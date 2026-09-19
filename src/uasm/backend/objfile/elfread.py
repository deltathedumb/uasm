"""Reading an ELF64 relocatable object back in.

THE OTHER DIRECTION FROM `elf.py`, and it exists for the same reason that one
does. A compiler that emits objects and then hands them to `ld` has not
finished: the program cannot be produced without a toolchain, and every
platform's toolchain is a different thing to find, version and be broken by.
`link/staticlink.py` is the linker; this is what feeds it.

WHAT IT READS AND WHAT IT REFUSES. ELF64, little-endian, relocatable
(`ET_REL`), with `SHT_RELA` relocations -- which is exactly what the emitters
next door write and what every ELF64 psABI this targets uses. `SHT_REL`
(implicit addends, 32-bit ELF's style) is refused rather than half-supported:
the addend lives in the field being patched and reading it back is a
per-relocation-type decision, so accepting one would mean claiming a
completeness this does not have.

WHY IT DOES NOT SHARE THE WRITER'S DATACLASSES. `Symbol` there names its
section by STRING, which is what a backend has in hand while it builds one.
A reader has a section INDEX, and two input objects each have a `.text` --
so resolving to a name would lose which one. Everything here is indexed, and
the index is what the linker groups by.

THE PARTS DELIBERATELY NOT MODELLED, so that a later reader knows they were
considered rather than missed:

  * SHN_COMMON. A tentative definition (C's `int x;` at file scope) is a
    symbol whose `st_value` is its ALIGNMENT and whose section is the
    `SHN_COMMON` pseudo-section, and the linker is expected to allocate it.
    Nothing this compiler emits produces one -- an uninitialised global goes
    to `.bss` with a real address -- so it is reported as an error naming the
    symbol rather than silently allocated wrong.
  * Section groups, COMDAT, and `.eh_frame` awareness. No emitter here makes
    them.
  * Any 32-bit ELF. `ptr` is 64 bits everywhere in this compiler (see the
    note in `ir/types.py`), so a 32-bit object could not be linked with the
    rest anyway.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .elf import (
    SHF_ALLOC, SHF_EXECINSTR, SHF_WRITE, SHN_UNDEF,
    SHT_NOBITS, SHT_PROGBITS, SHT_RELA, SHT_STRTAB, SHT_SYMTAB,
    STB_GLOBAL, STB_LOCAL, STB_WEAK,
)

#: `SHT_REL`, which 32-bit ELF uses and which this refuses: its addend lives
#: in the field being patched, so reading one back is a decision per
#: relocation TYPE rather than a field to unpack.
SHT_REL = 9

#: `e_type`. Only a relocatable can be an input to a static link: an
#: executable has had its relocations applied and thrown away, and a shared
#: object needs a dynamic loader this produces no use of.
ET_REL = 1

#: `st_shndx` values that are not section indices at all.
SHN_ABS = 0xFFF1
SHN_COMMON = 0xFFF2

_EHDR = "<16sHHIQQQIHHHHHH"
_SHDR = "<IIQQQQIIQQ"
_SYM = "<IBBHQQ"
_RELA = "<QQq"


class ElfError(Exception):
    """An object this cannot read. Carries a user-facing reason.

    NOT `ValueError` AND NOT AN ASSERTION, because the thing that raises it is
    a file the user named on a command line. A truncated object, an object for
    the wrong machine and an object in a format nothing here writes are all
    ordinary mistakes to make, and each deserves a sentence rather than a
    traceback through `struct.unpack_from`.
    """


@dataclass(slots=True)
class InSection:
    """One section of one input object, as read."""

    #: Its index in ITS OWN object's section table, which is what a symbol's
    #: `st_shndx` and a relocation section's `sh_info` refer to.
    index: int
    name: str
    kind: int
    flags: int
    align: int
    #: The bytes. Empty for `SHT_NOBITS`, whose size is still `size`.
    data: bytes
    size: int

    @property
    def is_alloc(self) -> bool:
        return bool(self.flags & SHF_ALLOC)

    @property
    def is_exec(self) -> bool:
        return bool(self.flags & SHF_EXECINSTR)

    @property
    def is_write(self) -> bool:
        return bool(self.flags & SHF_WRITE)

    @property
    def is_bss(self) -> bool:
        return self.kind == SHT_NOBITS


@dataclass(slots=True)
class InSymbol:
    """One entry of one input object's `.symtab`."""

    name: str
    #: `st_shndx`: a section index, `SHN_UNDEF`, `SHN_ABS` or `SHN_COMMON`.
    shndx: int
    value: int
    size: int
    binding: int
    kind: int

    @property
    def is_undefined(self) -> bool:
        return self.shndx == SHN_UNDEF

    @property
    def is_local(self) -> bool:
        return self.binding == STB_LOCAL

    @property
    def is_weak(self) -> bool:
        return self.binding == STB_WEAK

    @property
    def is_absolute(self) -> bool:
        return self.shndx == SHN_ABS


@dataclass(slots=True)
class InReloc:
    """One `Elf64_Rela`, with the symbol resolved to an index."""

    #: Offset within the section being patched.
    offset: int
    #: Index into the owning object's symbol list.
    symbol: int
    #: The architecture's relocation number.
    kind: int
    addend: int


@dataclass(slots=True)
class Relocatable:
    """One input object, read.

    `origin` IS FOR THE ERROR MESSAGE and nothing else. A missing symbol is
    reported as "undefined reference to X, needed by Y", and Y is this -- so
    it is the path the user named, kept whole rather than reduced to a
    basename that two directories could both produce.
    """

    origin: str
    machine: int
    sections: list[InSection]
    symbols: list[InSymbol]
    #: Relocations, keyed by the index of the section they patch. A section
    #: with none is absent rather than present and empty, so `in` is the
    #: question "does this need patching at all".
    relocs: dict[int, list[InReloc]] = field(default_factory=dict)


def _cstr(blob: bytes, at: int) -> str:
    """One NUL-terminated string out of a string table."""
    if at >= len(blob):
        raise ElfError(f"string table offset {at} is past its end")
    end = blob.find(b"\0", at)
    if end < 0:
        raise ElfError("string table is not NUL-terminated")
    return blob[at:end].decode("utf-8", "surrogateescape")


def _at(data: bytes, fmt: str, off: int, what: str):
    """`struct.unpack_from`, with the truncation reported in words.

    A SHORT FILE IS THE COMMON CORRUPTION -- a build killed part way through,
    a half-written artifact -- and `struct.error: unpack_from requires a
    buffer of at least 64 bytes` names nothing the user can act on.
    """
    if off < 0 or off + struct.calcsize(fmt) > len(data):
        raise ElfError(f"truncated: {what} runs past the end of the file")
    return struct.unpack_from(fmt, data, off)


def read(data: bytes, origin: str = "<memory>") -> Relocatable:
    """One relocatable ELF64 object, or `ElfError` saying why not."""
    if len(data) < 64 or data[:4] != b"\x7fELF":
        raise ElfError(f"{origin}: not an ELF file")
    if data[4] != 2:
        raise ElfError(f"{origin}: not ELF64 (only 64-bit objects link here, "
                       f"because `ptr` is 64 bits everywhere in this "
                       f"compiler)")
    if data[5] != 1:
        raise ElfError(f"{origin}: not little-endian")

    (_ident, e_type, e_machine, _ver, _entry, _phoff, e_shoff, _flags,
     _ehsize, _phentsize, _phnum, e_shentsize, e_shnum,
     e_shstrndx) = _at(data, _EHDR, 0, "the ELF header")
    if e_type != ET_REL:
        raise ElfError(
            f"{origin}: ELF type {e_type} is not a relocatable object; only "
            f"objects from `uasm build --emit` and other `.o` files can be "
            f"linked")
    if e_shnum == 0 or e_shoff == 0:
        raise ElfError(f"{origin}: has no section table")

    # ── the section headers ─────────────────────────────────────────────────
    raw: list[tuple] = []
    for i in range(e_shnum):
        raw.append(_at(data, _SHDR, e_shoff + i * e_shentsize,
                       f"section header {i}"))
    if e_shstrndx >= e_shnum:
        raise ElfError(f"{origin}: section-name table index is out of range")
    shstr_off, shstr_size = raw[e_shstrndx][4], raw[e_shstrndx][5]
    shstr = data[shstr_off:shstr_off + shstr_size]

    sections: list[InSection] = []
    for i, (nm, kind, flags, _addr, off, size, _link, _info, align,
            _entsize) in enumerate(raw):
        # `SHT_NOBITS` OCCUPIES NO FILE SPACE, so its `sh_offset` is where it
        # WOULD have been and slicing there would read whatever follows.
        body = b"" if kind == SHT_NOBITS else data[off:off + size]
        if kind != SHT_NOBITS and len(body) != size:
            raise ElfError(f"{origin}: section {_cstr(shstr, nm)!r} runs past "
                           f"the end of the file")
        sections.append(InSection(index=i, name=_cstr(shstr, nm), kind=kind,
                                  flags=flags, align=max(1, align),
                                  data=body, size=size))

    # ── the symbol table ────────────────────────────────────────────────────
    #
    # THE FIRST `SHT_SYMTAB` AND NOT ALL OF THEM. ELF permits several; no
    # producer writes more than one, and a linker that merged two would have
    # to decide what a relocation's `sh_link` meant when it named the other.
    symtabs = [i for i, r in enumerate(raw) if r[1] == SHT_SYMTAB]
    symbols: list[InSymbol] = []
    symtab_index = -1
    if symtabs:
        symtab_index = symtabs[0]
        sh = raw[symtab_index]
        strtab_index = sh[6]
        if strtab_index >= e_shnum or raw[strtab_index][1] != SHT_STRTAB:
            raise ElfError(f"{origin}: the symbol table's `sh_link` does not "
                           f"name a string table")
        st_off, st_size = raw[strtab_index][4], raw[strtab_index][5]
        strtab = data[st_off:st_off + st_size]
        entsize = sh[9] or 24
        for i in range(sh[5] // entsize):
            nameoff, info, _other, shndx, value, size = _at(
                data, _SYM, sh[4] + i * entsize, f"symbol {i}")
            symbols.append(InSymbol(
                name=_cstr(strtab, nameoff), shndx=shndx, value=value,
                size=size, binding=info >> 4, kind=info & 0xF))

    # ── the relocations ─────────────────────────────────────────────────────
    relocs: dict[int, list[InReloc]] = {}
    for i, (_nm, kind, _flags, _addr, off, size, link, info, _align,
            entsize) in enumerate(raw):
        if kind != SHT_RELA:
            if kind == SHT_REL:
                raise ElfError(
                    f"{origin}: section {sections[i].name!r} uses SHT_REL "
                    f"relocations; only SHT_RELA (with explicit addends) can "
                    f"be linked here")
            continue
        if link != symtab_index:
            raise ElfError(f"{origin}: relocation section "
                           f"{sections[i].name!r} refers to a symbol table "
                           f"that is not this object's")
        if info >= e_shnum:
            raise ElfError(f"{origin}: relocation section "
                           f"{sections[i].name!r} patches section {info}, "
                           f"which does not exist")
        step = entsize or 24
        out = relocs.setdefault(info, [])
        for k in range(size // step):
            r_off, r_info, r_add = _at(data, _RELA, off + k * step,
                                       f"relocation {k} of {sections[i].name}")
            sym = r_info >> 32
            if sym >= len(symbols):
                raise ElfError(f"{origin}: relocation names symbol {sym}, "
                               f"and the symbol table has {len(symbols)}")
            out.append(InReloc(offset=r_off, symbol=sym,
                               kind=r_info & 0xFFFFFFFF, addend=r_add))

    return Relocatable(origin=origin, machine=e_machine, sections=sections,
                       symbols=symbols, relocs=relocs)


def is_elf(data: bytes) -> bool:
    """Whether these bytes begin an ELF file.

    FOR TELLING INPUTS APART rather than for validating one: the linker takes
    a list of paths and has to know which are objects and which are archives
    or something else entirely, and it must answer that without raising.
    """
    return len(data) >= 4 and data[:4] == b"\x7fELF"


__all__ = [
    "ElfError", "InReloc", "InSection", "InSymbol", "Relocatable",
    "ET_REL", "SHN_ABS", "SHN_COMMON", "SHN_UNDEF",
    "STB_GLOBAL", "STB_LOCAL", "STB_WEAK",
    "SHT_NOBITS", "SHT_PROGBITS",
    "is_elf", "read",
]
