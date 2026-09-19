"""Reading a PE/COFF relocatable object back in.

The other direction from `coff.py`, and `elfread.py`'s twin: what
`link/staticlink.py` links on Windows. It answers the SAME dataclasses
`elfread` does, so the linker has one shape to work on and one place that
knows what a relocation number means.

THE DIFFERENCE THAT IS NOT COSMETIC, and it is the whole reason this file is
longer than it looks: COFF HAS NO ADDEND FIELD. ELF's `SHT_RELA` carries one
per relocation; COFF folds it into the section bytes under the field being
patched, so reading it back means knowing how wide that field is and whether
it is signed -- which is a fact about the architecture's relocation number,
not about the container. `elfread.py` refuses `SHT_REL` for exactly this
reason and says so; here there is no choice, so the knowledge is written
down, in `_ADDEND_WIDTH`, per relocation type, with nothing guessed.

AND THE IMPLICIT BIAS IS NOT RECOVERED, because it was never stored.
`IMAGE_REL_AMD64_REL32` means "relative to the byte AFTER the four the field
occupies", where ELF's `R_X86_64_PC32` means "relative to the field" and
carries -4 to say so. The bias is part of what the TYPE means, so it belongs
in the patch table beside the formula rather than in a number read out of the
file -- see `link/staticlink.py`. What comes back from here is the addend
COFF actually stored, which for everything this compiler emits is zero.

WHAT IT REFUSES. A COFF object with symbol auxiliary records it cannot skip,
a `.bss` with contents, or a machine no backend here targets. Anything with
`IMAGE_SCN_LNK_*` in it -- COMDAT sections, directives -- is dropped with the
section, because nothing here writes one and honouring one wrongly is worse
than not honouring it.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .coff import (
    IMAGE_FILE_MACHINE_AMD64, IMAGE_FILE_MACHINE_ARM64,
    IMAGE_FILE_MACHINE_I386,
    IMAGE_SCN_CNT_CODE, IMAGE_SCN_CNT_INITIALIZED_DATA,
    IMAGE_SCN_CNT_UNINITIALIZED_DATA, IMAGE_SCN_LNK_REMOVE,
    IMAGE_SCN_MEM_DISCARDABLE, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ,
    IMAGE_SCN_MEM_WRITE, IMAGE_SYM_CLASS_EXTERNAL, IMAGE_SYM_CLASS_STATIC,
    IMAGE_SYM_UNDEFINED,
)

_FILEHDR = "<HHIIIHH"
_SECHDR = "<8sIIIIIIHHI"
_SYM = "<8sIhHBB"
_RELOC = "<IIH"

#: How wide the field a relocation patches is, and whether it is signed --
#: which is what it takes to read an addend back out of the section data.
#: Keyed by (machine, type).
#:
#: ONLY THE TYPES THIS COMPILER'S OWN EMITTERS PRODUCE. An object from
#: elsewhere carrying another one is refused by name rather than guessed at:
#: reading four bytes where the field is two is a plausible wrong addend, and
#: a plausible wrong addend is a program that links and misbehaves.
_ADDEND_WIDTH = {
    # IMAGE_REL_AMD64_REL32 and its four biased cousins, all `word32`.
    (IMAGE_FILE_MACHINE_AMD64, 0x0001): (8, True),    # ADDR64
    (IMAGE_FILE_MACHINE_AMD64, 0x0002): (4, False),   # ADDR32
    (IMAGE_FILE_MACHINE_AMD64, 0x0003): (4, False),   # ADDR32NB
    (IMAGE_FILE_MACHINE_AMD64, 0x0004): (4, True),    # REL32
    (IMAGE_FILE_MACHINE_AMD64, 0x0005): (4, True),    # REL32_1
    (IMAGE_FILE_MACHINE_AMD64, 0x0006): (4, True),    # REL32_2
    (IMAGE_FILE_MACHINE_AMD64, 0x0007): (4, True),    # REL32_3
    (IMAGE_FILE_MACHINE_AMD64, 0x0008): (4, True),    # REL32_4
    (IMAGE_FILE_MACHINE_AMD64, 0x0009): (4, True),    # REL32_5
    # ARM64's are bitfields inside one instruction word, so the "addend" is
    # not a plain integer in the data at all. None of them carries one in
    # anything this compiler emits, and a reader that pretended to recover
    # one would be inventing it -- see `_addend_of`.
    (IMAGE_FILE_MACHINE_ARM64, 0x0001): (8, True),    # ADDR64
    (IMAGE_FILE_MACHINE_ARM64, 0x0003): (0, False),   # BRANCH26
    (IMAGE_FILE_MACHINE_ARM64, 0x0004): (0, False),   # PAGEBASE_REL21
    (IMAGE_FILE_MACHINE_ARM64, 0x0006): (0, False),   # PAGEOFFSET_12A
    (IMAGE_FILE_MACHINE_ARM64, 0x0007): (0, False),   # PAGEOFFSET_12L
}

#: The machines a backend here targets, so an object for another is refused
#: with its number rather than linked into nonsense.
KNOWN_MACHINES = (IMAGE_FILE_MACHINE_AMD64, IMAGE_FILE_MACHINE_ARM64,
                  IMAGE_FILE_MACHINE_I386)


class CoffError(Exception):
    """An object this cannot read. Carries a user-facing reason."""


@dataclass(slots=True)
class InSection:
    """One section of one input object, shaped as `elfread.InSection` is.

    `flags` IS SPELLED IN ELF'S VOCABULARY, translated on the way in. The
    linker groups sections by what they ARE -- executable, writable, without
    contents -- and those are the same three questions in every format; making
    it ask them in three vocabularies would put format knowledge in the one
    place that is meant to have none.
    """

    index: int
    name: str
    kind: int
    flags: int
    align: int
    data: bytes
    size: int

    @property
    def is_alloc(self) -> bool:
        return bool(self.flags & 0x2)          # SHF_ALLOC

    @property
    def is_exec(self) -> bool:
        return bool(self.flags & 0x4)          # SHF_EXECINSTR

    @property
    def is_write(self) -> bool:
        return bool(self.flags & 0x1)          # SHF_WRITE

    @property
    def is_bss(self) -> bool:
        return self.kind == 8                  # SHT_NOBITS


@dataclass(slots=True)
class InSymbol:
    name: str
    shndx: int
    value: int
    size: int
    binding: int
    kind: int

    @property
    def is_undefined(self) -> bool:
        return self.shndx == 0

    @property
    def is_local(self) -> bool:
        return self.binding == 0               # STB_LOCAL

    @property
    def is_weak(self) -> bool:
        return self.binding == 2               # STB_WEAK

    @property
    def is_absolute(self) -> bool:
        return self.shndx == 0xFFF1            # SHN_ABS


@dataclass(slots=True)
class InReloc:
    offset: int
    symbol: int
    kind: int
    addend: int


@dataclass(slots=True)
class Relocatable:
    origin: str
    machine: int
    sections: list[InSection]
    symbols: list[InSymbol]
    relocs: dict[int, list[InReloc]] = field(default_factory=dict)
    #: WHICH CONTAINER THIS CAME OUT OF. The linker keys its patch table on
    #: it, because `4` means `IMAGE_REL_AMD64_REL32` here and
    #: `R_X86_64_PLT32` in an ELF object -- the same number, different
    #: arithmetic, and nothing in the record itself says which.
    fmt: str = "coff"


def _name_of(raw: bytes, strings: bytes, what: str) -> str:
    """One section or symbol name, inline or spilled into the string table.

    A NAME LONGER THAN EIGHT BYTES SPILLS, and the two kinds spell the
    reference differently: a SECTION writes `/` and the offset in decimal
    ASCII, a SYMBOL writes four zero bytes and then the offset as a number.
    Both are handled here because both arrive as eight bytes.
    """
    if raw[:4] == b"\0\0\0\0" and what == "symbol":
        at, = struct.unpack_from("<I", raw, 4)
        return _cstr(strings, at, what)
    text = raw.rstrip(b"\0")
    if text[:1] == b"/" and what == "section":
        try:
            return _cstr(strings, int(text[1:]), what)
        except ValueError:
            raise CoffError(f"a {what} name begins with / and is not an "
                            f"offset: {text!r}") from None
    return text.decode("utf-8", "surrogateescape")


def _cstr(strings: bytes, at: int, what: str) -> str:
    # THE LENGTH COUNTS ITSELF, so an offset is from the start of the four
    # bytes that hold it and four is the first real character.
    if at < 4 or at >= len(strings):
        raise CoffError(f"a {what} name points outside the string table")
    end = strings.find(b"\0", at)
    if end < 0:
        raise CoffError("the string table is not NUL-terminated")
    return strings[at:end].decode("utf-8", "surrogateescape")


def _flags_of(characteristics: int) -> tuple[int, int]:
    """COFF characteristics as (ELF section type, ELF section flags).

    `.bss` IS THE ONE THAT MATTERS: `IMAGE_SCN_CNT_UNINITIALIZED_DATA` is
    what makes a section occupy memory and no file space, which is `SHT_NOBITS`
    on the other side. Everything else is a straight translation of the three
    memory bits.
    """
    kind = 8 if characteristics & IMAGE_SCN_CNT_UNINITIALIZED_DATA else 1
    flags = 0
    if characteristics & (IMAGE_SCN_CNT_CODE
                          | IMAGE_SCN_CNT_INITIALIZED_DATA
                          | IMAGE_SCN_CNT_UNINITIALIZED_DATA):
        flags |= 0x2                                   # SHF_ALLOC
    if characteristics & IMAGE_SCN_MEM_EXECUTE:
        flags |= 0x4                                   # SHF_EXECINSTR
    if characteristics & IMAGE_SCN_MEM_WRITE:
        flags |= 0x1                                   # SHF_WRITE
    # A SECTION THE LINKER IS TOLD TO DROP IS NOT PART OF THE PROGRAM.
    # `.drectve` carries linker switches and `IMAGE_SCN_MEM_DISCARDABLE`
    # marks debug data; neither belongs in an image, and neither is
    # something this linker can act on.
    if characteristics & (IMAGE_SCN_LNK_REMOVE | IMAGE_SCN_MEM_DISCARDABLE):
        flags &= ~0x2
    return kind, flags


def _align_of(characteristics: int) -> int:
    """The alignment COFF's one-based exponent means. Zero is unspecified."""
    exponent = (characteristics >> 20) & 0xF
    return 1 if exponent == 0 else 1 << (exponent - 1)


def _addend_of(body: bytes, rel_offset: int, machine: int, kind: int,
               origin: str) -> int:
    """The addend COFF stored in the section data under the field."""
    got = _ADDEND_WIDTH.get((machine, kind))
    if got is None:
        raise CoffError(
            f"{origin}: relocation type {kind:#x} is not one this linker "
            f"knows for machine {machine:#x}; its addend lives in the "
            f"section data and reading it needs the field's width")
    width, signed = got
    if width == 0:
        # A BITFIELD INSIDE AN INSTRUCTION. Nothing this compiler emits puts
        # an addend in one -- `backends/arm64/machoemit.py` REFUSES a non-zero
        # addend for the same reason rather than dropping it -- so zero is
        # the truth here and not a default.
        return 0
    if rel_offset + width > len(body):
        raise CoffError(f"{origin}: a relocation patches past the end of its "
                        f"section")
    fmt = {1: "b", 2: "h", 4: "i", 8: "q"}[width] if signed else \
        {1: "B", 2: "H", 4: "I", 8: "Q"}[width]
    return struct.unpack_from("<" + fmt, body, rel_offset)[0]


def read(data: bytes, origin: str = "<memory>") -> Relocatable:
    """One relocatable PE/COFF object, or `CoffError` saying why not."""
    if len(data) < 20:
        raise CoffError(f"{origin}: too short to be a COFF object")
    (machine, nsections, _stamp, symtab_at, nsymbols, opt_size,
     _chars) = struct.unpack_from(_FILEHDR, data, 0)
    if machine not in KNOWN_MACHINES:
        raise CoffError(f"{origin}: COFF machine {machine:#x} is not one any "
                        f"backend here targets")
    if opt_size:
        raise CoffError(f"{origin}: has an optional header, so it is an image "
                        f"rather than a relocatable object")
    if machine == IMAGE_FILE_MACHINE_I386:
        raise CoffError(f"{origin}: 32-bit COFF cannot be linked here, "
                        f"because `ptr` is 64 bits everywhere in this "
                        f"compiler")

    # THE STRING TABLE FOLLOWS THE SYMBOLS, and its own first four bytes are
    # its length. An object with no symbols at all has none.
    strings = b"\0\0\0\0"
    if symtab_at and nsymbols:
        at = symtab_at + 18 * nsymbols
        if at + 4 <= len(data):
            size, = struct.unpack_from("<I", data, at)
            strings = data[at:at + max(4, size)]

    sections: list[InSection] = []
    raw_bodies: dict[int, bytes] = {}
    for i in range(nsections):
        off = 20 + i * 40
        if off + 40 > len(data):
            raise CoffError(f"{origin}: section header {i} runs past the end")
        (nm, vsize, _vaddr, rawsize, rawptr, relptr, _lineptr, nrel,
         _nline, chars) = struct.unpack_from(_SECHDR, data, off)
        kind, flags = _flags_of(chars)
        # `.bss` CARRIES ITS SIZE IN `SizeOfRawData` here, which is what the
        # writer next door does and what the comment there explains: a size
        # kept in `VirtualSize` reads as nothing to every tool.
        size = rawsize if kind == 8 else rawsize
        body = (b"" if kind == 8 or not rawptr
                else data[rawptr:rawptr + rawsize])
        if kind != 8 and rawptr and len(body) != rawsize:
            raise CoffError(f"{origin}: section {i} runs past the end")
        sections.append(InSection(index=i + 1,
                                  name=_name_of(nm, strings, "section"),
                                  kind=kind, flags=flags,
                                  align=_align_of(chars), data=body,
                                  size=size))
        raw_bodies[i + 1] = body

    symbols: list[InSymbol] = []
    skip = 0
    for i in range(nsymbols):
        off = symtab_at + i * 18
        if off + 18 > len(data):
            raise CoffError(f"{origin}: symbol {i} runs past the end")
        if skip:
            # AUXILIARY RECORDS ARE PART OF THE PRECEDING SYMBOL and are not
            # symbols. They still occupy an index, so a placeholder keeps the
            # numbering a relocation refers to.
            skip -= 1
            symbols.append(InSymbol("", 0, 0, 0, 0, 0))
            continue
        nm, value, secnum, typ, storage, naux = struct.unpack_from(
            _SYM, data, off)
        skip = naux
        binding = 0 if storage == IMAGE_SYM_CLASS_STATIC else 1
        symbols.append(InSymbol(
            name=_name_of(nm, strings, "symbol"),
            # `SectionNumber` IS ONE-BASED, and the two negative values are
            # not sections: -1 is absolute and -2 is a debug symbol.
            shndx=(0 if secnum == IMAGE_SYM_UNDEFINED
                   else 0xFFF1 if secnum == -1 else secnum),
            value=value, size=0,
            binding=binding,
            kind=2 if (typ >> 4) == 0x2 else 0))

    relocs: dict[int, list[InReloc]] = {}
    for i in range(nsections):
        off = 20 + i * 40
        (_nm, _vsize, _vaddr, _rawsize, _rawptr, relptr, _lineptr, nrel,
         _nline, _chars) = struct.unpack_from(_SECHDR, data, off)
        if not nrel or not relptr:
            continue
        body = raw_bodies[i + 1]
        out = relocs.setdefault(i + 1, [])
        for k in range(nrel):
            at = relptr + k * 10
            if at + 10 > len(data):
                raise CoffError(f"{origin}: relocation {k} runs past the end")
            r_off, sym, kind = struct.unpack_from(_RELOC, data, at)
            if sym >= len(symbols):
                raise CoffError(f"{origin}: relocation names symbol {sym}, "
                                f"and the table has {len(symbols)}")
            out.append(InReloc(offset=r_off, symbol=sym, kind=kind,
                               addend=_addend_of(body, r_off, machine, kind,
                                                 origin)))

    return Relocatable(origin=origin, machine=machine, sections=sections,
                       symbols=symbols, relocs=relocs, fmt="coff")


def is_coff(data: bytes) -> bool:
    """Whether these bytes begin a COFF relocatable for a known machine.

    COFF HAS NO MAGIC NUMBER, which is the one genuinely awkward thing about
    telling it apart: the file begins with the machine, so recognising one
    means recognising the machines. That is why this answers only for the
    machines a backend here targets -- a wider answer would claim any file
    whose first two bytes happen to be 0x8664.
    """
    if len(data) < 20:
        return False
    machine, _n, _stamp, _symtab, _nsym, opt, _chars = struct.unpack_from(
        _FILEHDR, data, 0)
    return machine in KNOWN_MACHINES and opt == 0


__all__ = ["CoffError", "InReloc", "InSection", "InSymbol", "Relocatable",
           "KNOWN_MACHINES", "is_coff", "read"]
