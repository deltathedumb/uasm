"""Reading a Mach-O relocatable object back in.

The other direction from `macho.py`, and the third of `elfread.py`'s twins:
what `link/staticlink.py` links on macOS. It answers the SAME dataclasses the
other two readers do, so the linker has one shape to work on.

THE THREE PLACES MACH-O IS NOT LIKE THE OTHERS, which is what this file is
for:

  * SECTIONS ARE NAMED TWICE and neither name is `.text`. `__TEXT,__text` is
    code; `__TEXT,__const` is read-only data in the same segment; `__DATA`
    is writable. THE SEGMENT IS THE PERMISSION and the section flags say
    whether it holds instructions, so both have to be read to answer the
    three questions the linker actually asks -- executable, writable,
    without contents.

  * THE ADDEND IS IN THE SECTION DATA, as in COFF. What is not as in COFF is
    AArch64: there the addend is a SEPARATE RELOCATION RECORD,
    `ARM64_RELOC_ADDEND`, sitting immediately before the one it modifies,
    with the addend in the field where a symbol index normally goes. A
    reader that did not know this would resolve it as a symbol index and
    reach whatever symbol happened to be at that position -- so it is folded
    into the following record here, which is what it means.

  * RELOCATIONS COME IN TWO SHAPES. `r_address` with its top bit set is a
    SCATTERED relocation, a different struct entirely. Nothing this compiler
    writes is scattered and a 64-bit Mach-O is not supposed to contain one;
    it is refused by name rather than misread as an enormous offset.

WHAT IT REFUSES. A 32-bit Mach-O, a fat archive, a file that is not an
object, a scattered relocation, a relocation against a SECTION rather than a
symbol, and a relocation type no patcher here implements. Each by name: an
object this cannot link should say which part of it is the problem.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .macho import (
    CPU_TYPE_ARM64, CPU_TYPE_X86_64, LC_SEGMENT_64, LC_SYMTAB, MH_MAGIC_64,
    MH_OBJECT, N_EXT, N_SECT, N_STAB, N_TYPE, N_UNDF,
    S_ATTR_PURE_INSTRUCTIONS, S_ATTR_SOME_INSTRUCTIONS,
)

#: `MH_CIGAM_64`, `MH_MAGIC` and `MH_CIGAM`: a 64-bit big-endian object and
#: the two 32-bit ones. Recognised only so that the refusal can name what the
#: file actually is.
MH_CIGAM_64 = 0xCFFAEDFE
MH_MAGIC_32 = 0xFEEDFACE
MH_CIGAM_32 = 0xCEFAEDFE
#: A universal ("fat") archive, which holds several of the above.
FAT_MAGIC = 0xCAFEBABE
FAT_CIGAM = 0xBEBAFECA

#: Section types, from `mach-o/loader.h`. The three that mean "occupies
#: memory and no file space", which is `SHT_NOBITS` on the ELF side.
S_ZEROFILL = 0x1
S_GB_ZEROFILL = 0xC
S_THREAD_LOCAL_ZEROFILL = 0x12
_ZEROFILL = (S_ZEROFILL, S_GB_ZEROFILL, S_THREAD_LOCAL_ZEROFILL)

#: The addend carrier. Its `r_symbolnum` field is not a symbol index but a
#: 24-bit addend for the record that follows it.
ARM64_RELOC_ADDEND = 10
#: The one AArch64 type whose addend IS in the section data, because the
#: field is a plain pointer rather than a bitfield in an instruction.
ARM64_RELOC_UNSIGNED = 0

#: The machines a backend here targets.
KNOWN_MACHINES = (CPU_TYPE_X86_64, CPU_TYPE_ARM64)

_HEADER = "<IiiIIIII"
_SEGMENT = "<II16sQQQQiiII"
_SECTION = "<16s16sQQIIIIIIII"
_SYMTAB = "<IIIIII"
_NLIST = "<IBBHQ"
_RELOC = "<iI"


class MachoError(Exception):
    """An object this cannot read. Carries a user-facing reason."""


@dataclass(slots=True)
class InSection:
    """One section of one input object, shaped as `elfread.InSection` is.

    `flags` IS SPELLED IN ELF'S VOCABULARY, as in `coffread`, and for the
    same reason: the linker asks three questions of every section and they
    are the same three in every format.
    """

    index: int
    name: str
    kind: int
    flags: int
    align: int
    data: bytes
    size: int
    #: `__TEXT,__text`, kept for diagnostics: a section refused for its
    #: permissions should be named the way the file names it.
    origin_name: str = ""

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
    #: MACH-O PUTS BOTH IN THE RECORD where ELF implies them from the type,
    #: so both have to be carried through to the patcher. `length` is the
    #: log2 of the field width in bytes.
    pcrel: bool = False
    length: int = 2


@dataclass(slots=True)
class Relocatable:
    origin: str
    machine: int
    sections: list[InSection]
    symbols: list[InSymbol]
    relocs: dict[int, list[InReloc]] = field(default_factory=dict)
    fmt: str = "macho"


def _permissions(segment: str, section: str, flags: int) -> tuple[int, int]:
    """A section's (ELF type, ELF flags), from its segment and its flags.

    THE SEGMENT DECIDES WHETHER IT IS WRITABLE. `__TEXT,__const` and
    `__DATA,__const` hold the same kind of thing and land in different
    segments for exactly this reason, and a reader that keyed on the SECTION
    name would put a program's constants on a writable page or its data on a
    read-only one.
    """
    kind = 8 if flags & 0xFF in _ZEROFILL else 1
    out = 0x2                                            # SHF_ALLOC
    if flags & (S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS):
        out |= 0x4                                       # SHF_EXECINSTR
    if segment not in ("__TEXT", "__DATA", "__DATA_CONST", ""):
        # NOT A SEGMENT THIS LINKER PLACES. `__LINKEDIT` holds the symbol
        # table, `__LLVM` holds bitcode, `__DWARF` holds debug information:
        # none of them is part of a program's image, and dropping SHF_ALLOC
        # is how the linker is told so.
        return kind, 0
    if segment in ("__DATA", "__DATA_CONST"):
        out |= 0x1                                       # SHF_WRITE
    if segment == "" and not out & 0x4:
        # A NAMELESS SEGMENT is what an object written by this compiler
        # carries, because an object has one segment and it has no name. The
        # section's own `segname` field is the one read above; reaching here
        # means neither said anything, so the safe reading is writable data.
        out |= 0x1
    return kind, out


def _addend_of(data: bytes, offset: int, length: int, kind: int,
               cputype: int, where: str) -> int:
    """The addend Mach-O stored in the section data, if it stored one there.

    MACH-O HAS NO ADDEND FIELD, as COFF has none -- see `coffread._addend_of`,
    which is this function's twin and was written first. The value sits under
    the field being patched, so reading it back means knowing how wide that
    field is; Mach-O, unlike COFF, says so in the RECORD, which is what
    `length` is.

    WHICH TYPES KEEP IT THERE IS THE ARCHITECTURE'S ANSWER, not the
    container's. Every x86-64 type patches a plain little-endian integer, so
    every one of them carries its addend in the data. AArch64 packs its
    operands into instruction bitfields, so there is nowhere sensible to put
    one -- which is why Mach-O invented `ARM64_RELOC_ADDEND` as a separate
    record. The single exception is `ARM64_RELOC_UNSIGNED`, whose field is a
    pointer and not an instruction.

    THIS FILE SHIPPED WITHOUT IT, reading every x86-64 addend as zero. Its
    own emitter stores zero, so nothing here noticed; an object from clang
    with `leaq sym+8(%rip)` in it would have linked and been wrong by eight.
    """
    if cputype == CPU_TYPE_ARM64 and kind != ARM64_RELOC_UNSIGNED:
        return 0
    width = 1 << length
    form = {1: "<b", 2: "<h", 4: "<i", 8: "<q"}.get(width)
    if form is None or offset + width > len(data):
        raise MachoError(
            f"a relocation at {offset:#x} in {where} patches {width} bytes "
            f"that are not there")
    return struct.unpack_from(form, data, offset)[0]


def _cstr(raw: bytes, at: int) -> str:
    if at >= len(raw):
        raise MachoError("a symbol name points outside the string table")
    end = raw.find(b"\0", at)
    if end < 0:
        raise MachoError("the string table is not NUL-terminated")
    return raw[at:end].decode("utf-8", "surrogateescape")


def _name(raw: bytes) -> str:
    """A 16-byte Mach-O name. NOT NUL-terminated when it fills the field."""
    return raw.rstrip(b"\0").decode("utf-8", "surrogateescape")


def is_macho(blob: bytes) -> bool:
    """Whether these bytes begin like a Mach-O file of any kind.

    ANY KIND, including the ones `read` refuses: the caller uses this to
    decide WHICH reader to hand the bytes to, and a 32-bit Mach-O should be
    refused by the Mach-O reader with a reason rather than fall through to
    "not an object file this linker reads".
    """
    if len(blob) < 4:
        return False
    magic, = struct.unpack_from("<I", blob, 0)
    return magic in (MH_MAGIC_64, MH_CIGAM_64, MH_MAGIC_32, MH_CIGAM_32,
                     FAT_MAGIC, FAT_CIGAM)


def read(blob: bytes, origin: str = "<memory>") -> Relocatable:
    """One Mach-O 64 relocatable object, in the linker's own shape."""
    if len(blob) < struct.calcsize(_HEADER):
        raise MachoError(f"{origin}: too short to be a Mach-O object")
    magic, = struct.unpack_from("<I", blob, 0)
    if magic in (FAT_MAGIC, FAT_CIGAM):
        raise MachoError(
            f"{origin}: a universal binary, not a single object",
            )
    if magic in (MH_MAGIC_32, MH_CIGAM_32):
        raise MachoError(f"{origin}: a 32-bit Mach-O; this linker is 64-bit")
    if magic == MH_CIGAM_64:
        raise MachoError(
            f"{origin}: a big-endian Mach-O; no backend here writes one")
    if magic != MH_MAGIC_64:
        raise MachoError(f"{origin}: not a Mach-O object")

    (_, cputype, _subtype, filetype, ncmds, sizeofcmds, _flags,
     _pad) = struct.unpack_from(_HEADER, blob, 0)
    if filetype != MH_OBJECT:
        raise MachoError(
            f"{origin}: a Mach-O of type {filetype}, not a relocatable "
            f"object",
            )
    if cputype not in KNOWN_MACHINES:
        raise MachoError(f"{origin}: Mach-O cputype {cputype:#x} is not one "
                         f"this linker knows")

    sections: list[InSection] = []
    raw_relocs: list[tuple[int, int]] = []      # (reloff, nreloc) per section
    symbols: list[InSymbol] = []
    at = struct.calcsize(_HEADER)
    end_of_commands = at + sizeofcmds
    symtab: tuple[int, int, int, int] | None = None

    for _ in range(ncmds):
        if at + 8 > end_of_commands:
            raise MachoError(f"{origin}: the load commands run off the end")
        cmd, cmdsize = struct.unpack_from("<II", blob, at)
        if cmdsize < 8 or at + cmdsize > end_of_commands:
            raise MachoError(
                f"{origin}: a load command of {cmdsize} bytes does not fit")
        if cmd == LC_SEGMENT_64:
            (_, _, segname, _vmaddr, _vmsize, _fileoff, _filesize,
             _maxprot, _initprot, nsects,
             _segflags) = struct.unpack_from(_SEGMENT, blob, at)
            here = at + struct.calcsize(_SEGMENT)
            for _i in range(nsects):
                (sectname, sect_segname, _addr, size, offset, align, reloff,
                 nreloc, flags, _r1, _r2,
                 _r3) = struct.unpack_from(_SECTION, blob, here)
                here += struct.calcsize(_SECTION)
                # THE SECTION'S OWN `segname` WINS over the segment's, which
                # in an object file is empty. They agree in a linked image
                # and only the section's is filled in an object.
                segment = _name(sect_segname) or _name(segname)
                short = _name(sectname)
                kind, eflags = _permissions(segment, short, flags)
                data = b""
                if kind != 8 and size:
                    if offset + size > len(blob):
                        raise MachoError(
                            f"{origin}: section {segment},{short} claims "
                            f"bytes past the end of the file")
                    data = blob[offset:offset + size]
                sections.append(InSection(
                    index=len(sections) + 1, name=short, kind=kind,
                    flags=eflags, align=1 << align if align < 64 else 1,
                    data=data, size=size,
                    origin_name=f"{segment},{short}"))
                raw_relocs.append((reloff, nreloc))
        elif cmd == LC_SYMTAB:
            _, _, symoff, nsyms, stroff, strsize = struct.unpack_from(
                _SYMTAB, blob, at)
            symtab = (symoff, nsyms, stroff, strsize)
        at += cmdsize

    if symtab is not None:
        symoff, nsyms, stroff, strsize = symtab
        if stroff + strsize > len(blob):
            raise MachoError(f"{origin}: the string table runs off the end")
        strings = blob[stroff:stroff + strsize]
        for i in range(nsyms):
            place = symoff + i * struct.calcsize(_NLIST)
            if place + struct.calcsize(_NLIST) > len(blob):
                raise MachoError(f"{origin}: the symbol table runs off the "
                                 f"end")
            n_strx, n_type, n_sect, _n_desc, n_value = struct.unpack_from(
                _NLIST, blob, place)
            # EVERY ENTRY IS KEPT, debug entries included, because a
            # relocation names a symbol BY POSITION. Dropping one would
            # renumber every symbol after it.
            name = _cstr(strings, n_strx) if n_strx else ""
            if n_type & N_STAB:
                symbols.append(InSymbol(name, 0, 0, 0, 0, 0))
                continue
            kind = n_type & N_TYPE
            binding = 1 if n_type & N_EXT else 0
            if kind == N_UNDF:
                symbols.append(InSymbol(name, 0, 0, 0, binding, 0))
            elif kind == N_SECT:
                symbols.append(InSymbol(name, n_sect, n_value, 0, binding, 0))
            else:
                # N_ABS and N_INDR. An absolute symbol needs no section and
                # an indirect one is a dylib's business; both are passed
                # through as absolute so that a reference to one is resolved
                # rather than reported missing.
                symbols.append(InSymbol(name, 0xFFF1, n_value, 0, binding, 0))

    relocs: dict[int, list[InReloc]] = {}
    for sec, (reloff, nreloc) in zip(sections, raw_relocs, strict=True):
        if not nreloc:
            continue
        here: list[InReloc] = []
        pending = 0
        for i in range(nreloc):
            place = reloff + i * struct.calcsize(_RELOC)
            if place + struct.calcsize(_RELOC) > len(blob):
                raise MachoError(
                    f"{origin}: {sec.origin_name}'s relocations run off the "
                    f"end")
            address, packed = struct.unpack_from(_RELOC, blob, place)
            if address < 0:
                # THE TOP BIT OF `r_address` MARKS A SCATTERED RECORD, which
                # is a different struct. Reading it as this one gives a
                # plausible offset and a nonsense symbol.
                raise MachoError(
                    f"{origin}: {sec.origin_name} has a scattered "
                    f"relocation, which this linker does not read")
            symbolnum = packed & 0xFFFFFF
            pcrel = bool(packed & (1 << 24))
            length = (packed >> 25) & 3
            extern = bool(packed & (1 << 27))
            kind = (packed >> 28) & 0xF
            if kind == ARM64_RELOC_ADDEND and cputype == CPU_TYPE_ARM64:
                # SEE THE MODULE DOCSTRING. Not a relocation of its own: an
                # addend for the next one, sign-extended from 24 bits.
                pending = symbolnum - (1 << 24) \
                    if symbolnum & (1 << 23) else symbolnum
                continue
            if not extern:
                raise MachoError(
                    f"{origin}: {sec.origin_name} has a relocation against a "
                    f"section rather than a symbol, which this linker does "
                    f"not read")
            # THE TWO SOURCES OF AN ADDEND ARE EXCLUSIVE in anything a
            # Mach-O producer writes: a type that takes `ARM64_RELOC_ADDEND`
            # keeps nothing in the data, and a type that keeps it in the data
            # is never preceded by one. Summing them is what that means
            # rather than a choice between them.
            stored = _addend_of(sec.data, address, length, kind, cputype,
                                sec.origin_name)
            here.append(InReloc(offset=address, symbol=symbolnum, kind=kind,
                                addend=pending + stored, pcrel=pcrel,
                                length=length))
            pending = 0
        if pending:
            raise MachoError(
                f"{origin}: {sec.origin_name} ends with an ARM64_RELOC_ADDEND "
                f"that modifies nothing")
        relocs[sec.index] = here

    return Relocatable(origin=origin, machine=cputype, sections=sections,
                       symbols=symbols, relocs=relocs)


__all__ = ["InReloc", "InSection", "InSymbol", "KNOWN_MACHINES", "MachoError",
           "Relocatable", "is_macho", "read"]
