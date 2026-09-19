"""The linker, written here rather than found on the machine.

WHAT A STATIC LINK IS, since everything below is one of these four steps:

  1. GATHER.    Read every input object's sections, symbols and relocations.
  2. LAY OUT.   Decide where each input section goes in the output, and give
                it an address.
  3. RESOLVE.   Turn every symbol name into an address, and report the ones
                that have none.
  4. PATCH.     For each relocation, compute the number the architecture's
                formula asks for and write it into the field.

Then write an executable the kernel can load.

WHY THIS EXISTS. A compiler that emits objects and hands them to `ld` cannot
produce a program by itself: the user needs a toolchain, the toolchain differs
on every platform, and the compiler's output is only as portable as the
weakest tool in the chain. The backends here already encode their own
instructions and write their own object files -- `backend/objfile/elf.py` says
why -- and this is the last step that was still somebody else's.

THE ADDRESS TRICK THAT MAKES THE LAYOUT SIMPLE, and it is worth stating first
because every offset below depends on it. The kernel requires of each loadable
segment that

    p_offset ≡ p_vaddr   (mod the page size)

so that one `mmap` of the file can serve it. Satisfying that per segment means
tracking two counters that drift apart. Instead every byte of the image gets

    vaddr = BASE + file offset

for one BASE, which makes the congruence hold identically, for every segment,
with no arithmetic at all. The cost is that the gap between two segments is
the same in the file as in memory -- a page of padding on disk where a real
linker would have none. A page is not worth a second counter and the bugs it
brings.

WHAT IS DELIBERATELY NOT HERE. No shared libraries, no PLT, no GOT, no dynamic
symbol table, no `.eh_frame` rewriting, no garbage collection of unused
sections, no archive (`.a`) member selection. Each of those is a real feature
and none is needed to turn this compiler's own objects into a program; a
linker that claimed them and did them badly would be worse than one that says
what it does.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

from ..backend.objfile.elf import (
    EM_AARCH64, EM_X86_64, SHF_ALLOC, SHF_EXECINSTR, SHF_WRITE,
    STB_GLOBAL, STB_LOCAL, STB_WEAK,
)
from . import pewrite
from ..backend.objfile import coffread, elfread, machoread
from ..backend.objfile.elfread import (
    ElfError, InReloc, InSection, Relocatable, SHN_ABS, SHN_COMMON,
)

#: Where the image is loaded. 0x400000 is the traditional x86-64 text base and
#: is above the 64 KiB `mmap_min_addr` every Linux enforces, so a null
#: dereference still faults rather than reading the program's own header.
DEFAULT_BASE = 0x400000

#: The page size the congruence rule is stated in. 4 KiB on x86-64 and the
#: smallest AArch64 page; a kernel with larger pages still accepts an image
#: aligned to a divisor of its own page size, so this is the safe choice
#: rather than the host's answer.
PAGE = 0x1000

# ── ELF executable constants ────────────────────────────────────────────────
ET_EXEC = 2
PT_LOAD = 1
PT_GNU_STACK = 0x6474E551
PF_X, PF_W, PF_R = 1, 2, 4

_EHDR = "<16sHHIQQQIHHHHHH"
_PHDR = "<IIQQQQQQ"


class LinkFailed(Exception):
    """The link could not be completed. Carries a user-facing reason.

    ONE EXCEPTION FOR EVERY REASON, because from the caller's side they are
    the same event: the program was not produced and here is why. The driver
    turns it into a diagnostic; nothing catches it to distinguish cases.
    """

    def __init__(self, message: str, *, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


# ── the output image ────────────────────────────────────────────────────────

@dataclass(slots=True)
class Piece:
    """One input section, placed."""

    obj: int
    section: InSection
    #: Where its first byte lives, as a virtual address.
    addr: int
    #: Where its bytes go in the output buffer. -1 for `.bss`, which has none.
    at: int


@dataclass(slots=True)
class OutSection:
    """One output section: the input sections of one kind, concatenated."""

    name: str
    flags: int
    #: True for `.bss`: occupies memory and no file space.
    nobits: bool = False
    align: int = 1
    addr: int = 0
    size: int = 0
    pieces: list[Piece] = field(default_factory=list)


@dataclass(slots=True)
class Defined:
    """A global symbol that some object defines."""

    addr: int
    size: int
    weak: bool
    origin: str


def _round_up(value: int, to: int) -> int:
    return (value + to - 1) & ~(to - 1) if to > 1 else value


# ── step 2: layout ──────────────────────────────────────────────────────────

#: The four output sections, in image order, and the test that sends an input
#: section to each.
#:
#: ORDERED READ-ONLY FIRST so that the two segments are contiguous runs: the
#: executable and read-only sections make one `PT_LOAD`, the writable ones
#: make the other, and a section in the wrong group would need a third.
#:
#: `.rodata` IS ITS OWN SECTION EVEN THOUGH IT COULD LIVE IN `.text`. Merging
#: them costs nothing at run time and would mean a program's constant pool is
#: executable, which is a real difference in what the produced binary permits.
def _bucket(sec: InSection) -> str | None:
    """Which output section an input section belongs to, or None to drop it.

    A SECTION WITHOUT `SHF_ALLOC` IS NOT PART OF THE PROGRAM. `.comment`,
    `.note.GNU-stack` and the symbol and string tables all describe the object
    rather than belonging to the image, and copying them in would put bytes
    nothing points at into the middle of the text.
    """
    if not sec.is_alloc:
        return None
    if sec.is_exec:
        return ".text"
    if sec.is_bss:
        return ".bss"
    if sec.is_write:
        return ".data"
    return ".rodata"


@dataclass(slots=True)
class Reserved:
    """Space the LINKER contributes, rather than any input object.

    Today there is exactly one: the PE import table, whose size is known from
    the names before its address is, and whose address has to be known before
    its bytes can be written -- every pointer inside one is a virtual
    address. So it is laid out as a section like any other and filled in
    afterwards.
    """

    name: str
    size: int
    flags: int
    align: int = 8
    addr: int = 0


def _lay_out(objects: list[Relocatable], base: int,
             reserved: list[Reserved] | None = None, *,
             headroom: int | None = None,
             section_align: int = 1) -> tuple[
        list[OutSection], dict[tuple[int, int], int], int]:
    """Place every allocatable input section and give it an address.

    Returns the output sections, a map from (object, section index) to the
    address that section landed at, and the end of the image.

    THE FILE OFFSET AND THE ADDRESS ARE THE SAME COUNTER, less `base` -- see
    the module docstring. The one place they part company is `.bss`, which
    advances the address and not the offset, and that is why it is last.

    `headroom` is how many bytes the container's own headers need in front of
    the first section, and `section_align` how far apart the sections have to
    be. THE TWO CONTAINERS DISAGREE ON BOTH. An ELF loader reads SEGMENTS, so
    the sections inside one may be packed tight and the headers are the three
    structures below; a PE loader maps SECTIONS, and the format says each
    one's address is a multiple of `SectionAlignment` -- so on that side
    every section starts a page and the headers get a page of their own.
    Getting this wrong is not subtle: the first version wrote PE headers over
    the beginning of `.text`.
    """
    out = {
        ".text": OutSection(".text", SHF_ALLOC | SHF_EXECINSTR),
        ".rodata": OutSection(".rodata", SHF_ALLOC),
        ".data": OutSection(".data", SHF_ALLOC | SHF_WRITE),
        ".bss": OutSection(".bss", SHF_ALLOC | SHF_WRITE, nobits=True),
    }
    for i, obj in enumerate(objects):
        for sec in obj.sections:
            where = _bucket(sec)
            if where is None:
                continue
            got = out[where]
            got.align = max(got.align, sec.align)
            got.pieces.append(Piece(obj=i, section=sec, addr=0, at=0))

    # THE HEADERS COME FIRST AND ARE PART OF THE FIRST SEGMENT. A loader maps
    # whole pages, so the bytes before the text are mapped whether or not
    # anything says so -- and `PT_LOAD` starting at file offset 0 is what lets
    # a program read its own headers, which is how `sys.argv` and the auxv
    # walk find things on other platforms.
    phnum = 3  # two PT_LOAD plus PT_GNU_STACK
    cursor = (struct.calcsize(_EHDR) + phnum * struct.calcsize(_PHDR)
              if headroom is None else headroom)

    # A RESERVED AREA IS A SECTION OF ITS OWN, placed with the writable ones:
    # the loader writes into the import table, so it cannot share a page with
    # anything read-only.
    for got in reserved or ():
        out[got.name] = OutSection(got.name, got.flags,
                                   nobits=False, align=got.align)
        out[got.name].pieces.append(
            Piece(obj=-1, section=InSection(
                index=-1, name=got.name, kind=1, flags=got.flags,
                align=got.align, data=b"\0" * got.size, size=got.size),
                addr=0, at=0))

    where: dict[tuple[int, int], int] = {}
    order = [".text", ".rodata", ".data",
             *(r.name for r in reserved or ()), ".bss"]
    writable_started = False
    for name in order:
        sec = out[name]
        if not sec.pieces:
            continue
        if section_align > 1:
            # EVERY SECTION ON ITS OWN PAGE, which is what a PE asks for.
            cursor = _round_up(cursor, section_align)
        elif sec.flags & SHF_WRITE and not writable_started:
            # THE FIRST WRITABLE SECTION STARTS A NEW PAGE. Two segments
            # cannot share one: the kernel's protection is per page, so a
            # read-only byte on the same page as a writable one is writable.
            # The ones after it follow on, because they are all in that
            # second segment.
            cursor = _round_up(cursor, PAGE)
        writable_started = writable_started or bool(sec.flags & SHF_WRITE)
        cursor = _round_up(cursor, sec.align)
        sec.addr = base + cursor
        for piece in sec.pieces:
            cursor = _round_up(cursor, piece.section.align)
            piece.addr = base + cursor
            piece.at = -1 if sec.nobits else cursor
            where[(piece.obj, piece.section.index)] = piece.addr
            cursor += piece.section.size
        sec.size = base + cursor - sec.addr
    return [out[n] for n in order if out[n].pieces], where, base + cursor


# ── step 3: symbol resolution ───────────────────────────────────────────────

def _resolve(objects: list[Relocatable],
             where: dict[tuple[int, int], int],
             extra: dict[str, Defined] | None = None) -> tuple[
                 dict[str, Defined], list[list[int | None]]]:
    """Every global symbol's address, and every object's own symbol addresses.

    TWO ANSWERS BECAUSE THERE ARE TWO SCOPES. A relocation names a symbol by
    INDEX in its own object, and that entry may be a local -- a static
    function, a compiler-generated label -- which no other object may see. So
    each object gets a list parallel to its symbol table, and the globals get
    a shared map that the undefined entries are looked up in.

    A DUPLICATE STRONG DEFINITION IS AN ERROR AND A WEAK ONE IS NOT. That is
    the whole of what `STB_WEAK` means: a weak definition yields to a strong
    one wherever it is, and two weak ones are not a conflict. Nothing this
    compiler emits is weak today; the rule is here because an object the user
    brings might be, and silently taking the first would be a wrong program.
    """
    globals_: dict[str, Defined] = dict(extra or {})
    for i, obj in enumerate(objects):
        for sym in obj.symbols:
            if not sym.name or sym.is_local or sym.is_undefined:
                continue
            if sym.shndx == SHN_COMMON:
                raise LinkFailed(
                    f"{obj.origin}: {sym.name!r} is a COMMON symbol",
                    detail="a tentative definition has no address of its own "
                           "and this linker does not allocate one; nothing "
                           "uasm emits produces one, so this object came "
                           "from elsewhere")
            if sym.is_absolute:
                addr = sym.value
            else:
                at = where.get((i, sym.shndx))
                if at is None:
                    # A SYMBOL IN A SECTION THAT WAS DROPPED. Not an error:
                    # `.note.GNU-stack` and friends carry none, and a symbol
                    # in one is describing the object rather than the program.
                    continue
                addr = at + sym.value
            found = globals_.get(sym.name)
            if found is None:
                globals_[sym.name] = Defined(addr, sym.size, sym.is_weak,
                                             obj.origin)
            elif found.weak and not sym.is_weak:
                globals_[sym.name] = Defined(addr, sym.size, False, obj.origin)
            elif not found.weak and not sym.is_weak:
                raise LinkFailed(
                    f"{sym.name!r} is defined twice",
                    detail=f"first in {found.origin}, again in {obj.origin}")

    per_object: list[list[int | None]] = []
    for i, obj in enumerate(objects):
        addrs: list[int | None] = []
        for sym in obj.symbols:
            if sym.is_undefined:
                got = globals_.get(sym.name)
                addrs.append(None if got is None else got.addr)
            elif sym.is_absolute:
                addrs.append(sym.value)
            elif sym.shndx == SHN_COMMON:
                addrs.append(None)
            else:
                at = where.get((i, sym.shndx))
                addrs.append(None if at is None else at + sym.value)
        per_object.append(addrs)
    return globals_, per_object


def _undefined(objects: list[Relocatable],
               per_object: list[list[int | None]]) -> list[tuple[str, str]]:
    """Every name that is referenced by a relocation and defined nowhere.

    GATHERED FROM THE RELOCATIONS AND NOT FROM THE SYMBOL TABLES, which is the
    difference between "this object mentions a name" and "this object needs
    one". An object can carry an undefined entry nothing patches -- the
    emitters here add one per relocation, but an object from elsewhere may
    list more -- and reporting those would send the user looking for a
    dependency the program does not have.
    """
    missing: dict[str, str] = {}
    for i, obj in enumerate(objects):
        for rels in obj.relocs.values():
            for rel in rels:
                if per_object[i][rel.symbol] is None:
                    name = obj.symbols[rel.symbol].name or "<unnamed symbol>"
                    missing.setdefault(name, obj.origin)
    return sorted(missing.items())


# ── step 4: relocation ──────────────────────────────────────────────────────

def _signed_fits(value: int, bits: int) -> bool:
    return -(1 << (bits - 1)) <= value < (1 << (bits - 1))


def _patch_x86_64(buf: bytearray, at: int, rel: InReloc, s: int,
                  p: int, name: str) -> None:
    """One x86-64 relocation, applied.

    ONLY THE FOUR THIS COMPILER CAN PRODUCE, plus the two absolute forms an
    object from elsewhere might carry. `encode.py` emits exactly `PC32` and
    `PLT32`, both with an addend of -4 because the displacement is measured
    from the END of the instruction and the field sits four bytes before it.

    `PLT32` IS TREATED AS `PC32`, which is correct for a static link and is
    what every linker does when the target is defined locally: the PLT exists
    to reach a symbol whose address is not known until load time, and in a
    static image every address is known now.
    """
    a = rel.addend
    kind = rel.kind
    if kind in (2, 4):            # R_X86_64_PC32, R_X86_64_PLT32
        value = s + a - p
        if not _signed_fits(value, 32):
            raise LinkFailed(
                f"{name!r} is too far away to reach",
                detail=f"a 32-bit PC-relative field cannot hold {value}; the "
                       f"image is larger than 2 GiB")
        struct.pack_into("<i", buf, at, value)
    elif kind == 1:               # R_X86_64_64
        struct.pack_into("<Q", buf, at, (s + a) & 0xFFFFFFFFFFFFFFFF)
    elif kind == 10:              # R_X86_64_32
        value = s + a
        if not 0 <= value < (1 << 32):
            raise LinkFailed(f"{name!r} does not fit a 32-bit absolute field")
        struct.pack_into("<I", buf, at, value)
    elif kind == 11:              # R_X86_64_32S
        value = s + a
        if not _signed_fits(value, 32):
            raise LinkFailed(f"{name!r} does not fit a signed 32-bit field")
        struct.pack_into("<i", buf, at, value)
    else:
        raise LinkFailed(
            f"x86-64 relocation type {kind} is not implemented",
            detail=f"needed for {name!r}; the encoder here emits only PC32 "
                   f"and PLT32, so this object came from elsewhere")


#: The AArch64 `LDST<n>_ABS_LO12_NC` family, and how far the offset shifts.
#: The immediate in a scaled load/store counts ELEMENTS, not bytes, so the
#: low 12 bits of the address are divided by the access size before they are
#: placed -- which is also why the field cannot represent a misaligned one.
_LDST_SHIFT = {278: 0, 284: 1, 285: 2, 286: 3, 299: 4}


def _patch_aarch64(buf: bytearray, at: int, rel: InReloc, s: int,
                   p: int, name: str) -> None:
    """One AArch64 relocation, applied.

    THE IMMEDIATE IS NEVER CONTIGUOUS, which is the whole difficulty. AArch64
    packs instruction fields around the operand registers, so patching means
    reading the word, clearing exactly the bits the field owns, and putting
    the value back in the pieces the encoding uses -- `ADRP` in particular
    splits its 21-bit page offset into two bits at 29 and nineteen at 5.
    """
    kind = rel.kind
    a = rel.addend
    word, = struct.unpack_from("<I", buf, at)
    if kind == 257:               # R_AARCH64_ABS64
        struct.pack_into("<Q", buf, at, (s + a) & 0xFFFFFFFFFFFFFFFF)
        return
    if kind in (282, 283):        # JUMP26, CALL26
        value = s + a - p
        if value & 3:
            raise LinkFailed(f"{name!r} is not 4-byte aligned")
        value >>= 2
        if not _signed_fits(value, 26):
            raise LinkFailed(
                f"{name!r} is too far away for a branch",
                detail=f"a 26-bit branch reaches +-128 MiB and this needs "
                       f"{value * 4} bytes")
        word = (word & ~0x03FFFFFF) | (value & 0x03FFFFFF)
    elif kind == 275:             # ADR_PREL_PG_HI21
        value = ((s + a) >> 12) - (p >> 12)
        if not _signed_fits(value, 21):
            raise LinkFailed(f"{name!r} is too far away for ADRP")
        word = (word & ~((0x3 << 29) | (0x7FFFF << 5)))
        word |= (value & 0x3) << 29
        word |= ((value >> 2) & 0x7FFFF) << 5
    elif kind == 277:             # ADD_ABS_LO12_NC
        word = (word & ~(0xFFF << 10)) | (((s + a) & 0xFFF) << 10)
    elif kind in _LDST_SHIFT:
        shift = _LDST_SHIFT[kind]
        low = (s + a) & 0xFFF
        if low & ((1 << shift) - 1):
            raise LinkFailed(
                f"{name!r} is not aligned for the load or store that "
                f"reaches it")
        word = (word & ~(0xFFF << 10)) | ((low >> shift) << 10)
    else:
        raise LinkFailed(
            f"AArch64 relocation type {kind} is not implemented",
            detail=f"needed for {name!r}")
    struct.pack_into("<I", buf, at, word & 0xFFFFFFFF)


def _patch_coff_amd64(buf: bytearray, at: int, rel: InReloc, s: int,
                      p: int, name: str) -> None:
    """One COFF relocation for x86-64, applied.

    THE BIAS IS PART OF THE TYPE, not a number in the file. ELF says
    "relative to the field, and here is -4"; COFF says "relative to the byte
    AFTER the field" and stores nothing. `REL32_1` through `REL32_5` shift
    the reference point further on, which is how COFF spells a displacement
    in an instruction with bytes after it.
    """
    kind = rel.kind
    a = rel.addend
    if 0x0004 <= kind <= 0x0009:
        # REL32 is "past the field"; REL32_N adds N more.
        bias = 4 + (kind - 0x0004)
        value = s + a - (p + bias)
        if not _signed_fits(value, 32):
            raise LinkFailed(f"{name!r} is too far away to reach")
        struct.pack_into("<i", buf, at, value)
    elif kind == 0x0001:                  # ADDR64
        struct.pack_into("<Q", buf, at, (s + a) & 0xFFFFFFFFFFFFFFFF)
    elif kind == 0x0002:                  # ADDR32
        value = s + a
        if not 0 <= value < (1 << 32):
            raise LinkFailed(f"{name!r} does not fit a 32-bit absolute field")
        struct.pack_into("<I", buf, at, value)
    elif kind == 0x0003:                  # ADDR32NB: minus the image base
        value = s + a - _IMAGE_BASE[0]
        if not 0 <= value < (1 << 32):
            raise LinkFailed(f"{name!r} is not within 4 GiB of the image base")
        struct.pack_into("<I", buf, at, value)
    else:
        raise LinkFailed(
            f"COFF x86-64 relocation type {kind:#x} is not implemented",
            detail=f"needed for {name!r}")


#: WHERE THE IMAGE WAS LAID OUT, for the one relocation type that needs to
#: know. `ADDR32NB` is an address MINUS the image base, and the patch
#: functions are handed only S, A and P -- so the base is left here rather
#: than threaded through every signature for the one caller that reads it.
#: Set by `link` before any patching happens.
_IMAGE_BASE = [DEFAULT_BASE]


def _patch_coff_arm64(buf: bytearray, at: int, rel: InReloc, s: int,
                      p: int, name: str) -> None:
    """One COFF relocation for AArch64, applied.

    THE SAME BITFIELDS AS ELF, under different numbers -- which is the shape
    of most of this: the arithmetic is the architecture's and the numbering
    is the container's.
    """
    same = {0x0001: 257,          # ADDR64            -> R_AARCH64_ABS64
            0x0003: 283,          # BRANCH26          -> CALL26
            0x0004: 275,          # PAGEBASE_REL21    -> ADR_PREL_PG_HI21
            0x0006: 277}          # PAGEOFFSET_12A    -> ADD_ABS_LO12_NC
    if rel.kind not in same:
        raise LinkFailed(
            f"COFF AArch64 relocation type {rel.kind:#x} is not implemented",
            detail=f"needed for {name!r}")
    _patch_aarch64(buf, at, InReloc(rel.offset, rel.symbol, same[rel.kind],
                                    rel.addend), s, p, name)


def _patch_macho_x86_64(buf: bytearray, at: int, rel: InReloc, s: int,
                        p: int, name: str) -> None:
    """One Mach-O relocation for x86-64, applied.

    MACH-O PUTS `pcrel` AND `length` IN THE RECORD where ELF implies both
    from the type, so the arithmetic reads them rather than a table. What it
    does NOT put anywhere is an extra bias: a displacement is always from the
    byte after the four the field occupies, and an instruction with more
    bytes after it says so by storing a NEGATIVE ADDEND in the field --
    `X86_64_RELOC_SIGNED_4` is "signed displacement with a -4 addend", and
    the -4 is already there. So one formula serves every PC-relative type.
    """
    kind = rel.kind
    a = rel.addend
    width = 1 << rel.length
    if kind == 0:                         # X86_64_RELOC_UNSIGNED
        if rel.pcrel:
            raise LinkFailed(f"{name!r}: X86_64_RELOC_UNSIGNED cannot be "
                             f"PC-relative")
        form = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}[width]
        struct.pack_into(form, buf, at, (s + a) & ((1 << (width * 8)) - 1))
        return
    if kind in (3, 4):                    # GOT_LOAD, GOT
        # THERE IS NO GLOBAL OFFSET TABLE IN A STATIC IMAGE, and there does
        # not need to be: the symbol's address is known now. `GOT_LOAD` is
        # `movq sym@GOTPCREL(%rip),%reg`, which becomes `leaq sym(%rip),%reg`
        # by changing one opcode byte -- the relaxation every static linker
        # performs. A bare `GOT` is a reference to the SLOT and cannot be
        # relaxed, so it is refused.
        if kind == 4:
            raise LinkFailed(
                f"{name!r} is referenced through the global offset table",
                detail="a static image has none; only the `GOT_LOAD` form, "
                       "which relaxes to an address, can be linked here")
        if at >= 2 and buf[at - 2] == 0x8B:
            buf[at - 2] = 0x8D            # mov -> lea
        else:
            raise LinkFailed(
                f"{name!r}: a GOT load this linker cannot relax",
                detail="the four bytes are expected to follow a `movq` "
                       "opcode (0x8b)")
    elif kind not in (1, 2, 6, 7, 8):     # SIGNED, BRANCH, SIGNED_1/2/4
        raise LinkFailed(
            f"Mach-O x86-64 relocation type {kind} is not implemented",
            detail=f"needed for {name!r}")
    if width != 4:
        raise LinkFailed(
            f"{name!r}: a PC-relative Mach-O relocation of {width} bytes",
            detail="every type this linker patches is four bytes wide")
    value = s + a - (p + 4)
    if not _signed_fits(value, 32):
        raise LinkFailed(f"{name!r} is too far away to reach")
    struct.pack_into("<i", buf, at, value)


#: The load/store forms `ARM64_RELOC_PAGEOFF12` may be patching, and the ELF
#: relocation that spells the same arithmetic.
#:
#: MACH-O HAS ONE TYPE WHERE ELF HAS SIX. ELF names the access width in the
#: relocation -- `LDST8`, `LDST16` and the rest -- because a scaled load's
#: immediate counts ELEMENTS; Mach-O expects the linker to read the
#: instruction and work it out. So that is what happens here.
_PAGEOFF_LDST = {0: 278, 1: 284, 2: 285, 3: 286, 4: 299}


def _pageoff_kind(word: int) -> int:
    """Which ELF relocation `ARM64_RELOC_PAGEOFF12` means, for this word."""
    if word & 0x3B000000 != 0x39000000:
        # Not a scaled load or store: an `ADD` immediate, whose field counts
        # bytes and needs no shift.
        return 277                        # R_AARCH64_ADD_ABS_LO12_NC
    size = (word >> 30) & 3
    if size == 0 and word & 0x04800000 == 0x04800000:
        size = 4                          # the 128-bit SIMD form
    return _PAGEOFF_LDST[size]


def _patch_macho_arm64(buf: bytearray, at: int, rel: InReloc, s: int,
                       p: int, name: str) -> None:
    """One Mach-O relocation for AArch64, applied.

    THE SAME BITFIELDS AS ELF under different numbers, as in COFF -- with the
    one difference that `PAGEOFF12` has to be told apart from ELF's six by
    looking at the instruction. See `_pageoff_kind`.
    """
    kind = rel.kind
    if kind == 0:                         # ARM64_RELOC_UNSIGNED
        width = 1 << rel.length
        form = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}[width]
        struct.pack_into(form, buf, at,
                         (s + rel.addend) & ((1 << (width * 8)) - 1))
        return
    if kind in (5, 6):                    # GOT_LOAD_PAGE21, GOT_LOAD_PAGEOFF12
        raise LinkFailed(
            f"{name!r} is referenced through the global offset table",
            detail="a static image has none; relaxing an AArch64 GOT load "
                   "means rewriting an `ldr` as an `add`, which this linker "
                   "does not do")
    word, = struct.unpack_from("<I", buf, at)
    same = {2: 283,                       # BRANCH26      -> CALL26
            3: 275,                       # PAGE21        -> ADR_PREL_PG_HI21
            4: _pageoff_kind(word)}       # PAGEOFF12     -> one of seven
    if kind not in same:
        raise LinkFailed(
            f"Mach-O AArch64 relocation type {kind} is not implemented",
            detail=f"needed for {name!r}")
    _patch_aarch64(buf, at, InReloc(rel.offset, rel.symbol, same[kind],
                                    rel.addend), s, p, name)


#: WHICH PATCHER SERVES WHICH (container, machine). The numbers mean
#: different things in different containers -- 4 is `R_X86_64_PLT32` in ELF,
#: `IMAGE_REL_AMD64_REL32` in COFF and `X86_64_RELOC_GOT` in Mach-O -- so the
#: container is half the key.
_PATCH = {
    ("elf", EM_X86_64): _patch_x86_64,
    ("elf", EM_AARCH64): _patch_aarch64,
    ("coff", coffread.IMAGE_FILE_MACHINE_AMD64): _patch_coff_amd64,
    ("coff", coffread.IMAGE_FILE_MACHINE_ARM64): _patch_coff_arm64,
    ("macho", machoread.CPU_TYPE_X86_64): _patch_macho_x86_64,
    ("macho", machoread.CPU_TYPE_ARM64): _patch_macho_arm64,
}


def read_object(blob: bytes, origin: str):
    """One input object, whatever container it is in.

    TOLD APART BY THE BYTES rather than by the file name: `uasm link a.o b.o`
    names two files and says nothing about their format, and a `.o` that is
    really a COFF object is an ordinary thing to have on Windows.
    """
    if elfread.is_elf(blob):
        return elfread.read(blob, origin)
    if coffread.is_coff(blob):
        return coffread.read(blob, origin)
    if machoread.is_macho(blob):
        return machoread.read(blob, origin)
    raise LinkFailed(
        f"{origin}: not an object file this linker reads",
        detail="ELF, PE/COFF and Mach-O relocatables are understood")


# ── the whole of it ─────────────────────────────────────────────────────────

@dataclass(slots=True)
class Image:
    """A linked program, and what is worth knowing about it."""

    data: bytes
    entry: int
    machine: int
    sections: list[OutSection]
    symbols: dict[str, Defined]
    #: The container the inputs came out of, and therefore the one the
    #: program goes into: `executable` writes ELF for "elf" and PE for
    #: "coff". A link never mixes them -- see `link`.
    fmt: str = "elf"
    #: Where the image was laid out. Carried so that `executable` need not be
    #: told again, and cannot be told something different.
    base: int = DEFAULT_BASE
    #: The PE import directory's address and size, both zero for ELF and for
    #: a PE that imports nothing.
    import_rva: int = 0
    import_size: int = 0
    #: The Import Address Table's address and size, which is a data directory
    #: of its own: the loader writes into that range and some of them make it
    #: writable ahead of time on the strength of this.
    iat: tuple[int, int] = (0, 0)


def link(inputs: list[tuple[str, bytes]], *, entry: str = "_start",
         base: int = DEFAULT_BASE) -> Image:
    """Link relocatable objects into a static executable image.

    `inputs` is (origin, bytes) so that an object held in memory -- which is
    what a backend hands the driver -- needs no temporary file to be linked,
    and one read from disk still reports its path when something is wrong.
    """
    if not inputs:
        raise LinkFailed("nothing to link")

    objects: list[Relocatable] = []
    for origin, blob in inputs:
        try:
            objects.append(read_object(blob, origin))
        except (ElfError, coffread.CoffError,
                machoread.MachoError) as exc:
            raise LinkFailed(str(exc)) from None

    formats = {getattr(obj, "fmt", "elf") for obj in objects}
    if len(formats) > 1:
        raise LinkFailed(
            "the inputs are in different object formats",
            detail=", ".join(f"{obj.origin} ({getattr(obj, 'fmt', 'elf')})"
                             for obj in objects))
    fmt = formats.pop()
    machines = {obj.machine for obj in objects}
    if len(machines) > 1:
        named = ", ".join(f"{obj.origin} (machine {obj.machine})"
                          for obj in objects)
        raise LinkFailed("the inputs are for different machines",
                         detail=named)
    machine = machines.pop()
    if (fmt, machine) not in _PATCH:
        raise LinkFailed(
            f"no relocation support for {fmt} machine {machine:#x}",
            detail="; ".join(f"{f} {m:#x}" for f, m in sorted(_PATCH)))

    _IMAGE_BASE[0] = base

    # ── what the linker itself has to contribute ────────────────────────────
    #
    # A PE REACHES THE KERNEL THROUGH A DLL, so its undefined `__imp_` names
    # are not missing symbols but imports, and the table that answers them is
    # the linker's to build. Its SIZE is known from the names; its CONTENTS
    # need its address, so it is reserved now and filled once the layout is
    # decided. See `pewrite.py`.
    reserved: list[Reserved] = []
    by_dll: dict[str, list[str]] = {}
    if fmt == "coff":
        wanted = {obj.symbols[rel.symbol].name
                  for obj in objects for rels in obj.relocs.values()
                  for rel in rels if obj.symbols[rel.symbol].name}
        have = {sym.name for obj in objects for sym in obj.symbols
                if sym.name and not sym.is_undefined}
        try:
            by_dll = pewrite.imports_needed(wanted - have)
        except pewrite.PeError as exc:
            raise LinkFailed(str(exc)) from None
        if by_dll:
            blob, _, _ = pewrite.import_section(by_dll, 0)
            reserved.append(Reserved(".idata", len(blob),
                                     SHF_ALLOC | SHF_WRITE))

    if fmt == "coff":
        sections, place, end = _lay_out(
            objects, base, reserved,
            headroom=pewrite.header_space(PAGE), section_align=PAGE)
    else:
        sections, place, end = _lay_out(objects, base, reserved)
    if not sections:
        raise LinkFailed("the inputs contain no loadable sections")
    globals_, per_object = _resolve(objects, place)

    import_rva = import_size = 0
    import_blob = b""
    iat = (0, 0)
    if by_dll:
        area = next(s for s in sections if s.name == ".idata")
        import_blob, slots, iat = pewrite.import_section(by_dll,
                                                         area.addr - base)
        import_rva, import_size = area.addr - base, len(import_blob)
        # THE SLOTS COME BACK AS RVAs, because that is what everything
        # inside an import table is; a relocation is patched from ADDRESSES.
        # The first version seeded the RVA and every call through the IAT
        # went four megabytes short of it.
        seeded = {name: Defined(base + addr, 8, False, "<import table>")
                  for name, addr in slots.items()}
        # ASKED AGAIN WITH THE SLOTS IN HAND, rather than patched into the
        # answer: `per_object` was built from a table that did not have them,
        # so every `__imp_` reference in it is still None. Only the SLOTS are
        # seeded -- handing back the whole map would have every object's own
        # definitions arrive a second time, which `_resolve` correctly calls
        # a duplicate.
        globals_, per_object = _resolve(objects, place, extra=seeded)

    missing = _undefined(objects, per_object)
    if missing:
        lines = "\n".join(f"  {name}  (needed by {origin})"
                          for name, origin in missing)
        raise LinkFailed(
            f"{len(missing)} undefined "
            f"symbol{'s' if len(missing) > 1 else ''}",
            detail=lines)

    if entry not in globals_:
        raise LinkFailed(
            f"the entry symbol {entry!r} is not defined",
            detail="a program needs somewhere to start; pass a different "
                   "name if this one is wrong")

    # ── the image buffer ────────────────────────────────────────────────────
    #
    # SIZED FROM THE LAST SECTION THAT HAS BYTES. `.bss` is at the end and
    # contributes none, so the file stops where `.data` does -- which is what
    # makes a program with a megabyte of zeroed state a small file.
    filesz = 0
    for sec in sections:
        if not sec.nobits:
            filesz = max(filesz, sec.addr - base + sec.size)
    buf = bytearray(filesz)
    for sec in sections:
        for piece in sec.pieces:
            if piece.at >= 0:
                buf[piece.at:piece.at + len(piece.section.data)] = \
                    piece.section.data

    patch = _PATCH[(fmt, machine)]
    for i, obj in enumerate(objects):
        for sec_index, rels in obj.relocs.items():
            target = place.get((i, sec_index))
            if target is None:
                continue  # a relocation in a section that is not loaded
            for rel in rels:
                addr = target + rel.offset
                at = addr - base
                if not 0 <= at < len(buf):
                    raise LinkFailed(
                        f"{obj.origin}: a relocation patches outside the "
                        f"image")
                s = per_object[i][rel.symbol]
                assert s is not None  # `_undefined` ran first
                patch(buf, at, rel, s, addr,
                      obj.symbols[rel.symbol].name or "<unnamed>")

    if import_blob:
        area = next(s for s in sections if s.name == ".idata")
        at = area.addr - base
        buf[at:at + len(import_blob)] = import_blob
    return Image(data=bytes(buf), entry=globals_[entry].addr, machine=machine,
                 sections=sections, symbols=globals_, fmt=fmt, base=base,
                 import_rva=import_rva, import_size=import_size, iat=iat)


def executable(image: Image, *, base: int | None = None) -> bytes:
    """The image, wrapped in the container its inputs came out of."""
    where = image.base if base is None else base
    if image.fmt == "coff":
        try:
            return pewrite.executable(image, base=where, page=PAGE,
                                      import_rva=image.import_rva,
                                      import_size=image.import_size,
                                      iat=image.iat)
        except pewrite.PeError as exc:
            raise LinkFailed(str(exc)) from None
    return _elf_executable(image, base=where)


def _elf_executable(image: Image, *, base: int = DEFAULT_BASE) -> bytes:
    """The image, wrapped in an ELF executable the kernel will load.

    NO SECTION HEADER TABLE. A loader reads program headers and nothing else;
    section headers are for linkers and debuggers, and this file is neither's
    input. `file` reports such a binary as "statically linked, stripped",
    which is what it is.

    THREE PROGRAM HEADERS: one `PT_LOAD` for the read-only half, one for the
    writable half, and `PT_GNU_STACK` with no `PF_X` so the stack is not
    executable. The last is not optional in practice -- without it a kernel
    that honours the marker assumes the worst.
    """
    ro_end = 0
    rw_start = None
    rw_filesz = 0
    rw_memsz = 0
    for sec in image.sections:
        if sec.flags & SHF_WRITE:
            if rw_start is None:
                rw_start = sec.addr
            rw_memsz = max(rw_memsz, sec.addr + sec.size - rw_start)
            if not sec.nobits:
                rw_filesz = max(rw_filesz, sec.addr + sec.size - rw_start)
        else:
            ro_end = max(ro_end, sec.addr + sec.size)

    ehsize = struct.calcsize(_EHDR)
    phentsize = struct.calcsize(_PHDR)
    phnum = 3
    phdrs = []
    # The read-only segment starts at the very beginning so that the headers
    # are mapped with it -- see `_lay_out`.
    ro_filesz = max(ro_end - base, ehsize + phnum * phentsize)
    phdrs.append((PT_LOAD, PF_R | PF_X, 0, base, base,
                  ro_filesz, ro_filesz, PAGE))
    if rw_start is not None:
        phdrs.append((PT_LOAD, PF_R | PF_W, rw_start - base, rw_start,
                      rw_start, rw_filesz, rw_memsz, PAGE))
    else:
        # A PROGRAM WITH NO WRITABLE DATA still gets the header, because the
        # count is fixed in the layout: `_lay_out` reserved room for three and
        # a short table would leave the text starting in the wrong place.
        phdrs.append((PT_LOAD, PF_R | PF_W, 0, base, base, 0, 0, PAGE))
    phdrs.append((PT_GNU_STACK, PF_R | PF_W, 0, 0, 0, 0, 0, 0x10))

    ident = (b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\0" * 8)
    head = struct.pack(
        _EHDR, ident, ET_EXEC, image.machine, 1, image.entry, ehsize, 0, 0,
        ehsize, phentsize, phnum, 64, 0, 0)
    table = b"".join(struct.pack(_PHDR, *one) for one in phdrs)

    out = bytearray(image.data)
    if len(out) < len(head) + len(table):
        out.extend(b"\0" * (len(head) + len(table) - len(out)))
    out[0:len(head)] = head
    out[len(head):len(head) + len(table)] = table
    return bytes(out)


def link_files(paths: list[Path], *, entry: str = "_start",
               base: int = DEFAULT_BASE) -> bytes:
    """Read, link and wrap -- the whole job, for a caller with paths."""
    inputs = []
    for path in paths:
        try:
            inputs.append((str(path), path.read_bytes()))
        except OSError as exc:
            raise LinkFailed(f"cannot read {path}: {exc.strerror}") from None
    return executable(link(inputs, entry=entry, base=base), base=base)


__all__ = ["DEFAULT_BASE", "Image", "LinkFailed", "OutSection", "executable",
           "link", "link_files"]
