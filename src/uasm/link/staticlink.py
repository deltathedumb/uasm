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
from ..backend.objfile.elfread import (
    ElfError, InReloc, InSection, Relocatable, SHN_ABS, SHN_COMMON, read,
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


def _lay_out(objects: list[Relocatable], base: int) -> tuple[
        list[OutSection], dict[tuple[int, int], int], int]:
    """Place every allocatable input section and give it an address.

    Returns the output sections, a map from (object, section index) to the
    address that section landed at, and the end of the image.

    THE FILE OFFSET AND THE ADDRESS ARE THE SAME COUNTER, less `base` -- see
    the module docstring. The one place they part company is `.bss`, which
    advances the address and not the offset, and that is why it is last.
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
    cursor = struct.calcsize(_EHDR) + phnum * struct.calcsize(_PHDR)

    where: dict[tuple[int, int], int] = {}
    order = [".text", ".rodata", ".data", ".bss"]
    for name in order:
        sec = out[name]
        if not sec.pieces:
            continue
        # A WRITABLE SECTION STARTS A NEW PAGE. Two segments cannot share one:
        # the kernel's protection is per page, so a read-only byte on the same
        # page as a writable one is writable.
        if name == ".data" or (name == ".bss" and not out[".data"].pieces):
            cursor = _round_up(cursor, PAGE)
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
             where: dict[tuple[int, int], int]) -> tuple[
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
    globals_: dict[str, Defined] = {}
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


_PATCH = {EM_X86_64: _patch_x86_64, EM_AARCH64: _patch_aarch64}


# ── the whole of it ─────────────────────────────────────────────────────────

@dataclass(slots=True)
class Image:
    """A linked program, and what is worth knowing about it."""

    data: bytes
    entry: int
    machine: int
    sections: list[OutSection]
    symbols: dict[str, Defined]


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
            objects.append(read(blob, origin))
        except ElfError as exc:
            raise LinkFailed(str(exc)) from None

    machines = {obj.machine for obj in objects}
    if len(machines) > 1:
        named = ", ".join(f"{obj.origin} (machine {obj.machine})"
                          for obj in objects)
        raise LinkFailed("the inputs are for different machines",
                         detail=named)
    machine = machines.pop()
    if machine not in _PATCH:
        raise LinkFailed(
            f"no relocation support for ELF machine {machine}",
            detail="x86-64 (62) and AArch64 (183) are implemented")

    sections, place, end = _lay_out(objects, base)
    if not sections:
        raise LinkFailed("the inputs contain no loadable sections")
    globals_, per_object = _resolve(objects, place)

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

    patch = _PATCH[machine]
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

    return Image(data=bytes(buf), entry=globals_[entry].addr, machine=machine,
                 sections=sections, symbols=globals_)


def executable(image: Image, *, base: int = DEFAULT_BASE) -> bytes:
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
