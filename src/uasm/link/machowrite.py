"""Writing a Mach-O executable, and signing it.

`staticlink.py` lays out, resolves and patches; this is the container it
writes the result into when the inputs were Mach-O. It is the third of
these, after `staticlink._elf_executable` and `pewrite.executable`, and it
is the longest for two reasons that have nothing to do with headers.

THE FIRST IS THAT THE KERNEL DOES NOT READ `LC_MAIN`'S ENTRY OFFSET. The
comment in XNU's `load_main` says so in as many words -- dyld uses it -- and
what that function does instead is set `needs_dynlinker`, which fails the
load a moment later when there is no `LC_LOAD_DYLINKER`. A static image has
no dyld, so the entry point has to be handed over the way it was before
`LC_MAIN` existed: `LC_UNIXTHREAD`, which carries A WHOLE REGISTER STATE
with the program counter as one register in it. That is why this file knows
what an `x86_THREAD_STATE64` looks like. (The kernel does read `LC_MAIN`'s
`stacksize`; it is only the entry offset it leaves to dyld.)

THE SECOND IS THAT AN ARM64 MAC WILL NOT RUN AN UNSIGNED BINARY. Not "will
warn": it dies before its first instruction. The signature that gets past
that is an AD-HOC one -- no certificate, no Apple, no network -- and it is
nothing but a table of SHA-256 hashes, one per 4 KiB page of the file,
wrapped in two headers. A linker that could not produce one would be a
linker whose output does not run on half the Macs there are, so this one
produces it.

AND A THIRD THING, WHICH IS WHY THE ARM64 IMAGE THIS WRITES IS WELL FORMED
AND STILL WILL NOT RUN. `parse_machfile` gates `MH_EXECUTE` before any entry
command is looked at:

    case MH_EXECUTE:
        if (header->flags & MH_DYLDLINK) { ... needs_dynlinker = TRUE; }
        else if (header->cputype == CPU_TYPE_X86_64) { /* allowed */ }
        else { #if !(DEVELOPMENT || DEBUG) return LOAD_FAILURE; #endif }

So a static `MH_EXECUTE` is permitted on x86-64 BY NAME and refused on
arm64 on a release kernel -- and setting `MH_DYLDLINK` to get past the gate
sets `needs_dynlinker`, which then demands the dyld this image does not
have. There is no arrangement of load commands that loads a dyld-free arm64
executable on a shipping macOS. The image is written anyway, because it is
correct and a development kernel or a later change in Apple's policy would
take it, and `arm64_static_will_not_load` says so where a caller can see it
rather than leaving the user to find out on a Mac.

WHAT IS CONVENTION AND WHAT IS NOT, because the difference matters when
something does not load:

  * `__PAGEZERO` -- a segment at address zero, four gigabytes of it, with no
    permissions -- is NOT a convention. `load_machfile` requires the map's
    minimum offset to be at least one page for any 64-bit executable and at
    least 0x100000000 for an arm64 one, and only a segment with `vmaddr` 0,
    `filesize` 0, `vmsize` non-zero and BOTH protections `VM_PROT_NONE`
    raises it. A non-zero `maxprot` silently disqualifies it. The image
    starts at 0x100000000 because that is where `__PAGEZERO` ends.
  * EXACTLY ONE SEGMENT MAY MAP FILE OFFSET ZERO with contents, it must be
    readable and executable, and at least one must exist -- so `__TEXT`
    starts at file offset zero and holds the Mach header and the load
    commands. `__PAGEZERO` is exempt because its `filesize` is zero.
  * THE ENTRY MUST LAND INSIDE A SEGMENT WHOSE `initprot` IS EXACTLY
    READ|EXECUTE. For an `LC_UNIXTHREAD` image the kernel checks this and
    fails the load otherwise, so an entry in `__DATA` is not a program that
    crashes -- it is a program that does not start.
  * A SEGMENT'S ADDRESS AND FILE OFFSET ARE PAGE-ALIGNED. The page is 4 KiB
    on Intel and 16 KiB on Apple Silicon, and the kernel enforces the one
    for the architecture it is running.
  * The sections inside a segment are not page-aligned, unlike PE. Mach-O
    maps SEGMENTS.

NOTHING HERE HAS BEEN EXECUTED, for the same reason `freestanding.py`'s
Windows floor has not: there is no macOS on the machine this was written on.
What is checked is that `llvm-objdump --macho` reads back every structure
this writes, and that the signature's hashes are the hashes of the bytes
they claim to cover. That is a weaker claim than the Linux floor's, and it
is the honest one.
"""
from __future__ import annotations

import hashlib
import struct

from ..backend.objfile.macho import (
    CPU_SUBTYPE_ARM64_ALL, CPU_SUBTYPE_X86_64_ALL, CPU_TYPE_ARM64,
    CPU_TYPE_X86_64, LC_SEGMENT_64, MH_MAGIC_64,
    VM_PROT_EXECUTE, VM_PROT_READ, VM_PROT_WRITE,
)

MH_EXECUTE = 2
#: The image has no undefined symbols. TRUE OF EVERY IMAGE THIS WRITES, by
#: construction: the link refuses one that has.
MH_NOUNDEFS = 0x1
#: NOT `MH_PIE`. This image carries no relocation information, so it cannot
#: be moved, and a flag saying it can would be a lie the kernel acts on.

LC_UNIXTHREAD = 0x5
LC_CODE_SIGNATURE = 0x1D

#: `thread_status.h`: the flavour that describes the general-purpose
#: registers, the number of 32-bit words it takes, and the BYTE OFFSET of the
#: program counter inside it.
#:
#: x86: 21 registers of 64 bits is 42 words, and `rip` is the seventeenth --
#: after rax, rbx, rcx, rdx, rdi, rsi, rbp, rsp and r8 through r15.
#: arm64: x0 through x28 is 29, then fp, lr, sp and pc is 33 registers of 64
#: bits, then `cpsr` and a word of padding: 68 words, with `pc` last of the
#: wide ones.
_THREAD_STATE = {
    CPU_TYPE_X86_64: (4, 42, 16 * 8),     # x86_THREAD_STATE64
    CPU_TYPE_ARM64: (6, 68, 32 * 8),      # ARM_THREAD_STATE64
}

#: The page a kernel maps in, which is not the same on the two machines.
PAGE_OF = {CPU_TYPE_X86_64: 0x1000, CPU_TYPE_ARM64: 0x4000}

#: Where a 64-bit macOS image is loaded: immediately above `__PAGEZERO`.
DEFAULT_BASE = 0x100000000
PAGEZERO_SIZE = 0x100000000

_SUBTYPE_OF = {CPU_TYPE_X86_64: CPU_SUBTYPE_X86_64_ALL,
               CPU_TYPE_ARM64: CPU_SUBTYPE_ARM64_ALL}

#: Which Mach-O segment and section each of the linker's four buckets is.
#: The names are the platform's, not this compiler's -- see
#: `backend/objfile/macho.py`, which says why a section called `.text` is not
#: a section a Mach-O tool will treat as code.
_PLACE = {
    ".text": ("__TEXT", "__text"),
    ".rodata": ("__TEXT", "__const"),
    ".data": ("__DATA", "__data"),
    ".bss": ("__DATA", "__bss"),
}

S_REGULAR = 0x0
S_ZEROFILL = 0x1
S_ATTR_PURE_INSTRUCTIONS = 0x80000000
S_ATTR_SOME_INSTRUCTIONS = 0x00000400

_HEADER = "<IiiIIIII"
_SEGMENT = "<II16sQQQQiiII"
_SECTION = "<16s16sQQIIIIIIII"


class MachoWriteError(Exception):
    """An image this cannot write. Carries a user-facing reason."""


def _fixed(text: str, width: int = 16) -> bytes:
    raw = text.encode("ascii")
    if len(raw) > width:
        raise MachoWriteError(f"{text!r} does not fit a Mach-O name")
    return raw.ljust(width, b"\0")


def _round_up(value: int, to: int) -> int:
    return (value + to - 1) & ~(to - 1) if to > 1 else value


def arm64_static_will_not_load(machine: int) -> str:
    """Why this image will not run, if it will not. Empty when it will.

    ASKED OF THE MACHINE AND ANSWERED IN WORDS, because there is nothing to
    fix. See the module docstring: a release macOS kernel refuses a static
    `MH_EXECUTE` for any cputype but x86-64, before it looks at a single
    load command, and the flag that gets past the gate demands the dynamic
    linker this image does not have.
    """
    if machine != CPU_TYPE_ARM64:
        return ""
    return ("a static arm64 macOS executable is refused by the kernel: "
            "`parse_machfile` allows an MH_EXECUTE without MH_DYLDLINK only "
            "for CPU_TYPE_X86_64, and setting MH_DYLDLINK would require an "
            "LC_LOAD_DYLINKER that a freestanding image has no use for. The "
            "image written here is well formed and a development kernel "
            "will take it")


def header_space(page: int, segments: int = 4, sections: int = 4,
                 thread_words: int = 68) -> int:
    """How many bytes the headers need in front of the first section.

    ASKED BEFORE THE LAYOUT, as `pewrite.header_space` is and for the same
    reason: the answer decides where the first section goes, so it cannot be
    computed from where the sections went. The defaults are more of each than
    this linker produces.
    """
    commands = (72 * segments + 80 * sections
                + (16 + 4 * thread_words)       # LC_UNIXTHREAD
                + 16)                            # LC_CODE_SIGNATURE
    return _round_up(32 + commands, page)


# ── the ad-hoc code signature ───────────────────────────────────────────────
#
# WHAT IT IS, in one paragraph, because the names are worse than the thing.
# A `SuperBlob` is a length, a count, and a table of (what, where) pairs. The
# only entry an ad-hoc signature needs is a `CodeDirectory`, which is a
# header, an identifier string, and then a SHA-256 hash of every 4 KiB page
# of the file up to the point where the signature itself begins. Nothing is
# encrypted, nothing is certified, and there is no CMS blob: "ad-hoc" means
# the hashes vouch for the file and nobody vouches for the hashes. That is
# enough for the kernel to map the image, which is all that is being asked.
#
# EVERY FIELD IS BIG-ENDIAN, which is the trap: the rest of the file is
# little-endian and a signature written in the wrong order is a signature the
# kernel reads as an enormous length.

CSMAGIC_EMBEDDED_SIGNATURE = 0xFADE0CC0
CSMAGIC_CODEDIRECTORY = 0xFADE0C02
CSSLOT_CODEDIRECTORY = 0

#: The version that has the `execSeg` fields, which say which part of the
#: file is the executable segment. An arm64 kernel reads them.
CS_CODEDIRECTORY_VERSION = 0x20400
#: The fixed part of a `CodeDirectory` of that version, in bytes.
_CD_HEADER = 88

CS_ADHOC = 0x0000002
#: "A linker made this, not a signing tool." Both are set by `lld` and by
#: `ld64`, and the second is what lets the image be re-signed later without
#: the ad-hoc signature being mistaken for a real one.
CS_LINKER_SIGNED = 0x0020000
CS_EXECSEG_MAINBINARY = 0x1

CS_HASHTYPE_SHA256 = 2
CS_HASH_SIZE_SHA256 = 32
#: THE HASH PAGE IS ALWAYS 4 KiB, on both machines. It is the unit the
#: signature is divided into, and it has nothing to do with the page the
#: kernel maps -- an arm64 image is mapped in 16 KiB pages and hashed in
#: 4 KiB ones.
CS_PAGE_SHIFT = 12
CS_PAGE = 1 << CS_PAGE_SHIFT

#: Where the signature starts, rounded up to. `ld64` and `lld` both 16-align
#: it inside `__LINKEDIT`.
_SIGNATURE_ALIGN = 16


def _signature_size(identifier: str, code_limit: int, page: int) -> int:
    """How long the signature will be, asked before it can be written.

    IT HAS TO BE KNOWN IN ADVANCE because `LC_CODE_SIGNATURE` records it,
    and that load command is INSIDE the bytes the signature hashes. A writer
    that measured afterwards would change the file it had just hashed.
    """
    del page
    slots = -(-code_limit // CS_PAGE)
    ident = len(identifier.encode("ascii")) + 1
    return 12 + 8 + _CD_HEADER + ident + CS_HASH_SIZE_SHA256 * slots


def _signature(blob: bytes, identifier: str, page: int,
               exec_base: int = 0, exec_limit: int = 0) -> bytes:
    """The `SuperBlob` for `blob`, which must already be its final bytes.

    `blob` ENDS WHERE THE SIGNATURE BEGINS: everything in it is hashed and
    nothing after it is, which is what `codeLimit` says. Handing this
    anything but the finished image would sign a file that does not exist.
    """
    del page
    code_limit = len(blob)
    slots = -(-code_limit // CS_PAGE)
    ident = identifier.encode("ascii") + b"\0"

    ident_at = _CD_HEADER
    hash_at = ident_at + len(ident)
    cd_length = hash_at + CS_HASH_SIZE_SHA256 * slots

    cd = bytearray()
    cd += struct.pack(
        ">IIIIIIIII", CSMAGIC_CODEDIRECTORY, cd_length,
        CS_CODEDIRECTORY_VERSION, CS_ADHOC | CS_LINKER_SIGNED,
        hash_at, ident_at, 0, slots, code_limit)
    cd += struct.pack(">BBBB", CS_HASH_SIZE_SHA256, CS_HASHTYPE_SHA256,
                      0, CS_PAGE_SHIFT)
    cd += struct.pack(">I", 0)                       # spare2
    cd += struct.pack(">I", 0)                       # scatterOffset
    cd += struct.pack(">I", 0)                       # teamOffset
    cd += struct.pack(">I", 0)                       # spare3
    cd += struct.pack(">Q", 0)                       # codeLimit64
    cd += struct.pack(">QQQ", exec_base, exec_limit, CS_EXECSEG_MAINBINARY)
    assert len(cd) == _CD_HEADER, (len(cd), _CD_HEADER)
    cd += ident
    for i in range(slots):
        chunk = blob[i * CS_PAGE:(i + 1) * CS_PAGE]
        cd += hashlib.sha256(chunk).digest()
    assert len(cd) == cd_length, (len(cd), cd_length)

    total = 12 + 8 + cd_length
    out = struct.pack(">III", CSMAGIC_EMBEDDED_SIGNATURE, total, 1)
    out += struct.pack(">II", CSSLOT_CODEDIRECTORY, 12 + 8)
    return out + bytes(cd)


def _bucket_of(section) -> tuple[str, str]:
    """Which Mach-O segment and section one of the linker's buckets is."""
    try:
        return _PLACE[section.name]
    except KeyError:
        raise MachoWriteError(
            f"no Mach-O segment is known for {section.name!r}",
            ) from None


def _section_flags(section) -> int:
    """A section's Mach-O flags, from the linker's ELF-shaped ones."""
    if section.nobits:
        return S_ZEROFILL
    if section.flags & 0x4:                          # SHF_EXECINSTR
        return S_REGULAR | S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS
    return S_REGULAR


def _protection(segment: str) -> int:
    if segment == "__TEXT":
        return VM_PROT_READ | VM_PROT_EXECUTE
    return VM_PROT_READ | VM_PROT_WRITE


def _thread_command(machine: int, entry: int) -> bytes:
    """`LC_UNIXTHREAD`: a whole register state, with the entry in the PC.

    EVERY OTHER REGISTER IS ZERO, which is what a fresh thread wants: the
    kernel sets the stack pointer itself when it builds the argument area,
    and a program that started with someone else's `rsp` would write over it.
    """
    try:
        flavour, words, pc_at = _THREAD_STATE[machine]
    except KeyError:
        raise MachoWriteError(
            f"no thread state is known for Mach-O cputype {machine:#x}",
            ) from None
    state = bytearray(4 * words)
    struct.pack_into("<Q", state, pc_at, entry)
    return struct.pack("<IIII", LC_UNIXTHREAD, 16 + len(state),
                       flavour, words) + bytes(state)


def executable(image, *, base: int, page: int, sign: bool = False,
               identifier: str = "uasm") -> bytes:
    """Wrap a laid-out image in a Mach-O executable.

    `image` is `staticlink.Image`. The layout invariant `vaddr = base + file
    offset` is what makes every segment's `fileoff` fall out of its address,
    which is the same trick the other two writers use and for the same
    reason: a loader maps whole pages, and a file whose offsets and addresses
    agree modulo the page size can be mapped without a second table.
    """
    machine = image.machine
    if machine not in _THREAD_STATE:
        raise MachoWriteError(f"no Mach-O support for cputype {machine:#x}")

    # ── the sections, grouped into the segments they belong to ──────────────
    grouped: dict[str, list] = {}
    for sec in image.sections:
        if not sec.pieces:
            continue
        segment, name = _bucket_of(sec)
        grouped.setdefault(segment, []).append((name, sec))
    if not grouped:
        raise MachoWriteError("the image has no sections to write")
    order = [s for s in ("__TEXT", "__DATA") if s in grouped]
    if set(order) != set(grouped):
        raise MachoWriteError(
            f"sections landed in a segment this writes nothing for: "
            f"{sorted(set(grouped) - set(order))}")
    if order[0] != "__TEXT":
        raise MachoWriteError("__TEXT must be the first segment")

    # ── each segment's extent ───────────────────────────────────────────────
    #
    # `__TEXT` REACHES BACK TO FILE OFFSET ZERO, because the Mach header and
    # the load commands are inside it: the kernel maps the segment in order
    # to read them, so a `__TEXT` that began after them would describe an
    # image whose own header is not mapped.
    extent: dict[str, tuple[int, int, int]] = {}
    for segment in order:
        members = [sec for _n, sec in grouped[segment]]
        start = min(sec.addr for sec in members)
        if segment == "__TEXT":
            start = base
        vm_end = max(sec.addr + sec.size for sec in members)
        file_end = max((sec.addr + sec.size for sec in members
                        if not sec.nobits), default=start)
        extent[segment] = (start, _round_up(vm_end - start, page),
                           _round_up(file_end - start, page))

    nsections = sum(len(v) for v in grouped.values())
    thread = _thread_command(machine, image.entry)

    # ── __LINKEDIT, which exists only to hold the signature ─────────────────
    #
    # A STATIC IMAGE HAS NOTHING ELSE TO PUT THERE. No symbol table is
    # written: nothing reads one out of an image with no dynamic linking, and
    # an empty `LC_SYMTAB` says less than no `LC_SYMTAB` at all.
    body_end = max(start + file_size for start, _vm, file_size
                   in extent.values())
    linkedit_at = _round_up(body_end, page)
    signature_size = _signature_size(identifier, linkedit_at - base,
                                     page) if sign else 0
    text_base, _text_vm, text_file = extent["__TEXT"]
    commands: list[bytes] = []
    ncommands = 1 + len(order) + 1 + (2 if sign else 0)   # +__PAGEZERO, thread

    commands.append(struct.pack(
        _SEGMENT, LC_SEGMENT_64, 72, _fixed("__PAGEZERO"),
        0, PAGEZERO_SIZE, 0, 0, 0, 0, 0, 0))

    for segment in order:
        start, vmsize, filesize = extent[segment]
        members = grouped[segment]
        commands.append(struct.pack(
            _SEGMENT, LC_SEGMENT_64, 72 + 80 * len(members), _fixed(segment),
            start, vmsize, start - base, filesize,
            _protection(segment), _protection(segment), len(members), 0))
        for name, sec in members:
            commands[-1] += _fixed(name) + _fixed(segment) + struct.pack(
                "<QQIIIIIIII", sec.addr, sec.size,
                0 if sec.nobits else sec.addr - base,
                max(0, (sec.align - 1).bit_length()),
                0, 0, _section_flags(sec), 0, 0, 0)

    if sign:
        commands.append(struct.pack(
            _SEGMENT, LC_SEGMENT_64, 72, _fixed("__LINKEDIT"),
            base + _round_up(linkedit_at - base, page),
            _round_up(signature_size, page),
            linkedit_at - base, signature_size,
            VM_PROT_READ, VM_PROT_READ, 0, 0))
    commands.append(thread)
    if sign:
        commands.append(struct.pack("<IIII", LC_CODE_SIGNATURE, 16,
                                    linkedit_at - base, signature_size))

    head = struct.pack(_HEADER, MH_MAGIC_64, machine, _SUBTYPE_OF[machine],
                       MH_EXECUTE, len(commands),
                       sum(len(c) for c in commands), MH_NOUNDEFS, 0)
    blob = head + b"".join(commands)
    first = min(sec.addr - base for sec in image.sections
                if sec.pieces and not sec.nobits)
    if len(blob) > first:
        raise MachoWriteError(
            "the Mach-O headers do not fit in front of the first section",
            )

    out = bytearray(image.data)
    if len(out) < len(blob):
        out.extend(b"\0" * (len(blob) - len(out)))
    out[0:len(blob)] = blob
    if len(out) < linkedit_at - base:
        out.extend(b"\0" * ((linkedit_at - base) - len(out)))
    if sign:
        out.extend(_signature(bytes(out), identifier, page,
                              exec_base=text_base - base,
                              exec_limit=text_file))
    del ncommands, nsections
    return bytes(out)


__all__ = ["DEFAULT_BASE", "MachoWriteError", "PAGE_OF",
           "arm64_static_will_not_load", "executable", "header_space"]
