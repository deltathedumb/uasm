"""The platform floor as machine code, so a program needs no libc.

`objects/floor.py` says what the three functions mean and gives a C
implementation of them over `fwrite`, `exit` and `malloc`. That C is why every
native build links against libc, and linking against libc is why every native
build needs a C toolchain to find it. This module is the same three contracts
written as SYSCALLS, plus the `_start` that a static image has instead of
crt1.o -- assembled here, as literal bytes, and handed to the linker as an
ordinary object.

WHY THE BYTES ARE WRITTEN OUT BY HAND rather than emitted by the backend's own
encoder. The encoder compiles IR, and the IR has no way to SAY `syscall`: it
is a call to a named function everywhere else, which is exactly the thing
being bottomed out here. A floor written in the language above it would be
circular. So these are the only hand-encoded instructions in the compiler, and
each one is commented with its assembly.

WHAT `_start` MUST DO, which is more than it looks:

  * THE STACK IS NOT 16-BYTE ALIGNED ON ENTRY. The kernel puts `argc` at
    `%rsp` and the ABI requires alignment at a CALL, so a `_start` that calls
    anything without adjusting will fault the first time the callee touches a
    SIMD register with an aligned move -- which is not the first instruction,
    so it looks like a bug somewhere else entirely.
  * THERE IS NO RETURN ADDRESS. `_start` is jumped to, not called; returning
    from it returns to nowhere. It must exit.
  * NOTHING HAS INITIALISED ANYTHING. No `__libc_start_main`, no atexit, no
    stdio. That is the point.

THE HEAP IS ONE `mmap` PER CALL, and the contract in `objects/floor.py` is
written so that it can be: a caller chains regions rather than assuming one
growing block, precisely so the floor does not have to be `sbrk`. A page
allocator here would be a second memory manager under the one in
`runtime/arena.py`, which already does the chaining.
"""
from __future__ import annotations

import struct

from ..backend.objfile import (
    EM_AARCH64, EM_X86_64, SHF_ALLOC, SHF_EXECINSTR, STB_GLOBAL, STT_FUNC,
    ElfObject, Relocation, Symbol,
)

#: Linux syscall numbers. Per architecture, because they are not shared: the
#: x86-64 table is the historical one and AArch64 uses the "generic" table
#: every newer port takes, so `write` is 1 on one and 64 on the other.
_LINUX_X86_64 = {"write": 1, "mmap": 9, "exit_group": 231}
_LINUX_AARCH64 = {"write": 64, "mmap": 222, "exit_group": 94}

#: `mmap` arguments for an anonymous private mapping. Same numbers on both
#: architectures; they are Linux's, not the machine's.
PROT_READ, PROT_WRITE = 0x1, 0x2
MAP_PRIVATE, MAP_ANONYMOUS = 0x02, 0x20


def _x86_64_object(entry: str) -> ElfObject:
    """The floor and `_start` for Linux x86-64.

    THE SYSTEM CALL CONVENTION IS NOT THE FUNCTION ONE, which is the single
    thing to keep straight in every routine below: arguments go in
    rdi/rsi/rdx/r10/r8/r9 rather than rdi/rsi/rdx/rcx/r8/r9, the number goes
    in rax, and `syscall` clobbers rcx and r11. So a C-ABI argument in rcx has
    to be moved to r10 before the call, and nothing may expect rcx to survive.
    """
    sc = _LINUX_X86_64
    text = bytearray()
    symbols: list[Symbol] = []
    relocs: list[Relocation] = []

    def begin(name: str) -> int:
        return len(text)

    def end(name: str, at: int) -> None:
        symbols.append(Symbol(name=name, section=".text", value=at,
                              size=len(text) - at, binding=STB_GLOBAL,
                              kind=STT_FUNC))

    # ── _start ──────────────────────────────────────────────────────────────
    at = begin("_start")
    #   xor  %rbp, %rbp            ; the ABI's "outermost frame" marker, and
    text += b"\x48\x31\xED"       # what a debugger walks back to.
    #   and  $-16, %rsp            ; the kernel leaves argc at %rsp, so the
    text += b"\x48\x83\xE4\xF0"   # stack is 8 short of aligned at a call.
    #   call uasm_main
    text += b"\xE8" + b"\0\0\0\0"
    relocs.append(Relocation(offset=len(text) - 4, symbol=entry,
                             kind=4, addend=-4))          # R_X86_64_PLT32
    #   mov  %rax, %rdi            ; the program's answer becomes the status
    text += b"\x48\x89\xC7"
    #   mov  $exit_group, %eax
    text += b"\xB8" + struct.pack("<I", sc["exit_group"])
    #   syscall
    text += b"\x0F\x05"
    #   ud2                        ; exit_group does not return; if it somehow
    text += b"\x0F\x0B"           # does, fault here rather than run on.
    end("_start", at)

    # ── plat_write(fd, buf, n) -> i64 ───────────────────────────────────────
    #
    # THE ARGUMENTS ARE ALREADY WHERE THE SYSCALL WANTS THEM: the C ABI puts
    # fd, buf and n in rdi, rsi and rdx, and `write` takes them in the same
    # three. Only the number has to be loaded.
    at = begin("plat_write")
    text += b"\xB8" + struct.pack("<I", sc["write"])       # mov $write,%eax
    text += b"\x0F\x05"                                    # syscall
    # A NEGATIVE RETURN IS AN ERRNO, NOT A COUNT. The kernel answers -EBADF
    # rather than setting a variable, and the contract here is "bytes written,
    # or -1" -- so anything in the error range becomes -1 rather than being
    # handed back as a plausible byte count.
    text += b"\x48\x83\xF8\x00"                            # cmp $0,%rax
    text += b"\x7D\x07"                                    # jge +7
    text += b"\x48\xC7\xC0" + struct.pack("<i", -1)        # mov $-1,%rax
    text += b"\xC3"                                        # ret
    end("plat_write", at)

    # ── plat_exit(code) ─────────────────────────────────────────────────────
    at = begin("plat_exit")
    text += b"\xB8" + struct.pack("<I", sc["exit_group"])  # mov $exit,%eax
    text += b"\x0F\x05"                                    # syscall
    text += b"\x0F\x0B"                                    # ud2
    end("plat_exit", at)

    # ── plat_heap(n) -> ptr ─────────────────────────────────────────────────
    #
    # `mmap(NULL, n, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)`.
    # The kernel rounds the length up to a page, so a caller asking for eight
    # bytes gets a page and the arena above chains them.
    at = begin("plat_heap")
    text += b"\x48\x85\xFF"                                # test %rdi,%rdi
    text += b"\x7F\x05"                                    # jg +5
    text += b"\x48\x31\xC0"                                # xor %rax,%rax
    text += b"\xC3"                                        # ret
    text += b"\x48\x89\xFE"                                # mov %rdi,%rsi (len)
    text += b"\x48\x31\xFF"                                # xor %rdi,%rdi (addr)
    text += b"\xBA" + struct.pack("<I", PROT_READ | PROT_WRITE)   # mov ,%edx
    text += b"\x41\xBA" + struct.pack("<I", MAP_PRIVATE | MAP_ANONYMOUS)
    text += b"\x49\xC7\xC0" + struct.pack("<i", -1)        # mov $-1,%r8 (fd)
    text += b"\x4D\x31\xC9"                                # xor %r9,%r9 (off)
    text += b"\xB8" + struct.pack("<I", sc["mmap"])        # mov $mmap,%eax
    text += b"\x0F\x05"                                    # syscall
    # THE ERROR RANGE IS THE LAST PAGE, not a negative number: `mmap` answers
    # an address, and -4095..-1 is how the kernel spells a failure in a value
    # that is otherwise unsigned.
    text += b"\x48\x3D" + struct.pack("<i", -4095)         # cmp $-4095,%rax
    text += b"\x72\x03"                                    # jb +3
    text += b"\x48\x31\xC0"                                # xor %rax,%rax
    text += b"\xC3"                                        # ret
    end("plat_heap", at)

    obj = ElfObject(EM_X86_64)
    obj.section(".text", bytes(text), flags=SHF_ALLOC | SHF_EXECINSTR,
                align=16)
    for rel in relocs:
        obj.relocate(".text", rel)
    for sym in symbols:
        obj.symbol(sym)
    obj.symbol(Symbol(name=entry, section="", binding=STB_GLOBAL))
    return obj


def _aarch64_object(entry: str) -> ElfObject:
    """The same three functions and `_start` for Linux AArch64.

    THE STACK IS ALREADY 16-ALIGNED ON ENTRY here, which x86-64's is not: the
    AArch64 ABI requires `sp` to be 16-aligned at all times, and the kernel
    honours it. So `_start` has nothing to adjust.
    """
    sc = _LINUX_AARCH64
    words: list[int] = []
    symbols: list[Symbol] = []
    relocs: list[Relocation] = []

    def movz(rd: int, imm: int) -> int:
        """`mov xD, #imm` for a 16-bit immediate -- MOVZ, shift 0."""
        return 0xD2800000 | ((imm & 0xFFFF) << 5) | rd

    def here() -> int:
        return len(words) * 4

    def end(name: str, at: int) -> None:
        symbols.append(Symbol(name=name, section=".text", value=at,
                              size=here() - at, binding=STB_GLOBAL,
                              kind=STT_FUNC))

    SVC0 = 0xD4000001          # svc #0
    RET = 0xD65F03C0           # ret
    BRK0 = 0xD4200000          # brk #0

    # ── _start ──────────────────────────────────────────────────────────────
    at = here()
    # NOTHING TO ALIGN. The AArch64 ABI keeps `sp` 16-aligned at all times and
    # the kernel honours it on entry, so unlike x86-64 there is no adjustment
    # to make before the first call.
    #
    # `bl uasm_main` -- the offset is a relocation, so the word is the opcode
    # with a zero displacement.
    relocs.append(Relocation(offset=here(), symbol=entry, kind=283, addend=0))
    words.append(0x94000000)                       # bl 0
    words.append(0xAA0003E0)                       # mov x0, x0 (status)
    words.append(movz(8, sc["exit_group"]))        # mov x8, #exit_group
    words.append(SVC0)
    words.append(BRK0)
    end("_start", at)

    # ── plat_write(fd, buf, n) -> i64 ───────────────────────────────────────
    at = here()
    words.append(movz(8, sc["write"]))
    words.append(SVC0)
    # `cmp x0, #0` then `csel`-free clamp: b.ge over a `mov x0, #-1`.
    words.append(0xF100001F)                       # cmp x0, #0
    words.append(0x540000AA)                       # b.ge +8 (two words on)
    words.append(0x92800000)                       # movn x0, #0  -> -1
    words.append(RET)
    end("plat_write", at)

    # ── plat_exit(code) ─────────────────────────────────────────────────────
    at = here()
    words.append(movz(8, sc["exit_group"]))
    words.append(SVC0)
    words.append(BRK0)
    end("plat_exit", at)

    # ── plat_heap(n) -> ptr ─────────────────────────────────────────────────
    at = here()
    words.append(0xF100001F)                       # cmp x0, #0
    words.append(0x5400006C)                       # b.gt +12 (three words on)
    words.append(0xAA1F03E0)                       # mov x0, xzr
    words.append(RET)
    words.append(0xAA0003E1)                       # mov x1, x0   (length)
    words.append(0xAA1F03E0)                       # mov x0, xzr  (addr)
    words.append(movz(2, PROT_READ | PROT_WRITE))
    words.append(movz(3, MAP_PRIVATE | MAP_ANONYMOUS))
    words.append(0x92800004)                       # movn x4, #0  -> -1 (fd)
    words.append(0xAA1F03E5)                       # mov x5, xzr  (offset)
    words.append(movz(8, sc["mmap"]))
    words.append(SVC0)
    # THE ERROR RANGE IS THE LAST PAGE, as on x86-64. `cmn x0, #4095` sets the
    # flags that `cmp x0, #-4095` would, which is how a comparison against a
    # negative immediate is spelled when the field is unsigned.
    words.append(0xB103FC1F)                       # cmn x0, #4095
    words.append(0x54000043)                       # b.lo +8
    words.append(0xAA1F03E0)                       # mov x0, xzr
    words.append(RET)
    end("plat_heap", at)

    text = b"".join(struct.pack("<I", w) for w in words)
    obj = ElfObject(EM_AARCH64)
    obj.section(".text", text, flags=SHF_ALLOC | SHF_EXECINSTR, align=4)
    for rel in relocs:
        obj.relocate(".text", rel)
    for sym in symbols:
        obj.symbol(sym)
    obj.symbol(Symbol(name=entry, section="", binding=STB_GLOBAL))
    return obj


#: Which builder serves which ELF machine.
_BUILDERS = {EM_X86_64: _x86_64_object, EM_AARCH64: _aarch64_object}

#: The names this object defines, for the test that asserts the floor is still
#: three functions and for a caller deciding whether it is needed at all.
PROVIDES = ("_start", "plat_write", "plat_exit", "plat_heap")


def floor_object(machine: int, *, entry: str = "uasm_main") -> bytes:
    """The floor, `_start` and nothing else, as a relocatable ELF object."""
    build = _BUILDERS.get(machine)
    if build is None:
        raise KeyError(
            f"no freestanding floor for ELF machine {machine}; "
            f"x86-64 (62) and AArch64 (183) have one")
    return build(entry).to_bytes()


__all__ = ["PROVIDES", "floor_object"]
