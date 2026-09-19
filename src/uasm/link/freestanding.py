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

    # ── putchar(c) -> int ───────────────────────────────────────────────────
    #
    # NOT PART OF THE FLOOR, and here for a reason worth stating. The Python
    # frontend emits a CALL to `putchar` -- it is how `print` writes the
    # space between arguments and the newline at the end (`lower.py`) -- so a
    # linked program names it whether or not a runtime was asked for. libc
    # supplies it in a hosted build; uasm's own `<stdio.h>` declares it
    # `static`, so the runtime object compiled from that header never exports
    # one. `link/baremetal.py` reached the same conclusion and defines its
    # own; this is that, as machine code.
    at = begin("putchar")
    text += b"\x48\x83\xEC\x10"                            # sub $16,%rsp
    text += b"\x40\x88\x3C\x24"                            # mov %dil,(%rsp)
    text += b"\xB8" + struct.pack("<I", sc["write"])       # mov $write,%eax
    text += b"\xBF\x01\x00\x00\x00"                        # mov $1,%edi
    text += b"\x48\x89\xE6"                                # mov %rsp,%rsi
    text += b"\xBA\x01\x00\x00\x00"                        # mov $1,%edx
    text += b"\x0F\x05"                                    # syscall
    # THE CHARACTER IS THE ANSWER, read back out of the buffer rather than
    # kept in a register: `%edi` held it and the fd overwrote it, and every
    # register that survives a syscall is one more thing to have got right.
    text += b"\x0F\xB6\x04\x24"                            # movzbl (%rsp),%eax
    text += b"\x48\x83\xC4\x10"                            # add $16,%rsp
    text += b"\xC3"                                        # ret
    end("putchar", at)

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
    words.append(0x5400004A)                       # b.ge +8 (two words on)
    words.append(0x92800000)                       # movn x0, #0  -> -1
    words.append(RET)
    end("plat_write", at)

    # ── plat_exit(code) ─────────────────────────────────────────────────────
    at = here()
    words.append(movz(8, sc["exit_group"]))
    words.append(SVC0)
    words.append(BRK0)
    end("plat_exit", at)

    # ── putchar(c) -> int ───────────────────────────────────────────────────
    #
    # See the x86-64 twin for why this is here at all.
    at = here()
    words.append(0xD10043FF)                       # sub sp, sp, #16
    words.append(0x390003E0)                       # strb w0, [sp]
    words.append(0xAA0003E9)                       # mov x9, x0   (keep it)
    words.append(0xD2800020)                       # mov x0, #1   (fd)
    words.append(0x910003E1)                       # mov x1, sp   (buf)
    words.append(0xD2800022)                       # mov x2, #1   (len)
    words.append(movz(8, sc["write"]))
    words.append(SVC0)
    words.append(0x12001D20)                       # and w0, w9, #0xff
    words.append(0x910043FF)                       # add sp, sp, #16
    words.append(RET)
    end("putchar", at)

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
    words.append(0xB13FFC1F)                       # cmn x0, #4095
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


# ── Windows ─────────────────────────────────────────────────────────────────

#: `GetStdHandle`'s arguments, and the two `VirtualAlloc` flags. Windows
#: constants, not machine ones.
STD_OUTPUT_HANDLE = -11
STD_ERROR_HANDLE = -12
MEM_COMMIT_RESERVE = 0x3000
PAGE_READWRITE = 0x04


def _windows_x86_64_object(entry: str):
    """The floor and the entry point for Windows x86-64.

    THREE THINGS ARE DIFFERENT FROM LINUX, and all three are why this is a
    separate function rather than a table of syscall numbers:

      * THERE IS NO SYSCALL ABI A PROGRAM MAY USE. The numbers in `ntdll`
        change between builds of the operating system and Microsoft says so,
        so the kernel is reached through `kernel32.dll` -- which means an
        IMPORT TABLE, built by `link/pewrite.py`, and calls that go
        indirectly through it. `__imp_WriteFile` is the ADDRESS OF THE SLOT
        the loader writes the function's address into, which is MSVC's
        convention and what makes `call *__imp_WriteFile(%rip)` work.
      * THE CALLING CONVENTION IS MICROSOFT'S: rcx, rdx, r8, r9, then the
        stack -- and the caller reserves THIRTY-TWO BYTES OF SHADOW SPACE
        above the return address for the callee to spill those four into,
        whether or not it does. A call without it corrupts the caller's own
        frame.
      * EVERY VOLATILE REGISTER IS REALLY VOLATILE. rax, rcx, rdx and r8-r11
        do not survive a call, so `plat_write` keeps its arguments on the
        stack across `GetStdHandle` rather than in r10 and r11, which was
        the first version of this and would have failed only sometimes.

    NOTHING HERE HAS BEEN EXECUTED. There is no Windows and no emulator on
    the machine this was written on; the encodings are checked by
    disassembling the linked image, and the import table by reading it back.
    That is a weaker claim than the Linux floor's, which is run by
    `tests/uasm/integration/test_builtin_linker.py`, and it is the honest one.
    """
    from ..backend.objfile import (
        CoffObject, CoffRelocation, CoffSymbol, IMAGE_FILE_MACHINE_AMD64,
        IMAGE_SCN_CNT_CODE, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ,
    )

    IMAGE_REL_AMD64_REL32 = 0x0004
    text = bytearray()
    symbols: list = []
    relocs: list = []

    def call(name: str) -> None:
        """`call *__imp_<name>(%rip)` -- indirectly, through the IAT."""
        text.extend(b"\xFF\x15\x00\x00\x00\x00")
        relocs.append(CoffRelocation(len(text) - 4, "__imp_" + name,
                                     IMAGE_REL_AMD64_REL32))

    def end(name: str, at: int) -> None:
        symbols.append(CoffSymbol(name=name, section=".text", value=at,
                                  size=len(text) - at, binding=1, kind=2))

    # ── _start ──────────────────────────────────────────────────────────────
    at = len(text)
    text += b"\x48\x83\xE4\xF0"            # and $-16,%rsp
    text += b"\x48\x83\xEC\x20"            # sub $32,%rsp   (shadow space)
    text += b"\xE8\x00\x00\x00\x00"        # call uasm_main
    relocs.append(CoffRelocation(len(text) - 4, entry, IMAGE_REL_AMD64_REL32))
    text += b"\x89\xC1"                     # mov %eax,%ecx  (exit status)
    call("ExitProcess")
    text += b"\xCC"                          # int3 -- ExitProcess is noreturn
    end("_start", at)

    # ── plat_write(fd, buf, n) -> i64 ───────────────────────────────────────
    at = len(text)
    text += b"\x48\x83\xEC\x48"            # sub $72,%rsp
    text += b"\x48\x89\x54\x24\x30"        # mov %rdx,0x30(%rsp)  (buf)
    text += b"\x4C\x89\x44\x24\x38"        # mov %r8,0x38(%rsp)   (n)
    text += b"\x48\x83\xF9\x02"            # cmp $2,%rcx
    text += b"\xB9" + struct.pack("<i", STD_OUTPUT_HANDLE)
    text += b"\xB8" + struct.pack("<i", STD_ERROR_HANDLE)
    text += b"\x0F\x44\xC8"                # cmove %eax,%ecx
    call("GetStdHandle")
    text += b"\x48\x89\xC1"                # mov %rax,%rcx  (handle)
    text += b"\x48\x8B\x54\x24\x30"        # mov 0x30(%rsp),%rdx
    text += b"\x4C\x8B\x44\x24\x38"        # mov 0x38(%rsp),%r8
    text += b"\x4C\x8D\x4C\x24\x40"        # lea 0x40(%rsp),%r9  (written)
    text += b"\x48\xC7\x44\x24\x20\x00\x00\x00\x00"   # movq $0,0x20(%rsp)
    call("WriteFile")
    text += b"\x85\xC0"                     # test %eax,%eax
    text += b"\x74\x0A"                     # je +10  (the failure tail)
    text += b"\x48\x8B\x44\x24\x38"        # mov 0x38(%rsp),%rax  (n)
    text += b"\x48\x83\xC4\x48"            # add $72,%rsp
    text += b"\xC3"                          # ret
    text += b"\x48\xC7\xC0\xFF\xFF\xFF\xFF"  # mov $-1,%rax
    text += b"\x48\x83\xC4\x48"            # add $72,%rsp
    text += b"\xC3"                          # ret
    end("plat_write", at)

    # ── plat_exit(code) ─────────────────────────────────────────────────────
    at = len(text)
    text += b"\x48\x83\xEC\x28"            # sub $40,%rsp
    call("ExitProcess")
    text += b"\xCC"                          # int3
    end("plat_exit", at)

    # ── putchar(c) -> int ───────────────────────────────────────────────────
    at = len(text)
    text += b"\x48\x83\xEC\x48"            # sub $72,%rsp
    text += b"\x89\x4C\x24\x30"            # mov %ecx,0x30(%rsp)  (keep c)
    text += b"\x88\x4C\x24\x38"            # mov %cl,0x38(%rsp)   (the byte)
    text += b"\xB9" + struct.pack("<i", STD_OUTPUT_HANDLE)
    call("GetStdHandle")
    text += b"\x48\x89\xC1"                # mov %rax,%rcx
    text += b"\x48\x8D\x54\x24\x38"        # lea 0x38(%rsp),%rdx
    text += b"\x41\xB8\x01\x00\x00\x00"    # mov $1,%r8d
    text += b"\x4C\x8D\x4C\x24\x40"        # lea 0x40(%rsp),%r9
    text += b"\x48\xC7\x44\x24\x20\x00\x00\x00\x00"
    call("WriteFile")
    text += b"\x8B\x44\x24\x30"            # mov 0x30(%rsp),%eax
    text += b"\x48\x83\xC4\x48"            # add $72,%rsp
    text += b"\xC3"                          # ret
    end("putchar", at)

    # ── plat_heap(n) -> ptr ─────────────────────────────────────────────────
    #
    # `VirtualAlloc(NULL, n, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE)`, which
    # is Windows' answer to the anonymous `mmap` the Linux floor makes. It
    # rounds up to a page too, so the arena above chains them the same way.
    at = len(text)
    text += b"\x48\x83\xEC\x28"            # sub $40,%rsp
    text += b"\x48\x85\xC9"                # test %rcx,%rcx
    text += b"\x7E\x1C"                     # jle +28  (answer null)
    text += b"\x48\x89\xCA"                # mov %rcx,%rdx   (size)
    text += b"\x31\xC9"                     # xor %ecx,%ecx   (address)
    text += b"\x41\xB8" + struct.pack("<I", MEM_COMMIT_RESERVE)
    text += b"\x41\xB9" + struct.pack("<I", PAGE_READWRITE)
    call("VirtualAlloc")
    text += b"\x48\x83\xC4\x28"            # add $40,%rsp
    text += b"\xC3"                          # ret
    text += b"\x48\x31\xC0"                # xor %rax,%rax
    text += b"\x48\x83\xC4\x28"            # add $40,%rsp
    text += b"\xC3"                          # ret
    end("plat_heap", at)

    obj = CoffObject(IMAGE_FILE_MACHINE_AMD64)
    obj.section(".text", bytes(text), align=16,
                characteristics=IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE
                                | IMAGE_SCN_MEM_READ)
    for sym in symbols:
        obj.symbol(sym)
    for name in (entry, "__imp_ExitProcess", "__imp_WriteFile",
                 "__imp_GetStdHandle", "__imp_VirtualAlloc"):
        obj.symbol(CoffSymbol(name=name, section="", binding=1))
    for rel in relocs:
        obj.relocate(".text", rel)
    return obj


# ── macOS ───────────────────────────────────────────────────────────────────

#: Darwin syscall numbers, from `bsd/kern/syscalls.master`. THE SAME ON BOTH
#: MACHINES, unlike Linux's: Darwin has one BSD table and both architectures
#: use it.
_DARWIN = {"write": 4, "exit": 1, "mmap": 197}

#: THE CLASS GOES IN THE NUMBER ON INTEL AND NOWHERE ON APPLE SILICON. A
#: Darwin syscall number is (class << 24) | number, and the BSD class is 2 --
#: so `write` is 0x2000004 in `rax` on x86-64. On arm64 the number goes in
#: `x16` as itself, and the trap is `svc #0x80`.
SYSCALL_CLASS_UNIX = 2 << 24

#: `mmap` arguments for an anonymous private mapping. NOT LINUX'S NUMBERS:
#: `MAP_ANON` is 0x1000 on Darwin where it is 0x20 on Linux, and a floor that
#: reused the Linux constant would ask for a file mapping of file -1.
DARWIN_MAP_PRIVATE, DARWIN_MAP_ANON = 0x0002, 0x1000


def _macho_symbols(obj, text: bytes, defined, needed, machine) -> None:
    """The shared tail of both macOS floors: one section and its symbols.

    A NAME NEEDS ITS LEADING UNDERSCORE on this platform -- the backend
    applies it to everything it emits, so `uasm_main` is `_uasm_main` in the
    object -- and the entry point is spelled `_start` here for the same
    reason the other two floors spell it that way: it is the name the linker
    is told to start at, and one name across three containers is worth more
    than each platform's own.
    """
    from ..backend.objfile.macho import (
        S_ATTR_PURE_INSTRUCTIONS, S_ATTR_SOME_INSTRUCTIONS, S_REGULAR,
        Symbol,
    )
    del machine
    obj.section(".text", text, align=16,
                flags=S_REGULAR | S_ATTR_PURE_INSTRUCTIONS
                | S_ATTR_SOME_INSTRUCTIONS)
    for sym in defined:
        obj.symbol(sym)
    for name in needed:
        obj.symbol(Symbol(name=name, section="", binding=1))


def _macos_x86_64_object(entry: str):
    """The floor and the entry point for macOS x86-64.

    THE TRAP IS THE SAME INSTRUCTION AS LINUX'S and almost nothing else is.
    The arguments go in the same six registers, which is why `plat_write`
    below is three instructions; the NUMBER carries a class in its top byte,
    the error is signalled by the CARRY FLAG rather than by a negative
    return, and `mmap`'s flags are Darwin's.

    NOTHING HERE HAS BEEN EXECUTED. There is no macOS on the machine this was
    written on and no emulator; the encodings are checked by disassembling
    the linked image and the structures by reading them back with
    `llvm-objdump --macho`. That is a weaker claim than the Linux floor's,
    which `tests/uasm/integration/test_builtin_linker.py` runs, and it is the
    honest one.
    """
    from ..backend.objfile.macho import (
        CPU_SUBTYPE_X86_64_ALL, CPU_TYPE_X86_64, MachoObject, Relocation,
        Symbol,
    )

    X86_64_RELOC_BRANCH = 2
    sc = {k: SYSCALL_CLASS_UNIX | v for k, v in _DARWIN.items()}
    text = bytearray()
    defined: list = []
    relocs: list = []

    def end(name: str, at: int) -> None:
        # THE PLATFORM'S UNDERSCORE. Every C name on macOS wears one, so the
        # backend emits `_plat_write` and the floor has to DEFINE
        # `_plat_write` -- a floor spelling it without would link against
        # nothing and leave every call undefined. `_start` already begins
        # with one and keeps its single underscore: it is not a C name, it
        # is the string the linker is told to start at, and it is the same
        # string in all three containers.
        defined.append(Symbol(name=name if name == "_start" else "_" + name,
                              section=".text", value=at,
                              size=len(text) - at, binding=1, kind=2))

    # ── _start ──────────────────────────────────────────────────────────────
    at = len(text)
    text += b"\x48\x31\xED"                # xor %rbp,%rbp  (outermost frame)
    text += b"\x48\x83\xE4\xF0"            # and $-16,%rsp
    text += b"\xE8\x00\x00\x00\x00"        # call _uasm_main
    relocs.append(Relocation(len(text) - 4, "_" + entry, X86_64_RELOC_BRANCH,
                             addend=0, pcrel=True, length=2))
    text += b"\x48\x89\xC7"                # mov %rax,%rdi  (exit status)
    text += b"\xB8" + struct.pack("<I", sc["exit"])
    text += b"\x0F\x05"                     # syscall
    text += b"\x0F\x0B"                     # ud2 -- exit does not return
    end("_start", at)

    # ── plat_write(fd, buf, n) -> i64 ───────────────────────────────────────
    #
    # THE ARGUMENTS ARE ALREADY WHERE THE SYSCALL WANTS THEM, as on Linux.
    # What differs is the failure: Darwin sets the CARRY FLAG and puts the
    # errno in rax, where Linux returns a negative number -- so a test of the
    # sign would read `EBADF` as nine bytes written.
    at = len(text)
    text += b"\xB8" + struct.pack("<I", sc["write"])
    text += b"\x0F\x05"                     # syscall
    text += b"\x73\x07"                     # jnc +7
    text += b"\x48\xC7\xC0\xFF\xFF\xFF\xFF"   # mov $-1,%rax
    text += b"\xC3"                          # ret
    end("plat_write", at)

    # ── plat_exit(code) ─────────────────────────────────────────────────────
    at = len(text)
    text += b"\xB8" + struct.pack("<I", sc["exit"])
    text += b"\x0F\x05"                     # syscall
    text += b"\x0F\x0B"                     # ud2
    end("plat_exit", at)

    # ── putchar(c) -> int ───────────────────────────────────────────────────
    #
    # SEE THE LINUX FLOOR for why a platform floor defines this at all: the
    # Python frontend emits a call to it, and uasm's own `<stdio.h>` declares
    # it `static`, so nothing else in a freestanding link exports one.
    at = len(text)
    text += b"\x48\x83\xEC\x10"            # sub $16,%rsp
    text += b"\x40\x88\x3C\x24"            # mov %dil,(%rsp)
    text += b"\xB8" + struct.pack("<I", sc["write"])
    text += b"\xBF\x01\x00\x00\x00"        # mov $1,%edi
    text += b"\x48\x89\xE6"                # mov %rsp,%rsi
    text += b"\xBA\x01\x00\x00\x00"        # mov $1,%edx
    text += b"\x0F\x05"                     # syscall
    text += b"\x0F\xB6\x04\x24"            # movzbl (%rsp),%eax
    text += b"\x48\x83\xC4\x10"            # add $16,%rsp
    text += b"\xC3"                          # ret
    end("putchar", at)

    # ── plat_heap(n) -> ptr ─────────────────────────────────────────────────
    at = len(text)
    text += b"\x48\x85\xFF"                # test %rdi,%rdi
    text += b"\x7F\x05"                     # jg +5
    text += b"\x48\x31\xC0"                # xor %rax,%rax
    text += b"\xC3"                          # ret
    text += b"\x48\x89\xFE"                # mov %rdi,%rsi  (len)
    text += b"\x48\x31\xFF"                # xor %rdi,%rdi  (addr)
    text += b"\xBA" + struct.pack("<I", PROT_READ | PROT_WRITE)
    text += b"\x41\xBA" + struct.pack("<I", DARWIN_MAP_PRIVATE
                                      | DARWIN_MAP_ANON)
    text += b"\x49\xC7\xC0\xFF\xFF\xFF\xFF"   # mov $-1,%r8   (fd)
    text += b"\x4D\x31\xC9"                # xor %r9,%r9   (offset)
    text += b"\xB8" + struct.pack("<I", sc["mmap"])
    text += b"\x0F\x05"                     # syscall
    text += b"\x73\x03"                     # jnc +3
    text += b"\x48\x31\xC0"                # xor %rax,%rax
    text += b"\xC3"                          # ret
    end("plat_heap", at)

    obj = MachoObject(CPU_TYPE_X86_64, CPU_SUBTYPE_X86_64_ALL)
    _macho_symbols(obj, bytes(text), defined, ("_" + entry,), CPU_TYPE_X86_64)
    for rel in relocs:
        obj.relocate(".text", rel)
    return obj


def _macos_aarch64_object(entry: str):
    """The floor and the entry point for macOS on Apple Silicon.

    THREE THINGS DIFFER FROM LINUX AARCH64 and all three are the platform's,
    not the machine's: the syscall number goes in `x16` rather than `x8`, the
    trap is `svc #0x80` rather than `svc #0`, and a failure is the CARRY FLAG
    rather than a negative return. The last is the one that fails quietly --
    a floor that tested the sign would read `EBADF` as nine bytes written.

    NOTHING HERE HAS BEEN EXECUTED. See the x86-64 twin: the encodings are
    checked by disassembling the linked image, which found two real bugs in
    the LINUX AArch64 floor above, and that is the strongest check available
    without a Mac.
    """
    from ..backend.objfile.macho import (
        CPU_SUBTYPE_ARM64_ALL, CPU_TYPE_ARM64, MachoObject, Relocation,
        Symbol,
    )

    ARM64_RELOC_BRANCH26 = 2
    words: list[int] = []
    defined: list = []
    relocs: list = []

    def movz(rd: int, imm: int) -> int:
        """`mov xD, #imm` for a 16-bit immediate -- MOVZ, shift 0."""
        if not 0 <= imm <= 0xFFFF:
            raise ValueError(f"{imm} does not fit one MOVZ")
        return 0xD2800000 | ((imm & 0xFFFF) << 5) | rd

    def here() -> int:
        return len(words) * 4

    def end(name: str, at: int) -> None:
        # THE PLATFORM'S UNDERSCORE -- see the x86-64 twin.
        defined.append(Symbol(name=name if name == "_start" else "_" + name,
                              section=".text", value=at,
                              size=here() - at, binding=1, kind=2))

    SVC80 = 0xD4001001         # svc #0x80  -- Darwin's, not svc #0
    RET = 0xD65F03C0
    BRK0 = 0xD4200000
    #: `b.cc` two instructions on: the Darwin success path, since a syscall
    #: that failed sets the carry flag.
    BCC_TWO = 0x54000043
    NUM = 16                   # x16 holds the number

    # ── _start ──────────────────────────────────────────────────────────────
    at = here()
    relocs.append(Relocation(here(), "_" + entry, ARM64_RELOC_BRANCH26,
                             addend=0, pcrel=True, length=2))
    words.append(0x94000000)                       # bl _uasm_main
    words.append(0xAA0003E0)                       # mov x0, x0  (status)
    words.append(movz(NUM, _DARWIN["exit"]))
    words.append(SVC80)
    words.append(BRK0)                             # exit does not return
    end("_start", at)

    # ── plat_write(fd, buf, n) -> i64 ───────────────────────────────────────
    at = here()
    words.append(movz(NUM, _DARWIN["write"]))
    words.append(SVC80)
    words.append(BCC_TWO)                          # carry clear: it worked
    words.append(0x92800000)                       # movn x0, #0  -> -1
    words.append(RET)
    end("plat_write", at)

    # ── plat_exit(code) ─────────────────────────────────────────────────────
    at = here()
    words.append(movz(NUM, _DARWIN["exit"]))
    words.append(SVC80)
    words.append(BRK0)
    end("plat_exit", at)

    # ── putchar(c) -> int ───────────────────────────────────────────────────
    at = here()
    words.append(0xD10043FF)                       # sub sp, sp, #16
    words.append(0x390003E0)                       # strb w0, [sp]
    words.append(0xAA0003E9)                       # mov x9, x0   (keep it)
    words.append(0xD2800020)                       # mov x0, #1   (fd)
    words.append(0x910003E1)                       # mov x1, sp   (buf)
    words.append(0xD2800022)                       # mov x2, #1   (len)
    words.append(movz(NUM, _DARWIN["write"]))
    words.append(SVC80)
    words.append(0x12001D20)                       # and w0, w9, #0xff
    words.append(0x910043FF)                       # add sp, sp, #16
    words.append(RET)
    end("putchar", at)

    # ── plat_heap(n) -> ptr ─────────────────────────────────────────────────
    at = here()
    words.append(0xF100001F)                       # cmp x0, #0
    words.append(0x5400006C)                       # b.gt +12 (three words on)
    words.append(0xAA1F03E0)                       # mov x0, xzr
    words.append(RET)
    words.append(0xAA0003E1)                       # mov x1, x0   (length)
    words.append(0xAA1F03E0)                       # mov x0, xzr  (addr)
    words.append(movz(2, PROT_READ | PROT_WRITE))
    words.append(movz(3, DARWIN_MAP_PRIVATE | DARWIN_MAP_ANON))
    words.append(0x92800004)                       # movn x4, #0  -> -1 (fd)
    words.append(0xAA1F03E5)                       # mov x5, xzr  (offset)
    words.append(movz(NUM, _DARWIN["mmap"]))
    words.append(SVC80)
    words.append(BCC_TWO)                          # carry clear: it worked
    words.append(0xAA1F03E0)                       # mov x0, xzr
    words.append(RET)
    end("plat_heap", at)

    text = b"".join(struct.pack("<I", w) for w in words)
    obj = MachoObject(CPU_TYPE_ARM64, CPU_SUBTYPE_ARM64_ALL)
    _macho_symbols(obj, text, defined, ("_" + entry,), CPU_TYPE_ARM64)
    for rel in relocs:
        obj.relocate(".text", rel)
    return obj


def _windows_aarch64_object(entry: str):
    """The floor and the entry point for Windows on ARM.

    THE SAME IMPORT TABLE AS x86-64 WINDOWS and none of the same
    instructions. `kernel32.dll` is still the only way to the kernel, so
    every call here goes through the IAT -- but AArch64 has no indirect call
    through a PC-relative memory operand, so reaching a slot takes three
    instructions rather than one:

        adrp x16, __imp_WriteFile          ; the slot's page
        ldr  x16, [x16, :lo12:__imp_...]   ; the function's address
        blr  x16

    x16 IS THE RIGHT REGISTER FOR IT. The AAPCS calls it IP0 and reserves it
    for exactly this -- a scratch a veneer may use between the caller and the
    callee -- which is also why `emit.py` keeps it out of the allocator.

    AND THE CONVENTION IS AAPCS64, not the x64 one: arguments in x0 through
    x7, NO SHADOW SPACE (that is an x86-64 Windows rule and not an ARM one),
    and the stack 16-aligned at all times. `x18` is the thread environment
    block's on this platform and nothing here touches it.

    NOTHING HERE HAS BEEN EXECUTED. Every word was taken from `llvm-mc
    -show-encoding` and checked back by disassembling the linked image; see
    the x86-64 Windows twin, which says the same.
    """
    from ..backend.objfile import (
        CoffObject, CoffRelocation, CoffSymbol, IMAGE_FILE_MACHINE_ARM64,
        IMAGE_SCN_CNT_CODE, IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ,
    )

    BRANCH26 = 0x0003
    PAGEBASE_REL21 = 0x0004
    PAGEOFFSET_12L = 0x0007
    words: list[int] = []
    symbols: list = []
    relocs: list = []

    def here() -> int:
        return len(words) * 4

    def call(name: str) -> None:
        """`blr` through the IAT slot `__imp_<name>`."""
        slot = "__imp_" + name
        relocs.append(CoffRelocation(here(), slot, PAGEBASE_REL21))
        words.append(0x90000010)                  # adrp x16, slot
        relocs.append(CoffRelocation(here(), slot, PAGEOFFSET_12L))
        words.append(0xF9400210)                  # ldr x16, [x16, :lo12:]
        words.append(0xD63F0200)                  # blr x16

    def end(name: str, at: int) -> None:
        symbols.append(CoffSymbol(name=name, section=".text", value=at,
                                  size=here() - at, binding=1, kind=2))

    RET = 0xD65F03C0
    BRK0 = 0xD4200000
    PUSH_FP = 0xA9BF7BFD          # stp x29, x30, [sp, #-16]!
    POP_FP = 0xA8C17BFD           # ldp x29, x30, [sp], #16
    PUSH_48 = 0xA9BD7BFD          # stp x29, x30, [sp, #-48]!
    POP_48 = 0xA8C37BFD           # ldp x29, x30, [sp], #48
    MOV_FP_SP = 0x910003FD        # mov x29, sp

    # ── _start ──────────────────────────────────────────────────────────────
    at = here()
    # THE STACK IS ALIGNED BY HAND rather than trusted. The loader starts a
    # thread with `sp` 16-aligned, and an unaligned one faults on the first
    # store rather than misbehaving quietly -- so three instructions buy the
    # difference between a program that cannot start and one that cannot
    # start FOR A REASON NOBODY WOULD GUESS.
    words.append(0x910003E9)                      # mov x9, sp
    words.append(0x927CED29)                      # and x9, x9, #-16
    words.append(0x9100013F)                      # mov sp, x9
    relocs.append(CoffRelocation(here(), entry, BRANCH26))
    words.append(0x94000000)                      # bl uasm_main
    call("ExitProcess")                           # the status is in w0
    words.append(BRK0)
    end("_start", at)

    # ── plat_write(fd, buf, n) -> i64 ───────────────────────────────────────
    at = here()
    words.append(PUSH_48)
    words.append(MOV_FP_SP)
    words.append(0xF9000BE1)                      # str x1, [sp, #16]  (buf)
    words.append(0xF9000FE2)                      # str x2, [sp, #24]  (n)
    words.append(0xF100081F)                      # cmp x0, #2
    words.append(0x12800140)                      # mov w0, #-11 (STD_OUTPUT)
    words.append(0x12800169)                      # mov w9, #-12 (STD_ERROR)
    words.append(0x1A800120)                      # csel w0, w9, w0, eq
    call("GetStdHandle")                          # x0 = the handle
    words.append(0xF9400BE1)                      # ldr x1, [sp, #16]
    words.append(0xF9400FE2)                      # ldr x2, [sp, #24]
    words.append(0x910083E3)                      # add x3, sp, #32 (written)
    words.append(0xAA1F03E4)                      # mov x4, xzr (lpOverlapped)
    call("WriteFile")                             # w0 = the BOOL
    words.append(0x7100001F)                      # cmp w0, #0
    words.append(0xF9400FE0)                      # ldr x0, [sp, #24]  (n)
    words.append(0x54000041)                      # b.ne +8  (it worked)
    words.append(0x92800000)                      # mov x0, #-1
    words.append(POP_48)
    words.append(RET)
    end("plat_write", at)

    # ── plat_exit(code) ─────────────────────────────────────────────────────
    #
    # NO FRAME. AArch64 pushes no return address, so `sp` is still aligned on
    # entry and `blr` only clobbers x30 -- which this never returns to use.
    at = here()
    call("ExitProcess")
    words.append(BRK0)
    end("plat_exit", at)

    # ── putchar(c) -> int ───────────────────────────────────────────────────
    at = here()
    words.append(PUSH_48)
    words.append(MOV_FP_SP)
    words.append(0x390043E0)                      # strb w0, [sp, #16]
    words.append(0xF9000FE0)                      # str x0, [sp, #24] (keep c)
    words.append(0x12800140)                      # mov w0, #-11
    call("GetStdHandle")
    words.append(0x910043E1)                      # add x1, sp, #16 (the byte)
    words.append(0xD2800022)                      # mov x2, #1
    words.append(0x910083E3)                      # add x3, sp, #32 (written)
    words.append(0xAA1F03E4)                      # mov x4, xzr
    call("WriteFile")
    words.append(0xF9400FE0)                      # ldr x0, [sp, #24]
    words.append(0x12001C00)                      # and w0, w0, #0xff
    words.append(POP_48)
    words.append(RET)
    end("putchar", at)

    # ── plat_heap(n) -> ptr ─────────────────────────────────────────────────
    at = here()
    words.append(PUSH_FP)
    words.append(MOV_FP_SP)
    words.append(0xF100001F)                      # cmp x0, #0
    words.append(0x5400008C)                      # b.gt +16 (the allocation)
    words.append(0xAA1F03E0)                      # mov x0, xzr
    words.append(POP_FP)
    words.append(RET)
    words.append(0xAA0003E1)                      # mov x1, x0   (size)
    words.append(0xAA1F03E0)                      # mov x0, xzr  (address)
    words.append(0x52860002)                      # mov w2, #0x3000
    words.append(0x52800083)                      # mov w3, #4
    call("VirtualAlloc")
    words.append(POP_FP)
    words.append(RET)
    end("plat_heap", at)

    text = b"".join(struct.pack("<I", w) for w in words)
    obj = CoffObject(IMAGE_FILE_MACHINE_ARM64)
    obj.section(".text", text, align=4,
                characteristics=IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE
                                | IMAGE_SCN_MEM_READ)
    for sym in symbols:
        obj.symbol(sym)
    for name in (entry, "__imp_ExitProcess", "__imp_WriteFile",
                 "__imp_GetStdHandle", "__imp_VirtualAlloc"):
        obj.symbol(CoffSymbol(name=name, section="", binding=1))
    for rel in relocs:
        obj.relocate(".text", rel)
    return obj


#: Which builder serves which (container, machine).
_BUILDERS = {
    ("elf", EM_X86_64): _x86_64_object,
    ("elf", EM_AARCH64): _aarch64_object,
    ("coff", 0x8664): _windows_x86_64_object,
    ("coff", 0xAA64): _windows_aarch64_object,
    ("macho", 0x01000007): _macos_x86_64_object,     # CPU_TYPE_X86_64
    ("macho", 0x0100000C): _macos_aarch64_object,    # CPU_TYPE_ARM64
}

#: The names this object defines.
#:
#: THE FLOOR IS STILL THREE FUNCTIONS -- `objects/floor.py` says which, and
#: `test_platform_floor.py` holds it to that. This object is not the floor: it
#: is what a FREESTANDING IMAGE needs from the platform, which is the floor
#: plus the entry point the C runtime start files would have supplied, plus
#: the one libc function the Python frontend emits a call to.
PROVIDES = ("_start", "plat_write", "plat_exit", "plat_heap", "putchar")


def provides(fmt: str = "elf") -> tuple[str, ...]:
    """The same names, spelled the way `fmt`'s platform spells them.

    MACH-O PUTS AN UNDERSCORE IN FRONT OF EVERY C NAME, which is the
    platform's ABI and not this file's choice -- so the floor defines
    `_plat_write` there and `plat_write` everywhere else, and a caller asking
    "does anything here already define the floor" has to ask in the right
    spelling. `_start` is not a C name and keeps its one underscore.
    """
    if fmt != "macho":
        return PROVIDES
    return tuple(name if name == "_start" else "_" + name
                 for name in PROVIDES)


def floor_object(machine: int, *, entry: str = "uasm_main",
                 fmt: str = "elf") -> bytes:
    """The floor and `_start`, as a relocatable object in `fmt`."""
    build = _BUILDERS.get((fmt, machine))
    if build is None:
        known = ", ".join(f"{f} {m:#x}" for f, m in sorted(_BUILDERS))
        raise KeyError(
            f"no freestanding floor for {fmt} machine {machine:#x}; "
            f"there is one for: {known}")
    return build(entry).to_bytes()


__all__ = ["PROVIDES", "floor_object", "provides"]
