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


#: Which builder serves which (container, machine).
_BUILDERS = {
    ("elf", EM_X86_64): _x86_64_object,
    ("elf", EM_AARCH64): _aarch64_object,
    ("coff", 0x8664): _windows_x86_64_object,
}

#: The names this object defines.
#:
#: THE FLOOR IS STILL THREE FUNCTIONS -- `objects/floor.py` says which, and
#: `test_platform_floor.py` holds it to that. This object is not the floor: it
#: is what a FREESTANDING IMAGE needs from the platform, which is the floor
#: plus the entry point the C runtime start files would have supplied, plus
#: the one libc function the Python frontend emits a call to.
PROVIDES = ("_start", "plat_write", "plat_exit", "plat_heap", "putchar")


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


__all__ = ["PROVIDES", "floor_object"]
