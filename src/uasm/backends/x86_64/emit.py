"""x86-64 backend: assembly for either ABI, using the shared register allocator.

Emits GNU-syntax assembly rather than machine code. That is a deliberate
staging decision, not a shortcut: instruction SELECTION and instruction
ENCODING are independent problems, and doing them at once means a bug could be
in either. With assembly as the output, every selection decision is readable,
`as` validates the encoding, and an encoder can be added later underneath a
backend already known to select correctly.

WHAT THIS DEMONSTRATES that the C backend cannot: using `uasm.backend.regalloc`.
Values live in machine registers, spilled ones in frame slots, and the prologue
saves exactly the callee-saved registers the allocation actually used.

THE ABI comes from the target, and there are two:

    System V      args rdi rsi rdx rcx r8 r9;  callee-saved rbx r12-r15
    Microsoft x64 args rcx rdx r8 r9;          callee-saved rbx rsi rdi r12-r15,
                  and the caller reserves 32 bytes of shadow space

Hardcoding one produces code that links fine on the other platform and
corrupts its arguments, which reads as a miscompilation of the callee rather
than of the call.

Arguments beyond the register file go on the stack, and the moves into
argument registers are SCHEDULED rather than emitted in order -- see
`_emit_parallel_moves`, which exists because doing them in order silently
collapsed four arguments into one value.

FLOATS use SSE, and are kept in frame slots rather than allocated. The shared
allocator models one register file and xmm* is a second; handing floats to it
as `on_stack` is slower than allocating them and correct at every register
count, which is the order to do these in. The float ABI is the part worth
reading: System V indexes the integer and SSE argument sequences
independently, Microsoft x64 indexes both by argument POSITION, and the
difference is invisible until a call mixes types.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ...backend.base import (
    ENTRY_SYMBOL, Backend, BackendUnsupported, Target, register,
)
from ...backend.regalloc import (
    Allocation, InRegister, InSlot, RegisterFile, allocate, verify_allocation,
)
from ...ir import Function, Module, types as T
from ...ir.module import Global, Instruction, Linkage, Register
from ...ir.opcodes import Op

@dataclass(frozen=True, slots=True)
class ABI:
    """A calling convention. Selected by target, never hardcoded.

    x86-64 has two in common use and they disagree about everything that
    matters: which registers carry arguments, which survive a call, and whether
    the caller must reserve scratch space. A backend that assumes one produces
    code that links on the other platform and corrupts its arguments -- which
    looks like a miscompilation of the callee, not of the call.
    """

    name: str
    argument_registers: tuple[str, ...]
    callee_saved: frozenset[str]
    #: Bytes the CALLER must reserve above the return address before a call.
    #: Microsoft x64 requires 32 ("shadow space") for the callee to spill its
    #: register arguments into; System V requires none.
    shadow_space: int = 0
    #: SSE registers carrying floating-point arguments.
    float_argument_registers: tuple[str, ...] = ()
    #: True if an argument's POSITION consumes a slot in both sequences.
    #:
    #: The two conventions disagree in a way that is invisible until a call
    #: mixes types. On System V the sequences are independent: f(int, float)
    #: passes the int in rdi and the float in xmm0. On Microsoft x64 they are
    #: one sequence indexed by position: the same call passes the int in rcx
    #: and the float in xmm1, and xmm0 is skipped. Getting this wrong reads
    #: the wrong register in the callee and produces garbage, not a crash.
    positional_float_slots: bool = False

    @property
    def allocation_order(self) -> tuple[str, ...]:
        """Caller-saved first: free in a leaf function, so the allocator only
        reaches for callee-saved (which the prologue must preserve) when the
        volatile ones run out."""
        volatile = [r for r in _ALL_GENERAL if r not in self.callee_saved]
        saved = [r for r in _ALL_GENERAL if r in self.callee_saved]
        return tuple(volatile + saved)

    def register_file(self) -> RegisterFile:
        return RegisterFile(general=self.allocation_order,
                            callee_saved=self.callee_saved,
                            reserved=RESERVED, slot_size=8)


_ALL_GENERAL = ("rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9",
                "rbx", "r12", "r13", "r14", "r15")
RESERVED = frozenset({"rsp", "rbp", "r10", "r11"})

SYSTEM_V = ABI(
    "sysv",
    argument_registers=("rdi", "rsi", "rdx", "rcx", "r8", "r9"),
    callee_saved=frozenset({"rbx", "r12", "r13", "r14", "r15"}),
    float_argument_registers=("xmm0", "xmm1", "xmm2", "xmm3",
                              "xmm4", "xmm5", "xmm6", "xmm7"),
)
MICROSOFT_X64 = ABI(
    "win64",
    argument_registers=("rcx", "rdx", "r8", "r9"),
    # rsi and rdi are callee-saved on Windows and volatile on System V --
    # exactly the kind of difference that silently corrupts a value across a
    # call if the ABI is assumed rather than looked up.
    callee_saved=frozenset({"rbx", "rsi", "rdi", "r12", "r13", "r14", "r15"}),
    shadow_space=32,
    float_argument_registers=("xmm0", "xmm1", "xmm2", "xmm3"),
    positional_float_slots=True,
)


_ABIS = {"sysv": SYSTEM_V, "win64": MICROSOFT_X64}


def abi_for(target: Target) -> ABI:
    """The calling convention `target` declares.

    Reads the field rather than looking for "windows" in the name. Name
    sniffing worked for the two targets that shipped and silently gave System
    V to everything else -- a target called `win64-custom` would have compiled,
    linked, and passed its arguments in the wrong registers, which is a bug
    that appears as corrupted data in a callee and nowhere near this line.
    """
    try:
        return _ABIS[target.abi]
    except KeyError:
        raise UnsupportedOperation(
            f"target {target.name!r} declares ABI {target.abi!r}, which this "
            f"backend does not implement (knows: "
            f"{', '.join(sorted(_ABIS))})") from None


@dataclass(frozen=True, slots=True)
class AsmDialect:
    """Object-format-specific assembler directives.

    The instructions are identical across ELF and COFF; the DIRECTIVES around
    them are not. `.type x, @function` and `.size` are ELF-only and are a hard
    error for a COFF assembler, and `.note.GNU-stack` is meaningless outside
    ELF. Emitting one set everywhere produces output that assembles on one
    platform and fails on the other with a message about a stray character.
    """

    #: Prefix applied to every symbol. COFF on 32-bit prefixed with "_";
    #: x86-64 COFF does not, but the hook belongs here rather than in the
    #: emitter, where it would be a special case in twenty places.
    symbol_prefix: str = ""

    def function_header(self, name: str, exported: bool) -> list[str]:
        raise NotImplementedError

    def function_footer(self, name: str) -> list[str]:
        return []

    def file_footer(self) -> list[str]:
        return []

    def align(self, bytes_: int) -> str:
        """Align to `bytes_`, however this assembler spells it.

        A HOOK BECAUSE `.align` MEANS TWO DIFFERENT THINGS. On ELF and COFF
        for x86 it is a byte count; on Mach-O it is a POWER OF TWO, so the
        `.align 8` written for both here would ask a Mach-O assembler for
        256-byte alignment. Over-aligned rather than wrong, and silent, which
        is the kind of difference that survives until someone measures a
        binary and wonders.
        """
        return f"\t.align {bytes_}"


class ElfDialect(AsmDialect):
    def function_header(self, name: str, exported: bool) -> list[str]:
        out = [f"	.globl {name}"] if exported else []
        return out + [f"	.type {name}, @function", f"{name}:"]

    def function_footer(self, name: str) -> list[str]:
        return [f"	.size {name}, .-{name}"]

    def file_footer(self) -> list[str]:
        # Marks the stack non-executable. Without it the linker assumes the
        # worst and marks the whole binary's stack executable.
        return ['	.section .note.GNU-stack,"",@progbits']


class CoffDialect(AsmDialect):
    def function_header(self, name: str, exported: bool) -> list[str]:
        out = [f"	.globl {name}"] if exported else []
        return out + [f"	.def {name}; .scl 2; .type 32; .endef", f"{name}:"]


class MachoDialect(AsmDialect):
    """Mach-O, which is neither of the other two and was silently treated as ELF.

    `dialect_for` answered COFF or ELF, so `--target x86_64-macos` produced
    output byte-identical to the Linux one -- `.type`, `.size` and
    `.note.GNU-stack`, none of which a Mach-O assembler accepts, and no
    leading underscore on any symbol, so nothing would have resolved even if
    it had assembled. The target has been registered since the beginning and
    that is exactly the failure the `Target` docstring warns about: a platform
    a backend never asked about is a platform it gets wrong quietly.

    EVERY SYMBOL WEARS AN UNDERSCORE. Mach-O's C ABI prefixes them, so `main`
    is `_main` and a call to `printf` is a call to `_printf`. `symbol_prefix`
    already existed for exactly this and had no user; `Backend.symbol` applies
    it to definitions and call sites alike, which is what keeps the two from
    disagreeing.
    """

    def function_header(self, name: str, exported: bool) -> list[str]:
        # No `.type`: Mach-O has no such directive, and its assembler stops.
        out = [f"	.globl {name}"] if exported else []
        return out + [f"{name}:"]

    def align(self, bytes_: int) -> str:
        # A POWER OF TWO here, not a byte count. `.p2align` says so in the
        # spelling rather than in a comment, and means the same on all three.
        return f"\t.p2align {max(0, (bytes_ - 1).bit_length())}"


def _clobbers_volatiles(ins: Instruction) -> bool:
    """Whether this instruction destroys caller-saved registers.

    Not the same question as "is it a call" in the IR. This backend has no
    float remainder instruction and lowers `Op.REM` on a float to `call
    fmod`, which clobbers the volatiles exactly as any call does -- but the
    IR shows an arithmetic opcode, so the shared liveness saw nothing and the
    allocator put a value live across it in `rax`. `float(i0)` a few
    instructions later read whatever fmod had left behind.

    Any lowering that introduces a call belongs here. Forgetting one does not
    fail to build; it produces a wrong number, sometimes.
    """
    return (ins.op in (Op.CALL, Op.CALL_PTR)
            or (ins.op is Op.REM and ins.ty.is_float))


#: Sub-register names for the two scratch registers, by width. Needed because
#: a load or store must touch EXACTLY the type's width: `movq` on an `i8.load`
#: reads eight bytes, and `i8.load` of a byte table returned 0x07060504
#: instead of 4 -- a plausible integer, from four bytes past the one asked
#: for.
_SUBREG = {
    "r10": {8: "r10b", 16: "r10w", 32: "r10d", 64: "r10"},
    "r11": {8: "r11b", 16: "r11w", 32: "r11d", 64: "r11"},
}

#: Load mnemonic by width and signedness. A narrow load must extend into the
#: full register, because the slot holding the value is 64 bits wide and
#: everything downstream reads all of it. Signed types sign-extend so that
#: -1 as an i8 stays -1, matching what the interpreter's `_wrap` produces.
_LOAD = {
    (8, True): "movsbq", (8, False): "movzbq",
    (16, True): "movswq", (16, False): "movzwq",
    (32, True): "movslq", (32, False): "movl",
    (64, True): "movq", (64, False): "movq",
}


def _store_mnemonic(bits: int) -> str:
    return {8: "movb", 16: "movw", 32: "movl", 64: "movq"}[bits]


def _access_width(ty: T.Type) -> int:
    """The width a load or store of `ty` touches.

    `i1` is stored as a byte: it is the width the C backend uses for it too,
    and the two must agree or a value written by one is misread by the other.
    """
    return 8 if ty is T.I1 else ty.bits


def _normalise(e: "_Emitter", scratch: str, ty: T.Type) -> None:
    """Force `scratch` to hold a valid value of `ty`.

    Every arithmetic instruction here operates at 64 bits, because that is
    what the registers are. For a narrow type the result must then be brought
    back into range, and it was not: `i8.add 127, 1` produced 128, where the
    IR says an i8 wraps to -128. The C backend casts to the type and got this
    right from the start; this one computed in the wrong width and kept the
    answer.

    Signed types sign-extend and unsigned types zero-extend, so the 64-bit
    slot always holds the value the interpreter's `_wrap` would produce --
    which is what makes a comparison between the two meaningful.
    """
    bits = ty.bits
    if ty.is_ptr or ty.is_float or bits >= 64:
        return
    if bits == 1:
        e.emit(f"andq $1, %{scratch}")
        return
    sub = _SUBREG[scratch][bits]
    if bits == 32 and not ty.is_signed:
        # Writing a 32-bit register zeroes the upper half; no widening needed.
        e.emit(f"movl %{sub}, %{sub}")
        return
    widen = {(8, True): "movsbq", (8, False): "movzbq",
             (16, True): "movswq", (16, False): "movzwq",
             (32, True): "movslq"}[(bits, ty.is_signed)]
    e.emit(f"{widen} %{sub}, %{scratch}")


def block_label(fn_name: str, label: str) -> str:
    """The assembler label for one IR block.

    One function, used by the definition and by every branch. They were
    computed separately once and drifted the moment the function symbol was
    renamed, which the assembler accepted and the linker rejected.
    """
    return f".L_{fn_name}_{label}"


def dialect_for(target: Target) -> AsmDialect:
    """The directives for this target's object format.

    A TABLE RATHER THAN A CHAIN OF GUESSES. The two-way `coff else elf` this
    replaces gave Mach-O the ELF dialect, so macOS was wrong and said nothing;
    an unknown format now falls to ELF deliberately and in one visible place.
    """
    if target.object_format == "coff":
        return CoffDialect()
    if target.object_format == "macho":
        return MachoDialect(symbol_prefix="_")
    return ElfDialect()

#: 64-bit name -> the sub-register of a given width. Needed because a `mov`
#: into `al` leaves the upper 56 bits of `rax` untouched, so a comparison
#: result must be zero-extended before it is used as a 64-bit value.
_BYTE = {
    "rax": "al", "rcx": "cl", "rdx": "dl", "rsi": "sil", "rdi": "dil",
    "rbx": "bl", "r8": "r8b", "r9": "r9b", "r10": "r10b", "r11": "r11b",
    "r12": "r12b", "r13": "r13b", "r14": "r14b", "r15": "r15b",
}

_SET_FOR = {
    (Op.EQ, True): "sete", (Op.NE, True): "setne",
    (Op.LT, True): "setl", (Op.LE, True): "setle",
    (Op.GT, True): "setg", (Op.GE, True): "setge",
    (Op.EQ, False): "sete", (Op.NE, False): "setne",
    (Op.LT, False): "setb", (Op.LE, False): "setbe",
    (Op.GT, False): "seta", (Op.GE, False): "setae",
}

_SIMPLE_BINOP = {
    Op.ADD: "addq", Op.SUB: "subq", Op.MUL: "imulq",
    Op.AND: "andq", Op.OR: "orq", Op.XOR: "xorq",
}


def _fsuffix(ty: T.Type) -> str:
    """`sd` for a double, `ss` for a single. The whole float ISA keys on it."""
    return "sd" if ty is T.F64 else "ss"


def _fmov(ty: T.Type) -> str:
    return "movsd" if ty is T.F64 else "movss"


#: Scratch for breaking a cycle of register moves. Reserved, so the allocator
#: never holds a live value in it.
_MOVE_SCRATCH = "r11"


#: Staging register for a memory-to-memory move. Distinct from the cycle
#: scratch so the two can never be in flight at once.
_COPY_SCRATCH = "r10"


def _emit_parallel_moves(e: "_Emitter", moves: list[tuple[str, str]]) -> None:
    """Emit `dst <- src` moves as if they happened simultaneously.

    Both ends are operand strings -- `%rcx` or `-8(%rbp)` -- because this runs
    in two places that are mirror images:

        the caller   values in registers and slots  ->  argument registers
        the prologue argument registers             ->  registers and slots

    Both are parallel assignments: every source is read from the state before
    the transfer, not the half-updated state the earlier moves leave. Doing
    them in argument order is correct only when no destination is also a
    source, and silently wrong otherwise -- the overwritten value is gone and
    the callee gets a duplicate of another argument.

    That was fixed for the caller and not for the prologue, so a six-argument
    call still produced `movq %rcx, %r8` (p0 to its slot) ahead of
    `movq %r8, %rcx` (p2 from its slot), and p2 arrived holding p0.

    The rule: emit any move whose DESTINATION is nobody else's source, repeat,
    and break a remaining cycle through a scratch register. A cycle is real --
    `f(b, a)` with a and b already in each other's registers -- and no
    ordering resolves it.

    A memory source cannot be clobbered here, so it is never part of a cycle.
    Its DESTINATION still can be someone's source, which is the subtlety that
    made "emit the memory moves first" look safe and reintroduce the same bug
    one layer down.
    """
    def is_register(operand: str) -> bool:
        return operand.startswith("%")

    def emit(dst: str, src: str) -> None:
        if is_register(dst) or is_register(src):
            e.emit(f"movq {src}, {dst}")
        else:
            # Neither end is a register, so nothing else can observe the
            # staging register in between.
            e.emit(f"movq {src}, %{_COPY_SCRATCH}")
            e.emit(f"movq %{_COPY_SCRATCH}, {dst}")

    pending = [(dst, src) for dst, src in moves if dst != src]

    while pending:
        # Only a register source can be destroyed by another move, so only
        # those constrain the order.
        sources = {src for _, src in pending if is_register(src)}
        ready = [m for m in pending if m[0] not in sources]
        if not ready:
            dst, src = pending[0]
            e.emit(f"movq {src}, %{_MOVE_SCRATCH}")
            scratch = f"%{_MOVE_SCRATCH}"
            pending = [(d, scratch if s == src else s) for d, s in pending]
            continue
        for move in ready:
            emit(move[0], move[1])
            pending.remove(move)


@dataclass(frozen=True, slots=True)
class _Place:
    """Where one argument travels: a register, or a slot on the stack."""

    register: str = ""
    is_float: bool = False
    #: Byte offset from rsp at the call, or None for a register argument.
    stack_offset: int | None = None

    @property
    def on_stack(self) -> bool:
        return self.stack_offset is not None


def _ucomis(ty: T.Type) -> str:
    """The ordered-compare mnemonic.

    NOT `"ucomis" + _fsuffix(ty)`: that spells `ucomissd`, which is not an
    instruction. The suffix convention breaks for exactly this mnemonic
    because it already ends in `s`.
    """
    return "ucomisd" if ty is T.F64 else "ucomiss"


def _float_bits(ty: T.Type, value: float) -> int:
    """The IEEE-754 bit pattern of `value` at `ty`'s width.

    Emitted as an integer and moved into an SSE register, rather than placed
    in .rodata and loaded. Both work; this keeps the constant next to its use
    and avoids a second section, at the cost of two instructions.
    """
    import struct
    fmt = "<d" if ty is T.F64 else "<f"
    return int.from_bytes(struct.pack(fmt, float(value)), "little")


#: `ucomis*` sets the flags as an UNSIGNED comparison, so the set codes are
#: the unsigned ones. It also sets PF when either operand is NaN, and that is
#: the case worth being careful about: every comparison against NaN is false in
#: Python, including `nan == nan`.
#:
#: For `<` and `<=` the operands are swapped and the "above" forms used, so an
#: unordered result yields 0 rather than 1. `==` and `!=` cannot be expressed
#: that way and take an extra instruction on PF.
_FLOAT_CMP = {
    Op.LT: ("swap", "seta"),
    Op.LE: ("swap", "setae"),
    Op.GT: ("plain", "seta"),
    Op.GE: ("plain", "setae"),
}


class UnsupportedOperation(BackendUnsupported):
    """This backend cannot emit the requested operation.

    Raised rather than emitting something plausible. A backend that guesses
    produces a program that runs and is wrong, which costs far more to diagnose
    than one that refuses to build.

    It derives from `BackendUnsupported` so the driver turns it into a
    diagnostic. As a bare Exception it reached the user as a traceback with a
    compiler stack in it, which says "you found a bug in uasm" when the true
    message is "this backend does not do that yet; use --backend c".
    """


@dataclass
class _Emitter:
    fn: Function
    alloc: Allocation
    lines: list[str] = field(default_factory=list)
    frame: int = 0
    #: Byte offset from rbp of the word an indirect call parks its target in,
    #: or 0 for a function that makes none. See `Op.CALL_PTR`.
    call_slot: int = 0

    # ── register/slot access ────────────────────────────────────────────────
    def loc(self, reg: Register) -> str:
        place = self.alloc.location(reg)
        return f"%{place.name}" if isinstance(place, InRegister) \
            else f"-{place.offset}(%rbp)"

    def into_scratch(self, reg: Register, scratch: str) -> str:
        """An operand usable as a register, loading a spilled value first."""
        place = self.alloc.location(reg)
        if isinstance(place, InRegister):
            return f"%{place.name}"
        self.emit(f"movq -{place.offset}(%rbp), %{scratch}")
        return f"%{scratch}"

    def store_from(self, scratch: str, reg: Register) -> None:
        place = self.alloc.location(reg)
        if isinstance(place, InRegister):
            if place.name != scratch:
                self.emit(f"movq %{scratch}, %{place.name}")
        else:
            self.emit(f"movq %{scratch}, -{place.offset}(%rbp)")

    # ── floating point ──────────────────────────────────────────────────────
    # Float values always live in frame slots, never in an SSE register across
    # instructions. The shared allocator models ONE register file, and x86-64
    # floats are a second one; rather than teach it about classes (or fork a
    # second allocator that nothing would exercise under pressure), floats are
    # handed to `allocate(on_stack=...)` and moved through xmm0/xmm1 exactly as
    # spilled integers move through r10/r11. Slower than it could be, and
    # correct at every register count -- which is the order to do these in.

    def float_into(self, reg: Register, xmm: str) -> str:
        """Load a float value into `xmm` and return its operand form."""
        ty = self.fn.register_type(reg)
        self.emit(f"{_fmov(ty)} {self.loc(reg)}, %{xmm}")
        return f"%{xmm}"

    def float_store(self, xmm: str, reg: Register) -> None:
        ty = self.fn.register_type(reg)
        self.emit(f"{_fmov(ty)} %{xmm}, {self.loc(reg)}")

    def emit(self, text: str) -> None:
        self.lines.append(f"\t{text}")

    def label(self, text: str) -> None:
        self.lines.append(f"{text}:")


class X86_64Backend(Backend):
    name = "x86-64"
    #: An ELF, COFF or Mach-O object, or its assembly.
    artifacts = (".o", ".s")
    description = "x86-64 machine code: ELF, COFF and Mach-O objects"
    # The machine this is running on, not a platform fixed at
    # authoring time: `uasm build --backend x86-64` on Windows used to
    # emit ELF directives and hand them to a COFF assembler.
    default_target = "host"

    def symbol(self, name: str, dialect: AsmDialect) -> str:
        """The assembler symbol for an IR function name.

        One place, used by definitions AND call sites. They used to be
        computed separately -- definitions applied `dialect.symbol_prefix` and
        calls did not -- so on any dialect with a prefix a program defined
        `_f` and called `f`. Both are here now, so they cannot drift.

        The IR's `main` is renamed. It is not C's `main`: it returns i64 where
        C requires int, and the runtime that provides the real entry point
        would collide with it at link time.
        """
        if name == "main":
            name = ENTRY_SYMBOL
        return dialect.symbol_prefix + name

    def global_symbol(self, name: str, dialect: AsmDialect) -> str:
        """The assembler symbol for an IR global's name.

        ONE PLACE, FOR THE REASON `symbol` GIVES. Globals had the same split
        that functions used to: the definition applied `dialect.symbol_prefix`
        and `GLOBAL_ADDR` did not, so on Mach-O a program defined `___rodata0`
        and referenced `__rodata0`. Nothing caught it because nothing linked a
        Mach-O object -- and when something did, x86-64 reported it as a
        displacement four gigabytes out of range while AArch64 quietly
        resolved every constant to address zero.

        Separate from `symbol` because that one renames `main`, which is a
        fact about the entry point and not about names in general.
        """
        return dialect.symbol_prefix + name

    def emit(self, module: Module, target: Target) -> dict[str, bytes]:
        abi = abi_for(target)
        dialect = dialect_for(target)
        if target.object_format == "macho":
            from .machoemit import object_bytes as macho_bytes
            return {"out.o": macho_bytes(self, module, abi, dialect)}
        if target.object_format == "coff":
            from .coffemit import object_bytes as coff_bytes
            return {"out.obj": coff_bytes(self, module, abi, dialect)}
        if target.object_format == "elf":
            # THE BACKEND DECIDES EVERY BYTE. See `encode.py` and the three
            # object emitters beside it.
            from .objemit import object_bytes
            return {"out.o": object_bytes(self, module, abi, dialect)}
        # A FORMAT WITH NO WRITER YET falls back to assembly, so adding a
        # target is not blocked on adding an object writer for it. Every
        # format this backend currently serves has one, so nothing reaches
        # here -- and `assembly` keeps the text path callable on purpose.
        return self.assembly(module, target)

    def assembly(self, module: Module, target: Target) -> dict[str, bytes]:
        """What this backend generates, as text. See `Backend.assembly`."""
        abi = abi_for(target)
        dialect = dialect_for(target)
        out: list[str] = [
            f"# Generated by the x86-64 backend ({abi.name} ABI).",
            "# Assemble and link:  cc out.s -o out",
            "\t.text",
        ]
        for fn in module.defined_functions():
            out.extend(self._function(fn, abi, dialect))
            out.append("")

        if module.globals:
            out.append("\t.data")
            for g in module.globals:
                out.extend(self._global(g, dialect))

        out.extend(dialect.file_footer())
        return {"out.s": ("\n".join(out) + "\n").encode("utf-8")}

    # ── globals ─────────────────────────────────────────────────────────────
    def _global(self, g: Global, dialect: AsmDialect) -> list[str]:
        name = self.global_symbol(g.name, dialect)
        lines = [f"\t.globl {name}"] if g.linkage is Linkage.EXPORT else []
        lines.append(dialect.align(g.align or 8))
        lines.append(f"{name}:")
        if g.data is None:
            lines.append(f"\t.zero {max(1, g.size)}")
        else:
            body = ", ".join(str(b) for b in g.data)
            lines.append(f"\t.byte {body}")
        return lines

    # ── one function ────────────────────────────────────────────────────────
    def _function(self, fn: Function, abi: ABI,
                  dialect: AsmDialect) -> list[str]:
        # SSE registers are a second register file the shared allocator does
        # not model, so float values are kept in frame slots and moved through
        # xmm0/xmm1. See `_Emitter.float_into`.
        floats = frozenset(r for r, ty in fn.registers.items() if ty.is_float)
        file = abi.register_file()
        alloc = allocate(fn, file, on_stack=floats, is_call=_clobbers_volatiles)
        problems = verify_allocation(fn, alloc, file=file,
                                     is_call=_clobbers_volatiles)
        if problems:
            # An allocation conflict is a compiler bug, and it produces a
            # program that computes the wrong answer rather than crashing. It
            # costs microseconds to check and is never worth skipping.
            raise AssertionError(
                f"register allocation conflict in {fn.name}:\n  "
                + "\n  ".join(problems))

        e = _Emitter(fn, alloc)
        saved = sorted(alloc.used_callee_saved)
        # The frame holds spill slots plus the saved registers, kept to a
        # 16-byte multiple so the stack is aligned at any call we make.
        frame = alloc.frame_size + 8 * len(saved)
        # AND ONE WORD MORE FOR AN INDIRECT CALL'S TARGET. The address has to
        # survive the argument shuffle, and NO REGISTER CAN HOLD IT: an
        # allocated one is a destination the shuffle may write, and the two
        # reserved ones are the shuffle's own scratch. A frame slot is the
        # only place it cannot be reached from. Reserved only where there is
        # such a call, so no other frame grows. See `Op.CALL_PTR`.
        if any(ins.op is Op.CALL_PTR
               for block in fn.blocks for ins in block.instructions):
            frame += 8
            e.call_slot = frame
        frame = (frame + 15) & ~15
        e.frame = frame

        name = self.symbol(fn.name, dialect)
        e.lines.extend(
            dialect.function_header(name, fn.linkage is Linkage.EXPORT))
        e.emit("pushq %rbp")
        e.emit("movq %rsp, %rbp")
        if frame:
            e.emit(f"subq ${frame}, %rsp")
        for i, reg in enumerate(saved):
            e.emit(f"movq %{reg}, -{alloc.frame_size + 8 * (i + 1)}(%rbp)")

        # Arguments arrive in ABI registers; move them where they were
        # allocated. The SAME placement the caller used -- one function
        # computes it for both sides, so a disagreement is impossible rather
        # than merely unlikely.
        places = self._argument_places(
            [fn.register_type(p) for p in fn.params], abi)
        arrivals: list[tuple[str, str]] = []
        for param, place in zip(fn.params, places):
            # A stacked argument sits above the caller's frame: from in here
            # that is rbp + 16 (saved rbp and return address) + the offset
            # from rsp the caller used.
            source = (f"{16 + place.stack_offset}(%rbp)" if place.on_stack
                      else f"%{place.register}")
            if place.is_float:
                # Floats always land in slots, so no float destination is a
                # register and none can be another's source. They need no
                # scheduling and do not interact with the integer file.
                ty = fn.register_type(param)
                if place.on_stack:
                    e.emit(f"{_fmov(ty)} {source}, %xmm0")
                    e.float_store("xmm0", param)
                else:
                    e.float_store(place.register, param)
            else:
                arrivals.append((e.loc(param), source))
        _emit_parallel_moves(e, arrivals)

        for block in fn.blocks:
            # `fn.name`, not the exported symbol: branches inside the function
            # build their targets from `fn.name` too. Using the symbol here
            # made every jump in `main` reference a label that was defined
            # under the renamed one -- an undefined-symbol error at link time
            # for a function whose assembly reads correctly.
            e.label(block_label(fn.name, block.label))
            for ins in block.instructions:
                self._instruction(e, ins, saved, abi, dialect)

        e.lines.extend(dialect.function_footer(name))
        return e.lines

    # ── calls ───────────────────────────────────────────────────────────────
    @staticmethod
    def _argument_places(types: list, abi: ABI) -> list[_Place]:
        """Where each argument goes.

        Computed once for the caller AND the prologue, so the two cannot
        disagree about a convention neither of them owns.

        The conventions differ in two ways that only show on calls the simple
        cases never make. They index the register sequences differently --
        `f(int, float)` puts the float in xmm0 under System V and in xmm1
        under Microsoft x64, which skips a slot for the int. And once the
        registers run out, arguments go on the stack, in order, above the
        shadow space where the ABI reserves one.
        """
        places: list[_Place] = []
        int_index = float_index = stack_index = 0
        for position, ty in enumerate(types):
            pool = (abi.float_argument_registers if ty.is_float
                    else abi.argument_registers)
            index = position if abi.positional_float_slots else (
                float_index if ty.is_float else int_index)
            if index < len(pool):
                places.append(_Place(register=pool[index],
                                     is_float=ty.is_float))
            else:
                places.append(_Place(
                    is_float=ty.is_float,
                    stack_offset=abi.shadow_space + 8 * stack_index))
                stack_index += 1
            if ty.is_float:
                float_index += 1
            else:
                int_index += 1
        return places

    def _place_arguments(self, e: _Emitter, ins: Instruction, abi: ABI, *,
                         skip_first: bool) -> int:
        """Move a call's arguments where the ABI wants them.

        Returns the number of bytes subtracted from rsp, which the caller must
        add back. The adjustment covers the shadow space and any stacked
        arguments together, and is rounded to a multiple of 16: rsp is
        16-aligned at every call site here, and an ABI-conforming callee is
        entitled to assume that. Adjusting by a non-multiple works until the
        callee uses an aligned SSE store, and then faults.
        """
        args = ins.args[1:] if skip_first else list(ins.args)
        places = self._argument_places([e.fn.register_type(a) for a in args],
                                       abi)
        stacked = sum(1 for p in places if p.stack_offset is not None)
        adjust = abi.shadow_space + 8 * stacked
        adjust = (adjust + 15) & ~15
        if adjust:
            e.emit(f"subq ${adjust}, %rsp")

        # Stacked arguments first: loading a register argument and then
        # touching rsp would be fine, but doing the memory stores while the
        # scratch registers are still free keeps this in one direction.
        for arg, place in zip(args, places):
            if place.stack_offset is None:
                continue
            if place.is_float:
                e.float_into(arg, "xmm0")
                ty = e.fn.register_type(arg)
                e.emit(f"{_fmov(ty)} %xmm0, {place.stack_offset}(%rsp)")
            else:
                src = e.into_scratch(arg, "r10")
                if src != "%r10":
                    e.emit(f"movq {src}, %r10")
                e.emit(f"movq %r10, {place.stack_offset}(%rsp)")

        # Register arguments last, and ORDERED. Emitting them in argument
        # order is wrong whenever an argument's source register is another
        # argument's destination:
        #
        #     movq %rax, %rcx     arg0 (in rax) -> rcx
        #     movq %rcx, %rdx     arg1 was IN rcx, which the line above just
        #                         destroyed -- so arg1 becomes arg0
        #
        # It produced 30 for a call that should have summed to 36: four
        # arguments collapsed into one value, silently, on a program with
        # enough arguments to fill the register file. Floats never appear here
        # -- they live in frame slots, so their sources are memory and cannot
        # be clobbered by a register write.
        moves: list[tuple[str, str]] = []
        for arg, place in zip(args, places):
            if place.on_stack:
                continue
            if place.is_float:
                e.float_into(arg, place.register)
            else:
                moves.append((f"%{place.register}", e.loc(arg)))
        _emit_parallel_moves(e, moves)
        return adjust

    # ── floating point ──────────────────────────────────────────────────────
    #: The opcodes the SSE path implements. An explicit list, not a test on
    #: `ins.ty.is_float`: `ret` of a double also has a float type, and routing
    #: it here made returning a float from a function an "unimplemented
    #: operation" while returning an int worked. Calls and terminators handle
    #: their own float cases in the main dispatch, because the rest of what
    #: they do -- the ABI, the epilogue -- is identical either way.
    _FLOAT_PATH = frozenset({
        Op.CONST, Op.COPY, Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.REM, Op.NEG,
        Op.EQ, Op.NE, Op.LT, Op.LE, Op.GT, Op.GE,
        Op.FTOI, Op.ITOF, Op.FTOF, Op.LOAD, Op.STORE,
    })

    @classmethod
    def _is_float_op(cls, e: _Emitter, ins: Instruction) -> bool:
        """Whether this instruction belongs to the SSE path.

        `ins.ty` alone does not decide it. A comparison of two doubles has
        `ty=f64` and defines an i1; a conversion from f64 to i64 has `ty=i64`
        and reads a float. Both are float operations, and `ftoi` sent down the
        integer path would `movq` a bit pattern and call it an integer.
        """
        if ins.op not in cls._FLOAT_PATH:
            return False
        if ins.op in (Op.FTOI, Op.ITOF, Op.FTOF) or ins.ty.is_float:
            return True
        return bool(ins.args) and e.fn.register_type(ins.args[0]).is_float

    def _float_instruction(self, e: _Emitter, ins: Instruction,
                           abi: ABI) -> None:
        op, ty = ins.op, ins.ty
        sfx = _fsuffix(ty)

        match op:
            case Op.CONST:
                bits = _float_bits(ty, ins.imm)
                if ty is T.F64:
                    e.emit(f"movabsq ${bits}, %r10")
                    e.emit("movq %r10, %xmm0")
                else:
                    e.emit(f"movl ${bits}, %r10d")
                    e.emit("movd %r10d, %xmm0")
                e.float_store("xmm0", ins.dst)

            case Op.COPY:
                e.float_into(ins.args[0], "xmm0")
                e.float_store("xmm0", ins.dst)

            case Op.ADD | Op.SUB | Op.MUL | Op.DIV:
                mnemonic = {Op.ADD: "add", Op.SUB: "sub",
                            Op.MUL: "mul", Op.DIV: "div"}[op]
                e.float_into(ins.args[0], "xmm0")
                e.float_into(ins.args[1], "xmm1")
                e.emit(f"{mnemonic}{sfx} %xmm1, %xmm0")
                e.float_store("xmm0", ins.dst)

            case Op.REM:
                # SSE has no remainder instruction. `a - trunc(a/b)*b` loses
                # precision once the quotient is large, so this calls libm's
                # fmod, which is exact -- the link stage adds -lm where libm is
                # separate. Arguments in xmm0/xmm1 and the result in xmm0 under
                # both ABIs, so no placement logic is needed.
                e.float_into(ins.args[0], "xmm0")
                e.float_into(ins.args[1], "xmm1")
                if abi.shadow_space:
                    e.emit(f"subq ${abi.shadow_space}, %rsp")
                e.emit("call " + ("fmod" if ty is T.F64 else "fmodf"))
                if abi.shadow_space:
                    e.emit(f"addq ${abi.shadow_space}, %rsp")
                e.float_store("xmm0", ins.dst)

            case Op.NEG:
                # Flip the sign bit rather than subtracting from zero: `0.0 -
                # x` gives +0.0 for x = +0.0, where Python's `-0.0` is -0.0,
                # and the difference is observable through division.
                mask = 1 << (63 if ty is T.F64 else 31)
                e.float_into(ins.args[0], "xmm0")
                if ty is T.F64:
                    e.emit(f"movabsq ${mask}, %r10")
                    e.emit("movq %r10, %xmm1")
                    e.emit("xorpd %xmm1, %xmm0")
                else:
                    e.emit(f"movl ${mask}, %r10d")
                    e.emit("movd %r10d, %xmm1")
                    e.emit("xorps %xmm1, %xmm0")
                e.float_store("xmm0", ins.dst)

            case Op.EQ | Op.NE:
                # NaN sets PF, and `nan == nan` is false in Python while
                # `sete` alone would say true: unordered also sets ZF.
                e.float_into(ins.args[0], "xmm0")
                e.float_into(ins.args[1], "xmm1")
                e.emit(f"{_ucomis(ty)} %xmm1, %xmm0")
                if op is Op.EQ:
                    e.emit("sete %r10b")
                    e.emit("setnp %r11b")
                    e.emit("andb %r11b, %r10b")
                else:
                    e.emit("setne %r10b")
                    e.emit("setp %r11b")
                    e.emit("orb %r11b, %r10b")
                e.emit("movzbq %r10b, %r10")
                e.store_from("r10", ins.dst)

            case Op.LT | Op.LE | Op.GT | Op.GE:
                order, setcc = _FLOAT_CMP[op]
                a, b = ins.args
                if order == "swap":
                    a, b = b, a
                e.float_into(a, "xmm0")
                e.float_into(b, "xmm1")
                e.emit(f"{_ucomis(ty)} %xmm1, %xmm0")
                e.emit(f"{setcc} %r10b")
                e.emit("movzbq %r10b, %r10")
                e.store_from("r10", ins.dst)

            case Op.FTOI:
                # cvtt*, not cvt*: truncation toward zero, which is what C and
                # Python's int() do. The non-truncating form rounds to nearest
                # and would make int(2.7) either 2 or 3 depending on the
                # rounding mode the process happened to be in.
                src = e.fn.register_type(ins.args[0])
                e.float_into(ins.args[0], "xmm0")
                e.emit(f"cvtt{_fsuffix(src)}2si %xmm0, %r10")
                e.store_from("r10", ins.dst)

            case Op.ITOF:
                src = e.into_scratch(ins.args[0], "r10")
                if src != "%r10":
                    e.emit(f"movq {src}, %r10")
                e.emit(f"cvtsi2{sfx}q %r10, %xmm0")
                e.float_store("xmm0", ins.dst)

            case Op.FTOF:
                src = e.fn.register_type(ins.args[0])
                e.float_into(ins.args[0], "xmm0")
                if src is not ty:
                    e.emit(f"cvt{_fsuffix(src)}2{sfx} %xmm0, %xmm0")
                e.float_store("xmm0", ins.dst)

            case Op.LOAD:
                addr = e.into_scratch(ins.args[0], "r11")
                e.emit(f"{_fmov(ty)} ({addr}), %xmm0")
                e.float_store("xmm0", ins.dst)

            case Op.STORE:
                e.float_into(ins.args[0], "xmm0")
                addr = e.into_scratch(ins.args[1], "r11")
                e.emit(f"{_fmov(e.fn.register_type(ins.args[0]))} %xmm0, ({addr})")

            case _:
                raise UnsupportedOperation(
                    f"{e.fn.name}: {op.value} on {ty} is not implemented by "
                    f"this backend; use --backend c")

    # ── one instruction ─────────────────────────────────────────────────────
    def _instruction(self, e: _Emitter, ins: Instruction, saved: list[str],
                     abi: ABI, dialect: AsmDialect) -> None:
        op, ty = ins.op, ins.ty
        fn_name = e.fn.name

        # Floating point is a different instruction set on the same machine:
        # different registers, different mnemonics, different comparison
        # semantics. Split here rather than adding an `if ty.is_float` inside
        # each of twenty cases.
        if self._is_float_op(e, ins):
            self._float_instruction(e, ins, abi)
            return

        match op:
            case Op.CONST:
                e.emit(f"movq ${int(ins.imm)}, %r10")
                e.store_from("r10", ins.dst)

            case Op.COPY:
                src = e.into_scratch(ins.args[0], "r10")
                e.emit(f"movq {src}, %r10") if src != "%r10" else None
                e.store_from("r10", ins.dst)

            case Op.GLOBAL_ADDR | Op.FUNC_ADDR:
                # A FUNCTION'S NAME AND A GLOBAL'S ARE SPELLED DIFFERENTLY --
                # `symbol` renames `main` and `global_symbol` does not -- so
                # sharing the case must not mean sharing the lookup.
                target = (self.symbol(ins.sym, dialect) if op is Op.FUNC_ADDR
                          else self.global_symbol(ins.sym, dialect))
                e.emit(f"leaq {target}(%rip), %r10")
                e.store_from("r10", ins.dst)

            case _ if op in _SIMPLE_BINOP:
                a = e.into_scratch(ins.args[0], "r10")
                if a != "%r10":
                    e.emit(f"movq {a}, %r10")
                b = e.into_scratch(ins.args[1], "r11")
                e.emit(f"{_SIMPLE_BINOP[op]} {b}, %r10")
                _normalise(e, "r10", ty)
                e.store_from("r10", ins.dst)

            case Op.DIV | Op.REM:
                # idiv divides rdx:rax and clobbers both, so they are saved
                # around it -- the allocator does not model an instruction
                # demanding specific registers, and teaching it that is a much
                # larger change than spilling two registers here.
                #
                # ORDER MATTERS. The divisor must be moved somewhere safe
                # BEFORE rax is loaded with the dividend: if the allocator put
                # the divisor in rax, loading the dividend destroys it, and the
                # division silently uses the wrong operand. That produced
                # 17 % -5 == 0 instead of -3.
                e.emit("pushq %rax")
                e.emit("pushq %rdx")
                divisor = e.into_scratch(ins.args[1], "r11")
                if divisor != "%r11":
                    e.emit(f"movq {divisor}, %r11")
                dividend = e.into_scratch(ins.args[0], "r10")
                e.emit(f"movq {dividend}, %rax")
                if ins.ty.is_signed:
                    e.emit("cqto")
                    e.emit("idivq %r11")
                else:
                    e.emit("xorq %rdx, %rdx")
                    e.emit("divq %r11")
                e.emit(f"movq %{'rax' if op is Op.DIV else 'rdx'}, %r10")
                e.emit("popq %rdx")
                e.emit("popq %rax")
                _normalise(e, "r10", ty)
                e.store_from("r10", ins.dst)

            case Op.NEG:
                a = e.into_scratch(ins.args[0], "r10")
                if a != "%r10":
                    e.emit(f"movq {a}, %r10")
                e.emit("negq %r10")
                _normalise(e, "r10", ty)
                e.store_from("r10", ins.dst)

            case Op.NOT:
                a = e.into_scratch(ins.args[0], "r10")
                if a != "%r10":
                    e.emit(f"movq {a}, %r10")
                e.emit("notq %r10")
                _normalise(e, "r10", ty)
                e.store_from("r10", ins.dst)

            case Op.SHL | Op.SHR:
                a = e.into_scratch(ins.args[0], "r10")
                if a != "%r10":
                    e.emit(f"movq {a}, %r10")
                b = e.into_scratch(ins.args[1], "r11")
                e.emit("pushq %rcx")
                e.emit(f"movq {b}, %rcx")
                mnemonic = "shlq" if op is Op.SHL else (
                    "sarq" if ins.ty.is_signed else "shrq")
                e.emit(f"{mnemonic} %cl, %r10")
                e.emit("popq %rcx")
                _normalise(e, "r10", ty)
                e.store_from("r10", ins.dst)

            case Op.EQ | Op.NE | Op.LT | Op.LE | Op.GT | Op.GE:
                a = e.into_scratch(ins.args[0], "r10")
                if a != "%r10":
                    e.emit(f"movq {a}, %r10")
                b = e.into_scratch(ins.args[1], "r11")
                e.emit(f"cmpq {b}, %r10")
                e.emit(f"{_SET_FOR[(op, ins.ty.is_signed)]} %r10b")
                # setcc writes one byte; the upper 56 bits keep whatever was
                # there. Without this the i1 is not 0 or 1 and every later test
                # of it is wrong.
                e.emit("movzbq %r10b, %r10")
                e.store_from("r10", ins.dst)

            case Op.TRUNC | Op.EXTEND | Op.BITCAST:
                a = e.into_scratch(ins.args[0], "r10")
                if a != "%r10":
                    e.emit(f"movq {a}, %r10")
                # Masking alone is wrong for a signed narrow type:
                # truncating 200 to i8 is -56, and `andq $0xFF` leaves 200.
                _normalise(e, "r10", ty)
                e.store_from("r10", ins.dst)

            case Op.ALLOCA:
                size = (int(ins.imm) + 15) & ~15
                e.emit(f"subq ${size}, %rsp")
                e.emit("movq %rsp, %r10")
                e.store_from("r10", ins.dst)

            case Op.LOAD:
                addr = e.into_scratch(ins.args[0], "r11")
                bits = _access_width(ty)
                mnemonic = _LOAD[(bits, ty.is_signed)]
                dest = "%r10d" if mnemonic == "movl" else "%r10"
                e.emit(f"{mnemonic} ({addr}), {dest}")
                e.store_from("r10", ins.dst)

            case Op.STORE:
                value = e.into_scratch(ins.args[0], "r10")
                if value != "%r10":
                    e.emit(f"movq {value}, %r10")
                addr = e.into_scratch(ins.args[1], "r11")
                bits = _access_width(ty)
                e.emit(f"{_store_mnemonic(bits)} "
                       f"%{_SUBREG['r10'][bits]}, ({addr})")

            case Op.OFFSET:
                base = e.into_scratch(ins.args[0], "r10")
                if base != "%r10":
                    e.emit(f"movq {base}, %r10")
                off = e.into_scratch(ins.args[1], "r11")
                e.emit(f"addq {off}, %r10")
                e.store_from("r10", ins.dst)

            case Op.CALL | Op.CALL_PTR:
                if op is Op.CALL_PTR:
                    # THE TARGET IS PARKED BEFORE THE ARGUMENTS MOVE, and read
                    # back after. It used to be read AFTER the shuffle, out of
                    # wherever the allocator had put it -- and the shuffle
                    # writes the argument registers, so a callee living in one
                    # of them was overwritten by an argument and the program
                    # jumped to whatever that argument held:
                    #
                    #     mov %r10, %rdi    the callee, as allocated
                    #     mov %rcx, %rdi    argument 0 -- the callee is gone
                    #     ...
                    #     mov %rdi, %r11    read back: argument 0
                    #     call *%r11        jump into a heap address
                    #
                    # measured on `g = f4; g(1, 2, 3, 4)`, which is every
                    # four-argument call through a value: `apy_invoke` and the
                    # IR's `apy_call` both reach a function pointer this way.
                    # Three arguments left a register free and it happened to
                    # survive, which is why this went so long unseen.
                    #
                    # A SLOT AND NOT A REGISTER, because there is no register
                    # that survives: an allocated one may be written as an
                    # argument, and r10 and r11 are the shuffle's own scratch
                    # -- r11 is what it breaks a cycle through, which is what
                    # the comment that stood here had backwards.
                    held = e.into_scratch(ins.args[0], "r10")
                    if held != "%r10":
                        e.emit(f"movq {held}, %r10")
                    e.emit(f"movq %r10, -{e.call_slot}(%rbp)")
                adjust = self._place_arguments(e, ins, abi,
                                               skip_first=op is Op.CALL_PTR)
                if op is Op.CALL:
                    e.emit(f"call {self.symbol(ins.sym, dialect)}")
                else:
                    e.emit(f"movq -{e.call_slot}(%rbp), %r11")
                    e.emit("call *%r11")
                if adjust:
                    e.emit(f"addq ${adjust}, %rsp")
                if ins.dst is not None:
                    if e.fn.register_type(ins.dst).is_float:
                        e.float_store("xmm0", ins.dst)
                    else:
                        e.store_from("rax", ins.dst)

            case Op.JUMP:
                e.emit(f"jmp {block_label(fn_name, ins.labels[0])}")

            case Op.BRANCH:
                cond = e.into_scratch(ins.args[0], "r10")
                e.emit(f"testq {cond}, {cond}")
                e.emit(f"jne {block_label(fn_name, ins.labels[0])}")
                e.emit(f"jmp {block_label(fn_name, ins.labels[1])}")

            case Op.SWITCH:
                value = e.into_scratch(ins.args[0], "r10")
                for case_value, target in ins.cases:
                    e.emit(f"cmpq ${case_value}, {value}")
                    e.emit(f"je {block_label(fn_name, target)}")
                e.emit(f"jmp {block_label(fn_name, ins.labels[0])}")

            case Op.RET:
                if ins.args:
                    if e.fn.register_type(ins.args[0]).is_float:
                        e.float_into(ins.args[0], "xmm0")
                    else:
                        src = e.into_scratch(ins.args[0], "r10")
                        e.emit(f"movq {src}, %rax")
                for i, reg in enumerate(saved):
                    e.emit(f"movq -{e.alloc.frame_size + 8 * (i + 1)}(%rbp), %{reg}")
                e.emit("movq %rbp, %rsp")
                e.emit("popq %rbp")
                e.emit("ret")

            case Op.UNREACHABLE:
                e.emit("ud2")

            case _:
                raise UnsupportedOperation(
                    f"x86-64 backend has no rule for {op.value!r}")


register(X86_64Backend())
