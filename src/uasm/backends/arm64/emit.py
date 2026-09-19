"""AArch64 backend: ARM64 assembly, using the shared register allocator.

THE POINT OF A THIRD BACKEND is to find out whether the parts claimed to be
shared actually are. `liveness` and `regalloc` were written against one
machine and used by one machine, which proves nothing; this one uses them
unchanged, and everything it needed that they did not already provide is a
place the abstraction was wrong.

They provided everything. The only backend-specific input is the register
file and the "does this clobber the volatiles" predicate that x86-64 already
had to add for its own hidden call.

WHERE AARCH64 DIFFERS FROM x86-64, and what each difference costs:

    three-operand      `add x0, x1, x2` does not destroy an operand, so the
                       move-into-place that the x86-64 emitter does before
                       every binary operation is simply absent

    no immediate ops   an arbitrary 64-bit constant takes up to four
                       instructions (movz plus three movk), where x86-64 has
                       a 64-bit `movabsq`

    no memory operands every operand comes from a register, so a spilled
                       value is always an explicit load -- x86-64 can often
                       fold the load into the instruction

    remainder          there is no remainder instruction: `sdiv` then `msub`

    SP alignment       the stack pointer must be 16-byte aligned AT ALL
                       TIMES, not merely at a call. That rules out the
                       push/pop-around-an-instruction trick x86-64 uses for
                       division, and it is why alloca space is reserved in
                       the frame rather than taken by moving SP

FRAME LAYOUT. Slots are addressed from SP with positive offsets, because the
12-bit scaled immediate reaches 32KB there while the signed 9-bit form used
for frame-pointer-relative access reaches only 256 bytes. SP therefore never
moves inside a function body -- `alloca` is served from space reserved up
front, which the IR allows because its size is an immediate.

    sp + 0            spill slots
    sp + alloca_base  alloca space
    sp + saved_base   callee-saved registers
    x29/x30           saved above, by the entry `stp`
"""
from __future__ import annotations

import struct
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


class UnsupportedOperation(BackendUnsupported):
    """This backend cannot emit the requested operation.

    Derives from `BackendUnsupported` so the driver reports it as a
    diagnostic naming the backend, rather than letting a traceback reach the
    user for something that means "use --backend c".
    """


# ── the machine ─────────────────────────────────────────────────────────────
#: Allocatable general-purpose registers, caller-saved first so a leaf
#: function never touches one the prologue would have to preserve.
_VOLATILE = tuple(f"x{i}" for i in range(0, 16))
_CALLEE_SAVED = frozenset(f"x{i}" for i in range(19, 29))
_ALLOCATABLE = _VOLATILE + tuple(f"x{i}" for i in range(19, 29))

#: x16 and x17 are IP0/IP1, reserved by the ABI for linker veneers, which
#: makes them exactly the scratch registers r10/r11 are on x86-64. x18 is the
#: platform register and is left alone. x29/x30 are the frame pointer and
#: link register.
SCRATCH_A, SCRATCH_B = "x16", "x17"
RESERVED = frozenset({"x16", "x17", "x18", "x29", "x30", "sp"})

#: Scratch SIMD registers. Floats live in frame slots and move through these,
#: for the same reason they do on x86-64: the shared allocator models one
#: register file, and this is a second one.
FSCRATCH_A, FSCRATCH_B = "d0", "d1"

#: The largest frame this backend emits, and the encoding that decides it.
#:
#: NOT THE STACK-POINTER ADJUSTMENT. `sub sp, sp, #imm` carries twelve bits,
#: which is 4095 -- but the architecture also lets the immediate be shifted
#: left by twelve, so two instructions reach sixteen megabytes and
#: `_move_sp` below emits them. It was the adjustment that set the old limit
#: of 4088, and it never needed to.
#:
#: IT IS THE SLOT ACCESS, and specifically a THIRTY-TWO-BIT FLOAT. Every
#: `ldr`/`str` with an unsigned offset scales the twelve-bit immediate by the
#: access width: eight bytes for `x` and `d` reaches 32760, four bytes for
#: `s` reaches 16380. A float slot is eight-aligned like every other, so the
#: last one this backend can address is 16376.
#:
#: BEYOND THAT THE ADDRESS HAS TO BE MATERIALISED, which needs a scratch
#: register at every spill site, and a function whose frame is sixteen
#: kilobytes is a function that should be looked at rather than accommodated.
#: So it is refused, by name, with the number.
MAX_FRAME = 16376

#: What one `add`/`sub` immediate holds before the shifted form is needed.
_IMM12 = 4095


def _move_sp(e: "_Emitter", op: str, amount: int) -> None:
    """`add`/`sub sp, sp, #amount`, in as many instructions as it takes.

    THE SHIFTED FORM CARRIES ONLY A MULTIPLE OF 4096, so an amount with
    anything in its low twelve bits is two instructions: the high part
    shifted, then the rest. `encode.py` picks the shift itself for a value
    that is a clean multiple, which is why this splits rather than spelling
    `lsl #12` out.
    """
    if not amount:
        return
    if amount <= _IMM12:
        e.emit(f"{op} sp, sp, #{amount}")
        return
    high, low = amount & ~0xFFF, amount & 0xFFF
    e.emit(f"{op} sp, sp, #{high}")
    if low:
        e.emit(f"{op} sp, sp, #{low}")


def _address_in_frame(e: "_Emitter", dest: str, offset: int) -> None:
    """`add dest, sp, #offset`, in as many instructions as it takes."""
    if offset <= _IMM12:
        e.emit(f"add {dest}, sp, #{offset}")
        return
    high, low = offset & ~0xFFF, offset & 0xFFF
    e.emit(f"add {dest}, sp, #{high}")
    if low:
        e.emit(f"add {dest}, {dest}, #{low}")


@dataclass(frozen=True, slots=True)
class ABI:
    """AAPCS64. One convention, unlike x86-64 -- but still read from the
    target rather than assumed, so a platform that differs can say so."""

    name: str
    argument_registers: tuple[str, ...] = tuple(f"x{i}" for i in range(8))
    float_argument_registers: tuple[str, ...] = tuple(f"d{i}" for i in range(8))
    callee_saved: frozenset[str] = _CALLEE_SAVED

    def register_file(self) -> RegisterFile:
        return RegisterFile(general=_ALLOCATABLE, callee_saved=self.callee_saved,
                            reserved=RESERVED, slot_size=8)


AAPCS64 = ABI("aapcs64")
_ABIS = {"aapcs64": AAPCS64}


def abi_for(target: Target) -> ABI:
    try:
        return _ABIS[target.abi]
    except KeyError:
        raise UnsupportedOperation(
            f"target {target.name!r} declares ABI {target.abi!r}, which this "
            f"backend does not implement (knows: "
            f"{', '.join(sorted(_ABIS))})") from None


# ── condition codes ─────────────────────────────────────────────────────────
#: `cset` conditions, by opcode and signedness. AArch64 spells unsigned
#: comparisons lo/ls/hi/hs, and using the signed ones on a u64 gets every
#: value above 2^63 backwards.
_CONDITION = {
    (Op.EQ, True): "eq", (Op.NE, True): "ne",
    (Op.LT, True): "lt", (Op.LE, True): "le",
    (Op.GT, True): "gt", (Op.GE, True): "ge",
    (Op.EQ, False): "eq", (Op.NE, False): "ne",
    (Op.LT, False): "lo", (Op.LE, False): "ls",
    (Op.GT, False): "hi", (Op.GE, False): "hs",
}

#: Float comparisons through `fcmp`. Unordered (either operand NaN) must
#: yield false for every one of these, which the "mi/ls/gt/ge" set does and
#: the plain signed set does not: `lt` is true when unordered.
_FLOAT_CONDITION = {
    Op.EQ: "eq", Op.NE: "ne",
    Op.LT: "mi", Op.LE: "ls", Op.GT: "gt", Op.GE: "ge",
}

_SIMPLE_BINOP = {
    Op.ADD: "add", Op.SUB: "sub", Op.MUL: "mul",
    Op.AND: "and", Op.OR: "orr", Op.XOR: "eor",
}

_FLOAT_BINOP = {Op.ADD: "fadd", Op.SUB: "fsub", Op.MUL: "fmul", Op.DIV: "fdiv"}

#: The opcodes the SIMD path implements. An explicit set, not a test on
#: `ins.ty.is_float`: `ret` of a double is float-typed too, and routing it
#: here would make returning a float an unimplemented operation.
_FLOAT_PATH = frozenset({
    Op.CONST, Op.COPY, Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.REM, Op.NEG,
    Op.EQ, Op.NE, Op.LT, Op.LE, Op.GT, Op.GE,
    Op.FTOI, Op.ITOF, Op.FTOF, Op.LOAD, Op.STORE,
})


@dataclass(frozen=True, slots=True)
class AsmDialect:
    """Object-format-specific directives, of which this backend needs three.

    AArch64 REACHES THREE OPERATING SYSTEMS and this backend spoke to one.
    `.type`/`.size` are ELF-only and stop a Mach-O assembler dead, and Mach-O
    prefixes every symbol with an underscore -- so `--target aarch64-macos`
    was registered, selected, and emitted ELF, byte-identical to the Linux
    output. Alignment is NOT here, unlike the x86-64 backend's dialect: ARM
    `.align` is a power of two on every format, so there is nothing to vary.
    """

    #: Prepended to every symbol, at definitions AND call sites.
    symbol_prefix: str = ""
    #: Whether `.type`/`.size` exist at all.
    elf_symbol_attributes: bool = True

    def function_header(self, name: str, exported: bool) -> list[str]:
        out = [f"	.globl {name}"] if exported else []
        if self.elf_symbol_attributes:
            # `%function`, not `@function`: on ARM `@` begins a comment, so
            # the ELF spelling every x86 example uses is a silent truncation.
            out.append(f"	.type {name}, %function")
        return out

    def function_footer(self, name: str) -> list[str]:
        return [f"	.size {name}, .-{name}"] if self.elf_symbol_attributes else []


def dialect_for(target: Target) -> AsmDialect:
    """The directives for this target's object format."""
    if target.object_format == "macho":
        return AsmDialect(symbol_prefix="_", elf_symbol_attributes=False)
    return AsmDialect()


def block_label(fn_name: str, label: str) -> str:
    """The assembler label for one IR block.

    One function for the definition and every branch. Computing them
    separately is how x86-64 ended up defining a label under one name and
    jumping to another the moment its entry symbol was renamed.
    """
    return f".L_{fn_name}_{label}"


def _float_bits(ty: T.Type, value: float) -> int:
    fmt = "<d" if ty is T.F64 else "<f"
    return int.from_bytes(struct.pack(fmt, float(value)), "little")


def _fsuffix(ty: T.Type) -> str:
    return "d" if ty is T.F64 else "s"


def _clobbers_volatiles(ins: Instruction) -> bool:
    """Whether this instruction destroys caller-saved registers.

    Not the same as "is it a call in the IR". Float remainder becomes a call
    to fmod here exactly as it does on x86-64, and the shared liveness cannot
    know that. Any lowering that introduces a call belongs in this predicate;
    forgetting one does not fail to build, it produces a wrong number.
    """
    return (ins.op in (Op.CALL, Op.CALL_PTR)
            or (ins.op is Op.REM and ins.ty.is_float))


@dataclass(frozen=True, slots=True)
class _Place:
    """Where one argument travels."""

    register: str = ""
    is_float: bool = False
    stack_offset: int | None = None

    @property
    def on_stack(self) -> bool:
        return self.stack_offset is not None


@dataclass
class _Emitter:
    fn: Function
    alloc: Allocation
    lines: list[str] = field(default_factory=list)
    #: Byte offset from SP where alloca space begins.
    alloca_base: int = 0
    alloca_used: int = 0

    def emit(self, text: str) -> None:
        self.lines.append(f"\t{text}")

    def label(self, text: str) -> None:
        self.lines.append(f"{text}:")

    # ── integer values ──────────────────────────────────────────────────────
    def slot_offset(self, reg: Register) -> int:
        place = self.alloc.location(reg)
        assert isinstance(place, InSlot)
        # The allocator hands out 1-based slot offsets; SP-relative addressing
        # wants them from zero.
        return place.offset - 8

    def into(self, reg: Register, scratch: str, bias: int = 0) -> str:
        """A register holding `reg`'s value, loading a spilled one first.

        `bias` IS HOW FAR SP HAS MOVED since the frame was laid out. It is
        zero everywhere but inside a call sequence, where the outgoing
        argument area has already been pushed and every frame slot is that
        much further up.
        """
        place = self.alloc.location(reg)
        if isinstance(place, InRegister):
            return place.name
        self.emit(f"ldr {scratch}, [sp, #{self.slot_offset(reg) + bias}]")
        return scratch

    def out_register(self, reg: Register, scratch: str) -> str:
        """Where to compute `reg`'s value: its register, or a scratch."""
        place = self.alloc.location(reg)
        return place.name if isinstance(place, InRegister) else scratch

    def store(self, scratch: str, reg: Register) -> None:
        place = self.alloc.location(reg)
        if isinstance(place, InRegister):
            if place.name != scratch:
                self.emit(f"mov {place.name}, {scratch}")
        else:
            self.emit(f"str {scratch}, [sp, #{self.slot_offset(reg)}]")

    # ── float values ────────────────────────────────────────────────────────
    def float_into(self, reg: Register, vreg: str, bias: int = 0) -> str:
        """The float in `vreg`. `bias` as in `into` -- see there.

        THE BIAS WAS MISSING HERE AND PRESENT ON THE INTEGER SIDE, which is
        a call with a stacked argument and a float one reading the float
        from `adjust` bytes below where it lives. Every float in this
        backend lives in a frame slot, so it was every such call.
        """
        ty = self.fn.register_type(reg)
        name = vreg if ty is T.F64 else "s" + vreg[1:]
        self.emit(f"ldr {name}, [sp, #{self.slot_offset(reg) + bias}]")
        return name

    def float_store(self, vreg: str, reg: Register) -> None:
        ty = self.fn.register_type(reg)
        name = vreg if ty is T.F64 else "s" + vreg[1:]
        self.emit(f"str {name}, [sp, #{self.slot_offset(reg)}]")

    # ── constants ───────────────────────────────────────────────────────────
    def materialise(self, value: int, dest: str) -> None:
        """Put a 64-bit constant in `dest`.

        AArch64 has no 64-bit immediate move. `movz` sets one 16-bit field and
        zeroes the rest, and each `movk` overwrites one more without touching
        the others, so any value takes at most four instructions -- and small
        ones, which is nearly all of them, take one.
        """
        v = value & 0xFFFFFFFFFFFFFFFF
        if v == 0:
            self.emit(f"mov {dest}, xzr")
            return
        chunks = [(v >> shift) & 0xFFFF for shift in (0, 16, 32, 48)]
        first = True
        for index, chunk in enumerate(chunks):
            if chunk == 0 and not first:
                continue
            shift = index * 16
            if first:
                self.emit(f"movz {dest}, #{chunk}"
                          + (f", lsl #{shift}" if shift else ""))
                first = False
            else:
                self.emit(f"movk {dest}, #{chunk}"
                          + (f", lsl #{shift}" if shift else ""))
        if first:                       # every chunk was zero
            self.emit(f"mov {dest}, xzr")


def _emit_parallel_moves(e: _Emitter, moves: list[tuple[str, str]]) -> None:
    """Emit `dst <- src` moves as if they happened simultaneously.

    The same scheduler x86-64 needs, for the same reason and with the same
    consequence for getting it wrong: argument setup is a parallel
    assignment, and emitting it in argument order silently collapses two
    arguments into one whenever a destination is also a source.

    Operands are register names or `[sp, #n]` strings, so the caller and the
    prologue -- exact mirror images -- share this rather than each getting it
    slightly wrong.
    """
    def is_register(operand: str) -> bool:
        return not operand.startswith("[")

    def emit(dst: str, src: str) -> None:
        if is_register(dst) and is_register(src):
            e.emit(f"mov {dst}, {src}")
        elif is_register(dst):
            e.emit(f"ldr {dst}, {src}")
        elif is_register(src):
            e.emit(f"str {src}, {dst}")
        else:
            e.emit(f"ldr {SCRATCH_A}, {src}")
            e.emit(f"str {SCRATCH_A}, {dst}")

    pending = [(dst, src) for dst, src in moves if dst != src]
    while pending:
        sources = {src for _, src in pending if is_register(src)}
        ready = [m for m in pending if m[0] not in sources]
        if not ready:
            dst, src = pending[0]
            e.emit(f"mov {SCRATCH_B}, {src}")
            pending = [(d, SCRATCH_B if s == src else s) for d, s in pending]
            continue
        for move in ready:
            emit(move[0], move[1])
            pending.remove(move)


class Arm64Backend(Backend):
    name = "arm64"
    #: An ELF or Mach-O object, or the assembly it came from.
    artifacts = (".o", ".s")
    description = "AArch64 machine code (AAPCS64): ELF and Mach-O objects"
    default_target = "aarch64-none"

    def symbol(self, name: str, dialect: AsmDialect) -> str:
        """The IR's `main` is not C's; see `ENTRY_SYMBOL`.

        The prefix is applied HERE so definitions and call sites cannot
        disagree -- the same reason the x86-64 backend keeps one of these.
        """
        return dialect.symbol_prefix + (ENTRY_SYMBOL if name == "main" else name)

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
            # THE BACKEND DECIDES EVERY WORD. See `encode.py` and the three
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
            f"// Generated by the arm64 backend ({abi.name}).",
            "\t.text",
        ]
        for fn in module.defined_functions():
            out.extend(self._function(fn, abi, dialect))
            out.append("")

        if module.globals:
            out.append("\t.data")
            for g in module.globals:
                out.extend(self._global(g, dialect))

        return {"out.s": ("\n".join(out) + "\n").encode("utf-8")}

    def _global(self, g: Global, dialect: AsmDialect) -> list[str]:
        name = self.global_symbol(g.name, dialect)
        lines = [f"\t.globl {name}"] if g.linkage is Linkage.EXPORT else []
        lines.append(f"\t.align {max(3, (g.align or 8).bit_length() - 1)}")
        lines.append(f"{name}:")
        if g.data is None:
            lines.append(f"\t.zero {max(1, g.size)}")
        else:
            lines.append("\t.byte " + ", ".join(str(b) for b in g.data))
        return lines

    # ── one function ────────────────────────────────────────────────────────
    def _function(self, fn: Function, abi: ABI,
                  dialect: AsmDialect) -> list[str]:
        floats = frozenset(r for r, ty in fn.registers.items() if ty.is_float)
        file = abi.register_file()
        alloc = allocate(fn, file, on_stack=floats, is_call=_clobbers_volatiles)
        problems = verify_allocation(fn, alloc, file=file,
                                     is_call=_clobbers_volatiles)
        if problems:
            raise AssertionError(
                f"register allocation conflict in {fn.name}:\n  "
                + "\n  ".join(problems))

        e = _Emitter(fn, alloc)
        saved = sorted(alloc.used_callee_saved)

        # SP never moves inside the body, so alloca is served from space
        # reserved here. The IR allows it: an alloca's size is an immediate.
        alloca_bytes = sum((int(ins.imm) + 15) & ~15
                           for _, ins in fn.instructions()
                           if ins.op is Op.ALLOCA)
        e.alloca_base = alloc.frame_size
        frame = alloc.frame_size + alloca_bytes + 8 * len(saved)
        frame = (frame + 15) & ~15
        saved_base = alloc.frame_size + alloca_bytes

        if frame > MAX_FRAME:
            # SEE `MAX_FRAME`. Refusing beats emitting a frame whose upper
            # slots silently alias.
            raise UnsupportedOperation(
                f"{fn.name}: frame of {frame} bytes exceeds this backend's "
                f"limit of {MAX_FRAME}",
                )

        name = self.symbol(fn.name, dialect)
        e.lines.extend(
            dialect.function_header(name, fn.linkage is Linkage.EXPORT))
        e.label(name)
        e.emit("stp x29, x30, [sp, #-16]!")
        e.emit("mov x29, sp")
        _move_sp(e, "sub", frame)
        for i, reg in enumerate(saved):
            e.emit(f"str {reg}, [sp, #{saved_base + 8 * i}]")

        places = self._argument_places(
            [fn.register_type(p) for p in fn.params], abi)
        arrivals: list[tuple[str, str]] = []
        for param, place in zip(fn.params, places):
            if place.on_stack:
                # Above the caller's frame: x29 points at the saved x29/x30
                # pair, so the caller's outgoing area starts 16 bytes up.
                source = f"[x29, #{16 + place.stack_offset}]"
            else:
                source = place.register
            if place.is_float:
                ty = fn.register_type(param)
                vreg = FSCRATCH_A if ty is T.F64 else "s0"
                if place.on_stack:
                    e.emit(f"ldr {vreg}, {source}")
                    e.float_store(FSCRATCH_A, param)
                else:
                    e.float_store(place.register, param)
            else:
                location = e.alloc.location(param)
                destination = (location.name if isinstance(location, InRegister)
                               else f"[sp, #{e.slot_offset(param)}]")
                arrivals.append((destination, source))
        _emit_parallel_moves(e, arrivals)

        for block in fn.blocks:
            e.label(block_label(fn.name, block.label))
            for ins in block.instructions:
                self._instruction(e, ins, saved, saved_base, frame, abi,
                                  dialect)

        e.lines.extend(dialect.function_footer(name))
        return e.lines

    # ── calls ───────────────────────────────────────────────────────────────
    @staticmethod
    def _argument_places(types: list, abi: ABI) -> list[_Place]:
        """Where each argument goes.

        AAPCS64 indexes the general and SIMD sequences INDEPENDENTLY, like
        System V and unlike Microsoft x64 -- so `f(int, float)` passes the
        float in d0, not d1.
        """
        places: list[_Place] = []
        int_index = float_index = stack_index = 0
        for ty in types:
            if ty.is_float:
                if float_index < len(abi.float_argument_registers):
                    places.append(_Place(
                        register=abi.float_argument_registers[float_index],
                        is_float=True))
                    float_index += 1
                    continue
            elif int_index < len(abi.argument_registers):
                places.append(_Place(
                    register=abi.argument_registers[int_index]))
                int_index += 1
                continue
            places.append(_Place(is_float=ty.is_float,
                                 stack_offset=8 * stack_index))
            stack_index += 1
        return places

    def _place_arguments(self, e: _Emitter, ins: Instruction, abi: ABI, *,
                         skip_first: bool) -> int:
        """Move a call's arguments into place. Returns the SP adjustment."""
        args = ins.args[1:] if skip_first else list(ins.args)
        places = self._argument_places([e.fn.register_type(a) for a in args],
                                       abi)
        stacked = sum(1 for p in places if p.on_stack)
        adjust = (8 * stacked + 15) & ~15
        _move_sp(e, "sub", adjust)

        for arg, place in zip(args, places):
            if not place.on_stack:
                continue
            if place.is_float:
                vreg = e.float_into(arg, FSCRATCH_A, bias=adjust)
                e.emit(f"str {vreg}, [sp, #{place.stack_offset}]")
            else:
                source = e.into(arg, SCRATCH_A, bias=adjust)
                e.emit(f"str {source}, [sp, #{place.stack_offset}]")

        moves: list[tuple[str, str]] = []
        for arg, place in zip(args, places):
            if place.on_stack:
                continue
            if place.is_float:
                e.float_into(arg, place.register, bias=adjust)
            else:
                location = e.alloc.location(arg)
                source = (location.name if isinstance(location, InRegister)
                          else f"[sp, #{e.slot_offset(arg) + adjust}]")
                moves.append((place.register, source))
        _emit_parallel_moves(e, moves)
        return adjust

    # ── one instruction ─────────────────────────────────────────────────────
    def _instruction(self, e: _Emitter, ins: Instruction, saved: list[str],
                     saved_base: int, frame: int, abi: ABI,
                     dialect: AsmDialect) -> None:
        op, ty = ins.op, ins.ty
        fn_name = e.fn.name

        if self._is_float_op(e, ins):
            self._float_instruction(e, ins, abi)
            return

        match op:
            case Op.CONST:
                dest = e.out_register(ins.dst, SCRATCH_A)
                e.materialise(int(ins.imm), dest)
                e.store(dest, ins.dst)

            case Op.COPY:
                source = e.into(ins.args[0], SCRATCH_A)
                e.store(source, ins.dst)

            case Op.GLOBAL_ADDR | Op.FUNC_ADDR:
                dest = e.out_register(ins.dst, SCRATCH_A)
                target = (self.symbol(ins.sym, dialect) if op is Op.FUNC_ADDR
                          else self.global_symbol(ins.sym, dialect))
                e.emit(f"adrp {dest}, {target}")
                e.emit(f"add {dest}, {dest}, :lo12:{target}")
                e.store(dest, ins.dst)

            case _ if op in _SIMPLE_BINOP:
                # Three-operand: no move-into-place, unlike x86-64.
                a = e.into(ins.args[0], SCRATCH_A)
                b = e.into(ins.args[1], SCRATCH_B)
                dest = e.out_register(ins.dst, SCRATCH_A)
                e.emit(f"{_SIMPLE_BINOP[op]} {dest}, {a}, {b}")
                self._narrow(e, dest, ty)
                e.store(dest, ins.dst)

            case Op.DIV:
                a = e.into(ins.args[0], SCRATCH_A)
                b = e.into(ins.args[1], SCRATCH_B)
                dest = e.out_register(ins.dst, SCRATCH_A)
                e.emit(f"{'sdiv' if ty.is_signed else 'udiv'} {dest}, {a}, {b}")
                self._narrow(e, dest, ty)
                e.store(dest, ins.dst)

            case Op.REM:
                # No remainder instruction: divide, then multiply-subtract.
                a = e.into(ins.args[0], SCRATCH_A)
                b = e.into(ins.args[1], SCRATCH_B)
                if a == SCRATCH_A and b == SCRATCH_B:
                    pass
                e.emit(f"{'sdiv' if ty.is_signed else 'udiv'} "
                       f"{SCRATCH_A}, {a}, {b}")
                dest = e.out_register(ins.dst, SCRATCH_A)
                e.emit(f"msub {dest}, {SCRATCH_A}, {b}, {a}")
                self._narrow(e, dest, ty)
                e.store(dest, ins.dst)

            case Op.NEG | Op.NOT:
                a = e.into(ins.args[0], SCRATCH_A)
                dest = e.out_register(ins.dst, SCRATCH_A)
                e.emit(f"{'neg' if op is Op.NEG else 'mvn'} {dest}, {a}")
                self._narrow(e, dest, ty)
                e.store(dest, ins.dst)

            case Op.SHL | Op.SHR:
                a = e.into(ins.args[0], SCRATCH_A)
                b = e.into(ins.args[1], SCRATCH_B)
                dest = e.out_register(ins.dst, SCRATCH_A)
                mnemonic = ("lsl" if op is Op.SHL
                            else ("asr" if ty.is_signed else "lsr"))
                e.emit(f"{mnemonic} {dest}, {a}, {b}")
                self._narrow(e, dest, ty)
                e.store(dest, ins.dst)

            case Op.EQ | Op.NE | Op.LT | Op.LE | Op.GT | Op.GE:
                a = e.into(ins.args[0], SCRATCH_A)
                b = e.into(ins.args[1], SCRATCH_B)
                dest = e.out_register(ins.dst, SCRATCH_A)
                e.emit(f"cmp {a}, {b}")
                e.emit(f"cset {dest}, {_CONDITION[(op, ty.is_signed)]}")
                e.store(dest, ins.dst)

            case Op.TRUNC | Op.EXTEND | Op.BITCAST:
                a = e.into(ins.args[0], SCRATCH_A)
                dest = e.out_register(ins.dst, SCRATCH_A)
                if dest != a:
                    e.emit(f"mov {dest}, {a}")
                self._narrow(e, dest, ty)
                e.store(dest, ins.dst)

            case Op.ALLOCA:
                dest = e.out_register(ins.dst, SCRATCH_A)
                _address_in_frame(e, dest, e.alloca_base + e.alloca_used)
                e.alloca_used += (int(ins.imm) + 15) & ~15
                e.store(dest, ins.dst)

            case Op.LOAD:
                addr = e.into(ins.args[0], SCRATCH_B)
                dest = e.out_register(ins.dst, SCRATCH_A)
                e.emit(f"{self._load_mnemonic(ty)} "
                       f"{self._sized(dest, ty)}, [{addr}]")
                e.store(dest, ins.dst)

            case Op.STORE:
                value = e.into(ins.args[0], SCRATCH_A)
                addr = e.into(ins.args[1], SCRATCH_B)
                e.emit(f"{self._store_mnemonic(ty)} "
                       f"{self._sized(value, ty)}, [{addr}]")

            case Op.OFFSET:
                base = e.into(ins.args[0], SCRATCH_A)
                delta = e.into(ins.args[1], SCRATCH_B)
                dest = e.out_register(ins.dst, SCRATCH_A)
                e.emit(f"add {dest}, {base}, {delta}")
                e.store(dest, ins.dst)

            case Op.CALL | Op.CALL_PTR:
                if op is Op.CALL_PTR:
                    # Loaded before the arguments: x17 is reserved, so the
                    # argument moves cannot clobber it.
                    target = e.into(ins.args[0], SCRATCH_B)
                    if target != SCRATCH_B:
                        e.emit(f"mov {SCRATCH_B}, {target}")
                adjust = self._place_arguments(e, ins, abi,
                                               skip_first=op is Op.CALL_PTR)
                if op is Op.CALL:
                    e.emit(f"bl {self.symbol(ins.sym, dialect)}")
                else:
                    e.emit(f"blr {SCRATCH_B}")
                if adjust:
                    _move_sp(e, "add", adjust)
                if ins.dst is not None:
                    if e.fn.register_type(ins.dst).is_float:
                        e.float_store("d0", ins.dst)
                    else:
                        e.store("x0", ins.dst)

            case Op.JUMP:
                e.emit(f"b {block_label(fn_name, ins.labels[0])}")

            case Op.BRANCH:
                cond = e.into(ins.args[0], SCRATCH_A)
                e.emit(f"cbnz {cond}, {block_label(fn_name, ins.labels[0])}")
                e.emit(f"b {block_label(fn_name, ins.labels[1])}")

            case Op.SWITCH:
                value = e.into(ins.args[0], SCRATCH_A)
                for case_value, target in ins.cases:
                    e.materialise(case_value, SCRATCH_B)
                    e.emit(f"cmp {value}, {SCRATCH_B}")
                    e.emit(f"b.eq {block_label(fn_name, target)}")
                e.emit(f"b {block_label(fn_name, ins.labels[0])}")

            case Op.RET:
                if ins.args:
                    if e.fn.register_type(ins.args[0]).is_float:
                        e.float_into(ins.args[0], "d0")
                    else:
                        source = e.into(ins.args[0], SCRATCH_A)
                        if source != "x0":
                            e.emit(f"mov x0, {source}")
                for i, reg in enumerate(saved):
                    e.emit(f"ldr {reg}, [sp, #{saved_base + 8 * i}]")
                _move_sp(e, "add", frame)
                e.emit("ldp x29, x30, [sp], #16")
                e.emit("ret")

            case Op.UNREACHABLE:
                e.emit("brk #1")

            case _:
                raise UnsupportedOperation(
                    f"{e.fn.name}: {op.value} on {ty} is not implemented by "
                    f"this backend; use --backend c")

    # ── widths ──────────────────────────────────────────────────────────────
    @staticmethod
    def _narrow(e: _Emitter, dest: str, ty: T.Type) -> None:
        """Bring a result back into `ty`'s range.

        Everything is computed at 64 bits because the registers are, and a
        narrow type has to be put back afterwards -- x86-64 skipped this and
        `i8.add 127, 1` came out 128 where the IR says -128.
        """
        if ty.is_ptr or ty.is_float or ty.bits >= 64:
            return
        if ty.bits == 1:
            e.emit(f"and {dest}, {dest}, #1")
            return
        widen = {(8, True): "sxtb", (8, False): "uxtb",
                 (16, True): "sxth", (16, False): "uxth",
                 (32, True): "sxtw", (32, False): "uxtw"}[(ty.bits,
                                                           ty.is_signed)]
        if widen == "uxtw":
            # No `uxtw` on a 64-bit destination: writing the 32-bit view
            # zeroes the top half, which is the same thing.
            e.emit(f"mov {'w' + dest[1:]}, {'w' + dest[1:]}")
        else:
            e.emit(f"{widen} {dest}, {'w' + dest[1:]}")

    @staticmethod
    def _access_width(ty: T.Type) -> int:
        return 8 if ty is T.I1 else ty.bits

    @classmethod
    def _load_mnemonic(cls, ty: T.Type) -> str:
        bits, signed = cls._access_width(ty), ty.is_signed
        return {(8, True): "ldrsb", (8, False): "ldrb",
                (16, True): "ldrsh", (16, False): "ldrh",
                (32, True): "ldrsw", (32, False): "ldr",
                (64, True): "ldr", (64, False): "ldr"}[(bits, signed)]

    @classmethod
    def _store_mnemonic(cls, ty: T.Type) -> str:
        return {8: "strb", 16: "strh", 32: "str",
                64: "str"}[cls._access_width(ty)]

    @classmethod
    def _sized(cls, reg: str, ty: T.Type) -> str:
        """The register name at the access width.

        A 32-bit load or store names the `w` view; anything narrower still
        uses `w`, because the byte and halfword forms take a 32-bit register.
        A signed 32-bit load is the exception: `ldrsw` widens into `x`.
        """
        bits = cls._access_width(ty)
        if bits == 64:
            return reg
        if bits == 32 and ty.is_signed:
            return reg
        return "w" + reg[1:]

    # ── floating point ──────────────────────────────────────────────────────
    @classmethod
    def _is_float_op(cls, e: _Emitter, ins: Instruction) -> bool:
        if ins.op not in _FLOAT_PATH:
            return False
        if ins.op in (Op.FTOI, Op.ITOF, Op.FTOF) or ins.ty.is_float:
            return True
        return bool(ins.args) and e.fn.register_type(ins.args[0]).is_float

    def _float_instruction(self, e: _Emitter, ins: Instruction,
                           abi: ABI) -> None:
        op, ty = ins.op, ins.ty
        wide = ty is T.F64
        a_reg = FSCRATCH_A if wide else "s0"
        b_reg = FSCRATCH_B if wide else "s1"

        match op:
            case Op.CONST:
                e.materialise(_float_bits(ty, ins.imm), SCRATCH_A)
                e.emit(f"fmov {a_reg}, "
                       f"{SCRATCH_A if wide else 'w' + SCRATCH_A[1:]}")
                e.float_store(FSCRATCH_A, ins.dst)

            case Op.COPY:
                e.float_into(ins.args[0], FSCRATCH_A)
                e.float_store(FSCRATCH_A, ins.dst)

            case Op.ADD | Op.SUB | Op.MUL | Op.DIV:
                e.float_into(ins.args[0], FSCRATCH_A)
                e.float_into(ins.args[1], FSCRATCH_B)
                e.emit(f"{_FLOAT_BINOP[op]} {a_reg}, {a_reg}, {b_reg}")
                e.float_store(FSCRATCH_A, ins.dst)

            case Op.REM:
                # No float remainder instruction here either; libm's fmod,
                # which `_clobbers_volatiles` reports as a call so the
                # allocator keeps live values out of the volatiles.
                e.float_into(ins.args[0], FSCRATCH_A)
                e.float_into(ins.args[1], FSCRATCH_B)
                e.emit("bl " + ("fmod" if wide else "fmodf"))
                e.float_store(FSCRATCH_A, ins.dst)

            case Op.NEG:
                # `fneg` flips the sign bit, so -0.0 stays -0.0 -- subtracting
                # from zero would not.
                e.float_into(ins.args[0], FSCRATCH_A)
                e.emit(f"fneg {a_reg}, {a_reg}")
                e.float_store(FSCRATCH_A, ins.dst)

            case Op.EQ | Op.NE | Op.LT | Op.LE | Op.GT | Op.GE:
                e.float_into(ins.args[0], FSCRATCH_A)
                e.float_into(ins.args[1], FSCRATCH_B)
                e.emit(f"fcmp {a_reg}, {b_reg}")
                dest = e.out_register(ins.dst, SCRATCH_A)
                e.emit(f"cset {dest}, {_FLOAT_CONDITION[op]}")
                e.store(dest, ins.dst)

            case Op.FTOI:
                source = e.fn.register_type(ins.args[0])
                e.float_into(ins.args[0], FSCRATCH_A)
                dest = e.out_register(ins.dst, SCRATCH_A)
                # `fcvtzs` truncates toward zero, which is what int() does.
                e.emit(f"fcvtzs {dest}, "
                       f"{FSCRATCH_A if source is T.F64 else 's0'}")
                e.store(dest, ins.dst)

            case Op.ITOF:
                source = e.into(ins.args[0], SCRATCH_A)
                e.emit(f"scvtf {a_reg}, {source}")
                e.float_store(FSCRATCH_A, ins.dst)

            case Op.FTOF:
                source = e.fn.register_type(ins.args[0])
                e.float_into(ins.args[0], FSCRATCH_A)
                if source is not ty:
                    src_reg = FSCRATCH_A if source is T.F64 else "s0"
                    e.emit(f"fcvt {a_reg}, {src_reg}")
                e.float_store(FSCRATCH_A, ins.dst)

            case Op.LOAD:
                addr = e.into(ins.args[0], SCRATCH_B)
                e.emit(f"ldr {a_reg}, [{addr}]")
                e.float_store(FSCRATCH_A, ins.dst)

            case Op.STORE:
                source = e.fn.register_type(ins.args[0])
                e.float_into(ins.args[0], FSCRATCH_A)
                addr = e.into(ins.args[1], SCRATCH_B)
                e.emit(f"str {FSCRATCH_A if source is T.F64 else 's0'}, "
                       f"[{addr}]")

            case _:
                raise UnsupportedOperation(
                    f"{e.fn.name}: {op.value} on {ty} is not implemented by "
                    f"this backend; use --backend c")


register(Arm64Backend())
