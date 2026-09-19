"""x86-64 instruction encoding: the stage that used to be `as`.

WHY THIS EXISTS. A backend that emits assembly has not finished. `as` does the
encoding and `as` writes the object, so the backend has never once decided
what a byte of its output is -- which reads as working, because the file
appears and the linker takes it, and which means the compiler cannot produce a
program without an assembler for the target installed. `backend/objfile/elf.py`
says the same thing about the file format; this is the same argument one stage
earlier, about the instructions.

WHAT IT TAKES AND WHAT IT ANSWERS. One line of the AT&T assembly the emitter
already produces, and the bytes it means, plus any relocation that line needs.
The instruction SET is closed: it is exactly what `emit.py` selects, which is
about fifty opcodes over half a dozen operand shapes. Nothing here is general;
an instruction the emitter cannot produce is not encodable here and says so
rather than guessing.

WHY IT PARSES TEXT RATHER THAN TAKING RECORDS. Because instruction SELECTION
is 1,100 lines that are known to be right, and rewriting them to build records
would put a working stage at risk to reach a stage that is not written yet.
The text is an internal convention between two halves of one backend, not an
interface anything outside sees, and it is parsed the moment it is built. The
objection the objfile docstring raises is to handing the last stage to an
EXTERNAL assembler, which is a toolchain dependency and a claim the backend
cannot check; a parser this file owns is neither.

HOW IT IS KNOWN TO BE RIGHT. Every form is compared against GNU `as`, byte for
byte, in `tests/uasm/unit/test_x86_encode.py`. That oracle is the whole
reason this is tractable: an encoder is a large table of small facts, and a
table of small facts is exactly what a differential test is good at. `as` is
needed to TEST this file and not to USE it, which is the distinction that
matters.

THE OPERAND FORMS the emitter produces, and nothing else:

    %rax  %r10d  %r10b  %cl  %xmm0     a register, at some width
    $123                                an immediate
    -8(%rbp)   16(%rsp)   (%r11)        base plus displacement
    sym(%rip)                           RIP-relative, a relocation
    *%r11                               an indirect call target
    .Lname   sym                        a branch or call target
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

# ── relocation numbers, from the psABI ──────────────────────────────────────
#
# NOT INTERPRETED HERE beyond being written down: what the linker does with
# one is its business. They live in this file rather than in the ELF writer
# because they are the ARCHITECTURE's, and a format writer that knew them
# would be the same mistake as an assembly backend that knows about ELF.
R_X86_64_PC32 = 2
R_X86_64_PLT32 = 4

#: The 64-bit general registers, in encoding order. The index IS the number
#: the instruction encoding wants, and the top bit becomes REX.B/R/X.
_GPR64 = ("rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")
_GPR32 = ("eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi",
          "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d")
_GPR16 = ("ax", "cx", "dx", "bx", "sp", "bp", "si", "di",
          "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w")
#: THE REX-ERA BYTE REGISTERS. `spl`/`bpl`/`sil`/`dil` are reachable only with
#: a REX prefix present; without one those four encodings mean `ah`/`ch`/
#: `dh`/`bh` instead. The emitter only ever names `r10b`, `r11b` and `cl`, so
#: the trap is not reachable from here -- but the table is written the REX way
#: so that it stays unreachable if another register is ever named.
_GPR8 = ("al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil",
         "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b")

_WIDTH_OF: dict[str, tuple[int, int]] = {}
for _table, _bits in ((_GPR64, 64), (_GPR32, 32), (_GPR16, 16), (_GPR8, 8)):
    for _i, _name in enumerate(_table):
        _WIDTH_OF[_name] = (_i, _bits)
for _i in range(16):
    _WIDTH_OF[f"xmm{_i}"] = (_i, 128)


class EncodeError(Exception):
    """An instruction this file cannot encode.

    RAISED, NEVER GUESSED. A wrong encoding is a program that runs and does
    something else, which is the failure this whole stage exists to be able to
    check -- so an unrecognised mnemonic or operand shape stops the compile
    and names itself.
    """


# ── operands ────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Reg:
    """A register, at the width its name implies."""

    name: str

    @property
    def num(self) -> int:
        return _WIDTH_OF[self.name][0]

    @property
    def bits(self) -> int:
        return _WIDTH_OF[self.name][1]

    @property
    def is_xmm(self) -> bool:
        return self.name.startswith("xmm")


@dataclass(frozen=True, slots=True)
class Imm:
    value: int


@dataclass(frozen=True, slots=True)
class Mem:
    """`disp(%base)`, or `sym(%rip)` when `symbol` is set."""

    base: str | None = None
    disp: int = 0
    symbol: str | None = None


@dataclass(frozen=True, slots=True)
class Label:
    """A branch or call destination, by name.

    NOT `Target`: in this tree that word means a PLATFORM, and a second
    meaning inside one backend would read as the same thing.
    """

    name: str


@dataclass(frozen=True, slots=True)
class Indirect:
    """`*%r11` -- call through a register."""

    reg: str


Operand = Reg | Imm | Mem | Label | Indirect

_MEM = re.compile(r"^(-?\d+)?\((%\w+)\)$")
_RIP = re.compile(r"^([A-Za-z_.$][\w.$@]*)\(%rip\)$")


def parse_operand(text: str) -> Operand:
    """One operand, in the AT&T spelling the emitter writes."""
    text = text.strip()
    if text.startswith("*%"):
        return Indirect(text[2:])
    if text.startswith("%"):
        name = text[1:]
        if name not in _WIDTH_OF:
            raise EncodeError(f"unknown register {text!r}")
        return Reg(name)
    if text.startswith("$"):
        return Imm(int(text[1:], 0))
    rip = _RIP.match(text)
    if rip is not None:
        return Mem(base="rip", symbol=rip.group(1))
    mem = _MEM.match(text)
    if mem is not None:
        disp = int(mem.group(1)) if mem.group(1) else 0
        return Mem(base=mem.group(2)[1:], disp=disp)
    return Label(text)


def _split_operands(rest: str) -> list[str]:
    """Split on commas that are not inside parentheses.

    The emitter never writes a nested form, so a depth counter is more than
    is needed -- and it is what keeps `movq -8(%rbp), %rax` from splitting in
    the middle of its memory operand if one ever appears.
    """
    out, depth, at = [], 0, 0
    for i, ch in enumerate(rest):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(rest[at:i])
            at = i + 1
    if rest[at:].strip():
        out.append(rest[at:])
    return [one.strip() for one in out]


# ── byte-level helpers ──────────────────────────────────────────────────────

def _rex(w: int, reg: int, index: int, base: int, *, force: bool = False) -> bytes:
    """The REX prefix, or nothing when every bit of it would be zero.

    `force` is for the byte registers: `movb %r10b, (%rax)` needs no extension
    bits, and an 8-bit operation naming `spl`/`bpl`/`sil`/`dil` still needs the
    prefix PRESENT to mean those rather than `ah`/`ch`/`dh`/`bh`.
    """
    value = 0x40 | (w << 3) | ((reg >> 3) << 2) | ((index >> 3) << 1) | (base >> 3)
    return bytes([value]) if force or value != 0x40 else b""


def _imm(value: int, bits: int) -> bytes:
    """`value` as a little-endian immediate of `bits` width.

    MASKED RATHER THAN RANGE-CHECKED, because both spellings of a bit pattern
    reach here. The IR carries `~0` as 18446744073709551615 and `-1` as -1,
    and they are the same eight bytes; refusing the first would reject a
    constant the runtime uses on nearly every line, and `as` takes it too.
    """
    width = {8: "<B", 16: "<H", 32: "<I", 64: "<Q"}[bits]
    return struct.pack(width, value & ((1 << bits) - 1))


def _signed(value: int, bits: int) -> int:
    """`value` as the signed number those `bits` hold.

    EVERY RANGE DECISION IN THE FILE GOES THROUGH HERE FIRST, because the two
    spellings of a bit pattern must reach the same instruction. `andq $-1` and
    `andq $18446744073709551615` are the same operation, and a range test on
    the raw value sends them down different arms -- the second to a form that
    does not exist, so the encoder either refuses valid assembly or writes a
    different number. Normalising makes the pattern, not the spelling, decide.
    """
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def _fits32(value: int) -> bool:
    """Whether a 64-bit operation can carry this as its sign-extended imm32.

    x86-64 has no 64-bit immediate outside `movabs`: every other instruction
    writes 32 bits and the processor sign-extends them. So a value needing bit
    32 or above is not encodable in that form at all, and `as` says as much
    rather than truncating.
    """
    return -2147483648 <= value <= 2147483647


def _modrm(mod: int, reg: int, rm: int) -> int:
    return (mod << 6) | ((reg & 7) << 3) | (rm & 7)


def _mem_operand(reg: int, mem: Mem) -> tuple[bytes, int, str | None]:
    """ModRM (plus SIB and displacement) for a memory operand.

    Answers the bytes, the offset of the displacement WITHIN them, and the
    symbol the displacement must be relocated against, if any. The caller adds
    the instruction's own prefix and opcode, so the displacement's position in
    the finished instruction is that prefix length plus this offset.

    THE TWO ENCODING TRAPS, both reachable from the emitter's own output:

      * `rsp` AND `r12` CANNOT BE A BARE BASE. Their `rm` value means "a SIB
        byte follows", so `(%rsp)` needs SIB 0x24 -- base rsp, no index. The
        emitter writes `16(%rsp)` for stack arguments, so this is not a
        theoretical case.
      * `rbp` AND `r13` CANNOT USE mod=00. That combination means
        RIP-relative, so a zero displacement off `rbp` must be written as
        mod=01 with an explicit zero byte. Every spilled value is `-N(%rbp)`,
        and a slot at offset zero would silently become a RIP reference.
    """
    if mem.base == "rip":
        # mod=00, rm=101: the displacement is relative to the END of the
        # instruction, which is what the relocation's addend accounts for.
        return bytes([_modrm(0, reg, 5)]) + b"\0\0\0\0", 1, mem.symbol
    if mem.base is None:
        raise EncodeError("memory operand with no base")
    base = _WIDTH_OF[mem.base][0]
    needs_sib = (base & 7) == 4
    if mem.disp == 0 and (base & 7) != 5:
        mod, disp = 0, b""
    elif -128 <= mem.disp <= 127:
        mod, disp = 1, struct.pack("<b", mem.disp)
    else:
        mod, disp = 2, struct.pack("<i", mem.disp)
    out = bytes([_modrm(mod, reg, base)])
    if needs_sib:
        out += bytes([0x24])            # scale 1, no index, base as given
    return out + disp, len(out), None


def _needs_rex8(reg: Reg | Mem | int | None) -> bool:
    """Does naming this BYTE register require a REX prefix to be present?

    Only four of them: `spl`, `bpl`, `sil` and `dil` occupy encodings 4 to 7,
    which without REX mean `ah`, `ch`, `dh` and `bh` instead. `al` through
    `bl` need nothing, and `r8b` upwards set REX.B anyway, so the prefix is
    already there.

    FORCING IT FOR EVERY BYTE REGISTER is the obvious version and is wrong in
    a way that runs: `movb %cl, (%rax)` came out three bytes where `as` makes
    it two, both of which execute correctly. Only the differential test could
    tell, which is what it is for.
    """
    return isinstance(reg, Reg) and reg.bits == 8 and 4 <= reg.num <= 7


def _rm(reg: int, target: Reg | Mem) -> tuple[bytes, int, str | None]:
    """ModRM for either a register or a memory r/m operand."""
    if isinstance(target, Reg):
        return bytes([_modrm(3, reg, target.num)]), 0, None
    return _mem_operand(reg, target)


def _rm_here(reg: int, target: Reg | Mem, mnemonic: str) -> tuple[bytes, int]:
    """`_rm` for the forms that have nowhere to put a relocation.

    SIXTEEN INSTRUCTION FORMS IN THIS FILE take an r/m operand and ignore the
    symbol `_rm` hands back -- `movq $1, sym(%rip)`, `shlq $3, sym(%rip)`,
    `negq sym(%rip)`, `testq %rax, sym(%rip)` and the rest. Each of them
    emitted a zero displacement and NO RELOCATION, which assembles, links and
    disassembles without complaint and reads whatever is four bytes past the
    instruction.

    NONE OF THEM IS REACHABLE TODAY. `emit.py` writes exactly one
    RIP-relative operand -- the `leaq` that takes a global's address -- and
    every read-modify-write goes through a register after it. So this is a
    refusal and not a fix: recording the bias correctly is a per-form job
    (the displacement is measured from the END of the instruction, so a
    trailing `imm32` makes it -8 rather than -4, and Mach-O needs
    `X86_64_RELOC_SIGNED_4` rather than `SIGNED` to say so), and threading
    -4 through all sixteen would silently store four bytes off the global
    rather than loudly declining.
    """
    rm, disp_at, symbol = _rm(reg, target)
    if symbol is not None:
        raise EncodeError(
            f"{mnemonic} cannot carry a relocation for {symbol!r}: this form "
            f"has no place to record one")
    return rm, disp_at


def _rex_for(w: int, reg: Reg | int, target: Reg | Mem | None,
             *, force: bool = False) -> bytes:
    r = reg.num if isinstance(reg, Reg) else reg
    if isinstance(target, Reg):
        return _rex(w, r, 0, target.num, force=force)
    if isinstance(target, Mem) and target.base not in (None, "rip"):
        return _rex(w, r, 0, _WIDTH_OF[target.base][0], force=force)
    return _rex(w, r, 0, 0, force=force)


# ── the instruction tables ──────────────────────────────────────────────────

#: `op r/m, r` and `op r, r/m` for the ALU instructions that have both forms.
#: The first opcode stores INTO the r/m operand, the second reads from it --
#: which is the whole of the difference, and the reason AT&T's operand order
#: has to be read carefully: `addq %rax, %rbx` adds rax INTO rbx.
_ALU = {
    "add": (0x01, 0x03, 0),
    "or": (0x09, 0x0B, 1),
    "and": (0x21, 0x23, 4),
    "sub": (0x29, 0x2B, 5),
    "xor": (0x31, 0x33, 6),
    "cmp": (0x39, 0x3B, 7),
}

#: The condition codes `setcc` and `jcc` share. The number is the low nibble
#: of the opcode in both families.
_CC = {
    "o": 0, "no": 1, "b": 2, "ae": 3, "e": 4, "ne": 5, "be": 6, "a": 7,
    "s": 8, "ns": 9, "p": 10, "np": 11, "l": 12, "ge": 13, "le": 14, "g": 15,
}

#: SSE arithmetic: the mandatory prefix and the two-byte opcode's second byte.
_SSE_ARITH = {"add": 0x58, "mul": 0x59, "sub": 0x5C, "div": 0x5E}

#: The widening moves, by mnemonic: opcode bytes and whether REX.W is set.
_MOVX = {
    "movzbq": (b"\x0f\xb6", 1), "movzwq": (b"\x0f\xb7", 1),
    "movsbq": (b"\x0f\xbe", 1), "movswq": (b"\x0f\xbf", 1),
    "movslq": (b"\x63", 1),
    "movzbl": (b"\x0f\xb6", 0), "movzwl": (b"\x0f\xb7", 0),
}

#: Width by mnemonic suffix, for the plain `mov` family.
_SUFFIX_BITS = {"b": 8, "w": 16, "l": 32, "q": 64}


@dataclass(slots=True)
class Fixup:
    """A branch whose target is a label in this same function.

    Resolved after every instruction's length is known -- which is why the
    encoder runs twice over a function. Nothing here is variable-length by
    choice: every jump is written as a 32-bit displacement, so the second
    pass changes displacements and never lengths, and one extra pass is
    enough. A short-form pass would save three bytes per jump and would make
    lengths depend on distances that depend on lengths.
    """

    at: int
    label: str
    end: int


@dataclass(slots=True)
class Encoded:
    """One function's worth of bytes, with what still has to be patched."""

    code: bytearray = field(default_factory=bytearray)
    relocs: list[tuple[int, str, int, int]] = field(default_factory=list)
    fixups: list[Fixup] = field(default_factory=list)
    labels: dict[str, int] = field(default_factory=dict)


def encode_line(text: str, out: Encoded) -> None:
    """Encode one assembly line, appending to `out`.

    A LABEL IS A LINE TOO. The emitter writes `name:` for every block, and the
    branch that reaches it is resolved from `out.labels` once the function is
    complete.
    """
    text = text.strip()
    if not text or text.startswith("#"):
        return
    if text.endswith(":"):
        out.labels[text[:-1]] = len(out.code)
        return
    parts = text.split(None, 1)
    mnemonic = parts[0]
    operands = [parse_operand(one) for one in _split_operands(parts[1])] \
        if len(parts) > 1 else []
    out.code += _encode(mnemonic, operands, out, len(out.code))


def _reloc(out: Encoded, at: int, symbol: str, kind: int, addend: int) -> None:
    out.relocs.append((at, symbol, kind, addend))


def _encode(mnemonic: str, ops: list[Operand], out: Encoded,
            base: int) -> bytes:
    """The bytes for one instruction. See the module docstring for the set."""
    # ── no operands ─────────────────────────────────────────────────────────
    if mnemonic == "ret":
        return b"\xc3"
    if mnemonic == "ud2":
        return b"\x0f\x0b"
    if mnemonic == "cqto":
        return b"\x48\x99"
    if mnemonic == "cltd":
        return b"\x99"

    # ── control flow ────────────────────────────────────────────────────────
    if mnemonic in ("jmp", "je", "jne", "call") and ops and isinstance(
            ops[0], Label):
        name = ops[0].name
        if mnemonic == "call":
            body = b"\xe8" + b"\0\0\0\0"
        elif mnemonic == "jmp":
            body = b"\xe9" + b"\0\0\0\0"
        else:
            body = bytes([0x0F, 0x80 | _CC[mnemonic[1:]]]) + b"\0\0\0\0"
        at = base + len(body) - 4
        if name.startswith(".L"):
            out.fixups.append(Fixup(at=at, label=name, end=base + len(body)))
        else:
            # THE ADDEND IS -4 because the displacement is measured from the
            # END of the instruction and the relocation is applied at its
            # start. PLT32 rather than PC32: a call to a symbol the linker may
            # route through a stub is what a call to an external function is,
            # and PC32 refuses that routing.
            _reloc(out, at, name, R_X86_64_PLT32, -4)
        return body
    if mnemonic == "call" and ops and isinstance(ops[0], Indirect):
        reg = Reg(ops[0].reg)
        return _rex(0, 0, 0, reg.num) + b"\xff" + bytes([_modrm(3, 2, reg.num)])

    # ── stack ───────────────────────────────────────────────────────────────
    if mnemonic in ("pushq", "popq"):
        reg = ops[0]
        assert isinstance(reg, Reg)
        prefix = b"\x41" if reg.num >= 8 else b""
        return prefix + bytes([(0x50 if mnemonic == "pushq" else 0x58)
                               + (reg.num & 7)])

    # ── moves ───────────────────────────────────────────────────────────────
    if mnemonic == "movabsq":
        src, dst = ops
        assert isinstance(src, Imm) and isinstance(dst, Reg)
        return _rex(1, 0, 0, dst.num) + bytes([0xB8 + (dst.num & 7)]) \
            + _imm(src.value, 64)
    if mnemonic in _MOVX:
        opcode, w = _MOVX[mnemonic]
        src, dst = ops
        assert isinstance(dst, Reg)
        assert isinstance(src, (Reg, Mem))
        rm, _ = _rm_here(dst.num, src, mnemonic)
        return _rex_for(w, dst, src, force=_needs_rex8(src)) + opcode + rm
    if mnemonic in ("movq", "movl", "movw", "movb", "movd"):
        # AN SSE REGISTER ON EITHER SIDE MAKES THIS A DIFFERENT INSTRUCTION,
        # and the mnemonic does not say so -- see `_sse`. Asked first, because
        # the general encoding below would take `%xmm3` for `%rbx`.
        got = _sse(mnemonic, ops, out, base)
        if got is not None:
            return got
        bits = _SUFFIX_BITS[mnemonic[-1]]
        return _mov(bits, ops, out, base)

    # ── lea ─────────────────────────────────────────────────────────────────
    if mnemonic == "leaq":
        src, dst = ops
        assert isinstance(dst, Reg) and isinstance(src, Mem)
        rm, disp_at, sym = _mem_operand(dst.num, src)
        head = _rex_for(1, dst, src) + b"\x8d"
        if sym is not None:
            _reloc(out, base + len(head) + disp_at, sym, R_X86_64_PC32, -4)
        return head + rm

    # ── ALU ─────────────────────────────────────────────────────────────────
    stem, bits = _stem(mnemonic)
    if stem in _ALU:
        return _alu(stem, bits, ops, out, base)
    if stem in ("neg", "not", "idiv", "div", "imul", "mul"):
        return _unary(stem, bits, ops)
    if stem == "test":
        src, dst = ops
        assert isinstance(src, Reg) and isinstance(dst, (Reg, Mem))
        rm, _ = _rm_here(src.num, dst, mnemonic)
        return _rex_for(1 if bits == 64 else 0, src, dst) + b"\x85" + rm
    if stem in ("shl", "sar", "shr", "sal"):
        return _shift(stem, bits, ops)

    # ── setcc ───────────────────────────────────────────────────────────────
    if mnemonic.startswith("set") and mnemonic[3:] in _CC:
        dst = ops[0]
        assert isinstance(dst, (Reg, Mem))
        rm, _ = _rm_here(0, dst, mnemonic)
        return _rex_for(0, 0, dst, force=_needs_rex8(dst)) \
            + bytes([0x0F, 0x90 | _CC[mnemonic[3:]]]) + rm

    # ── SSE ─────────────────────────────────────────────────────────────────
    got = _sse(mnemonic, ops, out, base)
    if got is not None:
        return got
    raise EncodeError(f"cannot encode {mnemonic!r}")


def _stem(mnemonic: str) -> tuple[str, int]:
    """A mnemonic split into its opcode stem and its operand width.

    `addq` is `add` at 64 bits. A mnemonic with no width suffix keeps its
    whole name and answers 64, which is what the emitter means everywhere it
    writes one.
    """
    if len(mnemonic) > 2 and mnemonic[-1] in _SUFFIX_BITS:
        return mnemonic[:-1], _SUFFIX_BITS[mnemonic[-1]]
    return mnemonic, 64


def _prefix_for(bits: int) -> bytes:
    """The operand-size override, which only 16-bit operations need."""
    return b"\x66" if bits == 16 else b""


def _mov(bits: int, ops: list[Operand], out: Encoded, base: int) -> bytes:
    src, dst = ops
    # NOTHING SSE REACHES HERE. `_encode` sends every form naming an `%xmm` to
    # `_sse` first, so one arriving means a shape neither of them handles --
    # and the general encoding would silently write the like-numbered general
    # register instead. A refusal is a build that stops; the alternative is a
    # program that runs and computes with the wrong register.
    for operand in ops:
        if isinstance(operand, Reg) and operand.is_xmm:
            raise EncodeError(
                f"cannot encode a {bits}-bit move naming {operand.name!r}: "
                f"no SSE form matched these operands")
    w = 1 if bits == 64 else 0
    if isinstance(src, Imm):
        assert isinstance(dst, (Reg, Mem))
        value = _signed(src.value, bits)
        if isinstance(dst, Reg) and bits != 64:
            # THE SHORT FORM, `B0+r` and `B8+r`, which carries the register in
            # the opcode byte and needs no ModRM. It is the one `as` picks for
            # a narrow register destination, so an encoder using C7 here would
            # be a byte longer and differ from binutils on every constant.
            return _prefix_for(bits) \
                + _rex_for(w, 0, dst, force=_needs_rex8(dst)) \
                + bytes([(0xB0 if bits == 8 else 0xB8) + (dst.num & 7)]) \
                + _imm(value, bits)
        if bits == 64 and isinstance(dst, Reg) and not _fits32(value):
            # `movq $imm64, %reg` IS `movabsq`, and `as` promotes it silently
            # -- so an encoder that refused would reject assembly binutils
            # accepts, and one that truncated would put a different number in
            # the register. Only a pattern that will not survive sign
            # extension needs the ten-byte instruction.
            return _rex(1, 0, 0, dst.num) + bytes([0xB8 + (dst.num & 7)]) \
                + _imm(value, 64)
        if bits == 64 and not _fits32(value):
            # A MEMORY DESTINATION HAS NO WIDER FORM: `movabs` only ever names
            # a register. Truncating would store a different quadword, so this
            # is the emitter's to fix by loading the constant first.
            raise EncodeError(
                f"no 64-bit store of {src.value:#x}: it does not fit a "
                f"sign-extended imm32 and movabs cannot address memory")
        rm, disp_at = _rm_here(0, dst, "a mov of an immediate")
        head = _prefix_for(bits) + _rex_for(w, 0, dst) \
            + bytes([0xC6 if bits == 8 else 0xC7])
        return head + rm + _imm(value, min(bits, 32))
    if isinstance(src, Reg) and isinstance(dst, (Reg, Mem)):
        rm, disp_at, sym = _rm(src.num, dst)
        head = _prefix_for(bits) \
            + _rex_for(w, src, dst, force=_needs_rex8(src) or _needs_rex8(dst)) \
            + bytes([0x88 if bits == 8 else 0x89])
        if sym is not None:
            _reloc(out, base + len(head) + disp_at, sym, R_X86_64_PC32, -4)
        return head + rm
    if isinstance(src, Mem) and isinstance(dst, Reg):
        rm, disp_at, sym = _rm(dst.num, src)
        head = _prefix_for(bits) \
            + _rex_for(w, dst, src, force=_needs_rex8(dst) or _needs_rex8(src)) \
            + bytes([0x8A if bits == 8 else 0x8B])
        if sym is not None:
            _reloc(out, base + len(head) + disp_at, sym, R_X86_64_PC32, -4)
        return head + rm
    raise EncodeError(f"cannot encode mov with {src!r}, {dst!r}")


def _alu(stem: str, bits: int, ops: list[Operand], out: Encoded,
         base: int) -> bytes:
    to_rm, from_rm, ext = _ALU[stem]
    src, dst = ops
    w = 1 if bits == 64 else 0
    if isinstance(src, Imm):
        assert isinstance(dst, (Reg, Mem))
        rm, _ = _rm_here(ext, dst, stem)
        value = _signed(src.value, bits)
        # THE SHORT FORM when the immediate fits in a signed byte, which is
        # most of them: `addq $8, %rsp` is four bytes rather than seven, and
        # `as` picks it too -- so a differential test would fail without it
        # even though both encodings run.
        if -128 <= value <= 127 and bits != 8:
            return _prefix_for(bits) + _rex_for(w, ext, dst) \
                + b"\x83" + rm + _imm(value, 8)
        if bits == 64 and not _fits32(value):
            # NO ALU INSTRUCTION TAKES A 64-BIT IMMEDIATE. Writing the low 32
            # bits would compute against a sign-extended number the source
            # never mentioned -- `andq $0xffffffff` would mask nothing at all
            # rather than the low word. The emitter must materialise it.
            raise EncodeError(
                f"no 64-bit {stem} of {src.value:#x}: the immediate does not "
                f"fit a sign-extended imm32")
        return _prefix_for(bits) + _rex_for(w, ext, dst) \
            + bytes([0x80 if bits == 8 else 0x81]) + rm \
            + _imm(value, min(bits, 32))
    if isinstance(src, Reg) and isinstance(dst, (Reg, Mem)):
        rm, disp_at, sym = _rm(src.num, dst)
        head = _prefix_for(bits) \
            + _rex_for(w, src, dst, force=_needs_rex8(src) or _needs_rex8(dst)) \
            + bytes([to_rm - (1 if bits == 8 else 0)])
        if sym is not None:
            _reloc(out, base + len(head) + disp_at, sym, R_X86_64_PC32, -4)
        return head + rm
    if isinstance(src, Mem) and isinstance(dst, Reg):
        rm, disp_at, sym = _rm(dst.num, src)
        head = _prefix_for(bits) + _rex_for(w, dst, src) \
            + bytes([from_rm - (1 if bits == 8 else 0)])
        if sym is not None:
            _reloc(out, base + len(head) + disp_at, sym, R_X86_64_PC32, -4)
        return head + rm
    raise EncodeError(f"cannot encode {stem} with {src!r}, {dst!r}")


#: `/digit` for the F7 group, and for `imul`'s one-operand form.
_UNARY_EXT = {"not": 2, "neg": 3, "mul": 4, "imul": 5, "div": 6, "idiv": 7}


def _unary(stem: str, bits: int, ops: list[Operand]) -> bytes:
    if stem == "imul" and len(ops) == 2:
        # THE TWO-OPERAND FORM is a different opcode entirely: `0F AF` reads
        # r/m and writes the register, where the F7 group writes rdx:rax.
        src, dst = ops
        assert isinstance(dst, Reg) and isinstance(src, (Reg, Mem))
        rm, _ = _rm_here(dst.num, src, stem)
        return _rex_for(1 if bits == 64 else 0, dst, src) + b"\x0f\xaf" + rm
    dst = ops[0]
    assert isinstance(dst, (Reg, Mem))
    rm, _ = _rm_here(_UNARY_EXT[stem], dst, stem)
    return _prefix_for(bits) + _rex_for(1 if bits == 64 else 0,
                                        _UNARY_EXT[stem], dst) \
        + bytes([0xF6 if bits == 8 else 0xF7]) + rm


_SHIFT_EXT = {"shl": 4, "sal": 4, "shr": 5, "sar": 7}


def _shift(stem: str, bits: int, ops: list[Operand]) -> bytes:
    amount, dst = ops
    assert isinstance(dst, (Reg, Mem))
    ext = _SHIFT_EXT[stem]
    rm, _ = _rm_here(ext, dst, stem)
    w = 1 if bits == 64 else 0
    if isinstance(amount, Reg) and amount.name == "cl":
        return _prefix_for(bits) + _rex_for(w, ext, dst) \
            + bytes([0xD2 if bits == 8 else 0xD3]) + rm
    if isinstance(amount, Imm):
        if amount.value == 1:
            return _prefix_for(bits) + _rex_for(w, ext, dst) \
                + bytes([0xD0 if bits == 8 else 0xD1]) + rm
        return _prefix_for(bits) + _rex_for(w, ext, dst) \
            + bytes([0xC0 if bits == 8 else 0xC1]) + rm \
            + _imm(amount.value, 8)
    raise EncodeError(f"cannot encode {stem} by {amount!r}")


def _sse(mnemonic: str, ops: list[Operand], out: Encoded,
         base: int) -> bytes | None:
    """The SSE subset the emitter uses. None when this is not one of them."""

    def two(prefix: bytes, opcode: bytes, reg: Reg,
            target: Reg | Mem, w: int = 0) -> bytes:
        rm, disp_at, sym = _rm(reg.num, target)
        head = prefix + _rex_for(w, reg, target) + opcode
        if sym is not None:
            _reloc(out, base + len(head) + disp_at, sym, R_X86_64_PC32, -4)
        return head + rm

    # movsd / movss: 10 loads INTO the register, 11 stores FROM it.
    if mnemonic in ("movsd", "movss"):
        prefix = b"\xf2" if mnemonic == "movsd" else b"\xf3"
        src, dst = ops
        if isinstance(dst, Reg) and dst.is_xmm:
            assert isinstance(src, (Reg, Mem))
            return two(prefix, b"\x0f\x10", dst, src)
        assert isinstance(src, Reg) and isinstance(dst, (Reg, Mem))
        return two(prefix, b"\x0f\x11", src, dst)

    for stem, opcode in _SSE_ARITH.items():
        for suffix, prefix in (("sd", b"\xf2"), ("ss", b"\xf3")):
            if mnemonic == stem + suffix:
                src, dst = ops
                assert isinstance(dst, Reg) and isinstance(src, (Reg, Mem))
                return two(prefix, bytes([0x0F, opcode]), dst, src)

    if mnemonic in ("ucomisd", "ucomiss"):
        prefix = b"\x66" if mnemonic == "ucomisd" else b""
        src, dst = ops
        assert isinstance(dst, Reg) and isinstance(src, (Reg, Mem))
        return two(prefix, b"\x0f\x2e", dst, src)

    if mnemonic in ("xorpd", "xorps"):
        prefix = b"\x66" if mnemonic == "xorpd" else b""
        src, dst = ops
        assert isinstance(dst, Reg) and isinstance(src, (Reg, Mem))
        return two(prefix, b"\x0f\x57", dst, src)

    if mnemonic in ("cvtsi2sdq", "cvtsi2ssq", "cvtsi2sd", "cvtsi2ss"):
        prefix = b"\xf2" if "sd" in mnemonic else b"\xf3"
        src, dst = ops
        assert isinstance(dst, Reg) and isinstance(src, (Reg, Mem))
        w = 1 if mnemonic.endswith("q") else 0
        return two(prefix, b"\x0f\x2a", dst, src, w)

    if mnemonic in ("cvttsd2si", "cvttss2si"):
        prefix = b"\xf2" if "sd" in mnemonic else b"\xf3"
        src, dst = ops
        assert isinstance(dst, Reg) and isinstance(src, (Reg, Mem))
        return two(prefix, b"\x0f\x2c", dst, src, 1 if dst.bits == 64 else 0)

    if mnemonic in ("cvtss2sd", "cvtsd2ss"):
        prefix = b"\xf3" if mnemonic == "cvtss2sd" else b"\xf2"
        src, dst = ops
        assert isinstance(dst, Reg) and isinstance(src, (Reg, Mem))
        return two(prefix, b"\x0f\x5a", dst, src)

    # ── `movd` and `movq` naming an SSE register ────────────────────────────
    #
    # THE MNEMONIC DOES NOT SAY WHICH INSTRUCTION THIS IS: `movq` is a general
    # 64-bit move, a general-to-SSE move, an SSE-to-SSE move and an SSE load,
    # and only the OPERANDS tell them apart. That is why this is reached from
    # `_mov` rather than by name -- a `movq` whose operands include an `%xmm`
    # was, until this existed, encoded as a general move with the SSE
    # register's NUMBER, so `movq %r10, %xmm0` wrote `%rax`. It assembled, it
    # linked, it ran, and every float came out wrong.
    if mnemonic in ("movd", "movq"):
        src, dst = ops
        w = 1 if mnemonic == "movq" else 0
        src_xmm = isinstance(src, Reg) and src.is_xmm
        dst_xmm = isinstance(dst, Reg) and dst.is_xmm
        if src_xmm and dst_xmm:
            # SSE TO SSE, which has its own opcode: F3 0F 7E, and the register
            # being written is the ModRM `reg` field even though the general
            # form writes `rm`.
            return two(b"\xf3", b"\x0f\x7e", dst, src)
        if dst_xmm and isinstance(src, Mem):
            return two(b"\xf3", b"\x0f\x7e", dst, src)
        if src_xmm and isinstance(dst, Mem):
            return two(b"\x66", b"\x0f\xd6", src, dst)
        if dst_xmm:
            assert isinstance(src, Reg)
            return two(b"\x66", b"\x0f\x6e", dst, src, w)
        if src_xmm:
            assert isinstance(dst, Reg)
            return two(b"\x66", b"\x0f\x7e", src, dst, w)
        return None                     # neither side is SSE: a general move
    return None


def resolve(out: Encoded) -> None:
    """Patch every branch to a label in this function.

    Run once the whole function is encoded, because a forward branch cannot
    know its target's offset until then. Every displacement is 32 bits, so
    patching changes no lengths and one pass is enough.
    """
    for fix in out.fixups:
        if fix.label not in out.labels:
            raise EncodeError(f"branch to unknown label {fix.label!r}")
        rel = out.labels[fix.label] - fix.end
        out.code[fix.at:fix.at + 4] = struct.pack("<i", rel)


def encode_function(lines: list[str]) -> Encoded:
    """Every line of one function, encoded, with its branches resolved."""
    out = Encoded()
    for line in lines:
        encode_line(line, out)
    resolve(out)
    return out
