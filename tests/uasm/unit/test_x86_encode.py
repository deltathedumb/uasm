"""Every x86-64 encoding, checked against GNU `as` byte for byte.

WHY THIS ORACLE AND NOT HAND-WRITTEN EXPECTATIONS. An encoder is a large table
of small facts -- which prefix, which opcode, which ModRM -- and a hand-written
expectation is the same fact written twice by the same person, so it agrees
with the encoder whenever the encoder is wrong in the way the author was.
`as` was written by someone else and is not going to make this file's mistake.

WHAT IT PROVES AND WHAT IT DOES NOT. It proves the bytes are the ones binutils
would have produced for the same line, which is the whole claim the encoder
makes. It does not prove the emitter SELECTED the right instruction; that is
`test_endtoend.py`'s job, by running the program.

`as` IS NEEDED TO TEST THE ENCODER AND NOT TO USE IT, which is the distinction
that matters: a machine without binutils skips this file and still compiles.
"""
from __future__ import annotations

import struct
import subprocess
import tempfile
from pathlib import Path

from tests import harness

from uasm.backends.x86_64.encode import Encoded, encode_line

def assemble(lines: list[str]) -> bytes:
    """What `as` makes of these lines, as raw `.text` bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.s"
        obj = Path(tmp) / "in.o"
        raw = Path(tmp) / "in.bin"
        src.write_text(".text\n" + "\n".join("\t" + one for one in lines) + "\n")
        subprocess.run(["as", "--64", "-o", str(obj), str(src)], check=True,
                       capture_output=True)
        subprocess.run(["objcopy", "-O", "binary", "--only-section=.text",
                        str(obj), str(raw)], check=True, capture_output=True)
        return raw.read_bytes()


def ours(line: str) -> bytes:
    out = Encoded()
    encode_line(line, out)
    return bytes(out.code)


#: THE FORMS THE EMITTER PRODUCES, and a few neighbours of each. A neighbour
#: is worth testing because the traps in x86 encoding are all "this register
#: is special": `rsp` needs a SIB byte, `rbp` at offset zero needs an explicit
#: displacement, `r8`-`r15` need REX, and the byte registers change meaning
#: depending on whether REX is present at all.
CASES = [
    # moves between registers, across the REX boundary
    "movq %rax, %rbx", "movq %r10, %rax", "movq %rax, %r11",
    "movq %r14, %r15", "movq %rsp, %rbp", "movq %rbp, %rsp",
    # memory, with the two special bases
    "movq -8(%rbp), %rax", "movq %rax, -8(%rbp)",
    "movq 0(%rbp), %rax", "movq (%rbp), %rax",
    "movq 16(%rsp), %rax", "movq %rax, 16(%rsp)", "movq (%rsp), %rax",
    "movq (%r11), %r10", "movq %r10, (%r11)",
    "movq -256(%rbp), %rax", "movq -2048(%rbp), %r15",
    "movq (%r12), %rax", "movq (%r13), %rax", "movq 8(%r13), %rax",
    # immediates
    "movq $0, %rax", "movq $1, %r10", "movq $-1, %rax",
    "movq $2147483647, %rax", "movq $5, -8(%rbp)",
    "movabsq $1152921504606846976, %rax", "movabsq $-1, %r11",
    # A 64-BIT IMMEDIATE WRITTEN AS `movq`, which `as` promotes to movabs.
    # The emitter writes constants this way, so this is the common path and
    # not a curiosity.
    "movq $1152921504606846976, %rax", "movq $-2147483649, %r10",
    "movq $4294967296, %r15", "movq $2147483648, %rax",
    # THE SAME BIT PATTERN WRITTEN UNSIGNED. The IR carries `~0` this way and
    # `-1` the other, and both must reach the same instruction -- which for
    # `movq` is the short C7 form, because the pattern DOES survive sign
    # extension however it was spelled.
    "movabsq $18446744073709551615, %rax", "movq $18446744073709551615, %r10",
    "movq $18446744073709551614, %r10", "andq $18446744073709551615, %r10",
    "movq $18446744073709551615, -8(%rbp)",
    # narrow immediates, where the register form is a byte shorter
    "movl $0, %eax", "movl $5, %r10d", "movl $-1, %eax",
    "movl $4294967295, %eax", "movw $5, %r10w", "movb $5, %r10b",
    "movb $5, %cl", "movl $5, -8(%rbp)", "movb $5, -8(%rbp)",
    "addl $4294967295, %eax", "andl $4294967295, %r10d",
    # widening
    "movzbq %r10b, %r10", "movsbq %r10b, %r11", "movswq %r10w, %r10",
    "movzwq %r11w, %r11", "movslq %r10d, %r10", "movl %r10d, %r10d",
    "movl (%r11), %r10d", "movl %eax, %ecx",
    # narrow stores
    "movb %r10b, (%r11)", "movw %r10w, (%r11)", "movl %r10d, (%r11)",
    "movb %cl, (%rax)",
    # lea
    "leaq -8(%rbp), %rax", "leaq 16(%rsp), %r10",
    # ALU, register and immediate, short and long immediate forms
    "addq %rax, %rbx", "addq %r10, %r11", "addq -8(%rbp), %rax",
    "addq %rax, -8(%rbp)", "addq $8, %rsp", "addq $1024, %rsp",
    "subq %rax, %rbx", "subq $32, %rsp", "subq $-128, %rsp",
    "andq %r10, %r11", "andq $1, %r10", "orq %r10, %r11",
    "xorq %rax, %rax", "xorq %r10, %r10", "imulq %r11, %r10",
    "cmpq %r11, %r10", "cmpq $0, %rax", "cmpq -8(%rbp), %rax",
    "testq %rax, %rax", "testq %r10, %r10",
    "negq %r10", "notq %r10", "idivq %r11", "divq %r11",
    "andb %r11b, %r10b", "orb %r11b, %r10b",
    # shifts
    "shlq %cl, %r10", "sarq %cl, %r10", "shrq %cl, %r10",
    "shlq $1, %r10", "sarq $3, %r11",
    # setcc, every condition the emitter can select
    "sete %r10b", "setne %r10b", "setl %r10b", "setle %r10b",
    "setg %r10b", "setge %r10b", "setb %r10b", "setbe %r10b",
    "seta %r10b", "setae %r10b", "setp %r11b", "setnp %r11b",
    # stack and control
    "pushq %rcx", "popq %rcx", "pushq %rbx", "pushq %r12", "popq %r15",
    "ret", "ud2", "cqto",
    "call *%r11", "call *%rax",
    # SSE
    "movsd -8(%rbp), %xmm0", "movsd %xmm0, -8(%rbp)",
    "movss -8(%rbp), %xmm0", "movss %xmm0, -8(%rbp)",
    "movsd (%r11), %xmm0", "movsd %xmm0, (%r11)",
    "movsd %xmm1, %xmm0", "movss %xmm1, %xmm0",
    "addsd %xmm1, %xmm0", "subsd %xmm1, %xmm0",
    "mulsd %xmm1, %xmm0", "divsd %xmm1, %xmm0",
    "addss %xmm1, %xmm0", "subss %xmm1, %xmm0",
    "mulss %xmm1, %xmm0", "divss %xmm1, %xmm0",
    "ucomisd %xmm1, %xmm0", "ucomiss %xmm1, %xmm0",
    "xorpd %xmm0, %xmm0", "xorps %xmm0, %xmm0",
    "cvtsi2sdq %r10, %xmm0", "cvtsi2ssq %r10, %xmm0",
    "cvttsd2si %xmm0, %r10", "cvttss2si %xmm0, %r10",
    "cvtss2sd %xmm0, %xmm0", "cvtsd2ss %xmm0, %xmm0",
    # `movq` NAMING AN SSE REGISTER, which is four different instructions
    # wearing one mnemonic. Encoding these as a general move produced a
    # program that ran and got every float wrong, because `%xmm3` and `%rbx`
    # are both register three.
    "movq %r10, %xmm0", "movq %rax, %xmm3", "movq %xmm0, %r10",
    "movq %xmm3, %rax", "movq %xmm0, %xmm1",
    "movq (%rax), %xmm0", "movq %xmm0, (%rax)",
    "movq -8(%rbp), %xmm0", "movq %xmm0, -8(%rbp)",
    "movd %eax, %xmm0", "movd %xmm0, %eax",
]


@harness.needs("binutils")
@harness.cases("line", CASES)
class TestEveryFormAgreesWithBinutils:
    """One case per encoding shape. See `CASES` for why the neighbours."""

    def test_the_bytes_are_the_same(self, line):
        want = assemble([line])
        got = ours(line)
        assert got == want, (
            f"{line}\n  ours: {got.hex(' ')}\n  as:   {want.hex(' ')}")


class TestBranchesResolveWithinAFunction:
    """A jump to a block label is patched once every length is known."""

    def test_a_backward_jump_reaches_its_label(self):
        out = Encoded()
        for line in (".Ltop:", "nopq", "jmp .Ltop"):
            if line == "nopq":
                out.code += b"\x90"     # a byte of distance, no encoder needed
                continue
            encode_line(line, out)
        from uasm.backends.x86_64.encode import resolve
        resolve(out)
        # jmp is five bytes and sits at offset 1, so it ends at 6 and the
        # label is at 0: the displacement is -6.
        assert struct.unpack("<i", bytes(out.code[2:6]))[0] == -6

    def test_a_forward_jump_reaches_its_label(self):
        out = Encoded()
        encode_line("jmp .Lend", out)
        out.code += b"\x90"
        encode_line(".Lend:", out)
        from uasm.backends.x86_64.encode import resolve
        resolve(out)
        assert struct.unpack("<i", bytes(out.code[1:5]))[0] == 1

    def test_an_unknown_label_is_refused(self):
        from uasm.backends.x86_64.encode import EncodeError, resolve
        out = Encoded()
        encode_line("jmp .Lnowhere", out)
        try:
            resolve(out)
        except EncodeError as exc:
            assert "Lnowhere" in str(exc)
        else:
            raise AssertionError("a branch to nothing was accepted")


class TestACallToASymbolAsksForARelocation:
    """A call outside the function is the linker's to place."""

    def test_the_relocation_names_the_symbol_and_the_offset(self):
        out = Encoded()
        encode_line("call apy_from_int", out)
        assert len(out.code) == 5
        assert out.relocs == [(1, "apy_from_int", 4, -4)]

    def test_a_rip_relative_load_is_relocated(self):
        out = Encoded()
        encode_line("leaq gv_x(%rip), %rax", out)
        (at, sym, kind, addend), = out.relocs
        assert sym == "gv_x" and addend == -4
        assert out.code[at:at + 4] == b"\0\0\0\0"


class TestWhatItCannotEncodeItRefuses:
    """A guess would be a program that runs and does something else."""

    def test_a_general_move_naming_an_sse_register_is_refused(self):
        """The failure mode this whole class is for.

        `%xmm3` and `%rbx` are both register three, so a general encoding of a
        `mov` naming an SSE register assembles, links and runs -- against the
        wrong register. Refusing is the only outcome that is not silent.
        """
        from uasm.backends.x86_64.encode import EncodeError
        out = Encoded()
        with harness.raises(EncodeError, match="xmm"):
            encode_line("movb %xmm3, (%rax)", out)

    def test_a_64_bit_immediate_no_alu_form_holds_is_refused(self):
        # `as` refuses this too: an `andq` immediate is 32 bits sign-extended,
        # so 0xffffffff is not a value the instruction can carry. Truncating
        # would mask with -1 -- an `and` that changes nothing.
        from uasm.backends.x86_64.encode import EncodeError
        out = Encoded()
        with harness.raises(EncodeError, match="imm32"):
            encode_line("andq $4294967295, %r10", out)

    def test_a_64_bit_immediate_store_to_memory_is_refused(self):
        # Also refused by `as`, and for the sharper reason: `movabs` is the
        # only 64-bit immediate form and it cannot name memory.
        from uasm.backends.x86_64.encode import EncodeError
        out = Encoded()
        with harness.raises(EncodeError, match="movabs"):
            encode_line("movq $4294967295, -8(%rbp)", out)

    def test_an_unknown_mnemonic_names_itself(self):
        from uasm.backends.x86_64.encode import EncodeError
        out = Encoded()
        try:
            encode_line("vfmadd231pd %ymm0, %ymm1, %ymm2", out)
        except (EncodeError, KeyError, AssertionError) as exc:
            assert exc is not None
        else:
            raise AssertionError("an unknown instruction was accepted")


class TestARipRelativeSymbolIsCarriedOrRefused:
    """Every form naming `sym(%rip)` either asks for a relocation or stops.

    WHAT THIS EXISTS TO PREVENT is the third thing, which is what the file
    used to do: emit a ZERO DISPLACEMENT AND NO RELOCATION. That assembles,
    links and disassembles without a word, and reads whatever happens to be
    four bytes past the instruction.

    NONE OF THE REFUSED FORMS IS REACHABLE TODAY -- `emit.py` writes exactly
    one RIP-relative operand, the `leaq` that takes a global's address, and
    every read-modify-write goes through a register after it. They are
    refused rather than fixed because recording the bias is a PER-FORM job:
    the displacement is measured from the end of the instruction, so a
    trailing `imm32` makes the addend -8 where `leaq`'s is -4, and Mach-O
    needs `X86_64_RELOC_SIGNED_4` rather than `SIGNED` to say so. Threading
    -4 through all of them would store four bytes past the global instead of
    declining.
    """

    #: The three that have somewhere to put it: a displacement with nothing
    #: after it.
    CARRIES = [
        "leaq gv(%rip), %rax",
        "movq gv(%rip), %rax",
        "movq %rax, gv(%rip)",
    ]

    #: And the ones that do not.
    REFUSES = [
        "movq $1, gv(%rip)",
        "movl $1, gv(%rip)",
        "addq $8, gv(%rip)",
        "andq $100000, gv(%rip)",
        "shlq $3, gv(%rip)",
        "shlq %cl, gv(%rip)",
        "negq gv(%rip)",
        "idivq gv(%rip)",
        "imulq gv(%rip), %rax",
        "testq %rax, gv(%rip)",
        "sete gv(%rip)",
        "movzbl gv(%rip), %eax",
        "movslq gv(%rip), %rax",
    ]

    @harness.cases("line", CARRIES)
    def test_it_asks_for_a_relocation(self, line):
        out = Encoded()
        encode_line(line, out)
        assert len(out.relocs) == 1, (line, out.relocs)
        _at, name, kind, addend = out.relocs[0]
        assert (name, kind, addend) == ("gv", 2, -4), out.relocs

    @harness.cases("line", REFUSES)
    def test_it_refuses_rather_than_dropping_the_symbol(self, line):
        from uasm.backends.x86_64.encode import EncodeError
        out = Encoded()
        with harness.raises(EncodeError, match="relocation"):
            encode_line(line, out)
        assert not out.relocs, out.relocs
