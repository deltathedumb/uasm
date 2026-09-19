"""The AArch64 object, proved by linking it beside one llvm-mc built.

WHAT THIS ADDS TO `test_arm64_encode`. That file checks one instruction at a
time, which is the encoder's claim. This checks the OBJECT's: that the words
land in the right order, that a branch inside a function resolves to the right
distance, and that every relocation names the right symbol with the right kind
-- three things no per-line test can see, because each is a property of the
whole file rather than of any line in it.

HOW IT IS CHECKED WITHOUT AN AArch64 MACHINE. Assemble the same module's
assembly with llvm-mc, link both objects at the same address with `ld.lld`,
and compare `.text`. Linking is what makes the comparison possible at all: our
object leaves a relocation where llvm-mc resolves a call to a symbol in the
same section itself, so the two files differ before linking and must not
after. A wrong relocation kind, a wrong addend or a wrong symbol all show up
here as different bytes.

WHY THE ASSEMBLY IS REBUILT RATHER THAN TAKEN FROM THE BACKEND. Because the
backend no longer emits any: an ELF target goes straight to the object. The
lines come from `_function` and `_global`, which is what the assembly path
used, so the two sides describe the same program.
"""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

from tests import harness

from uasm.backend import get, load_builtin
from uasm.ir.module import Linkage
from uasm.target import get as get_target

from .test_endtoend import ALL_PROGRAMS, compile_module

TRIPLE = "aarch64-linux-gnu"

#: A SPREAD RATHER THAN ALL OF THEM. Each program here reaches something the
#: others do not -- a call, a loop, a float, a class with its globals -- and
#: linking a quarter-megabyte object thirty-one times is minutes of test time
#: to re-prove what the per-line differential already covers.
PROGRAMS = ["arithmetic", "calls", "loops", "float_arithmetic",
            "class_basics", "closure_cell_is_shared", "user_exception_classes"]


def _assembly(backend, module, abi, dialect) -> str:
    """The same program as text, laid out the way the object lays it out.

    THE ZEROED GLOBALS GO IN `.bss` HERE, which the assembly path did not do:
    it wrote `.zero` inside `.data`, so the two would disagree about every
    address after the first zeroed global. Both layouts are correct and only
    one can be compared, so the text is written to match the object.
    """
    lines = [".text"]
    for fn in module.defined_functions():
        lines.extend(backend._function(fn, abi, dialect))
    initialised = [g for g in module.globals if g.data is not None]
    zeroed = [g for g in module.globals if g.data is None]
    if initialised:
        lines.append("\t.data")
        for g in initialised:
            lines.extend(backend._global(g, dialect))
    if zeroed:
        lines.append('\t.section .bss,"aw",@nobits')
        for g in zeroed:
            name = dialect.symbol_prefix + g.name
            if g.linkage is Linkage.EXPORT:
                lines.append(f"\t.globl {name}")
            lines.append("\t.align 3")
            lines.append(f"{name}:")
            lines.append(f"\t.zero {max(1, g.size)}")
    return "\n".join(lines) + "\n"


def _text_after_linking(obj: Path, tmp: Path, tag: str) -> bytes:
    """`.text` once the linker has resolved everything it can.

    `--no-relax` MATTERS. lld rewrites an `adrp`/`add` pair into `nop`/`adr`
    when the target is near enough, and whether it is near enough depends on
    the layout -- so with relaxation on, two correct objects can differ purely
    because one placed a section a few bytes elsewhere.
    """
    linked, raw = tmp / f"{tag}.elf", tmp / f"{tag}.bin"
    done = subprocess.run(
        ["ld.lld", "-o", str(linked), str(obj),
         "--unresolved-symbols=ignore-all", "-e", "0",
         "--image-base=0x100000", "--no-rosegment", "--no-relax"],
        capture_output=True, text=True)
    assert done.returncode == 0, f"linking {tag} failed:\n{done.stderr}"
    subprocess.run(
        ["llvm-objcopy", "-O", "binary", "--only-section=.text",
         str(linked), str(raw)], check=True, capture_output=True)
    return raw.read_bytes()


@harness.needs("llvm-aarch64", "lld")
@harness.cases("name", PROGRAMS)
class TestTheLinkedProgramMatchesLlvm:

    def test_the_text_sections_are_identical(self, name, tmp_path):
        from uasm.backends.arm64.emit import abi_for, dialect_for
        load_builtin()
        backend = get("arm64")
        target = get_target("aarch64-linux")
        abi, dialect = abi_for(target), dialect_for(target)
        source = textwrap.dedent(ALL_PROGRAMS[name]).strip() + "\n"
        module = compile_module(source, tmp_path, True)

        (tmp_path / "ours.o").write_bytes(
            backend.emit(module, target)["out.o"])
        (tmp_path / "theirs.s").write_text(
            _assembly(backend, module, abi, dialect), encoding="utf-8")
        subprocess.run(
            ["llvm-mc", f"-triple={TRIPLE}", "-filetype=obj",
             "-o", str(tmp_path / "theirs.o"), str(tmp_path / "theirs.s")],
            check=True, capture_output=True)

        ours = _text_after_linking(tmp_path / "ours.o", tmp_path, "ours")
        theirs = _text_after_linking(tmp_path / "theirs.o", tmp_path, "theirs")
        assert len(ours) == len(theirs), (
            f"{len(ours)} bytes against {len(theirs)}")
        for at in range(0, len(ours), 4):
            assert ours[at:at + 4] == theirs[at:at + 4], (
                f"first difference at {at:#x}: "
                f"ours {ours[at:at + 4].hex()} llvm {theirs[at:at + 4].hex()}")


@harness.needs("readelf")
class TestTheObjectIsWellFormed:
    """What the file says about itself, read by something that is not us."""

    def _object(self, tmp_path) -> Path:
        load_builtin()
        source = textwrap.dedent(ALL_PROGRAMS["class_basics"]).strip() + "\n"
        module = compile_module(source, tmp_path, True)
        path = tmp_path / "out.o"
        path.write_bytes(
            get("arm64").emit(module, get_target("aarch64-linux"))["out.o"])
        return path

    def test_it_names_aarch64(self, tmp_path):
        out = subprocess.run(["readelf", "-h", str(self._object(tmp_path))],
                             capture_output=True, text=True, check=True).stdout
        assert "AArch64" in out, out
        assert "REL (Relocatable file)" in out

    def test_the_relocations_are_the_aarch64_ones(self, tmp_path):
        """A relocation number from the wrong architecture still WRITES.

        The field is just an integer, so an object carrying x86-64's numbers
        is well formed and the linker rejects it -- or worse, finds a number
        that happens to mean something else. Reading the names back is the
        check that these are AArch64's.
        """
        out = subprocess.run(["readelf", "-rW", str(self._object(tmp_path))],
                             capture_output=True, text=True, check=True).stdout
        for wanted in ("R_AARCH64_CALL26", "R_AARCH64_ADR_PREL_PG_HI21",
                       "R_AARCH64_ADD_ABS_LO12_NC"):
            assert wanted in out, f"no {wanted} in the object"
        assert "R_X86_64" not in out


# ── the frame, which no machine here can execute ────────────────────────────
#
# READ RATHER THAN RUN, and the file's own preamble argues against exactly
# that -- so it is worth saying why these two are the exception. Both check a
# property of the STACK POINTER over a whole sequence: how far it moves, and
# what the offsets around it are measured from. There is no ARM machine here
# and no emulator, `llvm-mc` agrees with whatever text it is handed so it
# cannot catch a wrong offset, and both bugs below shipped in code that
# assembled, linked and disassembled without complaint. Reading is what there
# is, and reading finds them.

def _lines_of(name: str, source: str, tmp_path: Path) -> list[str]:
    """The AArch64 assembly for one function of `source`."""
    from uasm.backends.arm64.emit import abi_for, dialect_for
    load_builtin()
    backend = get("arm64")
    target = get_target("aarch64-linux")
    abi, dialect = abi_for(target), dialect_for(target)
    module = compile_module(textwrap.dedent(source).strip() + "\n",
                            tmp_path, True)
    for fn in module.defined_functions():
        if fn.name == name:
            return backend._function(fn, abi, dialect)
    raise AssertionError(f"no function named {name!r} was compiled")


def _immediate(line: str, after: str) -> int | None:
    """The `#n` in a line beginning `after`, or None if it is not one."""
    import re
    got = re.fullmatch(rf"\s*{after}\s*#(\d+)\s*", line)
    return int(got.group(1)) if got else None


#: Eleven arguments, two of them floats. Nine integers fill x0-x7 and spill
#: one onto the stack, which is what makes the caller push an outgoing area
#: at all -- and the floats live in frame slots, which is what makes the
#: offsets around that push matter.
WIDE_CALL = """
    def wide(a: float, b: int, c: int, d: int, e: int, f: int,
             g: int, h: int, i: int, j: int, k: float) -> float:
        return a + float(b + c + d + e + f + g + h + i + j) + k


    def go() -> float:
        return wide(1.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2.5)


    print(go())
"""

#: A frame bigger than one `sub sp, sp, #imm` can carry. `alloca` is counted
#: into the frame, so this is the shortest way to ask for one -- and it is
#: written in the STATICALLY TYPED subset, because that is where `alloca`
#: lives, with a dynamic `main` to call it so the module compiles as Python.
BIG_FRAME = """
    def wide() -> i64:
        buf: ptr = alloca(8192)
        store(i64, 4242, buf)
        return load(i64, buf)


    def main() -> int:
        return int(wide())
"""


class TestTheStackPointerIsAccountedFor:

    def test_a_slot_read_inside_a_call_is_biased_by_the_push(self, tmp_path):
        """SP HAS ALREADY MOVED by the time the arguments are read.

        A call with a stacked argument pushes the outgoing area first, so
        every frame slot is that much further up while it is being filled.
        The integer side added the bias and the FLOAT side did not, which is
        a float argument read from the outgoing area instead of from its own
        slot -- on every call that had both.
        """
        lines = _lines_of("go", WIDE_CALL, tmp_path)
        pushes = [(i, _immediate(line, "sub sp, sp,"))
                  for i, line in enumerate(lines)]
        pushes = [(i, n) for i, n in pushes if n]
        # The first is the prologue; the rest are outgoing argument areas.
        assert len(pushes) >= 2, lines
        at, adjust = pushes[1]
        end = next(i for i, line in enumerate(lines[at:], at)
                   if line.strip().startswith("bl "))
        reads = [line for line in lines[at:end]
                 if line.strip().startswith(("ldr ", "ldr\t"))]
        assert reads, lines[at:end]
        import re
        for line in reads:
            got = re.search(r"\[sp, #(\d+)\]", line)
            assert got, line
            assert int(got.group(1)) >= adjust, (
                f"{line.strip()!r} reads below the {adjust} bytes just "
                f"pushed, which is the outgoing area and not a frame slot")

    def test_a_frame_too_big_for_one_immediate_is_split(self, tmp_path):
        """`sub sp, sp, #imm` carries twelve bits and the frame needs more.

        THE ARCHITECTURE ALREADY ALLOWS IT: the immediate may be shifted
        left by twelve, so two instructions reach sixteen megabytes. The
        backend refused anything over 4088 instead, which is what stopped an
        AArch64 object runtime from building at all -- one function in it
        needs 5216 bytes.
        """
        lines = _lines_of("wide", BIG_FRAME, tmp_path)
        down = sum(n for n in (_immediate(line, "sub sp, sp,")
                               for line in lines) if n)
        up = sum(n for n in (_immediate(line, "add sp, sp,")
                             for line in lines) if n)
        assert down > 4095, f"the frame is only {down} bytes; make it bigger"
        assert down == up, f"SP goes down {down} and comes back up {up}"
        # AND EVERY PIECE FITS ONE IMMEDIATE, which is the point: a value
        # over 4095 is emitted as its high and low halves, and `encode.py`
        # picks the shift for the high one because it is a clean multiple.
        for line in lines:
            for direction in ("sub sp, sp,", "add sp, sp,"):
                n = _immediate(line, direction)
                if n is not None:
                    assert n <= 4095 or not n & 0xFFF, line

    def test_the_limit_it_still_has_is_the_one_it_says(self, tmp_path):
        """And it is a LOAD/STORE limit, not a stack-pointer one.

        `ldr`/`str` scale their twelve-bit offset by the access width, so a
        32-bit float reaches 16380 where a 64-bit register reaches 32760.
        The frame check has to be the smaller of those, and the encoder
        refuses anything past it rather than truncating -- so this asserts
        the two numbers agree.
        """
        from uasm.backends.arm64 import encode
        from uasm.backends.arm64.emit import MAX_FRAME

        assert MAX_FRAME <= 16380, MAX_FRAME
        assert MAX_FRAME % 8 == 0, "slots are eight-aligned"
        got = encode.encode_function([f"\tldr s0, [sp, #{MAX_FRAME}]"])
        assert len(got.code) == 4, got.code
        try:
            encode.encode_function([f"\tldr s0, [sp, #{MAX_FRAME + 8}]"])
        except encode.EncodeError:
            pass
        else:
            harness.fail("an offset past the limit should not encode")
