"""The platform floor, executed on every path that can execute it.

STAGE 2 OF `docs/INERT-RUNTIME.md`. A backend that wants dynamic Python has to
define 229 `apy_*` functions today. The plan replaces them with a runtime
written in IR -- and a runtime written in IR still cannot talk to the machine,
so something has to. That something is three functions:

    plat_write(fd, buf, n)      plat_exit(code)      plat_heap(n)

`objects/floor.py` holds the contracts. THIS FILE IS THE CLAIM THAT THE NUMBER
IS REALLY THREE, and the only way to make that claim is to write a program that
uses nothing else and run it everywhere.

WHAT THE PROGRAM DOES, AND WHY IT IS THAT PROGRAM. It formats a signed 64-bit
integer to decimal and writes the bytes. That is the smallest thing that is
undeniably RUNTIME work rather than a smoke test: it needs memory it can index,
arithmetic at a fixed width, a loop whose trip count depends on the value, and
a way to emit bytes -- and it produces an answer that is either exactly right
or obviously wrong. There is no `put_int` anywhere in it, which is the point:
`put_int` is what every backend has to implement today, and after this it is
what one runtime implements once.

THE THREE PATHS MUST AGREE. The IR interpreter is the oracle -- it defines what
the program means -- and the C and JVM backends must produce its output byte
for byte. That is the same argument the corpus makes for the object runtime,
applied to the floor: two implementations of a contract that are never compared
are two contracts.

WHY `//` AND `%` ARE DONE UNSIGNED in the program below, spelled out because it
cost time: the subset is PYTHON, so `//` floors even at a machine width.
`-1234 // 10` is -124 and `-1 // 10` is -1, so a digit loop written on the
negative side never terminates -- it ran the index off the front of the buffer
and faulted. On `u64` both operands are non-negative, the floor correction is
dead code, and `0 - m` gives the magnitude of -9223372036854775808 too, where
negating in `i64` would give back the same number.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from tests import harness
from tests.harness import snapshot

SRC = snapshot.current(Path(__file__).resolve().parents[3])

#: See the module docstring. Every line of this is the machine subset; the only
#: symbols it does not define are the three the floor names.
FORMAT_AND_WRITE = """\
def write_i64(v: i64) -> None:
    buf: ptr = alloca(24)
    at: i64 = 24
    neg: bool = v < 0
    m: u64 = u64(v)
    if neg:
        m = 0 - m
    at = at - 1
    store(u8, u8(48 + m % 10), offset(buf, at))
    m = m // 10
    while m > 0:
        at = at - 1
        store(u8, u8(48 + m % 10), offset(buf, at))
        m = m // 10
    if neg:
        at = at - 1
        store(u8, 45, offset(buf, at))
    plat_write(1, offset(buf, at), 24 - at)


def newline() -> None:
    buf: ptr = alloca(1)
    store(u8, 10, buf)
    plat_write(1, buf, 1)


def main() -> int:
    write_i64(0)
    newline()
    write_i64(1234)
    newline()
    write_i64(-7)
    newline()
    write_i64(9223372036854775807)
    newline()
    write_i64(-9223372036854775808)
    newline()
    p: ptr = plat_heap(64)
    store(i64, 4242, p)
    write_i64(load(i64, p))
    newline()
    return 0
"""

EXPECTED = ("0\n1234\n-7\n9223372036854775807\n-9223372036854775808\n"
            "4242\n")

#: `plat_exit` ends the process and DOES NOT RETURN, so the second write must
#: never happen. A floor whose `exit` merely returned would pass a test that
#: only checked the status.
EXIT_PROGRAM = """\
def main() -> int:
    buf: ptr = alloca(3)
    store(u8, 104, buf)
    store(u8, 105, offset(buf, 1))
    store(u8, 10, offset(buf, 2))
    plat_write(1, buf, 3)
    plat_exit(7)
    plat_write(1, buf, 3)
    return 0
"""

#: See `test_jvm.py`: a JVM reserves its initial heap up front, and on a small
#: machine running one per worker that failed as an empty stdout -- which reads
#: exactly like a miscompile.
JAVA_FLAGS = ["-Xmx64m", "-XX:-UsePerfData"]


def _cli(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run([sys.executable, "-m", "uasm", *args],
                          capture_output=True, text=True, env=env)


def write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "prog.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def interpret(source: Path) -> subprocess.CompletedProcess:
    """The oracle: the IR interpreter, which defines what the program means."""
    return _cli("run", str(source))


def build_and_run(tmp_path: Path, source: Path, backend: str) -> \
        subprocess.CompletedProcess:
    out = tmp_path / ("prog.exe" if backend == "c" else "Prog.class")
    # `--link` BECAUSE `build` STOPS AT AN OBJECT. What this file checks is
    # that a PROGRAM built on the floor runs and prints the right thing, so
    # it asks for the program; a build without it writes the artifact the
    # backend produced and nothing more. See `test_builtin_linker.py`, whose
    # `TestBuildAndLinkAreSeparateVerbs` is the test of the split itself.
    built = _cli("build", str(source), "--backend", backend, "--link",
                 "-o", str(out), "--workdir", str(tmp_path / f"wd-{backend}"))
    assert built.returncode == 0, built.stdout + built.stderr
    if backend == "c":
        return subprocess.run([str(out)], capture_output=True, text=True)
    # The JVM backend writes a jar beside the class name it was given.
    jar = out.with_suffix(".jar")
    assert jar.is_file(), f"no jar at {jar}: {built.stdout}"
    return subprocess.run(["java", *JAVA_FLAGS, "-jar", str(jar)],
                          capture_output=True, text=True)


class TestTheFloorIsEnough:
    """A runtime operation written in the subset, over three functions."""

    def test_the_interpreter_formats_an_integer(self, tmp_path):
        """THE ORACLE. If this is wrong, the program is wrong and the backends
        agreeing with each other would prove nothing."""
        got = interpret(write(tmp_path, FORMAT_AND_WRITE))
        assert got.returncode == 0, got.stderr
        assert got.stdout == EXPECTED, repr(got.stdout)

    @harness.needs("gcc")
    def test_the_c_backend_agrees(self, tmp_path):
        got = build_and_run(tmp_path, write(tmp_path, FORMAT_AND_WRITE), "c")
        assert got.returncode == 0, got.stderr
        assert got.stdout == EXPECTED, repr(got.stdout)

    @harness.needs("java")
    def test_the_jvm_backend_agrees(self, tmp_path):
        """The one that matters most.

        `backends/jvm/emit.py` says in its own words that it cannot compile
        dynamic Python because the object runtime is C and nothing in a class
        file can call it. This program does real runtime work -- it formats a
        number -- and the class file runs it, because the only thing it needs
        from outside the IR is three methods the backend now has.
        """
        got = build_and_run(tmp_path, write(tmp_path, FORMAT_AND_WRITE), "jvm")
        assert "Exception" not in got.stderr, got.stderr
        assert got.returncode == 0, got.stderr
        assert got.stdout == EXPECTED, repr(got.stdout)


class TestExitDoesNotReturn:
    """The half of `plat_exit`'s contract a status check does not reach."""

    def test_the_interpreter_stops(self, tmp_path):
        got = interpret(write(tmp_path, EXIT_PROGRAM))
        assert got.returncode == 7, (got.returncode, got.stderr)
        assert got.stdout == "hi\n", repr(got.stdout)

    @harness.needs("gcc")
    def test_the_c_backend_stops(self, tmp_path):
        got = build_and_run(tmp_path, write(tmp_path, EXIT_PROGRAM), "c")
        assert got.returncode == 7, (got.returncode, got.stderr)
        assert got.stdout == "hi\n", repr(got.stdout)

    @harness.needs("java")
    def test_the_jvm_backend_stops(self, tmp_path):
        got = build_and_run(tmp_path, write(tmp_path, EXIT_PROGRAM), "jvm")
        assert "Exception" not in got.stderr, got.stderr
        assert got.returncode == 7, (got.returncode, got.stderr)
        assert got.stdout == "hi\n", repr(got.stdout)


class TestTheFloorStaysThree:
    """The deliverable is a NUMBER, so something has to hold it to it."""

    def test_there_are_exactly_three(self):
        from uasm.objects.floor import FLOOR
        assert set(FLOOR) == {"plat_write", "plat_exit", "plat_heap"}, FLOOR

    def test_every_backend_that_claims_the_floor_has_all_of_it(self):
        """A backend implementing two of three is worse than one implementing
        none: the program compiles and fails at load, or at the first call,
        naming a symbol the user never wrote."""
        from uasm.objects.floor import FLOOR
        from uasm.backends.jvm.runtime import HOST_NAMES as JVM_NAMES
        from uasm.objects.support import HOST_NAMES as C_NAMES
        assert set(FLOOR) <= set(JVM_NAMES), set(FLOOR) - set(JVM_NAMES)
        assert set(FLOOR) <= set(C_NAMES), set(FLOOR) - set(C_NAMES)

    def test_a_program_that_does_not_ask_does_not_pay(self, tmp_path):
        """Declaring the floor must not make every program depend on it.

        The same rule the rest of the runtime follows: a module that declares
        an import it never uses still gets it linked, and "no runtime
        dependencies" quietly stops being true for programs that only do
        arithmetic.
        """
        source = write(tmp_path, """\
            def main() -> int:
                print(1 + 1)
                return 0
        """)
        emitted = _cli("build", str(source), "--emit-ir",
                       "-o", str(tmp_path / "prog.ir"),
                       "--workdir", str(tmp_path / "wd"))
        assert emitted.returncode == 0, emitted.stdout + emitted.stderr
        text = (tmp_path / "prog.ir").read_text(encoding="utf-8")
        assert "plat_" not in text, text
