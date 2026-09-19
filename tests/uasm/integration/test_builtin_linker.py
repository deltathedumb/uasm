"""The builtin linker, exercised by running what it produces.

THE CLAIM THIS FILE CHECKS is a negative one -- that nothing outside this
repository is involved -- and a negative claim cannot be checked by looking
at the code. So every test here builds a real program, runs it, and compares
the output; and the two that matter most run it with `env -i`, so a `PATH`
pointing at a toolchain cannot be what made it work.

WHY THE PROGRAMS ARE THE ONES THEY ARE, smallest first:

  * `FLOOR_PROGRAM` uses `plat_write`, `plat_exit` and `plat_heap` and
    nothing else, so it fails if the syscall stubs in `link/freestanding.py`
    are wrong and passes with no object runtime at all. It is the same
    program `test_platform_floor.py` builds, for the same reason.
  * `print("hello")` is the whole Python runtime: the object runtime is
    compiled by uasm's own C frontend, linked, and run. It fails if any of
    the four pieces is wrong.

THESE TESTS ARE SLOW THE FIRST TIME -- the object runtime is 22k lines of C
and takes about twelve seconds to compile -- and fast afterwards, because
`link/runtimeobj.py` caches it under the workdir by the hash of its own
source. The cache is per workdir, and `tmp_path` gives each test its own, so
`test_the_runtime_is_cached` is the one that measures the reuse.
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

#: The floor and nothing else. See `test_platform_floor.py`, whose copy of
#: this is the oracle for what it should print.
FLOOR_PROGRAM = """\
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
    write_i64(-9223372036854775808)
    newline()
    p: ptr = plat_heap(64)
    store(i64, 4242, p)
    write_i64(load(i64, p))
    newline()
    return 0
"""

FLOOR_EXPECTED = "0\n-9223372036854775808\n4242\n"

#: `plat_exit` ENDS THE PROCESS, so the second write must never happen and
#: the status must be the one asked for. A floor whose exit merely returned
#: would pass a test that only looked at stdout.
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


def _cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    base = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run([sys.executable, "-m", "uasm", *args],
                          capture_output=True, text=True, env=env or base)


def _write(tmp_path: Path, source: str, name: str = "prog.py") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def _build(tmp_path: Path, source: Path, *extra: str) -> Path:
    """Build and link `source` with the builtin linker. Returns the program."""
    out = tmp_path / "prog"
    got = _cli("build", str(source), "--link", "-ln", "builtin",
               "-o", str(out), "--workdir", str(tmp_path / ".uasm"), *extra)
    assert got.returncode == 0, got.stderr[-3000:]
    assert out.exists(), got.stdout + got.stderr
    return out


def _run(program: Path, *, clean_env: bool = False) -> \
        subprocess.CompletedProcess:
    """Run it. `clean_env` empties the environment, PATH included.

    THAT IS THE WHOLE POINT OF THE FLAG. A statically linked program needs
    no loader, no library path and no PATH; running one with `env -i` is how
    this file's central claim is checked rather than asserted.
    """
    return subprocess.run([str(program)], capture_output=True,
                          env={} if clean_env else None, timeout=120)


class TestTheFloorAlone:
    """A program that uses only the three platform functions."""

    def test_it_runs_and_prints_what_the_interpreter_does(self, tmp_path):
        source = _write(tmp_path, FLOOR_PROGRAM)
        oracle = _cli("run", str(source))
        assert oracle.returncode == 0, oracle.stderr[-2000:]
        assert oracle.stdout == FLOOR_EXPECTED, oracle.stdout

        got = _run(_build(tmp_path, source))
        assert got.returncode == 0, got.stderr[-2000:]
        assert got.stdout.decode() == FLOOR_EXPECTED

    def test_it_runs_with_no_environment_at_all(self, tmp_path):
        program = _build(tmp_path, _write(tmp_path, FLOOR_PROGRAM))
        got = _run(program, clean_env=True)
        assert got.returncode == 0, got.stderr[-2000:]
        assert got.stdout.decode() == FLOOR_EXPECTED

    def test_plat_exit_ends_the_process_with_the_status(self, tmp_path):
        program = _build(tmp_path, _write(tmp_path, EXIT_PROGRAM))
        got = _run(program)
        assert got.returncode == 7, (got.returncode, got.stderr[-2000:])
        assert got.stdout == b"hi\n", got.stdout

    def test_the_image_is_static_and_has_no_interpreter(self, tmp_path):
        import struct
        program = _build(tmp_path, _write(tmp_path, FLOOR_PROGRAM))
        blob = program.read_bytes()
        assert blob[:4] == b"\x7fELF"
        e_type, = struct.unpack_from("<H", blob, 16)
        assert e_type == 2, "the image should be ET_EXEC"
        phoff, = struct.unpack_from("<Q", blob, 0x20)
        phentsize, phnum = struct.unpack_from("<HH", blob, 0x36)
        kinds = {struct.unpack_from("<I", blob, phoff + i * phentsize)[0]
                 for i in range(phnum)}
        # PT_INTERP is 3, and its absence is what "static" means to a loader.
        assert 3 not in kinds, "a static image must name no interpreter"
        # PT_DYNAMIC is 2: there is nothing for a dynamic loader to do.
        assert 2 not in kinds, "a static image must have no dynamic section"


class TestAWholePythonProgram:
    """`print("hello")`: the object runtime, compiled by uasm and linked."""

    def test_it_prints_what_the_interpreter_does(self, tmp_path):
        source = _write(tmp_path, 'print("hello")\nprint(2 + 3)\n')
        oracle = _cli("run", str(source))
        assert oracle.returncode == 0, oracle.stderr[-2000:]
        want = oracle.stdout

        got = _run(_build(tmp_path, source))
        assert got.returncode == 0, got.stderr[-2000:]
        assert got.stdout.decode() == want, (got.stdout, want)

    def test_it_runs_with_no_environment_at_all(self, tmp_path):
        source = _write(tmp_path, 'print("hello")\n')
        got = _run(_build(tmp_path, source), clean_env=True)
        assert got.returncode == 0, got.stderr[-2000:]
        assert got.stdout == b"hello\n"

    def test_the_runtime_is_cached_between_links(self, tmp_path):
        """The second link reuses the compiled runtime rather than rebuilding.

        MEASURED BY THE FILE AND NOT BY THE CLOCK. A timing assertion would
        be a flake on a loaded machine; the cache is a file whose name is the
        hash of its own source, so "was it reused" is "is there still exactly
        one of them, and is it older than the second program".
        """
        source = _write(tmp_path, 'print("hello")\n')
        _build(tmp_path, source)
        cache = tmp_path / ".uasm" / "runtime"
        objects = sorted(cache.glob("*.o"))
        assert len(objects) == 1, objects
        stamp = objects[0].stat().st_mtime_ns

        _build(tmp_path, source)
        again = sorted(cache.glob("*.o"))
        assert [p.name for p in again] == [p.name for p in objects]
        assert again[0].stat().st_mtime_ns == stamp, \
            "the runtime was recompiled when it should have been reused"


class TestBuildAndLinkAreSeparateVerbs:
    def test_build_writes_an_object_and_link_makes_the_program(self, tmp_path):
        source = _write(tmp_path, 'print("hello")\n')
        obj = tmp_path / "prog.o"
        built = _cli("build", str(source), "-o", str(obj),
                     "--workdir", str(tmp_path / ".uasm"))
        assert built.returncode == 0, built.stderr[-3000:]
        assert obj.exists(), built.stdout + built.stderr
        assert obj.read_bytes()[:4] == b"\x7fELF"
        # ET_REL: it is an object, not a program.
        import struct
        assert struct.unpack_from("<H", obj.read_bytes(), 16)[0] == 1

        out = tmp_path / "prog"
        linked = _cli("link", str(obj), "-o", str(out),
                      "--workdir", str(tmp_path / ".uasm"))
        assert linked.returncode == 0, linked.stderr[-3000:]
        got = _run(out)
        assert got.returncode == 0, got.stderr[-2000:]
        assert got.stdout == b"hello\n"


class TestItSaysWhatItCannotDo:
    def test_an_undefined_symbol_is_named(self, tmp_path):
        """A missing symbol is a diagnostic naming it, not a traceback."""
        from uasm.backend.objfile import (
            EM_X86_64, SHF_ALLOC, SHF_EXECINSTR, STB_GLOBAL, STT_FUNC,
            ElfObject, Relocation, Symbol,
        )
        from uasm.link.staticlink import LinkFailed, link

        obj = ElfObject(EM_X86_64)
        obj.section(".text", b"\xe8\x00\x00\x00\x00\xc3",
                    flags=SHF_ALLOC | SHF_EXECINSTR, align=16)
        obj.symbol(Symbol("_start", ".text", 0, 6, STB_GLOBAL, STT_FUNC))
        obj.symbol(Symbol("nowhere_at_all", "", binding=STB_GLOBAL))
        obj.relocate(".text", Relocation(1, "nowhere_at_all", 4, -4))
        try:
            link([("made-up.o", obj.to_bytes())])
        except LinkFailed as exc:
            assert "1 undefined symbol" in exc.message, exc.message
            assert "nowhere_at_all" in exc.detail, exc.detail
            assert "made-up.o" in exc.detail, exc.detail
        else:
            harness.fail("linking an undefined symbol should have failed")

    def test_two_definitions_of_one_name_are_refused(self, tmp_path):
        from uasm.backend.objfile import (
            EM_X86_64, SHF_ALLOC, SHF_EXECINSTR, STB_GLOBAL, STT_FUNC,
            ElfObject, Symbol,
        )
        from uasm.link.staticlink import LinkFailed, link

        def one(name: str) -> bytes:
            obj = ElfObject(EM_X86_64)
            obj.section(".text", b"\xc3", flags=SHF_ALLOC | SHF_EXECINSTR,
                        align=16)
            obj.symbol(Symbol("_start", ".text", 0, 1, STB_GLOBAL, STT_FUNC))
            obj.symbol(Symbol("twice", ".text", 0, 1, STB_GLOBAL, STT_FUNC))
            return obj.to_bytes()

        try:
            link([("a.o", one("a")), ("b.o", one("b"))])
        except LinkFailed as exc:
            assert "defined twice" in exc.message, exc.message
        else:
            harness.fail("two definitions should have been refused")
