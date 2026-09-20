"""`ctypes`: calling a native library from a compiled program.

THE POINT IS WHAT IT DID NOT COST. The obvious implementation is
`dlopen`/`dlsym` -- a FOURTH and FIFTH platform function, against a floor that
stage 2 of `docs/INERT-RUNTIME.md` spent its whole argument getting down to
three. None was added, because none was needed: the C backend already emits
`extern double sqrt(double);` for any external the IR declares and does not
define, and the toolchain resolves it. That was measured with hand-written IR
before a line of this was designed.

So `ctypes.CDLL("m")` is not a load, it is a promise to the linker, and
`libm.sqrt(...)` is an ordinary `Op.CALL`. Nothing happens at run time that
would not happen in a C program calling the same function -- which is also
why these tests check the OUTPUT rather than any machinery: if the answer is
right, the ABI was right.
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


def _cli(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run([sys.executable, "-m", "uasm", *args],
                          capture_output=True, text=True, env=env)


def write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "prog.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def build_and_run(tmp_path: Path, source: str) -> subprocess.CompletedProcess:
    # `--link` BECAUSE `build` STOPS AT AN UNLINKED OBJECT, and what every
    # test reached through here checks is the OUTPUT of a program that ran.
    out = tmp_path / "prog.exe"
    built = _cli("build", str(write(tmp_path, source)), "--backend", "c",
                 "--link", "-o", str(out), "--workdir", str(tmp_path / "wd"))
    assert built.returncode == 0, built.stdout + built.stderr
    return subprocess.run([str(out)], capture_output=True, text=True)


def run_only(tmp_path: Path, source: str) -> subprocess.CompletedProcess:
    """The same program under the IR INTERPRETER rather than compiled.

    THE TWO HAVE TO AGREE and they reach a native symbol by completely
    different routes -- the compiled one through an `extern` the system linker
    resolves, this one through `objects/natives_host.py`. A test that only built
    would not notice the interpreter answering something else, or refusing to
    answer at all, which is what it did until `natives_host` existed.
    """
    return _cli("run", str(write(tmp_path, source)))


def refused(tmp_path: Path, source: str) -> str:
    r = _cli("build", str(write(tmp_path, source)), "--emit-ir",
             "-o", str(tmp_path / "p.ir"), "--workdir", str(tmp_path / "wd"))
    assert r.returncode != 0, "expected a diagnostic"
    return r.stdout + r.stderr


class TestItCallsTheLibrary:
    @harness.needs("gcc")
    def test_a_double_returning_function(self, tmp_path):
        """`sqrt` out of libm, by value, in and out.

        A float argument passed where the callee wants a double is the ctypes
        mistake that does not fail -- it produces a plausible wrong number --
        so the assertion is the exact digits.
        """
        got = build_and_run(tmp_path, """\
            import ctypes
            libm = ctypes.CDLL("m")
            libm.sqrt.restype = ctypes.c_double
            libm.sqrt.argtypes = [ctypes.c_double]

            def main() -> int:
                print(libm.sqrt(9.0))
                print(libm.sqrt(2.0))
                return 0
        """)
        assert got.returncode == 0, got.stderr
        assert got.stdout.split("\n")[:2] == ["3.0", "1.4142135623730951"]

    @harness.needs("gcc")
    def test_two_arguments_and_a_narrow_integer(self, tmp_path):
        """`c_int32` is 32 bits and Python's int is 64. Getting that wrong
        does not raise, it truncates -- which is the whole reason `argtypes`
        is required here rather than guessed."""
        got = build_and_run(tmp_path, """\
            from ctypes import CDLL, c_double, c_int32

            libm = CDLL("m")
            libm.pow.restype = c_double
            libm.pow.argtypes = [c_double, c_double]
            libm.abs.restype = c_int32
            libm.abs.argtypes = [c_int32]

            def main() -> int:
                print(libm.pow(2.0, 10.0))
                print(libm.abs(-42))
                return 0
        """)
        assert got.returncode == 0, got.stderr
        assert got.stdout.split("\n")[:2] == ["1024.0", "42"]

    @harness.needs("gcc")
    def test_the_library_reaches_the_linker_without_a_flag(self, tmp_path):
        """`CDLL("m")` is the whole declaration. Needing `--link-input=-lm` as
        well would mean saying the same thing twice, in two places that can
        disagree."""
        got = build_and_run(tmp_path, """\
            import ctypes
            libm = ctypes.CDLL("libm.so.6")
            libm.sqrt.restype = ctypes.c_double
            libm.sqrt.argtypes = [ctypes.c_double]

            def main() -> int:
                print(libm.sqrt(16.0))
                return 0
        """)
        assert got.returncode == 0, got.stderr
        assert got.stdout.startswith("4.0")


class TestTheDeclarationsAreNotCode:
    """They describe the build, and emitting them was three separate bugs."""

    def test_a_declaration_does_not_make_the_module_the_entry(self, tmp_path):
        """A module whose only top-level statements are ctypes declarations
        has no top level. Counting them made `<module>` the entry, so `main`
        was renamed out of the way and never called -- and the program
        compiled, linked, and printed nothing at all."""
        r = _cli("build", str(write(tmp_path, """\
            import ctypes
            libm = ctypes.CDLL("m")
            libm.sqrt.restype = ctypes.c_double
            libm.sqrt.argtypes = [ctypes.c_double]

            def main() -> int:
                print(libm.sqrt(9.0))
                return 0
        """)), "--emit-ir", "-o", str(tmp_path / "p.ir"),
                 "--workdir", str(tmp_path / "wd"))
        assert r.returncode == 0, r.stdout + r.stderr
        text = (tmp_path / "p.ir").read_text(encoding="utf-8")
        assert "export func main()" in text
        # Renamed to `py_main` only when a module entry exists, which is the
        # symptom this test exists for.
        assert "py_main" not in text, text[:400]

    def test_the_native_call_is_a_direct_call(self, tmp_path):
        """No load, no lookup, no runtime machinery -- one `call` to a symbol
        the module declares and does not define."""
        r = _cli("build", str(write(tmp_path, """\
            import ctypes
            libm = ctypes.CDLL("m")
            libm.sqrt.restype = ctypes.c_double
            libm.sqrt.argtypes = [ctypes.c_double]

            def main() -> int:
                print(libm.sqrt(9.0))
                return 0
        """)), "--emit-ir", "-o", str(tmp_path / "p.ir"),
                 "--workdir", str(tmp_path / "wd"))
        assert r.returncode == 0, r.stdout + r.stderr
        text = (tmp_path / "p.ir").read_text(encoding="utf-8")
        assert "func sqrt(%0: f64) -> f64 external" in text
        assert "f64.call @sqrt" in text

    def test_no_platform_function_was_added(self):
        """The claim this whole feature is measured against."""
        sys.path.insert(0, str(SRC))
        try:
            from uasm.objects.floor import FLOOR
            assert set(FLOOR) == {"plat_write", "plat_exit", "plat_heap"}, FLOOR
        finally:
            del sys.path[0]


class TestWhatItRefuses:
    """Each refusal is a thing that cannot work this way, said plainly."""

    def test_a_computed_library_name(self, tmp_path):
        """The one case `dlopen` would buy. There is no symbol to hand the
        linker, so it is refused rather than half-supported."""
        assert "E0124" in refused(tmp_path, """\
            import ctypes
            name = "m"
            lib = ctypes.CDLL(name)

            def main() -> int:
                return 0
        """)

    def test_a_call_with_no_argtypes(self, tmp_path):
        """STRICTER THAN CPYTHON, deliberately: there a missing `argtypes`
        means guess from the values, and guessing is how a ctypes program
        corrupts a stack."""
        assert "E0125" in refused(tmp_path, """\
            import ctypes
            libm = ctypes.CDLL("m")

            def main() -> int:
                print(libm.sqrt(9.0))
                return 0
        """)

    def test_a_ctypes_name_this_frontend_does_not_have(self, tmp_path):
        """Structs, arrays, `byref` and callbacks are not here yet, and a
        program using one should hear that rather than `undefined name`."""
        assert "E0128" in refused(tmp_path, """\
            from ctypes import Structure

            def main() -> int:
                return 0
        """)

    def test_a_library_used_as_a_value(self, tmp_path):
        """`libm` is a compile-time name and there is nothing to pass around,
        so reading one is refused where it is written."""
        assert "E0127" in refused(tmp_path, """\
            import ctypes
            libm = ctypes.CDLL("m")

            def main() -> int:
                x = libm
                return 0
        """)


class TestPointerArguments:
    """The three things `bundled/pathlib.py` needs, each checked on BOTH paths.

    A POINTER ARGUMENT IS NOT A NUMBER, and that is the whole difficulty. The
    value a Python program has is an `apy_value` -- a tagged object, or in the
    interpreter a HANDLE into a table -- and the callee wants an address. The
    two paths solve it differently and must arrive at the same answer, so
    every case here runs twice.
    """

    #: Opened, written, read back and removed, using only symbols that no
    #: header the runtime includes declares. `_write`/`_read` take a pointer,
    #: which is what makes this more than a `sqrt` test.
    #:
    #: PER PLATFORM, because the SYMBOL NAMES AND THE FLAG VALUES BOTH ARE.
    #: `_open` is MSVC's spelling of `open` and `O_CREAT` is 256 there and 64
    #: here, so one hard-coded program can only ever run on one of them --
    #: and this file used to hold the Windows one, which meant three tests
    #: that could not pass on the machine most people run the suite on. What
    #: cannot vary is the set of names AVAILABLE: `mkdir` and `rmdir` are
    #: declared by headers the C runtime includes and the backend's `extern`
    #: conflicts with them, which is a fact about this compiler rather than
    #: about the platform, so both columns stay inside `<fcntl.h>` and
    #: `<unistd.h>` -- neither of which the runtime includes.
    if sys.platform == "win32":
        _LIB = "c"
        _OPEN, _CLOSE, _READ, _WRITE = "_open", "_close", "_read", "_write"
        #: `O_WRONLY | O_CREAT | O_TRUNC | O_BINARY`, MSVC's values.
        _WRONLY_CREATE = 1 | 256 | 512 | 32768
        _RDONLY = 0 | 32768
        _MODE = 128
    else:
        _LIB = "c"
        _OPEN, _CLOSE, _READ, _WRITE = "open", "close", "read", "write"
        #: `O_WRONLY | O_CREAT | O_TRUNC` on Linux. There is no `O_BINARY`:
        #: POSIX has one kind of file.
        _WRONLY_CREATE = 1 | 64 | 512
        _RDONLY = 0
        _MODE = 420                       # 0o644

    ROUNDTRIP = """\
        import ctypes
        libc = ctypes.CDLL("%(lib)s")
        libc.%(open)s.restype = ctypes.c_int
        libc.%(open)s.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        libc.%(close)s.restype = ctypes.c_int
        libc.%(close)s.argtypes = [ctypes.c_int]
        libc.%(read)s.restype = ctypes.c_int
        libc.%(read)s.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        libc.%(write)s.restype = ctypes.c_int
        libc.%(write)s.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

        name = "apy-ctypes-case.bin"
        fd = libc.%(open)s(name, %(wrflags)d, %(mode)d)
        print("write fd ok:", fd >= 0)
        print("wrote:", libc.%(write)s(fd, b"abc\\x00d", 5))
        libc.%(close)s(fd)

        fd = libc.%(open)s(name, %(rdflags)d, 0)
        buf = bytearray(5)
        print("read:", libc.%(read)s(fd, buf, 5))
        libc.%(close)s(fd)
        print("got:", bytes(buf))

        # REMOVED BY THE PROGRAM, because it runs in the repository root and a
        # test that leaves a file behind makes `git status` dirty for whoever
        # runs the suite next. On Windows `DeleteFileA` rather than `_unlink`:
        # the second is declared in MinGW's <stdio.h> and the extern conflicts.
        %(cleanup)s
    """ % {
        "lib": _LIB, "open": _OPEN, "close": _CLOSE, "read": _READ,
        "write": _WRITE, "wrflags": _WRONLY_CREATE, "rdflags": _RDONLY,
        "mode": _MODE,
        "cleanup": (
            'k32 = ctypes.CDLL("kernel32")\n'
            '        k32.DeleteFileA.restype = ctypes.c_int\n'
            '        k32.DeleteFileA.argtypes = [ctypes.c_char_p]\n'
            '        print("cleaned:", k32.DeleteFileA(name) != 0)'
            if sys.platform == "win32" else
            'libc.unlink.restype = ctypes.c_int\n'
            '        libc.unlink.argtypes = [ctypes.c_char_p]\n'
            '        print("cleaned:", libc.unlink(name) == 0)'),
    }

    @harness.needs("gcc")
    def test_a_buffer_is_filled_by_the_callee_when_compiled(self, tmp_path):
        """`bytearray` IS the writable buffer, and reading proves it.

        A `str` argument only has to be READ by the callee, so passing its
        bytes is enough. A read has to be WRITTEN THROUGH, so the address
        handed over must be the object's own storage rather than a copy --
        and the failure when it is not is a buffer of the right LENGTH full
        of zeroes, which no exception marks.
        """
        got = build_and_run(tmp_path, self.ROUNDTRIP)
        assert got.returncode == 0, got.stderr
        lines = [ln for ln in got.stdout.split("\n") if ln]
        assert lines == ["write fd ok: True", "wrote: 5", "read: 5",
                         "got: b'abc\\x00d'", "cleaned: True"], got.stdout

    def test_the_interpreter_answers_the_same_call(self, tmp_path):
        """The oracle reaches the same symbols through `natives_host`.

        Its pointers are offsets into a `bytearray` the interpreter owns, so
        the write-back is a SECOND copy rather than a shared address -- see
        `natives_host._writeback`. Whether that copy happens is invisible
        except here.
        """
        got = run_only(tmp_path, self.ROUNDTRIP)
        assert got.returncode == 0, got.stdout + got.stderr
        lines = [ln for ln in got.stdout.split("\n") if ln]
        assert lines[-5:] == ["write fd ok: True", "wrote: 5", "read: 5",
                              "got: b'abc\\x00d'", "cleaned: True"], got.stdout

    #: `0` where a pointer is declared. C admits a null pointer constant for a
    #: pointer parameter and `CreateDirectoryA(path, 0)` is the ordinary way
    #: to write "no security descriptor"; refusing it made every native call
    #: with a NULL argument a TypeError.
    #:
    #: PER PLATFORM for the same reason `ROUNDTRIP` is, and the POSIX column
    #: uses `getcwd(NULL, 0)` -- glibc's documented "allocate one for me"
    #: mode, and the most ordinary nullable pointer argument there is. The
    #: obvious POSIX mirror of the Windows case, `mkdir(path, mode)`, cannot
    #: be used: `<sys/stat.h>` declares it and the backend's `extern`
    #: conflicts, exactly as MinGW's `<stdio.h>` does for `_unlink`.
    NULL_ARGUMENT = ("""\
        import ctypes
        k32 = ctypes.CDLL("kernel32")
        k32.CreateDirectoryA.restype = ctypes.c_int
        k32.CreateDirectoryA.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        k32.RemoveDirectoryA.restype = ctypes.c_int
        k32.RemoveDirectoryA.argtypes = [ctypes.c_char_p]
        k32.GetFileAttributesA.restype = ctypes.c_uint32
        k32.GetFileAttributesA.argtypes = [ctypes.c_char_p]

        name = "apy-ctypes-case-dir"
        print("made:", k32.CreateDirectoryA(name, 0) != 0)
        print("is dir:", (k32.GetFileAttributesA(name) & 16) != 0)
        print("removed:", k32.RemoveDirectoryA(name) != 0)
        print("gone:", k32.GetFileAttributesA(name) == 4294967295)
    """ if sys.platform == "win32" else """\
        import ctypes
        libc = ctypes.CDLL("c")
        libc.getcwd.restype = ctypes.c_void_p
        libc.getcwd.argtypes = [ctypes.c_void_p, ctypes.c_int]
        libc.open.restype = ctypes.c_int
        libc.open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        libc.close.restype = ctypes.c_int
        libc.close.argtypes = [ctypes.c_int]
        libc.unlink.restype = ctypes.c_int
        libc.unlink.argtypes = [ctypes.c_char_p]

        name = "apy-ctypes-case-null.bin"
        print("made:", libc.getcwd(0, 0) != 0)
        fd = libc.open(name, 1 | 64 | 512, 420)
        print("is dir:", fd >= 0)
        libc.close(fd)
        print("removed:", libc.unlink(name) == 0)
        print("gone:", libc.open(name, 0, 0) < 0)
    """)

    @harness.needs("gcc")
    def test_an_int_where_a_pointer_is_declared_when_compiled(self, tmp_path):
        got = build_and_run(tmp_path, self.NULL_ARGUMENT)
        assert got.returncode == 0, got.stderr
        lines = [ln for ln in got.stdout.split("\n") if ln]
        assert lines == ["made: True", "is dir: True", "removed: True",
                         "gone: True"], got.stdout

    def test_an_int_where_a_pointer_is_declared_in_the_interpreter(
            self, tmp_path):
        got = run_only(tmp_path, self.NULL_ARGUMENT)
        assert got.returncode == 0, got.stdout + got.stderr
        lines = [ln for ln in got.stdout.split("\n") if ln]
        assert lines[-4:] == ["made: True", "is dir: True", "removed: True",
                              "gone: True"], got.stdout

    def test_a_pointer_argument_that_is_neither(self, tmp_path):
        """A float has no bytes and no address, and saying so is the point.

        The check exists because the alternative is passing whatever the
        object's payload happens to be as an address, which is a crash
        somewhere else with nothing pointing back to here.
        """
        got = run_only(tmp_path, """\
            import ctypes
            k32 = ctypes.CDLL("kernel32")
            k32.GetFileAttributesA.restype = ctypes.c_uint32
            k32.GetFileAttributesA.argtypes = [ctypes.c_char_p]

            try:
                k32.GetFileAttributesA(1.5)
            except TypeError as e:
                print("TypeError:", "str, bytes or int" in str(e))
        """)
        assert got.returncode == 0, got.stdout + got.stderr
        assert "TypeError: True" in got.stdout, got.stdout
