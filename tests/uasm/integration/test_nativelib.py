"""Declaring a native library, so a program can `import` one.

`frontends/python/nativelib.py` restores what `archived/docs/NATIVE_LIBRARIES.md`
described and the rewrite had lost: naming a shared library and the signatures
you want to call in it, rather than editing the compiler to use one.

WHAT IS ACTUALLY AT RISK, and it is not the happy path.

* **The signature.** A declaration is the only thing that says how many bytes
  an argument is. A wrong or guessed one does not fail -- it truncates, or it
  reads the wrong register -- so "no `args`" is refused rather than defaulted,
  and a type outside the vocabulary is refused rather than passed through.
* **Which library, on which platform.** A cross-platform program gives two
  libraries one module name. Picking the wrong one type-checks perfectly and
  fails at link, or worse, links against something that happens to export the
  name.
* **Shadowing.** A declaration that silently took the name `math` would be a
  build nobody could explain.

THE CALL PATH ITSELF IS NOT RE-TESTED HERE. A declared function becomes an
ordinary ctypes binding -- see `nativelib.py` for why that is the design --
so argument checking, conversion and lowering are `test_ctypes.py`'s subject
and are the same code. What is tested here is that the declaration ARRIVES.
"""
from __future__ import annotations

import json
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


def _declare(tmp_path: Path, *libraries: dict) -> Path:
    path = tmp_path / "libs.json"
    path.write_text(json.dumps(list(libraries)), encoding="utf-8")
    return path


def _program(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "prog.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


#: THE PLATFORM'S OWN LIBRARY, always present, so there is no fixture to build
#: and nothing that can be missing on the machine running the suite. Both
#: functions are pure and total, so the expected answer is arithmetic rather
#: than something about the machine.
if sys.platform == "win32":
    LIBRARY = {"module": "plat", "library": "user32.dll",
               "target_os": "windows",
               "functions": [{"name": "metric", "symbol": "GetSystemMetrics",
                              "args": ["c_int"], "ret": "c_int"}]}
    CALL, EXPECT = "plat.metric(0) > 0", "True"
else:
    LIBRARY = {"module": "plat", "library": "libm.so.6",
               "target_os": "linux",
               "functions": [{"name": "sqrt", "args": ["c_double"],
                              "ret": "c_double"}]}
    CALL, EXPECT = "plat.sqrt(9.0)", "3.0"


class TestImportingADeclaredLibrary:
    """The whole point: `import <library>` and call into it."""

    @harness.needs("gcc")
    def test_a_declared_library_is_imported_and_called(self, tmp_path):
        libs = _declare(tmp_path, LIBRARY)
        prog = _program(tmp_path, f"""
            import plat
            print({CALL})
        """)
        out = tmp_path / "prog.exe"
        # `--link` BECAUSE `build` STOPS AT AN UNLINKED OBJECT, and what this
        # class checks is a program that runs -- an object would only show the
        # declaration compiled, never that the library was found and called.
        #
        # `-ln cc` BECAUSE A DECLARED LIBRARY IS A SHARED LIBRARY, and the
        # builtin linker says outright that it links none; it is the default
        # now that `build` no longer goes through the C backend, so the one
        # toolchain here that can resolve `libm.so.6` has to be asked for by
        # name. The build in the test below wants both for the same reasons.
        built = _cli("build", str(prog), "--native-library", str(libs),
                     "--link", "-ln", "cc", "-o", str(out),
                     "--workdir", str(tmp_path / "wd"))
        assert built.returncode == 0, built.stdout + built.stderr
        ran = subprocess.run([str(out)], capture_output=True, text=True)
        assert ran.returncode == 0, ran.stdout + ran.stderr
        assert ran.stdout.strip() == EXPECT, ran.stdout

    @harness.needs("gcc")
    def test_the_declared_symbol_is_what_gets_called(self, tmp_path):
        """`symbol` renames: the program says `metric`, the linker sees
        `GetSystemMetrics`. Checked because a declaration that quietly called
        the IMPORTED name would fail at link with a name the source never
        wrote."""
        if sys.platform != "win32":
            harness.skip("the renaming case is written against user32")
        libs = _declare(tmp_path, LIBRARY)
        prog = _program(tmp_path, """
            import plat
            print(plat.metric(0) > 0)
        """)
        out = tmp_path / "prog.exe"
        built = _cli("build", str(prog), "--native-library", str(libs),
                     "--link", "-ln", "cc", "-o", str(out),
                     "--workdir", str(tmp_path / "wd"))
        assert built.returncode == 0, built.stdout + built.stderr

    def test_a_program_declaring_nothing_is_unaffected(self, tmp_path):
        prog = _program(tmp_path, """
            print(1 + 1)
        """)
        assert _cli("verify", str(prog)).returncode == 0


class TestWhatADeclarationMayNotDo:
    """The refusals, each of which would otherwise be a silent wrong build."""

    def test_it_cannot_shadow_a_module_the_compiler_has(self, tmp_path):
        libs = _declare(tmp_path, dict(LIBRARY, module="math"))
        prog = _program(tmp_path, """
            import math
            print(math.pi)
        """)
        done = _cli("verify", str(prog), "--native-library", str(libs))
        assert done.returncode != 0
        assert "E0131" in done.stdout + done.stderr

    def test_a_library_is_imported_whole(self, tmp_path):
        libs = _declare(tmp_path, LIBRARY)
        prog = _program(tmp_path, """
            from plat import sqrt
            print(sqrt(9.0))
        """)
        done = _cli("verify", str(prog), "--native-library", str(libs))
        assert "E0130" in done.stdout + done.stderr


class TestReadingADeclaration:
    """Parsing, where a wrong answer becomes a wrong number of bytes."""

    def _read(self, tmp_path, entry):
        sys.path.insert(0, str(SRC))
        try:
            from uasm.frontends.python import nativelib
            return nativelib, nativelib.read(_declare(tmp_path, entry))
        finally:
            del sys.path[0]

    def test_the_legacy_type_spellings_still_mean_something(self, tmp_path):
        """`archived/docs/NATIVE_LIBRARIES.md` writes `int`, `float`, `str`,
        and is still the description of this feature -- a reader following it
        must not be told the words it uses do not exist."""
        nativelib, registry = self._read(tmp_path, {
            "module": "m", "library": "libm.so.6",
            "functions": [{"name": "f", "args": ["int", "float", "str"],
                           "ret": "int"}]})
        fn = registry.get("m").member("f")
        assert fn.params == ("c_int", "c_double", "c_char_p")
        assert fn.ret == "c_int"

    @harness.cases("entry,fragment", [
        ({"module": "m", "library": "l",
          "functions": [{"name": "f", "args": ["c_wombat"], "ret": "c_int"}]},
         "not a native type"),
        ({"module": "m", "library": "l",
          "functions": [{"name": "f", "ret": "c_int"}]},
         "will not guess"),
        ({"module": "m"}, "needs a 'library'"),
        ({"library": "l"}, "needs a 'module'"),
        ({"module": "not an identifier", "library": "l"},
         "not a usable module name"),
    ])
    def test_a_bad_declaration_says_what_is_wrong(self, tmp_path, entry,
                                                  fragment):
        sys.path.insert(0, str(SRC))
        try:
            from uasm.frontends.python import nativelib
            try:
                nativelib.read(_declare(tmp_path, entry))
            except nativelib.DeclarationError as exc:
                assert fragment in str(exc), str(exc)
            else:
                raise AssertionError(f"{entry} was accepted")
        finally:
            del sys.path[0]

    def test_an_empty_args_list_is_a_declaration_and_not_an_omission(
            self, tmp_path):
        """The distinction the refusal above depends on: a function that
        takes nothing is declarable, and only LEAVING IT OUT is refused."""
        _, registry = self._read(tmp_path, {
            "module": "m", "library": "l",
            "functions": [{"name": "f", "args": [], "ret": "c_int"}]})
        assert registry.get("m").member("f").params == ()


class TestPlatformScoping:
    """Two libraries, one module name, and the target decides."""

    def _registry(self):
        sys.path.insert(0, str(SRC))
        try:
            from uasm.frontends.python import nativelib
        finally:
            del sys.path[0]
        registry = nativelib.Registry()
        registry.add(nativelib.NativeLibrary(
            module="gfx", library="user32.dll", target_os="windows"))
        registry.add(nativelib.NativeLibrary(
            module="gfx", library="libX11.so.6", target_os="linux"))
        return nativelib, registry

    @harness.cases("target,expected", [
        ("windows", "user32.dll"),
        ("linux", "libX11.so.6"),
    ])
    def test_the_target_picks_the_library(self, target, expected):
        _, registry = self._registry()
        assert registry.get("gfx", target).library == expected

    def test_a_target_nothing_was_scoped_to_gets_nothing(self):
        _, registry = self._registry()
        assert registry.get("gfx", "macos") is None

    def test_an_unscoped_declaration_is_the_fallback_and_never_the_winner(
            self):
        """A declaration naming this platform is more specific than one
        naming none, however they were written down -- so the order they
        appear in the file must not decide."""
        nativelib, registry = self._registry()
        registry.add(nativelib.NativeLibrary(module="gfx", library="any.so"))
        assert registry.get("gfx", "windows").library == "user32.dll"
        assert registry.get("gfx", "macos").library == "any.so"
