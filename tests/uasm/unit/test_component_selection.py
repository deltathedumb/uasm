"""What `-o thing.wasm` means without anyone naming a backend.

THE RULE THIS FILE PINS. A spelling chooses; a flag overrules. Two components
claiming one spelling is an ERROR naming both, never a pick -- a warning
followed by a build of the wrong thing is worse than stopping, because the
user reads the warning after the artifact already exists.

WHY THE AMBIGUOUS CASE IS TESTED WITH REAL COMPONENTS rather than a stub: `c`
and `cpyext` both write `.c` today, and that is not a hypothetical to be
mocked up -- it is the case the driver actually meets. A stub would keep
passing after the real collision was resolved or a third claimant appeared.
"""
from __future__ import annotations

from pathlib import Path

from tests import harness

from uasm import backend as backend_registry
from uasm import frontend as frontend_registry
from uasm import link as link_registry
from uasm import target as target_registry
from uasm.backend.families import SelectionError
from uasm.driver.select import (
    choose, choose_backend, choose_frontend, choose_linker,
)

backend_registry.load_builtin()
link_registry.load_builtin()
frontend_registry.load_builtin()


class TestTheSourceChoosesTheFrontend:
    def test_a_python_source_needs_no_flag(self):
        assert choose_frontend(Path("t.py"), None, frontend_registry) == "python"

    def test_a_named_frontend_overrules_the_spelling(self):
        assert choose_frontend(Path("t.py"), "uir", frontend_registry) == "uir"

    def test_an_unclaimed_extension_defers_rather_than_refusing(self):
        # NOT A MISTAKE BY ITSELF. `uasm run thing.ir` reads IR
        # directly and never asks a frontend, and the commands that do ask
        # already report the failure in their own words.
        assert choose_frontend(Path("t.ir"), None, frontend_registry) is None
        assert choose_frontend(Path("t.zzz"), None, frontend_registry) is None


class TestTheOutputChoosesTheLinker:
    def test_an_extension_module_is_the_cpyext_linker(self):
        got = choose_linker(Path("t.so"), None, link_registry, fallback="none")
        assert got == "cpyext"

    def test_a_jar_is_the_jar_linker(self):
        got = choose_linker(Path("t.jar"), None, link_registry, fallback="none")
        assert got == "jar"

    def test_an_unclaimed_extension_falls_back(self):
        got = choose_linker(Path("t.wasm"), None, link_registry,
                            fallback="none")
        assert got == "none"

    def test_no_output_at_all_falls_back(self):
        assert choose_linker(None, None, link_registry, fallback="cc") == "cc"

    def test_a_named_linker_overrules_the_spelling(self):
        got = choose_linker(Path("t.so"), "none", link_registry,
                            fallback="cc")
        assert got == "none"


class TestTheLinkerChoosesTheBackend:
    def test_cpyext_reaches_its_own_backend_not_c(self):
        # BOTH WRITE `.c`, so the extension cannot tell them apart and the
        # linker's own declaration is the only thing that can.
        got = choose_backend(Path("t.so"), None, "cpyext",
                             backend_registry, link_registry)
        assert got == "cpyext"

    def test_a_jar_reaches_the_jvm_backend(self):
        got = choose_backend(Path("t.jar"), None, "jar",
                             backend_registry, link_registry)
        assert got == "jvm"

    def test_cc_prefers_the_first_backend_it_declares(self):
        # A PREFERENCE IS NOT AN AMBIGUITY -- `-o thing` names no extension
        # at all, and `cc` listing `c` first is a declaration, not a tie.
        got = choose_backend(Path("thing"), None, "cc",
                             backend_registry, link_registry)
        assert got == "c"

    def test_an_unlinked_output_is_read_as_the_backends_own_artifact(self):
        got = choose_backend(Path("t.wasm"), None, "none",
                             backend_registry, link_registry)
        assert got == "wasm"

    def test_a_named_backend_overrules_everything(self):
        got = choose_backend(Path("t.so"), "llvm", "cpyext",
                             backend_registry, link_registry)
        assert got == "llvm"


class TestAmbiguityStopsAndNamesTheCandidates:
    def test_dot_c_does_not_say_whether_c_or_cpyext(self):
        try:
            choose_backend(Path("out.c"), None, "none",
                           backend_registry, link_registry)
        except SelectionError as exc:
            said = str(exc)
            assert "c" in said and "cpyext" in said, said
            assert "-bk/--backend" in said, said
        else:
            raise AssertionError(
                "'.c' is claimed by both c and cpyext; selection must refuse")

    def test_a_linker_taking_anything_with_no_spelling_refuses(self):
        try:
            choose_backend(None, None, "none",
                           backend_registry, link_registry)
        except SelectionError as exc:
            assert "-bk/--backend" in str(exc)
        else:
            raise AssertionError("expected a refusal")


class TestTheDeclarationsNameRealBackends:
    def test_every_linker_names_backends_that_exist(self):
        known = set(backend_registry.available())
        wrong = {name: [b for b in tc.backends if b not in known]
                 for name, tc in link_registry.available().items()
                 if [b for b in tc.backends if b not in known]}
        assert wrong == {}, f"linkers naming unregistered backends: {wrong}"


#: WHAT EACH SPELLING RESOLVES TO, end to end. The table is the contract: a
#: reader should be able to answer "what does `-o thing.jar` build?" without
#: reading the algorithm, and a change that moves any row is a change to what
#: the command means.
#:
#: THE FIRST AND LAST ROWS MOVED when the builtin linker arrived, and the move
#: is the point of it: an output that names no linker used to fall back to
#: `cc`, and `choose_backend` asks the LINKER what it takes input from -- so
#: `CcToolchain.backends` listing `"c"` first was the whole of "the default
#: backend is C". The builtin linker takes input from the machine backends,
#: so the same mechanism now answers `x86-64` on an x86-64 host. Every other
#: row is untouched: a spelling a linker or a backend claims is not a
#: fallback at all.
#: THE NATIVE BACKEND IS THE HOST'S, so the two rows that reach it are
#: derived rather than written: this file would otherwise pass on x86-64 and
#: fail on an ARM machine, which is a test asserting where it is running.
NATIVE = {"x86_64": "x86-64", "aarch64": "arm64",
          "x86": "x86-32", "arm": "arm32"}.get(
              getattr(target_registry.host(), "arch", ""), "x86-64")

RESOLVES = [
    ("thing",      "python", NATIVE,   "builtin"),
    ("thing.wasm", "python", "wasm",   "none"),
    ("thing.so",   "python", "cpyext", "cpyext"),
    ("thing.jar",  "python", "jvm",    "jar"),
    ("thing.pyc",  "python", "pybc",   "pyc"),
    ("thing.ll",   "python", "llvm",   "none"),
    (None,         "python", NATIVE,   "builtin"),
]


@harness.cases(
    "output,frontend,backend,linker",
    [harness.param(*row, id=str(row[0])) for row in RESOLVES])
def test_each_output_spelling_resolves(output, frontend, backend, linker):
    got = choose(Path("t.py"),
                 Path(output) if output is not None else None,
                 frontend=None, backend=None, linker=None, emit=False,
                 frontends=frontend_registry, backends=backend_registry,
                 linkers=link_registry)
    assert (got.frontend, got.backend, got.linker) == (
        frontend, backend, linker)


class TestTheWholeSpellingResolves:
    def test_emit_truncates_the_pipeline_it_does_not_change_it(self):
        # `--emit -o thing.so` wants the artifact of the pipeline that MAKES
        # a `.so`, so the spelling still picks `cpyext` and only the LINKING
        # is dropped. Choosing the linker after `emit` threw the spelling
        # away and refused a request that is perfectly clear -- `.so` is
        # claimed by the cpyext toolchain and by no backend at all.
        got = choose(Path("t.py"), Path("thing.so"), frontend=None,
                     backend=None, linker=None, emit=True,
                     frontends=frontend_registry, backends=backend_registry,
                     linkers=link_registry)
        assert (got.backend, got.linker) == ("cpyext", "none")

    def test_a_named_linker_survives_emit(self):
        got = choose(Path("t.py"), Path("thing.so"), frontend=None,
                     backend=None, linker="cpyext", emit=True,
                     frontends=frontend_registry, backends=backend_registry,
                     linkers=link_registry)
        assert got.linker == "cpyext"

    def test_an_artifact_spelling_is_not_linked_into_a_program(self):
        # THE CASE THAT USED TO BE WRONG. With the toolchain fixed at `cc`,
        # `-o out.ll` linked an executable and named it `out.ll`.
        got = choose(Path("t.py"), Path("out.ll"), frontend=None,
                     backend=None, linker=None, emit=False,
                     frontends=frontend_registry, backends=backend_registry,
                     linkers=link_registry)
        assert (got.backend, got.linker) == ("llvm", "none")


class TestWhoGetsTheLastWordOnTheLinker:
    """`linker_named` is what lets the TARGET override a fallback.

    THE CASE THIS EXISTS FOR. `-o Prog.class` names a backend ARTIFACT, so
    the linker falls back to `none` -- and the jvm target declares `jar`,
    because a class file is packaged rather than linked. The target has to
    win there, and must NOT win when the user said otherwise.

    IT USED TO BE DECIDED BY COMPARING THE TOOLCHAIN TO "cc", which worked
    only while "cc" was the hardcoded default and so stood in for "nobody
    chose". The moment the output's extension began choosing, that sentinel
    started reading real answers as decisions and the jar stopped being
    written.
    """

    def _named(self, output, **kw):
        return choose(Path("t.py"),
                      Path(output) if output is not None else None,
                      frontend=None, backend=None, linker=None, emit=False,
                      frontends=frontend_registry, backends=backend_registry,
                      linkers=link_registry, **kw).linker_named

    def test_a_backend_artifact_leaves_it_to_the_target(self):
        assert self._named("Prog.class") is False
        assert self._named("thing.ll") is False

    def test_no_output_leaves_it_to_the_target(self):
        assert self._named(None) is False

    def test_an_extension_a_linker_claims_is_a_decision(self):
        assert self._named("thing.so") is True
        assert self._named("thing.jar") is True

    def test_naming_the_linker_is_a_decision(self):
        got = choose(Path("t.py"), Path("Prog.class"), frontend=None,
                     backend=None, linker="none", emit=False,
                     frontends=frontend_registry, backends=backend_registry,
                     linkers=link_registry)
        assert got.linker_named is True

    def test_emit_is_a_decision_so_nothing_packages_it_again(self):
        # The user asked for the artifact and nothing after it; a target
        # adding its packaging step back would be answering a question that
        # was already settled.
        got = choose(Path("t.py"), Path("Prog.class"), frontend=None,
                     backend=None, linker=None, emit=True,
                     frontends=frontend_registry, backends=backend_registry,
                     linkers=link_registry)
        assert (got.linker, got.linker_named) == ("none", True)
