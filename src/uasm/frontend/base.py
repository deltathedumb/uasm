"""The frontend interface.

A frontend turns source text into a verified `Module`. Mirror image of a
backend, and the same size:

    class MyLang(Frontend):
        name = "mylang"
        extensions = (".ml",)
        def compile(self, source, sink) -> Module | None

Everything language-specific lives on this side of the line. The IR has no
opinion about Python's `//` flooring toward negative infinity, about boxing, or
about dynamic dispatch: those are things a frontend LOWERS into the IR's small
vocabulary plus whatever runtime functions it decides to call.

`Op.DIV` truncates toward zero, like C and like every machine. Python's `//`
floors. So a Python frontend emits more than one instruction for `//` -- paid
once here, rather than owed by each of the backends. That is the whole argument
for a small IR, and it is why this interface takes source text and returns IR
with nothing in between that a backend could reach into.

Returning None means "I reported errors to the sink". A frontend never raises
for bad user input; it reports and returns None so the driver can decide
whether to continue with other inputs.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path

from ..diagnostics import DiagnosticSink, SourceFile
from ..ir import Module
from ..options import Option, OptionError  # noqa: F401  (re-export)


@dataclass(frozen=True, slots=True)
class BuildContext:
    """What a frontend needs about the run that is not one of its own flags.

    A FLAG IS NOT THE WHOLE OF WHAT `configure` NEEDS. `--import-path` adds
    to a search path whose FIRST entry is the source's own directory, and a
    scoped native-library declaration picks a library by the platform being
    built for -- neither is anything the user typed, and both were reasons
    the driver did this work itself.

    HANDED OVER RATHER THAN LOOKED UP. The driver resolves the target once
    and passes what it resolved; a frontend working it out again is how a
    program type-checks against `user32.dll` and links against
    `libX11.so.6`.
    """

    #: The file being compiled. Its directory is `sys.path[0]` in CPython's
    #: terms -- first on the search path, unless a flag says otherwise.
    source: Path
    #: The platform the program is being built for, as `Target.os` spells it.
    #: None when nothing can say yet, which is not an error: it leaves only
    #: the declarations that named no platform applying.
    target_os: str | None = None

    #: NOTHING AFTER THE FRONTEND WILL RUN. `uasm verify` compiles and then
    #: throws the result away, so a frontend may skip work whose only
    #: consumer is a backend or a linker.
    #:
    #: WHAT IT DOES NOT LICENSE: skipping anything that could change a
    #: diagnostic. `verify` exists so that a program it passes is a program
    #: `build` accepts, and a frontend that analysed less under this flag
    #: would break exactly that -- silently, and only for the programs where
    #: it mattered.
    verifying: bool = False


class Frontend(abc.ABC):
    """Turn source text into a Module."""

    name: str = ""
    extensions: tuple[str, ...] = ()
    description: str = ""
    #: Language specifications this frontend can compile.  The driver does
    #: not interpret these strings; each frontend owns its standard's naming
    #: and validates its `--language-version` option in `configure`.
    language_versions: tuple[str, ...] = ()
    default_language_version: str = ""

    #: OPTIONS THIS FRONTEND TAKES from the command line, declared the way a
    #: backend has always declared its own. The flags a frontend needs used
    #: to sit on the DRIVER's parser -- `--import-path`, `--host-python`,
    #: `--no-site-packages` are the Python frontend's and nobody else's -- so
    #: the driver carried them as though they were facts about compiling in
    #: general, and a second frontend would have inherited flags that mean
    #: nothing to it.
    options: tuple[Option, ...] = ()

    def configure(self, values: dict, context: "BuildContext",
                  sink: DiagnosticSink) -> "Frontend | None":
        """The frontend to compile with, given this run's option values.

        THE MIRROR OF `Backend.configure`, and for the same reasons. `values`
        holds only the options this frontend declared, keyed by name without
        the dashes; a repeatable one arrives as a list and a switch as True.
        Raise `OptionError` for a value that cannot be used, and report
        anything advisory to `sink`.

        `context` IS WHAT A FRONTEND NEEDS THAT IS NOT ITS OWN FLAG -- the
        source, and the platform being built for. See `BuildContext` for why
        the driver hands them over rather than the frontend looking them up.

        RETURN A NEW INSTANCE rather than mutating `self`. The registry holds
        one shared frontend object, so a frontend that stored its flags on
        itself would leak them into the next compilation in the same process
        -- invisible in a command-line run and wrong in every test suite and
        every embedding tool.

        RETURN None, having reported to `sink`, for anything that stops the
        build before compiling: an interpreter that would not run, a
        declaration file that will not parse. A diagnostic and a return are
        the two halves of one refusal.
        """
        return self

    @abc.abstractmethod
    def compile(self, source: SourceFile, sink: DiagnosticSink) -> Module | None:
        """Lower `source` to a Module, or return None having reported errors."""

    def __repr__(self) -> str:
        return f"<frontend {self.name}>"


_REGISTRY: dict[str, Frontend] = {}


def register(fe: Frontend) -> Frontend:
    if not fe.name:
        raise ValueError(f"{type(fe).__name__} has no name")
    if fe.name in _REGISTRY:
        raise ValueError(f"frontend {fe.name!r} is already registered")
    _REGISTRY[fe.name] = fe
    return fe


def get(name: str) -> Frontend:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise SystemExit(f"unknown frontend {name!r}\navailable: {known}") from None


def for_path(path: Path) -> Frontend | None:
    matches = [f for f in _REGISTRY.values() if path.suffix in f.extensions]
    return matches[0] if len(matches) == 1 else None


def available() -> dict[str, Frontend]:
    return dict(_REGISTRY)


def load_builtin() -> None:
    from ..frontends import c, python  # noqa: F401
