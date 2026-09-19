"""Which frontend, backend and linker a build means.

THE SPELLING CHOOSES, and a name typed on the command line only overrules it.
`uasm build thing.py -o thing.wasm` is a complete request: `.py` is the
Python frontend and `.wasm` is the wasm backend, and neither had to be named.
This is the whole of `-bk`, `-fr` and `-ln` being optional.

THREE QUESTIONS IN ONE ORDER, because the answers depend on each other:

  * the FRONTEND from the source's extension, which is independent;
  * the LINKER from the output's, because what the user names with `-o` is
    the program and not the artifact on the way to it -- `-o thing.so` is an
    extension module, and the `.c` in between is nobody's business;
  * the BACKEND from the linker, which declares what it can take input from,
    falling back to the output's extension when no linker claimed it.

AMBIGUITY IS AN ERROR AND NEVER A GUESS. Two components claiming one
extension is a question this cannot answer, and a warning followed by a build
of the wrong thing is worse than stopping: the user reads the warning after
the artifact already exists. So it raises, names every candidate, and says
which flag settles it.

A PREFERENCE IS NOT AN AMBIGUITY. `-o thing` with no extension at all names
no linker and no backend, and `cc` listing `c` before the machine backends is
a declared preference rather than a tie -- see `Toolchain.backends`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..backend.families import SelectionError

#: The linker a request that names none gets. See `choose` for why this one
#: line is also the answer to "which backend is the default".
DEFAULT_LINKER = "builtin"


@dataclass(frozen=True)
class Choice:
    """The three components one build runs through."""

    #: NONE MEANS "NOT DECIDED HERE". No registered frontend claims the
    #: source's extension, which is not always a mistake -- `uasm run
    #: thing.ir` reads IR directly and never asks a frontend at all -- so the
    #: answer is deferred to the pipeline, which already words the real
    #: failure. Ambiguity is still refused here, because that one nothing
    #: downstream can tell apart from absence.
    frontend: str | None
    backend: str
    linker: str

    #: WHETHER THE LINKER WAS POSITIVELY DETERMINED -- named with `-ln`, or
    #: claimed by the output's extension -- rather than fallen back to.
    #:
    #: THE TARGET GETS THE LAST WORD WHEN IT WAS NOT. A `.class` file is
    #: packaged into a jar rather than linked, and the jvm target says so with
    #: `default_toolchain`; so does pybc, and so does cpyext. `-o Prog.class`
    #: names a backend ARTIFACT, which falls back to `none` here and would
    #: leave the jar unwritten -- the target knowing better is the whole point
    #: of it declaring a default, and this is what lets the driver tell the
    #: two cases apart. It used to tell them apart by comparing the toolchain
    #: to the string "cc", which was a sentinel standing in for this flag.
    linker_named: bool = False


def _claimants(suffix: str, registry, attr: str) -> list[str]:
    """Every registered component whose `attr` lists `suffix`, sorted."""
    if not suffix:
        return []
    return sorted(
        name for name, component in registry.available().items()
        if suffix in getattr(component, attr, ()))


def _one(candidates: list[str], *, what: str, spelling: str,
         flag: str) -> str | None:
    """The single candidate, or None for no candidate at all.

    RAISES ON TWO OR MORE rather than taking the first. Sorting them makes
    first-wins look deliberate when it is alphabetical, which is exactly the
    kind of decision a reader would believe and should not.
    """
    if not candidates:
        return None
    if len(candidates) > 1:
        raise SelectionError(
            f"{spelling!r} does not say which {what}: "
            f"{', '.join(candidates)} all produce it; "
            f"pick one with {flag}")
    return candidates[0]


def choose_frontend(source: Path, named: str | None,
                    registry) -> str | None:
    """The frontend the source's extension names, or None for nobody's.

    AMBIGUITY IS REFUSED AND ABSENCE IS NOT, which is the one asymmetry here
    and it is deliberate. Two frontends claiming `.py` is a question only the
    user can settle, and nothing downstream can tell it apart from having no
    frontend at all -- the registry's own `for_path` answers None to both,
    which is how two claimants came to read as none.

    NOBODY CLAIMING THE EXTENSION IS NOT ALWAYS A MISTAKE. `uasm run
    thing.ir` reads IR directly, and the commands that do never reach a
    frontend; the ones that do reach one already report the failure in their
    own words. So this defers rather than guessing that it knows better.
    """
    if named:
        return named
    return _one(_claimants(source.suffix, registry, "extensions"),
                what="frontend", spelling=source.suffix,
                flag="-fr/--frontend")


def choose_linker(output: Path | None, named: str | None, registry,
                  *, fallback: str) -> str:
    """The linker the OUTPUT names, or `fallback` when nothing claims it.

    `fallback` IS THE CALLER'S TO DECIDE because the honest answer differs by
    verb: a build with no `-o` at all is producing a program and wants the
    native toolchain, while one whose output extension is a backend's own
    artifact wants `none` -- the backend has already produced the file being
    asked for and there is nothing left to link.
    """
    if named:
        return named
    if output is None:
        return fallback
    picked = _one(_claimants(output.suffix, registry, "artifacts"),
                  what="linker", spelling=output.suffix, flag="-ln/--linker")
    return picked if picked is not None else fallback


def _all_machines(candidates: list[str]) -> bool:
    """Whether every candidate emits for a machine rather than for a tool."""
    from ..backend.families import _ARCH_OF
    return all(name in _ARCH_OF for name in candidates)


def _target_backend(candidates: list[str], target: str | None) -> str | None:
    """The candidate that emits for the named target's architecture.

    None when nothing was named, when the name is not a target, or when the
    architecture is one no candidate emits for -- every case where guessing
    would be worse than falling through to the host.
    """
    if not target:
        return None
    from ..backend.families import _ARCH_OF
    from .. import target as target_registry
    try:
        arch = target_registry.get(target).arch
    except Exception:                                  # noqa: BLE001
        return None
    matching = [c for c in candidates if _ARCH_OF.get(c) == arch]
    return matching[0] if len(matching) == 1 else None


def choose_backend(output: Path | None, named: str | None, linker: str,
                   backends, linkers, target: str | None = None) -> str:
    """The backend, from the linker first and the output's spelling second.

    THE LINKER KNOWS BEST. It declares what it can take input from, in
    preference order, so `-o thing.so` reaches the `cpyext` backend rather
    than `c` -- both write `.c`, and the extension alone cannot tell them
    apart.

    WHICH IS WHY THE OUTPUT IS THE SECOND QUESTION AND NOT THE FIRST: it is
    asked only for a linker that takes anything, which is `none`, and there
    the output really is the backend's own artifact.
    """
    if named:
        return named
    wanted = getattr(linkers.get(linker), "backends", ())
    if wanted:
        ready = [b for b in wanted if b in backends.available()]
        if not ready:
            raise SelectionError(
                f"the {linker} linker takes input from "
                f"{', '.join(wanted)}, and none of them is registered")
        if len(ready) > 1 and _all_machines(ready):
            # THE TARGET DECIDES WHICH MACHINE, when every candidate IS a
            # machine and there is more than one to decide between. The
            # builtin linker takes input from both, so `--target
            # aarch64-macos` with no `-bk` reached the x86-64 backend and
            # was refused for declaring an ABI it does not implement -- a
            # failure whose cause is two flags away from what it says.
            #
            # AND THE HOST DECIDES WHEN NOTHING ELSE DOES, because a build
            # that names no target is a build for the machine it is running
            # on. Only then does the order in `backends` break the tie.
            #
            # `_all_machines` IS WHAT KEEPS THIS OFF `cc`. That toolchain
            # declares `c` first and means it -- the order IS the preference
            # -- and a list holding both `c` and a machine backend is not a
            # question about which machine.
            picked = (_target_backend(ready, target)
                      or _host_backend(ready))
            if picked is not None:
                return picked
        return ready[0]
    if output is not None:
        claimed = _claimants(output.suffix, backends, "artifacts")
        # THE MACHINE SETTLES A TIE BETWEEN MACHINE BACKENDS, and only that.
        # `-o thing.o` is claimed by all four of them, which is a real
        # ambiguity in general and not one here: an object for a machine this
        # is not running on is a cross build, and a cross build says so with
        # `--target` or `-bk`. Refusing the plain case sent every user of
        # `build --emit -o x.o` to a flag to say the obvious.
        #
        # A TIE BETWEEN UNLIKE BACKENDS IS STILL REFUSED. This only ever
        # picks from candidates that all emit for an architecture, so `c` and
        # `cpyext` both claiming `.c` is untouched by it.
        if len(claimed) > 1:
            native = _host_backend(claimed)
            if native is not None:
                return native
        picked = _one(claimed, what="backend", spelling=output.suffix,
                      flag="-bk/--backend")
        if picked is not None:
            return picked
    raise SelectionError(
        f"the {linker} linker takes input from any backend and "
        f"{'the output names none' if output is None else repr(output.suffix)}"
        f" does not say which; pick one with -bk/--backend")


def _takes(linkers, linker: str, backend: str) -> bool:
    """Whether `linker` declares that it can take input from `backend`.

    AN EMPTY DECLARATION MEANS ANY, which only `none` can honestly say -- it
    writes what the backend produced and never reads it. See
    `Toolchain.backends`.
    """
    wanted = getattr(linkers.get(linker), "backends", ())
    return not wanted or backend in wanted


def _linker_for(linkers, backend: str) -> str | None:
    """A registered linker that takes input from `backend`, or None.

    SORTED AND FIRST, which is arbitrary between equals and is not a tie in
    practice: the registered linkers each name a disjoint set of backends
    apart from the machine ones, and those are ordered by the preference
    their own declarations state.
    """
    for name in sorted(linkers.available()):
        wanted = getattr(linkers.get(name), "backends", ())
        if wanted and backend in wanted:
            return name
    return None


def _host_backend(candidates: list[str]) -> str | None:
    """The candidate that emits for the machine this is running on.

    None when no candidate does, which is every case this must not answer:
    a tie between backends that are not machine backends, and a host whose
    architecture nothing here targets.
    """
    from ..backend.families import _ARCH_OF
    from .. import target as target_registry
    try:
        host = target_registry.host()
    except Exception:                                  # noqa: BLE001
        return None
    arch = getattr(host, "arch", None)
    matching = [c for c in candidates if _ARCH_OF.get(c) == arch]
    return matching[0] if len(matching) == 1 else None


def choose(source: Path, output: Path | None, *, frontend: str | None,
           backend: str | None, linker: str | None, emit: bool,
           frontends, backends, linkers, target: str | None = None) -> Choice:
    """The three components one build runs through.

    WHAT `-o` MEANS DECIDES WHETHER THERE IS A LINKER AT ALL, and the four
    cases are worth spelling out because they are the whole policy:

      * `--emit` says not to link, so there is nothing to choose;
      * an output a LINKER claims is a program -- `.so`, `.jar`, `.pyc`;
      * an output a BACKEND claims is an artifact the user asked for by
        name, and asking a linker to turn `out.c` into a program named
        `out.c` is not what was meant;
      * anything else -- `-o thing`, or no `-o` at all -- is a program, and
        the native toolchain is what makes one.

    THE THIRD CASE IS THE ONE THAT USED TO BE WRONG. With the toolchain
    fixed at `cc`, `-o out.c` linked an executable and called it `out.c`.
    """
    fe = choose_frontend(source, frontend, frontends)
    # THE SPELLING NAMES THE PIPELINE EVEN WHEN NOTHING WILL BE LINKED, which
    # is why this is asked before `emit` is looked at. `--emit -o thing.so`
    # wants the artifact of the pipeline that MAKES a `.so`, and forcing the
    # linker to `none` first threw that away: `.so` is claimed by the cpyext
    # TOOLCHAIN and by no backend, so the backend question then had nothing
    # to go on and refused a request that is perfectly clear.
    names = choose_linker(
        output, linker, linkers,
        # An artifact the backend itself writes needs no linker; a program
        # does. `builtin` is named here rather than derived because "the one
        # that makes a native executable" is a fact about this driver and not
        # something a registry can be asked.
        #
        # IT USED TO BE `cc`, AND THAT IS WHERE "the default backend is C"
        # CAME FROM -- by a hop nobody reading this line would guess.
        # `choose_backend` below asks the LINKER what it takes input from,
        # and `CcToolchain.backends` lists `"c"` first; so the linker falling
        # back to `cc` chose the backend too. The builtin linker takes input
        # from the machine backends only, which is the same mechanism
        # answering the other way.
        fallback=("none" if (output is not None
                             and _claimants(output.suffix, backends,
                                            "artifacts"))
                  else DEFAULT_LINKER))
    # POSITIVELY DETERMINED, rather than fallen back to: either the user
    # named it or the output's extension is one a linker claims.
    named = bool(linker) or bool(
        output is not None and _claimants(output.suffix, linkers, "artifacts"))
    chosen = choose_backend(output, backend, names, backends, linkers, target)
    # AND THE BACKEND GETS TO CORRECT THE LINKER, which is the same coupling
    # read the other way. `choose_backend` asks the LINKER what it takes
    # input from, so a linker nobody named implies a backend; when the user
    # NAMES the backend instead, the implication runs backwards and a linker
    # that cannot read what that backend writes is not a choice anyone made.
    #
    # `-bk c` IS THE CASE THAT MATTERS. The C backend writes C, the builtin
    # linker reads objects, and the two together were a build that failed
    # after compiling everything -- with a message telling the user to name
    # a flag the request had already implied.
    if not named and not _takes(linkers, names, chosen):
        instead = _linker_for(linkers, chosen)
        if instead is not None:
            names = instead
    return Choice(
        frontend=fe,
        backend=chosen,
        # `--emit` TRUNCATES THE PIPELINE, it does not choose a different
        # one. An explicitly named linker still wins, because a user who
        # types both has said which they meant.
        linker=(linker or "none") if emit else names,
        # `--emit` IS ITSELF A DECISION. The user asked for the artifact and
        # nothing after it, so the target must not add a packaging step back.
        linker_named=named or emit)
