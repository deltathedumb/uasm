"""The object runtime, as an object file, compiled by uasm itself.

WHAT THIS REPLACES. Every native build so far has ended with

    gcc out.o uasm_runtime.c -o prog -lm -ldl -lpthread

and that one line is the whole of why a compiler that encodes its own
instructions and writes its own object files still could not produce a
program by itself. The C it hands to `gcc` is not foreign code: it is
`objects/` generated as a translation unit, and uasm has a C frontend. So it
compiles its own runtime, and the last external tool leaves.

WHAT COMES OUT. One ELF relocatable with the entire object runtime in it and
exactly three undefined symbols -- `plat_write`, `plat_exit`, `plat_heap` --
which `freestanding.py` answers with syscalls. No libc, because uasm's
`<stdio.h>` is compiled from C that bottoms out on the same three.

THE RUNTIME HAS TO MATCH THE PROGRAM, which is the one subtlety. A program
object built with the ported runtime spliced in already DEFINES four hundred
of these functions, and a runtime that defined them again is a duplicate
symbol; a program built `--object-runtime c` defines none of them and needs
them all. So the C is generated per link, from what the program object
actually defines -- read out of its symbol table rather than taken on trust
from a flag, because the object is the thing being linked and a flag is a
claim about it.

CACHED, BECAUSE IT TAKES TWELVE SECONDS. Keyed by the SHA-256 of the exact C
text and the target, so a cache hit means the bytes being reused were
compiled from the same source for the same machine -- there is no staleness
window and no need to invalidate anything by hand.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ..objects import ir as objects_ir
from ..objects.csource import objects_c
from ..objects.support import RUNTIME_C, host_functions

#: Where a compiled runtime is kept between builds. Under the workdir rather
#: than in a home directory: a build tree is the thing whose contents it
#: matches, and a stale global cache shared between two checkouts of this
#: compiler is exactly the confusion the content hash is there to avoid.
CACHE_DIR = "runtime"


class RuntimeBuildFailed(Exception):
    """The runtime could not be compiled. Carries the diagnostics."""

    def __init__(self, message: str, *, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


def runtime_source(provided: frozenset[str] | set[str]) -> str:
    """The runtime C to compile, given what the program object already has.

    `provided` is the set of symbols the program object DEFINES. Every ported
    name in it is omitted from the C -- the IR has it -- and every SPLIT name
    in it keeps its body under `<name>_slow`, which is what the IR's fast half
    calls when it meets a case it does not handle.

    THE `main` SHIM AND THE FLOOR ARE BOTH LEFT OUT. See `objects/support.py`:
    the shim would become a second definition of the backend's entry symbol,
    and the floor would conflict with the declarations uasm's own standard
    library is compiled against.
    """
    omit = tuple(n for n in objects_ir.PORTED
                 if n in provided and n not in objects_ir.SPLIT)
    split = tuple(n for n in objects_ir.SPLIT if n in provided)
    text = RUNTIME_C.replace(
        "int main(void) { return (int)@ENTRY@(); }",
        "/* The `main` shim is omitted: `_start` calls the entry directly. */")
    return (text.replace("@HOST@", host_functions(floor=False))
            .replace("@OBJECTS@", objects_c(omit=omit, split=split))
            .replace("@ENTRY@", "uasm_main"))


def _cache_path(workdir: Path, text: str, backend: str,
                target_name: str) -> Path:
    digest = hashlib.sha256(
        text.encode("utf-8")
        + f"\0{backend}\0{target_name}".encode("utf-8")).hexdigest()[:32]
    return workdir / CACHE_DIR / f"runtime-{digest}.o"


def runtime_object(provided: frozenset[str] | set[str], *, backend: str,
                   target, workdir: Path, verbose: bool = False) -> bytes:
    """The object runtime as an object, compiled by uasm, cached by content."""
    text = runtime_source(provided)
    target_name = getattr(target, "name", "") or ""
    cached = _cache_path(workdir, text, backend, target_name)
    if cached.exists():
        try:
            return cached.read_bytes()
        except OSError:
            pass   # unreadable cache is a cache miss, not a failure

    source = cached.with_suffix(".c")
    cached.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(text, encoding="utf-8")

    blob = _compile(source, backend=backend, target=target,
                    workdir=cached.parent, verbose=verbose)
    # WRITTEN THROUGH A TEMPORARY AND RENAMED, so that two builds running at
    # once never leave a half-written object for the third to read. `rename`
    # is atomic within a directory on every platform this runs on.
    staging = cached.with_suffix(f".o.{os.getpid()}")
    staging.write_bytes(blob)
    staging.replace(cached)
    return blob


def _compile(source: Path, *, backend: str, target, workdir: Path,
             verbose: bool) -> bytes:
    """Run the compiler over the runtime C, in this process.

    IN PROCESS AND NOT AS A SUBPROCESS, which is worth saying because a
    compiler invoking itself usually means the latter. There is nothing to
    isolate: the frontend and the backend are ordinary modules, the work is
    deterministic, and a subprocess would have to find this interpreter again
    -- the one thing a build that claims no external tools should not have to
    do.

    `--object-runtime c` IS NOT OPTIONAL HERE. This translation unit IS the
    object runtime; splicing the IR one into it makes every ported function
    defined twice, which the splice reports as an internal error rather than
    as the ordinary mistake it is.

    IMPORTED AT CALL TIME because the driver imports the link registry and
    this is the link registry importing the driver back. The cycle is real
    and only at module scope; inside a function both halves already exist.
    """
    from ..diagnostics import DiagnosticSink
    from ..driver.pipeline import Options, compile_source

    sink = DiagnosticSink()
    opts = Options(source=source, output=None, frontend="c", backend=backend,
                   target=target, link=False, workdir=workdir,
                   verbose=verbose, object_runtime="c")
    try:
        result = compile_source(opts, sink)
    except Exception as exc:                       # noqa: BLE001
        raise RuntimeBuildFailed(
            "could not compile the object runtime",
            detail=f"{type(exc).__name__}: {exc}") from None
    if not result.ok or not result.artifacts:
        raise RuntimeBuildFailed(
            "could not compile the object runtime",
            detail=_render(sink))
    objects = [blob for name, blob in sorted(result.artifacts.items())
               if name.endswith(".o")]
    if len(objects) != 1:
        raise RuntimeBuildFailed(
            f"the {backend} backend produced "
            f"{len(objects)} objects for the runtime, not one",
            detail=", ".join(sorted(result.artifacts)))
    return objects[0]


def _render(sink) -> str:
    """The sink's diagnostics as text, however this sink spells that."""
    for attr in ("render", "text"):
        got = getattr(sink, attr, None)
        if callable(got):
            try:
                return str(got())
            except Exception:                      # noqa: BLE001
                break
    return ""


__all__ = ["RuntimeBuildFailed", "runtime_object", "runtime_source"]
