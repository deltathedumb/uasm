"""`uasm` -- the command line.

FIVE VERBS, and they are the five things a compiler does:

    uasm build prog.py                  source -> IR -> a program you can run
    uasm build prog.py --emit           stop at the artifacts; do not link
    uasm build prog.py --emit-ir        stop after the IR and print it
    uasm build prog.py -O --time-passes optimise, and show what each pass cost
    uasm build prog.py --target x86_64-linux --backend x86-64
    uasm build prog.py --backend jvm --java-version 21   -> a runnable jar
    uasm run prog.py                    execute in the reference interpreter
    uasm verify prog.py [--json]        compile and verify, produce nothing
    uasm link a.ir b.ir -o all.ir       join modules at the IR
    uasm link a.o b.o -o prog           or objects, into a program
    uasm plugin add|remove|list|show    install plugins and see what they are
    uasm plugin backends|frontends|linkers|targets|passes
    uasm plugin ops|types|libraries|port

THE LISTINGS ARE UNDER `plugin` because every one of them answers a question
about the INSTALLATION rather than about a program, and a plugin is the
reason the answer can differ between two machines. They were nine verbs of
their own, sitting beside `build` and `run` as though listing something were
the same kind of act as compiling.

Every stage can be stopped at and dumped, and `--emit-ir` writes text that
`uasm run` accepts. That round trip is what makes a backend debuggable: you can
read exactly what it was given, and run that same text through the interpreter
to learn what it should have produced.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import plugins
from .. import backend as backend_registry
from .. import frontend as frontend_registry
from .. import link as link_registry
from .. import target as target_registry
from ..diagnostics import DiagnosticSink, Renderer, SourceFile, error, is_real
from ..ir import opcodes, types as T, verify
from ..options import Option
from ..ir.interpreter import Interpreter, Trap
from ..ir.printer import ParseError, parse_module, print_module
from ..ir.verifier import VerifyError
from ..passes import available as available_passes
from .pipeline import DEFAULT_PASSES, Options, compile_source


def _sink(args) -> DiagnosticSink:
    return DiagnosticSink(
        max_errors=getattr(args, "max_errors", 100),
        warnings_are_errors=getattr(args, "werror", False),
    )


def _select(args):
    """Every component one build runs through, and its target.

    RESOLVED IN ONE PLACE so the pieces cannot be read in different orders by
    different commands. A contradiction is a usage error and exits saying
    which two disagree, rather than one silently winning.

    THE SPELLING CHOOSES AND A FLAG OVERRULES -- see `driver/select.py` for
    the rules and the table of what each output extension resolves to. This
    used to default to `c` and `cc` outright, so `-o thing.wasm` built C and
    `-o out.ll` linked an executable and named it `out.ll`.

    THE FAMILY RESOLUTION STILL RUNS AFTERWARDS. `--bits` and `--target`
    narrow WITHIN a backend -- `x86` plus 32 is `x86-32` -- which is a
    different question from which backend, and asking it second means the
    answer applies whether the backend was chosen or named.
    """
    from .. import frontend as frontend_registry
    from .. import link as link_registry
    from ..backend.families import (
        SelectionError, resolve_backend, resolve_target,
    )
    from . import select as selector
    backend_registry.load_builtin()
    frontend_registry.load_builtin()
    link_registry.load_builtin()
    bits = getattr(args, "bits", None)
    named = getattr(args, "target", None)
    out = getattr(args, "output", None)
    try:
        choice = selector.choose(
            Path(args.source), Path(out) if out else None,
            frontend=getattr(args, "frontend", None),
            backend=getattr(args, "backend", None),
            linker=getattr(args, "toolchain", None),
            # `--emit-asm` IMPLIES `--emit`, the same way it does for `link`
            # below: assembly is not something this driver's toolchains take.
            emit=(getattr(args, "emit", False)
                  or getattr(args, "emit_asm", False)),
            frontends=frontend_registry, backends=backend_registry,
            linkers=link_registry,
            # WHAT WAS ASKED FOR, not what was resolved: the target is
            # resolved FROM the backend two lines down, and the backend is
            # what is being chosen here. A named target is the only thing
            # that can say which machine a build is for before then.
            target=named)
        backend = resolve_backend(choice.backend, bits, None)
        target = resolve_target(backend, bits, named, target_registry)
    except SelectionError as exc:
        raise SystemExit(f"uasm: {exc}") from None
    return choice, backend, target


def _options(args) -> Options:
    choice, backend, target_name = _select(args)
    linker_options = dict(getattr(args, "linker_options", None) or {})
    return Options(
        source=Path(args.source),
        output=Path(args.output) if getattr(args, "output", None) else None,
        frontend=choice.frontend,
        frontend_options=dict(getattr(args, "frontend_options", None) or {}),
        backend=backend,
        backend_options=dict(getattr(args, "backend_options", None) or {}),
        target=(target_registry.get(target_name) if target_name else None),
        # `build` PRODUCES ARTIFACTS; `link` PRODUCES A PROGRAM. The two
        # verbs used to be one: `build` linked unless `--emit` said not to,
        # which made "compile this" and "compile and link this" the same
        # request and gave the second no name. Now the split is the default
        # and `--link` is how a one-step build is still asked for.
        #
        # `--emit` IS KEPT AND MEANS WHAT IT ALWAYS DID, which is now also
        # what happens anyway. Every documented invocation and every script
        # that passes it keeps working, and it still OVERRIDES `--link` --
        # a user who types both has asked for artifacts last.
        #
        # `--emit-asm` IMPLIES `--emit`: assembly is not something the
        # toolchain in this driver links, and asking it to would fail after
        # the user already has the file they wanted.
        link=(getattr(args, "link", False)
              and not (getattr(args, "emit", False)
                       or getattr(args, "emit_asm", False))),
        toolchain=choice.linker,
        toolchain_chosen=choice.linker_named,
        linker_options=linker_options,
        # SEEDED FROM THE FLAG, then added to: `_link_stage` merges the
        # libraries the SOURCE asked for into this same list, which is why
        # the driver keeps a field of its own rather than reading the
        # linker's table at the point of use.
        link_inputs=tuple(linker_options.get("link-input", ())),
        workdir=Path(args.workdir) if getattr(args, "workdir", None) else None,
        keep_intermediates=getattr(args, "keep_intermediates", False),
        verbose=getattr(args, "verbose", False),
        passes=tuple(p for p in (getattr(args, "passes", "") or "").split(",") if p),
        optimise=getattr(args, "optimise", False),
        emit_ir=getattr(args, "emit_ir", False),
        emit_asm=getattr(args, "emit_asm", False),
        show_spans=getattr(args, "show_spans", False),
        verify_each=getattr(args, "verify_each", False),
        time_passes=getattr(args, "time_passes", False),
        max_errors=getattr(args, "max_errors", 100),
        warnings_are_errors=getattr(args, "werror", False),
        object_runtime=getattr(args, "object_runtime", "ir"),
    )


def cmd_build(args) -> int:
    sink = _sink(args)
    opts = _options(args)
    result = compile_source(opts, sink)
    sink.emit()
    if not result.ok:
        return 1

    if result.pass_report:
        print(result.pass_report, file=sys.stderr)

    if result.ir_text is not None:
        if opts.output:
            opts.output.write_text(result.ir_text, encoding="utf-8")
            print(f"wrote {opts.output}")
        else:
            sys.stdout.write(result.ir_text)
        return 0

    if opts.verbose:
        for cmd in result.commands:
            print("$ " + " ".join(cmd), file=sys.stderr)

    if result.program is not None:
        print(f"wrote {result.program}")
        return 0

    out = opts.output
    for name, data in sorted(result.artifacts.items()):
        # One artifact goes exactly where -o said. Several are written beside
        # it under their own names -- renaming the second onto the requested
        # path would silently overwrite the first.
        dest = out if (out and len(result.artifacts) == 1) else \
            ((out.parent / name) if out else Path(name))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"wrote {dest} ({len(data)} bytes)")
    return 0


def _truthy(raw) -> bool:
    """A `1|0` flag, spelled the way `plugin add --cwd 1|0` spells one."""
    return str(raw).lower() in ("1", "true", "yes", "on")


#: The extensions that hold IR rather than something a platform linker reads.
#: `.ir` is the text form `--emit-ir` writes and `uasm run` accepts; `.uirb`
#: is the container the `uir` backend will write.
_IR_SUFFIXES = (".ir", ".uir", ".uirb")


def cmd_link(args) -> int:
    """Join inputs into one thing, at whichever level they are already at.

    TWO STAGES WEAR THE NAME `link` and this verb reaches both, because from
    where the user stands they are one act -- "make these into one" -- and
    which one happens is decided by what they handed over, not by a flag they
    have to know to pass:

      * IR in, IR out. `uasm link a.ir b.ir -o all.ir` resolves names BETWEEN
        modules, which is the thing a single-translation-unit compiler can
        never do: see `ir/link.py` for why that is a stage and not a
        convenience.
      * Objects in, a program out. `uasm link a.o b.o -o prog` is the
        platform link, run through the same toolchain `build` would have
        used and chosen the same way.

    MIXED IS AN ERROR rather than a guess. `a.ir b.o` could mean compile the
    IR and then link both, or it could be a typo; doing the first silently
    would mean a flag nobody passed decided which backend compiled `a.ir`,
    and the answer would be visible only in the program.
    """
    sink = _sink(args)
    inputs = [Path(p) for p in args.inputs]
    ir = [p for p in inputs if p.suffix in _IR_SUFFIXES]
    rest = [p for p in inputs if p.suffix not in _IR_SUFFIXES]
    if ir and rest:
        sink.report(
            error("E9112", "cannot link IR and object files together")
            .note("IR: " + ", ".join(str(p) for p in ir))
            .note("objects: " + ", ".join(str(p) for p in rest))
            .help("build the IR first (`uasm build x.ir -o x.o --emit`), "
                  "then link the objects"))
        sink.emit()
        return 1
    rc = _link_ir(args, inputs, sink) if ir else _link_objects(args, inputs, sink)
    sink.emit()
    return rc


def _link_ir(args, inputs, sink) -> int:
    """Merge IR modules into one, and write the IR out."""
    from ..ir.link import merge

    loaded = []
    for path in inputs:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            sink.report(error("E9113", f"cannot read {path}: {exc}"))
            return 1
        try:
            loaded.append((str(path), parse_module(text)))
        except ParseError as exc:
            # HAND-EDITED IR IS THE DOCUMENTED WAY to test a backend, so
            # malformed IR is expected input rather than an internal error.
            sink.report(error("E9113", f"{path}: {exc}"))
            return 1
    module = merge(loaded, sink)
    if module is None:
        return 1
    # VERIFIED BEFORE IT IS WRITTEN. A merge can produce IR that no single
    # input contained -- a call that was external in one file and is now
    # resolved -- and writing that out unchecked hands the next stage a file
    # this one was supposed to vouch for.
    try:
        verify(module)
    except VerifyError as exc:
        d = error("E9999", "internal error: linking produced invalid IR")
        d.note("This is a bug in the compiler, not in your inputs.")
        for problem in exc.problems[:10]:
            d.note(problem)
        sink.report(d)
        return 1
    text = print_module(module, show_spans=getattr(args, "show_spans", False))
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(text)
    return 0


def _link_objects(args, inputs, sink) -> int:
    """Hand the inputs to a toolchain, exactly as the build's link stage does."""
    if not args.output:
        sink.report(
            error("E9114", "linking objects needs an output path")
            .help("name it with -o; there is no source file to name it after"))
        return 1

    from . import select as selector

    target = (target_registry.get(args.target) if args.target
              else target_registry.get("host"))
    # `choose_linker` AND NOT `choose`, because `choose` starts from a SOURCE
    # and there is not one: nothing here is compiled, so no frontend and no
    # backend are chosen. What the output spells still picks the linker
    # exactly as it does for a build -- `-o thing.so` is an extension module
    # here too -- and the builtin linker is the fallback for the same reason
    # it is there: an output with no extension of its own is a native
    # program, and making one needs no external tool.
    name = selector.choose_linker(Path(args.output), args.toolchain,
                                  link_registry,
                                  fallback=selector.DEFAULT_LINKER)
    toolchain = link_registry.get(name)
    if not toolchain.supports(target):
        sink.report(
            error("E9115",
                  f"the {toolchain.name} linker cannot produce a program "
                  f"for {target.name}"))
        return 1

    workdir = (Path(args.workdir) if args.workdir
               else Path(args.output).parent / ".uasm")
    runtime_sources: tuple[Path, ...] = ()
    if _truthy(args.runtime):
        from ..objects import support as objects_support
        # `module=None` MEANS THE WHOLE RUNTIME. The build stage narrows it to
        # what one module needs; there is no module here, so nothing can be
        # left out and the linker drops what nothing reaches.
        runtime_sources = (objects_support.write_runtime(workdir),)
    request = link_registry.LinkRequest(
        # NO ARTIFACTS OF OUR OWN: everything being linked was named on the
        # command line, which is the difference between this and `build`.
        artifacts={}, target=target, output=Path(args.output),
        workdir=workdir,
        # `input_paths` AND NOT `extra_inputs`: these are objects the user
        # named, which exist where they are. `--link-input` is still the
        # other thing -- `-l` names and libraries -- and still comes after
        # them, which is the order a linker wants.
        input_paths=tuple(inputs),
        runtime_sources=runtime_sources,
        extra_inputs=tuple(
            (getattr(args, "linker_options", None) or {})
            .get("link-input", ())),
        keep_intermediates=getattr(args, "keep_intermediates", False),
        verbose=getattr(args, "verbose", False))
    try:
        program = toolchain.link(request)
    except link_registry.LinkError as exc:
        d = error("E9104", exc.message)
        if exc.detail:
            d.note(exc.detail)
        if exc.help:
            d.help(exc.help)
        sink.report(d)
        return 1
    finally:
        if getattr(args, "verbose", False):
            for cmd in request.commands:
                print("$ " + " ".join(cmd), file=sys.stderr)
    print(f"wrote {program}")
    return 0


def cmd_run(args) -> int:
    path = Path(args.source)
    if path.suffix == ".ir":
        from ..ir import verify
        from ..ir.printer import ParseError
        try:
            module = parse_module(path.read_text(encoding="utf-8"))
        except (ParseError, OSError) as exc:
            # Hand-edited IR is the documented way to test a backend, so
            # malformed IR is expected input, not an internal error.
            print(f"cannot read IR: {exc}", file=sys.stderr)
            return 2
        try:
            verify(module)
        except VerifyError as exc:
            print("invalid IR:\n" + exc.report(), file=sys.stderr)
            return 2
    else:
        sink = _sink(args)
        opts = _options(args)
        opts.emit_ir = True             # the interpreter runs IR, not artifacts
        opts.link = False
        result = compile_source(opts, sink)
        sink.emit()
        if not result.ok:
            return 1
        module = result.module

    interp = Interpreter(module)

    # WHAT THE TRAILING WORDS ARE depends on what the entry takes, and asking
    # the entry is the only way to tell: `uasm run --entry fib prog.py 30`
    # calls `fib(30)` and always did, while `uasm run prog.py a b` should
    # do what `python prog.py a b` does. An IR function's parameters are i64,
    # so a function that declares some wants integers and a word is not one;
    # a function that declares none can only be reading them as `sys.argv`.
    #
    # `argv[0]` IS THE SOURCE AS WRITTEN, exactly as CPython's is the script
    # as written and as a compiled binary's is the path it was invoked by.
    # It is set either way -- an entry taking parameters still leaves a
    # program able to ask its own name.
    entry = module.function(args.entry)
    takes = len(entry.params) if entry is not None else 0
    interp.argv = [args.source] + ([] if takes else list(args.args))
    try:
        value = interp.run(args.entry,
                           [int(a) for a in args.args] if takes else [])
    except Trap as trap:
        print(f"trap: {trap}", file=sys.stderr)
        return 70
    if args.print_result and value is not None:
        print(f"-> {value}")
    # THE PROGRAM'S EXIT STATUS IS THE PROGRAM'S, on this path as on every
    # other. Two ways to set one and both are honoured:
    #
    #   plat_exit(n)   ends the process, and a compiled binary really does
    #   return n       from the entry, which `objects/support.py` turns into
    #                  `int main(void) { return (int)ir_main(); }`
    #
    # The second used to be dropped here, so `def main() -> int: return 7`
    # exited 7 under every compiled backend and 0 under `uasm run`. That
    # is the frontend's own documented rule -- E0009 says the return value
    # BECOMES the process exit code -- disagreeing with the interpreter that
    # is the oracle for every backend, which makes it a defect in the
    # reference rather than a quirk of it.
    #
    # ONLY FOR THE DEFAULT ENTRY. `--entry other_function` runs something that
    # is not a program, and its result is an answer rather than a status:
    # `uasm run --entry fib prog.py 30` should not exit 88 because
    # fib(30) ends in those bits. `--print-result` is how you read that.
    if interp.exit_status is not None:
        return interp.exit_status & 0xFF
    if args.entry == "main" and isinstance(value, int):
        return value & 0xFF
    return 0


def cmd_verify(args) -> int:
    """`build` with everything after the frontend taken off.

    IT IS A BUILD AND NOT A SEPARATE ANALYSIS. The same frontend, the same
    flags, the same imports resolved the same way -- because the one thing
    this must never do is say a program is fine and then have `build` refuse
    it. So it runs the whole front half and the IR verifier, and stops where
    the backend would start.

    THE FRONTEND IS TOLD, through `BuildContext.verifying`, so it can skip
    work whose only consumer is a stage that will not run. Nothing that could
    change a diagnostic may be skipped -- that would be the same lie by a
    shorter route -- and what it costs to say so is one bool.
    """
    sink = _sink(args)
    opts = _options(args)
    opts.emit_ir = True                     # stop before any backend
    opts.link = False
    opts.verifying = True
    result = compile_source(opts, sink)
    if getattr(args, "json", False):
        # THE DIAGNOSTICS AS DATA, for an editor or a CI job. Printed
        # INSTEAD of the rendered form rather than beside it: two renderings
        # of one run on one stream is not something a reader or a parser can
        # use.
        print(_diagnostics_as_json(sink, result))
        return 0 if not sink.failed else 1
    sink.emit()
    if not result.ok:
        return 1
    stats = result.module.statistics()
    print("ok: " + ", ".join(f"{v} {k}" for k, v in stats.items() if v))
    return 0


def _diagnostics_as_json(sink, result) -> str:
    """This run, as one JSON object.

    THE SHAPE IS THE DIAGNOSTIC'S OWN and not a flattened string: `code`,
    `severity`, `message`, the notes and helps separately, and a position
    when there is one. A tool that wanted the rendered text could have run
    without `--json`; what it cannot reconstruct from that text is which
    code was reported, and that is the part a CI job filters on.

    POSITIONS ARE 1-BASED LINE AND COLUMN, as the rendered form prints them,
    and `bytes` carries the half-open byte range for an editor that wants to
    highlight exactly what the compiler pointed at.
    """
    import json

    def place(span):
        if not is_real(span):
            return None
        start, end = span.start_loc, span.end_loc
        return {"file": span.file.name,
                "line": start.line, "column": start.column,
                "end_line": end.line, "end_column": end.column,
                "bytes": [span.start, span.end]}

    items = []
    for d in sorted(sink.diagnostics, key=lambda x: x.sort_key()):
        items.append({
            "code": d.code,
            "severity": d.severity.label,
            "message": d.message,
            "at": place(d.primary_span),
            "notes": list(d.notes),
            "helps": list(d.helps),
            # EVERY LABEL AND NOT JUST THE PRIMARY ONE: a secondary label is
            # where the OTHER operand was, or where the name was defined
            # first, and dropping it leaves half the explanation behind.
            "labels": [{"at": place(label.span), "message": label.message,
                        "primary": label.primary} for label in d.labels],
        })
    return json.dumps({
        "ok": not sink.failed,
        "errors": sink.error_count,
        "warnings": sink.warning_count,
        "diagnostics": items,
        # WHAT WAS VERIFIED, when there is anything: a run that reported
        # nothing and compiled nothing is not the same as one that checked a
        # program, and the counts say which happened.
        "statistics": (result.module.statistics() if result.module is not None
                       else None),
    }, indent=2)


def cmd_ops(args) -> int:
    print(f"{len(opcodes.Op)} opcodes. `ty` is the width the opcode operates "
          f"at; for a comparison it is the OPERAND type (the result is i1).\n")
    width = max(len(o.value) for o in opcodes.Op)
    for op in opcodes.Op:
        s = opcodes.spec(op)
        arity = "*" if s.arity is None else str(s.arity)
        mark = "  [terminator]" if s.terminator else ""
        allowed = "any" if not s.allowed else " ".join(t.name for t in s.allowed)
        print(f"  {op.value:<{width}}  args={arity:<2} -> {s.result:<4}{mark}")
        print(f"  {'':<{width}}  types: {allowed}")
        for line in _wrap(s.doc, 64):
            print(f"  {'':<{width}}  {line}")
        print()
    return 0


def cmd_types(args) -> int:
    for name, ty in T.ALL.items():
        if ty.is_void:
            print(f"  {name:<5} no value")
            continue
        kind = ("float" if ty.is_float else "pointer" if ty.is_ptr
                else "signed integer" if ty.is_signed else "unsigned integer")
        print(f"  {name:<5} {ty.bits:>3} bits, {ty.size} byte(s)   {kind}")
    return 0


def _column(items, floor: int = 10) -> int:
    """How wide the name column has to be.

    Computed rather than fixed, because a fixed width is only ever right for
    the names that shipped. Every built-in fits in ten characters and a
    third party's `counting-link` does not, so the first registration from
    outside this repository broke the alignment -- a small thing that is also
    the only kind of bug an extension point can have that nobody in-tree sees.
    """
    return max([floor, *(len(name) for name, _ in items)])


def _print_options(options, width: int) -> None:
    """List a component's own flags under its line.

    WITH THE COMPONENT rather than only in `build --help`, where they sit
    among thirty options that apply to every build and give no hint which one
    they belong to. Shared by all three listings now that all three kinds can
    declare flags -- `uasm plugin frontends` said nothing about
    `--import-path`, which is the Python frontend's and nobody else's.
    """
    for option in options:
        spelling = option.flag
        if option.short:
            spelling = f"-{option.short}, {spelling}"
        if not option.switch:
            spelling = f"{spelling} {option.metavar}"
        print(f"  {'':<{width}}   {spelling}")
        for line in _wrap(option.help, 60):
            print(f"  {'':<{width}}     {line}")


def cmd_backends(args) -> int:
    backend_registry.load_builtin()
    items = sorted(backend_registry.available().items())
    width = _column(items)
    for name, be in items:
        flag = "" if be.ready else "   (unfinished)"
        print(f"  {name:<{width}} {be.description}{flag}")
        _print_options(be.options, width)
    # THE FAMILIES AFTER THE BACKENDS, and marked as not being backends. They
    # are selectable with --backend and are not code generators, so listing
    # them among the others would make `uasm plugin backends` show six entries
    # for four compilers; leaving them out entirely would hide a name the
    # help text tells the user to type.
    from ..backend.families import DEFAULT_BITS, FAMILIES
    print()
    print("  families -- select one with --backend and choose the width "
          "with --bits:")
    for family, members in sorted(FAMILIES.items()):
        spelled = ", ".join(f"{bits}: {name}"
                            for bits, name in sorted(members.items()))
        print(f"  {family:<{width}} {spelled}   (default {DEFAULT_BITS})")
    return 0


def cmd_libraries(args) -> int:
    """Which installed packages a build would resolve against, and from where.

    THE POINT OF PRINTING IT. A library point is a directory the user never
    typed, holding whatever pip last put there. "Why did `import yaml` work on
    my machine and not in CI" has one honest answer and this is where it is.
    """
    from ..frontends.python import hostlib
    host = hostlib.discover(getattr(args, "host_python", None))
    if host.unavailable:
        print(f"  no library point: {host.unavailable}")
        return 1
    print(f"  interpreter  {host.executable or '(this one)'}")
    print(f"  version      Python {host.version}")
    if host.prefix:
        print(f"  prefix       {host.prefix}")
    print()
    if not host.points:
        print("  no site-packages directory exists for that interpreter")
        return 0
    print("  searched after the bundled standard library, the source's own "
          "directory")
    print("  and every --import-path, in this order:")
    print()
    width = max(len(p.kind) for p in host.points)
    for point in host.points:
        count = _distribution_count(point.path)
        print(f"  {point.kind:<{width}}  {point.path}"
              + (f"   ({count} installed)" if count is not None else ""))
    print()
    print("  compiled extension modules are found and REFUSED with E0129: "
          "they are")
    print("  native binaries built against CPython, and there is no source "
          "to splice.")
    return 0


def _distribution_count(point) -> int | None:
    """How many distributions live on one point, or None if unreadable.

    Counted from `.dist-info` directories rather than by importing anything:
    the point may belong to another interpreter entirely, and reading its
    metadata with THIS one's machinery is exactly the mistake `hostlib`
    refuses to make elsewhere.
    """
    try:
        return sum(1 for entry in point.iterdir()
                   if entry.name.endswith((".dist-info", ".egg-info")))
    except OSError:
        return None


def cmd_targets(args) -> int:
    targets = target_registry.available()
    aliases: dict[str, list[str]] = {}
    for alias, canonical in target_registry.aliases().items():
        aliases.setdefault(canonical, []).append(alias)
    width = max((len(n) for n in targets), default=4)
    for name, t in sorted(targets.items()):
        alias = (f"   aka {', '.join(sorted(aliases.get(name, [])))}"
                 if aliases.get(name) else "")
        print(f"  {name:<{width}}  {t.arch}/{t.os}  abi={t.abi} "
              f"format={t.object_format}{alias}")
    print()
    print(f"  {'host':<{width}}  resolves to this machine "
          f"({target_registry.host().name}); the default for a machine backend")
    return 0


def cmd_toolchains(args) -> int:
    link_registry.load_builtin()
    items = sorted(link_registry.available().items())
    width = _column(items)
    for name, tc in items:
        print(f"  {name:<{width}} {tc.description}")
        _print_options(tc.options, width)
    return 0


def cmd_port(args) -> int:
    """How far the object runtime has moved to IR, and what is next.

    THE READY LIST IS THE POINT. Anything on it can be ported without deciding
    anything new: every function it calls is already in IR, and every libc
    function it calls is one the machine subset can write out. Everything else
    is reported as what stands in the way, so the next piece of work is a
    blocker rather than a guess.

    See `objects/ir.py` for why this asks about CLOSURE rather than about
    `static` helpers, and why a call to libc counts as a callee.
    """
    from ..objects.ir import survey

    s = survey()
    # THE TOTAL IS EVERY EXPORTED FUNCTION, which this used to get wrong: it
    # added the fully-replaced count to the untouched one and called that the
    # total, so the split functions were in neither and the runtime looked
    # smaller than it is.
    total = s["replaced"] + s["untouched"]
    print(f"object runtime: {s['replaced']} of {total} exported functions "
          f"in IR, {s['split']} of those split -- {s['untouched']} still C")
    print()
    if s["ready"]:
        print("ready to port -- closed over what IR already defines:")
        for name in s["ready"]:
            print(f"  {name}")
    else:
        print("nothing is ready: every remaining function needs something "
              "that is not in IR yet")
    if s["wont"]:
        print()
        print("ruled out, and not proposed again:")
        for name, why in sorted(s["wont"].items()):
            print(f"  {name}")
            print(f"      {why}")
    if s["near"]:
        print()
        print("one or two away -- what each is waiting on:")
        width = max(len(n) for n, _ in s["near"])
        for name, needs in s["near"]:
            print(f"  {name:<{width}}  {', '.join(needs)}")
    print()
    print("what stands in the way, by how many functions wait on it:")
    for name, count in s["blockers"]:
        why = s["reasons"].get(name)
        print(f"  {count:4d}  {name}" + (f"   {why}" if why else ""))
    return 0


def cmd_frontends(args) -> int:
    frontend_registry.load_builtin()
    items = sorted(frontend_registry.available().items())
    width = _column(items)
    for name, fe in items:
        print(f"  {name:<{width}} {fe.description:<42} {' '.join(fe.extensions)}")
        _print_options(fe.options, width)
    return 0


def cmd_passes(args) -> int:
    print(f"default pipeline for -O: {', '.join(DEFAULT_PASSES)}\n")
    items = sorted(available_passes().items())
    width = _column(items)
    for name, p in items:
        print(f"  {name:<{width}} {p.description}")
        tags = []
        for label, s in (("requires", p.requires), ("provides", p.provides),
                         ("invalidates", p.invalidates)):
            if s:
                tags.append(f"{label}={','.join(sorted(s))}")
        if tags:
            print(f"  {'':<12} {'  '.join(tags)}")
    return 0


def _flag(value: str | None, default: bool) -> bool:
    """`--pip 1` / `--pip 0`, with the flag absent meaning the default.

    Spelled as an explicit value rather than `--pip`/`--no-pip` because these
    three switch a SEARCH ORDER, and a reader of a script wants to see which
    sources were on without knowing what the defaults were that week.
    """
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _sources(args) -> plugins.Sources:
    return plugins.Sources(
        cwd=_flag(getattr(args, "cwd", None), True),
        pypath=_flag(getattr(args, "pypath", None), True),
        pip=_flag(getattr(args, "pip", None), False),
    )


def _confirm_replace(name: str, entry, args) -> bool:
    """Ask before replacing an installed plugin. True to go ahead.

    `--yes`/`--no` answer it without asking, which is what a script needs.
    With neither, a NON-INTERACTIVE run refuses rather than prompting: a
    build that blocks forever on a hidden question is worse than one that
    stops and says which flag to pass.
    """
    if getattr(args, "assume_no", False):
        return False
    if getattr(args, "assume_yes", False):
        return True
    if not sys.stdin.isatty():
        print(f"uasm: {name!r} is already installed "
              f"({entry.source or 'unknown'}: {entry.origin or '?'})",
              file=sys.stderr)
        print("  pass --yes to replace it, or --no to leave it alone",
              file=sys.stderr)
        return False
    print(f"{name!r} is already installed "
          f"({entry.source or 'unknown'}: {entry.origin or '?'})")
    for kind, names in (entry.provides or {}).items():
        if names:
            print(f"  {kind}: {', '.join(names)}")
    try:
        reply = input("replace it? [y/N] ").strip().lower()
    except EOFError:
        # isatty() is not a reliable "someone is there": a subprocess inherits
        # its parent's stdin, so a build runner can look interactive and still
        # have nothing to read. EOF means no answer, which means no.
        print("no answer; leaving it alone "
              "(pass --yes to replace without asking)", file=sys.stderr)
        return False
    return reply in ("y", "yes")


def cmd_plugin_invalidate(args) -> int:
    """Re-resolve installed plugins from where they came from."""
    if getattr(args, "all", False):
        names = [e.name for e in plugins.installed()]
        if not names:
            print("no plugins installed")
            return 0
    else:
        # Comma-separated OR repeated OR a single bare name -- all three are
        # the same thing to anyone typing it, so all three work.
        names = [n for chunk in (args.names or []) for n in chunk.split(",") if n]
        if not names:
            print("uasm: name an id, or pass --all", file=sys.stderr)
            return 2
    failed = 0
    for name in names:
        try:
            entry = plugins.invalidate([name])[0]
        except plugins.PluginError as exc:
            print(f"uasm: {exc}", file=sys.stderr)
            failed += 1
            continue
        print(f"invalidated {entry.name}  ({entry.source}: {entry.origin})")
        for kind, provided in (entry.provides or {}).items():
            if provided:
                print(f"  {kind:<10} {', '.join(provided)}")
    return 1 if failed else 0


def cmd_plugin_add(args) -> int:
    sources = _sources(args)
    existing = plugins.store.find(args.name)
    replace = False
    if existing is not None:
        if not _confirm_replace(args.name, existing, args):
            print("cancelled; nothing changed")
            return 1
        replace = True
    try:
        entry = plugins.install(args.name, sources, replace=replace)
    except plugins.AlreadyInstalled:
        # Only reachable if something installed the same name between the
        # check above and here.
        print(f"uasm: {args.name!r} is already installed", file=sys.stderr)
        return 2
    except plugins.PluginError as exc:
        print(f"uasm: {exc}", file=sys.stderr)
        print(f"  searched: {', '.join(sources.enabled()) or 'nothing'}",
              file=sys.stderr)
        if not sources.pip:
            print("  try --pip 1 to install it from an index",
                  file=sys.stderr)
        return 2
    print(f"added {entry.name}  ({entry.source}: {entry.origin})")
    contents = {k: v for k, v in entry.provides.items() if v}
    if contents:
        for kind, names in contents.items():
            print(f"  {kind:<10} {', '.join(names)}")
    else:
        # A module with no manifest registers on import; nothing here can say
        # what it added without diffing the registries, and claiming it added
        # nothing would be worse than saying so.
        print("  registered on import; declare __uasm_plugin__ to list "
              "its contents")
    print(f"stored in {plugins.store.config_file()}")
    return 0


def cmd_plugin_remove(args) -> int:
    if not plugins.uninstall(args.name):
        print(f"uasm: {args.name!r} is not installed", file=sys.stderr)
        return 2
    print(f"removed {args.name}")
    return 0


def cmd_plugin_list(args) -> int:
    entries = plugins.installed()
    if not entries:
        print("no plugins installed")
        print(f"  ({plugins.store.config_file()})")
        return 0
    width = max(len(e.name) for e in entries)
    for e in entries:
        print(f"  {e.name:<{width}}  {e.source or 'unknown'}: "
              f"{e.origin or '?'}")
        for kind, names in e.provides.items():
            if names:
                print(f"  {'':<{width}}  {kind}: {', '.join(names)}")
    return 0


def cmd_plugin_show(args) -> int:
    """What a module provides, without registering any of it."""
    try:
        found = plugins.resolve(args.name, _sources(args))
    except plugins.ResolveError as exc:
        print(f"uasm: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:                    # noqa: BLE001 -- reported
        # The module was found and raised on import. That is a bug in the
        # user's plugin, and a compiler traceback for it reads as a bug in
        # uasm.
        print(f"uasm: plugin {args.name!r} raised while importing: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    plugin = plugins.manifest_of(found.module)
    print(f"{args.name}  ({found.source}: {found.origin})")
    if plugin is None:
        print("  no __uasm_plugin__; this module registers on import")
        return 0
    if plugin.name or plugin.version:
        print(f"  {plugin.name} {plugin.version}".rstrip())
    if plugin.description:
        print(f"  {plugin.description}")
    for kind, names in plugin.contents().items():
        if names:
            print(f"  {kind:<10} {', '.join(names)}")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


class _CollectComponentOption(argparse.Action):
    """Store one component's flag into that component's own table.

    ONE TABLE PER KIND, not one table for everything: the driver hands a
    backend's `configure` the options a BACKEND declared, and treats anything
    else in that table as a flag the backend does not take (E9106). A
    frontend's flag arriving in the backend's table would be reported as the
    backend's mistake.

    KINDS AND NOT A KIND, because one NAME can span two of them -- `cpyext` is
    a backend and a linker both -- and `--cpyext:module-name` is one spelling
    whoever is reading it. If both halves declare the name, both are handed
    the value; the pipeline gives each component only what that component
    declared, so the half that does not want it never sees it.

    KEYED BY WHAT THE COMPONENT DECLARED, bound here rather than read back off
    the flag text, so `--opt-level`, `--pybc:opt-level` and a short letter all
    land on the same key -- the component never learns which spelling was
    used, because the spellings are for the parser and not for it.
    """

    def __init__(self, option_strings, dest, *, kinds, option, **kw):
        super().__init__(option_strings, dest, **kw)
        self.kinds = kinds
        # THE WHOLE DECLARATION and not its name alone: what to do with the
        # value is the option's own business -- a switch is its own value, a
        # repeatable one appends -- and carrying the `Option` means a shape
        # added to it is not also a constructor argument added here.
        self.option = option

    def __call__(self, parser, namespace, value, option_string=None):
        for kind in self.kinds:
            table = getattr(namespace, f"{kind}_options", None)
            if table is None:
                table = {}
                setattr(namespace, f"{kind}_options", table)
            if self.option.switch:
                # A SWITCH IS ITS OWN VALUE. `nargs=0` hands this `[]`, and
                # storing that would make `--library` read as falsey to a
                # component that asked whether it was given.
                table[self.option.name] = True
            elif self.option.repeat:
                # A REPEATABLE FLAG KEEPS EVERY VALUE, and in the order it was
                # typed: a search path is a list and its order is its meaning.
                table.setdefault(self.option.name, []).append(value)
            else:
                table[self.option.name] = value


class _AmbiguousOption(argparse.Action):
    """A spelling more than one component declares.

    REFUSED RATHER THAN AWARDED TO ONE. Registration order is alphabetical and
    looks deliberate, which is exactly the kind of accident a reader would
    believe -- and the wrong component would then be configured silently. The
    qualified spellings say which was meant, and they are always registered.

    CARRIES THE SPELLINGS AND NOT A NAME, because a contested SHORT letter is
    not a contested word: `-P` might be `--safe-path` to one component and
    `--pedantic` to another, and telling the user to write `--x:P` would name
    a flag that does not exist.
    """

    def __init__(self, option_strings, dest, *, owners, spellings, **kw):
        super().__init__(option_strings, dest, **kw)
        self.owners = owners
        self.spellings = spellings

    def __call__(self, parser, namespace, value, option_string=None):
        parser.error(f"{option_string} is ambiguous: "
                     f"{', '.join(self.owners)} all declare it. Write "
                     + " or ".join(self.spellings) + ".")


_ALL_KINDS = ("backend", "frontend", "linker")


def _add_component_options(parser: argparse.ArgumentParser,
                           kinds: tuple[str, ...] = _ALL_KINDS) -> None:
    """Give `parser` every registered component's own flags.

    EVERY COMPONENT, NOT JUST THE BACKENDS. A backend has declared its own
    flags since the beginning; a frontend and a linker could not, so the flags
    they needed sat on the driver's parser instead -- `--import-path` was
    offered to a build using any frontend, and `--link-input` to one that
    links nothing and could not say so.

    ALL of them, not just the selected one's: `--backend` is parsed by the
    same pass that would have to know the answer, and a parser that rejected
    `--class-version` before reading `--backend jvm` would depend on the order
    the flags were typed in. Passing one to a component that does not declare
    it is caught in the driver, which by then knows what was chosen and can
    say who does take it.

    `kinds` NARROWS WHICH STAGES A VERB HAS. `run` and `check` stop at the
    IR, so a backend's flags on them would be flags for a stage that never
    happens -- and, worse, could make a frontend's flag ambiguous against one
    nothing on that command line could ever reach.
    """
    frontend_registry.load_builtin()
    link_registry.load_builtin()

    # KEYED BY (OWNER NAME, OPTION NAME) and not by kind, so the two halves of
    # a name that spans kinds collapse into the one flag a user would expect
    # rather than colliding inside argparse.
    declarations: dict[tuple[str, str], tuple[list[str], Option]] = {}
    registries = {"backend": backend_registry, "frontend": frontend_registry,
                  "linker": link_registry}
    for kind in kinds:
        for name, component in sorted(registries[kind].available().items()):
            for option in component.options:
                owned, _ = declarations.setdefault(
                    (name, option.name), ([], option))
                owned.append(kind)

    # WHO CLAIMS WHAT. Words and letters are counted apart: two components can
    # want one letter for two different words, and losing the letter to that
    # tie should not cost either of them the word.
    words: dict[str, list[tuple[str, str]]] = {}
    letters: dict[str, list[tuple[str, str]]] = {}
    for name, flag in declarations:
        _, option = declarations[name, flag]
        words.setdefault(flag, []).append((name, flag))
        if option.short:
            letters.setdefault(option.short, []).append((name, flag))

    group = parser.add_argument_group("component options")

    def offer(spellings: list[str], option: Option, owned: list[str],
              help: str) -> None:
        extra = ({"nargs": 0} if option.switch
                 else {"metavar": option.metavar})
        group.add_argument(*spellings, action=_CollectComponentOption,
                           kinds=tuple(owned), option=option,
                           dest=argparse.SUPPRESS, **extra,
                           # SAID IN THE HELP because it changes what the
                           # flag MEANS: giving `--import-path` twice adds a
                           # second directory, where giving `--host-python`
                           # twice replaces the first.
                           help=help + (" (repeatable)" if option.repeat
                                        else ""))

    def agreed(claims: list[tuple[str, str]]):
        """The one option every claimant declared, or None if they differ.

        SEVERAL COMPONENTS DECLARING ONE FLAG IS NOT A COLLISION when what
        they declared is the same declaration: `cc`, `cpyext` and `baremetal`
        share `--link-input` because it means the same thing to all three,
        and refusing it as ambiguous would be telling the user to choose
        between three spellings of one flag. `Option` is a frozen dataclass,
        so "the same" is its own value -- a differing help text is a
        differing flag, which errs towards asking.
        """
        declared = {declarations[name, flag][1] for name, flag in claims}
        return declared.pop() if len(declared) == 1 else None

    def refuse(spelling: str, claims: list[tuple[str, str]]) -> None:
        # TWO COMPONENTS WANTING ONE SPELLING IS A QUESTION, not something to
        # settle by registration order -- the same shape of refusal an
        # ambiguous `-o` extension gets. `nargs="?"` so that the bare flag
        # stops here too, rather than argparse complaining about a missing
        # value before anyone gets to say the spelling was ambiguous.
        owners = [o for o, _ in claims]
        qualified = [f"--{o}:{n}" for o, n in claims]
        group.add_argument(
            spelling, action=_AmbiguousOption, owners=owners,
            spellings=qualified, nargs="?", dest=argparse.SUPPRESS,
            metavar="VALUE",
            help=("[" + ", ".join(owners) + "] ambiguous; write "
                  + " or ".join(qualified)))

    for (name, flag), (owned, option) in declarations.items():
        # THE QUALIFIED SPELLING IS ALWAYS REGISTERED, collision or not, so a
        # script written against `--pybc:opt-level` keeps working when a
        # plugin later claims the short name out from under it.
        offer([option.qualified(name)], option, owned,
              f"[{name} {'/'.join(owned)}] {option.help}")
    def kinds_of(claims):
        return sorted({k for name, flag in claims
                       for k in declarations[name, flag][0]})

    def owners_of(claims):
        return ", ".join(sorted({name for name, _ in claims}))

    for flag, claims in sorted(words.items()):
        option = agreed(claims)
        if option is None:
            refuse("--" + flag, sorted(claims))
            continue
        # THE LETTER RIDES WITH THE WORD when both are uncontested, so
        # `--help` prints `-P, --safe-path` the way every other flag here is
        # printed rather than listing the two as unrelated entries.
        rides = (option.short is not None
                 and agreed(letters.get(option.short, ())) is option)
        offer(([f"-{option.short}"] if rides else []) + [option.flag],
              option, kinds_of(claims),
              f"[{owners_of(claims)}] {option.help}")
    for letter, claims in sorted(letters.items()):
        option = agreed(claims)
        if option is None:
            refuse("-" + letter, sorted(claims))
            continue
        if agreed(words[option.name]) is option:
            continue                        # it rode with the word, above
        # THE WORD WAS CONTESTED AND THE LETTER WAS NOT, so the letter is the
        # only unqualified way in -- which is fine, and would not be if it
        # silently meant the other claimant's flag.
        offer(["-" + letter], option, kinds_of(claims),
              f"[{owners_of(claims)}] {option.help}")


def build_parser() -> argparse.ArgumentParser:
    # Backends load before the parser is built, not before it runs: each one
    # contributes its own flags, and argparse has to know a flag exists before
    # it can accept it. Plugins were loaded earlier still, in `main`, for the
    # same reason -- a third-party backend's options are not second class.
    backend_registry.load_builtin()
    ap = argparse.ArgumentParser(prog="uasm", description=__doc__.split("\n")[0])
    # Two dests, one flag. A subparser's argument OVERWRITES the namespace
    # attribute rather than appending to it, so sharing `dest` would make
    # `uasm --plugin a build x.py --plugin b` load only `b` -- silently,
    # and only when both positions are used.
    ap.add_argument("--plugin", action="append", default=[], metavar="MODULE",
                    dest="plugin_before",
                    help="import MODULE before running, so its backends, "
                         "targets, toolchains and frontends register "
                         "(repeatable; also accepted after the command)")
    sub = ap.add_subparsers(dest="command", required=True)

    def source_args(p):
        # `--import-path`, `-P`, `--no-site-packages`, `--host-python`,
        # `--native-library` and `--library` USED TO BE HERE, as did the C
        # frontend's `-I` and `-D`. Every one of them is about resolving
        # names in, or compiling, ONE language, so every one belongs to its
        # frontend: see `PythonFrontend.options` and the C frontend's, and
        # `_add_component_options` for how a component's flags reach a
        # parser. Registered per verb rather than here, because which STAGES
        # a verb has decides which kinds of component it can be given flags
        # for -- `run` and `check` stop at the IR.
        p.add_argument("source")
        p.add_argument("--frontend")
        p.add_argument("--max-errors", type=int, default=100)
        p.add_argument("--werror", action="store_true",
                       help="treat warnings as errors")

    def pass_args(p):
        p.add_argument("-O", "--optimise", action="store_true",
                       help=f"run the default pipeline ({', '.join(DEFAULT_PASSES)})")
        p.add_argument("--passes", help="comma-separated pass names")
        p.add_argument("--verify-each", action="store_true",
                       help="verify after every pass; names the pass that broke it")
        p.add_argument("--time-passes", action="store_true")
        # WHERE THE OBJECT RUNTIME COMES FROM. Part of it is written in
        # uasm's own machine subset and compiled in (`objects/ir.py`);
        # `c` uses the hand-written C for all of it, as every build did before
        # any of it was ported. Both are supported: the reason to write it in
        # IR is that a backend should not HAVE to define 229 functions, which
        # is an argument for making the C unnecessary rather than unavailable.
        p.add_argument("--object-runtime", choices=("ir", "c"), default="ir",
                       help="ir: compile the ported runtime from source and "
                            "splice it in (default); c: use the C runtime for "
                            "all of it")

    b = sub.add_parser("build", help="compile to backend artifacts")
    source_args(b)
    pass_args(b)
    b.add_argument("-o", "--output")
    b.add_argument("-bk", "--backend", default=None,
                   help="code generator, or a family: `x86` and `arm` pick "
                        "their member from --bits. Chosen from the output's "
                        "extension when not given")
    b.add_argument("--bits", type=int, choices=(32, 64), default=None,
                   help="word size to emit for. Selects within a backend "
                        "family, and is checked against --backend and "
                        "--target when those already imply one")
    b.add_argument("--target",
                   help="platform to emit for; see `uasm plugin targets`")
    b.add_argument("--link", action="store_true",
                   help="also link the artifacts into a program. Without it "
                        "`build` writes unlinked objects and `uasm link` "
                        "turns them into a program")
    b.add_argument("--emit", action="store_true",
                   help="write backend artifacts and stop; do not link. The "
                        "default, and kept because it says so explicitly")
    b.add_argument("--emit-asm", action="store_true",
                   help="write the backend's assembly instead of its object "
                        "file, and stop. For reading what was generated; a "
                        "backend whose artifact is already readable refuses")
    b.add_argument("-ln", "--linker", "--toolchain", dest="toolchain",
                   default=None,
                   help="how to turn artifacts into a program "
                        "(see `uasm plugin linkers`). Chosen from the "
                        "output's extension when not given")
    # `--link-input` USED TO BE HERE, and was offered for `jar` and `pyc`
    # too -- neither of which links anything, and neither of which could say
    # so. It is declared by the three toolchains that do link; see
    # `link/toolchains.py`.
    b.add_argument("--workdir", help="where intermediates go (default .uasm)")
    b.add_argument("--keep-intermediates", action="store_true")
    b.add_argument("-v", "--verbose", action="store_true",
                   help="print the external commands that were run")
    b.add_argument("--emit-ir", action="store_true")
    b.add_argument("--show-spans", action="store_true",
                   help="annotate each instruction with its source position")
    # EVERY KIND, because `build` is the verb that has every stage.
    _add_component_options(b)
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("run", help="execute in the reference interpreter")
    source_args(r)
    pass_args(r)
    r.add_argument("--entry", default="main")
    r.add_argument("args", nargs="*")
    r.add_argument("--print-result", action="store_true")
    # THE FRONTEND'S FLAGS AND NOT THE BACKEND'S: `run` stops at the IR, so
    # `--class-version` on it would name a stage this command never reaches.
    _add_component_options(r, kinds=("frontend",))
    r.set_defaults(fn=cmd_run)

    ln = sub.add_parser(
        "link", help="join IR modules, or objects, into one thing")
    ln.add_argument("inputs", nargs="+", metavar="INPUT",
                    help="IR files (.ir, .uirb) to merge, or objects and "
                         "archives to link. Not both: see the refusal")
    ln.add_argument("-o", "--output",
                    help="where the result goes; required when linking "
                         "objects, and IR goes to stdout without it")
    ln.add_argument("-ln", "--linker", "--toolchain", dest="toolchain",
                    default=None,
                    help="how to turn objects into a program (see `uasm "
                         "plugin linkers`). Chosen from the output's "
                         "extension when not given")
    ln.add_argument("--target",
                    help="platform to link for; see `uasm plugin targets`")
    # THE ENTRY SHIM AND THE OBJECT RUNTIME, off by default.
    #
    # OFF, because `link` means "link what I gave you" and anything else is a
    # guess about where the objects came from. ON in one word, because
    # objects from `uasm build --emit` cannot link without it: the IR's
    # `main` is emitted as `uasm_main` -- it returns i64 where C requires int
    # -- so nothing defines the `main` the platform's start files call.
    ln.add_argument("--runtime", metavar="1|0", default="0",
                    help="also link the object runtime and the `int main` "
                         "shim that calls uasm_main (default 0). Objects "
                         "from `uasm build --emit` need it; foreign objects "
                         "must not have it")
    ln.add_argument("--workdir", help="where intermediates go (default .uasm)")
    ln.add_argument("--keep-intermediates", action="store_true")
    ln.add_argument("-v", "--verbose", action="store_true",
                    help="print the external commands that were run")
    ln.add_argument("--show-spans", action="store_true",
                    help="annotate each instruction with its source position")
    # THE LINKERS' FLAGS AND NOBODY ELSE'S: nothing is compiled here, so a
    # frontend's flag or a backend's would name a stage this verb has not got.
    _add_component_options(ln, kinds=("linker",))
    ln.set_defaults(fn=cmd_link)

    # `check` UNTIL NOW, and renamed because the two words mean different
    # things: `check` reads as a lint, and this is a BUILD with everything
    # after the frontend taken off -- same frontend, same flags, same
    # imports. What it promises is that a program it passes is one `build`
    # accepts, which is a stronger claim than "looks fine".
    v = sub.add_parser("verify",
                       help="compile and verify, produce nothing")
    source_args(v)
    pass_args(v)
    _add_component_options(v, kinds=("frontend",))
    v.add_argument("--json", action="store_true",
                   help="print the diagnostics as JSON instead of rendering "
                        "them, for an editor or a CI job")
    v.set_defaults(fn=cmd_verify)

    pl = sub.add_parser("plugin",
                        help="install plugins, and describe this installation")
    pls = pl.add_subparsers(dest="plugin_command", required=True)

    def where(p):
        """The three sources, as explicit 1/0 values."""
        p.add_argument("--cwd", metavar="1|0",
                       help="look for NAME.py or NAME/ here (default 1)")
        p.add_argument("--pypath", metavar="1|0",
                       help="import NAME from the Python path (default 1)")
        p.add_argument("--pip", metavar="1|0",
                       help="pip install NAME first (default 0)")

    a = pls.add_parser("add", help="install a plugin and remember it")
    a.add_argument("name")
    where(a)
    a.add_argument("--yes", "-y", dest="assume_yes", action="store_true",
                   help="replace an already-installed plugin without asking")
    a.add_argument("--no", "-n", dest="assume_no", action="store_true",
                   help="never replace; fail instead")
    a.set_defaults(fn=cmd_plugin_add)

    inv = pls.add_parser(
        "invalidate",
        help="re-resolve installed plugins from where they came from")
    inv.add_argument("names", nargs="*", metavar="ID",
                     help="plugin ids, comma-separated or repeated")
    inv.add_argument("--all", action="store_true",
                     help="every installed plugin")
    inv.set_defaults(fn=cmd_plugin_invalidate)

    rm = pls.add_parser("remove", help="forget an installed plugin")
    rm.add_argument("name")
    rm.set_defaults(fn=cmd_plugin_remove)

    ls = pls.add_parser("list", help="what is installed")
    ls.set_defaults(fn=cmd_plugin_list)

    sh = pls.add_parser("show", help="what a module provides, registering none of it")
    sh.add_argument("name")
    where(sh)
    sh.set_defaults(fn=cmd_plugin_show)

    # ── what this installation is made of ──────────────────────────────────
    #
    # NINE VERBS OF THEIR OWN, until now. `uasm backends`, `uasm ops` and
    # seven more sat beside `build` and `run`, as though listing something
    # were the same kind of act as compiling. They are here because every one
    # of them answers a question about the INSTALLATION rather than about a
    # program, and because a plugin is the reason the answer can differ
    # between two machines: five of them print a registry a plugin extends,
    # two print the IR contract a plugin backend is written against, and two
    # describe the installation a plugin lands in.
    #
    # `plugin list` STAYS THE PLUGINS THEMSELVES, and these do not take its
    # place: "what did I install" and "what can this compiler do" are
    # different questions, and a plugin's whole point is that the second
    # answer changes when the first does.
    for name, fn, doc, aliases in (
        ("backends", cmd_backends, "code generators, and the flags each takes",
         ()),
        ("frontends", cmd_frontends, "languages, and the flags each takes", ()),
        # THE REGISTRY'S OWN WORD IS `toolchain` and the flag is `--linker`,
        # so both are accepted here for the same reason `-ln` accepts
        # `--toolchain`: one thing, and nobody should have to remember which
        # word this particular command wanted.
        ("linkers", cmd_toolchains, "ways of turning artifacts into a program",
         ("toolchains",)),
        ("targets", cmd_targets, "target platforms, with their aliases", ()),
        ("passes", cmd_passes, "optimisation passes", ()),
        ("ops", cmd_ops, "the IR's instruction set", ()),
        ("types", cmd_types, "the IR's type system", ()),
        ("port", cmd_port, "how far the object runtime has moved to IR", ()),
    ):
        p = pls.add_parser(name, help=doc, aliases=aliases)
        p.set_defaults(fn=fn)

    # Not in the loop above: this is the one noun whose answer depends on
    # WHICH interpreter is asked, so it takes the same flag `build` does.
    lib = pls.add_parser("libraries",
                         help="where installed packages are resolved from")
    lib.add_argument("--host-python", metavar="PATH",
                     help="the interpreter to ask; default is the one "
                          "running the compiler")
    lib.set_defaults(fn=cmd_libraries)

    # argparse accepts a parser-level flag only BEFORE the subcommand, and
    # `uasm build prog.py --plugin mine` is what people type. So every
    # subparser takes it too, and main() merges the two lists.
    # THE SUBVERBS TOO, not only the verbs: `uasm plugin backends --plugin
    # mine` is the whole point of the flag for a listing -- see what a plugin
    # adds without installing it -- and argparse would have taken it only in
    # front of `backends`, where nobody would think to put it.
    # BY IDENTITY, because an alias is the same parser under a second name:
    # `linkers` and `toolchains` are one entry in `choices` twice over, and
    # adding the flag to it twice is a conflict argparse raises at startup.
    seen: list[argparse.ArgumentParser] = []
    for p in list(sub.choices.values()) + list(pls.choices.values()):
        if any(p is done for done in seen):
            continue
        seen.append(p)
        p.add_argument("--plugin", action="append", default=[],
                       metavar="MODULE", help=argparse.SUPPRESS)
    return ap


def _named_plugins(argv: list[str]) -> list[str]:
    """The `--plugin` values in `argv`, wherever they appear.

    A pass of its own because plugins have to load BEFORE the real parser is
    built -- a third-party backend contributes flags, and argparse cannot
    accept a flag it has not been told about. `parse_known_args` on a parser
    that knows only this one option ignores everything else, including the
    subcommand it has no subparsers for.

    Two `--plugin` arguments exist on the real parser (before and after the
    subcommand) because a subparser OVERWRITES the namespace attribute rather
    than appending to it. Here there is one parser and one list, so both
    positions land in it.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--plugin", action="append", default=[])
    try:
        known, _ = pre.parse_known_args(argv)
    except SystemExit:
        # `--plugin` with no value. The real parser reports it properly a
        # moment from now; failing here would print the wrong usage line.
        return []
    return list(known.plugin)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the command.

    `LookupError` is caught here rather than at each call site. Every registry
    -- targets, toolchains, passes -- raises it for a name nobody registered,
    which is a typo, and a typo should not print a traceback with a compiler
    stack in it. Catching the type once means a new registry gets the same
    treatment without anyone remembering to wire it up.

    Nothing else is caught: an unexpected exception IS a compiler bug, and the
    traceback is the most useful thing to show.
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    # Plugins load before the PARSER, not merely before the command: a
    # third-party backend declares its own options (see `Backend.options`), and
    # those have to be on the parser by the time it reads the command line.
    try:
        report = plugins.load_all(_named_plugins(argv))
    except plugins.PluginError as exc:
        print(f"uasm: {exc}", file=sys.stderr)
        return 2
    for name, why in report.failed:
        # Installed but broken: said out loud every run, and never fatal.
        # `uasm plugin remove NAME` is how you fix it, and that command
        # cannot be the one thing a broken plugin prevents.
        print(f"uasm: warning: installed plugin {name!r} did not load "
              f"({why})", file=sys.stderr)

    args = build_parser().parse_args(argv)
    if report.loaded and getattr(args, "verbose", False):
        print(f"loaded plugins: {', '.join(report.loaded)}", file=sys.stderr)

    try:
        return args.fn(args)
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
