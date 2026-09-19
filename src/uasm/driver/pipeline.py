"""Orchestration: source in, artifacts out.

One function, `compile_source`, running the stages in order and stopping at the
first that fails. It exists so the CLI, the test suite and any embedding tool
drive the compiler through exactly the same path -- a test that reproduces a
bug through a different sequence of calls is testing something the user never
runs.

    parse + analyse   frontend      -> Module or None (errors reported)
    verify            ir.verify     -> internal error if the frontend is wrong
    optimise          passes        -> Module
    verify again      ir.verify     -> internal error if a pass is wrong
    emit              backend       -> {filename: bytes}

THE TWO VERIFY CALLS ARE NOT REDUNDANT. The first attributes bad IR to the
frontend, the second to the pass pipeline. Without both, a malformed module
reaching a backend is a crash whose cause could be either, and the difference
is which file you open.

Invalid IR is reported as an INTERNAL error, distinctly from a user error. A
user handed a list of IR invariants will reasonably assume their program is at
fault, and go looking in the wrong place.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from .. import backend as backend_registry
from .. import frontend as frontend_registry
from .. import target as target_registry
from ..target import Target
from ..diagnostics import DiagnosticSink, Severity, SourceFile, error, warning
from ..ir import Module, print_module, verify
from ..backend.base import BackendUnsupported
from ..ir.verifier import VerifyError
from ..objects import ir as objects_ir, support as objects_support
from ..passes import PassManager

#: Passes run when the user asks for optimisation but names none.
DEFAULT_PASSES = ("constfold", "copyprop", "dce", "simplifycfg")


@dataclass
class Options:
    """Everything that varies between invocations."""

    source: Path
    output: Path | None = None
    frontend: str | None = None
    #: Values for the options the chosen FRONTEND declares, keyed by option
    #: name without the dashes -- {"import-path": ["lib"], "library": True}.
    #: Not interpreted here: `--import-path`, `--host-python` and
    #: `--library` were the driver's own flags, which meant the driver knew
    #: how Python resolves an import.
    frontend_options: dict = field(default_factory=dict)
    backend: str = "c"
    #: Values for the options the chosen backend declares, keyed by option name
    #: without the dashes -- {"class-version": "75"}. Not interpreted here: the
    #: driver knows the flags exist because a backend said so, and knows
    #: nothing about what any of them mean.
    backend_options: dict[str, str] = field(default_factory=dict)
    target: Target | None = None
    passes: tuple[str, ...] = ()
    optimise: bool = False
    #: Produce a program, not just artifacts. False is `--emit`.
    link: bool = False
    #: THIS RUN WILL BE THROWN AWAY. `uasm verify` stops after the frontend
    #: and the IR verifier, so anything whose only consumer is a later stage
    #: is work nobody will read. Passed to the frontend rather than acted on
    #: here, because only the frontend knows which of its own work that is --
    #: and nothing that could change a DIAGNOSTIC may be skipped, or the
    #: command would be saying a program is fine on evidence it did not
    #: gather.
    verifying: bool = False
    toolchain: str = "cc"
    #: WHETHER `toolchain` WAS POSITIVELY DETERMINED -- named on the command
    #: line, or claimed by the output's extension. False means it was fallen
    #: back to, and the TARGET gets the last word: see below.
    toolchain_chosen: bool = False
    #: Values for the options the chosen LINKER declares, keyed by option
    #: name without the dashes. `--link-input` is one of these now: a linker
    #: that packages rather than links cannot honour it, and until it was a
    #: declaration neither `--help` nor the driver could say so.
    linker_options: dict = field(default_factory=dict)
    #: Extra objects/archives/-l names handed to the toolchain.
    #:
    #: NOT THE SAME LIST AS `linker_options["link-input"]`, though it starts
    #: as a copy of it: a `ctypes.CDLL("m")` in the SOURCE is a promise that
    #: `-lm` will be there, and the driver merges what the program needs into
    #: what the user named. See `_link_stage`.
    link_inputs: tuple[str, ...] = ()
    workdir: Path | None = None
    keep_intermediates: bool = False
    verbose: bool = False
    emit_ir: bool = False
    #: Ask the backend for its assembly instead of its artifacts. Only a
    #: machine backend has one; the rest refuse with a reason.
    emit_asm: bool = False
    show_spans: bool = False
    verify_each: bool = False
    time_passes: bool = False
    max_errors: int = 100
    warnings_are_errors: bool = False
    #: Where the object runtime comes from: `"ir"` compiles the ported part
    #: from `runtime/*.py` and splices it in; `"c"` uses the hand-written C for
    #: all of it, exactly as every build did before any of it was ported.
    #:
    #: BOTH ARE SUPPORTED ARRANGEMENTS, not a migration and a legacy. The
    #: reason to write the runtime in IR is that a backend should not HAVE to
    #: define 229 functions; that is an argument for making the C unnecessary,
    #: not for making it unavailable. `Backend.object_runtime` is the same
    #: choice at the granularity of one function.
    object_runtime: str = "ir"

    @property
    def effective_passes(self) -> tuple[str, ...]:
        if self.passes:
            return self.passes
        return DEFAULT_PASSES if self.optimise else ()


@dataclass
class Result:
    """What a compilation produced, plus how it got there."""

    module: Module | None = None
    artifacts: dict[str, bytes] = field(default_factory=dict)
    ir_text: str | None = None
    pass_report: str = ""
    #: The executable, if the link stage ran and succeeded.
    program: Path | None = None
    #: External commands the link stage ran, in order.
    commands: list[list[str]] = field(default_factory=list)
    #: The target actually used. Recorded because the link stage needs the
    #: object format and suffixes, and re-deriving them from the options
    #: would mean two places deciding what "the target" was.
    target: Target | None = None

    @property
    def ok(self) -> bool:
        return self.module is not None


def _target_os(opts: "Options", be) -> str | None:
    """The platform this build is for, before the emit stage resolves it.

    THE SAME EXPRESSION the emit stage uses, and deliberately so: a scoped
    native-library declaration picks a library by it, and two places deciding
    what "the target" was is how a program type-checks against `user32.dll`
    and links against `libX11.so.6`. None when nothing can say yet, which
    leaves only unscoped declarations applying.
    """
    chosen = None
    if opts.target is not None:
        chosen = opts.target.os
    elif be is not None:
        try:
            chosen = target_registry.get(be.default_target).os
        except Exception:
            chosen = None
    # `any` MEANS THE HOST, and this is the C backend's whole case. Its target
    # is `any/any` because it emits SOURCE -- but the toolchain then compiles
    # that source for the machine it is running on, so the program really is
    # a host program and `user32.dll` really is the library it wants. Left as
    # `any`, a declaration scoped to a platform could never apply to the
    # default backend, which is every ordinary build.
    if chosen in (None, "any"):
        try:
            return target_registry.host().os
        except Exception:
            return None
    return chosen


def _publish_backend_modules(be) -> None:
    """Tell the Python frontend what the selected backend makes importable.

    Reaching into the frontend from here rather than having the frontend ask
    for a backend: the frontend is one of several and none of them should know
    the backend registry exists. What travels is a table, not a dependency.
    """
    try:
        from uasm.frontends.python import modules as py_modules
    except ImportError:                     # the frontend is not installed
        return
    py_modules.use_backend(be.name if be else "", getattr(be, "modules", {}),
                           getattr(be, "java_classes", None))


def compile_source(opts: Options, sink: DiagnosticSink) -> Result:
    """Run the pipeline. Errors go to `sink`; `Result.ok` says whether to write."""
    frontend_registry.load_builtin()
    backend_registry.load_builtin()

    try:
        source = SourceFile.read(opts.source)
    except OSError as exc:
        sink.report(error("E9100", f"cannot read {opts.source}: {exc.strerror}"))
        return Result()

    # WHICH BACKEND IS COMPILING decides which names are importable, so it is
    # published BEFORE the frontend runs -- a backend for a board can offer
    # the board, and `import hw` has to resolve while the source is analysed
    # rather than when the artifacts are emitted. Republished every compile,
    # so two in one process do not see each other's backends.
    # Looked up through `available()` rather than `get()`: an unknown backend
    # is reported below, with the rest of the options checked, and `get` exits
    # the process rather than raising.
    #
    # CONFIGURED FIRST, because a backend's options can decide what it offers:
    # the JVM backend's `--classpath` is the whole of which Java packages are
    # importable, and publishing the unconfigured backend's modules would have
    # offered none of them.
    selected = backend_registry.available().get(opts.backend)
    if selected is not None:
        selected = _configure_backend(selected, opts, sink)
        if selected is None:
            return Result()
    _publish_backend_modules(selected)

    fe = (frontend_registry.get(opts.frontend) if opts.frontend
          else frontend_registry.for_path(opts.source))
    if fe is None:
        sink.report(
            error("E9101", f"no frontend claims {opts.source.suffix!r}")
            .help("choose one with --frontend "
                  + "|".join(sorted(frontend_registry.available()))))
        return Result()

    # THE FRONTEND'S OWN FLAGS, handed to the frontend. This used to be a
    # block of driver code importing `hostlib`, `imports` and `nativelib` out
    # of `frontends/python` by name: the driver knew how Python resolves an
    # import, and a second frontend could not have been added without it
    # growing a branch.
    #
    # THE TARGET IS RESOLVED HERE and passed, exactly as the emit stage
    # resolves it -- a scoped native-library declaration picks a library by
    # it, and two places deciding what "the target" was is how a program
    # type-checks against `user32.dll` and links against `libX11.so.6`.
    fe = _configure_frontend(
        fe, opts,
        frontend_registry.BuildContext(source=opts.source,
                                       target_os=_target_os(opts, selected),
                                       verifying=opts.verifying),
        sink)
    if fe is None:
        return Result()

    try:
        module = fe.compile(source, sink)
    except RecursionError:
        # A long expression is a deep tree, and analysis and lowering both
        # walk it recursively. `1 + 2 + ... + 999` exhausted the interpreter
        # stack and reached the user as a traceback ending in `_binop`, which
        # reads as a compiler crash. It is a real limit and it has a real
        # cause, so it gets said.
        sink.report(
            error("E9105", "expression is too deeply nested to compile")
            .note("analysis and lowering walk the expression tree "
                  "recursively, and this one is deeper than the stack allows")
            .help("split it across several statements"))
        return Result()
    if module is None or sink.failed:
        return Result()

    if not _verify_stage(module, sink, "the frontend"):
        return Result()

    # THE PART OF THE OBJECT RUNTIME THAT IS IR, merged in before anything
    # downstream looks at the module. Here rather than in the frontend because
    # it is not the frontend's: a second frontend producing the same `apy_*`
    # calls gets the same runtime, which is the whole point of writing it in
    # IR. And before the passes, so the runtime is optimised with everything
    # else rather than being the one part that is not.
    #
    # A program that does not reach the runtime gets nothing at all -- see
    # `objects_ir.wants_runtime`.
    objects_ir.splice(
        module, sink,
        provided=getattr(selected, "object_runtime", frozenset()),
        enabled=opts.object_runtime != "c")
    if not _verify_stage(module, sink, "the IR runtime splice"):
        return Result()

    result = Result(module=module)

    names = opts.effective_passes
    if names:
        pm = PassManager.from_names(list(names), verify_each=opts.verify_each)
        pm.run(module, sink)
        if sink.failed:
            return Result()
        if opts.time_passes:
            result.pass_report = pm.report()
        if not _verify_stage(module, sink, "the pass pipeline"):
            return Result()

    if opts.emit_ir:
        result.ir_text = print_module(module, show_spans=opts.show_spans)
        return result

    # The instance configured above, not a fresh one: a backend's `configure`
    # may have built state the frontend has since been reading from -- the JVM
    # backend records which Java calls were named while the source was
    # analysed, and emits exactly those.
    be = selected if selected is not None else backend_registry.get(opts.backend)
    if not be.ready:
        sink.report(
            error("W9102", f"backend {be.name!r} is not finished")
            .note("its output may be incorrect or incomplete"))
        sink.diagnostics[-1].severity = Severity.WARNING
    target = opts.target if opts.target is not None \
        else target_registry.get(be.default_target)
    result.target = target
    try:
        # BEFORE `emit`, so a missing capability is a diagnostic naming the
        # group rather than an undefined symbol naming an object file.
        be.check_host_services(module)
        result.artifacts = (be.assembly(module, target) if opts.emit_asm
                            else be.emit(module, target))
    except BackendUnsupported as exc:
        sink.report(
            error("E9103", f"the {be.name} backend cannot compile this program "
                           f"for {target.name}")
            .note(str(exc)))
        return Result()

    if opts.link:
        _link_stage(opts, result, be, target, module, sink)
    return result


#: Which flag selects each kind of component, for an error that has to say
#: how to reach the component that DOES take the option the user typed.
_SELECTS = {"backend": "--backend", "frontend": "--frontend",
            "linker": "--linker"}


def _who_takes(name: str) -> list[tuple[str, str]]:
    """Every registered component declaring an option called `name`.

    ACROSS ALL THREE KINDS and not just the one that refused. `--import-path
    --frontend apir` is a user reaching for the PYTHON frontend's flag, and
    an answer that searched only frontends would still be right -- but
    `--class-version --backend c` searching only backends was how a linker's
    flag or a frontend's got "no idea whose this is" when its owner was one
    registry over.
    """
    from .. import backend as backend_registry
    from .. import frontend as frontend_registry
    from .. import link as link_registry

    found = []
    for kind, registry in (("backend", backend_registry),
                           ("frontend", frontend_registry),
                           ("linker", link_registry)):
        for owner, component in sorted(registry.available().items()):
            if any(o.name == name for o in component.options):
                found.append((owner, kind))
    return found


def _reject_strays(kind: str, component, values: dict,
                   sink: DiagnosticSink) -> bool:
    """Report every option `component` does not declare. True if any was.

    AN ERROR RATHER THAN SOMETHING IGNORED. `--class-version 75 --backend c`
    reads as a request the C backend cannot honour, and quietly building
    without it hands back an artifact that is not what was asked for.
    """
    declared = {o.name for o in component.options}
    stray = sorted(set(values) - declared)
    for name in stray:
        d = error("E9106",
                  f"the {component.name} {kind} does not take --{name}")
        takers = _who_takes(name)
        if takers:
            owners = [o for o, _ in takers]
            spoken = (owners[0] if len(owners) == 1
                      else ", ".join(owners[:-1]) + " and " + owners[-1])
            kinds = sorted({k for _, k in takers})
            d.help(f"--{name} belongs to the {spoken} "
                   + "/".join(kinds)
                   + ("" if len(owners) == 1 else "s")
                   # NO ONE OF THEM SUGGESTED when several take it: which to
                   # pick is the user's decision about what to build, and
                   # naming the alphabetically first would read as advice.
                   + (f"; pass {_SELECTS[takers[0][1]]} {owners[0]}"
                      if len(owners) == 1 else ""))
        sink.report(d)
    return bool(stray)


def _configure_frontend(fe, opts: Options, context, sink: DiagnosticSink):
    """Hand the frontend its own options. Returns it, or None to stop."""
    from ..options import OptionError

    if _reject_strays("frontend", fe, opts.frontend_options, sink):
        return None
    declared = {o.name for o in fe.options}
    mine = {name: value for name, value in opts.frontend_options.items()
            if name in declared}
    try:
        return fe.configure(mine, context, sink)
    except OptionError as exc:
        message, _, detail = str(exc).partition("\n")
        d = error("E9107", f"{fe.name} frontend: {message}")
        for line in detail.splitlines():
            d.note(line)
        sink.report(d)
        return None


def _configure_backend(be, opts: Options, sink: DiagnosticSink):
    """Hand the backend its own options. Returns the backend, or None to stop."""
    from ..options import OptionError

    if _reject_strays("backend", be, opts.backend_options, sink):
        return None
    declared = {o.name for o in be.options}
    mine = {name: value for name, value in opts.backend_options.items()
            if name in declared}
    try:
        return be.configure(mine, sink)
    except OptionError as exc:
        message, _, detail = str(exc).partition("\n")
        d = error("E9107", f"{be.name} backend: {message}")
        for line in detail.splitlines():
            d.note(line)
        sink.report(d)
        return None


def _link_stage(opts: Options, result: Result, be, target: Target,
                module: Module, sink: DiagnosticSink) -> None:
    """Artifacts to a program.

    Separate from emission because they fail for unrelated reasons and the
    user needs to know which happened: a backend that cannot compile a
    construct is a compiler limitation, while a missing assembler is a machine
    that needs a package installed. Collapsing both into "build failed" sends
    people to the wrong place.
    """
    from .. import link as link_registry

    # A bare-metal target cannot be linked by the hosted toolchain: no libc,
    # no start files, and a linker script that has to match the machine. Nor
    # can a class file, which is packaged rather than linked. The default
    # follows the target rather than making every invocation say so.
    # THE TARGET GETS THE LAST WORD ONLY WHEN NOTHING ELSE SAID. This used
    # to compare the toolchain against the string "cc", which worked only
    # because "cc" was the hardcoded default and therefore stood in for "not
    # chosen". Once the driver began choosing from the output's extension,
    # `-o Prog.class` resolved to `none` -- a real answer, not the sentinel --
    # and the jvm target's `jar` never applied, so the class file was written
    # and the jar beside it was not.
    name = opts.toolchain
    if not opts.toolchain_chosen:
        if target.default_toolchain:
            name = target.default_toolchain
        elif target.os == "none":
            name = "baremetal"
    toolchain = link_registry.get(name)
    # THE LINKER'S OWN FLAGS, checked against the linker that was chosen.
    # `--link-input -o thing.jar` is a request the jar toolchain cannot
    # honour, and it used to be found out at link time, as a LinkError about
    # inputs, rather than said before anything was built.
    if _reject_strays("linker", toolchain, opts.linker_options, sink):
        result.module = None          # nothing usable was produced
        return
    workdir = opts.workdir or (opts.output or opts.source).parent / ".uasm"
    output = opts.output or opts.source.with_suffix(target.executable_suffix)
    if output.suffix != target.executable_suffix and target.executable_suffix:
        output = output.with_suffix(target.executable_suffix)

    # NATIVE LIBRARIES THE SOURCE NAMED, as `-l` flags. A `ctypes.CDLL("m")`
    # is a promise to the linker and this is where it is kept; see
    # `frontends/python/cffi.py`.
    try:
        from ..frontends.python import cffi as py_cffi
        named = tuple(py_cffi.link_flag(lib)
                      for lib in py_cffi.named_libraries())
    except ImportError:                     # the frontend is not installed
        named = ()
    if named:
        opts = replace(opts, link_inputs=opts.link_inputs + named)

    runtime_sources: tuple[Path, ...] = ()
    if not be.self_contained and objects_support.needs_runtime(module):
        # THE MODULE DECIDES which `apy_*` the C still defines: whatever this
        # program supplies in IR, the C stands aside for. See
        # `objects.ir.omitted_by`.
        runtime_sources = (
            objects_support.write_runtime(workdir, module=module),)

    request = link_registry.LinkRequest(
        artifacts=result.artifacts, target=target, output=output,
        workdir=workdir, extra_inputs=opts.link_inputs,
        runtime_sources=runtime_sources,
        keep_intermediates=opts.keep_intermediates, verbose=opts.verbose)
    try:
        result.program = toolchain.link(request)
    except link_registry.LinkError as exc:
        d = error("E9104", exc.message)
        if exc.detail:
            d.note(exc.detail)
        if exc.help:
            d.help(exc.help)
        sink.report(d)
        result.module = None          # nothing usable was produced
    finally:
        result.commands = [list(c) for c in request.commands]
        # WHAT THE TOOLCHAIN WANTS SAID ABOUT A LINK THAT WORKED. See
        # `LinkRequest.notes`: an image can be correct and still be one the
        # target platform will not start.
        for note in request.notes:
            sink.report(warning("W9116", note))


def _verify_stage(module: Module, sink: DiagnosticSink, who: str) -> bool:
    try:
        verify(module)
        return True
    except VerifyError as exc:
        d = error("E9999", f"internal error: {who} produced invalid IR")
        d.note("This is a bug in the compiler, not in your program.")
        for problem in exc.problems[:10]:
            d.note(problem)
        if len(exc.problems) > 10:
            d.note(f"... and {len(exc.problems) - 10} more")
        sink.report(d)
        return False
