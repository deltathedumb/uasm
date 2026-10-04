"""The backend interface and target description.

A backend turns a verified `Module` into artifacts. The interface is one
abstract method, because the module it receives has already passed `verify()`
and every invariant listed there holds -- a backend needs no defensive checks.

    class MyBackend(Backend):
        name = "my-machine"
        def emit(self, module, target) -> dict[str, bytes]: ...

REGISTER ALLOCATION IS OPTIONAL. The simplest correct backend gives every
virtual register its own stack slot and never allocates. `uasm.backend.regalloc`
is a library a backend may call; it is not a stage anyone must implement, and
`backends/c` never touches it.
"""
from __future__ import annotations

import abc
import copy
from dataclasses import dataclass

from ..diagnostics import is_real

from ..ir import Module
# SHARED WITH THE FRONTENDS AND THE LINKERS, so re-exported rather than defined
# here: a backend has declared its own flags since the beginning and every
# backend imports `Option` from this module. See `uasm/options.py` for why
# the other two kinds needed the same thing.
from ..options import Option, OptionError  # noqa: F401  (re-export)
# Re-exported so a backend author needs one import. The TYPE belongs to the
# backend interface -- every `emit` receives one -- but the INSTANCES do not
# live here: they are registered in `uasm.targets`, so adding a platform never
# means editing the compiler. See docs/TARGETS.md.
from ..target import Target

#: The symbol a backend emits the IR's `main` under.
#:
#: The IR's `main` is not C's `main` -- it returns i64 where C requires int --
#: so a backend that emitted it verbatim would collide with the entry point in
#: whatever runtime gets linked alongside. Named here because the backend
#: writing the symbol and the runtime calling it must agree, and two constants
#: that must agree are one constant.
ENTRY_SYMBOL = "uasm_main"

class Backend(abc.ABC):
    """Turn a verified module into named artifacts."""

    name: str = ""
    description: str = ""
    #: Options this backend takes from the command line. The driver turns each
    #: into a flag and hands the values back through `configure`.
    options: tuple[Option, ...] = ()
    #: False for a work in progress; the driver warns.
    ready: bool = True
    #: WHAT KIND OF ARTIFACT THIS PRODUCES, and therefore whether text is
    #: allowed out of it. Two answers and the rule is different for each:
    #:
    #:   "language"  Emits SOURCE IN ANOTHER LANGUAGE -- C, LLVM IR. Text is
    #:               the artifact; something downstream compiles it. There is
    #:               nothing to encode and no object format to write.
    #:   "binary"    Emits MACHINE CODE OR BYTECODE -- an ELF/COFF/Mach-O
    #:               object, a class file, a wasm module, a .pyc. The output
    #:               is bytes that a loader takes directly.
    #:
    #: A BINARY BACKEND MAY NOT EMIT ASSEMBLY. That is the whole reason this
    #: field exists rather than being obvious from the name. Assembly text is
    #: neither: it is a binary backend that stopped one stage early and handed
    #: the last one to `as`, which reads as working -- the file exists, the
    #: toolchain links it -- while the backend has in fact never encoded an
    #: instruction. Saying "binary" here is a claim that `emit` produces the
    #: bytes itself, and `test_backend_kinds.py` checks the claim.
    kind: str = "binary"
    #: Name of the target used when the user names none. A NAME, not a
    #: Target: holding an instance here would import the built-in targets
    #: to define the backend interface, putting platforms back inside the
    #: compiler.
    default_target: str = "c"
    #: True if the artifacts already form a complete program -- an entry point
    #: and every host function the frontend calls. The C backend is: it emits
    #: its own `main` wrapper and its own `print_int`. A machine backend is
    #: not, and the link stage supplies the runtime for it. Getting this wrong
    #: produces either a duplicate-symbol error or an undefined one, both at
    #: link time and both clear.
    self_contained: bool = False
    #: `apy_*` THIS BACKEND DEFINES ITSELF, and does not want supplied as IR.
    #:
    #: Part of the object runtime is now written in uasm's own machine
    #: subset and compiled into every program that needs it
    #: (`objects/ir.py`), which is how a backend stops having to define
    #: 229 functions. THIS IS THE OPT-OUT, and it is not an afterthought: the
    #: point of that work is to REMOVE an obligation, not to replace it with a
    #: different one. A backend with its own implementation of a function --
    #: hand-written, faster, or simply older and trusted -- names it here and
    #: keeps it, and the splice supplies only what is left.
    #:
    #: Empty for every built-in backend, which is what makes the shipped
    #: arrangement the ported one. `Options.object_runtime = "c"` is the same
    #: opt-out for a whole BUILD rather than for a backend, and either one
    #: leaves the C runtime exactly as it was before any of it was ported.
    object_runtime: frozenset[str] = frozenset()
    #: HOST SERVICE GROUPS THIS BACKEND PROVIDES -- "file", "net", "time",
    #: "random", "env", "text". See `objects/hostsvc.py` for the operations in
    #: each and for why they are optional rather than part of the floor.
    #:
    #: EMPTY IS A COMPLETE BACKEND. That is the whole reason this is declared
    #: rather than assumed: the platform floor is three functions because
    #: making it five was a cost every backend paid forever, and a filesystem
    #: is not something a bare-metal target has. A program that opens a file
    #: is refused HERE, at compile time, naming the operation and the
    #: capability -- not at link time as an undefined symbol, and not at run
    #: time as a wrong answer.
    host_services: frozenset[str] = frozenset()
    #: MODULES THIS BACKEND MAKES IMPORTABLE, as {name: {member: spec}} in the
    #: shape `frontends/python/modules.py` documents.
    #:
    #: A backend targeting a board can offer the board: `import hw` reads a
    #: pin. Which is fine until two backends, or a backend and the standard
    #: set, both want the name `json` -- so the rule is:
    #:
    #:   * `import <backend>.<name>` ALWAYS reaches this backend's module,
    #:     collision or not. The prefixed path is not a fallback, it is the
    #:     real name and it always works.
    #:   * `import <name>` reaches it only if nothing else already has that
    #:     name. A pre-existing module WINS, silently and deliberately: a
    #:     program that said `import json` before a backend grew one of its
    #:     own must keep meaning what it meant.
    #:
    #: So a backend author picks any name they like and the collision resolves
    #: itself, at the cost of the prefix for the loser -- which is exactly
    #: where the ambiguity was.
    modules: dict[str, dict] = {}

    def configure(self, values: dict[str, str], sink) -> "Backend":
        """The backend to emit with, given this run's option values.

        `values` holds only the options this backend declared, keyed by name
        without the dashes. Raise `OptionError` for a value that cannot be
        used; report anything advisory to `sink`.

        RETURN A NEW INSTANCE rather than mutating `self`. The registry holds
        one shared backend object, so a backend that stored its flags on itself
        would leak them into the next compilation in the same process -- which
        is invisible in a command-line run and wrong in every test suite and
        every embedding tool.

        The default ignores both arguments, which is right for a backend with
        no options: the driver rejects a flag no backend declared before it
        ever gets here.
        """
        return self

    def check_host_services(self, module: Module) -> None:
        """Refuse a program that needs a capability this backend has not got.

        HERE RATHER THAN AT LINK TIME, which is the whole point. A frontend
        emits `host_file_open` as an ordinary external call, so without this
        the failure is an undefined symbol naming an object file -- or, for a
        backend that resolves lazily, nothing at all until the program runs.
        Neither says "this target has no filesystem".

        NAMED BY CAPABILITY AND NOT ONLY BY FUNCTION, because the answer is
        never to implement one operation: a target with files has all of them
        or none. Telling an author that `host_file_open` is missing invites
        them to add one function; telling them the `file` group is missing
        says what the work actually is.

        The `core` group is not checked. It is the platform floor, every
        backend owes it, and a backend that has not implemented it fails in
        ways this could only describe worse.
        """
        from ..objects import hostsvc

        wanted: dict[str, str] = {}
        for fn in module.functions:
            if not fn.external:
                continue
            group = hostsvc.group_of(fn.name)
            if group is not None and group not in hostsvc.MANDATORY                     and group not in self.host_services:
                wanted.setdefault(group, fn.name)
        if not wanted:
            return
        listed = ", ".join(
            f"{g!r} (for {wanted[g]})" for g in sorted(wanted))
        raise BackendUnsupported(
            f"this program needs host services the {self.name} backend does "
            f"not provide: {listed}. See objects/hostsvc.py for what each group "
            f"is, and `Backend.host_services` for how a backend declares one.")

    #: THE EXTENSIONS THIS BACKEND'S OWN ARTIFACTS CARRY, most specific
    #: first -- what `emit` writes, BEFORE any linker has run.
    #:
    #: THE SYMMETRIC HALF OF `Frontend.extensions`, which has always let the
    #: source's spelling choose a frontend. A backend had no such declaration,
    #: so the driver could only default to one name and make every other
    #: choice the user's to type: `-o foo.wasm` still built C.
    #:
    #: NOT WHAT THE PROGRAM ENDS UP AS. `cpyext` writes `.c` here and the
    #: toolchain of the same name turns it into `.so` -- so an output named
    #: `.so` is the LINKER's artifact and chooses the backend only through it.
    #: See `Toolchain.artifacts`.
    #:
    #: EMPTY MEANS "NEVER CHOSEN BY SPELLING", which is honest for a backend
    #: whose artifact has no extension of its own to claim.
    artifacts: tuple[str, ...] = ()

    @abc.abstractmethod
    def emit(self, module: Module, target: Target) -> dict[str, bytes]:
        """Compile `module`. Returns {filename: contents}.

        `module` has passed verify(). Do not re-check its invariants.
        """

    def assembly(self, module: Module, target: Target) -> dict[str, bytes]:
        """The same program as ASSEMBLY, for a backend that has one.

        WHY THIS IS PART OF THE INTERFACE and not a private path. A machine
        backend that writes object files has no reason to produce text on the
        way there, and once it stops, its assembly is unreachable -- which
        loses two things worth keeping. A compiler that cannot show you what
        it generated is much harder to debug than one that can, and the x86
        lifter's whole input is x86 assembly this backend wrote.

        So the text stays, with a name and a caller: `build --emit-asm`.
        A backend with no assembly to show refuses, which is the honest
        answer for the C and JVM backends -- their artifact IS the readable
        form.
        """
        raise BackendUnsupported(
            f"the {self.name} backend has no assembly form; its artifacts "
            f"are already the readable output")

    def __repr__(self) -> str:
        return f"<backend {self.name}>"


@dataclass(frozen=True, slots=True)
class MachineConfiguration:
    """Machine choices that are independent of an operating-system target."""

    cpu: str
    features: frozenset[str]


class MachineBackend(Backend):
    """A native backend with explicit ISA configuration.

    The target owns ABI, object format and pointer width.  This class owns the
    ISA baseline and optional instructions the emitter is allowed to use.  A
    feature is refused until the emitter really implements it: silently
    accepting a compiler flag is worse than not offering it.
    """

    architecture: str = ""
    word_bits: int = 0
    default_cpu: str = "generic"
    cpus: frozenset[str] = frozenset(("generic",))
    supported_features: frozenset[str] = frozenset()
    required_features: frozenset[str] = frozenset()
    options = (
        Option("cpu", "ISA baseline to emit for", metavar="CPU"),
        Option("feature", "enable or disable an ISA feature (+name or -name)",
               metavar="FEATURE", repeat=True),
    )
    machine = MachineConfiguration("generic", frozenset())

    def configure(self, values: dict[str, str], sink) -> "MachineBackend":
        cpu = values.get("cpu", self.default_cpu).lower()
        if cpu not in self.cpus:
            raise OptionError(
                f"unsupported CPU {cpu!r}; supported: {', '.join(sorted(self.cpus))}")
        features = set(self.required_features)
        for raw in values.get("feature", ()):
            sign, name = raw[:1], raw[1:].lower()
            if sign not in ("+", "-") or not name:
                raise OptionError(f"--feature needs +name or -name, not {raw!r}")
            if name not in self.supported_features:
                raise OptionError(
                    f"the {self.name} backend cannot emit feature {name!r}")
            if sign == "+":
                features.add(name)
            elif name in self.required_features:
                raise OptionError(f"the {self.name} backend requires feature {name!r}")
            else:
                features.discard(name)
        clone = copy.copy(self)
        clone.machine = MachineConfiguration(cpu, frozenset(features))
        return clone

    def validate_target(self, target: Target) -> None:
        if target.arch != self.architecture:
            raise BackendUnsupported(
                f"target {target.name!r} has architecture {target.arch!r}, "
                f"but this backend emits {self.architecture!r}")
        if target.pointer_bits != self.word_bits:
            raise BackendUnsupported(
                f"target {target.name!r} has {target.pointer_bits}-bit pointers, "
                f"but the {self.name} backend emits {self.word_bits}-bit code")
        if not target.little_endian:
            raise BackendUnsupported(
                f"the {self.name} backend does not implement big-endian output")


def source_file_of(module: Module):
    """The `SourceFile` the frontend parsed `module` from, or `None`.

    `Module` carries no source text of its own -- `metadata["source"]` is a
    display name, not the text -- but every instruction, function and global
    the frontend produced was spanned against the file it came from, and a
    span's `.file` is the real `SourceFile`, text included. Shared by any
    backend that needs the ORIGINAL source rather than the IR: `pybc`, which
    hands it to the host's own `compile()`, and `cpyext`, which re-parses it
    to check a function's calling convention before exporting it.

    Checking function/global spans before descending into every instruction
    is just cheaper: most programs have one within the first few functions.
    """
    for fn in module.functions:
        if is_real(fn.span):
            return fn.span.file
        for block in fn.blocks:
            for instr in block.instructions:
                if is_real(instr.span):
                    return instr.span.file
    for g in module.globals:
        if is_real(g.span):
            return g.span.file
    return None


_REGISTRY: dict[str, Backend] = {}


def register(be: Backend) -> Backend:
    if not be.name:
        raise ValueError(f"{type(be).__name__} has no name")
    if be.name in _REGISTRY:
        raise ValueError(f"backend {be.name!r} is already registered")
    _REGISTRY[be.name] = be
    return be


def get(name: str) -> Backend:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise SystemExit(f"unknown backend {name!r}\navailable: {known}") from None


def available() -> dict[str, Backend]:
    return dict(_REGISTRY)


def load_builtin() -> None:
    # THE STUBS ARE IMPORTED TOO. Each declares `ready = False` and refuses to
    # emit -- so `uasm plugin backends` shows the whole matrix with the
    # unfinished half marked, rather than showing four and leaving the rest to
    # be discovered as "unknown backend".
    from ..backends import (                                   # noqa: F401
        arm32, arm64, c, cpyext, jvm, llvm, pybc, uir, wasm, x86_32, x86_64,
    )


class BackendUnsupported(Exception):
    """A backend cannot compile this program for this target.

    Distinct from a crash and from a user error: the program is valid and the
    compiler is working, but this particular code generator does not implement
    the construct yet. The driver turns it into a diagnostic naming the
    backend and the target, so the answer ("use --backend c") is visible
    rather than something to work out from a traceback.
    """
