"""Python frontend: it compiles Python.

    analysis.py   resolve names, assign a type to every expression, report
    lower.py      typed AST -> IR

Lowering runs only if analysis reported nothing, so it contains no validation.

NOT A SUBSET, and this docstring used to say it was. It read "THE SUBSET:
annotated int/float/bool parameters and locals... No objects, no dynamic
typing, no exceptions, no closures", which described this frontend before the
dynamic path existed and stayed there while the conformance suite went to
1668/1668 -- objects, closures, generators, coroutines, metaclasses,
`except*`, t-strings and `compile()` included. A reader checking what
uasm accepts found a sentence saying "almost nothing", four years of work
out of date.

TWO PATHS, AND THE CHOICE IS AN OPTIMISATION RATHER THAN A LIMIT:

  * STATIC -- a function whose every parameter and return is annotated keeps
    machine representations: an `int` is a 64-bit register, a `float` an xmm
    register, and nothing is allocated.
  * DYNAMIC -- module-level code and any function with an unannotated
    parameter. Every value is a runtime object carrying its own type. This is
    the path ordinary Python takes, and it is the one the conformance suite
    measures.

A dynamic function calling a static one unwraps each argument to the type it
declared and wraps the result back. That boundary is the only place the two
meet, which is what stops a value's representation depending on the slot it
was stored in.

WHERE THE TWO STILL DISAGREE is written down rather than left to be found:
`and`, `or` and `a if c else b` YIELD AN OPERAND in Python, so `x and y` over
an `int` and a `float` answers whichever one it picked -- and the static path
gives every expression one type, so it widens to `float` and prints `0.0`
where CPython and the dynamic path print `0`. See `_unify_all` in
`analysis.py`. That is a defect and not a documented feature of a subset.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path

from ...diagnostics import DiagnosticSink, SourceFile, error, warning
from ...frontend import BuildContext, Frontend, register
from ...ir import Module
from ...options import Option, OptionError
from .analysis import Analyzer, span_of
from . import cffi
from .bundled import _bound_locally, splice
from .imports import splice as user_splice
from .lower import Lowerer


#: Spellings that are not distinct types: PEP 3151 folded the old I/O error
#: names into `OSError`, and CPython keeps them only as aliases.
_EXC_ALIASES = {"IOError": "OSError", "EnvironmentError": "OSError",
                "WindowsError": "OSError"}


class _Aliases(ast.NodeTransformer):
    """Replace an aliased builtin name with the one it IS."""

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = _EXC_ALIASES.get(node.id, node.id)
        return node


#: The builtins that need a compiler at run time. Named here as well as in
#: `bundled.py` because the two ask different questions about them: that one
#: routes the call, this one says what it costs.
_RUNTIME_COMPILER = ("compile", "eval", "exec")


def _warn_about_runtime_compilation(tree, source, sink):
    """Say what `compile`, `eval` and `exec` cost, where they are written.

    A WARNING AND NOT AN ERROR. The program gets a real one: the parser, the
    validator and the code object are bundled Python spliced into it. What it
    does not get is the compiler that built the binary, and the difference is
    worth a line at the call site rather than a surprise later.

    TWO MESSAGES, because the costs differ. `compile()` answers whether source
    is valid Python and stops. `eval()` and `exec()` RUN it, and running it is
    interpretation -- far slower than the native code around it, which is the
    part a reader needs told.

    REPORTED BEFORE THE SPLICE, which is the last moment the bare name is
    still in the tree: afterwards it points at the bundled definition and
    nothing can tell it was ever written.
    """
    bound = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            bound.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    bound.add(target.id)
    said = set()

    def report(node):
        said.add(node.id)
        note = ("it answers whether source is valid Python, through the "
                "parser bundled into this binary -- not through the compiler "
                "that built it") if node.id == "compile" else (
            "the source it is given is INTERPRETED rather than compiled: it "
            "runs through the interpreter bundled into this binary, and is "
            "far slower than the code around it")
        sink.report(
            warning("W0091",
                    f"{node.id}() is not recommended in a compiled program")
            .at(span_of(source, node))
            .note(note)
            .help("prefer a function the compiler can see and call directly"))

    def walk(node, shadowed):
        # SCOPE BY SCOPE, not `ast.walk`. The module-level `bound` above is
        # not the whole of shadowing: a program with its own `def eval` INSIDE
        # a function was told its own function is not recommended, naming a
        # line that has nothing to do with the builtin. A nested scope's
        # bindings are added on the way in and dropped on the way out, so a
        # genuine `eval()` elsewhere in the module still gets the warning.
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.Lambda, ast.ClassDef)):
                walk(child, shadowed | _bound_locally(child))
                continue
            if isinstance(child, ast.Name) and child.id in _RUNTIME_COMPILER \
                    and child.id not in shadowed and child.id not in said:
                report(child)
            walk(child, shadowed)

    walk(tree, bound)


def _stringifies(tree: ast.Module) -> bool:
    """Did this module ask for PEP 563?"""
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            if any(a.name == "annotations" for a in stmt.names):
                return True
    return False


class _Stringify(ast.NodeTransformer):
    """Replace every annotation with its source text.

    `ast.unparse` is the round trip Python itself uses for this, so the text a
    program reads back is the text CPython would give it -- `list[int]` and
    not `list [ int ]`.
    """

    @staticmethod
    def _text(node):
        made = ast.copy_location(ast.Constant(value=ast.unparse(node)), node)
        # MARKED, so analysis can tell this string from a forward reference a
        # program wrote itself. A quoted `'int'` IS the static int; a PEP 563
        # annotation is TEXT the program asked not to have evaluated, and
        # using it as a type would contradict the directive that made it text.
        made.pep563 = True
        return made

    def visit_arg(self, node: ast.arg):
        if node.annotation is not None:
            node.annotation = self._text(node.annotation)
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.returns is not None:
            node.returns = self._text(node.returns)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self.generic_visit(node)
        node.annotation = self._text(node.annotation)
        return node


class PythonFrontend(Frontend):
    name = "python"
    extensions = (".py",)
    description = "Python 3.14"

    #: The language contract is explicit.  A compiler version is not a
    #: language version: users need to be able to pin the grammar and
    #: semantics their source was written for, and a future frontend can add
    #: another supported release without changing the driver's interface.
    language_versions = ("3.14",)
    default_language_version = "3.14"

    #: THE FLAGS THIS FRONTEND TAKES, and no longer the driver's. Every one
    #: of them is about resolving Python names or compiling Python: a search
    #: path, an interpreter to borrow packages from, a declaration of what
    #: may be imported from a shared library. The driver used to carry all of
    #: them, which meant a second frontend would inherit `--host-python` and
    #: have nothing to do with it.
    options = (
        Option("language-version", metavar="VERSION",
               help="Python language version (currently 3.14)"),
        Option("import-path", metavar="DIR", repeat=True,
               help="where to find the program's own modules; the source's "
                    "own directory is searched too unless -P"),
        # CPYTHON'S OWN FLAG, spelled the same. `-P` is what a program uses
        # when a file beside it is named after a module it imports and it
        # wants the real one.
        Option("safe-path", switch=True, short="P",
               help="do not search the source's own directory, "
                    "as CPython's -P does"),
        # THE HOST INSTALLATION'S PACKAGES. Searched LAST, after the bundled
        # standard library, the source's directory and every --import-path,
        # so nothing that resolved before this flag existed resolves
        # differently because of it. See `hostlib.py`.
        Option("no-site-packages", switch=True,
               help="do not search the host Python installation's "
                    "site-packages (see `uasm plugin libraries`)"),
        Option("host-python", metavar="PATH",
               help="the interpreter whose site-packages to search; "
                    "default is the one running the compiler"),
        # A SHARED LIBRARY THE PROGRAM MAY `import`. Declared rather than
        # discovered: a foreign symbol's argument kinds cannot be read out of
        # the library, and guessing them is how a native call corrupts a
        # stack. See `nativelib.py`.
        Option("native-library", metavar="FILE", repeat=True,
               help="JSON declaring shared libraries this program may "
                    "import, and the signatures it calls in them"),
        # A LIBRARY HAS NO ENTRY AND IS NOT SUPPOSED TO. Every top-level
        # `def` is exported instead of only `main`, and a module of nothing
        # but definitions is not "nothing to run" (E0003) but the point.
        Option("library", switch=True,
               help="compile definitions only, with no `main`; every "
                    "top-level function is exported (needed to target "
                    "`cpyext`, and useful with `run --entry` to call one "
                    "function directly)"),
    )

    #: Set by `configure` on the copy it returns; see `compile`.
    library = False
    #: Set by `configure` from `BuildContext.verifying`; see `compile`.
    verifying = False
    language_version = default_language_version

    def configure(self, values: dict, context: BuildContext,
                  sink: DiagnosticSink) -> "PythonFrontend | None":
        """Publish the search path and the native declarations, and take
        `--library` for this run.

        HERE AND NOT IN THE DRIVER, which is where it used to be: the driver
        imported `hostlib`, `imports` and `nativelib` out of this package by
        name, so "the compiler" knew how Python resolves an import and a
        second frontend could not have been added without the driver growing
        a branch for it.

        THROUGH MODULE GLOBALS, still: the frontend is handed a source and a
        sink, so anything the run knows and the compilation needs arrives
        the way it always has -- `imports.use` and `nativelib.use` are
        republished on every compilation, so two in one process cannot see
        each other's.
        """
        from . import hostlib, imports as py_imports, nativelib as py_nativelib

        version = values.get("language-version", self.default_language_version)
        if version not in self.language_versions:
            raise OptionError(
                f"unsupported Python language version {version!r}; "
                f"supported: {', '.join(self.language_versions)}")

        # THE HOST INSTALLATION'S PACKAGES GO LAST, so a name that resolved
        # before library points existed still resolves to what it resolved
        # to then.
        wanted = values.get("host-python")
        host = (hostlib.HostLibrary() if values.get("no-site-packages")
                else hostlib.discover(wanted))
        if host.unavailable and wanted:
            # ONLY WHEN THE USER NAMED ONE. A failure to introspect the
            # running interpreter means site-packages are simply not
            # available and the program may well not need them; a failure to
            # run the interpreter the user typed is about the flag they
            # typed, and is worth saying.
            sink.report(
                error("E9108", f"--host-python: {host.unavailable}")
                .help("give the path of a Python interpreter, or pass "
                      "--no-site-packages to search none"))
            return None
        # THE SOURCE'S DIRECTORY IS `sys.path[0]`, and comes first for the
        # same reason CPython puts it there -- unless `--safe-path` removes
        # it, as `-P` does.
        own = () if values.get("safe-path") else (context.source.parent,)
        extra = tuple(Path(p) for p in values.get("import-path", ()))
        py_imports.use(own + extra + host.roots, host)

        declared = py_nativelib.Registry()
        for path in values.get("native-library", ()):
            try:
                for library in py_nativelib.read(Path(path)).all():
                    declared.add(library)
            except py_nativelib.DeclarationError as exc:
                sink.report(error("E9109", f"--native-library: {exc}"))
                return None
        py_nativelib.use(declared, context.target_os)

        clone = copy.copy(self)
        clone.library = bool(values.get("library"))
        clone.verifying = context.verifying
        clone.language_version = version
        return clone

    def compile(self, source: SourceFile, sink: DiagnosticSink, *,
                library: bool | None = None) -> Module | None:
        # THE KEYWORD STILL WINS when a caller passes one. `objects/ir.py`
        # compiles the shipped runtime by calling this directly, with no
        # command line anywhere near it; `None` means "whatever `configure`
        # was told", which is what the driver's call means.
        library = self.library if library is None else library
        try:
            tree = ast.parse(source.text, filename=source.name)
        except SyntaxError as exc:
            sink.report(self._syntax_error(source, exc))
            return None

        # A bundled standard-library module becomes ordinary definitions
        # in this program, before anything looks at it. See `bundled.py`: it
        # is a splice, not an import system, and a program that uses none of
        # them comes back unchanged.
        # ASKED BEFORE THE SPLICE, because `__future__` is itself a bundled
        # module now and splicing consumes the very statement this reads.
        stringify = _stringifies(tree)
        # SAID BEFORE THE SPLICE CONSUMES THE NAME. `compile`, `eval` and
        # `exec` are rewritten to the bundled implementation, and after that
        # nothing downstream can tell they were ever written -- so the warning
        # about what they cost is reported from here, where the source still
        # says what the programmer wrote.
        _warn_about_runtime_compilation(tree, source, sink)
        # THE PROGRAM'S OWN MODULES FIRST, then the bundled standard library.
        # A spliced user module may `import functools`, and after the first
        # pass that statement is an ordinary one in the merged tree for the
        # second to resolve. The other order leaves it unspliced.
        tree = user_splice(tree, source, sink)
        tree = splice(tree, source, sink)
        # PEP 3151: `IOError` and `EnvironmentError` ARE `OSError` -- the same
        # object in CPython, not subclasses of it. Rewriting the name here is
        # what makes `IOError is OSError` True and an `except IOError` catch
        # an OSError; leaving them as distinct names would have given two
        # objects that print differently and never match each other.
        _Aliases().visit(tree)
        # PEP 563: `from __future__ import annotations` makes every annotation
        # its own SOURCE TEXT. Done here, as a rewrite, because that is what
        # the future import means -- nothing downstream needs to know about it
        # once each annotation is already the string it stands for.
        if stringify:
            _Stringify().visit(tree)

        analyzer = Analyzer(source, sink, library=library)
        functions = analyzer.run(tree)
        # WHAT THE LINKER HAS TO BE TOLD. `ctypes.CDLL("m")` is a promise that
        # `-lm` will be there; published here because the driver drives the
        # link and the frontend only knows what the source said.
        #
        # NOT ON A VERIFICATION RUN, because the linker is the only thing
        # that reads it and there will not be one. Analysis still WALKS the
        # `ctypes` calls -- `E0121` through `E0128` are its diagnostics and
        # they are the point of verifying -- so what is skipped is the
        # handoff and nothing else.
        if not self.verifying:
            cffi.name_libraries(analyzer.ctypes_libraries)
        if sink.failed:
            # Lowering assumes analysis succeeded. Running it anyway would
            # produce IR that fails the verifier, and the user would see an
            # internal-error report on top of the real diagnostics.
            return None
        return Lowerer(functions, source, analyzer, library=library).run()

    @staticmethod
    def _syntax_error(source: SourceFile, exc: SyntaxError):
        line = exc.lineno or 1
        col = max(0, (exc.offset or 1) - 1)
        starts = source.line_starts
        begin = starts[line - 1] + col if line <= len(starts) else 0
        return error("E0000", exc.msg or "invalid syntax").at(
            source.span(begin, begin + 1))


register(PythonFrontend())
