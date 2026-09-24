"""Standard-library modules written in Python and spliced into the program.

`functools.reduce` is twelve lines of ordinary Python. Writing it as runtime C
means writing it twice -- once in C and once in the interpreter's host -- and
keeping both agreeing with CPython forever. Writing it HERE means writing it
once, in the language it is specified in, and letting the compiler that is
being tested compile it.

WHAT THIS IS NOT: an import system. Nothing is loaded at run time and no
module object exists. A program that imports a bundled module has that
module's definitions SPLICED INTO IT, under names it cannot collide with, and
every reference rewritten to point at them. The result is one module, which is
the only shape the rest of the frontend knows.

THE CONSTRAINT THAT MAKES THIS HONEST: a bundled module is compiled by this
compiler, so it may only use what this compiler accepts. A construct one of
them cannot use is a gap worth closing rather than a reason to drop back to C.
"""
from __future__ import annotations

import ast
import pathlib

from .analysis import span_of
from .modules import resolve as _resolve
from ...diagnostics import error

#: Where the sources live. One file per module, named for it.
_HERE = pathlib.Path(__file__).parent / "bundled"

#: The prefix a spliced name gets. An IDENTIFIER, because the rest of the
#: frontend reads a dot in a function key as "nested inside", and a name with
#: one in it is not a module-level `def` to anything downstream. A program
#: could write this spelling itself; `splice` refuses rather than shadowing.
_MANGLE = "_asmpy_bundled_"


def available() -> set[str]:
    """Which modules are bundled."""
    return {p.stem for p in _HERE.glob("*.py")}


def _mangled(module: str, name: str, prefix: str = None) -> str:
    """One symbol for one (module, member), and never for two.

    THE DOT GOES. `collections.abc` is one bundled module with a dot in its
    name, and the rest of the frontend reads a dot in a key as "nested
    inside" -- so a name carrying one is not a module-level definition to
    anything downstream.

    AND THE DOT GOING IS WHY THIS NEEDS MORE THAN A REPLACE. It was
    `prefix + module.replace('.', '_') + '_' + name`, which is two collisions
    at once and both are SILENT -- one definition simply overwrites the other
    and the program prints the survivor:

        module `a.b`, member `X`  ->  ..._a_b_X
        module `a_b`, member `X`  ->  ..._a_b_X   (the dot and the underscore
                                                   flatten to the same thing)
        module `a`,   member `b_X` -> ..._a_b_X   (nothing says where the
                                                   module ends and the
                                                   member begins)

    Measured: a program importing `X` from `a.b` and from `a_b` printed
    'a_b' twice where CPython printed both.

    TWO CHANGES, ONE FOR EACH COLLISION. An underscore in the module doubles
    before the dot becomes one, which is the ordinary escape and makes the
    module component injective on its own. Then the component is LENGTH-
    PREFIXED, which is the only thing that can say where it ends, because a
    member name may contain an underscore too and no separator character is
    illegal in a Python identifier.

    The result is longer and still readable: `_asmpy_bundled_4_copy_Error`.
    Nothing parses it -- a mangled exception class is mapped back to its
    displayed name by LOOKING IT UP (`apy_exc_shown`), not by taking the
    string apart -- and the two places that strip a known prefix still do.
    """
    escaped = module.replace("_", "__").replace(".", "_")
    return f"{prefix or _MANGLE}{len(escaped)}_{escaped}_{name}"


#: BUILTINS A BUNDLED MODULE PROVIDES, and the module that provides each. A
#: program naming one gets that module spliced in, which is the whole
#: mechanism: nothing imports a builtin, so the name APPEARING is the only
#: signal there is.
#:
#: `open` IS HERE BECAUSE THE FILE OBJECTS ALREADY EXISTED. `io.open` and
#: everything under it -- `_RawFile`, `_TextReader`, `_TextWriter`, the modes,
#: the context manager, the line iteration -- were written and correct, and
#: the builtin name was simply never pointed at them, so `open(p)` was
#: `call to unknown function 'open'` while `io.open(p)` worked. Files were
#: reachable only through `pathlib`'s whole-file `read_text`/`write_text`.
#:
#: WHAT IT COSTS is `io` spliced into any program that says `open` -- the same
#: bargain `eval` already makes for `_pyrun`, and a better one, because a
#: program that opens a file was always going to want the module that opens
#: files.
_BUNDLED_BUILTINS = {"compile": "_pycompile",
                     "eval": "_pyrun", "exec": "_pyrun",
                     "open": "io"}


def module_of(mangled: str) -> str | None:
    """The bundled module a spliced name came from, or None for a program's
    own name.

    THE ONE PLACE THAT TAKES A MANGLED NAME APART, and it exists because
    `__module__` is a fact about WHERE a class was written and nothing else
    records it: a spliced class is an ordinary module-level definition by the
    time lowering sees it, and only its name still says it came from
    `fractions`. The length prefix is what makes the parse possible at all --
    see `_mangled` for the two collisions it is there to stop.
    """
    split = _split_mangled(mangled)
    if split is None:
        return None
    escaped, _bare = split
    # THE ESCAPE READ BACKWARDS: a doubled underscore was one in the module's
    # name and a single one was a dot. Walked rather than replaced, because
    # `a__b` and `a.b` both hold `__` after escaping and a blind replace
    # cannot tell which of the two it is looking at.
    out, i = "", 0
    while i < len(escaped):
        if escaped[i] == "_":
            if escaped[i:i + 2] == "__":
                out += "_"
                i += 2
            else:
                out += "."
                i += 1
            continue
        out += escaped[i]
        i += 1
    return out


def _split_mangled(mangled: str):
    """A spliced name as (escaped module, bare name), or None if it is not one.

    The length prefix is read once here so that `module_of` and `bare_of`
    cannot disagree about where the module ends and the name begins.
    """
    if not mangled.startswith(_MANGLE):
        return None
    rest = mangled[len(_MANGLE):]
    at = rest.find("_")
    if at <= 0 or not rest[:at].isdigit():
        return None
    width = int(rest[:at])
    escaped = rest[at + 1:at + 1 + width]
    if len(escaped) != width:
        return None
    after = rest[at + 1 + width:]
    if not after.startswith("_"):
        return None
    return escaped, after[1:]


def bare_of(mangled: str) -> str | None:
    """The name a spliced definition was WRITTEN under, or None for a
    program's own name.

    `fractions.Fraction` is spliced as one module-level class whose statement
    name carries the module, and a method of it takes its `__qualname__` from
    that statement -- so `Fraction.limit_denominator.__qualname__` read
    `_asmpy_bundled_9_fractions_Fraction.limit_denominator`, the mangling in
    plain sight. The splice restores `__name__` and `__qualname__` on the
    class itself (see the `_MANGLE` fixup below); this is the half a method
    needs, because its qualname was built from the class's key before any
    fixup runs.
    """
    split = _split_mangled(mangled)
    return None if split is None else split[1]


def _module_bindings(tree) -> set:
    """Every name the program itself binds at module level.

    A program with its own `def compile(...)` or `compile = something` means
    its own, and rewriting the call would send it somewhere else without
    saying so.
    """
    out = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            out.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    out.add(target.id)
        elif isinstance(stmt, ast.AnnAssign) \
                and isinstance(stmt.target, ast.Name):
            out.add(stmt.target.id)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                out.add(alias.asname or alias.name.split(".")[0])
    return out


class _Reserved(Exception):
    """A program used the prefix `splice` reserves for itself."""


def _no_member(sink, source, node, module: str, name: str, members,
               note: str = "") -> None:
    """`warnings.deprecated` when `warnings` is bundled and has no such name.

    THIS PASS IS THE ONLY ONE THAT CAN SAY IT. A bundled module is spliced and
    then GONE -- the import statement is dropped and no module object is left
    behind -- so by the time analysis runs there is nothing named `warnings`
    for it to have an opinion about. What it said instead was one of two wrong
    things, and both sent a reader to the wrong place:

      from warnings import deprecated
        -> `E0083: no module named 'warnings' is available; there is no import
           path`, listing modules that do not include it. Flatly false: the
           module is bundled, it was spliced into the program, and every other
           name in it worked. The reader goes looking for a missing module.

      import warnings; warnings.deprecated(...)
        -> NOTHING AT COMPILE TIME, and `NameError: name 'warnings' is not
           defined` at run time -- because the import was dropped and the
           attribute was the one reference the splice could not rewrite. The
           reader goes looking for a broken import.

    Neither is the truth, which is that the module is here and this member is
    not. Stating the coverage in a docstring (`docs/STDLIB.md`) is worth
    nothing if the compiler contradicts it, so the members it does have go in
    the diagnostic.
    """
    public = sorted(one for one in members[module] if not one.startswith("_"))
    # THE NOTE SAYS WHICH KIND OF MODULE IT IS. The mistake is the same for a
    # bundled module and for one of the program's own files, so the code and
    # the message are -- but "it covers part of CPython's module, see
    # docs/STDLIB.md" is false about a file the user wrote, and sends them to
    # read the standard library's coverage table about their own typo.
    report = (error("E0084", f"module {module!r} has no member {name!r}")
              .at(span_of(source, node))
              .note(note or
                    f"{module!r} is bundled: it is Python spliced into this "
                    f"program rather than an import, and it covers part of "
                    f"CPython's module. See docs/STDLIB.md"))
    if public:
        report = report.help("it provides: " + ", ".join(public))
    sink.report(report)


def _bound_locally(node) -> set:
    """The names a function body BINDS, so they are its own and not the
    module's.

    A local that happens to share a name with a module-level definition is a
    different variable -- `fields = [...]` inside a function that a module
    also defines `fields()` in -- and renaming it pointed the body at the
    function instead of at its own list. Silently: the code compiled and did
    something else.
    """
    out = set()
    for arg in getattr(node.args, "args", []) if hasattr(node, "args") else []:
        out.add(arg.arg)
    if hasattr(node, "args"):
        for group in ("posonlyargs", "kwonlyargs"):
            for arg in getattr(node.args, group, []) or []:
                out.add(arg.arg)
        for one in (node.args.vararg, node.args.kwarg):
            if one is not None:
                out.add(one.arg)
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and isinstance(inner.ctx,
                                                      (ast.Store, ast.Del)):
            out.add(inner.id)
        elif isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)):
            # A NESTED `def` OR `class` BINDS ITS NAME, and this did not
            # collect one -- so `def f(): def open(p): ...` left `open`
            # looking like the builtin and the body was rewritten to the
            # bundled `io.open`, calling something else entirely. The walk
            # starts AT `node`, so this adds the function's own name too;
            # `_Rename` already subtracts it, which is what that subtraction
            # was for.
            out.add(inner.name)
        elif isinstance(inner, ast.arg):
            out.add(inner.arg)
        elif isinstance(inner, ast.ExceptHandler) and inner.name:
            # THE BINDINGS THAT ARE PLAIN STRINGS, which a walk over `Name`
            # nodes cannot see: `except E as open`, `import io as open`, and
            # `case _ as open`. Each binds the name as surely as `open = ...`
            # does, and each was invisible here.
            out.add(inner.name)
        elif isinstance(inner, (ast.Import, ast.ImportFrom)):
            for alias in inner.names:
                out.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(inner, ast.MatchAs) and inner.name:
            out.add(inner.name)
        elif isinstance(inner, ast.MatchStar) and inner.name:
            out.add(inner.name)
        elif isinstance(inner, (ast.Global, ast.Nonlocal)):
            # DECLARED TO BE THE OUTER ONE, so it is not local after all.
            out.difference_update(inner.names)
    return out


def _declare_global(scope, names) -> None:
    """`global <names>` at the top of `scope`'s body -- after its docstring,
    which has to stay the first statement to go on being the docstring.

    WHY A REWRITE NEEDS ONE. `sys.stdout = buf` inside a function is an
    attribute store, and binds nothing in the function; rewritten to the
    spliced definition's name it becomes a NAME store, and a name stored to
    anywhere in a function is local to ALL of it. So the rewrite changed the
    meaning of every other mention of `sys.stdout` in that function: the
    `saved = sys.stdout` above the assignment read an unbound local and
    raised `UnboundLocalError` naming the mangled spelling, and the stream
    swap every redirecting helper is written around never reached the
    module. The declaration restores what the attribute store meant -- the
    module's binding, not a new one.
    """
    decl = ast.Global(names=sorted(names))
    body = scope.body
    at = 0
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        at = 1
    ast.copy_location(decl, body[min(at, len(body) - 1)])
    body.insert(at, decl)


def _dependencies(wanted, have):
    """Every bundled module needed, each before anything that imports it.

    A depth-first walk with the module emitted AFTER what it needs, which is
    the order the prelude has to be in: a spliced definition referring to
    another module's name is only correct once that name exists.

    `eval`, `exec` AND `compile` COUNT AS DEPENDENCIES HERE TOO. They are
    BUILTINS, so no import brings them in and nothing above this function
    would have noticed one -- the scan in `splice` walks the PROGRAM's tree
    and stops there. A bundled module calling `eval` in its own body
    therefore got `_pyrun` spliced only when the program happened to name
    `eval` as well, and otherwise failed with `error: 'eval'`.
    `bundled/annotationlib.py` documented it as the reason a stringized
    annotation could not be resolved.
    """
    order, seen = [], set()

    def visit(name):
        if name in seen:
            return
        seen.add(name)
        source = (_HERE / f"{name}.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename=f"<bundled {name}>")
        # `ast.walk` AND NOT `tree.body`, for the reason `splice` gives on
        # the program's side: `import copy` INSIDE A FUNCTION is the same
        # import. argparse makes six of them -- `copy`, `textwrap`,
        # `warnings`, `shutil`, `difflib` and `_colorize` are all imported
        # where they are used, to keep its own import cheap.
        for stmt in ast.walk(tree):
            if isinstance(stmt, ast.ImportFrom) and stmt.module in have:
                visit(stmt.module)
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.name in have:
                        visit(alias.name)
        # AFTER the imports and BEFORE this module is emitted, so the
        # compiler it needs is already in the prelude when its own body
        # arrives. `_module_bindings` keeps a module that defines its OWN
        # `compile` from being rewritten to somebody else's.
        own = _module_bindings(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in _BUNDLED_BUILTINS \
                    and node.id not in own:
                visit(_BUNDLED_BUILTINS[node.id])
        order.append(name)

    for one in dict.fromkeys(wanted):
        visit(one)
    return order


class _Rename(ast.NodeTransformer):
    """Rewrite the bundled module's own top-level names to mangled ones.

    Only the names the module DEFINES: a reference to `list` or `TypeError`
    inside it means the builtin, exactly as it would in any other program --
    and neither does a LOCAL that shares a name with one of them.
    """

    def __init__(self, module: str, defined: set[str],
                 borrowed: dict | None = None,
                 imported: dict | None = None,
                 members: dict | None = None,
                 prefix: str | None = None) -> None:
        #: WHOSE NAMESPACE THE MANGLED NAMES LIVE IN. `imports.py` reuses this
        #: transformer for a program's own modules and must not mint names
        #: under the bundled prefix -- `bundled.splice` refuses a tree that
        #: already contains one, on the grounds that a program writing the
        #: reserved spelling would otherwise have its own name replaced.
        self.prefix = prefix
        self.module = module
        self.defined = defined
        #: What this module imported FROM ANOTHER BUNDLED ONE, already
        #: mangled. `from _pylex import LexError` inside `_pycompile` binds
        #: nothing after the import statement is dropped, so the reference has
        #: to point at the name `_pylex` was spliced under.
        self.borrowed = borrowed or {}
        #: `import warnings` inside a bundled module: the local name, and the
        #: module it stands for. The statement is DROPPED -- the module is
        #: spliced instead -- so `warnings.warn(...)` has to be rewritten or
        #: the name is simply unbound, which is a NameError at run time for a
        #: module the source plainly imports.
        self.imported = imported or {}
        self.members = members or {}
        #: Names the enclosing function bodies bind. A name in here is theirs.
        self.shadowed: set = set()
        #: See `_Rewrite.stores`: the same bookkeeping, on this side.
        self.stores: list = []

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.shadowed:
            return node
        if node.id in self.defined:
            node.id = _mangled(self.module, node.id, self.prefix)
        elif node.id in self.borrowed:
            node.id = self.borrowed[node.id]
        elif node.id in _BUNDLED_BUILTINS and self.members is not None:
            # `eval`, `exec` AND `compile` IN A BUNDLED MODULE'S OWN BODY.
            # They are builtins, so nothing imports them and neither
            # `defined` nor `borrowed` has them -- the name went through
            # unrewritten and the program failed with `error: 'eval'`.
            # `_Rewrite` has had this branch for the PROGRAM's half since
            # `_pycompile` was written; this is the same rule on the other
            # side, and `_dependencies` is what makes sure the module
            # providing it is already in the prelude.
            #
            # `members` IS THE GATE, not a convenience: it holds what each
            # spliced module defines, so `compile in members["_pycompile"]`
            # is the test for "was the compiler actually spliced". A module
            # that names `eval` in a program where `_pyrun` was somehow not
            # brought in keeps the builtin's own refusal rather than being
            # rewritten to a name that does not exist.
            provider = _BUNDLED_BUILTINS[node.id]
            if node.id in self.members.get(provider, ()):
                node.id = _mangled(provider, node.id, self.prefix)
        return node

    def visit_FunctionDef(self, node):
        # The function's OWN name is bound outside it, so it is renamed with
        # the module's; everything the body binds is not.
        outer = self.shadowed
        self.shadowed = outer | (_bound_locally(node) - {node.name})
        self.stores.append(set())
        self.generic_visit(node)
        stored = self.stores.pop()
        if stored:
            _declare_global(node, stored)
        self.shadowed = outer
        if node.name in self.defined and node.name not in self.shadowed:
            node.name = _mangled(self.module, node.name, self.prefix)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Attribute(self, node):
        """`warnings.warn(...)` -> the name `warnings` was spliced under."""
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) \
                and node.value.id in self.imported \
                and node.value.id not in self.shadowed:
            module = self.imported[node.value.id]
            if node.attr in self.members.get(module, ()):
                name = _mangled(module, node.attr, self.prefix)
                if not isinstance(node.ctx, ast.Load) and self.stores:
                    self.stores[-1].add(name)
                return ast.copy_location(
                    ast.Name(id=name, ctx=node.ctx), node)
        return node

    def visit_ClassDef(self, node):
        """The class's own name is the module's; its methods' names are not.

        `_pyrun` has a top-level `eval()` AND a `_Walker.eval` method, and
        renaming the second with the first left the class without the method
        it plainly defines -- `'_Walker' object has no attribute 'eval'`, at
        run time, for a name the source has in front of it.

        ONLY THE NAME, and not references that happen to match it. Shadowing
        every mention inside the class was the first attempt and it was worse:
        `datetime` has a method called `date` and inherits from the CLASS
        `date`, so `class datetime(date)` stopped being rewritten and the base
        was reported as undefined. A base is evaluated OUTSIDE the body, and a
        method body cannot see its siblings unqualified either -- in both
        places the name means the module's.
        """
        methods = [one for one in node.body
                   if isinstance(one, (ast.FunctionDef, ast.AsyncFunctionDef))]
        kept = [one.name for one in methods]
        self.stores.append(set())
        self.generic_visit(node)
        stored = self.stores.pop()
        if stored:
            _declare_global(node, stored)
        # `generic_visit` renamed the method names along with everything
        # else; a method keeps the name its class body gave it.
        for one, name in zip(methods, kept):
            one.name = name
        if node.name in self.defined and node.name not in self.shadowed:
            node.name = _mangled(self.module, node.name, self.prefix)
        return node


class _Rewrite(ast.NodeTransformer):
    """Point the user's references at the spliced definitions.

    `functools.reduce(...)` becomes the mangled name; `from functools import
    reduce` becomes a binding of `reduce` to it, so the user's own spelling
    goes on working wherever it was already legal.
    """

    def __init__(self, imported: dict[str, str], members: dict[str, set],
                 names: dict[str, str], redirects: bool = False,
                 prefix: str | None = None) -> None:
        #: See `_Rename.prefix`.
        self.prefix = prefix
        #: Whether the program ASSIGNS to `sys.stdout`. See `visit_Name`.
        self.redirects = redirects
        #: local name -> module, for `import functools` and `import x as y`
        self.imported = imported
        self.members = members
        #: local name -> mangled, for `from functools import reduce`
        self.names = names
        #: Names the enclosing function bodies bind, as `_Rename` keeps for
        #: the other half of the same job. Only the BUILTIN rewrites below
        #: consult it, and they have to: a program with its own `open` in a
        #: function means its own, and the rewrite sent the call to `io`
        #: without saying so. A module-level binding was already safe by a
        #: different road -- it stops the provider being spliced at all, so
        #: `members` is empty and the branch never fires -- which is why this
        #: only ever went wrong one scope down.
        self.shadowed: set = set()
        #: One set per enclosing function or class body: the spliced names an
        #: attribute STORE there was rewritten to. See `_declare_global`.
        self.stores: list = []

    def _scoped(self, node):
        """Visit a function body with the names it binds held aside."""
        outer = self.shadowed
        self.shadowed = outer | _bound_locally(node)
        self.stores.append(set())
        try:
            self.generic_visit(node)
        finally:
            self.shadowed = outer
            stored = self.stores.pop()
        # A LAMBDA HAS NO STATEMENTS to declare anything in, and needs none:
        # its body is an expression, and an expression cannot store to an
        # attribute.
        if stored and not isinstance(node, ast.Lambda):
            _declare_global(node, stored)
        return node

    def visit_FunctionDef(self, node):
        return self._scoped(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        return self._scoped(node)

    def visit_ClassDef(self, node):
        # A CLASS BODY IS A SCOPE TOO, for this purpose: a name stored there
        # becomes a class attribute, where the attribute store it replaced
        # changed the module.
        self.stores.append(set())
        try:
            self.generic_visit(node)
        finally:
            stored = self.stores.pop()
        if stored:
            _declare_global(node, stored)
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        # REWRITTEN, not bound to a variable. Binding `wraps = <mangled>` made
        # it a module-level VALUE, and a decorator written `@wraps(fn)` is a
        # call whose callee the frontend then looked for among its functions.
        # Pointing the name straight at the definition keeps it a `def`.
        if node.id in self.names:
            node.id = self.names[node.id]
            return node
        # PEP 553: `breakpoint()` IS `sys.breakpointhook()`, by definition
        # rather than by convention. Rewritten only when the program brought
        # `sys` in -- without it there is nothing to have replaced the hook,
        # and the builtin's own no-op is the whole behaviour.
        if node.id in self.shadowed:
            # THE PROGRAM'S OWN, in this scope. Every rewrite below replaces a
            # BUILTIN name, and a name the body binds is not one.
            return node
        if node.id == "breakpoint"                 and "breakpointhook" in self.members.get("sys", ()):
            node.id = _mangled("sys", "breakpointhook")
        # `print` GOES THROUGH `sys.stdout` ONLY FOR A PROGRAM THAT REPLACES
        # IT. Routing every program's printing through Python would cost the
        # ones that never redirect anything, and the direct call is the whole
        # of what they need.
        if node.id == "print" and self.redirects:
            node.id = _mangled("sys", "_print")
        # `open()`, `compile()`, `eval()` and `exec()` ARE THE BUNDLED ONES.
        # See `_BUNDLED_BUILTINS`: the names are builtins, so nothing imports
        # them, and the module is spliced because the name appears at all.
        module = _BUNDLED_BUILTINS.get(node.id)
        if module is not None and node.id in self.members.get(module, ()):
            node.id = _mangled(module, node.id, self.prefix)
        return node

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in self.imported:
            module = self.imported[node.value.id]
            if node.attr in self.members[module]:
                name = _mangled(module, node.attr, self.prefix)
                if not isinstance(node.ctx, ast.Load) and self.stores:
                    self.stores[-1].add(name)
                return ast.copy_location(
                    ast.Name(id=name, ctx=node.ctx), node)
        return node


def _hoist_imports(parsed, have, defined, borrowed, brought) -> None:
    """A bundled module's imports of another bundled module that sit INSIDE
    a function or a class, dropped and recorded as if written at the top.

    `import copy` inside a function binds a LOCAL, and `_bound_locally` says
    so -- which is exactly what stopped `copy.copy(items)` from being
    rewritten to the spliced definition: a local is the function's own
    business. With the statement removed there is no local any more, and the
    name means the module everywhere in this file that does not bind it
    itself, which is also what it meant in CPython, where every one of those
    imports returns the same cached module object.

    A NAME THE MODULE ALSO DEFINES IS LEFT ALONE rather than guessed at: two
    meanings for one spelling in one file cannot share a module-wide map,
    and leaving the statement in place reports the import as unresolved
    instead of quietly pointing one of them at the other.
    """
    for node in ast.walk(parsed):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            continue
        for holder in ast.walk(node):
            for field in ("body", "orelse", "finalbody"):
                stmts = getattr(holder, field, None)
                if not isinstance(stmts, list):
                    continue
                kept = []
                for stmt in stmts:
                    if isinstance(stmt, ast.Import) \
                            and all(a.name in have for a in stmt.names) \
                            and not any((a.asname or a.name) in defined
                                        for a in stmt.names):
                        for alias in stmt.names:
                            brought[alias.asname or alias.name] = alias.name
                        continue
                    if isinstance(stmt, ast.ImportFrom) \
                            and stmt.module in have \
                            and not any((a.asname or a.name) in defined
                                        for a in stmt.names):
                        for alias in stmt.names:
                            borrowed[alias.asname or alias.name] = _mangled(
                                stmt.module, alias.name)
                        continue
                    kept.append(stmt)
                if len(kept) != len(stmts):
                    # A BODY CANNOT BE EMPTY, and one holding nothing but the
                    # import would be.
                    stmts[:] = kept or [ast.copy_location(ast.Pass(),
                                                          stmts[0])]


def splice(tree: ast.Module, source, sink) -> ast.Module:
    """Rewrite `tree` so its bundled imports become ordinary definitions.

    Returns the tree unchanged when it imports none of them, so a program that
    uses no bundled module pays nothing and looks exactly as it did.

    `source` and `sink` are REQUIRED rather than optional. This pass is the
    only one that can report a reference to a member a bundled module does not
    have -- see `_no_member` -- and a sink that defaults to None is a sink that
    silently swallows every one of those the day someone calls this from
    somewhere new.
    """
    have = available()
    wanted, imported, aliases = [], {}, []
    at = None
    # `compile`, `eval` and `exec` ARE BUILTINS, so no import brings them in
    # -- the name appearing is what does. `_pycompile` is spliced exactly as
    # an imported module is, and the name is rewritten to point at it.
    #
    # ONLY WHEN THE PROGRAM DOES NOT BIND THE NAME ITSELF: a program with its
    # own `def compile(...)` means its own, and rewriting the call would
    # silently send it somewhere else.
    bound = _module_bindings(tree)
    uses = {name for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in _BUNDLED_BUILTINS
            and node.id not in bound
            for name in (node.id,)}
    for one in sorted(uses):
        wanted.append(_BUNDLED_BUILTINS[one])
    if uses:
        at = at or next((n for n in ast.walk(tree)
                         if isinstance(n, ast.Name) and n.id in uses), None)
    # `ast.walk`, NOT `tree.body`: an `import` INSIDE A FUNCTION is the same
    # import. The scan was module-level only, so
    #
    #     def f():
    #         import heapq
    #
    # never made `heapq` wanted, nothing was spliced, the statement survived
    # into analysis and came back `E0083: no module named 'heapq' is
    # available` -- with `heapq` listed in that same diagnostic's own
    # "available:" line, which is as self-contradicting as a compiler gets.
    # A builtin module (`math`) worked there all along, so the gap was
    # invisible until a bundled one was tried.
    #
    # NOTHING DOWNSTREAM NEEDS THE DISTINCTION. A splice has no scoping to
    # respect: what it produces is module-level definitions under mangled
    # names, and `_Rewrite` already walks into function bodies to repoint the
    # references. Hoisting the DEFINITIONS out of the function that imported
    # them is what CPython effectively does too -- the module object is built
    # once and cached in `sys.modules`, and a function-level `import` after
    # the first is a dictionary lookup.
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name in have:
                    wanted.append(alias.name)
                    imported[alias.asname or alias.name] = alias.name
                    at = at or stmt
        elif isinstance(stmt, ast.ImportFrom) and stmt.module in have:
            wanted.append(stmt.module)
            at = at or stmt
    if not wanted:
        return tree
    # A program that writes the reserved prefix itself would have its own name
    # replaced silently. Refusing is the only honest answer, and it costs a
    # walk of a tree that is already in memory.
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith(_MANGLE):
            raise _Reserved(node.id)

    prelude, members = [], {}
    # A BUNDLED MODULE MAY IMPORT ANOTHER, and until `_pycompile` none did.
    # The list is walked while it grows, so a module pulled in by another is
    # spliced too -- `_pycompile` needs `_pylex`, `_pyparse` and
    # `_pyvalidate`, none of which the program ever names.
    #
    # BEFORE ITS IMPORTERS, so that the names it defines are already known
    # when the importer's own references are rewritten. `_dependencies` is
    # what puts them in that order.
    order = _dependencies(wanted, have)
    for module in order:
        # NOT `source`: that is the SourceFile of the program being compiled,
        # which `_no_member` needs to turn a node into a span.
        text = (_HERE / f"{module}.py").read_text(encoding="utf-8")
        parsed = ast.parse(text, filename=f"<bundled {module}>")
        defined = {s.name for s in parsed.body
                   if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef))}
        # A MODULE-LEVEL VARIABLE IS A MEMBER TOO. `sys.monitoring` is an
        # object the module builds rather than a class it defines, and a
        # `defined` set that held only definitions left every such name
        # unrewritten -- so the reference reached the builtin table, which
        # does not have it.
        for stmt in parsed.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        defined.add(target.id)
            elif isinstance(stmt, ast.AnnAssign)                     and isinstance(stmt.target, ast.Name):
                defined.add(stmt.target.id)
        members[module] = defined
        # WHAT THIS MODULE IMPORTED FROM ANOTHER BUNDLED ONE. `from _pylex
        # import LexError` inside `_pycompile` has to become the mangled name
        # `_pylex` was spliced under -- the import statement itself is dropped
        # below, so nothing else would bind it.
        borrowed, brought = {}, {}
        body = []
        for stmt in parsed.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module in have:
                for alias in stmt.names:
                    borrowed[alias.asname or alias.name] = _mangled(
                        stmt.module, alias.name)
                continue
            if isinstance(stmt, ast.Import) \
                    and all(a.name in have for a in stmt.names):
                # DROPPED, and remembered: `warnings.warn` has to point at the
                # name `warnings` was spliced under, or it is a NameError for
                # a module the source plainly imports.
                for alias in stmt.names:
                    brought[alias.asname or alias.name] = alias.name
                continue
            body.append(stmt)
        parsed.body = body
        _hoist_imports(parsed, have, defined, borrowed, brought)
        renamed = _Rename(module, defined, borrowed, brought,
                          members).visit(parsed)
        # The docstring goes: it is the module's, and a spliced statement that
        # binds nothing is one more thing for the class-body and entry rules
        # to have an opinion about.
        for stmt in renamed.body:
            if (isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)):
                continue
            prelude.append(stmt)
            # THE MANGLED NAME IS AN IMPLEMENTATION DETAIL AND A PROGRAM CAN
            # SEE IT: `type(C).__name__` is the metaclass's `__name__`, and
            # that is the name of the `class` statement. Renaming the
            # statement is how the binding avoids colliding with the user's
            # names; restoring `__name__` after it is how the collision stays
            # invisible.
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)) and stmt.name.startswith(
                                     _MANGLE):
                real = stmt.name[len(_mangled(module, "")):]
                for dunder in ("__name__", "__qualname__"):
                    fixup = ast.Assign(
                        targets=[ast.Attribute(
                            value=ast.Name(id=stmt.name, ctx=ast.Load()),
                            attr=dunder, ctx=ast.Store())],
                        value=ast.Constant(value=real))
                    # MARKED FOR THE ANALYSER, which is the only thing that
                    # knows whether this `def` will take the STATIC path. A
                    # static function is machine words and has no object to
                    # carry a `__name__`, so this assignment reads a global
                    # that is never stored and traps -- see
                    # `_is_splice_dunder`. The splice cannot decide it here:
                    # the rule is thirty lines of annotation analysis and a
                    # second copy of it would drift.
                    fixup.splice_dunder = True
                    prelude.append(fixup)

    # EVERY SPLICED NODE POINTS AT THE IMPORT. The positions it arrives with
    # are lines in the bundled file, which do not exist in the program being
    # compiled -- a diagnostic carrying one indexed past the end of the source
    # and crashed the compiler. The import is the honest place to point: it is
    # where this code entered the program.
    line = getattr(at, "lineno", 1)
    col = getattr(at, "col_offset", 0)
    for stmt in prelude:
        for node in ast.walk(stmt):
            if hasattr(node, "lineno"):
                node.lineno = node.end_lineno = line
                node.col_offset = col
                node.end_col_offset = col + 1

    # `from functools import reduce` REWRITES the user's spelling to the
    # spliced definition rather than binding a variable to it.
    names = {}

    def without_bundled(body: list) -> list:
        """`body` with its bundled imports dropped, or trimmed to the half
        this does not cover. Whatever `from ... import` bound is recorded in
        `names` for `_Rewrite` to repoint."""
        out = []
        for stmt in body:
            if isinstance(stmt, ast.Import) \
                    and all(a.name in have for a in stmt.names):
                # THE IMPORT SURVIVES when the module also exists as a builtin
                # one: `sys` is bundled IN PART -- `audit` and `monitoring` are
                # ordinary Python while `maxsize` is a compiler constant -- and
                # dropping the statement unbound the half this does not cover.
                also = [a for a in stmt.names if _resolve(a.name) is not None]
                if also:
                    out.append(ast.copy_location(ast.Import(names=also), stmt))
                continue
            if isinstance(stmt, ast.ImportFrom) and stmt.module in have:
                left = []
                for alias in stmt.names:
                    if alias.name not in members[stmt.module]:
                        # NOT SOMETHING THE BUNDLED MODULE DEFINES, so the
                        # import of it SURVIVES -- WHEN THERE IS SOMETHING ELSE
                        # TO SURVIVE INTO. A module can be bundled IN PART --
                        # `typing`'s special forms are already runtime values
                        # and only its classes need writing -- and dropping the
                        # whole statement unbound the half this does not cover.
                        #
                        # When nothing else provides the module, leaving the
                        # statement handed analysis an import it could not
                        # resolve and produced a diagnostic denying the module
                        # exists. See `_no_member`.
                        if _resolve(stmt.module) is None:
                            _no_member(sink, source, stmt, stmt.module,
                                       alias.name, members)
                            continue
                        left.append(alias)
                        continue
                    names[alias.asname or alias.name] = _mangled(stmt.module,
                                                                 alias.name)
                if left:
                    out.append(ast.copy_location(
                        ast.ImportFrom(module=stmt.module, names=left,
                                       level=stmt.level), stmt))
                continue
            out.append(stmt)
        return out

    kept = without_bundled(tree.body)
    # AND EVERY NESTED BODY, for the same reason the collection above walks
    # the whole tree: `def f(): import heapq` is an import of a bundled
    # module, and leaving the statement in place hands analysis one it cannot
    # resolve. Function bodies, class bodies, `if`/`try`/`with`/loop bodies --
    # anything holding statements, at any depth.
    #
    # Mutating during `ast.walk` is safe here because what is REMOVED is only
    # ever an `Import` or `ImportFrom`, and neither holds statements of its
    # own: a dropped node the walk has already queued contributes nothing when
    # its turn comes.
    for stmt in kept:
        for child in ast.walk(stmt):
            for field, value in ast.iter_fields(child):
                if isinstance(value, list) and any(
                        isinstance(one, ast.stmt) for one in value):
                    setattr(child, field, without_bundled(value))

    # `warnings.deprecated` WHERE `warnings` IS BUNDLED AND HAS NO SUCH NAME.
    # Reported HERE, before `_Rewrite` runs, because afterwards the attributes
    # that DID resolve are gone and the ones left look like an attribute of an
    # ordinary object. The condition is exactly the one under which
    # `_Rewrite.visit_Attribute` declines to rewrite and the import statement
    # was dropped -- so the diagnostic and the rewrite cannot disagree.
    for stmt in kept:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Attribute) \
                    or not isinstance(node.value, ast.Name):
                continue
            module = imported.get(node.value.id)
            if module is None or node.attr in members[module] \
                    or _resolve(module) is not None:
                continue
            _no_member(sink, source, node, module, node.attr, members)

    # DOES THE PROGRAM REPLACE `sys.stdout`? Only then is `print` routed
    # through it -- see `_Rewrite.visit_Name`.
    redirects = any(
        isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
        and n.attr in ("stdout", "stderr")
        and isinstance(n.value, ast.Name) and n.value.id in imported
        for n in ast.walk(tree))
    out = ast.Module(body=prelude + kept, type_ignores=tree.type_ignores)
    # The prelude is already mangled; only the user's half is rewritten, so a
    # bundled module that happens to define `reduce` is not renamed twice.
    out.body[len(prelude):] = [
        _Rewrite(imported, members, names, redirects).visit(s)
        for s in kept]
    ast.fix_missing_locations(out)
    return out
