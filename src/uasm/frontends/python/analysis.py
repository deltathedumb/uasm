"""Name resolution and type checking for the Python subset.

Analysis is one traversal producing two products: a symbol table saying what
each name refers to, and a type for every expression. They are computed
together because for an annotated subset they are mutually dependent -- the
type of `x` is the type of the symbol `x` resolves to -- and separating them
would mean two traversals passing a partially-filled table between them.

That is a real design choice with a real cost. A dynamically-typed frontend
could NOT do this: it would need resolution complete before inference starts.
The staging here is `analysis -> lowering`, and lowering may assume every
expression already has a type.

POISONING. When analysis cannot determine a type it yields `ERROR` and reports
once. Every operation on `ERROR` produces `ERROR` silently, so a single unknown
name generates one diagnostic rather than one per use. Lowering is never
reached if any error was reported, so it may assume no `ERROR` remains.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

from . import cffi, modules, nativelib
from .modules import importable, member, resolve

from ...diagnostics import (DiagnosticSink, SourceFile, Span, error,
                            warning)
from ...ir import types as T
from ...objects.floor import FLOOR as _FLOOR
from ...objects import hostsvc as _hostsvc

#: `object.<dunder>` -- the DEFAULT implementations, and the runtime call each
#: one is. A class that overrides a dunder reaches its default this way, which
#: is the only way out of the recursion `__getattribute__` would otherwise be:
#:
#:     def __getattribute__(self, name):
#:         log.append(name)
#:         return object.__getattribute__(self, name)
#:
#: `object` is not a value here -- there is no type object to hold these -- so
#: the attribute is resolved at the call site, by name.
OBJECT_DEFAULTS = {
    "__getattribute__": ("apy_default_getattr", 2),
    "__setattr__": ("apy_default_setattr", 3),
    "__delattr__": ("apy_default_delattr", 2),
    "__repr__": ("apy_default_repr", 1),
    "__str__": ("apy_default_repr", 1),
    "__eq__": ("apy_default_eq", 2),
    # NOT `apy_default_eq`, WHICH IS WHAT THIS SAID. The two names sat on
    # adjacent lines mapping to one symbol, so a written `object.__ne__(x, y)`
    # computed EQUALITY -- `object.__ne__(1, 1)` was True and
    # `object.__ne__(1, 2)` False, each the exact opposite of CPython's. It
    # reads as a copied line rather than a decision.
    "__ne__": ("apy_default_ne", 2),
    "__hash__": ("apy_default_hash", 1),
    "__init__": ("apy_default_init", 1),
}

#: Names that are VALUES without being assigned. `Ellipsis` is the spelling
#: `...` also has, and both are the one singleton cell.
_SINGLETON_NAMES = frozenset({"Ellipsis", "NotImplemented"})

#: THE TWO NODES A `def` CAN BE. Named because the distinction almost never
#: matters -- an `async def` binds a name, takes parameters and has a body
#: exactly as a `def` does, and the one place it differs is what calling it
#: builds. Every filter that reached for `ast.FunctionDef` alone silently
#: dropped every `async def` in the module.
_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)

#: The frontend's own view of a type. Distinct from `ir.types` on purpose:
#: `bool` and `int` are different here and both lower to integers, and ERROR
#: has no IR counterpart at all.
@dataclass(frozen=True, slots=True)
class SemType:
    name: str

    def __str__(self) -> str:
        return self.name

    @property
    def is_numeric(self) -> bool:
        """Whether `+` and friends apply. `ptr` is deliberately NOT numeric:
        pointer arithmetic goes through `offset`, which says in bytes what it
        is doing, rather than through a `+` whose meaning would depend on an
        element type this IR does not have."""
        return self.name in ("int", "float", "bool") or (
            self.is_machine and self.name != "ptr")

    @property
    def is_error(self) -> bool:
        return self.name == "<error>"

    # ── the machine subset ──────────────────────────────────────────────────
    @property
    def is_machine(self) -> bool:
        return self.name in MACHINE

    @property
    def is_machine_int(self) -> bool:
        return self.is_machine and self.name[0] in "iu"

    @property
    def is_machine_float(self) -> bool:
        return self.is_machine and self.name[0] == "f"

    @property
    def is_ptr(self) -> bool:
        return self.name == "ptr"


INT = SemType("int")
FLOAT = SemType("float")
BOOL = SemType("bool")
NONE = SemType("None")
ERROR = SemType("<error>")

#: The type of a value whose type is not known until it runs -- which in
#: Python is most of them. A DYNAMIC function (see `FunctionInfo.dynamic`)
#: gives every expression this type and every operation becomes a call into
#: the object runtime, where the value carries its own kind.
#:
#: This is not "unknown, guess later". It is a real type with a real
#: representation, and that distinction is the point: the compiler this
#: replaces used its `int` as a stand-in for "no idea", so a value's
#: representation followed the slot it was stored in rather than the value,
#: and one root cause surfaced as a dozen unrelated-looking bugs.
OBJ = SemType("object")

#: THE MACHINE SUBSET. The IR's own types, spelled as annotations, named
#: identically to `ir.types` so that a diagnostic, the IR text and the backend
#: all say `u32` and mean the one thing.
#:
#: WHY THESE EXIST AT ALL. uasm's object runtime is 15,560 lines of C plus
#: an 8,583-line Python re-implementation for the IR interpreter, and a backend
#: that wants dynamic Python has to go and find 229 `apy_*` symbols -- which is
#: why exactly one backend has them. `docs/INERT-RUNTIME.md` argues the way out
#: is to write that runtime in IR instead, and the static path is where it gets
#: written: the static path emits NO `apy_*` at all, so a runtime written in it
#: stands on the machine rather than on itself. These names are the vocabulary
#: for doing that.
#:
#: They are NOT for ordinary Python. A program that wants an integer writes
#: `int`; these are for code that has to know a width because something else
#: reads the same bytes.
#:
#: `bool` is already `i1` and `i1` is not spelled here -- one name per storage
#: class, and `bool` is the one Python has.
MACHINE = {name: SemType(name) for name in
           ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
            "f32", "f64", "ptr")}

PTR = MACHINE["ptr"]

#: The memory intrinsics, and how many arguments each takes.
#:
#: Each is ONE IR INSTRUCTION, not a call -- `load(i64, p)` emits `i64.load`
#: and nothing else. Naming them rather than inventing syntax (`p[0]`, `*p`,
#: `&x`) is deliberate: the systems subset stays parseable by `ast`, so it
#: stays readable by every tool that already reads Python, and the grammar
#: this frontend has to accept does not grow at all.
#:
#: A program that defines its own `def load(...)` keeps it. These are looked up
#: only after the user's own functions, exactly as `int` is.
#: `reserve(name, size)` is the odd one out and is here because it is the
#: fourth way to get an address: `alloca` gives frame storage that dies at the
#: return, `plat_heap` gives heap the program asked for, a parameter gives
#: someone else's -- and this gives STATIC storage, zeroed, with a name, for
#: the whole run.
#:
#: A runtime needs it and nothing else here provides it. The small-integer
#: cache is the motivating case: `a = 1; b = 1; a is b` is True in CPython
#: because -5..256 are shared cells, and a cache has to outlive every call that
#: touches it.
MEMORY_INTRINSICS = {"alloca": 1, "load": 2, "store": 3, "offset": 2,
                     "sizeof": 1, "reserve": 2,
                     # INDIRECT CALLS, deferred three times by
                     # docs/INERT-RUNTIME.md and the last thing standing
                     # between the ported runtime and classes, generators and
                     # every dunder lookup -- all of which dispatch through a
                     # function pointer rather than a kind tag.
                     "funcaddr": 1,
                     # CONSTANT BYTES, which the subset had no way to express
                     # at all: `b"..."` is `E0040: unsupported expression`, so
                     # every runtime function that builds a string from a
                     # literal was unportable -- the type name in a repr, the
                     # name of an exception, every message. The IR could
                     # always hold one (`global __str0 = "..." readonly`);
                     # what was missing was a spelling.
                     "rodata": 1,
                     # VARIADIC, which `-1` means here: at least a type and an
                     # address, then however many arguments the callee takes.
                     "callptr": -1}

#: What each yields when its arity was wrong, so one mistake makes one
#: diagnostic instead of a cascade about the expression it sits in.
_INTRINSIC_RESULT = {"alloca": PTR, "load": None, "store": NONE,
                     "offset": PTR, "sizeof": INT, "reserve": PTR,
                     "funcaddr": PTR, "callptr": None, "rodata": PTR}

#: THE PLATFORM FLOOR, callable from the static path. Three functions -- emit
#: bytes, stop, get memory -- and everything else a runtime needs is code that
#: has not been written in this subset yet. See `objects/floor.py` for the
#: contracts and `docs/INERT-RUNTIME.md` for why the number is three.
#:
#: Ordinary calls, not intrinsics: each lowers to `Op.CALL` of an external
#: symbol, which is what makes them the ONE thing a backend still has to
#: supply. They are reachable by name without an import because the set is
#: CLOSED by design -- a fourth would be a change to the contract every backend
#: implements, and it should read like one rather than appear as an import.
#:
#: READ OUT OF `objects/floor.py`, which is where the contracts are written and
#: where the C implementation sits. A hand-kept second copy of a signature list
#: drifted three times in one afternoon when `_OBJECT_RUNTIME` was one; the
#: fix there was to parse the definitions, and the fix here is to have one list.
BY_NAME = {"int": INT, "float": FLOAT, "bool": BOOL, "None": NONE,
           "object": OBJ, **MACHINE}

PLATFORM = {name: (tuple(BY_NAME[a] for a in args),
                   NONE if ret == "void" else BY_NAME[ret])
            for name, (args, ret) in _FLOOR.items()}

#: THE HOST SERVICES, callable from the static path exactly as the floor is.
#:
#: Everything in `objects/hostsvc.py` EXCEPT its `core` group, which is the floor
#: and is already above -- one name declared twice would be two IR imports of
#: the same symbol, and the second would win silently.
#:
#: NOT AN IMPORT, for the same reason the floor is not: these are the contract
#: between a frontend and a backend rather than a library, and a program that
#: uses one should read as though it is reaching for the machine. What differs
#: from the floor is that a backend may not HAVE the group -- which is checked
#: where the backend is known (`Backend.check_host_services`) rather than here,
#: because the frontend does not know which backend it is compiling for.
HOSTSVC = {name: (tuple(BY_NAME[a] for a in args),
                  NONE if ret == "void" else BY_NAME[ret])
           for name, (args, ret) in _hostsvc.ALL.items()
           if _hostsvc.GROUP_OF[name] not in _hostsvc.MANDATORY}


def _object_runtime() -> dict:
    """The `apy_*` runtime, callable FROM the machine subset.

    THE POINT OF THIS IS TO PORT IT. `docs/INERT-RUNTIME.md` replaces the C
    object runtime one kind at a time, and a half-ported runtime is one where
    the ported functions call the ones that are still C -- `apy_add`'s integer
    path is subset code, and its fallthrough for a str is not, yet. Without a
    way to call across that line, a port would have to be all-or-nothing.

    THE SIGNATURES ARE READ OUT OF THE C by `objects/csource.signatures()`, the
    same parse the frontend's own declarations come from, so this cannot
    disagree with what a program links against. `apy_value` is `ptr` there,
    which is why a runtime function written here takes and returns `ptr`.

    A user program can reach these too. That is not a hole: a program that
    writes `store(i64, ...)` can already do anything, and the `apy_` prefix is
    not something anyone arrives at by accident.
    """
    from ...objects.csource import signatures
    from ...objects.ir import SPLIT
    out = {}
    for name, (args, ret) in signatures().items():
        sig = (tuple(BY_NAME[a] for a in args),
               NONE if ret == "void" else BY_NAME[ret])
        out[name] = sig
        if name in SPLIT:
            # THE OTHER HALF OF A SPLIT FUNCTION. `apy_add`'s C body is
            # renamed `apy_add_slow` and the ported fast path calls it for
            # every kind it does not handle -- so the name has to be callable,
            # and it has the same signature because it IS the same function.
            # Derived rather than parsed: the rename happens in
            # `objects_c(split=...)`, long after this list is read.
            out[name + "_slow"] = sig
    return out


#: Built once. `signatures()` parses 15,560 lines of C and every `Analyzer`
#: would otherwise repeat it -- which showed up as the frontend taking longer
#: to start than to compile.
OBJECT_RUNTIME = _object_runtime()

#: `apy_*` object-runtime primitives a bundled DYNAMIC module may call by
#: bare name. See `dynamic.py`'s `_OBJRT_DYNAMIC_NAMES` -- the same small,
#: explicit set (duplicated rather than imported, the same relationship
#: `dynamic.py`'s own `_HOSTSVC_NAMES` already has with this file's
#: `HOSTSVC`: each side reads what it needs from the source of truth,
#: `objects/csource.py`'s signatures by way of `OBJECT_RUNTIME` here).
#: Every argument and the result are `ptr`, checked once, here, rather
#: than trusted per name -- see `_dyn_objrt_call`'s own comment for why
#: that has to stay true for every name added to this set.
_OBJRT_DYNAMIC_NAMES = frozenset({
    "apy_weakref_register", "apy_weakref_deref", "apy_weakref_count",
    "apy_weakref_existing", "apy_gc_collect",
})
assert _OBJRT_DYNAMIC_NAMES <= OBJECT_RUNTIME.keys()
assert all(all(t is PTR for t in args) and ret is PTR
          for name in _OBJRT_DYNAMIC_NAMES
          for args, ret in [OBJECT_RUNTIME[name]])

#: What the module's top-level statements are called internally. Not a legal
#: Python identifier, so it can never collide with a user function -- including
#: one actually named `main`, which stays an ordinary function that the entry
#: may call. Lowering renames it to `main`, which is what the backends and the
#: C runtime agree the entry point is called.
ENTRY_NAME = "<module>"


#: Builtins a dynamic function may call, and their arity (None = any). Small
#: on purpose: every entry is a thing the object runtime actually implements,
#: and a name accepted here that lowering cannot emit is a crash rather than a
#: diagnostic.
#: The builtin CLASSES that are values in their own right. `object` is what
#: `object.__new__(cls)` names and `type` is a metaclass's base; neither is a
#: kind the way `int` is, so neither travels as text.
_CLASS_VALUES = frozenset({"object", "type"})

#: The builtins that would need A COMPILER INSIDE THE PRODUCED BINARY.
#:
#: Refused with a message that says so, in call position AND as a value: a
#: program that merely names one has to hear the real reason rather than
#: `name 'eval' is not defined`, which is a false statement about Python.
#:
#: This is the one deliberate limitation that costs conformance cases -- 19 of
#: them, all calling `compile()` on a bad program and expecting a SyntaxError
#: -- so it is refused loudly and TESTED, rather than left to become a quiet
#: wrong answer the day somebody half-implements one of them.
_NEEDS_A_COMPILER = frozenset({"compile", "eval", "exec"})

#: The builtin kinds a class may extend. `class D(dict)` gives every instance
#: a real dict of its own for everything the body does not write, which is
#: what makes a subclass with only `__missing__` in it behave.
#:
#: A NAME HERE NEEDS A ROW IN `_BUILTIN_BASE_KIND` (dynamic.py) and nowhere
#: else in this file: this set is the whole of the frontend gate, and every
#: other thing the frontend does with a builtin base reads `info.builtin_base`
#: or that table. The pairing is not optional -- the lowering indexes the
#: table unguarded, so a name added here alone is a KeyError rather than a
#: diagnostic.
#:
#: `bytearray` IS NOT HERE and cannot be until the kind carries a mut flag:
#: it shares `APY_BYTES_K` with bytes, so a row for it would build instances
#: holding an immutable empty bytes. See `_BUILTIN_BASE_KIND`.
_BUILTIN_BASES = frozenset({"dict", "list", "set", "tuple", "str", "bytes"})

_DYN_BUILTINS = {
    # `None` WHERE THE COUNT VARIES. `list()`, `tuple()`, `str()` and
    # `float()` are all legal with no argument -- they answer the empty or
    # zero value of their type, which is how `defaultdict(list)` builds one --
    # and requiring exactly one rejected a program CPython accepts.
    "print": None, "int": None, "float": None, "str": None,
    "repr": 1, "len": 1, "list": None, "tuple": None, "bool": None,
    # `type(x)` asks, `type(name, bases, ns)` MAKES -- the three-argument
    # form is the `class` statement written out, and builds the same
    # object a metaclass's `super().__new__` does.
    "type": None,
    "sorted": 1, "min": None, "max": None, "sum": None, "reversed": 1,
    "enumerate": None, "zip": None, "range": None, "abs": 1, "round": None,
    "isinstance": 2, "set": None, "frozenset": None, "complex": None,
    "ord": 1, "chr": 1, "ascii": 1, "bin": 1, "hex": 1, "oct": 1, "hash": 1,
    "callable": 1, "all": 1, "any": 1, "divmod": 2, "pow": None,
    "id": 1,
    # `__import__(name)` -- a DYNAMIC import, which this compiler
    # cannot perform. It answers an ImportError either way, which is
    # what a program guarding an optional import already handles.
    "__import__": None,
    # `object()` -- a bare instance, which is what a program uses as a
    # UNIQUE SENTINEL: nothing else compares equal to it.
    "object": 0,
    # The runtime descriptors. Written as decorators nearly always, which is
    # an ordinary one-argument call by the time it reaches here.
    "property": 1, "classmethod": 1, "staticmethod": 1,
    # `slice(stop)`, `slice(start, stop)`, `slice(start, stop, step)`.
    "slice": None,
    # `dir(x)` -- the names it answers to, sorted. `dir()` with no argument
    # is the names IN SCOPE, which is `sorted(locals())`.
    "dir": None,
    # PEP 654: `ExceptionGroup(msg, [excs])`.
    "ExceptionGroup": 2, "BaseExceptionGroup": 2,
    "hasattr": 2, "getattr": None, "iter": None, "next": None,
    "dict": None, "bytes": None,
    # `bytearray()`, `bytearray(5)`, `bytearray(b"ab")`.
    # `memoryview` COUNTS ITS OWN ARGUMENTS now, along with every other type
    # constructor -- see `CTOR_PARAMS`. A count here refused
    # `memoryview(object=b"a")` as a diagnostic, which is a call CPython
    # answers, and put the refusal at compile time where CPython puts a
    # catchable TypeError.
    "bytearray": None, "memoryview": None,
    # A SNAPSHOT of the names in scope, built at the call site.
    "locals": 0, "globals": 0,
    "issubclass": 2, "vars": 1, "setattr": 3, "delattr": 2,
    "map": 2, "filter": 2, "format": None,
}

#: The keyword arguments each builtin accepts. A builtin not listed here takes
#: none, and naming one it does not have is reported where CPython reports it
#: -- at the call, by name, rather than as a wrong argument count.
_BUILTIN_KEYWORDS = {
    "sorted": ("key", "reverse"),
    "min": ("key", "default"),
    "max": ("key", "default"),
    "zip": ("strict",),
    "enumerate": ("start",),
    "print": ("sep", "end", "file", "flush"),
}

#: Builtin exception names. Calling one CONSTRUCTS an exception value, and
#: naming one in an `except` clause matches against the hierarchy in
#: `objects/csource.py`. Kept in the frontend as a plain set because the frontend
#: only has to recognise the name -- the runtime owns what it means.
#: Builtins that may be used as a VALUE, not only called. Each becomes a
#: one-argument function that calls it -- `sorted(xs, key=len)` needs `len` to
#: BE something, not merely to be callable.
#:
#: Exactly one argument. A variadic builtin has no single thunk shape, and a
#: two-argument one would need its arity carried alongside; neither is needed
#: by anything that passes a builtin as a value, which is nearly always a key
#: function.
#: NOT `type`. `type(x)` currently yields the type's NAME rather than a type
#: object, and the frontend hides that by special-casing `type(x).__name__`.
#: A thunk has no such special case, so `apply(type, 1).__name__` would fail
#: on a str -- refusing is better than being wrong in a new place, and the
#: entry goes back the moment `type` returns a real object.
_VALUE_BUILTINS = frozenset({
    "repr", "str", "len", "int", "float", "bool", "abs", "hash",
    "list", "tuple", "set", "frozenset", "sorted", "reversed", "sum", "min",
    "max", "ascii", "ord", "chr", "bin", "hex", "oct", "id", "complex",
    # These three take any number of arguments. Their thunk declares `*rest`
    # and hands the tuple to one runtime call, which is why they can be
    # values even though the one-argument thunk shape does not fit them.
    "print", "dict", "bytes",
    # `bytearray` IS A TYPE LIKE THE REST and was the only builtin type left
    # out. `isinstance(x, bytearray)` and `bytearray.__name__` already
    # worked, so the gap was narrow and total: `type(x) is bytearray`,
    # `map(bytearray, xs)` and `[bytearray][0](b"ab")` were COMPILE ERRORS
    # for a name Python hands out as an ordinary value.
    "bytearray",
    # `memoryview` IS A TYPE LIKE THE REST. It was the last builtin type
    # that could not be NAMED: `memoryview._from_flags(b, 0)`,
    # `memoryview[int]` and `type(m) is memoryview` were all
    # `'memoryview' is a builtin that cannot be used as a value` -- an
    # E0056 for a name CPython hands out as an ordinary object. The
    # constructor's signature was already declared (`CTOR_PARAMS`) and
    # `apy_builtin_ctor` already served it, so only the two sets that say
    # "this name has a value form" were missing it.
    "memoryview",
    # THE RUNTIME DESCRIPTORS, which are written `@property` far more often
    # than `property(f)`. A bare decorator is a NAME USED AS A VALUE, so
    # without these three here every `@staticmethod` was refused as a builtin
    # that cannot be one. Lowering recognises the decorator form and emits the
    # wrapping directly -- see `_dyn_decorated`.
    "property", "classmethod", "staticmethod",
})

#: Module attributes every Python file has without writing them, and what
#: they hold here. A compiled program IS the script being run, so `__name__`
#: is `"__main__"` -- which is what makes the `if __name__ == "__main__":`
#: guard at the bottom of a script take its branch, and that guard is in
#: enough real programs that not having it meant refusing them.
MODULE_DUNDERS = {
    "__name__": "__main__", "__doc__": None, "__package__": "",
    #: The SOURCE PATH, filled in by lowering -- the one dunder whose value is
    #: not a constant of every compilation. A program reads it to find files
    #: beside itself, and its absence was a NameError.
    "__file__": "",
}
_MODULE_DUNDERS = frozenset(MODULE_DUNDERS)
#: `__builtins__` is a name every module has and no module assigns, like
#: the dunders above -- but it holds an OBJECT rather than a constant, so
#: lowering builds it rather than reading a value from here.
_BUILTINS_NAME = "__builtins__"

#: Builtin type names, and the constructors reachable through them --
#: `dict.fromkeys`, `int.from_bytes`. Not unbound methods: there is no
#: receiver of that type to be the first argument.
_BUILTIN_TYPE_NAMES = frozenset({"dict", "int", "bytes", "str", "list",
                                 "tuple", "set", "frozenset", "float",
                                 # `bytearray.maketrans` is the same
                                 # staticmethod `bytes` has; the name is here
                                 # for that call shape and for no other.
                                 "bytearray"})
_TYPE_STATIC_NAMES = frozenset({"fromkeys", "from_bytes", "fromhex",
                                "maketrans"})

_EXC_NAMES = frozenset({
    # The WARNING categories are exceptions like any other: `Warning`
    # inherits `Exception`, and a program both raises them and asks
    # `issubclass(DeprecationWarning, Warning)`.
    "Warning", "UserWarning", "DeprecationWarning",
    "PendingDeprecationWarning", "SyntaxWarning", "RuntimeWarning",
    "FutureWarning", "ImportWarning", "UnicodeWarning",
    "BytesWarning", "ResourceWarning", "EncodingWarning",
    "BaseException", "Exception", "SystemExit", "KeyboardInterrupt",
    "GeneratorExit", "ArithmeticError", "ZeroDivisionError", "OverflowError",
    "FloatingPointError", "LookupError", "IndexError", "KeyError",
    "NameError", "UnboundLocalError", "AttributeError", "TypeError",
    "ValueError", "UnicodeError", "UnicodeDecodeError",
    "UnicodeEncodeError", "UnicodeTranslateError",
    "RuntimeError", "NotImplementedError",
    "RecursionError", "AssertionError", "ImportError", "ModuleNotFoundError",
    "OSError", "FileNotFoundError", "StopIteration", "StopAsyncIteration",
    # PEP 3151. `IOError` and `EnvironmentError` ARE `OSError` -- the same
    # object -- and the errno-specific ones are real classes under it.
    "IOError", "EnvironmentError", "PermissionError", "IsADirectoryError",
    "NotADirectoryError", "FileExistsError", "InterruptedError",
    "BlockingIOError", "ChildProcessError", "ProcessLookupError",
    "ConnectionError", "BrokenPipeError", "ConnectionAbortedError",
    "ConnectionRefusedError", "ConnectionResetError", "TimeoutError",
    "MemoryError", "EOFError",
    # WHAT `compile()` RAISES. `IndentationError` and `TabError` are
    # SUBCLASSES of SyntaxError, and a program catches each by name -- which
    # is the whole of what the `syntax/*` cases do.
    "SyntaxError", "IndentationError", "TabError",
    # `asyncio`. `CancelledError` inherits BaseException and not Exception,
    # since 3.8: `except Exception:` inside a task must not swallow the
    # cancellation the loop just delivered.
    "CancelledError", "InvalidStateError",
    # PEP 654. `BaseExceptionGroup` catches groups of BaseExceptions;
    # `ExceptionGroup` is the narrower one every ordinary program means.
    "ExceptionGroup", "BaseExceptionGroup",
})


def _import_names(node) -> list:
    """The names an `import` statement binds.

    `import a.b` binds `a`; `import a.b as c` binds `c`; `from m import x as y`
    binds `y`. Written out because the three forms bind different things and
    reading `alias.name` for all of them binds the wrong one twice.
    """
    out = []
    for alias in node.names:
        if isinstance(node, ast.ImportFrom):
            out.append(alias.asname or alias.name)
        else:
            out.append(alias.asname or alias.name.split(".")[0])
    return out


def _members_of(table):
    """Every member spec in a module table, however it is stored.

    A backend's table may be lazy -- the JVM backend builds a package's
    contents only when something looks inside -- so this asks for values
    rather than assuming a plain dict.
    """
    try:
        return list(table.values()) if hasattr(table, "values")             else [table[k] for k in table]
    except Exception:                          # noqa: BLE001 -- best effort
        return []


def _has_yield(node) -> bool:
    """Does this `def`'s OWN body contain a `yield`?

    Not a nested one's: a `def` inside a generator is an ordinary function
    unless it yields itself, and walking into it would make every enclosing
    function a generator. A lambda cannot contain `yield` at all, so it is
    skipped for free by the same rule.
    """
    return any(_owns_yield(stmt) for stmt in node.body)


def _owns_yield(node) -> bool:
    """`yield` anywhere under `node`, not entering a nested function."""
    if isinstance(node, (ast.Yield, ast.YieldFrom)):
        return True
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return False
    return any(_owns_yield(child) for child in ast.iter_child_nodes(node))


def _pattern_names(pat) -> list:
    """Every name a `case` pattern BINDS, in the order it binds them.

    A pattern is a mix of tests and bindings and the two are not separable by
    node type: `case Point(x, y)` tests the class and binds two names, while
    `case Point(0, y)` tests one of them instead. What decides is POSITION --
    a bare `Name` in a pattern is a capture, the same `Name` under
    `MatchValue` is a value to compare against.
    """
    if pat is None:
        return []
    if isinstance(pat, ast.MatchAs):
        # `case [x] as whole` binds `whole` AND whatever the inner pattern
        # binds. A bare `case name` is a MatchAs with no pattern, and `case _`
        # is one with no name -- the wildcard, which binds nothing.
        inner = _pattern_names(pat.pattern)
        return inner + ([pat.name] if pat.name else [])
    if isinstance(pat, ast.MatchStar):
        return [pat.name] if pat.name else []
    if isinstance(pat, ast.MatchOr):
        # Every alternative must bind the SAME names -- Python requires it --
        # so the first is representative and duplicates would only be noise.
        return _pattern_names(pat.patterns[0]) if pat.patterns else []
    if isinstance(pat, ast.MatchSequence):
        return [n for sub in pat.patterns for n in _pattern_names(sub)]
    if isinstance(pat, ast.MatchMapping):
        out = [n for sub in pat.patterns for n in _pattern_names(sub)]
        return out + ([pat.rest] if pat.rest else [])
    if isinstance(pat, ast.MatchClass):
        out = [n for sub in pat.patterns for n in _pattern_names(sub)]
        return out + [n for sub in pat.kwd_patterns
                      for n in _pattern_names(sub)]
    # MatchValue and MatchSingleton test; they bind nothing.
    return []


def _target_names(node) -> list:
    """Every plain name in an assignment or loop target.

    Nested targets (`for a, (b, c) in ...`) flatten here and are rejected by
    lowering, which knows the arity it is unpacking into; a name list is all
    analysis needs to declare them.
    """
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Starred):
        # `a, *rest = xs`. The star is punctuation on the target, not a target
        # of its own: `rest` is an ordinary name that happens to be bound to a
        # list. Missing it here left `rest` undeclared and reported the
        # program's own variable as an undefined name.
        return _target_names(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        out = []
        for e in node.elts:
            out.extend(_target_names(e))
        return out
    return []


def _declared_global(body: list) -> set:
    """Names a function body declares `global`.

    Walks the whole body including nested statements, because `global` is
    function-wide wherever it appears -- Python even allows it after the first
    use, which is a wart but a real one.
    """
    found: set = set()
    for node in body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Global):
                found.update(sub.names)
    return found


def _stored_names(node) -> set:
    """Every name a statement BINDS, however deeply.

    A class body's general statements are the ones that bind through control
    flow, so there is no one target to read: `try: x = f() / except E: x = 0`
    binds `x` twice in two branches, and an `import json` binds `json`. The
    walk is over the whole statement because that is where the bindings are.
    """
    out = set()
    for one in ast.walk(node):
        if isinstance(one, ast.Name) and isinstance(one.ctx, ast.Store):
            out.add(one.id)
        elif isinstance(one, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)):
            out.add(one.name)
        elif isinstance(one, ast.alias):
            out.add(one.asname or one.name.split(".")[0])
        elif isinstance(one, ast.ExceptHandler) and one.name:
            out.add(one.name)
    return out


def _one_handler_name(node) -> str:
    """The exception name ONE `except` clause entry gives.

    A DOTTED NAME ANSWERS ITS LAST PART. `except asyncio.CancelledError:`
    means the exception called CancelledError -- the hierarchy here is a table
    of names, so the module qualifying it adds nothing to match on and reading
    the attribute as an empty name refused the clause outright. A name the
    table does not know is still refused, one line further on, so this cannot
    turn an unknown type into a silent catch-all.
    """
    if isinstance(node, ast.Attribute):
        return node.attr
    return getattr(node, "id", "")


def _handler_names(node) -> list:
    """The exception names an `except` clause lists. `except (A, B):` is a
    tuple; `except A:` is one name."""
    if isinstance(node, ast.Tuple):
        return [_one_handler_name(e) for e in node.elts]
    return [_one_handler_name(node)]


#: The literal types the object runtime has a kind for. `Ellipsis` is not one
#: and neither is `bytes` or `complex`; a literal of any other type is refused
#: rather than passed to a lowering that has no case for it.
_CONSTANT_KINDS = frozenset({bool, int, float, complex, str, bytes,
                             type(None)})


def _is_docstring(node) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
        and isinstance(node.value.value, str)

#: How each maps into the IR. `bool` becomes i1 so a comparison result and a
#: `bool` variable are the same thing at the machine level.
#: The prefix a Java handle type carries. A value of one of these types is an
#: integer standing for an entry in a table the generated class keeps -- see
#: `backends/jvm/interop.py`. The class is in the NAME so that a diagnostic can
#: say `com.minecraft.block.Block` rather than "object", and so that a method
#: call can be resolved against the right class.
JAVA = "jvm:"


def sem_type(name: str) -> SemType:
    """The SemType for a type NAME, interned.

    `SemType` is compared with `is` in places -- `print` picks its writer that
    way -- so a fresh `SemType("float")` is not `FLOAT` and a float printed as
    an integer. Java signatures arrive as names, so they come through here.
    """
    return BY_NAME.get(name) or SemType(name)


def is_java(ty) -> bool:
    """Whether a type is a handle. A `SemType` OR ITS NAME.

    Both, because a Java signature arrives as names and an expression's type
    arrives as a `SemType`, and the assignability rules compare one against the
    other -- `_java_fits(got, want)` is exactly that pair. Insisting on one kind
    meant a subclass reaching a superclass parameter crashed on `.name`.
    """
    return (ty if isinstance(ty, str) else ty.name).startswith(JAVA)


def java_class_of(ty) -> str:
    """The internal class name behind a handle type."""
    return (ty if isinstance(ty, str) else ty.name)[len(JAVA):]


class _IrTypes(dict):
    """SemType -> IR type, with every Java handle answering `i64`.

    A dict with a fallback rather than a function, because every existing use
    site indexes it and a handle can reach any of them -- an argument, a
    return, a local's declared width.
    """

    def __missing__(self, key):
        if is_java(key):
            return T.I64
        raise KeyError(key)


TO_IR = _IrTypes({INT: T.I64, FLOAT: T.F64, BOOL: T.I1, NONE: T.VOID,
                  OBJ: T.PTR,
                  # The machine types are their IR counterparts by NAME, which
                  # is not laziness: it is the property that keeps the two
                  # tables from ever disagreeing about a width.
                  **{sem: T.ALL[name] for name, sem in MACHINE.items()}})


#: Where a name's value actually lives, which is not the same question as what
#: it is called. Three answers, and the difference between them is the whole
#: of closure support:
#:
#:   LOCAL  a register. What every name was before nested `def` existed.
#:   CELL   a register holding a runtime CELL, because some inner function
#:          captures this name. Reads and writes go through the box.
#:   FREE   a cell that arrived through the closure environment.
#:
#: A captured variable has to be a box rather than a copy because
#: `functions/closure-cell-is-shared` requires two closures over one name to
#: see each other's writes. Copying the value at capture time passes every
#: case with a single closure and fails that one, which is the worst possible
#: split: the feature looks finished.
LOCAL, CELL, FREE = "local", "cell", "free"


@dataclass(slots=True)
class Symbol:
    name: str
    type: SemType
    span: Span
    #: Set once lowering allocates a register for it.
    register: int | None = None
    is_param: bool = False
    used: bool = False
    storage: str = LOCAL
    #: Position in the owning function's cell list (CELL) or in the
    #: environment a closure was built with (FREE). The two orders must agree:
    #: the maker writes cell `i` and the callee reads env slot `i`.
    index: int = -1


@dataclass(slots=True)
class FunctionInfo:
    """Everything lowering needs about one function, after analysis."""

    #: The `def` this came from -- or, for the module entry, the `ast.Module`
    #: holding the top-level statements. Both answer to `.body`, which is all
    #: analysis and lowering ever ask of it.
    node: ast.FunctionDef | ast.Module
    name: str
    params: list[Symbol]
    ret: SemType
    locals: dict[str, Symbol] = field(default_factory=dict)
    #: Expression node id -> its type. Keyed by id() because ast nodes are not
    #: hashable in a way that survives equality, and analysis and lowering walk
    #: the same tree objects.
    expr_types: dict[int, SemType] = field(default_factory=dict)
    #: Call node id -> the Java overload it resolved to, as the backend
    #: published it. Lowering reads this rather than resolving again: two
    #: resolutions of an overload set are two chances to pick differently.
    java_calls: dict[int, dict] = field(default_factory=dict)
    #: Call node id -> the native function it names, as
    #: `{symbol, params, ret, library}`. Recorded by analysis and read by
    #: lowering for the same reason `java_calls` is: resolving twice is two
    #: chances to resolve differently.
    ctypes_calls: dict[int, dict] = field(default_factory=dict)
    #: Constant node id -> the string literal that has to become a Java
    #: `String`. Only literals in a Java argument position are here; the
    #: subset has no string type for them to have anywhere else.
    java_strings: dict[int, str] = field(default_factory=dict)
    #: Call node id -> the key of the method of a Python class over a Java
    #: type. `b.tick()` where `tick` is written in THIS source is a direct call
    #: to an ordinary static function, not an `invokevirtual` -- the class path
    #: has never heard of the method, and going through the JVM to reach a
    #: function in the same class file would be slower and no more correct.
    sub_calls: dict[int, str] = field(default_factory=dict)
    #: The IR symbol this function must be emitted under, when the backend
    #: chose it rather than lowering. Only a method of a class over a Java
    #: type has one: the symbol encodes the generated class, its base, the
    #: method and its descriptor, and the backend rebuilds the class file from
    #: exactly that -- so a name invented in lowering would be a second
    #: spelling of something with one authority.
    java_symbol: str = ""
    #: True when this function's values are runtime objects rather than
    #: machine words. Set for the module's top-level statements, which have no
    #: annotations at all, and for any function with an unannotated parameter
    #: -- ordinary Python, in other words. A fully annotated function keeps the
    #: static path, which is what every already-written program uses.
    dynamic: bool = False
    #: Names this function reads from MODULE scope rather than from its own
    #: frame. Python's rule: a name assigned anywhere in a function body is
    #: local for the whole body, and every other name falls through to the
    #: module. Recorded per function because the answer depends on the body,
    #: not on the name.
    #: Whether this function calls `locals()`, which is a READ OF EVERY
    #: LOCAL and the only expression that is. It matters because a local no
    #: ordinary read could reach unassigned still has to be absent from the
    #: mapping rather than present as a null -- so `locals()` is what decides
    #: a register needs its null initialiser, not the read analysis.
    reads_all_locals: bool = False
    module_reads: set = field(default_factory=set)
    #: Names this function ASSIGNS at module scope -- those it declared
    #: `global`. Without the declaration an assignment makes a local, even if a
    #: module-level name of the same spelling exists.
    module_writes: set = field(default_factory=set)
    #: Default-value expressions, for the LAST `len(defaults)` parameters.
    #: Evaluated ONCE, where the `def` runs, not at each call -- which is
    #: observable: `def f(xs=[])` shares one list across calls, and the suite
    #: checks exactly that.
    defaults: list = field(default_factory=list)
    #: The `*rest` parameter's name, or None. It is not in `params`: it takes
    #: no argument position and the arity check must not count it.
    vararg: str | None = None
    #: True when the body contains a `yield`. Such a function does not run
    #: when called -- it builds a generator -- so it is lowered as TWO IR
    #: functions and every local lives in the generator object. See
    #: `_dyn_generator`.
    is_generator: bool = False
    #: True for an `async def`. A COROUTINE IS A GENERATOR here -- it needs the
    #: same frame, the same step function and the same "none of the body runs
    #: until something drives it" rule -- so `is_generator` is set alongside
    #: this one and the lowering is shared. What this flag decides is what the
    #: object CALLS itself: `type(f()).__name__` is 'coroutine', and a program
    #: that awaits a generator or iterates a coroutine is making an error the
    #: name is how it finds out about.
    is_coroutine: bool = False
    #: True for an `async def` that also contains `yield`. NEITHER a plain
    #: generator NOR a plain coroutine: it is driven by `async for`, its
    #: `yield` produces values and its `await` suspends, and CPython names it
    #: `async_generator`. Told apart from both because a program reads that
    #: name, and because the two things it does travel the same channel.
    is_async_generator: bool = False
    #: Local name -> its slot in the generator's frame. Empty for an ordinary
    #: function, whose locals are registers.
    slots: dict = field(default_factory=dict)
    #: Locals whose assignment could not be PROVED to happen before every
    #: read. On the dynamic path they are checked at run time and raise
    #: `UnboundLocalError`, exactly as CPython does; the static path rejects
    #: them outright, having no representation for "unset".
    maybe_unbound: set = field(default_factory=set)
    #: How many LEADING parameters are positional-only -- the ones before a
    #: `/` in the signature. They take an argument position like any other and
    #: differ in one way: a keyword cannot reach them, so their names are not
    #: recorded on the function value and `f(a=1)` against `def f(a, /)` is an
    #: unexpected keyword rather than a second value for `a`.
    posonly: int = 0
    #: How many TRAILING parameters are keyword-only -- the ones after a `*`.
    #: The mirror image: a position cannot reach them, so positional filling
    #: stops short of them and only a name arrives.
    kwonly: int = 0
    #: How many of the TRAILING DEFAULTS are the keyword-only ones'.
    #: Not `kwonly`: one of those may be required, and `def f(a, b=1,
    #: *args, c)` has one keyword-only parameter and one default that
    #: is not its.
    kwdefaults: int = 0
    #: The `**kw` parameter's name, or None. Like `vararg` it is not in
    #: `params`: it takes no argument position, and it is bound to a dict of
    #: whatever keywords the call could not place -- empty when there were
    #: none, because `def f(**kw)` called as `f()` binds `{}` and not nothing.
    kwarg: str | None = None
    #: AugAssign node id -> the equivalent BinOp, built once by analysis and
    #: reused by lowering. `x += 1` is `x = x + 1` and both passes need that
    #: tree; building it twice meant only one of them checked it, and
    #: `x **= n` type-checked as ordinary arithmetic before hitting a
    #: lowering table that requires a literal exponent.
    aug_nodes: dict[int, ast.BinOp] = field(default_factory=dict)

    #: The key of the function this one is written inside, or None at module
    #: level. A METHOD's parent is the function containing its `class`, not the
    #: class -- a class body is not in the lookup chain, so a method cannot see
    #: names the class body bound. Getting that wrong makes a method resolve a
    #: sibling method's name as a free variable, which then captures nothing.
    parent: str | None = None
    #: Dotted name for diagnostics: `outer.<locals>.inner`, `Point.move`.
    qualname: str = ""
    #: Locals some inner function captures, in CELL ORDER. Each lives in a
    #: runtime cell instead of a plain register.
    cellvars: list = field(default_factory=list)
    #: Names captured FROM an enclosing function, in ENV ORDER. The maker
    #: writes cell `i` and this function reads env slot `i`; the two orders are
    #: one agreement and both sides read this list.
    freevars: list = field(default_factory=list)
    #: The class this is a method of, or None. What `super()` needs: the class
    #: the method was DEFINED in, which is not `type(self)`.
    owner: str | None = None
    #: True for a function that is a VALUE bound where its `def` runs -- every
    #: nested `def` and every method. A module-level `def` is one too, but it
    #: also keeps its direct-call path.
    is_value: bool = False

    @property
    def signature(self) -> tuple[list[SemType], SemType]:
        return [p.type for p in self.params], self.ret

    @property
    def takes_env(self) -> bool:
        """Whether the compiled function's first parameter is the closure env.

        EVERY dynamic function has one, including those that capture nothing,
        and the module entry has none. Uniformity rather than a per-function
        flag: `apy_call` reaches a function it cannot see the definition of, so
        a convention that varied would have to be recorded in the value and
        checked on every call. An unused first parameter costs one register.
        """
        return self.dynamic and self.name != ENTRY_NAME


@dataclass(slots=True)
class ClassInfo:
    """One `class` statement.

    Not a scope in the closure sense -- see `FunctionInfo.parent`. A class body
    is executed where it is written and its bindings go into the type object,
    so what analysis needs from it is an ordered list of what to bind.
    """

    node: ast.ClassDef
    name: str
    #: PEP 3155's `__qualname__`, as the SOURCE spells it: `mk.<locals>.D`
    #: for a class written inside a function, `C.Inner` for one written
    #: inside another class, and the bare name at module level. Not the key
    #: -- see `_register_class` for the three ways the two differ.
    qualname: str
    #: The FIRST base's name as written, or None. Everything that asks a
    #: single question -- `__base__`, a walk up one chain -- reads this.
    base: str | None = None
    #: Every base, in the order written. The MRO is linearised from these at
    #: run time, because whether a base itself has bases is a property of a
    #: value rather than of this statement.
    bases: list = field(default_factory=list)
    #: PEP 560: bases written as EXPRESSIONS rather than names. What each one
    #: contributes is decided by its `__mro_entries__` at run time, so there
    #: is nothing to record here but the expression.
    base_exprs: list = field(default_factory=list)
    #: `class D(dict)` -- the BUILTIN KIND this class extends, by name, or
    #: None. Not a class, so it cannot go in `bases`.
    builtin_base: str | None = None
    #: Keys into `Analyzer.functions`, in definition order.
    methods: list = field(default_factory=list)
    #: (name, value expression) per class-level assignment, in order.
    attrs: list = field(default_factory=list)
    #: Bare expression statements in the body, in order. A class body RUNS,
    #: and a statement that binds nothing still has its effect.
    body_exprs: list = field(default_factory=list)
    #: (name, annotation expression) per annotated class-level name, in order
    #: -- including one with no value. PEP 649 builds `C.__annotations__` from
    #: these, lazily, exactly as a function's are built.
    annotations: list = field(default_factory=list)
    #: The key of the function whose body this `class` statement runs in.
    scope: str = ENTRY_NAME
    #: True for `class Meta(type):` -- this class is a METACLASS, so
    #: calling it builds a class rather than an instance and its `__new__`
    #: reaches `type.__new__` through `super()`.
    is_meta: bool = False
    #: The metaclass NAME from `class C(metaclass=Meta)`, or None.
    metaclass: str | None = None
    #: `class C(metaclass=Meta, flavour="x")` -- the keywords that are not
    #: `metaclass`, in order. They reach `__new__`, `__init__` and
    #: `__init_subclass__` as keyword arguments.
    class_keywords: list = field(default_factory=list)
    #: True for `class MyError(ValueError):` -- a name in the exception
    #: hierarchy rather than a type object. See `_class_statement`.
    is_exception: bool = False
    #: STATEMENTS IN THE BODY THAT ARE NOT MEMBERS -- an `if`, a `try`, a
    #: `for`. A class body is a block that runs, and these are the parts of it
    #: that bind through control flow rather than in one line.
    body_stmts: list = field(default_factory=list)
    #: What those statements BIND. Recorded because the lowering has to route
    #: their stores into the class namespace: a plain local would make the
    #: name vanish when the body ended, and the class attribute the source
    #: plainly writes would never exist.
    body_stmt_names: set = field(default_factory=set)


def int_literal(node) -> int | None:
    """The value of `node` if it is an integer literal, else None.

    `-1` is not `Constant(-1)` -- Python parses it as `UnaryOp(USub,
    Constant(1))`, and code that only checks for Constant silently treats a
    negative literal as "not a literal". That mistake made `range(5, 0, -1)`
    compile to an ascending loop that ran zero times.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant) \
            and isinstance(node.operand.value, int) \
            and not isinstance(node.operand.value, bool):
        if isinstance(node.op, ast.USub):
            return -node.operand.value
        if isinstance(node.op, ast.UAdd):
            return node.operand.value
    return None


def _string_literal(node) -> str | None:
    """The text of `node` if it is a plain string literal, else None.

    The static path has no string TYPE -- a str is an object and the static
    path allocates nothing -- so this is not "evaluate an expression", it is
    "read a name the compiler was given". `reserve` is the only user.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def const_int(node, shadowed=frozenset()) -> int | None:
    """An integer the COMPILER knows: a literal, `sizeof(T)`, or arithmetic
    between two of those. None if it is not one.

    `sizeof(i64)` is a literal that happens to be spelled as a call -- it
    folds to 8 and emits a constant -- and treating it as one is what lets a
    struct layout be written the way it should be read:

        p: ptr = alloca(sizeof(i64) + sizeof(ptr) + sizeof(i64) * 4)

    That position needs a number BEFORE the program runs, because `alloca`
    carries its size in an immediate. Without folding it would have to be
    written `48` with a comment explaining where 48 came from -- which is how
    a layout and the code that reads it stop agreeing.

    NOT A CONSTANT EVALUATOR. A NAME does not fold, however constant it looks:
    resolving one means knowing the scope it is in, and this is a function
    over a node. `sizeof(...)` and arithmetic on it is the whole of it.

    `shadowed` is the names the program defined itself. A program with its own
    `def sizeof(...)` keeps it, exactly as it keeps its own `load`.
    """
    value = int_literal(node)
    if value is not None:
        return value
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "sizeof" and node.func.id not in shadowed
            and len(node.args) == 1
            and getattr(node.args[0], "id", None) in MACHINE):
        return T.ALL[node.args[0].id].size
    if isinstance(node, ast.BinOp) and type(node.op) in _CONST_FOLD:
        left = const_int(node.left, shadowed)
        right = const_int(node.right, shadowed)
        if left is not None and right is not None:
            return _CONST_FOLD[type(node.op)](left, right)
    return None


#: What `const_int` folds. Deliberately not division: `//` by zero would have
#: to be reported from a helper that has no diagnostic sink, and a layout that
#: needs division is one that should name its parts.
_CONST_FOLD = {ast.Add: lambda a, b: a + b,
               ast.Sub: lambda a, b: a - b,
               ast.Mult: lambda a, b: a * b}


def _written_qual(qualname: str) -> str:
    """The qualname as the SOURCE spells it: `Fraction.limit_denominator`.

    The first component of a nested qualname is the enclosing definition's
    key, and a spliced class's key carries the mangling -- so a method of a
    bundled class answered `__qualname__` as
    `_asmpy_bundled_9_fractions_Fraction.limit_denominator`, the mangling in
    plain sight. The key itself stays mangled: it is what keeps a spliced
    class from colliding with a program's own class of the same name. Only
    the spelling a program can READ is restored.
    """
    # Deferred: `bundled` imports `span_of` from this module.
    from .bundled import bare_of
    head, dot, tail = qualname.partition(".")
    real = bare_of(head)
    return qualname if real is None else real + dot + tail


def span_of(source: SourceFile, node) -> Span:
    """The source range an AST node occupies.

    Shared by analysis and lowering so that a diagnostic and the instruction it
    describes point at exactly the same bytes -- if they computed it
    separately they would drift, and an optimiser's message would land one
    column off the error the type checker reported for the same expression.
    """
    lineno = getattr(node, "lineno", None)
    if lineno is None:
        return source.span(0, 0)
    starts = source.line_starts
    begin = starts[lineno - 1] + getattr(node, "col_offset", 0)
    end_line = getattr(node, "end_lineno", lineno)
    end_col = getattr(node, "end_col_offset", None)
    if end_col is None or end_line > len(starts):
        return source.span(begin, begin + 1)
    return source.span(begin, starts[end_line - 1] + end_col)


# ── scopes ──────────────────────────────────────────────────────────────────
# Everything down to `Analyzer` answers ONE question: for each name a function
# mentions, where does its storage live. Before nested `def` there was nothing
# to compute -- every name was a register in the current frame, or a module
# global. A closure makes it three answers and makes them depend on the whole
# nesting, so it becomes a pass of its own that runs before any body is
# checked. It has to run first because a closure may capture a name assigned
# LATER in the enclosing function (`functions/inner-function-sees-later-
# binding`), so the decision cannot be made as the body is walked.


class _Scope:
    """One binding region: the module, a function body, or a class body."""

    __slots__ = ("key", "kind", "parent", "bound", "globals", "nonlocals",
                 "reads", "cells", "frees", "node", "lambdas")

    def __init__(self, key: str, kind: str, parent, node=None) -> None:
        self.key = key
        self.kind = kind                # "module" | "function" | "class"
        self.parent = parent
        self.node = node
        self.bound: set = set()
        self.globals: set = set()
        self.nonlocals: set = set()
        self.reads: set = set()
        #: Resolved by `_resolve_closures`. ORDERED, because the function that
        #: builds a closure writes cell `i` and the closure reads env slot `i`;
        #: the two sides index into these lists and nothing else keeps them in
        #: step.
        self.cells: list = []
        self.frees: list = []
        #: Lambdas written in this scope's expressions, awaiting a scope of
        #: their own. Recorded rather than registered on the spot because
        #: `_expr_names` is a free function with no analyzer to register with,
        #: and it is called from a dozen places -- putting the hook in one of
        #: them and not the others is how a lambda in a `while` test would
        #: silently get no scope.
        self.lambdas: list = []

    @property
    def locals(self) -> set:
        """Names whose storage is this scope's own frame."""
        return self.bound - self.globals - self.nonlocals

    def enclosing_function(self):
        """The nearest enclosing FUNCTION scope, skipping class bodies.

        A class body is NOT in the lookup chain. A method that reads `n` does
        not see `n = 1` from its own class body -- it sees the enclosing
        function's `n`, or the module's. Walking the parent chain without this
        skip makes a method capture a class attribute CPython never gives it,
        and the symptom is a method that reads a stale value rather than one
        that fails.
        """
        s = self.parent
        while s is not None and s.kind == "class":
            s = s.parent
        return s


def parameter_names(args: ast.arguments) -> list[str]:
    """Every name a parameter list BINDS.

    ALL FIVE GROUPS, which is the point of writing it once. The closure scopes
    bound `args.args` and the two star-parameters and nothing else, which is
    right for every function whose parameters are ordinary and silently wrong
    for one written `def f(arg, /)`: a positional-only parameter was not bound
    in its own scope, so a nested function reading it could not see it. The
    diagnostic that came out was `E0052: call to unknown function 'arg'` --
    about a parameter three lines above, listing every function in the program
    except the one meant. `warnings.deprecated` is written that way, and CPython
    writes it that way deliberately, so the gap had to be closed before the
    module could exist.
    """
    names = [a.arg for a in
             list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)]
    for star in (args.vararg, args.kwarg):
        if star is not None:
            names.append(star.arg)
    return names


def lambda_def(node: ast.Lambda) -> ast.FunctionDef:
    """The `def` a lambda is, built once and cached on the node.

    Cached because analysis and lowering must see the SAME synthetic node:
    scope registration keys off its identity, and building a second one would
    leave lowering looking for a function nobody registered.
    """
    made = getattr(node, "_uasm_def", None)
    if made is not None:
        return made
    made = ast.FunctionDef(
        name="<lambda>", args=node.args,
        body=[ast.Return(value=node.body)], decorator_list=[], returns=None,
        type_params=[])
    ast.copy_location(made, node)
    ast.copy_location(made.body[0], node)
    node._uasm_def = made
    return made


def genexp_def(node: ast.GeneratorExp) -> ast.FunctionDef:
    """The generator `def` a generator expression IS.

    `(f(x) for x in xs if p(x))` is exactly

        def <genexpr>(.0):
            for x in .0:
                if p(x):
                    yield f(x)

    called with `xs` already evaluated. Everything the expression form is
    supposed to do falls out of that shape rather than being arranged
    separately:

      * it is LAZY, because the body is a generator body;
      * its target does not leak, because the body is a function;
      * and the OUTERMOST iterable is evaluated eagerly, because it is an
        argument -- which is why `(x for x in boom())` raises at the
        expression and `(x for x in xs if boom())` does not.

    Cached on the node for the reason `lambda_def` is: analysis and lowering
    must see the same synthetic node, since scope registration keys off its
    identity.
    """
    made = getattr(node, "_uasm_def", None)
    if made is not None:
        return made

    body: list = [ast.Expr(value=ast.Yield(value=node.elt))]
    for i, gen in enumerate(reversed(node.generators)):
        for test in reversed(gen.ifs):
            body = [ast.If(test=test, body=body, orelse=[])]
        # The FIRST generator's iterable becomes the parameter; the rest are
        # evaluated inside, which is what makes only the outermost eager.
        source = (ast.Name(id=".0", ctx=ast.Load())
                  if i == len(node.generators) - 1 else gen.iter)
        body = [ast.For(target=gen.target, iter=source, body=body,
                        orelse=[], type_comment=None)]

    made = ast.FunctionDef(
        name="<genexpr>",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=".0")],
                           vararg=None, kwonlyargs=[], kw_defaults=[],
                           kwarg=None, defaults=[]),
        body=body, decorator_list=[], returns=None, type_params=[])
    for sub in ast.walk(made):
        if not hasattr(sub, "lineno"):
            ast.copy_location(sub, node)
    ast.fix_missing_locations(made)
    node._uasm_def = made
    return made


def _lambdas_in(node):
    """Every lambda or GENERATOR EXPRESSION in an expression, outermost first,
    without descending into one already nested inside another -- that inner
    one belongs to the outer one's scope and is collected with its body.

    Both are nested functions written inline, and both need a scope of their
    own, so one walk finds them and one registration handles them.
    """
    found = []

    def walk(sub):
        if sub is None:
            return
        for child in ast.iter_child_nodes(sub):
            if isinstance(child, (ast.Lambda, ast.GeneratorExp)):
                found.append(child)
            else:
                walk(child)

    if isinstance(node, (ast.Lambda, ast.GeneratorExp)):
        return [node]
    walk(node)
    return found


def _inside_lambda(target, root) -> bool:
    """Whether `target` sits inside a lambda somewhere under `root`."""
    for lam in _lambdas_in(root):
        if target is lam:
            return True
        for sub in ast.walk(lam):
            if sub is target:
                return True
    return False


def _expr_names(node, scope: _Scope) -> None:
    """Record what an expression reads and what it binds.

    A Store-context name inside an EXPRESSION is a comprehension target or a
    walrus. Comprehensions do not get a scope of their own here -- a stated
    divergence, `[i for i in xs]` leaves `i` behind -- so their targets bind in
    the enclosing scope, which is exactly what recording them here does.
    """
    if node is None:
        return
    for lam in _lambdas_in(node):
        scope.lambdas.append(lam)
    for sub in ast.walk(node):
        if _inside_lambda(sub, node):
            # A lambda's PARAMETERS and body are its own, not the enclosing
            # scope's. Walking into one bound them here, so `lambda x: x` made
            # `x` a local of whatever contained it.
            continue
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                and sub.func.id == "super" and not sub.func.id in scope.bound \
                and not sub.args:
            # `super()` READS THE CLASS IT WAS WRITTEN IN, and nothing in the
            # source says so. The no-argument form is sugar for
            # `super(TheClass, self)`, and the frontend supplies the first
            # half from the method's owner -- but it supplies it as a NAME to
            # be loaded, and a name nobody recorded a read of gets no cell.
            #
            # For a class at module level that was invisible: the load fell
            # through to a global read and the global was there. For a class
            # NESTED IN A FUNCTION the class is a local of that function, so
            # the load emitted `global_addr` for a global that is never
            # created and the compile failed with `unknown global`. Recording
            # the read here is what makes the enclosing function hand the
            # method a cell, which is the same mechanism CPython's `__class__`
            # closure is.
            #
            # HARMLESS AT MODULE LEVEL, where the closure pass finds the name
            # bound in no enclosing FUNCTION and leaves it a global read --
            # which is what it already was.
            owner = scope.parent
            while owner is not None and owner.kind != "class":
                owner = owner.parent
            if owner is not None and owner.node is not None:
                scope.reads.add(owner.node.name)
        if isinstance(sub, ast.Name):
            if isinstance(sub.ctx, ast.Load):
                scope.reads.add(sub.id)
            else:
                scope.bound.add(sub.id)


class Analyzer:
    """Resolves names and assigns a type to every expression."""

    def __init__(self, source: SourceFile, sink: DiagnosticSink, *,
                 library: bool = False) -> None:
        self.source = source
        self.sink = sink
        #: Compiling a RUNTIME MODULE rather than a program: definitions only,
        #: no entry point, and every function exported because the whole point
        #: of it is to be called from somewhere else.
        self.library = library
        self.functions: dict[str, FunctionInfo] = {}
        #: Bound name -> the module it names. `import a.b as n` puts `n` here;
        #: `import a.b` puts `a` here mapped to "", meaning "a package root,
        #: the module is further along the attribute chain".
        self.namespaces: dict[str, str] = {}
        #: What `import ctypes` and `from ctypes import ...` bound, as
        #: {local name: "ctypes" | "ctypes.type" | "ctypes.loader"}.
        self.ctypes_names: dict[str, str] = {}
        #: Local name -> the library `CDLL(...)` named. See `cffi.py`.
        self.ctypes_libs: dict[str, str] = {}
        #: (library-local, function) -> {"params": [...], "ret": str}, built
        #: from the `argtypes` and `restype` assignments a program writes.
        self.ctypes_sigs: dict = {}
        #: Every library named, so the driver can hand them to the linker.
        self.ctypes_libraries: list = []
        #: Which platform a DECLARED native library is being chosen for, so a
        #: cross-platform program declaring `user32.dll` and `libX11.so.6`
        #: under one module name gets the one that applies. None means every
        #: unscoped declaration applies and no scoped one does.
        self.native_os: str | None = nativelib.target_os()
        #: Statement ids that WERE a ctypes declaration. Lowering skips them:
        #: `libm = ctypes.CDLL("m")` describes the build rather than doing
        #: anything, and lowering it as ordinary code looks for a run-time
        #: `ctypes` that was never going to exist.
        self.ctypes_stmts: set = set()
        #: A splice's `__name__` restoration on a STATICALLY typed `def`.
        #: Same shape and same reason as `ctypes_stmts`: the statement
        #: describes the compile rather than doing anything, and lowering
        #: it looks for a function object that was never going to exist.
        #: See `_is_splice_dunder`.
        self.splice_dunder_stmts: set = set()
        #: Python class name -> the backend's table for the type it declares.
        #: `class MyBlock(block.Block)` is not a class in the runtime-object
        #: sense at all: it is a TYPE the backend generates, its instances are
        #: handles like any other Java object's, and its methods are ordinary
        #: static functions. Nothing here reaches `self.classes`.
        self.java_subclasses: dict[str, dict] = {}
        #: Python class name -> {method name: its key in `functions`}.
        self.java_subclass_methods: dict[str, dict] = {}
        #: The keys of those methods, in source order, for the body pass.
        self._java_method_keys: list[str] = []
        self.current: FunctionInfo | None = None
        #: `break`/`continue` are only meaningful inside a loop, and Python
        #: makes that a syntax error. ast.parse does NOT -- it happily parses
        #: a bare `break` in a function body -- so it is checked here.
        self.loop_depth = 0
        #: Every name the module's top level binds. Set by `run`.
        self.module_names: set = set()
        #: One `ClassInfo` per `class` statement, keyed the way `functions` is.
        #: Named static regions this module asked for, by size. See `_reserve`.
        #: Lowering turns each into one IR global; the same name twice is one
        #: region, which is how two functions share a cache.
        self.reserved: dict[str, int] = {}
        self.classes: dict = {}
        #: User exception class name -> its base's name. Separate from
        #: `classes` because these are NOT type objects: `raise MyError(x)`
        #: builds an exception cell, exactly as `raise ValueError(x)` does.
        self.exc_classes: dict = {}
        #: The scope tree, flat, plus an index from a function's key to its
        #: scope. Built by `_collect_scopes` before any body is analysed.
        self.scopes: list = []
        self.scope_of: dict = {}
        #: ClassDef node id -> its key in `classes`. Keyed by id() for the same
        #: reason `expr_types` is: analysis and lowering walk the same objects.
        self.class_of_node: dict = {}
        #: FunctionDef node id -> its key. A nested `def` is a STATEMENT that
        #: builds a value, so lowering needs to get from the statement back to
        #: the function analysis registered for it.
        self.def_of_node: dict = {}
        #: Module-level names more than one `def` binds. A call to one of them
        #: means whichever has RUN, so it cannot be resolved at the call site.
        self.rebound: set = set()
        #: Calls whose argument count is provably wrong. Python reports these
        #: at run time and a program may catch them, so they are lowered as
        #: value calls and left to the runtime.
        self.late_arity: set = set()
        #: Keys of every `def` below module level, in registration order --
        #: which is source order, so a method is checked after the class that
        #: owns it exists.
        self._nested_keys: list = []
        #: The names local to the function being checked, or None outside one
        #: and for the module entry (whose names ARE the globals).
        self._function_locals: set | None = None

    # ── entry point ─────────────────────────────────────────────────────────
    def run(self, tree: ast.Module) -> dict[str, FunctionInfo]:
        # `async def` COUNTS AS A DEF EVERYWHERE HERE. Filtering on
        # `ast.FunctionDef` alone left every `async def` out of the module's
        # function table, so a call to one from anywhere reported "call to
        # unknown function" while the note listed it among the known ones.
        # NAMESPACES FIRST, before signatures and before any body. A
        # parameter may be annotated with a Java type, and whether an
        # annotation is one decides whether the function takes the static path
        # at all -- so the imports have to be read before the first signature
        # is built, not merely before the first body.
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._remember_namespace(alias)
        # AND THEN THE CLASSES OVER JAVA TYPES, for the same reason one step
        # further on: `def place(b: MyBlock)` names one, so what those classes
        # are has to be settled before any signature is built either.
        self._collect_ctypes(tree)
        self._declare_java_subclasses(tree)

        defs = [n for n in tree.body if isinstance(n, _DEF_NODES)]
        # WHETHER THERE IS AN ENTRY AT ALL is decided by the RUNNABLE
        # statements: a module of pure definitions has none and runs `main`
        # instead.
        # An `import` of a BACKEND NAMESPACE is not runnable. It binds a name
        # the analyser recognises and emits nothing at all -- so a module whose
        # only top-level statements are those still has no entry, and keeps the
        # `main`-is-the-program shape. Without this, `import com.minecraft.block`
        # at the top of a file made the module body the entry, and a module
        # body is DYNAMIC: the program acquired the whole object runtime for an
        # import that costs no instructions.
        # A `class` over a JAVA TYPE is not runnable either, and for the same
        # reason: it declares a type the backend generates and emits not one
        # instruction where it is written. Counting it would make the module
        # body the entry, and a module body is dynamic.
        # A CTYPES DECLARATION IS NOT RUNNABLE, for the same reason as the two
        # above: `libm = ctypes.CDLL("m")` names a library for the linker and
        # `libm.sqrt.restype = ...` names a signature for the caller, and
        # neither emits an instruction. Counting them made the module body the
        # entry -- so a program whose `main` did all the work had `main`
        # renamed out of the way and never called, and printed nothing at all
        # while compiling and linking perfectly.
        runnable = [n for n in tree.body
                    if not isinstance(n, _DEF_NODES) and not _is_docstring(n)
                    and not self._is_namespace_import(n)
                    and not self._is_java_subclass(n)
                    and id(n) not in self.ctypes_stmts]
        # WHAT THE ENTRY CONTAINS keeps the `def` STATEMENTS too, in source
        # order. Their own bodies are analysed separately, through `defs` --
        # but the statement still has to RUN where it is written, because that
        # is where Python evaluates the defaults and applies the decorators.
        # Lifting them out meant `n = 1` / `def f(v=n)` / `n = 99` reached `n`
        # before the module had bound it, and a decorator naming something the
        # module assigns did the same.
        body = ([n for n in tree.body
                 if not _is_docstring(n) and not self._is_java_subclass(n)
                 and id(n) not in self.ctypes_stmts]
                if runnable else [])

        # The module's own names, before anything else: a function body may
        # read one, and which names exist at module scope is the question that
        # decides whether an unresolved name is a global or a mistake.
        self.module_names = self._module_names(
            [n for n in tree.body if not isinstance(n, _DEF_NODES)
             and not self._is_java_subclass(n)])
        # PEP 695's type parameters, from the DEFINITIONS TOO. `_module_names`
        # is given only the non-definition statements -- a `def`'s body binds
        # its own names, not the module's -- so a `def first[T]` had nowhere
        # for `T` to live and the annotation thunk read an unknown global.
        for node in tree.body:
            for one in getattr(node, "type_params", ()):
                self.module_names.add(one.name)

        # Signatures first, so a function may call one defined later.
        for node in defs:
            if node.name in self.functions:
                # DEFINED TWICE IS LEGAL AND MEANS REBINDING. The second `def`
                # replaces the name; the first stays reachable only through
                # whatever already held it -- `@f.register` over two `def _`s
                # is the idiom that makes this worth supporting, and refusing
                # it rejected a program CPython runs. The earlier one is MOVED
                # ASIDE under its own key so its body is still compiled, and
                # the bare name is the LAST one, which is what a call written
                # after both resolves to.
                first = self.functions[node.name]
                key, n = node.name, 1
                while key in self.functions:
                    n += 1
                    key = f"{node.name}#{n}"
                first.name = key
                self.functions[key] = first
                self.def_of_node[id(first.node)] = key
                self.rebound.add(node.name)
                self.functions[node.name] = self._signature(node)
                # A REBOUND NAME IS A VALUE, and a value is an `apy_value`.
                # The same rule a NESTED `def` follows and for the same
                # reason: there is no way to hand a machine-word function to
                # `apy_call`, and a call to a name defined twice has to go
                # through the value -- it is the only thing that knows which
                # definition has run. Annotations do not buy the static path
                # here either.
                #
                # BOTH DEFINITIONS, not just the later one. Leaving the
                # first on the static path gave the name no module storage
                # to rebind, so a call after the second reached the FIRST
                # signature: `f(y=1)` was refused over a parameter the `def`
                # above it plainly names, and forcing the value form on top
                # of a machine-word body handed `apy_call` an unboxed
                # argument instead.
                for one in (first, self.functions[node.name]):
                    if not one.dynamic:
                        one.dynamic = True
                        one.ret = OBJ
                        for p in one.params:
                            p.type = OBJ
            else:
                self.functions[node.name] = self._signature(node)
            self.def_of_node[id(node)] = node.name

        # A module-level `def` or `class` BINDS A MODULE NAME, and now that
        # both are values something may read it rather than call it. Added
        # only when there is an entry to initialise the storage: a module of
        # pure definitions runs `main` directly and never executes a `def`
        # statement, so a global there would be permanently unset.
        if any(not isinstance(n, _DEF_NODES) and not _is_docstring(n)
               for n in tree.body):
            self.module_names |= {
                n.name for n in tree.body
                if (isinstance(n, ast.ClassDef)
                    and not self._is_java_subclass(n))
                or (isinstance(n, _DEF_NODES)
                    and self.functions[n.name].dynamic)}

        # Scopes next, and before any body: a closure may capture a name the
        # enclosing function assigns LATER than the `def`, so where a name
        # lives cannot be decided while walking the body that mentions it.
        self._collect_scopes(tree, defs)
        self._resolve_closures()

        entry = self._entry(body)
        for node in defs:
            # THROUGH THE NODE'S OWN KEY, not the bare name: a name defined
            # twice has the earlier one filed under a suffixed key, and
            # looking the name up would compile the survivor twice and the
            # other never.
            self._body(self.functions[self.def_of_node[id(node)]])
        for key in self._java_method_keys:
            self._body(self.functions[key])
        for key in self._nested_keys:
            self._body(self.functions[key])
        if entry is not None:
            self._body(entry)
            # A module-level `def`'s DECORATORS run at module level, not in
            # the function they decorate, so they are checked here -- inside
            # the entry's scope and after its body, which is what makes an
            # unknown one a NAME ERROR rather than an unbound global that
            # reaches lowering and fails the IR verifier.
            self.current = entry
            self.dynamic = True
            for node in defs:
                for deco in node.decorator_list:
                    self._expr(deco)
        return self.functions

    # ── scope collection ────────────────────────────────────────────────────
    def _collect_scopes(self, tree: ast.Module, defs: list) -> None:
        """Build the scope tree and register every nested `def` and `class`.

        Registration happens HERE rather than as each body is walked, so that a
        name can be resolved against the whole nesting before any of it is
        checked. The module scope is created even though module names are
        handled by `module_names` -- it terminates the walk upward, and a
        function scope whose search reaches it has found a global.
        """
        module = self._new_scope(ENTRY_NAME, "module", None, tree)
        module.bound |= set(self.module_names)
        for node in tree.body:
            self._collect(node, module,
                          qual="" if not isinstance(node, ast.FunctionDef)
                          else node.name)
        # Lambdas last, and by INDEX over a list that grows: registering one
        # collects its body, which may contain more. Draining after the whole
        # statement walk rather than at each `_expr_names` call site is what
        # makes a lambda in a `while` test or an `except` clause get a scope
        # like any other -- there are a dozen such sites and hooking some of
        # them is how one silently would not.
        i = 0
        while i < len(self.scopes):
            scope = self.scopes[i]
            i += 1
            j = 0
            while j < len(scope.lambdas):
                self._register_lambda(scope.lambdas[j], scope)
                j += 1

    def _new_scope(self, key: str, kind: str, parent, node=None) -> _Scope:
        scope = _Scope(key, kind, parent, node)
        self.scopes.append(scope)
        self.scope_of[key] = scope
        return scope

    def _register_lambda(self, lam, scope: _Scope) -> None:
        """Give one lambda a scope of its own.

        A lambda is a nested function that happens to be written inline, so it
        gets exactly what a nested `def` gets -- and by going through the same
        `_register_nested`, it gets closures and cells for free rather than a
        second mechanism that would have to grow them separately.
        """
        genexp = isinstance(lam, ast.GeneratorExp)
        made = genexp_def(lam) if genexp else lambda_def(lam)
        key = self._register_nested(made, scope,
                                    "<genexpr>" if genexp else "<lambda>")
        self.def_of_node[id(lam)] = key
        child = self._new_scope(key, "function", scope, made)
        child.bound.update(parameter_names(made.args))
        for stmt in made.body:
            self._collect(stmt, child)

    def _collect(self, node, scope: _Scope, qual: str = "") -> None:
        """Record one statement's bindings and reads into `scope`.

        A `def` or `class` gets a child scope and is NOT descended into here;
        everything else contributes to this one. Written out per statement kind
        rather than over `ast.iter_fields`, because the two questions -- which
        sub-node is a nested SCOPE and which is an expression evaluated HERE --
        have different answers per statement and a generic walk gets the
        `def`'s default arguments wrong: they are evaluated where the `def` is
        written, not where the function runs.
        """
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope.bound.add(node.name)
            for d in node.args.defaults:
                _expr_names(d, scope)
            for dec in node.decorator_list:
                _expr_names(dec, scope)
            key = self._register_nested(node, scope, qual or node.name)
            child = self._new_scope(key, "function", scope, node)
            child.bound.update(parameter_names(node.args))
            for s in node.body:
                self._collect(s, child)
            return
        if isinstance(node, ast.ClassDef):
            scope.bound.add(node.name)
            if self._is_java_subclass(node):
                # Declared already, and not a class in the sense the rest of
                # this understands. Its BASE is not a name to resolve either:
                # it is a namespace attribute, which binds nothing.
                self._collect_java_subclass(node, scope)
                return
            for b in node.bases:
                _expr_names(b, scope)
            key = self._register_class(node, scope, qual or node.name)
            child = self._new_scope(key, "class", scope, node)
            for s in node.body:
                self._collect(s, child)
            return

        match node:
            case ast.Global(names=names):
                scope.globals.update(names)
            case ast.Nonlocal(names=names):
                scope.nonlocals.update(names)
            # A TARGET goes through `_expr_names` as well as `_target_names`,
            # and the pair is not redundant. In `x.attr = v` the base `x` is a
            # LOAD inside a Store-context Attribute, so it is a read of x and
            # binds nothing; `x[i] = v` reads both x and i. `_target_names`
            # answers the plain-name half and `_expr_names` the rest, and
            # neither alone covers `a, b[i] = ...`.
            case ast.Assign(targets=targets, value=value):
                _expr_names(value, scope)
                for t in targets:
                    scope.bound.update(_target_names(t))
                    _expr_names(t, scope)
            case ast.AnnAssign(target=target, value=value):
                _expr_names(value, scope)
                scope.bound.update(_target_names(target))
                _expr_names(target, scope)
            case ast.AugAssign(target=target, value=value):
                _expr_names(value, scope)
                # `x += 1` READS x as well as binding it.
                scope.reads.update(_target_names(target))
                scope.bound.update(_target_names(target))
                _expr_names(target, scope)
            case ast.For(target=target, iter=it, body=b, orelse=o):
                _expr_names(it, scope)
                scope.bound.update(_target_names(target))
                _expr_names(target, scope)
                for s in list(b) + list(o):
                    self._collect(s, scope)
            case ast.While(test=test, body=b, orelse=o) \
                    | ast.If(test=test, body=b, orelse=o):
                _expr_names(test, scope)
                for s in list(b) + list(o):
                    self._collect(s, scope)
            case ast.Try():
                for s in list(node.body) + list(node.orelse) \
                        + list(node.finalbody):
                    self._collect(s, scope)
                for h in node.handlers:
                    _expr_names(h.type, scope)
                    if h.name:
                        scope.bound.add(h.name)
                    for s in h.body:
                        self._collect(s, scope)
            case _:
                for sub in ast.iter_child_nodes(node):
                    if isinstance(sub, ast.expr):
                        _expr_names(sub, scope)
                    elif isinstance(sub, ast.stmt):
                        self._collect(sub, scope)

    def _scope_qual(self, scope: _Scope) -> str:
        """The WRITTEN qualname of the scope a definition sits in, or "" at
        module level.

        Read from whatever registered that scope rather than from its key,
        so that the three differences between a key and a qualname are
        already gone by the time a nested name is built on it. See
        `_register_nested`.
        """
        if scope.kind == "module":
            return ""
        info = self.functions.get(scope.key)
        if info is not None and info.qualname:
            return info.qualname
        held = self.classes.get(scope.key)
        if held is not None and held.qualname:
            return _written_qual(held.qualname)
        return _written_qual(scope.key)

    def _register_nested(self, node, scope: _Scope, qual: str) -> str:
        """Give a `def` below module level its own FunctionInfo."""
        known = self.def_of_node.get(id(node))
        if scope.kind == "module" and known is not None \
                and self.functions.get(known) is not None \
                and self.functions[known].node is node:
            # A module-level `def`, already registered by `run`. Its key is
            # its bare name -- or a suffixed one where the module defines that
            # name twice, in which case `run` has recorded which is which.
            info = self.functions[known]
            info.qualname = node.name
            info.is_value = info.dynamic
            return known

        owner = scope.key if scope.kind == "class" else None
        parent = scope
        while parent is not None and parent.kind == "class":
            parent = parent.parent
        qualname = (f"{scope.key}.{node.name}" if scope.kind == "class"
                    else f"{scope.key}.<locals>.{node.name}")
        key = qualname
        n = 1
        while key in self.functions:
            n += 1
            key = f"{qualname}#{n}"      # two `def f` in one body: both exist
        info = self._signature(node)
        info.name = key
        # THE NAME A PROGRAM READS, BUILT FROM THE ENCLOSING SCOPE'S OWN
        # QUALNAME AND NOT FROM ITS KEY. The two differ in three ways, and a
        # key leaking into `__qualname__` showed all three at once -- a
        # generator expression inside a module-level lambda read
        # `<module>.<locals>.<lambda>#19.<locals>.<genexpr>` where CPython
        # says `<lambda>.<locals>.<genexpr>`:
        #
        #   the `<module>` prefix   CPython's qualname names the enclosing
        #                           functions and classes and never the
        #                           module, so a definition at module level
        #                           is qualified by nothing
        #   the `#19` suffix        what tells two `def f` in one body apart,
        #                           which is a fact about this compiler
        #   the mangling           `_asmpy_bundled_9_fractions_Fraction` is
        #                           how a spliced class avoids colliding
        #                           with the program's own names
        #
        # The key keeps all three; only the spelling a program can READ is
        # rebuilt. See `_written_qual` for the third.
        outer = self._scope_qual(scope)
        if scope.kind == "module":
            info.qualname = node.name
        elif not outer:
            info.qualname = node.name
        elif scope.kind == "class":
            info.qualname = f"{outer}.{node.name}"
        else:
            info.qualname = f"{outer}.<locals>.{node.name}"
        info.parent = parent.key if parent is not None else None
        info.owner = owner
        info.is_value = True
        if not info.dynamic:
            # A nested function is a VALUE, and a value is an `apy_value`.
            # There is no way to hand a machine-word function to `apy_call`,
            # so annotations do not buy the static path here.
            info.dynamic = True
            info.ret = OBJ
            for p in info.params:
                p.type = OBJ
        self.functions[key] = info
        self.def_of_node[id(node)] = key
        self._nested_keys.append(key)
        if owner is not None:
            self.classes[owner].methods.append(key)
        return key

    def _register_class(self, node: ast.ClassDef, scope: _Scope,
                        qual: str) -> str:
        qualname = (node.name if scope.kind == "module"
                    else f"{scope.key}.{node.name}")
        key = qualname
        n = 1
        while key in self.classes:
            n += 1
            key = f"{qualname}#{n}"
        # PEP 3155, BUILT FROM THE ENCLOSING SCOPE'S OWN QUALNAME and not
        # from the key, for the three reasons `_register_nested` sets out at
        # length: the key carries a `<module>` prefix a qualname never does,
        # a `#2` suffix telling two statements of one name apart, and the
        # mangling a spliced definition is spliced under. A class written
        # inside a FUNCTION qualifies through `<locals>` and one written
        # inside another class does not -- which is the whole difference
        # between `mk.<locals>.D` and `C.Inner`.
        outer = self._scope_qual(scope)
        if scope.kind == "module" or not outer:
            shown = node.name
        elif scope.kind == "class":
            shown = f"{outer}.{node.name}"
        else:
            shown = f"{outer}.<locals>.{node.name}"
        base = None
        bases: list = []
        base_exprs: list = []
        builtin_base = None
        is_meta = False
        for i, written in enumerate(node.bases):
            # PEP 560: `class Box(Generic[T])` has `Generic` as its base, not
            # the subscript -- `__mro_entries__` on the alias answers the
            # origin, and for every generic in the standard library that IS
            # the origin. The type arguments say nothing the runtime keeps.
            if isinstance(written, ast.Subscript)                     and isinstance(written.value, ast.Name):
                written = ast.copy_location(written.value, written)
                node.bases[i] = written
            if not isinstance(written, ast.Name):
                # PEP 560: A BASE NEED NOT BE A CLASS. `class C(Fake())` asks
                # the object for `__mro_entries__`, and what that answers is
                # what the class actually inherits -- which is how a generic
                # alias and every library that builds bases at run time work.
                # Kept as an expression because there is no name to record.
                # CHECKED WHERE THE STATEMENT IS, not here: this runs during
                # the scope pass, when the classes it may name are not all
                # registered yet -- `class C(Fake())` above `class Fake` is
                # ordinary and was reported as an unknown call.
                base_exprs.append(written)
                continue
            if written.id == "type":
                # `class Meta(type)` -- A METACLASS. `type` is not a class
                # defined in this module and never will be; what the base
                # records is that calling this builds a CLASS.
                is_meta = True
                continue
            # `class C(object):` is `class C:` -- every class already has
            # `object` at the root of its chain, so naming it explicitly says
            # nothing this runtime does not already do. Writing it is a Python
            # 2 habit that plenty of code still carries, and refusing it
            # rejected programs over punctuation.
            if written.id == "object":
                continue
            if written.id in _BUILTIN_BASES:
                # `class D(dict)`. The base is a KIND and not a class, so it
                # is recorded separately: an instance of D carries a real dict
                # for everything the body does not write, and `isinstance(d,
                # dict)` answers True because the class says so.
                builtin_base = written.id
                continue
            bases.append(written.id)
        # `base` IS THE FIRST ONE and stays for everything that asks a single
        # question -- `__base__`, and the places that walk one chain. The full
        # list is what the MRO is linearised from.
        if bases:
            base = bases[0]
        metaclass, class_keywords = None, []
        for kw in node.keywords:
            if kw.arg is None:
                self._error("E0090", "`**` in a class statement is not "
                                     "supported", node)
            elif kw.arg == "metaclass":
                if isinstance(kw.value, ast.Name):
                    metaclass = kw.value.id
                else:
                    self._error("E0073", "a metaclass must be a plain name",
                                node)
            else:
                # Every other keyword travels to `__new__`, `__init__` and
                # `__init_subclass__` -- which is the whole of what a class
                # keyword does.
                class_keywords.append(kw.arg)
        info = ClassInfo(node, node.name, shown, base, bases=bases,
                         base_exprs=base_exprs, builtin_base=builtin_base,
                         scope=scope.key, is_meta=is_meta,
                         metaclass=metaclass,
                         class_keywords=class_keywords)
        # WHETHER THIS IS AN EXCEPTION CLASS is decided HERE, in the scope
        # pass, and not where the `class` statement is checked. Function
        # bodies are analysed before the module's, so a `def` that says
        # `except MyError:` is checked before the `class MyError` statement
        # is reached -- and reported the name as unknown.
        if base is not None and (base in _EXC_NAMES or base in self.exc_classes):
            info.is_exception = True
            self.exc_classes[node.name] = base
        self.classes[key] = info
        self.class_of_node[id(node)] = key
        return key

    def _resolve_closures(self) -> None:
        """Decide, for every mentioned name, whether it is captured.

        The rule is Python's: a name not bound in this function, but bound in
        some enclosing FUNCTION, is free here and a cell there. Every function
        BETWEEN the two gets it as free as well, because the closure is built
        one level at a time -- an inner function two levels down receives the
        box from its immediate parent, which must therefore have it to give.
        """
        for scope in self.scopes:
            if scope.kind != "function":
                continue
            wanted = (scope.reads | scope.nonlocals) - scope.globals
            for name in sorted(wanted):
                if name in scope.locals:
                    continue
                chain: list = []
                binder = scope.enclosing_function()
                while binder is not None and binder.kind == "function":
                    if name in binder.locals:
                        break
                    chain.append(binder)
                    binder = binder.enclosing_function()
                if binder is None or binder.kind != "function":
                    # Nothing encloses it: a module global or a builtin, both
                    # of which the existing `module_names` path already
                    # answers. `nonlocal` with no binder is the one error.
                    if name in scope.nonlocals:
                        self._error(
                            "E0067",
                            f"no binding for nonlocal {name!r} was found in "
                            f"an enclosing function", scope.node)
                    continue
                if name not in binder.cells:
                    binder.cells.append(name)
                for mid in chain + [scope]:
                    if name not in mid.frees:
                        mid.frees.append(name)

        for scope in self.scopes:
            info = self.functions.get(scope.key)
            if info is not None and scope.kind == "function":
                info.cellvars = list(scope.cells)
                info.freevars = list(scope.frees)

    def _module_names(self, body: list) -> set:
        """Every name the module's top level binds.

        Collected BEFORE any function body is checked, because a function may
        refer to a global defined after it -- the lookup happens when the
        function runs, not where it is written, and rejecting that would refuse
        a shape most Python files have.
        """
        found: set = set()
        for node in body:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        found.update(_target_names(t))
                elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                    found.update(_target_names(sub.target))
                elif isinstance(sub, ast.For):
                    found.update(_target_names(sub.target))
                elif isinstance(sub, ast.ExceptHandler) and sub.name:
                    found.add(sub.name)
                elif getattr(sub, "type_params", None):
                    # PEP 695: `def first[T](...)` and `class Box[T]` put `T`
                    # in scope for the annotations and the body. Bound at
                    # MODULE level here rather than scoped to the definition,
                    # which is a stated divergence: writing `T` after the
                    # `def` is a NameError in CPython and finds the TypeVar
                    # here. The annotations are the reason -- they are built
                    # by a thunk that runs later and reads its names as
                    # globals, so a scope that ended at the `def` would leave
                    # the thunk with nothing to read.
                    for one in sub.type_params:
                        found.add(one.name)
                elif isinstance(sub, ast.With):
                    # `with a as x:` binds x. A `withitem` is not a statement
                    # or an expression, so neither of the branches above sees
                    # it and the name had no module storage.
                    for item in sub.items:
                        if isinstance(item.optional_vars, ast.Name):
                            found.add(item.optional_vars.id)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    # An import BINDS A MODULE NAME, so a function below can
                    # read it -- which is the whole shape of a script that
                    # imports at the top and uses it further down.
                    found.update(_import_names(sub))
        return found

    def _entry(self, body: list) -> FunctionInfo | None:
        """Make the module's top-level statements the program.

        Running a Python file runs its top-level statements; a `def main` in it
        is an ordinary name that happens to be spelled `main`. This frontend
        began the other way round -- a module was a bag of definitions and
        `main` was the entry point by convention -- which meant no program that
        looks like a Python script would compile, and every conformance case is
        a script.

        Both shapes work now, under one rule: the module body is the entry, and
        a module with no top-level statements falls back to the old convention
        of `main` being it. That fallback is what keeps every already-written
        `def main() -> int:` program compiling and doing what it did.

        The synthesised entry is registered under a name no Python identifier
        can collide with, and lowering gives it the IR name `main` that the
        backends and the C runtime agree on. A user function actually spelled
        `main` in a module that HAS top-level statements is then an ordinary
        function that the entry may call, and lowering renames it out of the
        way -- two definitions of the entry symbol is a C compiler error
        naming a function the user never wrote.
        """
        info = self.functions.get("main")
        # A LIBRARY HAS NO ENTRY and is not supposed to. The object runtime's
        # ported pieces are definitions and nothing else -- they are called by
        # a program, not run -- so "nothing to run" is the correct shape rather
        # than the mistake it usually is. See `objects/ir.py`.
        if self.library:
            if body:
                self.sink.report(
                    error("E0037", "a runtime module has no top level")
                    .at(self._span(body[0]))
                    .note("it is compiled into other programs, so a statement "
                          "here has no moment at which it would run")
                    .help("put it in a function"))
            return None
        if body:
            node = ast.Module(body=body, type_ignores=[])
            node.lineno, node.col_offset = getattr(body[0], "lineno", 1), 0
            entry = FunctionInfo(node, ENTRY_NAME, [], INT, dynamic=True)
            self.functions[ENTRY_NAME] = entry
            return entry

        if info is None:
            # Only definitions, and no `main` to call: nothing would run. That
            # is a legal (empty) Python program, but it is far more often a
            # mistake than an intention, and the diagnostic costs nothing --
            # the moment the module grows one executable statement it stops
            # applying, because that statement IS the program.
            self.sink.report(error("E0003", "nothing to run")
                             .note("this module is only definitions, and "
                                   "defines no `main` to call")
                             .help("add top-level statements, or a "
                                   "`def main() -> int:`"))
            return None
        if info.params:
            self.sink.report(
                error("E0008", "`main` takes no parameters")
                .at(self._span(info.node))
                .note("with no top-level statements it is the program's entry "
                      "point, and there is nothing to pass it")
                .help("read arguments from a runtime function instead"))
        if info.ret is not INT and not info.ret.is_error:
            self.sink.report(
                error("E0009", f"`main` must return int, not {info.ret}")
                .at(self._span(info.node))
                .note("the return value becomes the process exit code"))
        # No top-level statements: `main` IS the entry, exactly as before. Not
        # a synthesised `return main()` wrapper -- that would work, but it puts
        # a trampoline where every reader (and every test) expects the IR
        # function called `main` to be the one the user wrote.
        return None

    # ── signatures ──────────────────────────────────────────────────────────
    def _signature(self, node: ast.FunctionDef,
                   self_type: SemType | None = None) -> FunctionInfo:
        """The types of one `def`'s parameters and result.

        `self_type` is given for a method of a class over a Java type, whose
        first parameter is `self` and carries no annotation -- writing one
        would mean naming the class inside its own body, which Python does not
        allow, and demanding it would be ceremony for a type there is only one
        possible answer to.
        """
        params: list[Symbol] = []
        seen: set[str] = set()
        # One unannotated parameter makes the WHOLE function dynamic, not just
        # that parameter. A function with a machine-word `int` in one slot and
        # a runtime object in another would need a conversion at every use of
        # either, and deciding per-parameter is how a compiler ends up with a
        # value whose representation depends on where it came from.
        # An `object` annotation is as dynamic as a missing one: the value's
        # kind is decided at run time either way, and `def f(v: object)` is how
        # a program says so explicitly. Read from the SOURCE rather than from
        # the resolved types, because resolving a parameter's annotation is
        # what this loop is about to do.
        # ANY annotation the static path cannot represent is dynamic, not an
        # error. `int`, `float`, `bool` and `None` are the machine words it
        # knows; `str`, `list[int]`, `Optional[int]`, a forward reference in
        # quotes and every typing construct are runtime objects, and saying so
        # is exactly what the dynamic path is for. Refusing them instead meant
        # a program that merely ANNOTATES did not compile at all -- and an
        # annotation is documentation, so a compiler that cannot use one
        # should ignore it rather than reject the program.
        def _is_dynamic_annotation(a) -> bool:
            if a is None:
                return True
            name = getattr(a, "id", None)
            if getattr(a, "pep563", False):
                # PEP 563 made this text. The program asked for the annotation
                # NOT to be evaluated, so it cannot also be the type.
                return True
            if name is None and isinstance(a, ast.Constant):
                name = str(a.value)
            if self._java_annotation(a) is not None:
                # A JAVA TYPE is a machine word -- a handle -- so a function
                # annotated with one stays static. Falling through to the
                # dynamic path here made every parameter an `object`, which is
                # a representation the JVM backend has no runtime for. The
                # bare-name form is one of these too, and only for a class
                # THIS source declares over a Java type.
                return False
            return name not in BY_NAME or name == "object"
        # A MISSING RETURN ANNOTATION is dynamic too, and this is not a
        # detail: `def f():` has no parameters, so "every parameter is
        # annotated" is vacuously true and a zero-argument function in an
        # ordinary script was taking the static path -- where module-level
        # names do not exist, so `def f(): return top` could not see `top`.
        # POSITIONAL-ONLY FIRST, then ordinary, then keyword-only: that is
        # the order they occupy argument positions in, and every count below
        # -- defaults, arity, the slot map -- is stated against it.
        declared = (list(node.args.posonlyargs) + list(node.args.args)
                    + list(node.args.kwonlyargs))
        # `self` is already typed and carries no annotation, so it is excluded
        # from the question the rest of them answer -- including it would make
        # every such method dynamic on the strength of the one parameter that
        # cannot be annotated.
        rest = declared[1:] if self_type is not None else declared
        dynamic = (any(_is_dynamic_annotation(a.annotation) for a in rest)
                   or _is_dynamic_annotation(node.returns))
        for i, arg in enumerate(declared):
            if arg.arg in seen:
                self._error("E0004",
                            f"duplicate parameter {arg.arg!r}", arg)
            seen.add(arg.arg)
            if self_type is not None and i == 0:
                ty = self_type
            else:
                ty = (OBJ if dynamic
                      else self._annotation(arg.annotation, arg,
                                            f"parameter {arg.arg!r}"))
            params.append(Symbol(arg.arg, ty, self._span(arg), is_param=True))
        if not dynamic and (node.args.posonlyargs or node.args.kwonlyargs):
            # The static path has no call-site name matching to restrict: its
            # arguments are machine words in registers, filled by position and
            # nothing else.
            self._error("E0005", "positional-only and keyword-only parameters "
                                 "are not supported here", node)
        if node.args.kwarg and not dynamic:
            # The static path has nowhere to put one: its parameters are
            # machine words and a `**kw` is a dict.
            self._error("E0005", "**kwargs are not supported here", node)
        if node.args.vararg and not dynamic:
            self._error("E0005", "*args are not supported here", node)
        if node.args.defaults and not dynamic:
            # The static path has no place to keep a default: its parameters
            # are machine words in registers and there is no per-function
            # storage to evaluate one into.
            self._error("E0006", "default arguments are not supported here",
                        node)
        if node.decorator_list and not dynamic:
            # The static path has no value to hand a decorator: its functions
            # are machine-word signatures, not objects, so there is nothing
            # for `f = deco(f)` to rebind.
            self._error("E0007", "decorators are not supported here", node)

        if dynamic:
            ret = OBJ
        else:
            ret = (self._annotation(node.returns, node, "the return type")
                   if node.returns else NONE)
        info = FunctionInfo(node, node.name, params, ret, dynamic=dynamic)
        # A `yield` ANYWHERE in the body makes this a generator, whatever else
        # the body does -- so `def g(): yield 1` and `def g(): ... yield ...`
        # deep inside a loop are the same kind of function. It is decided here
        # rather than while checking the body, because a call to it has to
        # know before the body is reached.
        info.is_coroutine = dynamic and isinstance(node, ast.AsyncFunctionDef)
        # An `async def` is lowered as a generator whether or not it yields:
        # `await` suspends it, so its locals have to survive the return that
        # a suspension compiles to, exactly as a `yield`'s do.
        info.is_generator = dynamic and (_has_yield(node) or info.is_coroutine)
        # `async def` WITH `yield` is a third thing, and awaiting one is an
        # error where iterating it is not.
        info.is_async_generator = info.is_coroutine and _has_yield(node)
        if info.is_generator:
            info.ret = OBJ
        if dynamic:
            info.defaults = list(node.args.defaults)
            info.vararg = node.args.vararg.arg if node.args.vararg else None
            info.kwarg = node.args.kwarg.arg if node.args.kwarg else None
            info.posonly = len(node.args.posonlyargs)
            info.kwonly = len(node.args.kwonlyargs)
            # A keyword-only parameter's default lives in its own list, and a
            # None there means REQUIRED rather than "defaults to None". Both
            # kinds end up as trailing defaults, which is the layout the
            # function value already has.
            kwd = [d for d in node.args.kw_defaults if d is not None]
            info.defaults = list(node.args.defaults) + kwd
            info.kwdefaults = len(kwd)
            if info.vararg:
                # `*rest` is an ordinary local holding a tuple. Declared as a
                # parameter would make the arity checks count it.
                pass
        return info

    def _no_compiler(self, name: str, at) -> None:
        """`compile`, `eval` and `exec` -- allowed, and said out loud.

        A WARNING AND NOT AN ERROR. The program gets a real `compile()`: the
        parser, the validator and the code object are bundled Python, spliced
        in like any other module -- see `bundled/_pycompile.py`. What it is
        not is the compiler that built the binary, and the difference is worth
        a line at the call site rather than a surprise later.

        TWO DIFFERENT WARNINGS, because the costs differ. `compile()` answers
        whether source is valid Python and stops there. `eval()` and `exec()`
        RUN it, and running it is interpretation -- everything reached that
        way is orders of magnitude slower than the native code around it, and
        that is the part a reader needs told.
        """
        said = warning(
            "W0091", f"{name}() is not recommended in a compiled program")
        said = said.at(self._span(at))
        if name == "compile":
            said = said.note(
                "it answers whether source is valid Python, through the "
                "parser bundled into this binary -- not through the compiler "
                "that built it")
        else:
            said = said.note(
                f"the source {name}() is given is INTERPRETED, not compiled: "
                f"it runs through the interpreter bundled into this binary "
                f"and is far slower than the code around it")
        self.sink.report(said.help(
            "prefer a function you can call directly, which the compiler can "
            "see and optimise"))

    def _annotation(self, ann, at, what: str) -> SemType:
        if ann is None:
            self.sink.report(
                error("E0010", f"{what} needs a type annotation")
                .at(self._span(at))
                .help("annotate with int, float, bool or None"))
            return ERROR
        # A JAVA TYPE, written the way it is imported: `block.Block`. Without
        # this a handle could be produced and used inside one function and
        # never leave it, because there was no way to spell the parameter that
        # would receive it.
        java = self._java_annotation(ann)
        if java is not None:
            return java
        name = getattr(ann, "id", None)
        if name is None and isinstance(ann, ast.Constant):
            name = str(ann.value)
        if name not in BY_NAME:
            self.sink.report(
                error("E0011", f"unsupported type {ast.unparse(ann)!r} for {what}")
                .at(self._span(ann))
                # THE MACHINE TYPES ARE MENTIONED SECOND, and briefly. Someone
                # who wrote `x: str` needs to hear about `int` and `object`;
                # putting eleven width names in front of those buries the
                # answer under a vocabulary they did not ask for.
                .note("this frontend understands: int, float, bool, None, "
                      "object")
                .note("and the machine types, for code that has to know a "
                      "width: " + ", ".join(MACHINE)))
            return ERROR
        return BY_NAME[name]

    def _java_annotation(self, ann) -> SemType | None:
        """`block.Block` as a type, or None if this is not one.

        The attribute form for a type from the class path: a bare `Block` would
        need the name bound to something, and `import a.b as n` binds `n`
        rather than what is in it. `from a.b import Block` is the spelling that
        would bind the bare name, and it is not supported here yet.

        A class THIS source declares over a Java type is the exception, and the
        bare name is the only spelling it has -- `class MyBlock(block.Block)`
        binds `MyBlock` and nothing else.
        """
        if ann is None or not isinstance(ann, (ast.Attribute, ast.Name)):
            return None
        table = self._java_type_table(ann)
        return None if table is None else sem_type(table["type"])

    # ── bodies ──────────────────────────────────────────────────────────────
    @property
    def class_names(self) -> set:
        """Bare class names, for resolving `C()` and `isinstance(x, C)`."""
        return {c.name for c in self.classes.values()}

    def _body(self, info: FunctionInfo) -> None:
        self.current = info
        self.dynamic = info.dynamic
        # A FREE variable is already bound -- the enclosing frame made the box
        # before this function existed -- so it is declared here and marked
        # assigned. Without that, reading it looks like use-before-assignment,
        # which is exactly the shape `inner-function-sees-later-binding` has:
        # the enclosing function assigns AFTER the `def`.
        for i, name in enumerate(info.freevars):
            info.locals[name] = Symbol(name, OBJ, self._span(info.node),
                                       storage=FREE, index=i)
        if info.kwarg:
            info.locals[info.kwarg] = Symbol(info.kwarg, OBJ,
                                             info.params[0].span
                                             if info.params else None,
                                             is_param=True)
        if info.vararg:
            # `is_param`, like the `**kw` above it: `*rest` ARRIVES IN A
            # REGISTER from the caller. Without the flag, boxing it for a
            # closure built the cell from None and threw the tuple away, so a
            # nested function reading `args` saw None while `kw` worked.
            info.locals[info.vararg] = Symbol(info.vararg, OBJ,
                                              self._span(info.node),
                                              is_param=True)
        scope = self.scope_of.get(info.name)
        if info.dynamic and info.name != ENTRY_NAME:
            declared = (set(scope.globals) if scope is not None
                        else _declared_global(info.node.body))
            info.module_writes = declared & self.module_names
            #: Local for the whole body if bound anywhere in it, minus what
            #: `global` hands back to the module, plus what a closure brought
            #: in -- a free variable is not this frame's storage but it is
            #: NOT the module's either, and leaving it out would make an
            #: enclosing function's `n` read the global of the same name.
            #:
            #: Taken from the SCOPE rather than by walking the body: the walk
            #: descends into nested `def`s, so an inner function's locals
            #: would count as this one's and shadow a global it really reads.
            self._function_locals = (
                (scope.locals if scope is not None
                 else self._module_names([info.node]) - declared)
                | {p.name for p in info.params} | set(info.freevars))
        else:
            self._function_locals = None
        for p in info.params:
            info.locals[p.name] = p
        if info.dynamic and self._function_locals is not None:
            # AN ASSIGNMENT ANYWHERE MAKES THE NAME LOCAL FOR THE WHOLE BODY.
            # That is Python's rule, and it is what makes reading a name
            # before its assignment an `UnboundLocalError` rather than finding
            # the global of the same spelling:
            #
            #     n = 10                  # module level
            #     def f():
            #         print(n)            # UnboundLocalError, not 10
            #         n = 5
            #
            # Declared up front so the read below finds the symbol. A name
            # nothing assigns is NOT here, so a misspelling is still an
            # undefined name reported at compile time.
            for name in sorted(self._function_locals):
                if name not in info.locals:
                    info.locals[name] = Symbol(name, OBJ, self._span(info.node))
        #: Names definitely assigned at the current point. Parameters always
        #: are; everything else has to be reached by an assignment on EVERY
        #: path, which is what makes the intersection at a join the whole
        #: analysis.
        self.assigned = {p.name for p in info.params} | set(info.freevars)
        if info.vararg:
            self.assigned.add(info.vararg)
        if info.kwarg:
            self.assigned.add(info.kwarg)
        self._block(info.node.body)
        # Which locals are BOXED, decided now that every name is declared. A
        # parameter can be one too: `def outer(n)` with an inner function that
        # reads `n` boxes the parameter on entry.
        for i, name in enumerate(info.cellvars):
            sym = info.locals.get(name)
            if sym is not None:
                sym.storage, sym.index = CELL, i
        if info.is_generator:
            # EVERY LOCAL GETS A SLOT, now that every name is declared. A
            # register does not survive the return a `yield` compiles to, so a
            # generator's frame is the object itself -- see `_dyn_generator`.
            # Numbered here rather than during lowering because both halves of
            # the pair, the constructor and the step, must agree on the map.
            info.slots = {name: i for i, name in enumerate(info.locals)}
        self.current = None

    def _block(self, body: list) -> bool:
        """Analyse a list of statements. Returns whether control falls through.

        Falling through matters for the join: a branch that always returns
        contributes nothing to what is assigned afterwards, and treating it
        as a normal path would reject

            if c:
                x: int = 1
            else:
                return 0
            print(x)

        which is correct code and a common shape.
        """
        for stmt in body:
            if not self._stmt(stmt):
                return False
        return True

    def _stmt(self, node) -> bool:
        """Analyse one statement; returns whether control falls through."""
        info = self.current
        assert info is not None
        self.dynamic = info.dynamic
        match node:
            case ast.Expr(value=ast.Constant()):
                # A BARE CONSTANT IS A NO-OP, and the one that matters is a
                # DOCSTRING. Walking it as an expression reported "unsupported
                # expression: Constant" for `def f() -> int: """doc"""` on the
                # static path -- so a statically typed function could not be
                # documented, and the diagnostic said nothing about why.
                # Lowering already skips these; analysis did not.
                pass
            case ast.Expr():
                self._expr(node.value)
            case ast.Assign() if self._ctypes_assign(node):
                pass
            case ast.Assign(targets=[ast.Name(id=name)]):
                self._bind(name, self._expr(node.value), node,
                           value=node.value)
                self.assigned.add(name)
            case ast.Assign(targets=[(ast.Tuple() | ast.List()) as target]) \
                    if self.dynamic:
                self._expr(node.value)
                for name in _target_names(target):
                    self._bind(name, OBJ, node)
                    self.assigned.add(name)
            case ast.Assign(targets=[ast.Subscript() as target]) if self.dynamic:
                self._expr(target.value)
                self._expr(target.slice)
                self._expr(node.value)
            case ast.Assign(targets=[ast.Attribute()]) \
                    if self._is_splice_dunder(node):
                # DROPPED. A splice restores `__name__` on every definition it
                # renamed, and a STATICALLY typed function has nothing to
                # restore it on -- it is a machine-word signature with no
                # object and no module storage, so the read of its global
                # trapped at run time with `NameError` naming the mangled
                # symbol. `from sm import f` where `sm.py` is
                # `def f() -> int: return 1` -- the language's own annotated
                # style, across the file boundary R1 is about.
                #
                # HERE AND NOT IN THE SPLICE, because this is the only place
                # that knows which path a `def` takes. Recorded rather than
                # merely skipped: lowering walks the same tree and would
                # otherwise emit the read this pass just declined to check.
                self.splice_dunder_stmts.add(id(node))
            case ast.Assign(targets=[ast.Attribute() as target]) if self.dynamic:
                # `x.attr = v`. Nothing static to check -- whether `x` has that
                # attribute, and whether it accepts one, are both runtime
                # questions, and the runtime answers them where CPython does.
                self._expr(target.value)
                self._expr(node.value)
            case ast.Assign(targets=[_, _, *_]) if self.dynamic:
                # `a = b = value` -- ONE value, bound to each target left to
                # right. The value is evaluated once, which is what makes
                # `a = b = []` two names for the same list.
                self._expr(node.value)
                for target in node.targets:
                    match target:
                        case ast.Name(id=name):
                            self._bind(name, OBJ, node)
                            self.assigned.add(name)
                        case ast.Tuple() | ast.List():
                            for name in _target_names(target):
                                self._bind(name, OBJ, node)
                                self.assigned.add(name)
                        case ast.Subscript() | ast.Attribute():
                            self._expr(target.value)
                        case _:
                            self._error("E0020",
                                        "only simple `name = value` "
                                        "assignment is supported", node)
            case ast.Assign():
                self._error("E0020", "only simple `name = value` assignment "
                                     "is supported", node)
            case ast.TypeAlias() if self.dynamic:
                # PEP 695: `type Alias = list[int]`. The TYPE PARAMETERS ARE
                # IN SCOPE FOR THE VALUE and nowhere else, so they are bound
                # here for the walk and dropped after it -- writing `T` after
                # the statement is a NameError, as it is in CPython.
                names = {p.name for p in node.type_params}
                for one in names:
                    self._bind(one, OBJ, node)
                    self.assigned.add(one)
                self._expr(node.value)
                self._bind(node.name.id, OBJ, node)
                self.assigned.add(node.name.id)
            case ast.AnnAssign(target=(ast.Attribute() | ast.Subscript()))                     if self.dynamic:
                # `self.items: list[T] = []`. THE ANNOTATION IS NOT KEPT --
                # only a NAME's annotation goes into `__annotations__`, and
                # CPython does not record one on an attribute either -- so
                # this is the assignment it also is, and refusing it rejected
                # ordinary Python over the annotation alone.
                if isinstance(node.target, ast.Attribute):
                    self._expr(node.target.value)
                else:
                    self._expr(node.target.value)
                    self._expr(node.target.slice)
                if node.value is not None:
                    self._expr(node.value)
            case ast.AnnAssign(target=ast.Name(id=name)) if self.dynamic:
                # `x: int = 1` inside a dynamic function. The annotation is
                # not enforced: the value is an object either way, and
                # honouring it would mean one slot in the function held a
                # machine word while its neighbours held objects.
                if node.value is not None:
                    self._expr(node.value)
                    self._bind(name, OBJ, node)
                    self.assigned.add(name)
                else:
                    self._declare(name, OBJ, node)
            case ast.AnnAssign(target=ast.Name(id=name)):
                declared = self._annotation(node.annotation, node, f"{name!r}")
                if node.value is not None:
                    actual = self._expr(node.value)
                    self._check_assignable(actual, declared, node,
                                           value=node.value)
                self._declare(name, declared, node)
                # `x: int` with no value declares a type and assigns nothing,
                # exactly as in Python.
                if node.value is not None:
                    self.assigned.add(name)
            case ast.AugAssign(target=(ast.Subscript() | ast.Attribute())) \
                    if self.dynamic:
                # `xs[i] += v`. The TARGET EXPRESSION IS EVALUATED ONCE --
                # `xs[idx()] += 5` calls `idx` a single time -- which is why
                # this is a statement of its own rather than a rewrite to
                # `xs[i] = xs[i] + v`.
                if isinstance(node.target, ast.Subscript):
                    self._expr(node.target.value)
                    self._expr(node.target.slice)
                else:
                    self._expr(node.target.value)
                self._expr(node.value)
            case ast.AugAssign(target=ast.Name(id=name)):
                # `x += 1` READS x first, so an unassigned x is an error here
                # for the same reason it is in an expression.
                have = self._lookup(name, node)
                # Analysed as the binary operation it is, through the same
                # path a written-out `x = x + 1` takes. Checking only the
                # right-hand side let `x **= n` and `x @= y` past the operator
                # rules and into lowering, which raised at the user.
                synthetic = ast.BinOp(
                    left=ast.Name(id=name, ctx=ast.Load()),
                    op=node.op, right=node.value)
                ast.copy_location(synthetic, node)
                ast.copy_location(synthetic.left, node)
                info.aug_nodes[id(node)] = synthetic
                result = self._expr(synthetic)
                self._check_assignable(result, have, node,
                                       what=f"assignment to {name!r}")
                self.assigned.add(name)
            case ast.Return():
                # A BARE `return` IN A DYNAMIC FUNCTION yields None AS AN
                # OBJECT, which is assignable to the `object` a dynamic
                # function returns. Typing it as the static NONE made
                # `def f(): return` -- and every early exit written that way
                # -- a narrowing error for a program CPython runs.
                got = (self._expr(node.value) if node.value
                       else (OBJ if self.dynamic else NONE))
                # In a GENERATOR a `return` does not produce the call's value
                # -- it ends the iteration -- so there is nothing to check it
                # against. `return` and `return v` are both legal there, and
                # `v` becomes StopIteration's `value` rather than the result.
                if not info.is_generator:
                    self._check_assignable(got, info.ret, node,
                                           what="return value",
                                           value=node.value)
                return False
            case ast.If():
                self._expr(node.test, True)
                before = set(self.assigned)
                then_falls = self._block(node.body)
                then_assigned = set(self.assigned)
                self.assigned = before
                else_falls = self._block(node.orelse)
                else_assigned = set(self.assigned)
                # A name is assigned after the `if` only if every path that
                # reaches here assigned it.
                if then_falls and else_falls:
                    self.assigned = then_assigned & else_assigned
                elif then_falls:
                    self.assigned = then_assigned
                elif else_falls:
                    self.assigned = else_assigned
                else:
                    self.assigned = then_assigned | else_assigned
                return then_falls or else_falls
            case ast.Match() if self.dynamic:
                self._expr(node.subject)
                # EVERY CASE BINDS INTO THE ENCLOSING SCOPE, and none of them
                # is guaranteed to run -- so the names are declared here but
                # NOT added to `assigned`. A `match` that falls through binds
                # nothing, and reading a capture afterwards is the
                # UnboundLocalError CPython raises rather than something to
                # reject at compile time.
                before = set(self.assigned)
                for item in node.cases:
                    for name in _pattern_names(item.pattern):
                        self._declare(name, OBJ, node)
                        self.assigned.add(name)
                    # The GUARD is checked with the captures in scope: `case n
                    # if n > 10` reads the name the pattern just bound.
                    if item.guard is not None:
                        self._expr(item.guard)
                    self._value_pattern_names(item.pattern)
                    self._block(item.body)
                    self.assigned = set(before)
                self.assigned = before
            case ast.Match():
                self._error("E0088", "`match` needs a dynamic function; this "
                                     "one is statically typed", node)
            case ast.While():
                self._expr(node.test, True)
                # The body may run zero times, so nothing it assigns is
                # definitely assigned afterwards.
                before = set(self.assigned)
                self.loop_depth += 1
                self._block(node.body)
                self.loop_depth -= 1
                self.assigned = before
                if node.orelse and not self.dynamic:
                    # The static path's lowering has no `else` on a loop, so
                    # accepting one here would drop it in silence. Rejecting is
                    # a limit; dropping is a wrong answer.
                    self._error("E0026", "`while ... else` is not supported "
                                         "in an annotated function", node)
                self._block(node.orelse)
                self.assigned = before
            case ast.AsyncFor() if self.current.is_coroutine:
                # `async for` is the ordinary loop's shape with an awaitable
                # source, so the same checks apply -- what differs is entirely
                # in the lowering. Outside a coroutine there is nothing to
                # suspend, which is why the guard is on the enclosing function
                # and not on the loop.
                self._expr(node.iter)
                before = set(self.assigned)
                if isinstance(node.target, (ast.Tuple, ast.List)):
                    for nm in _target_names(node.target):
                        self._bind(nm, OBJ, node)
                        self.assigned.add(nm)
                else:
                    self._bind(node.target.id, OBJ, node)
                    self.assigned.add(node.target.id)
                self.loop_depth += 1
                self._block(node.body)
                self.loop_depth -= 1
                self.assigned = before
                self._block(node.orelse)
                self.assigned = before
            case ast.AsyncFor():
                self._error("E0087", "`async for` outside async function", node)
            case ast.For(target=(ast.Tuple() | ast.List())) if self.dynamic:
                self._for_unpack(node)
            case ast.For(target=ast.Name(id=name)):
                self._for_range(node, name)
            case ast.For():
                self._error("E0021", "loop target must be a plain name", node)
            case ast.Break() | ast.Continue():
                if self.loop_depth == 0:
                    self._error(
                        "E0027",
                        f"`{type(node).__name__.lower()}` outside a loop", node)
                return False
            case ast.Global() if self.dynamic:
                for name in node.names:
                    if name not in self.module_names:
                        self._error("E0066",
                                    f"`global {name}` names nothing at module "
                                    f"scope", node)
            case ast.Nonlocal() if self.dynamic:
                # The binding was resolved in `_resolve_closures`, which is
                # also where a nonlocal with nothing to bind to is reported.
                # Nothing left to check here, and nothing to emit: the
                # declaration only changes which storage the assignments use.
                pass
            case ast.Nonlocal():
                self._error("E0067", "`nonlocal` needs a dynamic function",
                            node)
            case ast.FunctionDef() | ast.AsyncFunctionDef() if self.dynamic:
                # A nested `def`, or an `async def` -- the same thing here.
                # Its body was registered and checked on its own; what happens
                # HERE is the binding of its name to a function value, which
                # is an ordinary assignment.
                self._bind(node.name, OBJ, node)
                self.assigned.add(node.name)
            case ast.ClassDef() if self.dynamic:
                self._class_statement(node)
            case ast.FunctionDef() | ast.ClassDef():
                self._error("E0074",
                            f"a nested {'class' if isinstance(node, ast.ClassDef) else 'def'}"
                            f" needs a dynamic function; this one is "
                            f"statically typed", node)
            case ast.Raise() if self.dynamic:
                if node.exc is not None:
                    self._expr(node.exc)
                if node.cause is not None:
                    self._expr(node.cause)
                return False
            case ast.Delete() if self.dynamic:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._lookup(target.id, target)
                        # `del x` UNBINDS. For a LOCAL that is a compile-time
                        # fact, so dropping it from `assigned` turns a later
                        # read into the error CPython raises as
                        # UnboundLocalError, reported earlier.
                        #
                        # For a MODULE name it is not: the cell goes back to
                        # zero and every read of one is already guarded, so
                        # the NameError happens where CPython's does -- at run
                        # time, where `try: print(x) except NameError:` can
                        # catch it. Rejecting that at compile time would make
                        # a program that HANDLES the error impossible to run,
                        # which is a larger divergence than the one the
                        # diagnostic was protecting against.
                        if target.id not in self.module_names:
                            self.assigned.discard(target.id)
                    elif isinstance(target, ast.Subscript):
                        self._expr(target.value)
                        self._expr(target.slice)
                    elif isinstance(target, ast.Attribute) and self.dynamic:
                        # `del obj.attr` -- the runtime already has
                        # `apy_delattr`, and a `__delattr__` or a descriptor's
                        # `__delete__` hangs off it. Only the frontend refused.
                        self._expr(target.value)
                    else:
                        self._error("E0081",
                                    f"`del` of a "
                                    f"{type(target).__name__} is not "
                                    f"supported", target)
            case ast.NamedExpr():
                pass        # only ever reached as an expression
            case ast.Assert() if self.dynamic:
                self._expr(node.test)
                if node.msg is not None:
                    self._expr(node.msg)
            case ast.Import() if self._ctypes_import(node):
                pass
            case ast.ImportFrom() if self._ctypes_import(node):
                pass
            # A DECLARED NATIVE LIBRARY, imported inside a function body. The
            # module-level ones were taken by `_collect_ctypes` and are
            # excluded from the entry, so they never arrive here twice.
            case ast.Import() if self._nativelib_import(node):
                pass
            case ast.ImportFrom() if self._nativelib_import(node):
                pass
            case ast.Import() if not self.dynamic:
                for alias in node.names:
                    self._static_import(alias, node)
            case ast.Import() if self.dynamic:
                for alias in node.names:
                    # Recorded for the STATIC functions below, whichever path
                    # this statement itself took: a module-level `import` is
                    # analysed dynamically because module-level code always is,
                    # and `main` still has to be able to name what it bound.
                    self._remember_namespace(alias)
                    if resolve(alias.name) is None:
                        self._no_module(alias.name, node)
                        continue
                    # `import a.b` BINDS `a`, as Python does: the head is a
                    # package holding the module, so `c.math.sqrt` reads two
                    # attributes off one name. `import a.b as n` binds `n` to
                    # the module itself instead, which is the other half of
                    # the same rule.
                    bound = alias.asname or alias.name.split(".")[0]
                    self._bind(bound, OBJ, node)
                    self.assigned.add(bound)
            case ast.ImportFrom() if self.dynamic:
                if node.module is None or resolve(node.module) is None:
                    self._no_module(node.module or "", node)
                else:
                    for alias in node.names:
                        if member(node.module, alias.name) is None:
                            self._error("E0084",
                                        f"module {node.module!r} has no member "
                                        f"{alias.name!r}", node)
                            continue
                        bound = alias.asname or alias.name
                        self._bind(bound, OBJ, node)
                        self.assigned.add(bound)
            case ast.AsyncWith() if self.current.is_coroutine:
                # The same checks as `with`: the difference is entirely in the
                # lowering, which awaits each half of the protocol.
                return self._check_with(node)
            case ast.AsyncWith():
                self._error("E0089", "`async with` outside async function",
                            node)
            case ast.With() if self.dynamic:
                return self._check_with(node)
            case ast.Try() | ast.TryStar() if self.dynamic:
                # `except*` DIVIDES rather than selects, and that
                # difference is entirely in the lowering: the same
                # names are bound and the same paths reach past the
                # statement, so the same definite-assignment
                # reasoning applies to both.
                return self._dyn_try(node)
            case ast.Pass():
                pass
            case _:
                self._error("E0022",
                            f"unsupported statement: {type(node).__name__}", node)
        return True

    def _check_with(self, node) -> bool:
        """`with` and `async with` -- the same checks for both.

        What differs between them is entirely in the lowering, which awaits
        each half of the protocol; the names bound and the expressions checked
        are identical.
        """
        for item in node.items:
            self._expr(item.context_expr)
            if item.optional_vars is not None:
                if not isinstance(item.optional_vars, ast.Name):
                    self._error("E0082", "`with ... as` needs a plain name",
                                node)
                else:
                    self._bind(item.optional_vars.id, OBJ, node)
                    self.assigned.add(item.optional_vars.id)
        # THE BODY IS NOT GUARANTEED TO FINISH, which is the whole of what an
        # `__exit__` returning true means: the exception is swallowed and the
        # statement after the `with` runs having skipped the rest of the
        # body. So a name the body binds may be UNBOUND there --
        #
        #     with C():          # whose __exit__ answers True
        #         raise ValueError
        #         held = 1
        #     return held        # UnboundLocalError in CPython
        #
        # -- and treating the body's bindings as definite made the lowering
        # emit a read the verifier refused as "reads %1 before any path
        # writes it", which the driver reports as an internal compiler error.
        # An ordinary `with` that binds a name and reads it afterwards would
        # not compile at all.
        #
        # THE `as` NAMES ARE DEFINITE and stay so: `__enter__` has already
        # returned on every path that reaches the body, and one that raises
        # propagates rather than arriving past the statement.
        before = set(self.assigned)
        self._block(node.body)
        self.assigned = before
        # AND THE STATEMENT AFTER A `with` IS ALWAYS REACHABLE, however the
        # body ends: a body that only raises still falls through here when
        # `__exit__` swallows. Reporting it unreachable stopped the analysis
        # at the `with`, so the code past it was never walked -- and the
        # LOWERING walked it anyway, emitting a read of a register nothing
        # had seeded. Whether a given `__exit__` swallows is a run-time
        # question, so the answer here is the one that is safe either way.
        return True

    def _class_statement(self, node: ast.ClassDef) -> None:
        """Check a `class` statement where it is WRITTEN.

        The methods were registered and are checked as functions of their own.
        What is left is the class body's other bindings -- which are class
        attributes, shared by every instance -- and the name the statement
        binds in the enclosing scope.
        """
        info = self.classes[self.class_of_node[id(node)]]
        # A USER EXCEPTION CLASS IS BOTH THINGS AT ONCE. Its NAME goes into
        # the runtime's exception hierarchy -- see `apy_exc_register` in
        # objects/csource.py -- which is what makes `except ValueError:` catch it
        # through the same walk that makes `except LookupError:` catch a
        # KeyError. Its BODY builds an ordinary class, because
        #
        #     class AppError(Exception):
        #         def __init__(self, code, message):
        #             super().__init__(f"{code}: {message}")
        #             self.code = code
        #
        # is how most programs write one, and a name in a table has nowhere to
        # put an `__init__`. So the checks below are the checks any class gets;
        # what an exception class skips is only the question of whether its
        # bases are classes THIS module defines, since its base is a name in
        # the hierarchy and usually a builtin one.
        if info.base is not None and not info.is_exception:
            # EVERY base, not just the first: a second one that names
            # nothing is the same mistake as a first one that does.
            known = {c.name for c in self.classes.values()}
            for one in info.bases:
                if one not in known:
                    self._error("E0076",
                                f"base class {one!r} is not a class "
                                f"defined in this module", node)
        # PEP 560's expression bases are checked HERE, where every class the
        # module defines is registered.
        for one in info.base_exprs:
            self._expr(one)
        for stmt in node.body:
            match stmt:
                case ast.FunctionDef() | ast.AsyncFunctionDef():
                    # `async def` IS A METHOD LIKE ANY OTHER here -- matching
                    # only `FunctionDef` rejected every class with an
                    # `__aenter__` or an async method on it, which is most of
                    # them once a program uses `async with`.
                    #
                    # The body was registered and is checked on its own; the
                    # DECORATORS are not part of it -- they run where the
                    # class body runs, so an unknown one has to be reported
                    # here or it reaches lowering as an unbound global.
                    for deco in stmt.decorator_list:
                        self._expr(deco)
                case ast.Assign(targets=[ast.Name(id=name)], value=value):
                    self._expr(value)
                    info.attrs.append((name, value))
                case ast.AnnAssign(target=ast.Name(id=name), value=value):
                    # THE ANNOTATION IS RECORDED whether or not there is a
                    # value: `a: int` with none declares nothing to bind and
                    # still appears in `C.__annotations__`, which is the whole
                    # point of writing it.
                    if stmt.annotation is not None:
                        info.annotations.append((name, stmt.annotation))
                    if value is not None:
                        self._expr(value)
                        info.attrs.append((name, value))
                case ast.Expr() if not _is_docstring(stmt):
                    # A BARE EXPRESSION IN A CLASS BODY. A class body is a
                    # block that RUNS, once, where it is written -- `log.append
                    # ("body")` in one is ordinary Python, and refusing it
                    # rejected the program over a statement that binds nothing.
                    self._expr(stmt.value)
                    info.body_exprs.append(stmt.value)
                case ast.Pass():
                    pass
                case _ if _is_docstring(stmt):
                    pass
                case _:
                    # A CLASS BODY IS A BLOCK THAT RUNS. `try: import x /
                    # except ImportError: x = None`, `if TYPE_CHECKING:`, a
                    # `for` that builds several attributes -- all ordinary
                    # Python, and refusing them rejected the program over the
                    # SHAPE of a statement rather than anything it did.
                    #
                    # WHAT IT BINDS IS A CLASS ATTRIBUTE, which is why the
                    # names are recorded: `_dyn_class` routes their stores
                    # into the class namespace, and a plain local would make
                    # the name vanish with the body.
                    info.body_stmts.append(stmt)
                    info.body_stmt_names |= _stored_names(stmt)
                    self._block([stmt])
        self._bind(node.name, OBJ, node)
        self.assigned.add(node.name)

    def _for_unpack(self, node: ast.For) -> None:
        """`for a, b in pairs:` -- the target is a tuple of names."""
        self._expr(node.iter)
        names = _target_names(node.target)
        before = set(self.assigned)
        for name in names:
            self._declare(name, OBJ, node)
            self.assigned.add(name)
        self.loop_depth += 1
        self._block(node.body)
        self.loop_depth -= 1
        self.assigned = before
        self._block(node.orelse)
        self.assigned = before

    def _dyn_try(self, node: ast.Try) -> bool:
        """Analyse a try statement. Falls through unless every path returns.

        The definite-assignment set is intersected the way `if` does it, and
        for the same reason -- except that the BODY may have stopped anywhere,
        so nothing it assigned is definitely assigned in a handler.
        """
        before = set(self.assigned)
        body_falls = self._block(node.body)
        after_body = set(self.assigned)

        # THE PATHS THAT REACH PAST THE STATEMENT, each with what it assigned.
        # Intersected at the end, exactly as `if`/`else` does it -- and for
        # the same reason a branch that returns does not dilute an `if`, a
        # HANDLER THAT CANNOT FALL THROUGH does not dilute this:
        #
        #     try:
        #         result = run()
        #     except Error as exc:
        #         raise Wrapped from exc     # never reaches the line below
        #     use(result)                    # `result` IS assigned here
        #
        # Resetting to `before` unconditionally -- which is what this did --
        # rejected that shape, and it is the ordinary way to wrap an
        # exception. A handler ending in `continue`, `break` or `return` is
        # the same case.
        exits: list[set] = []
        if body_falls:
            self.assigned = after_body
            # `else` runs only when the body finished without raising, so it
            # extends that path rather than being one of its own.
            if not node.orelse or self._block(node.orelse):
                exits.append(set(self.assigned))
        for handler in node.handlers:
            # A HANDLER STARTS FROM `before`: the body may have stopped
            # anywhere, so nothing it assigned is definitely assigned here.
            self.assigned = set(before)
            if handler.type is not None:
                for name in _handler_names(handler.type):
                    if name not in _EXC_NAMES and name not in self.exc_classes:
                        self._error("E0062",
                                    f"unknown exception type {name!r}",
                                    handler)
            if handler.name:
                self._declare(handler.name, OBJ, handler)
                self.assigned.add(handler.name)
            if self._block(handler.body):
                exits.append(set(self.assigned))

        # `finally` runs on EVERY path out, so what it assigns is assigned
        # afterwards whichever way control got there.
        after_final: set = set()
        if node.finalbody:
            self.assigned = set(before)
            self._block(node.finalbody)
            after_final = set(self.assigned)

        if exits:
            out = set(exits[0])
            for reached in exits[1:]:
                out &= reached
            self.assigned = out | after_final
            return True
        # Nothing falls through: every path returned, raised or jumped.
        self.assigned = before | after_final
        return False

    def _for_range(self, node: ast.For, name: str) -> None:
        call = node.iter
        if not (isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "range"):
            if self.dynamic:
                # Any sequence, walked by index. Not an iterator protocol --
                # there is no generator yet, and a `for` over a list does not
                # need one.
                self._expr(node.iter)
                self._declare(name, OBJ, node)
                before = set(self.assigned)
                self.assigned.add(name)
                self.loop_depth += 1
                self._block(node.body)
                self.loop_depth -= 1
                self.assigned = before
                self._block(node.orelse)
                self.assigned = before
                return
            self._error("E0023", "only `for ... in range(...)` is supported",
                        node.iter)
            return
        if not 1 <= len(call.args) <= 3:
            self._error("E0024", "range() takes 1 to 3 arguments", call)
        for a in call.args:
            got = self._expr(a)
            if not self.dynamic:
                self._check_assignable(got, INT, a, what="range() argument")
            # In a dynamic function every argument is an object, and whether
            # it holds an int is a runtime question. Lowering unwraps it; a
            # value that is not an int fails there, where CPython fails too.
        if len(call.args) == 3:
            # The step's SIGN decides whether the loop test is `<` or `>`.
            # A runtime step would need both tests and a branch on the sign;
            # accepting one and emitting `<` would make every descending loop
            # run zero times and report success.
            step = int_literal(call.args[2])
            if step is None:
                self._error("E0028", "range() step must be a literal",
                            call.args[2])
            elif step == 0:
                self._error("E0029", "range() step must not be zero",
                            call.args[2])
        self._declare(name, OBJ if self.dynamic else INT, node)
        # `range(0)` runs zero times, so neither the loop variable nor
        # anything the body assigns is definitely assigned afterwards --
        # `for i in range(0): pass` then `print(i)` is an UnboundLocalError
        # in Python too.
        before = set(self.assigned)
        self.assigned.add(name)
        self.loop_depth += 1
        self._block(node.body)
        self.loop_depth -= 1
        self.assigned = before
        # `for ... else` runs the else clause when the loop finished WITHOUT a
        # break -- so it is reachable, and nothing it assigns is definitely
        # assigned afterwards either. Only on the dynamic path: the static
        # lowering has no `else` on a loop and would drop it in silence.
        if node.orelse and not self.dynamic:
            self._error("E0025", "`for ... else` is not supported in an "
                                 "annotated function", node)
        self._block(node.orelse)
        self.assigned = before

    # ── symbols ─────────────────────────────────────────────────────────────
    def _declare(self, name: str, ty: SemType, at) -> Symbol:
        info = self.current
        assert info is not None
        existing = info.locals.get(name)
        if existing is not None:
            if not ty.is_error and not existing.type.is_error and existing.type != ty:
                self.sink.report(
                    error("E0030", f"{name!r} was declared {existing.type} "
                                   f"and is being redeclared {ty}")
                    .at(self._span(at))
                    .also(existing.span, "first declared here")
                    .note("a variable keeps one type for its whole life"))
            return existing
        sym = Symbol(name, ty, self._span(at))
        info.locals[name] = sym
        return sym

    def _bind(self, name: str, ty: SemType, at, value=None) -> None:
        info = self.current
        assert info is not None
        if name in info.module_writes:
            return          # the module owns the storage; nothing local to bind
        existing = info.locals.get(name)
        if existing is None:
            self._declare(name, ty, at)
            return
        self._check_assignable(ty, existing.type, at,
                               what=f"assignment to {name!r}", value=value)

    def _lookup(self, name: str, at) -> SemType:
        info = self.current
        assert info is not None
        sym = info.locals.get(name)
        if sym is not None and name not in self.assigned and info.dynamic:
            # NOT PROVED, so checked at run time -- see `maybe_unbound`. Some
            # correct programs cannot be proved: an interpreter's dispatch
            # loop assigns under one condition and reads under its complement,
            # and nothing short of a solver relates the two.
            info.maybe_unbound.add(name)
        if sym is None:
            if info.dynamic:
                # A NAME NOTHING IN THIS PROGRAM ASSIGNS is a GLOBAL READ, and
                # a global that was never set raises `NameError` when the read
                # runs -- which is CPython's rule, and the difference between
                # refusing a program and running the part of it that works:
                #
                #     gen = list(j for j in range(2))
                #     try:
                #         print(j)            # NameError, and catchable
                #     except NameError:
                #         ...
                #
                # `j` is the generator expression's own, so nothing at module
                # level assigns it -- and CPython still compiles the read.
                # Reporting here refused a program that CPython runs.
                # REGISTERED AS MODULE STORAGE, so the read has somewhere to
                # come from. The cell is zero-initialised and zero is never a
                # value, so reading one nothing assigned is the NameError
                # above rather than a crash -- the same mechanism a global
                # read before its assignment already uses.
                self.module_names.add(name)
                self.current.module_reads.add(name)
                return OBJ
            if name in self.ctypes_libs:
                # A CTYPES LIBRARY IS NOT A VALUE. `libm = ctypes.CDLL("m")`
                # binds a compile-time namespace, exactly as `import block`
                # does for Java: `libm.sqrt(...)` is resolved to an external
                # symbol while compiling and nothing survives to be read.
                #
                # So the name is legal here and would be undefined ANYWHERE
                # else -- a statically typed function cannot see module-level
                # storage, and this is not module-level storage. Without this
                # the one place a native call belongs, where the arguments are
                # already machine scalars, was the one place it was refused.
                return ERROR if self._used_as_value(at) else OBJ
            self.sink.report(
                error("E0031", f"undefined name {name!r}")
                .at(self._span(at))
                .help("assign it before use, or annotate it"))
            return ERROR
        if name not in self.assigned and not info.dynamic:
            # Python raises UnboundLocalError for this at runtime; saying it
            # at compile time is strictly better. Reporting it here also
            # keeps it out of the IR, where the verifier catches the same
            # thing as "reads %1 before any path writes it" and the driver
            # correctly but unhelpfully calls it an internal compiler error.
            self.sink.report(
                error("E0032", f"{name!r} may be used before it is assigned")
                .at(self._span(at))
                .also(sym.span, "assigned here, but not on every path")
                .note("Python raises UnboundLocalError for this at runtime")
                .help("give it a value before the branch, or assign it in "
                      "every branch"))
            # Treated as assigned from here on, so one mistake yields one
            # diagnostic rather than one per use.
            self.assigned.add(name)
        sym.used = True
        return sym.type

    # ── expressions ─────────────────────────────────────────────────────────
    def _expr(self, node, as_condition: bool = False) -> SemType:
        """The type of `node`.

        `as_condition` says only its TRUTH is observed -- it is the test of an
        `if`, a `while`, a conditional expression, or the operand of `not`.
        That matters for `and`, `or` and `a if c else b`, which YIELD AN
        OPERAND in Python: what they answer has the type of whichever one they
        picked, so a static path that gives every expression one type can only
        be right when the operands agree. In a condition it does not have to
        be -- `0` and `0.0` are both false -- so the strictness applies where
        the value can actually be seen and nowhere else.

        DEFAULTS TO FALSE, which is the strict reading. A call site that
        forgets to say "condition" gets a diagnostic on a program that would
        have worked; one that defaulted the other way would let a wrong value
        through, which is the failure worth preventing.
        """
        ty = self._expr_inner(node, as_condition)
        info = self.current
        if info is not None:
            info.expr_types[id(node)] = ty
        return ty

    def _expr_inner(self, node, as_condition: bool = False) -> SemType:
        if getattr(self, "dynamic", False):
            # The dynamic path already answers an operand: every value
            # carries its own type, so there is nothing to unify and
            # nothing to refuse.
            return self._dyn_expr(node)
        match node:
            case ast.Constant(value=bool()):
                return BOOL
            case ast.Constant(value=int()):
                return INT
            case ast.Constant(value=float()):
                return FLOAT
            case ast.Constant(value=None):
                return NONE
            case ast.Name(id=name):
                return self._lookup(name, node)
            case ast.UnaryOp(op=ast.Not()):
                self._expr(node.operand, True)
                return BOOL
            case ast.UnaryOp():
                return self._expr(node.operand)
            case ast.BinOp():
                return self._binop(node)
            case ast.BoolOp():
                # THE OPERANDS INHERIT THE POSITION. `if (a and b) or c:`
                # observes only truth all the way down, so nothing inside it
                # needs to agree about type either.
                types = [self._expr(v, as_condition) for v in node.values]
                return self._yielded(
                    types, node, as_condition,
                    "`and`" if isinstance(node.op, ast.And) else "`or`")
            case ast.Compare():
                left_node, left = node.left, self._expr(node.left)
                for op, c in zip(node.ops, node.comparators):
                    right = self._expr(c)
                    self._machine_compare(op, left_node, left, c, right)
                    left_node, left = c, right
                return BOOL
            case ast.Call():
                return self._call(node)
            case ast.IfExp():
                self._expr(node.test, True)
                return self._yielded(
                    [self._expr(node.body, as_condition),
                     self._expr(node.orelse, as_condition)],
                    node, as_condition, "a conditional expression")
        self._error("E0040",
                    f"unsupported expression: {type(node).__name__}", node)
        return ERROR

    def _dyn_expr(self, node) -> SemType:
        """Check an expression in a dynamic function. Every result is OBJ.

        There is nothing to infer -- the value's kind is decided at run time --
        so this walks for the questions that still have static answers: an
        undefined name, a call to something that does not exist, a call with
        the wrong number of arguments. Everything else the object runtime
        answers, including the type errors, which it raises where CPython does.
        """
        match node:
            case ast.Lambda():
                # Its body was checked with its own scope during collection;
                # the expression itself is a function value.
                return OBJ
            case ast.Attribute(value=ast.Name(id=base)) if (
                    base in _DYN_BUILTINS and base not in self.current.locals):
                # `str.lower` -- an unbound method of a builtin type. Lowering
                # makes it a one-argument callable; there is nothing to check
                # here, because whether the method exists is the receiver's
                # business at run time.
                return OBJ
            case ast.NamedExpr(target=ast.Name(id=name)):
                # PEP 572: binds `name` and evaluates to the same value. The
                # binding is in the enclosing scope, not a scope of its own,
                # which is what makes `while (n := next_one()) > 0:` work.
                self._expr(node.value)
                self._bind(name, OBJ, node)
                self.assigned.add(name)
                return OBJ
            case ast.Constant(value=v) if type(v) in _CONSTANT_KINDS:
                return OBJ
            case ast.Constant(value=v):
                # A literal whose type the runtime has no kind for. Accepting
                # it here and letting lowering assert breaks the project's
                # stated invariant -- a result or a diagnostic, never a
                # traceback -- and it did, for every `b'..'` in the corpus.
                # Whether the kind is missing is decided HERE, where there is
                # a span to point at and a name to say.
                if v is Ellipsis:
                    return OBJ
                self._error("E0080",
                            f"a {type(v).__name__} literal is not supported "
                            f"yet", node)
                return ERROR
            case ast.Yield() | ast.YieldFrom() if self.current.is_generator:
                if node.value is not None:
                    self._expr(node.value)
                return OBJ
            case ast.Await() if self.current.is_coroutine:
                self._expr(node.value)
                return OBJ
            case ast.Await():
                # OUTSIDE an `async def` this is a syntax error in CPython,
                # and reporting it as one here rather than lowering something
                # that cannot suspend is the difference between a diagnostic
                # and a wrong answer.
                self._error("E0086", "'await' outside async function", node)
                return ERROR
            case ast.Slice() if self.dynamic:
                # A SLICE AS A VALUE -- `c[1:2, 3]` puts one in a tuple. Its
                # bounds are ordinary expressions and an omitted one is None.
                for part in (node.lower, node.upper, node.step):
                    if part is not None:
                        self._expr(part)
                return OBJ
            case ast.Name(id=name) if name in _SINGLETON_NAMES:
                return OBJ
            case ast.Attribute(value=ast.Name(id="object"), attr=attr) \
                    if attr in OBJECT_DEFAULTS \
                    and "object" not in self.current.locals:
                # `object.__getattribute__` and friends -- see
                # `OBJECT_DEFAULTS`. Shadowed by a local named `object`, which
                # is why the table is consulted only when nothing else claims
                # the name.
                return OBJ
            case ast.Name(id=name) if (name in _MODULE_DUNDERS
                                       and name not in self.module_names
                                       and name not in self.current.locals):
                # `__name__` and friends. Not assigned by the program, so no
                # storage exists for them; lowering answers each with its
                # constant. Shadowed by a real binding of the same spelling,
                # which is why both tables are checked first.
                return OBJ
            case ast.Name(id=name):
                if (self._function_locals is not None
                        and name not in self._function_locals
                        and name in self.module_names):
                    # A module-level name, read from inside a function.
                    self.current.module_reads.add(name)
                    return OBJ
                if name not in self.current.locals:
                    if name in _NEEDS_A_COMPILER:
                        # AS A VALUE TOO, and not only in call position. Left
                        # to the general path this compiled to a runtime
                        # `name 'eval' is not defined` -- which says the wrong
                        # thing about a name Python does define.
                        self._no_compiler(name, node)
                        return ERROR
                    if name in _EXC_NAMES or name in self.exc_classes:
                        return OBJ
                    if name in self.functions or name in self.class_names:
                        # A `def` or a `class` NAMED but not called. Both are
                        # values now -- a function object and a type object --
                        # so this is an ordinary read, not the refusal it used
                        # to be.
                        #
                        # A STATICALLY TYPED `def` is the exception: its
                        # parameters are machine words and there is no object
                        # to hand out, so it has no value form and no module
                        # storage. Saying so here is what stops lowering
                        # emitting a read of a global that was never made --
                        # which reached the IR verifier as an internal error
                        # on a program the user wrote.
                        if (name in self.functions
                                and not self.functions[name].dynamic):
                            self._error(
                                "E0085",
                                f"{name!r} is a statically typed function and "
                                f"has no value form", node)
                            return ERROR
                        return OBJ
                    if name in _DYN_BUILTINS:
                        # A builtin used as a VALUE -- `key=repr`,
                        # `map(str, xs)`. Lowering synthesises a one-argument
                        # function that calls it, so it becomes a value like
                        # any other. Only the one-argument builtins qualify:
                        # the thunk's SHAPE is what makes it callable, and a
                        # variadic like `print` has no single shape.
                        if name in _VALUE_BUILTINS or name in _CLASS_VALUES:
                            return OBJ
                        self._error("E0056",
                                    f"{name!r} is a builtin that cannot be "
                                    f"used as a value", node)
                        return ERROR
                self._lookup(name, node)
                return OBJ
            case ast.Attribute():
                # ANY attribute, on any value. This used to accept only
                # `type(x).__name__` because neither `type(x)` nor a bound
                # method was a value; both are now, so the general form is
                # what the runtime answers and a missing attribute is an
                # AttributeError where CPython raises one rather than a
                # compile error where CPython has none.
                self._expr(node.value)
                return OBJ
            case ast.Set(elts=elts):
                for e in elts:
                    self._arg_expr(e)
                return OBJ
            case ast.TemplateStr(values=parts):
                # THE SAME CHECKS AS AN F-STRING: the pieces are expressions
                # either way, and what differs is only whether the result is
                # joined. `Interpolation` carries `format_spec` as a nested
                # f-string exactly as `FormattedValue` does.
                for part in parts:
                    if isinstance(part, ast.Constant):
                        continue
                    if part.format_spec is not None:
                        self._expr(part.format_spec)
                    self._expr(part.value)
                return OBJ
            case ast.JoinedStr(values=parts):
                for part in parts:
                    if isinstance(part, ast.Constant):
                        continue
                    if part.format_spec is not None:
                        # A spec is itself an f-string -- `{x:{w}}` nests --
                        # so it is checked as one rather than as text.
                        self._expr(part.format_spec)
                    self._expr(part.value)
                return OBJ
            case ast.List(elts=elts) | ast.Tuple(elts=elts):
                # `_arg_expr`, not `_expr`: a display may contain `*xs`, and
                # the star itself has no type -- what it spreads does.
                for e in elts:
                    self._arg_expr(e)
                return OBJ
            case ast.GeneratorExp():
                # A SCOPE OF ITS OWN, registered alongside the lambdas -- so
                # only the outermost iterable is typed here, the rest being
                # inside the synthetic body. See `genexp_def`.
                self._expr(node.generators[0].iter)
                return OBJ
            case ast.ListComp() | ast.SetComp():
                return self._dyn_comprehension(node, [node.elt])
            case ast.DictComp():
                return self._dyn_comprehension(node, [node.key, node.value])
            case ast.Dict(keys=keys, values=values):
                for k, v in zip(keys, values):
                    if k is None:
                        # `{**other}` -- the value is a MAPPING spread into
                        # this one, and there is no key to check.
                        self._expr(v)
                        continue
                    self._expr(k)
                    self._expr(v)
                return OBJ
            case ast.Subscript():
                self._expr(node.value)
                if isinstance(node.slice, ast.Slice):
                    for part in (node.slice.lower, node.slice.upper,
                                 node.slice.step):
                        if part is not None:
                            self._expr(part)
                    return OBJ
                self._expr(node.slice)
                return OBJ
            case ast.Call():
                return self._dyn_call(node)
            case ast.BinOp():
                self._expr(node.left); self._expr(node.right)
                if not isinstance(node.op, self._DYN_BINOPS):
                    self.sink.report(
                        error("E0045",
                              f"operator {_op_symbol(node.op)} is not supported")
                        .at(self._span(node)))
                return OBJ
            case ast.UnaryOp():
                self._expr(node.operand)
                return OBJ
            case ast.BoolOp():
                for v in node.values:
                    self._expr(v)
                return OBJ
            case ast.Compare():
                self._expr(node.left)
                for c in node.comparators:
                    self._expr(c)
                return OBJ
            case ast.IfExp():
                self._expr(node.test); self._expr(node.body)
                self._expr(node.orelse)
                return OBJ
        self._error("E0040",
                    f"unsupported expression: {type(node).__name__}", node)
        return ERROR

    def _dyn_comprehension(self, node, results: list) -> SemType:
        """A comprehension binds its loop variables in its OWN scope.

        Python gives a comprehension a scope of its own, so `[i for i in xs]`
        does not leave `i` behind. This frontend has one flat scope per
        function, so the names ARE visible afterwards -- a stated divergence.
        What it must not do is reject the comprehension for using a name that
        the enclosing code never assigned, which is why the targets are
        declared here before the element expression is checked.
        """
        for gen in node.generators:
            if gen.is_async and not self.current.is_coroutine:
                # Outside a coroutine there is nothing to suspend, which is
                # what `async for` in a comprehension needs. CPython calls
                # this a syntax error for the same reason.
                self._error("E0064", "an async comprehension outside an "
                                     "async function", node)
            self._expr(gen.iter)
            for name in _target_names(gen.target):
                self._declare(name, OBJ, node)
                self.assigned.add(name)
            for cond in gen.ifs:
                self._expr(cond)
        for r in results:
            self._expr(r)
        return OBJ

    def _value_pattern_names(self, pat) -> None:
        """Check the parts of a pattern that are READ rather than bound.

        `case Color.RED` compares against a value and `case Point(x, y)` names
        a class -- both are ordinary expressions, and an undefined name in one
        is the same mistake as anywhere else. Walked separately from the
        binding names because the two are interleaved and only position tells
        them apart.
        """
        if pat is None:
            return
        if isinstance(pat, ast.MatchValue):
            self._expr(pat.value)
        elif isinstance(pat, ast.MatchClass):
            self._expr(pat.cls)
            for sub in list(pat.patterns) + list(pat.kwd_patterns):
                self._value_pattern_names(sub)
        elif isinstance(pat, ast.MatchMapping):
            for key in pat.keys:
                self._expr(key)
            for sub in pat.patterns:
                self._value_pattern_names(sub)
        elif isinstance(pat, (ast.MatchSequence, ast.MatchOr)):
            for sub in pat.patterns:
                self._value_pattern_names(sub)
        elif isinstance(pat, ast.MatchAs):
            self._value_pattern_names(pat.pattern)

    def _arg_expr(self, arg) -> SemType:
        """Type one call argument, seeing through `*xs`.

        The star itself has no type; what it spreads does, and typing that is
        what catches `f(*7)`. Every place a call's arguments are walked goes
        through here, because a missed one reaches lowering as a bare
        `ast.Starred` and asserts.
        """
        if isinstance(arg, ast.Starred):
            return self._expr(arg.value)
        return self._expr(arg)

    def _dyn_call(self, node: ast.Call) -> SemType:
        # A NATIVE CALL FROM DYNAMIC CODE. `cffi.py` resolves a ctypes symbol
        # while compiling and the STATIC path has always lowered one; this
        # path never looked, so `libm.sqrt(x)` inside an untyped function
        # lowered as an ordinary attribute access on `libm` -- a name the
        # splice removes -- and raised `NameError: name 'libm' is not
        # defined` at run time, about a library the source plainly declares.
        #
        # THAT MATTERS BEYOND THE ANNOYANCE: every BUNDLED module is dynamic
        # Python, so the standard library could not reach a C library at all.
        # See docs/STDLIB.md on why that is what stands between here and a
        # concrete `pathlib`.
        if self._is_ctypes_call(node):
            return self._dyn_ctypes_call(node)
        if isinstance(node.func, ast.Name) and node.func.id == "locals"                 and "locals" not in self.current.locals:
            self.current.reads_all_locals = True
        for kw in node.keywords:
            self._expr(kw.value)
        if isinstance(node.func, ast.Attribute)                 and isinstance(node.func.value, ast.Name)                 and node.func.value.id in _BUILTIN_TYPE_NAMES                 and node.func.attr in _TYPE_STATIC_NAMES                 and node.func.value.id not in self.current.locals:
            # `dict.fromkeys(...)` -- a constructor on the TYPE. The receiver
            # is not typed because there is none; see `_TYPE_STATICS`.
            for a in node.args:
                self._arg_expr(a)
            return OBJ
        if isinstance(node.func, ast.Attribute)                 and isinstance(node.func.value, ast.Name)                 and node.func.value.id == "object"                 and node.func.attr in OBJECT_DEFAULTS                 and "object" not in self.current.locals:
            # `object.__getattribute__(self, name)` -- a DEFAULT, resolved by
            # name. The receiver is not typed because there is no `object`
            # value to type; see `OBJECT_DEFAULTS`.
            for a in node.args:
                self._arg_expr(a)
            return OBJ
        if isinstance(node.func, ast.Attribute):
            # A method call. A name in the table lowers to the runtime
            # operation it names; anything else is looked up on the receiver
            # and called, which is what a user class needs and what makes
            # `"abc".nosuch()` an AttributeError at run time -- where CPython
            # raises it -- instead of a compile error CPython does not have.
            self._expr(node.func.value)
            for a in node.args:
                self._arg_expr(a)
            return OBJ
        if not isinstance(node.func, ast.Name):
            # A call on something computed: `fs[0](x)`, `obj.f()(y)`. The
            # callee is a value, so this is `apy_call` like any other.
            self._expr(node.func)
            for a in node.args:
                self._arg_expr(a)
            return OBJ
        name = node.func.id
        if name == "super":
            # `super(C, self)` IS THE EXPLICIT FORM of the same thing: the
            # class to start past, and the receiver to bind to. The
            # no-argument spelling is sugar for exactly this pair, so both go
            # to one runtime call and only where the two come from differs.
            if len(node.args) == 2:
                for a in node.args:
                    self._arg_expr(a)
            elif node.args:
                self._error("E0078", "`super()` takes no arguments or two",
                            node)
            elif self.current.owner is None:
                self._error("E0079", "`super()` outside a method", node)
            return OBJ
        for a in node.args[:1] if name == "isinstance" else node.args:
            self._arg_expr(a)
        if name in _EXC_NAMES or name in self.exc_classes:
            # A GROUP TAKES TWO: the message and the exceptions it carries.
            # Every other exception takes at most one argument, which is what
            # makes the general rule worth stating -- and why the exception to
            # it is named rather than left to a count.
            if name in ("ExceptionGroup", "BaseExceptionGroup"):
                if len(node.args) != 2:
                    self._error("E0061", f"{name} takes the message and the "
                                         f"exceptions it carries", node)
                return OBJ
            # EVERY ARGUMENT IS KEPT. `e.args` is a tuple of all of them and
            # `OSError(errno, strerror)` reads two back by name, so refusing
            # more than one rejected programs CPython accepts.
            return OBJ
        if name == "isinstance":
            # THE SECOND ARGUMENT IS AN EXPRESSION, and usually a dotted one:
            # `isinstance(node, ast.Name)` is what every AST walk is made of.
            # A tuple of them is the other legal form and means "any of
            # these".
            #
            # Only a BARE BUILTIN NAME is special. There is no `int` value to
            # compare against -- the builtin types are kinds, not objects --
            # so that one name travels as text and everything else travels as
            # the class it evaluates to. See `_dyn_type_name`.
            second = node.args[1] if len(node.args) == 2 else None
            for element in (second.elts
                            if isinstance(second, (ast.Tuple, ast.List))
                            else [second] if second is not None else []):
                if isinstance(element, ast.Name)                         and element.id not in self.current.locals                         and element.id not in self.class_names:
                    continue      # a builtin or exception name: travels as text
                self._expr(element)
            return OBJ
        if name in _DYN_BUILTINS:
            want = _DYN_BUILTINS[name]
            starred = any(isinstance(a, ast.Starred) for a in node.args)
            allowed = _BUILTIN_KEYWORDS.get(name)
            if allowed is not None:
                # These are keywords, not positional, so the positional count
                # is unaffected by them.
                #
                # A `**` SPLAT NAMES NOTHING HERE. `sorted(xs, **opts)` has
                # one keyword whose `arg` is None, and checking that against
                # the allowed list refused a call CPython accepts, naming
                # `None` as the offending keyword. Which names the dict holds
                # is a run-time question, and the lowering reads them out of
                # it -- see `dynamic._KEYWORD_THUNKS`.
                for kw in node.keywords:
                    if kw.arg is not None and kw.arg not in allowed:
                        self._error("E0068",
                                    f"{name}() got an unexpected keyword "
                                    f"argument {kw.arg!r}", node)
            if want is not None and not starred and len(node.args) != want:
                self._error("E0054",
                            f"{name}() takes exactly {want} argument(s), "
                            f"got {len(node.args)}", node)
            return OBJ
        if name in HOSTSVC and name not in self.functions \
                and name not in self.current.locals:
            # A HOST SERVICE FROM DYNAMIC CODE, which is the shape every
            # BUNDLED module has. The static path admits these through
            # `_hostsvc_call`; without this the same call from an untyped
            # function was `call to unknown function 'host_file_kind'`, and
            # the standard library is entirely untyped -- so `pathlib` could
            # not reach the layer built for it. Exactly the gap `ctypes` had
            # before `_dyn_ctypes_call`, and fixed the same way.
            want = len(HOSTSVC[name][0])
            if not any(isinstance(a, ast.Starred) for a in node.args) \
                    and len(node.args) != want:
                self._error("E0054",
                            f"{name}() takes exactly {want} argument(s), "
                            f"got {len(node.args)}", node)
            for a in node.args:
                self._arg_expr(a)
            # OBJ, NOT INT, even though every host service answers `i64`.
            # `_dyn_hostsvc_call` boxes the result with `apy_from_int`, so
            # what the surrounding dynamic code receives is an object like
            # every other value it handles -- and claiming INT here made the
            # analyser reject the assignment that stores it.
            return OBJ
        if name in _OBJRT_DYNAMIC_NAMES and name not in self.functions \
                and name not in self.current.locals:
            # THE SAME GAP `HOSTSVC` HAS JUST ABOVE, for the small,
            # explicit set of object-runtime primitives a bundled dynamic
            # module is allowed to call by bare name -- see
            # `_OBJRT_DYNAMIC_NAMES`'s own comment. Every one of them
            # takes and returns `ptr`, which is already an ordinary
            # dynamic value's own representation, so unlike `HOSTSVC`
            # there is no boxing/unboxing question here at all.
            want = len(OBJECT_RUNTIME[name][0])
            if not any(isinstance(a, ast.Starred) for a in node.args) \
                    and len(node.args) != want:
                self._error("E0054",
                            f"{name}() takes exactly {want} argument(s), "
                            f"got {len(node.args)}", node)
            for a in node.args:
                self._arg_expr(a)
            return OBJ
        info = self.functions.get(name)
        if info is None:
            # Not a module-level `def`. It may still be a callable VALUE -- a
            # class, a nested function, a parameter holding one -- in which
            # case this is `apy_call` and the arity is the callee's business
            # at run time. Only a name that resolves to nothing is an error.
            if name in self.class_names or name in self.current.locals \
                    or (self._function_locals is not None
                        and name not in self._function_locals
                        and name in self.module_names):
                self._expr(node.func)
                return OBJ
            if name in _NEEDS_A_COMPILER:
                self._no_compiler(name, node)
                return ERROR
            self.sink.report(
                error("E0052", f"call to unknown function {name!r}")
                .at(self._span(node.func))
                .note("known functions: "
                      + (", ".join(sorted(n for n in self.functions
                                          if n != ENTRY_NAME)) or "(none)")))
            return ERROR
        self._check_arity(node, name, info)
        return OBJ

    def _check_arity(self, node: ast.Call, name: str,
                     info: FunctionInfo) -> None:
        """Positional count, keyword names, and what the defaults cover.

        A STARRED argument makes the count unknowable here -- it is whatever
        the spread sequence turns out to hold -- so every count check is
        skipped and the runtime reports a mismatch instead. Skipping is not
        accepting: `f(*xs)` against a two-parameter `f` is still an error,
        just one nobody can see until `xs` exists.
        """
        if any(isinstance(a, ast.Starred) for a in node.args):
            return
        if name in self.rebound:
            # A NAME THE MODULE DEFINES TWICE has no single signature to
            # check against. `info` is the LAST definition, and a call
            # written between the two means the FIRST -- so checking it here
            # reported `f() got an unexpected keyword argument 'x'` about a
            # call whose `x` is exactly what the definition above it names.
            # A compile error for correct Python, which is worse than the
            # wrong answer this check exists to prevent.
            #
            # THE RUNTIME ANSWERS IT INSTEAD, and it is already arranged to:
            # `dynamic._dyn_call`'s `by_value` routes every call to a rebound
            # name through the VALUE, which is whichever function has been
            # bound by the time the call runs. A real mismatch is reported
            # there, which is where CPython reports this shape too.
            return
        if info.kwarg is not None or any(kw.arg is None
                                         for kw in node.keywords):
            # `**kw` accepts any keyword and `f(**d)` supplies names that do
            # not exist yet, so neither the count nor the names can be checked
            # here. The runtime reports a real mismatch, which is where
            # CPython reports this shape too.
            return
        if getattr(info.node, "decorator_list", ()):
            # A DECORATED function's signature is the WRAPPER's, and that is
            # decided at run time -- `@tag('hi') def g(x)` is called as `g(3)`
            # through a `wrapper(*a)` that takes anything. Checking the `def`'s
            # own parameters here rejected calls the program makes happily.
            return
        params = [p.name for p in info.params]
        # A KEYWORD-ONLY PARAMETER CANNOT BE FILLED BY POSITION. Every
        # parameter was treated as reachable positionally, so
        # `def b(x, *args, c=3)` called `b(1, 2, c=9)` was rejected as two
        # values for `c` -- the `2` was counted as having filled it. The
        # runtime already models the split (see `apy_func_kwonly`); only this
        # check did not, which is why the error was a refusal rather than a
        # wrong answer.
        positional = params[:len(params) - info.kwonly] if info.kwonly             else params
        required = len(params) - len(info.defaults)
        given = list(node.args)
        by_name = {kw.arg: kw for kw in node.keywords if kw.arg}
        # A POSITIONAL-ONLY PARAMETER IS NOT REACHABLE BY NAME, and this
        # check did not know that: `/` was never consulted, so a keyword
        # naming one read either as a slot already filled by position
        # (`multiple values`) or, when it was some OTHER name that did not
        # match, as an unexpected keyword. Both are the wrong complaint about
        # the wrong argument.
        #
        # GATHERED, IN DECLARATION ORDER, AND ALL AT ONCE, which is the rule
        # CPython follows and the one the runtime binder was taught: `pos(a=1,
        # zz=2)` against `def pos(a, b, /, c)` names `a` and never mentions
        # `zz`, because the positional-only complaint outranks the unknown
        # name.
        #
        # THE `**kw` CASE NEVER REACHES HERE. A callee with one takes the name
        # into the collection rather than refusing it, and this whole check
        # returned above for that.
        hit = [p for p in params[:info.posonly] if p in by_name]
        if hit:
            self._call_keyword_fault(
                node, info, name, "E0069",
                f"{name}() got some positional-only arguments passed as "
                f"keyword arguments: '{', '.join(hit)}'")
            return
        for kw in by_name:
            if kw not in params:
                self._call_keyword_fault(
                    node, info, name, "E0068",
                    f"{name}() got an unexpected keyword argument {kw!r}")
                return
            if kw in positional and positional.index(kw) < len(given):
                self._call_keyword_fault(
                    node, info, name, "E0069",
                    f"{name}() got multiple values for argument {kw!r}")
                return
        if info.vararg is None and len(given) > len(positional):
            self._arity_mismatch(
                node, info,
                f"{name}() takes at most {len(positional)} "
                f"positional argument(s), got {len(given)}", name)
            return
        filled = set(positional[:len(given)]) | set(by_name)
        missing = [p for p in params[:required] if p not in filled]
        if missing:
            self._arity_mismatch(
                node, info,
                f"{name}() missing {len(missing)} required argument(s): "
                + ", ".join(repr(m) for m in missing), name)

    def _call_keyword_fault(self, node: ast.Call, info: FunctionInfo,
                            name: str, code: str, message: str) -> None:
        """A keyword this call gets wrong.

        A REFUSAL FOR A STATIC FUNCTION and a RUNTIME RAISE for a dynamic one
        -- the split `_arity_mismatch` already draws, for the same reason and
        now applied to the keywords too. Python's answer is a TypeError a
        program may CATCH, and a module whose wrong call sits on a branch
        nothing takes compiles under CPython; while this was an error it did
        not compile here at all.

        THE CODE IS KEPT for the static half, because `got an unexpected
        keyword argument` and `got multiple values` are different mistakes
        and a reader of the refusal wants to know which. The dynamic half is
        the same warning an arity mismatch raises, since what it says is the
        same thing: this is reported when the line runs.
        """
        if not info.dynamic:
            self.sink.report(
                error(code, message)
                .at(self._span(node))
                .also(self._span_of_def(info), f"{name} is defined here"))
            return
        self.sink.report(
            warning("W0053", message + " -- raised at run time")
            .at(self._span(node))
            .also(self._span_of_def(info), f"{name} is defined here"))
        # LOWERING HAS TO KNOW -- see `_arity_mismatch`. The direct call path
        # hands the arguments straight to the symbol, which has no slot for a
        # keyword the callee does not declare; routing the call through the
        # value path is what lets the runtime report it.
        self.late_arity.add(id(node))

    def _arity_mismatch(self, node: ast.Call, info: FunctionInfo,
                        message: str, name: str) -> None:
        """A call whose argument count cannot be right.

        A REFUSAL FOR A STATIC FUNCTION and a RUNTIME RAISE for a dynamic one.
        The two differ because Python's answer is a TypeError, which a program
        may catch -- and one that does could not be compiled at all while this
        was a compile error. A statically typed function has a fixed machine
        signature and genuinely cannot be called wrongly, so there the refusal
        is the only thing that can be produced.
        """
        if not info.dynamic:
            self.sink.report(
                error("E0053", message)
                .at(self._span(node))
                .also(self._span_of_def(info), f"{name} is defined here"))
            return
        self.sink.report(
            warning("W0053", message + " -- raised at run time")
            .at(self._span(node))
            .also(self._span_of_def(info), f"{name} is defined here"))
        # LOWERING HAS TO KNOW. The direct call path passes the arguments
        # straight to the symbol, and a count the callee cannot accept is a C
        # compile error rather than a Python one; routing this call through
        # the value path is what lets the runtime report it.
        self.late_arity.add(id(node))

    #: Every binary operator this frontend lowers. Checked explicitly, because
    #: the failure mode of NOT checking is a traceback rather than an error:
    #: `@` and `**` both type-checked as ordinary arithmetic here and then hit
    #: a lowering table that had never heard of them. An operator missing from
    #: this set is refused by the type checker, which is where a user can see
    #: it.
    #: SHARED BY BOTH PATHS, which is why `@` is not in it. `a @ b` has no
    #: meaning for a machine word -- it reaches `__matmul__` or nothing -- so
    #: on the static path it stays unsupported, and putting it here made a
    #: statically typed `1 @ 2` reach a lowering with no entry for it and
    #: raise KeyError instead of reporting.
    _LOWERABLE_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                         ast.Mod, ast.Pow, ast.BitAnd, ast.BitOr, ast.BitXor,
                         ast.LShift, ast.RShift)

    #: The dynamic path additionally has `@`, which is dunder dispatch and
    #: nothing else.
    _DYN_BINOPS = _LOWERABLE_BINOPS + (ast.MatMult,)

    def _binop(self, node: ast.BinOp) -> SemType:
        left, right = self._expr(node.left), self._expr(node.right)
        if not isinstance(node.op, self._LOWERABLE_BINOPS):
            self.sink.report(
                error("E0045",
                      f"operator {_op_symbol(node.op)} is not supported")
                .at(self._span(node)))
            return ERROR
        if left.is_error or right.is_error:
            return ERROR
        if not (left.is_numeric and right.is_numeric):
            self.sink.report(
                error("E0041", f"cannot apply {_op_symbol(node.op)} to "
                               f"{left} and {right}")
                .at(self._span(node.op) if hasattr(node.op, "lineno")
                    else self._span(node), _op_symbol(node.op))
                .also(self._span(node.left), str(left))
                .also(self._span(node.right), str(right)))
            return ERROR
        if left.is_machine or right.is_machine:
            return self._machine_binop(node, left, right)
        if isinstance(node.op, ast.Div):
            return FLOAT          # Python's `/` is always float
        if isinstance(node.op, ast.Pow):
            return self._pow(node, left, right)
        if isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor,
                                ast.LShift, ast.RShift)):
            if FLOAT in (left, right):
                self.sink.report(
                    error("E0042", f"{_op_symbol(node.op)} requires integers")
                    .at(self._span(node))
                    .note("floats have no bitwise representation here"))
                return ERROR
            return INT
        return self._unify_all([left, right], node)

    def _pow(self, node: ast.BinOp, left: SemType, right: SemType) -> SemType:
        """`**`, restricted to a non-negative integer literal exponent.

        Python's `**` is not one operation. `2 ** 10` is an int, `2 ** -1` is
        the float 0.5, and `2.0 ** x` is a libm call. A statically typed
        subset cannot give one expression two types, so accepting a runtime
        exponent would mean picking one and being silently wrong about the
        other -- `2 ** n` yielding 0 for negative n is exactly the kind of
        plausible wrong answer that never gets reported as a bug.

        A literal exponent has none of that ambiguity: the result type is the
        base's, and lowering expands it to multiplications with no loop and no
        runtime at all. Anything else is refused with the expression that does
        work.
        """
        exponent = node.right
        value = int_literal(exponent)
        if value is None:
            self.sink.report(
                error("E0043", "`**` needs a literal integer exponent")
                .at(self._span(node.right), "not a literal")
                .note("`x ** n` for a runtime n is float-valued in Python "
                      "when n may be negative, and this subset gives every "
                      "expression one static type")
                .help("for a square write `x * x`; for a float power call "
                      "into a runtime yourself"))
            return ERROR
        if value < 0:
            self.sink.report(
                error("E0044", "`**` with a negative exponent is float-valued")
                .at(self._span(node.right), f"{value}")
                .help(f"write `1.0 / ({ast.unparse(node.left)} "
                      f"** {-value})`"))
            return ERROR
        return FLOAT if left is FLOAT else INT

    # ── the machine subset: arithmetic ──────────────────────────────────────
    #: The bitwise operators, which machine floats have no answer for.
    _BITWISE = (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)

    def _machine_binop(self, node: ast.BinOp, left: SemType,
                       right: SemType) -> SemType:
        """`u32 + u32`, and every way that can go wrong.

        THE RULE IS THAT WIDTHS DO NOT CONVERT THEMSELVES. Python's numeric
        tower widens `bool` to `int` to `float` because none of those loses
        anything; `i64` to `i32` loses half the value, and the whole reason to
        write a width down is that something else is reading the same bytes.
        An implicit conversion here would be a silent disagreement about a
        struct layout, which is the single worst bug this subset can have.

        A LITERAL IS DIFFERENT and adapts to the other side, because `n + 1`
        should not have to be written `n + u32(1)` -- the literal has no width
        of its own to lose, and one that does not fit is refused rather than
        wrapped.
        """
        if left.is_machine and not right.is_machine:
            if self._adapt_literal(node.right, left):
                right = left
        elif right.is_machine and not left.is_machine:
            if self._adapt_literal(node.left, right):
                left = right
        if left.is_error or right.is_error:
            return ERROR
        if left != right:
            self._width_mismatch(node.op, node.left, left, node.right, right)
            return ERROR

        if isinstance(node.op, ast.Div):
            # `/` ON AN INTEGER WIDTH IS REFUSED. Python's `/` is float-valued,
            # so `a / b` on two i32s would have to produce a float and lose the
            # width the author asked for -- and `//` is what the operation they
            # meant is actually called.
            if left.is_machine_int:
                self.sink.report(
                    error("E0014", f"`/` is float-valued, and {left} is an "
                                   f"integer width")
                    .at(self._span(node))
                    .help("write `//` for integer division, or convert both "
                          "sides with f64(...)"))
                return ERROR
            return left
        if isinstance(node.op, ast.Pow):
            # REFUSED RATHER THAN GUESSED. `**` on Python's `int` expands to
            # multiplications and on `float` calls a runtime; neither answer is
            # obviously right for a fixed width, and nothing that needs a width
            # needs `**`. Refusing costs nothing and cannot be silently wrong.
            self.sink.report(
                error("E0015", f"`**` is not defined for {left}")
                .at(self._span(node))
                .help("write the multiplications out, or compute in float and "
                      f"convert back with {left}(...)"))
            return ERROR
        if isinstance(node.op, self._BITWISE):
            if left.is_machine_float:
                self.sink.report(
                    error("E0042", f"{_op_symbol(node.op)} requires integers")
                    .at(self._span(node))
                    .note("floats have no bitwise representation here"))
                return ERROR
            return left
        return left

    def _adapt_literal(self, node, want: SemType) -> bool:
        """Retype an untyped numeric LITERAL to a machine type, if it fits.

        Returns True if the literal was taken care of -- either retyped, or
        refused with a diagnostic already reported. False means this was not a
        literal at all and the caller should report its own mismatch.

        The retyping is a write into `expr_types`, which is what lowering
        reads: analysis and lowering walk the same AST objects, so changing
        this node's recorded type is how a `5` becomes a `u8` five.

        THE RANGE IS CHECKED HERE. `x: u8 = 300` silently becoming 44 is the
        exact failure that makes width annotations worse than no annotations.
        """
        if not want.is_machine or want.is_ptr or self.current is None:
            return False
        if want.is_machine_float:
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, (int, float))
                    and not isinstance(node.value, bool)):
                return False
            self._retype(node, want)
            return True
        value = const_int(node, self.functions)
        if value is None:
            return False
        bits = T.ALL[want.name].bits
        lo, hi = ((-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
                  if want.name[0] == "i" else (0, (1 << bits) - 1))
        if not lo <= value <= hi:
            self.sink.report(
                error("E0012", f"{value} does not fit in {want}")
                .at(self._span(node), f"{want} holds {lo} to {hi}")
                .help("widen the type, or wrap the value explicitly with "
                      f"{want}(...) if truncation is what you mean"))
            return True
        self._retype(node, want)
        return True

    def _machine_compare(self, op, left_node, left: SemType,
                         right_node, right: SemType) -> None:
        """Both sides of ONE `<` must share a width, exactly as `+` does.

        Checked per pair rather than over the chain: `a < b < c` is two
        comparisons and only the adjacent operands ever meet.

        The same rule and the same diagnostic as `_machine_binop`: comparing a
        signed width against an unsigned one has no answer that is right for
        both, which is the same fact as not being able to add them.
        """
        if not (left.is_machine or right.is_machine):
            return
        if left.is_machine and not right.is_machine:
            if self._adapt_literal(right_node, left):
                right = left
        elif right.is_machine and not left.is_machine:
            if self._adapt_literal(left_node, right):
                left = right
        if left.is_error or right.is_error or left == right:
            return
        self._width_mismatch(op, left_node, left, right_node, right)

    def _width_mismatch(self, op, left_node, left: SemType,
                        right_node, right: SemType) -> None:
        """The one diagnostic for "these two machine types had to match"."""
        self.sink.report(
            error("E0013", f"cannot apply {_op_symbol(op)} to "
                           f"{left} and {right}")
            .at(self._span(left_node), str(left))
            .also(self._span(right_node), str(right))
            .note("machine types never convert implicitly -- a change of "
                  "width has to be visible at the point it happens")
            .help(f"convert one side with {left}(...) or {right}(...)"))

    def _machine_conversion(self, name: str, node: ast.Call) -> SemType:
        """`i32(x)`, `f64(x)`, `ptr(x)` -- spelled exactly like `int(x)`.

        This is the ONLY way a value changes width, which is what makes the
        rule in `_machine_binop` enforceable: every conversion in a program
        that uses these types is a thing you can grep for.
        """
        want = MACHINE[name]
        if len(node.args) != 1:
            self._error("E0054", f"{name}() takes exactly one argument, "
                                 f"got {len(node.args)}", node)
            for a in node.args:
                self._expr(a)
            return want
        # A LITERAL ARGUMENT TAKES THE CAST'S WIDTH, which is the whole point
        # of writing the cast: `u64(0x8000000000000000)` means that bit
        # pattern, and typing the literal on its own made it an i64 first --
        # so the value wrapped to INT64_MIN before the cast ever saw it, and
        # the range check that exists for exactly this never ran.
        #
        # A FLOAT TARGET STILL CHECKS AGAINST i64, because the subset has no
        # float literal: `f64(n)` converts FROM an integer, so `n` has to be
        # one that exists. `f64(9223372036854775808)` looked like a way to
        # write 2**63 and silently wrote INT64_MIN instead -- which made
        # `apy_trunc_of`'s guard true for every value, and `(1.5).is_integer()`
        # answer True.
        got = self._expr(node.args[0])
        # AFTER `_expr`, WHICH WRITES THE TYPE IT COMPUTED. Retyping before it
        # is overwritten by it, and the literal goes back to the default
        # width -- which is the whole thing this is here to prevent.
        if int_literal(node.args[0]) is not None:
            if want.is_machine_int:
                self._fit_literal(node.args[0], want)
                got = want
            elif want.name in ("f32", "f64"):
                self._fit_literal(node.args[0], MACHINE["i64"])
        if got.is_error:
            return want
        if want.is_ptr:
            # An address arrives as an integer -- from an allocator, or from a
            # platform call. Nothing else can become one: `ptr(1.5)` and
            # `ptr(some_ptr)` are both mistakes rather than conversions.
            if not (got is INT or got.is_machine_int):
                self.sink.report(
                    error("E0033", f"cannot make a ptr from {got}")
                    .at(self._span(node.args[0]), str(got))
                    .note("a pointer comes from an integer address, and "
                          "nothing else here is one"))
                return ERROR
            return want
        if got.is_ptr:
            # AND BACK. Only at the pointer's own width: narrowing an address
            # to 32 bits is a real thing to want on a 32-bit target and a
            # silent disaster on this one, so it has to be written as two
            # steps.
            if not (want.is_machine_int and T.ALL[want.name].bits == 64):
                self.sink.report(
                    error("E0034", f"cannot convert a ptr to {want}")
                    .at(self._span(node.args[0]), "ptr")
                    .note("a pointer is 64 bits wide")
                    .help("convert to i64 or u64 first, then narrow"))
                return ERROR
            return want
        if not got.is_numeric:
            self.sink.report(
                error("E0055", f"cannot convert {got} to {want}")
                .at(self._span(node.args[0]), str(got)))
            return ERROR
        return want

    def _fit_literal(self, node, want: SemType) -> None:
        """Report an integer literal that does not fit `want`. Nothing else.

        SEPARATE FROM `_retype` because a cast does not change the argument's
        TYPE -- `u64(x)` converts a value of whatever width `x` already has --
        but a LITERAL has no width of its own until something gives it one,
        and the cast is the only thing here that ever will.
        """
        value = int_literal(node)
        if value is None or not want.is_machine_int:
            return
        bits = T.ALL[want.name].bits
        lo, hi = ((-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
                  if want.name[0] == "i" else (0, (1 << bits) - 1))
        if lo <= value <= hi:
            # AND IT TAKES THE WIDTH, which is the half that matters: checking
            # the range and then leaving the literal an i64 lets the value
            # wrap on its way to the cast, which is the bug this exists to
            # stop rather than merely to report.
            self._retype(node, want)
            return
        self.sink.report(
            error("E0012", f"{value} does not fit in {want}")
            .at(self._span(node), f"{want} holds {lo} to {hi}")
            .help("build it from parts that do fit -- 2**63 is "
                  "`f64(4611686018427387904) * f64(2)`"))

    def _memory_intrinsic(self, name: str, node: ast.Call) -> SemType:
        """`alloca`, `load`, `store`, `offset`, `sizeof`.

        Not runtime calls: each is one IR instruction, and giving them names
        instead of syntax is what keeps the systems subset from needing new
        grammar. `load` and `store` take the TYPE FIRST, spelled the way the
        IR spells it and the way LLVM writes the same instruction -- so the
        width being read is the first thing on the line rather than something
        you infer from the variable it lands in.
        """
        want = MEMORY_INTRINSICS[name]
        if want < 0:
            # VARIADIC. `callptr` is the only one, and its floor is the type
            # and the address; everything after is the callee's business.
            if len(node.args) < 2:
                self._error("E0054", f"{name}() takes a type, an address and "
                                     f"then the arguments, got "
                                     f"{len(node.args)}", node)
                for a in node.args:
                    self._expr(a)
                return ERROR
            return self._callptr(node)
        if len(node.args) != want:
            self._error("E0054", f"{name}() takes exactly {want} argument(s), "
                                 f"got {len(node.args)}", node)
            for a in node.args:
                self._expr(a)
            return _INTRINSIC_RESULT[name] or ERROR

        if name == "sizeof":
            self._type_arg(node.args[0], name)
            return INT
        if name == "funcaddr":
            return self._funcaddr(node)
        if name == "rodata":
            return self._rodata(node)
        if name == "reserve":
            return self._reserve(node)
        if name == "alloca":
            # A COMPILE-TIME CONSTANT, because `Op.ALLOCA` carries its size in
            # an immediate: frame storage is laid out before the function runs
            # and a backend has nowhere to put a size it only learns later.
            # A runtime-sized allocation is the allocator's job, not the
            # frame's.
            self._expr(node.args[0])
            size = const_int(node.args[0], self.functions)
            if size is None or size <= 0:
                self.sink.report(
                    error("E0018", "alloca() needs a positive size the "
                                   "compiler can work out")
                    .at(self._span(node.args[0]),
                        "not a constant" if size is None else str(size))
                    .note("frame storage is sized before the function runs, "
                          "so the size cannot be computed")
                    .help("for a runtime size, call an allocator"))
                return PTR
            return PTR
        if name == "load":
            ty = self._type_arg(node.args[0], name)
            self._want_ptr(node.args[1], name, "address")
            return ty
        if name == "store":
            ty = self._type_arg(node.args[0], name)
            got = self._expr(node.args[1])
            if not ty.is_error:
                self._check_assignable(got, ty, node.args[1],
                                       what="value stored")
            self._want_ptr(node.args[2], name, "address")
            return NONE
        # offset
        self._want_ptr(node.args[0], name, "base")
        got = self._expr(node.args[1])
        if not (got.is_error or got is INT or got.is_machine_int):
            self._bad_argument(
                node.args[1], name, "byte count", got, "an integer",
                "there is no element type here -- an index is multiplied by "
                "the element size where it is written")
        return PTR

    def _runtime_call(self, name: str, node: ast.Call) -> SemType:
        """A call into the `apy_*` object runtime. See `_object_runtime`."""
        return self._declared_call(name, OBJECT_RUNTIME[name], node,
                                   "it is part of the object runtime -- see "
                                   "objects/csource.py for what it does")

    def _platform_call(self, name: str, node: ast.Call) -> SemType:
        """`plat_write`, `plat_exit`, `plat_heap` -- the whole platform floor.

        Checked exactly as a call to a declared function is, because that is
        what it is: the signature comes from `objects/floor.py` instead of from
        a `def` in this module, and nothing else about it is special.
        """
        return self._declared_call(
            name, PLATFORM[name], node,
            "it is the platform floor -- see objects/floor.py for what each "
            "argument means")

    def _hostsvc_call(self, name: str, node: ast.Call) -> SemType:
        """One host service, called like any other declared function.

        THE SAME SHAPE AS `_platform_call`, deliberately: a host service is a
        call with a signature and nothing more, which is the argument for not
        making it an opcode. What it is NOT checked for here is whether the
        backend provides its group -- the frontend does not know the backend,
        and `Backend.check_host_services` asks that question where the answer
        exists.
        """
        return self._declared_call(
            name, HOSTSVC[name], node,
            "it is a host service -- see objects/hostsvc.py for what each "
            "argument means and which backends provide it")

    def _declared_call(self, name: str, signature, node: ast.Call,
                       note: str) -> SemType:
        """One call to a function declared outside this module."""
        params, ret = signature
        if len(node.args) != len(params):
            self.sink.report(
                error("E0053", f"{name}() takes {len(params)} argument(s), "
                               f"got {len(node.args)}")
                .at(self._span(node))
                .note(note))
            for a in node.args:
                self._expr(a)
            return ret
        for arg, want in zip(node.args, params):
            got = self._expr(arg)
            self._check_assignable(got, want, arg,
                                   what=f"argument to {name}()", value=arg)
        return ret

    # ── the machine subset: indirect calls ──────────────────────────────
    #
    # DEFERRED THREE TIMES BY docs/INERT-RUNTIME.md, and honestly each time:
    # laying out an object does not need them, dispatching on one does. `str`,
    # `list` and `dict` are tag-dispatched and needed none of this; classes,
    # generators, exceptions and every dunder lookup are function pointers all
    # the way down, and none of them can leave C until the subset can spell one.
    #
    # TWO NAMES RATHER THAN A BARE FUNCTION VALUE. `f: ptr = helper` would
    # read well and it is exactly the mistake this subset exists to refuse: a
    # forgotten `()` would silently become an address instead of a call, which
    # type-checks as `ptr` and produces a plausible wrong number. `funcaddr`
    # makes taking the address something you asked for, and leaves a bare
    # `helper` the error it already was.

    def _funcaddr(self, node: ast.Call) -> SemType:
        """`funcaddr(f)` -- the address of a function defined in this module.

        A NAME, NOT AN EXPRESSION. `Op.FUNC_ADDR` carries its target in
        `Instr.sym`, a symbol fixed at compile time, so there is nothing for a
        computed argument to lower to. Refusing it here says so; accepting it
        would mean inventing a meaning the IR has no room for.
        """
        arg = node.args[0]
        if not isinstance(arg, ast.Name):
            self._error("E0055", "funcaddr() takes the NAME of a function",
                        arg)
            self._expr(arg)
            return PTR
        info = self.functions.get(arg.id)
        if info is None:
            self._error("E0031", f"undefined function {arg.id!r}", arg)
            return PTR
        return PTR

    def _callptr(self, node: ast.Call) -> SemType:
        """`callptr(T, addr, *args)` -- call through a function pointer.

        THE TYPE COMES FIRST, as it does for `load` and `store` and for the
        same reason: what the call ANSWERS is the first thing on the line
        rather than something inferred from where it lands. There is no
        signature to check against -- an address is not a declaration -- so
        the type argument is the whole of what the compiler knows, and it is
        the caller's promise rather than a fact anything verified.
        """
        ty = self._type_arg(node.args[0], "callptr")
        self._want_ptr(node.args[1], "callptr", "address")
        for arg in node.args[2:]:
            got = self._expr(arg)
            if got.is_error:
                continue
            # NO FLOATS THROUGH HERE, and it is a real limit rather than an
            # oversight: the C backend lowers CALL_PTR by casting the address
            # to a signature of `int64_t` parameters, so a double would arrive
            # in the wrong register file and read as an integer. Refused
            # rather than mis-passed -- a wrong number out of a correct call
            # is the failure this subset is built to make impossible.
            if got is FLOAT or (got.is_machine and got.is_machine_float):
                self._bad_argument(
                    arg, "callptr", "argument", got,
                    "an integer or a pointer",
                    "an indirect call passes every argument as a machine "
                    "word; pass a float through memory instead")
        return ty

    def _rodata(self, node: ast.Call) -> SemType:
        """`rodata(b"...")` -- the address of those bytes, read-only.

        THE SIBLING `reserve` NEEDED. That one gives named storage that is
        ZEROED, which serves a cache or a free list and cannot serve a
        constant: there is nowhere to run an initialiser, so bytes that have
        to be there before any code runs could not get there at all.

        BYTES AND NOT `str`, because what this answers is an address and a
        length in BYTES, and a `str` would raise the question of an encoding
        the subset has no opinion about. `b"..."` is unambiguous, and it is
        what the callers want: `apy_from_bytes(rodata(b"StopIteration"), 13)`.

        NO TERMINATOR IS ADDED. The C reads many of its literals as C strings
        and this does not know which -- so a caller that needs one writes it:
        `rodata(b"StopIteration\\x00")`. Appending one silently would make
        `len` wrong for every caller that did not.

        DEDUPLICATED BY CONTENT at lowering, so naming the same bytes twice
        costs one global -- which matters because a runtime says
        `"TypeError"` in a great many places.
        """
        value = node.args[0]
        if not (isinstance(value, ast.Constant)
                and isinstance(value.value, bytes)):
            self.sink.report(
                error("E0070", "rodata() needs a bytes literal")
                .at(self._span(value))
                .note("it becomes the initialiser of a read-only global, so "
                      "it has to be known at compile time")
                .help('e.g. rodata(b"StopIteration")'))
            return PTR
        if not value.value:
            self.sink.report(
                error("E0070", "rodata() needs at least one byte")
                .at(self._span(value))
                .note("a zero-length global has no address to hand back"))
        return PTR

    def _reserve(self, node: ast.Call) -> SemType:
        """`reserve("name", bytes)` -> the address of named static storage.

        BOTH ARGUMENTS ARE COMPILE-TIME. The name becomes an IR global's
        symbol, so it cannot be computed; the size becomes that global's, so it
        cannot either. Being able to write it anywhere -- rather than only in
        some module-level declaration section -- is what keeps the region's
        definition next to the code that reads it.

        Two `reserve`s with the same name are the SAME storage, which is the
        point: two functions share a cache by naming it. With different sizes
        they are a mistake, and one that would otherwise show up as whichever
        of them the emitter happened to see second.
        """
        text = _string_literal(node.args[0])
        size = const_int(node.args[1], self.functions)
        self._expr(node.args[1])
        if text is None or not text.isidentifier():
            self.sink.report(
                error("E0035", "reserve() needs a literal name")
                .at(self._span(node.args[0]))
                .note("it becomes the symbol of a global, so it has to be "
                      "an identifier known at compile time")
                .help('e.g. reserve("small_ints", 2096)'))
            return PTR
        if size is None or size <= 0:
            self.sink.report(
                error("E0018", "reserve() needs a positive size the "
                               "compiler can work out")
                .at(self._span(node.args[1]),
                    "not a constant" if size is None else str(size)))
            return PTR
        seen = self.reserved.get(text)
        if seen is not None and seen != size:
            self.sink.report(
                error("E0036", f"{text!r} is reserved twice, as {seen} bytes "
                               f"and as {size}")
                .at(self._span(node))
                .note("two reserves of one name are one region, so the sizes "
                      "have to agree"))
            return PTR
        self.reserved[text] = size
        return PTR

    def _type_arg(self, node, name: str) -> SemType:
        """The type argument of `load`/`store`/`sizeof`, read as a TYPE.

        NOT walked as an expression, which is the point: `i64` never becomes a
        value, so it can never leak into arithmetic and there is no first-class
        type object anyone has to be told about.
        """
        text = getattr(node, "id", None)
        if text in MACHINE:
            return MACHINE[text]
        self.sink.report(
            error("E0017", f"the first argument to {name}() is a type")
            .at(self._span(node),
                ast.unparse(node) if text is None else text)
            .note("one of: " + ", ".join(MACHINE))
            .help(f"e.g. {name}(i64, ...)"))
        return ERROR

    def _want_ptr(self, node, name: str, what: str) -> None:
        got = self._expr(node)
        if got.is_error or got.is_ptr:
            return
        self._bad_argument(node, name, what, got, "a ptr",
                           help="wrap an integer address with ptr(...)")

    def _bad_argument(self, node, name: str, what: str, got: SemType,
                      want: str, note: str = "", help: str = "") -> None:
        """The one diagnostic for an intrinsic argument of the wrong type."""
        d = error("E0019", f"the {what} {name}() takes is {want}, not {got}") \
            .at(self._span(node), str(got))
        if note:
            d.note(note)
        if help:
            d.help(help)
        self.sink.report(d)

    def _retype(self, node, want: SemType) -> None:
        """Record `want` as this literal's type, and its operand's.

        BOTH, because `-1` is `UnaryOp(USub, Constant(1))` and lowering reads
        the OPERAND's type to pick the width it negates at. Retyping only the
        outer node left every negative literal at the default width, which
        showed up as a verifier complaint about mismatched operand types
        rather than as anything pointing here.
        """
        self.current.expr_types[id(node)] = want
        if isinstance(node, ast.UnaryOp):
            self.current.expr_types[id(node.operand)] = want

    # ── importing a namespace ───────────────────────────────────────────────
    def _static_import(self, alias, node) -> None:
        """`import a.b.c` and `import a.b.c as n`, on the static path.

        A namespace here is a COMPILE-TIME thing: there is no module object at
        run time and nothing is emitted for the statement. What it binds is a
        name the analyser will recognise at the head of an attribute chain --
        which is enough for `block.Block()` and is all a subset with no first
        class objects can honestly offer.

        `import a.b.c as n` binds `n` to the module. `import a.b.c` binds `a`,
        as Python does, and the rest of the chain is read at the use site.
        """
        if resolve(alias.name) is None and not self._is_package_root(alias.name):
            self._no_module(alias.name, node)
            return
        self._remember_namespace(alias)
        self._bind_namespace(alias.asname or alias.name.split(".")[0], node)

    # ── extending a Java type ───────────────────────────────────────────────
    #
    #     class MyBlock(block.Block):
    #         def getHardness(self) -> int:
    #             return 99
    #
    # THE OTHER DIRECTION. Everything else here calls INTO the class path;
    # this declares a type the backend adds TO it, so that Java can call back
    # -- which is what every mod loader in existence requires, because
    # registration means handing it something it will invoke.
    #
    # It is not a class in the runtime-object sense and never reaches
    # `self.classes`. `self` is a HANDLE like any other Java value, each method
    # is an ordinary static function of `(self, ...)`, and `MyBlock()` is a
    # constructor call resolved exactly as `block.Block()` is. So the static
    # path needs no notion of objects to gain this, and the dynamic path --
    # where `class` means what Python means -- is untouched.

    def _declare_java_subclasses(self, tree: ast.Module) -> None:
        """Find them, name them, and register their methods as functions.

        IN TWO PASSES over the same statements. The first settles which names
        are types, because a method of one may be annotated with another
        written further down; the second builds the signatures, which needs
        every one of those names already to be a type.
        """
        classes = [(n, self._java_base(n)) for n in tree.body
                   if isinstance(n, ast.ClassDef)]
        classes = [(n, b) for n, b in classes if b is not None]
        seen: set = set()
        kept: list = []
        for node, base in classes:
            if node.name in seen:
                # Two of them produce ONE class file, so the second silently
                # replaces the first -- and a Python programmer reasonably
                # expects rebinding rather than a merge. Only one can be
                # generated, so this is refused rather than resolved.
                self._error("E0110",
                            f"{node.name} is declared over a Java type twice; "
                            f"only one class can be generated", node)
                continue
            seen.add(node.name)
            kept.append((node, base))
        classes = kept
        for node, base in classes:
            # Provisional: enough to BE a type -- a name, a handle type, and
            # what it is assignable to. The constructors and the method
            # symbols arrive in the second pass, from the backend.
            self.java_subclasses[node.name] = {
                "internal": node.name,
                "type": JAVA + node.name,
                "supers": [base["internal"]] + list(base["supers"]),
                "abstract": False, "interface": False,
                "new": [], "static": base["static"],
                "instance": base["instance"], "impl": {},
            }
            self.java_subclass_methods[node.name] = {}
        for node, base in classes:
            self._declare_java_subclass(node, base)

    def _java_base(self, node: ast.ClassDef) -> dict | None:
        """The Java type this `class` extends, as a table, or None.

        None is the ordinary answer and means "an ordinary Python class",
        which is checked and lowered by everything that was already here.
        """
        if len(node.bases) != 1 or node.keywords:
            return None
        return self._java_type_table(node.bases[0])

    def _is_java_subclass(self, node) -> bool:
        return (isinstance(node, ast.ClassDef)
                and node.name in self.java_subclasses)

    def _declare_java_subclass(self, node: ast.ClassDef, base: dict) -> None:
        methods: list = []
        for stmt in node.body:
            if isinstance(stmt, _DEF_NODES):
                key = self._java_method(node, stmt)
                if key is not None:
                    info = self.functions[key]
                    methods.append((stmt.name,
                                    [p.type.name for p in info.params[1:]],
                                    info.ret.name))
            elif isinstance(stmt, ast.Pass) or _is_docstring(stmt):
                pass
            else:
                self._error("E0103",
                            "only methods are supported in a class over a "
                            "Java type; there are no fields", stmt)
        if node.decorator_list:
            self._error("E0104", "a class over a Java type takes no "
                                 "decorators", node)

        table = modules.declare_subclass(node.name, base["internal"], methods)
        if table is None:
            self._error("E0105",
                        f"this backend cannot extend {base['internal']}; "
                        f"a class over a Java type needs the jvm backend",
                        node)
            return
        for where, message in table.get("errors", ()):
            # Reported at the METHOD when the backend named one, because a
            # signature that does not override is a fact about that line and
            # not about the class.
            at = next((s for s in node.body
                       if isinstance(s, _DEF_NODES) and s.name == where),
                      node)
            self._error("E0106", message, at)
        table.setdefault("impl", {})
        self.java_subclasses[node.name] = table
        for name, key in self.java_subclass_methods[node.name].items():
            found = table["impl"].get(name)
            if found is not None:
                # THE SYMBOL COMES FROM THE BACKEND, like every other Java
                # symbol: it encodes the generated class, its base and the
                # descriptor, and the backend rebuilds the whole class file
                # from it. A name invented here would be a second spelling.
                self.functions[key].java_symbol = found["symbol"]

    def _java_method(self, node: ast.ClassDef, stmt) -> str | None:
        """Register one method as a static function of `(self, ...)`."""
        if isinstance(stmt, ast.AsyncFunctionDef):
            self._error("E0107", "a method of a class over a Java type cannot "
                                 "be `async`", stmt)
            return None
        if not stmt.args.args or stmt.args.args[0].annotation is not None:
            self._error("E0108",
                        "the first parameter of such a method is `self` and "
                        "is not annotated", stmt)
            return None
        key = f"{node.name}.{stmt.name}"
        if key in self.functions:
            self._error("E0109", f"{node.name} defines {stmt.name!r} twice",
                        stmt)
            return None
        info = self._signature(stmt, self_type=sem_type(JAVA + node.name))
        info.name = key
        info.qualname = key
        self.functions[key] = info
        self.def_of_node[id(stmt)] = key
        self.java_subclass_methods[node.name][stmt.name] = key
        self._java_method_keys.append(key)
        return key

    def _collect_java_subclass(self, node: ast.ClassDef,
                               scope: _Scope) -> None:
        """Scopes for its methods. Each is a MODULE-LEVEL function.

        Not a class scope: there is no class body to run, no attribute to bind
        and nothing to capture. A method here is a `def` that happens to be
        written indented, and treating it as one is what lets closures, name
        resolution and everything else stay exactly as they were.
        """
        for stmt in node.body:
            key = self.def_of_node.get(id(stmt))
            if key is None or not isinstance(stmt, _DEF_NODES):
                continue
            child = self._new_scope(key, "function", scope, stmt)
            for a in stmt.args.args:
                child.bound.add(a.arg)
            for s in stmt.body:
                self._collect(s, child)

    def _java_type_table(self, node) -> dict | None:
        """The backend's table for a Java type named by an expression.

        `block.Block` and `jvm.com.minecraft.block.Block` -- the attribute
        forms -- and the bare name of a class this source declares over one.
        """
        if isinstance(node, ast.Name):
            return self.java_subclasses.get(node.id)
        if not isinstance(node, ast.Attribute):
            return None
        found = self._namespace_of(node)
        if found is None:
            return None
        module, path = found
        if len(path) != 1:
            return None
        entry = member(module, path[0])
        if entry is None or entry[0] != "jclass":
            return None
        return entry[1]

    def _type_table(self, internal: str) -> dict | None:
        """Everything known about one handle type, generated or found."""
        found = self.java_subclasses.get(internal)
        return found if found is not None else modules.type_of(internal)

    def _remember_namespace(self, alias) -> None:
        """Record what an `import` makes nameable, for every later function."""
        if alias.asname:
            self.namespaces[alias.asname] = alias.name
        else:
            self.namespaces.setdefault(alias.name.split(".")[0], "")

    def _is_namespace_import(self, node) -> bool:
        """True for `import x` where every name is a backend namespace."""
        if not isinstance(node, ast.Import):
            return False
        return all(self._namespace_import(alias) for alias in node.names)

    def _namespace_import(self, alias) -> bool:
        table = resolve(alias.name)
        if table is not None:
            return any(entry[0] == "jclass"
                       for entry in _members_of(table)) or not len(table)
        return self._is_package_root(alias.name)

    def _is_package_root(self, dotted: str) -> bool:
        """True if anything importable starts with this name.

        `import jvm.com.minecraft.block` names a module; `import com.minecraft`
        may name only a prefix of one, and binding the head is still right --
        the use site is what decides whether the whole chain resolves.
        """
        prefix = dotted + "."
        return any(name.startswith(prefix) for name in importable())

    def _bind_namespace(self, name: str, node) -> None:
        """A namespace occupies a name without being a value.

        Recorded as assigned so that reading it is not "used before
        assignment", and NOT given a type: there is no value, and an
        expression that tries to use one gets a diagnostic saying so.
        """
        self.assigned.add(name)

    def _namespace_of(self, node) -> tuple[str, list[str]] | None:
        """Split an attribute chain into (module, remaining names).

        `block.Block` with `block` bound to `com.minecraft.block` is
        (`com.minecraft.block`, ["Block"]). `jvm.com.minecraft.block.Block`,
        where only the head `jvm` was bound, is the same pair -- found by
        trying the longest prefix first, because `com.minecraft.block` and
        `com.minecraft` may both resolve and the longer one is meant.
        """
        parts: list[str] = []
        at = node
        while isinstance(at, ast.Attribute):
            parts.append(at.attr)
            at = at.value
        if not isinstance(at, ast.Name) or at.id not in self.namespaces:
            return None
        parts.reverse()
        bound = self.namespaces[at.id]
        if bound:
            return bound, parts
        chain = [at.id] + parts
        for cut in range(len(chain) - 1, 0, -1):
            candidate = ".".join(chain[:cut])
            if resolve(candidate) is not None:
                return candidate, chain[cut:]
        return None

    #: Type conversions. Not "functions" -- they are the only calls whose
    #: result type depends on which one you named rather than on a signature.
    _CONVERSIONS = {"int": INT, "float": FLOAT, "bool": BOOL}

    def _call(self, node: ast.Call) -> SemType:
        if isinstance(node.func, ast.Attribute):
            # A NATIVE CALL BEFORE A JAVA ONE. Both are attribute calls and
            # only the receiver tells them apart, so the one with the narrower
            # test goes first.
            if self._is_ctypes_call(node):
                return self._ctypes_call(node)
            return self._java_call(node)
        if not isinstance(node.func, ast.Name):
            self._error("E0050", "only direct calls by name are supported", node)
            return ERROR
        name = node.func.id
        if node.keywords:
            self._error("E0051", "keyword arguments are not supported", node)
        if name in self.java_subclasses:
            # `MyBlock(7)` -- CONSTRUCTING the generated class. Resolved
            # against the same overload set the base's constructors give, so
            # `MyBlock(7)` and `block.Block(7)` agree about which one they
            # mean and disagree only about which class comes out.
            return self._pick(node, self.java_subclasses[name]["new"], name)
        if name == "print":
            for a in node.args:
                self._arg_expr(a)
            return NONE
        if name in self._CONVERSIONS and name not in self.functions:
            want = self._CONVERSIONS[name]
            if len(node.args) != 1:
                self._error("E0054",
                            f"{name}() takes exactly one argument, "
                            f"got {len(node.args)}", node)
                for a in node.args:
                    self._expr(a)
                return want
            got = self._expr(node.args[0])
            if not (got.is_numeric or got.is_error):
                self.sink.report(
                    error("E0055", f"cannot convert {got} to {want}")
                    .at(self._span(node.args[0]), str(got)))
                return ERROR
            return want
        if name in MACHINE and name not in self.functions:
            return self._machine_conversion(name, node)
        if name in MEMORY_INTRINSICS and name not in self.functions:
            return self._memory_intrinsic(name, node)
        if name in PLATFORM and name not in self.functions:
            return self._platform_call(name, node)
        if name in HOSTSVC and name not in self.functions:
            return self._hostsvc_call(name, node)
        if name in OBJECT_RUNTIME and name not in self.functions:
            return self._runtime_call(name, node)
        info = self.functions.get(name)
        if info is None:
            if name in _NEEDS_A_COMPILER:
                self._no_compiler(name, node)
                return ERROR
            self.sink.report(
                error("E0052", f"call to unknown function {name!r}")
                .at(self._span(node.func))
                .note("known functions: "
                      + ", ".join(sorted(self.functions)) or "(none)"))
            for a in node.args:
                self._arg_expr(a)
            return ERROR
        params, ret = info.signature
        if len(node.args) != len(params):
            self.sink.report(
                error("E0053", f"{name}() takes {len(params)} argument(s), "
                               f"got {len(node.args)}")
                .at(self._span(node))
                .also(self._span_of_def(info), f"{name} is defined here"))
        for arg, want in zip(node.args, params):
            got = self._expr(arg)
            self._check_assignable(got, want, arg, what=f"argument to {name}()")
        for extra in node.args[len(params):]:
            self._expr(extra)
        return ret



    # ── ctypes ──────────────────────────────────────────────────────────────
    #
    # A NATIVE CALL RESOLVED WHILE COMPILING. See `cffi.py` for why this needs
    # no `dlopen` and therefore adds nothing to the platform floor: the C
    # backend already emits an `extern` for any external the IR declares and
    # does not define, and the toolchain resolves it. `CDLL("m")` is a promise
    # to the linker, not a load.

    def _collect_ctypes(self, tree) -> None:
        """Read every ctypes declaration BEFORE any function body.

        A declaration is a module-level statement and a use is usually inside
        a function, and function bodies are analysed first -- so collecting
        these where they are written left `libm` undefined in the one place
        anybody writes `libm.sqrt(...)`. The same reason `_declare_java_
        subclasses` runs here: a name a body may use has to exist before the
        body is read.

        THIS PASS REPORTS, and it is the only one that can. A declaration is
        excluded from the entry's body -- it describes the build and emits no
        instruction -- so the statement is never analysed again, and a pass
        that stayed quiet here swallowed every diagnostic about one. `from
        ctypes import Structure` compiled cleanly and did nothing.

        Nothing is reported twice, for the same reason: these statements are
        visited here and nowhere else.
        """
        for stmt in tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                # A DECLARED NATIVE LIBRARY IS ASKED ABOUT SECOND, so `import
                # ctypes` keeps meaning `ctypes` even if someone declares a
                # library under that name -- and `_nativelib_import` refuses
                # any module name that already resolves anyway.
                self._ctypes_import(stmt) or self._nativelib_import(stmt)
            elif isinstance(stmt, ast.Assign):
                self._ctypes_assign(stmt)

    def _ctypes_import(self, node) -> bool:
        """`import ctypes` / `from ctypes import ...`. True if it was one.

        Recognised BEFORE the ordinary import rules and on either path, because
        `ctypes` is not a module this compiler has -- it is a spelling this
        compiler understands, in the same way `plat_write` is a name rather
        than an import.
        """
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "ctypes":
                    return False
            for alias in node.names:
                self.ctypes_names[alias.asname or "ctypes"] = "ctypes"
            self.ctypes_stmts.add(id(node))
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "ctypes" \
                and not node.level:
            for alias in node.names:
                local = alias.asname or alias.name
                if alias.name in cffi.TYPES:
                    self.ctypes_names[local] = "ctypes.type"
                elif alias.name in cffi.LOADERS:
                    self.ctypes_names[local] = "ctypes.loader"
                else:
                    self._error("E0128",
                                f"ctypes.{alias.name} is not supported; this "
                                f"frontend has the scalar types, the library "
                                f"loaders, and calls with declared signatures",
                                node)
            self.ctypes_stmts.add(id(node))
            return True
        return False

    def _nativelib_import(self, node) -> bool:
        """`import user32`, where a declaration said what `user32` is.

        True if this statement named a DECLARED NATIVE LIBRARY, in which case
        it is handled here and the ordinary import rules never see it.

        WHAT IT BINDS, AND WHY THERE IS NOTHING ELSE TO IT. A declared
        function is a ctypes function whose `argtypes` are already known, so
        this fills in exactly the state those three statements would have
        left behind -- the library for the linker, and one signature per
        function -- and every path afterwards is the ctypes path. See
        `nativelib.py` for why that is the whole design and not a shortcut.

        A DECLARATION CANNOT SHADOW A MODULE THAT ALREADY RESOLVES. `resolve`
        is asked FIRST, so pointing `import math` at a DLL leaves `math`
        meaning what it meant. Refused rather than silently ignored, because
        a declaration that does nothing is a build someone will debug.
        """
        registry = nativelib.current()
        if not registry or not isinstance(node, (ast.Import, ast.ImportFrom)):
            return False
        if isinstance(node, ast.ImportFrom):
            # `from user32 import GetSystemMetrics` would bind the function as
            # a bare name, and a native call is recognised by the shape
            # `lib.f(...)` -- there is no value to bind. Reported rather than
            # left to fail later as an undefined name.
            if node.level or node.module is None:
                return False
            if registry.get(node.module, self.native_os) is None:
                return False
            self._error("E0130",
                        f"a native library is imported whole; write "
                        f"`import {node.module}` and call "
                        f"`{node.module}.{node.names[0].name}(...)`", node)
            self.ctypes_stmts.add(id(node))
            return True
        claimed = False
        for alias in node.names:
            library = registry.get(alias.name, self.native_os)
            if library is None:
                continue
            if resolve(alias.name) is not None:
                self._error("E0131",
                            f"{alias.name!r} is declared as a native library "
                            f"and is already a module this compiler has",
                            node)
                claimed = True
                continue
            local = alias.asname or alias.name
            self.ctypes_libs[local] = library.library
            if library.library not in self.ctypes_libraries:
                self.ctypes_libraries.append(library.library)
            for fn in library.functions:
                self.ctypes_sigs[(local, fn.name)] = {
                    "params": list(fn.params), "ret": fn.ret,
                    "symbol": fn.c_symbol,
                }
            claimed = True
        if claimed:
            self.ctypes_stmts.add(id(node))
        return claimed

    def _ctypes_assign(self, node) -> bool:
        """The three statements a ctypes declaration is made of.

            lib = ctypes.CDLL("m")
            lib.sqrt.restype = ctypes.c_double
            lib.sqrt.argtypes = [ctypes.c_double]

        True if this statement was one of them, so the caller stops.
        """
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            return False
        target = node.targets[0]
        claimed = self.ctypes_stmts.add

        # `lib = ctypes.CDLL("m")`
        if isinstance(target, ast.Name):
            library = cffi.loader_call(node.value, self.ctypes_names)
            if library is None:
                return False
            if not library:
                self._error("E0124",
                            "a ctypes library must be named by a literal",
                            node.value)
                self.ctypes_libs[target.id] = ""
                return True
            self.ctypes_libs[target.id] = library
            if library not in self.ctypes_libraries:
                self.ctypes_libraries.append(library)
            claimed(id(node))
            return True

        # `lib.sqrt.restype = ...` / `lib.sqrt.argtypes = [...]`
        if not (isinstance(target, ast.Attribute)
                and target.attr in ("restype", "argtypes")
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id in self.ctypes_libs):
            return False
        key = (target.value.value.id, target.value.attr)
        signature = self.ctypes_sigs.setdefault(key, {})
        if target.attr == "restype":
            name = cffi.type_name(node.value, self.ctypes_names)
            if name is None:
                self._error("E0121", "restype must be a ctypes scalar type",
                            node.value)
                claimed(id(node))
                return True
            signature["ret"] = name
            claimed(id(node))
            return True
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            self._error("E0122", "argtypes must be a list",
                        node.value)
            claimed(id(node))
            return True
        params = []
        for item in node.value.elts:
            name = cffi.type_name(item, self.ctypes_names)
            if name is None:
                self._error("E0123",
                            "this is not a ctypes scalar type", item)
                claimed(id(node))
                return True
            params.append(name)
        signature["params"] = params
        claimed(id(node))
        return True

    def _used_as_value(self, at) -> bool:
        """Whether a ctypes library name is being read rather than called.

        `libm.sqrt(1.0)` never reaches `_lookup` -- `_call` intercepts it --
        so anything that does get here is `libm` on its own or `libm.sqrt`
        without a call, neither of which is a thing. Reported where it is
        written rather than left to become a stranger error downstream.
        """
        self._error("E0127",
                    "a ctypes library is a compile-time name, not a value; "
                    "it can only be called through", at)
        return True

    def _is_ctypes_call(self, node) -> bool:
        return (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in self.ctypes_libs)

    def _dyn_ctypes_call(self, node: ast.Call) -> SemType:
        """The same call from code with no static types.

        THE SIGNATURE IS STILL REQUIRED and still not guessed -- that rule is
        about the CALLEE and does not depend on what the caller knows. What
        changes is the arguments: a dynamic value's type is a run-time fact,
        so each one is CONVERTED at the call rather than checked at the
        compile. That is what CPython's ctypes does, and here it is the only
        thing available -- refusing every dynamic argument would refuse the
        feature.
        """
        local, symbol = node.func.value.id, node.func.attr
        signature = self.ctypes_sigs.get((local, symbol), {})
        params = signature.get("params")
        if params is None:
            self._error("E0125",
                        f"{local}.{symbol} has no argtypes; this frontend "
                        f"will not guess a native signature", node)
            for a in node.args:
                self._expr(a)
            return OBJ
        ret = signature.get("ret", cffi.DEFAULT_RESTYPE)
        if len(node.args) != len(params):
            self._error("E0126",
                        f"{local}.{symbol}() takes {len(params)} argument(s) "
                        f"by its argtypes, got {len(node.args)}", node)
        for a in node.args:
            self._expr(a)
        # A POINTER ARGUMENT IS A STRING'S BYTES, checked at run time. The
        # kind cannot be known here -- that is what dynamic means -- so
        # `apy_str_bytes` does the checking and fails with the callee's name
        # rather than handing a native function the address of an integer
        # cell. NUL termination and lifetime are both already guaranteed: see
        # the accessor.
        wide = [cffi.TYPES[p] for p in params]
        if self.current is not None:
            self.current.ctypes_calls[id(node)] = {
                # THE DECLARED C SYMBOL WINS, when a declaration gave one.
                # `nativelib` allows a function to be imported under a name
                # that is not the symbol's; a program writing `ctypes` gives
                # no such mapping, and there the two are the same string.
                "symbol": signature.get("symbol", symbol),
                "params": wide,
                "ret": cffi.TYPES[ret],
                "library": self.ctypes_libs.get(local, ""),
                "dynamic": True,
            }
        return OBJ

    def _ctypes_call(self, node: ast.Call) -> SemType:
        """`lib.sqrt(9.0)` -- an ordinary call to an external symbol."""
        local, symbol = node.func.value.id, node.func.attr
        signature = self.ctypes_sigs.get((local, symbol), {})
        params = signature.get("params")
        if params is None:
            # STRICTER THAN CPYTHON, and deliberately. There a missing
            # `argtypes` means "guess from the values", and guessing is how a
            # ctypes program corrupts a stack -- a Python int passed where the
            # callee wants a 32-bit value does not fail, it truncates. A
            # compiler that knows the types at build time has no reason to
            # guess.
            self._error("E0125",
                        f"{local}.{symbol} has no argtypes; this frontend "
                        f"will not guess a native signature", node)
            for a in node.args:
                self._expr(a)
            return ERROR
        # `restype` DEFAULTS TO `c_int`, which is what ctypes documents and
        # what a program relying on it is entitled to.
        ret = signature.get("ret", cffi.DEFAULT_RESTYPE)
        if len(node.args) != len(params):
            self._error("E0126",
                        f"{local}.{symbol}() takes {len(params)} argument(s) "
                        f"by its argtypes, got {len(node.args)}", node)
        for arg, want in zip(node.args, params):
            got = self._expr(arg)
            self._check_assignable(got, BY_NAME[cffi.TYPES[want]], arg,
                                   what=f"argument to {symbol}()", value=arg)
        for extra in node.args[len(params):]:
            self._expr(arg)
        if self.current is not None:
            self.current.ctypes_calls[id(node)] = {
                # THE DECLARED C SYMBOL WINS, when a declaration gave one.
                # `nativelib` allows a function to be imported under a name
                # that is not the symbol's; a program writing `ctypes` gives
                # no such mapping, and there the two are the same string.
                "symbol": signature.get("symbol", symbol),
                "params": [cffi.TYPES[p] for p in params],
                "ret": cffi.TYPES[ret],
                "library": self.ctypes_libs.get(local, ""),
            }
        return BY_NAME[cffi.TYPES[ret]]

    # ── calling Java ────────────────────────────────────────────────────────
    def _java_call(self, node: ast.Call) -> SemType:
        """`block.Block()`, `block.Block.count()`, `my_block.setName(x)`.

        Three shapes and one resolution: find the overload set, then pick the
        overload the arguments fit. The set comes from the backend, which read
        it out of the class path, so a call that type-checks here names a
        method that is really there.
        """
        found = self._namespace_of(node.func)
        if found is not None:
            module, path = found
            return self._namespace_call(node, module, path)

        # Not a namespace, so the base is a VALUE -- an instance method call.
        receiver = self._expr(node.func.value)
        if receiver.is_error:
            return ERROR
        if not is_java(receiver):
            self._error("E0092",
                        f"only direct calls by name are supported; "
                        f"{ast.unparse(node.func.value)} is {receiver}", node)
            return ERROR
        # A method written in THIS source, on a class this source declares.
        # Called directly: the class path has never heard of it, and reaching
        # a function in the same class file through the JVM's dispatch would
        # be slower and no more correct.
        own = self.java_subclass_methods.get(
            java_class_of(receiver), {}).get(node.func.attr)
        if own is not None:
            return self._sub_call(node, own)
        table = self._type_table(java_class_of(receiver))
        if table is None:
            self._error("E0093", f"nothing is known about {receiver}", node)
            return ERROR
        overloads = table["instance"].get(node.func.attr)
        if not overloads:
            self._error("E0094",
                        f"{java_class_of(receiver).replace('/', '.')} has no "
                        f"method {node.func.attr!r}", node)
            return ERROR
        return self._pick(node, overloads, node.func.attr, receiver=True)

    def _sub_call(self, node: ast.Call, key: str) -> SemType:
        """`b.tick(2)` where `tick` is a method of a class declared here.

        An ordinary call to an ordinary function, with the receiver as the
        first argument -- which is what `self` already is.
        """
        info = self.functions[key]
        params, ret = info.signature
        if len(node.args) != len(params) - 1:
            self.sink.report(
                error("E0053", f"{key}() takes {len(params) - 1} argument(s), "
                               f"got {len(node.args)}")
                .at(self._span(node))
                .also(self._span_of_def(info), f"{key} is defined here"))
        for arg, want in zip(node.args, params[1:]):
            got = self._java_arg(arg, want.name)
            self._check_assignable(got, want, arg,
                                   what=f"argument to {key}()")
        for extra in node.args[len(params) - 1:]:
            self._expr(extra)
        if self.current is not None:
            self.current.sub_calls[id(node)] = key
        return ret

    def _namespace_call(self, node, module: str, path: list) -> SemType:
        if not path:
            self._error("E0095", f"{module} is a module, not a function", node)
            return ERROR
        entry = member(module, path[0])
        if entry is None:
            self._error("E0084",
                        f"module {module!r} has no member {path[0]!r}", node)
            return ERROR
        if entry[0] != "jclass":
            self._error("E0096",
                        f"{module}.{path[0]} is not a Java type", node)
            return ERROR
        table = entry[1]
        if len(path) == 1:
            # `block.Block()` -- constructing one.
            if table["abstract"]:
                self._error("E0097",
                            f"{path[0]} is abstract and cannot be "
                            f"constructed", node)
                return ERROR
            if not table["new"]:
                self._error("E0098",
                            f"{path[0]} has no public constructor", node)
                return ERROR
            return self._pick(node, table["new"], path[0])
        if len(path) == 2:
            overloads = table["static"].get(path[1])
            if not overloads:
                self._error("E0099",
                            f"{path[0]} has no static method {path[1]!r}", node)
                return ERROR
            return self._pick(node, overloads, f"{path[0]}.{path[1]}")
        self._error("E0100", f"cannot call {'.'.join(path)}", node)
        return ERROR

    def _pick(self, node, overloads: list, what: str,
              receiver: bool = False) -> SemType:
        """Choose the overload the arguments fit, and record it.

        BY ARITY FIRST, then by whether each argument is assignable. Java
        resolves overloads by a much larger rule -- widening, boxing,
        varargs -- and implementing a fraction of it would be worse than
        implementing an obvious part of it: this picks the first candidate that
        fits exactly, and says so plainly when none does.
        """
        fitting = [o for o in overloads if len(o["params"]) == len(node.args)]
        if not fitting:
            counts = sorted({len(o["params"]) for o in overloads})
            self._error("E0101",
                        f"{what}() takes {' or '.join(map(str, counts))} "
                        f"argument(s), got {len(node.args)}", node)
            for a in node.args:
                self._expr(a)
            return ERROR

        types = [self._java_arg(a, fitting[0]["params"][i])
                 for i, a in enumerate(node.args)]
        for overload in fitting:
            if all(self._java_fits(got, want)
                   for got, want in zip(types, overload["params"])):
                if self.current is not None:
                    self.current.java_calls[id(node)] = dict(
                        overload, receiver=receiver)
                return sem_type(overload["returns"])

        wanted = " or ".join("(" + ", ".join(o["params"]) + ")"
                             for o in fitting)
        self._error("E0102",
                    f"no overload of {what}() takes "
                    f"({', '.join(str(t) for t in types)}); it takes {wanted}",
                    node)
        return ERROR

    def _java_arg(self, arg, want: str) -> SemType:
        """One argument to a Java method.

        A STRING LITERAL is special and only here: the subset has no string
        type, so `"stone"` is not an expression anywhere else in the language.
        In a `String` parameter it is the only thing that could be meant, and
        refusing it would leave every `setName` in the world unreachable.
        """
        if want == JAVA + "java/lang/String" and isinstance(arg, ast.Constant)                 and isinstance(arg.value, str):
            if self.current is not None:
                self.current.java_strings[id(arg)] = arg.value
                self.current.expr_types[id(arg)] = sem_type(want)
            return sem_type(want)
        return self._expr(arg)

    def _java_fits(self, got: SemType, want: str) -> bool:
        if got.is_error:
            return True                       # already reported
        if got.name == want:
            return True
        if is_java(got) and is_java(want):
            # A subclass fills a superclass parameter, and a class fills an
            # interface it implements. Java's own rule, and the one that makes
            # an API usable: nearly every method in one takes a base type.
            table = self._type_table(java_class_of(got))
            return bool(table) and java_class_of(want) in table["supers"]
        if want in ("int", "float", "bool"):
            # The same widening the language already does between its own
            # numbers, and nothing more: an `int` fills a `double` parameter.
            return got.name in ("int", "bool") if want != "float"                 else got.is_numeric
        return False

    # ── type rules ──────────────────────────────────────────────────────────
    def _yielded(self, types: list[SemType], at, as_condition: bool,
                 what: str) -> SemType:
        """The type of an expression that YIELDS ONE OF ITS OPERANDS.

        `and`, `or` and `a if c else b` do not compute a value -- they pick
        one. So the result's type is whichever operand was chosen, and a
        static path that gives every expression a single type can only be
        right when they agree.

        IT DID NOT REFUSE, IT WIDENED. `_unify_all` answers `float` for an
        `int` and a `float`, which is correct for arithmetic (`a + b` really
        does widen, as Python's numeric tower says) and wrong here: for
        `a: int = 0` and `b: float`, CPython prints `0` and this printed
        `0.0`. The dynamic path printed `0` all along, so the same source
        answered differently depending on whether its function was annotated
        -- which `archived/docs/LANGUAGE.md` described as "the static path is
        a subset with its own rules rather than an optimisation of the dynamic
        one". It is a defect, and this is the fix.

        NOTHING CAUGHT IT because every conformance case is a script, and a
        script's top level is dynamic. The divergence lives only inside a
        fully annotated function, which is the half the corpus never measures.

        IN A CONDITION THE VALUE IS NOT OBSERVED, so nothing has to agree:
        `if a and b:` asks whether the result is truthy and `0` and `0.0`
        answer that identically. Refusing there would reject working programs
        to prevent a difference nobody can see.
        """
        real = [ty for ty in types if not ty.is_error]
        if not real:
            return ERROR
        # DEFER WHERE `_unify_all` ALREADY REFUSES, which is every case
        # involving a machine width: it reports E0016, whose note says "a
        # machine width unifies only with itself" and whose help names the
        # conversion. This rule exists to close the gap where unification was
        # PERMISSIVE -- `int` widening to `float` -- so overriding a sharper
        # diagnostic with a blunter one would be a loss. A condition defers
        # for the other reason: its value is never observed.
        if as_condition or any(ty.is_machine for ty in real):
            return self._unify_all(types, at)
        if any(ty != real[0] for ty in real):
            self.sink.report(
                error("E0065",
                      f"{what} yields one of its operands, and these do not "
                      f"have the same type: "
                      + ", ".join(sorted({str(ty) for ty in real})))
                .at(self._span(at))
                .note("Python answers whichever operand it picked, so the "
                      "result has that operand's type -- there is no single "
                      "type for this expression to have")
                .help("convert the operands to one type, or test it as a "
                      "condition, where the value is not observed"))
            return ERROR
        return real[0]

    def _unify_all(self, types: list[SemType], at) -> SemType:
        real = [t for t in types if not t.is_error]
        if not real:
            return ERROR
        # A MACHINE WIDTH DOES NOT UNIFY WITH ANYTHING BUT ITSELF. Without
        # this, `x if c else 0` over an `i32` fell through to the `INT in real`
        # line below and the whole expression was typed `int` -- so the i32 arm
        # was silently widened and the register the result went into was the
        # wrong width, which the verifier reports about a register rather than
        # about this expression.
        machine = next((t for t in real if t.is_machine), None)
        if machine is not None:
            if any(t != machine for t in real):
                self.sink.report(
                    error("E0016", "these do not all have the same type: "
                          + ", ".join(sorted({str(t) for t in real})))
                    .at(self._span(at))
                    .note("a machine width unifies only with itself")
                    .help(f"convert the others with {machine}(...)"))
                return ERROR
            return machine
        if FLOAT in real:
            return FLOAT
        if INT in real:
            return INT
        return real[0]

    def _check_assignable(self, got: SemType, want: SemType, at,
                          what: str = "value", value=None) -> None:
        """`at` is where the diagnostic points; `value` is the EXPRESSION.

        They differ at the three sites that report against a whole statement
        -- `x: u8 = 5`, `return 5`, `x = 5` -- and the difference matters
        because a literal adapts to a machine width and a statement is not a
        literal. Reported spans are unchanged: `at` still decides those.
        """
        if got.is_error or want.is_error or got == want:
            return
        # A MACHINE WIDTH ON EITHER SIDE ends the numeric tower. See
        # `_machine_binop`: a width exists because something else reads the
        # same bytes, so it converts only where the source says it does.
        if want.is_machine or got.is_machine:
            if self._adapt_literal(at if value is None else value, want):
                return
            d = error("E0060", f"{what} has type {got}, expected {want}") \
                .at(self._span(at), str(got)) \
                .note("machine types never convert implicitly -- a change of "
                      "width has to be visible at the point it happens")
            if not want.is_ptr and not got.is_ptr:
                d.help(f"convert explicitly with {want}(...)")
            self.sink.report(d)
            return
        # bool widens to int and int to float, as Python's own numeric tower
        # does. Nothing narrows implicitly: losing precision silently is how a
        # program computes the wrong answer without ever failing.
        if (want, got) in {(INT, BOOL), (FLOAT, INT), (FLOAT, BOOL)}:
            return
        # A subclass where a superclass is wanted, or a class where one of its
        # interfaces is. Java's rule, and the one that makes an API usable at
        # all: nearly every method in one is declared over a base type.
        if is_java(got) and is_java(want) and self._java_fits(got, want.name):
            return
        d = error("E0060", f"{what} has type {got}, expected {want}") \
            .at(self._span(at), str(got))
        if (want, got) == (INT, FLOAT):
            d.help("convert explicitly with int(...) -- narrowing is never implicit")
        self.sink.report(d)

    # ── positions ───────────────────────────────────────────────────────────
    def _span(self, node) -> Span:
        return span_of(self.source, node)

    def _span_of_def(self, info: FunctionInfo) -> Span:
        return self._span(info.node)

    def _is_splice_dunder(self, node) -> bool:
        """A splice's `__name__` restoration on a function with no object.

        TRUE ONLY FOR THE COMPILER'S OWN STATEMENT, which is why the splices
        mark it rather than this matching on the shape: a program is perfectly
        entitled to write `f.__name__ = 'g'`, and that must go on meaning what
        it says.

        And true only when the target IS static. A dynamic `def` and every
        `class` get a real object, and the restoration is what makes
        `g.__name__` answer `g` instead of `_asmpy_module_m_g` -- dropping it
        for those would trade a trap for a wrong answer.
        """
        if not getattr(node, "splice_dunder", False):
            return False
        target = node.targets[0].value
        if not isinstance(target, ast.Name):
            return False
        found = self.functions.get(target.id)
        return found is not None and not found.dynamic

    def _no_module(self, name: str, node) -> None:
        """`name` did not resolve to source. Say WHY, when the answer is known.

        A COMPILED EXTENSION MODULE is the one case where the file exists, is
        unambiguously the module that was asked for, and still cannot be used.
        `E0083` would list the directory it was sitting in among the ones it
        searched, which reads as "not found" about a file that is plainly
        there -- so that case gets a code of its own and names the file.
        """
        from . import imports as _imports
        finder = _imports.current()
        native = finder.native(name)
        if native is None:
            report = (error("E0083",
                            f"no module named {name!r} is available")
                      .at(self._span(node)))
            # WHERE IT LOOKED, rather than a claim that it did not. The text
            # was "there is no import path" unconditionally, and the file
            # being compiled always contributes its own directory as a root
            # -- so the clause was false on every single invocation, about
            # the one place a reader would go and check.
            if finder.roots:
                report = report.note(
                    "searched: " + ", ".join(str(r) for r in finder.roots))
            else:
                report = report.note("there is no import path")
            self.sink.report(report.help("available: "
                                         + ", ".join(importable())))
            return
        report = (error("E0129",
                        f"{name!r} is a compiled extension module, which "
                        f"cannot be compiled from source")
                  .at(self._span(node))
                  .note(f"found: {native}")
                  .note("it is a native binary built against CPython's C "
                        "API, so there is no Python source to splice and no "
                        "interpreter here for it to call back into")
                  .help("use a pure-Python package instead, or call the "
                        "underlying library directly with ctypes"))
        self.sink.report(report)

    def _error(self, code: str, message: str, node, choices=None) -> None:
        """One diagnostic. `choices` lists what WOULD have worked, which for
        an import is the whole answer -- the set is small and knowing it is
        the difference between "no" and "here is what there is"."""
        report = error(code, message).at(self._span(node))
        if choices:
            report = report.help("available: " + ", ".join(choices))
        self.sink.report(report)


_OP_SYMBOLS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
    ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
    ast.LShift: "<<", ast.RShift: ">>", ast.MatMult: "@",
    # THE COMPARISONS TOO, so that a mismatch of machine widths reads the same
    # whichever operator found it: `a < b` and `a + b` break the same rule and
    # say so with one diagnostic rather than two that have to be kept in step.
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
    ast.Gt: ">", ast.GtE: ">=", ast.Is: "is", ast.IsNot: "is not",
    ast.In: "in", ast.NotIn: "not in",
}


def _op_symbol(op) -> str:
    return _OP_SYMBOLS.get(type(op), type(op).__name__)
