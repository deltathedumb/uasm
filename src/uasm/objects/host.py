"""The object runtime, for the reference interpreter.

`objects/csource.py` is the dynamic runtime as C, linked into every compiled
program. The interpreter cannot link C, so `uasm run` trapped on
`apy_from_int` -- which is to say it could not run any program the frontend
now compiles dynamically, which is most of them. That breaks the project's
central invariant from the other side: CPython, the C backend and the machine
backends agreed with each other, and the thing that is supposed to ADJUDICATE
between them could not execute the program at all.

This is the same runtime, backed by REAL PYTHON OBJECTS. An `apy_value` is an
integer handle into a table of ordinary Python values, so `apy_add` is `a + b`
on an `int` and a `float` and every Python semantic -- the numeric tower,
lexicographic list ordering, dict insertion order, `nan` being unordered --
comes for free instead of being restated. A second implementation of the C
would be a second thing to get wrong, and the whole reason this file exists is
to be the thing that is right.

WHAT DOES NOT COME FOR FREE, and why each one is here rather than left to
Python:

  * IDENTITY. `apy_is` is pointer equality in C, and the C shares one cell for
    None, True, False and every int in -5..256 -- CPython's own cache range.
    So the handle table interns exactly those and hands out a fresh handle for
    everything else, and `a = 1; b = 1; a is b` answers the same in both.
  * INTEGERS NO LONGER WRAP, and this entry used to say the opposite. Every
    integer result was truncated here the way the C's `int64_t` truncated,
    "because the invariant under test is that this and the compiled program
    agree -- including where they are both, knowingly, not CPython". The C
    grew arbitrary precision, so the truncation became the thing that made
    the two disagree: `2 ** 64` is a 20-digit number in the compiled program
    and was 0 here. Python's own integers are exactly what the C now
    implements, so the agreement costs nothing to keep.
    `_wrap64` survives for `apy_as_int` alone -- see the note there.
  * MESSAGE TEXT. Most of CPython's messages are what the C was built to
    reproduce, so most come free. The handful that do not are listed at
    `_C_DECIDES` below with what the C says and why.
  * FLOAT REPR is Python's `repr`, which is what `py_repr_double` in
    objects/support.py computes the hard way. That one agrees by construction.

A NULL HANDLE IS NOT A VALUE. In C a failed operation returns NULL and the
frontend is expected to check the error flag before using the result; passing
that NULL to another operation dereferences it and the program dies somewhere
unrelated. Here it raises immediately, naming the operation, because a
reference interpreter that hid a missing check would be worse than useless.

THE ERROR FLAG is sticky and first-writer-wins, exactly as in the C, and for
the same reason: an operation that runs on a value produced by a failure must
still report the ORIGINAL failure.
"""
from __future__ import annotations

import builtins
import math
import sys

from ..ir import types as _ir_types

#: Returned by `call` for a name this file does not provide, so the
#: interpreter's own host table keeps its `putchar`/`put_int`/... bindings and
#: its "no host binding" trap for genuinely unknown symbols.
NOT_MINE = object()

#: CPython's small-int cache, which `apy_from_int` reproduces. Not an
#: optimisation in either place -- it is the reason `a is b` is True for 256
#: and False for 257, and a program can see the difference.
_SMALL_LO, _SMALL_HI = -5, 256

#: What a handle is offset by, so that NO HANDLE CAN EVER EQUAL AN ADDRESS.
#:
#: THIS IS NOT COSMETIC. The IR has one `ptr` type for an object handle and
#: for a raw memory address (a global's, an `alloca`'s), and
#: `interpreter.py`'s shadow refcounting cannot tell which a `ptr` register
#: holds -- it just calls `incref`/`decref`, which are no-ops for a number
#: that is not a live handle. That was safe only while the two numbers could
#: not collide, and they collided CONSTANTLY: handles used to be indices
#: counting up from 1, and globals are laid out from address 8 up, so
#: `ptr.global_addr` handed a register the number 136 in a program whose
#: 136th object was a live instance.
#:
#: The failure was not symmetric, which is why it corrupted rather than
#: cancelling: writing that address into a register increfs handle 136 only
#: if 136 is ALREADY a handle, and the address is usually written first --
#: but the frame's teardown decrefs it unconditionally later, by which time
#: 136 IS a live object. One spurious decref, and an object with a real
#: reference still outstanding finalized early. Measured on a cycle whose
#: members died at their function's return instead of surviving to
#: `gc.collect()`.
#:
#: `1 << 32` sits above any address this interpreter's arena can produce
#: (`Memory` is `1 << 22` bytes) and below `interpreter._FUNC_TAG`
#: (`1 << 40`), which tags a function pointer the same way.
_HANDLE_BASE = 1 << 32


def _wrap64(v: int) -> int:
    """An integer as the C's `int64_t` holds it."""
    v &= (1 << 64) - 1
    return v - (1 << 64) if v >= (1 << 63) else v


#: The exceptions a CORRECT host method can raise, and which are therefore the
#: PROGRAM's answer rather than a bug in this file.
#:
#: DELIBERATELY NOT `Exception`. An `AttributeError` or a `NameError` raised
#: while dispatching to a host method is a typo in a spec table here -- a
#: method name that does not exist -- and catching it would file the
#: interpreter's own bug under the program's, which is exactly the failure this
#: module's header warns about. Narrow enough to keep that distinction.
#:
#: `ArithmeticError` IS WHAT THE LIST WAS MISSING, and it covers OverflowError.
#: `'a'.center(2 ** 100)` raised one straight out of the bridge and KILLED THE
#: INTERPRETER, where CPython and both compiled paths raise a catchable
#: OverflowError. A sweep of 9,339 generated expressions produced 631 lines and
#: 8,708 blanks because of it, and every blank read as a divergence.
#:
#: `LookupError` covers IndexError and KeyError, which a host method reaches
#: the same way -- `[].pop()` and `{}.popitem()` are the program's errors.
_HOST_RAISES = (TypeError, ValueError, ArithmeticError, LookupError,
                MemoryError, RecursionError)


class Exc:
    """An exception VALUE -- what `except ValueError as e` binds.

    Not a Python exception instance: the C carries a type NAME and one
    argument and nothing else, and building a real `ValueError` here would
    give this file behaviour (a class hierarchy, `args`, chaining) the
    compiled program does not have.
    """

    __slots__ = ("name", "arg", "has_arg", "context", "cause", "suppress",
                 "notes", "rendered", "subs", "argv", "dict", "cls", "pos")

    def __init__(self, name: str, arg, has_arg: bool = True) -> None:
        #: THE CLASS THE PROGRAM WROTE, and this exception's own attributes --
        #: both empty for the exceptions the runtime raises itself, which have
        #: neither.
        #:
        #: `raise` and `except` match on the NAME and always did; what these
        #: add is everything a class body puts on an exception that a name
        #: alone cannot hold. `self.code = 404` goes in `dict` and `def
        #: summary(self)` is found through `cls`, so a user exception is an
        #: ordinary object in every way except how it is caught.
        self.dict: dict = {}
        self.cls = None
        #: WHERE IT WAS RAISED, as an index into the position table, or -1
        #: for one that never was -- which is what makes
        #: `ValueError("x").__traceback__` None. See `_apy_raise`.
        self.pos = -1
        #: EVERY argument, when there is more than one. `OSError(2, "No such
        #: file")` carries both and reads them back as `errno`/`strerror`; one
        #: field could hold only the first, so a program passing two lost one.
        self.argv = None
        #: CHAINING. `context` is whatever was being handled when this one was
        #: raised, set implicitly; `cause` is what `raise X from Y` said. They
        #: are separate because `raise ... from None` SUPPRESSES the context
        #: without having a cause, which one field could not express.
        self.context = None
        self.cause = None
        self.suppress = False
        self.notes = None
        #: THE EXCEPTIONS AN `ExceptionGroup` CARRIES, or None for an ordinary
        #: one. A group is an exception like any other; this is the only thing
        #: that distinguishes it.
        self.subs = None
        #: WHETHER `arg` is the already-formatted message rather than the
        #: object the program raised -- see the C's `rendered`.
        self.rendered = False
        #: WHETHER there was an argument, not whether it is None. `E()` and
        #: `E(None)` both hold None and are different exceptions: `e.args` is
        #: `()` for one and `(None,)` for the other, and `repr` shows `E()`
        #: against `E(None)`. The C carries the same flag for the same reason.
        self.has_arg = has_arg
        self.name = name
        self.arg = arg


#: Every place the C's answer is NOT Python's, with the C's answer. Each entry
#: is enforced at the call site below; this table is the index, so the next
#: divergence has an obvious place to be recorded rather than being discovered
#: as a backend disagreement.
#:
#: `str % x`      -- Python FORMATS (`'a%d' % 7` is 'a7'). The C has no
#:                   printf-style formatting and reports CPython's
#:                   "not all arguments converted during string formatting"
#:                   for every `%` with a str on the left.
#: `(-8) ** 0.5`  -- Python returns a complex. There is no complex kind, so
#:                   the C reports a ValueError; see its `apy_pow`.
#: `list + tuple` -- Python says "can only concatenate list (not "tuple") to
#:                   list". The C falls through to the generic operand text.
#: `tuple[9]`     -- Python says "tuple index out of range". The C says
#:                   "list index out of range" for both sequence kinds.
#: `xs['a']`      -- Python says "list indices must be integers or slices, not
#:                   str". The C omits the slices, which it does not have.
#: `s[i]`         -- indexes BYTES, not characters, in both. The C's own
#:                   docstring records this; `len` is the only place the
#:                   byte/character distinction is currently resolved.
#: `int('9'*30)`  -- the C's `strtoll` SATURATES at int64. Python would give a
#:                   big int. Neither is CPython's answer, and the point of
#:                   this file is that both paths give the SAME answer.
_C_DECIDES = "see the docstring above and the call sites marked `C DECIDES`"


class ObjectHost:
    """`apy_*` for the interpreter, over a handle table of Python objects."""

    def __init__(self, interp) -> None:
        self._interp = interp
        # Index 0 is never handed out: a 0 handle is the C's NULL, which means
        # "an error was set" and never a value.
        #
        # A HANDLE IS `_HANDLE_BASE + index`, NOT the index itself, and the
        # offset is load-bearing -- see `_HANDLE_BASE`.
        self._cells: list = [None]
        #: id(object) -> its handle, for the kinds `is` asks about. See
        #: `_value`: the C compares pointers, so one object must be one
        #: handle or the two paths disagree about identity.
        self._identity: dict = {}
        #: WHICH COMPARISON A CONSUMER IS DRIVING -- `<` for `sorted`, `min`
        #: and `list.sort`, `>` for `max`. It cannot be read off the dunder
        #: Python reaches, because `Instance.__gt__` is `max`'s own
        #: comparison and also the REFLECTED half of the `<` a sort drives,
        #: and a refusal names its operands the other way round in each case.
        #: See `_make_order`, which words that refusal.
        self.order_driver = "<"
        self._small: dict[int, int] = {}
        #: `type(x)` results, interned by kind name -- see `_type_of`.
        self._types: dict = {}
        #: Builtin type thunks, by name -- see `_apy_func_is_type`. One per
        #: name, so `int == int` is True.
        self._type_thunks: dict = {}
        #: Where each mutable buffer handed to a native call was copied to,
        #: as address -> the `bytearray` itself. See `_apy_str_bytes`: the
        #: bytes of a host object live in the host, so a native call that
        #: FILLS a buffer fills a copy, and `natives_host` uses this to put
        #: what it wrote back where the program can see it.
        self._native_buffers: dict[int, bytearray] = {}
        #: `member_descriptor`, the class a slot read through its own class
        #: answers -- see the `__slots__` branch in `_apy_getattr`.
        self._member_class = None
        #: `code`, the class `f.__code__` answers -- see `_code_class`.
        self._code_cls = None
        #: `frame`, the class `g.gi_frame` answers -- see `_gen_frame_class`.
        self._gen_frame_cls = None
        #: `asyncio.TaskGroup` -- see `_taskgroup_class`.
        self._taskgroup_cls = None
        #: PEP 657's three classes, and the code objects made from them.
        self._tb_code_cls = None
        self._tb_frame_cls = None
        self._tb_cls = None
        self._code_objects: dict = {}
        #: PEP 657. One (function, line, end_line, col, end_col) per statement
        #: lowered, in source order; the index is what `apy_at` stores.
        #:
        #: ONE POSITION PER STATEMENT, not one per operation -- the frontend
        #: sets a span per statement and that is the granularity it has.
        #:
        #: PER HOST, like every other table here. The C keeps this in file
        #: statics and each compiled program is its own process, so one table
        #: is one program; this file runs MANY programs in one process, and a
        #: module-level list handed the second one the first's rows -- a
        #: traceback then reported a line six too small, which is a wrong
        #: answer and not a crash.
        self.positions: list = []
        #: Every task the program handed to the loop -- see
        #: `_apy_asyncio_create_task`. PER HOST, for the reason `positions`
        #: is: one process runs many programs here, and a shared list would
        #: let the second one step what the first left unfinished.
        self.tasks: list = []
        #: EVERY ASYNC GENERATOR MADE DURING A RUN, so the loop can close the
        #: ones a program abandoned. Per host for the reason `tasks` is: one
        #: process runs many programs, and a shared list let one program's
        #: `asyncio.run` close what another had left suspended.
        self.live_agens: list = []
        #: Which statement is running, and where the last failure happened.
        #: Two because a handler's own statements move the first, and what a
        #: traceback reports is where the exception came from.
        self.pos_here = -1
        self.pos_err = -1
        #: PEP 750's two classes -- see `_template_class`.
        self._template_cls = None
        self._interp_cls = None
        #: `object`'s defaults as callable values -- see `_object_default`.
        self._defaults: dict = {}
        #: `typing` forms, by name -- see `_apy_typing_form` for why one per
        #: name. Per host, like every other table here: a handle indexes THIS
        #: host's cells, and a shared one would hand a second run the first's.
        self._forms: dict = {}
        #: The containers `_text` is inside, by id. A container that holds
        #: itself would otherwise recur until Python's own limit fired.
        self._rendering: set = set()
        #: `class MyError(ValueError):` -> its base name. See
        #: `_apy_exc_register`.
        #: `class AppError(Exception):` WITH A BODY -> the class its name
        #: stands for. See `_apy_exc_class_bind`.
        self.exc_class: dict = {}
        #: id(Class) -> the builtin exception name it stands for. `_type_of`
        #: interns one bare `Class` per kind, so the object standing for
        #: `ValueError` is otherwise indistinguishable from the one standing
        #: for `int` -- and calling it found no `__init__` and answered
        #: `ValueError() takes no arguments`. See `_invoke`.
        self.exc_types: dict = {}
        #: (kept beside `user_exc`, which holds the same names' bases)
        self.user_exc: dict = {}
        self._none = self._new(None)
        #: `...`, as one cell -- see `apy_ellipsis`.
        self._ellipsis = self._new(Ellipsis)
        #: `NotImplemented`, AS ONE CELL. `_new` mints a fresh handle every
        #: call and `x is NotImplemented` compares handles, so a new one per
        #: mention would answer False -- the same identity trap the suspend
        #: token fell into.
        self._notimplemented = self._new(NotImplemented)
        #: The exhaustion sentinel -- see `apy_stop`.
        self._stop = self._new(_STOP)
        #: THE SUSPENSION TOKEN, AS ONE CELL. `_new` mints a fresh handle
        #: every call, and the lowered code compares against this by IDENTITY
        #: -- `async for` tells "suspended" from "produced an item" that way.
        #: A new handle per suspension compared unequal to every other, so the
        #: loop took the token for an item and never terminated. The C
        #: interns it for exactly this reason.
        self._suspend = self._new(_SUSPEND)
        self._true = self._new(True)
        self._false = self._new(False)
        self.err: tuple[str, str] | None = None
        #: The exception OBJECT, when the pending error came from a `raise`
        #: rather than an operation failing. None for the latter, which never
        #: had one -- see `_fail_raised`.
        self.err_value = None
        #: THE exception being handled right now, for implicit chaining: a
        #: `raise` inside an `except` records it as the new exception's
        #: `__context__`. One slot rather than a stack, matching the C.
        self.handling = None

        # ── shadow reference counting ───────────────────────────────────────
        #: OBSERVABLE object lifetime, not real memory management -- nothing
        #: here is ever freed; the arena's own "leak forever" model
        #: (`docs/INERT-RUNTIME.md`) is untouched. What this tracks is WHEN a
        #: `__del__`/a `weakref` callback would fire and what
        #: `sys.getrefcount` would answer, so a program that observes those
        #: three things sees what CPython's program would show it.
        #:
        #: Keyed by handle, and populated ONLY for the INTERNED kinds
        #: (`_INTERNED` below, plus the named classes `_new` interns) --
        #: exactly the kinds `__del__`/`weakref` are ever meaningfully used
        #: on. A plain int/str/float/tuple has no stable per-VALUE handle to
        #: count against (`_new` mints a fresh one most of the time; see its
        #: own docstring), and CPython's `__del__`/`weakref` story is about
        #: OBJECTS, not scalars -- so those are left untracked, permanently
        #: "alive" as far as this table is concerned, same as CPython
        #: effectively treats a small int or an interned string.
        self._refcount: dict[int, int] = {}
        #: target handle -> list of `(ref_self, callback)`, one entry per
        #: `weakref.ref(target, callback)` created against it -- `ref_self`
        #: is the `weakref.ref` INSTANCE's own handle (what a callback
        #: receives, per CPython's convention), `callback` its handle or 0
        #: for none. See `_apy_weakref_register`/`_apy_weakref_deref`.
        self._weakrefs: dict[int, list[tuple[int, int]]] = {}
        #: `ref_self` handle -> its target's handle, the REVERSE of (the
        #: keys of) `_weakrefs`. Purely host-side bookkeeping, never
        #: touched by `_attr_store`/`incref`/`decref` -- which is the
        #: point: a `weakref.ref` instance storing its target as an
        #: ordinary Python attribute (`self._target = obj`) would incref
        #: it like any other attribute, making the "weak" reference a
        #: strong one and defeating the entire feature. Looked up by
        #: `apy_weakref_deref`, keyed by the `ref` instance's OWN handle
        #: rather than anything stored inside it.
        self._weakref_target: dict[int, int] = {}
        #: Handles currently inside `_finalize`, so a `__del__` that reaches
        #: back into its own object (directly, or through the cascade this
        #: object's own decref triggers) neither recurses nor runs twice.
        #: CPython's own GC has the same guard (`_PyGC_FINALIZED`).
        self._finalizing: set = set()
        #: Handles `_finalize_once` has already run for, permanently. A
        #: SEPARATE guard from `_finalizing`: that one covers ONE call's
        #: own recursion, this one covers two SEPARATE calls -- an extra,
        #: erroneous `decref` landing on this handle again after it is
        #: already at (or past) zero, from a bookkeeping bug elsewhere,
        #: would otherwise run `__del__` a second time, which CPython
        #: never does no matter how a refcount implementation gets there.
        self._dead: set = set()
        #: handle -> pin count. See `_protect`/`_unprotect`: a value that
        #: exists only as a HOST Python local (not yet stored into any
        #: register `put()` will incref) but is about to be handed to a
        #: nested `_call` -- the receiver of a bound method, most of all a
        #: `self` under construction -- would otherwise be observed
        #: at its true, transient refcount of zero the moment that nested
        #: call's own frame teardown drops its copy, finalizing something
        #: still very much alive one level up. Pinning it holds `decref`
        #: off; `_unprotect` replays the zero-check once the pin lifts.
        self._protected: dict[int, int] = {}

    # ── the handle table ────────────────────────────────────────────────────
    #: Kinds whose handle is INTERNED, so `is` answers about the object rather
    #: than about which handle it came back through. The mutable containers
    #: are here for the same reason the object kinds are: `xs.append(xs)` then
    #: `xs[1] is xs` is True in a compiled program.
    #:
    #: TUPLES AND FROZENSETS WERE LEFT OUT, on the reasoning that they are
    #: compared by value and that interning would keep every one alive for
    #: the run. Both halves were wrong. `_cells` already holds a reference
    #: to every object it mints a handle for, so nothing is freed either
    #: way -- and value comparison is not the only question asked: `xs =
    #: [t]` then `xs[0] is t` answered FALSE here and True in CPython,
    #: because reading the element back minted a SECOND handle for the same
    #: tuple. Interning by `id(obj)` keeps that correct without making two
    #: separately built equal tuples the same object.
    #:
    #: THE LIFETIME HALF MATTERS MORE. Only an interned kind gets a
    #: reference count (`_new`), and only a counted kind can hold its
    #: contents alive: an uncounted tuple neither kept its elements alive
    #: nor released them, so `return x, y` finalized BOTH before the caller
    #: ever saw them -- a premature `__del__`, the one direction this
    #: scheme must never be wrong in. A str is still left out: `is` on one
    #: is unpredictable in CPython too, and a str holds nothing.
    _INTERNED = (list, dict, set, frozenset, tuple, bytearray, memoryview)

    def _cell(self, h: int):
        """The object handle `h` names. The one place the handle-to-index
        arithmetic lives -- see `_HANDLE_BASE` for why there is any.

        THE LOWER BOUND IS CHECKED, not left to the list: `h` below the
        base is not a handle at all (a raw address, most likely), and
        `self._cells[136 - 2**32]` is a large NEGATIVE index, which
        Python resolves by counting from the END of the list and handing
        back a real, wrong object rather than raising. `IndexError` is
        what `_get` above is written to turn into a diagnosis.
        """
        i = h - _HANDLE_BASE
        if i < 0:
            raise IndexError(h)
        return self._cells[i]

    def _new(self, obj) -> int:
        self._cells.append(obj)
        made = len(self._cells) - 1 + _HANDLE_BASE
        # RECORDED HERE, not only in `_value`. An object first handed out by
        # `_new` and later reached through `_value` got two handles and
        # compared unequal to itself.
        if isinstance(obj, (int, str, bytes, float, complex)):
            # IDENTITY BUT NO COUNT. An immutable value is one handle per
            # OBJECT for the reason `_value` gives, and it is not something
            # the shadow refcount answers questions about -- a str has no
            # `__del__` and nothing asks what its count is. Recorded here as
            # well as in `_value`, because a literal is minted by `apy_lit`
            # straight through `_new`: without this the literal's handle and
            # the one `_value` later made for the same object were two, and
            # `alias(V) is V` was False for a call through a value.
            self._identity.setdefault(id(obj), made)
        elif isinstance(obj, self._INTERNED) or type(obj).__name__ in (
                "Instance", "Class", "Exc", "Func", "Gen", "Iterator",
                "Alias", "Cell"):
            self._identity.setdefault(id(obj), made)
            # BORN AT ZERO. Every INCREF from here is a real, counted
            # reference; nothing after this line double-counts the handle
            # itself the way "start at 1 for the handle table's own entry"
            # would -- `_cells` holding `obj` is bookkeeping, not a Python
            # reference this scheme answers questions about.
            self._refcount[made] = 0
        return made

    # ── shadow reference counting ───────────────────────────────────────────
    def incref(self, h: int) -> None:
        """One more live reference to handle `h`.

        A no-op for the null handle and for anything `_new` did not enrol
        (a plain int/str/float/tuple/...) -- see `_refcount`'s own comment
        for why those are out of scope. Safe to call on any handle for
        exactly that reason: a caller never has to ask first whether `h` is
        the kind of thing this tracks.
        """
        if h and h in self._refcount:
            self._refcount[h] += 1

    def decref(self, h: int) -> None:
        """One fewer live reference to handle `h`.

        At zero: `__del__` runs if the class defines one, every `weakref`
        pointing here dies and fires its callback, and then -- CASCADING --
        every reference THIS object itself held is dropped in turn. That
        last step is not optional: it is what makes `a = [Foo()]; del a`
        finalize `Foo()` in the same statement rather than only the list,
        exactly as CPython's `tp_dealloc` recursing into its container does.
        It is also, with NOTHING extra, why a genuine reference CYCLE does
        NOT collapse this way: each member's count never reaches zero on
        its own, because the other member is still holding one -- the same
        limitation CPython's own refcounting has, which is why `gc.collect`
        exists as a SEPARATE mechanism (`_apy_gc_collect`) rather than
        something this function needs to know about.
        """
        if not h or h not in self._refcount:
            return
        self._refcount[h] -= 1
        if self._refcount[h] > 0 or h in self._protected:
            return
        self._finalize_once(h)

    def _finalize_once(self, h: int) -> None:
        """Run `_finalize(h)`, but never twice for the same handle -- see
        `_dead`. The shared tail of `decref` and `_unprotect`, the two
        places a count landing at (or already past) zero decides to
        finalize.
        """
        if h in self._dead:
            return
        self._dead.add(h)
        self._finalizing.add(h)
        try:
            self._finalize(h)
        finally:
            self._finalizing.discard(h)

    def _protect(self, h: int) -> None:
        """Pin `h` so a `decref` landing on zero while it is pinned does
        not finalize it -- see `_protected`'s own comment. Nests: two
        pins need two lifts, which is what lets `_invoke_obj` pin the
        same handle its own re-entrant call might also pin (`self`
        passed to a method that itself constructs and passes `self`
        onward) without the inner lift exposing the outer's window.
        """
        if h and h in self._refcount:
            self._protected[h] = self._protected.get(h, 0) + 1

    def _unprotect(self, h: int) -> None:
        """Lift one pin from `_protect`. THE ZERO-CHECK IS REPLAYED HERE,
        not skipped: a pinned handle can still be decrefed to zero while
        pinned (that's the whole point), so once the last pin lifts,
        whatever the true count settled at has to be re-examined -- a
        pin that was never decremented past zero is a no-op read of a
        positive count, and one that was is finalized now, exactly where
        `decref` would have if it had not been pinned.
        """
        if not h or h not in self._protected:
            return
        n = self._protected[h] - 1
        if n > 0:
            self._protected[h] = n
            return
        del self._protected[h]
        if self._refcount.get(h, 1) > 0:
            return
        self._finalize_once(h)

    def protect_handoff(self, h: int) -> None:
        """Pin `h` across a RETURN -- `interpreter.py`'s frame teardown
        calls this for the value being returned, just before dropping
        this frame's own reference to it. Same mechanism as `_protect`,
        named separately because the two call sites are answering
        different questions and a reader tracing one should not have to
        rule out the other.
        """
        self._protect(h)

    def release_handoff(self, h: int) -> None:
        """The returned value went NOWHERE -- a call whose result the
        program discards (`f()` as a statement). `put()` calls this when
        it has no destination register to store into, since nothing else
        will ever lift the pin `protect_handoff` put on it, and a pin
        nothing lifts is an object that never finalizes.

        `_unprotect` rather than `settle`: this drops ONE layer and
        re-checks the count, so a value that is ALSO under construction
        somewhere up the stack keeps that separate pin, and a value whose
        count really did reach zero finalizes here -- which is where
        CPython finalizes a discarded temporary too.
        """
        self._unprotect(h)

    def settle(self, h: int) -> None:
        """`h` just found a real, counted home -- `interpreter.py`'s
        `put()` calls this right after its own `incref(h)`, when storing
        `h` into a register. Whatever transient pin `_instantiate` (or
        anywhere else) left on it while it was in flight, with nothing
        BUT that pin standing between it and a premature `decref` in a
        nested call, is no longer doing any work: the register itself is
        a real reference now. Dropping the pin here rather than leaving
        it for `_unprotect` to unwind matters when the pinning frame is
        still several nested calls away from returning -- `_instantiate`
        never unpins its own `obj` at all, precisely so this is the one
        place that does. A no-op for a handle nothing ever pinned.
        """
        self._protected.pop(h, None)

    def refcount(self, h: int) -> int:
        """The shadow count `sys.getrefcount` reads, or `-1` for a handle
        this scheme does not track (a plain scalar) -- the caller decides
        what an untracked object's refcount should read as."""
        return self._refcount.get(h, -1)

    def _finalize(self, h: int) -> None:
        obj = self._cell(h)
        if isinstance(obj, Instance):
            m = obj.cls.find("__del__")
            if m is not None and isinstance(m, (Func, Native)):
                saved_err, saved_val = self.err, self.err_value
                self.err = None
                self.err_value = None
                try:
                    self._invoke_obj(m.bind(obj), [])
                except _Trap:
                    raise
                except _UserFailed:
                    # The flag is already set; the report below is what
                    # this does with it. `_invoke_obj` RAISES rather than
                    # returning on a failed call, so without this the
                    # `__del__`'s own exception escaped `_finalize` --
                    # out of whatever `decref` happened to be running --
                    # instead of being swallowed the way the comment
                    # below says it is.
                    pass
                # A `__del__` THAT FAILS IS REPORTED AND SWALLOWED, exactly
                # as CPython's does (`Exception ignored in: <bound method
                # ...__del__...>`) -- to stderr, never stdout, so it never
                # touches the byte-for-byte comparison the differential
                # suite runs; and swallowed because the finalizer runs at a
                # moment of this object's choosing, not the caller's, and a
                # caller decref-ing something unrelated must not inherit a
                # `__del__`'s own bug as its own failure.
                if self.err is not None:
                    kind, msg = self.err
                    print(f"Exception ignored in: {obj!r}.__del__\n"
                         f"{kind}: {msg}", file=sys.stderr)
                self.err, self.err_value = saved_err, saved_val
        self._fire_weakrefs(h)
        self._decref_contents(obj)

    def _fire_weakrefs(self, h: int, doomed: frozenset[int] = frozenset()) -> None:
        """Run every callback registered against `h`, and forget them.

        SPLIT OUT OF `_finalize` so the cycle collector can run this pass
        FIRST, over the whole garbage set, before it clears any of it --
        which is the order CPython documents and the order a program can
        SEE. `gc` "invokes callbacks for objects that would be collected"
        before it breaks the cycle, so a callback observes a world where
        the doomed objects are already unreachable but the OTHER members
        of its own garbage set have not yet been torn apart. Firing them
        one-at-a-time inside each member's own `_finalize` instead means
        the first member's `__del__` runs before the second member's
        callback, and a program counting callbacks during `gc.collect()`
        sees a different number.

        `doomed` is the rest of the set being collected, if this is a
        collection: a callback is SKIPPED when its own `ref` is in there
        too. That is CPython's rule, not a convenience -- `handle_weakrefs`
        checks `gc_is_collecting` on the weakref itself and clears it
        without calling, because a callback reachable only from the
        garbage is about to stop existing and running it would resurrect
        (or run code out of) a half-dismantled set. Measured, not assumed:
        a `ref` stored on a cycle member, with a callback, fires zero
        times under CPython 3.14.

        `pop`, so the loop `_finalize` still reaches is a no-op for a
        handle the collector already fired: exactly-once either way.
        """
        for ref_self, callback in self._weakrefs.pop(h, ()):
            # CLEARED BEFORE THE CALLBACK RUNS, not after: CPython breaks
            # the weak reference first and calls back second, so a
            # callback that asks `r()` -- the overwhelmingly common thing
            # for one to do -- gets `None`. Left in place, `_apy_gc_collect`
            # would fire callbacks while the target's cell is still
            # standing (nothing has reached `_dead` yet at that point in
            # the collection) and `r()` would hand back the doomed object.
            self._weakref_target.pop(ref_self, None)
            if callback and ref_self not in self._dead and ref_self not in doomed:
                # `ref_self not in self._dead`: the weakref OBJECT itself
                # (a bundled `weakref.ref` instance) is tracked like any
                # other, and can die before its own target does -- nothing
                # left holding it, `del r`. CPython does not fire a
                # callback for a `ref` that no longer exists to pass to
                # it, and neither does this.
                saved_err, saved_val = self.err, self.err_value
                self.err = None
                self.err_value = None
                try:
                    # THE CALLBACK RECEIVES THE WEAKREF, not the (now
                    # gone) target -- CPython's own convention
                    # (`weakref.ref(obj, callback)`'s docs: "the weak
                    # reference object will be passed as the only
                    # parameter"), and the only object left to pass.
                    #
                    # DEREFERENCED, both of them: `_weakrefs` stores
                    # HANDLES (an `apy_value` is what the primitive was
                    # handed), and `_invoke` takes host VALUES -- passing
                    # the raw ints reached `f.bound` on an `int` and took
                    # the whole interpreter down with an AttributeError.
                    self._invoke(self._cell(callback),
                                 [self._cell(ref_self)])
                except _Trap:
                    raise
                except _UserFailed:
                    # The callback's own failure is already on the error
                    # flag, and is reported and swallowed below exactly as
                    # a failing `__del__` is -- see that comment.
                    pass
                if self.err is not None:
                    kind, msg = self.err
                    print(f"Exception ignored in weakref callback\n"
                         f"{kind}: {msg}", file=sys.stderr)
                self.err, self.err_value = saved_err, saved_val

    def _referent(self, obj) -> int | None:
        """The handle for `obj`, if `_new` ever minted one for it -- the
        lookup every heap-mutation incref/decref site uses, since a
        container holds the dereferenced Python VALUE and not the handle
        (see `_apy_seq_push` etc.): `id(obj)` recovers it through the same
        `_identity` table `_new` populates. `None` for anything untracked.
        """
        return self._identity.get(id(obj))

    def _held_by(self, obj):
        """Every TRACKED handle `obj` holds a reference to.

        ONE DEFINITION, TWO READERS, and they have to agree or the
        second one is wrong: `_decref_contents` drops exactly these when
        `obj` is finalized, and `_apy_gc_collect` subtracts exactly
        these to decide which references are internal to a cycle. A kind
        walked by one and not the other would make the collector see a
        count it could not explain.

        Conservative by kind -- anything not listed is a kind this
        scheme does not track the CONTENTS of yet (`docs/STDLIB.md`
        names them), and is left alone rather than guessed at: missing
        one leaks a shadow count (safe), inventing one finalizes
        something still alive.

        A TUPLE OR FROZENSET IS WALKED THROUGH, not reported: neither is
        interned, so `_new` never gave it a handle of its own (see
        `_INTERNED`) and there is nothing to count against it -- but it
        can still be HOLDING something tracked, and `xs = [(Foo(),)]`
        has to reach that `Foo` when `xs` dies.
        """
        out = []

        def walk(v):
            h = self._referent(v)
            if h is not None:
                out.append(h)
            elif isinstance(v, (tuple, frozenset)):
                for item in v:
                    walk(item)

        if isinstance(obj, (list, tuple, set, frozenset)):
            # LAST ELEMENT FIRST, because the order several objects
            # dying together are finalized in is observable and this is
            # the order CPython's own `list_dealloc` uses -- it walks the
            # item array backwards. Measured: a list of three, and a
            # comprehension's three results, both printed in reverse
            # there and forward here until this. A `set`'s order is its
            # own business in both, and a dict's entries stay forward,
            # which is the order CPython's dict teardown takes them in.
            for item in reversed(list(obj)):
                walk(item)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(k)
                walk(v)
        elif isinstance(obj, Instance):
            for v in obj.dict.values():
                walk(v)
            if obj.held is not None:
                out.extend(self._held_by(obj.held))
        elif isinstance(obj, Cell):
            # A CLOSURE'S BOX HOLDS WHAT IS IN IT. Counted since
            # `_apy_cell_new`/`_apy_cell_set` began increfing, and this is
            # the matching drop -- without it the cell's reference would be
            # one nothing ever released, and `x = Foo(); def f(): return x`
            # would keep `Foo` alive for the rest of the run even after both
            # the name and the closure were gone.
            walk(obj.slot)
        elif isinstance(obj, Func):
            for one in obj.cells or ():
                walk(one)
            for one in obj.defaults or ():
                walk(one)
        return out

    def callee_keeps_nothing(self, h: int) -> bool:
        """Does INVOKING the value behind `h` retain the value itself?

        THE CALLEE OF A DYNAMIC CALL IS AN ARGUMENT LIKE ANY OTHER, and
        `apy_call` is a host primitive, so its argument registers are not
        retired -- which made `alias()` on a module-level global read one
        reference high per CALL SITE and is why a function's own count was
        never trustworthy enough to cascade from.

        A PLAIN FUNCTION IS THE CASE THAT CAN BE ANSWERED HERE. Calling one
        runs interpreted Python, which counts its parameters by construction
        and keeps no reference to the function object itself, so the caller's
        temporary is free the moment the call returns.

        EVERYTHING ELSE ANSWERS FALSE, and each for its own reason. A CLASS
        builds an instance that points BACK at it through `Instance.cls`, and
        that back-pointer is not counted -- retiring the class's temporary
        could finalize a class an instance still names. A BUILTIN TYPE used
        as a constructor goes the same way. A `Native` is a host lambda this
        file cannot certify one at a time. An `Instance` with `__call__` is
        probably safe by the interpreted argument and is left out until
        something measures it, because "probably" is not the standard this
        list is held to.
        """
        try:
            v = self._cell(int(h))
        except (IndexError, ValueError):
            return False
        return isinstance(v, Func) and not getattr(v, "is_type", False)

    def _decref_contents(self, obj) -> None:
        """The cascade: dropping the last reference to a container drops
        one reference to everything it was holding, exactly as CPython's
        `tp_dealloc` walks a `list`/`dict`/instance `__dict__` on the way
        down. `_held_by` says what "everything it was holding" means.
        """
        for h in self._held_by(obj):
            self.decref(h)

    def _get(self, h, where: str):
        h = int(h)
        if h == 0:
            raise _Trap(
                f"{where}: operand is a null value -- an earlier operation "
                f"failed and its result was used without checking "
                f"apy_error_occurred()")
        try:
            return self._cell(h)
        except IndexError:
            raise _Trap(f"{where}: {h} is not a runtime value handle") from None

    def _int(self, v: int) -> int:
        """A handle for an int, interned over the small-int range.

        NOT truncated to 64 bits. The C demotes any big that fits an int64 to
        its small-int representation and keeps exactly one representation per
        value, so a Python `int` of any size is the faithful model of what it
        holds -- and the small-int interning below still reproduces the one
        thing about the C's representation that a program can observe.
        """
        if _SMALL_LO <= v <= _SMALL_HI:
            h = self._small.get(v)
            if h is None:
                h = self._small[v] = self._new(v)
            return h
        return self._new(v)

    def _bool(self, b) -> int:
        return self._true if b else self._false

    def _value(self, obj) -> int:
        """A handle for a computed result, interning what the C interns."""
        if obj is None:
            return self._none
        if obj is NotImplemented:
            # Interned for the same reason `_STOP` is: a dunder returning it
            # is compared by identity to decide whether to ask the other
            # operand.
            return self._notimplemented
        if obj is _STOP:
            # `apy_stop` IS ONE CELL IN THE C, so identity has to survive a
            # round trip through a frame slot: a loop that stores the step's
            # answer and then compares the SLOT against `apy_stop()` got a
            # fresh handle back and never saw the end of the sequence.
            return self._stop
        if obj is _SUSPEND:
            # THE SUSPENSION TOKEN IS INTERNED IN THE C, and the lowered code
            # compares against it by identity -- that is how `async for` tells
            # "suspended on an await" from "produced an item". A fresh handle
            # per suspension compared unequal to every other, so the loop took
            # the token for an item and ran forever.
            return self._suspend
        if obj is True or obj is False:
            return self._bool(obj)
        if isinstance(obj, int) and _SMALL_LO <= obj <= _SMALL_HI:
            return self._int(obj)
        # A BYTEARRAY AND A VIEW BELONG HERE TOO, and their absence was two
        # lists disagreeing: `_new` records an identity for everything in
        # `_INTERNED`, which names both, and this decided whether to LOOK --
        # so `memoryview(h).obj is h` and `xs[0] is h` for a bytearray `h`
        # were False here and True in both compiled runtimes.
        if isinstance(obj, (Instance, Class, Exc, Func, Gen, Iterator,
                            list, dict, set, frozenset, tuple, Alias, Cell,
                            bytearray, memoryview,
                            # AND EVERY IMMUTABLE VALUE, which is the whole
                            # of what was left out. See below.
                            int, str, bytes, float, complex)):
            # ONE HANDLE PER OBJECT, so `is` answers about the object and not
            # about which handle it came back through. The C compares
            # pointers, so a fresh handle for the same instance made
            # `m.__self__ is obj` False here and True in a compiled program --
            # the paths disagreeing about identity, which is the one thing
            # identity must not depend on.
            #
            # MUTABLE CONTAINERS TOO. `xs.append(xs)` then `xs[1] is xs` is
            # True in a compiled program and was False here, because reading
            # an element minted a second handle for the same list.
            #
            # AND A STR, A BYTES, A FLOAT, A COMPLEX AND A BIG INT, which
            # were the ones left out -- "compared by value everywhere that
            # matters" was not true: `xs[0] is xs[0]` and `r = lambda x: x;
            # r(v) is v` are both True in CPython and in both compiled
            # runtimes, and both were False here for every kind that minted
            # a fresh handle per read. THE OBJECT IS THE ANSWER: two reads of
            # one element hand back one Python object, and two equal strings
            # built separately hand back two -- which is exactly the line
            # CPython draws, because the host IS CPython. Interning costs no
            # retention either: `_cells` already holds every object a handle
            # was ever made for, so one handle per object holds FEWER of
            # them than one per read did.
            #
            # THE SMALL INTEGERS ARE NOT HERE. They are one cell per value
            # rather than one per object -- see `_int` -- which is the range
            # the C shares and the only part of an int's identity a program
            # can rely on.
            #
            # AND A CELL, which no program can see but which `apy_env_cell`
            # hands back on every call into a closure. A second handle for
            # the same box gets its OWN entry in `_refcount`, starting at
            # zero: the callee's register increfed that one to 1 and its
            # teardown dropped it to 0, which finalized the box -- and now
            # that `_held_by` walks a Cell's slot, that dropped the CAPTURED
            # VALUE. `def inner(): print(captured.name)` printed
            # `captured died` between two calls to `inner`, with `inner`
            # still holding it. Interning is what makes the box's count one
            # count.
            got = self._identity.get(id(obj))
            if got is not None and self._cell(got) is obj:
                return got
            made = self._new(obj)
            self._identity[id(obj)] = made
            return made
        return self._new(obj)

    # ── the error flag ──────────────────────────────────────────────────────
    def _fail(self, kind: str, msg: str) -> int:
        # First writer wins, as `apy_fail` does. A second failure downstream of
        # the first must not overwrite the report of what actually went wrong.
        if self.err is None:
            # WHERE IT HAPPENED, taken at the moment the flag goes up: by the
            # time a handler asks, its own statements have moved the cursor.
            self.pos_err = self.pos_here
            self.err = (kind, msg)
            self.err_value = None
        return 0

    def _fail_raised(self, exc) -> int:
        """An explicit `raise`, which REPLACES whatever was pending and keeps
        the exception OBJECT.

        Two differences from `_fail`, both matching the C:

        * a `raise` overrides a pending error rather than losing to it --
          `try: raise A / finally: raise B` propagates B;
        * the object survives, so `except E as e` binds what was raised and
          `e.args[0]` is the value rather than the text it rendered to.
          Rebuilding from the message made `raise E(42)` catchable as
          `E('42')`, a different value of a different type.
        """
        self.err = (exc.name, "" if not exc.has_arg
                    else self._text(exc.arg, False))
        self.err_value = exc
        return 0

    def _fail_like(self, exc: BaseException) -> int:
        """Report a Python exception as the runtime would report it.

        The message is Python's own wherever the C reproduces it, which is
        nearly everywhere -- the C's texts were written against CPython's and
        differentially tested against them.
        """
        return self._fail(type(exc).__name__, str(exc))

    # ── kind names ──────────────────────────────────────────────────────────
    @staticmethod
    def kind_name(v) -> str:
        if v is None:
            return "NoneType"
        if v is True or v is False:
            return "bool"
        if isinstance(v, Exc):
            return v.name
        if isinstance(v, Instance):
            # An instance answers with its CLASS's name, which is what makes
            # every TypeError about a user object name the user's type.
            return v.cls.name
        if isinstance(v, Class):
            # `type(C).__name__` IS THE METACLASS'S NAME when one made it.
            return v.meta.name if v.meta is not None else "type"
        if isinstance(v, Native):
            # A DUNDER WITH A RANGE IS NOT A SLOT: `(5).__round__` is a
            # `builtin_function_or_method` in CPython, because int writes the
            # method out rather than filling a slot and an optional argument
            # is exactly what a slot cannot carry.
            if v.ranged:
                return "builtin_function_or_method"
            # REACHED OFF THE TYPE, which CPython calls a DESCRIPTOR and not
            # a function or a bound method: `list.append` is a
            # `method_descriptor` and `list.__len__` a `wrapper_descriptor`.
            # Asked of CPython -- see `_descr_of`.
            if getattr(v, "descr", ""):
                named = type(_descr_of(v)).__name__
                return (named if named.endswith("_descriptor")
                        else "method_descriptor")
            # THE RUNTIME'S OWN CODE, and `Native` is a class in this file --
            # `type([1].index).__name__` answered that name, an implementation
            # detail of this compiler in a string the program prints. Python
            # says `builtin_function_or_method`, or `method-wrapper` for a
            # bound slot, which is what a dunder name means here.
            if not (len(v.name) >= 5 and v.name.startswith("__")
                    and v.name.endswith("__")):
                return "builtin_function_or_method"
            # AND THE ARITY WAS ONLY STANDING IN FOR THE REAL QUESTION,
            # which is whether the type WRITES the method out or fills a slot
            # with it -- `list.__getitem__` takes exactly one argument and is
            # written out, `tuple.__getitem__` is slotted, and nothing in
            # either signature says so. ASKED OF CPYTHON, which is the same
            # oracle the generated table is built from; the compiled halves
            # cannot ask at run time and read `apy_kind_meth_written`.
            who = v.bound if v.bound is not None else (
                v.owner if v.owner is not _NO_OWNER else None)
            if isinstance(who, _KIND_SAMPLES):
                got = getattr(who, v.name, None)
                if type(got).__name__ == "builtin_function_or_method":
                    return "builtin_function_or_method"
            # A SLOT IS FILLED ON A VALUE, never on a TYPE. What binds to a
            # type is a CLASSMETHOD -- `list.__class_getitem__` is the only
            # one here -- and CPython calls a bound one a
            # `builtin_function_or_method` whatever its name looks like.
            if isinstance(who, Class) or (isinstance(who, Func)
                                          and getattr(who, "is_type", False)):
                return "builtin_function_or_method"
            return "method-wrapper"
        if isinstance(v, Func):
            # A BUILTIN reached as a value is not a plain function:
            # `type(print).__name__` is `builtin_function_or_method`.
            if getattr(v, "is_type", False):
                return "type"
            # REACHED OFF THE TYPE, which CPython calls a DESCRIPTOR and not
            # a function: `list.append` is a `method_descriptor` and
            # `list.__len__` a `wrapper_descriptor`, and which of the two is
            # whether the type writes the method out or fills a slot with it.
            # Asked of CPython -- see `_descr_of`.
            if getattr(v, "descr", False):
                named = type(_descr_of(v)).__name__
                return (named if named.endswith("_descriptor")
                        else "method_descriptor")
            if getattr(v, "builtin", False):
                return "builtin_function_or_method"
            # A BOUND ONE IS A `method`, a type of its own in CPython:
            # `type(C().m).__name__` is `method` where `type(C.m).__name__`
            # is `function`. The receiver is the whole difference.
            return "method" if v.bound is not None else "function"
        if isinstance(v, Cell):
            return "cell"
        if isinstance(v, Super):
            return "super"
        if isinstance(v, Alias):
            # A UNION IS NOT A GENERIC ALIAS to a program that asks:
            # `type(int | str).__name__` is how it tells the two apart.
            return "typing.Union" if isinstance(v.origin, Instance)                 else "types.GenericAlias"
        if isinstance(v, Func) and getattr(v, "is_type", False):
            return "type"
        if isinstance(v, _VIEW_TYPES):
            return type(v).__name__
        if isinstance(v, slice):
            return "slice"
        if isinstance(v, Descr):
            return ("classmethod" if v.kind == PROP_CLASSMETHOD else
                    "staticmethod" if v.kind == PROP_STATICMETHOD else
                    "property")
        if isinstance(v, Gen):
            # All three share every field; only the name differs, and a
            # program reads it to tell them apart.
            if v.wrapper:
                return "coroutine_wrapper"
            if v.agen:
                return "async_generator"
            return "coroutine" if v.coro else "generator"
        if isinstance(v, Iterator):
            # A CURSOR names what MADE it: `map(str, xs)` is a `map`, which is
            # what `type(...).__name__` answers and what tells a reader why it
            # is lazy. A plain or reversed one is named after what it WALKS --
            # see `_cursor_named`.
            return {Iterator.MAP: "map", Iterator.FILTER: "filter",
                    Iterator.ENUMERATE: "enumerate",
                    Iterator.ZIP: "zip"}.get(v.mode, v.named)
        return type(v).__name__

    # ── text ────────────────────────────────────────────────────────────────
    def _under_oserror(self, name: str) -> bool:
        """Is `name` anywhere under `OSError`?

        PYTHON'S OWN CLASS TREE FIRST, which is where `_apy_exc_parent_of`
        reads the hierarchy from too -- one source here, so the two cannot
        drift. A user class declared with an `OSError` base is not in
        builtins, so `user_exc` is walked after it.
        """
        seen = set()
        at = name
        while at is not None and at not in seen:
            if at == "OSError":
                return True
            seen.add(at)
            cls = getattr(__import__("builtins"), at, None)
            if isinstance(cls, type) and issubclass(cls, BaseException):
                return issubclass(cls, OSError)
            at = self.user_exc.get(at)
        return False

    def _text(self, v, quoted: bool) -> str:
        # See `_class_module` for why the two reprs below qualify a name.
        if isinstance(v, Class):
            # PRINTING A CLASS IS THE METACLASS'S BUSINESS when it says so.
            # `repr(Colour)` is `type(Colour).__repr__(Colour)`, which is how
            # an enum prints as `<enum 'Colour'>` -- the fourth of the
            # metaclass dunders to need saying so, beside `__iter__`,
            # `__len__` and `__contains__`.
            if v.meta is not None:
                hook = v.meta.lookup("__repr__")
                if hook is not _ABSENT:
                    got = self._invoke(hook, [v])
                    if self.err is None and isinstance(got, str):
                        return got
            where = _class_module(v)
            return (f"<class '{where}.{v.name}'>" if where
                    else f"<class '{v.name}'>")
        if isinstance(v, Alias):
            return _alias_text(v)
        if isinstance(v, Func):
            # A BUILTIN TYPE NAME PRINTS AS A CLASS. `print(int)` says
            # `<class 'int'>` and it reaches here as a callable thunk, so the
            # flag -- not the kind -- decides what it is called.
            if getattr(v, "is_type", False):
                return f"<class '{v.name}'>"
            # A DESCRIPTOR NAMES THE TYPE IT CAME OFF and carries no address:
            # `<method 'append' of 'list' objects>`, and `<slot wrapper
            # '__len__' of 'list' objects>` for one the type fills a slot
            # with. Taken from CPython's own -- see `_descr_of`.
            if getattr(v, "descr", False):
                got = _descr_of(v)
                if got is not None:
                    return repr(got)
            # A BUILTIN REACHED AS A VALUE HAS NO ADDRESS IN ITS REPR.
            # `repr(len)` is `<built-in function len>` -- CPython writes no
            # address for one, because there is only ever the one. The flag
            # is the same one `type()` reads.
            if getattr(v, "builtin", False) and v.bound is None:
                return f"<built-in function {v.name}>"
            # A BOUND METHOD SHOWS WHAT IT IS BOUND TO, with the owner's name
            # before its own -- `<bound method C.m of <C object at 0x...>>`.
            if v.bound is not None:
                if getattr(v, "builtin", False):
                    return (f"<built-in method {v.name} of "
                            f"{self.kind_name(v.bound)} object at "
                            f"0x{id(v.bound):x}>")
                return (f"<bound method {self.kind_name(v.bound)}.{v.name} "
                        f"of {self._text(v.bound, True)}>")
            # THE QUALIFIED NAME, which is what CPython writes: `repr(C.m)`
            # is `<function C.m at 0x...>` and not `<function m at 0x...>`.
            return (f"<function {v.qualname if v.qualname is not None else v.name}"
                    f" at 0x{id(v):x}>")
        if isinstance(v, Instance):
            # A TYPING FORM PRINTS AS `typing.Name`. It is an instance with no
            # `__repr__`, so the default `<_SpecialForm object at 0x...>` came
            # out -- an address where CPython prints the name a program wrote.
            form = _form_name(v)
            if form is not None:
                return "typing." + form
            # `repr()` and `str()` reach `Instance.__repr__`/`__str__`, which
            # dispatch to the user's methods; the str/repr asymmetry lives
            # there so that a container printing its elements gets it too.
            return repr(v) if quoted else str(v)
        if isinstance(v, Exc):
            # A CLASS'S OWN `__str__`/`__repr__` WINS, exactly as it does
            # for an `Instance`. An exception is the one kind of object
            # whose text is nearly always overridden -- `CalledProcessError`
            # exists to say "Command 'x' returned non-zero exit status 1"
            # -- and without this the override was ignored and the ARGS
            # were printed instead, so every bundled module's exception
            # said something CPython's did not.
            #
            # BEFORE EVERY FIXED RULE BELOW, because those are
            # BaseException's own and an override is written to replace
            # them. The runtime's OWN exceptions have no `cls` and skip
            # this entirely.
            if v.cls is not None:
                hook = v.cls.find("__repr__" if quoted else "__str__")
                if isinstance(hook, (Func, Native)):
                    try:
                        # `_invoke_obj` ANSWERS A VALUE, not a handle --
                        # it ends in `self._get(result, ...)` itself, and
                        # dereferencing again read the text as a handle
                        # and tried `int()` on it.
                        return str(self._invoke_obj(hook.bind(v), []))
                    except _UserFailed:
                        # The override failed and the flag is already set.
                        # Falling through renders what BaseException would,
                        # which is better than a NULL reaching a caller
                        # that only wanted some text.
                        pass
            # MORE THAN ONE ARGUMENT PRINTS AS THE TUPLE. `str(ValueError(
            # 'a','b'))` is `('a', 'b')` and its repr is `ValueError('a',
            # 'b')` -- CPython shows the whole of `args` once there is more
            # than one, and rendering only the first dropped the rest.
            # THE NAME IT SHOWS, not the one it MATCHES on. A bundled
            # module's exception class is spliced under a mangled name and
            # renamed back, while its cells keep the mangled spelling --
            # because `except copy.Error` compiles to that and the hierarchy
            # walks by name. Mirrors the C's `apy_exc_shown`.
            held = v.cls if v.cls is not None else self.exc_class.get(v.name)
            shown_name = held.name if held is not None else v.name
            argv = getattr(v, "argv", None)
            # `[Errno 2] No such file`, and `: 'f.txt'` when a filename came
            # too. THE WHOLE FAMILY ARRIVES UNDER ITS OWN NAME -- opening a
            # missing file raises FileNotFoundError -- so this walks the
            # hierarchy rather than testing for `OSError` itself. Mirrors the
            # C's `apy_os_text` and IR's `apy_errno_text`; the three have to
            # agree, and before they did the compiled paths said
            # `[Errno 2] No such file` while `uasm run` said
            # `(2, 'No such file')`.
            if not quoted and argv is not None and 2 <= len(argv) <= 3:
                if self._under_oserror(v.name) and isinstance(argv[0], int):
                    body = f"[Errno {argv[0]}] " + self._text(argv[1], False)
                    if len(argv) == 3:
                        body += ": " + self._text(argv[2], True)
                    return body
            if argv is not None and len(argv) > 1:
                shown = self._text(tuple(argv), True)
                # The tuple's own parentheses ARE the call's.
                return f"{shown_name}{shown}" if quoted else shown
            # `str(e)` is the argument alone, `repr(e)` is `ValueError('x')`.
            if not quoted:
                # `str(KeyError('k'))` is `"'k'"` -- the REPR of the argument.
                # KeyError alone does this, so a missing key whose text is
                # empty is still visible in the report.
                return ("" if not v.has_arg else self._text(
                    v.arg, not v.rendered and v.name == "KeyError"))
            # A KeyError REBUILT FROM A FAILED LOOKUP already holds the
            # repr of the key -- that is what `rendered` records -- so
            # repr'ing it again gave `KeyError("'k'")` where CPython says
            # `KeyError('k')`. Every other type stores the plain message and
            # does want the quotes. `e.args[0]` is still the repr text; see
            # the C for why that half needs the key retained.
            twice = v.rendered and v.name == "KeyError"
            shown = "" if not v.has_arg else self._text(v.arg, not twice)
            return f"{shown_name}({shown})"
        if isinstance(v, range):
            # `range(0, 10, 2)` -- and `range(0, 3)` when the step is 1, which
            # is how CPython prints one. Before the container branch, which
            # would render its ELEMENTS and undo the laziness.
            if v.step == 1:
                return f"range({v.start}, {v.stop})"
            return f"range({v.start}, {v.stop}, {v.step})"
        if isinstance(v, (list, tuple, dict, set, frozenset)):
            # A container always shows its ELEMENTS with repr, whichever of
            # str/repr was asked of the container -- `print(['a'])` is `['a']`.
            #
            # RECURSED THROUGH `_text`, not handed to Python's `repr`. Every
            # object this file defines -- a user instance, an exception, a
            # builtin type used as a value -- has a repr that only `_text`
            # knows, and Python's printed its ADDRESS instead. Worse, `repr()`
            # on an `Instance` reaches the user's `__repr__` outside the
            # bridge's guard, so `print([P(1)])` raised out of the interpreter
            # rather than printing. Handing the job to Python looked like it
            # could not disagree with CPython, and it disagreed with it for
            # every element that was not a plain Python value.
            here = id(v)
            if here in self._rendering:
                # `xs.append(xs)`. CPython prints the ellipsis rather than
                # recurring, and that was Python's repr doing it for us.
                return ("[...]" if isinstance(v, list)
                        else "(...)" if isinstance(v, tuple) else "{...}")
            self._rendering.add(here)
            try:
                if isinstance(v, dict):
                    return "{" + ", ".join(
                        f"{self._text(k, True)}: {self._text(x, True)}"
                        for k, x in v.items()) + "}"
                body = ", ".join(self._text(x, True) for x in v)
                if isinstance(v, tuple):
                    # The TRAILING COMMA in a one-element tuple, which is what
                    # tells `(1,)` from `(1)`.
                    return f"({body},)" if len(v) == 1 else f"({body})"
                if isinstance(v, list):
                    return f"[{body}]"
                if isinstance(v, frozenset):
                    return f"frozenset({{{body}}})" if v else "frozenset()"
                # An empty set has no braces form -- `{}` is a dict.
                return "{" + body + "}" if v else "set()"
            finally:
                self._rendering.discard(here)
        if isinstance(v, str):
            return repr(v) if quoted else v
        if isinstance(v, memoryview):
            # `<memory at 0x...>` -- `memory`, not `memoryview`, which makes
            # it the one kind whose repr does not use its type name.
            return f"<memory at 0x{id(v):x}>"
        if isinstance(v, Descr):
            # A CLASSMETHOD OR STATICMETHOD SHOWS WHAT IT WRAPS, which is
            # what CPython writes. A `property` does not.
            if v.kind == PROP_PROPERTY:
                return f"<property object at 0x{id(v):x}>"
            return (f"<{self.kind_name(v)}("
                    f"{self._text(v.get, True)})>")
        if isinstance(v, Native):
            # A RUNTIME METHOD IS A `built-in method`, named after what it is
            # bound to. It answered `<uasm.objects.host.Native object
            # at 0x...>` -- a name from inside the compiler, in text a
            # program prints.
            # A DESCRIPTOR NAMES THE TYPE IT CAME OFF and carries no
            # address: `<method 'append' of 'list' objects>`. Taken from
            # CPython's own -- see `_descr_of`.
            if getattr(v, "descr", ""):
                got = _descr_of(v)
                if got is not None:
                    return repr(got)
            held = (v.bound if v.bound is not None
                    else v.owner if v.owner is not _NO_OWNER else None)
            if held is None:
                return f"<built-in function {v.name}>"
            # A BOUND SLOT SAYS `method-wrapper` AND QUOTES THE NAME:
            # `[].__len__` is `<method-wrapper '__len__' of list object at
            # 0x...>` where `[].append` is `<built-in method append of list
            # object at 0x...>`. Which of the two it is, is the question
            # `type()` already answers -- so it is asked rather than
            # restated.
            if self.kind_name(v) == "method-wrapper":
                return (f"<method-wrapper '{v.name}' of "
                        f"{self.kind_name(held)} object at 0x{id(held):x}>")
            return (f"<built-in method {v.name} of "
                    f"{self.kind_name(held)} object at 0x{id(held):x}>")
        if isinstance(v, Gen):
            # A GENERATOR NAMES THE `def` IT CAME FROM: `<generator object
            # gen at 0x...>`, and a generator expression the qualified name
            # of the scope it was written in. A coroutine and an async
            # generator take the same shape under their own kind names. The
            # four the async machinery builds have no step and fall back to
            # the bare object shape below.
            step = v.step
            if isinstance(step, Func):
                who = step.qualname if step.qualname is not None else step.name
                return (f"<{self.kind_name(v)} object {who} "
                        f"at 0x{id(v):x}>")
        if isinstance(v, (Iterator, Gen, Cell, Super)):
            # THE KIND, NOT THIS FILE'S CLASS NAME. Python's own `repr` of an
            # internal object here answered
            # `<uasm.objects.host.Iterator object at 0x...>` -- a
            # name from inside the compiler, in text a program prints. The
            # compiled runtimes say `<map object at 0x...>`.
            return f"<{self.kind_name(v)} object at 0x{id(v):x}>"
        return repr(v)

    # ── calling compiled code ───────────────────────────────────────────────
    def _template_class(self):
        """`Template`, interned so `type(a) is type(b)` for two t-strings."""
        if self._template_cls is None:
            self._template_cls = Class("Template")
        return self._template_cls

    def _interp_class(self):
        """`Interpolation`, interned for the reason `Template` is."""
        if self._interp_cls is None:
            self._interp_cls = Class("Interpolation")
        return self._interp_cls

    def _traceback_code_class(self):
        """`code`, as a traceback names it. A METHOD for `co_positions`,
        because that is how CPython spells it and a program calls it."""
        if self._tb_code_cls is None:
            cls = Class("code")
            cls.dict["co_positions"] = Native(
                "co_positions", lambda c: c.dict["_positions"])
            self._tb_code_cls = cls
        return self._tb_code_cls

    def _traceback_frame_class(self):
        if self._tb_frame_cls is None:
            self._tb_frame_cls = Class("frame")
        return self._tb_frame_cls

    def _traceback_class(self):
        if self._tb_cls is None:
            self._tb_cls = Class("traceback")
        return self._tb_cls

    def _taskgroup_class(self):
        """`TaskGroup`, interned so two groups share a type."""
        if self._taskgroup_cls is None:
            cls = Class("TaskGroup")
            cls.dict["__aenter__"] = Native(
                "__aenter__",
                lambda g: _coro_value(self, g))
            cls.dict["__aexit__"] = Native(
                "__aexit__",
                lambda g, *rest: _tgexit_coro(g))
            cls.dict["create_task"] = Native(
                "create_task",
                lambda g, coro: self._get(
                    _tg_create(self, g, coro), "create_task"))
            self._taskgroup_cls = cls
        return self._taskgroup_cls

    def _code_class(self):
        """The class `f.__code__` answers, interned so two of them share a
        type the way CPython has it."""
        if self._code_cls is None:
            self._code_cls = Class("code")
        return self._code_cls

    def _gen_frame_class(self):
        """`frame`, the class `g.gi_frame` answers, interned so two suspended
        generators share a type the way CPython has it."""
        if self._gen_frame_cls is None:
            self._gen_frame_cls = Class("frame")
        return self._gen_frame_cls

    def _member_descriptor_class(self):
        """The class a slot read through its own class answers, interned so
        `type(A.v) is type(B.w)` the way CPython has it."""
        if self._member_class is None:
            self._member_class = Class("member_descriptor")
        return self._member_class

    def _no_attr(self, obj, name: str) -> int:
        # Every builtin kind's lookup ends here, which is why the protocol
        # table is consulted HERE rather than in each of a dozen branches: a
        # kind that HAS `__iter__` reaches this line for it exactly as one
        # that has nothing does.
        found = _kind_attr(self, obj, name)
        if found is not None:
            return found
        # THE TYPE ITSELF answers for its instances: `issubclass(dict,
        # Mapping)` is a question about what a dict CAN DO, asked of `dict`
        # and never of one.
        if isinstance(obj, Func) and getattr(obj, "is_type", False):
            proto = _kind_prototype(obj.name)
            found = (None if proto is None
                     else _kind_attr(self, proto, name))
            if found is not None:
                # `__class_getitem__` IS A CLASSMETHOD and binds the TYPE,
                # which is why `list.__class_getitem__(int)` takes only the
                # key where `list.append(xs, 9)` takes a receiver first --
                # and why a bound one is a `builtin_function_or_method`
                # rather than the `method_descriptor` the stamp below would
                # make it.
                if name == "__class_getitem__":
                    # THE TYPE IS ITS OWNER, which is the receiver CPython
                    # names: `<built-in method __class_getitem__ of type
                    # object at ...>`. The OWNER and not `bound`, because
                    # binding would prepend a receiver the body does not
                    # take -- a classmethod's argument is the key alone. The
                    # body keeps the prototype in its closure, so the
                    # alias's origin is unchanged.
                    return self._new(Native(
                        name, self._get(found, name).body, owner=obj))
                # UNBOUND, because `dict.keys` is unbound in CPython too and
                # `dict.keys(d)` is how it is called -- binding it to the
                # prototype would answer for an empty dict, and a mutating
                # method would write into the prototype itself.
                return self._new(Native(name, _unbound_kind(self, name),
                                        descr=obj.name))
        # A TYPE IS NAMED, NOT DESCRIBED. CPython says `type object 'list'
        # has no attribute 'nope'` where a VALUE gets `'int' object has no
        # attribute 'nope'` -- the type's own name is what the reader needs,
        # and `kind_name` of one is only ever the word `type`.
        if isinstance(obj, Func) and getattr(obj, "is_type", False):
            return self._fail("AttributeError",
                              f"type object '{obj.name}' has no "
                              f"attribute '{name}'")
        return self._fail("AttributeError",
                          f"'{self.kind_name(obj)}' object has no "
                          f"attribute '{name}'")

    def _type_of(self, v):
        """`type(x)` as a VALUE, interned by name so `type(1) is type(2)`.

        By NAME rather than by Python class, because every exception shares one
        `Exc` class here and names one of thirty types -- interning by class
        would make `type(KeyError('k')) is type(ValueError('v'))` answer True,
        which the C does not.
        """
        if isinstance(v, Instance):
            return v.cls
        # AN EXCEPTION OF A CLASS THE PROGRAM WROTE answers that class, so
        # `type(e).__name__` and `type(e) is AppError` say what the source
        # does. Without one it falls through to the name-keyed table below,
        # which is what every exception the runtime raises itself has.
        if isinstance(v, Exc) and v.cls is not None:
            return v.cls
        # `type(C)` IS THE METACLASS when one made it. An ordinary class has
        # none recorded and reads as `type`, which is what it is.
        if isinstance(v, Class) and v.meta is not None:
            return v.meta
        key = "type" if isinstance(v, Class) else self.kind_name(v)
        if key == "type":
            # THE ONE `type` OBJECT, which is also what the bare word
            # evaluates to. A second one built here made `type(C) is type`
            # False while `print(type(C))` said `<class 'type'>` -- the worst
            # pair of answers to have. A builtin type reached as a value
            # lands here too: its kind name is `type`, because that is what
            # it is.
            return self._get(_apy_type_class(self, ()), "apy_type_class")
        got = self._types.get(key)
        if got is None:
            got = self._types[key] = Class(key, None)
        return got

    def _require_str(self, value, which: str) -> str:
        """What `__str__`/`__repr__` gave back, which must be a str.

        Converting instead would let `def __str__(self): return self` recurse
        until Python's own recursion limit fired, and that traps rather than
        reporting. The C raises here; so does this, with the same words.
        """
        if isinstance(value, str):
            return value
        self._fail("TypeError", f"{which} returned non-string "
                                f"(type {self.kind_name(value)})")
        raise _UserFailed

    def _invoke(self, f, args: list, kwrest=None, bound: bool = False):
        """Call any callable VALUE with Python-object arguments.

        One entry point for every kind, exactly as `apy_call_n` is in the C:
        a call site never has to know whether it holds a class or a function.

        `kwrest` is the `**kw` dict the caller could not place, threaded down
        rather than appended to `args`: a callee with both `*rest` and `**kw`
        packs the surplus positionals FIRST, so a dict sitting in `args` would
        be swallowed into `rest` instead of landing past it.
        """
        # AN EXCEPTION TYPE IS CALLABLE. `ValueError("v")` is resolved at the
        # call site by the frontend and never reaches here; `c = ValueError;
        # c("v")` does, and answered `ValueError() takes no arguments` about a
        # class every program constructs. `warnings.warn` is why it surfaced:
        # `raise category(message)` holds the class in a parameter.
        #
        # A TYPE CARRIES NO ARGUMENT and an instance does, which is the whole
        # distinction -- it is what stops `e = ValueError("v"); e()` from being
        # read as a second construction.
        #
        # THE SAME RULE AS THE C's, and it has to be: `apy_call_nk` grew this
        # case in the same commit. Two runtimes disagreeing about whether a
        # class is callable is exactly the drift docs/INERT-RUNTIME.md exists
        # to end.
        if isinstance(f, Class) and id(f) in self.exc_types and not f.dict:
            name = self.exc_types[id(f)]
            if (args and name == "OSError"
                    and isinstance(args[0], int)
                    and not isinstance(args[0], bool)):
                name = _ERRNO_CLASS.get(args[0], name)
            made = Exc(name, args[0] if args else None, bool(args))
            made.argv = tuple(args)
            # THE OBJECT, not a handle. `_invoke` speaks in objects and the
            # `_apy_*` bindings speak in handles; returning what
            # `_apy_make_excn` returns printed the handle as an integer --
            # `11` where the message should have been.
            return made
        # AND A CLASS THE PROGRAM WROTE BY SUBCLASSING ONE. `class
        # AppError(ValueError)` has a body, so the test above declines it, and
        # `_instantiate` built an ordinary Instance instead: `str(e)` read
        # `<AppError object at 0x...>` and `raise e` said `exceptions must
        # derive from BaseException, not 'AppError'` -- about a class whose
        # `class` statement names ValueError as its base. Only through a
        # VARIABLE, because the frontend resolves the written spelling at the
        # call site, which is what kept it hidden.
        #
        # `is f` rather than a membership test: a program may bind an ordinary
        # class to a name an exception class also has, and the registered
        # object is the one this holds for.
        if isinstance(f, Class) and self.exc_class.get(f.name) is f:
            made = Exc(f.name, args[0] if args else None, bool(args))
            made.argv = tuple(args)
            made.cls = f
            # ITS OWN `__init__` STILL RUNS, over the arguments the call
            # supplied and after the defaults are in place -- the same order
            # `_exc_construct` uses for the written spelling, so the two ways
            # of naming the class build the same object. Called through
            # `_invoke_obj` and not `_exc_construct`, which answers a HANDLE
            # where everything here speaks in objects.
            init = f.find("__init__")
            if isinstance(init, (Func, Native)):
                self._invoke_obj(init.bind(made), args, kwrest, bound)
            return made
        if isinstance(f, Class):
            if f.meta is not None:
                # THE METACLASS DECIDES WHAT CALLING THE CLASS DOES, if it
                # says so: `type(C).__call__(C, ...)` is what `C(...)` means,
                # and it is how `ABCMeta` refuses to instantiate a class with
                # abstract methods. Looked for only when there IS a metaclass
                # -- the default is the allocate-and-init below.
                hook = f.meta.lookup("__call__")
                # THE DEFAULT `type.__call__` IS THE FALL-THROUGH, not a hook
                # to run. Every metaclass inherits it, so a metaclass that
                # writes no `__call__` of its own still found one here -- and
                # `_invoke_obj` has no case for that Native the way `_invoke`
                # does, so it called its body directly and `C()` answered
                # None. Skipping it lands on `_instantiate` below, which IS
                # what `type.__call__` means.
                if isinstance(hook, (Func, Native)) and not (
                        isinstance(hook, Native)
                        and hook.name == "<type.__call__>"):
                    return self._invoke_obj(hook, [f] + list(args), kwrest)
            return self._instantiate(f, args, kwrest, bound)
        if isinstance(f, Native) and f.name == "<type.__call__>":
            # `type.__call__(cls, ...)` -- the ordinary instantiation with the
            # metaclass hook deliberately skipped, which is what a metaclass's
            # own `__call__` delegates to. Consulting the hook from here would
            # be that `__call__` calling itself forever.
            if f.bound is not None:
                return self._instantiate(f.bound, args, kwrest)
            if not args:
                self._fail("TypeError", "type.__call__() needs a type")
                raise _UserFailed
            return self._instantiate(args[0], args[1:], kwrest)
        if isinstance(f, Instance):
            return self._invoke_rest(f, args, kwrest, bound)
        return self._invoke_rest(f, args, kwrest, bound)

    def _instantiate(self, f, args: list, kwrest=None, bound: bool = False):
        """`C(...)` -- allocate, then run `__init__` if there is one.

        SEPARATE FROM `_invoke` because `type.__call__` is exactly this and
        nothing else: a metaclass that overrides `__call__` and ends by
        delegating upward has to reach the default without re-entering it.
        """
        # A BUILTIN TYPE OBJECT IS CALLABLE. `type(5)()` is `0` and
        # `x.__class__()` is an empty one of whatever `x` is -- both are the
        # ordinary way to make a fresh value of a type you were handed, and a
        # generic copy or merge helper does exactly that. A type INVENTED for
        # a kind the program never NAMED had no constructor at all and
        # allocated an instance of a nameless class.
        made = _builtin_ctor(self, f, args, kwrest)
        if made is not _NO_CTOR:
            return made
        if True:
            maker = f.find("__new__")
            if isinstance(maker, (Func, Native)):
                # `__new__` IS AN IMPLICIT STATICMETHOD: it receives the CLASS
                # as its first argument, not an instance, so it is called
                # unbound with the class pushed in front. It was ignored
                # entirely -- the instance was allocated and `__new__` never
                # ran, which is a wrong answer rather than a missing feature.
                obj = self._invoke_obj(maker, [f] + list(args), kwrest, bound)
                # `__init__` RUNS ONLY IF `__new__` RETURNED ONE OF THESE.
                # Returning something else is how a `__new__` deliberately
                # bypasses initialisation, and CPython honours that.
                # For a METACLASS the thing `__new__` answered is a CLASS
                # whose metaclass is this one, which is the same test through
                # `_type_of`. Comparing only against `Instance.cls` skipped a
                # metaclass's `__init__` entirely.
                if self._type_of(obj) is not f:
                    return obj
            else:
                obj = Instance(f, self)
            # PINNED FROM HERE TO WHEN THIS FUNCTION'S RETURN VALUE FINDS
            # A REAL HOME. `obj` may not have a handle yet, and is about
            # to be passed into `__init__` (below) where, absent a pin,
            # its ONLY tracked reference for that call's whole length
            # would be the `self` parameter binding `_call` itself
            # creates and then drops at frame teardown -- landing on
            # zero and finalizing the object under construction before
            # `_instantiate` has even returned it to whoever asked for
            # `C(...)`. Deliberately NOT unpinned before this function
            # returns: the pin travels with the handle up through
            # `_invoke`/`_apy_call` and is lifted by `ObjectHost.settle`,
            # which `interpreter.py`'s `put()` calls once the value is
            # actually stored into a register -- a real, counted owner.
            h = self._value(obj)
            self._protect(h)
            init = f.find("__init__")
            if isinstance(init, (Func, Native)):
                self._invoke_obj(init.bind(obj), args, kwrest, bound)
            elif (args or kwrest) and not isinstance(maker, (Func, Native)):
                # A CLASS EXTENDING A BUILTIN INHERITS ITS CONSTRUCTOR.
                # `class L(list): pass` then `L([1, 2, 3])` is a list of
                # three, because `list.__init__` is what the empty body left
                # in place -- and reporting `L() takes no arguments` about it
                # names the wrong thing entirely: the class HAS a constructor,
                # inherited, and the arguments are exactly what it wants.
                #
                # `builtin_kind()` IS THE HOST TYPE, which is what makes this
                # two lines rather than a second constructor per kind: the
                # instance already carries a real `dict`/`list`/`tuple`/`set`/
                # `str`, so building the initial one is calling that type.
                #
                # NOT WHEN THE CLASS WROTE `__new__`. That constructor has
                # already decided what the instance holds -- a `namedtuple`
                # packs its arguments into one tuple -- and filling it a
                # second time from the RAW arguments would either undo the
                # packing or, for `P(1, 2)`, ask `tuple(1, 2)` for a tuple of
                # two separate things and report the arity as an error.
                # CPython draws the same line: `object.__init__` complains
                # about surplus arguments only when `__new__` is not
                # overridden.
                kind = f.builtin_kind() if isinstance(f, Class) else None
                if kind is None:
                    self._fail("TypeError", f"{f.name}() takes no arguments")
                    raise _UserFailed
                # THE ARGUMENTS ARE HOST VALUES ALREADY, not handles --
                # `_instantiate` is reached from `_invoke`, which unwraps.
                # Putting a `_get` here read `int(a_list)` and reported the
                # builtin's complaint about an int.
                try:
                    obj.held = kind(*[_content_of(one) for one in args],
                                    **(kwrest or {}))
                except _UserFailed:
                    raise
                except Exception as exc:
                    # THE BUILTIN'S OWN REFUSAL, forwarded rather than
                    # reworded: `L(5)` is a TypeError about an int not being
                    # iterable, and that is the message CPython gives.
                    self._fail(type(exc).__name__, str(exc))
                    raise _UserFailed
            return obj

    def _invoke_rest(self, f, args: list, kwrest=None, bound: bool = False):
        """Everything `_invoke` dispatches that is not a class."""
        if isinstance(f, Instance):
            # THROUGH `_invoke_obj`, not `_send`: a callable instance's
            # `__call__` may take `**kw`, and `_send` has no way to carry the
            # keywords the caller could not place -- they were silently lost.
            m = f.cls.find("__call__")
            if not isinstance(m, Func):
                self._fail("TypeError",
                           f"'{f.cls.name}' object is not callable")
                raise _UserFailed
            return self._invoke_obj(m.bind(f), args, kwrest, bound)
        if not isinstance(f, (Func, Native)):
            self._fail("TypeError",
                       f"'{self.kind_name(f)}' object is not callable")
            raise _UserFailed
        return self._invoke_obj(f, args, kwrest, bound)

    def _invoke_obj(self, f: "Func", args: list, kwrest=None,
                    bound: bool = False):
        """Run one compiled function. THE re-entry into the interpreter.

        `env` is the function object itself and it is passed FIRST, which is
        the calling convention every backend shares -- see the comment on
        `apy_func_new` in objects/csource.py. The env is a fresh handle for the
        BOUND method rather than for the underlying function, because the
        receiver travels in the value and the callee reads its cells out of
        whichever object the call came through.
        """
        if isinstance(f, Native):
            # A NATIVE has no compiled body to enter and no signature to match
            # against: the receiver, if it was bound, is simply the first
            # argument. Tested here rather than at each call site, so every
            # route into a callable reaches one the same way.
            given = ([f.bound] if f.bound is not None else []) + list(args)
            # THE POSITIONAL BOUND, for the two methods whose is narrower
            # than their total. Only where the caller has NOT already
            # matched names to slots -- after that the count says nothing
            # about how the call was written. See `_meth_positional`.
            if not bound and _meth_positional(self, f, args):
                raise _UserFailed
            try:
                return f.body(*given)
            except TypeError as exc:
                # THE LAMBDA'S NAME IS AN IMPLEMENTATION DETAIL OF THIS FILE.
                # A wrong argument count against a runtime-owned callable
                # reported `_kind_attr.<locals>.<lambda>() takes 1 positional
                # argument but 3 were given` -- a name from here, in a message
                # the program sees and may print. Rewritten to name the METHOD
                # instead, and ONLY when the failure is the body's own
                # signature: a TypeError raised BY the body is the program's
                # and travels unchanged.
                where = getattr(f.body, "__qualname__", "")
                text = str(exc)
                if not where or not text.startswith(where + "()"):
                    raise
                # CPYTHON WORDS A BUILTIN DIFFERENTLY from a function it
                # compiled: `expected 0 arguments, got 2`, with no
                # parentheses -- and it names the method only when the method
                # has a name a program wrote. A bound SLOT is anonymous, the
                # same line `type()` draws between `method-wrapper` and
                # `builtin_function_or_method`.
                # A BOUND BUILTIN METHOD IS WORDED BY ITS RECEIVER, and
                # there are nine wordings rather than the three below --
                # `[].append()` is `list.append() takes exactly one argument
                # (0 given)` in CPython. See `_meth_arity_words`. Anything
                # the generated table does not know -- a generator's `send`,
                # a bound slot, a native this runtime invented -- falls
                # through to the plainer wording.
                who = (f.bound if f.bound is not None
                       else f.owner if f.owner is not _NO_OWNER else None)
                packed = (KINDMETH_WORDS.get((f.name, _meth_kind(who)))
                          if who is not None else None)
                if packed:
                    _meth_arity_words(self, who, f.name, packed, len(args))
                    raise _UserFailed
                code = getattr(f.body, "__code__", None)
                count = code.co_argcount if code else 0
                ndef = len(getattr(f.body, "__defaults__", None) or ())
                if f.ranged:
                    most, least = count, count - ndef
                else:
                    # THE DEFAULTS IN THESE LAMBDAS ARE CLOSURE CAPTURES
                    # (`lambda o, _w=want: ...`) and not optional parameters;
                    # only a native that says it is ranged really has one.
                    most = least = count - ndef
                got = len(args)
                # A FIXED ARITY IS SAID PLAINLY whichever way the count is
                # wrong; only a method with a RANGE says which end it missed.
                if least == most:
                    how = (f"expected {most} "
                           f"argument{'' if most == 1 else 's'}")
                elif got < least:
                    how = (f"expected at least {least} "
                           f"argument{'' if least == 1 else 's'}")
                else:
                    how = (f"expected at most {most} "
                           f"argument{'' if most == 1 else 's'}")
                # A DUNDER WITH A FIXED ARITY IS A BOUND SLOT, and CPython
                # leaves a slot anonymous here. One with a RANGE is a method
                # CPython wrote out rather than a slot, and those are named.
                lead = ("" if f.name.startswith("__") and least == most
                        else f.name + " ")
                self._fail("TypeError", f"{lead}{how}, got {got}")
                raise _UserFailed
        slots = [f.bound] if f.bound is not None else []
        declared = f.arity - (1 if f.vararg else 0) - (1 if f.kwarg else 0)
        # WHERE POSITIONS STOP. A keyword-only parameter is declared but not
        # reachable by position, so a surplus argument belongs to `*rest` --
        # or is an error -- rather than landing in it.
        #
        # `bound` means the caller ALREADY matched names to slots, so every
        # argument here belongs where it is: re-applying the limit would
        # truncate a list `apy_call_kw` had just completed and then refill the
        # tail from defaults, discarding what the keywords supplied.
        byslot = declared if bound else declared - f.kwonly
        take = len(args)
        if len(slots) + take > byslot:
            take = max(0, byslot - len(slots))
        slots += list(args[:take])
        # A missing trailing argument comes from the default the `def`
        # evaluated -- one object, shared across calls, which is what
        # `aliasing/default-argument-is-shared` measures.
        while len(slots) < declared:
            d = _fn_default(f, len(slots))
            if d is _NO_DEFAULT:
                break
            slots.append(d)
        if f.vararg:
            slots.append(tuple(args[take:]))
        # `**kw` is the LAST parameter and is bound even when empty: `def
        # f(**kw)` called as `f()` gets `{}`, not nothing.
        if f.kwarg:
            slots.append(dict(kwrest) if kwrest else {})
        # A SURPLUS POSITIONAL IS AN ERROR, not something to drop. The
        # packing above CAPS what it copies at the positional capacity, so
        # the length compared below can only ever come out too SMALL -- which
        # is exactly how `two(0, 1, 2)` answered `(0, 1)` instead of
        # refusing, on every path there is. A `*rest` is what a surplus is
        # FOR, and `bound` says the caller matched names to slots and counted
        # them itself, so neither is one.
        if not bound and not f.vararg:
            byslot = f.arity - (1 if f.kwarg else 0) - f.kwonly
            if (1 if f.bound is not None else 0) + len(args) > byslot:
                _fn_arity_failure(
                    self, f, (1 if f.bound is not None else 0) + len(args), 0)
                raise _UserFailed
        if len(slots) != f.arity:
            _fn_arity_failure(self, f,
                              (1 if f.bound is not None else 0) + len(args), 0)
            raise _UserFailed
        fn = self._interp.module.functions[f.code & ~_FUNC_TAG]
        # THE INTERNED HANDLE FOR AN UNBOUND FUNCTION, a fresh one only for a
        # BOUND method. `_new` mints a new cell every time and gives it its
        # own `_refcount` entry starting at zero, so a closure reached
        # through here got a SECOND identity per call -- and `_call`'s
        # parameter incref and teardown decref took that second count from
        # one to zero, finalizing the function on every call. Invisible until
        # `_held_by` learned to cascade from a `Func` to its cells, at which
        # point it dropped the captured value while the closure was still
        # live. A bound method is a fresh object per access and must keep
        # its own handle: the receiver travels in the value.
        env = self._value(f) if f.bound is None else self._new(f)
        values = [self._value(s) for s in slots]
        # PINNED FOR THE CALL. `env` and every argument handle here may
        # exist ONLY as this host frame's own local right now -- nothing
        # has stored them into a register yet, which is the only place
        # this scheme's `put()` choke point would count a reference. The
        # clearest case is `self` for a constructor: `_instantiate` mints
        # its handle and reaches `_invoke_obj` in the very same
        # expression, so the object's ONLY tracked reference for the
        # length of `__init__` is the parameter binding `_call` itself
        # creates below -- which its own frame teardown correctly drops
        # back out when `__init__` returns. Without a pin that drop lands
        # on zero and finalizes the object being constructed, before
        # `_instantiate` has even returned it to whoever asked for a new
        # `Foo()`. Pinning costs nothing when a caller-side reference
        # already exists (the pin just outlives a decref that would not
        # have reached zero anyway) and is exactly what closes the gap
        # when one does not yet.
        self._protect(env)
        for v in values:
            self._protect(v)
        try:
            result = self._interp._call(fn, [env] + values)
        finally:
            self._unprotect(env)
            for v in values:
                self._unprotect(v)
        if self.err is not None:
            # The callee failed. Its report is already the first one, so there
            # is nothing to add -- only to stop, so that a NULL result never
            # reaches an operator as though it were a value.
            raise _UserFailed
        return self._get(result, f.name)

    # ── dispatch ────────────────────────────────────────────────────────────
    def call(self, name: str, args: list):
        """Dispatch one runtime symbol to its host binding.

        THE BACKSTOP FOR A HOST METHOD THAT RAISES lives here, at the one
        place every binding is reached, rather than in each of the two hundred
        bindings. Most of them wrap their own call and convert the failure;
        the ones that forget used to take the WHOLE INTERPRETER down with a
        Python traceback, where CPython and both compiled paths raise an
        ordinary catchable exception.

        THAT IS NOT A SMALL FAILURE. A sweep of 9,339 generated expressions
        stopped at case 631 -- `'abc'.index(5)`, whose binding calls
        `v.index(item)` with no guard at all -- so 93% of the run produced no
        output and read as divergence. Two separate bindings did it, which is
        what says the fix belongs here and not in either of them.

        `_HOST_RAISES` AND NOT `Exception`, for the reason given where it is
        defined: an AttributeError here is this file's own bug, and burying it
        as the program's error is the failure mode that would cost most.
        """
        fn = _TABLE.get(name)
        if fn is None:
            return NOT_MINE
        try:
            return fn(self, args)
        except _HOST_RAISES as exc:
            return self._fail_like(exc)


# Imported late so this module can be read without chasing the interpreter's
# own definitions; `Trap` is the interpreter's "the program did something
# undefined" signal and this file raises exactly one kind of it.
from ..ir.interpreter import Trap as _Trap, _FUNC_TAG  # noqa: E402

#: "not a builtin type" -- distinct from every value a constructor can
#: answer, including None and 0, which `bool()` and `int()` really do give.
_NO_CTOR = object()

#: The builtin type NAMES and what calling one with nothing and with one
#: argument answers. BY NAME AND NOT BY A PROTOTYPE'S KIND: `bool` and `int`
#: share a prototype -- the attribute question cannot tell them apart and
#: does not need to -- and `type(True)()` is `False`, not `0`.
#:
#: CPYTHON'S OWN CONSTRUCTOR where the answer and every refusal already
#: agree, and a RUNTIME SYMBOL where they do not: the three text ones go
#: through `_BY_SYMBOL` below, because this compiler's `str(x)` asks the
#: value for its text the way the compiled halves do.
_BUILTIN_CTORS = {
    "int": (lambda: 0, int), "bool": (lambda: False, bool),
    "float": (lambda: 0.0, float), "str": (lambda: "", None),
    "bytes": (lambda: b"", None), "bytearray": (lambda: bytearray(), None),
    "list": (list, list), "tuple": (tuple, tuple), "dict": (dict, dict),
    "set": (set, set), "frozenset": (frozenset, frozenset),
    "complex": (complex, complex),
    # `range(stop)` and `slice(stop)` PUT THE LONE ARGUMENT IN STOP, and
    # neither has an empty form at all -- `_ctor_make` answers both, and the
    # entry here is what says the name is a builtin type.
    "range": (None, None), "slice": (None, None),
    "memoryview": (None, memoryview),
}

#: THE ONE-ARGUMENT FORMS GO THROUGH THE RUNTIME, every one of them, and
#: not through CPython's constructor of the same name.
#:
#: The argument may be an INSTANCE of a user class, which is an interpreter
#: object and not the thing it stands for: `int(Fraction(6, 3))` has to reach
#: that class's `__int__`, and CPython's own `int` sees only the host object
#: and reports `'Fraction' object cannot be interpreted as an integer`. The
#: compiled halves have always gone through these symbols; the interpreter
#: reached them only for the three text ones, which is what made `f = int`
#: then `f(fraction)` fail where the written `int(fraction)` answered.
_BY_SYMBOL = {
    "int": "apy_to_int", "float": "apy_to_float", "bool": "apy_to_bool",
    "str": "apy_str", "bytes": "apy_to_bytes", "bytearray": "apy_to_bytearray",
    "dict": "apy_to_dict", "set": "apy_to_set", "frozenset": "apy_to_frozenset",
    "memoryview": "apy_memoryview",
}


def _ctor_run(h, symbol, values, raw=()):
    """One runtime entry point, with values in and a value out.

    The host's constructors answer VALUES -- that is what `_instantiate`
    answers everywhere else -- and the runtime symbols take and answer
    HANDLES, so the wrapping happens here rather than at each call.

    `raw` IS THE TAIL THAT IS NOT A VALUE. `apy_bytes_ctor` ends in an
    `int64_t mut`, a machine word rather than an object, and handing it a
    handle made every `bytes(s, "utf-8")` answer a bytearray -- a handle is
    a nonzero index, so the flag read as set.
    """
    got = _TABLE[symbol](h, [h._new(v) for v in values] + list(raw))
    if not got:
        raise _UserFailed
    return h._get(got, symbol)


def _ctor_make(h, tn, vals):
    """The positional forms of a builtin constructor, once its keywords are
    folded into slots. The same arms `apy_ctor_make` holds in the C, so the
    interpreter and the compiled runtimes answer one thing."""
    n = len(vals)
    if tn in ("range", "slice"):
        start = vals[0] if n > 1 else None
        stop = vals[1] if n > 1 else vals[0]
        step = vals[2] if n > 2 else None
        if tn == "slice":
            return slice(start, stop, step)
        return range(*((stop,) if n == 1 else (start, stop)
                       if n == 2 else (start, stop, step)))
    if n == 0:
        return _BUILTIN_CTORS[tn][0]()
    if n == 1:
        if tn in _BY_SYMBOL:
            # `bytes(xs)` ASKS FOR A LENGTH HINT and `bytearray(xs)` does
            # not: CPython builds the immutable one from the iterator in a
            # single allocation and grows the mutable one. None of the other
            # names here asks. See `_apy_length_hint`.
            if tn == "bytes" and not _apy_length_hint(h, [h._new(vals[0])]):
                raise _UserFailed
            return _ctor_run(h, _BY_SYMBOL[tn], vals)
        if tn in ("list", "tuple"):
            # `tuple(t)` ON A TUPLE IS `t`, as the written form already
            # answers -- see `_dyn_convert_sequence`.
            if tn == "tuple" and type(vals[0]) is tuple:
                return vals[0]
            # ONLY A LIST ASKS FOR A LENGTH HINT. `tuple(x)` does not, in
            # CPython, and the difference is measurable from inside the
            # program -- see `_apy_length_hint`.
            if tn == "list" and not _apy_length_hint(h, [h._new(vals[0])]):
                raise _UserFailed
            # BUILT AND FILLED, the way `apy_call_kind` fills one: `apy_extend`
            # is the one thing that already knows how to drain every kind of
            # source -- a generator, a dict (its keys), a str, an instance.
            made = _TABLE["apy_tuple_new" if tn == "tuple"
                          else "apy_list_new"](h, [4])
            if not _TABLE["apy_extend"](h, [made, h._new(vals[0])]):
                raise _UserFailed
            return h._get(made, tn)
        if tn == "complex":
            # NONE FOR "NOT GIVEN", not the number 0: `complex(x)` asks the
            # class through `__complex__` and `complex(x, 0)` builds from
            # parts, and a zero default made the two indistinguishable.
            return _ctor_run(h, "apy_complex_of", [vals[0], None])
        return _BUILTIN_CTORS[tn][1](vals[0])
    if n == 2 and tn == "int":
        return _ctor_run(h, "apy_to_int_base", vals)
    if n == 2 and tn == "complex":
        return _ctor_run(h, "apy_complex_of", vals)
    if tn == "str" and n in (2, 3):
        return _ctor_run(h, "apy_str_ctor", list(vals) + [None] * (3 - n))
    if tn in ("bytes", "bytearray") and n in (2, 3):
        return _ctor_run(h, "apy_bytes_ctor", list(vals) + [None] * (3 - n),
                         raw=(1 if tn == "bytearray" else 0,))
    return _NO_CTOR


def _ctor_call(h, tn, args, kwrest):
    """A builtin type constructor: the keywords folded, then the positions.

    THE FRONTEND'S OWN FOLD, read here rather than restated -- `int(x,
    base=16)` written out and `type(5)(x, base=16)` reached through a value
    are arranged against one signature and refused in one set of words. See
    `CTOR_PARAMS`.
    """
    if tn not in _BUILTIN_CTORS:
        return _NO_CTOR
    try:
        plan = fold_ctor_keywords(tn, len(args), list(kwrest or ()))
    except KeywordError as exc:
        h._fail("TypeError", str(exc))
        raise _UserFailed
    if plan is None:
        vals = list(args)
        if tn == "dict" and kwrest:
            made = dict(args[0]) if args else {}
            made.update(kwrest)
            return made
    else:
        vals = []
        for kind, value in plan:
            vals.append(args[value] if kind == "pos"
                        else kwrest[value] if kind == "kw" else value)
    try:
        return _ctor_make(h, tn, vals)
    except _HOST_RAISES as exc:
        h._fail_like(exc)
        raise _UserFailed


def _apy_kw_owner(h, a):
    """`"abc".upper(x=1)` -- a method that takes no keyword at all. CPython
    names the OWNER here, and the owner is the receiver's kind."""
    recv = h._get(a[0], "apy_kw_owner")
    return h._fail("TypeError", f"{h.kind_name(recv)}."
                                f"{h._get(a[1], 'apy_kw_owner')}() takes no "
                                f"keyword arguments")


def _apy_kw_none(h, a):
    """The same refusal, for a `**` mapping that may or may not hold one. The
    empty case is the common one and passes straight through."""
    if not h._get(a[2], "apy_kw_none"):
        return h._new(None)
    return _apy_kw_owner(h, a)


def _kw_twice(h, recv, meth: str, key):
    """`f(**{"k": v}, k=w)` -- the same name twice, once through a mapping and
    once written out. CPython refuses it whichever order the two come in, and
    NAMES THE OWNER. The C twin is `apy_kw_twice`."""
    return h._fail("TypeError",
                   f"{h.kind_name(recv)}.{meth}() got multiple values for "
                   f"keyword argument {key!r}")


def _apy_kw_put(h, a):
    """One written keyword into the bag, refused if the bag already holds it.

    OFFERED ONE KEY AT A TIME, because a merge cannot see a collision: the
    later key simply wins, and `"a,b".split(**{"sep": ","}, sep=";")` answered
    `['a,b']` with nothing to mark it. Two WRITTEN names cannot collide --
    that is a SyntaxError -- so any repeat here is the mapping meeting a name.
    """
    bag = h._get(a[0], "apy_kw_put")
    key = h._get(a[1], "apy_kw_put")
    val = h._get(a[2], "apy_kw_put")
    recv = h._get(a[3], "apy_kw_put") if a[3] else None
    meth = h._get(a[4], "apy_kw_put")
    if key in bag:
        return _kw_twice(h, recv, meth, key)
    bag[key] = val
    return h._none


def _apy_kw_merge(h, a):
    """A `**` mapping into the bag, key by key. See `_apy_kw_put`."""
    bag = h._get(a[0], "apy_kw_merge")
    src = h._get(a[1], "apy_kw_merge")
    recv = h._get(a[2], "apy_kw_merge") if a[2] else None
    meth = h._get(a[3], "apy_kw_merge")
    if not isinstance(src, dict):
        # NOT A MAPPING AT ALL, which `**` requires. Left to `apy_update`,
        # which already words it.
        return _TABLE["apy_update"](h, [a[0], a[1]])
    for key, val in src.items():
        if isinstance(key, str) and key in bag:
            return _kw_twice(h, recv, meth, key)
        bag[key] = val
    return h._none


def _apy_kw_check(h, a):
    """`"aaa".replace("a", "b", **opts)` -- what the mapping holds, checked
    where it is finally known.

    A `**` MAPPING CANNOT BE FOLDED AT COMPILE TIME: its keys are a run-time
    value. The frontend emits one `apy_dict_get_or` per parameter against the
    names `METHOD_PARAMS` records; this is the half that cannot be decided
    there -- a key that names no parameter, or one naming a slot a positional
    already filled. The wordings are CPython's, the same ones `fold_keywords`
    raises for the written spelling.

    A PARAMETER CPYTHON MARKS POSITIONAL-ONLY ARRIVES AS AN EMPTY STRING: a
    key can never name one, and leaving a hole would misalign every slot
    after it.
    """
    bag = h._get(a[0], "apy_kw_check")
    known = list(h._get(a[1], "apy_kw_check"))
    meth = h._get(a[2], "apy_kw_check")
    argc = int(a[3])
    # TOO MANY BEATS EVERY OTHER COMPLAINT, counting the keywords in. A call
    # with no positionals at all is worded as KEYWORD arguments -- the same
    # split `fold_keywords` makes for the written spelling.
    if argc + len(bag) > len(known):
        plural = "" if len(known) == 1 else "s"
        if argc == 0:
            return h._fail("TypeError",
                           f"{meth}() takes at most {len(known)} keyword "
                           f"argument{plural} ({len(bag)} given)")
        return h._fail("TypeError",
                       f"{meth}() takes at most {len(known)} argument{plural} "
                       f"({argc + len(bag)} given)")
    for key in bag:
        if not isinstance(key, str):
            return h._fail("TypeError", "keywords must be strings")
        at = next((i for i, p in enumerate(known) if p and p == key), -1)
        if at < 0:
            near = _suggest(key, [p for p in known if p])
            return h._fail("TypeError",
                           f"{meth}() got an unexpected keyword argument "
                           f"{key!r}"
                           + (f". Did you mean {near!r}?" if near else ""))
        if at < argc:
            return h._fail("TypeError",
                           f"argument for {meth}() given by name ({key!r}) "
                           f"and position ({at + 1})")
    return h._new(None)


def _apy_ctor_call(h, a):
    """A BUILTIN TYPE REACHED AS A VALUE: `f = int` then `f("ff", 16)`,
    `map(int, xs)`, `defaultdict(list)`. The thunk the frontend synthesises
    for a named builtin type calls exactly this, so the value form, the
    written form and the type-object form are one implementation."""
    tn = h._get(a[0], "apy_ctor_call")
    args = list(h._get(a[1], "apy_ctor_call"))
    # AN EMPTY BAG IS NO BAG: the thunk always declares the `**kw` slot, so
    # an ordinary `list(xs)` arrives with an empty dict in it.
    kwrest = h._get(a[2], "apy_ctor_call") if a[2] else None
    made = _ctor_call(h, tn, args, kwrest or None)
    if made is _NO_CTOR:
        return h._fail("TypeError", f"{tn}() takes no arguments")
    return h._new(made)


def _builtin_ctor(h, f, args, kwrest):
    """`type(5)()` and `x.__class__(v)`, or `_NO_CTOR` for anything else.

    ONLY FOR A TYPE WITH NOTHING OF ITS OWN -- no `__new__`, no `__init__`,
    no metaclass -- which is exactly the shape a type invented for a builtin
    kind has and nothing a program writes.
    """
    name = getattr(f, "name", None)
    if (name not in _BUILTIN_CTORS
            or getattr(f, "meta", None) is not None
            or f.find("__new__") is not None
            or f.find("__init__") is not None):
        return _NO_CTOR
    return _ctor_call(h, name, list(args),
                      dict(kwrest) if kwrest else None)


def _user(h, body, fail=0):
    """Run `body`, turning a user method's failure into the C's NULL return.

    Every entry point that can reach compiled code needs this: `_UserFailed`
    means the error flag is already set and the only thing left is to stop.
    A TypeError raised by the bridge itself -- "has no len()" -- is a report
    this file owes, and is converted with the message Python built.
    """
    try:
        return body()
    except _UserFailed:
        return fail
    except (TypeError, ValueError, ZeroDivisionError, OverflowError,
            KeyError, IndexError) as e:
        return h._fail_like(e)


# ── construction ────────────────────────────────────────────────────────────

def _apy_none(h, a):
    return h._none


def _apy_ellipsis(h, a):
    """`...` and the name `Ellipsis` -- one cell, so `is` answers True."""
    return h._ellipsis


def _apy_notimplemented(h, a):
    """`NotImplemented` -- ONE CELL, so `x is NotImplemented` answers True.

    Python's own singleton is used rather than a stand-in of this file's, so
    a dunder returning it is the same object the comparison below tests for.
    """
    return h._notimplemented


def _apy_from_bool(h, a):
    return h._bool(int(a[0]) != 0)


def _apy_from_int(h, a):
    return h._int(int(a[0]))


def _apy_obj_alloc(h, a):
    """`apy_obj_alloc(kind)` -- the C runtime's allocation hook.

    IT HAS NO HONEST ANSWER HERE, and refusing is the honest answer. The C
    hands back a 160-byte cell whose payload the caller then writes through a
    pointer; this host has no cells and no pointers, only handles into a Python
    list, so there is nothing to return that the caller could write into.

    Nothing reaches it. `apy_alloc` is `static` C and never enters the IR, and
    the ported runtime that calls this is not run by the interpreter at all
    (see `Interpreter._call`: the host owns every `apy_*` it claims). The
    binding exists because every exported symbol must have one -- a compiled
    program can call what `uasm run` cannot, and a MISSING binding fails
    as `unknown host function`, which says nothing about why.
    """
    raise _Trap(
        "apy_obj_alloc: the interpreter allocates objects as host values, not "
        "as cells, so there is no address to hand back. This is the ported "
        "allocator's entry point (runtime/arena.py) and the interpreter runs "
        "the host object runtime instead -- see Interpreter._call.")


def _apy_from_float(h, a):
    return h._new(float(a[0]))


def _decode_cell(raw: bytes) -> str:
    """The str a run of the runtime's bytes stands for.

    TWO ERROR HANDLERS, TRIED IN ORDER, because the cell holds two different
    kinds of not-quite-UTF-8 and only one handler reads each correctly.

    A LONE SURROGATE IS A LEGAL str -- `errors="surrogateescape"` produces
    one and `os.fsdecode` hands one back for an undecodable filename -- and
    the compiler spells it in the cell as WTF-8: three bytes with an `ED`
    lead, which is what every character walk in the compiled runtimes already
    reads as one character. `surrogateescape` maps those three bytes to THREE
    surrogates, so the interpreter counted a three-character string holding
    one as five, where both compiled runtimes said three.

    NO SURROGATE IS WRITTEN OUT IN THIS DOCSTRING, deliberately: a module
    whose own text carries one cannot be imported on a UTF-8 stream, and
    writing the example cost a debugging session.

    `surrogatepass` reads them back as the one character they are and refuses
    genuinely invalid bytes, which is what the fallback is for: a cell holding
    raw undecodable data still round-trips the way it did.
    """
    try:
        return raw.decode("utf-8", "surrogatepass")
    except UnicodeDecodeError:
        return raw.decode("utf-8", "surrogateescape")


def _read_cstr(h, addr: int) -> str:
    buf = h._interp.mem.buf
    end = buf.index(0, addr)
    return _decode_cell(bytes(buf[addr:end]))


def _apy_from_cstr(h, a):
    return h._new(_read_cstr(h, int(a[0])))


def _apy_from_bytes(h, a):
    addr, n = int(a[0]), int(a[1])
    buf = h._interp.mem.buf
    return h._new(_decode_cell(bytes(buf[addr:addr + n])))


# ── extraction ──────────────────────────────────────────────────────────────
# No kind check, matching the C: the frontend calls these only where it has
# proved the kind, and a check here would hide a compiler bug behind a zero.

#: What a slice bound past any real length becomes. The compiled runtime uses
#: `1 << 62` so that adding a length cannot overflow; the host matches it so
#: the two clamp to the same place.
_HUGE_BOUND = 1 << 62


def _apy_slice_bound(h, a):
    """A SLICE BOUND, WHICH IS NOT AN INDEX.

    `xs[2 ** 100]` is a request neither runtime can serve and CPython refuses
    it too; `xs[:2 ** 100]` is the whole list, and refusing THAT would be
    wrong. A value too large clamps, keeping its sign so that
    `xs[-(2 ** 100):]` is the whole list as well.
    """
    v = h._get(a[0], "apy_slice_bound")
    if v is None:
        # AN EXPLICIT `None` IS NOT A BOUND, and the caller pairs this with
        # `apy_slice_given`, which reports that it was not given at all.
        return 0
    if _is_int_like(v) and not -(1 << 63) <= int(v) < (1 << 63):
        return -_HUGE_BOUND if int(v) < 0 else _HUGE_BOUND
    if _is_int_like(v):
        return _wrap64(int(v))
    if isinstance(v, Instance) and v.cls.find("__index__") is not None:
        try:
            got = v._send("__index__")
        except _UserFailed:
            return 0
        if _is_int_like(got):
            # A BIG FROM `__index__` CLAMPS TOO. The bound rule is about the
            # POSITION, not about where the number came from.
            if not -(1 << 63) <= int(got) < (1 << 63):
                return -_HUGE_BOUND if int(got) < 0 else _HUGE_BOUND
            return _wrap64(int(got))
    # NOT `apy_index`'s MESSAGE. CPython words a bad slice bound differently
    # from a bad index, and now that None is accepted the difference matters:
    # its wording NAMES None as one of the things a bound may be.
    h._fail("TypeError", "slice indices must be integers or None or have an "
                         "__index__ method")
    return 0


def _apy_slice_given(h, a):
    """Whether this bound was GIVEN, in the sense a slice means.

    `xs[None:None]` IS `xs[:]`. Python reads an explicitly written `None`
    bound exactly as it reads an omitted one, which is also why
    `slice(None, None, None)` is what a bare `[:]` builds. The frontend knows
    which bounds were WRITTEN and cannot know which of them evaluate to None;
    this answers the second half.
    """
    return 0 if h._get(a[0], "apy_slice_given") is None else 1


def _apy_slice_step(h, a):
    """A slice's step, where `None` means ONE rather than nothing.

    Separate from the bound because the default differs and a step of zero is
    an error: `xs[::None]` is `xs[::1]`.
    """
    if h._get(a[0], "apy_slice_step") is None:
        return 1
    return _apy_slice_bound(h, a)


def _apy_index(h, a):
    """A VALUE AS AN INDEX, checked -- a `range` argument or a subscript, which
    came from the program and may be anything, including a user object with
    `__index__`.

    AN INTEGER THAT DOES NOT FIT IS REFUSED rather than truncated. It used to
    go through `_wrap64`, which turned `2 ** 100` into whatever its low 64
    bits happened to be -- a silent wrong index. The compiled runtime had the
    same bug from the other direction: it tested `apy_is_int_like` first,
    which is true of a big, and read the limb POINTER as the value.
    """
    v = h._get(a[0], "apy_index")
    if _is_int_like(v) and not -(1 << 63) <= int(v) < (1 << 63):
        h._fail("OverflowError",
                "cannot fit 'int' into an index-sized integer")
        return 0
    if _is_int_like(v):
        return _wrap64(int(v))
    if isinstance(v, Instance) and v.cls.find("__index__") is not None:
        try:
            got = v._send("__index__")
        except _UserFailed:
            return 0
        if _is_int_like(got) and not -(1 << 63) <= int(got) < (1 << 63):
            h._fail("OverflowError",
                    "cannot fit 'int' into an index-sized integer")
            return 0
        if _is_int_like(got):
            return _wrap64(int(got))
    h._fail("TypeError",
            f"'{h.kind_name(v)}' object cannot be interpreted as an integer")
    return 0


def _as_index(h, v):
    """A value as an index, or `_UserFailed` with the refusal already set.

    `[1, 2].pop(1.0)` IS A TypeError AND NOT A POSITION. The conversion used
    to be a bare `int()`, which truncated -- the compiled halves read the
    bits of the double instead, which is the same bug from the other side.
    See `apy_index_arg_of`, which the C and the subset share.
    """
    if _is_int_like(v):
        return int(v)
    if isinstance(v, Instance) and v.cls.find("__index__") is not None:
        got = v._send("__index__")
        if _is_int_like(got):
            return int(got)
    h._fail("TypeError",
            f"'{h.kind_name(v)}' object cannot be interpreted as an integer")
    raise _UserFailed


def _apy_as_int(h, a):
    """The MACHINE WORD behind a value, for the frontend's own arithmetic.

    Still truncating, and it is the only thing left that does: this models the
    C reading `v.i` out of the cell, which is a 64-bit field however large the
    integer it came from. The C's own `apy_as_int` has no kind check for the
    same reason, and an integer too big to fit is a frontend bug in both.
    """
    return _wrap64(int(h._get(a[0], "apy_as_int")))


def _apy_as_float(h, a):
    return float(h._get(a[0], "apy_as_float"))


def _apy_as_bool(h, a):
    return 1 if h._get(a[0], "apy_as_bool") else 0


# ── inspection ──────────────────────────────────────────────────────────────

def _apy_bind_of(h, a):
    """A bound method: the function plus the receiver it was found on."""
    f = h._get(a[0], "apy_bind_of")
    self_ = h._get(a[1], "apy_bind_of")
    return h._new(f.__get__(self_) if hasattr(f, "__get__") else f)


def _apy_dunder_of(h, a):
    """The bound `name` method of an instance, or 0.

    ONLY AN INSTANCE HAS ONE, and only when the CLASS defines it -- looking on
    the instance would find an attribute that happened to share the name, which
    is not how Python resolves a dunder.
    """
    v = h._get(a[0], "apy_dunder_of")
    name = str(h._get(a[1], "apy_dunder_of"))
    m = getattr(type(v), name, None)
    if m is None or not callable(m):
        return 0
    return h._new(m.__get__(v))


def _apy_unary_dunder_of(h, a):
    """`v.__name__()`, or 0 when the class does not define one.

    ZERO MEANS TWO THINGS and the caller tells them apart by the error flag:
    "no such method" and "it ran and failed". Every operator dispatch in this
    runtime reads it that way, which is why nothing is returned to say which.
    """
    m = _apy_dunder_of(h, a)
    if m == 0:
        return 0
    try:
        return h._value(h._invoke(h._get(m, "apy_unary_dunder_of"), []))
    except _UserFailed:
        return 0


def _apy_method1_of(h, a):
    """`v.__name__(arg)`, or 0 when the class does not define one."""
    m = _apy_dunder_of(h, a[:2])
    if m == 0:
        return 0
    arg = h._get(a[2], "apy_method1_of")
    try:
        return h._value(h._invoke(h._get(m, "apy_method1_of"), [arg]))
    except _UserFailed:
        return 0


def _apy_clamp_range_of(h, a):
    """Slice bounds, which the host does not keep in memory.

    THE COMPILED VERSION WRITES THROUGH TWO POINTERS and the host has no
    addresses to write to -- every function that would clamp a range is itself
    bound in this file and uses Python's own slicing.
    """
    raise RuntimeError(
        "apy_clamp_range_of has no host equivalent: it writes through two "
        "int64 pointers, and the interpreter slices with Python's own rules")


def _apy_int_arg_of(h, a):
    """An integer argument, which the host reads without an out-parameter."""
    raise RuntimeError(
        "apy_int_arg_of has no host equivalent: it writes through an int64 "
        "pointer, and the interpreter passes values")


def _apy_slice_arg_of(h, a):
    """A slice bound. See `_apy_int_arg_of`."""
    raise RuntimeError(
        "apy_slice_arg_of has no host equivalent: it writes through an int64 "
        "pointer, and the interpreter passes values")


def _apy_affix1_of(h, a):
    """Does a prefix or suffix sit at one end of a window?"""
    s = h._get(a[0], "apy_affix1_of")
    fix = h._get(a[1], "apy_affix1_of")
    lo, hi, at_end = int(a[2]), int(a[3]), int(a[4])
    window = s[lo:hi]
    return 1 if (window.endswith(fix) if at_end
                 else window.startswith(fix)) else 0


def _apy_name_of(h, a):
    """An interned str cell for an attribute name.

    NO CACHE HERE, because there is nothing to save: the compiled runtime
    interns to avoid allocating a str cell per attribute access, and the host
    is handing back a Python string it already has.
    """
    return h._new(str(h._get(a[0], "apy_name_of")))


def _apy_name_rows(h, a):
    """The cache's table, which the host does not have. See `_apy_name_of`."""
    raise RuntimeError(
        "apy_name_rows has no host equivalent: the interpreter does not "
        "intern attribute names, so there is no table")


def _apy_name_slot(h, a):
    """The cache's count. See `_apy_name_rows`."""
    raise RuntimeError(
        "apy_name_slot has no host equivalent: the interpreter does not "
        "intern attribute names, so there is nothing to count")


def _apy_type_for(h, a):
    """`type(v)` -- the object, not the name.

    PYTHON'S OWN `type`, which interns by construction: `type(1) is type(2)`
    is true here because there is one `int`, where the compiled runtime has
    to remember the cell it made.
    """
    return h._new(h._type_of(h._get(a[0], "apy_type_for")))


def _apy_type_rows(h, a):
    """The interning table, which the host does not keep. See `_apy_type_for`."""
    raise RuntimeError(
        "apy_type_rows has no host equivalent: the interpreter uses Python's "
        "own type objects, which are already unique")


def _apy_type_slot_count(h, a):
    """The table's count. See `_apy_type_rows`."""
    raise RuntimeError(
        "apy_type_slot_count has no host equivalent: there is no table to "
        "count")


def _apy_canonical_slot(h, a):
    """Where program-declared classes are remembered.

    THE HOST KEEPS THEM ON ITSELF -- `h.exc_class` and the class objects it
    builds -- so there is no single word to hand out.
    """
    raise RuntimeError(
        "apy_canonical_slot has no host equivalent: the interpreter keeps "
        "declared classes on the host object")


def _apy_gen_step_of(h, a):
    """Resume a generator once, filling `done` with whether it finished.

    THE HOST'S GENERATORS ARE REAL PYTHON ONES, resumed by `send` and ended
    by a StopIteration the interpreter catches -- there is no state word to
    read and no `done` cell in its memory to write. Every caller of this is
    bound in this file and steps them that way.
    """
    raise RuntimeError(
        "apy_gen_step_of has no host equivalent: the interpreter resumes "
        "real Python generators")


def _apy_is_data_descriptor_of(h, a):
    """Does `v` want to intercept a WRITE as well as a read?

    `_apy_default_setattr` DOES THIS INLINE, through `_descr_set` -- the
    compiled runtime factors the question out because two callers want it and
    the interpreter has only the one.
    """
    raise RuntimeError(
        "apy_is_data_descriptor_of has no host equivalent: the interpreter "
        "asks inside its own setattr")


def _apy_descr_set_of(h, a):
    """Hand a write to a data descriptor. See `_apy_is_data_descriptor_of`."""
    raise RuntimeError(
        "apy_descr_set_of has no host equivalent: the interpreter hands a "
        "write to a descriptor inside its own setattr")


def _apy_slot_allows_of(h, a):
    """May `name` be stored on an instance of `cls`? Same reason."""
    raise RuntimeError(
        "apy_slot_allows_of has no host equivalent: the interpreter checks "
        "__slots__ inside its own setattr")


def _apy_binary_dunder_of(h, a):
    """`a.__op__(b)` first, then `b.__rop__(a)`.

    EVERY CALLER IS BOUND IN THIS FILE. The interpreter dispatches each
    operator to Python's own, which already tries the reflected form and
    already understands `NotImplemented` -- so there is nothing left for a
    shared worker to do.
    """
    raise RuntimeError(
        "apy_binary_dunder_of has no host equivalent: the interpreter lets "
        "Python try the reflected operator itself")


def _apy_extreme_of(h, a):
    """`max(xs)` and `min(xs)` over one sequence.

    `_apy_max` AND `_apy_min` EACH DO IT DIRECTLY, through Python's own --
    the compiled runtime shares one body because the two differ by a sign and
    the interpreter has two small ones.
    """
    raise RuntimeError(
        "apy_extreme_of has no host equivalent: the interpreter binds max() "
        "and min() to Python's own")


def _apy_extreme_by_of(h, a):
    """`max(xs, key=f)`. See `_apy_extreme_of`."""
    raise RuntimeError(
        "apy_extreme_by_of has no host equivalent: the interpreter binds "
        "max() and min() to Python's own")


def _apy_base_text_of(h, a):
    """`bin`, `oct` and `hex`, which are one body three times.

    `_apy_bin`, `_apy_oct` AND `_apy_hex` EACH REACH PYTHON'S OWN, so the
    shared worker has no caller on this path -- the compiled runtime shares
    one because the three differ by a base and a prefix.
    """
    raise RuntimeError(
        "apy_base_text_of has no host equivalent: the interpreter binds bin, "
        "oct and hex to Python's own")


def _apy_big_base_text_of(h, a):
    """The same for a big. See `_apy_base_text_of` -- Python's integers are
    already arbitrary-precision here, so there is no separate path."""
    raise RuntimeError(
        "apy_big_base_text_of has no host equivalent: the interpreter's "
        "integers are Python's, which need no separate big path")


def _apy_splitlines_impl_of(h, a):
    """`s.splitlines()`. `_apy_str_splitlines` and its keepends twin each
    reach Python's own, so the shared worker has no caller here."""
    raise RuntimeError(
        "apy_splitlines_impl_of has no host equivalent: the interpreter "
        "binds splitlines to Python's own")


def _apy_arg_must_be_str_of(h, a):
    """The TypeError a string method raises for a non-string argument.

    THE HOST LETS PYTHON RAISE IT. Its string methods are Python's own, so
    the refusal and its wording both come from there -- there is nothing for
    a shared message builder to do.
    """
    raise RuntimeError(
        "apy_arg_must_be_str_of has no host equivalent: Python's own string "
        "methods raise this")


def _apy_inst_held_of(h, a):
    """The builtin an instance wraps.

    THE HOST READS `Instance.held` DIRECTLY wherever it needs it, so the
    accessor the compiled runtime factors out has no caller on this path.
    """
    v = h._get(a[0], "apy_inst_held_of")
    if not isinstance(v, Instance) or v.held is None:
        return 0
    return h._new(v.held)


def _apy_str_count_in_of(h, a):
    """`s.count(sub)` with bounds. `_apy_str_count2` and its three-argument
    twin each reach Python's own, so the shared worker has no caller here."""
    raise RuntimeError(
        "apy_str_count_in_of has no host equivalent: the interpreter binds "
        "count to Python's own")


def _apy_group_select_of(h, a):
    """The part of an exception group matching a type.

    `_apy_group_split`, `_apy_group_subgroup` AND `_apy_group_dispatch` each
    walk the group themselves here, so the shared selector the compiled
    runtime factors out has no caller on this path.
    """
    raise RuntimeError(
        "apy_group_select_of has no host equivalent: the interpreter walks "
        "an exception group inside each of split, subgroup and dispatch")


def _apy_descr_get_of(h, a):
    """Read through a descriptor.

    THE HOST'S ATTRIBUTE LOOKUP DOES THIS INLINE, through `_descr_get` -- the
    compiled runtime factors it out because two callers want it, and the
    interpreter reaches it from one place.
    """
    raise RuntimeError(
        "apy_descr_get_of has no host equivalent: the interpreter reads "
        "through a descriptor inside its own getattr")


def _apy_kind_method_opt(h, a):
    """One builtin method as a value, with an OPTIONAL TAIL.

    THE HOST NEEDS NO RANGE OF ITS OWN: a Native here wraps a Python callable
    and Python checks the count, so the declared arity never mattered. The
    binding exists because the runtime names this and a name the table does
    not carry is a link error rather than a fallback -- and it is never
    reached, for the reason `_apy_kind_method_of` beside it gives.
    """
    raise RuntimeError(
        "apy_kind_method_opt has no host equivalent: the interpreter answers "
        "Python's own bound method, which carries its own arity")


def _apy_kind_class(h, a):
    """The class object standing for a builtin kind.

    THE SAME OBJECT `type(x)` ANSWERS, which here needs no cache of its own:
    Python already keeps one type object per type, and the compiled runtimes
    reach `apy_type_for` for exactly that reason.
    """
    return _apy_type_object(h, a)


def _apy_member_descriptor(h, a):
    """One `member_descriptor`, which is what a slot reads as on the class."""
    raise RuntimeError(
        "apy_member_descriptor has no host equivalent: the interpreter "
        "answers a slot read on the class its own way")


def _apy_object_default(h, a):
    """`object`'s own implementation of a dunder, by name.

    THE HOST HAS NO SELECTORS. Its dunders are Python functions found on real
    classes, so there is nothing to hand back by name -- every caller of this
    is bound in this file and looks the method up instead.
    """
    raise RuntimeError(
        "apy_object_default has no host equivalent: the interpreter finds "
        "object's dunders on real Python classes")


def _apy_kind_attr(h, a):
    """The builtin method or field a name means on a builtin value.

    THE HOST ASKS PYTHON. Its lists and dicts are Python's, so `getattr` on
    one answers a real bound method and there is no table of what exists to
    consult -- the compiled runtime needs one because the methods live in the
    frontend's dispatch and not in any class.
    """
    raise RuntimeError(
        "apy_kind_attr has no host equivalent: the interpreter asks Python "
        "for a builtin's attributes")


def _apy_kind_attr_of(h, a):
    """The same, unbound. See `_apy_kind_attr`."""
    raise RuntimeError(
        "apy_kind_attr_of has no host equivalent: the interpreter asks "
        "Python for a builtin's attributes")


def _apy_meth_arity(h, a):
    """Refuse `got` arguments to this builtin method on this receiver the way
    CPython does, or answer None so the call goes ahead.

    THE LOWERING EMITS THIS ahead of a written call whose argument count SOME
    kind refuses -- ten `(name, count)` pairs, no more, because the frontend
    knows both statically and emits nothing where every kind that has the
    name accepts the count. Without it `set().pop(1)` was `'set' object has
    no attribute 'pop'` about a method a set plainly has, and `{}.pop()` was
    a `KeyError` from a symbol that took a call it should have refused.
    """
    recv = h._get(a[0], "apy_meth_arity")
    want = h._get(a[1], "apy_meth_arity")
    # THE COUNT IS AN `int64_t` and arrives unboxed, not as a handle.
    got = int(a[2])
    packed = KINDMETH_WORDS.get((want, _meth_kind(recv)))
    # A NAME THIS KIND DOES NOT HAVE really is an AttributeError, and the
    # caller's own path words it.
    if packed is None:
        return h._new(None)
    if (packed & 15) <= got <= ((packed >> 4) & 15):
        return h._new(None)
    return _meth_arity_words(h, recv, want, packed, got)


def _apy_kind_method_var(h, a):
    """A builtin method that takes WHATEVER IT IS GIVEN, by name.

    The compiled runtimes need this one because their native carries a
    declared arity and `str.format` has none -- see `apy_kind_method_var`.
    The host has nothing to declare: `_kind_attr` hands out a `Native` with
    `variadic` set and Python's own `*args, **kw` does the rest, so there is
    nothing here to build.
    """
    raise RuntimeError(
        "apy_kind_method_var has no host equivalent: the interpreter's "
        "natives take what they are given")


def _apy_kind_method_of(h, a):
    """One builtin method as a callable value. See `_apy_kind_attr`."""
    raise RuntimeError(
        "apy_kind_method_of has no host equivalent: the interpreter answers "
        "Python's own bound method")


def _apy_kind_prototype(h, a):
    """An empty value of a named builtin kind. See `_apy_kind_attr`."""
    raise RuntimeError(
        "apy_kind_prototype has no host equivalent: the interpreter asks "
        "Python's own type for a builtin's attributes")


def _apy_no_attribute(h, a):
    """The last thing attribute lookup tries, and the error if it fails.

    `_apy_default_getattr` RAISES IT DIRECTLY here, because the host has no
    separate table of builtin attributes to consult first.
    """
    raise RuntimeError(
        "apy_no_attribute has no host equivalent: the interpreter raises "
        "the AttributeError from its own getattr")


def _apy_traceback_of(h, a):
    """`e.__traceback__`.

    THE HOST BUILDS ITS OWN, from the position it keeps on the interpreter
    rather than from a table the compiled runtime reserves.
    """
    raise RuntimeError(
        "apy_traceback_of has no host equivalent: the interpreter builds a "
        "traceback from its own position record")


def _apy_exc_shown_of(h, a):
    """The name an exception SHOWS, which is not always the one it matches.

    THE HOST'S EXCEPTIONS ARE REAL PYTHON CLASSES, so the name it shows is
    already the name it has -- the mangling this exists to hide belongs to
    the compiled path's bundled-module splice.
    """
    name = str(h._get(a[0], "apy_exc_shown_of"))
    cls = h.exc_class.get(name)
    return h._new(cls.name if cls is not None else name)


def _apy_special_form_class(h, a):
    """The class every interned typing form is an instance of.

    THE HOST KEEPS ITS TYPING FORMS AS PYTHON OBJECTS, so there is no single
    class cell to hand out -- `_apy_is_special_form` asks about them
    directly.
    """
    raise RuntimeError(
        "apy_special_form_class has no host equivalent: the interpreter "
        "keeps typing forms as Python objects")


def _apy_text_result_of(h, a):
    """What a user `__repr__` answered, if it answered a string.

    THE HOST CHECKS INLINE wherever it calls one, because Python's own
    `repr()` already refuses a non-string and there is one place that has to
    reproduce the wording.
    """
    raise RuntimeError(
        "apy_text_result has no host equivalent: the interpreter checks a "
        "dunder's result where it calls it")


def _apy_bytes_repr(h, a):
    """`repr(b"...")`, and `bytearray(...)` around a mutable one.

    THE HOST ASKS PYTHON, whose bytes and bytearray already render
    themselves -- including the quote choice and the escapes this exists to
    reproduce on the compiled path.
    """
    v = h._get(a[0], "apy_bytes_repr")
    return h._new(repr(v))


def _apy_text_of(h, a):
    """`str(v)` or `repr(v)`, depending on the flag.

    THE HOST ASKS PYTHON, which renders its own objects -- and for the kinds
    the interpreter keeps as real Python values that is the same answer the
    compiled runtime builds by hand.
    """
    v = h._get(a[0], "apy_text_of")
    quoted = int(h._get(a[1], "apy_text_of"))
    return h._new(repr(v) if quoted else str(v))


def _apy_exc_text_of(h, a):
    """`repr(e)` and `str(e)`.

    THE HOST ASKS PYTHON, whose exceptions already know both spellings --
    including the two the compiled version is written out for: the tuple an
    exception with several arguments shows, and the `[Errno n] msg` form the
    OSError family puts on a message.
    """
    v = h._get(a[0], "apy_exc_text_of")
    quoted = int(h._get(a[1], "apy_exc_text_of"))
    return _user(h, lambda: h._new(repr(v) if quoted else str(v)))


def _apy_seq_text_of(h, a):
    """`[1, 2]` and `(1, 2)`.

    THE HOST ASKS PYTHON, whose lists and tuples already render themselves --
    including the one-element tuple's comma and the `[...]` a cycle prints,
    which are the two things the compiled version exists to get right.
    """
    return _user(h, lambda: h._new(repr(h._get(a[0], "apy_seq_text_of"))))


def _apy_dict_text_of(h, a):
    """`{'a': 1}`. THE HOST ASKS PYTHON, as above."""
    return _user(h, lambda: h._new(repr(h._get(a[0], "apy_dict_text_of"))))


def _apy_set_text_of(h, a):
    """`{1, 2}` and `frozenset({1, 2})`. THE HOST ASKS PYTHON, as above."""
    return _user(h, lambda: h._new(repr(h._get(a[0], "apy_set_text_of"))))


def _apy_big_text(h, a):
    """A big in base ten.

    THE HOST'S INTEGERS ARE PYTHON'S, which are already arbitrary-precision
    and already know how to print themselves -- there is no separate big
    representation here to render.
    """
    raise RuntimeError(
        "apy_big_text has no host equivalent: the interpreter's integers are "
        "Python's own")


def _apy_every_of(h, a):
    """`all(v)` and `any(v)`, which are one walk with two answers.

    `_apy_all` AND `_apy_any` BELOW EACH DO IT DIRECTLY, so nothing on this
    path reaches the shared worker -- the compiled runtime shares one body
    because it is the same walk, and the interpreter has two small ones.
    """
    raise RuntimeError(
        "apy_every_of has no host equivalent: the interpreter binds all() "
        "and any() to their own walks")


def _apy_gen_stop(h, a):
    """The StopIteration a finished generator ends with, carrying its return.

    THE HOST\'S GENERATORS ARE REAL PYTHON ONES, so the StopIteration is
    raised by Python itself and already carries the value -- there is no
    result slot here to read one out of.
    """
    raise RuntimeError(
        "apy_gen_stop has no host equivalent: the interpreter lets Python "
        "raise its own StopIteration")


def _apy_exc_class_named_of(h, a):
    """The class a program declared for exceptions of this name, or 0.

    THE HOST KEEPS THE TABLE ON ITSELF -- `h.exc_class`, keyed by name -- so
    this reads it directly rather than through the slot the compiled runtime
    reserves.
    """
    name = str(h._get(a[0], "apy_exc_class_named_of"))
    cls = h.exc_class.get(name)
    return h._new(cls) if cls is not None else 0


def _apy_exc_construct_of(h, a):
    """Run a program-written exception class's `__init__` over a raised cell.

    THE HOST BUILDS ITS EXCEPTIONS AS REAL PYTHON OBJECTS, so a class it
    declared is instantiated by Python itself and there is no separate cell
    to run a constructor over afterwards. Every caller of this is bound in
    this file and builds the exception the host's own way.
    """
    raise RuntimeError(
        "apy_exc_construct_of has no host equivalent: the interpreter builds "
        "exceptions as real Python objects")


def _apy_exc_class_slot(h, a):
    """Where the name-to-class table for program exceptions lives.

    THE HOST KEEPS IT ON ITSELF -- `h.exc_class`, a real dict keyed by name --
    so there is no single word to hand out. `apy_exc_class_bind` and the
    lookup beside it are both bound in this file and reach it directly.
    """
    raise RuntimeError(
        "apy_exc_class_slot has no host equivalent: the interpreter keeps "
        "the exception-class table on the host object")


def _apy_live_agens_slot(h, a):
    """Where the list of started async generators lives.

    THE HOST TRACKS THEM ON ITSELF, because its generators are real Python
    ones and the shutdown walk is a list on the interpreter rather than a
    runtime value.
    """
    raise RuntimeError(
        "apy_live_agens_slot has no host equivalent: the interpreter keeps "
        "live async generators on the host object")


def _apy_tasks_slot(h, a):
    """Where the list of handed-over tasks lives.

    THE HOST RUNS ITS OWN LOOP over a list it keeps, so there is no runtime
    value holding them and nothing to hand back.
    """
    raise RuntimeError(
        "apy_tasks_slot has no host equivalent: the interpreter keeps the "
        "task list on the host object")


def _apy_affix_of(h, a):
    """`startswith`/`endswith` with bounds and a tuple of prefixes.

    EVERY CALLER IS BOUND IN THIS FILE -- the six exported spellings each
    reach Python's own method directly -- so the shared worker is never the
    thing the interpreter is asked for.
    """
    raise RuntimeError(
        "apy_affix_of has no host equivalent: the interpreter binds each "
        "startswith/endswith spelling to Python's own method")


def _apy_str_slice_of(h, a):
    """`s[lo:hi]` by BYTE bounds.

    THE HOST HAS NO BYTE BOUNDS TO SLICE BY. Its strings are Python strings
    and its indices are characters, so a byte-addressed cut has no meaning
    here -- every caller of this is itself bound in this file and slices
    with Python's own indices.
    """
    raise RuntimeError(
        "apy_str_slice_of has no host equivalent: the interpreter slices "
        "Python strings by character")


def _apy_split_ws_of(h, a):
    """Split on runs of whitespace.

    EVERY CALLER IS BOUND IN THIS FILE -- the six exported split spellings
    each reach Python's own `split`/`rsplit`, which is the same algorithm
    including the two modes.
    """
    raise RuntimeError(
        "apy_split_ws_of has no host equivalent: the interpreter binds each "
        "split spelling to Python's own method")


def _apy_split_sep_of(h, a):
    """Split on each occurrence of a separator. See `_apy_split_ws_of`."""
    raise RuntimeError(
        "apy_split_sep_of has no host equivalent: the interpreter binds each "
        "split spelling to Python's own method")


def _apy_str_split_impl_of(h, a):
    """Which split algorithm a call means. See `_apy_split_ws_of`."""
    raise RuntimeError(
        "apy_str_split_impl_of has no host equivalent: the interpreter binds "
        "each split spelling to Python's own method")


def _apy_native_of(h, a):
    """The runtime's own implementation of a dunder, as a callable value.

    THE HOST HAS NO SELECTORS. Its dunders are Python functions found on real
    classes, so there is nothing to cache and nothing to hand back -- every
    caller of this is itself bound in this file and looks the method up
    instead.
    """
    raise RuntimeError(
        "apy_native_of has no host equivalent: the interpreter finds dunders "
        "on real Python classes rather than by selector")


def _apy_type_class(h, a):
    """The class `type` itself is."""
    return h._new(type)


def _apy_abs64_of(h, a):
    """|v|, with INT64_MIN answering its own bit pattern.

    THE WRAP IS DELIBERATE and matches the compiled half: the magnitude of the
    most negative int64 does not fit an int64, and every caller wants the bits
    rather than a number to do arithmetic on.
    """
    return _wrap64(abs(int(a[0])))


def _apy_binop_error_of(h, a):
    """`unsupported operand type(s) for OP` -- what every operator refuses with."""
    op = str(h._get(a[0], "apy_binop_error_of"))
    x = h._get(a[1], "apy_binop_error_of")
    y = h._get(a[2], "apy_binop_error_of")
    h.err = ("TypeError",
             f"unsupported operand type(s) for {op}: "
             f"'{h.kind_name(x)}' and '{h.kind_name(y)}'")
    h.err_value = None
    h.pos_err = h.pos_here
    return 0


def _set_rhs(h, b, who):
    """The right operand as a set, or None if it cannot be one."""
    if isinstance(b, (set, frozenset)):
        return b
    # A CLASS OF THIS FILE'S IS WALKED THROUGH THE FUNNEL, not handed to
    # `set()`. An `Instance` is not a Python iterable however well a `for`
    # loop walks it, so `s.union(obj)` and `s | obj` reported that it could
    # not be one -- about a class whose `__iter__` plainly says how. The
    # refusal `_seq_items` sets on the way out is PUT BACK THE WAY IT WAS
    # FOUND, because both callers word their own: `|` says `unsupported
    # operand type(s)` and `union` names the kind.
    if isinstance(b, (Instance, Class, Gen, Iterator)):
        keep = (h.err, h.err_value, h.pos_err)
        got = _seq_items(h, b, who)
        if got is None:
            h.err, h.err_value, h.pos_err = keep
            return None
        b = got
    try:
        return set(b)
    except TypeError:
        return None


def _apy_set_algebra_of(h, a):
    """`|`, `&`, `-` and `^`, and the methods that spell them out.

    THE RESULT KEEPS THE LEFT SIDE'S KIND, so `frozenset({1}) | {2}` is a
    frozenset -- which Python's own operators do, so the four are reached
    through the type of the left operand rather than reimplemented.
    """
    op = str(h._get(a[0], "apy_set_algebra_of"))
    x = h._get(a[1], "apy_set_algebra_of")
    y = h._get(a[2], "apy_set_algebra_of")
    which, strict = int(a[3]), int(a[4])
    if not isinstance(x, (set, frozenset)):
        return _apy_binop_error_of(h, a[:3])
    if strict and not isinstance(y, (set, frozenset)):
        return _apy_binop_error_of(h, a[:3])
    rhs = _set_rhs(h, y, op)
    if rhs is None:
        return _apy_binop_error_of(h, a[:3])
    made = (x | rhs if which == 0 else x & rhs if which == 1
            else x - rhs if which == 2 else x ^ rhs)
    return h._new(made)


def _apy_set_method_of(h, a):
    """`s.union(x)` and its three siblings -- the non-strict spelling."""
    x = h._get(a[1], "apy_set_method_of")
    if not isinstance(x, (set, frozenset)):
        name = str(h._get(a[0], "apy_set_method_of"))
        h.err = ("AttributeError",
                 f"'{h.kind_name(x)}' object has no attribute '{name}'")
        h.err_value = None
        h.pos_err = h.pos_here
        return 0
    return _apy_set_algebra_of(h, (a[0], a[1], a[2], a[3], 0))


def _apy_set_relate_of(h, a):
    """`issubset`, `issuperset` and `isdisjoint` -- all answer a bool."""
    name = str(h._get(a[0], "apy_set_relate_of"))
    x = h._get(a[1], "apy_set_relate_of")
    y = h._get(a[2], "apy_set_relate_of")
    which = int(a[3])
    if not isinstance(x, (set, frozenset)):
        h.err = ("AttributeError",
                 f"'{h.kind_name(x)}' object has no attribute '{name}'")
        h.err_value = None
        h.pos_err = h.pos_here
        return 0
    rhs = _set_rhs(h, y, name)
    if rhs is None:
        return _apy_binop_error_of(h, (a[0], a[1], a[2]))
    got = (x <= rhs if which == 0 else x >= rhs if which == 1
           else x.isdisjoint(rhs))
    return h._new(bool(got))


def _apy_hash_raw_of(h, a):
    """A hash for anything that can be a key.

    PYTHON'S OWN `hash`, which gets every rule for free -- `hash(5) ==
    hash(5.0)`, a tuple ordered and a frozenset not, a class's `__hash__`.
    The compiled runtime reproduces those; here they are the definition.
    """
    return _wrap64(hash(h._get(a[0], "apy_hash_raw_of")))


def _apy_set_mask_of(h, a):
    """The mask an n-element set orders by. Arithmetic, so it is the same."""
    n, size = int(a[0]), 8
    while n * 5 >= size * 3:
        size *= 2
    return size - 1


def _apy_q_append_of(h, a):
    """Append to a sequence, which on this side is a Python list."""
    h._get(a[0], "apy_q_append_of").append(h._get(a[1], "apy_q_append_of"))
    return None


def _apy_set_find_of(h, a):
    """Where an item sits in a set, or -1.

    BY POSITION, because the compiled set is an ordered table its callers
    index. The host keeps a Python set, which has no positions -- so the
    order is the one iteration gives, which is what every caller of this
    actually uses it for.
    """
    s = h._get(a[0], "apy_set_find_of")
    item = h._get(a[1], "apy_set_find_of")
    for i, v in enumerate(s):
        if v == item and type(v) is type(item):
            return i
    return -1


def _apy_set_reorder_of(h, a):
    """Put a set back in hash order, which the host does not keep."""
    return None


def _apy_unhashable_elem_of(h, a):
    """The TypeError a SET raises for an element it cannot hash."""
    item = h._get(a[0], "apy_unhashable_elem_of")
    inner = str(h._get(a[1], "apy_unhashable_elem_of"))
    h.err = ("TypeError",
             f"cannot use '{h.kind_name(item)}' as a set element "
             f"(unhashable type: '{inner}')")
    h.err_value = None
    h.pos_err = h.pos_here
    return 0


def _apy_subset_of(h, a):
    """Is every element of the first in the second?"""
    x = h._get(a[0], "apy_subset_of")
    y = h._get(a[1], "apy_subset_of")
    return 1 if all(v in y for v in x) else 0


def _apy_mutable_set_of(h, a):
    """Is this a set that may be changed? A frozenset is not."""
    name = str(h._get(a[0], "apy_mutable_set_of"))
    s = h._get(a[1], "apy_mutable_set_of")
    if isinstance(s, set) and not isinstance(s, frozenset):
        return 1
    h.err = ("AttributeError",
             f"'{h.kind_name(s)}' object has no attribute '{name}'")
    h.err_value = None
    h.pos_err = h.pos_here
    return 0


def _apy_set_insert_of(h, a):
    """Add to a set. 1 if new, 0 if already there, -1 if unhashable."""
    s = h._get(a[0], "apy_set_insert_of")
    item = h._get(a[1], "apy_set_insert_of")
    try:
        hash(item)
    except TypeError:
        return _apy_unhashable_elem_of(
            h, (a[1], h._new(h.kind_name(item)))) or -1
    if item in s:
        return 0
    s.add(item)
    return 1


def _apy_set_from_of(h, a):
    """A set or frozenset holding everything in the source."""
    kind = int(a[0])
    src = h._get(a[1], "apy_set_from_of")
    # THROUGH `_seq_items` AND NOT PYTHON'S OWN `for`. A cursor and a
    # generator are classes of this file's and Python can walk neither, so
    # `set(map(f, xs))` reported `'Iterator' object is not iterable` -- a name
    # from inside the compiler, about a thing that is plainly iterable. That
    # function is the one place that knows the whole protocol.
    items = _seq_items(h, src, "apy_set_from_of")
    if items is None:
        return 0
    try:
        made = set(items)
    except TypeError:
        return _apy_unhashable_elem_of(h, (a[1], h._new("list"))) or 0
    return h._new(frozenset(made) if kind == 10 else made)


def _apy_unhashable_of(h, a):
    """The kind name to complain about, or 0 if this may be a key.

    ASKED OF PYTHON ITSELF, by trying to hash it -- which gets the tuple
    recursion and the `__eq__`-without-`__hash__` rule for free, and cannot
    drift from what the host's own dicts will accept.
    """
    v = h._get(a[0], "apy_unhashable_of")
    try:
        hash(v)
    except TypeError:
        return h._new(h.kind_name(v))
    return 0


def _apy_unhashable_key_of(h, a):
    """The TypeError a dict raises for a key it cannot hash."""
    key = h._get(a[0], "apy_unhashable_key_of")
    inner = str(h._get(a[1], "apy_unhashable_key_of"))
    h.err = ("TypeError",
             f"cannot use '{h.kind_name(key)}' as a dict key "
             f"(unhashable type: '{inner}')")
    h.err_value = None
    h.pos_err = h.pos_here
    return 0


def _apy_dict_find_of(h, a):
    """Where a key sits in a dict, or -1.

    BY POSITION, because the compiled dict keeps two parallel arrays and its
    callers index them. The host keeps a Python dict, so the position is
    recovered by walking the keys -- which is what the compiled version does
    anyway, and at the same cost.
    """
    d = h._get(a[0], "apy_dict_find_of")
    if not isinstance(d, dict):
        return -1
    key = h._get(a[1], "apy_dict_find_of")
    for i, k in enumerate(d):
        if k == key and type(k) is type(key):
            return i
    return -1


def _apy_class_find_of(h, a):
    """Find a name on a class or above it, or 0.

    THROUGH PYTHON'S OWN LOOKUP, which walks the real MRO -- the same order
    the compiled version walks, because that list is what built this class.
    """
    cls = h._get(a[0], "apy_class_find_of")
    name = str(h._get(a[1], "apy_class_find_of"))
    if not isinstance(cls, type):
        return 0
    for base in getattr(cls, "__mro__", (cls,)):
        if name in vars(base):
            return h._new(vars(base)[name])
    return 0


def _apy_str_cmp_of(h, a):
    """-1, 0 or 1 for two strings or two bytes.

    PYTHON'S OWN COMPARISON, which orders strings by code point and bytes by
    byte -- the same order the compiled version reaches by comparing UTF-8
    bytes, because that encoding was designed so the two agree.
    """
    x = h._get(a[0], "apy_str_cmp_of")
    y = h._get(a[1], "apy_str_cmp_of")
    return 0 if x == y else (-1 if x < y else 1)


def _apy_cursor_of(h, a):
    """A cursor cell, which the host does not build.

    THE INTERPRETER MAKES REAL PYTHON ITERATORS -- `map`, `filter` and the
    rest are the builtins on this side -- so nothing reaches here: every
    function that would make a cursor is itself bound in this file. It raises
    rather than answering, for the reason `_apy_err_slots` does.
    """
    raise RuntimeError(
        "apy_cursor_of has no host equivalent: the interpreter uses Python's "
        "own iterators rather than a cursor cell")


def _apy_is_int_like_of(h, a):
    """Is this an integer, in the sense Python means? A bool is one.

    ANSWERED FROM PYTHON'S OWN TYPES here, which gets the same three kinds:
    `bool` is a subclass of `int`, and the host has no separate big -- an
    integer that outgrew a machine word is still an `int` on this side.
    """
    v = h._get(a[0], "apy_is_int_like_of")
    return 1 if isinstance(v, (int, bool)) and not isinstance(v, float) else 0


def _apy_is_num_of(h, a):
    """The same three, plus float. No complex: the callers want a double."""
    v = h._get(a[0], "apy_is_num_of")
    return 1 if isinstance(v, (int, bool, float)) else 0


def _held_text(v):
    """An instance of a class extending str, bytes or bytearray, AS the value
    it holds. Identity for everything else.

    `class S(str)` makes something CPython's `find`, `join`, `replace`,
    `split` and `in` all take without a second thought -- they read the
    C-level layout, which a subclass has. Every one of them refused it here,
    and the refusal named `Instance`, a class of this file's.

    UNGATED BY WHAT THE CLASS WROTE, unlike the dunder routes: `"abc".find(s)`
    is 1 for an `s` whose `__str__` answers something else entirely, because
    `str.find` never asks. Mirrors the C's `apy_text_like`.
    """
    if isinstance(v, Instance) and isinstance(v.held, (str, bytes, bytearray)):
        return v.held
    return v


def _apy_str_self_of(h, a):
    """Is this a str or bytes receiver? Raises naming the method if not.

    RAISES AS A SIDE EFFECT AND ANSWERS A NUMBER, which is the C's shape --
    forty-odd string methods open with it -- and the host has to keep it so
    that a compiled program and an interpreted one refuse the same call the
    same way.
    """
    name = str(h._get(a[0], "apy_str_self_of"))
    v = _held_text(h._get(a[1], "apy_str_self_of"))
    if isinstance(v, (str, bytes, bytearray)):
        return 1
    h.err = ("AttributeError",
             f"'{h.kind_name(v)}' object has no attribute '{name}'")
    h.err_value = None
    h.pos_err = h.pos_here
    return 0


def _apy_seq_new_of(h, a):
    """An empty sequence of the given kind.

    THE KIND IS A NUMBER on the compiled side and a Python type here, so this
    maps the three the C uses. Anything else would be a kind the frontend
    never emits.
    """
    kind = int(a[0])
    from uasm.objects.ir import PORTED  # noqa: F401  (import kept local)
    made = {9: set(), 10: frozenset(), 7: (), 6: []}.get(kind)
    return h._new([] if made is None else made)


def _apy_kind_name_of(h, a):
    """The type name a message would use, as a host string.

    THE COMPILED RUNTIME ANSWERS A C STRING POINTER and this answers a value,
    which is the ordinary difference between the two sides -- everything that
    reads it is bound in this file and takes a value either way.

    THROUGH THE SAME `kind_name` the rest of this file uses, so the fused
    `type(x).__name__` and a TypeError about `x` cannot disagree about what
    `x` is.
    """
    return h._new(h.kind_name(h._get(a[0], "apy_kind_name_of")))


def _apy_type_name(h, a):
    """`type(x).__name__`, fused by the frontend into one call.

    THROUGH `_type_of` AND NOT STRAIGHT TO `kind_name`. The fused shape has to
    answer what the unfused one does, and `kind_name` reads an exception's own
    NAME STRING rather than the class behind it -- so a bundled module's
    exception, whose class the splice renames from `_asmpy_bundled_4_copy_Error`
    back to `Error`, showed the internal spelling. `t = type(e); t.__name__`
    was right and `type(e).__name__` was not, for the same object, because
    only one of them came through here.
    """
    v = h._get(a[0], "apy_type_name")
    got = h._type_of(v)
    if isinstance(got, Class):
        return h._new(_bare_name(got.name))
    return h._new(_bare_name(h.kind_name(v)))


def _apy_truth(h, a):
    v = h._get(a[0], "apy_truth")
    if isinstance(v, Instance):
        return _user(h, lambda: 1 if v else 0, fail=0)
    return 1 if v else 0


def _apy_len(h, a):
    v = h._get(a[0], "apy_len")
    if isinstance(v, Class) and v.meta is not None:
        # THE LENGTH OF A CLASS IS THE METACLASS'S BUSINESS, exactly as
        # iterating one is: `len(Colour)` is `type(Colour).__len__(Colour)`,
        # which is how an enum says how many members it has. `_apy_iter` grew
        # this case and this did not, so `for c in Colour` worked and
        # `len(Colour)` answered `object of type 'EnumMeta' has no len()` --
        # about a class whose metaclass plainly defines one.
        hook = v.meta.lookup("__len__")
        if hook is not _ABSENT:
            got = h._invoke(hook, [v])
            if h.err is not None:
                return 0
            return h._value(got)
    if isinstance(v, Instance):
        # A CLASS THAT EXTENDS A BUILTIN has one for everything it did not
        # write, so a `class D(dict)` whose body says nothing about length
        # still has one.
        if v.cls.find("__len__") is None and v.held is not None:
            return h._int(len(v.held))
        return _user(h, lambda: h._int(len(v)))
    if isinstance(v, _VIEW_TYPES):
        # THROUGH THE VIEW to the dict: a view has no length of its own, and
        # taking one when it was made is what a snapshot does.
        return h._int(len(v))
    if isinstance(v, (list, tuple, dict, str, bytes, set, frozenset,
                      bytearray, memoryview, range)):
        # A str's length is in CHARACTERS -- the one place the C resolves the
        # byte/character distinction, via its `apy_str_chars`. A RANGE's is
        # arithmetic on its three numbers, in both paths.
        return h._int(len(v))
    return h._fail("TypeError",
                   f"object of type '{h.kind_name(v)}' has no len()")


def _apy_raw_len(h, a):
    """The length as a machine word, for the frontend's own loop bounds.

    A str's is in CHARACTERS, as `apy_len` is and as indexing now is. It used
    to be BYTES here to match the C, which indexed bytes -- so every string
    holding a non-ASCII character was iterated one byte at a time and yielded
    halves of characters. Both runtimes count characters everywhere now.
    """
    v = h._get(a[0], "apy_raw_len")
    if isinstance(v, Gen):
        # A GENERATOR has no length until it has been run, so asking for one
        # DRAINS it. `apy_key_at` then reads the same list.
        got = _gen_cache(h, v)
        return 0 if got is None else len(got)
    if isinstance(v, Iterator):
        # A PLAIN cursor over a real container knows WHAT REMAINS without
        # walking it. One that transforms as it goes does not -- filtering may
        # drop any number -- so it is DRAINED, which turns it into a plain
        # cursor over what it produced.
        if v.mode == Iterator.PLAIN and not isinstance(v.src, (Gen, Iterator,
                                                               Instance)):
            return max(0, len(_iter_items(v.src)) - v.i)
        got = _drain_cursor(h, v)
        return 0 if got is None else len(got)
    if isinstance(v, str):
        # IN CHARACTERS, matching `apy_len` and matching what indexing counts.
        # This answered BYTES while `apy_len` answered characters, and the two
        # disagreeing was written down as a limitation -- but a `for` loop
        # takes its bound from here and its elements from the subscript, so a
        # string with any non-ASCII character in it walked off the end.
        return len(v)
    # A BYTEARRAY IS NOT A `bytes` TO `isinstance` -- it is not a subclass
    # of it -- so every list here that says `bytes` and means "the bytes
    # cell" left one out. Both compiled runtimes hold the two in one cell
    # and iterate both; only this path refused, so `list(bytearray(b'ab'))`
    # was `'bytearray' object is not iterable` about a thing whose whole
    # point is being a sequence of octets.
    # AND A VIEW IS A SEQUENCE TOO, which both compiled runtimes already
    # walked and this refused: `list(memoryview(b"ab"))` was "not iterable"
    # about the one kind whose whole point is showing a sequence of octets.
    if isinstance(v, (list, tuple, dict, set, frozenset, bytes, bytearray,
                      range, memoryview)):
        return len(v)
    # A user object with `__len__`. Together with `apy_key_at` falling through
    # to `__getitem__`, that is the whole `__len__`/`__getitem__` iteration
    # protocol -- the one the index walk here fits exactly.
    if isinstance(v, Instance) and v.cls.find("__len__") is not None:
        try:
            return int(v._send("__len__"))
        except _UserFailed:
            return 0
    # A CLASS EXTENDING A BUILTIN has the builtin's length, which is what the
    # index walk needs to bound itself. Without it the walk refused before it
    # started, so `class D(dict)` was iterable through `for` and not through
    # a comprehension -- the same object, two answers.
    if isinstance(v, Instance) and v.held is not None:
        return len(v.held)
    h._fail("TypeError", f"'{h.kind_name(v)}' object is not iterable")
    return 0


def _apy_repr(h, a):
    return _user(h, lambda: h._new(h._text(h._get(a[0], "apy_repr"), True)))


def _apy_str(h, a):
    # THROUGH `_value`, not `_new`. `str(s)` on a str IS `s` -- CPython hands
    # the receiver back and so do both compiled runtimes -- and `_text`
    # already returns the same object, so a fresh handle for it was the
    # whole of the difference. See `_value` for why one handle per object.
    return _user(h, lambda: h._value(h._text(h._get(a[0], "apy_str"), False)))


def _apy_print_with(h, a):
    """`print(..., sep='-', end='!')`. A separator or terminator of None means
    the default, which is what an omitted one lowers to -- so "not given" and
    "given as None" are the same request."""
    addr, n = int(a[0]), int(a[1])
    # A str SUBCLASS IS A SEPARATOR -- see `_held_text`.
    sep = _held_text(h._get(a[2], "apy_print_with"))
    end = _held_text(h._get(a[3], "apy_print_with"))
    parts = []
    for i in range(n):
        handle = h._interp.mem.read(addr + i * 8, _PTR)
        parts.append(h._text(h._get(handle, "apy_print_with"), False))
    text = (sep if isinstance(sep, str) else " ").join(parts)
    h._interp._emit(text + (end if isinstance(end, str) else "\n"))
    return None


def _apy_print(h, a):
    """`items` is the ADDRESS of an array of n handles, not a value.

    The IR has no varargs, so the frontend builds the array in a stack slot
    and passes its address -- the same shape the C reads.
    """
    addr, n = int(a[0]), int(a[1])
    parts = []
    for i in range(n):
        handle = h._interp.mem.read(addr + i * 8, _PTR)
        parts.append(h._text(h._get(handle, "apy_print"), False))
    h._interp._emit(" ".join(parts) + "\n")
    return None


# ── sequences ───────────────────────────────────────────────────────────────

def _apy_list_new(h, a):
    return h._new([])


def _apy_tuple_new(h, a):
    return h._new(())


def _apy_seq_push(h, a):
    """Append, for both the list and the tuple builder.

    A tuple literal is `apy_tuple_new` followed by one push per element, and a
    Python tuple has nothing to push onto. So the CELL is replaced with a
    longer tuple each time, which is quadratic in the element count and is the
    right trade here: the handle does not change, so identity survives exactly
    as it does in the C (which mutates one cell in place), and every reader --
    `repr`, `len`, `==`, `hash`, iteration -- sees a REAL tuple with no
    finalisation step that could be forgotten at one call site.

    The first shape of this held a mutable builder and converted it when
    something asked. That is one more state than the C has, and the extra
    state escaped: a tuple pushed into a list was stored still-unconverted and
    printed as the builder's `repr`.
    """
    seq = h._get(a[0], "apy_seq_push")
    item = h._get(a[1], "apy_seq_push")
    if isinstance(seq, tuple):
        # THE CELL IS REPLACED, so the IDENTITY TABLE HAS TO MOVE WITH IT.
        # A tuple is interned by `id(obj)` (see `_INTERNED`), and the old
        # tuple is unreferenced the moment this line runs -- CPython is
        # free to free it and hand its `id` to something else, which
        # would then resolve to THIS handle. Popping the old entry and
        # adding the new one is what keeps that from happening.
        made = seq + (item,)
        h._cells[int(a[0]) - _HANDLE_BASE] = made
        h._identity.pop(id(seq), None)
        h._identity[id(made)] = int(a[0])
        # The tuple holds a reference now, like the list below, and
        # `_held_by` releases it when the tuple itself reaches zero.
        h.incref(int(a[1]))
        return h._none
    if isinstance(seq, bytearray):
        # A BYTEARRAY APPENDS AN INT, and refuses anything else with the
        # message CPython uses. It was missing entirely -- `.append` on one
        # answered "'bytearray' object has no attribute 'append'" about the
        # most ordinary way there is to build one up, which
        # `bundled/subprocess.py` found and had to work around. NOT
        # REFERENCE COUNTED, because an `int` is not a tracked handle: see
        # `_refcount`'s own comment on why plain scalars are out of scope.
        byte = _byte_arg(h, item)
        if byte is None:
            return 0
        seq.append(byte)
        return h._none
    if not isinstance(seq, list):
        return h._fail(
            "AttributeError",
            f"'{h.kind_name(seq)}' object has no attribute 'append'")
    seq.append(item)
    # THE LIST NOW HOLDS A REFERENCE TOO, not only whatever register or
    # slot handed `item` here -- `xs = [Foo()]` finalizing `Foo()` the
    # moment the CALL result's own temp register was consumed (see
    # `_consume` in `interpreter.py`), before `xs.append` ever ran, is
    # exactly the bug this closes. `a[1]` is the ORIGINAL handle, still
    # the right one to count against even though `item` above is the
    # dereferenced value `_decref_contents` will later look back up via
    # `_referent`.
    h.incref(int(a[1]))
    return h._none


def _apy_getitem(h, a):
    seq = h._get(a[0], "apy_getitem")
    index = h._get(a[1], "apy_getitem")
    if isinstance(seq, Instance) and seq.cls is _SPECIAL_FORM_CLASS:
        # `Literal["a", "b"]`, `TypeGuard[int]` -- PARAMETERISING a typing
        # form, which is the same thing `list[int]` does to a builtin type.
        # Above the general instance path, which would look for a
        # `__getitem__` the form has no reason to define.
        args = index if isinstance(index, tuple) else (index,)
        if _form_name(seq) == "Optional":
            # `Optional[X]` IS `X | None`. 3.14 unified the two spellings, so
            # a program that prints the annotation sees the union rather than
            # the form it was written with, and `get_args` answers two arms.
            arms = []
            for one in args:
                arms.extend(_union_arms(one))
            arms.extend(_union_arms(None))
            return h._new(Alias(_union_form(h), tuple(_dedup_arms(arms))))
        return h._new(Alias(seq, args))
    if isinstance(seq, Instance):
        if seq.cls.find("__getitem__") is None and seq.held is not None:
            # A CLASS THAT EXTENDS A BUILTIN IS one for everything it did not
            # write. `class D(dict)` with only a `__missing__` still has to
            # answer `d[k]`, and this is the dict it answers from.
            if isinstance(seq.held, dict):
                if index in seq.held:
                    return h._value(seq.held[index])
                # `__missing__` IS WHAT A dict SUBCLASS IS FOR: a key that is
                # not there is the class's question to answer.
                if seq.cls.find("__missing__") is not None:
                    return _user(h, lambda: h._value(
                        seq._send("__missing__", index)))
                return h._fail("KeyError", h._text(index, True))
            return _apy_getitem(h, [h._new(seq.held), a[1]])
        return _user(h, lambda: h._value(seq[index]))
    if isinstance(seq, dict):
        return _dict_get(h, seq, index)
    if isinstance(seq, Func) and getattr(seq, "is_type", False):
        # `list[int]` -- PARAMETERISING a builtin type, not indexing it.
        args = index if isinstance(index, tuple) else (index,)
        return h._new(Alias(seq, args))
    if isinstance(seq, Class):
        # `C[int]`. A CLASS IS NOT A CONTAINER: subscripting one asks
        # `__class_getitem__`, an implicit classmethod, and a class without it
        # is not subscriptable at all -- CPython says so, and parameterising
        # silently would turn a mistake into an object.
        hook = seq.find("__class_getitem__")
        if hook is None and seq.meta is not None:
            # THE METACLASS DECIDES, if it has an opinion: `Box[int]` where
            # `Box` inherits `Generic` is `type(Box).__getitem__(Box, int)`,
            # which is how a generic class is parameterised without every
            # class in the program becoming subscriptable.
            m = seq.meta.lookup("__getitem__")
            if m is not _ABSENT:
                return _user(h, lambda: h._value(h._invoke(m, [seq, index])))
        if hook is None:
            return h._fail("TypeError",
                           f"type '{seq.name}' is not subscriptable")
        return _user(h, lambda: h._value(h._invoke(hook, [seq, index])))
    if isinstance(seq, memoryview):
        # A SLICE OF A VIEW IS STILL A VIEW -- `mv[1:3][0] = 9` writes to the
        # original buffer, which a copy here would silently lose.
        try:
            return h._new(seq[index])
        except (TypeError, IndexError) as exc:
            return h._fail_like(exc)
    if isinstance(index, slice) and isinstance(seq, (list, tuple, str, bytes)):
        # `xs[slice(1, 5)]`. `xs[1:5]` never comes this way -- the frontend
        # slices it directly -- but a slice built as a VALUE has to work as a
        # subscript too.
        return h._new(seq[index])
    if not _is_int_like(index) and isinstance(index, Instance)             and index.cls.find("__index__") is not None:
        # `__index__` -- how a user object BECOMES an index. PEP 357, and a
        # separate dunder from `__int__` for a reason: a float has `__int__`
        # and is still not a valid subscript.
        try:
            got = index._send("__index__")
        except _UserFailed:
            return 0
        if _is_int_like(got):
            index = got
    if not _is_int_like(index):
        # TWO TEXTS, NOT ONE with a substituted noun, and the C has the same
        # pair: CPython says `list indices must be integers or slices, not
        # float` for a list or a tuple -- both of which DO accept a slice --
        # and `string indices must be integers, not 'float'` for a str, with
        # the kind quoted and no mention of slices. `byte`, SINGULAR, is what
        # it calls a bytes.
        if isinstance(seq, str):
            return h._fail("TypeError",
                           f"string indices must be integers, "
                           f"not '{h.kind_name(index)}'")
        named = ("byte" if isinstance(seq, bytes) and
                 not isinstance(seq, bytearray) else h.kind_name(seq))
        return h._fail("TypeError",
                       f"{named} indices must be integers or slices, "
                       f"not {h.kind_name(index)}")
    if not -_INDEX_LIMIT <= index < _INDEX_LIMIT:
        # AN INDEX THAT DOES NOT FIT A MACHINE WORD. The compiled halves
        # cannot hold one and say so; answering `list index out of range`
        # here made the interpreter disagree about a program neither can
        # serve. See `apy_index_arg_of`.
        return h._fail("IndexError",
                       "cannot fit 'int' into an index-sized integer")
    i = int(index)
    if isinstance(seq, (list, tuple)):
        if i < 0:
            i += len(seq)
        if not 0 <= i < len(seq):
            # NAMES THE KIND IT WAS GIVEN, as the C does: `(1,)[9]` is a
            # complaint about a tuple and not about a list.
            return h._fail("IndexError",
                           f"{h.kind_name(seq)} index out of range")
        return h._value(seq[i])
    if isinstance(seq, (bytes, bytearray)):
        # An INT, not a one-byte bytes. Slicing gives bytes back and indexing
        # does not, which is the one asymmetry a reader will not expect and
        # the C reproduces in `apy_bytes_getitem`.
        #
        # BYTEARRAY TOO: the C represents bytes and bytearray as one kind
        # with a mutable flag (`APY_BYTES_K`), so `apy_bytes_getitem` never
        # distinguished them and `ba[i]` has always returned an int there.
        # This `isinstance(seq, bytes)` check did distinguish them -- Python's
        # own `bytearray` is not a `bytes` subclass -- so `ba[i]` fell through
        # every case below and reported "'bytearray' object is not
        # subscriptable" for a value the C backend indexes correctly. Found
        # writing `struct.unpack_from`, which reads a buffer byte by byte and
        # is the ordinary way a bytearray is read at all.
        if i < 0:
            i += len(seq)
        if not 0 <= i < len(seq):
            return h._fail("IndexError", "index out of range")
        return h._int(seq[i])
    if isinstance(seq, range):
        # ARITHMETIC, not a walk: the element at an index is one
        # multiplication whatever the range's length.
        try:
            return h._new(seq[i])
        except IndexError:
            return h._fail("IndexError", "range object index out of range")
    if isinstance(seq, str):
        # BY CHARACTER, as the C now indexes. Both paths used to index BYTES,
        # so `s[1]` on a string with a non-ASCII character in it was the first
        # HALF of one -- identical to CPython for ASCII and wrong for anything
        # else, in both paths equally, which is why nothing caught it.
        if i < 0:
            i += len(seq)
        if not 0 <= i < len(seq):
            return h._fail("IndexError", "string index out of range")
        return h._new(seq[i])
    return h._fail("TypeError",
                   f"'{h.kind_name(seq)}' object is not subscriptable")


def _apy_setitem(h, a):
    seq = h._get(a[0], "apy_setitem")
    index = h._get(a[1], "apy_setitem")
    item = h._get(a[2], "apy_setitem")
    if isinstance(seq, Instance) and seq.cls.find("__setitem__") is None             and seq.held is not None:
        # A CLASS THAT EXTENDS A BUILTIN writes into the one it carries.
        return _apy_setitem(h, [h._new(seq.held), a[1], a[2]])
    if isinstance(seq, Instance):
        def store():
            seq[index] = item
            return h._none
        return _user(h, store)
    if isinstance(seq, dict):
        return _dict_set(h, seq, index, item)
    if isinstance(index, slice) and isinstance(seq, list):
        # THE SPAN IS REPLACED, and the replacement need not be the same
        # length -- `xs[1:3] = [9]` shortens the list. In place, so every
        # other name bound to it sees the change.
        if index.step not in (None, 1):
            return h._fail("ValueError",
                           "only step 1 slice assignment is supported")
        try:
            seq[index] = list(item)
        except TypeError as exc:
            return h._fail_like(exc)
        return h._none
    if isinstance(seq, (bytearray, memoryview)):
        # THROUGH to the buffer, which for a memoryview is someone else's --
        # that write being visible in the original is what a view is for.
        # Python's own types raise exactly what CPython raises here, so the
        # bounds, the range and the read-only refusal are not restated.
        try:
            seq[int(index)] = int(item)
        except (TypeError, ValueError, IndexError) as exc:
            return h._fail_like(exc)
        return h._none
    if not isinstance(seq, list):
        return h._fail(
            "TypeError",
            f"'{h.kind_name(seq)}' object does not support item assignment")
    if not _is_int_like(index):
        return h._fail("TypeError",
                       f"list indices must be integers, "
                       f"not '{h.kind_name(index)}'")
    i = int(index)
    if i < 0:
        i += len(seq)
    if not 0 <= i < len(seq):
        return h._fail("IndexError", "list assignment index out of range")
    # NEW BEFORE OLD, same reason as everywhere else this pattern
    # appears: `xs[i] = xs[i]` has the same handle on both sides, and
    # decrefing the slot's old occupant before the new one has its own
    # reference would pass through zero for something not actually
    # going away.
    old = seq[i]
    h.incref(int(a[2]))
    seq[i] = item
    old_h = h._referent(old)
    if old_h is not None:
        h.decref(old_h)
    return h._none


# ── dict ────────────────────────────────────────────────────────────────────
# A real Python dict, so insertion order, `d[1] is d[True]` key identity and
# order-free equality all come free -- and all three are things the C had to
# be written carefully to get right.

# Hashability is left to Python's own dict FOR BUILT-IN KINDS, whose message
# is already 3.14's exact text -- `cannot use 'tuple' as a dict key
# (unhashable type: 'list')`, naming the key's kind and then the innermost
# offender, recursive through tuples. Reproducing that would be three rules to
# keep in step for no gain.
#
# A USER OBJECT IS THE EXCEPTION, and the reason is that the message names a
# TYPE: Python sees the key as an `objects_host.Instance` and says so, leaking
# this file's internals into a message a program prints -- `cannot use
# 'uasm.objects.host.Instance' as a dict key`, where a compiled binary
# says `OnlyEq2`. So instances are checked here, with the C's wording.


def _unhashable_name(v):
    """The class name of an unhashable USER OBJECT, or None.

    A class defining `__eq__` and not `__hash__` is unhashable -- the same
    rule `Instance.__hash__` enforces, asked before the container tries.
    """
    if isinstance(v, Instance) and v.cls.find("__hash__") is None             and v.cls.find("__eq__") is not None:
        return v.cls.name
    if isinstance(v, tuple):
        for item in v:
            got = _unhashable_name(item)
            if got:
                return got
    return None


def _apy_dict_new(h, a):
    return h._new({})


def _dict_set(h, d, key, val):
    bad = _unhashable_name(key)
    if bad:
        return h._fail("TypeError", f"cannot use '{bad}' as a dict key "
                                    f"(unhashable type: '{bad}')")
    had_key = key in d
    old = d.get(key) if had_key else None
    try:
        d[key] = val
    except TypeError as e:
        return h._fail_like(e)
    # THE DICT NOW HOLDS ITS OWN REFERENCE to `val` (and to `key`, the
    # first time this key is used) -- `h._value` recovers the SAME
    # handle `_referent`/`decref` would later look it up as, whether it
    # already existed (an object stored elsewhere too) or is minted here
    # for the first time. NEW BEFORE OLD, same reason as `put()`'s own
    # comment: `d[k] = d[k]` has the same handle on both sides.
    h.incref(h._value(val))
    if not had_key:
        h.incref(h._value(key))
    if had_key:
        old_h = h._referent(old)
        if old_h is not None:
            h.decref(old_h)
    return h._none


def _attr_store(h, d: dict, name: str, value) -> None:
    """`d[name] = value` for an object's own attribute dict
    (`Instance.dict`/`Exc.dict`/`Class.dict`/`Func.dict`), with the same
    incref-new/decref-old bookkeeping `_dict_set` does for a Python
    dict -- an attribute holds a reference exactly as a dict value does.
    `name` is never incref'd: it is always a plain `str`, and a `str`
    handle is never tracked (see `_INTERNED`) -- there is nothing there
    for `incref` to do.
    """
    had = name in d
    old = d.get(name) if had else None
    d[name] = value
    h.incref(h._value(value))
    if had:
        old_h = h._referent(old)
        if old_h is not None:
            h.decref(old_h)


def _apy_clear(h, a):
    """`.clear()` -- empties in place and answers None."""
    v = h._get(a[0], "apy_clear")
    if isinstance(v, bytearray):
        v.clear()
        return h._none
    if not isinstance(v, (list, dict, set)):
        return h._fail("AttributeError",
                       f"'{h.kind_name(v)}' object has no attribute 'clear'")
    # EVERY REFERENCE `v` HELD IS DROPPED, same walk `_decref_contents`
    # already does for a container reaching zero itself -- `.clear()` is
    # exactly that, just without `v` itself going away too.
    h._decref_contents(v)
    v.clear()
    return h._none


def _apy_copy(h, a):
    """`.copy()` -- SHALLOW, like Python's. A frozenset's copy is itself,
    which is what CPython returns and is safe for the same reason
    `frozenset(f)` is."""
    v = h._get(a[0], "apy_copy")
    if isinstance(v, frozenset):
        return a[0]
    # A BYTEARRAY COPIES ITS BYTES and `bytes` has no `copy` at all -- the
    # compiled runtimes share one cell between the two and decide on the
    # `mut` flag, which here is the Python type itself.
    if isinstance(v, bytearray):
        return h._new(bytearray(v))
    if not isinstance(v, (list, dict, set)):
        return h._fail("AttributeError",
                       f"'{h.kind_name(v)}' object has no attribute 'copy'")
    return h._new(v.copy())


def _apy_hash(h, a):
    """`hash(x)`. THE VALUES ARE NOT CPYTHON'S -- CPython salts str and bytes
    hashes per process, so there is no fixed number to match. What must hold
    is the CONTRACT: equal values hash equally, and this file and the C agree
    with each other because both go through Python's own `hash` on the same
    objects."""
    v = h._get(a[0], "apy_hash")
    if isinstance(v, Instance):
        return _user(h, lambda: h._int(hash(v)))
    try:
        return h._int(hash(v))
    except TypeError as exc:
        return h._fail_like(exc)


def _byte_arg(h, v):
    """The octet a bytearray method's argument stands for, or None raising.

    A BYTEARRAY HOLDS NUMBERS AND NOT ONE-BYTE STRINGS -- `b.append(b"z")` is
    a TypeError and `b.append(300)` a ValueError -- and `append`, `insert`
    and `remove` all say it the same way.
    """
    # A BOOL IS AN INT, in Python and in both compiled runtimes:
    # `b.append(True)` appends 1 rather than refusing. Excluding it made this
    # the one path of the three that said no.
    if not isinstance(v, int):
        h._fail("TypeError", f"'{h.kind_name(v)}' object cannot be "
                             f"interpreted as an integer")
        return None
    if v < 0 or v > 255:
        h._fail("ValueError", "byte must be in range(0, 256)")
        return None
    return v


def _apy_iadd(h, a):
    """`x += y` -- NOT sugar for `x = x + y`.

    A list EXTENDS ITSELF and hands itself back, so every other name bound to
    it sees the new elements; a tuple has no in-place form and falls through to
    `+`, which builds a new one. That difference is observable from another
    frame, which is why it cannot be a frontend rewrite.
    """
    x = h._get(a[0], "apy_iadd")
    y = h._get(a[1], "apy_iadd")
    if isinstance(x, Instance) and x.cls.find("__iadd__") is not None:
        try:
            got = x._send("__iadd__", y)
        except _UserFailed:
            return 0
        if got is not NotImplemented:
            return h._value(got)
    if isinstance(x, list):
        # `xs += it` IS `list.__iadd__`, which is `list_extend` -- so it asks
        # the argument for a length hint, exactly as `xs.extend(it)` does.
        # A bytearray below does not, and neither does a tuple.
        got = _apy_extend_meth(h, [a[0], a[1]])
        return a[0] if got else 0
    # A BYTEARRAY EXTENDS ITSELF TOO, and for the same reason -- it is
    # mutable, so `b += data` has to be visible through every other name for
    # it. Falling through to `apy_add` built a NEW bytearray and rebound the
    # one name that was written. ONLY FROM SOMETHING BYTES-LIKE: CPython
    # refuses a list of ints with `can't concat list to bytearray`.
    if isinstance(x, bytearray):
        if not isinstance(y, (bytes, bytearray, memoryview)):
            return h._fail("TypeError",
                           f"can't concat {h.kind_name(y)} to bytearray")
        x.extend(bytes(y))
        return a[0]
    return _TABLE["apy_add"](h, a)


_IOP_DUNDER = {"|": "__ior__", "&": "__iand__", "^": "__ixor__",
               "-": "__isub__", "*": "__imul__"}


def _apy_iop(h, a):
    """`s |= other` and the rest of the in-place set and dict operators."""
    x = h._get(a[0], "apy_iop")
    y = h._get(a[1], "apy_iop")
    op = str(h._get(a[2], "apy_iop"))
    if isinstance(x, Instance):
        name = _IOP_DUNDER[op]
        if x.cls.find(name) is not None:
            try:
                got = x._send(name, y)
            except _UserFailed:
                return 0
            if got is not NotImplemented:
                return h._value(got)
    if isinstance(x, dict) and op == "|":
        got = _apy_update(h, [a[0], a[1]])
        return a[0] if got else 0
    # `xs *= k` REPEATS IN PLACE for the two mutable sequences -- the `*=`
    # half of the rule `+=` follows. Rebinding left every other name for the
    # list holding the old contents.
    if op == "*" and isinstance(x, (list, bytearray)):
        made = _TABLE["apy_mul"](h, [a[0], a[1]])
        if not made:
            return 0
        got = h._get(made, "apy_iop")
        x.clear()
        x.extend(got)
        return a[0]
    if isinstance(x, set):
        if not isinstance(y, (set, frozenset)):
            return h._binop_error(op, x, y)
        # Computed then copied back, rather than mutated as it goes: the two
        # operands may be the SAME set.
        out = ({"|": x | y, "&": x & y, "^": x ^ y}.get(op)
               if op != "-" else x - y)
        x.clear()
        x |= out
        return a[0]
    sym = {"|": "apy_bitor", "&": "apy_bitand", "^": "apy_bitxor",
           "-": "apy_sub", "*": "apy_mul"}[op]
    return _TABLE[sym](h, [a[0], a[1]])


def _apy_list_insert(h, a):
    """`xs.insert(i, v)`. The index is CLAMPED, not checked -- which is why
    `insert` never raises where `xs[i] = v` does."""
    seq = h._get(a[0], "apy_list_insert")
    where = h._get(a[1], "apy_list_insert")
    if isinstance(seq, bytearray):
        # A BYTEARRAY INSERTS AN OCTET, and the position clamps the same way.
        if not isinstance(where, int):
            return h._fail("TypeError",
                           f"'{h.kind_name(where)}' object cannot be "
                           f"interpreted as an integer")
        byte = _byte_arg(h, h._get(a[2], "apy_list_insert"))
        if byte is None:
            return 0
        seq.insert(where, byte)
        return h._none
    if not isinstance(seq, list):
        return h._fail("AttributeError",
                       f"'{h.kind_name(seq)}' object has no attribute 'insert'")
    # A BOOL IS AN INT: `xs.insert(True, 9)` inserts at 1 in Python and in
    # both compiled runtimes, and refusing it here was a three-way split.
    if not isinstance(where, int):
        return h._fail("TypeError", f"'{h.kind_name(where)}' object cannot be "
                                    f"interpreted as an integer")
    seq.insert(where, h._get(a[2], "apy_list_insert"))
    return h._none


def _apy_list_sort(h, a):
    """`xs.sort()` -- IN PLACE and answering None, which is the whole
    difference from `sorted(xs)`."""
    seq = h._get(a[0], "apy_list_sort")
    if not isinstance(seq, list):
        return h._fail("AttributeError",
                       f"'{h.kind_name(seq)}' object has no attribute 'sort'")
    key = h._get(a[1], "apy_list_sort")
    rev = bool(h._get(a[2], "apy_list_sort"))
    try:
        if key is None:
            _driving(h, "<", lambda: seq.sort(reverse=rev))
        else:
            return _user(h, lambda: _driving(h, "<", lambda: (
                seq.sort(key=lambda v: h._invoke(key, [v]), reverse=rev)
                or h._none)))
    except _HOST_RAISES as exc:
        return h._fail_like(exc)
    return h._none


def _apy_list_reverse(h, a):
    seq = h._get(a[0], "apy_list_reverse")
    if isinstance(seq, bytearray):
        seq.reverse()
        return h._none
    if not isinstance(seq, list):
        return h._fail("AttributeError", f"'{h.kind_name(seq)}' object has no "
                                         f"attribute 'reverse'")
    seq.reverse()
    return h._none


def _apy_setdefault(h, a):
    """Read, and INSERT when missing -- one lookup's worth of difference from
    `d.get(k, v)`, and the difference is the whole point."""
    d = h._get(a[0], "apy_setdefault")
    if not isinstance(d, dict):
        return h._fail("AttributeError", f"'{h.kind_name(d)}' object has no "
                                         f"attribute 'setdefault'")
    key = h._get(a[1], "apy_setdefault")
    if key in d:
        return h._value(d[key])
    return _dict_set(h, d, key, h._get(a[2], "apy_setdefault")) and a[2]         or a[2]


def _apy_format(h, a):
    """`format(v, spec)`, `f"{v:spec}"` and `"{:spec}".format(v)` -- one
    function, because they are one mini-language.

    Python's own `format` IS the language, so this file uses it and the C
    reimplements it; the two are checked against each other by the differential
    fuzzer and by the integration corpus, which is the only way a hand-written
    parser of a spec stays honest.
    """
    v = h._get(a[0], "apy_format")
    spec = h._get(a[1], "apy_format")
    # A user object formats ITSELF, and is asked BEFORE the empty-spec
    # shortcut: `f"{obj}"` is `format(obj, "")`, which calls `__format__("")`
    # and not `str(obj)`. A class defining both can tell the difference.
    if isinstance(v, Instance):
        try:
            got = v._send("__format__", spec)
        except _UserFailed:
            return 0
        if h.err is not None:
            return 0
        if got is not NotImplemented:
            return h._value(got)
        return h._new(format(h._text(v, False), spec))
    if not spec:
        return h._new(h._text(v, False))
    if isinstance(v, Exc):
        return h._new(format(h._text(v, False), spec))
    try:
        return h._new(format(v, spec))
    except _HOST_RAISES as exc:
        return h._fail_like(exc)


def _apy_str_format(h, a):
    """`"{} {:>5} {name!r}".format(...)` -- the replacement-field syntax around
    the spec `apy_format` reads."""
    # A str SUBCLASS IS THE FORMAT STRING TOO -- see `_held_text`.
    fmt = _held_text(h._get(a[0], "apy_str_format"))
    if not isinstance(fmt, str):
        return h._fail("AttributeError",
                       f"'{h.kind_name(fmt)}' object has no attribute 'format'")
    args = list(h._get(a[1], "apy_str_format"))
    kw = dict(h._get(a[2], "apy_str_format"))
    try:
        return h._new(fmt.format(*args, **kw))
    except (IndexError, KeyError, ValueError, TypeError) as exc:
        return h._fail_like(exc)


#: The codec names the C canonicalises. Kept here so the two agree about
#: which spellings exist -- a name one accepts and the other refuses is a
#: divergence in what a program is allowed to write, not in what it computes.
_CODECS = {
    "utf-8": "utf-8", "utf8": "utf-8", "u8": "utf-8",
    "ascii": "ascii", "us-ascii": "ascii", "646": "ascii",
    "latin-1": "latin-1", "latin1": "latin-1", "iso-8859-1": "latin-1",
    "l1": "latin-1", "8859": "latin-1",
    "utf-16": "utf-16", "utf16": "utf-16",
    "utf-16-le": "utf-16-le", "utf-16le": "utf-16-le",
    "utf-16-be": "utf-16-be", "utf-16be": "utf-16-be",
    "utf-32": "utf-32", "utf32": "utf-32",
    "utf-32-le": "utf-32-le", "utf-32le": "utf-32-le",
    "utf-32-be": "utf-32-be", "utf-32be": "utf-32-be",
}


def _codec_args(h, a, who):
    """The encoding and error handler a call named, canonicalised.

    Answers None for a name no codec matches, which is a LookupError and not a
    silent fall back to UTF-8 -- the C refuses it and so must this.
    """
    # A str SUBCLASS NAMES A CODEC -- see `_held_text`.
    enc = _held_text(h._get(a[1], who)) if len(a) > 1 else None
    err = _held_text(h._get(a[2], who)) if len(a) > 2 else None
    name = _CODECS.get(str(enc).lower().replace("_", "-")) if enc is not None         else "utf-8"
    handler = str(err) if err is not None else "strict"
    # EVERY HANDLER CPYTHON ANSWERS WITHOUT A REGISTERED CALLBACK, which is
    # the same set `apy_errors_of` knows. `namereplace` is the one left out:
    # it spells a character by its UNICODE NAME, and the name table is a
    # bundled module rather than something the runtime carries.
    if handler not in ("strict", "replace", "ignore", "backslashreplace",
                       "xmlcharrefreplace", "surrogateescape",
                       "surrogatepass"):
        handler = "strict"
    return (name, handler)


def _apy_str_encode(h, a):
    # A str SUBCLASS IS A str HERE TOO -- see `_held_text`. The method
    # spelling arrives unwrapped through `apy_method_self`; the constructor
    # `bytes(S("ab"), "utf-8")` reaches this with the instance itself.
    v = _held_text(h._get(a[0], "apy_str_encode"))
    if not isinstance(v, str):
        return h._fail("AttributeError", f"'{h.kind_name(v)}' object has no "
                                         f"attribute 'encode'")
    if _codec_given(h, "encode", "encoding", a[1]):
        return 0
    if _codec_given(h, "encode", "errors", a[2]):
        return 0
    name, handler = _codec_args(h, a, "apy_str_encode")
    if name is None:
        return h._fail("LookupError",
                       f"unknown encoding: {h._get(a[1], 'encode')}")
    try:
        return h._new(v.encode(name, handler))
    except UnicodeEncodeError as exc:
        # CPYTHON'S OWN SENTENCE, character, position and reason included:
        # `'utf-8' codec can't encode character '\udcff' in position 1:
        # surrogates not allowed`. It used to stop after `character`, which
        # is the half that says nothing.
        return h._fail("UnicodeEncodeError", str(exc))


def _codec_arg(h, who, slot, handle):
    """Refuse a non-string encoding or errors, in the CONSTRUCTOR's wording.

    NAMES THE PARAMETER where the method family beside it NUMBERS it, which
    is CPython's own split. None is not checked: it is how "not given"
    travels to the codec pair, and `str(b, errors="replace")` defaults the
    encoding to UTF-8 rather than refusing.
    """
    v = _held_text(h._get(handle, "apy_codec_arg"))
    if v is None or isinstance(v, str):
        return 0
    h._fail("TypeError", f"{who}() argument {slot!r} must be str, "
                         f"not {h.kind_name(v)}")
    return 1


def _codec_or(h, handle, filled: str):
    """The handle, or one holding `filled` where the slot holds None -- the
    constructor's reading of a slot the call never wrote."""
    if handle and h._get(handle, "apy_codec_arg") is not None:
        return handle
    return h._new(filled)


def _codec_given(h, who, slot, handle):
    """The same, WHERE NONE IS NOT A DEFAULT EITHER.

    `"a".encode(None)` is a TypeError in CPython and `"a".encode()` is
    `"a".encode("utf-8")` -- the two are told apart by the lowering, which
    pads a slot the call left out with the DEFAULT'S OWN TEXT rather than
    with None. The constructors are the exception and keep `_codec_arg`: a
    None encoding there means the slot was never written.

    `not None` AND NOT `not NoneType`, because the arg clinic writes the
    VALUE for these two and the type for everything else.
    """
    v = _held_text(h._get(handle, "apy_codec_arg")) if handle else None
    if isinstance(v, str):
        return 0
    h._fail("TypeError",
            f"{who}() argument {slot!r} must be str, "
            f"not {'None' if v is None else h.kind_name(v)}")
    return 1


def _apy_bytes_ctor(h, a):
    """`bytes(s, encoding)` and `bytearray(s, encoding, errors)` -- THE
    CONSTRUCTOR SPELLING OF `.encode()`, and a different constructor from the
    one-argument form: `bytes(xs)` is a sequence of octets and
    `bytes(s, "utf-8")` is `s.encode("utf-8")`.

    A NON-STR WITH AN ENCODING IS THE OTHER HALF of the refusal the
    one-argument form already makes.
    """
    who = "bytearray" if int(a[3]) else "bytes"
    if _codec_arg(h, who, "encoding", a[1]):
        return 0
    if _codec_arg(h, who, "errors", a[2]):
        return 0
    v = _held_text(h._get(a[0], "apy_bytes_ctor"))
    if not isinstance(v, str):
        return h._fail("TypeError", "encoding without a string argument")
    # A NONE SLOT IS ONE THE CALL NEVER WROTE, which is the constructor's own
    # reading of it -- so the default is filled in HERE rather than accepted
    # by `encode`, which refuses a None the way CPython does for the method
    # spelling.
    made = _apy_str_encode(h, [a[0], _codec_or(h, a[1], "utf-8"),
                               _codec_or(h, a[2], "strict")])
    if not made:
        return 0
    if int(a[3]):
        return h._new(bytearray(h._get(made, "apy_bytes_ctor")))
    return made


def _apy_str_ctor(h, a):
    """`str(b, encoding)` -- the constructor spelling of `.decode()`, and the
    same split: `str(b)` is the REPR of the bytes and `str(b, "utf-8")` is
    the text they spell."""
    if _codec_arg(h, "str", "encoding", a[1]):
        return 0
    if _codec_arg(h, "str", "errors", a[2]):
        return 0
    v = h._get(a[0], "apy_str_ctor")
    if not isinstance(v, (bytes, bytearray, memoryview)):
        return h._fail("TypeError", f"decoding to str: need a bytes-like "
                                    f"object, {h.kind_name(v)} found")
    # The same, and for the same reason -- see `_apy_bytes_ctor`. A VIEW IS A
    # BYTES-LIKE OBJECT HERE, which `decode` itself no longer accepts as a
    # receiver.
    held = h._new(bytes(v)) if isinstance(v, memoryview) else a[0]
    return _apy_bytes_decode(h, [held, _codec_or(h, a[1], "utf-8"),
                                 _codec_or(h, a[2], "strict")])


def _apy_bytes_decode(h, a):
    v = h._get(a[0], "apy_bytes_decode")
    # A VIEW IS NOT A RECEIVER FOR THIS. `memoryview(b"a").decode()` is an
    # AttributeError in CPython -- a view has no `decode` -- and converting
    # one here answered the text instead. The CONSTRUCTOR spelling does take
    # a view, and `_apy_str_ctor` converts it before calling.
    if not isinstance(v, (bytes, bytearray)):
        return h._fail("AttributeError", f"'{h.kind_name(v)}' object has no "
                                         f"attribute 'decode'")
    if _codec_given(h, "decode", "encoding", a[1]):
        return 0
    if _codec_given(h, "decode", "errors", a[2]):
        return 0
    name, handler = _codec_args(h, a, "apy_bytes_decode")
    if name is None:
        return h._fail("LookupError",
                       f"unknown encoding: {h._get(a[1], 'decode')}")
    try:
        return h._new(bytes(v).decode(name, handler))
    except UnicodeDecodeError as exc:
        # CPYTHON'S OWN SENTENCE again -- the byte, the position and which of
        # its three reasons it was. See `apy_utf8_why` for the C's copy.
        return h._fail("UnicodeDecodeError", str(exc))


#: str, bytes and bytearray -- the three receivers that walk TEXT rather than
#: elements, and the list several gates here were missing the last of.
#: `isinstance(bytearray(b""), bytes)` is False in Python, so every one of
#: them refused a bytearray for a method it plainly has.
_TEXTY = (str, bytes, bytearray)


def _apy_bytes_hex(h, a):
    """`b.hex()` and `b.hex(sep)`. ONE BYTE PER GROUP, which is the default
    CPython declares -- see `_apy_bytes_hex_n`, which both spellings are.
    The count slot is EMPTY because this call did not write one."""
    return _apy_bytes_hex_n(h, [a[0], a[1], 0])


def _apy_bytes_hex_n(h, a):
    """`b.hex()`, `b.hex(sep)` and `b.hex(sep, bytes_per_sep)`.

    THE GROUPING IS COUNTED FROM AN END AND WHICH END IS THE SIGN -- see
    `apy_bytes_hex_n` in the C. Python's own `bytes.hex` has the rule, so
    the count is handed straight to it rather than reimplemented.
    """
    b = h._get(a[0], "apy_bytes_hex")
    # AN EMPTY SLOT IS ONE THE CALL DID NOT WRITE, and it is the only way to
    # tell `b.hex()` from `b.hex(None)` -- the first is the no-argument form
    # whose default gets filled in, the second a separator CPython refuses.
    sep = h._get(a[1], "apy_bytes_hex") if a[1] else None
    per = h._get(a[2], "apy_bytes_hex") if a[2] else None
    # A VIEW HEXES THE BYTES IT SHOWS, which is most of what a program makes
    # one to look at.
    if isinstance(b, memoryview):
        b = b.tobytes()
    if not isinstance(b, (bytes, bytearray)):
        return h._fail("AttributeError",
                       f"'{h.kind_name(b)}' object has no attribute 'hex'")
    # THE COUNT IS CONVERTED FIRST, which is CPython's order: `b.hex(None,
    # None)` complains about the integer and `b.hex(None)` about the
    # separator's length.
    if a[2] and not isinstance(per, int):
        return h._fail("TypeError",
                       f"'{h.kind_name(per)}' object cannot be interpreted "
                       f"as an integer")
    if sep is None and not a[1]:
        return h._new(b.hex())
    # CPython ASKS THE SEPARATOR FOR ITS LENGTH, so anything without one is a
    # TypeError about `len` rather than about hex.
    if not isinstance(sep, (str, bytes, bytearray)):
        return h._fail("TypeError",
                       f"object of type '{h.kind_name(sep)}' has no len()")
    if len(sep) != 1:
        return h._fail("ValueError", "sep must be length 1.")
    if per is None:
        return h._new(b.hex(sep))
    return h._new(b.hex(sep, per))


def _apy_bytes_fromhex(h, a):
    # The RECEIVER is `a[0]` and ignored: the shape matches the method table's
    # so that `b.fromhex(s)` and `bytes.fromhex(s)` are one implementation.
    held = h._get(a[0], "apy_bytes_fromhex")
    text = h._get(a[1], "apy_bytes_fromhex")
    if not isinstance(text, str):
        return h._fail("TypeError", "fromhex() argument must be str")
    try:
        got = bytes.fromhex(text)
    except ValueError as exc:
        return h._fail_like(exc)
    # AND THE ANSWER IS THE KIND IT WAS REACHED THROUGH.
    return h._new(bytearray(got) if isinstance(held, bytearray) else got)


def _mview_live(h, view) -> bool:
    """Whether the view still has its buffer, raising here if it does not.

    `release` drops the borrow so the object underneath can be resized again,
    and every operation on the view from then on is refused.
    """
    try:
        view.nbytes
    except ValueError as exc:
        h._fail_like(exc)
        return False
    return True


def _mview_call(h, view, name, args):
    """One of a view's methods, with Python's own failures passed through.

    THE REFUSALS ARE PART OF THE METHOD: `del m[0]` is "cannot delete memory"
    and `m[0] = 1` on a read-only view is "cannot modify read-only memory",
    and both are worth reaching rather than being an AttributeError about a
    method a memoryview plainly has.
    """
    try:
        return getattr(view, name)(*args)
    except (TypeError, ValueError, IndexError) as exc:
        h._fail_like(exc)
        raise _UserFailed


def _apy_any_fromhex(h, a):
    """Whichever `fromhex` the receiver meant.

    A float's is a different reading entirely -- `0x1.8p+0` is one number,
    not three bytes -- and the receiver is the only thing that says which was
    written. THE SPLIT LIVES HERE rather than inside the bytes body because
    that body is IR, and the ported runtime has to define everything it
    calls: the hex float parser is the C's.
    """
    if isinstance(h._get(a[0], "apy_any_fromhex"), float):
        return _apy_float_fromhex(h, [a[1]])
    return _apy_bytes_fromhex(h, a)


def _apy_bytearray_fromhex(h, a):
    """`bytearray.fromhex(text)` -- the same reading, a MUTABLE answer.

    CPython gives back the kind the method was reached through, so this is a
    bytearray where the shared body above hands out bytes.
    """
    got = _apy_bytes_fromhex(h, a)
    if not got:
        return 0
    return h._new(bytearray(h._get(got, "apy_bytearray_fromhex")))


def _apy_to_bytes_n(h, a):
    v = h._get(a[0], "apy_to_bytes_n")
    length = h._get(a[1], "apy_to_bytes_n")
    order = h._get(a[2], "apy_to_bytes_n")
    # THE THIRD ARGUMENT IS `signed`, and it is always supplied: lowering pads
    # every `to_bytes` call to three, so the keyword-only parameter reaches
    # here as a position rather than being dropped on the way.
    signed = h._get(a[3], "apy_to_bytes_n")
    if not _is_int_like(v):
        return h._fail("AttributeError", f"'{h.kind_name(v)}' object has no "
                                         f"attribute 'to_bytes'")
    # CPYTHON'S OWN TWO REFUSALS, which are not one: the LENGTH is named by
    # the value that was handed over -- the wording every index-taking slot
    # uses -- and the BYTEORDER by the parameter, the arg clinic's form.
    if not _is_int_like(length):
        return h._fail("TypeError", f"'{h.kind_name(length)}' object cannot "
                                    f"be interpreted as an integer")
    if not isinstance(order, str):
        return h._fail("TypeError",
                       f"to_bytes() argument 'byteorder' must be str, not "
                       f"{'None' if order is None else h.kind_name(order)}")
    try:
        return h._new(int(v).to_bytes(
            int(length), "little" if order == "little" else "big",
            signed=bool(signed)))
    except (OverflowError, ValueError) as exc:
        return h._fail_like(exc)


def _apy_as_integer_ratio(h, a):
    """The EXACT fraction the double holds. `0.1` is not one tenth, and this
    is the method that says so."""
    v = h._get(a[0], "apy_as_integer_ratio")
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        if not isinstance(v, int):
            return h._fail("AttributeError",
                           f"'{h.kind_name(v)}' object has no attribute "
                           f"'as_integer_ratio'")
    try:
        return h._new(v.as_integer_ratio())
    except (OverflowError, ValueError) as exc:
        return h._fail_like(exc)


def _apy_str_expandtabs(h, a):
    """`s.expandtabs(width)`. BYTES TOO -- `b"a\tb".expandtabs()` is an
    ordinary call in Python, and a column there is a BYTE rather than a
    character, which is the only thing the receiver's kind changes.

    THE WIDTH IS REQUIRED AND MUST BE AN INTEGER. It arrives always -- the
    frontend supplies 8 for the no-argument form -- so a value that is not an
    integer is one the PROGRAM wrote, and `"a\tb".expandtabs(None)` is a
    TypeError in Python. Reading a non-integer as the default made the
    written None answer what the omitted argument answers.
    """
    s = h._get(a[0], "apy_str_expandtabs")
    width = h._get(a[1], "apy_str_expandtabs")
    if not isinstance(s, (str, bytes, bytearray)):
        return h._fail("AttributeError", f"'{h.kind_name(s)}' object has no "
                                         f"attribute 'expandtabs'")
    if not _is_int_like(width):
        return h._fail("TypeError", f"'{h.kind_name(width)}' object cannot "
                                    f"be interpreted as an integer")
    return h._new(s.expandtabs(int(width)))


#: The one class every `typing` special form is an instance of, made on first
#: use and kept -- so `Final.__class__ is Optional.__class__`, as in CPython.
_SPECIAL_FORM_CLASS = None


def _apy_type_class(h, a):
    """`type` AS A CLASS OBJECT, so `class Meta(type)` has a real base rather
    than a special case. Its dict holds the two natives a metaclass reaches
    through `super()`. Interned: `Meta.__base__ is type` has to hold."""
    made = h._defaults.get("<type>")
    if made is not None:
        return made
    cls = Class("type")
    cls.dict["__new__"] = Native("__new__", lambda mcls, name, bases, ns:
                                 _type_from_ns(h, mcls, name, bases, ns))
    cls.dict["__init__"] = Native("__init__", lambda *a: None)
    # What `C(...)` MEANS, reachable by name so a metaclass's own `__call__`
    # can delegate to it. The dict key is what a program spells; the Native's
    # own name is the marker `_invoke` dispatches on, and it is deliberately
    # unspellable so no user `__call__` is mistaken for this one.
    cls.dict["__call__"] = Native("<type.__call__>", lambda *a: None)
    made = h._new(cls)
    h._defaults["<type>"] = made
    return made


#: The builtin a `class` statement named, waiting for the class to exist:
#: `[name, kind]` or `[None, None]`. See `_apy_type_builtin_pending`, which
#: is where the whole of it is explained. Mirrors the C's two statics.
_BUILTIN_PENDING = [None, None]


def _apy_type_builtin_pending(h, a):
    """`class Colour(str, Enum)` -- the kind, announced before the class is
    built rather than recorded after it.

    `apy_type_builtin` runs once `apy_class_build` has ANSWERED, which for a
    class with a metaclass is after the metaclass body has finished. An
    `EnumMeta` makes every member inside that body, so the members were built
    against a class that did not yet know it extended anything, and
    `isinstance(Colour.RED, str)` was False, `len(Colour.RED)` a TypeError,
    and `Colour.RED == "red"` False.

    THE NAME IS PART OF THE CELL so that a `class` statement inside a
    metaclass body cannot take the tag meant for the class being built: only
    a creation of the same name consumes it. The lowering sets it immediately
    before `apy_class_build` and that call clears it again, so nothing
    outlives one statement.
    """
    # THE KIND IS A PLAIN NUMBER, not a handle -- the lowering passes the
    # enum constant straight through, as it does to `apy_type_builtin`.
    _BUILTIN_PENDING[0] = str(h._get(a[0], "apy_type_builtin_pending"))
    _BUILTIN_PENDING[1] = _BUILTIN_KINDS.get(int(a[1]))
    return h._none


def _type_from_ns(h, mcls, name, bases, ns):
    """`type(name, bases, ns)` as an OBJECT: what `super().__new__` inside a
    metaclass's `__new__` answers."""
    listed = [b for b in bases if isinstance(b, Class)] \
        if isinstance(bases, (list, tuple)) else []
    base = listed[0] if listed else None
    cls = Class(str(name), base, listed)
    # THE KIND IS RECORDED HERE, where a metaclass's `super().__new__`
    # reaches -- see `_apy_type_builtin_pending`.
    if _BUILTIN_PENDING[1] is not None and _BUILTIN_PENDING[0] == str(name):
        cls.builtin = _BUILTIN_PENDING[1]
        _BUILTIN_PENDING[0] = None
        _BUILTIN_PENDING[1] = None
    if listed:
        order = c3(cls, listed)
        if order is None:
            # ANSWERS None, not raises. The flag is set and the caller returns
            # NULL, which is the convention every runtime entry point here
            # follows -- raising out of it would escape the interpreter rather
            # than become an exception the program can catch.
            h._fail("TypeError",
                    "Cannot create a consistent method resolution order "
                    "(MRO) for bases")
            return None
        cls.mro = order
    if isinstance(mcls, Class):
        cls.meta = mcls
    # COPIED IN, not adopted: `__prepare__` may hand back a mapping the
    # program goes on using, and a class sharing it would see later writes.
    if isinstance(ns, dict):
        cls.dict.update(ns)
    # SUPPLIED HERE when the namespace carried none, as CPython's `type_new`
    # supplies it from the caller's globals: a class written out brings one
    # from its body, and one built by `type(name, bases, {})` -- `enum`,
    # `dataclasses` and `namedtuple` all do -- had none at all, so it printed
    # unqualified. There is no frame to ask and only a program's own module
    # can reach this. See `apy_type_from_ns`.
    cls.dict.setdefault("__module__", "__main__")
    return cls


def _meta_for(given, bases):
    """Which metaclass builds this class.

    THE WRITTEN ONE IF THERE IS ONE, otherwise a base's -- a class with no
    `metaclass=` still has one when its base does, and that cannot be answered
    where the class is compiled because it is a property of a run-time value.
    """
    if isinstance(given, Class):
        return given
    if isinstance(bases, (list, tuple)):
        for base in bases:
            if isinstance(base, Class) and base.meta is not None:
                return base.meta
    return None


def _apy_mro_entries(h, a):
    """PEP 560: WHAT A NON-CLASS BASE CONTRIBUTES.

    `class C(Fake())` asks the object for `__mro_entries__(bases)` and
    inherits whatever it answers. A class contributes itself, which is the
    ordinary case and costs one test.
    """
    written = h._get(a[0], "apy_mro_entries")
    if isinstance(written, Class):
        return a[0]
    if not isinstance(written, Instance)             or written.cls.find("__mro_entries__") is None:
        return h._fail("TypeError",
                       f"bases must be types, not '{h.kind_name(written)}'")
    got = _user(h, lambda: h._invoke(written.cls.find("__mro_entries__"),
                                     [written, h._get(a[1],
                                                      "apy_mro_entries")]),
                fail=_FAILED)
    if got is _FAILED:
        return 0
    # THE FIRST ENTRY -- see the C for why one and not the tuple.
    if isinstance(got, (list, tuple)) and got:
        return h._value(got[0])
    return _apy_object_class(h, [])


def _apy_meta_for(h, a):
    made = _meta_for(h._get(a[0], "apy_meta_for"),
                     h._get(a[1], "apy_meta_for"))
    return h._new(made) if made is not None else h._new(None)


def _class_build(h, a, kw):
    """Build the class a `class` statement describes.

    THROUGH THE METACLASS when there is one, which is what lets `ABCMeta`
    refuse an instantiation. Without one this is the plain construction, and
    the two meet here so the lowering does not have to tell them apart.
    """
    meta = h._get(a[0], "apy_class_build")
    name = h._get(a[1], "apy_class_build")
    bases = h._get(a[2], "apy_class_build")
    ns = h._get(a[3], "apy_class_build")
    if not _bases_ok(h, bases):
        return 0
    use = _meta_for(meta, bases)
    if use is not None:
        # THE CLASS KEYWORDS GO TO THE METACLASS -- `class C(metaclass=M,
        # kind="x")` is `M(name, bases, ns, kind="x")`. Only when there IS
        # one: without a metaclass they are for `__init_subclass__`, which the
        # caller announces separately, and handing them to the plain
        # construction would make them an arity error.
        if kw:
            # THROUGH THE NAME-MATCHING PATH. `_invoke` only fills a `**kw`
            # parameter, so a keyword naming a declared one was dropped and
            # the parameter took its default.
            built = _call_kwargs(h, use, [name, bases, ns], dict(kw))
            _BUILTIN_PENDING[0] = None
            _BUILTIN_PENDING[1] = None
            return built
        # `h._value`, NOT `h._new`. `is` compares HANDLES, and the class the
        # metaclass answered already has one -- it was built inside
        # `Meta.__new__`, where the body could keep it. Minting a second made
        # `C is Meta.seen` False for a class the metaclass had just handed
        # back, and left every member an enum built against the class its
        # metaclass saw rather than against the one the statement bound, so
        # `type(Colour.RED) is Colour` was False too.
        #
        # The same rule `_apy_exc_type` states for `OSError is OSError`. The
        # plain path below keeps `_new`, because there the class is made HERE
        # and this is the first handle it has ever had.
        built = h._value(h._invoke(use, [name, bases, ns]))
        # NOTHING PENDING OUTLIVES ONE STATEMENT. A metaclass that answers
        # something it did not build with `type()` leaves the cell unread,
        # and the next class of the same name would take it.
        _BUILTIN_PENDING[0] = None
        _BUILTIN_PENDING[1] = None
        return built
    made = _type_from_ns(h, None, name, bases, ns)
    _BUILTIN_PENDING[0] = None
    _BUILTIN_PENDING[1] = None
    return h._new(made) if made is not None else 0


def _bases_ok(h, bases) -> bool:
    """Can these bases be linearised at all? Reported HERE rather than left
    to `_type_from_ns`, because the metaclass path never reaches it."""
    listed = [b for b in bases if isinstance(b, Class)]         if isinstance(bases, (list, tuple)) else []
    if len(listed) < 2:
        return True
    if c3(Class("<probe>"), listed) is None:
        h._fail("TypeError",
                "Cannot create a consistent method resolution order (MRO) "
                "for bases")
        return False
    return True


def _apy_class_build(h, a):
    return _class_build(h, a, None)


def _apy_class_build_kw(h, a):
    return _class_build(h, a, h._get(a[4], "apy_class_build_kw"))


def _apy_object_class(h, a):
    """`object` AS A CLASS OBJECT -- what `C.__base__` answers for a class
    with no written base. Its dict carries the same defaults `super()` falls
    back to.

    NOT INSTALLED AS AN ACTUAL BASE on every class: it is the honest ANSWER to
    a question about the hierarchy, and making it a real link would put
    `__eq__` and friends into every lookup.
    """
    made = h._defaults.get("<object>")
    if made is not None:
        return made
    cls = Class("object")
    for nm in ("__init__", "__new__", "__repr__", "__str__", "__eq__",
               "__ne__", "__hash__"):
        cls.dict[nm] = _object_default(h, nm)
    made = h._new(cls)
    h._defaults["<object>"] = made
    return made


def _apy_type_make(h, a):
    """`type(name, bases, ns)` -- the three-argument form, which is the
    `class` statement written out. No metaclass recorded: one made this way
    IS a plain `type`."""
    return h._new(_type_from_ns(h, None,
                                h._get(a[0], "apy_type_make"),
                                h._get(a[1], "apy_type_make"),
                                h._get(a[2], "apy_type_make")))


def _apy_prepare(h, a):
    """PEP 3115: the mapping a class body is executed into."""
    meta = h._get(a[0], "apy_prepare")
    name = h._get(a[1], "apy_prepare")
    bases = h._get(a[2], "apy_prepare")
    if not isinstance(meta, Class):
        return h._new({})
    hook = meta.lookup("__prepare__")
    if hook is _ABSENT:
        return h._new({})
    if isinstance(hook, Descr) and hook.get is not None:
        # THE METACLASS IS ITS FIRST ARGUMENT, which is what `@classmethod`
        # on it means.
        got = h._invoke(hook.get, [meta, name, bases])
    else:
        got = h._invoke(hook, [name, bases])
    if not isinstance(got, dict):
        return h._fail("TypeError", "__prepare__() must return a mapping")
    return h._new(got)


#: The class `None` is an instance of, as a value. Made once and kept,
#: because `get_args(int | None)[1] is get_args(str | None)[1]` holds in
#: CPython -- a fresh class per union would answer False.
_NONE_TYPE = None


def _none_type():
    global _NONE_TYPE
    if _NONE_TYPE is None:
        _NONE_TYPE = Class("NoneType")
    return _NONE_TYPE


def _apy_typevar(h, a):
    """PEP 695's type PARAMETER, and PEP 484's `TypeVar` under one object:
    both are a name and nothing else at run time."""
    global _TYPEVAR_CLASS
    if _TYPEVAR_CLASS is None:
        _TYPEVAR_CLASS = Class("TypeVar")
    if "has_default" not in _TYPEVAR_CLASS.dict:
        # PEP 696: `has_default()` is a METHOD and not an attribute, so the
        # class needs one. Native, because the whole of it is "is the default
        # slot filled".
        _TYPEVAR_CLASS.dict["has_default"] = Native(
            "has_default",
            lambda self: self.dict.get("__default__") is not None)
    made = Instance(_TYPEVAR_CLASS, h)
    made.dict["__name__"] = str(h._get(a[0], "apy_typevar"))
    made.dict["__default__"] = None
    return h._new(made)


#: The modules this build resolves at compile time. A name here cannot be
#: imported DYNAMICALLY either -- there is no import machinery in a produced
#: binary -- but the error says which of the two reasons it is.
_KNOWN_MODULES = frozenset({
    "__future__", "_pyast", "_pycompile", "_pylex", "_pyparse", "_pyrun",
    "_pyvalidate", "abc", "annotationlib", "asyncio", "bisect",
    "collections", "collections.abc", "contextlib", "contextvars", "copy",
    "dataclasses", "datetime", "decimal", "enum", "fractions", "functools",
    "gc", "heapq", "inspect", "io", "itertools", "json", "keyword", "math",
    "numbers", "operator", "os", "pathlib", "random", "re", "select",
    "socket", "statistics", "string", "struct", "subprocess", "sys",
    "textwrap", "threading", "time", "tomllib", "traceback", "types",
    "typing", "unicodedata", "warnings", "weakref",
})


def _apy_import(h, a):
    """`__import__(name)` -- a DYNAMIC import, which this compiler cannot do.

    Both answers are honest and neither is silently wrong: a module this build
    does not have is a ModuleNotFoundError exactly as in CPython, and one it
    does have is an ImportError saying the import cannot be done dynamically.
    """
    want = h._get(a[0], "apy_import")
    want = want if isinstance(want, str) else ""
    if want in _KNOWN_MODULES:
        return h._fail("ImportError",
                       f"cannot import {want!r} dynamically: this build "
                       f"resolves imports at compile time")
    return h._fail("ModuleNotFoundError", f"No module named {want!r}")


def _apy_typevar_default(h, a):
    """PEP 696: the DEFAULT a type parameter was written with."""
    tv = h._get(a[0], "apy_typevar_default")
    tv.dict["__default__"] = h._get(a[1], "apy_typevar_default")
    return a[0]


def _apy_type_alias(h, a):
    """PEP 695: `type Alias = list[int]`.

    A NAME plus the thing it stands for, and neither means anything to the
    runtime beyond being readable back. Its class is interned so
    `type(Alias).__name__` answers `TypeAliasType`, which is what a program
    asks to tell an alias from the type it aliases.
    """
    global _ALIAS_CLASS
    if _ALIAS_CLASS is None:
        _ALIAS_CLASS = Class("TypeAliasType")
    made = Instance(_ALIAS_CLASS, h)
    made.dict["__name__"] = h._get(a[0], "apy_type_alias")
    made.dict["__value__"] = h._get(a[1], "apy_type_alias")
    made.dict["__type_params__"] = tuple(h._get(a[2], "apy_type_alias") or ())
    return h._new(made)


#: Interned, so two mentions of an alias's class are one object.
_TYPEVAR_CLASS = None
_ALIAS_CLASS = None


def _apy_typing_form(h, a):
    """A `typing` special form -- `Final`, `LiteralString`, `Self`.

    A program may name one, annotate with it, and print its class; nothing
    else about it is observable, so one object carrying its own name covers
    the lot. The class is `_SpecialForm` because that is what CPython reports.

    INTERNED BY NAME. `get_origin(Literal["a"]) is Literal` is what a
    program tests, and it is True only if the two mentions are one object.
    """
    global _SPECIAL_FORM_CLASS
    name = str(h._get(a[0], "apy_typing_form"))
    if _SPECIAL_FORM_CLASS is None:
        _SPECIAL_FORM_CLASS = Class("_SpecialForm")
    made = h._forms.get(name)
    if made is not None:
        return made
    inst = Instance(_SPECIAL_FORM_CLASS, h)
    inst.dict["_name"] = name
    made = h._new(inst)
    h._forms[name] = made
    return made


#: What a parameterised type answers `get_origin`/`get_args` from when it is
#: NOT the native alias kind -- see `_apy_get_origin`.
_ALIAS_ATTR = ("__origin__", "__args__")


def _generic_attr(h, a, which: str, empty):
    """The `__origin__`/`__args__` a PYTHON OBJECT carries.

    `typing.Generic`'s own `__class_getitem__` is written in Python -- see
    `bundled/typing.py` -- so `Box[int]` is an ordinary instance and not the
    alias kind `list[int]` builds. Both spellings answer the same two
    attributes, so reading the attribute covers the user generic without
    changing a single answer for the builtin one, which is what makes this
    worth doing here rather than rewriting the two functions in the bundled
    module (where every program that only wants `get_origin` would pay for
    the whole of `typing`).

    A MISS IS NOT AN ERROR: `get_origin(3)` is None in CPython, not a
    TypeError, and every library that probes an annotation relies on it.
    """
    return _apy_getattr_default(h, [a[0], h._new(which), empty])


def _apy_get_origin(h, a):
    """`get_origin(x)` -- what was subscripted, or None."""
    v = h._get(a[0], "apy_get_origin")
    if isinstance(v, Alias):
        return h._value(v.origin)
    if isinstance(v, Instance):
        return _generic_attr(h, a, _ALIAS_ATTR[0], h._none)
    return h._none


def _apy_get_args(h, a):
    """`get_args(x)` -- what it was subscripted WITH, or the empty tuple."""
    v = h._get(a[0], "apy_get_args")
    if isinstance(v, Alias):
        return h._new(tuple(v.args))
    if isinstance(v, Instance):
        return _generic_attr(h, a, _ALIAS_ATTR[1], h._new(()))
    return h._new(())


def _apy_typing_final(h, a):
    """`@final` on a class. RETURNS ITS ARGUMENT -- a decorator that returned
    anything else would replace the thing it marks, and the marking is the
    whole of what it does."""
    obj = h._get(a[0], "apy_typing_final")
    if isinstance(obj, (Class, Instance)):
        obj.dict["__final__"] = True
    elif isinstance(obj, Func):
        if obj.dict is None:
            obj.dict = {}
        obj.dict["__final__"] = True
    return a[0]


def _apy_typing_override(h, a):
    """`@override` on a method. Marks and hands the function straight back."""
    obj = h._get(a[0], "apy_typing_override")
    if isinstance(obj, Func):
        if obj.dict is None:
            obj.dict = {}
        obj.dict["__override__"] = True
    elif isinstance(obj, (Class, Instance)):
        obj.dict["__override__"] = True
    return a[0]


def _apy_typing_mark(h, a):
    """`@runtime_checkable`, `@no_type_check`. A decorator that marks a thing
    for a CHECKER and does nothing a running program can see, so the honest
    implementation hands the argument back untouched."""
    return a[0]


def _apy_str_maketrans(h, a):
    """`str.maketrans(a, b)` and `str.maketrans(a, b, drop)`.

    The result is an ORDINARY DICT keyed by code point -- the documented
    shape, not an internal one, so a program may build the same table by hand
    and hand it to `translate`. None as the third argument is the
    two-argument form; the frontend always passes three.
    """
    # A str SUBCLASS IS A str HERE TOO -- see `_held_text`.
    first = _held_text(h._get(a[0], "apy_str_maketrans"))
    second = _held_text(h._get(a[1], "apy_str_maketrans"))
    drop = _held_text(h._get(a[2], "apy_str_maketrans"))
    if not isinstance(first, str) or not isinstance(second, str):
        return h._fail("TypeError", "maketrans() arguments must be strings")
    if len(first) != len(second):
        return h._fail("ValueError", "the first two maketrans arguments must "
                                     "have equal length")
    out = {}
    for src, dst in zip(first, second):
        out[ord(src)] = ord(dst)
    if isinstance(drop, str):
        # The third argument names characters to DELETE, recorded as a None
        # value -- what a hand-written table uses to mean "drop this one".
        for ch in drop:
            out[ord(ch)] = None
    return h._new(out)


def _apy_str_translate(h, a):
    """`s.translate(table)`. A character with no entry is KEPT -- translate
    maps what it knows and passes the rest through, which is what makes a
    table holding one key a useful thing to write."""
    s = h._get(a[0], "apy_str_translate")
    table = h._get(a[1], "apy_str_translate")
    # A BYTES RECEIVER IS A DIFFERENT METHOD -- bytes through a 256-byte
    # table, not code points through a dict.
    #
    # AN EMPTY DELETE SET, NOT None. `b.translate(table)` deletes nothing and
    # `b.translate(table, None)` is a TypeError -- the one-argument form
    # reaches HERE and the two-argument one reaches `_apy_bytes_translate`
    # directly, which is the only thing that tells them apart. `b""` is the
    # default CPython's own signature declares.
    if isinstance(s, (bytes, bytearray)):
        return _apy_bytes_translate(h, [a[0], a[1], h._new(b"")])
    if not isinstance(s, str):
        return h._fail("AttributeError", f"'{h.kind_name(s)}' object has no "
                                         f"attribute 'translate'")
    if not isinstance(table, dict):
        return h._fail("TypeError",
                       f"'{h.kind_name(table)}' object is not subscriptable")
    out = []
    for ch in s:
        if ord(ch) not in table:
            out.append(ch)
            continue
        to = table[ord(ch)]
        if to is None:
            continue
        if isinstance(to, str):
            out.append(to)
        elif _is_int_like(to):
            out.append(chr(int(to)))
        else:
            return h._fail("TypeError",
                           "character mapping must be in range(0x110000)")
    return h._new("".join(out))


#: What counts as bytes for a translation table or a delete set. A MEMORYVIEW
#: IS BYTES-LIKE and every other bytes method here already takes one --
#: `b"abcd".find(memoryview(b"bc"))` answers 1 -- so these take one too.
_BYTES_LIKE = (bytes, bytearray, memoryview)


def _bytes_like_bad(h, v):
    """CPython's refusal for a table, a delete set or a maketrans argument
    that is not bytes."""
    return h._fail("TypeError",
                   f"a bytes-like object is required, not '{h.kind_name(v)}'")


def _apy_bytes_maketrans(h, a):
    """`bytes.maketrans(frm, to)` -- the 256-byte table `translate` wants.

    A FULL MAPPING AND NOT A SPARSE ONE: every byte the caller did not name
    maps to itself, which is why the table must be exactly 256 long and why
    `translate` can refuse any other length.
    """
    frm = h._get(a[0], "apy_bytes_maketrans")
    to = h._get(a[1], "apy_bytes_maketrans")
    if not isinstance(frm, _BYTES_LIKE):
        return _bytes_like_bad(h, frm)
    if not isinstance(to, _BYTES_LIKE):
        return _bytes_like_bad(h, to)
    if len(frm) != len(to):
        return h._fail("ValueError",
                       "maketrans arguments must have same length")
    return h._new(bytes.maketrans(bytes(frm), bytes(to)))


def _apy_bytes_translate(h, a):
    """`b.translate(table)` and `b.translate(table, delete)`.

    A NONE TABLE IS THE IDENTITY and not an error: `b.translate(None, b"a")`
    is the ordinary way to spell "delete these bytes and change nothing
    else". The delete set is consulted BEFORE the table, which is CPython's
    order and is observable -- a byte that is both mapped and deleted goes.
    """
    s = h._get(a[0], "apy_bytes_translate")
    table = h._get(a[1], "apy_bytes_translate")
    drop = h._get(a[2], "apy_bytes_translate")
    if not isinstance(s, (str, bytes, bytearray)):
        return h._fail("AttributeError", f"'{h.kind_name(s)}' object has no "
                                         f"attribute 'translate'")
    if isinstance(s, str):
        return h._fail("TypeError",
                       "str.translate() takes exactly one argument (2 given)")
    if table is not None and not isinstance(table, _BYTES_LIKE):
        return _bytes_like_bad(h, table)
    if table is not None and len(table) != 256:
        return h._fail("ValueError",
                       "translation table must be 256 characters long")
    # NO None HERE. A delete set that was WRITTEN must be bytes-like --
    # `b"abc".translate(None, None)` is `a bytes-like object is required, not
    # 'NoneType'` in CPython -- and the one-argument form passes an empty one
    # rather than a None.
    if not isinstance(drop, _BYTES_LIKE):
        return _bytes_like_bad(h, drop)
    out = bytes(s).translate(None if table is None else bytes(table),
                             bytes(drop))
    return h._new(bytearray(out) if isinstance(s, bytearray) else out)


def _apy_translate_kw(h, a):
    """`x.translate(table, delete=...)` -- the form written with the keyword.

    THE ONLY DIFFERENCE IS THE MESSAGE. `str.translate` takes no keyword at
    all, and CPython says so rather than complaining about the count; the
    keyword is folded into its slot before any runtime symbol sees the call,
    so the frontend picks this name and the wording survives.
    """
    if isinstance(h._get(a[0], "apy_translate_kw"), str):
        return h._fail("TypeError",
                       "str.translate() takes no keyword arguments")
    return _apy_bytes_translate(h, a)


# ── math ────────────────────────────────────────────────────────────────────
# `import math`. Python's own module IS the specification, so this file uses
# it and the C reimplements it; the two are checked against each other by the
# integration corpus, which is the only way a hand-written `isqrt` stays
# honest.
#
# The INTEGER-PRESERVING ones are the point: `math.floor(-2.5)` is the int -3,
# not the float -3.0, and a float there would print and compare differently.

def _math_real(h, v, fn):
    # A BOOL IS A REAL NUMBER, because a bool is an int. `math.floor(True)` is
    # `1` in CPython and `math.sqrt(True)` is `1.0`; every math function takes
    # one. Rejecting them here made the interpreter refuse the whole module
    # for an argument both compiled paths accepted -- a three-way split on
    # `math.floor(True)`, which answered `1`, `True` and a TypeError.
    if not isinstance(v, (int, float)):
        h._fail("TypeError", f"must be real number, not {h.kind_name(v)}")
        return None
    return v


def _math1(name, fn, want_int=False, dunder=None):
    def run(h, a):
        v = h._get(a[0], name)
        if dunder is not None and isinstance(v, Instance):
            # PEP 3141'S HOOK, and the reason `math.floor` has one at all:
            # `Fraction(7, 2)` and `Decimal("3.5")` are real numbers that
            # are not `float`, and the only way `math.floor` can round one
            # is to ask it. CPython's `floor`/`ceil`/`trunc` each look for
            # their own dunder BEFORE trying to make a float, so a class
            # defining it never reaches the float conversion -- which is
            # what made `math.floor(Fraction(7, 2))` a TypeError here
            # while CPython answered 3. `bundled/fractions.py` found it.
            #
            # NOT `__index__`, and not a fallback to it: a class with
            # `__index__` and no `__floor__` is an integer-like object
            # whose float conversion is exact, so the ordinary path below
            # is right for it.
            if v.cls.find(dunder) is not None:
                try:
                    return h._value(v._send(dunder))
                except _UserFailed:
                    return 0
        if want_int:
            # A BOOL IS AN INT HERE TOO -- `math.isqrt(True)` is 1. See
            # `_math_real`.
            if not isinstance(v, int):
                if not isinstance(v, float):
                    return h._fail("TypeError",
                                   f"'{h.kind_name(v)}' object cannot be "
                                   f"interpreted as an integer")
        elif _math_real(h, v, name) is None:
            return 0
        try:
            return h._value(fn(v))
        except (ValueError, OverflowError) as exc:
            return h._fail_like(exc)
    return run


def _math2(name, fn, want_int=False):
    def run(h, a):
        x = h._get(a[0], name)
        y = h._get(a[1], name)
        if want_int:
            # A BOOL IS AN INT -- `math.gcd(True, 6)` is 1. See `_math_real`.
            for v in (x, y):
                if not isinstance(v, int):
                    return h._fail("TypeError",
                                   "'float' object cannot be interpreted as "
                                   "an integer")
        else:
            if _math_real(h, x, name) is None or _math_real(h, y, name) is None:
                return 0
        try:
            return h._value(fn(x, y))
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            return h._fail_like(exc)
    return run


def _apy_math_log(h, a):
    """`log(x)` and `log(x, base)` -- see the C's own comment on why the
    base defaults to a sentinel rather than to `e`."""
    x = _math_real(h, h._get(a[0], "apy_math_log"), "apy_math_log")
    if x is None:
        return 0
    if x <= 0:
        return h._fail("ValueError", "math domain error")
    base = h._get(a[1], "apy_math_log") if len(a) > 1 and a[1] else None
    if base is None:
        return h._value(math.log(x))
    base = _math_real(h, base, "apy_math_log")
    if base is None:
        return 0
    if base <= 0:
        return h._fail("ValueError", "math domain error")
    try:
        return h._value(math.log(x, base))
    except (ValueError, ZeroDivisionError) as exc:
        return h._fail_like(exc)


def _apy_math_frexp(h, a):
    """`frexp(x)` -- `(mantissa, exponent)` with `x == m * 2 ** e`. A PAIR,
    which is why it is not in the one-argument family above."""
    v = _math_real(h, h._get(a[0], "apy_math_frexp"), "apy_math_frexp")
    if v is None:
        return 0
    return h._new(math.frexp(v))


def _apy_math_modf(h, a):
    """`modf(x)` -- `(fractional, integral)`, both floats. FRACTIONAL
    FIRST, which is CPython's order and the opposite of what the name
    suggests to most readers."""
    v = _math_real(h, h._get(a[0], "apy_math_modf"), "apy_math_modf")
    if v is None:
        return 0
    return h._new(math.modf(v))


def _seq_of_reals(h, v, name):
    """A sequence of floats from whatever a caller passed -- a list, a
    tuple, a range or a generator, drained the way `apy_iterable` drains
    one on the compiled side so the two agree about what is iterable."""
    if isinstance(v, (Gen, Iterator)) or not isinstance(
            v, (list, tuple, set, frozenset, range)):
        v = h._get(_apy_iterable(h, [h._value(v)]), name)
    out = []
    for one in v:
        got = _math_real(h, one, name)
        if got is None:
            return None
        out.append(got)
    return out


def _apy_math_fsum(h, a):
    """`fsum(iterable)` -- an EXACT sum, which is the whole difference from
    `sum`: `sum([0.1] * 10)` is 0.9999999999999999 and this is 1.0."""
    items = _seq_of_reals(h, h._get(a[0], "apy_math_fsum"), "apy_math_fsum")
    if items is None:
        return 0
    try:
        return h._value(math.fsum(items))
    except (ValueError, OverflowError) as exc:
        return h._fail_like(exc)


def _apy_math_prod(h, a):
    """`prod(iterable, start=1)`.

    THROUGH THE ORDINARY MULTIPLY, so an integer product stays exact and
    promotes to a big rather than losing low bits in a double --
    `prod(range(1, 21))` is 2432902008176640000. The `start` decides the
    type of an empty product, as in CPython.
    """
    v = h._get(a[0], "apy_math_prod")
    acc = h._get(a[1], "apy_math_prod") if len(a) > 1 and a[1] else 1
    if isinstance(v, (Gen, Iterator)) or not isinstance(
            v, (list, tuple, set, frozenset, range)):
        v = h._get(_apy_iterable(h, [h._value(v)]), "apy_math_prod")
    for one in v:
        if not isinstance(one, (int, float)):
            return h._fail("TypeError",
                           f"unsupported operand type(s) for *: "
                           f"'{h.kind_name(acc)}' and '{h.kind_name(one)}'")
        acc = acc * one
    return h._value(acc)


def _apy_math_fma(h, a):
    """`fma(x, y, z)` -- `x * y + z` with ONE rounding."""
    got = []
    for one in a[:3]:
        v = _math_real(h, h._get(one, "apy_math_fma"), "apy_math_fma")
        if v is None:
            return 0
        got.append(v)
    try:
        return h._value(math.fma(got[0], got[1], got[2]))
    except (ValueError, OverflowError) as exc:
        return h._fail_like(exc)


def _apy_math_sumprod(h, a):
    """`sumprod(p, q)` -- the dot product, INTEGER-PRESERVING like `prod`."""
    p = h._get(a[0], "apy_math_sumprod")
    q = h._get(a[1], "apy_math_sumprod")
    pairs = []
    for v in (p, q):
        if isinstance(v, (Gen, Iterator)) or not isinstance(
                v, (list, tuple, set, frozenset, range)):
            v = h._get(_apy_iterable(h, [h._value(v)]), "apy_math_sumprod")
        pairs.append(list(v))
    if len(pairs[0]) != len(pairs[1]):
        return h._fail("ValueError", "Inputs are not the same length")
    acc = 0
    for x, y in zip(pairs[0], pairs[1]):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return h._fail("TypeError",
                           f"unsupported operand type(s) for *: "
                           f"'{h.kind_name(x)}' and '{h.kind_name(y)}'")
        acc = acc + x * y
    return h._value(acc)


def _apy_math_dist(h, a):
    """`dist(p, q)` -- the Euclidean distance between two points."""
    p = _seq_of_reals(h, h._get(a[0], "apy_math_dist"), "apy_math_dist")
    if p is None:
        return 0
    q = _seq_of_reals(h, h._get(a[1], "apy_math_dist"), "apy_math_dist")
    if q is None:
        return 0
    if len(p) != len(q):
        return h._fail("ValueError",
                       "both points must have the same number of dimensions")
    return h._value(math.dist(p, q))


def _math_choose(name, fn):
    """`math.comb` and `math.perm`, which are `_math2` plus one refusal.

    A BIG ARGUMENT IS REFUSED, and the refusal is what this wrapper exists
    for rather than `_math2`. Both functions loop `k` times in the
    compiled arrangements (`objects/c/_math.py`, `runtime/mathints.py`),
    so a `k` that does not fit a machine word is a loop that never
    finishes -- and a big `n` needs the whole falling factorial in big
    arithmetic, which those two do not have. CPython answers both,
    eventually. Refusing HERE TOO is what keeps the three arrangements
    saying the same thing, which is the property this runtime is measured
    on; answering where the others cannot would be a divergence nothing
    tests for.
    """
    def run(h, a):
        x = h._get(a[0], name)
        y = h._get(a[1], name)
        for v in (x, y):
            if not isinstance(v, int):
                return h._fail("TypeError",
                               "'float' object cannot be interpreted as "
                               "an integer")
        if not -(1 << 63) < int(x) < (1 << 63) \
                or not -(1 << 63) < int(y) < (1 << 63):
            return h._fail("OverflowError",
                           "comb()/perm() arguments must fit in a machine "
                           "word here")
        try:
            return h._value(fn(int(x), int(y)))
        except (ValueError, OverflowError) as exc:
            return h._fail_like(exc)
    return run


def _apy_math_isclose(h, a):
    """`isclose(a, b, rel_tol=1e-09, abs_tol=0.0)`. The relative tolerance is
    taken against the LARGER magnitude, which is what makes the relation
    symmetric -- a version dividing by one side is not."""
    vals = []
    for i in range(4):
        v = h._get(a[i], "apy_math_isclose")
        if _math_real(h, v, "isclose") is None:
            return 0
        vals.append(v)
    try:
        return h._bool(math.isclose(vals[0], vals[1], rel_tol=vals[2],
                                    abs_tol=vals[3]))
    except ValueError as exc:
        return h._fail_like(exc)


def _apy_is_integer(h, a):
    v = h._get(a[0], "apy_is_integer")
    if isinstance(v, bool) or isinstance(v, int):
        return h._bool(True)
    if not isinstance(v, float):
        return h._fail("AttributeError", f"'{h.kind_name(v)}' object has no "
                                         f"attribute 'is_integer'")
    return h._bool(v.is_integer())


def _apy_conjugate(h, a):
    v = h._get(a[0], "apy_conjugate")
    if not isinstance(v, (int, float, complex)):
        return h._fail("AttributeError", f"'{h.kind_name(v)}' object has no "
                                         f"attribute 'conjugate'")
    return h._value(v.conjugate())


def _apy_update(h, a):
    """`d.update(other)`, and the `**d` of a call.

    NOT ONLY A MAPPING: `d.update([(1, 2)])` is legal, and so is any iterable
    of two-element iterables -- which is why a list of two-character strings
    works and a list of characters does not.
    """
    target = h._get(a[0], "apy_update")
    src = h._get(a[1], "apy_update")
    if not isinstance(target, dict):
        # A SET UPDATES TOO, and `d.update` and `s.update` are one runtime
        # function -- so refusing everything that is not a dict here made
        # `s.update(x)` report a missing attribute for a method sets have.
        # A frozenset genuinely has none, and is refused BY NAME the way
        # every other set mutator refuses one.
        if isinstance(target, set):
            items = _seq_items(h, src, "apy_update")
            if items is None:
                return 0
            for item in items:
                bad = _unhashable_name(item)
                if bad:
                    return h._fail("TypeError",
                                   f"cannot use '{bad}' as a set element "
                                   f"(unhashable type: '{bad}')")
                try:
                    target.add(item)
                except TypeError as exc:
                    return h._fail_like(exc)
            return h._none
        return h._fail("AttributeError",
                       f"'{h.kind_name(target)}' object has no attribute "
                       f"'update'")
    if isinstance(src, Instance) and isinstance(src.held, dict):
        # A dict SUBCLASS UPDATES FROM ITS MAPPING, not from its keys.
        # Iterating a dict yields keys, so the pair walk below read
        # `d.update(Counter())` as a sequence of single keys and reported an
        # element of length 1 -- about a perfectly good mapping.
        src = src.held
    if isinstance(src, dict):
        for k, v in src.items():
            if _dict_set(h, target, k, v) == 0:
                return 0
        return h._none
    items = _seq_items(h, src, "apy_update")
    if items is None:
        return 0
    for at, pair in enumerate(items):
        # TWO DIFFERENT ERRORS AND CPYTHON DISTINGUISHES THEM: an element
        # that is not a sequence AT ALL is a TypeError naming nothing, and
        # one that IS a sequence of the wrong length is a ValueError naming
        # its position and its length. Reporting the second for both meant
        # `d.update([5])` was told about a length an int does not have, and
        # `except TypeError:` did not catch it. The compiled twin draws the
        # line between the same kinds.
        if not isinstance(pair, (list, tuple, str, set, frozenset, dict)):
            return h._fail("TypeError", "object is not iterable")
        got = list(pair)
        if len(got) != 2:
            # THE INDEX IS PART OF THE MESSAGE, which CPython and the compiled
            # runtime both say and this did not -- a program with a long
            # sequence was told a length and not which element had it.
            return h._fail("ValueError",
                           f"dictionary update sequence element #{at} has "
                           f"length {len(got)}; 2 is required")
        if _dict_set(h, target, got[0], got[1]) == 0:
            return 0
    return h._none


def _apy_dict_set(h, a):
    d = h._get(a[0], "apy_dict_set")
    key = h._get(a[1], "apy_dict_set")
    val = h._get(a[2], "apy_dict_set")
    if not isinstance(d, dict):
        return h._fail(
            "TypeError",
            f"'{h.kind_name(d)}' object does not support item assignment")
    return _dict_set(h, d, key, val)


def _dict_get(h, d, key):
    try:
        return h._value(d[key])
    except KeyError:
        # KeyError's message is the repr of the key, which is what the C
        # builds too -- and what CPython prints for an uncaught one.
        return h._fail("KeyError", h._text(key, True))
    except TypeError as e:
        return h._fail_like(e)


def _apy_key_at(h, a):
    v = h._get(a[0], "apy_key_at")
    i = int(a[1])
    if isinstance(v, Gen):
        got = _gen_cache(h, v)          # what the length query drained
        if got is None:
            return 0
        return h._value(got[i]) if 0 <= i < len(got) else h._none
    if isinstance(v, Iterator):
        # IGNORES `i` AND ADVANCES, exactly as the C's `apy_key_at` does and
        # for the reason documented there: a cursor has a position of its own,
        # and a consumer walking from zero would replay it.
        if v.mode != Iterator.PLAIN or isinstance(v.src, (Gen, Iterator,
                                                          Instance)):
            got = _drain_cursor(h, v)
            if got is None:
                return 0
        items = _iter_items(v.src)
        if v.i >= len(items):
            return h._none
        item = items[v.i]
        v.i += 1
        return h._value(item)
    if isinstance(v, dict):
        # Insertion order, which Python guarantees since 3.7 and the C gets by
        # appending to an association list.
        return h._value(list(v)[i])
    if isinstance(v, (set, frozenset)):
        # A SET IS ITERABLE AND NOT SUBSCRIPTABLE, and this function means
        # "iterate": the C reaches its `v.q` items directly here, so falling
        # through to `apy_getitem` reported `list({1, 2})` as unsubscriptable
        # -- a TypeError on a program CPython runs.
        return h._value(list(v)[i])
    return _apy_getitem(h, [a[0], h._int(i)])


# ── exceptions ──────────────────────────────────────────────────────────────

def _exc_construct(h, exc, args):
    """Give a fresh exception its class and, where the class writes one, run
    its `__init__` over the arguments the `raise` supplied.

    AFTER the defaults are in place, not instead of them: CPython sets `args`
    in `BaseException.__new__` and only then calls `__init__`, so a class whose
    `__init__` never calls `super().__init__` still reads back what was passed.
    One that does call it overwrites them, which is the whole reason
    `AppError(404, "missing")` can report `('404: missing',)`.
    """
    cls = h.exc_class.get(exc.name)
    if cls is None:
        return h._new(exc)
    exc.cls = cls
    init = cls.find("__init__")
    if init is None:
        return h._new(exc)

    def run():
        h._invoke(init, [exc] + list(args))
        return h._new(exc)

    return _user(h, run)


def _rename_exception(h, was: str, now: str, cls) -> None:
    """Carry a class's EXCEPTION REGISTRATION across a rename.

    The hierarchy is a table of NAMES, and a class that is renamed after it was
    registered leaves the two disagreeing. That is not a hypothetical: a
    BUNDLED module's classes are spliced under mangled names and the splice
    then restores `__name__` -- precisely so the mangling stays invisible --
    so `copy.Error` registered as `_asmpy_bundled_4_copy_Error` and then started
    calling itself `Error`. `issubclass(copy.Error, Exception)` asks the table
    for `Error`, finds nothing, and answers False for a class whose `class`
    statement plainly names Exception as its base.

    BOTH SPELLINGS ARE KEPT rather than the old one moved. Generated code
    raises through the mangled name -- that is the name in the compiled
    program -- while everything a reader sees uses the restored one, so the
    table has to answer for both or one of the two stops working.
    """
    if was == now:
        return
    if was in h.user_exc:
        h.user_exc[now] = h.user_exc[was]
    # BOTH SPELLINGS, and the `not registered` case is the one that matters.
    # An exception class with an EMPTY BODY is never handed to
    # `apy_exc_class_bind` at all -- the lowering early-returns because there
    # is nothing to build -- so nothing was registered under the mangled name,
    # and every display fell back to the name the CELL carries. Right for a
    # user's class, wrong for a bundled one. Mirrors the C's `__name__` arm.
    held = h.exc_class.get(was)
    if held is cls or (held is None and was in h.user_exc):
        h.exc_class[now] = cls
        h.exc_class[was] = cls


def _apy_exc_class_bind(h, a):
    """A user exception class, findable by NAME.

    The hierarchy is a table of names -- that is what makes `except
    LookupError:` catch a KeyError without either being a value -- so a class
    with a body has to be findable from the name alone, at the moment an
    exception of that name is made. Nothing else knows: `apy_make_exc` is
    handed a string, and the `class` statement that wrote the body may be in
    another function entirely.
    """
    h.exc_class[str(h._get(a[0], "apy_exc_class_bind"))] = h._get(
        a[1], "apy_exc_class_bind")
    return h._none


def _apy_make_exc(h, a):
    name = h._get(a[0], "apy_make_exc")
    arg = h._get(a[1], "apy_make_exc")
    return _exc_construct(h, Exc(str(name), arg), [arg])


#: Which OSError subclass an errno names. PEP 3151: `OSError(2, ...)` IS a
#: FileNotFoundError, so a program can catch the precise failure without
#: inspecting `errno`. The numbers are the POSIX values CPython maps.
_ERRNO_CLASS = {
    1: "PermissionError", 2: "FileNotFoundError", 3: "ProcessLookupError",
    4: "InterruptedError", 10: "ChildProcessError", 11: "BlockingIOError",
    13: "PermissionError", 17: "FileExistsError", 20: "NotADirectoryError",
    21: "IsADirectoryError", 32: "BrokenPipeError",
    103: "ConnectionAbortedError", 104: "ConnectionResetError",
    110: "TimeoutError", 111: "ConnectionRefusedError",
    115: "BlockingIOError",
}


def _apy_make_excn(h, a):
    """`E(a, b, ...)` -- an exception built from MORE THAN ONE argument."""
    name = str(h._get(a[0], "apy_make_excn"))
    addr, count = int(a[1]), int(a[2])

    def cell(i):
        return h._get(h._interp.mem.read(addr + i * 8, _PTR), "apy_make_excn")

    args = [cell(i) for i in range(count)]
    if args and name == "OSError" and isinstance(args[0], int)             and not isinstance(args[0], bool):
        name = _ERRNO_CLASS.get(args[0], name)
    made = Exc(name, args[0] if args else None, bool(args))
    made.argv = tuple(args)
    return _exc_construct(h, made, args)


def _apy_raise(h, a):
    exc = h._get(a[0], "apy_raise")
    if not isinstance(exc, Exc):
        return h._fail(
            "TypeError",
            f"exceptions must derive from BaseException, "
            f"not '{h.kind_name(exc)}'")
    # The exception being HANDLED becomes this one's `__context__`, unless a
    # `raise ... from` already spoke for the chain. Set here rather than at the
    # `except` because only a raise creates a link.
    # Set even when `from` suppressed it: `__suppress_context__` is whether to
    # PRINT the context, not whether to have one.
    # WHERE IT WAS RAISED. An exception carries no traceback until this
    # runs, which is what makes `ValueError("x").__traceback__` None and the
    # same object's traceback real once it has been raised.
    exc.pos = h.pos_here
    if (exc.context is None
            and h.handling is not None and h.handling is not exc):
        exc.context = h.handling
    # A raise while an error is still PENDING -- `try: raise A finally: raise
    # B` -- chains too, and nothing was "being handled" there: the A is in
    # flight rather than caught.
    if exc.context is None and h.err is not None:
        pending = h.err_value
        if pending is None:
            name, msg = h.err
            pending = Exc(name, msg, has_arg=bool(msg))
            pending.rendered = True
            pending.pos = h.pos_err
        if pending is not exc:
            exc.context = pending
    return h._fail_raised(exc)


def _apy_user_exc_rows(h, a):
    """The table's address, which the host does not have.

    THE REGISTRATIONS ARE A DICT here -- `h.user_exc`, keyed by name -- so
    everything that would walk sixty-four rows walks that instead, and is
    itself bound in this file. It raises for the reason `_apy_err_slots`
    does: a zero would be indexed.
    """
    raise RuntimeError(
        "apy_user_exc_rows has no host equivalent: the interpreter keeps "
        "user exception classes in a dict on the host object, not a table")


def _apy_user_exc_slot(h, a):
    """Where the count lives, which for the same reason is nowhere."""
    raise RuntimeError(
        "apy_user_exc_slot has no host equivalent: the interpreter counts "
        "user exception classes with len(), not a stored word")


def _apy_exc_register(h, a):
    """A `class MyError(ValueError):` the program wrote.

    Registered into the same hierarchy the builtins use, so `except
    ValueError:` catches it through the code path that already makes `except
    LookupError:` catch a KeyError. See `apy_exc_register` in objects/csource.py
    for why this is a NAME and not a type object, and what that costs.
    """
    h.user_exc[str(h._get(a[0], "apy_exc_register"))] = str(
        h._get(a[1], "apy_exc_register"))
    return h._none


def _exc_chain(h, name: str):
    """`name` and every ancestor, builtin or user-defined."""
    seen = []
    while name is not None and name not in seen:
        seen.append(name)
        parent = h.user_exc.get(name)
        if parent is None:
            cls = getattr(builtins, name, None)
            if isinstance(cls, type) and issubclass(cls, BaseException):
                bases = cls.__mro__[1:]
                parent = bases[0].__name__ if bases else None
            else:
                parent = None
        name = parent
    return seen


def _apy_error_handling(h, a):
    """WHAT IS BEING HANDLED right now, for implicit chaining: a `raise`
    inside an `except` records it as the new exception's `__context__`."""
    exc = h._get(a[0], "apy_error_handling")
    was = h.handling
    h.handling = exc if isinstance(exc, Exc) else None
    return h._value(was)


def _apy_add_note(h, a):
    """`e.add_note(text)` -- PEP 678."""
    exc = h._get(a[0], "apy_add_note")
    text = h._get(a[1], "apy_add_note")
    if not isinstance(exc, Exc):
        return h._fail("AttributeError", f"'{h.kind_name(exc)}' object has no "
                                         f"attribute 'add_note'")
    if not isinstance(text, str):
        return h._fail("TypeError", "note must be a str")
    if exc.notes is None:
        exc.notes = []
    exc.notes.append(text)
    return h._none


def _apy_raise_from(h, a):
    """`raise X from Y`. The cause is EXPLICIT, and `from None` suppresses the
    implicit context rather than setting a cause."""
    exc = h._get(a[0], "apy_raise_from")
    cause = h._get(a[1], "apy_raise_from")
    if isinstance(exc, Exc):
        exc.suppress = True
        exc.cause = cause if int(a[2]) and isinstance(cause, Exc) else None
    return _apy_raise(h, [a[0]])


def _apy_exc_parent_of(h, a):
    """The name of an exception's base class, or 0 if it has none.

    ANSWERED FROM PYTHON'S OWN CLASS TREE, the way `_apy_error_matches` below
    answers its question -- one source for the hierarchy here, so the two
    cannot drift. The compiled runtime reads a packed table instead, and
    `tests/uasm/integration/test_exc_tree.py` is what keeps THAT in step
    with the C's.

    USER CLASSES FIRST HERE, and built-ins first in the compiled version --
    which is not a disagreement: `h.user_exc` cannot contain a built-in name,
    because the compiled table is consulted before a name reaches it.
    """
    name = str(h._get(a[0], "apy_exc_parent_of"))
    if name in h.user_exc:
        return h._new(h.user_exc[name])
    cls = getattr(__import__("builtins"), name, None)
    if isinstance(cls, type) and issubclass(cls, BaseException):
        bases = cls.__bases__
        if bases and bases[0] is not object:
            return h._new(bases[0].__name__)
    return 0


def _apy_error_matches(h, a):
    """Does the current error match a handler named by `a[0]`?

    The hierarchy comes from PYTHON'S OWN builtin exception classes rather
    than from a table -- `except LookupError` catching a KeyError is then a
    fact about the real class tree, not about a list this file keeps in step
    with the C's. The C's `APY_EXC_TREE` is a transcription of that same tree,
    so the two agree by construction on every name in it.

    A name that is not a builtin exception matches only itself, which is the
    C's rule for a name missing from its table and stays right until user
    classes exist and can declare a base.
    """
    want = str(h._get(a[0], "apy_error_matches"))
    if h.err is None:
        return 0
    have = h.err[0]
    if want in h.user_exc or have in h.user_exc:
        # A user-defined name is in neither `builtins` nor Python's class
        # tree, so the walk below cannot see it. The chain does.
        return 1 if want in _exc_chain(h, have) else 0
    want_cls = getattr(builtins, want, None)
    have_cls = getattr(builtins, have, None)
    if (isinstance(want_cls, type) and issubclass(want_cls, BaseException)
            and isinstance(have_cls, type)
            and issubclass(have_cls, BaseException)):
        return 1 if issubclass(have_cls, want_cls) else 0
    return 1 if have == want else 0


def _apy_check_bound(h, a):
    """A LOCAL READ BEFORE IT WAS ASSIGNED.

    Null is the runtime's "no value" and never a legitimate one, so a null
    here means the assignment has not run -- a different thing from having
    been assigned None, which is the distinction `UnboundLocalError` makes.
    """
    if a[0]:
        return a[0]
    return h._fail("UnboundLocalError",
                   f"cannot access local variable "
                   f"'{h._get(a[1], 'apy_check_bound')}' where it is not "
                   f"associated with a value")


def _apy_error_value(h, a):
    if h.err is None:
        return h._none
    # The object that was RAISED, when there was one. An operation that failed
    # never had one, and its message text is all there is to rebuild from.
    if h.err_value is not None:
        return h._new(h.err_value)
    name, msg = h.err
    built = Exc(name, msg, has_arg=bool(msg))
    # The message is ALREADY formatted -- `'k'` for a KeyError from a failed
    # lookup -- so `str(e)` must not repr it a second time.
    built.rendered = True
    # THE FAILING STATEMENT, not the one running now: a handler's own
    # statements have already moved the cursor by the time this is built.
    built.pos = h.pos_err
    return h._new(built)


def _apy_err_slots(h, a):
    """THE HOST HAS NO SUCH STORAGE, and that is not a gap.

    The compiled runtime keeps the pending error in two words and 256 bytes,
    and these two functions hand out their addresses so the C and the IR can
    share them. The host keeps the same state as Python objects on `h` --
    `h.err` and `h.err_value` -- because it is not compiling anything and has
    nowhere to put a buffer.

    So nothing reaches here: every function that would read those words is
    itself bound in this file and reads `h.err` instead. The binding exists
    because `objects_host.py` must answer for every exported symbol, and
    answering with a raise is more honest than answering with a zero that
    would be dereferenced.
    """
    raise RuntimeError(
        "apy_err_slots has no host equivalent: the interpreter keeps the "
        "pending error on the host object, not in runtime memory")


def _apy_err_text(h, a):
    """The message buffer's address. See `_apy_err_slots`."""
    raise RuntimeError(
        "apy_err_text has no host equivalent: the interpreter keeps the "
        "pending error's message on the host object, not in a buffer")


def _apy_raise_at(h, a):
    """Record a pending error, first writer winning.

    THE HOST HAS THE SAME TWO RULES and keeps them in Python: `h.err` is the
    pair, and a second failure while one is pending changes nothing. What it
    does not have is the 256-byte buffer, so nothing here truncates -- a
    difference only a message longer than 255 bytes could see, and the C's
    own messages are literals well short of it.
    """
    if h.err is not None:
        return 0
    h.pos_err = h.pos_here
    h.err = (str(h._get(a[0], "apy_raise_at")),
             str(h._get(a[1], "apy_raise_at")))
    h.err_value = None
    return 0


def _apy_raise_over(h, a):
    """Record a pending error, replacing whatever was there. A `raise`."""
    h.pos_err = h.pos_here
    h.err = (str(h._get(a[0], "apy_raise_over")),
             str(h._get(a[1], "apy_raise_over")))
    h.err_value = None
    return 0


def _apy_raise_fmt(h, a):
    """Record a pending error whose text is a template with two strings in it.

    `%s` AND NOTHING ELSE, which is not a simplification: all 153 call sites
    in the C pass 280 conversions between them and every one is `%s`. Python's
    own `%` operator would accept more, so the substitution is written out to
    keep the host and the compiled runtime agreeing about what a format may
    contain.
    """
    fmt = str(h._get(a[1], "apy_raise_fmt"))
    args = [str(h._get(a[2], "apy_raise_fmt")),
            str(h._get(a[3], "apy_raise_fmt"))]
    out, i, which = [], 0, 0
    while i < len(fmt):
        if fmt.startswith("%s", i):
            out.append(args[which] if which < 2 else "")
            which += 1
            i += 2
        else:
            out.append(fmt[i])
            i += 1
    return _apy_raise_at(h, (a[0], h._new("".join(out))))


def _apy_error_occurred(h, a):
    return 1 if h.err is not None else 0


def _apy_error_type(h, a):
    return h._none if h.err is None else h._new(h.err[0])


def _apy_error_message(h, a):
    return h._none if h.err is None else h._new(h.err[1])


def _apy_error_clear(h, a):
    h.err = None
    # The object goes with the flag. Leaving it would let the NEXT error --
    # one from a failing operation, which has no object -- report the previous
    # exception's payload.
    h.err_value = None
    return None


def _apy_fatal_if_error(h, a):
    """Stop the program the way the compiled one stops.

    The C writes `<Type>: <msg>` to STDERR and exits 1 -- deliberately not
    stdout, because the conformance suite diffs stdout and a traceback there
    would turn a correct failure into a wrong answer. Here that is a `Trap`
    carrying the same text: whatever was printed before it has already gone to
    the output stream, so the two paths agree on stdout, which is what is
    compared.
    """
    if h.err is None:
        return None
    kind, msg = h.err
    raise _Trap(f"{kind}: {msg}" if msg else kind)


# ── the numeric tower ───────────────────────────────────────────────────────

#: WHAT A MACHINE INDEX HOLDS. The compiled halves convert a subscript to an
#: `int64_t` and refuse anything wider; the interpreter has no such limit and
#: would happily index with a number no compiled program can hold.
_INDEX_LIMIT = 1 << 63


def _is_int_like(v) -> bool:
    return isinstance(v, int) and not isinstance(v, float)


def _is_num(v) -> bool:
    return v is not None and isinstance(v, (int, float))


def _result(h, v):
    """A computed result as a handle, truncating an integer to 64 bits.

    Python's integers are unbounded and the C's are not. Where they differ the
    C is the one a compiled program will produce, and the invariant this file
    exists to serve is that the interpreter and that program agree -- so the
    truncation is deliberate, and it is the same truncation `int64_t` performs.
    """
    return h._value(v)


#: Operator symbol -> (`__op__`, `__rop__`). Mirrors `APY_OP_DUNDERS` in the
#: C, and exists for the same reason: the reflected call has to be made
#: explicitly here, because every user object shares one Python class.
_OP_DUNDER = {
    "+": ("__add__", "__radd__"), "-": ("__sub__", "__rsub__"),
    "*": ("__mul__", "__rmul__"), "/": ("__truediv__", "__rtruediv__"),
    "//": ("__floordiv__", "__rfloordiv__"), "%": ("__mod__", "__rmod__"),
    "**": ("__pow__", "__rpow__"), "&": ("__and__", "__rand__"),
    "|": ("__or__", "__ror__"), "^": ("__xor__", "__rxor__"),
    "<<": ("__lshift__", "__rlshift__"), ">>": ("__rshift__", "__rrshift__"),
    "@": ("__matmul__", "__rmatmul__"),
}


def _as_builtin(v, dunders):
    """A builtin-extending instance ACTING AS its builtin, for one operator.

    `class T(tuple)` has no `__add__` in its body, so `t + (4,)` reported an
    unsupported operand pair between `'T'` and `'tuple'` -- about an object
    that is a tuple and an operation tuples support. The same miss made
    `D({'a': 1}) == {'a': 1}` answer False, which is worse than an error
    because nothing marks it.

    ONLY WHEN THE CLASS IS SILENT. A class that writes `__eq__` means its own,
    and substituting the held value would run the builtin's comparison
    instead -- so both the direct and the reflected name are checked before
    anything is swapped.
    """
    if isinstance(v, Instance) and v.held is not None \
            and all(v.cls.find(d) is None for d in dunders):
        return v.held
    return v


def _binop(name, op, sym):
    """One arithmetic binding: Python's operator, the C's dispatch.

    The kinds the C rejects are rejected here with the C's text, and anything
    it accepts is computed by `op` on the real objects -- so promotion
    (`True + 1` is the int 2, `1 + 1.0` the float 2.0) is Python's rather than
    a restatement of it.
    """
    def run(h, a):
        x = h._get(a[0], name)
        y = h._get(a[1], name)
        # A VIEW BEHAVES AS A SET under `&`, `|`, `-` and `^`, as the C does
        # it. Converted BEFORE the kind rules run: `_reject` sees a view as
        # neither a set nor a number and reports an unsupported pair, so the
        # conversion has to happen first or it never gets the chance.
        if sym in ("&", "|", "-", "^"):
            if isinstance(x, _VIEW_TYPES):
                x = set(x)
            if isinstance(y, _VIEW_TYPES):
                y = set(y)
        # PEP 604: `int | str` IS A TYPE, not an arithmetic operation. The
        # arms flatten, so `int | str | None` is one three-armed union.
        if sym == "|" and _is_type_like(x) and _is_type_like(y):
            arms = _union_arms(x) + _union_arms(y)
            return h._new(Alias(_union_form(h), tuple(_dedup_arms(arms))))
        # THE REFLECTED CALL IS MADE HERE, not left to Python's protocol.
        # Every user object is the same Python class (`Instance`), and Python
        # SKIPS the reflected method when both operands have the same type --
        # so `A() + B()` never reached `B.__radd__` and reported an
        # unsupported pair instead. The C dispatches explicitly; so does this.
        #
        # THREE STEPS, AND THE BUILTIN IS THE MIDDLE ONE -- the same order
        # `_cmpop` takes and for the same reason, which `_order_plan` gives at
        # length. This read the builtin FIRST, before either dunder and before
        # `_reject`, so `OnlyRadd([1]) + OnlyRadd([2])` answered `'RADD'`
        # where CPython concatenates: `type(a).__add__` is the one `list`
        # gives the subclass, and an inherited slot wins outright.
        dunder = _OP_DUNDER.get(sym.split(" ")[0])
        if dunder and (isinstance(x, Instance) or isinstance(y, Instance)):
            direct, reflected = dunder
            for step in _order_plan(x, y, reflected):
                if step is _HELD:
                    # BOTH SIDES MUST COME OUT AS BUILTINS, as in `_cmpop`:
                    # `list.__add__` answers NotImplemented for an operand it
                    # knows nothing about, which is when CPython goes on to
                    # the mirror AND hands it the operands the program wrote.
                    bx = _as_builtin(x, (direct,))
                    by = y.held if isinstance(y, Instance) \
                        and y.held is not None else y
                    if isinstance(bx, Instance):
                        continue
                    # `str % anything` IS NOT A KIND-DECLINE, and it is the
                    # one operator that is not. Every other builtin answers
                    # NotImplemented for an operand it does not know, which
                    # is what sends the operation on to the mirror -- but
                    # `str.__mod__` accepts whatever it is handed and
                    # formats it, so `SubS("%d") % obj` raises the format
                    # error and never reaches `obj.__rmod__`. CPython does
                    # the same.
                    if isinstance(by, Instance) and not (
                            sym == "%" and isinstance(bx, str)):
                        continue
                    # A BUILTIN THAT RAN AND REFUSED IS NOT A DECLINE, and
                    # the mirror does not get a turn after it: CPython reports
                    # `can only concatenate list (not "int") to list` for
                    # `Sub([1]) + 5`, which is `_reject`'s own text.
                    bad = _reject(h, sym, bx, by)
                    if bad is not None:
                        return bad
                    try:
                        return _result(h, op(bx, by))
                    except _UserFailed:
                        return 0
                    except TypeError as e:
                        # PYTHON'S OWN TEXT WHEN IT NAMES THE RIGHT KINDS,
                        # which for two unwrapped builtins it does -- `[1] +
                        # 5` is `can only concatenate list (not "int") to
                        # list`, the message CPython reports for
                        # `Sub([1]) + 5`. The same test the plain path makes
                        # below, and for the same reason.
                        if (type(bx).__name__ != h.kind_name(bx)
                                or type(by).__name__ != h.kind_name(by)):
                            return h._binop_error(sym.split(" ")[0], x, y)
                        return h._fail_like(e)
                    except (ValueError, ZeroDivisionError, OverflowError) as e:
                        return h._fail_like(e)
                who, other, which = ((x, y, direct) if step is _DIRECT
                                     else (y, x, reflected))
                if not isinstance(who, Instance) or who.cls.find(which) is None:
                    continue
                got = _user(h, lambda: who._send(which, other), fail=_FAILED)
                if got is _FAILED:
                    return 0
                if got is not NotImplemented:
                    return h._value(got)
            # NOBODY ANSWERED, and the refusal names what the PROGRAM wrote
            # rather than what the builtin step unwrapped it to.
            return h._binop_error(sym.split(" ")[0], x, y)
        bad = _reject(h, sym, x, y)
        if bad is not None:
            return bad
        try:
            return _result(h, op(x, y))
        except _UserFailed:
            return 0
        except TypeError as e:
            # Python's own message would name `Instance` -- this file's class
            # and not the program's type -- and the same is true of every
            # other wrapper here: `list[int] + 1` reported `'Alias'`, a name
            # no program has ever seen, where CPython says
            # `'types.GenericAlias'`. THE TEST IS WHETHER THE TWO NAMES AGREE:
            # for an int, a str or a list the Python class IS the kind and the
            # message is already right, so it is kept along with the more
            # specific text CPython has for those pairs.
            if (type(x).__name__ != h.kind_name(x)
                    or type(y).__name__ != h.kind_name(y)):
                return h._binop_error(sym.split(" ")[0], x, y)
            return h._fail_like(e)
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            return h._fail_like(e)
    return run


#: The conversions that take a NUMBER, grouped by what they take and by how
#: CPython words the refusal -- which is three different ways, not one
#: generalised one. `%d` takes any real number; `%x` REFUSES A FLOAT, which
#: `%d` accepts; and the floating conversions do not name the conversion at
#: all.
_PERCENT_REAL = ("d", "i", "u")
_PERCENT_INT = ("x", "X", "o")
_PERCENT_FLOAT = ("e", "E", "f", "F", "g", "G")


def _percent_needs(h, conv, value):
    """CPython's refusal for an argument a `%` conversion cannot take, or None.

    THE WRONG EXCEPTION TYPE IS THE PART THAT MATTERS, and it is why this
    exists rather than letting the failure surface on its own. `%` is
    implemented by translating into the format MINI-LANGUAGE and handing the
    argument to `format()` -- which is what keeps `%05.2f` and `{:05.2f}`
    from being written twice -- and what `format()` complains about is an
    unknown FORMAT CODE, a ValueError. What `%` complains about is the
    ARGUMENT, a TypeError. So `"%d" % "a"` raised `Unknown format code 'd'
    for object of type 'str'` where CPython raises `%d format: a real number
    is required, not str`, and a program catching TypeError around a `%`
    missed it entirely.
    """
    name = h.kind_name(value)
    numeric = _is_int_like(value) or isinstance(value, float)
    if conv in _PERCENT_REAL and not numeric:
        return h._fail("TypeError",
                       f"%{conv} format: a real number is required, "
                       f"not {name}")
    if conv in _PERCENT_INT and not _is_int_like(value):
        # A FLOAT IS REFUSED HERE AND ACCEPTED BY `%d`, which is CPython's
        # rule and not an oversight: `%x` of a non-integer has no answer,
        # where `%d` of one truncates.
        return h._fail("TypeError",
                       f"%{conv} format: an integer is required, not {name}")
    if conv in _PERCENT_FLOAT and not numeric:
        # NO CONVERSION IN THE TEXT. CPython's message for these names only
        # what was wanted, which is the one place it does not say `%e
        # format:`.
        return h._fail("TypeError", f"must be real number, not {name}")
    return None


def _percent(h, fmt, right):
    """`"%d %s" % (1, "a")` -- printf-style formatting.

    TRANSLATED INTO THE MINI-LANGUAGE and handed to `format()`, which is what
    the C does with its own -- `%05.2f` and `{:05.2f}` mean the same thing, so
    the padding and the presentation types are not written twice.

    NOT Python's own `%`, which is the obvious implementation: an argument may
    be one of this file's objects, and `"%s" % P(1)` has to reach the user's
    `__str__` through `_text` rather than printing an address. That is the
    same mistake the container renderer made.
    """
    raw = isinstance(fmt, (bytes, bytearray))
    text = fmt.decode("latin-1") if raw else fmt
    many = isinstance(right, tuple)
    # A MAPPING ON THE RIGHT supplies NAMED fields only, and nothing is
    # consumed positionally -- so an unused entry is not an error.
    mapping = isinstance(right, dict)
    args = right if many else (right,)
    out, i, at, n = [], 0, 0, len(text)
    while i < n:
        if text[i] != "%":
            out.append(text[i]); i += 1; continue
        i += 1
        if i < n and text[i] == "%":
            out.append("%"); i += 1; continue
        named = None
        if i < n and text[i] == "(":
            # `%(name)s` -- the MAPPING FORM. The key runs to the matching
            # `)`; what follows is an ordinary spec.
            if not mapping:
                return h._fail("TypeError", "format requires a mapping")
            i += 1
            key = ""
            while i < n and text[i] != ")":
                key += text[i]; i += 1
            if i < n and text[i] == ")":
                i += 1
            if key not in right:
                return h._fail("KeyError", repr(key))
            named = right[key]
        # THE FLAGS ARE COLLECTED, NOT EMITTED: two of them depend on the
        # conversion, which has not been read yet, and the mini-language fixes
        # an order (align, sign, `#`, `0`, width) that printf does not.
        flags, width, prec = set(), "", ""
        while i < n and text[i] in "-+ 0#":
            flags.add(text[i]); i += 1
        while i < n and text[i].isdigit():
            width += text[i]; i += 1
        if i < n and text[i] == ".":
            prec += text[i]; i += 1
            while i < n and text[i].isdigit():
                prec += text[i]; i += 1
        if i >= n:
            return h._fail("ValueError", "incomplete format")
        conv = text[i]; i += 1
        is_text = conv in ("s", "r", "a", "c", "b")
        # PRINTF RIGHT-ALIGNS A STRING; the mini-language left-aligns one.
        # The only difference between the two that is not a spelling.
        spec = "<" if "-" in flags else (">" if is_text else "")
        spec += "+" if "+" in flags else (" " if " " in flags else "")
        spec += "#" if "#" in flags else ""
        spec += "0" if ("0" in flags and "-" not in flags
                        and not is_text) else ""
        spec += width + prec
        if named is None:
            if at >= len(args):
                return h._fail("TypeError",
                               "not enough arguments for format string")
            value = args[at]; at += 1
        else:
            value = named
        if conv in ("s", "b"):
            if raw and isinstance(value, (bytes, bytearray)):
                # `b"%s" % b"ab"` inserts THE BYTES, not their repr.
                value = bytes(value).decode("latin-1")
            else:
                value = h._text(value, False)
        elif conv == "r":
            value = h._text(value, True)
        elif conv == "a":
            # The same `backslashreplace` the `ascii` builtin uses, so the
            # two cannot drift on which escape width a code point gets.
            value = h._text(value, True).encode(
                "ascii", "backslashreplace").decode("ascii")
        elif conv == "c":
            # ONE CHARACTER OR AN INT, and nothing else. A string of any
            # other length is refused BY ITS LENGTH, which is how CPython
            # words it -- `%c requires an int or a unicode character, not a
            # string of length 2`.
            if _is_int_like(value):
                value = chr(int(value))
            elif isinstance(value, str) and len(value) == 1:
                pass
            else:
                wanted = ("a string of length "
                          + str(len(value)) if isinstance(value, str)
                          else h.kind_name(value))
                return h._fail("TypeError",
                               f"%c requires an int or a unicode character, "
                               f"not {wanted}")
        else:
            spec += "d" if conv in ("i", "u") else conv
        if isinstance(value, Instance):
            # A USER OBJECT REACHES A NUMERIC CONVERSION THROUGH ITS NUMBER
            # and not through `__format__`: CPython's `%d` asks `__index__`
            # and `%f` asks `__float__`. Handing the object straight to
            # `format()` reached `object.__format__`, which refuses a
            # non-empty spec -- so `"%d" % obj` failed with `unsupported
            # format string passed to Instance.__format__`, naming a dunder
            # the user never wrote and `%` never consults.
            want = "__float__" if conv in "eEfFgG" else "__index__"
            try:
                got = value._send(want)
                if got is NotImplemented and want == "__index__":
                    got = value._send("__int__")
            except _UserFailed:
                return 0
            if h.err is not None:
                return 0
            # A CLASS THAT ANSWERED NEITHER is refused in the SAME WORDS a
            # builtin of the wrong kind is, below: what `%d` wants does not
            # depend on whether the argument came from a class.
            if got is not NotImplemented:
                value = got
        if conv not in ("s", "b", "r", "a", "c"):
            bad = _percent_needs(h, conv, value)
            if bad is not None:
                return bad
            # `%d` OF A FLOAT TRUNCATES, which is why it takes a real number
            # and `%x` takes an integer. The mini-language's `d` refuses one
            # outright, so the truncation has to happen before the handoff --
            # `"%d" % 1.5` is `'1'`, not an unknown format code.
            if conv in _PERCENT_REAL and isinstance(value, float):
                value = int(value)
        try:
            out.append(format(value, spec))
        except (ValueError, TypeError) as exc:
            return h._fail_like(exc)
    # A MAPPING has nothing to leave unconsumed: its entries are reached by
    # name, and an unused one is ordinary.
    if not mapping and at < len(args):
        return h._fail("TypeError",
                       "not all arguments converted during string formatting")
    joined = "".join(out)
    return h._new(joined.encode("latin-1") if raw else joined)


def _reject(h, sym: str, x, y):
    """The C's kind rules, for the cases where they are not Python's."""
    if sym == "%" and isinstance(x, (str, bytes, bytearray)):
        # `%` on a str or on bytes is PRINTF-STYLE FORMATTING, not arithmetic.
        return _percent(h, x, y)
    if sym == "+" and isinstance(x, str) and not isinstance(y, str):
        # UNLESS THE RIGHT OPERAND WRITES `__radd__`. `"the " + obj` is
        # `str.__add__` answering NotImplemented and CPython then asking
        # `type(obj).__radd__`, which is how a `StrEnum` member concatenates.
        # Rejecting here ran BEFORE the reflected dispatch below and reported
        # `can only concatenate str (not "Colours") to str` about a class that
        # defines exactly the method for it. `None` means "not rejected", so
        # the reflected call gets its turn.
        if isinstance(y, Instance) and y.cls.find("__radd__") is not None:
            return None
        return h._fail("TypeError",
                       f'can only concatenate str (not "{h.kind_name(y)}") '
                       f'to str')
    if sym == "+" and isinstance(x, (list, tuple)) and type(x) is not type(y):
        # THE CONCATENATION WORDING, FROM THE LEFT OPERAND'S KIND, which is
        # what CPython says: `[1] + (2,)` is `can only concatenate list (not
        # "tuple") to list`. This deferred to the C and said so, back when the
        # C answered the generic operand text -- `apy_add` has worded it this
        # way for some time and this was the half that did not follow, so
        # `[1] + 5` read as an unsupported operand pair here and as a failed
        # concatenation everywhere else.
        return h._fail("TypeError",
                       f'can only concatenate {h.kind_name(x)} '
                       f'(not "{h.kind_name(y)}") to {h.kind_name(x)}')
    if sym == "*" and (isinstance(x, str) or isinstance(y, str)):
        if not (_is_int_like(x) or _is_int_like(y)):
            other = y if isinstance(x, str) else x
            return h._fail(
                "TypeError",
                f"can't multiply sequence by non-int of type "
                f"'{h.kind_name(other)}'")
    if sym in ("&", "|", "^", "-") and isinstance(x, (set, frozenset)):
        # Set algebra. `&`, `|`, `^` and `-` are the four the C implements on
        # sets, and both operands must BE sets -- Python's operators say the
        # same, unlike its methods, which accept any iterable.
        if not isinstance(y, (set, frozenset)):
            return h._binop_error(sym, x, y)
        return None
    if sym == "|" and isinstance(x, dict) and isinstance(y, dict):
        # `d1 | d2` MERGES, with the right-hand side winning -- PEP 584. Not
        # the bitwise operator despite the spelling.
        return None
    if sym in ("&", "|", "^", "<<", ">>"):
        # A USER OBJECT WRITING THE DUNDER IS NOT AN INT-LIKE and is still
        # entitled to the operator: `Perm.R | Perm.W` is `Flag.__or__`, which
        # is the whole of what a flag enum is. This rule ran BEFORE the
        # reflected dispatch and reported `unsupported operand type(s) for |`
        # about two objects whose class defines `__or__`. `None` means "not
        # rejected", so the dispatch below gets its turn and answers the
        # unsupported-pair error itself when neither side has the method.
        if isinstance(x, Instance) or isinstance(y, Instance):
            return None
        if not (_is_int_like(x) and _is_int_like(y)):
            return h._binop_error(sym, x, y)
    return None


def _is_type_like(v) -> bool:
    """Is this something `|` should read as a TYPE? A builtin type used as a
    value, a user class, `None`, or ANY parameterised type.

    ANY ALIAS, not only a union: `list[int] | None` and
    `Annotated[str, 'o'] | None` are both ordinary unions in CPython, and
    requiring the origin to be a `typing` special form refused the first and
    mis-read the second.
    """
    if isinstance(v, Func) and getattr(v, "is_type", False):
        return True
    if isinstance(v, Class) or v is None:
        return True
    return isinstance(v, Alias)


def _union_arms(v) -> list:
    """`v`'s arms. A union contributes its own rather than itself, so unions
    flatten instead of nesting.

    ONLY A UNION FLATTENS. Every `typing` special form has an `Instance`
    origin, so testing for one flattened `Annotated[str, 'o']` too and
    `Optional[Annotated[str, 'o']]` came out as `str | 'o' | None` -- the
    metadata promoted to an arm of the union.
    """
    if isinstance(v, Alias) and _form_name(v.origin) == "Union":
        return list(v.args)
    # `None` IN A UNION IS `NoneType`. `int | None` is written with the VALUE
    # and holds the TYPE -- `get_args` answers `<class 'NoneType'>` in
    # CPython, and keeping the singleton answered `None`, a different object
    # with a different repr.
    if v is None:
        return [_none_type()]
    return [v]


def _form_name(v):
    """The name of a `typing` special form, or None if `v` is not one."""
    if isinstance(v, Instance) and v.cls is _SPECIAL_FORM_CLASS:
        got = v.dict.get("_name")
        return str(got) if got is not None else None
    return None


def _arm_key(args):
    """A union's arms as an ORDER-FREE, DUPLICATE-FREE key.

    A `frozenset` would be the obvious one and cannot be used: an arm may be
    an `Instance` whose `__hash__` is the program's own, and hashing one from
    inside `Alias.__hash__` re-enters user code. Identity is what the arms are
    compared by everywhere else here, so identity is what keys them.
    """
    return frozenset(id(one) for one in args)


def _dedup_arms(arms):
    """`int | str | int` HAS TWO ARMS. CPython drops a repeat when it builds
    the union, so the repr shows it once; keeping it showed `int | str | int`
    where CPython shows `int | str`."""
    out = []
    for one in arms:
        if not any(x is one for x in out):
            out.append(one)
    return out


def _union_form(h):
    """The `Union` special form every union is built on, interned by name like
    every other form."""
    return h._get(_apy_typing_form(h, [h._new("Union")]), "apy_bitor")


def _binop_error(h, sym: str, x, y):
    return h._fail(
        "TypeError",
        f"unsupported operand type(s) for {sym}: "
        f"'{h.kind_name(x)}' and '{h.kind_name(y)}'")


ObjectHost._binop_error = _binop_error


def _pow(x, y):
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        if x < 0 and isinstance(y, float) and y != math.floor(y):
            # C DECIDES: CPython returns a complex here. There is no complex
            # kind, so the runtime reports -- the one place it knowingly
            # raises where CPython answers.
            raise ValueError("negative number cannot be raised to a "
                             "fractional power (no complex support)")
    return x ** y


def _apy_neg(h, a):
    v = h._get(a[0], "apy_neg")
    if isinstance(v, Instance):
        return _user(h, lambda: h._value(-v))
    # Complex negates both parts. `_is_num` deliberately says no to it -- that
    # predicate gates the ORDERED numeric paths, which complex is not part of.
    if isinstance(v, complex):
        return h._new(-v)
    if not _is_num(v):
        return h._fail("TypeError",
                       f"bad operand type for unary -: '{h.kind_name(v)}'")
    return _result(h, -v)


def _apy_pos(h, a):
    v = h._get(a[0], "apy_pos")
    if isinstance(v, Instance):
        return _user(h, lambda: h._value(+v))
    if isinstance(v, complex):
        return h._new(+v)
    if not _is_num(v):
        return h._fail("TypeError",
                       f"bad operand type for unary +: '{h.kind_name(v)}'")
    return _result(h, +v)


def _apy_invert(h, a):
    v = h._get(a[0], "apy_invert")
    if isinstance(v, Instance):
        return _user(h, lambda: h._value(~v))
    if not _is_int_like(v):
        return h._fail("TypeError",
                       f"bad operand type for unary ~: '{h.kind_name(v)}'")
    # `int(v)` rather than `~v`: `~True` is deprecated on a bool and this is
    # the int operation the C performs on the payload.
    return _result(h, ~int(v))


# ── comparison ──────────────────────────────────────────────────────────────

#: The comparison operators and their dunders, with the REFLECTED name being
#: the MIRRORED operator rather than an `__r`-prefixed one: `a < b` falls back
#: to `b.__gt__(a)`, because what b is asked is the comparison as seen from
#: its side. Comparisons are the one family where that is true.
_CMP_DUNDER = {
    "apy_lt": ("__lt__", "__gt__"), "apy_le": ("__le__", "__ge__"),
    "apy_gt": ("__gt__", "__lt__"), "apy_ge": ("__ge__", "__le__"),
    "apy_eq": ("__eq__", "__eq__"), "apy_ne": ("__ne__", "__ne__"),
}


#: The three things a comparison between an instance and anything else may
#: try, in the order `_order_plan` puts them.
_DIRECT, _HELD, _MIRROR = "direct", "held", "mirror"


def _order_plan(x, y, mirror):
    """The order `x OP y` tries the written dunders and the builtin between.

    THE BUILTIN SITS IN THE MIDDLE, and that is the whole of divergence #112.
    `type(a).__lt__` is the one the class WROTE or -- for a class extending a
    builtin that wrote none -- the one it INHERITED, and in CPython an
    inherited slot wins OUTRIGHT over a `__gt__` the other side wrote:
    `OnlyGt([1]) < OnlyGt([1, 2])` is True, because `list.__lt__` answers and
    the reflected `__gt__` is never reached. Trying both written dunders
    first and only then reading the builtin answered `'GT'` -- the class's
    own method, for a comparison the class never claimed.

    THE ONE REORDERING RULE IS CPYTHON'S OWN. A right operand whose type is a
    PROPER SUBCLASS of the left's and which overrides the mirror goes first,
    so `[1] < OnlyGt([1, 2])` asks `OnlyGt.__gt__` before `list.__lt__` --
    the subclass is presumed to know more about the pair than its base does.
    Its own direct dunder still follows, because the mirror may answer
    NotImplemented. Without this the reorder above would have broken the
    three cases that were already right.
    """
    if isinstance(y, Instance) and y.cls.find(mirror) is not None:
        if isinstance(x, Instance):
            proper = y.cls is not x.cls and y.cls.is_sub(x.cls)
        else:
            # X IS NOT AN INSTANCE, so "a proper subclass of type(x)" is
            # asking whether y extends the very builtin x is one of.
            kind = y.cls.builtin_kind()
            proper = kind is not None and type(x) is kind
        if proper:
            return (_MIRROR, _DIRECT, _HELD)
    return (_DIRECT, _HELD, _MIRROR)


def _cmpop(name, op):
    def run(h, a):
        x = h._get(a[0], name)
        y = h._get(a[1], name)
        # WHAT THE PROGRAM COMPARED, kept for the refusal below: `x` and `y`
        # are replaced by the builtin a class extends before the comparison,
        # and naming those said `'int' and 'tuple'` about a namedtuple.
        wrote = (x, y)
        # A USER CLASS ANSWERS WITH WHATEVER ITS DUNDER RETURNS, not with a
        # bool: `__lt__` returning a string is legal and its result is the
        # value of `<`. Coercing it -- which `h._bool` does -- turned every
        # such answer into True.
        if isinstance(x, Instance) or isinstance(y, Instance):
            direct, reflected = _CMP_DUNDER[name]
            for step in _order_plan(x, y, reflected):
                if step is _HELD:
                    # THE BUILTIN THE LEFT SIDE EXTENDS, which is its
                    # INHERITED direct dunder -- `list.__lt__` for a class
                    # extending list that wrote no `__lt__` of its own. The
                    # right side is unwrapped too and WITHOUT ASKING what it
                    # wrote, because it is only the operand here: CPython's
                    # `list.__lt__(a, b)` takes any list subclass for `b` and
                    # never consults its methods.
                    bx = _as_builtin(x, (direct,))
                    by = y.held if isinstance(y, Instance) \
                        and y.held is not None else y
                    if isinstance(bx, Instance) or isinstance(by, Instance):
                        # NOT A BUILTIN COMPARISON AFTER ALL. One side is
                        # still an object the builtin knows nothing about, so
                        # `list.__lt__` would answer NotImplemented -- which
                        # is exactly when CPython goes on to the mirror, AND
                        # gives it the operands the PROGRAM wrote. Comparing
                        # the unwrapped pair here instead reached the mirror
                        # with a bare list where `Sub([1]) < Watcher()` should
                        # hand `Watcher.__gt__` the Sub.
                        continue
                    try:
                        return h._bool(op(bx, by))
                    except _UserFailed:
                        return 0
                    except TypeError:
                        # NOT ORDERABLE AS THE BUILTINS EITHER, which is not
                        # the end: `list.__lt__` answering NotImplemented is
                        # exactly when CPython goes on to the mirror.
                        continue
                who, other, which = ((x, y, direct) if step is _DIRECT
                                     else (y, x, reflected))
                if not isinstance(who, Instance) or who.cls.find(which) is None:
                    continue
                try:
                    got = who._send(which, other)
                except _UserFailed:
                    return 0
                if got is not NotImplemented:
                    return h._value(got)
            # NOTHING ANSWERED, so the comparison below is the one that words
            # the refusal. Both sides are read as the builtin they extend for
            # the same reason the plan reads one: `Counter(a=1) == {'a': 1}`
            # is True in CPython because a class that wrote no `__eq__`
            # compares as the builtin.
            x = _as_builtin(x, (direct, reflected))
            y = _as_builtin(y, (direct, reflected))
        try:
            return h._bool(op(x, y))
        except _UserFailed:
            return 0
        except TypeError as e:
            if name in _CMP_SYMBOL and (isinstance(wrote[0], Instance)
                                        or isinstance(wrote[1], Instance)):
                # PYTHON'S MESSAGE NAMES `Instance`, which is THIS FILE'S class
                # and not the program's type. Every user object shares it, so a
                # dataclass whose `__lt__` answered NotImplemented reported
                # `'<' not supported between instances of 'Instance' and 'int'`
                # -- naming a class the program has never heard of. The
                # arithmetic path already rewrote this; comparison did not, so
                # `a + b` blamed the right types and `a < b` did not.
                return h._fail(
                    "TypeError",
                    f"'{_CMP_SYMBOL[name]}' not supported between instances "
                    f"of '{h.kind_name(wrote[0])}' and "
                    f"'{h.kind_name(wrote[1])}'")
            return h._fail_like(e)
    return run


#: The operator each ORDERING binding spells, for the message above. Only the
#: four orderings: `==` and `!=` never reach it, because equality between any
#: two objects is always answerable, and membership of this table is what
#: selects the branch.
_CMP_SYMBOL = {"apy_lt": "<", "apy_le": "<=", "apy_gt": ">", "apy_ge": ">="}


def _apy_is(h, a):
    """Identity, which is HANDLE equality -- not `is` on the Python objects.

    Python interns far more than the runtime does (every string literal, for
    one), so asking `x is y` about the backing objects would answer True where
    a compiled program answers False. The handle is the address.
    """
    h._get(a[0], "apy_is")
    h._get(a[1], "apy_is")
    return h._bool(int(a[0]) == int(a[1]))


def _apy_contains(h, a):
    needle = h._get(a[0], "apy_contains")
    hay = h._get(a[1], "apy_contains")
    # `x in gen` CONSUMES the generator up to the match and leaves the rest --
    # a generator is consumed once. Draining it reported the generator as not
    # iterable at all.
    if isinstance(hay, Gen):
        while True:
            item, done = _gen_step(h, hay, None)
            if done is None:
                return 0
            if done:
                return h._new(False)
            if item == needle or item is needle:
                return h._new(True)
    # `x in it` CONSUMES the cursor up to the match and leaves the rest, for
    # the same reason `x in gen` does. Without this arm the fallback asked
    # PYTHON's `in` about a class of this file's, and Python said so:
    # `argument of type 'Iterator' is not a container or iterable`, about
    # every `map`, `filter`, `zip`, `enumerate`, `reversed` and `iter(x)`
    # there is.
    if isinstance(hay, Iterator):
        while True:
            got = _apy_step(h, [h._value(hay)])
            if not got:
                return 0
            item = h._get(got, "apy_contains")
            if item is _STOP:
                return h._new(False)
            if item == needle or item is needle:
                return h._new(True)
    # AN ALIAS IS ITERABLE and yields one item, so membership is that one
    # comparison. Written out rather than left to the fallback below, which
    # would ask PYTHON about a class of this file's and be told it is not a
    # container.
    if isinstance(hay, Alias):
        return h._new(needle == Alias(hay.origin, hay.args, True))
    if isinstance(hay, Class) and hay.meta is not None:
        # MEMBERSHIP IN A CLASS IS THE METACLASS'S BUSINESS, as iterating and
        # measuring one are: `Colour.RED in Colour` is
        # `type(Colour).__contains__(Colour, RED)`. The third of the three to
        # need saying so -- see `_apy_iter` and `_apy_len`.
        hook = hay.meta.lookup("__contains__")
        if hook is not _ABSENT:
            got = h._invoke(hook, [hay, needle])
            if h.err is not None:
                return 0
            return h._bool(bool(got))
    if isinstance(hay, Instance) and hay.cls.find("__contains__") is None:
        # No `__contains__`. `in` falls back to ITERATION, which is CPython's
        # rule and the reason a class with only `__getitem__` supports it.
        walked = _apy_iterable(h, [a[1]])
        if not walked:
            # `in` REPORTS ITS OWN REFUSAL and not the iteration's, as
            # CPython does: what was asked is a membership question, so a
            # class with none of the three is `argument of type 'C' is not a
            # container or iterable` rather than `'C' object is not
            # iterable`, which is a claim about something else.
            h.err = None
            h.err_value = None
            return h._fail("TypeError",
                           f"argument of type '{h.kind_name(hay)}' is not "
                           f"a container or iterable")
        if walked != a[1]:
            return _apy_contains(h, [a[0], walked])
        # `_apy_iterable` left it alone, which means `__len__` plus
        # `__getitem__`: the index walk IS its protocol, so walk it.
        n = _apy_raw_len(h, [a[1]])
        if h.err is not None:
            return 0
        for i in range(n):
            try:
                item = hay._send("__getitem__", i)
            except _UserFailed:
                return 0
            if h.err is not None:
                return 0
            if item == needle:
                return h._bool(True)
        return h._bool(False)
    # PYTHON'S OWN `in` FOR THE KINDS IT SHARES, which is what keeps the
    # unhashable-key rule and the str/bytes substring searches from being
    # restated here -- and for an Instance, which reaches its `__contains__`.
    if isinstance(hay, (list, tuple, set, frozenset, dict, str, bytes,
                        bytearray, range, memoryview, Instance)) \
            or isinstance(hay, _VIEW_TYPES):
        # A str SUBCLASS IS A str FOR THE SUBSTRING SEARCH. Only for one:
        # `S("b") in ["a", S("b")]` is a comparison and must stay the
        # instance's, which `__eq__` may have been written for.
        if isinstance(hay, (str, bytes, bytearray)):
            needle = _held_text(needle)
        try:
            return h._bool(needle in hay)
        except _UserFailed:
            return 0
        except TypeError as e:
            return h._fail_like(e)
    # EVERYTHING ELSE IS REFUSED HERE AND NOT BY PYTHON, which would name a
    # class of this file's: `1 in f` for a function said `argument of type
    # 'Func' is not a container or iterable`.
    return h._fail("TypeError", f"argument of type '{h.kind_name(hay)}' "
                                f"is not a container or iterable")


# ── conversions ─────────────────────────────────────────────────────────────

def _dunder_number(h, v, names):
    """The first of `names` the class defines, called. None when it has none.

    A CLASS SAYS WHAT ITS NUMBER IS -- answering from the numeric tower
    instead converts something the class never claimed was a number.
    """
    if not isinstance(v, Instance):
        return None
    for name in names:
        if v.cls.find(name) is not None:
            got = _user(h, lambda: v._send(name), fail=_FAILED)
            return None if got is _FAILED else got
    return None


def _apy_to_int(h, a):
    v = h._get(a[0], "apy_to_int")
    # `__int__` first and `__index__` after it: the two are not the same
    # question, and a class may define only the second.
    got = _dunder_number(h, v, ("__int__", "__index__"))
    if got is not None:
        return h._value(got)
    if h.err is not None:
        return 0
    # AND WITH NEITHER DUNDER, THE BUILTIN IT EXTENDS ANSWERS. `int(S("12"))`
    # for a `class S(str)` parses the text in CPython -- the conversion reads
    # the C-level layout, which a subclass has.
    v = _held_text(v)
    if not isinstance(v, (int, float, str)):
        return h._fail("TypeError",
                       f"int() argument must be a string, a bytes-like "
                       f"object or a real number, not '{h.kind_name(v)}'")
    try:
        n = int(v)
    except (ValueError, OverflowError) as e:
        return h._fail_like(e)
    # THE SATURATION THAT USED TO BE HERE IS GONE. `int('1' + '0' * 40)` was
    # clamped to INT64_MAX, because the C reached `strtoll` and strtoll
    # saturates -- "neither is CPython's answer, but the two paths have to
    # give the SAME wrong answer". The C parses arbitrary length now and gives
    # CPython's answer, so keeping the clamp would make this the only one of
    # the three still wrong.
    return _result(h, n)


def _apy_to_float(h, a):
    _v = h._get(a[0], "apy_to_float")
    # `__float__`, and `__index__` after it: an object that can be an integer
    # can be a float, which is the rule CPython follows too.
    _got = _dunder_number(h, _v, ("__float__", "__index__"))
    if _got is not None:
        return h._value(float(_got) if isinstance(_got, int) else _got)
    if h.err is not None:
        return 0
    # THE BUILTIN A CLASS EXTENDS ANSWERS -- see `_apy_to_int`.
    v = _held_text(h._get(a[0], "apy_to_float"))
    if not isinstance(v, (int, float, str)):
        return h._fail("TypeError",
                       f"float() argument must be a string or a real number, "
                       f"not '{h.kind_name(v)}'")
    try:
        # THROUGH `_value`: `float(x)` on a float is `x`, which is what
        # Python's own `float` answers and what both compiled runtimes do.
        return h._value(float(v))
    except ValueError as e:
        return h._fail_like(e)


def _apy_to_bool(h, a):
    return h._bool(bool(h._get(a[0], "apy_to_bool")))


# ── the table ───────────────────────────────────────────────────────────────
# Written out rather than derived from `OBJECT_NAMES`, so a symbol added to
# the C without a binding here is a KeyError naming it rather than a program
# that silently does nothing. `tests/.../test_endtoend.py` asserts the two
# sets are equal.


# ── arbitrary precision integers ────────────────────────────────────────────
# Every one of these is a delegation, because Python's integers are exactly
# what the C's second integer kind implements. The C had to be written; this
# only has to agree with it, and the differential tool checks that the two
# do -- see tools/objects_diff.py.

def _apy_pow3(h, a):
    base = h._get(a[0], "apy_pow3")
    exp = h._get(a[1], "apy_pow3")
    mod = h._get(a[2], "apy_pow3")
    if not all(isinstance(v, int) for v in (base, exp, mod)):
        return h._fail("TypeError",
                       "pow() 3rd argument not allowed unless all arguments "
                       "are integers")
    try:
        return h._int(pow(int(base), int(exp), int(mod)))
    except (ValueError, ZeroDivisionError) as exc:
        return h._fail_like(exc)


def _apy_divmod(h, a):
    x = h._get(a[0], "apy_divmod")
    y = h._get(a[1], "apy_divmod")
    # `__divmod__`/`__rdivmod__`, the same explicit direct-then-reflected
    # shape `_binop`'s `_OP_DUNDER` loop uses for `+`/`-`/`*`/... -- but
    # `divmod()` is a BUILTIN rather than an operator, so it was never
    # routed through that table, and a class defining `__divmod__`
    # (`bundled/datetime.py`'s `timedelta`, for `divmod(td, other_td)`)
    # got "unsupported operand type(s) for divmod(): 'Instance' and
    # 'Instance'" -- naming this file's own class, about two objects that
    # plainly define the method being asked for. The same miss `_apy_abs`
    # above already guards against for `__abs__`.
    if isinstance(x, Instance) or isinstance(y, Instance):
        for who, other, which in ((x, y, "__divmod__"),
                                  (y, x, "__rdivmod__")):
            if not isinstance(who, Instance) or who.cls.find(which) is None:
                continue
            got = _user(h, lambda who=who, other=other, which=which:
                       who._send(which, other), fail=_FAILED)
            if got is _FAILED:
                return 0
            if got is not NotImplemented:
                return h._value(got)
        return h._binop_error("divmod()", x, y)
    bad = _reject(h, "divmod()", x, y)
    if bad is not None:
        return bad
    try:
        q, r = divmod(x, y)
    except ZeroDivisionError:
        return h._fail("ZeroDivisionError", "division by zero")
    # RAW VALUES, not handles: a tuple object's Python-level body holds the
    # elements themselves (see `Interpreter._text`'s container branch, and
    # `Iterator.ENUMERATE` just below building `(at, v)` the same way) --
    # `h._value()` mints a HANDLE, and wrapping each element in one before
    # `h._new` double-boxed them, so reading the tuple back printed the
    # handle's cell INDEX instead of the quotient/remainder it pointed to.
    return h._new((q, r))


def _apy_bit_length(h, a):
    v = h._get(a[0], "apy_bit_length")
    if not isinstance(v, int):
        return h._fail("AttributeError",
                       f"'{h.kind_name(v)}' object has no attribute "
                       f"'bit_length'")
    return h._int(int(v).bit_length())


def _apy_bit_count(h, a):
    v = h._get(a[0], "apy_bit_count")
    if not isinstance(v, int):
        return h._fail("AttributeError",
                       f"'{h.kind_name(v)}' object has no attribute "
                       f"'bit_count'")
    return h._int(int(v).bit_count())


def _base_text(h, a, fn, render):
    v = _base_arg(h, h._get(a[0], fn), fn)
    if v is None:
        return 0
    return h._new(render(int(v)))


def _base_arg(h, v, fn):
    """`hex(obj)` goes through `__index__` -- PEP 357 names these three as the
    operations it exists for, alongside subscripting."""
    if isinstance(v, Instance) and v.cls.find("__index__") is not None:
        try:
            got = v._send("__index__")
        except _UserFailed:
            return None
        if _is_int_like(got):
            return got
    if not _is_int_like(v):
        h._fail("TypeError",
                f"{fn}() argument can't be interpreted as an integer")
        return None
    return v


def _apy_bin(h, a):
    return _base_text(h, a, "bin", bin)


def _apy_oct(h, a):
    return _base_text(h, a, "oct", oct)


def _apy_hex(h, a):
    return _base_text(h, a, "hex", hex)


def _apy_to_int_base(h, a):
    """`int(s, base)` for any base from 2 to 36, and base 0 -- which means
    "read the prefix", so `int('0x1f', 0)` is 31 and `int('17', 0)` is 17."""
    # A str SUBCLASS IS A str HERE TOO -- see `_held_text`.
    v = _held_text(h._get(a[0], "apy_to_int_base"))
    base = h._get(a[1], "apy_to_int_base")
    if not isinstance(v, str):
        return h._fail("TypeError",
                       "int() can't convert non-string with explicit base")
    if isinstance(base, bool) or not isinstance(base, int):
        return h._fail("TypeError",
                       f"'{h.kind_name(base)}' object cannot be interpreted "
                       f"as an integer")
    if base != 0 and not 2 <= base <= 36:
        return h._fail("ValueError", "int() base must be >= 2 and <= 36, or 0")
    try:
        return h._int(int(v, base))
    except ValueError:
        return h._fail("ValueError",
                       f"invalid literal for int() with base {base}: "
                       f"{v!r}")


# ── the object model ────────────────────────────────────────────────────────
# The kinds the C's `apy_obj` union holds that Python has no equivalent for. A
# str is a str and a list is a list -- those are modelled by the real thing,
# which is why `sorted` here IS `sorted` and its stability comes from CPython
# rather than from a restatement of it. These five have no such stand-in.

#: The IR type of a pointer-sized value, for reading an argument array out of
#: interpreter memory. Named here rather than imported into every use so the
#: import stays one line at the top of the file.
_PTR = _ir_types.PTR


class _UserFailed(Exception):
    """A user method reached from inside a runtime entry point failed.

    NOT an error report of its own: the error flag is already set, and the
    only thing left is to stop. Every entry point that can re-enter compiled
    code catches this and answers the C's NULL, so a failed `__eq__` inside a
    `sorted` looks exactly like a failed `apy_add` to the caller.
    """


def _class_module(cls) -> str | None:
    """The module a class was written in, for the two reprs that qualify a
    name: `<class '__main__.C'>` and `<__main__.C object at 0x...>`.

    Answers None for a class with none -- and for `builtins`, which CPython
    leaves OUT: `int` is `<class 'int'>` and not `<class 'builtins.int'>`.
    The class body puts it in the dict, which is where CPython keeps it too;
    see `_dyn_class`. The C twin is `apy_class_module`.
    """
    if not isinstance(cls, Class):
        return None
    where = cls.dict.get("__module__")
    if not isinstance(where, str) or where == "builtins":
        return None
    return where


def _inst_repr(v) -> str:
    """`<__main__.C object at 0x...>` -- the repr an instance gets when its
    class wrote none.

    The module qualifies the name here exactly as it does in `<class
    '__main__.C'>`; the C twin is the APY_INST_K arm of `apy_text_of`. Three
    call sites need it: `Instance.__repr__`, the `object.__repr__` default a
    class inherits, and `apy_default_repr` reached by name.
    """
    where = _class_module(v.cls)
    at = f"0x{id(v):x}"
    return (f"<{where}.{v.cls.name} object at {at}>" if where
            else f"<{v.cls.name} object at {at}>")


class Cell:
    """A captured variable's box.

    What a closure captures is the BOX and not the value in it, so two
    closures over one variable see each other's writes. Copying the value at
    capture time passes every test where only one closure exists and fails the
    one where two do.
    """

    __slots__ = ("slot",)

    def __init__(self, initial=None) -> None:
        self.slot = initial


def _bare_name(name: str) -> str:
    """A type's `__name__` is the LAST COMPONENT of its dotted name.

    Two kinds here are named the way CPython names them in a message --
    `types.GenericAlias` and `typing.Union` -- because that is what
    `unsupported operand type(s) for +` prints and what `<class '...'>` shows.
    `__name__` is the other half of the same rule: CPython answers
    `GenericAlias` and `Union`, with the module in `__module__` and never in
    the name. One dotted string serves both, split here.

    EVERY OTHER NAME PASSES THROUGH UNCHANGED. No identifier can hold a dot,
    so a class the program wrote is never affected.
    """
    return name.rsplit(".", 1)[-1]


#: "no such attribute", told apart from an attribute whose VALUE is None.
_ABSENT = object()


class Class:
    """A user class: its name, its bases, and what its body bound.

    MULTIPLE INHERITANCE through the C3 linearisation in `c3`. `base` is the
    FIRST base and stays for everything that asks a single question; `mro` is
    the order every lookup walks, and it is None for a class built with one
    base or none, where the chain and the order are the same walk.
    """

    __slots__ = ("name", "base", "dict", "meta", "bases", "mro", "builtin")

    def __init__(self, name: str, base=None, bases=None) -> None:
        self.name = name
        self.base = base
        self.dict: dict = {}
        #: The METACLASS that made this class, or None for an ordinary
        #: `class`, which reads as `type`. `type(C)` answers it, and a
        #: metaclass's `__instancecheck__` is reached through it.
        self.meta = None
        self.bases = list(bases) if bases else ([base] if base else [])
        self.mro = None
        #: `class D(dict)` -- the BUILTIN KIND this class extends, as the
        #: Python type, or None. Recorded rather than derived: the base list
        #: holds only classes and a builtin is not one.
        self.builtin = None

    def builtin_kind(self):
        """The builtin this class extends, looked up the whole chain: a
        subclass of a subclass of `dict` is still a dict."""
        for here in self.order():
            if here.builtin is not None:
                return here.builtin
        return None

    def order(self) -> list:
        """The classes a lookup walks, in order, starting with this one."""
        if self.mro is not None:
            return self.mro
        out, here = [], self
        while isinstance(here, Class):
            out.append(here)
            here = here.base
        return out

    def find(self, name: str):
        """The attribute, searching this class and then its bases. None when
        no class in the order has it -- the caller decides whether that is an
        error, because an instance still has its own dict to try."""
        for here in self.order():
            if name in here.dict:
                return here.dict[name]
        return None

    def lookup(self, name: str):
        """The attribute, or `_ABSENT` when no class in the order has it.

        Distinguished from `find`, which answers None for BOTH a missing name
        and one bound to None -- so a class attribute written `_one = None`,
        which is how every lazily-filled slot starts, was invisible.
        """
        for here in self.order():
            if name in here.dict:
                return here.dict[name]
        return _ABSENT

    def is_sub(self, other) -> bool:
        """Is `other` anywhere in this class's order? The `isinstance` rule
        for user classes."""
        return any(here is other for here in self.order())


def c3(cls, bases):
    """THE C3 LINEARISATION: the order attribute lookup walks.

    With one base it is the base chain and nothing is gained by computing it.
    With several it is the only order that keeps two promises at once -- a
    class comes before its bases, and the bases keep the order they were
    written in -- and no simple walk keeps both.

    Answers None when no order satisfies both, which is what CPython refuses
    for `class Z(X, Y)` where X and Y disagree.
    """
    queues = [list(b.order()) for b in bases] + [list(bases)]
    out = [cls]
    while any(queues):
        chosen = None
        for queue in queues:
            if not queue:
                continue
            head = queue[0]
            # A CANDIDATE IN SOME OTHER LIST'S TAIL MUST WAIT for that list,
            # or the result would put it before something that comes first.
            if any(head in other[1:] for other in queues):
                continue
            chosen = head
            break
        if chosen is None:
            return None
        if chosen not in out:
            out.append(chosen)
        for queue in queues:
            if queue and queue[0] is chosen:
                del queue[0]
    return out


class Instance:
    """A user object: its class, its own attributes, and the host that made it.

    The host reference is what lets a dunder re-enter the interpreter. One
    ObjectHost exists per interpreter, so this is a back-pointer and not a
    second source of truth.
    """

    __slots__ = ("cls", "dict", "h", "held")

    def __init__(self, cls, h) -> None:
        self.cls = cls
        self.dict: dict = {}
        self.h = h
        # AN INSTANCE OF A BUILTIN-EXTENDING CLASS CARRIES ONE. `class
        # D(dict)` with only a `__missing__` in its body still has to BE a
        # dict for everything it did not write, and this is that dict.
        kind = cls.builtin_kind() if isinstance(cls, Class) else None
        self.held = kind() if kind is not None else None

    # -- the dunder bridge --------------------------------------------------
    def _send(self, name: str, *args):
        """Call a method of this object's class, if it has one.

        `NotImplemented` means THE CLASS DOES NOT DEFINE IT, which every
        caller distinguishes from the method having returned None -- a class
        with no `__str__` falls back to `__repr__`, and one whose `__str__`
        returns None is an error.
        """
        m = self.cls.find(name)
        # A NATIVE COUNTS. A class the RUNTIME builds -- `asyncio.TaskGroup`
        # is one -- has natives in its dict rather than compiled functions,
        # and refusing them here reported every protocol method on such a
        # class as absent: `async with TaskGroup()` awaited NotImplemented.
        if m is None or not isinstance(m, (Func, Native)):
            return NotImplemented
        return self.h._invoke_obj(m.bind(self), list(args))

    def __repr__(self):
        out = self._send("__repr__")
        if out is not NotImplemented:
            return self.h._require_str(out, "__repr__")
        # A CLASS EXTENDING A BUILTIN SHOWS THE BUILTIN. `class D(dict)` with
        # no `__repr__` prints `{'a': 1}` in CPython, and printing
        # `<D object at 0x...>` instead hides the entire contents of the thing
        # -- which for a Counter or a defaultdict is the whole value. The
        # class name is not shown here because CPython does not show it
        # either: `repr(D({'a': 1}))` is `{'a': 1}`, and a subclass wanting
        # its name in the repr writes one, as `Counter` and `deque` do.
        if self.held is not None:
            return repr(self.held)
        return _inst_repr(self)

    def __str__(self):
        out = self._send("__str__")
        if out is NotImplemented:
            # STR, BYTES AND BYTEARRAY WRITE A `__str__` OF THEIR OWN, and
            # it beats a subclass's `__repr__` the way any inherited method
            # beats a fallback: `class S(str)` with only a `__repr__` still
            # prints its TEXT under `str()`. Every other builtin leaves
            # `tp_str` at object's, which reaches `tp_repr` and so the
            # written `__repr__` -- which is why `class D(dict)` with a
            # `__repr__` does print with it. Without this, `str(S("text"))`
            # was `"'text'"`: six characters where the program wrote four,
            # so the text no longer compared equal to itself. Mirrors the
            # C's `apy_text_of`.
            if isinstance(self.held, (str, bytes, bytearray)):
                return str(self.held)
            return self.__repr__()
        return self.h._require_str(out, "__str__")

    def __format__(self, spec):
        """`"{:03d}".format(obj)` -- reached through PYTHON'S formatting
        machinery rather than through `apy_format`.

        `_apy_str_format` hands the arguments to `str.format`, which asks the
        HOST object for `__format__`; without one it found `object`'s, which
        refuses a non-empty spec and reported `unsupported format string
        passed to Instance.__format__` -- naming a class the program has never
        heard of. `apy_format` had the case and this did not, so
        `format(obj, "03d")` worked and `"{:03d}".format(obj)` did not.
        """
        out = self._send("__format__", spec)
        if out is NotImplemented:
            return format(self.__str__(), spec)
        return self.h._require_str(out, "__format__")

    def __bool__(self):
        out = self._send("__bool__")
        if out is NotImplemented:
            # A CLASS THAT EXTENDS A BUILTIN has one for everything it did
            # not write -- `_apy_len` already asks `held` before walking the
            # dunder for exactly this reason, and this had not: `class
            # Counter(dict)` with no `__len__` of its own fell straight to
            # "no `__bool__` and no `__len__`", so `bool(Counter())` was
            # ALWAYS True and `not Counter()` was ALWAYS False, regardless
            # of content. `multimode('')` (an empty `Counter`) read its
            # `if not counts: return []` as unreachable and called
            # `max(counts.values())` on nothing.
            if self.cls.find("__len__") is None and self.held is not None:
                return bool(self.held)
            out = self._send("__len__")
            if out is NotImplemented:
                # A CLASS EXTENDING A BUILTIN IS TRUTHY BY THE BUILTIN'S OWN
                # RULE, exactly as `_apy_len` already answers `len(d)` from
                # `held` for the same class rather than raising -- `class
                # S(list): pass` with nothing in its own body still has to
                # answer False once emptied, the way an ordinary `[]` does.
                # `_send` only ever finds a dunder the CLASS ITSELF wrote,
                # which is why this is not already covered above.
                if self.held is not None:
                    return bool(self.held)
                # No `__bool__`, no `__len__`, and nothing held is ALWAYS
                # TRUE. Answering False would silently invert every bare
                # `if obj:`.
                return True
            return bool(out)
        return bool(out)

    def __len__(self):
        out = self._send("__len__")
        if out is NotImplemented:
            raise TypeError(f"object of type '{self.cls.name}' has no len()")
        return int(out)

    def __eq__(self, other):
        out = self._send("__eq__", other)
        if out is NotImplemented:
            # A CLASS EXTENDING A BUILTIN COMPARES AS THE BUILTIN. Identity is
            # the right default only for an object with no content to compare;
            # for two equal namedtuples it answers False, and together with a
            # hash that agreed they would both sit in the same set.
            got = _held_binary(self, "__eq__", other)
            return (self is other) if got is NotImplemented else got
        return out

    def __ne__(self, other):
        out = self._send("__ne__", other)
        if out is NotImplemented:
            eq = self._send("__eq__", other)
            if eq is NotImplemented:
                got = _held_binary(self, "__eq__", other)
                return (self is not other) if got is NotImplemented else not got
            return not eq
        return out

    def __hash__(self):
        out = self._send("__hash__")
        if out is NotImplemented:
            # A CLASS EXTENDING A BUILTIN HASHES AS THE BUILTIN, so two equal
            # namedtuples land in the same bucket. Identity would not: they
            # compare equal through the held tuple and would still both sit in
            # a set, which is the wrong ANSWER rather than an error.
            if self.held is not None and self.cls.find("__eq__") is None:
                return hash(self.held)
            # A class defining `__eq__` and not `__hash__` is UNHASHABLE in
            # Python, and saying so is what makes a dict key of one an error
            # rather than a silently wrong lookup.
            if self.cls.find("__eq__") is not None:
                raise TypeError(f"unhashable type: '{self.cls.name}'")
            return id(self) >> 3
        return int(out)

    def __getitem__(self, key):
        out = self._send("__getitem__", key)
        if out is NotImplemented:
            raise TypeError(f"'{self.cls.name}' object is not subscriptable")
        return out

    def __setitem__(self, key, value):
        out = self._send("__setitem__", key, value)
        if out is NotImplemented:
            raise TypeError(f"'{self.cls.name}' object does not support item "
                            f"assignment")

    def __contains__(self, needle):
        out = self._send("__contains__", needle)
        if out is NotImplemented:
            raise TypeError(f"argument of type '{self.cls.name}' is not "
                            f"a container or iterable")
        return bool(out)

    def __neg__(self):
        out = self._send("__neg__")
        if out is NotImplemented:
            raise TypeError(f"bad operand type for unary -: "
                            f"'{self.cls.name}'")
        return out

    def __pos__(self):
        out = self._send("__pos__")
        if out is NotImplemented:
            raise TypeError(f"bad operand type for unary +: "
                            f"'{self.cls.name}'")
        return out

    def __invert__(self):
        out = self._send("__invert__")
        if out is NotImplemented:
            raise TypeError(f"bad operand type for unary ~: "
                            f"'{self.cls.name}'")
        return out

    def __abs__(self):
        out = self._send("__abs__")
        if out is NotImplemented:
            raise TypeError(f"bad operand type for abs(): '{self.cls.name}'")
        return out

    def __index__(self):
        out = self._send("__index__")
        if out is NotImplemented:
            raise TypeError(f"'{self.cls.name}' object cannot be interpreted "
                            f"as an integer")
        return int(out)


class Func:
    """A callable: compiled code, its captured boxes, and maybe a receiver.

    Defaults and `*rest` live HERE and not at the call site, exactly as in the
    C: a call through `apy_call` reaches a function whose definition the caller
    never saw, and every method call is one of those.
    """

    __slots__ = ("annotate", "qualname", "module", "descr", "code", "arity",
                 "name", "cells",
                 "bound",
                 "defaults",
                 "vararg", "pnames", "kwarg", "kwonly", "posonly", "doc",
                 "dict", "coro", "is_type", "builtin", "nkwdefault")

    def __init__(self, code, arity, name, ncells, ndefaults=0,
                 vararg=False) -> None:
        self.code = code
        self.arity = arity
        self.name = name
        self.cells = [None] * ncells
        self.bound = None
        self.defaults = [None] * ndefaults
        self.vararg = vararg
        #: ARBITRARY ATTRIBUTES set on the function itself, or None until one
        #: is. Mirrors `v.fn.dict` in the C so both paths answer the same.
        self.dict = None
        #: WHETHER CALLING THIS BUILDS A COROUTINE -- an `async def`.
        self.coro = False
        #: WHETHER THIS IS A BUILTIN TYPE NAME used as a value -- `int`,
        #: `str`, `list`. It stays callable; the flag is what makes
        #: `print(int)` say `<class 'int'>`.
        self.is_type = False
        #: WHETHER the last declared parameter is `**kw`.
        self.kwarg = False
        #: How many of the TRAILING DEFAULTS are the keyword-only
        #: parameters'. Not derivable from `kwonly`: one of those may be
        #: REQUIRED, and `def f(a, b=1, *args, c)` has one keyword-only
        #: parameter and one default that is not its.
        self.nkwdefault = 0
        #: How many TRAILING declared parameters are keyword-only. A position
        #: cannot reach one, so positional filling stops short of them.
        self.kwonly = 0
        #: How many LEADING declared parameters are positional-only. Their
        #: names ARE recorded -- so a keyword call can be told which mistake
        #: it made -- but the matcher skips them.
        self.posonly = 0
        #: Parameter names in declaration order, including `self`. Only a
        #: KEYWORD argument through a value needs them -- see the C's
        #: `pnames`.
        self.pnames = [None] * arity
        #: The DOCSTRING, or None. Recorded because `f.__doc__` is the one
        #: piece of a `def` a program routinely reads back.
        self.doc = None
        #: PEP 649: the thunk that BUILDS `__annotations__`, or None. Lazy
        #: because an annotation may name something that does not exist yet.
        self.annotate = None
        #: PEP 3155: the QUALIFIED name -- `C.m` -- or None for the plain one.
        self.qualname = None
        #: WHERE THE `def` WAS WRITTEN -- `fractions` for a spliced one -- or
        #: None, which reads as `__main__`. Only a spliced `def` is told; see
        #: `_apy_func_module`.
        self.module = None
        #: WHETHER THIS IS A BUILTIN reached as a value -- `print`, `len`.
        self.builtin = False
        #: WHETHER THIS WAS REACHED OFF A TYPE rather than off a value.
        #: `list.append` is a `method_descriptor` in CPython and
        #: `list.__len__` a `wrapper_descriptor`; the qualname carries which
        #: type. See the C's `apy_func_descr`.
        self.descr = False

    def __eq__(self, other) -> bool:
        """A BOUND METHOD IS A FRESH OBJECT PER ACCESS -- `c.m is c.m` is
        False -- and two of them are EQUAL when they wrap the same function
        and the same receiver, which is what CPython compares.

        Only bound ones: two closures over the same `def` are distinct objects
        with distinct cells, and CPython calls those unequal. Without this the
        host answered identity and disagreed with the C, which compares the
        pair explicitly.
        """
        if self is other:
            return True
        if not isinstance(other, Func):
            return NotImplemented
        return (self.bound is not None and other.bound is not None
                and self.code == other.code and self.bound is other.bound)

    def __hash__(self) -> int:
        # Defined because `__eq__` is: without it a Func is unhashable, and
        # one goes into a dict or a set wherever a program keys on a method.
        # Bound methods that compare equal must hash equal, so the receiver's
        # identity is what it hashes on.
        return hash((self.code, id(self.bound) if self.bound is not None
                     else id(self)))

    def bind(self, receiver) -> "Func":
        out = Func(self.code, self.arity, self.name, 0)
        # SHARED, not copied: a bound method sees the same boxes and the same
        # default objects as the function it came from.
        out.cells = self.cells
        out.defaults = self.defaults
        out.vararg = self.vararg
        out.kwarg = self.kwarg
        out.kwonly = self.kwonly
        # WHERE THE POSITIONAL DEFAULTS END AND THE KEYWORD-ONLY ONES BEGIN.
        # Carried like every other shape of the signature: the C copies the
        # whole struct when it binds, so leaving this one out was the two
        # halves disagreeing about `def __init__(self, name, *, default=x)`
        # -- reachable the moment a bound method is what a call enters, which
        # is every method call there is.
        out.nkwdefault = self.nkwdefault
        out.posonly = self.posonly
        out.pnames = self.pnames
        out.doc = self.doc
        out.annotate = self.annotate
        out.qualname = self.qualname
        out.builtin = self.builtin
        # WHETHER CALLING IT BUILDS A COROUTINE, and the attributes set on
        # the function itself. SHARED like the cells and the defaults, not
        # copied: `C.m.tag = 1` is readable through `C().m` because they are
        # one dict, and writing through the bound one is meant to be seen by
        # the other.
        out.coro = self.coro
        out.dict = self.dict
        out.bound = receiver
        return out


#: "no receiver recorded" -- distinct from None, which is a real receiver.
_NO_OWNER = object()

#: The kinds a builtin method can be bound to, and the only ones CPython may
#: be asked about directly: an Instance or a Class here is the interpreter's
#: own object, and asking Python about one would answer about the wrong
#: thing entirely.
_KIND_SAMPLES = (str, bytes, bytearray, list, tuple, dict, set, frozenset,
                 int, float, range, complex,
                 # A VIEW TOO, for the same question: `type(m.release)` is a
                 # `builtin_function_or_method` in CPython and this is the
                 # oracle that says so.
                 memoryview)

#: THE SAME ORACLE, REACHED BY NAME. A descriptor carries its owner as the
#: head of its qualname -- `list` out of `list.append` -- where a bound
#: method carries a receiver, so this is how the owner becomes a type CPython
#: can be asked about. The C's twin walks `apy_kind_prototype`.
_KIND_BY_NAME = {one.__name__: one for one in _KIND_SAMPLES}


def _descr_of(v):
    """What CPython's own type answers for the name a DESCRIPTOR carries, or
    None when the owner is not a builtin kind.

    Asking CPython is exact where a table is a copy: whether `list.__len__`
    is a `wrapper_descriptor` or a `method_descriptor`, and what its repr
    says, are both read straight off the real thing.
    """
    owner = (getattr(v, "descr", "") if isinstance(v, Native)
             else (v.qualname or "").split(".")[0])
    kind = _KIND_BY_NAME.get(owner)
    return None if kind is None else getattr(kind, v.name, None)


class Native:
    """A callable the RUNTIME owns, standing for one of `object`'s defaults.

    Every callable here used to be compiled code, which meant a `super()`
    whose base chain had run out had nothing to hand back -- the default
    behaviours existed and no VALUE named them, so `super().__init__()` in a
    class with no explicit base was an AttributeError. This is the value.

    Interned per name by `_object_default`, so `super().__init__` reached
    twice is the same object, as any other attribute would be.
    """

    __slots__ = ("name", "body", "bound", "ranged", "owner", "variadic",
                 "descr")

    def __init__(self, name: str, body, ranged: bool = False,
                 owner=_NO_OWNER, variadic: bool = False,
                 descr: str = "") -> None:
        self.name = name
        #: THE TYPE THIS WAS REACHED OFF, by name, or "" for one reached off
        #: a value. `list.append` is a `method_descriptor` in CPython and not
        #: a function: it names the type in its repr, qualifies its own name
        #: with it, answers it to `__objclass__` and has no `__self__`. The
        #: compiled halves carry the same fact as a flag plus the qualname;
        #: see the C's `apy_func_descr`.
        self.descr = descr
        self.body = body
        self.bound = None
        #: THE RECEIVER, FOR A MESSAGE AND NOTHING ELSE. A builtin method
        #: built from the table holds its receiver in the BODY's closure, so
        #: `bound` is empty and `str.upper() takes no keyword arguments` had
        #: no owner to name. Recorded rather than derived, and read only
        #: where CPython writes the owner out.
        self.owner = owner
        #: Whether the method really takes a RANGE of argument counts.
        #: NOT DERIVABLE FROM THE BODY: nearly every lambda in `_kind_attr`
        #: carries a default (`lambda o, _w=want: ...`) as a CLOSURE CAPTURE,
        #: which looks exactly like an optional parameter and is not one.
        #: Only `__round__` and its like say so here, and the difference
        #: decides both what `type()` answers and how a wrong count is worded.
        self.ranged = ranged
        #: Whether it takes WHATEVER IT IS GIVEN, keywords included.
        #: `format` is the one builtin method shaped that way, and the
        #: keyword binder has to hand the names over rather than match them
        #: against a signature it has none of.
        self.variadic = variadic

    def bind(self, receiver) -> "Native":
        out = Native(self.name, self.body, self.ranged, self.owner,
                     self.variadic)
        out.bound = receiver
        return out

    def __call__(self, *args):
        return self.body(*args)



#: An EMPTY VALUE of the kind a builtin type names, so `_kind_attr` can
#: answer for the type without a second copy of it. Nothing is done with the
#: prototype but ask its kind.
_KIND_PROTOTYPES = {"list": [], "tuple": (), "dict": {}, "set": set(),
                    "frozenset": frozenset(), "str": "", "bytes": b"",
                    "int": 0, "bool": False, "float": 0.0}


def _kind_prototype(name: str):
    """An empty value of the kind `name` names, or None.

    WHAT A TYPE ANSWERS ATTRIBUTES FROM. `list.append` has no list to ask, so
    one is made; `type(iter([])).__next__` has no cursor, so one is made
    here. The C twin is `apy_kind_prototype`.
    """
    found = _KIND_PROTOTYPES.get(name)
    if found is not None:
        return found
    # A GENERATOR PROTOTYPE IS A GENERATOR WITH NO STEP, for the reason a
    # cursor prototype has no source: nothing is ever run, and what the type
    # carries is all that is asked of it.
    if name in ("generator", "coroutine", "async_generator",
                "coroutine_wrapper"):
        made = Gen(None, 0)
        made.coro = name != "generator"
        made.agen = name == "async_generator"
        # A WRAPPER PROTOTYPE WRAPS NOTHING, for the same reason: what the
        # type carries is all that is asked of it.
        made.wrapper = name == "coroutine_wrapper"
        return made
    mode = _CURSOR_PROTO_MODES.get(name)
    if mode is None:
        return None
    it = Iterator([], None, mode)
    it.named = name
    return it


def _unbound_kind(h, want: str):
    """`dict.keys` as a value: the receiver is its first argument."""
    def body(recv, *rest):
        found = _kind_attr(h, recv, want)
        if found is None:
            h._fail("TypeError", f"descriptor '{want}' needs an argument")
            raise _UserFailed
        return h._get(found, want).body(*rest)
    return body


#: The dunders `object` gives to every value, with their arities.
#:
#: WHAT MAKES `hasattr(x, "__eq__")` TRUE FOR EVERYTHING. Comparison is the
#: part programs read rather than call: `functools.total_ordering` asks a
#: class which orderings it already has, and `abc` asks structurally. The
#: same table is `runtime/slots.py`'s `apy_object_arity`, written for the
#: machine subset.
_OBJECT_ARITY = {"__eq__": 2, "__ne__": 2, "__lt__": 2, "__le__": 2,
                 "__gt__": 2, "__ge__": 2, "__str__": 1, "__repr__": 1,
                 "__format__": 2,
                 # AND THE TWELVE `object` HANDS DOWN THAT NOTHING
                 # OVERRIDES. `__reduce_ex__` is how pickle finds a value,
                 # `__dir__` is what `dir()` reads, and `__init__` and
                 # `__new__` are on everything there is. `_kind_attr` answers
                 # each of them; this is the same list as a table, for the
                 # runtime symbol that states it.
                 "__init__": 1, "__new__": 2, "__getattribute__": 2,
                 "__setattr__": 3, "__delattr__": 2, "__init_subclass__": 1,
                 "__subclasshook__": 2, "__dir__": 1, "__sizeof__": 1,
                 "__reduce__": 1, "__reduce_ex__": 2, "__getstate__": 1}

#: What every number has, what an int and a float share, and what only an
#: int carries. See `runtime/slots.py`'s `apy_number_arity`, which is the
#: same three sets. `__round__` IS NOT HERE because it is not a fixed
#: arity -- it takes an optional `ndigits`, and is answered separately.
_NUM_ANY = {"__add__": 2, "__radd__": 2, "__sub__": 2, "__rsub__": 2,
            "__mul__": 2, "__rmul__": 2, "__truediv__": 2, "__rtruediv__": 2,
            "__pow__": 2, "__rpow__": 2, "__neg__": 1, "__pos__": 1,
            "__abs__": 1, "__bool__": 1}
_NUM_REAL = {"__floordiv__": 2, "__rfloordiv__": 2, "__mod__": 2,
             "__rmod__": 2, "__divmod__": 2, "__rdivmod__": 2, "__int__": 1,
             "__float__": 1, "__trunc__": 1, "__floor__": 1, "__ceil__": 1}
_NUM_INT = {"__index__": 1, "__invert__": 1, "__and__": 2, "__rand__": 2,
            "__or__": 2, "__ror__": 2, "__xor__": 2, "__rxor__": 2,
            "__lshift__": 2, "__rlshift__": 2, "__rshift__": 2,
            "__rrshift__": 2}


def _number_arity(want: str, is_int: bool, is_complex: bool) -> int:
    """The arity of a number's dunder, or 0 where that kind has none."""
    if want in _NUM_ANY:
        return _NUM_ANY[want]
    if is_complex:
        return 1 if want == "__complex__" else 0
    if want in _NUM_REAL:
        return _NUM_REAL[want]
    return _NUM_INT.get(want, 0) if is_int else 0


def _apy_object_arity(h, a):
    """The arity of a dunder every object has, or 0 for anything else.

    A BINDING THAT ANSWERS RATHER THAN REFUSING, unlike `_apy_kind_attr`
    beside it: the compiled runtime needs the table because a builtin has no
    class dict to search, and the table is a fact about Python that the
    interpreter can state as readily.
    """
    return _OBJECT_ARITY.get(str(h._get(a[0], "apy_object_arity")), 0)


def _apy_number_arity(h, a):
    """The arity of a number's dunder, or 0. See `_apy_object_arity`."""
    return _number_arity(str(h._get(a[0], "apy_number_arity")),
                         bool(a[1]), bool(a[2]))


#: The six rich comparisons, which answer a sentinel rather than raising.
_RICH_COMPARISONS = ("__eq__", "__ne__", "__lt__", "__le__", "__gt__",
                     "__ge__")


#: The kinds whose comparison is CPython's own, because the interpreter
#: holds a real one of each.
_PLAIN = (int, float, complex, str, bytes, bytearray, list, tuple, set,
          frozenset, dict, range, type(None))


def _rich_compare(h, want: str, obj, other):
    """One rich comparison by name, with CPython's `NotImplemented` rule.

    THE SENTINEL IS THE POINT. `(1).__eq__("a")` is NotImplemented and not
    False: it is `==` above the method that turns a pair of them into False,
    and `<` that turns them into the TypeError a program sees. A method that
    answered False directly would break `functools.total_ordering`, which
    reads the sentinel to decide whether to try the reflected operation.

    CPYTHON'S OWN METHOD DECIDES where both sides are real Python values,
    which is the whole reason this path is short: the widening rule between
    the numbers, the one-way reach from a bytearray to bytes, and which kinds
    refuse an order are all already right there. `runtime/slots.py`'s
    `apy_compare_how` is the same table written out, for the runtimes that
    have no Python underneath.

    ANYTHING THIS INTERPRETER MADE GETS `object`'s: identity for `__eq__`,
    and the sentinel for everything else. An instance reaching here has no
    comparison of its own, since the class chain was searched first.
    """
    if isinstance(obj, _PLAIN) and isinstance(other, _PLAIN):
        return getattr(obj, want)(other)
    if want == "__eq__" and obj is other:
        return True
    if want == "__ne__" and obj is other:
        return False
    return NotImplemented


#: The ordinary builtin methods, and which symbol serves each arity. THE
#: FRONTEND'S OWN TABLE, read here rather than restated: `x.upper()` is
#: lowered through it and `getattr(x, "upper")()` is dispatched through it,
#: so the two spellings cannot drift into two implementations. The generated
#: C half reads the same table; see `objects/c/_gen_kindmeth.py`.
from uasm.frontends.python.methods import (  # noqa: E402
    DYN_METHOD_TABLE, METHOD_PARAMS, REQUIRED, KeywordError, _suggest,
    fold_ctor_keywords, method_symbol)
from uasm.objects.c.kindmeth_table import (  # noqa: E402
    KINDMETH_WORDS, KIND_DIR, KIND_DOC, CURSOR_SAMPLES,
    # THE TWO SAMPLES THAT ARE NOT EXPRESSIONS. The table names them
    # and the `eval` below needs them in scope; see the generated
    # file, which carries both.
    _sample_coroutine, _sample_coroutine_wrapper, _sample_async_generator)

#: EVERY CURSOR TYPE JOINS THE ORACLE, through the same sample expressions
#: the generated tables were built from. `type(iter([])).__next__` is a
#: descriptor off `list_iterator`, and only the real type can say whether
#: that is a `wrapper_descriptor` or a `method_descriptor` and what its repr
#: reads. Read from the generated module rather than written again here, so
#: all three arrangements ask CPython about the same twenty-two types.
_KIND_BY_NAME.update(
    (name, type(eval(expr)))            # noqa: S307 -- a generated literal
    for name, expr in CURSOR_SAMPLES.items())

#: What kind a value COUNTS AS when a builtin method is looked up on it.
#: `True` is an `int` here, exactly as `apy_kind_bit` folds the bool kind
#: into the int one -- `True.bit_count()` IS int's method, and CPython says
#: `int.bit_count() takes no arguments` when it is handed one.
_METH_KIND = {bool: "int"}


def _meth_kind(v) -> str:
    return _METH_KIND.get(type(v), type(v).__name__)


#: WHAT `_fn_default` ANSWERS FOR A PARAMETER WITH NO DEFAULT. A distinct
#: object and not `None`, because `None` is the commonest default there is.
_NO_DEFAULT = object()


def _fn_default(f, idx: int):
    """The default the `def` evaluated for declared parameter `idx`, or
    `_NO_DEFAULT` where it has none.

    NOT SIMPLY THE TRAILING ONES. `defaults` holds the POSITIONAL defaults
    first and the KEYWORD-ONLY defaults after them, and a keyword-only
    parameter WITHOUT a default may follow a positional one WITH: `def f(a,
    b=1, *, k)` records one default, for `b`, while `k` has none. Indexing
    from the end of the whole declaration looked for `b`'s default two places
    before the start of the list, found nothing, and reported `b` as a hole
    -- so `f(1, k=2)` refused a call CPython answers.

    `nkwdefault` is on the function object for exactly this: it says where
    the first run ends and the second begins.
    """
    declared = f.arity - (1 if f.vararg else 0) - (1 if f.kwarg else 0)
    bypos = declared - f.kwonly
    nkw = max(0, f.nkwdefault)
    npos = max(0, len(f.defaults) - nkw)
    if idx < 0:
        return _NO_DEFAULT
    if idx < bypos:
        # The positional defaults cover the LAST `npos` positional
        # parameters, and sit at the FRONT of the list.
        d = idx - (bypos - npos)
        return f.defaults[d] if 0 <= d < npos else _NO_DEFAULT
    # And the keyword-only defaults cover the last `nkw` of the keyword-only
    # tail, sitting behind the positional ones.
    if idx < declared - nkw or idx >= declared:
        return _NO_DEFAULT
    d = npos + (idx - (declared - nkw))
    return f.defaults[d] if 0 <= d < len(f.defaults) else _NO_DEFAULT


def _name_list(names) -> str:
    """CPython's list of parameter names: `'a'`, `'a' and 'b'`, then
    `'a', 'b', and 'c'`. The Oxford comma appears only from three on, which
    is why the separator cannot be chosen from the position alone.

    TAKES THE NAMES AND NOT A RANGE OF THEM, because the holes a keyword call
    leaves need not be adjacent: `f(b=2)` against `def f(a, b, c)` is missing
    `'a' and 'c'`."""
    got = [f"'{n or '?'}'" for n in names]
    if len(got) < 3:
        return " and ".join(got)
    return ", ".join(got[:-1]) + ", and " + got[-1]


def _fn_arity_failure(h, f, given: int, kwo: int) -> None:
    """Word the arity of a function the program wrote as CPython words it.

    Four messages, and they are one family: which end the count fell off, and
    whether a default makes the low end a RANGE rather than a number.

        f() takes 2 positional arguments but 3 were given
        f() takes from 1 to 2 positional arguments but 3 were given
        f() missing 1 required positional argument: 'b'
        f() missing 2 required keyword-only arguments: 'k' and 'j'

    `given` COUNTS THE RECEIVER, because CPython does: `C().m(1, 2)` is
    `C.m() takes 2 positional arguments but 3 were given`, self included at
    both ends. The caller adds it, because only the caller knows whether the
    receiver is on the function (a bound method) or still ahead of it (a
    class being instantiated).

    `kwo` is how many keywords landed on KEYWORD-ONLY parameters, which
    CPython names in the surplus message and nowhere else.

    NAMED BY ITS QUALNAME -- `C.m()`, `outer.<locals>.inner()` -- with the
    plain name for a function that never got one. Fails through `h._fail`
    and leaves the raise to the caller, the same shape `_meth_positional`
    has.
    """
    who = f.qualname or f.name
    declared = f.arity - (1 if f.vararg else 0) - (1 if f.kwarg else 0)
    bypos = max(0, declared - f.kwonly)
    # ONLY A POSITIONAL PARAMETER'S DEFAULT widens the low end. The
    # keyword-only defaults sit past `bypos`, and counting those said
    # `takes from 0 to 1 positional arguments` for `def f(a, *, k=0)`.
    ndef = max(0, len(f.defaults) - f.nkwdefault)
    least = max(0, bypos - ndef)

    if given > bypos and not f.vararg:
        if least == bypos:
            lead = (f"{who}() takes {bypos} positional "
                    f"argument{'' if bypos == 1 else 's'}")
        else:
            lead = (f"{who}() takes from {least} to {bypos} "
                    f"positional arguments")
        # A KEYWORD-ONLY ARGUMENT IS COUNTED SEPARATELY OR NOT AT ALL --
        # CPython switches the whole tail of the sentence on whether one was
        # passed, rather than adding a clause to it.
        if kwo > 0:
            h._fail("TypeError",
                    f"{lead} but {given} positional "
                    f"argument{'' if given == 1 else 's'} (and {kwo} "
                    f"keyword-only argument{'' if kwo == 1 else 's'}) "
                    f"were given")
        else:
            h._fail("TypeError", f"{lead} but {given} "
                                 f"{'was' if given == 1 else 'were'} given")
        return
    # TOO FEW IS SAID IN NAMES, not in a count of what the function takes.
    # The names still wanted start where the arguments ran out -- and that
    # index already skips a receiver, which is why `given` counts it.
    if given < least and f.pnames and least <= len(f.pnames):
        n = least - given
        h._fail("TypeError",
                f"{who}() missing {n} required positional "
                f"argument{'' if n == 1 else 's'}: "
                f"{_name_list(f.pnames[given:least])}")
        return
    # A REQUIRED KEYWORD-ONLY PARAMETER IS MISSED BY NAME and never by count:
    # no position could have reached it, so a message about how many
    # positional arguments the function takes sends the reader to the wrong
    # half of the signature. They are the parameters past `bypos` that the
    # trailing keyword defaults do not cover.
    end = declared - f.nkwdefault
    if f.kwonly and f.pnames and bypos < end <= len(f.pnames):
        n = end - bypos
        h._fail("TypeError",
                f"{who}() missing {n} required keyword-only "
                f"argument{'' if n == 1 else 's'}: "
                f"{_name_list(f.pnames[bypos:end])}")
        return
    # NOTHING THE FAMILY ABOVE COVERS -- a function whose parameter names the
    # frontend never recorded, or a count wrong in some way the signature does
    # not explain. The plainer wording, with the receiver taken back out of
    # the number because that wording never counted it.
    got = given - (1 if f.bound is not None else 0)
    h._fail("TypeError",
            f"{f.name}() takes {bypos} positional argument"
            f"{'' if bypos == 1 else 's'} but {got} "
            f"{'was' if got == 1 else 'were'} given")


def _meth_positional(h, f, args) -> bool:
    """Refuse this many POSITIONAL arguments to a bound builtin method, for
    the two whose positional bound is narrower than their total.

    WHERE THE POSITIONAL COUNT IS STILL KNOWN, which is only before the
    keywords are folded into slots: a call three slots wide may have been
    written with three positionals -- a TypeError -- or with two names, which
    is not one. The C twin is `apy_meth_positional`.
    """
    # THE OWNER IS THE RECEIVER when the native was made unbound, which is
    # what `_made_table_method` hands out: the receiver travels in `owner`
    # and the body closes over it.
    recv = (f.bound if f.bound is not None
            else f.owner if f.owner is not _NO_OWNER else None)
    if recv is None:
        return False
    packed = KINDMETH_WORDS.get((f.name, _meth_kind(recv)))
    if not packed:
        return False
    family = (packed >> 16) & 15
    # FAMILY 4 IS `takes at most N positional arguments` and family 9 is
    # `takes no positional arguments`, whose N is nought. Nothing else in
    # the table counts positions.
    if family == 4:
        allowed = (packed >> 20) & 15
    elif family == 9:
        allowed = 0
    else:
        return False
    if len(args) <= allowed:
        return False
    _meth_arity_words(h, recv, f.name, packed, len(args))
    return True


def _meth_arity_words(h, recv, want: str, packed: int, got: int):
    """CPython's own refusal for the wrong number of arguments to a builtin
    method, worded for THIS receiver.

    THE C TWIN IS `apy_meth_arity_words`, and the packed word both read is
    generated by asking CPython -- see `_gen_kindmeth.py`. Nine shapes,
    because CPython really does have nine and they disagree about every
    visible thing: whether the type qualifies the name, whether there are
    parentheses, and whether the count is `exactly one` or `at most 3`.

    RECORDS THE FAILURE AND ANSWERS 0, which is what a `_TABLE` binding
    returns; a caller inside a native's body raises `_UserFailed` on top.
    """
    least = packed & 15
    if got < least:
        fam, n = (packed >> 8) & 15, (packed >> 12) & 15
    else:
        fam, n = (packed >> 16) & 15, (packed >> 20) & 15
        # A SECOND UPPER WORDING, past a second bound. `(5).to_bytes` says
        # `takes at most 2 positional arguments` for a third POSITIONAL and
        # `takes at most 3 arguments` for a fourth argument -- its `signed`
        # is keyword-only, so the two counts differ and so do the words.
        over = (packed >> 25) & 15
        if over and got >= over:
            fam, n = (packed >> 29) & 15, (packed >> 33) & 15
    plural = "" if n == 1 else "s"
    kind = _meth_kind(recv)
    if fam == 1:
        said = f"{kind}.{want}() takes exactly one argument ({got} given)"
    elif fam == 2:
        said = f"{kind}.{want}() takes no arguments ({got} given)"
    elif fam == 3:
        said = f"{want}() takes at most {n} argument{plural} ({got} given)"
    elif fam == 4:
        said = (f"{want}() takes at most {n} positional "
                f"argument{plural} ({got} given)")
    elif fam == 5:
        said = (f"{want}() takes at least {n} positional "
                f"argument{plural} ({got} given)")
    elif fam == 6:
        said = f"{want} expected at least {n} argument{plural}, got {got}"
    elif fam == 7:
        said = f"{want} expected at most {n} argument{plural}, got {got}"
    elif fam == 9:
        # NO COUNT AT ALL, which is what a signature with nothing but
        # keyword-only parameters says: `[].sort(None)`.
        said = f"{want}() takes no positional arguments"
    else:
        said = f"{want} expected {n} argument{plural}, got {got}"
    return h._fail("TypeError", said)

#: The names that table answers -- the dunders are left out, because every
#: one of them is a PROTOCOL name the arms above answer with their own
#: gating, and a table keyed by name alone cannot make those distinctions.
_TABLE_METHODS = frozenset(n for n in DYN_METHOD_TABLE
                           if not n.startswith("__"))


def _table_shape(h, name: str, held, args):
    """The handles one runtime entry point takes, which is not always the
    receiver followed by the arguments.

    SEVEN SHAPES ARE IRREGULAR and every one is mirrored from the frontend's
    `_dyn_builtin_method` and from the generated C: the two spellings have to
    reach the same call.
    """
    argc = len(args)
    if name in ("keys", "values", "items"):
        return [held, {"keys": 0, "values": 1, "items": 2}[name]]
    if name == "pop" and argc < 2:
        return [held, args[0] if args else h._none, 1 if args else 0]
    if name in ("encode", "decode"):
        # THE DEFAULTS THEMSELVES, NOT None -- see `_dyn_builtin_method`,
        # which pads the written spelling the same way, and `_codec_given`,
        # which refuses a None once a slot really can only hold one the
        # program wrote.
        pad = [h._new("utf-8"), h._new("strict")]
        return [held] + [args[i] if i < argc else pad[i] for i in range(2)]
    if name in ("hex", "expandtabs") and argc == 0:
        # `hex` WITH NOTHING has an EMPTY separator slot, which is how the
        # no-argument form is told from `b.hex(None)` -- see `_apy_hex_of`.
        return [held, h._int(8) if name == "expandtabs" else 0]
    if name in ("get", "setdefault") and argc == 1:
        return [held, args[0], h._none]
    if name == "sort":
        # PADDED FROM THE DEFAULTS, so the no-argument form and the folded
        # two-keyword form reach one entry point.
        pad = [h._none, h._new(False)]
        return [held] + [args[i] if i < argc else pad[i] for i in range(2)]
    if name == "update" and argc == 0:
        return [held, h._new({})]
    if name == "to_bytes":
        pad = [h._int(1), h._new("big"), h._new(False)]
        return [held] + [args[i] if i < argc else pad[i] for i in range(3)]
    return [held, *args]


#: The six set methods that take ANY NUMBER of other sets. `s.union()` is a
#: copy, `s.union(a, b)` folds both in, and no row in an arity table can say
#: "any count" -- so each was declared as taking exactly one and refused
#: every other call CPython answers. The compiled twin is `apy_set_fold`.
_SET_FOLDS = frozenset({"union", "intersection", "difference", "update",
                        "intersection_update", "difference_update"})


def _made_set_fold(h, obj, want: str):
    """One of the six, as a variadic bound callable value."""
    def body(*args, _w=want, _o=obj, **named):
        if named:
            # NONE OF THE SIX TAKES A KEYWORD, and it is named by its owner:
            # CPython says `set.union() takes no keyword arguments`.
            h._fail("TypeError",
                    f"{h.kind_name(_o)}.{_w}() takes no keyword arguments")
            raise _UserFailed
        # EACH ARGUMENT WALKED THROUGH THE FUNNEL FIRST. These six take any
        # iterable, and a class of this file's is not a Python one however
        # well a `for` loop walks it -- so `s.union(obj)` reported that a
        # class whose `__iter__` says exactly how was not iterable.
        walked = list(args)
        for at, one in enumerate(walked):
            if isinstance(one, (Instance, Class, Gen, Iterator)):
                got = _seq_items(h, one, _w)
                if got is None:
                    raise _UserFailed
                walked[at] = got
        args = walked
        try:
            return getattr(_o, _w)(*args)
        except TypeError as exc:
            h._fail("TypeError", str(exc))
            raise _UserFailed
    return h._new(Native(want, body, owner=obj, variadic=True))


def _made_table_method(h, obj, want: str):
    """One ordinary builtin method, as a bound callable value."""
    # THE SIX SET METHODS TAKE WHATEVER THEY ARE GIVEN -- see `_SET_FOLDS`.
    if want in _SET_FOLDS and isinstance(obj, (set, frozenset)):
        return _made_set_fold(h, obj, want)
    held = h._new(obj)
    live = [i for i, sym in enumerate(DYN_METHOD_TABLE[want]) if sym]
    lo, hi = (0, 3) if want == "to_bytes" else (min(live), max(live))

    def body(*args, _w=want, _h=held, _o=obj):
        sym = method_symbol(_w, len(args))
        if _w == "to_bytes":
            sym = DYN_METHOD_TABLE[_w][3]
        # THE RANGE IS THE RECEIVER'S, not the union over every kind that
        # has the name. `str.count` takes three arguments and `list.count`
        # takes one, and a union said three for both -- so `[1].count(1, 2)`
        # went through to a symbol that could not serve it.
        packed = KINDMETH_WORDS.get((_w, _meth_kind(_o)))
        low, high = (packed & 15, (packed >> 4) & 15) if packed else (lo, hi)
        # A SECOND BOUND MEANS THE ACCEPTED RANGE IS THE WIDER ONE.
        # `to_bytes` takes at most TWO positionals and THREE arguments, and
        # `sort` takes none and two, because each has a keyword-only
        # parameter -- and the keywords are folded into slots before the body
        # sees them. The POSITIONAL bound is checked where the positional
        # count is still known; see `_meth_positional`.
        if packed and (packed >> 25) & 15:
            high = ((packed >> 25) & 15) - 1
        got = len(args)
        if got < low or got > high:
            if packed:
                _meth_arity_words(h, _o, _w, packed, got)
                raise _UserFailed
            # A NAME THE GENERATED TABLE DOES NOT KNOW keeps the plainer
            # wording `apy_arity_error` falls back to.
            if lo == hi:
                how = f"expected {hi} argument{'' if hi == 1 else 's'}"
            elif got < lo:
                how = (f"expected at least {lo} "
                       f"argument{'' if lo == 1 else 's'}")
            else:
                how = (f"expected at most {hi} "
                       f"argument{'' if hi == 1 else 's'}")
            h._fail("TypeError", f"{_w} {how}, got {got}")
            raise _UserFailed
        if sym is None:
            h._fail("TypeError",
                    f"{_w} expected {hi} argument{'' if hi == 1 else 's'}, "
                    f"got {got}")
            raise _UserFailed
        # A NATIVE'S BODY IS HANDED VALUES and a runtime symbol takes
        # HANDLES, which is the whole of the conversion here.
        shaped = _table_shape(h, _w, _h, [h._new(v) for v in args])
        got = _TABLE[sym](h, shaped)
        if not got:
            raise _UserFailed
        return h._get(got, _w)

    return h._new(Native(want, body, ranged=lo != hi, owner=obj))


#: How big one runtime cell is, in bytes -- `sizeof(apy_obj)` in the C, which
#: is what `apy_sizeof` counts from. `__sizeof__` IS AN IMPLEMENTATION
#: NUMBER: CPython's own answer differs between builds, and what a program can
#: rely on is that it is an int and that it grows with what the value holds.
#: Written here so the three runtimes agree with EACH OTHER.
_CELL_BYTES = 160


def _cell_bytes(v) -> int:
    """`x.__sizeof__()` -- the same arithmetic `apy_sizeof` does in the C."""
    if isinstance(v, (str, bytes, bytearray)):
        return _CELL_BYTES + len(v) + 1
    if isinstance(v, (list, tuple, set, frozenset)):
        return _CELL_BYTES + len(v) * 8
    if isinstance(v, dict):
        return _CELL_BYTES + len(v) * 16
    if isinstance(v, int) and not isinstance(v, bool):
        # A BIG INT IS LIMBS, and a small one lives in the cell itself.
        if v.bit_length() > 63:
            return _CELL_BYTES + ((v.bit_length() + 63) // 64) * 8
    return _CELL_BYTES


def _cursor_left(it) -> int:
    """How many steps a cursor has left, for `__length_hint__`.

    AN ESTIMATE AND NOT A PROMISE, which is what the hint is for: the source
    may grow or shrink under the walk, and CPython's own answer goes stale
    the same way. What it must never be is negative -- `operator.length_hint`
    raises ValueError on one.

    A REVERSED CURSOR COUNTS DOWN from where `reversed` started it, and -1 is
    its exhaustion, so what it has left is the position plus one.
    """
    if it.mode == Iterator.REV:
        return max(0, it.i + 1)
    if not isinstance(it.src, (list, tuple, str, bytes, bytearray, dict, set,
                               frozenset, range)):
        return 0
    return max(0, len(it.src) - it.i)


def _cursor_setstate(h, it, where):
    """`it.__setstate__(i)` -- WHERE THE WALK IS, written rather than read.
    The other half of `__reduce__`: pickle remakes a partly consumed
    iterator and then says how far it had got.

    THE CLAMPING IS CPYTHON'S, measured rather than derived. A forward
    cursor takes 0..len and anything outside that -- above OR BELOW -- leaves
    it exhausted, so `iter([1,2,3]).__setstate__(-1)` yields nothing. A
    reversed one counts down from `i` and -1 is already its exhaustion, so a
    position past the end is pulled back to the last element and a negative
    one stays the end of the walk.

    `zip` AND `map` KEEP NO POSITION OF THEIR OWN -- what CPython pickles for
    them is their sub-iterators -- so theirs takes the argument and changes
    nothing, which is what the same call does there. The C twin is
    `apy_cursor_setstate`.
    """
    if isinstance(where, bool) or not isinstance(where, int):
        return h._fail("TypeError", "an integer is required")
    if it.mode in (Iterator.MAP, Iterator.ZIP):
        return h._none
    n = (len(it.src)
         if isinstance(it.src, (list, tuple, str, bytes, bytearray, dict,
                                set, frozenset, range))
         else 0)
    if it.mode == Iterator.REV:
        at = min(where, n - 1)
        it.i = -1 if at < 0 else at
        return h._none
    it.i = n if (where < 0 or where > n) else where
    return h._none


def _unwrap(h, got):
    """A handle an `_apy_*` entry point answered, as a value -- or the
    failure it left pending, re-raised so the caller sees it."""
    if got == 0:
        raise _UserFailed
    return h._get(got, "__getattribute__")


def _kind_attr(h, obj, want: str):
    """A BUILTIN'S PROTOCOL METHODS, AS VALUES.

    `[].append` and `{}.keys` are lowered at the call site by the frontend,
    which means they exist as CALLS and never as attributes -- so
    `hasattr([1], "__iter__")` answered False for the most iterable object in
    the language, and every structural type test written against
    `collections.abc` said no.

    This does not make the whole method table reachable by name; it makes the
    PROTOCOL reachable, which is the part a program asks about rather than
    calls. Answers None for a name the kind does not have -- that is what
    keeps `hasattr` honest -- and the None VALUE where CPython has the
    attribute set to None, which is how a mutable container says it cannot be
    hashed.
    """
    # A RANGE IS INDEXABLE AND WALKABLE but is not a list: `+` and `*` do
    # not apply to one, so it is its own case rather than part of `seq`.
    rng = isinstance(obj, range)
    seq = isinstance(obj, (list, tuple))
    text = isinstance(obj, (str, bytes, bytearray))
    dict_ = isinstance(obj, dict)
    set_ = isinstance(obj, (set, frozenset))
    walks = seq or text or dict_ or set_ or rng         or isinstance(obj, _VIEW_TYPES)
    mutable = isinstance(obj, (list, dict, set, bytearray))

    def made(name, body):
        # THE RECEIVER TRAVELS WITH IT, for the one message that names the
        # owner: `list.append() takes no keyword arguments`. The body holds
        # it in a closure, so nothing else could say what it was.
        return h._new(Native(name, body, owner=obj))

    num = isinstance(obj, (int, float, complex))
    is_int = isinstance(obj, int)
    is_complex = isinstance(obj, complex)

    # `x.__class__` IS `type(x)`, for every value there is -- and it was
    # missing from all of them. `obj.__class__.__name__` is an everyday
    # idiom, and a program reaching for it got an AttributeError about the one
    # attribute Python guarantees. NOT A METHOD but the type object itself.
    if want == "__class__":
        return _apy_type_object(h, [h._new(obj)])
    # PEP 257 FOR THE BUILTINS. `"".__doc__` IS `str.__doc__` -- the text
    # `help` prints and every docstring tool reads -- and it was an
    # AttributeError on every builtin value there is. The COMPILED halves
    # read a generated table for it, because the text is CPython's and not a
    # fact this compiler could derive; here CPython is in the room.
    if want == "__doc__" and h.kind_name(obj) in KIND_DOC:
        # None FOR A KIND WHOSE TYPE HAS NO DOCSTRING, and not a refusal:
        # `iter([]).__doc__` is None in CPython and `__doc__` is on the list
        # `dir()` gives, so refusing it would be a list that lies. The table
        # holds the None, which is what tells it from a kind nothing knows.
        return h._value(KIND_DOC[h.kind_name(obj)])

    if want == "__hash__":
        # THE ATTRIBUTE EXISTS EITHER WAY. `[].__hash__ is None` is how a
        # program asks whether a list can be a dict key, and answering "no
        # such attribute" is a different claim from the one CPython makes.
        return h._none if mutable else made("__hash__", lambda: hash(obj))
    # `object` GIVES THESE TO EVERYTHING, which is why they are gated on no
    # kind at all: `hasattr(x, "__eq__")` is True for every value in Python.
    if want in ("__str__", "__repr__"):
        # THE INTERPRETER'S OWN TEXT and not CPython's `repr`, because a
        # container here holds the interpreter's values and Python would
        # print them its way.
        return made(want, lambda _w=want: h._text(obj, _w == "__repr__"))
    if want == "__format__":
        # THE SPEC MINI-LANGUAGE IS `_apy_format`'s, not a second copy of it
        # -- and its answer is a handle, which this body must hand back
        # unwrapped because the caller wraps what it gets.
        return made("__format__", lambda spec: h._get(
            _apy_format(h, [h._new(obj), h._new(spec)]), "__format__"))
    if want in _RICH_COMPARISONS:
        return made(want, lambda o, _w=want: _rich_compare(h, _w, obj, o))
    # AND THE TWELVE `object` HANDS DOWN THAT NOTHING OVERRIDES. Every one
    # was missing from every builtin value, which is 144 attributes a program
    # may ask for and Python guarantees: `__reduce_ex__` is how pickle finds
    # a value, `__dir__` is what `dir()` reads, and `__init__` and `__new__`
    # are on everything there is. THE SAME TWELVE the compiled halves answer
    # in `apy_object_arity`, with the same bodies.
    if want in ("__init__", "__init_subclass__"):
        return made(want, lambda *a: None)
    if want == "__getstate__":
        # A BUILTIN HAS NO `__dict__`, and `object.__getstate__` answers None
        # for anything that carries no state of its own.
        return made("__getstate__", lambda: None)
    if want == "__subclasshook__":
        # `object`'s ANSWER IS "I HAVE NO OPINION", which is what lets `abc`
        # fall through to its own structural test.
        return made("__subclasshook__", lambda c: NotImplemented)
    if want == "__dir__":
        return made("__dir__",
                    lambda: h._get(_apy_dir(h, [h._new(obj)]), "__dir__"))
    if want == "__sizeof__":
        return made("__sizeof__", lambda: _cell_bytes(obj))
    if want == "__new__":
        # `[].__new__(list)` IS AN EMPTY LIST: an implicit staticmethod, so
        # the receiver is ignored and the CLASS decides what is built.
        return made("__new__", lambda cls, *rest: h._invoke(cls, []))
    if want == "__getattribute__":
        return made("__getattribute__", lambda n: _unwrap(
            h, _apy_getattr(h, [h._new(obj), h._new(n)])))
    if want in ("__setattr__", "__delattr__"):
        # A BUILTIN VALUE HAS NO `__dict__` to write into, which is the whole
        # of what CPython says here.
        def _no_dict(n, *rest, _o=obj):
            h._fail("AttributeError",
                    f"'{h.kind_name(_o)}' object has no attribute "
                    f"'{n}' and no __dict__ for setting new attributes")
            raise _UserFailed
        return made(want, _no_dict)
    if want in ("__reduce__", "__reduce_ex__"):
        # PICKLING IS NOT SERVED HERE, and CPython refuses most of these in
        # exactly these words. The four it answers -- bytearray, set,
        # frozenset and range -- build a tuple through `copyreg`, which this
        # runtime has no counterpart for.
        def _no_pickle(*rest, _o=obj):
            h._fail("TypeError",
                    f"cannot pickle '{h.kind_name(_o)}' object")
            raise _UserFailed
        return made(want, _no_pickle)
    if want == "__len__" and walks:
        return made("__len__", lambda: len(obj))
    # THROUGH `_unwrap`, like `__getattribute__` above and for the same
    # reason: an `_apy_*` entry point answers a HANDLE, and handing that back
    # as the value made `it.__next__()` answer `4294967317` where CPython
    # answers `1` -- a silent wrong answer, and one only the interpreter gave,
    # so the three arrangements disagreed with each other. It also made
    # `it.__iter__() is it` False, because an int is not the cursor.
    #
    # ONLY THE DUNDER REACHED AS AN ATTRIBUTE was wrong; `next(it)` and a
    # `for` loop never come this way, which is why it lasted. A program sees
    # it the moment it drives an iterator by hand.
    if want == "__iter__" and (walks or isinstance(obj, (Gen, Iterator))):
        return made("__iter__",
                    lambda: _unwrap(h, _apy_iter(h, [h._new(obj)])))
    if want == "__next__" and isinstance(obj, (Gen, Iterator)):
        return made("__next__",
                    lambda: _unwrap(h, _apy_next(h, [h._new(obj), 0, 0])))
    # ONLY THE CURSORS THAT WALK A SIZED SOURCE have a length hint, which is
    # CPython's line: a list, tuple, str, bytes, range, dict, set or reversed
    # iterator carries one, and `map`, `filter`, `enumerate`, `zip` and a
    # `callable_iterator` do not. Those are the LAZY modes, and what they
    # have left is not a question their source can answer.
    # AND A `memory_iterator` HAS NONE, which is the one sized walk without
    # one: `iter(mv)` in CPython carries `__iter__` and `__next__` and
    # nothing else, where `reversed(mv)` -- a plain `reversed` -- does have
    # the hint.
    if want == "__length_hint__" and isinstance(obj, Iterator) \
            and obj.mode in (Iterator.PLAIN, Iterator.REV) \
            and obj.named != "memory_iterator":
        return made("__length_hint__", lambda: _cursor_left(obj))
    # `it.__setstate__(i)` AND `enumerate[int]`. WHICH CURSORS CARRY EACH is
    # read straight out of the generated `dir()` table rather than restated:
    # the list is CPython's own, and a name this answers that the list omits
    # -- or the other way about -- is the one mistake nobody would see. The
    # compiled halves draw the same line by mode and source kind; see
    # `apy_cursor_setstate_p`.
    if isinstance(obj, Iterator) and want in ("__setstate__",
                                              "__class_getitem__"):
        if want in KIND_DIR.get(h.kind_name(obj), ()):
            if want == "__setstate__":
                return made("__setstate__",
                            lambda i: _unwrap(h, _cursor_setstate(h, obj, i)))
            return made("__class_getitem__", lambda k: Alias(
                h._get(_apy_type_object(h, [h._new(obj)]),
                       "__class_getitem__"),
                k if isinstance(k, tuple) else (k,)))
    if want == "__contains__" and walks:
        return made("__contains__", lambda x: x in obj)
    if want == "__getitem__" and (seq or text or dict_):
        return made("__getitem__", lambda i: obj[i])
    if want == "__setitem__" and (isinstance(obj, (list, dict, bytearray))):
        return made("__setitem__", lambda i, v: obj.__setitem__(i, v))
    # WHATEVER `del x[k]` WOULD REACH -- exactly the kinds that take a
    # `__setitem__`, because a container that cannot be written cannot have a
    # piece taken out of it either.
    if want == "__delitem__" and isinstance(obj, (list, dict, bytearray)):
        return made("__delitem__", lambda i: obj.__delitem__(i))
    # `reversed(x)` WALKS ANYTHING INDEXABLE, but only three kinds carry the
    # method that names it: a str or a tuple is reversed through `__len__`
    # and `__getitem__`, and CPython gives neither a `__reversed__`.
    if want == "__reversed__" and isinstance(obj, (list, dict, range)):
        return made("__reversed__", lambda: list(reversed(obj)))
    # CONCATENATION AND REPETITION, which a sequence has and a set, a dict
    # and a range do not -- `range(3) * 2` is a TypeError in Python and the
    # attribute is absent, not a method that refuses.
    if (seq or text) and want in ("__add__", "__mul__", "__rmul__"):
        return made(want, lambda o, _w=want: getattr(obj, _w)(o))
    # `__round__` TAKES AN OPTIONAL `ndigits`, which the arity table cannot
    # say -- `(1.55).__round__()` and `(1.55).__round__(1)` are one method.
    # A COMPLEX HAS NONE: there is no rounding of one.
    if num and not is_complex and want == "__round__":
        return h._new(Native("__round__",
                             lambda nd=None: (obj.__round__() if nd is None
                                              else obj.__round__(nd)),
                             ranged=True, owner=obj))
    if num and _number_arity(want, is_int, is_complex):
        return made(want, lambda *r, _w=want: getattr(obj, _w)(*r))
    if rng and want == "__bool__":
        return made("__bool__", lambda: bool(obj))
    # AND None SAYS SO TOO, which is the only other kind here that writes the
    # method out rather than being read through `__len__`.
    if obj is None and want == "__bool__":
        return made("__bool__", lambda: False)
    # THE IN-PLACE OPERATORS, which belong to the MUTABLE kinds and to no
    # other: `(1,).__iadd__` is an AttributeError in Python and `[1].__iadd__`
    # is the method that makes `xs += ys` change the list every other name for
    # it also sees. A FROZENSET HAS NONE of them, which is why the set arm
    # asks for `set` rather than `set_`.
    if isinstance(obj, (list, bytearray)) and want in ("__iadd__",
                                                       "__imul__"):
        return made(want, lambda o, _w=want: getattr(obj, _w)(o))
    if isinstance(obj, (dict, set)) and want == "__ior__":
        return made("__ior__", lambda o: obj.__ior__(o))
    if isinstance(obj, set) and want in ("__iand__", "__isub__", "__ixor__"):
        return made(want, lambda o, _w=want: getattr(obj, _w)(o))
    # `%` ON TEXT IS FORMATTING, not arithmetic -- which is why it belongs to
    # str and bytes and to no other sequence.
    if isinstance(obj, (str, bytes, bytearray)) and want == "__mod__":
        return made("__mod__", lambda o: obj.__mod__(o))
    # `d | e` MERGES TWO DICTS, and the same four spellings are a set's
    # operations.
    if dict_ and want in ("__or__", "__ror__"):
        return made(want, lambda o, _w=want: getattr(obj, _w)(o))
    if set_ and want in ("__or__", "__and__", "__sub__", "__xor__",
                         "__ror__", "__rand__", "__rsub__", "__rxor__"):
        return made(want, lambda o, _w=want: getattr(obj, _w)(o))
    # `index`, `count`, `append`, `insert`, `add`, `discard` and
    # `isdisjoint` ARE ORDINARY TABLE METHODS and are answered below, by
    # `_made_table_method`, which reads each receiver's real bounds out of
    # the generated table. They were arms of their own here with a
    # fixed-arity lambda apiece, which refused `"ab".count("a", 0, 2)` --
    # a call CPython answers -- and named no kind when it did.
    # PEP 688: whatever can be handed to `memoryview` HAS `__buffer__`. It is
    # a protocol a program asks about far more often than it calls, and
    # answering False for `bytes` said this runtime has no buffers at all.
    # THE STATICS A TYPE CARRIES AND A VALUE CARRIES TOO. `bytes.fromhex`
    # and `b"".fromhex` are ONE method in Python -- an implicit staticmethod,
    # so the receiver decides which body and is otherwise ignored -- and
    # every one of these was reachable only as a written call on the type's
    # own name. A program holding the VALUE found nothing.
    # `format` TAKES WHATEVER IT IS GIVEN, positionally and by keyword, which
    # is why it is not in the method table: no row there can say "any count".
    # The WRITTEN form is lowered at the call site; this is the same call
    # reached by name.
    if want == "format" and isinstance(obj, str):
        def _formatted(*rest, _o=obj, **named):
            return _unwrap(h, _apy_str_format(
                h, [h._new(_o), h._new(list(rest)), h._new(dict(named))]))
        return h._new(Native("format", _formatted, owner=obj,
                             variadic=True))
    if want == "maketrans" and isinstance(obj, (str, bytes, bytearray)):
        return made("maketrans",
                    lambda *r, _t=type(obj): _t.maketrans(*r))
    if want == "fromhex" and isinstance(obj, (bytes, bytearray, float)):
        return made("fromhex", lambda text, _t=type(obj): _t.fromhex(text))
    if want == "fromkeys" and dict_:
        return made("fromkeys", lambda keys, value=None:
                    dict.fromkeys(keys, value))
    if want == "from_bytes" and isinstance(obj, int):
        return made("from_bytes", lambda data, order="big":
                    int.from_bytes(data, order))
    if want == "from_number" and isinstance(obj, (float, complex)):
        return made("from_number",
                    lambda v, _t=type(obj): _t.from_number(v))
    if want == "__buffer__" and isinstance(obj, (bytes, bytearray,
                                                 memoryview)):
        return made("__buffer__", lambda flags=0: memoryview(obj))
    # PEP 688's OTHER HALF, and the one bytearray internal a program can
    # read. `__release_buffer__` is what a `with memoryview(...)` block calls
    # on the way out, and only the buffer that can be RESIZED carries it --
    # bytes cannot move under a view and has nothing to be told.
    if isinstance(obj, bytearray):
        if want == "__release_buffer__":
            # PEP 688 SAYS WHAT IT IS HANDED: the view being closed, and one
            # over THIS buffer. A `with memoryview(...)` block that ends on
            # the wrong object is a mistake worth naming.
            def _release(view, _o=obj):
                if not isinstance(view, memoryview):
                    h._fail("TypeError", "expected a memoryview object")
                    raise _UserFailed
                if view.obj is not _o:
                    h._fail("ValueError",
                            "memoryview's buffer is not this object")
                    raise _UserFailed
                return None
            return made("__release_buffer__", _release)
        if want == "__alloc__":
            # CPython's answer for a freshly built one is its length plus the
            # terminator, and nought for an empty one, which holds no buffer.
            return made("__alloc__", lambda: len(obj) + 1 if obj else 0)
    # WHAT `copy` AND `pickle` REBUILD A VALUE FROM. Every immutable builtin
    # answers `(self,)` -- a complex answers its two halves -- and a mutable
    # one has none at all, because anything it handed back would be SHARED
    # with the copy rather than rebuild it.
    if want == "__getnewargs__" and isinstance(obj, (str, bytes, tuple, int,
                                                     float, complex)):
        if is_complex:
            return made("__getnewargs__", lambda: (obj.real, obj.imag))
        return made("__getnewargs__", lambda: (obj,))
    # `bytes(x)` ASKS `x` FOR ITSELF FIRST, and bytes is the kind that
    # answers -- a bytearray does not, which is why `bytes(ba)` copies.
    if want == "__bytes__" and isinstance(obj, bytes):
        return made("__bytes__", lambda: obj)
    # `list[int]` REACHED BY NAME. The written form is a subscript the
    # frontend lowers; this is the method behind it, which `typing` calls
    # directly when it parameterises a container. TEXT HAS NONE: `str[int]`
    # is a TypeError in Python and the attribute is absent.
    # AND `generator[int]` BESIDE THEM, which a generic annotation on an
    # `async def` or a `Generator[...]`-shaped alias reaches by name.
    if want == "__class_getitem__" and (seq or dict_ or set_
                                        or isinstance(obj, Gen)):
        return made("__class_getitem__", lambda k: Alias(
            h._get(_apy_type_object(h, [h._new(obj)]), "__class_getitem__"),
            k if isinstance(k, tuple) else (k,)))
    # A GENERATOR HAS A FINALISER and is the only builtin value here that
    # does: closing an abandoned one runs its `finally` blocks, which is why
    # CPython gives the type a `__del__` where a list has none. CALLING IT
    # DOES NOTHING, which is what CPython's does for a generator that has
    # already finished -- and one that has not is closed by the runtime
    # rather than by the program.
    if want == "__del__" and isinstance(obj, Gen):
        return made("__del__", lambda: None)
    # AND THE PROTOCOL EACH OF THE THREE ANSWERS TO. A coroutine is what
    # `await` walks and carries `__await__`; an async generator is what
    # `async for` walks and carries the two halves of that protocol; a plain
    # generator has neither, which is what `dir()` over each says.
    #
    # `c.__await__()` ANSWERS A WRAPPER, which is a second object and not
    # the coroutine -- see `_apy_coro_wrapper`. `__aiter__` on an async
    # generator does hand the receiver back, because there is no second
    # object there. `__anext__` is `asend(None)`, which is what CPython's is.
    if isinstance(obj, Gen) and obj.coro and not obj.agen and not obj.wrapper:
        if want == "__await__":
            return made("__await__",
                        lambda: _unwrap(h, _apy_coro_wrapper(
                            h, [h._new(obj)])))
    if isinstance(obj, Gen) and obj.agen:
        if want == "__aiter__":
            return made("__aiter__", lambda: obj)
        if want == "__anext__":
            return made("__anext__", lambda: h._get(
                _apy_agen_asend(h, [h._value(obj), h._none]), "__anext__"))
    # WHICH FLOATING-POINT FORMAT THIS BUILD USES. One answer, and a float is
    # the only kind ever asked.
    if want == "__getformat__" and isinstance(obj, float):
        def _format_of(key):
            if key not in ("double", "float"):
                h._fail("ValueError", "__getformat__() argument 1 must be "
                        "'double' or 'float'")
                raise _UserFailed
            return "IEEE, little-endian"
        return made("__getformat__", _format_of)
    # THE REFLECTED `%`, which text carries and always refuses -- `1 % "a"`
    # is a TypeError the OPERATOR raises after this answers NotImplemented.
    # A NUMBER'S IS ELSEWHERE: `_NUM_REAL` already has it.
    if want == "__rmod__" and text:
        return made("__rmod__", lambda o: NotImplemented)
    if want in _TABLE_METHODS and hasattr(type(obj), want):
        # AND THE WHOLE METHOD TABLE. Everything above answers a PROTOCOL
        # name or a field; this answers the ORDINARY methods, which existed
        # only as calls the frontend lowered and so could not be reached by
        # NAME at all -- `getattr("abc", "upper")` was an AttributeError
        # about a method the object plainly has.
        #
        # THROUGH THE RUNTIME'S OWN SYMBOLS rather than through Python's
        # method, so the written form and the looked-up form are ONE
        # implementation: `DYN_METHOD_TABLE` says which symbol serves which
        # arity, and the compiled halves read the same table.
        return _made_table_method(h, obj, want)
    if rng:
        # THE THREE NUMBERS A RANGE IS, read back -- and `index`/`count`,
        # which are arithmetic on them rather than a walk.
        if want in ("start", "stop", "step"):
            return h._value(getattr(obj, want))
        if want in ("index", "count"):
            return made(want, lambda x, _w=want: getattr(obj, _w)(x))
        if want == "__len__":
            return made("__len__", lambda: len(obj))
        if want == "__iter__":
            # THROUGH `_unwrap` -- see the `__iter__` branch above: an
            # `_apy_*` entry point answers a HANDLE, and handing that back
            # made `range(3).__iter__()` an int rather than a cursor.
            return made("__iter__",
                        lambda: _unwrap(h, _apy_iter(h, [h._new(obj)])))
        if want == "__contains__":
            return made("__contains__", lambda x: x in obj)
        if want == "__getitem__":
            return made("__getitem__", lambda i: obj[i])
    return None


def _object_new_of(h):
    """`object.__new__(cls)` -- and the two refusals CPython makes here.

    A CLASS EXTENDING A BUILTIN IS REFUSED, which is CPython's rule and not
    an arbitrary one: `object.__new__` would build the shell and leave the
    builtin half empty, where `str.__new__(S, value)` fills it. CPython walks
    up to the first base that is not a Python class and refuses when its
    `__new__` is not object's -- which for `class S(str)` is `str`'s.

    AN ARGUMENT BEYOND THE CLASS IS FOR `__init__` TO TAKE, and only when
    there is one to take it. CPython draws the line twice: a class that wrote
    `__new__` gets the argument count complained about, and a class with
    neither is told it takes none.
    """
    def _new(cls, *rest):
        who = cls.name if isinstance(cls, Class) else h.kind_name(cls)
        if isinstance(cls, Class) and cls.builtin_kind() is not None:
            h._fail("TypeError",
                    f"object.__new__({who}) is not safe, "
                    f"use {who}.__new__()")
            raise _UserFailed
        # BY VALUE AND NOT BY COUNT, which this path alone could do. The
        # compiled halves reach the same native with a trailing slot the
        # arity check has already filled from `defaults`, so they see two
        # arguments whether or not one was written and read a None as
        # "omitted". Counting here instead would make the interpreter right
        # and the other two wrong, which is worse than all three agreeing.
        # See #160, which is that one spelling.
        rest = [one for one in rest if one is not None]
        if rest and isinstance(cls, Class):
            if cls.find("__new__") is not None:
                h._fail("TypeError", "object.__new__() takes exactly one "
                                     "argument (the type to instantiate)")
                raise _UserFailed
            if cls.find("__init__") is None:
                h._fail("TypeError", f"{who}() takes no arguments")
                raise _UserFailed
        return Instance(cls, h)

    return _new


def _object_default(h, name: str):
    """`object`'s own version of a dunder, as a callable value.

    The bodies are the ones the C reaches through `apy_default_*`; what was
    missing in both runtimes was a value naming them.
    """
    made = h._defaults.get(name)
    if made is not None:
        return made
    if name == "__init__":
        body = lambda *a: None
    elif name == "__new__":
        # An implicit STATICMETHOD: the argument is the class.
        #
        # A CONTENT ARGUMENT FILLS THE BUILTIN HALF, which is the only way to
        # build an immutable one: `class P(tuple)` has to be filled here or
        # never, and an enum member of a `class Colour(str, Enum)` is its
        # VALUE rather than an empty string. The same fill
        # `super().__new__(cls, x)` already does -- see `_builtin_new` -- and
        # CPython reaches it as `str.__new__(cls, value)`, a spelling this
        # runtime has no type object to write.
        body = _object_new_of(h)
    elif name in ("__repr__", "__str__"):
        body = lambda v, *a: (_inst_repr(v) if isinstance(v, Instance)
                              else h._text(v, True))
    elif name == "__eq__":
        body = lambda a, b, *r: a is b
    elif name == "__ne__":
        body = lambda a, b, *r: a is not b
    elif name == "__hash__":
        body = lambda v, *a: id(v)
    elif name == "__init_subclass__":
        # Every class has one, and a user hook ends by calling it: `object`'s
        # is the no-op that terminates the chain.
        body = lambda *a, **kw: None
    else:
        return None
    made = Native(name, body)
    h._defaults[name] = made
    return made


class Super:
    """What `super()` evaluates to: where to start looking, and for whom.

    `frm` is the class the calling method was DEFINED IN, not `type(recv)`.
    With `B(A)` and `C(B)`, a `super().m()` written in B's `m` must find A's;
    starting from `type(recv)` would find B's own and recurse until the stack
    ran out.
    """

    __slots__ = ("frm", "recv")

    def __init__(self, frm, recv) -> None:
        self.frm = frm
        self.recv = recv


def _held_binary(self, name, other):
    """The operation performed by the BUILTIN the class extends, or
    NotImplemented if that is not what this instance is.

    THE HOST'S OWN OPERATORS NEED IT TOO, which is easy to miss. `_binop` and
    `_cmpop` are what the IR calls, and they already substitute -- but
    `sorted`, `min`, `max` and `list.sort` compare with PYTHON's `<` on the
    `Instance` objects directly, and that reaches these generated dunders
    instead. Without the same fallback, `sorted([P(2, 1), P(1, 9)])` on a
    `namedtuple` reported `'<' not supported between instances of 'Instance'
    and 'Instance'` -- naming this file's class, about two tuples.
    """
    if self.held is None or self.cls.find(name) is not None:
        return NotImplemented
    theirs = other
    if isinstance(other, Instance) and other.held is not None \
            and other.cls.find(name) is None:
        theirs = other.held
    method = getattr(self.held, name, None)
    if method is None:
        return NotImplemented
    return method(theirs)


def _make_binary(name, rname):
    def run(self, other):
        got = self._send(name, other)
        return _held_binary(self, name, other) if got is NotImplemented \
            else got

    def rrun(self, other):
        got = self._send(rname, other)
        return _held_binary(self, rname, other) if got is NotImplemented \
            else got
    return run, rrun


for _op, _rop in (("add", "radd"), ("sub", "rsub"), ("mul", "rmul"),
                  ("truediv", "rtruediv"), ("floordiv", "rfloordiv"),
                  ("mod", "rmod"), ("pow", "rpow"), ("and", "rand"),
                  ("or", "ror"), ("xor", "rxor"), ("lshift", "rlshift"),
                  ("rshift", "rrshift")):
    _f, _rf = _make_binary(f"__{_op}__", f"__{_rop}__")
    setattr(Instance, f"__{_op}__", _f)
    setattr(Instance, f"__{_rop}__", _rf)

#: The four orderings, their MIRRORED reflection and the symbol each spells.
#: The reflected form of a comparison is the mirrored operator and not an
#: `__r`-prefixed one: `a < b` retries as `b.__gt__(a)`, because what b is
#: asked is the comparison as seen from its side.
_ORDERINGS = (("lt", "gt", "<"), ("le", "ge", "<="),
              ("gt", "lt", ">"), ("ge", "le", ">="))


def _make_order(name, mirror, symbol):
    """One ordering dunder, WITH ITS OWN REFUSAL.

    THE REFLECTION IS TRIED HERE rather than left to Python, and the refusal
    is worded here rather than left to Python, because Python words it from
    `type(a).__name__` -- which for every user object is `Instance`, this
    file's class. `sorted([Bare(), Bare()])` reported `'<' not supported
    between instances of 'Instance' and 'Instance'` about two objects of a
    class the program named itself, while `Bare() < Bare()` -- which goes
    through `_cmpop`, and which rewrites the same message -- named it right.

    Returning the sentinel and letting Python raise is what could not be
    kept: by the time Python raises, both sides have refused and there is
    nowhere left to correct the text.
    """
    def run(self, other):
        got = self._send(name, other)
        if got is not NotImplemented:
            return got
        got = _held_binary(self, name, other)
        if got is not NotImplemented:
            return got
        if not isinstance(other, Instance):
            # AGAINST ANYTHING ELSE THIS MAY BE THE REFLECTED CALL -- Python
            # reaches `__gt__` both as `max`'s own comparison and as the
            # mirror of the `<` a sort drives -- so which operator the
            # program would see, and in which order its operands, comes from
            # the consumer rather than from the dunder that was reached.
            # Letting Python word it kept both right and named this file's
            # `Instance` as the type; wording it from the dunder alone named
            # the program's type and flipped the comparison.
            driver = self.h.order_driver
            first, second = ((self, other) if symbol == driver
                             else (other, self))
            raise TypeError(f"'{driver}' not supported between instances of "
                            f"'{self.h.kind_name(first)}' and "
                            f"'{self.h.kind_name(second)}'")
        back = other._send(mirror, self)
        if back is not NotImplemented:
            return back
        back = _held_binary(other, mirror, self)
        if back is not NotImplemented:
            return back
        raise TypeError(f"'{symbol}' not supported between instances of "
                        f"'{self.h.kind_name(self)}' and "
                        f"'{self.h.kind_name(other)}'")
    return run


for _op, _mirror, _symbol in _ORDERINGS:
    setattr(Instance, f"__{_op}__", _make_order(f"__{_op}__", f"__{_mirror}__",
                                                _symbol))

_UNARY_SYMBOL = {"__neg__": "-", "__pos__": "+", "__invert__": "~"}

for _op in ("neg", "pos", "invert"):
    def _unary(self, _name=f"__{_op}__"):
        out = self._send(_name)
        if out is NotImplemented:
            raise TypeError(f"bad operand type for unary "
                            f"{_UNARY_SYMBOL[_name]}: '{self.cls.name}'")
        return out
    setattr(Instance, f"__{_op}__", _unary)


# ── the entry points ────────────────────────────────────────────────────────

def _apy_cell_new(h, a):
    """A CELL COUNTS WHAT IS IN IT, exactly as a list counts its elements.

    A closure's box is the one place a value could be RETAINED without
    anything counting it, and that made it the one place the interpreter
    could not retire a call argument at its last read -- see
    `ir/interpreter.py`'s `_consume`. `def outer(o): def inner(): return o`
    keeps `o` alive through `inner`, with nothing but this cell holding it
    once `outer`'s frame is gone; without the incref here, retiring the
    CALLER's temporary finalized `o` while `inner` still closed over it.

    The matching decref is `_held_by`/`_decref_contents`: a Cell reports
    its slot, so the whole cascade -- finalization order, the cycle
    collector's subtraction -- treats it like any other container.
    """
    v = h._get(a[0], "apy_cell_new")
    held = h._referent(v)
    if held is not None:
        h.incref(held)
    return h._new(Cell(v))


def _apy_cell_get(h, a):
    return h._value(h._get(a[0], "apy_cell_get").slot)


def _apy_cell_set(h, a):
    """Rebinding a closed-over variable. NEW BEFORE OLD, and see
    `_apy_cell_new` for why either happens at all."""
    cell = h._get(a[0], "apy_cell_set")
    v = h._get(a[1], "apy_cell_set")
    old = h._referent(cell.slot)
    held = h._referent(v)
    cell.slot = v
    if held is not None:
        h.incref(held)
    if old is not None and old != held:
        h.decref(old)
    return h._none


def _apy_func_new(h, a):
    """`a[0]` is a FUNC_ADDR, which is not a value handle.

    The interpreter tags a function address as `_FUNC_TAG | index`, so it can
    never be confused with a handle, and it is stored verbatim -- decoding
    happens once, where the call is made.
    """
    return h._new(Func(int(a[0]), int(a[1]),
                       str(h._get(a[2], "apy_func_new")), int(a[3]),
                       int(a[4]), bool(int(a[5]))))


def _apy_func_default(h, a):
    """`def f(x=<value>)` -- the default lives on the FUNCTION, so the
    function holds a reference to it. Counted, and dropped by
    `_held_by(Func)`; a default is written once, at the `def`, so there is
    no old value to release."""
    f = h._get(a[0], "apy_func_default")
    v = h._get(a[2], "apy_func_default")
    held = h._referent(v)
    if held is not None:
        h.incref(held)
    f.defaults[int(a[1])] = v
    return a[0]


def _apy_func_doc(h, a):
    h._get(a[0], "apy_func_doc").doc = str(h._get(a[1], "apy_func_doc"))
    return a[0]


def _apy_func_posonly(h, a):
    h._get(a[0], "apy_func_posonly").posonly = int(a[1])
    return a[0]


def _apy_func_kwonly(h, a):
    h._get(a[0], "apy_func_kwonly").kwonly = int(a[1])
    return a[0]


def _apy_func_kwdefaults(h, a):
    """How many of the TRAILING DEFAULTS are the keyword-only parameters'.

    Not derivable from `kwonly`: one of those may be REQUIRED, and `def f(a,
    b=1, *args, c)` has one keyword-only parameter and one default that is not
    its -- so splitting on `kwonly` reported `b`'s default as `c`'s.
    """
    h._get(a[0], "apy_func_kwdefaults").nkwdefault = int(a[1])
    return a[0]


def _apy_func_kwarg(h, a):
    h._get(a[0], "apy_func_kwarg").kwarg = int(a[1]) != 0
    return a[0]


def _apy_func_param(h, a):
    f = h._get(a[0], "apy_func_param")
    i = int(a[1])
    if 0 <= i < len(f.pnames):
        f.pnames[i] = str(h._get(a[2], "apy_func_param"))
    return a[0]


def _apy_func_cell(h, a):
    """Attach one of the closure's boxes. THE FUNCTION HOLDS THE CELL, and
    the cell holds the value (`_apy_cell_new`) -- that chain is what keeps a
    captured variable alive for exactly as long as some closure over it is,
    and no longer. Dropped by `_held_by(Func)`; attached once, at the `def`,
    so there is no old cell to release."""
    f = h._get(a[0], "apy_func_cell")
    cell = h._get(a[2], "apy_func_cell")
    held = h._referent(cell)
    if held is not None:
        h.incref(held)
    f.cells[int(a[1])] = cell
    return a[0]


def _apy_env_cell(h, a):
    env = h._get(a[0], "apy_env_cell")
    if not isinstance(env, Func):
        return h._fail("SystemError", "closure environment is not a function")
    return h._value(env.cells[int(a[1])])


def _apy_type_new(h, a):
    base = h._get(a[1], "apy_type_new")
    return h._new(Class(str(h._get(a[0], "apy_type_new")),
                        base if isinstance(base, Class) else None))


def _apy_type_set(h, a):
    cls = h._get(a[0], "apy_type_set")
    cls.dict[str(h._get(a[1], "apy_type_set"))] = h._get(a[2], "apy_type_set")
    return h._none


def _apy_init_subclass(h, a):
    """`__init_subclass__` -- the hook a base runs when a subclass is created.

    CALLED ON THE BASE, NOT THE NEW CLASS, and with the new class as its
    argument: a class does not announce its own creation to itself. Run after
    the body has been filled, because the hook routinely reads what it bound.
    """
    cls = h._get(a[0], "apy_init_subclass")
    if not isinstance(cls, Class) or not isinstance(cls.base, Class):
        return h._none
    hook = cls.base.find("__init_subclass__")
    if hook is None:
        return h._none
    # THE CLASS KEYWORDS TRAVEL WITH IT: `class A(Base, tag="a")` is how a
    # program configures the hook, and dropping them left every subclass
    # looking identically unconfigured.
    kwd = h._get(a[1], "apy_init_subclass") if len(a) > 1 else {}
    # Whatever it answers is discarded -- it is called for its effect.
    # MATCHED BY NAME against the hook's declared parameters, as the C's
    # `apy_call_kw` does. `_invoke` alone puts every keyword into `**kw`, so
    # `def __init_subclass__(cls, tag=None, **kw)` saw `tag` in `kw` and left
    # the parameter on its default -- the wrong answer this whole path is
    # about.
    names = list(getattr(hook, "pnames", None) or [])
    extra = dict(kwd)
    args = [cls]
    for pname in names[1:]:
        if pname is None or pname not in extra:
            break
        args.append(extra.pop(pname))
    if _user(h, lambda: (h._invoke(hook, args, kwrest=extra), 1)[1]) == 0:
        return 0
    return h._none


#: The C's kind enum, as the Python types the host uses. The two lists must
#: agree -- a wrong number gives an instance the wrong kind of storage.
_BUILTIN_KINDS = {4: str, 5: list, 6: tuple, 7: dict, 9: set}


def _apy_builtin_new(h, a):
    """`str.__new__(cls, content)` -- the builtin half of an instance, built
    through the type it extends. This is how CPython's own `enum` makes a
    mixin member: `member_type.__new__(enum_class, value)`.

    AN IMPLICIT STATICMETHOD, so the first argument is the CLASS TO BUILD and
    not a receiver of this type. The ordinary unbound-method check --
    `apy_descr_applies` -- therefore does not apply to it, and did.

    THE CONTENT IS TAKEN ONLY BY THE IMMUTABLE KINDS. `list.__new__(L,
    [1, 2])` is an EMPTY list in CPython and `tuple.__new__(T, [1, 2])` is
    `(1, 2)`: a mutable builtin fills in `__init__`, and an immutable one has
    nowhere else to do it.

    THE BUILTIN ITSELF ANSWERS A PLAIN ONE. `str.__new__(str, "ab")` is
    `"ab"` -- there is no class to put it in.
    """
    want = str(h._get(a[0], "apy_builtin_new"))
    kind = _BUILTIN_KINDS.get(int(a[1]))
    cls = h._get(a[2], "apy_builtin_new")
    content = h._get(a[3], "apy_builtin_new")
    has = content is not None
    given = None
    if isinstance(cls, Class):
        given = cls.name
        if cls.builtin_kind() is kind and kind is not None:
            made = Instance(cls, h)
            if kind in (str, tuple) and has:
                made.held = kind(_content_of(content))
            return h._new(made)
    elif isinstance(cls, (Func, Native)) and getattr(cls, "is_type", False):
        given = cls.name
        if given == want:
            return h._value(kind(_content_of(content)) if has else kind())
    else:
        # NOT A TYPE AT ALL, which CPython words differently from a type that
        # is simply the wrong one -- and writes as a literal `X`.
        return h._fail("TypeError",
                       f"{want}.__new__(X): X is not a type object "
                       f"({h.kind_name(cls)})")
    return h._fail("TypeError", f"{want}.__new__({given}): {given} is not a "
                                f"subtype of {want}")


def _apy_type_builtin(h, a):
    """`class D(dict)` -- which builtin kind this class extends."""
    cls = h._get(a[0], "apy_type_builtin")
    cls.builtin = _BUILTIN_KINDS.get(int(a[1]))
    return a[0]


def _apy_instance_new(h, a):
    cls = h._get(a[0], "apy_instance_new")
    if not isinstance(cls, Class):
        return h._fail("TypeError",
                       f"'{h.kind_name(cls)}' object is not callable")
    return h._new(Instance(cls, h))


def _apy_aenter(h, a):
    """`async with cm:` -- the `__aenter__` half.

    ANSWERS A COROUTINE rather than a value: the caller awaits it, which is
    the whole difference from `__enter__` and the reason these cannot share an
    entry point.
    """
    cm = h._get(a[0], "apy_aenter")
    if not isinstance(cm, Instance) or cm.cls.find("__aenter__") is None:
        return h._fail("TypeError",
                       f"'{h.kind_name(cm)}' object does not support the "
                       f"asynchronous context manager protocol")
    return _user(h, lambda: h._value(cm._send("__aenter__")))


def _apy_aexit(h, a):
    """`__aexit__(type, value, traceback)`, answering a coroutine.

    All three arguments come from the one value, as `apy_exit` does it: the
    TYPE is what `et.__name__` reads, the VALUE is the exception itself, and
    the traceback is None because there are none here.
    """
    cm = h._get(a[0], "apy_aexit")
    exc = h._get(a[1], "apy_aexit")
    if not isinstance(cm, Instance) or cm.cls.find("__aexit__") is None:
        return h._fail("TypeError",
                       f"'{h.kind_name(cm)}' object does not support the "
                       f"asynchronous context manager protocol")
    kind = h._type_of(exc) if isinstance(exc, Exc) else None
    return _user(h, lambda: h._value(
        cm._send("__aexit__", kind, exc if isinstance(exc, Exc) else None,
                 None)))


def _apy_enter(h, a):
    """`with cm:` -- the `__enter__` half.

    A missing method is reported as "not a context manager" rather than as a
    missing attribute, matching the C, because that is the sentence that tells
    the reader which half of the protocol to write.
    """
    cm = h._get(a[0], "apy_enter")
    # A MEMORYVIEW IS THE ONE BUILTIN THAT IS A CONTEXT MANAGER, and its two
    # halves are this runtime's own code reached BY KIND -- the instance
    # lookup below would never find them. What the block binds is the view
    # itself; what leaving it does is release.
    if isinstance(cm, memoryview):
        if not _mview_live(h, cm):
            return 0
        return a[0]
    if not isinstance(cm, Instance) or cm.cls.find("__enter__") is None:
        return h._fail("TypeError",
                       f"'{h.kind_name(cm)}' object does not support the "
                       f"context manager protocol (missed __enter__ method)")
    return _user(h, lambda: h._value(cm._send("__enter__")))


def _apy_exit(h, a):
    """`__exit__(type, value, tb)` on every path out of a `with`.

    The exception's TYPE is passed, not just the value: a manager that logs
    `et.__name__` needs it, and a true return has to SWALLOW the exception
    rather than merely observe it -- which is why the caller branches on what
    this returns.
    """
    cm = h._get(a[0], "apy_exit")
    exc = h._get(a[1], "apy_exit")
    # THE VIEW HANDS ITS BUFFER BACK on the way out, and swallows nothing.
    if isinstance(cm, memoryview):
        cm.release()
        return h._none
    if not isinstance(cm, Instance) or cm.cls.find("__exit__") is None:
        return h._fail("TypeError",
                       f"'{h.kind_name(cm)}' object does not support the "
                       f"context manager protocol (missed __exit__ method)")
    if isinstance(exc, Exc):
        args = (h._type_of(exc), exc, None)
    else:
        args = (None, None, None)
    return _user(h, lambda: h._value(cm._send("__exit__", *args)))


def _content_of(v):
    """What a builtin constructor should be handed for `v`.

    `OrderedDict(other_ordered_dict)` reaches `dict(instance)`, and the host's
    `dict` cannot read one: it subscripts what it is given, and an Instance is
    not subscriptable. The instance's own dict is what the caller meant, and
    handing that over is also what CPython does -- a mapping argument is
    copied as a mapping.
    """
    return v.held if isinstance(v, Instance) and v.held is not None else v


def _builtin_init(*args):
    """`super().__init__(...)` where the base chain ends at a BUILTIN.

    `class M(dict)` writing `super().__init__(other)` means `dict.__init__`,
    which FILLS the instance -- and the base chain has no Python in it to find
    that on, so the walk ran out, `object.__init__` answered, and the call
    quietly did nothing. An empty dict where the program asked for a full one
    is the shape of failure this whole arrangement is worst at showing: no
    error, just a container that is not there.
    """
    if not args or not isinstance(args[0], Instance):
        return None
    recv, rest = args[0], list(args[1:])
    kind = recv.cls.builtin_kind()
    if kind is None:
        return None
    # IN PLACE, not a new object: `super().__init__()` initialises the
    # instance the caller already has, and the lines after it in the same
    # `__init__` go on using it.
    recv.held = kind(*[_content_of(one) for one in rest])
    return None


def _exc_init(*args):
    """`BaseException.__init__(*args)`: it SETS THE MESSAGE AND `args`, which
    is the whole of what it does and the reason a class writing
    `super().__init__(f"{code}: {message}")` prints that text."""
    if not args or not isinstance(args[0], Exc):
        return None
    exc, rest = args[0], list(args[1:])
    exc.arg = rest[0] if rest else None
    exc.has_arg = bool(rest)
    exc.argv = tuple(rest)
    # The text is the ARGUMENT again, not something already rendered -- see
    # `Exc.rendered` for what that flag stops twice over.
    exc.rendered = False
    return None


def _apy_super(h, a):
    frm = h._get(a[0], "apy_super")
    if not isinstance(frm, Class):
        return h._fail("TypeError", "super(type, obj): obj must be an "
                                    "instance or subtype of type")
    return h._new(Super(frm, h._get(a[1], "apy_super")))


def _apy_descr_applies(h, a):
    """`str.upper(5)` -- an unbound method applied to the wrong kind.

    CPython checks the receiver against the type the descriptor was found on
    before it runs, and words the refusal as the DESCRIPTOR's. Without the
    check the call went through as `(5).upper()` and reported a missing
    attribute -- true of the int, and not what the program got wrong.

    THE RECEIVER COMES BACK, so the check sits in the expression rather than
    beside it. `apy_isinstance` and not a kind comparison, because a SUBCLASS
    passes: `int.bit_length(True)` is 1 and a bool is an int.
    """
    recv = h._get(a[0], "apy_descr_applies")
    want = str(h._get(a[1], "apy_descr_applies"))
    method = str(h._get(a[2], "apy_descr_applies"))
    ok = _apy_isinstance(h, (a[0], a[1]))
    if ok == 0:
        return 0
    if h._get(ok, "apy_descr_applies"):
        return a[0]
    return h._fail("TypeError",
                   f"descriptor '{method}' for '{want}' objects doesn't "
                   f"apply to a '{h.kind_name(recv)}' object")


def _apy_is_instance(h, a):
    return 1 if isinstance(h._get(a[0], "apy_is_instance"), Instance) else 0


def _apy_alloc_block(h, a):
    """`n` bytes of interpreter memory that the caller may hand back.

    REACHED ONLY BY PORTED IR. The host's own containers are Python lists and
    dicts, so nothing here has a `v.q.items` to allocate -- but an exported
    symbol the interpreter cannot answer is precisely the drift the ported
    runtime exists to remove, and `test_every_exported_symbol_has_a_host_
    binding` is what says so.

    NO FREE LIST, deliberately. `runtime/blocks.py` reuses blocks because a
    compiled program runs until it exits and its arena is all the memory it
    will ever have; the interpreter is a Python process whose memory outlives
    any one program and whose `mem` grows on demand. Reusing here would be
    machinery serving no measurement, so `apy_free_block` is a no-op and this
    hands out fresh bytes each time -- which is CORRECT and not thrifty, and
    the two paths still agree on every value.
    """
    n = int(h._get(a[0], "apy_alloc_block"))
    return h._interp.mem.alloc(n if n > 0 else 1)


def _apy_realloc_block(h, a):
    """`want` bytes holding the first `was` of what `p` held."""
    p = int(h._get(a[0], "apy_realloc_block"))
    was = int(h._get(a[1], "apy_realloc_block"))
    want = int(h._get(a[2], "apy_realloc_block"))
    if want < 1:
        want = 1
    fresh = h._interp.mem.alloc(want)
    if p:
        keep = was if was < want else want
        if keep > 0:
            buf = h._interp.mem.buf
            buf[fresh:fresh + keep] = buf[p:p + keep]
    return fresh


def _apy_free_block(h, a):
    """A no-op. See `_apy_alloc_block` for why there is no free list here."""
    return 0


def _ascii_pred(which):
    """One of the five ASCII predicates, for the interpreter.

    PROMOTED SYMBOLS NEED A HOST BINDING, which is what
    `test_every_exported_symbol_has_a_host_binding` insists on and the reason
    it exists: an exported symbol only the C can answer is exactly the drift
    the ported runtime is meant to remove.

    THE ARGUMENT IS A BYTE, arriving as the machine word the C widened it to.
    Anything outside 0..127 is not ASCII and every one of these answers False
    for it, which is what makes them safe to run over UTF-8: every byte of a
    multi-byte sequence has its high bit set.
    """
    def run(h, a):
        c = int(h._get(a[0], "apy_c_" + which)) & 0xFF
        if which == "lower":
            return 1 if 0x61 <= c <= 0x7A else 0
        if which == "upper":
            return 1 if 0x41 <= c <= 0x5A else 0
        if which == "alpha":
            return 1 if (0x41 <= c <= 0x5A or 0x61 <= c <= 0x7A) else 0
        if which == "digit":
            return 1 if 0x30 <= c <= 0x39 else 0
        return 1 if c in (0x20, 0x09, 0x0A, 0x0D, 0x0C, 0x0B) else 0
    return run


def _apy_method_is_builtin(h, a):
    """Whether `obj.name(...)` should take the BUILTIN side of the call.

    See the C of the same name for the argument. In short: the class body
    wins, a class that extends a builtin gets the builtin's method for
    everything it did not write, and an ordinary instance stays on the user
    side so `__getattr__` still gets its chance.
    """
    obj = h._get(a[0], "apy_method_is_builtin")
    if not isinstance(obj, Instance):
        return 1
    name = str(h._get(a[1], "apy_method_is_builtin"))
    if obj.cls.find(name) is not None:
        return 0
    return 1 if obj.held is not None else 0


def _apy_method_self(h, a):
    """The receiver the builtin side should act on: the held value, when the
    class did not define the method itself."""
    obj = h._get(a[0], "apy_method_self")
    if isinstance(obj, Instance) and obj.held is not None:
        name = str(h._get(a[1], "apy_method_self"))
        if obj.cls.find(name) is None:
            return h._new(obj.held)
    return a[0]


def _apy_getattr(h, a):
    """`x.name`. `__getattribute__` INTERCEPTS EVERYTHING, before the instance
    dict is even looked at -- that is what distinguishes it from
    `__getattr__`, which is consulted only after a miss."""
    obj = h._get(a[0], "apy_getattr")
    if isinstance(obj, Instance)             and obj.cls.find("__getattribute__") is not None:
        return _user(h, lambda: h._value(
            obj._send("__getattribute__", str(h._get(a[1], "apy_getattr")))))
    return _apy_default_getattr(h, a)


def _apy_getattr_default(h, a):
    """`getattr(x, 'a', fallback)`. A MISS IS NOT AN ERROR HERE, which is the
    whole difference from the two-argument form -- so the pending
    AttributeError is dropped and the fallback answered.

    ONLY an AttributeError is swallowed: a `__getattr__` that raised something
    of its own is the program's error and has to survive, or every failure
    inside a property would turn into a silent default.
    """
    before = h.err
    got = _apy_getattr(h, a[:2])
    if got:
        return got
    if h.err is not None and h.err is not before and h.err[0] == "AttributeError":
        h.err = before
        h.err_value = None
        # The HANDLE, not the unwrapped Python object: every host binding
        # answers a machine word, and `_get` would hand back the value behind
        # it for the caller to try to write into memory as an integer.
        return a[2]
    return 0


def _func_code(h, obj):
    """`f.__code__` -- ENOUGH OF ONE to answer what a program asks a function
    about its own signature. Not a real code object: there is no bytecode
    here to describe, and `co_argcount` and `co_varnames` are what
    introspection actually reads.

    TWO CALLERS AND ONE BODY. A generator's `gi_code` is the code of the `def`
    it came from -- in CPython literally the same object -- and the generator
    carries a Func describing that `def`, so both questions are this one.
    """
    declared = obj.arity - (1 if obj.vararg else 0) - (1 if obj.kwarg else 0)
    code = Instance(h._code_class(), h)
    code.dict["co_argcount"] = declared - obj.kwonly
    code.dict["co_posonlyargcount"] = obj.posonly
    code.dict["co_kwonlyargcount"] = obj.kwonly
    # `*rest` AND `**kw` COME LAST, after every declared parameter -- where
    # CPython puts them and where a signature rebuilt from this expects to
    # find them. Omitted before, so the rebuilt signature had no variadic
    # parts at all.
    code.dict["co_varnames"] = tuple(
        n for n in (obj.pnames or ())[:obj.arity] if n)
    # 0x04 is `*rest` and 0x08 is `**kw`, which is how a signature knows the
    # variadic parts exist without a second field.
    code.dict["co_flags"] = ((4 if obj.vararg else 0)
                             | (8 if obj.kwarg else 0))
    code.dict["co_name"] = obj.name
    code.dict["co_qualname"] = (obj.qualname if obj.qualname is not None
                                else obj.name)
    return code


def _gen_frame(h, g):
    """THE FRAME A SUSPENDED GENERATOR IS SITTING IN -- enough of one, for
    the reason the code object above is enough of one. `gi_frame` is read to
    ask where a generator is and what it can still see, and the two things a
    program reads off it are its code and whether it is there at all.
    """
    came = g.sig if g.sig is not None else g.step
    if not isinstance(came, Func):
        return None
    frame = Instance(h._gen_frame_class(), h)
    frame.dict["f_code"] = _func_code(h, came)
    frame.dict["f_back"] = None
    frame.dict["f_lineno"] = 0
    # WHERE THE BODY IS, as far as this runtime can say it: the resume state
    # a `yield` left behind, and -1 before the first step -- which is the
    # number CPython uses for "has not run an instruction yet".
    frame.dict["f_lasti"] = g.state if g.state > 0 else -1
    frame.dict["f_globals"] = {}
    frame.dict["f_locals"] = {}
    frame.dict["f_builtins"] = {}
    frame.dict["f_trace"] = None
    return frame


def _apy_default_getattr(h, a):
    """The DEFAULT lookup: instance dict, then class, then `__getattr__`.

    Named separately because a class that overrides `__getattribute__` needs a
    way to do what it overrode, and `object.__getattribute__(self, name)` is
    how Python spells that -- the only way out of the recursion.
    """
    obj = h._get(a[0], "apy_getattr")
    name = str(h._get(a[1], "apy_getattr"))
    if isinstance(obj, Instance):
        # A DATA DESCRIPTOR ON THE CLASS BEATS THE INSTANCE DICT -- the one
        # place "instance wins" does not hold, and what makes a property a
        # property. A NON-data one loses to it instead.
        klass_found = obj.cls.find(name)
        if klass_found is not None and _is_data_descriptor(klass_found):
            return _descr_get(h, klass_found, obj, obj.cls)
        # The INSTANCE DICT WINS over the class, so `self.x = 1` shadows a
        # class attribute of the same name.
        if name in obj.dict:
            return h._value(obj.dict[name])
        found = obj.cls.find(name)
        if found is not None:
            # A NON-DATA descriptor is asked HERE, after the instance dict
            # has missed -- `staticmethod`, `classmethod`, or a user class
            # with only `__get__`.
            if _is_descriptor(found):
                return _descr_get(h, found, obj, obj.cls)
            # A function on the class becomes a BOUND METHOD, and a fresh one
            # per access -- which is what CPython does and what
            # `datamodel/method-objects-are-created-per-access` measures.
            # A NATIVE BINDS TOO: the runtime's own methods take the receiver
            # as their first argument exactly as a written method does, so
            # leaving one unbound called it with nothing.
            return h._new(found.bind(obj)) \
                if isinstance(found, (Func, Native)) else h._value(found)
        if name == "__class__":
            return h._value(obj.cls)
        # THE INSTANCE'S OWN attributes, and the real dict rather than a copy:
        # `obj.__dict__["x"] = 1` is how a program sets an attribute
        # dynamically, and a copy would accept the write and lose it.
        # ABSENT under `__slots__`, which is the point of declaring it --
        # `hasattr(p, "__dict__")` is how a program checks.
        if name == "__dict__":
            if not _slot_allows(obj.cls, "__dict__"):
                return h._no_attr(obj, name)
            return h._new(obj.dict)
        # A CLASS THAT EXTENDS A BUILTIN answers with the builtin's own
        # method for everything its body did not define. `class D(dict)` with
        # only a `__missing__` still has `keys`, `items`, `get` and `update`,
        # and this is where they come from -- asked of `held`, the real dict
        # the instance carries.
        #
        # HERE AND NOT EARLIER, because the class body wins: a `Counter`
        # defining `update` must shadow `dict.update` rather than be shadowed
        # by it. And here and not LATER, because in CPython these arrive
        # through the MRO, which is consulted before `__getattr__` -- a class
        # extending a builtin AND defining `__getattr__` would otherwise route
        # every inherited method through the fallback.
        #
        # THE MISS IS NOT THE ANSWER. A name neither the class nor the builtin
        # has must still reach `__getattr__`, so the AttributeError the
        # delegation raised is rolled back rather than reported -- the same
        # rollback `_apy_getattr_default` does for its fallback.
        if obj.held is not None:
            before = h.err
            got = _apy_default_getattr(h, [h._new(obj.held), a[1]])
            if got:
                return got
            if h.err is not None and h.err is not before \
                    and h.err[0] == "AttributeError":
                h.err = before
                h.err_value = None
            else:
                return got
        # `__getattr__` -- the LAST resort, asked only after the instance dict
        # and the class have both missed. That ordering is the whole protocol.
        if obj.cls.find("__getattr__") is not None:
            return _user(h, lambda: h._value(obj._send("__getattr__", name)))
        return h._no_attr(obj, name)
    if isinstance(obj, slice):
        # `s.start`, `s.stop`, `s.step` -- what a `__getitem__` reads off the
        # slice it was handed. None where the bound was omitted.
        if name in ("start", "stop", "step"):
            return h._value(getattr(obj, name))
        return h._no_attr(obj, name)
    if isinstance(obj, Descr):
        # `p.fget` / `p.fset` -- the functions a property was built from. A
        # subclass overriding a property reaches the base getter through
        # `fget`, the only way to extend rather than replace it. None where
        # there is no such half, as CPython answers.
        if name == "fget":
            return h._value(obj.get) if obj.get is not None else h._none
        if name == "fset":
            return h._value(obj.set) if obj.set is not None else h._none
        if name == "fdel":
            return h._value(obj.del_) if obj.del_ is not None else h._none
        if name == "__func__" and obj.get is not None:
            return h._value(obj.get)
        # THE PROTOCOL ITSELF, as values. `hasattr(p, "__get__")` is how a
        # program asks whether something is a descriptor, and a property that
        # answered False to it was reported as an ordinary attribute.
        if name == "__get__":
            # A NATIVE'S BODY ANSWERS A RAW VALUE, and `_descr_get` answers a
            # HANDLE -- the two conventions meet here, and returning the
            # handle unchanged handed the program the integer index.
            def _read(inst, cls=None, _d=obj):
                got = _descr_get(h, _d, inst, cls)
                return h._get(got, "__get__") if got else None
            return h._new(Native("__get__", _read))
        if name == "__set__":
            return h._new(Native("__set__", lambda inst, value:
                                 _descr_write(h, obj, inst, value)))
        if name == "__delete__":
            return h._new(Native("__delete__", lambda inst:
                                 _descr_write(h, obj, inst, None,
                                              delete=True)))
        return h._no_attr(obj, name)
    if isinstance(obj, Class):
        if name == "__name__":
            return h._new(_bare_name(obj.name))
        # What the class BODY bound, not what it inherited -- the difference
        # `"x" in vars(C)` asks about. A copy: a type's dict is a mapping
        # proxy in CPython and is not writable.
        if name == "__dict__":
            return h._new(dict(obj.dict))
        # A SLOT NAME reached through the class is a DESCRIPTOR, not a
        # missing attribute: `__slots__` declares storage, and the class dict
        # holds nothing for it.
        #
        # THE WHOLE CHAIN, not this class's own `__slots__` alone: a slot
        # declared on a BASE is inherited through the MRO, so `Sub.held` for
        # a `held` declared on `Base` answers the descriptor in CPython and
        # answered None here -- while the C, which walks, answered the
        # descriptor. Two runtimes disagreeing about one class attribute.
        if name not in obj.dict and _slot_declares(obj, name):
            return h._new(Instance(h._member_descriptor_class(), h))
        # THE HIERARCHY, as a program reads it back. `object` is the root
        # of every chain even though no class links to it, so a class with no
        # written base still has one base and only `object` itself has none.
        # Answering the empty tuple there said the chain stopped at the class.
        # PEP 3155. A class nested in another would qualify differently;
        # only the top-level spelling is recorded, which is the same limit the
        # frontend's own keys have for classes.
        if name == "__qualname__":
            return h._new(obj.name)
        # A TYPE OBJECT REACHED THROUGH `type(x)` IS A CLASS CELL, not the
        # thunk the program's own `bytes` names -- `apy_type_for` mints one
        # keyed by the kind's name for anything the canonical table has no
        # entry for. So `type(b"").__doc__` took this path and `bytes.__doc__`
        # took the Func one, and only the second answered. A class the PROGRAM
        # wrote that shares a builtin's name is excluded by its own `__doc__`,
        # which its body bound and which is found first.
        if name == "__doc__" and "__doc__" not in obj.dict:
            # THROUGH THE GENERATED TABLE, which holds a None for a kind
            # whose type has no docstring -- a cursor's. `in` and not `get`,
            # because None is an ANSWER here and not an absence.
            if obj.name in KIND_DOC:
                return h._value(KIND_DOC[obj.name])
        # `C.__class__` IS THE METACLASS, which is `type` unless the class
        # named one. Every other kind answers this and a class did not, so
        # `C.__class__` was an AttributeError about a class that plainly has
        # one -- and `isinstance(x, C.__class__)` is how a program asks.
        if name == "__class__" and "__class__" not in obj.dict:
            # `_value` AND NOT `_new`: identity has to survive the handle, so
            # `C.__class__ is type` holds. A fresh handle for the same object
            # answers False to `is`, which is the one question this is for.
            return h._value(h._type_of(obj))
        # PEP 649 for a CLASS: `C.__annotations__` is built on access by the
        # thunk the body left in the dict, for the same reason a function's
        # is -- an annotation may name something that does not exist yet.
        if name == "__annotations__":
            # WHAT WAS STORED WINS OVER THE THUNK. A program may set
            # `C.__annotations__` directly, or hand a class body to
            # `type(name, bases, {"__annotations__": ...})`, and both write an
            # ordinary dict entry -- which this read ignored entirely, so the
            # write appeared to succeed and the read still answered `{}`.
            #
            # That is not an exotic spelling: it is how every library that
            # builds a class dynamically declares its fields, and it is why
            # `dataclasses.make_dataclass` could not work here at all. The
            # thunk stays the fallback, because for a class written out in
            # source PEP 649 is what defers evaluating the annotations.
            stored = obj.dict.get("__annotations__")
            if stored is not None:
                return h._value(stored)
            thunk = obj.dict.get("__annotate__")
            if thunk is None:
                return h._new({})
            return _user(h, lambda: h._value(h._invoke(thunk, [])))
        if name in ("__bases__", "__base__", "__mro__"):
            root = h._get(_apy_object_class(h, []), "apy_getattr")
            # AN EXCEPTION TYPE'S PARENT IS IN THE NAME TABLE, not in a base
            # pointer -- the builtin hierarchy is a table because `raise` and
            # `except` match on the name. So the walk had to ask the table, or
            # `Exception.__bases__` answered `object`.
            if obj.base is None and obj is not root:
                chain = _exc_chain(h, obj.name)
                # `_exc_chain` walks all the way to `object`, and the root is
                # appended below -- keeping both listed it twice.
                if chain and chain[-1] == "object":
                    chain = chain[:-1]
                if len(chain) > 1:
                    types = [h._get(_apy_exc_type(h, [h._new(nm)]),
                                    "apy_getattr") for nm in chain[1:]]
                    if name == "__base__":
                        return h._value(types[0])
                    if name == "__bases__":
                        return h._new((types[0],))
                    return h._new(tuple([obj] + types + [root]))
            if name == "__mro__":
                # THE RECORDED ORDER when the class has one -- that IS the
                # answer, and rebuilding it from the base chain would give a
                # different one for a class with several bases.
                if obj.mro is not None:
                    return h._new(tuple(obj.mro + [root]))
                walk, out = obj, []
                while isinstance(walk, Class) and walk is not root:
                    out.append(walk)
                    walk = walk.base
                return h._new(tuple(out + [root]))
            if obj is root:
                return h._new(() if name == "__bases__" else None)
            # ALL OF THEM for `__bases__`, in the order written; `__base__`
            # is the first alone.
            if name == "__bases__" and len(obj.bases) > 1:
                return h._new(tuple(obj.bases))
            one = obj.base if obj.base is not None else root
            # `h._value`, not `h._new`: `B.__base__ is A` compares handles, so
            # minting a fresh one for a class that already has a handle makes
            # it False. The C compares pointers and cannot make that mistake.
            return h._new((one,)) if name == "__bases__" else h._value(one)
        # DEFINING `__eq__` AND NOT `__hash__` SETS `__hash__` TO None, and a
        # program reads it back: `C.__hash__ is None` is how it asks whether
        # instances are hashable. `Instance.__hash__` already refuses to hash
        # one; without this the two disagreed about the same class.
        if (name == "__hash__" and obj.find("__hash__") is None
                and obj.find("__eq__") is not None):
            return h._none
        # THROUGH THE CLASS, a descriptor is asked with no instance:
        # `C.make()` binds the class, `C.plain` hands the plain function back,
        # and `Base.v` is the property object itself.
        klass_d = obj.find(name)
        if klass_d is not None and _is_descriptor(klass_d):
            return _descr_get(h, klass_d, None, obj)
        found = obj.lookup(name)
        # Through the CLASS a method is UNBOUND: `C.m(x)` passes x as self.
        if found is not _ABSENT:
            return h._value(found)
        # AN ATTRIBUTE OF THE CLASS'S OWN TYPE. `Quacks.register(Duck)` is a
        # method the METACLASS defines, and the class is its receiver -- the
        # same relationship an instance has to its class, one level up. Asked
        # LAST, because a name the class itself binds wins over one its
        # metaclass does.
        if obj.meta is not None:
            m = obj.meta.lookup(name)
            if m is not _ABSENT:
                if _is_descriptor(m):
                    return _descr_get(h, m, obj, obj)
                if isinstance(m, (Func, Native)):
                    return h._new(m.bind(obj))
                return h._value(m)
        # A CLASS `_type_of` MINTED ANSWERS WHAT ITS KIND CARRIES.
        # `type(iter([]))` is not the canonical thunk a NAMED builtin gets --
        # no program writes `list_iterator` -- so `_type_of` interns a bare
        # `Class` under the kind's name, and every method that kind has was
        # out of reach: `type(it).__next__` was an AttributeError about the
        # one method an iterator is FOR.
        #
        # AN EMPTY DICT, NO BASE AND NO METACLASS is what identifies one. A
        # class the program wrote is a `Class` too, and its body always
        # leaves `__module__` behind -- so a class named `list` cannot fall
        # in here and collect a list's methods by accident.
        if not obj.dict and obj.base is None and obj.meta is None:
            proto = _kind_prototype(obj.name)
            found = None if proto is None else _kind_attr(h, proto, name)
            if found is not None:
                # `__class_getitem__` IS A CLASSMETHOD and binds the TYPE --
                # see `_no_attr`, which draws the same line.
                if name == "__class_getitem__":
                    # THE TYPE IS ITS OWNER -- see `_no_attr`.
                    return h._new(Native(
                        name, h._get(found, name).body, owner=obj))
                # UNBOUND, as it is off a builtin type: `type(it).__next__`
                # takes the cursor as its argument, which is what
                # `type(it).__next__(it)` means.
                return h._new(Native(name, _unbound_kind(h, name),
                                     descr=obj.name))
        return h._fail("AttributeError",
                       f"type object '{obj.name}' has no attribute '{name}'")
    if isinstance(obj, Super):
        found = None
        # THE RECEIVER'S ORDER, PAST THE DEFINING CLASS. This is what makes a
        # diamond work: inside B's method, `super()` on a D instance must
        # reach C and not A, and only the RECEIVER's order knows that C sits
        # between them. With a single base this is the base chain again, which
        # is what the fallback below still walks.
        host = h._type_of(obj.recv) if obj.recv is not None else None
        if isinstance(host, Class) and host.mro is not None:
            order = host.order()
            at = next((i for i, k in enumerate(order) if k is obj.frm), -1)
            if at >= 0:
                for here in order[at + 1:]:
                    if name in here.dict:
                        found = here.dict[name]
                        break
        if found is None:
            base = obj.frm.base
            found = base.find(name) if base is not None else None
        # THE BASE CHAIN HAS RUN OUT OF PYTHON and the receiver is an
        # exception, so what `super().__init__(msg)` means is
        # `BaseException.__init__` -- which is not a function anywhere in that
        # chain, because the hierarchy above a user exception class is a table
        # of names rather than classes. AFTER the walk, not before: a subclass
        # writing `super().__init__(...)` must reach its own base's `__init__`
        # first, and intercepting early sent every one of them straight past
        # it. Falling through instead would find `object`'s, which takes the
        # message and does nothing with it.
        if found is None and isinstance(obj.recv, Exc) and name == "__init__":
            return h._new(Native("__init__", _exc_init).bind(obj.recv))
        # THE SAME ARRANGEMENT FOR A BUILTIN BASE, and for the same reason:
        # the chain above `class M(dict)` is a KIND rather than a class, so
        # the walk finds nothing and `object.__init__` would accept the
        # arguments and drop them. See `_builtin_init`.
        if found is None and name == "__init__"                 and isinstance(obj.recv, Instance) and obj.recv.held is not None:
            return h._new(Native("__init__", _builtin_init).bind(obj.recv))
        # `super().__new__(cls, x)` PAST A BUILTIN BASE, which is the only way
        # to build an immutable one. A tuple's contents cannot be set after
        # the fact, so `class P(tuple)` has to fill it here or never -- and
        # `object.__new__` below would answer a bare instance with an EMPTY
        # tuple inside, which is a wrong value rather than an error.
        #
        # NOT BOUND. `__new__` is an implicit staticmethod: the class arrives
        # as an ordinary first argument that the caller writes out, and
        # binding the receiver in front of it would land every argument one
        # place late.
        if found is None and name == "__new__" and isinstance(obj.frm, Class) \
                and obj.frm.builtin_kind() is not None:
            def _builtin_new(cls, *rest):
                made = Instance(cls, h)
                kind = cls.builtin_kind() if isinstance(cls, Class) else None
                if kind is not None:
                    made.held = kind(*[_content_of(one) for one in rest]) \
                        if rest else kind()
                return made

            return h._new(Native("__new__", _builtin_new))
        if found is None:
            # THE BASE CHAIN HAS RUN OUT, which means `object` -- and every
            # class has one. `super().__init__()` inside a class with no
            # explicit base is ordinary Python and was an AttributeError.
            found = _object_default(h, name)
            if found is None:
                return h._fail("AttributeError",
                               f"'super' object has no attribute '{name}'")
        # A NATIVE BINDS TOO. `object`'s defaults take the receiver as their
        # first argument exactly as a written method does, so leaving one
        # unbound called it with nothing.
        #
        # EXCEPT `__new__`, which is an implicit STATICMETHOD: it receives the
        # class as an ordinary first argument that the caller writes out --
        # `super().__new__(cls)`. Binding it put the receiver in front of that
        # and every argument landed one place late.
        if name == "__new__":
            return h._value(found)
        return h._new(found.bind(obj.recv)) \
            if isinstance(found, (Func, Native)) else h._value(found)
    if isinstance(obj, Gen):
        # WHICH `def` IT CAME FROM. `g.__name__` is the plain name and
        # `g.__qualname__` the dotted one -- `C.m` for a method's generator,
        # `f.<locals>.<genexpr>` for an expression's -- which is also what
        # the repr prints. Recorded on the STEP function, because the
        # generator cell is a frame and the frame's code is the step.
        if name in ("__name__", "__qualname__"):
            step = obj.step
            if not isinstance(step, Func):
                return h._no_attr(obj, name)
            if name == "__qualname__" and step.qualname is not None:
                return h._new(step.qualname)
            return h._new(step.name)
        # WHERE THE BODY IS, which is the whole of a generator's
        # introspection and was missing entirely -- so `dir(g)` could not
        # honestly list a single one of these and did not list anything.
        #
        # `gi_running` is True only WHILE the frame is executing, which a
        # program outside it can only see through a callback the body made;
        # `gi_suspended` is True between the first `next` and the last, and
        # False both before it has started and after it has finished. The
        # state is 0, k and -1 for those three, and the two questions are
        # different halves of it.
        #
        # A PLAIN GENERATOR ONLY, which is why all five are gated. A
        # coroutine and an async generator are this same cell, and CPython
        # gives them the same facts under `cr_` and `ag_`, which are not
        # these names: `c().gi_code` is an AttributeError where `c().cr_code`
        # is the code. `not coro` is what `_apy_inspect_isgenerator` already
        # reads, because an async generator carries both flags.
        if not obj.coro:
            if name == "gi_running":
                return h._bool(obj.running)
            if name == "gi_suspended":
                return h._bool(obj.state > 0)
            # WHAT A `yield from` IS DELEGATING TO, or None. Recorded by the
            # lowered loop, because the delegate otherwise lives in a frame
            # slot nothing outside the generator can name.
            if name == "gi_yieldfrom":
                return h._value(obj.yieldfrom)
            # THE CODE OF THE `def` IT CAME FROM. The step function IS that
            # code here -- the generator cell is the frame and the step is
            # what the frame runs -- so this is `step.__code__` under the
            # name a generator gives it.
            if name == "gi_code":
                came = obj.sig if obj.sig is not None else obj.step
                if not isinstance(came, Func):
                    return h._no_attr(obj, name)
                return h._new(_func_code(h, came))
            # AND THE FRAME ITSELF, which is None once the body has
            # finished: CPython drops the frame at that point, and
            # `g.gi_frame is None` is how a program asks whether a generator
            # is spent.
            if name == "gi_frame":
                if obj.state < 0:
                    return h._none
                made = _gen_frame(h, obj)
                return h._none if made is None else h._new(made)
        # AND THE SAME FIVE UNDER TWO OTHER PREFIXES. A coroutine and an
        # async generator are this same cell, and CPython gives each of the
        # three its own spelling: `gi_` for a generator, `cr_` for a
        # coroutine, `ag_` for an async generator, and `dir()` over each
        # lists only its own.
        #
        # `cr_await` AND `ag_await` ARE THE DELEGATE, which is the same slot
        # a `yield from` records: what the frame is waiting on is what it is
        # waiting on, whichever keyword put it there.
        #
        # `cr_origin` IS ALWAYS None -- it is filled by coroutine origin
        # tracking, which nothing here turns on, and None is what CPython
        # answers by default.
        elif name[:3] == ("ag_" if obj.agen else "cr_"):
            rest = name[3:]
            if rest == "running":
                return h._bool(obj.running)
            if rest == "suspended":
                return h._bool(obj.state > 0)
            if rest == "await":
                return h._value(obj.yieldfrom)
            if rest == "origin" and not obj.agen:
                return h._none
            if rest == "code":
                came = obj.sig if obj.sig is not None else obj.step
                if not isinstance(came, Func):
                    return h._no_attr(obj, name)
                return h._new(_func_code(h, came))
            if rest == "frame":
                if obj.state < 0:
                    return h._none
                made = _gen_frame(h, obj)
                return h._none if made is None else h._new(made)
        # THE THREE METHODS, AS VALUES. They are dispatched by name at the
        # call site, so nothing needed a value for them -- until a program
        # asked `hasattr(g, "close")`, which every duck-typed consumer does,
        # and got False for a method it can plainly call.
        # A TASK'S OWN METHODS. A task is a generator cell like any other
        # here, so these sit beside `send` and `throw` rather than on a class
        # of their own.
        if obj.builtin == _CORO_TASK and name in (
                "cancel", "result", "done", "cancelled"):
            fn = {"cancel": _apy_task_cancel, "result": _apy_task_result,
                  "done": _apy_task_done,
                  "cancelled": _apy_task_cancelled}[name]
            return h._new(Native(
                name, lambda g, _fn=fn: h._get(_fn(h, [h._value(g)]), name)
            ).bind(obj))
        # AN ASYNC GENERATOR HAS THE OTHER THREE. CPython gives `asend`,
        # `athrow` and `aclose` to one and `send`, `throw` and `close` to a
        # coroutine and a plain generator, and neither has the other's --
        # `hasattr(a, "send")` is False. Each answers an AWAITABLE; see
        # `_apy_agen_asend`.
        if obj.agen:
            if name in ("asend", "athrow", "aclose"):
                body = {"asend": lambda g, v: h._get(
                            _apy_agen_asend(h, [h._value(g), h._value(v)]),
                            name),
                        "athrow": lambda g, e: h._get(
                            _apy_agen_athrow(h, [h._value(g), h._value(e)]),
                            name),
                        "aclose": lambda g: h._get(
                            _apy_agen_aclose(h, [h._value(g)]), name)}[name]
                return h._new(Native(name, body).bind(obj))
            return h._no_attr(obj, name)
        if name in ("send", "throw", "close"):
            body = {"send": lambda g, v: h._get(
                        _apy_gen_send(h, [h._value(g), h._value(v)]), name),
                    "throw": lambda g, e: h._get(
                        _apy_gen_throw(h, [h._value(g), h._value(e)]), name),
                    "close": lambda g: h._get(
                        _apy_gen_close(h, [h._value(g)]), name)}[name]
            return h._new(Native(name, body).bind(obj))
        return h._no_attr(obj, name)
    if isinstance(obj, Alias):
        # `list[int].__origin__` is `list` and `.__args__` is `(int,)`. A
        # program that inspects an annotation reads exactly these two.
        if name == "__origin__":
            return h._value(obj.origin)
        if name == "__args__":
            return h._new(tuple(obj.args))
        return h._no_attr(obj, name)
    if isinstance(obj, memoryview):
        # `itemsize` and `format` are the one-byte unsigned format's, which is
        # what a bytes-like source always has -- Python's own view reports the
        # same for the same reason, so these are read off it rather than
        # written out.
        # THE BUFFER'S SHAPE joins them, because a program that asks is
        # usually checking exactly that -- `m.ndim == 1` before it indexes --
        # and a slice with a step is NOT contiguous, which is the one of
        # these that is not constant. Both orders agree for a single
        # dimension, which is why C and Fortran answer alike.
        if name in ("readonly", "nbytes", "itemsize", "format", "obj",
                    "ndim", "shape", "strides", "suboffsets",
                    "c_contiguous", "f_contiguous", "contiguous"):
            return h._value(getattr(obj, name))
        # THE TWO WAYS A VIEW HANDS ITS CONTENTS OVER, and the reason a
        # program makes one at all: `mv.tobytes()` copies them out and
        # `mv.tolist()` reads them as numbers. Neither existed, so a view
        # could be indexed and sliced and never emptied.
        if name in ("tobytes", "tolist"):
            return h._new(Native(name,
                                 lambda _n=name, _o=obj: getattr(_o, _n)(),
                                 owner=obj))
        # WHAT A VIEW CARRIES BESIDE THOSE TWO. `hex`, `count` and `index`
        # read the bytes it shows -- a memoryview IS a sequence in Python --
        # and the three subscript dunders are the methods behind the `m[i]` a
        # program writes. `__delitem__` exists and always refuses, which is
        # not the same claim as having no such method.
        if name == "hex":
            return h._new(Native("hex", lambda *r, _o=obj: _o.hex(*r),
                                 ranged=True, owner=obj))
        # HANDING THE BUFFER BACK, and the `with` block that does it for a
        # program. A RELEASED VIEW STILL ANSWERS ITS METHOD NAMES -- CPython
        # raises when one is CALLED, not when it is looked up.
        if name == "__enter__":
            # THE RECEIVER'S OWN HANDLE, because identity here is handle
            # equality: `m.__enter__() is m` has to hold, and a fresh handle
            # over the same view would answer False.
            def _enter(_o=obj, _at=a[0]):
                if not _mview_live(h, _o):
                    raise _UserFailed
                return h._get(_at, "__enter__")
            return h._new(Native("__enter__", _enter, owner=obj))
        if name == "_from_flags":
            # PEP 688's private constructor. The flags pick which buffer
            # REQUEST to make, and every source this runtime can wrap is a
            # flat contiguous byte buffer that answers them all alike.
            return h._new(Native("_from_flags",
                                 lambda src, flags: memoryview(src),
                                 owner=obj))
        if name in ("release", "toreadonly", "cast", "__exit__"):
            return h._new(Native(
                name, lambda *r, _n=name, _o=obj: _mview_call(h, _o, _n, r),
                ranged=name == "__exit__", owner=obj))
        if name in ("count", "index", "__setitem__", "__delitem__",
                    "__getitem__", "__len__", "__iter__",
                    "__release_buffer__"):
            return h._new(Native(
                name, lambda *r, _n=name, _o=obj: _mview_call(h, _o, _n, r),
                owner=obj))
        if name == "__class_getitem__":
            return h._new(Native("__class_getitem__", lambda k, _o=obj: Alias(
                h._get(_apy_type_object(h, [h._new(_o)]), "__class_getitem__"),
                k if isinstance(k, tuple) else (k,)), owner=obj))
        return h._no_attr(obj, name)
    if isinstance(obj, complex):
        # `.real` and `.imag` are floats, not complexes -- `(1+2j).real` is
        # `1.0`. `h._value` would keep whatever Python's attribute gives,
        # which is already a float, so this is just the two names.
        if name in ("real", "imag"):
            return h._new(getattr(obj, name))
        return h._no_attr(obj, name)
    if isinstance(obj, (int, float)):
        # EVERY NUMBER HAS `real` AND `imag`, not only a complex one -- that
        # is what makes the numeric tower uniform. An int's imaginary part is
        # the INT zero, not the float, which `type()` on it can tell apart, so
        # Python's own attribute is used rather than a literal 0.
        if name in ("real", "imag"):
            return h._value(getattr(obj, name))
        # AND A RATIONAL'S TWO HALVES, which an int carries because it IS one
        # in lowest terms -- `numbers.Rational` is the rung of the tower that
        # names them, and a float is not on it.
        if isinstance(obj, int) and name in ("numerator", "denominator"):
            return h._value(getattr(obj, name))
    if isinstance(obj, Func):
        # WHAT A PROGRAM PUT THERE WINS over the built-in attributes below,
        # as it does in the C: `f.__name__ = 'other'` has to read back as it
        # was written, not as the function was defined.
        if obj.dict is not None and name in obj.dict:
            return h._value(obj.dict[name])
        if name == "__code__":
            return h._new(_func_code(h, obj))
        # `f.__defaults__` is the POSITIONAL defaults as a tuple and
        # `__kwdefaults__` the keyword-only ones as a dict -- each None rather
        # than empty when there are none, which is how a program tells "no
        # defaults" from "a default that is falsey". They are stored as one
        # trailing run, keyword-only last.
        if name in ("__defaults__", "__kwdefaults__"):
            nd = len(obj.defaults or ())
            # THE RECORDED COUNT, not the number of keyword-only parameters.
            at_kw = min(getattr(obj, "nkwdefault", 0) or 0, nd)
            npos = nd - at_kw
            declared = obj.arity - (1 if obj.vararg else 0)                 - (1 if obj.kwarg else 0)
            if name == "__defaults__":
                return h._new(tuple(obj.defaults[:npos])) if npos > 0                     else h._none
            if at_kw <= 0 or not obj.pnames:
                return h._none
            return h._new({obj.pnames[declared - at_kw + i]:
                           obj.defaults[npos + i] for i in range(at_kw)})
        if name == "__qualname__":
            # PEP 3155. The frontend's key is already the qualified name.
            return h._value(obj.qualname if obj.qualname is not None
                            else obj.name)
        if name == "__module__":
            # WHERE THE `def` WAS WRITTEN. `builtins` for a builtin thunk or
            # a builtin type name, the recorded module for a spliced `def`,
            # and `__main__` for a program's own -- which is the default
            # rather than an absence, because every `def` a program writes is
            # in a module and there is only one it can be in. The C twin is
            # the `__module__` arm of `apy_kind_attr`.
            if obj.builtin or obj.is_type:
                return h._new("builtins")
            return h._new(obj.module if obj.module is not None
                          else "__main__")
        if name in ("__name__", "__qualname__"):
            # No qualified name is recorded -- a nested `def` knows its own
            # name and not its enclosing scope's -- so the plain one is what
            # there is.
            return h._new(obj.name)
        # `list.append.__objclass__` is `list` -- the type a DESCRIPTOR came
        # off, which is how `inspect` and every `functools` wrapper finds out
        # what a method belongs to. Absent on anything else, as in CPython.
        if name == "__objclass__":
            owner = ((obj.qualname or "").split(".")[0]
                     if getattr(obj, "descr", False) else "")
            found = _KIND_BY_NAME.get(owner)
            if found is None:
                return h._no_attr(obj, name)
            return _apy_kind_class(h, [h._new(found())])
        # `m.__self__` is the RECEIVER of a bound method, and its absence is
        # how a program tells a bound method from a plain function.
        if name == "__self__":
            if obj.bound is None:
                return h._no_attr(obj, name)
            return h._value(obj.bound)
        # `m.__func__` is the UNDERLYING function of a bound method -- the one
        # the class holds, without the receiver.
        if name == "__func__":
            # THE OBJECT THE CLASS HOLDS, looked up by the method's own name
            # -- not a copy. `c.m.__func__ is C.m` is the identity that says
            # what `__func__` means, and a fresh binding answers False to it.
            if obj.bound is None:
                return h._no_attr(obj, name)
            if isinstance(obj.bound, Instance):
                found = obj.bound.cls.find(obj.name)
                if found is not None:
                    return h._value(found)
            return h._no_attr(obj, name)
        if name == "__doc__":
            # A TYPE OBJECT'S IS ITS KIND'S. `str.__doc__` and `"".__doc__`
            # are the same text in Python, and this is the half that reaches
            # it through the name the type carries.
            if obj.doc is None and getattr(obj, "is_type", False):
                if obj.name in KIND_DOC:
                    return h._value(KIND_DOC[obj.name])
            return h._value(obj.doc)
        # PEP 649: `__annotations__` is BUILT ON ACCESS, by the thunk the
        # `def` recorded. Evaluating them at the `def` would make
        # `def f(x: Undefined)` an error where Python accepts it -- only
        # reading them is. A function with none answers the empty dict.
        if name == "__annotate__":
            return h._value(obj.annotate)
        if name == "__annotations__":
            if obj.annotate is None:
                return h._new({})
            return _user(h, lambda: h._value(h._invoke(obj.annotate, [])))
        return h._no_attr(obj, name)
    if isinstance(obj, Native) and name in ("__name__", "__qualname__",
                                            "__self__", "__module__",
                                            "__objclass__"):
        # A BUILTIN METHOD REACHED AS A VALUE, which had none of these here
        # while both compiled runtimes answered them from the FUNC arm.
        # CPython splits the four by whether there is a receiver:
        # `list.append` is a method_descriptor with no `__module__` at all,
        # `[].append` a builtin_function_or_method whose `__module__` is
        # None. The name is the method's own either way.
        #
        # NOT A FULL ARM: anything else about a Native -- `__class__` above
        # all -- is answered by the generic paths below, and swallowing the
        # rest here would cut this kind off from them.
        # THE RECEIVER TRAVELS IN `owner` for a method built from the kind
        # table -- its body closes over the value, so `bound` is empty -- and
        # in `bound` for one `apy_bind` made. Either is a receiver.
        held = obj.bound if obj.bound is not None else (
            None if obj.owner is _NO_OWNER else obj.owner)
        if name == "__name__":
            return h._new(obj.name)
        if name == "__qualname__":
            # QUALIFIED BY THE TYPE IT WAS DEFINED ON, which is the same text
            # bound or unbound: `[].append.__qualname__` and
            # `list.append.__qualname__` are both `list.append` in CPython,
            # because binding does not change where a method was defined.
            owner = getattr(obj, "descr", "") or (
                h.kind_name(held) if held is not None else "")
            return h._new(f"{owner}.{obj.name}" if owner else obj.name)
        if name == "__objclass__":
            # `list.append.__objclass__` is `list` -- the type a DESCRIPTOR
            # came off, and absent on anything else as it is in CPython.
            found = _KIND_BY_NAME.get(getattr(obj, "descr", ""))
            if found is None:
                return h._no_attr(obj, name)
            return _apy_kind_class(h, [h._new(found())])
        if name == "__self__":
            # A DESCRIPTOR HAS NONE: it was reached off the type, so there is
            # no receiver to answer with.
            if getattr(obj, "descr", "") or held is None:
                return h._no_attr(obj, name)
            return h._value(held)
        return h._none if held is not None else h._no_attr(obj, name)
    if isinstance(obj, Exc):
        # WHAT THE PROGRAM STORED WINS over what the kind offers.
        # `value`, `message` and `exceptions` are answered for every
        # exception here -- one cell serves them all -- but in CPython
        # they belong to StopIteration and ExceptionGroup alone, so a
        # class of its own setting `self.value` owns that name and
        # nothing should take it. It did: `raise _Returned(42)` then
        # `caught.value` gave back the MESSAGE rather than the 42, which
        # is a wrong answer and not a missing feature.
        #
        # THE DUNDERS AND `args` ARE NOT IN THIS, because those really
        # are BaseException's and writing one means writing through it.
        if name in obj.dict and not name.startswith('_') \
                and name != 'args':
            return h._value(obj.dict[name])
        # `g.exceptions` -- what an `ExceptionGroup` carries. Absent on an
        # ordinary exception, which is how a program tells the two apart
        # without asking about the type.
        # `g.message` -- the text an ExceptionGroup was built with, which is
        # its FIRST argument and separate from the exceptions it carries.
        # Present only on a group, like `exceptions`.
        if name == "message":
            if obj.subs is None:
                return h._no_attr(obj, name)
            return h._value(obj.arg if obj.has_arg else "")
        if name == "exceptions":
            if not getattr(obj, "subs", None):
                return h._no_attr(obj, name)
            return h._new(list(obj.subs))
        if name == "args":
            if getattr(obj, "argv", None) is not None:
                return h._new(tuple(obj.argv))
            return h._new(() if not obj.has_arg else (obj.arg,))
        # PEP 3151: `OSError(2, "No such file")` READS BOTH BACK. They are
        # positions in `args`, not fields, which is why they are answered from
        # it rather than stored twice.
        if name in ("errno", "strerror"):
            argv = getattr(obj, "argv", None) or ()
            at = 0 if name == "errno" else 1
            return h._value(argv[at]) if len(argv) > at else h._none
        # Both are None when unset, never absent: `e.__cause__` is an
        # attribute every exception has, and code that reads it to decide
        # whether to print "the direct cause" would get an AttributeError.
        # `e.value` -- what a generator's `return` gave. Every exception has
        # it in CPython, so answering None keeps the bare form working too.
        if name == "value":
            return h._value(obj.arg if obj.has_arg else None)
        if name == "__context__":
            return h._value(obj.context)
        if name == "__cause__":
            return h._value(obj.cause)
        if name == "__suppress_context__":
            return h._bool(obj.suppress)
        if name == "__notes__":
            if obj.notes is None:
                return h._no_attr(obj, name)
            return h._value(obj.notes)
        # There is no traceback OBJECT -- no frames are recorded -- but an
        # exception that was raised HAS one, and `e.__traceback__ is not None`
        # is the test programs actually write. An empty tuple is the least
        # dishonest stand-in.
        if name == "__traceback__":
            # A REAL TRACEBACK where the position table exists, and the old
            # empty-tuple stand-in where it does not -- a program that never
            # asks about positions gets none recorded, and `e.__traceback__ is
            # not None` still has to answer True.
            if getattr(obj, "pos", -1) >= 0:
                return h._new(_traceback_of(h, obj))
            # NEVER RAISED, so there is nothing to point at -- which is what
            # CPython answers, and how a program tells a caught exception from
            # one it merely built.
            if h.positions:
                return h._none
            # No positions recorded at all: the old stand-in, which is not
            # None because an exception that WAS raised has a traceback and
            # nothing here can tell the two apart without them.
            return h._new(())
        if name == "__class__":
            return h._new(h._type_of(obj))
        # THIS EXCEPTION'S OWN ATTRIBUTES, then its class's -- the ordinary
        # two-step, arriving late because the fixed names above are what
        # BaseException itself defines and a class body cannot shadow.
        if name in obj.dict:
            return h._value(obj.dict[name])
        if obj.cls is not None:
            found = obj.cls.find(name)
            if found is not None:
                if isinstance(found, (Func, Native)):
                    return h._new(found.bind(obj))
                # A DESCRIPTOR IS READ THROUGH, exactly as it is on an
                # `Instance`. Without this a `@property` on an exception
                # subclass handed the program the `Descr` OBJECT --
                # `subprocess.CalledProcessError.stdout` printed
                # `<...objects_host.Descr object at 0x...>` where CPython
                # prints the output, which is a wrong answer rather than a
                # missing feature. `classmethod` and `staticmethod` were
                # wrong the same way.
                if isinstance(found, Descr) or (
                        isinstance(found, Instance)
                        and found.cls.find("__get__") is not None):
                    return _descr_get(h, found, obj, obj.cls)
                return h._value(found)
        return h._no_attr(obj, name)
    return h._no_attr(obj, name)


def _apy_default_repr(h, a):
    """`object.__repr__(x)` -- what a `__repr__` override overrode."""
    v = h._get(a[0], "apy_default_repr")
    if not isinstance(v, Instance):
        return h._new(h._text(v, True))
    return h._new(_inst_repr(v))


def _apy_default_eq(h, a):
    """IDENTITY, and `__hash__` agrees with it. That pairing is the contract:
    two objects that compare equal must hash equally, and the default
    satisfies it by comparing nothing but identity."""
    return h._bool(h._get(a[0], "apy_default_eq")
                   is h._get(a[1], "apy_default_eq"))


def _apy_default_hash(h, a):
    return h._int(id(h._get(a[0], "apy_default_hash")) >> 3)


def _apy_default_init(h, a):
    """`object.__init__(self)` does nothing: a subclass calling it is saying
    the base has no state to set up."""
    return h._none


def _apy_setattr(h, a):
    """`x.name = v`. `__setattr__` INTERCEPTS EVERY assignment, the mirror of
    `__getattribute__` -- and the default stays callable from within the
    override, which is the only way an override can actually store."""
    obj = h._get(a[0], "apy_setattr")
    if isinstance(obj, Instance) and obj.cls.find("__setattr__") is not None:
        return _user(h, lambda: h._value(obj._send(
            "__setattr__", str(h._get(a[1], "apy_setattr")),
            h._get(a[2], "apy_setattr"))))
    return _apy_default_setattr(h, a)


def _slot_declares(cls, name: str) -> bool:
    """Is `name` DECLARED in this class's `__slots__`, or an ancestor's?

    NOT `_slot_allows`, which answers a DIFFERENT QUESTION: that one says
    whether an INSTANCE may carry this attribute at all, and answers yes for
    every name the moment one class in the chain omits `__slots__`. Right for
    a write, wrong for deciding whether reading the name off the CLASS gives
    a `member_descriptor`. See `objects/c/_descriptors.py`'s
    `apy_slot_declares`, which is the same walk and was the same mistake.
    """
    here = cls
    while isinstance(here, Class):
        declared = here.dict.get("__slots__")
        if declared is not None:
            # A BARE STRING IS ONE SLOT, not a run of one-character ones.
            names = [declared] if isinstance(declared, str) else None
            if names is None:
                try:
                    names = list(declared)
                except TypeError:
                    names = []
            if name in names:
                return True
        here = here.base
    return False


def _slot_allows(cls, name: str) -> bool:
    """Is this instance restricted to a fixed set of attributes, and is `name`
    one of them?

    An instance is unrestricted unless EVERY class in its chain declares
    `__slots__` -- one that does not gives the dict back, and with it the
    freedom to set anything.
    """
    here = cls
    while isinstance(here, Class):
        if "__slots__" not in here.dict:
            return True                        # no `__slots__`: a dict
        here = here.base
    here = cls
    while isinstance(here, Class):
        declared = here.dict.get("__slots__")
        if isinstance(declared, str):
            if declared == name:
                return True
        elif declared is not None:
            try:
                if name in declared:
                    return True
            except TypeError:
                return True
        here = here.base
    return False


def _apy_default_setattr(h, a):
    obj = h._get(a[0], "apy_setattr")
    name = str(h._get(a[1], "apy_setattr"))
    if isinstance(obj, Instance) and not _slot_allows(obj.cls, name):
        return h._fail("AttributeError",
                       f"'{h.kind_name(obj)}' object has no attribute "
                       f"'{name}' and no __dict__ for setting new attributes")
    value = h._get(a[2], "apy_setattr")
    if isinstance(obj, Instance):
        # A DATA DESCRIPTOR ON THE CLASS TAKES THE WRITE -- otherwise the next
        # read would find the stored value and the property would never be
        # consulted again.
        klass_d = obj.cls.find(name)
        if klass_d is not None:
            handled = _descr_set(h, klass_d, obj, value)
            if handled == 0:
                return 0
            if handled == 1:
                return h._none
        _attr_store(h, obj.dict, name, value)
        return h._none
    if isinstance(obj, Exc):
        # `self.code = code` in a user exception's `__init__`.
        _attr_store(h, obj.dict, name, value)
        return h._none
    if isinstance(obj, Class):
        # `C.__name__ = ...` CHANGES WHAT THE CLASS IS CALLED. The name is a
        # field on the class, not an entry in its dict, so storing it as an
        # ordinary attribute left `__name__` reading the old one -- the write
        # appeared to succeed and changed nothing.
        if name == "__name__":
            was = obj.name
            obj.name = str(value)
            _rename_exception(h, was, obj.name, obj)
            return h._none
        _attr_store(h, obj.dict, name, value)
        return h._none
    if isinstance(obj, Func):
        # A function carries whatever a program hangs on it, as it does in the
        # C -- `f.__override__ = True` is a decorator doing exactly that. The
        # dict is made on first write so an ordinary `def` costs nothing.
        if obj.dict is None:
            obj.dict = {}
        _attr_store(h, obj.dict, name, value)
        return h._none
    return h._no_attr(obj, name)


def _apy_call(h, a):
    """`a[1]` is the ADDRESS of an argument array, as `apy_print` takes one."""
    addr, n = int(a[1]), int(a[2])
    args = [h._get(h._interp.mem.read(addr + i * 8, _PTR), "apy_call")
            for i in range(n)]
    try:
        return h._value(h._invoke(h._get(a[0], "apy_call"), args))
    except _UserFailed:
        return 0


#: A slot no argument has reached yet, so the default can be told from a
#: legitimately-passed None. `f(c=3)` leaves a HOLE at `b`, and filling it
#: with None instead of `b`'s default is a different call.
_MISSING = object()


def _apy_call_kw(h, a):
    """`f(a, k=v, **d)`. The positional arguments are in the buffer; the
    keyword ones are a DICT built at the call site.

    The names are matched against the CALLEE's parameters, which is why the
    function object carries them: the callee is a value here, so which
    parameter `k=` names is known only to whatever it turns out to hold.
    """
    addr, argc = int(a[1]), int(a[2])

    def cell(i):
        return h._get(h._interp.mem.read(addr + i * 8, _PTR), "apy_call_kw")

    f = h._get(a[0], "apy_call_kw")
    args = [cell(i) for i in range(argc)]
    kwargs = dict(h._get(a[3], "apy_call_kw"))
    return _call_kwargs(h, f, args, kwargs)


def _native_kwargs(h, f, args, kwargs):
    """A BUILTIN METHOD REACHED AS A VALUE, handed keywords.

    `g = x.split` then `g(",", maxsplit=1)` is a call CPython answers, and so
    is the one the COLLISION path makes when a module defines a class
    extending a builtin. A native declares no parameters, so the binder above
    had nothing to match the names against and dropped them in silence: the
    answer came back as though the keyword had never been written.

    THE SAME TABLE THE WRITTEN SPELLING FOLDS AGAINST -- `METHOD_PARAMS`, read
    here rather than restated, so the two arrangements cannot drift. The C
    reads a generated copy of it; see `apy_kind_meth_sign`. The refusals are
    CPython's and in CPython's order: too many first, then a slot given twice,
    then a name no parameter has.
    """
    # A VARIADIC BUILTIN COLLECTS ITS KEYWORDS rather than matching them
    # against declared names: `"{a}".format(a=1)` has no parameter called
    # `a`, and the dict IS the argument. Its body takes `**named` and this
    # hands them straight over.
    if f.variadic:
        return h._value(f.body(*args, **kwargs))
    params = METHOD_PARAMS.get(f.name) if f.name in _TABLE_METHODS else None
    if params is None:
        # A METHOD WITH NO KEYWORD SIGNATURE AT ALL names its owner, which is
        # the receiver's kind: `str.upper() takes no keyword arguments`.
        who = (f.bound if f.bound is not None
               else f.owner if f.owner is not _NO_OWNER
               else args[0] if args else None)
        # `d.update(a=1)` IS THE EXCEPTION: the keywords ARE the value, and
        # no signature can say that -- any name at all becomes a key. A SET'S
        # `update` really does take no keyword, so this is the dict's alone.
        if f.name == "update" and isinstance(who, dict):
            for one in args:
                if not isinstance(one, (dict, _VIEW_TYPES)):
                    return h._fail("TypeError",
                                   f"'{h.kind_name(one)}' object is not a "
                                   f"mapping")
                who.update(one)
            who.update(kwargs)
            return h._none
        return h._fail("TypeError", f"{h.kind_name(who)}.{f.name}() takes no "
                                    f"keyword arguments")
    named = [p for p, _ in params]
    slots = list(args[:len(params)])
    taken = [True] * len(slots) + [False] * (len(params) - len(slots))
    slots += [None] * (len(params) - len(slots))
    if len(args) + len(kwargs) > len(params):
        plural = "" if len(params) == 1 else "s"
        return h._fail("TypeError",
                       f"{f.name}() takes at most {len(params)} "
                       f"argument{plural} ({len(args) + len(kwargs)} given)")
    for name, value in kwargs.items():
        at = named.index(name) if name in named and name is not None else -1
        if at < 0:
            return h._fail("TypeError", f"{f.name}() got an unexpected "
                                        f"keyword argument {name!r}")
        if taken[at]:
            return h._fail("TypeError",
                           f"argument for {f.name}() given by name "
                           f"({name!r}) and position ({at + 1})")
        slots[at], taken[at] = value, True
    top = max((i + 1 for i, got in enumerate(taken) if got), default=0)
    for i in range(len(params)):
        if taken[i]:
            continue
        _, default = params[i]
        if default is REQUIRED:
            if i < top:
                return h._fail("TypeError",
                               f"{f.name}() takes at least {i + 1} positional "
                               f"argument{'' if i == 0 else 's'} "
                               f"({len(args)} given)")
            default = None
        # PADDED TO THE FULL ARITY, which is what the frontend's own fold does
        # for the written spelling: the entry point takes every parameter and
        # the defaults supply the rest.
        slots[i] = default
    try:
        # `bound` SAYS THE NAMES ARE ALREADY IN THEIR SLOTS, which is what
        # keeps the positional bound from being re-applied to a call that
        # filled those slots by name. See `_meth_positional`.
        return h._value(h._invoke(f, slots, bound=True))
    except _UserFailed:
        return 0


def _call_kwargs(h, f, args, kwargs):
    """One call whose keywords are matched by NAME against the callee.

    Split out of `apy_call_kw` because `apy_class_build_kw` needs exactly this
    and reads its arguments from somewhere else: `class C(metaclass=M,
    kind="x")` is `M(name, bases, ns, kind="x")`, and the names have to land
    on `M.__new__`'s parameters the same way any other call's do.
    """
    if isinstance(f, Native) and kwargs:
        return _native_kwargs(h, f, args, kwargs)
    target, skip = f, 0
    if isinstance(f, Class):
        # `__new__` DECLARES THE KEYWORDS when a class writes one and leaves
        # `__init__` to the default -- which is exactly a metaclass taking
        # class keywords: `M.__new__(mcls, name, bases, ns, kind=None)` with
        # `type.__init__` behind it. Reading the names off `__init__` there
        # matched them against a native that declares none.
        target, skip = f.find("__init__"), 1
        if not isinstance(target, Func):
            maker = f.find("__new__")
            if isinstance(maker, Func):
                target = maker
    elif isinstance(f, Instance):
        target, skip = f.cls.find("__call__"), 1
    elif isinstance(f, Func) and f.bound is not None:
        skip = 1
    if not isinstance(target, Func):
        # No signature to match against. `_invoke` words both the missing
        # `__init__` and the non-callable, so let it.
        #
        # WITH THE KEYWORDS, which this used to drop: `class C(dict): pass`
        # then `C(a=1)` has no `__init__` to match against and arrived here,
        # so the names went nowhere and the instance came back with an EMPTY
        # dict and no error -- a wrong answer where a refusal was intended.
        try:
            return h._value(h._invoke(f, args, kwrest=kwargs))
        except _UserFailed:
            return 0
    declared = (target.arity - (1 if target.vararg else 0)
                - (1 if target.kwarg else 0))
    want = max(0, declared - skip)
    # WHERE POSITIONS STOP. Names reach all of `want`; positions reach only
    # as far as the keyword-only tail, which is the whole of `*` in a
    # signature.
    bypos = max(0, want - target.kwonly)
    # PADDED TO `want`, so a keyword-only parameter nobody named still has a
    # slot for its default. Sizing it from the highest keyword bound left the
    # tail off entirely.
    slots = list(args[:bypos])
    slots += [_MISSING] * max(0, want - len(slots))
    # The surplus belongs to `*rest`. It is kept OUT of `slots` so it cannot
    # overwrite the keyword-only tail, and appended after them below -- which
    # is the layout `_invoke(bound=True)` reads.
    extra = args[bypos:]
    rest = {} if target.kwarg else None
    # NAMED BY ITS QUALNAME in every refusal below -- `K.m()`,
    # `outer.<locals>.inner()` -- which is what CPython prints and what the
    # arity messages already say.
    who = target.qualname or target.name
    # POSITIONAL-ONLY NAMES PASSED BY KEYWORD ARE GATHERED, all of them at
    # once: CPython lists every offender in ONE refusal (`'a, b'`) and says
    # it BEFORE any other keyword complaint -- `pos(a=1, zz=2)` names `a` and
    # never mentions `zz`. Scanned in DECLARATION order, which is the order
    # CPython prints and not the order the call wrote; refusing at the first
    # match named one of them, and named whichever the CALL put first.
    #
    # SKIPPED WHEN THERE IS A `**kw`. The name is not refused then, it lands
    # in the collection: `def f(a, /, **kw)` called `f(1, a=2)` binds
    # `kw = {'a': 2}`.
    if rest is None and target.posonly and kwargs:
        hit = [n for n in target.pnames[skip:target.posonly]
               if n and n in kwargs]
        if hit:
            return h._fail("TypeError",
                           f"{who}() got some positional-only arguments "
                           f"passed as keyword arguments: '{', '.join(hit)}'")
    # COUNTED FOR THE REFUSAL AND NOTHING ELSE. A surplus positional
    # alongside these is worded by CPython as `3 positional arguments (and 1
    # keyword-only argument)`, and this loop is the only place that can tell
    # a keyword-only landing from any other.
    kwonly_given = 0
    for name, value in kwargs.items():
        try:
            at = target.pnames.index(name) - skip
        except ValueError:
            at = -1
        # POSITIONAL-ONLY: the name is recorded so the message below can be
        # specific, but it does not match.
        posonly_hit = 0 <= at + skip < target.posonly
        if posonly_hit:
            at = -1
        # A `*rest` OR `**kw` PARAMETER CANNOT BE FILLED BY NAME. The name
        # belongs to the COLLECTION and not to a slot, so a keyword spelling
        # either of them goes into `**kw` like any other unmatched name --
        # and both sit PAST the declared positions, which is what tells them
        # apart. `def f(*a, **kw)` called as `f(a=1)` bound the tuple to `1`
        # and then tried to walk it; the compiled runtimes collect it.
        if at >= want:
            at = -1
        # NOT `or extra`. Surplus positionals belong to `*rest`; they do
        # not stop a keyword from naming a parameter, and least of all a
        # keyword-only one, which no position could have filled.
        if at < 0:
            if rest is None:
                # THE POSITIONAL-ONLY CASE NEVER REACHES HERE. The gather
                # above owns it, and owns every name at once; by this point
                # the only keyword still unplaced is one the signature does
                # not declare at all.
                return h._fail("TypeError",
                               f"{who}() got an unexpected keyword "
                               f"argument '{name}'")
            rest[name] = value
            continue
        if 0 <= at < len(slots) and slots[at] is not _MISSING:
            return h._fail("TypeError", f"{who}() got multiple values "
                                        f"for argument '{name}'")
        while len(slots) <= at:
            slots.append(_MISSING)
        slots[at] = value
        if at >= bypos:
            kwonly_given += 1
    # A SURPLUS POSITIONAL IS AN ERROR, not something to hand on. Appending
    # it to `slots` sent it to `_invoke`, whose packing CAPS what it copies
    # and dropped it in silence -- and by then `slots` is the width of the
    # array the NAMES filled, so nothing downstream could say how the call
    # was written. `*rest` is what a surplus is for; anything else refuses.
    if extra and not target.vararg:
        _fn_arity_failure(h, target, skip + len(args), kwonly_given)
        return 0
    # EVERY HOLE AT ONCE, and the two kinds of hole kept apart. CPython names
    # all the parameters a call left unfilled in one refusal -- `'a' and
    # 'c'`, and the holes need not be adjacent -- where refusing at the first
    # meant a caller who forgot two learned about one, fixed it, and came
    # straight back. Keyword-only holes are a SEPARATE message, said only
    # when no positional one is outstanding: no position could have filled
    # one, so counting it among the positional arguments sends the reader to
    # the wrong half of the signature.
    #
    # GATHERED BEFORE ANY DEFAULT IS WRITTEN, because the fill is what hides
    # them: a hole a default covers stops being one.
    miss, kwmiss = [], []
    for i, value in enumerate(slots):
        if value is not _MISSING:
            continue
        if _fn_default(target, i + skip) is not _NO_DEFAULT:
            continue
        (kwmiss if i >= bypos else miss).append(target.pnames[i + skip])
    if miss or kwmiss:
        names = miss or kwmiss
        kind = "positional" if miss else "keyword-only"
        return h._fail("TypeError",
                       f"{who}() missing {len(names)} required {kind} "
                       f"argument{'' if len(names) == 1 else 's'}: "
                       f"{_name_list(names)}")
    for i, value in enumerate(slots):
        if value is not _MISSING:
            continue
        slots[i] = _fn_default(target, i + skip)
    try:
        return h._value(h._invoke(f, slots + extra, kwrest=rest, bound=True))
    except _UserFailed:
        return 0


# ── builtins over sequences ─────────────────────────────────────────────────
# Each of these is the Python operation on the real object, which is the whole
# reason the handle table holds real objects: `sorted` here IS `sorted`, so
# stability, the ordering rules and the TypeError all come from CPython rather
# than from a restatement of them that could drift from the C.

def _gen_cache(h, g):
    """A generator's elements, drained once. See `Gen.cache`."""
    if g.cache is None:
        got = _apy_gen_drain(h, [h._value(g)])
        if not got:
            return None
        g.cache = h._get(got, "apy_gen_drain")
    return g.cache


#: Python's own dict views. `d.keys()` answers one of these rather than a
#: class of this file's: they are already live, already set-like, and already
#: iterate the way the C's do, so the two cannot drift.
_VIEW_TYPES = (type({}.keys()), type({}.values()), type({}.items()))

#: Distinguishes "the user's method failed" from "it answered handle 0".
#: `_user` defaults to 0 on failure, and 0 is a real handle.
_FAILED = object()


def _seq_items(h, v, where: str):
    """The elements of a sequence, or the KEYS of a dict -- what `apy_key_at`
    walks, so that every consumer of a container agrees on what iterating it
    yields."""
    if isinstance(v, Iterator):
        # CONSUMING, and from the current position -- the same rule the C's
        # `apy_step` follows. A consumer that read from zero would replay what
        # the cursor had already yielded and leave it unconsumed. Walked
        # through the protocol rather than reached into, because a `map`
        # cursor's elements do not exist until it is stepped.
        rest = _drain_cursor(h, v)
        if rest is None:
            return None
        v.i = len(rest)
        return list(rest)
    if isinstance(v, Gen):
        got = _gen_cache(h, v)
        return None if got is None else list(got)
    if isinstance(v, dict):
        return list(v)
    if isinstance(v, (list, tuple, set, frozenset, str, bytes, bytearray,
                      range, memoryview)):
        # A MEMORYVIEW IS ITERABLE and yields ints, which every eager consumer
        # here expects and this funnel did not list.
        return list(v)
    if isinstance(v, _VIEW_TYPES):
        # READ WHEN WALKED, which is what makes a view live: the keys are the
        # ones the dict has now, not the ones it had when the view was made.
        return list(v)
    if isinstance(v, Class):
        # ITERATING A CLASS IS THE METACLASS'S BUSINESS: `for c in Color` is
        # `type(Color).__iter__(Color)`, which is how an enum lists its
        # members. `_apy_iterable` has known that all along; this funnel did
        # not, so `sorted(Color)` and `",".join(Color)` refused what a `for`
        # loop over the very same class walked.
        walked = _apy_iterable(h, [h._new(v)])
        if walked == 0:
            return None
        got = h._get(walked, where)
        if got is v:
            # No metaclass, or one that says nothing about walking -- and
            # the refusal below is still the right answer for it.
            h._fail("TypeError", f"'{h.kind_name(v)}' object is not iterable")
            return None
        return _seq_items(h, got, where)
    if isinstance(v, Instance):
        # THROUGH `_apy_iterable`, NOT A SECOND COPY OF THE RULES. It already
        # knows the whole protocol -- `__iter__`, the iterator check, the
        # `__len__`-bounded `__getitem__` walk -- and restating any of it here
        # is how the two drift. They did: one copy accepted a non-iterator.
        walked = _apy_iterable(h, [h._new(v)])
        if walked == 0:
            return None
        got = h._get(walked, where)
        # `_apy_iterable` answers the object unchanged when the index walk is
        # already the right thing, which for an Instance means `__len__` plus
        # `__getitem__`.
        if got is v:
            limit = _user(h, lambda: v._send("__len__"), fail=_FAILED)
            if limit is _FAILED:
                return None
            out = []
            for i in range(int(limit)):
                item = _apy_getitem(h, [h._new(v), h._int(i)])
                if item == 0:
                    return None
                out.append(h._get(item, where))
            return out
        # WHATEVER `__iter__` GAVE, ASKED AGAIN. It may be a cursor or a
        # generator, and handing one back as if it were the elements left
        # every caller holding a class of this file's -- which Python's own
        # `for` cannot walk, so `bytearray().extend(obj)` reported
        # `'Iterator' object is not iterable` about a class that plainly is.
        if isinstance(got, (list, tuple)):
            return list(got)
        return _seq_items(h, got, where)
    h._fail("TypeError", f"'{h.kind_name(v)}' object is not iterable")
    return None


def _driving(h, symbol, run):
    """`run()`, with `symbol` recorded as the comparison being driven.

    EVERY EAGER CONSUMER THAT COMPARES SAYS WHICH OPERATOR IT USES, because
    the refusal a program sees names that one -- `sorted` and `min` name `<`
    and `max` names `>` -- and the dunder Python reaches cannot say which,
    since the mirror of one is the other. See `_make_order`.
    """
    was = h.order_driver
    h.order_driver = symbol
    try:
        return run()
    finally:
        h.order_driver = was


def _apy_sorted(h, a):
    # ASKED FIRST, ON THE ARGUMENT, before anything drains it. See
    # `_apy_length_hint`.
    if not _apy_length_hint(h, a[:1]):
        return 0
    items = _seq_items(h, h._get(a[0], "apy_sorted"), "apy_sorted")
    if items is None:
        return 0
    try:
        return h._new(_driving(h, "<", lambda: sorted(items)))
    except TypeError as exc:
        return h._fail_like(exc)


def _apy_min(h, a):
    items = _seq_items(h, h._get(a[0], "apy_min"), "apy_min")
    if items is None:
        return 0
    try:
        return h._value(_driving(h, "<", lambda: min(items)))
    except _HOST_RAISES as exc:
        return h._fail_like(exc)


def _apy_max(h, a):
    items = _seq_items(h, h._get(a[0], "apy_max"), "apy_max")
    if items is None:
        return 0
    try:
        return h._value(_driving(h, ">", lambda: max(items)))
    except _HOST_RAISES as exc:
        return h._fail_like(exc)


def _apy_sum(h, a):
    items = _seq_items(h, h._get(a[0], "apy_sum"), "apy_sum")
    if items is None:
        return 0
    try:
        return h._value(sum(items))
    except TypeError as exc:
        return h._fail_like(exc)


def _apy_sum_from(h, a):
    # `sum` REFUSES STRINGS, and it is not an oversight in CPython: joining
    # this way is quadratic and `''.join(...)` is the answer. Python's own
    # `sum` refuses too, but with `start` first in the message -- the C names
    # the sequence type, so the check is made here to keep the two identical.
    start = h._get(a[1], "apy_sum_from")
    if isinstance(start, str):
        return h._fail("TypeError",
                       "sum() can't sum strings [use ''.join(seq) instead]")
    if isinstance(start, (bytes, bytearray)):
        return h._fail("TypeError",
                       "sum() can't sum bytes [use b''.join(seq) instead]")
    items = _seq_items(h, h._get(a[0], "apy_sum_from"), "apy_sum_from")
    if items is None:
        return 0
    try:
        return h._value(sum(items, h._get(a[1], "apy_sum_from")))
    except TypeError as exc:
        return h._fail_like(exc)


def _argv(h, a, where: str, n: int, at: int = 1):
    """The values in a stack array, as the C's `apy_value *` argument is."""
    addr = int(a[at])
    return [h._get(h._interp.mem.read(addr + i * 8, _PTR), where)
            for i in range(n)]


def _apy_extreme_n(h, a):
    """`min(a, b, ...)` -- the MULTI-ARGUMENT form, which compares the
    arguments themselves rather than the elements of one of them."""
    args = _argv(h, a, "apy_extreme_n", int(a[1]), 0)
    if not args:
        return h._fail("TypeError", "min expected at least 1 argument")
    try:
        return h._value((max if int(a[2]) else min)(args))
    except _HOST_RAISES as exc:
        return h._fail_like(exc)


def _apy_extreme_or(h, a):
    """`min(xs, default=v)`. Only an EMPTY iterable reaches the default."""
    items = _seq_items(h, h._get(a[0], "apy_extreme_or"), "apy_extreme_or")
    if items is None:
        return 0
    if not items:
        return a[2]
    key = h._get(a[1], "apy_extreme_or")
    pick = max if int(a[3]) else min
    try:
        if key is None:
            return h._value(pick(items))
        return _user(h, lambda: h._value(
            pick(items, key=lambda v: h._invoke(key, [v]))))
    except _HOST_RAISES as exc:
        return h._fail_like(exc)


def _apy_enumerate(h, a):
    """`enumerate(xs, start)` -- LAZY, and consumed once, like the C's."""
    src = _apy_getiter(h, [a[0]])
    if not src:
        return 0
    return h._new(Iterator(h._get(src, "apy_enumerate"), None,
                           Iterator.ENUMERATE, int(a[1])))


def _apy_reversed(h, a):
    """`reversed(xs)` -- a CURSOR counting down, not a list built backwards.

    LAZY, because CPython's is: `reversed(xs)` is an iterator with no length,
    no indexing and one walk in it, and a list answered all three wrongly
    while copying the whole sequence before the first element was wanted.

    REVERSING NEEDS A SEQUENCE: a length and indexing, or a `__reversed__`. A
    set has no order, and a cursor or a generator has a POSITION rather than a
    length -- CPython refuses all three by name. The refusal is explicit
    because the walk would otherwise answer confidently: a set in an
    arbitrary order, and an iterator drained FORWARDS by a walk that believes
    it is going backwards.
    """
    v = h._get(a[0], "apy_reversed")
    # `__reversed__` WINS OVER THE INDEX WALK. A class may define both it and
    # `__getitem__`, and they need not agree -- the hook is the answer the
    # class chose.
    if isinstance(v, Instance) and v.cls.find("__reversed__") is not None:
        got = _user(h, lambda: v._send("__reversed__"), fail=_FAILED)
        if got is _FAILED:
            return 0
        return _apy_iterable(h, [h._value(got)])
    if isinstance(v, _VIEW_TYPES):
        # A VIEW IS READ NOW, as it is for a forward walk, and the copy is the
        # cursor's source from here on -- so which view it was is recorded
        # before it is gone.
        part = ("keys" if "keys" in type(v).__name__
                else "values" if "values" in type(v).__name__ else "items")
        items = list(v)
        made = Iterator(items, None, Iterator.REV, len(items) - 1)
        made.named = _CURSOR_NAMES[part][1]
        return h._new(made)
    ok = isinstance(v, (list, tuple, str, bytes, bytearray, dict, range,
                        memoryview)) or (
        # An instance with the older protocol -- or one extending a builtin,
        # whose length and indexing are the builtin's.
        isinstance(v, Instance)
        and (v.held is not None or v.cls.find("__getitem__") is not None))
    if not ok:
        return h._fail("TypeError",
                       f"'{h.kind_name(v)}' object is not reversible")
    n = _apy_raw_len(h, [a[0]])
    if h.err is not None:
        return 0
    return h._new(Iterator(v, None, Iterator.REV, int(n) - 1))


def _apy_zip_n(h, a):
    """`zip(...)` -- LAZY, stopping at the shortest. `strict` reports an
    uneven zip instead, which is the whole point of PEP 618: the lossiness is
    useful and is also the bug, so the caller says which it meant."""
    cursors = []
    for v in _argv(h, a, "apy_zip_n", int(a[1]), 0):
        got = _apy_getiter(h, [h._value(v)])
        if not got:
            return 0
        cursors.append(h._get(got, "apy_zip_n"))
    return h._new(Iterator(cursors, int(a[2]) != 0, Iterator.ZIP))


def _apy_zip2(h, a):
    return _apy_zip_n(h, [a[0], 0, 0]) if False else _zip_two(h, a)


def _zip_two(h, a):
    cursors = []
    for handle in (a[0], a[1]):
        got = _apy_getiter(h, [handle])
        if not got:
            return 0
        cursors.append(h._get(got, "apy_zip2"))
    return h._new(Iterator(cursors, False, Iterator.ZIP))


def _apy_range(h, a):
    """`range(...)` as A LAZY SEQUENCE, which is Python's own `range`.

    Materialised into a list before, so `type(range(3)).__name__` said `list`
    and `range(10**9)` would have built a billion elements. Python's `range`
    IS the object the C's `APY_RANGE_K` reproduces -- length, indexing,
    slicing, membership and equality are all arithmetic on three numbers in
    both -- so using it here is the closest the two paths can be.
    """
    start, stop, step = int(a[0]), int(a[1]), int(a[2])
    if step == 0:
        return h._fail("ValueError", "range() arg 3 must not be zero")
    return h._new(range(start, stop, step))


def _apy_abs(h, a):
    v = h._get(a[0], "apy_abs")
    if isinstance(v, Instance) and v.cls.find("__abs__") is not None:
        return _user(h, lambda: h._value(v._send("__abs__")))
    # `abs(complex)` is its MODULUS, and a float -- the one kind for which abs
    # changes the type rather than the sign.
    if not isinstance(v, (int, float, complex)):
        return h._fail("TypeError",
                       f"bad operand type for abs(): '{h.kind_name(v)}'")
    return h._value(abs(v))


def _apy_index_obj(h, a):
    """`v.__index__()` called directly -- the BOXED twin of `_apy_index`
    above, which the compiler emits for a subscript and unpacks to a machine
    word instead. `operator.index` reaches this one, and asked for it: the
    method was simply absent from `DYN_METHOD_TABLE`
    (`frontends/python/methods.py`), so `(7).__index__()` traded a working
    call for `AttributeError: 'int' object has no attribute '__index__'`.

    A BOOL ANSWERS AN int, NOT ITSELF -- `True.__index__()` is `1` and
    `type(1)` is not `bool`, same as `_apy_abs` below has to keep straight
    for `abs(True)`. Python's own arbitrary-precision `int` is what a value
    already IS here, so there is no separate big-integer case to keep in
    step with the boxed C twin's.
    """
    v = h._get(a[0], "apy_index_obj")
    if isinstance(v, Instance) and v.cls.find("__index__") is not None:
        return _user(h, lambda: h._value(v._send("__index__")))
    if isinstance(v, bool):
        return h._value(int(v))
    if isinstance(v, int):
        return h._value(v)
    return h._fail("TypeError",
                   f"'{h.kind_name(v)}' object cannot be interpreted as "
                   f"an integer")


def _apy_round(h, a):
    _v = h._get(a[0], "apy_round")
    # `__round__` WITH NO DIGITS. A class defining it decides what rounding
    # itself means.
    _got = _dunder_number(h, _v, ("__round__",))
    if _got is not None:
        return h._value(_got)
    if h.err is not None:
        return 0
    v = h._get(a[0], "apy_round")
    if isinstance(v, bool) or isinstance(v, int):
        return h._int(int(v))
    if not isinstance(v, float):
        return h._fail("TypeError",
                       f"type '{h.kind_name(v)}' doesn't define __round__ method")
    # Python's `round` is half-to-EVEN and returns an int with no digits
    # argument. C's `round` is half-away-from-zero, which is why the C has its
    # own; here the builtin already has the right rule.
    return h._int(round(v))


def _apy_round_to(h, a):
    """`round(x, n)` -- a different function from `round(x)`, and not only in
    precision: this one answers a float where that one answers an int."""
    v = h._get(a[0], "apy_round_to")
    nd = h._get(a[1], "apy_round_to")
    # `__round__(ndigits)` -- the two-argument form, a different call into the
    # same hook. Asked BEFORE the None check, because a class may well be
    # rounded to no digits through it.
    if isinstance(v, Instance) and v.cls.find("__round__") is not None:
        got = _user(h, lambda: v._send("__round__", nd), fail=_FAILED)
        return 0 if got is _FAILED else h._value(got)
    if nd is None:
        return _apy_round(h, a)
    if isinstance(nd, bool) or not isinstance(nd, int):
        return h._fail("TypeError", f"'{h.kind_name(nd)}' object cannot be "
                                    f"interpreted as an integer")
    if isinstance(v, bool) or isinstance(v, int):
        return h._int(round(v, nd))
    if not isinstance(v, float):
        return h._fail("TypeError", f"type '{h.kind_name(v)}' doesn't define "
                                    f"__round__ method")
    return h._new(round(v, nd))


def _meta_check(h, cls, other, hook_name):
    """`__instancecheck__` / `__subclasscheck__` on the metaclass, if it has
    one. Asked BEFORE anything structural, which is what lets a metaclass
    claim a class it has no relationship to -- and answers a BOOL whatever the
    hook returned, as CPython does.

    `_UserFailed` IS CAUGHT HERE, matching every other call site that invokes
    a value the program wrote: a hook that RAISES -- `typing.Protocol`'s
    `__instancecheck__` refusing a non-`@runtime_checkable` class with
    `TypeError`, for one -- otherwise propagated as a raw Python exception
    out of the whole interpreter instead of the program's own `try/except`,
    because nothing between here and the top level was watching for it. `0`
    is NULL/failure, the same sentinel every other guarded host binding
    returns; the caller already treats "not None" as "decided", so a failure
    here is handed back exactly like a real answer would be.
    """
    if not isinstance(cls, Class) or cls.meta is None:
        return None
    hook = cls.meta.lookup(hook_name)
    if hook is _ABSENT:
        return None
    try:
        return h._new(bool(h._invoke(hook, [cls, other])))
    except _UserFailed:
        return 0


def _names_object(h, v) -> bool:
    """Is `v` the `object` type, however it was spelled?

    TWO SPELLINGS REACH HERE. `object` in source is the one class cell the
    runtime hands out; a builtin type name held in a variable arrives as a
    function marked as standing for a type. Both carry the name.
    """
    if isinstance(v, Class):
        return v.name == "object"
    if isinstance(v, Func) and getattr(v, "is_type", False):
        return str(v.name) == "object"
    return v == "object"


def _is_classlike(v) -> bool:
    """Is `v` something `issubclass` may be ASKED about?"""
    if isinstance(v, Class):
        return True
    return isinstance(v, Func) and getattr(v, "is_type", False)


def _apy_is_subclass(h, a):
    x = h._get(a[0], "apy_is_subclass")
    y = h._get(a[1], "apy_is_subclass")
    # BUILTIN TYPES REACHED AS VALUES: `issubclass(bool, int)`. Each side is a
    # callable thunk carrying its name, so the question is asked of the names
    # -- the same rule `isinstance` uses.
    if isinstance(x, Func) and getattr(x, "is_type", False)             and isinstance(y, Func) and getattr(y, "is_type", False):
        return h._new(x.name == y.name or y.name == "object"
                      or (x.name == "bool" and y.name == "int"))
    decided = _meta_check(h, y, x, "__subclasscheck__")
    if decided is not None:
        return decided
    # EVERYTHING IS A SUBCLASS OF `object`, and nothing's base chain contains
    # it: `object` is one class cell that no class names as a base, so the
    # walk below answered False for every class -- and `issubclass(int,
    # object)`, where the first argument is a builtin NAME, was refused
    # outright. Both are wrong about the most basic relation there is.
    if _names_object(h, y) and _is_classlike(x):
        return h._bool(True)
    if not isinstance(x, Class):
        return h._fail("TypeError", "issubclass() arg 1 must be a class")
    # A BUILTIN AS THE SECOND ARGUMENT, with a class extending it as the
    # first: `issubclass(S, str)` for a `class S(str)` is True in CPython and
    # was `issubclass() arg 2 must be a class or tuple of classes` here -- a
    # builtin kind has no class cell, so the walk below had nothing to
    # compare against. The relation is the one `builtin_kind` records, and
    # the name is how it travels, exactly as it does for `isinstance`.
    if not isinstance(y, Class):
        want = (y.name if isinstance(y, (Func, Native))
                and getattr(y, "is_type", False)
                else y if isinstance(y, str) else None)
        if want is not None:
            kind = x.builtin_kind()
            return h._bool(kind is not None and kind.__name__ == want)
        return h._fail("TypeError", "issubclass() arg 2 must be a class or "
                                    "tuple of classes")
    # THROUGH THE ORDER, not the base chain. With several bases the two are
    # different walks, and the chain from `class M(A, C)` reaches only A --
    # so `issubclass(M, C)` answered False for a class that names C as a
    # base. `is_sub` is the same linearisation attribute lookup uses.
    if x.is_sub(y):
        return h._bool(True)
    # An EXCEPTION type has no base pointer -- the builtin hierarchy is a
    # table of NAMES, because `raise` and `except` match on the name and never
    # hold a class. So the same question is asked again, of that table.
    return h._bool(y.name in _exc_chain(h, x.name))


def _apy_exc_type(h, a):
    """A builtin exception NAME as a value. Interned by `_type_of`, so the
    same name is the same object and `type(e) is ValueError` holds.

    `h._value`, not `h._new`: `is` compares HANDLES, and minting a fresh one
    for an object that already has a handle makes `OSError is OSError` False
    -- which is exactly what `IOError is OSError` asks after the alias rewrite
    points both spellings at the one name.
    """
    name = str(h._get(a[0], "apy_exc_type"))
    # THE CLASS THE PROGRAM WROTE, when it wrote one. `except AppError:`,
    # `isinstance(e, AppError)` and `super()` inside its own method must all
    # reach the SAME object, or a method found through one would be missing
    # through another.
    if name in h.exc_class:
        return h._value(h.exc_class[name])
    # TAGGED AS CONSTRUCTIBLE. `_type_of` interns a bare `Class` per kind, so
    # the object standing for `ValueError` is indistinguishable from the one
    # standing for `int` -- and calling it found no `__init__` and answered
    # `ValueError() takes no arguments`. Recording the name here is what lets
    # `_invoke` build one; see the case there.
    cls = h._type_of(Exc(name, None, False))
    # RECORDED SEPARATELY, not on the class. `Class.builtin` already means
    # something else, and writing the name there rendered every constructed
    # exception as a number.
    h.exc_types[id(cls)] = name
    return h._value(cls)


def _apy_id(h, a):
    """`id(x)` -- a number that is distinct for distinct live objects.

    THE HANDLE, not Python's `id()`. A handle already names one cell of this
    host, which is exactly the identity `is` compares, and Python's own id
    would answer about the box a small int is cached in rather than about the
    value the program is holding.
    """
    return h._int(int(a[0]))


def _apy_dict_get(h, a):
    """`d[k]` for a dict, with the KeyError a miss raises.

    Not `dict.get`, which has a default and never raises: this is the
    subscript, reached by name from code that already knows it holds a dict.
    """
    d = h._get(a[0], "apy_dict_get")
    key = h._get(a[1], "apy_dict_get")
    return _dict_get(h, d, key)


def _apy_vars(h, a):
    obj = h._get(a[0], "apy_vars")
    # A CLASS has a `__dict__` too, holding the names its body bound, and
    # `"x" in vars(C)` is how a program asks whether the class defines one.
    if isinstance(obj, Class):
        return h._new(dict(obj.dict))
    if not isinstance(obj, Instance):
        return h._fail("TypeError",
                       "vars() argument must have __dict__ attribute")
    return h._new(dict(obj.dict))


def _apy_delattr(h, a):
    obj = h._get(a[0], "apy_delattr")
    if isinstance(obj, Instance) and obj.cls.find("__delattr__") is not None:
        return _user(h, lambda: h._value(obj._send(
            "__delattr__", str(h._get(a[1], "apy_delattr")))))
    return _apy_default_delattr(h, a)


def _apy_default_delattr(h, a):
    obj = h._get(a[0], "apy_delattr")
    name = str(h._get(a[1], "apy_delattr"))
    if isinstance(obj, Instance):
        # A DATA DESCRIPTOR OWNS THE DELETION. `del obj.v` on a property runs
        # its `@v.deleter`; without this it looked in the instance dict, found
        # nothing there -- a property never puts anything there -- and reported
        # a missing attribute for one the class plainly defines.
        found = obj.cls.lookup(name)
        if isinstance(found, Descr) and found.del_ is not None:
            return _user(h, lambda: h._value(
                h._invoke(found.del_, [obj])))
        # A PROPERTY WITH NO DELETER REFUSES; it does not fall through to the
        # instance dict. The dict never holds a property's name, so falling
        # through reported a missing attribute for one the class plainly
        # defines -- `del p.p` on a read-only property said
        # `'P' object has no attribute 'p'` here while both compiled paths
        # said `can't delete attribute`. Mirrors the C's `apy_delattr`.
        if isinstance(found, Descr) and _is_data_descriptor(found):
            return h._fail("AttributeError", "can't delete attribute")
        # A USER DESCRIPTOR'S `__delete__` IS THE THIRD OF THE THREE, and it
        # is not a `Descr` -- it is an ordinary instance whose class defines
        # the method. Only the property form was handled here, so
        # `del h.d` on a class-level descriptor object looked in the instance
        # dict, found nothing, and reported an attribute the class plainly
        # has. Mirrors the C's `apy_delattr`.
        if isinstance(found, Instance):
            drop = found.cls.find("__delete__")
            if drop is not None:
                return _user(h, lambda: h._value(
                    h._invoke(drop, [found, obj])))
    if not isinstance(obj, Instance) or name not in obj.dict:
        return h._fail("AttributeError",
                       f"'{h.kind_name(obj)}' object has no attribute "
                       f"'{name}'")
    old = obj.dict[name]
    del obj.dict[name]
    # THE INSTANCE'S OWN REFERENCE IS GONE -- the counterpart to
    # `_attr_store`'s incref: `del self.x` drops the one `self.x = ...`
    # added, exactly as `del a` drops a register's.
    old_h = h._referent(old)
    if old_h is not None:
        h.decref(old_h)
    return h._none


def _apy_iter_until(h, a):
    """`iter(f, sentinel)` -- the CALLABLE form.

    A CURSOR IN THE CALL MODE, which calls on every step. The calls USED TO
    ALL HAPPEN HERE, into a list, with a guard at a million to stop a
    callable that never answers the sentinel from running forever. Two things
    were wrong with that and both are what a program sees: the calls happened
    at CONSTRUCTION, so `iter(f, 4)` had already called `f` four times before
    the first `next()`, and a callable that never answers the sentinel made
    `for v in iter(f, s): break` -- which CPython leaves after ONE call --
    run a million times and then stop short of what it was asked for.

    NAMED AFTER THE FORM: CPython calls this a `callable_iterator`, and
    nothing about the mode or the sentinel says so.
    """
    fn = h._get(a[0], "apy_iter_until")
    # REFUSED WHEN IT IS MADE, not when it is walked. CPython checks the
    # first argument here and says so in those words; leaving it to the walk
    # reported `'int' object is not callable` from the first `next()`, which
    # names the wrong moment and the wrong thing.
    if not h._get(_apy_callable(h, [a[0]]), "apy_iter_until"):
        return h._fail("TypeError", "iter(v, w): v must be callable")
    made = Iterator(h._get(a[1], "apy_iter_until"), fn, Iterator.CALL)
    made.named = _CURSOR_NAMES["callable"][0]
    # THE SENTINEL IS NOT A SOURCE TO WALK. `Iterator` reads a dict source's
    # size to refuse one that changes under the walk, and a dict SENTINEL is
    # not being walked at all -- `iter(f, {})` would have measured it and
    # then reported it as changed.
    made.n0 = -1
    return h._new(made)


# ── match ───────────────────────────────────────────────────────────────────
# The predicates a `case` pattern needs that nothing else does. Class and value
# patterns reuse `apy_isinstance` and `apy_eq`; these are the parts with rules
# of their own.


def _apy_match_seq(h, a):
    """Does a SEQUENCE pattern -- `case [a, b]` -- apply to this value?

    A str is NOT a sequence for matching, and neither is bytes. `case [x, y]`
    against "ab" must not bind 'a' and 'b': Python excludes them because
    matching a string element-wise is almost never what was meant, and the
    loop that did it would silently succeed.
    """
    v = h._get(a[0], "apy_match_seq")
    return 1 if isinstance(v, (list, tuple)) else 0


def _apy_match_map(h, a):
    """Does a MAPPING pattern -- `case {"k": v}` -- apply?"""
    return 1 if isinstance(h._get(a[0], "apy_match_map"), dict) else 0


def _apy_match_args(h, a):
    """`cls.__match_args__`, or an empty tuple.

    POSITIONAL SUB-PATTERNS ARE ATTRIBUTE NAMES: `case Point(0, y)` means
    "attribute `x` equals 0, bind attribute `y`", and the class says which
    attributes those are. A class without the declaration accepts no
    positional patterns, which the length check that follows reports.
    """
    cls = h._get(a[0], "apy_match_args")
    if isinstance(cls, Class):
        got = cls.find("__match_args__")
        if isinstance(got, (tuple, list)):
            return h._value(tuple(got))
    return h._new(())


def _apy_match_rest(h, a):
    """What `**rest` in a mapping pattern binds: the dict MINUS the keys the
    pattern named. A copy, because the subject must not change shape because
    something matched it."""
    d = h._get(a[0], "apy_match_rest")
    used = h._get(a[1], "apy_match_rest")
    if not isinstance(d, dict):
        return h._new({})
    drop = list(used) if isinstance(used, (list, tuple)) else []
    return h._new({k: v for k, v in d.items()
                   if not any(k == u for u in drop)})


# ── slice ───────────────────────────────────────────────────────────────────
# `a:b:c` AS A VALUE. Built only where one is needed as an object -- a user
# `__getitem__` receives it, and `c[1:2, 3]` puts one in a tuple. Slicing a
# list or a str still goes straight through without allocating.
#
# Python's own `slice` is used rather than a class of this file's own, so its
# repr, its `.indices` and its equality all come free and match the C by
# construction.


def _apy_slice_new(h, a):
    """`slice(start, stop, step)`. Each may be None -- an omitted bound is not
    the same as any number, which is why the three are kept rather than
    resolved to indices here."""
    return h._new(slice(h._get(a[0], "apy_slice_new"),
                        h._get(a[1], "apy_slice_new"),
                        h._get(a[2], "apy_slice_new")))


def _apy_slice_indices(h, a):
    """`s.indices(n)` -- the (start, stop, step) a walk over a sequence of that
    length would really use, with omitted and negative bounds resolved."""
    sl = h._get(a[0], "apy_slice_indices")
    if not isinstance(sl, slice):
        return h._fail("AttributeError",
                       f"'{h.kind_name(sl)}' object has no attribute "
                       f"'indices'")
    try:
        return h._new(sl.indices(int(h._get(a[1], "apy_slice_indices"))))
    except _HOST_RAISES as exc:
        return h._fail_like(exc)


def _apy_sizeof(h, a):
    """`x.__sizeof__()` -- how many bytes the value occupies HERE.

    AN IMPLEMENTATION NUMBER, and CPython says so: its own answer differs
    between builds and between a 32- and a 64-bit one. What a program can
    rely on is that it is an int and that it grows with what the value
    holds, and `_cell_bytes` is the same arithmetic `apy_sizeof` does in the
    C so the three runtimes agree with each other.
    """
    return h._new(_cell_bytes(h._get(a[0], "apy_sizeof")))


def _apy_dir(h, a):
    """`dir(x)` -- the names it answers to, SORTED.

    `__dir__` overrides the whole computation, and its answer is sorted but
    NOT deduplicated: CPython sorts what the method returned and hands it
    back, so `["b", "a", "a"]` becomes `["a", "a", "b"]`. Deduplicating would
    be tidier and would disagree.

    Without the hook it is the instance's own attributes plus every class in
    the chain -- which IS deduplicated, because a subclass overriding a method
    must not make it appear twice.
    """
    v = h._get(a[0], "apy_dir")
    if isinstance(v, Instance) and v.cls.find("__dir__") is not None:
        got = _user(h, lambda: v._send("__dir__"))
        if got == 0:
            return 0
        return h._new(sorted(got))
    names = []
    def add(seq):
        for n in seq:
            if n not in names:
                names.append(n)
    if isinstance(v, Instance):
        add(v.dict)
        cls = v.cls
        while isinstance(cls, Class):
            add(cls.dict)
            cls = cls.base
    elif (isinstance(v, Class) and not v.dict and v.base is None
            and v.meta is None and v.name in KIND_DIR):
        # A CLASS `_type_of` MINTED reads its KIND's row rather than a class
        # chain it has none of: `dir(type(it))` and `dir(it)` are one list in
        # CPython. The empty dict, no base and no metaclass are what tell
        # such a cell from a class the program wrote, whose chain is walked
        # below. See `apy_dir`.
        add(KIND_DIR[v.name])
    elif isinstance(v, Class):
        cls = v
        while isinstance(cls, Class):
            add(cls.dict)
            cls = cls.base
    elif h.kind_name(v) in KIND_DIR:
        # A BUILT-IN KIND, READ OUT OF THE GENERATED TABLE -- the same one
        # the compiled runtimes read, rather than live CPython. Asking
        # CPython worked while the only kinds were the fifteen this
        # interpreter represents with real Python values, and stopped
        # working at the first CURSOR: `iter([])` here is an object of this
        # compiler's own, and `dir()` over one lists that object's members
        # rather than a `list_iterator`'s. See `apy_kind_dir`.
        add(KIND_DIR[h.kind_name(v)])
    elif (isinstance(v, Func) and getattr(v, "is_type", False)
            and v.name in KIND_DIR):
        # A BUILTIN TYPE IS A FUNC WEARING `is_type`, and `dir(str)` is the
        # same list as `dir("")` -- so it is the same table row.
        add(KIND_DIR[v.name])
    return h._new(sorted(names))


# ── exception groups ────────────────────────────────────────────────────────
# PEP 654. A group IS an exception -- it has a name and a message and
# propagates the same way -- and what distinguishes it is the exceptions it
# carries.


def _apy_excgroup_new(h, a):
    """`ExceptionGroup(msg, [excs])`."""
    msg = h._get(a[0], "apy_excgroup_new")
    excs = h._get(a[1], "apy_excgroup_new")
    if not isinstance(excs, (list, tuple)):
        return h._fail("TypeError",
                       "second argument (exceptions) must be a sequence")
    if not excs:
        return h._fail("ValueError",
                       "second argument (exceptions) must be a non-empty "
                       "sequence")
    g = Exc("ExceptionGroup", msg)
    g.subs = list(excs)
    return h._new(g)


def _group_select(h, g, want, keep):
    """Every leaf of `g` that matches `want`, as a group of the SAME SHAPE --
    or None when nothing in it does.

    The nesting is preserved: a match inside an inner group comes back inside
    an inner group, because `split` is defined to give back something you
    could have raised. That is what makes the two halves add up.
    """
    picked = []
    for one in getattr(g, "subs", None) or []:
        if isinstance(one, Exc) and getattr(one, "subs", None):
            inner = _group_select(h, one, want, keep)
            if inner is not None:
                picked.append(inner)
            continue
        hit = _apy_isinstance(h, [h._new(one), h._new(want)])
        if hit == 0:
            return None
        if bool(h._get(hit, "split")) is bool(keep):
            picked.append(one)
    if not picked:
        return None
    out = Exc("ExceptionGroup", g.arg if g.has_arg else None)
    out.subs = picked
    return out


def _apy_group_subgroup(h, a):
    """`g.subgroup(T)` -- the part that matches, or None."""
    g = h._get(a[0], "apy_group_subgroup")
    if not isinstance(g, Exc) or not getattr(g, "subs", None):
        return h._fail("AttributeError",
                       f"'{h.kind_name(g)}' object has no attribute "
                       f"'subgroup'")
    got = _group_select(h, g, h._get(a[1], "apy_group_subgroup"), True)
    return h._new(got) if got is not None else h._none


def _apy_group_split(h, a):
    """`g.split(T)` -- `(matching, rest)`, either of which may be None.
    Together they hold every leaf the original did."""
    g = h._get(a[0], "apy_group_split")
    if not isinstance(g, Exc) or not getattr(g, "subs", None):
        return h._fail("AttributeError",
                       f"'{h.kind_name(g)}' object has no attribute 'split'")
    want = h._get(a[1], "apy_group_split")
    hit = _group_select(h, g, want, True)
    miss = _group_select(h, g, want, False)
    return h._new((hit if hit is not None else None,
                   miss if miss is not None else None))


def _tgexit_coro(group):
    """What `__aexit__` answers: a coroutine that finishes once every task
    the group started has."""
    g = Gen(None, 1)
    g.coro = True
    g.builtin = _CORO_TGWAIT
    g.slots[0] = group.dict["_tasks"]
    return g


def _tg_create(h, group, coro):
    made = _apy_asyncio_create_task(h, [h._new(coro)])
    if made == 0:
        return 0
    group.dict["_tasks"].append(h._get(made, "create_task"))
    return made


def _apy_pos_add(h, a):
    """Record one statement's position. Called once each, at program start."""
    h.positions.append((str(h._get(a[0], "apy_pos_add")),
                        int(a[1]), int(a[2]), int(a[3]), int(a[4])))
    return h._none


def _apy_pos_count(h, a):
    """How many positions have been recorded.

    A REAL ANSWER HERE, unlike `_apy_pos_rows` below: the count is a number
    whatever the table is made of, and the C asks it to decide whether a
    program was compiled with positions at all.
    """
    return len(h.positions)


def _apy_pos_rows(h, a):
    """The table's address, which the host does not have.

    THE POSITIONS ARE A LIST OF TUPLES here, not rows of a struct -- see
    `_apy_pos_add`. Everything that would walk the rows is itself bound in
    this file and walks the list instead, so nothing reaches this. It raises
    for the reason `_apy_err_slots` does: a zero would be indexed.
    """
    raise RuntimeError(
        "apy_pos_rows has no host equivalent: the interpreter keeps source "
        "positions as a list on the host object, not as a table in memory")


def _apy_at(h, a):
    """Say which statement is running now."""
    h.pos_here = int(a[0])
    return h._none


def _apy_code_of(h, a):
    """The code object a frame names, built from what `_apy_pos_add`
    recorded. Interned per function name, so two tracebacks out of one
    function name the same object."""
    name = str(h._get(a[0], "apy_code_of"))
    made = h._code_objects.get(name)
    if made is not None:
        return h._new(made)
    cls = h._traceback_code_class()
    code = Instance(cls, h)
    rows = [(line, end_line, col, end_col)
            for fn, line, end_line, col, end_col in h.positions if fn == name]
    code.dict["co_name"] = name
    code.dict["co_qualname"] = name
    code.dict["co_filename"] = "<compiled>"
    code.dict["_positions"] = rows
    code.dict["co_firstlineno"] = rows[0][0] if rows else 0
    h._code_objects[name] = code
    return h._new(code)


def _traceback_of(h, exc):
    """The traceback an exception carries.

    ONE FRAME DEEP -- there is no call stack here, so the chain a real
    traceback walks has a single link and `tb_next` is None. Saying that is
    better than inventing frames the runtime never had.
    """
    at = getattr(exc, "pos", -1)
    row = h.positions[at] if 0 <= at < len(h.positions) else None
    where = row[0] if row else "<module>"
    code = h._get(_apy_code_of(h, [h._new(where)]), "traceback")
    frame = Instance(h._traceback_frame_class(), h)
    frame.dict["f_code"] = code
    frame.dict["f_lineno"] = row[1] if row else 0
    frame.dict["f_globals"] = {}
    frame.dict["f_locals"] = {}
    tb = Instance(h._traceback_class(), h)
    tb.dict["tb_frame"] = frame
    tb.dict["tb_lineno"] = row[1] if row else 0
    tb.dict["tb_lasti"] = -1
    tb.dict["tb_next"] = None
    return tb


def _apy_ns_get(h, a):
    """A class body's namespace, read by name.

    NOT `apy_dict_get`, whose miss is a KeyError: this stands for reading a
    NAME, and a name that is not bound is a NameError. Both spellings are
    passed because the KEY is mangled -- `__x` inside `class C` is stored as
    `_C__x` -- and the message has to say what the program wrote.
    """
    ns = h._get(a[0], "apy_ns_get")
    key = h._get(a[1], "apy_ns_get")
    if key in ns:
        return h._value(ns[key])
    shown = h._get(a[2], "apy_ns_get")
    return h._fail("NameError", f"name '{shown}' is not defined")


# ── PEP 750 template strings ────────────────────────────────────────────────
# A t-string DOES NOT JOIN. That is the whole of it: an f-string decides at the
# point of writing that the answer is text and throws away everything it used
# to get there, and a template keeps the pieces apart so that whatever consumes
# it decides instead -- which is what makes one safe to hand a SQL or HTML
# builder and the other not.


def _apy_interpolation_new(h, a):
    """One replacement field: what was WRITTEN and what it CAME TO.

    The expression source is kept because a consumer that reports an error, or
    builds a query with named parameters, needs the text -- and an f-string has
    already discarded it by the time anyone could ask.
    """
    one = Instance(h._interp_class(), h)
    one.dict["value"] = h._get(a[0], "apy_interpolation_new")
    one.dict["expression"] = h._get(a[1], "apy_interpolation_new")
    one.dict["conversion"] = h._get(a[2], "apy_interpolation_new")
    one.dict["format_spec"] = h._get(a[3], "apy_interpolation_new")
    return h._new(one)


def _apy_template_new(h, a):
    """The template itself. `strings` is ALWAYS one longer than
    `interpolations` -- an empty piece stands between two adjacent fields, and
    one stands at each end -- so a consumer walks them in lockstep without
    having to ask which of the two came first."""
    t = Instance(h._template_class(), h)
    t.dict["strings"] = h._get(a[0], "apy_template_new")
    t.dict["interpolations"] = h._get(a[1], "apy_template_new")
    t.dict["values"] = h._get(a[2], "apy_template_new")
    return h._new(t)


def _apy_group_dispatch(h, a):
    """PEP 654's `except*` dispatch, whole, in one call.

    EVERY CLAUSE RUNS -- not the first that matches. A group carrying a
    ValueError and a TypeError enters both handlers, each holding its own
    half, and that is the entire difference from `except`. Splitting here
    rather than as a chain of calls in the lowering keeps the LEFTOVER in one
    place, and the leftover is what has to be re-raised once the clauses
    between them have not accounted for everything.

    Answers one entry per clause -- what it catches, or None -- and last what
    nothing caught, or None.
    """
    raised = h._get(a[0], "apy_group_dispatch")
    types = h._get(a[1], "apy_group_dispatch")
    wrapped = not (isinstance(raised, Exc) and getattr(raised, "subs", None))
    if wrapped:
        # A BARE EXCEPTION IS A GROUP OF ONE to `except*`, which is why a
        # handler binds a group even where the program raised a plain
        # ValueError.
        rest = Exc("ExceptionGroup", "")
        rest.subs = [raised]
    else:
        rest = raised
    out, any_hit = [], False
    for want in types:
        hit = _group_select(h, rest, want, True) if rest is not None else None
        if hit is not None:
            any_hit = True
            rest = _group_select(h, rest, want, False)
        out.append(hit)
    if not any_hit:
        # NOTHING MATCHED, so the ORIGINAL propagates -- not the wrapper this
        # made to split with. A program catching the plain ValueError outside
        # must see the ValueError.
        rest = raised
    elif rest is not None and wrapped:
        subs = getattr(rest, "subs", None) or []
        if len(subs) == 1:
            rest = subs[0]
    out.append(rest)
    return h._new(tuple(out))


def _apy_split_of(h, a):
    """`x.split(y)` where `x` may be a str OR an ExceptionGroup.

    ONE METHOD NAME, TWO RECEIVERS, and which is meant is not known until run
    time -- the same shape `count` and `index` have.
    """
    x = h._get(a[0], "apy_split_of")
    if isinstance(x, Exc) and getattr(x, "subs", None):
        return _apy_group_split(h, a)
    return _TABLE["apy_str_split"](h, a)


# ── generic aliases and builtin type values ─────────────────────────────────
# A builtin type name reaches a program as a one-argument thunk -- that is what
# makes `map(str, xs)` work -- but it is also a TYPE: `print(int)` says
# `<class 'int'>` and `list[int]` parameterises it. The thunk carries a flag
# rather than becoming a real class, because the call path would otherwise have
# to learn to construct from one for the same observable result.


class Alias:
    """`list[int]` -- a PARAMETERISED type.

    Not a type itself: nothing is instantiated from one here, and what a
    program does with it is print it or write it in an annotation. Kept as the
    origin and the arguments so the repr can be rebuilt exactly.
    """

    __slots__ = ("origin", "args", "unpacked")

    def __init__(self, origin, args, unpacked: bool = False) -> None:
        self.origin = origin
        self.args = args
        #: PEP 646's `*list[int]`, and the reason it exists here: ITERATING
        #: an alias is what CPython does with one -- `iter(list[int])` yields
        #: exactly this, once. That is why `1 in list[int]` answers False
        #: rather than raising, and why the flag has to take part in equality:
        #: without it the starred form would equal the plain one.
        self.unpacked = unpacked

    def __eq__(self, other):
        """`list[int] == list[int]` -- SAME ORIGIN, SAME ARGUMENTS.

        Without this a parameterised type compared by identity, so two
        spellings of one annotation were never equal, `hash` disagreed with
        `==`, and a set of them kept every duplicate. `typing.get_type_hints`
        style code that keys on an alias got a fresh bucket per lookup.

        THE ORIGIN IS COMPARED BY IDENTITY on purpose. A union's origin is a
        `typing` special form, which is an `Instance` -- and `Instance.__eq__`
        runs the program's own `__eq__`, which must not be re-entered from
        inside a host dunder that a container is already walking. Every origin
        a program can build is one interned object, so identity is the same
        answer.

        A UNION'S ARMS ARE COMPARED ORDER-FREE, because `int | str` and
        `str | int` are the same type in CPython, and duplicates do not count
        (`int | str | int` equals `int | str`). ONLY a union: `tuple[int, str]`
        and `tuple[str, int]` are different types, so every other form -- a
        builtin origin, `Callable`, `Tuple` -- compares its arguments in
        order.
        """
        if not isinstance(other, Alias):
            return NotImplemented
        # `*list[int] != list[int]`, which is what lets membership walk the
        # one item iteration yields and correctly not match it.
        if self.unpacked != other.unpacked:
            return False
        if self.origin is not other.origin:
            return False
        if _form_name(self.origin) == "Union":
            return _arm_key(self.args) == _arm_key(other.args)
        return self.args == other.args

    def __hash__(self):
        # EQUAL ALIASES HASH EQUALLY, which is the whole reason this is here
        # alongside `__eq__`: one without the other puts two equal aliases in
        # two buckets. `id` for the origin mirrors the identity comparison
        # above; an unhashable argument (`list[[]]`) raises TypeError from
        # here, which is what CPython does too.
        if _form_name(self.origin) == "Union":
            return hash((id(self.origin), self.unpacked,
                         _arm_key(self.args)))
        return hash((id(self.origin), self.unpacked, self.args))


def _alias_text(v) -> str:
    """`list[int]`, `int | str`, `typing.Annotated[int, 'm']` -- a whole alias.

    THE UNION IS THE ONLY FORM THAT PRINTS WITH BARS. PEP 604 made `int | str`
    the spelling for that one; every other form keeps the subscript it was
    written with, and testing "the origin is an instance" made
    `Annotated[int, 'x']` print as a union.

    AT MODULE LEVEL rather than inside `_text`, because `_alias_part` below
    has to reach it for a NESTED one: `list[Annotated[int, 'n']]` printed the
    inner alias through `repr()`, which is this file's own object address.
    """
    if _form_name(v.origin) == "Union":
        return " | ".join(_alias_part(x) for x in v.args)
    # `list[int]`, not `list[<class 'int'>]` -- see `_alias_part`.
    inner = ", ".join(_alias_part(x) for x in v.args) \
        if isinstance(v.args, (list, tuple)) else _alias_part(v.args)
    # `*list[int]` for the unpacked form, which is what CPython prints and
    # what iterating an alias hands out.
    star = "*" if getattr(v, "unpacked", False) else ""
    return f"{star}{_alias_part(v.origin)}[{inner}]"


def _alias_part(x) -> str:
    """How one piece of an alias renders.

    A TYPE ARGUMENT USES ITS NAME even though `str(int)` is `<class 'int'>`:
    CPython's alias repr uses the qualname, and the difference shows in every
    annotation a program prints.
    """
    if isinstance(x, Alias):
        return _alias_text(x)
    if isinstance(x, Func) and getattr(x, "is_type", False):
        return x.name
    if isinstance(x, Class):
        # `NoneType` IS SPELLED `None` inside a union, which is how CPython
        # prints `int | None`: the class is what the union HOLDS and `None` is
        # what it is written as.
        return "None" if x.name == "NoneType" else x.name
    form = _form_name(x)
    if form is not None:
        return "typing." + form
    return repr(x)


def _apy_alias_new(h, a):
    return h._new(Alias(h._get(a[0], "apy_alias_new"),
                        h._get(a[1], "apy_alias_new")))


def _can_iterate(h, v) -> bool:
    """CAN THIS BE WALKED AT ALL, asked without running a line of the program.

    The structural half of `_apy_getiter`'s rule. One caller needs the
    question ANSWERED rather than attempted: `f(*x)` owes a different
    TypeError from the one iteration raises, and a `__iter__` the program
    wrote may raise a TypeError of its own that CPython propagates unchanged.
    Rewording every TypeError would have swallowed it.
    """
    if isinstance(v, Instance):
        return (v.held is not None
                or v.cls.find("__iter__") is not None
                or v.cls.find("__getitem__") is not None)
    if isinstance(v, Class):
        return v.meta is not None and v.meta.lookup("__iter__") is not _ABSENT
    if isinstance(v, (Gen, Iterator, Alias)):
        return True
    if isinstance(v, _VIEW_TYPES):
        return True
    # A RANGE IS A REAL `range` HERE, as a list is a real list: this file
    # keeps a Python object for every kind it has not had to model.
    return isinstance(v, (list, tuple, set, frozenset, dict, str, bytes,
                          bytearray, range, memoryview))


def _callee_str(h, f) -> str:
    """`f()` as CPython names a callee in a message.

    THE MODULE IS PART OF THE NAME except for a builtin: CPython prints
    `print() argument after *` and `__main__.f() argument after *`, and the
    difference between them is the module and nothing else.
    """
    nm = getattr(f, "qualname", None) or getattr(f, "name", None)
    # THE SAME RULE `__module__` ANSWERS BY. A program's own function usually
    # has no module recorded and reads as `__main__`, which is what CPython
    # prints; a builtin has no prefix at all.
    bare = bool(getattr(f, "builtin", False) or getattr(f, "is_type", False)
                or getattr(f, "native", False))
    mod = getattr(f, "module", None) or ("builtins" if bare else "__main__")
    if isinstance(f, Class):
        nm = f.name
        bare = False
        mod = (f.dict.get("__module__") if isinstance(f.dict, dict)
               else None) or "__main__"
    if not isinstance(nm, str):
        return h.kind_name(f)
    if bare or not isinstance(mod, str) or mod == "builtins":
        return nm
    return f"{mod}.{nm}"


def _apy_length_hint(h, a):
    """`PyObject_LengthHint` -- what `list(x)`, `sorted(x)` and the other
    spellings below ask before they drain, so they can size the result in
    one allocation.

    OBSERVABLE, which is why it exists: the answer is thrown away, but a
    class that writes `__len__` or `__length_hint__` sees the call, and a
    program that prints from one prints a line.

    TWO VERBS, IN ORDER, AND ONLY ONE OF THEM. `__len__` first; only when
    there is none is `__length_hint__` asked. A TypeError from `__len__` is
    swallowed and `__length_hint__` tried in its place -- that is how
    CPython lets a type say "I have no length" -- and any OTHER exception is
    the program's own and propagates.

    WHICH SPELLINGS ASK IS MEASURED, not reasoned about: `list(x)`,
    `bytes(x)`, `sorted(x)`, `xs.extend(x)` on a list and on a bytearray,
    `xs += x` on a list, and a starred display -- `[*x]` AND `(*x,)`.
    `tuple(x)`, `set(x)`, `frozenset(x)`, `bytearray(x)`, `{*x}`,
    `s.update(x)`, a comprehension, `f(*x)`, unpacking, `in`, `sum`, `max`,
    `join` and a plain `for` do NOT, which is why this is a function each of
    them names rather than a line in the funnel they share.
    """
    src = h._get(a[0], "apy_length_hint")
    if not isinstance(src, Instance):
        return h._none
    if src.cls.find("__len__") is not None:
        try:
            got = src._send("__len__")
        except _UserFailed:
            # THE FAILURE IS THE ANSWER HERE, not a reason to stop: `__len__`
            # raising is how a type says "no length", and the caller below
            # decides which raisings count. `_UserFailed` carries nothing --
            # `h.err` is already set and is the thing to read.
            got = None
        if h.err is not None:
            if not _apy_error_matches(h, [h._new("TypeError")]):
                return 0
            h.err = None
        elif got is not NotImplemented:
            return h._none
    if src.cls.find("__length_hint__") is not None:
        try:
            src._send("__length_hint__")
        except _UserFailed:
            return 0
        if h.err is not None:
            return 0
    return h._none


def _apy_extend_arg(h, a):
    """`f(*xs)` -- the flattening, with the refusal the CALL SITE owes.

    `extend` can only say what the argument is; CPython names the FUNCTION
    and what the position expected. The call site is the only place that
    knows which function was being called.
    """
    more = h._get(a[1], "apy_extend_arg")
    if _can_iterate(h, more):
        return _apy_extend(h, a[:2])
    who = _callee_str(h, h._get(a[2], "apy_extend_arg"))
    return h._fail("TypeError",
                   f"{who}() argument after * must be an iterable, "
                   f"not {h.kind_name(more)}")


def _apy_alias_unpack(h, a):
    """`*list[int]` -- what iterating `list[int]` yields, once.

    A COPY rather than a flag flipped in place: the alias being iterated is
    the program's own value, and marking it would change what the program
    holds. Anything that is not an alias comes back untouched, which is what
    lets the callers write this without a kind test of their own.
    """
    v = h._get(a[0], "apy_alias_unpack")
    if not isinstance(v, Alias) or v.unpacked:
        return a[0]
    return h._new(Alias(v.origin, v.args, True))


def _apy_func_descr(h, a):
    """Mark this as REACHED OFF A TYPE: `list.append` and `str.upper` are
    DESCRIPTORS in CPython and not functions, and the qualname carries which
    type. See the C's `apy_func_descr`."""
    f = h._get(a[0], "apy_func_descr")
    if isinstance(f, Func):
        f.descr = True
    return a[0]


def _apy_func_module(h, a):
    """`__module__` -- where a SPLICED `def` was written.

    Emitted for those only, because a program's own `def` is in `__main__`
    and the read defaults to it. The C twin writes `v.fn.module`.
    """
    f = h._get(a[0], "apy_func_module")
    if isinstance(f, Func):
        f.module = str(h._get(a[1], "apy_func_module"))
    return a[0]


def _apy_func_qualname(h, a):
    """PEP 3155: record a function's qualified name."""
    f = h._get(a[0], "apy_func_qualname")
    if isinstance(f, Func):
        f.qualname = str(h._get(a[1], "apy_func_qualname"))
    return a[0]


def _apy_func_annotate(h, a):
    """PEP 649: record the thunk that builds this function's annotations."""
    f = h._get(a[0], "apy_func_annotate")
    if isinstance(f, Func):
        f.annotate = h._get(a[1], "apy_func_annotate")
    return a[0]


def _apy_func_builtin(h, a):
    """Mark a thunk as standing for a BUILTIN reached as a value."""
    f = h._get(a[0], "apy_func_builtin")
    if isinstance(f, Func):
        f.builtin = True
    return a[0]


def _apy_func_is_type(h, a):
    """Mark a thunk as standing for a builtin TYPE, and hand back the
    CANONICAL one for that name. It stays callable.

    `int` mentioned twice built two thunks, so `int == int` was False. See the
    C for why interning by name is not the `type(1) is int` registry that was
    tried and reverted.
    """
    f = h._get(a[0], "apy_func_is_type")
    if not isinstance(f, Func):
        return a[0]
    f.is_type = True
    found = h._type_thunks.get(f.name)
    if found is not None:
        return found
    h._type_thunks[f.name] = a[0]
    return a[0]


def _apy_type_object(h, a):
    """`type(x)` -- a TYPE OBJECT, not its name.

    THE SAME OBJECT THE NAME ANSWERS when the program mentions that builtin
    type anywhere, so `type(1) is int` holds. The frontend registers each at
    the top of the entry, before any statement, which takes the evaluation
    order out of it.
    """
    v = h._get(a[0], "apy_type_object")
    if not isinstance(v, (Instance, Class)):
        found = h._type_thunks.get(h.kind_name(v))
        if found is not None:
            return found
    return h._value(h._type_of(v))


def _apy_ascii(h, a):
    """`ascii(x)` -- `repr(x)` with every non-ASCII character escaped.

    NOT AN ALIAS FOR `repr`, which is what it was: `repr('aé')` is
    `'aé'` and `ascii('aé')` is `'a\xe9'`. The whole point is that
    the answer survives a channel that cannot carry the character, so handing
    back the character defeats it.
    """
    shown = _apy_repr(h, a)
    if shown == 0:
        return 0
    # `backslashreplace` produces exactly Python's three escape widths --
    # `\xNN`, `\uNNNN`, `\UNNNNNNNN` -- so the C and this cannot drift on
    # which one a code point gets.
    text = h._get(shown, "apy_ascii")
    return h._new(text.encode("ascii", "backslashreplace").decode("ascii"))


def _apy_hex_of(h, a):
    """`x.hex()` where `x` may be BYTES or a FLOAT.

    ONE METHOD NAME, TWO RECEIVERS, and the two answer entirely different
    things: bytes give their contents in hex digits, a float gives the exact
    binary value it holds. Python's own `float.hex` is used rather than a
    format string, so the thirteen-digit mantissa and the `0x0.0p+0` spelling
    of zero cannot drift from the C.
    """
    x = h._get(a[0], "apy_hex_of")
    if isinstance(x, float):
        return h._new(x.hex())
    # NONE HERE IS THE DEFAULT THIS SYMBOL SUPPLIES and not a separator the
    # program wrote: `b.hex()` reaches here and `b.hex(None)` reaches
    # `apy_bytes_hex`, which refuses it the way CPython does. An empty slot
    # is how that is said.
    sep = h._get(a[1], "apy_hex_of") if a[1] else None
    return _apy_bytes_hex_n(h, [a[0], 0 if sep is None else a[1], 0])


def _apy_float_from_number(h, a):
    """`float.from_number(x)` -- 3.14's way of converting a NUMBER.

    The whole reason it exists beside the constructor is that `float("5")`
    reads text and this refuses it: a program that means "convert this
    number" can say so and have a string rejected rather than parsed. The
    RECEIVER is `a[0]` and ignored, so the type form and the value form are
    one implementation.
    """
    v = h._get(a[1], "apy_float_from_number")
    if isinstance(v, (int, float)) and not isinstance(v, bool) \
            or isinstance(v, bool):
        return h._new(float(v))
    return h._fail("TypeError", f"must be real number, not {h.kind_name(v)}")


def _apy_complex_from_number(h, a):
    """`complex.from_number(x)`. See `_apy_float_from_number`."""
    v = h._get(a[1], "apy_complex_from_number")
    if isinstance(v, complex):
        return h._new(v)
    if isinstance(v, (int, float)):
        return h._new(complex(v, 0.0))
    return h._fail("TypeError", f"must be real number, not {h.kind_name(v)}")


def _apy_float_fromhex(h, a):
    """`float.fromhex('0x1.4p+1')` -- exact, which is the point of it."""
    text = h._get(a[0], "apy_float_fromhex")
    if not isinstance(text, str):
        return h._fail("TypeError", f"fromhex() argument must be str, not "
                                    f"{h.kind_name(text)}")
    try:
        return h._new(float.fromhex(text))
    except ValueError as exc:
        return h._fail_like(exc)


def _apy_dict_view(h, a):
    """`d.keys()` -- A WINDOW ON THE DICT, not a copy of it.

    Python's own view objects are used rather than a class of this file's: they
    are already live, already set-like, and already compare and iterate the way
    the C's do -- so the two cannot drift on any of it.
    """
    d = h._get(a[0], "apy_dict_view")
    if not isinstance(d, dict):
        return h._fail("AttributeError",
                       f"'{h.kind_name(d)}' object has no attribute 'keys'")
    which = int(a[1])
    return h._new(d.keys() if which == 0
                  else d.values() if which == 1 else d.items())


def _apy_view_items(h, a):
    """What the view shows RIGHT NOW, as a list. Read at the moment it is
    asked for rather than when the view was made -- that is the liveness."""
    v = h._get(a[0], "apy_view_items")
    return h._new(list(v))


def _apy_isinstance(h, a):
    # A TUPLE OF TYPES means ANY OF THESE, and there is no ambiguity with
    # asking about the tuple type itself: `isinstance(x, tuple)` arrives as
    # the STRING "tuple", because a builtin kind has no value form.
    v = h._get(a[0], "apy_isinstance")
    spec = h._get(a[1], "apy_isinstance")
    decided = _meta_check(h, spec, v, "__instancecheck__")
    if decided is not None:
        return decided
    # PEP 604: `isinstance(x, int | str)` asks each ARM, which is the same
    # question a tuple of types asks and is answered the same way.
    if isinstance(spec, Alias) and isinstance(spec.origin, Instance):
        return _apy_isinstance(h, [a[0], h._new(tuple(spec.args))])
    if isinstance(spec, tuple):
        for element in spec:
            got = _apy_isinstance(h, [a[0], h._value(element)])
            if not got:
                return 0
            if h._get(got, "apy_isinstance"):
                return h._bool(True)
        return h._bool(False)
    want = h._get(a[1], "apy_isinstance")
    # A real TYPE OBJECT, for a user class: the frontend passes the class
    # itself when the name resolves to one, and its text otherwise. Comparing
    # names would make two classes both called `Node` instances of each other.
    if isinstance(want, Func) and getattr(want, "is_type", False):
        # `t = int; isinstance(x, t)` -- the same question as the literal
        # form, which the frontend rewrites to a name at the call site.
        return _apy_isinstance(h, [a[0], h._new(want.name)])
    if isinstance(want, Class):
        # AN EXCEPTION TYPE REACHED AS A VALUE. `isinstance(e, ValueError)` is
        # rewritten to the NAME at the call site, but `t = ValueError;
        # isinstance(e, t)` -- and `g.split(ValueError)` -- hand the type
        # object over, and an exception is not an `Instance`. It answered
        # False for every such test: a wrong answer, not a refusal.
        if isinstance(v, Exc):
            return _apy_isinstance(h, [a[0], h._new(want.name)])
        return h._bool(isinstance(v, Instance) and v.cls.is_sub(want))
    # ANYTHING ELSE IS NOT A TYPE. A tuple, a union and a class are each
    # handled above, so what is left here is a builtin kind's NAME -- and if
    # it is not even that, the argument was never a type at all. This used to
    # fall through to the name comparison below and answer False, which is a
    # wrong answer where the compiled runtime refuses.
    if not isinstance(want, str):
        return h._fail("TypeError",
                       "isinstance() arg 2 must be a type, a tuple of "
                       "types, or a union")
    have = h.kind_name(v)
    # An INSTANCE never matches a builtin name. Its kind_name is its class's
    # name, so without this a class called `int` would answer True.
    if isinstance(v, Instance):
        # EXCEPT WHEN ITS CLASS EXTENDS ONE. `class D(dict)` makes every D an
        # instance of `dict`, which is what the base says and what a program
        # testing `isinstance(d, dict)` is asking.
        if v.held is not None:
            if h.kind_name(v.held) == want:
                return h._bool(True)
            if want != "object":
                return _apy_isinstance(h, [h._new(v.held), a[1]])
        return h._bool(want == "object")
    if have == want or want == "object":
        return h._bool(True)
    # bool is a SUBCLASS of int in Python, so this is not a name comparison.
    if isinstance(v, bool) and want == "int":
        return h._bool(True)
    if isinstance(v, Exc) and (want in h.user_exc or v.name in h.user_exc):
        return h._bool(want in _exc_chain(h, v.name))
    if isinstance(v, Exc):
        # An exception instance is an instance of every base in its chain.
        # Resolved against Python's own classes, which is what
        # `_apy_error_matches` does -- one source for the hierarchy, so the
        # two answers cannot drift.
        have_cls = getattr(__import__("builtins"), v.name, None)
        want_cls = getattr(__import__("builtins"), want, None)
        if (isinstance(want_cls, type) and issubclass(want_cls, BaseException)
                and isinstance(have_cls, type)
                and issubclass(have_cls, BaseException)):
            return h._bool(issubclass(have_cls, want_cls))
    return h._bool(False)


def _apy_slice(h, a):
    v = h._get(a[0], "apy_slice")
    start, stop, step = int(a[1]), int(a[2]), int(a[3])
    has_start, has_stop = int(a[4]), int(a[5])
    if step == 0:
        return h._fail("ValueError", "slice step cannot be zero")
    sl = slice(start if has_start else None,
               stop if has_stop else None, None if step == 1 else step)
    # A USER OBJECT GETS THE SLICE AS AN OBJECT. `c[1:2]` on an instance is
    # `c.__getitem__(slice(1, 2, None))` -- the class decides what a slice of
    # it means, and it can only do that if it is handed one.
    if isinstance(v, Instance):
        return _apy_getitem(h, [a[0], h._new(sl)])
    # A SLICE OF A RANGE IS A RANGE, not a list -- Python's own `range`
    # answers one, which is the behaviour the C reproduces.
    if isinstance(v, range):
        return h._new(v[sl])
    if not isinstance(v, (list, tuple, str, bytes, bytearray, memoryview)):
        return h._fail("TypeError",
                       f"'{h.kind_name(v)}' object is not subscriptable")
    # SLICING A MEMORYVIEW ANSWERS A MEMORYVIEW, not a copy -- Python's own
    # type does that, which is why this one line covers it.
    return h._new(v[sl])


# ── list and dict methods ───────────────────────────────────────────────────

def _dict_pop(h, d, key, fallback, has_default):
    """`d.pop(k)` and `d.pop(k, default)`. A MISSING KEY WITH NO DEFAULT IS A
    KeyError, which is the whole difference from `d.get(k)`."""
    if key not in d:
        if has_default:
            return fallback
        # The KEY'S REPR is the message, as every missing-key report here
        # does it -- `KeyError: 'a'` and not `KeyError: a`.
        return h._fail("KeyError", repr(key))
    return h._value(d.pop(key))


def _apy_list_pop(h, a):
    v = h._get(a[0], "apy_list_pop")
    # A DICT OR A SET REACHES HERE TOO -- one method name, three receivers,
    # and the frontend cannot know which it has until run time. `d.pop(k)`
    # takes a KEY where `xs.pop(i)` takes an index, so those split off before
    # anything treats the argument as a position.
    if isinstance(v, dict):
        return _dict_pop(h, v, h._get(a[1], "apy_list_pop"), None, False)
    if isinstance(v, (set, frozenset)):
        if not v:
            return h._fail("KeyError", "'pop from an empty set'")
        return h._value(next(iter(v)) if isinstance(v, frozenset) else v.pop())
    # A BYTEARRAY POPS AN INT, because that is what indexing one answers,
    # and the empty message names the kind.
    if isinstance(v, bytearray):
        if not v:
            return h._fail("IndexError", "pop from empty bytearray")
        try:
            at = (_as_index(h, h._get(a[1], "apy_list_pop"))
                  if int(a[2]) else -1)
        except _UserFailed:
            return 0
        try:
            return h._value(v.pop(at))
        except IndexError:
            return h._fail("IndexError", "pop index out of range")
    if not isinstance(v, list):
        return h._fail("AttributeError",
                       f"'{h.kind_name(v)}' object has no attribute 'pop'")
    if not v:
        return h._fail("IndexError", "pop from empty list")
    try:
        index = (_as_index(h, h._get(a[1], "apy_list_pop"))
                 if int(a[2]) else -1)
    except _UserFailed:
        return 0
    try:
        return h._value(v.pop(index))
    except IndexError:
        return h._fail("IndexError", "pop index out of range")


def _apy_pop_or(h, a):
    """`d.pop(k, default)`. Its own entry point because the method table is
    keyed by ARGUMENT COUNT, and at two arguments `xs.pop(i)` cannot be
    meant."""
    d = h._get(a[0], "apy_pop_or")
    if not isinstance(d, dict):
        return h._fail("TypeError",
                       f"pop() takes at most 1 argument for "
                       f"'{h.kind_name(d)}'")
    return _dict_pop(h, d, h._get(a[1], "apy_pop_or"), a[2], True)


def _apy_dict_popitem(h, a):
    """`d.popitem()` -- the LAST pair, and removed. Last rather than
    arbitrary: CPython has taken it from the end since dicts became ordered,
    so a loop that pops a dict empty sees the reverse of insertion order."""
    d = h._get(a[0], "apy_dict_popitem")
    if not isinstance(d, dict):
        return h._fail("AttributeError",
                       f"'{h.kind_name(d)}' object has no attribute "
                       f"'popitem'")
    if not d:
        # QUOTED: `str(KeyError(x))` is the REPR of the argument, so the
        # message carries its own quotes rather than being re-quoted.
        return h._fail("KeyError", "'popitem(): dictionary is empty'")
    return h._value(d.popitem())


def _apy_index_of(h, a):
    v = h._get(a[0], "apy_index_of")
    item = h._get(a[1], "apy_index_of")
    # A VIEW IS A SEQUENCE OF NUMBERS, and CPython COMPARES the element
    # rather than refusing a needle of the wrong kind: `m.index("a")` is the
    # not-found ValueError, where the same needle handed to a bytes receiver
    # is a TypeError.
    if isinstance(v, memoryview):
        for at, one in enumerate(v.tolist()):
            if one == item:
                return h._int(at)
        return h._fail("ValueError", "memoryview.index(x): x not found")
    # A str OR BYTES receiver means SUBSTRING search, not element search. The
    # element loop below answers for a one-character needle and silently
    # wrongly for any longer one.
    if isinstance(v, _TEXTY):
        # A str SUBCLASS IS A str FOR A SUBSTRING SEARCH -- and only for
        # one: the element loop below is a COMPARISON, which must stay the
        # instance's so a written `__eq__` has its say.
        item = _held_text(item)
        try:
            return h._int(v.index(item))
        except ValueError as exc:
            # AN INTEGER NEEDLE OUT OF A BYTE'S RANGE IS ITS OWN COMPLAINT,
            # not a miss: `b"abc".index(300)` is `byte must be in range(0,
            # 256)` and rewriting it to "not found" claimed the search ran.
            if isinstance(item, int) and not isinstance(v, str):
                return h._fail_like(exc)
            # A BYTES RECEIVER HAS NO SUBSTRINGS: CPython says `subsection
            # not found` for one.
            return h._fail("ValueError", "substring not found"
                           if isinstance(v, str) else "subsection not found")
    items = _seq_items(h, v, "apy_index_of")
    if items is None:
        return 0
    for i, x in enumerate(items):
        if x == item:
            return h._int(i)
    return h._fail("ValueError", f"{h._text(item, True)} is not in list")


def _index_bounded(h, a, has_end):
    """`xs.index(v, start)` and `xs.index(v, start, stop)`.

    THE BOUNDED SEARCH, which a list and a tuple have as well as a str --
    `xs.index(v, start)` is how a scan resumes past the last hit. A str or
    bytes receiver means SUBSTRING, exactly as at one argument. A RANGE HAS
    NO BOUNDED FORM: CPython says `range.index() takes exactly one argument`,
    so it is refused rather than given a window it has no method for.
    """
    seq = h._get(a[0], "apy_index_of")
    item = h._get(a[1], "apy_index_of")
    lo = h._get(a[2], "apy_index_of")
    hi = h._get(a[3], "apy_index_of") if has_end else None
    if not isinstance(seq, (str, bytes, bytearray, list, tuple, memoryview)):
        return h._fail("AttributeError",
                       f"'{h.kind_name(seq)}' object has no attribute 'index'")
    # A str SUBCLASS IS A str FOR A SUBSTRING SEARCH -- and ONLY for one:
    # `["a"].index(S("a"))` is a comparison, and must stay the instance's so
    # that a written `__eq__` has its say.
    if isinstance(seq, (str, bytes, bytearray)):
        item = _held_text(item)
    # A SEQUENCE'S `index` TAKES NO None AT ALL, where a string's does --
    # CPython leaves the `or None` out of this one, and it is the only thing
    # that tells the two messages apart. A VIEW is one of the sequences.
    if not isinstance(seq, (str, bytes, bytearray)):
        # NONE IS NOT A BOUND HERE. `hi` is absent rather than None when the
        # call wrote no third argument, which is what `has_end` says.
        for bound in ((lo, hi) if has_end else (lo,)):
            if not _is_int_like(bound):
                return h._fail("TypeError",
                               "slice indices must be integers or have an "
                               "__index__ method")
    try:
        if hi is None:
            return h._int(seq.index(item, 0 if lo is None else lo))
        return h._int(seq.index(item, 0 if lo is None else lo,
                                len(seq) if hi is None else hi))
    except ValueError:
        if isinstance(seq, (str, bytes, bytearray)):
            # A BYTES RECEIVER HAS NO SUBSTRINGS: CPython says `subsection
            # not found` for one.
            return h._fail("ValueError", "substring not found"
                           if isinstance(seq, str) else "subsection not found")
        if isinstance(seq, memoryview):
            return h._fail("ValueError", "memoryview.index(x): x not found")
        return h._fail("ValueError",
                       f"{h.kind_name(seq)}.index(x): x not in "
                       f"{h.kind_name(seq)}")
    except _HOST_RAISES as exc:
        return h._fail_like(exc)


def _apy_index_of2(h, a):
    """`xs.index(v, start)`."""
    return _index_bounded(h, a, False)


def _apy_index_of3(h, a):
    """`xs.index(v, start, stop)`."""
    return _index_bounded(h, a, True)


def _str_rindex(h, a, has_end):
    """`s.rindex(sub, start)` and `s.rindex(sub, start, stop)` -- str and
    bytes only, which is why this does not share the sequence walk above."""
    s = h._get(a[0], "apy_str_rindex")
    sub = h._get(a[1], "apy_str_rindex")
    lo = h._get(a[2], "apy_str_rindex")
    hi = h._get(a[3], "apy_str_rindex") if has_end else None
    if not isinstance(s, (str, bytes, bytearray)):
        return h._fail("AttributeError", f"'{h.kind_name(s)}' object has no "
                                         f"attribute 'rindex'")
    # A str SUBCLASS IS A str FOR A SUBSTRING SEARCH -- see `_held_text`.
    sub = _held_text(sub)
    try:
        at = s.rindex(sub, 0 if lo is None else lo,
                      len(s) if hi is None else hi)
    except ValueError:
        return h._fail("ValueError", "substring not found"
                       if isinstance(s, str) else "subsection not found")
    except _HOST_RAISES as exc:
        return h._fail_like(exc)
    return h._int(at)


def _apy_str_rindex2(h, a):
    """`s.rindex(sub, start)`."""
    return _str_rindex(h, a, False)


def _apy_str_rindex3(h, a):
    """`s.rindex(sub, start, stop)`."""
    return _str_rindex(h, a, True)


def _apy_str_index2(h, a):
    """`s.index(sub, start)` -- the str half, reached when the receiver is
    known to be text."""
    return _index_bounded(h, a, False)


def _apy_str_index3(h, a):
    """`s.index(sub, start, stop)`."""
    return _index_bounded(h, a, True)


def _apy_count_of(h, a):
    _v = h._get(a[0], "apy_count_of")
    # A VIEW IS A SEQUENCE OF NUMBERS. See `_apy_index_of`.
    if isinstance(_v, memoryview):
        want = h._get(a[1], "apy_count_of")
        return h._int(sum(1 for one in _v.tolist() if one == want))
    # Substring counting for a str or bytes, for the same reason `index`
    # splits.
    if isinstance(_v, _TEXTY):
        # A NON-STRING ARGUMENT IS A RUNTIME ERROR AND NOT AN INTERPRETER
        # ONE. `"abc".count(5)` let Python's TypeError escape this binding
        # and take the whole interpreter down with a traceback, where a
        # compiled program raises it and a handler catches it.
        try:
            return h._int(_v.count(_held_text(h._get(a[1], "apy_count_of"))))
        except TypeError as exc:
            return h._fail_like(exc)
    seq = h._get(a[0], "apy_count_of")
    item = h._get(a[1], "apy_count_of")
    # SUBSTRING counting for a str, element counting for a sequence -- the
    # same split `index` makes, and the reason `'aaaa'.count('aa')` is 2 and
    # not 0. Walking a str as a list of characters answered 0 for every
    # multi-character needle.
    if isinstance(seq, _TEXTY):
        # A str SUBCLASS IS A str FOR A SUBSTRING COUNT -- see `_held_text`.
        item = _held_text(item)
        # A BYTEARRAY COUNTS BYTES, so `bytearray(b"a-b").count(b"-")` is an
        # ordinary call -- the needle need not be the receiver's exact type,
        # only its FAMILY.
        family = (bytes, bytearray) if isinstance(seq, (bytes, bytearray)) \
            else str
        if not isinstance(item, family):
            return h._fail("TypeError",
                           f"must be {h.kind_name(seq)}, not "
                           f"{h.kind_name(item)}")
        return h._int(seq.count(item))
    items = _seq_items(h, seq, "apy_count_of")
    if items is None:
        return 0
    return h._int(sum(1 for x in items if x == item))


def _apy_list_remove(h, a):
    v = h._get(a[0], "apy_list_remove")
    item = h._get(a[1], "apy_list_remove")
    # A BYTEARRAY REMOVES AN OCTET BY VALUE: `remove(98)`, not `remove(b"b")`.
    if isinstance(v, bytearray):
        byte = _byte_arg(h, item)
        if byte is None:
            return 0
        try:
            v.remove(byte)
        except ValueError:
            return h._fail("ValueError", "value not found in bytearray")
        return h._none
    if not isinstance(v, list):
        return h._fail("AttributeError",
                       f"'{h.kind_name(v)}' object has no attribute 'remove'")
    try:
        v.remove(item)
    except ValueError:
        return h._fail("ValueError", "list.remove(x): x not in list")
    return h._none


def _apy_dict_parts(h, a):
    """`d.keys()`, `d.values()`, `d.items()` -- A VIEW, not a snapshot.

    `ks = d.keys()` then `d['b'] = 2` and `len(ks)` is 2. A snapshot is the
    obvious implementation and is wrong exactly when a program relies on the
    view being live.
    """
    return _apy_dict_view(h, a)


def _apy_dict_get_or(h, a):
    v = h._get(a[0], "apy_dict_get_or")
    if not isinstance(v, dict):
        return h._fail("AttributeError",
                       f"'{h.kind_name(v)}' object has no attribute 'get'")
    key = h._get(a[1], "apy_dict_get_or")
    try:
        hash(key)
    except TypeError as exc:
        return h._fail_like(exc)
    return a[2] if key not in v else h._value(v[key])


#: `apy_name` -> the function that implements it.
#:
#: DERIVED from the definitions above rather than listed, because the naming
#: is exact -- `_apy_foo` implements `apy_foo` -- and a hand-written list of
#: two hundred entries is a second place for the same fact to live. The one
#: it drifts from is the runtime, and `tests/uasm/unit/test_objects_host`
#: is the ratchet that catches that: every symbol `objects/csource.py` exports
#: must be reachable here.
#:
#: The arithmetic and comparison operators are added separately: they share
#: one implementation parameterised by the Python operator, so there is no
#: `_apy_add` to find.
_TABLE: dict = {}

_TABLE.update({
    "apy_add": _binop("apy_add", lambda x, y: x + y, "+"),
    "apy_sub": _binop("apy_sub", lambda x, y: x - y, "-"),
    "apy_mul": _binop("apy_mul", lambda x, y: x * y, "*"),
    "apy_truediv": _binop("apy_truediv", lambda x, y: x / y, "/"),
    "apy_floordiv": _binop("apy_floordiv", lambda x, y: x // y, "//"),
    "apy_mod": _binop("apy_mod", lambda x, y: x % y, "%"),
    "apy_pow": _binop("apy_pow", _pow, "**"),
    "apy_bitand": _binop("apy_bitand", lambda x, y: x & y, "&"),
    "apy_bitor": _binop("apy_bitor", lambda x, y: x | y, "|"),
    "apy_bitxor": _binop("apy_bitxor", lambda x, y: x ^ y, "^"),
    "apy_lshift": _binop("apy_lshift", lambda x, y: x << y, "<<"),
    "apy_rshift": _binop("apy_rshift", lambda x, y: x >> y, ">>"),
    # PEP 465. NO BUILT-IN KIND IMPLEMENTS IT -- there are no matrices here --
    # so this reaches `__matmul__` or reports the operand pair.
    "apy_matmul": _binop("apy_matmul", lambda x, y: x @ y, "@"),
    "apy_eq": _cmpop("apy_eq", lambda x, y: x == y),
    "apy_ne": _cmpop("apy_ne", lambda x, y: x != y),
    "apy_lt": _cmpop("apy_lt", lambda x, y: x < y),
    "apy_le": _cmpop("apy_le", lambda x, y: x <= y),
    "apy_gt": _cmpop("apy_gt", lambda x, y: x > y),
    "apy_ge": _cmpop("apy_ge", lambda x, y: x >= y),
})

_TABLE.update({
    "apy_sorted": _apy_sorted,
    "apy_min": _apy_min,
    "apy_max": _apy_max,
    "apy_sum": _apy_sum,
    "apy_reversed": _apy_reversed,
    "apy_enumerate": _apy_enumerate,
    "apy_zip2": _apy_zip2,
    "apy_range": _apy_range,
    "apy_abs": _apy_abs,
    "apy_index_obj": _apy_index_obj,
    "apy_round": _apy_round,
    "apy_isinstance": _apy_isinstance,
    "apy_slice": _apy_slice,
    "apy_list_pop": _apy_list_pop,
    "apy_index_of": _apy_index_of,
    "apy_count_of": _apy_count_of,
    "apy_list_remove": _apy_list_remove,
    "apy_dict_parts": _apy_dict_parts,
    "apy_dict_get_or": _apy_dict_get_or,
})


# ── str methods, and set algebra ────────────────────────────────────────────
# Every one of these is the Python method on the real object. That is the whole
# argument for a handle table over real objects rather than a reimplementation:
# `'ab'.split(',')` here IS `str.split`, so the empty-separator ValueError, the
# whitespace-runs rule and the exact return shapes come from CPython instead of
# from a second description of them that could drift from the C's.
#
# The spec is (symbol, Python method, argument count) and the wrappers are
# built from it, because forty near-identical functions written out is forty
# chances to bind one to the wrong method.

_STR_METHOD_SPEC = (
    ("apy_str_upper", "upper", 0), ("apy_str_lower", "lower", 0),
    ("apy_str_title", "title", 0), ("apy_str_capitalize", "capitalize", 0),
    ("apy_str_swapcase", "swapcase", 0), ("apy_str_casefold", "casefold", 0),
    ("apy_str_strip", "strip", 0), ("apy_str_lstrip", "lstrip", 0),
    ("apy_str_rstrip", "rstrip", 0),
    ("apy_str_strip_chars", "strip", 1),
    ("apy_str_lstrip_chars", "lstrip", 1),
    ("apy_str_rstrip_chars", "rstrip", 1),
    ("apy_str_split_ws", "split", 0), ("apy_str_split", "split", 1),
    ("apy_str_split_n", "split", 2),
    ("apy_str_rsplit_ws", "rsplit", 0), ("apy_str_rsplit", "rsplit", 1),
    ("apy_str_rsplit_n", "rsplit", 2),
    ("apy_str_splitlines", "splitlines", 0),
    ("apy_str_splitlines_keep", "splitlines", 1),
    ("apy_str_partition", "partition", 1),
    ("apy_str_rpartition", "rpartition", 1),
    ("apy_str_join", "join", 1),
    ("apy_str_replace", "replace", 2), ("apy_str_replace_n", "replace", 3),
    ("apy_str_find", "find", 1), ("apy_str_find2", "find", 2),
    ("apy_str_find3", "find", 3),
    ("apy_str_rfind", "rfind", 1), ("apy_str_rfind2", "rfind", 2),
    ("apy_str_rfind3", "rfind", 3),
    ("apy_str_rindex", "rindex", 1),
    ("apy_str_startswith", "startswith", 1),
    ("apy_str_startswith2", "startswith", 2),
    ("apy_str_startswith3", "startswith", 3),
    ("apy_str_endswith", "endswith", 1),
    ("apy_str_endswith2", "endswith", 2),
    ("apy_str_endswith3", "endswith", 3),
    ("apy_str_count2", "count", 2), ("apy_str_count3", "count", 3),
    ("apy_str_zfill", "zfill", 1),
    ("apy_str_center", "center", 1), ("apy_str_center_fill", "center", 2),
    ("apy_str_ljust", "ljust", 1), ("apy_str_ljust_fill", "ljust", 2),
    ("apy_str_rjust", "rjust", 1), ("apy_str_rjust_fill", "rjust", 2),
    ("apy_str_removeprefix", "removeprefix", 1),
    ("apy_str_removesuffix", "removesuffix", 1),
    ("apy_str_isalpha", "isalpha", 0), ("apy_str_isdigit", "isdigit", 0),
    ("apy_str_isdecimal", "isdecimal", 0),
    ("apy_str_isnumeric", "isnumeric", 0),
    ("apy_str_isalnum", "isalnum", 0), ("apy_str_isspace", "isspace", 0),
    ("apy_str_islower", "islower", 0), ("apy_str_isupper", "isupper", 0),
    ("apy_str_istitle", "istitle", 0), ("apy_str_isascii", "isascii", 0),
    ("apy_str_isprintable", "isprintable", 0),
    ("apy_str_isidentifier", "isidentifier", 0),
)


def _apy_str_like(h, a):
    """Re-tag a str method's result to match its receiver.

    A no-op here: the host calls Python's own methods, so a bytes receiver
    already answers bytes. The binding exists because the frontend emits the
    call for the C, which shares one implementation between the two kinds and
    has to put the tag back.
    """
    return a[1]


def _is_iterator(got) -> bool:
    """Is this an ITERATOR -- the rule `__iter__`'s ANSWER has to satisfy.

    A generator or a cursor is one; a str is not, however walkable it looks,
    and accepting one turned a broken class into a working one that iterated
    something else entirely. `_apy_iterable` enforces the same rule where it
    walks, and `str.join` asks it HERE so it can word the refusal its own way
    -- see the call site for why clearing the funnel's error instead would
    swallow more than CPython does.
    """
    # A GENERATOR, A CURSOR, OR A CLASS WITH `__next__`, AND NOTHING ELSE.
    # A list is not an iterator and neither is a dict, a set or a tuple --
    # `iter()` makes one FROM each of them, which is a different thing, and
    # this used to accept all four. A class whose `__iter__` answered a list
    # then iterated it happily where CPython refuses, so a broken class
    # silently worked and the author was never told which method to fix.
    return (isinstance(got, (Gen, Iterator))
            or (isinstance(got, Instance)
                and got.cls.find("__next__") is not None))


def _joinable(v) -> bool:
    """Whether `join` can walk `v`.

    THE SAME THREE ROADS `for` TAKES -- `__iter__`, the older `__getitem__`
    protocol, or a builtin underneath -- and for a class object, whatever its
    metaclass says, which is how `",".join(Color)` walks an enum's members.
    """
    if isinstance(v, Instance):
        return (v.held is not None or v.cls.find("__iter__") is not None
                or v.cls.find("__getitem__") is not None)
    if isinstance(v, Class):
        return v.meta is not None and v.meta.lookup("__iter__") is not _ABSENT
    return True


def _make_str_method(symbol: str, method: str, argc: int):
    def binding(h, a, _m=method, _n=argc, _sym=symbol):
        receiver = h._get(a[0], _sym)
        # BYTES AND BYTEARRAY TOO. `b.strip()` is the same operation on the
        # same layout, and Python's own types spell it the same way -- so the
        # receiver being accepted is the whole of it here, and the result
        # comes back correctly tagged without any re-tagging. The C needs
        # `apy_str_like` for that.
        #
        # A BYTEARRAY WAS LEFT OFF THIS LIST and it is not a third kind: the
        # compiled runtimes hold one as a bytes cell that admits it is
        # mutable, and answered every method. Here it was an AttributeError
        # about a method Python plainly has.
        # AN INSTANCE OF A CLASS EXTENDING ONE OF THE THREE IS ONE, both as
        # the receiver and as an argument -- see `_held_text`. The receiver
        # arrives unwrapped for a written `s.upper()`, which goes through
        # `apy_method_self`; the unbound `str.upper(s)` does not.
        receiver = _held_text(receiver)
        if not isinstance(receiver, (str, bytes, bytearray)):
            return h._fail(
                "AttributeError",
                f"'{h.kind_name(receiver)}' object has no attribute '{_m}'")
        given = [h._get(a[i + 1], _sym) for i in range(_n)]
        args = [_held_text(one) for one in given]
        # A TUPLE OF PREFIXES IS HELD TO THE SAME RULE, element by element:
        # `"abc".startswith((S("z"), S("a")))` is True in CPython, and the
        # whole-argument unwrap above leaves a tuple alone.
        if _m in ("startswith", "endswith") and isinstance(args[0], tuple):
            args[0] = tuple(_held_text(one) for one in args[0])
        # A GENERATOR OR CURSOR ARGUMENT IS DRAINED FIRST. `join` is the one
        # that takes an iterable, and once generator expressions became real
        # generators `sep.join(f(x) for x in xs)` started arriving here as a
        # `Gen` -- which Python's own `join` refuses, reporting `can only join
        # an iterable` about something that plainly was one. The C fixed
        # exactly this by reaching `apy_iterable` first; this is the same fix
        # on the other path.
        for at, arg in enumerate(args):
            if isinstance(arg, (Gen, Iterator)):
                drained = _seq_items(h, arg, _sym)
                if drained is None:
                    return 0
                args[at] = drained
        # A CLASS IS WALKED FOR `join` ALONE. It is the only one of these
        # methods that takes an iterable -- every other argument is a string,
        # and draining one would replace Python's own complaint about it with
        # a complaint about a list. Handing an `Instance` to `str.join` made
        # Python object to a SUBSCRIPT, about a class a `for` loop walks;
        # and for one that cannot be walked the refusal is `join`'s own,
        # which names no kind where `_seq_items` names one.
        if _m == "join" and isinstance(args[0], (Instance, Class)):
            if not _joinable(args[0]):
                return h._fail("TypeError", "can only join an iterable")
            # AND WHAT `__iter__` GIVES BACK IS `join`'S QUESTION TOO.
            # `_seq_items` asks it as well, but its refusal names the kind --
            # `iter() returned non-iterator of type 'int'` -- where `join`'s
            # names none. CPython reaches iteration through
            # `PySequence_Fast(seq, "can only join an iterable")`, which
            # REPLACES every TypeError that comes out of GETTING the iterator
            # and lets the ones raised while WALKING through; `_joinable`
            # above already covers the other way of failing to get one.
            #
            # NOT BY CLEARING THE FUNNEL'S ERROR afterwards: `_seq_items`
            # gets the iterator AND drains it, so that would also swallow a
            # TypeError raised inside a user's `__next__`, which CPython
            # propagates. `__iter__` IS CALLED ONCE, and what it answered is
            # what goes on to the funnel.
            if isinstance(args[0], Instance) \
                    and args[0].cls.find("__iter__") is not None:
                got = _user(h, lambda: args[0]._send("__iter__"),
                            fail=_FAILED)
                if got is _FAILED or h.err is not None:
                    return 0
                if got is not NotImplemented:
                    if not _is_iterator(got):
                        return h._fail("TypeError",
                                       "can only join an iterable")
                    args[0] = got
            drained = _seq_items(h, args[0], _sym)
            if drained is None:
                return 0
            args[0] = drained
        # ITS ELEMENTS ARE WHAT MUST BE str, whatever they arrived in --
        # a written list reaches here without the drain above, and it was
        # `"-".join(["a", S("b")])` that reported an `Instance` where
        # CPython answered `a-b`.
        if _m == "join" and isinstance(args[0], (list, tuple)):
            args[0] = [_held_text(one) for one in args[0]]
        try:
            got = getattr(receiver, _m)(*args)
        except _HOST_RAISES as exc:
            return h._fail_like(exc)
        # PARTITION HANDS THE SEPARATOR BACK AS THE OBJECT IT WAS GIVEN,
        # which for a str subclass means the INSTANCE and not the text
        # inside it: CPython's `partition` increfs and returns `sep` itself,
        # so `type("a-b".partition(S("-"))[1])` is `S`.
        #
        # ONLY FOR AN INSTANCE, and only when the separator was FOUND. A
        # bytearray receiver answers three BYTEARRAYS whatever it was
        # handed, so putting a plain `b"-"` back in the middle made
        # `bytearray(b"a-b").partition(b"-")` disagree with CPython; and a
        # miss puts an empty str there, which is str's own.
        if _m in ("partition", "rpartition") and got[1] \
                and isinstance(given[0], Instance):
            got = (got[0], given[0], got[2])
        return h._value(got)
    return binding


for _sym, _method, _argc in _STR_METHOD_SPEC:
    _TABLE[_sym] = _make_str_method(_sym, _method, _argc)


def _apy_str_format_map(h, a):
    """`s.format_map(m)` -- `format` with the mapping handed over WHOLE.

    CPython does not CHECK the argument, it SUBSCRIPTS it, so what a program
    sees for a non-mapping is whatever the subscript refused with -- which is
    why the two messages here are a subscript's and not a signature's.
    """
    fmt = _held_text(h._get(a[0], "apy_str_format_map"))
    mapping = h._get(a[1], "apy_str_format_map")
    if not isinstance(fmt, str):
        return h._fail("AttributeError",
                       f"'{h.kind_name(fmt)}' object has no attribute "
                       f"'format_map'")
    # THROUGH PYTHON'S OWN `format_map`, which is what makes the mapping
    # LAZY: it is subscripted only once a field asks it for a key, so
    # `"".format_map(None)` is `''` rather than a complaint about None. A
    # check here could not tell the two apart.
    try:
        return h._new(fmt.format_map(mapping))
    except (IndexError, KeyError, ValueError, TypeError) as exc:
        return h._fail_like(exc)


def _apy_bytearray_resize(h, a):
    """`ba.resize(n)` -- 3.14's way of saying how long a bytearray is to be.

    GROWING FILLS WITH NUL and shrinking truncates, which is what makes it a
    buffer a program can hand out and then fill.
    """
    held = h._get(a[0], "apy_bytearray_resize")
    want = h._get(a[1], "apy_bytearray_resize")
    if not isinstance(held, bytearray):
        return h._fail("AttributeError",
                       f"'{h.kind_name(held)}' object has no attribute "
                       f"'resize'")
    if not isinstance(want, int) or isinstance(want, bool):
        return h._fail("TypeError",
                       f"'{h.kind_name(want)}' object cannot be interpreted "
                       f"as an integer")
    if want < 0:
        return h._fail("ValueError",
                       f"Can only resize to positive sizes, got {want}")
    if want > len(held):
        held.extend(b"\0" * (want - len(held)))
    else:
        del held[want:]
    return h._none


# Set algebra. These symbols serve the METHOD spelling, which takes ANY
# ITERABLE -- `{1}.union([2])` is `{1, 2}` where `{1} | [2]` is a TypeError,
# and the operator has its own path and its own wording. Refusing a list here
# refused eight calls the compiled runtimes and CPython both answer.
_SET_OP_SPEC = (
    ("apy_set_union", "union"),
    ("apy_set_intersection", "intersection"),
    ("apy_set_difference", "difference"),
    ("apy_set_symdiff", "symmetric_difference"),
    ("apy_set_issubset", "issubset"),
    ("apy_set_issuperset", "issuperset"),
    ("apy_set_isdisjoint", "isdisjoint"),
)


def _make_set_op(symbol: str, method: str):
    def binding(h, a, _m=method, _sym=symbol):
        left = h._get(a[0], _sym)
        right = h._get(a[1], _sym)
        if not isinstance(left, (set, frozenset)):
            return h._fail(
                "AttributeError",
                f"'{h.kind_name(left)}' object has no attribute '{_m}'")
        if not isinstance(right, (set, frozenset)):
            # THROUGH THE FUNNEL, not through `set()`. A class of this file's
            # is not a Python iterable however well a `for` loop walks it, so
            # `{1}.union(obj)` reported that a class whose `__iter__` says
            # exactly how was not iterable -- and one with the OLDER protocol
            # was worse than refused, because Python's own `set()` walked it
            # and the user's IndexError came back as this file's internal
            # failure instead of as the end of the walk. The in-place three
            # below already read their operand this way. `_seq_items` words
            # its refusal exactly as this one did, so nothing is reworded.
            items = _seq_items(h, right, _sym)
            if items is None:
                return 0
            # HANDED OVER AS A LIST, not made into a set first. Python's set
            # methods take any iterable, and building a set here changed the
            # ORDER the elements arrived in -- which is the order the answer
            # iterates and prints in, so `{1}.union(obj)` printed its members
            # in an order CPython's own `union` does not.
            right = items
        try:
            result = getattr(left, _m)(right)
        except TypeError as exc:
            # An UNHASHABLE element, which is the method's own complaint and
            # not a claim about the argument as a whole.
            return h._fail_like(exc)
        # A frozenset operand keeps the LEFT operand's kind, as Python does.
        if isinstance(result, (set, frozenset)) and isinstance(left, frozenset):
            result = frozenset(result)
        return h._new(result) if not isinstance(result, bool) else h._bool(result)
    return binding


for _sym, _method in _SET_OP_SPEC:
    _TABLE[_sym] = _make_set_op(_sym, _method)


# The same three IN PLACE, and they are NOT `_make_set_op`: each mutates and
# answers None, a FROZENSET HAS NONE OF THEM -- there is nothing to update --
# and each takes any iterable, which is the split the algebra above already
# draws between a method and the operator behind it.
_SET_UPDATE_SPEC = (
    ("apy_set_inter_update", "intersection_update"),
    ("apy_set_diff_update", "difference_update"),
    ("apy_set_symdiff_update", "symmetric_difference_update"),
)


def _make_set_update(symbol: str, method: str):
    def binding(h, a, _m=method, _sym=symbol):
        left = h._get(a[0], _sym)
        if not isinstance(left, set):
            return h._fail(
                "AttributeError",
                f"'{h.kind_name(left)}' object has no attribute '{_m}'")
        right = h._get(a[1], _sym)
        if not isinstance(right, (set, frozenset)):
            items = _seq_items(h, right, _sym)
            if items is None:
                return 0
            # AS A LIST, for the reason `_make_set_op` gives: making a set of
            # it first reorders what the answer holds.
            right = items
        try:
            getattr(left, _m)(right)
        except TypeError as exc:
            return h._fail_like(exc)
        return h._none
    return binding


for _sym, _method in _SET_UPDATE_SPEC:
    _TABLE[_sym] = _make_set_update(_sym, _method)


def _apy_set_new(h, a):
    return h._new(set())


def _apy_frozenset_new(h, a):
    return h._new(frozenset())


def _apy_set_push(h, a):
    s = h._get(a[0], "apy_set_push")
    item = h._get(a[1], "apy_set_push")
    bad = _unhashable_name(item)
    if bad:
        return h._fail("TypeError", f"cannot use '{bad}' as a set element "
                                    f"(unhashable type: '{bad}')")
    try:
        hash(item)
    except TypeError as exc:
        return h._fail_like(exc)
    if isinstance(s, set):
        was_new = item not in s
        s.add(item)
        if was_new:
            h.incref(int(a[1]))
        return h._none
    return h._fail("AttributeError",
                   f"'{h.kind_name(s)}' object has no attribute 'add'")


def _apy_set_discard(h, a):
    s = h._get(a[0], "apy_set_discard")
    if not isinstance(s, set):
        return h._fail("AttributeError",
                       f"'{h.kind_name(s)}' object has no attribute 'discard'")
    item = h._get(a[1], "apy_set_discard")
    was_present = item in s
    s.discard(item)
    if was_present:
        removed_h = h._referent(item)
        if removed_h is not None:
            h.decref(removed_h)
    return h._none


def _apy_same_tuple(h, a):
    """`t` IF IT IS ALREADY A TUPLE, and 0 otherwise. See the C twin."""
    v = h._get(a[0], "apy_same_tuple")
    return a[0] if type(v) is tuple else 0


def _to_set(h, a, frozen: bool):
    v = h._get(a[0], "apy_to_set")
    if not isinstance(v, (list, tuple, set, frozenset, dict, str, range)):
        # NOT ONE OF THE FAST CONTAINER TYPES -- which used to mean an
        # outright refusal, so `frozenset(x for x in y)` failed with
        # "'generator' object is not iterable" while `list(x for x in y)`
        # worked: `list()`'s path already drains a generator (and any other
        # iterable) through `_apy_iterable`/`_apy_gen_drain`, and `set()`/
        # `frozenset()` had never been routed through the same machinery.
        # Same funnel as `_apy_iterable`'s own class-extending-a-builtin
        # comment describes: patching one caller and not this one just moves
        # which builtin call reports the object as not iterable.
        #
        # THROUGH `_seq_items`, WHICH CONSUMES A CURSOR. `_apy_iterable`
        # hands one straight back -- only a generator is drained there -- so
        # this still reached Python's own `set()` holding a class of this
        # file's, and `set(map(f, xs))` reported `'Iterator' object is not
        # iterable`. `_seq_items` routes an instance through `_apy_iterable`
        # itself, so nothing is lost by asking it instead.
        items = _seq_items(h, v, "apy_to_set")
        if items is None:
            return 0
        v = items
    try:
        # A FROZENSET OF A FROZENSET IS THE SAME OBJECT, which Python's own
        # `frozenset` already answers -- `_value` is what keeps it one handle
        # here. `set()` always copies, because a set is mutable.
        return h._value(frozenset(v) if frozen else set(v))
    except TypeError as exc:
        return h._fail_like(exc)


_TABLE.update({
    "apy_set_new": _apy_set_new,
    "apy_frozenset_new": _apy_frozenset_new,
    "apy_set_push": _apy_set_push,
    "apy_set_add": _apy_set_push,
    "apy_set_discard": _apy_set_discard,
    "apy_to_set": lambda h, a: _to_set(h, a, False),
    "apy_to_frozenset": lambda h, a: _to_set(h, a, True),
    "apy_same_tuple": _apy_same_tuple,
})


#: The class machinery's bindings, registered where they are DEFINED rather
#: than in the table above. They live at the bottom of the file because they
#: need `Instance` and the re-entry into the interpreter, and a forward
#: reference in the table would be a NameError at import -- which is the same
#: protection the written-out table gives, arriving in the same way.
_TABLE.update({
    "apy_cell_new": _apy_cell_new,
    "apy_cell_get": _apy_cell_get,
    "apy_cell_set": _apy_cell_set,
    "apy_func_new": _apy_func_new,
    "apy_func_cell": _apy_func_cell,
    "apy_func_param": _apy_func_param,
    "apy_func_kwarg": _apy_func_kwarg,
    "apy_func_kwonly": _apy_func_kwonly,
    "apy_func_posonly": _apy_func_posonly,
    "apy_func_doc": _apy_func_doc,
    "apy_update": _apy_update,
    "apy_clear": _apy_clear,
    "apy_copy": _apy_copy,
    "apy_hash": _apy_hash,
    "apy_iadd": _apy_iadd,
    "apy_iop": _apy_iop,
    "apy_list_insert": _apy_list_insert,
    "apy_list_sort": _apy_list_sort,
    "apy_list_reverse": _apy_list_reverse,
    "apy_setdefault": _apy_setdefault,
    "apy_format": _apy_format,
    "apy_str_format": _apy_str_format,
    "apy_str_encode": _apy_str_encode,
    "apy_bytes_decode": _apy_bytes_decode,
    "apy_bytes_hex": _apy_bytes_hex,
    "apy_bytes_hex_n": _apy_bytes_hex_n,
    "apy_bytes_fromhex": _apy_bytes_fromhex,
    "apy_to_bytes_n": _apy_to_bytes_n,
    "apy_as_integer_ratio": _apy_as_integer_ratio,
    "apy_str_expandtabs": _apy_str_expandtabs,
    "apy_math_sqrt": _math1("apy_math_sqrt", math.sqrt),
    "apy_math_floor": _math1("apy_math_floor", math.floor,
                             dunder="__floor__"),
    "apy_math_ceil": _math1("apy_math_ceil", math.ceil, dunder="__ceil__"),
    "apy_math_trunc": _math1("apy_math_trunc", math.trunc,
                             dunder="__trunc__"),
    "apy_math_fabs": _math1("apy_math_fabs", math.fabs),
    "apy_math_isnan": _math1("apy_math_isnan", math.isnan),
    "apy_math_isinf": _math1("apy_math_isinf", math.isinf),
    "apy_math_isfinite": _math1("apy_math_isfinite", math.isfinite),
    # THE REST OF libm, matching `objects/c/_math.py` name for name.
    # `math` here IS the specification -- see this section's own
    # header -- so each is the CPython function of the same name, and
    # the C reimplements it.
    "apy_math_acos": _math1("apy_math_acos", math.acos),
    "apy_math_asin": _math1("apy_math_asin", math.asin),
    "apy_math_acosh": _math1("apy_math_acosh", math.acosh),
    "apy_math_asinh": _math1("apy_math_asinh", math.asinh),
    "apy_math_atanh": _math1("apy_math_atanh", math.atanh),
    "apy_math_cosh": _math1("apy_math_cosh", math.cosh),
    "apy_math_sinh": _math1("apy_math_sinh", math.sinh),
    "apy_math_tanh": _math1("apy_math_tanh", math.tanh),
    "apy_math_expm1": _math1("apy_math_expm1", math.expm1),
    "apy_math_log1p": _math1("apy_math_log1p", math.log1p),
    "apy_math_erf": _math1("apy_math_erf", math.erf),
    "apy_math_erfc": _math1("apy_math_erfc", math.erfc),
    "apy_math_gamma": _math1("apy_math_gamma", math.gamma),
    "apy_math_lgamma": _math1("apy_math_lgamma", math.lgamma),
    "apy_math_cbrt": _math1("apy_math_cbrt", math.cbrt),
    "apy_math_exp2": _math1("apy_math_exp2", math.exp2),
    "apy_math_ulp": _math1("apy_math_ulp", math.ulp),
    "apy_math_fmod": _math2("apy_math_fmod", math.fmod),
    "apy_math_remainder": _math2("apy_math_remainder", math.remainder),
    "apy_math_nextafter": _math2("apy_math_nextafter", math.nextafter),
    "apy_math_ldexp": _math2("apy_math_ldexp", math.ldexp),
    "apy_math_frexp": _apy_math_frexp,
    "apy_math_modf": _apy_math_modf,
    "apy_math_fsum": _apy_math_fsum,
    "apy_math_prod": _apy_math_prod,
    "apy_math_dist": _apy_math_dist,
    "apy_math_fma": _apy_math_fma,
    "apy_math_sumprod": _apy_math_sumprod,
    "apy_math_isqrt": _math1("apy_math_isqrt", math.isqrt, want_int=True),
    "apy_math_comb": _math_choose("apy_math_comb", math.comb),
    "apy_math_perm": _math_choose("apy_math_perm", math.perm),
    "apy_math_factorial": _math1("apy_math_factorial", math.factorial,
                                 want_int=True),
    "apy_math_exp": _math1("apy_math_exp", math.exp),
    "apy_math_log": _apy_math_log,
    "apy_math_log2": _math1("apy_math_log2", math.log2),
    "apy_math_log10": _math1("apy_math_log10", math.log10),
    "apy_math_sin": _math1("apy_math_sin", math.sin),
    "apy_math_cos": _math1("apy_math_cos", math.cos),
    "apy_math_tan": _math1("apy_math_tan", math.tan),
    "apy_math_atan": _math1("apy_math_atan", math.atan),
    "apy_math_degrees": _math1("apy_math_degrees", math.degrees),
    "apy_math_radians": _math1("apy_math_radians", math.radians),
    "apy_math_gcd": _math2("apy_math_gcd", math.gcd, want_int=True),
    "apy_math_lcm": _math2("apy_math_lcm", math.lcm, want_int=True),
    "apy_math_copysign": _math2("apy_math_copysign", math.copysign),
    "apy_math_pow": _math2("apy_math_pow", math.pow),
    "apy_math_atan2": _math2("apy_math_atan2", math.atan2),
    "apy_math_hypot": _math2("apy_math_hypot", math.hypot),
    "apy_math_isclose": _apy_math_isclose,
    "apy_is_integer": _apy_is_integer,
    "apy_conjugate": _apy_conjugate,
    "apy_func_default": _apy_func_default,
    "apy_env_cell": _apy_env_cell,
    "apy_call": _apy_call,
    "apy_sum_from": _apy_sum_from,
    "apy_extreme_n": _apy_extreme_n,
    "apy_extreme_or": _apy_extreme_or,
    "apy_zip_n": _apy_zip_n,
    "apy_round_to": _apy_round_to,
    "apy_is_subclass": _apy_is_subclass,
    "apy_exc_type": _apy_exc_type,
    "apy_vars": _apy_vars,
    "apy_delattr": _apy_delattr,
    "apy_iter_until": _apy_iter_until,
    "apy_call_kw": _apy_call_kw,
    "apy_unary_dunder_of": _apy_unary_dunder_of,
    "apy_method1_of": _apy_method1_of,
    "apy_type_new": _apy_type_new,
    "apy_type_set": _apy_type_set,
    "apy_instance_new": _apy_instance_new,
    "apy_exc_class_slot": _apy_exc_class_slot,
    "apy_exc_class_named_of": _apy_exc_class_named_of,
    "apy_gen_step_of": _apy_gen_step_of,
    "apy_every_of": _apy_every_of,
    "apy_big_text": _apy_big_text,
    "apy_text_of": _apy_text_of,
    "apy_exc_text_of": _apy_exc_text_of,
    "apy_seq_text_of": _apy_seq_text_of,
    "apy_dict_text_of": _apy_dict_text_of,
    "apy_set_text_of": _apy_set_text_of,
    "apy_bytes_repr": _apy_bytes_repr,
    "apy_exc_shown_of": _apy_exc_shown_of,
    "apy_special_form_class": _apy_special_form_class,
    "apy_text_result_of": _apy_text_result_of,
    "apy_traceback_of": _apy_traceback_of,
    "apy_kind_prototype": _apy_kind_prototype,
    "apy_no_attribute": _apy_no_attribute,
    "apy_kind_attr": _apy_kind_attr,
    "apy_kind_attr_of": _apy_kind_attr_of,
    "apy_kind_method_of": _apy_kind_method_of,
    "apy_object_default": _apy_object_default,
    "apy_descr_get_of": _apy_descr_get_of,
    "apy_kind_class": _apy_kind_class,
    "apy_member_descriptor": _apy_member_descriptor,
    "apy_group_select_of": _apy_group_select_of,
    "apy_str_count_in_of": _apy_str_count_in_of,
    "apy_arg_must_be_str_of": _apy_arg_must_be_str_of,
    "apy_inst_held_of": _apy_inst_held_of,
    "apy_base_text_of": _apy_base_text_of,
    "apy_big_base_text_of": _apy_big_base_text_of,
    "apy_splitlines_impl_of": _apy_splitlines_impl_of,
    "apy_extreme_of": _apy_extreme_of,
    "apy_extreme_by_of": _apy_extreme_by_of,
    "apy_binary_dunder_of": _apy_binary_dunder_of,
    "apy_is_data_descriptor_of": _apy_is_data_descriptor_of,
    "apy_descr_set_of": _apy_descr_set_of,
    "apy_slot_allows_of": _apy_slot_allows_of,
    "apy_gen_stop": _apy_gen_stop,
    "apy_exc_construct_of": _apy_exc_construct_of,
    "apy_live_agens_slot": _apy_live_agens_slot,
    "apy_tasks_slot": _apy_tasks_slot,
    "apy_affix_of": _apy_affix_of,
    "apy_str_slice_of": _apy_str_slice_of,
    "apy_split_ws_of": _apy_split_ws_of,
    "apy_split_sep_of": _apy_split_sep_of,
    "apy_str_split_impl_of": _apy_str_split_impl_of,
    "apy_type_object": _apy_type_object,
    "apy_enter": _apy_enter,
    "apy_exit": _apy_exit,
    "apy_default_getattr": _apy_default_getattr,
    "apy_default_repr": _apy_default_repr,
    "apy_default_eq": _apy_default_eq,
    "apy_default_hash": _apy_default_hash,
    "apy_default_init": _apy_default_init,
    "apy_getattr": _apy_getattr,
    "apy_is_instance": _apy_is_instance,
    "apy_alloc_block": _apy_alloc_block,
    "apy_realloc_block": _apy_realloc_block,
    "apy_free_block": _apy_free_block,
    "apy_c_lower": _ascii_pred("lower"),
    "apy_c_upper": _ascii_pred("upper"),
    "apy_c_alpha": _ascii_pred("alpha"),
    "apy_c_digit": _ascii_pred("digit"),
    "apy_c_space": _ascii_pred("space"),
    "apy_method_is_builtin": _apy_method_is_builtin,
    "apy_method_self": _apy_method_self,
    "apy_exc_register": _apy_exc_register,
    "apy_default_setattr": _apy_default_setattr,
    "apy_default_delattr": _apy_default_delattr,
    "apy_setattr": _apy_setattr,
    "apy_super": _apy_super,
    "apy_descr_applies": _apy_descr_applies,
})


# ── bytes ───────────────────────────────────────────────────────────────────
# A `bytes` value is a Python `bytes`, so every operation on it below is the
# Python operation -- the escaping in its repr, the octet ordering, the
# int-from-indexing and bytes-from-slicing asymmetry, all of them CPython's
# rather than a second description of them.
#
# The one thing that needs saying: `kind_name` answers "bytes" for these
# because `type(b'') is bytes`, and `apy_text` must give the REPR even when
# asked for `str` -- `str(b'ab')` is "b'ab'", not "ab". Both fall out of
# holding a real `bytes` and calling `repr` on it.

def _apy_bytes_literal(h, a):
    """A bytes literal read out of the program's own data.

    The LENGTH is passed rather than measured, because a bytes literal may
    contain a NUL -- `b"a\\x00b"` is three bytes -- and stopping at one would
    silently truncate every literal that has one. That is the difference
    between bytes and str, so getting it wrong here would erase the reason the
    kind exists.
    """
    addr, n = int(a[0]), int(a[1])
    return h._new(bytes(h._interp.mem.buf[addr:addr + n]))


_TABLE["apy_bytes_literal"] = _apy_bytes_literal


def _apy_weakref_register(h, a):
    """`weakref.ref(target, callback=None).__init__`'s own primitive.

    `a[0]` is the TARGET being weakly referenced; `a[1]` is the `ref`
    INSTANCE itself (`self`, from the bundled class's own `__init__`) --
    kept only to hand to `callback` later, exactly as CPython passes the
    weakref object itself, not the (by then gone) target; `a[2]` is
    `callback`'s handle -- ITSELF, not `0`, for "none": `callback=None`
    reaches here as `h._none`'s own handle (a real, non-zero cell, like
    every `None` in this runtime), not the literal `0` a bare truthiness
    check on the handle would need. Normalised to `0` here, once, so
    `_finalize`'s firing loop can keep using a plain truthiness check.

    THE TARGET IS RECORDED ONLY IN `_weakref_target`, never as an
    ordinary Python attribute on the `ref` instance (`self._target =
    obj` in the bundled class would have `_attr_store` incref it like
    any other stored attribute, making a WEAK reference hold a STRONG
    one and keeping every referent alive forever -- measured directly:
    the first version of this module did exactly that, and `del a`
    after `weakref.ref(a)` stopped finalizing `a` at all, not even
    late).
    """
    target = int(a[0])
    if target not in h._refcount:
        return h._fail("TypeError",
                       f"cannot create weak reference to "
                       f"'{h.kind_name(h._get(target, 'weakref'))}' object")
    callback = int(a[2])
    if callback == h._none:
        callback = 0
    ref_self = int(a[1])
    if ref_self in h._weakref_target:
        # ALREADY REGISTERED, and this is not a mistake: `ref.__new__`
        # hands back an existing callback-free `ref` for the same target
        # (CPython interns those -- `ref(o) is ref(o)`), and returning an
        # instance OF THE CLASS from `__new__` means `__init__` runs on
        # it a second time. Idempotent here rather than guarded there,
        # so the bundled module does not need a second flag attribute
        # whose only job is to remember this.
        return h._none
    h._weakrefs.setdefault(target, []).append((ref_self, callback))
    h._weakref_target[ref_self] = target
    return h._none


_TABLE["apy_weakref_register"] = _apy_weakref_register


def _apy_weakref_deref(h, a):
    """`weakref.ref.__call__`'s own primitive -- the live target, or
    `None` once `_finalize` has run for it. `a[0]` is the `ref`
    INSTANCE's own handle (`self`), looked up in `_weakref_target`
    rather than read off a stored attribute -- see
    `_apy_weakref_register`'s own comment for why there is no such
    attribute to read.

    `_dead`, not a refcount read, is the right question for the
    TARGET: a handle whose count is transiently at zero while
    `_protect`ed has not actually finalized yet, and CPython's own
    weakref does not go dead until the object really is gone.
    """
    target = h._weakref_target.get(int(a[0]))
    if not target or target in h._dead:
        return h._none
    return target


_TABLE["apy_weakref_deref"] = _apy_weakref_deref


def _apy_weakref_existing(h, a):
    """The live, CALLBACK-FREE `weakref.ref` already pointing at `a[0]`,
    or `None`.

    CPython INTERNS these: `ref(o) is ref(o)` is True, and
    `getweakrefcount(o)` counts one, because the second call hands back
    the first object rather than building a second. A `ref` WITH a
    callback is never shared -- two callbacks both have to run -- which
    is why only the callback-free ones are looked up here.
    """
    target = int(a[0])
    for ref_self, callback in h._weakrefs.get(target, ()):
        if not callback and ref_self not in h._dead:
            return ref_self
    return h._none


_TABLE["apy_weakref_existing"] = _apy_weakref_existing


def _apy_gc_collect(h, a):
    """`gc.collect()` -- find every object kept alive ONLY by a cycle,
    finalize it, and answer how many there were.

    CPYTHON'S OWN ALGORITHM, and it needs nothing this scheme does not
    already have -- in particular it never asks what the interpreter's
    frames are holding, which is the part `ObjectHost` cannot see:

    1. Take every tracked, still-live handle as a CANDIDATE, and give
       each a scratch count starting at its real reference count.
    2. For each candidate, walk what it HOLDS (`_held_by`) and subtract
       one from each held candidate's scratch count. That removes
       exactly the references candidates make to each other.
    3. A candidate whose scratch count is still above zero is referenced
       from OUTSIDE the candidate set -- a register, a global, a
       container that is itself externally held. It is reachable, and so
       is everything reachable FROM it.
    4. Whatever is left is reachable only from inside the set: a cycle,
       or something hanging off one. That is the garbage.

    WHY THIS IS THE MECHANISM CPYTHON HAS TOO, rather than an extra one
    bolted on: plain reference counting cannot collect a cycle, because
    every member's count is held up by another member -- which is why
    `gc.collect()` exists there at all, and why a program that never
    calls it sees a cycle survive in both.

    BEING WRONG HERE IS BEING TOO CONSERVATIVE. Every count this scheme
    over-holds (`docs/STDLIB.md` lists them -- a temporary passed into a
    call, a closure cell) makes a member look externally referenced, so
    its cycle is left alone. Nothing is collected that should not be;
    some things are not collected that could be.
    """
    del a
    scratch = {c: h._refcount[c] for c in h._refcount
               if c not in h._dead and h._refcount[c] > 0}
    held = {c: h._held_by(h._cell(c)) for c in scratch}
    for owner in scratch:
        for one in held[owner]:
            if one in scratch:
                scratch[one] -= 1
    reachable = {c for c in scratch if scratch[c] > 0}
    stack = list(reachable)
    while stack:
        for one in held[stack.pop()]:
            if one in scratch and one not in reachable:
                reachable.add(one)
                stack.append(one)
    garbage = [c for c in scratch if c not in reachable]
    doomed = frozenset(garbage)
    for one in garbage:
        # EVERY CALLBACK IN THE SET, BEFORE ANY OF THE SET IS TORN DOWN.
        # CPython's collector clears the weak references to an
        # unreachable set and runs their callbacks as one pass, and only
        # then finalizes and clears the members -- so a callback sees the
        # whole cycle still assembled, and a program counting callbacks
        # during `gc.collect()` counts all of them. Leaving each
        # member's callbacks to its own `_finalize` interleaves them with
        # the `__del__`s instead, and a callback whose own `ref` is a
        # member of the same set can find itself already dead (and so
        # never fire) by the time its turn comes.
        h._fire_weakrefs(one, doomed)
    for one in garbage:
        # `_finalize_once`, not `decref`: their counts are held up by
        # each other and will never reach zero on their own -- deciding
        # they are garbage IS the collection. The `_dead` latch keeps
        # the cascade from finalizing a member twice when one member's
        # own teardown reaches another.
        h._finalize_once(one)
    return h._int(len(garbage))


_TABLE["apy_gc_collect"] = _apy_gc_collect


def _apy_sys_getrefcount(h, a):
    """`sys.getrefcount(obj)` -- the shadow count, plus one.

    MINUS ONE, and the one being subtracted is this runtime's, not the
    program's. A `("call", ...)` member of `sys` is materialized as a
    callable value, so reaching this primitive goes through a generated
    wrapper frame (`pyn_apy_sys_getrefcount`) that BINDS the argument --
    one counted reference, held for exactly the length of a call the
    program did not write and cannot see. Subtracting it answers about
    the program's own bindings, which is what the question means and
    what CPython 3.14 answers: `x = Foo(); sys.getrefcount(x)` is 1
    there, and is 1 here.

    (CPython's older convention of answering one HIGHER, for the
    argument its own C function receives, is gone in 3.14 -- checked
    against the installed interpreter rather than assumed from the
    docs, which still describe the old behaviour.)

    ONE SHAPE READS ONE HIGHER HERE, and it is worth naming rather than
    leaving to be discovered: a function asking about its OWN PARAMETER.
    `def f(o): return sys.getrefcount(o)` called with a caller's local
    answers 1 in CPython 3.14 and 2 here, because that binding IS a
    counted reference in this runtime and CPython's interpreter borrows
    it from the caller's frame instead. Every reference a PROGRAM makes
    -- a name, a list or set membership, a dict value, an attribute --
    is counted identically in both; the difference is confined to a
    reference that exists only because a call is in progress, which is
    an interpreter's own bookkeeping and is exactly the part CPython's
    own documentation warns is implementation-specific.

    `0` for a handle this scheme never tracked (a plain int or str --
    see `_refcount`'s own comment). CPython answers a large number for
    those, since it really does count references to interned scalars;
    nothing here can, and `0` is the honest "not counted" rather than a
    number made up to look like one.
    """
    n = h.refcount(int(a[0]))
    return h._int(max(0, n - 1) if n >= 0 else 0)


_TABLE["apy_sys_getrefcount"] = _apy_sys_getrefcount


def _apy_weakref_count(h, a):
    """`weakref.getweakrefcount(obj)` -- how many live `ref`s point at
    `obj`. `0` for a handle nothing tracks, matching CPython's own
    `getweakrefcount` on an object that cannot be weakly referenced at
    all, rather than raising over a question with an obvious answer.
    """
    target = int(a[0])
    if target not in h._refcount:
        return h._int(0)
    return h._int(sum(1 for ref_self, _ in h._weakrefs.get(target, ())
                      if ref_self not in h._dead))


_TABLE["apy_weakref_count"] = _apy_weakref_count


def _apy_str_bytes(h, a):
    """The bytes of a string as an address in the interpreter's memory.

    A HOST STRING HAS NO BYTES until someone asks, so this writes them and
    answers where. That is not the C's arrangement -- there the cell already
    points at them -- but it is the same promise: NUL-terminated, and valid
    for as long as the program runs, because this memory is never released
    either.

    IT IS REACHED NOW, which the previous version of this comment said it
    would not be. `ctypes` still resolves at LINK time, but `natives_host.py`
    binds the handful of symbols a bundled module declares -- otherwise the
    oracle could not run a program that opens a file, and the compiled
    behaviour of `pathlib` would be measured against nothing.

    AN INT IS AN ADDRESS AND IS PASSED THROUGH, matching `apy_str_bytes` in
    `objects/csource.py`: C's rule for a pointer parameter admits a null pointer
    constant, and `CreateDirectoryA(path, 0)` relies on it.

    A `bytearray` IS REGISTERED FOR WRITE-BACK, and that is the one part of
    this that the C does not need. There, the cell already points at the
    bytes and a native call that fills a buffer fills the object itself.
    Here the bytes are COPIED into interpreter memory, so a call that writes
    -- `_read` is the only one so far -- would fill a copy and the program
    would see the buffer it passed still zeroed. `_native_buffers` records
    where each mutable buffer went so `natives_host` can copy it back. A
    `str` or `bytes` is not registered, because nothing may write to one.
    """
    v = h._get(a[0], "apy_str_bytes")
    if isinstance(v, bool):
        return h._fail("TypeError",
                       "a pointer argument must be str, bytes or int, not "
                       "bool")
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        raw = v.encode("utf-8", "surrogateescape")
    elif isinstance(v, (bytes, bytearray)):
        raw = bytes(v)
    else:
        return h._fail("TypeError",
                       f"a pointer argument must be str, bytes or int, not "
                       f"{h.kind_name(v)}")
    at = h._interp.mem.alloc(len(raw) + 1)
    h._interp.mem.buf[at:at + len(raw)] = raw
    h._interp.mem.buf[at + len(raw)] = 0
    if isinstance(v, bytearray):
        h._native_buffers[at] = v
    return at


_TABLE["apy_str_bytes"] = _apy_str_bytes


# ── the constructors that OWN their bytes ───────────────────────────────────
#
# `apy_lit`, `apy_str_take`, `apy_str_copy` and `apy_bytes_copy` were `static`
# in the C until stage 5 of docs/INERT-RUNTIME.md needed to name them: a port
# cannot replace a symbol `_definition_of` cannot find, and `signatures()`
# never typed one it could not see. Promoting them made them EXPORTED, and an
# exported symbol the interpreter cannot answer is the drift this whole
# arrangement exists to remove -- `test_every_exported_symbol_has_a_host_
# binding` said so immediately, which is the ratchet doing its job.
#
# THE COPY IS INVISIBLE HERE, and that is not a shortcut. The C distinguishes
# borrowing a caller's buffer (`take`) from duplicating it (`copy`) because it
# has to decide who calls `free`. A host string is an immutable Python value
# with no buffer behind it, so all three collapse to "read those bytes and
# make a str" -- and the distinction they encode is one the interpreter
# genuinely does not have.

def _apy_lit(h, a):
    """A NUL-terminated literal. Interned in the C; a value here."""
    return h._new(_read_cstr(h, int(a[0])))


def _apy_str_take(h, a):
    addr, n = int(a[0]), int(a[1])
    return h._new(bytes(h._interp.mem.buf[addr:addr + n])
                  .decode("utf-8", "surrogateescape"))


def _apy_str_copy(h, a):
    return _apy_str_take(h, a)


def _apy_bytes_copy(h, a):
    addr, n = int(a[0]), int(a[1])
    return h._new(bytes(h._interp.mem.buf[addr:addr + n]))


_TABLE["apy_lit"] = _apy_lit
_TABLE["apy_str_take"] = _apy_str_take
_TABLE["apy_str_copy"] = _apy_str_copy


def _apy_str_copy_bytes(h, a):
    """The half of `apy_str_copy` that allocates, which the IR replaces.

    The C is two functions so that this one can take an `apy_value` and match
    what an IR `ptr` compiles to; the shim keeps the pointer-typed name its 24
    call sites already use. On the host both spellings are the same read --
    there is no buffer to own, so the split that exists for the ABI's sake has
    nothing behind it here.
    """
    return _apy_str_take(h, a)


_TABLE["apy_str_copy_bytes"] = _apy_str_copy_bytes
_TABLE["apy_bytes_copy"] = _apy_bytes_copy


def _apy_delitem(h, a):
    """`del d[k]` and `del xs[i]`.

    A tuple is refused: immutability is the whole distinction from a list, and
    Python's own message for it names item DELETION rather than assignment.
    """
    seq = h._get(a[0], "apy_delitem")
    key = h._get(a[1], "apy_delitem")
    if isinstance(seq, Instance):
        # `del obj[k]` IS `obj.__delitem__(k)`. Never dispatched before, so a
        # class that wrote one had it ignored and the delete was reported as
        # unsupported -- a wrong answer about the class's own method.
        if seq.cls.find("__delitem__") is not None:
            return _user(h, lambda: h._value(seq._send("__delitem__", key)))
        # A CLASS THAT EXTENDS A BUILTIN deletes from the one it carries.
        if seq.held is not None:
            return _apy_delitem(h, [h._new(seq.held), a[1]])
    # A VIEW REFUSES IN ITS OWN WORDS. Deleting from one is not "this kind
    # has no such operation" -- a memoryview HAS `__delitem__` and it always
    # refuses, because a window onto a buffer cannot make the buffer shorter.
    if isinstance(seq, memoryview):
        return h._fail("TypeError", "cannot delete memory")
    if isinstance(seq, dict):
        try:
            hash(key)
        except TypeError as exc:
            return h._fail_like(exc)
        if key not in seq:
            return h._fail("KeyError", h._text(key, True))
        old_val = seq[key]
        del seq[key]
        # THE VALUE'S reference, mirroring `_dict_set`'s incref -- not
        # the KEY'S: `seq[key]` looked the entry up by EQUALITY, and the
        # object actually stored as the key (what a real decref needs
        # to target) need not be identical to `key` here, only equal to
        # it. Left as a documented, safe-direction gap -- see
        # `_decref_contents`'s own for the matching one on the read
        # side.
        old_h = h._referent(old_val)
        if old_h is not None:
            h.decref(old_h)
        return h._none
    # A BYTEARRAY DELETES TOO, and its buffer holds BYTES rather than values
    # -- so nothing here is a reference and there is nothing to decref.
    # `bytes` is the same Python type family and does not delete, which is
    # what the `bytearray` test rather than a wider one is for.
    if isinstance(seq, bytearray):
        if isinstance(key, slice):
            try:
                del seq[key]
            except ValueError as exc:
                return h._fail_like(exc)
            return h._none
        if not isinstance(key, (int, bool)):
            return h._fail("TypeError",
                           f"bytearray indices must be integers or slices, "
                           f"not {h.kind_name(key)}")
        i = int(key)
        if i < 0:
            i += len(seq)
        if not 0 <= i < len(seq):
            return h._fail("IndexError", "bytearray index out of range")
        del seq[i]
        return h._none
    if not isinstance(seq, list):
        return h._fail("TypeError",
                       f"'{h.kind_name(seq)}' object doesn't support item "
                       f"deletion")
    # `del xs[1:3]` AND `del xs[::2]` REMOVE A SET OF POSITIONS. Falling
    # through to the index path asked `int()` for the slice and raised out of
    # the bridge.
    if isinstance(key, slice):
        try:
            removed = seq[key]
            del seq[key]
        except ValueError as exc:
            return h._fail_like(exc)
        for item in removed:
            item_h = h._referent(item)
            if item_h is not None:
                h.decref(item_h)
        return h._none
    i = int(key)
    if i < 0:
        i += len(seq)
    if not 0 <= i < len(seq):
        return h._fail("IndexError", "list assignment index out of range")
    removed = seq[i]
    del seq[i]
    removed_h = h._referent(removed)
    if removed_h is not None:
        h.decref(removed_h)
    return h._none


_TABLE["apy_delitem"] = _apy_delitem


def _apy_del_run(h, a):
    """The ascending run of positions a slice deletes. See the ported one.

    NO HOST EQUIVALENT, and the reason is the same one `_apy_kind_attr` gives:
    the compiled runtime needs the arithmetic because it compacts a buffer by
    hand, and the interpreter holds a real Python list and deletes through
    Python's own slice -- which needs no run at all.
    """
    raise RuntimeError(
        "apy_del_run has no host equivalent: the interpreter deletes "
        "through Python's own slice")


def _apy_del_bytes(h, a):
    """`del ba[i]` on a bytearray. See `_apy_del_run`, and `_apy_delitem`,
    which answers the bytearray itself."""
    raise RuntimeError(
        "apy_del_bytes has no host equivalent: `apy_delitem` answers the "
        "bytearray through Python's own delete")


def _apy_call_spread(h, a):
    """`f(*xs)`. The arguments arrive as a LIST, not an address.

    That is the whole difference from `apy_call`: a starred call's argument
    count is a value rather than a constant, so the frontend has no number to
    emit and builds a list instead.
    """
    args = h._get(a[1], "apy_call_spread")
    try:
        return h._value(h._invoke(h._get(a[0], "apy_call_spread"), list(args)))
    except _UserFailed:
        return 0


def _apy_call_spread_kw(h, a):
    """`f(*xs, **kw)`.

    The keyword half travels SEPARATELY, as it does for every other call shape
    here: appended to the argument list it would arrive as one more
    positional. Dropping it made every keyword in a spread call vanish.
    """
    args = list(h._get(a[1], "apy_call_spread_kw"))
    kwd = h._get(a[2], "apy_call_spread_kw")
    callee = h._get(a[0], "apy_call_spread_kw")
    if not kwd:
        try:
            return h._value(h._invoke(callee, args))
        except _UserFailed:
            return 0
    # MATCHED BY NAME against the callee's parameters, THROUGH THE SAME
    # BINDER `apy_call_kw` uses -- `_invoke` alone would put every keyword
    # into `**kw`. This used to be a shorter binder of its own, which walked
    # the parameter names only while they arrived in order and knew nothing
    # about a builtin method's: `fwd("aaa".replace, "a", "b", count=1)`
    # answered 'bbb', the keyword dropped in silence.
    return _call_kwargs(h, callee, args, dict(kwd))


def _apy_extend(h, a):
    """Every element of a sequence, appended. A dict spreads its KEYS, which
    is what iterating one yields."""
    seq = h._get(a[0], "apy_extend")
    other = h._get(a[1], "apy_extend")
    # A TUPLE UNDER CONSTRUCTION counts. `(*xs, 3)` is `apy_tuple_new` plus a
    # push per element, and a starred one extends the same partly-built cell --
    # the C fills a tuple by pushing too, so refusing one here made the two
    # paths disagree on a display the compiled program built happily.
    if isinstance(seq, bytearray):
        # BYTES IN, BYTES ON THE END. Same omission as `append` above and
        # the same finder. A `bytearray` extends from anything byte-like or
        # from a sequence of ints, which is what CPython accepts.
        if isinstance(other, (bytes, bytearray, memoryview)):
            seq.extend(bytes(other))
            return h._none
        # A STR IS ITERABLE AND ITS ELEMENTS ARE NOT INTEGERS, which CPython
        # says in its own words: `expected iterable of integers; got: 'str'`.
        # That is NOT `can't extend bytearray with str`, which is what it
        # says for something it cannot walk at all.
        if isinstance(other, str):
            return h._fail("TypeError",
                           f"expected iterable of integers; got: "
                           f"'{h.kind_name(other)}'")
        # ANY ITERABLE OF INTEGERS, which is what CPython accepts: a range, a
        # cursor, a generator, a dict, a class with `__iter__` or with
        # `__len__` plus `__getitem__`. Only a list and a tuple were walked
        # here, so `ba.extend(map(int, xs))` reported that a `map` cannot
        # extend a bytearray while the compiled program extended happily.
        if isinstance(other, (list, tuple, set, frozenset, dict, range,
                              Iterator, Gen, Instance)) \
                or isinstance(other, _VIEW_TYPES):
            items = _seq_items(h, other, "apy_extend")
            if items is None:
                return 0
            for one in items:
                # THE TWO REFUSALS ARE DIFFERENT QUESTIONS, and CPython words
                # them differently: something that is not a number at all is a
                # TypeError, and a number outside a byte is a ValueError. Both
                # used to answer the second, which named the wrong problem.
                # A BOOL IS AN INT AND CPYTHON TAKES IT: `bytearray([True])`
                # is `b"\x01"`, because bool is a subclass of int.
                if not isinstance(one, int):
                    return h._fail("TypeError",
                                   f"'{h.kind_name(one)}' object cannot be "
                                   f"interpreted as an integer")
                if not 0 <= one <= 255:
                    return h._fail("ValueError",
                                   "byte must be in range(0, 256)")
            seq.extend(bytes(items))
            return h._none
        return h._fail("TypeError",
                       f"can't extend bytearray with "
                       f"{h.kind_name(other)}")
    if not isinstance(seq, (list, tuple)):
        return h._fail("AttributeError",
                       f"'{h.kind_name(seq)}' object has no attribute 'extend'")
    # A GENERATOR IS ONE TOO, and `_seq_items` below already drains it -- only
    # this gate did not know, so `xs.extend(g())` reported that a generator is
    # not iterable while the compiled program extended happily. The C reaches
    # `apy_iterable` first, which drains it; this is where the interpreter
    # does the same thing.
    # AN INSTANCE IS ONE TOO, through `__iter__` or through the older
    # `__len__` plus `__getitem__` -- `_seq_items` below knows both, and only
    # this gate did not, so `[*obj]` and `xs.extend(obj)` reported that a
    # class is not iterable about the very protocol that makes it so. A
    # memoryview for the same reason.
    if not isinstance(other, (list, tuple, set, frozenset, str, bytes,
                              bytearray, dict, memoryview,
                              Iterator, Gen, range, Instance)) \
            and not isinstance(other, _VIEW_TYPES):
        return h._fail("TypeError",
                       f"'{h.kind_name(other)}' object is not iterable")
    items = _seq_items(h, other, "apy_extend")
    if items is None:
        return 0
    if isinstance(seq, list):
        seq.extend(items)
        return h._none
    # A tuple grows by REPLACING the cell, which is what `apy_seq_push` does
    # and what keeps the handle -- and so the identity -- unchanged.
    for item in items:
        _apy_seq_push(h, [a[0], h._new(item)])
    return h._none


_TABLE["apy_call_spread"] = _apy_call_spread
_TABLE["apy_extend"] = _apy_extend


def _apy_extend_meth(h, a):
    """`xs.extend(other)` as the program wrote it -- `_apy_extend` with the
    length hint CPython asks for in front of it.

    A SEPARATE ENTRY POINT AND NOT A LINE INSIDE `_apy_extend`, because that
    one is the shared drain: `f(*xs)` reaches it, and so does the temporary
    list `set(x)` collects into, and CPython asks in NEITHER.

    NO KIND TEST -- both receivers `extend` has ask, a list and a bytearray
    alike. See the C's `apy_extend_meth`.
    """
    if not _apy_length_hint(h, a[1:2]):
        return 0
    return _apy_extend(h, a)


_TABLE["apy_extend_meth"] = _apy_extend_meth


def _apy_make_exc0(h, a):
    """An exception with NO argument, as distinct from one whose argument is
    None. `E().args` is `()` and `E(None).args` is `(None,)`, so the two
    cannot share a constructor -- see `Exc.has_arg`."""
    name = h._get(a[0], "apy_make_exc0")
    return _exc_construct(h, Exc(name, None, has_arg=False), [])


def _apy_int_literal(h, a):
    """An integer literal too large for a machine word, as decimal text.

    The IR's `const` is a machine word, so the frontend cannot emit one --
    `9223372036854775808` wrapped to its negative. Python's own `int()` parses
    it here, which is exactly what the C's digit-at-a-time parser computes.
    """
    addr, n, neg = int(a[0]), int(a[1]), int(a[2])
    digits = bytes(h._interp.mem.buf[addr:addr + n]).decode("ascii")
    return h._int(-int(digits) if neg else int(digits))


_TABLE["apy_make_exc0"] = _apy_make_exc0
_TABLE["apy_make_excn"] = _apy_make_excn
_TABLE["apy_int_literal"] = _apy_int_literal


def _apy_from_complex(h, a):
    return h._new(complex(float(a[0]), float(a[1])))


def _apy_complex_of(h, a):
    _re = h._get(a[0], "apy_complex_of")
    _im = h._get(a[1], "apy_complex_of")
    # `complex(x)` WITH ONE ARGUMENT ASKS THE CLASS. `complex(x, 0)` is
    # building from parts and has nothing to ask, which is why the frontend
    # passes None for an omitted imaginary part rather than zero.
    if isinstance(_re, Instance) and _im is None:
        _got = _dunder_number(h, _re, ("__complex__",))
        if _got is not None:
            return h._value(_got)
        if h.err is not None:
            return 0
    # AND `complex(z)` ON A COMPLEX IS `z`. One argument and nothing to
    # convert, which CPython answers by handing the receiver back.
    if isinstance(_re, complex) and _im is None:
        return h._value(_re)
    # `complex("1+2j")` -- THE STRING FORM, which is a parse rather than an
    # arithmetic conversion, and only exists for the one-argument shape.
    if isinstance(_re, str) and _im is None:
        try:
            return h._new(complex(_re))
        except ValueError:
            return h._fail("ValueError",
                           "complex() arg is a malformed string")
    if _im is None:
        a = [a[0], h._int(0)]
    """`complex(re, im)` from two runtime values.

    NOT `re + im * 1j`: that loses a signed zero, and the sign of a zero is
    observable in the repr -- `complex(0, -0.0)` is `-0j`. Two real arguments
    are stored as they are, which is what the C does for the same reason.
    """
    re = h._get(a[0], "apy_complex_of")
    im = h._get(a[1], "apy_complex_of")
    for v in (re, im):
        if not isinstance(v, (int, float, complex)):
            return h._fail("TypeError",
                           f"complex() argument must be a string or a "
                           f"number, not '{h.kind_name(v)}'")
    if not isinstance(re, complex) and not isinstance(im, complex):
        return h._new(complex(re, im))
    return h._new(complex(re) + complex(im) * 1j)


_TABLE["apy_from_complex"] = _apy_from_complex
_TABLE["apy_complex_of"] = _apy_complex_of


# ── the small builtins ──────────────────────────────────────────────────────
# Each is the Python function on the real object. `hash` is the one that
# cannot be: Python's is salted per process and the C's is not, so two runs of
# the same program would disagree with each other -- see below.

def _apy_ord(h, a):
    # A str SUBCLASS IS A str HERE TOO -- see `_held_text`.
    v = _held_text(h._get(a[0], "apy_ord"))
    if isinstance(v, bytes):
        if len(v) != 1:
            return h._fail("TypeError", "ord() expected a character, but "
                                        "string of length != 1 found")
        return h._int(v[0])
    if not isinstance(v, str):
        return h._fail("TypeError", f"ord() expected string of length 1, but "
                                    f"{h.kind_name(v)} found")
    if len(v) != 1:
        return h._fail("TypeError", "ord() expected a character, but string "
                                    "of length != 1 found")
    return h._int(ord(v))


def _apy_chr(h, a):
    """A code point as a one-character str. UTF-8 underneath, because that is
    how a str is stored -- so this is one to four bytes and `len` decodes them
    again. Refusing anything above ASCII, which is what this did, made
    `chr(233)` an error on a runtime that handles the byte fine."""
    v = h._get(a[0], "apy_chr")
    # A BOOL IS AN INT HERE, as it is in Python and as the C's
    # `apy_is_int_like` already had it: `chr(True)` is `chr(1)` is `'\x01'`.
    # Excluding it made this the ONE path of the three that refused --
    # CPython answered, the compiled backend answered, and the interpreter
    # raised `an integer is required (got type bool)`. That is the drift
    # docs/INERT-RUNTIME.md exists to end, found by porting `apy_chr` and
    # watching the two runtimes disagree about which inputs the C half is
    # even asked about.
    if not isinstance(v, int):
        return h._fail("TypeError",
                       f"an integer is required (got type {h.kind_name(v)})")
    if not 0 <= v <= 0x10FFFF:
        return h._fail("ValueError", "chr() arg not in range(0x110000)")
    return h._new(chr(v))


def _apy_callable(h, a):
    v = h._get(a[0], "apy_callable")
    if isinstance(v, (Func, Class)):
        return h._bool(True)
    if isinstance(v, Instance):
        return h._bool(v.cls.find("__call__") is not None)
    return h._bool(False)


def _apy_hasattr(h, a):
    got = _apy_getattr(h, a)
    if got:
        return h._bool(True)
    # ANSWERS rather than propagating: a missing attribute is False, not the
    # AttributeError the lookup raised.
    h.err = None
    h.err_value = None
    return h._bool(False)


def _apy_all(h, a):
    """SHORT-CIRCUITS, which is why it steps rather than indexing: with a
    generator argument that is observable, because the generator is simply not
    resumed again."""
    it = _apy_getiter(h, [a[0]])
    if not it:
        return 0
    for _ in range(100000000):
        got = _apy_step(h, [it])
        if not got:
            return 0
        item = h._get(got, "apy_all")
        if item is _STOP:
            return h._bool(True)
        if bool(item) is False:
            return h._bool(False)
    return h._bool(True)


def _apy_any(h, a):
    """SHORT-CIRCUITS, which is why it steps rather than indexing: with a
    generator argument that is observable, because the generator is simply not
    resumed again."""
    it = _apy_getiter(h, [a[0]])
    if not it:
        return 0
    for _ in range(100000000):
        got = _apy_step(h, [it])
        if not got:
            return 0
        item = h._get(got, "apy_any")
        if item is _STOP:
            return h._bool(False)
        if bool(item) is True:
            return h._bool(True)
    return h._bool(False)


def _call_key(h, keyfn, item):
    """Apply a key function to one element, or return it unchanged."""
    if keyfn is None:
        return item
    return h._invoke(keyfn, [item])


def _apy_sorted_by(h, a):
    """`sorted(xs, key=f, reverse=r)`.

    Python's own `sorted` with a key computed once per element -- so the
    stability that `reverse=True` must preserve comes from CPython rather than
    from a restatement of it. Reversing the RESULT would reverse equal
    elements too, which is the subtle half.
    """
    # ASKED FIRST, ON THE ARGUMENT AS WRITTEN, exactly as `apy_sorted` asks:
    # `key=` changes what is compared and not what was drained, so a class
    # that writes `__len__` sees the same one call either way. This is also
    # the route the VALUE form takes -- `g = sorted; g(it)` arrives here
    # with the keywords defaulted. See `_apy_length_hint`.
    if not _apy_length_hint(h, a[:1]):
        return 0
    items = _seq_items(h, h._get(a[0], "apy_sorted_by"), "apy_sorted_by")
    if items is None:
        return 0
    keyfn = h._get(a[1], "apy_sorted_by")
    keyfn = None if keyfn is None else keyfn
    reverse = bool(h._get(a[2], "apy_sorted_by"))
    try:
        if keyfn is None:
            return h._new(_driving(h, "<",
                                   lambda: sorted(items, reverse=reverse)))
        return h._new(_driving(h, "<", lambda: sorted(
            items, key=lambda v: _call_key(h, keyfn, v), reverse=reverse)))
    except TypeError as exc:
        return h._fail_like(exc)


def _extreme_by(h, a, which, name):
    items = _seq_items(h, h._get(a[0], name), name)
    if items is None:
        return 0
    keyfn = h._get(a[1], name)
    symbol = ">" if which is max else "<"
    try:
        if keyfn is None:
            return h._value(_driving(h, symbol, lambda: which(items)))
        return h._value(_driving(h, symbol, lambda: which(
            items, key=lambda v: _call_key(h, keyfn, v))))
    except _HOST_RAISES as exc:
        return h._fail_like(exc)


def _apy_min_by(h, a):
    return _extreme_by(h, a, min, "apy_min_by")


def _apy_max_by(h, a):
    return _extreme_by(h, a, max, "apy_max_by")


_TABLE.update({
    "apy_sorted_by": _apy_sorted_by,
    "apy_min_by": _apy_min_by,
    "apy_max_by": _apy_max_by,
})


#: What CPython calls the iterator over each kind of source, forward and
#: reversed. A source not in it is an `iterator` -- or a `reversed`, which is
#: the name CPython's generic reverse wrapper carries.
#:
#: A RANGE REVERSED IS STILL A RANGE ITERATOR, because `reversed(r)` in
#: CPython hands back a walk over the range with its step negated rather than
#: a wrapper around it.
_CURSOR_NAMES = {
    "list": ("list_iterator", "list_reverseiterator"),
    "tuple": ("tuple_iterator", "reversed"),
    "dict": ("dict_keyiterator", "dict_reversekeyiterator"),
    "range": ("range_iterator", "range_iterator"),
    "set": ("set_iterator", "reversed"),
    "frozenset": ("set_iterator", "reversed"),
    "bytes": ("bytes_iterator", "reversed"),
    "bytearray": ("bytearray_iterator", "reversed"),
    "memoryview": ("memory_iterator", "reversed"),
    "keys": ("dict_keyiterator", "dict_reversekeyiterator"),
    "values": ("dict_valueiterator", "dict_reversevalueiterator"),
    "items": ("dict_itemiterator", "dict_reverseitemiterator"),
    "callable": ("callable_iterator", "callable_iterator"),
}


def _cursor_named(src, mode: int) -> str:
    """What a cursor over `src` is called, which CPython takes from what it
    walks: `iter([1])` is a `list_iterator` and `reversed([1])` a
    `list_reverseiterator`.

    `str_ascii_iterator` IS A NAME OF ITS OWN IN CPYTHON, which records the
    width on the string; a str is UTF-8 bytes in the compiled runtimes, so
    the same question there is whether any byte has its high bit set. The C
    twin is `apy_cursor_name`.
    """
    rev = 1 if mode == Iterator.REV else 0
    if isinstance(src, str):
        if rev:
            return "reversed"
        return "str_ascii_iterator" if src.isascii() else "str_iterator"
    found = _CURSOR_NAMES.get(type(src).__name__)
    if found is None:
        return "reversed" if rev else "iterator"
    return found[rev]


class Iterator:
    """A cursor over something indexable -- what `iter(x)` returns.

    Not a Python iterator: the compiled runtime walks by INDEX, so an iterator
    there is a container plus a position, and reproducing that exactly is what
    keeps the two paths agreeing about a half-consumed one.
    """

    __slots__ = ("src", "i", "fn", "mode", "n0", "named")

    #: WHAT A CURSOR DOES on the way. A plain one walks; the rest apply
    #: something as they go, which is what makes `map(f, xs)` lazy -- `f` runs
    #: when the result is walked, not when it is made. `REV` walks the same
    #: source by index, counting DOWN -- see `_apy_reversed`.
    #: `CALL` is `iter(f, sentinel)`: call `fn` on every step until it
    #: answers the sentinel, which is what `src` holds. A mode and not a list
    #: drained at construction, because CPython's is lazy -- the calls happen
    #: as the walk asks for them, and the position slot is 0 until the
    #: sentinel arrives and 1 after.
    PLAIN, MAP, FILTER, ENUMERATE, ZIP, REV, CALL = range(7)

    def __init__(self, src, fn=None, mode: int = 0, start: int = 0) -> None:
        self.src = src
        self.fn = fn
        self.mode = mode
        self.i = start
        #: WHAT THIS CURSOR IS CALLED -- `list_iterator`,
        #: `dict_valueiterator`, `list_reverseiterator`. Decided here and
        #: never again, because the source does not always survive: a view's
        #: items are copied out at construction and a drained `map` becomes a
        #: plain cursor over the list it produced. A caller whose source is
        #: already gone overwrites this; see `_cursor_named`.
        self.named = _cursor_named(src, mode)
        # THE SIZE THE WALK STARTED WITH, for a dict. Growing or shrinking
        # one while iterating it rehashes the table and the walk would
        # silently skip or repeat entries, so CPython refuses. Only a dict:
        # a list may change under a walk and CPython allows it.
        self.n0 = len(src) if isinstance(src, dict) else -1


#: ONE PROTOTYPE PER CURSOR, by the mode it walks in. A cursor's prototype
#: is a cursor with no source, which is all `_kind_attr` reads of one -- the
#: mode, and the name it goes by. Nothing here is ever stepped: a prototype
#: exists to be asked which attributes its kind carries and is then thrown
#: away.
#:
#: THE NAME IS THE `named` SLOT, written over. `Iterator.__init__` computes
#: one from the source and there is no source here, so the KEY is the answer
#: -- which is why this table needs only the mode beside it. The C twin is
#: `apy_cursor_protos`, which carries a `named` integer for the same reason.
#:
#: BESIDE `Iterator` AND NOT BESIDE `_KIND_PROTOTYPES`, because the modes are
#: its own constants and naming them is the whole readability of the table.
#: `_kind_prototype` reads this at call time, as a global.
_CURSOR_PROTO_MODES = {
    "list_iterator":             Iterator.PLAIN,
    "tuple_iterator":            Iterator.PLAIN,
    "str_iterator":              Iterator.PLAIN,
    "str_ascii_iterator":        Iterator.PLAIN,
    "bytes_iterator":            Iterator.PLAIN,
    "bytearray_iterator":        Iterator.PLAIN,
    "range_iterator":            Iterator.PLAIN,
    "set_iterator":              Iterator.PLAIN,
    "memory_iterator":           Iterator.PLAIN,
    "dict_keyiterator":          Iterator.PLAIN,
    "dict_valueiterator":        Iterator.PLAIN,
    "dict_itemiterator":         Iterator.PLAIN,
    "list_reverseiterator":      Iterator.REV,
    "reversed":                  Iterator.REV,
    "dict_reversekeyiterator":   Iterator.REV,
    "dict_reversevalueiterator": Iterator.REV,
    "dict_reverseitemiterator":  Iterator.REV,
    "callable_iterator":         Iterator.CALL,
    "enumerate":                 Iterator.ENUMERATE,
    "zip":                       Iterator.ZIP,
    "map":                       Iterator.MAP,
    "filter":                    Iterator.FILTER,
}


def _apy_iterable(h, a):
    """WHAT TO WALK. `v` itself for anything indexable, and for a user object
    the iterator protocol DRAINED into a list.

    Eager where CPython is lazy, which is visible: a `for` over an infinite
    `__next__` never starts rather than never ending. Laziness needs a
    resumable frame -- the same thing `yield` needs.
    """
    v = h._get(a[0], "apy_iterable")
    if isinstance(v, Gen):
        # DRAINED: the walk below is by index and an index walk needs a
        # length. See `_apy_gen_drain` for what that costs.
        return _apy_gen_drain(h, a)
    if isinstance(v, _VIEW_TYPES):
        # A VIEW IS READ WHEN IT IS WALKED. The index walk below cannot step
        # one, and materialising here is the moment the liveness happens.
        return h._new(list(v))
    if isinstance(v, Alias):
        # AN ALIAS YIELDS ONE ITEM -- itself, starred. That is what CPython
        # does with `iter(list[int])`, and it is why `1 in list[int]` answers
        # False rather than raising: membership walks this one item and does
        # not match it. A one-item tuple is the container, so every eager
        # consumer gets the answer without a branch of its own.
        return h._new((Alias(v.origin, v.args, True),))
    if isinstance(v, Class) and v.meta is not None:
        # ITERATING A CLASS IS THE METACLASS'S BUSINESS: `for c in Color` is
        # `type(Color).__iter__(Color)`, which is how an enum lists its
        # members. A class with no metaclass cannot be iterated, and the
        # refusal further down is still the right answer for it.
        hook = v.meta.lookup("__iter__")
        if hook is not _ABSENT:
            got = h._invoke(hook, [v])
            if h.err is not None:
                return 0
            # THE SAME RULE A CLASS'S OWN `__iter__` ANSWERS UNDER: a
            # metaclass writing one is still writing `__iter__`.
            if not _is_iterator(got):
                return h._fail("TypeError",
                               f"iter() returned non-iterator of type "
                               f"'{h.kind_name(got)}'")
            return h._new(got)
    if not isinstance(v, Instance):
        return a[0]
    # A CLASS EXTENDING A BUILTIN WALKS THE BUILTIN, unless its body says
    # otherwise. This is the funnel every eager walk comes through -- a
    # comprehension, `sorted`, `list(x)`, `in` -- so the delegation belongs
    # here as well as in `_apy_iter`: patching only the latter left
    # `sorted(k for k in d)` reporting `'D' object is not iterable` while
    # `for k in d` worked, which is a worse state than neither.
    if v.cls.find("__iter__") is None and v.cls.find("__getitem__") is None             and v.held is not None:
        return h._new(list(v.held))
    try:
        got = v._send("__iter__")
    except _UserFailed:
        return 0
    if h.err is not None:
        return 0
    if got is NotImplemented:
        # No `__iter__`. THE OLDER PROTOCOL IS `__getitem__` ALONE, walked
        # from 0 until it reports IndexError -- `__len__` is no part of it
        # and CPython never consults it. Reading it as a BOUND made a class
        # whose two disagree answer differently, and made a class with a
        # `__len__` and no `__getitem__` iterable when CPython says it is
        # not. See `apy_iterable` in `objects/c/_builtins.py`, which said the
        # same thing and has stopped.
        if v.cls.find("__getitem__") is None:
            return h._fail("TypeError",
                           f"'{h.kind_name(v)}' object is not iterable")
        out = []
        for i in range(1000000):
            try:
                item = v._send("__getitem__", i)
            except _UserFailed:
                item = None
            if h.err is not None:
                if "IndexError" in _exc_chain(h, h.err[0]):
                    h.err = None
                    h.err_value = None
                    break
                return 0
            out.append(item)
        return h._new(out)
    if not isinstance(got, Instance) or got.cls.find("__next__") is None:
        # WHAT `__iter__` RETURNS MUST BE AN ITERATOR -- see `_is_iterator`,
        # which is the rule itself and which `str.join` asks for its own
        # wording. Recursing into something that is not one turned a broken
        # class into a working one that iterated something else entirely,
        # which is a wrong answer rather than a refusal.
        if _is_iterator(got):
            return _apy_iterable(h, [h._value(got)])
        return h._fail("TypeError", f"iter() returned non-iterator of type "
                                    f"'{h.kind_name(got)}'")
    out = []
    for i in range(1000000):
        try:
            item = got._send("__next__")
        except _UserFailed:
            item = None
        if h.err is not None:
            if "StopIteration" in _exc_chain(h, h.err[0]):
                h.err = None
                h.err_value = None
                break
            return 0
        out.append(item)
    return h._new(out)


class Gen:
    """A SUSPENDED FUNCTION -- what a `def` containing `yield` builds.

    Its locals live here rather than in registers, because a register does not
    survive the return a `yield` compiles to. `state` is 0 before the first
    step, k while suspended at yield k, and -1 once the body has finished.
    See `apy_gen_new` in objects/csource.py for the whole shape.
    """

    __slots__ = ("step", "slots", "state", "sent", "running", "cache",
                 "pending", "result", "coro", "builtin", "deadline",
                 "agen", "cancel", "yieldfrom", "sig", "wrapper")

    def __init__(self, step, nslots: int) -> None:
        self.step = step
        self.slots = [None] * nslots
        self.state = 0
        #: WHETHER THIS CAME FROM `async def`. The machinery is identical --
        #: a coroutine is a generator with a frame and a step -- so this
        #: decides only what the object calls itself.
        self.coro = False
        #: Which BUILT-IN coroutine this is (`sleep`, `gather`), or 0 for one
        #: lowered from an `async def`. The built-ins have no Python body and
        #: so no step to re-enter; `_apy_await_step` drives them directly.
        self.builtin = 0
        #: WHEN A `sleep` WANTS TO WAKE, on the virtual clock. Unused by
        #: everything else.
        self.deadline = 0.0
        #: CANCELLATION, for a task: 0 none, 1 asked for, 2 delivered. Three
        #: states and not two, because `cancel()` does not raise -- it asks,
        #: and the exception arrives at the task's next suspension point,
        #: which is where a `try` around the `await` inside it can catch it.
        self.cancel = 0
        #: WHETHER THIS IS AN `async def` CONTAINING `yield` -- an async
        #: generator, which is neither a coroutine nor a plain generator.
        self.agen = False
        self.sent = None
        self.running = False
        #: THE SUB-ITERATOR A `yield from` IS DELEGATING TO, or None when
        #: this generator is not inside one. `g.gi_yieldfrom` is the only way
        #: a program can see the delegation from outside, and the delegate
        #: otherwise lives in a frame slot the outside cannot name. Set and
        #: cleared by the lowered loop; see `_dyn_yield_from`.
        self.yieldfrom = None
        #: A COROUTINE_WRAPPER, which `c.__await__()` answers and nothing
        #: else builds. It has no body of its own: every `send`, `throw` and
        #: `close` goes to the coroutine in `yieldfrom`, which is what
        #: `yieldfrom` has always meant. CPython has a separate type for it
        #: and a program can see the difference.
        self.wrapper = False
        #: THE `def`'S OWN SIGNATURE, as a Func that describes it and is
        #: never called. `g.gi_code` is the code of the function the
        #: generator came from, and the STEP cannot carry it: the step is an
        #: ordinary compiled function of one argument and `_gen_step` reaches
        #: it through the ordinary call machinery, so its arity is the number
        #: of arguments that call passes and not the number the `def`
        #: declared. One per `def` and shared; see `_dyn_generator`.
        self.sig = None
        #: What a LENGTH QUERY drained. `sum(g)` walks by index and an index
        #: walk needs a length, so the first one to ask consumes the generator
        #: and every later index reads from the same list.
        self.cache = None
        #: An exception to raise AT THE SUSPENSION POINT, from `throw` or
        #: `close`. Delivered by the resume block, so a `try` inside the body
        #: catches it -- the whole difference between `throw` and raising at
        #: the call site.
        self.pending = None
        #: WHAT `return` GAVE, waiting to become `StopIteration.value`. A
        #: generator's return value is not the call's result -- the call
        #: already returned the generator -- so this is the only place it can
        #: live between the `return` and the exception that carries it.
        self.result = None


def _apy_gen_new(h, a):
    return h._new(Gen(h._get(a[0], "apy_gen_new"), int(a[1])))


def _apy_gen_sig(h, a):
    """WHICH `def` THIS GENERATOR CAME FROM, as a Func that describes its
    signature and is never called. See the `sig` slot for why the step
    cannot be it."""
    g = h._get(a[0], "apy_gen_sig")
    if isinstance(g, Gen):
        g.sig = h._get(a[1], "apy_gen_sig")
    return h._none


def _apy_gen_delegate(h, a):
    """WHAT A `yield from` IS DELEGATING TO, recorded while the loop runs.

    Cleared with None when the loop ends, which is why this takes whatever it
    is handed rather than asking for a generator.
    """
    g = h._get(a[0], "apy_gen_delegate")
    it = h._get(a[1], "apy_gen_delegate")
    if isinstance(g, Gen):
        g.yieldfrom = None if it is None else it
    return h._none


class _Slot:
    """A frame slot's contents, as the HANDLE rather than the object.

    The indirection is the point: the C stores a pointer and sees any later
    replacement of what it names. See `_apy_gen_set`.
    """

    __slots__ = ("handle",)

    def __init__(self, handle: int) -> None:
        self.handle = handle


def _apy_gen_slot(h, a):
    g = h._get(a[0], "apy_gen_slot")
    i = int(a[1])
    if 0 <= i < len(g.slots) and isinstance(g.slots[i], _Slot):
        return g.slots[i].handle
    # An UNSET slot reads as None rather than as a null: a local a `yield` has
    # not reached yet is not an error to read, and a null would be taken for a
    # pending exception by the next operation that touched it.
    return h._value(g.slots[i] if 0 <= i < len(g.slots) else None)


def _apy_gen_set(h, a):
    """Store into a frame slot.

    THE HANDLE, not the object it names. The C stores a pointer, and a value
    that is REPLACED IN ITS CELL afterwards -- which is how a tuple under
    construction grows here, see `_apy_seq_push` -- has to be visible through
    the slot. Storing the object froze the slot at whatever the value was when
    it went in, so `(await a(), await b())` spilled an empty tuple and read an
    empty tuple back however many elements had been pushed since.
    """
    g = h._get(a[0], "apy_gen_set")
    i = int(a[1])
    if 0 <= i < len(g.slots):
        h._get(a[2], "apy_gen_set")     # the null check the C's callers get
        g.slots[i] = _Slot(int(a[2]))
    return h._none


def _apy_gen_state(h, a):
    return h._get(a[0], "apy_gen_state").state


def _apy_gen_iget(h, a):
    """A slot holding a RAW machine word -- a `for` index that must survive a
    yield, kept unboxed because boxing it would allocate per iteration."""
    g = h._get(a[0], "apy_gen_iget")
    i = int(a[1])
    got = g.slots[i] if 0 <= i < len(g.slots) else 0
    return int(got) if isinstance(got, int) else 0


def _apy_gen_iset(h, a):
    g = h._get(a[0], "apy_gen_iset")
    i = int(a[1])
    if 0 <= i < len(g.slots):
        g.slots[i] = int(a[2])
    return h._none


def _apy_gen_taken(h, a):
    """What a delegated generator RETURNED, for `yield from` to answer with.

    Read off the object rather than caught as a StopIteration: the delegation
    drains rather than stepping, so there is no exception left to catch.
    """
    g = h._get(a[0], "apy_gen_taken")
    return h._value(g.result if isinstance(g, Gen) else None)


def _apy_gen_result(h, a):
    """`return v` inside the body, held until exhaustion is reported."""
    h._get(a[0], "apy_gen_result").result = h._get(a[1], "apy_gen_result")
    return h._none


def _gen_stop(h, g):
    """The exhaustion signal, carrying whatever `return` gave. A generator
    that returned nothing raises a bare StopIteration, which is not the same
    as one that returned None.

    A WRAPPER HAS NO RESULT OF ITS OWN -- the coroutine it delegates to does,
    and `next(c.__await__())` carries what `c` returned.
    """
    g = _coro_wrapped(g)
    if g.result is None:
        return h._fail("StopIteration", "")
    return h._fail_raised(Exc("StopIteration", g.result))


def _apy_gen_goto(h, a):
    h._get(a[0], "apy_gen_goto").state = int(a[1])
    return h._none


def _apy_gen_sent(h, a):
    return h._value(h._get(a[0], "apy_gen_sent").sent)


def _apy_gen_throwing(h, a):
    """Is an exception waiting to be raised here? Asked by every resume
    block."""
    return 1 if h._get(a[0], "apy_gen_throwing").pending is not None else 0


def _apy_gen_pending(h, a):
    g = h._get(a[0], "apy_gen_pending")
    got, g.pending = g.pending, None      # CLEARING, so it fires once
    return h._value(got)


def _gen_step(h, g, sent):
    """One step. Returns (value, done) or (None, None) on failure.

    A generator ALREADY RUNNING cannot be re-entered: `next(g)` from inside
    `g` is a ValueError, and without the guard it would corrupt the slots it
    is halfway through writing.
    """
    # A WRAPPER DELEGATES, having no body of its own -- see
    # `_apy_coro_wrapper`. Unwrapped HERE, which is the one funnel every
    # driver goes through: `send`, `throw` and `close` each unwrap too, but
    # `next(w)` and a `for` over one reach the step directly.
    g = _coro_wrapped(g)
    if not isinstance(g, Gen):
        h._fail("TypeError", f"'{h.kind_name(g)}' object is not a generator")
        return None, None
    if g.running:
        h._fail("ValueError", "generator already executing")
        return None, None
    if g.state < 0:
        return None, True
    g.sent = sent
    g.running = True
    try:
        out = h._invoke_obj(g.step, [g])
    except _UserFailed:
        g.state = -1
        # PEP 479: a `StopIteration` that ESCAPES a generator body becomes a
        # RuntimeError, with the original as its `__cause__`. Left alone it
        # was indistinguishable from the generator finishing normally, so a
        # bug inside the body read as a clean end of iteration -- which is the
        # entire reason the PEP exists.
        if h.err is not None and h.err[0] == "StopIteration":
            cause = h._get(_apy_error_value(h, []), "gen_step")
            wrapped = Exc("RuntimeError", "generator raised StopIteration")
            wrapped.cause = cause
            # `_fail_raised`, not `_fail`: the object has to survive so
            # `e.__cause__` reads back the StopIteration it replaced.
            h.err = None
            h._fail_raised(wrapped)
        return None, None
    finally:
        g.running = False
    # The body sets the state to -1 on its way out, so "did this call finish
    # the generator" is a question about the state AFTER it, not about the
    # value -- a generator may legitimately yield None.
    return out, g.state < 0


# ── asyncio ─────────────────────────────────────────────────────────────────
# A COROUTINE IS A GENERATOR, so this mirrors the C exactly: the same suspend
# token, the same round-robin in `gather`, the same "a built-in coroutine has
# no step function and is driven here" rule. The two files have to agree
# because `uasm run` and a compiled binary must answer the same program
# the same way -- and the ordering `gather` produces is observable.

#: Which built-in coroutine a `Gen` is. Mirrors the enum in objects/csource.py.
_CORO_SLEEP = 1
_CORO_GATHER = 2
_CORO_ANEXT = 3
_CORO_TASK = 4
_CORO_WAITFOR = 5
_CORO_VALUE = 6
_CORO_TGWAIT = 7
#: WHAT `a.asend(v)`, `a.athrow(e)` AND `a.aclose()` HAND BACK. An async
#: generator's three methods answer an AWAITABLE rather than doing the work,
#: which is the whole reason they are spelled with the `a` -- so each is a
#: built-in coroutine holding the generator and what to do to it, and the
#: work happens when it is awaited.
_CORO_ASEND = 8
_CORO_ATHROW = 9
_CORO_ACLOSE = 10

#: THE VIRTUAL CLOCK. Not a real one -- nothing here waits, and `sleep(10)`
#: returns as fast as `sleep(0)`. What it buys is ORDER: two coroutines
#: sleeping for different times must wake shortest-first, which is observable
#: from the program and which round-robin gets wrong.
#:
#: A list so the value is shared rather than rebound; the C keeps a static
#: double and the two must agree.
_NOW = [0.0]

#: WHERE A SUSPENSION LEAVES ITS WAKE TIME. Not in the value it hands back: an
#: async generator yields values of its own through that channel, and
#: `yield 0.05` would be read as a deadline. Only one coroutine steps at a
#: time, so the time need not travel as a value. `[set?, when]`.
_WAKE = [False, 0.0]


def _wake_clear():
    _WAKE[0], _WAKE[1] = False, 0.0


def _wake_note(when):
    """The EARLIEST request wins, so a short sleep beside a long one decides
    how far the clock moves."""
    if not _WAKE[0] or when < _WAKE[1]:
        _WAKE[1] = when
    _WAKE[0] = True


#: What a suspension hands back. An OPAQUE token carrying nothing, so anything
#: a program yields on purpose is unambiguous.
_SUSPEND = "<suspend>"


def _apy_coro_wrapper(h, a):
    """`c.__await__()` -- THE OBJECT BETWEEN THE COROUTINE AND WHAT DRIVES IT.

    CPython answers a `coroutine_wrapper`, not the coroutine, and a program
    can see all three differences: it is not `c`, `type(...).__name__` is
    `coroutine_wrapper`, and `dir()` of it holds `close`, `send` and `throw`
    and nothing else. Handing the coroutine back made `w is c` True and
    `dir(w)` eight names longer.
    """
    co = h._get(a[0], "apy_coro_wrapper")
    if not isinstance(co, Gen):
        return a[0]
    made = Gen(None, 0)
    made.wrapper = True
    made.yieldfrom = co
    return h._new(made)


def _apy_coro_wrapped(h, a):
    """The coroutine a wrapper stands in front of, or the value itself."""
    return h._value(_coro_wrapped(h._get(a[0], "apy_coro_wrapped")))


def _coro_wrapped(v):
    """The same, as a plain function: one test and one substitution, so it
    drops in front of an existing body rather than beside it."""
    if isinstance(v, Gen) and v.wrapper and v.yieldfrom is not None:
        return v.yieldfrom
    return v


def _apy_coro_mark(h, a):
    """Mark a freshly built frame as a coroutine. Same object, different name:
    `type(f()).__name__` is 'coroutine' and that is how a program tells one
    from a generator."""
    g = h._get(a[0], "apy_coro_mark")
    if isinstance(g, Gen):
        g.coro = True
    return a[0]


#: EVERY ASYNC GENERATOR MADE DURING A RUN, so the loop can close the ones a
#: program abandoned. `async for ...: break` leaves one suspended inside its
#: own `try` with its `finally` unrun; CPython closes those at loop shutdown.
def _apy_agen_mark(h, a):
    """Mark a frame as an ASYNC GENERATOR -- `async def` with `yield`."""
    g = h._get(a[0], "apy_agen_mark")
    if isinstance(g, Gen):
        g.coro = True
        g.agen = True
        h.live_agens.append(g)
    return a[0]


# ── descriptors ─────────────────────────────────────────────────────────────
# `property`, `classmethod`, `staticmethod`, and the protocol a user class
# joins by defining `__get__`. TWO KINDS, differing only in which side of the
# instance dict they sit on: a DATA descriptor defines `__set__` and wins over
# it, a NON-DATA one defines only `__get__` and loses to it. That is what lets
# `c.v = 4` reach a property's setter while a method can still be shadowed by
# an attribute of the same name.

PROP_PROPERTY, PROP_CLASSMETHOD, PROP_STATICMETHOD = 0, 1, 2


class Descr:
    """A runtime descriptor. Mirrors `v.p` in objects/csource.py."""

    __slots__ = ("get", "set", "del_", "kind")

    def __init__(self, fn, kind: int) -> None:
        self.get = fn
        self.set = None
        self.del_ = None
        self.kind = kind


def _is_descriptor(v) -> bool:
    if isinstance(v, Descr):
        return True
    return isinstance(v, Instance) and v.cls.find("__get__") is not None


def _is_data_descriptor(v) -> bool:
    if isinstance(v, Descr):
        return v.kind == PROP_PROPERTY
    return isinstance(v, Instance) and (v.cls.find("__set__") is not None
                                        or v.cls.find("__delete__") is not None)


def _descr_get(h, d, obj, cls):
    """Read through a descriptor: `d.__get__(obj, type)`."""
    if isinstance(d, Descr):
        if d.kind == PROP_CLASSMETHOD:
            # Bound to the CLASS the lookup started from, so `D.make()` on a
            # subclass sees `D`.
            return h._value(d.get.bind(cls))
        if d.kind == PROP_STATICMETHOD:
            return h._value(d.get)          # no binding at all
        # THROUGH THE CLASS, A PROPERTY IS ITSELF -- which is how
        # `Base.v.fget(self)` reaches the base getter from an override.
        if obj is None:
            return h._value(d)
        if d.get is None:
            return h._fail("AttributeError", "unreadable attribute")
        # Through `_user`, like every other re-entry into compiled code: a
        # getter that failed has already set the flag, and `_UserFailed` means
        # the only thing left is to answer NULL.
        return _user(h, lambda: h._value(h._invoke(d.get, [obj])))
    m = d.cls.find("__get__")
    return _user(h, lambda: h._value(
        h._invoke_obj(m.bind(d), [obj, cls if cls is not None else None])))


def _descr_write(h, d, obj, value, delete: bool = False):
    """`p.__set__(obj, v)` and `p.__delete__(obj)` as a program calls them.

    `_descr_set` answers a code the attribute machinery reads; a program
    calling the method sees None, so the two are not the same function.
    """
    half = d.del_ if delete else d.set
    if half is None:
        h._fail("AttributeError",
                "can't delete attribute" if delete else "can't set attribute")
        return None
    h._invoke(half, [obj] if delete else [obj, value])
    return None


def _pin_for_native(h, who, value) -> None:
    """Incref `value` when `who` is a HOST primitive rather than Python.

    WHY THIS EXISTS AT ALL: `apy_setattr` is on `interpreter._NON_RETAINING`,
    which certifies that every value reaching it is either counted or handed
    to interpreted Python (which counts by construction -- see
    `Interpreter._interpreted`). Every branch of `_apy_default_setattr` meets
    one of those. A property whose SETTER IS A NATIVE is the single shape
    that meets neither: the body is a host lambda this file cannot certify
    one at a time, and if one of them retained the value uncounted, retiring
    the caller's register would finalize something still reachable -- the one
    direction the whole scheme refuses to be wrong in.

    So the value is pinned instead. That LEAKS in a shape no program is
    known to write (`property(g).setter(some_builtin)`), and leaking is the
    allowed direction: late, never early.
    """
    if isinstance(who, Native):
        h.incref(h._value(value))


def _descr_set(h, d, obj, value):
    """1 when the descriptor took the write, -1 when it is not a data
    descriptor and the caller should store normally, 0 on failure."""
    if not _is_data_descriptor(d):
        return -1
    if isinstance(d, Descr):
        if d.set is None:
            h._fail("AttributeError", "can't set attribute")
            return 0
        _pin_for_native(h, d.set, value)
        if _user(h, lambda: (h._invoke(d.set, [obj, value]), 1)[1]) == 0:
            return 0
        return 0 if h.err is not None else 1
    m = d.cls.find("__set__")
    if m is None:
        return -1
    _pin_for_native(h, m, value)
    if _user(h, lambda: (h._invoke_obj(m.bind(d), [obj, value]), 1)[1]) == 0:
        return 0
    return 0 if h.err is not None else 1


def _apy_descr_new(h, a):
    """`property(f)`, `classmethod(f)`, `staticmethod(f)` -- one constructor,
    because they differ only in what reading one does."""
    return h._new(Descr(h._get(a[0], "apy_descr_new"), int(a[1])))


def _apy_prop_setter(h, a):
    """`@v.setter` -- a NEW property carrying the original getter and this
    setter. New rather than mutated: the decorator's result is rebound
    afterwards, and a program holding the old object must not see it change."""
    prop = h._get(a[0], "apy_prop_setter")
    if not isinstance(prop, Descr):
        return h._fail("AttributeError", f"'{h.kind_name(prop)}' object has "
                                         f"no attribute 'setter'")
    out = Descr(prop.get, PROP_PROPERTY)
    out.set = h._get(a[1], "apy_prop_setter")
    out.del_ = prop.del_
    return h._new(out)


def _apy_prop_deleter(h, a):
    """`@v.deleter` -- the third of the three, and the one that was missing:
    `del obj.v` had a slot to read and no way to fill it."""
    prop = h._get(a[0], "apy_prop_deleter")
    if not isinstance(prop, Descr):
        return h._fail("AttributeError", f"'{h.kind_name(prop)}' object has "
                                         f"no attribute 'deleter'")
    out = Descr(prop.get, PROP_PROPERTY)
    out.set = prop.set
    out.del_ = h._get(a[1], "apy_prop_deleter")
    return h._new(out)


def _apy_check_slots(h, a):
    """`__slots__ = ("v",)` and `v = 1` IN THE SAME BODY is a ValueError, at
    class creation. The slot and the class attribute would share a name and
    the attribute would win silently, which is why CPython refuses it rather
    than picking one -- and why it is raised rather than refused: a program
    may catch it."""
    cls = h._get(a[0], "apy_check_slots")
    if not isinstance(cls, Class):
        return h._none
    slots = cls.dict.get("__slots__")
    if slots is None:
        return h._none
    names = [slots] if isinstance(slots, str) else list(slots)
    for one in names:
        if isinstance(one, str) and one in cls.dict:
            return h._fail("ValueError",
                           f"'{one}' in __slots__ conflicts with class "
                           f"variable")
    return h._none


def _apy_set_names(h, a):
    """PEP 487: every descriptor a class body bound is TOLD ITS OWN NAME, once,
    after the body is complete. A descriptor cannot know it otherwise -- the
    expression that built it had no idea what it was about to be assigned to.
    """
    cls = h._get(a[0], "apy_set_names")
    if not isinstance(cls, Class):
        return h._none
    for key, member in list(cls.dict.items()):
        if not isinstance(member, Instance):
            continue
        hook = member.cls.find("__set_name__")
        if hook is None:
            continue
        h._invoke(hook.bind(member), [cls, key])
    return h._none


def _apy_prop_getter(h, a):
    prop = h._get(a[0], "apy_prop_getter")
    if not isinstance(prop, Descr):
        return h._fail("AttributeError", f"'{h.kind_name(prop)}' object has "
                                         f"no attribute 'getter'")
    out = Descr(h._get(a[1], "apy_prop_getter"), PROP_PROPERTY)
    out.set = prop.set
    out.del_ = prop.del_
    return h._new(out)


def _apy_func_coro(h, a):
    """Mark a FUNCTION as one whose call builds a coroutine -- `async def`.
    Recorded on the function and not only on what it returns, because
    `inspect.iscoroutinefunction(f)` asks before anything has been called."""
    f = h._get(a[0], "apy_func_coro")
    if isinstance(f, Func):
        f.coro = True
    return a[0]


def _apy_inspect_iscoroutine(h, a):
    v = h._get(a[0], "apy_inspect_iscoroutine")
    return h._bool(isinstance(v, Gen) and v.coro and not v.agen)


def _apy_inspect_isgenerator(h, a):
    v = h._get(a[0], "apy_inspect_isgenerator")
    return h._bool(isinstance(v, Gen) and not v.coro)


def _apy_inspect_isasyncgen(h, a):
    v = h._get(a[0], "apy_inspect_isasyncgen")
    return h._bool(isinstance(v, Gen) and v.agen)


def _apy_inspect_iscoroutinefunction(h, a):
    v = h._get(a[0], "apy_inspect_iscoroutinefunction")
    return h._bool(isinstance(v, Func) and v.coro)


def _apy_suspend_value(h, a):
    """The suspend token as a value, so lowered code can compare against it.
    `async for` has to tell "suspended on an await" from "produced an item",
    and those arrive through one channel."""
    return h._suspend


def _apy_aiter(h, a):
    """What `async for` actually iterates: `__aiter__` of whatever was written.

    AN ASYNC GENERATOR IS ITS OWN ITERATOR, which is why this was skipped for
    so long -- every `async for` in the suite ran over one, and `__aiter__` on
    it answers itself. A CLASS is the other half of the protocol and the more
    common one in real code: `__aiter__` hands back the object that has
    `__anext__`, and `__anext__` is an `async def`, so each item arrives
    through a coroutine that may suspend before it produces one.

    That coroutine has to survive between steps -- the loop asks for one item
    at a time and a suspension means "no item yet, ask again" -- so the class
    is wrapped in a generator cell whose slots hold the iterator and whatever
    `__anext__` call is currently in flight. Nothing else here has somewhere
    to keep it.
    """
    src = h._get(a[0], "apy_aiter")
    if isinstance(src, Gen) and src.agen:
        return h._value(src)
    if not isinstance(src, Instance) or src.cls.find("__aiter__") is None:
        return h._fail("TypeError", f"'{h.kind_name(src)}' object does not "
                                    f"support asynchronous iteration")
    it = _user(h, lambda: src._send("__aiter__"), fail=_FAILED)
    if it is _FAILED:
        return 0
    if isinstance(it, Gen) and it.agen:
        return h._value(it)
    if not isinstance(it, Instance) or it.cls.find("__anext__") is None:
        return h._fail("TypeError", f"'{h.kind_name(it)}' object does not "
                                    f"support asynchronous iteration")
    wrap = Gen(None, 2)
    wrap.agen = True
    wrap.builtin = _CORO_ANEXT
    wrap.slots[0] = it
    wrap.slots[1] = None
    return h._new(wrap)


def _anext_step(h, g):
    """One step of an `async for` over a CLASS -- see `_apy_aiter`.

    The same three answers every other step here gives: the next item, the
    suspend token, or exhaustion. `StopAsyncIteration` out of `__anext__` is
    exhaustion and not an error, which is the convention the whole protocol
    is built on.
    """
    pending = g.slots[1]
    if pending is None:
        got = _user(h, lambda: g.slots[0]._send("__anext__"), fail=_FAILED)
        if got is _FAILED:
            if h.err is not None and h.err[0] == "StopAsyncIteration":
                h.err, h.err_value = None, None
                return h._stop
            return 0
        # `__anext__` WRITTEN AS A PLAIN `def` answers the item itself rather
        # than a coroutine. Ordinary Python: `async for` awaits what it gets,
        # and awaiting something not awaitable is the error -- so a class
        # answering directly is only an unusual way to write it.
        if not isinstance(got, Gen):
            return h._value(got)
        pending = g.slots[1] = got
    value, done = _gen_step(h, pending, None)
    if done is None:
        g.slots[1] = None
        if h.err is not None and h.err[0] == "StopAsyncIteration":
            h.err, h.err_value = None, None
            return h._stop
        return 0
    if not done:
        return h._value(value)              # suspended: ask again
    g.slots[1] = None
    return h._value(pending.result)


def _apy_agen_step(h, a):
    """One step of `async for v in agen`.

    THREE OUTCOMES FROM ONE STEP: an item, a suspension, or exhaustion. A
    suspension is the opaque token and nothing else can be -- which is why the
    token stopped carrying the deadline.
    """
    g = h._get(a[0], "apy_agen_step")
    if not isinstance(g, Gen) or not g.agen:
        return h._fail("TypeError", f"'{h.kind_name(g)}' object does not "
                                    f"support asynchronous iteration")
    if g.builtin == _CORO_ANEXT:
        return _anext_step(h, g)
    if g.state < 0:
        return h._stop
    value, done = _gen_step(h, g, None)
    if done is None:
        return 0
    return h._stop if done else h._value(value)


def _asend_step(h, g):
    """ONE STEP OF `asend`, `athrow` OR `aclose`, whichever built this
    awaitable.

    THE WORK HAPPENS HERE and not where the method was called, which is what
    makes `a.asend(1)` without an `await` do nothing -- as CPython's does.
    The generator is in slot 0 and the argument in slot 1; the builtin says
    which of the three this is.

    SUSPENSION PASSES THROUGH: an `await` inside the async generator's own
    body suspends it, and the token travels out to whoever is awaiting this,
    which re-enters here and steps the generator again.

    EXHAUSTION IS `StopAsyncIteration`, the protocol's own end marker.
    """
    agen, arg = g.slots[0], g.slots[1]
    if g.builtin == _CORO_ACLOSE:
        if not _apy_gen_close(h, [h._value(agen)]):
            return 0
        g.result = None
        return h._stop
    if g.builtin == _CORO_ATHROW and arg is not None:
        # THE EXCEPTION IS DELIVERED ONCE: a second entry after a suspension
        # must not raise it again.
        g.slots[1] = None
        got = _apy_gen_throw(h, [h._value(agen), h._value(arg)])
        if not got:
            return 0
        out = h._get(got, "athrow")
        if out is _SUSPEND:
            return got
        g.result = out
        return h._stop
    if agen.state < 0:
        return h._fail("StopAsyncIteration", "")
    value, done = _gen_step(h, agen, arg)
    if done is None:
        return 0
    if value is _SUSPEND:
        return h._suspend
    if done:
        return h._fail("StopAsyncIteration", "")
    # SENT ONCE. A resumption after a suspension carries None, exactly as a
    # plain generator's does.
    g.slots[1] = None
    g.result = value
    return h._stop


def _agen_await(h, agen, arg, which):
    """THE THREE METHODS THEMSELVES, as the awaitables they answer."""
    if not isinstance(agen, Gen) or not agen.agen:
        return h._fail("TypeError",
                       f"'{h.kind_name(agen)}' object is not an async "
                       f"generator")
    g = Gen(None, 2)
    g.coro = True
    g.builtin = which
    g.slots[0] = agen
    g.slots[1] = arg
    return h._new(g)


def _apy_agen_asend(h, a):
    return _agen_await(h, h._get(a[0], "apy_agen_asend"),
                       h._get(a[1], "apy_agen_asend"), _CORO_ASEND)


def _apy_agen_athrow(h, a):
    return _agen_await(h, h._get(a[0], "apy_agen_athrow"),
                       h._get(a[1], "apy_agen_athrow"), _CORO_ATHROW)


def _apy_agen_aclose(h, a):
    return _agen_await(h, h._get(a[0], "apy_agen_aclose"), None,
                       _CORO_ACLOSE)


def _apy_await_step(h, a):
    """`await x`, one step of it.

    DELEGATION, NOT DRAINING: each step of the awaited coroutine is handed
    back so the awaiting one can suspend too. Finishing is reported as the
    `apy_stop` sentinel, the same one iteration uses, because the IR cannot
    pass a pointer to a local for an out-parameter.
    """
    awaited = h._get(a[0], "apy_await_step")
    if not isinstance(awaited, Gen):
        return h._fail("TypeError", f"object {h.kind_name(awaited)} can't be "
                                    f"used in 'await' expression")
    if awaited.step is None or awaited.step == 0:
        # A BUILT-IN coroutine: no Python body, so no step to re-enter.
        if awaited.builtin == _CORO_GATHER:
            return _gather_step(h, awaited)
        if awaited.builtin == _CORO_TASK:
            return _task_step(h, awaited)
        if awaited.builtin == _CORO_WAITFOR:
            return _waitfor_step(h, awaited)
        if awaited.builtin == _CORO_TGWAIT:
            return _tgwait_step(h, awaited)
        if awaited.builtin == _CORO_VALUE:
            awaited.result = awaited.slots[0]
            return h._stop
        if awaited.builtin in (_CORO_ASEND, _CORO_ATHROW, _CORO_ACLOSE):
            return _asend_step(h, awaited)
        # A `sleep` ALWAYS SUSPENDS AT LEAST ONCE, before the clock is even
        # consulted: `sleep(0)` is how a program hands control to the loop on
        # purpose. Returning immediately when the deadline had passed ran each
        # coroutine straight to the end -- concurrency gone, every conformance
        # case still green, because they only check the results.
        if awaited.state == 0:
            awaited.state = 1
            _wake_note(awaited.deadline)
            return h._suspend
        if _NOW[0] < awaited.deadline:
            _wake_note(awaited.deadline)
            return h._suspend
        return h._stop
    value, done = _gen_step(h, awaited, h._get(a[1], "apy_await_step"))
    if done is None:
        return 0
    return h._stop if done else h._value(value)


def _apy_asyncio_sleep(h, a):
    """`asyncio.sleep(delay)`. A coroutine that suspends once.

    It does NOT wait. There is no clock here, so what it does is suspend --
    which is the only part of it a program can observe when nothing else is
    competing for the loop.
    """
    delay = h._get(a[0], "apy_asyncio_sleep")
    g = Gen(None, 0)
    g.coro = True
    g.builtin = _CORO_SLEEP
    # THE DEADLINE IS TAKEN NOW, when `sleep` is called, not when it is first
    # awaited -- as a real loop does it.
    d = float(delay) if isinstance(delay, (int, float)) and not isinstance(delay, bool) else 0.0
    g.deadline = _NOW[0] + (d if d > 0.0 else 0.0)
    return h._new(g)


def _apy_asyncio_gather(h, a):
    """`asyncio.gather(*coros)` -- run them concurrently, results IN ARGUMENT
    ORDER rather than completion order."""
    coros = h._get(a[0], "apy_asyncio_gather")
    if not isinstance(coros, (list, tuple)):
        return h._fail("TypeError",
                       f"gather() takes coroutines, not {h.kind_name(coros)}")
    g = Gen(None, 2)
    g.coro = True
    g.builtin = _CORO_GATHER
    g.slots[0] = list(coros)
    # Full of None and filled IN PLACE, so a coroutine finishing third still
    # lands at its own index. Appending as they complete loses the ordering.
    g.slots[1] = [None] * len(coros)
    return h._new(g)


def _gather_step(h, g):
    """One round of `gather`: advance every unfinished child once.

    ROUND-ROBIN AND NOT ONE-AT-A-TIME -- the difference between running
    concurrently and merely running, and the ordering it produces is
    observable from the program.
    """
    coros, out = g.slots[0], g.slots[1]
    pending = 0
    soonest = None
    for i, child in enumerate(coros):
        if not isinstance(child, Gen):
            out[i] = child
            continue
        if child.state < 0:
            continue                      # already finished
        # CLEARED BEFORE EACH CHILD, so what is read back afterwards is that
        # child's request and not a sibling's left over from this round.
        _wake_clear()
        if child.step is None or child.step == 0:
            stepped = _apy_await_step(h, [h._new(child), h._none])
            if stepped == 0:
                return 0
            if stepped == h._stop:
                out[i] = child.result
            else:
                pending += 1
                when = _WAKE[1] if _WAKE[0] else _NOW[0]
                soonest = when if soonest is None else min(soonest, when)
            continue
        value, done = _gen_step(h, child, None)
        if done is None:
            return 0
        if done:
            out[i] = child.result
        else:
            pending += 1
            # A child that suspended without naming a time is ready now, which
            # keeps a plain `await` beside a sleep from stalling the clock.
            when = _WAKE[1] if _WAKE[0] else _NOW[0]
            soonest = when if soonest is None else min(soonest, when)
    if pending:
        # THE EARLIEST MOMENT ANY CHILD COULD MAKE PROGRESS -- the minimum,
        # not the first seen, so a short sleep wakes before a long one.
        _wake_note(soonest if soonest is not None else _NOW[0])
        return h._suspend
    g.result = out
    return h._stop


# ── tasks ───────────────────────────────────────────────────────────────────
# A TASK IS A COROUTINE THE LOOP OWNS. `await coro` runs it inside the awaiting
# one; `create_task(coro)` hands it to the loop, which runs it whenever
# anything else suspends -- and that difference is the whole of what a task is
# for. Everything below exists to give the loop somewhere to keep them and
# something to do with one that has been cancelled.


def _apy_asyncio_create_task(h, a):
    coro = h._get(a[0], "apy_asyncio_create_task")
    if not isinstance(coro, Gen) or not coro.coro:
        return h._fail("TypeError",
                       f"a coroutine was expected, got {h.kind_name(coro)}")
    t = Gen(None, 3)
    t.coro = True
    t.builtin = _CORO_TASK
    t.slots[0] = coro           # what it runs
    t.slots[1] = None           # what it returned
    t.slots[2] = None           # how it failed, if it did
    h.tasks.append(t)
    return h._new(t)


def _task_step(h, t):
    """One step of a task, whether the loop is running it in a gap or a
    program is awaiting it. The two are the same act -- which is why `await
    task` after the loop has already finished it answers what it finished
    with rather than running it again."""
    child = t.slots[0]
    if t.state < 0:
        if t.slots[2] is not None:
            return _apy_raise(h, [h._new(t.slots[2])])
        return h._stop
    if t.cancel == 1:
        t.cancel = 2
        if isinstance(child, Gen) and child.state > 0:
            # AT THE SUSPENSION POINT, which is where CPython delivers it: a
            # `try`/`except CancelledError` around the `await` inside the task
            # catches it, and being catchable there is the whole of what
            # cancellation means.
            child.pending = Exc("CancelledError", None, has_arg=False)
        else:
            # NEVER STARTED, so there is no point to raise at and the task
            # simply never runs.
            t.state = -1
            t.slots[2] = Exc("CancelledError", None, has_arg=False)
            return _apy_raise(h, [h._new(t.slots[2])])
    stepped = _apy_await_step(h, [h._new(child), h._none])
    if stepped == 0:
        # HOW IT FAILED IS KEPT, because the loop may be the one that found
        # out and the program may ask later. The flag stays set, so whoever
        # was awaiting sees it now.
        t.state = -1
        t.slots[2] = h.err_value if h.err_value is not None else (
            Exc(h.err[0], h.err[1], has_arg=bool(h.err[1]))
            if h.err else None)
        return 0
    if stepped == h._stop:
        t.state = -1
        t.slots[1] = child.result if isinstance(child, Gen) else None
        # AND WHERE `await` LOOKS FOR IT: it reads what a finished coroutine
        # returned off the cell it awaited, not off the child.
        t.result = t.slots[1]
        return h._stop
    return stepped


def _tasks_turn(h):
    """Every task the loop owns, advanced once. Called wherever the thing
    being driven suspends -- that gap is exactly when a task may run."""
    soonest, have = 0.0, False
    for t in h.tasks:
        if t.state < 0:
            continue
        _wake_clear()
        stepped = _task_step(h, t)
        if stepped == 0:
            # A TASK THAT FAILED IS NOT THE LOOP'S ERROR. It is recorded on
            # the task and raised where the task is awaited -- which is what
            # `asyncio` does, and why an un-awaited failing task is quiet.
            h.err, h.err_value = None, None
            continue
        if stepped != h._stop:
            here = _WAKE[1] if _WAKE[0] else _NOW[0]
            if not have or here < soonest:
                soonest, have = here, True
    _wake_clear()
    if have:
        _wake_note(soonest)


def _apy_task_cancel(h, a):
    """`t.cancel()` -- ASK, do not raise. The exception arrives at the task's
    next suspension point; here it is only recorded."""
    t = h._get(a[0], "apy_task_cancel")
    if not isinstance(t, Gen) or t.builtin != _CORO_TASK:
        return h._fail("AttributeError",
                       f"'{h.kind_name(t)}' object has no attribute 'cancel'")
    if t.state < 0:
        return h._new(False)
    if not t.cancel:
        t.cancel = 1
    return h._new(True)


def _apy_task_result(h, a):
    """`t.result()` -- what it returned, or the exception it ended with."""
    t = h._get(a[0], "apy_task_result")
    if not isinstance(t, Gen) or t.builtin != _CORO_TASK:
        return h._fail("AttributeError",
                       f"'{h.kind_name(t)}' object has no attribute 'result'")
    if t.state >= 0:
        return h._fail("InvalidStateError", "Result is not set.")
    if t.slots[2] is not None:
        return _apy_raise(h, [h._new(t.slots[2])])
    return h._value(t.slots[1])


def _apy_task_done(h, a):
    t = h._get(a[0], "apy_task_done")
    if not isinstance(t, Gen) or t.builtin != _CORO_TASK:
        return h._fail("AttributeError",
                       f"'{h.kind_name(t)}' object has no attribute 'done'")
    return h._new(t.state < 0)


def _apy_task_cancelled(h, a):
    t = h._get(a[0], "apy_task_cancelled")
    if not isinstance(t, Gen) or t.builtin != _CORO_TASK:
        return h._fail("AttributeError",
                       f"'{h.kind_name(t)}' object has no attribute "
                       f"'cancelled'")
    return h._new(t.state < 0 and isinstance(t.slots[2], Exc)
                  and t.slots[2].name == "CancelledError")


def _apy_asyncio_wait_for(h, a):
    """`asyncio.wait_for(coro, timeout)` -- run it, and give up at the
    deadline.

    THE CLOCK IS VIRTUAL and moves only to the next moment something can
    happen, so a timeout is not an approximation of waiting: it is a deadline
    among the others, and the loop reaching it first is exactly what "the
    coroutine took too long" means here.
    """
    coro = h._get(a[0], "apy_asyncio_wait_for")
    timeout = h._get(a[1], "apy_asyncio_wait_for")
    if not isinstance(coro, Gen):
        return h._fail("TypeError",
                       f"a coroutine was expected, got {h.kind_name(coro)}")
    t = float(timeout) if isinstance(timeout, (int, float)) \
        and not isinstance(timeout, bool) else -1.0
    w = Gen(None, 1)
    w.coro = True
    w.builtin = _CORO_WAITFOR
    w.slots[0] = coro
    # A negative or absent timeout is no deadline at all, which is what
    # `wait_for(c, None)` means.
    w.deadline = _NOW[0] + t if t >= 0.0 else -1.0
    return h._new(w)


def _waitfor_step(h, w):
    child = w.slots[0]
    limit = w.deadline
    if limit >= 0.0 and _NOW[0] >= limit and child.state >= 0:
        # THE CHILD IS STOPPED FIRST. `wait_for` promises not to leave it
        # running, and a `finally` inside it runs on the way out.
        if child.state > 0 and _apy_gen_close(h, [h._new(child)]) == 0:
            return 0
        child.state = -1
        return h._fail("TimeoutError", "")
    _wake_clear()
    stepped = _apy_await_step(h, [h._new(child), h._none])
    if stepped == 0:
        return 0
    if stepped == h._stop:
        w.result = child.result
        return h._stop
    # THE EARLIER OF the child's own wake and the deadline. Noting only the
    # child's would let the clock jump past the moment this gives up.
    if limit >= 0.0:
        _wake_note(limit)
    return h._suspend


def _apy_asyncio_taskgroup(h, a):
    """`asyncio.TaskGroup()`.

    AN OBJECT, not a coroutine: `async with` asks it for `__aenter__` and
    `__aexit__`, and `tg.create_task(...)` is an ordinary call between them.
    """
    cls = h._taskgroup_class()
    g = Instance(cls, h)
    g.dict["_tasks"] = []
    return h._new(g)


def _coro_value(h, v):
    """A coroutine that is already finished, carrying one value. `__aenter__`
    has to answer an awaitable and has nothing to wait for."""
    g = Gen(None, 1)
    g.coro = True
    g.builtin = _CORO_VALUE
    g.slots[0] = v
    return g


def _tgwait_step(h, w):
    """Leaving the `async with`: every task the group started runs to the end.

    THAT IS THE WHOLE PROMISE of a task group -- the block does not finish
    while its children are still going -- and it is why `t.result()` after
    the block is a question with an answer.

    WHAT IT DOES NOT DO is cancel the others when one of them fails, or when
    the block is left by an exception. CPython's group does both and collects
    what it cancelled into an `ExceptionGroup`; here every child runs to its
    end and a failing one is reported where it is awaited.
    """
    pending, soonest, have = 0, 0.0, False
    for t in w.slots[0]:
        if t.state < 0:
            continue
        _wake_clear()
        stepped = _task_step(h, t)
        if stepped == 0:
            return 0
        if stepped != h._stop:
            pending += 1
            here = _WAKE[1] if _WAKE[0] else _NOW[0]
            if not have or here < soonest:
                soonest, have = here, True
    _wake_clear()
    if pending:
        _wake_note(soonest if have else _NOW[0])
        return h._suspend
    # FALSE, not None: `__aexit__` answering truthy would swallow whatever
    # exception was leaving the block.
    w.result = False
    return h._stop


def _apy_asyncio_run(h, a):
    """Drive one coroutine to completion and answer what it returned."""
    coro = h._get(a[0], "apy_asyncio_run")
    if not isinstance(coro, Gen) or not coro.coro:
        return h._fail("ValueError",
                       f"a coroutine was expected, got {h.kind_name(coro)}")
    for _ in range(100_000_000):
        _wake_clear()
        stepped = _apy_await_step(h, [a[0], h._none])
        if stepped == 0:
            return 0
        if stepped == h._stop:
            # THE LOOP CLOSES WHAT THE PROGRAM ABANDONED, before answering.
            # An `async for` left by `break` holds a generator suspended
            # inside its own `try`, and its `finally` has not run yet.
            result = coro.result
            while h.live_agens:
                ag = h.live_agens.pop()
                if ag.state > 0 and _apy_gen_close(h, [h._new(ag)]) == 0:
                    return 0
            return h._value(result)
        # THE GAP IS WHERE A TASK RUNS. What was being driven has
        # suspended, so this is the moment anything the program handed to the
        # loop can make progress -- and without it `create_task` would be an
        # elaborate way of writing `await`.
        mine, had = (_WAKE[1] if _WAKE[0] else _NOW[0]), _WAKE[0]
        _tasks_turn(h)
        if had and (not _WAKE[0] or mine < _WAKE[1]):
            _WAKE[0], _WAKE[1] = True, mine
        # THE CLOCK ONLY EVER MOVES FORWARD, and only to the next moment
        # something can happen. Everything is blocked when this is reached.
        if _WAKE[0] and _WAKE[1] > _NOW[0]:
            _NOW[0] = _WAKE[1]
    return h._fail("RuntimeError", "coroutine did not finish")


def _apy_gen_next(h, a):
    value, done = _gen_step(h, h._get(a[0], "apy_gen_next"), None)
    if done is None:
        return 0
    if done:
        if int(a[2]):
            return a[1]
        return _gen_stop(h, h._get(a[0], "apy_gen_next"))
    return h._value(value)


def _apy_gen_send(h, a):
    # A WRAPPER DELEGATES, having no body of its own -- see
    # `_apy_coro_wrapper`.
    g = _coro_wrapped(h._get(a[0], "apy_gen_send"))
    v = h._get(a[1], "apy_gen_send")
    if isinstance(g, Gen) and g.state == 0 and v is not None:
        return h._fail("TypeError", "can't send non-None value to a "
                                    "just-started generator")
    value, done = _gen_step(h, g, v)
    if done is None:
        return 0
    if done:
        return _gen_stop(h, g)
    return h._value(value)


def _apy_gen_throw(h, a):
    """`g.throw(exc)` -- raise AT the suspension point, so a `try` around the
    `yield` inside the body catches it."""
    # A WRAPPER DELEGATES, having no body of its own -- see
    # `_apy_coro_wrapper`.
    g = _coro_wrapped(h._get(a[0], "apy_gen_throw"))
    exc = h._get(a[1], "apy_gen_throw")
    if not isinstance(g, Gen):
        return h._fail("AttributeError",
                       f"'{h.kind_name(g)}' object has no attribute 'throw'")
    # NOT YET STARTED, or already finished: there is no suspension point to
    # raise at, so it is raised here.
    if g.state <= 0:
        g.state = -1
        return _apy_raise(h, [a[1]])
    g.pending = exc
    value, done = _gen_step(h, g, None)
    if done is None:
        return 0
    if done:
        return h._fail("StopIteration", "")
    return h._value(value)


def _apy_gen_close(h, a):
    """A GeneratorExit at the suspension point, so a `finally` in the body
    runs. The exception is SWALLOWED if it comes back out, which is what makes
    `close` quiet; anything else the body raised propagates."""
    # A WRAPPER DELEGATES, having no body of its own -- see
    # `_apy_coro_wrapper`.
    g = _coro_wrapped(h._get(a[0], "apy_gen_close"))
    if not isinstance(g, Gen):
        return h._fail("AttributeError",
                       f"'{h.kind_name(g)}' object has no attribute 'close'")
    if g.state > 0:
        g.pending = Exc("GeneratorExit", None, has_arg=False)
        _gen_step(h, g, None)
        if h.err is not None:
            if "GeneratorExit" in _exc_chain(h, h.err[0]):
                h.err = None
                h.err_value = None
            else:
                g.state = -1
                return 0
    g.state = -1
    return h._none


def _apy_gen_drain(h, a):
    """Everything a generator will yield, as a list. EAGER, where CPython is
    lazy -- see `apy_gen_drain`."""
    g = h._get(a[0], "apy_gen_drain")
    out = []
    for _ in range(1000000):
        value, done = _gen_step(h, g, None)
        if done is None:
            return 0
        if done:
            break
        out.append(value)
    return h._new(out)


#: The exhaustion SENTINEL -- a cell, not a null, because null already means
#: "an error is set" and running out is not an error. One object, so the test
#: is an identity compare.
_STOP = object()


def _apy_is_stop(h, a):
    """Is this the exhaustion sentinel? A comparison the IR cannot spell.

    A RAW MACHINE WORD, not a handle: the caller branches on it directly, and
    a handle is a small non-zero integer -- so answering one made every test
    say "stopped" and the delegation loop ran zero times.
    """
    return 1 if h._get(a[0], "apy_is_stop") is _STOP else 0


def _apy_delegate_step(h, a):
    """ONE STEP OF A `yield from`, with the value the outer generator was SENT.

    Delegation has to STEP the inner generator rather than drain it: `got =
    yield ...` inside the inner one reads what the OUTER was sent, and a
    drained generator has already run past every such point with nothing. A
    source that is not a generator has nowhere to put the sent value and is
    simply advanced.
    """
    src = h._get(a[0], "apy_delegate_step")
    if isinstance(src, Gen):
        # THROUGH `_gen_step`, not `apy_gen_send`: the send entry point raises
        # StopIteration at the end, and delegation needs the SENTINEL so the
        # loop that drives it can stop rather than propagate.
        value, done = _gen_step(h, src, h._get(a[1], "apy_delegate_step"))
        if done is None:
            return 0
        return h._stop if done else h._value(value)
    return _apy_step(h, [a[0]])


def _apy_stop(h, a):
    return h._stop


def _apy_getiter(h, a):
    _v = h._get(a[0], "apy_getiter")
    if isinstance(_v, _VIEW_TYPES):
        # READ WHEN THE WALK STARTS -- `for k in d.keys()` sees the keys the
        # dict has now.
        return _apy_getiter(h, [h._new(list(_v))])
    """What to step. See `apy_getiter` for why iteration advances rather than
    walking by index."""
    v = h._get(a[0], "apy_getiter")
    if isinstance(v, (Gen, Iterator)):
        return a[0]               # a generator IS its own cursor
    if isinstance(v, Alias):
        # AN ALIAS YIELDS ONE ITEM -- itself, starred. Here as well as in
        # `_apy_iterable` because this is the LAZY entry: `for x in list[int]`
        # comes through here and `list(list[int])` comes through there.
        # Fixing only the eager one left the `for` reporting the alias as not
        # iterable, which is the same loop written twice.
        return _apy_getiter(h, [h._new((Alias(v.origin, v.args, True),))])
    if isinstance(v, Instance):
        try:
            got = v._send("__iter__")
        except _UserFailed:
            return 0
        if h.err is not None:
            return 0
        if got is not NotImplemented:
            if isinstance(got, Instance) and got.cls.find("__next__") is not None:
                return h._value(got)
            return _apy_getiter(h, [h._value(got)])
        # A CLASS EXTENDING A BUILTIN steps the builtin's own cursor. This is
        # the entry a GENERATOR's `for` uses -- it advances rather than
        # walking by index, because an index walk cannot survive a suspension
        # -- so without this `[k for k in d]` worked and `(k for k in d)` did
        # not, which is the same loop written two ways.
        if v.held is not None:
            return _apy_getiter(h, [h._new(v.held)])
        if v.cls.find("__getitem__") is None:
            return h._fail("TypeError",
                           f"'{h.kind_name(v)}' object is not iterable")
    # A MEMORYVIEW IS WALKED BY INDEX, which is what it already answers to
    # everywhere else: `list(mv)`, `[*mv]`, `98 in mv` and `reversed(mv)` all
    # worked and `iter(mv)` alone said it was not iterable.
    elif not isinstance(v, (list, tuple, set, frozenset, dict, str, bytes,
                            bytearray, range, memoryview)):
        return h._fail("TypeError",
                       f"'{h.kind_name(v)}' object is not iterable")
    return h._new(Iterator(v))


def _apy_step(h, a):
    it = h._get(a[0], "apy_step")
    if isinstance(it, Iterator) and it.mode == Iterator.REV:
        # BACKWARDS, from the index `reversed` started it at. The position
        # counts DOWN and -1 is exhaustion, so the forward walk's own slot
        # serves without a second one for the length. A SOURCE THAT SHRANK
        # ends the walk rather than reading past the end, which is what
        # CPython's reverse iterator does.
        if it.i < 0:
            return h._stop
        have = _apy_raw_len(h, [h._value(it.src)])
        if h.err is not None:
            return 0
        if it.i >= have:
            it.i = -1
            return h._stop
        at = it.i
        it.i -= 1
        return _apy_key_at(h, [h._value(it.src), at])
    if isinstance(it, Iterator) and it.mode != Iterator.PLAIN:
        return _step_cursor(h, it)
    if isinstance(it, Gen):
        value, done = _gen_step(h, it, None)
        if done is None:
            return 0
        return h._stop if done else h._value(value)
    if isinstance(it, Instance):
        # A user iterator: `__next__` until StopIteration, which is the
        # protocol rather than a sentinel here.
        try:
            got = it._send("__next__")
        except _UserFailed:
            got = NotImplemented
        if h.err is not None:
            if "StopIteration" in _exc_chain(h, h.err[0]):
                h.err = None
                h.err_value = None
                return h._stop
            return 0
        if got is NotImplemented:
            return h._fail("TypeError",
                           f"'{h.kind_name(it)}' object is not an iterator")
        return h._value(got)
    if not isinstance(it, Iterator):
        return h._fail("TypeError",
                       f"'{h.kind_name(it)}' object is not an iterator")
    src = it.src
    if isinstance(src, Instance):
        # Walked through `__getitem__`, ending on the IndexError the class
        # raises -- CPython's rule for the older protocol.
        try:
            got = src._send("__getitem__", it.i)
        except _UserFailed:
            got = None
        if h.err is not None:
            if "IndexError" in _exc_chain(h, h.err[0]):
                h.err = None
                h.err_value = None
                return h._stop
            return 0
        it.i += 1
        return h._value(got)
    # THE LENGTH IS READ EVERY STEP, which is the point: a body that appends
    # to the list it is walking sees the new elements.
    items = list(src)
    # A DICT THAT CHANGED SIZE UNDER THE WALK. The table is rehashed by the
    # write, so continuing would skip or repeat entries; the refusal is what
    # makes the loss impossible rather than occasional.
    if it.n0 >= 0 and isinstance(src, dict) and len(items) != it.n0:
        return h._fail("RuntimeError",
                       "dictionary changed size during iteration")
    if it.i >= len(items):
        return h._stop
    got = items[it.i]
    it.i += 1
    return h._value(got)


def _step_cursor(h, it):
    """One step of a cursor that TRANSFORMS as it goes."""
    def pull(src):
        return h._get(_apy_step(h, [h._value(src)]), "apy_step")

    if it.mode == Iterator.MAP:
        v = pull(it.src)
        if v is _STOP or h.err is not None:
            return 0 if h.err is not None else h._stop
        return _user(h, lambda: h._value(h._invoke(it.fn, [v])))
    if it.mode == Iterator.FILTER:
        for _ in range(100000000):
            v = pull(it.src)
            if h.err is not None:
                return 0
            if v is _STOP:
                return h._stop
            # `filter(None, xs)` keeps the truthy elements -- a real form, and
            # why the callable is TESTED rather than simply called.
            if it.fn is None:
                keep = v
            else:
                try:
                    keep = h._invoke(it.fn, [v])
                except _UserFailed:
                    return 0
            if keep:
                return h._value(v)
        return h._stop
    if it.mode == Iterator.ENUMERATE:
        v = pull(it.src)
        if h.err is not None:
            return 0
        if v is _STOP:
            return h._stop
        at, it.i = it.i, it.i + 1
        return h._new((at, v))
    if it.mode == Iterator.CALL:
        # `iter(f, sentinel)`. ONE CALL PER STEP, and the sentinel ends it
        # for good: CPython drops the callable the moment it arrives, so a
        # second `next()` answers StopIteration without calling again.
        if it.i:
            return h._stop
        try:
            v = h._invoke(it.fn, [])
        except _UserFailed:
            return 0
        if h.err is not None:
            return 0
        # COMPARED BY VALUE and not by identity: `iter(f, "")` stops on any
        # empty string, which is what CPython does.
        if v == it.src:
            it.i = 1
            return h._stop
        return h._value(v)
    # zip. `zip()` with no arguments is EMPTY, not endless.
    if not it.src:
        return h._stop
    row = []
    for k, cur in enumerate(it.src):
        v = pull(cur)
        if h.err is not None:
            return 0
        if v is _STOP:
            # STOPS AT THE SHORTEST, which is what makes zip lossy; `strict`
            # reports instead.
            if it.fn and k > 0:
                return h._fail("ValueError",
                               "zip() argument 2 is shorter than argument 1")
            return h._stop
        row.append(v)
    return h._new(tuple(row))


def _drain_cursor(h, it):
    """Walk a cursor to the end and BECOME a plain one over what it produced.

    Asking a lazy thing for its length is asking it to run, so it runs once
    and keeps the result -- a length query followed by an index walk sees the
    same elements. What is consumed stays consumed.
    """
    out = []
    for _ in range(100000000):
        got = _apy_step(h, [h._value(it)])
        if not got:
            return None
        v = h._get(got, "apy_step")
        if v is _STOP:
            break
        out.append(v)
    it.src, it.fn, it.mode, it.i = out, None, Iterator.PLAIN, 0
    return out


def _apy_iter(h, a):
    v = h._get(a[0], "apy_iter")
    if isinstance(v, Iterator):
        # `iter(it)` IS `it`, and the handle has to be the same one: a fresh
        # `h._new` wraps the same cursor in a new value, so the two compare
        # unequal under `is` while walking identically. The compiled path
        # returns its argument and answered True where this answered False.
        return a[0]
    if isinstance(v, Gen):
        # `iter(g)` IS `g`, so a half-consumed generator keeps its position.
        return a[0]
    if isinstance(v, Class) and v.meta is not None:
        # ITERATING A CLASS IS THE METACLASS'S BUSINESS: `for c in Color` is
        # `type(Color).__iter__(Color)`, which is how an enum lists its
        # members. A class with no metaclass cannot be iterated, and the
        # refusal further down is still the right answer for it.
        hook = v.meta.lookup("__iter__")
        if hook is not _ABSENT:
            got = h._invoke(hook, [v])
            if h.err is not None:
                return 0
            # THE SAME RULE A CLASS'S OWN `__iter__` ANSWERS UNDER: a
            # metaclass writing one is still writing `__iter__`.
            if not _is_iterator(got):
                return h._fail("TypeError",
                               f"iter() returned non-iterator of type "
                               f"'{h.kind_name(got)}'")
            return _apy_iter(h, [h._new(got)])
    if isinstance(v, Instance):
        # `iter(obj)` answers what `__iter__` did, UNCHANGED, so that
        # `iter(it) is it` holds for a class that returns self.
        #
        # UNCHANGED BUT NOT UNCHECKED: what `__iter__` answers has to BE an
        # iterator -- see `_is_iterator`, which is the rule and which the
        # eager funnel applies too. Handing a list straight back meant
        # `next(iter(obj))` then reported `'list' object is not an iterator`,
        # which is what `next` says about something that never was one rather
        # than what `iter` says about what `__iter__` gave it.
        got = v._send("__iter__")
        if h.err is not None:
            return 0
        if got is not NotImplemented:
            if not _is_iterator(got):
                return h._fail("TypeError",
                               f"iter() returned non-iterator of type "
                               f"'{h.kind_name(got)}'")
            return h._value(got)
        # A CLASS EXTENDING A BUILTIN IS ITERABLE BECAUSE THE BUILTIN IS.
        # `for k in d` over a `class D(dict)` walks its keys, and without this
        # it reported `'D' object is not iterable` about a thing whose whole
        # content is iterable -- `__iter__` is not in the body, so the miss
        # above is not the answer.
        if v.held is not None:
            return _apy_iter(h, [h._new(v.held)])
        drained = _apy_iterable(h, a)
        if not drained:
            return 0
        return _apy_iter(h, [drained])
    # A VIEW WALKS WHAT IT IS A VIEW OF, which `apy_getiter` and
    # `apy_iterable` both already do -- `iter(d.items())` refused a thing
    # `list(d.items())` accepts, on this path and on the compiled one.
    if isinstance(v, _VIEW_TYPES):
        made = Iterator(list(v))
        part = ("keys" if "keys" in type(v).__name__
                else "values" if "values" in type(v).__name__ else "items")
        made.named = _CURSOR_NAMES[part][0]
        return h._new(made)
    # A MEMORYVIEW IS WALKED BY INDEX, which is what it already answers to
    # everywhere else: `list(mv)`, `[*mv]`, `98 in mv` and `reversed(mv)` all
    # worked and `iter(mv)` alone said it was not iterable.
    if not isinstance(v, (list, tuple, set, frozenset, dict, str, bytes,
                          bytearray, range, memoryview)):
        return h._fail("TypeError",
                       f"'{h.kind_name(v)}' object is not iterable")
    return h._new(Iterator(v))


def _iter_items(v):
    """What an iterator's source yields, in order."""
    return list(v) if not isinstance(v, dict) else list(v)


def _apy_next(h, a):
    """ONE STEP, through the same protocol `for` uses.

    Anything with a position -- a generator, a cursor, a user iterator --
    advances the same way, so `next(map(f, xs))` calls `f` once rather than
    draining. Exhaustion is a StopIteration here and a sentinel there, which
    is the only difference between the two spellings.
    """
    it = h._get(a[0], "apy_next")
    if not isinstance(it, (Gen, Iterator, Instance)):
        return h._fail("TypeError",
                       f"'{h.kind_name(it)}' object is not an iterator")
    got = _apy_step(h, [a[0]])
    if not got:
        return 0
    if h._get(got, "apy_next") is _STOP:
        if int(a[2]):
            return a[1]
        # A GENERATOR CARRIES ITS RETURN VALUE OUT IN THE EXCEPTION:
        # `return "done"` becomes `StopIteration("done")`, and `e.value` is
        # how a program reads it. A bare one threw that away, while
        # `yield from` read it off the object -- two spellings disagreeing
        # about the same generator.
        # Only a GENERATOR has a return value to carry; a cursor or a user
        # iterator running out is a bare StopIteration.
        return _gen_stop(h, it) if isinstance(it, Gen)             else h._fail("StopIteration", "")
    return got


def _apy_map(h, a):
    """`map(f, xs)` -- LAZY. `f` runs when the result is walked, not when it
    is made, and a program with a side-effecting `f` can tell."""
    src = _apy_getiter(h, [a[1]])
    if not src:
        return 0
    return h._new(Iterator(h._get(src, "apy_map"),
                           h._get(a[0], "apy_map"), Iterator.MAP))


def _apy_filter(h, a):
    src = _apy_getiter(h, [a[1]])
    if not src:
        return 0
    return h._new(Iterator(h._get(src, "apy_filter"),
                           h._get(a[0], "apy_filter"), Iterator.FILTER))


def _apy_print_seq(h, a):
    """`print` reached through a VALUE. The arguments arrive as a tuple
    because a value-form has no compile-time count."""
    items = h._get(a[0], "apy_print_seq")
    h._interp._emit(" ".join(h._text(v, False) for v in items) + "\n")
    return h._none


def _apy_print_seq_with(h, a):
    """`print(*xs, sep=..., end=...)`.

    The starred form builds its arguments at run time, so it cannot use the
    stack-array entry point the fixed form does -- and routing it through
    `_apy_print_seq`, which has nowhere to put them, DROPPED the separator
    silently.
    """
    items = h._get(a[0], "apy_print_seq_with")
    # A str SUBCLASS IS A SEPARATOR -- see `_held_text`.
    sep = _held_text(h._get(a[1], "apy_print_seq_with"))
    end = _held_text(h._get(a[2], "apy_print_seq_with"))
    text = (sep if isinstance(sep, str) else " ").join(
        h._text(v, False) for v in items)
    h._interp._emit(text + (end if isinstance(end, str) else "\n"))
    return h._none


def _apy_dict_of(h, a):
    items = list(h._get(a[0], "apy_dict_of"))
    if not items:
        return h._new({})
    return _apy_to_dict(h, [h._value(items[0])])


def _apy_bytes_of(h, a):
    items = list(h._get(a[0], "apy_bytes_of"))
    if not items:
        return h._new(b"")
    return _apy_to_bytes(h, [h._value(items[0])])


def _apy_dict_fromkeys(h, a):
    """`dict.fromkeys(keys, value)` -- every key mapped to the SAME value.

    The sharing is the point and the trap: `dict.fromkeys(ks, [])` gives every
    key the same list, and appending through one key is visible through all.
    """
    keys = _seq_items(h, h._get(a[0], "apy_dict_fromkeys"), "apy_dict_fromkeys")
    if keys is None:
        return 0
    value = h._get(a[1], "apy_dict_fromkeys")
    out = {}
    for key in keys:
        if _dict_set(h, out, key, value) == 0:
            return 0
    return h._new(out)


def _apy_from_bytes_n(h, a):
    """`int.from_bytes(b, byteorder)`, unsigned -- the signed form takes a
    keyword this does not offer, and guessing would turn a large positive
    number negative."""
    b = h._get(a[0], "apy_from_bytes_n")
    order = h._get(a[1], "apy_from_bytes_n")
    if not isinstance(b, bytes):
        return h._fail("TypeError",
                       f"cannot convert '{h.kind_name(b)}' object to bytes")
    if len(b) > 8:
        return h._fail("OverflowError", "int too big to convert")
    return h._int(int.from_bytes(b, "little" if order == "little" else "big"))


def _apy_to_dict(h, a):
    src = h._get(a[0], "apy_to_dict")
    # A COPY, not the same dict: `dict(d)` is a constructor and the result is
    # a new object, which only became visible once `|=` mutated in place.
    if isinstance(src, dict):
        return h._new(dict(src))
    # `dict(d)` WHERE `d` IS A dict SUBCLASS copies the MAPPING, not the keys.
    # Iterating a dict yields keys, so the pair walk below read `dict(["a"])`
    # and reported a sequence element of the wrong length -- about a
    # `defaultdict` that is a perfectly good mapping.
    if isinstance(src, Instance) and isinstance(src.held, dict):
        return h._new(dict(src.held))
    items = _seq_items(h, src, "apy_to_dict")
    if items is None:
        return 0
    out = {}
    for i, pair in enumerate(items):
        # TWO DIFFERENT ERRORS, and Python distinguishes them: an element that
        # is not a sequence at all is a TypeError -- `dict([3, 1])` cannot
        # convert an int to a pair -- while one that IS a sequence of the
        # wrong length is a ValueError.
        # A STR IS A SEQUENCE HERE. `dict(['ab', 'cd'])` is
        # `{'a': 'b', 'c': 'd'}`, so the length check applies to it and the
        # TypeError does not.
        if not isinstance(pair, (list, tuple, str, bytes)):
            return h._fail("TypeError",
                           f"cannot convert dictionary update sequence "
                           f"element #{i} to a sequence")
        if len(pair) != 2:
            return h._fail("ValueError",
                           f"dictionary update sequence element #{i} has "
                           f"length {len(pair)}; 2 is required")
        out[pair[0]] = pair[1]
    return h._new(out)


def _apy_unpack_check(h, a):
    """`a, b = xs` -- THE ARITY, checked before anything is bound.

    Without it a short sequence read past the end and reported an IndexError
    from a subscript the program never wrote, and a long one bound the leading
    names and silently dropped the rest.
    """
    v = h._get(a[0], "apy_unpack_check")
    want, at_least = int(a[1]), int(a[2])
    # WHETHER THE SURPLUS IS COUNTED, decided by the SOURCE as CPython decides
    # it: an exact list, tuple or dict knows its own length, and everything
    # else is unpacked through the iterator protocol, which can only say that
    # there was one element too many. So `a, b = [1, 2, 3]` is `(expected 2,
    # got 3)` and `a, b = "abc"`, `range(3)` or `map(f, xs)` is `(expected
    # 2)`. Asked before the drain below replaces `v`. The shortfall is counted
    # whatever the source, because the unpack took what there was.
    counted = type(v) in (list, tuple, dict)
    # WHAT CAN BE READ BY INDEX, which is what the unpack does: `a, *b, c =
    # xs` takes `c` from `xs[-1]` without ever computing a length. A generator
    # and a cursor hold a POSITION instead, so they are drained here -- and
    # answered, because the caller indexes what this hands back. See the C's
    # `apy_unpack_check`.
    handle = _apy_iterable(h, a)
    if not handle:
        return 0
    v = h._get(handle, "apy_unpack_check")
    # THE UNPACK WORDS ITS OWN REFUSAL. `_apy_iterable` answers a value it
    # does not recognise UNCHANGED -- only a user object can fail there -- so
    # the kinds that can be walked are tested here, and CPython's wording for
    # this operation names it: `cannot unpack non-iterable int object`, not
    # `'int' object is not iterable`.
    if not isinstance(v, (list, tuple, set, frozenset, dict, str, bytes,
                          bytearray, range, memoryview, Iterator, Gen,
                          Instance)) \
            and not isinstance(v, _VIEW_TYPES):
        return h._fail("TypeError", f"cannot unpack non-iterable "
                                    f"{h.kind_name(v)} object")
    # ANYTHING NOT READ BY POSITION IS COPIED INTO A LIST. A dict subscripts
    # by KEY, so `a, b = {1: 0, 2: 0}` asked for key 0; a set cannot be
    # subscripted at all; a cursor holds a position instead of indices. AN
    # INSTANCE IS WALKED TOO, even the `__len__` plus `__getitem__` kind whose
    # subscript is its protocol: the `*rest` branch reads a SLICE, and
    # `a, *rest = obj` handed one to a `__getitem__` written for integers.
    if not isinstance(v, (list, tuple, str, bytes, bytearray, range,
                          memoryview)):
        rest = _seq_items(h, v, "apy_unpack_check")
        if rest is None:
            return 0
        handle = h._new(list(rest))
        v = h._get(handle, "apy_unpack_check")
    n = _apy_raw_len(h, [handle])
    if h.err is not None:
        return 0
    if n < want:
        return h._fail("ValueError",
                       f"not enough values to unpack (expected "
                       f"{'at least ' if at_least else ''}{want}, got {n})")
    if not at_least and n > want:
        if counted:
            return h._fail("ValueError",
                           f"too many values to unpack (expected {want}, "
                           f"got {n})")
        return h._fail("ValueError",
                       f"too many values to unpack (expected {want})")
    return handle


def _apy_name_or(h, a):
    """A module-level name that is ALSO a builtin: what the global holds, or
    the builtin when it holds nothing.

    `a[0]` is read RAW, as in `_apy_locals_put`: a zero means the global was
    never assigned or has been deleted, which is the question rather than a
    failure.
    """
    return int(a[0]) or int(a[1])


def _apy_locals_put(h, a):
    """One name into the dict `locals()` is building.

    `a[2]` is read RAW rather than through `h._get`, which is the whole point:
    every other binding treats handle 0 as "an earlier call failed and its
    result was used unchecked" and traps on it. Here 0 means the name is not
    bound on the path taken, and the name is simply left out.
    """
    if int(a[2]) == 0:
        return a[0]
    d = h._get(a[0], "apy_locals_put")
    d[h._get(a[1], "apy_locals_put")] = h._get(a[2], "apy_locals_put")
    return a[0]


def _apy_to_bytearray(h, a):
    """`bytearray(...)`. Always a fresh buffer -- see the C."""
    src = h._get(a[0], "apy_to_bytearray")
    if isinstance(src, int) and not isinstance(src, bool):
        if src < 0:
            return h._fail("ValueError", "negative count")
        return h._new(bytearray(src))
    frozen = _apy_to_bytes(h, [a[0]])
    if not frozen:
        return 0
    return h._new(bytearray(h._get(frozen, "apy_to_bytearray")))


def _apy_mview_live(h, a):
    """Whether the view still has its buffer.

    `release` drops the borrow so the object underneath can be resized again,
    and every operation on the view from then on is refused. Answers 1 while
    live, 0 having raised.
    """
    return h._int(1 if _mview_live(h, h._get(a[0], "apy_mview_live")) else 0)


def _apy_mview_from_flags(h, a):
    """`memoryview._from_flags(obj, flags)` -- PEP 688's private
    constructor. See the C's `apy_mview_from_flags` for why the flags are
    read only to be refused when absent."""
    return _apy_memoryview(h, [a[1]])


def _apy_mview_cast(h, a):
    """`m.cast(fmt)` -- THE SAME BYTES, READ AS SOMETHING ELSE.

    Only a C-contiguous view can be cast and its length has to divide by the
    new element size; Python's own `cast` says both, in the words the
    compiled halves repeat.
    """
    view = h._get(a[0], "apy_mview_cast")
    fmt = h._get(a[1], "apy_mview_cast")
    if not isinstance(view, memoryview):
        return h._fail("AttributeError",
                       f"'{h.kind_name(view)}' object has no attribute "
                       f"'cast'")
    if not _mview_live(h, view):
        return 0
    try:
        return h._new(view.cast(fmt))
    except (TypeError, ValueError) as exc:
        return h._fail_like(exc)


def _apy_mview_item(h, a):
    """One element of a view, decoded as its format says."""
    view = h._get(a[0], "apy_mview_item")
    if not _mview_live(h, view):
        return 0
    try:
        return h._value(view[int(a[1])])
    except (IndexError, ValueError) as exc:
        return h._fail_like(exc)


def _apy_mview_release(h, a):
    """`m.release()` -- HAND THE BUFFER BACK. Releasing twice is not an
    error, which is what lets a `with` block close a view a program already
    closed."""
    view = h._get(a[0], "apy_mview_release")
    if not isinstance(view, memoryview):
        return h._fail("AttributeError",
                       f"'{h.kind_name(view)}' object has no attribute "
                       f"'release'")
    view.release()
    return h._none


def _apy_mview_readonly(h, a):
    """`m.toreadonly()` -- the same window, writes refused."""
    view = h._get(a[0], "apy_mview_readonly")
    if not isinstance(view, memoryview):
        return h._fail("AttributeError",
                       f"'{h.kind_name(view)}' object has no attribute "
                       f"'toreadonly'")
    if not _mview_live(h, view):
        return 0
    return h._new(view.toreadonly())


def _apy_memoryview(h, a):
    src = h._get(a[0], "apy_memoryview")
    if isinstance(src, memoryview):
        # A RELEASED VIEW HAS NO BUFFER TO WRAP. Python's own `memoryview(m)`
        # over one raises, and answering a second released view instead hands
        # back something every use of it refuses.
        if not _mview_live(h, src):
            return 0
        return h._new(memoryview(src))
    if not isinstance(src, (bytes, bytearray)):
        return h._fail("TypeError",
                       "memoryview: a bytes-like object is required, not "
                       f"'{h.kind_name(src)}'")
    return h._new(memoryview(src))


def _apy_mview_bytes(h, a):
    """What the view shows RIGHT NOW, as bytes."""
    return h._new(bytes(h._get(a[0], "apy_mview_bytes")))


def _apy_to_bytes(h, a):
    src = h._get(a[0], "apy_to_bytes")
    if isinstance(src, (bytearray, memoryview)):
        # A COPY: `bytes(ba)` is a snapshot, and the bytearray goes on being
        # written to.
        return h._new(bytes(src))
    if isinstance(src, bytes):
        # THE RECEIVER ITSELF. `bytes(b)` on a bytes IS `b` in CPython and in
        # both compiled runtimes, and `_value` is what hands one handle back
        # per object -- `_new` minted a second and the two compared unequal.
        return h._value(src)
    if isinstance(src, str):
        return h._fail("TypeError", "string argument without an encoding")
    # `bytes(3)` is THREE ZERO BYTES, not the digit three -- the same rule
    # `bytearray(3)` follows, and the reason a count is tested before the
    # sequence walk below asks an int to be iterable.
    if isinstance(src, int) and not isinstance(src, bool):
        if src < 0:
            return h._fail("ValueError", "negative count")
        return h._new(bytes(src))
    items = _seq_items(h, src, "apy_to_bytes")
    if items is None:
        return 0
    out = bytearray()
    for item in items:
        value = int(item)
        if not 0 <= value <= 255:
            return h._fail("ValueError", "bytes must be in range(0, 256)")
        out.append(value)
    return h._new(bytes(out))


_TABLE.update({
    "apy_gen_new": _apy_gen_new,
    "apy_gen_slot": _apy_gen_slot,
    "apy_gen_set": _apy_gen_set,
    "apy_gen_state": _apy_gen_state,
    "apy_gen_iget": _apy_gen_iget,
    "apy_gen_iset": _apy_gen_iset,
    "apy_gen_result": _apy_gen_result,
    "apy_gen_taken": _apy_gen_taken,
    "apy_gen_goto": _apy_gen_goto,
    "apy_gen_delegate": _apy_gen_delegate,
    "apy_gen_sig": _apy_gen_sig,
    "apy_agen_asend": _apy_agen_asend,
    "apy_agen_athrow": _apy_agen_athrow,
    "apy_agen_aclose": _apy_agen_aclose,
    "apy_gen_sent": _apy_gen_sent,
    "apy_gen_next": _apy_gen_next,
    "apy_gen_send": _apy_gen_send,
    "apy_gen_throwing": _apy_gen_throwing,
    "apy_gen_pending": _apy_gen_pending,
    "apy_gen_throw": _apy_gen_throw,
    "apy_gen_close": _apy_gen_close,
    "apy_gen_drain": _apy_gen_drain,
    "apy_stop": _apy_stop,
    "apy_getiter": _apy_getiter,
    "apy_step": _apy_step,
    "apy_iter": _apy_iter,
    "apy_alias_unpack": _apy_alias_unpack,
    "apy_extend_arg": _apy_extend_arg,
    "apy_length_hint": _apy_length_hint,
    "apy_iterable": _apy_iterable,
    "apy_next": _apy_next,
    "apy_map": _apy_map,
    "apy_filter": _apy_filter,
    "apy_print_seq": _apy_print_seq,
    "apy_print_seq_with": _apy_print_seq_with,
    "apy_prop_deleter": _apy_prop_deleter,
    "apy_func_annotate": _apy_func_annotate,
    "apy_func_qualname": _apy_func_qualname,
    "apy_func_module": _apy_func_module,
    "apy_func_descr": _apy_func_descr,
    "apy_func_builtin": _apy_func_builtin,
    "apy_call_spread_kw": _apy_call_spread_kw,
    "apy_str_like": _apy_str_like,
    "apy_set_names": _apy_set_names,
    "apy_check_slots": _apy_check_slots,
    "apy_dict_of": _apy_dict_of,
    "apy_bytes_of": _apy_bytes_of,
    "apy_dict_fromkeys": _apy_dict_fromkeys,
    "apy_from_bytes_n": _apy_from_bytes_n,
    "apy_to_dict": _apy_to_dict,
    "apy_to_bytes": _apy_to_bytes,
    "apy_to_bytearray": _apy_to_bytearray,
    "apy_locals_put": _apy_locals_put,
    "apy_name_or": _apy_name_or,
    "apy_unpack_check": _apy_unpack_check,
    "apy_get_origin": _apy_get_origin,
    "apy_type_class": _apy_type_class,
    "apy_type_object": _apy_type_object,
    "apy_prepare": _apy_prepare,
    "apy_builtin_new": _apy_builtin_new,
    "apy_coro_wrapper": _apy_coro_wrapper,
    "apy_coro_wrapped": _apy_coro_wrapped,
    "apy_type_builtin_pending": _apy_type_builtin_pending,
    "apy_class_build": _apy_class_build,
    "apy_class_build_kw": _apy_class_build_kw,
    "apy_meta_for": _apy_meta_for,
    "apy_type_make": _apy_type_make,
    "apy_object_class": _apy_object_class,
    "apy_get_args": _apy_get_args,
    "apy_memoryview": _apy_memoryview,
    "apy_mview_bytes": _apy_mview_bytes,
})


def _apy_set_update(h, a):
    """`{*xs, y}` -- every element of a sequence ADDED to a set.

    `extend` appends and would let a duplicate through; that distinction is
    the whole difference between a set display and a list one.
    """
    target = h._get(a[0], "apy_set_update")
    src = h._get(a[1], "apy_set_update")
    if not isinstance(target, set):
        return h._fail("AttributeError",
                       f"'{h.kind_name(target)}' object has no attribute "
                       f"'update'")
    items = _seq_items(h, src, "apy_set_update")
    if items is None:
        return 0
    for item in items:
        try:
            hash(item)
        except TypeError as exc:
            return h._fail_like(exc)
        target.add(item)
    return h._none


_TABLE["apy_set_update"] = _apy_set_update



# ── the error path's state, as accessors ───────────────────────────────────
#
# THESE FOUR ARE NEW SYMBOLS, added to the C so that the storage behind them
# could move into the IR runtime -- see `runtime/errstate.py`. The C used to
# read `apy_pos_here`, `apy_err_pos` and `apy_handling` directly, which the
# machine subset cannot do; routing those reads through functions is what let
# the variables cross.
#
# THEY LAND HERE FOR THE REASON `test_every_exported_symbol_has_a_host_binding`
# gives: a compiled program can call these and `uasm run` must be able to
# as well, or the interpreter stops being the thing that adjudicates between
# the backends. The port added them to the C and this is the other half of
# that edit; the test caught it the same run.


def _apy_pos_now(h, a):
    """Where execution is. -1 if nothing has recorded a position."""
    return h.pos_here


def _apy_pos_latch(h, a):
    """Remember the current position as where the failure happened.

    ONE OPERATION, as in the C and in the ported version: `apy_fail` and
    `apy_fail2` both did this pair together, and a getter plus a setter would
    let a caller do half of it.
    """
    h.pos_err = h.pos_here
    return h.pos_err


def _apy_pos_latched(h, a):
    """Where the failure was raised. -1 if none has been."""
    return h.pos_err


def _apy_handling_now(h, a):
    """The exception whose `except` block is running, or None."""
    return h._value(h.handling)

# DEFINED ABOVE THE SWEEP BELOW, and that is not a style choice:
# `_TABLE.update` reads `globals()` once, so a binding written after it
# is never registered. Appending these four to the end of the file left
# `apy_handling_now` unbound and the ratchet said so.

# EVERY `_apy_x` REGISTERED AS `apy_x`, swept at the end of the module so the
# definitions below the explicit blocks are seen too -- a sweep in the middle
# silently missed them, and a missing binding is a symbol the compiled program
# can call and `uasm run` cannot.
#
# An explicit entry above WINS: a few names map to a function whose own name
# differs (`apy_zip2` is `_zip_two`), and the arithmetic operators share one
# parameterised implementation with no `_apy_add` to find.
#
# `tests/uasm/unit/test_objects_host` is the ratchet: every symbol
# `objects/csource.py` exports must be reachable here.
_TABLE.update({name[1:]: fn for name, fn in list(globals().items())
               if name.startswith("_apy_") and callable(fn)
               and name[1:] not in _TABLE})
