# `object`'s own behaviour, in the machine subset.
#
# STAGE 5 OF docs/INERT-RUNTIME.md. What every class inherits and most never
# override: equality by identity, a hash derived from the address, and an
# `__init__` that does nothing. A `super()` call whose base chain has run out
# lands on these, so they are reached by ordinary programs rather than only by
# the ones that name `object`.
#
# THEY PORT EARLY BECAUSE THEY DEPEND ON NOTHING. Each is one expression over
# a pointer, with no cell layout to know beyond the address itself -- so they
# could move as soon as `apy_from_bool` and `apy_from_int` did, which is to
# say as soon as the singletons went.


def apy_default_eq(a: ptr, b: ptr) -> ptr:
    """`object.__eq__`: equal to itself, and NOT UNEQUAL to anything else.

    NEARLY THE SAME COMPARISON AS `apy_is`, and deliberately a separate
    function. They are short enough to share and they answer different
    questions: `is` is what a program WRITES, and this is what a class
    INHERITS. A class overriding `__eq__` replaces this and leaves `is`
    alone, so a shared implementation would tie two things together that are
    supposed to come apart -- and they no longer even agree, which is the
    change here.

    NotImplemented AND NOT False FOR A PAIR IT CANNOT JUDGE.
    `object_richcompare` answers Py_True for identity and NotImplemented for
    everything else: it does not claim two different objects are unequal, it
    declines to say. `object.__eq__(1, 2)` is NotImplemented in CPython and
    was False here, and False is a CLAIM where CPython makes none.

    THE `==` OPERATOR IS NOT THIS. `apy_eq` asks the written dunders and
    falls back to a raw comparison, so `a == b` for two plain instances still
    answers False -- the identity fallback lives there, where CPython's
    `do_richcompare` keeps it, and not in the method.
    """
    if u64(a) == u64(b):
        return apy_from_bool(1)
    return apy_notimplemented()


def apy_default_hash(v: ptr) -> ptr:
    """`object.__hash__`: the address, shifted.

    SHIFTED BY THREE because every cell is eight-aligned, so the low three
    bits are always zero and would make every hash a multiple of eight --
    which collides catastrophically in a table that masks off low bits to
    choose a bucket. CPython rotates for the same reason.

    CONSTANT FOR THE OBJECT'S LIFETIME, which it must be: nothing here is
    ever freed or moved, so an address is a stable identity. A collector that
    moved cells would break this before it broke anything else.
    """
    return apy_from_int(i64(u64(v) >> 3))


def apy_default_init(v: ptr) -> ptr:
    """`object.__init__`: accepts anything, does nothing, answers None.

    IT HAS TO EXIST rather than being skipped, because `super().__init__()`
    at the end of a user's `__init__` has to land somewhere -- and a class
    whose base chain runs out is the common case, not an edge one.
    """
    return apy_none()


def apy_typing_mark(obj: ptr) -> ptr:
    """A `typing` decorator: mark the object and hand it straight back.

    INERT BY DESIGN. `@final`, `@override` and their relatives say something
    to a type checker and nothing at all to a running program, so the honest
    implementation is the identity function. Returning something else -- a
    wrapper, a copy -- would make a decorator that is documented to have no
    runtime effect have one.
    """
    return obj


def apy_name_or(got: ptr, fallback: ptr) -> ptr:
    """`got` unless it is null, in which case `fallback`.

    NULL IS NOT A VALUE anywhere in this runtime -- it is how a failed lookup
    answers -- so this is the "was there one" test rather than a truthiness
    test. A `got` holding `None`, `0` or an empty list is a real answer and
    is returned unchanged.
    """
    if got:
        return got
    return fallback
