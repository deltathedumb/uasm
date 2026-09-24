# COVERAGE: @dataclass bare and called; init/repr/eq/order/unsafe_hash/frozen/
# match_args/kw_only and their illegal combinations; field() with default,
# default_factory, init, repr, hash, compare, metadata, kw_only; the mutable-
# default refusal; fields(); is_dataclass(); asdict/astuple RECURSING;
# replace(); make_dataclass; InitVar; ClassVar exclusion; __post_init__;
# inheritance and field override order; the __hash__ table; the split
# overwrite policy; the non-default-after-default error; and the four __init__
# arity errors; the generated methods' names, qualnames and `__wrapped__`.
# NOT covered here: slots/weakref_slot (refused by name, tested
# as refusals), field(doc=), Field[int], and abc.update_abstractmethods.
#
# TWO THINGS ARE DELIBERATELY NOT COMPARED, both because they cannot be:
#   * repr(MISSING) and repr(a Field) embed an object address.
#   * a NESTED class's __qualname__ is the bare name under uasm, so a
#     nested dataclass's repr differs. Every dataclass here is top-level.
import typing

import dataclasses
from dataclasses import dataclass, field, fields, InitVar

# ---- the basic shape -------------------------------------------------------
@dataclass
class Point:
    x: int
    y: int = 2


p = Point(1)
print(p, p.x, p.y)
print(Point(1, 5), Point(1, 5) == Point(1, 5), Point(1) == Point(1, 3))
print(dataclasses.is_dataclass(Point), dataclasses.is_dataclass(p))
print([f.name for f in fields(Point)], [f.type.__name__ for f in fields(Point)])
print(Point.__match_args__)


@dataclass()
class Called:
    a: int


print(Called(1))

# EQUALITY IS EXACT-CLASS, and answers NotImplemented -> False, not a compare.
@dataclass
class Other:
    x: int
    y: int = 2


print(Point(1, 2) == Other(1, 2), Point(1, 2) == 7, Point(1, 2) != Other(1, 2))


# ---- the default dataclass is UNHASHABLE ----------------------------------
print("__hash__" in vars(Point), vars(Point)["__hash__"] is None)
try:
    hash(Point(1))
except TypeError as exc:
    print("TypeError:", exc)


@dataclass(frozen=True)
class Frozen:
    x: int


print(hash(Frozen(1)) == hash(Frozen(1)), len({Frozen(1), Frozen(1)}))
try:
    Frozen(1).x = 5
except dataclasses.FrozenInstanceError as exc:
    print("FrozenInstanceError:", exc)
try:
    del Frozen(1).x
except dataclasses.FrozenInstanceError as exc:
    print("FrozenInstanceError:", exc)
# COMPARED NOW: a bundled module's exception class keeps its place in the
# hierarchy across the splice's rename. `FrozenInstanceError` inheriting
# AttributeError is what lets `except AttributeError` and hasattr-style probing
# behave the way real code expects.
print(issubclass(dataclasses.FrozenInstanceError, AttributeError))


@dataclass(eq=False)
class NoEq:
    x: int


print("__hash__" in vars(NoEq), NoEq(1) == NoEq(1))


@dataclass(unsafe_hash=True)
class Unsafe:
    x: int


print(hash(Unsafe(1)) == hash(Unsafe(1)), Unsafe(1) == Unsafe(1))


@dataclass(eq=False, unsafe_hash=True)
class HashNoEq:
    x: int


print(hash(HashNoEq(1)) == hash(HashNoEq(1)), HashNoEq(1) == HashNoEq(1))


# ---- field() --------------------------------------------------------------
@dataclass
class Bag:
    tags: list = field(default_factory=list)
    label: str = "x"
    seen: int = field(default=0, repr=False)
    computed: int = field(init=False, default=9)
    meta: int = field(default=1, metadata={"unit": "cm"})


b = Bag()
print(b)
print(b.tags, b.computed, b.seen)
# ONE LIST PER INSTANCE, which is the whole reason the factory is a callable.
b.tags.append(1)
print(Bag().tags, b.tags)
print([f.name for f in fields(Bag)])
print(fields(Bag)[4].metadata, fields(Bag)[4].metadata == {"unit": "cm"})
print(fields(Bag)[0].metadata == {}, len(fields(Bag)[0].metadata))
try:
    fields(Bag)[4].metadata["unit"] = "m"
except TypeError as exc:
    print("TypeError:", exc)
# `init=False` WITH A PLAIN DEFAULT ASSIGNS NOTHING -- the value is read from
# the class, so it is not in the instance dict.
print("computed" in vars(b), Bag.computed)

# `field(3)` IS NOT TESTED: `field` is keyword-only and this compiler does not
# enforce keyword-only at run time, so the positional is accepted here and
# refused by CPython. The module states it; it is a permissive superset in one
# spot rather than a wrong answer.
try:
    field(default=1, default_factory=list)
except ValueError as exc:
    print("ValueError:", exc)


# A MUTABLE DEFAULT IS REFUSED, which the module's whole docstring is about.
# Written as real class statements because `make_dataclass` cannot work here.
try:
    @dataclass
    class MutList:
        xs: list = []
except ValueError as exc:
    print("ValueError:", exc)
try:
    @dataclass
    class MutDict:
        xs: dict = {}
except ValueError as exc:
    print("ValueError:", exc)
try:
    @dataclass
    class MutSet:
        xs: set = field(default=set())
except ValueError as exc:
    print("ValueError:", exc)


# `hash` IS THREE-STATE: None follows `compare`.
@dataclass(frozen=True)
class Hashing:
    a: int
    b: int = field(compare=False)
    c: int = field(hash=False)
    d: int = field(hash=True, compare=False)


h1 = Hashing(1, 2, 3, 4)
h2 = Hashing(1, 99, 3, 4)
h3 = Hashing(1, 2, 99, 4)
print(h1 == h2, hash(h1) == hash(h2))
print(h1 == h3, hash(h1) == hash(h3))


# ---- ordering -------------------------------------------------------------
@dataclass(order=True)
class Ordered:
    a: int
    b: int = 0


print(Ordered(1) < Ordered(2), Ordered(1, 1) > Ordered(1, 0))
print(sorted([Ordered(2), Ordered(1), Ordered(1, 5)]))
try:
    Ordered(1) < 5
except TypeError as exc:
    print("TypeError:", exc)
try:
    Ordered(1) < Point(1)
except TypeError as exc:
    print("TypeError:", exc)
try:
    @dataclass(order=True, eq=False)
    class Bad:
        x: int
except ValueError as exc:
    print("ValueError:", exc)


# ---- the overwrite policy is ASYMMETRIC ----------------------------------
@dataclass
class KeepsBody:
    x: int

    def __repr__(self):
        return "BODY-REPR"

    def __eq__(self, other):
        return "BODY-EQ"


print(repr(KeepsBody(1)), KeepsBody(1) == KeepsBody(1))

try:
    @dataclass(order=True)
    class ClashesOrder:
        x: int

        def __lt__(self, other):
            return True
except TypeError as exc:
    print("TypeError:", exc)

try:
    @dataclass(unsafe_hash=True)
    class ClashesHash:
        x: int

        def __hash__(self):
            return 1
except TypeError as exc:
    print("TypeError:", exc)


# ---- the arity errors -----------------------------------------------------
@dataclass
class Two:
    a: int
    b: int = 2


# WRITTEN OUT RATHER THAN THROUGH `eval`. A compiled program's `eval` runs in
# the embedded interpreter, which cannot see the program's own globals, so
# `eval("Two()")` is a NameError about a class three lines above.
try:
    Two()
except TypeError as exc:
    print("Two() ->", exc)
try:
    Two(1, 2, 3)
except TypeError as exc:
    print("Two(1, 2, 3) ->", exc)
try:
    Two(1, z=5)
except TypeError as exc:
    print("Two(1, z=5) ->", exc)
try:
    Two(1, a=2)
except TypeError as exc:
    print("Two(1, a=2) ->", exc)

try:
    @dataclass
    class Wrong:
        a: int = 1
        b: int
except TypeError as exc:
    print("TypeError:", exc)


# ---- kw_only --------------------------------------------------------------
@dataclass
class KwSome:
    a: int
    b: int = field(kw_only=True, default=0)
    c: int = 3


print(KwSome(1, c=9, b=8), KwSome.__match_args__)
try:
    KwSome(1, 2)
except TypeError as exc:
    print("TypeError:", exc)


@dataclass(kw_only=True)
class AllKw:
    a: int
    b: int = 2


print(AllKw(a=1), AllKw.__match_args__)
try:
    AllKw(1)
except TypeError as exc:
    print("TypeError:", exc)


@dataclass
class Marked:
    a: int
    _: dataclasses.KW_ONLY
    b: int
    c: int = 4


print(Marked(1, b=2), Marked.__match_args__)
print([f.name for f in fields(Marked)])


# ---- ClassVar, InitVar, __post_init__ ------------------------------------
# A REAL `typing.ClassVar` AND NOT THE STRING "ClassVar[str]". CPython
# resolves a string annotation in the defining module's namespace, so without
# `typing` imported there the string form is an ordinary field -- which is
# CPython's answer and not one worth reproducing a divergence over.
@dataclass
class WithClassVar:
    kind: typing.ClassVar[str] = "shared"
    x: int = 1


print([f.name for f in fields(WithClassVar)], WithClassVar(2), WithClassVar.kind)


@dataclass
class WithInit:
    x: int
    doubled: InitVar[int] = 3
    total: int = field(init=False, default=0)

    def __post_init__(self, doubled):
        self.total = self.x + doubled


w = WithInit(1, 10)
print(w, w.total, [f.name for f in fields(WithInit)])
print(WithInit(1).total, WithInit.__match_args__)
print("doubled" in vars(w))


# ---- inheritance ---------------------------------------------------------
@dataclass
class Base:
    a: int
    b: int = 2
    c: int = 3


@dataclass
class Sub(Base):
    b: int = 99
    d: int = 4


# THE OVERRIDDEN FIELD KEEPS THE BASE'S POSITION AND TAKES THE SUB'S VALUE.
print([f.name for f in fields(Sub)])
print(Sub(1), Sub(1, 2, 3, 4))


@dataclass
class Left:
    a: int = 1


@dataclass
class Right:
    b: int = 2


@dataclass
class Both(Left, Right):
    c: int = 3


print([f.name for f in fields(Both)], Both())


class Plain:
    ignored: int = 7


@dataclass
class OverPlain(Plain):
    x: int = 1


print([f.name for f in fields(OverPlain)], OverPlain())

# An undecorated subclass of a dataclass still answers is_dataclass.
class Undecorated(Base):
    pass


print(dataclasses.is_dataclass(Undecorated),
      [f.name for f in fields(Undecorated)])

try:
    @dataclass
    class NonFrozenOverFrozen(Frozen):
        y: int = 1
except TypeError as exc:
    print("TypeError:", exc)

try:
    @dataclass(frozen=True)
    class FrozenOverPlain(Base):
        y: int = 1
except TypeError as exc:
    print("TypeError:", exc)


# A FROZEN SUBCLASS'S NON-FIELD ATTRIBUTE still goes through.
@dataclass(frozen=True)
class FrozenSub(Frozen):
    y: int = 1


class LooseChild(FrozenSub):
    pass


lc = LooseChild(1)
lc.whatever = 5
print(lc.whatever)
try:
    lc.x = 9
except dataclasses.FrozenInstanceError as exc:
    print("FrozenInstanceError:", exc)


# ---- asdict / astuple RECURSE -------------------------------------------
@dataclass
class Inner:
    v: int


@dataclass
class Outer:
    one: Inner
    many: list
    named: dict


o = Outer(Inner(1), [Inner(2), Inner(3)], {"k": Inner(4)})
print(dataclasses.asdict(o))
print(dataclasses.astuple(o))
print(dataclasses.asdict(o)["many"][0]["v"])
# The result must not SHARE the nested objects.
got = dataclasses.asdict(o)
got["many"][0]["v"] = 99
print(o.many[0].v)
# A LEAF IS DEEP-COPIED, NOT SHARED. The recursion set is exactly dataclass /
# list / tuple / dict and everything else goes through deepcopy, so a plain
# object in a field comes back as a copy -- which is the whole point of calling
# asdict to snapshot something.
@dataclass
class Holder:
    raw: object


held = Holder({"nested": [1, 2]})
snap = dataclasses.asdict(held)
print("leaf shared:", snap["raw"] is held.raw, snap["raw"] == held.raw)
snap["raw"]["nested"].append(3)
print("original untouched:", held.raw)

try:
    dataclasses.asdict(Point)
except TypeError as exc:
    print("TypeError:", exc)


# ---- replace ------------------------------------------------------------
print(dataclasses.replace(Point(1, 2), y=9))
print(dataclasses.replace(Sub(1), d=8))
try:
    dataclasses.replace(Bag(), computed=1)
except TypeError as exc:
    print("TypeError:", exc)
# AN UNKNOWN KEYWORD IS THE CONSTRUCTOR'S ERROR, so it names the class.
try:
    dataclasses.replace(Point(1), zzz=5)
except TypeError as exc:
    print("TypeError:", exc)
# A SET IS NOT RECURSED INTO -- its members are deep-copied instances, because
# converting them to dicts would make them unhashable and crash.
@dataclass(frozen=True)
class Key:
    k: int


@dataclass
class HasSet:
    s: object


snap_set = dataclasses.asdict(HasSet({Key(3)}))
print(snap_set, type(snap_set["s"]).__name__)
# A CLASS IS NOT AN INSTANCE: asdict/astuple/replace take instances only.
for fn in (dataclasses.asdict, dataclasses.astuple):
    try:
        fn(Point)
        print("ACCEPTED a class")
    except TypeError as exc:
        print("TypeError:", exc)
print(dataclasses.replace(WithInit(1, 10), x=5, doubled=1).total)


@dataclass
class NeedsInitVar:
    x: int
    v: InitVar[int]

    def __post_init__(self, v):
        self.x = self.x + v


try:
    dataclasses.replace(NeedsInitVar(1, 2))
except TypeError as exc:
    print("TypeError:", exc)


# ---- fields() on a non-dataclass ---------------------------------------
for bad in (42, Plain, Plain()):
    try:
        fields(bad)
        print("ACCEPTED", bad)
    except TypeError as exc:
        print("TypeError:", exc)


# ---- make_dataclass ----------------------------------------------------
# It needed a COMPILER fix, not a workaround: the runtime's `__annotations__`
# read consulted only the PEP 649 thunk and ignored a stored entry, so the
# namespace this function writes was invisible and every made class had no
# fields.
Made = dataclasses.make_dataclass(
    "Made", ["a", ("b", int), ("c", int, field(default=3))])
print(Made(1, 2), [f.name for f in fields(Made)])
print(dataclasses.is_dataclass(Made), Made.__name__, Made.__match_args__)
print(Made(1, 2) == Made(1, 2), Made(1, 2) == Made(1, 9))

MadeFrozen = dataclasses.make_dataclass("MadeFrozen", [("x", int)],
                                        frozen=True, order=True)
print(hash(MadeFrozen(1)) == hash(MadeFrozen(1)),
      MadeFrozen(1) < MadeFrozen(2))
try:
    MadeFrozen(1).x = 5
except dataclasses.FrozenInstanceError as exc:
    print("FrozenInstanceError:", exc)

# `y` NEEDS A DEFAULT: `Base` ends in defaulted fields, and a non-default one
# after them is the same TypeError whether the class is written out or made.
MadeSub = dataclasses.make_dataclass(
    "MadeSub", [("y", int, field(default=7))], bases=(Base,))
print([f.name for f in fields(MadeSub)], MadeSub(1))

for bad in ("class", "not-an-identifier"):
    try:
        dataclasses.make_dataclass("Bad", [(bad, int)])
        print("ACCEPTED", bad)
    except TypeError as exc:
        print("TypeError:", exc)
try:
    dataclasses.make_dataclass("Dup", [("a", int), ("a", int)])
except TypeError as exc:
    print("TypeError:", exc)


# ---- weakref_slot without slots is an error in BOTH -------------------
# `slots=True` ITSELF IS NOT TESTED HERE: CPython implements it and this module
# refuses it by name, so asserting the refusal would be asserting a difference
# and could only ever fail. It is measured against uasm alone, in
# test_stdlib.py::test_dataclasses_refuses_slots.
try:
    @dataclass(weakref_slot=True)
    class NeedsSlots:
        x: int
except TypeError as exc:
    print("TypeError:", exc)

# ---- the generated methods' names -------------------------------------
# `P.__init__.__qualname__` is `P.__init__`: the class and the method, not
# the closure the method was made in. `__repr__` keeps its body as
# `__wrapped__`, whose qualname names `__create_fn__` -- the shape a program
# (pprint is one) reads to tell a generated repr from a written one -- and
# `__replace__` is CPython's one module-level `_replace`.
@dataclass(order=True, frozen=True)
class Named:
    x: int


for name in ("__init__", "__repr__", "__eq__", "__lt__", "__le__", "__gt__",
             "__ge__", "__hash__", "__setattr__", "__delattr__",
             "__replace__"):
    method = Named.__dict__[name]
    wrapped = getattr(method, "__wrapped__", None)
    print(name, method.__name__, method.__qualname__,
          wrapped.__qualname__ if wrapped is not None else None)


class Holder:
    @dataclass
    class Inside:
        a: int


print(Holder.Inside.__init__.__qualname__, Holder.Inside(3))

print("done")
