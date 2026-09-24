"""Classes whose boilerplate is generated from their annotations.

COVERAGE: `dataclass` bare and called, with `init`, `repr`, `eq`, `order`,
`unsafe_hash`, `frozen`, `match_args` and `kw_only`; `field` with `default`,
`default_factory`, `init`, `repr`, `hash`, `compare`, `metadata` and
`kw_only`; `Field`, `fields`, `is_dataclass`, `asdict`, `astuple`, `replace`,
`make_dataclass`, `MISSING`, `KW_ONLY`, `InitVar`, `FrozenInstanceError`;
`__dataclass_fields__`, `__dataclass_params__`, `__match_args__`,
`__post_init__`, `__replace__`; `ClassVar` exclusion; field inheritance in
reverse-MRO order; the whole `(eq, frozen, unsafe_hash, explicit hash)` table;
and the four `__init__` arity errors.

NOT COVERED, and each REFUSED BY NAME rather than accepted and ignored:
`slots=True` and `weakref_slot=True`. `slots` replaces the class object and
rewrites the `__class__` cell behind every zero-argument `super()` in it, and a
half-built version of that is worse than none -- a program gets a class whose
`super()` calls fail at run time and nowhere near the decorator.

ALSO NOT COVERED: `field(doc=...)`, `Field[int]`, and
`abc.update_abstractmethods` (so a dataclass over an ABC keeps abstract methods
it has implemented). These are declined quietly because nothing observable
breaks; the two above are refused loudly because something does.

WHY IT WAS REBUILT RATHER THAN RESTORED. The archived version accepted
`frozen=True` and then never read it, so a frozen dataclass was fully mutable;
it had no `__hash__` handling, so the default dataclass compared equal and
stayed hashable by identity and silently corrupted every set and dict it went
into; it read only the class's own `__annotations__`, so inheritance dropped
every base field without a word; and its `asdict` did not recurse. Each of
those is a WRONG ANSWER rather than a missing feature, which is what this
rebuild exists to find. See `docs/STDLIB.md`.

## Two places this cannot match CPython, and they are the compiler's

`__repr__` uses `__qualname__`, and uasm's `__qualname__` for a NESTED
class is the bare name -- so `Outer.Inner(x=1)` reads `Inner(x=1)` here. The
code is right and the qualified name is not yet.

The mutable-default check is CPython's `type(default).__hash__ is None`, and
that test does not work here: `type([]).__hash__` is not None under uasm
and `type({}).__hash__` raises. So the check is a list of the builtin mutable
types plus the `__hash__ is None` test for anything else, which agrees with
CPython for every default a program actually writes and differs for exotica
like `{}.keys()`. Written down rather than discovered.

`field(3)` IS ACCEPTED HERE AND REFUSED BY CPYTHON. `field` is declared
keyword-only and this compiler does not enforce that at run time -- it warns
where it can see the call and lets it through -- so a positional argument
becomes the `default`. The declaration is right and the enforcement is the
compiler's; recorded because it makes this module a permissive superset in one
spot rather than a divergence in what a correct program computes.

## No `exec`, so `__init__` has no real signature

CPython builds `__init__` by compiling source. There is no `exec` here, so the
generated one is a closure over the field list taking `*args, **kwargs`. That
is invisible until something goes wrong, and then it is very visible: the
interpreter's own arity errors do not exist, so `_arity` reproduces them by
hand. A dataclass that accepted `C(1, 2, 3)` on a two-field class, or ignored
`C(1, z=5)`, would be a silently permissive superset of Python -- which is the
one thing a differential suite cannot catch, because no correct program does
it.
"""
import copy as _copy
import keyword


class _MissingType:
    """The absence of a default, as an object -- because None is a value a
    field may legitimately default to.

    NO `__repr__`. CPython's `_MISSING_TYPE` defines none, so `repr(MISSING)`
    is an object address there and here; giving it a tidy `MISSING` would be a
    difference, and every internal test is `is MISSING` anyway.
    """


MISSING = _MissingType()


class _KwOnlyType:
    """The `KW_ONLY` marker's type. See `KW_ONLY`."""


#: A pseudo-field whose annotation is this puts every field AFTER it in the
#: class body into keyword-only. Scoped to the body it appears in.
KW_ONLY = _KwOnlyType()


class _FactoryType:
    """What `inspect.signature` shows for a `default_factory` parameter.

    A SENTINEL AND NOT AN "IS THE ARGUMENT MISSING" TEST: passing `None` or `0`
    explicitly must not call the factory, and only this object means nobody
    passed anything.
    """

    def __repr__(self):
        return "<factory>"


_HAS_DEFAULT_FACTORY = _FactoryType()


class _FieldKind:
    """Which of the three things an entry in `__dataclass_fields__` is.

    `__repr__` answers the bare name because `Field.__repr__` formats this one
    value with `str()` where it uses `repr()` for every other.
    """

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return self._name


_FIELD = _FieldKind("_FIELD")
_FIELD_CLASSVAR = _FieldKind("_FIELD_CLASSVAR")
_FIELD_INITVAR = _FieldKind("_FIELD_INITVAR")

_PARAMS = "__dataclass_params__"
_FIELDS = "__dataclass_fields__"
_POST_INIT = "__post_init__"


class FrozenInstanceError(AttributeError):
    """Assigning to a field of a frozen instance. An AttributeError, because
    that is what a refused attribute write is."""


class InitVar:
    """A constructor argument that is NOT a field.

    It reaches `__init__` and `__post_init__` and is never stored, never in
    `fields()`, never in the repr and never compared. `InitVar[int]` and the
    bare `InitVar` are both valid, which is why the detection tests identity
    AND type.
    """

    def __init__(self, type=None):
        self.type = type

    def __repr__(self):
        name = getattr(self.type, "__name__", None)
        return "dataclasses.InitVar[%s]" % (name if name is not None
                                            else repr(self.type),)

    def __class_getitem__(cls, item):
        return InitVar(item)


class _MappingProxy:
    """A read-only view of a dict, for `Field.metadata`.

    A LIVE VIEW AND NOT A COPY, which is CPython's behaviour and the
    counter-intuitive half: mutating the dict a program passed to `field()`
    shows through `f.metadata` afterwards. Copying it defensively is the
    obvious thing to write and diverges.

    Hand-written because there is no `types.MappingProxyType` here and no
    subclassing of a builtin. It answers `mappingproxy({...})` to `repr`, as
    CPython's does, and compares equal to a plain dict in both directions so
    that `f.metadata == {"unit": "cm"}` holds.
    """

    def __init__(self, held):
        self._held = held

    def __getitem__(self, key):
        return self._held[key]

    def __setitem__(self, key, value):
        raise TypeError("'mappingproxy' object does not support item "
                        "assignment")

    def __delitem__(self, key):
        raise TypeError("'mappingproxy' object does not support item deletion")

    def __contains__(self, key):
        return key in self._held

    def __iter__(self):
        return iter(self._held)

    def __len__(self):
        return len(self._held)

    def __eq__(self, other):
        if isinstance(other, _MappingProxy):
            return self._held == other._held
        return self._held == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "mappingproxy(%r)" % (self._held,)

    def __str__(self):
        # THE DICT'S, NOT THE PROXY'S. CPython's mappingproxy answers
        # `mappingproxy({...})` to `repr` and `{...}` to `str`, so `print(f.metadata)`
        # shows a plain mapping. Letting `str` fall back to `__repr__` -- which is
        # what happens if this is not written -- prints the wrapper.
        return str(self._held)

    def get(self, key, default=None):
        return self._held.get(key, default)

    def keys(self):
        return self._held.keys()

    def values(self):
        return self._held.values()

    def items(self):
        return self._held.items()

    def copy(self):
        """A PLAIN DICT, as CPython's proxy answers -- a copy of a read-only
        view that was still read-only would be useless."""
        return dict(self._held)


#: ONE SHARED EMPTY PROXY. `field().metadata is field().metadata` is True in
#: CPython, so a fresh one per field would be equal and not identical.
_EMPTY_METADATA = _MappingProxy({})


class Field:
    """One field's description. `fields()` hands these out.

    `Field()` TAKES EVERY ARGUMENT AND HAS NO DEFAULTS, as CPython's does: it
    is not a public constructor, `field()` is, and giving it defaults would
    make `Field()` succeed where CPython raises.
    """

    def __init__(self, default, default_factory, init, repr, hash, compare,
                 metadata, kw_only):
        self.name = None
        self.type = None
        self.default = default
        self.default_factory = default_factory
        self.init = init
        self.repr = repr
        self.hash = hash
        self.compare = compare
        self.metadata = (_EMPTY_METADATA if metadata is None
                         else _MappingProxy(metadata))
        self.kw_only = kw_only
        self._field_type = None

    def __repr__(self):
        # NO SPACE AFTER THE COMMAS, which is CPython's format and not the one
        # anybody writes by hand. `_field_type` is the single value formatted
        # with `str()` rather than `repr()`.
        return ("Field(name=%r,type=%r,default=%r,default_factory=%r,"
                "init=%r,repr=%r,hash=%r,compare=%r,metadata=%r,"
                "kw_only=%r,_field_type=%s)"
                % (self.name, self.type, self.default, self.default_factory,
                   self.init, self.repr, self.hash, self.compare,
                   self.metadata, self.kw_only, self._field_type))

    def __set_name__(self, owner, name):
        """PEP 487, FORWARDED TO THE DEFAULT.

        A descriptor supplied as `field(default=D())` never sits in the class
        body -- the Field does -- so without this the descriptor is never told
        its name.
        """
        hook = getattr(type(self.default), "__set_name__", None)
        if hook is not None:
            hook(self.default, owner, name)


def field(*, default=MISSING, default_factory=MISSING, init=True, repr=True,
          hash=None, compare=True, metadata=None, kw_only=MISSING):
    """Describe one field. KEYWORD-ONLY, as CPython's is.

    `hash` IS THREE-STATE and not a boolean: None means "follow `compare`",
    which is what lets `field(hash=False)` drop a compared field from the hash
    and `field(hash=True, compare=False)` add an uncompared one. Reading it as
    falsey drops every field from `__hash__` and every frozen instance then
    hashes alike -- a wrong answer no repr or equality test notices.

    `kw_only` DEFAULTS TO MISSING and not to False, so that "unspecified" can
    be told from "explicitly positional" when the class says `kw_only=True`.
    """
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("cannot specify both default and default_factory")
    return Field(default, default_factory, init, repr, hash, compare,
                 metadata, kw_only)


class _Params:
    """What `__dataclass_params__` holds. The frozen flag is read off BASES by
    the inheritance check, which is the reason this exists at all."""

    def __init__(self, init, repr, eq, order, unsafe_hash, frozen,
                 match_args, kw_only, slots, weakref_slot):
        self.init = init
        self.repr = repr
        self.eq = eq
        self.order = order
        self.unsafe_hash = unsafe_hash
        self.frozen = frozen
        self.match_args = match_args
        self.kw_only = kw_only
        self.slots = slots
        self.weakref_slot = weakref_slot

    def __repr__(self):
        return ("_DataclassParams(init=%r,repr=%r,eq=%r,order=%r,"
                "unsafe_hash=%r,frozen=%r)"
                % (self.init, self.repr, self.eq, self.order,
                   self.unsafe_hash, self.frozen))


# ── the mutable-default rule ────────────────────────────────────────────────

def _is_mutable_default(value):
    """Would CPython refuse this as a plain default?

    ITS OWN TEST -- `type(default).__hash__ is None` -- DOES NOT WORK HERE:
    `type([]).__hash__` is not None under uasm and `type({}).__hash__`
    raises outright. So the builtin mutable types are named, and the
    `__hash__ is None` test (which does work) catches everything else.

    WRITTEN AS SEPARATE `isinstance` CALLS AND NOT A TUPLE, because `bytearray`
    is not a name this compiler lets a program use as a VALUE -- a tuple of
    types containing it is `E0056: 'bytearray' is a builtin that cannot be used
    as a value`, while `isinstance(x, bytearray)` is fine.
    """
    if isinstance(value, list) or isinstance(value, dict):
        return True
    if isinstance(value, set) or isinstance(value, bytearray):
        return True
    # A USER CLASS SAYING `__hash__ = None` is refused too, and that half of
    # the test does work.
    try:
        return type(value).__hash__ is None
    except AttributeError:
        return False


# ── recognising the annotations ─────────────────────────────────────────────

def _is_classvar(annotation):
    """Is this annotation a `typing.ClassVar`?

    TEXTUALLY, because `typing` is not a rebuilt module and an annotation may
    be a string anyway -- under `from __future__ import annotations` every one
    is. CPython also falls back to a textual test for the string case, and the
    shapes accepted here are the ones it accepts: `ClassVar`, `ClassVar[...]`
    and either spelled through a module alias.
    """
    text = annotation if isinstance(annotation, str) else _annotation_text(
        annotation)
    if not text:
        return False
    head = text.split("[")[0].strip()
    return head == "ClassVar" or head.endswith(".ClassVar")


def _annotation_text(annotation):
    name = getattr(annotation, "_name", None)
    if isinstance(name, str) and name:
        return name
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        got = getattr(origin, "_name", None)
        if isinstance(got, str) and got:
            return got
    try:
        return repr(annotation)
    except Exception:
        return ""


def _is_initvar(annotation):
    """`InitVar` bare and `InitVar[int]` are both one."""
    if annotation is InitVar or type(annotation) is InitVar:
        return True
    if isinstance(annotation, str):
        head = annotation.split("[")[0].strip()
        return head == "InitVar" or head.endswith(".InitVar")
    return False


def _is_kw_only_marker(annotation):
    if annotation is KW_ONLY:
        return True
    if isinstance(annotation, str):
        head = annotation.strip()
        return head == "KW_ONLY" or head.endswith(".KW_ONLY")
    return False


# ── collecting the fields ───────────────────────────────────────────────────

def _own_annotations(cls):
    """The annotations THIS class wrote, and not its bases'.

    PEP 649 MADE THIS THE PLAIN ATTRIBUTE READ. Under 3.14 annotations are
    computed from `__annotate__` on demand and `__annotations__` is NEVER in
    `cls.__dict__` -- so the `__dict__` lookup that used to be the careful way
    to avoid reading a base's annotations now answers `{}` for every class, and
    a dataclass came out with no fields at all.

    The plain read is correct and was verified on both runtimes: a class that
    annotates nothing answers `{}` rather than its base's annotations, which is
    the only thing the `__dict__` lookup was protecting against.
    """
    return getattr(cls, "__annotations__", {}) or {}


def _collect(cls, kw_only_default):
    """Every field of `cls`, base fields first, merged by NAME.

    A DICT UPDATE AND NOT AN APPEND. When a subclass redeclares a base field
    the field keeps the BASE's position and takes the SUBCLASS's value, which
    is what `fields[name] = f` gives and what `base + own` does not:
    `Base(a, b=2, c=3)` with `Sub(b=99)` is `(a, b=99, c=3)`.

    THE MRO IS WALKED IN REVERSE -- most-base first, `cls` excluded. Forwards,
    or over `__bases__` left to right, silently reverses the order under
    multiple inheritance, and single inheritance hides it completely.
    """
    found = {}
    for base in cls.__mro__[-1:0:-1]:
        inherited = getattr(base, _FIELDS, None)
        if inherited:
            for name in inherited:
                found[name] = inherited[name]

    kw_only = kw_only_default
    for name in _own_annotations(cls):
        annotation = _own_annotations(cls)[name]
        if _is_kw_only_marker(annotation):
            # THE MARKER IS NOT A FIELD. Everything after it in THIS body is
            # keyword-only, and the marker itself leaves no attribute and no
            # entry behind.
            kw_only = True
            continue
        found[name] = _make_field(cls, name, annotation, kw_only)
    return found


def _make_field(cls, name, annotation, kw_only):
    got = getattr(cls, name, MISSING)
    if isinstance(got, Field):
        # A FRESH Field RATHER THAN THE ONE THE BODY HELD. Mutating that one
        # rewrites a `field()` result a program may have bound to a name and
        # reused, and once inheritance merges base Field objects into a
        # subclass's dict it would rewrite the BASE's field too.
        spec = Field(got.default, got.default_factory, got.init, got.repr,
                     got.hash, got.compare, None, got.kw_only)
        spec.metadata = got.metadata
    else:
        spec = Field(got, MISSING, True, True, None, True, None, MISSING)
    spec.name = name
    spec.type = annotation
    if _is_classvar(annotation):
        spec._field_type = _FIELD_CLASSVAR
    elif _is_initvar(annotation):
        spec._field_type = _FIELD_INITVAR
    else:
        spec._field_type = _FIELD
    if spec.kw_only is MISSING:
        spec.kw_only = kw_only
    if spec._field_type is _FIELD_CLASSVAR or spec._field_type is _FIELD_INITVAR:
        if spec.default_factory is not MISSING:
            raise TypeError("field %s cannot have a default factory" % (name,))
    elif spec.default is not MISSING and _is_mutable_default(spec.default):
        raise ValueError("mutable default %r for field %s is not allowed: "
                         "use default_factory" % (type(spec.default), name))
    return spec


# ── the generated methods ───────────────────────────────────────────────────

def _init_params(specs):
    """The fields `__init__` takes, partitioned as its signature is.

    A STABLE PARTITION: the non-keyword-only ones in declaration order, then
    the keyword-only ones in declaration order. Only the SIGNATURE moves --
    `fields()`, the repr, equality and the ordering all keep declaration
    order, so sorting the stored list corrupts every one of them.
    """
    std, kw = [], []
    for spec in specs:
        if spec._field_type is _FIELD_CLASSVAR or not spec.init:
            continue
        (kw if spec.kw_only else std).append(spec)
    return std, kw


def _has_default(spec):
    return spec.default is not MISSING or spec.default_factory is not MISSING


def _names(items):
    """`'a'`, `'a' and 'b'`, `'a', 'b', and 'c'` -- the interpreter's own
    joining, Oxford comma and all."""
    quoted = ["'%s'" % (one,) for one in items]
    if len(quoted) == 1:
        return quoted[0]
    if len(quoted) == 2:
        return "%s and %s" % (quoted[0], quoted[1])
    return "%s, and %s" % (", ".join(quoted[:-1]), quoted[-1])


def _make_init(cls, specs, frozen, has_post_init):
    std, kw = _init_params(specs)
    lowest = 1 + len([s for s in std if not _has_default(s)])
    highest = 1 + len(std)
    by_name = {}
    for spec in std:
        by_name[spec.name] = spec
    for spec in kw:
        by_name[spec.name] = spec
    initvars = [s for s in specs if s._field_type is _FIELD_INITVAR]

    def __init__(self, *args, **kwargs):
        who = "%s.__init__()" % (cls.__qualname__,)
        if len(args) > len(std):
            got = len(args) + 1
            # SINGULAR AND PLURAL BOTH WAYS, as the interpreter writes them:
            # `takes 1 positional argument but 2 were given`, and `but 1 was
            # given` when only one arrived.
            were = "was" if got == 1 else "were"
            if lowest == highest:
                raise TypeError("%s takes %d positional argument%s but %d %s "
                                "given"
                                % (who, highest, "" if highest == 1 else "s",
                                   got, were))
            raise TypeError("%s takes from %d to %d positional arguments but "
                            "%d %s given"
                            % (who, lowest, highest, got, were))
        given = {}
        for i in range(len(args)):
            given[std[i].name] = args[i]
        for name in kwargs:
            if name not in by_name:
                raise TypeError("%s got an unexpected keyword argument %r"
                                % (who, name))
            if name in given:
                raise TypeError("%s got multiple values for argument %r"
                                % (who, name))
            given[name] = kwargs[name]

        missing_pos = [s.name for s in std
                       if s.name not in given and not _has_default(s)]
        if missing_pos:
            raise TypeError("%s missing %d required positional argument%s: %s"
                            % (who, len(missing_pos),
                               "" if len(missing_pos) == 1 else "s",
                               _names(missing_pos)))
        missing_kw = [s.name for s in kw
                      if s.name not in given and not _has_default(s)]
        if missing_kw:
            raise TypeError("%s missing %d required keyword-only argument%s: "
                            "%s"
                            % (who, len(missing_kw),
                               "" if len(missing_kw) == 1 else "s",
                               _names(missing_kw)))

        # ASSIGNED IN DECLARATION ORDER, which is not the parameter order when
        # keyword-only fields are interleaved.
        for spec in specs:
            if spec._field_type is _FIELD_CLASSVAR:
                continue
            if spec.name in given:
                value = given[spec.name]
            elif spec.default_factory is not MISSING:
                # ONE CALL PER INSTANCE, which is the whole reason the factory
                # is a callable rather than a value.
                value = spec.default_factory()
            elif spec.default is not MISSING:
                if not spec.init:
                    # `init=False` WITH A PLAIN DEFAULT ASSIGNS NOTHING: the
                    # value is read through the class attribute, so `vars(c)`
                    # stays empty and changing `C.x` afterwards changes every
                    # instance. Assigning it here is the tidy-looking thing
                    # and breaks that aliasing.
                    continue
                value = spec.default
            elif not spec.init:
                # No default and not in `__init__`: the attribute is genuinely
                # absent, and reading it is an AttributeError as it should be.
                continue
            else:
                continue
            if spec._field_type is _FIELD_INITVAR:
                continue
            if frozen:
                object.__setattr__(self, spec.name, value)
            else:
                setattr(self, spec.name, value)

        if has_post_init:
            # LOOKED UP FRESH, so that replacing `__post_init__` after the
            # class was built is honoured; the BOOLEAN was snapshotted, which
            # is what CPython does.
            supplied = []
            for spec in initvars:
                if spec.name in given:
                    supplied.append(given[spec.name])
                elif spec.default_factory is not MISSING:
                    supplied.append(spec.default_factory())
                else:
                    supplied.append(spec.default)
            getattr(self, _POST_INIT)(*supplied)

    return __init__


def _make_repr(cls, specs):
    shown = [s for s in specs if s._field_type is _FIELD and s.repr]
    #: THE RECURSION GUARD. A self-referential instance must print `...` for
    #: the whole re-entered object rather than blow the stack, and the key is
    #: the object's identity.
    running = set()

    def body(self):
        parts = []
        for spec in shown:
            parts.append("%s=%r" % (spec.name, getattr(self, spec.name)))
        # `__qualname__`, NOT `__name__`, and read off the instance's own
        # class at call time so a plain subclass prints its own name.
        return "%s(%s)" % (self.__class__.__qualname__, ", ".join(parts))

    # TWO FUNCTIONS, as CPython has: the body its `__create_fn__` makes, and
    # the recursion guard around it, which keeps the body as `__wrapped__`.
    # A program -- `pprint` is one -- tells a GENERATED repr from a written
    # one by exactly that shape, so it is reproduced rather than folded away.
    body.__name__ = "__repr__"
    body.__qualname__ = "__create_fn__.<locals>.__repr__"

    def __repr__(self):
        key = id(self)
        if key in running:
            return "..."
        running.add(key)
        try:
            return body(self)
        finally:
            running.discard(key)

    __repr__.__wrapped__ = body
    return __repr__


def _class_of(value):
    """`value.__class__`, falling back to `type(value)`.

    CPython's generated `__eq__` tests `other.__class__ is self.__class__` and
    not `type(other) is type(self)` -- the two differ for anything overriding
    `__class__`, so the spelling is deliberate. Under uasm an `int` has no
    `__class__` at all, so comparing a dataclass against one raised
    `AttributeError: 'int' object has no attribute '__class__'` from inside
    `__eq__`, where CPython answers NotImplemented and the operator answers
    False. The fallback keeps CPython's spelling wherever it is available.
    """
    got = getattr(value, "__class__", None)
    return type(value) if got is None else got


def _compare_names(specs):
    return [s.name for s in specs if s._field_type is _FIELD and s.compare]


def _make_eq(cls, specs):
    names = _compare_names(specs)

    def __eq__(self, other):
        # AN IDENTITY FAST PATH, which the ordering methods do NOT have.
        if self is other:
            return True
        # `other.__class__ is self.__class__`, not `type(other) is type(self)`
        # -- and `NotImplemented` rather than False, so the reflected operand
        # gets its turn.
        if _class_of(other) is not _class_of(self):
            return NotImplemented
        # A CHAINED `and` AND NOT A TUPLE COMPARE. Two consequences a tuple
        # gets wrong: a tuple short-circuits on element IDENTITY, so a field
        # holding nan compares equal to itself where CPython says False; and
        # `and` yields the OPERAND, so a field whose `__eq__` answers a
        # non-bool is passed through verbatim.
        got = True
        for name in names:
            got = getattr(self, name) == getattr(other, name)
            if not got:
                return got
        return got

    return __eq__


def _make_order(cls, specs, symbol):
    names = _compare_names(specs)

    def compare(self, other):
        if _class_of(other) is not _class_of(self):
            return NotImplemented
        mine = tuple([getattr(self, name) for name in names])
        theirs = tuple([getattr(other, name) for name in names])
        # A TUPLE COMPARE HERE, where `__eq__` is a chained `and` -- CPython
        # generates the two differently and the difference is observable.
        if symbol == "<":
            return mine < theirs
        if symbol == "<=":
            return mine <= theirs
        if symbol == ">":
            return mine > theirs
        return mine >= theirs

    return compare


def _hash_names(specs):
    """`f.compare if f.hash is None else f.hash` -- a three-state flag."""
    return [s.name for s in specs if s._field_type is _FIELD
            and (s.compare if s.hash is None else s.hash)]


def _make_hash(cls, specs):
    names = _hash_names(specs)

    def __hash__(self):
        return hash(tuple([getattr(self, name) for name in names]))

    return __hash__


def _make_frozen_setattr(cls, names):
    def __setattr__(self, name, value):
        # NOT A BLANKET BLOCK. On an instance of the decorated class every
        # name is refused, field or not; on a SUBCLASS instance only real
        # field names are, and anything else goes through. An unconditional
        # raise breaks ordinary subclasses that add attributes of their own.
        if type(self) is cls or name in names:
            raise FrozenInstanceError("cannot assign to field %r" % (name,))
        object.__setattr__(self, name, value)

    return __setattr__


def _make_frozen_delattr(cls, names):
    def __delattr__(self, name):
        if type(self) is cls or name in names:
            raise FrozenInstanceError("cannot delete field %r" % (name,))
        object.__delattr__(self, name)

    return __delattr__


# ── the decorator ───────────────────────────────────────────────────────────

def _qualified(cls, value, name):
    """A generated method named as CPython names it: `P.__init__`.

    Without this a program reading `P.__init__.__qualname__` saw the closure
    it was made in -- under the SPLICED name of this module's own function,
    `_asmpy_bundled_11_dataclasses__make_init.<locals>.__init__` -- where
    CPython answers the class and the method. A tuple (`__match_args__`)
    and None (`__hash__`) are not functions and are left alone.

    THE NAME TOO, not only the qualname: the four ordering methods are made
    by one factory, so each was called `compare` where CPython's are
    `__lt__`, `__le__`, `__gt__` and `__ge__`."""
    if callable(value) and not isinstance(value, type)             and hasattr(value, "__name__"):
        value.__name__ = name
        value.__qualname__ = cls.__qualname__ + "." + name
    return value


def _set_new(cls, name, value):
    """Install a generated method only if the body did not write one.

    `cls.__dict__` and not `getattr`: inheriting the method from a dataclass
    base is not the same as writing it here, and only the second wins.
    """
    if name in cls.__dict__:
        return False
    if name != "__replace__":
        _qualified(cls, value, name)
    setattr(cls, name, value)
    return True


def _set_or_raise(cls, name, value):
    """Install a generated method, or REFUSE if the body wrote one.

    The other half of a policy that is deliberately asymmetric: `__init__`,
    `__repr__`, `__eq__` and `__match_args__` yield to the body silently, and
    the ordering methods, `__hash__` and the frozen pair raise. One rule for
    all seven either swallows a real mistake or breaks code that hand-writes a
    `__repr__`.
    """
    if name in cls.__dict__:
        raise TypeError("Cannot overwrite attribute %s in class %s%s"
                        % (name, cls.__name__,
                           " Consider using functools.total_ordering"
                           if name in ("__lt__", "__le__", "__gt__", "__ge__")
                           else ""))
    setattr(cls, name, _qualified(cls, value, name))
    return True


def _process(cls, init, repr, eq, order, unsafe_hash, frozen, match_args,
             kw_only, slots, weakref_slot):
    if order and not eq:
        raise ValueError("eq must be true if order is true")
    if slots:
        raise TypeError(
            "slots=True is not supported by this implementation: it replaces "
            "the class object and rewrites the __class__ cell behind every "
            "zero-argument super() in it. See docs/STDLIB.md")
    if weakref_slot and not slots:
        raise TypeError("weakref_slot is True but slots is False")

    # THE FROZEN INHERITANCE RULES SCAN EVERY DATACLASS BASE, not the
    # immediate one, so a plain class sandwiched between does not launder
    # frozen-ness and a field-less dataclass base still counts. `all_frozen`
    # is deliberately three-state: None means no dataclass base at all, and
    # confusing it with False makes a frozen dataclass over an ordinary class
    # an error.
    any_frozen = False
    all_frozen = None
    for base in cls.__mro__[-1:0:-1]:
        params = getattr(base, _PARAMS, None)
        if params is None:
            continue
        all_frozen = params.frozen if all_frozen is None \
            else (all_frozen and params.frozen)
        if params.frozen:
            any_frozen = True
    if all_frozen is not None:
        if any_frozen and not frozen:
            raise TypeError("cannot inherit non-frozen dataclass from a "
                            "frozen one")
        if not all_frozen and frozen:
            raise TypeError("cannot inherit frozen dataclass from a "
                            "non-frozen one")

    found = _collect(cls, kw_only)
    setattr(cls, _FIELDS, found)
    setattr(cls, _PARAMS, _Params(init, repr, eq, order, unsafe_hash, frozen,
                                  match_args, kw_only, slots, weakref_slot))
    specs = [found[name] for name in found]

    # THE CLASS ATTRIBUTE THE BODY LEFT IS A `Field` OBJECT, and only then is
    # it rewritten. Unconditionally setting it re-installs an inherited
    # default onto a subclass; unconditionally deleting it attacks names the
    # decorator never placed.
    for name in _own_annotations(cls):
        spec = found.get(name)
        if spec is None or not isinstance(cls.__dict__.get(name), Field):
            continue
        if spec.default is MISSING:
            try:
                delattr(cls, name)
            except AttributeError:
                pass
        else:
            setattr(cls, name, spec.default)

    real = [s for s in specs if s._field_type is _FIELD]

    if init:
        # THE ORDERING CHECK RUNS OVER `__init__` PARAMETERS and not over
        # fields: InitVars count, ClassVars and `init=False` fields do not,
        # and keyword-only ones are partitioned out before the scan. It names
        # the MOST RECENT defaulted field, not the first.
        std, _kw = _init_params(specs)
        seen_default = None
        for spec in std:
            if _has_default(spec):
                seen_default = spec.name
            elif seen_default is not None:
                raise TypeError("non-default argument %r follows default "
                                "argument %r" % (spec.name, seen_default))
        _set_new(cls, "__init__",
                 _make_init(cls, specs, frozen, hasattr(cls, _POST_INIT)))

    if repr:
        _set_new(cls, "__repr__", _make_repr(cls, specs))

    if eq:
        _set_new(cls, "__eq__", _make_eq(cls, specs))
        # NO `__ne__` IS GENERATED. Adding one is observable through
        # `vars(cls)` and coerces a non-bool `__eq__` result to a bool.

    if order:
        for symbol, name in (("<", "__lt__"), ("<=", "__le__"),
                             (">", "__gt__"), (">=", "__ge__")):
            _set_or_raise(cls, name, _make_order(cls, specs, symbol))

    _apply_hash(cls, specs, eq, frozen, unsafe_hash)

    if match_args:
        # NOT THE FIELD LIST: the names of the POSITIONAL `__init__`
        # parameters, so InitVars are in and `init=False` and keyword-only
        # fields are out -- and it is generated even when `init=False`, from
        # the parameters `__init__` would have had.
        std, _kw = _init_params(specs)
        _set_new(cls, "__match_args__", tuple([s.name for s in std]))

    if frozen:
        names = frozenset([s.name for s in real])
        _set_or_raise(cls, "__setattr__", _make_frozen_setattr(cls, names))
        _set_or_raise(cls, "__delattr__", _make_frozen_delattr(cls, names))

    def __replace__(self, **changes):
        return replace(self, **changes)

    # NAMED `_replace`, which is what CPython's is: its `__replace__` is one
    # module-level function assigned to every dataclass, never qualified by
    # a class.
    __replace__.__name__ = "_replace"
    __replace__.__qualname__ = "_replace"

    # THROUGH THE SILENT-SKIP PATH, so a body-defined `__replace__` wins --
    # and installed unconditionally otherwise, including under `init=False`,
    # where it raises when called rather than being absent.
    _set_new(cls, "__replace__", __replace__)
    return cls


def _explicit_hash(cls):
    """Did the class BODY write a `__hash__` that counts?

    A body holding both `__eq__` and `__hash__ = None` does NOT count: that is
    Python's own automatic nulling rather than a decision, so `unsafe_hash`
    may still install one. Get this backwards and a frozen class with a custom
    `__eq__` stops being hashable.
    """
    if "__hash__" not in cls.__dict__:
        return False
    if cls.__dict__["__hash__"] is None and "__eq__" in cls.__dict__:
        return False
    return True


def _apply_hash(cls, specs, eq, frozen, unsafe_hash):
    """The whole `(unsafe_hash, eq, frozen, explicit)` table.

    THE DEFAULT DATACLASS IS UNHASHABLE, and that is the row everything else
    is written around: `eq=True, frozen=False` sets `__hash__ = None`. Python
    only nulls it automatically when `__eq__` is in the class body at creation
    time, and this assigns `__eq__` afterwards -- so without this line the
    class compares by value and still hashes by identity, and every set and
    dict it goes into silently holds duplicates.
    """
    explicit = _explicit_hash(cls)
    if unsafe_hash:
        if explicit:
            raise TypeError("Cannot overwrite attribute __hash__ in class %s"
                            % (cls.__name__,))
        cls.__hash__ = _qualified(cls, _make_hash(cls, specs), "__hash__")
        return
    if explicit:
        return
    if eq and frozen:
        cls.__hash__ = _qualified(cls, _make_hash(cls, specs), "__hash__")
    elif eq:
        cls.__hash__ = None


def dataclass(cls=None, /, *, init=True, repr=True, eq=True, order=False,
              unsafe_hash=False, frozen=False, match_args=True,
              kw_only=False, slots=False, weakref_slot=False):
    """Turn a class of annotations into one with the methods written out.

    `cls` IS POSITIONAL-ONLY AND EVERY FLAG IS KEYWORD-ONLY, which is not
    cosmetic: written the other way `dataclass(C, False)` silently means
    `init=False` where CPython raises, and `dataclass(cls=None)` returns a
    decorator where CPython raises.
    """
    def wrap(target):
        return _process(target, init, repr, eq, order, unsafe_hash, frozen,
                        match_args, kw_only, slots, weakref_slot)
    if cls is None:
        return wrap
    return wrap(cls)


# ── reading a dataclass back ────────────────────────────────────────────────

def fields(class_or_instance):
    """Every REAL field, as a tuple. Pseudo-fields are filtered out here.

    `__dataclass_fields__` holds the ClassVar and InitVar entries too, because
    inheritance needs them -- a subclass demoting a base field to a ClassVar
    has to overwrite the base entry. This is the filter; that is the store.
    """
    target = class_or_instance if isinstance(class_or_instance, type) \
        else type(class_or_instance)
    found = getattr(target, _FIELDS, None)
    if found is None:
        raise TypeError("must be called with a dataclass type or instance")
    return tuple([found[name] for name in found
                  if found[name]._field_type is _FIELD])


def is_dataclass(obj):
    """SEEN THROUGH INHERITANCE, so an undecorated subclass of a dataclass
    answers True and reports its base's fields. That is CPython's answer and
    not a bug to fix."""
    target = obj if isinstance(obj, type) else type(obj)
    return hasattr(target, _FIELDS)


def _is_dataclass_instance(obj):
    return not isinstance(obj, type) and is_dataclass(obj)


def asdict(obj, *, dict_factory=dict):
    """A dict of the fields, RECURSING into nested dataclasses and containers.

    The recursion is the whole point, and it is what the archived version
    lacked: a dataclass holding a list of dataclasses came back holding the
    objects themselves. Values are deep-copied through the structure rather
    than shared.
    """
    if not _is_dataclass_instance(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_inner(obj, dict_factory)


def _asdict_inner(obj, dict_factory):
    if _is_dataclass_instance(obj):
        out = []
        for spec in fields(obj):
            out.append((spec.name,
                        _asdict_inner(getattr(obj, spec.name), dict_factory)))
        return dict_factory(out)
    if isinstance(obj, tuple):
        return type(obj)([_asdict_inner(one, dict_factory) for one in obj]) \
            if _is_namedtuple(obj) \
            else tuple([_asdict_inner(one, dict_factory) for one in obj])
    if isinstance(obj, (list,)):
        return type(obj)([_asdict_inner(one, dict_factory) for one in obj])
    if isinstance(obj, dict):
        return type(obj)([(_asdict_inner(k, dict_factory),
                           _asdict_inner(v, dict_factory))
                          for k, v in obj.items()])
    # A LEAF IS DEEP-COPIED AND NOT SHARED. This is the half of `asdict` that
    # is easy to miss: the recursion set is exactly dataclass / list / tuple /
    # dict, and everything else goes through `deepcopy`. Passing leaves through
    # gives a dict whose values are the SAME objects the instance holds, so
    # mutating the result mutates the original -- which is precisely what
    # somebody calling `asdict` to serialise or snapshot is trying to avoid.
    return _copy.deepcopy(obj)


def _is_namedtuple(obj):
    """CPython's own test: a tuple that knows its field names.

    UNTESTED AGAINST THE ORACLE HERE, and said so rather than left to look
    covered. A namedtuple is a tuple SUBCLASS, `collections` is not rebuilt
    yet, and this compiler cannot subclass a builtin -- so no program it
    compiles can currently produce one. The branch is written correctly for
    when `collections` lands.
    """
    return isinstance(obj, tuple) and hasattr(obj, "_fields")


def astuple(obj, *, tuple_factory=tuple):
    if not _is_dataclass_instance(obj):
        raise TypeError("astuple() should be called on dataclass instances")
    return _astuple_inner(obj, tuple_factory)


def _astuple_inner(obj, tuple_factory):
    if _is_dataclass_instance(obj):
        return tuple_factory([_astuple_inner(getattr(obj, spec.name),
                                             tuple_factory)
                              for spec in fields(obj)])
    if _is_namedtuple(obj):
        # A NAMEDTUPLE SURVIVES `astuple` TOO, which is easy to miss because
        # the function's whole job looks like "make everything a tuple".
        return type(obj)(*[_astuple_inner(one, tuple_factory) for one in obj])
    if isinstance(obj, tuple):
        return tuple([_astuple_inner(one, tuple_factory) for one in obj])
    if isinstance(obj, list):
        return type(obj)([_astuple_inner(one, tuple_factory) for one in obj])
    if isinstance(obj, dict):
        return type(obj)([(_astuple_inner(k, tuple_factory),
                           _astuple_inner(v, tuple_factory))
                          for k, v in obj.items()])
    return obj


def replace(obj, /, **changes):
    """A new instance with some fields changed.

    `init=False` FIELDS MAY NOT BE CHANGED and are not re-supplied: they are
    not `__init__` parameters, so the new instance computes them again.
    An InitVar has to be given, because it was never stored -- and a required
    one that is missing is an error rather than a silently wrong object.
    """
    if not _is_dataclass_instance(obj):
        raise TypeError("replace() should be called on dataclass instances")
    found = getattr(type(obj), _FIELDS)
    # STARTS FROM WHAT THE CALLER PASSED, so a name that is not a field at all
    # stays in and reaches the constructor -- which is where CPython's
    # `P.__init__() got an unexpected keyword argument 'zzz'` comes from.
    # Building a fresh dict of known fields instead silently DROPPED the
    # unknown one and answered an unchanged object, so `replace(cfg,
    # tiemout=5)` was a typo nobody ever found out about.
    values = dict(changes)
    for name in found:
        spec = found[name]
        if spec._field_type is _FIELD_CLASSVAR:
            continue
        if not spec.init:
            if name in changes:
                raise TypeError("field %s is declared with init=False, it "
                                "cannot be specified with replace()" % (name,))
            continue
        if name in changes:
            continue
        if spec._field_type is _FIELD_INITVAR:
            if spec.default is MISSING and spec.default_factory is MISSING:
                raise TypeError("InitVar %r must be specified with "
                                "replace()" % (name,))
            continue
        values[name] = getattr(obj, name)
    # AN UNKNOWN KEYWORD IS LEFT TO THE CONSTRUCTOR, which names the class:
    # `P.__init__() got an unexpected keyword argument 'zzz'`. Checking it
    # here produced the same sentence without the class in front of it, and a
    # message that differs is a message that does not match.
    return type(obj)(**values)


def make_dataclass(cls_name, fields, *, bases=(), namespace=None, init=True,
                   repr=True, eq=True, order=False, unsafe_hash=False,
                   frozen=False, match_args=True, kw_only=False, slots=False,
                   weakref_slot=False):
    """Build a dataclass from a name and a list of fields.

    Each entry is a name, a `(name, type)` pair, or a `(name, type, field())`
    triple -- and a bare name types as `typing.Any` in CPython, which is
    spelled here as the string `"typing.Any"` because `typing` is not a
    rebuilt module.
    """
    # THIS NEEDED A COMPILER FIX RATHER THAN A WORKAROUND. `make_dataclass`
    # builds a class at RUN TIME and its fields come from annotations written
    # into the namespace -- and the runtime's `__annotations__` read consulted
    # only the PEP 649 thunk, ignoring a stored entry, so both
    # `type(name, bases, {"__annotations__": ...})` and a later
    # `C.__annotations__ = ...` answered `{}`. The write appeared to succeed
    # and the read did not see it. Refusing this function was the honest
    # answer while that held; the read now prefers what was stored.
    annotations = {}
    body = dict(namespace) if namespace else {}
    seen = set()
    for entry in fields:
        if isinstance(entry, str):
            name, kind, spec = entry, "typing.Any", MISSING
        elif len(entry) == 2:
            name, kind = entry
            spec = MISSING
        elif len(entry) == 3:
            name, kind, spec = entry
        else:
            raise TypeError("Invalid field: %r" % (entry,))
        if not isinstance(name, str) or not name.isidentifier():
            raise TypeError("Field names must be valid identifiers: %r"
                            % (name,))
        if keyword.iskeyword(name):
            raise TypeError("Field names must not be keywords: %r" % (name,))
        if name in seen:
            raise TypeError("Field name duplicated: %r" % (name,))
        seen.add(name)
        annotations[name] = kind
        if spec is not MISSING:
            body[name] = spec
    body["__annotations__"] = annotations
    made = type(cls_name, tuple(bases), body)
    return _process(made, init, repr, eq, order, unsafe_hash, frozen,
                    match_args, kw_only, slots, weakref_slot)


__all__ = [
    "dataclass", "field", "Field", "FrozenInstanceError", "InitVar",
    "KW_ONLY", "MISSING", "fields", "asdict", "astuple", "make_dataclass",
    "replace", "is_dataclass",
]
