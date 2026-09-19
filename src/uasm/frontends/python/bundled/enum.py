"""Enumerations, as ordinary Python this compiler compiles.

COVERAGE: `Enum`, `IntEnum`, `StrEnum`, `Flag`, `IntFlag`, `EnumMeta` (and its
3.11 spelling `EnumType`), `auto`, `unique`, `__members__`, aliases, lookup by
value and by name, iteration in definition order, and the 3.11+ reprs.

NOT COVERED: `ReprEnum` as a base to inherit from, `verify` and `EnumCheck`,
`FlagBoundary` and `boundary=`, `global_enum`, `member`/`nonmember`,
`_missing_`, `_generate_next_value_` overridden by a subclass, pickling, and
functional creation (`Enum("Color", "RED GREEN")`).

The WORDING of one error differs. A flag given a bit nobody declared is a
ValueError here as it is in CPython, and CPython's message is three lines
showing the given and allowed bits in binary; this one is the ordinary
`8 is not a valid Perm`. The refusal is the contract and the ASCII art is not.

THE ONE DIVERGENCE WORTH KNOWING. CPython's `IntEnum` members ARE ints and its
`StrEnum` members ARE strs -- they inherit the builtin, so `isinstance(Num.ONE,
int)` is True and `Num.ONE` can be handed to anything expecting one. There is
no subclassing of a builtin here, so the operators are written out instead:
`Num.ONE == 1`, `Num.ONE + 1`, `Num.ONE < 2`, `str(Colour.RED) == "red"` and
sorting all behave, and `isinstance(Num.ONE, int)` is False. That is stated
because a program that TESTS for int-ness will take a different branch here,
and finding that out from behaviour rather than from a docstring is the thing
this rebuild exists to stop.

## Why a metaclass

An `Enum` is a class whose body names constants and whose metaclass turns each
of them into an INSTANCE of that class. `Color.RED` is not the integer 1: it
is an object that knows it is called RED and that its value is 1, and
`Color(1)` finds it by that value. The rewriting has to happen while the class
is being built, which is what a metaclass is and why nothing simpler will do.
"""


class auto:
    """A value the metaclass fills in.

    One past the highest so far in an ordinary enum, and THE NEXT BIT in a
    flag -- `A = auto()` through `D = auto()` is 1, 2, 4, 8. Two rules rather
    than one because a flag whose members were 1, 2, 3 could not be combined:
    `A | B` would collide with `C` and mean something the author did not write.
    """


def _is_descriptor(value):
    return (hasattr(value, "__get__") or hasattr(value, "__set__")
            or hasattr(value, "__delete__"))


def _next_bit(highest):
    if highest <= 0:
        return 1
    bit = 1
    while bit <= highest:
        bit = bit * 2
    return bit


def _kind(bases, marker):
    for base in bases:
        if getattr(base, marker, False):
            return True
    return False


class EnumMeta(type):
    """What turns a class body full of constants into a class full of members."""

    def __new__(mcls, name, bases, namespace):
        # WHICH NAMES ARE MEMBERS: the ones that are neither dunders nor
        # callables nor descriptors. A method in an enum body stays a method
        # -- that is how `Color.describe()` works -- and a `_private` name is
        # the enum's own bookkeeping.
        flag = _kind(bases, "_flag_") or namespace.get("_flag_", False)
        chosen = []
        rest = {}
        highest = 0
        for key in namespace:
            value = namespace[key]
            if key.startswith("_") or callable(value) or _is_descriptor(value):
                rest[key] = value
                continue
            if isinstance(value, auto):
                if flag:
                    value = _next_bit(highest)
                else:
                    value = highest + 1
            if isinstance(value, int) and not isinstance(value, bool):
                # `auto()` COUNTS FROM THE HIGHEST WRITTEN VALUE and not from
                # the number of members: `RED = 5` then `GREEN = auto()` is 6.
                if value > highest:
                    highest = value
            chosen.append((key, value))
            rest[key] = None
        cls = super().__new__(mcls, name, bases, rest)
        members = []
        by_value = {}
        by_name = {}
        for pair in chosen:
            key = pair[0]
            value = pair[1]
            # AN ALIAS IS THE SAME OBJECT, not a second member of equal value.
            # `CRIMSON = 1` beside `RED = 1` makes `Color.CRIMSON is Color.RED`
            # true and leaves CRIMSON out of iteration -- it is another
            # spelling rather than another thing.
            found = by_value.get(value)
            if found is not None:
                setattr(cls, key, found)
                by_name[key] = found
                continue
            # THE VALUE GOES IN THE BUILTIN HALF, which is what makes a
            # member of `class Colour(str, Enum)` a str -- and it is
            # `member_type.__new__(enum_class, value)` that puts it there,
            # which is what CPython's own enum writes. THE CLASS DECIDES
            # AND NOT THE VALUE: `class Colour(Enum)` with `RED = "red"`
            # extends nothing, and asking the value would hand a plain enum
            # to `str.__new__` and be told it is not a subtype of str.
            # `str` and `tuple` are the two mixins that take their content at
            # construction; every other kind fills in `__init__`, and an enum
            # extending nothing has no half to fill.
            if issubclass(cls, str):
                member = str.__new__(cls, value)
            elif issubclass(cls, tuple):
                member = tuple.__new__(cls, value)
            else:
                member = object.__new__(cls)
            member._name_ = key
            member._value_ = value
            setattr(cls, key, member)
            members.append(member)
            by_value[value] = member
            by_name[key] = member
        cls._member_list_ = members
        cls._value2member_ = by_value
        cls._member_map_ = by_name
        # SET RATHER THAN COMPUTED BY A PROPERTY ON THE METACLASS. A property
        # there is read through the metaclass for the CLASS and through the
        # class for a MEMBER, and the two would need different answers.
        # ALIASES ARE IN IT, which is the whole difference from iterating the
        # class: `__members__` is every spelling and iteration is every thing.
        cls.__members__ = by_name
        return cls

    def __call__(cls, *args, **kwargs):
        """`Color(1)` LOOKS A MEMBER UP; it does not build one.

        An enum class is not instantiable by a program -- every member was
        made when the class was -- so calling it is a query, and a value with
        no member is a ValueError rather than a new object.
        """
        if len(args) == 1 and not kwargs:
            found = cls._value2member_
            if args[0] in found:
                return found[args[0]]
            # A COMBINATION NOBODY DECLARED IS STILL A VALUE OF A FLAG, but
            # only if every bit in it was declared: `Colour(3)` is RED|GREEN
            # and `Colour(8)` is an error, because 8 is not a colour and
            # answering with a nameless member would invent one.
            if getattr(cls, "_flag_", False) and isinstance(args[0], int) \
                    and not isinstance(args[0], bool) and args[0] >= 0:
                whole = 0
                for one in cls._member_list_:
                    whole = whole | one._value_
                if (args[0] & ~whole) == 0:
                    return _compose(cls, args[0])
            raise ValueError("%r is not a valid %s" % (args[0], cls.__name__))
        return super().__call__(*args, **kwargs)

    def __iter__(cls):
        return iter(cls._member_list_)

    def __len__(cls):
        return len(cls._member_list_)

    def __getitem__(cls, key):
        return cls._member_map_[key]

    def __contains__(cls, item):
        return item in cls._member_list_

    def __repr__(cls):
        return "<enum %r>" % (cls.__name__,)


#: The 3.11 spelling. The same object, because a program that writes one and a
#: library that writes the other must agree about what a metaclass IS.
EnumType = EnumMeta


class Enum(metaclass=EnumMeta):
    @property
    def name(self):
        return self._name_

    @property
    def value(self):
        return self._value_

    def __repr__(self):
        return "<%s.%s: %r>" % (type(self).__name__, self._name_,
                                self._value_)

    def __str__(self):
        return "%s.%s" % (type(self).__name__, self._name_)

    def __format__(self, spec):
        return str(self)

    def __hash__(self):
        return hash(self._name_)


def _as_int(value):
    return value._value_ if isinstance(value, Enum) else value


class IntEnum(Enum):
    """An enum whose members compare and calculate as their integer values.

    `__str__` IS THE NUMBER'S, and that is 3.11's change rather than an
    oversight: `IntEnum` became a `ReprEnum`, so `str(Num.ONE)` is `1` and
    only `repr` still names the member. A program printing one into a
    template depends on it.
    """

    def __str__(self):
        return str(self._value_)

    def __format__(self, spec):
        return format(self._value_, spec)

    def __eq__(self, other):
        if isinstance(other, Enum):
            return self is other
        return self._value_ == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._value_)

    def __int__(self):
        return self._value_

    def __index__(self):
        return self._value_

    def __bool__(self):
        return bool(self._value_)

    def __lt__(self, other):
        return self._value_ < _as_int(other)

    def __le__(self, other):
        return self._value_ <= _as_int(other)

    def __gt__(self, other):
        return self._value_ > _as_int(other)

    def __ge__(self, other):
        return self._value_ >= _as_int(other)

    def __add__(self, other):
        return self._value_ + _as_int(other)

    def __radd__(self, other):
        return _as_int(other) + self._value_

    def __sub__(self, other):
        return self._value_ - _as_int(other)

    def __rsub__(self, other):
        return _as_int(other) - self._value_

    def __mul__(self, other):
        return self._value_ * _as_int(other)

    def __rmul__(self, other):
        return _as_int(other) * self._value_


class StrEnum(str, Enum):
    """An enum whose members ARE their strings.

    A `ReprEnum` like `IntEnum`, so `str(Colour.RED)` is `red` and `repr` is
    what still names the member.

    EXTENDS `str`, which is what CPython's does -- `class StrEnum(str,
    ReprEnum)`. Written as a plain `Enum` it emulated the string through
    `__str__`, `__eq__` and the orderings, which covered printing and
    comparison and nothing else: `Name.A.upper()` was an AttributeError and
    `"-".join([Name.A])` refused the member by kind.
    """

    def __str__(self):
        return self._value_

    def __format__(self, spec):
        return format(self._value_, spec)

    def __eq__(self, other):
        if isinstance(other, Enum):
            return self is other
        return self._value_ == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._value_)

    def __lt__(self, other):
        return self._value_ < _as_str(other)

    def __le__(self, other):
        return self._value_ <= _as_str(other)

    def __gt__(self, other):
        return self._value_ > _as_str(other)

    def __ge__(self, other):
        return self._value_ >= _as_str(other)

    def __add__(self, other):
        return self._value_ + _as_str(other)

    def __radd__(self, other):
        return _as_str(other) + self._value_

    def __len__(self):
        return len(self._value_)


def _as_str(value):
    return value._value_ if isinstance(value, Enum) else value


class Flag(Enum):
    """Members meant to be COMBINED, so their values are single bits.

    `RED | GREEN` is a member too -- one nobody wrote, with both bits set and
    a name made of the members it holds. That is what makes a flag different
    from an enum that happens to hold numbers: the combination is a value of
    the type rather than a plain integer that has left the type behind.
    """

    _flag_ = True

    def __iter__(self):
        return iter(_parts(type(self), self._value_))

    def __contains__(self, other):
        want = _as_int(other)
        return (self._value_ & want) == want

    def __bool__(self):
        return bool(self._value_)

    def __or__(self, other):
        return _compose(type(self), self._value_ | _as_int(other))

    def __and__(self, other):
        return _compose(type(self), self._value_ & _as_int(other))

    def __xor__(self, other):
        return _compose(type(self), self._value_ ^ _as_int(other))

    def __invert__(self):
        # WITHIN THE DECLARED BITS. `~RED` is every other member, not an
        # integer with the sign bit set: the complement of a flag has to stay
        # a value of the same type or it is not a flag any more.
        whole = 0
        for one in type(self)._member_list_:
            whole = whole | one._value_
        return _compose(type(self), whole & ~self._value_)

    def __repr__(self):
        return "<%s.%s: %r>" % (type(self).__name__, _flag_name(self),
                                self._value_)

    def __str__(self):
        return "%s.%s" % (type(self).__name__, _flag_name(self))

    def __hash__(self):
        return hash(self._value_)

    def __eq__(self, other):
        # A PLAIN FLAG IS NOT ITS NUMBER. `Colour.RED == 1` is False in
        # CPython, and only `IntFlag` makes it True -- which is the entire
        # difference between the two and the reason both exist.
        # AND NOT ANOTHER FLAG'S MEMBER EITHER. Two unrelated flags both
        # numbering their first member 1 are not equal, so the test is the
        # CLASS and not the value -- `IntFlag` is where equality reaches
        # across, because there the members really are integers.
        if type(self) is type(other):
            return self._value_ == other._value_
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


class IntFlag(Flag):
    """A `Flag` that is also arithmetic, as `IntEnum` is to `Enum`."""

    def __str__(self):
        return str(self._value_)

    def __format__(self, spec):
        return format(self._value_, spec)

    def __eq__(self, other):
        if isinstance(other, Enum):
            return self._value_ == other._value_
        return self._value_ == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._value_)

    def __int__(self):
        return self._value_

    def __index__(self):
        return self._value_

    def __or__(self, other):
        return _compose(type(self), self._value_ | _as_int(other))

    def __ror__(self, other):
        return _compose(type(self), _as_int(other) | self._value_)

    def __and__(self, other):
        return _compose(type(self), self._value_ & _as_int(other))

    def __rand__(self, other):
        return _compose(type(self), _as_int(other) & self._value_)


def _parts(cls, value):
    """The declared members whose bit is set, in definition order."""
    out = []
    for one in cls._member_list_:
        bit = one._value_
        if bit and (value & bit) == bit:
            out.append(one)
    return out


def _flag_name(member):
    if member._name_ is not None:
        return member._name_
    found = _parts(type(member), member._value_)
    if not found:
        return str(member._value_)
    return "|".join([one._name_ for one in found])


def _compose(cls, value):
    """The member for `value`, made and REMEMBERED if nobody declared it.

    Remembered so that `(RED | GREEN) is (RED | GREEN)` -- a program comparing
    two combinations with `is`, or using one as a dict key, needs the same
    object for the same bits.
    """
    found = cls._value2member_.get(value)
    if found is not None:
        return found
    made = object.__new__(cls)
    made._name_ = None
    made._value_ = value
    cls._value2member_[value] = made
    return made


def unique(enumeration):
    """Refuse a class that has aliases. A decorator, so the class says so."""
    duplicates = []
    for name in enumeration._member_map_:
        member = enumeration._member_map_[name]
        if name != member._name_:
            duplicates.append("%s -> %s" % (name, member._name_))
    if duplicates:
        raise ValueError("duplicate values found in %r: %s"
                         % (enumeration, ", ".join(duplicates)))
    return enumeration
