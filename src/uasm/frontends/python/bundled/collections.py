"""Container datatypes.

COVERAGE: the whole documented surface -- `namedtuple`, `deque`,
`defaultdict`, `Counter`, `OrderedDict`, `ChainMap`, `UserDict`, `UserList`,
`UserString`.

NOT COVERED: `deque` is a LIST INSIDE, so `appendleft` and `popleft` are O(n)
where CPython's doubly-linked block list makes them O(1). Every answer is the
same; only the cost is not, and a program that can tell is one timing itself.
`namedtuple`'s generated class has no `__slots__`, and `_make`/`_replace`/
`_asdict`/`_fields`/`_field_defaults` are all here.

THREE OF THESE SUBCLASS `dict`, AND THAT IS WHAT MAKES THEM RIGHT. CPython's
`defaultdict`, `Counter` and `OrderedDict` are dicts -- `isinstance(c, dict)`
is True, `c == {...}` compares by content, `dict(c)` copies. A version built
by composition around a `.data` attribute answers all three differently, which
is why the shape of this module was decided by fixing the compiler rather than
by working around it: an instance of a builtin-extending class now carries a
real dict and delegates to it for everything its body does not define.

`UserDict`, `UserList` AND `UserString` DO NOT SUBCLASS, and that is equally
deliberate -- wrapping is their entire purpose. They exist so a program can
override one method without inheriting a builtin's C-level shortcuts, and
their content lives in `.data` because CPython's does.
"""

#: AT MODULE LEVEL, because a bundled module is SPLICED rather than imported
#: and a nested `import` has no import path to resolve against. `keyword` is
#: the only thing this module needs from outside, and only to refuse a field
#: name that would be a syntax error.
import keyword

class _NamedTuple(tuple):
    """The base every `namedtuple` class extends, and where all the work is.

    DRIVEN BY `cls._fields`, not by closures. The generated class carries the
    field names and defaults as class attributes, so every method here is
    written once and reads them off whichever subclass it was called on --
    which is shorter than generating a method per class and has no chance of
    the classic loop-closure mistake, where each accessor captures the
    variable and every field ends up reading the last index.

    `__new__` AND NOT `__init__`, because a tuple's contents cannot be set
    after it exists. That is also why `super().__new__(cls, vals)` has to
    reach the builtin base and build a real tuple; see the runtime's own
    `super()` handling for a class whose chain ends in a kind.
    """

    def __new__(cls, *args, **kwargs):
        names = cls._fields
        fallbacks = cls._field_defaults
        typename = cls.__name__
        if len(args) > len(names):
            raise TypeError(typename + "() takes " + str(len(names))
                            + " positional arguments but "
                            + str(len(args)) + " were given")
        for key in kwargs:
            if key not in names:
                raise TypeError(typename
                                + "() got an unexpected keyword argument '"
                                + key + "'")
        vals = list(args)
        for key in names[len(args):]:
            if key in kwargs:
                vals.append(kwargs[key])
            elif key in fallbacks:
                vals.append(fallbacks[key])
            else:
                raise TypeError(typename
                                + "() missing required argument: '"
                                + key + "'")
        return super().__new__(cls, vals)

    def __repr__(self):
        names = type(self)._fields
        parts = []
        for i in range(len(names)):
            parts.append(names[i] + "=" + repr(self[i]))
        return type(self).__name__ + "(" + ", ".join(parts) + ")"

    def _asdict(self):
        """A plain dict, in FIELD ORDER. CPython answers a `dict` since 3.8 --
        it used to be an OrderedDict, and a program that prints one can tell
        the difference."""
        names = type(self)._fields
        out = {}
        for i in range(len(names)):
            out[names[i]] = self[i]
        return out

    def _replace(self, **kwargs):
        names = type(self)._fields
        vals = list(self)
        for key in kwargs:
            if key not in names:
                raise ValueError("Got unexpected field names: ["
                                 + repr(key) + "]")
            vals[names.index(key)] = kwargs[key]
        return type(self)(*vals)

    def __getnewargs__(self):
        return tuple(self)


def namedtuple(typename, field_names, rename=False, defaults=None,
               module=None):
    """A tuple subclass with named fields.

    THE FIELDS ARE PROPERTIES OVER AN INDEX, which is the whole trick: the
    instance IS the tuple, so `p[1]`, `len(p)`, unpacking, `==` against a
    plain tuple and `isinstance(p, tuple)` all come from the base and cost
    nothing. Only the names are new.

    NOT `exec`. CPython builds the class by compiling generated source, which
    is how it gets a real signature and `__slots__`. Here the class is made
    with `type()` over `_NamedTuple` -- the observable difference is that
    `inspect.signature` sees `*args`, which `docs/STDLIB.md` records.
    """
    names = _parse_fields(field_names, rename)
    body = {
        "_fields": tuple(names),
        "_field_defaults": _defaults_for(names, defaults),
        "__doc__": typename + "(" + ", ".join(names) + ")",
        "__module__": module if module is not None else "__main__",
    }
    for i in range(len(names)):
        body[names[i]] = _field_at(i, names[i])
    made = type(typename, (_NamedTuple,), body)
    made._make = classmethod(_make_from)
    return made


def _make_from(cls, iterable):
    """`P._make(it)` -- the classmethod every namedtuple class gets.

    A FUNCTION AT MODULE LEVEL, wrapped once per class, because `classmethod`
    has to be bound to the class it is set on and a shared instance would
    report the wrong `cls`.
    """
    return cls(*list(iterable))


def _field_at(index, name):
    """One named accessor, over a fixed index. See `namedtuple`."""

    def read(self):
        return self[index]

    read.__name__ = name
    read.__doc__ = "Alias for field number " + str(index)
    return property(read)


def _parse_fields(field_names, rename):
    """`"x y"`, `"x,y"` or `["x", "y"]` -- all three, as CPython takes them."""
    if isinstance(field_names, str):
        names = field_names.replace(",", " ").split()
    else:
        names = [str(one) for one in field_names]
    seen = []
    out = []
    for i in range(len(names)):
        one = names[i]
        bad = (not one.isidentifier()) or _is_keyword(one) \
            or one.startswith("_") or one in seen
        if bad:
            if not rename:
                if one in seen:
                    raise ValueError("Encountered duplicate field name: "
                                     + repr(one))
                if one.startswith("_"):
                    raise ValueError(
                        "Field names cannot start with an underscore: "
                        + repr(one))
                # THE KEYWORD CASE IS ITS OWN MESSAGE, because `class` IS a
                # valid identifier and reporting it as one that is not says
                # something false about the name.
                if _is_keyword(one):
                    raise ValueError("Type names and field names cannot be a "
                                     "keyword: " + repr(one))
                raise ValueError("Type names and field names must be valid "
                                 "identifiers: " + repr(one))
            one = "_" + str(i)
        seen.append(one)
        out.append(one)
    return out


def _is_keyword(word):
    return keyword.iskeyword(word)


def _defaults_for(names, defaults):
    """`defaults` applies to the LAST fields, which is what makes it useful:
    `namedtuple("P", "x y", defaults=[0])` gives `y` the default, not `x`."""
    if defaults is None:
        return {}
    vals = list(defaults)
    if len(vals) > len(names):
        raise TypeError("Got more default values than field names")
    out = {}
    start = len(names) - len(vals)
    for i in range(len(vals)):
        out[names[start + i]] = vals[i]
    return out


class deque:
    """A double-ended queue, optionally bounded.

    A LIST INSIDE, which is a real difference in COST and in nothing else --
    see the module docstring. It is not a `list` SUBCLASS, because CPython's
    is not: `isinstance(deque(), list)` is False, and a deque has no `sort`,
    no `+` and no slicing.

    `maxlen` DISCARDS FROM THE FAR END, which is the whole point of a bounded
    deque: appending to a full one drops the leftmost, and `appendleft` drops
    the rightmost. Getting that backwards keeps the length right and the
    contents wrong.
    """

    def __init__(self, iterable=(), maxlen=None):
        if maxlen is not None and maxlen < 0:
            raise ValueError("maxlen must be non-negative")
        self.maxlen = maxlen
        self._items = []
        self.extend(iterable)

    def _trim(self, from_left):
        if self.maxlen is None:
            return
        while len(self._items) > self.maxlen:
            if from_left:
                self._items.pop(0)
            else:
                self._items.pop()

    def append(self, item):
        self._items.append(item)
        self._trim(True)

    def appendleft(self, item):
        self._items.insert(0, item)
        self._trim(False)

    def extend(self, iterable):
        for one in iterable:
            self.append(one)

    def extendleft(self, iterable):
        """LEFT-TO-RIGHT ONTO THE LEFT END, so the result is REVERSED with
        respect to the argument -- `extendleft([1, 2])` leaves `[2, 1]`.
        CPython does this and it surprises people; it falls out of each
        element being pushed onto the same end in turn."""
        for one in iterable:
            self.appendleft(one)

    def pop(self):
        if not self._items:
            raise IndexError("pop from an empty deque")
        return self._items.pop()

    def popleft(self):
        if not self._items:
            raise IndexError("pop from an empty deque")
        return self._items.pop(0)

    def rotate(self, n=1):
        """POSITIVE ROTATES RIGHT: the last `n` elements move to the front."""
        if not self._items:
            return None
        count = n % len(self._items)
        if count:
            self._items = self._items[-count:] + self._items[:-count]
        return None

    def clear(self):
        self._items = []
        return None

    def copy(self):
        return deque(self._items, self.maxlen)

    def count(self, item):
        return self._items.count(item)

    def index(self, item, start=0, stop=None):
        """WALKED BY HAND, because `list.index` takes only the value in this
        frontend -- passing a start and a stop to it is an arity error about a
        method the program never wrote."""
        end = len(self._items) if stop is None else stop
        at = start
        while at < end and at < len(self._items):
            if self._items[at] == item:
                return at
            at = at + 1
        raise ValueError(repr(item) + " is not in deque")

    def insert(self, at, item):
        if self.maxlen is not None and len(self._items) >= self.maxlen:
            raise IndexError("deque already at its maximum size")
        self._items.insert(at, item)
        return None

    def remove(self, item):
        if item not in self._items:
            raise ValueError("deque.remove(x): x not in deque")
        self._items.remove(item)
        return None

    def reverse(self):
        self._items.reverse()
        return None

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __reversed__(self):
        return iter(list(reversed(self._items)))

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError("sequence index must be integer, not 'slice'")
        return self._items[index]

    def __setitem__(self, index, value):
        self._items[index] = value
        return None

    def __delitem__(self, index):
        del self._items[index]
        return None

    def __contains__(self, item):
        return item in self._items

    def __bool__(self):
        return len(self._items) != 0

    def __eq__(self, other):
        if isinstance(other, deque):
            return self._items == other._items
        return NotImplemented

    def __ne__(self, other):
        got = self.__eq__(other)
        return got if got is NotImplemented else not got

    def __add__(self, other):
        if isinstance(other, deque):
            return deque(self._items + other._items, self.maxlen)
        return NotImplemented

    def __repr__(self):
        if self.maxlen is None:
            return "deque(" + repr(self._items) + ")"
        return "deque(" + repr(self._items) + ", maxlen=" \
            + str(self.maxlen) + ")"


class defaultdict(dict):
    """A dict that FILLS A MISSING KEY instead of raising.

    `__missing__` IS THE WHOLE MECHANISM, and it is the runtime's rather than
    this class's: `d[k]` on a dict subclass consults it when the lookup
    misses. So the body here is one method, which is also what CPython's is.

    A MISS THROUGH `[]` INSERTS; a miss through `get` DOES NOT. That asymmetry
    is CPython's and it is the reason `d["b"]` then `"b" in d` answers True
    while `d.get("b")` leaves the dict alone.
    """

    def __init__(self, default_factory=None, *args, **kwargs):
        if default_factory is not None and not callable(default_factory):
            raise TypeError("first argument must be callable or None")
        self.default_factory = default_factory
        # THE KEYWORDS ARE SET ONE AT A TIME rather than forwarded. A `**kw`
        # handed to the builtin base's constructor does not reach it here --
        # the stand-in for `dict.__init__` takes positional content only --
        # and the failure is a silently EMPTY mapping. Writing the loop out
        # also means every key goes through `__setitem__`, which is what a
        # subclass overriding it would expect.
        if args:
            super().__init__(args[0])
        for key in kwargs:
            self[key] = kwargs[key]

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        made = self.default_factory()
        self[key] = made
        return made

    def copy(self):
        return defaultdict(self.default_factory, dict(self))

    def __copy__(self):
        return self.copy()

    def __repr__(self):
        return "defaultdict(" + repr(self.default_factory) + ", " \
            + repr(dict(self)) + ")"


class Counter(dict):
    """A dict counting hashable things, whose MISSING KEY IS ZERO.

    ZERO WITHOUT INSERTING, which is the difference from `defaultdict(int)`
    and the reason `__missing__` here does not assign: `c["z"]` on an absent
    key answers 0 and leaves the Counter with the same length. A Counter that
    grew every time it was queried would report a different `len` after a
    read, and reading is what a Counter is for.

    NEGATIVE AND ZERO COUNTS ARE KEPT by `update` and dropped by the operators
    -- `+`, `-`, `&` and `|` all return only positive counts, which is
    CPython's rule and the one people trip over.
    """

    def __init__(self, iterable=None, **kwargs):
        super().__init__()
        self.update(iterable, **kwargs)

    def __missing__(self, key):
        return 0

    def update(self, iterable=None, **kwargs):
        """ADDS rather than replaces, which is the opposite of `dict.update`
        and deliberate: `c.update("ab")` counts another a and another b."""
        if iterable is not None:
            if isinstance(iterable, dict):
                for key in iterable:
                    self[key] = self[key] + iterable[key]
            else:
                for one in iterable:
                    self[one] = self[one] + 1
        for key in kwargs:
            self[key] = self[key] + kwargs[key]
        return None

    def subtract(self, iterable=None, **kwargs):
        """The mirror of `update`, and it KEEPS negatives -- that is what
        distinguishes it from `-`."""
        if iterable is not None:
            if isinstance(iterable, dict):
                for key in iterable:
                    self[key] = self[key] - iterable[key]
            else:
                for one in iterable:
                    self[one] = self[one] - 1
        for key in kwargs:
            self[key] = self[key] - kwargs[key]
        return None

    def most_common(self, n=None):
        """Highest count first, TIES IN INSERTION ORDER.

        CPython's sort is stable over the dict's own order, so two keys with
        the same count come back in the order they were first seen. A sort
        that broke ties by key would be tidier and would not match.
        """
        pairs = []
        for key in list(self):
            pairs.append((key, self[key]))
        ordered = sorted(pairs, key=_second, reverse=True)
        return ordered if n is None else ordered[:n]

    def elements(self):
        """Each key repeated `count` times. A count of zero or less yields
        nothing, which is what makes this the inverse of the constructor."""
        out = []
        for key in list(self):
            times = self[key]
            i = 0
            while i < times:
                out.append(key)
                i = i + 1
        return iter(out)

    def total(self):
        """The sum of the counts, negatives included. Added in 3.10."""
        got = 0
        for key in self:
            got = got + self[key]
        return got

    def copy(self):
        return Counter(dict(self))

    def __copy__(self):
        return self.copy()

    def __repr__(self):
        # EMPTY IS `Counter()`, as it is for `OrderedDict`.
        if not len(self):
            return "Counter()"
        pairs = self.most_common()
        parts = []
        for one in pairs:
            parts.append(repr(one[0]) + ": " + repr(one[1]))
        return "Counter({" + ", ".join(parts) + "})"

    def __add__(self, other):
        if not isinstance(other, Counter):
            return NotImplemented
        out = Counter()
        for key in list(self):
            got = self[key] + other[key]
            if got > 0:
                out[key] = got
        for key in list(other):
            if key not in self and other[key] > 0:
                out[key] = other[key]
        return out

    def __sub__(self, other):
        if not isinstance(other, Counter):
            return NotImplemented
        out = Counter()
        for key in list(self):
            got = self[key] - other[key]
            if got > 0:
                out[key] = got
        return out

    def __or__(self, other):
        """The larger of the two counts -- a union."""
        if not isinstance(other, Counter):
            return NotImplemented
        out = Counter()
        for key in list(self):
            got = self[key] if self[key] > other[key] else other[key]
            if got > 0:
                out[key] = got
        for key in list(other):
            if key not in self and other[key] > 0:
                out[key] = other[key]
        return out

    def __and__(self, other):
        """The smaller of the two counts -- an intersection."""
        if not isinstance(other, Counter):
            return NotImplemented
        out = Counter()
        for key in list(self):
            got = self[key] if self[key] < other[key] else other[key]
            if got > 0:
                out[key] = got
        return out

    def __pos__(self):
        out = Counter()
        for key in list(self):
            if self[key] > 0:
                out[key] = self[key]
        return out

    def __neg__(self):
        out = Counter()
        for key in list(self):
            if self[key] < 0:
                out[key] = -self[key]
        return out


def _second(pair):
    return pair[1]


class OrderedDict(dict):
    """A dict that remembers insertion order and COMPARES BY IT.

    A PLAIN DICT HAS KEPT INSERTION ORDER SINCE 3.7, so what is left of this
    class is the two things a plain dict still does not do: `move_to_end`, and
    an `__eq__` that is ORDER-SENSITIVE against another OrderedDict. The
    second is the one worth stating -- `OrderedDict(a=1, b=2)` and
    `OrderedDict(b=2, a=1)` are NOT equal, while the plain dicts with the same
    contents are.

    AGAINST A PLAIN DICT IT IS ORDER-INSENSITIVE, which looks inconsistent and
    is CPython's rule: the comparison is only strict when both sides claim to
    care about order.
    """

    def __init__(self, *args, **kwargs):
        # See `defaultdict.__init__` for why the keywords are not forwarded.
        if args:
            super().__init__(args[0])
        for key in kwargs:
            self[key] = kwargs[key]

    def move_to_end(self, key, last=True):
        if key not in self:
            raise KeyError(key)
        value = self[key]
        del self[key]
        if last:
            self[key] = value
        else:
            # TO THE FRONT means rebuilding: a dict can only append, so the
            # rest is removed and put back after the key that moved.
            rest = []
            for other in list(self):
                rest.append((other, self[other]))
            for pair in rest:
                del self[pair[0]]
            self[key] = value
            for pair in rest:
                self[pair[0]] = pair[1]
        return None

    def popitem(self, last=True):
        keys = list(self)
        if not keys:
            raise KeyError("dictionary is empty")
        key = keys[-1] if last else keys[0]
        value = self[key]
        del self[key]
        return (key, value)

    def copy(self):
        return OrderedDict(self)

    def __eq__(self, other):
        if isinstance(other, OrderedDict):
            return list(self) == list(other) \
                and dict(self) == dict(other)
        if isinstance(other, dict):
            return dict(self) == other
        return NotImplemented

    def __ne__(self, other):
        got = self.__eq__(other)
        return got if got is NotImplemented else not got

    def __repr__(self):
        # EMPTY IS `OrderedDict()`, with no braces -- CPython's spelling, and
        # the one a program reading its own repr back expects. The braces
        # only appear once there is something to put in them.
        if not len(self):
            return "OrderedDict()"
        parts = []
        for key in list(self):
            parts.append(repr(key) + ": " + repr(self[key]))
        return "OrderedDict({" + ", ".join(parts) + "})"


class ChainMap:
    """Several mappings searched in order, as one.

    NOTHING IS COPIED, which is the entire point: a lookup walks the maps
    front to back, and a WRITE goes to the FIRST one only. That asymmetry is
    what makes a ChainMap useful for layered configuration and what makes
    `cm["y"] = 99` change `maps[0]` while the shadowed `y` further back stays
    exactly where it was.
    """

    def __init__(self, *maps):
        self.maps = list(maps) if maps else [{}]

    def __getitem__(self, key):
        for one in self.maps:
            if key in one:
                return one[key]
        return self.__missing__(key)

    def __missing__(self, key):
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.maps[0][key] = value
        return None

    def __delitem__(self, key):
        if key not in self.maps[0]:
            raise KeyError("Key not found in the first mapping: "
                           + repr(key))
        del self.maps[0][key]
        return None

    def __contains__(self, key):
        for one in self.maps:
            if key in one:
                return True
        return False

    def __iter__(self):
        return iter(self._keys())

    def __len__(self):
        return len(self._keys())

    def __bool__(self):
        return len(self._keys()) != 0

    def _keys(self):
        """Every key once. LAST MAP FIRST so that a key shadowed by an earlier
        map keeps the POSITION of its first appearance, which is the order
        CPython's own `__iter__` produces."""
        out = []
        for one in reversed(self.maps):
            for key in one:
                if key not in out:
                    out.append(key)
        return out

    def keys(self):
        return self._keys()

    def values(self):
        out = []
        for key in self._keys():
            out.append(self[key])
        return out

    def items(self):
        out = []
        for key in self._keys():
            out.append((key, self[key]))
        return out

    def get(self, key, default=None):
        return self[key] if key in self else default

    def pop(self, key, *args):
        if key not in self.maps[0]:
            if args:
                return args[0]
            raise KeyError("Key not found in the first mapping: "
                           + repr(key))
        value = self.maps[0][key]
        del self.maps[0][key]
        return value

    def popitem(self):
        if not self.maps[0]:
            raise KeyError("No keys found in the first mapping.")
        return self.maps[0].popitem()

    def clear(self):
        self.maps[0].clear()
        return None

    def setdefault(self, key, default=None):
        if key not in self:
            self.maps[0][key] = default
        return self[key]

    def new_child(self, m=None):
        """A NEW FRONT MAP, sharing the rest. The usual way to enter a scope."""
        return ChainMap(m if m is not None else {}, *self.maps)

    @property
    def parents(self):
        """Everything but the front map -- leaving a scope."""
        return ChainMap(*self.maps[1:])

    def copy(self):
        return ChainMap(dict(self.maps[0]), *self.maps[1:])

    def __repr__(self):
        parts = []
        for one in self.maps:
            parts.append(repr(one))
        return "ChainMap(" + ", ".join(parts) + ")"


class UserDict:
    """A dict WRAPPED rather than extended, with its content in `.data`.

    WHY WRAPPING IS THE POINT. Overriding `__setitem__` on a real `dict`
    subclass does not catch `update` or the constructor, because those go
    through C-level shortcuts -- so a subclass that meant to validate every
    write silently misses most of them. Everything here goes through
    `__setitem__`, so overriding it catches all of them. That is the whole
    reason this class exists next to `dict`, and it is why it must NOT be a
    dict subclass however tempting the symmetry.
    """

    def __init__(self, dict=None, **kwargs):
        self.data = {}
        if dict is not None:
            self.update(dict)
        if kwargs:
            self.update(kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        missing = getattr(self.__class__, "__missing__", None)
        if missing is not None:
            return missing(self, key)
        raise KeyError(key)

    def __setitem__(self, key, item):
        self.data[key] = item
        return None

    def __delitem__(self, key):
        del self.data[key]
        return None

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, key):
        return key in self.data

    def __eq__(self, other):
        if isinstance(other, UserDict):
            return self.data == other.data
        if isinstance(other, dict):
            return self.data == other
        return NotImplemented

    def __ne__(self, other):
        got = self.__eq__(other)
        return got if got is NotImplemented else not got

    def __repr__(self):
        return repr(self.data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def get(self, key, default=None):
        return self.data[key] if key in self.data else default

    def pop(self, key, *args):
        if key in self.data:
            value = self.data[key]
            del self.data[key]
            return value
        if args:
            return args[0]
        raise KeyError(key)

    def popitem(self):
        return self.data.popitem()

    def setdefault(self, key, default=None):
        if key not in self.data:
            self[key] = default
        return self.data[key]

    def update(self, other=None, **kwargs):
        if other is not None:
            keys = other.keys() if hasattr(other, "keys") else None
            if keys is not None:
                for key in keys:
                    self[key] = other[key]
            else:
                for pair in other:
                    self[pair[0]] = pair[1]
        for key in kwargs:
            self[key] = kwargs[key]
        return None

    def clear(self):
        self.data.clear()
        return None

    def copy(self):
        return UserDict(dict(self.data))


class UserList:
    """A list WRAPPED rather than extended, with its content in `.data`.
    See `UserDict` for why wrapping is the point."""

    def __init__(self, initlist=None):
        self.data = []
        if initlist is not None:
            self.data = list(initlist)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return UserList(self.data[index])
        return self.data[index]

    def __setitem__(self, index, item):
        self.data[index] = item
        return None

    def __delitem__(self, index):
        del self.data[index]
        return None

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item):
        return item in self.data

    def __eq__(self, other):
        if isinstance(other, UserList):
            return self.data == other.data
        if isinstance(other, list):
            return self.data == other
        return NotImplemented

    def __ne__(self, other):
        got = self.__eq__(other)
        return got if got is NotImplemented else not got

    def __lt__(self, other):
        return self.data < (other.data if isinstance(other, UserList)
                            else other)

    def __le__(self, other):
        return self.data <= (other.data if isinstance(other, UserList)
                             else other)

    def __gt__(self, other):
        return self.data > (other.data if isinstance(other, UserList)
                            else other)

    def __ge__(self, other):
        return self.data >= (other.data if isinstance(other, UserList)
                             else other)

    def __add__(self, other):
        return UserList(self.data
                        + (other.data if isinstance(other, UserList)
                           else list(other)))

    def __radd__(self, other):
        return UserList(list(other) + self.data)

    def __mul__(self, n):
        return UserList(self.data * n)

    def __rmul__(self, n):
        return UserList(self.data * n)

    def __repr__(self):
        return repr(self.data)

    def append(self, item):
        self.data.append(item)
        return None

    def insert(self, at, item):
        self.data.insert(at, item)
        return None

    def pop(self, index=-1):
        return self.data.pop(index)

    def remove(self, item):
        self.data.remove(item)
        return None

    def clear(self):
        self.data.clear()
        return None

    def copy(self):
        return UserList(self.data)

    def count(self, item):
        return self.data.count(item)

    def index(self, item, start=0, end=None):
        at = start
        stop = len(self.data) if end is None else end
        while at < stop and at < len(self.data):
            if self.data[at] == item:
                return at
            at = at + 1
        raise ValueError(repr(item) + " is not in list")

    def reverse(self):
        self.data.reverse()
        return None

    def sort(self, key=None, reverse=False):
        self.data.sort(key=key, reverse=reverse)
        return None

    def extend(self, other):
        self.data.extend(other.data if isinstance(other, UserList)
                         else list(other))
        return None


class UserString:
    """A str WRAPPED rather than extended, with its content in `.data`.
    See `UserDict` for why wrapping is the point.

    EVERY OPERATION ANSWERS A UserString, not a str -- that is what makes a
    subclass of this one useful, since a method that returned a bare str
    would drop back out of the subclass on the first `.upper()`.

    THE OPTIONAL ARGUMENTS ARE WRITTEN OUT rather than forwarded as `*args`.
    A starred call is lowered as a SPREAD, which reaches the receiver's method
    through `getattr` -- and a `str` in this frontend does not answer its
    method names that way, so `us.replace("a", "b")` reported that a str has
    no attribute `replace`. Naming each parameter keeps the call direct.
    """

    def __init__(self, seq=""):
        if isinstance(seq, UserString):
            self.data = seq.data
        elif isinstance(seq, str):
            self.data = seq
        else:
            self.data = str(seq)

    def __str__(self):
        return self.data

    def __repr__(self):
        return repr(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return UserString(self.data[index])

    def __contains__(self, item):
        return str(item) in self.data

    def __iter__(self):
        out = []
        for ch in self.data:
            out.append(UserString(ch))
        return iter(out)

    def __eq__(self, other):
        return self.data == _text_of(other)

    def __ne__(self, other):
        return self.data != _text_of(other)

    def __lt__(self, other):
        return self.data < _text_of(other)

    def __le__(self, other):
        return self.data <= _text_of(other)

    def __gt__(self, other):
        return self.data > _text_of(other)

    def __ge__(self, other):
        return self.data >= _text_of(other)

    def __hash__(self):
        return hash(self.data)

    def __add__(self, other):
        return UserString(self.data + _text_of(other))

    def __radd__(self, other):
        return UserString(_text_of(other) + self.data)

    def __mul__(self, n):
        return UserString(self.data * n)

    def __rmul__(self, n):
        return UserString(self.data * n)

    def __mod__(self, args):
        return UserString(self.data % args)

    def capitalize(self):
        return UserString(self.data.capitalize())

    def casefold(self):
        return UserString(self.data.casefold())

    def center(self, width, fillchar=" "):
        return UserString(self.data.center(width, fillchar))

    def count(self, sub, start=0, end=None):
        return self.data.count(_text_of(sub), start,
                               len(self.data) if end is None else end)

    def encode(self, encoding="utf-8", errors="strict"):
        return self.data.encode(encoding, errors)

    def endswith(self, suffix, start=0, end=None):
        return self.data.endswith(suffix, start,
                                  len(self.data) if end is None else end)

    def find(self, sub, start=0, end=None):
        return self.data.find(_text_of(sub), start,
                              len(self.data) if end is None else end)

    def index(self, sub, start=0, end=None):
        return self.data.index(_text_of(sub), start,
                               len(self.data) if end is None else end)

    def isalpha(self):
        return self.data.isalpha()

    def isalnum(self):
        return self.data.isalnum()

    def isdigit(self):
        return self.data.isdigit()

    def isidentifier(self):
        return self.data.isidentifier()

    def islower(self):
        return self.data.islower()

    def isnumeric(self):
        return self.data.isnumeric()

    def isspace(self):
        return self.data.isspace()

    def istitle(self):
        return self.data.istitle()

    def isupper(self):
        return self.data.isupper()

    def join(self, seq):
        return UserString(self.data.join(seq))

    def ljust(self, width, fillchar=" "):
        return UserString(self.data.ljust(width, fillchar))

    def lower(self):
        return UserString(self.data.lower())

    def lstrip(self, chars=None):
        return UserString(self.data.lstrip(chars))

    def partition(self, sep):
        got = self.data.partition(sep)
        return (UserString(got[0]), UserString(got[1]), UserString(got[2]))

    def removeprefix(self, prefix):
        return UserString(self.data.removeprefix(_text_of(prefix)))

    def removesuffix(self, suffix):
        return UserString(self.data.removesuffix(_text_of(suffix)))

    def replace(self, old, new, count=-1):
        return UserString(self.data.replace(_text_of(old), _text_of(new),
                                            count))

    def rfind(self, sub, start=0, end=None):
        return self.data.rfind(_text_of(sub), start,
                               len(self.data) if end is None else end)

    def rindex(self, sub, start=0, end=None):
        return self.data.rindex(_text_of(sub), start,
                                len(self.data) if end is None else end)

    def rjust(self, width, fillchar=" "):
        return UserString(self.data.rjust(width, fillchar))

    def rpartition(self, sep):
        got = self.data.rpartition(sep)
        return (UserString(got[0]), UserString(got[1]), UserString(got[2]))

    def rsplit(self, sep=None, maxsplit=-1):
        return self.data.rsplit(sep, maxsplit)

    def rstrip(self, chars=None):
        return UserString(self.data.rstrip(chars))

    def split(self, sep=None, maxsplit=-1):
        return self.data.split(sep, maxsplit)

    def splitlines(self, keepends=False):
        return self.data.splitlines(keepends)

    def startswith(self, prefix, start=0, end=None):
        return self.data.startswith(prefix, start,
                                    len(self.data) if end is None else end)

    def strip(self, chars=None):
        return UserString(self.data.strip(chars))

    def swapcase(self):
        return UserString(self.data.swapcase())

    def title(self):
        return UserString(self.data.title())

    def upper(self):
        return UserString(self.data.upper())

    def zfill(self, width):
        return UserString(self.data.zfill(width))


def _text_of(value):
    return value.data if isinstance(value, UserString) else value
