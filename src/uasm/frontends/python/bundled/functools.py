"""`functools`, as ordinary Python this compiler compiles.

COVERAGE: reduce, wraps, total_ordering, partial, cached_property, lru_cache, cache, singledispatch.

NOT COVERED: partialmethod, singledispatchmethod, cmp_to_key, update_wrapper as a public name, and
`reduce`'s C fast path.

Restored from `archived/stdlib-prerefactor/` and measured against CPython by
`tests/stdlib/functools.py`. The coverage line above is the module's contract; see
`docs/STDLIB.md`.

A module written HERE rather than as runtime C is one that any Python
programmer can read and fix, and one that cannot drift from the semantics it
is copying -- because it IS the semantics. The frontend splices these
definitions into the program that imports them; see `bundled.py`.

What may be written here is the subset the compiler accepts. That is a real
constraint and a useful one: a construct this module cannot use is one the
compiler should probably support.
"""


def reduce(fn, xs, *rest):
    """`reduce(f, xs)` and `reduce(f, xs, initial)`."""
    items = list(xs)
    if rest:
        acc = rest[0]
        start = 0
    else:
        if not items:
            raise TypeError(
                "reduce() of empty iterable with no initial value")
        acc = items[0]
        start = 1
    for i in range(start, len(items)):
        acc = fn(acc, items[i])
    return acc


def wraps(fn):
    """Copy the identifying attributes of `fn` onto the wrapper.

    `__wrapped__` is part of it: a decorator that hides the function it wraps
    still has to hand it back to anyone who asks.
    """
    def deco(inner):
        inner.__name__ = fn.__name__
        inner.__doc__ = fn.__doc__
        inner.__wrapped__ = fn
        return inner
    return deco


# ── total_ordering ───────────────────────────────────────────────────────
#
# THE ROOT OPERATOR IS CALLED DIRECTLY, not spelled as an operator, and that
# is the whole of what these twelve exist to get right. `not (self < other)`
# RAISES when the class's own `__lt__` answers NotImplemented and nothing
# else claims the pair -- so a synthesised `__gt__` written that way raises
# from inside, and that error escapes before the comparison the PROGRAM
# wrote can word its own refusal. `1 < Num(2)` reported `'<' not supported
# between instances of 'Num' and 'int'`: the right operator name, the wrong
# operands, and both belonging to a call the program never made.
#
# `type(self).__lt__(self, other)` REACHES THE METHOD AND NOT THE OPERATOR,
# so NotImplemented comes back as a value and is handed straight on. That is
# what CPython's `functools` does, and each body below is its arithmetic.


def _gt_from_lt(self, other):
    """`a > b` from `not a < b and a != b`."""
    got = type(self).__lt__(self, other)
    if got is NotImplemented:
        return got
    return not got and self != other


def _le_from_lt(self, other):
    """`a <= b` from `a < b or a == b`."""
    got = type(self).__lt__(self, other)
    if got is NotImplemented:
        return got
    return got or self == other


def _ge_from_lt(self, other):
    """`a >= b` from `not a < b`."""
    got = type(self).__lt__(self, other)
    if got is NotImplemented:
        return got
    return not got


def _lt_from_le(self, other):
    """`a < b` from `a <= b and a != b`."""
    got = type(self).__le__(self, other)
    if got is NotImplemented:
        return got
    return got and self != other


def _gt_from_le(self, other):
    """`a > b` from `not a <= b`."""
    got = type(self).__le__(self, other)
    if got is NotImplemented:
        return got
    return not got


def _ge_from_le(self, other):
    """`a >= b` from `not a <= b or a == b`."""
    got = type(self).__le__(self, other)
    if got is NotImplemented:
        return got
    return not got or self == other


def _lt_from_gt(self, other):
    """`a < b` from `not a > b and a != b`."""
    got = type(self).__gt__(self, other)
    if got is NotImplemented:
        return got
    return not got and self != other


def _ge_from_gt(self, other):
    """`a >= b` from `a > b or a == b`."""
    got = type(self).__gt__(self, other)
    if got is NotImplemented:
        return got
    return got or self == other


def _le_from_gt(self, other):
    """`a <= b` from `not a > b`."""
    got = type(self).__gt__(self, other)
    if got is NotImplemented:
        return got
    return not got


def _lt_from_ge(self, other):
    """`a < b` from `not a >= b`."""
    got = type(self).__ge__(self, other)
    if got is NotImplemented:
        return got
    return not got


def _gt_from_ge(self, other):
    """`a > b` from `a >= b and a != b`."""
    got = type(self).__ge__(self, other)
    if got is NotImplemented:
        return got
    return got and self != other


def _le_from_ge(self, other):
    """`a <= b` from `not a >= b or a == b`."""
    got = type(self).__ge__(self, other)
    if got is NotImplemented:
        return got
    return not got or self == other


def total_ordering(cls):
    """Fill in the ordering operators a class did not write.

    ONE ROOT AND THREE DERIVED, and which one is the root is decided by what
    the class actually defined: `__lt__` for preference, then `__le__`, then
    `__gt__`, then `__ge__` -- CPython's own order. A class that defined NONE
    of them has nothing to derive from and is refused by name; this used to
    assume `__lt__` and write all four regardless, so a class that wrote only
    `__gt__` got a `__gt__` back that recursed through a `__lt__` it never
    had.

    ONLY THE MISSING ONES ARE FILLED. An operator the class wrote is its own
    and is left alone, which is the other half of the same rule.

    NO `__ne__`. CPython's `total_ordering` does not touch it and neither
    does this: Python derives `!=` from `__eq__` on its own, and writing one
    here shadowed that for no gain.
    """
    # AGAINST `object`'s OWN, not `hasattr`. Every class inherits all four
    # orderings from `object` -- `hasattr(C, "__gt__")` is True in CPython
    # for a class that wrote none -- so the question is not whether the name
    # RESOLVES but whether it resolves to something other than the default.
    # CPython's own `total_ordering` is written exactly this way, and this
    # said `hasattr`: it worked only while a class inherited nothing, and the
    # moment it did, every class looked as though it had written all four and
    # none was ever filled in.
    has_lt = getattr(cls, "__lt__", None) is not getattr(object, "__lt__", None)
    has_le = getattr(cls, "__le__", None) is not getattr(object, "__le__", None)
    has_gt = getattr(cls, "__gt__", None) is not getattr(object, "__gt__", None)
    has_ge = getattr(cls, "__ge__", None) is not getattr(object, "__ge__", None)
    if not has_lt and not has_le and not has_gt and not has_ge:
        raise ValueError(
            "must define at least one ordering operation: < > <= >=")
    if has_lt:
        if not has_gt:
            cls.__gt__ = _gt_from_lt
        if not has_le:
            cls.__le__ = _le_from_lt
        if not has_ge:
            cls.__ge__ = _ge_from_lt
    elif has_le:
        if not has_ge:
            cls.__ge__ = _ge_from_le
        if not has_lt:
            cls.__lt__ = _lt_from_le
        if not has_gt:
            cls.__gt__ = _gt_from_le
    elif has_gt:
        if not has_lt:
            cls.__lt__ = _lt_from_gt
        if not has_ge:
            cls.__ge__ = _ge_from_gt
        if not has_le:
            cls.__le__ = _le_from_gt
    else:
        if not has_le:
            cls.__le__ = _le_from_ge
        if not has_gt:
            cls.__gt__ = _gt_from_ge
        if not has_lt:
            cls.__lt__ = _lt_from_ge
    return cls


def partial(fn, *bound, **held):
    """`partial(f, 1)` -- a callable with some arguments already supplied.

    A CLOSURE rather than a class, which is the whole implementation: the
    attributes CPython puts on the object (`func`, `args`, `keywords`) are set
    on the returned function, so a program that reads them still can.
    """
    def applied(*rest, **more):
        merged = dict(held)
        merged.update(more)
        return fn(*(bound + rest), **merged)

    applied.func = fn
    applied.args = bound
    applied.keywords = held
    return applied


class cached_property:
    """A property computed ONCE and then stored on the instance.

    A descriptor with `__get__` and no `__set__`: that is what makes the
    instance's own attribute win every later read, so the second access never
    reaches this at all and the value costs nothing to fetch.
    """

    def __init__(self, func):
        self.func = func
        self.attrname = "_cached_" + func.__name__
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name):
        # PEP 487: the descriptor is told the name it was bound to, which is
        # the only way it can know where to cache -- the expression that built
        # it had no idea what it was about to be assigned to.
        self.attrname = "_cached_" + name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if not hasattr(instance, self.attrname):
            setattr(instance, self.attrname, self.func(instance))
        return getattr(instance, self.attrname)


class _CacheInfo:
    def __init__(self, hits, misses, maxsize, currsize):
        self.hits = hits
        self.misses = misses
        self.maxsize = maxsize
        self.currsize = currsize

    def __repr__(self):
        return "CacheInfo(hits=" + str(self.hits) + ", misses=" \
               + str(self.misses) + ", maxsize=" + repr(self.maxsize) \
               + ", currsize=" + str(self.currsize) + ")"

    def __eq__(self, other):
        return (self.hits, self.misses, self.maxsize, self.currsize) == other


def lru_cache(maxsize=128, typed=False):
    """Remember what a function answered, and answer again without calling it.

    LEAST RECENTLY USED IS WHAT GOES when the cache is full -- so a key that
    keeps being asked for stays, and the eviction order is a property of the
    reads rather than of the writes. `maxsize=None` never evicts.

    Written to accept BOTH spellings CPython does: `@lru_cache` bare, where
    the decorated function arrives as `maxsize`, and `@lru_cache(maxsize=2)`.
    """
    if callable(maxsize):
        return _lru_wrap(maxsize, 128)
    return lambda fn: _lru_wrap(fn, maxsize)


def _lru_wrap(fn, maxsize):
    state = {"hits": 0, "misses": 0}
    cache = {}
    order = []

    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items()))) if kwargs else args
        if key in cache:
            state["hits"] = state["hits"] + 1
            order.remove(key)
            order.append(key)
            return cache[key]
        state["misses"] = state["misses"] + 1
        made = fn(*args, **kwargs)
        cache[key] = made
        order.append(key)
        if maxsize is not None and len(order) > maxsize:
            del cache[order[0]]
            del order[0]
        return made

    def cache_info():
        return _CacheInfo(state["hits"], state["misses"], maxsize, len(cache))

    def cache_clear():
        cache.clear()
        del order[:]
        state["hits"] = 0
        state["misses"] = 0

    wrapper.cache_info = cache_info
    wrapper.cache_clear = cache_clear
    wrapper.__wrapped__ = fn
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def cache(fn):
    """`lru_cache(maxsize=None)` -- unbounded, so nothing is ever evicted."""
    return _lru_wrap(fn, None)


def singledispatch(fn):
    """A function whose body is chosen by the TYPE of its first argument.

    The default is the function this decorates; every `register` adds a more
    specific one. Dispatch tries an EXACT type match first and falls back to
    `isinstance` in registration order, which is what makes `describe(True)`
    reach the `int` implementation -- a bool is an int, and no one registered
    a bool.
    """
    exact = {}
    order = []

    def dispatch(kind):
        if kind in exact:
            return exact[kind]
        return fn

    def register(first, second=None):
        # BOTH SPELLINGS. `@f.register` reads the type off the annotation of
        # the function it decorates; `@f.register(int)` is given it.
        if second is None and not isinstance(first, type):
            hints = getattr(first, "__annotations__", {})
            kind = None
            for key in hints:
                if key != "return":
                    kind = hints[key]
                    break
            if kind is None:
                raise TypeError("Invalid first argument to register(): "
                                "no type annotation found")
            exact[kind] = first
            order.append((kind, first))
            return first
        if second is not None:
            exact[first] = second
            order.append((first, second))
            return second

        def keep(impl):
            exact[first] = impl
            order.append((first, impl))
            return impl
        return keep

    def wrapper(*args, **kwargs):
        if args:
            found = exact.get(type(args[0]))
            if found is None:
                for pair in order:
                    if isinstance(args[0], pair[0]):
                        found = pair[1]
                        break
            if found is not None:
                return found(*args, **kwargs)
        return fn(*args, **kwargs)

    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = exact
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper
