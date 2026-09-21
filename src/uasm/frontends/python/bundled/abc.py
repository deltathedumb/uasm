"""Abstract base classes.

COVERAGE: `ABCMeta`, `ABC`, `abstractmethod`, `abstractproperty`,
`abstractclassmethod`, `abstractstaticmethod`, `register`,
`__abstractmethods__`, `__subclasshook__`, `update_abstractmethods`,
`get_cache_token`, and `@property`/`@classmethod` stacked over
`@abstractmethod`.

NOT COVERED: the negative cache CPython keeps per class. It is invisible
except through `get_cache_token`, which is here and answers a counter that
moves when a registration happens -- the one thing a caller can DO with the
token is compare two of them, and that comparison is right.

THE THREE DEPRECATED DECORATORS ARE FUNCTIONS, not subclasses of `property`,
`classmethod` and `staticmethod` as in CPython -- this frontend cannot extend
those three. What a caller can observe is identical; see `abstractproperty`.

`ABCMeta` IS A METACLASS AND NOTHING MORE. It collects the names a body marked
abstract and refuses to instantiate a class that still has any; `register` and
`__subclasshook__` decide what counts as a subclass without inheritance. All
of it is ordinary Python -- `type.__new__` through `super()`, and a `__call__`
that decides what calling the class does -- which is why this is written here
rather than in the runtime.

TWO WAYS TO BE A SUBCLASS, and `collections.abc` needs both. REGISTRATION is a
claim made from outside: `Sequence.register(tuple)` says so, nothing about
`tuple` changes, and only `isinstance`/`issubclass` consult the claim.
`__subclasshook__` is STRUCTURAL: the ABC inspects the candidate and decides,
which is how `isinstance(SomeUserClass(), Iterable)` answers True for a class
that never heard of `Iterable` and only wrote `__iter__`.
"""

#: Bumped whenever a registration changes what `issubclass` would answer.
#: CPython's token is an opaque object identity; a counter compares the same
#: way and is the only thing a caller can do with one.
_token = [0]


def get_cache_token():
    """A value that CHANGES when the ABC hierarchy does.

    A caller caches an `issubclass` answer alongside this and re-asks when the
    two stop matching. That is the whole contract -- the value is otherwise
    opaque, and CPython documents it as such.
    """
    return _token[0]


def abstractmethod(funcobj):
    """Mark a method as one a concrete subclass must define.

    THE MARK IS AN ATTRIBUTE ON THE FUNCTION, which is where `ABCMeta` looks
    for it -- exactly as in CPython, so a decorator stack that puts
    `@classmethod` outside it still works as long as the wrapper forwards the
    flag.
    """
    funcobj.__isabstractmethod__ = True
    return funcobj


def abstractproperty(func):
    """`@abstractproperty`, deprecated in CPython since 3.3 and still there.

    A FUNCTION HERE, A CLASS THERE. CPython spells it as a subclass of
    `property` carrying the flag; this frontend cannot subclass `property`,
    and the flag can live on the underlying function instead -- which is
    where `_is_abstract` looks anyway, because it has to look there for the
    MODERN spelling too. The observable behaviour is the same: the name is
    a property, and the class will not instantiate until it is overridden.
    """
    func.__isabstractmethod__ = True
    return property(func)


def abstractclassmethod(func):
    """`@abstractclassmethod`. See `abstractproperty` for why it is a
    function; prefer `@classmethod` over `@abstractmethod`, which is what
    CPython has recommended since 3.3 and what `_is_abstract` unwraps."""
    func.__isabstractmethod__ = True
    return classmethod(func)


def abstractstaticmethod(func):
    """`@abstractstaticmethod`. See `abstractclassmethod`."""
    func.__isabstractmethod__ = True
    return staticmethod(func)


def _is_abstract(value):
    """Whether a class body bound something still waiting for an override.

    THE FLAG MAY BE ONE LEVEL DOWN, and that is the ordinary case rather than
    a corner. The spelling CPython recommends is `@property` over
    `@abstractmethod`, or `@classmethod` over it -- so what the body BOUND is
    a descriptor, and the mark is on the function inside it. Reading only the
    outer object made every such name concrete, and a class that had not
    overridden it instantiated cleanly.
    """
    if getattr(value, "__isabstractmethod__", False):
        return True
    for attr in ("fget", "__func__"):
        inner = getattr(value, attr, None)
        if inner is not None and getattr(inner, "__isabstractmethod__", False):
            return True
    return False


def _collect(cls, bases, namespace):
    """The names still abstract on a freshly built class.

    TWO SOURCES, and the second is the one that is easy to get wrong. A name
    this body marked is abstract. A name a BASE left abstract is still
    abstract unless something concrete has been bound over it -- and that is
    checked by looking the name up ON THE CLASS, not in this namespace, so an
    override two levels up counts.
    """
    out = []
    for key in namespace:
        if _is_abstract(namespace[key]):
            out.append(key)
    for base in bases:
        for key in getattr(base, "__abstractmethods__", ()):
            if key in out:
                continue
            if _is_abstract(getattr(cls, key, None)):
                out.append(key)
    return out


def update_abstractmethods(cls):
    """Recompute `__abstractmethods__` after the class body has been changed.

    FOR A DECORATOR THAT ADDS METHODS. `dataclass` is the motivating one: it
    binds `__init__` and friends AFTER `ABCMeta.__new__` has already counted,
    so a class whose only abstract name the decorator filled would still
    refuse to instantiate. CPython added this in 3.10 for exactly that.
    """
    if not hasattr(cls, "__abstractmethods__"):
        return cls
    out = []
    for key in getattr(cls, "__abstractmethods__", ()):
        if _is_abstract(getattr(cls, key, None)):
            out.append(key)
    for base in getattr(cls, "__mro__", ()):
        for key in getattr(base, "__abstractmethods__", ()):
            if key not in out and _is_abstract(getattr(cls, key, None)):
                out.append(key)
    cls.__abstractmethods__ = frozenset(out)
    return cls


class ABCMeta(type):
    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        cls.__abstractmethods__ = frozenset(_collect(cls, bases, namespace))
        # ONE LIST PER CLASS, made here rather than as a class attribute of
        # `ABCMeta`: a single shared list would make every ABC in the program
        # answer for every registration any of them accepted.
        cls._abc_registry = []
        # AND ONE LIST OF ITS CHILDREN, for the same reason and kept by hand:
        # `type.__subclasses__` is what CPython walks to inherit a
        # registration DOWNWARD, and this compiler does not have it yet.
        cls._abc_children = []
        for base in bases:
            children = getattr(base, "_abc_children", None)
            if children is not None:
                children.append(cls)
        return cls

    def register(cls, subclass):
        """Declare `subclass` a subclass of `cls` WITHOUT INHERITANCE.

        Nothing about `subclass` changes -- its `__mro__` does not mention
        `cls` -- so the claim lives here, and only `isinstance` and
        `issubclass` are taught to consult it.

        RETURNS THE ARGUMENT, so it works as a decorator.
        """
        cls._abc_registry.append(subclass)
        _token[0] = _token[0] + 1
        return subclass

    def __subclasscheck__(cls, subclass):
        # THE REAL HIERARCHY FIRST, walked by hand: asking `issubclass` here
        # would come straight back to this hook and never stop.
        for entry in getattr(subclass, "__mro__", ()):
            if entry is cls:
                return True
        # THE STRUCTURAL ANSWER SECOND. `__subclasshook__` may say True, False
        # or NotImplemented, and only the last means "carry on asking" --
        # a hook answering False is a REFUSAL and must not be overridden by
        # the registry, which is what lets `Hashable` reject a class that set
        # `__hash__ = None`.
        hook = getattr(cls, "__subclasshook__", None)
        if hook is not None:
            got = hook(subclass)
            if got is not NotImplemented:
                return bool(got)
        for entry in cls._abc_registry:
            if subclass is entry:
                return True
            # A SUBCLASS OF A REGISTERED CLASS COUNTS TOO -- registration is
            # about the whole subtree, not the one class named.
            for walk in getattr(subclass, "__mro__", ()):
                if walk is entry:
                    return True
        # A REGISTRATION IS INHERITED DOWNWARD, so the classes to ask are
        # `cls`'s CHILDREN and not its bases. `MutableMapping` is a
        # `Mapping`, so a class registered with `MutableMapping` answers True
        # for `Mapping` -- and the registry is per class, so `Mapping` has to
        # ask the classes BELOW it. CPython walks `cls.__subclasses__()` here
        # and this walks the list `__new__` keeps, which is the same set.
        #
        # THE OTHER DIRECTION IS WRONG AND WAS HERE, dead, until `isinstance`
        # learned that a class is an instance of its metaclass: asking the
        # BASES made `issubclass(float, Rational)` True, because `float` is
        # registered with `Real` and `Real` is a base of `Rational`. That is
        # the relation upside down -- a Real is not a Rational.
        for child in cls._abc_children:
            if child.__subclasscheck__(subclass):
                return True
        return False

    def __instancecheck__(cls, instance):
        return cls.__subclasscheck__(type(instance))

    def __call__(cls, *args, **kwargs):
        missing = cls.__abstractmethods__
        if missing:
            names = sorted(missing)
            listed = ""
            for name in names:
                if listed:
                    listed = listed + ", "
                listed = listed + "'" + name + "'"
            plural = "s" if len(names) != 1 else ""
            raise TypeError("Can't instantiate abstract class " + cls.__name__
                            + " without an implementation for abstract method"
                            + plural + " " + listed)
        return super().__call__(*args, **kwargs)


class ABC(metaclass=ABCMeta):
    """A base whose subclasses get `ABCMeta` without repeating `metaclass=`."""
