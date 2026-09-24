"""`pprint`, as ordinary Python this compiler compiles.

COVERAGE: `pprint`, `pformat`, `pp`, `saferepr`, `isreadable`,
`isrecursive`, and `PrettyPrinter` with every constructor parameter --
`indent`, `width`, `depth`, `stream`, `compact`, `sort_dicts`,
`underscore_numbers` -- and its `pprint`, `pformat`, `isreadable`,
`isrecursive` and `format` methods. Laid out as CPython lays them out:
dicts (sorted, or in insertion order), lists, tuples, sets and frozensets,
strings split at word boundaries into implicitly concatenated pieces, bytes
and bytearrays split every four bytes, dataclasses whose `__repr__` was
generated, `SimpleNamespace`, and `collections`' `OrderedDict`,
`defaultdict`, `Counter`, `ChainMap`, `deque`, `UserDict`, `UserList` and
`UserString`; recursion, `depth`, and the readable/recursive flags.

NOT COVERED: `types.MappingProxyType`, which is not bundled, so a
`mappingproxy` is printed by its repr on one line. Registering a printer by
writing into `PrettyPrinter._dispatch` -- a private table CPython keys by
`__repr__` function -- does nothing here.

## How a printer is chosen

CPython asks `type(obj).__repr__` and looks that FUNCTION up in a table, so
a `dict` subclass that inherits `dict`'s repr is laid out as a dict and one
that writes its own `__repr__` is printed by it. The question is really
"which class does this object's repr come from", and that is how it is
asked here: the first class in the object's MRO that is either a class this
module lays out or one that defines `__repr__` itself decides. Asking by
function identity would not work -- `dict.__repr__ is dict.__repr__` is
False under this compiler -- and asking by class gives the same answer for
every object CPython's table can see.

A BUILTIN BASE MAY BE MISSING FROM THE MRO here (`class D(dict)` lists `D`
and `object`), so after the walk an `isinstance` check stands in for it.
"""

import collections as _collections
import sys as _sys
import types as _types
from io import StringIO as _StringIO

__all__ = ["pprint", "pformat", "isreadable", "isrecursive", "saferepr",
           "PrettyPrinter", "pp"]


def pprint(object, stream=None, indent=1, width=80, depth=None, *,
           compact=False, sort_dicts=True, underscore_numbers=False):
    """Pretty-print a Python object to a stream [default is sys.stdout]."""
    printer = PrettyPrinter(stream=stream, indent=indent, width=width,
                            depth=depth, compact=compact,
                            sort_dicts=sort_dicts,
                            underscore_numbers=underscore_numbers)
    printer.pprint(object)


def pformat(object, indent=1, width=80, depth=None, *, compact=False,
            sort_dicts=True, underscore_numbers=False):
    """Format a Python object into a pretty-printed representation."""
    return PrettyPrinter(indent=indent, width=width, depth=depth,
                         compact=compact, sort_dicts=sort_dicts,
                         underscore_numbers=underscore_numbers
                         ).pformat(object)


def pp(object, *args, sort_dicts=False, **kwargs):
    """Pretty-print a Python object -- in insertion order by default."""
    pprint(object, *args, sort_dicts=sort_dicts, **kwargs)


def saferepr(object):
    """Version of repr() which can handle recursive data structures."""
    return PrettyPrinter()._safe_repr(object, {}, None, 0)[0]


def isreadable(object):
    """Determine if saferepr(object) is readable by eval()."""
    return PrettyPrinter()._safe_repr(object, {}, None, 0)[1]


def isrecursive(object):
    """Determine if object requires a recursive representation."""
    return PrettyPrinter()._safe_repr(object, {}, None, 0)[2]


class _safe_key:
    """A sort key that orders values `<` cannot: by the text of their type,
    then by identity. Values that do compare, compare as themselves."""

    __slots__ = ['obj']

    def __init__(self, obj):
        self.obj = obj

    def __lt__(self, other):
        try:
            return self.obj < other.obj
        except TypeError:
            return ((str(type(self.obj)), id(self.obj))
                    < (str(type(other.obj)), id(other.obj)))


def _safe_tuple(t):
    """A sort key for a (key, value) pair."""
    return _safe_key(t[0]), _safe_key(t[1])


#: What `repr` alone answers: no layout, always readable, never recursive.
_builtin_scalars = frozenset({str, bytes, bytearray, float, complex, bool,
                              type(None)})

#: The builtins an MRO here may leave out -- see the module docstring -- in
#: the order they are tried. `bool` is an int, and is a scalar long before it
#: gets here.
_BUILTIN_BASES = (dict, list, tuple, set, frozenset, str, bytes, bytearray,
                  int)


def _repr_owner(value):
    """The class whose `__repr__` this value's repr comes from, when it is
    one this module knows; None when it is a class of the program's own.

    `value` AND NOT `object`, which is what every other function here calls
    its argument: this one has to compare against the builtin.

    A BUILTIN TYPE HAS NO `__mro__` under this compiler, so one is its own:
    for `dict` and `list` that is the whole of the answer anyway."""
    cls = type(value)
    mro = getattr(cls, "__mro__", None)
    if mro is None:
        mro = (cls,)
    for base in mro:
        if base in _KNOWN:
            return base
        if base is object:
            break
        if "__repr__" in vars(base):
            return None
    for base in _BUILTIN_BASES:
        if isinstance(value, base):
            return base
    return None


def _recursion(object):
    return ("<Recursion on %s with id=%s>"
            % (type(object).__name__, id(object)))


def _split_words(line):
    """`re.findall(r'\\S*\\s*', line)` without its final empty match: the
    line as runs of non-space each followed by the space after it."""
    parts = []
    i = 0
    n = len(line)
    while i < n:
        start = i
        while i < n and not line[i].isspace():
            i += 1
        while i < n and line[i].isspace():
            i += 1
        parts.append(line[start:i])
    return parts


def _wrap_bytes_repr(object, width, allowance):
    """The reprs of `object`'s pieces, each as long as fits, split only at
    multiples of four bytes."""
    current = b''
    last = len(object) // 4 * 4
    for i in range(0, len(object), 4):
        part = object[i:i + 4]
        candidate = current + part
        if i == last:
            width -= allowance
        if len(repr(candidate)) > width:
            if current:
                yield repr(current)
            current = part
        else:
            current = candidate
    if current:
        yield repr(current)


class PrettyPrinter:
    def __init__(self, indent=1, width=80, depth=None, stream=None, *,
                 compact=False, sort_dicts=True, underscore_numbers=False):
        """Handle pretty printing operations onto a stream using a set of
        configured parameters."""
        indent = int(indent)
        width = int(width)
        if indent < 0:
            raise ValueError('indent must be >= 0')
        if depth is not None and depth <= 0:
            raise ValueError('depth must be > 0')
        if not width:
            raise ValueError('width must be != 0')
        self._depth = depth
        self._indent_per_level = indent
        self._width = width
        # THE STREAM AS IT IS NOW, when none is given: a program that swaps
        # `sys.stdout` later keeps printing where it was when this was made.
        if stream is not None:
            self._stream = stream
        else:
            self._stream = _sys.stdout
        self._compact = bool(compact)
        self._sort_dicts = sort_dicts
        self._underscore_numbers = underscore_numbers

    def pprint(self, object):
        if self._stream is not None:
            self._format(object, self._stream, 0, 0, {}, 0)
            self._stream.write("\n")

    def pformat(self, object):
        sio = _StringIO()
        self._format(object, sio, 0, 0, {}, 0)
        return sio.getvalue()

    def isrecursive(self, object):
        return self.format(object, {}, 0, 0)[2]

    def isreadable(self, object):
        s, readable, recursive = self.format(object, {}, 0, 0)
        return readable and not recursive

    def _format(self, object, stream, indent, allowance, context, level):
        """Write `object`: on one line when its repr fits in what is left of
        the width, and laid out by its printer when it does not.

        `allowance` IS WHAT MUST FOLLOW ON THE SAME LINE -- the closing
        brackets and the comma of every level still open -- so the last
        element of a nested structure is measured with room for them."""
        objid = id(object)
        if objid in context:
            stream.write(_recursion(object))
            self._recursive = True
            self._readable = False
            return
        rep = self._repr(object, context, level)
        max_width = self._width - indent - allowance
        if len(rep) > max_width:
            owner = _repr_owner(object)
            printer = self._dispatch.get(owner) if owner is not None else None
            if printer is not None:
                context[objid] = 1
                printer(self, object, stream, indent, allowance, context,
                        level + 1)
                del context[objid]
                return
            if self._generated_dataclass_repr(object):
                context[objid] = 1
                self._pprint_dataclass(object, stream, indent, allowance,
                                       context, level + 1)
                del context[objid]
                return
        stream.write(rep)

    def _generated_dataclass_repr(self, object):
        """Is this a dataclass INSTANCE whose `__repr__` is the one
        `@dataclass` generated? A written one is the program's own layout,
        and is used as written."""
        from dataclasses import is_dataclass
        if not is_dataclass(object) or isinstance(object, type):
            return False
        if not object.__dataclass_params__.repr:
            return False
        method = object.__repr__
        if not hasattr(method, "__wrapped__"):
            return False
        return "__create_fn__" in method.__wrapped__.__qualname__

    def _pprint_dataclass(self, object, stream, indent, allowance, context,
                          level):
        from dataclasses import fields as dataclass_fields
        cls_name = object.__class__.__name__
        indent += len(cls_name) + 1
        items = [(f.name, getattr(object, f.name))
                 for f in dataclass_fields(object) if f.repr]
        stream.write(cls_name + '(')
        self._format_namespace_items(items, stream, indent, allowance,
                                     context, level)
        stream.write(')')

    def _pprint_dict(self, object, stream, indent, allowance, context,
                     level):
        write = stream.write
        write('{')
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * ' ')
        if len(object):
            if self._sort_dicts:
                items = sorted(object.items(), key=_safe_tuple)
            else:
                items = object.items()
            self._format_dict_items(items, stream, indent, allowance + 1,
                                    context, level)
        write('}')

    def _pprint_ordered_dict(self, object, stream, indent, allowance,
                             context, level):
        if not len(object):
            stream.write(repr(object))
            return
        cls = object.__class__
        stream.write(cls.__name__ + '(')
        self._format(list(object.items()), stream,
                     indent + len(cls.__name__) + 1, allowance + 1,
                     context, level)
        stream.write(')')

    def _pprint_list(self, object, stream, indent, allowance, context,
                     level):
        stream.write('[')
        self._format_items(object, stream, indent, allowance + 1, context,
                           level)
        stream.write(']')

    def _pprint_tuple(self, object, stream, indent, allowance, context,
                      level):
        stream.write('(')
        endchar = ',)' if len(object) == 1 else ')'
        self._format_items(object, stream, indent,
                           allowance + len(endchar), context, level)
        stream.write(endchar)

    def _pprint_set(self, object, stream, indent, allowance, context,
                    level):
        if not len(object):
            stream.write(repr(object))
            return
        typ = object.__class__
        if typ is set:
            stream.write('{')
            endchar = '}'
        else:
            stream.write(typ.__name__ + '({')
            endchar = '})'
            indent += len(typ.__name__) + 1
        object = sorted(object, key=_safe_key)
        self._format_items(object, stream, indent,
                           allowance + len(endchar), context, level)
        stream.write(endchar)

    def _pprint_str(self, object, stream, indent, allowance, context,
                    level):
        """A long string as the implicitly concatenated pieces it can be
        written as: split at its own line ends first, then, for a line that
        is still too long, between words -- each piece a repr that fits. At
        the top level the pieces are wrapped in parentheses, which is what
        makes the output a valid expression."""
        write = stream.write
        if not len(object):
            write(repr(object))
            return
        chunks = []
        lines = object.splitlines(True)
        if level == 1:
            indent += 1
            allowance += 1
        max_width1 = max_width = self._width - indent
        rep = ''
        for i, line in enumerate(lines):
            rep = repr(line)
            if i == len(lines) - 1:
                max_width1 -= allowance
            if len(rep) <= max_width1:
                chunks.append(rep)
            else:
                parts = _split_words(line)
                max_width2 = max_width
                current = ''
                for j, part in enumerate(parts):
                    candidate = current + part
                    if j == len(parts) - 1 and i == len(lines) - 1:
                        max_width2 -= allowance
                    if len(repr(candidate)) > max_width2:
                        if current:
                            chunks.append(repr(current))
                        current = part
                    else:
                        current = candidate
                if current:
                    chunks.append(repr(current))
        if len(chunks) == 1:
            write(rep)
            return
        if level == 1:
            write('(')
        for i, rep in enumerate(chunks):
            if i > 0:
                write('\n' + ' ' * indent)
            write(rep)
        if level == 1:
            write(')')

    def _pprint_bytes(self, object, stream, indent, allowance, context,
                      level):
        write = stream.write
        if len(object) <= 4:
            write(repr(object))
            return
        parens = level == 1
        if parens:
            indent += 1
            allowance += 1
            write('(')
        delim = ''
        for rep in _wrap_bytes_repr(object, self._width - indent, allowance):
            write(delim)
            write(rep)
            if not delim:
                delim = '\n' + ' ' * indent
        if parens:
            write(')')

    def _pprint_bytearray(self, object, stream, indent, allowance, context,
                          level):
        write = stream.write
        write('bytearray(')
        self._pprint_bytes(bytes(object), stream, indent + 10,
                           allowance + 1, context, level + 1)
        write(')')

    def _pprint_simplenamespace(self, object, stream, indent, allowance,
                                context, level):
        # `namespace(...)` for the type itself, which is how its repr names
        # it; a subclass is named by its own class.
        if type(object) is _types.SimpleNamespace:
            cls_name = 'namespace'
        else:
            cls_name = object.__class__.__name__
        indent += len(cls_name) + 1
        items = object.__dict__.items()
        stream.write(cls_name + '(')
        self._format_namespace_items(items, stream, indent, allowance,
                                     context, level)
        stream.write(')')

    def _format_dict_items(self, items, stream, indent, allowance, context,
                           level):
        write = stream.write
        indent += self._indent_per_level
        delimnl = ',\n' + ' ' * indent
        items = list(items)
        last_index = len(items) - 1
        for i, (key, ent) in enumerate(items):
            last = i == last_index
            rep = self._repr(key, context, level)
            write(rep)
            write(': ')
            self._format(ent, stream, indent + len(rep) + 2,
                         allowance if last else 1, context, level)
            if not last:
                write(delimnl)

    def _format_namespace_items(self, items, stream, indent, allowance,
                                context, level):
        write = stream.write
        delimnl = ',\n' + ' ' * indent
        items = list(items)
        last_index = len(items) - 1
        for i, (key, ent) in enumerate(items):
            last = i == last_index
            write(key)
            write('=')
            # A VALUE ALREADY BEING PRINTED is `...`, the way a dataclass's
            # own recursive repr spells it, rather than the recursion text.
            if id(ent) in context:
                write("...")
            else:
                self._format(ent, stream, indent + len(key) + 1,
                             allowance if last else 1, context, level)
            if not last:
                write(delimnl)

    def _format_items(self, items, stream, indent, allowance, context,
                      level):
        """The elements of a sequence, one per line -- or, with `compact`,
        as many to a line as fit."""
        write = stream.write
        indent += self._indent_per_level
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * ' ')
        delimnl = ',\n' + ' ' * indent
        delim = ''
        width = max_width = self._width - indent + 1
        it = iter(items)
        try:
            next_ent = next(it)
        except StopIteration:
            return
        last = False
        while not last:
            ent = next_ent
            try:
                next_ent = next(it)
            except StopIteration:
                last = True
                max_width -= allowance
                width -= allowance
            if self._compact:
                rep = self._repr(ent, context, level)
                w = len(rep) + 2
                if width < w:
                    width = max_width
                    if delim:
                        delim = delimnl
                if width >= w:
                    width -= w
                    write(delim)
                    delim = ', '
                    write(rep)
                    continue
            write(delim)
            delim = delimnl
            self._format(ent, stream, indent, allowance if last else 1,
                         context, level)

    def _repr(self, object, context, level):
        repr, readable, recursive = self.format(object, context.copy(),
                                                self._depth, level)
        if not readable:
            self._readable = False
        if recursive:
            self._recursive = True
        return repr

    def format(self, object, context, maxlevels, level):
        """Format object for a specific context, returning a string
        and flags indicating whether the representation is 'readable'
        and whether the object represents a recursive construct.
        """
        return self._safe_repr(object, context, maxlevels, level)

    def _pprint_default_dict(self, object, stream, indent, allowance,
                             context, level):
        if not len(object):
            stream.write(repr(object))
            return
        rdf = self._repr(object.default_factory, context, level)
        cls = object.__class__
        indent += len(cls.__name__) + 1
        stream.write('%s(%s,\n%s' % (cls.__name__, rdf, ' ' * indent))
        self._pprint_dict(object, stream, indent, allowance + 1, context,
                          level)
        stream.write(')')

    def _pprint_counter(self, object, stream, indent, allowance, context,
                        level):
        if not len(object):
            stream.write(repr(object))
            return
        cls = object.__class__
        stream.write(cls.__name__ + '({')
        if self._indent_per_level > 1:
            stream.write((self._indent_per_level - 1) * ' ')
        items = object.most_common()
        self._format_dict_items(items, stream,
                                indent + len(cls.__name__) + 1,
                                allowance + 2, context, level)
        stream.write('})')

    def _pprint_chain_map(self, object, stream, indent, allowance, context,
                          level):
        if not len(object.maps):
            stream.write(repr(object))
            return
        cls = object.__class__
        stream.write(cls.__name__ + '(')
        indent += len(cls.__name__) + 1
        for i, m in enumerate(object.maps):
            if i == len(object.maps) - 1:
                self._format(m, stream, indent, allowance + 1, context,
                             level)
                stream.write(')')
            else:
                self._format(m, stream, indent, 1, context, level)
                stream.write(',\n' + ' ' * indent)

    def _pprint_deque(self, object, stream, indent, allowance, context,
                      level):
        if not len(object):
            stream.write(repr(object))
            return
        cls = object.__class__
        stream.write(cls.__name__ + '(')
        indent += len(cls.__name__) + 1
        stream.write('[')
        if object.maxlen is None:
            self._format_items(object, stream, indent, allowance + 2,
                               context, level)
            stream.write('])')
        else:
            self._format_items(object, stream, indent, 2, context, level)
            rml = self._repr(object.maxlen, context, level)
            stream.write('],\n%smaxlen=%s)' % (' ' * indent, rml))

    def _pprint_user_dict(self, object, stream, indent, allowance, context,
                          level):
        self._format(object.data, stream, indent, allowance, context,
                     level - 1)

    def _pprint_user_list(self, object, stream, indent, allowance, context,
                          level):
        self._format(object.data, stream, indent, allowance, context,
                     level - 1)

    def _pprint_user_string(self, object, stream, indent, allowance,
                            context, level):
        self._format(object.data, stream, indent, allowance, context,
                     level - 1)

    def _safe_repr(self, object, context, maxlevels, level):
        """(repr, readable, recursive) -- the one-line text, whether `eval`
        could read it back, and whether it had to break a cycle."""
        typ = type(object)
        if typ in _builtin_scalars:
            return repr(object), True, False

        owner = _repr_owner(object)
        if owner is int:
            if self._underscore_numbers:
                return f"{object:_d}", True, False
            return repr(object), True, False

        if owner is dict:
            if not object:
                return "{}", True, False
            objid = id(object)
            if maxlevels and level >= maxlevels:
                return "{...}", False, objid in context
            if objid in context:
                return _recursion(object), False, True
            context[objid] = 1
            readable = True
            recursive = False
            components = []
            level += 1
            if self._sort_dicts:
                items = sorted(object.items(), key=_safe_tuple)
            else:
                items = object.items()
            for k, v in items:
                krepr, kreadable, krecur = self.format(k, context,
                                                       maxlevels, level)
                vrepr, vreadable, vrecur = self.format(v, context,
                                                       maxlevels, level)
                components.append("%s: %s" % (krepr, vrepr))
                readable = readable and kreadable and vreadable
                if krecur or vrecur:
                    recursive = True
            del context[objid]
            return "{%s}" % ", ".join(components), readable, recursive

        if owner is list or owner is tuple:
            if owner is list:
                if not object:
                    return "[]", True, False
                format = "[%s]"
            elif len(object) == 1:
                format = "(%s,)"
            else:
                if not object:
                    return "()", True, False
                format = "(%s)"
            objid = id(object)
            if maxlevels and level >= maxlevels:
                return format % "...", False, objid in context
            if objid in context:
                return _recursion(object), False, True
            context[objid] = 1
            readable = True
            recursive = False
            components = []
            level += 1
            for o in object:
                orepr, oreadable, orecur = self.format(o, context,
                                                       maxlevels, level)
                components.append(orepr)
                if not oreadable:
                    readable = False
                if orecur:
                    recursive = True
            del context[objid]
            return format % ", ".join(components), readable, recursive

        rep = repr(object)
        return rep, (rep and not rep.startswith('<')), False


#: The printers, by the class a repr comes from. See the module docstring.
PrettyPrinter._dispatch = {
    dict: PrettyPrinter._pprint_dict,
    _collections.OrderedDict: PrettyPrinter._pprint_ordered_dict,
    list: PrettyPrinter._pprint_list,
    tuple: PrettyPrinter._pprint_tuple,
    set: PrettyPrinter._pprint_set,
    frozenset: PrettyPrinter._pprint_set,
    str: PrettyPrinter._pprint_str,
    bytes: PrettyPrinter._pprint_bytes,
    bytearray: PrettyPrinter._pprint_bytearray,
    _types.SimpleNamespace: PrettyPrinter._pprint_simplenamespace,
    _collections.defaultdict: PrettyPrinter._pprint_default_dict,
    _collections.Counter: PrettyPrinter._pprint_counter,
    _collections.ChainMap: PrettyPrinter._pprint_chain_map,
    _collections.deque: PrettyPrinter._pprint_deque,
    _collections.UserDict: PrettyPrinter._pprint_user_dict,
    _collections.UserList: PrettyPrinter._pprint_user_list,
    _collections.UserString: PrettyPrinter._pprint_user_string,
}

#: Every class whose repr this module knows: the printers' classes, and
#: `int`, whose repr `underscore_numbers` rewrites.
_KNOWN = frozenset(list(PrettyPrinter._dispatch) + [int])
