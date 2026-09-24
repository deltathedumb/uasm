# COVERAGE: pprint, pformat, pp, saferepr, isreadable, isrecursive, and
# PrettyPrinter with indent, width, depth, stream, compact, sort_dicts and
# underscore_numbers, and its pprint/pformat/isreadable/isrecursive/format
# methods; the layout of dicts (sorted, insertion order, unorderable keys),
# lists, tuples (the one-element comma), sets and frozensets, long strings
# (split at line ends and between words, parenthesised at the top level),
# bytes and bytearray (split every four bytes), a dataclass with a generated
# repr and one with a written repr, SimpleNamespace, OrderedDict,
# defaultdict, Counter, ChainMap, deque (with and without maxlen), UserDict,
# UserList, UserString, a dict subclass that inherits dict's repr and one
# that writes its own; recursion (the flags, not the id), depth; the
# ValueErrors.
#
# Run under CPython and under uasm; the outputs must be identical. So
# the assertions below are written against what the module IS SPECIFIED to
# do, not against what uasm currently does.
#
# A RECURSIVE STRUCTURE IS SHOWN WITH ITS ID ELIDED: `<Recursion on list with
# id=...>` carries the address, which is the one part of the text two
# processes cannot agree on.
import collections
import io
import pprint
import re
import types
from dataclasses import dataclass, field


def attempt(label, f):
    try:
        print(label, repr(f()))
    except (TypeError, ValueError) as e:
        print(label, type(e).__name__ + ":", e)


def no_ids(text):
    return re.sub(r"id=\d+", "id=...", text)


# ---- the basics ----------------------------------------------------------
data = {"b": [1, 2, 3], "a": {"x": 1, "y": (1, 2)}, "c": "text",
        "d": {"nested": {"deeper": list(range(12))}}}
pprint.pprint(data)
pprint.pprint(data, width=30)
pprint.pp(data, width=30)
pprint.pprint(data, sort_dicts=False, width=30)
for width in (10, 25, 60):
    print(pprint.pformat(list(range(25)), width=width))
    print(pprint.pformat(list(range(25)), width=width, compact=True))
print(pprint.pformat(tuple(range(20)), width=30))
print(pprint.pformat((1,)), pprint.pformat(("only-one-long-element" * 3,),
                                           width=20))
print(pprint.pformat([[1, 2, 3], [4, 5, 6], [7, 8, 9]] * 3, width=20,
                     indent=4))
print(pprint.pformat({"key%d" % i: i for i in range(10)}, width=40,
                     indent=2))
print(pprint.pformat(set(range(15)), width=20))
print(pprint.pformat(frozenset("abcdefghij"), width=20))
print(pprint.pformat(set()), pprint.pformat(frozenset()), pprint.pformat({}),
      pprint.pformat([]), pprint.pformat(()))
print(pprint.pformat({1: "a", "b": 2, (3,): None}, width=15))

# ---- depth, numbers, readability -------------------------------------------
nested = [1, [2, [3, [4, [5]]]], {"k": {"k2": {"k3": 1}}}]
print(pprint.pformat(nested, depth=2))
print(pprint.pformat(nested, depth=3, width=20))
print(pprint.pformat([10 ** 12, -1234567, 3.5, 10 ** 6], underscore_numbers=True))
print(pprint.pformat({"n": 123456789}, underscore_numbers=True))
print(pprint.saferepr({"z": 1, "a": [1, 2]}), pprint.saferepr("x" * 5))
print(pprint.isreadable([1, "a", (2, 3)]), pprint.isreadable([object()]))
print(pprint.isreadable({"a": print}), pprint.isrecursive([1, 2]))
loop = [1, 2]
loop.append(loop)
print(no_ids(pprint.saferepr(loop)), pprint.isrecursive(loop),
      pprint.isreadable(loop))
print(no_ids(pprint.pformat(loop)))
d = {"self": None}
d["self"] = d
print(no_ids(pprint.pformat(d)), pprint.isrecursive(d))
printer = pprint.PrettyPrinter(width=30, indent=2, depth=2)
print(printer.pformat(nested), printer.isreadable(nested),
      printer.isrecursive(nested))
print(printer.format([1, 2], {}, 0, 0))

# ---- strings and bytes -----------------------------------------------------
text = ("The quick brown fox jumps over the lazy dog. " * 4).strip()
print(pprint.pformat(text, width=40))
print(pprint.pformat([text], width=40))
print(pprint.pformat({"key": text}, width=50))
print(pprint.pformat("line one\nline two is a little longer\nthree\n",
                     width=20))
print(pprint.pformat("a" * 60, width=20))
print(pprint.pformat(bytes(range(40)), width=30))
print(pprint.pformat([bytes(range(40))], width=30))
print(pprint.pformat(bytearray(b"bytearray contents that run on and on"),
                     width=25))
print(pprint.pformat(b"abcd"), pprint.pformat(b""))

# ---- collections and namespaces ---------------------------------------------
print(pprint.pformat(collections.OrderedDict((str(i), i) for i in range(8)),
                     width=30))
print(pprint.pformat(collections.OrderedDict()))
dd = collections.defaultdict(list)
for i in range(6):
    dd["k%d" % i].append(i)
print(pprint.pformat(dd, width=30))
print(pprint.pformat(collections.defaultdict(int)))
print(pprint.pformat(collections.Counter("mississippi river runs"), width=30))
print(pprint.pformat(collections.ChainMap({"a": 1, "b": 2},
                                          {"c": list(range(10))}),
                     width=30))
print(pprint.pformat(collections.deque(range(20)), width=30))
print(pprint.pformat(collections.deque(range(20), maxlen=25), width=30))
print(pprint.pformat(collections.deque()))
print(pprint.pformat(collections.UserDict({"a": list(range(12))}), width=30))
print(pprint.pformat(collections.UserList(list(range(20))), width=30))
print(pprint.pformat(collections.UserString("user string " * 6), width=30))
print(pprint.pformat(types.SimpleNamespace(alpha=list(range(10)),
                                           beta="b" * 20), width=30))


class Space(types.SimpleNamespace):
    pass


print(pprint.pformat(Space(a=list(range(15))), width=30))


class Plain(dict):
    pass


class Written(dict):
    def __repr__(self):
        return "Written(%d keys)" % len(self)


print(pprint.pformat(Plain({"k%d" % i: i for i in range(8)}), width=30))
print(pprint.pformat(Written({"k%d" % i: i for i in range(8)}), width=30))


@dataclass
class Point:
    x: int
    y: int
    tags: list = field(default_factory=list)
    hidden: int = field(default=0, repr=False)


@dataclass
class Custom:
    x: int

    def __repr__(self):
        return "Custom!"


print(pprint.pformat(Point(1, 2, list(range(20))), width=30))
print(pprint.pformat([Point(3, 4)], width=10))
print(pprint.pformat(Custom(5), width=5))

# ---- streams and refusals ---------------------------------------------------
buf = io.StringIO()
pprint.pprint({"to": "a stream"}, stream=buf)
pprint.PrettyPrinter(stream=buf, width=10).pprint(list(range(5)))
print(repr(buf.getvalue()))
attempt("indent", lambda: pprint.PrettyPrinter(indent=-1))
attempt("depth", lambda: pprint.PrettyPrinter(depth=0))
attempt("width", lambda: pprint.PrettyPrinter(width=0))
print("done")
