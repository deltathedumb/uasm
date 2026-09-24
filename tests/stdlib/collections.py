# COVERAGE: namedtuple (construction by position and keyword, defaults,
# rename, _make/_replace/_asdict/_fields/_field_defaults, tuple-ness);
# deque (both ends, maxlen discarding from the far end, rotate, extendleft's
# reversal, the sequence protocol); defaultdict (__missing__ inserting and
# `get` not); Counter (missing is zero without inserting, most_common ties,
# elements, subtract keeping negatives, the four operators dropping them);
# OrderedDict (move_to_end, order-sensitive equality against another
# OrderedDict and insensitive against a plain dict); ChainMap (lookup order,
# writes hitting the first map only, new_child/parents); UserDict, UserList
# and UserString.
#
# THE dict SUBCLASSES ARE CHECKED FOR BEING DICTS, not just for behaving like
# them: `isinstance(c, dict)`, `c == {...}` and `dict(c)` are the three things
# a composition-based implementation gets wrong while every method still
# looks right, so they are asserted directly.
import collections
from collections import (ChainMap, Counter, OrderedDict, UserDict, UserList,
                         UserString, defaultdict, deque, namedtuple)

# ---- namedtuple ------------------------------------------------------------
P = namedtuple("P", "x y")
p = P(1, 2)
print(p, p.x, p.y, p[0], p[1], len(p))
print(P._fields, P._field_defaults)
print(p._asdict(), p._replace(x=9), p._replace(y=8))
print(P._make([3, 4]), P(y=6, x=5), P(7, y=8))
print(isinstance(p, tuple), p == (1, 2), tuple(p), list(p))
a, b = p
print(a, b)
print(sorted([P(2, 1), P(1, 9)]))
print(len({P(1, 2), P(1, 2)}))

# `"x,y"` and a list are both accepted spellings of the field names.
print(namedtuple("Q", "x,y")._fields, namedtuple("R", ["a", "b"])._fields)

# DEFAULTS APPLY TO THE LAST FIELDS, which is what makes them useful.
D = namedtuple("D", "a b c", defaults=[20, 30])
print(D(1), D(1, 2), D(1, 2, 3), D._field_defaults)

# `rename` turns an unusable name into its position rather than raising.
print(namedtuple("N", ["ok", "class", "ok"], rename=True)._fields)
try:
    namedtuple("Bad", ["class"])
except ValueError as e:
    print("keyword refused:", "identifiers" in str(e))
try:
    namedtuple("Bad", ["a", "a"])
except ValueError as e:
    print("duplicate refused:", "duplicate" in str(e))
try:
    P(1, 2, 3)
except TypeError:
    print("too many refused")
try:
    P(1)
except TypeError:
    print("too few refused")
try:
    P(1, 2, z=3)
except TypeError:
    print("unexpected keyword refused")

# ---- deque -----------------------------------------------------------------
d = deque([1, 2, 3])
d.appendleft(0)
d.append(4)
print(list(d), len(d), d[0], d[-1])
print(d.popleft(), d.pop(), list(d))
d.rotate(1)
print(list(d))
d.rotate(-1)
print(list(d))
d.extend([7, 8])
d.extendleft([9, 10])
# EXTENDLEFT REVERSES with respect to its argument, which surprises people and
# falls out of each element going onto the same end in turn.
print(list(d))
print(d.count(7), d.index(7), 7 in d, 99 in d)
d.remove(7)
print(list(d))
d.reverse()
print(list(d))
print(list(deque(d)), deque([1, 2]) == deque([1, 2]), deque([1]) == deque([2]))

bounded = deque(maxlen=2)
bounded.extend([1, 2, 3])
print(list(bounded), bounded.maxlen)
bounded.appendleft(0)
print(list(bounded))
empty = deque()
print(len(empty), bool(empty), bool(deque([1])))
try:
    empty.pop()
except IndexError:
    print("pop from empty refused")
try:
    empty.popleft()
except IndexError:
    print("popleft from empty refused")

# ---- defaultdict -----------------------------------------------------------
dd = defaultdict(list)
dd["a"].append(1)
print(sorted(dd.items()), isinstance(dd, dict))
# A MISS THROUGH `[]` INSERTS; a miss through `get` does not.
print("b" in dd, dd["b"], "b" in dd)
print(dd.get("c"), "c" in dd, dd.get("c", 5))
print(dd == {"a": [1], "b": []}, dict(dd) == {"a": [1], "b": []})
counts = defaultdict(int)
for ch in "aab":
    counts[ch] += 1
print(sorted(counts.items()), counts.default_factory is int)
plain = defaultdict(None)
try:
    plain["x"]
except KeyError:
    print("no factory raises")
print(sorted(defaultdict(int, {"z": 9}).items()))

# ---- Counter ---------------------------------------------------------------
c = Counter("aabbbc")
print(sorted(c.items()), isinstance(c, dict), len(c))
# MISSING IS ZERO AND DOES NOT INSERT, which is the difference from
# defaultdict(int) and the reason a read leaves the length alone.
print(c["z"], "z" in c, len(c))
print(c.most_common(1), c.most_common(2), c.most_common())
print(sorted(c.elements()))
print(c.total())
c.update("a")
print(sorted(c.items()))
c.subtract("aaaa")
# SUBTRACT KEEPS NEGATIVES; the operators do not.
print(sorted(c.items()))
print(sorted((+c).items()), sorted((-c).items()))

x = Counter("aab")
y = Counter("abb")
print(sorted((x + y).items()))
print(sorted((x - y).items()))
print(sorted((x | y).items()))
print(sorted((x & y).items()))
print(x == Counter("aab"), x == {"a": 2, "b": 1})
print(sorted(Counter(a=3, b=1).items()))
print(sorted(Counter({"a": 2}).items()))

# TIES COME BACK IN INSERTION ORDER, because the sort is stable over the
# dict's own order.
print(Counter("abc").most_common())

# ---- OrderedDict -----------------------------------------------------------
od = OrderedDict([("a", 1), ("b", 2)])
print(list(od), isinstance(od, dict), od["a"])
od.move_to_end("a")
print(list(od))
od.move_to_end("a", last=False)
print(list(od))
# ORDER-SENSITIVE against another OrderedDict, INSENSITIVE against a plain
# dict -- which looks inconsistent and is CPython's rule.
print(OrderedDict([("a", 1), ("b", 2)]) == OrderedDict([("b", 2), ("a", 1)]))
print(OrderedDict([("a", 1), ("b", 2)]) == OrderedDict([("a", 1), ("b", 2)]))
print(OrderedDict([("a", 1), ("b", 2)]) == {"b": 2, "a": 1})
print({"a": 1, "b": 2} == {"b": 2, "a": 1})
print(od.popitem(), list(od))
od2 = OrderedDict([("x", 1), ("y", 2), ("z", 3)])
print(od2.popitem(last=False), list(od2))
print(sorted(OrderedDict(a=1).items()), list(od2.copy()))
try:
    OrderedDict().popitem()
except KeyError:
    print("popitem on empty refused")

# ---- ChainMap --------------------------------------------------------------
first = {"x": 1, "y": 2}
second = {"y": 20, "z": 30}
cm = ChainMap(first, second)
print(cm["x"], cm["y"], cm["z"], len(cm.maps))
# A WRITE GOES TO THE FIRST MAP ONLY; the shadowed key stays where it was.
cm["y"] = 99
print(sorted(first.items()), sorted(second.items()))
print(sorted(cm.keys()), len(cm), "x" in cm, "nope" in cm)
print(sorted(cm.items()), sorted(cm.values()))
print(cm.get("z"), cm.get("nope"), cm.get("nope", 4))
child = cm.new_child({"w": 0})
print(child["w"], child["z"], len(child.maps))
print(len(child.parents.maps), sorted(child.parents.keys()))
try:
    cm["z"]
    del cm["z"]
except KeyError as e:
    print("delete from a later map refused")
try:
    ChainMap({}).popitem()
except KeyError:
    print("popitem on empty refused")
print(ChainMap().maps, bool(ChainMap()), bool(ChainMap({"a": 1})))
print(cm.setdefault("new", 5), cm["new"], sorted(first.items()))
print(cm.pop("new"), "new" in cm)

# ---- UserDict / UserList / UserString --------------------------------------
ud = UserDict({"a": 1})
ud["b"] = 2
print(sorted(ud.items()), len(ud), ud["a"], "a" in ud, ud.data)
print(ud == {"a": 1, "b": 2}, isinstance(ud, dict))
print(ud.get("zz", 3), ud.pop("b"), sorted(ud.keys()))
ud.update({"c": 9})
print(sorted(ud.items()), ud.setdefault("d", 4), sorted(ud.items()))
del ud["d"]
print(sorted(ud.items()), repr(ud))


# THE POINT OF WRAPPING: overriding `__setitem__` catches EVERY write,
# including the ones the constructor and `update` make.
class Upper(UserDict):
    def __setitem__(self, key, value):
        super().__setitem__(key.upper(), value)


u = Upper({"a": 1})
u.update({"b": 2})
u["c"] = 3
print(sorted(u.items()))

ul = UserList([3, 1, 2])
ul.append(4)
ul.sort()
print(list(ul), len(ul), ul[0], ul.data, isinstance(ul, list))
print(ul == [1, 2, 3, 4], ul.index(3), ul.count(1), 2 in ul)
ul.reverse()
print(list(ul))
ul.insert(0, 0)
print(list(ul), ul.pop(), list(ul))
ul.remove(0)
print(list(ul), list(ul + UserList([9])), list(UserList([0]) + ul))
print(list(ul * 2), list(ul[1:]))

us = UserString("hi there")
print(us, len(us), us.upper(), us.data, isinstance(us, str))
print(us == "hi there", us + "!", "hi" in us, us[0])
print(us.split(), us.replace("hi", "yo"), us.title(), us.strip())
print(us.startswith("hi"), us.endswith("re"), us.find("there"))
print(UserString("a") * 3, UserString("A").lower(), hash(us) == hash("hi there"))
print(sorted([UserString("b"), UserString("a")]))

# EMPTY REPRS: `OrderedDict()` and `Counter()` drop their braces when there is
# nothing to put in them; `defaultdict` and `ChainMap` keep theirs.
print(repr(OrderedDict()), repr(Counter()), repr(defaultdict(int)),
      repr(deque()), repr(ChainMap()), repr(OrderedDict(a=1)),
      repr(Counter("aab")))
