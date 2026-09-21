"""Ordinary Python, run three ways and compared against CPython.

`test_endtoend.py` does this for the statically typed subset. This does it for
the DYNAMIC path -- the one a Python script actually takes, where every value
is a runtime object and every operation a call into `objects/csource.py`.

Each program is checked against CPython through:

    1. the reference interpreter, on the IR the frontend produced
    2. the C backend, compiled and executed

Two paths rather than the four `test_endtoend` uses, because these programs
are large enough that a per-program x86-64 assemble-and-link would dominate the
suite's runtime, and the machine backends are covered by the differential
fuzzer instead.

WHY THESE PROGRAMS. Each was written while landing the feature it covers and
each one caught something. They are kept as a corpus rather than folded into
one file because a failure then names the feature: if `exceptions` fails and
`sequences` passes, the handler chain is what broke.

`tools/dynamic_diff.py` generates programs from the same grammar and is the
better bug-finder -- it found a falsy-empty-`Block` that made every `else`
branch unreachable. This is the regression half: what is known to work, kept
working, named.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from tests import harness

from uasm.diagnostics import DiagnosticSink
from uasm.driver import Options, compile_source
from uasm.ir.interpreter import Interpreter

HAS_CC = shutil.which("gcc") or shutil.which("cc")

PROGRAMS = {
    # A KEYWORD DOES NOT CHANGE THE ARGUMENT COUNT, and the built-in method
    # dispatch is indexed by count -- so every one of these used to be
    # DROPPED rather than refused. `split(",", maxsplit=1)` answered three
    # pieces, `replace(count=1)` replaced all of them, and an unknown keyword
    # was accepted in silence. Ten methods take a keyword and eight were
    # wrong; the two that were not, `sort` and `update`, have branches of
    # their own and are here to keep them that way.
    "builtin_method_keywords": """
        print("a,b,c".split(",", maxsplit=1))
        print("a,b,c".rsplit(",", maxsplit=1))
        print("a b c".split(maxsplit=1))
        print("a\\nb".splitlines(keepends=True))
        print("aaa".replace("a", "b", count=1))
        print("a\\tb".expandtabs(tabsize=2))
        print("a".encode(encoding="utf-8", errors="strict"))
        print(b"a".decode(encoding="utf-8", errors="strict"))
        xs = [3, 1, 2]
        xs.sort(reverse=True)
        print(xs)
        ys = ["bb", "a"]
        ys.sort(key=len)
        print(ys)
        d = {"a": 1}
        d.update(b=2)
        print(sorted(d.items()))
    """,
    # AN UNKNOWN KEYWORD IS A TypeError, not a value. Every message here is
    # CPython 3.14's own wording, because a program may be reading it.
    "builtin_method_keyword_errors": """
        def show(label, fn):
            try:
                print(label, repr(fn()))
            except Exception as e:
                print(label, type(e).__name__ + ":", e)
        show("unknown   ", lambda: (5).to_bytes(2, "little", nonsense=True))
        show("duplicate ", lambda: "a,b".split(",", sep=";"))
        try:
            "abc".count("a", x=1)
        except TypeError as e:
            # CPYTHON NAMES THE OWNER: `str.count() takes no keyword
            # arguments`. There is no static type for the receiver here --
            # that is the whole reason the dispatch is by arity -- so the
            # bare method name is as close as this gets, and the prefix is
            # normalised away rather than the claim being dropped.
            print("none taken TypeError:", str(e).replace("str.", ""))
    """,
    # A NAME COLLISION KEEPS THE USER METHOD'S KEYWORDS. When a program defines
    # its own `replace` or `split`, the receiver picks between the builtin and
    # the user method at RUN TIME, so the keyword cannot be folded into a
    # builtin's slot at compile time -- `datetime.replace(tzinfo=...)` against
    # `str.replace`'s `count` is why refusing it is not an option either.
    #
    # THE BUILTIN HALF OF SUCH A CALL STILL DROPS ITS KEYWORDS, and that is a
    # known gap rather than a fixed one: in a module that defines `split`,
    # `"a,b,c".split(",", maxsplit=1)` still answers three pieces. Folding it
    # needs the keyword VALUES, and the user half of the same branch lowers
    # `node.keywords` itself -- so producing both from one place means
    # lowering each argument twice, which in the bundled parser consumed the
    # token stream twice and made every starred form a SyntaxError. Those two
    # lines are therefore not asserted here; what is asserted is that the
    # user method, which is the common case, gets what it was passed.
    "builtin_method_name_collision": """
        class Thing:
            def __init__(self, a, b):
                self.a, self.b = a, b
            def replace(self, a=None, b=None):
                return Thing(a if a is not None else self.a,
                             b if b is not None else self.b)
            def split(self, sep=None, maxsplit=-1):
                return ("user", sep, maxsplit)
        t = Thing(1, 2)
        r = t.replace(b=9)
        print(r.a, r.b)
        print(t.split(",", maxsplit=3))
        print(t.split(","))
    """,
    # AN EMPTY `**` MAPPING IS THE COMMON ONE, and it has an exact answer:
    # the positional call. A wrapper forwarding `*args, **kwargs` passes one
    # nearly always. What a NON-EMPTY mapping holds is a run-time fact the
    # compile-time arrangement cannot use, so that is refused -- loudly, where
    # it used to be dropped. See `test_method_keywords.py` for the refusal.
    #
    # A `*args` SPREAD ON THE POSITIONAL SIDE is a different and OLDER gap and
    # is not tested here: `s.split(*args)` cannot be counted at compile time,
    # so it falls to the generic attribute path -- which cannot produce a
    # bound builtin method at all, and answers AttributeError. That predates
    # this change and is unaffected by it.
    "builtin_method_empty_spread": """
        kw = {}
        print("a,b,c".split(",", **kw))
        print("aaa".replace("a", "z", **kw))
        print((5).to_bytes(2, "little", **kw))
    """,
    # EACH ARGUMENT RUNS ONCE. Folding a keyword into its slot means deciding
    # the arrangement AFTER the arguments are lowered, and a first attempt
    # lowered the positional ones twice -- so the separator expression here
    # printed twice and any call with a side effect ran twice.
    "builtin_method_keyword_evaluation": """
        def sep():
            print("evaluated")
            return ","
        print("a,b,c".split(sep(), maxsplit=1))
    """,
    # `to_bytes` READ A BIG INTEGER'S CELL AS A NUMBER, and a big integer's
    # cell holds a POINTER to its limbs -- so this printed a heap address
    # formatted as data, on both compiled paths, while the interpreter was
    # right. `signed=` was dropped on the way in, and the no-argument and
    # one-argument forms did not exist.
    "int_to_bytes": """
        print((2 ** 63).to_bytes(16, "little"))
        print((2 ** 63).to_bytes(16, "little", signed=True))
        print((2 ** 70).to_bytes(16, "little"))
        print((2 ** 70).to_bytes(16, "big"))
        print((-(2 ** 70)).to_bytes(16, "little", signed=True))
        print((-1).to_bytes(2, "little", signed=True))
        print((-128).to_bytes(1, "big", signed=True))
        print((127).to_bytes(1, "big", signed=True))
        print((255).to_bytes())
        print((255).to_bytes(2))
        print((0).to_bytes(0, "big"))
        def show(label, fn):
            try:
                print(label, repr(fn()))
            except Exception as e:
                print(label, type(e).__name__ + ":", e)
        show("negative  ", lambda: (-1).to_bytes(2, "little"))
        show("too big   ", lambda: (300).to_bytes(1, "little"))
        show("signed edge", lambda: (128).to_bytes(1, "big", signed=True))
        show("big too big", lambda: (2 ** 70).to_bytes(4, "little"))
        show("bad length", lambda: (1).to_bytes(-1, "big"))
    """,
    # A BOUND THAT IS WRITTEN AND A BOUND THAT IS GIVEN ARE TWO FACTS. The
    # frontend knew only the first: `has_start` was a compile-time constant
    # saying whether the source spelled a bound, and a bound that was spelled
    # but evaluated to None went down the integer path -- so every container
    # answered a TypeError about NoneType for `xs[None:None]`, which is
    # CPython's way of spelling the whole thing. `apy_slice_given` is the
    # second fact and it can only be a run-time one.
    "slice_none_bounds": """
        s = "abcdef"
        xs = [1, 2, 3, 4]
        t = (1, 2, 3)
        b = b"abcd"
        n = None
        print(s[None:None], s[None:3], s[1:None], s[::None])
        print(s[None:None:None], s[n:n:-1], s[None::2])
        print(xs[None:None], t[None:None], b[None:None])
        print(bytearray(b"abc")[None:None], range(10)[None:None:2])
        print(xs[n:n:n], s[n:3], s[1:n], s[::n])
        def show(label, fn):
            try:
                print(label, repr(fn()))
            except Exception as e:
                print(label, type(e).__name__ + ":", e)
        show("step zero  ", lambda: s[::0])
        show("str bound  ", lambda: s["a":])
        show("float bound", lambda: s[1.5:])
        show("list bound ", lambda: xs[[]:])
    """,
    # A BUILTIN HAS NO CLASS DICT TO SEARCH, so the list of what it carries
    # is written out -- and it had eight names on it. `hasattr([1], "__eq__")`
    # was False for the most comparable object in the language, and so was
    # every `__add__`, `__lt__`, `__repr__` and `__delitem__` a structural
    # test or a numeric tower reads.
    "builtin_protocol_methods": """
        names = ["__len__", "__iter__", "__contains__", "__getitem__",
                 "__setitem__", "__delitem__", "__hash__", "__eq__",
                 "__ne__", "__lt__", "__add__", "__mul__", "__reversed__",
                 "__buffer__", "__index__", "__int__", "__float__",
                 "__abs__", "__neg__", "__bool__", "__str__", "__repr__",
                 "__format__"]
        vals = [("list", [1]), ("dict", {1: 2}), ("set", {1}),
                ("frozenset", frozenset([1])), ("tuple", (1,)),
                ("str", "a"), ("bytes", b"a"),
                ("bytearray", bytearray(b"a")), ("int", 1), ("float", 1.0),
                ("range", range(3))]
        for label, v in vals:
            print(label, " ".join(n for n in names if hasattr(v, n)))

        # AND THEY HAVE TO WORK, not merely answer `hasattr`.
        xs = [1, 2, 3]
        xs.__delitem__(0)
        d = {"a": 1, "b": 2}
        d.__delitem__("a")
        print(xs, d)
        print(list([1, 2].__reversed__()), list(range(3).__reversed__()))
        print([1].__add__([2]), "a".__add__("b"), "ab".__mul__(2))
        print((7).__floordiv__(2), (7).__mod__(2), (7).__divmod__(2))
        print((6).__and__(3), (6).__xor__(3), (1).__lshift__(4))
        print((5).__invert__(), (-5).__abs__(), True.__index__())
        print((2.7).__floor__(), (2.1).__ceil__(), (2.9).__int__())
        print((2 ** 70).__add__(1), (3 + 4j).__abs__())
        print("%d-%s".__mod__((1, "a")), {"a": 1}.__or__({"b": 2}))
        print(sorted({1, 2}.__xor__({2, 3})), (255).__format__("x"))

        # THE SENTINEL, WHICH IS THE PART THAT IS NOT OBVIOUS.
        # `(1).__eq__("a")` is `NotImplemented` and not False: `==` above the
        # method turns a pair of them into False and `<` turns them into the
        # TypeError a program sees, and `functools.total_ordering` reads the
        # sentinel to decide whether to try the reflected operation. The
        # numbers widen ONE WAY -- `(1).__eq__(1.0)` declines and
        # `(1.0).__eq__(1)` answers -- which is how Python arranges for
        # exactly one side to decide.
        pairs = [(1, 1), (1, 1.0), (1.0, 1), (1j, 1), (1, 1j),
                 ("a", "a"), ("a", 1), (b"a", bytearray(b"a")),
                 (bytearray(b"a"), b"a"), ([1], (1,)), ([1], [1]),
                 ({1}, frozenset([1])), ({}, {}), (None, None), (None, 1),
                 (range(3), range(3))]
        for a, b in pairs:
            print(repr(a.__eq__(b)), repr(a.__lt__(b)))

        class P:
            pass
        p, q = P(), P()
        print(p.__eq__(p), p.__eq__(q), p.__ne__(p), p.__lt__(q))
        print(type(p.__repr__()) is str, p.__format__("") == str(p))
    """,
    # A SLICE DELETES A SET OF POSITIONS AND NOT A SPAN. `del xs[::2]` was
    # refused with a message of this runtime's own invention -- "only step 1
    # slice deletion is supported" -- and a bytearray could not be deleted
    # from at all, by index or by slice, on any of the three runtimes.
    "delete_slices_and_bytearrays": """
        def show(label, fn):
            try:
                print(label, repr(fn()))
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        def onlist(fn):
            def go():
                xs = [0, 1, 2, 3, 4, 5]
                fn(xs)
                return xs
            return go

        def onbytes(fn):
            def go():
                ba = bytearray(b"abcdef")
                fn(ba)
                return ba
            return go

        show("xs [::2]   ", onlist(lambda x: x.__delitem__(slice(None, None, 2))))
        show("xs [::-2]  ", onlist(lambda x: x.__delitem__(slice(None, None, -2))))
        show("xs [1:5:2] ", onlist(lambda x: x.__delitem__(slice(1, 5, 2))))
        show("xs [1:3]   ", onlist(lambda x: x.__delitem__(slice(1, 3))))
        show("xs [:]     ", onlist(lambda x: x.__delitem__(slice(None))))
        show("xs [::0]   ", onlist(lambda x: x.__delitem__(slice(None, None, 0))))
        show("xs [2]     ", onlist(lambda x: x.__delitem__(2)))
        show("ba [0]     ", onbytes(lambda b: b.__delitem__(0)))
        show("ba [-1]    ", onbytes(lambda b: b.__delitem__(-1)))
        show("ba [1:3]   ", onbytes(lambda b: b.__delitem__(slice(1, 3))))
        show("ba [:]     ", onbytes(lambda b: b.__delitem__(slice(None))))
        show("ba [::2]   ", onbytes(lambda b: b.__delitem__(slice(None, None, 2))))
        show("ba [::-2]  ", onbytes(lambda b: b.__delitem__(slice(None, None, -2))))
        show("ba [1:9]   ", onbytes(lambda b: b.__delitem__(slice(1, 9))))
        show("ba oob     ", onbytes(lambda b: b.__delitem__(99)))
        show("ba bad key ", onbytes(lambda b: b.__delitem__("x")))

        # AND THE STATEMENT, which is what a program actually writes.
        ys = [0, 1, 2, 3, 4, 5]
        del ys[::2]
        zs = [0, 1, 2, 3, 4, 5]
        del zs[1::2]
        ba = bytearray(b"abcdef")
        del ba[2]
        bb = bytearray(b"abcdef")
        del bb[::3]
        print(ys, zs, ba, bb)
        # THE BUFFER KEEPS ITS TERMINATOR: a shrunk bytearray is still a
        # string to the C that reads it, so what follows the last byte
        # matters even though nothing reads past the length.
        print(bytes(bb), len(bb), bb.decode())
        show("bytes del  ", lambda: b"abc".__delitem__(0))
    """,
    # A POSITION AND A WIDTH ARE CHARACTERS, and five places counted bytes.
    # Every one of them was invisible in ASCII and wrong the moment a string
    # held anything else: `"café".count("")` answered 6 because it counted
    # byte boundaries, one of which is inside the two bytes of the `é`.
    "character_positions_and_widths": """
        def show(label, fn):
            try:
                print(label, repr(fn()))
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        s = "éàbcé"
        # AN EMPTY NEEDLE COUNTS THE PLACES BETWEEN CHARACTERS.
        print("café".count(""), "abc".count(""), "".count(""))
        print(b"abc".count(b""), b"".count(b""))
        # AND THE BOUNDS NAME CHARACTERS, which is what `find` already did.
        print(s.count("é"), s.count("é", 1, 5), s.count("é", 1), s.count("é", 0, 1))
        print(b"a-b".count(b"-", 1, 3))
        # AN EMPTY NEEDLE IN `replace` INSERTS BETWEEN CHARACTERS TOO.
        print("café".replace("", "-"))
        print("日本".replace("", "-"), "日本".replace("", "-", 2))
        print(b"ab".replace(b"", b"-"))
        # A TAB STOP COUNTS COLUMNS AND A COLUMN IS A CHARACTER.
        print(repr("é\tx".expandtabs(4)), repr("ab\tx".expandtabs(4)))
        print(repr("日\tx".expandtabs(2)))
        # A FORMAT WIDTH IS A CHARACTER COUNT, its fill is a CHARACTER, and
        # its precision truncates characters.
        print("{:>7}".format(s), "{:<7}".format(s), "{:^7}".format(s))
        print("{:*>7}".format(s), "{:é>7}".format("ab"), "{:é^7}".format("ab"))
        print("{:.2}".format(s), format(s, ">7"), f"{s:>7}")
        print("{:>7}".format(42), "{:08.2f}".format(-1.5))
    """,
    # CASE MAPPING WAS ASCII PLUS A HAND-CODED LATIN-1 BRANCH, and the C
    # said so: "a full Unicode case table is not here". So `"Σ".lower()`
    # answered its input, `"İ".lower()` did too where Python gives an `i` and
    # a combining dot, and `"ﬁx".title()` came out `"ﬁX"` -- the ligature
    # unraised and the `x` raised, which is both halves wrong at once.
    "unicode_case_mapping": """
        # A CHARACTER CAN GROW. None of these has a single-character form.
        print("ß".upper(), "ﬁ".upper(), "ﬃ".upper(), "ﬅ".upper())
        print("İ".lower(), "ǰ".upper(), "ᾼ".upper())
        # TITLECASE IS NOT UPPERCASE, and exactly one class of character can
        # show the difference: `ß` capitalizes to `Ss` and uppercases to `SS`.
        print("ß".title(), "ß".capitalize(), "ﬁ".title(), "ǆ".title())
        print("ǅ".upper(), "ǅ".lower(), "ǅ".title(), "ǅ".swapcase())
        # A GREEK CAPITAL SIGMA AT THE END OF A WORD TAKES THE FINAL FORM.
        print("ΟΣ".lower(), "ΣΟ".lower(), "Ο Σ".lower(), "ΟΣ.".lower())
        print("ΑΣΑ".lower(), "Σ".lower(), "ΟΣ".swapcase(), "ΟΣ".casefold())
        print("ΟΣ".title(), "Ο Σ".capitalize())
        # AND THE REST OF THE ALPHABETS, which a Latin-1 branch could not see.
        print("ΑΒΓ".lower(), "αβγ".upper(), "ЖУК".lower(), "жук".upper())
        print("İstanbul".upper(), "İstanbul".swapcase())
        # WHAT WAS ALREADY RIGHT, kept right: the word rule, and the ASCII
        # the table deliberately leaves out.
        print("a1b".title(), "don't".title(), "hello World".capitalize())
        print("a中b".title(), "中a".upper(), "éàbcé".title())
        print("ﬁx".title(), "ﬁx".upper(), "ﬁx".capitalize())
        print("ẞ".lower(), "ẞ".casefold(), "ı".upper(), "ｱ".upper())
    """,
    # A BYTEARRAY IS A BYTES CELL THAT ADMITS IT IS MUTABLE, and `mut` was
    # not travelling with the tag: every method came back as plain bytes, so
    # `bytearray(b"ab").upper()` answered something a program could not write
    # into. The interpreter had the other half of the same gap and refused
    # the methods outright, reporting an AttributeError about names Python
    # plainly has.
    "bytearray_methods_and_type": """
        b = bytearray(b"a-b-c")
        print(b.replace(b"-", b"+"), b.upper(), b.upper().lower())
        print(bytearray(b" x ").strip(), b.split(b"-"))
        print(bytearray(b"-").join([b"a", b"b"]), b[1:3])
        print(b + b"z", bytearray(b"ab") * 2, 2 * bytearray(b"ab"))
        print(bytearray(b"ab").center(6, b"-"), bytearray(b"ab cd").title())
        # THE SEPARATOR COMES BACK AS THE RECEIVER'S KIND, not as the kind it
        # was passed in. All three pieces are a bytearray in Python.
        print(b.partition(b"-"), b.rpartition(b"-"))
        print(bytearray(b"a\\nb").splitlines(), bytearray(b"ab").zfill(4))
        print(bytearray(b"ab").removeprefix(b"a"))
        # AND THE ONES THAT ARE NOT BYTES AT ALL stay what they are.
        print(bytearray(b"ab").hex(), bytearray(b"ab").decode())
        print(bytearray(b"ab").find(b"b"), bytearray(b"ab").isalpha())
        # A BYTES RECEIVER IS UNTOUCHED, which is what the `mut` test is for.
        print(b"ab".upper(), b"a-b".split(b"-"), b"c" + bytearray(b"ab"))
    """,
    # `bytes.translate` IS A DIFFERENT METHOD WEARING `str.translate`'s NAME,
    # and neither it, `bytes.maketrans` nor `bytearray.copy` existed: all
    # three answered AttributeError about names Python plainly has. A str maps
    # code points through a DICT and may replace one character with a whole
    # string; bytes map BYTES through a 256-byte table and take a second
    # `delete` argument the str form has no parameter for at all.
    "bytes_translate_and_copy": """
        t = bytes.maketrans(b"ab", b"xy")
        print(len(t), type(t) is bytes, t[97:99])
        print(b"abc".translate(t), b"abc".translate(None))
        print(b"abc".translate(t, b"c"), b"abc".translate(t, delete=b"c"))
        print(b"abc".translate(None, b"a"))
        # THE RECEIVER'S KIND TRAVELS WITH THE RESULT.
        print(bytearray(b"abc").translate(t),
              type(bytearray(b"abc").translate(t)).__name__)
        print(bytearray.maketrans(b"ab", b"xy")[97:99])
        # A MEMORYVIEW IS BYTES-LIKE, here as everywhere else.
        print(b"abc".translate(memoryview(t)), b"abc".translate(bytearray(t)))
        # THE STR METHOD IS UNTOUCHED, and refuses what only bytes can take.
        print("abc".translate({97: "X"}))
        try:
            "abc".translate({97: "X"}, b"c")
        except TypeError as e:
            print("positional:", e)
        try:
            "abc".translate({97: "X"}, delete=b"c")
        except TypeError as e:
            print("keyword:", e)
        for bad in (b"xy", "x" * 256):
            try:
                print(b"abc".translate(bad))
            except (TypeError, ValueError) as e:
                print(type(e).__name__ + ":", e)
        try:
            bytes.maketrans(b"ab", b"xyz")
        except ValueError as e:
            print("maketrans:", e)
        # `.copy()` IS THE BYTEARRAY'S ALONE, and it copies the bytes rather
        # than sharing them.
        original = bytearray(b"ab")
        made = original.copy()
        made[0] = 122
        print(original, made, made is original)
        for immutable in (b"ab", "ab"):
            try:
                immutable.copy()
            except AttributeError as e:
                print("copy:", e)
    """,
    # `bytearray` WAS THE ONE BUILTIN TYPE THAT COULD NOT BE A VALUE. Calling
    # it worked, `isinstance(x, bytearray)` worked and `bytearray.__name__`
    # worked -- but naming it anywhere a value belongs was a COMPILE ERROR
    # (`E0056`), for a name Python hands out like any other type object. So
    # `type(x) is bytearray` did not fail at run time: the program did not
    # build.
    "bytearray_as_a_value": """
        print(bytearray(b"ab"), bytearray())
        print(repr(bytearray), bytearray.__name__)
        print(type(bytearray(b"a")) is bytearray, type(b"a") is bytearray)
        print(isinstance(bytearray(b"a"), bytearray), isinstance(b"a", bytearray))
        print(isinstance(bytearray(b"a"), (bytes, bytearray)))
        print(list(map(bytearray, [b"ab", b"c"])))
        # ONE OBJECT PER NAME, which is what `is` and a dict key both need.
        print(bytearray is bytearray, {bytearray: 1}[bytearray])
        # THE VALUE FORM TAKES THE ZERO-ARGUMENT CALL TOO, which is what
        # `defaultdict(bytearray)` does with it.
        make = bytearray
        print(make(), make(b"ab"), make([1, 2]))
        # AND BYTES IS STILL ITS OWN TYPE, not swept up by the new name.
        print(type(b"a") is bytes, [type(x).__name__ for x in
                                    [bytearray(b"a"), b"a"]])
    """,
    # A BYTEARRAY IS A MUTABLE SEQUENCE AND HAD HALF THE PROTOCOL. `append`
    # and `extend` worked; `insert`, `pop`, `remove`, `clear` and `reverse`
    # answered AttributeError about methods Python plainly gives it.
    #
    # AND `+=` REBOUND RATHER THAN EXTENDING, which is the worse half and the
    # one no error marks: `b += data` built a NEW bytearray, so every other
    # name for the old one kept the old bytes. `*=` had it too, on a LIST as
    # well -- the two mutable sequences are the kinds where the distinction
    # between `x += y` and `x = x + y` is visible at all.
    "bytearray_is_a_mutable_sequence": """
        b = bytearray(b"abc")
        b.append(100); b.extend(b"ef"); b.insert(0, 122)
        print(b, b.pop(), b.pop(0), b)
        b.remove(98)
        print(b)
        b.reverse()
        print(b)
        b.clear()
        print(b, len(b), bool(b))
        # AN ALIAS SEES THE CHANGE, which is the whole of what in place means.
        grown = bytearray(b"ab")
        alias = grown
        grown += b"z"
        print(grown, alias, grown is alias)
        grown *= 2
        print(grown, alias, grown is alias)
        xs = [1]
        ys = xs
        xs *= 3
        print(xs, ys, xs is ys)
        # AND THE DUNDERS THAT NAME THEM ARE VALUES, on the mutable kinds
        # only -- a tuple falls through to `+` and has none.
        print([1].__iadd__([2]), bytearray(b"a").__iadd__(b"z"))
        print(hasattr([1], "__iadd__"), hasattr((1,), "__iadd__"))
        print(hasattr({1}, "__ior__"), hasattr(frozenset({1}), "__ior__"))
        # THE REFUSALS ARE CPYTHON'S, and a bytearray holds NUMBERS.
        for body in (lambda: bytearray().pop(),
                     lambda: bytearray(b"ab").pop(5),
                     lambda: bytearray(b"ab").remove(122),
                     lambda: bytearray(b"ab").remove(b"a"),
                     lambda: bytearray(b"ab").insert(0, 300),
                     lambda: bytearray(b"ab").append(b"z")):
            try:
                body()
            except (IndexError, ValueError, TypeError) as e:
                print(type(e).__name__ + ":", e)
        # BYTES HAS NONE OF THEM, which is the flag doing its work.
        for name in ("pop", "clear", "reverse", "insert", "remove"):
            try:
                getattr(b"ab", name)
            except AttributeError as e:
                print(e)
    """,
    # `x.__class__` IS THE ONE ATTRIBUTE PYTHON GUARANTEES, and every builtin
    # value was missing it: `(5).__class__` was an AttributeError while
    # `type(5)` answered perfectly well. `obj.__class__.__name__` is an
    # everyday idiom -- duck typing, repr helpers, copy and serialisation
    # code -- so a program reaching for it heard that an int has no class.
    "class_of_a_builtin_value": """
        values = [b"a", bytearray(b"a"), "a", [1], (1,), {1: 2}, {1},
                  frozenset({1}), 5, 1.5, True, None, range(3), 1j]
        print([v.__class__.__name__ for v in values])
        # THE SAME OBJECT `type(x)` ANSWERS, which is what makes a comparison
        # against it work rather than merely printing the same.
        print((5).__class__ is type(5), [1].__class__ is type([1]))
        print((5).__class__ is (7).__class__)
        print(hasattr(5, "__class__"), hasattr("a", "__class__"))
        # AND A USER CLASS AND A FUNCTION STILL ANSWER THEIR OWN.
        class Point:
            pass
        print(Point().__class__.__name__, (lambda: 1).__class__.__name__)
    """,
    # `index` TAKES THE SAME BOUNDS `find` DOES and did not: at two or three
    # arguments it fell off the arity table, reached the generic attribute
    # lookup, and reported about a one-argument method -- naming an internal
    # lambda of the interpreter while it did so. The window is how a scan
    # resumes past the last hit, and a list and a tuple have it too.
    #
    # `expandtabs` IS IN THE SAME PARAGRAPH for the opposite reason: it had a
    # width it should not accept. The no-argument form padded the slot with
    # None, so `s.expandtabs(None)` -- a TypeError in Python -- answered what
    # the omitted argument answers. Once a default has been folded into a
    # slot nothing downstream can tell it from a written value, so the slot
    # carries 8. And bytes never had the method at all.
    "bounded_index_and_expandtabs": """
        print("abcabc".index("c", 3), "abcabc".index("c", 0, 3))
        print("abcabc".index("c", None, None), "abcabc".rindex("c", 0, 3))
        print(b"abcabc".index(b"c", 3), bytearray(b"abcabc").index(b"c", 3))
        print([1, 2, 3, 2].index(2, 2), [1, 2, 3, 2].index(2, 0, 2))
        print((1, 2, 3, 2).index(2, 2))
        # A POSITION IS A CHARACTER, in the bounded form as in the bare one.
        print("h\u00e9llo h\u00e9llo".index("llo", 3))
        for body in (lambda: "abcabc".index("z", 0, 3),
                     lambda: "abcabc".rindex("z"),
                     lambda: [1, 2, 3].index(2, 2)):
            try:
                body()
            except ValueError as e:
                print("ValueError:", e)
        print(repr("a\tb".expandtabs()), repr("a\tb".expandtabs(4)))
        print(repr("a\tb".expandtabs(tabsize=4)), repr("a\tb".expandtabs(True)))
        print(b"a\tb".expandtabs(), b"a\tb".expandtabs(4))
        print(bytearray(b"a\tb").expandtabs())
        try:
            "a\tb".expandtabs(None)
        except TypeError as e:
            print("TypeError:", e)
    """,
    # A LONE SURROGATE IS A LEGAL str AND CRASHED THE COMPILER. It is what
    # `errors="surrogateescape"` produces and what `os.fsdecode` hands back
    # for an undecodable filename, so any program touching a non-UTF-8 path
    # can hold one -- and the literal was encoded with plain UTF-8, which
    # raises on it. Not a diagnostic: the compiler died with a Python
    # traceback and never reached the program.
    #
    # NOTHING HERE PRINTS THE CHARACTER ITSELF, because CPython's own stdout
    # refuses it too -- `repr` and the comparisons are what a program can
    # actually do with one.
    "a_lone_surrogate_is_a_string": """
        s = "a\\udcffb"
        print(len(s), repr(s))
        print(s[0], s[2], len(s[1]), ord(s[1]))
        print(s == "a\\udcffb", s != "a\\udcfeb")
        print(s.upper() == "A\\udcffB", s.startswith("a"), s.endswith("b"))
        print("x".join([s, s]) == s + "x" + s)
        print([len(part) for part in s.split("x")])
    """,
    # BYTES AND STR SHARE ONE IMPLEMENTATION and it took the STR view of
    # every question, so every bytes method was wrong on a byte above 0x7F:
    # it decoded two bytes as one character, asked the Unicode table about
    # it, and answered accordingly. Python's bytes methods are ASCII-ONLY --
    # a byte above 0x7F is not a letter, has no case, is not whitespace --
    # and a bytes WIDTH is a byte count where a str's is a character count,
    # which is the opposite of the fix the str side needed.
    "bytes_methods_are_byte_methods": """
        b = b"\\xc3\\xa9"
        print(b.upper(), b"\\xc3\\x89".lower(), b"\\xc3\\xa9ab".title())
        print(b"\\xc3\\xa9ab".capitalize(), b"\\xc3\\xa9aB".swapcase())
        print(b.isalpha(), b.isalnum(), b.isascii(), b"\\xc2\\xa0".isspace())
        print(b"\\xc3\\x89".isupper(), b.islower(), b"\\xc3\\x89\\xc3\\xa9".istitle())
        # A WIDTH IS A BYTE COUNT. Two bytes, so nine leaves seven to pad.
        print(b.center(9), b.ljust(9), b.rjust(9), b.zfill(9), len(b))
        print(b"\\xc2\\xa0a\\xc2\\xc2".strip())
        # AND THE STR SIDE IS UNTOUCHED, which is the half the shared body
        # was right about all along.
        print("\\u00e9".upper(), "\\u00e9".isalpha(), "\\u00e9".center(5))
        try:
            b"".index(b"b")
        except ValueError as e:
            print("bytes:", e)
        try:
            "".index("b")
        except ValueError as e:
            print("str:", e)
        # A BYTEARRAY IS A SEQUENCE OF OCTETS and the interpreter alone
        # refused to walk one -- `bytearray` is not a `bytes` to isinstance.
        print(list(bytearray(b"abc")), [x for x in bytearray(b"ab")])
        print(sum(bytearray(b"ab")), sorted(bytearray(b"ba")))
        print(list(zip(bytearray(b"ab"), "xy")))
        print(list(reversed(bytearray(b"ab"))), 97 in bytearray(b"abc"))
    """,
    # `s.split()` SPLITS ON WHITESPACE and knew only the six ASCII bytes, so
    # every Unicode space was an ordinary character to split around:
    # `"\u00a0".split()` answered `["\u00a0"]` where Python answers `[]`.
    # `strip()` was fixed by asking the character table and `split()` was
    # not. The backward walk `rsplit` does is the only new piece -- stepping
    # UTF-8 in reverse means finding the lead byte behind the continuations.
    "splitting_on_unicode_whitespace": """
        print("\\u00a0".split(), "\\u2003x".split())
        print("a\\u00a0b c".split(), "a\\u00a0b c".rsplit())
        print("a\\u2003b\\u2003c".rsplit(None, 1))
        print("a\\u2003b\\u2003c".split(None, 1))
        print("x\\u00a0".split(), "\\u00a0x".split())
        # THE ASCII CASES ARE UNTOUCHED, including where the remainder keeps
        # the whitespace the limit stopped at.
        print("  a  b  ".split(), "  a  b  ".rsplit())
        print("  a  b  ".split(None, 1), "  a  b  ".rsplit(None, 1))
        print("".split(), " \\t\\n ".split())
        # AND A BYTES RECEIVER KEEPS THE ASCII ANSWER: those two bytes are
        # not a character and neither of them is whitespace.
        print(b"a\\xc2\\xa0b".split(), b"  a  b  ".rsplit(None, 1))
    """,
    # A BOUND BUILTIN METHOD IS NOT A FUNCTION. `type([1].index).__name__` is
    # `builtin_function_or_method` in Python and `method-wrapper` for a bound
    # slot like `[1].__len__`; both compiled paths said `function` and the
    # interpreter said `Native` -- the name of a class in objects_host.py, an
    # implementation detail of this compiler in a string a program prints.
    "the_type_of_a_builtin_method": """
        print(type([1].index).__name__, type({}.keys).__name__)
        print(type([].append).__name__, type([1].count).__name__)
        # A DUNDER IS THE ONE THAT DIFFERS: a bound slot is a method-wrapper.
        print(type([1].__len__).__name__, type([1].__eq__).__name__)
        print(type("a".__hash__).__name__, type((5).__abs__).__name__)
        # EXCEPT A DUNDER WITH A RANGE, which is not a slot at all: int
        # writes `__round__` out, and an optional argument is exactly what a
        # slot cannot carry.
        print(type((5).__round__).__name__)
        # AND THE THREE THAT WERE ALREADY RIGHT stay right.
        print(type(print).__name__, type(len).__name__)
        print(type(lambda: 1).__name__, type((5).__class__).__name__)
        def written():
            pass
        print(type(written).__name__)
    """,
    # A NATIVE CARRIED ONE ARITY AND THE CALL MACHINERY TRUNCATED A SURPLUS,
    # so `getattr([1], "__len__")(9)` answered 1 rather than refusing -- the
    # count was capped to the expected number and then found to match. It
    # also meant no method with an OPTIONAL argument could be reached by
    # name: `__round__` was left out of the protocol set for exactly that
    # reason, and is back now that a native can say it takes a range.
    "a_builtin_method_counts_its_arguments": """
        print((5).__round__(), (1234).__round__(-2))
        print((1.55).__round__(), (1.55).__round__(1), True.__round__())
        print(hasattr(5, "__round__"), hasattr(1j, "__round__"))
        print(hasattr("a", "__round__"), round(1.55, 1))
        for body in (lambda: getattr([1], "__iter__")(1, 2),
                     lambda: getattr([1], "__len__")(9),
                     lambda: getattr([1], "__contains__")(),
                     lambda: (5).__round__(1, 2)):
            try:
                body()
            except TypeError as e:
                print("TypeError:", e)
    """,
    # `"abc".upper` IS LOWERED AT THE CALL SITE, so it existed as a CALL and
    # never as an attribute: `getattr("abc", "upper")` was an AttributeError
    # about a method the object plainly has, and `hasattr` said False. Every
    # ordinary builtin method was unreachable by NAME -- which is what
    # `operator.methodcaller`, a plugin table and `getattr(x, name)(...)` all
    # do. The table is generated from CPython's own `hasattr` and from the
    # frontend's own `DYN_METHOD_TABLE`, so the written form and the
    # looked-up form are ONE implementation rather than two that can drift.
    "a_builtin_method_by_name": """
        print(getattr("abc", "upper")(), getattr("abcabc", "find")("c"))
        print(getattr("abcabc", "find")("c", 0, 3), getattr("a,b", "split")(","))
        print(getattr("a b", "split")(), getattr("aaa", "replace")("a", "b"))
        print(getattr("aaa", "replace")("a", "b", 1))
        print(getattr({"a": 1}, "keys")(), getattr({"a": 1}, "get")("z", 7))
        print(getattr(b"a", "decode")(), getattr("a", "encode")())
        print(getattr(b"ab", "hex")(), getattr("a\\tb", "expandtabs")())
        print(getattr(258, "to_bytes")(4, "little"), getattr([1], "copy")())
        print(sorted(getattr({1}, "union")({2})))
        # THE IN-PLACE ONES CHANGE THE RECEIVER, which is the half a value
        # comparison cannot see.
        for name, args in (("append", (9,)), ("pop", ()), ("pop", (0,)),
                           ("sort", ()), ("reverse", ())):
            v = [3, 1]
            print(name, getattr(v, name)(*args), v)
        d = {"a": 1}
        print(getattr(d, "update")({"b": 2}), d)
        # AND THE GATING IS CPYTHON'S: a list has no `upper` and a str no
        # `append`, which is what keeps `hasattr` honest.
        print(hasattr("abc", "upper"), hasattr("abc", "append"))
        print(hasattr([1], "append"), hasattr([1], "upper"))
        print(hasattr({}, "keys"), hasattr([1], "keys"))
        for body in (lambda: getattr("abc", "nosuch"),
                     lambda: getattr([1], "upper")):
            try:
                body()
            except AttributeError as e:
                print(e)
    """,
    # THE ORDER OF CPYTHON'S KEYWORD REFUSALS IS NOT THE ORDER THEY OCCUR TO
    # A READER, and this compiler had it wrong four ways. A MISSING REQUIRED
    # POSITIONAL BEATS EVERYTHING -- `"aaa".replace(count=1)` reports the two
    # positionals it did not get, not the keyword it did. TOO MANY counts the
    # keywords in. Only then does a slot given twice, and only then an
    # unknown name. One of the four used to print `None`: a positional-only
    # parameter has no name, and the message said so out loud.
    #
    # AND THE SUGGESTION IS CPYTHON'S OWN EDIT DISTANCE, ported rather than
    # approximated: a suggestion this compiler makes where CPython makes none
    # is a new divergence, not a kindness. `SEP` finds `sep` and `TABSIZE`
    # does not find `tabsize`, which is the case cost doing its work.
    "the_keyword_refusals": """
        def show(f):
            try:
                print(repr(f()))
            except TypeError as e:
                print("TypeError:", e)
        show(lambda: "aaa".replace(count=1))
        show(lambda: "aaa".replace(nosuch=1))
        show(lambda: "aaa".replace("a", nosuch=1))
        show(lambda: "aaa".replace("a", "b", nosuch=1))
        show(lambda: "a,b".split(",", None, maxsplit=1))
        show(lambda: "a,b".split(SEP=1))
        show(lambda: "a,b".split(nosuch=1))
        show(lambda: "a,b".split(masplit=1))
        show(lambda: "a".encode(nosuch=1))
        show(lambda: "a".encode("u", encoding="x"))
        show(lambda: "a".encode(a=1, b=2, c=3))
        show(lambda: "a".splitlines(keepends=1, x=2))
        show(lambda: "a".splitlines(1, x=2))
        show(lambda: "a".splitlines(x=2))
        show(lambda: "a".expandtabs(TABSIZE=1))
        show(lambda: "a".expandtabs(tabsiz=1))
        show(lambda: (5).to_bytes(x=1))
        # AND THE CALLS THAT ARE FINE STAY FINE.
        show(lambda: "a,b,c".split(",", maxsplit=1))
        show(lambda: "aaa".replace("a", "b", count=1))
        show(lambda: "a\\tb".expandtabs(tabsize=4))
    """,
    # `bytes(s, "utf-8")` IS A DIFFERENT CONSTRUCTOR from `bytes(xs)` -- one
    # encodes text and the other reads octets -- and only the second existed,
    # so the constructor spelling of `.encode()` and `.decode()` raised
    # TypeError for ordinary Python. What tells the two apart is whether an
    # ENCODING was given, by position or by name; `errors` alone is enough,
    # because `str(b, errors="replace")` is a call CPython answers.
    "the_encoding_constructors": """
        print(bytes("caf\\u00e9", "utf-8"), bytes("a", encoding="utf-8"))
        print(bytes("a\\u00ff", "ascii", "replace"), bytes("\\u00e9", "latin-1"))
        print(bytearray("caf\\u00e9", "utf-8"),
              type(bytearray("a", "utf-8")).__name__)
        print(str(b"caf\\xc3\\xa9", "utf-8"), str(b"a", encoding="utf-8"))
        print(str(b"a\\xff", "utf-8", "replace"), str(b"a", errors="replace"))
        print(str(bytearray(b"ab"), "utf-8"))
        # AND THE ONE-ARGUMENT FORMS ARE UNTOUCHED, including the two
        # refusals that are each other's mirror.
        print(bytes(b"ab"), bytes([1, 2]), bytes(3), str(b"ab"), str(5))
        for body in (lambda: bytes("a"), lambda: bytes(b"a", "utf-8"),
                     lambda: str(5, "utf-8"), lambda: bytearray("a")):
            try:
                body()
            except TypeError as e:
                print("TypeError:", e)
    """,
    # A BUILTIN TYPE OBJECT WAS ONLY CALLABLE WHEN THE PROGRAM HAPPENED TO
    # WRITE THAT TYPE'S NAME. `type(5)()` answered `<int object at 0x...>` --
    # a fresh empty instance of a class with no constructor -- because the
    # canonical thunk that turns a type object back into a call is registered
    # by the sweep that SEES the bare word `int`. A module that never writes
    # `int` got a type object that instantiated like a user class, so whether
    # `type(x)()` worked depended on unrelated text elsewhere in the file.
    #
    # The fix keys the instantiation on the TYPE'S OWN NAME rather than on
    # what the module mentions, so `type(x)()` and `x.__class__(...)` reach
    # the same constructor the bare word does. The three kinds that have no
    # empty form -- range, slice and memoryview -- refuse in CPython's words
    # rather than answering an empty instance.
    "a_builtin_type_object_is_callable": """
        class C:
            def __init__(self):
                self.x = 1

        def show(label, f):
            try:
                print(label, repr(f()))
            except TypeError as e:
                print(label, "TypeError:", e)

        # THE EMPTY FORM of every builtin type, reached through a VALUE
        # rather than through the bare word.
        show("list", lambda: type([])())
        show("int", lambda: type(5)())
        show("str", lambda: type("")())
        show("dict", lambda: type({})())
        show("set", lambda: type(set())())
        show("tuple", lambda: type(())())
        show("float", lambda: type(1.5)())
        show("bytes", lambda: type(b"")())
        show("bytearray", lambda: type(bytearray())())
        show("frozenset", lambda: type(frozenset())())
        show("bool", lambda: type(True)())
        show("complex", lambda: type(1j)())
        # `__class__` IS THE SAME OBJECT, so it calls the same way.
        show("int cls", lambda: (5).__class__())
        show("list cls", lambda: [1].__class__())
        # THE CONVERTING FORMS convert, exactly as the named spelling does.
        show("list(it)", lambda: type([])((1, 2)))
        show("int(str)", lambda: type(5)("7"))
        show("str(int)", lambda: type("")(5))
        show("bool(1)", lambda: type(True)(1))
        show("dict(pairs)", lambda: type({})([("a", 1)]))
        show("set(it)", lambda: sorted(type(set())([1, 2, 1])))
        show("complex(2)", lambda: type(1j)(2))
        # AND THE THREE WITH NO EMPTY FORM refuse rather than invent one.
        show("range", lambda: type(range(3))())
        show("slice", lambda: type(slice(1))())
        show("memoryview", lambda: type(memoryview(b"a"))())
        # A USER CLASS was never the broken half, and stays whole.
        print(type(C())().x, type(C()) is C)
    """,
    # THE BUILTIN CONSTRUCTORS HAD NO SIGNATURES. Every keyword written to
    # one was DROPPED without a word -- `int(x="1")` answered 0, `list(x=1)`
    # answered `[]`, `str(object=5)` answered `''` and `complex(real=1,
    # imag=2)` answered `0j`. Surplus positionals were dropped the same way:
    # `int("1", 10, 3)` answered 1 and `bool(1, 2)` answered True. Two of
    # them did not even reach the runtime: `range(1, 2, 3, 4)` emitted C that
    # would not compile and `slice(start=1)` crashed the compiler outright.
    #
    # The signatures are CPython's, read out of it and kept in `CTOR_PARAMS`
    # beside the method table that has done the same job since the keyword
    # round. So is the wording of all six refusals -- and their ORDER, which
    # is not the order they occur to a reader: a type that takes no keyword
    # at all says so ahead of the first argument it also did not get.
    #
    # THE SURPLUS WORDING IS TWO WORDINGS and both are reproduced: most types
    # say `list expected at most 1 argument, got 2` and four say
    # `bytes() takes at most 3 arguments (4 given)` -- with the second form
    # taking over for every type once a KEYWORD is in the count.
    "the_constructor_signatures": """
        def show(label, f):
            try:
                print(label, repr(f()))
            except TypeError as e:
                print(label, "TypeError:", e)

        # THE KEYWORD FORMS, which are calls CPython answers.
        show("int base", lambda: int("ff", base=16))
        show("str object", lambda: str(object=5))
        show("str decode", lambda: str(b"a", encoding="utf-8"))
        show("str errors", lambda: str(b"a", errors="replace"))
        show("bytes source", lambda: bytes(source="a", encoding="utf-8"))
        show("bytearray src", lambda: bytearray(source=b"ab"))
        show("complex parts", lambda: complex(real=1, imag=2))
        show("complex imag", lambda: complex(imag=2))
        show("headless str", lambda: str(encoding="utf-8"))
        # A TYPE THAT TAKES NO KEYWORD AT ALL says exactly that, and says it
        # ahead of every other complaint it could make.
        for body in (lambda: list(iterable=[1]), lambda: float(x=1),
                     lambda: bool(x=1, y=2), lambda: range(start=1),
                     lambda: slice(start=1), lambda: tuple(x=1)):
            show("no keywords", body)
        # THEN A MISSING REQUIRED FIRST ARGUMENT, in the type's own wording.
        show("range none", lambda: range())
        show("slice none", lambda: slice())
        show("mview none", lambda: memoryview())
        show("mview wrong", lambda: memoryview(x=1))
        show("int headless", lambda: int(base=16))
        show("bytes headless", lambda: bytes(encoding="utf-8"))
        show("errors alone", lambda: bytes(b"a", errors="replace"))
        # THEN TOO MANY, in both of CPython's two wordings.
        show("list surplus", lambda: list(1, 2))
        show("int surplus", lambda: int("1", 10, 3))
        show("bool surplus", lambda: bool(1, 2))
        show("dict surplus", lambda: dict(1, 2))
        show("str surplus", lambda: str(1, 2, 3, 4))
        show("range surplus", lambda: range(1, 2, 3, 4))
        show("bytes surplus", lambda: bytes(1, 2, 3, 4))
        show("complex surplus", lambda: complex(1, 2, 3))
        show("mview surplus", lambda: memoryview(b"a", "x"))
        show("kw counted in", lambda: int("1", 10, base=2))
        show("kw only count", lambda: memoryview(object=b"a", x=1))
        # THEN A SLOT GIVEN TWICE, and only then an unknown name -- with
        # CPython's own edit distance deciding whether there is a suggestion
        # to make. `bas` finds `base` and `nos` finds nothing.
        show("twice", lambda: complex(1, real=2))
        show("twice str", lambda: str(b"a", object=1))
        show("unknown", lambda: int(x="1"))
        show("suggested", lambda: int("1", bas=16))
        show("suggested enc", lambda: str(b"a", encodng="utf-8"))
        show("unsuggested", lambda: str(b"a", nos=1))
        show("suggested real", lambda: complex(rea=1))
        # AND A NON-STRING ENCODING is refused rather than looked up: it used
        # to split three ways, LookupError in the interpreter and IGNORED
        # when compiled.
        show("encoding kind", lambda: str(b"a", 5))
        show("errors kind", lambda: str(b"a", "utf-8", 5))
        show("bytes enc kind", lambda: bytes("a", 5))
        show("barr err kind", lambda: bytearray("a", "utf-8", 5))
        # AND THE CALLS THAT WERE ALWAYS FINE STAY FINE.
        print(int("ff", 16), str(b"a", "utf-8"), bytes("a", "utf-8"))
        print(dict(a=1), dict([("a", 1)], b=2), complex(1, 2))
        print(range(1, 2), slice(1, 2), bytes(b"ab"), str(5))
        print(int(), str(), bool(), float(), list(), tuple(), dict())
        print(bytes(), bytearray(), set(), frozenset(), complex())
    """,
    # THE SAME SIGNATURE, THROUGH A TYPE OBJECT AND THROUGH A VALUE. A
    # builtin type reached as a value became a ONE-ARGUMENT thunk, so
    # `f = int` then `f("ff", 16)` dropped the base and `f(x=1)` dropped the
    # keyword; a type object reached through `type(x)` served only the empty
    # and one-argument forms and reported `int() takes no arguments` for the
    # rest. All three spellings are one implementation now -- the frontend
    # folds what it can see, and `apy_ctor_call` folds the rest against the
    # same table at run time.
    "a_builtin_type_is_a_constructor_everywhere": """
        def show(label, f):
            try:
                print(label, repr(f()))
            except TypeError as e:
                print(label, "TypeError:", e)

        # THROUGH A TYPE OBJECT, which no module has to name for this to work.
        show("int base", lambda: type(5)("ff", 16))
        show("int by name", lambda: type(5)("ff", base=16))
        show("str decode", lambda: type("")(b"a", "utf-8"))
        show("str by name", lambda: type("")(b"a", encoding="utf-8"))
        show("bytes encode", lambda: type(b"")("a", "utf-8"))
        show("complex parts", lambda: type(1j)(real=1, imag=2))
        show("dict kw", lambda: type({})(a=1))
        show("dict both", lambda: type({})([("a", 1)], b=2))
        show("range two", lambda: type(range(3))(1, 4))
        show("slice two", lambda: type(slice(1))(1, 4))
        show("mview", lambda: bytes(type(memoryview(b"a"))(b"xy")))
        show("surplus", lambda: type([])(1, 2))
        show("unknown kw", lambda: type([])(x=1))
        show("suggested", lambda: type(5)("1", bas=16))
        show("twice", lambda: type(1j)(1, real=2))
        # AND AS A VALUE, which is what `map`, a key function and a
        # `defaultdict` factory all hold.
        show("value base", lambda: (lambda f: f("ff", 16))(int))
        show("value kw", lambda: (lambda f: f("ff", base=16))(int))
        show("value decode", lambda: (lambda f: f(b"a", "utf-8"))(str))
        show("value dict kw", lambda: (lambda f: f(a=1))(dict))
        show("value surplus", lambda: (lambda f: f(1, 2))(list))
        show("value bad kw", lambda: (lambda f: f(x=1))(list))
        print(list(map(int, ["1", "2"])), sorted([10, 9], key=str))
        print(list(map(str, [1, 2])), list(filter(bool, [0, 1, 2, 0])))
        # THE EMPTY CALL through a value is the type's own zero value, which
        # is what a `defaultdict` factory asks for.
        for kind in (int, float, str, bytes, bytearray, list, tuple, dict,
                     set, frozenset, bool, complex):
            print(kind.__name__, repr(kind()))
        # AND THE TYPE IS STILL A TYPE: one object per name, and the answer
        # to `type(x) is int` rather than merely something callable.
        print(type(5) is int, type("") is str, isinstance(1, int))
        print(int is int, repr(int.__name__), type(5).__name__)
    """,
    # A `**` MAPPING WITH ANYTHING IN IT WAS REFUSED. What it holds is a
    # run-time value, so the frontend could not fold it into a slot and said
    # so out loud: `replace() does not take a non-empty ** mapping in this
    # compiler`. That refused `f(*args, **kwargs)` forwarding through a
    # wrapper, which is the shape the feature exists for.
    #
    # WHICH NAMES IT MAY HOLD IS KNOWN even when the values are not --
    # `METHOD_PARAMS` has said so since the keyword round -- so each
    # parameter is read out of the mapping by name at run time and the
    # ordinary arity dispatch sees the call it should have seen all along.
    # `apy_kw_check` is the half that cannot be decided at compile time: a
    # key naming no parameter, or one naming a slot a positional already
    # filled. Its wordings are CPython's, and so is their order -- too many
    # beats a slot given twice, which beats an unknown name.
    #
    # AND A METHOD WITH NO KEYWORD SIGNATURE now refuses in CPython's words
    # too, naming the OWNER: `str.upper() takes no keyword arguments`, which
    # the runtime can say because it has the receiver and the frontend did
    # not.
    "a_star_star_mapping_reaches_its_slots": """
        def show(label, f):
            try:
                print(label, repr(f()))
            except TypeError as e:
                print(label, "TypeError:", e)

        opts = {"count": 1}
        show("replace", lambda: "aaa".replace("a", "b", **opts))
        show("split", lambda: "a,b,c".split(",", **{"maxsplit": 1}))
        show("split sep", lambda: "a,b".split(**{"sep": ","}))
        show("rsplit", lambda: "a,b,c".rsplit(",", **{"maxsplit": 1}))
        show("encode", lambda: "a".encode(**{"encoding": "utf-8"}))
        show("encode both",
             lambda: "a".encode(**{"encoding": "ascii", "errors": "strict"}))
        show("decode", lambda: b"a".decode(**{"encoding": "utf-8"}))
        show("decode errors", lambda: b"a" .decode(**{"errors": "replace"}))
        show("expandtabs", lambda: "a\\tb".expandtabs(**{"tabsize": 4}))
        show("splitlines", lambda: "a\\nb".splitlines(**{"keepends": True}))
        show("to_bytes", lambda: (258).to_bytes(
            **{"length": 4, "byteorder": "little"}))
        show("to_bytes signed", lambda: (-2).to_bytes(
            **{"length": 2, "byteorder": "big", "signed": True}))
        show("translate", lambda: b"abc".translate(None, **{"delete": b"b"}))
        # MIXED WITH A WRITTEN KEYWORD, in source order: a later key wins
        # over one a `**` brought, which is how CPython reads the two.
        show("mixed", lambda: "a,b,c".split(",", maxsplit=1, **{}))
        show("map first", lambda: "a,b,c".split(",", **{}, maxsplit=1))
        show("two maps",
             lambda: "aaa".replace("a", "b", **{"count": 1}, **{}))
        # AND THE EMPTY MAPPING, which is what a forwarding wrapper passes
        # nearly always, still reaches the positional call.
        show("empty", lambda: "aaa".replace("a", "b", **{}))
        show("empty plain", lambda: "abc".upper(**{}))
        show("empty translate", lambda: b"abc".translate(**{}))
        # THE REFUSALS, in CPython's order: too many first, then a slot given
        # twice, then a name no parameter has -- with CPython's own edit
        # distance deciding whether there is a suggestion to make.
        show("too many", lambda: "a,b".split(",", 1, **{"maxsplit": 2}))
        show("too many mixed",
             lambda: "a,b,c".split(",", maxsplit=1, **{"sep": ","}))
        show("keyword count",
             lambda: "a".encode(**{"encoding": "a", "errors": "b", "x": "c"}))
        show("twice", lambda: "a".encode("utf-8", **{"encoding": "ascii"}))
        show("unknown", lambda: "aaa".replace("a", "b", **{"nope": 1}))
        show("suggested", lambda: "aaa".replace("a", "b", **{"coun": 1}))
        show("positional only", lambda: "aaa".replace(**{"old": "a"}))
        show("non-str key", lambda: "aaa".replace("a", "b", **{1: 2}))
        # A METHOD WITH NO KEYWORD SIGNATURE names its owner, and the owner
        # is whatever the receiver turned out to be.
        show("str", lambda: "abc".upper(**{"x": 1}))
        show("bytes", lambda: b"abc".upper(**{"x": 1}))
        show("list", lambda: [1].append(**{"x": 1}))
        show("dict", lambda: {}.keys(**{"x": 1}))
        show("int", lambda: (5).bit_length(**{"x": 1}))
        # AND A USER METHOD IS UNTOUCHED by any of it. Its name is its own:
        # a class writing `replace` puts the call on the COLLISION path,
        # where the builtin half still drops what it was given -- which is a
        # divergence of its own and not this one.
        class C:
            def shuffled(self, a, count=0):
                return (a, count)
        show("user", lambda: C().shuffled("a", **{"count": 3}))
        show("user empty", lambda: C().shuffled("a", **{}))
    """,
    # A NAME COLLISION DROPPED THE BUILTIN HALF'S KEYWORDS. A module that
    # defines its own `split` puts every `x.split(...)` on the two-way
    # dispatch, where the receiver decides at run time which half is meant --
    # and the builtin half had nowhere to put the keywords, so
    # `"a,b,c".split(",", maxsplit=1)` answered THREE pieces and
    # `"a,b".split(",", nope=1)` answered two rather than reporting the name.
    # Both are wrong answers with nothing to mark them.
    #
    # WHAT STOOD IN THE WAY was double evaluation: the user half lowers the
    # keywords itself, so folding for the builtin half as well ran every
    # argument twice. They are lowered ONCE now, before the branch, and each
    # arm is handed registers -- the user arm through a `_Given` node that
    # gives the register straight back.
    #
    # THE ARRANGEMENT BELONGS INSIDE THE BUILTIN BLOCK, refusals and all: a
    # keyword the builtin cannot take may be exactly the one the user method
    # wants, which is the whole reason the call is in two halves.
    #
    # A CLASS EXTENDING A BUILTIN puts EVERY method name on this path, which
    # is why the cases below are not only the two the classes name.
    "a_colliding_name_keeps_both_halves_keywords": """
        class Thing:
            def split(self, sep, maxsplit=-1):
                return ("user split", sep, maxsplit)
            def encode(self, encoding="x", errors="y"):
                return ("user encode", encoding, errors)
            def sort(self, key=None, reverse=False):
                return ("user sort", reverse)
            def update(self, other=None, **kw):
                return ("user update", other, sorted(kw.items()))

        class L(list):
            pass

        class D(dict):
            pass

        def show(label, f):
            try:
                print(label, repr(f()))
            except TypeError as e:
                print(label, "TypeError:", e)

        # THE BUILTIN HALF, which is the half that was losing them.
        show("split", lambda: "a,b,c".split(",", maxsplit=1))
        show("split positional", lambda: "a,b,c".split(",", 1))
        show("rsplit", lambda: "a,b,c".rsplit(",", maxsplit=1))
        show("replace", lambda: "aaa".replace("a", "b", count=1))
        show("encode", lambda: "a".encode(encoding="utf-8"))
        show("encode both",
             lambda: "a\\u00ff".encode(encoding="ascii", errors="replace"))
        show("decode", lambda: b"a".decode(errors="replace"))
        show("expandtabs", lambda: "a\\tb".expandtabs(tabsize=2))
        show("splitlines", lambda: "a\\nb".splitlines(keepends=True))
        show("mapping", lambda: "a,b,c".split(",", **{"maxsplit": 1}))
        # AND ITS REFUSALS, which used to be silent acceptance.
        show("unknown", lambda: "a,b".split(",", nope=1))
        show("suggested", lambda: "a,b".split(",", masplit=1))
        show("twice", lambda: "a,b".split(",", 1, maxsplit=2))
        show("no keywords at all", lambda: "abc".upper(x=1))
        # THE USER HALF is untouched, keywords and all.
        show("user split", lambda: Thing().split(",", maxsplit=1))
        show("user positional", lambda: Thing().split(",", 1))
        show("user encode", lambda: Thing().encode(encoding="z"))
        show("user mapping", lambda: Thing().split(",", **{"maxsplit": 1}))
        # THE TWO WITH KEYWORDS OF THEIR OWN take the branch they always did.
        show("user sort", lambda: Thing().sort(reverse=True))
        show("user update", lambda: Thing().update(x=1))
        v = [1, 3, 2]
        v.sort(reverse=True)
        print("list sort", v)
        d = {"a": 1}
        d.update(b=2)
        print("dict update", sorted(d.items()))
        # A CLASS EXTENDING A BUILTIN reaches the builtin through the
        # instance, which is the other way into this dispatch.
        w = L([1, 3, 2])
        w.sort(reverse=True)
        print("extended sort", list(w))
        e = D(a=1)
        e.update(b=2)
        print("extended update", sorted(e.items()))
        print("extended pop", L([1, 2, 3]).pop(0))
        # AND EVERY ARGUMENT RUNS ONCE. Lowering both halves is what this
        # change is made of, and running a keyword's expression twice is the
        # way it would go wrong without being visible in any answer above.
        box = []
        print("once", "a,b,c".split(",", maxsplit=(box.append(1) or 1)),
              len(box))

        def rows():
            yield "a,b,c".split(",", maxsplit=1)
            yield Thing().split(",", maxsplit=1)

        print("in a generator", list(rows()))
    """,
    # A BOUND BUILTIN METHOD REACHED AS A VALUE REFUSED ITS KEYWORDS.
    # `f = x.split` then `f(",", maxsplit=1)` dropped the keyword in the
    # interpreter and reported `got an unexpected keyword argument` on both
    # compiled paths -- a native declares no parameters, so the keyword
    # binder had nothing to match the names against.
    #
    # IT REACHED ORDINARY CODE through the collision path: a module defining
    # a class that extends a builtin puts EVERY method name on the two-way
    # dispatch, and an int's `to_bytes` arrives at the value binder rather
    # than at the frontend's own fold. `(5).to_bytes(2, byteorder="big")` was
    # refused for that reason alone.
    #
    # THE NAMES ARE THE SAME ONES the written spelling folds against.
    # `METHOD_PARAMS` is read by the host directly and generated into the C
    # as `apy_kind_meth_sign`, so the two arrangements cannot drift -- and
    # the refusals are CPython's, in CPython's order.
    "a_bound_builtin_method_takes_its_keywords": """
        class L(list):
            pass

        def show(label, f):
            try:
                print(label, repr(f()))
            except TypeError as e:
                print(label, "TypeError:", e)

        def call(f, *args, **kw):
            return f(*args, **kw)

        # AS A VALUE, which is what a forwarding wrapper holds.
        show("replace", lambda: (lambda f: f("a", "b", count=1))("aaa".replace))
        show("split", lambda: (lambda f: f(",", maxsplit=1))("a,b,c".split))
        show("split gap", lambda: (lambda f: f(maxsplit=1))("a b c".split))
        show("split plain", lambda: (lambda f: f(","))("a,b,c".split))
        show("split none", lambda: (lambda f: f())("a b c".split))
        show("encode", lambda: (lambda f: f(encoding="utf-8"))("a".encode))
        show("decode", lambda: (lambda f: f(errors="replace"))(b"a".decode))
        show("expandtabs",
             lambda: (lambda f: f(tabsize=2))("a\\tb".expandtabs))
        show("splitlines",
             lambda: (lambda f: f(keepends=True))("a\\nb".splitlines))
        show("to_bytes", lambda: (lambda f: f(2, byteorder="big"))(
            (5).to_bytes))
        show("to_bytes named", lambda: (lambda f: f(length=2, signed=False))(
            (5).to_bytes))
        # THROUGH A WRAPPER, which is the shape the whole thing exists for.
        show("forwarded", lambda: call("aaa".replace, "a", "b", count=1))
        show("forwarded map",
             lambda: call("a,b,c".split, ",", **{"maxsplit": 1}))
        show("forwarded plain", lambda: call("aaa".upper))
        # AND WRITTEN OUT, which the collision path sends the same way
        # because `L` extends a builtin.
        print("written", (5).to_bytes(2, byteorder="big"),
              "a,b,c".split(",", maxsplit=1))
        print("extended", L([3, 1]).pop(0))
        # THE REFUSALS, in CPython's order.
        show("too many", lambda: (lambda f: f(",", 1, maxsplit=2))(
            "a,b".split))
        show("twice", lambda: (lambda f: f("utf-8", encoding="ascii"))(
            "a".encode))
        show("unknown", lambda: (lambda f: f(",", nope=1))("a,b".split))
        show("no keywords", lambda: (lambda f: f(x=1))("a".upper))
        show("no keywords list", lambda: (lambda f: f(x=1))([1].append))
        show("no keywords dict", lambda: (lambda f: f(x=1))({}.keys))
        # AND A FIXED-ARITY SLOT still counts its arguments.
        show("len", lambda: (lambda f: f())([1, 2].__len__))
        show("len surplus", lambda: (lambda f: f(9))([1, 2].__len__))
        show("round", lambda: (lambda f: f(1))((1.23).__round__))
        show("round none", lambda: (lambda f: f())((1.23).__round__))
        show("find", lambda: (lambda f: f("b"))("abcabc".find))
        show("find window", lambda: (lambda f: f("b", 2, 6))("abcabc".find))
    """,
    # A FLOAT INDEX WAS ACCEPTED WHERE CPYTHON REFUSES IT. `[1, 2].pop(1.0)`
    # answered 2 in the interpreter and `pop index out of range` when
    # compiled: the conversion behind it read the PAYLOAD of whatever it was
    # handed, so the bits of a double became a position. One check, in the
    # one place the C and the machine subset share, refuses it and asks a
    # user object for its `__index__` instead.
    #
    # AND THE MESSAGES ARE THREE, not one with a substituted noun. CPython
    # says `list indices must be integers or slices, not float` for a list or
    # a tuple, `string indices must be integers, not 'float'` for a str with
    # the kind quoted and no mention of slices, and `byte` -- singular -- for
    # a bytes. The C had the pair and the interpreter did not; a bytes
    # subscript reached neither and answered `'bytes' object is not
    # subscriptable`, a sentence about the receiver for a complaint about the
    # subscript.
    #
    # A BOUND IS A SLICE INDEX AND SAYS SO: `"abc".find("b", 1.0)` names
    # `__index__` and mentions None, and a sequence's `index` names
    # `__index__` and does not -- which is the only thing telling the two
    # apart.
    "a_float_is_not_an_index": """
        def show(label, f):
            try:
                print(label, repr(f()))
            except (TypeError, IndexError, OverflowError) as e:
                print(label, type(e).__name__ + ":", e)

        # NAMED, so the literal forms do not become a SyntaxWarning CPython
        # emits and this compiler does not.
        h = 1.0
        word = "x"
        nothing = None
        huge = 2 ** 100

        class Two:
            def __index__(self):
                return 1

        show("list pop", lambda: [1, 2].pop(h))
        show("bytearray pop", lambda: bytearray(b"ab").pop(h))
        show("list subscript", lambda: [1, 2][h])
        show("tuple subscript", lambda: (1, 2)[h])
        show("str subscript", lambda: "ab"[h])
        show("bytes subscript", lambda: b"ab"[h])
        show("bytearray subscript", lambda: bytearray(b"ab")[h])
        show("by a str", lambda: [1, 2][word])
        show("by None", lambda: [1, 2][nothing])
        show("str by a str", lambda: "ab"[word])
        show("insert", lambda: (lambda v: (v.insert(h, 9), v)[1])([1, 2]))
        show("setitem",
             lambda: (lambda v: (v.__setitem__(h, 9), v)[1])([1, 2]))
        show("delitem", lambda: (lambda v: (v.__delitem__(h), v)[1])([1, 2]))
        show("index bound", lambda: [1, 2].index(1, h))
        show("find bound", lambda: "abc".find("b", h))
        show("slice bound", lambda: [1, 2, 3][h:2])
        show("range", lambda: range(h))
        show("repeat", lambda: [1] * h)
        show("ljust", lambda: "a".ljust(h))
        # A DICT IS KEYED AND NOT INDEXED, so 1.0 finds the key 1.
        show("dict subscript", lambda: {1: "a"}[h])
        # AN INDEX TOO WIDE FOR A MACHINE WORD is refused rather than
        # answered: the compiled halves cannot hold one, and the interpreter
        # used to disagree by reporting the range instead.
        show("huge", lambda: [1, 2][huge])
        show("huge pop", lambda: [1, 2].pop(huge))
        # AND WHAT IS STILL AN INDEX stays one. A bool is an integer, and a
        # class saying `__index__` is saying it IS one.
        show("bool", lambda: [1, 2][True])
        show("__index__", lambda: [1, 2][Two()])
        show("__index__ pop", lambda: [1, 2].pop(Two()))
        show("ordinary", lambda: ([1, 2][1], "ab"[1], b"ab"[1],
                                  bytearray(b"ab")[1], (1, 2)[0]))
        show("out of range", lambda: [1, 2][9])
        show("tuple out of range", lambda: (1, 2)[9])
        show("str out of range", lambda: "ab"[9])
    """,
    # ENCODING A STRING HOLDING A SURROGATE SPLIT THREE WAYS. CPython gives
    # seven different answers for `"a\udcffb".encode("utf-8", errors)`; the
    # interpreter refused the first three and both compiled runtimes handed
    # back the raw WTF-8 bytes for ALL of them, which is six wrong answers
    # wearing one shape. The cell has held a surrogate as WTF-8 since the
    # surrogate-literal round, so the encoder finally has something real to
    # work from.
    #
    # EVERY HANDLER CPYTHON ANSWERS WITHOUT A REGISTERED CALLBACK is here --
    # `namereplace` is the one left out, because it spells a character by its
    # Unicode NAME and the name table is a bundled module rather than
    # something the runtime carries.
    #
    # AND THE MESSAGES ARE WHOLE SENTENCES NOW. They used to stop after
    # `can't encode character`, saying nothing about which character, where,
    # or why. CPython names all three, reports a RUN of them together, and
    # from the other side counts MAXIMAL SUBPARTS: a truncated four-byte
    # sequence is one error and a WTF-8 surrogate is three, because `ED`
    # accepts only `80..9F` and the bytes after it start nothing.
    "a_surrogate_survives_every_error_handler": """
        def show(label, f):
            try:
                print(label, repr(f()))
            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                print(label, type(e).__name__ + ":", e)

        low = "a\\udcffb"
        high = "a\\ud800b"
        handlers = ("strict", "replace", "ignore", "backslashreplace",
                    "xmlcharrefreplace", "surrogateescape", "surrogatepass")
        for how in handlers:
            show("encode " + how, lambda h=how: low.encode("utf-8", h))
        show("encode default", lambda: low.encode("utf-8"))
        show("high pass", lambda: high.encode("utf-8", "surrogatepass"))
        show("high escape", lambda: high.encode("utf-8", "surrogateescape"))
        show("escape out of range",
             lambda: "a\\udc00b".encode("utf-8", "surrogateescape"))
        # THE CELL IS UNCHANGED BY ANY OF IT: the surrogate is one character
        # and reads back as one.
        print("held", len(low), ord(low[1]), repr(low))
        # THE NARROW CODECS take the same seven.
        for how in handlers:
            show("ascii " + how, lambda h=how: "caf\\u00e9".encode("ascii", h))
        show("ascii default", lambda: "caf\\u00e9".encode("ascii"))
        show("latin-1 wide", lambda: "\\u0100".encode("latin-1"))
        show("latin-1 escape", lambda: low.encode("latin-1", "surrogateescape"))
        show("emoji backslash",
             lambda: "\\U0001F600".encode("ascii", "backslashreplace"))
        show("emoji xml",
             lambda: "\\U0001F600".encode("ascii", "xmlcharrefreplace"))
        # A RUN OF UNENCODABLE CHARACTERS IS ONE COMPLAINT, naming a range
        # and none of them; one on its own is named.
        show("run", lambda: "a\\udcff\\udcffb".encode("utf-8"))
        show("ascii run", lambda: "\\u00e9\\u00e9x".encode("ascii"))
        show("ascii apart", lambda: "\\u00e9x\\u00e9".encode("ascii"))
        # AND FROM THE OTHER SIDE. A byte string spelling a surrogate is not
        # UTF-8, whatever the cell does internally.
        pairs = [(b"a\\xed\\xb3\\xbfb", "wtf8"), (b"a\\xffb", "lone byte"),
                 (b"a\\xc3", "truncated 2"), (b"\\xf0\\x9f\\x98", "truncated 4"),
                 (b"a\\xc3(", "bad continuation"), (b"\\x80", "bad start"),
                 (b"\\xc0\\x80", "overlong"), (b"\\xe0\\x80", "short range"),
                 (b"\\xf4\\x90\\x80\\x80", "past the top")]
        for raw, name in pairs:
            for how in ("strict", "replace", "ignore", "surrogateescape",
                        "surrogatepass"):
                show(name + " " + how,
                     lambda r=raw, h=how: r.decode("utf-8", h))
        show("ascii decode", lambda: b"a\\xffb".decode("ascii"))
        show("ascii decode escape",
             lambda: b"a\\xffb".decode("ascii", "surrogateescape"))
        show("ascii decode replace",
             lambda: b"a\\xffb".decode("ascii", "replace"))
        # THE ROUND TRIP IS THE POINT OF PEP 383: a byte that would not
        # decode comes back out as that byte.
        show("roundtrip", lambda: b"a\\xffb".decode("utf-8", "surrogateescape")
             .encode("utf-8", "surrogateescape"))
        show("roundtrip wide",
             lambda: b"\\xf0\\x9f\\x98".decode("utf-8", "surrogateescape")
             .encode("utf-8", "surrogateescape"))
        # AND ORDINARY TEXT IS UNTOUCHED by every one of them.
        for how in handlers:
            show("plain " + how, lambda h=how: "caf\\u00e9".encode("utf-8", h))
        print("empty", "".encode("utf-8"), b"".decode("utf-8"))
        print("round", "caf\\u00e9".encode("utf-8").decode("utf-8"))
    """,
    # `type(C) is type` WAS FALSE while `print(type(C))` said
    # `<class 'type'>` -- the worst pair of answers to have, because the
    # printed one says the identity question was already settled. There were
    # THREE objects named `type`: the one the bare word evaluates to, one the
    # name-keyed table built for a class, and one an attribute read built
    # again. Which of them a program got depended on how it asked.
    #
    # A BUILTIN TYPE IS A CLASS TOO. The canonical thunk standing for one is
    # a FUNC carrying `is_type` rather than a TYPE cell, so it fell past the
    # class arm entirely and `type(int) is type` was False on every path.
    #
    # AND A CLASS HAS A `__class__`, which every other kind already answered:
    # it was an AttributeError about the one attribute Python guarantees, and
    # `isinstance(x, C.__class__)` is how a program asks.
    "one_type_object_answers_every_way_of_asking": """
        class C:
            pass

        class M(type):
            pass

        class D(metaclass=M):
            pass

        print("user", type(C) is type, C.__class__ is type)
        print("builtin", type(int) is type, int.__class__ is type)
        print("value", type(5).__class__ is type, type("").__class__ is type)
        print("itself", type is type, type(type) is type)
        print("metaclass", type(D) is M, D.__class__ is M, type(M) is type)
        print("names", type(C).__name__, type(int).__name__,
              C.__class__.__name__)
        print("shown", type(C), type(int), C.__class__)
        # ONE OBJECT, however it was reached -- which is what `is` asks and
        # what `id` measures.
        print("one", id(type(C)) == id(type), id(type(int)) == id(type),
              id(C.__class__) == id(type(C)))
        # AND THE QUESTIONS BUILT ON IT.
        print("isinstance", isinstance(C, type), isinstance(int, type),
              isinstance(5, int), isinstance(C(), C))
        print("issubclass", issubclass(M, type), issubclass(C, object))
        print("base", M.__base__ is type)
    """,
    # THE TEXT GATE ASKED "IS THIS STR OR BYTES" AND NOT "against WHAT". So
    # either kind passed for either receiver: `"abc".find(b"a")` answered 0
    # and `b"abc".find("a")` answered 0, two WRONG ANSWERS where CPython
    # refuses and the interpreter already did. And a MEMORYVIEW -- which is
    # bytes-like, and the reason the gate has to see the receiver at all --
    # was refused by every one of them.
    #
    # THE REFUSALS ARE RECEIVER-DEPENDENT, in CPython's own words: the
    # searches say `argument should be integer or bytes-like object` because
    # an INTEGER is a legal needle for them, everything else says `a
    # bytes-like object is required`, and `startswith` names the kind of
    # tuple it wanted.
    #
    # A VIEW ALSO HANDS ITS CONTENTS OVER, which is the reason a program
    # makes one: `tobytes` and `tolist` did not exist, so a view could be
    # indexed and sliced and never emptied. And it hashes as the bytes it
    # views -- as, it turns out, BYTES THEMSELVES did not in the C runtime,
    # which hashed every one of them by ADDRESS.
    "a_memoryview_is_bytes_like_and_a_str_is_not": """
        def show(label, f):
            try:
                print(label, repr(f()))
            except TypeError as e:
                print(label, "TypeError:", e)

        m = memoryview(b"abc")
        one = memoryview(b"a")
        # A BYTES RECEIVER TAKES A VIEW wherever it takes bytes.
        show("find", lambda: b"xabcx".find(m))
        show("rfind", lambda: b"xabcx".rfind(m))
        show("index", lambda: b"xabcx".index(m))
        show("count", lambda: b"xabcx".count(m))
        show("startswith", lambda: b"abcx".startswith(m))
        show("endswith", lambda: b"xabc".endswith(m))
        show("startswith tuple", lambda: b"abcx".startswith((m, b"z")))
        show("split", lambda: b"1abc2".split(m))
        show("rsplit", lambda: b"1abc2".rsplit(m))
        show("replace", lambda: b"xabcx".replace(m, b"Z"))
        show("removeprefix", lambda: b"abcx".removeprefix(m))
        show("removesuffix", lambda: b"xabc".removesuffix(m))
        show("join", lambda: b",".join([m, m]))
        show("contains", lambda: m in b"xabcx")
        show("constructors", lambda: (bytes(m), bytearray(m)))
        # A STR RECEIVER TAKES NONE OF IT, and says so as CPython does.
        show("str find", lambda: "abc".find(m))
        show("str replace", lambda: "abc".replace(m, "z"))
        show("str startswith", lambda: "abc".startswith(m))
        # AND NEITHER KIND MAY STAND IN FOR THE OTHER.
        show("str find bytes", lambda: "abc".find(b"a"))
        show("bytes find str", lambda: b"abc".find("a"))
        show("str startswith bytes", lambda: "abc".startswith(b"a"))
        show("bytes startswith str", lambda: b"abc".startswith("a"))
        show("str tuple bytes", lambda: "abc".startswith((b"a",)))
        show("bytes tuple str", lambda: b"abc".startswith(("a",)))
        show("str replace bytes", lambda: "abc".replace(b"a", b"z"))
        show("bytes replace str", lambda: b"abc".replace("a", "z"))
        show("str split bytes", lambda: "a,b".split(b","))
        show("bytes split str", lambda: b"a,b".split(","))
        show("str prefix bytes", lambda: "abc".removeprefix(b"a"))
        show("bytes prefix str", lambda: b"abc".removeprefix("a"))
        show("str join bytes", lambda: ",".join([b"a"]))
        show("bytes join str", lambda: b",".join(["a"]))
        show("str in bytes", lambda: "a" in b"abc")
        show("bytes in str", lambda: b"a" in "abc")
        # A BYTEARRAY IS BYTES-LIKE TOO and always was.
        show("bytearray arg", lambda: b"xabc".find(bytearray(b"abc")))
        show("bytearray join", lambda: b",".join([bytearray(b"a"), b"b"]))
        # WHAT A VIEW ANSWERS ABOUT ITSELF.
        print("view", len(m), m[1], bytes(m[1:]), m.tobytes(), m.tolist())
        print("same", m == b"abc", hash(m) == hash(b"abc"))
        print("one", one.tobytes(), one.tolist(), bytes(one))
        # AND EQUAL BYTES HASH EQUAL, which the C half did not do at all:
        # every bytes value fell through to its ADDRESS.
        made = b"ab" + b"c"
        print("content", b"abc" == made, hash(b"abc") == hash(made),
              hash("abc") == hash("ab" + "c"))
    """,
    # A DUNDER A BUILTIN TYPE WRITES OUT READ AS `method-wrapper` HERE.
    # `type([1, 2].__getitem__).__name__` is `builtin_function_or_method` in
    # CPython: a list DEFINES that method where a tuple fills a slot with
    # it, and nothing in either signature says which. The rule in place was
    # the ARITY -- a dunder with a range is written out -- which is a proxy,
    # and it got twenty-seven pairs wrong.
    #
    # THE ANSWER IS READ OUT OF CPYTHON, per kind and per name, by the
    # generator that already asks which kinds own which names. It has to be
    # per KIND as well as per name: `__contains__` and `__getitem__` really
    # do differ between a dict and a tuple.
    #
    # THE PAIRS BELOW ARE THE ONES THIS RUNTIME HAS. Nineteen dunder names
    # are missing from every builtin value and are a gap of their own, not
    # this one.
    "a_written_out_dunder_is_not_a_slot": """
        def show(label, f):
            try:
                print(label, f())
            except AttributeError as e:
                print(label, "AttributeError:", e)

        for name in ('__add__', '__buffer__', '__class__', '__contains__', '__delitem__', '__eq__', '__format__', '__ge__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__imul__', '__iter__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__repr__', '__rmul__', '__setitem__', '__str__'):
            print("bytearray", name,
                  type(getattr(bytearray(), name)).__name__)
        for name in ('__add__', '__buffer__', '__class__', '__contains__', '__eq__', '__format__', '__ge__', '__getitem__', '__gt__', '__hash__', '__iter__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__repr__', '__rmul__', '__str__'):
            print("bytes", name,
                  type(getattr(b"", name)).__name__)
        for name in ('__abs__', '__add__', '__bool__', '__class__', '__complex__', '__eq__', '__format__', '__ge__', '__gt__', '__hash__', '__le__', '__lt__', '__mul__', '__ne__', '__neg__', '__pos__', '__pow__', '__radd__', '__repr__', '__rmul__', '__rpow__', '__rsub__', '__rtruediv__', '__str__', '__sub__', '__truediv__'):
            print("complex", name,
                  type(getattr(1j, name)).__name__)
        for name in ('__class__', '__contains__', '__delitem__', '__eq__', '__format__', '__ge__', '__getitem__', '__gt__', '__hash__', '__ior__', '__iter__', '__le__', '__len__', '__lt__', '__ne__', '__or__', '__repr__', '__reversed__', '__ror__', '__setitem__', '__str__'):
            print("dict", name,
                  type(getattr({}, name)).__name__)
        for name in ('__abs__', '__add__', '__bool__', '__ceil__', '__class__', '__divmod__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__gt__', '__hash__', '__int__', '__le__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__pos__', '__pow__', '__radd__', '__rdivmod__', '__repr__', '__rfloordiv__', '__rmod__', '__rmul__', '__round__', '__rpow__', '__rsub__', '__rtruediv__', '__str__', '__sub__', '__truediv__', '__trunc__'):
            print("float", name,
                  type(getattr(1.5, name)).__name__)
        for name in ('__and__', '__class__', '__contains__', '__eq__', '__format__', '__ge__', '__gt__', '__hash__', '__iter__', '__le__', '__len__', '__lt__', '__ne__', '__or__', '__rand__', '__repr__', '__ror__', '__rsub__', '__rxor__', '__str__', '__sub__', '__xor__'):
            print("frozenset", name,
                  type(getattr(frozenset(), name)).__name__)
        for name in ('__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__divmod__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__gt__', '__hash__', '__index__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__str__', '__sub__', '__truediv__', '__trunc__', '__xor__'):
            print("int", name,
                  type(getattr(5, name)).__name__)
        for name in ('__add__', '__class__', '__contains__', '__delitem__', '__eq__', '__format__', '__ge__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__imul__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__repr__', '__reversed__', '__rmul__', '__setitem__', '__str__'):
            print("list", name,
                  type(getattr([], name)).__name__)
        for name in ('__bool__', '__class__', '__contains__', '__eq__', '__format__', '__ge__', '__getitem__', '__gt__', '__hash__', '__iter__', '__le__', '__len__', '__lt__', '__ne__', '__repr__', '__reversed__', '__str__'):
            print("range", name,
                  type(getattr(range(3), name)).__name__)
        for name in ('__and__', '__class__', '__contains__', '__eq__', '__format__', '__ge__', '__gt__', '__hash__', '__iand__', '__ior__', '__isub__', '__iter__', '__ixor__', '__le__', '__len__', '__lt__', '__ne__', '__or__', '__rand__', '__repr__', '__ror__', '__rsub__', '__rxor__', '__str__', '__sub__', '__xor__'):
            print("set", name,
                  type(getattr(set(), name)).__name__)
        for name in ('__add__', '__class__', '__contains__', '__eq__', '__format__', '__ge__', '__getitem__', '__gt__', '__hash__', '__iter__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__repr__', '__rmul__', '__str__'):
            print("str", name,
                  type(getattr("", name)).__name__)
        for name in ('__add__', '__class__', '__contains__', '__eq__', '__format__', '__ge__', '__getitem__', '__gt__', '__hash__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__repr__', '__rmul__', '__str__'):
            print("tuple", name,
                  type(getattr((), name)).__name__)
    """,
    "a_builtin_value_carries_what_object_hands_down": """
        def show(label, f):
            try:
                print(label, f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        # THE TWELVE `object` HANDS DOWN, on a builtin value rather
        # than an instance. Every one was an AttributeError about an
        # attribute Python guarantees to every object there is.
        names = ('__init__', '__init_subclass__', '__getstate__',
                 '__subclasshook__',
                 '__dir__', '__sizeof__', '__new__', '__getattribute__',
                 '__setattr__', '__delattr__', '__reduce__', '__reduce_ex__')
        for sample in ("", b"", bytearray(), [], (), {}, set(), frozenset(),
                       5, 1.5, range(3), 1j):
            print(type(sample).__name__,
                  [n for n in names if hasattr(sample, n)] == list(names))
        show("init", lambda: [].__init__())
        show("init_subclass", lambda: (5).__init_subclass__())
        show("getstate", lambda: "a".__getstate__())
        show("subclasshook", lambda: [].__subclasshook__(int))
        show("new", lambda: [].__new__(list))
        show("getattribute", lambda: "abc".__getattribute__("upper")())
        show("getattribute miss", lambda: "abc".__getattribute__("nope"))
        show("setattr", lambda: [].__setattr__("a", 1))
        show("delattr", lambda: {}.__delattr__("a"))
        show("reduce", lambda: "a".__reduce__())
        show("dir answers a list", lambda: type("a".__dir__()).__name__)
        # `__sizeof__` IS AN IMPLEMENTATION NUMBER -- CPython's own differs between
        # builds -- so what is checked is what a program can rely on: an int, and one
        # that grows with what the value holds.
        show("sizeof", lambda: (type("abc".__sizeof__()).__name__,
                                "abcd".__sizeof__() > "a".__sizeof__()))
    """,
    "a_kind_carries_the_dunders_only_it_has": """
        def show(label, f):
            try:
                print(label, f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        # WHAT `copy` AND `pickle` REBUILD A VALUE FROM. Every immutable builtin
        # answers a COPY of itself -- a complex answers its two halves as floats,
        # and a tuple is the one kind that shares -- and a mutable one does not
        # carry the attribute at all.
        show("str", lambda: "ab".__getnewargs__())
        show("bytes", lambda: b"ab".__getnewargs__())
        show("tuple", lambda: (1, 2).__getnewargs__())
        show("int", lambda: (5).__getnewargs__())
        show("float", lambda: (1.5).__getnewargs__())
        show("complex", lambda: (1 + 2j).__getnewargs__())
        # A COPY AND NOT THE RECEIVER, which is the whole of what these bodies do
        # and what every path here used to skip. The four that answer True are the
        # ones whose copy lands back in a shared cell -- the empty str, the 256
        # one-octet bytes and the small ints go through the same constructor every
        # other one does -- plus the tuple, which is shared outright.
        print([v.__getnewargs__()[0] is v
               for v in ("ab", b"ab", (1, 2), 5, 10 ** 30, 1.5, True,
                         "", b"", b"a", "a")])
        # AND A bool ANSWERS THE int 1: `_PyLong_Copy` builds an integer out of the
        # bool's value, so the kind changes and the `True` object is not the answer.
        one = 5 - 4
        print(True.__getnewargs__(), type(True.__getnewargs__()[0]).__name__,
              True.__getnewargs__()[0] is one)
        print([hasattr(v, "__getnewargs__")
               for v in (bytearray(), [], {}, set(), range(3))])
        # THE REFLECTED `%`, which text carries and always refuses -- the TypeError a
        # program sees is the operator's, raised after this answers.
        print("ab".__rmod__(1), b"ab".__rmod__(1), bytearray(b"ab").__rmod__(1))
        # `bytes(x)` ASKS `x` FOR ITSELF FIRST, and bytes is the kind that answers.
        print(b"ab".__bytes__(), hasattr(bytearray(), "__bytes__"))
        # THE ONE BYTEARRAY INTERNAL A PROGRAM CAN READ.
        print(bytearray(b"abc").__alloc__(), bytearray().__alloc__(),
              hasattr(b"", "__alloc__"))
        show("format double", lambda: (1.5).__getformat__("double"))
        show("format bad", lambda: (1.5).__getformat__("x"))
        print(hasattr(5, "__getformat__"))
        # `list[int]` REACHED BY NAME rather than written as a subscript.
        print([].__class_getitem__(int), {}.__class_getitem__(str),
              ().__class_getitem__(int).__args__)
        print([hasattr(v, "__class_getitem__") for v in ("", b"", 5, 1.5)])
        # PEP 688's OTHER HALF, and the two arguments it refuses.
        held = bytearray(b"abc")
        print(held.__release_buffer__(memoryview(held)))
        show("release none", lambda: held.__release_buffer__(None))
        show("release other",
             lambda: held.__release_buffer__(memoryview(bytearray(b"x"))))
        print(hasattr(b"", "__release_buffer__"))
    """,
    "a_builtin_method_cpython_has_is_here_too": """
        def show(label, f):
            try:
                print(label, f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        # THE THREE IN-PLACE SET UPDATES, which a frozenset does not have. Each takes
        # any iterable where the operator behind it demands a set, and each answers
        # None because it changed the set every other name for it also sees.
        def ran(op, arg):
            s = {1, 2, 3}
            got = getattr(s, op)(arg)
            return got, sorted(s)
        show("difference_update", lambda: ran("difference_update", {2}))
        show("difference_update list", lambda: ran("difference_update", [2, 3]))
        show("intersection_update", lambda: ran("intersection_update", {2, 3, 9}))
        show("intersection_update list", lambda: ran("intersection_update", [1]))
        show("symmetric_difference_update", lambda: ran("symmetric_difference_update", {3, 4}))
        show("emptied by itself", lambda: ran("difference_update", {1, 2, 3}))
        show("unhashable", lambda: ran("difference_update", [[1]]))
        print([hasattr(frozenset(), n) for n in ("difference_update",
                                                 "intersection_update",
                                                 "symmetric_difference_update")])
        # `s.format_map(m)` -- `format` with the mapping handed over whole. CPython
        # does not check the argument, it SUBSCRIPTS it, so a non-mapping is refused
        # in a subscript's words and not a signature's.
        show("format_map", lambda: "{a}-{b}".format_map({"a": 1, "b": 2}))
        show("format_map missing", lambda: "{a}".format_map({}))
        show("format_map int", lambda: "{a}".format_map(5))
        show("format_map list", lambda: "{a}".format_map([1]))
        # 3.14's `resize`: growing fills with NUL and shrinking truncates.
        def resized(start, n):
            b = bytearray(start)
            got = b.resize(n)
            return got, bytes(b)
        show("resize grow", lambda: resized(b"ab", 4))
        show("resize shrink", lambda: resized(b"abcd", 2))
        show("resize to nothing", lambda: resized(b"ab", 0))
        show("resize negative", lambda: resized(b"ab", -1))
        show("resize str", lambda: resized(b"ab", "x"))
        # `bytearray.fromhex` ANSWERS THE KIND IT WAS REACHED THROUGH, which is not
        # the bytes the shared reading gives.
        show("bytearray.fromhex", lambda: bytearray.fromhex("41 42"))
        show("bytes.fromhex", lambda: bytes.fromhex("41 42"))
        # A RATIONAL'S TWO HALVES, which an int carries because it IS one. `True`
        # converts on these and not on `real`, which CPython does too.
        show("int", lambda: ((5).numerator, (5).denominator))
        show("bool", lambda: (True.numerator, True.denominator, True.real))
        show("big", lambda: (2 ** 100).numerator == 2 ** 100)
        print([hasattr(1.5, "numerator"), hasattr(1j, "numerator")])
        # And None writes `__bool__` out rather than being read through `__len__`.
        show("None", lambda: (None.__bool__(), type(None.__bool__()).__name__))
        print([type(getattr(v, n)).__name__ for v, n in
               (({1}, "difference_update"), ("", "format_map"),
                (bytearray(), "resize"), (None, "__bool__"))])
    """,
    "a_view_says_what_shape_its_buffer_is": """
        # WHAT SHAPE A BUFFER IS, which a program asks before it indexes one:
        # `m.ndim == 1` is the everyday check, and it was an AttributeError about
        # the answer it wanted.
        whole = memoryview(bytearray(b"abcd"))
        print(whole.ndim, whole.shape, whole.strides, whole.suboffsets)
        print(whole.c_contiguous, whole.f_contiguous, whole.contiguous)
        # A SLICE WITH A STEP IS NOT CONTIGUOUS -- it skips bytes -- and its stride
        # says by how many. Both orders agree for a single dimension, which is why
        # C and Fortran answer alike.
        stepped = whole[::2]
        print(stepped.ndim, stepped.shape, stepped.strides)
        print(stepped.c_contiguous, stepped.f_contiguous, stepped.contiguous)
        print(stepped.nbytes, stepped.tolist())
        part = whole[1:3]
        print(part.shape, part.strides, part.contiguous, part.tolist())
        # And over bytes, where the view cannot be written through.
        frozen = memoryview(b"xy")
        print(frozen.readonly, frozen.shape, frozen.strides, frozen.contiguous)
        print(whole.readonly, whole.itemsize, whole.format)
    """,
    "a_static_is_reachable_from_a_value_too": """
        def show(label, f):
            try:
                print(label, f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        # AN IMPLICIT STATICMETHOD IS ONE METHOD, and the receiver decides which body
        # and is otherwise ignored: `bytes.fromhex(s)` and `b"".fromhex(s)` are the
        # same call. Every one of these was reachable only as a written call on the
        # type's own name, so a program holding the VALUE found nothing.
        show("str maketrans", lambda: "".maketrans("a", "b"))
        show("str maketrans deleting", lambda: "".maketrans("a", "b", "c"))
        show("str maketrans mapping", lambda: "".maketrans({"a": "b"}))
        show("bytes maketrans", lambda: type(b"".maketrans(b"a", b"b")).__name__)
        show("bytes fromhex", lambda: b"".fromhex("41 42"))
        show("bytearray fromhex", lambda: bytearray().fromhex("41 42"))
        show("float fromhex", lambda: (0.0).fromhex("0x1.8p+0"))
        show("dict fromkeys", lambda: {}.fromkeys([1, 2]))
        show("dict fromkeys with value", lambda: {}.fromkeys([1], 9))
        show("int from_bytes", lambda: (0).from_bytes(b"\\x01\\x00"))
        show("int from_bytes little", lambda: (0).from_bytes(b"\\x01\\x00", "little"))
        # 3.14's `from_number` converts a NUMBER and nothing else, which is the whole
        # reason it exists beside the constructor: `float("5")` reads text and this
        # refuses it.
        show("float from_number", lambda: (0.0).from_number(5))
        show("float from_number float", lambda: (0.0).from_number(1.5))
        show("float from_number str", lambda: (0.0).from_number("5"))
        show("float from_number complex", lambda: (0.0).from_number(1j))
        show("complex from_number", lambda: (0j).from_number(5))
        show("complex from_number cx", lambda: (0j).from_number(2j))
        show("complex from_number str", lambda: (0j).from_number("5"))
        # The type-object form still answers the same way, which is what makes them
        # one implementation rather than two that can drift.
        show("on the type", lambda: (bytes.fromhex("41"), bytearray.fromhex("41"),
                                     float.fromhex("0x1.8p+0"),
                                     dict.fromkeys([1]), int.from_bytes(b"\\x02"),
                                     float.from_number(7), complex.from_number(7)))
        print([type(getattr(v, n)).__name__ for v, n in
               (("", "maketrans"), (b"", "fromhex"), ({}, "fromkeys"),
                (0, "from_bytes"), (0.0, "from_number"), (0j, "from_number"))])
        # AND NO KIND CARRIES ONE IT DOES NOT HAVE.
        print([hasattr(v, n) for v, n in
               (("", "fromkeys"), ([], "fromkeys"), ("", "fromhex"),
                (1.5, "maketrans"), (0j, "from_bytes"), (5, "from_number"))])
    """,
    "a_local_wins_over_a_module_level_def": """
        def g():
            return "global g"
        def call(g):
            return g()
        print(call(lambda: "parameter"))
        # AN ASSIGNMENT IN THE BODY SHADOWS TOO, from the point it is a local -- which
        # in Python is the whole function, not the line after.
        def assigned():
            g = lambda: "assigned"
            return g()
        print(assigned())
        # A RECURSIVE MODULE-LEVEL `def` STILL REACHES ITSELF: its own name is a
        # global there, not a local, so nothing shadows it.
        def fact(n):
            return 1 if n < 2 else n * fact(n - 1)
        print(fact(5))
        # A NESTED `def` IS A LOCAL OF ITS ENCLOSING SCOPE and wins over a module one.
        def outer():
            def g():
                return "nested g"
            return g()
        print(outer())
        # AND A PLAIN CALL TO THE MODULE'S OWN `def` IS UNCHANGED.
        print(g())
        # A keyword argument through a shadowed parameter still lands.
        def kw(sorted):
            return sorted([3, 1, 2], reverse=True)
        print(kw(lambda xs, reverse=False: sorted(xs, reverse=reverse)))
        # A DEFAULT THAT NAMES THE GLOBAL is evaluated where the `def` runs.
        def defaulted(g=g):
            return g()
        print(defaulted())
    """,
    "a_docstring_survives_where_python_keeps_one": """
        def show(label, f):
            try:
                print(label, repr(f())[:48])
            except Exception as e:
                print(label, type(e).__name__ + ":", str(e)[:40])

        # PEP 257: A CLASS BODY OPENING WITH A STRING BINDS `__doc__`, and a class
        # without one binds None -- CPython puts the name in every class dict either
        # way. It was a bare expression here, so it ran and vanished, and `C.__doc__`
        # was an AttributeError about the attribute `help` is built on.
        class Base:
            "the base"
            def m(self):
                "a method"
        class Sub(Base):
            pass
        class Own:
            __doc__ = "assigned"
        class Late:
            "written"
            __doc__ = "then assigned"
        def fn():
            "a function"
        def plain():
            pass
        show("class", lambda: Base.__doc__)
        show("instance", lambda: Base().__doc__)
        show("method", lambda: Base.m.__doc__)
        show("bound method", lambda: Base().m.__doc__)
        show("no docstring", lambda: Sub.__doc__)
        show("in every class dict", lambda: "__doc__" in Base.__dict__)
        show("assigned instead", lambda: Own.__doc__)
        show("assigned after", lambda: Late.__doc__)
        show("function", lambda: fn.__doc__)
        show("no docstring function", lambda: plain.__doc__)
        # AND THE BUILTINS. `"".__doc__` IS `str.__doc__` -- the same text -- and it
        # was an AttributeError on every builtin value there is.
        show("a value has its type's", lambda: "".__doc__ == str.__doc__)
        show("reached through type()", lambda: b"".__doc__ == type(b"").__doc__)
        show("bool is not int's", lambda: True.__doc__ == int.__doc__)
        show("every kind answers", lambda: all(
            v.__doc__ == type(v).__doc__
            for v in ("", b"", bytearray(), [], (), {}, set(), frozenset(),
                      5, 1.5, range(3), 1j, True, None, memoryview(b"a"))))
        show("none of them is empty", lambda: min(
            len(type(v).__doc__) for v in ("", 5, None, memoryview(b"a"))) > 10)
        show("str says what it takes", lambda: "".__doc__.startswith("str(object="))
    """,
    "a_view_is_a_sequence_and_carries_one_s_methods": """
        def show(label, f):
            try:
                print(label, f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        # A VIEW IS A SEQUENCE, and it carries what a sequence carries. `hex`, `count`
        # and `index` read the bytes it shows, and the subscript dunders are the
        # methods behind the `m[i]` a program writes.
        view = memoryview(bytearray(b"abcd"))
        show("hex", lambda: view.hex())
        show("hex with a separator", lambda: view.hex(" "))
        show("count", lambda: view.count(98))
        show("count of nothing there", lambda: view.count(200))
        show("index", lambda: view.index(98))
        # THE NEEDLE IS COMPARED, NOT CHECKED: a view answers "not found" for one of
        # the wrong kind where the same needle handed to bytes is a TypeError.
        show("index of nothing there", lambda: view.index(200))
        show("count of the wrong kind", lambda: view.count("a"))
        show("index of the wrong kind", lambda: view.index("a"))
        show("iterating one", lambda: list(memoryview(b"ab")))
        def written(op):
            held = memoryview(bytearray(b"abcd"))
            if op == "set":
                held[0] = 65
            else:
                held.__setitem__(1, 66)
            return bytes(held)
        show("subscript assignment", lambda: written("set"))
        show("the method behind it", lambda: written("dunder"))
        # `__delitem__` EXISTS AND ALWAYS REFUSES, which is not the same claim as
        # having no such method: a window onto a buffer cannot shorten the buffer.
        def deleted():
            held = memoryview(bytearray(b"abcd"))
            del held[0]
            return bytes(held)
        show("deleting from one", deleted)
        show("the delete method", lambda: view.__delitem__(0))
        show("writing a read-only one", lambda: memoryview(b"ab").__setitem__(0, 65))
        show("parameterised by name", lambda: view.__class_getitem__(int))
        # AND AN INTEGER IS A LEGAL NEEDLE FOR THE BYTES SEARCHES, which their own
        # wording says and which they refused anyway. One byte, so anything outside
        # a byte's range is a ValueError rather than a wrong answer.
        show("bytes index of a byte", lambda: b"abcd".index(98))
        show("bytes count of a byte", lambda: b"abcd".count(98))
        show("bytearray index", lambda: bytearray(b"abcd").index(98))
        show("bytes find of a byte", lambda: b"abcd".find(98))
        show("out of a byte's range", lambda: b"abcd".index(300))
        show("below it", lambda: b"abcd".index(-1))
        show("still a substring search", lambda: b"abcd".index(b"bc"))
        print([type(getattr(view, n)).__name__ for n in
               ("hex", "count", "index", "__setitem__", "__delitem__")])
    """,
    "an_object_with_no_repr_of_its_own_says_what_it_is": """
        # `repr` OF A KIND WITH NO REPR OF ITS OWN read the value as a STR -- the cell
        # union's other half -- so `repr(map(len, xs))` walked a wild pointer and DIED
        # on both compiled runtimes while `repr(property(f))` answered the empty
        # string. The interpreter did not crash and leaked its own class names
        # instead: `<uasm.objects.host.Iterator object at 0x...>`.
        #
        # AN ADDRESS CANNOT MATCH CPYTHON, so what is checked here is the shape: the
        # kind CPython names and the punctuation around it.
        def show(label, f):
            try:
                print(label, "->", f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        def shape(text):
            # The repr with any hexadecimal address replaced, so two runs agree.
            out, i = "", 0
            while i < len(text):
                if text[i:i + 2] == "0x":
                    out += "0x_"
                    i += 2
                    while i < len(text) and text[i] in "0123456789abcdef":
                        i += 1
                    continue
                out += text[i]
                i += 1
            return out

        show("map", lambda: shape(repr(map(len, ["a"]))))
        show("filter", lambda: shape(repr(filter(None, "ab"))))
        show("zip", lambda: shape(repr(zip([1], [2]))))
        show("enumerate", lambda: shape(repr(enumerate([1]))))
        # A GENERATOR NAMES THE `def` IT CAME FROM, and a generator expression
        # the qualified name of the scope it was written in -- which for one
        # inside a lambda is `<lambda>.<locals>.<genexpr>`.
        show("generator", lambda: shape(repr(x for x in [1])))
        show("memoryview", lambda: shape(repr(memoryview(b"ab"))))
        show("property", lambda: shape(repr(property(len))))
        show("staticmethod", lambda: shape(repr(staticmethod(len))))
        show("classmethod", lambda: shape(repr(classmethod(len))))
        show("slice", lambda: repr(slice(1, 2)))
        show("range", lambda: repr(range(3)))
        show("dict keys", lambda: repr({}.keys()))

        # AND WHAT A CALLABLE IS CALLED. A builtin reached as a value has NO address
        # -- there is only ever the one -- and a bound method names what it is bound
        # to, with `built-in method` for the runtime's own.
        class C:
            def m(self):
                pass
        def g():
            pass
        show("builtin function", lambda: repr(len))
        show("builtin type", lambda: repr(int))
        show("user function", lambda: shape(repr(g)))
        show("user method", lambda: shape(repr(C().m)))
        show("builtin method", lambda: shape(repr("a".upper)))
        show("a view's method", lambda: shape(repr(memoryview(b"ab").tobytes)))
        show("printed, not repred", lambda: shape(str(len)))
    """,
    "a_method_reached_off_its_type_is_a_descriptor": """
        # REACHED OFF THE TYPE IS NOT THE SAME THING AS REACHED OFF A VALUE, and
        # CPython has four names for the two: `list.append` is a
        # `method_descriptor` and `list.__len__` a `wrapper_descriptor`, while
        # `[].append` is a `builtin_function_or_method` and `[].__len__` a
        # `method-wrapper`. Both unbound forms answered `function` or
        # `builtin_function_or_method` here, printed as one, and qualified their
        # name with nothing.
        #
        # TWO ROUTES REACH THE SAME PLACE: `str.upper` is a thunk the frontend
        # synthesises and `list.append` a native the type's prototype hands
        # over. Both are marked where they are built, and the qualname carries
        # which type -- see `apy_func_descr`.
        def show(label, f):
            try:
                print(label, "->", f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        def shape(text):
            out, i = "", 0
            while i < len(text):
                if text[i:i + 2] == "0x":
                    out += "0x_"
                    i += 2
                    while i < len(text) and text[i] in "0123456789abcdef":
                        i += 1
                    continue
                out += text[i]
                i += 1
            return out

        show("a native", lambda: type(list.append).__name__)
        show("its repr", lambda: repr(list.append))
        show("its qualname", lambda: list.append.__qualname__)
        show("its name", lambda: list.append.__name__)
        show("its objclass", lambda: list.append.__objclass__)
        show("it has no self", lambda: list.append.__self__)
        show("a thunk", lambda: type(str.upper).__name__)
        show("its repr", lambda: repr(str.upper))
        show("its qualname", lambda: str.upper.__qualname__)
        show("its name", lambda: str.upper.__name__)
        show("a dict method", lambda: type(dict.get).__name__)
        show("its repr", lambda: repr(dict.get))
        show("an int method", lambda: type(int.to_bytes).__name__)
        show("its qualname", lambda: int.to_bytes.__qualname__)
        show("a bytes method", lambda: repr(bytes.hex))
        show("a set method", lambda: repr(set.add))
        # A SLOT IS A `wrapper_descriptor` AND PRINTS AS ONE. Which of the two
        # it is, is whether the type WRITES the method out or fills a slot with
        # it: `list.__getitem__` is written out and `tuple.__getitem__` slotted,
        # and nothing in either signature says so.
        show("a slot", lambda: type(list.__len__).__name__)
        show("its repr", lambda: repr(list.__len__))
        show("its qualname", lambda: list.__len__.__qualname__)
        show("a written dunder", lambda: type(list.__getitem__).__name__)
        show("a slotted one", lambda: type(tuple.__getitem__).__name__)
        # A BOUND ONE KEEPS ITS OWN NAMES, and is qualified by its receiver's
        # kind -- the same text the descriptor answers, because binding does not
        # change where a method was defined.
        show("bound", lambda: type([].append).__name__)
        show("its repr", lambda: shape(repr([].append)))
        show("its qualname", lambda: [].append.__qualname__)
        show("its name", lambda: [].append.__name__)
        show("its self", lambda: [1].append.__self__)
        show("a bound slot", lambda: type([].__len__).__name__)
        show("its repr", lambda: shape(repr([].__len__)))
        # A USER METHOD IS NOT A DESCRIPTOR. `C.m` is an ordinary function and
        # `C().m` a `method`, which is a type of its own.
        class C:
            def m(self):
                pass
        show("a user method", lambda: type(C.m).__name__)
        show("its repr", lambda: shape(repr(C.m)))
        show("its qualname", lambda: C.m.__qualname__)
        show("bound", lambda: type(C().m).__name__)
        show("its repr", lambda: shape(repr(C().m)))
        show("a plain def", lambda: type(show).__name__)
        show("its repr", lambda: shape(repr(show)))
        show("a builtin", lambda: type(len).__name__)
        show("its repr", lambda: repr(len))
        show("a type", lambda: type(int).__name__)
        # AND THE REFUSAL NAMES THE TYPE rather than describing it.
        show("missing on a type", lambda: list.nope)
        show("missing on a value", lambda: (5).nope)
        show("missing on a class", lambda: C.nope)
        show("missing on an instance", lambda: C().nope)
        # CALLING ONE STILL WORKS, which is what a descriptor is for.
        def pushed():
            xs = [1]
            list.append(xs, 2)
            return xs
        show("called", lambda: str.upper("ab"))
        show("its effect", pushed)
    """,
    "an_iterator_is_named_after_what_it_walks": """
        # EVERY PLAIN CURSOR WAS AN `iterator`, where CPython names it after its
        # source: `list_iterator`, `dict_valueiterator`, `str_ascii_iterator`.
        # The name is what `type(it).__name__` answers and what the repr prints.
        #
        # `reversed(xs)` WAS A LIST, which answered three questions wrongly --
        # `type()` said `list`, `isinstance(..., list)` was True, and it could be
        # walked twice and indexed -- and copied the whole sequence before the
        # first element was wanted. It is a cursor counting down now.
        def show(label, f):
            try:
                print(label, "->", f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        def shape(text):
            # The repr with any hexadecimal address replaced, so two runs agree.
            out, i = "", 0
            while i < len(text):
                if text[i:i + 2] == "0x":
                    out += "0x_"
                    i += 2
                    while i < len(text) and text[i] in "0123456789abcdef":
                        i += 1
                    continue
                out += text[i]
                i += 1
            return out

        d = {1: 2}
        show("list", lambda: type(iter([1])).__name__)
        show("its repr", lambda: shape(repr(iter([1]))))
        show("tuple", lambda: type(iter((1,))).__name__)
        show("str", lambda: type(iter("a")).__name__)
        # A STRING WIDER THAN ASCII IS A DIFFERENT ITERATOR in CPython, which
        # records the width on the string; a str is UTF-8 bytes in the compiled
        # runtimes, so the same question there is whether any byte is above 0x7f.
        show("wide str", lambda: type(iter(chr(233))).__name__)
        show("bytes", lambda: type(iter(b"a")).__name__)
        show("bytearray", lambda: type(iter(bytearray(b"a"))).__name__)
        show("dict", lambda: type(iter(d)).__name__)
        show("keys", lambda: type(iter(d.keys())).__name__)
        show("values", lambda: type(iter(d.values())).__name__)
        show("items", lambda: type(iter(d.items())).__name__)
        show("set", lambda: type(iter({1})).__name__)
        show("frozenset", lambda: type(iter(frozenset({1}))).__name__)
        show("range", lambda: type(iter(range(3))).__name__)
        show("a callable and a sentinel", lambda: type(iter(int, 0)).__name__)
        show("map", lambda: type(map(len, [])).__name__)
        show("filter", lambda: type(filter(None, "")).__name__)

        show("reversed list", lambda: type(reversed([1])).__name__)
        show("its repr", lambda: shape(repr(reversed([1]))))
        show("reversed elements", lambda: list(reversed([1, 2, 3])))
        show("reversed tuple", lambda: type(reversed((1,))).__name__)
        show("reversed str", lambda: type(reversed("ab")).__name__)
        show("its elements", lambda: list(reversed("ab")))
        show("reversed range", lambda: type(reversed(range(3))).__name__)
        show("its elements", lambda: list(reversed(range(3))))
        show("reversed dict", lambda: type(reversed(d)).__name__)
        show("reversed keys", lambda: type(reversed(d.keys())).__name__)
        show("reversed values", lambda: type(reversed(d.values())).__name__)
        show("reversed items", lambda: type(reversed(d.items())).__name__)
        show("its elements", lambda: list(reversed({1: 2, 3: 4}.items())))
        # NOT EVERYTHING ITERABLE IS REVERSIBLE, and CPython refuses these four
        # by name: reversing needs a length and indexing, or a `__reversed__`.
        show("a set", lambda: reversed({1}))
        show("a frozenset", lambda: reversed(frozenset({1})))
        show("a cursor", lambda: reversed(iter([1])))
        show("a generator", lambda: reversed(x for x in [1]))
        show("a map", lambda: reversed(map(len, [])))
        show("an int", lambda: reversed(5))
        # A CLASS SAYS WHAT ITS REVERSE IS, and the hook wins over the index
        # walk -- a class may define both and they need not agree.
        class Own:
            def __reversed__(self):
                return iter("own")
        show("its own", lambda: "".join(reversed(Own())))
        class Indexed:
            def __len__(self):
                return 2
            def __getitem__(self, i):
                return "ab"[i]
        show("through the index walk", lambda: list(reversed(Indexed())))
    """,
    "an_unpack_reads_whatever_it_was_handed": """
        # THE UNPACK READS BY INDEX -- `a, *b, c = xs` takes `c` from `xs[-1]`
        # without ever computing a length -- and four ordinary sources have no
        # indices to read, each failing in its own way while `for` over the same
        # value worked:
        #
        #   a dict         subscripts by KEY, so `a, b = {1: 0, 2: 0}` asked for
        #                  key 0 and raised KeyError
        #   a set          cannot be subscripted at all
        #   a cursor       holds a position instead of indices
        #   an instance    with `__len__` and `__getitem__` was subscripted, and
        #                  the `*rest` branch handed its `__getitem__` a SLICE
        #
        # So the check that establishes the length ANSWERS WHAT TO INDEX, and
        # anything not read by position is copied into a list there.
        def show(label, f):
            try:
                print(label, "->", f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        def two(src):
            a, b = src
            return [a, b]

        def star(src):
            a, *r = src
            return [a, r]

        def mid(src):
            a, *r, z = src
            return [a, r, z]

        class Seq:
            # `__len__` AND `__getitem__` -- the older protocol, and the one
            # whose `__getitem__` must never be handed a slice.
            def __init__(self, n):
                self.n = n
            def __len__(self):
                return self.n
            def __getitem__(self, i):
                if not isinstance(i, int):
                    raise TypeError("no slices here")
                if i >= self.n:
                    raise IndexError(i)
                return i + 1

        class Walks:
            def __iter__(self):
                return iter([4, 5, 6])

        show("a list", lambda: two([1, 2]))
        show("a tuple", lambda: two((1, 2)))
        show("a str", lambda: two("ab"))
        show("bytes", lambda: two(b"ab"))
        show("a bytearray", lambda: two(bytearray(b"ab")))
        show("a range", lambda: two(range(2)))
        show("a memoryview", lambda: two(memoryview(b"ab")))
        show("a dict", lambda: two({"x": 1, "y": 2}))
        show("a set", lambda: sorted(two({1, 2})))
        show("a frozenset", lambda: sorted(two(frozenset({1, 2}))))
        show("keys", lambda: two({1: 0, 2: 0}.keys()))
        show("values", lambda: two({1: 7, 2: 8}.values()))
        show("items", lambda: two({1: 7, 2: 8}.items()))
        show("a cursor", lambda: two(iter([1, 2])))
        show("a map", lambda: two(map(int, "12")))
        show("a reversed", lambda: two(reversed([1, 2])))
        show("a generator", lambda: two(x for x in [1, 2]))
        show("a sequence class", lambda: two(Seq(2)))
        show("a class with iter", lambda: two(Walks()))
        show("its star", lambda: star(Walks()))
        show("its middle", lambda: mid(Walks()))
        show("a sequence class star", lambda: star(Seq(3)))
        show("a sequence class middle", lambda: mid(Seq(3)))
        show("a cursor middle", lambda: mid(iter([1, 2, 3, 4])))
        show("a str middle", lambda: mid("abcd"))
        # THE MESSAGES. Whether the surplus is COUNTED is decided by the source:
        # an exact list, tuple or dict knows its own length, and everything else
        # is unpacked through the iterator protocol, which can only say that
        # there was one element too many. The shortfall is always counted.
        show("a long list", lambda: two([1, 2, 3]))
        show("a long tuple", lambda: two((1, 2, 3)))
        show("a long dict", lambda: two({1: 0, 2: 0, 3: 0}))
        show("a long str", lambda: two("abc"))
        show("a long range", lambda: two(range(3)))
        show("a long cursor", lambda: two(iter([1, 2, 3])))
        show("a long generator", lambda: two(x for x in [1, 2, 3]))
        show("a long class", lambda: two(Seq(3)))
        show("a short list", lambda: two([1]))
        show("a short cursor", lambda: two(iter([1])))
        show("a short star", lambda: mid([1]))
        show("not iterable", lambda: two(5))
        show("None", lambda: two(None))
        show("a float", lambda: two(1.5))
    """,
    "a_consumer_walks_a_cursor_by_stepping_it": """
        # FOUR CONSUMERS COULD NOT WALK A CURSOR, each for the same reason: the
        # walk is by INDEX here and `apy_iterable` hands a cursor straight back
        # rather than draining it, so a `map` reached a loop that could only
        # read a container. Every one of them reported a thing that is plainly
        # iterable as not iterable:
        #
        #   `x in map(f, xs)`     every path        argument of type 'map' ...
        #   `set(map(f, xs))`     the interpreter   'Iterator' object is not ...
        #   `f(*map(f, xs))`      both compiled     'map' object is not ...
        #   `"".join(map(...))`   both compiled     can only join an iterable
        #
        # STEPPING, NOT DRAINING, is what makes `x in it` consume only as far as
        # the match and leave the rest -- which is the same rule `x in gen`
        # follows, and the reason a cursor is walked once.
        def show(label, f):
            try:
                print(label, "->", f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        def three(a, b, c):
            return [a, b, c]

        def _ext(into, src):
            into.extend(src)
            return into

        class Bare:
            pass

        class Seq:
            # `__len__` AND `__getitem__`: the older protocol, whose walk IS
            # the index walk -- and whose `__getitem__` must never be handed
            # a slice.
            def __len__(self):
                return 3
            def __getitem__(self, i):
                if not isinstance(i, int):
                    raise TypeError("no slices here")
                if i >= 3:
                    raise IndexError(i)
                return i + 1

        class Only:
            # `__getitem__` ALONE, walked until it reports IndexError.
            def __getitem__(self, i):
                if i >= 2:
                    raise IndexError(i)
                return i * 10

        show("in a map", lambda: 1 in map(int, "12"))
        show("in a filter", lambda: 2 in filter(None, [1, 2]))
        show("in a plain cursor", lambda: 1 in iter([1, 2]))
        show("in a zip", lambda: (1, 2) in zip([1], [2]))
        show("in an enumerate", lambda: (0, 1) in enumerate([1]))
        show("in a reversed", lambda: 1 in reversed([1, 2]))
        show("absent", lambda: 9 in map(int, "12"))
        # CONSUMED ONLY AS FAR AS THE MATCH, and what is left is what a later
        # walk sees -- the whole difference between stepping and draining.
        def partly():
            it = iter([1, 2, 3])
            return [1 in it, list(it)]
        show("consumed to the match", partly)
        show("set", lambda: sorted(set(map(int, "121"))))
        show("frozenset", lambda: sorted(frozenset(iter([2, 1, 2]))))
        show("splat into a call", lambda: three(*map(int, "123")))
        show("splat a plain cursor", lambda: three(*iter([1, 2, 3])))
        show("splat a reversed", lambda: three(*reversed([1, 2, 3])))
        show("a list display", lambda: [*map(int, "12")])
        show("a tuple display", lambda: (*iter([1, 2]),))
        show("a set display", lambda: sorted({*map(int, "12")}))
        show("extend", lambda: [0] + list(map(int, "12")))
        show("join", lambda: "".join(map(str, [1, 2])))
        show("join a reversed", lambda: "".join(reversed("abc")))
        show("join a plain cursor", lambda: "-".join(iter(["a", "b"])))
        # THE REFUSALS STAY REFUSALS. `join` words its own, and the three
        # iterables that yield ints get CPython's message about the element.
        show("join an int", lambda: "".join(5))
        show("join a range", lambda: "".join(range(3)))
        show("join bytes", lambda: "".join(b"ab"))
        show("in an int", lambda: 1 in 5)
        show("in a bare class", lambda: 1 in Bare())
        # A MEMORYVIEW IS A SEQUENCE OF INTS, and it fell past every arm to
        # the refusal at the end.
        show("in a memoryview", lambda: 97 in memoryview(b"ab"))
        show("absent from one", lambda: 1 in memoryview(b"ab"))
        show("in a class", lambda: 2 in Seq())
        show("in only getitem", lambda: 10 in Only())
        # AND THE OLDER PROTOCOL, which the same walks could not read: a class
        # with `__len__` and `__getitem__`, or with `__getitem__` alone.
        show("a class display", lambda: [*Seq()])
        show("into a call", lambda: three(*Seq()))
        show("its extend", lambda: _ext([0], Seq()))
        show("only getitem", lambda: [*Only()])
        show("a memoryview display", lambda: [*memoryview(b"ab")])
        show("a memoryview extend", lambda: _ext([0], memoryview(b"ab")))
        # A BYTEARRAY EXTENDS FROM ANY ITERABLE OF INTEGERS, and the two
        # refusals are different questions: a non-number is a TypeError and a
        # number outside a byte a ValueError.
        show("a bytearray from a map",
             lambda: _ext(bytearray(), map(int, "12")))
        show("a bytearray from a class", lambda: _ext(bytearray(), Seq()))
        show("a bytearray from a range", lambda: _ext(bytearray(), range(3)))
        show("a bytearray from a bool", lambda: _ext(bytearray(), [True]))
        show("a bytearray from a str", lambda: _ext(bytearray(), ["a"]))
        show("a bytearray out of range", lambda: _ext(bytearray(), [256]))
    """,
    "an_eager_consumer_walks_a_class_that_says_how_to_iterate": """
        # THIRTEEN CONSUMERS COULD NOT WALK A USER ITERATOR, all for one
        # reason: each asked `apy_raw_len` for a bound straight out instead of
        # funnelling through `apy_iterable` first. `apy_raw_len` knows the
        # OLDER protocol -- `__len__` plus `__getitem__` -- and nothing else,
        # so a class whose `__iter__` answers an object with `__next__` was
        # reported as not iterable by `sorted`, `sum`, `min`, `max`, `set`,
        # `frozenset`, `bytes`, `dict.fromkeys` and `s.union`, while a `for`
        # loop over the very same object walked it. `apy_iterable` had drained
        # one all along; the consumers just never asked it.
        #
        # `join` IS THE SAME GAP BY ANOTHER ROAD. It funnelled, then refused
        # every instance the funnel handed back -- which is exactly the class
        # with `__len__` and `__getitem__`, whose walk IS the index walk.
        #
        # AND A CLASS OBJECT IS WALKED BY ITS METACLASS, which the interpreter
        # funnel did not know: `for c in Color` worked and `sorted(Color)` did
        # not.
        def show(label, f):
            try:
                print(label, "->", f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        class Cursor:
            # A REAL ITERATOR: `__iter__` answers itself and `__next__` ends
            # on StopIteration.
            def __init__(self, n):
                self.i = 0
                self.n = n
            def __iter__(self):
                return self
            def __next__(self):
                if self.i >= self.n:
                    raise StopIteration
                self.i = self.i + 1
                return self.i * 10

        class Steps:
            # `__iter__` ANSWERS SOMETHING ELSE, which is the shape nothing
            # here could read: the object walked is not the object handed in.
            def __init__(self, n):
                self.n = n
            def __iter__(self):
                return Cursor(self.n)

        class Older:
            # `__len__` AND `__getitem__`, the protocol the index walk IS.
            def __len__(self):
                return 2
            def __getitem__(self, i):
                if i >= 2:
                    raise IndexError(i)
                return "ab"[i]

        class Bare:
            pass

        class Meta(type):
            def __iter__(cls):
                return iter(["x", "y"])

        class Walkable(metaclass=Meta):
            pass

        show("sorted", lambda: sorted(Steps(3)))
        show("sum", lambda: sum(Steps(3)))
        show("sum from a start", lambda: sum(Steps(3), 100))
        show("min", lambda: min(Steps(3)))
        show("max", lambda: max(Steps(3)))
        show("min by a key", lambda: min(Steps(3), key=lambda x: -x))
        show("max by a key", lambda: max(Steps(3), key=lambda x: -x))
        show("min of an empty one", lambda: min(Steps(0), default=-1))
        show("sorted by a key", lambda: sorted(Steps(3), key=lambda x: -x))
        show("set", lambda: sorted(set(Steps(3))))
        show("frozenset", lambda: sorted(frozenset(Steps(3))))
        show("bytes", lambda: bytes(Steps(3)))
        show("fromkeys", lambda: dict.fromkeys(Steps(2), 0))
        show("union", lambda: sorted({0}.union(Steps(2))))
        show("update", lambda: sorted({0} | set(Steps(2))))
        show("isdisjoint", lambda: {5}.isdisjoint(Steps(2)))
        # THE LAZY ROADS ALREADY WORKED, and still do -- they are what made
        # the eager ones look wrong rather than merely incomplete.
        show("a for loop", lambda: [x for x in Steps(3)])
        show("list", lambda: list(Steps(3)))
        show("tuple", lambda: tuple(Steps(3)))
        show("a display", lambda: [*Steps(3)])
        show("in", lambda: 20 in Steps(3))
        # `join` OVER THE OLDER PROTOCOL, which it refused by name.
        show("join a class", lambda: ",".join(Older()))
        show("join an iterator class", lambda: ",".join(Steps(0)))
        show("join a generator", lambda: ",".join(str(x) for x in range(3)))
        # `update` IS THE SAME GAP TWICE OVER -- once for the dict spelling
        # and once for the set one, which are ONE runtime function.
        def dupd(src):
            d = {0: 0}
            d.update(src)
            return sorted(d.items())

        def supd(src):
            s = {0}
            s.update(src)
            return sorted(s)

        class Pairs:
            def __init__(self):
                self.done = False
            def __iter__(self):
                return self
            def __next__(self):
                if self.done:
                    raise StopIteration
                self.done = True
                return (1, 2)

        class Sub(dict):
            pass

        show("dict update", lambda: dupd(Pairs()))
        show("set update", lambda: supd(Steps(2)))
        # AND A dict SUBCLASS UPDATES FROM ITS MAPPING, not from its keys:
        # iterating a dict yields keys, so the pair walk read one key per
        # element and reported a sequence element of length 1.
        show("update from a subclass", lambda: dupd(Sub({5: 6})))
        show("dict from a subclass", lambda: sorted(dict(Sub({5: 6})).items()))
        # TWO DIFFERENT ERRORS AND THEY STAY DIFFERENT: an element that is
        # not a sequence at all is a TypeError naming nothing, one of the
        # wrong length a ValueError naming its position.
        show("update a non-pair", lambda: dupd([5]))
        show("update a long pair", lambda: dupd([(1, 2, 3)]))
        show("update an int", lambda: dupd(5))
        # A CLASS OBJECT, walked by whatever its metaclass says.
        show("a metaclass walk", lambda: sorted(Walkable))
        show("join a class object", lambda: ",".join(Walkable))
        show("a metaclass set", lambda: sorted(set(Walkable)))
        # AND THE REFUSALS STAY REFUSALS, each in its own words: `join` names
        # no kind and everything else does.
        show("sorted an int", lambda: sorted(5))
        show("sorted a bare class", lambda: sorted(Bare()))
        show("sum a bare class", lambda: sum(Bare()))
        show("set a bare class", lambda: set(Bare()))
        show("union a bare class", lambda: {0}.union(Bare()))
        show("join a bare class", lambda: ",".join(Bare()))
        show("join an int", lambda: ",".join(5))
        show("join a class object", lambda: ",".join(Bare))
        show("sorted a class object", lambda: sorted(Bare))
    """,
    "an_ordering_names_the_pair_it_actually_stopped_on": """
        # `(1,) < ("a",)` SAID `'tuple' and 'tuple'`, about a comparison
        # tuples support perfectly well. CPython compares two sequences
        # lexicographically and names the first pair of ELEMENTS it could not
        # order -- int and str -- at whatever depth the walk reached. The
        # ordering answers a number and has nowhere to carry a witness out,
        # so the pair is found by walking again on the failure path.
        #
        # AND `sorted` WORDED IT AS AN OPERATOR. `sorted([1, "a"])` reported
        # `unsupported operand type(s) for <`, which is what `+` says about a
        # pair it cannot add; a comparison failing is worded differently, and
        # the operator itself already said so. `max` names `>` where `min`
        # and `sorted` name `<`, so which one it was travels with the
        # refusal.
        #
        # A CLASS EXTENDING A BUILTIN ORDERS AS THE BUILTIN, which equality
        # has always done and the ordering did not: `sorted` over two
        # namedtuples reported that two of them could not be compared, while
        # `==` between the same two answered.
        from collections import namedtuple

        P = namedtuple("P", "a b")

        class Sub(tuple):
            pass

        class Own(tuple):
            def __lt__(self, other):
                return self[0] > other[0]

        class Bare:
            pass

        class Num:
            def __init__(self, v):
                self.v = v
            def __lt__(self, other):
                return self.v < other.v

        def show(label, f):
            try:
                print(label, "->", f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        def srt(xs):
            out = list(xs)
            out.sort()
            return out

        show("two tuples", lambda: (1,) < ("a",))
        show("two lists", lambda: [1] < ["a"])
        show("past an equal head", lambda: (0, 1) < (0, "a"))
        show("nested", lambda: ((1,),) < (("a",),))
        show("and the answers", lambda: [(1, 2) < (1, 3), [1] < [1, 2]])
        show("sorted", lambda: sorted([1, "a"]))
        show("sorted of tuples", lambda: sorted([(0, 0), ("a", "b")]))
        show("sorted reversed", lambda: sorted([1, "a"], reverse=True))
        show("sorted by a key", lambda: sorted([1, "a"], key=lambda x: x))
        show("list.sort", lambda: srt([1, "a"]))
        show("min", lambda: min([1, "a"]))
        show("max", lambda: max([1, "a"]))
        show("min of several", lambda: min(1, "a"))
        show("max of several", lambda: max((1,), ("a",)))
        show("min by a key", lambda: min([1, "a"], key=lambda x: x))
        show("max by a key", lambda: max([1, "a"], key=lambda x: x))
        # A CLASS EXTENDING A BUILTIN, and one that writes its own order --
        # which must win over the builtin underneath it.
        show("a namedtuple", lambda: sorted([P(2, 1), P(1, 9)]))
        show("its min", lambda: min([P(2, 1), P(1, 9)]))
        show("its max", lambda: max([P(2, 1), P(1, 9)]))
        show("its sort", lambda: srt([P(2, 1), P(1, 9)]))
        show("a tuple subclass", lambda: Sub((1,)) < Sub((2,)))
        # AND THE MESSAGE STAYS THE ORIGINAL PAIR'S: unwrapping is how the
        # comparison is made, not what the program compared.
        show("against an int", lambda: sorted([P(1, 1), 5]))
        show("its operator", lambda: 5 < P(1, 1))
        show("sorted", lambda: [tuple(x) for x in sorted([Sub((2,)), Sub((1,))])])
        show("its own order wins", lambda: [tuple(x) for x in sorted([Own((1,)), Own((2,))])])
        show("and its operator", lambda: Own((1,)) < Own((2,)))
        # A CLASS WITH NO ORDER AT ALL names itself, not this file's class.
        show("two bare classes", lambda: sorted([Bare(), Bare()]))
        show("its operator", lambda: Bare() < Bare())
        show("its max", lambda: max([Bare(), Bare()]))
        show("against an int", lambda: sorted([Bare(), 5]))
        show("its max", lambda: max([Bare(), 5]))
        # AND A CLASS THAT DOES ORDER still orders.
        show("a user order", lambda: [x.v for x in sorted([Num(3), Num(1)])])
        show("its min", lambda: min([Num(3), Num(1)]).v)
        show("a nan", lambda: sorted([1.0, 0.5]))
        show("two sets", lambda: {1} < {1, 2})
    """,
    "an_unbound_builtin_method_is_called_with_its_receiver_first": """
        # `dict.get(d, 1)` RAISED `'type' object has no attribute 'get'`, for
        # a call `list.append([1], 2)` answered. `_dyn_method` picks its
        # symbol by NAME AND ARGUMENT COUNT, and the count it saw included the
        # receiver -- so the two-argument `get` row matched and
        # `apy_dict_get_or` was handed the `dict` type object. The five that
        # worked did so only because their method has no row at the shifted
        # count and fell through to `apy_getattr`, which finds the descriptor
        # on the type's prototype and applies it. Naming the same methods as
        # VALUES worked everywhere, which is what made the gap a lowering one.
        #
        # REWRITTEN TO THE BOUND SPELLING, so one implementation serves both
        # and the keyword folding, the name-collision test and the arity table
        # are the ones an ordinary call site uses.
        class Box:
            # A USER CLASS WITH THE SAME METHOD NAMES, which is what the
            # collision test is for: `Box().get(2)` must stay Box's.
            def __init__(self, v):
                self.v = v
            def get(self, k):
                return ("box", k)
            def upper(self):
                return "BOX"

        class MyList(list):
            pass

        def show(label, f):
            try:
                print(label, "->", f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        def srt(xs):
            out = list(xs)
            list.sort(out)
            return out

        def shadowed(dict):
            return dict.get(1)

        show("dict.get", lambda: dict.get({1: 2}, 1))
        show("int.to_bytes", lambda: int.to_bytes(258, 2, "little"))
        show("bytes.hex", lambda: bytes.hex(b"ab"))
        show("tuple.count", lambda: tuple.count((1, 1), 1))
        show("str.split", lambda: str.split("a b"))
        show("bytearray.hex", lambda: bytearray.hex(bytearray(b"ab")))
        show("complex.conjugate", lambda: complex.conjugate(2 + 3j))
        show("memoryview.tobytes", lambda: memoryview.tobytes(memoryview(b"ab")))
        show("float.is_integer", lambda: float.is_integer(2.0))
        show("bool.bit_length", lambda: bool.bit_length(True))
        show("set.union", lambda: srt(set.union({1}, {2})))
        show("frozenset.copy", lambda: srt(frozenset.copy(frozenset({2, 1}))))
        show("list.append", lambda: list.append([1], 2))
        show("str.upper", lambda: str.upper("ab"))
        show("list.__len__", lambda: list.__len__([1, 2]))
        show("dict.items", lambda: srt(dict.items({1: 2})))
        show("str.__contains__", lambda: str.__contains__("abc", "b"))
        # A KEYWORD TRAVELS WITH THE REST, folded into the slot it names --
        # the rewrite hands the ordinary call site its own arguments.
        show("a keyword", lambda: str.split("a b c", maxsplit=1))
        show("three of them", lambda: str.replace("aaa", "a", "b", 1))
        show("sort in place", lambda: srt([3, 1, 2]))
        # THE CONSTRUCTORS ON A TYPE are a different shape and keep theirs.
        show("fromkeys", lambda: srt(dict.fromkeys([2, 1], 0).items()))
        show("from_bytes", lambda: int.from_bytes(b"\x01\x02", "big"))
        show("fromhex", lambda: bytes.fromhex("6162"))
        # AND THE VALUE FORM, which always worked and still does.
        show("as a key", lambda: sorted(["B", "a"], key=str.lower))
        show("through map", lambda: list(map(str.upper, ["a", "b"])))
        # A SUBCLASS PASSES, because the check is `isinstance` and not a kind
        # comparison: a bool is an int and a bytearray is bytes-like.
        show("a list subclass", lambda: list.append(MyList([1]), 2))
        show("a bool as an int", lambda: int.bit_length(True))
        # THE REFUSALS ARE THE DESCRIPTOR'S OWN, which naming the method on
        # the receiver could not produce.
        show("no receiver", lambda: str.upper())
        show("the wrong kind", lambda: str.upper(5))
        show("another", lambda: dict.get([1], 1))
        # AND A NAME THE PROGRAM TOOK BACK is the program's.
        show("a user method", lambda: Box([1]).get(2))
        show("a parameter", lambda: shadowed({1: 5}))
    """,
    "a_builtin_a_bundled_module_provides_is_still_shadowable": """
        # `open`, `eval`, `exec` and `compile` ARE BUNDLED MODULES' FUNCTIONS
        # here: nothing imports a builtin, so the name APPEARING is the only
        # signal there is, and `_BUNDLED_BUILTINS` rewrites it to the spliced
        # definition. A program that binds the name itself means its own --
        # and one scope down, it did not get it.
        #
        # A MODULE-LEVEL BINDING WAS ALREADY SAFE, by a different road: it
        # stops the provider being spliced at all, so the members table is
        # empty and the rewrite cannot fire. A binding inside a function had
        # no such luck, and `open` is the name programs shadow most.
        #
        # THE BINDINGS THAT ARE PLAIN STRINGS were the last to be found:
        # `except E as open`, `import io as open` and `case _ as open` bind
        # the name as surely as an assignment, and none of them is a `Name`
        # node for a walk to see.
        def show(label, f):
            try:
                print(label, "->", f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        def a_parameter(open):
            return open("p")

        def an_assignment():
            open = lambda x: "assigned:" + x
            return open("a")

        def a_nested_def():
            def open(x):
                return "nested:" + x
            return open("n")

        def a_loop_target():
            seen = []
            for open in ["x", "y"]:
                seen.append(open)
            return seen

        def a_lambda_parameter():
            return (lambda open: open * 2)("z")

        def an_except_name():
            try:
                raise ValueError("caught")
            except ValueError as open:
                return str(open)

        def a_with_name():
            class Held:
                def __enter__(self):
                    return "held"
                def __exit__(self, *rest):
                    return False
            with Held() as open:
                return open

        def its_own_eval():
            def eval(text):
                return "mine:" + text
            return eval("1+1")

        def its_own_compile():
            compile = 5
            return compile + 1

        show("a parameter", lambda: a_parameter(lambda x: "param:" + x))
        show("an assignment", an_assignment)
        show("a nested def", a_nested_def)
        show("a loop target", a_loop_target)
        show("a lambda parameter", a_lambda_parameter)
        show("an except name", an_except_name)
        show("a with name", a_with_name)
        show("its own eval", its_own_eval)
        show("its own compile", its_own_compile)
        # AND THE BUILTIN IS STILL THE BUILTIN where nothing took the name --
        # the shadowing is per scope, not per module.
        show("still the builtin", lambda: open("no-such-file-here-at-all"))
    """,
    "a_generator_names_the_def_it_came_from": """
        # `repr(gen())` WAS `<generator object at 0x...>` -- CPython writes the
        # qualified name of the `def` between `object` and `at`, and a program
        # that logs a generator or reads a traceback sees the difference. Nor
        # was there a `__name__` or a `__qualname__` to read.
        #
        # THE NAME LIVES ON THE STEP FUNCTION, because the generator cell is a
        # frame and the frame's code is the step.
        def show(label, f):
            try:
                print(label, "->", f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        def shape(text):
            out, i = "", 0
            while i < len(text):
                if text[i:i + 2] == "0x":
                    out += "0x_"
                    i += 2
                    while i < len(text) and text[i] in "0123456789abcdef":
                        i += 1
                    continue
                out += text[i]
                i += 1
            return out

        def gen():
            yield 1

        class C:
            def m(self):
                yield 1

        def outer():
            def inner():
                yield 1
            return inner()

        show("a def", lambda: shape(repr(gen())))
        show("its name", lambda: gen().__name__)
        show("its qualname", lambda: gen().__qualname__)
        show("a method", lambda: shape(repr(C().m())))
        show("its qualname", lambda: C().m().__qualname__)
        show("its name", lambda: C().m().__name__)
        show("a nested def", lambda: outer().__qualname__)
        # A GENERATOR EXPRESSION IS `<genexpr>`, qualified by the scope it was
        # written in -- and at module level by nothing at all, because CPython's
        # `__qualname__` never names the module.
        show("an expression", lambda: shape(repr(x for x in [1])))
        show("its name", lambda: (x for x in [1]).__name__)
        show("its qualname", lambda: (x for x in [1]).__qualname__)
        show("in a def", lambda: outer.__qualname__)
        def holds():
            return (y for y in [1])
        show("one inside a def", lambda: holds().__qualname__)
        show("a lambda's", lambda: (lambda: 0).__qualname__)
        show("one inside a lambda",
             lambda: (lambda: (z for z in [1]))().__qualname__)
        # A COROUTINE AND AN ASYNC GENERATOR take the same shape under their own
        # kind names.
        async def waits():
            return 1
        made = waits()
        show("a coroutine", lambda: shape(repr(made)))
        made.close()
    """,
    "a_class_and_its_instances_are_qualified_by_their_module": """
        # `__module__` WAS NOT RECORDED AT ALL, so every class printed as
        # `<class 'C'>` where CPython says `<class '__main__.C'>` and every instance
        # as `<C object at 0x...>`. A class WRITTEN OUT gets it from its body; one
        # built by `type(name, bases, ns)` gets it from the constructor, as CPython's
        # own `type_new` supplies it -- and `enum`, `dataclasses` and `namedtuple`
        # all build classes that way.
        #
        # `builtins` IS THE ONE CPYTHON LEAVES OUT of a repr: `int` prints as
        # `<class 'int'>` and never `<class 'builtins.int'>`.
        def show(label, f):
            try:
                print(label, "->", f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        def shape(text):
            # The repr with any hexadecimal address replaced, so two runs agree.
            out, i = "", 0
            while i < len(text):
                if text[i:i + 2] == "0x":
                    out += "0x_"
                    i += 2
                    while i < len(text) and text[i] in "0123456789abcdef":
                        i += 1
                    continue
                out += text[i]
                i += 1
            return out

        class C:
            pass

        class D(C):
            pass

        show("written", lambda: C.__module__)
        show("in the dict", lambda: "__module__" in C.__dict__)
        show("inherited", lambda: D.__module__)
        show("class repr", lambda: repr(C))
        show("instance repr", lambda: shape(repr(C())))
        show("builtin type", lambda: repr(int))
        show("builtin type module", lambda: int.__module__)
        made = type("Made", (), {})
        show("by the constructor", lambda: made.__module__)
        show("its repr", lambda: repr(made))
        told = type("Told", (), {"__module__": "elsewhere"})
        show("told which", lambda: told.__module__)
        show("its repr", lambda: repr(told))
        # A SPLICED CLASS IS NOT `__main__`. The module a bundled definition was
        # written in is the only thing its mangled name still says, and this is the
        # half that reads it back out.
        import fractions
        show("bundled", lambda: fractions.Fraction.__module__)
        show("bundled repr", lambda: repr(fractions.Fraction))
    """,
    "a_callable_says_which_module_it_was_written_in": """
        # `f.__module__` WAS AN AttributeError on every path, for every callable.
        # CPython answers five different ways and the receiver decides which:
        #
        #   `len`                     `builtins`      a builtin reached as a value
        #   `int`                     `builtins`      a builtin TYPE as a value
        #   a program's own `def`     `__main__`      including a lambda
        #   a spliced `def`           `fractions`     where it was written
        #   `[].append`               None            a BOUND builtin method
        #
        # `__main__` IS A DEFAULT AND NOT A RECORD: every `def` a program writes is
        # in one module and there is only one it can be in, so nothing is emitted to
        # say so and only a spliced `def` is told.
        def show(label, f):
            try:
                print(label, "->", f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        def g():
            pass

        class C:
            def m(self):
                pass

        show("builtin", lambda: len.__module__)
        show("builtin printer", lambda: print.__module__)
        show("builtin type", lambda: int.__module__)
        show("own def", lambda: g.__module__)
        show("a lambda", lambda: (lambda: 0).__module__)
        show("a method", lambda: C.m.__module__)
        show("a bound method", lambda: C().m.__module__)
        show("a bound builtin", lambda: [].append.__module__)
        show("its receiver", lambda: [1].append.__self__)
        # THE MANGLED NAME IS AN IMPLEMENTATION DETAIL AND A PROGRAM CAN SEE IT: a
        # method of a spliced class took its `__qualname__` from the class's key, so
        # it read `_asmpy_bundled_9_fractions_Fraction.limit_denominator`.
        import fractions
        show("spliced module", lambda: fractions.Fraction.limit_denominator.__module__)
        show("spliced qualname",
             lambda: fractions.Fraction.limit_denominator.__qualname__)
        show("spliced name", lambda: fractions.Fraction.limit_denominator.__name__)
        show("its own name", lambda: g.__qualname__)
        show("a method's", lambda: C.m.__qualname__)
    """,
    "a_keyword_reaches_sort_and_update_by_every_route": """
        # `sort` AND `update` PLACE THEIR OWN KEYWORDS -- `key` and `reverse` travel as
        # VALUES so the key runs once per element, and `update`'s keywords ARE the value
        # -- and four routes got past that branch. Three of them DROPPED the keyword in
        # silence: the call answered as though it had never been written.
        def show(label, f):
            try:
                print(label, "->", repr(f()))
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        def mk():
            return ["bb", "a", "ccc"]

        def sorted_with(*a, **kw):
            held = mk()
            held.sort(*a, **kw)
            return held

        # A NAME `sort` DOES NOT TAKE, written out.
        show("unknown name", lambda: mk().sort(nope=1))
        show("a near miss", lambda: mk().sort(revers=1))
        show("a known name and an unknown one", lambda: mk().sort(reverse=True, nope=1))
        # A `**` MAPPING, whose contents are a run-time value.
        show("spread reverse", lambda: (lambda xs: (xs.sort(**{"reverse": True}), xs)[1])(mk()))
        show("spread key", lambda: (lambda xs: (xs.sort(**{"key": len}), xs)[1])(mk()))
        show("spread both",
             lambda: (lambda xs: (xs.sort(**{"key": len, "reverse": True}), xs)[1])(mk()))
        show("spread empty", lambda: (lambda xs: (xs.sort(**{}), xs)[1])(mk()))
        show("spread unknown", lambda: mk().sort(**{"nope": 1}))
        show("spread and written", lambda: sorted_with(**{"reverse": True}))
        # SOURCE ORDER DECIDES: a later key wins over one a `**` brought.
        show("written wins", lambda: (lambda xs: (xs.sort(**{"reverse": True}, reverse=False), xs)[1])(mk()))
        show("mapping wins", lambda: (lambda xs: (xs.sort(reverse=False, **{"reverse": True}), xs)[1])(mk()))
        # AND A POSITIONAL IS STILL REFUSED, which is what keyword-only means.
        show("positional", lambda: mk().sort(None))

        # `update`'s KEYWORDS ARE THE VALUE, so any name at all becomes a key.
        def merged(*a, **kw):
            held = {"x": 0}
            held.update(*a, **kw)
            return sorted(held.items())

        show("written name", lambda: merged(a=1))
        show("a mapping", lambda: merged({"b": 2}))
        show("both", lambda: merged({"b": 2}, c=3))
        show("spread", lambda: merged(**{"a": 1}))
        show("spread and written", lambda: merged(**{"a": 1}, b=2))
        show("spread empty", lambda: merged(**{}))
        show("by name", lambda: (lambda d: (getattr(d, "update")(a=1), sorted(d.items()))[1])({"x": 0}))
        show("by name with a mapping",
             lambda: (lambda d: (getattr(d, "update")({"b": 2}, c=3), sorted(d.items()))[1])({"x": 0}))
        # A SET'S `update` REALLY DOES TAKE NO KEYWORD, which is the line this is on
        # the other side of.
        show("a set by name", lambda: (lambda s: getattr(s, "update")(a=1))({1}))
        show("a set with others", lambda: sorted((lambda s: (s.update({2}, {3}), s)[1])({1})))

        # A NAME GIVEN BOTH BY A MAPPING AND WRITTEN OUT is a TypeError, whichever
        # order the two come in -- and a merge cannot see it, so the later key simply
        # won and `"a,b".split(**{"sep": ","}, sep=";")` answered `['a,b']` with
        # nothing to mark it. Every spread-folded method had it.
        show("split twice", lambda: "a,b".split(**{"sep": ","}, sep=";"))
        show("split the other way", lambda: "a,b".split(sep=";", **{"sep": ","}))
        show("replace twice", lambda: "aaa".replace("a", "b", **{"count": 1}, count=2))
        show("encode twice", lambda: "a".encode(**{"encoding": "utf-8"}, encoding="ascii"))
        show("two mappings", lambda: "a,b".split(**{"sep": ","}, **{"sep": ";"}))
        show("a mapping and a different name",
             lambda: "a,b,c".split(**{"sep": ","}, maxsplit=1))
        show("still one name", lambda: "a,b".split(**{"sep": ","}))
    """,
    "a_keyword_named_like_a_rest_parameter_is_collected": """
        def show(label, f):
            try:
                print(label, "->", repr(f()))
            except Exception as e:
                print(label, type(e).__name__ + ":", e)
        def only_rest(*a):
            return a
        def only_kw(**kw):
            return sorted(kw.items())
        def both(*a, **kw):
            return (a, sorted(kw.items()))
        def mixed(p, q=9, *a, r=8, **kw):
            return (p, q, a, r, sorted(kw.items()))
        show("kw named kw", lambda: only_kw(kw=1))
        show("both a", lambda: both(a=1))
        show("both kw", lambda: both(kw=1))
        show("mixed a", lambda: mixed(1, a=2))
        show("mixed kw", lambda: mixed(1, kw=2))
        show("mixed r", lambda: mixed(1, r=2))
        show("mixed q", lambda: mixed(1, q=2))
        show("mixed all", lambda: mixed(1, 2, 3, r=4, z=5))
        show("posonly", lambda: (lambda: None)())
        def po(x, /, y):
            return (x, y)
        show("positional only", lambda: po(x=1, y=2))
        show("dup", lambda: mixed(1, 2, q=3))
        # A `*rest` OR `**kw` PARAMETER CANNOT BE FILLED BY NAME: the name belongs to
        # the COLLECTION and not to a slot. The interpreter matched a keyword against
        # every declared parameter including those two, so `f(a=1)` on a `def f(*a,
        # **kw)` bound the tuple to 1 and then tried to walk it -- an interpreter-only
        # wrong answer where both compiled runtimes collected the name.
        show("a name the tuple has", lambda: both(a=1))
        show("a name the mapping has", lambda: both(kw=1))
        show("both at once", lambda: both(a=1, kw=2))
        show("through a spread", lambda: both(**{"a": 1, "kw": 2}))
        def carries(*a, **kw):
            held = {"x": 0}
            held.update(*a, **kw)
            return sorted(held.items())
        show("forwarded to update", lambda: carries(a=1))
        show("forwarded with a mapping", lambda: carries({"b": 2}, c=3))
    """,
    "an_argument_of_the_wrong_type_is_worded_by_cpython": """
        # A BUILTIN METHOD'S ARGUMENT-TYPE REFUSALS were worded differently from
        # CPython's in seven places, and three of them were not refusals at all: a None
        # encoding was read as "the default", a None delete set as "delete nothing", and
        # a view's bounded `index` said the view had no such method.
        def show(label, f):
            try:
                print(label, "->", repr(f()))
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        # A CODEC ARGUMENT IS NAMED, and None is not a default once it is WRITTEN: the
        # lowering pads a slot the call left out with the default's own text, so the
        # two can be told apart at all.
        show("encode", lambda: "a".encode())
        show("encode named", lambda: "a".encode("utf-8", "strict"))
        show("encode None", lambda: "a".encode(None))
        show("encode errors None", lambda: "a".encode("utf-8", None))
        show("encode int", lambda: "a".encode(5))
        show("encode by keyword", lambda: "a".encode(encoding=None))
        show("decode None", lambda: b"a".decode(None))
        show("bytearray decode None", lambda: bytearray(b"a").decode(None))
        show("a view has no decode", lambda: memoryview(b"a").decode())
        show("the constructor still defaults", lambda: bytes("a", "utf-8"))
        show("and so does str's", lambda: str(memoryview(b"a"), "utf-8"))

        # A BYTEARRAY NAMES WHAT IT COULD NOT WALK, and a str is walkable while its
        # elements are not integers -- which CPython says in different words again.
        show("extend None", lambda: bytearray(b"a").extend(None))
        show("extend int", lambda: bytearray(b"a").extend(5))
        show("extend str", lambda: bytearray(b"a").extend("ab"))
        show("extend bytes", lambda: (lambda t: (t.extend(b"bc"), bytes(t))[1])(bytearray(b"a")))
        show("extend ints", lambda: (lambda t: (t.extend([1, 2]), bytes(t))[1])(bytearray(b"a")))
        show("a list still says iterable", lambda: [].extend(None))

        # `format_map` SUBSCRIPTS RATHER THAN CHECKING, and only once a field asks it
        # for a key -- so a string with no field never touches the mapping at all.
        show("no field", lambda: "".format_map(None))
        show("no field but text", lambda: "a".format_map(None))
        show("a field", lambda: "{a}".format_map(None))
        show("a field and a list", lambda: "{a}".format_map([]))
        show("a field and a mapping", lambda: "{a}".format_map({"a": 1}))

        # A SEQUENCE'S BOUNDED `index` TAKES NO None, where a string's does -- and a
        # view is one of the sequences.
        show("view index", lambda: memoryview(b"abc").index(98))
        show("view index window", lambda: memoryview(b"abcb").index(98, 2, 4))
        show("view index None", lambda: memoryview(b"ab").index(97, None, None))
        show("view index missing", lambda: memoryview(b"ab").index(99))
        show("list index None", lambda: [1, 2].index(2, None, None))
        show("str index None", lambda: "ab".index("b", None, None))
        show("bytes index None", lambda: b"ab".index(b"b", None, None))

        # THE TWO PARTITION REFUSALS, which differ by RECEIVER: a bytes one quotes the
        # type and a str one does not.
        show("bytes partition", lambda: b"ab".partition(None))
        show("bytearray partition", lambda: bytearray(b"ab").partition(None))
        show("str partition", lambda: "ab".partition(None))
        show("bytes rpartition", lambda: b"ab".rpartition(None))

        # `to_bytes` NAMES ITS LENGTH BY THE VALUE and its byteorder by the parameter.
        show("to_bytes None", lambda: (5).to_bytes(None))
        show("to_bytes str", lambda: (5).to_bytes("x"))
        show("to_bytes order None", lambda: (5).to_bytes(2, None))
        show("to_bytes ok", lambda: (258).to_bytes(2, "little"))

        # A DELETE SET THAT WAS WRITTEN must be bytes-like; the one-argument form
        # passes an empty one rather than a None.
        show("translate table only", lambda: b"abc".translate(None))
        show("translate delete", lambda: b"abc".translate(None, b"b"))
        show("translate delete None", lambda: b"abc".translate(None, None))
        show("str translate None", lambda: "abc".translate(None))
    """,
    "a_sort_reached_by_name_still_takes_its_keywords": """
        # `list.sort`'s KEYWORDS WERE REFUSED when the method was reached by name.
        # `getattr(xs, "sort")(reverse=True)` was `list.sort() takes no keyword
        # arguments` for a call CPython sorts: the written form has a branch of its own
        # -- `key` and `reverse` travel as VALUES so the key runs once per element --
        # and the by-name spelling had no signature to match a name against.
        def show(label, f):
            try:
                print(label, "->", repr(f()))
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        def mk():
            return ["bb", "a", "ccc"]

        def sorted_by(**named):
            held = mk()
            getattr(held, "sort")(**named) if False else held.sort(**named)
            return held

        show("written plain", lambda: (lambda xs: (xs.sort(), xs)[1])(mk()))
        show("written reverse", lambda: (lambda xs: (xs.sort(reverse=True), xs)[1])(mk()))
        show("written key", lambda: (lambda xs: (xs.sort(key=len), xs)[1])(mk()))
        show("written both", lambda: (lambda xs: (xs.sort(key=len, reverse=True), xs)[1])(mk()))
        show("by name plain", lambda: getattr(mk(), "sort")())
        show("by name reverse", lambda: (lambda xs: (getattr(xs, "sort")(reverse=True), xs)[1])(mk()))
        show("by name key", lambda: (lambda xs: (getattr(xs, "sort")(key=len), xs)[1])(mk()))
        show("by name both",
             lambda: (lambda xs: (getattr(xs, "sort")(key=len, reverse=True), xs)[1])(mk()))
        show("by name unknown", lambda: getattr(mk(), "sort")(nope=1))
        # A POSITIONAL IS STILL REFUSED, by the arity gate rather than by a signature:
        # both parameters are keyword-only.
        show("written positional", lambda: mk().sort(None))
        show("by name positional", lambda: getattr(mk(), "sort")(None))
        # AND THE METHOD VALUE IS STILL A BUILTIN METHOD of the right owner.
        show("its type", lambda: type(getattr(mk(), "sort")).__name__)
        show("a tuple has none", lambda: ().sort())
    """,
    "a_wrong_argument_count_is_worded_by_its_receiver": """
        # A WRONG NUMBER OF ARGUMENTS to a builtin method was reported with ONE wording
        # where CPython has nine -- and sometimes not as a wrong count at all: a set's
        # `pop` said the set had no such method, and a dict's said `KeyError: None`.
        def show(label, f):
            try:
                print(label, "->", f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        # THE NINE WORDINGS, one example apiece. They disagree about every visible
        # thing: whether the type qualifies the name, whether the parentheses are
        # there, and whether the count is `exactly one` or `at most 3`.
        show("qualified exact", lambda: [].append())
        show("qualified none", lambda: (5).bit_count(1))
        show("at most", lambda: "a".encode(1, 2, 3))
        show("at most positional", lambda: (5).to_bytes(1, 2, 3))
        show("at least positional", lambda: "a".replace("a"))
        show("expected at least", lambda: "a".count())
        show("expected at most", lambda: "a".lstrip(1, 2))
        show("expected exactly", lambda: [].insert(1))
        show("no positional", lambda: [].sort(None))

        # THE SAME NAME WORDED BY THE RECEIVER. `list.count` takes exactly one and
        # `str.count` takes a window, so one refusal names the type and the other
        # does not -- a table keyed by name alone could not hold both.
        show("list count", lambda: [1].count(1, 2))
        show("str count", lambda: "a".count("a", 0, 1, 2))
        show("str count window", lambda: "abcabc".count("a", 1, 6))
        show("dict pop none", lambda: {}.pop())
        show("set pop one", lambda: set().pop(1))
        show("list pop two", lambda: [1].pop(0, 1))
        show("float hex one", lambda: (1.5).hex(1))

        # AND REACHED BY NAME, which is the same question asked the other way: the
        # method value used to declare the widest arity any kind has.
        show("by name append", lambda: getattr([], "append")())
        show("by name count", lambda: getattr("abcabc", "count")("a", 1, 6))
        show("by name pop", lambda: getattr({}, "pop")())
        show("by name hex", lambda: getattr((1.5), "hex")(1))

        # `bytes.hex` TAKES A GROUP SIZE, and which end it counts from is the SIGN.
        show("hex grouped", lambda: bytes(range(1, 8)).hex(":", 2))
        show("hex from the left", lambda: bytes(range(1, 8)).hex(":", -3))
        show("hex wider than the whole", lambda: bytes(range(1, 8)).hex(":", 8))
        show("hex bad separator", lambda: b"ab".hex(":::"))
        show("hex no separator", lambda: b"ab".hex())

        # THE SIX SET METHODS TAKE ANY NUMBER OF OTHERS, which no arity table can say.
        s = {1, 2, 3}
        show("union none", lambda: sorted(s.union()))
        show("union two", lambda: sorted(s.union({4}, {5})))
        show("intersection two", lambda: sorted(s.intersection({1, 2}, {2, 3})))
        show("difference two", lambda: sorted(s.difference({1}, {2})))
        show("union keeps the kind", lambda: type(frozenset({1}).union()).__name__)
        show("union takes an iterable", lambda: sorted(s.union([9])))
        show("union refuses keywords", lambda: s.union(x=1))
        def updated(name, *others):
            held = {1, 2, 3}
            getattr(held, name)(*others)
            return sorted(held)
        show("update two", lambda: updated("update", {4}, {5}))
        show("intersection_update two", lambda: updated("intersection_update", {1, 2}, {2, 3}))
        show("difference_update two", lambda: updated("difference_update", {1}, {2}))
    """,
    "dir_lists_every_name_getattr_answers": """
        # `dir(x)` OVER A BUILTIN ANSWERED AN EMPTY LIST, on every path: a builtin has no
        # class chain to walk here, so `dir(5)`, `dir("")` and `dir({})` were all `[]`
        # where CPython lists seventy to ninety names apiece.
        vals = ["", b"", bytearray(), [], (), {}, set(), frozenset(), 5, 1.5,
                range(3), 1j, True, None, memoryview(b"ab")]
        for v in vals:
            names = dir(v)
            print(type(v).__name__, len(names), names[:2], names[-2:])

        # A LIST THAT ADVERTISES A NAME `getattr` REFUSES would be worse than the empty
        # one it replaces, so the two are held against each other here.
        missing = []
        for v in vals:
            for n in dir(v):
                try:
                    getattr(v, n)
                except Exception as e:
                    missing.append((type(v).__name__, n, type(e).__name__))
        print("refused:", missing)
        # SORTED AND DEDUPLICATED, and `x.__dir__()` is the same question.
        print([dir(v) == sorted(set(v.__dir__())) for v in vals])
        # A BUILTIN TYPE IS THE SAME LIST AS A VALUE OF IT.
        print(dir(str) == dir(""), dir(int) == dir(5), dir(dict) == dir({}))
        # A USER CLASS STILL WALKS ITS OWN CHAIN, and `__dir__` still overrides it --
        # sorted but NOT deduplicated, which is what CPython does with what it returned.
        class Base:
            def inherited(self): pass
        class Sub(Base):
            def mine(self): pass
        print([n for n in dir(Sub()) if not n.startswith("_")])
        print([n for n in dir(Sub) if not n.startswith("_")])
        class Proxy:
            def __dir__(self): return ["b", "a", "a"]
        print(dir(Proxy()))
    """,
    # A CURSOR IS A KIND TOO, and `dir()` over one answered the empty list
    # that `dir(5)` used to -- for every walk there is: `iter([])`,
    # `reversed(xs)`, a dict's three, a set's, `map`, `filter`, `enumerate`,
    # `zip` and `iter(f, sentinel)`.
    #
    # THE NAME IS THE KEY and the names were already right: `type(iter([]))`
    # has said `list_iterator` here for a while. What was missing was a row
    # per name in the generated table, which is CPython's own `dir()` read at
    # generation time -- so this asks for the LENGTH of each list rather than
    # its contents, which would be ninety lines of dunder.
    #
    # AND THE SAME HONESTY RULE the builtin values are held to: a list that
    # advertises a name `getattr` then refuses is worse than the empty one it
    # replaces. Two names made that real work -- `__setstate__`, which is
    # where a walk IS and which only some cursors carry, and `enumerate`'s
    # `__class_getitem__`.
    #
    # A GENERATOR IS DELIBERATELY ABSENT from the table: seven of its
    # thirty-eight names are frame introspection this runtime does not have,
    # and listing them would be the lying list the rule exists to prevent.
    #
    # `memory_iterator` IS IN THE TABLE AND NOT IN THIS PROGRAM, because
    # `iter(memoryview(b"a"))` is refused here at all -- a divergence of its
    # own, and not this one.
    "dir_over_a_cursor_lists_what_cpython_lists": """
        def named():
            return [
                ("list_iterator", iter([1])),
                ("list_reverseiterator", reversed([1])),
                ("tuple_iterator", iter((1,))),
                ("reversed", reversed((1,))),
                ("str_ascii_iterator", iter("a")),
                ("bytes_iterator", iter(b"a")),
                ("bytearray_iterator", iter(bytearray(b"a"))),
                ("range_iterator", iter(range(1))),
                ("set_iterator", iter({1})),
                ("frozen_iterator", iter(frozenset({1}))),
                ("dict_keyiterator", iter({1: 2}.keys())),
                ("dict_valueiterator", iter({1: 2}.values())),
                ("dict_itemiterator", iter({1: 2}.items())),
                ("dict_reversekeyiterator", reversed({1: 2}.keys())),
                ("callable_iterator", iter(lambda: None, None)),
                ("enumerate", enumerate([1])),
                ("zip", zip([1])),
                ("map", map(str, [1])),
                ("filter", filter(None, [1])),
            ]
        for want, it in named():
            print(want, type(it).__name__, len(dir(it)))
        # THE LIST IS ONLY HONEST IF EVERY NAME ON IT ANSWERS.
        missing = []
        for want, it in named():
            for n in dir(it):
                try:
                    getattr(it, n)
                except Exception as e:
                    missing.append((want, n, type(e).__name__))
        print("refused:", missing)
        # `__doc__` IS None FOR MOST OF THEM and real text for four, and
        # None is an ANSWER rather than a refusal -- it is on the list.
        for want, it in named():
            d = it.__doc__
            print(want, "doc",
                  "None" if d is None else d.split(chr(10))[0][:40])
        # `it.__setstate__(i)` -- WHERE THE WALK IS, written. The clamping is
        # CPython's: a forward cursor takes 0..len and anything outside it,
        # above or below, is exhausted; a reversed one counts down from `i`.
        def at(make, n):
            it = make()
            it.__setstate__(n)
            return list(it)
        print("fwd:", [at(lambda: iter([1, 2, 3]), n)
                       for n in (0, 1, 3, 9, -1)])
        print("rev:", [at(lambda: reversed([1, 2, 3]), n)
                       for n in (0, 1, 2, -1, 9)])
        print("tuple:", at(lambda: iter((1, 2, 3)), 1))
        print("str:", at(lambda: iter("abc"), 1))
        print("bytes:", at(lambda: iter(b"abc"), 1))
        print("range:", at(lambda: iter(range(3)), 1))
        # `zip` AND `map` KEEP NO POSITION OF THEIR OWN.
        print("zip:", at(lambda: zip([1, 2]), 1))
        print("map:", at(lambda: map(str, [1, 2]), 1))
        try:
            iter([1]).__setstate__("x")
        except Exception as e:
            print("bad arg:", type(e).__name__, e)
        # AND `enumerate` IS THE ONE CURSOR WITH A `__class_getitem__`.
        e = enumerate([1])
        print("subscript:", e.__class_getitem__(int),
              type(e.__class_getitem__(int)).__name__)
    """,
    # A CURSOR'S TYPE OBJECT CARRIED NOTHING. `type(iter([]))` says
    # `list_iterator` and always has, and every method that name stands for
    # was out of reach: `type(it).__next__` was an AttributeError about the
    # one method an iterator is FOR, and `dir()` over it answered [].
    #
    # THE MACHINERY WAS ALREADY THERE, for the other shape of the same
    # question: `list.append` has no list to ask, so `apy_kind_prototype`
    # makes an empty one and `apy_no_attribute` asks THAT what the kind
    # carries. What was missing was a prototype per cursor -- which is a
    # cursor with NO SOURCE, because the mode and the name are all that is
    # read of one -- and the route from a TYPE CELL to it. A cell
    # `apy_type_for` minted is told from a class the program wrote by its
    # empty dict, no base and no metaclass.
    #
    # AND THE DESCRIPTOR IT HANDS BACK IS CPYTHON'S KIND. `__next__` fills a
    # slot and `__length_hint__` is written out, so one reprs as a `slot
    # wrapper` and the other as a `method` -- which the bit-keyed table
    # could not say, a cursor having no bit.
    "a_cursor_type_object_carries_its_kinds_methods": """
        it = iter([1, 2, 3])
        t = type(it)
        print("name:", t.__name__, "dir:", len(dir(t)), "doc:", t.__doc__)
        print("same list:", dir(t) == dir(it))
        print("next:", t.__next__(it), t.__next__(it))
        print("hint:", t.__length_hint__(iter([1, 2, 3])))
        print("reprs:", repr(t.__next__), "|", repr(t.__length_hint__))
        print("qualnames:", t.__next__.__qualname__,
              t.__length_hint__.__qualname__)
        try:
            t.nope
        except Exception as e:
            print("missing:", type(e).__name__, e)
        # EVERY CURSOR KIND, through the same route.
        for v in (reversed([1]), iter((1,)), iter("a"), iter(b"a"),
                  iter(range(1)), iter({1}), iter({1: 2}.keys()),
                  iter({1: 2}.values()), iter({1: 2}.items()),
                  iter(lambda: None, None), enumerate([1]), zip([1]),
                  map(str, [1]), filter(None, [1])):
            k = type(v)
            print(" ", k.__name__, len(dir(k)), dir(k) == dir(v),
                  hasattr(k, "__next__"))
        # A CLASS THE PROGRAM WROTE STILL WALKS ITS OWN CHAIN, which is what
        # the empty-dict test is there to protect.
        class Base:
            def inherited(self):
                pass
        class Sub(Base):
            def mine(self):
                pass
        print("user:", [n for n in dir(Sub) if not n.startswith("_")])
        print("user instance:", [n for n in dir(Sub())
                                 if not n.startswith("_")])
        # AND A BUILTIN TYPE REACHED AS A VALUE IS UNCHANGED.
        print("builtin:", repr(type([]).append), len(dir(type([]))))
        xs = [1]
        type([]).append(xs, 9)
        print("written on a type:", xs, type("").upper("ab"))
    """,
    # A MEMORYVIEW WAS ITERABLE EVERY WAY BUT `iter`. `list(mv)`, `[*mv]`,
    # `98 in mv`, `reversed(mv)` and a `for` all walked one, and `iter(mv)`
    # alone said `'memoryview' object is not iterable` -- the same view, the
    # same walk, one spelling refused. The cursor steps through
    # `apy_getitem`, and `mv[i]` is the int CPython yields.
    #
    # AND `__class_getitem__` IS A CLASSMETHOD, which is why
    # `list.__class_getitem__(int)` takes the key alone where
    # `list.append(xs, 9)` takes a receiver first. Reached off the type it
    # was handed back UNBOUND and the call was `expected 2 arguments, got 1`
    # about a spelling CPython answers. Binding the TYPE -- not the
    # prototype -- is what makes both the call and the repr right: the alias
    # takes a type straight as its origin, and CPython's repr reads `of type
    # object at ...`. A bound classmethod is a `builtin_function_or_method`
    # and not the `method_descriptor` an unbound one would be, and not the
    # `method-wrapper` its dunder name would otherwise make it -- a slot is
    # filled on a VALUE, never on a type.
    "a_view_walks_and_a_classmethod_binds_its_type": """
        mv = memoryview(b"abc")
        it = iter(mv)
        print("named:", type(it).__name__)
        print("stepped:", next(it), next(it), next(it))
        try:
            next(it)
        except StopIteration:
            print("exhausted")
        print("again:", list(iter(mv)), [x for x in iter(mv)])
        # A `memory_iterator` CARRIES NO LENGTH HINT -- the one sized walk
        # without one, and `reversed(mv)` does have it.
        print("hint:", hasattr(iter(mv), "__length_hint__"),
              reversed(mv).__length_hint__())
        print("dir:", len(dir(iter(mv))), len(dir(type(iter(mv)))))
        print("honest:", [n for n in dir(iter(mv))
                          if not hasattr(iter(mv), n)])
        print("the other ways still work:", list(mv), [*mv], 98 in mv,
              list(reversed(mv)), len(mv))
        # AND THE CLASSMETHOD.
        for t in (list, tuple, dict, set, frozenset):
            got = t.__class_getitem__(int)
            print(" ", t.__name__, got, type(got).__name__)
        g = list.__class_getitem__
        print("kind:", type(g).__name__, "| name:", g.__name__)
        print("prefix:", repr(g)[:len("<built-in method __class_getitem__")])
        print("receiver:", repr(g).split(" of ")[1].split(" object")[0])
        print("call:", g(str), getattr(list, "__class_getitem__")(bytes))
        # OFF A VALUE IT BINDS THE TYPE TOO, which is what a classmethod is.
        v = [].__class_getitem__
        print("off a value:", type(v).__name__, v(int))
        # AND THE UNBOUND DESCRIPTORS BESIDE IT ARE UNCHANGED.
        print("beside it:", repr(list.append), type(list.append).__name__,
              repr(list.__len__))
        e = enumerate([1])
        print("cursor:", type(e).__class_getitem__(int),
              type(type(e).__class_getitem__).__name__)
    """,
    # A GENERATOR KNOWS WHERE ITS BODY IS. `dir(g)` answered the empty list
    # where CPython lists thirty-eight names, because seven of them had
    # nothing to answer with: `gi_running`, `gi_suspended`, `gi_yieldfrom`,
    # `gi_code`, `gi_frame`, `__del__` and `__class_getitem__`. The first
    # five are the frame, which the generator cell IS and the step function
    # is the code of; `gi_yieldfrom` is the one with nowhere to live, so the
    # lowered `yield from` records it.
    "a_generator_knows_where_its_body_is": """
        def gen(a, b=2):
            x = a + b
            yield x
            yield from [7, 8]
            return 99

        g = gen(1)
        d = dir(g)
        print(len(d), type(g).__name__)
        print([n for n in d if not n.startswith("_")])
        # THE LIST IS ONLY HONEST IF EVERY NAME ON IT ANSWERS.
        refused = []
        for n in d:
            try:
                getattr(g, n)
            except Exception as e:
                refused.append((n, type(e).__name__))
        print("refused:", refused)
        # WHERE THE BODY IS, before it has started. The code object is the
        # "enough of one" the `__code__` arm builds, so only what that one
        # carries is read off it -- `co_varnames` holds the parameters and
        # not the locals, and `co_flags` has neither `CO_GENERATOR` nor the
        # two every function sets.
        print(g.gi_running, g.gi_suspended, g.gi_yieldfrom)
        print(g.gi_code.co_name, g.gi_code.co_argcount,
              g.gi_code.co_kwonlyargcount, g.gi_code.co_posonlyargcount)
        print(g.gi_frame is None, g.gi_frame.f_code.co_name, g.gi_frame.f_back)
        print(type(g.__del__).__name__, type(g).__class_getitem__(int))
        # AND AFTER EACH STEP. `gi_suspended` is True between the first
        # `next` and the last; `gi_yieldfrom` is the sub-iterator only while
        # the `yield from` is running, and None on either side of it.
        print(next(g), g.gi_suspended, g.gi_running, g.gi_yieldfrom)
        print(next(g), type(g.gi_yieldfrom).__name__)
        print(next(g), type(g.gi_yieldfrom).__name__)
        for _ in g:
            pass
        print(g.gi_suspended, g.gi_yieldfrom, g.gi_frame is None)
        # A GENERATOR EXPRESSION IS ONE TOO, named for the scope it was
        # written in.
        q = (i for i in [1])
        print(q.gi_code.co_name, q.gi_suspended, q.gi_frame is None)
        # A COROUTINE AND AN ASYNC GENERATOR ARE THE SAME CELL HERE, and
        # CPython gives them these facts under `cr_` and `ag_` -- so `gi_` is
        # an AttributeError on both, and the two dunders every one of them
        # has are not.
        async def c():
            return 1
        async def ag():
            yield 1
        co = c()
        a = ag()
        for v in (co, a):
            try:
                v.gi_code
                print("answered")
            except AttributeError as e:
                print(type(v).__name__, "|", e)
        print(type(co.__del__).__name__, type(a.__del__).__name__)
        co.close()
    """,
    # AND AN EXPRESSION OVER CONSTANTS IS A CONSTANT. `10 ** 20` written
    # twice is ONE object in CPython -- its compiler folds the expression
    # and the table above then shares the answer -- and was two here,
    # because every operator was lowered as a call and the value was built
    # where it was written. Everything that decides WHETHER is CPython's
    # own line, measured: see `_const_fold`.
    #
    # ONE NAME PER LINE, never `a, b = x, y`: reading an element back out of
    # a tuple loses the handle on the interpreter, which is #145 and would
    # answer False here for a reason that is not this.
    "a_constant_expression_is_a_constant": """
        a1 = 10 ** 20
        a2 = 10 ** 20
        print("10**20:", a1 is a2, a1)
        # THE CEILINGS ARE EXACT, because a program can see the boundary.
        b1 = 2 ** 64
        b2 = 2 ** 64
        c1 = 2 ** 65
        c2 = 2 ** 65
        print("2**64:", b1 is b2, "| 2**65:", c1 is c2, b1, c1)
        e1 = "a" * 4096
        e2 = "a" * 4096
        f1 = "a" * 4097
        f2 = "a" * 4097
        print("text:", e1 is e2, f1 is f2, len(e1), len(f1))
        h1 = (1,) * 256
        h2 = (1,) * 256
        i1 = (1,) * 257
        i2 = (1,) * 257
        print("tuple:", h1 is h2, i1 is i2, len(h1), len(i1))
        p1 = 1 << 127
        p2 = 1 << 127
        q1 = 1 << 128
        q2 = 1 << 128
        print("shift:", p1 is p2, q1 is q2, p1 == q1 >> 1)
        # A GUARD IS ON THE OPERANDS, so `10 ** 40` is refused and the add
        # above it goes with it -- while two huge LITERALS still fold.
        r1 = 10 ** 40 + 1
        r2 = 10 ** 40 + 1
        s1 = 10000000000000000000000000000000000000000 + 1
        s2 = 10000000000000000000000000000000000000000 + 1
        print("compose:", r1 is r2, s1 is s2, r1 == s1)
        d1 = "a" * 3
        d2 = "a" * 3
        g1 = (1, 2) + (3,)
        g2 = (1, 2) + (3,)
        l1 = b"ab" * 3
        l2 = b"ab" * 3
        print("built:", d1 is d2, g1 is g2, l1 is l2, d1, g1, l1)
        j1 = 1.5 * 2
        j2 = 1.5 * 2
        k1 = 1j * 2
        k2 = 1j * 2
        print("numbers:", j1 is j2, k1 is k2, j1, k1)
        print("unary:", -5, ~7, not 0, +5, -(2 ** 3))
        print("odds:", 2 ** -1, 0 ** 100000, (-2) ** 64, 3 & 5, 3 | 5,
              3 ^ 5, 7 >> 1, 7 // 2, 7 % 3, 1 / 2)
        # A SUBSCRIPT OF CONSTANTS FOLDS, and a SLICE takes literal bounds
        # only -- which is CPython's behaviour rather than its intention,
        # and visible, so it is transcribed rather than tidied up.
        m1 = "abc"[1]
        m2 = "abc"[1]
        n1 = "abcd"[:3]
        n2 = "abcd"[:3]
        o1 = "abcd"[:-1]
        o2 = "abcd"[:-1]
        print("read:", m1 is m2, n1 is n2, o1 is o2, m1, n1, o1)
        print("slices:", (1, 2, 3)[1:], "abc"[::2], (1, 2, 3)[::2],
              "abc"[:], (1, 2)[0], b"a"[0], "abc"[1 + 1])
        # A MUTABLE RESULT IS NEVER SHARED, however equal two of them look.
        t1 = [1] * 2
        t2 = [1] * 2
        print("mutable:", t1 is t2, t1)
        # AND A SIGNED ZERO IS TWO CONSTANTS, not one: they compare equal
        # and hash alike, so the slot key cannot be the value alone.
        pz = 0.0
        nz = -0.0
        print("zero:", pz is nz, pz, nz, pz == nz)
        # NOTHING THAT RAISES IS FOLDED -- it fails where it is written.
        try:
            print(1 / 0)
        except ZeroDivisionError as e:
            print("raises:", e)
        try:
            print("abc"[10])
        except IndexError as e:
            print("raises:", e)
        try:
            print(1 % 0)
        except ZeroDivisionError as e:
            print("raises:", e)
        print("left alone:", "%s!" % 1, 5 % 3)
    """,
    # AND IDENTITY SURVIVES A READ AND A CALL. Reading a value back out of
    # a container, and returning one through a callable reached as a VALUE,
    # each minted a fresh handle on the interpreter -- so `xs[0] is xs[0]`
    # and `r = lambda x: x; r(v) is v` answered False there and True in
    # CPython and in both compiled runtimes, for every kind the interpreter
    # did not already intern by object.
    "identity_survives_a_read_and_a_call": """
        name = "hello world"
        big = 10 ** 20
        xs = [name, 1.5, big, b"bytes here", (1, 2), [3], 2j]
        print("elements:", [xs[i] is xs[i] for i in range(len(xs))])
        print("by name:", xs[0] is name, xs[2] is big)
        t = (name, 1.5)
        print("tuple:", t[0] is t[0], t[0] is name)
        d = {"k": name}
        print("dict:", d["k"] is d["k"], d["k"] is name)

        class C:
            def __init__(self):
                self.v = name

        c = C()
        print("attribute:", c.v is c.v, c.v is name)
        for item in [name]:
            print("loop:", item is name)

        # AND THROUGH A CALL REACHED AS A VALUE, which is where a DIRECT
        # call always agreed: the frontend lowers one to the symbol and the
        # handle never leaves the interpreter.
        def through(x):
            return x

        print("direct:", through(name) is name)
        alias = through
        print("aliased:", alias(name) is name, alias(big) is big)
        lam = lambda x: x
        print("lambda:", lam(name) is name, lam(big) is big)

        def outer():
            def inner(x):
                return x
            return inner(name) is name

        print("nested:", outer())
        made = lambda: "one object"
        print("built inside:", made() is made())
        print("mapped:", list(map(through, [name]))[0] is name)

        def second(a, b):
            return b

        print("second:", second(1, name) is name)
        pick = second
        print("second aliased:", pick(1, name) is name)

        class D:
            def m(self, x):
                return x

        print("method:", D().m(name) is name)
    """,
    # AND A COROUTINE AND AN ASYNC GENERATOR KNOW WHERE THEIR BODIES ARE.
    # They are the same cell as a generator here and three different types
    # in CPython, which gives each of them the same five facts under its own
    # prefix -- `gi_`, `cr_`, `ag_` -- and its own three methods. `dir()`
    # over either was the empty list, for the reason `dir(g)` was: seven
    # names apiece had nothing to answer with.
    "a_coroutine_and_an_async_generator_know_where_they_are": """
        import asyncio

        async def co(a, b=2):
            x = a + b
            await asyncio.sleep(0)
            return x

        async def ag(a, b=2):
            yield a
            yield b

        c = co(1)
        g = ag(1)
        print("kinds:", type(c).__name__, type(g).__name__)
        print("dir:", len(dir(c)), len(dir(g)))
        print("c named:", [n for n in dir(c) if not n.startswith("_")])
        print("g named:", [n for n in dir(g) if not n.startswith("_")])
        # THE LIST IS ONLY HONEST IF EVERY NAME ON IT ANSWERS.
        print("refused:", [(n, k) for k, v in (("c", c), ("g", g))
                           for n in dir(v) if not hasattr(v, n)])
        print("c where:", c.cr_running, c.cr_suspended, c.cr_await,
              c.cr_origin, c.cr_code.co_name, c.cr_code.co_argcount,
              c.cr_frame is None)
        print("g where:", g.ag_running, g.ag_suspended, g.ag_await,
              g.ag_code.co_name, g.ag_code.co_argcount, g.ag_frame is None)
        # AND `gi_` IS NOT THEIRS, which is the other half of the same rule.
        for v in (c, g):
            try:
                v.gi_code
                print("answered")
            except AttributeError as e:
                print(type(v).__name__, "|", e)
        # NOR IS EACH OTHER'S SET OF METHODS.
        print("crossed:", hasattr(c, "asend"), hasattr(g, "send"),
              hasattr(c, "send"), hasattr(g, "asend"))
        print("dunders:", type(c.__del__).__name__, type(g.__del__).__name__,
              type(c).__class_getitem__(int),
              type(g).__class_getitem__(int))

        async def drive():
            a = ag(1)
            print("aiter is self:", a.__aiter__() is a)
            print("asend:", await a.asend(None))
            print("mid:", a.ag_suspended, a.ag_running)
            print("anext:", await a.__anext__())
            try:
                await a.asend(None)
            except StopAsyncIteration:
                print("exhausted")
            print("spent:", a.ag_frame is None)
            b = ag(1)
            print("aclose:", await b.aclose())
            print("closed:", b.ag_frame is None)
            total = []
            async for v in ag(1):
                total.append(v)
            print("async for still works:", total)

        asyncio.run(drive())
        c.close()
    """,
    # AND A GENERATOR'S FRAME CAN HOLD A BOX. A generator never ran the
    # prologue that makes this frame's boxes and unpacks the ones it was
    # handed -- that one puts each in a REGISTER, which a generator has no
    # use for -- so what it captured from its enclosing scope read as None
    # and an inner `def` of its own captured a box nothing had made. Both
    # are everyday Python and both were silently wrong rather than refused.
    "a_generators_frame_holds_its_boxes": """
        def captures(n):
            def gen():
                yield n
                yield n + 1
            return list(gen())

        print("captured parameter:", captures(5))

        def captures_local():
            # NOT THROUGH A LAZY READ: a generator is drained where it is
            # walked here, so anything measuring the captured value between
            # two yields measures the eagerness rather than the capture.
            v = [1]

            def gen():
                yield v
                v.append(2)
                yield v

            got = list(gen())
            return len(got), got[0] is v, got[1] is v

        print("captured local:", captures_local())

        def writes_back():
            seen = []

            def gen():
                for i in range(3):
                    seen.append(i)
                    yield i

            return list(gen()), seen

        print("writes back:", writes_back())

        def gen_owns_a_box():
            v = 7

            def inner():
                return v

            yield inner()
            v = 9
            yield inner()

        print("owns a box:", list(gen_owns_a_box()))

        def gen_boxed_param(n):
            def inner():
                return n * 2

            yield inner()
            n = 10
            yield inner()

        print("boxed parameter:", list(gen_boxed_param(3)))

        def gen_siblings():
            def inner():
                return 7

            def mid():
                return inner()

            yield mid()

        print("siblings:", list(gen_siblings()))

        def two_levels(a):
            def gen():
                def deeper():
                    return a
                yield deeper()
            return list(gen())

        print("two levels:", two_levels(4))
        print("genexp:", list(x * 3 for x in range(3)))

        import asyncio

        async def coro_captures(n):
            async def inner():
                return n

            async def mid():
                return await inner()

            return await mid()

        print("coroutine siblings:", asyncio.run(coro_captures(6)))

        async def agen_captures(n):
            async def inner():
                return n + 1
            yield await inner()

        async def drive():
            out = []
            async for v in agen_captures(1):
                out.append(v)
            return out

        print("async generator:", asyncio.run(drive()))

        # AND A CAPTURED `*rest` OR `**kw` IS BOXED WITH ITS OWN VALUE IN
        # IT, which is the half a parameter-only fix would have left out.
        def gen_both(a, *rest, **kw):
            def inner():
                return a, list(rest), sorted(kw)
            yield inner()

        print("variadic:", list(gen_both(1, 2, 3, z=4)))
    """,
    # `c.__await__()` ANSWERS A WRAPPER, not the coroutine. CPython has a
    # `coroutine_wrapper` type and a program can see all three differences:
    # the wrapper is not `c`, `type(...).__name__` says so, and `dir()` of it
    # holds `close`, `send` and `throw` and nothing else -- none of the `cr_`
    # introspection the coroutine carries. Handing the coroutine back made
    # `w is c` True and `dir(w)` eight names longer.
    "an_await_answers_a_wrapper_and_not_the_coroutine": """
        import asyncio

        async def inner(n):
            await asyncio.sleep(0)
            return n * 2

        async def drive():
            co = inner(1)
            w = co.__await__()
            print("type:", type(w).__name__)
            print("not the coroutine:", w is co)
            print("iter:", iter(w) is w)
            print("dir:", sorted(n for n in dir(w) if not n.startswith("_")))
            # THE WRAPPER HAS NO BODY OF ITS OWN: every step goes to the
            # coroutine, and what it returned is what the StopIteration
            # carries.
            try:
                while True:
                    next(w)
            except StopIteration as e:
                print("drove:", e.value)
            # AND CLOSING ONE CLOSES WHAT IT WRAPS.
            second = inner(2)
            w2 = second.__await__()
            w2.close()
            print("closed:", type(w2).__name__)
            # THE COROUTINE ITSELF IS UNCHANGED, and so is a generator.
            third = inner(3)
            print("coroutine:", type(third).__name__,
                  "cr_code" in dir(third))
            print("awaited:", await third)
            g = (i for i in (1, 2))
            print("generator:", type(g).__name__, "gi_code" in dir(g))
            return 0

        asyncio.run(drive())
    """,
    # THE EMPTY AND ONE-CHARACTER SINGLETONS, and the identity rules that go
    # with them. CPython keeps one empty string, one string per latin-1
    # character, one empty bytes, one bytes per octet and one empty tuple --
    # and nothing else about a str, a bytes or a tuple is shared, so
    # `chr(256) is chr(256)` is False at the boundary. It also hands the
    # RECEIVER back where an immutable sequence operation has nothing to do:
    # a whole slice, a repeat by one, a concatenation with nothing. A list
    # and a bytearray copy in all three, because either can be written to
    # afterwards, and that is what `xs[:]` is for.
    #
    # Every one of these was False here. The runtime built a fresh cell for
    # each, so `"" is str()` and `s[:] is s` answered no, and a program that
    # printed either got a different answer from CPython.
    "the_empty_and_one_character_values_are_shared": """
        s = "abcd"
        b = b"abcd"
        t = (1, 2, 3)
        xs = [1, 2]
        ba = bytearray(b"ab")
        e = ""
        eb = b""
        et = ()
        one = "a"
        oneb = b"a"
        print("slice self:", s[:] is s, b[:] is b, t[:] is t)
        print("slice step1:", s[::1] is s, b[::1] is b, t[::1] is t)
        print("slice copies:", xs[:] is xs, ba[:] is ba)
        print("empties:", str() is e, bytes() is eb, tuple() is et)
        print("empty slices:", s[0:0] is e, b[0:0] is eb, t[0:0] is et)
        print("one char:", s[0:1] is one, b[0:1] is oneb)
        print("chr:", chr(97) is one, chr(233) is chr(233),
              chr(256) is chr(256))
        print("join:", "".join([]) is e, "-".join([]) is e)
        print("bjoin:", b"".join([]) is eb)
        print("mult one:", s * 1 is s, t * 1 is t, b * 1 is b)
        print("mult zero:", s * 0 is e, t * 0 is et)
        print("concat:", s + e is s, e + s is s, t + et is t, et + t is t)
        print("tuple conv:", tuple([]) is et, tuple(t) is t)
        print("bytes seq:", bytes([97]) is oneb, bytes(b) is b)
        print("ba fresh:", bytearray(b"") is not eb,
              bytearray() is not bytearray())
        # AND THE VALUES ARE STILL RIGHT, which is the half a sharing bug
        # would break silently: a bytearray that came back as the shared
        # empty bytes would be unwritable, and a bytes cell re-tagged from a
        # shared string would turn every later `""` into `b""`.
        ba.append(99)
        print("values:", repr(s[:]), repr(t[0:0]), repr(s[0:1]), repr(ba),
              repr(e), repr(eb))
    """,
    # A REPEAT SIZES AN ALLOCATION AND COPIES INTO IT, and both halves can
    # go wrong at a count the guard beside them does not cover.
    #
    # THE PRODUCT was computed and then looked at, which is not a check:
    # `b"abcd" * (2 ** 62 + 2)` is 2**64 + 8, wraps to EIGHT, and the
    # `malloc(9)` SUCCEEDED before the copy loop walked off the heap.
    # CPython divides first -- `if (n > 0 && Py_SIZE(a) > PY_SSIZE_T_MAX / n)`
    # -- and so does each of the three now, with CPython's own wording for
    # each kind: "repeated bytes are too long", "repeated string is too
    # long", and a bare MemoryError for a tuple or a list.
    #
    # THE COPY LOOP ran `k` times whatever the receiver's length was, so an
    # EMPTY receiver and a large count was `2 ** 62` zero-byte copies -- a
    # HANG, and one no overflow check can catch, because the product really
    # is zero. Measured: the C object runtime spun for 82 minutes on
    # `"" * (2 ** 62)` without printing a line. An immutable empty receiver
    # comes back as ITSELF, which is what CPython's `size == Py_SIZE(a)` test
    # says; a list and a bytearray are fresh, because either may be written
    # to afterwards.
    #
    # AND THE COUNT ITSELF has to fit an index. Every kind reports that as an
    # OverflowError; bytes alone said IndexError, because it asked for the
    # conversion with the form a SUBSCRIPT uses.
    "a_repeat_checks_its_product_and_copies_nothing_for_nothing": """
        BIG = 2 ** 62

        def rep(v, k):
            return v * k

        def show(label, f):
            try:
                r = f()
                print(f"{label:22} ok len={len(r)} {type(r).__name__}")
            except Exception as e:
                print(f"{label:22} {type(e).__name__}: {e}")

        # The product does not fit, at three widths and for five kinds.
        show("bytes big", lambda: rep(b"ab", BIG))
        show("str big", lambda: rep("ab", BIG))
        show("tuple big", lambda: rep((1, 2), BIG))
        show("list big", lambda: rep([1, 2], BIG))
        show("bytearray big", lambda: rep(bytearray(b"ab"), BIG))
        show("bytes maxsize", lambda: rep(b"ab", 2 ** 63 - 1))
        # The one that wrapped to a SMALL POSITIVE number and smashed the
        # heap: 2**64 + 8 as a size_t is 8, so malloc(9) succeeded.
        show("bytes wraps small", lambda: rep(b"abcd", BIG + 2))
        show("str wraps small", lambda: rep("abcd", BIG + 2))
        show("tuple wraps small", lambda: rep((1, 2, 3, 4), BIG + 2))

        # Nothing to copy, however large the count.
        show("str empty", lambda: rep("", BIG))
        show("bytes empty", lambda: rep(b"", BIG))
        show("bytearray empty", lambda: rep(bytearray(), BIG))
        show("tuple empty", lambda: rep((), BIG))
        show("list empty", lambda: rep([], BIG))
        show("str empty neg", lambda: rep("", -3))
        show("bytearray empty max", lambda: rep(bytearray(), 2 ** 63 - 1))

        # The count does not fit an index: one report for every kind.
        show("bytes count", lambda: rep(b"ab", 2 ** 63))
        show("str count", lambda: rep("ab", 2 ** 63))
        show("tuple count", lambda: rep((1,), 2 ** 63))
        show("list count", lambda: rep([1], 2 ** 63))
        show("bytearray count", lambda: rep(bytearray(b"ab"), 2 ** 63))

        # And which of them is the receiver itself afterwards.
        s, b, t, xs, ba = "", b"", (), [], bytearray()
        print("empty self:", rep(s, BIG) is s, rep(b, BIG) is b,
              rep(t, BIG) is t)
        print("empty fresh:", rep(xs, BIG) is xs, rep(ba, BIG) is ba)
        full = (1, 2)
        print("one self:", rep(full, 1) is full, rep(xs, 1) is xs)
    """,
    # A BUILTIN'S `__new__` BUILDS THE SUBCLASS IT IS HANDED. It is an
    # implicit staticmethod, so its first argument is the CLASS TO BUILD and
    # not a receiver of that type -- and the unbound-method check that every
    # other method wants said so: `str.__new__(S, "hi")` was `descriptor
    # '__new__' for 'str' objects doesn't apply to a 'type' object`, about a
    # class that is exactly what the call meant to name. `object.__new__`
    # refuses the same shapes CPython refuses, now that there is somewhere
    # else for them to go.
    "a_builtins_new_builds_the_subclass_it_is_handed": """
        class S(str):
            pass

        class T(tuple):
            pass

        class L(list):
            pass

        class D(dict):
            pass

        class Own:
            def __new__(cls, *rest):
                return super().__new__(cls)

        class WithInit:
            def __init__(self, x):
                self.x = x

        class Plain:
            pass

        # THE BUILTIN ITSELF ANSWERS A PLAIN ONE -- there is no class to put
        # it in.
        print("plain:", repr(str.__new__(str, "ab")),
              type(str.__new__(str, "ab")).__name__)
        print("sub:", repr(str.__new__(S, "ab")),
              type(str.__new__(S, "ab")).__name__)
        print("empty:", repr(str.__new__(S)), len(str.__new__(S)))
        # THE CONTENT IS TAKEN ONLY BY THE IMMUTABLE KINDS: a mutable builtin
        # fills in `__init__`, and an immutable one has nowhere else to.
        print("tuple:", repr(tuple.__new__(T, [1, 2])))
        print("list:", repr(list.__new__(L, [1, 2])))
        print("dict:", repr(dict.__new__(D)))
        # AND IT IS A REAL str, methods and all.
        s = str.__new__(S, "hi")
        print("methods:", s.upper(), s == "hi", isinstance(s, str), len(s))
        try:
            str.__new__(L, "x")
        except TypeError as e:
            print("wrong type:", e)
        try:
            str.__new__(5)
        except TypeError as e:
            print("not a type:", e)
        # `object.__new__` REFUSES A CLASS EXTENDING A BUILTIN, because it
        # would build the shell and leave the builtin half empty.
        try:
            object.__new__(S)
        except TypeError as e:
            print("unsafe:", e)
        try:
            object.__new__(S, 1)
        except TypeError as e:
            print("unsafe with arg:", e)
        # AN ARGUMENT BEYOND THE CLASS IS FOR `__init__` TO TAKE, and only
        # when there is one to take it.
        print("own:", type(object.__new__(Own)).__name__)
        try:
            object.__new__(Own, 1)
        except TypeError as e:
            print("own with arg:", e)
        print("with init:", type(object.__new__(WithInit, 1)).__name__)
        print("plain class:", type(object.__new__(Plain)).__name__)
        try:
            object.__new__(Plain, 1)
        except TypeError as e:
            print("plain with arg:", e)
        # AND THE CLASS-LEVEL QUESTION IS ANSWERABLE, which is how a caller
        # knows which `__new__` to reach for. A builtin kind has no class
        # cell and travels as a NAME, the shape `isinstance` already reads,
        # and `issubclass` refused it outright.
        print("issubclass:", issubclass(S, str), issubclass(T, tuple),
              issubclass(L, list), issubclass(D, dict))
        print("issubclass no:", issubclass(S, tuple), issubclass(Plain, str),
              issubclass(L, str))
        print("issubclass self:", issubclass(str, str), issubclass(S, object))
    """,
    # AND A CLASS LEARNS ITS BUILTIN BEFORE ITS METACLASS RUNS. The kind
    # used to be recorded once `apy_class_build` had ANSWERED, which for a
    # class with a metaclass is after the metaclass body has finished -- and
    # an `EnumMeta` makes every member inside that body. So a member of a
    # `class Colour(str, Enum)` was built against a class that did not yet
    # know it extended anything: `isinstance(Colour.RED, str)` was False,
    # `len(Colour.RED)` a TypeError and `Colour.RED == "red"` False. And
    # `StrEnum` was written as a plain `Enum`, so `Name.A.upper()` was an
    # AttributeError and `"-".join([Name.A])` refused the member by kind.
    "an_enum_member_is_the_builtin_its_enum_extends": """
        from enum import Enum, IntEnum, StrEnum, auto

        class Colour(str, Enum):
            RED = "red"
            BLUE = "blue"

        class Name(StrEnum):
            A = "a"
            B = "b"

        class Num(IntEnum):
            ONE = 1
            TWO = 2

        class Plain(Enum):
            X = auto()
            Y = auto()

        # A MIXIN MEMBER IS THE TEXT, and `Enum` writing `__str__` and
        # `__repr__` still decides how it PRINTS.
        print("mixin:", isinstance(Colour.RED, str), len(Colour.RED),
              Colour.RED == "red")
        print("mixin shows:", str(Colour.RED), repr(Colour.RED),
              Colour.RED.value)
        print("mixin methods:", Colour.RED.upper(), Colour.RED + "!",
              "-".join([Colour.RED, Colour.BLUE]))
        # A StrEnum MEMBER PRINTS AS ITS TEXT, which is what makes it one.
        print("strenum:", str(Name.A), repr(Name.A), Name.A == "a")
        print("strenum methods:", Name.A.upper(), "-".join([Name.A, Name.B]),
              Name.A in "abc", f"{Name.A}")
        # AND THE REST OF THE MODULE IS UNCHANGED.
        print("int:", str(Num.ONE), Num.ONE + 1, Num.ONE == 1)
        print("plain:", str(Plain.X), repr(Plain.X), Plain.X.value)
        print("lookup:", Colour("red") is Colour.RED,
              Colour["RED"] is Colour.RED, Name("a") is Name.A)
        print("members:", [m.name for m in Colour], [m.value for m in Name])
        print("keys:", {Name.A: 1}[Name.A], Colour.RED in Colour)
        # THE CLASS THE METACLASS SAW IS THE ONE THE STATEMENT BOUND.
        print("type:", type(Colour.RED) is Colour, type(Name.A) is Name)
    """,
    # AND A CLASS EXTENDING str IS A str WHEREVER ONE IS EXPECTED. `class
    # S(str)` makes something CPython's `find`, `join`, `replace`, `split`,
    # `in`, `int()`, `format` and `getattr` all take without a second
    # thought -- they read the C-level layout, which a subclass has. Every
    # one of them refused it here, by kind: `"-".join(["a", S("b")])` was
    # `sequence item 1: expected str instance, S found`.
    "a_builtin_extending_class_is_the_builtin_it_extends": """
        class S(str):
            def __str__(self):
                return "nope"

            def __repr__(self):
                return "<nope>"

        class Plain(str):
            pass

        s = S("b")
        p = Plain("b")
        # THE SEARCHES read the buffer and never ask what the class wrote,
        # which is why `s` above defines `__str__` and still finds as "b".
        print("find:", "abc".find(s), "abcb".rfind(s), "abc".index(s))
        print("count:", "abcb".count(s))
        print("in:", s in "abc")
        print("join:", "-".join(["a", s]), "-".join([s, p]))
        print("replace:", "abc".replace(s, "X"))
        print("split:", "a-b".split(Plain("-")), "a-b".rsplit(Plain("-")))
        print("affix:", "abc".startswith(Plain("a")),
              "abc".endswith(Plain("c")))
        # A TUPLE OF PREFIXES IS HELD TO THE SAME RULE, element by element.
        print("affix tuple:", "abc".startswith((Plain("z"), Plain("a"))),
              "abc".endswith((Plain("z"), Plain("c"))))
        print("trim:", "xbx".strip(Plain("x")), "xbx".lstrip(Plain("x")))
        print("pad:", "x".center(5, Plain("-")), "x".ljust(3, Plain(".")),
              "x".rjust(3, Plain("-")))
        print("affixes:", "abc".removeprefix(Plain("a")),
              "abc".removesuffix(Plain("c")))
        # THE SEPARATOR COMES BACK AS THE OBJECT IT WAS GIVEN, which for a
        # subclass means the INSTANCE and not the text inside it.
        parts = "a-b".partition(Plain("-"))
        print("partition:", parts, type(parts[1]).__name__)
        print("rpartition:", "a-b".rpartition(Plain("-")))
        # THE CONVERSIONS read the buffer too.
        print("int:", int(Plain("12")) + 1, int(Plain("ff"), 16))
        print("float:", float(Plain("1.5")) + 0.5)
        print("encode:", Plain("ab").encode(Plain("utf-8")),
              bytes(Plain("ab"), "utf-8"))
        print("decode:", b"ab".decode(Plain("utf-8")))
        print("ord:", ord(Plain("a")))
        print("maketrans:", "ab".translate(str.maketrans(Plain("a"),
                                                         Plain("z"))))
        # AND SO DO THE FORMAT STRING, THE SPEC AND THE SEPARATOR.
        print("format:", Plain("{}-{}").format(1, 2))
        print("percent:", Plain("%s!") % ("x",))
        print("spec:", "{:>4}".format(p))
        print("a", "b", sep=Plain("-"))
        # AN ATTRIBUTE NAME IS TEXT, whatever class the text arrived in.
        print("getattr:", getattr("abc", Plain("upper"))(),
              hasattr("abc", Plain("upper")))
        # THE UNBOUND SPELLING reaches the receiver gate without the unwrap
        # a written `p.upper()` gets from `apy_method_self`.
        print("unbound:", str.upper(p))
        # AND A COMPARISON IS STILL THE INSTANCE'S, because a subclass may
        # have written one: only the buffer readers unwrap.
        print("compare:", s == "b", p == "b", hash(p) == hash("b"))
        print("keys:", {p: 1}["b"], ["b"].index(p), p in ["b"])
    """,
    # `object.__init_subclass__` TAKES NOTHING AT ALL, which is what makes a
    # class keyword an error rather than a silence. CPython's is a
    # classmethod over a function of one parameter, `cls`, already bound by
    # the time a program can reach it -- so the signature a caller sees takes
    # no argument and no keyword.
    #
    # THE BINDING IS BY ARGUMENT HERE AND BY CLOSURE THERE, which is the
    # whole difficulty: `super().__init_subclass__(**kw)` passes the class
    # OUT, so a leading class is the bound receiver and is dropped before the
    # count is taken. On the compiled paths the optional slot arrives filled
    # with None for the same reason, and that is dropped too.
    #
    # WHAT IT COST BEFORE: `object.__init_subclass__(1, 2)` answered None on
    # the interpreter, the compiled paths counted the bound class as a
    # parameter and said `takes from 0 to 1 positional arguments`, and a
    # class keyword that reached the end of the chain was swallowed -- so
    # `class D(Plain, extra=1)` built a class on every path where CPython
    # raises.
    "objects_init_subclass_takes_nothing_at_all": """
        def w(label, f):
            try:
                print(f"{label:30} {f()!r}")
            except Exception as e:
                print(f"{label:30} !{type(e).__name__}: {e}")

        # A USER HOOK CONSUMES ITS KEYWORDS and ends by calling object's,
        # which must still answer: this is the ordinary spelling.
        class Base:
            def __init_subclass__(cls, tag=None, **kw):
                super().__init_subclass__(**kw)
                cls.tag = tag

        class C(Base, tag="x"):
            pass

        w("C.tag", lambda: C.tag)

        class Eat:
            def __init_subclass__(cls, **kw):
                super().__init_subclass__()
                cls.saw = sorted(kw)

        class E(Eat, a=1, b=2):
            pass

        w("E.saw", lambda: E.saw)

        # NOBODY WROTE ONE, so object's runs and the keyword is an error --
        # named after the class BEING CREATED, as CPython names it.
        class Plain:
            pass

        try:
            class DTop(Plain, extra=1):
                pass
        except TypeError as e:
            print(f"{'keyword, no hook':30} !TypeError: {e}")

        class D2(Plain):
            pass

        w("no keyword, no hook", lambda: D2.__name__)
        # AND WRITTEN OUT, where nothing is bound.
        w("isc()", lambda: object.__init_subclass__())
        w("isc(1)", lambda: object.__init_subclass__(1))
        w("isc(1, 2)", lambda: object.__init_subclass__(1, 2))
        w("isc(k=1)", lambda: object.__init_subclass__(k=1))
    """,
    # A CLASSMETHOD READ MINTS, AND A STATIC TYPE'S `__doc__` MINTS TOO --
    # two rules about WHERE an answer is built, which is what `is` measures.
    #
    # `__init_subclass__` AND `__subclasshook__` sit in `object.__dict__` as
    # classmethod_descriptors in CPython, so every read binds the class and
    # hands back a NEW bound method, exactly as `C.m is C.m` is False for a
    # written `@classmethod`. Here the dict holds a native and not a wrapper,
    # so the descriptor arm never saw them and both reads landed on the one
    # cell. The DICT ENTRY is still one cell -- only the attribute read
    # mints, and `object.__dict__["__subclasshook__"]` twice is one object in
    # CPython too.
    #
    # `__doc__` IS `type.__doc__`, A GETSET, for a class: CPython's
    # `type_get_doc` builds a fresh str from `tp_doc` for a STATIC type and
    # hands a HEAP type its dict entry straight back. So `object.__doc__ is
    # object.__doc__` and `str.__doc__ is str.__doc__` are both False while a
    # written docstring read twice is True. A VALUE's is the opposite: it is
    # an ordinary lookup that finds the one str the type carries, so
    # `"".__doc__ is "".__doc__` is True -- and the interpreter and the
    # compiled paths had the two halves exactly backwards from each other.
    "objects_a_classmethod_and_a_static_types_doc_mint_per_read": """
        def w(label, f):
            try:
                print(f"{label:30} {f()!r}")
            except Exception as e:
                print(f"{label:30} !{type(e).__name__}: {e}")

        class Plain:
            pass

        class Written:
            '''Its own text.'''

        d = object.__dict__
        # THE TWO CLASSMETHODS MINT PER READ, off the root and off a class.
        w("hook is hook",
          lambda: object.__subclasshook__ is object.__subclasshook__)
        w("isc is isc",
          lambda: object.__init_subclass__ is object.__init_subclass__)
        w("P.hook is P.hook",
          lambda: Plain.__subclasshook__ is Plain.__subclasshook__)
        w("P.isc is P.isc",
          lambda: Plain.__init_subclass__ is Plain.__init_subclass__)
        # THE DICT ENTRY IS ONE CELL, which is the other half of the rule.
        w("d[hook] is d[hook]",
          lambda: d["__subclasshook__"] is d["__subclasshook__"])
        w("d[isc] is d[isc]",
          lambda: d["__init_subclass__"] is d["__init_subclass__"])
        # AND THE REST OF object's NAMES ARE STILL ONE CELL EACH -- which
        # `functools.total_ordering` reads, through
        # `getattr(cls, op, None) is not getattr(object, op, None)`.
        w("lt is lt", lambda: object.__lt__ is object.__lt__)
        w("eq is eq", lambda: object.__eq__ is object.__eq__)
        w("repr is repr", lambda: object.__repr__ is object.__repr__)
        w("new is new", lambda: object.__new__ is object.__new__)
        # THEY STILL ANSWER WHAT THEY ANSWERED.
        w("hook(int)", lambda: object.__subclasshook__(int))
        w("P.hook(int)", lambda: Plain.__subclasshook__(int))
        w("isc()", lambda: object.__init_subclass__())
        w("P.isc()", lambda: Plain.__init_subclass__())
        # A TYPE'S `__doc__` MINTS; A VALUE'S IS ONE CELL; A WRITTEN ONE IS
        # THE OBJECT THE BODY BOUND.
        w("object doc", lambda: object.__doc__ is object.__doc__)
        w("str doc", lambda: str.__doc__ is str.__doc__)
        w("int doc", lambda: int.__doc__ is int.__doc__)
        w("list doc", lambda: list.__doc__ is list.__doc__)
        w("written doc", lambda: Written.__doc__ is Written.__doc__)
        w("written text", lambda: Written.__doc__)
        w("plain doc", lambda: Plain.__doc__)
        w("value str doc", lambda: "".__doc__ is "".__doc__)
        w("value int doc", lambda: (1).__doc__ is (1).__doc__)
        w("value list doc", lambda: [].__doc__ is [].__doc__)
        w("value doc == type", lambda: "".__doc__ == str.__doc__)
        w("obj doc == dict", lambda: object.__doc__ == d["__doc__"])
        w("obj doc is dict", lambda: object.__doc__ is d["__doc__"])
        w("d[doc] is d[doc]", lambda: d["__doc__"] is d["__doc__"])
    """,
    # `__class__` IS THE TWENTY-FOURTH NAME `object` CARRIES, and it was in
    # no dict at all. `dir(object)` listed it from a rule of its own, so
    # `len(object.__dict__)` was 23 against CPython's 24, `"__class__" in
    # object.__dict__` was False, and `sorted(object.__dict__) ==
    # sorted(dir(object))` was False -- three readings of one absence.
    #
    # A GETSET DESCRIPTOR AND NOT A SLOT, which is why the entry could not
    # simply be the `type` cell: CPython's is a pair of C functions CALLED
    # with whoever asked, so `object.__class__` is `type` and
    # `object().__class__` is `object`, and one plain slot cannot be both
    # answers. The entry is a descriptor cell and the two reads stay rules;
    # what the entry buys is that a program looking at the dict sees what
    # CPython's holds. It is not callable there and is not here.
    #
    # AND THE CLASS READ IS THE DATA DESCRIPTOR'S, which the dict test that
    # guarded the rule had backwards: `type.__dict__["__class__"]` is a DATA
    # descriptor, so for a CLASS read it wins over the class's own dict --
    # `class Own: __class__ = 7` has `Own.__class__` as `type` in CPython and
    # answered 7 here. The INSTANCE read is the other way round and still is:
    # `Own().__class__` IS 7, because there the MRO finds Own's entry before
    # object's getset.
    "objects_class_is_the_twenty_fourth_name_object_carries": """
        def w(label, f):
            try:
                print(f"{label:32} {f()!r}")
            except Exception as e:
                print(f"{label:32} !{type(e).__name__}: {e}")

        class Meta(type):
            pass

        class C:
            pass

        class M(metaclass=Meta):
            pass

        class S(str):
            pass

        class Own:
            __class__ = 7

        d = object.__dict__
        w("len(object.__dict__)", lambda: len(d))
        w("class in dict", lambda: "__class__" in d)
        w("dict == dir", lambda: sorted(d) == sorted(dir(object)))
        w("type(d[class])", lambda: type(d["__class__"]).__name__)
        w("d[class] callable", lambda: callable(d["__class__"]))
        w("d[class] is d[class]", lambda: d["__class__"] is d["__class__"])
        w("object.__class__ is type", lambda: object.__class__ is type)
        w("object().__class__ is object",
          lambda: object().__class__ is object)
        w("C.__class__ is type", lambda: C.__class__ is type)
        w("C().__class__ is C", lambda: C().__class__ is C)
        w("M.__class__ is Meta", lambda: M.__class__ is Meta)
        w("S('a').__class__ is S", lambda: S("a").__class__ is S)
        w("Own.__class__ is type", lambda: Own.__class__ is type)
        w("Own().__class__", lambda: Own().__class__)
        w("(1).__class__ is int", lambda: (1).__class__ is int)
        # AND `dir` LISTS IT ONCE, from the dict rather than from a push of
        # its own -- for a class, for an instance and for a builtin value.
        w("len(dir(object))", lambda: len(dir(object)))
        w("dir(object) class", lambda: dir(object).count("__class__"))
        w("dir(object()) class", lambda: dir(object()).count("__class__"))
        w("dir(C) class", lambda: dir(C).count("__class__"))
        w("dir(C()) class", lambda: dir(C()).count("__class__"))
        w("dir(1) class", lambda: dir(1).count("__class__"))
        # `dir(x)` AND `dir(type(x))` ARE ONE LIST for a class the program
        # wrote, which is what the shared chain walk is for: the C's instance
        # arm had no tail of its own and `dir(C())` was TWO names there where
        # the interpreter and the IR answered twenty-five.
        w("dir(C()) == dir(C)", lambda: dir(C()) == dir(C))
        w("dir(S('a')) == dir(S)", lambda: dir(S("a")) == dir(S))
        w("vars(C) class", lambda: "__class__" in vars(C))
        # AND THE LIST DOES NOT LIE, which is what makes widening it safe:
        # every name `dir` gives back has to answer `getattr`, and the
        # interpreter listed `__doc__` for an instance and refused it. A
        # class WITHOUT a docstring binds `__doc__ = None` rather than
        # nothing, and the instance walk asked through `find`, which answers
        # None for both a missing name and one bound to None.
        bad = []
        for recv, label in ((C(), "C()"), (S("a"), "S"), (object(), "o"),
                            (C, "C"), (1, "1"), ("", "s")):
            for n in dir(recv):
                try:
                    getattr(recv, n)
                except Exception as e:
                    bad.append(f"{label}.{n}: {type(e).__name__}")
        print("unanswered:", len(bad), sorted(bad))
        w("C().__doc__", lambda: C().__doc__)
        # AND A BUILTIN BASE'S DOCSTRING IS NOT THE SUBCLASS'S: `S("a")`
        # fell through to the held str and answered str's whole text.
        w("S('a').__doc__", lambda: S("a").__doc__)
    """,
    # A CALL THROUGH A VALUE CARRIES ALL ITS ARGUMENTS, and past three it
    # carried the program into a heap address instead. The x86-64 emitter
    # read an indirect call's TARGET after placing the arguments, out of
    # whichever register the allocator had given it -- and the argument
    # shuffle writes those registers:
    #
    #     mov %r10, %rdi    the callee, as allocated
    #     mov %rcx, %rdi    argument 0 -- the callee is gone
    #     mov %rdx, %r11    the shuffle breaking a cycle, through r11
    #     mov %rsi, %rdx
    #     mov %r11, %rsi
    #     mov %rdi, %r11    read the target back: argument 0
    #     call *%r11        jump to whatever argument 0 held
    #
    # THREE ARGUMENTS LEFT A REGISTER FREE and the callee happened to survive
    # in it, which is why this went unseen: every shape below with three or
    # fewer worked, and every shape with four segfaulted -- silently, exit
    # 139, with no output at all because the crash came before the first
    # `print` could flush.
    #
    # IT IS ONE BUG UNDER MANY FACES. `apy_invoke` in the C runtime and
    # `apy_call` in the IR one both reach a function pointer this way, so
    # `g(1, 2, 3, 4)`, `f(*(1, 2, 3, 4))`, `C(1, 2, 3, 4)` and a metaclass's
    # `super().__new__(mcls, name, bases, ns)` were all the same crash --
    # which is why no `TypedDict`, `enum` or `ABCMeta` could be compiled.
    "calls_through_a_value_carry_every_argument": """
        def w(label, f):
            try:
                print(f"{label:34} {f()!r}")
            except Exception as e:
                print(f"{label:34} !{type(e).__name__}: {e}")

        def f1(a):
            return a

        def f4(a, b, c, d):
            return a + b + c + d

        def f6(a, b, c, d, e, g):
            return a + b + c + d + e + g

        class Four:
            def __init__(self, a, b, c, d):
                self.v = a + b + c + d

            def m(self, a, b, c, d):
                return self.v + a + b + c + d

        # THE METACLASS IS THE SHAPE THAT FOUND IT. `super().__new__` takes
        # four arguments, so every metaclass with a `__new__` crashed.
        class Meta(type):
            def __new__(mcls, name, bases, ns):
                cls = super().__new__(mcls, name, bases, ns)
                cls.made = True
                return cls

        class WithMeta(metaclass=Meta):
            pass

        w("direct 4", lambda: f4(1, 2, 3, 4))
        g4 = f4
        w("value 4", lambda: g4(1, 2, 3, 4))
        w("splat 4", lambda: f4(*(1, 2, 3, 4)))
        g6 = f6
        w("value 6", lambda: g6(1, 2, 3, 4, 5, 6))
        w("splat 6", lambda: f6(*(1, 2, 3, 4, 5, 6)))
        w("value 1", lambda: f1(9))
        w("ctor 4", lambda: Four(1, 2, 3, 4).v)
        o = Four(1, 2, 3, 4)
        w("bound 4", lambda: o.m(1, 2, 3, 4))
        bm = o.m
        w("bound value 4", lambda: bm(1, 2, 3, 4))
        w("metaclass new", lambda: (WithMeta.made, type(WithMeta).__name__))
    """,
    # `object.__ne__` IS NOT `object.__eq__`, and the frontend's table said it
    # was: `OBJECT_DEFAULTS` mapped both names to `apy_default_eq` on adjacent
    # lines, so a written `object.__ne__(x, y)` computed EQUALITY.
    # `object.__ne__(1, 1)` answered True and `object.__ne__(1, 2)` False,
    # each the exact opposite of CPython's. It reads as a copied line rather
    # than a decision.
    #
    # AND `object.__eq__` DECLINES WHERE IT USED TO CLAIM.
    # `object_richcompare` answers True for identity and NotImplemented for
    # everything else -- it does not say two different objects are unequal, it
    # says it cannot judge. `object.__eq__(1, 2)` is NotImplemented there and
    # was False here, and False is a claim CPython does not make.
    #
    # THE `==` OPERATOR IS NOT THIS and is unchanged: `a == b` for two plain
    # instances is still False, because the identity fallback lives in the
    # operator -- where CPython's `do_richcompare` keeps it -- and not in the
    # method. Both are measured below, side by side, because moving the one
    # would have been the easy way to break the other.
    #
    # WHAT `__ne__` ASKS IS THE RECEIVER'S `__eq__`, and only the receiver's:
    # `object.__ne__(A(), B())` is NotImplemented even when B writes one, and
    # `object.__ne__(B(), A())` is False when B's says True. There is no
    # reflection here, which is what makes it different from `!=`.
    "objects_ne_is_not_eq_and_eq_declines_to_judge": """
        def w(label, f):
            try:
                print(f"{label:30} {f()!r}")
            except Exception as e:
                print(f"{label:30} !{type(e).__name__}: {e}")

        class P:
            pass

        class Q:
            def __eq__(self, other):
                return NotImplemented

        class R:
            def __eq__(self, other):
                return True

        class A2:
            pass

        class B2:
            def __eq__(self, other):
                return True

        class S(str):
            pass

        a, b, q = P(), P(), Q()
        # `object.__eq__` IS IDENTITY OR NOTHING.
        w("eq 1 2", lambda: object.__eq__(1, 2))
        w("eq 1 1", lambda: object.__eq__(1, 1))
        w("eq a b", lambda: object.__eq__(a, b))
        w("eq a a", lambda: object.__eq__(a, a))
        w("eq R R", lambda: object.__eq__(R(), R()))
        w("eq S S", lambda: object.__eq__(S("a"), S("a")))
        # `object.__ne__` DERIVES FROM THE RECEIVER'S `__eq__`.
        w("ne 1 2", lambda: object.__ne__(1, 2))
        w("ne 1 1", lambda: object.__ne__(1, 1))
        w("ne a b", lambda: object.__ne__(a, b))
        w("ne a a", lambda: object.__ne__(a, a))
        w("ne str", lambda: object.__ne__("a", "b"))
        w("ne list", lambda: object.__ne__([1], [1]))
        w("ne R", lambda: object.__ne__(R(), R()))
        w("ne Q", lambda: object.__ne__(Q(), Q()))
        w("ne q q", lambda: object.__ne__(q, q))
        # NO REFLECTION: only the LEFT operand's type is asked.
        w("ne A2 B2", lambda: object.__ne__(A2(), B2()))
        w("ne B2 A2", lambda: object.__ne__(B2(), A2()))
        # A CLASS EXTENDING A BUILTIN COMPARES AS THE BUILTIN.
        w("ne S S", lambda: object.__ne__(S("a"), S("a")))
        # AND THE OPERATORS ARE UNTOUCHED, which is the other half.
        w("a == b", lambda: a == b)
        w("a != b", lambda: a != b)
        w("a == a", lambda: a == a)
        w("a != a", lambda: a != a)
        w("q == q", lambda: q == q)
        w("q != q", lambda: q != q)
        w("S == S", lambda: S("a") == S("a"))
        w("1 == 1", lambda: 1 == 1)
        w("[1] != [1]", lambda: [1] != [1])
    """,
    # A STATICMETHOD OR CLASSMETHOD ON A BUILTIN TYPE HAS NO RECEIVER, and
    # both spellings ate the first argument as if it had one:
    #
    #     m = str.maketrans; m("ab", "xy")
    #     TypeError: if you give only one argument to maketrans it must be a
    #                dict
    #     list.__class_getitem__(int)
    #     TypeError: descriptor '__class_getitem__' for 'list' objects
    #                doesn't apply
    #
    # with `"ab"` and `int` each swallowed as a receiver that is not there.
    # `dict.keys(d)` and `str.upper(x)` DO mean exactly that, which is why
    # the rule is a table and not a guess: every row was read off CPython as
    # the names in a builtin type's `__dict__` whose value is a
    # `staticmethod` or a `classmethod_descriptor`.
    #
    # THE PROTOTYPE IS WHAT GETS BOUND, because it carries the KIND: a
    # classmethod needs it -- `bytes.fromhex` and `bytearray.fromhex` differ
    # only in what they build, which the last row measures -- and a
    # staticmethod ignores its receiver, so one rule serves both.
    # `__class_getitem__` is the exception that binds the TYPE, because its
    # body reads it as the alias's origin.
    "builtin_static_and_class_methods_take_no_receiver": """
        def w(label, f):
            try:
                print(f"{label:30} {f()!r}")
            except Exception as e:
                print(f"{label:30} !{type(e).__name__}: {e}")

        # WRITTEN OUT, then read as a value: one rule has to serve both.
        w("maketrans written", lambda: str.maketrans("ab", "xy"))
        m = str.maketrans
        w("maketrans value", lambda: m("ab", "xy"))
        w("maketrans three", lambda: m("ab", "xy", "z"))
        w("fromkeys written", lambda: dict.fromkeys([1, 2], 0))
        fk = dict.fromkeys
        w("fromkeys value", lambda: fk([1, 2], 0))
        w("from_bytes written", lambda: int.from_bytes(b"\x01\x02", "big"))
        ib = int.from_bytes
        w("from_bytes value", lambda: ib(b"\x01\x02", "big"))
        w("fromhex written", lambda: bytes.fromhex("41 42"))
        fh = bytes.fromhex
        w("fromhex value", lambda: fh("41 42"))
        w("float.fromhex written", lambda: float.fromhex("0x1.8p+1"))
        ff = float.fromhex
        w("float.fromhex value", lambda: ff("0x1.8p+1"))
        w("class_getitem written", lambda: list.__class_getitem__(int))
        cg = list.__class_getitem__
        w("class_getitem value", lambda: cg(int))
        w("dict class_getitem", lambda: dict.__class_getitem__((int, str)))
        # AND THE UNBOUND INSTANCE METHODS ARE UNTOUCHED, which is the other
        # half: for these the first argument IS the receiver.
        w("dict.keys unbound", lambda: list(dict.keys({"a": 1})))
        k = dict.keys
        w("dict.keys value", lambda: list(k({"a": 1})))
        w("str.upper unbound", lambda: str.upper("ab"))
        u = str.upper
        w("str.upper value", lambda: u("ab"))
        w("str.replace unbound", lambda: str.replace("aba", "a", "z"))
        w("str.join unbound", lambda: str.join("-", ["a", "b"]))
        w("bytes.hex unbound", lambda: bytes.hex(b"AB"))
        # THE KIND THE CLASSMETHOD WAS REACHED OFF DECIDES WHAT IT BUILDS,
        # which is why the prototype and not the type is bound.
        w("kinds differ", lambda: (type(bytes.fromhex("41")).__name__,
                                   type(bytearray.fromhex("41")).__name__))
    """,
    # THE NAMES A `class` STATEMENT CREATES BESIDE THE BODY'S, and the
    # ORDER they land in -- PEP 520 makes a class dict's order readable, so
    # where a name sits is part of the answer. CPython's is `__module__`,
    # `__firstlineno__`, the body, `__static_attributes__`, `__dict__`,
    # `__weakref__`, and last a `__doc__` the body did not write.
    #
    # TWO ARE THE COMPILER'S and two are `type.__new__`'s. `__firstlineno__`
    # is where the STATEMENT begins, which for a decorated class is the first
    # decorator's line; `__static_attributes__` is the names its functions
    # assign through `self`, and its rule is stranger than it sounds -- see
    # `_static_attributes`. `__dict__` and `__weakref__` stand for storage
    # the instance layout provides, which is why a class declaring
    # `__slots__` gets neither, a subclass of a class that has them gets
    # neither, and a builtin base answers for itself: the variable-sized ones
    # give no `__weakref__`, and an exception already carries an instance
    # dict so it takes only that.
    "a_class_statement_creates_four_names_beside_the_bodys": """
        class Plain:
            x = 1

            def m(self):
                self.y = 2
                self.z = 3

        class Doc:
            'A docstring.'

            k = 1

        class Sub(Plain):
            pass

        class Slotted:
            __slots__ = ("a",)

        class D(dict):
            pass

        class T(tuple):
            pass

        class E(Exception):
            tag = 1

        class Odd:
            def m(this):
                this.a = 1

            def n(self):
                self.b = 1

            @staticmethod
            def s():
                self = Plain()
                self.c = 1

            def deep(self):
                def inner():
                    self.d = 1
                return inner

            def aug(self):
                self.r += 1
                (self.s1, self.s2) = (1, 2)

        def w(label, f):
            try:
                print(f"{label:22} {f()!r}")
            except Exception as e:
                print(f"{label:22} !{type(e).__name__}: {e}")

        w("plain order", lambda: list(Plain.__dict__))
        w("doc order", lambda: list(Doc.__dict__))
        w("firstlineno", lambda: Plain.__firstlineno__)
        w("doc firstlineno", lambda: Doc.__firstlineno__)
        w("static", lambda: Plain.__static_attributes__)
        w("none static", lambda: Doc.__static_attributes__)
        # THE FOUR WAYS THE RULE SURPRISES, in one tuple: `this.a` is left
        # out, a staticmethod's local `self` is counted, a nested `def`
        # counts, and `self.r += 1` is not a store.
        w("odd static", lambda: Odd.__static_attributes__)
        for label, cls in (("sub", Sub), ("slotted", Slotted), ("dict", D),
                           ("tuple", T), ("exception", E)):
            w(label + " slots", lambda cls=cls: ("__dict__" in cls.__dict__,
                                                 "__weakref__" in cls.__dict__))
        w("descriptor kind", lambda: type(Plain.__dict__["__dict__"]).__name__)
        # AND THE STORAGE THEY STAND FOR, read off an INSTANCE: the class
        # carries the descriptor and the instance answers its own dict.
        w("instance dict", lambda: type(Plain().__dict__).__name__)
        w("instance weakref", lambda: Plain().__weakref__)
        w("slotted instance", lambda: Slotted().__dict__)
        w("runtime built", lambda: list(type("R", (), {}).__dict__))
        w("runtime sub", lambda: list(type("R", (Plain,), {}).__dict__))
        w("exception works", lambda: isinstance(E("x"), Exception))
    """,
    # PEP 3155: A CLASS QUALIFIES THROUGH WHATEVER IT WAS WRITTEN INSIDE.
    # `mk.<locals>.D` for a class written in a function, `C.Inner` for one
    # written in another class -- and the bare name at module level, which
    # is why a program that never nests a class sees nothing of this.
    #
    # THE NAME AND THE QUALNAME ARE SEPARATE, which is the whole shape of
    # the fix: CPython's REFUSALS say `'D' object has no attribute` for a
    # nested class and only its REPRS carry the dotted spelling. So the
    # class cell grew a field rather than having the dots put into the name
    # it already had -- see `apy_type_qualname` -- and the frontend writes
    # it, because where a `class` statement was written is a fact about the
    # source and nothing the runtime holds could recover it.
    "a_nested_class_qualifies_through_what_it_was_written_in": """
        class Plain:
            def m(self):
                return 1

            class Inner:
                def q(self):
                    return 2

        def mk():
            class D(Plain):
                def n(self):
                    return 3
            return D

        def twice():
            def inner():
                class Buried:
                    pass
                return Buried
            return inner()

        class Named:
            pass

        def w(label, f):
            try:
                print(f"{label:26} {f()!r}")
            except Exception as e:
                print(f"{label:26} !{type(e).__name__}: {e}")

        D = mk()
        w("module class", lambda: Plain.__qualname__)
        w("method", lambda: Plain.m.__qualname__)
        w("class in class", lambda: Plain.Inner.__qualname__)
        w("its method", lambda: Plain.Inner.q.__qualname__)
        w("class in function", lambda: D.__qualname__)
        w("its name", lambda: D.__name__)
        w("its method", lambda: D.n.__qualname__)
        w("two deep", lambda: twice().__qualname__)
        w("class repr", lambda: repr(D))
        w("inner repr", lambda: repr(Plain.Inner))
        w("instance repr", lambda: repr(D())[:21])
        w("type of instance", lambda: str(type(D())))
        # THE REFUSALS NAME THE CLASS PLAINLY, which is CPython's rule and
        # the reason the qualname is a field of its own.
        w("attribute miss", lambda: D().zz)
        w("type attribute miss", lambda: D.zz)
        w("no len", lambda: len(D()))
        # RENAMING MOVES ONE AND NOT THE OTHER.
        w("rename", lambda: (setattr(Named, "__name__", "Renamed"),
                             Named.__qualname__, Named.__name__)[1:])
        w("assign qualname", lambda: (setattr(Plain.Inner, "__qualname__",
                                              "Elsewhere"),
                                      Plain.Inner.__qualname__)[1])
        w("assign an int", lambda: setattr(Plain, "__qualname__", 1))
        # A CLASS BUILT AT RUN TIME QUALIFIES AS ITS NAME: there is no
        # statement behind it to have been written inside anything.
        w("built at run time", lambda: type("Runtime", (), {}).__qualname__)
        # THE ONE REFUSAL THAT DOES QUALIFY. CPython words
        # `object.__init_subclass__`'s complaint with the qualname while
        # every refusal about an INSTANCE says plainly `'D' object`, so the
        # two spellings are measured side by side.
        def kw_nested():
            class K(Plain, extra=1):
                pass
            return K

        w("nested keyword", lambda: kw_nested())
        try:
            class TopKeyword(Plain, extra=1):
                pass
        except Exception as e:
            print("top keyword", type(e).__name__, e)
    """,
    # AN EMPTY-BODIED EXCEPTION CLASS GOT NEITHER OF THE TWO NAMES. Its
    # `class` statement builds nothing -- there is nothing in the body to
    # build -- so the lowering binds an interned cell by name and returned,
    # skipping the `__module__` every other class statement writes: the same
    # statement printed `<class 'MyError'>` with `pass` in it and
    # `<class '__main__.MyError'>` with one attribute in it.
    #
    # THE TEST FOR "THIS CELL IS A BARE EXCEPTION TYPE" HAD TO MOVE WITH IT.
    # The interpreter asked whether the class dict was EMPTY, which was the
    # same answer for every program that could reach it until the dict
    # gained a `__module__`; the C asks whether the class has an `__init__`
    # or a `__new__`, which is what actually distinguishes a class the
    # program wrote over the top of a builtin exception name. The two now
    # ask the same question.
    "an_empty_bodied_exception_class_is_named_like_every_other": """
        class TopEmpty(ValueError):
            pass

        class TopBody(ValueError):
            tag = 1

        def mk():
            class InEmpty(ValueError):
                pass

            class InBody(ValueError):
                tag = 2
            return InEmpty, InBody

        def w(label, f):
            try:
                print(f"{label:24} {f()!r}")
            except Exception as e:
                print(f"{label:24} !{type(e).__name__}: {e}")

        A, B = mk()
        for label, cls in (("top empty", TopEmpty), ("top body", TopBody),
                           ("nested empty", A), ("nested body", B)):
            w(label, lambda cls=cls: (cls.__name__, cls.__qualname__,
                                      repr(cls)))
        # AND IT IS STILL AN EXCEPTION, which is what the class the lowering
        # binds is for: constructible, catchable, and carrying its argument.
        w("construct", lambda: str(A("boom")))
        w("isinstance", lambda: isinstance(A("boom"), ValueError))
        w("args", lambda: A("boom").args)
        try:
            raise A("thrown")
        except ValueError as e:
            print("caught by base", type(e).__name__, type(e).__qualname__)
        try:
            raise TopEmpty("flat")
        except TopEmpty as e:
            print("caught by name", str(e), repr(type(e)))
    """,
    # A CLASS REACHED AS A TYPE INHERITS, and did not. `P.__eq__`,
    # `P.__init__`, `P.__repr__` and eight more were every one an
    # AttributeError about an attribute Python guarantees; `S.upper` for a
    # `class S(str)` was one too; and `dir(P)` was the two names the class
    # body leaves behind where CPython answers twenty-nine.
    #
    # TWO LOOKUPS THE CHAIN DOES NOT LINK TO, and they are separate. The
    # builtin a class extends is a KIND and not a class, so `S.builtin` is
    # what makes S a str and the base walk reads dicts. And `object` is not
    # installed as a real base on anything -- deliberately, since making it
    # one would put `__eq__` into every lookup -- so the root of every chain
    # is reachable from nothing.
    #
    # THE ORDER IS WHAT MAKES IT RIGHT: the class's own body, then its base
    # chain, then its metaclass, then the builtin it extends, then object's.
    # `S.__repr__` is str's and not object's for exactly that reason, and a
    # class that writes its own `__eq__` still wins over both.
    "a_class_reached_as_a_type_inherits": """
        class P:
            pass

        class S(str):
            pass

        class Own:
            def __eq__(self, other):
                return True

            def upper(self):
                return "mine"

        class SOwn(str):
            def upper(self):
                return "mine"

        def w(label, f):
            try:
                got = f()
            except Exception as e:
                print(f"{label:24} !{type(e).__name__}: {e}")
                return
            print(f"{label:24} {got!r}")

        # OBJECT'S NAMES, EVERY ONE REACHABLE. What `type()` CALLS each of
        # them is a separate question this does not ask -- they read as
        # `method-wrapper` here where CPython has three finer names, which
        # is the value side's own divergence and older than this.
        for nm in ("__init__", "__eq__", "__ne__", "__repr__", "__str__",
                   "__hash__", "__format__", "__reduce__", "__sizeof__",
                   "__init_subclass__", "__subclasshook__", "__getattribute__",
                   "__setattr__", "__delattr__", "__dir__", "__lt__"):
            w(f"has {nm}", lambda n=nm: hasattr(P, n))
        # AND THEY ANSWER THE SAME THING `object` ANSWERS, which is what
        # makes the fallback a reach rather than a second implementation.
        w("P.__sizeof__", lambda: P.__sizeof__(P()) == object.__sizeof__(P()))
        w("P.__format__", lambda: P.__format__(1, "") == object.__format__(1, ""))
        w("P.__dir__ len", lambda: len(P.__dir__(P())) == len(dir(P())))
        # THE BUILTIN A CLASS EXTENDS, unbound as it is off a type.
        w("S.upper", lambda: type(S.upper).__name__)
        w("S.upper call", lambda: S.upper("abc"))
        w("S.join", lambda: S.join("-", ["a", "b"]))
        w("S.__len__", lambda: S.__len__("abc"))
        # `__new__` IS THE BUILTIN'S, not object's: it is an implicit
        # staticmethod whose first argument is the class to build.
        w("S.__new__", lambda: S.__new__(S, "zz"))
        w("S.__new__ kind", lambda: type(S.__new__(S, "zz")).__name__)
        w("S.__new__ empty", lambda: S.__new__(S))
        w("P.__new__ kind", lambda: type(P.__new__(P)).__name__)
        # A WRITTEN NAME STILL WINS over both inherited routes.
        w("Own.__eq__ is obj", lambda: Own.__eq__ is object.__eq__)
        w("Own eq", lambda: Own() == 1)
        w("SOwn.upper", lambda: SOwn("ab").upper())
        w("SOwn unbound", lambda: SOwn.upper(SOwn("ab")))
        # AND `dir` LISTS WHAT THE LOOKUP NOW ANSWERS, for the class and for
        # an instance alike -- they are one list in CPython.
        w("__eq__ in dir(P)", lambda: "__eq__" in dir(P))
        w("__new__ in dir(P)", lambda: "__new__" in dir(P))
        w("upper in dir(S)", lambda: "upper" in dir(S))
        w("upper in dir(S())", lambda: "upper" in dir(S("a")))
        w("__eq__ in dir(S)", lambda: "__eq__" in dir(S))
        w("dir(P) is dir(P())", lambda: dir(P) == dir(P()))
        w("dir(S) is dir(S())", lambda: dir(S) == dir(S("a")))
        # `dir(str)` and `dir(object)` must not have moved.
        w("dir(str) len", lambda: len(dir(str)))
        w("dir(object) len", lambda: len(dir(object)))
    """,
    # A BUILTIN BASE'S CONSTRUCTOR IS THE WHOLE OF ITS SIGNATURE. `bytes`
    # and `str` take three arguments, so `class B(bytes)` then
    # `B("Ab", "utf-8")` is `b'Ab'` and `class S(str)` then
    # `S(b"Ab", "utf-8")` is `'Ab'`. Both compiled paths reported `B() takes
    # no arguments` -- about a class that HAS a constructor, inherited, and
    # was handed exactly what it wants -- while the interpreter answered,
    # so the paths disagreed with each other as well as with CPython.
    #
    # THE KEYWORD MIX IS WHY THIS ROUTES THROUGH `apy_builtin_ctor` rather
    # than calling the two constructors directly, and why it sits AHEAD of
    # the one-argument branch: `B("Ab", encoding="utf-8")` has argc == 1, so
    # that branch would take it, pass the source alone, and report `string
    # argument without an encoding` about a call that gave one.
    "a_builtin_bases_whole_constructor_is_inherited": """
        class B(bytes):
            pass

        class S(str):
            pass

        def w(label, f):
            try:
                got = f()
            except Exception as e:
                print(f"{label:22} !{type(e).__name__}: {e}")
                return
            print(f"{label:22} {got!r} {type(got).__name__}")

        # The one-argument forms, which must keep working.
        w("B one arg", lambda: B(b"Ab"))
        w("B empty", lambda: B())
        w("B from int", lambda: B(3))
        w("B from list", lambda: B([65, 98]))
        w("S one arg", lambda: S("Ab"))
        w("S from int", lambda: S(65))
        # The two- and three-argument forms, positional and by name.
        w("B two args", lambda: B("Ab", "utf-8"))
        w("B three args", lambda: B("Ab", "utf-8", "strict"))
        w("B kwargs", lambda: B("Ab", encoding="utf-8"))
        w("B all kwargs", lambda: B(source="Ab", encoding="utf-8"))
        w("B kw errors", lambda: B("Ab", "utf-8", errors="strict"))
        w("S two args", lambda: S(b"Ab", "utf-8"))
        w("S three args", lambda: S(b"Ab", "utf-8", "strict"))
        w("S kwargs", lambda: S(b"Ab", encoding="utf-8"))
        # THE REFUSALS ARE THE CONSTRUCTOR'S OWN, not an arity message that
        # would hide what was actually wrong.
        w("B bad encoding", lambda: B("Ab", 5))
        w("S bad encoding", lambda: S(b"Ab", 5))
        w("B bad codec", lambda: B("Ab", "nosuch"))
        w("B four args", lambda: B("Ab", "utf-8", "strict", "x"))
        w("B unknown kw", lambda: B("Ab", nosuch="x"))
        w("B twice", lambda: B("Ab", "utf-8", encoding="utf-8"))
        # And the non-subclass spelling answers the same, as it always did.
        w("bytes two args", lambda: bytes("Ab", "utf-8"))
        w("bytes kwargs", lambda: bytes("Ab", encoding="utf-8"))
        w("str two args", lambda: str(b"Ab", "utf-8"))
    """,
    # `int()` AND `float()` TAKE A BYTES, and the refusal said so while
    # refusing one: `int() argument must be a string, a bytes-like object or
    # a real number, not 'bytes'` is worse than a plain refusal, because a
    # reader checking it against the argument concludes the value is not
    # what it is. The parse is byte-wise already and bytes shares the str
    # layout, so the arm serves both; a VIEW is flattened first.
    #
    # AND AN EMBEDDED NUL IS NOT A TERMINATOR, which the bytes rows are what
    # exposed: the parse copies into a C string and both callers measured it
    # with `strlen`, so `int("1\x002")` answered 1 and `float("1\x002")`
    # answered 1.0 where CPython raises. A wrong answer, not a missing
    # feature -- and a str bug, found only because bytes made it worth
    # asking twice.
    "int_and_float_read_a_bytes_and_stop_at_no_nul": """
        def w(label, f):
            try:
                print(f"{label:22} {f()!r}")
            except Exception as e:
                print(f"{label:22} !{type(e).__name__}: {e}")

        w("int bytes", lambda: int(b"12"))
        w("int bytearray", lambda: int(bytearray(b"12")))
        w("int memoryview", lambda: int(memoryview(b"12")))
        w("float bytes", lambda: float(b"1.5"))
        w("float bytearray", lambda: float(bytearray(b"1.5")))
        w("int bytes b16", lambda: int(b"ff", 16))
        w("int bytearray b16", lambda: int(bytearray(b"ff"), 16))
        w("int bytes spaces", lambda: int(b"  12  "))
        w("int bytes sign", lambda: int(b"-12"))
        w("int bytes under", lambda: int(b"1_2"))
        # THE REFUSALS name the value and repr it as the kind it is.
        w("int bytes bad", lambda: int(b"zz"))
        w("int bytes b16 bad", lambda: int(b"zz", 16))
        w("float bytes bad", lambda: float(b"zz"))
        w("int list", lambda: int([1]))
        w("float list", lambda: float([1]))
        # A NUL IS A BYTE LIKE ANY OTHER, wherever it falls.
        w("int str nul", lambda: int("1\\x002"))
        w("int bytes nul", lambda: int(b"1\\x002"))
        w("float str nul", lambda: float("1\\x002"))
        w("float bytes nul", lambda: float(b"1\\x002"))
        w("int bytes trailing", lambda: int(b"12\\x00"))
        w("int str leading", lambda: int("\\x0012"))
        # AND A SUBCLASS RIDES ON THE SAME GATES, through the held value.
        class B(bytes):
            pass

        class S(str):
            pass

        w("int B", lambda: int(B(b"12")))
        w("float B", lambda: float(B(b"1.5")))
        w("int B b16", lambda: int(B(b"ff"), 16))
        w("int S", lambda: int(S("12")))
        w("int S b16", lambda: int(S("ff"), 16))
    """,
    # AND THE BYTES TWIN OF IT, which could not be written until `class
    # B(bytes)` compiled at all: the base was refused with E0076, so the
    # whole of the bytes half of the subclass contract was unmeasurable.
    #
    # WHAT DIFFERS FROM THE STR TWIN IS THE POINT. Indexing yields an INT
    # where str yields a one-character str, so an element walk and a
    # substring search stop being the same thing -- which is what exposed
    # `in` falling back to iteration for both bases. `str()` of one is its
    # REPR, because bytes leaves `tp_str` at object's. And `__format__` is
    # object's too, so a non-empty spec is a TypeError where a str subclass
    # takes the whole mini-language.
    "a_bytes_extending_class_is_the_bytes_it_extends": """
        class B(bytes):
            def __repr__(self):
                return "<nope>"

        class Plain(bytes):
            pass

        b = B(b"Ab")
        p = Plain(b"Ab")
        # WHAT IT IS. The kind is the class, the builtin is behind it, and every
        # question a bytes answers it answers.
        print("kind:", type(b).__name__, isinstance(b, bytes), issubclass(B, bytes))
        print("len:", len(p), "index:", p[0], "slice:", p[:1], type(p[:1]).__name__)
        print("iter:", list(p), "in one:", b"A" in p, "in two:", b"Ab" in p)
        print("in absent:", b"zz" in p, "octet:", 65 in p)
        print("eq:", p == b"Ab", "hash:", hash(p) == hash(b"Ab"), "key:", {p: 1}[b"Ab"])
        # A METHOD ANSWERS A PLAIN BYTES, not a str and not the subclass.
        print("upper:", p.upper(), type(p.upper()).__name__)
        print("strip:", Plain(b" a ").strip(), "split:", Plain(b"a-b").split(b"-"))
        print("replace:", p.replace(b"A", b"z"), "hex:", p.hex())
        print("decode:", p.decode(), "unbound:", bytes.upper(p), bytes.decode(p))
        print("concat:", p + b"c", type(p + b"c").__name__)
        print("repeat:", p * 2, type(p * 2).__name__)
        # THE SEARCHES read the buffer and never ask what the class wrote, which is
        # why `b` above defines `__repr__` and still finds as b"Ab".
        print("find:", b"zAbz".find(b), b"zAbz".count(b), b"zAbz".index(b))
        print("join:", b"-".join([b, p]), "startswith:", b"Abc".startswith(p))
        print("pad:", b"x".center(5, Plain(b"-")), b"x".ljust(3, Plain(b".")))
        print("trim:", b"xAbx".strip(Plain(b"x")))
        # THE CONSTRUCTORS AND THE CONVERSIONS read the buffer too.
        print("bytes:", bytes(p), type(bytes(p)).__name__, "bytearray:", bytearray(p))
        print("str:", str(p, "utf-8"), "view:", memoryview(p).tobytes())
        print("maketrans:", p.translate(bytes.maketrans(Plain(b"A"), Plain(b"z"))))
        print("empty:", B(), "from int:", B(3), "from list:", B([65, 98]))
        print("new:", bytes.__new__(B, b"zz"), type(bytes.__new__(B, b"zz")).__name__)
        # `str()` OF ONE IS ITS REPR, as it is for a bytes: bytes leaves `tp_str` at
        # object's, which reaches `tp_repr` -- so a written `__repr__` DOES print.
        print("str of:", str(p), "repr:", repr(p), "written:", str(b), repr(b))
        # AND `__format__` IS object's. bytes has none of its own, so a non-empty
        # spec is a TypeError where a str subclass takes the whole mini-language.
        print("format:", format(p), repr(f"{p}"))
        try:
            format(p, "5")
        except TypeError as e:
            print("spec:", e)
        # A NO-OP HANDS BACK A DIFFERENT OBJECT, because the receiver is a subclass:
        # CPython's `PyBytes_CheckExact` refuses the handback and copies.
        q = Plain(b"Ab")
        print("self:", (p * 1) is p, p[:] is p, (p * 1) is (q * 1))
    """,
    # STR, BYTES AND BYTEARRAY WRITE A `__str__`, so it beats a subclass's
    # `__repr__`. `str(S("text"))` on a `class S(str)` answered `"'text'"` --
    # the REPR, quotes and all: six characters where the program wrote four,
    # and a text that no longer compared equal to itself. Every other builtin
    # leaves `tp_str` at object's, which reaches `tp_repr`, so `class
    # D(dict)` with a written `__repr__` does print with it.
    "a_builtin_extending_class_prints_the_builtins_text": """
        class S(str):
            pass

        class Mid(S):
            pass

        class SR(str):
            def __repr__(self):
                return "<SR>"

        class SS(str):
            def __str__(self):
                return "<SS>"

            def __repr__(self):
                return "<SS repr>"

        class D(dict):
            pass

        class DR(dict):
            def __repr__(self):
                return "<DR>"

        class L(list):
            pass

        class T(tuple):
            pass

        s = S("text")
        print("str:", str(s), "| len:", len(str(s)), "| eq:", str(s) == "text")
        print("repr:", repr(s))
        # THE TEXT REACHES EVERY FUNNEL, not just the written `str()`.
        print("print:", s)
        print("fstring:", f"{s}")
        print("percent:", "%s" % (s,))
        print("format:", "{}".format(s))
        # AN INHERITED `__str__` IS STILL THE BUILTIN'S, two levels down.
        print("mid:", str(Mid("mid")), repr(Mid("mid")))
        # A WRITTEN `__repr__` DOES NOT WIN under `str()` for a str
        # subclass, because `str.__str__` is found first...
        print("sr:", str(SR("text")), repr(SR("text")))
        # ...but a written `__str__` wins over both.
        print("ss:", str(SS("text")), repr(SS("text")))
        # AND EVERY OTHER BUILTIN KEEPS THE OLD RULE: str falls back to
        # repr, so a written one is what prints.
        d = D()
        d["a"] = 1
        print("dict:", str(d), repr(d))
        print("dict repr:", str(DR()), repr(DR()))
        print("list:", str(L([1, 2])), repr(L([1, 2])))
        print("tuple:", str(T((1,))), repr(T((1,))))
        # THE EMPTY CASE, where the repr is two characters and the text none.
        print("empty:", repr(str(S(""))), len(str(S(""))))
        # AND A CONTAINER SHOWS ITS ELEMENTS WITH REPR, which is unchanged.
        print("nested:", str([s]), str({"k": s}))
    """,
    # AND A CONVERSION THAT HAS NOTHING TO DO HANDS THE RECEIVER BACK.
    # `str(s)`, `bytes(b)`, `tuple(t)`, `frozenset(f)`, `float(x)`,
    # `complex(z)` and `int(n)` are each their own receiver in CPython when
    # it is already exactly that type -- there is nothing to copy about an
    # immutable object. The interpreter built a second one of every kind;
    # the compiled halves did for a tuple and a complex.
    "a_no_op_conversion_hands_the_receiver_back": """
        s = "spam here"
        b = b"spam here"
        t = (1, 2, 3)
        f = frozenset({1, 2})
        n = 10 ** 20
        x = 1.5
        z = 2j
        print("str:", str(s) is s)
        print("bytes:", bytes(b) is b)
        print("tuple:", tuple(t) is t)
        print("frozenset:", frozenset(f) is f)
        print("int:", int(n) is n)
        print("float:", float(x) is x)
        print("complex:", complex(z) is z)
        # AND THE MUTABLE ONES STILL COPY, because two mentions of a list
        # must be two objects however equal they look.
        xs = [1]
        d = {1: 2}
        print("mutable:", list(xs) is xs, dict(d) is d,
              type(set({1})).__name__, type(bytearray(b)).__name__)
        # A SUBCLASS IS CONVERTED, not handed back.

        class T(tuple):
            pass

        sub = T((1, 2))
        print("subclass:", type(tuple(sub)).__name__, tuple(sub) is sub,
              tuple(sub) == (1, 2))
        print("bool through int:", int(True), int(True) is True)
        # AND THE SAME THROUGH THE NAME AS A VALUE, which is a different
        # constructor path from the written form.

        def through(fn, v):
            return fn(v)

        print("as a value:", through(tuple, t) is t, through(str, s) is s,
              through(float, x) is x, through(frozenset, f) is f)
        print("values:", str(s), bytes(b), tuple(t), sorted(frozenset(f)),
              int(n), float(x), complex(z))
    """,
    # A LITERAL IS ONE OBJECT, module-wide. `"hello" is "hello"` is True in
    # CPython and was False here, and so was every other pair of equal
    # literals: two names bound to the same text, `b"ab" is b"ab"`, `1.5 is
    # 1.5`, `(1, 2) is (1, 2)`, `() is ()`. Only the integers the runtime
    # already shares agreed -- the bytes of a literal were interned into one
    # read-only global, and the CELL built from them was not.
    #
    # ACROSS THE WHOLE MODULE, which is CPython's line measured rather than
    # assumed: two functions each returning `"a b c"` return the same object
    # in 3.14, and so do a tuple constant and a big integer written in two
    # different places. Not only the identifier-like strings, and not only
    # within one code object.
    #
    # A TUPLE OF CONSTANTS IS ITSELF A CONSTANT, nested ones included, and a
    # LIST display never is: it is mutable, so two of them must be two
    # objects however equal they look.
    #
    # KEYED BY TYPE AS WELL AS VALUE, because `1 == 1.0 == True` and the
    # three are different constants -- sharing a slot would hand one of them
    # back as the wrong kind.
    #
    # `is` IS WRITTEN THROUGH A FUNCTION here because CPython warns about
    # `"a" is "a"` written out, and the warning is not what this is about.
    "literals_are_one_object_each": """
        def two(a, b):
            return a is b
        print("str:", two("hello", "hello"))
        a = "hello"
        b = "hello"
        print("two names:", a is b)
        print("small ints:", two(5, 5), two(256, 256), two(257, 257))
        print("empty str:", two("", ""))
        print("empty tuple:", two((), ()))
        print("bytes:", two(b"ab", b"ab"))
        print("float:", two(1.5, 1.5))
        print("complex:", two(2j, 2j))
        print("tuple:", two((1, 2), (1, 2)))
        print("nested tuple:", two(((1, 2), 3), ((1, 2), 3)))
        def f():
            return "a b c"
        def g():
            return "a b c"
        print("across fns:", f() is g())
        print("module vs fn:", two("a b c", f()))
        print("built at run time:", two("abc", "".join(["a", "b", "c"])))
        print("a list is never shared:", two([1], [1]))
        print("kinds stay apart:", two(1000, 1000.0), 1000 == 1000.0,
              type((1000.0,)[0]).__name__, type((1000,)[0]).__name__)
        # AND THE VALUES STILL WORK, which is the half a sharing bug would
        # break in silence: an operation that read a shared cell as its own
        # to write would corrupt every other mention of that literal.
        print("values:", "hello" + "!", (1, 2) + (3,), 1.5 * 2, b"ab" * 2,
              "ab" * 3, (1, 2)[1], len(""), 257 + 1, 2j * 2)
        xs = [1]
        xs.append(2)
        ys = [1]
        print("mutation is local:", xs, ys)
    """,
    "a_builtin_type_is_an_ordinary_value": """
        # A BUILTIN TYPE IS AN ORDINARY VALUE, and `memoryview` was the last one that was
        # not: naming it at all was `'memoryview' is a builtin that cannot be used as a
        # value`, so its constructor, its subscript and its statics were all out of reach.
        def show(label, f):
            try:
                print(label, f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        held = b"xy"
        print(memoryview, memoryview.__name__, memoryview.__qualname__)
        show("the written call", lambda: memoryview(held).tobytes())
        show("by keyword", lambda: memoryview(object=held).tobytes())
        show("through a name", lambda: (lambda f: f(held).tobytes())(memoryview))
        show("through map", lambda: [m.tobytes() for m in map(memoryview, [held, b"ab"])])
        show("out of a list", lambda: [memoryview][0](held).tobytes())
        show("its type", lambda: type(memoryview(held)) is memoryview)
        show("isinstance", lambda: isinstance(memoryview(held), memoryview))
        show("a subscript of the type", lambda: memoryview[int])
        show("the private constructor", lambda: memoryview._from_flags(held, 0).tobytes())
        show("it on a value too", lambda: memoryview(held)._from_flags(held, 0).tobytes())
        show("with nothing", lambda: memoryview())
        show("with too much", lambda: memoryview(held, 1))
        show("equal to itself", lambda: memoryview == memoryview)
    """,
    "a_view_hands_its_buffer_back": """
        def show(label, f):
            try:
                print(label, f())
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        # A VIEW BORROWS ITS SOURCE, and `release` hands the borrow back so the object
        # underneath can be resized again. Every operation on the view afterwards is
        # refused -- which is not the same as the view being empty.
        held = bytearray(b"abcd")
        view = memoryview(held)
        show("readonly view", lambda: view.toreadonly().readonly)
        show("the source still accepts", lambda: view.readonly)
        show("writing through it", lambda: view.toreadonly().__setitem__(0, 65))
        show("reading through it", lambda: view.toreadonly().tolist())
        show("a slice of it", lambda: view.toreadonly()[1:].readonly)
        show("release", lambda: view.release())
        show("releasing twice", lambda: view.release())
        show("then tolist", lambda: view.tolist())
        show("then len", lambda: len(view))
        show("then a subscript", lambda: view[0])
        show("then a field", lambda: view.readonly)
        show("then hex", lambda: view.hex())
        show("then wrapping it", lambda: memoryview(view))
        # THE `with` BLOCK IS THE SAME RELEASE, and what it binds is the view itself.
        def in_a_block():
            one = memoryview(bytearray(b"xy"))
            with one as got:
                inside = got.tolist(), got is one
            return inside, one
        show("inside the block", lambda: in_a_block()[0])
        show("released on the way out", lambda: in_a_block()[1].tolist())
        print([type(getattr(memoryview(b"ab"), n)).__name__ for n in
               ("release", "toreadonly", "__enter__", "__exit__")])

        # AND A NAME A `with` BODY BINDS IS READABLE AFTER IT. A with body is not a
        # scope: what it binds belongs to the enclosing function, exactly as an `if`
        # body's binding does.
        class Swallows:
            def __enter__(self):
                return 7
            def __exit__(self, *rest):
                return True
        class Propagates:
            def __enter__(self):
                return 7
            def __exit__(self, *rest):
                return None
        def bound_inside():
            with Propagates() as got:
                seen = got + 1
            return seen
        # THE BODY IS NOT GUARANTEED TO FINISH, which is the whole of what a true
        # `__exit__` means: the statement after the `with` runs having skipped the
        # rest of the body, so a name the body binds may be unbound there.
        def skipped_by_a_swallow():
            with Swallows():
                raise ValueError("x")
                never = 1
            return never
        def the_as_name_survives():
            with Swallows() as got:
                raise ValueError("x")
            return got
        def two_managers():
            with Propagates() as a, Propagates() as b:
                both = a + b
            return both
        def a_loop_inside():
            with Propagates():
                for i in (1, 2, 3):
                    last = i
            return last
        show("bound inside a with", bound_inside)
        show("skipped by a swallow", skipped_by_a_swallow)
        show("the as name survives", the_as_name_survives)
        show("two managers", two_managers)
        show("a loop inside", a_loop_inside)
    """,
    "traceback_positions": """
        try:
            (1).missing
        except AttributeError as e:
            tb = e.__traceback__
            print(tb is not None)
            print(tb.tb_lineno)
            code = tb.tb_frame.f_code
            print(hasattr(code, "co_positions"))
            rows = list(code.co_positions())
            print(len(rows) > 0, all(len(p) == 4 for p in rows))
            # Every row is a real span: it starts no later than it ends. A row of
            # Nones is CPython's for a synthetic instruction, and counts as one.
            print(all(p[0] is None or p[0] <= p[1] for p in rows))
            # And the line the traceback reports is one of them.
            print(tb.tb_lineno in [p[0] for p in rows])

        try:
            d = {}
            d["missing"]
        except KeyError as e:
            print(e.__traceback__.tb_lineno)

        try:
            n = 1 / 0
        except ZeroDivisionError as e:
            print(e.__traceback__.tb_lineno)

        try:
            raise ValueError("written out")
        except ValueError as e:
            print(e.__traceback__.tb_lineno)

        # An exception that was never raised has no traceback.
        print(ValueError("unraised").__traceback__)
    """,
    "buffers_grow_and_are_released": """
        # THE BLOCK ALLOCATOR, exercised through the things that use it. A
        # list's item array is the one allocation this runtime genuinely frees
        # -- it DOUBLES on growth and a slice assignment hands the old one
        # back -- so `runtime/blocks.py` puts size classes and free lists over
        # the arena, and every buffer below is handed out, grown, released and
        # handed out again.
        #
        # WHAT A BROKEN ALLOCATOR LOOKS LIKE HERE: not a crash, but an element
        # read out of a block that was reused while still live. So every case
        # checks CONTENTS after the reuse rather than only lengths.
        xs = []
        for i in range(200):
            xs.append(i)
        print(len(xs), xs[0], xs[199], sum(xs))

        # A release, then a rebuild that should take the same blocks back.
        xs[0:150] = []
        print(len(xs), xs[0], xs[-1])
        for i in range(300):
            xs.append(i * 2)
        print(len(xs), xs[49], xs[50], xs[-1])

        # Several live at once, so the free lists cannot hand the same block
        # to two of them.
        a = list(range(100))
        b = list(range(100, 200))
        c = list(range(200, 300))
        a[0:50] = []
        print(len(a), len(b), len(c), a[0], b[0], c[0], a[-1], b[-1], c[-1])
        for i in range(200):
            a.append(-i)
        print(len(a), a[0], a[49], a[50], a[-1], b[0], b[99], c[0], c[99])

        # Dicts grow the same way, two buffers at a time.
        d = {}
        for i in range(150):
            d[i] = i * i
        print(len(d), d[0], d[149], sorted(d)[:3])
        for i in range(75):
            del d[i]
        print(len(d), d[75], d[149], min(d), max(d))

        # Sets, and a nesting so the blocks are interleaved rather than
        # allocated and freed in order.
        s = set()
        for i in range(200):
            s.add(i % 91)
        print(len(s), min(s), max(s))
        rows = [list(range(k, k + 30)) for k in range(40)]
        for row in rows:
            row[0:10] = []
        print(len(rows), len(rows[0]), rows[0][0], rows[39][0], rows[39][-1])

        # Tuples share the sequence buffer and are built once.
        t = tuple(range(120))
        print(len(t), t[0], t[119], t[60])

        # An empty list, a one-element list and a list that never grows are
        # the three sizes the smallest class has to get right.
        print(len([]), len([1]), [1][0], len([1, 2]), [1, 2][1])
    """,
    "extending_a_builtin": """
        # `class D(dict)` and its four siblings, which `collections` is built
        # on. An instance carries a real dict/list/tuple/set and DELEGATES to
        # it for everything the class body does not define -- attributes,
        # iteration, `in`, `len`, the operators, `repr`, `hash` and the
        # constructor. Every one of those was a separate entry point and each
        # is here, because getting one wrong is a silently wrong answer rather
        # than an error: the object still claims `isinstance(d, dict)`.
        class D(dict):
            def __missing__(self, k):
                return "missing:" + str(k)


        d = D()
        d["a"] = 1
        print(d, d["a"], len(d), sorted(d.keys()), d["zz"])
        print(sorted(d.items()), d.get("a"), d.get("q", 7))
        print(isinstance(d, dict), d == {"a": 1}, dict(d))
        print([k for k in d], sorted(k for k in d), list(d))
        print("a" in d, "zz" in d)
        d.update({"b": 2})
        print(sorted(d.items()), d.pop("b"), sorted(d.items()))


        class Init(dict):
            def __init__(self, *a, **k):
                super().__init__(*a)
                self.tag = "t"


        i = Init({"x": 1})
        print(i, i.tag, len(i))


        class L(list):
            def second(self):
                return self[1]


        l = L([3, 1, 2])
        print(l, l.second(), len(l), isinstance(l, list))
        l.append(4)
        l.sort()
        print(l, l.index(3), l.count(1), l[1:], list(reversed(l)))


        class T(tuple):
            def __new__(cls, *args):
                return super().__new__(cls, list(args))


        t = T(1, 2, 3)
        print(t, len(t), t[1], t + (4,), t == (1, 2, 3))
        a, b, c = t
        print(a, b, c, len({T(1, 2), T(1, 2)}), hash(t) == hash((1, 2, 3)))


        class S(set):
            pass


        s = S([1, 2])
        s.add(3)
        print(sorted(s), 2 in s, len(s))


        # THE CLASS BODY WINS over the builtin, which is what lets a `Counter`
        # define `update` next to `dict.update`.
        class Own(dict):
            def keys(self):
                return "mine"

            def __repr__(self):
                return "Own!"


        o = Own()
        o["k"] = 1
        print(o.keys(), repr(o), len(o), o["k"])


        # A plain instance is untouched by any of it.
        class Plain:
            def __getattr__(self, name):
                return "fallback:" + name


        print(Plain().whatever, Plain().append)
    """,
    "spread_calls_onto_every_callable": """
        class B:
            def m(self, a, b):
                return (a, b)


        class C(B):
            def m(self, *args):
                return super().m(*args)


        def plain(a, b, c=0):
            return (a, b, c)


        xs = [1, 2]
        print(C().m(*xs))
        print(plain(*xs))
        print(plain(*xs, c=3))
        print(plain(**{"a": 1, "b": 2}))
        print(B().m(*xs))
        # `max(*xs)` IS `max(xs)` -- both ask for the largest of these.
        print(max(*xs), min(*xs))
        print(max(*[[1], [2, 3]]))
        print(sorted(*[[3, 1, 2]]))
        # `str.format` is chosen by NAME at the call site, so the spread has to
        # reach it the same way an ordinary call does.
        print("{}-{}".format(*xs))
        print("{a}!".format(**{"a": 5}))
        print("{}{}{c}".format(*xs, c="!"))
        print(*xs, sep="/")

        d = {"a": 1}
        print(sorted(dict(**d, b=2).items()))
        print(sorted(dict(**d, **{"c": 3}).items()))
        print(sorted(dict(x=1, y=2).items()))
        print(sorted(dict(d).items()), dict())
    """,
    "await_in_slices_specs_and_asserts": """
        import asyncio


        async def v(x):
            await asyncio.sleep(0)
            return x


        async def ag(n):
            for i in range(n):
                await asyncio.sleep(0)
                yield i


        def plain(a, b):
            return (a, b)


        async def main():
            out = []
            xs = [10, 20, 30, 40]
            # A slice whose bounds suspend.
            out.append(xs[await v(1):await v(3)])
            out.append(xs[await v(0):await v(4):await v(2)])
            # A format spec that suspends.
            out.append(f"{7:>{await v(4)}}")
            # An assert whose message suspends.
            try:
                assert await v(False), await v("why")
            except AssertionError as e:
                out.append(str(e))
            # An async comprehension held across a call.
            out.append(plain([x async for x in ag(2)], 1))
            out.append({"k": [x async for x in ag(2)]})
            out.append(len([x async for x in ag(3)]) < 9)
            # `**` with awaits in the values.
            out.append(plain(**{"a": await v(1), "b": await v(2)}))
            return out


        for one in asyncio.run(main()):
            print(one)
    """,
    "generators_and_coroutines_mixed": """
        import asyncio


        async def v(x):
            await asyncio.sleep(0)
            return x


        async def agen(n):
            # An async generator: `yield` and `await` in one frame, and both of them
            # inside expressions.
            total = 0
            for i in range(n):
                total += await v(i)
                yield (await v(i), total)


        async def main():
            out = []
            async for pair in agen(3):
                out.append(pair)
            # An async comprehension, which is a frame of its own.
            out.append([x async for x in agen(2)])
            out.append([await v(i) for i in range(3)])
            return out


        def plain_gen():
            got = []
            got.append((yield "a"))
            got.append((yield "b"))
            return got


        g = plain_gen()
        g.send(None)
        g.send(1)
        try:
            g.send(2)
        except StopIteration as stop:
            print(stop.value)
        for one in asyncio.run(main()):
            print(one)
    """,
    "yield_in_every_expression_position": """
        def plain(a, b, c=0):
            return (a, b, c)


        class Box:
            def __init__(self):
                self.items = []

            def take(self, a, b):
                self.items.append((a, b))
                return len(self.items)


        def gen():
            # `yield` SUSPENDS exactly as `await` does, so every expression position
            # that holds a value across one has the same problem -- and every one of
            # these used to produce invalid IR.
            xs = [10, 20, 30]
            d = {"k": 5}
            b = Box()
            out = []
            out.append(plain((yield 1), (yield 2)))
            out.append(plain(1, (yield 3), c=(yield 4)))
            out.append(b.take((yield 5), (yield 6)))
            out.append(b.items)
            out.append(xs[(yield 0) % 3])
            out.append(d["k" if (yield "k") else "k"])
            out.append((yield 7) + (yield 8))
            out.append([(yield 9), 2, (yield 10)])
            out.append({"a": (yield 11), "b": (yield 12)})
            out.append(((yield 13), (yield 14)))
            out.append((yield 15) < (yield 16))
            out.append(f"{(yield 17)}-{(yield 18)}")
            n = 1
            n += (yield 19)
            out.append(n)
            out.append(plain(*[(yield 20), (yield 21)]))
            out.append(sorted({(yield 22), (yield 23)}))
            return out


        g = gen()
        sent = 0
        try:
            got = g.send(None)
            while True:
                sent += 1
                # WHAT IS SENT BACK is what the expression evaluates to, so the values
                # below are the sent ones and not the yielded ones.
                got = g.send(sent)
        except StopIteration as stop:
            for one in stop.value:
                print(one)
        print("yields", sent)
    """,
    "await_in_every_expression_position": """
        import asyncio


        async def v(x):
            await asyncio.sleep(0)
            return x


        def plain(a, b, c=0):
            return (a, b, c)


        class Box:
            def __init__(self):
                self.items = []

            def take(self, a, b):
                self.items.append((a, b))
                return len(self.items)


        async def main():
            out = []
            # A call with an await among the arguments.
            out.append(plain(await v(1), await v(2)))
            out.append(plain(1, await v(2), c=await v(3)))
            # A method call on an object, with awaits.
            b = Box()
            out.append(b.take(await v("a"), await v("b")))
            out.append(b.items)
            # A subscript indexed by an await.
            xs = [10, 20, 30]
            out.append(xs[await v(1)])
            d = {"k": 5}
            out.append(d[await v("k")])
            # Binary operators either side of a suspension.
            out.append(await v(3) + await v(4))
            out.append([await v(1), 2, await v(3)])
            out.append({"a": await v(1), "b": await v(2)})
            out.append((await v(1), await v(2)))
            # A comparison and a boolean operator.
            out.append(await v(1) < await v(2))
            # An f-string with awaits in it.
            out.append(f"{await v('x')}-{await v('y')}")
            # A nested call.
            out.append(plain(plain(await v(1), 2), await v(3)))
            return out


        for one in asyncio.run(main()):
            print(one)
    """,
    "asyncio_tasks_and_groups": """
        import asyncio

        log = []


        async def slow():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                log.append("cancelled")
                raise


        async def val(v):
            await asyncio.sleep(0)
            return v


        async def late():
            await asyncio.sleep(5)
            return "never"


        async def main():
            task = asyncio.create_task(slow())
            await asyncio.sleep(0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                log.append("awaited-cancel")

            # A task that finishes normally: the result is readable after.
            t = asyncio.create_task(val(7))
            log.append(await t)
            log.append((t.done(), t.result(), t.cancelled()))

            # `wait_for` gives up at its deadline.
            try:
                await asyncio.wait_for(late(), timeout=0.01)
            except asyncio.TimeoutError:
                log.append("timeout")
            # And answers normally when the coroutine is quick enough.
            log.append(await asyncio.wait_for(val(3), timeout=1))

            # A task group does not finish while its children are running.
            async with asyncio.TaskGroup() as tg:
                made = [tg.create_task(val(n)) for n in (1, 2, 3)]
            log.append([t.result() for t in made])
            return log


        print(asyncio.run(main()))
        print(asyncio.TimeoutError is TimeoutError)
    """,
    "async_iterator_on_a_class": """
        import asyncio


        class Counting:
            def __init__(self, n):
                self.n = n
                self.i = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.i >= self.n:
                    raise StopAsyncIteration
                # A REAL SUSPENSION between items, which is what makes this more than
                # a synchronous loop wearing an `async` hat.
                await asyncio.sleep(0)
                self.i += 1
                return self.i


        class Pairs:
            # `__aiter__` may answer something OTHER than self.
            def __init__(self, items):
                self.items = items

            def __aiter__(self):
                return Counting(len(self.items))


        async def main():
            out = []
            async for v in Counting(3):
                out.append(v)
            async for v in Counting(0):
                out.append("never")
            async for v in Pairs(["a", "b"]):
                out.append(("pair", v))
            # An async generator still works, and is its own iterator.
            async def gen():
                for i in range(2):
                    await asyncio.sleep(0)
                    yield i * 10
            async for v in gen():
                out.append(v)
            return out


        print(asyncio.run(main()))
    """,
    "class_body_runs_as_a_block": """
        class C:
            values = [1, 2, 3]
            # The OUTERMOST iterable is evaluated in the class scope; everything else
            # in the comprehension is not, which is what the NameError below is.
            doubled = [v * 2 for v in values]
            try:
                bad = [v * len(values) for v in range(2)]
            except NameError:
                bad = "NameError"

            if len(values) > 2:
                flag = "big"
            else:
                flag = "small"

            total = 0
            for n in values:
                total = total + n

            def scaled(self):
                # A METHOD reads the class attribute through `self`, which is the
                # ordinary way and unaffected by any of the above.
                return [x * self.total for x in self.values]

            label = flag + ":" + str(total)


        print(C.doubled, C.bad)
        print(C.flag, C.total, C.label)
        print(C.n)
        print(C().scaled())

        # A `try`/`except ImportError` around an import, which is how a class picks
        # an optional dependency.
        class D:
            try:
                import math
                have = True
            except ImportError:
                have = False

            name = "D" if have else "?"


        print(D.have, D.name)

        # A while loop, and a name bound only in one branch.
        class E:
            i = 0
            while i < 3:
                i = i + 1
            if i == 3:
                done = True
            parts = []
            for w in ("a", "b"):
                parts.append(w * i)


        print(E.i, E.done, E.parts)
    """,
    "exception_class_with_a_body": """
        class AppError(Exception):
            # An exception class with a body, which is how most are written.
            kind = "app"

            def __init__(self, code, message):
                super().__init__(f"{code}: {message}")
                self.code = code

            def summary(self):
                return f"{self.kind}/{self.code}"


        class NotFound(AppError):
            def __init__(self, what):
                super().__init__(404, what)
                self.what = what


        try:
            raise AppError(500, "boom")
        except AppError as e:
            print(e.code, str(e), e.args)
            print(e.summary(), e.kind)
            print(type(e).__name__)

        # A subclass reaches its base's `__init__` through `super()`.
        try:
            raise NotFound("page")
        except AppError as e:
            print(e.code, e.what, str(e))
            print(type(e).__name__, isinstance(e, AppError))

        # It is still caught by its BASE, which is what the hierarchy is for.
        try:
            raise NotFound("again")
        except Exception as e:
            print("base caught", e.code)

        # An exception class with an EMPTY body still works, and is not a class with
        # a body in disguise.
        class Plain(ValueError):
            pass


        try:
            raise Plain("p")
        except ValueError as e:
            print(type(e).__name__, str(e), e.args)

        # Built without raising, then raised later.
        held = AppError(1, "held")
        print(held.code, str(held))
        try:
            raise held
        except AppError as e:
            print("reraised", e.code)

        # The class is a value: `issubclass` and `__name__` both answer.
        print(AppError.__name__, issubclass(NotFound, AppError))

        # A default `__init__` that never calls super still reads back what it was
        # passed, because `args` is set before it runs.
        class Quiet(RuntimeError):
            def __init__(self, tag):
                self.tag = tag


        try:
            raise Quiet("t")
        except RuntimeError as e:
            print(e.tag, e.args)
    """,
    "template_strings_keep_the_pieces": """
        name = "world"
        n = 5
        w = 8

        t = t"hello {name}!"
        print(type(t).__name__)
        print(t.strings)
        print([i.expression for i in t.interpolations])
        print([i.value for i in t.interpolations])
        print(t.values)

        # The conversion and the spec are RECORDED, not applied.
        t2 = t"{n!r:>{w}}"
        i = t2.interpolations[0]
        print(i.expression, i.conversion, repr(i.format_spec), i.value)
        print(t2.strings)

        # Adjacent fields leave an empty piece between them, so `strings` stays one
        # longer than `interpolations`.
        t3 = t"{n}{name}"
        print(t3.strings, len(t3.interpolations))

        # No fields at all.
        t4 = t"plain"
        print(t4.strings, t4.interpolations, t4.values)

        # Two templates are the same type.
        print(type(t) is type(t4))

        # Joining is the CONSUMER's job, which is the whole point.
        print(t.strings[0] + str(t.values[0]) + t.strings[1])

        # The expression source survives, which an f-string throws away.
        total = t"{n + w}"
        print(total.interpolations[0].expression, total.values[0])
    """,
    "except_star_divides": """
        log = []

        # A BARE exception is a group of one to `except*`.
        try:
            raise ValueError("v")
        except* ValueError as eg:
            log.append(("bare", type(eg).__name__, len(eg.exceptions)))

        # Nothing matches: the ORIGINAL propagates, not a wrapper.
        try:
            try:
                raise KeyError("k")
            except* ValueError as eg:
                log.append(("wrong", 1))
        except KeyError as e:
            log.append(("through", type(e).__name__))

        # Part matches, part is left over.
        try:
            try:
                raise ExceptionGroup("g", [ValueError("a"), KeyError("b")])
            except* ValueError as eg:
                log.append(("half", len(eg.exceptions)))
        except ExceptionGroup as e:
            log.append(("left", len(e.exceptions), type(e.exceptions[0]).__name__))

        # A tuple of types in one clause.
        try:
            raise ExceptionGroup("g", [ValueError("a"), KeyError("b"), TypeError("c")])
        except* (ValueError, KeyError) as eg:
            log.append(("tuple", len(eg.exceptions)))
        except* TypeError as eg:
            log.append(("rest", len(eg.exceptions)))

        # `finally` runs on the way out, matched or not.
        try:
            try:
                raise ExceptionGroup("g", [OSError("o")])
            except* ValueError as eg:
                log.append(("no", 1))
            finally:
                log.append(("finally", 1))
        except ExceptionGroup:
            log.append(("escaped", 1))

        # No exception at all: the clauses are skipped and `else` runs.
        try:
            log.append(("body", 1))
        except* ValueError as eg:
            log.append(("never", 1))
        else:
            log.append(("else", 1))
        finally:
            log.append(("fin2", 1))

        for item in log:
            print(item)
    """,
    "with_statement": """
        class Tracer:
            def __init__(self, name, swallow):
                self.name = name
                self.swallow = swallow
            def __enter__(self):
                print('enter', self.name)
                return self.name
            def __exit__(self, et, ev, tb):
                print('exit', self.name, et.__name__ if et else None)
                return self.swallow

        with Tracer('plain', False) as who:
            print('body', who)
        with Tracer('outer', True):
            with Tracer('inner', False):
                raise ValueError('boom')
        print('after')
        def early():
            with Tracer('returning', False):
                return 'value'
        print(early())
        with Tracer('a', False), Tracer('b', False):
            print('both')
    """,
    "parameter_kinds": """
        # `/` and `*` in a signature: which arguments a position can reach,
        # and which only a name can.
        def pos(a, b, /, c, d=4):
            return (a, b, c, d)
        p = pos
        print(p(1, 2, 3), p(1, 2, c=3), p(1, 2, 3, 4))
        try:
            p(1, b=2, c=3)
        except TypeError as e:
            print('TypeError', e)
        def kwo(a, *, b, c=3):
            return (a, b, c)
        k = kwo
        print(k(1, b=2), k(1, b=2, c=9))
        try:
            k(1, 2)
        except TypeError as e:
            print('TypeError', e)
        def both(a, /, b, *, c):
            return (a, b, c)
        bo = both
        print(bo(1, 2, c=3), bo(1, b=2, c=3))
        # A positional-only name is COLLECTED by `**kw` rather than matching.
        def collect(a, /, **kw):
            return (a, sorted(kw.items()))
        co = collect
        print(co(1, a=9, z=1))
        class Box:
            def __call__(self, /, *args, **kwargs):
                return (args, sorted(kwargs.items()))
        print(Box()(1, 2, x=3))
    """,
    "definite_assignment_joins": """
        # A handler that cannot fall through does not dilute the join: the
        # only path reaching the use DID assign.
        def wrap(n):
            try:
                result = 10 // n
            except ZeroDivisionError as exc:
                raise ValueError('bad') from exc
            return result
        print(wrap(2))
        def scan(items):
            out = []
            for item in items:
                try:
                    value = int(item)
                except ValueError:
                    continue
                out.append(value)
            return out
        print(scan(['1', 'x', '3']))
        def both(n):
            try:
                v = n * 2
            except ValueError:
                v = 0
            return v
        print(both(3))
        def fin():
            try:
                a = 1
            finally:
                b = 2
            return a + b
        print(fin())
        def orelse(n):
            try:
                x = n
            except ValueError:
                return 'no'
            else:
                y = x + 1
            return y
        print(orelse(1))
    """,
    "isinstance_forms": """
        class A:
            pass
        class B(A):
            pass
        class C:
            pass
        b = B()
        print(isinstance(b, (A, C)), isinstance(b, (C,)), isinstance(b, ()))
        print(isinstance(1, (str, int)), isinstance(1.0, (str, int)))
        print(isinstance(True, (int,)), isinstance('x', (bytes, str)))
        print(isinstance([1], (list, tuple)))
        print(isinstance(ValueError('x'), (KeyError, ValueError)))
        # A TUPLE HELD IN A VARIABLE is the same question as a literal one.
        kinds = (A, C)
        alias = A
        print(isinstance(b, kinds), isinstance(1, kinds), isinstance(b, alias))
        def kind(node):
            if isinstance(node, (A, B)):
                return 'ab'
            return 'other'
        print(kind(b), kind(C()))
    """,
    "generator_return_values": """
        # `return v` in a generator becomes StopIteration.value -- and that is
        # what `yield from` reads as the delegated generator's answer.
        def gen():
            x = yield 1
            y = yield x
            return (x, y)
        g = gen()
        print(next(g), g.send('a'))
        try:
            g.send('b')
        except StopIteration as e:
            print(e.value)
        def bare():
            yield 1
        b = bare()
        next(b)
        try:
            next(b)
        except StopIteration as e:
            print(repr(e), e.value)
        def inner():
            yield 'i'
            return 'from-inner'
        def outer():
            got = yield from inner()
            yield got
        print(list(outer()))
    """,
    "type_constructors_and_codepoints": """
        # Constructors reached through the TYPE: no receiver to be the first
        # argument, which is what separates these from `str.lower`.
        print(dict.fromkeys(['a', 'b'], 0))
        print((255).to_bytes(2, 'big'), int.from_bytes(bytes([1, 0]), 'big'))
        print(bytes.fromhex('01ff'))
        d = {'a': 1}
        d.update(b=2)
        d.update({'c': 3}, d=4)
        print(sorted(d.items()))
        # A str is UTF-8 underneath, so `chr` builds one to four bytes and
        # `ord` decodes them -- both counting CHARACTERS, not bytes.
        print(len(chr(233)), len(chr(0x4e2d)), ord(chr(233)), ord(chr(0x4e2d)))
        print(chr(233) == chr(233), chr(65), len('a' + chr(233)))
    """,
    "lazy_builtin_cursors": """
        # `map`, `filter`, `enumerate` and `zip` are CURSORS, not lists: the
        # function runs when the result is walked, and each is consumed once.
        log = []
        def keep(v):
            log.append(v)
            return v % 2 == 0
        f = filter(keep, [1, 2, 3, 4])
        print(log)
        print(list(f), log, list(f))
        m = map(str, [1, 2])
        print(list(m), list(m))
        e = enumerate('ab')
        print(next(e), list(e), list(e))
        print(type(enumerate([])).__name__, type(map(str, [])).__name__)
        print(type(filter(None, [])).__name__, type(zip()).__name__)
        print(list(filter(None, [0, 1, '', 'a', [], [1]])))
        print(list(zip([1, 2], 'ab')), list(zip()), list(zip([1, 2, 3], 'ab')))
        print(list(enumerate('ab', 5)))
        # A cursor over an INFINITE generator, stepped by hand.
        def naturals():
            n = 0
            while True:
                yield n
                n += 1
        doubled = map(lambda v: v * 2, naturals())
        print(next(doubled), next(doubled), next(doubled))
        print(sum(map(int, ['1', '2', '3'])), sorted(filter(keep, [3, 2, 1])))
    """,
    "lazy_iteration": """
        # ADVANCE UNTIL DONE, not walk by index. A generator has no length
        # until it has been run, and a body that appends to the list it is
        # walking sees the new elements -- neither works under an index walk.
        def naturals():
            n = 0
            while True:
                yield n
                n += 1
        for v in naturals():
            if v > 3:
                break
            print('lazy', v)
        xs = [1, 2, 3]
        seen = []
        for v in xs:
            seen.append(v)
            if len(seen) == 1:
                xs.append(99)
        print(seen, xs)
        shrinking = [1, 2, 3, 4]
        walked = []
        for v in shrinking:
            walked.append(v)
            shrinking.pop()
        print(walked, shrinking)
        for a, b in [(1, 'a'), (2, 'b')]:
            print(a, b)
        for k in {'x': 1, 'y': 2}:
            print(k)
        for c in 'ab':
            print(c)
        for v in {5}:
            print(v)
        for x in [1, 2, 3]:
            if x == 2:
                continue
            print('c', x)
        else:
            print('else')
        class Manual:
            def __init__(self):
                self.i = 0
            def __iter__(self):
                return self
            def __next__(self):
                if self.i >= 2:
                    raise StopIteration
                self.i += 1
                return self.i
        for v in Manual():
            print('manual', v)
        class Old:
            def __getitem__(self, i):
                if i >= 2:
                    raise IndexError
                return i * 5
        for v in Old():
            print('old', v)
    """,
    "generators": """
        def counter(n):
            i = 0
            while i < n:
                yield i
                i += 1
        print(list(counter(4)))
        for v in counter(3):
            print('v', v)
        # `send` -- the yield EXPRESSION's value is what was sent in.
        def echo():
            got = yield 'first'
            while got != 'stop':
                got = yield ('got', got)
            yield 'done'
        e = echo()
        print(next(e), e.send('a'), e.send('b'), e.send('stop'))
        # NONE OF THE BODY RUNS until the first `next`.
        def lazy():
            print('side')
            yield 1
        g = lazy()
        print('made')
        print(next(g))
        def withret():
            yield 1
            return
            yield 2
        print(list(withret()))
        # A `for` INSIDE a generator: its index lives in the frame, because a
        # register does not survive the return a `yield` compiles to.
        def nested():
            for x in [1, 2]:
                for y in 'ab':
                    yield (x, y)
        print(list(nested()))
        h = counter(9)
        print(next(h))
        h.close()
        try:
            next(h)
        except StopIteration:
            print('closed')
        print(sum(counter(5)), sorted(counter(3), reverse=True))
        try:
            next(iter(counter(0)))
        except StopIteration:
            print('empty')
        # `yield from` -- delegation, including recursively.
        def inner():
            yield 1
            yield 2
        def outer():
            yield 0
            yield from inner()
            yield 3
        print(list(outer()))
        def flat(xs):
            for x in xs:
                if isinstance(x, list):
                    yield from flat(x)
                else:
                    yield x
        print(list(flat([1, [2, [3, 4]], 5])))
        # `throw` raises AT the suspension point, so a `try` in the body
        # catches it; `close` sends GeneratorExit, so a `finally` runs.
        def plain():
            yield 1
            yield 2
        p = plain()
        print(next(p))
        try:
            p.throw(ValueError('boom'))
        except ValueError as e:
            print('ValueError', e)
        def guarded():
            try:
                yield 1
                yield 2
            except ValueError:
                yield 'caught'
        q = guarded()
        print(next(q), q.throw(ValueError('x')))
        def cleaning():
            try:
                yield 1
                yield 2
            finally:
                print('cleanup')
        c = cleaning()
        print(next(c))
        c.close()
        class Tracked:
            def __enter__(self):
                print('enter')
                return self
            def __exit__(self, *a):
                print('exit')
                return False
        def held():
            with Tracked():
                yield 1
                yield 2
        hh = held()
        print(next(hh), next(hh))
        try:
            next(hh)
        except StopIteration:
            print('stop')
    """,
    "comprehension_scope_and_print": """
        # A COMPREHENSION HAS ITS OWN SCOPE: the outer name is untouched, and
        # one only the comprehension binds is unbound afterwards.
        i = 'outer'
        squares = [i * 2 for i in range(3)]
        print(squares, i)
        gen = list(j for j in range(2))
        print(gen)
        try:
            print(j)
        except NameError:
            print('NameError')
        # A WALRUS writes the ENCLOSING scope, which is the difference.
        total = 0
        sums = [total := total + n for n in (1, 2, 3)]
        print(sums, total)
        def inner():
            k = 'local'
            got = [k for k in range(2)]
            return got, k
        print(inner())
        print('a', 'b', sep='-', end='!')
        print()
        print('x', 'y', sep='')
        print(1, 2, 3, sep=', ')
        for at, ch in enumerate('ab', start=10):
            print(at, ch)
        for at, ch in enumerate('ab', 5):
            print(at, ch)
    """,
    "importing_math": """
        import math
        import math as m
        from math import pi, sqrt
        from math import pi as PI
        print(round(math.pi, 5), round(m.pi, 5), round(pi, 5), round(PI, 5))
        # A module is built ONCE per program, so the two names are one object.
        print(math is m)
        print(math.floor(-2.5), math.ceil(-2.5), math.trunc(-2.5))
        print(math.trunc(2.5), math.gcd(12, 18), math.lcm(4, 6))
        print(math.isqrt(17), math.factorial(5), round(math.log(math.e), 10))
        print(math.inf > 0, math.isnan(math.nan), math.isfinite(1.0))
        print(math.isclose(1.0, 1.0 + 1e-12), math.isclose(1.0, 1.1, rel_tol=0.2))
        print(math.copysign(1.0, -0.0), sqrt(9), math.pow(2, 3))
        print(sorted([4.0, 1.0, 9.0], key=math.sqrt))
        print(round(math.tau, 5) == round(2 * math.pi, 5))
    """,
    "unpacking_arity_and_except_target": """
        # THE ARITY IS CHECKED BEFORE ANYTHING IS BOUND. A short sequence read
        # past the end and reported an IndexError from a subscript the program
        # never wrote; a long one bound the leading names and silently dropped
        # the rest, which is the worse of the two.
        try:
            a, b = [1]
        except ValueError as e:
            print(str(e))
        try:
            a, b = [1, 2, 3]
        except ValueError as e:
            print(str(e))
        try:
            p, *q = []
        except ValueError as e:
            print(str(e))
        a, b = [1, 2]
        (c, d), f = (1, 2), 3
        x, *rest = [1, 2, 3]
        print(a, b, c, d, f, x, rest)

        # `except ... as e` DELETES `e` when the clause ends, and a program
        # reads that back: `e` afterwards is a NameError, not the caught value.
        try:
            raise ValueError("x")
        except ValueError as e:
            print(type(e).__name__)
        try:
            print(e)
        except NameError:
            print("NameError")

        # A handler that RETURNS or RAISES has already ended its block, so
        # there is nothing to delete and nowhere to emit the deletion.
        def returns_from_handler():
            try:
                raise ValueError("x")
            except ValueError as e:
                return str(e)

        def raises_from_handler():
            try:
                raise KeyError("k")
            except KeyError as e:
                raise RuntimeError("wrapped")

        print(returns_from_handler())
        try:
            raises_from_handler()
        except RuntimeError as e:
            print(str(e))
    """,
    "name_resolution_and_nan": """
        # A CONTAINER ASKS "is this the same object?" before it asks the
        # object, which is why `[nan] == [nan]` is True while `nan == nan` is
        # False. Membership and dict comparison rest on the same rule.
        nan = float("nan")
        print(nan == nan, nan != nan)
        print([nan] == [nan], nan in [nan], {"k": nan} == {"k": nan})
        print(sorted([1.0, nan, 2.0]) == [1.0, nan, 2.0])

        # LOCAL, THEN GLOBAL, THEN BUILTINS. Shadowing a builtin at module
        # level made every use of that name a global read, so `del` left it
        # raising NameError for a name that is always defined.
        print(len([1, 2]))
        len = 5
        print(len)
        del len
        print(len([1, 2]))

        # `print(*xs, sep=)` -- the starred form builds its arguments at run
        # time and went through a path with nowhere to put the separator,
        # which it dropped rather than refusing.
        print(*[1, 2, 3], sep=",")
        print(*[1, 2], sep="-", end="!")
        print("")
        print("a", "b", sep="-")
        print(*[], sep=",")
    """,
    "metaclasses": """
        # `class Meta(type)` gets a REAL BASE -- the runtime's `type` class,
        # whose dict holds the two natives `super()` reaches -- so a
        # metaclass's `__new__` builds a class through the same path any
        # other `__new__` takes.
        log = []

        class Meta(type):
            def __new__(mcls, name, bases, ns, **kw):
                log.append(("new", name, sorted(kw.items())))
                return super().__new__(mcls, name, bases, ns)
            def __init__(cls, name, bases, ns, **kw):
                log.append(("init", name))
                super().__init__(name, bases, ns)

        class C(metaclass=Meta, flavour="x"):
            pass

        print(log)
        # `type(C)` IS THE METACLASS. An ordinary class reads as `type`.
        print(type(C).__name__)

        class Plain:
            pass

        print(type(Plain).__name__)

        # PEP 3115: the body runs into whatever `__prepare__` supplies, so a
        # seeded mapping is visible on the class and the body's own bindings
        # are readable as a mapping before the class exists.
        class OrderedMeta(type):
            @classmethod
            def __prepare__(mcls, name, bases, **kw):
                return {"seeded": 7}
            def __new__(mcls, name, bases, ns):
                cls = super().__new__(mcls, name, bases, dict(ns))
                cls.declared = [k for k in ns if not k.startswith("_")]
                return cls

        class D(metaclass=OrderedMeta):
            b = 1
            a = 2

        print(D.declared, D.b, D.a, D.seeded, type(D).__name__)

        # THE METACLASS DECIDES both checks, and both answer a BOOL whatever
        # the hook returned.
        class Quacky(type):
            def __instancecheck__(cls, obj):
                return "quacks"
            def __subclasscheck__(cls, sub):
                return True

        class Duck(metaclass=Quacky):
            pass

        print(isinstance(42, Duck), issubclass(int, Duck))
        print(isinstance(42, int), isinstance("a", int))

        # `type(name, bases, ns)` is the `class` statement written out, and
        # builds the same object a metaclass's `super().__new__` does.
        class B:
            def m(self):
                return "b"

        X = type("X", (), {"a": 1})
        Y = type("Y", (B,), {})
        print(X.a, X.__name__, type(X).__name__)
        print(Y().m(), isinstance(Y(), B), type(3).__name__)

        # `object` is the ROOT of every chain even though no class links to
        # it, so a class with no written base still has one base. The empty
        # tuple said the chain stopped at the class.
        print(B.__base__.__name__, len(B.__bases__), B.__bases__[0].__name__)

        class Deriv(B):
            pass

        print(Deriv.__base__ is B, len(Deriv.__bases__))
    """,
    "new_and_object_defaults": """
        # `__new__` was IGNORED -- the instance was allocated and the method
        # never ran -- and `super().__init__()` in a class with no explicit
        # base was an AttributeError, because `object`'s defaults existed as
        # behaviours and no VALUE named them.
        class P:
            def __new__(cls, *a):
                print("new", a)
                return super().__new__(cls)
            def __init__(self, n):
                print("init", n)
                self.n = n

        print(P(5).n)

        # A class attribute BOUND TO NONE is a real attribute. The lazily
        # filled slot every singleton starts from was invisible.
        class Singleton:
            _one = None
            def __new__(cls):
                if cls._one is None:
                    cls._one = super().__new__(cls)
                return cls._one

        print(Singleton() is Singleton())

        # `__init__` RUNS ONLY IF `__new__` RETURNED ONE OF THESE.
        class Weird:
            def __new__(cls):
                return 42
            def __init__(self):
                print("never")

        print(Weird())

        class Base:
            def __init__(self):
                super().__init__()
                self.tag = "base"

        class Sub(Base):
            def __init__(self):
                super().__init__()
                self.tag = self.tag + "+sub"

        print(Sub().tag)

        class E:
            def __eq__(self, o):
                return super().__eq__(o)
            def __hash__(self):
                return super().__hash__()

        e = E()
        print(e == e, e == E(), type(hash(e)).__name__)
        # NOT the default repr's text: CPython qualifies it with the module
        # (`<__main__.R object ...>`) and no qualified name is recorded here.
        # What is pinned is that `super().__repr__()` reaches the DEFAULT
        # rather than being rewritten to the repr of the super object.
        class R:
            def __repr__(self):
                return "R<" + super().__repr__()[0] + ">"

        print(str(R()), repr(R()))
    """,
    "bound_methods_compare_by_receiver": """
        # A BOUND METHOD IS A FRESH OBJECT PER ACCESS, so `c.m is c.m` is
        # False -- and two of them are EQUAL when they wrap the same function
        # and the same receiver. Only bound ones: two closures over one `def`
        # are distinct objects, and CPython calls those unequal.
        class C:
            def m(self):
                return 1

        c, d = C(), C()
        print(c.m == c.m, c.m is c.m, C.m is C.m)
        print(c.m == d.m, c.m == C.m, [c.m] == [c.m])

        def outer():
            def inner():
                pass
            return inner

        print(outer() == outer(), outer() is outer())
        held = c.m
        print(held == c.m, held() == 1, len({c.m, c.m}))
    """,
    "equality_is_never_a_pointer_accident": """
        # Every kind that is not a number used to fall through to the NUMERIC
        # comparison, which reads a union member that for most kinds is a
        # pointer. Two slices sharing a `start`, two views onto one dict and
        # two memoryviews over one buffer each compared EQUAL because the
        # pointers matched. What this pins is that each kind either compares
        # by content deliberately or by identity, and never by accident.
        d = {"a": 1}
        print(d.keys() == {"a"}, d.items() == {("a", 1)}, d.keys() == {"b"})
        # `values()` is the one view that is not set-like: it defines no
        # equality, so it is equal only to itself.
        v = d.values()
        print(d.keys() == d.values(), v == v, d.keys() == d.keys())
        print(memoryview(b"ab") == b"ab", memoryview(b"ab") == b"ac")
        ba = bytearray(b"abcd")
        print(memoryview(ba)[0:2] == memoryview(ba)[0:3])
        print(slice(1, 2) == slice(1, 3), slice(1, 2) == slice(1, 2))
        print({1} == frozenset({1}), 1 == 1.0, None == None, [1] == [1])
    """,
    "keyword_only_parameters": """
        # A KEYWORD-ONLY PARAMETER TAKES NO ARGUMENT POSITION. Three places
        # treated it as though it did: the arity check refused
        # `b(1, 2, c=9)`, the call lowering let the `2` land in `c`, and the
        # runtime binder sent `c=3` into `**kw` whenever there were surplus
        # positionals bound for `*args` -- so `c` kept its default and the
        # tuple came back empty.
        def f(a, b=2, *args, c=3, **kw):
            return (a, b, args, c, kw)

        print(f(1), f(1, b=9))
        print(f(1, 2, 3, c=4, d=5), f(1, 2, 3, 4), f(1, 2, nope=1, c=2))

        def g(a, /, b, *, c):
            return (a, b, c)

        print(g(1, 2, c=3), g(1, b=2, c=3))
        # A POSITIONAL-ONLY PARAMETER CANNOT BE NAMED. The runtime binder
        # refused this for a call through a value; a DIRECT call resolves
        # names at the call site and filled the slot anyway, so `f(a=1)`
        # against `def f(a, /)` quietly worked.
        try:
            g(a=1, b=2, c=3)
        except TypeError:
            print("positional-only")
        # FEWER positionals than there are positions, with a keyword-only
        # tail behind them -- the slot list has to reach the tail so its
        # defaults land, and sizing it from the arguments given did not.
        def few(a, b=2, *, c=3, d=4):
            return (a, b, c, d)

        print(few(1), few(1, 5), few(1, d=9), few(1, 5, c=8, d=9))

        def required_kwonly(x, *args, c):
            return (x, args, c)

        print(required_kwonly(1, 2, c=3), required_kwonly(1, c=2))

        def h(*args, **kw):
            return (args, kw)

        print(h(), h(1, 2, x=3))

        class K:
            def __init__(self, a, *rest, b=1, **kw):
                self.v = (a, rest, b, kw)

        print(K(1, 2, b=3, z=4).v, K(1).v)
        # NOT a try/except for the missing `c`: a provably missing required
        # argument is REFUSED at compile time here, as every provable arity
        # mistake is, so there is no run to catch it in.
    """,
    "exception_repr": """
        # `str(KeyError('k'))` is the REPR of the key -- KeyError alone does
        # that, so a missing key whose text is empty is still visible. The
        # trap is a KeyError REBUILT from a failed lookup: its argument is
        # already the repr, and repr'ing it again said `KeyError("'k'")`.
        print(repr(KeyError("k")), str(KeyError("k")))
        print(repr(ValueError("v")), str(ValueError("v")), repr(KeyError()))
        try:
            {}["k"]
        except KeyError as e:
            print(repr(e), str(e))
        try:
            [][0]
        except IndexError as e:
            print(repr(e), str(e))
        print([KeyError("k"), ValueError("v")])
    """,
    "percent_formatting": """
        # Translated into the format mini-language rather than reimplemented:
        # `%05.2f` and `{:05.2f}` mean the same thing. What is pinned here is
        # the printf SPELLING -- which flags mean what, and the two
        # conversions the mini-language has no type character for.
        class P:
            def __init__(self, n):
                self.n = n
            def __repr__(self):
                return "P!" + str(self.n)
            def __str__(self):
                return "p" + str(self.n)

        print("%d %s" % (1, "a"))
        print("%05.2f|%x|%X|%o|%e" % (3.14159, 255, 255, 8, 1234.5))
        # A USER OBJECT through `%s` and `%r` -- the reason Python's own `%`
        # is not what does this: it would print an address.
        print("%r %s" % (P(1), P(2)))
        print("%-6d|%+d|% d|%#x|%08.3f" % (42, 5, 5, 255, 3.14159))
        print("%s" % [1, 2], "%s" % (1,), "%s" % "x", "%%")
        print("%c%c" % (65, "B"), "%.3s" % "abcdef", "%5s|" % "ab")
        # `b"%s"` inserts THE BYTES, not their repr, and the answer is bytes.
        print(b"%d %s %x" % (3, b"ab", 255), type(b"%d" % 3).__name__)
        try:
            "%d %d" % (1,)
        except TypeError:
            print("too few")
        try:
            "%d" % (1, 2)
        except TypeError:
            print("too many")
        # A MAPPING ON THE RIGHT supplies NAMED fields only, and nothing is
        # consumed positionally -- so an unused entry is ordinary rather than
        # "not all arguments converted".
        print("ab" % {"ab": 1}, "%(x)s%(y)s" % {"x": "a", "y": "b"})
        print("%(n)05.1f|" % {"n": 3.14159})
        try:
            "%(z)s" % {"a": 1}
        except KeyError:
            print("KeyError")
    """,
    "definition_time_defaults_and_decorators": """
        # A module-level `def` STATEMENT runs where it is written. Its
        # defaults and its decorators were evaluated at program start instead,
        # before the module body had bound anything -- so a default naming a
        # global was a NameError for a program CPython runs, and a decorator
        # naming one was too.
        log = []

        def outer(fn):
            log.append("outer")
            return fn

        def inner(fn):
            log.append("inner")
            return fn

        @outer
        @inner
        def decorated():
            pass

        print(log)

        n = 1

        def f(v=n):
            return v

        n = 99
        print(f(), f(5), n)
    """,
    "bundled_warnings": """
        # The WARNING CATEGORIES are exceptions like any other -- `Warning`
        # inherits `Exception` -- and were missing from the hierarchy
        # entirely, so `issubclass(DeprecationWarning, Warning)` could not
        # even be asked.
        print(issubclass(DeprecationWarning, Warning), UserWarning.__name__)
        try:
            raise DeprecationWarning("d")
        except Warning as e:
            print(type(e).__name__, str(e))

        # A COMPILED PROGRAM HAS NO WARNING FILTERS TO INHERIT, so the module
        # is the whole mechanism rather than a view onto one: an action, a
        # place to record, and a context manager that saves and restores both.
        import warnings
        from warnings import deprecated

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("dep", DeprecationWarning)
            warnings.warn("usr", UserWarning)
        print(sorted(w.category.__name__ for w in caught))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("ignore")
            warnings.warn("dep", DeprecationWarning)
        print(len(caught))

        # The action is restored on the way OUT, which is what makes this
        # usable around code expected to fail.
        with warnings.catch_warnings(record=True) as after:
            warnings.warn("again", UserWarning)
        print(len(after))

        @deprecated("use g instead")
        def f():
            return 1

        print(f(), f.__deprecated__)
    """,
    "bundled_itertools_and_contextlib": """
        # Two more modules written in PYTHON and spliced in. `itertools` is
        # all generators, so the laziness is the language's rather than a data
        # structure imitating it; `contextmanager` is a generator wearing the
        # `with` protocol, and written here that sentence IS the
        # implementation.
        import itertools
        import contextlib

        print(list(itertools.chain([1], [2, 3])))
        print(list(itertools.islice(itertools.count(5), 3)))
        print(list(itertools.islice(range(10), 2, 5)))
        print(list(itertools.repeat("x", 3)))
        print([list(g) for _, g in itertools.groupby([1, 1, 2])])
        print(list(itertools.product([1, 2], "ab")))
        # THE ORDER IS PART OF THE ANSWER: the right tuples in the wrong order
        # is still wrong to a program that prints them.
        print(list(itertools.combinations([1, 2, 3], 2)))
        print(list(itertools.combinations([1], 2)), list(itertools.chain()))

        @contextlib.contextmanager
        def tracked(log):
            log.append("enter")
            yield log
            log.append("exit")

        log = []
        with tracked(log) as held:
            held.append("body")
        print(log)

        # A `try` around the `yield` is how the block's exception reaches the
        # generator, and swallowing it there SUPPRESSES it.
        @contextlib.contextmanager
        def swallow():
            try:
                yield
            except ValueError:
                pass

        with swallow():
            raise ValueError("x")
        print("suppressed")
    """,
    "bundled_functools": """
        # `functools` is written in PYTHON and spliced into the program that
        # imports it -- see `frontends/python/bundled.py`. Compiling the
        # standard library with the compiler under test is the point: it
        # cannot drift from the semantics it copies, because it IS them.
        #
        # It also found two compiler bugs on its first day, both pinned below.
        import functools
        from functools import reduce, wraps

        print(functools.reduce(lambda a, b: a + b, [1, 2, 3]))
        # NOT `reduce(max, ...)`: a builtin passed as a VALUE becomes a
        # one-argument thunk, so a two-argument builtin cannot be one.
        # A lambda can, which is what this uses.
        print(reduce(lambda a, b: a + b, [1, 2, 3], 10))

        def add(a, b, c=0):
            return a + b + c

        # A NESTED FUNCTION CAPTURING `*args`: the vararg arrives in a
        # register like any parameter, but was not marked as one, so boxing it
        # for a closure built the cell from None and threw the tuple away.
        # `**kw` worked, which is what made it look like something else.
        def outer(*a, **k):
            def inner():
                return a, k
            return inner()

        print(outer(1, 2, x=3), outer())

        # `f(*xs, **kw)` DROPPED EVERY KEYWORD. The keyword half has to travel
        # separately, as it does for every other call shape -- appended to the
        # argument list it would arrive as one more positional.
        args = (1, 2)
        kw = {"c": 3}
        print(add(*args, **kw), add(*args), add(1, 2, **kw))

        p = functools.partial(add, 1)
        q = functools.partial(add, c=10)
        print(p(2), p(2, c=3), q(1, 2), p.func is add, p.args)

        @functools.total_ordering
        class V:
            def __init__(self, n):
                self.n = n
            def __eq__(self, o):
                return self.n == o.n
            def __lt__(self, o):
                return self.n < o.n

        print(V(1) < V(2), V(2) >= V(1), V(1) <= V(1), V(2) > V(1))

        @wraps(add)
        def wrapper(*a, **k):
            return add(*a, **k)

        print(wrapper.__name__, wrapper(1, 2), wrapper.__wrapped__ is add)
    """,
    "await_in_expression_position": """
        # A REGISTER DOES NOT SURVIVE A SUSPENSION -- that is why a
        # generator's locals live in frame slots -- and neither does an
        # INTERMEDIATE. `await a() + await b()` computed the left operand into
        # a register, the right operand suspended, and the resume path read a
        # register no path had written: invalid IR, reported against a block
        # the program never wrote. Every display holding an accumulator across
        # its elements had the same shape.
        import asyncio

        async def v(x):
            return x

        async def main():
            total = await v(1) + await v(2)
            xs = [await v(n) for n in (1, 2)]
            d = {await v(1): await v(2)}
            s = {await v(3)}
            trio = (await v(1), await v(2), await v(3))
            nested = [await v(n) + await v(n) for n in (1, 2)]
            return total, xs, d, s, trio, nested

        print(asyncio.run(main()))

        # A frame slot holds the HANDLE, not the object: a tuple under
        # construction is REPLACED in its cell as it grows, and a slot holding
        # the object froze at the empty tuple it went in as.
        async def builds():
            return (await v(1), await v(2), await v(3))

        print(asyncio.run(builds()))
    """,
    "chained_assignment_and_bare_return": """
        # `a = b = value` -- ONE evaluation, bound to each target left to
        # right, which is what makes `a = b = []` two names for the SAME list.
        a = b = [1]
        a.append(2)
        print(a, b, a is b)
        p = q = r = 5
        print(p, q, r)
        d1 = {}
        d1["k"] = n = 7
        print(d1, n)
        x, y = 1, 2
        x, y = y, x
        print(x, y)

        # A BARE `return` in a dynamic function yields None AS AN OBJECT.
        # Typing it as the static None made `def f(): return` -- and every
        # early exit written that way -- a narrowing error for a program
        # CPython runs.
        def bare():
            return

        def falls():
            pass

        def early():
            for _ in range(1):
                return "early"

        print(bare(), falls(), early())
    """,
    "exception_group_message": """
        # `g.message` is the text a group was built with -- its FIRST argument,
        # separate from the exceptions it carries, and present only on a group.
        inner = ExceptionGroup("inner", [TypeError("t"), ValueError("v")])
        outer = ExceptionGroup("outer", [inner])
        print(sorted(type(e).__name__ for e in inner.exceptions))
        print(outer.message, len(inner.exceptions))
        try:
            ValueError("v").message
        except AttributeError:
            print("AttributeError")
    """,
    "function_introspection": """
        # `__defaults__` is the POSITIONAL defaults as a tuple and
        # `__kwdefaults__` the keyword-only ones as a dict -- each NONE rather
        # than empty when there are none, which is how a program tells "no
        # defaults" from "a default that is falsey". They are stored as one
        # trailing run, keyword-only last, so the split is the number of
        # keyword-only parameters that have one.
        def f(a, b=1, *args, c=2, **kw):
            '''Doc.'''
            return a

        print(f.__defaults__, f.__kwdefaults__)
        print(f.__name__, f.__doc__, f.__qualname__)

        def g(x):
            return x

        def h(a, b=1, d=2):
            return a

        print(g.__defaults__, g.__kwdefaults__)
        print(h.__defaults__, h.__kwdefaults__)

        # PEP 3155: the frontend's own key for a function is already CPython's
        # spelling of the qualified name.
        class C:
            def m(self):
                pass

        print(C.__qualname__, C.m.__qualname__, C.m.__name__)
        # PEP 649 for a CLASS: a body's ANNOTATED NAMES build
        # `C.__annotations__` the same lazy way a function's parameters do,
        # and a name with NO value still appears -- which is the whole point
        # of writing `a: int` on its own.
        class Ann:
            a: int = 1
            b: str

        print(sorted(Ann.__annotations__), Ann.a)

        class NoAnn:
            pass

        print(NoAnn.__annotations__)
        # A function is an object a program may hang anything on.
        f.custom = "attached"
        print(f.custom)
        # `f.__code__` -- ENOUGH OF ONE to answer what a program asks a
        # function about its own signature. There is no bytecode here to
        # describe; these are what introspection actually reads.
        print(f.__code__.co_argcount, f.__code__.co_varnames[:2])
        print(f.__code__.co_kwonlyargcount, f.__code__.co_name)
    """,
    "membership_consumes_a_generator": """
        # `x in gen` CONSUMES the generator up to the match and leaves the
        # rest -- a generator is consumed once. Draining it to a list to
        # answer the question reported the generator as not iterable at all.
        print(sum(v for v in range(3)))
        squares = (v * v for v in range(3))
        print(2 in squares)
        print(list(squares))
        g2 = (v for v in range(4))
        print(1 in g2, list(g2))
        # A container is unaffected -- it has no position to consume.
        xs = [1, 2, 3]
        print(2 in xs, xs, 9 in xs)
    """,
    "slots_conflict_and_descriptor": """
        # `__slots__ = ("v",)` and `v = 1` IN THE SAME BODY is a ValueError at
        # class creation: the slot and the attribute would share a name and
        # the attribute would win silently. RAISED, not refused at compile
        # time -- a program may catch it, and this one does.
        try:
            class Bad:
                __slots__ = ("v",)
                v = 1
        except ValueError:
            print("ValueError")

        class Ok:
            __slots__ = ("v",)

        o = Ok()
        o.v = 1
        print(o.v, Ok.__slots__)
        # A SLOT READ THROUGH THE CLASS is a descriptor, not a missing
        # attribute: `__slots__` declares storage and the class dict holds
        # nothing for it.
        print(type(Ok.v).__name__)
        try:
            o.other = 2
        except AttributeError:
            print("AttributeError")
    """,
    "bytes_methods_mirror_str": """
        # bytes shares the str LAYOUT -- a pointer and a length -- so every
        # one of these is the same operation. What was missing is that the
        # receiver check rejected the kind and that the RESULT has to come
        # back tagged bytes, which one wrapper at the call site does for all
        # fifty-odd methods at once.
        b = b"  Hello  "
        print(b.strip(), b.upper(), b.lower())
        print(b.split(), b.replace(b"l", b"L"), b.find(b"e"))
        print(b"a,b".split(b","), b"ab".startswith(b"a"), b"ab".endswith(b"b"))
        print(b"-".join([b"a", b"b"]), b"ab".index(b"b"), b"aa".count(b"a"))
        print(b"ab".ljust(4), b"ab".partition(b"b"), b"AB".title())
        # A METHOD THAT ANSWERS AN INT is left alone, which is what makes the
        # wrapper safe to apply everywhere.
        print(b"abc".find(b"z"), len(b"abc"), b"abc"[1])
        # str is untouched -- the wrapper returns its argument for any
        # receiver that is not bytes.
        s = "  Hi  "
        print(s.strip(), s.upper(), s.split(), "a,b".split(","))
        print("x".join(["1", "2"]), "abc".find("b"), "abc".partition("b"))
        print([1, 2, 1].count(1), [1, 2].index(2))
        # THE CONVERSIONS ARE NOT MIRRORED. `b.hex()` and `b.decode()` answer
        # a str FROM bytes -- that is what they are for -- so re-tagging their
        # result would undo the conversion the program asked for. Wrapping
        # every method without excluding these two made `b.hex()` bytes.
        raw = bytes([1, 255, 16])
        print(raw.hex(), raw.hex(":"), type(raw.hex()).__name__)
        print(b"ab".decode(), "ab".encode(), bytes.fromhex("01ff10"))
        # `bytes(3)` is THREE ZERO BYTES, not the digit three.
        print(bytes(3), list(raw), bytes([1, 2]))
        # MIXING BYTES AND str IS A TypeError -- PEP 3112's whole point. An
        # equality body ended up in `apy_add`'s mixed branch during this work
        # and returned a C int as a value, so this SEGFAULTED rather than
        # raising. Nothing else in the corpus added the two kinds.
        try:
            b"a" + "a"
        except TypeError:
            print("TypeError")
        print(b"a" == "a", b"ab" + b"c", "ab" + "c")
        print(b"ab"[0], "ab"[0], len(b"ab"), len("ab"))
    """,
    "complex_strings_and_signed_zero": """
        # `complex("1+2j")` is a PARSE, not an arithmetic conversion, and the
        # TypeError it used to raise already promised a string was acceptable.
        print(complex("1+2j"), complex("2j"), complex("3"), complex("-1-1j"))
        print(complex("(1+2j)"), complex(1, 2), complex(1.5))
        try:
            complex("bad")
        except ValueError:
            print("ValueError")

        # PEP 682: `z` turns a negative zero into a positive one, and it is
        # about the ROUNDED value -- so `-0.001` at one decimal place is `0.0`
        # too. `signbit`, not `< 0`: negative zero is not less than zero, and
        # it is the value the flag exists for.
        print(format(-0.0, "z.1f"), format(-0.0, ".1f"), format(0.0, "z.1f"))
        print(format(-0.001, "z.1f"), f"{-0.0:z.2f}", format(-1.5, "z.1f"))
    """,
    "class_body_order_and_slice_del": """
        # A class body runs TOP TO BOTTOM, attributes and methods together.
        # They were lowered in two passes, so an attribute could not see a
        # method defined above it and the class dict came out in pass order
        # rather than definition order -- which PEP 520 makes observable.
        class C:
            b = 1
            a = 2

            def m(self):
                return self.b

            z = 3

        print([k for k in vars(C) if not k.startswith("_")])

        class D:
            def f(self):
                return 1
            g = f
            x = 2

        print(D().g(), D.x, [k for k in vars(D) if not k.startswith("_")])

        # A CLASS BODY IS A BLOCK THAT RUNS, once, where it is written -- and
        # a statement that binds nothing still has its effect, in its own
        # place among the rest. Refusing a bare expression rejected the
        # program over a statement that binds nothing.
        log = []

        class E:
            log.append("body")
            v = 1
            log.append("after v")

            def m(self):
                return self.v

        print(log)
        E()
        E()
        print(log, E().m(), E.v)

        # `del xs[1:3]` REMOVES A SPAN. It fell through to the index path,
        # which asked for an integer, got the slice, and reported an
        # IndexError about a subscript the program never wrote.
        xs = [0, 1, 2, 3]
        del xs[1:]
        ys = [0, 1, 2, 3]
        del ys[1:3]
        zs = [0, 1, 2]
        del zs[:]
        ws = [0, 1, 2]
        del ws[1]
        print(xs, ys, zs, ws)
    """,
    "annotations_are_lazy": """
        # PEP 649: `__annotations__` is BUILT ON ACCESS, by a thunk the `def`
        # records. Evaluating them at the `def` would make
        # `def f(x: Undefined)` an error where Python accepts it -- only
        # READING them is, and that is the whole point of the PEP.
        def f(a: int, b: "str" = "x") -> bool:
            return True

        print(sorted(f.__annotations__), f.__annotations__["a"], f(1))

        def plain(x):
            return x

        print(plain.__annotations__, plain(2), hasattr(plain, "__annotate__"))

        def lazy(x: Undefined) -> AlsoUndefined:
            return x

        print(callable(lazy), lazy(1), hasattr(lazy, "__annotate__"))
        try:
            lazy.__annotations__
        except NameError:
            print("NameError")

        class C:
            def m(self, n: int) -> str:
                return "m"

        print(sorted(C.m.__annotations__), C().m(1))
    """,
    "generic_aliases_and_unions": """
        # PEP 604: `int | str` IS A TYPE, not an arithmetic operation, and its
        # repr uses the bars rather than `Union[...]`. The arms flatten, so
        # `int | str | float` is one three-armed union and `isinstance` over
        # it is a single walk.
        u = int | str
        print(u, isinstance(3, u), isinstance("a", u), isinstance(3.0, u))
        v = int | str | float
        print(v, isinstance(1.5, v), isinstance(None, u))

        # PEP 585: `list[int].__origin__` is `list` and `.__args__` is
        # `(int,)` -- the two a program reads off an annotation.
        t = list[int]
        print(t, t.__origin__ is list, t.__args__)
        print(dict[str, int], len(dict[str, int].__args__))

        # A generator's three methods are dispatched by NAME at the call site,
        # so nothing needed a VALUE for them -- until `hasattr(g, "close")`,
        # which every duck-typed consumer asks.
        def gen():
            yield 1

        g = gen()
        print(hasattr(g, "close"), hasattr(g, "throw"), hasattr(g, "send"))
        print(next(g), hasattr(g, "nosuch"))
    """,
    "generator_flow_rules": """
        # A `return` INSIDE A `finally` discards whatever was in flight. The
        # finally body ran, decided the function's answer, and the exception it
        # was cleaning up after went on propagating anyway.
        def wins():
            try:
                raise ValueError("lost")
            finally:
                return "finally wins"

        print(wins())

        def kept():
            try:
                raise ValueError("kept")
            finally:
                pass

        try:
            kept()
        except ValueError as e:
            print("ValueError", e)

        # PEP 479: a `StopIteration` that ESCAPES a generator body becomes a
        # RuntimeError with the original as its `__cause__`. Left alone it was
        # indistinguishable from the generator finishing normally, so a bug in
        # the body read as a clean end of iteration.
        def raises_stop():
            yield 1
            raise StopIteration("inner")

        g = raises_stop()
        print(next(g))
        try:
            next(g)
        except RuntimeError as e:
            print("RuntimeError", type(e.__cause__).__name__)
    """,
    "exception_classes_are_values": """
        # `raise` and `except` match on the NAME, so registering the name is
        # what makes an exception class work -- but a program also reads
        # `MyError.__mro__` and passes the class to `issubclass`, and binding
        # nothing made every such use a NameError for a class it just defined.
        class AppError(Exception):
            pass

        class SubError(AppError):
            pass

        try:
            raise SubError("boom")
        except AppError as e:
            print(type(e).__name__, str(e), isinstance(e, AppError))

        print(SubError.__mro__[1].__name__, issubclass(SubError, AppError))
        # An exception type's parent is in the NAME TABLE, not in a base
        # pointer, so the walk had to ask the table -- `Exception.__bases__`
        # answered `object` where CPython says `BaseException`.
        print(Exception.__bases__[0].__name__, issubclass(Exception, BaseException))
        print([c.__name__ for c in Exception.__mro__])

        class A:
            pass

        class B(A):
            pass

        print([c.__name__ for c in B.__mro__])
    """,
    "format_fields_and_module_dunders": """
        # `{x[0]}` and `{a.real}`: the NAME stops at the first `.` or `[`, and
        # what follows is a chain of accessors. Treating the whole field as one
        # keyword looked for an argument called `x[0]`.
        print("{x[0]}".format(x=[9]), "{a.real}".format(a=1.5))
        print("{p[1]}{p[0]}".format(p=("a", "b")), "{m[k]}".format(m={"k": 7}))
        print("{:{}}".format(42, ">6") + "|", "{0}{1}{0}".format("a", "b"))
        print("{{literal}}".format(), "{}{}".format("a", "b"))

        # `__file__` is the one module dunder whose value is not a constant of
        # every compilation, and `globals()` carries the dunders too --
        # `"__builtins__" in globals()` is a question programs ask.
        # NO MODULE DUNDERS HERE. This corpus runs CPython through `exec`
        # with a fresh globals dict, which has neither `__file__` nor
        # `__doc__` nor `__builtins__` -- so the reference would disagree with
        # itself rather than with us. `statements/module-dunder-attributes`
        # covers them, as a real script, which is the only way they mean
        # anything.
        print(__name__)
    """,
    "strings_are_measured_in_characters": """
        # A str is stored as UTF-8. `len` counted CHARACTERS while indexing,
        # slicing, iteration and `find` all counted BYTES -- identical for
        # ASCII and wrong for everything else, in BOTH paths equally, which is
        # why nothing caught it. `s[1]` was the first half of a character and
        # a `for` loop took its bound from one count and its elements from the
        # other, so it walked off the end.
        s = "héllö"
        print(len(s), [ord(c) for c in s])
        print(ord(s[0]), ord(s[1]), ord(s[-1]))
        print(len(s[1:4]), ord(s[1:4][0]), s[::-1] == "ölleéh")
        print(s.find("ll"), s.rfind("l"), s.index("l"), s.find("z"))
        print(s.find("l", 3), s.count("l"), len(s[::2]))
        # ASCII is unchanged, which is the whole reason this stayed hidden.
        a = "abc"
        print(a[0], a[-1], a[1:], a[::-1], len(a), list(a))
        print("abcabc".find("bc"), "abcabc".rfind("bc"), "abc".index("c"))
        # Beyond the BMP: four bytes, still one character.
        e = "😀x"
        print(len(e), ord(e[0]), e[1], e[0:1] == "😀")
    """,
    "builtin_protocol_is_reachable_by_name": """
        # `[].append` and `{}.keys` are lowered at the call site, which means
        # they exist as CALLS and never as attributes -- so `hasattr([1],
        # "__iter__")` answered False for the most iterable object in the
        # language, and every structural type test said no.
        print(hasattr([1], "__iter__"), hasattr([1], "__len__"))
        print(hasattr({}, "keys"), hasattr((1,), "index"))
        print([1, 2].__len__(), [1, 2].__contains__(2), [1, 2].__getitem__(0))
        print(sorted({"a": 1, "b": 2}.keys()))
        # SET TO None MEANS WITHDRAWN: `[].__hash__` is None rather than
        # missing, which is how a mutable container says it cannot be hashed.
        print([].__hash__, (1, 2).__hash__() == hash((1, 2)))
        # THE TYPE ANSWERS FOR ITS INSTANCES, and unbound -- `dict.keys` is
        # unbound in CPython too.
        print(hasattr(dict, "keys"), hasattr(list, "append"))

    """,
    "dict_views_print_as_views": """
        # A dict view rendered as the empty string: it had no repr of its own
        # and fell through to the default, so `print(d.keys())` printed
        # nothing at all.
        d = {"a": 1, "b": 2}
        print(d.keys())
        print(d.values())
        print(d.items())

    """,
    "typing_forms_print_as_written": """
        # THE UNION IS THE ONLY FORM THAT PRINTS WITH BARS. Testing "the
        # origin is an instance" made every parameterised form print as one,
        # so `Annotated[int, 'x']` came out as `int | 'x'`.
        from typing import Annotated, Literal, Optional, get_args

        print(Annotated, Literal)
        print(Annotated[int, "positive"])
        print(Literal["a", "b"])
        # `Optional[X]` IS `X | None` in 3.14, and `None` in a union is the
        # NoneType CLASS rather than the singleton.
        print(Optional[int], int | str)
        print(get_args(Optional[int]))
        print(get_args(int | None))

    """,
    "zero_argument_constructors": """
        # `list()` is `[]` -- the type's zero value, not a conversion of
        # nothing. Requiring exactly one argument rejected a program CPython
        # accepts, and `defaultdict(list)` is the one that hits it.
        print(int(), list(), tuple(), repr(str()), float())
        print(dict(), set(), repr(bytes()), bool(), frozenset())
        # AS A VALUE TOO: the thunk's parameter has to be optional, or the
        # call through it reports an arity error the direct call does not.
        make = list
        empty = dict
        print(make(), empty(), make([1, 2]))

    """,
    "module_def_can_be_rebound": """
        # A NAME DEFINED TWICE IS REBOUND, not an error: the second `def`
        # replaces it and the first stays reachable only through whatever
        # already held it. Refusing this rejected a program CPython runs --
        # `@f.register` over two `def _`s is the idiom that needs it.
        def f():
            return 1

        # Called BETWEEN the two, so it must reach the FIRST body. Resolving
        # the name where the call is compiled picked the survivor instead.
        print(f())

        def f():
            return 2

        print(f())

        def g():
            return "a"

        held = g

        def g():
            return "b"

        print(held(), g())

    """,
    "oserror_subsumes_the_old_names": """
        # PEP 3151: `IOError` and `EnvironmentError` ARE `OSError` -- the same
        # object, not subclasses -- and the errno decides which specific class
        # a two-argument construction builds.
        print(IOError is OSError, EnvironmentError is OSError)
        print(issubclass(FileNotFoundError, OSError),
              issubclass(BrokenPipeError, OSError))
        e = OSError(2, "No such file")
        # EVERY ARGUMENT IS KEPT: one was all the constructor took, so the
        # rest were dropped and `e.args` reported a one-element tuple.
        print(e.errno, e.strerror, e.args)
        print(type(e).__name__)
        # AND MORE THAN ONE PRINTS AS THE TUPLE, which is how CPython shows
        # `args` once there is more than one of them to show.
        two = ValueError("a", "b")
        print(str(two), repr(two), two.args)
        try:
            raise IOError("late")
        except OSError as caught:
            print("caught", caught)

    """,
    "dict_resized_while_walked": """
        # The table is rehashed by the write, so continuing the walk would
        # skip or repeat entries. CPython refuses rather than losing them.
        d = {"a": 1, "b": 2}
        try:
            for k in d:
                d["c"] = 3
        except RuntimeError:
            print("RuntimeError")
        print(len(d))
        # A LIST MAY CHANGE UNDER A WALK and CPython allows it -- the walk
        # simply sees the new length, so this must NOT refuse.
        xs = [1, 2, 3]
        seen = []
        for x in xs:
            seen.append(x)
            if len(xs) < 5:
                xs.append(x * 10)
        print(len(seen) > 3, xs[:5])

    """,
    "codecs_are_real_conversions": """
        # `encode` and `decode` IGNORED THE ENCODING ENTIRELY: text is held as
        # UTF-8, so the utf-8 spelling was right by accident and every other
        # one silently answered the internal bytes.
        s = "a\\u00e9\\u4e2d"
        for enc in ("utf-8", "utf-16", "utf-32"):
            b = s.encode(enc)
            print(enc, len(b), b.decode(enc) == s)
        try:
            s.encode("ascii")
        except UnicodeEncodeError:
            print("UnicodeEncodeError")
        # THE ERROR HANDLER DECIDES whether a bad byte is a refusal or a
        # replacement, so it has to reach the runtime rather than be dropped.
        print(ascii(s.encode("ascii", "replace")))
        raw = b"\\xff\\x61"
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            print("UnicodeDecodeError")
        print(ascii(raw.decode("utf-8", "replace")))
        print(ascii(raw.decode("utf-8", "ignore")))
        # EVERY BYTE IS A CODE POINT in latin-1, which is what makes it the
        # round trip for arbitrary octets.
        print(ascii(raw.decode("latin-1")), len(raw.decode("latin-1")))

    """,
    "module_objects_are_namespaces": """
        import types

        m = types.ModuleType("m")
        m.__getattr__ = lambda name: "dynamic:" + name
        print(m.__getattr__("anything"))
        # The CLASS is called `module`; `ModuleType` is the name it is
        # exported under and not the one the type carries.
        print(type(m).__name__)
        ns = types.SimpleNamespace(b=2, a=1)
        print(ns, ns.a + ns.b)

    """,
    "multiple_inheritance_and_the_mro": """
        # A second base used to be REFUSED, because every lookup walked a
        # straight chain of base pointers. C3 is the only order that keeps two
        # promises at once -- a class comes before its bases, and the bases
        # keep the order they were written in -- and no simple walk keeps both.
        class A:
            v = "A"

            def who(self):
                return "A"

        class B(A):
            v = "B"
            w = "Bw"

            def who(self):
                # `super()` HERE must reach C on a D instance and A on a B
                # one: only the RECEIVER's order knows what sits between.
                return "B" + super().who()

        class C(A):
            def who(self):
                return "C" + super().who()

        class D(B, C):
            def who(self):
                return "D" + super().who()

        print(D().who())
        print([k.__name__ for k in D.__mro__])
        print([k.__name__ for k in B.__mro__])

        # TWO UNRELATED BASES: the FIRST one wins for a name both define,
        # which is what "the written order is binding" means.
        class Mix:
            v = "Mix"
            m = "Mm"

        class Left(A, Mix):
            pass

        class Right(Mix, A):
            pass

        print(Left.v, Left.m, Right.v)
        print([k.__name__ for k in Left.__mro__])
        print([k.__name__ for k in Left.__bases__])
        print(isinstance(D(), A), isinstance(D(), C), issubclass(D, B))

    """,
    "mro_conflict_is_rejected": """
        # No order satisfies both promises, so there is no class to make.
        class A:
            pass

        class B(A):
            pass

        try:
            class Bad(A, B):
                pass
        except TypeError:
            print("TypeError")

        class Ok(B, A):
            pass

        print([k.__name__ for k in Ok.__mro__])

    """,
    "yield_from_steps_the_inner_generator": """
        # Delegation DRAINED the inner generator, so a value sent to the outer
        # one arrived after the inner had already run past every `yield` that
        # could read it -- and an infinite inner generator never started.
        def inner():
            got = yield "inner-1"
            yield ("inner-got", got)
            return "inner-return"

        def outer():
            result = yield from inner()
            yield ("outer-saw", result)

        g = outer()
        print(next(g))
        print(g.send("x"))
        print(next(g))

        # AND A NON-GENERATOR STILL WORKS: it has nowhere to put the sent
        # value and is simply advanced.
        def over(xs):
            yield from xs

        print(list(over([1, 2, 3])))

    """,
    "signature_is_recoverable_from_code": """
        # `co_varnames` left `*args` and `**kw` out entirely, and the defaults
        # were split on the COUNT OF KEYWORD-ONLY PARAMETERS -- so `def f(a,
        # b=1, *args, c)` reported `b`'s default as `c`'s.
        import inspect

        def f(a, b=1, *args, c, **kw):
            pass

        print(f.__defaults__, f.__kwdefaults__)
        code = f.__code__
        print(code.co_varnames, code.co_argcount, code.co_kwonlyargcount)
        sig = inspect.signature(f)
        print(str(sig))
        print(list(sig.parameters))
        print(sig.parameters["b"].default, sig.parameters["c"].kind.name)

    """,
    "pep695_type_parameters": """
        # PEP 695. A type alias is a NAME plus what it stands for; the type
        # parameters are in scope for the value and readable off the thing
        # they belong to.
        type Alias = list[int]

        def first[T](xs: list[T]) -> T:
            return xs[0]

        class Box[T]:
            def __init__(self, v: T):
                self.v = v

        print(Alias.__name__, Alias.__value__, type(Alias).__name__)
        print(first([1, 2]), Box("x").v)
        print(first.__type_params__[0].__name__)

    """,
    "unicode_predicates_are_exact": """
        # The predicates walked BYTES, so a multi-byte character was asked
        # about its own continuation bytes -- which belong to no class -- and
        # every non-ASCII string answered False. And `isdecimal`, `isdigit`
        # and `isnumeric` shared one test, which they are not.
        for s in ("abc123", "123", "Abc", "\u00b2", "\u2167", "\u00e9"):
            print(s.isalnum(), s.isalpha(), s.isdigit(), s.isdecimal(),
                  s.isnumeric(), s.isidentifier())
        print("\u03bb".isidentifier(), "caf\u00e9".isidentifier())
        print("1\u03bb".isidentifier(), "_\u4e2d".isidentifier())
        print("\u00c9".isupper(), "\u00e9".islower(), "\u00c9\u00e9".istitle())

    """,
    "range_is_a_lazy_sequence": """
        # `range` was MATERIALISED into a list, so `type(range(3)).__name__`
        # said `list` and `range(10**12)` would have built a trillion
        # elements. It is three numbers now, and every question about one is
        # arithmetic on them.
        r = range(0, 10, 2)
        print(r.start, r.stop, r.step, len(r))
        print(r[2], r[-1], r.index(4), r.count(4), r.count(5))
        print(list(r[1:3]), list(r[::-1]))
        print(range(3) == range(3), range(0, 3, 1) == range(3),
              range(3) == range(4))
        print(type(r).__name__, repr(range(3)), repr(r))
        big = range(10 ** 12)
        print(len(big), 10 ** 11 in big, -1 in big)
        print(list(range(5)), list(range(10, 0, -3)), sum(range(5)))
        print(sorted(range(3), reverse=True), list(reversed(range(4))))
        print(min(range(2, 9)), max(range(2, 9)), tuple(range(3)))
        for v in range(3):
            print("v", v)

    """,
    "a_base_need_not_be_a_class": """
        # PEP 560: `class C(Fake())` asks the object for `__mro_entries__` and
        # inherits whatever it answers. A base had to be a plain NAME, so
        # every library that builds one at run time was refused.
        class Base:
            def who(self):
                return "Base"

        class Fake:
            def __mro_entries__(self, bases):
                return (Base,)

        class C(Fake()):
            pass

        print([k.__name__ for k in C.__mro__])
        print(C().who())
        # `__orig_bases__` is what the statement ACTUALLY SAID, which the
        # resolved base has lost.
        print(C.__orig_bases__[0].__class__.__name__)

    """,
    "a_class_may_extend_a_builtin": """
        # `class D(dict)` was refused: a base had to be a class the module
        # defines. An instance of one now carries a real dict, and everything
        # the body does not write is answered from it.
        class WithMissing(dict):
            def __missing__(self, k):
                return "default:" + k

        w = WithMissing()
        print(w["nope"], len(w), isinstance(w, dict), isinstance(w, object))
        w["a"] = 1
        print(w["a"], len(w))
        del w["a"]
        print(len(w), isinstance({}, WithMissing))

        # `del obj[k]` IS `obj.__delitem__(k)` -- never dispatched before, so
        # a class that wrote one had it ignored.
        class Store:
            def __init__(self):
                self.d = {}

            def __setitem__(self, k, v):
                self.d[k] = v

            def __delitem__(self, k):
                del self.d[k]

            def __len__(self):
                return len(self.d)

        s = Store()
        s["x"] = 1
        print(len(s))
        del s["x"]
        print(len(s))

    """,
    "metaclass_is_inherited": """
        # A class with no `metaclass=` still has one when its BASE does, and
        # that cannot be decided where the class is compiled -- it is a
        # property of a run-time value. The lowering used to pick between two
        # shapes statically and so a subclass of a metaclassed base was built
        # as a plain type, silently losing every hook.
        class Meta(type):
            def __call__(cls, *a, **kw):
                print("through", cls.__name__)
                # `type.__call__` is the ordinary instantiation. Reaching it
                # is the only way this hook can end without calling itself.
                return super().__call__(*a, **kw)

            def tag(cls):
                return "tag:" + cls.__name__

        class Base(metaclass=Meta):
            def __init__(self, x):
                self.x = x

        class Sub(Base):
            pass

        print(type(Base).__name__, type(Sub).__name__)
        print(Sub(7).x)
        # A method the METACLASS defines is reached through the class, bound
        # to it -- the same relationship an instance has to its class.
        print(Base.tag(), Sub.tag())

    """,
    "metaclass_takes_class_keywords": """
        # `class C(metaclass=M, kind="x")` is `M(name, bases, ns, kind="x")`,
        # and the keywords have to be matched against `M.__new__` -- which is
        # where they are declared, `__init__` being `type`'s default.
        class M(type):
            def __new__(mcls, name, bases, ns, kind=None):
                cls = super().__new__(mcls, name, bases, ns)
                cls.kind = kind
                return cls

            def __iter__(cls):
                # Iterating a CLASS is the metaclass's business.
                return iter(cls.members)

        class C(metaclass=M, kind="x"):
            members = [1, 2, 3]

        print(C.kind, list(C), [n * 2 for n in C])

    """,
    "class_name_is_writable": """
        # `C.__name__ = ...` changes what the class is called. The name is a
        # field on the type rather than an entry in its dict, so storing it as
        # an ordinary attribute left `__name__` reading the old one -- a write
        # that appeared to succeed and changed nothing.
        class C:
            pass

        C.__name__ = "Renamed"
        # `__name__` only: CPython's class repr is built from `__qualname__`
        # and the module, neither of which this write touches, and the module
        # half is a divergence of its own.
        print(C.__name__)

        def f():
            pass

        f.__name__ = "g"
        print(f.__name__)

    """,
    "property_is_a_named_descriptor": """
        # `hasattr(p, "__get__")` is how a program asks whether something is a
        # descriptor. The behaviour existed with nothing naming it, so a
        # property answered False and read as an ordinary attribute.
        class C:
            def __init__(self):
                self._v = 1

            @property
            def v(self):
                return self._v

            @v.setter
            def v(self, new):
                self._v = new

        p = C.__dict__["v"]
        print(hasattr(p, "__get__"), hasattr(p, "__set__"))
        obj = C()
        print(p.__get__(obj, C))
        p.__set__(obj, 9)
        print(obj.v)

    """,
    "object_and_type_are_values": """
        # `object.__new__(cls)` is what a metaclass calls to build an instance
        # without running the class's own `__new__`, and neither `object` nor
        # `type` is a kind the way `int` is -- both are class objects the
        # moment a program names one.
        class C:
            def __init__(self):
                raise AssertionError("__init__ must not run")

        made = object.__new__(C)
        print(type(made).__name__, isinstance(made, C))
        print(object.__name__, type.__name__)

        class Meta(type):
            pass

        print(Meta.__base__ is type)

    """,
    "init_subclass_keywords": """
        # `class A(Base, tag="a")` is how a program CONFIGURES the hook, and
        # the keywords were dropped -- so every subclass looked identically
        # unconfigured, which is a wrong answer rather than a refusal. They
        # have to be matched by NAME against the hook's parameters, not handed
        # over as `**kw`.
        seen = []

        class Base:
            def __init_subclass__(cls, tag=None, **kw):
                seen.append((cls.__name__, tag, sorted(kw.items())))
                # `object.__init_subclass__` EXISTS and is the no-op that
                # terminates the chain; it had no value naming it.
                super().__init_subclass__(**kw)

        class A(Base, tag="a"):
            pass

        class B(Base):
            pass

        # NOT a class with an unconsumed keyword: `object.__init_subclass__`
        # takes none, so CPython raises there too and the reference would
        # disagree with itself.
        class C(Base, tag="c"):
            pass

        print(seen)
    """,
    "descriptors_learn_their_names": """
        # PEP 487: a descriptor is TOLD ITS OWN NAME after the class body is
        # complete. It cannot know it otherwise -- the expression that built
        # it had no idea what it was about to be assigned to.
        class Field:
            def __set_name__(self, owner, name):
                self.name = name
                self.owner = owner.__name__
            def __get__(self, obj, objtype=None):
                return "field:" + self.name + "@" + self.owner

        class C:
            a = Field()
            b = Field()

        print(C().a, C().b)

        # `@v.deleter` -- the third of the three. `del obj.v` had a slot to
        # read and no way to fill it, so it looked in the instance dict, found
        # nothing (a property never puts anything there) and reported a
        # missing attribute for one the class plainly defines.
        class P:
            def __init__(self):
                self._v = 1
            @property
            def v(self):
                return self._v
            @v.setter
            def v(self, n):
                self._v = n
            @v.deleter
            def v(self):
                self._v = "deleted"

        p = P()
        print(p.v)
        p.v = 5
        print(p.v)
        del p.v
        print(p.v, type(P.v).__name__)
    """,
    "type_is_an_object": """
        # `type(x)` answers a TYPE OBJECT, not its name. The name was a str,
        # so `type(a) is type(b)` compared two separately-built strings and was
        # False for two ints -- and `print(type(x))` said `int` where CPython
        # says `<class 'int'>`.
        #
        # `type(1) is int` holds because the frontend registers the canonical
        # thunk for every builtin type the module names at the TOP OF THE
        # ENTRY, before any statement. Registering lazily made the answer
        # depend on which side was evaluated first, which is why an earlier
        # attempt at this was reverted.
        print(type(1) is int, type("a") is str, type(1.5) is float)
        print(type([]) is list, type({}) is dict, type(1) is str)
        print(type(1), type("a"), type([]), type(1).__name__)
        small, big = 1, 10 ** 30
        print(type(small) is type(big), type(True) is bool)
        print(isinstance(1, type(2)), isinstance(1, object))
        print(issubclass(bool, int), issubclass(int, int), issubclass(int, str))

        class C:
            pass

        print(type(C()) is C, type(C).__name__, type(None).__name__)
        # STILL CALLABLE, which is the whole reason it is a thunk.
        print(int("7") + 1, list(map(int, ["1", "2"])), str(9) + "!")
    """,
    "a_builtin_type_is_one_object": """
        # `int` mentioned twice built two thunks, so `int == int` was False
        # and a set of types compared unequal to itself. Interning by NAME has
        # no evaluation order to depend on -- unlike the registry tried and
        # reverted for `type(1) is int`, which made the answer depend on which
        # of `type(1)` and `int` the program reached first.
        print(int is int, int == int, int is str, str == str)
        print({int, str} == {str, int}, [int] == [int], {int: 1}[int])
        # STILL CALLABLE, which is the whole reason it is a thunk.
        print(list(map(int, ["1", "2"])), int("7") + 1, isinstance(1, int))
        print(int, str, bool, float, list, dict)
        # A BUILTIN REACHED AS A VALUE is not a plain function:
        # `type(print).__name__` is `builtin_function_or_method`. A
        # synthesised thunk is an ordinary compiled function without the flag.
        def written():
            pass

        print(type(print).__name__, type(len).__name__)
        print(type(written).__name__, type(int).__name__)
    """,
    "typing_introspection": """
        # `get_origin(Literal["a"]) is Literal` is only True if the two
        # mentions of `Literal` are ONE object -- forms are interned by name
        # for the same reason the suspension token and `NotImplemented` are.
        from typing import Literal, TypeGuard, get_args, get_origin
        L = Literal["a", "b"]
        print(get_args(L), get_origin(L) is Literal, get_origin(L) is L)
        print(get_args(TypeGuard[int]), get_origin(list[int]))
        print(get_args(list[int]), get_args(dict[str, int]))
        # Anything that is not a parameterised type answers None and ().
        print(get_origin(42), get_args(42), get_origin(list), get_args("a"))
        # A UNION IS NOT A GENERIC ALIAS to a program that asks -- the name is
        # how it tells the two apart.
        print(type(int | str).__name__)
    """,
    "containers_render_their_elements": """
        # A container shows its elements with REPR, whichever of str/repr was
        # asked of the container -- and every element has to go through the
        # same renderer the element alone would. The interpreter handed the
        # job to Python's `repr`, which printed an ADDRESS for anything the
        # runtime defines and raised out of the bridge for a user instance.
        class P:
            def __init__(self, n):
                self.n = n
            def __repr__(self):
                return "P(" + str(self.n) + ")"

        print(P(1), [P(1), P(2)], (P(3),))
        print({"k": P(4)}, [KeyError("k")])
        print([int], (int,), {"t": str})
        print([], (), {}, set(), frozenset(), frozenset([1]))
        print((1,), (1, 2), ["a"], {"a": "b"})
        # A container that holds itself prints the ellipsis rather than
        # recurring -- which Python's repr was doing for us.
        xs = [1]
        xs.append(xs)
        print(xs)
    """,
    "locals_and_globals": """
        # PEP 667: an INDEPENDENT SNAPSHOT. Writing to the dict must not reach
        # the local, and assigning the local afterwards must not show up in
        # the dict -- both fall out of it being an ordinary dict built at the
        # call site, which is the only place the name-to-register mapping
        # still exists.
        g = "global"

        def f():
            x = 1
            snapshot = locals()
            x = 2
            snapshot["x"] = 99
            return snapshot.get("x"), locals().get("x"), x

        print(f())

        # A local the branch did not bind is ABSENT, not present as None --
        # and no ordinary read reaches it, so nothing but `locals()` itself
        # can be what decides its register needs the null it starts from.
        def maybe(flag):
            if flag:
                v = 1
            return locals().get("v", "unbound"), sorted(locals())

        print(maybe(True))
        print(maybe(False))
        print("g" in globals(), "f" in globals(), "nosuch" in globals())

        # `dir()` WITH NO ARGUMENT is the names in scope, sorted -- which is
        # `sorted(locals())`, and now that `locals()` exists there is nothing
        # else to build.
        def scoped():
            a = 1
            b = 2
            return dir()

        print(scoped(), "g" in dir(), "nosuch" in dir())

        def reads_the_global():
            return sorted(k for k in globals() if not k.startswith("_"))

        print(reads_the_global()[:2])
        # NOT `__builtins__` here. It is a module when a script runs and a
        # plain dict under `exec`, which is how this corpus runs CPython --
        # so the reference would disagree with itself, not with us.
        # `scoping/locals-and-globals-builtins` covers it as a real script.
    """,
    "mutable_buffers": """
        # A bytearray is the bytes kind with the buffer writable, and a
        # memoryview is a window on someone else's -- so what this pins is
        # that a write is SEEN through the other name, and that `bytes()` of
        # either is a snapshot that then stops changing.
        ba = bytearray(b"abcd")
        mv = memoryview(ba)
        frozen = bytes(mv)
        mv[0] = 122
        print(ba, frozen, bytes(mv))
        print(bytes(mv[1:3]), bytes(mv[::-1]))
        print(len(mv), mv.readonly, mv.nbytes, mv.itemsize, mv.format)
        print(memoryview(b"xy").readonly, bytes(bytearray(3)))
        # Slicing a bytearray gives a bytearray; adding to one keeps the
        # LEFT operand's kind, which is what CPython does.
        print(ba[1:3], ba + b"e", b"e" + ba)
        print(ba == bytearray(b"zbcd"), ba == b"zbcd", type(ba).__name__)
        try:
            {ba: 1}
        except TypeError:
            print("unhashable")
        try:
            memoryview(b"xy")[0] = 1
        except TypeError:
            print("read-only")
        # BY CONTENT, not by buffer address. Two identical literals share one
        # static buffer in the compiled program, so comparing those two would
        # have passed while a bytes value BUILT at run time compared unequal
        # to the literal it matches -- and every dict lookup and `in` test on
        # a bytes key went the same way.
        built = b"a" + b"b"
        print(built == b"ab", built != b"ab", b"ab" == b"ac")
        print(b"ab" in [b"a" + b"b"], {b"ab": 1}[built])
    """,
    "dict_views_are_live": """
        # A VIEW IS A WINDOW ON THE DICT, not a copy. A snapshot is the
        # obvious implementation and is wrong exactly when a program relies on
        # the view being live -- after the dict changes.
        d = {'a': 1}
        ks = d.keys()
        print(sorted(ks))
        d['b'] = 2
        print(sorted(ks), len(ks))
        print(sorted(d.values()), sorted(d.items()))
        print('a' in d.keys(), 'z' in d.keys(), list(d.keys()))
        for k in d.keys():
            print('k', k)
        # A view is SET-LIKE: `&`, `|`, `-` all work against a real set.
        print(sorted(d.keys() & {'a'}), sorted(d.keys() | {'c'}))
        print(sorted(d.keys() - {'a'}))
    """,
    "float_hex": """
        # `hex` REACHES BYTES OR A FLOAT, and the two answer entirely
        # different things -- so the no-argument form dispatches on the
        # receiver, as `pop`, `split` and `count` already do.
        print((2.5).hex(), (0.0).hex(), (-0.0).hex())
        print((1.0).hex(), (0.1).hex(), float.fromhex('0x1.4p+1'))
        print(bytes([1, 255]).hex())
        print((2.5).is_integer(), (2.0).is_integer(), (2.5).as_integer_ratio())
    """,
    "dunders_on_builtins": """
        # DUNDERS CALLED DIRECTLY ON A BUILTIN are ordinary Python, and each
        # is the operation the runtime already performs for the operator form
        # -- the same symbol reached by another spelling, not a second
        # implementation that could disagree with it.
        print((-5).__abs__(), (0.0).__bool__(), (1.5).__trunc__())
        print((3).__neg__(), [1, 2].__len__(), (5).__repr__(), 'a'.__str__())
        # EVERY NUMBER HAS `real` AND `imag`, not only a complex one. An int's
        # imaginary part is the INT zero, which `type()` can tell apart.
        print((7).conjugate(), (7).real, (7).imag)
        print((2.5).real, (2.5).imag, type((7).imag).__name__)
    """,
    "floor_and_ceil_cross_into_big": """
        # A DOUBLE OF MAGNITUDE 2**63 OR MORE IS ALREADY A WHOLE NUMBER, and
        # flooring it has to produce an integer that big -- which means
        # building a big out of the mantissa and the exponent. Nothing is
        # rounded here and nothing may be lost: the value has no fractional
        # part left, so the answer is exact or it is wrong.
        import math
        for v in (0.0, 1.5, -1.5, 2.5, -2.5, 1e17, 1e18, 1e19, -1e19,
                  2.0 ** 62, 2.0 ** 63, -(2.0 ** 63), 2.0 ** 64,
                  2.0 ** 100, -(2.0 ** 100), 1e30):
            print(math.floor(v), math.ceil(v), math.trunc(v))
        print(2 ** 100 == math.floor(float(2 ** 100)))
        print(math.floor(1e19) // 2, math.ceil(-(2.0 ** 100)) % 7)
        # AN INTEGER IS ANSWERED UNCHANGED, and a BOOL becomes an int:
        # `math.floor(True)` is `1` and not `True`.
        print(math.floor(5), math.ceil(2 ** 100), math.trunc(-7))
        print(repr(math.floor(True)), repr(math.ceil(False)))
    """,
    "deletion_reports_what_it_could_not_find": """
        # FOUR SHAPES THAT LOOK ALIKE AND ARE NOT. A dict deletes by KEY and
        # names the key; a list deletes by INDEX and names a range; a slice
        # deletes a SPAN and is not an index at all; an attribute deletes from
        # the instance dict and owes an AttributeError rather than the
        # KeyError the dict underneath it would raise.
        d = {"a": 1, "b": 2, "c": 3}
        del d["b"]
        print(d, list(d))
        try:
            del d["zz"]
        except KeyError as e:
            print("KeyError:", e)
        xs = list(range(10))
        del xs[3]
        del xs[-1]
        del xs[1:4]
        print(xs)
        try:
            del xs[50]
        except IndexError as e:
            print("IndexError:", e)
        try:
            del (1, 2)[0]
        except TypeError as e:
            print("TypeError:", e)

        class Box:
            def __init__(self): self.d = {}
            def __delitem__(self, k):
                print("__delitem__", k)

        del Box()["x"]

        class C:
            def __init__(self): self.x = 1

            @property
            def p(self): return 1

        c = C()
        del c.x
        print(hasattr(c, "x"))
        try:
            del c.x
        except AttributeError as e:
            print("AttributeError:", e)
        # A PROPERTY WITH NO DELETER REFUSES rather than falling through to
        # the instance dict, which never held it.
        #
        # THE TYPE AND NOT THE MESSAGE, because CPython's wording here --
        # `property 'p' of 'C' object has no deleter` -- is one this runtime
        # does not reproduce, and a case that asserted it would be asserting
        # a known divergence rather than the behaviour. What matters and is
        # checked is that it refuses at all: it used to report a missing
        # attribute for one the class plainly defines.
        try:
            del c.p
        except AttributeError as e:
            print("refused:", type(e).__name__)
    """,
    "ascii_escapes_what_repr_keeps": """
        # `ascii(x)` IS `repr(x)` WITH NOTHING ABOVE 0x7F LEFT IN IT. The two
        # agree for every ASCII value, which is why `!a` could be folded into
        # `!r` for a long time without anything noticing.
        #
        # THE PERMISSIVE UTF-8 STEP, not the validating one: `ascii` names
        # whatever byte is there rather than refusing it, because dropping a
        # bad byte or emitting it raw are both worse than `ÿ`.
        for s in ("abc", "café", "中文", "😀",
                  " ", "it's", "tab	here"):
            print(ascii(s))
        print(ascii([1, "café"]), ascii({"é": "ü"}))
        print(ascii(5), ascii(None), ascii(b"ab"))
        # AND THE THREE PLACES A CONVERSION IS SPELLED, which reach it by
        # three different routes: an f-string is lowered by the frontend,
        # `str.format` parses its own replacement fields, and `%a` goes
        # through the percent formatter.
        print(f"{'café'!a}", "{!a}".format("café"), "%r" % "café")
        print(f"{'é'!a:>10}|{None!a}")
    """,
    "oserror_shows_its_errno": """
        # `str()` OF THE OSError FAMILY IS NOT ITS ARGUMENTS. CPython puts
        # `[Errno n] message` on the two-argument form and appends the quoted
        # filename on the three -- and the whole family arrives under its own
        # name, so this is a walk up the hierarchy and not a test for
        # `OSError` itself. `repr` is unaffected and still shows the call.
        #
        # THE FILENAME IS QUOTED AND THE MESSAGE IS NOT, which reads as an
        # inconsistency and is deliberate: the message is prose, and a path
        # with a trailing space is invisible unrendered.
        for e in (OSError(2, "No such file"),
                  OSError(2, "No such file", "f.txt"),
                  PermissionError(13, "Permission denied"),
                  OSError("plain"),
                  OSError()):
            print(str(e))
        print(str(ValueError(1, "not an oserror")))
    """,
    "self_referential_repr": """
        # A CONTAINER THAT HOLDS ITSELF is an ordinary thing to build, and
        # rendering it naively recurses until the stack runs out. Python
        # prints the repeat as `[...]`, which is what makes it finite.
        xs = [1]
        xs.append(xs)
        print(xs, len(xs), xs[1] is xs)
        d = {'a': 1}
        d['self'] = d
        print(d)
        t = ([1],)
        t[0].append(t)
        print(t)
        # Nested but NOT cyclic still renders in full -- the check is about
        # re-entry, not about depth.
        print([[1, 2], [3]], {'x': {'y': 1}})
    """,
    "raise_a_variable": """
        # `raise e` WHERE `e` IS A VARIABLE re-raises the object it holds.
        # Only a name that IS an exception type means "make one of these" --
        # treating every name that way built an exception named after the
        # variable, so the handler for the real type never fired.
        e = ValueError('v')
        try:
            raise e
        except ValueError as x:
            print('caught', x, type(x).__name__)
        for exc in (ValueError('v'), TypeError('t'), KeyError('k'),
                    IndexError('i')):
            try:
                raise exc
            except LookupError as x:
                print('LookupError', type(x).__name__)
            except (ValueError, TypeError) as x:
                print('ValueError-or-TypeError', type(x).__name__)
        print(issubclass(KeyError, LookupError),
              issubclass(ZeroDivisionError, ArithmeticError))
        # `sum` REFUSES STRINGS -- the concatenation works, which is exactly
        # why the refusal has to be explicit.
        print(sum([1, 2, 3]), sum([1.0], 0.0), sum([True, True, False]))
        try:
            sum(['a', 'b'], '')
        except TypeError as x:
            print('TypeError:', x)
    """,
    "notimplemented_falls_back": """
        # `NotImplemented` MEANS "ASK THE OTHER OPERAND", not "the answer is
        # NotImplemented". Returning it as the result made the comparison
        # answer the sentinel instead of falling back.
        class Left:
            def __eq__(self, other):
                return NotImplemented
        class Right:
            def __eq__(self, other):
                return 'right-wins'
        print(Left() == Right())
        # Neither side answers: the default is IDENTITY for `==`.
        print(Left() == Left())
        class A:
            def __add__(self, o):
                return NotImplemented
        class B:
            def __radd__(self, o):
                return 'B-wins'
        print(A() + B())
        # Neither side answers for arithmetic: a TypeError naming the pair.
        try:
            A() + A()
        except TypeError as e:
            print('TypeError:', e)
        print(NotImplemented is NotImplemented, str(NotImplemented))
    """,
    "numeric_conversion_dunders": """
        # A CLASS SAYS WHAT ITS NUMBER IS. Answering from the numeric tower
        # instead converted something the class never claimed was a number.
        class N:
            def __int__(self):
                return 7
            def __float__(self):
                return 7.5
            def __complex__(self):
                return complex(1, 2)
            def __round__(self, nd=None):
                return 'round:' + str(nd)
            def __bool__(self):
                return False
        n = N()
        print(int(n), float(n), complex(n))
        print(round(n), round(n, 2))
        print(bool(n))
        # `__index__` stands in for both when the class defines only it.
        class I:
            def __index__(self):
                return 3
        print(int(I()), float(I()))
        # `complex(x)` asks the class; `complex(x, 0)` is building from parts
        # and has nothing to ask -- so "omitted" and "given as 0" are not the
        # same thing.
        print(complex(), complex(1), complex(1, 2), complex(1.5, -2))
        print(round(2.5), round(2.567, 2), round(7))
    """,
    "iteration_protocol_hooks": """
        # `__reversed__` WINS OVER THE INDEX WALK. A class may define both it
        # and `__getitem__`, and they need not agree -- walking indices
        # backwards instead silently produced a different sequence.
        class C:
            def __reversed__(self):
                return iter(['z', 'y'])
            def __len__(self):
                return 2
            def __getitem__(self, i):
                return 'ab'[i]
        print(list(reversed(C())))
        class Seq:
            def __len__(self):
                return 3
            def __getitem__(self, i):
                return i * 10
        print(list(reversed(Seq())))
        # WHAT `__iter__` RETURNS MUST BE AN ITERATOR. A str is not one however
        # walkable it looks; accepting it turned a broken class into a working
        # one that iterated something else entirely.
        class BadIter:
            def __iter__(self):
                return 'not-an-iterator'
        try:
            list(BadIter())
        except TypeError as e:
            print('TypeError:', e)
        class GoodIter:
            def __iter__(self):
                return iter([1, 2])
        print(list(GoodIter()))
        # A SET HAS NO ORDER TO REVERSE, and it has a length -- so the index
        # walk would have answered confidently.
        try:
            reversed({1, 2})
        except TypeError as e:
            print('TypeError:', e)
        print(list(reversed([1, 2, 3])), list(reversed('abc')))
    """,
    "ascii_escapes": """
        # `ascii` IS NOT `repr`. Its answer has to survive a channel that
        # cannot carry the character, so handing back the character defeats
        # the whole point of it.
        s = 'a' + chr(233)
        print(repr(s))
        print(ascii(s))
        print(ascii([s]))
        print(ascii('plain'), ascii(1), ascii(b'a'))
        print(ascii(chr(0x4e2d)))
    """,
    "sys_module": """
        import sys
        # ONLY WHAT CAN BE ANSWERED HONESTLY: most of `sys` describes a running
        # interpreter that is not there. What it does know is which
        # implementation compiled the program.
        print(type(sys.implementation.name).__name__)
        print(len(sys.implementation.version) >= 3)
        print(sys.implementation.name == sys.implementation.name.lower())
        print(isinstance(sys.implementation.hexversion, int))
        print(sys.byteorder, sys.maxsize > 0)
    """,
    "builtin_types_as_values": """
        # A BUILTIN TYPE NAME IS BOTH. As a value it is a callable -- which is
        # what `map(str, xs)` needs -- and it is also a class, which is what
        # `print(int)` and `isinstance(x, t)` ask about. Answering
        # `<function int at 0x...>` to the second was a wrong answer.
        print(int, str, list, dict, bool)
        print(list[int], dict[str, int], tuple[int, str])
        t = int
        print(isinstance(1, t), isinstance(1.0, t), isinstance(True, t))
        print(list(map(str, [1, 2])), sorted([3, 1], key=int))
        print(int('42'), str(9), list((1, 2)), dict([('a', 1)]))
        print(type(1).__name__)
        class C:
            def __class_getitem__(cls, item):
                return str(cls.__name__) + '[' + str(item) + ']'
        print(C[int])
        # A class WITHOUT the hook is not subscriptable, as CPython says.
        class D:
            pass
        try:
            D[int]
        except TypeError as e:
            print('TypeError:', e)
    """,
    "init_subclass_hook": """
        # ANNOUNCED TO THE BASE, not to the class itself, and after the body
        # has been filled -- the hook routinely reads what the body bound.
        seen = []
        class Base:
            def __init_subclass__(cls, **kw):
                seen.append(cls.__name__)
        class A(Base):
            pass
        class B(Base):
            pass
        class C(A):
            pass
        print(seen)
        names = []
        class Reg:
            def __init_subclass__(cls, **kw):
                names.append((cls.__name__, getattr(cls, 'tag', None)))
        class R1(Reg):
            tag = 'one'
        class R2(Reg):
            tag = 'two'
        print(names)
    """,
    "exception_groups": """
        # PEP 654. A group IS an exception -- it propagates the same way -- and
        # what distinguishes it is the exceptions it carries.
        eg = ExceptionGroup('outer', [ValueError('v1'),
                                      ExceptionGroup('inner', [TypeError('t1')])])
        print(type(eg).__name__, len(eg.exceptions))
        # `split` PRESERVES THE NESTING: a match inside an inner group comes
        # back inside an inner group, so the two halves add up to the original.
        m, rest = eg.split(ValueError)
        print(type(m).__name__, len(m.exceptions))
        print(type(rest).__name__, len(rest.exceptions))
        print(eg.subgroup(TypeError) is not None, eg.subgroup(KeyError) is None)
        print(isinstance(eg, Exception), isinstance(eg, ExceptionGroup))
        try:
            raise ExceptionGroup('boom', [ValueError('a')])
        except ExceptionGroup as g:
            print('caught', len(g.exceptions))
        # An exception type reached AS A VALUE, which `split` relies on and
        # which used to answer False for every exception.
        t = ValueError
        print(isinstance(ValueError('x'), t), isinstance(ValueError('x'), KeyError))
        print('a b'.split(), 'a,b'.split(','))
    """,
    "dir_builtin": """
        class C:
            def __dir__(self):
                return ['b', 'a', 'a']
        # SORTED BUT NOT DEDUPLICATED: CPython sorts what the hook returned
        # and hands it back.
        print(dir(C()))
        class D:
            x = 1
            def m(self):
                pass
        d = D()
        d.own = 2
        names = dir(d)
        print('x' in names, 'm' in names, 'own' in names)
        class E(D):
            y = 3
        print('x' in dir(E()), 'y' in dir(E()))
        print(dir(D) == sorted(dir(D)))
    """,
    "async_context_manager": """
        import asyncio
        log = []
        class ACM:
            def __init__(self, swallow):
                self.swallow = swallow
            async def __aenter__(self):
                log.append('aenter')
                await asyncio.sleep(0)
                return 'value'
            async def __aexit__(self, et, ev, tb):
                log.append(('aexit', et.__name__ if et else None))
                await asyncio.sleep(0)
                return self.swallow
        async def main():
            async with ACM(False) as v:
                log.append(('body', v))
            # THE EXCEPTION PATH SUSPENDS: `await __aexit__(...)` returns from
            # the step function between computing the live exception and
            # re-raising it, so neither can live in a register. That is what
            # made the first version produce IR the verifier rejected.
            try:
                async with ACM(False):
                    raise ValueError('boom')
            except ValueError as e:
                log.append(('caught', str(e)))
            # A true return from `__aexit__` SWALLOWS rather than observes.
            async with ACM(True):
                raise KeyError('swallowed')
            log.append('after')
            return log
        print(asyncio.run(main()))
    """,
    "slice_objects": """
        # A SLICE REACHES A USER `__getitem__` AS AN OBJECT -- the class
        # decides what a slice of it means, and it can only do that if it is
        # handed one. Slicing a list still goes straight through without
        # allocating; only these paths build the object.
        class C:
            def __getitem__(self, key):
                if isinstance(key, slice):
                    return ('slice', key.start, key.stop, key.step)
                return ('index', key)
        c = C()
        print(c[1])
        print(c[1:2])
        print(c[::2])
        print(c[1:2, 3])
        print(slice(5), slice(1, 5), slice(1, 10, 2))
        s = slice(1, 10, 2)
        print(s.start, s.stop, s.step)
        print([0, 1, 2, 3, 4, 5][s])
        print('abcdef'[slice(2, 4)])
        print([1, 2, 3, 4][slice(None, None, -1)])
        print(slice(1, 5, 2).indices(10), slice(None, None, -1).indices(5))
        print(slice(-3, None).indices(10))
        # Assigning through a slice may CHANGE THE LENGTH, and does it in
        # place so every other name bound to the list sees it.
        xs = [0, 1, 2, 3]
        xs[1:3] = [9]
        print(xs)
        ys = [0, 1, 2, 3, 4]
        alias = ys
        ys[:] = [7, 8]
        print(ys, alias)
        zs = [1, 2, 3]
        zs[1:1] = [9, 9]
        print(zs)
    """,
    "match_statement": """
        class Point:
            __match_args__ = ('x', 'y')
            def __init__(self, x, y):
                self.x, self.y = x, y
        def f(v):
            match v:
                case []:
                    return 'empty'
                case [1, *rest]:
                    return 'one-then:' + str(rest)
                case [a, b] if a == b:
                    return 'pair-equal:' + str(a)
                case [a, *mid, b]:
                    return 'ends:' + str(a) + ',' + str(b) + ' mid=' + str(mid)
                case {'t': 'a', 'v': val}:
                    return 'tagged-a:' + str(val)
                case {'t': t, **rest}:
                    return 'tagged:' + str(t) + ' rest=' + str(sorted(rest))
                case Point(0, 0):
                    return 'origin'
                case Point(x=0, y=y):
                    return 'on-y:' + str(y)
                case Point(px, py):
                    return 'point:' + str(px) + ',' + str(py)
                case str(s):
                    return 'str:' + s
                case int(n) if n > 100:
                    return 'big:' + str(n)
                # `True` reaches this and NOT the `case True` below, because
                # bool is an int subclass -- the ordering is observable.
                case (int() | float()) as num:
                    return 'num:' + str(num)
                case None:
                    return 'none'
                case True:
                    return 'true'
                case other:
                    return 'other:' + str(other)
        for v in ([], [1, 2, 3], [4, 4], [1], [7, 8, 9], {'t': 'a', 'v': 5},
                  {'t': 'b', 'z': 1}, Point(0, 0), Point(0, 7), Point(3, 4),
                  'hi', 500, 3.5, None, True):
            print(f(v))
        def g(v):
            match v:
                case [[a, b], [c, d]]:
                    return a + b + c + d
                case _:
                    return -1
        print(g([[1, 2], [3, 4]]), g([1, 2]))
        # A `match` with nothing matching does nothing -- `case _` is optional.
        def h(v):
            out = 'untouched'
            match v:
                case 99:
                    out = 'ninetynine'
            return out
        print(h(99), h(1))
        # A str is NOT a sequence pattern: this must fall through.
        def s(v):
            match v:
                case [x, y]:
                    return 'seq:' + str(x) + str(y)
                case _:
                    return 'not-a-sequence'
        print(s('ab'), s([1, 2]))
    """,
    "reraise_runs_finally": """
        log = []
        def f():
            try:
                raise ValueError('x')
            except ValueError:
                log.append('caught')
                # A BARE `raise` RE-RAISES WHAT THE HANDLER CAUGHT. Entering a
                # handler clears the error flag, so this used to propagate
                # nothing at all -- the exception vanished and the outer
                # `except` never fired.
                raise
            finally:
                # AND THE `finally` STILL RUNS on the way out. Handler bodies
                # are lowered with their own `try` already popped, so a raise
                # there jumped straight to the enclosing handler.
                log.append('finally')
        try:
            f()
        except ValueError:
            log.append('outer')
        print(log)
        def g():
            try:
                yield 1
            except GeneratorExit:
                log.append('exit')
                raise
            finally:
                log.append('gfinally')
        it = g()
        next(it)
        it.close()
        it.close()
        print(log[3:])
        # `else` and `finally` together, with a handler that never runs: the
        # rethrow path must still terminate its block.
        try:
            print('fine')
        except ValueError:
            print('not reached')
        else:
            print('else ran')
        finally:
            print('finally ran')
        def h():
            yield 1
            return 'done'
        it2 = h()
        print(next(it2))
        try:
            next(it2)
        except StopIteration as e:
            # `next()` carries the generator's return value out; `yield from`
            # read it correctly while this spelling answered None.
            print('StopIteration', e.value)
    """,
    "descriptors": """
        class Base:
            @property
            def v(self):
                return 'base'
        class Sub(Base):
            @property
            def v(self):
                # THROUGH THE CLASS a property is ITSELF, which is the only
                # way an override reaches the getter it is extending.
                return 'sub:' + Base.v.fget(self)
        print(Base().v, Sub().v)
        class P:
            def __init__(self):
                self._v = 1
            @property
            def v(self):
                return self._v * 10
            @v.setter
            def v(self, n):
                self._v = n + 1
        p = P()
        print(p.v)
        p.v = 4
        print(p.v)
        class C:
            tag = 'base'
            @classmethod
            def make(cls):
                return cls.tag
            @staticmethod
            def plain(n):
                return n * 2
        class D(C):
            tag = 'derived'
        print(C.make(), D.make(), C.plain(3))
        # A DATA descriptor beats the instance dict; a NON-data one loses to
        # it. That difference is the whole protocol.
        class Data:
            def __get__(self, obj, t=None):
                return 'data'
            def __set__(self, obj, v):
                obj.__dict__['v'] = v
        class NonData:
            def __get__(self, obj, t=None):
                return 'non-data'
        class Holder:
            v = Data()
            n = NonData()
        h = Holder()
        h.v = 1
        h.__dict__['n'] = 'instance'
        print(h.v, h.n, h.__dict__['v'])
        # A class body is a SCOPE: it runs top to bottom and reads what it
        # has already bound.
        class Scoped:
            x = 1
            y = x + 1
        print(Scoped.y)
    """,
    "pop_across_receivers": """
        # ONE METHOD NAME, THREE RECEIVERS. `xs.pop(i)` takes an index,
        # `d.pop(k)` takes a key, and `s.pop()` takes nothing -- and which one
        # is meant is not known until run time.
        xs = [1, 2, 3]
        print(xs.pop(), xs.pop(0), xs)
        s = {9}
        print(s.pop(), len(s))
        d = {'a': 1, 'b': 2}
        print(d.pop('a'), d.pop('zz', 'dflt'), sorted(d))
        try:
            d.pop('nope')
        except KeyError as e:
            print('KeyError', e)
        print(d.popitem(), len(d))
        try:
            {}.popitem()
        except KeyError as e:
            print('empty:', e)
        try:
            [].pop()
        except IndexError as e:
            print('IndexError:', e)
    """,
    "set_iteration_order": """
        # CPython holds a set in an open-addressed table, so `{3, 1, 2}`
        # iterates as 1, 2, 3 -- the three land in slots 3, 1 and 2 of an
        # eight-slot table whatever order they were written in. Insertion
        # order, which this used to produce, is the one thing CPython's order
        # is never about, and seven conformance cases read it back.
        s = {3, 1, 2}
        print(list(s), len(s), sorted(s))
        print(list(set(range(20)))[:10])
        print(list({10, 3, 7, 1}))
        print(sorted({'b', 'a'}))
        d = {1, 2, 3}
        d.add(4)
        d.add(0)
        d.discard(2)
        print(list(d))
        print(list({1, 2} | {3, 4}), list({1, 2, 3} & {2, 3, 4}))
        print(list(frozenset({5, 2, 9})))
        print({1, 2, 3} == {3, 2, 1}, 2 in d, 99 in d)
        e = set()
        for i in [5, 3, 8, 1]:
            e.add(i)
        print(list(e))
        print([v for v in {4, 2, 6}], tuple({7, 3}))
    """,
    "inspect_coroutine_questions": """
        import asyncio
        import inspect
        async def coro():
            return 1
        def gen():
            yield 1
        async def agen():
            yield 1
        c = coro()
        print(inspect.iscoroutine(c), inspect.isgenerator(c))
        print(inspect.isgenerator(gen()), inspect.iscoroutine(gen()))
        # An async generator is NEITHER, which is the distinction that makes
        # three flags rather than one.
        a = agen()
        print(inspect.iscoroutine(a), inspect.isgenerator(a),
              inspect.isasyncgen(a))
        print(inspect.iscoroutinefunction(coro),
              inspect.iscoroutinefunction(gen))
        print(asyncio.run(c))
    """,
    "async_generators": """
        import asyncio
        log = []
        async def agen():
            try:
                for i in range(5):
                    yield i
            finally:
                log.append('cleanup')
        async def main():
            out = []
            async for v in agen():
                out.append(v)
                if v == 1:
                    break
            return out
        print(asyncio.run(main()))
        # The `finally` runs when the loop closes what the program abandoned,
        # not when the `async for` is left -- `break` leaves the generator
        # suspended inside its own `try`.
        print(log)
        async def nums(n):
            for i in range(n):
                await asyncio.sleep(0)
                yield i
        async def forms():
            lst = [v async for v in nums(3)]
            st = {v async for v in nums(3)}
            dct = {v: v * 2 async for v in nums(2)}
            filt = [v async for v in nums(3) if v]
            return lst, sorted(st), sorted(dct.items()), filt
        print(asyncio.run(forms()))
        print(type(nums(1)).__name__)
    """,
    "coroutines_and_gather": """
        import asyncio
        log = []
        async def task(name, rounds):
            for i in range(rounds):
                log.append((name, i))
                await asyncio.sleep(0)
            return name
        async def main():
            return await asyncio.gather(task('a', 3), task('b', 2))
        print(asyncio.run(main()))
        # THE INTERLEAVING IS THE POINT. A drained `await` -- one that ran the
        # inner coroutine to completion instead of suspending -- prints the
        # same results list and a different log, and passes every conformance
        # case either way. This line is what tells the two apart.
        print(log)
        async def val(v):
            await asyncio.sleep(0)
            return v
        async def seq():
            a = await val(1)
            b = await val(2)
            return a + b
        print(asyncio.run(seq()))
        never = []
        async def unused():
            never.append('ran')
        c = unused()
        print(never, type(c).__name__)
        asyncio.run(c)
        print(never)
        # SLEEP DURATION DECIDES WAKE ORDER. A `sleep` that ignored its delay
        # completed these round-robin -- 'slow' first, because it was stepped
        # first -- and no conformance case noticed, since every one of them
        # sleeps for 0 and only checks `gather`'s results, which are ordered
        # by argument either way.
        order = []
        async def timed(name, delay):
            await asyncio.sleep(delay)
            order.append(name)
            return name
        async def race():
            return await asyncio.gather(timed('slow', 0.05), timed('fast', 0.001))
        print(asyncio.run(race()))
        print(order)
    """,
    "int_passed_to_a_float_parameter": """
        def scale(x: float) -> float:
            return x * 2.0
        def half(x: float) -> float:
            return x / 2.0
        print(scale(42), scale(3.5))
        print(half(7), half(7.0))
        print(scale(0), scale(-3))
        total = 0.0
        for i in range(4):
            total = total + scale(i)
        print(total)
    """,
    "format_mini_language_numbers": """
        print(format(42, '08.2f'))
        print(format(1234, 'e'), format(1234, '.2e'))
        print(format(0.5, '%'), format(0.5, '.1%'))
        print(format(1234, 'g'), format(0.000012345, 'g'))
        print(format(3.14159, '.3'), format(3.14159, '10.3f') + '|')
        print(format(255, 'c') == chr(255))
        print(format(255, '#x'), format(255, '#o'), format(255, '#b'))
        print(format(1234567, ','), format(-42, '=+8d'))
        print(format('ab', '*^8') + '|', format('abcdef', '.3'))
    """,
    "typing_is_inert": """
        from typing import Final, final, override, Optional, LiteralString
        MAX: Final = 10
        print(MAX)
        @final
        class Sealed:
            pass
        class Still(Sealed):
            pass
        print(Still.__name__, getattr(Sealed, '__final__', False))
        class Base:
            def m(self):
                return 'base'
        class Sub(Base):
            @override
            def m(self):
                return 'sub'
        print(Sub().m(), Sub.m.__override__)
        def q(s: LiteralString) -> str:
            return s
        print(q('ok'))
        print(Optional.__class__.__name__ != '')
    """,
    "string_translation": """
        table = str.maketrans('ab', 'xy')
        print('aabb'.translate(table))
        print('abc'.translate(str.maketrans('', '', 'b')))
        print('hello'.translate({ord('l'): 'L'}))
        print('abcabc'.count('a', 1), 'abc'.count(''), 'aaaa'.count('aa'))
        print(chr(223).upper(), chr(223).casefold())
        print('a\\tb'.expandtabs(4))

        # THE SINGLE-ARGUMENT MAPPING FORM, which is a different operation
        # wearing the same name: nothing is paired off, a table that is
        # already a table is copied with its string keys turned into the code
        # points `translate` looks up.
        #
        # AND THE FORM IS CHOSEN BY THE ARGUMENTS THAT ARE NOT THERE.
        # CPython's `unicode_maketrans_impl` branches on `y == NULL` and only
        # then asks whether `x` is a dict, so `str.maketrans('ab')` -- one
        # argument, and a str -- is this form refusing a non-dict. Branching
        # on the first argument's kind instead is what made the compiled
        # runtimes answer the pairing form's complaint for it.
        def refused(label, fn):
            try:
                print(label, '->', fn())
            except Exception as e:
                print(label, '->', type(e).__name__ + ':', e)

        print(str.maketrans({'a': 'z'}), str.maketrans({97: 'z'}),
              str.maketrans({}))
        # A VALUE IS NOT CHECKED HERE, only a key: it may be None for a
        # deletion, an integer code point, a string LONGER than one character,
        # or something `translate` will refuse when it is reached.
        print(str.maketrans({'a': None, 'b': 98, 'c': 'zz', 'd': 1.0}))
        print('abcde'.translate(str.maketrans({'a': 'zz', 'b': None,
                                               99: 'Q', 'd': 101})))
        refused('long key', lambda: str.maketrans({'ab': 'z'}))
        refused('float key', lambda: str.maketrans({1.0: 'z'}))
        refused('bytes key', lambda: str.maketrans({b'a': 'z'}))
        refused('one str', lambda: str.maketrans('ab'))
        refused('one list', lambda: str.maketrans([]))
        refused('a dict and a second', lambda: str.maketrans({'a': 'z'}, 'b'))

        # THE TWO CHECKS ARE NOT THE SAME CHECK, and CPython's own source is
        # the only thing that says so: `unicode_maketrans_impl` admits the
        # argument with `PyDict_CheckExact` -- so a dict SUBCLASS is refused
        # exactly as a list is -- while the loop inside it admits a key with
        # `PyUnicode_Check`, which is NOT the exact check, so a str SUBCLASS
        # key is read for its one character like any other string.
        class MyDict(dict):
            pass

        class MyStr(str):
            pass

        refused('a dict subclass', lambda: str.maketrans(MyDict({'a': 'z'})))
        print(str.maketrans({MyStr('a'): 'z'}),
              str.maketrans({MyStr(chr(233)): 'z'}))
        refused('a long subclass key',
                lambda: str.maketrans({MyStr('ab'): 'z'}))
    """,
    "function_attributes": """
        def f():
            return 1
        f.tag = 'x'
        f.count = 3
        print(f.tag, f.count, f())
        print(getattr(f, 'tag'), getattr(f, 'nope', 'fallback'))
        print(hasattr(f, 'tag'), hasattr(f, 'nope'))
        def mark(fn):
            fn.marked = True
            return fn
        @mark
        def g():
            return 2
        print(g.marked, g(), g.__name__)
        class C:
            def m(self):
                return 3
        print(getattr(C, 'missing', 'none'))
    """,
    "hash_and_eq_contract": """
        class Point:
            def __init__(self, x):
                self.x = x
            def __eq__(self, o):
                return isinstance(o, Point) and self.x == o.x
            def __hash__(self):
                return hash(self.x)
        a, b = Point(1), Point(1)
        print(a == b, hash(a) == hash(b))
        print(len({a, b}))
        print({a: 'v'}[b])
        class OnlyEq:
            def __eq__(self, o):
                return True
        try:
            print(hash(OnlyEq()))
        except TypeError as e:
            print('TypeError:', e)
        class Plain:
            pass
        print(isinstance(hash(Plain()), int))
        # `__eq__` WITHOUT `__hash__` makes a class unhashable, and a CONTAINER
        # has to find that out too. `hash(x)` refused it already; `{x: 1}` did
        # not, and built a mapping whose key could never be looked up again --
        # a silent wrong answer where CPython raises.
        class OnlyEq2:
            def __eq__(self, o):
                return True
        print(OnlyEq2.__hash__)
        for make in ('dict', 'set'):
            try:
                if make == 'dict':
                    {OnlyEq2(): 1}
                else:
                    {OnlyEq2()}
            except TypeError as e:
                print(make, '->', e)
    """,
    "dunder_protocols": """
        class Odd:
            def __lt__(self, o):
                return 'lt'
            def __gt__(self, o):
                return 'gt'
            def __eq__(self, o):
                return 'eq'
        odd = Odd()
        print(odd < 1, odd > 1, odd == 1, 1 > odd, 1 < odd)
        class Unary:
            def __neg__(self):
                return 'neg'
            def __pos__(self):
                return 'pos'
            def __abs__(self):
                return 'abs'
        u = Unary()
        print(-u, +u, abs(u), -5, +5, abs(-5), ~5)
        class Two:
            def __index__(self):
                return 2
        print([10, 20, 30][Two()], 'abcd'[Two():], hex(Two()), bin(Two()))
        class Missing:
            def __init__(self):
                self.real = 1
            def __getattr__(self, name):
                return 'missing:' + name
        m = Missing()
        print(m.real, m.nope)
        class Private:
            def __init__(self):
                self.__hidden = 1
            def peek(self):
                return self.__hidden
            def __helper(self):
                return 'helped'
            def call_helper(self):
                return self.__helper()
        pv = Private()
        print(pv.peek(), pv._Private__hidden, hasattr(pv, '__hidden'))
        print(pv.call_helper(), sorted(vars(pv)))
        # THE RIGHT-HAND SIDE FIRST, which is the opposite of reading order.
        order = []
        def probe(n):
            order.append(n)
            return n
        class Sink:
            def __setitem__(self, k, v):
                order.append(('set', k, v))
        Sink()[probe('key')] = probe('value')
        print(order)
    """,
    "in_place_operators": """
        # `x += y` is NOT `x = x + y`: a list extends itself, so every other
        # name bound to it sees the change -- observable from another frame.
        def extend(xs):
            xs += [99]
        xs = [1]
        extend(xs)
        print(xs)
        def rebind(t):
            t += (99,)
        t = (1,)
        rebind(t)
        print(t)
        a = [1, 2]
        b = a
        a += [3]
        print(a, b, a is b)
        s1 = {1, 2}
        s2 = s1
        s1 |= {3}
        print(sorted(s1), sorted(s2), s1 is s2)
        s1 -= {1}
        print(sorted(s1))
        d = {'a': 1}
        e = d
        d |= {'b': 2}
        print(sorted(d.items()), d is e)
        n = 5
        n += 1
        st = 'a'
        st += 'b'
        print(n, st)
        row = [[0]]
        row[0] += [1]
        print(row)
        class Box:
            def __init__(self):
                self.v = [0]
        box = Box()
        box.v += [1]
        print(box.v)
    """,
    "finally_on_every_exit": """
        log = []
        for i in range(3):
            try:
                if i == 0:
                    continue
                if i == 2:
                    break
                log.append(('body', i))
            finally:
                log.append(('finally', i))
        print(log)
        def nested():
            out = []
            for i in range(3):
                try:
                    try:
                        if i == 1:
                            break
                        out.append(i)
                    finally:
                        out.append('inner')
                finally:
                    out.append('outer')
            return out
        print(nested())
        def wins():
            while True:
                try:
                    break
                finally:
                    return 'from-finally'
        print(wins())
        # The returned value is computed BEFORE the finally runs.
        def snapshot():
            n = 1
            try:
                return n
            finally:
                n = 99
        print(snapshot())
        marks = []
        def handler(mode):
            try:
                if mode == 'raise':
                    raise ValueError('x')
                return 'returned'
            except ValueError:
                return 'caught'
            finally:
                marks.append(mode)
        print(handler('ok'), handler('raise'), marks)
        try:
            try:
                raise ValueError('original')
            finally:
                raise KeyError('from-finally')
        except KeyError as e:
            print(type(e).__name__, type(e.__context__).__name__)
    """,
    "dicts_and_dunder_attributes": """
        a = {'x': 1, 'y': 2}
        b = {'y': 20, 'z': 30}
        print(sorted((a | b).items()), sorted((b | a).items()))
        c = dict(a)
        c |= b
        print(sorted(c.items()), sorted(a.items()))
        class Holder:
            shared = 1
            def __init__(self):
                self.own = 2
            def m(self):
                return 'm'
        h = Holder()
        print(sorted(vars(h)), vars(h)['own'])
        print('shared' in vars(h), 'shared' in vars(Holder))
        print(sorted(h.__dict__))
        h.__dict__['dynamic'] = 9
        print(h.dynamic)
        def plain(x):
            return x
        print(plain.__name__, plain.__qualname__, plain.__annotations__)
        # ONE HANDLE PER OBJECT: `is` has to answer about the object, not
        # about which access it came back through.
        print(h.m.__self__ is h)
    """,
    "format_mini_language": """
        print(f"{3.14159:.2f}", f"{42:5d}", f"{42:<5}|", f"{42:^7}|")
        print(f"{42:*>6}", f"{255:x}", f"{255:X}", f"{255:#x}", f"{255:b}")
        print(f"{255:#b}", f"{8:o}", f"{1234567:,}", f"{1234567:_}")
        print(f"{1234567.891:,.2f}", f"{-1.5:08.2f}", f"{1.5:+.1f}")
        print(f"{1.5: .1f}", f"{0.25:%}", f"{1234.5:e}", f"{1234.5:.3g}")
        print(f"{'hi':>6}|", f"{'hello':.3}", f"{'hi':-^8}", f"{'a'!r:>5}|")
        width = 8
        print(f"{3.14159:{width}.3f}|")
        print("{} {} {}".format(1, 2, 3), "{0} {2} {1}".format('a', 'b', 'c'))
        print("{name}: {v:.1f}".format(name='x', v=2.55))
        print("{{literal}} {}".format(9), "{:>{}}".format('q', 5) + "|")
        print(format(3.14159, '.3f'), format(42, 'b'), format('hi'))
        print("{!r}".format('a'), "{0!r} {0}".format('b'))
        class Point:
            def __init__(self, x):
                self.x = x
            def __format__(self, spec):
                return 'P<' + spec + '>' + str(self.x)
            def __str__(self):
                return 'P' + str(self.x)
        p = Point(3)
        print(f"{p}", f"{p:.2f}", format(p, 'wide'))
    """,
    "iterator_protocol": """
        class Count:
            def __init__(self, n):
                self.n = n
                self.i = 0
            def __iter__(self):
                return self
            def __next__(self):
                if self.i >= self.n:
                    raise StopIteration
                self.i += 1
                return self.i
        print(list(Count(3)))
        it = Count(1)
        print(next(it))
        try:
            next(it)
        except StopIteration:
            print('StopIteration')
        for v in Count(2):
            print('v', v)
        print([x * 2 for x in Count(3)])
        class Seq:
            def __len__(self):
                return 3
            def __getitem__(self, i):
                if i >= 3:
                    raise IndexError
                return i * 10
        s = Seq()
        print(len(s), s[1], list(s), 20 in s, 5 in s)
        class Bare:
            def __getitem__(self, i):
                if i >= 2:
                    raise IndexError
                return i
        print(list(Bare()), 1 in Bare())
    """,
    "exceptions_leave_functions": """
        # An exception raised inside a call has to reach the CALLER's handler.
        # It leaves a compiled function the same way it leaves an `apy_add`:
        # with the flag set and a null result.
        def inner():
            raise ValueError('deep')
        def middle():
            inner()
            return 'unreachable'
        def guarded():
            try:
                middle()
            except ValueError as e:
                return 'caught ' + str(e)
            return 'no'
        print(guarded())
        try:
            middle()
        except ValueError as e:
            print('outer', e)
        class Box:
            def check(self, n):
                if n < 0:
                    raise ValueError('negative')
                return n
        b = Box()
        try:
            b.check(-1)
        except ValueError as e:
            print('method', e)
        print(b.check(2))
        def finallys():
            try:
                inner()
            finally:
                print('finally ran')
        try:
            finallys()
        except ValueError as e:
            print('after finally', e)
    """,
    "exception_chaining": """
        try:
            try:
                raise KeyError('inner')
            except KeyError:
                raise ValueError('outer')
        except ValueError as e:
            print(type(e).__name__, e.args,
                  type(e.__context__).__name__, e.__cause__)
        try:
            try:
                raise KeyError('k')
            except KeyError as k:
                raise ValueError('v') from k
        except ValueError as e:
            print(type(e.__cause__).__name__, type(e.__context__).__name__,
                  e.__suppress_context__)
        try:
            try:
                raise KeyError('k')
            except KeyError:
                raise ValueError('v') from None
        except ValueError as e:
            print(e.__cause__, e.__context__, e.__suppress_context__)
        try:
            raise ValueError('n')
        except ValueError as e:
            e.add_note('extra')
            print(e.__notes__, e.__traceback__ is None)
    """,
    "mutating_methods": """
        xs = [3, 1, 2]
        xs.insert(0, 9)
        xs.insert(99, 7)
        xs.insert(-99, 8)
        print(xs)
        ys = [3, 1, 2]
        ys.sort()
        print(ys)
        ys.sort(reverse=True)
        print(ys)
        zs = ['bb', 'a', 'ccc']
        zs.sort(key=len)
        print(zs)
        zs.reverse()
        copied = zs.copy()
        copied.clear()
        print(zs, copied)
        zs.extend(['x', 'y'])
        print(zs)
        d = {'a': 1}
        d.update({'b': 2})
        print(sorted(d.items()), d.setdefault('a', 9), d.setdefault('c', 3))
        print(sorted(d.items()))
        print('hi'.encode(), b'hi'.decode())
        print((255).bit_length(), (5).is_integer(), (2.0).is_integer())
        print((2.5).is_integer(), complex(1, 2).conjugate(), (5).conjugate())
    """,
    "star_kwargs_and_decorators": """
        def f(a, b=2, **kw):
            return (a, b, sorted(kw.items()))
        g = f
        print(g(1), g(1, 3, x=9, y=8), g(1, x=1))
        d = {'p': 1, 'q': 2}
        print(g(5, **d), g(5, b=7, **d))
        def h(*rest, **kw):
            return (rest, sorted(kw.items()))
        hh = h
        print(hh(1, 2, 3, k=1), hh())
        class Opts:
            def __init__(self, n, **opts):
                self.n = n
                self.opts = sorted(opts.items())
        made = Opts(1, colour='red', size=2)
        print(made.n, made.opts)

        def twice(fn):
            def wrapper(x):
                return fn(fn(x))
            return wrapper
        def shout(fn):
            def wrapper(x):
                return str(fn(x)) + '!'
            return wrapper
        @shout
        @twice
        def inc(n):
            return n + 1
        print(inc(1))
        def tag(label):
            def deco(fn):
                def wrapper(*a):
                    return (label, fn(*a))
                return wrapper
            return deco
        @tag('hi')
        def doubled(x):
            return x * 2
        print(doubled(3))
        def marked(cls):
            cls.mark = True
            return cls
        @marked
        class Plain:
            def m(self):
                return 'm'
        print(Plain.mark, Plain().m())
    """,
    "builtin_call_shapes": """
        # The builtins that are two functions wearing one name: the argument
        # count picks, so both shapes have to be checked.
        print(round(2.675, 2), round(2.5), round(2.5, 0), round(2.345, 2))
        print(round(1234.5678, -2), round(1234, -2), round(-2.5), round(0.5))
        print(int('ff', 16), int('101', 2), int('0x1f', 16), int('7'))
        print(sum([1, 2, 3], 10), sum([], 0.0), sum([1, 2]))
        print(min(3, 1, 2), max(3, 1, 2), min([3, 1, 2]), max([3, 1, 2]))
        print(min([], default=9), max([1, 2], default=9))
        print(list(zip()), list(zip([1, 2])), list(zip([1, 2], [3, 4], [5, 6])))
        class Base:
            pass
        class Derived(Base):
            pass
        print(issubclass(Derived, Base), issubclass(Base, Derived))
        class Holder:
            def __init__(self):
                self.x = 1
                self.y = 2
        h = Holder()
        print(sorted(vars(h).items()))
        setattr(h, 'z', 3)
        print(h.z, hasattr(h, 'z'))
        delattr(h, 'z')
        print(hasattr(h, 'z'))
        seen = [0]
        def tick():
            seen[0] += 1
            return seen[0] if seen[0] < 4 else 0
        print(list(iter(tick, 0)))
    """,
    "keyword_arguments": """
        def f(a, b=2, c=3):
            return (a, b, c)
        g = f
        print(f(1, c=30), g(1), g(1, c=30), g(1, b=20, c=30), g(c=9, a=8))
        class Point:
            def __init__(self, x, y=0, label='p'):
                self.x = x
                self.y = y
                self.label = label
            def moved(self, dx=0, dy=0):
                return (self.x + dx, self.y + dy, self.label)
        print(Point(1, label='q').label, Point(1, 2, 'r').moved(dx=1, dy=1))
        print(Point(1).moved(dy=5))
        # Through the ALIAS, so the callee is a value and the mismatch is
        # reported where CPython reports it -- at run time. Calling `f` by
        # name is checked at compile time instead, which is a difference in
        # WHEN and not in what.
        try:
            g(1, z=5)
        except TypeError as e:
            print('TypeError', e)
        try:
            g(1, a=5)
        except TypeError as e:
            print('TypeError', e)
        try:
            g(b=1)
        except TypeError as e:
            print('TypeError', e)
    """,
    "arithmetic_and_kinds": """
        print(1 + 2, 3.5, True, False, None)
        print(7 // 2, -7 // 2, 7 % 3, -7 % 3)
        print(1 / 4, 2 ** 10, -5, +5, ~5)
        print(True + 1, 1 == True, 1.0 == 1)
        print(1 and 2, 0 or 'x', not 0)
        print(1 < 2 < 3, 1 < 2 > 3)
        print(type(1).__name__, type(1.0).__name__)
        print(type(True).__name__, type(None).__name__, type('a').__name__)
    """,
    "strings": """
        s = 'abc'
        print(s + 'de', s * 2, 3 * s, len(s))
        print(s[0], s[-1], s[1:], s[:2], s[::-1])
        print('b' in s, 'z' in s)
        print(repr('a"b'), repr("it's"))
        print('  pad  '.strip(), 'A,B'.split(','), '-'.join(['x', 'y']))
        print('abc'.upper(), 'ABC'.lower(), 'abc'.replace('b', 'X'))
        print('abc'.find('b'), 'abc'.startswith('ab'), '7'.zfill(3))
    """,
    "sequences": """
        xs = [1, 2, 3]
        print(xs, xs[0], xs[-1], len(xs))
        xs.append(4)
        xs[0] = 'a'
        print(xs, xs.pop(), xs.index(2), xs.count(2))
        t = (1, 'two', 3.5)
        print(t, t[1], len(t), (7,), ())
        print([1, 2] + [3], (1,) + (2,), [0] * 3)
        print(1 in xs, 9 in xs, [1] in xs)
        print([1, 2] == [1, 2], [1, 2] == (1, 2), [1, 2] < [1, 3])
        print(sorted([3, 1, 2]), min([3, 1]), max([3, 1]), sum([1, 2]))
        print(list(reversed([1, 2])), list(enumerate('ab')))
        print(list(zip([1, 2], 'ab')), list(range(3)))
    """,
    "dicts_and_sets": """
        d = {'a': 1, 'b': 2}
        d['c'] = 3
        d['a'] = 9
        print(d, d['a'], len(d), 'a' in d, 'z' in d)
        print(list(d.keys()), list(d.values()), list(d.items()))
        print(d.get('a'), d.get('z'), d.get('z', 0))
        print({}, {1: [2], (3,): 'x'}, {'a': 1} == {'a': 1})
        xs = {1, 2, 3}
        xs.add(4)
        print(sorted(xs), len(xs), 2 in xs, type(xs).__name__)
        print(sorted({1, 2} | {2, 3}), sorted({1, 2} & {2, 3}))
        print(set(), frozenset([1, 2]) == frozenset([2, 1]))
    """,
    "control_flow": """
        total = 0
        for i in range(5):
            if i == 3:
                continue
            total = total + i
        print(total)
        n = 0
        while n < 3:
            n += 1
        print(n)
        for v in [10, 20]:
            print(v)
        else:
            print('for-else')
        for v in [10, 20]:
            break
        else:
            print('not reached')
        print('yes' if n else 'no')
        if n > 99:
            print('big')
        else:
            print('small')
    """,
    "exceptions": """
        try:
            raise ValueError('boom')
        except ValueError as e:
            print('caught', e, type(e).__name__)
        try:
            print(1 / 0)
        except ZeroDivisionError as e:
            print('zde', e)
        try:
            raise KeyError('k')
        except LookupError:
            print('base class caught it')
        try:
            print(1 + 'a')
        except TypeError as e:
            print('te', e)
        try:
            print('fine')
        except ValueError:
            print('not reached')
        else:
            print('else ran')
        finally:
            print('finally ran')
        try:
            [1][5]
        except IndexError as e:
            print('idx', e)
        print(repr(ValueError('v')))
    """,
    "functions_and_globals": """
        top = 5
        items = [1, 2]
        def read():
            return top, items
        print(read())
        count = 0
        def bump():
            global count
            count = count + 1
        bump()
        bump()
        print(count)
        def shadow():
            top = 99
            return top
        print(shadow(), top)
        def greet(name, greeting='hi'):
            return greeting + ' ' + name
        print(greet('a'), greet('a', 'yo'), greet(greeting='hey', name='b'))
        def collect(xs=[]):
            xs.append(1)
            return xs
        print(collect(), collect())
        def wide(a, *rest):
            return a, rest
        print(wide(1), wide(1, 2, 3))
    """,
    "comprehensions_and_unpacking": """
        print([x * 2 for x in [1, 2, 3]])
        print([x for x in range(6) if x % 2 == 0])
        print([(a, b) for a in [1, 2] for b in 'xy'])
        print({k: v for k, v in [('a', 1), ('b', 2)]})
        print(sorted({x * 2 for x in [1, 2, 2]}))
        print(sum(x for x in [1, 2, 3]))
        a, b = (1, 2)
        print(a, b)
        for i, ch in enumerate('hi'):
            print(i, ch)
        for k, v in {'p': 1}.items():
            print(k, v)
    """,
    "bytes": """
        b = b'ab'
        print(b, len(b), b[0], b[-1], b[1:], b + b'cd', b * 2)
        print(b == b'ab', b == 'ab', b < b'ac', b'a' in b, 97 in b)
        print(type(b).__name__, b'', b'q' * 0)
        print(b'abc'[::-1], b'abcdef'[1:5:2])
        print({b'k': 1}[b'k'], b'x' in {b'x': 1})
        print(repr(b'a\\tb\\nc'), repr(b"it's"))
        print(sorted([b'c', b'a', b'b']))
    """,
    "del_and_walrus": """
        d = {'a': 1, 'b': 2}
        del d['a']
        xs = [1, 2, 3]
        del xs[1]
        del xs[-1]
        print(d, xs)
        top = 5
        del top
        try:
            print(top)
        except NameError as e:
            print('NameError', e)
        try:
            del d['zz']
        except KeyError as e:
            print('KeyError', e)
        n = 0
        while (n := n + 1) < 4:
            print('walrus', n)
        print([y for x in [1, 2, 3] if (y := x * 2) > 2])
    """,
    "star_args": """
        xs = [1, 2, 3]
        def take(*a):
            return a
        print(take(*xs))
        def two(a, b):
            return a - b
        print(two(*[10, 4]), two(1, *[9]))
        def mixed(a, *rest):
            return a, rest
        print(mixed(*xs))
        print(take(*'ab'), take(*(7, 8)))
    """,
    "big_integers": """
        big = 9223372036854775808
        print(big, big - 1, -big, big * 2)
        print(2 ** 100, 10 ** 30)
        n = 1
        for i in range(1, 26):
            n = n * i
        print(n)
        print(big // 2, big % 7, big > 5, big == big)
        back = (big + 1) - big
        print(back, back == 1, type(back).__name__)
        d = {back: 'x'}
        print(d[1], 1 in d, sorted([big, 1, -big]))
        print(str(big), len(str(2 ** 100)))
        print(int('123456789012345678901234567890') + 1)
    """,
    "exception_payloads": """
        class E(Exception):
            pass
        def move(v):
            try:
                raise E(v)
            except E as e:
                return e.args[0]
        for original in [42, 'abc', [1, 2], None, 3.5, b'ab', 9223372036854775808]:
            moved = move(original)
            print(moved, moved == original, type(moved).__name__)
        try:
            raise ValueError()
        except ValueError as e:
            print(repr(e), e.args, repr(str(e)))
        try:
            raise ValueError(None)
        except ValueError as e:
            print(repr(e), e.args, str(e))
        try:
            assert False
        except AssertionError as e:
            print(repr(e), e.args)
    """,
    "complex_numbers": """
        print(1j, 3+4j, (1+2j))
        print(complex(0,2), complex(1,2), complex(1,-2), complex(-0.0,2))
        print(complex(0,-0.0), complex(0,0), complex(1.5,0), complex(), complex(5))
        print((1+2j)+(3+4j), (1+2j)*(3+4j), (1+2j)-(3+4j), (1+2j)/(3+4j))
        print((1+2j)==(1+2j), (1+2j)==1, complex(1,0)==1, 1j=='a')
        print((1+2j).real, (1+2j).imag, type(1j).__name__, bool(0j), bool(1j))
        print(1j + 2, 2 + 1j, 1j * 2.0, -(1+2j), +(1+2j))
        try:
            print(1j < 2j)
        except TypeError as e:
            print('TypeError', e)
        try:
            print(1j / 0)
        except ZeroDivisionError:
            print('ZeroDivisionError')
        print([1j, 2+3j], {1j: 'a'}[1j], 1j in [1j])
    """,
    "builtins_as_values": """
        print(sorted([3, 1, 2], key=str))
        print(sorted(['bb', 'a', 'ccc'], key=len))
        f = repr
        print(f('x'), f(1))
        g = abs
        print(g(-3))
        def apply(fn, v):
            return fn(v)
        print(apply(len, 'abcd'), apply(repr, 'q'), apply(str, 9))
        print(apply(hex, 255), apply(ord, 'A'), apply(chr, 66))
    """,
    "lambdas_and_keys": """
        f = lambda x: x * 2
        g = lambda a, b: a + b
        print(f(3), g(1, 2))
        n = 10
        h = lambda: n
        print(h(), (lambda x=5: x)(), (lambda x=5: x)(9))
        def make(k):
            return lambda v: v * k
        print(make(3)(4), make(10)(4))
        print(sorted([3, 1, 2], key=lambda v: -v))
        print(sorted([3, 1, 2], reverse=True))
        print(sorted(['bb', 'a', 'ccc'], key=len, reverse=True))
        print(min([3, 1, 2], key=lambda v: -v), max([3, 1, 2], key=lambda v: -v))
        print(sorted([(1, 'b'), (1, 'a'), (0, 'c')], key=lambda p: p[0]))
        adders = [lambda v, k=i: v + k for i in range(3)]
        print([a(10) for a in adders])
    """,
    "iterators": """
        it = iter([1, 2, 3])
        print(next(it), next(it), next(it))
        try:
            next(it)
        except StopIteration:
            print('StopIteration')
        print(next(iter([]), 'dflt'))
        s = iter('ab')
        print(next(s), next(s))
        it2 = iter([7, 8])
        print(list(it2), list(it2))
        it3 = iter([1, 2, 3])
        next(it3)
        for v in it3:
            print(v)
        print(sum(iter([1, 2, 3])), sorted(iter([3, 1])))
        print(dict(), dict([('a', 1), ('b', 2)]))
        print(bytes(), bytes([1, 255, 16]), bytes(b'ab'))
    """,
    "star_displays_and_augmented": """
        xs = [1, 2]
        print([*xs, 3], [0, *xs], [*xs, *xs])
        print((*xs, 3), sorted({*xs, 3}))
        print([*'ab'], [*{'k': 1}])
        log = []
        def idx():
            log.append('idx')
            return 0
        ys = [10]
        ys[idx()] += 5
        print(ys, log)
        d = {'a': 1}
        d['a'] += 2
        print(d)
        class P:
            def __init__(self):
                self.n = 1
        p = P()
        p.n += 4
        print(p.n)
        zs = [[1], [2]]
        zs[0] += [9]
        print(zs)
    """,
    "an_inherited_comparison_beats_a_mirror_the_class_wrote": """
        # `OnlyGt([1]) < OnlyGt([1, 2])` ANSWERED `'GT'`. `type(a).__lt__` is
        # the one `list` gives the subclass, and in CPython an INHERITED slot
        # wins outright over a `__gt__` the class wrote -- the reflected
        # method is never consulted, and the answer is True. All three
        # arrangements tried the written direct dunder, then the written
        # MIRROR, and only then read the builtin the class extends, so a
        # class that mentioned `__gt__` once answered every ordering with it.
        #
        # THE BUILTIN BELONGS IN THE MIDDLE, which is the whole fix: written
        # direct, then the builtin (the inherited direct), then written
        # mirror. `apy_order_mirror_first_of` and `apy_written_dunder_of`
        # exist to make that order expressible -- `apy_binary_dunder` runs
        # both written halves back to back and there was nowhere to put the
        # builtin between them.
        #
        # CPYTHON HAS ONE REORDERING RULE and it had to come with the fix: a
        # right operand whose type is a PROPER SUBCLASS of the left's and
        # which overrides the mirror goes first, so `[1] < OnlyGt([1, 2])` is
        # still `'GT'`. Three cases that were already right depend on it.
        #
        # AND `sorted` HAD TO MOVE TOO. It compares with the very `<` the
        # operator spells, through `apy_order_rich_of` rather than
        # `apy_cmp` -- so before this, `sorted` over these ordered by the
        # written `__gt__` while `a < b` on the same two ordered as lists.
        class OnlyGt(list):
            def __gt__(self, other):
                return "GT"

        class OnlyLt(list):
            def __lt__(self, other):
                return "LT"

        class Both(list):
            def __lt__(self, other):
                return "B.LT"

            def __gt__(self, other):
                return "B.GT"

        class Plain(list):
            pass

        class T(tuple):
            def __gt__(self, other):
                return "T.GT"

        def show(label, f):
            try:
                print(label, repr(f()))
            except TypeError as e:
                print(label, "TypeError:", e)

        a1 = OnlyGt([1])
        a2 = OnlyGt([1, 2])
        # THE INHERITED `__lt__` ANSWERS and the written `__gt__` is not asked.
        show("inherited lt", lambda: a1 < a2)
        # THE WRITTEN ONE STILL ANSWERS ITS OWN OPERATOR.
        show("written gt", lambda: a1 > a2)
        # THE SUBCLASS GOES FIRST when it is one and overrides the mirror.
        show("subclass first", lambda: [1] < a2)
        # AND NOT WHEN IT OVERRIDES THE OTHER NAME.
        show("no reorder", lambda: [1] > a2)
        show("against a plain list", lambda: a1 < [1, 2])
        show("against an int", lambda: 5 < a1)
        # THE MIRROR SIDE OF THE SAME RULE.
        b1 = OnlyLt([1])
        b2 = OnlyLt([1, 2])
        show("written lt", lambda: b1 < b2)
        show("inherited gt", lambda: b1 > b2)
        show("reflected lt", lambda: [1] > b2)
        # A CLASS THAT WROTE BOTH is never read as its builtin.
        show("both lt", lambda: Both([1]) < Both([1, 2]))
        show("both gt", lambda: Both([1]) > Both([1, 2]))
        # SIBLINGS ARE NOT SUBCLASSES OF EACH OTHER.
        show("plain pair", lambda: Plain([1]) < Plain([1, 2]))
        show("plain vs onlygt", lambda: Plain([1]) < a2)
        show("onlygt vs plain", lambda: a1 < Plain([1, 2]))
        # THE SAME RULE ON A TUPLE, because the held value is any builtin.
        show("tuple lt", lambda: T((1,)) < T((2,)))
        show("tuple gt", lambda: T((1,)) > T((2,)))
        show("tuple eq", lambda: T((1,)) == T((1,)))
        # AND THE CONSUMERS, which reach it through `apy_order_rich_of`.
        xs = [OnlyGt([1, 2]), OnlyGt([1]), OnlyGt([1, 2, 3])]
        print("sorted", [len(v) for v in sorted(xs)])
        print("extremes", len(min(xs)), len(max(xs)))
        ts = [T((2,)), T((1,))]
        print("sorted tuples", [t[0] for t in sorted(ts)])

        # AND THE GUARD THE REORDER NEEDED, which the first attempt at it got
        # wrong: reading the builtin only counts when BOTH sides come out of
        # it as builtins. `list.__lt__` answers NotImplemented for an operand
        # it knows nothing about, and that is when CPython goes on to the
        # mirror -- giving it the operands THE PROGRAM WROTE. Comparing the
        # half-unwrapped pair instead handed `Watcher.__gt__` a bare list
        # where it is owed the Sub, and then coerced the string it answered
        # into True.
        class Watcher:
            def __gt__(self, other):
                return "gt got " + type(other).__name__

            def __radd__(self, other):
                return "radd got " + type(other).__name__

        show("held pair must be whole", lambda: Plain([1]) < Watcher())
        show("and from a bare builtin", lambda: [1] < Watcher())
        show("the tuple side too", lambda: T((1,)) < Watcher())
    """,
    "arithmetic_asks_the_builtin_a_class_extends": """
        # THE OTHER OPERATOR FAMILY, with the same three steps comparison
        # got: the dunder a wrote, the builtin a extends, the mirror b wrote.
        # Arithmetic had BOTH halves of that wrong and one gap of its own.
        #
        # THE GAP: a BUILTIN on the left and an INSTANCE on the right never
        # reached the dispatch at all. `apy_add`'s concatenation refusals
        # return before `apy_binop_fallback` at the bottom is ever read, so
        # `[9] + Sub([2])` on a `class Sub(list)` reported `can only
        # concatenate list (not "Sub") to list` -- about an object that IS a
        # list. `apy_mul` had the same shape. There WAS a guard for str + an
        # instance, added for `StrEnum`, but it asked only the written
        # dunders, so a subclass that writes nothing still fell through.
        #
        # THE ORDER: `apy_binop_fallback` ran both written halves and only
        # then read the builtin, and the interpreter read the builtin FIRST,
        # before either. Two different wrong answers for one rule.
        #
        # AND THE GUARD: reading the builtin only counts when BOTH sides come
        # out of it as builtins, because that is when CPython's inherited
        # slot answers NotImplemented and hands the mirror the operands the
        # program wrote. `Sub([1]) + Mirror()` owes `Mirror.__radd__` the Sub.
        #
        # WITH ONE EXCEPTION, which is `str % anything`: `str.__mod__` takes
        # whatever it is handed and formats it rather than declining by kind,
        # so it must run even against an instance.
        class Sub(list):
            pass

        class SubT(tuple):
            pass

        class SubS(str):
            pass

        class SubD(dict):
            pass

        class OnlyRadd(list):
            def __radd__(self, other):
                return "RADD"

        class OnlyAdd(list):
            def __add__(self, other):
                return "ADD"

        class Mirror:
            def __radd__(self, other):
                return "RADD " + type(other).__name__

        def show(label, f):
            try:
                print(label, repr(f()))
            except TypeError as e:
                print(label, "TypeError:", e)

        # A BUILTIN ON THE LEFT reaches the instance on the right.
        show("list + sub", lambda: [9] + Sub([2]))
        show("tuple + sub", lambda: (9,) + SubT((2,)))
        show("str + sub", lambda: "b" + SubS("a"))
        show("dict | sub", lambda: {"b": 2} | SubD(a=1))
        # AND THE OTHER DIRECTION, which already worked and must stay.
        show("sub + list", lambda: Sub([1]) + [9])
        show("sub + sub", lambda: Sub([1]) + Sub([2]))
        show("sub * 2", lambda: Sub([1]) * 2)
        show("2 * sub", lambda: 2 * Sub([1]))
        show("subs * 2", lambda: SubS("a") * 2)
        # THE INHERITED SLOT BEATS A WRITTEN MIRROR, as in comparison.
        show("inherited add", lambda: OnlyRadd([1]) + OnlyRadd([2]))
        show("inherited vs list", lambda: OnlyRadd([1]) + [9])
        # AND THE SUBCLASS STILL GOES FIRST when it overrides the mirror.
        show("subclass first", lambda: [9] + OnlyRadd([2]))
        show("no reorder", lambda: [9] + OnlyAdd([2]))
        # THE HELD PAIR MUST BE WHOLE, or the mirror gets the wrong operand.
        show("mirror keeps the sub", lambda: Sub([1]) + Mirror())
        show("mirror from a builtin", lambda: [1] + Mirror())
        # A BUILTIN THAT RAN AND REFUSED IS NOT A DECLINE, and it words the
        # refusal as a failed CONCATENATION rather than as an operand pair.
        # The interpreter said the generic text and deferred to the C in a
        # comment -- back when the C said it too; `apy_add` has worded it
        # from the left operand's kind for some time and this was the half
        # that did not follow. Two plain builtins show it without any class
        # at all, which is why they are here.
        show("refused by the builtin", lambda: Sub([1]) + 5)
        show("plain list and int", lambda: [1] + 5)
        show("plain list and tuple", lambda: [1] + (2,))
        show("plain tuple and list", lambda: (1,) + [2])
        show("plain str and int", lambda: "a" + 1)
        show("multiply by a list", lambda: [1] * [2])
        # AND `%` ON A str RUNS AGAINST ANYTHING.
        show("percent", lambda: SubS("%d") % 5)
    """,
    "percent_refuses_an_argument_its_conversion_cannot_take": """
        # `"%d" % "a"` RAISED A ValueError. `%` is implemented by translating
        # into the format MINI-LANGUAGE and handing the argument to
        # `format()` -- which is what keeps `%05.2f` and `{:05.2f}` from
        # being written twice -- and what that complains about is an unknown
        # FORMAT CODE. What `%` complains about is the ARGUMENT. So the
        # exception TYPE was wrong, and a program catching TypeError around a
        # `%` missed it entirely.
        #
        # THREE WORDINGS, which are CPython's own rather than one
        # generalised: `%d` takes any real number, `%x` takes an INTEGER and
        # refuses a float that `%d` accepts, and the floating conversions do
        # not name the conversion at all.
        #
        # AND `%c` WAS ANSWERING. `"%c" % "ab"` gave back `'ab'` and
        # `"%c" % obj` its repr -- wrong answers rather than errors, on the
        # compiled paths, which is the failure that does not announce itself.
        #
        # A CLASS REACHES A NUMERIC CONVERSION THROUGH ITS NUMBER, which the
        # interpreter did and the C did not: `%d` asks `__index__` and `%f`
        # asks `__float__`, so a class defining exactly the method for it was
        # reported as an unknown format code.
        class Obj:
            def __repr__(self):
                return "<obj>"

            def __str__(self):
                return "obj!"

        class HasIndex:
            def __index__(self):
                return 42

        class HasFloat:
            def __float__(self):
                return 2.5

        def show(label, f):
            try:
                print(label, repr(f()))
            except TypeError as e:
                print(label, "TypeError:", e)
            except ValueError as e:
                print(label, "ValueError:", e)

        # A REAL NUMBER IS REQUIRED, and a float is one -- it truncates.
        show("d str", lambda: "%d" % "a")
        show("d obj", lambda: "%d" % Obj())
        show("i str", lambda: "%i" % "a")
        show("u str", lambda: "%u" % "a")
        show("d float", lambda: "%d" % 1.5)
        show("d negative float", lambda: "%d" % -1.9)
        show("d bool", lambda: "%d" % True)
        # AN INTEGER IS REQUIRED, and a float is NOT one.
        show("x str", lambda: "%x" % "a")
        show("x float", lambda: "%x" % 1.5)
        show("o str", lambda: "%o" % "a")
        show("X obj", lambda: "%X" % Obj())
        show("x int", lambda: "%x" % 255)
        # THE FLOATING ONES NAME ONLY WHAT THEY WANTED.
        show("f str", lambda: "%f" % "a")
        show("e obj", lambda: "%e" % Obj())
        show("g str", lambda: "%g" % "a")
        show("f float", lambda: "%.2f" % 1.5)
        # `%c` TAKES ONE CHARACTER OR AN INT.
        show("c long str", lambda: "%c" % "ab")
        show("c obj", lambda: "%c" % Obj())
        show("c int", lambda: "%c" % 65)
        show("c char", lambda: "%c" % "z")
        # A CLASS THROUGH ITS NUMBER.
        show("d index", lambda: "%d" % HasIndex())
        show("x index", lambda: "%x" % HasIndex())
        show("f float dunder", lambda: "%f" % HasFloat())
        show("d float dunder", lambda: "%d" % HasFloat())
        # AND THE SHAPES THAT MUST NOT HAVE MOVED.
        show("s", lambda: "%s|%s|%s" % (1, "a", Obj()))
        show("r", lambda: "%r|%r" % ("a", Obj()))
        show("pad", lambda: "%5d|%-5d|%05d|%+d" % (42, 42, 42, 42))
        show("bytes", lambda: b"%s|%d" % (b"ab", 5))
        show("map", lambda: "%(a)s-%(b)d" % {"a": "x", "b": 2})
        show("too few", lambda: "%s %s" % ("one",))
        show("too many", lambda: "%s" % ("one", "two"))
    """,
    "join_words_a_broken_iter_as_join_does": """
        # `",".join(obj)` WHERE `obj.__iter__` ANSWERS A NON-ITERATOR
        # reported `iter() returned non-iterator of type 'int'` -- the
        # FUNNEL's message, naming the kind. CPython reaches iteration
        # through `PySequence_Fast(seq, "can only join an iterable")`, which
        # REPLACES every TypeError that comes out of GETTING the iterator
        # with that one sentence and names no kind.
        #
        # ERRORS FROM WALKING STILL PROPAGATE, which is why this is not done
        # by clearing the funnel's flag afterwards: the funnel gets the
        # iterator AND drains it, so a cleared TypeError would also swallow
        # one raised inside a user's `__next__`. `join` asks the question
        # itself instead, calling `__iter__` ONCE and handing what it
        # answered on -- a class whose `__iter__` has a side effect must not
        # have it twice.
        log = []

        class BadIter:
            def __iter__(self):
                return 42

        class NoNext:
            def __iter__(self):
                return self

        class RaisesValue:
            def __iter__(self):
                raise ValueError("boom")

        class NextRaisesType:
            def __init__(self):
                self.n = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.n = self.n + 1
                if self.n == 1:
                    return "a"
                raise TypeError("from next")

        class GenRaisesType:
            def __iter__(self):
                def gen():
                    yield "a"
                    raise TypeError("from gen")
                return gen()

        class Counts:
            def __iter__(self):
                log.append("iter")
                return iter(["a", "b"])

        class Seq:
            def __getitem__(self, i):
                if i < 3:
                    return "s" + str(i)
                raise IndexError

        class Meta(type):
            def __iter__(cls):
                return iter(["m1", "m2"])

        class Members(metaclass=Meta):
            pass

        class SubList(list):
            pass

        def show(label, f):
            try:
                print(label, repr(f()))
            except TypeError as e:
                print(label, "TypeError:", e)
            except ValueError as e:
                print(label, "ValueError:", e)

        # GETTING THE ITERATOR FAILED -- join's own wording, naming no kind.
        show("non-iterator", lambda: ",".join(BadIter()))
        show("no next", lambda: ",".join(NoNext()))
        # NOT A TypeError, so it is not replaced.
        show("raises value", lambda: ",".join(RaisesValue()))
        # WALKING FAILED, and those propagate whatever they are.
        show("next raises", lambda: ",".join(NextRaisesType()))
        show("gen raises", lambda: ",".join(GenRaisesType()))
        # `__iter__` ONCE, not twice.
        show("counts", lambda: ",".join(Counts()))
        print("iter calls", len(log))
        # AND EVERY WAY OF BEING JOINABLE STILL IS.
        show("getitem", lambda: ",".join(Seq()))
        show("sublist", lambda: ",".join(SubList(["x", "y"])))
        show("metaclass", lambda: ",".join(Members))
        show("genexp", lambda: ",".join(str(x) for x in range(3)))
        show("map", lambda: ",".join(map(str, range(3))))
        show("list", lambda: ",".join(["a", "b"]))
        show("tuple", lambda: ",".join(("a", "b")))
        show("dict", lambda: ",".join({"a": 1, "b": 2}))
        show("str", lambda: ",".join("abc"))
        show("empty", lambda: ",".join([]))
        # AND THE REFUSALS THAT WERE ALREADY RIGHT.
        show("int", lambda: ",".join(5))
        show("ints inside", lambda: ",".join([1, 2]))
    """,
    "what_iter_answers_must_be_an_iterator": """
        # `__iter__` MUST RETURN AN ITERATOR -- an object with `__next__`. A
        # list is not one, and neither is a dict, a set or a tuple: `iter()`
        # makes one FROM each of them, which is a different thing. Every
        # funnel here accepted all four, so `[x for x in GivesList()]`
        # answered `[1, 2]` where CPython refuses -- a broken class silently
        # working, and the author never told which method to fix.
        #
        # AND THE LAZY PATH REFUSED IT IN THE WRONG WORDS. `iter(obj)` handed
        # back whatever `__iter__` gave, so the complaint came later from
        # `next`: `'list' object is not an iterator`, which is what `next`
        # says about something that never was one rather than what `iter`
        # says about what `__iter__` gave it.
        #
        # ONE PREDICATE PER ARRANGEMENT now, which is what lets `iter`, the
        # eager funnel and `str.join` agree: `apy_is_iterator_of` in the C
        # and in the ported subset, `_is_iterator` in the host.
        class GivesList:
            def __iter__(self):
                return [1, 2]

        class GivesDict:
            def __iter__(self):
                return {"k": 1}

        class GivesSet:
            def __iter__(self):
                return {"a"}

        class GivesTuple:
            def __iter__(self):
                return (1, 2)

        class GivesStr:
            def __iter__(self):
                return "ab"

        class BadMeta(type):
            def __iter__(cls):
                return [1, 2]

        class BadMembers(metaclass=BadMeta):
            pass

        class SelfIter:
            def __init__(self):
                self.n = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.n = self.n + 1
                if self.n > 2:
                    raise StopIteration
                return self.n

        class GenIter:
            def __iter__(self):
                yield "g1"
                yield "g2"

        class WrapsIter:
            def __iter__(self):
                return iter([1, 2, 3])

        class GetItem:
            def __getitem__(self, i):
                if i < 2:
                    return i * 10
                raise IndexError

        class Meta(type):
            def __iter__(cls):
                return iter(["m"])

        class Members(metaclass=Meta):
            pass

        def show(label, f):
            try:
                print(label, repr(f()))
            except TypeError as e:
                print(label, "TypeError:", e)

        # REFUSED, and by `iter`'s words rather than `next`'s -- on the eager
        # path, the lazy one and `join` alike.
        for name, k in (("list", GivesList), ("dict", GivesDict),
                        ("set", GivesSet), ("tuple", GivesTuple),
                        ("str", GivesStr)):
            show("for " + name, lambda k=k: [x for x in k()])
            show("list " + name, lambda k=k: list(k()))
            show("iter " + name, lambda k=k: next(iter(k())))
            show("join " + name, lambda k=k: ",".join(k()))
        show("metaclass gives list", lambda: list(BadMembers))
        # AND EVERY REAL WAY OF BEING ITERABLE STILL IS.
        show("self-iter", lambda: [x for x in SelfIter()])
        show("iter is self", lambda: (lambda o: iter(o) is o)(SelfIter()))
        show("generator method", lambda: list(GenIter()))
        show("wraps a cursor", lambda: list(WrapsIter()))
        show("getitem walk", lambda: list(GetItem()))
        show("metaclass", lambda: list(Members))
        show("builtins", lambda: (list([1, 2]), list((3,)), list({"k": 1})))
        show("str and range", lambda: (list("ab"), list(range(3))))
        show("cursors", lambda: (list(map(str, [1])), list(zip([1], [2]))))
        show("iter builtins", lambda: (next(iter([7])), next(iter({"z": 1})),
                                       next(iter("q"))))
        show("half consumed",
             lambda: (lambda i: (next(i), list(i)))(iter([1, 2, 3])))
        show("unpack", lambda: (lambda a, b: (a, b))(*SelfIter()))
        show("sum", lambda: sum(SelfIter()))
        show("not iterable at all", lambda: list(object()))
    """,
    "a_big_integer_formats_in_the_base_it_was_asked_for": """
        # `format(2 ** 70, "x")` ANSWERED THE DECIMAL DIGITS, with nothing to
        # say the base had been ignored -- a wrong answer rather than a
        # missing feature, and `%x` inherits it because `%` is translated
        # into this language. `hex()` has converted a big integer all along;
        # this path just never asked it, and said so in a comment.
        #
        # AND THE BIG PATH DID NOT DO WHAT THE MACHINE-WORD PATH DOES. It
        # returned the decimal text padded, so a sign flag was dropped and
        # grouping never happened: `format(2 ** 70, "+d")` lost the `+` and
        # `format(2 ** 70, ",d")` came back ungrouped. The two paths now
        # build the same layout -- the sign the spec asks for, the optional
        # prefix, the digits, then grouping -- and only the DIGITS differ,
        # which is the only part that has to.
        def show(label, f):
            try:
                print(label, repr(f()))
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        n = 2 ** 70
        m = 2 ** 100 + 12345
        # THE BASE THAT WAS ASKED FOR.
        show("x", lambda: format(n, "x"))
        show("X", lambda: format(n, "X"))
        show("o", lambda: format(n, "o"))
        show("b", lambda: format(n, "b"))
        show("neg", lambda: (format(-n, "x"), format(-n, "o"), format(-n, "b")))
        show("alt", lambda: (format(n, "#x"), format(n, "#o"), format(n, "#b")))
        show("alt neg", lambda: (format(-n, "#x"), format(-n, "#X")))
        show("upper", lambda: (format(m, "X"), format(m, "#X")))
        show("width", lambda: (format(n, "30x"), format(n, "<30x"),
                               format(n, "^30x")))
        show("sign", lambda: (format(n, "+x"), format(n, " x"),
                              format(-n, "+x")))
        # THE DECIMAL SIDE, which had the same gap.
        show("plus", lambda: format(n, "+d"))
        show("space", lambda: format(n, " d"))
        show("plus neg", lambda: format(-n, "+d"))
        show("comma", lambda: format(n, ",d"))
        show("under", lambda: format(n, "_d"))
        show("comma neg", lambda: format(-n, ",d"))
        show("plus comma", lambda: format(n, "+,d"))
        show("comma width", lambda: format(n, ">35,d"))
        show("plain", lambda: (format(n, "d"), format(n, ""), format(-n, "d")))
        # `c` OF A BIG IS NO CHARACTER, and refusing it is what keeps
        # `apy_chr` from reading a machine word out of a value that has none.
        show("c", lambda: format(n, "c"))
        # THE MACHINE-WORD PATH MUST NOT HAVE MOVED.
        show("small", lambda: (format(255, "x"), format(-255, "#X"),
                               format(0, "x"), format(0, "#b")))
        show("boundary", lambda: (format(2 ** 63 - 1, "x"), format(2 ** 63, "x"),
                                  format(2 ** 64, "x"), format(-(2 ** 63), "x")))
        show("small dec", lambda: (format(1234567890, "+d"),
                                   format(1234567890, ",d"),
                                   format(-1234567890, "030d")))
        # AND THE OTHER SPELLINGS OF THE SAME CONVERSION.
        show("fstring", lambda: f"{n:x}|{n:#X}|{-n:+#o}")
        show("percent", lambda: ("%x" % n, "%X" % n, "%o" % n, "%d" % n))
        show("builtins", lambda: (bin(n), oct(n), hex(n), hex(-n)))
        show("floats", lambda: (format(1.5, ".2f"), format(1e21, ".3e")))
    """,
    "a_bytes_method_keeps_its_tag_through_the_two_way_dispatch": """
        # `b",".join([b"a", b"b"])` ANSWERED A str. A str method on a bytes
        # receiver answers bytes -- the two share a layout, so the operation
        # is the same one and only the TAG on the result differs, which
        # `apy_str_like` puts back. The ordinary path applied it; the
        # COLLISION path returned without it.
        #
        # WHICH MAKES THE TRIGGER A CLASS, not the method. A module defining
        # ONE class that extends a builtin puts every method name on the
        # two-way dispatch -- see `a_bound_builtin_method_takes_its_keywords`
        # for the same mechanism biting keywords -- so `SubList` below is
        # what turns every line here from right to wrong. Without it they all
        # answer correctly, which is why this went unnoticed.
        #
        # A WRONG TYPE, NOT AN ERROR, with nothing to announce it until
        # something later writes the result to a descriptor or compares it
        # against a literal. `split`, `replace`, `partition`, `strip`,
        # `upper` and the rest were all in scope, not just `join`.
        class SubList(list):
            pass

        class Own:
            def split(self, sep):
                return "OWN split " + repr(sep)

            def upper(self):
                return "OWN upper"

        def show(label, f):
            try:
                print(label, repr(f()))
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        b = b"a-b-c"
        ba = bytearray(b"a-b-c")
        show("split", lambda: b.split(b"-"))
        show("rsplit", lambda: b.rsplit(b"-", 1))
        show("splitlines", lambda: b"x\\ny".splitlines())
        show("partition", lambda: b.partition(b"-"))
        show("rpartition", lambda: b.rpartition(b"-"))
        show("strip", lambda: b"  x  ".strip())
        show("lstrip", lambda: b"..x".lstrip(b"."))
        show("replace", lambda: b.replace(b"-", b"+"))
        show("case", lambda: (b.upper(), b"AB".lower(), b"aB".swapcase()))
        show("pad", lambda: (b"a".ljust(3, b"."), b"a".rjust(3, b"."),
                             b"a".center(5, b"*"), b"7".zfill(3)))
        show("join", lambda: b",".join([b"q", b"r"]))
        show("expandtabs", lambda: b"a\\tb".expandtabs(2))
        show("affix", lambda: (b"abc".removeprefix(b"a"),
                               b"abc".removesuffix(b"c")))
        show("bytearray", lambda: (ba.split(b"-"), ba.upper()))
        # THE CONVERSIONS ANSWER A str ON PURPOSE and must not be re-tagged.
        show("hex and decode", lambda: (b"ab".hex(), b"ab".decode()))
        # AND A RESULT THAT IS NOT TEXT is left alone either way.
        show("not text", lambda: (b.find(b"-"), b.count(b"-"),
                                  b.startswith(b"a")))
        # A str RECEIVER STAYS str.
        show("str", lambda: ("a-b-c".split("-"), "A".lower(),
                             "a-b".partition("-")))
        # AND THE USER-CLASS HALF of the very same dispatch still reaches the
        # class: `apy_str_like` hands an instance straight back.
        show("own split", lambda: Own().split("-"))
        show("own upper", lambda: Own().upper())
        show("inherited", lambda: SubList([3, 1]).pop(0))
    """,
    "the_format_spec_groups_and_pads_and_refuses_as_cpython_does": """
        # FIVE RULES, every one of them wrong at MACHINE-WORD size, so none
        # of them is about big integers -- they surfaced while fixing the
        # base conversion for those and were each checked against a small
        # value before being separated out.
        #
        # 1. ZERO PADDING WENT BEFORE THE PREFIX: `format(255, "#030x")` was
        #    `000000000000000000000000000xff`, with the `0x` stranded in the
        #    middle of the fill where it reads as neither prefix nor digit.
        #    The `=` fill belongs after the sign AND after the prefix.
        # 2. GROUPING USED THREE DIGITS IN EVERY BASE. CPython groups the
        #    non-decimal bases by FOUR and only `d`/`n` by three.
        # 3. GROUPING WAS SKIPPED PAST 120 DIGITS, on account of a fixed
        #    scratch buffer -- so every big integer past that came back
        #    unseparated, which is a wrong answer and a silent one.
        # 4. `,` AND `_` WERE ACCEPTED WITH `n`, and `,` with the
        #    non-decimal bases, where CPython refuses all of them by name.
        #    `_x` IS allowed and `,x` is not, so the rule is about the
        #    SEPARATOR and not only about the base.
        # 5. `n` WAS AN INTEGER TYPE ONLY: `format(1.5, "n")` was refused as
        #    an unknown format code for a float, where it means `g`.
        #
        # AND `c` OUT OF RANGE was a ValueError from `chr` where CPython
        # raises an OverflowError from `%c` -- the wrong TYPE is what a
        # program catching it sees.
        def show(label, f):
            try:
                print(label, repr(f()))
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        n = 10 ** 301
        b = 2 ** 500
        # PAST THE OLD CAP.
        show("302 digits", lambda: format(n, ",d")[:50])
        show("302 length", lambda: len(format(n, ",d")))
        show("big hex", lambda: format(b, "_x")[:50])
        show("big bin length", lambda: len(format(b, "_b")))
        # THE GROUP SIZES.
        show("hex", lambda: (format(0x1, "_x"), format(0xFFFF, "_x"),
                             format(0x12345, "_x"), format(0x123456789, "_x")))
        show("bin", lambda: (format(0b1, "_b"), format(0b11111, "_b")))
        show("oct", lambda: (format(0o7, "_o"), format(0o1234567, "_o")))
        show("dec", lambda: (format(1, "_d"), format(1234, "_d"),
                             format(1234567890, ",d")))
        show("float", lambda: (format(1234567.891, ",.2f"), format(1e6, ",.1f")))
        # THE FILL AND THE PREFIX.
        show("alt zero", lambda: (format(255, "#030x"), format(-255, "#030x"),
                                  format(8, "#012o"), format(5, "#012b")))
        show("alt zero upper", lambda: format(255, "#030X"))
        show("plain zero", lambda: (format(255, "030x"), format(-255, "030d")))
        show("explicit =", lambda: (format(255, "*=10x"), format(-255, "*=10d")))
        show("no width", lambda: (format(255, "#x"), format(255, "#o")))
        # WHAT A SEPARATOR MAY GO WITH.
        for spec in (",n", "_n", ",x", ",X", ",o", ",b"):
            show("refuse " + spec, lambda spec=spec: format(255, spec))
        for spec in ("_x", "_X", "_o", "_b", ",d", "_d", "n"):
            show("allow " + spec, lambda spec=spec: format(255, spec))
        # `n` ON EITHER KIND OF NUMBER.
        show("float n", lambda: (format(1.5, "n"), format(1234.5678, "10.3n"),
                                 format(1e21, "n"), format(0.0001, "n")))
        show("int n", lambda: (format(255, "n"), format(10 ** 40, "n")))
        # AND `c`.
        show("c ok", lambda: (format(65, "c"), format(255, "c")))
        show("c big", lambda: format(2 ** 70, "c"))
        show("c over", lambda: format(2 ** 40, "c"))
        show("c negative", lambda: format(-1, "c"))
        show("chr is its own", lambda: chr(2 ** 40))
    """,
    "the_older_iteration_protocol_ignores_dunder_len": """
        # A CLASS WITH `__len__` AND `__getitem__` WAS WALKED BY READING
        # `__len__` ONCE and subscripting that many times. CPython never
        # consults `__len__` for iteration: the older protocol is
        # `__getitem__` alone, called from 0 until it reports IndexError.
        #
        # SO A CLASS WHOSE TWO DISAGREE ANSWERED DIFFERENTLY. One with a
        # `__len__` of 2 and a `__getitem__` good for five stopped at two --
        # three elements silently missing, from `for`, from `list`, from
        # `in`, from `max`, from every consumer at once.
        #
        # AND `__len__` ALONE DOES NOT MAKE AN OBJECT ITERABLE, which is the
        # other half of the same rule: the shortcut answered the object
        # itself for anything with a `__len__`, so a class with no
        # `__getitem__` was walked by index anyway.
        #
        # THE LAZY PATH ALREADY HAD THIS RIGHT -- a cursor over an instance
        # reads through `__getitem__` and stops on IndexError -- which is why
        # only the EAGER funnel needed changing, in all three arrangements.
        log = []

        class LenLies:
            def __len__(self):
                return 2

            def __getitem__(self, i):
                log.append(i)
                if i < 5:
                    return i * 10
                raise IndexError

        class OnlyLen:
            def __len__(self):
                return 3

        class OnlyGet:
            def __getitem__(self, i):
                if i < 3:
                    return i
                raise IndexError

        def show(label, f):
            try:
                print(label, repr(f()))
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        # THE WALK GOES PAST WHAT `__len__` CLAIMS.
        show("comprehension", lambda: [x for x in LenLies()])
        print("getitem calls", log)
        show("tuple", lambda: tuple(LenLies()))
        show("sum", lambda: sum(LenLies()))
        show("in", lambda: 30 in LenLies())
        show("max", lambda: max(LenLies()))
        show("sorted", lambda: sorted(LenLies()))
        show("star", lambda: [*LenLies()])
        # AND `__len__` ITSELF STILL ANSWERS, which is what it is for.
        show("len", lambda: len(LenLies()))
        show("subscript", lambda: LenLies()[3])
        # A `__len__` WITH NO `__getitem__` IS NOT ITERABLE.
        show("only len iterated", lambda: [x for x in OnlyLen()])
        show("only len listed", lambda: list(OnlyLen()))
        show("only len measured", lambda: len(OnlyLen()))
        # AND `__getitem__` WITH NO `__len__` WALKS AS IT ALWAYS DID.
        show("only getitem", lambda: [x for x in OnlyGet()])
        show("only getitem star", lambda: [*OnlyGet()])
        show("only getitem unpack",
             lambda: (lambda a, b, c: (a, b, c))(*OnlyGet()))
    """,
    "a_surplus_positional_argument_is_refused_and_not_dropped": """
        # A CALL WITH MORE POSITIONAL ARGUMENTS THAN THE FUNCTION DECLARES
        # answered, silently, from the first few. `two(0, 1, 2)` gave
        # `(0, 1)` on the interpreter and on both compiled runtimes -- a
        # WRONG ANSWER where CPython raises, and the one shape of mistake a
        # caller is least likely to catch by reading the output.
        #
        # THE PACKING CAPS WHAT IT COPIES at the positional capacity, so the
        # count compared afterwards could only ever come out too SMALL: every
        # surplus was gone before anything looked. The count is taken from
        # what the CALL WROTE instead, ahead of the packing.
        #
        # AND THE FOUR WORDINGS ARE ONE FAMILY. Which end the count fell off,
        # and whether a default makes the low end a range:
        #
        #     f() takes 2 positional arguments but 3 were given
        #     f() takes from 1 to 2 positional arguments but 3 were given
        #     f() missing 1 required positional argument: 'b'
        #     f() missing 2 required keyword-only arguments: 'k' and 'j'
        #
        # `self` IS COUNTED AT BOTH ENDS of the `takes` message and at
        # neither end of the `missing` one, and the function is named by its
        # QUALNAME -- `C.m()`, `outer.<locals>.inner()` -- which is what sent
        # a reader of `__init__() missing ...` looking for a free function.
        def show(label, f):
            try:
                print(label, repr(f()))
            except TypeError as e:
                print(label, "TypeError:", e)

        def two(a, b):
            return (a, b)

        def one(a):
            return a

        def withdef(a, b=9):
            return (a, b)

        def withrest(a, *rest):
            return (a, rest)

        def kwonly(a, *, k=1):
            return (a, k)

        def needskw(a, *, k):
            return (a, k)

        def needstwokw(a, *, k, j):
            return (a, k, j)

        def dfl(a, b=1, *, k=0):
            return (a, b, k)

        def kwrest(a, **kw):
            return (a, kw)

        def outer():
            def inner(a, b):
                return (a, b)
            return inner

        class C:
            def m(self, a):
                return a

        class D:
            def __init__(self, a, *, k):
                self.a = a

        class E:
            def __init__(self, a, b=2):
                self.a = a

        # A SURPLUS IS REFUSED, written out and splatted alike.
        show("surplus", lambda: two(0, 1, 2))
        show("surplus splat", lambda: two(*[0, 1, 2]))
        show("surplus two over", lambda: two(*[0, 1, 2, 3]))
        show("surplus of one", lambda: one(1, 2))
        show("surplus lambda", lambda: (lambda a, b: (a, b))(*[0, 1, 2]))
        # A DEFAULT MAKES THE LOW END A RANGE.
        show("surplus with default", lambda: withdef(0, 1, 2))
        # A KEYWORD-ONLY PARAMETER IS NOT A POSITION.
        show("surplus past kwonly", lambda: kwonly(0, 1))
        # `*rest` IS WHAT A SURPLUS IS FOR.
        show("rest takes it", lambda: withrest(0, 1, 2))
        # `self` IS COUNTED, AND THE METHOD IS NAMED BY ITS CLASS.
        show("surplus on a method", lambda: C().m(1, 2))
        show("surplus on a class", lambda: E(1, 2, 3))
        # TOO FEW IS SAID IN NAMES.
        show("one short", lambda: two(0))
        show("two short", lambda: two())
        show("short with default", lambda: withdef())
        show("short method", lambda: C().m())
        show("short class", lambda: E())
        show("short nested", lambda: outer()(1))
        # A KEYWORD-ONLY PARAMETER IS MISSED BY NAME, never by count.
        show("kwonly short", lambda: needskw(1))
        show("kwonly two short", lambda: needstwokw(1))
        # A SURPLUS ALONGSIDE KEYWORDS COUNTS THEM SEPARATELY.
        show("surplus and kwonly", lambda: needskw(1, 2, k=3))
        show("surplus range and kwonly", lambda: dfl(1, 2, 3, k=1))
        show("surplus and kwrest", lambda: kwrest(1, 2, x=1))
        show("surplus class and kwonly", lambda: D(1, 2, k=1))
        # AND A CALL THAT FITS STILL ANSWERS.
        show("exact", lambda: two(0, 1))
        show("default taken", lambda: withdef(0))
        show("kwonly named", lambda: needskw(1, k=2))
        show("kwrest fits", lambda: kwrest(1, x=1))
    """,
    "a_keyword_refusal_gathers_every_name_and_says_the_qualname": """
        # THREE REFUSALS STOPPED AT THE FIRST NAME THEY MET and none of them
        # said which function they were about.
        #
        # POSITIONAL-ONLY NAMES ARE GATHERED. `pos(a=1, b=2, c=3)` against
        # `def pos(a, b, /, c)` is `got some positional-only arguments passed
        # as keyword arguments: 'a, b'` -- every offender in ONE refusal, in
        # DECLARATION order. Returning at the first match named one of them,
        # and named whichever the CALL happened to put first, so
        # `pos(b=2, a=1, c=3)` said 'b'. The scan also runs BEFORE any other
        # keyword complaint -- `pos(a=1, zz=2, c=3)` names `a` and never
        # mentions `zz` -- and is skipped where a `**kw` exists for the names
        # to land in.
        #
        # MISSING PARAMETERS ARE GATHERED TOO, and the holes need not be
        # adjacent: `f(b=2)` against `def f(a, b, c)` is missing `'a' and
        # 'c'`. Reporting the first meant a caller who forgot two learned
        # about one, fixed it, and came straight back. A keyword-only hole is
        # a SEPARATE message, said only when no positional one is
        # outstanding.
        #
        # AND ALL OF THEM NAME THE FUNCTION BY ITS QUALNAME -- `K.m()`,
        # `K.__init__()`, `outer.<locals>.inner()` -- which is what CPython
        # prints and what the arity messages already said.
        # CALLED THROUGH A VALUE, not written out. The frontend refuses
        # some of these spellings at COMPILE time with wording of its own --
        # see the static-checker divergence -- and what is measured here is
        # the RUN-TIME binder, which is what a call through a value reaches.
        def show(label, fn, args, kw):
            try:
                print(label, repr(fn(*args, **kw)))
            except TypeError as e:
                print(label, "TypeError:", e)

        def pos(a, b, /, c):
            return (a, b, c)

        def poskw(a, b, /, **kw):
            return (a, b, kw)

        def three(a, b, c):
            return (a, b, c)

        def four(a, b, c, d):
            return (a, b, c, d)

        def mixed(a, b=1, *, k, j):
            return (a, b, k, j)

        def outer():
            def inner(a, b, /, c):
                return (a, b, c)
            return inner

        class K:
            def __init__(self, a, /, b=0):
                self.a = a

            def m(self, a, b, /, c):
                return (a, b, c)

        class Q:
            def m(self, a):
                return a

        # EVERY POSITIONAL-ONLY NAME, IN DECLARATION ORDER.
        show("both by name", pos, [], {"a": 1, "b": 2, "c": 3})
        show("both, call order reversed", pos, [], {"b": 2, "a": 1, "c": 3})
        show("only one by name", pos, [1, 2], {"a": 9, "c": 3})
        # AND BEFORE ANY OTHER KEYWORD COMPLAINT.
        show("alongside an unknown name", pos, [], {"a": 1, "zz": 2, "c": 3})
        # UNLESS THERE IS A `**kw` FOR THEM TO LAND IN.
        show("kwrest takes them", poskw, [], {"a": 1, "b": 2})
        show("kwrest keeps its own", poskw, [1, 2], {"a": 1})
        # EVERY HOLE AT ONCE, ADJACENT OR NOT.
        show("holes at the ends", three, [], {"b": 2})
        show("three holes", four, [], {"b": 2})
        show("no argument at all", three, [], {})
        # A KEYWORD-ONLY HOLE IS ITS OWN MESSAGE, AND WAITS ITS TURN.
        show("positional first", mixed, [], {"k": 1})
        show("then the keyword-only", mixed, [1], {})
        show("a default is not a hole", mixed, [1], {"k": 1, "j": 2})
        # AND THE QUALNAME NAMES THE FUNCTION.
        show("method", K(1).m, [], {"a": 1, "b": 2, "c": 3})
        show("init", K, [], {"a": 1})
        show("nested", outer(), [], {"a": 1, "b": 2, "c": 3})
        show("unknown name, nested", outer(), [], {"zz": 1})
        show("unknown name, method", Q().m, [], {"zz": 1})
        show("by name and position both", Q().m, [1], {"a": 2})
    """,
    "a_default_belongs_to_its_own_parameter": """
        # `def mixed(a, b=1, *, k, j)` called `mixed(1, k=1, j=2)` refused,
        # with `missing 1 required positional argument: 'b'`, for a call
        # CPython answers.
        #
        # THE DEFAULTS WERE INDEXED AS IF EVERY ONE BELONGED TO A TRAILING
        # PARAMETER. That is false the moment a keyword-only parameter
        # WITHOUT a default follows a positional one WITH: `b`'s default sits
        # at index 0 of a one-element list, and the arithmetic looked for it
        # two places before the start.
        #
        # THE LAYOUT IS TWO RUNS. The positional defaults first, covering the
        # LAST that many POSITIONAL parameters; then the keyword-only ones,
        # covering the last that many of the keyword-only tail. `nkwdefault`
        # is recorded on the function for exactly this, and says where the
        # first run ends.
        def show(label, f):
            try:
                print(label, repr(f()))
            except TypeError as e:
                print(label, "TypeError:", e)

        def a1(a, b=1, *, k):
            return (a, b, k)

        def a2(a, b=1, *, k=5):
            return (a, b, k)

        def a3(a=0, b=1, *, k, j=7):
            return (a, b, k, j)

        def a4(a, *, k, j=7):
            return (a, k, j)

        def a5(a, b=1, c=2, *rest, k, j=7, **kw):
            return (a, b, c, rest, k, j, kw)

        def a6(a, /, b=1, *, k=2):
            return (a, b, k)

        def a7(*, k=1, j=2):
            return (k, j)

        # A POSITIONAL DEFAULT BEHIND A REQUIRED KEYWORD-ONLY PARAMETER.
        show("one default, one required", lambda: a1(1, k=9))
        show("both positions written", lambda: a1(1, 2, k=9))
        show("and the keyword-only missing", lambda: a1(1))
        # BOTH RUNS AT ONCE.
        show("both defaulted", lambda: a2(1))
        show("both written", lambda: a2(1, 2, k=9))
        show("two positional, two keyword-only", lambda: a3(k=9))
        show("all four written", lambda: a3(1, 2, k=9, j=8))
        show("keyword-only still required", lambda: a3())
        # NO POSITIONAL DEFAULT AT ALL.
        show("keyword-only pair", lambda: a4(1, k=9))
        show("keyword-only pair written", lambda: a4(1, k=9, j=8))
        # WITH `*rest` AND `**kw` AROUND THEM.
        show("rest and kw empty", lambda: a5(1, k=9))
        show("rest and kw filled", lambda: a5(1, 2, 3, 4, 5, k=9, j=8, z=0))
        # AND WITH A `/` IN FRONT.
        show("positional-only lead", lambda: a6(1))
        show("its default named", lambda: a6(1, b=2))
        # KEYWORD-ONLY ONLY.
        show("all defaulted", lambda: a7())
        show("one of two named", lambda: a7(j=9))
        # AND ALL OF IT THROUGH A BOUND METHOD, which is where the two halves
        # came apart: the C copies a function's whole shape when it binds a
        # receiver, while the interpreter copied it field by field and left
        # three out -- where the keyword-only defaults begin, whether calling
        # it builds a coroutine, and the attributes set on the function
        # itself. The first of those is this rule, and a class whose
        # `__init__` takes a keyword-only default is the commonest shape
        # there is: `ContextVar(name, *, default=...)` built by name refused
        # every construction that left the default alone.
        import inspect

        class Var:
            def __init__(self, name, *, default=None):
                self.name = name
                self.default = default

            def m(self, a, b=1, *, k=2):
                return (a, b, k)

            async def am(self):
                return 1

            m.tag = "kept"

        show("built with the default", lambda: Var("x").name)
        show("the default itself", lambda: Var("x").default)
        show("built with it named", lambda: Var("y", default=5).default)
        show("bound method defaults", lambda: Var("z").m(1))
        show("bound method written", lambda: Var("z").m(1, 2, k=3))
        # AND THE OTHER TWO THINGS A BIND MUST CARRY.
        show("an attribute set on the method",
             lambda: Var("z").m.tag)
        show("a coroutine function stays one",
             lambda: inspect.iscoroutinefunction(Var("z").am))
        show("and unbound too",
             lambda: inspect.iscoroutinefunction(Var.am))
    """,
    "iter_with_a_sentinel_calls_as_the_walk_asks": """
        # `iter(f, sentinel)` CALLED `f` UNTIL THE SENTINEL AT CONSTRUCTION
        # and handed back a cursor over the results. Two things were wrong
        # with that and a program sees both.
        #
        # THE CALLS HAPPENED TOO EARLY. `iter(feed, 4)` had already called
        # `feed` four times before the first `next()`, so anything with a side
        # effect -- reading a file a block at a time, which is what this form
        # is FOR -- ran to the end before the caller asked for one item.
        #
        # AND A CALLABLE THAT NEVER ANSWERS THE SENTINEL was capped at a
        # million calls. `for v in iter(f, s): break` leaves after ONE call in
        # CPython; here it made a million and then stopped, so a program that
        # works there did not terminate in any useful time here.
        #
        # IT IS A CURSOR MODE NOW, alongside map, filter, enumerate, zip and
        # reversed, which were already lazy for the same reason. The sentinel
        # lives where a plain cursor keeps its source and the callable where
        # `map` keeps its function.
        #
        # EXHAUSTED STAYS EXHAUSTED: CPython drops the callable the moment the
        # sentinel arrives, so a second `next()` answers StopIteration without
        # calling again -- which the call count below is what proves.
        def show(label, f):
            try:
                print(label, repr(f()))
            except Exception as e:
                print(label, type(e).__name__ + ":", e)

        calls = []

        def feed():
            calls.append(len(calls))
            return len(calls)

        it = iter(feed, 4)
        print("named", type(it).__name__)
        # NOTHING HAS BEEN CALLED YET.
        print("before", calls)
        print("first", next(it), calls)
        print("second", next(it), calls)
        print("rest", list(it), len(calls))

        seen = []

        def forever():
            seen.append(1)
            return 0

        # A CALLABLE THAT NEVER ANSWERS THE SENTINEL IS ENDLESS, and a walk
        # that leaves early has made exactly as many calls as it took.
        inf = iter(forever, -1)
        n = 0
        for v in inf:
            n += 1
            if n >= 5:
                break
        print("bounded", n, len(seen))
        print("resumes", next(inf), len(seen))

        done = []

        def three():
            done.append(1)
            return len(done)

        walk = iter(three, 3)
        print("drained", list(walk), len(done))
        print("exhausted", next(walk, "gone"), len(done))
        print("still exhausted", next(walk, "gone"), len(done))

        # THE SENTINEL IS COMPARED BY VALUE and not by identity: a callable
        # that answers a fresh empty string still stops.
        def empties():
            return ""

        print("by value", list(iter(empties, "")))

        # AND THE ERRORS THE FORM ITSELF REPORTS.
        show("not callable", lambda: list(iter(5, 1)))
        show("through next", lambda: next(iter(feed, 4)))
    """,
    "a_written_def_beats_the_builtin_of_the_same_name": """
        # `def len(a)` WAS COMPILED, BOUND, AND NEVER CALLED. `len(x)`
        # reached the runtime's `apy_len` and answered the builtin's answer,
        # with nothing said about it -- and the same for `min`, `sorted`,
        # `sum`, `repr`, `hex`, `isinstance` and every other name the
        # frontend dispatches on.
        #
        # THE BRANCHES DISPATCH ON THE NAME and ran before the module's own
        # functions were consulted at all. Shadowing a builtin at module
        # level is ordinary, legal Python -- a `def id(...)` over a record
        # key, a `def input(...)` over a prompt -- and a program that does it
        # got a wrong answer here with no diagnostic.
        #
        # AN ASSIGNMENT ALREADY WORKED. `len = f` binds a name, and a bound
        # name is a local, which those branches did ask about. Only the `def`
        # spelling was missed, which is why it lasted.
        # PRINTED AND NOT `repr`-ed, because `repr` is one of the names
        # shadowed below and the program's own would be what ran -- which is
        # right, and would leave every line here reading "my repr".
        def show(label, f):
            try:
                print(label, f())
            except Exception as e:
                print(label, e.__class__.__name__ + ":", e)

        def len(a):
            return "my len"

        def min(a, b):
            return "my min"

        def sorted(a):
            return "my sorted"

        def sum(a):
            return "my sum"

        def repr(a):
            return "my repr"

        def hex(a):
            return "my hex"

        def isinstance(a, b):
            return "my isinstance"

        def dict(*a, **k):
            return "my dict"

        def type(a):
            return "my type"

        def next(a, b=None):
            return "my next"

        def zip(*a):
            return "my zip"

        def enumerate(a):
            return "my enumerate"

        def getattr(a, b):
            return "my getattr"

        class K:
            pass

        # THE ONE-CALL BUILTINS.
        show("len", lambda: len([1, 2]))
        show("min", lambda: min(1, 2))
        show("sorted", lambda: sorted([2, 1]))
        show("sum", lambda: sum([1]))
        show("repr", lambda: repr(1))
        show("hex", lambda: hex(9))
        # NOT `abs`, though it is the same branch as `len` and `sum`: a
        # program that writes `def abs` does not BUILD under the C runtime at
        # all, because an emitted function takes its Python name as a
        # top-level C symbol and libc already has that one. A separate fault
        # with a separate fix; `len` and the rest cover this rule.
        # THE ONES WITH BRANCHES OF THEIR OWN.
        show("isinstance", lambda: isinstance(K(), K))
        show("type", lambda: type(1))
        show("next", lambda: next(1))
        show("zip", lambda: zip([1]))
        show("enumerate", lambda: enumerate([1]))
        show("getattr", lambda: getattr(1, "x"))
        # AND THE CONSTRUCTOR SHAPES, whose keywords and empty call are
        # folded at compile time.
        show("dict with keywords", lambda: dict(a=1))
        show("dict empty", lambda: dict())
        show("dict from pairs", lambda: dict([(1, 2)]))
        # A SPLAT STILL REACHES IT.
        show("splatted", lambda: min(*[1, 2]))
        show("keyword splatted", lambda: sorted(*[[2, 1]]))
    """,
    "a_function_named_like_a_c_library_symbol_still_builds": """
        # A MODULE-LEVEL `def` KEEPS ITS BARE NAME AS ITS IR SYMBOL, and the
        # C backend emits that name as a top-level function. So a program
        # that wrote `def abs(a)` did not BUILD:
        #
        #     error: conflicting types for 'abs';
        #            have 'uintptr_t(uintptr_t, uintptr_t)'
        #     note: previous declaration of 'abs' with type 'int(int)'
        #
        # `abs`, `time`, `index`, `pow`, `log`, `exit`, `remove`, `div` and
        # `send` are ordinary Python function names and CPython runs every one
        # of them. Twelve of the seventeen names tried stopped the build.
        #
        # THE NAMES MOVE ASIDE NOW, the same way a nested `def` already did --
        # its key is not a C identifier, so it gets a `pyf_` symbol, and a key
        # the C runtime already owns is no more usable as a symbol than that.
        #
        # DECIDED BEFORE ANY BACKEND rather than in the C emitter, because the
        # collision is not only C's: the assembler backends emit an object
        # file that links against libc, and a global `abs` there preempts the
        # real one at LINK time -- a wrong answer rather than an error.
        def abs(a):
            return ("my abs", a)

        def time():
            return "my time"

        def index(a):
            return ("my index", a)

        def pow(a, b):
            return ("my pow", a, b)

        def log(a):
            return ("my log", a)

        def remove(a):
            return ("my remove", a)

        def div(a, b):
            return ("my div", a, b)

        def send(a):
            return ("my send", a)

        def free(a):
            return ("my free", a)

        def malloc(n):
            return ("my malloc", n)

        def printf(s):
            return ("my printf", s)

        def sqrt(a):
            return ("my sqrt", a)

        def floor(a):
            return ("my floor", a)

        def system(a):
            return ("my system", a)

        def rename(a, b):
            return ("my rename", a, b)

        def stdout():
            return "my stdout"

        def errno():
            return "my errno"

        print(abs(-1), time(), index(2), pow(2, 3), log(4))
        print(remove(5), div(6, 7), send(8), free(9), malloc(10))
        print(printf("x"), sqrt(11), floor(12), system("y"), rename("a", "b"))
        print(stdout(), errno())

        # AND THEY ARE STILL ORDINARY VALUES, reachable by name and callable
        # through one -- the renaming is a SYMBOL, not a binding.
        def apply(f, a):
            return f(a)

        print(apply(abs, -2), apply(index, 3))
        fs = [abs, log, send]
        print([f(1) for f in fs])
    """,
    "driving_an_iterator_by_hand_answers_values_not_handles": """
        # `it.__next__()` ANSWERED A NUMBER. `iter([1, 2, 3]).__next__()` gave
        # `4294967317` where CPython gives `1` -- the interpreter's internal
        # HANDLE for the value, handed back without being unwrapped. A silent
        # wrong answer, and one only the interpreter gave, so the three
        # arrangements disagreed with each other.
        #
        # EVERY CURSOR KIND AND A GENERATOR ALIKE: a list iterator, a
        # `reversed`, a str, a dict, `map`, `filter`, `enumerate`, `zip`.
        #
        # AND `it.__iter__() is it` WAS FALSE, where CPython says True -- the
        # same missing unwrap, because an int is not the cursor.
        #
        # ONLY THE DUNDER REACHED AS AN ATTRIBUTE was wrong. `next(it)` and a
        # `for` loop never come this way, which is why it lasted; a program
        # sees it the moment it drives an iterator by hand, which is exactly
        # what a wrapper class with its own `__iter__`/`__next__` does.
        def show(label, f):
            try:
                print(label, f())
            except Exception as e:
                print(label, e.__class__.__name__ + ":", e)

        def g():
            yield 7
            yield 8

        it = iter([1, 2, 3])
        show("first", lambda: it.__next__())
        show("second", lambda: it.__next__())
        show("generator", lambda: g().__next__())
        show("map", lambda: map(lambda x: x * 2, [5]).__next__())
        show("filter", lambda: filter(None, [6]).__next__())
        show("enumerate", lambda: enumerate(["a"]).__next__())
        show("zip", lambda: zip([1], [2]).__next__())
        show("reversed", lambda: reversed([1, 2]).__next__())
        show("str", lambda: iter("ab").__next__())
        show("dict", lambda: iter({5: 6}).__next__())
        show("range", lambda: range(3).__iter__().__next__())
        # AN ITERATOR IS ITS OWN `__iter__`.
        show("cursor is its own", lambda: (lambda i: i.__iter__() is i)(
            iter([1])))
        show("generator is its own", lambda: (lambda i: i.__iter__() is i)(g()))
        # AND A CONTAINER'S `__iter__` ANSWERS A CURSOR, not a number.
        show("list", lambda: list([1, 2].__iter__()))
        show("tuple", lambda: list((1, 2).__iter__()))
        show("bytes", lambda: list(b"ab".__iter__()))
        show("range walked", lambda: list(range(3).__iter__()))
        show("set", lambda: sorted({1, 2}.__iter__()))
        # THE WHOLE PROTOCOL DRIVEN BY HAND, which is the shape that made this
        # matter: a wrapper that forwards to the cursor it holds.
        class Twice:
            def __init__(self, src):
                self.it = src.__iter__()

            def __iter__(self):
                return self

            def __next__(self):
                return self.it.__next__() * 2

        print("wrapped", [x for x in Twice([1, 2, 3])])
        print("wrapped listed", list(Twice([4, 5])))
        # AND `send` STILL CARRIES ITS OWN VALUE THROUGH.
        show("send", lambda: (lambda i: (i.__next__(), i.send(None)))(g()))
    """,
    "a_cursor_over_a_sized_source_says_what_it_has_left": """
        # `__length_hint__` WAS MISSING FROM EVERY CURSOR. A list iterator, a
        # `reversed`, a str, a tuple, a dict, a set and a range iterator all
        # carry one in CPython and answered AttributeError here.
        #
        # ONLY THE CURSORS THAT WALK A SIZED SOURCE HAVE ONE, which is
        # CPython's line and not an approximation of it: `map`, `filter`,
        # `enumerate`, `zip` and a `callable_iterator` have NO length hint,
        # because those are the lazy modes and what they have left is not a
        # question their source can answer.
        #
        # AN ESTIMATE AND NOT A PROMISE. The source may grow or shrink under
        # the walk and CPython's answer goes stale the same way; what it must
        # never be is negative, which `operator.length_hint` raises on.
        #
        # A REVERSED CURSOR COUNTS DOWN from where `reversed` started it, so
        # what it has left is its position plus one.
        def show(label, f):
            try:
                print(label, f())
            except Exception as e:
                print(label, e.__class__.__name__ + ":", e)

        it = iter([1, 2, 3])
        show("fresh", lambda: it.__length_hint__())
        show("after one", lambda: (next(it), it.__length_hint__())[1])
        show("after two", lambda: (next(it), it.__length_hint__())[1])
        show("exhausted", lambda: (next(it), it.__length_hint__())[1])
        show("past the end", lambda: it.__length_hint__())
        show("reversed", lambda: reversed([1, 2, 3]).__length_hint__())
        show("reversed stepped",
             lambda: (lambda r: (next(r), r.__length_hint__())[1])(
                 reversed([1, 2, 3])))
        show("tuple", lambda: iter((1, 2)).__length_hint__())
        show("str", lambda: iter("abc").__length_hint__())
        show("bytes", lambda: iter(b"abcd").__length_hint__())
        show("dict", lambda: iter({1: 2, 3: 4}).__length_hint__())
        show("set", lambda: iter({1}).__length_hint__())
        show("range", lambda: iter(range(5)).__length_hint__())
        show("empty", lambda: iter([]).__length_hint__())
        # AND THE LAZY MODES HAVE NONE.
        show("map", lambda: hasattr(map(str, [1]), "__length_hint__"))
        show("filter", lambda: hasattr(filter(None, [1]), "__length_hint__"))
        show("enumerate", lambda: hasattr(enumerate([1]), "__length_hint__"))
        show("zip", lambda: hasattr(zip([1], [2]), "__length_hint__"))
        show("callable", lambda: hasattr(iter(lambda: 0, 1),
                                         "__length_hint__"))
        show("generator", lambda: hasattr((x for x in [1]),
                                          "__length_hint__"))
        # A SOURCE THAT SHRANK UNDER THE WALK never answers a negative.
        xs = [1, 2, 3, 4]
        walk = iter(xs)
        next(walk)
        next(walk)
        del xs[1:]
        show("source shrank", lambda: walk.__length_hint__())
    """,
    "a_type_object_has_its_methods_without_being_named": """
        # `type(x)` ANSWERED A TYPE OBJECT WITH NO METHODS unless the program
        # happened to MENTION that type's name somewhere:
        #
        #     L = type([1])
        #     hasattr(L, "append")      # False, against CPython's True
        #
        #     _x = list                 # one unused line, anywhere
        #     L = type([1])
        #     hasattr(L, "append")      # True
        #
        # Action at a distance: an unrelated statement elsewhere in the module
        # decided what an unrelated expression answered. The canonical thunk
        # -- which is what carries a builtin type's methods -- was registered
        # only for the type names appearing as an `ast.Name` in the source, on
        # the reasoning that a program which never writes `int` needs no thunk
        # for it. The name and the repr survived that; the METHODS did not,
        # and what a program will ask `type()` about is not knowable from the
        # names it writes.
        #
        # NOT ONE BUILTIN TYPE IS NAMED BELOW, which is the whole point of the
        # test: every type here is reached through `type(...)` alone. Writing
        # `list` once anywhere would register the thunk and prove nothing.
        def show(label, f):
            try:
                print(label, f())
            except Exception as e:
                print(label, e.__class__.__name__ + ":", e)

        L = type([1])
        S = type("a")
        D = type({1: 2})
        T = type((1,))
        I = type(1)
        F = type(1.5)
        B = type(b"a")
        E = type({1})

        show("list name", lambda: L.__name__)
        show("list append", lambda: hasattr(L, "append"))
        show("list len", lambda: hasattr(L, "__len__"))
        show("list unbound len", lambda: L.__len__([1, 2, 3]))
        show("str upper", lambda: S.upper("ab"))
        show("str unbound len", lambda: S.__len__("abc"))
        show("str join", lambda: S.join("-", ["a", "b"]))
        show("dict keys", lambda: sorted(D.keys({1: 2, 3: 4})))
        show("dict len", lambda: D.__len__({1: 2}))
        show("int bit_length", lambda: I.bit_length(5))
        show("float is_integer", lambda: F.is_integer(2.0))
        show("set union sorted", lambda: sorted(E.union({1}, {2})))
        # REACHED THROUGH `getattr` for the three whose WRITTEN spelling is a
        # separate fault: `T.count(...)` on a runtime type object lands on the
        # compile-time name dispatch and refuses, where `getattr` takes the
        # value path and answers. Filed on its own; what this program is
        # about is whether the methods are THERE.
        show("tuple count", lambda: getattr(T, "count")((1, 1, 2), 1))
        show("list count", lambda: getattr(L, "count")([1, 2, 1], 1))
        show("bytes hex", lambda: getattr(B, "hex")(b"ab"))
        # AND EACH IS THE SAME OBJECT the name would have given.
        show("interned", lambda: type([1]) is L)
        show("still named", lambda: [L.__name__, S.__name__, D.__name__])
        # `dir` OVER ONE LISTS WHAT IT REALLY HAS.
        show("list dir has append", lambda: "append" in dir(L))
        show("str dir has upper", lambda: "upper" in dir(S))
    """,
    "a_positional_only_keyword_is_refused_when_the_line_runs": """
        # TWO FAULTS IN THE COMPILE-TIME KEYWORD CHECK, and the written
        # spelling is the only one that met them -- the same calls made
        # through a value were right, because the RUNTIME binder knows both
        # rules.
        #
        # IT DID NOT KNOW ABOUT `/`. A keyword naming a positional-only
        # parameter read either as a slot already filled by position
        # (`multiple values for argument 'a'`) or, when some OTHER name did
        # not match, as an unexpected keyword -- the wrong complaint about
        # the wrong argument. CPython gathers every positional-only name into
        # ONE refusal, in DECLARATION order, and says it BEFORE anything else
        # about the keywords: `pos(a=1, zz=2)` names `a` and never mentions
        # `zz`.
        #
        # AND IT WAS AN ERROR, not a warning. Python's answer is a TypeError
        # a program may CATCH, so a module whose wrong call sits on a branch
        # nothing takes compiles under CPython -- and did not compile here at
        # all. The frontend already drew that line for a wrong argument
        # COUNT; the keywords now get the same treatment, so the call reaches
        # the runtime and raises when the line runs.
        def show(label, f):
            try:
                print(label, f())
            except TypeError as e:
                print(label, "TypeError:", e)

        def pos(a, b, /, c):
            return (a, b, c)

        def poskw(a, b, /, **kw):
            return (a, b, kw)

        def pos1(a, /, b):
            return (a, b)

        def plain(a, b):
            return (a, b)

        # EVERY POSITIONAL-ONLY NAME, GATHERED AND IN DECLARATION ORDER.
        show("all by name", lambda: pos(a=1, b=2, c=3))
        show("call order reversed", lambda: pos(b=2, a=1, c=3))
        show("one of them", lambda: pos(1, 2, a=9, c=3))
        # AND AHEAD OF THE UNKNOWN NAME.
        show("alongside an unknown", lambda: pos(a=1, zz=2, c=3))
        # WHICH IS STILL WHAT IS SAID WHEN NO POSITIONAL-ONLY IS NAMED.
        show("unknown alone", lambda: pos(1, 2, 3, zz=4))
        show("unknown, no slash", lambda: plain(1, zz=2))
        show("multiple values", lambda: plain(1, a=2))
        # A `**kw` TAKES THEM IN rather than refusing.
        show("kwrest takes them", lambda: poskw(a=1, b=2))
        show("kwrest keeps its own", lambda: poskw(1, 2, a=1))
        show("one positional-only", lambda: pos1(a=1, b=2))
        # AND A CALL THAT FITS STILL ANSWERS.
        show("fits", lambda: pos(1, 2, 3))
        show("fits by name", lambda: pos(1, 2, c=3))
        show("fits with rest", lambda: poskw(1, 2, z=3))
    """,
    "fstrings": """
        n = 42
        s = 'ab'
        print(f'n={n} s={s}')
        print(f'{n}', f'{s!r}', f'', f'no interp')
        print(f'{n + 1} {[1, 2]}')
    """,
    # AN ALIAS IS ITERABLE, and yields exactly one item: itself with the star
    # PEP 646 spells unpacking with. That one fact is the whole reason
    # `1 in list[int]` answers False in CPython instead of raising, which is
    # what it used to do here -- `in` walks that item and does not match it.
    #
    # THE STARRED FORM IS A DISTINCT VALUE, not a rendering trick: it compares
    # unequal to the alias it came from and hashes apart from it, which is
    # what `len({ga, starred})` holds this to. Answering False by fiat would
    # have passed the membership line and failed every line under it.
    "generic_alias_is_iterable": """
        ga = list[int]
        print(1 in ga, int in ga, ga in ga)
        starred = list(ga)[0]
        print(repr(starred), starred in ga, starred == list(ga)[0])
        print(starred == ga, len({ga, starred}))
        d = dict[str, int]
        print(list(d), str in d)
        print([repr(x) for x in ga])
        got = []
        for one in dict[str, bytes]:
            got.append(repr(one))
        print(got)
    """,
    # `f(*x)` OWES A DIFFERENT REFUSAL from the one iteration raises. The
    # plain `extend` says only what the argument is -- `'NI' object is not
    # iterable` -- where CPython names the FUNCTION and what the position
    # expected. The call site is the only place that knows which function was
    # being called.
    #
    # THE MODULE IS PART OF THE NAME, and only for a function the program
    # wrote: `__main__.f()` against `print()`. That is the same rule
    # `__module__` answers by, which is why this checks a def, a method, a
    # class and a builtin rather than one of them.
    #
    # A `__iter__` THAT RAISES TypeError IS NOT REWORDED -- CPython
    # propagates the program's own complaint, so "try it and reword the
    # failure" would have swallowed it. The question is asked structurally
    # instead, which is what `apy_can_iterate` is for, and the
    # `__getitem__`-only line holds that predicate to the older protocol.
    "star_splat_names_the_callee": """
        class NI:
            pass
        def f(*a):
            return len(a)
        class C:
            def m(self, *a):
                return len(a)
            def __init__(self, *a):
                pass
        class Raises:
            def __iter__(self):
                raise TypeError("my own complaint")
        class Getitem:
            def __getitem__(self, i):
                raise IndexError
        def show(lbl, fn):
            try:
                fn()
            except Exception as e:
                print(lbl, "|", type(e).__name__, "|", e)
            else:
                print(lbl, "| ok")
        show("plain def", lambda: f(*NI()))
        show("method", lambda: C().m(*NI()))
        show("class", lambda: C(*NI()))
        show("builtin", lambda: print(*NI()))
        show("int", lambda: f(*3))
        show("none", lambda: f(*None))
        show("user TypeError propagates", lambda: f(*Raises()))
        show("old protocol is iterable", lambda: f(*Getitem()))
        show("a real iterable still works", lambda: f(*[1, 2]))
        show("a generator still works", lambda: f(*(x for x in [1])))
        show("a string still works", lambda: f(*"ab"))
        show("a range still works", lambda: f(*range(2)))
        show("a dict still works", lambda: f(*{"a": 1}))
    """,
    # `list(x)` AND `sorted(x)` ASK THE ARGUMENT HOW LONG IT WILL BE before
    # they drain it, so they can size the result in one allocation. The
    # answer is thrown away -- and the CALL is still observable, because a
    # class that writes `__len__` or `__length_hint__` prints from it.
    #
    # WHICH SPELLINGS ASK IS MEASURED, not reasoned about -- the first
    # attempt put the ask in the shared iteration funnel and was caught
    # running a line of the program CPython never runs. Against 3.14 the
    # askers are: `list(x)`, `bytes(x)`, `sorted(x)` with a key or without,
    # `xs.extend(x)` on a list and on a bytearray, `xs += x` on a list, and
    # a starred display -- `[*x]` AND `(*x,)`, the tuple one because CPython
    # builds a tuple display by extending a list and converting it.
    #
    # AND WHICH DO NOT, which is the half that makes the list above mean
    # anything: `tuple(x)`, `set(x)`, `frozenset(x)`, `bytearray(x)`,
    # `{*x}`, `s.update(x)`, a comprehension, `f(*x)`, unpacking, `in`,
    # `sum`, `max`, `join` and a plain `for`.
    #
    # TWO VERBS, IN ORDER, AND ONLY ONE OF THEM: `__len__` first, and
    # `__length_hint__` only when there is no `__len__`. A class with both
    # sees exactly one call.
    #
    # A TypeError FROM `__len__` IS SWALLOWED and `__length_hint__` tried in
    # its place, which is how a type says "I have no length"; any other
    # exception is the program's own and travels.
    #
    # THE VALUE FORMS ARE HERE TOO. `f = list`, `g = sorted` and `h = bytes`
    # reach the runtime by another road -- the type object's constructor and
    # the keyword-taking thunk -- and each had to be given the ask
    # separately.
    "list_asks_for_a_length_hint": """
        class Base:
            def __init__(self):
                self.n = 0
            def __iter__(self):
                return self
            def __next__(self):
                self.n += 1
                if self.n > 2:
                    raise StopIteration
                return self.n
        class Neither(Base):
            pass
        class HasLen(Base):
            def __len__(self):
                print("  __len__")
                return 2
        class HasHint(Base):
            def __length_hint__(self):
                print("  __length_hint__")
                return 2
        class HasBoth(Base):
            def __len__(self):
                print("  __len__")
                return 2
            def __length_hint__(self):
                print("  __length_hint__")
                return 2
        class Refuses(Base):
            def __len__(self):
                print("  __len__ refuses")
                raise TypeError("no length")
            def __length_hint__(self):
                print("  __length_hint__")
                return 2
        class Breaks(Base):
            def __len__(self):
                raise ValueError("broken")
        def takes(*a):
            return len(a)
        for name, C in (("neither", Neither), ("len", HasLen),
                        ("hint", HasHint), ("both", HasBoth),
                        ("refuses", Refuses)):
            print(name)
            print(" list:", list(C()))
            print(" bytes:", bytes(C()))
            print(" sorted:", sorted(C()))
            print(" sorted key:", sorted(C(), key=lambda v: -v))
            print(" star list:", [*C()])
            print(" star tuple:", (*C(),))
            xs = []
            xs.extend(C())
            print(" extend:", xs)
            ys = []
            ys += C()
            print(" iadd:", ys)
            ba = bytearray()
            ba.extend(C())
            print(" bytearray extend:", list(ba))
            print(" tuple:", tuple(C()))
            print(" bytearray:", list(bytearray(C())))
            print(" set:", len(set(C())))
            print(" frozenset:", len(frozenset(C())))
            print(" star set:", len({*C()}))
            zs = set()
            zs.update(C())
            print(" update:", len(zs))
            print(" comp:", [y for y in C()])
            print(" splat call:", takes(*C()))
            print(" sum:", sum(C()))
            print(" max:", max(C()))
            print(" in:", 9 in C())
            print(" join:", "-".join(str(v) for v in C()))
            for y in C():
                pass
            print(" for: done")
        f = list
        g = sorted
        h = bytes
        print("value list:", f(HasLen()))
        print("value sorted:", g(HasLen()))
        print("value bytes:", h(HasLen()))
        try:
            list(Breaks())
        except ValueError as e:
            print("other exception travels:", e)
    """,
    # A MUTABLE ANSWER MUST NEVER BE A SHARED CELL, which is the hazard the
    # shared empty and one-byte values created and the one thing a program
    # can see go wrong everywhere at once. `bytearray.fromhex("41")` used to
    # build the SHARED `b"A"` and then write "mutable" onto it, so every
    # later one-byte bytes in the program -- literals included -- was a
    # bytearray. The order below is the test: the same expression before and
    # after, and a literal at the end that nothing in between went near.
    "a_bytearray_does_not_poison_the_shared_bytes": """
        print("fromhex before:", bytes.fromhex("41"),
              type(bytes.fromhex("41")).__name__)
        made = bytearray.fromhex("41")
        print("the bytearray:", made, type(made).__name__)
        print("fromhex after:", bytes.fromhex("41"),
              type(bytes.fromhex("41")).__name__)
        print("a literal:", b"A", type(b"A").__name__)
        print("the empty one:", bytearray.fromhex(""),
              bytes.fromhex(""), b"", type(b"").__name__)

        # AND THE RECEIVER DECIDES, which is the behaviour the re-tagging was
        # there to produce: reached through a bytearray the answer is one,
        # reached through bytes or the type it is not.
        print("through a value:", type(bytearray(b"x").fromhex("41")).__name__,
              type(b"x".fromhex("41")).__name__)
        print("still a literal:", b"A", type(b"A").__name__)

        # THE SAME SHAPE FOR THE OTHER CONSTRUCTOR that takes a shared cell.
        one = bytearray(b"A")
        one.append(66)
        print("appended:", one, b"A", type(b"A").__name__)
    """,
    # A METHOD WITH NOTHING TO DO MAY HAND THE RECEIVER BACK ONLY WHEN THE
    # RECEIVER CANNOT BE WRITTEN INTO. `strip` and `replace` were guarded and
    # these eight were not, so `bytearray(b"abc").center(3)` WAS the receiver:
    # two live names for one buffer, and the answer changed under the program
    # at the next `ba[0] = ...`. Nothing was raised at the call that caused
    # it, which is why the write below is part of the test and not just the
    # `is`.
    #
    # BOTH DIRECTIONS ARE HERE, because over-copying is the other way to get
    # this wrong and it is just as visible: an immutable receiver must STILL
    # come straight back, or `'abc'.center(3) is 'abc'` stops being True and
    # CPython says it is.
    "a_bytearray_receiver_is_never_handed_back": """
        def watch(label, make):
            ba = bytearray(b"abc")
            got = make(ba)
            same = got is ba
            ba[0] = 122
            print(label, same, bytes(got))

        watch("center      ", lambda ba: ba.center(3))
        watch("ljust       ", lambda ba: ba.ljust(3))
        watch("rjust       ", lambda ba: ba.rjust(3))
        watch("center fill ", lambda ba: ba.center(3, b"."))
        watch("ljust fill  ", lambda ba: ba.ljust(3, b"."))
        watch("rjust fill  ", lambda ba: ba.rjust(3, b"."))
        watch("zfill       ", lambda ba: ba.zfill(3))
        watch("zfill under ", lambda ba: ba.zfill(0))
        watch("removeprefix", lambda ba: ba.removeprefix(b"z"))
        watch("removeempty ", lambda ba: ba.removeprefix(b""))
        watch("removesuffix", lambda ba: ba.removesuffix(b"z"))
        watch("partition   ", lambda ba: ba.partition(b"z")[0])
        watch("rpartition  ", lambda ba: ba.rpartition(b"z")[2])
        watch("strip       ", lambda ba: ba.strip())
        watch("replace     ", lambda ba: ba.replace(b"z", b"y"))
        watch("full slice  ", lambda ba: ba[:])

        # THE COPY IS OF THE RECEIVER'S OWN KIND: a bytearray in, a bytearray
        # out. Handing back a plain `bytes` would be a second bug wearing the
        # first one's fix, and the program could no longer write into it.
        kept = bytearray(b"abc").center(3)
        print("kind:", type(kept).__name__)
        kept[1] = 122
        print("writable:", bytes(kept))

        # AND AN IMMUTABLE RECEIVER STILL COMES STRAIGHT BACK.
        s = "abc"
        b = b"abc"
        print("str  ", s.center(3) is s, s.ljust(3) is s, s.rjust(3) is s,
              s.zfill(3) is s, s.removeprefix("z") is s,
              s.removesuffix("z") is s, s.partition("z")[0] is s,
              s.rpartition("z")[2] is s)
        print("bytes", b.center(3) is b, b.ljust(3) is b, b.rjust(3) is b,
              b.zfill(3) is b, b.removeprefix(b"z") is b,
              b.removesuffix(b"z") is b, b.partition(b"z")[0] is b,
              b.rpartition(b"z")[2] is b)
    """,
}


def cpython(src: str) -> list[str]:
    """What CPython prints. The oracle, with no compensation of any kind --
    every divergence the runtime has is one this corpus avoids rather than one
    the comparison papers over.

    REAL STDOUT, through the real `print`. A stub that joined its arguments
    with a space stood in for it once, and it was not the oracle it claimed to
    be: it ignored `sep` and `end`, so a program using either was compared
    against an answer CPython does not give. Redirecting the stream costs
    nothing and is the thing itself.
    """
    stream = StringIO()
    with redirect_stdout(stream):
        # `dont_inherit=True`, because THIS MODULE says `from __future__
        # import annotations` and `compile` inherits future flags from its
        # caller. Every corpus program was compiled in a language mode a
        # script does not use -- annotations stringified -- so the oracle
        # disagreed with CPython about any program that reads
        # `__annotations__`, and it disagreed silently.
        exec(compile(src, "<case>", "exec", dont_inherit=True),
             {"__name__": "__main__"})
    return stream.getvalue().split("\n")[:-1]


def compile_it(src: str, tmp_path: Path, optimise: bool = False):
    path = tmp_path / "prog.py"
    path.write_text(src, encoding="utf-8")
    sink = DiagnosticSink()
    result = compile_source(Options(source=path, optimise=optimise), sink)
    assert result.ok, [d.message for d in sink.diagnostics]
    return result.module


@harness.cases("name", sorted(PROGRAMS))
class TestEveryPathAgrees:
    def source(self, name: str) -> str:
        return textwrap.dedent(PROGRAMS[name]).strip() + "\n"

    def test_the_interpreter_matches_cpython(self, name, tmp_path):
        src = self.source(name)
        out = StringIO()
        Interpreter(compile_it(src, tmp_path), out=out).run("main")
        assert out.getvalue().split("\n")[:-1] == cpython(src)

    def test_the_interpreter_matches_cpython_optimised(self, name, tmp_path):
        """The same, on optimised IR. This is what catches a pass that changes
        meaning -- which no amount of testing a pass in isolation will find,
        because a pass can be individually correct and wrong in combination."""
        src = self.source(name)
        out = StringIO()
        Interpreter(compile_it(src, tmp_path, optimise=True), out=out).run("main")
        assert out.getvalue().split("\n")[:-1] == cpython(src)

    @harness.needs("cc")
    def test_the_c_backend_matches_cpython(self, name, tmp_path):
        from uasm.backend import get, load_builtin
        from uasm.target import get as get_target
        load_builtin()
        src = self.source(name)
        module = compile_it(src, tmp_path)
        c_file = tmp_path / "out.c"
        c_file.write_bytes(get("c").emit(module, get_target("c"))["out.c"])
        exe = tmp_path / "out.exe"
        # -lm -ldl: the object runtime calls libm (floor, fmod, ...) and libdl
        # unconditionally, needed on every ELF host -- see toolchains.py.
        system_libs = [] if sys.platform == "win32" else ["-lm", "-ldl"]
        built = subprocess.run([HAS_CC, str(c_file), "-o", str(exe), *system_libs],
                               capture_output=True, text=True)
        assert built.returncode == 0, built.stderr
        # UTF-8, NOT THE LOCALE ENCODING. A str is stored as UTF-8 by this
        # runtime and the compiled program writes those bytes straight out;
        # decoding them as cp1252 turns every non-ASCII character into
        # mojibake and compares it against CPython's correct output. The
        # conformance shim had the same bug, and it failed cases the compiler
        # was getting exactly right.
        ran = subprocess.run([str(exe)], capture_output=True, text=True,
                             encoding="utf-8")
        assert ran.stdout.split("\n")[:-1] == cpython(src)
