"""The object runtime against CPython, over generated cases.

WHY THIS EXISTS. The conformance suite is at 1668/1668 -- full marks -- and it
still cannot see the class of bug this finds. Five real divergences turned up
in one session purely by accident, while porting unrelated functions: `str()`
of the OSError family, `f"{x!a}"`, `"{!a}".format()`, a descriptor's
`__delete__` on the interpreter, and `math.floor(True)` answering three
different things on three paths. Not one is in the suite, because a construct
the suite does not write is a construct the suite cannot find.

A SATURATED SCORE IS NOT A MEASUREMENT. It says the cases pass; it says
nothing about the cases nobody wrote. This writes them by the thousand.

FOUR PATHS, FOUR RUNS, NOT FOUR RUNS PER CASE. Every case goes into ONE
program, so the cost is four executions and two `gcc` invocations however many
cases there are. That is what makes a sweep of this size affordable at all.

  cpython      the reference
  interp       `uasm run` -- the host object runtime
  ir           compiled, object runtime in the machine subset
  c            compiled, `--object-runtime c`

WHAT A DISAGREEMENT MEANS depends on WHO disagrees, and the tool says so:

  ir != c                 the port is unfaithful -- an IR half does not match
                          the C body it replaced. This is the one that means
                          somebody's change is wrong RIGHT NOW.
  ir == c != interp       the host and the compiled runtime have drifted.
  all three != cpython    a real conformance gap, and an old one: every path
                          agrees, so nothing recent caused it.

EVERY CASE IS WRAPPED so that a raising case is a RESULT and not the end of
the run. `repr` of the exception type and its message is the answer, which is
what makes an error message a thing this can compare at all -- three of the
divergences found by hand were message wording.

Usage:
    py -3.14 tools/objects_diff.py                  # the whole sweep
    py -3.14 tools/objects_diff.py --only str       # one group
    py -3.14 tools/objects_diff.py --list           # what groups exist
"""
# FOUR GROUPS WERE ADDED AFTER THE FIRST SWEEP SATURATED, and each exists
# because a whole SHAPE of call was missing rather than a few cases:
#
#   keywords            Every case above is written POSITIONALLY, and a
#                       keyword is a different lowering path -- so 9,339
#                       generated expressions found five real divergences and
#                       none of the eight keyword bugs. 26 distinct shapes on
#                       the first run.
#   bytes               The bytes family had no group at all, and it shares
#                       its implementation with str -- which turns out to mean
#                       it shares str's UTF-8 view of a receiver that has no
#                       characters in it.
#   bytearray-mutation  An in-place method answers None and puts the change in
#                       the RECEIVER, so a group printing only the result
#                       cannot see it. Both are printed, and for the operators
#                       an ALIAS is printed too: that is the whole question.
#   attributes          `x.upper` and `getattr(x, "upper")` are different
#                       lowerings and only the first was generated. 167
#                       divergences of 650.
from __future__ import annotations

import argparse
import itertools
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# THIS TOOL PRINTS PYTHON SOURCE BACK AT YOU, and the cases are deliberately
# full of non-ASCII -- a Turkish dotted capital I is exactly the sort of case
# worth generating. On a cp1252 console that is a UnicodeEncodeError raised
# while REPORTING A FINDING, which loses the finding and the run with it.
# Escaping rather than failing: a mangled character in a report is a nuisance,
# a dead reporter is a lost measurement.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

# ---------------------------------------------------------------------------
# The cases.
#
# EACH GROUP IS A LIST OF EXPRESSION STRINGS, and they are built by product
# rather than written out: the point is to cover combinations nobody would
# think to write by hand. `str.center` against a two-character fill and a
# width shorter than the string is not a case anyone writes deliberately.
#
# AN EXPRESSION AND NOT A STATEMENT, so that the answer is a value this can
# print and compare. A few groups need a setup line; those spell it with `;`.
# ---------------------------------------------------------------------------

STRINGS = ["", "a", "ab", "abc", "Hello World", "  pad  ", "MiXeD",
           "caf\u00e9", "\u4e2d\u6587", "\U0001F600", "a\tb", "a\nb",
           "\u00a0", "123", "12.5", "-7", "0", "aaa", "abcabc", "'q'",
           '"d"', "\\", "\u0130", "\u00df", "\u2003x"]
INTS = [0, 1, -1, 2, 7, -7, 255, 256, 10 ** 18, -(10 ** 18),
        2 ** 63 - 1, -(2 ** 63), 2 ** 63, 2 ** 100, -(2 ** 100)]
FLOATS = [0.0, -0.0, 1.0, -1.0, 0.5, 1.5, -1.5, 2.5, -2.5, 1e17, 1e19,
          2.0 ** 63, -(2.0 ** 63), 2.0 ** 100, 1e308, 5e-324]
SEQS = ["[]", "[1]", "[1, 2, 3]", "[1, 'a', None]", "()", "(1,)", "(1, 2)",
        "{}", "{'a': 1}", "{1: 'x', 2: 'y'}", "set()", "{1}", "{1, 2, 3}",
        "frozenset()", "frozenset({1, 2})", "b''", "b'ab'",
        "bytearray(b'ab')", "range(3)", "range(0, 10, 3)"]

NO_ARG_STR = ["upper", "lower", "title", "capitalize", "swapcase", "casefold",
              "strip", "lstrip", "rstrip", "split", "isalpha", "isdigit",
              "isdecimal", "isnumeric", "isalnum", "isspace", "isprintable",
              "isascii", "islower", "isupper", "istitle", "isidentifier",
              "encode", "splitlines"]
ONE_ARG_STR = ["count", "find", "rfind", "index", "rindex", "startswith",
               "endswith", "split", "rsplit", "partition", "rpartition",
               "join", "lstrip", "rstrip", "strip", "removeprefix",
               "removesuffix", "center", "ljust", "rjust", "zfill"]


def _lit(v) -> str:
    return repr(v)


def group_str_unary():
    return [f"{_lit(s)}.{m}()" for s in STRINGS for m in NO_ARG_STR]


def group_str_binary():
    args = ["'a'", "'ab'", "''", "'\\u00e9'", "0", "1", "3", "10", "-1"]
    return [f"{_lit(s)}.{m}({a})"
            for s in STRINGS[:12] for m in ONE_ARG_STR for a in args]


def group_str_slice():
    idx = ["None", "0", "1", "2", "-1", "-2", "10", "-10"]
    return [f"{_lit(s)}[{a}:{b}:{c}]"
            for s in STRINGS[:10]
            for a, b, c in itertools.product(idx[:6], idx[:6], ["None", "2", "-1"])]


def group_str_repr():
    return ([f"repr({_lit(s)})" for s in STRINGS]
            + [f"ascii({_lit(s)})" for s in STRINGS]
            + [f"str({_lit(s)})" for s in STRINGS]
            + [f"format({_lit(s)}, {_lit(spec)})"
               for s in STRINGS[:8]
               for spec in ["", ">10", "<10", "^10", "*^12", ".2", "!r"]]
            + [f"f'{{{_lit(s)}!{c}}}'" for s in STRINGS for c in "ars"])


def group_int_arith():
    ops = ["+", "-", "*", "//", "%", "&", "|", "^", "<<", ">>",
           "<", "<=", "==", "!=", ">", ">="]
    out = []
    for a in INTS:
        for b in INTS:
            for op in ops:
                if op in ("<<", ">>") and (b < 0 or abs(b) > 200):
                    continue
                out.append(f"{_lit(a)} {op} {_lit(b)}")
    return out


def group_int_unary():
    fns = ["abs", "repr", "str", "hex", "oct", "bin", "bool", "float",
           "hash", "len(str(%s))" % "{}"]
    out = []
    for v in INTS:
        for f in fns[:-1]:
            out.append(f"{f}({_lit(v)})")
        out.append(f"({_lit(v)}).bit_length()")
        out.append(f"({_lit(v)}).bit_count()")
        out.append(f"({_lit(v)}).to_bytes(16, 'little', signed=True)")
        out.append(f"(-{_lit(v)} if {_lit(v)} else 0)")
        out.append(f"~({_lit(v)})")
    return out


def group_float():
    out = []
    for v in FLOATS:
        for f in ["abs", "repr", "str", "bool", "int"]:
            out.append(f"{f}({_lit(v)})")
        out.append(f"({_lit(v)}).is_integer()")
        out.append(f"({_lit(v)}).as_integer_ratio()")
        for m in ["floor", "ceil", "trunc", "fabs", "isnan", "isinf"]:
            out.append(f"__import__('math').{m}({_lit(v)})")
    for a in FLOATS[:8]:
        for b in FLOATS[:8]:
            for op in ["+", "-", "*", "<", "==", ">"]:
                out.append(f"{_lit(a)} {op} {_lit(b)}")
    return out


def group_containers():
    out = []
    for s in SEQS:
        for f in ["len", "repr", "str", "bool", "list", "tuple", "sorted"]:
            out.append(f"{f}({s})")
        out.append(f"[x for x in {s}]")
        out.append(f"list(reversed(list({s})))")
        for probe in ["1", "'a'", "None"]:
            out.append(f"{probe} in {s}")
    return out


def group_container_methods():
    out = []
    for s in ["[3, 1, 2]", "[1]", "[]"]:
        for m in ["pop()", "pop(0)", "pop(5)", "sort()", "reverse()",
                  "index(1)", "count(1)", "remove(1)", "clear()",
                  "insert(0, 9)", "append(9)", "extend([7])", "copy()"]:
            out.append(f"(lambda v: (v.{m}, v))({s})")
    for d in ["{'a': 1}", "{}", "{1: 2, 3: 4}"]:
        for m in ["keys()", "values()", "items()", "get('a')", "get('z', 5)",
                  "pop('a', None)", "popitem()", "copy()", "setdefault('b', 2)"]:
            out.append(f"(lambda v: (str(v.{m}), v))({d})")
    for st in ["{1, 2}", "set()", "{1}"]:
        for m in ["union({3})", "intersection({1})", "difference({1})",
                  "symmetric_difference({2})", "issubset({1, 2})",
                  "issuperset({1})", "isdisjoint({9})", "pop()", "copy()"]:
            out.append(f"(lambda v: (str(v.{m}), sorted(v)))({st})")
    return out


def group_exceptions():
    excs = ["ValueError('bad')", "ValueError()", "ValueError(None)",
            "ValueError('a', 'b')", "KeyError('k')", "KeyError('')",
            "OSError(2, 'No such file')", "OSError(2, 'm', 'f.txt')",
            "OSError('plain')", "OSError()", "PermissionError(13, 'nope')",
            "TypeError('x')", "IndexError('i')", "StopIteration()",
            "ZeroDivisionError('division by zero')",
            "ExceptionGroup('g', [ValueError('v')])"]
    out = []
    for e in excs:
        out.append(f"repr({e})")
        out.append(f"str({e})")
        out.append(f"({e}).args")
        out.append(f"repr([{e}])")
    raisers = ["1/0", "[][3]", "{}['k']", "int('zz')", "'a'+1", "len(5)",
               "(1,2)[9]", "[].pop()", "{}.popitem()", "set().pop()",
               "'abc'.index('z')", "b'a'.decode('utf-8', 'strict')",
               "[1,2].remove(9)", "float('zz')", "'a'.center(2**100)",
               "[1,2][2**100]", "[1,2]*(2**100)"]
    for r in raisers:
        out.append(r)
    return out


def group_conversions():
    out = []
    for s in ["'12'", "'-7'", "'0'", "' 5 '", "'0x1f'", "'zz'", "''",
              "'1_0'", "'+3'", "'1.5'", "'inf'", "'nan'", "'1e3'"]:
        for f in ["int", "float"]:
            out.append(f"{f}({s})")
        out.append(f"int({s}, 16)")
        out.append(f"int({s}, 0)")
    for v in ["1", "1.5", "'a'", "None", "True", "[1]", "b'a'"]:
        for f in ["bool", "repr", "str", "hash", "type", "id(0)*0 or type"]:
            if f.startswith("id"):
                continue
            out.append(f"{f}({v})" if f != "type" else f"type({v}).__name__")
    return out


def group_translate():
    tables = ["{}", "{ord('a'): 'A'}", "{ord('a'): None}",
              "{ord('a'): 'AAA'}", "{ord('\\u00e9'): 'e'}",
              "{ord('a'): ord('b')}", "str.maketrans('abc', 'xyz')",
              "str.maketrans('', '', 'lo')"]
    return [f"{_lit(s)}.translate({t})" for s in STRINGS[:14] for t in tables]


def group_keywords():
    """EVERY BUILTIN METHOD THAT TAKES A KEYWORD, written as one.

    THE GAP THIS FILLS. 9,339 generated expressions found five real
    divergences and none of the keyword bugs, for one reason: every case
    above is written POSITIONALLY, and a keyword is a different lowering
    path. `"a,b,c".split(",", maxsplit=1)` and `"a,b,c".split(",", 1)` reach
    the same runtime symbol by different roads, and only one of the roads was
    ever generated.

    THE TABLE IS THE ORACLE FOR WHAT TO WRITE and CPython for what it should
    answer -- so a method that grows a parameter is covered the day
    `METHOD_PARAMS` learns about it.

    FOUR SHAPES PER PARAMETER, because each is a different failure:
    the keyword alone, the keyword after a positional, the keyword given
    twice, and a keyword the method does not have.
    """
    sys.path.insert(0, str(SRC))
    try:
        from uasm.frontends.python.methods import (
            METHOD_PARAMS, POSITIONAL_ONLY)
    finally:
        sys.path.remove(str(SRC))
    #: A receiver each method can be called on, and arguments its positional
    #: parameters accept. Written out because a keyword call has to be
    #: WELL-FORMED to reach the interesting part -- a generated argument of
    #: the wrong type fails before the keyword matters.
    subjects = {
        "split": ("'a,b,c'", ["','", "None"]),
        "rsplit": ("'a,b,c'", ["','", "None"]),
        "splitlines": ("'a\\nb\\nc'", ["True", "False"]),
        "replace": ("'aaa'", ["'a'", "'b'"]),
        "expandtabs": ("'a\\tb'", ["4"]),
        "encode": ("'caf\\u00e9'", ["'utf-8'", "'strict'"]),
        "decode": ("b'caf\\xc3\\xa9'", ["'utf-8'", "'strict'"]),
        "to_bytes": ("(258)", ["4", "'little'"]),
        "translate": ("b'abc'", ["bytes.maketrans(b'ab', b'xy')"]),
        "sort": ("[3, 1, 2]", ["None", "False"]),
    }
    values = {"sep": "','", "maxsplit": "1", "keepends": "True",
              "count": "2", "tabsize": "4", "encoding": "'utf-8'",
              "errors": "'replace'", "length": "4", "byteorder": "'little'",
              "signed": "True", "delete": "b'c'", "key": "None",
              "reverse": "True"}
    out = []
    for name, params in sorted(METHOD_PARAMS.items()):
        recv, args = subjects[name]
        nameable = [p for p, _ in params if p is not POSITIONAL_ONLY]
        for i, (param, _) in enumerate(params):
            if param is POSITIONAL_ONLY:
                continue
            value = values.get(param, "1")
            # The keyword ALONE, with the parameters before it left out.
            out.append(f"{recv}.{name}({param}={value})")
            # And after each number of positionals it can legally follow.
            for k in range(min(i, len(args)) + 1):
                given = ", ".join(args[:k])
                lead = given + ", " if given else ""
                out.append(f"{recv}.{name}({lead}{param}={value})")
            # GIVEN BY NAME AND BY POSITION AT ONCE, which CPython refuses
            # with a message naming the slot.
            if i < len(args):
                given = ", ".join(args[:i + 1])
                out.append(f"{recv}.{name}({given}, {param}={value})")
        # A KEYWORD THE METHOD DOES NOT HAVE, and one it does spelled wrong.
        out.append(f"{recv}.{name}(nosuch=1)")
        if nameable:
            out.append(f"{recv}.{name}({nameable[0].upper()}=1)")
        # AND A `**` MAPPING, empty and not.
        out.append(f"{recv}.{name}(**{{}})")
        if nameable:
            out.append(f"{recv}.{name}(**{{{nameable[0]!r}: "
                       f"{values.get(nameable[0], '1')}}})")
    return out


#: The bytes family, which had no group of its own and holds a third of the
#: divergences found by hand: `translate`, `maketrans`, `copy`, the mutable
#: sequence methods, `expandtabs`, and the retagging that makes a bytearray
#: method answer a bytearray.
BYTES_RECEIVERS = ["b''", "b'a'", "b'abc'", "b'a-b-c'", "b'  pad  '",
                   "b'MiXeD'", "b'\\xc3\\xa9'", "b'a\\tb'", "b'a\\nb'",
                   "bytearray(b'')", "bytearray(b'abc')",
                   "bytearray(b'a-b-c')", "bytearray(b'MiXeD')"]


def group_bytes():
    out = []
    for r in BYTES_RECEIVERS:
        for m in ["upper", "lower", "title", "capitalize", "swapcase",
                  "strip", "lstrip", "rstrip", "split", "splitlines",
                  "isalpha", "isdigit", "isalnum", "isspace", "isupper",
                  "islower", "istitle", "isascii", "hex", "expandtabs",
                  "decode", "copy"]:
            out.append(f"{r}.{m}()")
        for m, a in [("find", "b'b'"), ("rfind", "b'b'"), ("index", "b'b'"),
                     ("count", "b'b'"), ("startswith", "b'a'"),
                     ("endswith", "b'c'"), ("partition", "b'-'"),
                     ("rpartition", "b'-'"), ("split", "b'-'"),
                     ("join", "[b'x', b'y']"), ("center", "9"),
                     ("ljust", "9"), ("rjust", "9"), ("zfill", "9"),
                     ("removeprefix", "b'a'"), ("removesuffix", "b'c'"),
                     ("lstrip", "b'a'"), ("rstrip", "b'c'"),
                     ("strip", "b'ac'"), ("expandtabs", "4")]:
            out.append(f"{r}.{m}({a})")
        out.append(f"{r}.replace(b'a', b'z')")
        out.append(f"{r}.translate(bytes.maketrans(b'ab', b'xy'))")
        out.append(f"{r}.translate(None, b'a')")
        out.append(f"{r}.translate(bytes.maketrans(b'ab', b'xy'), b'c')")
        # THE KIND OF THE ANSWER, which is the half a value comparison hides:
        # a bytearray method answers a bytearray and `hex` answers a str.
        out.append(f"type({r}.upper()).__name__")
        out.append(f"type({r}.split(b'-')[0]).__name__")
        out.append(f"[{r} + b'z', b'z' + {r}, {r} * 2, 2 * {r}]")
        out.append(f"[len({r}), bool({r}), list({r})]")
        for probe in ["b'a'", "97", "b'zz'"]:
            out.append(f"{probe} in {r}")
    return out


def group_bytearray_mutation():
    """THE IN-PLACE HALF, which a value comparison cannot see on its own: the
    method answers None and the change is in the receiver, so both are
    printed."""
    out = []
    for m in ["append(122)", "append(b'z')", "append(300)", "append(True)",
              "extend(b'de')", "extend([1, 2])", "insert(0, 122)",
              "insert(99, 122)", "insert(-99, 122)", "insert(0, b'z')",
              "pop()", "pop(0)", "pop(-1)", "pop(9)", "remove(98)",
              "remove(122)", "remove(b'b')", "clear()", "reverse()",
              "copy()", "__iadd__(b'z')", "__imul__(2)"]:
        for start in ["b''", "b'abc'", "b'a'"]:
            out.append(f"(lambda v: (v.{m}, v))(bytearray({start}))")
    # AND THE OPERATORS, where the question is whether the ALIAS sees it.
    for recv in ["bytearray(b'ab')", "[1, 2]", "(1, 2)", "'ab'", "b'ab'"]:
        out.append(f"_iop({recv}, b'z' if isinstance({recv}, (bytes, "
                   f"bytearray)) else [9], '+')")
        for k in ["2", "0", "-1"]:
            out.append(f"_iop({recv}, {k}, '*')")
    for recv in ["{1, 2}", "frozenset({1})", "{1: 2}"]:
        for which in ["|", "&", "-", "^"]:
            other = "{3: 4}" if which == "|" else "{3}"
            out.append(f"_iop({recv}, {other} if {recv!r}.startswith('{{1:')"
                       f" else {{3}}, {which!r})")
    return out


def group_attributes():
    """`getattr(x, name)` OVER EVERY KIND AND EVERY METHOD NAME.

    The written form `x.upper` and the looked-up form `getattr(x, "upper")`
    are DIFFERENT LOWERINGS -- the first is a call the frontend recognises,
    the second goes through the runtime's attribute table -- and only the
    first was ever generated. Anything doing dynamic dispatch by method name
    takes the second.
    """
    kinds = ["'abc'", "b'abc'", "bytearray(b'abc')", "[1, 2]", "(1, 2)",
             "{'a': 1}", "{1, 2}", "frozenset({1})", "5", "1.5", "range(3)",
             "None", "1j"]
    names = ["upper", "append", "keys", "add", "index", "count", "copy",
             "find", "split", "strip", "join", "translate", "pop",
             "__class__", "__len__", "__iter__", "__contains__", "__add__",
             "__iadd__", "__eq__", "__hash__", "__getitem__", "__round__",
             "__sizeof__", "nosuch"]
    out = []
    for k in kinds:
        for n in names:
            out.append(f"hasattr({k}, {n!r})")
            out.append(f"type(getattr({k}, {n!r})).__name__")
    return out


def group_slices():
    return [f"slice({a}, {b}, {c}).indices({n})"
            for a in ["None", "0", "2", "-3", "100"]
            for b in ["None", "1", "5", "-2", "-100"]
            for c in ["None", "1", "2", "-1", "-3"]
            for n in [0, 1, 5, 10]]


GROUPS = {
    "str-unary": group_str_unary,
    "str-binary": group_str_binary,
    "str-slice": group_str_slice,
    "str-repr": group_str_repr,
    "int-arith": group_int_arith,
    "int-unary": group_int_unary,
    "float": group_float,
    "containers": group_containers,
    "container-methods": group_container_methods,
    "exceptions": group_exceptions,
    "conversions": group_conversions,
    "translate": group_translate,
    "slices": group_slices,
    "keywords": group_keywords,
    "bytes": group_bytes,
    "bytearray-mutation": group_bytearray_mutation,
    "attributes": group_attributes,
}


# ---------------------------------------------------------------------------
# Running.
# ---------------------------------------------------------------------------

PRELUDE = '''\
import sys
def _iop(v, o, which):
    """`v op= o` with the ALIAS kept, which is the whole question for the
    in-place operators: a list and a bytearray must change in place, so `w`
    sees it and `v is w` stays True; everything else rebinds. A STATEMENT and
    so a function -- `lambda: (v += o)` is not Python."""
    w = v
    if which == "+":
        v += o
    elif which == "*":
        v *= o
    elif which == "|":
        v |= o
    elif which == "&":
        v &= o
    elif which == "-":
        v -= o
    else:
        v ^= o
    return (v, w, v is w)
def _s(i, f):
    try:
        v = f()
    except BaseException as e:
        v = type(e).__name__ + ": " + str(e)
    try:
        print(i, repr(v))
    except BaseException as e:
        print(i, "<unprintable>", type(e).__name__)
'''


def build_program(cases) -> str:
    """One program holding every case, each printing an indexed line.

    A LAMBDA PER CASE, so a case that raises is caught and RECORDED rather
    than ending the run. The index is printed with it, because a case that
    prints nothing at all would otherwise shift every line after it and
    report the whole tail as divergent.
    """
    out = [PRELUDE]
    for i, c in enumerate(cases):
        out.append(f"_s({i}, lambda: ({c}))")
    return "\n".join(out) + "\n"


def run_cpython(path: pathlib.Path) -> list[str]:
    r = subprocess.run([sys.executable, str(path)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace",
                       env=_env())
    return r.stdout.splitlines()


def _env():
    import os
    e = dict(os.environ)
    e["PYTHONPATH"] = str(SRC)
    e["PYTHONIOENCODING"] = "utf-8"
    return e


def run_interp(path: pathlib.Path) -> list[str]:
    r = subprocess.run([sys.executable, "-m", "uasm", "run", str(path)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=_env())
    return [ln for ln in r.stdout.splitlines() if ln != "no diagnostics"]


def run_compiled(path: pathlib.Path, mode: str, work: pathlib.Path):
    c_file = work / f"prog_{mode}.c"
    exe = work / f"prog_{mode}.exe"
    build = subprocess.run(
        [sys.executable, "-m", "uasm", "build", str(path),
         "--backend", "c", "--object-runtime", mode, "-o", str(c_file),
         "--workdir", str(work / mode)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=_env())
    if build.returncode != 0:
        return None, (build.stdout + build.stderr)[-2000:]
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        return None, "no C compiler found (cc or gcc)"
    libs = [] if sys.platform == "win32" else ["-lm", "-ldl"]
    link = subprocess.run(
        [cc, str(c_file), "-o", str(exe), *libs],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if link.returncode != 0:
        return None, link.stderr[-2000:]
    r = subprocess.run([str(exe)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=_env())
    return r.stdout.splitlines(), None


def indexed(lines):
    """`{index: text}` from the printed lines, ignoring anything unindexed."""
    out = {}
    for ln in lines:
        head, _, rest = ln.partition(" ")
        if head.isdigit():
            out[int(head)] = rest
    return out


VERDICTS = [
    ("ir != c", "THE PORT IS UNFAITHFUL -- an IR half does not match the C "
                "body it replaced. Somebody's change is wrong right now."),
    ("compiled != interp", "the host and the compiled runtime have drifted."),
    ("all != cpython", "a conformance gap, and an old one: every path agrees "
                       "with every other, so nothing recent caused it."),
]


def classify(cp, it, ir, c):
    if ir != c:
        return "ir != c"
    if ir != it:
        return "compiled != interp"
    if ir != cp:
        return "all != cpython"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", action="append", default=None,
                    help="run just this group (repeatable)")
    ap.add_argument("--list", action="store_true", help="list the groups")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the cases per group (0 = no cap)")
    ap.add_argument("--show", type=int, default=25,
                    help="how many divergences to print per class")
    ap.add_argument("--skip-compiled", action="store_true",
                    help="cpython and the interpreter only -- no gcc")
    args = ap.parse_args()

    if args.list:
        for name, fn in GROUPS.items():
            print(f"{name:20} {len(fn()):7,} cases")
        return 0

    names = args.only or list(GROUPS)
    cases, owner = [], []
    for name in names:
        if name not in GROUPS:
            print(f"no such group: {name}", file=sys.stderr)
            return 2
        got = GROUPS[name]()
        if args.limit:
            got = got[:args.limit]
        cases.extend(got)
        owner.extend([name] * len(got))
    print(f"{len(cases):,} cases over {len(names)} group(s)")

    work = pathlib.Path(tempfile.mkdtemp(prefix="objects_diff_"))
    prog = work / "cases.py"
    prog.write_text(build_program(cases), encoding="utf-8")

    print("  cpython ...", flush=True)
    cp = indexed(run_cpython(prog))
    print("  interp  ...", flush=True)
    it = indexed(run_interp(prog))
    if args.skip_compiled:
        ir = c = None
    else:
        print("  ir      ...", flush=True)
        ir_lines, err = run_compiled(prog, "ir", work)
        if err:
            print("BUILD FAILED (ir):\n" + err, file=sys.stderr)
            return 1
        ir = indexed(ir_lines)
        print("  c       ...", flush=True)
        c_lines, err = run_compiled(prog, "c", work)
        if err:
            print("BUILD FAILED (c):\n" + err, file=sys.stderr)
            return 1
        c = indexed(c_lines)

    MISSING = "<no line>"
    buckets: dict[str, list] = {k: [] for k, _ in VERDICTS}
    for i, case in enumerate(cases):
        a = cp.get(i, MISSING)
        b = it.get(i, MISSING)
        d = ir.get(i, MISSING) if ir is not None else b
        e = c.get(i, MISSING) if c is not None else b
        verdict = classify(a, b, d, e)
        if verdict:
            buckets[verdict].append((owner[i], case, a, b, d, e))

    total = sum(len(v) for v in buckets.values())
    print(f"\n{total:,} divergent of {len(cases):,}\n")
    for key, why in VERDICTS:
        rows = buckets[key]
        if not rows:
            continue
        # ASCII, because this prints to whatever console the developer
        # has and a box-drawing character is a UnicodeEncodeError on a
        # cp1252 one -- the tool would die reporting its own findings.
        print(f"-- {key}  ({len(rows):,}) ".ljust(72, "-"))
        print(f"   {why}\n")
        seen = set()
        shown = 0
        for grp, case, a, b, d, e in rows:
            sig = (grp, a, b, d, e)
            if sig in seen:
                continue
            seen.add(sig)
            shown += 1
            if shown > args.show:
                break
            print(f"   [{grp}] {case}")
            print(f"       cpython {a}")
            if key != "all != cpython":
                print(f"       interp  {b}")
                print(f"       ir      {d}")
                print(f"       c       {e}")
            else:
                print(f"       ours    {d}")
        if len(rows) > shown:
            print(f"   ... {len(rows) - shown:,} more, "
                  f"{len(seen)} distinct shapes\n")
        print()
    print(f"work kept in {work}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
