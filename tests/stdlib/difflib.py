# COVERAGE: SequenceMatcher -- construction with isjunk/a/b/autojunk,
# set_seqs/set_seq1/set_seq2, find_longest_match with and without bounds,
# get_matching_blocks, get_opcodes, get_grouped_opcodes (including the
# in-place trim of the cached opcodes), ratio/quick_ratio/real_quick_ratio,
# the b2j/bjunk/bpopular attributes and the autojunk heuristic on a 200+
# element sequence, sequences that are not strings; Match; get_close_matches
# with its tie-breaking and refusals; Differ.compare and ndiff with intraline
# `?` marking, tabs, and the plain-replace fallback; restore; unified_diff and
# context_diff with file names, dates, n=0/1/3, lineterm='', identical inputs
# and the type refusals; diff_bytes over non-ASCII bytes; IS_LINE_JUNK and
# IS_CHARACTER_JUNK.
#
# Run under CPython and under uasm; the outputs must be identical. So
# the assertions below are written against what the module IS SPECIFIED to
# do, not against what uasm currently does.
import difflib
import keyword
from difflib import SequenceMatcher


def attempt(label, f):
    try:
        print(label, repr(f()))
    except (TypeError, ValueError, KeyError) as e:
        print(label, type(e).__name__ + ":", e)


# ---- SequenceMatcher -------------------------------------------------------
s = SequenceMatcher(lambda x: x == " ", "private Thread currentThread;",
                    "private volatile Thread currentThread;")
print(round(s.ratio(), 3), s.get_matching_blocks())
for block in s.get_matching_blocks():
    print("a[%d] and b[%d] match for %d elements" % block)
for opcode in s.get_opcodes():
    print("%6s a[%d:%d] b[%d:%d]" % opcode)

s = SequenceMatcher(None, " abcd", "abcd abcd")
print(s.find_longest_match(0, 5, 0, 9), s.find_longest_match())
print(s.find_longest_match(1, 3, 0, 9), s.find_longest_match(0, 5, 5, 9))
s = SequenceMatcher(lambda x: x == " ", " abcd", "abcd abcd")
print(s.find_longest_match(0, 5, 0, 9), sorted(s.bjunk))
print(SequenceMatcher(None, "ab", "c").find_longest_match(0, 2, 0, 1))
m = SequenceMatcher(None, "abxcd", "abcd").get_matching_blocks()
print(m, m[0].a, m[0].b, m[0].size, m[-1] == (5, 4, 0))

a, b = "qabxcd", "abycdf"
s = SequenceMatcher(None, a, b)
for tag, i1, i2, j1, j2 in s.get_opcodes():
    print("%7s a[%d:%d] (%s) b[%d:%d] (%s)"
          % (tag, i1, i2, a[i1:i2], j1, j2, b[j1:j2]))
s = SequenceMatcher(None, "abcd", "bcde")
print(s.ratio(), s.quick_ratio(), s.real_quick_ratio())
s.set_seq1("bcde")
print(s.ratio())
s.set_seq2("abcd")
print(s.ratio())
s.set_seqs("", "")
print(s.ratio(), s.get_opcodes(), s.get_matching_blocks())
print(SequenceMatcher(None, "abc", "").ratio(),
      SequenceMatcher(None, "", "abc").get_opcodes())

# THE LONGEST BLOCK WINS, and ties go to the earliest in `a`, then in `b`.
print(SequenceMatcher(None, "ab", "acab").get_opcodes())
print(SequenceMatcher(None, "xyxy", "yxyx").get_matching_blocks())
print(SequenceMatcher(None, [1, 2, 3, 4], [1, 3, 4, 5]).get_opcodes())
print(SequenceMatcher(None, ("a", "b"), ("b", "a")).ratio())
print(SequenceMatcher(None, "abcde" * 3, "edcba" * 3).get_matching_blocks())

# THE INDEX, junk and popular elements left out of it.
s = SequenceMatcher(lambda c: c in "xy", "", "axbyca")
print(sorted(s.b2j.items()), sorted(s.bjunk), sorted(s.bpopular))
calls = []


def counting(elt):
    calls.append(elt)
    return elt == "b"


SequenceMatcher(counting, "", "abcabcb")
print("isjunk asked:", calls)
long_b = "x" * 150 + "abc" * 30 + "q"
s = SequenceMatcher(None, "xxabcq", long_b)
print(len(long_b), sorted(s.bpopular), sorted(s.b2j))
print(s.get_matching_blocks()[:3], round(s.ratio(), 4))
s = SequenceMatcher(None, "xxabcq", long_b, autojunk=False)
print(sorted(s.bpopular), s.get_matching_blocks()[:3], round(s.ratio(), 4))

# GROUPED, and the cached opcodes it trims in place.
a = [str(i) for i in range(1, 40)]
b = a[:]
b[8:8] = ["i"]
b[20] += "x"
b[23:28] = []
b[30] += "y"
for group in SequenceMatcher(None, a, b).get_grouped_opcodes():
    print(group)
print(list(SequenceMatcher(None, a, b).get_grouped_opcodes(0)))
print(list(SequenceMatcher(None, a, a).get_grouped_opcodes()))
print(list(SequenceMatcher(None, [], []).get_grouped_opcodes()))
s = SequenceMatcher(None, "abcdefghij", "abcdefghiX")
print(s.get_opcodes())
print(list(s.get_grouped_opcodes(1)))
print(s.get_opcodes())

# ---- get_close_matches -----------------------------------------------------
print(difflib.get_close_matches("appel", ["ape", "apple", "peach", "puppy"]))
print(difflib.get_close_matches("wheel", keyword.kwlist))
print(difflib.get_close_matches("Apple", keyword.kwlist))
print(difflib.get_close_matches("accept", keyword.kwlist))
print(difflib.get_close_matches("ab", ["ba", "ab", "aa", "bb", "ac"], n=5))
print(difflib.get_close_matches("xyz", ["xya", "xyb", "xyc"], n=2))
print(difflib.get_close_matches("hello", ["hallo", "hello", "help"], 1, 0.0))
attempt("n", lambda: difflib.get_close_matches("a", ["a"], n=0))
attempt("cutoff", lambda: difflib.get_close_matches("a", ["a"], cutoff=1.5))

# ---- Differ and ndiff --------------------------------------------------------
text1 = """  1. Beautiful is better than ugly.
  2. Explicit is better than implicit.
  3. Simple is better than complex.
  4. Complex is better than complicated.
""".splitlines(keepends=True)
text2 = """  1. Beautiful is better than ugly.
  3.   Simple is better than complex.
  4. Complicated is better than complex.
  5. Flat is better than nested.
""".splitlines(keepends=True)
result = list(difflib.Differ().compare(text1, text2))
for line in result:
    print(repr(line))
print("".join(result), end="")
diff = list(difflib.ndiff("one\ntwo\nthree\n".splitlines(keepends=True),
                          "ore\ntree\nemu\n".splitlines(keepends=True)))
print("".join(diff), end="")
print("".join(difflib.restore(diff, 1)), end="")
print("".join(difflib.restore(diff, 2)), end="")
attempt("restore", lambda: list(difflib.restore(diff, 3)))
print(list(difflib.ndiff(["\tabcDefghiJkl\n"], ["\tabcdefGhijkl\n"])))
print(list(difflib.ndiff(["a\n", "b\n"], ["x\n", "y\n", "z\n"])))
print(list(difflib.ndiff(["same\n"], ["same\n"])))
print(list(difflib.Differ(charjunk=difflib.IS_CHARACTER_JUNK).compare(
    ["private Thread currentThread;\n"],
    ["private volatile Thread currentThread;\n"])))
print(list(difflib.Differ(linejunk=difflib.IS_LINE_JUNK).compare(
    ["a\n", "\n", "b\n"], ["a\n", "#\n", "b\n"])))
many_a = ["line %d is here\n" % i for i in range(30)]
many_b = [line.replace("here", "there") if i % 7 == 0 else line
          for i, line in enumerate(many_a)]
print("".join(difflib.ndiff(many_a, many_b)), end="")

# ---- unified and context diffs ---------------------------------------------
for line in difflib.unified_diff("one two three four".split(),
                                 "zero one tree four".split(), "Original",
                                 "Current", "2005-01-26 23:30:50",
                                 "2010-04-02 10:20:52", lineterm=""):
    print(line)
print("".join(difflib.context_diff(
    "one\ntwo\nthree\nfour\n".splitlines(True),
    "zero\none\ntree\nfour\n".splitlines(True), "Original", "Current")),
    end="")
for n in (0, 1, 3):
    print("n =", n)
    print("".join(difflib.unified_diff(a, b, "a.txt", "b.txt", n=n,
                                       lineterm="\n")), end="")
    print("".join(difflib.context_diff(a, b, n=n)), end="")
print(list(difflib.unified_diff(["x\n"], ["x\n"])))
print(list(difflib.context_diff(["x\n"], ["x\n"])))
print(list(difflib.unified_diff([], ["new\n"])))
print(list(difflib.unified_diff(["gone\n"], [])))
print(list(difflib.context_diff([], ["new\n"], lineterm="")))
print(list(difflib.context_diff(["gone\n"], [], lineterm="")))
attempt("bytes lines", lambda: list(difflib.unified_diff([b"a"], ["b"])))
attempt("str input", lambda: list(difflib.unified_diff("abc", ["b"])))
attempt("bytes name", lambda: list(difflib.context_diff(["a"], ["b"],
                                                          fromfile=b"x")))
for line in difflib.diff_bytes(difflib.unified_diff, [b"caf\xe9\n", b"x\n"],
                               [b"cafe\n", b"x\n"], b"old", b"new"):
    print(line)
attempt("diff_bytes str", lambda: list(difflib.diff_bytes(
    difflib.unified_diff, ["a"], [b"b"])))

# ---- junk predicates ---------------------------------------------------------
print([difflib.IS_LINE_JUNK(line) for line in
       ("\n", "  #   \n", "hello\n", "##\n", "", "   ")])
print([difflib.IS_CHARACTER_JUNK(ch) for ch in (" ", "\t", "\n", "x")])
print(difflib.IS_CHARACTER_JUNK("x", ws="xyz"))
print(difflib.Match(1, 2, 3), difflib.Match(1, 2, 3)._asdict())
print("done")
