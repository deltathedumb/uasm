# COVERAGE: ArgumentParser end to end -- positional allocation for every
# nargs (None, N, ?, *, +, REMAINDER, PARSER) including the greedy
# give-one-back cases, `--`, option clusters (-xyz, -xyzVALUE), --opt=value
# and -ovalue, unambiguous and ambiguous abbreviations, allow_abbrev=False,
# negative numbers with and without negative-looking options; every action
# (store, store_const, store_true/false, append, append_const, extend,
# count, help, version, BooleanOptionalAction, a program's own Action);
# type (a callable raising ArgumentTypeError, int, float), choices (a list,
# a range, a string), default (converted when it is a string), required,
# metavar (tuples), dest, deprecated; argument groups, mutually exclusive
# groups (required or not), subparsers (aliases, help, title/description,
# metavar, dest, required, deprecated, a subparser's own defaults and
# errors); parents and conflict_handler='resolve'; set_defaults/get_default,
# argument_default=SUPPRESS, a namespace passed in; parse_known_args,
# parse_intermixed_args, parse_known_intermixed_args; exit_on_error=False
# and ArgumentError's attributes; every add_argument misuse CPython refuses;
# format_usage/format_help/print_usage/print_help at several widths with the
# usage wrap for short and long program names, all five formatter classes
# and a subclass of one; fromfile_prefix_chars with nested files and
# convert_arg_line_to_args; FileType; Namespace repr, ==, in; the reprs of
# an action and a parser; exit and error.
#
# Run under CPython and under uasm; the outputs must be identical. Every
# error path is captured -- stdout and stderr swapped for StringIO around
# the call -- and printed, so the message text is compared too, and the
# SystemExit a failing parse raises is caught and its code printed.
#
# EVERY PARSER NAMES ITS PROG. The default is the basename of argv[0], and
# that is checked once against `os.path.basename` rather than printed: the
# two sides run the same file under different spellings of its path.
import argparse
import io
import os
import sys


def run(parser, argv, how="parse_args"):
    saved_out, saved_err = sys.stdout, sys.stderr
    out = io.StringIO()
    err = io.StringIO()
    sys.stdout = out
    sys.stderr = err
    code = "-"
    result = None
    try:
        try:
            result = getattr(parser, how)(argv)
        except SystemExit as e:
            code = e.code
    finally:
        sys.stdout = saved_out
        sys.stderr = saved_err
    print("argv", argv, "->", result, "exit", code)
    if out.getvalue():
        print("  stdout:", repr(out.getvalue()))
    if err.getvalue():
        print("  stderr:", repr(err.getvalue()))


def attempt(label, f):
    try:
        print(label, repr(f()))
    except argparse.ArgumentError as e:
        print(label, "ArgumentError:", e)
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        print(label, type(e).__name__ + ":", e)


def narrow(width, **kwargs):
    return lambda prog: argparse.HelpFormatter(prog, width=width, **kwargs)


def positionals(*specs):
    p = argparse.ArgumentParser(prog="p")
    for name, nargs in specs:
        if nargs == "":
            p.add_argument(name)
        else:
            p.add_argument(name, nargs=nargs)
    return p


# ---- positional allocation ----------------------------------------------
# A greedy `*` gives strings back one at a time until what follows fits.
run(positionals(("a", "*"), ("b", 1)), ["x", "y", "z"])
run(positionals(("a", "*"), ("b", "?")), ["x", "y", "z"])
run(positionals(("a", "?"), ("b", "*")), ["x", "y", "z"])
run(positionals(("a", "+"), ("b", 2)), ["1", "2", "3", "4"])
run(positionals(("a", ""), ("b", "*"), ("c", "")), ["1", "2", "3", "4"])
run(positionals(("a", ""), ("b", "?"), ("c", "?")), ["1", "2"])
run(positionals(("a", 2), ("b", "?")), ["1"])
run(positionals(("a", 2), ("b", "?")), ["1", "2", "3", "4"])
run(positionals(("a", "*")), [])
run(positionals(("a", "+")), [])
run(positionals(("a", "")), ["--", "-x"])
run(positionals(("a", "*")), ["1", "--", "-x", "--", "y"])
run(positionals(("a", "?"), ("b", "")), ["--", "x"])
run(positionals(("a", argparse.REMAINDER)), ["1", "--", "-x", "--y"])
run(positionals(("a", ""), ("b", argparse.REMAINDER)), ["1", "-h", "x"])

p = argparse.ArgumentParser(prog="p")
p.add_argument("-f")
p.add_argument("a", nargs="*")
p.add_argument("b")
run(p, ["1", "2", "-f", "x", "3"])
run(p, ["1", "-f", "x", "2", "3"])
run(p, ["-f", "x"])
run(p, ["1", "2", "-f", "x", "3"], "parse_known_args")
run(p, ["1", "-f", "x", "2", "3"], "parse_intermixed_args")
run(p, ["1", "-f", "x", "2", "-g", "3"], "parse_known_intermixed_args")

# ---- options ---------------------------------------------------------------
p = argparse.ArgumentParser(prog="p")
p.add_argument("-x", action="store_true")
p.add_argument("-y", action="store_true")
p.add_argument("-z")
p.add_argument("--long")
p.add_argument("--longer", nargs="?", const="C", default="D")
for argv in (["-xy"], ["-xyz5"], ["-xyzfoo"], ["-xz", "5"], ["-x=1"],
             ["-z=5"], ["-z5"], ["--lo", "a"], ["--long=a"], ["--longe"],
             ["--longer"], ["--longer", "v"], ["--long", "-x"], ["-z", "-1"],
             ["-x", "-1"], ["-1"], ["--nope"], ["-q"], ["-xq"], ["a b"],
             ["-z", "--", "x"], ["--long=-x"], ["-z", "--long"], ["--x"],
             ["---x"], ["-", "-x"]):
    run(p, argv)

dep = argparse.ArgumentParser(prog="dep", allow_abbrev=False)
dep.add_argument("--old", deprecated=True)
dep.add_argument("pos", nargs="?", deprecated=True)
dep.add_argument("-1", dest="one", action="store_true")
run(dep, ["--old", "v", "--old", "w", "p"])
run(dep, ["--ol", "v"])
run(dep, ["-1"])
run(dep, ["-2"])
neg = argparse.ArgumentParser(prog="neg")
neg.add_argument("-n", type=int)
neg.add_argument("vals", nargs="*", type=float)
run(neg, ["-n", "-5", "-1.5", "-.5", "3"])
run(neg, ["-1e3"])

plus = argparse.ArgumentParser(prog="plus", prefix_chars="+/")
plus.add_argument("+x", action="store_true")
plus.add_argument("//long")
print(plus.format_help())
run(plus, ["+x", "//long", "v"])
run(plus, ["++help"])

# ---- help --------------------------------------------------------------------
p = argparse.ArgumentParser(
    prog="tool",
    description="Tool does %(prog)s things.  It is\n\n\n    wrapped   and squashed.",
    epilog="See %(prog)s --help.", formatter_class=narrow(78))
p.add_argument("source", help="where from")
p.add_argument("dest", nargs="?", default="out",
               help="where to (default: %(default)s)")
p.add_argument("extra", nargs="*", metavar="X")
p.add_argument("-v", "--verbose", action="count", default=0,
               help="more noise; repeat for even more noise than you would "
                    "ever want in a lifetime of runs")
p.add_argument("-q", "--quiet", action="store_true")
p.add_argument("--level", type=int, choices=[1, 2, 3], help="one of %(choices)s")
p.add_argument("--pair", nargs=2, metavar=("LEFT", "RIGHT"), help="two things")
p.add_argument("--many", nargs="+", help="at least one")
p.add_argument("--maybe", nargs="*", metavar=("A", "B"))
p.add_argument("--tail", nargs=argparse.REMAINDER)
p.add_argument("--hidden", help=argparse.SUPPRESS)
p.add_argument("--required-thing", required=True, dest="rt",
               help="must be given")
p.add_argument("--a-very-long-option-name-indeed", metavar="VALUE",
               help="long invocation goes on its own line")
g = p.add_argument_group("output", "Where results go.")
g.add_argument("--format", choices=("json", "text"), default="text")
g.add_argument("-o", "--output", metavar="FILE")
m = p.add_mutually_exclusive_group()
m.add_argument("--fast", action="store_true", help="go fast")
m.add_argument("--slow", action="store_true", help="go slow")
m2 = p.add_mutually_exclusive_group(required=True)
m2.add_argument("--red", action="store_const", const="r", dest="colour")
m2.add_argument("--blue", action="store_const", const="b", dest="colour")
p.add_argument("--version", action="version", version="%(prog)s 1.2.3")
print(p.format_usage())
print(p.format_help())
run(p, ["--version"])
run(p, ["-h"])
run(p, ["s", "--red", "--required-thing", "x", "--fast", "--slow"])
run(p, ["s", "--required-thing", "x"])
run(p, ["s", "--red"])
run(p, ["s", "--red", "--required-thing", "x", "--level", "7"])
run(p, ["s", "--red", "--required-thing", "x", "--level", "x"])
run(p, ["s", "--red", "--required-thing", "x", "-vvv", "--pair", "a", "b",
        "--tail", "-x", "--y"])

for width in (30, 50, 100):
    q = argparse.ArgumentParser(prog="w" * 12, formatter_class=narrow(width))
    for i in range(6):
        q.add_argument("--option-%d" % i, metavar="V%d" % i,
                       help="help text number %d that is long enough to "
                            "wrap around" % i)
    q.add_argument("positional_one")
    q.add_argument("positional_two", nargs="+")
    print(q.format_help())

tight = argparse.ArgumentParser(
    prog="tight", formatter_class=narrow(60, max_help_position=10,
                                         indent_increment=4))
tight.add_argument("--alpha", help="the alpha option, placed below its name")
tight.add_argument("beta", help="beta")
print(tight.format_help())

long_prog = "a-very-long-program-name-that-takes-most-of-the-line"
long1 = argparse.ArgumentParser(prog=long_prog, formatter_class=narrow(60))
for i in range(4):
    long1.add_argument("--opt%d" % i, metavar="X")
long1.add_argument("pos", nargs=2)
print(long1.format_usage())
long2 = argparse.ArgumentParser(prog=long_prog, formatter_class=narrow(60))
long2.add_argument("--o")
print(long2.format_usage())

for cls in (argparse.RawDescriptionHelpFormatter,
            argparse.RawTextHelpFormatter,
            argparse.ArgumentDefaultsHelpFormatter,
            argparse.MetavarTypeHelpFormatter):
    r = argparse.ArgumentParser(prog="raw", formatter_class=cls,
                                description="  keep\n    this   layout\n",
                                epilog="  and\n  this")
    r.add_argument("--n", type=int, default=5,
                   help="  first line\n  second   line")
    r.add_argument("x", type=float, nargs="?", default=1.5, help="an x")
    r.add_argument("--s", type=str, help="no default shown?")
    print(cls.__name__)
    print(r.format_help())


class Shouting(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        return [line.upper() for line in super()._split_lines(text, width)]


shout = argparse.ArgumentParser(prog="shout", formatter_class=Shouting)
shout.add_argument("--quiet", help="say it quietly, which this will not do")
print(shout.format_help())

u = argparse.ArgumentParser(prog="custom", usage="%(prog)s [options] FILE...")
u.add_argument("f")
print(u.format_help())
e = argparse.ArgumentParser(prog="empty", add_help=False)
print(repr(e.format_usage()), repr(e.format_help()))
out = io.StringIO()
u.print_usage(out)
u.print_help(out)
print(repr(out.getvalue()))

# ---- subcommands ---------------------------------------------------------------
p = argparse.ArgumentParser(prog="git")
p.add_argument("--verbose", action="store_true")
sub = p.add_subparsers(dest="cmd", help="sub-command help", required=True)
c = sub.add_parser("commit", help="record changes", aliases=["ci"])
c.add_argument("-m", "--message", required=True)
c.add_argument("paths", nargs="*")
push = sub.add_parser("push", help="send them", deprecated=True)
push.add_argument("remote", nargs="?", default="origin")
push.set_defaults(force=False)
sub.add_parser("status")
print(p.format_help())
print(c.format_help())
run(p, ["commit", "-m", "hi", "a", "b"])
run(p, ["ci", "-m", "x"])
run(p, ["--verbose", "push"])
run(p, ["push", "up", "--extra"])
run(p, ["push", "up", "--extra"], "parse_known_args")
run(p, ["nope"])
run(p, [])
run(p, ["commit"])
run(p, ["commit", "-h"])

t = argparse.ArgumentParser(prog="t")
tsub = t.add_subparsers(title="commands", description="what to do",
                        metavar="CMD")
a = tsub.add_parser("add")
a.add_argument("x", type=int)
a.set_defaults(func="adder")
print(t.format_help())
run(t, ["add", "4"])
run(t, ["add", "four"])
run(t, [])

# ---- parents and conflicts -----------------------------------------------------
base = argparse.ArgumentParser(add_help=False)
base.add_argument("--shared", default="s")
g = base.add_argument_group("shared group")
g.add_argument("--grouped", action="store_true")
mx = base.add_mutually_exclusive_group()
mx.add_argument("--one", action="store_true")
mx.add_argument("--two", action="store_true")
child = argparse.ArgumentParser(prog="child", parents=[base])
child.add_argument("--own")
print(child.format_help())
run(child, ["--one", "--two"])
run(child, ["--shared", "x", "--grouped"])
attempt("conflict", lambda: child.add_argument("--own"))
res = argparse.ArgumentParser(prog="res", conflict_handler="resolve")
res.add_argument("-x", "--xx", help="first")
res.add_argument("--xx", help="second")
res.add_argument("-x", help="third")
print(res.format_help())
attempt("bad handler", lambda: argparse.ArgumentParser(conflict_handler="no"))

# ---- actions -------------------------------------------------------------------
q = argparse.ArgumentParser(prog="q")
q.add_argument("--app", action="append", type=int)
q.add_argument("--ext", action="extend", nargs="+")
q.add_argument("--const", action="append_const", const="C", dest="consts")
q.add_argument("--const2", action="append_const", const="D", dest="consts")
q.add_argument("-c", action="count")
q.add_argument("--flag", action=argparse.BooleanOptionalAction, default=True,
               help="toggle it")
q.add_argument("--sc", action="store_const", const=42)
q.add_argument("--sf", action="store_false")
q.add_argument("--default-list", action="append", default=["base"])
print(q.format_help())
run(q, ["--app", "1", "--app", "2", "--ext", "a", "b", "--ext", "c",
        "--const", "--const2", "-ccc", "--no-flag", "--sc", "--sf",
        "--default-list", "more"])
run(q, [])
run(q, ["--app", "x"])
run(q, ["--flag", "--no-flag", "--flag"])


class Upper(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest,
                values.upper() + "|" + str(option_string))


def pos_int(text):
    n = int(text)
    if n <= 0:
        raise argparse.ArgumentTypeError("%r is not positive" % text)
    return n


r = argparse.ArgumentParser(prog="r", argument_default=argparse.SUPPRESS)
r.add_argument("--up", action=Upper)
r.add_argument("--n", type=pos_int)
r.add_argument("--f", type=float)
r.add_argument("--choice", type=int, choices=range(1, 4))
r.add_argument("--letters", choices="abc")
run(r, ["--up", "hi", "--n", "3"])
run(r, [])
run(r, ["--n", "-2"])
run(r, ["--n", "x"])
run(r, ["--f", "1e3", "--choice", "3", "--letters", "b"])
run(r, ["--choice", "9"])
run(r, ["--letters", "ab"])

# ---- errors --------------------------------------------------------------------
e = argparse.ArgumentParser(prog="e", exit_on_error=False)
e.add_argument("--n", type=int)
e.add_argument("pos")
attempt("type", lambda: e.parse_args(["--n", "x", "p"]))
attempt("missing", lambda: e.parse_args([]))
attempt("unrecognized", lambda: e.parse_args(["p", "--zzz"]))
attempt("expected", lambda: e.parse_args(["p", "--n"]))
try:
    e.parse_args(["--n", "bad", "p"])
except argparse.ArgumentError as err:
    print("attrs:", err.argument_name, "|", err.message)


class Raising(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("refused: " + message)


attempt("error override", lambda: Raising(prog="x").parse_args(["--what"]))

m = argparse.ArgumentParser(prog="m")
attempt("store nargs 0", lambda: m.add_argument("--x", nargs=0))
attempt("const", lambda: m.add_argument("--y", const=1))
attempt("bad action", lambda: m.add_argument("--z", action="bogus"))
attempt("pos required", lambda: m.add_argument("p", required=True))
attempt("pos flag", lambda: m.add_argument("p", action="store_true"))
attempt("bad prefix", lambda: m.add_argument("-a", "b"))
attempt("dest twice", lambda: m.add_argument("p", dest="q"))
attempt("metavar", lambda: m.add_argument("--mv", nargs=2, metavar=("A",)))
attempt("bad help", lambda: m.add_argument("--h2", help="%(nope)s"))
attempt("bad type", lambda: m.add_argument("--t", type="int"))
attempt("filetype class",
        lambda: m.add_argument("--ft", type=argparse.FileType))
attempt("bad nargs", lambda: m.add_argument("--bn", nargs="x"))
attempt("no dest", lambda: m.add_argument("--"))
attempt("nested group", lambda: m.add_argument_group().add_argument_group())
attempt("nested mutex", lambda: m.add_mutually_exclusive_group()
        .add_mutually_exclusive_group())
attempt("required in mutex", lambda: m.add_mutually_exclusive_group()
        .add_argument("--rq", required=True))
attempt("two subparsers", lambda: (m.add_subparsers(), m.add_subparsers()))
attempt("bool no-", lambda: m.add_argument(
    "--no-thing", action=argparse.BooleanOptionalAction))
attempt("intermixed parser", lambda: m.parse_intermixed_args([]))
attempt("parents", lambda: argparse.ArgumentParser(parents=[object()]))
run(argparse.ArgumentParser(prog="ex"), [], "exit")

# ---- defaults, namespaces, reprs -------------------------------------------------
d = argparse.ArgumentParser(prog="d")
d.add_argument("--x", default="7", type=int)
d.add_argument("--y", default=[1], nargs="*")
d.set_defaults(x="8", z="zz")
print(d.get_default("x"), d.get_default("z"), d.get_default("nope"))
run(d, [])
ns = argparse.Namespace(a=1, b="two")
ns2 = argparse.Namespace(b="two", a=1)
print(ns, ns == ns2, ns != ns2, "a" in ns, "c" in ns, vars(ns))
print(argparse.Namespace(**{"not-ident": 1, "ok": 2}))
print(d.parse_args([], namespace=argparse.Namespace(x=100)))
act = d.add_argument("--rep", type=int, choices=[1, 2], help="h")
print(repr(act))
print(repr(d))
print(argparse.SUPPRESS, argparse.OPTIONAL, argparse.ZERO_OR_MORE,
      argparse.ONE_OR_MORE, argparse.PARSER, argparse.REMAINDER)
print(argparse.ArgumentParser().prog == os.path.basename(sys.argv[0]))

# ---- files -------------------------------------------------------------------------
_PATH = "apy-argparse-case.txt"
_INNER = "apy-argparse-inner.txt"
with open(_PATH, "w") as f:
    f.write("--name\nfrom a file\n@" + _INNER + "\n")
with open(_INNER, "w") as f:
    f.write("pos-from-inner\n")
ff = argparse.ArgumentParser(prog="ff", fromfile_prefix_chars="@")
ff.add_argument("--name")
ff.add_argument("rest", nargs="*")
run(ff, ["@" + _PATH, "tail"])
run(ff, ["@apy-argparse-missing.txt"])


class Words(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()


words = Words(prog="words", fromfile_prefix_chars="@")
words.add_argument("rest", nargs="*")
run(words, ["@" + _PATH])

ft = argparse.ArgumentParser(prog="ft")
ft.add_argument("--in", dest="infile", type=argparse.FileType("r"))
ft.add_argument("--out", type=argparse.FileType("w", encoding="utf-8"))
print(repr(argparse.FileType("rb")), repr(argparse.FileType("w", 1, "utf-8")))
args = ft.parse_args(["--in", _PATH, "--out", "-"])
print(args.infile.read().splitlines(), args.out is sys.stdout)
args.infile.close()
run(ft, ["--in", "apy-argparse-missing.txt"])
os.remove(_PATH)
os.remove(_INNER)
print("done")
