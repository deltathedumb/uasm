# COVERAGE: getopt and gnu_getopt over short options (flags, required and
# optional arguments, clusters, glued and separate arguments) and long
# options (flags, `=` and `=?`, `--name=value`, unique and ambiguous
# abbreviations, an exact name that is also a prefix), `--`, a lone `-`,
# longopts given as one string; gnu_getopt's intermixing, its `+` and `-`
# prefixes; GetoptError and `error` with msg, opt, args and str for every
# message.
#
# Run under CPython and under uasm; the outputs must be identical. So
# the assertions below are written against what the module IS SPECIFIED to
# do, not against what uasm currently does.
import getopt


def attempt(label, f):
    try:
        print(label, f())
    except getopt.GetoptError as e:
        print(label, "GetoptError:", str(e), "|", e.msg, "|", repr(e.opt),
              "|", e.args)


SHORT = "ab:c::"
LONG = ["alpha", "beta=", "gamma=?", "delta", "deltoid"]
for argv in (["-a", "x"], ["-ab", "val", "x"], ["-abval", "x"],
             ["-b", "-a"], ["-c", "x"], ["-cvalue", "x"], ["-ac", "-b", "1"],
             ["--alpha", "x"], ["--beta", "v", "x"], ["--beta=v"],
             ["--gamma", "x"], ["--gamma=g", "x"], ["--al", "x"],
             ["--delta"], ["--del"], ["--deltoid"], ["--", "-a", "x"],
             ["-", "-a"], ["x", "-a"], [], ["--alpha", "--", "--beta"]):
    attempt("getopt %r" % argv, lambda: getopt.getopt(argv, SHORT, LONG))
for argv in (["-z"], ["-b"], ["--beta"], ["--alpha=x"], ["--zeta"],
             ["--de"], ["-c::"], ["--"]):
    attempt("error %r" % argv, lambda: getopt.getopt(argv, SHORT, LONG))
attempt("colon letter", lambda: getopt.getopt(["-:"], "a:"))
attempt("string longopts", lambda: getopt.getopt(["--only"], "", "only"))

for argv in (["x", "-a", "y", "--beta", "v", "z"], ["-a", "--", "-b", "x"],
             ["x", "y"], ["-", "-a"]):
    attempt("gnu %r" % argv, lambda: getopt.gnu_getopt(argv, SHORT, LONG))
    attempt("gnu+ %r" % argv,
            lambda: getopt.gnu_getopt(argv, "+" + SHORT, LONG))
    attempt("gnu- %r" % argv,
            lambda: getopt.gnu_getopt(argv, "-" + SHORT, LONG))
attempt("gnu error", lambda: getopt.gnu_getopt(["x", "-q"], SHORT, LONG))
print(getopt.error is getopt.GetoptError,
      issubclass(getopt.GetoptError, Exception))
e = getopt.GetoptError("plain")
print(str(e), repr(e.opt), e.args)
print("done")
