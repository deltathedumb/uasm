# COVERAGE: sys.exit -- the exception it raises, the status it carries, and
# the one place `sys.exit(None)` and `SystemExit(None)` part company. The
# ESCAPING form is not here: this runner requires both sides to exit 0 and
# compares stdout, so a program that really exits cannot be one of these
# cases -- `tests/uasm/integration/test_cli.py` measures the status instead.
# Also sys.getrefcount -- one binding, a second binding, a binding
# dropped, membership in a list, a dict value, a set member, an instance
# attribute, and each of those going away again. A function asking about
# its OWN PARAMETER reads one higher here than in CPython 3.14 and is
# left out on purpose -- see `objects_host._apy_sys_getrefcount` for
# why (that binding is a counted reference here and a borrowed one
# there).
import sys


class Foo:
    def __init__(self):
        self.child = None


def bindings():
    x = Foo()
    print("one name:", sys.getrefcount(x))
    y = x
    print("two names:", sys.getrefcount(x))
    z = x
    print("three names:", sys.getrefcount(x))
    del z
    print("back to two:", sys.getrefcount(x))
    del y
    print("back to one:", sys.getrefcount(x))


bindings()
print("--- bindings done ---")


def containers():
    x = Foo()
    print("start:", sys.getrefcount(x))
    xs = [x, x]
    print("twice in a list:", sys.getrefcount(x))
    d = {"k": x}
    print("plus a dict value:", sys.getrefcount(x))
    s = {x}
    print("plus a set member:", sys.getrefcount(x))
    holder = Foo()
    holder.child = x
    print("plus an attribute:", sys.getrefcount(x))
    holder.child = None
    print("attribute gone:", sys.getrefcount(x))
    s.discard(x)
    print("set member gone:", sys.getrefcount(x))
    del d["k"]
    print("dict entry gone:", sys.getrefcount(x))
    xs.clear()
    print("list cleared:", sys.getrefcount(x))


containers()
print("--- containers done ---")


def overwritten():
    x = Foo()
    xs = [x]
    print("in the list:", sys.getrefcount(x))
    xs[0] = None
    print("overwritten:", sys.getrefcount(x))


overwritten()
print("--- overwritten done ---")


# --- a reference a CALL made, and gave back --------------------------
#
# Passing an object into a function does not leave a reference behind:
# the callee's binding is gone the moment it returns. Written out because
# the reverse -- an argument the interpreter went on counting -- is
# invisible to every test that does not ask this question, and shows up
# instead as a `__del__` that never runs and a `weakref` that never dies.
def looked_at(o):
    return 0


def calls_leave_nothing():
    x = Foo()
    print("before any call: ", sys.getrefcount(x))
    looked_at(x)
    print("after a plain call:", sys.getrefcount(x))
    looked_at(x)
    looked_at(x)
    print("after three:      ", sys.getrefcount(x))
    x.method_like = None
    repr(x)
    print("after a builtin:  ", sys.getrefcount(x))


calls_leave_nothing()
print("done")


# --- the bundled half: streams, breakpoint, audit, monitoring ---------
print("stdout writable:", sys.stdout.writable(), sys.stdout.readable(),
      sys.stdout.seekable())
print("descriptors:", sys.stdout.fileno(), sys.stderr.fileno())
print("closed/isatty:", sys.stdout.closed, sys.stdout.isatty(),
      sys.stdout.flush())
try:
    sys.stdout.write(7)
except TypeError as exc:
    print("write type:", exc)

calls = []
sys.breakpointhook = lambda *a, **kw: calls.append((a, sorted(kw)))
breakpoint()
breakpoint(1, k=2)
print("breakpoint:", calls, callable(breakpoint))

seen = []
sys.addaudithook(lambda event, args: seen.append((event, args)))
sys.audit("stdlib.test", 1, "two")
sys.audit("stdlib.other")
print("audit:", seen, callable(sys.audit), callable(sys.addaudithook))

mon = sys.monitoring
print("ids:", mon.DEBUGGER_ID, mon.COVERAGE_ID, mon.PROFILER_ID,
      mon.OPTIMIZER_ID)
print("events:", mon.events.NO_EVENTS, mon.events.PY_START, mon.events.CALL,
      mon.events.LINE, mon.events.BRANCH)
print("unclaimed:", mon.get_tool(mon.DEBUGGER_ID))
mon.use_tool_id(mon.DEBUGGER_ID, "stdlib")
print("claimed:", mon.get_tool(mon.DEBUGGER_ID))
try:
    mon.use_tool_id(mon.DEBUGGER_ID, "again")
except ValueError as exc:
    print("reuse:", exc)
print("events before:", mon.get_events(mon.DEBUGGER_ID))
mon.set_events(mon.DEBUGGER_ID, mon.events.CALL | mon.events.LINE)
print("events after:", mon.get_events(mon.DEBUGGER_ID))
print("callback:", mon.register_callback(mon.DEBUGGER_ID, mon.events.CALL,
                                         lambda *a: None))
mon.free_tool_id(mon.DEBUGGER_ID)
print("freed:", mon.get_tool(mon.DEBUGGER_ID))
for bad in (9, -1):
    try:
        mon.get_tool(bad)
    except ValueError as exc:
        print("bad id:", exc)


# --- A FUNCTION'S COUNT IS CPYTHON'S, before and after any number of calls.
# The callee of a dynamic call is an argument like any other, and leaving
# that slot alone made a callable held in a global read one high per CALL
# SITE -- which is also why a function's count could not be cascaded from.
#
# INSIDE A FUNCTION, like every other count measured in this file: a module
# body containing a loop does not retire its temporaries, so a count read at
# module level here would be measuring that divergence instead of this one.
def counted():
    return 1


def makes_one():
    captured = 5

    def held():
        return captured

    return held


def function_counts():
    named = counted
    print("alias:", sys.getrefcount(named))
    named()
    named()
    print("alias after two calls:", sys.getrefcount(named))

    closed = makes_one()
    print("closure:", sys.getrefcount(closed))
    closed()
    closed()
    closed()
    print("closure after three calls:", sys.getrefcount(closed))
    also = closed
    print("two names:", sys.getrefcount(closed))
    del also
    print("back to one:", sys.getrefcount(closed))


function_counts()
print("--- function counts done ---")


# COVERAGE: sys.argv -- what it is, what is in it, and that it is an
# ORDINARY MUTABLE LIST rather than a view onto the host. The last part is
# what a program relying on `argparse`-shaped code needs, and it is the part
# a lazy implementation would get wrong.
#
# `argv[0]` IS COMPARED BY SHAPE, not printed: the oracle runs
# `python tests/stdlib/sys.py` and uasm runs `uasm run
# tests/stdlib/sys.py`, so both see the same path -- but printing it would
# put a machine-specific absolute path in the compared output for no gain.

def command_line():
    print("type:", type(sys.argv).__name__)
    print("never empty:", len(sys.argv) >= 1)
    print("argv[0] is a str:", isinstance(sys.argv[0], str))
    print("argv[0] names this file:",
          sys.argv[0].replace("\\", "/").endswith("tests/stdlib/sys.py"))
    # NO ARGUMENTS WERE PASSED, so everything after the name is empty -- on
    # both sides, which is the point of asserting it.
    print("tail:", sys.argv[1:])

    # A REAL LIST. Every one of these would fail against a host-backed view.
    same = sys.argv
    print("same object:", same is sys.argv)
    same.append("appended")
    print("append is visible:", sys.argv[-1])
    print("length after append:", len(sys.argv))
    sys.argv.pop()
    print("length after pop:", len(sys.argv))
    print("a copy is not the list:", list(sys.argv) == sys.argv,
          list(sys.argv) is sys.argv)


command_line()
print("--- command line done ---")


def command_line_replaced():
    """A program that REWRITES `sys.argv`, which real ones do.

    `argparse`-shaped code assigns a list of its own before re-parsing, and a
    test suite that drives a `main(argv)` does it every case. What it reads
    back has to be what it wrote, with nothing from the host left in it.
    """
    sys.argv = ["prog", "--flag", "value"]
    print("replaced:", sys.argv)
    print("name:", sys.argv[0], "rest:", sys.argv[1:])
    print("length:", len(sys.argv))
    for one in sys.argv:
        print("  *", one)


command_line_replaced()
print("--- command line replaced done ---")


def exiting():
    """`sys.exit` RAISES; it does not stop the process.

    That is the whole of what it does, and the distinction is observable:
    the exception unwinds, so a `finally` runs and an enclosing `except
    SystemExit` may decline to exit at all. A function that ended the
    process would skip both.
    """
    for label, call in (("2", lambda: sys.exit(2)),
                        ("bare", lambda: sys.exit()),
                        ("None", lambda: sys.exit(None)),
                        ("0", lambda: sys.exit(0)),
                        ("message", lambda: sys.exit("no such file"))):
        try:
            call()
        except SystemExit as e:
            print(f"{label:8} code={e.code!r} args={e.args!r}")
        else:
            print(f"{label:8} did not raise")


exiting()
print("--- exit done ---")


def exit_unwinds():
    """The `finally` runs and the handler may decline, both observable."""
    try:
        try:
            sys.exit(9)
        finally:
            print("finally ran")
    except SystemExit as e:
        print("declined", e.code)
    print("carried on")


exit_unwinds()
print("--- exit unwinding done ---")


def the_constructor_keeps_what_it_was_handed():
    """`sys.exit(None)` and `SystemExit(None)` are NOT the same object.

    `sys.exit` drops the None -- CPython raises a SystemExit with no
    arguments for both the bare call and the explicit None, so `args` is
    `()` -- while the CONSTRUCTOR keeps it. A reimplementation that passes
    its default straight through gets the second row wrong and nothing else
    notices.
    """
    print("ctor None :", SystemExit(None).args, SystemExit(None).code)
    print("ctor bare :", SystemExit().args, SystemExit().code)
    print("ctor two  :", SystemExit(1, 2).args, SystemExit(1, 2).code)


the_constructor_keeps_what_it_was_handed()
print("--- exit constructor done ---")
