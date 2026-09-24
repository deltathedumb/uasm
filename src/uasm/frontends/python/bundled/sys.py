"""`sys`, for the half of it that is ORDINARY PYTHON.

COVERAGE: `stdout` and `stderr` as real writable streams on file descriptors
1 and 2, replaceable by assignment -- `sys.stdout = io.StringIO()` captures
what `print` writes, which is the whole reason they are objects here rather
than compiler constants; `argv`, the command line the process was started
with; `breakpointhook` and `__breakpointhook__`, PEP 553; `audit` and
`addaudithook`, PEP 578; `monitoring`, PEP 669's tool-id registry and event
constants.

BUNDLED IN PART, and `bundled.py` says so at two places that were written
before this file existed. The other half of `sys` -- `maxsize`, `platform`,
`byteorder`, `implementation`, `getrefcount` -- stays in
`frontends/python/modules.py`'s native `_SYS` table, because each of those is
a COMPILER CONSTANT or a runtime primitive rather than Python. A program
importing `sys` reaches both halves through one name; see `bundled.py`'s
`without_bundled`, which keeps the `import` statement alive precisely so the
native half stays bound.

## What `print` has to do with any of this

`print` is a COMPILER BUILTIN here -- it lowers to a call that writes through
the platform floor, and that is what makes it fast and what makes a
freestanding image able to print at all. A program that replaces
`sys.stdout` needs it to go somewhere else instead, and the compiler cannot
know at the call site whether it will.

`bundled.py` decides that by SCANNING FOR THE ASSIGNMENT: if the program
stores to `sys.stdout` or `sys.stderr` anywhere, every `print` in it is
rewritten to `_print` below, which reads the module-level `stdout` at call
time. A program that never redirects keeps the direct call and pays nothing.
That is `_Rewrite.visit_Name`'s `redirects` branch, and `_print` is the
function it names.

## What does NOT happen, said plainly

AN AUDIT HOOK NEVER FIRES BY ITSELF. CPython's own runtime raises audit
events from inside `open`, `exec`, `socket.connect` and about a hundred other
places; nothing in this runtime raises one. `sys.audit(...)` called BY THE
PROGRAM reaches every registered hook exactly as CPython's does, and that is
the whole of what works -- a program auditing its own events is served, a
program watching for the interpreter's is not, and would silently see
nothing. Said here because silence is the failure mode that looks like
success.

A MONITORING CALLBACK NEVER FIRES either, for the same reason: PEP 669's
events come from the interpreter's eval loop, and compiled code has none. The
TOOL REGISTRY below is real state -- `use_tool_id` refuses an id already
taken, `get_tool` answers what was registered, `free_tool_id` releases it --
because that half is bookkeeping a program can do and observe without any
interpreter at all. `set_events` and `register_callback` record what they
were given and answer what CPython answers; nothing consults either.

A HOOK IS NOT REMOVED. CPython has no `removeaudithook` and neither does
this; hooks accumulate for the life of the program, which is the documented
behaviour rather than a limitation of this file.

`breakpoint()` WITH THE DEFAULT HOOK RAISES. CPython's default enters `pdb`,
and there is no debugger in a compiled binary to enter -- so the default
refuses BY NAME rather than doing nothing, which would make `breakpoint()`
look like it worked. Replacing the hook, which is what PEP 553 is for, works
exactly as specified.

## The `file` and `env` host-service groups

`stdout` and `stderr` write through `host_file_write`, which is the same
primitive `bundled/io.py` uses and which belongs to the `file` service group
-- so a program that imports `sys` needs a backend providing it. Descriptors
1 and 2 are guaranteed to mean standard output and standard error there; see
`objects/hostsvc.py`, which says so where it defines them. Routing `stderr`
to `print` instead would have avoided the dependency and sent the error
stream to standard output, which is a wrong answer rather than a smaller one.

`argv` adds `env` to that, for `host_arg_count` and `host_arg_get`, and adds
it to EVERY program that imports `sys` rather than only to one that reads
`argv` -- because the list is built by module-level code, and module-level
code always runs. That is the price of `sys.argv` being a real list; see the
section that builds it for why it cannot be lazy. It costs nothing today:
the two backends that provide `file` provide `env` too, so no program that
compiled before this stops compiling.
"""


class _Console:
    """`sys.stdout` or `sys.stderr` -- a writable text stream on a descriptor.

    NOT `io.TextIOWrapper`, and not built on `bundled/io.py`: a program that
    imports `sys` would then carry the whole of `io` for two objects that
    only ever write. What a program does with either is `write`, `flush`, and
    ask whether it can be written to, so that is what is here.

    THE WRITE LOOP IS NOT DECORATION. `host_file_write` may return a SHORT
    COUNT -- a pipe with a full buffer is the ordinary case -- so a single
    call is not a write, and the same loop is in `bundled/io.py` for the same
    reason.
    """

    def __init__(self, fd, name):
        self._fd = fd
        self.name = name
        self.mode = "w"
        self.encoding = "utf-8"
        self.errors = "strict"
        self.newlines = None
        self.line_buffering = False

    def write(self, text):
        if not isinstance(text, str):
            raise TypeError("write() argument must be str, not "
                            + type(text).__name__)
        rest = text.encode("utf-8")
        while len(rest) > 0:
            wrote = host_file_write(self._fd, rest, len(rest))
            if wrote <= 0:
                raise OSError("[Errno 5] Input/output error: " + self.name)
            rest = rest[wrote:]
        # THE LENGTH IN CHARACTERS, not in bytes: CPython's text stream
        # answers what it was given, and a non-ASCII string encodes longer.
        return len(text)

    def writelines(self, lines):
        for one in lines:
            self.write(one)

    def flush(self):
        # NOTHING TO FLUSH: `host_file_write` reaches the descriptor on the
        # way out, so there is no buffer of this file's own to empty.
        # Answering None rather than refusing is what makes a stream here
        # substitutable for one a program wrote.
        return None

    def fileno(self):
        return self._fd

    def isatty(self):
        # FALSE RATHER THAN A GUESS. There is no `isatty` primitive in
        # `objects/hostsvc.py`, and a program asking is deciding whether to
        # colour its output -- for which "no" is the answer that cannot
        # corrupt a redirected stream.
        return False

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False

    def read(self, size=-1):
        raise OSError("not readable")

    def readline(self, size=-1):
        raise OSError("not readable")

    def close(self):
        # STANDARD OUTPUT IS NOT CLOSABLE. CPython lets it be closed and
        # every later write then raises; closing the process's own output
        # from a library is a mistake with no recovery, so this refuses.
        raise ValueError("cannot close " + self.name)

    @property
    def closed(self):
        return False

    def __repr__(self):
        return ("<_io.TextIOWrapper name=" + repr(self.name)
                + " mode='w' encoding='utf-8'>")


stdout = _Console(1, "<stdout>")
stderr = _Console(2, "<stderr>")


# ── the command line ─────────────────────────────────────────────────────
#
# A REAL LIST, BUILT ONCE, exactly as CPython's is. A program may assign to
# `sys.argv`, slice it, or hand it to `argparse`, and every one of those needs
# an ordinary mutable list rather than a view that asks the host again -- which
# is also why this is read eagerly here rather than on first use: a lazy
# `argv` would answer the command line as it is NOW to a program that had
# already replaced it.
#
# WHERE THE WORDS COME FROM is `objects/hostsvc.py`'s `env` group. A compiled
# binary gets them from its `main`, which hands them to `apy_host_args_take`
# on the way in; `uasm run` gets them from the CLI, which sets
# `interp.argv` before it runs anything. Both routes answer the same two
# primitives, so a program reads the same list whichever one compiled it.

def _read_arg(i):
    """Word `i` of the command line, as `str`.

    A GROWING BUFFER, the shape `bundled/os.py`'s `_getenv_raw` uses and for
    the same reason: `host_arg_get` answers the length it NEEDED rather than
    the length it wrote, so a caller whose first guess was too small asks
    again with a buffer that big instead of keeping a truncated answer.

    `surrogateescape` IS NOT DECORATION. A command line is bytes on every
    platform this targets and nothing promises they are UTF-8 -- a file name
    typed in another encoding reaches `argv` verbatim. CPython decodes it
    with this error handler (that is what `os.fsdecode` is), so the bytes
    survive as lone surrogates and re-encode to exactly what came in. A plain
    `decode("utf-8")` raised instead, and raised INSIDE THE MODULE BODY --
    so one undecodable argument killed every program that imported `sys`,
    including the ones that never read `argv`.
    """
    cap = 256
    buf = bytearray(cap)
    got = host_arg_get(i, buf, cap)
    if got < 0:
        # THE EMPTY STRING RATHER THAN A REFUSAL. The only way here is a
        # count that disagrees with what `host_arg_get` will answer, and a
        # program that cannot import `sys` because of it is worse served than
        # one whose argument list has a gap in it.
        return ""
    if got > cap:
        cap = got
        buf = bytearray(cap)
        got = host_arg_get(i, buf, cap)
        if got < 0:
            return ""
    return bytes(buf[:got]).decode("utf-8", "surrogateescape")


def _read_argv():
    out = []
    n = host_arg_count()
    i = 0
    while i < n:
        out.append(_read_arg(i))
        i = i + 1
    # NEVER EMPTY, which is CPython's rule and not a convenience: `sys.argv[0]`
    # is documented to be the empty string when there was no script name, so a
    # program that reads it unguarded -- and most do -- works on a target with
    # no command line instead of raising IndexError there and nowhere else.
    if len(out) == 0:
        out.append("")
    return out


argv = _read_argv()

def exit(status=None):
    """`sys.exit(status)` -- RAISE `SystemExit`, do not stop the process.

    That is the whole of what CPython's does, and the distinction matters:
    the exception unwinds, so every `finally` runs and an enclosing
    `except SystemExit` may decline to exit at all. A function that ended
    the process here would skip both.

    None MEANS NO STATUS, and this is the one place the two spellings of
    "nothing" part company. `sys.exit(None)` and `sys.exit()` are the same
    call -- CPython raises a SystemExit with NO arguments for both, so
    `e.args` is `()` -- while the CONSTRUCTOR keeps what it was handed:
    `SystemExit(None).args` really is `(None,)`. So the None is dropped
    HERE rather than passed on, which is why this takes an ordinary default
    and not a sentinel; a sentinel would have preserved a None nobody wants.

    THE STATUS IS READ BACK OFF THE EXCEPTION, as `e.code` -- see
    `objects/host.py`'s Exc arm, which answers it from `args`. 0 and None
    both mean success, any other int is the status, and anything else is a
    MESSAGE: CPython prints it to stderr and exits 1, which is what makes
    `sys.exit("no such file")` a complete way to fail.
    """
    if status is None:
        raise SystemExit()
    raise SystemExit(status)


def _print(*args, sep=" ", end="\n", file=None, flush=False):
    """What `print` becomes in a program that replaces `sys.stdout`.

    `stdout` IS READ AT CALL TIME, which is the entire point: the program
    assigns to it between two calls and the second goes somewhere else. A
    default argument of `stdout` would have captured the object at definition
    time, which is before the program has run at all.

    `str()` ON EACH ARGUMENT, and `sep`/`end` of None meaning their defaults
    -- both are CPython's, and `print(None)` printing `None` rather than
    nothing depends on the first.
    """
    where = stdout if file is None else file
    text = ""
    first = True
    for one in args:
        if not first:
            text = text + (" " if sep is None else sep)
        first = False
        text = text + (one if isinstance(one, str) else str(one))
    text = text + ("\n" if end is None else end)
    where.write(text)
    if flush:
        where.flush()
    return None


# ── PEP 553: breakpoint() ────────────────────────────────────────────────

def __breakpointhook__(*args, **kwargs):
    """The DEFAULT `breakpoint()`, which has no debugger to enter.

    Kept under CPython's own name for it so a program that replaced
    `sys.breakpointhook` can put the original back, which is the only reason
    CPython keeps the second name either.
    """
    raise RuntimeError(
        "breakpoint() has no debugger to enter in a compiled program -- "
        "there is no pdb here. Assign sys.breakpointhook to call your own. "
        "See bundled/sys.py.")


def breakpointhook(*args, **kwargs):
    """`breakpoint(*args, **kwargs)` calls this, and a program REPLACES it.

    `breakpoint` is rewritten to this name by `bundled.py`'s
    `_Rewrite.visit_Name`, so the call site reads whatever is bound here when
    it runs -- which is what makes `sys.breakpointhook = ...` take effect for
    a `breakpoint()` written before the assignment.
    """
    return __breakpointhook__(*args, **kwargs)


# ── PEP 578: audit hooks ─────────────────────────────────────────────────

#: Every hook `addaudithook` was given, in the order they were added, which
#: is the order `audit` calls them in. There is no removal -- see the module
#: docstring.
_audit_hooks = []


def addaudithook(hook):
    """Add an auditing hook. CPython has no way to remove one, and neither
    has this."""
    if not callable(hook):
        raise TypeError("expected a callable, not " + type(hook).__name__)
    _audit_hooks.append(hook)
    return None


def audit(event, *args):
    """Raise an audit event, calling every hook with `(event, args)`.

    THE ARGUMENTS ARE ONE TUPLE, not spread: a hook is written
    `def hook(event, args)` and reads `args[0]`, which is CPython's shape and
    what a hook written against CPython expects.

    NOTHING IN THIS RUNTIME CALLS THIS. See the module docstring: only the
    program's own events exist here.
    """
    if not isinstance(event, str):
        raise TypeError("expected str for event, not "
                        + type(event).__name__)
    for hook in _audit_hooks:
        hook(event, args)
    return None


# ── PEP 669: sys.monitoring ──────────────────────────────────────────────

class _Events:
    """The event bitmask constants, WITH CPython'S OWN VALUES.

    The numbers matter: a program combines them with `|` and compares the
    result against what `get_events` answers, so a set of values that were
    merely distinct would agree with itself and with nothing else.
    """

    NO_EVENTS = 0
    PY_START = 1
    PY_RESUME = 2
    PY_RETURN = 4
    PY_YIELD = 8
    CALL = 16
    LINE = 32
    INSTRUCTION = 64
    JUMP = 128
    BRANCH_LEFT = 256
    BRANCH_RIGHT = 512
    STOP_ITERATION = 1024
    RAISE = 2048
    EXCEPTION_HANDLED = 4096
    PY_UNWIND = 8192
    PY_THROW = 16384
    RERAISE = 32768
    C_RETURN = 65536
    C_RAISE = 131072
    BRANCH = 262144


class _Monitoring:
    """PEP 669's namespace -- the TOOL REGISTRY, really.

    A CLASS AND ONE INSTANCE, because `sys.monitoring` is a submodule in
    CPython and there are no module objects here. `bundled.py`'s splice
    records a module-level assignment as a member for exactly this shape --
    the comment there names `sys.monitoring` as the reason.

    NO CALLBACK EVER FIRES. What is real is the bookkeeping: which tool id is
    taken, by what name, and what each was asked to watch. See the module
    docstring.
    """

    #: CPython's four named tools. Ids 3 and 4 have no name there either;
    #: `use_tool_id` accepts them, which is why the bound is 5 rather than
    #: the number of names.
    DEBUGGER_ID = 0
    COVERAGE_ID = 1
    PROFILER_ID = 2
    OPTIMIZER_ID = 5

    def __init__(self):
        self.events = _Events()
        # `MISSING` and `DISABLE` are SENTINELS a callback compares against
        # by identity, so what matters is that each is one object distinct
        # from every other value -- which is all CPython promises about them.
        self.MISSING = object()
        self.DISABLE = object()
        self._tools = {}
        self._watched = {}
        self._callbacks = {}

    def _check(self, tool_id):
        if not isinstance(tool_id, int) or isinstance(tool_id, bool) \
                or tool_id < 0 or tool_id > 5:
            raise ValueError("invalid tool " + str(tool_id)
                             + " (must be between 0 and 5)")
        return tool_id

    def use_tool_id(self, tool_id, name):
        """Claim `tool_id` under `name`. REFUSES A SECOND CLAIM, which is the
        whole purpose: two profilers cannot both own one id."""
        self._check(tool_id)
        if tool_id in self._tools:
            raise ValueError("tool " + str(tool_id) + " is already in use")
        self._tools[tool_id] = name
        self._watched[tool_id] = 0
        return None

    def free_tool_id(self, tool_id):
        """Release `tool_id`. NOT AN ERROR when it was never claimed, which
        is what lets a `finally` clause hold one unconditionally."""
        self._check(tool_id)
        if tool_id in self._tools:
            del self._tools[tool_id]
        if tool_id in self._watched:
            del self._watched[tool_id]
        return None

    def get_tool(self, tool_id):
        """The name `tool_id` was claimed under, or None."""
        self._check(tool_id)
        return self._tools.get(tool_id)

    def clear_tool_id(self, tool_id):
        """Forget what `tool_id` was watching, keeping the claim itself."""
        self._check(tool_id)
        self._watched[tool_id] = 0
        return None

    def get_events(self, tool_id):
        """The event mask `tool_id` last set, or 0."""
        self._check(tool_id)
        return self._watched.get(tool_id, 0)

    def set_events(self, tool_id, event_set):
        """Record what `tool_id` watches. RECORDED AND NOT ACTED ON -- see
        the class docstring."""
        self._check(tool_id)
        if tool_id not in self._tools:
            raise ValueError("tool " + str(tool_id) + " is not in use")
        self._watched[tool_id] = event_set
        return None

    def register_callback(self, tool_id, event, func):
        """Record a callback and answer the one it replaced, which is what
        CPython answers -- None the first time."""
        self._check(tool_id)
        key = (tool_id, event)
        was = self._callbacks.get(key)
        if func is None:
            if key in self._callbacks:
                del self._callbacks[key]
        else:
            self._callbacks[key] = func
        return was

    def get_local_events(self, tool_id, code):
        self._check(tool_id)
        return 0

    def set_local_events(self, tool_id, code, event_set):
        self._check(tool_id)
        return None

    def restart_events(self):
        return None


monitoring = _Monitoring()
