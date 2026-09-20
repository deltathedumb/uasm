"""Host services: THE WHOLE CONTRACT between a frontend and a backend.

WHAT THIS IS. One table of named operations with fixed signatures. A frontend
emits calls to these names and knows nothing else about the target; a backend
implements them and is then a complete backend. Nothing else crosses.

That is the destination rather than today's state, and the difference is worth
being exact about. A backend today ALSO owes whatever of the object runtime is
not yet written in the machine subset -- 394 symbols as this is written, and
docs/INERT-RUNTIME.md is the work of removing them. When that finishes, this
table is all that is left. The two halves are the same project seen from
opposite ends: port the runtime so the obligation shrinks, and name the
obligation so it stops growing.

THE FLOOR IS THE MANDATORY GROUP OF THIS TABLE, not a separate thing.
`objects/floor.py` holds the contracts and the C for `plat_write`,
`plat_exit` and `plat_heap`, and the whole of stage 2 was an argument for why
that number is three. Nothing here changes it -- `core` below IS that set,
read from there so there is one list. What this file adds is everything a real
program needs that a bare-metal target cannot have: a filesystem, a clock,
entropy, a network, a character database.

WHY THOSE CANNOT SIMPLY JOIN THE FLOOR. Because the floor is what EVERY
backend owes, and stage 2's achievement was getting it from five to three. A
mandatory thirty would undo it and would make a target without a filesystem
impossible to write a backend for. So the rest of this table is OPTIONAL and
DECLARED: a backend says which groups it provides, and a program using one it
does not have is refused at compile time.

WHY NOT `ctypes`, WHICH ALREADY WORKS. `frontends/python/cffi.py` resolves a
native symbol at COMPILE time -- a promise to the linker, not a `dlopen`. That
is exactly right for what it is, and it is why `bundled/pathlib.py` can call
`_open` and `GetFileAttributesA`. But a promise to the linker is a promise
only a LINKING backend can keep: the JVM backend has no linker and no `_open`,
and a bare-metal target has neither. So `pathlib` works on the C backend and
cannot be made to work anywhere else, and the same would be true of every
module that touches a file. The symbol names are the problem -- `_open` is
MSVC's spelling, `open` is POSIX's, and `java.nio` is neither.

So: a NAMED SET OF OPERATIONS with fixed signatures, which each backend
satisfies however it can. The C backend calls libc. The JVM backend calls
`java.nio`. The interpreter calls Python's `os`. A frontend calls one name and
does not know which.

THE ONE RULE, INHERITED FROM THE FLOOR, AND IT IS THE SAME RULE. Nothing here
may know what a Python value is. `plat_write` takes bytes, not a str, and the
floor's own documentation says why: `put_bool` knows that Python spells a true
value `True`, so every backend implementing it owes the LANGUAGE rather than
the machine. Every signature below takes and answers machine words and byte
buffers. A backend author implementing all of them still has not been told
what a `list` is.

NOT AN OPCODE, and the floor is the precedent. `plat_write` is an ordinary
`Op.CALL` of an external symbol; nothing in the instruction set knows it
exists. An opcode would have to be implemented by all five backends plus the
verifier, printer, liveness and interpreter -- eleven places -- to express
what a call with a signature already expresses. What a new opcode buys is a
new SHAPE of instruction, and these are not a new shape; they are calls.

WHAT A GROUP IS. A capability a target either has or does not: a filesystem, a
network stack, a clock. Grouping rather than listing each operation is
deliberate -- a target with files has all of the file operations or none of
them, and a backend author answering "do you have a filesystem" once is a
better question than answering it eleven times.
"""
from __future__ import annotations

from .floor import FLOOR as _FLOOR

#: THE ERROR CODES, and they are NOT `errno`.
#:
#: `errno` is a C concept, it is thread-local, and its numbers differ between
#: platforms -- which is the portability bug this file exists to avoid, in
#: miniature. So an operation that fails answers a NEGATIVE number from this
#: table and nothing else, and a caller that wants to tell "no such file" from
#: "permission denied" gets the same answer on every target.
#:
#: SMALL AND CLOSED. Every code here is one a caller can act on differently.
#: A richer set would be a translation table in every backend, and the ones
#: nobody branches on would drift.
ERRORS: dict[str, int] = {
    "HOST_ERR": -1,          # something failed and nothing more is known
    "HOST_ENOENT": -2,       # no such file or directory
    "HOST_EACCES": -3,       # permission denied
    "HOST_EEXIST": -4,       # already exists
    "HOST_ENOTDIR": -5,      # a path component is not a directory
    "HOST_ENOTEMPTY": -6,    # a directory that is not empty
    "HOST_EAGAIN": -7,       # would block; try again
    "HOST_EPIPE": -8,        # the other end is gone
    "HOST_EINVAL": -9,       # the arguments do not make sense
}

#: HOW A FILE IS OPENED, and NOT libc's `O_*` flags.
#:
#: `O_BINARY` is 0x8000 on MSVC and does not exist on POSIX; `O_CREAT` is 0x100
#: on one platform and 0o100 on another. A frontend that passed those through
#: would be writing platform-specific code in a portable module, which is what
#: `bundled/pathlib.py` had to do and what this replaces.
#:
#: ALWAYS BINARY. Newline translation is a property of TEXT, and text is a
#: language concept -- `str` knows about it and this layer does not. A frontend
#: that wants CRLF writes CRLF.
OPEN_MODES: dict[str, int] = {
    "HOST_OPEN_READ": 0,     # must exist
    "HOST_OPEN_WRITE": 1,    # created if absent, truncated if present
    "HOST_OPEN_APPEND": 2,   # created if absent, position at the end
    "HOST_OPEN_UPDATE": 3,   # read and write, must exist, not truncated
}

#: WHAT A PATH IS, as `host_file_kind` answers it.
#:
#: THREE NUMBERS RATHER THAN A `struct stat`. A struct means a layout every
#: backend agrees on and a frontend can read, which is a second ABI to keep in
#: step; three quarters of what `stat` is asked for is this question, and
#: `bundled/pathlib.py` reached for `GetFileAttributesA` to ask exactly it.
#: Size is a separate call because it is the other quarter.
KINDS: dict[str, int] = {
    "HOST_KIND_MISSING": 0,
    "HOST_KIND_FILE": 1,
    "HOST_KIND_DIR": 2,
    "HOST_KIND_OTHER": 3,    # a device, a socket, a symlink to nothing
}

#: WHERE A SEEK STARTS FROM.
SEEK: dict[str, int] = {
    "HOST_SEEK_SET": 0,
    "HOST_SEEK_CUR": 1,
    "HOST_SEEK_END": 2,
}

#: THE OPERATIONS, grouped by the capability a target either has or has not.
#:
#: name -> (argument IR types, result IR type)
#:
#: A PATH IS A POINTER AND A LENGTH, never a NUL-terminated string. The IR has
#: no C-string convention, `plat_write` already takes a buffer and a count, and
#: a JVM backend handed a bare pointer would have to scan for a terminator it
#: has no reason to believe in. It also means a path may contain a NUL and be
#: rejected honestly rather than silently truncated.
#:
#: DESCRIPTORS SHARE THE FLOOR'S NUMBERING. 0, 1 and 2 are standard input,
#: output and error, so `host_file_write(1, ...)` writes where
#: `plat_write(1, ...)` writes. Two numbering schemes for the same thing is
#: how a program ends up with interleaved output nobody can explain.
GROUPS: dict[str, dict[str, tuple[tuple[str, ...], str]]] = {
    # ── the floor: emit bytes, stop, get memory ─────────────────────────
    #
    # THE ONE MANDATORY GROUP, and it is `objects/floor.py`'s list rather than
    # a copy of it -- a hand-kept second copy of a signature list drifted
    # three times in one afternoon the last time this project kept two.
    # docs/INERT-RUNTIME.md stage 2 is the argument for why it is these three
    # and not the five it used to be, and that argument is unchanged.
    "core": dict(_FLOOR),
    # ── a filesystem ────────────────────────────────────────────────────
    "file": {
        "host_file_open":   (("ptr", "i64", "i64"), "i64"),
        "host_file_read":   (("i64", "ptr", "i64"), "i64"),
        "host_file_write":  (("i64", "ptr", "i64"), "i64"),
        "host_file_close":  (("i64",), "i64"),
        "host_file_seek":   (("i64", "i64", "i64"), "i64"),
        "host_file_kind":   (("ptr", "i64"), "i64"),
        "host_file_size":   (("ptr", "i64"), "i64"),
        "host_file_remove": (("ptr", "i64"), "i64"),
        "host_dir_make":    (("ptr", "i64"), "i64"),
        "host_dir_remove":  (("ptr", "i64"), "i64"),
    },
    # ── a clock ─────────────────────────────────────────────────────────
    #
    # NANOSECONDS AS AN i64, which runs to the year 2262 from the epoch and is
    # what every modern platform's clock answers anyway. A float would lose
    # precision at exactly the scale a profiler cares about, and a pair of
    # words would need a struct.
    #
    # TWO CLOCKS BECAUSE THERE ARE TWO QUESTIONS. `host_time_unix` answers
    # what time it is and may jump backwards when the machine is corrected;
    # `host_time_monotonic` never goes backwards and means nothing absolute.
    # A program that measures a duration with the first has a bug that appears
    # twice a year.
    "time": {
        "host_time_unix":      ((), "i64"),
        "host_time_monotonic": ((), "i64"),
        "host_sleep":          (("i64",), "i64"),
    },
    # ── entropy ─────────────────────────────────────────────────────────
    #
    # THE SYSTEM'S, not a PRNG. A backend fills the buffer from whatever its
    # platform calls cryptographically secure. A seeded generator is a
    # LANGUAGE feature -- `random.Random` is reproducible on purpose -- and
    # belongs above this line, seeded from here.
    "random": {
        "host_random_bytes": (("ptr", "i64"), "i64"),
    },
    # ── the environment the process was started in ──────────────────────
    #
    # COPIED OUT INTO A CALLER'S BUFFER, answering the length it needed. A
    # caller that guessed too small gets the true length and calls again,
    # which is the only shape that works without the layer allocating -- and
    # the layer must not allocate, because who frees it is a question with a
    # different answer in every backend.
    "env": {
        "host_env_get":  (("ptr", "i64", "ptr", "i64"), "i64"),
        "host_arg_count": ((), "i64"),
        "host_arg_get":  (("i64", "ptr", "i64"), "i64"),
    },
    # ── a network ───────────────────────────────────────────────────────
    #
    # STREAMS ONLY, and blocking. Datagrams, non-blocking sockets and TLS are
    # each a larger contract than this whole file, and a backend that has one
    # can offer it as a group of its own rather than by widening this one.
    # A DESCRIPTOR, and the same kind of number the `file` group hands out:
    # opaque, belonging to whoever answered it, given back and never
    # interpreted. `host_net_connect` takes a host name and a port and
    # answers one; `host_net_listen` takes a port and a backlog and answers
    # the LISTENER's, which `host_net_accept` turns into a connection's.
    # `host_net_read` answers 0 at end of stream, which is distinguishable
    # from every error because every error here is negative.
    #
    # `host_net_port` IS HOW A CALLER LEARNS WHICH PORT IT GOT. Passing 0 to
    # `host_net_listen` asks the operating system for a free one -- what
    # every ephemeral server and every test does -- and without this the
    # answer was unobtainable, so the group could only be used by a program
    # willing to hard-code a number and race whatever else chose the same.
    #
    # `host_net_ready` ASKS WHETHER ONE DESCRIPTOR WOULD BLOCK, and is what
    # `select` is built from. `want` is 1 for readable and 2 for writable;
    # `timeout` is nanoseconds, with a negative value meaning wait
    # indefinitely and 0 meaning poll. It answers 1 for ready, 0 for timed
    # out, and a negative code for an error. ONE DESCRIPTOR AND NOT A SET,
    # because a set means an array layout every backend agrees on and a
    # frontend can build -- a second ABI to keep in step, for a loop the
    # caller can write. `bundled/select.py` writes that loop.
    "net": {
        "host_net_connect": (("ptr", "i64", "i64"), "i64"),
        "host_net_listen":  (("i64", "i64"), "i64"),
        "host_net_accept":  (("i64",), "i64"),
        "host_net_read":    (("i64", "ptr", "i64"), "i64"),
        "host_net_write":   (("i64", "ptr", "i64"), "i64"),
        "host_net_close":   (("i64",), "i64"),
        "host_net_port":    (("i64",), "i64"),
        "host_net_ready":   (("i64", "i64", "i64"), "i64"),
    },
    # ── threads ─────────────────────────────────────────────────────────
    #
    # THE ONE GROUP WHOSE ABSENCE IS NOT ABOUT HARDWARE. A target without a
    # filesystem is a target without a filesystem; a target without threads
    # is usually one whose RUNTIME cannot be re-entered, which is a property
    # of the backend rather than of the machine. Either way the answer is
    # the same as for the rest: declare it, or be refused by name.
    #
    # A HANDLE IS OPAQUE, as everywhere else in this file: the C backend
    # widens a `pthread_t`, another may answer an index into a table. A
    # caller may only hand it back.
    #
    # THE THREAD FUNCTION IS A FUNCTION POINTER AND AN ARGUMENT, which is
    # what `Op.FUNC_ADDR` already produces and `Op.CALL_PTR` already calls:
    # nothing new is needed in the instruction set to start one. It takes a
    # `ptr` and answers an `i64`, which is `thrd_start_t` with the
    # `void *`/`int` spelled as the machine words they are.
    #
    # WHY MUTEXES AND CONDITION VARIABLES ARE HERE and not built above this
    # line out of something smaller. A spin lock needs an atomic exchange,
    # which the IR does not have and should not grow for one caller; a
    # condition variable needs a way to sleep until woken, which nothing
    # here can express. Both are primitive to the host, so both are named.
    #
    # `host_mutex_new` TAKES A KIND: 0 is plain and 1 is recursive, which is
    # C's `mtx_plain` and `mtx_recursive`. A timed mutex is not a kind --
    # `host_mutex_timedlock` takes the timeout, so the same mutex can be
    # waited on either way.
    "thread": {
        "host_thread_start":  (("ptr", "ptr"), "i64"),
        "host_thread_join":   (("i64", "ptr"), "i64"),
        "host_thread_detach": (("i64",), "i64"),
        "host_thread_self":   ((), "i64"),
        "host_thread_yield":  ((), "i64"),
        "host_thread_exit":   (("i64",), "i64"),
        "host_mutex_new":     (("i64",), "i64"),
        "host_mutex_lock":    (("i64",), "i64"),
        "host_mutex_trylock": (("i64",), "i64"),
        "host_mutex_timedlock": (("i64", "i64"), "i64"),
        "host_mutex_unlock":  (("i64",), "i64"),
        "host_mutex_free":    (("i64",), "i64"),
        "host_cond_new":      ((), "i64"),
        "host_cond_wait":     (("i64", "i64", "i64"), "i64"),
        "host_cond_signal":   (("i64",), "i64"),
        "host_cond_broadcast": (("i64",), "i64"),
        "host_cond_free":     (("i64",), "i64"),
        # PER-THREAD STORAGE, which is four operations and not a storage
        # class: a key, a pointer per thread under it, and a destructor the
        # host runs when a thread ends. `_Thread_local` is compiled onto
        # these too -- see the C frontend's `lower._tls_slot`.
        "host_tss_new":       (("ptr",), "i64"),
        "host_tss_get":       (("i64",), "ptr"),
        "host_tss_set":       (("i64", "ptr"), "i64"),
        "host_tss_free":      (("i64",), "i64"),
    },
    # ── another program ─────────────────────────────────────────────────
    #
    # RUN TO COMPLETION AND CAPTURE, which is one operation rather than the
    # six a `Popen` would need. That is the shape because of what a
    # single-threaded runtime can actually do: a pipe you write to while the
    # child writes back needs someone to read the other end, and there is
    # nobody. `subprocess.run(capture_output=True)` -- start it, wait for it,
    # collect what it said -- is the whole of what a program here can ask
    # for, so it is the whole of what this promises.
    #
    # ARGV IS NUL-SEPARATED, in one buffer, with a count. Not a shell string:
    # quoting a list into one is where command injection comes from, and the
    # separation a caller already has must not be thrown away and guessed at
    # again. Not an array of pointers either -- that is a layout every
    # backend would have to agree on, and this file's own rule is that a
    # string crosses as a pointer and a length.
    #
    # THE TWO CAPTURE BUFFERS ANSWER THE LENGTH THEY NEEDED, exactly as
    # `host_env_get` does: a caller that guessed too small sees a number
    # larger than its buffer and calls again with a bigger one. The exit
    # STATUS is written through `status`, because the return value is
    # already carrying the stdout length and one call cannot answer two
    # numbers.
    "proc": {
        "host_proc_run": (("ptr", "i64", "i64", "ptr", "i64", "ptr", "i64",
                           "ptr"), "i64"),
    },
    # ── a dynamic library ───────────────────────────────────────────────
    #
    # WHY THIS EXISTS WHEN `ctypes` ALREADY WORKS, and the answer is the same
    # one this file's header gives for `file`. `frontends/python/cffi.py`
    # resolves a native symbol at COMPILE time -- a promise to the linker --
    # which is exactly right and is why `CDLL("m")` needs nothing here. But a
    # promise to the linker requires the library and the symbol to be
    # LITERALS, because there has to be a name to hand the linker, and
    # `E0112` refuses every program that computes one. A plugin directory
    # read at startup is not a program that can be written that way.
    #
    # SO THIS IS THE OTHER HALF, not a replacement. A backend that has both
    # keeps the link-time path for a literal -- it is faster, it needs no
    # group, and a missing symbol is caught while building -- and reaches
    # here only when the name is not known until the program runs.
    #
    # AN ADDRESS, AND THE IR ALREADY KNOWS WHAT TO DO WITH ONE. `Op.CALL_PTR`
    # calls through a value with a declared signature, so nothing new is
    # needed in the instruction set to call what `host_dl_sym` answers. That
    # is why this group is three operations and not four: "call it" is not a
    # host service, it is an instruction that already exists.
    #
    # A HANDLE IS OPAQUE and belongs to whoever answered it. The C backend
    # widens `HMODULE`/`void *`; a backend with a table of its own may answer
    # an index. A caller may only hand it back.
    #
    # ZERO IS "NO SUCH SYMBOL", not an error, and it is distinguishable from
    # one because every error in this table is NEGATIVE. A symbol's address is
    # never zero, so a caller tests for it directly rather than needing a
    # second call to ask whether the last one failed -- which is what
    # `dlerror` is, and it is thread-local global state this layer will not
    # have.
    "dynlib": {
        "host_dl_open":  (("ptr", "i64"), "i64"),
        "host_dl_sym":   (("i64", "ptr", "i64"), "i64"),
        "host_dl_close": (("i64",), "i64"),
    },
    # ── the character database ──────────────────────────────────────────
    #
    # NOT I/O, AND THAT IS THE POINT OF PUTTING IT HERE. A host operation is
    # anything a backend can do that the IR cannot express and that differs by
    # target -- the JVM has `Character.toUpperCase` and a generated table would
    # be dead weight beside it, while the C backend has the table and no JVM.
    # Same question, two answers, one name.
    #
    # CODE POINTS, NOT BYTES. Case folding is defined on characters, and a
    # layer taking bytes would have to pick an encoding. `str_code.py` already
    # turns UTF-8 into code points in the subset.
    "text": {
        "host_char_upper": (("i64",), "i64"),
        "host_char_lower": (("i64",), "i64"),
        "host_char_class": (("i64",), "i64"),
    },
}

#: The groups a backend owes no matter what. Everything else is declared.
MANDATORY = ("core",)

#: The groups a backend may or may not offer.
OPTIONAL = tuple(g for g in ("file", "time", "random", "env", "net",
                             "proc", "dynlib", "text", "thread"))

#: Every operation, flattened, for the places that want one dictionary.
ALL: dict[str, tuple[tuple[str, ...], str]] = {
    name: sig for ops in GROUPS.values() for name, sig in ops.items()}

#: Which group an operation belongs to, so a refusal can name the capability
#: rather than only the function.
GROUP_OF: dict[str, str] = {
    name: group for group, ops in GROUPS.items() for name in ops}

NAMES = tuple(ALL)


def group_of(name: str) -> str | None:
    """The capability `name` needs, or None if it is not a host service."""
    return GROUP_OF.get(name)


def signature(name: str):
    """The IR signature of one operation, or None."""
    return ALL.get(name)


#: THE C IMPLEMENTATION, per group. `@STATIC@` and `@PTR@` are substituted
#: exactly as `objects/floor.py` substitutes them, because a second convention
#: would be a second thing to keep in step.
#:
#: PER GROUP, and that is not tidiness. A bare-metal C target declares no
#: `file` group, and emitting `fopen` for it would fail to link -- which is
#: the same portability bug in C that `ctypes` had in the frontend. A backend
#: gets exactly the groups it said it has.
C_SOURCE: dict[str, str] = {}

C_SOURCE["file"] = r"""/* --- host services: file ------------------------------------------------ */

/* EVERY PLATFORM FUNCTION THIS NEEDS IS DECLARED HERE, and no header is
   included for them. `<sys/stat.h>` and `<direct.h>` both pull in `<io.h>` on
   MinGW, which declares `_open`, `_read`, `_write` and `_close` -- the very
   names a `ctypes` program declares for itself, and two prototypes for one
   symbol do not compile. Including them would have re-created the obstacle
   `cffi.py` documents, from the other side, and broken every program that
   reaches libc through `ctypes`.

   The prototypes below are the platform's own, written out. That is a real
   cost -- a wrong one is undefined behaviour rather than a compile error --
   which is why there are five of them and not fifty. */
#ifdef _WIN32
int _mkdir(const char *);
int _rmdir(const char *);
#define APY_HOST_MKDIR(p) _mkdir(p)
#else
int mkdir(const char *, unsigned int);
int rmdir(const char *);
#define APY_HOST_MKDIR(p) mkdir((p), 0777)
#define _rmdir rmdir
#endif

/* DIRECTORY DETECTION WITHOUT `stat`, which is what avoids the header. A
   directory is the thing `opendir` accepts and `fopen` does not; `DIR *` is a
   pointer on every platform, so declaring the return as `void *` is
   ABI-identical and needs no `<dirent.h>` either. */
void *opendir(const char *);
int closedir(void *);

/* A PATH ARRIVES AS A POINTER AND A LENGTH and every platform call below
   wants a NUL-terminated string, so it is copied into a bounded buffer. The
   copy is also where an embedded NUL is caught: a path containing one is
   rejected rather than silently truncated at it, which is a real class of
   security bug and costs one comparison to close. */
#define APY_HOST_PATH_MAX 4096
static int apy_host_path(@PTR@ p, int64_t n, char *out)
{
    int64_t i;
    if (n < 0 || n >= APY_HOST_PATH_MAX) return 0;
    for (i = 0; i < n; i++) {
        char c = ((const char *)p)[i];
        if (c == 0) return 0;
        out[i] = c;
    }
    out[n] = 0;
    return 1;
}

/* `errno` TRANSLATED ONCE, HERE. The whole reason this layer exists is that
   `errno` numbers differ between platforms, so a caller must never see one.
   Anything not named becomes the generic failure rather than a number the
   caller would have to look up. */
static int64_t apy_host_err(void)
{
    switch (errno) {
    case ENOENT:  return -2;
    case EACCES:  return -3;
    case EPERM:   return -3;
    case EEXIST:  return -4;
    case ENOTDIR: return -5;
    case ENOTEMPTY: return -6;
    case EAGAIN:  return -7;
    case EPIPE:   return -8;
    case EINVAL:  return -9;
    default:      return -1;
    }
}

@STATIC@int64_t host_file_open(@PTR@ path, int64_t n, int64_t mode)
{
    char buf[APY_HOST_PATH_MAX];
    const char *how;
    FILE *f;
    if (!apy_host_path(path, n, buf)) return -9;
    /* BINARY ALWAYS. Newline translation is a property of TEXT and text is a
       language concept; a frontend that wants CRLF writes CRLF. Without the
       `b` this layer would silently rewrite a program's bytes on Windows. */
    if (mode == 0)      how = "rb";
    else if (mode == 1) how = "wb";
    else if (mode == 2) how = "ab";
    else if (mode == 3) how = "r+b";
    else return -9;
    f = fopen(buf, how);
    if (!f) return apy_host_err();
    /* THE HANDLE IS A `FILE *` WIDENED, not an index into a table this file
       keeps. A table would need a size, a policy for exhausting it, and would
       not survive a backend that wanted its own; a pointer is what the
       platform already gave us and the caller only ever hands it back. */
    return (int64_t)(intptr_t)f;
}

/* THE THREE STANDARD DESCRIPTORS BY NUMBER, so `host_file_write(1, ...)`
   writes where `plat_write(1, ...)` writes. Two numbering schemes for the
   same thing is how a program ends up with interleaved output nobody can
   explain. Every other value is a handle `host_file_open` answered. */
static FILE *apy_host_stream(int64_t fd)
{
    if (fd == 0) return stdin;
    if (fd == 1) return stdout;
    if (fd == 2) return stderr;
    return (FILE *)(intptr_t)fd;
}

@STATIC@int64_t host_file_read(int64_t fd, @PTR@ buf, int64_t n)
{
    FILE *s = apy_host_stream(fd);
    size_t got;
    if (n < 0) return -9;
    got = fread((void *)buf, 1, (size_t)n, s);
    if (got == 0 && ferror(s)) return apy_host_err();
    return (int64_t)got;
}

@STATIC@int64_t host_file_write(int64_t fd, @PTR@ buf, int64_t n)
{
    FILE *s = apy_host_stream(fd);
    size_t put;
    if (n < 0) return -9;
    put = fwrite((const void *)buf, 1, (size_t)n, s);
    if (put != (size_t)n) return apy_host_err();
    /* FLUSHED, for the same reason `plat_write` is: stdout to a pipe is
       block-buffered, so without this the interleaving of stdout and stderr
       depends on where the output is going. */
    if ((fd == 1 || fd == 2) && fflush(s) != 0) return apy_host_err();
    return (int64_t)put;
}

@STATIC@int64_t host_file_close(int64_t fd)
{
    if (fd >= 0 && fd <= 2) return 0;          /* never close the standard three */
    if (fclose((FILE *)(intptr_t)fd) != 0) return apy_host_err();
    return 0;
}

@STATIC@int64_t host_file_seek(int64_t fd, int64_t off, int64_t whence)
{
    FILE *s = apy_host_stream(fd);
    int w = whence == 1 ? SEEK_CUR : whence == 2 ? SEEK_END : SEEK_SET;
    if (whence < 0 || whence > 2) return -9;
    if (fseek(s, (long)off, w) != 0) return apy_host_err();
    return (int64_t)ftell(s);
}

/* THREE NUMBERS RATHER THAN A `struct stat`, which would be a second ABI for
   a frontend to agree with -- and, as it turns out, a header this file cannot
   afford to include. */
@STATIC@int64_t host_file_kind(@PTR@ path, int64_t n)
{
    char buf[APY_HOST_PATH_MAX];
    void *d;
    FILE *f;
    if (!apy_host_path(path, n, buf)) return -9;
    d = opendir(buf);
    if (d) { closedir(d); return 2; }
    f = fopen(buf, "rb");
    if (f) { fclose(f); return 1; }
    /* NEITHER A DIRECTORY NOR READABLE. `ENOENT` is missing; anything else --
       a permission failure, a device -- is something that IS there and is not
       an ordinary file, which is what kind 3 is for. */
    return errno == ENOENT ? 0 : 3;
}

/* SIZE BY SEEKING TO THE END, because `stat` is the header this cannot have.
   Exact for a regular file, which is the only thing a caller asks about. */
@STATIC@int64_t host_file_size(@PTR@ path, int64_t n)
{
    char buf[APY_HOST_PATH_MAX];
    FILE *f;
    long at;
    if (!apy_host_path(path, n, buf)) return -9;
    f = fopen(buf, "rb");
    if (!f) return apy_host_err();
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    at = ftell(f);
    fclose(f);
    return at < 0 ? -1 : (int64_t)at;
}

@STATIC@int64_t host_file_remove(@PTR@ path, int64_t n)
{
    char buf[APY_HOST_PATH_MAX];
    if (!apy_host_path(path, n, buf)) return -9;
    if (remove(buf) != 0) return apy_host_err();
    return 0;
}

@STATIC@int64_t host_dir_make(@PTR@ path, int64_t n)
{
    char buf[APY_HOST_PATH_MAX];
    if (!apy_host_path(path, n, buf)) return -9;
    if (APY_HOST_MKDIR(buf) != 0) return apy_host_err();
    return 0;
}

@STATIC@int64_t host_dir_remove(@PTR@ path, int64_t n)
{
    char buf[APY_HOST_PATH_MAX];
    if (!apy_host_path(path, n, buf)) return -9;
    if (_rmdir(buf) != 0) return apy_host_err();
    return 0;
}
"""

C_SOURCE["time"] = r"""/* --- host services: time ------------------------------------------------ */

/* NANOSECONDS AS AN i64 -- good to the year 2262 from the epoch, and what
   every modern platform's clock answers anyway. A double would lose precision
   at exactly the scale a profiler cares about. */
@STATIC@int64_t host_time_unix(void)
{
    return (int64_t)time(NULL) * 1000000000;
}

@STATIC@int64_t host_time_monotonic(void)
{
    /* NEVER GOES BACKWARDS, which is the whole difference from the one above:
       a duration measured with a wall clock is wrong twice a year. `clock()`
       is the portable fallback and measures CPU rather than elapsed time --
       stated rather than hidden, because a backend with something better
       should use it. */
    return (int64_t)clock() * (1000000000 / CLOCKS_PER_SEC);
}

@STATIC@int64_t host_sleep(int64_t nanos)
{
    if (nanos <= 0) return 0;
    {
        clock_t until = clock() + (clock_t)(nanos / (1000000000 / CLOCKS_PER_SEC));
        while (clock() < until) { }
    }
    return 0;
}
"""

C_SOURCE["random"] = r"""/* --- host services: random ---------------------------------------------- */

/* THE SYSTEM'S ENTROPY, not a PRNG. A seeded generator is a LANGUAGE feature
   -- `random.Random` is reproducible on purpose -- and belongs above this
   line, seeded from here. */
@STATIC@int64_t host_random_bytes(@PTR@ buf, int64_t n)
{
    unsigned char *out = (unsigned char *)buf;
    int64_t i;
    if (n < 0) return -9;
#ifdef _WIN32
    {
        /* `rand_s` is the CRT's cryptographic one and needs no handle.
           DECLARED HERE because <stdlib.h> hides it behind `_CRT_RAND_S`,
           which has to be defined BEFORE that include -- and this C is
           spliced in after it. Declaring the symbol is the smaller of the two
           evils; the alternative is reaching into how the prelude is built
           from a file that should not know. */
        int rand_s(unsigned int *);
        unsigned int v;
        for (i = 0; i < n; i++) {
            if (rand_s(&v) != 0) return -1;
            out[i] = (unsigned char)(v & 0xFF);
        }
        return n;
    }
#else
    {
        FILE *f = fopen("/dev/urandom", "rb");
        size_t got;
        if (!f) return -1;
        got = fread(out, 1, (size_t)n, f);
        fclose(f);
        if (got != (size_t)n) return -1;
        return n;
    }
#endif
}
"""

C_SOURCE["env"] = r"""/* --- host services: env ------------------------------------------------- */

/* COPIED OUT INTO THE CALLER'S BUFFER, answering the length it NEEDED rather
   than the length it wrote. A caller that guessed too small sees a number
   larger than its buffer and calls again -- which is the only shape that
   works without this layer allocating, and it must not allocate, because who
   frees it has a different answer in every backend.
   -2 for a name that is not set, so "absent" and "empty" stay distinct. */
@STATIC@int64_t host_env_get(@PTR@ name, int64_t n, @PTR@ out, int64_t cap)
{
    char key[512];
    const char *got;
    int64_t len, i;
    if (n < 0 || n >= (int64_t)sizeof key) return -9;
    for (i = 0; i < n; i++) {
        char c = ((const char *)name)[i];
        if (c == 0) return -9;
        key[i] = c;
    }
    key[n] = 0;
    got = getenv(key);
    if (!got) return -2;
    len = (int64_t)strlen(got);
    for (i = 0; i < len && i < cap; i++) ((char *)out)[i] = got[i];
    return len;
}

/* THE COMMAND LINE, stashed by the entry wrapper because C only offers it to
   `main`. Zero arguments is a legitimate answer for a backend whose target has
   no command line. */
static int apy_host_argc = 0;
static char **apy_host_argv = 0;

/* THE ENTRY WRAPPER'S DOOR, and the only writer of the two above. `main` is
   the one place C hands a program its command line, so the wrapper an emitter
   writes around the IR's entry calls this on the way in -- see the C
   backend's `main` in `backends/c/emit.py`, which is where the words come
   from. NOT A DECLARED SERVICE: it is not in the `env` group's table above
   and no frontend can reach it, because a program setting its own command
   line is not a thing a program does.

   A BACKEND WHOSE TARGET HAS NO COMMAND LINE simply never calls it, and
   `host_arg_count` then answers 0 -- which `sys.argv` turns into CPython's
   `['']` rather than an empty list, because CPython's is never empty. */
@STATIC@void apy_host_args_take(int argc, char **argv)
{
    apy_host_argc = argc;
    apy_host_argv = argv;
}

@STATIC@int64_t host_arg_count(void)
{
    return (int64_t)apy_host_argc;
}

@STATIC@int64_t host_arg_get(int64_t i, @PTR@ out, int64_t cap)
{
    int64_t len, k;
    const char *s;
    if (i < 0 || i >= (int64_t)apy_host_argc) return -9;
    s = apy_host_argv[i];
    len = (int64_t)strlen(s);
    for (k = 0; k < len && k < cap; k++) ((char *)out)[k] = s[k];
    return len;
}
"""

C_SOURCE["net"] = r"""/* --- host services: net -------------------------------------------------- */

/* NO HEADER, for the reason the `file` and `dynlib` groups both give at
   length: `<sys/socket.h>` and `<winsock2.h>` declare a great deal besides
   the seven calls wanted here, and every name they declare is one a `ctypes`
   program may declare for itself -- two prototypes for one symbol do not
   compile. So the prototypes are written out, and the two structs with them.

   THE STRUCT IS THE PART THAT HAS TO BE EXACTLY RIGHT, because a wrong
   layout is undefined behaviour rather than a compile error. `sockaddr_in`
   is the same sixteen bytes on every platform this targets -- it is wire
   format, not an ABI choice: a 16-bit family, a 16-bit port in NETWORK byte
   order, a 32-bit address in network byte order, and eight bytes of padding
   nothing reads. Written as a byte array plus explicit stores rather than as
   a struct with named fields, so there is nothing for a compiler to pad
   differently. */
#ifdef _WIN32
/* Winsock needs starting, and its descriptors are `SOCKET` (an unsigned
   pointer-sized handle) rather than ints -- widened to int64_t here, which
   is what the contract answers anyway. */
typedef unsigned long long apy_socket_t;
__declspec(dllimport) int __stdcall WSAStartup(unsigned short, void *);
__declspec(dllimport) apy_socket_t __stdcall socket(int, int, int);
__declspec(dllimport) int __stdcall connect(apy_socket_t, const void *, int);
__declspec(dllimport) int __stdcall bind(apy_socket_t, const void *, int);
__declspec(dllimport) int __stdcall listen(apy_socket_t, int);
__declspec(dllimport) apy_socket_t __stdcall accept(apy_socket_t, void *, int *);
__declspec(dllimport) int __stdcall recv(apy_socket_t, char *, int, int);
__declspec(dllimport) int __stdcall send(apy_socket_t, const char *, int, int);
__declspec(dllimport) int __stdcall closesocket(apy_socket_t);
__declspec(dllimport) int __stdcall setsockopt(apy_socket_t, int, int,
                                               const char *, int);
__declspec(dllimport) int __stdcall getsockname(apy_socket_t, void *, int *);
__declspec(dllimport) unsigned long __stdcall inet_addr(const char *);
__declspec(dllimport) void * __stdcall gethostbyname(const char *);
__declspec(dllimport) int __stdcall WSAPoll(void *, unsigned long, int);
#define APY_NET_CLOSE(s)   closesocket(s)
#define APY_NET_INVALID    ((apy_socket_t)~0)
#define APY_NET_SOL_SOCKET 0xffff
#define APY_NET_REUSEADDR  0x0004
#else
typedef int apy_socket_t;
int socket(int, int, int);
int connect(int, const void *, unsigned int);
int bind(int, const void *, unsigned int);
int listen(int, int);
int accept(int, void *, unsigned int *);
long recv(int, void *, unsigned long, int);
long send(int, const void *, unsigned long, int);
int close(int);
int setsockopt(int, int, int, const void *, unsigned int);
int getsockname(int, void *, unsigned int *);
unsigned int inet_addr(const char *);
void *gethostbyname(const char *);
int poll(void *, unsigned long, int);
#define APY_NET_CLOSE(s)   close(s)
#define APY_NET_INVALID    (-1)
#define APY_NET_SOL_SOCKET 1
#define APY_NET_REUSEADDR  2
#endif

#define APY_AF_INET      2
#define APY_SOCK_STREAM  1

/* `sockaddr_in`, as the sixteen bytes it is. See the comment above on why
   this is a byte array: the layout is wire format and writing it out by hand
   is the only way to be sure a compiler has not padded it. */
static void apy_net_addr(unsigned char *sa, unsigned int ip, int64_t port)
{
    int i;
    for (i = 0; i < 16; i++) sa[i] = 0;
    sa[0] = APY_AF_INET;                       /* sin_family, little end   */
    sa[1] = 0;
    sa[2] = (unsigned char)((port >> 8) & 0xff);   /* sin_port, network    */
    sa[3] = (unsigned char)(port & 0xff);
    /* `inet_addr` already answers network byte order, so the four bytes go
       out in memory order rather than being byte-swapped again. */
    sa[4] = (unsigned char)(ip & 0xff);
    sa[5] = (unsigned char)((ip >> 8) & 0xff);
    sa[6] = (unsigned char)((ip >> 16) & 0xff);
    sa[7] = (unsigned char)((ip >> 24) & 0xff);
}

#define APY_NET_NAME_MAX 256
static int apy_net_name(@PTR@ p, int64_t n, char *out)
{
    int64_t i;
    if (n < 0 || n >= APY_NET_NAME_MAX) return 0;
    for (i = 0; i < n; i++) {
        char c = ((const char *)p)[i];
        if (c == 0) return 0;
        out[i] = c;
    }
    out[n] = 0;
    return 1;
}

static void apy_net_start(void)
{
#ifdef _WIN32
    /* ONCE. Winsock refuses every call before `WSAStartup`, and calling it
       twice is harmless but pointless. 0x0202 is version 2.2. */
    static int done = 0;
    if (!done) { char data[512]; WSAStartup(0x0202, data); done = 1; }
#endif
}

@STATIC@int64_t host_net_connect(@PTR@ host, int64_t n, int64_t port)
{
    char name[APY_NET_NAME_MAX];
    unsigned char sa[16];
    unsigned int ip;
    apy_socket_t s;
    if (port < 0 || port > 65535) return -9;
    if (!apy_net_name(host, n, name)) return -9;
    apy_net_start();
    ip = (unsigned int)inet_addr(name);
    /* NUMERIC ADDRESSES ONLY. `gethostbyname` would resolve a name, and
       resolution is a larger contract than this group has -- it needs a
       resolver, a timeout and an error vocabulary of its own. A caller with
       a name resolves it before getting here; a caller without one passes
       `127.0.0.1` and this works. */
    if (ip == 0xffffffffu) return -9;
    s = socket(APY_AF_INET, APY_SOCK_STREAM, 0);
    if (s == APY_NET_INVALID) return -1;
    apy_net_addr(sa, ip, port);
    if (connect(s, sa, 16) != 0) { APY_NET_CLOSE(s); return -1; }
    return (int64_t)s;
}

@STATIC@int64_t host_net_listen(int64_t port, int64_t backlog)
{
    unsigned char sa[16];
    apy_socket_t s;
    int on = 1;
    if (port < 0 || port > 65535) return -9;
    apy_net_start();
    s = socket(APY_AF_INET, APY_SOCK_STREAM, 0);
    if (s == APY_NET_INVALID) return -1;
    /* REUSEADDR, or a listener that has just closed leaves the port in
       TIME_WAIT and the next run of the same program is refused. The
       interpreter's implementation sets it too, so the two agree about
       whether a program can be run twice. */
    setsockopt(s, APY_NET_SOL_SOCKET, APY_NET_REUSEADDR,
               (const char *)&on, (unsigned int)sizeof on);
    apy_net_addr(sa, inet_addr("127.0.0.1"), port);
    if (bind(s, sa, 16) != 0) { APY_NET_CLOSE(s); return -1; }
    if (listen(s, (int)(backlog > 0 ? backlog : 1)) != 0) {
        APY_NET_CLOSE(s); return -1;
    }
    return (int64_t)s;
}

@STATIC@int64_t host_net_accept(int64_t fd)
{
    apy_socket_t c = accept((apy_socket_t)fd, 0, 0);
    if (c == APY_NET_INVALID) return -1;
    return (int64_t)c;
}

@STATIC@int64_t host_net_read(int64_t fd, @PTR@ buf, int64_t n)
{
    long got;
    if (n < 0) return -9;
    got = (long)recv((apy_socket_t)fd, (char *)buf, (unsigned long)n, 0);
    /* ZERO IS END OF STREAM and is not an error: the peer closed its end.
       Every error in this table is negative, so the two never collide. */
    if (got < 0) return -1;
    return (int64_t)got;
}

@STATIC@int64_t host_net_write(int64_t fd, @PTR@ buf, int64_t n)
{
    long put;
    if (n < 0) return -9;
    put = (long)send((apy_socket_t)fd, (const char *)buf, (unsigned long)n, 0);
    if (put < 0) return -8;                    /* EPIPE: the peer is gone  */
    return (int64_t)put;
}

@STATIC@int64_t host_net_close(int64_t fd)
{
    return APY_NET_CLOSE((apy_socket_t)fd) == 0 ? 0 : -1;
}

@STATIC@int64_t host_net_ready(int64_t fd, int64_t want, int64_t timeout)
{
    /* `poll`, and NOT `select`. Two reasons, and the second is the one that
       decided it: `select`'s `fd_set` is a bitmap on POSIX and a counted
       array on Windows, so it needs two implementations -- and glibc's
       `<stdlib.h>` already brings `select` into scope through
       `<sys/select.h>`, so declaring it here is a conflicting prototype and
       does not compile. `poll` is declared by `<poll.h>`, which nothing here
       includes, and its `struct pollfd` is three fields with no padding
       question on either platform.

       WINDOWS SPELLS IT `WSAPoll` and its descriptor is a pointer-sized
       SOCKET rather than an int, which is the only difference -- so the
       struct is written out per platform and the call is the same shape. */
    int ready;
#ifdef _WIN32
    struct { apy_socket_t fd; short events; short revents; } pfd;
    pfd.fd = (apy_socket_t)fd;
    /* POLLRDNORM / POLLWRNORM: Winsock's names for the two conditions,
       and the only two `WSAPoll` accepts on input. */
    pfd.events = (short)(want == 1 ? 0x0100 : 0x0010);
    pfd.revents = 0;
    if (want != 1 && want != 2) return -9;
    ready = WSAPoll(&pfd, 1, timeout < 0 ? -1 : (int)(timeout / 1000000));
#else
    struct { int fd; short events; short revents; } pfd;
    if (want != 1 && want != 2) return -9;
    pfd.fd = (int)fd;
    /* POLLIN is 0x001 and POLLOUT is 0x004 on every POSIX platform this
       targets -- they are in the standard, not the implementation. */
    pfd.events = (short)(want == 1 ? 0x001 : 0x004);
    pfd.revents = 0;
    /* MILLISECONDS, which is what `poll` takes and why the contract's
       nanoseconds are divided here rather than at every caller. A negative
       timeout means wait indefinitely, in the contract and in `poll`. */
    ready = poll(&pfd, 1, timeout < 0 ? -1 : (int)(timeout / 1000000));
#endif
    if (ready < 0) return -1;
    return ready > 0 ? 1 : 0;
}


@STATIC@int64_t host_net_port(int64_t fd)
{
    unsigned char sa[16];
    unsigned int len = 16;
    int i;
    for (i = 0; i < 16; i++) sa[i] = 0;
    if (getsockname((apy_socket_t)fd, sa, &len) != 0) return -1;
    /* NETWORK BYTE ORDER, read back the way `apy_net_addr` wrote it. */
    return (int64_t)(((unsigned int)sa[2] << 8) | (unsigned int)sa[3]);
}
"""


C_SOURCE["proc"] = r"""/* --- host services: proc ------------------------------------------------ */

/* NO HEADER, as everywhere else in this file: `<unistd.h>` and
   `<sys/wait.h>` declare a great deal besides the seven calls wanted here,
   and every name they declare is one a `ctypes` program may declare for
   itself. The prototypes are written out.

   `fork` + `execvp` + `waitpid` AND NOT `system` OR `popen`. Both of those
   take a SHELL COMMAND, which means quoting an argv list into one string --
   and that is where command injection comes from. The caller already has
   the arguments separated; throwing that away and guessing at it again
   would make every program built on this less safe than the same program
   written in C. A shell, when one is wanted, is `["/bin/sh", "-c", cmd]`
   passed in as an ordinary argv by whoever wanted it. */
#ifdef _WIN32
/* `_spawnvp` RUNS AND WAITS in one call, which is the shape this contract
   wants -- but it cannot CAPTURE, and there is no way to capture on Windows
   without `CreateProcess` and its two structs (`STARTUPINFOA` is eighteen
   fields whose layout a wrong prototype gets silently wrong, which is
   exactly the hazard this file's `file` group warns about). So the Windows
   half runs the child with INHERITED stdio and reports empty capture
   buffers, and `bundled/subprocess.py` turns that into the refusal a
   program can act on. Honest and incomplete beats a struct written from
   memory. */
__declspec(dllimport) intptr_t _spawnvp(int, const char *, const char *const *);
#define APY_SPAWN_WAIT 0
#else
/* THE PIPES ARE READ THROUGH `FILE *`, not through `read`, and the reason
   is a collision this file cannot win. Every name declared here shares one
   namespace with the C the BACKEND emits for the program's own functions,
   and a Python program with `def read(...)` compiles to `uintptr_t
   read(uintptr_t)` -- which conflicts with any prototype for libc's, and
   with the implicit declaration if there is none. Corpus program
   `functions_and_globals` has exactly that function, and it failed to
   build the moment this group appeared.

   `fdopen`/`fread`/`fclose` are `<stdio.h>`'s, which is already included,
   so NOTHING IS DECLARED for them and there is nothing to collide. The
   same argument applies to every name below: each is declared because
   nothing else in the translation unit has it, and each was checked by
   compiling, which is the only way to know. */
int pipe(int *);
int fork(void);
int execvp(const char *, char *const *);
int dup2(int, int);
int waitpid(int, int *, int);
void _exit(int);
/* DECLARED HERE TOO, though `net` declares it as well: a group must not
   depend on another group being emitted, because a backend may take one
   and not the other. Two identical prototypes for one symbol are legal C;
   two DIFFERENT ones are not, which is why both say `int close(int)`. */
int close(int);
#endif

#define APY_PROC_ARGS_MAX 256

/* THE PACKED ARGV, SPLIT IN PLACE. The buffer arrives NUL-separated with a
   count, which is already the shape `execvp` wants -- a pointer per
   argument into the same bytes, and a NULL at the end. Copied into a local
   because `execvp` takes `char *const *` and the caller's buffer is
   `const`; the strings themselves are not copied. */
static int apy_proc_argv(@PTR@ packed, int64_t n, int64_t count,
                         char **out)
{
    char *base = (char *)packed;
    int64_t i, seen = 0;
    if (count <= 0 || count >= APY_PROC_ARGS_MAX || n < 0) return 0;
    out[seen++] = base;
    for (i = 0; i < n && seen <= count; i++)
        if (base[i] == 0 && i + 1 < n && seen < count)
            out[seen++] = base + i + 1;
    if (seen != count) return 0;
    out[count] = 0;
    return 1;
}

@STATIC@int64_t host_proc_run(@PTR@ packed, int64_t count, int64_t n,
                              @PTR@ out, int64_t out_cap,
                              @PTR@ err, int64_t err_cap,
                              @PTR@ status)
{
    char *argv[APY_PROC_ARGS_MAX];
    if (!apy_proc_argv(packed, n, count, argv)) return -9;
#ifdef _WIN32
    {
        intptr_t code = _spawnvp(APY_SPAWN_WAIT, argv[0],
                                 (const char *const *)argv);
        if (code < 0) return -2;
        *(int64_t *)status = (int64_t)code;
        (void)out; (void)out_cap; (void)err; (void)err_cap;
        /* NOTHING CAPTURED, and the module above knows what that means. */
        return 0;
    }
#else
    {
        int outfd[2], errfd[2], state = 0, pid, i;
        int64_t got_out = 0, got_err = 0;
        if (pipe(outfd) != 0) return -1;
        if (pipe(errfd) != 0) { close(outfd[0]); close(outfd[1]); return -1; }
        pid = fork();
        if (pid < 0) {
            close(outfd[0]); close(outfd[1]);
            close(errfd[0]); close(errfd[1]);
            return -1;
        }
        if (pid == 0) {
            /* THE CHILD. Its writing ends become its stdout and stderr, and
               every other descriptor this function opened is closed --
               leaving one open would hold the pipe alive after the child
               exits and the parent's read below would never see end of
               file. `_exit` and not `exit`: the child must not run the
               parent's atexit handlers or flush its buffers twice. */
            dup2(outfd[1], 1);
            dup2(errfd[1], 2);
            close(outfd[0]); close(outfd[1]);
            close(errfd[0]); close(errfd[1]);
            execvp(argv[0], argv);
            _exit(127);                  /* the shell's "not found" status */
        }
        close(outfd[1]);
        close(errfd[1]);
        {
        FILE *fout = fdopen(outfd[0], "rb");
        FILE *ferr = fdopen(errfd[0], "rb");
        if (!fout || !ferr) {
            if (fout) fclose(fout); else close(outfd[0]);
            if (ferr) fclose(ferr); else close(errfd[0]);
            waitpid(pid, &state, 0);
            return -1;
        }
        /* READ BOTH BEFORE WAITING. A child that fills one pipe blocks
           until someone drains it, so waiting first and reading second
           deadlocks on any output larger than a pipe buffer. Alternating
           between the two is what a `select` would do properly; reading
           stdout to the end and then stderr is enough here because the
           capture buffers are sized by the caller and the common case is a
           program that writes to one of them. */
        if (out_cap > 0)
            got_out = (int64_t)fread((char *)out, 1, (size_t)out_cap, fout);
        if (err_cap > 0)
            got_err = (int64_t)fread((char *)err, 1, (size_t)err_cap, ferr);
        fclose(fout);
        fclose(ferr);
        }
        if (waitpid(pid, &state, 0) < 0) return -1;
        /* WIFEXITED / WEXITSTATUS, written out. The low seven bits are the
           signal that killed it and the next eight are the exit status;
           CPython reports a signal as a NEGATIVE return code, so that is
           what goes back. */
        if ((state & 0x7f) == 0)
            *(int64_t *)status = (int64_t)((state >> 8) & 0xff);
        else
            *(int64_t *)status = -(int64_t)(state & 0x7f);
        (void)i;
        return got_out;
    }
#endif
}
"""


C_SOURCE["thread"] = r"""/* --- host services: thread ---------------------------------------------- */

/* NO HEADER, for the reason every other group here gives: `<pthread.h>`
   declares a great deal besides the dozen calls wanted, and every name it
   declares is one a `ctypes` program may declare for itself -- two
   prototypes for one symbol do not compile.

   THE OPAQUE TYPES ARE BYTE ARRAYS OF THE RIGHT SIZE, allocated with
   `malloc` and never inspected. `pthread_mutex_t` is 40 bytes on x86-64
   Linux and 64 on some other platforms; a `pthread_cond_t` is 48. Asking
   for 128 costs nothing measurable and cannot be too small on any platform
   this targets -- and a handle a caller may only hand back is exactly the
   kind of thing this file's header says should be opaque.

   THE THREAD FUNCTION'S SIGNATURE IS THE CONTRACT'S, not pthreads': it
   takes a `ptr` and answers an `i64`. `pthread_create` wants `void *(*)(void
   *)`, so the start routine below is a trampoline that calls the real one
   and keeps its answer for `join`. */
#ifdef _WIN32
/* WINDOWS IS NOT IMPLEMENTED HERE, and says so rather than pretending: the
   Win32 thread API needs `CRITICAL_SECTION` and `CONDITION_VARIABLE`, which
   are STRUCTS whose layout a wrong prototype gets silently wrong -- the
   hazard this file's `file` group warns about at length, and the reason its
   `proc` group stops where it does. `thrd_create` answers `thrd_error`
   there, which is an answer C defines and a program can act on. */
@STATIC@int64_t host_thread_start(@PTR@ fn, @PTR@ arg)
{ (void)fn; (void)arg; return -1; }
@STATIC@int64_t host_thread_join(int64_t h, @PTR@ out)
{ (void)h; (void)out; return -1; }
@STATIC@int64_t host_thread_detach(int64_t h) { (void)h; return -1; }
@STATIC@int64_t host_thread_self(void) { return 0; }
@STATIC@int64_t host_thread_yield(void) { return 0; }
@STATIC@int64_t host_thread_exit(int64_t code) { (void)code; return -1; }
@STATIC@int64_t host_mutex_new(int64_t kind) { (void)kind; return -1; }
@STATIC@int64_t host_mutex_lock(int64_t m) { (void)m; return -1; }
@STATIC@int64_t host_mutex_trylock(int64_t m) { (void)m; return -1; }
@STATIC@int64_t host_mutex_timedlock(int64_t m, int64_t ns)
{ (void)m; (void)ns; return -1; }
@STATIC@int64_t host_mutex_unlock(int64_t m) { (void)m; return -1; }
@STATIC@int64_t host_mutex_free(int64_t m) { (void)m; return -1; }
@STATIC@int64_t host_cond_new(void) { return -1; }
@STATIC@int64_t host_cond_wait(int64_t c, int64_t m, int64_t ns)
{ (void)c; (void)m; (void)ns; return -1; }
@STATIC@int64_t host_cond_signal(int64_t c) { (void)c; return -1; }
@STATIC@int64_t host_cond_broadcast(int64_t c) { (void)c; return -1; }
@STATIC@int64_t host_cond_free(int64_t c) { (void)c; return -1; }
@STATIC@int64_t host_tss_new(@PTR@ dtor) { (void)dtor; return -1; }
@STATIC@@PTR@ host_tss_get(int64_t k) { (void)k; return 0; }
@STATIC@int64_t host_tss_set(int64_t k, @PTR@ v) { (void)k; (void)v; return -1; }
@STATIC@int64_t host_tss_free(int64_t k) { (void)k; return -1; }
#else
#if defined(Py_PYTHON_H)
/* THE CONSUMER ALREADY HAS THEM, which makes declaring them again the very
   conflict the rule above exists to avoid -- from the other side, exactly as
   `clock_gettime` below. `Python.h` includes `pythread.h`, which includes
   `<pthread.h>`, so every name this group calls is already in scope with the
   platform's own signatures. A second prototype saying `void *` where the
   real one says `pthread_mutex_t *` is a hard error and not a warning, which
   is why no extension module the cpyext backend produced would compile at
   all: `conflicting types for 'pthread_create'`, then the same for
   `pthread_cond_destroy`, and the build stopped there.

   THE TEST IS ANSWERED BY THE TIME THE PREPROCESSOR REACHES IT. The cpyext
   backend writes `#include <Python.h>` FIRST and says why -- it sets
   feature-test macros that later headers read -- so `Py_PYTHON_H` is defined
   before this line whenever this C is part of an extension module, and
   defined nowhere else.

   NOTHING BELOW CHANGES. The calls pass `void *` handles into these, and a
   `void *` converts to any object pointer on its own, so the bodies compile
   against the real prototypes unaltered. `apy_thread_t` becomes the
   platform's `pthread_t` rather than a guess at it, which is the same
   improvement the rule gives up elsewhere for the sake of not needing the
   header at all. */
#include <pthread.h>
#include <sched.h>
typedef pthread_t apy_thread_t;
#else
typedef unsigned long apy_thread_t;
int pthread_create(apy_thread_t *, const void *, void *(*)(void *), void *);
int pthread_join(apy_thread_t, void **);
int pthread_detach(apy_thread_t);
apy_thread_t pthread_self(void);
int sched_yield(void);
void pthread_exit(void *);
int pthread_mutex_init(void *, const void *);
int pthread_mutex_lock(void *);
int pthread_mutex_trylock(void *);
int pthread_mutex_unlock(void *);
int pthread_mutex_destroy(void *);
int pthread_mutexattr_init(void *);
int pthread_mutexattr_settype(void *, int);
int pthread_cond_init(void *, const void *);
int pthread_cond_wait(void *, void *);
int pthread_cond_timedwait(void *, void *, const void *);
int pthread_cond_signal(void *);
int pthread_cond_broadcast(void *);
int pthread_cond_destroy(void *);
int pthread_key_create(unsigned int *, void (*)(void *));
int pthread_key_delete(unsigned int);
void *pthread_getspecific(unsigned int);
int pthread_setspecific(unsigned int, const void *);
#endif
/* `clock_gettime` IS NOT DECLARED HERE, and it is the one exception to this
   file's rule. Every consumer of this C already includes `<time.h>` -- the
   `time` group's `time()` and `CLOCKS_PER_SEC` need it -- so the name and
   its `struct timespec` are already in scope, and declaring them again is
   the conflicting-prototype error the rule exists to avoid, from the other
   side. The struct is also one this file must not guess at: it is two words
   on every platform this targets and is exactly the kind of layout the
   `file` group's comment says not to write from memory. */

/* ROOM FOR THE LARGEST OF THESE OBJECTS ON ANY PLATFORM THIS TARGETS. See
   the comment above: the size is deliberately generous and the contents are
   never read here. */
#define APY_THREAD_OBJ 128
#define APY_MUTEX_RECURSIVE 1

/* THE TRAMPOLINE AND WHAT IT CARRIES. A thread's answer is an `int64_t` and
   `pthread_join` hands back a `void *`, which is the same width on every
   platform with threads -- but going through one loses the sign, so the
   answer is kept in the block instead and the pointer is only a token. */
struct apy_thread_slot {
    apy_thread_t id;
    int64_t (*fn)(void *);
    void *arg;
    int64_t result;
    int done;
};

static void *apy_thread_run(void *p)
{
    struct apy_thread_slot *s = (struct apy_thread_slot *)p;
    s->result = s->fn(s->arg);
    s->done = 1;
    return p;
}

@STATIC@int64_t host_thread_start(@PTR@ fn, @PTR@ arg)
{
    struct apy_thread_slot *s;
    if (fn == 0) return -9;
    s = (struct apy_thread_slot *)malloc(sizeof *s);
    if (!s) return -1;
    s->fn = (int64_t (*)(void *))fn;
    s->arg = (void *)arg;
    s->result = 0;
    s->done = 0;
    if (pthread_create(&s->id, 0, apy_thread_run, s) != 0) {
        free(s);
        return -1;
    }
    /* THE HANDLE IS THE BLOCK, not the `pthread_t`: the answer has to live
       somewhere until somebody joins, and the block is where it is. */
    return (int64_t)(intptr_t)s;
}

@STATIC@int64_t host_thread_join(int64_t h, @PTR@ out)
{
    struct apy_thread_slot *s = (struct apy_thread_slot *)(intptr_t)h;
    void *ignored;
    if (!s) return -9;
    if (pthread_join(s->id, &ignored) != 0) return -1;
    if (out) *(int64_t *)out = s->result;
    free(s);
    return 0;
}

@STATIC@int64_t host_thread_detach(int64_t h)
{
    struct apy_thread_slot *s = (struct apy_thread_slot *)(intptr_t)h;
    if (!s) return -9;
    if (pthread_detach(s->id) != 0) return -1;
    /* THE BLOCK LEAKS, ON PURPOSE. Nothing will join, so nothing can know
       when the thread is finished with it; one block per detached thread is
       the price of not freeing memory another thread is still running in. */
    return 0;
}

@STATIC@int64_t host_thread_self(void)
{
    /* NOT THE BLOCK -- a thread does not know its own -- but a number that
       is this thread's and nobody else's, which is all `thrd_current` is
       for: comparing. */
    return (int64_t)pthread_self();
}

@STATIC@int64_t host_thread_yield(void) { return (int64_t)sched_yield(); }

@STATIC@int64_t host_thread_exit(int64_t code)
{
    pthread_exit((void *)(intptr_t)code);
    return 0;
}

@STATIC@int64_t host_mutex_new(int64_t kind)
{
    void *m = malloc(APY_THREAD_OBJ);
    char attr[APY_THREAD_OBJ];
    if (!m) return -1;
    if (kind == APY_MUTEX_RECURSIVE) {
        pthread_mutexattr_init(attr);
        /* PTHREAD_MUTEX_RECURSIVE is 1 on Linux and 2 on macOS, and there
           is no portable name without the header. The common case is the
           plain mutex; a recursive one falls back to plain where the number
           is wrong, which C's `mtx_recursive` does not promise to detect. */
        pthread_mutexattr_settype(attr, 1);
        if (pthread_mutex_init(m, attr) != 0) { free(m); return -1; }
        return (int64_t)(intptr_t)m;
    }
    if (pthread_mutex_init(m, 0) != 0) { free(m); return -1; }
    return (int64_t)(intptr_t)m;
}

@STATIC@int64_t host_mutex_lock(int64_t h)
{
    if (!h) return -9;
    return pthread_mutex_lock((void *)(intptr_t)h) == 0 ? 0 : -1;
}

@STATIC@int64_t host_mutex_trylock(int64_t h)
{
    if (!h) return -9;
    /* ONE FOR "SOMEBODY ELSE HAS IT", which is not an error: every error in
       this table is negative, so the two are distinguishable. */
    return pthread_mutex_trylock((void *)(intptr_t)h) == 0 ? 0 : 1;
}

@STATIC@int64_t host_mutex_timedlock(int64_t h, int64_t nanos)
{
    /* SPINNING WITH A YIELD, because `pthread_mutex_timedlock` takes a
       `struct timespec` at an ABSOLUTE time -- a struct this file would
       have to declare and a clock it would have to read. The wait is
       correct and is not efficient; a program that waits on a mutex for
       long enough to notice wants a condition variable. */
    int64_t waited = 0;
    if (!h) return -9;
    for (;;) {
        if (pthread_mutex_trylock((void *)(intptr_t)h) == 0) return 0;
        if (nanos >= 0 && waited >= nanos) return 1;
        sched_yield();
        waited += 1000;
    }
}

@STATIC@int64_t host_mutex_unlock(int64_t h)
{
    if (!h) return -9;
    return pthread_mutex_unlock((void *)(intptr_t)h) == 0 ? 0 : -1;
}

@STATIC@int64_t host_mutex_free(int64_t h)
{
    if (!h) return -9;
    pthread_mutex_destroy((void *)(intptr_t)h);
    free((void *)(intptr_t)h);
    return 0;
}

@STATIC@int64_t host_cond_new(void)
{
    void *c = malloc(APY_THREAD_OBJ);
    if (!c) return -1;
    if (pthread_cond_init(c, 0) != 0) { free(c); return -1; }
    return (int64_t)(intptr_t)c;
}

@STATIC@int64_t host_cond_wait(int64_t c, int64_t m, int64_t nanos)
{
    if (!c || !m) return -9;
    if (nanos < 0)
        return pthread_cond_wait((void *)(intptr_t)c,
                                 (void *)(intptr_t)m) == 0 ? 0 : -1;
    {
        /* THE TIMED FORM NEEDS AN ABSOLUTE DEADLINE, which is `<time.h>`'s
           own `struct timespec` -- see the note beside the prototypes above
           for why this one type is not written out. CLOCK_REALTIME is 0. */
        struct timespec until;
        clock_gettime(0, &until);
        until.tv_sec += (long)(nanos / 1000000000);
        until.tv_nsec += (long)(nanos % 1000000000);
        if (until.tv_nsec >= 1000000000) {
            until.tv_nsec -= 1000000000;
            until.tv_sec += 1;
        }
        {
            int r = pthread_cond_timedwait((void *)(intptr_t)c,
                                           (void *)(intptr_t)m, &until);
            if (r == 0) return 0;
            /* ETIMEDOUT is 110 on Linux and 60 on macOS; anything that is
               not success and not an argument error is a timeout here, and
               the caller's own deadline tells it which. */
            return 1;
        }
    }
}

@STATIC@int64_t host_cond_signal(int64_t c)
{
    if (!c) return -9;
    return pthread_cond_signal((void *)(intptr_t)c) == 0 ? 0 : -1;
}

@STATIC@int64_t host_cond_broadcast(int64_t c)
{
    if (!c) return -9;
    return pthread_cond_broadcast((void *)(intptr_t)c) == 0 ? 0 : -1;
}

@STATIC@int64_t host_cond_free(int64_t c)
{
    if (!c) return -9;
    pthread_cond_destroy((void *)(intptr_t)c);
    free((void *)(intptr_t)c);
    return 0;
}

/* A KEY IS A `pthread_key_t`, WHICH IS AN UNSIGNED INT, and the handle is
   that number plus one: zero is a perfectly good key and this table's
   handles are negative only for errors, so the offset keeps a valid key
   from looking like "nothing". */
@STATIC@int64_t host_tss_new(@PTR@ dtor)
{
    unsigned int key = 0;
    if (pthread_key_create(&key, (void (*)(void *))dtor) != 0) return -1;
    return (int64_t)key + 1;
}

@STATIC@@PTR@ host_tss_get(int64_t key)
{
    if (key <= 0) return 0;
    return (@PTR@)pthread_getspecific((unsigned int)(key - 1));
}

@STATIC@int64_t host_tss_set(int64_t key, @PTR@ value)
{
    if (key <= 0) return -9;
    return pthread_setspecific((unsigned int)(key - 1),
                               (const void *)value) == 0 ? 0 : -1;
}

@STATIC@int64_t host_tss_free(int64_t key)
{
    if (key <= 0) return -9;
    return pthread_key_delete((unsigned int)(key - 1)) == 0 ? 0 : -1;
}
#endif
"""

C_SOURCE["dynlib"] = r"""/* --- host services: dynlib ---------------------------------------------- */

/* NO HEADER FOR EITHER PLATFORM, and for the reason the `file` group gives at
   length: `<dlfcn.h>` and `<windows.h>` both declare a great deal besides the
   three functions wanted here, and every name they declare is one a `ctypes`
   program may declare for itself -- two prototypes for one symbol do not
   compile. `<windows.h>` in particular would put `CreateDirectoryA` in scope,
   which `bundled/pathlib.py` reaches through `ctypes` precisely because no
   header this runtime includes declares it.

   So the prototypes are written out. On Windows they are spelled with the
   types the ABI actually uses rather than the `WINAPI` typedefs: `HMODULE` is
   a pointer, `FARPROC` is a function pointer, and `LPCSTR` is `const char *`.
   Declaring them as plain pointers is ABI-identical and needs no header. */
#ifdef _WIN32
__declspec(dllimport) void *__stdcall LoadLibraryA(const char *);
__declspec(dllimport) int __stdcall FreeLibrary(void *);
/* GetProcAddress answers a function pointer. It is declared returning
   `void *` here and converted through `uintptr_t` at the one call site --
   which is the conversion C says is implementation-defined and every
   platform with a `dlsym` defines, because the whole point of the call is to
   get an address a caller will call. */
__declspec(dllimport) void *__stdcall GetProcAddress(void *, const char *);
#define APY_DL_OPEN(p)     LoadLibraryA(p)
#define APY_DL_SYM(h, s)   GetProcAddress((h), (s))
#define APY_DL_CLOSE(h)    (FreeLibrary(h) ? 0 : -1)
#else
void *dlopen(const char *, int);
void *dlsym(void *, const char *);
int dlclose(void *);
/* RTLD_NOW | RTLD_LOCAL, written as the numbers they are on Linux and macOS.
   Resolving eagerly means a missing symbol is a failed OPEN rather than a
   crash at the first call, which is the difference between an error a program
   can report and one it cannot. */
#define APY_DL_OPEN(p)     dlopen((p), 0x00002 | 0x00004)
#define APY_DL_SYM(h, s)   dlsym((h), (s))
#define APY_DL_CLOSE(h)    (dlclose(h) ? -1 : 0)
#endif

/* A NAME ARRIVES AS A POINTER AND A LENGTH, as every path in this file does,
   and every platform call below wants it NUL-terminated. The copy is also
   where an embedded NUL is caught rather than silently truncated at. */
#define APY_DL_NAME_MAX 4096
static int apy_dl_name(@PTR@ p, int64_t n, char *out)
{
    int64_t i;
    if (n < 0 || n >= APY_DL_NAME_MAX) return 0;
    for (i = 0; i < n; i++) {
        char c = ((const char *)p)[i];
        if (c == 0) return 0;
        out[i] = c;
    }
    out[n] = 0;
    return 1;
}

/* THE NAME IS PASSED THROUGH UNDECORATED. `CDLL("m")` becomes `-lm` on the
   link-time path because a LINKER wants that spelling; a loader does not, and
   wants `libm.so.6` or `user32.dll` exactly as the program wrote it. The two
   paths therefore disagree about what a library is called, which is a
   property of the two mechanisms rather than a wart -- and the reason the
   decoration is stripped in `cffi.link_flag` and not here. */
@STATIC@int64_t host_dl_open(@PTR@ name, int64_t n)
{
    char buf[APY_DL_NAME_MAX];
    void *h;
    if (!apy_dl_name(name, n, buf)) return -9;
    h = APY_DL_OPEN(buf);
    /* NO `dlerror` AND NO `GetLastError`. Both are thread-local global state
       read by a SECOND call, which is the shape this layer refuses -- see the
       header. A library that will not load is `HOST_ENOENT`, which is what it
       almost always is. */
    if (!h) return -2;
    return (int64_t)(intptr_t)h;
}

@STATIC@int64_t host_dl_sym(int64_t handle, @PTR@ name, int64_t n)
{
    char buf[APY_DL_NAME_MAX];
    void *addr;
    if (handle <= 0) return -9;
    if (!apy_dl_name(name, n, buf)) return -9;
    addr = APY_DL_SYM((void *)(intptr_t)handle, buf);
    /* ZERO IS "NO SUCH SYMBOL" AND IS NOT AN ERROR. A symbol's address is
       never zero, and every error in this table is negative, so the two are
       distinguishable without a second call. */
    if (!addr) return 0;
    return (int64_t)(intptr_t)addr;
}

@STATIC@int64_t host_dl_close(int64_t handle)
{
    if (handle <= 0) return -9;
    return APY_DL_CLOSE((void *)(intptr_t)handle) ? -1 : 0;
}
"""


def c_source(groups, *, static: bool = False, ptr: str = "void *") -> str:
    """The C for the groups a backend declared, substitutions made.

    UNKNOWN GROUP NAMES ARE IGNORED rather than refused, because a backend may
    declare a capability it implements ITSELF -- the JVM backend will answer
    `file` from `java.nio` and wants no C at all. This function answers "what
    C do you need from me", and for such a backend the answer is none.

    IN `GROUPS` ORDER, NOT THE CALLER'S. A backend declares its set as a
    `frozenset`, whose iteration order is a hash artefact -- so the emitted
    C came out in a different order on different runs, and a group whose
    prototypes another one relies on landed after it about half the time.
    That is exactly what happened: `proc` calls `close`, which `net`
    declares, and the build failed with an implicit declaration whenever
    the set happened to yield `proc` first. Ordering here is one line; each
    group ALSO declaring what it calls is the other half of the fix, and
    both are needed -- this makes the output reproducible and that makes
    each group independent.
    """
    parts = [C_SOURCE[g] for g in GROUPS if g in groups and g in C_SOURCE]
    text = "\n".join(parts)
    return (text.replace("@PTR@", ptr)
                .replace("@STATIC@", "static " if static else ""))
