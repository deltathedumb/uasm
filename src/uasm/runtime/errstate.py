# The error path's own state, in the machine subset.
#
# STAGE 5 OF docs/INERT-RUNTIME.md, and the first piece of SHARED STATE to
# move since the singleton cells. The pattern is the same one 5b established
# and the reason is the same: the subset cannot read a C static, so the
# storage has to come across before anything that reads it can.
#
# WHY THIS AND NOT THE WHOLE ERROR FLAG. `apy_err_type`, `apy_err_msg` and
# `apy_err_value` are the rest of it, and they are a harder problem: the type
# is a `const char *` compared with `strcmp` inside functions that will stay
# in C for a long time, and the message is a 256-byte buffer written with
# `snprintf`, which the subset has no way to do. What moves here is the part
# with no formatting and no C strings in it -- two integers and a handle --
# which is enough to port `apy_at` and `apy_error_handling` and is a rehearsal
# for the rest.
#
# ── zero means "none", and that took a decision ────────────────────────────
#
# The C wrote `static int64_t apy_pos_here = -1, apy_err_pos = -1;` -- MINUS
# ONE for "no position recorded", because 0 is a real position. `reserve`
# gives storage that is zeroed and there is nowhere to run an initialiser, so
# the two cannot start at -1.
#
# SO WHAT IS STORED IS THE POSITION PLUS ONE. Zeroed storage reads back as
# -1, which is exactly what the C's initialiser meant, and every caller sees
# the same values it always did. The bias is confined to this file: three
# functions add one and three subtract it, and nothing outside knows.
#
# The alternative -- an "initialised" flag beside each -- costs another word
# and another branch to express what a bias expresses for free, and it has the
# failure mode a bias does not: the flag and the value can disagree.


def apy_pos_state() -> ptr:
    """Two int64s: the current position, then the latched one.

    ONE RESERVATION FOR THE PAIR because they are written together and read
    together, and because `reserve` is per-NAME -- two would be two globals
    where the C had one declaration of two variables.
    """
    return reserve("apy_pos_state_ir", 16)


def apy_handling_state() -> ptr:
    """The exception currently being handled, or null.

    SEPARATE FROM THE POSITIONS, because a handle is not an integer and
    putting them in one reservation would mean this file deciding a layout
    that nothing else can check. Two names cost nothing.
    """
    return reserve("apy_handling_ir", 8)


# ── the current position ───────────────────────────────────────────────────


def apy_at(which: i64) -> None:
    """Record where execution is, for a traceback to report later.

    CALLED PER STATEMENT in a program compiled with positions, so it is the
    hottest function in this file by a wide margin -- one store, and the bias
    is an add the backend folds into the constant.

    `-> None` AND NOT `-> void`. The C returns `void` and `signatures()` types
    it that way, but `void` is not a name the machine subset has: its
    vocabulary is the widths, `ptr`, and Python's own `None`. Writing the C
    spelling gets `E0052: call to unknown function 'store'`, which names the
    line after the annotation rather than the annotation.
    """
    store(i64, which + 1, apy_pos_state())


def apy_pos_now() -> i64:
    """Where execution is. -1 if nothing has recorded a position."""
    return load(i64, apy_pos_state()) - 1


def apy_pos_latch() -> i64:
    """Remember the current position as where the failure happened.

    ONE OPERATION, NOT A GETTER AND A SETTER. Both `apy_fail` and `apy_fail2`
    did exactly this, and splitting it would let a caller do half -- reading
    the position without latching it, or latching a position it did not read.
    What has to happen together is one function.
    """
    now: i64 = load(i64, apy_pos_state())
    store(i64, now, offset(apy_pos_state(), 8))
    return now - 1


def apy_pos_latched() -> i64:
    """Where the failure was raised. -1 if none has been."""
    return load(i64, offset(apy_pos_state(), 8)) - 1


# ── the handler being run ──────────────────────────────────────────────────


def apy_handling_now() -> ptr:
    """The exception whose `except` block is running, or null."""
    return ptr(load(u64, apy_handling_state()))


def apy_error_handling(exc: ptr) -> ptr:
    """Enter or leave a handler. Answers what was being handled before.

    THE SWAP IS THE POINT. A handler saves what this returns and passes it
    back on the way out, so nesting restores rather than clears -- which is
    what makes `e.__context__` chain correctly through nested `except`
    blocks instead of losing the outer one.

    A NON-EXCEPTION CLEARS IT, which is how leaving a handler is spelled: the
    frontend passes something that is not an exception cell rather than
    needing a second entry point.
    """
    was: ptr = ptr(load(u64, apy_handling_state()))
    if i64(load(i32, offset(exc, 0))) == apy_exc_kind():
        store(u64, u64(exc), apy_handling_state())
    else:
        store(u64, 0, apy_handling_state())
    if not was:
        return apy_none()
    return was


def apy_exc_kind() -> i64:
    return 8


# ── the pending error ──────────────────────────────────────────────────────
#
# STAGE 5, AND THE LEVER THE SURVEY POINTED AT. `apy_error_occurred` is one
# line of C and ten other functions are waiting on it: they are closed over
# everything else IR defines, and this single flag is what keeps them in the C.
#
# IT COULD NOT MOVE ALONE. A flag is only as portable as the storage behind it,
# and `apy_err_type` was a C static -- so this moves the STATE, and the flag
# follows. The same thing happened to the None cell in `runtime/singletons.py`
# and to the source positions above.
#
# THREE PIECES, AND THEY ARE NOT ONE RESERVATION. The type name and the
# exception object are two words written together; the message is 256 bytes
# written by `snprintf`. Putting the text in the same block would make this
# file decide where a buffer starts relative to two pointers, which is a layout
# nothing outside could check.
#
# THE TYPE IS A C STRING POINTER and stays one. Every raise site passes a
# literal -- "TypeError", "ValueError" -- and the C compares them with `strcmp`
# and hands them to `apy_lit`. Storing the address is what the C did; copying
# the text would be a second owner for something that never changes.


def apy_err_slots() -> ptr:
    """Two words: the pending type's name, then the exception object.

    ZERO MEANS NO ERROR for the first and MEANS "NOT FROM A `raise`" for the
    second, and both readings are what the C's zero-initialised statics
    meant -- so unlike the positions above, this needs no bias.
    """
    return reserve("apy_err_slots_ir", 16)


def apy_err_text() -> ptr:
    """The 256-byte message buffer. `snprintf`'s destination, still."""
    return reserve("apy_err_text_ir", 256)


def apy_err_kind() -> ptr:
    """The pending error's type name, or null."""
    return ptr(load(u64, apy_err_slots()))


def apy_err_set_kind(name: ptr) -> None:
    """Record the pending error's type name."""
    store(u64, u64(name), apy_err_slots())


def apy_err_obj() -> ptr:
    """The exception object, or null when the error came from an operation."""
    return ptr(load(u64, offset(apy_err_slots(), 8)))


def apy_err_set_obj(exc: ptr) -> None:
    """Record the exception object a `raise` supplied."""
    store(u64, u64(exc), offset(apy_err_slots(), 8))


def apy_error_occurred() -> i64:
    """Is an error pending?

    THE ONE LINE TEN FUNCTIONS WERE WAITING FOR.
    """
    if apy_err_kind():
        return 1
    return 0


def apy_error_clear() -> None:
    """Drop the pending error.

    THE MESSAGE IS NOT CLEARED, exactly as the C did not clear it: nothing
    reads the text without first finding a type, so blanking 256 bytes would
    be work no reader can observe. `apy_err_msg[0]` is only consulted on the
    path where a type was already found.
    """
    apy_err_set_kind(ptr(0))
    apy_err_set_obj(ptr(0))


# ── the source-position table ──────────────────────────────────────────────
#
# WHERE EVERY STATEMENT IS, recorded once at startup so a traceback can name a
# line later. `apy_at` above stores an INDEX into this table; this is the table.
#
# THE SAME MOVE AS THE PENDING ERROR, for the same reason: the function was
# closed over everything IR defines except its own storage. Three words behind
# an accessor with a C body, so the runtime still stands alone.
#
# IT GROWS BY DOUBLING, which is why it needs `runtime/blocks.py` and could not
# have moved at stage 4 -- the bump arena can neither resize nor reclaim, and a
# table that doubles releases the block it grew out of.


# THE NUMBERS BELOW ARE THE C COMPILER'S. See `runtime/slots.py`.


def apy_pos_row_size() -> i64:
    return 24


def apy_pos_fn_offset() -> i64:
    return 0


def apy_pos_line_offset() -> i64:
    return 8


def apy_pos_end_line_offset() -> i64:
    return 12


def apy_pos_col_offset() -> i64:
    return 16


def apy_pos_end_col_offset() -> i64:
    return 20


def apy_pos_table() -> ptr:
    """Three words: the rows, how many are used, how many fit."""
    return reserve("apy_pos_table_ir", 24)


def apy_pos_rows() -> ptr:
    """The table itself. Null until the first position is recorded."""
    return ptr(load(u64, apy_pos_table()))


def apy_pos_count() -> i64:
    """How many positions have been recorded."""
    return load(i64, offset(apy_pos_table(), 8))


def apy_pos_add(name: ptr, line: i64, end_line: i64, col: i64,
                end_col: i64) -> None:
    """Record one statement's position.

    A FAILED GROW DROPS THE POSITION AND NOTHING ELSE, exactly as the C's
    `if (!apy_pos_tab) return;` did. A program that cannot record where its
    statements are still runs; it just cannot say where it stopped, which is
    a far better failure than refusing to start.

    128 ROWS FIRST, THEN DOUBLING, which is the C's schedule. The first
    reservation is the one that matters: a module of any size records
    hundreds, so starting at one would mean seven reallocations before the
    table is even useful.
    """
    state: ptr = apy_pos_table()
    n: i64 = load(i64, offset(state, 8))
    cap: i64 = load(i64, offset(state, 16))
    if n == cap:
        grown: i64 = 128
        if cap:
            grown = cap * 2
        rows: ptr = apy_realloc_block(ptr(load(u64, state)),
                                      cap * apy_pos_row_size(),
                                      grown * apy_pos_row_size())
        if not rows:
            return
        store(u64, u64(rows), state)
        store(i64, grown, offset(state, 16))
    at: ptr = offset(apy_pos_rows(), n * apy_pos_row_size())
    store(u64, u64(name), offset(at, apy_pos_fn_offset()))
    store(i32, i32(line), offset(at, apy_pos_line_offset()))
    store(i32, i32(end_line), offset(at, apy_pos_end_line_offset()))
    store(i32, i32(col), offset(at, apy_pos_col_offset()))
    store(i32, i32(end_col), offset(at, apy_pos_end_col_offset()))
    store(i64, n + 1, offset(state, 8))


# ── raising ────────────────────────────────────────────────────────────────
#
# `apy_fail` AND `apy_fail2` ARE THE TWO BIGGEST WALLS the survey reports, and
# neither is what it looks like.
#
# `apy_fail` LOOKED LIKE STDIO and is not: its `snprintf(buf, cap, "%s", msg)`
# has no conversion to perform. It is a bounded string copy written in the
# spelling C reaches for.
#
# `apy_fail2` LOOKED LIKE FORMATTING and is not either. Counted rather than
# assumed: 153 call sites pass 280 conversions between them and EVERY ONE IS
# `%s`. Nothing anywhere asks for a number, a width or a precision -- the
# whole family is one template with two strings dropped into it.
#
# So the wall was a spelling. Both are written out here, and the C keeps
# `apy_fail` and `apy_fail2` as one-line delegates so its 153 call sites do
# not move.
#
# TRUNCATION IS COPIED EXACTLY, because it is observable. `snprintf` writes at
# most cap-1 bytes and always terminates, so a message longer than 255 loses
# its tail and not its terminator -- and `apy_fail2` truncates TWICE, once
# building the text and once storing it. Both are reproduced; a version that
# truncated only once would differ from the C on exactly the long messages
# nobody writes a test for.


def apy_fmt_scratch() -> ptr:
    """Where `apy_raise_fmt` builds its text before storing it.

    A SECOND BUFFER, because the C has one: `apy_fail2` fills a local and
    hands it to `apy_fail`, which copies it again. Expanding straight into
    the message buffer would read an argument out of the buffer it was
    overwriting if a caller ever passed the pending message back in.
    """
    return reserve("apy_fmt_scratch_ir", 256)


def apy_err_cap() -> i64:
    """The message buffer's size, terminator included."""
    return 256


def apy_cstr_into(dst: ptr, at: i64, cap: i64, src: ptr) -> i64:
    """Copy a NUL-terminated string, stopping `cap` bytes from the start.

    ANSWERS WHERE THE NEXT BYTE GOES, so a caller appending several pieces
    keeps no running total of its own. It writes no terminator: the caller
    does that once, which is what makes a truncated result still terminated.
    """
    i: i64 = at
    while i < cap:
        b: u8 = load(u8, offset(src, i - at))
        if b == u8(0):
            return i
        store(u8, b, offset(dst, i))
        i = i + 1
    return i


def apy_raise_at(kind: ptr, msg: ptr) -> ptr:
    """Record a pending error. The IR half of the C's `apy_fail`.

    FIRST ERROR WINS, like a real traceback: a second failure while one is
    pending changes nothing. The position is latched HERE rather than by the
    caller, because this is the choke point every failed operation comes
    through -- by the time a handler asks, its own statements have moved the
    cursor.

    ANSWERS ZERO, which is what every caller returns straight back up. A
    failed operation has no value to give.
    """
    if apy_err_kind():
        return ptr(0)
    apy_pos_latch()
    apy_err_set_kind(kind)
    apy_err_set_obj(ptr(0))
    at: i64 = apy_cstr_into(apy_err_text(), 0, apy_err_cap() - 1, msg)
    store(u8, u8(0), offset(apy_err_text(), at))
    return ptr(0)


def apy_raise_fmt(kind: ptr, fmt: ptr, a: ptr, b: ptr) -> ptr:
    """Record a pending error whose text is `fmt` with `a` and `b` in it.

    TWO ARGUMENTS AND NO MORE, which is the C's signature and is enough for
    all 153 sites -- the ones needing only one pass `""` as the second, and
    the template ends in `%s` so the empty string lands where nothing shows.

    A `%` NOT FOLLOWED BY `s` IS COPIED LITERALLY rather than refused. No
    format in the runtime contains one, so this is a rule about what happens
    to a format nobody has written yet: it appears in the message instead of
    swallowing the character after it.
    """
    buf: ptr = apy_fmt_scratch()
    cap: i64 = apy_err_cap() - 1
    out: i64 = 0
    i: i64 = 0
    which: i64 = 0
    while out < cap:
        c: u8 = load(u8, offset(fmt, i))
        if c == u8(0):
            break
        if c == u8(37) and load(u8, offset(fmt, i + 1)) == u8(115):
            arg: ptr = b
            if which == 0:
                arg = a
            which = which + 1
            out = apy_cstr_into(buf, out, cap, arg)
            i = i + 2
        else:
            store(u8, c, offset(buf, out))
            out = out + 1
            i = i + 1
    store(u8, u8(0), offset(buf, out))
    return apy_raise_at(kind, buf)


def apy_raise_over(kind: ptr, msg: ptr) -> ptr:
    """Record a pending error, replacing whatever was already there.

    AN EXPLICIT `raise` IS NOT A FAILED OPERATION, which is the whole reason
    this exists beside `apy_raise_at`. After one error the frontend may run
    more operations before it checks, and the second failure is a consequence
    of the first rather than the cause -- so the first writer wins there. A
    `raise` statement is not that: reaching one means control deliberately got
    there, and Python's rule is that the new exception wins.

        try:
            raise ValueError("original")
        finally:
            raise KeyError("from-finally")   # this is what propagates

    THE SAME BODY WITHOUT THE GUARD, written out rather than sharing one with
    a flag: a flag would put the choice at the call site, and the choice is
    the difference between these two functions.
    """
    apy_pos_latch()
    apy_err_set_kind(kind)
    apy_err_set_obj(ptr(0))
    at: i64 = apy_cstr_into(apy_err_text(), 0, apy_err_cap() - 1, msg)
    store(u8, u8(0), offset(apy_err_text(), at))
    return ptr(0)


# ── the first ported function that names a string ──────────────────────────
#
# `rodata` WAS DECLARED AND UNUSED until this. It is what lets the subset
# point at bytes that are compiled in rather than built at run time, and
# without it every function reaching `apy_fail` or `apy_fail2` was unportable
# no matter what else had moved -- because all 153 of those call sites pass
# LITERALS, and a literal is not a call for a survey to notice.
#
# NO TERMINATOR IS ADDED, which its own docstring says and which a probe
# confirmed the hard way: `rodata(b"hello")` is five bytes and a scan for the
# NUL runs into whatever the linker put next. Every literal below writes its
# own, because these are C strings the rest of the runtime reads with `strlen`.


def apy_check_bound(v: ptr, name: ptr) -> ptr:
    """`v` if the local is bound, else the UnboundLocalError Python words.

    THE MESSAGE ENDS IN `%s` WITH NOTHING FOR IT, which looks like a mistake
    and is the C's own shape: `apy_raise_fmt` takes exactly two arguments
    because that is what its 153 call sites needed between them, and a message
    wanting only one passes an empty second. The alternative is a second
    function differing in one parameter.

    THE NAME IS READ AS A C STRING, not as a cell: `APY_CSTR` in the C is the
    string's byte pointer, and the substitution walks it to its NUL.
    """
    if v:
        return v
    return apy_raise_fmt(
        rodata(b"UnboundLocalError\0"),
        rodata(b"cannot access local variable '%s' where it is not "
        b"associated with a value%s\0"),
        apy_str_data(name),
        rodata(b"\0"))


# ── the user exception table ───────────────────────────────────────────────
#
# WHAT `class MyError(ValueError)` LEAVES BEHIND: a name and its parent's name,
# so `except ValueError` can catch it later. Sixty-four of them, in a fixed
# table, because the C's is fixed and a program that declares a
# sixty-fifth gets a RuntimeError rather than a reallocation.
#
# THE STORAGE MOVED FIRST, as it had to: `apy_exc_register` calls nothing the
# IR lacks, and was still unportable because it names a C static. That is the
# third kind of dependency `uasm plugin port` now reports, after calls and libc.


def apy_user_exc_max() -> i64:
    """How many user exception classes a program may declare."""
    return 64


def apy_user_exc_rows() -> ptr:
    """The table: `max` pairs of C string pointers, name then parent."""
    return reserve("apy_user_exc_ir", 1024)


def apy_user_exc_slot() -> ptr:
    """Where the count lives."""
    return reserve("apy_user_exc_n_ir", 8)


def apy_user_exc_at(i: i64, half: i64) -> ptr:
    """One half of row `i` -- 0 is the name, 1 is the parent's name."""
    return ptr(load(u64, offset(apy_user_exc_rows(), i * 16 + half * 8)))


def apy_cstr_eq(a: ptr, b: ptr) -> bool:
    """`strcmp(a, b) == 0`, written out.

    STOPS AT THE FIRST DIFFERENCE OR THE FIRST NUL, and the order of those
    two tests is what makes it right: comparing the bytes first would walk
    past the end of the shorter string when one is a prefix of the other.
    """
    i: i64 = 0
    while True:
        x: u8 = load(u8, offset(a, i))
        if x != load(u8, offset(b, i)):
            return False
        if x == u8(0):
            return True
        i = i + 1
    return False


def apy_exc_register(name: ptr, parent: ptr) -> ptr:
    """Remember that `name` is an exception class derived from `parent`.

    REGISTERING TWICE IS NOT AN ERROR and answers None, which matters because
    a class statement inside a function runs every time the function is
    called. The C searches before it appends for exactly that.

    BOTH NAMES ARE READ AS C STRINGS out of their cells and kept as POINTERS,
    not copied. That is the C's arrangement and it is safe for the same
    reason: a class name is a compiled-in literal, so the bytes outlive any
    table that points at them.
    """
    n: ptr = apy_str_data(name)
    used: i64 = load(i64, apy_user_exc_slot())
    i: i64 = 0
    while i < used:
        if apy_cstr_eq(apy_user_exc_at(i, 0), n):
            return apy_none()
        i = i + 1
    if used >= apy_user_exc_max():
        return apy_raise_at(rodata(b"RuntimeError\0"),
                            rodata(b"too many user-defined exception classes\0"))
    store(u64, u64(n), offset(apy_user_exc_rows(), used * 16))
    store(u64, u64(apy_str_data(parent)),
          offset(apy_user_exc_rows(), used * 16 + 8))
    store(i64, used + 1, apy_user_exc_slot())
    return apy_none()



# ── the exception hierarchy ────────────────────────────────────────────────
#
# WHICH EXCEPTION INHERITS WHICH, so `except ArithmeticError` catches a
# ZeroDivisionError. The C keeps it as `APY_EXC_TREE`, a table of string
# pointers -- and a table of POINTERS is the one thing `rodata` cannot give,
# because it hands back bytes and a pointer needs the linker to fill it in.
#
# SO THE SHAPE CHANGES AND THE CONTENT DOES NOT: name, NUL, parent, NUL,
# repeated, and one more NUL to end it. Walking it is the same linear scan the
# C does -- `apy_exc_parent` compares against every entry in order -- so the
# packing costs nothing it was not already paying.
#
# THE TWO COPIES CANNOT DRIFT, because a test compares them:
# `tests/uasm/integration/test_ported_int.py` reads `APY_EXC_TREE` out of
# the C and this blob out of here, and fails if either grows an entry the
# other does not have. That is the same arrangement the layout constants use,
# and for the same reason: two hand-written copies of one fact is a fact that
# will eventually be two.


def apy_exc_tree() -> ptr:
    """The built-in hierarchy, packed. See the note above."""
    return rodata(
        b"Exception\0BaseException\0"
        b"SystemExit\0BaseException\0"
        b"KeyboardInterrupt\0BaseException\0"
        b"GeneratorExit\0BaseException\0"
        b"CancelledError\0BaseException\0"
        b"InvalidStateError\0Exception\0"
        b"BaseExceptionGroup\0BaseException\0"
        b"ExceptionGroup\0Exception\0"
        b"ArithmeticError\0Exception\0"
        b"ZeroDivisionError\0ArithmeticError\0"
        b"OverflowError\0ArithmeticError\0"
        b"FloatingPointError\0ArithmeticError\0"
        b"LookupError\0Exception\0"
        b"IndexError\0LookupError\0"
        b"KeyError\0LookupError\0"
        b"NameError\0Exception\0"
        b"UnboundLocalError\0NameError\0"
        b"AttributeError\0Exception\0"
        b"TypeError\0Exception\0"
        b"ValueError\0Exception\0"
        b"UnicodeError\0ValueError\0"
        b"UnicodeDecodeError\0UnicodeError\0"
        b"UnicodeEncodeError\0UnicodeError\0"
        b"UnicodeTranslateError\0UnicodeError\0"
        b"RuntimeError\0Exception\0"
        b"NotImplementedError\0RuntimeError\0"
        b"RecursionError\0RuntimeError\0"
        b"AssertionError\0Exception\0"
        b"ImportError\0Exception\0"
        b"ModuleNotFoundError\0ImportError\0"
        b"OSError\0Exception\0"
        b"FileNotFoundError\0OSError\0"
        b"PermissionError\0OSError\0"
        b"IsADirectoryError\0OSError\0"
        b"NotADirectoryError\0OSError\0"
        b"FileExistsError\0OSError\0"
        b"InterruptedError\0OSError\0"
        b"BlockingIOError\0OSError\0"
        b"ChildProcessError\0OSError\0"
        b"ProcessLookupError\0OSError\0"
        b"ConnectionError\0OSError\0"
        b"BrokenPipeError\0ConnectionError\0"
        b"ConnectionAbortedError\0ConnectionError\0"
        b"ConnectionRefusedError\0ConnectionError\0"
        b"ConnectionResetError\0ConnectionError\0"
        b"TimeoutError\0OSError\0"
        b"StopIteration\0Exception\0"
        b"Warning\0Exception\0"
        b"UserWarning\0Warning\0"
        b"DeprecationWarning\0Warning\0"
        b"PendingDeprecationWarning\0Warning\0"
        b"SyntaxWarning\0Warning\0"
        b"RuntimeWarning\0Warning\0"
        b"FutureWarning\0Warning\0"
        b"ImportWarning\0Warning\0"
        b"UnicodeWarning\0Warning\0"
        b"BytesWarning\0Warning\0"
        b"ResourceWarning\0Warning\0"
        b"EncodingWarning\0Warning\0"
        b"StopAsyncIteration\0Exception\0"
        b"MemoryError\0Exception\0"
        b"EOFError\0Exception\0"
        b"SyntaxError\0Exception\0"
        b"IndentationError\0SyntaxError\0"
        b"TabError\0IndentationError\0"
        b"\0")


def apy_exc_parent_of(name: ptr) -> ptr:
    """The name of `name`'s base class, or null if it has none.

    BUILT-INS FIRST, THEN USER CLASSES, which is the C's order and is not
    arbitrary: a program may declare `class ValueError(Exception)` of its own,
    and the built-in answer has to win or the shadowing class would rewire the
    hierarchy for everything that never asked for it.

    NULL FOR AN UNKNOWN NAME, including for `BaseException` -- which is the
    root and genuinely has no parent, so the caller's walk up the chain ends
    there rather than needing a name to stop at.
    """
    at: ptr = apy_exc_tree()
    while load(u8, at) != u8(0):
        parent: ptr = apy_cstr_end(at)
        if apy_cstr_eq(at, name):
            return parent
        at = apy_cstr_end(parent)
    used: i64 = load(i64, apy_user_exc_slot())
    i: i64 = 0
    while i < used:
        if apy_cstr_eq(apy_user_exc_at(i, 0), name):
            return apy_user_exc_at(i, 1)
        i = i + 1
    return ptr(0)


def apy_cstr_end(p: ptr) -> ptr:
    """Just past `p`'s terminator -- where the next packed string starts."""
    n: i64 = 0
    while load(u8, offset(p, n)) != u8(0):
        n = n + 1
    return offset(p, n + 1)


def apy_error_matches(handler: ptr) -> i64:
    """Does the pending error match a handler named `handler`?

    WALKS UP FROM THE RAISED TYPE, which is what makes a base class catch
    every derived one: a KeyError asks `apy_exc_parent_of` for LookupError,
    then Exception, then BaseException, and matches `except Exception` on the
    way. Walking DOWN would need the tree indexed the other way and a list of
    every subclass, which is the same fact stored twice.

    THE CHAIN ENDS AT A NAME WITH NO PARENT, and `BaseException` is that name
    -- so nothing has to know it is the root. A user class whose parent was
    never registered ends there too, which is why `test_exc_tree.py` checks
    that every parent is itself declared.
    """
    have: ptr = apy_err_kind()
    want: ptr = apy_str_data(handler)
    while have:
        if apy_cstr_eq(have, want):
            return 1
        have = apy_exc_parent_of(have)
    return 0


def apy_error_message() -> ptr:
    """The pending error's text as a string, or None if none is pending.

    A COPY, NOT A VIEW OF THE BUFFER. The message lives in 256 bytes that the
    next error overwrites, so a cell pointing into them would change its own
    value the moment anything else failed -- and `except E as e` keeps `e`
    long past that point.
    """
    if not apy_err_kind():
        return apy_none()
    p: ptr = apy_err_text()
    n: i64 = 0
    while load(u8, offset(p, n)) != u8(0):
        n = n + 1
    return apy_str_copy_bytes(p, n)


# ── the bundled module names ───────────────────────────────────────────────
#
# WHICH MODULES THIS BUILD RESOLVES AT COMPILE TIME. `import x` is a compile-
# time decision here, so a dynamic `__import__` can never succeed -- but the
# two ways it fails say different things, and both are honest: a module this
# build does not have is a ModuleNotFoundError exactly as in CPython, and one
# it does have is an ImportError saying the import cannot be done that way.
#
# PACKED LIKE THE EXCEPTION TREE, and for the same reason: `rodata` gives
# bytes and a table of POINTERS needs the linker. Names, each terminated,
# then one more terminator to end the list.
#
# `tests/uasm/integration/test_exc_tree.py` COMPARES THE TWO COPIES so
# that adding a bundled module to one and not the other cannot happen
# quietly.


def apy_known_modules() -> ptr:
    """The modules this build bundles, packed. See the note above."""
    return rodata(
        b"__future__\0"
        b"_pyast\0"
        b"_pycompile\0"
        b"_pylex\0"
        b"_pyparse\0"
        b"_pyrun\0"
        b"_pyvalidate\0"
        b"abc\0"
        b"annotationlib\0"
        b"argparse\0"
        b"asyncio\0"
        b"bisect\0"
        b"collections\0"
        b"collections.abc\0"
        b"contextlib\0"
        b"contextvars\0"
        b"copy\0"
        b"dataclasses\0"
        b"datetime\0"
        b"decimal\0"
        b"difflib\0"
        b"enum\0"
        b"fractions\0"
        b"functools\0"
        b"gc\0"
        b"heapq\0"
        b"inspect\0"
        b"io\0"
        b"itertools\0"
        b"json\0"
        b"keyword\0"
        b"math\0"
        b"numbers\0"
        b"operator\0"
        b"os\0"
        b"pathlib\0"
        b"pprint\0"
        b"random\0"
        b"re\0"
        b"select\0"
        b"shlex\0"
        b"socket\0"
        b"statistics\0"
        b"string\0"
        b"struct\0"
        b"subprocess\0"
        b"sys\0"
        b"textwrap\0"
        b"threading\0"
        b"time\0"
        b"tomllib\0"
        b"traceback\0"
        b"types\0"
        b"typing\0"
        b"unicodedata\0"
        b"warnings\0"
        b"weakref\0"
        b"\0")


def apy_module_is_known(want: ptr) -> bool:
    """Is `want` one of the bundled names?"""
    at: ptr = apy_known_modules()
    while load(u8, at) != u8(0):
        if apy_cstr_eq(at, want):
            return True
        at = apy_cstr_end(at)
    return False


def apy_import(name: ptr) -> ptr:
    """`__import__(name)` -- which always fails, in one of two ways.

    A NON-STRING ARGUMENT BECOMES THE EMPTY NAME rather than a TypeError,
    which is the C's behaviour and worth keeping deliberately: the message
    then reports `No module named ''`, and a program that reached here with
    a non-string was already going to fail.
    """
    want: ptr = apy_empty_name()
    if apy_is_str(name):
        want = apy_str_data(name)
    if apy_module_is_known(want):
        return apy_raise_fmt(
            rodata(b"ImportError\0"),
            rodata(b"cannot import '%s' dynamically: this build "
                   b"resolves imports at compile time%s\0"),
            want, rodata(b"\0"))
    return apy_raise_fmt(
        rodata(b"ModuleNotFoundError\0"),
        rodata(b"No module named '%s'%s\0"),
        want, rodata(b"\0"))


def apy_empty_name() -> ptr:
    """An empty C string, for an argument that was not one."""
    return rodata(b"\0")


def apy_exc_class_slot() -> ptr:
    """Where the name-to-class table for program exceptions lives."""
    return reserve("apy_exc_class_ir", 8)


def apy_exc_class_bind(name: ptr, cls: ptr) -> ptr:
    """Remember that `name` is an exception class this program wrote.

    BY NAME AND NOT BY OBJECT, because that is all the raising side has:
    `apy_make_exc` is handed a string, and the `class` statement that wrote
    the body may be in another function entirely. Nothing else connects the
    two.
    """
    slot: ptr = apy_exc_class_slot()
    held: ptr = ptr(load(u64, slot))
    if not held:
        held = apy_dict_new(8)
        if not held:
            return ptr(0)
        store(u64, u64(held), slot)
    if not apy_dict_set(held, name, cls):
        return ptr(0)
    return apy_none()


def apy_decimal_slots() -> ptr:
    """Two 24-byte cells, for the two numbers a message can name.

    TWO BECAUSE `apy_raise_fmt` TAKES TWO ARGUMENTS and both may be numbers --
    "expected 2, got 3" is exactly that shape. A single buffer would have the
    second render overwrite the first before either was read.

    TWENTY-FOUR EACH is room for the longest int64 (`-9223372036854775808`,
    twenty characters) and its terminator, rounded up.
    """
    return reserve("apy_decimal_slots_ir", 48)


def apy_decimal_of(v: i64, which: i64) -> ptr:
    """`v` written in decimal, as a C string, in slot `which`.

    WHAT `snprintf("%lld")` DID. The C builds its messages with it and the
    subset has no varargs and no libc -- so a number that has to appear in a
    message is rendered here and handed to `apy_raise_fmt` as text.

    THE DIGITS COME OUT BACKWARDS and are reversed in place, which is what
    dividing by ten gives you and is cheaper than counting the digits first.

    INT64_MIN IS WHY THE MAGNITUDE IS UNSIGNED: its positive form does not fit
    an int64, so negating it in signed arithmetic answers itself and the
    number would print without its digits.
    """
    buf: ptr = offset(apy_decimal_slots(), which * 24)
    at: i64 = 0
    if v < 0:
        store(u8, u8(45), buf)
        at = 1
    m: u64 = u64(v)
    if v < 0:
        m = u64(0) - u64(v)
    start: i64 = at
    if m == 0:
        store(u8, u8(48), offset(buf, at))
        at = at + 1
    while m != 0:
        store(u8, u8(48 + i64(m % u64(10))), offset(buf, at))
        m = m // u64(10)
        at = at + 1
    store(u8, u8(0), offset(buf, at))
    i: i64 = 0
    half: i64 = (at - start) // 2
    while i < half:
        lo: ptr = offset(buf, start + i)
        hi: ptr = offset(buf, at - 1 - i)
        c: u8 = load(u8, lo)
        store(u8, load(u8, hi), lo)
        store(u8, c, hi)
        i = i + 1
    return buf


def apy_nat_positions() -> i64:
    return 25


# -- the `code` object a traceback frame names ------------------------------


def apy_code_slots() -> ptr:
    """Two words: the one `code` class, and the cache of made ones."""
    return reserve("apy_code_slots_ir", 16)


def apy_code_class() -> ptr:
    """The class every code object is an instance of, made once.

    `co_positions` IS A METHOD AND NOT AN ATTRIBUTE, because that is how
    CPython spells it -- a program CALLS it, and `hasattr(code,
    "co_positions")` is the test written before it does.
    """
    held: ptr = ptr(load(u64, apy_code_slots()))
    if held:
        return held
    cls: ptr = apy_type_new(apy_from_cstr(rodata(b"code\0")), ptr(0))
    if not cls:
        return cls
    store(u64, u64(cls), apy_code_slots())
    apy_type_set(cls, apy_from_cstr(rodata(b"co_positions\0")),
                 apy_native_of(apy_nat_positions(), 1,
                               rodata(b"co_positions\0")))
    return cls


def apy_code_cache() -> ptr:
    """One code object per function NAME, so `f.__code__ is f.__code__`."""
    slot: ptr = offset(apy_code_slots(), 8)
    held: ptr = ptr(load(u64, slot))
    if held:
        return held
    made: ptr = apy_dict_new(8)
    if made:
        store(u64, u64(made), slot)
    return made


def apy_code_of(name: ptr) -> ptr:
    """The `code` object for the function called `name`.

    BUILT FROM THE POSITION TABLE, which is the only record a compiled
    program keeps of where its source was: every row the frontend emitted for
    this function becomes one `co_positions()` tuple.

    `co_filename` IS `<compiled>` AND NOT THE SOURCE PATH, because the path
    is a compile-time fact and the binary does not carry it -- saying so is
    more honest than naming a file that may not be there.

    CACHED BY NAME, so `f.__code__ is f.__code__` holds -- an identity a
    program can and does compare.
    """
    cls: ptr = apy_code_class()
    if not cls:
        return cls
    made: ptr = apy_code_cache()
    if not made:
        return made
    seen: ptr = apy_dict_get_or(made, name, ptr(0))
    if seen:
        return seen
    code: ptr = apy_instance_new(cls)
    if not code:
        return code
    apy_setattr(code, apy_from_cstr(rodata(b"co_name\0")), name)
    apy_setattr(code, apy_from_cstr(rodata(b"co_qualname\0")), name)
    apy_setattr(code, apy_from_cstr(rodata(b"co_filename\0")),
                apy_from_cstr(rodata(b"<compiled>\0")))
    rows: ptr = apy_seq_new_of(apy_list_kind(), 8)
    if not rows:
        return rows
    n: i64 = apy_pos_count()
    table: ptr = apy_pos_rows()
    first: i64 = 0
    i: i64 = 0
    while i < n:
        row_at: ptr = offset(table, i * apy_pos_row_size())
        who: ptr = ptr(load(u64, offset(row_at, apy_pos_fn_offset())))
        if apy_eq_raw_of(who, name):
            line: i64 = i64(load(i32, offset(row_at, apy_pos_line_offset())))
            row: ptr = apy_seq_new_of(apy_tuple_kind(), 4)
            if not row:
                return row
            apy_seq_push(row, apy_from_int(line))
            apy_seq_push(row, apy_from_int(i64(load(
                i32, offset(row_at, apy_pos_end_line_offset())))))
            apy_seq_push(row, apy_from_int(i64(load(
                i32, offset(row_at, apy_pos_col_offset())))))
            apy_seq_push(row, apy_from_int(i64(load(
                i32, offset(row_at, apy_pos_end_col_offset())))))
            apy_seq_push(rows, row)
            if not first:
                first = line
        i = i + 1
    apy_setattr(code, apy_from_cstr(rodata(b"_positions\0")), rows)
    apy_setattr(code, apy_from_cstr(rodata(b"co_firstlineno\0")),
                apy_from_int(first))
    if apy_error_occurred():
        return ptr(0)
    apy_dict_set(made, name, code)
    return code


def apy_tb_slots() -> ptr:
    """Two words: the `traceback` class and the `frame` class."""
    return reserve("apy_tb_slots_ir", 16)


def apy_traceback_of(exc: ptr) -> ptr:
    """`e.__traceback__` -- one frame, built from the position it carries.

    ONE FRAME AND NOT A CHAIN. A compiled program keeps the position where
    the exception was raised and nothing above it, so `tb_next` is None --
    the chain CPython walks does not exist here and pretending otherwise
    would mean inventing frames.

    `tb_lasti` IS -1 because there is no bytecode index to give: the number
    is meaningless for compiled code, and -1 is what CPython uses for "not
    available" rather than a made-up offset.

    THE CLASSES ARE MADE ONCE, so two tracebacks are instances of one class
    and `isinstance` between them holds.
    """
    slots: ptr = apy_tb_slots()
    tbcls: ptr = ptr(load(u64, slots))
    if not tbcls:
        tbcls = apy_type_new(apy_from_cstr(rodata(b"traceback\0")), ptr(0))
        if not tbcls:
            return ptr(0)
        frcls_new: ptr = apy_type_new(
            apy_from_cstr(rodata(b"frame\0")), ptr(0))
        if not frcls_new:
            return ptr(0)
        store(u64, u64(tbcls), slots)
        store(u64, u64(frcls_new), offset(slots, 8))
    frcls: ptr = ptr(load(u64, offset(slots, 8)))
    at: i64 = load(i64, offset(exc, apy_e_pos_offset()))
    n: i64 = apy_pos_count()
    known: i64 = 0
    if at >= 0 and at < n:
        known = 1
    where: ptr = apy_from_cstr(rodata(b"<module>\0"))
    line: i64 = 0
    if known:
        row: ptr = offset(apy_pos_rows(), at * apy_pos_row_size())
        where = ptr(load(u64, offset(row, apy_pos_fn_offset())))
        line = i64(load(i32, offset(row, apy_pos_line_offset())))
    code: ptr = apy_code_of(where)
    if not code:
        return ptr(0)
    frame: ptr = apy_instance_new(frcls)
    if not frame:
        return ptr(0)
    apy_setattr(frame, apy_from_cstr(rodata(b"f_code\0")), code)
    apy_setattr(frame, apy_from_cstr(rodata(b"f_lineno\0")),
                apy_from_int(line))
    apy_setattr(frame, apy_from_cstr(rodata(b"f_globals\0")),
                apy_dict_new(1))
    apy_setattr(frame, apy_from_cstr(rodata(b"f_locals\0")),
                apy_dict_new(1))
    tb: ptr = apy_instance_new(tbcls)
    if not tb:
        return ptr(0)
    apy_setattr(tb, apy_from_cstr(rodata(b"tb_frame\0")), frame)
    apy_setattr(tb, apy_from_cstr(rodata(b"tb_lineno\0")),
                apy_from_int(line))
    apy_setattr(tb, apy_from_cstr(rodata(b"tb_lasti\0")),
                apy_from_int(-1))
    apy_setattr(tb, apy_from_cstr(rodata(b"tb_next\0")), apy_none())
    if apy_error_occurred():
        return ptr(0)
    return tb
