# The generator frame's fields, in the machine subset.
#
# STAGE 5 OF docs/INERT-RUNTIME.md, and the arm that exists because a
# `yield` breaks the one assumption every other kind makes: that a function's
# locals live in registers. A register does not survive the return a `yield`
# compiles to, so a generator's locals live in a heap array -- `v.g.slots` --
# and `state` records which `yield` it stopped at, so the next `send` can
# branch straight back to it.
#
# WHAT IS HERE is the frame's accessors: read and write a slot, read and set
# the resume point, hand over what was sent in, record what was returned. What
# is NOT here is `apy_gen_new` and the stepping itself -- those decide the
# layout and drive the switch, and they move when the generator does.
#
# TWO SLOT PAIRS, AND THEY ARE NOT THE SAME. `apy_gen_set`/`apy_gen_iget`
# both index `slots`, and one takes an `apy_value` while the other takes a
# machine integer: a generator over the STATIC path holds machine words in
# the same array a dynamic one holds handles in. The C casts between them and
# so does this -- the array is untyped storage, and which it is depends on the
# function that owns it.


def apy_gen_kind() -> i64:
    return 22


def apy_g_sent_offset() -> i64:
    return 16


def apy_g_slots_offset() -> i64:
    return 24


def apy_g_result_offset() -> i64:
    return 40


def apy_g_yieldfrom_offset() -> i64:
    """What this frame is delegating to -- and, for a coroutine_wrapper, the
    coroutine it stands in front of. See the C's `apy_coro_wrapper`."""
    return 104


def apy_g_wrapper_offset() -> i64:
    """Is this a coroutine_wrapper? See the C's `apy_coro_wrapper`."""
    return 120


def apy_g_pending_offset() -> i64:
    return 48


def apy_g_n_offset() -> i64:
    return 56


def apy_g_state_offset() -> i64:
    return 64


def apy_g_coro_offset() -> i64:
    return 76


def apy_g_agen_offset() -> i64:
    return 84


# ── the resume point ───────────────────────────────────────────────────────


def apy_gen_state(g: ptr) -> i64:
    """Which `yield` this generator stopped at.

    A NUMBER THE FRONTEND CHOSE, not an address: the generator body compiles
    to a switch over this, so the value only has to agree with the labels
    that same compilation emitted. That is what lets a suspended frame
    survive being written to memory and read back.
    """
    return load(i64, offset(g, apy_g_state_offset()))


def apy_gen_goto(g: ptr, k: i64) -> ptr:
    """Set the resume point. Answers None, as a statement does."""
    store(i64, k, offset(g, apy_g_state_offset()))
    return apy_none()


# ── the frame's slots ──────────────────────────────────────────────────────
#
# BOUNDS-CHECKED AND SILENT, as the C is: the index comes from a count the
# frontend computed, so an out-of-range one is a compiler bug rather than a
# program's. The check keeps that bug from becoming a wild write.


def apy_gen_set(g: ptr, i: i64, v: ptr) -> ptr:
    """Store a VALUE in slot `i`."""
    if i >= 0:
        if i < load(i64, offset(g, apy_g_n_offset())):
            slots: ptr = ptr(load(u64, offset(g, apy_g_slots_offset())))
            store(u64, u64(v), offset(slots, i * 8))
    return apy_none()


def apy_gen_iset(g: ptr, i: i64, v: i64) -> ptr:
    """Store a MACHINE INTEGER in slot `i`.

    The same array as `apy_gen_set` writes handles into. A generator in a
    statically typed function holds machine words; one on the dynamic path
    holds handles. The storage does not know which, and neither does this.
    """
    if i >= 0:
        if i < load(i64, offset(g, apy_g_n_offset())):
            slots: ptr = ptr(load(u64, offset(g, apy_g_slots_offset())))
            store(i64, v, offset(slots, i * 8))
    return apy_none()


def apy_gen_iget(g: ptr, i: i64) -> i64:
    """Read slot `i` as a machine integer. Zero if the index is out of range."""
    if i < 0:
        return 0
    if i >= load(i64, offset(g, apy_g_n_offset())):
        return 0
    slots: ptr = ptr(load(u64, offset(g, apy_g_slots_offset())))
    return load(i64, offset(slots, i * 8))


# ── what crosses the suspension ────────────────────────────────────────────


def apy_gen_sent(g: ptr) -> ptr:
    """What `gen.send(x)` handed in. The value the `yield` expression takes."""
    return ptr(load(u64, offset(g, apy_g_sent_offset())))


def apy_gen_result(g: ptr, v: ptr) -> ptr:
    """Record what the generator RETURNED, as distinct from yielded.

    `return v` inside a generator is not a value the caller iterates -- it
    becomes `StopIteration.value`, and `yield from` answers it. So it is kept
    apart from the yield channel entirely.
    """
    store(u64, u64(v), offset(g, apy_g_result_offset()))
    return apy_none()


def apy_gen_taken(g: ptr) -> ptr:
    """That recorded return value, or None.

    GUARDED ON THE KIND because `yield from` reaches here with whatever it was
    delegating to, which may be any iterable rather than a generator -- and an
    ordinary list has no `result` field, only whatever `v.q` keeps at that
    offset.
    """
    if i64(load(i32, offset(g, 0))) != apy_gen_kind():
        return apy_none()
    got: ptr = ptr(load(u64, offset(g, apy_g_result_offset())))
    if not got:
        return apy_none()
    return got


def apy_gen_throwing(g: ptr) -> i64:
    """Whether an exception is waiting to be raised at the resume point.

    `gen.throw(E)` parks the exception here rather than raising it at once:
    it has to surface INSIDE the generator, at the `yield` it is suspended
    on, so that the generator's own `try` can catch it.
    """
    if load(i64, offset(g, apy_g_pending_offset())) != 0:
        return 1
    return 0


def apy_coro_mark(g: ptr) -> ptr:
    """Mark this frame a coroutine rather than a plain generator.

    The same machinery drives both -- `await` is a suspension like `yield` --
    and this flag is what makes `inspect.iscoroutine` and the asyncio task
    layer able to tell them apart.
    """
    if i64(load(i32, offset(g, 0))) == apy_gen_kind():
        store(i32, 1, offset(g, apy_g_coro_offset()))
    return g


# ── what `inspect` asks ────────────────────────────────────────────────────


def apy_inspect_isgenerator(v: ptr) -> ptr:
    """A generator, and NOT a coroutine. The flag is what separates them."""
    if i64(load(i32, offset(v, 0))) != apy_gen_kind():
        return apy_from_bool(0)
    if load(i32, offset(v, apy_g_coro_offset())) != 0:
        return apy_from_bool(0)
    return apy_from_bool(1)


def apy_inspect_isasyncgen(v: ptr) -> ptr:
    """An `async def` containing `yield`: a frame with its own flag."""
    if i64(load(i32, offset(v, 0))) != apy_gen_kind():
        return apy_from_bool(0)
    if load(i32, offset(v, apy_g_agen_offset())) != 0:
        return apy_from_bool(1)
    return apy_from_bool(0)


def apy_inspect_iscoroutine(v: ptr) -> ptr:
    """A suspended `async def`. The frame, not the function that made it.

    AND NOT AN ASYNC GENERATOR, which is the clause this was written without
    and the differential suite caught within the hour. `apy_agen_mark` sets
    BOTH `coro` and `agen` -- an `async def` containing `yield` is a
    coroutine-flavoured frame that is not a coroutine -- so testing `coro`
    alone answers True for one, and `inspect.iscoroutine(agen())` said True
    where CPython says False.

    Three flags rather than one is exactly the distinction: a plain generator
    has neither, a coroutine has `coro`, an async generator has both, and
    each of the three questions has to name what it is NOT as well as what it
    is.
    """
    if i64(load(i32, offset(v, 0))) != apy_gen_kind():
        return apy_from_bool(0)
    if load(i32, offset(v, apy_g_coro_offset())) == 0:
        return apy_from_bool(0)
    if load(i32, offset(v, apy_g_agen_offset())) != 0:
        return apy_from_bool(0)
    return apy_from_bool(1)


def apy_inspect_iscoroutinefunction(v: ptr) -> ptr:
    """The FUNCTION, not the frame -- `async def f` before it is called."""
    if i64(load(i32, offset(v, 0))) != apy_func_kind():
        return apy_from_bool(0)
    if load(i32, offset(v, apy_fn_coro_offset())) != 0:
        return apy_from_bool(1)
    return apy_from_bool(0)


# ── two more frame reads ───────────────────────────────────────────────────


def apy_gen_pending(g: ptr) -> ptr:
    """Take the exception `gen.throw` parked, CLEARING it. None if there is none.

    IT CLEARS AS IT READS, and that is the whole protocol rather than an
    optimisation: the resumed generator raises what it takes, and a pending
    exception left in place would be raised again at the next `yield` -- so
    the same `throw` would fire twice and the generator's own `except` would
    see it in a loop.
    """
    got: ptr = ptr(load(u64, offset(g, apy_g_pending_offset())))
    store(i64, 0, offset(g, apy_g_pending_offset()))
    if not got:
        return apy_none()
    return got


def apy_gen_slot(g: ptr, i: i64) -> ptr:
    """Read slot `i` as a VALUE. None if out of range, and None if empty.

    A NULL SLOT IS `None` HERE, not an error. A frame's slots are allocated
    for every local the function has and filled as execution reaches them, so
    a generator suspended at its first `yield` has slots that were never
    written -- and reading one has to answer something a program can hold.
    """
    if i < 0:
        return apy_none()
    if i >= load(i64, offset(g, apy_g_n_offset())):
        return apy_none()
    slots: ptr = ptr(load(u64, offset(g, apy_g_slots_offset())))
    got: ptr = ptr(load(u64, offset(slots, i * 8)))
    if not got:
        return apy_none()
    return got


def apy_g_running_offset() -> i64:
    return 72


def apy_gen_step_of(g: ptr, sent: ptr, done: ptr) -> ptr:
    """Resume a generator once. `done` is filled with whether it finished.

    TWO ANSWERS, AND BOTH ARE NEEDED: a generator may legitimately yield
    None, so "did this call finish it" is a question about the STATE after
    the call rather than about the value. The body sets the state to -1 on
    its way out.

    THE RUNNING FLAG IS WHAT CATCHES RE-ENTRY -- `next(g)` from inside `g` --
    which would otherwise resume a frame that is already live and corrupt
    both walks.

    PEP 479. A `StopIteration` that ESCAPES a generator body becomes a
    RuntimeError with the original as its `__cause__`. Left alone it is
    indistinguishable from the generator finishing normally, so a bug inside
    the body reads as a clean end of iteration -- which is the entire reason
    the PEP exists.

    THROUGH `apy_call` AND NOT A RAW `callptr`: a generator\'s step is an
    ordinary compiled function of one argument, so the split\'s fast path
    takes it -- and anything unusual is still answered correctly by the half
    underneath rather than crashing here.
    """
    store(i64, 0, done)
    # A WRAPPER DELEGATES, having no body of its own -- see the C's
    # `apy_coro_wrapper`. Read out HERE rather than called for: the ported
    # runtime reaches nothing in the C but the `_slow` halves of a split,
    # which `test_ported_int.py` holds it to.
    if i64(load(i32, offset(g, 0))) == apy_gen_kind():
        if load(i32, offset(g, apy_g_wrapper_offset())):
            held: ptr = ptr(load(u64, offset(g, apy_g_yieldfrom_offset())))
            if held:
                g = held
    if i64(load(i32, offset(g, 0))) != apy_gen_kind():
        apy_raise_fmt(rodata(b"TypeError\0"),
                      rodata(b"'%s' object is not a generator%s\0"),
                      apy_kind_name_of(g), rodata(b"\0"))
        return ptr(0)
    if load(i32, offset(g, apy_g_running_offset())) != 0:
        apy_raise_at(rodata(b"ValueError\0"),
                     rodata(b"generator already executing\0"))
        return ptr(0)
    if load(i64, offset(g, apy_g_state_offset())) < 0:
        store(i64, 1, done)
        return apy_none()
    store(u64, u64(sent), offset(g, apy_g_sent_offset()))
    store(i32, i32(1), offset(g, apy_g_running_offset()))
    argv: ptr = alloca(8)
    store(u64, u64(g), argv)
    out: ptr = apy_call(ptr(load(u64, offset(g, apy_g_step_offset()))),
                        argv, 1)
    store(i32, i32(0), offset(g, apy_g_running_offset()))
    if not out:
        store(i64, -1, offset(g, apy_g_state_offset()))
        if apy_error_occurred():
            if apy_cstr_eq(apy_err_kind(), rodata(b"StopIteration\0")):
                cause: ptr = apy_error_value()
                wrapped: ptr = apy_make_exc(
                    apy_from_cstr(rodata(b"RuntimeError\0")),
                    apy_from_cstr(rodata(
                        b"generator raised StopIteration\0")))
                if wrapped:
                    if cause:
                        store(u64, u64(cause),
                              offset(wrapped, apy_e_cause_offset()))
                apy_error_clear()
                if wrapped:
                    apy_raise(wrapped)
        return ptr(0)
    if load(i64, offset(g, apy_g_state_offset())) < 0:
        store(i64, 1, done)
    return out


def apy_gen_drain(g: ptr) -> ptr:
    """Run a generator to the end and answer a list of what it yielded.

    A GUARD RATHER THAN NO BOUND, because a generator with no end is a normal
    thing to write and `list(g)` on one has to stop somewhere. A million is
    the C\'s number and is kept.
    """
    out: ptr = apy_seq_new_of(apy_list_kind(), 8)
    if not out:
        return out
    slot: ptr = alloca(8)
    guard: i64 = 0
    going: i64 = 1
    while going:
        if guard >= 1000000:
            going = 0
        else:
            v: ptr = apy_gen_step_of(g, apy_none(), slot)
            if not v:
                return ptr(0)
            if load(i64, slot):
                going = 0
            else:
                apy_seq_push(out, v)
                guard = guard + 1
    return out


def apy_gen_close(g: ptr) -> ptr:
    """`g.close()` -- ask a suspended generator to unwind.

    A `GeneratorExit` IS THROWN IN AND EXPECTED BACK. The body\'s `finally`
    clauses run, and the exception reaching here again is the normal ending
    rather than a failure -- so it is cleared. Anything ELSE the body raises
    is real and propagates.

    A GENERATOR NOT STARTED OR ALREADY FINISHED just goes to done, which is
    what `state > 0` gates: there is no frame to unwind.
    """
    # A WRAPPER DELEGATES, having no body of its own -- see the C's
    # `apy_coro_wrapper`. Read out here rather than called for: the ported
    # runtime reaches nothing in the C but the `_slow` halves of a split,
    # which `test_ported_int.py` holds it to.
    if i64(load(i32, offset(g, 0))) == apy_gen_kind():
        if load(i32, offset(g, apy_g_wrapper_offset())):
            wrapped: ptr = ptr(load(u64, offset(g, apy_g_yieldfrom_offset())))
            if wrapped:
                g = wrapped
    if i64(load(i32, offset(g, 0))) != apy_gen_kind():
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute "
                   b"'close'%s\0"),
            apy_kind_name_of(g), rodata(b"\0"))
    if load(i64, offset(g, apy_g_state_offset())) > 0:
        exc: ptr = apy_make_exc(
            apy_from_cstr(rodata(b"GeneratorExit\0")), apy_none())
        store(u64, u64(exc), offset(g, apy_g_pending_offset()))
        slot: ptr = alloca(8)
        apy_gen_step_of(g, apy_none(), slot)
        if apy_error_occurred():
            if apy_error_matches(
                    apy_from_cstr(rodata(b"GeneratorExit\0"))):
                apy_error_clear()
            else:
                store(i64, -1, offset(g, apy_g_state_offset()))
                return ptr(0)
    store(i64, -1, offset(g, apy_g_state_offset()))
    return apy_none()


def apy_gen_throw(g: ptr, exc: ptr) -> ptr:
    """`g.throw(exc)` -- raise `exc` at the point the generator is suspended.

    A GENERATOR NOT YET STARTED CANNOT CATCH IT, which is Python\'s rule: the
    body has not run, so there is no `try` around anything and the exception
    simply propagates from the throw.

    FINISHING ON A THROW IS A StopIteration, because the generator ended --
    the exception was caught inside and the body returned.
    """
    # A WRAPPER DELEGATES, having no body of its own -- see the C's
    # `apy_coro_wrapper`. Read out here rather than called for: the ported
    # runtime reaches nothing in the C but the `_slow` halves of a split,
    # which `test_ported_int.py` holds it to.
    if i64(load(i32, offset(g, 0))) == apy_gen_kind():
        if load(i32, offset(g, apy_g_wrapper_offset())):
            wrapped: ptr = ptr(load(u64, offset(g, apy_g_yieldfrom_offset())))
            if wrapped:
                g = wrapped
    if i64(load(i32, offset(g, 0))) != apy_gen_kind():
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute "
                   b"'throw'%s\0"),
            apy_kind_name_of(g), rodata(b"\0"))
    if load(i64, offset(g, apy_g_state_offset())) <= 0:
        store(i64, -1, offset(g, apy_g_state_offset()))
        apy_raise(exc)
        return ptr(0)
    store(u64, u64(exc), offset(g, apy_g_pending_offset()))
    slot: ptr = alloca(8)
    out: ptr = apy_gen_step_of(g, apy_none(), slot)
    if not out:
        return ptr(0)
    if load(i64, slot):
        return apy_raise_at(rodata(b"StopIteration\0"), rodata(b"\0"))
    return out


def apy_gen_stop(g: ptr) -> ptr:
    """The StopIteration a finished generator ends with, carrying its return.

    A GENERATOR\'S RETURN VALUE TRAVELS IN THE EXCEPTION, which is how
    `yield from` collects it: `x = yield from inner()` binds what `inner`
    returned, and the only place it exists is `StopIteration.value`.

    A BARE ONE FOR None, because `return` and `return None` end a generator
    the same way and neither carries anything worth attaching.
    """
    # A WRAPPER HAS NO RESULT OF ITS OWN -- the coroutine it delegates to
    # does, and `next(c.__await__())` carries what `c` returned. Read
    # out here rather than called for; see the C's
    # `apy_coro_wrapper`. Read out HERE rather than called for: the ported
    # runtime reaches nothing in the C but the `_slow` halves of a split,
    # which `test_ported_int.py` holds it to.
    if i64(load(i32, offset(g, 0))) == apy_gen_kind():
        if load(i32, offset(g, apy_g_wrapper_offset())):
            held: ptr = ptr(load(u64, offset(g, apy_g_yieldfrom_offset())))
            if held:
                g = held
    carried: ptr = ptr(load(u64, offset(g, apy_g_result_offset())))
    if not carried:
        return apy_raise_at(rodata(b"StopIteration\0"), rodata(b"\0"))
    if i64(load(i32, offset(carried, 0))) == apy_none_kind():
        return apy_raise_at(rodata(b"StopIteration\0"), rodata(b"\0"))
    return apy_raise(apy_make_exc(
        apy_from_cstr(rodata(b"StopIteration\0")), carried))


def apy_gen_next(g: ptr, fallback: ptr, has_default: i64) -> ptr:
    """`next(g)` on a generator, with `next(g, default)` beside it.

    A DEFAULT TURNS THE END INTO A VALUE. Without one the end is the
    StopIteration a finished generator carries -- which is where its return
    value lives, so the two are not interchangeable.
    """
    slot: ptr = alloca(8)
    out: ptr = apy_gen_step_of(g, apy_none(), slot)
    if not out:
        return ptr(0)
    if load(i64, slot):
        if has_default:
            return fallback
        return apy_gen_stop(g)
    return out


def apy_gen_send(g: ptr, v: ptr) -> ptr:
    """`g.send(v)` -- resume a generator, handing `v` to the `yield`.

    A JUST-STARTED GENERATOR CANNOT BE SENT ANYTHING but None, because there
    is no `yield` waiting to receive it: the body has not run, so the value
    would have nowhere to go. `g.send(None)` is the same as `next(g)` and is
    how a generator is primed.
    """
    # A WRAPPER DELEGATES, having no body of its own -- see the C's
    # `apy_coro_wrapper`. Read out here rather than called for: the ported
    # runtime reaches nothing in the C but the `_slow` halves of a split,
    # which `test_ported_int.py` holds it to.
    if i64(load(i32, offset(g, 0))) == apy_gen_kind():
        if load(i32, offset(g, apy_g_wrapper_offset())):
            wrapped: ptr = ptr(load(u64, offset(g, apy_g_yieldfrom_offset())))
            if wrapped:
                g = wrapped
    if i64(load(i32, offset(g, 0))) == apy_gen_kind():
        if load(i64, offset(g, apy_g_state_offset())) == 0:
            if i64(load(i32, offset(v, 0))) != apy_none_kind():
                return apy_raise_at(
                    rodata(b"TypeError\0"),
                    rodata(b"can't send non-None value to a "
                           b"just-started generator\0"))
    slot: ptr = alloca(8)
    out: ptr = apy_gen_step_of(g, v, slot)
    if not out:
        return ptr(0)
    if load(i64, slot):
        return apy_gen_stop(g)
    return out
