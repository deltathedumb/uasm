# The task predicates, in the machine subset.
#
# A TASK IS A GENERATOR WITH A FLAG. There is no separate task cell: `builtin`
# says a coroutine was wrapped by `asyncio.create_task`, and the three
# questions below are all about that same generator's state.


# THE NUMBERS BELOW ARE THE C COMPILER'S. See `runtime/slots.py`.


def apy_g_builtin_offset() -> i64:
    return 80


def apy_g_cancel_offset() -> i64:
    return 96


def apy_coro_anext() -> i64:
    return 3


def apy_coro_gather() -> i64:
    return 2


def apy_coro_task() -> i64:
    return 4


def apy_is_task(t: ptr) -> bool:
    """Is `t` a coroutine that `create_task` wrapped?

    TWO TESTS, NOT ONE. A plain coroutine is a generator too, so the kind
    alone would let `await`ing one answer `.done()` -- and `builtin` alone
    would read a field that means something else in every other arm.
    """
    if i64(load(i32, offset(t, 0))) != apy_gen_kind():
        return False
    return i64(load(i32, offset(t, apy_g_builtin_offset()))) == apy_coro_task()


def apy_task_refuse(t: ptr, name: ptr) -> ptr:
    """The AttributeError all three share, worded for the one that asked."""
    return apy_raise_fmt(
        rodata(b"AttributeError\0"),
        rodata(b"'%s' object has no attribute '%s'\0"),
        apy_kind_name_of(t), name)


def apy_task_done(t: ptr) -> ptr:
    """`task.done()` -- has it finished, however it finished?

    A NEGATIVE STATE MEANS FINISHED, which is the generator machinery's
    convention rather than this function's: the state is an index into the
    resumption points while a coroutine is running, and no index is negative.
    A task that raised and one that returned are both done.
    """
    if not apy_is_task(t):
        return apy_task_refuse(t, rodata(b"done\0"))
    # `apy_from_bool` TAKES A NUMBER and a comparison is a `bool` here --
    # one machine bit, not the i64 its parameter is. The branch says which
    # number rather than widening one silently.
    if load(i64, offset(t, apy_g_state_offset())) < 0:
        return apy_from_bool(1)
    return apy_from_bool(0)


def apy_task_cancelled(t: ptr) -> ptr:
    """`task.cancelled()` -- did it finish BECAUSE it was cancelled?

    NOT THE SAME QUESTION AS `cancel()` HAVING BEEN CALLED. A task may be
    asked to cancel and finish normally first, so this reads the exception
    it actually ended with: slot 2 holds it, and only a CancelledError there
    counts.
    """
    if not apy_is_task(t):
        return apy_task_refuse(t, rodata(b"cancelled\0"))
    if load(i64, offset(t, apy_g_state_offset())) >= 0:
        return apy_from_bool(0)
    slots: ptr = ptr(load(u64, offset(t, apy_g_slots_offset())))
    if not slots:
        return apy_from_bool(0)
    exc: ptr = ptr(load(u64, offset(slots, 2 * apy_value_size())))
    if not exc:
        return apy_from_bool(0)
    name: ptr = ptr(load(u64, offset(exc, apy_e_name_offset())))
    if apy_cstr_eq(name, rodata(b"CancelledError\0")):
        return apy_from_bool(1)
    return apy_from_bool(0)


def apy_task_cancel(t: ptr) -> ptr:
    """`task.cancel()` -- ask it to stop, and say whether asking was possible.

    A FINISHED TASK ANSWERS False and is not an error: cancelling something
    that already ended is a race a program cannot avoid, so Python reports it
    rather than raising.

    THE FLAG IS SET, NOT ACTED ON. Cancellation is delivered the next time
    the task is resumed -- this only records the request, which is why it can
    answer immediately.
    """
    if not apy_is_task(t):
        return apy_task_refuse(t, rodata(b"cancel\0"))
    if load(i64, offset(t, apy_g_state_offset())) < 0:
        return apy_from_bool(0)
    store(i32, i32(1), offset(t, apy_g_cancel_offset()))
    return apy_from_bool(1)


# ── three that came ready with `apy_is_seq` ────────────────────────────────


def apy_str_like(recv: ptr, out: ptr) -> ptr:
    """Re-tag a str result to match a bytes receiver.

    ONE IMPLEMENTATION SERVES BOTH KINDS. `b.strip()` runs the same code as
    `s.strip()` -- same layout, same walk -- and answers a str; this is what
    puts the bytes tag back afterwards, so the sharing costs one pass rather
    than a second copy of every string method.

    THE COPY IS WHY THE TAG CAN BE WRITTEN. Re-tagging the result in place
    would re-tag a LITERAL if the method answered its own receiver -- and
    `'abc'.strip()` does exactly that when there is nothing to strip. A fresh
    cell has no other owner.

    A LIST OF STRINGS IS WALKED, because `split` answers one and every piece
    of it needs the same treatment. Recursion rather than a loop over one
    level: `partition` answers a tuple, and a tuple of tuples is not ruled
    out by anything here.

    `mut` TRAVELS WITH THE TAG, which is the whole of what separates a
    bytearray from bytes: `bytearray(b"ab").upper()` is a bytearray in Python
    and came back as bytes here, which a program then could not write into.

    AND A RESULT THAT IS ALREADY BYTES STILL MAY NOT BE THE RIGHT ONE.
    `bytearray(b"a-b").partition(b"-")` hands back the SEPARATOR the caller
    passed, which is immutable; copying is what makes fixing it safe, since
    setting `mut` in place would turn the caller's own `b"-"` into a
    bytearray.
    """
    if not out:
        return out
    if i64(load(i32, offset(recv, 0))) != apy_bytes_kind():
        return out
    want: i64 = i64(load(i32, offset(recv, apy_s_mut_offset())))
    retag: i64 = 0
    if i64(load(i32, offset(out, 0))) == apy_str_kind():
        retag = 1
    elif i64(load(i32, offset(out, 0))) == apy_bytes_kind():
        if i64(load(i32, offset(out, apy_s_mut_offset()))) != want:
            retag = 1
    if retag:
        # THROUGH THE BYTES CONSTRUCTOR, not a string re-tagged in place: the
        # string constructor answers SHARED cells now, and writing the bytes
        # kind over the shared empty string turns every later `""` into `b""`
        # at once. See `apy_bytes_made_of`.
        return apy_bytes_made_of(apy_str_data(out), apy_str_byte_len(out),
                                 want)
    if apy_is_seq_of(out):
        n: i64 = load(i64, offset(out, apy_q_n_offset()))
        items: ptr = ptr(load(u64, offset(out, apy_q_items_offset())))
        i: i64 = 0
        while i < n:
            at: ptr = offset(items, i * apy_value_size())
            store(u64, u64(apy_str_like(recv, ptr(load(u64, at)))), at)
            i = i + 1
    return out


def apy_meta_for(given: ptr, bases: ptr) -> ptr:
    """The metaclass a `class` statement should use.

    AN EXPLICIT `metaclass=` WINS, and only then is it worth looking at the
    bases: a class that names one has said what it wants.

    THE FIRST BASE THAT HAS ONE, not the most derived: this walks in order
    and stops, which is what CPython's rule reduces to for the hierarchies a
    program actually writes. A real metaclass conflict is not detected here.
    """
    if given:
        if i64(load(i32, offset(given, 0))) == apy_type_kind():
            return given
    if bases:
        if apy_is_seq_of(bases):
            n: i64 = load(i64, offset(bases, apy_q_n_offset()))
            items: ptr = ptr(load(u64, offset(bases, apy_q_items_offset())))
            i: i64 = 0
            while i < n:
                base: ptr = ptr(load(u64, offset(
                    items, i * apy_value_size())))
                if i64(load(i32, offset(base, 0))) == apy_type_kind():
                    meta: ptr = ptr(load(u64, offset(
                        base, apy_t_meta_offset())))
                    if meta:
                        return meta
                i = i + 1
    return apy_none()


def apy_asyncio_gather(coros: ptr) -> ptr:
    """`asyncio.gather(*coros)` -- a coroutine that awaits several.

    A GENERATOR WITH A FLAG, like a task: `builtin` says which builtin
    coroutine this is, and the two slots hold what it needs -- the coroutines
    to run and the results as they arrive.

    THE RESULT LIST IS FILLED WITH None UP FRONT rather than appended to,
    because results arrive in completion order and have to be STORED in
    argument order. Appending would report them in whichever order they
    finished, which is the one thing `gather` promises not to do.
    """
    if not apy_is_seq_of(coros):
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"gather() takes coroutines, not %s%s\0"),
            apy_kind_name_of(coros), rodata(b"\0"))
    n: i64 = load(i64, offset(coros, apy_q_n_offset()))
    g: ptr = apy_gen_new(ptr(0), 2)
    if not g:
        return g
    store(i32, i32(1), offset(g, apy_g_coro_offset2()))
    store(i32, i32(apy_coro_gather()), offset(g, apy_g_builtin_offset()))
    slots: ptr = ptr(load(u64, offset(g, apy_g_slots_offset())))
    store(u64, u64(coros), slots)
    room: i64 = n
    if room < 1:
        room = 1
    out: ptr = apy_list_new(room)
    if not out:
        return out
    store(u64, u64(out), offset(slots, apy_value_size()))
    i: i64 = 0
    while i < n:
        apy_seq_push(out, apy_none())
        i = i + 1
    return g


def apy_live_agens_slot() -> ptr:
    """The list of async generators the program has started."""
    return reserve("apy_live_agens_ir", 8)


def apy_agen_mark(g: ptr) -> ptr:
    """Mark a generator as an ASYNC one, and remember it.

    BOTH FLAGS, because an async generator is a coroutine too -- `await`
    inside one is what makes it async, and the coroutine flag is what lets
    that through.

    REMEMBERED FOR THE SHUTDOWN, which is the only reason a list exists:
    `aclose` has to reach every one still live at the end, including the
    ones a program abandoned midway. Nothing is ever removed -- a run is
    short and the list is walked once.

    ANYTHING THAT IS NOT A GENERATOR PASSES THROUGH, because the caller
    applies this to whatever the `async def` evaluated to.
    """
    if i64(load(i32, offset(g, 0))) == apy_gen_kind():
        store(i32, i32(1), offset(g, apy_g_coro_offset()))
        store(i32, i32(1), offset(g, apy_g_agen_offset()))
        slot: ptr = apy_live_agens_slot()
        held: ptr = ptr(load(u64, slot))
        if not held:
            held = apy_seq_new_of(apy_list_kind(), 4)
            if not held:
                return g
            store(u64, u64(held), slot)
        apy_seq_push(held, g)
    return g


def apy_tasks_slot() -> ptr:
    """Every task the program handed over, so the loop can run them."""
    return reserve("apy_tasks_ir", 8)


def apy_asyncio_create_task(coro: ptr) -> ptr:
    """`asyncio.create_task(coro)` -- a task wrapping a coroutine.

    A TASK IS A GENERATOR TOO, with three slots: the coroutine it drives,
    the result once there is one, and whether it was cancelled. That is why
    it is built by `apy_gen_new` rather than being a kind of its own -- the
    loop steps it exactly as it steps anything else.

    NO STEP FUNCTION, which is what the `builtin` tag replaces: the task is
    driven by the runtime and not by compiled code, so there is no resume
    point to hold.

    ONLY A COROUTINE, and the refusal names what it got -- an ordinary
    generator looks identical from the outside and would otherwise be run
    to completion by an event loop that had no business touching it.
    """
    if i64(load(i32, offset(coro, 0))) != apy_gen_kind():
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"a coroutine was expected, got %s%s\0"),
            apy_kind_name_of(coro), rodata(b"\0"))
    if load(i32, offset(coro, apy_g_coro_offset())) == 0:
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"a coroutine was expected, got %s%s\0"),
            apy_kind_name_of(coro), rodata(b"\0"))
    t: ptr = apy_gen_new(ptr(0), 3)
    if not t:
        return t
    store(i32, i32(1), offset(t, apy_g_coro_offset()))
    store(i32, i32(apy_coro_task()), offset(t, apy_g_builtin_offset()))
    slots: ptr = ptr(load(u64, offset(t, apy_g_slots_offset())))
    store(u64, u64(coro), slots)
    store(u64, u64(0), offset(slots, apy_value_size()))
    store(u64, u64(0), offset(slots, 2 * apy_value_size()))
    slot: ptr = apy_tasks_slot()
    held: ptr = ptr(load(u64, slot))
    if not held:
        held = apy_seq_new_of(apy_list_kind(), 4)
        if not held:
            return t
        store(u64, u64(held), slot)
    apy_seq_push(held, t)
    return t


def apy_task_result(t: ptr) -> ptr:
    """`task.result()` -- what the coroutine answered, or what it raised.

    ONLY A FINISHED TASK HAS ONE, which is what the state test is: asking a
    running task for its result is an InvalidStateError in asyncio and not a
    wait.

    A STORED EXCEPTION IS RE-RAISED rather than returned, so the failure
    reaches whoever asked for the result -- which is the whole reason a task
    keeps it instead of reporting it where it happened.
    """
    if i64(load(i32, offset(t, 0))) != apy_gen_kind():
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute "
                   b"'result'%s\0"),
            apy_kind_name_of(t), rodata(b"\0"))
    if i64(load(i32, offset(t, apy_g_builtin_offset()))) != apy_coro_task():
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute "
                   b"'result'%s\0"),
            apy_kind_name_of(t), rodata(b"\0"))
    if load(i64, offset(t, apy_g_state_offset())) >= 0:
        return apy_raise_at(rodata(b"InvalidStateError\0"),
                            rodata(b"Result is not set.\0"))
    slots: ptr = ptr(load(u64, offset(t, apy_g_slots_offset())))
    failed: ptr = ptr(load(u64, offset(slots, 2 * apy_value_size())))
    if failed:
        apy_raise(failed)
        return ptr(0)
    got: ptr = ptr(load(u64, offset(slots, apy_value_size())))
    if got:
        return got
    return apy_none()


def apy_nat_tg_enter() -> i64:
    return 30


def apy_nat_tg_exit() -> i64:
    return 31


def apy_nat_tg_create() -> i64:
    return 32


def apy_taskgroup_slot() -> ptr:
    """Where the one `TaskGroup` class lives once it is built."""
    return reserve("apy_taskgroup_cls_ir", 8)


def apy_asyncio_taskgroup() -> ptr:
    """`asyncio.TaskGroup()` -- an async context manager holding tasks.

    THREE NATIVE METHODS AND A LIST. Entering answers the group itself,
    leaving waits for everything in it, and `create_task` adds to the list --
    which is all a task group is.

    THE CLASS IS MADE ONCE, so two groups are instances of one class and
    `isinstance` between them holds, as it must.
    """
    cls: ptr = ptr(load(u64, apy_taskgroup_slot()))
    if not cls:
        cls = apy_type_new(apy_from_cstr(rodata(b"TaskGroup\0")), ptr(0))
        if not cls:
            return ptr(0)
        store(u64, u64(cls), apy_taskgroup_slot())
        apy_type_set(cls, apy_from_cstr(rodata(b"__aenter__\0")),
                     apy_native_of(apy_nat_tg_enter(), 1,
                                   rodata(b"__aenter__\0")))
        apy_type_set(cls, apy_from_cstr(rodata(b"__aexit__\0")),
                     apy_native_of(apy_nat_tg_exit(), 4,
                                   rodata(b"__aexit__\0")))
        apy_type_set(cls, apy_from_cstr(rodata(b"create_task\0")),
                     apy_native_of(apy_nat_tg_create(), 2,
                                   rodata(b"create_task\0")))
    g: ptr = apy_instance_new(cls)
    if not g:
        return ptr(0)
    apy_setattr(g, apy_from_cstr(rodata(b"_tasks\0")),
                apy_seq_new_of(apy_list_kind(), 4))
    if apy_error_occurred():
        return ptr(0)
    return g
