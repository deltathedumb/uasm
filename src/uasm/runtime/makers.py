# The function and generator cells, in the machine subset.
#
# STAGE 5 OF docs/INERT-RUNTIME.md, and the last two constructors that
# `apy_alloc` was gating. `runtime/funcs.py` and `runtime/gens.py` already read
# and write these two arms; what was missing was the ability to MAKE one.
#
# ── most of the C's body is not here, and that is the point ────────────────
#
# `apy_func_new` writes eighteen fields and `apy_gen_new` writes eleven. Ten
# and seven of those respectively are a literal zero, and every one of them is
# already zero: `apy_obj_alloc` (`runtime/arena.py`) clears the whole payload
# before it returns, which is a guarantee its docstring makes and which the
# C's own allocator makes too.
#
# SO ONLY THE FIELDS THAT CARRY SOMETHING ARE WRITTEN. That is not a saving of
# instructions -- it is what makes these readable: what a new function or a new
# generator actually STARTS WITH is now eight lines and six, rather than being
# spelled out among two dozen zeroes that say nothing about either.
#
# THE DEPENDENCE IS REAL AND IS NAMED HERE so that anyone who changes the
# allocator knows these two are standing on it.


# THE NUMBERS BELOW ARE THE C COMPILER'S. See `runtime/slots.py`.


def apy_fn_code_offset() -> i64:
    return 8


def apy_fn_name_offset() -> i64:
    return 24


def apy_fn_vararg_offset() -> i64:
    return 72


def apy_g_cache_offset() -> i64:
    return 32


def apy_g_step_offset() -> i64:
    return 8


def apy_func_new(code: ptr, arity: i64, name: ptr, ncells: i64,
                 ndefaults: i64, vararg: i64) -> ptr:
    """A function cell.

    THE CELL ARRAYS ARE ASKED FOR ONLY IF THEY ARE WANTED. A function with no
    closure and no defaults gets two null pointers, which is what the C does
    and what `runtime/funcs.py` already expects to find -- `apy_func_cell`
    reads `ncells` before it reads `cells`, so a null there is never
    dereferenced.

    THEY ARE ZEROED, which `calloc` did for the C and `apy_alloc_block` does
    NOT: a block off the free list holds whatever the last owner left in it.
    A stale pointer read as a closure cell is the exact shape of bug this
    runtime is arranged to make impossible, so the clearing is explicit.

    `code` IS AN ADDRESS, not a value -- it is where the function's body
    starts, and the C casts it through `uintptr_t` for the same reason the
    subset spells it `ptr`.
    """
    o: ptr = apy_obj_alloc(apy_func_kind())
    if not o:
        return o
    store(u64, u64(code), offset(o, apy_fn_code_offset()))
    store(i64, arity, offset(o, apy_fn_arity_offset()))
    store(u64, u64(name), offset(o, apy_fn_name_offset()))
    store(i64, ncells, offset(o, apy_fn_ncells_offset()))
    store(i64, ndefaults, offset(o, apy_fn_ndefaults_offset()))
    store(i32, i32(vararg), offset(o, apy_fn_vararg_offset()))
    if ncells > 0:
        store(u64, u64(apy_slots_new(ncells)),
              offset(o, apy_fn_cells_offset()))
    if ndefaults > 0:
        store(u64, u64(apy_slots_new(ndefaults)),
              offset(o, apy_fn_defaults_offset()))
    return o


def apy_slots_new(n: i64) -> ptr:
    """`n` value slots, cleared. What `calloc` was for.

    THE SIZE IS `apy_value_size()` AND NOT EIGHT, because that is the number
    the rest of the runtime indexes these arrays by -- `runtime/list_cell.py`
    uses it for `v.q.items` and this is the same kind of array.
    """
    room: i64 = n * apy_value_size()
    p: ptr = apy_alloc_block(room)
    if not p:
        return p
    at: i64 = 0
    while at < room:
        store(i64, 0, offset(p, at))
        at = at + 8
    return p


def apy_gen_new(step: ptr, nslots: i64) -> ptr:
    """A generator cell, suspended before its first resumption.

    `sent` STARTS AS None AND NOT AS ZERO, which is the one field here the
    allocator cannot supply: a generator that has never been sent anything
    still yields `None` from its first `yield`, and a zero there is not a
    value at all. It is also why this file needs `runtime/singletons.py`.

    `state` OF ZERO MEANS "AT THE TOP", and it is left to the allocator
    because that is genuinely the number -- unlike `sent`, where zero would be
    a hole rather than a beginning.
    """
    o: ptr = apy_obj_alloc(apy_gen_kind())
    if not o:
        return o
    store(u64, u64(step), offset(o, apy_g_step_offset()))
    store(i64, nslots, offset(o, apy_g_n_offset()))
    store(u64, u64(apy_none()), offset(o, apy_g_sent_offset()))
    if nslots > 0:
        store(u64, u64(apy_slots_new(nslots)),
              offset(o, apy_g_slots_offset()))
    return o



def apy_rg_start_offset() -> i64:
    return 8


def apy_rg_stop_offset() -> i64:
    return 16


def apy_rg_step_offset() -> i64:
    return 24


def apy_sup_from_offset() -> i64:
    return 8


def apy_sup_self_offset() -> i64:
    return 16


def apy_mv_src_offset() -> i64:
    return 8


def apy_mv_off_offset() -> i64:
    return 16


def apy_mv_n_offset() -> i64:
    return 24


def apy_mv_step_offset() -> i64:
    return 32


def apy_mv_wide_offset() -> i64:
    """How many bytes one element of the view is.

    ONE FOR EVERY VIEW `memoryview(b)` BUILDS, because every source this
    runtime can wrap is flat bytes; `cast` is the only thing that changes it.
    """
    return 48


def apy_mv_fmt_offset() -> i64:
    """The format character one element reads as -- `struct`'s spelling, so
    'B' for a fresh view and whatever `cast` was given after one."""
    return 56


def apy_mv_ro_offset() -> i64:
    """Whether the VIEW refuses writes, which is not the buffer's answer.

    `m.toreadonly()` hands out a view that cannot be written over a buffer
    that can, so this cannot be derived from the source's mutability flag.
    """
    return 40


def apy_mview_at_of(v: ptr, i: i64) -> i64:
    """Where the view's `i`th byte sits in the buffer underneath.

    THE STRIDE IS APPLIED HERE and nowhere else, so a reversed view reads,
    writes and converts through one piece of arithmetic.
    """
    return (load(i64, offset(v, apy_mv_off_offset()))
            + i * load(i64, offset(v, apy_mv_step_offset())))


def apy_mview_live(v: ptr) -> i64:
    """Whether the view still has its buffer.

    `m.release()` -- and the `with` block that calls it -- drops the borrow so
    the object underneath can be resized again, and every operation on the
    view from then on is refused. A RELEASED VIEW IS ONE WHOSE SOURCE IS
    NULL, which needs no field of its own and is what makes this one test.
    Answers 1 while live, 0 having raised.
    """
    if i64(load(i32, offset(v, 0))) == apy_mview_kind():
        if not ptr(load(u64, offset(v, apy_mv_src_offset()))):
            apy_raise_at(
                rodata(b"ValueError\0"),
                rodata(b"operation forbidden on released memoryview "
                       b"object\0"))
            return 0
    return 1


def apy_mview_release(v: ptr) -> ptr:
    """`m.release()` -- HAND THE BUFFER BACK.

    A view borrows its source and stops that source being resized; releasing
    drops the borrow. RELEASING TWICE IS NOT AN ERROR, which is what lets a
    `with` block close a view a program already closed.
    """
    if i64(load(i32, offset(v, 0))) != apy_mview_kind():
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute 'release'%s\0"),
            apy_kind_name_of(v), rodata(b"\0"))
    store(u64, u64(0), offset(v, apy_mv_src_offset()))
    return apy_none()


def apy_mview_readonly(v: ptr) -> ptr:
    """`m.toreadonly()` -- THE SAME WINDOW, WRITES REFUSED.

    The flag is the VIEW's and not the buffer's: this hands out one that
    refuses over a bytearray that accepts, which is the whole point of
    handing one out.
    """
    if i64(load(i32, offset(v, 0))) != apy_mview_kind():
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute 'toreadonly'%s\0"),
            apy_kind_name_of(v), rodata(b"\0"))
    if not apy_mview_live(v):
        return ptr(0)
    o: ptr = apy_obj_alloc(apy_mview_kind())
    if not o:
        return o
    store(u64, load(u64, offset(v, apy_mv_src_offset())),
          offset(o, apy_mv_src_offset()))
    store(i64, load(i64, offset(v, apy_mv_off_offset())),
          offset(o, apy_mv_off_offset()))
    store(i64, load(i64, offset(v, apy_mv_n_offset())),
          offset(o, apy_mv_n_offset()))
    store(i64, load(i64, offset(v, apy_mv_step_offset())),
          offset(o, apy_mv_step_offset()))
    store(i64, load(i64, offset(v, apy_mv_wide_offset())),
          offset(o, apy_mv_wide_offset()))
    store(i64, load(i64, offset(v, apy_mv_fmt_offset())),
          offset(o, apy_mv_fmt_offset()))
    store(i64, 1, offset(o, apy_mv_ro_offset()))
    return o


def apy_fmt_width_of(c: i64) -> i64:
    """How wide one element is for a format character, or 0 for one that is
    not a native single-character format. `struct`'s spelling and sizes.

    THE `_of` SUFFIX IS NOT DECORATION: the C has a static of the same idea
    under the plain name, and the two halves are ONE translation unit -- a
    ported helper keeping a C static's name is `conflicting types` from gcc
    rather than a second copy sitting harmlessly beside it.
    """
    if c == 66 or c == 98 or c == 99 or c == 63:
        return 1
    if c == 104 or c == 72:
        return 2
    if c == 105 or c == 73 or c == 102:
        return 4
    if c == 108 or c == 76 or c == 113 or c == 81:
        return 8
    if c == 110 or c == 78 or c == 100 or c == 80:
        return 8
    return 0


def apy_mview_cast(v: ptr, fmt: ptr) -> ptr:
    """`m.cast(fmt)` -- THE SAME BYTES, READ AS SOMETHING ELSE.

    Only a C-contiguous view can be cast, because a stride over the buffer has
    no whole elements to regroup, and the length has to divide by the new
    element size.
    """
    if i64(load(i32, offset(v, 0))) != apy_mview_kind():
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute 'cast'%s\0"),
            apy_kind_name_of(v), rodata(b"\0"))
    if not apy_mview_live(v):
        return ptr(0)
    bad: i64 = 0
    if i64(load(i32, offset(fmt, 0))) != apy_str_kind():
        bad = 1
    written: i64 = 0
    if bad == 0:
        written = load(i64, offset(fmt, apy_str_len_offset()))
        if written < 1:
            bad = 1
        if written > 2:
            bad = 1
    c: i64 = 0
    wide: i64 = 0
    if bad == 0:
        text: ptr = ptr(load(u64, offset(fmt, apy_str_ptr_offset())))
        c = i64(load(u8, offset(text, written - 1)))
        wide = apy_fmt_width_of(c)
        if wide == 0:
            bad = 1
        # `@` IS THE ONLY PREFIX, and it means the platform's own order --
        # which is the only one a cast here produces.
        if written == 2:
            if i64(load(u8, text)) != 64:
                bad = 1
    if bad:
        return apy_raise_at(
            rodata(b"ValueError\0"),
            rodata(b"memoryview: destination format must be a native "
                   b"single character format prefixed with an optional "
                   b"'@'\0"))
    if load(i64, offset(v, apy_mv_step_offset())) != 1:
        return apy_raise_at(
            rodata(b"TypeError\0"),
            rodata(b"memoryview: casts are restricted to C-contiguous "
                   b"views\0"))
    if load(i64, offset(v, apy_mv_n_offset())) % wide:
        return apy_raise_at(
            rodata(b"TypeError\0"),
            rodata(b"memoryview: length is not a multiple of itemsize\0"))
    o: ptr = apy_obj_alloc(apy_mview_kind())
    if not o:
        return o
    store(u64, load(u64, offset(v, apy_mv_src_offset())),
          offset(o, apy_mv_src_offset()))
    store(i64, load(i64, offset(v, apy_mv_off_offset())),
          offset(o, apy_mv_off_offset()))
    store(i64, load(i64, offset(v, apy_mv_n_offset())),
          offset(o, apy_mv_n_offset()))
    store(i64, 1, offset(o, apy_mv_step_offset()))
    store(i64, load(i64, offset(v, apy_mv_ro_offset())),
          offset(o, apy_mv_ro_offset()))
    store(i64, wide, offset(o, apy_mv_wide_offset()))
    store(i64, c, offset(o, apy_mv_fmt_offset()))
    return o


def apy_mview_bytes(v: ptr) -> ptr:
    """What the view shows right now, as bytes.

    `bytes(mv)` goes through here, and so does every consumer that wants a
    bytes-like argument to be one -- a translation table, a delete set.
    COPIED rather than aliased: a view can be a stride over somebody else's
    buffer, so its bytes are not contiguous and there is nothing to point at.
    """
    if not apy_mview_live(v):
        return ptr(0)
    n: i64 = load(i64, offset(v, apy_mv_n_offset()))
    src: ptr = ptr(load(u64, offset(v, apy_mv_src_offset())))
    buf: ptr = apy_str_data(src)
    out: ptr = apy_alloc_bytes(n + 1)
    if not out:
        return out
    i: i64 = 0
    while i < n:
        store(u8, load(u8, offset(buf, apy_mview_at_of(v, i))),
              offset(out, i))
        i = i + 1
    store(u8, u8(0), offset(out, n))
    return apy_bytes_literal(out, n)


def apy_t_base_offset() -> i64:
    return 16


def apy_t_mro_offset() -> i64:
    return 48


def apy_t_dict_offset() -> i64:
    return 24


# ── four more cells, and what each refuses ─────────────────────────────────


def apy_range(start: i64, stop: i64, step: i64) -> ptr:
    """`range(start, stop, step)`.

    THE THREE BOUNDS ARE MACHINE INTEGERS, not cells, which is what makes a
    range cheap: it holds three words and computes its elements, so
    `range(10 ** 6)` allocates one object rather than a million.

    A ZERO STEP IS THE ONLY REFUSAL. A range that never advances is not
    empty, it is endless, and Python raises rather than hanging.
    """
    if step == 0:
        return apy_raise_at(rodata(b"ValueError\0"),
                            rodata(b"range() arg 3 must not be zero\0"))
    o: ptr = apy_obj_alloc(apy_range_kind())
    if not o:
        return o
    store(i64, start, offset(o, apy_rg_start_offset()))
    store(i64, stop, offset(o, apy_rg_stop_offset()))
    store(i64, step, offset(o, apy_rg_step_offset()))
    return o


def apy_super(from_: ptr, self_: ptr) -> ptr:
    """What `super()` evaluates to: a class and a receiver.

    THE CLASS IS WHERE THE METHOD WAS DEFINED, not the receiver's own class,
    and that distinction is the whole reason this holds two things. Lookup
    starts at that class's BASE, so a method calling `super().m()` in a
    two-level hierarchy reaches the parent rather than itself.
    """
    if i64(load(i32, offset(from_, 0))) != apy_type_kind():
        return apy_raise_at(
            rodata(b"TypeError\0"),
            rodata(b"super(type, obj): obj must be an instance or subtype "
                   b"of type\0"))
    o: ptr = apy_obj_alloc(apy_super_kind())
    if not o:
        return o
    store(u64, u64(from_), offset(o, apy_sup_from_offset()))
    store(u64, u64(self_), offset(o, apy_sup_self_offset()))
    return o


def apy_memoryview(src: ptr) -> ptr:
    """`memoryview(b)` -- a window onto bytes that copies nothing.

    A MEMORYVIEW OF A MEMORYVIEW IS A NEW ONE OVER THE SAME BUFFER, and
    the C returned the argument itself instead -- so `memoryview(m) is m`
    was True where Python says False. Nothing observable rested on it
    beyond that identity, which is exactly why it survived: the two views
    show the same bytes either way.

    THE WINDOW IS COPIED, NOT RESET, when wrapping a view: an `m[2:5]`
    handed here must stay three bytes long. Wrapping BYTES starts at zero
    with a step of one, because a fresh view is the whole buffer and
    contiguous.
    """
    k: i64 = i64(load(i32, offset(src, 0)))
    o: ptr = apy_obj_alloc(apy_mview_kind())
    if not o:
        return o
    if k == apy_mview_kind():
        if not apy_mview_live(src):
            return ptr(0)
        store(u64, load(u64, offset(src, apy_mv_src_offset())),
              offset(o, apy_mv_src_offset()))
        # A COPY OF A READ-ONLY VIEW IS READ-ONLY, and a copy of a cast one
        # reads the same way. Both are the view's own and not the buffer's,
        # so both travel with the window.
        store(i64, load(i64, offset(src, apy_mv_ro_offset())),
              offset(o, apy_mv_ro_offset()))
        store(i64, load(i64, offset(src, apy_mv_wide_offset())),
              offset(o, apy_mv_wide_offset()))
        store(i64, load(i64, offset(src, apy_mv_fmt_offset())),
              offset(o, apy_mv_fmt_offset()))
        store(i64, load(i64, offset(src, apy_mv_off_offset())),
              offset(o, apy_mv_off_offset()))
        store(i64, load(i64, offset(src, apy_mv_n_offset())),
              offset(o, apy_mv_n_offset()))
        store(i64, load(i64, offset(src, apy_mv_step_offset())),
              offset(o, apy_mv_step_offset()))
        return o
    if k != apy_bytes_kind():
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"memoryview: a bytes-like object is required, not "
                   b"'%s'%s\0"),
            apy_kind_name_of(src), rodata(b"\0"))
    store(u64, u64(src), offset(o, apy_mv_src_offset()))
    store(i64, apy_str_byte_len(src), offset(o, apy_mv_n_offset()))
    store(i64, 1, offset(o, apy_mv_step_offset()))
    # BYTES, ONE WIDE, which is every source this runtime can wrap; `cast` is
    # the only thing that changes either.
    store(i64, 1, offset(o, apy_mv_wide_offset()))
    store(i64, 66, offset(o, apy_mv_fmt_offset()))
    return o


def apy_type_new(name: ptr, base: ptr) -> ptr:
    """A class object.

    A BASE OF None IS NO BASE, and is accepted rather than refused: the
    frontend passes None for `class C:` with no bases, so treating it as an
    error would make every plain class one.

    THE DICT IS MADE HERE AND THE REST IS LEFT ZERO. `bases`, `mro` and
    `meta` are filled in by whatever builds a more complicated class than
    this; `builtin` marks the ones this runtime provides, and a class a
    program wrote is not one.
    """
    if base:
        k: i64 = i64(load(i32, offset(base, 0)))
        if k != apy_type_kind():
            if k != apy_none_kind():
                return apy_raise_fmt(
                    rodata(b"TypeError\0"),
                    rodata(b"bases must be types, not '%s'%s\0"),
                    apy_kind_name_of(base), rodata(b"\0"))
    o: ptr = apy_obj_alloc(apy_type_kind())
    if not o:
        return o
    store(u64, u64(name), offset(o, apy_t_name_offset()))
    if base:
        if i64(load(i32, offset(base, 0))) == apy_type_kind():
            store(u64, u64(base), offset(o, apy_t_base_offset()))
    store(u64, u64(apy_dict_new(4)), offset(o, apy_t_dict_offset()))
    return o


# ── the native function cache, and the `type` class above it ───────────────


def apy_nat_kind() -> i64:
    return 17


def apy_nat_builtin_init() -> i64:
    return 23


def apy_nat_builtin_new() -> i64:
    return 24


def apy_nat_type_new() -> i64:
    return 11


def apy_nat_type_init() -> i64:
    return 12


def apy_nat_type_call() -> i64:
    return 13


def apy_nat_count() -> i64:
    """How many native selectors there are, which is how long the cache is.

    THE NUMBER IS THE C's, and the C sizes its own cache from the enum's last
    member. `APY_NAT_OBJ_ONLY` was appended for the six dunders `object`
    hands down that are NOT the receiver's, so this went from 37 to 38. A
    selector past the end of this table is a write past the end of it.
    """
    return 38


def apy_fn_native_offset() -> i64:
    return 112


def apy_native_rows() -> ptr:
    """One cell per selector, made on first use. See `apy_native_of`."""
    return reserve("apy_native_rows_ir", 512)


def apy_native_absent() -> ptr:
    """A one-element default list every builtin `__init__` shares.

    SHARED RATHER THAN PER-FUNCTION, which the C does too: the value is
    always None and nothing writes through it, so one array serves every
    selector that needs a default.
    """
    return reserve("apy_native_absent_ir", 8)


def apy_native_of(sel: i64, arity: i64, name: ptr) -> ptr:
    """The runtime's own implementation of a dunder, as a callable value.

    MADE ONCE AND REMEMBERED, because these are identities a program can
    compare: `type(x).__repr__ is type(y).__repr__` for two builtins of the
    same kind, and building a fresh cell per lookup would make that False.

    `APY_NAT_KIND` IS THE EXCEPTION and is never cached: it stands for
    whatever this kind is meant to dispatch to, so two callers asking
    for it mean different functions and must not share one.

    AND SO IS `APY_NAT_OBJ_ONLY`, for exactly the same reason: one selector
    serves the four orderings, `__subclasshook__` and `__format__`, and the
    body reads the NAME to tell them apart. Cached by selector, whichever
    was asked for first would have been handed back for all six -- so
    `object.__format__` would have answered NotImplemented because
    `object.__lt__` was built before it.

    A BUILTIN `__init__` TAKES ONE OPTIONAL ARGUMENT, which is what the
    default list is for: `int()` and `int(5)` are both calls to the same
    function, and the arity check would refuse one of them without it.
    """
    if sel != apy_nat_kind() and sel != apy_nat_obj_only():
        held: ptr = ptr(load(u64, offset(apy_native_rows(),
                                         sel * apy_value_size())))
        if held:
            return held
    o: ptr = apy_obj_alloc(apy_func_kind())
    if not o:
        return o
    store(i32, i32(sel), offset(o, apy_fn_native_offset()))
    store(i64, arity, offset(o, apy_fn_arity_offset()))
    store(u64, u64(apy_from_cstr(name)), offset(o, apy_fn_name_offset()))
    # AND `object.__init_subclass__()` IS WRITTEN WITH NO ARGUMENT AT ALL.
    # CPython's is a classmethod, so the class is already bound and the empty
    # call is the whole spelling; the class-creation path here hands the class
    # over explicitly instead, so the slot is OPTIONAL rather than absent. The
    # body answers None whichever way it is reached, and without this the two
    # spellings disagreed: `object.__init_subclass__()` was an arity error on
    # the compiled halves and None in the interpreter.
    if (sel == apy_nat_builtin_init() or sel == apy_nat_builtin_new()
            or sel == apy_nat_new() or sel == apy_nat_init_subclass()):
        store(u64, u64(apy_none()), apy_native_absent())
        store(i64, 1, offset(o, apy_fn_ndefaults_offset()))
        store(u64, u64(apy_native_absent()),
              offset(o, apy_fn_defaults_offset()))
    if sel == apy_nat_kind():
        return o
    store(u64, u64(o), offset(apy_native_rows(), sel * apy_value_size()))
    return o


def apy_type_slot() -> ptr:
    """Where the one `type` class lives once it is built."""
    return reserve("apy_type_class_ir", 8)


def apy_type_class() -> ptr:
    """The class `type` itself is, made once.

    ONE OBJECT FOR THE WHOLE PROGRAM, which is what makes `type(int) is
    type(str)` true -- both answer this, and a second copy would compare
    unequal while looking identical.

    THREE METHODS AND NO MORE. `__new__` and `__init__` are what a metaclass
    calls through `super()`, and `__call__` is what makes `C(...)`
    instantiate. Anything else a program asks of `type` falls through to the
    attribute machinery.
    """
    held: ptr = ptr(load(u64, apy_type_slot()))
    if held:
        return held
    cls: ptr = apy_type_new(apy_from_cstr(rodata(b"type\0")), ptr(0))
    if not cls:
        return cls
    store(u64, u64(cls), apy_type_slot())
    d: ptr = ptr(load(u64, offset(cls, apy_t_dict_offset())))
    apy_dict_set(d, apy_name_of(rodata(b"__new__\0")),
                 apy_native_of(apy_nat_type_new(), 4, rodata(b"__new__\0")))
    apy_dict_set(d, apy_name_of(rodata(b"__init__\0")),
                 apy_native_of(apy_nat_type_init(), 4,
                               rodata(b"__init__\0")))
    apy_dict_set(d, apy_name_of(rodata(b"__call__\0")),
                 apy_native_of(apy_nat_type_call(), 1,
                               rodata(b"__call__\0")))
    return cls


def apy_abs64_of(v: i64) -> i64:
    """`|v|`, and the one value that has no positive form.

    INT64_MIN NEGATES TO ITSELF, which is why the C casts through an unsigned
    type: the magnitude of the most negative int64 does not fit an int64, and
    every caller here wants the BITS rather than a number to do arithmetic
    on.
    """
    if v < 0:
        return i64(u64(0) - u64(v))
    return v


# ── the seven set methods, and what a value's type IS ──────────────────────


def apy_set_union(a: ptr, b: ptr) -> ptr:
    """`s.union(x)`."""
    return apy_set_method_of(rodata(b"union\0"), a, b, apy_op_union())


def apy_set_intersection(a: ptr, b: ptr) -> ptr:
    """`s.intersection(x)`."""
    return apy_set_method_of(rodata(b"intersection\0"), a, b,
                             apy_op_inter())


def apy_set_difference(a: ptr, b: ptr) -> ptr:
    """`s.difference(x)`."""
    return apy_set_method_of(rodata(b"difference\0"), a, b, 2)


def apy_set_symdiff(a: ptr, b: ptr) -> ptr:
    """`s.symmetric_difference(x)`."""
    return apy_set_method_of(rodata(b"symmetric_difference\0"), a, b,
                             apy_op_symdiff())


def apy_set_issubset(a: ptr, b: ptr) -> ptr:
    """`s.issubset(x)`."""
    return apy_set_relate_of(rodata(b"issubset\0"), a, b, 0)


def apy_set_issuperset(a: ptr, b: ptr) -> ptr:
    """`s.issuperset(x)`."""
    return apy_set_relate_of(rodata(b"issuperset\0"), a, b, 1)


def apy_set_isdisjoint(a: ptr, b: ptr) -> ptr:
    """`s.isdisjoint(x)`."""
    return apy_set_relate_of(rodata(b"isdisjoint\0"), a, b, 2)


def apy_canonical_slot() -> ptr:
    """Where the dict of program-declared classes lives.

    ONE WORD, NOT A DICT, because it starts empty and is only made when a
    program declares a class -- `apy_func_is_type` fills it, and most programs
    never do.
    """
    return reserve("apy_canonical_types_ir", 8)


def apy_type_rows() -> ptr:
    """Sixty-four pairs: a kind name, and the type object made for it."""
    return reserve("apy_type_rows_ir", 1024)


def apy_type_slot_count() -> ptr:
    """How many of those pairs are used."""
    return reserve("apy_type_count_ir", 8)


def apy_type_max() -> i64:
    """How many builtin kinds get a remembered type object."""
    return 64


def apy_type_for(v: ptr) -> ptr:
    """`type(v)` -- the object, not the name.

    FOUR ANSWERS BEFORE THE TABLE, and each is a place the type is already
    written down: an instance carries its class, a raised exception may carry
    the class it was built from, a class made by a metaclass carries that,
    and a class made by anything else is a `type`.

    A CLASS THE PROGRAM WROTE WINS OVER A BUILTIN OF THE SAME NAME, which is
    what the canonical table is for: `class int: ...` makes `type(5)` still
    answer the builtin, but `type(MyInt())` answers the program's -- and both
    are found by NAME, so the lookup order is what keeps them apart.

    THE REST ARE INTERNED, sixty-four of them, because `type(1) is type(2)`
    has to be true and a fresh cell per call would make it false. A
    sixty-fifth kind gets a fresh one -- slower and still correct, the same
    bargain `apy_name_of` makes.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_inst_kind():
        return ptr(load(u64, offset(v, apy_o_cls_offset())))
    if k == apy_exc_kind():
        cls: ptr = ptr(load(u64, offset(v, apy_e_cls_offset())))
        if cls:
            return cls
    if k == apy_type_kind():
        meta: ptr = ptr(load(u64, offset(v, apy_t_meta_offset())))
        if meta:
            return meta
        return apy_type_class()
    # A BUILTIN TYPE IS A CLASS TOO, and the canonical thunk standing for one
    # is a FUNC carrying `is_type` rather than a TYPE cell. Without this it
    # fell through to the name-keyed table below, which built a SECOND object
    # named `type` -- so `type(int) is type` was False while `print(type(int))`
    # said `<class 'type'>`, which is the worst pair of answers to have.
    if k == apy_func_kind():
        if load(i32, offset(v, apy_fn_is_type_offset())):
            return apy_type_class()
    key: ptr = apy_kind_name_of(v)
    canon: ptr = apy_canonical_slot()
    held: ptr = ptr(load(u64, canon))
    if held:
        found: ptr = apy_dict_get_or(held, apy_from_cstr(key), ptr(0))
        if found:
            return found
    used: i64 = load(i64, apy_type_slot_count())
    i: i64 = 0
    while i < used:
        row: ptr = offset(apy_type_rows(), i * 16)
        if apy_cstr_eq(ptr(load(u64, row)), key):
            return ptr(load(u64, offset(row, 8)))
        i = i + 1
    made: ptr = apy_type_new(apy_from_cstr(key), ptr(0))
    if used >= apy_type_max():
        return made
    if not made:
        return made
    at: ptr = offset(apy_type_rows(), used * 16)
    store(u64, u64(key), at)
    store(u64, u64(made), offset(at, 8))
    store(i64, used + 1, apy_type_slot_count())
    return made


def apy_type_object(v: ptr) -> ptr:
    """`type(v)` as a program spells it."""
    return apy_type_for(v)


def apy_nat_has_default() -> i64:
    return 18


def apy_typevar_slot() -> ptr:
    """Where the one `TypeVar` class lives once it is built."""
    return reserve("apy_typevar_cls_ir", 8)


def apy_typevar(name: ptr) -> ptr:
    """`T = TypeVar("T")`, and the class every one of them is an instance of.

    ONE CLASS FOR THE WHOLE PROGRAM, made on the first `TypeVar` and never
    again -- which is what makes `type(T) is type(U)` true for two of them,
    the same bargain every interned class here makes.

    `has_default()` IS A METHOD AND NOT AN ATTRIBUTE, which PEP 696 spells
    out, so the class carries one. Native, because the whole of it is asking
    whether the default slot is filled and there is no Python here to write
    it in.

    `__default__` IS SET TO None RATHER THAN LEFT OUT, so a program that
    reads it on a TypeVar written without one gets None instead of an
    AttributeError -- `apy_typevar_default` overwrites it afterwards when
    the definition had one.
    """
    cls: ptr = ptr(load(u64, apy_typevar_slot()))
    if not cls:
        cls = apy_type_new(apy_from_cstr(rodata(b"TypeVar\0")), ptr(0))
        if not cls:
            return cls
        store(u64, u64(cls), apy_typevar_slot())
        apy_dict_set(
            ptr(load(u64, offset(cls, apy_t_dict_offset()))),
            apy_name_of(rodata(b"has_default\0")),
            apy_native_of(apy_nat_has_default(), 1,
                          rodata(b"has_default\0")))
    o: ptr = apy_instance_new(cls)
    if not o:
        return o
    d: ptr = ptr(load(u64, offset(o, apy_o_dict_offset())))
    apy_dict_set(d, apy_from_cstr(rodata(b"__name__\0")), name)
    apy_dict_set(d, apy_from_cstr(rodata(b"__default__\0")), apy_none())
    return o


def apy_func_is_type(f: ptr) -> ptr:
    """Mark a function as standing for a class, and keep one per name.

    A `class` STATEMENT LOWERS TO A FUNCTION here, and this is what says so.
    Anything that is not a function passes through untouched rather than
    refusing, because the caller applies this to whatever the class body
    evaluated to.

    THE CANONICAL TABLE IS KEYED BY NAME AND THE FIRST ONE WINS, which is
    what `type(x) is C` needs: re-running a class body -- a module imported
    twice, a class defined in a loop -- would otherwise make a second class
    object that compares unequal to the first while looking identical.
    """
    if i64(load(i32, offset(f, 0))) != apy_func_kind():
        return f
    store(i32, i32(1), offset(f, apy_fn_is_type_offset()))
    canon: ptr = apy_canonical_slot()
    held: ptr = ptr(load(u64, canon))
    if not held:
        held = apy_dict_new(16)
        if not held:
            return f
        store(u64, u64(held), canon)
    name: ptr = ptr(load(u64, offset(f, apy_fn_name_offset())))
    found: ptr = apy_dict_get_or(held, name, ptr(0))
    if found:
        return found
    apy_dict_set(held, name, f)
    return f
