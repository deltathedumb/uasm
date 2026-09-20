# `join` and the two partitions, in the machine subset.
#
# STAGE 5 OF docs/INERT-RUNTIME.md. Both of these READ a sequence or BUILD
# one, which is what `runtime/str_split.py` established was possible; what is
# new here is reading a list's items directly rather than only pushing onto
# one.
#
# NO ASCII GATE for either, and it is the `str_split.py` argument both times:
# `join` never inspects a character, it concatenates whole strings; and
# `partition` cuts at a separator, which is a character boundary. Neither asks
# a question that bytes and characters could answer differently.
#
# ── join's fast path is narrow on purpose ──────────────────────────────────
#
# The C's `join` takes ANY iterable -- it drains a generator first, accepts a
# set or a dict, and words two different TypeErrors about what it was given.
# Reaching all of that needs `apy_iterable`, `apy_key_at`, `apy_is_seq` and
# `apy_kind_name`, none of which the subset can name.
#
# WHAT IS LEFT IS THE CASE PROGRAMS WRITE: a plain list of plain strings. The
# check for it is the first pass, which has to walk the items anyway to add up
# their lengths -- so declining costs a comparison per item and no second
# walk. Anything else, including a list with one non-string in it, goes back
# to the C with nothing built and nothing half-written.


def apy_bytes_move(dst: ptr, at: i64, src: ptr, n: i64) -> i64:
    """Copy `n` bytes into `dst` at `at`; answer where the next one goes.

    `memcpy` WRITTEN OUT, and answering the new offset rather than nothing so
    the caller does not keep a running total the loop already knows. A backend
    with a real `memcpy` is free to recognise the shape.
    """
    i: i64 = 0
    while i < n:
        store(u8, load(u8, offset(src, i)), offset(dst, at + i))
        i = i + 1
    return at + n


def apy_str_item_at(seq: ptr, i: i64) -> ptr:
    """Item `i` of a list or tuple, without the index cell `apy_getitem`
    wants.

    NO BOUNDS CHECK, because both callers here walk `0 .. v.q.n` themselves.
    It is not exported and nothing else may use it.
    """
    return ptr(load(u64, offset(ptr(load(u64, offset(seq,
               apy_q_items_offset()))), i * apy_value_size())))


def apy_join_total(sep: ptr, parts: ptr, n: i64) -> i64:
    """The joined length, or -1 if any item is not a plain string.

    ONE WALK ANSWERS BOTH QUESTIONS. The length has to be known before the
    buffer is asked for, and every item has to be checked before a byte is
    written; doing them together is what makes the decline free.

    THE SEPARATORS ARE `n - 1`, NOT `n` -- the one place an off-by-one here
    would be a buffer overrun rather than a wrong string.
    """
    total: i64 = 0
    i: i64 = 0
    while i < n:
        it: ptr = apy_str_item_at(parts, i)
        if not apy_is_str(it):
            return -1
        total = total + apy_str_byte_len(it)
        i = i + 1
    if n > 1:
        total = total + apy_str_byte_len(sep) * (n - 1)
    return total


def apy_str_join(sep: ptr, parts: ptr) -> ptr:
    """`sep.join(parts)` for a plain separator and a plain list of strings."""
    if not apy_is_str(sep):
        return apy_str_join_slow(sep, parts)
    if i64(load(i32, offset(parts, 0))) != apy_list_kind():
        return apy_str_join_slow(sep, parts)
    n: i64 = load(i64, offset(parts, apy_q_n_offset()))
    total: i64 = apy_join_total(sep, parts, n)
    if total < 0:
        return apy_str_join_slow(sep, parts)
    # ONE ELEMENT IS NOTHING TO JOIN, and CPython hands that element
    # straight back: both `PyUnicode_Join` and the `bytes_join` every
    # bytes-like shares test `seqlen == 1` and the element's exact type
    # before they write a byte, so the SEPARATOR is never even looked at and
    # `"-".join([s]) is s` is True. Nothing is copied because nothing was
    # going to be inserted.
    #
    # AFTER THE MEASURING WALK AND NOT BEFORE IT. `apy_join_total` is what
    # refuses a list holding something that is not a plain string, so
    # answering `n == 1` ahead of it would hand `"-".join([5])` back the 5
    # instead of letting the C word its TypeError.
    #
    # THE EXACT-TYPE TEST IS ALREADY SPENT: that walk admits only an exactly
    # str element and the gate above only an exactly str separator, so
    # neither can be a bytearray or a subclass instance here. The bytes
    # family reaches the C, which asks `apy_str_may_return_self` of both --
    # `bytearray(b"-").join([b"abc"])` must answer a bytearray, and an
    # element that is itself a bytearray is a buffer the caller can write
    # into afterwards.
    if n == 1:
        return apy_str_item_at(parts, 0)
    buf: ptr = apy_alloc_bytes(total + 1)
    if not buf:
        return apy_str_join_slow(sep, parts)
    sn: i64 = apy_str_byte_len(sep)
    sp: ptr = apy_str_data(sep)
    out: i64 = 0
    i: i64 = 0
    while i < n:
        if i > 0:
            out = apy_bytes_move(buf, out, sp, sn)
        it: ptr = apy_str_item_at(parts, i)
        out = apy_bytes_move(buf, out, apy_str_data(it),
                             apy_str_byte_len(it))
        i = i + 1
    store(u8, u8(0), offset(buf, total))
    return apy_from_bytes(buf, total)


# ── partition and rpartition ───────────────────────────────────────────────
#
# A THREE-TUPLE, ALWAYS, which is what makes these easier than `split`: the
# size is known before anything is searched, so `apy_tuple_new(3)` is exact
# and nothing can grow.


def apy_str_empty_like(s: ptr) -> ptr:
    """An empty string cell.

    BUILT FROM `s`'s OWN POINTER with a length of zero, which copies no bytes
    and dereferences nothing -- the subset has no null to pass and no string
    literal to point at, and borrowing a pointer that is certainly valid is
    cheaper than either.
    """
    return apy_str_copy_bytes(apy_str_data(s), 0)


def apy_str_partition(s: ptr, sep: ptr) -> ptr:
    """`s.partition(sep)` -- before, the separator, after.

    THE NO-MATCH ANSWER IS NOT THREE EMPTY STRINGS. Python gives the whole
    receiver FIRST and two empties after it, so `'abc'.partition('x')[0]` is
    `'abc'`. `rpartition` puts the receiver LAST for the same reason: each
    keeps the text on the side it did not search past.

    AND IT IS THE RECEIVER ITSELF THAT GOES INTO THE SLOT, which is the one
    place this file puts a cell somewhere the program keeps. That is only
    safe because `apy_str_split_ok` admits an exactly-str receiver: a
    BYTEARRAY in a tuple slot is still two live names for one writable
    buffer, and the piece would change under the program at the next
    `ba[0] = ...`. CPython reaches a different function for a mutable
    receiver for exactly this reason -- `stringlib_partition` hands slot 0
    back with a plain `Py_INCREF` that is right only for the immutable kinds,
    and the mutable instantiation copies -- so
    `bytearray(b"abc").partition(b"z")[0] is` it is False there. A bytearray
    declines to the C, which copies before it fills the slot.
    """
    m: i64 = apy_str_split_ok(s, sep)
    if m < 0:
        return apy_str_partition_slow(s, sep)
    n: i64 = apy_str_byte_len(s)
    out: ptr = apy_tuple_new(3)
    if not out:
        return apy_str_partition_slow(s, sep)
    at: i64 = apy_str_find_at(s, sep, 0, n)
    if at < 0:
        apy_seq_push(out, s)
        apy_seq_push(out, apy_str_empty_like(s))
        apy_seq_push(out, apy_str_empty_like(s))
        return out
    apy_seq_push(out, apy_str_slice_new(s, 0, at))
    apy_seq_push(out, sep)
    apy_seq_push(out, apy_str_slice_new(s, at + m, n))
    return out


def apy_str_rpartition(s: ptr, sep: ptr) -> ptr:
    """`s.rpartition(sep)` -- the last occurrence, not the first."""
    m: i64 = apy_str_split_ok(s, sep)
    if m < 0:
        return apy_str_rpartition_slow(s, sep)
    n: i64 = apy_str_byte_len(s)
    out: ptr = apy_tuple_new(3)
    if not out:
        return apy_str_rpartition_slow(s, sep)
    at: i64 = apy_str_rfind_at(s, sep, 0, n)
    if at < 0:
        apy_seq_push(out, apy_str_empty_like(s))
        apy_seq_push(out, apy_str_empty_like(s))
        apy_seq_push(out, s)
        return out
    apy_seq_push(out, apy_str_slice_new(s, 0, at))
    apy_seq_push(out, sep)
    apy_seq_push(out, apy_str_slice_new(s, at + m, n))
    return out
