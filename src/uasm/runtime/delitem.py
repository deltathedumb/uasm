# `del d[k]`, `del xs[i]`, `del xs[1:3]`, and `del obj.name`.
#
# TWO CONTAINERS AND FOUR FAILURE MODES, and CPython's own wording for each.
# The shapes look alike and are not: a dict deletes by KEY and reports the key
# it could not find, a list deletes by INDEX and reports a range, and a slice
# deletes a SPAN and is not an index at all.
#
# THE SLICE CASE IS CHECKED FIRST FOR THAT LAST REASON. Falling through to the
# index path asked `apy_index_arg` for an integer, got a slice, and reported
# an IndexError about a subscript the program never wrote.
#
# A SLICE DELETES A SET OF POSITIONS AND NOT A SPAN, which is what makes
# `del xs[::2]` no harder than `del xs[1:3]`: one pass copies the survivors
# down over the gaps, and `apy_del_run` turns a negative step round first so
# that pass only ever moves forwards.
#
# INSERTION ORDER IS PRESERVED BY SHIFTING, not by swapping the last entry
# into the hole. Dict order has been part of the language since 3.7, so the
# swap would be a WRONG ANSWER rather than a faster one -- and the list has
# never been allowed to reorder on delete at all.


def apy_delitem(seq: ptr, key: ptr) -> ptr:
    """`del seq[key]`. None on success, zero with an error set on failure."""
    if i64(load(i32, offset(seq, 0))) == apy_inst_kind():
        # `del obj[k]` IS `obj.__delitem__(k)`. Never dispatched before, so a
        # class that wrote one had it ignored and the delete was reported as
        # unsupported -- a wrong answer about the class's own method.
        r: ptr = apy_method1_of(seq, rodata(b"__delitem__\0"), key)
        if r:
            return r
        if apy_err_kind():
            return ptr(0)
        # A CLASS THAT EXTENDS A BUILTIN deletes from the one it carries.
        held: ptr = apy_inst_held_of(seq)
        if held:
            return apy_delitem(held, key)
    if i64(load(i32, offset(seq, 0))) == apy_dict_kind():
        # A mappingproxy IS READ-ONLY TO A PROGRAM, and CPython words this
        # one differently from the assignment it refuses beside it.
        if load(i32, offset(seq, apy_d_ro_offset())):
            return apy_raise_fmt(
                rodata(b"TypeError\0"),
                rodata(b"'%s' object does not support item deletion%s\0"),
                apy_kind_name_of(seq), rodata(b"\0"))
        bad: ptr = apy_unhashable_of(key)
        if bad:
            return apy_raise_fmt(rodata(b"TypeError\0"),
                                 rodata(b"unhashable type: '%s'%s\0"),
                                 bad, rodata(b"\0"))
        i: i64 = apy_dict_find_of(seq, key)
        if i < 0:
            shown: ptr = apy_repr(key)
            if not shown:
                return shown
            # THE BYTES AND NOT THE CELL. `apy_raise_fmt` copies its two
            # arguments with `apy_cstr_into`, so both are C strings -- handing
            # it a str cell copied from the header, whose first byte is the
            # kind tag's low byte, and the KeyError came out empty.
            return apy_raise_fmt(
                rodata(b"KeyError\0"), rodata(b"%s%s\0"),
                ptr(load(u64, offset(shown, apy_str_ptr_offset()))),
                rodata(b"\0"))
        keys: ptr = ptr(load(u64, offset(seq, apy_d_keys_offset())))
        vals: ptr = ptr(load(u64, offset(seq, apy_d_vals_offset())))
        n: i64 = load(i64, offset(seq, apy_d_n_offset()))
        while i + 1 < n:
            store(u64, load(u64, offset(keys, (i + 1) * apy_value_size())),
                  offset(keys, i * apy_value_size()))
            store(u64, load(u64, offset(vals, (i + 1) * apy_value_size())),
                  offset(vals, i * apy_value_size()))
            i = i + 1
        store(i64, n - 1, offset(seq, apy_d_n_offset()))
        return apy_none()
    # A BYTEARRAY DELETES TOO, and its buffer is BYTES rather than values --
    # which is the whole reason it is a case of its own rather than a wider
    # test above. `mut` is what separates it from bytes, which does not.
    if i64(load(i32, offset(seq, 0))) == apy_bytes_kind():
        if load(i32, offset(seq, apy_s_mut_offset())):
            return apy_del_bytes(seq, key)
    # A VIEW REFUSES IN ITS OWN WORDS. Deleting from one is not "this kind
    # has no such operation" -- a memoryview HAS `__delitem__` and it always
    # refuses, because a window onto a buffer cannot make the buffer shorter.
    if i64(load(i32, offset(seq, 0))) == apy_mview_kind():
        return apy_raise_at(
            rodata(b"TypeError\0"),
            rodata(b"cannot delete memory\0"))
    if i64(load(i32, offset(seq, 0))) != apy_list_kind():
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"'%s' object doesn't support item deletion%s\0"),
            apy_kind_name_of(seq), rodata(b"\0"))
    items: ptr = ptr(load(u64, offset(seq, apy_q_items_offset())))
    have: i64 = load(i64, offset(seq, apy_q_n_offset()))
    if key:
        if i64(load(i32, offset(key, 0))) == apy_slice_kind():
            return apy_del_span(seq, key)
    # THE SUBSCRIPT'S OWN WORDING. `del xs[1.0]` is a complaint about the
    # subscript, and CPython words it as one: `list indices must be integers
    # or slices, not float`, and not the index conversion's `'float' object
    # cannot be interpreted as an integer`. AN INSTANCE IS LEFT ALONE, since
    # one with `__index__` is a valid subscript and the conversion asks.
    if apy_is_int_like_of(key) == 0:
        if i64(load(i32, offset(key, 0))) != apy_inst_kind():
            return apy_raise_fmt(
                rodata(b"TypeError\0"),
                rodata(b"list indices must be integers or slices, "
                       b"not %s%s\0"),
                apy_kind_name_of(key), rodata(b"\0"))
    slot: ptr = apy_delitem_slot()
    if not apy_index_arg_of(key, slot, apy_idx_sub()):
        return ptr(0)
    at: i64 = load(i64, slot)
    if at < 0:
        at = at + have
    if at < 0 or at >= have:
        return apy_raise_at(rodata(b"IndexError\0"),
                            rodata(b"list assignment index out of range\0"))
    while at + 1 < have:
        store(u64, load(u64, offset(items, (at + 1) * apy_value_size())),
              offset(items, at * apy_value_size()))
        at = at + 1
    store(i64, have - 1, offset(seq, apy_q_n_offset()))
    return apy_none()


def apy_delitem_slot() -> ptr:
    """One word, for the index `apy_index_arg_of` writes back."""
    return reserve("apy_delitem_slot_ir", 8)


def apy_del_run(key: ptr, n: i64, out: ptr) -> i64:
    """The ASCENDING run of positions a slice deletes: first, stride, count.

    Three words written to `out`; 0 with an error set if the slice refuses.

    A NEGATIVE STEP WALKS THE SAME SET BACKWARDS, and a delete is about the
    SET and not the order -- `del xs[::-2]` and `del xs[::2]` take the same
    three positions out of six, starting from opposite ends. Turning it round
    here is what lets the compaction that follows only ever move forwards.

    THE COUNT IS ARITHMETIC AND NOT A WALK, because the compaction needs to
    know where the NEXT position is while it is copying, and recomputing it
    from `first + taken * stride` is one multiply against a second pass.
    """
    bounds: ptr = apy_slice_indices(key, apy_from_int(n))
    if not bounds:
        return 0
    b: ptr = ptr(load(u64, offset(bounds, apy_q_items_offset())))
    start: i64 = apy_int_payload(ptr(load(u64, b)))
    stop: i64 = apy_int_payload(ptr(load(u64, offset(b, apy_value_size()))))
    step: i64 = apy_int_payload(
        ptr(load(u64, offset(b, 2 * apy_value_size()))))
    count: i64 = 0
    if step > 0:
        if stop > start:
            count = (stop - start + step - 1) // step
    else:
        if stop < start:
            count = (stop - start + step + 1) // step
    # PAST THE END WHEN NOTHING MATCHES, so the copy loop's position test is
    # false at every step without a count test of its own.
    first: i64 = n
    stride: i64 = 1
    if count > 0:
        if step > 0:
            first = start
            stride = step
        else:
            first = start + (count - 1) * step
            stride = -step
    store(i64, first, out)
    store(i64, stride, offset(out, 8))
    store(i64, count, offset(out, 16))
    return 1


def apy_del_span(seq: ptr, key: ptr) -> ptr:
    """`del xs[1:3]` and `del xs[::2]` -- a slice removes a SET of positions.

    ONE PASS THAT COMPACTS, which is what makes the strided case no harder
    than the contiguous one: the survivors are copied down over the gaps as
    they are met, and the length is what the copy ended at.
    """
    n: i64 = load(i64, offset(seq, apy_q_n_offset()))
    run: ptr = alloca(24)
    if not apy_del_run(key, n, run):
        return ptr(0)
    first: i64 = load(i64, run)
    stride: i64 = load(i64, offset(run, 8))
    count: i64 = load(i64, offset(run, 16))
    items: ptr = ptr(load(u64, offset(seq, apy_q_items_offset())))
    at: i64 = first
    taken: i64 = 0
    frm: i64 = 0
    to: i64 = 0
    while frm < n:
        drop: i64 = 0
        if taken < count:
            if frm == at:
                drop = 1
        if drop:
            taken = taken + 1
            at = first + taken * stride
        else:
            store(u64, load(u64, offset(items, frm * apy_value_size())),
                  offset(items, to * apy_value_size()))
            to = to + 1
        frm = frm + 1
    store(i64, to, offset(seq, apy_q_n_offset()))
    return apy_none()


def apy_del_bytes(seq: ptr, key: ptr) -> ptr:
    """`del ba[i]` and `del ba[a:b:c]`, on a bytearray.

    THE SAME TWO SHAPES AS A LIST over a different buffer, and the messages
    name `bytearray` rather than `list` because CPython's do.

    THE BUFFER KEEPS ITS TERMINATOR. Nothing here reads past `n`, but the
    cell is shared with str, whose bytes are handed to C as a string -- so
    shrinking writes the NUL rather than leaving the old byte behind.
    """
    p: ptr = ptr(load(u64, offset(seq, apy_str_ptr_offset())))
    n: i64 = load(i64, offset(seq, apy_str_len_offset()))
    if key:
        if i64(load(i32, offset(key, 0))) == apy_slice_kind():
            run: ptr = alloca(24)
            if not apy_del_run(key, n, run):
                return ptr(0)
            first: i64 = load(i64, run)
            stride: i64 = load(i64, offset(run, 8))
            count: i64 = load(i64, offset(run, 16))
            spot: i64 = first
            taken: i64 = 0
            frm: i64 = 0
            to: i64 = 0
            while frm < n:
                drop: i64 = 0
                if taken < count:
                    if frm == spot:
                        drop = 1
                if drop:
                    taken = taken + 1
                    spot = first + taken * stride
                else:
                    store(u8, load(u8, offset(p, frm)), offset(p, to))
                    to = to + 1
                frm = frm + 1
            store(i64, to, offset(seq, apy_str_len_offset()))
            store(u8, u8(0), offset(p, to))
            return apy_none()
    if not apy_is_int_like_of(key):
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"bytearray indices must be integers or slices, "
                   b"not %s%s\0"),
            apy_kind_name_of(key), rodata(b"\0"))
    slot: ptr = apy_delitem_slot()
    if not apy_index_arg_of(key, slot, apy_idx_sub()):
        return ptr(0)
    at: i64 = load(i64, slot)
    if at < 0:
        at = at + n
    if at < 0 or at >= n:
        return apy_raise_at(rodata(b"IndexError\0"),
                            rodata(b"bytearray index out of range\0"))
    while at + 1 < n:
        store(u8, load(u8, offset(p, at + 1)), offset(p, at))
        at = at + 1
    store(i64, n - 1, offset(seq, apy_str_len_offset()))
    store(u8, u8(0), offset(p, n - 1))
    return apy_none()


def apy_default_delattr(obj: ptr, name: ptr) -> ptr:
    """`del obj.name` -- `object.__delattr__`, after the hooks have declined.

    ONE MESSAGE FOR TWO FAILURES, deliberately: an object with no instance
    dict at all and an object whose dict does not hold the name are the same
    thing from the program's side, and CPython reports them alike.

    THE LOOKUP IS SEPARATE FROM THE DELETE because `apy_delitem` reports a
    KeyError and this owes an AttributeError. Letting the delete raise its own
    message would name a dict the program never wrote.
    """
    if i64(load(i32, offset(obj, 0))) != apy_inst_kind():
        return apy_no_such_attr(obj, name)
    d: ptr = ptr(load(u64, offset(obj, apy_o_dict_offset())))
    if apy_dict_find_of(d, name) < 0:
        return apy_no_such_attr(obj, name)
    return apy_delitem(d, name)


def apy_no_such_attr(obj: ptr, name: ptr) -> ptr:
    """`'C' object has no attribute 'x'`.

    THE NAME'S BYTES AND NOT ITS CELL: `apy_raise_fmt` copies both arguments
    with `apy_cstr_into`, so it wants C strings.
    """
    return apy_raise_fmt(
        rodata(b"AttributeError\0"),
        rodata(b"'%s' object has no attribute '%s'\0"),
        apy_kind_name_of(obj),
        ptr(load(u64, offset(name, apy_str_ptr_offset()))))


def apy_delattr(obj: ptr, name: ptr) -> ptr:
    """`del obj.name`, with the hooks that may take it first.

    THREE CHANCES BEFORE THE INSTANCE DICT, in CPython's order:

      `__delattr__` on the class takes EVERY delete, whatever the name --
      the same rule `__setattr__` has, and the reason it is asked before
      anything is looked up at all.

      A DATA DESCRIPTOR TAKES THE DELETE exactly as it takes the write.
      `__delete__` is the third of the three, and a property or a user
      descriptor defining it never reaches the instance dict. Without this,
      `del c.d` on a descriptor attribute looked in the dict, found nothing
      -- a descriptor never puts anything there -- and reported an attribute
      the class plainly has.

      A PROPERTY WITH NO DELETER REFUSES rather than falling through, for
      the same reason: falling through would report the attribute missing.

    ANYTHING ELSE IS `object.__delattr__`.
    """
    if i64(load(i32, offset(obj, 0))) == apy_inst_kind():
        cls: ptr = ptr(load(u64, offset(obj, apy_o_cls_offset())))
        hook: ptr = apy_class_find_of(cls, apy_name_of(rodata(b"__delattr__\0")))
        if hook:
            argv: ptr = alloca(8)
            store(u64, u64(name), argv)
            return apy_call(apy_bind_of(hook, obj), argv, 1)
        found: ptr = apy_class_find_of(cls, name)
        if found:
            if apy_is_data_descriptor_of(found):
                one: ptr = alloca(8)
                store(u64, u64(obj), one)
                if i64(load(i32, offset(found, 0))) == apy_prop_kind():
                    dele: ptr = ptr(load(
                        u64, offset(found, apy_prop_del_offset())))
                    if not dele:
                        return apy_raise_at(
                            rodata(b"AttributeError\0"),
                            rodata(b"can't delete attribute\0"))
                    return apy_call(dele, one, 1)
                m: ptr = apy_class_find_of(
                    ptr(load(u64, offset(found, apy_o_cls_offset()))),
                    apy_name_of(rodata(b"__delete__\0")))
                if m:
                    return apy_call(apy_bind_of(m, found), one, 1)
    return apy_default_delattr(obj, name)
