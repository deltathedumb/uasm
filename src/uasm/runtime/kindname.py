# The kind names, in the machine subset.
#
# WHAT EVERY TypeError SAYS. `apy_kind_name` has sat at the top of
# `uasm plugin port` since the first survey: a hundred messages name the kind
# they were handed, and none of them could move while this did not.
#
# A SPLIT, AND ONLY ONE CASE MAKES IT ONE. Every kind here answers with a
# literal or with a field read -- except an exception, whose displayed name is
# not the name it matches: a bundled module's classes are spliced under
# mangled names, so `copy.Error` carries `_asmpy_bundled_4_copy_Error` and
# `apy_exc_shown` maps it back. That mapping is a class lookup, which is
# still C, so an exception goes to the slow half and everything else stays
# here.
#
# THE LITERALS ARE PER-CASE `rodata`, not one packed table with offsets into
# it. A table would be smaller and would need a second thing to be right: the
# offsets. These are compared against nothing, so the shape that cannot drift
# is the one where each name sits at its own use.


# THE NUMBERS BELOW ARE THE C COMPILER'S. See `runtime/slots.py`.


def apy_big_kind() -> i64:
    return 16


def apy_iter_kind() -> i64:
    return 19


def apy_mview_kind() -> i64:
    return 27


def apy_range_kind() -> i64:
    return 28


def apy_super_kind() -> i64:
    return 15


def apy_view_kind() -> i64:
    return 25


def apy_o_dict_offset() -> i64:
    return 16


def apy_o_held_offset() -> i64:
    return 24


def apy_o_cls_offset() -> i64:
    return 8


def apy_t_name_offset() -> i64:
    return 8


def apy_e_name_offset() -> i64:
    return 8


def apy_it_mode_offset() -> i64:
    return 40


def apy_it_named_offset() -> i64:
    return 44


# `apy_ga_origin_offset` IS `runtime/alias.py`'S, not repeated here. Two
# identical definitions of one name in the ported runtime are one definition
# too many: the frontend reads a name defined twice as REBINDING, files the
# earlier one aside and puts the later on the value path -- which is right for
# a program and wrong for a runtime whose whole point is machine words.


def apy_vw_dict_offset() -> i64:
    return 8


def apy_vw_part_offset() -> i64:
    return 16


def apy_fn_bound_offset() -> i64:
    return 48


def apy_big_neg_offset() -> i64:
    return 24


def apy_fn_span() -> i64:
    return 152


def apy_fn_is_type_offset() -> i64:
    return 116


def apy_g_coro_offset2() -> i64:
    return 76


def apy_g_agen_offset2() -> i64:
    return 84


def apy_g_wrapper_offset2() -> i64:
    """Is this a coroutine_wrapper? See the C's `apy_coro_wrapper`."""
    return 120


def apy_s_mut_offset() -> i64:
    return 24


def apy_it_map() -> i64:
    return 1


def apy_it_filter() -> i64:
    return 2


def apy_it_enumerate() -> i64:
    return 3


def apy_it_zip() -> i64:
    return 4


# `reversed(x)`: the same index walk, counting DOWN from where the cursor was
# started. A mode and not an eagerly built list, because CPython's `reversed`
# is an ITERATOR -- no length, no indexing, walkable once.
def apy_it_rev() -> i64:
    return 5


# `iter(f, sentinel)`: call `f` on every step until it answers the sentinel.
# A mode and not a list drained at construction, because CPython's is lazy --
# the calls happen as the walk asks for them. `fn` is the callable and `src`
# the sentinel; the position slot is 0 until the sentinel arrives and 1 after.
def apy_it_call() -> i64:
    return 6


# WHAT A CURSOR IS NAMED AFTER, in its `named` slot: the KIND of what it was
# made from, or `apy_it_viewed() + part` for a dict view, or
# `apy_it_callable()` for `iter(f, sentinel)`. Both are past every kind, so
# one slot carries all three. See `apy_cursor_name` in the C.
def apy_it_viewed() -> i64:
    return 64


def apy_it_callable() -> i64:
    return 68


# A REVERSED cursor's tag, added to whichever of the above applies. IN THE TAG
# AND NOT IN THE MODE, because a length query DRAINS a cursor and resets its
# mode to plain -- so `len(reversed(xs))`, of all things, would have renamed
# it.
def apy_it_revof() -> i64:
    return 128


def apy_part_keys() -> i64:
    return 0


def apy_part_values() -> i64:
    return 1


def apy_part_items() -> i64:
    return 2


def apy_prop_classmethod() -> i64:
    return 1


def apy_prop_staticmethod() -> i64:
    return 2


def apy_name_is_dunder(name: ptr) -> i64:
    """Whether this C string is spelled `__like_this__`.

    FOUR UNDERSCORES AND AT LEAST ONE CHARACTER BETWEEN THEM, which is
    Python's own rule for the names a bound slot wears.
    """
    n: i64 = 0
    while load(u8, offset(name, n)) != u8(0):
        n = n + 1
    if n < 5:
        return 0
    if load(u8, offset(name, 0)) != u8(95):
        return 0
    if load(u8, offset(name, 1)) != u8(95):
        return 0
    if load(u8, offset(name, n - 1)) != u8(95):
        return 0
    if load(u8, offset(name, n - 2)) != u8(95):
        return 0
    return 1


def apy_kind_name_of(v: ptr) -> ptr:
    """The type name a message would use for `v`, as a C string.

    AN EXCEPTION GOES TO THE SLOW HALF and nothing else does. See the header.

    A BIG IS AN `int`, and that is deliberate rather than an omission. There
    is one integer type in Python and the width is an implementation detail
    this runtime hides -- a program that could tell `2 ** 100` from `5` by its
    type name would be seeing a seam that should not exist.

    AN INSTANCE ANSWERS WITH ITS CLASS'S NAME, which is what makes
    `type(p).__name__` say `Point` and every TypeError about a user object
    name the user's type rather than a word from this file.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_exc_kind():
        return apy_kind_name_of_slow(v)
    if k == apy_none_kind():
        return rodata(b"NoneType\0")
    if k == apy_bool_kind():
        return rodata(b"bool\0")
    if k == apy_int_kind():
        return rodata(b"int\0")
    if k == apy_big_kind():
        return rodata(b"int\0")
    if k == apy_float_kind():
        return rodata(b"float\0")
    if k == apy_complex_kind():
        return rodata(b"complex\0")
    if k == apy_str_kind():
        return rodata(b"str\0")
    if k == apy_bytes_kind():
        # ONE KIND, TWO NAMES: a bytearray is a bytes cell that admits it is
        # mutable, and `mut` is the only thing telling them apart.
        if load(i32, offset(v, apy_s_mut_offset())):
            return rodata(b"bytearray\0")
        return rodata(b"bytes\0")
    if k == apy_list_kind():
        return rodata(b"list\0")
    if k == apy_tuple_kind():
        return rodata(b"tuple\0")
    if k == apy_set_kind():
        return rodata(b"set\0")
    if k == apy_frozen_kind():
        return rodata(b"frozenset\0")
    if k == apy_dict_kind():
        # A READ-ONLY DICT IS A `mappingproxy`, which is what `C.__dict__`
        # answers and what `type()` of it says. See the C's `struct apy_obj`
        # for why it is a flag on the dict rather than a kind of its own.
        if load(i32, offset(v, apy_d_ro_offset())):
            return rodata(b"mappingproxy\0")
        return rodata(b"dict\0")
    if k == apy_inst_kind():
        return apy_str_data(ptr(load(u64, offset(
            ptr(load(u64, offset(v, apy_o_cls_offset()))),
            apy_t_name_offset()))))
    if k == apy_type_kind():
        return rodata(b"type\0")
    if k == apy_func_kind():
        # A CLASS IS A FUNCTION HERE -- `is_type` is what a `class` statement
        # sets, and a program asking `type(C).__name__` must see `type`.
        if load(i32, offset(v, apy_fn_is_type_offset())):
            return rodata(b"type\0")
        # A DESCRIPTOR GOES TO THE SLOW HALF, like an exception. `list.append`
        # is a `method_descriptor` and `list.__len__` a `wrapper_descriptor`,
        # and telling those two apart means taking the owner out of the
        # qualname and finding a value of that kind to ask about -- a string
        # slice and a lookup, which is the C's `apy_descr_owner` and not
        # something worth a second copy here.
        if load(i32, offset(v, apy_fn_descr_offset())):
            return apy_kind_name_of_slow(v)
        if load(i32, offset(v, apy_fn_builtin_offset())):
            return rodata(b"builtin_function_or_method\0")
        # A NATIVE IS THE RUNTIME'S OWN CODE, not a compiled function --
        # `[1].index` and `[1].__len__` are `builtin_function_or_method` and
        # `method-wrapper` in Python, and both answered `function` here. The
        # DUNDER is the one that differs: Python calls a bound slot a
        # method-wrapper, whatever the slot is.
        if load(i32, offset(v, apy_fn_native_offset())):
            # A DUNDER WITH A RANGE IS NOT A SLOT. `(5).__round__` is a
            # `builtin_function_or_method` in CPython because int writes the
            # method out rather than filling a slot, and an optional argument
            # is exactly what a slot cannot carry.
            if load(i64, offset(v, apy_fn_ndefaults_offset())) != 0:
                if not ptr(load(u64, offset(v, apy_fn_defaults_offset()))):
                    return rodata(b"builtin_function_or_method\0")
            named: ptr = ptr(load(u64, offset(
                ptr(load(u64, offset(v, apy_fn_name_offset()))),
                apy_str_ptr_offset())))
            if apy_name_is_dunder(named):
                # AND THE ARITY WAS ONLY STANDING IN FOR THE REAL QUESTION,
                # which is whether the type WRITES the method out or fills a
                # slot with it -- `list.__getitem__` takes exactly one
                # argument and is written out, `tuple.__getitem__` is
                # slotted, and nothing in either signature says so. The
                # answer is read out of CPython per kind and per name; see
                # `apy_kind_meth_written_of`.
                held: ptr = ptr(load(u64, offset(v, apy_fn_bound_offset())))
                if held:
                    if apy_kind_meth_written_of(named, apy_kind_bit_of(held)):
                        return rodata(b"builtin_function_or_method\0")
                    # A SLOT IS FILLED ON A VALUE, never on a TYPE. What
                    # binds to a type is a CLASSMETHOD --
                    # `list.__class_getitem__` is the only one here -- and
                    # CPython calls a bound one a
                    # `builtin_function_or_method` whatever its name looks
                    # like. The kind-and-name table cannot say so: a type
                    # has no kind bit of its own.
                    hk: i64 = i64(load(i32, offset(held, 0)))
                    if hk == apy_type_kind():
                        return rodata(b"builtin_function_or_method\0")
                    if hk == apy_func_kind():
                        if load(i32, offset(held, apy_fn_is_type_offset())):
                            return rodata(
                                b"builtin_function_or_method\0")
                return rodata(b"method-wrapper\0")
            return rodata(b"builtin_function_or_method\0")
        # A BOUND ONE IS A `method`, a type of its own in CPython:
        # `type(C().m).__name__` is `method` where `type(C.m).__name__` is
        # `function`. The receiver is the whole difference.
        if ptr(load(u64, offset(v, apy_fn_bound_offset()))):
            return rodata(b"method\0")
        return rodata(b"function\0")
    if k == apy_cell_kind():
        return rodata(b"cell\0")
    if k == apy_super_kind():
        return rodata(b"super\0")
    if k == apy_ellipsis_kind():
        return rodata(b"ellipsis\0")
    if k == apy_notimpl_kind():
        return rodata(b"NotImplementedType\0")
    if k == apy_slice_kind():
        return rodata(b"slice\0")
    if k == apy_mview_kind():
        return rodata(b"memoryview\0")
    if k == apy_range_kind():
        return rodata(b"range\0")
    if k == apy_iter_kind():
        # A CURSOR NAMES WHAT MADE IT: `map(str, xs)` is a `map`, which is
        # what `type(...).__name__` answers and what tells a reader why it is
        # lazy. A plain or reversed one is named after what it WALKS, as
        # CPython names those -- `list_iterator`, `dict_valueiterator`,
        # `list_reverseiterator`. See `apy_cursor_name` in the C, whose arms
        # these are.
        m: i64 = i64(load(i32, offset(v, apy_it_mode_offset())))
        if m == apy_it_map():
            return rodata(b"map\0")
        if m == apy_it_filter():
            return rodata(b"filter\0")
        if m == apy_it_enumerate():
            return rodata(b"enumerate\0")
        if m == apy_it_zip():
            return rodata(b"zip\0")
        tag: i64 = i64(load(i32, offset(v, apy_it_named_offset())))
        rev: i64 = 0
        if tag >= apy_it_revof():
            rev = 1
            tag = tag - apy_it_revof()
        if tag == apy_it_viewed() + apy_part_keys():
            if rev:
                return rodata(b"dict_reversekeyiterator\0")
            return rodata(b"dict_keyiterator\0")
        if tag == apy_it_viewed() + apy_part_values():
            if rev:
                return rodata(b"dict_reversevalueiterator\0")
            return rodata(b"dict_valueiterator\0")
        if tag == apy_it_viewed() + apy_part_items():
            if rev:
                return rodata(b"dict_reverseitemiterator\0")
            return rodata(b"dict_itemiterator\0")
        if tag == apy_it_callable():
            return rodata(b"callable_iterator\0")
        if tag == apy_list_kind():
            if rev:
                return rodata(b"list_reverseiterator\0")
            return rodata(b"list_iterator\0")
        if tag == apy_dict_kind():
            if rev:
                return rodata(b"dict_reversekeyiterator\0")
            return rodata(b"dict_keyiterator\0")
        # A RANGE REVERSED IS STILL A RANGE ITERATOR: CPython answers
        # `range_iterator` both ways, because `reversed(r)` hands back a walk
        # over the range with its step negated rather than a wrapper.
        if tag == apy_range_kind():
            return rodata(b"range_iterator\0")
        if tag == apy_set_kind():
            return rodata(b"set_iterator\0")
        if tag == apy_frozen_kind():
            return rodata(b"set_iterator\0")
        if rev:
            return rodata(b"reversed\0")
        if tag == apy_tuple_kind():
            return rodata(b"tuple_iterator\0")
        if tag == apy_mview_kind():
            return rodata(b"memory_iterator\0")
        src: ptr = ptr(load(u64, offset(v, apy_it_src_offset())))
        if tag == apy_bytes_kind():
            # A bytearray and a bytes share the kind, so the mutable flag
            # decides -- read from the source while it is still there.
            if src:
                if i64(load(i32, offset(src, 0))) == apy_bytes_kind():
                    if load(i32, offset(src, apy_s_mut_offset())):
                        return rodata(b"bytearray_iterator\0")
            return rodata(b"bytes_iterator\0")
        if tag == apy_str_kind():
            # `str_ascii_iterator` IS A NAME OF ITS OWN IN CPYTHON, which
            # records the width on the string. A str here is UTF-8 bytes, so
            # "every character is one byte" is the same question as "no byte
            # has its high bit set" -- asked here, where nothing reads the
            # answer in a loop.
            if src:
                if i64(load(i32, offset(src, 0))) == apy_str_kind():
                    text: ptr = apy_str_data(src)
                    wide: i64 = apy_str_byte_len(src)
                    seen: i64 = 0
                    while seen < wide:
                        if i64(load(u8, offset(text, seen))) >= 128:
                            return rodata(b"str_iterator\0")
                        seen = seen + 1
            return rodata(b"str_ascii_iterator\0")
        return rodata(b"iterator\0")
    if k == apy_view_kind():
        p: i64 = i64(load(i32, offset(v, apy_vw_part_offset())))
        if p == apy_part_keys():
            return rodata(b"dict_keys\0")
        if p == apy_part_values():
            return rodata(b"dict_values\0")
        return rodata(b"dict_items\0")
    if k == apy_prop_kind():
        d: i64 = i64(load(i32, offset(v, apy_prop_kind_offset())))
        if d == apy_prop_classmethod():
            return rodata(b"classmethod\0")
        if d == apy_prop_staticmethod():
            return rodata(b"staticmethod\0")
        return rodata(b"property\0")
    if k == apy_gen_kind():
        # ALL FOUR SHARE EVERY FIELD and only the name differs, which a
        # program reads to tell them apart: `async def` with `yield` is an
        # async generator, which is neither of the other two, and what
        # `c.__await__()` answers is a fourth.
        #
        if load(i32, offset(v, apy_g_wrapper_offset2())):
            return rodata(b"coroutine_wrapper\0")
        if load(i32, offset(v, apy_g_agen_offset2())):
            return rodata(b"async_generator\0")
        if load(i32, offset(v, apy_g_coro_offset2())):
            return rodata(b"coroutine\0")
        return rodata(b"generator\0")
    if k == apy_alias_kind():
        # A UNION IS NOT A GENERIC ALIAS to a program that asks. `int | str`
        # is built on the `Union` form, and `type(...).__name__` is how a
        # program tells the two apart.
        if i64(load(i32, offset(ptr(load(u64, offset(
                v, apy_ga_origin_offset()))), 0))) == apy_inst_kind():
            return rodata(b"typing.Union\0")
        return rodata(b"types.GenericAlias\0")
    # THE C'S `default` IS `str` AND NOT A REFUSAL, which looks arbitrary and
    # is load-bearing: several cells that never reach a message share the str
    # arm's layout, and answering something is what keeps a stray kind from
    # printing a null pointer.
    return rodata(b"str\0")


# ── two lookups that answer with a pointer ─────────────────────────────────


def apy_str_bytes(s: ptr) -> ptr:
    """The bytes behind a str, bytes or int, for a native call.

    AN INT IS ALLOWED AND MEANS THE ADDRESS ITSELF, because C's own rule for a
    pointer parameter is that a null pointer constant fits it -- and
    `CreateDirectoryA(path, 0)` is the ordinary way to pass "no security
    descriptor". Refusing it would make every native call with a NULL argument
    a TypeError, which is not what the C being declared says.

    THE MESSAGE NAMES THE KIND because `objects_host` words the same refusal
    for the interpreter, and a program that prints the exception would
    otherwise get different text from the two paths -- which the corpus
    compares.
    """
    k: i64 = i64(load(i32, offset(s, 0)))
    if k == apy_int_kind():
        return ptr(apy_int_payload(s))
    if k != apy_str_kind():
        if k != apy_bytes_kind():
            return apy_raise_fmt(
                rodata(b"TypeError\0"),
                rodata(b"a pointer argument must be str, bytes or int, "
                       b"not %s%s\0"),
                apy_kind_name_of(s), rodata(b"\0"))
    return apy_str_data(s)


def apy_type_name(v: ptr) -> ptr:
    """`type(x).__name__`, which the frontend fuses into one call.

    THE CLASS'S OWN NAME VALUE, NOT A FRESH COPY, for an instance: two
    instances of one class must give `type(a).__name__ is type(b).__name__`,
    as they do in CPython, and building a string here would give two.

    `type(C).__name__` IS THE METACLASS'S NAME when one made the class, and
    an ordinary class has no metaclass recorded and is a `type`.

    EVERYTHING ELSE GOES THROUGH THE KIND NAME, which is why this could not
    move until `runtime/kindname.py` did -- and why an exception's answer here
    is the DISPLAYED name rather than the internal one: that split is
    `apy_kind_name_of`'s to make, not this function's.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_inst_kind():
        return ptr(load(u64, offset(
            ptr(load(u64, offset(v, apy_o_cls_offset()))),
            apy_t_name_offset())))
    if k == apy_type_kind():
        meta: ptr = ptr(load(u64, offset(v, apy_t_meta_offset())))
        if meta:
            return ptr(load(u64, offset(meta, apy_t_name_offset())))
        return apy_from_cstr(rodata(b"type\0"))
    # THE BARE NAME. A builtin kind is named the way CPython names it in a
    # message -- `types.GenericAlias`, `typing.Union` -- and `__name__` is the
    # last component of that, with the module belonging in `__module__`. The
    # kind name is a static literal, so the tail of a dotted one is a
    # NUL-terminated string of its own and needs no copy.
    nm: ptr = apy_kind_name_of(v)
    i: i64 = 0
    at: i64 = -1
    while load(u8, offset(nm, i)):
        if i64(load(u8, offset(nm, i))) == 46:
            at = i
        i = i + 1
    if at >= 0:
        return apy_from_cstr(offset(nm, at + 1))
    return apy_from_cstr(nm)
