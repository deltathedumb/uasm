# The set, frozenset and dict constructors, in the machine subset.
#
# STAGE 5 OF docs/INERT-RUNTIME.md. `uasm plugin port` named `apy_seq_new` and
# `apy_alloc` as two of the walls in front of the remaining runtime; these are
# the functions behind them that need nothing else.
#
# `runtime/list_cell.py` PREDICTED THIS. Its `apy_seq_alloc` says it is "not
# exported, and deliberately: a helper that displaced `apy_seq_new` would take
# the tuple and set constructors with it before either was ported". Both are
# ported now, so the helper it was holding back for is simply called.
#
# THE TWO SET CONSTRUCTORS ARE THE SAME FUNCTION WITH A DIFFERENT TAG, exactly
# as list and tuple are. A frozenset is not a separate structure -- it is a set
# that refuses to be added to, and the refusal lives in the methods rather than
# in the layout.
#
# A DICT IS NOT A SEQUENCE and so does not go through `apy_seq_alloc`: it
# carries TWO parallel arrays where a sequence carries one, and its count and
# capacity live at their own offsets. Sharing the allocator would mean
# pretending one arm is the other.


# THE NUMBERS BELOW ARE THE C COMPILER'S. See `runtime/slots.py`.


def apy_set_kind() -> i64:
    return 9


def apy_frozen_kind() -> i64:
    return 10


def apy_d_keys_offset() -> i64:
    return 8


def apy_d_vals_offset() -> i64:
    return 16


def apy_d_n_offset() -> i64:
    return 24


def apy_d_cap_offset() -> i64:
    return 32


def apy_set_new(cap: i64) -> ptr:
    """An empty set with room for `cap`."""
    return apy_seq_alloc(apy_set_kind(), cap)


def apy_frozenset_new(cap: i64) -> ptr:
    """An empty frozenset with room for `cap`."""
    return apy_seq_alloc(apy_frozen_kind(), cap)


def apy_dict_new(cap: i64) -> ptr:
    """An empty dict with room for `cap` pairs.

    TWO BLOCKS, NOT ONE OF PAIRS. Keys and values are separate arrays, which
    is what lets `apy_dict_keys` hand back a run of memory rather than walking
    a stride -- and it is why a failed second allocation has to be checked
    even though the first succeeded.

    A CAPACITY OF ZERO BECOMES ONE, for the reason `apy_seq_alloc` gives: a
    zero-capacity table doubles to zero forever.
    """
    if cap < 1:
        cap = 1
    cell: ptr = apy_obj_alloc(apy_dict_kind())
    if not cell:
        return cell
    keys: ptr = apy_alloc_block(cap * apy_value_size())
    if not keys:
        return keys
    vals: ptr = apy_alloc_block(cap * apy_value_size())
    if not vals:
        return vals
    store(u64, u64(keys), offset(cell, apy_d_keys_offset()))
    store(u64, u64(vals), offset(cell, apy_d_vals_offset()))
    store(i64, 0, offset(cell, apy_d_n_offset()))
    store(i64, cap, offset(cell, apy_d_cap_offset()))
    return cell


# ── emptying, viewing, and two conversions ─────────────────────────────────


def apy_clear(v: ptr) -> ptr:
    """`d.clear()`, `xs.clear()`, `s.clear()` -- the count goes to zero.

    THE ITEMS ARE NOT ERASED and do not need to be. Nothing reads past `n`,
    the block stays for the next append, and this runtime never frees a
    string's bytes anyway -- so zeroing them would be work whose only effect
    is to make the next `append` slower.

    A TUPLE AND A FROZENSET ARE NOT HERE, which is the whole reason the kinds
    are named one at a time rather than tested with `v.q`: all four share the
    arm, and immutability is the only thing telling them apart.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_dict_kind():
        store(i64, 0, offset(v, apy_d_n_offset()))
        return apy_none()
    if k == apy_list_kind() or k == apy_set_kind():
        store(i64, 0, offset(v, apy_q_n_offset()))
        return apy_none()
    if apy_is_bytearray_of(v):
        # THE TERMINATOR IS WRITTEN and the count is not merely dropped: two
        # hundred places read these bytes as a C string, so an emptied
        # bytearray whose first byte still said `a` would print as `a` to
        # every one of them. Only when there is a byte to write -- an
        # already-empty one may be pointing at a literal.
        if load(i64, offset(v, apy_str_len_offset())):
            store(u8, u8(0), apy_str_data(v))
        store(i64, 0, offset(v, apy_str_len_offset()))
        return apy_none()
    return apy_raise_fmt(
        rodata(b"AttributeError\0"),
        rodata(b"'%s' object has no attribute 'clear'%s\0"),
        apy_kind_name_of(v), rodata(b"\0"))



def apy_dict_part_name(part: i64) -> ptr:
    """`keys`, `values` or `items`, for an error that has to name one.

    THE C HARDCODED `keys` HERE and reported that a `.values()` call on
    a non-dict had no attribute `keys`. Both halves name the part now.
    """
    if part == apy_part_keys():
        return rodata(b"keys\0")
    if part == apy_part_values():
        return rodata(b"values\0")
    return rodata(b"items\0")

def apy_dict_view(d: ptr, part: i64) -> ptr:
    """`d.keys()`, `d.values()` or `d.items()`.

    A VIEW HOLDS THE DICT, NOT A COPY OF IT, which is what makes
    `list(d.keys())` after a `d[k] = v` show the new key -- and what makes a
    view over a dict that is later cleared show nothing rather than the old
    contents.

    THE MESSAGE ALWAYS SAYS 'keys' even for `.values()` on a non-dict, which
    is the C's wording and is kept: the three share one entry point, and the
    part is an argument rather than something the error can see.
    """
    if i64(load(i32, offset(d, 0))) != apy_dict_kind():
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute '%s'\0"),
            apy_kind_name_of(d), apy_dict_part_name(part))
    o: ptr = apy_obj_alloc(apy_view_kind())
    if not o:
        return o
    store(u64, u64(d), offset(o, apy_vw_dict_offset()))
    store(i32, i32(part), offset(o, apy_vw_part_offset()))
    return o


def apy_from_bytes_n(b: ptr, order: ptr) -> ptr:
    """`int.from_bytes(b, order)` for up to eight bytes.

    BIG-ENDIAN UNLESS THE ORDER IS EXACTLY 'little', which is the C's rule
    and is not the same as "little if it says little": anything that is not
    that one string -- a different word, a non-string, a missing argument --
    reads as big-endian rather than raising. Python would raise; matching the
    C is what keeps the two arrangements agreeing, and changing it is a
    decision about the C rather than about this port.

    EIGHT BYTES IS THE LIMIT because the result is an `int64`, and a ninth
    would need the big integers. The C says so with an OverflowError and so
    does this.

    UNSIGNED THROUGHOUT. The accumulator is a `u64` so a leading 0xFF shifts
    in without becoming a sign, and it leaves as a MAGNITUDE rather than
    through `apy_from_int` -- which is what turned eight 0xFF bytes into -1
    where Python answers 18446744073709551615.
    """
    if i64(load(i32, offset(b, 0))) != apy_bytes_kind():
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"cannot convert '%s' object to bytes%s\0"),
            apy_kind_name_of(b), rodata(b"\0"))
    n: i64 = apy_str_byte_len(b)
    if n > 8:
        return apy_raise_at(rodata(b"OverflowError\0"),
                            rodata(b"int too big to convert\0"))
    little: i64 = 0
    if apy_is_str(order):
        if apy_cstr_eq(apy_str_data(order), rodata(b"little\0")):
            little = 1
    p: ptr = apy_str_data(b)
    acc: u64 = u64(0)
    i: i64 = 0
    while i < n:
        at: i64 = i
        if little:
            at = n - 1 - i
        acc = (acc << u64(8)) | u64(load(u8, offset(p, at)))
        i = i + 1
    # PAST AN i64 IS A BIG INTEGER, not a negative one. Eight unsigned bytes
    # reach 1.8e19 and this conversion is unsigned, so handing the
    # accumulator to `apy_from_int` answered -1 for the largest of them --
    # a wrong VALUE rather than a refusal.
    return apy_big_of_u64_of(acc)


def apy_env_cell(env: ptr, i: i64) -> ptr:
    """Closure cell `i` of `env`.

    THE BOUNDS CHECK IS A SystemError, NOT AN IndexError, and the difference
    is who is at fault: an index out of range here means the compiler emitted
    a lookup the function was not built for, which no program could have
    caused and no handler should catch.
    """
    if i64(load(i32, offset(env, 0))) != apy_func_kind():
        return apy_env_refuse()
    if i < 0:
        return apy_env_refuse()
    if i >= load(i64, offset(env, apy_fn_ncells_offset())):
        return apy_env_refuse()
    cells: ptr = ptr(load(u64, offset(env, apy_fn_cells_offset())))
    return ptr(load(u64, offset(cells, i * apy_value_size())))


def apy_env_refuse() -> ptr:
    """The one error `apy_env_cell` can raise, written once."""
    return apy_raise_at(
        rodata(b"SystemError\0"),
        rodata(b"closure environment is not the one this function was "
               b"compiled for\0"))


def apy_dict_popitem(d: ptr) -> ptr:
    """`d.popitem()` -- remove and return the LAST pair.

    LAST AND NOT ARBITRARY, which is Python 3.7 onwards: a dict remembers
    insertion order, so `popitem` is what makes one usable as a stack. The C
    takes the last slot for the same reason, and taking the first would be
    the same amount of code and a different language.

    THE COUNT DROPS AND THE SLOTS STAY, as everywhere else in this runtime:
    nothing reads past `n`.
    """
    if i64(load(i32, offset(d, 0))) != apy_dict_kind():
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute 'popitem'%s\0"),
            apy_kind_name_of(d), rodata(b"\0"))
    n: i64 = load(i64, offset(d, apy_d_n_offset()))
    if n == 0:
        # THE QUOTES ARE INSIDE THE MESSAGE, which looks like a mistake and
        # is CPython's: a KeyError renders its argument with `repr`, and this
        # one's argument is a sentence rather than a key.
        return apy_raise_at(rodata(b"KeyError\0"),
                            rodata(b"'popitem(): dictionary is empty'\0"))
    out: ptr = apy_tuple_new(2)
    if not out:
        return out
    keys: ptr = ptr(load(u64, offset(d, apy_d_keys_offset())))
    vals: ptr = ptr(load(u64, offset(d, apy_d_vals_offset())))
    apy_seq_push(out, ptr(load(u64, offset(keys, (n - 1) * apy_value_size()))))
    apy_seq_push(out, ptr(load(u64, offset(vals, (n - 1) * apy_value_size()))))
    store(i64, n - 1, offset(d, apy_d_n_offset()))
    return out


# ── two lookups the equality split unblocked ───────────────────────────────


def apy_dict_find_of(d: ptr, key: ptr) -> i64:
    """Where `key` sits in `d`, or -1.

    A LINEAR SCAN, WHICH IS NOT A MISTAKE TO FIX HERE. The dict keeps
    insertion order in two parallel arrays and finds by walking them; that is
    the C's design and changing it is a change to the DATA STRUCTURE, not a
    port. Everything that reads a dict goes through this, so replacing it
    with a hash table is a piece of work worth doing on its own.

    A NULL DICT FINDS NOTHING rather than faulting: `apy_class_find` passes
    `v.t.dict` straight in, and a class built without one is a real state.
    """
    if not d:
        return -1
    n: i64 = load(i64, offset(d, apy_d_n_offset()))
    keys: ptr = ptr(load(u64, offset(d, apy_d_keys_offset())))
    i: i64 = 0
    while i < n:
        if apy_eq_raw_of(ptr(load(u64, offset(keys, i * apy_value_size()))),
                         key):
            return i
        i = i + 1
    return -1


def apy_dict_value_at(d: ptr, at: i64) -> ptr:
    """Value `at` of `d`, for an index `apy_dict_find_of` just answered."""
    return ptr(load(u64, offset(
        ptr(load(u64, offset(d, apy_d_vals_offset()))),
        at * apy_value_size())))


def apy_class_find_of(cls: ptr, name: ptr) -> ptr:
    """Find `name` on `cls` or above it. Null if it is nowhere.

    THE MRO WINS WHEN THERE IS ONE, and the base chain is what happens when
    there is not. Both walks exist because a class built by a `class`
    statement with one base never needs a linearisation, and computing one
    for it would be work no lookup would use.

    A NON-TYPE IN THE MRO IS SKIPPED rather than refused. The list is built
    elsewhere and this is a lookup; a lookup that raised would turn a
    malformed hierarchy into an error at every attribute access rather than
    at the place that built it.

    `name` IS AN INTERNED VALUE AND NOT A C STRING -- `apy_name_of(rodata(
    b"__next__" + a terminator))`, never the `rodata` alone. Both are `ptr`
    in this subset, so passing the wrong one compiles, links, and answers
    "not found" for a method the class plainly has. `apy_dunder_of` beside it
    takes the OTHER shape, which is what makes the mistake easy: it interns
    for itself. The interpreter cannot catch this either, because its
    bindings look names up in Python -- only a compiled differential run
    shows it, and it took four of them.
    """
    if not cls:
        return ptr(0)
    if i64(load(i32, offset(cls, 0))) != apy_type_kind():
        return ptr(0)
    order: ptr = ptr(load(u64, offset(cls, apy_t_mro_offset())))
    if order:
        n: i64 = load(i64, offset(order, apy_q_n_offset()))
        items: ptr = ptr(load(u64, offset(order, apy_q_items_offset())))
        i: i64 = 0
        while i < n:
            here: ptr = ptr(load(u64, offset(items, i * apy_value_size())))
            if i64(load(i32, offset(here, 0))) == apy_type_kind():
                d: ptr = ptr(load(u64, offset(here, apy_t_dict_offset())))
                at: i64 = apy_dict_find_of(d, name)
                if at >= 0:
                    return apy_dict_value_at(d, at)
            i = i + 1
        return ptr(0)
    walk: ptr = cls
    while walk:
        if i64(load(i32, offset(walk, 0))) != apy_type_kind():
            return ptr(0)
        d2: ptr = ptr(load(u64, offset(walk, apy_t_dict_offset())))
        at2: i64 = apy_dict_find_of(d2, name)
        if at2 >= 0:
            return apy_dict_value_at(d2, at2)
        walk = ptr(load(u64, offset(walk, apy_t_base_offset())))
    return ptr(0)


# ── three that came ready with the two lookups ─────────────────────────────


def apy_dict_get_or(d: ptr, key: ptr, fallback: ptr) -> ptr:
    """`d.get(key, fallback)`.

    A MISSING KEY IS NOT AN ERROR, which is the whole difference from `d[k]`
    -- and the fallback is returned as given rather than copied, so
    `d.get(k, [])` hands back the very list the caller passed.
    """
    if i64(load(i32, offset(d, 0))) != apy_dict_kind():
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute 'get'%s\0"),
            apy_kind_name_of(d), rodata(b"\0"))
    at: i64 = apy_dict_find_of(d, key)
    if at < 0:
        return fallback
    return apy_dict_value_at(d, at)


def apy_method_is_builtin(obj: ptr, name: ptr) -> i64:
    """Should `obj.name()` reach the runtime's own method rather than a class's?

    A CLASS METHOD ALWAYS WINS. If the instance's class defines `name`, that
    is what a program meant -- even when the name is one this runtime also
    provides, which is how a user class gets its own `append` or `keys`.

    OTHERWISE IT DEPENDS ON WHETHER THE INSTANCE HOLDS SOMETHING. `held` is
    the builtin an instance wraps -- what `class C(list)` puts inside -- and
    a method call falls through to it. An instance holding nothing has no
    builtin to fall through TO, so the answer is no.
    """
    if i64(load(i32, offset(obj, 0))) != apy_inst_kind():
        return 1
    cls: ptr = ptr(load(u64, offset(obj, apy_o_cls_offset())))
    if apy_class_find_of(cls, name):
        return 0
    if ptr(load(u64, offset(obj, apy_o_held_offset()))):
        return 1
    return 0


def apy_method_self(obj: ptr, name: ptr) -> ptr:
    """The receiver a builtin method should actually run on.

    THE HELD OBJECT, WHEN THERE IS ONE AND THE CLASS DOES NOT OVERRIDE. A
    `class C(list)` instance passed to the runtime's `append` has to reach
    the list inside it, not the wrapper -- the wrapper has no `v.q` to push
    onto.

    THE SAME TWO TESTS AS `apy_method_is_builtin`, and deliberately not
    factored into one: that one answers WHETHER and this answers WHAT, and a
    caller needs each without the other.
    """
    if i64(load(i32, offset(obj, 0))) != apy_inst_kind():
        return obj
    held: ptr = ptr(load(u64, offset(obj, apy_o_held_offset())))
    if not held:
        return obj
    if apy_class_find_of(ptr(load(u64, offset(obj, apy_o_cls_offset()))),
                         name):
        return obj
    return held


# ── the name cache, and what a dict may use as a key ───────────────────────


def apy_name_max() -> i64:
    """How many distinct attribute names get an interned cell."""
    return 48


def apy_name_rows() -> ptr:
    """`max` pairs: the C string, then the cell built for it."""
    return reserve("apy_name_rows_ir", 768)


def apy_name_slot() -> ptr:
    """How many are used."""
    return reserve("apy_name_count_ir", 8)


def apy_name_of(text: ptr) -> ptr:
    """A str cell for an attribute name, made once and remembered.

    WHY INTERN AT ALL: every attribute access builds one of these to look up
    with, and the lookup compares by VALUE -- so the cell is thrown away
    immediately. Forty-eight of them cover the dunders and the common names,
    and the arena never frees, so without this a loop calling `x.foo()` would
    allocate a string per iteration and keep it.

    A FULL CACHE STILL ANSWERS, with a fresh cell rather than an error. That
    is what makes the size a performance number and not a limit: a program
    with a forty-ninth name is slower and not broken.

    THE TEXT IS COMPARED, NOT THE POINTER. Two call sites passing the same
    literal may or may not share storage -- that is the C compiler's
    business -- so the cache would miss half the time if it compared
    addresses.
    """
    used: i64 = load(i64, apy_name_slot())
    i: i64 = 0
    while i < used:
        row: ptr = offset(apy_name_rows(), i * 16)
        if apy_cstr_eq(ptr(load(u64, row)), text):
            return ptr(load(u64, offset(row, 8)))
        i = i + 1
    made: ptr = apy_from_cstr(text)
    if used >= apy_name_max():
        return made
    if not made:
        return made
    at: ptr = offset(apy_name_rows(), used * 16)
    store(u64, u64(text), at)
    store(u64, u64(made), offset(at, 8))
    store(i64, used + 1, apy_name_slot())
    return made


def apy_unhashable_of(v: ptr) -> ptr:
    """The kind name to complain about, or null if `v` may be a key.

    A MUTABLE bytes IS A bytearray, which is why the bytes arm is tested for
    `mut` rather than refused outright: the two share a layout and only one
    of them can be hashed.

    A TUPLE IS AS HASHABLE AS ITS CONTENTS, checked by walking them -- which
    is what makes `{(1, [2]): x}` refuse while `{(1, 2): x}` does not.

    A USER OBJECT WITH `__eq__` AND NO `__hash__` IS UNHASHABLE, and this is
    where a CONTAINER finds that out. `hash(x)` already refused it, but
    `{x: 1}` went through here, found nothing to complain about, and built a
    mapping whose key could never be looked up again -- two objects that
    compare equal, hashed by address, in different places. A silent wrong
    answer where CPython raises.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_list_kind():
        return apy_kind_name_of(v)
    if k == apy_dict_kind():
        return apy_kind_name_of(v)
    if k == apy_set_kind():
        return apy_kind_name_of(v)
    if k == apy_bytes_kind():
        if load(i32, offset(v, apy_s_mut_offset())):
            return apy_kind_name_of(v)
        return ptr(0)
    if k == apy_tuple_kind():
        n: i64 = load(i64, offset(v, apy_q_n_offset()))
        items: ptr = ptr(load(u64, offset(v, apy_q_items_offset())))
        i: i64 = 0
        while i < n:
            bad: ptr = apy_unhashable_of(
                ptr(load(u64, offset(items, i * apy_value_size()))))
            if bad:
                return bad
            i = i + 1
        return ptr(0)
    if k == apy_alias_kind():
        # AN ALIAS IS AS HASHABLE AS ITS ARGUMENTS, for the same reason a
        # tuple is: the hash combines them, so `list[[]]` would otherwise be
        # hashed through a list -- by address, silently, where CPython raises
        # `unhashable type: 'list'`.
        return apy_unhashable_of(
            ptr(load(u64, offset(v, apy_ga_args_offset()))))
    if k == apy_inst_kind():
        cls: ptr = ptr(load(u64, offset(v, apy_o_cls_offset())))
        if not apy_class_find_of(cls, apy_name_of(rodata(b"__hash__\0"))):
            if apy_class_find_of(cls, apy_name_of(rodata(b"__eq__\0"))):
                return apy_str_data(
                    ptr(load(u64, offset(cls, apy_t_name_offset()))))
    return ptr(0)


def apy_unhashable_key_of(key: ptr, inner: ptr) -> ptr:
    """The TypeError a dict raises for a key it cannot hash.

    CPYTHON 3.14 WRAPS THE PLAIN TEXT when the value is used AS A KEY, which
    is why this exists beside the message `hash()` gives: the same object
    refused by `hash(x)` and by `{x: 1}` gets two different sentences, and
    the second names both the outer kind and the inner one.
    """
    return apy_raise_fmt(
        rodata(b"TypeError\0"),
        rodata(b"cannot use '%s' as a dict key "
               b"(unhashable type: '%s')\0"),
        apy_kind_name_of(key), inner)


def apy_dict_set(d: ptr, key: ptr, val: ptr) -> ptr:
    """`d[key] = val`.

    A RE-ASSIGNED KEY KEEPS ITS ORIGINAL POSITION, which is Python's rule and
    easy to get wrong the other way: insertion order is about FIRST insertion,
    not last write, so `d['a'] = 1; d['b'] = 2; d['a'] = 3` still lists `a`
    first.

    THE KEY IS CHECKED BEFORE ANYTHING IS WRITTEN. An unhashable key that got
    as far as being stored would make a dict nothing could look up again --
    which is exactly the silent wrong answer `apy_unhashable_of` exists to
    prevent.

    BOTH ARRAYS DOUBLE TOGETHER, and the OLD size is what `apy_realloc_block`
    needs -- it returns the previous block to the free list, and a wrong size
    there puts it in the wrong size class. That is why `was` is computed
    before the capacity changes.
    """
    if i64(load(i32, offset(d, 0))) != apy_dict_kind():
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"'%s' object does not support item "
                   b"assignment%s\0"),
            apy_kind_name_of(d), rodata(b"\0"))
    bad: ptr = apy_unhashable_of(key)
    if bad:
        return apy_unhashable_key_of(key, bad)
    at: i64 = apy_dict_find_of(d, key)
    if at >= 0:
        store(u64, u64(val), offset(
            ptr(load(u64, offset(d, apy_d_vals_offset()))),
            at * apy_value_size()))
        return apy_none()
    n: i64 = load(i64, offset(d, apy_d_n_offset()))
    cap: i64 = load(i64, offset(d, apy_d_cap_offset()))
    if n == cap:
        was: i64 = cap * apy_value_size()
        grown: i64 = cap * 2
        keys: ptr = apy_realloc_block(
            ptr(load(u64, offset(d, apy_d_keys_offset()))), was,
            grown * apy_value_size())
        if not keys:
            return keys
        store(u64, u64(keys), offset(d, apy_d_keys_offset()))
        vals: ptr = apy_realloc_block(
            ptr(load(u64, offset(d, apy_d_vals_offset()))), was,
            grown * apy_value_size())
        if not vals:
            return vals
        store(u64, u64(vals), offset(d, apy_d_vals_offset()))
        store(i64, grown, offset(d, apy_d_cap_offset()))
    store(u64, u64(key), offset(
        ptr(load(u64, offset(d, apy_d_keys_offset()))), n * apy_value_size()))
    store(u64, u64(val), offset(
        ptr(load(u64, offset(d, apy_d_vals_offset()))), n * apy_value_size()))
    store(i64, n + 1, offset(d, apy_d_n_offset()))
    return apy_none()


# ── seven that came ready with `apy_dict_set` ──────────────────────────────


def apy_callable(v: ptr) -> ptr:
    """`callable(x)`.

    A CLASS IS CALLABLE because calling it makes an instance, and a function
    obviously is. An INSTANCE is callable only if its class defines
    `__call__` -- which is the one case that needs a lookup rather than a
    kind test.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_func_kind():
        return apy_from_bool(1)
    if k == apy_type_kind():
        return apy_from_bool(1)
    if k == apy_inst_kind():
        if apy_class_find_of(ptr(load(u64, offset(v, apy_o_cls_offset()))),
                             apy_name_of(rodata(b"__call__\0"))):
            return apy_from_bool(1)
    return apy_from_bool(0)


def apy_locals_put(d: ptr, name: ptr, v: ptr) -> ptr:
    """Add one binding to a `locals()` dict, skipping the unbound ones.

    A NULL VALUE IS A LOCAL THAT HAS NOT BEEN ASSIGNED YET, and `locals()`
    leaves those out rather than reporting them as None -- which is what
    makes the dict it answers match what the function can actually see.
    """
    if not v:
        return d
    if not apy_dict_set(d, name, v):
        return ptr(0)
    return d


def apy_setdefault(d: ptr, key: ptr, fallback: ptr) -> ptr:
    """`d.setdefault(key, fallback)`.

    THE STORED VALUE COMES BACK, not the fallback, when the key was already
    there -- which is the whole point: `d.setdefault(k, []).append(x)` has to
    append to the list already in the dict.
    """
    if i64(load(i32, offset(d, 0))) != apy_dict_kind():
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute "
                   b"'setdefault'%s\0"),
            apy_kind_name_of(d), rodata(b"\0"))
    at: i64 = apy_dict_find_of(d, key)
    if at >= 0:
        return apy_dict_value_at(d, at)
    if not apy_dict_set(d, key, fallback):
        return ptr(0)
    return fallback


def apy_type_set(cls: ptr, name: ptr, value: ptr) -> ptr:
    """`C.name = value` -- a write into the class's own dict."""
    if i64(load(i32, offset(cls, 0))) != apy_type_kind():
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"'%s' object is not a class%s\0"),
            apy_kind_name_of(cls), rodata(b"\0"))
    if not apy_dict_set(ptr(load(u64, offset(cls, apy_t_dict_offset()))),
                        name, value):
        return ptr(0)
    return apy_none()


def apy_typevar_default(tv: ptr, value: ptr) -> ptr:
    """`T = TypeVar('T', default=int)` -- PEP 696.

    ANSWERS THE TYPEVAR, not the default, because the caller is an expression
    that has to evaluate to the variable itself.
    """
    apy_dict_set(ptr(load(u64, offset(tv, apy_o_dict_offset()))),
                 apy_from_cstr(rodata(b"__default__\0")), value)
    return tv


def apy_match_args(cls: ptr) -> ptr:
    """`C.__match_args__`, or an empty tuple.

    AN EMPTY TUPLE FOR ANYTHING THAT HAS NONE, which is what makes
    `case C(x)` a no-match rather than an error for a class that never
    declared positional fields.

    A LIST IS ACCEPTED as well as a tuple: a program may write either, and
    the walk that reads it does not care.
    """
    if i64(load(i32, offset(cls, 0))) != apy_type_kind():
        return apy_tuple_new(1)
    got: ptr = apy_class_find_of(cls, apy_name_of(rodata(b"__match_args__\0")))
    if got:
        k: i64 = i64(load(i32, offset(got, 0)))
        if k == apy_tuple_kind():
            return got
        if k == apy_list_kind():
            return got
    return apy_tuple_new(1)


def apy_match_rest(d: ptr, used: ptr) -> ptr:
    """What a `**rest` capture in a mapping pattern binds.

    THE KEYS THE PATTERN ALREADY NAMED ARE LEFT OUT, which is the whole job:
    `case {'a': x, **rest}` binds everything BUT `a`, and the comparison is
    by value rather than by position because the pattern names keys.
    """
    if i64(load(i32, offset(d, 0))) != apy_dict_kind():
        return apy_dict_new(1)
    n: i64 = load(i64, offset(d, apy_d_n_offset()))
    room: i64 = n
    if room < 1:
        room = 1
    out: ptr = apy_dict_new(room)
    if not out:
        return out
    keys: ptr = ptr(load(u64, offset(d, apy_d_keys_offset())))
    vals: ptr = ptr(load(u64, offset(d, apy_d_vals_offset())))
    un: i64 = load(i64, offset(used, apy_q_n_offset()))
    uitems: ptr = ptr(load(u64, offset(used, apy_q_items_offset())))
    i: i64 = 0
    while i < n:
        key: ptr = ptr(load(u64, offset(keys, i * apy_value_size())))
        skip: i64 = 0
        k2: i64 = 0
        while k2 < un:
            if apy_eq_raw_of(key, ptr(load(u64, offset(
                    uitems, k2 * apy_value_size())))):
                skip = 1
                k2 = un
            else:
                k2 = k2 + 1
        if not skip:
            if not apy_dict_set(out, key, ptr(load(u64, offset(
                    vals, i * apy_value_size())))):
                return ptr(0)
        i = i + 1
    return out


# ── the set predicate, and the chain above it ──────────────────────────────


def apy_is_set_of(v: ptr) -> i64:
    """Is `v` a set or a frozenset?

    THE OTHER HALF OF `apy_is_seq_of`. All four kinds share the `v.q` arm and
    these two are the ones WITHOUT positions -- which is why the pair of
    predicates exists rather than one test for the arm.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_set_kind():
        return 1
    if k == apy_frozen_kind():
        return 1
    return 0


def apy_class_builtin_of(cls: ptr) -> i64:
    """Which builtin kind a class extends, walking up, or 0.

    UP THE BASE CHAIN, NOT THE MRO, because this is a question about what the
    instance has to HOLD -- and a class can only hold one thing however many
    bases it lists. The first answer up the chain is the one that decides the
    layout.
    """
    walk: ptr = cls
    while walk:
        if i64(load(i32, offset(walk, 0))) != apy_type_kind():
            return 0
        got: i64 = i64(load(i32, offset(walk, apy_t_builtin_offset())))
        if got:
            return got
        walk = ptr(load(u64, offset(walk, apy_t_base_offset())))
    return 0


def apy_instance_new(cls: ptr) -> ptr:
    """A fresh instance of `cls`.

    `held` IS WHAT `class C(list)` PUTS INSIDE. An instance of a class that
    extends a builtin carries one, and every builtin method call on the
    instance reaches it -- see `apy_method_self`. A class extending nothing
    holds nothing, and the allocator has already made that zero.

    THE INSTANCE DICT IS ALWAYS MADE, even for a class that extends a
    builtin: `self.x = 1` has to go somewhere, and the held object is not it.
    """
    if i64(load(i32, offset(cls, 0))) != apy_type_kind():
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"'%s' object is not callable%s\0"),
            apy_kind_name_of(cls), rodata(b"\0"))
    o: ptr = apy_obj_alloc(apy_inst_kind())
    if not o:
        return o
    store(u64, u64(cls), offset(o, apy_o_cls_offset()))
    store(u64, u64(apy_dict_new(4)), offset(o, apy_o_dict_offset()))
    kind: i64 = apy_class_builtin_of(cls)
    if kind == apy_dict_kind():
        store(u64, u64(apy_dict_new(4)), offset(o, apy_o_held_offset()))
    elif kind == apy_list_kind():
        store(u64, u64(apy_list_new(4)), offset(o, apy_o_held_offset()))
    elif kind == apy_set_kind():
        store(u64, u64(apy_set_new(4)), offset(o, apy_o_held_offset()))
    elif kind == apy_tuple_kind():
        store(u64, u64(apy_tuple_new(1)), offset(o, apy_o_held_offset()))
    elif kind == apy_str_kind():
        store(u64, u64(apy_from_cstr(rodata(b"\0"))),
              offset(o, apy_o_held_offset()))
    elif kind == apy_bytes_kind():
        # `class B(bytes)`. THE SHARED IMMUTABLE EMPTY, which is what
        # `apy_bytes_literal` answers for a length of zero. The cell must
        # have `mut` clear, and it does: only `bytes` can reach this kind as
        # a base -- `bytearray` has no row in `_BUILTIN_BASE_KIND`, because
        # the two share this kind and a number cannot carry the difference.
        store(u64, u64(apy_bytes_literal(rodata(b"\0"), 0)),
              offset(o, apy_o_held_offset()))
    return o


def apy_key_at(v: ptr, i: i64) -> ptr:
    """Element `i` of anything a `for` loop or an unpacking can walk.

    ONE FUNCTION FOR SEVEN SHAPES, which is what lets `for x in y` compile to
    a counter and a call rather than to a protocol. Each arm answers where
    its elements actually live.

    AN INSTANCE FALLS THROUGH TO WHAT IT HOLDS, but only if its class does
    not define `__getitem__` -- a `class T(tuple)` unpacks like a tuple, and
    one that overrides subscripting does not.

    A GENERATOR ANSWERS FROM ITS CACHE OR ANSWERS None. It cannot be indexed
    and it cannot be rewound; the cache is what a previous full walk left
    behind, and None is the honest answer when there is none.

    A CURSOR ADVANCES ITSELF AND IGNORES `i`, which is the one arm that is
    not a lookup: `map` and `filter` have no positions, so walking one is a
    step rather than an index. The counter the caller keeps is what stops the
    loop, and the cursor's own is what feeds it.

    EVERYTHING ELSE IS A SUBSCRIPT, so a str or a user class with
    `__getitem__` walks by the rule it defines.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_inst_kind():
        held: ptr = ptr(load(u64, offset(v, apy_o_held_offset())))
        if held:
            if not apy_class_find_of(
                    ptr(load(u64, offset(v, apy_o_cls_offset()))),
                    apy_name_of(rodata(b"__getitem__\0"))):
                return apy_key_at(held, i)
    if k == apy_view_kind():
        return apy_key_at(apy_view_items(v), i)
    if k == apy_gen_kind():
        cache: ptr = ptr(load(u64, offset(v, apy_g_cache_offset())))
        if cache:
            return apy_key_at(cache, i)
        return apy_none()
    if k == apy_dict_kind():
        return ptr(load(u64, offset(
            ptr(load(u64, offset(v, apy_d_keys_offset()))),
            i * apy_value_size())))
    if apy_is_set_of(v):
        return ptr(load(u64, offset(
            ptr(load(u64, offset(v, apy_q_items_offset()))),
            i * apy_value_size())))
    if k == apy_iter_kind():
        src: ptr = ptr(load(u64, offset(v, apy_it_src_offset())))
        n: i64 = apy_raw_len(src)
        if apy_error_occurred():
            return apy_none()
        at: i64 = load(i64, offset(v, apy_it_i_offset()))
        if at >= n:
            return apy_none()
        store(i64, at + 1, offset(v, apy_it_i_offset()))
        return apy_key_at(src, at)
    return apy_getitem(v, apy_from_int(i))


# ── three walks over anything `apy_key_at` can index ───────────────────────
#
# ALL THREE HAVE THE SAME SHAPE: ask how long it is, then ask for each
# element. That is what `apy_key_at` bought -- a list, a dict, a set, a
# generator and a `map` are all walked by the same two calls, so these three
# needed no arms of their own.


def apy_dict_fromkeys(keys: ptr, value: ptr) -> ptr:
    """`dict.fromkeys(keys, value)`.

    ONE VALUE, SHARED BY EVERY KEY, and shared rather than copied -- which is
    the trap `dict.fromkeys(ks, [])` sets: all the keys get the SAME list, so
    appending through one appends through all. That is Python's behaviour and
    is reproduced rather than corrected.
    """
    # Drained first, so a user iterator can be walked -- see `apy_sorted`.
    keys = apy_iterable(keys)
    if not keys:
        return ptr(0)
    n: i64 = apy_raw_len(keys)
    if apy_error_occurred():
        return ptr(0)
    out: ptr = apy_dict_new(n + 1)
    if not out:
        return out
    i: i64 = 0
    while i < n:
        k: ptr = apy_key_at(keys, i)
        if not k:
            return ptr(0)
        if not apy_dict_set(out, k, value):
            return ptr(0)
        i = i + 1
    return out


def apy_sum(seq: ptr) -> ptr:
    """`sum(seq)`.

    STARTS AT THE INTEGER ZERO, which is why `sum([])` is `0` and not `None`
    -- and why `sum([1.5])` is `1.5`: the addition promotes, so the start
    being an int costs nothing.

    THROUGH `apy_add`, so a total that outgrows an int64 becomes a big the
    same way `2 ** 100` does.
    """
    # Drained first, so a user iterator can be walked -- see `apy_sorted`.
    seq = apy_iterable(seq)
    if not seq:
        return ptr(0)
    n: i64 = apy_raw_len(seq)
    if apy_error_occurred():
        return ptr(0)
    total: ptr = apy_from_int(0)
    i: i64 = 0
    while i < n:
        total = apy_add(total, apy_key_at(seq, i))
        if not total:
            return total
        i = i + 1
    return total


def apy_sum_from(seq: ptr, start: ptr) -> ptr:
    """`sum(seq, start)`.

    STRINGS AND BYTES ARE REFUSED BY THE START, not by the elements, which is
    CPython's rule and is the cheap place to check: `sum(xs, '')` is the way
    someone tries to join strings, and it is quadratic. The refusal names the
    method that is not.

    THE TEST IS ON `start` ALONE, so `sum(['a', 'b'])` -- with no start --
    gets a TypeError from the ADDITION instead, complaining that an int and a
    str will not add. Both refuse; only one explains.
    """
    k: i64 = i64(load(i32, offset(start, 0)))
    if k == apy_str_kind():
        return apy_raise_at(
            rodata(b"TypeError\0"),
            rodata(b"sum() can't sum strings [use ''.join(seq) "
                   b"instead]\0"))
    if k == apy_bytes_kind():
        return apy_raise_at(
            rodata(b"TypeError\0"),
            rodata(b"sum() can't sum bytes [use b''.join(seq) "
                   b"instead]\0"))
    # Drained first, so a user iterator can be walked -- see `apy_sorted`.
    seq = apy_iterable(seq)
    if not seq:
        return ptr(0)
    n: i64 = apy_raw_len(seq)
    if apy_error_occurred():
        return ptr(0)
    total: ptr = start
    i: i64 = 0
    while i < n:
        total = apy_add(total, apy_key_at(seq, i))
        if not total:
            return total
        i = i + 1
    return total


# ── binding, and reading an integer argument ───────────────────────────────


def apy_bind_of(f: ptr, self_: ptr) -> ptr:
    """A bound method: the function, plus the receiver it was found on.

    THE WHOLE `fn` ARM IS COPIED, not just the pointers a call needs. A bound
    method has to answer `__name__`, `__doc__` and its defaults exactly as
    the unbound one does -- and copying the arm is one memcpy where naming
    the fields would be nineteen stores and a new place to forget one.

    A FRESH CELL EVERY TIME, which is why `a.m is a.m` is False in Python:
    each attribute access binds again. Reusing one would make two methods of
    the same object identical and a method of two objects share a receiver.
    """
    o: ptr = apy_obj_alloc(apy_func_kind())
    if not o:
        return o
    at: i64 = 0
    while at < apy_fn_span():
        store(i64, load(i64, offset(f, apy_payload_offset() + at)),
              offset(o, apy_payload_offset() + at))
        at = at + 8
    store(u64, u64(self_), offset(o, apy_fn_bound_offset()))
    return o


def apy_dunder_of(v: ptr, name: ptr) -> ptr:
    """The bound `name` method of an instance, or null.

    ONLY AN INSTANCE HAS ONE. A dunder on a builtin is this runtime's own
    code, reached by kind rather than by lookup -- so a str has no `__add__`
    to find here, and the caller falls through to the arithmetic.
    """
    if i64(load(i32, offset(v, 0))) != apy_inst_kind():
        return ptr(0)
    m: ptr = apy_class_find_of(
        ptr(load(u64, offset(v, apy_o_cls_offset()))), apy_name_of(name))
    if not m:
        return m
    if i64(load(i32, offset(m, 0))) != apy_func_kind():
        return ptr(0)
    return apy_bind_of(m, v)


def apy_clamp_range_of(n: i64, lo: ptr, hi: ptr) -> None:
    """Turn a slice's bounds into indices inside `0 .. n`.

    A NEGATIVE BOUND COUNTS FROM THE END and then clamps at zero, which is
    why `xs[-99:]` is the whole list rather than an error. The high bound
    clamps at `n` for the same reason and the low one does not need to: a low
    bound past the end gives an empty slice by being greater than `hi`.
    """
    a: i64 = load(i64, lo)
    if a < 0:
        a = a + n
        if a < 0:
            a = 0
        store(i64, a, lo)
    b: i64 = load(i64, hi)
    if b < 0:
        b = b + n
        if b < 0:
            b = 0
    if b > n:
        b = n
    store(i64, b, hi)


def apy_int_arg_of(v: ptr, out: ptr) -> i64:
    """Read an integer argument into `out`. 0 and a pending error if not one.

    A BIG IS AN OverflowError RATHER THAN A TypeError, and the difference is
    real: `xs[2 ** 100]` is a well-typed request this runtime cannot serve,
    where `xs['a']` is not well typed at all. CPython draws the same line.
    """
    if not apy_is_int_like_of(v):
        apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"'%s' object cannot be interpreted as "
                   b"an integer%s\0"),
            apy_kind_name_of(v), rodata(b"\0"))
        return 0
    if apy_is_big_of(v):
        apy_raise_at(
            rodata(b"OverflowError\0"),
            rodata(b"Python int too large to convert to C ssize_t\0"))
        return 0
    store(i64, apy_int_payload(v), out)
    return 1


def apy_slice_arg_of(v: ptr, out: ptr) -> i64:
    """Read a slice bound, where None means "leave it alone".

    A BIG BECOMES A HUGE BOUND RATHER THAN AN ERROR, which is the difference
    from `apy_int_arg_of`: `xs[:2 ** 100]` is the whole list in Python, so
    the bound is clamped to something past any real length instead of
    refusing. The sign is kept, so `xs[-2 ** 100:]` is the whole list too.
    """
    if i64(load(i32, offset(v, 0))) == apy_none_kind():
        return 1
    # A BOUND IS A SLICE INDEX AND SAYS SO. `"abc".find("b", 1.0)` is
    # `slice indices must be integers or None or have an __index__ method` in
    # CPython, not the general wording `apy_int_arg_of` gives every other
    # integer argument. AN INSTANCE FALLS THROUGH, since one with `__index__`
    # is a valid bound and the conversion below is what asks.
    if apy_is_int_like_of(v) == 0:
        if i64(load(i32, offset(v, 0))) != apy_inst_kind():
            apy_raise_at(
                rodata(b"TypeError\0"),
                rodata(b"slice indices must be integers or None or have an "
                       b"__index__ method\0"))
            return 0
    if apy_is_big_of(v):
        big: i64 = 4611686018427387904
        if load(i32, offset(v, apy_big_neg_offset())):
            big = -big
        store(i64, big, out)
        return 1
    return apy_int_arg_of(v, out)


def apy_affix1_of(s: ptr, fix: ptr, lo: i64, hi: i64, at_end: i64) -> i64:
    """Does `fix` sit at one end of `s[lo:hi]`?

    BYTES, AND THAT IS EXACT: a prefix relation over UTF-8 bytes is the same
    relation as over characters, because a valid encoding is prefix-free at
    character boundaries. See `runtime/str_affix.py`, which splits the
    two-argument forms on the same reasoning.
    """
    m: i64 = apy_str_byte_len(fix)
    if m > hi - lo:
        return 0
    at: i64 = lo
    if at_end:
        at = hi - m
    if apy_bytes_equal_at(apy_str_data(s), at, apy_str_data(fix), m):
        return 1
    return 0



# ── hashing, and the set core above it ─────────────────────────────────────
#
# THE SET FAMILY IS ONE CHAIN and this is the bottom of it: nine functions
# stand on `apy_set_insert`, which stands on `apy_hash_raw`. None of it
# touches the call machinery, which is what makes it reachable while that
# still is not.


def apy_hash_raw_of(v: ptr) -> i64:
    """A hash for anything that can be a key. The IR half of a split.

    THREE KINDS GO TO THE SLOW HALF and each for its own reason: a FLOAT
    needs `floor` to decide whether it is integral (and `5.0` must hash as
    `5`, or `{5: x}[5.0]` would miss); a BIG folds its limbs; an INSTANCE
    asks its class, which is a call.

    AN int HASHES TO ITSELF, which is CPython's rule too and is what makes
    small integers land in order -- and what makes `True` and `1` collide,
    as they must, since they are equal.

    A TUPLE AND A FROZENSET COMBINE THEIR ELEMENTS DIFFERENTLY, and the
    difference is order: a tuple multiplies as it goes so `(1, 2)` and
    `(2, 1)` differ, while a frozenset XORs so `{1, 2}` and `{2, 1}` agree.
    Getting that backwards would make one of the two containers unusable as
    a key.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_none_kind():
        return 99339021
    if k == apy_int_kind():
        return apy_int_payload(v)
    if k == apy_bool_kind():
        return apy_int_payload(v)
    if k == apy_str_kind():
        return apy_hash_bytes(v)
    if k == apy_bytes_kind():
        return apy_hash_bytes(v)
    if k == apy_mview_kind():
        # A VIEW HASHES AS THE BYTES IT VIEWS, which is what makes
        # `hash(memoryview(b"a")) == hash(b"a")` -- the equality between them
        # already holds, and a hash that disagreed would put the two in
        # different buckets of the same dict.
        return apy_hash_bytes(apy_mview_bytes(v))
    if k == apy_tuple_kind():
        n: i64 = load(i64, offset(v, apy_q_n_offset()))
        items: ptr = ptr(load(u64, offset(v, apy_q_items_offset())))
        h: i64 = 3430008
        i: i64 = 0
        while i < n:
            h = i64(u64(h) * u64(1000003)) ^ apy_hash_raw_of(
                ptr(load(u64, offset(items, i * apy_value_size()))))
            i = i + 1
        return h ^ n
    if k == apy_frozen_kind():
        fn: i64 = load(i64, offset(v, apy_q_n_offset()))
        fitems: ptr = ptr(load(u64, offset(v, apy_q_items_offset())))
        fh: i64 = 0
        j: i64 = 0
        while j < fn:
            fh = fh ^ i64(u64(apy_hash_raw_of(ptr(load(u64, offset(
                fitems, j * apy_value_size()))))) * u64(11400714819323198485))
            j = j + 1
        return fh ^ fn
    return apy_hash_raw_of_slow(v)


def apy_hash_bytes(v: ptr) -> i64:
    """FNV-1a over a str or bytes cell's bytes.

    BY BYTE, WHICH IS RIGHT HERE for the reason `apy_str_cmp_of` gives: two
    equal strings have equal bytes, so a byte hash cannot separate values
    that compare equal -- which is the only property a hash must have.
    """
    n: i64 = apy_str_byte_len(v)
    p: ptr = apy_str_data(v)
    h: i64 = i64(u64(14695981039346656037))
    i: i64 = 0
    while i < n:
        h = h ^ i64(load(u8, offset(p, i)))
        h = i64(u64(h) * u64(1099511628211))
        i = i + 1
    return h


def apy_set_mask_of(n: i64) -> i64:
    """The mask an `n`-element set orders by.

    GROWS AT THREE FIFTHS FULL, which is what `n * 5 >= size * 3` says
    without a division. The mask rather than the size, because every use is
    `hash & mask`.
    """
    size: i64 = 8
    while n * 5 >= size * 3:
        size = size * 2
    return size - 1


def apy_q_append_of(q: ptr, item: ptr) -> None:
    """Append to a sequence cell, doubling when full.

    THE OLD SIZE IS COMPUTED FIRST, because `apy_realloc_block` returns the
    previous block to the free list and a wrong size puts it in the wrong
    size class.
    """
    n: i64 = load(i64, offset(q, apy_q_n_offset()))
    cap: i64 = load(i64, offset(q, apy_q_cap_offset()))
    if n == cap:
        was: i64 = cap * apy_value_size()
        grown: i64 = cap * 2
        items: ptr = apy_realloc_block(
            ptr(load(u64, offset(q, apy_q_items_offset()))), was,
            grown * apy_value_size())
        if not items:
            return
        store(u64, u64(items), offset(q, apy_q_items_offset()))
        store(i64, grown, offset(q, apy_q_cap_offset()))
    store(u64, u64(item), offset(
        ptr(load(u64, offset(q, apy_q_items_offset()))), n * apy_value_size()))
    store(i64, n + 1, offset(q, apy_q_n_offset()))


def apy_set_find_of(s: ptr, item: ptr) -> i64:
    """Where `item` sits in `s`, or -1.

    A LINEAR SCAN over a table kept in hash order -- which sounds like a
    contradiction and is not: the order is what makes iteration stable and
    the scan is what the C does. Replacing it with a probe is a change to the
    data structure, the same judgement `apy_dict_find_of` records.
    """
    n: i64 = load(i64, offset(s, apy_q_n_offset()))
    items: ptr = ptr(load(u64, offset(s, apy_q_items_offset())))
    i: i64 = 0
    while i < n:
        if apy_eq_raw_of(ptr(load(u64, offset(items, i * apy_value_size()))),
                         item):
            return i
        i = i + 1
    return -1


def apy_set_reorder_of(s: ptr, mask: i64) -> None:
    """Put the table back in hash order after a rehash.

    AN INSERTION SORT, which is right rather than lazy: the table was already
    ordered under the OLD mask, and a wider mask only splits buckets -- so
    almost every element is already where it belongs and the sort is close to
    linear. A general sort would be slower here.
    """
    items: ptr = ptr(load(u64, offset(s, apy_q_items_offset())))
    n: i64 = load(i64, offset(s, apy_q_n_offset()))
    i: i64 = 1
    while i < n:
        held: ptr = ptr(load(u64, offset(items, i * apy_value_size())))
        want: i64 = apy_hash_raw_of(held) & mask
        # THE LOOP ENDS AT THE INSERTION POINT, and `j` IS that point -- so
        # it cannot double as the stop flag. Setting it to zero to break
        # would store every displaced element at index 0.
        j: i64 = i
        going: i64 = 1
        while going:
            if j <= 0:
                going = 0
            else:
                prev: ptr = ptr(load(u64, offset(
                    items, (j - 1) * apy_value_size())))
                if (apy_hash_raw_of(prev) & mask) <= want:
                    going = 0
                else:
                    store(u64, u64(prev),
                          offset(items, j * apy_value_size()))
                    j = j - 1
        store(u64, u64(held), offset(items, j * apy_value_size()))
        i = i + 1


# ── inserting into a set, and what stands on it ────────────────────────────


def apy_unhashable_elem_of(item: ptr, inner: ptr) -> ptr:
    """The TypeError a SET raises for an element it cannot hash.

    A SEPARATE SENTENCE FROM THE DICT ONE, which `apy_unhashable_key_of`
    words: CPython 3.14 says 'as a set element' where a dict says 'as a dict
    key', and the two paths are far enough apart that sharing a message would
    mean losing which container refused.
    """
    return apy_raise_fmt(
        rodata(b"TypeError\0"),
        rodata(b"cannot use '%s' as a set element "
               b"(unhashable type: '%s')\0"),
        apy_kind_name_of(item), inner)


def apy_subset_of(a: ptr, b: ptr) -> i64:
    """Is every element of `a` in `b`?

    THE CALLERS PAIR IT WITH A LENGTH TEST to get equality, which is why this
    answers only the one-way question: `a == b` is `len(a) == len(b) and
    subset(a, b)`, and a subset test alone would call `{1}` equal to
    `{1, 2}`.
    """
    n: i64 = load(i64, offset(a, apy_q_n_offset()))
    items: ptr = ptr(load(u64, offset(a, apy_q_items_offset())))
    i: i64 = 0
    while i < n:
        if apy_set_find_of(
                b, ptr(load(u64, offset(items, i * apy_value_size())))) < 0:
            return 0
        i = i + 1
    return 1


def apy_mutable_set_of(name: ptr, s: ptr) -> i64:
    """Is `s` a set that may be changed? A frozenset is not.

    THE MESSAGE NAMES THE METHOD, so `frozenset().add(1)` reports that a
    frozenset has no `add` -- which is true and is how Python words it, rather
    than saying the set is immutable.
    """
    if i64(load(i32, offset(s, 0))) == apy_set_kind():
        return 1
    apy_raise_fmt(
        rodata(b"AttributeError\0"),
        rodata(b"'%s' object has no attribute '%s'\0"),
        apy_kind_name_of(s), name)
    return 0


def apy_set_insert_of(s: ptr, item: ptr) -> i64:
    """Add `item` to `s`. 1 if added, 0 if already there, -1 on refusal.

    THREE ANSWERS, NOT TWO, and the caller needs all of them: `s.add(x)` does
    not care whether it was new, but `s | t` counts and an unhashable element
    has to stop the whole operation rather than be skipped.

    THE TABLE MAY REHASH ON THIS INSERT, which is why the mask is computed
    both before and after: if the load factor crossed, every element moves and
    a full reorder is cheaper than placing one element into a table that is
    about to be rebuilt anyway.

    OTHERWISE ONE ELEMENT SLIDES INTO PLACE, the same insertion step
    `apy_set_reorder_of` does for all of them -- written out here rather than
    called, because the reorder starts at 1 and this one starts at `n`.
    """
    bad: ptr = apy_unhashable_of(item)
    if bad:
        apy_unhashable_elem_of(item, bad)
        return -1
    if apy_set_find_of(s, item) >= 0:
        return 0
    n: i64 = load(i64, offset(s, apy_q_n_offset()))
    was: i64 = apy_set_mask_of(n)
    mask: i64 = apy_set_mask_of(n + 1)
    apy_q_append_of(s, item)
    if mask != was:
        apy_set_reorder_of(s, mask)
        return 1
    items: ptr = ptr(load(u64, offset(s, apy_q_items_offset())))
    want: i64 = apy_hash_raw_of(item) & mask
    i: i64 = n
    going: i64 = 1
    while going:
        if i <= 0:
            going = 0
        else:
            prev: ptr = ptr(load(u64, offset(
                items, (i - 1) * apy_value_size())))
            if (apy_hash_raw_of(prev) & mask) <= want:
                going = 0
            else:
                store(u64, u64(prev), offset(items, i * apy_value_size()))
                i = i - 1
    store(u64, u64(item), offset(items, i * apy_value_size()))
    return 1


def apy_set_from_of(kind: i64, src: ptr) -> ptr:
    """A set or frozenset of `kind` holding everything in `src`.

    THROUGH `apy_set_insert_of`, so duplicates collapse and an unhashable
    element refuses the whole construction -- `set([1, [2]])` is a TypeError,
    not a one-element set.
    """
    # Drained first, so a user iterator can be walked -- see `apy_sorted`.
    src = apy_iterable(src)
    if not src:
        return ptr(0)
    n: i64 = apy_raw_len(src)
    if apy_error_occurred():
        return ptr(0)
    out: ptr = apy_seq_new_of(kind, n + 1)
    if not out:
        return out
    i: i64 = 0
    while i < n:
        item: ptr = apy_key_at(src, i)
        if not item:
            return ptr(0)
        if apy_set_insert_of(out, item) < 0:
            return ptr(0)
        i = i + 1
    return out


# ── seven that came ready with the set layer ───────────────────────────────


def apy_set_add(s: ptr, item: ptr) -> ptr:
    """`s.add(item)` -- and a frozenset has no `add`."""
    if not apy_mutable_set_of(rodata(b"add\0"), s):
        return ptr(0)
    if apy_set_insert_of(s, item) < 0:
        return ptr(0)
    return apy_none()


def apy_set_push(s: ptr, item: ptr) -> ptr:
    """Add to a set the runtime itself is building.

    NO MUTABILITY CHECK, which is the whole difference from `apy_set_add`: a
    frozenset is immutable to a PROGRAM, and this is how the runtime fills
    one in before anyone can see it.
    """
    if apy_set_insert_of(s, item) < 0:
        return ptr(0)
    return apy_none()


def apy_set_discard(s: ptr, item: ptr) -> ptr:
    """`s.discard(item)` -- remove it if present, and say nothing if not.

    A MISSING ELEMENT IS NOT AN ERROR, which is the whole difference from
    `remove`. An UNHASHABLE one still is: `s.discard([1])` cannot be absent
    in a way the set could have checked, so it refuses rather than
    pretending.

    THE TAIL SLIDES DOWN, keeping the table in hash order without a reorder:
    removing one element cannot change any other element's bucket.
    """
    if not apy_mutable_set_of(rodata(b"discard\0"), s):
        return ptr(0)
    bad: ptr = apy_unhashable_of(item)
    if bad:
        return apy_unhashable_elem_of(item, bad)
    at: i64 = apy_set_find_of(s, item)
    if at >= 0:
        n: i64 = load(i64, offset(s, apy_q_n_offset()))
        items: ptr = ptr(load(u64, offset(s, apy_q_items_offset())))
        k: i64 = at
        while k + 1 < n:
            store(u64, load(u64, offset(items, (k + 1) * apy_value_size())),
                  offset(items, k * apy_value_size()))
            k = k + 1
        store(i64, n - 1, offset(s, apy_q_n_offset()))
    return apy_none()


def apy_to_set(v: ptr) -> ptr:
    """`set(v)`."""
    return apy_set_from_of(apy_set_kind(), v)


def apy_to_frozenset(v: ptr) -> ptr:
    """`frozenset(v)`.

    A FROZENSET COMES BACK UNCHANGED, which is safe only because it cannot be
    modified afterwards -- the same shortcut would be a bug for `set()`, and
    is not taken there.
    """
    if i64(load(i32, offset(v, 0))) == apy_frozen_kind():
        return v
    return apy_set_from_of(apy_frozen_kind(), v)


def apy_copy(v: ptr) -> ptr:
    """`x.copy()` for a dict, a list or a set. Shallow.

    A FROZENSET COPIES TO ITSELF, for the reason `apy_to_frozenset` gives.

    THE ELEMENTS ARE SHARED, NOT COPIED, which is what 'shallow' means and
    what makes `d.copy()` cheap: the new container holds the same objects, so
    mutating one THROUGH the copy is visible through the original.

    A BYTEARRAY COPIES ITS BYTES, and `bytes` HAS NO `copy` AT ALL -- the two
    share this cell and the `mut` flag is the whole difference, so the flag is
    what decides here. Copying the bytes rather than sharing the pointer is
    the point of the method: `b.copy()[0] = 1` must not reach into `b`.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_frozen_kind():
        return v
    if k == apy_bytes_kind():
        if i64(load(i32, offset(v, apy_s_mut_offset()))) != 0:
            # A BYTES CELL OF ITS OWN, and not a string re-tagged: the string
            # constructor answers SHARED cells for the empty and the
            # one-character values, and writing the bytes kind and the `mut`
            # flag over one of those would make every later `""` a writable
            # `b""`. See `apy_bytes_made_of`.
            return apy_bytes_made_of(apy_str_data(v), apy_str_byte_len(v), 1)
    if k == apy_dict_kind():
        n: i64 = load(i64, offset(v, apy_d_n_offset()))
        out: ptr = apy_dict_new(n + 1)
        if not out:
            return out
        keys: ptr = ptr(load(u64, offset(v, apy_d_keys_offset())))
        vals: ptr = ptr(load(u64, offset(v, apy_d_vals_offset())))
        i: i64 = 0
        while i < n:
            if not apy_dict_set(
                    out,
                    ptr(load(u64, offset(keys, i * apy_value_size()))),
                    ptr(load(u64, offset(vals, i * apy_value_size())))):
                return ptr(0)
            i = i + 1
        return out
    if k == apy_list_kind() or k == apy_set_kind():
        qn: i64 = load(i64, offset(v, apy_q_n_offset()))
        made: ptr = apy_seq_new_of(k, qn + 1)
        if not made:
            return made
        qitems: ptr = ptr(load(u64, offset(v, apy_q_items_offset())))
        j: i64 = 0
        while j < qn:
            apy_q_append_of(made, ptr(load(u64, offset(
                qitems, j * apy_value_size()))))
            j = j + 1
        return made
    return apy_raise_fmt(
        rodata(b"AttributeError\0"),
        rodata(b"'%s' object has no attribute 'copy'%s\0"),
        apy_kind_name_of(v), rodata(b"\0"))


def apy_list_insert(seq: ptr, where: ptr, item: ptr) -> ptr:
    """`xs.insert(where, item)`.

    THE POSITION CLAMPS AT BOTH ENDS rather than raising: `xs.insert(99, x)`
    appends and `xs.insert(-99, x)` prepends, which is Python's rule and
    unlike every other index in the language.

    APPENDED FIRST, THEN SLID INTO PLACE, so the growth and the shift are one
    pass each and neither needs to know about the other.
    """
    if apy_is_bytearray_of(seq):
        # A BYTEARRAY INSERTS AN OCTET, and the position clamps the same way.
        if not apy_is_int_like_of(where):
            return apy_raise_fmt(
                rodata(b"TypeError\0"),
                rodata(b"'%s' object cannot be interpreted as "
                       b"an integer%s\0"),
                apy_kind_name_of(where), rodata(b"\0"))
        byte: i64 = apy_byte_arg_of(item)
        if byte < 0:
            return ptr(0)
        bn: i64 = load(i64, offset(seq, apy_str_len_offset()))
        bat: i64 = apy_int_payload(where)
        if bat < 0:
            bat = bat + bn
        if bat < 0:
            bat = 0
        if bat > bn:
            bat = bn
        # GROWN BY ONE THROUGH THE ORDINARY APPEND, so the reallocation and
        # the terminator are somebody else's problem, then slid up.
        if not apy_seq_push(seq, apy_from_int(0)):
            return ptr(0)
        bp: ptr = apy_str_data(seq)
        j: i64 = bn
        while j > bat:
            store(u8, load(u8, offset(bp, j - 1)), offset(bp, j))
            j = j - 1
        store(u8, u8(byte), offset(bp, bat))
        return apy_none()
    if i64(load(i32, offset(seq, 0))) != apy_list_kind():
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute 'insert'%s\0"),
            apy_kind_name_of(seq), rodata(b"\0"))
    if not apy_is_int_like_of(where):
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"'%s' object cannot be interpreted as "
                   b"an integer%s\0"),
            apy_kind_name_of(where), rodata(b"\0"))
    n: i64 = load(i64, offset(seq, apy_q_n_offset()))
    at: i64 = apy_int_payload(where)
    if at < 0:
        at = at + n
    if at < 0:
        at = 0
    if at > n:
        at = n
    apy_q_append_of(seq, item)
    items: ptr = ptr(load(u64, offset(seq, apy_q_items_offset())))
    i: i64 = load(i64, offset(seq, apy_q_n_offset())) - 1
    while i > at:
        store(u64, load(u64, offset(items, (i - 1) * apy_value_size())),
              offset(items, i * apy_value_size()))
        i = i - 1
    store(u64, u64(item), offset(items, at * apy_value_size()))
    return apy_none()


# ── the set algebra, and the relations beside it ───────────────────────────


def apy_op_union() -> i64:
    return 0


def apy_op_inter() -> i64:
    return 1


def apy_op_symdiff() -> i64:
    return 3


def apy_binop_error_of(op: ptr, a: ptr, b: ptr) -> ptr:
    """`unsupported operand type(s) for OP` -- the shape every operator uses.

    THREE SUBSTITUTIONS AND `apy_raise_fmt` TAKES TWO, which is why the
    operator goes in first and the two kinds follow: the message is built in
    one pass over a template that names all three, rather than by formatting
    twice.
    """
    return apy_raise_fmt(
        rodata(b"TypeError\0"),
        apy_binop_template(op),
        apy_kind_name_of(a), apy_kind_name_of(b))


def apy_binop_template(op: ptr) -> ptr:
    """The message with the operator already in it.

    BUILT IN THE MESSAGE BUFFER'S NEIGHBOUR, because `apy_raise_fmt` reads
    its template while writing the buffer -- so the template cannot BE the
    buffer. `apy_fmt_scratch` is the one it expands into, so this needs a
    third piece of storage of its own.
    """
    buf: ptr = apy_binop_scratch()
    at: i64 = apy_cstr_into(
        buf, 0, 200, rodata(b"unsupported operand type(s) for \0"))
    at = apy_cstr_into(buf, at, 200, op)
    at = apy_cstr_into(buf, at, 200, rodata(b": '%s' and '%s'\0"))
    store(u8, u8(0), offset(buf, at))
    return buf


def apy_binop_scratch() -> ptr:
    """Where a binary-operator message template is assembled."""
    return reserve("apy_binop_template_ir", 256)


def apy_set_algebra_of(op: ptr, a: ptr, b: ptr, which: i64,
                       strict: i64) -> ptr:
    """`|`, `&`, `-` and `^`, and the methods that spell them out.

    THE OPERATOR FORM IS STRICT AND THE METHOD FORM IS NOT: `{1} | [2]` is a
    TypeError and `{1}.union([2])` is a set, which is Python's rule and the
    only thing `strict` controls.

    THE RESULT KEEPS THE LEFT SIDE'S KIND, so `frozenset({1}) | {2}` is a
    frozenset. That is why the capacity is `a + rhs` rather than either
    alone -- a union can hold both.

    INTERSECTION WALKS THE SHORTER SIDE, which is worth the swap: the test is
    a scan of the other set, so walking the longer one does more of them. The
    swap is only safe when `b` was ALREADY a set -- if it was converted, the
    conversion order is what decides the result's order, and Python takes it
    from the left.
    """
    if not apy_is_set_of(a):
        return apy_binop_error_of(op, a, b)
    if strict:
        if not apy_is_set_of(b):
            return apy_binop_error_of(op, a, b)
    rhs: ptr = b
    if not apy_is_set_of(b):
        rhs = apy_set_from_of(apy_set_kind(), b)
    if not rhs:
        return rhs
    an: i64 = load(i64, offset(a, apy_q_n_offset()))
    rn: i64 = load(i64, offset(rhs, apy_q_n_offset()))
    out: ptr = apy_seq_new_of(i64(load(i32, offset(a, 0))), an + rn + 1)
    if not out:
        return out
    if which == apy_op_inter():
        walk: ptr = rhs
        test: ptr = a
        if apy_is_set_of(b):
            if rn > an:
                walk = a
                test = rhs
        wn: i64 = load(i64, offset(walk, apy_q_n_offset()))
        witems: ptr = ptr(load(u64, offset(walk, apy_q_items_offset())))
        i: i64 = 0
        while i < wn:
            item: ptr = ptr(load(u64, offset(
                witems, i * apy_value_size())))
            if apy_set_find_of(test, item) >= 0:
                apy_q_append_of(out, item)
            i = i + 1
        return out
    aitems: ptr = ptr(load(u64, offset(a, apy_q_items_offset())))
    j: i64 = 0
    while j < an:
        left: ptr = ptr(load(u64, offset(aitems, j * apy_value_size())))
        there: i64 = 0
        if apy_set_find_of(rhs, left) >= 0:
            there = 1
        if not there or which == apy_op_union():
            apy_q_append_of(out, left)
        j = j + 1
    if which == apy_op_union() or which == apy_op_symdiff():
        ritems: ptr = ptr(load(u64, offset(rhs, apy_q_items_offset())))
        k: i64 = 0
        while k < rn:
            right: ptr = ptr(load(u64, offset(
                ritems, k * apy_value_size())))
            if apy_set_find_of(a, right) < 0:
                apy_q_append_of(out, right)
            k = k + 1
    return out


def apy_set_method_of(name: ptr, a: ptr, b: ptr, which: i64) -> ptr:
    """`s.union(x)` and its three siblings -- the non-strict spelling."""
    if not apy_is_set_of(a):
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute '%s'\0"),
            apy_kind_name_of(a), name)
    return apy_set_algebra_of(name, a, b, which, 0)


def apy_set_relate_of(name: ptr, a: ptr, b: ptr, which: i64) -> ptr:
    """`issubset`, `issuperset` and `isdisjoint`.

    ALL THREE ANSWER A BOOL and none builds a set, which is why they are not
    part of the algebra above: the work is a scan, and the only thing they
    share with it is turning a non-set argument into one first.
    """
    if not apy_is_set_of(a):
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute '%s'\0"),
            apy_kind_name_of(a), name)
    rhs: ptr = b
    if not apy_is_set_of(b):
        rhs = apy_set_from_of(apy_set_kind(), b)
    if not rhs:
        return rhs
    if which == 0:
        return apy_from_bool(apy_subset_of(a, rhs))
    if which == 1:
        return apy_from_bool(apy_subset_of(rhs, a))
    n: i64 = load(i64, offset(a, apy_q_n_offset()))
    items: ptr = ptr(load(u64, offset(a, apy_q_items_offset())))
    i: i64 = 0
    while i < n:
        if apy_set_find_of(
                rhs, ptr(load(u64, offset(items, i * apy_value_size())))) >= 0:
            return apy_from_bool(0)
        i = i + 1
    return apy_from_bool(1)


def apy_set_update(target: ptr, src: ptr) -> ptr:
    """`s.update(x)` -- add everything in `x` to `s`.

    THROUGH `apy_set_push` RATHER THAN `apy_set_add`, which skips the
    mutability check: the caller has already established `target` is a set
    the program may change, and the runtime uses this to fill sets a program
    cannot see yet.
    """
    # Drained first, so a user iterator can be walked -- see `apy_sorted`.
    src = apy_iterable(src)
    if not src:
        return ptr(0)
    n: i64 = apy_raw_len(src)
    if apy_error_occurred():
        return ptr(0)
    i: i64 = 0
    while i < n:
        item: ptr = apy_key_at(src, i)
        if not item:
            return ptr(0)
        if not apy_set_push(target, item):
            return ptr(0)
        i = i + 1
    return apy_none()


def apy_vars(obj: ptr) -> ptr:
    """`vars(obj)` -- a copy of what the object's `__dict__` holds.

    A CLASS HAS ONE TOO, holding the names its body bound: methods and class
    attributes, which is what `"x" in vars(C)` asks about.

    A COPY RATHER THAN THE DICT ITSELF, so writing through the result cannot
    reach into the object -- CPython answers a mappingproxy for a class for
    the same reason, and a copy is the closest thing here.
    """
    k: i64 = i64(load(i32, offset(obj, 0)))
    if k == apy_type_kind():
        return apy_copy(ptr(load(u64, offset(obj, apy_t_dict_offset()))))
    if k != apy_inst_kind():
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"vars() argument must have __dict__ attribute%s%s\0"),
            rodata(b"\0"), rodata(b"\0"))
    return apy_copy(ptr(load(u64, offset(obj, apy_o_dict_offset()))))


def apy_hash(v: ptr) -> ptr:
    """`hash(v)`.

    AN UNHASHABLE VALUE NAMES ITSELF, and the name is the INNER one: a list
    inside a tuple makes the tuple unhashable, and reporting "list" is what
    tells a program which element to look at.
    """
    bad: ptr = apy_unhashable_of(v)
    if bad:
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"unhashable type: '%s'%s\0"),
            bad, rodata(b"\0"))
    return apy_from_int(apy_hash_raw_of(v))


def apy_ns_get(ns: ptr, key: ptr, shown: ptr) -> ptr:
    """A name looked up in a namespace dict, or the NameError for it.

    THE NAME REPORTED IS `shown` AND NOT `key`, because the two differ: a
    bundled module\'s globals are mangled, and a program that misspells one
    should be told the name it wrote rather than the one the splice made.
    """
    at: i64 = apy_dict_find_of(ns, key)
    if at >= 0:
        vals: ptr = ptr(load(u64, offset(ns, apy_d_vals_offset())))
        return ptr(load(u64, offset(vals, at * apy_value_size())))
    return apy_raise_fmt(
        rodata(b"NameError\0"),
        rodata(b"name '%s' is not defined%s\0"),
        ptr(load(u64, offset(shown, apy_str_ptr_offset()))),
        rodata(b"\0"))


def apy_unpack_check(v: ptr, want: i64, at_least: i64) -> ptr:
    """`a, b = xs` -- does `xs` have the right number of elements?

    `at_least` TURNS THE EXACT COUNT INTO A FLOOR, which is what a `*rest`
    means: `a, *b = xs` wants at least one and the message says so.

    TWO MESSAGES AND NOT ONE, because Python words the two directions
    differently -- "not enough values" and "too many values" -- and a program
    matching on the text would see the wrong one.

    ANSWERS WHAT TO INDEX, because the unpack reads BY INDEX -- `a, *b, c =
    xs` takes `c` from `xs[-1]` without ever computing a length -- and not
    everything iterable has indices. A generator, a cursor and a user object
    with `__iter__` are drained here. See the C's `apy_unpack_check`.
    """
    # WHETHER THE SURPLUS IS COUNTED, decided by the SOURCE as CPython decides
    # it: an exact list, tuple or dict knows its own length, and everything
    # else is unpacked through the iterator protocol, which can only say there
    # was one element too many. Asked before the drain replaces `v`.
    k: i64 = i64(load(i32, offset(v, 0)))
    counted: i64 = 0
    if k == apy_list_kind():
        counted = 1
    if k == apy_tuple_kind():
        counted = 1
    if k == apy_dict_kind():
        counted = 1
    src: ptr = apy_iterable(v)
    if not src:
        return ptr(0)
    # THE UNPACK WORDS ITS OWN REFUSAL. `apy_iterable` answers a value it does
    # not recognise UNCHANGED -- only a user object can fail there -- so the
    # kinds that can be walked are tested here, and CPython's wording for this
    # operation names it: `cannot unpack non-iterable int object`.
    sk: i64 = i64(load(i32, offset(src, 0)))
    walks: i64 = apy_is_seq_of(src) or apy_is_set_of(src)
    if sk == apy_str_kind() or sk == apy_bytes_kind():
        walks = 1
    if sk == apy_dict_kind() or sk == apy_range_kind():
        walks = 1
    if sk == apy_mview_kind() or sk == apy_iter_kind():
        walks = 1
    if sk == apy_gen_kind() or sk == apy_inst_kind():
        walks = 1
    if not walks:
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"cannot unpack non-iterable %s object%s\0"),
            apy_kind_name_of(v), rodata(b"\0"))
    # ANYTHING NOT READ BY POSITION IS COPIED INTO A LIST. A dict subscripts
    # by KEY, so `a, b = {1: 0, 2: 0}` asked for key 0 and raised KeyError; a
    # set cannot be subscripted at all; a cursor holds a position instead of
    # indices.
    indexed: i64 = 0
    if sk == apy_list_kind() or sk == apy_tuple_kind():
        indexed = 1
    if sk == apy_str_kind() or sk == apy_bytes_kind():
        indexed = 1
    if sk == apy_range_kind() or sk == apy_mview_kind():
        indexed = 1
    # AN INSTANCE IS WALKED AND NOT SUBSCRIPTED, even the `__len__` plus
    # `__getitem__` kind whose subscript is its protocol: the `*rest` branch
    # reads a SLICE, and `a, *rest = obj` handed one to a `__getitem__`
    # written for integers.
    if not indexed:
        drained: ptr = apy_list_new(8)
        if not drained:
            return ptr(0)
        if not apy_extend(drained, src):
            return ptr(0)
        src = drained
    v = src
    n: i64 = apy_raw_len(v)
    if apy_error_occurred():
        return ptr(0)
    if n < want:
        prefix: ptr = rodata(b"\0")
        if at_least:
            prefix = rodata(b"at least \0")
        buf: ptr = apy_fmt_scratch()
        at: i64 = apy_cstr_into(
            buf, 0, 200,
            rodata(b"not enough values to unpack (expected \0"))
        at = apy_cstr_into(buf, at, 200, prefix)
        at = apy_cstr_into(buf, at, 200, apy_decimal_of(want, 0))
        at = apy_cstr_into(buf, at, 200, rodata(b", got \0"))
        at = apy_cstr_into(buf, at, 200, apy_decimal_of(n, 1))
        at = apy_cstr_into(buf, at, 200, rodata(b")\0"))
        store(u8, u8(0), offset(buf, at))
        return apy_raise_at(rodata(b"ValueError\0"), buf)
    if not at_least and n > want:
        if counted:
            return apy_raise_fmt(
                rodata(b"ValueError\0"),
                rodata(b"too many values to unpack (expected %s, got %s)\0"),
                apy_decimal_of(want, 0), apy_decimal_of(n, 1))
        return apy_raise_fmt(
            rodata(b"ValueError\0"),
            rodata(b"too many values to unpack (expected %s)%s\0"),
            apy_decimal_of(want, 0), rodata(b"\0"))
    return v


def apy_check_slots(cls: ptr) -> ptr:
    """`__slots__` naming something the class body also binds is an error.

    CPYTHON REFUSES IT AT CLASS CREATION, and for a reason worth keeping: the
    slot and the class variable would occupy the same name, so one of them is
    silently unreachable. Which one depends on lookup order, which is exactly
    the kind of thing a program should not have to know.

    A BARE STRING IS ONE SLOT, the same rule `apy_slot_allows_of` follows.

    AN UNREADABLE `__slots__` IS LET THROUGH rather than refused: this runs at
    every class creation and a malformed declaration should fail where it is
    written, not here.
    """
    if i64(load(i32, offset(cls, 0))) != apy_type_kind():
        return apy_none()
    d: ptr = ptr(load(u64, offset(cls, apy_t_dict_offset())))
    slots: ptr = apy_dict_get_or(
        d, apy_name_of(rodata(b"__slots__\0")), ptr(0))
    if not slots:
        return apy_none()
    bare: i64 = 0
    if i64(load(i32, offset(slots, 0))) == apy_str_kind():
        bare = 1
    n: i64 = 1
    if not bare:
        n = apy_raw_len(slots)
    if apy_error_occurred():
        apy_error_clear()
        return apy_none()
    i: i64 = 0
    while i < n:
        one: ptr = slots
        if not bare:
            one = apy_key_at(slots, i)
        if one:
            if i64(load(i32, offset(one, 0))) == apy_str_kind():
                if apy_dict_find_of(d, one) >= 0:
                    return apy_raise_fmt(
                        rodata(b"ValueError\0"),
                        rodata(b"'%s' in __slots__ conflicts with "
                               b"class variable%s\0"),
                        ptr(load(u64, offset(one, apy_str_ptr_offset()))),
                        rodata(b"\0"))
        i = i + 1
    return apy_none()


def apy_pair_len(pair: ptr) -> i64:
    """How many elements a `dict()` argument\'s element has, or -1.

    A STR IS A SEQUENCE HERE. `dict(["ab", "cd"])` is `{"a": "b", "c": "d"}`
    -- each element is walked as a pair of characters -- so the length check
    applies to it and the not-a-sequence refusal does not.
    """
    if apy_is_seq_of(pair):
        return load(i64, offset(pair, apy_q_n_offset()))
    k: i64 = i64(load(i32, offset(pair, 0)))
    if k == apy_str_kind() or k == apy_bytes_kind():
        return apy_raw_len(pair)
    return -1


def apy_to_dict(src: ptr) -> ptr:
    """`dict(src)`.

    A DICT COPIES and anything else is read as a sequence of pairs, which is
    what `dict([(1, 2)])` means. A class wrapping a dict copies what it holds,
    so `dict(Counter())` answers the counts rather than walking the instance.

    TWO REFUSALS AND THEY ARE DIFFERENT ERRORS: an element that is not a
    sequence at all is a TypeError, and one of the wrong LENGTH is a
    ValueError. Reporting ValueError for both meant `except TypeError:` did
    not catch the first.
    """
    k: i64 = i64(load(i32, offset(src, 0)))
    if k == apy_dict_kind():
        return apy_copy(src)
    if k == apy_inst_kind():
        held: ptr = ptr(load(u64, offset(src, apy_o_held_offset())))
        if held:
            if i64(load(i32, offset(held, 0))) == apy_dict_kind():
                return apy_copy(held)
    # Drained first, so a user iterator can be walked -- see `apy_sorted`.
    src = apy_iterable(src)
    if not src:
        return ptr(0)
    n: i64 = apy_raw_len(src)
    if apy_error_occurred():
        return ptr(0)
    out: ptr = apy_dict_new(n + 1)
    if not out:
        return out
    i: i64 = 0
    while i < n:
        pair: ptr = apy_key_at(src, i)
        if not pair:
            return ptr(0)
        plen: i64 = apy_pair_len(pair)
        if plen < 0:
            return apy_raise_fmt(
                rodata(b"TypeError\0"),
                rodata(b"cannot convert dictionary update sequence element "
                       b"#%s to a sequence%s\0"),
                apy_decimal_of(i, 0), rodata(b"\0"))
        if plen != 2:
            return apy_raise_fmt(
                rodata(b"ValueError\0"),
                rodata(b"dictionary update sequence element #%s has length "
                       b"%s; 2 is required\0"),
                apy_decimal_of(i, 0), apy_decimal_of(plen, 1))
        key: ptr = apy_key_at(pair, 0)
        val: ptr = apy_key_at(pair, 1)
        if not key:
            return ptr(0)
        if not val:
            return ptr(0)
        if not apy_dict_set(out, key, val):
            return ptr(0)
        i = i + 1
    return out


def apy_update(target: ptr, src: ptr) -> ptr:
    """`d.update(x)` and `s.update(x)` -- one runtime function for both.

    NOT ONLY A MAPPING. `d.update([(1, 2), (3, 4)])` is legal and so is
    `d.update(["ab"])` -- any iterable of two-element iterables, which is why
    a str of two-character strings works and a str of characters does not.

    A SET UPDATES THROUGH ITS OWN INSERT, so duplicates collapse and an
    unhashable element refuses the whole operation rather than being skipped.
    """
    if i64(load(i32, offset(target, 0))) == apy_dict_kind():
        # A dict SUBCLASS UPDATES FROM ITS MAPPING, not from its keys.
        # Iterating a dict yields keys, so the pair walk below read
        # `d.update(Counter())` as a sequence of single keys and reported an
        # element of length 1 -- about a perfectly good mapping.
        if i64(load(i32, offset(src, 0))) == apy_inst_kind():
            inner: ptr = ptr(load(u64, offset(src, apy_o_held_offset())))
            if inner:
                if i64(load(i32, offset(inner, 0))) == apy_dict_kind():
                    src = inner
        if i64(load(i32, offset(src, 0))) == apy_dict_kind():
            dn: i64 = load(i64, offset(src, apy_d_n_offset()))
            keys: ptr = ptr(load(u64, offset(src, apy_d_keys_offset())))
            vals: ptr = ptr(load(u64, offset(src, apy_d_vals_offset())))
            j: i64 = 0
            while j < dn:
                if not apy_dict_set(
                        target,
                        ptr(load(u64, offset(keys, j * apy_value_size()))),
                        ptr(load(u64, offset(vals, j * apy_value_size())))):
                    return ptr(0)
                j = j + 1
            return apy_none()
        # Drained first, so a user iterator can be walked -- see
        # `apy_sorted`. `d.update(pairs())` is an ordinary spelling.
        src = apy_iterable(src)
        if not src:
            return ptr(0)
        n: i64 = apy_raw_len(src)
        if apy_error_occurred():
            return ptr(0)
        i: i64 = 0
        while i < n:
            pair: ptr = apy_key_at(src, i)
            if not pair:
                return ptr(0)
            plen: i64 = apy_pair_len(pair)
            if plen < 0:
                if i64(load(i32, offset(pair, 0))) != apy_dict_kind():
                    return apy_raise_at(
                        rodata(b"TypeError\0"),
                        rodata(b"object is not iterable\0"))
                plen = apy_raw_len(pair)
            if apy_error_occurred():
                return ptr(0)
            if plen != 2:
                return apy_raise_fmt(
                    rodata(b"ValueError\0"),
                    rodata(b"dictionary update sequence element #%s has "
                           b"length %s; 2 is required\0"),
                    apy_decimal_of(i, 0), apy_decimal_of(plen, 1))
            if not apy_dict_set(target, apy_key_at(pair, 0),
                                apy_key_at(pair, 1)):
                return ptr(0)
            i = i + 1
        return apy_none()
    if not apy_mutable_set_of(rodata(b"update\0"), target):
        return ptr(0)
    # Drained first, so a user iterator can be walked -- see `apy_sorted`.
    src = apy_iterable(src)
    if not src:
        return ptr(0)
    sn: i64 = apy_raw_len(src)
    if apy_error_occurred():
        return ptr(0)
    k: i64 = 0
    while k < sn:
        item: ptr = apy_key_at(src, k)
        if not item:
            return ptr(0)
        if apy_set_insert_of(target, item) < 0:
            return ptr(0)
        k = k + 1
    return apy_none()


# -- isinstance, which is six different questions wearing one name ----------


def apy_inst_held_of(v: ptr) -> ptr:
    """The builtin an instance wraps -- what `class C(list)` puts inside."""
    if i64(load(i32, offset(v, 0))) != apy_inst_kind():
        return ptr(0)
    return ptr(load(u64, offset(v, apy_o_held_offset())))


def apy_type_is_sub_of(of: ptr, cls: ptr) -> i64:
    """Is `cls` anywhere in `of`'s order? The `isinstance` rule for classes.

    THROUGH THE MRO WHEN THERE IS ONE, for the same reason attribute lookup
    is: `isinstance(D(), C)` for `class D(B, C)` is True and the base chain
    from D reaches only B and A.
    """
    if of:
        if i64(load(i32, offset(of, 0))) == apy_type_kind():
            order: ptr = ptr(load(u64, offset(of, apy_t_mro_offset())))
            if order:
                n: i64 = load(i64, offset(order, apy_q_n_offset()))
                items: ptr = ptr(load(u64, offset(
                    order, apy_q_items_offset())))
                i: i64 = 0
                while i < n:
                    if ptr(load(u64, offset(
                            items, i * apy_value_size()))) == cls:
                        return 1
                    i = i + 1
                return 0
    here: ptr = of
    going: i64 = 1
    while going:
        if not here:
            going = 0
        elif i64(load(i32, offset(here, 0))) != apy_type_kind():
            going = 0
        elif here == cls:
            return 1
        else:
            here = ptr(load(u64, offset(here, apy_t_base_offset())))
    return 0


def apy_isinstance(v: ptr, type_name: ptr) -> ptr:
    """`isinstance(v, t)` -- and `t` arrives in six different shapes.

    THE METACLASS DECIDES, if it says so: `__instancecheck__` is asked before
    anything structural, which is what makes `isinstance(42, Duck)` able to
    answer True. The answer is coerced to a BOOL however truthy the hook was
    -- returning the hook's own value printed `quacks`.

    A TUPLE MEANS ANY OF THESE, and there is no ambiguity with asking about
    the tuple type itself: `isinstance(x, tuple)` arrives as the STRING
    "tuple", because a builtin kind has no value form. So a tuple here is
    always the multi-type form -- including one built at run time and held in
    a variable.

    PEP 604: `isinstance(x, int | str)` asks each ARM, which is the same
    question a tuple asks and is answered the same way.

    AN INSTANCE NEVER MATCHES A BUILT-IN NAME on its own, because its kind
    name is its CLASS's name -- without that rule a class called `object`
    would answer True to everything. It matches through what it HOLDS
    instead, which is what makes `class D(dict)` an instance of `dict`.

    AN EXCEPTION IS AN INSTANCE OF EVERY BASE IN ITS CHAIN, walked by name:
    the hierarchy is a table of names and that is the only form it has.
    """
    if type_name:
        if i64(load(i32, offset(type_name, 0))) == apy_type_kind():
            meta: ptr = ptr(load(u64, offset(type_name, apy_t_meta_offset())))
            if meta:
                hook: ptr = apy_class_find_of(
                    meta, apy_name_of(rodata(b"__instancecheck__\0")))
                if hook:
                    args: ptr = alloca(16)
                    store(u64, u64(type_name), args)
                    store(u64, u64(v), offset(args, apy_value_size()))
                    got: ptr = apy_call(hook, args, 2)
                    if not got:
                        return got
                    flag: i64 = 0
                    if apy_truth(got):
                        flag = 1
                    return apy_from_bool(flag)
    k: i64 = i64(load(i32, offset(type_name, 0)))
    if k == apy_alias_kind():
        origin: ptr = ptr(load(u64, offset(type_name, apy_ga_origin_offset())))
        if i64(load(i32, offset(origin, 0))) == apy_inst_kind():
            return apy_isinstance(v, ptr(load(u64, offset(
                type_name, apy_ga_args_offset()))))
    if k == apy_tuple_kind():
        n: i64 = load(i64, offset(type_name, apy_q_n_offset()))
        items: ptr = ptr(load(u64, offset(type_name, apy_q_items_offset())))
        i: i64 = 0
        while i < n:
            got2: ptr = apy_isinstance(v, ptr(load(u64, offset(
                items, i * apy_value_size()))))
            if not got2:
                return ptr(0)
            if apy_truth(got2):
                return apy_from_bool(1)
            i = i + 1
        return apy_from_bool(0)
    if k == apy_func_kind():
        if load(i32, offset(type_name, apy_fn_is_type_offset())):
            return apy_isinstance(v, ptr(load(u64, offset(
                type_name, apy_fn_name_offset()))))
    if k == apy_type_kind():
        vk: i64 = i64(load(i32, offset(v, 0)))
        if vk == apy_exc_kind():
            return apy_isinstance(v, ptr(load(u64, offset(
                type_name, apy_t_name_offset()))))
        if vk == apy_inst_kind():
            return apy_from_bool(apy_type_is_sub_of(
                ptr(load(u64, offset(v, apy_o_cls_offset()))), type_name))
        return apy_isinstance(v, ptr(load(u64, offset(
            type_name, apy_t_name_offset()))))
    if k != apy_str_kind():
        return apy_raise_at(
            rodata(b"TypeError\0"),
            rodata(b"isinstance() arg 2 must be a type, a tuple of types, "
                   b"or a union\0"))
    want: ptr = ptr(load(u64, offset(type_name, apy_str_ptr_offset())))
    have: ptr = apy_kind_name_of(v)
    if i64(load(i32, offset(v, 0))) == apy_inst_kind():
        held: ptr = apy_inst_held_of(v)
        if held:
            if apy_cstr_eq(apy_kind_name_of(held), want):
                return apy_from_bool(1)
            if not apy_cstr_eq(want, rodata(b"object\0")):
                return apy_isinstance(held, type_name)
        plain: i64 = 0
        if apy_cstr_eq(want, rodata(b"object\0")):
            plain = 1
        return apy_from_bool(plain)
    if apy_cstr_eq(have, want):
        return apy_from_bool(1)
    if i64(load(i32, offset(v, 0))) == apy_bool_kind():
        if apy_cstr_eq(want, rodata(b"int\0")):
            return apy_from_bool(1)
    if apy_cstr_eq(want, rodata(b"object\0")):
        return apy_from_bool(1)
    if i64(load(i32, offset(v, 0))) == apy_exc_kind():
        chain: ptr = ptr(load(u64, offset(v, apy_e_name_offset())))
        walking: i64 = 1
        while walking:
            if not chain:
                walking = 0
            elif apy_cstr_eq(chain, want):
                return apy_from_bool(1)
            else:
                chain = apy_exc_parent_of(chain)
    return apy_from_bool(0)


def apy_names_object(v: ptr) -> i64:
    """Is `v` the `object` type, however it was spelled?

    TWO SPELLINGS REACH HERE. `object` in source lowers to the one type cell
    `apy_object_class` hands out; a builtin type name held in a variable
    arrives as a function marked as standing for a type. Both carry the name,
    and the name is what settles it.
    """
    if not v:
        return 0
    k: i64 = i64(load(i32, offset(v, 0)))
    name: ptr = ptr(0)
    if k == apy_type_kind():
        name = ptr(load(u64, offset(v, apy_t_name_offset())))
    elif k == apy_func_kind():
        if load(i32, offset(v, apy_fn_is_type_offset())):
            name = ptr(load(u64, offset(v, apy_fn_name_offset())))
    if not name:
        return 0
    if i64(load(i32, offset(name, 0))) != apy_str_kind():
        return 0
    if apy_cstr_eq(ptr(load(u64, offset(name, apy_str_ptr_offset()))),
                   rodata(b"object\0")):
        return 1
    return 0


def apy_is_classlike(v: ptr) -> i64:
    """Is `v` something `issubclass` may be ASKED about?"""
    if not v:
        return 0
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_type_kind():
        return 1
    if k == apy_func_kind():
        if load(i32, offset(v, apy_fn_is_type_offset())):
            return 1
    return 0


def apy_is_subclass(a: ptr, b: ptr) -> ptr:
    """`issubclass(a, b)`.

    THE METACLASS DECIDES, if it says so: `__subclasscheck__` is the mirror
    of `__instancecheck__` and is asked before anything structural, which is
    what lets an ABC claim a class it never saw.

    TWO BUILTIN TYPE NAMES COMPARE BY NAME, with two relations written out --
    everything is a subclass of `object`, and `bool` is a subclass of `int`.
    A builtin kind has no class object to walk, so the names are all there is.

    A REAL CLASS WALKS ITS ORDER FIRST and falls back to the exception
    hierarchy by NAME, because a class that subclasses a builtin exception
    has no type object above it -- the chain is a table of names.
    """
    if b:
        if i64(load(i32, offset(b, 0))) == apy_type_kind():
            meta: ptr = ptr(load(u64, offset(b, apy_t_meta_offset())))
            if meta:
                hook: ptr = apy_class_find_of(
                    meta, apy_name_of(rodata(b"__subclasscheck__\0")))
                if hook:
                    args: ptr = alloca(16)
                    store(u64, u64(b), args)
                    store(u64, u64(a), offset(args, apy_value_size()))
                    got: ptr = apy_call(hook, args, 2)
                    if not got:
                        return got
                    truthy: i64 = 0
                    if apy_truth(got):
                        truthy = 1
                    return apy_from_bool(truthy)
    # EVERYTHING IS A SUBCLASS OF `object`, and nothing's base chain
    # contains it: `object` lowers to one type cell that no class names as a
    # base, so walking the chain answered False for every class -- and
    # `issubclass(int, object)`, where the first argument is a builtin NAME
    # and the second a type cell, matched neither shape below and was refused
    # outright. Both are wrong about the most basic relation there is.
    if apy_names_object(b) and apy_is_classlike(a):
        return apy_from_bool(1)
    both_names: i64 = 0
    if a and b:
        if i64(load(i32, offset(a, 0))) == apy_func_kind():
            if i64(load(i32, offset(b, 0))) == apy_func_kind():
                if load(i32, offset(a, apy_fn_is_type_offset())):
                    if load(i32, offset(b, apy_fn_is_type_offset())):
                        both_names = 1
    if both_names:
        have: ptr = ptr(load(u64, offset(
            ptr(load(u64, offset(a, apy_fn_name_offset()))),
            apy_str_ptr_offset())))
        want: ptr = ptr(load(u64, offset(
            ptr(load(u64, offset(b, apy_fn_name_offset()))),
            apy_str_ptr_offset())))
        yes: i64 = 0
        if apy_cstr_eq(have, want):
            yes = 1
        if apy_cstr_eq(want, rodata(b"object\0")):
            yes = 1
        if apy_cstr_eq(have, rodata(b"bool\0")):
            if apy_cstr_eq(want, rodata(b"int\0")):
                yes = 1
        return apy_from_bool(yes)
    if i64(load(i32, offset(a, 0))) != apy_type_kind():
        return apy_raise_at(rodata(b"TypeError\0"),
                            rodata(b"issubclass() arg 1 must be a class\0"))
    if i64(load(i32, offset(b, 0))) != apy_type_kind():
        # A BUILTIN AS THE SECOND ARGUMENT, with a class extending it as the
        # first: `issubclass(S, str)` for a `class S(str)` is True in CPython
        # and was the refusal below -- a builtin kind has no type cell, so
        # the walk had nothing to compare against. The relation is the one
        # `apy_class_builtin_kind` records, and the name is how it travels,
        # exactly as it does for `isinstance`.
        named: ptr = ptr(0)
        if i64(load(i32, offset(b, 0))) == apy_func_kind():
            if load(i32, offset(b, apy_fn_is_type_offset())):
                named = ptr(load(u64, offset(
                    ptr(load(u64, offset(b, apy_fn_name_offset()))),
                    apy_str_ptr_offset())))
        if i64(load(i32, offset(b, 0))) == apy_str_kind():
            named = ptr(load(u64, offset(b, apy_str_ptr_offset())))
        if named:
            extends: i64 = apy_class_builtin_kind(a)
            mine: ptr = rodata(b"\0")
            if extends == apy_str_kind():
                mine = rodata(b"str\0")
            if extends == apy_bytes_kind():
                mine = rodata(b"bytes\0")
            if extends == apy_list_kind():
                mine = rodata(b"list\0")
            if extends == apy_tuple_kind():
                mine = rodata(b"tuple\0")
            if extends == apy_dict_kind():
                mine = rodata(b"dict\0")
            if extends == apy_set_kind():
                mine = rodata(b"set\0")
            same: i64 = 0
            if extends:
                if apy_cstr_eq(mine, named):
                    same = 1
            return apy_from_bool(same)
        return apy_raise_at(
            rodata(b"TypeError\0"),
            rodata(b"issubclass() arg 2 must be a class or tuple of "
                   b"classes\0"))
    if apy_type_is_sub_of(a, b):
        return apy_from_bool(1)
    chain: ptr = ptr(load(u64, offset(
        ptr(load(u64, offset(a, apy_t_name_offset()))), apy_str_ptr_offset())))
    target: ptr = ptr(load(u64, offset(
        ptr(load(u64, offset(b, apy_t_name_offset()))), apy_str_ptr_offset())))
    walking: i64 = 1
    while walking:
        if not chain:
            walking = 0
        elif apy_cstr_eq(chain, target):
            return apy_from_bool(1)
        else:
            chain = apy_exc_parent_of(chain)
    return apy_from_bool(0)


def apy_split_of(x: ptr, arg: ptr) -> ptr:
    """`x.split(a)` -- which `split` depends on what `x` is.

    ONE NAME, TWO METHODS. An exception GROUP splits into a matching part and
    a rest; a string splits on a separator. The frontend cannot tell which
    from the call site, so the receiver decides here.
    """
    if i64(load(i32, offset(x, 0))) == apy_exc_kind():
        if ptr(load(u64, offset(x, apy_e_subs_offset()))):
            return apy_group_split(x, arg)
    return apy_str_split(x, arg)
