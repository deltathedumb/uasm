# The slice and the descriptor, in the machine subset.
#
# STAGE 5 OF docs/INERT-RUNTIME.md, and the first port outside the string and
# number work. What made it reachable is not a new capability -- it is a
# better question, and the question is CLOSURE.
#
# ── the metric that matters ────────────────────────────────────────────────
#
# The count that guided this port for a long time was "how many exported
# functions are blocked by a `static` C helper", and it said 89% and pointed
# at `apy_fail2` and `apy_kind_name`. That is a real obstacle but it is the
# WRONG MEASURE, because the invariant the inert runtime actually keeps is
# stronger: the ported runtime CALLS NOTHING IT DOES NOT DEFINE. That is what
# lets a backend owe three functions and no more.
#
# Under the right measure most of the "unblocked" functions turn out to be
# unportable after all -- `apy_hasattr` uses no static at all, but it calls
# `apy_getattr`, so porting it would open a hole in the closure. And a few
# functions nobody had noticed turn out to be ready, which is how these two
# were found: every callee they have is already in IR.
#
# ── what these two are ─────────────────────────────────────────────────────
#
# CELL CONSTRUCTION AND NOTHING ELSE. Both ask the arena for a cell, fill in
# three or four fields and hand it back; there is no walk, no allocation of
# bytes, no decision. That is why they move WHOLE rather than splitting: there
# is no case they cannot handle, so there is nothing for a slow half to do.
#
# `apy_obj_alloc` ZEROES THE PAYLOAD, which both of these rely on and only one
# of them says so in the C. `apy_descr_new` writes `set` and `del_` as 0
# explicitly; `apy_slice_new` writes all three of its fields, so it never had
# to care. The zeroing is kept in view here because a fourth field added to
# either arm would otherwise be uninitialised in exactly one of them.


# THE NUMBERS BELOW ARE THE C COMPILER'S, not anyone's reading of the enum:
# `tests/uasm/integration/test_ported_int.py` asks a real compiler for
# each one and fails if a line here disagrees. They are written without
# docstrings because that probe reads them as one-line constants.


def apy_slice_kind() -> i64:
    return 23


def apy_prop_kind() -> i64:
    return 26


def apy_slice_start_offset() -> i64:
    return 8


def apy_slice_stop_offset() -> i64:
    return 16


def apy_slice_step_offset() -> i64:
    return 24


def apy_prop_get_offset() -> i64:
    return 8


def apy_prop_set_offset() -> i64:
    return 16


def apy_prop_del_offset() -> i64:
    return 24


def apy_prop_kind_offset() -> i64:
    return 32


def apy_slice_new(start: ptr, stop: ptr, step: ptr) -> ptr:
    """`slice(start, stop, step)`.

    THE THREE FIELDS ARE VALUES, NOT NUMBERS. A slice holds whatever it was
    given -- None for an omitted bound is the common case, and `a[::2]` puts
    None in two of the three. Nothing here interprets them; that is the job of
    whoever subscripts with the slice.
    """
    o: ptr = apy_obj_alloc(apy_slice_kind())
    if not o:
        return o
    store(u64, u64(start), offset(o, apy_slice_start_offset()))
    store(u64, u64(stop), offset(o, apy_slice_stop_offset()))
    store(u64, u64(step), offset(o, apy_slice_step_offset()))
    return o


def apy_descr_new(fn: ptr, kind: i64) -> ptr:
    """A descriptor over `fn` -- a property, a classmethod, a staticmethod.

    ONLY THE GETTER IS SET. `property(f)` has no setter and no deleter until
    `@x.setter` builds a new descriptor, so both are written as zero rather
    than left to the allocator -- which does zero them, and which the C also
    does not trust for exactly this pair.

    `kind` IS THE ONE NUMBER, and it is what tells a `classmethod` from a
    `staticmethod` from a `property` at lookup time. It is stored as a
    narrower field than the rest, which is why it gets its own store width.
    """
    o: ptr = apy_obj_alloc(apy_prop_kind())
    if not o:
        return o
    store(u64, u64(fn), offset(o, apy_prop_get_offset()))
    store(u64, u64(0), offset(o, apy_prop_set_offset()))
    store(u64, u64(0), offset(o, apy_prop_del_offset()))
    store(i32, i32(kind), offset(o, apy_prop_kind_offset()))
    return o


# ── the property decorators, and one list method ───────────────────────────
#
# THE FIRST THINGS THE FIXED SURVEY FOUND. `uasm plugin port` was reading a
# fraction of the C -- its comment-and-literal stripper deleted most of the
# runtime -- and these four had never once appeared on its list.
#
# `PROPERTY` IS ZERO, which is why `apy_descr_new` needs no kind argument
# here: it is the first member of the enum and every one of these builds a
# property rather than a classmethod or a staticmethod.


def apy_prop_kind_property() -> i64:
    """The descriptor kind a `@property` makes. First of its enum."""
    return 0


def apy_prop_new_from(prop: ptr, get: ptr, set_: ptr, del_: ptr) -> ptr:
    """A fresh property carrying three functions.

    A NEW DESCRIPTOR EVERY TIME, NOT A MUTATION. `@x.setter` reads as though
    it changes `x`, and it must not: the class body binds the RESULT, and a
    property shared by two classes would otherwise gain a setter in both.
    That is also why each of the three below passes the two halves it is not
    replacing rather than leaving them for the allocator.
    """
    out: ptr = apy_descr_new(get, apy_prop_kind_property())
    if not out:
        return out
    store(u64, u64(set_), offset(out, apy_prop_set_offset()))
    store(u64, u64(del_), offset(out, apy_prop_del_offset()))
    return out


def apy_prop_part(prop: ptr, which: i64) -> ptr:
    """One of a property's three functions, by offset."""
    return ptr(load(u64, offset(prop, which)))


def apy_prop_refuse(prop: ptr, name: ptr) -> ptr:
    """The AttributeError all three share, worded for the one that asked."""
    return apy_raise_fmt(
        rodata(b"AttributeError\0"),
        rodata(b"'%s' object has no attribute '%s'\0"),
        apy_kind_name_of(prop), name)


def apy_prop_getter(prop: ptr, fn: ptr) -> ptr:
    """`@x.getter` -- the same property with a new reader."""
    if i64(load(i32, offset(prop, 0))) != apy_prop_kind():
        return apy_prop_refuse(prop, rodata(b"getter\0"))
    return apy_prop_new_from(prop, fn,
                             apy_prop_part(prop, apy_prop_set_offset()),
                             apy_prop_part(prop, apy_prop_del_offset()))


def apy_prop_setter(prop: ptr, fn: ptr) -> ptr:
    """`@x.setter` -- the same property with a new writer."""
    if i64(load(i32, offset(prop, 0))) != apy_prop_kind():
        return apy_prop_refuse(prop, rodata(b"setter\0"))
    return apy_prop_new_from(prop,
                             apy_prop_part(prop, apy_prop_get_offset()),
                             fn,
                             apy_prop_part(prop, apy_prop_del_offset()))


def apy_prop_deleter(prop: ptr, fn: ptr) -> ptr:
    """`@x.deleter` -- the same property with a new remover."""
    if i64(load(i32, offset(prop, 0))) != apy_prop_kind():
        return apy_prop_refuse(prop, rodata(b"deleter\0"))
    return apy_prop_new_from(prop,
                             apy_prop_part(prop, apy_prop_get_offset()),
                             apy_prop_part(prop, apy_prop_set_offset()),
                             fn)


def apy_list_reverse(seq: ptr) -> ptr:
    """`xs.reverse()` -- in place, and only for a list.

    A TUPLE IS REFUSED BY THE SAME CHECK THAT ADMITS A LIST, which is the
    point of testing the kind rather than testing for `v.q`: both arms have
    items and a count, and reversing a tuple in place would erase the one
    distinction between them.

    HALF THE WALK, because each swap places two elements. A loop to `n`
    would reverse the list and then reverse it back.
    """
    if apy_is_bytearray_of(seq):
        # THE SAME HALF-WALK OVER BYTES rather than over an items array.
        bn: i64 = load(i64, offset(seq, apy_str_len_offset()))
        bp: ptr = apy_str_data(seq)
        k: i64 = 0
        while k < bn // 2:
            kept: u8 = load(u8, offset(bp, k))
            store(u8, load(u8, offset(bp, bn - 1 - k)), offset(bp, k))
            store(u8, kept, offset(bp, bn - 1 - k))
            k = k + 1
        return apy_none()
    if i64(load(i32, offset(seq, 0))) != apy_list_kind():
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute 'reverse'%s\0"),
            apy_kind_name_of(seq), rodata(b"\0"))
    n: i64 = load(i64, offset(seq, apy_q_n_offset()))
    items: ptr = ptr(load(u64, offset(seq, apy_q_items_offset())))
    i: i64 = 0
    while i < n // 2:
        lo: ptr = offset(items, i * apy_value_size())
        hi: ptr = offset(items, (n - 1 - i) * apy_value_size())
        held: u64 = load(u64, lo)
        store(u64, load(u64, hi), lo)
        store(u64, held, hi)
        i = i + 1
    return apy_none()


def apy_fn_dict_offset() -> i64:
    return 144


# -- setting an attribute, which is six exported functions deep --------------


def apy_is_data_descriptor_of(v: ptr) -> i64:
    """Does `v` want to intercept a WRITE as well as a read?

    THE DISTINCTION DECIDES WHO WINS. A data descriptor beats the instance
    dict; a non-data one (a plain method, a `classmethod`) loses to it, which
    is what lets `c.m = 5` shadow a method.

    A `classmethod` OR `staticmethod` IS NOT ONE, which is why the kind is
    checked rather than the tag alone -- all three share a cell.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_prop_kind():
        if i64(load(i32, offset(v, apy_prop_kind_offset()))) == \
                apy_prop_kind_property():
            return 1
        return 0
    if k != apy_inst_kind():
        return 0
    cls: ptr = ptr(load(u64, offset(v, apy_o_cls_offset())))
    if apy_class_find_of(cls, apy_name_of(rodata(b"__set__\0"))):
        return 1
    if apy_class_find_of(cls, apy_name_of(rodata(b"__delete__\0"))):
        return 1
    return 0


def apy_descr_set_of(d: ptr, obj: ptr, value: ptr) -> i64:
    """Hand a write to a data descriptor. -1 if it is not one.

    THREE ANSWERS AND THE CALLER NEEDS ALL OF THEM: -1 means "not mine, store
    it yourself", 1 means "taken", 0 means "taken and it failed". Folding the
    last two together would make a setter that raised look like a name the
    instance dict should have stored.

    A `property` WITH NO SETTER IS AN ERROR AND NOT A FALLTHROUGH: `c.v = 4`
    on a read-only property must refuse rather than quietly shadow it.
    """
    if not apy_is_data_descriptor_of(d):
        return -1
    if i64(load(i32, offset(d, 0))) == apy_prop_kind():
        setter: ptr = ptr(load(u64, offset(d, apy_prop_set_offset())))
        if not setter:
            apy_raise_at(rodata(b"AttributeError\0"),
                         rodata(b"can't set attribute\0"))
            return 1
        argv: ptr = alloca(16)
        store(u64, u64(obj), argv)
        store(u64, u64(value), offset(argv, apy_value_size()))
        if apy_call(setter, argv, 2):
            return 1
        return 0
    m: ptr = apy_class_find_of(
        ptr(load(u64, offset(d, apy_o_cls_offset()))),
        apy_name_of(rodata(b"__set__\0")))
    if not m:
        return -1
    argv2: ptr = alloca(16)
    store(u64, u64(obj), argv2)
    store(u64, u64(value), offset(argv2, apy_value_size()))
    if apy_call(apy_bind_of(m, d), argv2, 2):
        return 1
    return 0


def apy_slot_allows_of(cls: ptr, name: ptr) -> i64:
    """May `name` be stored on an instance of `cls`?

    TWO WALKS, AND THE FIRST IS THE POINT: a class anywhere in the chain
    WITHOUT `__slots__` gives every instance a dict, so any name is allowed
    and the second walk never runs. Only when every class declares one does
    the name have to appear in one of them.

    A BARE STRING IS ONE SLOT. `__slots__ = "x"` is legal and means the same
    as `("x",)`, which is why the string case is tested before the walk --
    iterating it would allow "x" and also allow nothing else of length one.

    AN UNREADABLE `__slots__` ALLOWS EVERYTHING rather than refusing: this is
    a permission check, and a malformed declaration should not turn every
    assignment into an error far from the class that wrote it.
    """
    here: ptr = cls
    walking: i64 = 1
    while walking:
        if not here:
            walking = 0
        elif i64(load(i32, offset(here, 0))) != apy_type_kind():
            walking = 0
        else:
            d: ptr = ptr(load(u64, offset(here, apy_t_dict_offset())))
            if apy_dict_find_of(
                    d, apy_name_of(rodata(b"__slots__\0"))) < 0:
                return 1
            here = ptr(load(u64, offset(here, apy_t_base_offset())))
    here = cls
    going: i64 = 1
    while going:
        if not here:
            going = 0
        elif i64(load(i32, offset(here, 0))) != apy_type_kind():
            going = 0
        else:
            d2: ptr = ptr(load(u64, offset(here, apy_t_dict_offset())))
            at: i64 = apy_dict_find_of(
                d2, apy_name_of(rodata(b"__slots__\0")))
            if at >= 0:
                vals: ptr = ptr(load(u64, offset(d2, apy_d_vals_offset())))
                names: ptr = ptr(load(u64, offset(
                    vals, at * apy_value_size())))
                n: i64 = apy_raw_len(names)
                if apy_error_occurred():
                    apy_error_clear()
                    return 1
                if i64(load(i32, offset(names, 0))) == apy_str_kind():
                    return apy_eq_raw_of(names, name)
                i: i64 = 0
                while i < n:
                    if apy_eq_raw_of(apy_key_at(names, i), name):
                        return 1
                    i = i + 1
            here = ptr(load(u64, offset(here, apy_t_base_offset())))
    return 0


def apy_default_setattr(obj: ptr, name: ptr, value: ptr) -> ptr:
    """Store `name` on `obj`, the way `object.__setattr__` does.

    A DATA DESCRIPTOR ON THE CLASS TAKES THE WRITE. `c.v = 4` where the class
    has a `property` runs its setter and stores nothing in the instance dict
    -- otherwise the next read would find the stored value and the property
    would never be consulted again.

    FOUR KINDS CARRY A DICT and each makes it on first write: an instance
    always has one, an exception and a function get one only if a program
    hangs something on them, and a class stores through `apy_type_set`
    because a class attribute is not simply a dict entry.
    """
    k: i64 = i64(load(i32, offset(obj, 0)))
    if k == apy_inst_kind():
        cls: ptr = ptr(load(u64, offset(obj, apy_o_cls_offset())))
        if not apy_slot_allows_of(cls, name):
            return apy_raise_fmt(
                rodata(b"AttributeError\0"),
                rodata(b"'%s' object has no attribute '%s' "
                       b"and no __dict__ for setting new attributes\0"),
                apy_kind_name_of(obj),
                ptr(load(u64, offset(name, apy_str_ptr_offset()))))
        found: ptr = apy_class_find_of(cls, name)
        if found:
            handled: i64 = apy_descr_set_of(found, obj, value)
            if handled == 0:
                return ptr(0)
            if handled == 1:
                return apy_none()
        if not apy_dict_set(
                ptr(load(u64, offset(obj, apy_o_dict_offset()))),
                name, value):
            return ptr(0)
        return apy_none()
    if k == apy_exc_kind():
        held: ptr = ptr(load(u64, offset(obj, apy_e_dict_offset())))
        if not held:
            held = apy_dict_new(4)
            if not held:
                return ptr(0)
            store(u64, u64(held), offset(obj, apy_e_dict_offset()))
        if not apy_dict_set(held, name, value):
            return ptr(0)
        return apy_none()
    if k == apy_type_kind():
        return apy_type_set(obj, name, value)
    if k == apy_func_kind():
        fheld: ptr = ptr(load(u64, offset(obj, apy_fn_dict_offset())))
        if not fheld:
            fheld = apy_dict_new(4)
            if not fheld:
                return ptr(0)
            store(u64, u64(fheld), offset(obj, apy_fn_dict_offset()))
        if not apy_dict_set(fheld, name, value):
            return ptr(0)
        return apy_none()
    return apy_raise_fmt(
        rodata(b"AttributeError\0"),
        rodata(b"'%s' object has no attribute '%s'\0"),
        apy_kind_name_of(obj),
        ptr(load(u64, offset(name, apy_str_ptr_offset()))))


def apy_setattr(obj: ptr, name: ptr, value: ptr) -> ptr:
    """`obj.name = value`.

    `__setattr__` INTERCEPTS EVERY assignment, the mirror of
    `__getattribute__`. Asked HERE rather than inside the default so that the
    default stays callable from within the override -- which is what
    `object.__setattr__(self, name, value)` is for, and the only way an
    override can actually store anything.

    `C.__name__ = ...` CHANGES WHAT THE CLASS IS CALLED. The name is a field
    on the type, not an entry in its dict, so storing it as an ordinary
    attribute left `__name__` reading the old one -- the write appeared to
    succeed and changed nothing.

    AND THE EXCEPTION REGISTRATION FOLLOWS THE RENAME. The hierarchy is a
    table of NAMES, so a class renamed after it was registered leaves the two
    disagreeing -- and a bundled module\'s classes are spliced under mangled
    names and then restore `__name__`, precisely so the mangling stays
    invisible. BOTH spellings are kept, because generated code raises through
    the mangled one.
    """
    # A str SUBCLASS IS AN ATTRIBUTE NAME -- see `apy_getattr`.
    nheld: ptr = ptr(0)
    if i64(load(i32, offset(name, 0))) == apy_inst_kind():
        nheld = ptr(load(u64, offset(name, apy_o_held_offset())))
    if nheld:
        if i64(load(i32, offset(nheld, 0))) == apy_str_kind():
            name = nheld
    if i64(load(i32, offset(obj, 0))) == apy_type_kind():
        if i64(load(i32, offset(name, 0))) == apy_str_kind():
            if apy_cstr_eq(
                    ptr(load(u64, offset(name, apy_str_ptr_offset()))),
                    rodata(b"__name__\0")):
                return apy_rename_class(obj, value)
    if i64(load(i32, offset(obj, 0))) == apy_inst_kind():
        hook: ptr = apy_class_find_of(
            ptr(load(u64, offset(obj, apy_o_cls_offset()))),
            apy_name_of(rodata(b"__setattr__\0")))
        if hook:
            argv: ptr = alloca(16)
            store(u64, u64(name), argv)
            store(u64, u64(value), offset(argv, apy_value_size()))
            return apy_call(apy_bind_of(hook, obj), argv, 2)
    return apy_default_setattr(obj, name, value)


def apy_rename_class(obj: ptr, value: ptr) -> ptr:
    """`C.__name__ = "D"`, and the exception table that has to follow it.

    `!found` IS THE CASE THAT MATTERS. An exception class with an EMPTY BODY
    is never handed to `apy_exc_class_bind` at all -- there is nothing to
    build -- so nothing was registered under the mangled name, the
    construction could not find a class, and every display fell back to the
    name the CELL carries. Right for a user\'s class, wrong for a bundled one
    whose cells carry the mangled spelling.
    """
    was: ptr = ptr(load(u64, offset(
        ptr(load(u64, offset(obj, apy_t_name_offset()))),
        apy_str_ptr_offset())))
    parent: ptr = apy_exc_parent_of(was)
    found: ptr = apy_exc_class_named_of(was)
    store(u64, u64(value), offset(obj, apy_t_name_offset()))
    if parent:
        now: ptr = ptr(load(u64, offset(value, apy_str_ptr_offset())))
        if not apy_cstr_eq(was, now):
            apy_exc_register(value, apy_from_cstr(parent))
            if found == obj or not found:
                apy_exc_class_bind(value, obj)
                apy_exc_class_bind(apy_from_cstr(was), obj)
    return apy_none()


def apy_typing_final(obj: ptr) -> ptr:
    """`@final` -- record that this must not be subclassed or overridden.

    A FLAG AND NOTHING ELSE. PEP 591 is a checker\'s rule, not a runtime
    one, and CPython does the same thing: it sets the attribute so a tool can
    read it and lets the program run either way.
    """
    apy_setattr(obj, apy_from_cstr(rodata(b"__final__\0")),
                apy_from_bool(1))
    if apy_error_occurred():
        return ptr(0)
    return obj


def apy_typing_override(obj: ptr) -> ptr:
    """`@override` -- record that this is meant to override a base method.

    PEP 698, and the same bargain `apy_typing_final` makes: the attribute is
    for a checker to read.
    """
    apy_setattr(obj, apy_from_cstr(rodata(b"__override__\0")),
                apy_from_bool(1))
    if apy_error_occurred():
        return ptr(0)
    return obj


def apy_interp_slot() -> ptr:
    """Where the one `Interpolation` class lives once it is built."""
    return reserve("apy_interp_cls_ir", 8)


def apy_interpolation_new(value: ptr, expression: ptr, conversion: ptr,
                          spec: ptr) -> ptr:
    """PEP 750: one `{...}` of a t-string, as an object.

    FOUR FIELDS AND NO METHODS. An interpolation is what a template hands its
    consumer, and everything interesting is done BY that consumer -- so the
    class exists to carry names and nothing else.
    """
    cls: ptr = ptr(load(u64, apy_interp_slot()))
    if not cls:
        cls = apy_type_new(apy_from_cstr(rodata(b"Interpolation\0")),
                           ptr(0))
        if not cls:
            return ptr(0)
        store(u64, u64(cls), apy_interp_slot())
    one: ptr = apy_instance_new(cls)
    if not one:
        return ptr(0)
    apy_setattr(one, apy_from_cstr(rodata(b"value\0")), value)
    apy_setattr(one, apy_from_cstr(rodata(b"expression\0")), expression)
    apy_setattr(one, apy_from_cstr(rodata(b"conversion\0")), conversion)
    apy_setattr(one, apy_from_cstr(rodata(b"format_spec\0")), spec)
    if apy_error_occurred():
        return ptr(0)
    return one


def apy_template_slot() -> ptr:
    """Where the one `Template` class lives once it is built."""
    return reserve("apy_template_cls_ir", 8)


def apy_template_new(strings: ptr, interps: ptr, values: ptr) -> ptr:
    """PEP 750: a t-string, as an object.

    THE LITERAL PIECES AND THE INTERPOLATIONS ARE KEPT APART, which is the
    whole point of a template: a consumer decides what to do with each
    substituted value rather than being handed a finished string.
    """
    cls: ptr = ptr(load(u64, apy_template_slot()))
    if not cls:
        cls = apy_type_new(apy_from_cstr(rodata(b"Template\0")), ptr(0))
        if not cls:
            return ptr(0)
        store(u64, u64(cls), apy_template_slot())
    t: ptr = apy_instance_new(cls)
    if not t:
        return ptr(0)
    apy_setattr(t, apy_from_cstr(rodata(b"strings\0")), strings)
    apy_setattr(t, apy_from_cstr(rodata(b"interpolations\0")), interps)
    apy_setattr(t, apy_from_cstr(rodata(b"values\0")), values)
    if apy_error_occurred():
        return ptr(0)
    return t


def apy_nat_init() -> i64:
    return 1

def apy_nat_new() -> i64:
    return 2

def apy_nat_repr() -> i64:
    return 3

def apy_nat_str() -> i64:
    return 4

def apy_nat_eq() -> i64:
    return 5

def apy_nat_ne() -> i64:
    return 6

def apy_nat_hash() -> i64:
    return 7

def apy_nat_getattr() -> i64:
    return 8

def apy_nat_setattr() -> i64:
    return 9

def apy_nat_delattr() -> i64:
    return 10

def apy_nat_init_subclass() -> i64:
    return 19


def apy_nat_obj_only() -> i64:
    """The selector for the six dunders `object` hands down that are NOT the
    receiver's -- the four orderings, `__subclasshook__` and `__format__`.

    LAST IN THE C's ENUM, which is why it is 37 and why appending it there
    renumbered nothing. `apy_nat_count` is 38 to match.
    """
    return 37


# -- the pieces attribute lookup stands on ---------------------------------


def apy_class_builtin_kind(cls: ptr) -> i64:
    """Which builtin kind, if any, this class extends.

    THROUGH THE BASE CHAIN, because `class D(C)` where `class C(dict)` is
    still a dict -- the tag is recorded on the class that named the builtin
    and inherited by everything under it.
    """
    here: ptr = cls
    going: i64 = 1
    while going:
        if not here:
            going = 0
        elif i64(load(i32, offset(here, 0))) != apy_type_kind():
            going = 0
        else:
            k: i64 = i64(load(i32, offset(here, apy_t_builtin_offset())))
            if k:
                return k
            here = ptr(load(u64, offset(here, apy_t_base_offset())))
    return 0


def apy_is_descriptor_of(v: ptr) -> i64:
    """Does `v` want to intercept a READ?

    ANY `__get__` COUNTS, unlike `apy_is_data_descriptor_of` which wants
    `__set__`: a plain method is a descriptor for reading and loses to the
    instance dict for writing, which is the whole non-data/data distinction.
    """
    if i64(load(i32, offset(v, 0))) == apy_prop_kind():
        return 1
    if i64(load(i32, offset(v, 0))) != apy_inst_kind():
        return 0
    if apy_class_find_of(ptr(load(u64, offset(v, apy_o_cls_offset()))),
                         apy_name_of(rodata(b"__get__\0"))):
        return 1
    return 0


def apy_member_slot() -> ptr:
    """Where the one `member_descriptor` class lives."""
    return reserve("apy_member_cls_ir", 8)


def apy_member_descriptor() -> ptr:
    """One `member_descriptor`, which is what a slot reads as on the CLASS.

    `C.x` FOR A SLOTTED CLASS IS NOT THE VALUE -- there is no instance to
    read it from -- and CPython answers a descriptor object. Answering the
    slot's value would be answering some other instance's.
    """
    held: ptr = ptr(load(u64, apy_member_slot()))
    if not held:
        held = apy_type_new(
            apy_from_cstr(rodata(b"member_descriptor\0")), ptr(0))
        if not held:
            return held
        store(u64, u64(held), apy_member_slot())
    return apy_instance_new(held)


def apy_kind_class(obj: ptr) -> ptr:
    """The class object standing for `obj`'s builtin kind.

    THE SAME OBJECT `type(x)` ANSWERS, and it kept its OWN interning table
    until `__class__` reached the ordinary kinds: two tables, each interned
    correctly, each handing out a different type object for `list` -- so
    `[1].__class__ is type([1])` was False while `type([1]) is type([2])` was
    True. One concept has one table, and `apy_type_for`'s is the one that
    also honours the canonical registration, which is what makes
    `type(1) is int` hold.
    """
    return apy_type_for(obj)


def apy_object_default(want: ptr) -> ptr:
    """`object`'s own implementation of a dunder, by name.

    ELEVEN NAMES AND NO MORE. These are what `super().__repr__()` reaches
    from a class that overrode it, and what `object` carries so that
    `hasattr(x, "__eq__")` is True for everything.
    """
    if apy_cstr_eq(want, rodata(b"__init__\0")):
        return apy_native_of(apy_nat_init(), 1, rodata(b"__init__\0"))
    if apy_cstr_eq(want, rodata(b"__new__\0")):
        # TWO SLOTS, THE SECOND OPTIONAL. `object.__new__(cls)` is the
        # ordinary spelling and `object.__new__(cls, content)` fills the
        # builtin half of a class that extends one -- see the native.
        return apy_native_of(apy_nat_new(), 2, rodata(b"__new__\0"))
    if apy_cstr_eq(want, rodata(b"__repr__\0")):
        return apy_native_of(apy_nat_repr(), 1, rodata(b"__repr__\0"))
    if apy_cstr_eq(want, rodata(b"__str__\0")):
        return apy_native_of(apy_nat_str(), 1, rodata(b"__str__\0"))
    if apy_cstr_eq(want, rodata(b"__eq__\0")):
        return apy_native_of(apy_nat_eq(), 2, rodata(b"__eq__\0"))
    if apy_cstr_eq(want, rodata(b"__ne__\0")):
        return apy_native_of(apy_nat_ne(), 2, rodata(b"__ne__\0"))
    if apy_cstr_eq(want, rodata(b"__hash__\0")):
        return apy_native_of(apy_nat_hash(), 1, rodata(b"__hash__\0"))
    if apy_cstr_eq(want, rodata(b"__getattribute__\0")):
        return apy_native_of(apy_nat_getattr(), 2,
                             rodata(b"__getattribute__\0"))
    if apy_cstr_eq(want, rodata(b"__setattr__\0")):
        return apy_native_of(apy_nat_setattr(), 3,
                             rodata(b"__setattr__\0"))
    if apy_cstr_eq(want, rodata(b"__delattr__\0")):
        return apy_native_of(apy_nat_delattr(), 2,
                             rodata(b"__delattr__\0"))
    if apy_cstr_eq(want, rodata(b"__init_subclass__\0")):
        return apy_native_of(apy_nat_init_subclass(), 1,
                             rodata(b"__init_subclass__\0"))
    # AND THE ELEVEN WITH NO SELECTOR OF THEIR OWN -- the four orderings,
    # `__format__`, `__dir__`, `__sizeof__`, `__subclasshook__`,
    # `__getstate__` and the two pickle hooks. Every one of them already has
    # a body, in `apy_nat_kind()`'s dispatch, which is where a builtin VALUE
    # reaches it; what was missing was a value naming it from `object`. A
    # selector per name would be eleven more numbers that differ only in
    # which name they carry, so the name IS the selector -- and
    # `apy_object_arity` is already the table of which names `object` hands
    # down and how many slots each takes.
    #
    # UNBOUND, because `object.__sizeof__(x)` writes its receiver out: the
    # reader binds one of these when it finds it on a class, exactly as it
    # binds the ten above.
    # SIX OF THEM ARE NOT THE RECEIVER'S, and must not reach that dispatch.
    # `apy_kind_method_of` answers with the KIND's own body, which is right
    # for `__dir__`, `__sizeof__`, `__getstate__` and the two pickle hooks --
    # `object.__sizeof__(x)` IS `x`'s size -- and wrong for these, because
    # `object` does not define them at all. Routed there,
    # `object.__lt__(1, 2)` became int's `<` and answered True; CPython
    # answers NotImplemented for every pair, because `object_richcompare`
    # has no ordering case. `__subclasshook__` is the same answer for the
    # same reason, and `object.__format__` takes an EMPTY spec only. The C's
    # `APY_NAT_OBJ_ONLY` is the twin of this, and holds the bodies.
    if apy_cstr_eq(want, rodata(b"__lt__\0")):
        return apy_native_of(apy_nat_obj_only(), 2, want)
    if apy_cstr_eq(want, rodata(b"__le__\0")):
        return apy_native_of(apy_nat_obj_only(), 2, want)
    if apy_cstr_eq(want, rodata(b"__gt__\0")):
        return apy_native_of(apy_nat_obj_only(), 2, want)
    if apy_cstr_eq(want, rodata(b"__ge__\0")):
        return apy_native_of(apy_nat_obj_only(), 2, want)
    if apy_cstr_eq(want, rodata(b"__subclasshook__\0")):
        # TWO SLOTS, THE SECOND OPTIONAL -- see `apy_native_of`. Read off a
        # VALUE the class is bound and both are filled; read off `object`
        # only one is, and it is a classmethod either way.
        return apy_native_of(apy_nat_obj_only(), 2, want)
    if apy_cstr_eq(want, rodata(b"__format__\0")):
        return apy_native_of(apy_nat_obj_only(), 2, want)
    common: i64 = apy_object_arity(want)
    if common:
        return apy_kind_method_of(ptr(0), common, want, 0)
    return ptr(0)


def apy_object_slot() -> ptr:
    """Where the one `object` class lives once it is built."""
    return reserve("apy_object_cls_ir", 8)


def apy_object_class() -> ptr:
    """The class `object` itself is, made once and filled with its dunders.

    THE DICT IS THE ANSWER TO `dir(object)`. CPython's `dir` over a class is
    the merge of its MRO's dicts and `object`'s MRO is itself, so the two
    questions are one question -- and this filled ten of the twenty-four
    names 3.14 answers. Nothing on it was wrong; fourteen were absent, so
    `dir(object)` was a list that under-reported by more than half and
    `object.__sizeof__` was an AttributeError about an attribute every value
    in the language carries.

    TWENTY-TWO ARE FILLED HERE, each through `apy_object_default`.
    `__doc__` is the twenty-third and is set below, because it is TEXT and
    not a method that function could answer. `__class__` is the
    twenty-fourth and is in no dict at all -- see `apy_dir_chain`.
    """
    held: ptr = ptr(load(u64, apy_object_slot()))
    if held:
        return held
    cls: ptr = apy_type_new(apy_from_cstr(rodata(b"object\0")), ptr(0))
    if not cls:
        return cls
    store(u64, u64(cls), apy_object_slot())
    d: ptr = ptr(load(u64, offset(cls, apy_t_dict_offset())))
    apy_object_fill(d, rodata(b"__init__\0"))
    apy_object_fill(d, rodata(b"__new__\0"))
    apy_object_fill(d, rodata(b"__repr__\0"))
    apy_object_fill(d, rodata(b"__str__\0"))
    apy_object_fill(d, rodata(b"__eq__\0"))
    apy_object_fill(d, rodata(b"__ne__\0"))
    apy_object_fill(d, rodata(b"__hash__\0"))
    apy_object_fill(d, rodata(b"__getattribute__\0"))
    apy_object_fill(d, rodata(b"__setattr__\0"))
    apy_object_fill(d, rodata(b"__delattr__\0"))
    apy_object_fill(d, rodata(b"__init_subclass__\0"))
    apy_object_fill(d, rodata(b"__lt__\0"))
    apy_object_fill(d, rodata(b"__le__\0"))
    apy_object_fill(d, rodata(b"__gt__\0"))
    apy_object_fill(d, rodata(b"__ge__\0"))
    apy_object_fill(d, rodata(b"__format__\0"))
    apy_object_fill(d, rodata(b"__dir__\0"))
    apy_object_fill(d, rodata(b"__sizeof__\0"))
    apy_object_fill(d, rodata(b"__subclasshook__\0"))
    apy_object_fill(d, rodata(b"__getstate__\0"))
    apy_object_fill(d, rodata(b"__reduce__\0"))
    apy_object_fill(d, rodata(b"__reduce_ex__\0"))
    # PEP 257, AND THE TEXT IS CPYTHON'S OWN. The generated doc table is
    # keyed by the KINDS this runtime models and `object` is not one of them,
    # so the string is written out here rather than looked up -- it is
    # `object.__doc__` in 3.14, copied verbatim. Both readers find it: the
    # class through its dict, and `object()` through its class.
    apy_dict_set(d, apy_name_of(rodata(b"__doc__\0")),
                 apy_from_cstr(rodata(
                     b"The base class of the class hierarchy.\n"
                     b"\n"
                     b"When called, it accepts no arguments and returns a new"
                     b" featureless\n"
                     b"instance that has no instance attributes and cannot be"
                     b" given any.\n\0")))
    return cls


def apy_object_fill(d: ptr, name: ptr) -> None:
    """Put one of `object`'s defaults into its dict, under its own name."""
    apy_dict_set(d, apy_name_of(name), apy_object_default(name))


def apy_descr_get_of(d: ptr, obj: ptr, cls: ptr) -> ptr:
    """Read through a descriptor.

    A `property` READ ON THE CLASS ANSWERS THE DESCRIPTOR, not a value:
    `C.v` has no instance to compute from, and CPython hands back the
    property object so `C.v.setter` works.

    A `classmethod` BINDS THE CLASS and a `staticmethod` binds nothing,
    which is the whole difference between the two.
    """
    if i64(load(i32, offset(d, 0))) == apy_prop_kind():
        kind: i64 = i64(load(i32, offset(d, apy_prop_kind_offset())))
        getter: ptr = ptr(load(u64, offset(d, apy_prop_get_offset())))
        if kind == apy_prop_kind_property():
            if not obj:
                return d
            if not getter:
                return apy_raise_at(rodata(b"AttributeError\0"),
                                    rodata(b"unreadable attribute\0"))
            one: ptr = alloca(8)
            store(u64, u64(obj), one)
            return apy_call(getter, one, 1)
        if kind == apy_prop_classmethod():
            return apy_bind_of(getter, cls)
        return getter
    m: ptr = apy_class_find_of(
        ptr(load(u64, offset(d, apy_o_cls_offset()))),
        apy_name_of(rodata(b"__get__\0")))
    argv: ptr = alloca(16)
    who: ptr = obj
    if not who:
        who = apy_none()
    what: ptr = cls
    if not what:
        what = apy_none()
    store(u64, u64(who), argv)
    store(u64, u64(what), offset(argv, apy_value_size()))
    return apy_call(apy_bind_of(m, d), argv, 2)


# -- what a BUILTIN answers to, which is a table and not a class ------------
#
# A builtin kind has no class object to look a name up in: the methods live in
# the frontend's dispatch table, which is a compile-time fact. So when a
# program asks for one as a VALUE -- `[].append`, `d.keys`, `x.__len__` -- the
# answer has to be synthesised, and this is where the list of what exists
# lives.


def apy_kind_method_of(obj: ptr, arity: i64, name: ptr, bind: i64) -> ptr:
    """One builtin method, as a callable value, optionally bound.

    `APY_NAT_KIND` STANDS FOR "WHATEVER THIS KIND'S IS", which is why it is
    never cached: two callers asking for it mean different functions.
    """
    fn: ptr = apy_native_of(apy_nat_kind(), arity, name)
    if bind:
        return apy_bind_of(fn, obj)
    return fn


def apy_kind_method_var(obj: ptr, name: ptr, bind: i64) -> ptr:
    """The same, TAKING WHATEVER IT IS GIVEN.

    `"{} {}".format(a, b)` has no argument count to declare and keywords
    besides, so neither a fixed arity nor an optional tail can say what it
    accepts. DECLARED AS THREE WITH BOTH VARIADIC PARTS -- receiver, the
    surplus as a tuple, the keywords as a dict -- which is the shape the call
    machinery already packs for a `def f(self, *rest, **kw)`, and the body
    reads the three slots.
    """
    fn: ptr = apy_native_of(apy_nat_kind(), 3, name)
    if not fn:
        return fn
    store(i32, i32(1), offset(fn, apy_fn_vararg_offset()))
    store(i32, i32(1), offset(fn, apy_fn_kwarg_offset()))
    if bind:
        return apy_bind_of(fn, obj)
    return fn


def apy_kind_bit_of(v: ptr) -> i64:
    """The bit the generated method table wants for this receiver.

    ONE PLACE THAT KNOWS THE PAIRING, so the kind tags and the generated
    table cannot drift apart silently. The bits are `KINDS` in
    `objects/c/_gen_kindmeth.py` and are private to the pair.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_str_kind():
        return 1
    # A VIEW IS IN THE PAIRING FOR THE WRITTEN TABLE ALONE -- which is about
    # what `type()` calls a bound method -- and has no row in the arity one:
    # its methods are hand-written above, because several of them mean
    # something a bytes receiver would answer differently for.
    if k == apy_mview_kind():
        return 4096
    if k == apy_bytes_kind():
        if i64(load(i32, offset(v, apy_s_mut_offset()))) != 0:
            return 4
        return 2
    if k == apy_list_kind():
        return 8
    if k == apy_tuple_kind():
        return 16
    if k == apy_dict_kind():
        return 32
    if k == apy_set_kind():
        return 64
    if k == apy_frozen_kind():
        return 128
    if k == apy_int_kind():
        return 256
    if k == apy_big_kind():
        return 256
    if k == apy_bool_kind():
        return 256
    if k == apy_float_kind():
        return 512
    if k == apy_range_kind():
        return 1024
    if k == apy_complex_kind():
        return 2048
    return 0


def apy_kind_method_opt(obj: ptr, arity: i64, nopt: i64, name: ptr,
                        bind: i64) -> ptr:
    """The same, WITH AN OPTIONAL TAIL: the method may be called with up to
    `nopt` fewer arguments than it declares.

    `x.find(sub)`, `x.find(sub, i)` and `x.find(sub, i, j)` are ONE METHOD,
    and a cell carrying one arity could not say so: the call machinery
    TRUNCATES a surplus argument and then finds the count it expected, so
    `getattr([1], "__len__")(9)` answered 1 rather than refusing, and no
    method with an optional argument could be reached by name at all.

    RECORDED AS `ndefaults` WITH A NULL `defaults`, which no other cell has:
    there are no VALUES to fill a missing slot with -- the body dispatches on
    how many it actually got. A separate entry point rather than a parameter
    on the one above, because eighty-five call sites take a fixed arity and
    say so.
    """
    fn: ptr = apy_native_of(apy_nat_kind(), arity, name)
    if not fn:
        return fn
    store(i64, nopt, offset(fn, apy_fn_ndefaults_offset()))
    store(u64, u64(0), offset(fn, apy_fn_defaults_offset()))
    if bind:
        return apy_bind_of(fn, obj)
    return fn


def apy_kind_is(obj: ptr, kind: i64) -> i64:
    """`obj`'s kind tag against one value, as an i64.

    AN i64 AND NOT A BOOL, because the tests below combine these with
    `apy_is_seq_of` and its siblings -- which answer i64 -- and the subset
    will not mix the two widths in one `and`. Written once here rather than
    at each of the twenty places that needs it.
    """
    if i64(load(i32, offset(obj, 0))) == kind:
        return 1
    return 0


def apy_name_is(want: ptr, name: ptr) -> i64:
    """A C-string comparison, as an i64. See `apy_kind_is`."""
    if apy_cstr_eq(want, name):
        return 1
    return 0



def apy_object_arity(want: ptr) -> i64:
    """The arity of a dunder EVERY object has, or 0 for anything else.

    WHAT `object` CARRIES, and the reason this is one table rather than a
    test per kind: `hasattr(x, "__eq__")` is True for every value in Python,
    a list and an int and a function alike, so the answer cannot depend on
    what `x` is. Comparison is the part programs actually read --
    `functools.total_ordering` asks a class which orderings it already has,
    and `abc` asks the same question structurally.

    THE ORDERINGS ARE HERE EVEN THOUGH MOST KINDS REFUSE THEM, because that
    is what CPython does: `{}.__lt__({})` answers `NotImplemented` rather
    than raising, and it is the operator above it that turns that into the
    TypeError a program sees.
    """
    if apy_cstr_eq(want, rodata(b"__eq__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__ne__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__lt__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__le__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__gt__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__ge__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__str__\0")):
        return 1
    if apy_cstr_eq(want, rodata(b"__repr__\0")):
        return 1
    if apy_cstr_eq(want, rodata(b"__format__\0")):
        return 2
    # AND THE TWELVE `object` HANDS DOWN THAT NOTHING OVERRIDES. Every one
    # was missing from every builtin value, which is 144 attributes a
    # program can ask for and Python guarantees: `__reduce_ex__` is how
    # pickle finds a value, `__dir__` is what `dir()` reads, and `__init__`
    # and `__new__` are on everything there is.
    if apy_cstr_eq(want, rodata(b"__init__\0")):
        return 1
    if apy_cstr_eq(want, rodata(b"__new__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__getattribute__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__setattr__\0")):
        return 3
    if apy_cstr_eq(want, rodata(b"__delattr__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__init_subclass__\0")):
        return 1
    if apy_cstr_eq(want, rodata(b"__subclasshook__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__dir__\0")):
        return 1
    if apy_cstr_eq(want, rodata(b"__sizeof__\0")):
        return 1
    if apy_cstr_eq(want, rodata(b"__reduce__\0")):
        return 1
    if apy_cstr_eq(want, rodata(b"__reduce_ex__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__getstate__\0")):
        return 1
    return 0


def apy_number_arity(want: ptr, is_int: i64, is_complex: i64) -> i64:
    """The arity of a NUMBER's dunder, or 0 where that kind has none.

    THREE OVERLAPPING SETS, and the overlaps are what make this one function.
    Every number adds, multiplies, divides, negates and has a truth; only an
    int and a float floor-divide, take a remainder and convert between the
    two; only an int has bits. A complex has none of the last two groups --
    `(1j).__floordiv__` is an AttributeError in Python and not a method that
    refuses -- which is the distinction a `numbers` ABC registration reads.

    THE REFLECTED HALVES ARE REAL METHODS. `(1).__radd__(2)` is 3, and a
    class implementing a numeric tower calls them by name.
    """
    if apy_cstr_eq(want, rodata(b"__add__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__radd__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__sub__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__rsub__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__mul__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__rmul__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__truediv__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__rtruediv__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__pow__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__rpow__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__neg__\0")):
        return 1
    if apy_cstr_eq(want, rodata(b"__pos__\0")):
        return 1
    if apy_cstr_eq(want, rodata(b"__abs__\0")):
        return 1
    if apy_cstr_eq(want, rodata(b"__bool__\0")):
        return 1
    if is_complex:
        if apy_cstr_eq(want, rodata(b"__complex__\0")):
            return 1
        return 0
    # AN INT AND A FLOAT, BOTH: the whole-number operations and the two
    # conversions between them. A complex has left above.
    if apy_cstr_eq(want, rodata(b"__floordiv__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__rfloordiv__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__mod__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__rmod__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__divmod__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__rdivmod__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__int__\0")):
        return 1
    if apy_cstr_eq(want, rodata(b"__float__\0")):
        return 1
    if apy_cstr_eq(want, rodata(b"__trunc__\0")):
        return 1
    if apy_cstr_eq(want, rodata(b"__floor__\0")):
        return 1
    if apy_cstr_eq(want, rodata(b"__ceil__\0")):
        return 1
    # `__round__` IS NOT HERE because it is not a fixed arity: it takes an
    # OPTIONAL `ndigits`, which this table has no way to say. It is answered
    # in `apy_kind_attr_of` through `apy_kind_method_opt` instead -- the
    # entry point that exists for exactly this.
    if not is_int:
        return 0
    # AN INT'S OWN: the bit operations and `__index__`, which is the promise
    # that this value may stand where a position is wanted.
    if apy_cstr_eq(want, rodata(b"__index__\0")):
        return 1
    if apy_cstr_eq(want, rodata(b"__invert__\0")):
        return 1
    if apy_cstr_eq(want, rodata(b"__and__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__rand__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__or__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__ror__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__xor__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__rxor__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__lshift__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__rlshift__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__rshift__\0")):
        return 2
    if apy_cstr_eq(want, rodata(b"__rrshift__\0")):
        return 2
    return 0


def apy_kind_method_ranged_of(obj: ptr, want: ptr, bind: i64) -> ptr:
    """The native for a builtin method whose real bounds the generated words
    table knows -- or 0 for a name it does not have, and for the six set
    methods that take ANY number of others.

    THE BOUNDS ARE THIS KIND'S. The arity table holds one row per NAME:
    `str.count` takes three arguments and `list.count` takes one, and the
    union of the two said three for both -- so a method reached BY NAME
    accepted counts its receiver refuses and refused counts its receiver
    accepts. The C twin is `apy_kind_method_ranged`.
    """
    said: i64 = apy_kind_meth_words_of(want, apy_kind_bit_of(obj))
    if not said:
        return ptr(0)
    most: i64 = (said >> 4) & 15
    # NO UPPER END AT ALL is the six set methods, and a declared arity
    # cannot say so: they take whatever they are given. The FOLD is in the
    # C -- `apy_set_fold` -- which both compiled runtimes share.
    if most == 15:
        return apy_kind_method_var(obj, want, bind)
    least: i64 = said & 15
    # A SECOND BOUND MEANS THE DECLARED RANGE IS THE WIDER ONE. `to_bytes`
    # takes at most TWO positionals and THREE arguments, and `sort` takes
    # none and two, because each has a keyword-only parameter -- and once the
    # keywords are folded into slots the call arrives at its full width. The
    # POSITIONAL bound is checked where the positional count is still known;
    # the C's `apy_meth_positional` does it for both compiled runtimes.
    over: i64 = (said >> 25) & 15
    if over:
        most = over - 1
    return apy_kind_method_opt(obj, most + 1, most - least, want, bind)


def apy_kind_attr_of(obj: ptr, want: ptr, bind: i64) -> ptr:
    """The builtin method or field `want` names on `obj`, or null.

    GATED BY KIND, every one of them: `[].keys` has to be an AttributeError
    and `{}.keys` a method, and the only thing that separates them here is
    which test the name sits behind.

    `__hash__` EXISTS EITHER WAY and answers None for a mutable kind, because
    `[].__hash__ is None` is how a program asks whether a list can be a dict
    key -- "no such attribute" is a different claim from the one CPython
    makes.
    """
    seq: i64 = apy_is_seq_of(obj)
    sset: i64 = apy_is_set_of(obj)
    is_str: i64 = apy_kind_is(obj, apy_str_kind())
    is_bytes: i64 = apy_kind_is(obj, apy_bytes_kind())
    is_dict: i64 = apy_kind_is(obj, apy_dict_kind())
    is_list: i64 = apy_kind_is(obj, apy_list_kind())
    is_set: i64 = apy_kind_is(obj, apy_set_kind())
    is_mview: i64 = apy_kind_is(obj, apy_mview_kind())
    is_view: i64 = apy_kind_is(obj, apy_view_kind())
    is_gen: i64 = apy_kind_is(obj, apy_gen_kind())
    is_iter: i64 = apy_kind_is(obj, apy_iter_kind())
    is_range: i64 = apy_kind_is(obj, apy_range_kind())
    text: i64 = is_str or is_bytes
    walks: i64 = seq or sset or text or is_dict or is_mview or is_view
    writable_bytes: i64 = 0
    if is_bytes:
        if load(i32, offset(obj, apy_s_mut_offset())):
            writable_bytes = 1
    mut: i64 = is_list or is_dict or is_set or writable_bytes
    is_int: i64 = apy_is_int_like_of(obj)
    is_float: i64 = apy_kind_is(obj, apy_float_kind())
    is_complex: i64 = apy_kind_is(obj, apy_complex_kind())
    num: i64 = is_int or is_float or is_complex
    if apy_name_is(want, rodata(b"__hash__\0")):
        if mut:
            return apy_none()
        return apy_kind_method_of(obj, 1, rodata(b"__hash__\0"), bind)
    # `x.__class__` IS `type(x)`, for every value there is -- and it was
    # missing from all of them. `obj.__class__.__name__` is an everyday
    # idiom, and a program that reaches for it got an AttributeError about
    # the one attribute Python guarantees. NOT A METHOD but the type object
    # itself, which is why it answers before `apy_kind_method_of` is reached;
    # `apy_type_for` interns per kind, so `x.__class__ is type(x)` holds.
    if apy_name_is(want, rodata(b"__class__\0")):
        return apy_type_for(obj)
    # `object` GIVES THESE TO EVERYTHING, which is why they are gated on no
    # kind at all: `hasattr(x, "__eq__")` is True for every object there is,
    # and a structural test written against `collections.abc` -- or
    # `functools.total_ordering`, which asks which orderings a class already
    # has -- reads them rather than calling them.
    common: i64 = apy_object_arity(want)
    if common != 0:
        return apy_kind_method_of(obj, common, want, bind)
    if apy_name_is(want, rodata(b"__len__\0")) and walks:
        return apy_kind_method_of(obj, 1, rodata(b"__len__\0"), bind)
    if apy_name_is(want, rodata(b"__iter__\0")):
        if walks or is_gen or is_iter:
            return apy_kind_method_of(obj, 1, rodata(b"__iter__\0"), bind)
    if apy_name_is(want, rodata(b"__next__\0")):
        if is_gen or is_iter:
            return apy_kind_method_of(obj, 1, rodata(b"__next__\0"), bind)
    # ONLY THE CURSORS THAT WALK A SIZED SOURCE have a length hint, which is
    # CPython's line: a list, tuple, str, bytes, range, dict, set or reversed
    # iterator carries one, and `map`, `filter`, `enumerate`, `zip` and a
    # `callable_iterator` do not. Those are the LAZY modes, and what they
    # have left is not a question their source can answer. The plain mode is
    # zero, which is why it is compared against and not named.
    # AND A `memory_iterator` HAS NONE, which is the one sized walk without
    # one: `iter(mv)` in CPython carries `__iter__` and `__next__` and
    # nothing else, where `reversed(mv)` -- a plain `reversed` -- does have
    # the hint.
    if apy_name_is(want, rodata(b"__length_hint__\0")):
        if is_iter:
            mode: i64 = i64(load(i32, offset(obj, apy_it_mode_offset())))
            hnamed: i64 = i64(load(i32,
                                   offset(obj, apy_it_named_offset())))
            if mode == 0 or mode == apy_it_rev():
                bare: i64 = 0
                if mode == 0:
                    if hnamed == apy_mview_kind():
                        bare = 1
                if not bare:
                    return apy_kind_method_of(
                        obj, 1, rodata(b"__length_hint__\0"), bind)
    if apy_name_is(want, rodata(b"__contains__\0")) and walks:
        return apy_kind_method_of(obj, 2, rodata(b"__contains__\0"), bind)
    if apy_name_is(want, rodata(b"__getitem__\0")):
        if seq or text or is_dict or is_mview:
            return apy_kind_method_of(obj, 2, rodata(b"__getitem__\0"),
                                      bind)
    if apy_name_is(want, rodata(b"__setitem__\0")):
        if is_list or is_dict or writable_bytes:
            return apy_kind_method_of(obj, 3, rodata(b"__setitem__\0"),
                                      bind)
    # WHATEVER `del x[k]` WOULD REACH. Exactly the three kinds that take a
    # `__setitem__`, because a container that cannot be written cannot have
    # a piece taken out of it either.
    if apy_name_is(want, rodata(b"__delitem__\0")):
        if is_list or is_dict or writable_bytes:
            return apy_kind_method_of(obj, 2, rodata(b"__delitem__\0"),
                                      bind)
    # `reversed(x)` WALKS ANYTHING INDEXABLE, but only three kinds carry the
    # method that names it: a str or a tuple is reversed through `__len__`
    # and `__getitem__`, and CPython gives neither a `__reversed__`.
    if apy_name_is(want, rodata(b"__reversed__\0")):
        if is_list or is_dict or is_range:
            return apy_kind_method_of(obj, 1, rodata(b"__reversed__\0"),
                                      bind)
    # CONCATENATION AND REPETITION, which a sequence has and a set, a dict
    # and a range do not -- `range(3) * 2` is a TypeError in Python and the
    # attribute is absent, not a method that refuses.
    if seq or text:
        if apy_name_is(want, rodata(b"__add__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__mul__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__rmul__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
    if num:
        # `__round__` TAKES AN OPTIONAL `ndigits`, which the arity table
        # cannot say -- `(1.55).__round__()` and `(1.55).__round__(1)` are
        # one method. A COMPLEX HAS NONE: there is no rounding of one.
        if is_complex == 0:
            if apy_name_is(want, rodata(b"__round__\0")):
                return apy_kind_method_opt(obj, 2, 1, want, bind)
        arity: i64 = apy_number_arity(want, is_int, is_complex)
        if arity != 0:
            return apy_kind_method_of(obj, arity, want, bind)
    # A RANGE IS FALSE WHEN IT IS EMPTY and says so with `__bool__` rather
    # than through `__len__`, which is the one place it parts company with
    # the other walkable kinds.
    if is_range and apy_name_is(want, rodata(b"__bool__\0")):
        return apy_kind_method_of(obj, 1, rodata(b"__bool__\0"), bind)
    # AND None SAYS SO TOO, which is the only other kind here that writes the
    # method out rather than being read through `__len__`.
    if apy_kind_is(obj, apy_none_kind()):
        if apy_name_is(want, rodata(b"__bool__\0")):
            return apy_kind_method_of(obj, 1, rodata(b"__bool__\0"), bind)
    # THE IN-PLACE OPERATORS, which belong to the MUTABLE kinds and to no
    # other: `(1,).__iadd__` is an AttributeError in Python and `[1].__iadd__`
    # is the method that makes `xs += ys` change the list every other name for
    # it also sees. A tuple falls through to `+` and has nothing to name.
    if is_list or writable_bytes:
        if apy_name_is(want, rodata(b"__iadd__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__imul__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
    # A DICT HAS ONLY `__ior__`; a set has the four. A FROZENSET HAS NONE,
    # which is why this asks `is_set` and not `sset`.
    if is_dict or is_set:
        if apy_name_is(want, rodata(b"__ior__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
    if is_set:
        if apy_name_is(want, rodata(b"__iand__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__isub__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__ixor__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
    # `%` ON TEXT IS FORMATTING, not arithmetic -- which is why it belongs to
    # str and bytes and to no other sequence.
    if is_str or is_bytes:
        if apy_name_is(want, rodata(b"__mod__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
    # `d | e` MERGES TWO DICTS, and the same four spellings are a set's
    # operations. A `collections.abc` mixin composes them by name, which is
    # the reading that needed them to exist as values.
    if is_dict:
        if apy_name_is(want, rodata(b"__or__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__ror__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
    if sset:
        if apy_name_is(want, rodata(b"__or__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__and__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__sub__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__xor__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__ror__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__rand__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__rsub__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__rxor__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
    if is_dict:
        if apy_name_is(want, rodata(b"keys\0")):
            return apy_kind_method_of(obj, 1, want, bind)
        if apy_name_is(want, rodata(b"values\0")):
            return apy_kind_method_of(obj, 1, want, bind)
        if apy_name_is(want, rodata(b"items\0")):
            return apy_kind_method_of(obj, 1, want, bind)
    if seq or text:
        # A WINDOW, ON THE KINDS THAT TAKE ONE. `"ab".count("a", 0, 2)` is a
        # call CPython answers and `[1].count(1, 2)` is not, so the bounds
        # come from the table rather than from this line.
        if apy_name_is(want, rodata(b"index\0")):
            ranged: ptr = apy_kind_method_ranged_of(obj, want, bind)
            if ranged:
                return ranged
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"count\0")):
            counted: ptr = apy_kind_method_ranged_of(obj, want, bind)
            if counted:
                return counted
            return apy_kind_method_of(obj, 2, want, bind)
    if is_list:
        if apy_name_is(want, rodata(b"append\0")):
            return apy_kind_method_of(obj, 2, rodata(b"append\0"), bind)
        if apy_name_is(want, rodata(b"insert\0")):
            return apy_kind_method_of(obj, 3, rodata(b"insert\0"), bind)
    if is_set:
        if apy_name_is(want, rodata(b"add\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"discard\0")):
            return apy_kind_method_of(obj, 2, want, bind)
    if sset and apy_name_is(want, rodata(b"isdisjoint\0")):
        return apy_kind_method_of(obj, 2, rodata(b"isdisjoint\0"), bind)
    # THE STATICS A TYPE CARRIES AND A VALUE CARRIES TOO. `bytes.fromhex`
    # and `b"".fromhex` are ONE method in Python -- an implicit
    # staticmethod, so the receiver decides which body and is otherwise
    # ignored -- and every one of these was reachable only as a written call
    # on the type's own name. A program holding the VALUE, or reaching the
    # method through `getattr`, found nothing.
    if is_str:
        if apy_name_is(want, rodata(b"maketrans\0")):
            return apy_kind_method_opt(obj, 4, 2, want, bind)
        # `format` TAKES WHATEVER IT IS GIVEN, positionally and by keyword,
        # which is why it is not in the method table: no row there can say
        # "any count". The WRITTEN form is lowered at the call site; this is
        # the same call reached by name.
        if apy_name_is(want, rodata(b"format\0")):
            return apy_kind_method_var(obj, want, bind)
    if is_bytes:
        if apy_name_is(want, rodata(b"maketrans\0")):
            return apy_kind_method_of(obj, 3, want, bind)
        if apy_name_is(want, rodata(b"fromhex\0")):
            return apy_kind_method_of(obj, 2, want, bind)
    if is_float:
        if apy_name_is(want, rodata(b"fromhex\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"from_number\0")):
            return apy_kind_method_of(obj, 2, want, bind)
    if is_complex:
        if apy_name_is(want, rodata(b"from_number\0")):
            return apy_kind_method_of(obj, 2, want, bind)
    if is_dict:
        if apy_name_is(want, rodata(b"fromkeys\0")):
            return apy_kind_method_opt(obj, 3, 1, want, bind)
    if is_int:
        if apy_name_is(want, rodata(b"from_bytes\0")):
            return apy_kind_method_opt(obj, 3, 1, want, bind)
    if apy_name_is(want, rodata(b"__buffer__\0")):
        if is_bytes or is_mview:
            return apy_kind_method_of(obj, 2, rodata(b"__buffer__\0"),
                                      bind)
    # PEP 688's OTHER HALF, and the one bytearray internal a program can
    # read. `__release_buffer__` is what a `with memoryview(...)` block calls
    # on the way out, and only the buffer that can be RESIZED carries it --
    # bytes cannot move under a view and has nothing to be told.
    if writable_bytes:
        if apy_name_is(want, rodata(b"__release_buffer__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__alloc__\0")):
            return apy_kind_method_of(obj, 1, want, bind)
    # WHAT `copy` AND `pickle` REBUILD A VALUE FROM. Every immutable builtin
    # answers `(self,)` -- a complex answers its two halves -- and a mutable
    # one has none at all, because anything it handed back would be shared
    # with the copy rather than rebuild it.
    if apy_name_is(want, rodata(b"__getnewargs__\0")):
        if is_str or is_int or is_float or is_complex:
            return apy_kind_method_of(obj, 1, want, bind)
        if seq:
            if is_list == 0:
                return apy_kind_method_of(obj, 1, want, bind)
        if is_bytes:
            if writable_bytes == 0:
                return apy_kind_method_of(obj, 1, want, bind)
    # `bytes(x)` ASKS `x` FOR ITSELF FIRST, and bytes is the kind that
    # answers -- a bytearray does not, which is why `bytes(ba)` copies.
    if apy_name_is(want, rodata(b"__bytes__\0")):
        if is_bytes:
            if writable_bytes == 0:
                return apy_kind_method_of(obj, 1, want, bind)
    # `list[int]` REACHED BY NAME. The written form is a subscript the
    # frontend lowers; this is the method behind it, which `typing` calls
    # directly when it parameterises a container. TEXT HAS NONE: `str[int]`
    # is a TypeError in Python and the attribute is absent.
    if apy_name_is(want, rodata(b"__class_getitem__\0")):
        if seq or is_dict or sset:
            return apy_kind_method_of(obj, 2, want, bind)
        # AND `enumerate[int]`, which is the one CURSOR CPython gives one
        # to: `zip`, `map` and `filter` have none, and `dir()` over each
        # says so.
        if is_iter:
            emode: i64 = i64(load(i32, offset(obj, apy_it_mode_offset())))
            if emode == apy_it_enumerate():
                return apy_kind_method_of(obj, 2, want, bind)
        # AND `generator[int]`, which a generic annotation on an `async def`
        # or a `Generator[...]`-shaped alias reaches by name.
        if is_gen:
            return apy_kind_method_of(obj, 2, want, bind)
    # A GENERATOR HAS A FINALISER and is the only builtin value here that
    # does: closing an abandoned one runs its `finally` blocks, which is why
    # CPython gives the type a `__del__` where a list has none. Answerable
    # rather than useful -- there is nothing for a program to do by calling
    # it -- but `dir(g)` lists it and a list that lies is worse than the
    # empty one this replaced.
    if apy_name_is(want, rodata(b"__del__\0")):
        if is_gen:
            return apy_kind_method_of(obj, 1, want, bind)
    # AND THE PROTOCOL EACH OF THE THREE ANSWERS TO. A coroutine is what
    # `await` walks and carries `__await__`; an async generator is what
    # `async for` walks and carries the two halves of that protocol; a plain
    # generator has neither, which is what `dir()` over each says.
    if is_gen:
        gcoro: i64 = i64(load(i32, offset(obj, apy_g_coro_offset())))
        gagen: i64 = i64(load(i32, offset(obj, apy_g_agen_offset())))
        if apy_name_is(want, rodata(b"__await__\0")):
            if gcoro:
                if not gagen:
                    return apy_kind_method_of(obj, 1, want, bind)
        if gagen:
            if apy_name_is(want, rodata(b"__aiter__\0")):
                return apy_kind_method_of(obj, 1, want, bind)
            if apy_name_is(want, rodata(b"__anext__\0")):
                return apy_kind_method_of(obj, 1, want, bind)
    # `it.__setstate__(i)` -- WHERE THE WALK IS, written rather than read.
    # WHICH CURSORS CARRY IT is CPython's own `dir()`, transcribed: the
    # sequence walks -- a list, tuple, str, bytes or range, forward or
    # reversed, and a reversed memoryview, which CPython calls a plain
    # `reversed` -- and `zip` and `map`. A set, a dict's three walks, a
    # `callable_iterator`, a forward `memory_iterator`, `enumerate` and
    # `filter` do NOT. See the C's `apy_cursor_setstate_p`, which draws the
    # same line.
    if apy_name_is(want, rodata(b"__setstate__\0")):
        if is_iter:
            smode: i64 = i64(load(i32, offset(obj, apy_it_mode_offset())))
            if smode == apy_it_map() or smode == apy_it_zip():
                return apy_kind_method_of(obj, 2, want, bind)
            if smode == apy_it_plain() or smode == apy_it_rev():
                named: i64 = i64(load(i32,
                                      offset(obj, apy_it_named_offset())))
                rev: i64 = 0
                if named >= apy_it_revof():
                    rev = 1
                    named = named - apy_it_revof()
                if rev:
                    if named == apy_mview_kind():
                        return apy_kind_method_of(obj, 2, want, bind)
                if named == apy_list_kind():
                    return apy_kind_method_of(obj, 2, want, bind)
                if named == apy_tuple_kind():
                    return apy_kind_method_of(obj, 2, want, bind)
                if named == apy_str_kind():
                    return apy_kind_method_of(obj, 2, want, bind)
                if named == apy_bytes_kind():
                    return apy_kind_method_of(obj, 2, want, bind)
                if named == apy_range_kind():
                    return apy_kind_method_of(obj, 2, want, bind)
    # WHICH FLOATING-POINT FORMAT THIS BUILD USES. One answer, and a float is
    # the only kind ever asked.
    if is_float:
        if apy_name_is(want, rodata(b"__getformat__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
    # THE REFLECTED `%`, which text carries and always refuses -- `1 % "a"`
    # is a TypeError the OPERATOR raises after this answers NotImplemented.
    # A NUMBER'S IS ELSEWHERE: `apy_number_arity` already has it.
    if text:
        if apy_name_is(want, rodata(b"__rmod__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
    # THE TWO WAYS A VIEW HANDS ITS CONTENTS OVER, and the reason a program
    # makes one at all: `mv.tobytes()` copies them out and `mv.tolist()`
    # reads them as numbers. Neither existed, so a view could be indexed and
    # sliced and never emptied.
    # WHAT A VIEW CARRIES BESIDE ITS TWO CONVERSIONS. `hex`, `count` and
    # `index` read the bytes it shows -- a memoryview IS a sequence in Python
    # -- and the three subscript dunders are the methods behind the `m[i]` a
    # program writes. `__delitem__` exists and always refuses, which is not
    # the same claim as having no such method.
    if is_mview:
        # THE SAME BOUNDS A BYTES RECEIVER HAS, because a view's `hex` and
        # `index` read the bytes it shows.
        if apy_name_is(want, rodata(b"hex\0")):
            hexed: ptr = apy_kind_method_ranged_of(obj, want, bind)
            if hexed:
                return hexed
            return apy_kind_method_opt(obj, 2, 1, want, bind)
        if apy_name_is(want, rodata(b"count\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"index\0")):
            viewed: ptr = apy_kind_method_ranged_of(obj, want, bind)
            if viewed:
                return viewed
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__delitem__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__class_getitem__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__release_buffer__\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__setitem__\0")):
            return apy_kind_method_of(obj, 3, want, bind)
        # HANDING THE BUFFER BACK, and the `with` block that does it for a
        # program. `__exit__` takes the three exception slots whether or not
        # there was one, which is why it declares four.
        if apy_name_is(want, rodata(b"release\0")):
            return apy_kind_method_of(obj, 1, want, bind)
        if apy_name_is(want, rodata(b"toreadonly\0")):
            return apy_kind_method_of(obj, 1, want, bind)
        if apy_name_is(want, rodata(b"cast\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"_from_flags\0")):
            return apy_kind_method_of(obj, 3, want, bind)
        if apy_name_is(want, rodata(b"__enter__\0")):
            return apy_kind_method_of(obj, 1, want, bind)
        if apy_name_is(want, rodata(b"__exit__\0")):
            return apy_kind_method_of(obj, 4, want, bind)
    if is_mview:
        if apy_name_is(want, rodata(b"tobytes\0")):
            return apy_kind_method_of(obj, 1, rodata(b"tobytes\0"), bind)
        if apy_name_is(want, rodata(b"tolist\0")):
            return apy_kind_method_of(obj, 1, rodata(b"tolist\0"), bind)
    if is_range:
        if apy_name_is(want, rodata(b"start\0")):
            return apy_from_int(load(i64, offset(obj,
                                                 apy_rg_start_offset())))
        if apy_name_is(want, rodata(b"stop\0")):
            return apy_from_int(load(i64, offset(obj, apy_rg_stop_offset())))
        if apy_name_is(want, rodata(b"step\0")):
            return apy_from_int(load(i64, offset(obj, apy_rg_step_offset())))
        if apy_name_is(want, rodata(b"index\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"count\0")):
            return apy_kind_method_of(obj, 2, want, bind)
        if apy_name_is(want, rodata(b"__len__\0")):
            return apy_kind_method_of(obj, 1, rodata(b"__len__\0"), bind)
        if apy_name_is(want, rodata(b"__iter__\0")):
            return apy_kind_method_of(obj, 1, rodata(b"__iter__\0"), bind)
        if apy_name_is(want, rodata(b"__contains__\0")):
            return apy_kind_method_of(obj, 2, rodata(b"__contains__\0"),
                                      bind)
        if apy_name_is(want, rodata(b"__getitem__\0")):
            return apy_kind_method_of(obj, 2, rodata(b"__getitem__\0"),
                                      bind)
    # AND THE WHOLE METHOD TABLE, generated. Everything above answers a
    # PROTOCOL name or a field; this answers the ORDINARY methods, which
    # existed only as calls the frontend lowered and so could not be reached
    # by NAME at all -- `getattr("abc", "upper")` was an AttributeError about
    # a method the object plainly has.
    #
    # LAST, so every arm above still decides first: a few names mean
    # different things to different kinds and are split there rather than in
    # a table keyed by name.
    #
    # THE BOUNDS ARE THIS KIND'S where the words table knows them -- see
    # `apy_kind_method_ranged_of`. The arity table decides for a name it does
    # not carry, and for the six set methods whose range has no upper end.
    tabled: ptr = apy_kind_method_ranged_of(obj, want, bind)
    if tabled:
        return tabled
    packed: i64 = apy_kind_meth_arity_of(want, apy_kind_bit_of(obj))
    if packed:
        return apy_kind_method_opt(obj, packed >> 8, packed & 255, want, bind)
    return ptr(0)


def apy_kind_attr(obj: ptr, want: ptr) -> ptr:
    """`apy_kind_attr_of` with the result BOUND, which is what a read wants."""
    return apy_kind_attr_of(obj, want, 1)


def apy_kind_prototype(type_name: ptr) -> ptr:
    """An empty value of the kind `type_name` names, or null.

    WHAT A BUILTIN TYPE USED AS A VALUE ANSWERS ATTRIBUTES FROM. `list.append`
    has no list to ask, so one is made -- empty, thrown away, and only ever
    used to decide which methods that kind has.
    """
    if apy_name_is(type_name, rodata(b"list\0")):
        return apy_list_new(1)
    if apy_name_is(type_name, rodata(b"tuple\0")):
        return apy_tuple_new(1)
    if apy_name_is(type_name, rodata(b"dict\0")):
        return apy_dict_new(1)
    if apy_name_is(type_name, rodata(b"set\0")):
        return apy_set_new(1)
    if apy_name_is(type_name, rodata(b"frozenset\0")):
        return apy_frozenset_new(1)
    if apy_name_is(type_name, rodata(b"str\0")):
        return apy_from_cstr(rodata(b"\0"))
    if apy_name_is(type_name, rodata(b"bytes\0")):
        return apy_bytes_literal(rodata(b"\0"), 0)
    if apy_name_is(type_name, rodata(b"int\0")):
        return apy_from_int(0)
    if apy_name_is(type_name, rodata(b"bool\0")):
        return apy_from_int(0)
    if apy_name_is(type_name, rodata(b"float\0")):
        return apy_from_float(f64(0))
    if apy_name_is(type_name, rodata(b"list_iterator\0")):
        return apy_cursor_proto_of(apy_it_plain(), apy_list_kind())
    if apy_name_is(type_name, rodata(b"list_reverseiterator\0")):
        return apy_cursor_proto_of(
            apy_it_rev(), apy_list_kind() + apy_it_revof())
    if apy_name_is(type_name, rodata(b"tuple_iterator\0")):
        return apy_cursor_proto_of(apy_it_plain(), apy_tuple_kind())
    if apy_name_is(type_name, rodata(b"reversed\0")):
        return apy_cursor_proto_of(
            apy_it_rev(), apy_tuple_kind() + apy_it_revof())
    if apy_name_is(type_name, rodata(b"str_iterator\0")):
        return apy_cursor_proto_of(apy_it_plain(), apy_str_kind())
    if apy_name_is(type_name, rodata(b"str_ascii_iterator\0")):
        return apy_cursor_proto_of(apy_it_plain(), apy_str_kind())
    if apy_name_is(type_name, rodata(b"bytes_iterator\0")):
        return apy_cursor_proto_of(apy_it_plain(), apy_bytes_kind())
    if apy_name_is(type_name, rodata(b"bytearray_iterator\0")):
        return apy_cursor_proto_of(apy_it_plain(), apy_bytes_kind())
    if apy_name_is(type_name, rodata(b"range_iterator\0")):
        return apy_cursor_proto_of(apy_it_plain(), apy_range_kind())
    if apy_name_is(type_name, rodata(b"set_iterator\0")):
        return apy_cursor_proto_of(apy_it_plain(), apy_set_kind())
    if apy_name_is(type_name, rodata(b"memory_iterator\0")):
        return apy_cursor_proto_of(apy_it_plain(), apy_mview_kind())
    if apy_name_is(type_name, rodata(b"dict_keyiterator\0")):
        return apy_cursor_proto_of(apy_it_plain(), apy_dict_kind())
    if apy_name_is(type_name, rodata(b"dict_valueiterator\0")):
        return apy_cursor_proto_of(
            apy_it_plain(), apy_it_viewed() + apy_part_values())
    if apy_name_is(type_name, rodata(b"dict_itemiterator\0")):
        return apy_cursor_proto_of(
            apy_it_plain(), apy_it_viewed() + apy_part_items())
    if apy_name_is(type_name, rodata(b"dict_reversekeyiterator\0")):
        return apy_cursor_proto_of(
            apy_it_rev(), apy_dict_kind() + apy_it_revof())
    if apy_name_is(type_name, rodata(b"dict_reversevalueiterator\0")):
        return apy_cursor_proto_of(
            apy_it_rev(), apy_it_viewed() + apy_part_values() + apy_it_revof())
    if apy_name_is(type_name, rodata(b"dict_reverseitemiterator\0")):
        return apy_cursor_proto_of(
            apy_it_rev(), apy_it_viewed() + apy_part_items() + apy_it_revof())
    if apy_name_is(type_name, rodata(b"callable_iterator\0")):
        return apy_cursor_proto_of(apy_it_call(), apy_it_callable())
    if apy_name_is(type_name, rodata(b"enumerate\0")):
        return apy_cursor_proto_of(apy_it_enumerate(), apy_list_kind())
    if apy_name_is(type_name, rodata(b"zip\0")):
        return apy_cursor_proto_of(apy_it_zip(), apy_list_kind())
    if apy_name_is(type_name, rodata(b"map\0")):
        return apy_cursor_proto_of(apy_it_map(), apy_list_kind())
    if apy_name_is(type_name, rodata(b"filter\0")):
        return apy_cursor_proto_of(apy_it_filter(), apy_list_kind())
    # A GENERATOR PROTOTYPE IS A GENERATOR WITH NO STEP, for the reason a
    # cursor prototype has no source: nothing is ever run, and what the type
    # carries is all that is asked of it.
    if apy_name_is(type_name, rodata(b"generator\0")):
        return apy_gen_new(ptr(0), 0)
    if apy_name_is(type_name, rodata(b"coroutine\0")):
        return apy_coro_mark(apy_gen_new(ptr(0), 0))
    if apy_name_is(type_name, rodata(b"async_generator\0")):
        return apy_agen_mark(apy_gen_new(ptr(0), 0))
    # A WRAPPER PROTOTYPE WRAPS NOTHING, for the same reason: what the type
    # carries is all that is asked of it. The flag is SET HERE rather than
    # through the C's `apy_coro_wrapper`, because the ported runtime reaches
    # nothing in the C but the `_slow` halves of a split.
    if apy_name_is(type_name, rodata(b"coroutine_wrapper\0")):
        made: ptr = apy_coro_mark(apy_gen_new(ptr(0), 0))
        if made:
            store(i32, i32(1), offset(made, apy_g_wrapper_offset()))
        return made
    return ptr(0)


def apy_cursor_proto_of(mode: i64, named: i64) -> ptr:
    """A CURSOR WITH NO SOURCE, which is all `apy_kind_attr_of` reads of one:
    the kind it is, the mode it walks in and what it is named after.

    Nothing here is ever stepped -- a prototype exists to be asked which
    attributes its kind carries and is then thrown away -- so there is
    nothing for a source to be. The C twin is `apy_cursor_protos`.
    """
    it: ptr = apy_cursor_of(ptr(0), ptr(0), mode, 0)
    if not it:
        return it
    store(i32, i32(named), offset(it, apy_it_named_offset()))
    return it


def apy_no_attribute(obj: ptr, name: ptr) -> ptr:
    """The last thing attribute lookup tries, and the error if it fails.

    A BUILTIN TYPE NAME IS ASKED THROUGH A PROTOTYPE: `list.append` is a real
    attribute, and the only way to answer it is to make an empty list and ask
    THAT what it has. Nothing else knows which methods a kind carries.
    """
    want: ptr = ptr(load(u64, offset(name, apy_str_ptr_offset())))
    found: ptr = apy_kind_attr(obj, want)
    if found:
        return found
    if i64(load(i32, offset(obj, 0))) == apy_func_kind():
        if load(i32, offset(obj, apy_fn_is_type_offset())):
            proto: ptr = apy_kind_prototype(ptr(load(u64, offset(
                ptr(load(u64, offset(obj, apy_fn_name_offset()))),
                apy_str_ptr_offset()))))
            if proto:
                # `__class_getitem__` IS A CLASSMETHOD and binds the TYPE,
                # which is why `list.__class_getitem__(int)` takes only the
                # key where `list.append(xs, 9)` takes a receiver first. The
                # prototype's own is already bound to it and reads
                # `type(...)` of what it was bound to, which IS the type
                # asked. Unbound, the call was `expected 2 arguments, got 1`
                # about a spelling CPython answers.
                bindit: i64 = 0
                if apy_cstr_eq(want, rodata(b"__class_getitem__\0")):
                    bindit = 1
                found = apy_kind_attr_of(proto, want, 0)
                if found:
                    if bindit:
                        # BOUND TO THE TYPE ITSELF and not to the prototype,
                        # because the receiver is what the repr names --
                        # CPython's reads `of type object at ...` -- and the
                        # body reads it too: the `__class_getitem__` arm
                        # takes a type straight as the alias's origin.
                        #
                        # AND A BOUND ONE IS NOT A DESCRIPTOR:
                        # `list.append` is a `method_descriptor` and
                        # `list.__class_getitem__` a
                        # `builtin_function_or_method`, so the stamp below
                        # is not reached.
                        return apy_bind_of(found, obj)
                if found:
                    # REACHED OFF THE TYPE MAKES IT A DESCRIPTOR, which
                    # CPython names, reprs and qualifies differently from a
                    # function: `list.append` is a `method_descriptor`,
                    # prints as `<method 'append' of 'list' objects>` and
                    # answers `list.append` to `__qualname__`. This is the
                    # only place that knows WHICH type handed it over, and
                    # the cell `apy_kind_attr_of` builds is fresh per call.
                    if i64(load(i32, offset(found, 0))) == apy_func_kind():
                        buf: ptr = apy_fmt_scratch()
                        at: i64 = apy_cstr_into(
                            buf, 0, 200,
                            ptr(load(u64, offset(
                                ptr(load(u64, offset(
                                    obj, apy_fn_name_offset()))),
                                apy_str_ptr_offset()))))
                        at = apy_cstr_into(buf, at, 200, rodata(b".\0"))
                        at = apy_cstr_into(buf, at, 200, want)
                        store(u64, u64(apy_str_copy_bytes(buf, at)),
                              offset(found, apy_fn_qualname_offset()))
                        store(i32, 1, offset(found, apy_fn_descr_offset()))
                    return found
    # A TYPE IS NAMED, NOT DESCRIBED. CPython says `type object 'list' has no
    # attribute 'nope'` where a VALUE gets `'int' object has no attribute
    # 'nope'` -- the type's own name is what the reader needs, and
    # `apy_kind_name_of` one is only ever the word `type`.
    if i64(load(i32, offset(obj, 0))) == apy_func_kind():
        if load(i32, offset(obj, apy_fn_is_type_offset())):
            return apy_raise_fmt(
                rodata(b"AttributeError\0"),
                rodata(b"type object '%s' has no attribute '%s'\0"),
                ptr(load(u64, offset(
                    ptr(load(u64, offset(obj, apy_fn_name_offset()))),
                    apy_str_ptr_offset()))), want)
    return apy_raise_fmt(
        rodata(b"AttributeError\0"),
        rodata(b"'%s' object has no attribute '%s'\0"),
        apy_kind_name_of(obj), want)


def apy_mro_entries(written: ptr, bases: ptr) -> ptr:
    """PEP 560: what a non-class written as a base RESOLVES to.

    `class C(Generic[T])` names something that is not a class, and
    `__mro_entries__` is how it says what should stand in its place. A
    generic alias answers `Generic`; anything without the hook is simply not
    a base.

    THE FIRST ENTRY IS TAKEN and the rest dropped, which is a stated
    simplification: the full protocol substitutes the whole sequence into the
    base list, and one base is what every use here needs.
    """
    if i64(load(i32, offset(written, 0))) == apy_type_kind():
        return written
    if i64(load(i32, offset(written, 0))) != apy_inst_kind():
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"bases must be types, not '%s'%s\0"),
            apy_kind_name_of(written), rodata(b"\0"))
    hook: ptr = apy_class_find_of(
        ptr(load(u64, offset(written, apy_o_cls_offset()))),
        apy_name_of(rodata(b"__mro_entries__\0")))
    if not hook:
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"bases must be types, not '%s'%s\0"),
            apy_kind_name_of(written), rodata(b"\0"))
    one: ptr = alloca(8)
    store(u64, u64(bases), one)
    got: ptr = apy_call(apy_bind_of(hook, written), one, 1)
    if not got:
        return ptr(0)
    if apy_is_seq_of(got):
        if load(i64, offset(got, apy_q_n_offset())) > 0:
            return ptr(load(u64, ptr(load(u64, offset(
                got, apy_q_items_offset())))))
    return apy_object_class()


# -- reading an attribute off an instance -----------------------------------
#
# SPLIT, and the line is drawn at the KIND. `apy_default_getattr` answers ten
# of them -- a class, a super, a slice, a property, a function, a generic
# alias, a generator, a memoryview, a complex, an exception -- and each is a
# page of its own field names. This half answers the one a program actually
# spends its time in: `self.x` on an instance.
#
# EVERYTHING ELSE GOES BACK TO THE C, which is the same bargain
# `apy_num_order_of` makes for bigs and `apy_call` makes for keyword matching:
# the shape that runs in a loop is ported, and the shapes that run once keep
# working exactly as they did.


def apy_inst_getattr(obj: ptr, name: ptr) -> ptr:
    """`object.__getattribute__(obj, name)` for an instance.

    THIS IS THE DEFAULT LOOKUP, not the entry point. A class overriding
    `__getattribute__` is asked first by `apy_getattr` and reaches this by
    calling the default explicitly, which is the only way out of the
    recursion.

    A DATA DESCRIPTOR ON THE CLASS BEATS THE INSTANCE DICT, and it is the one
    place the "instance wins" rule does not hold -- it is what makes a
    `property` a property: `c.v = 4` runs its setter and the instance dict
    never gets a `v` to shadow it with. A NON-data descriptor loses to the
    dict instead, which is how a method can be shadowed by an attribute of
    the same name.

    A FUNCTION FOUND ON THE CLASS BINDS AND ANYTHING ELSE DOES NOT. That
    single test is the whole of the "methods take self" rule.

    A HELD BUILTIN IS ASKED AFTER THE CLASS AND BEFORE `__getattr__`. The
    class body wins -- a `Counter` defining `update` shadows `dict.update` --
    and the fallback loses, because in CPython these arrive through the MRO,
    which is consulted first. THE MISS IS NOT THE ANSWER: a name neither has
    must still reach `__getattr__`, so the AttributeError the delegation
    raised is cleared rather than reported.
    """
    cls: ptr = ptr(load(u64, offset(obj, apy_o_cls_offset())))
    d: ptr = ptr(load(u64, offset(obj, apy_o_dict_offset())))
    found: ptr = apy_class_find_of(cls, name)
    if found:
        if apy_is_data_descriptor_of(found):
            return apy_descr_get_of(found, obj, cls)
    at: i64 = apy_dict_find_of(d, name)
    if at >= 0:
        vals: ptr = ptr(load(u64, offset(d, apy_d_vals_offset())))
        return ptr(load(u64, offset(vals, at * apy_value_size())))
    if found:
        if apy_is_descriptor_of(found):
            return apy_descr_get_of(found, obj, cls)
        if i64(load(i32, offset(found, 0))) == apy_func_kind():
            return apy_bind_of(found, obj)
        return found
    want: ptr = ptr(load(u64, offset(name, apy_str_ptr_offset())))
    if apy_cstr_eq(want, rodata(b"__class__\0")):
        return cls
    if apy_cstr_eq(want, rodata(b"__dict__\0")):
        # ABSENT UNDER `__slots__`, which is the point of declaring it --
        # `hasattr(p, "__dict__")` is how a program checks. The dict ITSELF
        # and not a copy: `obj.__dict__["x"] = 1` is how an attribute is set
        # dynamically, and a copy would accept the write and lose it.
        if not apy_slot_allows_of(cls, apy_from_cstr(
                rodata(b"__dict__\0"))):
            return apy_no_attribute(obj, name)
        return d
    held: ptr = ptr(load(u64, offset(obj, apy_o_held_offset())))
    if held:
        got: ptr = apy_default_getattr(held, name)
        if got:
            return got
        if apy_cstr_eq(apy_err_kind(), rodata(b"AttributeError\0")):
            apy_error_clear()
        else:
            return got
    hook: ptr = apy_class_find_of(cls,
                                  apy_name_of(rodata(b"__getattr__\0")))
    if hook:
        one: ptr = alloca(8)
        store(u64, u64(name), one)
        return apy_call(apy_bind_of(hook, obj), one, 1)
    return apy_no_attribute(obj, name)


def apy_default_getattr(obj: ptr, name: ptr) -> ptr:
    """`object.__getattribute__` -- the default read, for every kind."""
    if i64(load(i32, offset(obj, 0))) == apy_inst_kind():
        return apy_inst_getattr(obj, name)
    return apy_default_getattr_slow(obj, name)


def apy_getattr(obj: ptr, name: ptr) -> ptr:
    """`obj.name` -- the entry point, with `__getattribute__` in front of it.

    `__getattribute__` INTERCEPTS EVERYTHING, which is the whole difference
    from `__getattr__`: one is asked before the lookup and sees every name,
    the other only after it has missed. A class overriding it reaches the
    ordinary rules by calling `object.__getattribute__` explicitly, which is
    the only way out of the recursion.

    ONLY THE NAME IS PASSED. The hook is bound to the receiver already, so
    handing it `obj` as well would put the object in front of its own
    argument and every name would arrive one place late.
    """
    # A str SUBCLASS IS AN ATTRIBUTE NAME. `getattr("abc", S("upper"))` is
    # ordinary Python; the instance reached the lookup as a name and the text
    # read out of it was whatever lay at the instance's address. Mirrors the
    # C's `apy_text_like`.
    nheld: ptr = ptr(0)
    if i64(load(i32, offset(name, 0))) == apy_inst_kind():
        nheld = ptr(load(u64, offset(name, apy_o_held_offset())))
    if nheld:
        if i64(load(i32, offset(nheld, 0))) == apy_str_kind():
            name = nheld
    if i64(load(i32, offset(obj, 0))) == apy_inst_kind():
        hook: ptr = apy_class_find_of(
            ptr(load(u64, offset(obj, apy_o_cls_offset()))),
            apy_name_of(rodata(b"__getattribute__\0")))
        if hook:
            one: ptr = alloca(8)
            store(u64, u64(name), one)
            return apy_call(apy_bind_of(hook, obj), one, 1)
    return apy_default_getattr(obj, name)


def apy_getattr_default(obj: ptr, name: ptr, fallback: ptr) -> ptr:
    """`getattr(obj, name, default)`.

    ONLY AN AttributeError IS SWALLOWED. A `__getattr__` that raises
    something else -- a ValueError from a computed property, say -- must
    reach the caller: `getattr(x, "n", 0)` asks whether the attribute is
    there, not whether reading it worked.
    """
    got: ptr = apy_getattr(obj, name)
    if got:
        return got
    if apy_error_matches(apy_from_cstr(rodata(b"AttributeError\0"))):
        apy_error_clear()
        return fallback
    return ptr(0)


def apy_hasattr(v: ptr, name: ptr) -> ptr:
    """`hasattr(obj, name)`.

    ONLY AN AttributeError IS SWALLOWED, which is Python's rule since 3.2 --
    before that `hasattr` caught everything, and the change was made because
    a property raising a ValueError read as "no such attribute" and hid a
    real failure. This caught everything too, and answered False where
    CPython propagates.
    """
    got: ptr = apy_getattr(v, name)
    if got:
        return apy_from_bool(1)
    if apy_error_matches(apy_from_cstr(rodata(b"AttributeError\0"))):
        apy_error_clear()
        return apy_from_bool(0)
    return ptr(0)


def apy_typing_form_slot() -> ptr:
    """The cache of one interned form per name."""
    return reserve("apy_typing_form_ir", 8)


def apy_typing_form(name: ptr) -> ptr:
    """One of the interned typing forms -- `Literal`, `TypeGuard`, their kin.

    INTERNED BY NAME, because a program compares them by identity: `x is
    Literal` is how a checker-shaped library asks which form it holds, and a
    fresh instance per mention would answer False.
    """
    slot: ptr = apy_typing_form_slot()
    seen: ptr = ptr(load(u64, slot))
    if not seen:
        seen = apy_dict_new(8)
        if not seen:
            return seen
        store(u64, u64(seen), slot)
    found: ptr = apy_dict_get_or(seen, name, ptr(0))
    if found:
        return found
    cls: ptr = apy_special_form_class()
    if not cls:
        return cls
    o: ptr = apy_instance_new(cls)
    if not o:
        return o
    apy_setattr(o, apy_from_cstr(rodata(b"_name\0")), name)
    if apy_error_occurred():
        return ptr(0)
    apy_dict_set(seen, name, o)
    return o
