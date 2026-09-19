"""The object runtime, in C: sequences, and printing.

ONE PART OF ONE TRANSLATION UNIT. `c/__init__.py` concatenates
these in order and the result is the file it always was, so a
definition here may rely on anything in an earlier part and
nothing in a later one. Sections, in order:
  * sequences
  * printing
"""

C = r"""/* --- sequences --------------------------------------------------------- */
static int apy_is_int_like(apy_value v);
static int apy_eq_raw(apy_value a, apy_value b);
static apy_value apy_text(apy_value v, int quoted);
/* An instance of a class extending str or bytes, AS the value it holds.
   Defined in `_strings.c`; declared here because `apy_print_with` below is
   the first thing to need it. */
static apy_value apy_text_like(apy_value v);

static const char *apy_exc_shown(const char *name);
APY_API apy_value apy_kind_name_of(apy_value v);

/* `str(e)` is the ARGUMENT alone and `repr(e)` is `ValueError('x')`.
   Printing an exception shows its message, which is why the two differ
   here and not for any other kind. */
static int apy_is_seq(apy_value v);

/* `[Errno 2] No such file`, and `: 'f.txt'` when a filename came too.
   0 when this is not that shape, so the caller renders the ordinary way.

   TWO OR THREE ARGUMENTS ONLY: `OSError('plain')` is an ordinary exception
   with an ordinary message and `OSError()` has nothing to say at all --
   CPython puts the errno form on exactly this shape and leaves the rest.

   A WALK UP THE HIERARCHY AND NOT A NAME TEST, because the whole family
   arrives under its own name: opening a missing file raises
   FileNotFoundError, and `errno` maps a dozen others onto their own classes.
   Asking whether the name IS `OSError` answers no for every one a real
   program catches. */
/* The parent chain is defined far below, and this walks it. */
static const char *apy_exc_parent(const char *name);
static int apy_exc_is_os(apy_value v) {
    const char *at = O(v)->v.e.name;
    while (at) {
        if (strcmp(at, "OSError") == 0) return 1;
        at = apy_exc_parent(at);
    }
    return 0;
}
static apy_value apy_os_text(apy_value v, apy_value argv) {
    int64_t n = O(argv)->v.q.n, out;
    apy_value a0, msg, tail = 0;
    char *buf;
    size_t room;
    (void)v;
    if (n < 2 || n > 3) return 0;
    a0 = O(argv)->v.q.items[0];
    if (O(a0)->kind != APY_INT_K) return 0;
    msg = apy_text(O(argv)->v.q.items[1], 0);
    if (!msg) return 0;
    if (n == 3) {
        tail = apy_text(O(argv)->v.q.items[2], 1);
        if (!tail) return 0;
    }
    room = (size_t)O(msg)->v.s.n + 32
         + (size_t)(tail ? O(tail)->v.s.n + 2 : 0);
    buf = (char *)malloc(room + 1);
    out = (int64_t)sprintf(buf, "[Errno %lld] ", (long long)O(a0)->v.i);
    memcpy(buf + out, O(msg)->v.s.p, (size_t)O(msg)->v.s.n);
    out += O(msg)->v.s.n;
    /* THE FILENAME IS QUOTED AND THE MESSAGE IS NOT, which looks inconsistent
       and is CPython's deliberate choice: the message is prose meant to be
       read, and the filename is a value whose exact bytes may matter -- a
       trailing space in a path is invisible otherwise. */
    if (tail) {
        buf[out++] = ':';
        buf[out++] = ' ';
        memcpy(buf + out, O(tail)->v.s.p, (size_t)O(tail)->v.s.n);
        out += O(tail)->v.s.n;
    }
    buf[out] = 0;
    return apy_str_take(buf, out);
}

APY_API apy_value apy_exc_text_of(apy_value v, int64_t quoted) {
    apy_value arg = O(v)->v.e.arg;
    /* MORE THAN ONE ARGUMENT PRINTS AS THE TUPLE. `str(ValueError('a','b'))`
       is `('a', 'b')` and its repr is `ValueError('a', 'b')` -- CPython shows
       the whole of `args` once there is more than one to show, and rendering
       only the first silently dropped the rest. */
    if (O(v)->v.e.argv && apy_is_seq(O(v)->v.e.argv) && !quoted
            && apy_exc_is_os(v)) {
        apy_value errno_text = apy_os_text(v, O(v)->v.e.argv);
        if (errno_text) return errno_text;
    }
    if (O(v)->v.e.argv && apy_is_seq(O(v)->v.e.argv)
            && O(O(v)->v.e.argv)->v.q.n > 1) {
        apy_value shown = apy_text(O(v)->v.e.argv, 1);
        int64_t n, out;
        char *buf;
        if (!quoted) return shown;
        n = (int64_t)strlen(apy_exc_shown(O(v)->v.e.name)) + O(shown)->v.s.n + 1;
        buf = (char *)malloc((size_t)n + 1);
        out = (int64_t)strlen(apy_exc_shown(O(v)->v.e.name));
        memcpy(buf, apy_exc_shown(O(v)->v.e.name), (size_t)out);
        /* The tuple's own parentheses ARE the call's, which is why the text
           is spliced in whole rather than wrapped again. */
        memcpy(buf + out, O(shown)->v.s.p, (size_t)O(shown)->v.s.n);
        out += O(shown)->v.s.n;
        buf[out] = 0;
        return apy_str_take(buf, out);
    }
    /* WHETHER there was an argument, not whether it is None. `str(E())` is
       empty and `str(E(None))` is "None"; `repr` shows `E()` and `E(None)`.
       Testing the argument's kind conflated the two, so an exception
       deliberately carrying None lost it. */
    int has = O(v)->v.e.has_arg;
    if (!quoted)
        /* `str(KeyError('k'))` is `"'k'"` -- the REPR of the argument, not the
           argument. KeyError alone does this, so that a missing key whose text
           is empty or is itself a message is still visible in the report. */
        return !has ? apy_lit("")
                    : apy_text(arg, !O(v)->v.e.rendered
                                    && strcmp(O(v)->v.e.name, "KeyError") == 0);
    {
        /* A KeyError REBUILT FROM A FAILED LOOKUP already holds the repr of
           the key -- that is what `rendered` records -- so repr'ing it again
           gave `KeyError("'k'")` where CPython says `KeyError('k')`. Every
           other type stores the plain message and does want the quotes.

           WHAT THIS DOES NOT FIX: `e.args[0]` is still the repr TEXT rather
           than the key, because a failed operation keeps a type and a message
           and never the object -- see `apy_err_value`. Retaining the key
           would fix both; this fixes the half that can be fixed without it. */
        int twice = O(v)->v.e.rendered
            && strcmp(O(v)->v.e.name, "KeyError") == 0;
        apy_value shown = !has ? apy_lit("") : apy_text(arg, !twice);
        int64_t n = (int64_t)strlen(apy_exc_shown(O(v)->v.e.name)) + O(shown)->v.s.n + 2;
        char *buf = (char *)malloc((size_t)n + 1);
        int64_t out = (int64_t)strlen(apy_exc_shown(O(v)->v.e.name));
        memcpy(buf, apy_exc_shown(O(v)->v.e.name), (size_t)out);
        buf[out++] = '(';
        memcpy(buf + out, O(shown)->v.s.p, (size_t)O(shown)->v.s.n);
        out += O(shown)->v.s.n;
        buf[out++] = ')';
        buf[out] = 0;
        return apy_str_take(buf, out);
    }
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now, and
   the exported half above stands in when nothing is ported. */
static apy_value apy_exc_text(apy_value v, int quoted) {
    return apy_exc_text_of(v, (int64_t)quoted);
}
APY_API apy_value apy_repr(apy_value v);
APY_API apy_value apy_lit(const char *p);
APY_API apy_value apy_getitem(apy_value seq, apy_value index);
/* A view's contents, taken when asked -- and the snapshot helper the
   non-dict path still uses. Both run above where they are defined. */
APY_API apy_value apy_view_items(apy_value v);
static apy_value apy_dict_parts_snapshot(apy_value d, int64_t which);
/* The set operators convert a view at the boundary, above where the helper
   that does it is defined. */
static apy_value apy_view_as_set(apy_value v);
/* `x.hex()` dispatches to the bytes form well above where it is defined. */
APY_API apy_value apy_bytes_hex(apy_value b, apy_value sep);
APY_API apy_value apy_bytes_hex_n(apy_value b, apy_value sep,
                                  apy_value perv);
/* `ascii` walks a str by CHARACTER to escape it, far above where the UTF-8
   step is defined. */
static int64_t apy_utf8_at(const unsigned char *p, int64_t n, int64_t i,
                           int64_t *len);
/* Subscripting a CLASS builds a generic alias, far above where aliases are
   defined. */
APY_API apy_value apy_alias_new(apy_value origin, apy_value args);
APY_API apy_value apy_alias_unpack(apy_value v);
/* Subscripting with a slice OBJECT resolves its bounds and slices, far above
   where the slicing itself is defined. */
APY_API apy_value apy_slice(apy_value seq, int64_t start, int64_t stop,
                            int64_t step, int64_t has_start, int64_t has_stop);
/* Slice ASSIGNMENT resolves the bounds through this, far above it. */
APY_API apy_value apy_slice_indices(apy_value sl, apy_value len_v);
static apy_value apy_dict_text(apy_value v);
/* The runtime's own callables, defined beside `apy_invoke` and reached from
   the `super()` lookup far above it. */
/* A str's length in CHARACTERS, which is what indexing and slicing it count
   -- defined with the other string measures, far below the subscript. */
/* A format field may carry accessors -- `{x[0]}`, `{a.real}` -- and attribute
   lookup is defined with the object model, well below the formatter. */
APY_API apy_value apy_getattr(apy_value obj, apy_value name);
/* PEP 604's `int | str` builds a union out of a `typing` form,
   which is defined with the rest of `typing` far below the
   operators. */
APY_API apy_value apy_typing_form(apy_value name);
static int64_t apy_str_chars(apy_value v);
APY_API apy_value apy_object_default(apy_value want);
static apy_value apy_kind_class(apy_value obj);
APY_API apy_value apy_dict_get_or(apy_value d, apy_value key,
                                  apy_value fallback);
static void apy_union_arms(apy_value into, apy_value v);
APY_API apy_value apy_typing_form(apy_value name);
/* `__init_subclass__` is called with the class KEYWORDS, and the keyword-call
   entry point is defined with the rest of the calling machinery far below. */
APY_API apy_value apy_call_kw(apy_value f, apy_value buf, int64_t argc,
                              apy_value kwd);
/* `f(*xs, **kw)` -- the spread form, defined beside the plain spread call and
   reaching the keyword binder above. */
APY_API apy_value apy_call_spread_kw(apy_value f, apy_value args,
                                     apy_value kwd);
static apy_value apy_native(int sel, int64_t arity, const char *name);
/* `object` and `type` as class OBJECTS, defined beside the natives they hold
   and reached from the attribute lookup far above. */
APY_API apy_value apy_object_class(void);
APY_API apy_value apy_type_class(void);
/* `%` formatting, defined beside `str.format` -- it borrows the whole of the
   mini-language -- and reached from `apy_mod`, far above. */
static apy_value apy_str_percent(apy_value fmt, apy_value right);
/* `typing`'s forms and the two introspection calls, defined at the very
   bottom beside the rest of `typing` and reached from `apy_getitem`. */
APY_API int64_t apy_is_special_form(apy_value v);
/* memoryview. Defined beside the dict views, which is far below `apy_getitem`
   and `apy_setitem` -- the two places that matter most. */
static const char *apy_mview_buf(apy_value v);
static int64_t apy_mview_at(apy_value v, int64_t i);
static apy_value apy_mview_slice(apy_value v, int64_t off, int64_t n,
                                 int64_t step);
APY_API apy_value apy_mview_bytes(apy_value v);
APY_API int64_t apy_mview_live(apy_value v);
APY_API apy_value apy_mview_item(apy_value v, int64_t i);
/* Cycle detection for `repr`, used by the dict renderer well above where the
   sequence one defines it. */
APY_API int64_t apy_repr_entered(apy_value v);
APY_API void apy_repr_left(apy_value v);
/* Declared with APY_API, not `static`: these two are host functions the IR
   can call, so their storage class has to match their definition -- and in
   the linked build that is external. A `static` forward declaration of an
   external definition is a C error, not a warning. */
APY_API apy_value apy_str(apy_value v);
APY_API apy_value apy_none(void);
static apy_value apy_fail(const char *type, const char *msg);
static apy_value apy_fail2(const char *type, const char *fmt,
                           const char *a, const char *b);
static apy_value apy_dict_get(apy_value d, apy_value key);
APY_API apy_value apy_dict_set(apy_value d, apy_value key, apy_value val);
static const char *apy_kind_name(apy_value v);
APY_API apy_value apy_bytes_repr(apy_value v);
APY_API apy_value apy_str_copy_bytes(apy_value p, int64_t n);
APY_API apy_value apy_str_copy(const char *p, int64_t n);
/* The class machinery, declared here and defined at the very bottom of this
   file. The order is deliberate and it is the reverse of the dependency: the
   operators -- `+`, `str`, `len`, `==` -- were all written against a closed
   set of kinds, and an instance dispatching to a user method is a hook INTO
   each of them. Defining the hooks first would put four hundred lines of
   class machinery between the reader and the arithmetic; declaring them here
   costs eight lines and leaves each operator reading as itself with one
   `if (...) dispatch` at the top. */
static apy_value apy_dunder(apy_value v, const char *name);
static apy_value apy_unary_dunder(apy_value v, const char *name);
static apy_value apy_method1(apy_value v, const char *name, apy_value arg);
static apy_value apy_binary_dunder(apy_value a, apy_value b,
                                   const char *name, const char *rname);
/* The two halves of the one above, plus the ungated read of what an instance
   holds: a comparison asks the written direct dunder, then the builtin the
   class extends, then the written mirror -- see `apy_order_mirror_first_of`
   for why the middle one is in the middle. */
static apy_value apy_written_dunder(apy_value who, apy_value other,
                                    const char *name);
static int apy_order_mirror_first(apy_value a, apy_value b,
                                  const char *mirror);
static apy_value apy_held_value(apy_value v);
static int apy_either_inst(apy_value a, apy_value b);
static apy_value apy_call_n(apy_value f, apy_value *argv, int64_t argc);
static apy_value apy_invoke(apy_value f, apy_value *a, int64_t n);
static apy_value apy_type_of(apy_value v);
APY_API apy_value apy_copy(apy_value v);
APY_API apy_value apy_update(apy_value target, apy_value src);
APY_API apy_value apy_extend(apy_value seq, apy_value other);
APY_API apy_value apy_extend_meth(apy_value seq, apy_value other);
static apy_value apy_type_kind_attr(apy_value type_obj,
                                    apy_value owner, apy_value name);
APY_API apy_value apy_length_hint(apy_value src);
APY_API apy_value apy_extend_arg(apy_value into, apy_value more,
                                 apy_value callee);
APY_API apy_value apy_bitor(apy_value a, apy_value b);
APY_API apy_value apy_bitand(apy_value a, apy_value b);
APY_API apy_value apy_bitxor(apy_value a, apy_value b);
APY_API apy_value apy_sub(apy_value a, apy_value b);
APY_API apy_value apy_mul(apy_value a, apy_value b);
APY_API apy_value apy_add(apy_value a, apy_value b);
/* The rest of the arithmetic, declared here because `apy_op_apply` names all
   of them and sits with the operator fallback, well above where most are
   defined. */
APY_API apy_value apy_truediv(apy_value a, apy_value b);
APY_API apy_value apy_floordiv(apy_value a, apy_value b);
APY_API apy_value apy_mod(apy_value a, apy_value b);
APY_API apy_value apy_bitor(apy_value a, apy_value b);
APY_API apy_value apy_lshift(apy_value a, apy_value b);
APY_API apy_value apy_rshift(apy_value a, apy_value b);
/* `class D(dict)` acting as its dict for one operation -- defined with the
   two method-dispatch helpers near the end, used by the operator fallback
   near the beginning. */
static apy_value apy_as_builtin(apy_value v, const char *dunder);
/* One builtin container built from one argument, by KIND. Defined with the
   instantiation machinery, named by the `super().__init__` native well above
   it. */
static apy_value apy_call_kind(int kind, apy_value src);
/* The builtin kind a class extends, looked up the WHOLE chain: a subclass of
   a subclass of `tuple` is still a tuple. 0 for a class that extends none. */
APY_API int64_t apy_class_builtin_kind(apy_value cls);
APY_API apy_value apy_to_dict(apy_value src);
APY_API apy_value apy_iter(apy_value v);
/* `callable(x)`, defined with the descriptors because an instance is callable
   only when its class writes `__call__`. Declared here because `iter(f, s)`
   refuses a non-callable when the iterator is MADE, and that is six parts
   earlier. */
APY_API apy_value apy_callable(apy_value v);
APY_API apy_value apy_getiter(apy_value v);
APY_API apy_value apy_step(apy_value it);
APY_API apy_value apy_stop(void);
static apy_value apy_drain_cursor(apy_value it);
static apy_value apy_cursor(apy_value src, apy_value fn, int mode,
                            int64_t start);
APY_API apy_value apy_gen_next(apy_value g, apy_value fallback,
                               int64_t has_default);
APY_API apy_value apy_gen_drain(apy_value g);
static apy_value apy_gen_step(apy_value g, apy_value sent, int *done);
APY_API apy_value apy_iterable(apy_value v);
/* The rule `__iter__`'s ANSWER has to satisfy, which `str.join` asks for
   itself so it can word the refusal its own way. See `apy_is_iterator_of`. */
static int apy_is_iterator(apy_value it);
APY_API apy_value apy_isinstance(apy_value v, apy_value type_name);
APY_API int64_t apy_error_matches(apy_value handler);
APY_API void apy_error_clear(void);
APY_API apy_value apy_getitem(apy_value seq, apy_value index);
APY_API apy_value apy_to_bytes(apy_value src);
/* `range` is three numbers and every question about one is arithmetic on
   them; the helpers live with the constructor, far below the operators that
   ask. */
APY_API apy_value apy_range(int64_t start, int64_t stop, int64_t step);
static int64_t apy_range_len(apy_value r);
static int64_t apy_range_at(apy_value r, int64_t i);
static int64_t apy_range_find(apy_value r, int64_t want);
static apy_value apy_name(const char *text);
/* `class D(dict)` -- the builtin an instance carries, defined with the object
   model far below the operators that delegate to it. */
static apy_value apy_inst_held(apy_value v);
static int64_t apy_dict_find(apy_value d, apy_value key);
APY_API apy_value apy_dict_get(apy_value d, apy_value key);
static apy_value apy_class_find(apy_value cls, apy_value name);
static apy_value apy_bind(apy_value f, apy_value self);
static int apy_type_is_sub(apy_value of, apy_value cls);
/* `format(n, 'c')` encodes a code point where the formatter runs, well above
   where `chr` is defined. */
APY_API apy_value apy_chr(apy_value v);
/* `next()` reports a generator's exhaustion with the value its `return`
   carried, and runs well above where generators are defined. */
static apy_value apy_gen_stop(apy_value g);
/* `del obj.attr` consults a descriptor, well above where the descriptor
   protocol itself is defined. */
static int apy_is_data_descriptor(apy_value v);
/* Declared here and not just above its definition: `__dict__` access reads
   it several hundred lines earlier than `__slots__` enforcement defines it. */
static int apy_slot_allows(apy_value cls, apy_value name);

/* A str value's bytes as a NUL-terminated C string, for comparing an
   attribute name against a literal. Every str cell keeps a NUL after its
   bytes (see `apy_str_take`), so this is the pointer itself.

   The parameter is `x` and NOT `v`. With `v` the body's `->v.s.p` is itself a
   use of the parameter, so `APY_CSTR(name)` expanded to `O(name)->name.s.p`
   and every call site failed with "no member named 'name'" -- a diagnostic
   that points at the macro's argument rather than at the macro. */
#define APY_CSTR(x) (O(x)->v.s.p)
/* The set section is above the numeric tower and reports a bad operand pair
   the same way `+` does; the shared wording lives down there with the
   arithmetic that first needed it. */
static apy_value apy_binop_error(const char *op, apy_value a, apy_value b);

APY_API int64_t apy_is_seq_of(apy_value v) {
    return O(v)->kind == APY_LIST_K || O(v)->kind == APY_TUPLE_K;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_is_seq(apy_value v) {
    return (int)apy_is_seq_of(v);
}

/* Deliberately NOT part of `apy_is_seq`. A set shares the `v.q` layout but is
   not a sequence in any way a caller of `apy_is_seq` means: it has no order to
   index, `+` and `*` do not apply, and `list == tuple` is False while
   `set == frozenset` is True. Every place that wants "has v.q" rather than "is
   a sequence" says so by calling both. */
APY_API int64_t apy_is_set_of(apy_value v) {
    return O(v)->kind == APY_SET_K || O(v)->kind == APY_FROZEN_K;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_is_set(apy_value v) {
    return (int)apy_is_set_of(v);
}

/* THE EXPORTED HALF, which `runtime/list_cell.py` replaces. The `static`
   below keeps the name its callers use and the cast they do not have to
   write; this body is what the runtime uses when nothing is ported. */
/* Declared here because the delegate is defined below its first use. */
static apy_value apy_seq_new(int kind, int64_t cap);
APY_API apy_value apy_seq_new_of(int64_t kind, int64_t cap) {
    apy_obj *o = apy_alloc((int)kind);
    if (cap < 1) cap = 1;
    o->v.q.items = (apy_value *)(uintptr_t)apy_alloc_block(
        cap * (int64_t)sizeof(apy_value));
    o->v.q.n = 0;
    o->v.q.cap = cap;
    return V(o);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above is what stands in when nothing is
   ported. The cast to a machine word happens here, once. */
static apy_value apy_seq_new(int kind, int64_t cap) {
    return apy_seq_new_of((int64_t)kind, cap);
}

APY_API apy_value apy_list_new(int64_t cap) { return apy_seq_new(APY_LIST_K, cap); }
APY_API apy_value apy_tuple_new(int64_t cap) { return apy_seq_new(APY_TUPLE_K, cap); }

/* THE ONE EMPTY TUPLE. CPython keeps exactly one and nothing else about a
   tuple is shared: `() is tuple()`, `(1, 2)[0:0] is ()` and `t * 0 is ()` are
   all True there, while two equal non-empty tuples built separately are two
   objects. A program sees it with `is` and with `id()`.

   BUILT ON FIRST ASK and never freed, like the string singletons in
   `_core.py`. Nothing may ever push into it -- a tuple is immutable, so the
   only way one grows is the constructor that has not finished yet, and this
   is only ever handed out as a FINISHED value. */
static apy_value apy_tuple_shared;

APY_API apy_value apy_tuple_empty(void) {
    if (!apy_tuple_shared) apy_tuple_shared = apy_seq_new(APY_TUPLE_K, 1);
    return apy_tuple_shared;
}

/* THE TUPLE TO ANSWER WITH, once one has been built. An empty one becomes
   the shared cell; everything else is itself. Written as a filter over a
   finished value rather than as a test inside `apy_tuple_new`, because a
   tuple is allocated with room and filled afterwards -- at the moment it is
   made nobody knows yet whether it will stay empty. */
APY_API apy_value apy_tuple_done(apy_value q) {
    if (q && O(q)->kind == APY_TUPLE_K && O(q)->v.q.n == 0)
        return apy_tuple_empty();
    return q;
}

/* Append with NO checking at all -- not that the cell is a list, not that a
   set already holds an equal element. Split out from `apy_seq_push` because
   the set code appends to a `v.q` that `apy_is_seq` rejects, and because a set
   operation whose inputs are already sets cannot produce a duplicate and so
   must not pay for the scan that would prove it. */
APY_API void apy_q_append_of(apy_value q, apy_value item) {
    apy_obj *o = O(q);
    if (o->v.q.n == o->v.q.cap) {
        int64_t was = o->v.q.cap * (int64_t)sizeof(apy_value);
        o->v.q.cap *= 2;
        o->v.q.items = (apy_value *)(uintptr_t)apy_realloc_block(
            (apy_value)(uintptr_t)o->v.q.items, was,
            o->v.q.cap * (int64_t)sizeof(apy_value));
    }
    o->v.q.items[o->v.q.n++] = item;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static void apy_q_append(apy_value q, apy_value item) {
    apy_q_append_of(q, item);
}

/* Used both to BUILD a literal and to implement `list.append`. A tuple is
   built with it too and then never appended to again: immutability is a rule
   the frontend enforces, not a property of the cell. */
/* GROW A BYTEARRAY BY ONE BYTE. `v.s` has no capacity field -- a str is
   immutable and never needed one -- so this reallocates on every append,
   which is quadratic and is the honest trade against adding a field to
   every string in the program. A caller building a large buffer builds it
   with `+` and one copy instead. */
static int apy_bytes_push(apy_value b, int byte) {
    int64_t n = O(b)->v.s.n;
    char *buf = (char *)malloc((size_t)n + 2);
    if (!buf) return 0;
    if (n) memcpy(buf, O(b)->v.s.p, (size_t)n);
    buf[n] = (char)byte;
    buf[n + 1] = 0;
    O(b)->v.s.p = buf;
    O(b)->v.s.n = n + 1;
    return 1;
}

/* Whether `v` is a bytearray: the bytes cell that admits it is writable. The
   flag is the whole of what separates the two, so it is what every mutating
   method tests -- `bytes` reaching one of them has no such method and must
   say so rather than quietly rewrite a literal. */
static int apy_is_bytearray(apy_value v) {
    return O(v)->kind == APY_BYTES_K && O(v)->v.s.mut;
}

/* THE OCTET A BYTEARRAY METHOD'S ARGUMENT STANDS FOR, or -1 with CPython's
   own refusal already raised. A bytearray holds NUMBERS and not one-byte
   strings -- `b.append(b"z")` is a TypeError and `b.append(300)` a
   ValueError -- and `append`, `insert` and `remove` all say it the same way,
   which is why it is said once. */
static int64_t apy_byte_arg(apy_value v) {
    int64_t byte;
    /* THE TYPE IS CHECKED BEFORE THE PAYLOAD IS READ. `apy_index_arg` takes
       an argument already known to be an integer and reads the payload word
       straight out -- so handing it `b"z"` read a POINTER as a number and
       reported `byte must be in range(0, 256)` about a value the program
       never wrote. The original inline copy of this had the same hole. */
    if (!apy_is_int_like(v)) {
        apy_fail2("TypeError",
                  "'%s' object cannot be interpreted as an integer%s",
                  apy_kind_name(v), "");
        return -1;
    }
    /* AND A BIG IS OUT OF RANGE RATHER THAN AN OVERFLOW: `2 ** 100` is an
       int like any other here, and CPython says so in those words. */
    if (apy_is_big(v)) {
        apy_fail("ValueError", "byte must be in range(0, 256)");
        return -1;
    }
    if (!apy_index_arg(v, &byte, APY_IDX_SUB)) return -1;
    if (byte < 0 || byte > 255) {
        apy_fail("ValueError", "byte must be in range(0, 256)");
        return -1;
    }
    return byte;
}

APY_API apy_value apy_seq_push(apy_value seq, apy_value item) {
    /* A BYTEARRAY APPENDS AN INT. It was missing here and in the
       interpreter both -- `.append` on the most ordinary way there is to
       build one up answered "'bytearray' object has no attribute
       'append'", which `bundled/subprocess.py` found. */
    if (apy_is_bytearray(seq)) {
        int64_t byte = apy_byte_arg(item);
        if (byte < 0) return 0;
        if (!apy_bytes_push(seq, (int)byte))
            return apy_fail("MemoryError", "out of memory");
        return apy_none();
    }
    if (!apy_is_seq(seq))
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'append'%s",
                         apy_kind_name(seq), "");
    apy_q_append(seq, item);
    return apy_none();
}

/* `b'ab'[0]` is 97, not `b'a'`. The one place bytes differs from str in KIND
   rather than in units: indexing a str gives a one-character str and indexing
   bytes gives the integer octet, while SLICING either gives the same kind
   back. Getting this wrong is invisible until something does arithmetic on
   the result. */
static apy_value apy_bytes_getitem(apy_value seq, int64_t i) {
    int64_t n = O(seq)->v.s.n;
    if (i < 0) i += n;
    if (i < 0 || i >= n)
        return apy_fail("IndexError", "index out of range");
    return apy_from_int((int64_t)(unsigned char)O(seq)->v.s.p[i]);
}

/* `b'ab' * 3`. A negative or zero count gives empty, as every sequence
   repetition in Python does -- not an error. */
static apy_value apy_bytes_repeat(apy_value v, apy_value count) {
    int64_t k, n, i;
    if (!apy_index_arg(count, &k, APY_IDX_SUB)) return 0;
    /* `b * 1 IS b` for immutable bytes, exactly as `s * 1 is s` and
       `t * 1 is t` -- see `apy_str_repeat`. A BYTEARRAY repeated once is a
       fresh bytearray, because it can be written to. */
    if (k == 1 && !O(v)->v.s.mut) return v;
    if (k < 0) k = 0;
    n = O(v)->v.s.n;
    { char *out = (char *)malloc((size_t)(n * k) + 1);
      if (!out) { fputs("uasm: out of memory\n", stderr); exit(1); }
      for (i = 0; i < k; i++) memcpy(out + i * n, O(v)->v.s.p, (size_t)n);
      out[n * k] = 0;
      /* A BYTEARRAY REPEATS INTO A BYTEARRAY, as `+` already carried:
         `mut` is the whole of what separates the two kinds and it has to
         travel with the tag.
         AND NOT THROUGH THE SHARED CELLS. CPython allocates a repetition
         directly rather than through the constructor that consults its
         cache, so `b"ab" * 0 is b""` is False there. */
      return apy_bytes_own(out, n * k, O(v)->v.s.mut); }
}


APY_API apy_value apy_getitem(apy_value seq, apy_value index) {
    int64_t i, n;
    if (O(seq)->kind == APY_MVIEW_K) {
        if (!apy_mview_live(seq)) return 0;
        /* A CAST VIEW IS INDEXED IN ELEMENTS and each one is decoded as its
           format says, which is the whole of what `cast` did. A fresh view
           is one byte wide and takes the plain path below. */
        if (apy_is_int_like(index) && O(seq)->v.mv.wide > 1) {
            int64_t at, many = O(seq)->v.mv.n / O(seq)->v.mv.wide;
            if (!apy_index_arg(index, &at, APY_IDX_SUB)) return 0;
            if (at < 0) at += many;
            if (at < 0 || at >= many)
                return apy_fail("IndexError",
                                "index out of bounds on dimension 1");
            return apy_mview_item(seq, at);
        }
        if (apy_is_int_like(index)) {
            if (!apy_index_arg(index, &i, APY_IDX_SUB)) return 0;
            n = O(seq)->v.mv.n;
            if (i < 0) i += n;
            if (i < 0 || i >= n)
                return apy_fail("IndexError", "index out of bounds on "
                                              "dimension 1");
            return apy_from_int((int64_t)(unsigned char)
                                apy_mview_buf(seq)[apy_mview_at(seq, i)]);
        }
        if (index && O(index)->kind == APY_SLICE_K) {
            /* STILL A VIEW. `mv[1:3][0] = 9` writes to the original buffer,
               which a copy here would silently lose. */
            apy_value bounds = apy_slice_indices(index,
                                                 apy_from_int(O(seq)->v.mv.n));
            int64_t start, stop, step, count;
            if (!bounds) return 0;
            start = O(O(bounds)->v.q.items[0])->v.i;
            stop = O(O(bounds)->v.q.items[1])->v.i;
            step = O(O(bounds)->v.q.items[2])->v.i;
            count = step > 0 ? (stop > start ? (stop - start + step - 1) / step
                                             : 0)
                             : (start > stop ? (start - stop - step - 1)
                                               / -step
                                             : 0);
            return apy_mview_slice(seq, apy_mview_at(seq, start), count,
                                   O(seq)->v.mv.step * step);
        }
        return apy_fail2("TypeError",
                         "memoryview indices must be integers%s%s", "", "");
    }
    if (O(seq)->kind == APY_BYTES_K) {
        /* NOT GATED ON THE INDEX BEING AN INT. It was, so `b"ab"[1.0]` fell
           past every arm to the guard below and answered `'bytes' object is
           not subscriptable` -- a sentence about the receiver, for a
           complaint about the subscript. */
        if (!apy_is_int_like(index) && O(index)->kind == APY_INST_K) {
            apy_value got = apy_unary_dunder(index, "__index__");
            if (apy_error_occurred()) return 0;
            if (got && apy_is_int_like(got)) index = got;
        }
        if (!apy_is_int_like(index))
            /* `byte`, SINGULAR, for a bytes -- CPython's own wording, and
               not the plural its type name would suggest. */
            return apy_fail2("TypeError",
                             "%s indices must be integers or slices, not %s",
                             apy_is_bytearray(seq) ? "bytearray" : "byte",
                             apy_kind_name(index));
        if (!apy_index_arg(index, &i, APY_IDX_SUB)) return 0;
        return apy_bytes_getitem(seq, i);
    }
    if (O(seq)->kind == APY_INST_K) {
        /* No fallthrough for an ordinary class: one without `__getitem__` is
           "not subscriptable", and `apy_method1` answering 0 with no error
           set lands on exactly that message below. */
        apy_value r = apy_method1(seq, "__getitem__", index);
        if (r || apy_error_occurred()) return r;
        /* A CLASS THAT EXTENDS A BUILTIN IS one for everything it did not
           write. `class D(dict)` with only a `__missing__` in it still has to
           answer `d[k]`, and this is the dict it answers from. */
        {
            apy_value held = apy_inst_held(seq);
            if (held) {
                if (O(held)->kind == APY_DICT_K) {
                    int64_t at = apy_dict_find(held, index);
                    if (at >= 0) return O(held)->v.d.vals[at];
                    /* `__missing__` IS WHAT A dict SUBCLASS IS FOR: a key
                       that is not there is the class's question to answer,
                       and a KeyError only when it declines to. */
                    {
                        apy_value miss = apy_method1(seq, "__missing__",
                                                     index);
                        if (miss || apy_error_occurred()) return miss;
                    }
                    return apy_dict_get(held, index);
                }
                return apy_getitem(held, index);
            }
        }
    }
    if (O(seq)->kind == APY_TYPE_K) {
        /* `C[int]`. A CLASS IS NOT A CONTAINER: subscripting one asks
           `__class_getitem__`, and a class without it is parameterised into a
           generic alias -- which is what `list[int]` is and why it prints
           rather than indexing anything. */
        apy_value hook = apy_class_find(seq, apy_name("__class_getitem__"));
        if (hook) {
            /* AN IMPLICIT CLASSMETHOD: it receives the class as its first
               argument, so the call is bound to the class rather than
               passing the subscript alone. */
            apy_value arg = index;
            return apy_call_n(apy_bind(hook, seq), &arg, 1);
        }
        /* THE METACLASS DECIDES, if it has an opinion: `Box[int]` where
           `Box` inherits `Generic` is `type(Box).__getitem__(Box, int)`, and
           that is how a generic class is parameterised without every class
           in the program becoming subscriptable. */
        if (O(seq)->v.t.meta) {
            apy_value m = apy_class_find(O(seq)->v.t.meta,
                                         apy_name("__getitem__"));
            if (m) {
                apy_value arg = index;
                return apy_call_n(apy_bind(m, seq), &arg, 1);
            }
        }
        /* A CLASS WITHOUT THE HOOK IS NOT SUBSCRIPTABLE. CPython says so --
           `class D: pass` then `D[int]` is a TypeError -- and only the
           builtin containers answer a generic alias, because they carry
           `__class_getitem__` of their own. Making every class parameterise
           silently would turn a mistake into an object. */
        return apy_fail2("TypeError", "type '%s' is not subscriptable%s",
                         APY_CSTR(O(seq)->v.t.name), "");
    }
    if (O(seq)->kind == APY_RANGE_K) {
        int64_t n = apy_range_len(seq);
        if (index && O(index)->kind == APY_SLICE_K) {
            /* A SLICE OF A RANGE IS A RANGE, not a list -- `r[1:3]` in
               CPython answers `range(2, 6, 2)`, and materialising would
               undo the whole point of the kind. */
            apy_value a = O(index)->v.sl.start, b = O(index)->v.sl.stop;
            apy_value c = O(index)->v.sl.step;
            int64_t lo = 0, hi = n, by = 1;
            if (c && O(c)->kind != APY_NONE_K
                    && !apy_index_arg(c, &by, APY_IDX_SIZE)) return 0;
            if (by == 0) return apy_fail("ValueError",
                                         "slice step cannot be zero");
            if (a && O(a)->kind != APY_NONE_K) {
                if (!apy_index_arg(a, &lo, APY_IDX_SIZE)) return 0;
                if (lo < 0) lo += n;
                if (lo < 0) lo = 0;
                if (lo > n) lo = n;
            } else if (by < 0) lo = n - 1;
            if (b && O(b)->kind != APY_NONE_K) {
                if (!apy_index_arg(b, &hi, APY_IDX_SIZE)) return 0;
                if (hi < 0) hi += n;
                if (hi < 0) hi = by < 0 ? -1 : 0;
                if (hi > n) hi = n;
            } else if (by < 0) hi = -1;
            {
                int64_t start = apy_range_at(seq, lo);
                int64_t step = O(seq)->v.rg.step * by;
                int64_t stop = O(seq)->v.rg.start + hi * O(seq)->v.rg.step;
                return apy_range(start, stop, step);
            }
        }
        {
            int64_t at;
            if (!apy_index_arg(index, &at, APY_IDX_SUB)) return 0;
            if (at < 0) at += n;
            if (at < 0 || at >= n)
                return apy_fail("IndexError", "range object index out of "
                                              "range");
            return apy_from_int(apy_range_at(seq, at));
        }
    }
    if (apy_is_special_form(seq)) {
        apy_value args = apy_tuple_new(2);
        if (O(index)->kind == APY_TUPLE_K) args = index;
        else apy_seq_push(args, index);
        /* `Optional[X]` IS `X | None`. 3.14 unified the two spellings, so a
           program that prints the annotation sees the union rather than the
           form it was written with, and `get_args` answers two arms. */
        apy_value form_name = apy_dict_get_or(O(seq)->v.o.dict,
                                              apy_lit("_name"), 0);
        if (form_name && strcmp(APY_CSTR(form_name), "Optional") == 0) {
            apy_value arms = apy_tuple_new(4);
            int64_t i;
            for (i = 0; i < O(args)->v.q.n; i++)
                apy_union_arms(arms, O(args)->v.q.items[i]);
            apy_union_arms(arms, apy_none());
            return apy_alias_new(apy_typing_form(apy_lit("Union")), arms);
        }
        return apy_alias_new(seq, args);
    }
    if (O(seq)->kind == APY_FUNC_K && O(seq)->v.fn.is_type) {
        /* `list[int]` -- PARAMETERISING a builtin type. It is not indexing:
           nothing is looked up, and what comes back is an object a program
           prints or writes in an annotation. */
        apy_value args = apy_tuple_new(2);
        if (O(index)->kind == APY_TUPLE_K) args = index;
        else apy_seq_push(args, index);
        return apy_alias_new(seq, args);
    }
    if (O(seq)->kind == APY_SLICE_K) { /* not a container */ }
    else if (index && O(index)->kind == APY_SLICE_K) {
        /* THE BOUNDS ARE RESOLVED HERE, not by the caller: which of the three
           were written decides what an omitted one means, and for a negative
           step that is the end rather than the beginning. None is "not
           written"; anything else is an index. */
        apy_value a = O(index)->v.sl.start, b = O(index)->v.sl.stop;
        apy_value c = O(index)->v.sl.step;
        int64_t start = 0, stop = 0, step = 1;
        int64_t has_start = a && O(a)->kind != APY_NONE_K;
        int64_t has_stop = b && O(b)->kind != APY_NONE_K;
        if (has_start && !apy_index_arg(a, &start, APY_IDX_SIZE)) return 0;
        if (has_stop && !apy_index_arg(b, &stop, APY_IDX_SIZE)) return 0;
        if (c && O(c)->kind != APY_NONE_K
                && !apy_index_arg(c, &step, APY_IDX_SIZE)) return 0;
        return apy_slice(seq, start, stop, step, has_start, has_stop);
    }
    if (O(seq)->kind == APY_DICT_K) return apy_dict_get(seq, index);
    /* "Is this subscriptable at all" is asked BEFORE "is the index an int",
       because a set answers the first question and CPython reports that:
       `{1, 2}[0]` is `'set' object is not subscriptable`, not a complaint
       about the index. Testing the index first put every non-subscriptable
       kind on the wrong message whenever the index happened to be an int. */
    if (!apy_is_seq(seq) && O(seq)->kind != APY_STR_K)
        return apy_fail2("TypeError", "'%s' object is not subscriptable%s",
                         apy_kind_name(seq), "");
    if (!apy_is_int_like(index) && O(index)->kind == APY_INST_K) {
        /* `__index__` -- how a user object BECOMES an index. PEP 357, and the
           reason it is a separate dunder from `__int__`: a float has `__int__`
           and is still not a valid subscript, so accepting the general
           conversion here would make `xs[1.5]` work. */
        apy_value got = apy_unary_dunder(index, "__index__");
        if (apy_error_occurred()) return 0;
        if (got && apy_is_int_like(got)) index = got;
    }
    if (!apy_is_int_like(index)) {
        /* Two texts, not one with a substituted noun. CPython says
           `list indices must be integers or slices, not float` for a list or
           a tuple -- both of which DO accept a slice -- and
           `string indices must be integers, not 'float'` for a str, with the
           kind quoted and no mention of slices. The single generic form was
           wrong for whichever half it was not written from. */
        if (O(seq)->kind == APY_STR_K)
            return apy_fail2("TypeError",
                             "string indices must be integers, not '%s'%s",
                             apy_kind_name(index), "");
        return apy_fail2("TypeError",
                         "%s indices must be integers or slices, not %s",
                         apy_kind_name(seq), apy_kind_name(index));
    }
    if (!apy_index_arg(index, &i, APY_IDX_SUB)) return 0;
    if (apy_is_seq(seq)) {
        n = O(seq)->v.q.n;
        /* A negative index counts from the end, and it is the LENGTH that
           decides whether it lands: `xs[-1]` on an empty list is an
           IndexError, not element -1. */
        if (i < 0) i += n;
        if (i < 0 || i >= n)
            /* NAMES THE KIND IT WAS GIVEN. `apy_is_seq` is true of a tuple
               too, so `(1,)[9]` reported "list index out of range" -- a
               sentence about a type the program never mentioned. */
            return apy_fail2("IndexError", "%s index out of range%s",
                             apy_kind_name(seq), "");
        return O(seq)->v.q.items[i];
    }
    if (O(seq)->kind == APY_STR_K) {
        /* BY CHARACTER, not by byte. A str is stored as UTF-8, and `len`
           already counts characters -- so indexing bytes made `s[1]` on
           `"héllo"` the first HALF of a character, which `ord` then
           reported as a string of length != 1: a complaint about a string the
           program never wrote. Slicing and iteration follow the same rule. */
        const unsigned char *p = (const unsigned char *)O(seq)->v.s.p;
        int64_t bytes = O(seq)->v.s.n, at = 0, seen = 0, used;
        n = apy_str_chars(seq);
        if (i < 0) i += n;
        if (i < 0 || i >= n)
            return apy_fail("IndexError", "string index out of range");
        while (seen < i && at < bytes) {
            apy_utf8_at(p, bytes, at, &used);
            at += used;
            seen++;
        }
        apy_utf8_at(p, bytes, at, &used);
        return apy_str_copy(O(seq)->v.s.p + at, used);
    }
    return apy_fail2("TypeError", "'%s' object is not subscriptable%s",
                     apy_kind_name(seq), "");   /* unreachable; see the guard */
}

APY_API apy_value apy_str(apy_value v) { return apy_text(v, 0); }

/* --- printing ---------------------------------------------------------- */
/* `items` is the ADDRESS of an array of n values -- the frontend builds one in
   a stack slot, because the IR has no varargs and `print` takes any number. */
/* `print(a, b, sep='-', end='!')`. The separator and the terminator are
   VALUES, not constants, because they are ordinary keyword arguments and a
   program may compute them.

   `sep=None` and `end=None` mean the defaults, which is what CPython accepts
   and what an omitted one lowers to -- so "not given" and "given as None" are
   the same request, and neither needs a second entry point. */
APY_API void apy_print_with(apy_value items, int64_t n, apy_value sep,
                            apy_value end) {
    const apy_value *a = (const apy_value *)items;
    int64_t i;
    const char *sp = " ";
    int64_t spn = 1;
    /* A str SUBCLASS IS A SEPARATOR: `print("a", "b", sep=S("-"))` writes
       `a-b` in CPython and the kind test alone fell back to a space --
       a wrong answer rather than a refusal. */
    sep = apy_text_like(sep);
    end = apy_text_like(end);
    if (O(sep)->kind == APY_STR_K) { sp = APY_CSTR(sep); spn = O(sep)->v.s.n; }
    for (i = 0; i < n; i++) {
        apy_value s;
        if (i) fwrite(sp, 1, (size_t)spn, stdout);
        s = apy_str(a[i]);
        if (!s) return;
        fwrite(O(s)->v.s.p, 1, (size_t)O(s)->v.s.n, stdout);
    }
    if (O(end)->kind == APY_STR_K)
        fwrite(APY_CSTR(end), 1, (size_t)O(end)->v.s.n, stdout);
    else
        fputc('\n', stdout);
}

APY_API void apy_print(apy_value items, int64_t n) {
    const apy_value *a = (const apy_value *)items;
    int64_t i;
    for (i = 0; i < n; i++) {
        apy_value s;
        if (i) fputc(' ', stdout);
        s = apy_str(a[i]);
        fwrite(O(s)->v.s.p, 1, (size_t)O(s)->v.s.n, stdout);
    }
    fputc('\n', stdout);
}

APY_API apy_value apy_setitem(apy_value seq, apy_value index, apy_value item) {
    int64_t i, n;
    if (O(seq)->kind == APY_INST_K) {
        /* Two arguments, so this cannot go through `apy_method1`. */
        apy_value m = apy_dunder(seq, "__setitem__"), args[2];
        if (m) { args[0] = index; args[1] = item;
                 return apy_call_n(m, args, 2); }
        /* A CLASS THAT EXTENDS A BUILTIN writes into the one it carries for
           everything its body did not define. */
        if (apy_inst_held(seq))
            return apy_setitem(apy_inst_held(seq), index, item);
    }
    if (O(seq)->kind == APY_DICT_K) return apy_dict_set(seq, index, item);
    if (index && O(index)->kind == APY_SLICE_K && O(seq)->kind == APY_LIST_K) {
        /* THE SPAN IS REPLACED, and the replacement need not be the same
           length -- `xs[1:3] = [9]` shortens the list. So this is a rebuild
           rather than a write through the existing cells, which is also why
           it cannot share the index path below. */
        apy_value bounds = apy_slice_indices(index, apy_from_int(
            O(seq)->v.q.n));
        apy_value fresh, given;
        int64_t start, stop, step, k;
        if (!bounds) return 0;
        start = O(O(bounds)->v.q.items[0])->v.i;
        stop = O(O(bounds)->v.q.items[1])->v.i;
        step = O(O(bounds)->v.q.items[2])->v.i;
        if (step != 1)
            return apy_fail("ValueError",
                            "only step 1 slice assignment is supported");
        given = apy_iterable(item);
        if (!given) return 0;
        if (!apy_is_seq(given) && O(given)->kind != APY_SET_K)
            return apy_fail2("TypeError",
                             "can only assign an iterable, not %s%s",
                             apy_kind_name(item), "");
        fresh = apy_seq_new(APY_LIST_K, O(seq)->v.q.n + 4);
        for (k = 0; k < start && k < O(seq)->v.q.n; k++)
            apy_seq_push(fresh, O(seq)->v.q.items[k]);
        for (k = 0; k < O(given)->v.q.n; k++)
            apy_seq_push(fresh, O(given)->v.q.items[k]);
        for (k = stop; k < O(seq)->v.q.n; k++)
            if (k >= 0) apy_seq_push(fresh, O(seq)->v.q.items[k]);
        /* IN PLACE: every other name bound to this list has to see the
           change, which is what makes `xs[:] = ys` the idiom it is. */
        apy_free_block((apy_value)(uintptr_t)O(seq)->v.q.items,
                       O(seq)->v.q.cap * (int64_t)sizeof(apy_value));
        O(seq)->v.q.items = O(fresh)->v.q.items;
        O(seq)->v.q.n = O(fresh)->v.q.n;
        O(seq)->v.q.cap = O(fresh)->v.q.cap;
        O(fresh)->v.q.items = NULL;
        O(fresh)->v.q.n = 0;
        O(fresh)->v.q.cap = 0;
        return apy_none();
    }
    if (O(seq)->kind == APY_MVIEW_K) {
        int64_t byte;
        if (!apy_mview_live(seq)) return 0;
        /* THE FLAG IS THE VIEW'S, not the buffer's: `m.toreadonly()` hands
           out one that refuses over a bytearray that accepts. */
        if (O(seq)->v.mv.ro || !O(O(seq)->v.mv.src)->v.s.mut)
            return apy_fail("TypeError", "cannot modify read-only memory");
        if (!apy_index_arg(index, &i, APY_IDX_SUB)) return 0;
        n = O(seq)->v.mv.n;
        if (i < 0) i += n;
        if (i < 0 || i >= n)
            return apy_fail("IndexError", "index out of bounds on "
                                          "dimension 1");
        if (!apy_index_arg(item, &byte, APY_IDX_SUB)) return 0;
        if (byte < 0 || byte > 255)
            return apy_fail("ValueError", "memoryview: invalid value for "
                                          "format 'B'");
        /* THROUGH to the buffer the view was taken over -- that write being
           visible in the original is the whole of what a memoryview is. */
        ((char *)apy_mview_buf(seq))[apy_mview_at(seq, i)] = (char)byte;
        return apy_none();
    }
    if (O(seq)->kind == APY_BYTES_K) {
        int64_t byte;
        if (!O(seq)->v.s.mut)
            return apy_fail("TypeError",
                            "'bytes' object does not support item assignment");
        if (!apy_index_arg(index, &i, APY_IDX_SUB)) return 0;
        n = O(seq)->v.s.n;
        if (i < 0) i += n;
        if (i < 0 || i >= n)
            return apy_fail("IndexError", "bytearray index out of range");
        if (!apy_index_arg(item, &byte, APY_IDX_SUB)) return 0;
        if (byte < 0 || byte > 255)
            return apy_fail("ValueError", "byte must be in range(0, 256)");
        /* The const is the STR half of the shared layout talking; this buffer
           came from `apy_bytes_copy` and is this object's to write. */
        ((char *)O(seq)->v.s.p)[i] = (char)byte;
        return apy_none();
    }
    if (O(seq)->kind != APY_LIST_K)
        return apy_fail2("TypeError",
                         "'%s' object does not support item assignment%s",
                         apy_kind_name(seq), "");
    if (!apy_is_int_like(index))
        return apy_fail2("TypeError",
                         "list indices must be integers or slices, not %s%s",
                         apy_kind_name(index), "");
    if (!apy_index_arg(index, &i, APY_IDX_SUB)) return 0;
    n = O(seq)->v.q.n;
    if (i < 0) i += n;
    if (i < 0 || i >= n)
        return apy_fail("IndexError", "list assignment index out of range");
    O(seq)->v.q.items[i] = item;
    return apy_none();
}

/* The length as a machine word, for the frontend's own use -- a `for` loop
   bound, not a value the program ever sees. `apy_len` is the builtin. */
APY_API int64_t apy_raw_len(apy_value v) {
    if (O(v)->kind == APY_VIEW_K)
        return O(O(v)->v.vw.dict)->v.d.n;
    if (O(v)->kind == APY_MVIEW_K) {
        if (!apy_mview_live(v)) return 0;
        /* IN ELEMENTS, NOT BYTES: eight bytes read as `i` are two. */
        return O(v)->v.mv.n / (O(v)->v.mv.wide ? O(v)->v.mv.wide : 1);
    }
    if (O(v)->kind == APY_RANGE_K) return apy_range_len(v);
    if (O(v)->kind == APY_DICT_K) return O(v)->v.d.n;
    if (apy_is_seq(v) || apy_is_set(v)) return O(v)->v.q.n;
    if (O(v)->kind == APY_BYTES_K) return O(v)->v.s.n;
    /* IN CHARACTERS, matching `apy_len` and matching what indexing counts.
       This used to answer BYTES while `apy_len` answered characters, and the
       two disagreeing was written down as a limitation -- but a `for` loop
       takes its bound from here and its elements from the subscript, so a
       string with any non-ASCII character in it walked off the end. */
    if (O(v)->kind == APY_STR_K) return apy_str_chars(v);
    if (O(v)->kind == APY_ITER_K) {
        /* A PLAIN cursor over a real container knows WHAT REMAINS without
           walking it, and answering cheaply keeps ordinary code off the slow
           path. Anything that transforms as it goes -- map, filter, zip --
           does not: filtering may drop any number of elements, so the only
           way to know is to walk. Those are DRAINED, which turns the cursor
           into a plain one over what it produced, exactly as asking a
           generator for its length does. */
        int src_kind = O(O(v)->v.it.src)->kind;
        if (O(v)->v.it.mode == APY_IT_PLAIN && src_kind != APY_GEN_K
                && src_kind != APY_ITER_K && src_kind != APY_INST_K) {
            int64_t n = apy_raw_len(O(v)->v.it.src);
            return n > O(v)->v.it.i ? n - O(v)->v.it.i : 0;
        }
        {
            apy_value got = apy_drain_cursor(v);
            if (!got) return 0;
            return O(got)->v.q.n;
        }
    }
    /* A GENERATOR has no length until it has been run, so asking for one
       DRAINS it -- see the `cache` field. `apy_key_at` then reads the same
       list, so the length and the elements agree. */
    if (O(v)->kind == APY_GEN_K) {
        if (!O(v)->v.g.cache) {
            apy_value got = apy_gen_drain(v);
            if (!got) return 0;
            O(v)->v.g.cache = got;
        }
        return O(O(v)->v.g.cache)->v.q.n;
    }
    /* A user object with `__len__`. Together with `apy_key_at` falling
       through to `__getitem__`, that is the whole `__len__`/`__getitem__`
       iteration protocol -- the one a sequence-like class implements, and the
       one this runtime's index-based walk fits exactly. */
    if (O(v)->kind == APY_INST_K) {
        apy_value n = apy_unary_dunder(v, "__len__");
        if (n && apy_is_int_like(n)) return O(n)->v.i;
        if (apy_error_occurred()) return 0;
        /* A CLASS EXTENDING A BUILTIN has the builtin's length, which is what
           bounds the index walk. `a, b, c = t` on a `class T(tuple)` takes
           its count from here, so without this an unpack refused before it
           started -- while `for x in t` worked, the same elements by another
           road. */
        if (O(v)->v.o.held) return apy_raw_len(O(v)->v.o.held);
    }
    apy_fail2("TypeError", "'%s' object is not iterable%s",
              apy_kind_name(v), "");
    return 0;
}

"""
