"""The object runtime, in C: extraction, inspection, repr and str.

ONE PART OF ONE TRANSLATION UNIT. `c/__init__.py` concatenates
these in order and the result is the file it always was, so a
definition here may rely on anything in an earlier part and
nothing in a later one. Sections, in order:
  * extraction
  * inspection
  * repr and str
"""

C = r"""
/* Defined in a LATER part; see `apy_mview_live` there. */
APY_API int64_t apy_mview_live(apy_value v);
/* --- extraction -------------------------------------------------------- */
/* The frontend calls these only where it has proved the kind, so they do not
   check. A wrong call here is a compiler bug, not a user error, and a check
   would hide it behind a plausible zero. */
APY_API int64_t apy_as_int(apy_value v) { return O(v)->v.i; }

/* A VALUE AS AN INDEX, checked. `apy_as_int` is a raw extraction the frontend
   calls where it has proved the kind; this is for the places where the value
   came from the program -- a slice bound, a `range` argument -- and may be
   anything, including a user object with `__index__`.

   A wrong kind reports rather than reading whatever the union happens to hold
   at that offset, which for an instance is its class pointer. */
/* Defined just below; the bound converter calls it. */
APY_API int64_t apy_index(apy_value v);
/* DECLARED AHEAD: both live past the generated method table, which is
   spliced into `_calling.py` -- see `apy_kind_name_of` for what they decide. */
static int64_t apy_kind_meth_written(const char *w, unsigned bit);
static unsigned apy_kind_bit(apy_value v);
APY_API apy_value apy_kind_prototype(apy_value type_name);

/* A SLICE BOUND, WHICH IS NOT AN INDEX. `xs[2 ** 100]` is a request this
   runtime cannot serve and CPython refuses it too; `xs[:2 ** 100]` is the
   whole list, and refusing THAT would be wrong. So a big clamps to a value
   past any real length rather than raising, keeping its sign so that
   `xs[-(2 ** 100):]` is the whole list as well.

   `apy_index` cannot make this distinction because it does not know which it
   was asked for -- which is why the frontend picks the converter rather than
   the converter guessing. */
APY_API int64_t apy_slice_bound(apy_value v) {
    /* AN EXPLICIT `None` IS NOT A BOUND, and the caller pairs this with
       `apy_slice_given`, which reports that it was not given at all. Zero
       is answered rather than refused because refusing is what
       `xs[None:None]` used to do. */
    if (O(v)->kind == APY_NONE_K)
        return 0;
    if (apy_is_big(v))
        return O(v)->v.big.neg ? -((int64_t)1 << 62) : ((int64_t)1 << 62);
    if (apy_is_int_like(v))
        return O(v)->v.i;
    if (O(v)->kind == APY_INST_K) {
        apy_value got = apy_unary_dunder(v, "__index__");
        if (apy_error_occurred()) return 0;
        if (got) {
            /* A BIG FROM `__index__` CLAMPS TOO. The bound rule is about the
               POSITION, not about where the number came from. */
            if (apy_is_big(got))
                return O(got)->v.big.neg ? -((int64_t)1 << 62)
                                         : ((int64_t)1 << 62);
            if (apy_is_int_like(got)) return O(got)->v.i;
        }
    }
    /* NOT `apy_index`'s MESSAGE. CPython words a bad slice bound differently
       from a bad index, and now that None is accepted the difference
       matters: its wording NAMES None as one of the things a bound may be. */
    apy_fail("TypeError",
             "slice indices must be integers or None or have an "
             "__index__ method");
    return 0;
}

/* Whether this bound was GIVEN, in the sense a slice means.

   `xs[None:None]` IS `xs[:]`. Python reads an explicitly written `None`
   bound exactly as it reads an omitted one -- which is also why
   `slice(None, None, None)` is the object a bare `[:]` builds, and why
   `xs[None::2]` steps from the start rather than raising.

   THE FRONTEND KNOWS WHICH BOUNDS WERE WRITTEN and cannot know which of them
   EVALUATE to None: `xs[a:b]` is a pair of expressions. So it settles the
   first half at compile time and this settles the second at run time. */
APY_API int64_t apy_slice_given(apy_value v) {
    return O(v)->kind == APY_NONE_K ? 0 : 1;
}

/* A slice's step, where `None` means ONE rather than nothing.

   Separate from the bound because the default differs and because a step of
   zero is an error: `xs[::None]` is `xs[::1]`, and answering 0 for the None
   would turn it into one. */
APY_API int64_t apy_slice_step(apy_value v) {
    return O(v)->kind == APY_NONE_K ? 1 : apy_slice_bound(v);
}

APY_API int64_t apy_index(apy_value v) {
    /* THE BIG TEST COMES FIRST, and it has to: `apy_is_int_like` is TRUE for
       a big -- that is the whole point of it, since a big is an integer --
       so testing it first sent every big down the fast path and returned
       `v.i`, which on a big is the LIMB POINTER read as an integer.

       Nothing crashed. A slice bound became a large positive address, so
       `xs[-(2 ** 100):]` was empty and `xs[:-(2 ** 100)]` was the whole
       list -- exactly inverted, and silent. The refusal below was
       unreachable. */
    if (apy_is_big(v)) {
        apy_fail("OverflowError",
                 "cannot fit 'int' into an index-sized integer");
        return 0;
    }
    if (apy_is_int_like(v)) return O(v)->v.i;
    if (O(v)->kind == APY_INST_K) {
        apy_value got = apy_unary_dunder(v, "__index__");
        if (apy_error_occurred()) return 0;
        /* AND THE SAME TEST ON WHAT `__index__` ANSWERED, which may be a big
           just as easily as the argument was. */
        if (got && apy_is_big(got)) {
            apy_fail("OverflowError",
                     "cannot fit 'int' into an index-sized integer");
            return 0;
        }
        if (got && apy_is_int_like(got)) return O(got)->v.i;
    }
    apy_fail2("TypeError",
              "'%s' object cannot be interpreted as an integer%s",
              apy_kind_name(v), "");
    return 0;
}
/* KIND-AWARE, unlike its int and bool neighbours. Those are raw extractions
   the frontend only emits where it has proved the kind; this one is reached
   with an int whenever a program passes one to a `float` parameter, which
   Python allows and people write. Reading `v.f` there reinterpreted the
   integer bits as a double and `f(42)` answered 4.15e-322. */
APY_API double apy_as_float(apy_value v) {
    if (O(v)->kind == APY_FLOAT_K) return O(v)->v.f;
    if (apy_is_big(v)) return apy_big_double(O(v));
    if (apy_is_int_like(v)) return (double)O(v)->v.i;
    return O(v)->v.f;
}
APY_API int64_t apy_as_bool(apy_value v) { return O(v)->v.i != 0; }

/* --- inspection -------------------------------------------------------- */
/* `b'ab'`, with CPython's escaping rules.

   Which are NOT the same as str's, and the differences are the whole function:
   every byte outside printable ASCII becomes `\\xNN` (never `\\uNNNN`, since
   there is no character here to have a code point), `\\t`, `\\n` and `\\r` keep
   their short forms, and the quote is single unless the value contains one and
   no double. */
APY_API apy_value apy_bytes_repr(apy_value v) {
    const unsigned char *p = (const unsigned char *)O(v)->v.s.p;
    if (O(v)->v.s.mut) {
        /* `bytearray(b'abc')` -- the repr of the bytes it holds, wrapped.
           Built by clearing the flag round the recursive call rather than by
           a second escaping loop, so the two spellings cannot drift. */
        apy_value inner;
        char *wrapped;
        int64_t m;
        O(v)->v.s.mut = 0;
        inner = apy_bytes_repr(v);
        O(v)->v.s.mut = 1;
        if (!inner) return 0;
        m = O(inner)->v.s.n;
        wrapped = (char *)malloc((size_t)m + 12);
        if (!wrapped) { fputs("uasm: out of memory\n", stderr); exit(1); }
        memcpy(wrapped, "bytearray(", 10);
        memcpy(wrapped + 10, O(inner)->v.s.p, (size_t)m);
        wrapped[10 + m] = ')';
        wrapped[11 + m] = 0;
        return apy_str_take(wrapped, m + 11);
    }
    int64_t n = O(v)->v.s.n, i;
    int has_single = 0, has_double = 0;
    for (i = 0; i < n; i++) {
        if (p[i] == '\'') has_single = 1;
        if (p[i] == '"') has_double = 1;
    }
    char quote = (has_single && !has_double) ? '"' : '\'';

    /* Four characters is the widest any one byte becomes (`\\xNN`), plus the
       quotes and the `b`. */
    int64_t cap = n * 4 + 4;
    char *out = (char *)malloc((size_t)cap + 1);
    if (!out) { fputs("uasm: out of memory\n", stderr); exit(1); }
    int64_t k = 0;
    out[k++] = 'b';
    out[k++] = quote;
    for (i = 0; i < n; i++) {
        unsigned char c = p[i];
        if (c == (unsigned char)quote || c == '\\') {
            out[k++] = '\\'; out[k++] = (char)c;
        } else if (c == '\t') { out[k++] = '\\'; out[k++] = 't';
        } else if (c == '\n') { out[k++] = '\\'; out[k++] = 'n';
        } else if (c == '\r') { out[k++] = '\\'; out[k++] = 'r';
        } else if (c >= 32 && c < 127) {
            out[k++] = (char)c;
        } else {
            static const char *hex = "0123456789abcdef";
            out[k++] = '\\'; out[k++] = 'x';
            out[k++] = hex[c >> 4]; out[k++] = hex[c & 15];
        }
    }
    out[k++] = quote;
    out[k] = 0;
    return apy_str_take(out, k);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */

/* THE EXPORTED HALF, which `runtime/kindname.py` splits: the IR answers
   every kind but an exception, and an exception comes back here because its
   DISPLAYED name is a class lookup rather than a literal. */
APY_API apy_value apy_kind_name_of(apy_value v) {
    return (apy_value)(uintptr_t)apy_kind_name(v);
}
/* WHAT A PLAIN OR REVERSED CURSOR IS CALLED, which CPython takes from what
   it walks: `iter([1])` is a `list_iterator`, `iter(d.values())` a
   `dict_valueiterator`, `reversed([1])` a `list_reverseiterator`. The name
   is what `type(it).__name__` answers and what its repr prints, and a
   program that logs either sees the difference.

   READ FROM `named` AND NOT FROM `src`, because the source does not always
   survive: a view's items are copied out at construction, and a drained
   `map` becomes a plain cursor over the list it produced. See
   `APY_IT_VIEWED`.

   `str_ascii_iterator` IS A NAME OF ITS OWN IN CPYTHON, which records the
   width on the string. A str here is UTF-8 bytes, so "every character is one
   byte" is the same question as "no byte has its high bit set" -- asked here,
   where nothing reads the answer in a loop, rather than paid for on every
   `for c in s`. */
/* THE TYPE A DESCRIPTOR CAME OFF, as a value of that kind. A descriptor
   carries its owner as the HEAD OF ITS QUALNAME -- `list` out of
   `list.append` -- where a bound method carries a receiver; both questions
   below want a value to ask `apy_kind_bit` about, and this is where one
   comes from. 0 for a name no builtin kind answers to. */
static apy_value apy_descr_owner(apy_value v) {
    char owner[64];
    const char *q = O(v)->v.fn.qualname ? APY_CSTR(O(v)->v.fn.qualname) : 0;
    const char *dot = q ? strchr(q, '.') : 0;
    size_t n;
    if (!dot) return 0;
    n = (size_t)(dot - q);
    if (n == 0 || n >= sizeof owner) return 0;
    memcpy(owner, q, n);
    owner[n] = 0;
    return apy_kind_prototype((apy_value)(uintptr_t)owner);
}

/* Does the owner WRITE this dunder out, or fill a slot with it? That is the
   whole difference between a `method_descriptor` and a `wrapper_descriptor`,
   and nothing in the signature says which: `list.__getitem__` is written out
   and `tuple.__getitem__` is slotted. Read out of CPython per kind and per
   name -- see `apy_kind_meth_written`, which the bound half asks too. */
/* Generated, and spliced into a LATER part -- see `_gen_kindmeth.py`. */
static int apy_cursor_meth_written(const char *w);

static int apy_descr_written(apy_value v) {
    apy_value proto = apy_descr_owner(v);
    if (!proto) return 0;
    /* A CURSOR HAS NO BIT. The table above is keyed by the thirteen builtin
       kinds' bits and `apy_kind_bit` of a cursor is 0, so every name read
       off one came back "slotted" -- and `type(it).__length_hint__` printed
       as a slot wrapper where CPython prints a method. One row answers for
       every cursor; see `apy_cursor_meth_written`. */
    if (O(proto)->kind == APY_ITER_K)
        return apy_cursor_meth_written(APY_CSTR(O(v)->v.fn.name));
    return apy_kind_meth_written(APY_CSTR(O(v)->v.fn.name),
                                 apy_kind_bit(proto)) != 0;
}

static const char *apy_cursor_name(apy_value v) {
    apy_value src = O(v)->v.it.src;
    /* REVERSEDNESS IS IN `named` AND NOT IN THE MODE, because a length query
       DRAINS a cursor and resets its mode to plain -- so
       `len(reversed(xs))`, of all things, would have renamed it. */
    int rev = O(v)->v.it.named >= APY_IT_REVOF;
    switch (O(v)->v.it.named - (rev ? APY_IT_REVOF : 0)) {
    case APY_IT_VIEWED + APY_PART_KEYS:
        return rev ? "dict_reversekeyiterator" : "dict_keyiterator";
    case APY_IT_VIEWED + APY_PART_VALUES:
        return rev ? "dict_reversevalueiterator" : "dict_valueiterator";
    case APY_IT_VIEWED + APY_PART_ITEMS:
        return rev ? "dict_reverseitemiterator" : "dict_itemiterator";
    case APY_IT_CALLABLE: return "callable_iterator";
    case APY_LIST_K:  return rev ? "list_reverseiterator" : "list_iterator";
    case APY_TUPLE_K: return rev ? "reversed" : "tuple_iterator";
    case APY_DICT_K:
        return rev ? "dict_reversekeyiterator" : "dict_keyiterator";
    /* A RANGE REVERSED IS STILL A RANGE ITERATOR: CPython answers
       `range_iterator` both ways, because `reversed(r)` hands back a walk
       over the range with its step negated rather than a wrapper around it. */
    case APY_RANGE_K: return "range_iterator";
    case APY_SET_K:
    case APY_FROZEN_K: return "set_iterator";
    case APY_MVIEW_K: return rev ? "reversed" : "memory_iterator";
    case APY_BYTES_K:
        if (rev) return "reversed";
        /* A bytearray and a bytes share the kind, so the mutable flag
           decides -- read from the source while it is still there. */
        return (src && O(src)->kind == APY_BYTES_K && O(src)->v.s.mut)
            ? "bytearray_iterator" : "bytes_iterator";
    case APY_STR_K:
        if (rev) return "reversed";
        if (src && O(src)->kind == APY_STR_K) {
            int64_t i;
            for (i = 0; i < O(src)->v.s.n; i++)
                if ((unsigned char)O(src)->v.s.p[i] >= 0x80)
                    return "str_iterator";
        }
        return "str_ascii_iterator";
    default: break;
    }
    return rev ? "reversed" : "iterator";
}

static const char *apy_kind_name(apy_value v) {
    switch (O(v)->kind) {
    case APY_NONE_K:  return "NoneType";
    case APY_BOOL_K:  return "bool";
    case APY_INT_K:   return "int";
    case APY_FLOAT_K: return "float";
    /* A READ-ONLY DICT IS A `mappingproxy`, which is what `C.__dict__`
       answers and what `type()` of it says. See the `ro` flag. */
    case APY_DICT_K:  return O(v)->v.d.ro ? "mappingproxy" : "dict";
    case APY_EXC_K:   return apy_exc_shown(O(v)->v.e.name);
    case APY_LIST_K:  return "list";
    case APY_TUPLE_K: return "tuple";
    case APY_SET_K:   return "set";
    case APY_FROZEN_K: return "frozenset";
    /* A big is an `int`. There is one integer type in Python and the width is
       an implementation detail this file is deliberately hiding -- a program
       that can tell `2 ** 100` from `5` by its type name is seeing a seam
       that should not exist. */
    case APY_BIG_K:   return "int";
    /* An instance answers with its CLASS's name, which is what makes
       `type(p).__name__` say `Point` and every TypeError about a user object
       name the user's type rather than a word from this file. */
    case APY_INST_K:  return APY_CSTR(O(O(v)->v.o.cls)->v.t.name);
    case APY_TYPE_K:  return "type";
    case APY_FUNC_K:
        if (O(v)->v.fn.is_type) return "type";
        /* REACHED OFF THE TYPE, which CPython calls a DESCRIPTOR and not a
           function: `list.append` is a `method_descriptor` and `list.__len__`
           a `wrapper_descriptor`. The two differ by whether the type WRITES
           the method out or fills a slot with it -- the same question the
           bound half asks below, and answered from the same table. */
        if (O(v)->v.fn.descr) {
            const char *w = APY_CSTR(O(v)->v.fn.name);
            size_t len = strlen(w);
            if (len >= 5 && w[0] == '_' && w[1] == '_'
                    && w[len - 1] == '_' && w[len - 2] == '_'
                    && !apy_descr_written(v))
                return "wrapper_descriptor";
            return "method_descriptor";
        }
        if (O(v)->v.fn.builtin) return "builtin_function_or_method";
        /* A NATIVE IS THE RUNTIME'S OWN CODE, not a compiled function --
           `[1].index` and `[1].__len__` are `builtin_function_or_method` and
           `method-wrapper` in Python, and both answered `function` here. The
           DUNDER is the one that differs: Python calls a bound slot a
           method-wrapper, whatever the slot is. */
        if (O(v)->v.fn.native) {
            const char *w = APY_CSTR(O(v)->v.fn.name);
            size_t len = strlen(w);
            /* A DUNDER WITH A RANGE IS NOT A SLOT. `(5).__round__` is a
               `builtin_function_or_method` in CPython because int writes the
               method out rather than filling a slot, and an optional argument
               is exactly what a slot cannot carry. */
            if (!O(v)->v.fn.defaults && O(v)->v.fn.ndefaults)
                return "builtin_function_or_method";
            if (len >= 5 && w[0] == '_' && w[1] == '_'
                    && w[len - 1] == '_' && w[len - 2] == '_') {
                /* AND THE ARITY WAS ONLY STANDING IN FOR THE REAL QUESTION,
                   which is whether the type WRITES the method out or fills a
                   slot with it -- `list.__getitem__` takes exactly one
                   argument and is written out, `tuple.__getitem__` is
                   slotted, and nothing in either signature says so. The
                   answer is read out of CPython per kind and per name; see
                   `apy_kind_meth_written`. */
                if (O(v)->v.fn.bound
                        && apy_kind_meth_written(w,
                                                 apy_kind_bit(O(v)->v.fn.bound)))
                    return "builtin_function_or_method";
                /* A SLOT IS FILLED ON A VALUE, never on a TYPE. What binds
                   to a type is a CLASSMETHOD -- `list.__class_getitem__` is
                   the only one here -- and CPython calls a bound one a
                   `builtin_function_or_method` whatever its name looks
                   like. The kind-and-name table cannot say so: a type has no
                   kind bit of its own. */
                if (O(v)->v.fn.bound
                        && ((O(O(v)->v.fn.bound)->kind == APY_FUNC_K
                             && O(O(v)->v.fn.bound)->v.fn.is_type)
                            || O(O(v)->v.fn.bound)->kind == APY_TYPE_K))
                    return "builtin_function_or_method";
                return "method-wrapper";
            }
            return "builtin_function_or_method";
        }
        /* A BOUND ONE IS A `method`, a type of its own in CPython:
           `type(C().m).__name__` is `method` where `type(C.m).__name__` is
           `function`. The receiver is the whole difference. */
        if (O(v)->v.fn.bound) return "method";
        return "function";
    case APY_CELL_K:  return "cell";
    case APY_SUPER_K: return "super";
    case APY_BYTES_K: return O(v)->v.s.mut ? "bytearray" : "bytes";
    case APY_COMPLEX_K: return "complex";
    /* A CURSOR names what MADE it: `map(str, xs)` is a `map`, which is what
       `type(...).__name__` answers and what tells a reader why it is lazy.
       A plain or reversed one is named after what it WALKS, as CPython names
       those -- see `apy_cursor_name`. */
    case APY_ITER_K:
        switch (O(v)->v.it.mode) {
        case APY_IT_MAP:       return "map";
        case APY_IT_FILTER:    return "filter";
        case APY_IT_ENUMERATE: return "enumerate";
        case APY_IT_ZIP:       return "zip";
        default:               return apy_cursor_name(v);
        }
    case APY_ELLIPSIS_K: return "ellipsis";
    case APY_NOTIMPL_K: return "NotImplementedType";
    /* All three share every field; only the name differs, and a program reads
       it to tell them apart -- `async def` with `yield` is an async
       generator, which is neither of the other two. */
    case APY_SLICE_K: return "slice";
    case APY_ALIAS_K:
        /* A UNION IS NOT A GENERIC ALIAS to a program that asks. `int | str`
           is built on the `Union` form, and `type(...).__name__` is how a
           program tells the two apart. */
        return O(O(v)->v.ga.origin)->kind == APY_INST_K
            ? "typing.Union" : "types.GenericAlias";
    case APY_MVIEW_K: return "memoryview";
    case APY_RANGE_K: return "range";
    case APY_VIEW_K:
        switch (O(v)->v.vw.part) {
        case APY_PART_KEYS:   return "dict_keys";
        case APY_PART_VALUES: return "dict_values";
        default:              return "dict_items";
        }
    case APY_PROP_K:
        switch (O(v)->v.p.kind) {
        case APY_PROP_CLASSMETHOD:  return "classmethod";
        case APY_PROP_STATICMETHOD: return "staticmethod";
        default:                    return "property";
        }
    case APY_GEN_K:
        if (O(v)->v.g.wrapper) return "coroutine_wrapper";
        if (O(v)->v.g.agen) return "async_generator";
        return O(v)->v.g.coro ? "coroutine" : "generator";
    default:          return "str";
    }
}

APY_API apy_value apy_str_slice_of(apy_value s, int64_t lo, int64_t hi);

/* A type's `__name__` is the LAST COMPONENT of its dotted name.

   Two kinds here are named the way CPython names them in a message --
   `types.GenericAlias` and `typing.Union` -- because that is what
   `unsupported operand type(s) for +` prints and what `<class '...'>` shows.
   `__name__` is the other half of the same rule: CPython answers
   `GenericAlias` and `Union`, keeping the module in `__module__` and never in
   the name. One dotted string serves both, split here.

   THE SAME VALUE COMES BACK when there is no dot, which is what keeps
   `type(a).__name__ is type(b).__name__` true for two instances of one class.
   No identifier can hold a dot, so a class the program wrote never takes the
   slicing branch -- and one whose `__name__` was ASSIGNED a dotted string is
   sliced safely rather than read past its end, which is why the scan is
   bounded by the length rather than looking for a NUL. */
static apy_value apy_bare_name(apy_value name) {
    int64_t i;
    for (i = O(name)->v.s.n - 1; i >= 0; i--)
        if (O(name)->v.s.p[i] == '.')
            return apy_str_slice_of(name, i + 1, O(name)->v.s.n);
    return name;
}

/* PEP 3155: the class's `__qualname__`, which is its name unless a `class`
   statement was written inside a function or another class.

   THE FIELD WHEN THERE IS ONE, and the bare name otherwise -- a class at
   module level qualifies as itself, so the frontend records nothing for it
   and there is nothing to store. Read by the two reprs that qualify a name
   and by the `__qualname__` arm of `apy_default_getattr`; the ERROR
   MESSAGES deliberately do not read it, because CPython's say `'D' object`
   for a nested class and reserve the qualified spelling for its reprs. */
static apy_value apy_type_qualname(apy_value cls) {
    if (!cls || O(cls)->kind != APY_TYPE_K) return 0;
    if (O(cls)->v.t.qual) return O(cls)->v.t.qual;
    return apy_bare_name(O(cls)->v.t.name);
}

APY_API apy_value apy_type_name(apy_value v) {
    /* The class's own name value, not a fresh copy: `type(a).__name__ is
       type(b).__name__` for two instances of one class, as in CPython. */
    if (O(v)->kind == APY_INST_K) return O(O(v)->v.o.cls)->v.t.name;
    /* `type(C).__name__` IS THE METACLASS'S NAME when one made the class.
       An ordinary class has no metaclass recorded and is a `type`. */
    if (O(v)->kind == APY_TYPE_K)
        return O(v)->v.t.meta ? O(O(v)->v.t.meta)->v.t.name : apy_lit("type");
    {
        /* The kind name is a static literal, so the tail of a dotted one is
           a NUL-terminated string of its own and needs no copy. */
        const char *nm = apy_kind_name(v), *dot = strrchr(nm, '.');
        return apy_lit(dot ? dot + 1 : nm);
    }
}

APY_API int64_t apy_truth(apy_value v) {
    switch (O(v)->kind) {
    case APY_NONE_K:  return 0;
    case APY_BOOL_K:
    case APY_INT_K:   return O(v)->v.i != 0;
    case APY_FLOAT_K: return O(v)->v.f != 0.0;
    case APY_COMPLEX_K: return O(v)->v.z.re != 0.0 || O(v)->v.z.im != 0.0;
    case APY_DICT_K:  return O(v)->v.d.n != 0;
    case APY_EXC_K:   return 1;
    /* Never zero: a zero-valued big demotes to the int 0 on construction. */
    case APY_BIG_K:   return 1;
    case APY_LIST_K:
    case APY_TUPLE_K:
    case APY_SET_K:
    case APY_FROZEN_K: return O(v)->v.q.n != 0;
    case APY_INST_K: {
        /* `__bool__` first, then `__len__`, then true -- CPython's order, and
           the fallback matters: an object with neither is ALWAYS truthy, so a
           bare `if obj:` on a plain instance takes the then-branch. Answering
           0 there would silently invert every such test. */
        apy_value r = apy_unary_dunder(v, "__bool__");
        if (r) return apy_truth(r);
        if (apy_error_occurred()) return 0;
        /* A CLASS THAT EXTENDS A BUILTIN has one for everything it did not
           write -- `apy_len` below already asks `held` before walking the
           dunder for exactly this reason, and this had not: `class
           Counter(dict)` with no `__len__` of its own fell straight through
           to "no `__bool__` and no `__len__`", so `bool(Counter())` was
           ALWAYS True regardless of content. */
        if (apy_inst_held(v)
                && !apy_class_find(O(v)->v.o.cls, apy_name("__len__")))
            return apy_truth(apy_inst_held(v));
        r = apy_unary_dunder(v, "__len__");
        if (r) return apy_truth(r);
        return 1;
    }
    /* Emptiness is truth only for things that HAVE a length. */
    case APY_STR_K:
    case APY_BYTES_K: return O(v)->v.s.n != 0;
    /* Everything else -- a function, a type, an iterator, a cell -- is an
       object with no emptiness to speak of, and Python calls those true.
       Reading `v.s.n` for them read whatever field the union happened to
       overlap, which for a type is its base pointer: `if et:` on a caught
       exception's type answered FALSE for every class with no base, so
       `et.__name__ if et else None` in a `__exit__` reported None. */
    default:          return 1;
    }
}

/* A str is stored as UTF-8 BYTES, but Python's `len` counts CHARACTERS:
   `len('e')` is 1 and `len('é')` is also 1, while the byte counts are 1
   and 2. Counting bytes is right for pure ASCII and silently wrong for
   everything else, which is the worst shape a bug can have -- so count the
   bytes that are not UTF-8 continuation bytes (`10xxxxxx`), which is the
   codepoint count for any well-formed UTF-8 and degrades to the byte count
   for ASCII.

   This is the only place the byte/character distinction is resolved today.
   Indexing and slicing will need the same treatment when they arrive; they
   are not in v1, and pretending otherwise by leaving `len` in bytes would
   only hide the problem. */
static int64_t apy_str_chars(apy_value v) {
    const unsigned char *p = (const unsigned char *)O(v)->v.s.p;
    int64_t i, n = O(v)->v.s.n, chars = 0;
    /* A BYTES RECEIVER HAS NO CHARACTERS IN IT, so its count IS its byte
       count -- the two words mean the same thing for bytes in Python, and
       every caller here wants whichever the receiver's own unit is. Running
       the UTF-8 walk over one read `b"\xc3\xa9"` as ONE character, so
       `b"\xc3\xa9".center(9)` padded to nine CHARACTERS and answered ten
       bytes, and every bounded search counted the same way. */
    if (O(v)->kind != APY_STR_K) return n;
    for (i = 0; i < n; i++)
        if ((p[i] & 0xC0) != 0x80) chars++;
    return chars;
}

APY_API apy_value apy_len(apy_value v) {
    /* THE LENGTH OF A CLASS IS THE METACLASS'S BUSINESS, exactly as iterating
       one is: `len(Colour)` is `type(Colour).__len__(Colour)`, which is how an
       enum says how many members it has. Iteration grew this case and length
       did not, so `for c in Colour` worked and `len(Colour)` reported
       `object of type 'EnumMeta' has no len()` -- about a class whose
       metaclass plainly defines one. */
    if (O(v)->kind == APY_TYPE_K && O(v)->v.t.meta) {
        apy_value hook = apy_class_find(O(v)->v.t.meta, apy_name("__len__"));
        if (hook) return apy_call_n(apy_bind(hook, v), NULL, 0);
    }
    /* A CLASS THAT EXTENDS A BUILTIN has one for everything it did not write.
       Asked before the dunder walk below, which would report "has no len()"
       for a `class D(dict)` whose body says nothing about length. */
    if (O(v)->kind == APY_INST_K && apy_inst_held(v)
            && !apy_class_find(O(v)->v.o.cls, apy_name("__len__")))
        return apy_len(apy_inst_held(v));
    /* THROUGH THE VIEW to the dict: a view has no length of its own, and
       taking one when it was made is what a snapshot does. */
    if (O(v)->kind == APY_VIEW_K)
        return apy_from_int(O(O(v)->v.vw.dict)->v.d.n);
    if (O(v)->kind == APY_MVIEW_K) {
        if (!apy_mview_live(v)) return 0;
        /* IN ELEMENTS, NOT BYTES: eight bytes read as `i` are two. */
        return apy_from_int(O(v)->v.mv.n
                            / (O(v)->v.mv.wide ? O(v)->v.mv.wide : 1));
    }
    if (O(v)->kind == APY_RANGE_K) return apy_from_int(apy_range_len(v));
    if (O(v)->kind == APY_DICT_K) return apy_from_int(O(v)->v.d.n);
    if (apy_is_seq(v) || apy_is_set(v)) return apy_from_int(O(v)->v.q.n);
    if (O(v)->kind == APY_INST_K) {
        apy_value r = apy_unary_dunder(v, "__len__");
        if (r || apy_error_occurred()) return r;
        /* No `__len__` falls through to the same "has no len()" the runtime
           reports for an int, naming the user's class -- which is exactly
           what CPython says for an instance without one. */
    }
    /* bytes counts OCTETS and str counts characters, so this cannot fall
       through to the str arm below -- which measures characters. */
    if (O(v)->kind == APY_BYTES_K) return apy_from_int(O(v)->v.s.n);
    if (O(v)->kind != APY_STR_K)
        return apy_fail2("TypeError", "object of type '%s' has no len()%s",
                         apy_kind_name(v), "");
    return apy_from_int(apy_str_chars(v));
}

/* --- repr and str ------------------------------------------------------ */
/* `repr` quotes a string and `str` does not; everything else is the same for
   the kinds here. Python prints with str() and shows with repr(), and getting
   that backwards prints `'abc'` where CPython prints `abc`. */
/* DEFINED IN THE STRING PART, which is joined after this one into the same
   translation unit. `repr` walks by code point and asks whether a character
   is printable; both answers live where the Unicode table does. */
static int apy_cp_printable(uint32_t cp);
static int64_t apy_utf8_step(const unsigned char *p, int64_t n, int64_t i,
                             uint32_t *out);

static apy_value apy_text(apy_value v, int quoted);

/* What `__str__` or `__repr__` gave back, which MUST be a str.

   Converting it instead -- calling `apy_text` on the result -- looks more
   forgiving and is a trap: `def __str__(self): return self` would then
   recurse until the C stack ran out, and a stack overflow is not a diagnosis.
   CPython raises here, so this does, with CPython's wording. */
APY_API apy_value apy_text_result_of(apy_value r, apy_value whichv) {
    const char *which = (const char *)whichv;
    char buf[128];
    if (O(r)->kind == APY_STR_K) return r;
    snprintf(buf, sizeof buf, "%s returned non-string (type %s)",
             which, apy_kind_name(r));
    return apy_fail("TypeError", buf);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_text_result(apy_value r, const char *which) {
    return apy_text_result_of(r, (apy_value)(uintptr_t)which);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */

/* A container always shows its ELEMENTS with repr, whichever of str/repr was
   asked of the container: `print(['a'])` is `['a']`, not `[a]`. A one-element
   tuple keeps its trailing comma, because `(1)` is not a tuple. */
/* CONTAINERS CURRENTLY BEING RENDERED. A list that holds itself is an
   ordinary thing to build -- `xs.append(xs)` -- and rendering it naively
   recurses until the stack runs out. Python prints `[...]` for the repeat,
   which is what makes the output finite and readable.

   A small array rather than a set: the depth of a repr is a handful of frames
   in every real case, and a linear scan of it is cheaper than a hash. */
static apy_value apy_repr_active[64];
static int apy_repr_depth;

APY_API int64_t apy_repr_entered(apy_value v) {
    int i;
    for (i = 0; i < apy_repr_depth; i++)
        if (apy_repr_active[i] == v) return 1;
    if (apy_repr_depth < 64) apy_repr_active[apy_repr_depth++] = v;
    return 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */

APY_API void apy_repr_left(apy_value v) {
    if (apy_repr_depth > 0 && apy_repr_active[apy_repr_depth - 1] == v)
        apy_repr_depth--;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */

APY_API apy_value apy_seq_text_of(apy_value v) {
    int tup = O(v)->kind == APY_TUPLE_K;
    int64_t n, i, len = 2, out = 0;
    apy_value *parts;
    char *buf;
    /* ALREADY BEING RENDERED: this is the cycle, and Python writes `[...]`
       for it. A tuple that contains itself cannot be built directly but can
       be reached through a list, so both spellings are needed. */
    if (apy_repr_entered(v)) return apy_lit(tup ? "(...)" : "[...]");
    n = O(v)->v.q.n;
    parts = (apy_value *)malloc((size_t)(n ? n : 1) * sizeof(apy_value));
    for (i = 0; i < n; i++) {
        parts[i] = apy_text(O(v)->v.q.items[i], 1);
        len += O(parts[i])->v.s.n + 2;
    }
    apy_repr_left(v);
    if (tup && n == 1) len += 1;
    buf = (char *)malloc((size_t)len + 1);
    buf[out++] = tup ? '(' : '[';
    for (i = 0; i < n; i++) {
        if (i) { buf[out++] = ','; buf[out++] = ' '; }
        memcpy(buf + out, O(parts[i])->v.s.p, (size_t)O(parts[i])->v.s.n);
        out += O(parts[i])->v.s.n;
    }
    if (tup && n == 1) buf[out++] = ',';
    buf[out++] = tup ? ')' : ']';
    buf[out] = '\0';
    free(parts);
    return apy_str_take(buf, out);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now, and
   the exported half above stands in when nothing is ported. */
static apy_value apy_seq_text(apy_value v) { return apy_seq_text_of(v); }

/* THE MODULE A CLASS WAS WRITTEN IN, for the two reprs that qualify a name:
   `<class '__main__.C'>` and `<__main__.C object at 0x...>`. Answers 0 for a
   class with none -- and for `builtins`, which CPython leaves OUT: `int` is
   `<class 'int'>` and not `<class 'builtins.int'>`.

   The class body puts it in the dict, which is where CPython keeps it too;
   see `_dyn_class`. */
static const char *apy_class_module(apy_value cls) {
    apy_value held;
    if (!cls || O(cls)->kind != APY_TYPE_K || !O(cls)->v.t.dict) return 0;
    held = apy_dict_get_or(O(cls)->v.t.dict, apy_lit("__module__"), 0);
    if (!held || O(held)->kind != APY_STR_K) return 0;
    if (strcmp(APY_CSTR(held), "builtins") == 0) return 0;
    return APY_CSTR(held);
}

APY_API apy_value apy_text_of(apy_value v, int64_t quoted) {
    /* WIDE ENOUGH FOR THE LONGEST SHAPE, which is a bound builtin method's
       `<built-in method tobytes of memoryview object at 0x...>` -- at 64 it
       was TRUNCATED, and a repr missing its closing bracket is a wrong
       answer that reads like a right one. */
    char buf[192];
    switch (O(v)->kind) {
    /* IGNORES `quoted`. `str(b'ab')` is "b'ab'" in Python 3 -- bytes has no
       separate str, which is the wart CPython emits a BytesWarning about
       under -b. Reproducing it means `print(b'ab')` shows the repr, and a
       `str()` that stripped the prefix would disagree with CPython on every
       line that printed one. */
    case APY_BYTES_K: return apy_bytes_repr(v);
    case APY_NONE_K: return apy_lit("None");
    case APY_BOOL_K: return apy_lit(O(v)->v.i ? "True" : "False");
    case APY_INT_K:
        snprintf(buf, sizeof buf, "%lld", (long long)O(v)->v.i);
        return apy_str_copy(buf, (int64_t)strlen(buf));
    case APY_BIG_K:   return apy_big_text((apy_value)O(v));
    case APY_COMPLEX_K: {
        /* CPython's rules exactly, and they are fussier than they look:

             1+2j   -> "(1+2j)"     parenthesised, sign always shown
             2j     -> "2j"         a ZERO real part is omitted, and so are
                                    the brackets
             -0+2j  -> "(-0+2j)"    but only a POSITIVELY signed zero is
                                    omitted; `-0.0` is a real part
             1-2j   -> "(1-2j)"
             0j     -> "0j"

           The sign test is on the BIT, not the value, because `-0.0 == 0.0`
           and the two print differently. Writing this as "if re is zero" made
           `complex(-0.0, 2)` print `2j`, which reads back as a different
           number. */
        char rbuf[64], ibuf[64];
        double re = O(v)->v.z.re, im = O(v)->v.z.im;
        int re_is_pos_zero = (re == 0.0) && !signbit(re);
        apy_complex_part(ibuf, sizeof ibuf, im);
        if (re_is_pos_zero) {
            snprintf(buf, sizeof buf, "%sj", ibuf);
            return apy_str_copy(buf, (int64_t)strlen(buf));
        }
        apy_complex_part(rbuf, sizeof rbuf, re);
        /* The imaginary part carries its own sign when negative, so the `+`
           is only written when it does not. `nan` has no sign to read, and
           CPython writes `+nanj`; `signbit` on a nan is unreliable, so the
           leading character of the rendered text is what decides. */
        if (ibuf[0] == '-')
            snprintf(buf, sizeof buf, "(%s%sj)", rbuf, ibuf);
        else
            snprintf(buf, sizeof buf, "(%s+%sj)", rbuf, ibuf);
        return apy_str_copy(buf, (int64_t)strlen(buf));
    }
    case APY_FLOAT_K:
        py_repr_double(buf, sizeof buf, O(v)->v.f);
        return apy_str_copy(buf, (int64_t)strlen(buf));
    case APY_DICT_K: {
        /* A READ-ONLY DICT WEARS ITS NAME. `repr(C.__dict__)` is
           `mappingproxy({...})` in CPython -- the mapping's own repr inside
           the wrapper's, which is what the wrapper is. */
        apy_value inner = apy_dict_text(v);
        if (!O(v)->v.d.ro || !inner) return inner;
        {
            int64_t n = O(inner)->v.s.n;
            char *out = (char *)malloc((size_t)n + 16);
            memcpy(out, "mappingproxy(", 13);
            memcpy(out + 13, O(inner)->v.s.p, (size_t)n);
            out[13 + n] = ')';
            out[14 + n] = 0;
            return apy_str_take(out, n + 14);
        }
    }
    case APY_RANGE_K: {
        /* `range(0, 10, 2)` -- and `range(0, 3)` when the step is 1, which
           is how CPython prints one. */
        char rbuf[96];
        int wrote;
        if (O(v)->v.rg.step == 1)
            wrote = snprintf(rbuf, sizeof rbuf, "range(%lld, %lld)",
                             (long long)O(v)->v.rg.start,
                             (long long)O(v)->v.rg.stop);
        else
            wrote = snprintf(rbuf, sizeof rbuf, "range(%lld, %lld, %lld)",
                             (long long)O(v)->v.rg.start,
                             (long long)O(v)->v.rg.stop,
                             (long long)O(v)->v.rg.step);
        return apy_str_copy(rbuf, wrote);
    }
    case APY_ELLIPSIS_K: return apy_lit("Ellipsis");
    case APY_NOTIMPL_K: return apy_lit("NotImplemented");
    case APY_ALIAS_K: {
        /* `list[int]`, not `list[<class 'int'>]`. A TYPE ARGUMENT RENDERS AS
           ITS NAME here even though `str(int)` is `<class 'int'>` -- CPython's
           alias repr uses the qualname, and the difference is visible in
           every annotation a program prints. */
        apy_value origin = O(v)->v.ga.origin;
        /* THE UNION IS THE ONLY FORM THAT PRINTS WITH BARS. PEP 604 made `int
           | str` the spelling for that one; every other form keeps the
           subscript it was written with, and testing "the origin is an
           instance" made `Annotated[int, 'x']` print as a union. */
        apy_value form_nm = O(origin)->kind == APY_INST_K
            ? apy_dict_get_or(O(origin)->v.o.dict, apy_lit("_name"), 0) : 0;
        int is_union = form_nm && strcmp(APY_CSTR(form_nm), "Union") == 0;
        apy_value head = (O(origin)->kind == APY_FUNC_K
                          && O(origin)->v.fn.is_type) ? O(origin)->v.fn.name
                         : O(origin)->kind == APY_TYPE_K
                           ? O(origin)->v.t.name : apy_text(origin, 0);
        apy_value args = O(v)->v.ga.args;
        int64_t i, n = apy_is_seq(args) ? O(args)->v.q.n : 0;
        int64_t room = O(head)->v.s.n + 8;
        char *out;
        int64_t at;
        apy_value *parts = (apy_value *)malloc(
            (size_t)(n ? n : 1) * sizeof(apy_value));
        for (i = 0; i < n; i++) {
            apy_value one = O(args)->v.q.items[i];
            /* `NoneType` IS SPELLED `None` INSIDE A UNION, which is how
               CPython prints `int | None` -- the class is what the union
               HOLDS and `None` is what it is written as. */
            parts[i] = (O(one)->kind == APY_TYPE_K
                        && strcmp(APY_CSTR(O(one)->v.t.name), "NoneType") == 0)
                ? apy_lit("None")
                : O(one)->kind == APY_TYPE_K ? O(one)->v.t.name
                : (O(one)->kind == APY_FUNC_K && O(one)->v.fn.is_type)
                    ? O(one)->v.fn.name
                    : apy_text(one, 1);
            room += O(parts[i])->v.s.n + 2;
        }
        /* ROOM FOR THE STAR that an unpacked alias prints with. Counted
           here rather than at the memcpy, because `room` is what the
           allocation is sized by. */
        room += 1;
        out = (char *)malloc((size_t)room + 1);
        /* A UNION PRINTS WITH BARS, not as `Union[...]`: PEP 604 made `int |
           str` the spelling, and that is what CPython's repr answers. It is
           an alias like any other underneath. */
        if (is_union) {
            at = 0;
            for (i = 0; i < n; i++) {
                if (i) { out[at++] = ' '; out[at++] = '|'; out[at++] = ' '; }
                memcpy(out + at, O(parts[i])->v.s.p,
                       (size_t)O(parts[i])->v.s.n);
                at += O(parts[i])->v.s.n;
            }
            out[at] = 0;
            free(parts);
            return apy_str_take(out, at);
        }
        at = 0;
        /* `*list[int]`, which is what CPython prints for the unpacked form
           and the one item iterating `list[int]` hands out. */
        if (O(v)->v.ga.unpacked) out[at++] = '*';
        memcpy(out + at, O(head)->v.s.p, (size_t)O(head)->v.s.n);
        at += O(head)->v.s.n;
        out[at++] = '[';
        for (i = 0; i < n; i++) {
            if (i) { out[at++] = ','; out[at++] = ' '; }
            memcpy(out + at, O(parts[i])->v.s.p, (size_t)O(parts[i])->v.s.n);
            at += O(parts[i])->v.s.n;
        }
        out[at++] = ']';
        out[at] = 0;
        free(parts);
        return apy_str_take(out, at);
    }
    case APY_SLICE_K: {
        /* `slice(1, 2, None)` -- always all three, and always the repr of
           each, which is how CPython prints one whether or not the bound was
           written. */
        apy_value a = apy_text(O(v)->v.sl.start, 1);
        apy_value b = apy_text(O(v)->v.sl.stop, 1);
        apy_value c = apy_text(O(v)->v.sl.step, 1);
        int64_t n = O(a)->v.s.n + O(b)->v.s.n + O(c)->v.s.n + 12;
        char *out = (char *)malloc((size_t)n + 1);
        int wrote = snprintf(out, (size_t)n + 1, "slice(%.*s, %.*s, %.*s)",
                             (int)O(a)->v.s.n, O(a)->v.s.p,
                             (int)O(b)->v.s.n, O(b)->v.s.p,
                             (int)O(c)->v.s.n, O(c)->v.s.p);
        return apy_str_take(out, wrote);
    }
    case APY_EXC_K: {
        /* A CLASS'S OWN `__str__`/`__repr__` WINS, exactly as it does for an
           instance and exactly as the interpreter's `_text` already had it:
           an exception is the one kind of object whose text is nearly always
           overridden. Without this `str(AppError(404, "missing"))` printed
           the ARGS -- `(404, 'missing')` -- on both compiled paths, where the
           class wrote `404: missing`. BEFORE BaseException's own rules,
           because an override is written to replace them; the runtime's own
           exceptions have no class and skip it. */
        const char *which = quoted ? "__repr__" : "__str__";
        if (O(v)->v.e.cls) {
            apy_value m = apy_class_find(O(v)->v.e.cls, apy_name(which));
            if (m && O(m)->kind == APY_FUNC_K) {
                apy_value r = apy_call_n(apy_bind(m, v), NULL, 0);
                return r ? apy_text_result(r, which) : r;
            }
        }
        return apy_exc_text(v, quoted);
    }
    case APY_LIST_K:
    case APY_TUPLE_K: return apy_seq_text(v);
    case APY_SET_K:
    case APY_FROZEN_K: return apy_set_text(v);
    case APY_VIEW_K: {
        /* `dict_keys(['a'])` -- the KIND NAME around the list of what the
           view is looking at, which is how CPython prints one. Falling
           through to the default answered the empty string, so a program that
           printed `d.keys()` printed nothing at all. */
        apy_value items = apy_seq_text(apy_view_items(v));
        const char *head = apy_kind_name(v);
        int64_t room = (int64_t)strlen(head) + O(items)->v.s.n + 3;
        char *out = (char *)malloc((size_t)room + 1);
        int wrote = snprintf(out, (size_t)room + 1, "%s(%.*s)", head,
                             (int)O(items)->v.s.n, O(items)->v.s.p);
        return apy_str_take(out, wrote);
    }
    case APY_INST_K: {
        /* A TYPING FORM PRINTS AS `typing.Name`. It is an instance with no
           `__repr__`, so the default `<_SpecialForm object at 0x...>` came
           out -- an address where CPython prints the name a program wrote. */
        if (apy_is_special_form(v)) {
            apy_value nm = apy_dict_get_or(O(v)->v.o.dict, apy_lit("_name"), 0);
            if (nm) {
                char *outf = (char *)malloc((size_t)O(nm)->v.s.n + 8);
                int wrotef = snprintf(outf, (size_t)O(nm)->v.s.n + 8,
                                      "typing.%.*s", (int)O(nm)->v.s.n,
                                      O(nm)->v.s.p);
                return apy_str_take(outf, wrotef);
            }
        }
        /* `str(x)` asks `__str__` and FALLS BACK to `__repr__`; `repr(x)`
           asks only `__repr__`. That asymmetry is Python's and it is load
           bearing: a class defining only `__repr__` prints with it, and one
           defining only `__str__` still shows its default repr in a list. */
        apy_value r = quoted ? 0 : apy_unary_dunder(v, "__str__");
        if (r || apy_error_occurred())
            return r ? apy_text_result(r, "__str__") : r;
        /* STR, BYTES AND BYTEARRAY WRITE A `__str__` OF THEIR OWN, and it
           beats a subclass's `__repr__` the way any inherited method beats a
           fallback: `class S(str)` with only a `__repr__` still prints its
           TEXT under `str()`, because `str.__str__` is what `str()` finds
           first and it answers the value. Every other builtin leaves
           `tp_str` at object's, which reaches `tp_repr` and so the written
           `__repr__` -- which is why `class D(dict)` with a `__repr__` does
           print with it. Without this, `str(S("text"))` was `"'text'"`:
           SIX characters where the program wrote four, so the text no longer
           compared equal to itself. A bytearray reaches here too, since one
           is a BYTES_K with `mut` set. */
        if (!quoted && O(v)->v.o.held
            && (O(O(v)->v.o.held)->kind == APY_STR_K
                || O(O(v)->v.o.held)->kind == APY_BYTES_K))
            /* A FRESH PLAIN str, AND NOT THE ONE INSIDE THE INSTANCE.
               `str()` of an exact str IS that str, so this answered the held
               cell -- which two instances built from one literal share, and
               `str(S("ab")) is str(S("ab"))` was True where CPython says
               False. See `apy_inst_text_result`. */
            return apy_inst_text_result(v, apy_text_of(O(v)->v.o.held, 0));
        r = apy_unary_dunder(v, "__repr__");
        if (r || apy_error_occurred())
            return r ? apy_text_result(r, "__repr__") : r;
        /* A CLASS EXTENDING A BUILTIN SHOWS THE BUILTIN. `class D(dict)`
           with no `__repr__` prints `{'a': 1}` in CPython, and the default
           below would hide the entire contents -- which for a Counter or a
           defaultdict is the whole value. The class name is not added,
           because CPython does not add it either; a subclass that wants its
           name in the repr writes one, as `Counter` and `deque` do. */
        if (O(v)->v.o.held) return apy_repr(O(v)->v.o.held);
        /* The default. CPython prints the ADDRESS, which no two runs agree on
           and which no conformance case can therefore assert -- every case
           that prints a bare instance defines `__repr__`. The address is
           printed anyway rather than omitted, because a program that prints
           one is telling the reader it did not define one. */
        {
            /* THE CLASS'S QUALNAME, for the same reason the class's own
               repr uses it: `<__main__.mk.<locals>.D object at 0x...>`. */
            const char *where = apy_class_module(O(v)->v.o.cls);
            const char *what = APY_CSTR(apy_type_qualname(O(v)->v.o.cls));
            if (where)
                snprintf(buf, sizeof buf, "<%s.%s object at 0x%llx>", where,
                         what, (unsigned long long)v);
            else
                snprintf(buf, sizeof buf, "<%s object at 0x%llx>",
                         what, (unsigned long long)v);
        }
        return apy_str_copy(buf, (int64_t)strlen(buf));
    }
    case APY_TYPE_K:
        /* PRINTING A CLASS IS THE METACLASS'S BUSINESS when it says so.
           `repr(Colour)` is `type(Colour).__repr__(Colour)`, which is how an
           enum prints as `<enum 'Colour'>` -- the fourth of the metaclass
           dunders to need saying so, beside `__iter__`, `__len__` and
           `__contains__`. */
        if (O(v)->v.t.meta) {
            apy_value hook = apy_class_find(O(v)->v.t.meta,
                                            apy_name("__repr__"));
            if (hook)
                return apy_call_n(apy_bind(hook, v), NULL, 0);
        }
        {
            /* THE QUALNAME AND NOT THE NAME: `repr` of a class written
               inside a function is `<class '__main__.mk.<locals>.D'>` in
               CPython, which is the one place the nesting shows. */
            const char *where = apy_class_module(v);
            const char *what = APY_CSTR(apy_type_qualname(v));
            if (where)
                snprintf(buf, sizeof buf, "<class '%s.%s'>", where, what);
            else
                snprintf(buf, sizeof buf, "<class '%s'>", what);
        }
        return apy_str_copy(buf, (int64_t)strlen(buf));
    case APY_FUNC_K:
        /* A BUILTIN TYPE NAME PRINTS AS A CLASS. `print(int)` says
           `<class 'int'>` in Python, and it reaches here as a callable thunk
           -- so the flag, not the kind, decides what it is called. */
        if (O(v)->v.fn.is_type) {
            snprintf(buf, sizeof buf, "<class '%s'>",
                     APY_CSTR(O(v)->v.fn.name));
            return apy_str_copy(buf, (int64_t)strlen(buf));
        }
        /* A DESCRIPTOR NAMES THE TYPE IT CAME OFF and carries no address,
           for the same reason a builtin does: `<method 'append' of 'list'
           objects>`, and `<slot wrapper '__len__' of 'list' objects>` for
           one the type fills a slot with. The owner is the head of the
           qualname -- see `apy_descr_owner`. */
        if (O(v)->v.fn.descr) {
            const char *q = O(v)->v.fn.qualname
                ? APY_CSTR(O(v)->v.fn.qualname) : 0;
            const char *dot = q ? strchr(q, '.') : 0;
            const char *w = APY_CSTR(O(v)->v.fn.name);
            size_t len = strlen(w);
            int slot = len >= 5 && w[0] == '_' && w[1] == '_'
                && w[len - 1] == '_' && w[len - 2] == '_'
                && !apy_descr_written(v);
            if (dot)
                snprintf(buf, sizeof buf, "<%s '%s' of '%.*s' objects>",
                         slot ? "slot wrapper" : "method", w,
                         (int)(dot - q), q);
            else
                snprintf(buf, sizeof buf, "<%s '%s'>",
                         slot ? "slot wrapper" : "method", w);
            return apy_str_copy(buf, (int64_t)strlen(buf));
        }
        /* A BUILTIN REACHED AS A VALUE HAS NO ADDRESS IN ITS REPR.
           `repr(len)` is `<built-in function len>` -- CPython writes no
           address for one, because there is only ever the one. The flag is
           the same one `type()` reads. */
        if (O(v)->v.fn.builtin && !O(v)->v.fn.bound) {
            snprintf(buf, sizeof buf, "<built-in function %s>",
                     APY_CSTR(O(v)->v.fn.name));
            return apy_str_copy(buf, (int64_t)strlen(buf));
        }
        /* A BOUND METHOD SHOWS WHAT IT IS BOUND TO, and a BUILTIN one is
           called a `built-in method`: `"a".upper` is `<built-in method
           upper of str object at 0x...>` where a compiled method is
           `<bound method C.m of <C object at 0x...>>`. Both name the
           receiver, which is most of what makes the text useful. */
        if (O(v)->v.fn.bound) {
            apy_value held = O(v)->v.fn.bound;
            if (O(v)->v.fn.native || O(v)->v.fn.builtin) {
                /* A BOUND SLOT SAYS `method-wrapper` AND QUOTES THE NAME:
                   `[].__len__` is `<method-wrapper '__len__' of list object
                   at 0x...>` where `[].append` is `<built-in method append
                   of list object at 0x...>`. Which of the two it is, is the
                   question `type()` already answers -- so it is asked here
                   rather than restated. */
                if (strcmp(apy_kind_name(v), "method-wrapper") == 0)
                    snprintf(buf, sizeof buf,
                             "<method-wrapper '%s' of %s object at 0x%llx>",
                             APY_CSTR(O(v)->v.fn.name), apy_kind_name(held),
                             (unsigned long long)held);
                else
                    snprintf(buf, sizeof buf,
                             "<built-in method %s of %s object at 0x%llx>",
                             APY_CSTR(O(v)->v.fn.name), apy_kind_name(held),
                             (unsigned long long)held);
                return apy_str_copy(buf, (int64_t)strlen(buf));
            }
            {
                /* THE OWNER'S NAME BEFORE THE METHOD'S, which is CPython's
                   `C.m` -- the qualified name a reader needs to find it. */
                apy_value shown = apy_text_of(held, 1);
                char *big;
                if (!shown) return 0;
                big = (char *)malloc((size_t)O(shown)->v.s.n + 128);
                if (!big) { fputs("uasm: out of memory\n", stderr); exit(1); }
                sprintf(big, "<bound method %s.%s of %s>",
                        apy_kind_name(held), APY_CSTR(O(v)->v.fn.name),
                        APY_CSTR(shown));
                return apy_str_take(big, (int64_t)strlen(big));
            }
        }
        /* THE QUALIFIED NAME, which is what CPython writes: `repr(C.m)` is
           `<function C.m at 0x...>` and not `<function m at 0x...>`. */
        snprintf(buf, sizeof buf, "<function %s at 0x%llx>",
                 APY_CSTR(O(v)->v.fn.qualname ? O(v)->v.fn.qualname
                                              : O(v)->v.fn.name),
                 (unsigned long long)v);
        return apy_str_copy(buf, (int64_t)strlen(buf));
    /* A STRING IS THE ONLY KIND THE QUOTING BELOW IS FOR. */
    case APY_STR_K: break;
    /* A MEMORYVIEW IS `<memory at 0x...>` -- `memory`, not `memoryview`,
       which makes it the one kind whose repr does not use its type name. */
    case APY_MVIEW_K: {
        char buf[64];
        snprintf(buf, sizeof buf, "<memory at 0x%llx>",
                 (unsigned long long)v);
        return apy_str_copy(buf, (int64_t)strlen(buf));
    }
    case APY_GEN_K: {
        /* A GENERATOR NAMES THE `def` IT CAME FROM: `<generator object gen
           at 0x...>`, and a generator expression the qualified name of the
           scope it was written in -- `<generator object f.<locals>.<genexpr>
           at 0x...>`. A coroutine and an async generator take the same shape
           under their own kind names.

           THE NAME IS THE STEP FUNCTION'S, because the generator cell is a
           frame and the frame's code is the step. The four built by the async
           machinery have no step and fall back to the bare object shape. */
        apy_value step = O(v)->v.g.step;
        apy_value who = (step && O(step)->kind == APY_FUNC_K)
            ? (O(step)->v.fn.qualname ? O(step)->v.fn.qualname
                                      : O(step)->v.fn.name)
            : 0;
        char named[256];
        if (who)
            snprintf(named, sizeof named, "<%s object %s at 0x%llx>",
                     apy_kind_name(v), APY_CSTR(who),
                     (unsigned long long)v);
        else
            snprintf(named, sizeof named, "<%s object at 0x%llx>",
                     apy_kind_name(v), (unsigned long long)v);
        return apy_str_copy(named, (int64_t)strlen(named));
    }
    /* A CLASSMETHOD OR STATICMETHOD SHOWS WHAT IT WRAPS, which is what
       CPython writes: `<staticmethod(<function f at 0x...>)>`. A `property`
       does not, and takes the ordinary object shape below. */
    case APY_PROP_K:
        if (O(v)->v.p.kind != APY_PROP_PROPERTY) {
            apy_value inner = O(v)->v.p.get
                ? apy_text_of(O(v)->v.p.get, 1) : apy_lit("None");
            char *buf2;
            int64_t n2;
            const char *word = O(v)->v.p.kind == APY_PROP_CLASSMETHOD
                ? "classmethod" : "staticmethod";
            if (!inner) return 0;
            n2 = (int64_t)strlen(word) + O(inner)->v.s.n + 4;
            buf2 = (char *)malloc((size_t)n2 + 1);
            if (!buf2) { fputs("uasm: out of memory\n", stderr); exit(1); }
            snprintf(buf2, (size_t)n2 + 1, "<%s(%s)>", word,
                     APY_CSTR(inner));
            return apy_str_take(buf2, (int64_t)strlen(buf2));
        }
        /* FALLS THROUGH to the object shape, which is what a property has. */
        /* fall through */
    default: {
        /* EVERY OTHER KIND IS AN OBJECT WITH NO REPR OF ITS OWN, and the
           quoting below reads the value as a STR: `v.s.p` on an iterator
           cell is whatever the union's other half left there, so
           `repr(map(len, xs))` WALKED A WILD POINTER AND DIED while
           `repr(property(f))` answered the empty string. */
        char buf[96];
        snprintf(buf, sizeof buf, "<%s object at 0x%llx>",
                 apy_kind_name(v), (unsigned long long)v);
        return apy_str_copy(buf, (int64_t)strlen(buf));
    }
    }
    if (!quoted) return v;
    {
        /* Python prefers single quotes and switches to double only when the
           text contains a single quote and no double. */
        const char *p = O(v)->v.s.p;
        int64_t n = O(v)->v.s.n, i, out = 0;
        int has_sq = 0, has_dq = 0;
        char q, *buf2;
        for (i = 0; i < n; i++) {
            if (p[i] == '\'') has_sq = 1;
            if (p[i] == '"') has_dq = 1;
        }
        q = (has_sq && !has_dq) ? '"' : '\'';
        buf2 = (char *)malloc((size_t)n * 4 + 3);
        buf2[out++] = q;
        /* BY CODE POINT ABOVE 0x7F, because whether a character is
           printable is a question about the CHARACTER: U+00A0 and U+2003
           are spaces Python escapes, and asked a byte at a time neither one
           is anything at all. Below 0x80 a byte IS a character and the walk
           this replaces still stands.

           THE THREE WIDTHS ARE PYTHON'S: `\\xNN` under 0x100, `\\uNNNN`
           under 0x10000, `\\UNNNNNNNN` above. An unprintable character
           written out in its own bytes would come back from `eval`
           unchanged and LOOK like the space it is not. */
        for (i = 0; i < n; ) {
            unsigned char c = (unsigned char)p[i];
            if (c < 0x80) {
                if (c == (unsigned char)q || c == '\\') {
                    buf2[out++] = '\\'; buf2[out++] = (char)c;
                } else if (c == '\n') { buf2[out++] = '\\'; buf2[out++] = 'n'; }
                else if (c == '\r') { buf2[out++] = '\\'; buf2[out++] = 'r'; }
                else if (c == '\t') { buf2[out++] = '\\'; buf2[out++] = 't'; }
                else if (c < 0x20 || c == 0x7f) {
                    out += (int64_t)sprintf(buf2 + out, "\\x%02x", c);
                } else buf2[out++] = (char)c;
                i++;
                continue;
            }
            {
                uint32_t cp;
                int64_t used = apy_utf8_step((const unsigned char *)p, n,
                                             i, &cp);
                /* A BYTE THAT IS NOT VALID UTF-8 is escaped as itself rather
                   than replaced: `repr` is what a person reads to find out
                   what is ACTUALLY in a string, so a lie about its bytes is
                   the one thing it must not tell. */
                if (!used) {
                    out += (int64_t)sprintf(buf2 + out, "\\x%02x", c);
                    i++;
                } else if (apy_cp_printable(cp)) {
                    int64_t k;
                    for (k = 0; k < used; k++) buf2[out++] = p[i + k];
                    i += used;
                } else if (cp < 0x100) {
                    out += (int64_t)sprintf(buf2 + out, "\\x%02x",
                                            (unsigned)cp);
                    i += used;
                } else if (cp < 0x10000) {
                    out += (int64_t)sprintf(buf2 + out, "\\u%04x",
                                            (unsigned)cp);
                    i += used;
                } else {
                    out += (int64_t)sprintf(buf2 + out, "\\U%08x",
                                            (unsigned)cp);
                    i += used;
                }
            }
        }
        buf2[out++] = q;
        buf2[out] = '\0';
        return apy_str_take(buf2, out);
    }
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_text(apy_value v, int quoted) {
    return apy_text_of(v, (int64_t)quoted);
}

APY_API apy_value apy_repr(apy_value v) { return apy_text(v, 1); }

/* `del d[k]` and `del xs[i]`.

   Two containers, two failure modes, and CPython's own messages for each: a
   missing dict key is a KeyError naming the key's repr, and an out-of-range
   list index is "list assignment index out of range" -- the ASSIGNMENT text,
   because deleting is a store-shaped operation and CPython says so.

   A tuple is refused: immutability is the whole distinction from a list, and
   letting a `del` through would erase it. */
/* The ASCENDING run of positions a slice deletes: first, stride and count.

   A NEGATIVE STEP WALKS THE SAME SET BACKWARDS, and a delete is about the
   SET and not the order -- `del xs[::-2]` and `del xs[::2]` take the same
   three positions out of six, starting from opposite ends. Turning it round
   here is what lets the compaction that follows only ever move forwards.

   `first` IS PAST THE END WHEN NOTHING MATCHES, so the copy loop's position
   test is false at every step without a count test of its own. */
APY_API int64_t apy_del_run(apy_value key, int64_t n, apy_value outv) {
    /* AN `apy_value` AND NOT AN `int64_t *`, because the ported half takes a
       `ptr` and the two halves are one translation unit: the declarations
       have to agree before the bodies can. */
    int64_t *out = (int64_t *)(uintptr_t)outv;
    apy_value bounds = apy_slice_indices(key, apy_from_int(n));
    int64_t start, stop, step, count = 0;
    if (!bounds) return 0;
    start = O(O(bounds)->v.q.items[0])->v.i;
    stop = O(O(bounds)->v.q.items[1])->v.i;
    step = O(O(bounds)->v.q.items[2])->v.i;
    if (step > 0) {
        if (stop > start) count = (stop - start + step - 1) / step;
    } else {
        if (stop < start) count = (stop - start + step + 1) / step;
    }
    out[0] = n;
    out[1] = 1;
    out[2] = count;
    if (count > 0) {
        out[0] = step > 0 ? start : start + (count - 1) * step;
        out[1] = step > 0 ? step : -step;
    }
    return 1;
}

/* `del ba[i]` and `del ba[a:b:c]`, on a bytearray.

   THE SAME TWO SHAPES AS A LIST over a different buffer, and the messages
   name `bytearray` rather than `list` because CPython's do.

   THE BUFFER KEEPS ITS TERMINATOR. Nothing here reads past `n`, but the cell
   is shared with str, whose bytes are handed to C as a string -- so shrinking
   writes the NUL rather than leaving the old byte behind. */
APY_API apy_value apy_del_bytes(apy_value seq, apy_value key) {
    char *p = (char *)O(seq)->v.s.p;
    int64_t n = O(seq)->v.s.n, i;
    if (key && O(key)->kind == APY_SLICE_K) {
        int64_t run[3], from, to = 0, taken = 0, at;
        if (!apy_del_run(key, n, (apy_value)(uintptr_t)run)) return 0;
        at = run[0];
        for (from = 0; from < n; from++) {
            if (taken < run[2] && from == at) {
                taken++;
                at = run[0] + taken * run[1];
                continue;
            }
            p[to++] = p[from];
        }
        O(seq)->v.s.n = to;
        p[to] = 0;
        return apy_none();
    }
    if (!apy_is_int_like(key))
        return apy_fail2("TypeError", "bytearray indices must be integers "
                                      "or slices, not %s%s",
                         apy_kind_name(key), "");
    if (!apy_index_arg(key, &i, APY_IDX_SUB)) return 0;
    if (i < 0) i += n;
    if (i < 0 || i >= n)
        return apy_fail("IndexError", "bytearray index out of range");
    for (; i + 1 < n; i++) p[i] = p[i + 1];
    O(seq)->v.s.n = n - 1;
    p[n - 1] = 0;
    return apy_none();
}

APY_API apy_value apy_delitem(apy_value seq, apy_value key) {
    int64_t i;
    if (O(seq)->kind == APY_INST_K) {
        /* `del obj[k]` IS `obj.__delitem__(k)`. Never dispatched before, so
           a class that wrote one had it ignored and the delete was reported
           as unsupported -- a wrong answer about the class's own method. */
        apy_value r = apy_method1(seq, "__delitem__", key);
        if (r || apy_error_occurred()) return r;
        /* A CLASS THAT EXTENDS A BUILTIN deletes from the one it carries. */
        if (apy_inst_held(seq)) return apy_delitem(apy_inst_held(seq), key);
    }
    if (O(seq)->kind == APY_DICT_K) {
        /* A mappingproxy IS READ-ONLY TO A PROGRAM -- and CPython words
           this one differently from the assignment it refuses beside it. */
        const char *bad;
        if (O(seq)->v.d.ro)
            return apy_fail2("TypeError",
                             "'%s' object does not support item deletion%s",
                             apy_kind_name(seq), "");
        bad = apy_unhashable(key);
        if (bad) return apy_fail2("TypeError", "unhashable type: '%s'%s",
                                  bad, "");
        i = apy_dict_find(seq, key);
        if (i < 0) {
            apy_value shown = apy_repr(key);
            return apy_fail2("KeyError", "%s%s", APY_CSTR(shown), "");
        }
        /* Shift the survivors down, preserving INSERTION ORDER -- which is
           part of the language since 3.7, so swapping the last entry into the
           hole would be a wrong answer rather than a faster one. */
        for (; i + 1 < O(seq)->v.d.n; i++) {
            O(seq)->v.d.keys[i] = O(seq)->v.d.keys[i + 1];
            O(seq)->v.d.vals[i] = O(seq)->v.d.vals[i + 1];
        }
        O(seq)->v.d.n--;
        return apy_none();
    }
    /* A BYTEARRAY DELETES TOO, and its buffer is BYTES rather than values --
       which is the whole reason it is a case of its own rather than a wider
       test below. `mut` is what separates it from bytes, which does not. */
    if (O(seq)->kind == APY_BYTES_K && O(seq)->v.s.mut)
        return apy_del_bytes(seq, key);
    /* A VIEW REFUSES IN ITS OWN WORDS. Deleting from one is not "this kind
       has no such operation" -- a memoryview HAS `__delitem__` and it always
       refuses, because a window onto a buffer cannot make the buffer
       shorter. CPython says so and a program may print it. */
    if (O(seq)->kind == APY_MVIEW_K)
        return apy_fail("TypeError", "cannot delete memory");
    if (O(seq)->kind != APY_LIST_K)
        return apy_fail2("TypeError", "'%s' object doesn't support item deletion%s",
                         apy_kind_name(seq), "");
    /* `del xs[1:3]` AND `del xs[::2]` REMOVE A SET OF POSITIONS. Falling
       through to the index path asked `apy_index_arg` for an integer, got
       the slice, and reported an IndexError about a subscript the program
       never wrote. ONE PASS THAT COMPACTS is what makes the strided case no
       harder than the contiguous one. */
    if (key && O(key)->kind == APY_SLICE_K) {
        int64_t run[3], from, to = 0, taken = 0, at, n = O(seq)->v.q.n;
        if (!apy_del_run(key, n, (apy_value)(uintptr_t)run)) return 0;
        at = run[0];
        for (from = 0; from < n; from++) {
            if (taken < run[2] && from == at) {
                taken++;
                at = run[0] + taken * run[1];
                continue;
            }
            O(seq)->v.q.items[to++] = O(seq)->v.q.items[from];
        }
        O(seq)->v.q.n = to;
        return apy_none();
    }
    /* THE SUBSCRIPT'S OWN WORDING. `del xs[1.0]` is a complaint about the
       subscript, and CPython words it as one: `list indices must be integers
       or slices, not float`, and not the index conversion's `'float' object
       cannot be interpreted as an integer`. AN INSTANCE IS LEFT ALONE, since
       one with `__index__` is a valid subscript and the conversion asks. */
    if (!apy_is_int_like(key) && O(key)->kind != APY_INST_K)
        return apy_fail2("TypeError",
                         "list indices must be integers or slices, not %s%s",
                         apy_kind_name(key), "");
    if (!apy_index_arg(key, &i, APY_IDX_SUB)) return 0;
    if (i < 0) i += O(seq)->v.q.n;
    if (i < 0 || i >= O(seq)->v.q.n)
        return apy_fail("IndexError", "list assignment index out of range");
    for (; i + 1 < O(seq)->v.q.n; i++)
        O(seq)->v.q.items[i] = O(seq)->v.q.items[i + 1];
    O(seq)->v.q.n--;
    return apy_none();
}


"""
