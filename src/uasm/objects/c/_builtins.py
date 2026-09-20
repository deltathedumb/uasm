"""The object runtime, in C: integer methods, and the builtins over sequences.

ONE PART OF ONE TRANSLATION UNIT. `c/__init__.py` concatenates
these in order and the result is the file it always was, so a
definition here may rely on anything in an earlier part and
nothing in a later one. Sections, in order:
  * integer methods and the base builtins
  * builtins over sequences
"""

C = r"""/* --- integer methods and the base builtins ------------------------------ */
/* `pow(a, b, m)`, which is not `a ** b % m` and cannot be: `pow(2, 1000,
   1000003)` is instant and `2 ** 1000 % 1000003` builds a 302-digit number
   first. Reducing at every step is the whole point of the three-argument
   form, and it is the reason it exists in the language. */
APY_API apy_value apy_pow3(apy_value a, apy_value b, apy_value m);

APY_API apy_value apy_pow3(apy_value a, apy_value b, apy_value m) {
    apy_value r, base;
    int64_t n;
    if (!apy_is_int_like(a) || !apy_is_int_like(b) || !apy_is_int_like(m))
        return apy_fail("TypeError",
                        "pow() 3rd argument not allowed unless all arguments "
                        "are integers");
    if (!apy_is_big(m) && O(m)->v.i == 0)
        return apy_fail("ValueError", "pow() 3rd argument cannot be 0");
    /* A NEGATIVE exponent is the MODULAR INVERSE raised to its magnitude,
       which CPython grew in 3.8. Done here with the extended Euclidean
       algorithm, written in terms of the public operators rather than in
       limbs: every one of them already promotes and already floors the way
       Python does, so there is nothing about big integers left to get wrong
       in it, and the whole algorithm is the six lines it is on paper. */
    if ((apy_is_big(b) && O(b)->v.big.neg) || (!apy_is_big(b) && O(b)->v.i < 0)) {
        apy_value tt = apy_from_int(0), newt = apy_from_int(1);
        apy_value rr = m, newr = apy_mod(a, m);
        if (!newr) return 0;
        while (!apy_is_big(newr) && O(newr)->v.i != 0) {
            apy_value q = apy_floordiv(rr, newr), st, sr;
            if (!q) return 0;
            st = apy_sub(tt, apy_mul(q, newt));
            sr = apy_sub(rr, apy_mul(q, newr));
            if (!st || !sr) return 0;
            tt = newt; newt = st;
            rr = newr; newr = sr;
        }
        /* A base with a common factor with the modulus has NO inverse, and
           `gcd != 1` is how that shows: 2 has none mod 4, and neither does 0
           mod anything. CPython reports rather than answering 0. */
        if (apy_is_big(rr) || (O(rr)->v.i != 1 && O(rr)->v.i != -1))
            return apy_fail("ValueError",
                            "base is not invertible for the given modulus");
        if (!apy_is_big(rr) && O(rr)->v.i == -1) tt = apy_neg(tt);
        tt = apy_mod(tt, m);
        if (!tt) return 0;
        /* And now the positive-exponent case, on the inverse. */
        return apy_pow3(tt, apy_neg(b), m);
    }
    if (apy_is_big(b)) return apy_big_too_large();
    n = O(b)->v.i;
    r = apy_mod(apy_from_int(1), m);
    if (!r) return 0;
    base = apy_mod(a, m);
    if (!base) return 0;
    while (n) {
        if (n & 1) {
            r = apy_mul(r, base);
            if (r) r = apy_mod(r, m);
            if (!r) return 0;
        }
        n >>= 1;
        if (n) {
            base = apy_mul(base, base);
            if (base) base = apy_mod(base, m);
            if (!base) return 0;
        }
    }
    return r;
}

/* `divmod(a, b)`. NOT two calls: the quotient and the remainder come out of
   one division, and for two bigs that division is the expensive part -- doing
   it twice is the whole cost twice. It is also the only way to be sure the
   two answers are consistent, which is the property `divmod(a, b)[0] * b +
   divmod(a, b)[1] == a` is asserting. Floats go through the existing `//` and
   `%`, which already carry CPython's transcribed `float_divmod`. */
APY_API apy_value apy_divmod(apy_value a, apy_value b) {
    apy_value out, q, r;
    if (!apy_is_num(a) || !apy_is_num(b))
        return apy_binop_error("divmod()", a, b);
    if (apy_is_int_like(a) && apy_is_int_like(b) && apy_either_big(a, b)) {
        if (!apy_is_big(b) && O(b)->v.i == 0)
            return apy_fail("ZeroDivisionError", APY_DIV0);
        apy_big_floordivmod(apy_as_big(a), apy_as_big(b), &q, &r);
    } else {
        q = apy_floordiv(a, b);
        if (!q) return 0;
        r = apy_mod(a, b);
        if (!r) return 0;
    }
    if (!q || !r) return 0;
    out = apy_seq_new(APY_TUPLE_K, 2);
    apy_q_append(out, q);
    apy_q_append(out, r);
    return out;
}

/* `n.bit_length()` -- the bits needed for the MAGNITUDE, so `(-255)` and
   `255` both answer 8 and `0` answers 0. */
APY_API apy_value apy_bit_length(apy_value v) {
    if (!apy_is_int_like(v))
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'bit_length'%s",
                         apy_kind_name(v), "");
    if (apy_is_big(v)) return apy_from_int(apy_mag_bits(O(v)));
    {
        uint64_t m = apy_abs64(O(v)->v.i);
        int64_t n = 0;
        while (m) { n++; m >>= 1; }
        return apy_from_int(n);
    }
}

/* `n.bit_count()` -- the number of ONE bits in the magnitude, again ignoring
   the sign, which is what CPython counts. */
APY_API apy_value apy_bit_count(apy_value v) {
    if (!apy_is_int_like(v))
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'bit_count'%s",
                         apy_kind_name(v), "");
    if (apy_is_big(v)) return apy_from_int(apy_big_popcount((apy_value)O(v)));
    {
        uint64_t m = apy_abs64(O(v)->v.i);
        int64_t n = 0;
        while (m) { n += (int64_t)(m & 1); m >>= 1; }
        return apy_from_int(n);
    }
}

/* `bin`, `oct` and `hex`. The prefix goes AFTER the sign -- `bin(-10)` is
   `-0b1010` and not `0b-1010` -- which is the only thing about these that is
   easy to get backwards. */
APY_API apy_value apy_base_text_of(apy_value v, int64_t bits_per,
                                  apy_value prefixv, apy_value fnv) {
    const char *prefix = (const char *)prefixv;
    const char *fn = (const char *)fnv;
    /* `hex(obj)` goes through `__index__` -- PEP 357 names these three as the
       operations it exists for, alongside subscripting. */
    if (O(v)->kind == APY_INST_K) {
        apy_value got = apy_unary_dunder(v, "__index__");
        if (apy_error_occurred()) return 0;
        if (got && apy_is_int_like(got)) v = got;
    }
    if (!apy_is_int_like(v))
        return apy_fail2("TypeError",
                         "%s() argument can't be interpreted as an integer%s",
                         fn, "");
    if (apy_is_big(v)) return apy_big_base_text(O(v), bits_per, prefix);
    {
        uint64_t m = apy_abs64(O(v)->v.i);
        char buf[80];
        int out = 0, i, start;
        if (O(v)->v.i < 0) buf[out++] = '-';
        buf[out++] = prefix[0];
        buf[out++] = prefix[1];
        start = out;
        if (m == 0) buf[out++] = '0';
        while (m) {
            buf[out++] = "0123456789abcdef"[m & (((uint64_t)1 << bits_per) - 1)];
            m >>= bits_per;
        }
        for (i = 0; i < (out - start) / 2; i++) {
            char c = buf[start + i];
            buf[start + i] = buf[out - 1 - i];
            buf[out - 1 - i] = c;
        }
        buf[out] = '\0';
        return apy_str_copy(buf, out);
    }
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_base_text(apy_value v, int bits_per,
                               const char *prefix, const char *fn) {
    return apy_base_text_of(v, (int64_t)bits_per,
                            (apy_value)(uintptr_t)prefix,
                            (apy_value)(uintptr_t)fn);
}

APY_API apy_value apy_bin(apy_value v) { return apy_base_text(v, 1, "0b", "bin"); }
APY_API apy_value apy_oct(apy_value v) { return apy_base_text(v, 3, "0o", "oct"); }
APY_API apy_value apy_hex(apy_value v) { return apy_base_text(v, 4, "0x", "hex"); }

/* `int(s, base)` for ANY base from 2 to 36, and base 0 -- which means "read
   the prefix", so `int('0x1f', 0)` is 31 and `int('17', 0)` is 17.

   Multiply-and-add rather than the shift the power-of-two bases allow: base
   36 has no shift, and having one loop means base 3 cannot be the one nobody
   tested. The multiply goes through `apy_mul` so the promotion to a big
   integer is the one the operators already know how to do. */
APY_API apy_value apy_to_int_base(apy_value v, apy_value base) {
    int64_t b, i, lo, hi;
    int neg = 0;
    apy_value acc;
    /* A str SUBCLASS IS A str HERE TOO -- see `apy_text_like`. */
    v = apy_text_like(v);
    if (O(v)->kind != APY_STR_K)
        return apy_fail2("TypeError",
                         "int() can't convert non-string with explicit base%s%s",
                         "", "");
    if (!apy_int_arg(base, &b)) return 0;
    if (b != 0 && (b < 2 || b > 36))
        return apy_fail("ValueError", "int() base must be >= 2 and <= 36, or 0");
    lo = 0;
    hi = O(v)->v.s.n;
    while (lo < hi && apy_is_space(O(v)->v.s.p[lo])) lo++;
    while (hi > lo && apy_is_space(O(v)->v.s.p[hi - 1])) hi--;
    if (lo < hi && (O(v)->v.s.p[lo] == '+' || O(v)->v.s.p[lo] == '-')) {
        neg = O(v)->v.s.p[lo] == '-';
        lo++;
    }
    /* The `0x`/`0o`/`0b` prefix is OPTIONAL when the base says the same
       thing, and DECIDES the base when the base is 0 -- which is the whole of
       what base 0 means. */
    if (hi - lo >= 2 && O(v)->v.s.p[lo] == '0') {
        char c = O(v)->v.s.p[lo + 1];
        int said = (c == 'x' || c == 'X') ? 16
                 : (c == 'o' || c == 'O') ? 8
                 : (c == 'b' || c == 'B') ? 2 : 0;
        if (said && (b == 0 || b == said)) { b = said; lo += 2; }
    }
    if (b == 0) b = 10;
    if (lo >= hi) {
        char msg[128];
        snprintf(msg, sizeof msg, "invalid literal for int() with base %lld: ",
                 (long long)b);
        return apy_conv_error(msg, v);
    }
    acc = apy_from_int(0);
    for (i = lo; i < hi; i++) {
        unsigned char c = (unsigned char)O(v)->v.s.p[i];
        int d;
        if (c == '_') continue;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'z') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'Z') d = c - 'A' + 10;
        else d = 99;
        if (d >= b) {
            char msg[128];
            snprintf(msg, sizeof msg,
                     "invalid literal for int() with base %lld: ", (long long)b);
            return apy_conv_error(msg, v);
        }
        acc = apy_mul(acc, apy_from_int(b));
        if (acc) acc = apy_add(acc, apy_from_int(d));
        if (!acc) return 0;
    }
    return neg ? apy_neg(acc) : acc;
}

/* --- builtins over sequences ------------------------------------------- */
/* `sorted`, `min`, `max`, `sum`, `reversed`, `enumerate`, `zip` all produce a
   LIST here, not an iterator. Python's return an iterator for the last three,
   and the difference is observable -- `type(enumerate(x)).__name__` is
   'enumerate' and a second pass over one yields nothing. That is a stated
   divergence, not an oversight: a real iterator needs a resumable frame, and
   every other use of these is a `for` loop or a `list(...)`, which a list
   satisfies exactly. */

/* `PyObject_LengthHint` -- what `list(x)`, `sorted(x)` and the other
   spellings below ask BEFORE they drain, so they can size the result in one
   allocation.

   OBSERVABLE, which is why it is here at all: the answer is thrown away,
   but a class that writes `__len__` or `__length_hint__` sees the call, and
   a program that prints from one prints a line. Not asking made `list(it)`
   differ from CPython in what it RAN, not in what it produced.

   TWO VERBS, IN ORDER, AND ONLY ONE OF THEM. `__len__` first; only when
   there is none does `__length_hint__` get asked. A class with both sees
   exactly one call, and it is `__len__`.

   A TypeError FROM `__len__` IS SWALLOWED and `__length_hint__` tried in
   its place -- that is how CPython lets a type say "I have no length" --
   and any OTHER exception is the program's own and propagates. Swallowing
   all of them would hide a bug in the very method being asked.

   WHICH SPELLINGS ASK IS MEASURED, not reasoned about: the first attempt at
   this put the call in the shared iteration funnel and made every consumer
   run a line of the program CPython never runs. Against 3.14 the askers are
   `list(x)`, `bytes(x)`, `sorted(x)`, `xs.extend(x)` on a list and on a
   bytearray, `xs += x` on a list, and a starred display -- `[*x]` AND
   `(*x,)`, the tuple one because CPython builds a tuple display by
   extending a list and converting it. `tuple(x)`, `set(x)`, `frozenset(x)`,
   `bytearray(x)`, `{*x}`, `s.update(x)`, a comprehension, `f(*x)`,
   unpacking, `in`, `sum`, `max`, `join` and a plain `for` do NOT ask, which
   is why this is a function each of them names rather than a line in the
   funnel they share.

   ANSWERS 0 ONLY TO PROPAGATE. The hint itself is advisory and no caller
   reads it; what a caller must not do is carry on with an error pending. */
APY_API apy_value apy_length_hint(apy_value src) {
    apy_value got;
    if (O(src)->kind != APY_INST_K) return apy_none();
    if (apy_class_find(O(src)->v.o.cls, apy_name("__len__"))) {
        got = apy_unary_dunder(src, "__len__");
        if (got) return apy_none();
        if (!apy_error_matches(apy_lit("TypeError"))) return 0;
        apy_error_clear();
    }
    if (apy_class_find(O(src)->v.o.cls, apy_name("__length_hint__"))) {
        got = apy_unary_dunder(src, "__length_hint__");
        if (!got) return 0;
    }
    return apy_none();
}

APY_API apy_value apy_sorted(apy_value seq) {
    int64_t n, i, j;
    apy_value out;
    /* ASKED FIRST, ON THE ARGUMENT, before anything unwraps or drains it:
       CPython sizes the result from the hint and a class that writes one
       sees the call. See `apy_length_hint`. */
    if (!apy_length_hint(seq)) return 0;
    /* A USER ITERATOR IS DRAINED BEFORE THE WALK. Everything below is by
       index, and a class whose `__iter__` answers an object with `__next__`
       has no index -- `apy_iterable` turns one into a list exactly as it does
       a generator. Without this the walk asked `apy_raw_len` straight out and
       was told the argument was not iterable, about something a `for` loop
       over the very same object walked happily. */
    seq = apy_iterable(seq);
    if (!seq) return 0;
    n = apy_raw_len(seq);
    if (apy_error_occurred()) return 0;
    out = apy_seq_new(APY_LIST_K, n + 1);
    for (i = 0; i < n; i++) apy_seq_push(out, apy_key_at(seq, i));
    /* Insertion sort, which is STABLE -- equal elements keep their input
       order, and Python guarantees that. A quicksort here would be faster and
       would quietly reorder them. */
    for (i = 1; i < n; i++) {
        apy_value key = O(out)->v.q.items[i];
        j = i - 1;
        while (j >= 0) {
            int c = apy_order_rich(key, O(out)->v.q.items[j]);
            if (c == 2) {
                apy_order_error(0, key, O(out)->v.q.items[j]);
                return 0;
            }
            if (c >= 0) break;
            O(out)->v.q.items[j + 1] = O(out)->v.q.items[j];
            j--;
        }
        O(out)->v.q.items[j + 1] = key;
    }
    return out;
}

/* `sorted(xs, key=f)` and `sorted(xs, reverse=True)`.

   The key is computed ONCE PER ELEMENT, before the sort, not inside the
   comparison. That is what CPython does and it is observable: a key function
   with a side effect runs exactly n times, and one that raises does so before
   any comparison happens.

   Reversal is applied by flipping the comparison rather than by reversing the
   result, because reversing afterwards also reverses the order of EQUAL
   elements -- and `sorted(reverse=True)` is still stable. That distinction is
   invisible until two items compare equal.

   `keyfn` of 0 means no key, so this is also the plain sort; keeping one
   implementation means the stability argument above only has to be right
   once. */
static apy_value apy_sort_with(apy_value seq, apy_value keyfn, int reverse) {
    int64_t n, i, j;
    apy_value out, keys;
    /* Drained first, so a user iterator can be walked -- see `apy_sorted`. */
    seq = apy_iterable(seq);
    if (!seq) return 0;
    n = apy_raw_len(seq);
    if (apy_error_occurred()) return 0;
    out = apy_seq_new(APY_LIST_K, n + 1);
    keys = apy_seq_new(APY_LIST_K, n + 1);
    for (i = 0; i < n; i++) {
        apy_value item = apy_key_at(seq, i);
        if (!item) return 0;
        apy_seq_push(out, item);
        if (keyfn) {
            apy_value k = apy_call_n(keyfn, &item, 1);
            if (!k) return 0;
            apy_seq_push(keys, k);
        } else {
            apy_seq_push(keys, item);
        }
    }
    for (i = 1; i < n; i++) {
        apy_value item = O(out)->v.q.items[i];
        apy_value k = O(keys)->v.q.items[i];
        j = i - 1;
        while (j >= 0) {
            int c = apy_order_rich(k, O(keys)->v.q.items[j]);
            if (c == 2) {
                /* REVERSED, THE PAIR IS NAMED THE OTHER WAY ROUND. CPython
                   sorts in reverse by reversing the list, sorting it
                   ascending and reversing it back -- so the comparison a
                   program sees refused is still an ascending one, over the
                   two elements in the opposite order from the one this loop
                   (which flips the test instead) reached them in. */
                if (reverse) apy_order_error(0, O(keys)->v.q.items[j], k);
                else apy_order_error(0, k, O(keys)->v.q.items[j]);
                return 0;
            }
            if (reverse ? c <= 0 : c >= 0) break;
            O(out)->v.q.items[j + 1] = O(out)->v.q.items[j];
            O(keys)->v.q.items[j + 1] = O(keys)->v.q.items[j];
            j--;
        }
        O(out)->v.q.items[j + 1] = item;
        O(keys)->v.q.items[j + 1] = k;
    }
    return out;
}

/* `iter(x)`. An iterator OVER an iterator is itself, which is what makes
   `for v in it` work on a partly-consumed one and what `iter(iter(x)) is
   iter(x)` asserts. */
/* `dict(pairs)` -- a sequence of two-element sequences. Not `dict(**kw)` and
   not `dict(mapping)`; both are shapes nothing here can produce yet. */
/* `{*xs, y}` -- every element of a sequence, ADDED to a set. `apy_extend`
   appends and would let a duplicate through; the distinction is the whole
   difference between a set display and a list one. */
APY_API apy_value apy_set_update(apy_value target, apy_value src) {
    int64_t i, n;
    /* Drained first, so a user iterator can be walked -- see `apy_sorted`. */
    src = apy_iterable(src);
    if (!src) return 0;
    n = apy_raw_len(src);
    if (apy_error_occurred()) return 0;
    for (i = 0; i < n; i++) {
        apy_value item = apy_key_at(src, i);
        if (!item || !apy_set_push(target, item)) return 0;
    }
    return apy_none();
}

/* The VARIADIC builtins, reached through a value rather than a call site:
   `map(print, xs)`, `key=dict`. Each takes the argument tuple a `*rest`
   thunk was handed, because a value-form has no compile-time argument count
   -- which is exactly what kept these three out of `_VALUE_BUILTINS` until
   the thunk learned to be variadic.

   The bodies are the SAME operations the direct call sites emit, reached by
   one more hop, so there is no second implementation to drift. */
/* `map(f, xs)` and `filter(f, xs)`.

   The calls all happen HERE and the result is a cursor over what they
   returned, where CPython's are lazy. That difference is visible: a `map` over
   a side-effecting function runs it eagerly, and one over an infinite source
   would not terminate. Laziness needs a resumable frame, which is the same
   thing `yield` needs and neither has yet.

   `filter(None, xs)` keeps the truthy elements -- a real form, and the reason
   the callable is tested for None rather than simply called. */
APY_API apy_value apy_map(apy_value fn, apy_value seq) {
    apy_value src = apy_getiter(seq);
    if (!src) return 0;
    return apy_cursor(src, fn, APY_IT_MAP, 0);
}

APY_API apy_value apy_filter(apy_value fn, apy_value seq) {
    apy_value src = apy_getiter(seq);
    if (!src) return 0;
    return apy_cursor(src, fn, APY_IT_FILTER, 0);
}

APY_API apy_value apy_print_seq(apy_value args) {
    apy_print((apy_value)O(args)->v.q.items, O(args)->v.q.n);
    return apy_none();
}

/* `print(*xs, sep=..., end=...)`. The starred form builds its arguments as a
   sequence at run time, so it cannot go through the stack-array entry point
   the fixed form uses -- and routing it through `apy_print_seq`, which has
   nowhere to put them, DROPPED the separator silently. */
APY_API apy_value apy_print_seq_with(apy_value args, apy_value sep,
                                     apy_value end) {
    apy_print_with((apy_value)O(args)->v.q.items, O(args)->v.q.n, sep, end);
    return apy_none();
}

APY_API apy_value apy_dict_of(apy_value args) {
    if (O(args)->v.q.n == 0) return apy_dict_new(1);
    return apy_to_dict(O(args)->v.q.items[0]);
}

APY_API apy_value apy_bytes_of(apy_value args) {
    if (O(args)->v.q.n == 0) return apy_bytes_literal((apy_value)"", 0);
    return apy_to_bytes(O(args)->v.q.items[0]);
}

/* `dict.fromkeys(keys, value)` -- one dict with every key mapped to the SAME
   value. The sharing is the point and the trap: `dict.fromkeys(ks, [])` gives
   every key the same list, and appending through one key is visible through
   all of them. */
APY_API apy_value apy_dict_fromkeys(apy_value keys, apy_value value) {
    int64_t n, i;
    apy_value out;
    /* Drained first, so a user iterator can be walked -- see `apy_sorted`. */
    keys = apy_iterable(keys);
    if (!keys) return 0;
    n = apy_raw_len(keys);
    if (apy_error_occurred()) return 0;
    out = apy_dict_new(n + 1);
    for (i = 0; i < n; i++) {
        apy_value k = apy_key_at(keys, i);
        if (!k || !apy_dict_set(out, k, value)) return 0;
    }
    return out;
}

/* `int.from_bytes(b, byteorder)`. The inverse of `to_bytes`, and unsigned:
   the signed form takes a keyword this does not offer, and guessing at it
   would turn a large positive number negative. */
APY_API apy_value apy_from_bytes_n(apy_value b, apy_value order) {
    int64_t i, n;
    uint64_t acc = 0;
    int big;
    if (O(b)->kind != APY_BYTES_K)
        return apy_fail2("TypeError",
                         "cannot convert '%s' object to bytes%s",
                         apy_kind_name(b), "");
    big = !(O(order)->kind == APY_STR_K
            && strcmp(APY_CSTR(order), "little") == 0);
    n = O(b)->v.s.n;
    if (n > 8) return apy_fail("OverflowError", "int too big to convert");
    for (i = 0; i < n; i++) {
        unsigned char byte = (unsigned char)
            O(b)->v.s.p[big ? i : n - 1 - i];
        acc = (acc << 8) | byte;
    }
    /* PAST AN i64 IS A BIG INTEGER, not a negative one. Eight unsigned
       bytes reach 1.8e19 and `int.from_bytes` is unsigned by default, so
       casting the accumulator would answer -1 for the largest of them --
       a wrong VALUE rather than a refusal. Through the decimal spelling,
       because that is the one constructor a big integer has. */
    if ((int64_t)acc < 0) {
        char text[24];
        int len = snprintf(text, sizeof text, "%llu",
                           (unsigned long long)acc);
        return apy_int_literal((apy_value)(uintptr_t)text, len, 0);
    }
    return apy_from_int((int64_t)acc);
}

APY_API apy_value apy_to_dict(apy_value src) {
    int64_t i, n;
    apy_value out;
    /* A COPY, not the same dict. `dict(d)` is a constructor and the result is
       a new object -- which only became visible once `|=` mutated in place,
       and then `c = dict(a); c |= b` changed `a` too. */
    if (O(src)->kind == APY_DICT_K) return apy_copy(src);
    /* `dict(d)` WHERE `d` IS A dict SUBCLASS copies the MAPPING, not the
       keys. Iterating a dict yields keys, so the pair walk below read
       `dict(["a"])` and reported a sequence element of the wrong length --
       about a `defaultdict` that is a perfectly good mapping. */
    if (O(src)->kind == APY_INST_K && O(src)->v.o.held
            && O(O(src)->v.o.held)->kind == APY_DICT_K)
        return apy_copy(O(src)->v.o.held);
    /* Drained first, so a user iterator can be walked -- see `apy_sorted`. */
    src = apy_iterable(src);
    if (!src) return 0;
    n = apy_raw_len(src);
    if (apy_error_occurred()) return 0;
    out = apy_dict_new(n + 1);
    for (i = 0; i < n; i++) {
        apy_value pair = apy_key_at(src, i);
        if (!pair) return 0;
        /* TWO DIFFERENT ERRORS, and Python distinguishes them: an element
           that is not a sequence at all is a TypeError -- `dict([3, 1])`
           cannot convert an int to a pair -- while one that IS a sequence of
           the wrong length is a ValueError. Reporting ValueError for both
           meant `except TypeError:` did not catch the first. */
        /* A STR IS A SEQUENCE HERE. `dict(['ab', 'cd'])` is `{'a': 'b', 'c':
           'd'}` -- each element is walked as a pair of characters -- so the
           length check applies to it and the TypeError does not. */
        {
            int64_t plen = apy_is_seq(pair) ? O(pair)->v.q.n
                : (O(pair)->kind == APY_STR_K || O(pair)->kind == APY_BYTES_K)
                    ? apy_raw_len(pair) : -1;
            if (plen < 0) {
                char b[128];
                snprintf(b, sizeof b,
                         "cannot convert dictionary update sequence element "
                         "#%d to a sequence", (int)i);
                return apy_fail("TypeError", b);
            }
            if (plen != 2) {
                char b[128];
                snprintf(b, sizeof b,
                         "dictionary update sequence element #%d has length "
                         "%d; 2 is required", (int)i, (int)plen);
                return apy_fail("ValueError", b);
            }
            {
                apy_value k = apy_key_at(pair, 0), v = apy_key_at(pair, 1);
                if (!k || !v) return 0;
                if (!apy_dict_set(out, k, v)) return 0;
            }
        }
    }
    return out;
}

/* `bytes(xs)` -- a sequence of integers in range(256). `bytes(str)` needs an
   encoding argument in Python 3 and is a TypeError without one, which is what
   the non-integer path below reports. */
APY_API apy_value apy_to_bytes(apy_value src) {
    int64_t i, n;
    char *buf;
    if (O(src)->kind == APY_MVIEW_K) return apy_mview_bytes(src);
    if (O(src)->kind == APY_BYTES_K) {
        if (!O(src)->v.s.mut) return src;
        /* A COPY, not the same buffer with the flag cleared: `bytes(ba)` is a
           snapshot, and the bytearray goes on being written to. */
        return apy_bytes_copy(O(src)->v.s.p, O(src)->v.s.n);
    }
    if (O(src)->kind == APY_STR_K)
        return apy_fail("TypeError", "string argument without an encoding");
    /* `bytes(3)` is THREE ZERO BYTES, not the digit three -- the same rule
       `bytearray(3)` follows, and the reason a count has to be tested before
       the sequence walk below asks an int to be iterable. */
    if (apy_is_int_like(src)) {
        int64_t count, k;
        char *zeros;
        if (!apy_index_arg(src, &count, APY_IDX_SUB)) return 0;
        if (count < 0) return apy_fail("ValueError", "negative count");
        zeros = (char *)malloc((size_t)(count ? count : 1) + 1);
        if (!zeros) { fputs("uasm: out of memory\n", stderr); exit(1); }
        for (k = 0; k <= count; k++) zeros[k] = 0;
        /* NOT `apy_bytes_take`: CPython builds this one straight rather
           than through the constructor that consults its cache, so
           `bytes(1) is bytes(1)` is False there -- while `bytes(0)` still
           answers the one empty. */
        if (count == 0) return apy_bytes_copy("", 0);
        return apy_bytes_own(zeros, count, 0);
    }
    /* Drained first, so a user iterator can be walked -- see `apy_sorted`. */
    src = apy_iterable(src);
    if (!src) return 0;
    n = apy_raw_len(src);
    if (apy_error_occurred()) return 0;
    buf = (char *)malloc((size_t)(n ? n : 1) + 1);
    if (!buf) { fputs("uasm: out of memory\n", stderr); exit(1); }
    for (i = 0; i < n; i++) {
        int64_t byte;
        apy_value item = apy_key_at(src, i);
        if (!item) { free(buf); return 0; }
        if (!apy_index_arg(item, &byte, APY_IDX_SUB)) { free(buf); return 0; }
        if (byte < 0 || byte > 255) {
            free(buf);
            return apy_fail("ValueError", "bytes must be in range(0, 256)");
        }
        buf[i] = (char)byte;
    }
    buf[n] = 0;
    return apy_bytes_take(buf, n);
}

/* `bytearray(...)`. Always a fresh heap buffer, never the argument's --
   `ba = bytearray(b)` then `ba[0] = 1` must not reach through to `b`, and
   the bytes it was built from may be a literal in read-only memory. */
APY_API apy_value apy_to_bytearray(apy_value src) {
    apy_value out;
    if (apy_is_int_like(src)) {
        /* `bytearray(5)` is five zero bytes, not the digit five. */
        int64_t n, i;
        char *buf;
        if (!apy_index_arg(src, &n, APY_IDX_SUB)) return 0;
        if (n < 0) return apy_fail("ValueError", "negative count");
        buf = (char *)malloc((size_t)(n ? n : 1) + 1);
        if (!buf) { fputs("uasm: out of memory\n", stderr); exit(1); }
        for (i = 0; i <= n; i++) buf[i] = 0;
        out = apy_bytes_own(buf, n, 1);
    } else {
        out = apy_to_bytes(src);
        if (!out) return 0;
        /* `apy_to_bytes` hands back its argument unchanged when it is already
           bytes; copy so the flag lands on a buffer this owns. A BYTEARRAY
           COPY AND NOT A BYTES ONE: the plain copy answers a SHARED cell for
           the empty and the one-byte values, and `bytearray(b"")` made
           writable in place would be `b""` made writable. */
        out = apy_bytearray_copy(O(out)->v.s.p, O(out)->v.s.n);
    }
    O(out)->v.s.mut = 1;
    return out;
}

/* One name into the dict `locals()` is building. A name the path taken never
   bound arrives NULL -- an unassigned register and an unset global are both
   zero, and zero is never a value -- so this SKIPS rather than storing, which
   is what makes `locals().get("v", "unbound")` answer the default after a
   branch that did not run. Every other runtime entry treats a null argument
   as "an earlier call failed"; this is the one place it means "not bound",
   and that is why the test is here rather than at the call site. */
/* A module-level name that is ALSO a builtin: what the global holds, or the
   builtin when the global holds nothing. That is the last step of Python's
   name resolution -- local, then global, then builtins -- and it is why
   `len = 5` followed by `del len` leaves `len([1, 2])` working again rather
   than raising NameError.

   `got` is read WITHOUT the null check every other entry point makes: a zero
   here means the global was never assigned or has been deleted, which is the
   question being asked rather than a failure. */
/* `a, b = xs` -- THE ARITY, checked before anything is bound. Without it a
   short sequence read past the end and reported IndexError from a subscript
   the program never wrote, and a long one bound the first two and silently
   dropped the rest, which is the worse of the two.

   `at_least` is set when the target has a `*rest`, which turns the exact
   count into a floor and changes the message to match CPython's. */
APY_API apy_value apy_unpack_check(apy_value v, int64_t want,
                                   int64_t at_least) {
    char buf[128];
    int64_t n;
    /* WHETHER THE SURPLUS IS COUNTED, which CPython decides by the SOURCE: an
       exact list, tuple or dict knows its own length, and everything else is
       unpacked through the iterator protocol, which can only say that there
       was one element too many. So `a, b = [1, 2, 3]` is `(expected 2, got
       3)` and `a, b = "abc"`, `range(3)` or `map(f, xs)` is `(expected 2)`.
       Asked of the value as it ARRIVES, before the drain below replaces it
       with a list. The shortfall is counted whatever the source, because the
       unpack took what there was and knows how many that was. */
    int counted = O(v)->kind == APY_LIST_K || O(v)->kind == APY_TUPLE_K
        || O(v)->kind == APY_DICT_K;
    /* WHAT CAN BE READ BY INDEX, which is what the unpack does: `a, *b, c =
       xs` takes `c` from `xs[-1]` without ever computing a length. So
       anything holding a POSITION rather than indices is drained here -- a
       generator, a cursor, a user object with `__iter__`.

       THE FRONTEND USED TO EMIT `apy_iterable` FOR THIS, ahead of the call.
       That drained a generator and handed a cursor straight back, so `a, b =
       map(f, xs)` reported that an `iterator` is not subscriptable -- true of
       the lowering and not of the language. Doing it here also keeps
       `counted` above answering about the source rather than about the list a
       drain produced. */
    apy_value src = apy_iterable(v);
    int k;
    if (!src) return 0;
    k = O(src)->kind;
    /* THE UNPACK WORDS ITS OWN REFUSAL. `apy_iterable` answers a value it
       does not recognise UNCHANGED -- only a user object can fail there --
       so the kinds that can be walked are tested here, and CPython's wording
       for this operation names it: `cannot unpack non-iterable int object`,
       not `'int' object is not iterable`. */
    if (!(apy_is_seq(src) || apy_is_set(src) || k == APY_STR_K
          || k == APY_BYTES_K || k == APY_DICT_K || k == APY_RANGE_K
          || k == APY_MVIEW_K || k == APY_ITER_K || k == APY_GEN_K
          || k == APY_INST_K))
        return apy_fail2("TypeError",
                         "cannot unpack non-iterable %s object%s",
                         apy_kind_name(v), "");
    /* ANYTHING NOT READ BY POSITION IS COPIED INTO A LIST. A dict subscripts
       by KEY, so `a, b = {1: 0, 2: 0}` asked for key 0 and raised KeyError; a
       set and a frozenset cannot be subscripted at all; a cursor holds a
       position instead of indices. All three are ordinary Python and all
       three failed, each in its own way, while `for` over the same value
       worked -- the same elements by another road. */
    if (!(k == APY_LIST_K || k == APY_TUPLE_K || k == APY_STR_K
          || k == APY_BYTES_K || k == APY_RANGE_K || k == APY_MVIEW_K)) {
        /* AN INSTANCE IS WALKED AND NOT SUBSCRIPTED, even the `__len__` plus
           `__getitem__` kind whose subscript is its protocol: the `*rest`
           branch reads a SLICE, and `a, *rest = obj` handed one to a
           `__getitem__` written for integers -- `unsupported operand type(s)
           for +: 'slice' and 'int'` out of a program that does not mention
           slices. CPython never asks, because it unpacks through the
           iterator. */
        apy_value into = apy_list_new(8);
        if (!into) return 0;
        if (!apy_extend(into, src)) return 0;
        src = into;
    }
    v = src;
    n = apy_raw_len(v);
    if (apy_error_occurred()) return 0;
    if (n < want) {
        snprintf(buf, sizeof buf,
                 "not enough values to unpack (expected %s%lld, got %lld)",
                 at_least ? "at least " : "", (long long)want, (long long)n);
        return apy_fail("ValueError", buf);
    }
    if (!at_least && n > want) {
        if (counted)
            snprintf(buf, sizeof buf,
                     "too many values to unpack (expected %lld, got %lld)",
                     (long long)want, (long long)n);
        else
            snprintf(buf, sizeof buf,
                     "too many values to unpack (expected %lld)",
                     (long long)want);
        return apy_fail("ValueError", buf);
    }
    /* THE SEQUENCE TO INDEX, not None: this is already the call that
       establishes the length, so it is where "what can be indexed" is
       answered -- and a second entry point for it would be one more thing
       three runtimes have to agree about. */
    return v;
}

APY_API apy_value apy_name_or(apy_value got, apy_value fallback) {
    return got ? got : fallback;
}

APY_API apy_value apy_locals_put(apy_value d, apy_value name, apy_value v) {
    if (!v) return d;
    return apy_dict_set(d, name, v) ? d : 0;
}

APY_API apy_value apy_iter(apy_value v) {
    apy_obj *o;
    if (O(v)->kind == APY_ITER_K) return v;
    /* A VIEW WALKS WHAT IT IS A VIEW OF, which `apy_getiter` and
       `apy_iterable` both already do -- this one did not, so
       `iter(d.items())` refused a thing `list(d.items())` accepts. */
    if (O(v)->kind == APY_VIEW_K) {
        /* THE ITEMS ARE COPIED OUT HERE, so the cursor's source is a list
           from now on and only this call still knows it was a view. CPython
           names the three apart -- `dict_keyiterator`,
           `dict_valueiterator`, `dict_itemiterator` -- so which one is
           recorded before the view is gone. */
        apy_value out = apy_iter(apy_view_items(v));
        if (out && O(out)->kind == APY_ITER_K)
            O(out)->v.it.named = APY_IT_VIEWED + O(v)->v.vw.part;
        return out;
    }
    /* ITERATING A CLASS IS THE METACLASS'S BUSINESS: `for c in Color` is
       `type(Color).__iter__(Color)`, which is how an enum lists its members.
       A class with no metaclass cannot be iterated, and the refusal further
       down is still the right answer for it. */
    if (O(v)->kind == APY_TYPE_K && O(v)->v.t.meta) {
        apy_value hook = apy_class_find(O(v)->v.t.meta, apy_name("__iter__"));
        if (hook) {
            apy_value got = apy_call_n(apy_bind(hook, v), NULL, 0);
            if (!got) return 0;
            /* THE SAME RULE A CLASS'S OWN `__iter__` ANSWERS UNDER: a
               metaclass writing one is still writing `__iter__`. */
            if (!apy_is_iterator(got))
                return apy_fail2("TypeError",
                                 "iter() returned non-iterator of type '%s'%s",
                                 apy_kind_name(got), "");
            return got;
        }
    }

    /* `iter(g)` IS `g`, so a half-consumed generator handed to `iter` keeps
       its position -- which is what makes `for v in g` after two `next`s
       start from the third. */
    if (O(v)->kind == APY_GEN_K) return v;
    if (O(v)->kind == APY_INST_K) {
        /* `iter(obj)` answers what `__iter__` did, unchanged, so that
           `iter(it) is it` holds for a class that returns self -- the identity
           `for v in it` on a half-consumed iterator relies on.

           UNCHANGED BUT NOT UNCHECKED: what `__iter__` answers has to BE an
           iterator, which is `apy_is_iterator_of`'s rule and the same one
           `apy_getiter` and `apy_iterable` apply. Handing a list straight
           back meant `next(iter(obj))` then reported `'list' object is not
           an iterator` -- what `next` says about something that never was
           one, rather than what `iter` says about what `__iter__` gave it. */
        apy_value got = apy_unary_dunder(v, "__iter__");
        if (got) {
            if (!apy_is_iterator(got))
                return apy_fail2("TypeError",
                                 "iter() returned non-iterator of type '%s'%s",
                                 apy_kind_name(got), "");
            return got;
        }
        if (apy_error_occurred()) return 0;
        /* A CLASS EXTENDING A BUILTIN IS ITERABLE BECAUSE THE BUILTIN IS.
           `for k in d` over a `class D(dict)` walks its keys; the miss above
           is not the answer, because `__iter__` was never in the body. */
        if (O(v)->v.o.held) return apy_iter(O(v)->v.o.held);
        got = apy_iterable(v);
        if (!got) return 0;
        if (got != v) return apy_iter(got);
    }
    /* A MEMORYVIEW IS WALKED BY INDEX, which is what it already answers to
       everywhere else: `list(mv)`, `[*mv]`, `98 in mv` and `reversed(mv)`
       all worked and `iter(mv)` alone said it was not iterable. The cursor
       steps through `apy_getitem`, and `mv[i]` is the int CPython yields. */
    if (!apy_is_seq(v) && !apy_is_set(v) && O(v)->kind != APY_STR_K
        && O(v)->kind != APY_BYTES_K && O(v)->kind != APY_DICT_K
        && O(v)->kind != APY_MVIEW_K && O(v)->kind != APY_RANGE_K)
        return apy_fail2("TypeError", "'%s' object is not iterable%s",
                         apy_kind_name(v), "");
    o = apy_alloc(APY_ITER_K);
    o->v.it.src = v;
    o->v.it.i = 0;
    /* NOT INHERITED FROM THE UNION. `fn`, `mode` and `n0` are read on every
       step; leaving them as whatever the recycled cell held made a plain
       `iter(d)` report the dict as resized on its very first `next`. */
    o->v.it.fn = 0;
    o->v.it.mode = APY_IT_PLAIN;
    o->v.it.n0 = (O(v)->kind == APY_DICT_K) ? O(v)->v.d.n : -1;
    /* WHAT IT IS NAMED AFTER -- see `apy_cursor_name`. */
    o->v.it.named = O(v)->kind;
    return V(o);
}

/* WHAT TO WALK. Answers `v` itself for anything this runtime can index, and
   for a user object DRAINS the iterator protocol into a list.

   Draining is eager where CPython is lazy, and that is visible: a `for` over
   an infinite `__next__` never starts rather than never ending, and a
   generator-shaped class's side effects all happen before the first pass.
   Laziness needs a resumable frame, which is the same thing `yield` needs.

   Called once where a loop begins, so the cost is one pass and not one per
   element -- and so a class with `__len__` and `__getitem__` is left alone
   entirely, because the index walk below already IS its protocol. */
/* IS THIS AN ITERATOR -- the rule `__iter__`'s ANSWER has to satisfy. A str
   is not one however walkable it looks, and accepting one turned a broken
   class into a working one that iterated something else entirely.

   FACTORED OUT so `str.join` can ask the same question and word the refusal
   its own way: CPython reaches iteration through `PySequence_Fast(seq, "can
   only join an iterable")`, which REPLACES any TypeError that comes out of
   GETTING the iterator with that one sentence and names no kind, while
   errors raised while WALKING still propagate. Asking here rather than
   clearing `apy_iterable`'s flag afterwards is what keeps that distinction:
   this funnel gets the iterator AND drains it, so a cleared flag would also
   swallow a TypeError from a user's `__next__`. */
APY_API int64_t apy_is_iterator_of(apy_value it) {
    /* A GENERATOR, A CURSOR, OR A CLASS WITH `__next__`, AND NOTHING ELSE.
       A list is not an iterator and neither is a dict, a set or a tuple --
       `iter()` makes one FROM each of them, which is a different thing, and
       this funnel used to accept all four. A class whose `__iter__` answered
       a list then iterated it happily where CPython refuses, so a broken
       class silently worked and the author was never told which method to
       fix. */
    return O(it)->kind == APY_GEN_K || O(it)->kind == APY_ITER_K
        || (O(it)->kind == APY_INST_K
            && apy_class_find(O(it)->v.o.cls, apy_name("__next__")) != 0);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_is_iterator(apy_value it) {
    return (int)apy_is_iterator_of(it);
}

APY_API apy_value apy_iterable(apy_value v) {
    apy_value it, out;
    int64_t guard;
    /* A VIEW IS READ WHEN IT IS WALKED, which is what makes it live. Every
       consumer that wants a sequence comes through here, so this one line is
       where the liveness actually happens. */
    if (O(v)->kind == APY_VIEW_K) return apy_view_items(v);
    /* A GENERATOR is drained: the walk below is by index and an index walk
       needs a length. See `apy_gen_drain` for what that costs. */
    if (O(v)->kind == APY_GEN_K) return apy_gen_drain(v);
    /* AN ALIAS YIELDS ONE ITEM -- itself, starred. Here as well as in
       `apy_getiter` because this is the EAGER funnel: `list(list[int])` and
       `*list[int],` come through here, and the lazy `for` comes through
       there. See `apy_getiter` for why an alias is iterable at all. */
    if (O(v)->kind == APY_ALIAS_K) {
        apy_value one = apy_tuple_new(1);
        apy_q_append_of(one, apy_alias_unpack(v));
        return one;
    }
    /* ITERATING A CLASS IS THE METACLASS'S BUSINESS: `for c in Color` is
       `type(Color).__iter__(Color)`, which is how an enum lists its members.
       A class with no metaclass cannot be iterated, and the refusal further
       down is still the right answer for it. */
    if (O(v)->kind == APY_TYPE_K && O(v)->v.t.meta) {
        apy_value hook = apy_class_find(O(v)->v.t.meta, apy_name("__iter__"));
        if (hook) {
            apy_value got = apy_call_n(apy_bind(hook, v), NULL, 0);
            if (!got) return 0;
            /* THE SAME RULE A CLASS'S OWN `__iter__` ANSWERS UNDER: a
               metaclass writing one is still writing `__iter__`. */
            if (!apy_is_iterator(got))
                return apy_fail2("TypeError",
                                 "iter() returned non-iterator of type '%s'%s",
                                 apy_kind_name(got), "");
            return apy_iterable(got);
        }
    }
    if (O(v)->kind != APY_INST_K) return v;
    /* A CLASS EXTENDING A BUILTIN WALKS THE BUILTIN. This is the EAGER
       funnel -- unpacking, `sorted`, `list(x)` -- as `apy_getiter` is the
       lazy one, and `a, b, c = t` on a `class T(tuple)` comes through here.
       Before the dunder walk, which would find nothing and refuse. */
    if (O(v)->v.o.held && !apy_class_find(O(v)->v.o.cls, apy_name("__iter__")))
        return apy_iterable(O(v)->v.o.held);
    it = apy_unary_dunder(v, "__iter__");
    if (apy_error_occurred()) return 0;
    if (it) {
        /* WHAT `__iter__` RETURNS MUST BE AN ITERATOR -- see `apy_getiter`,
           which enforces the same rule on the lazy path, and
           `apy_is_iterator_of`, which is the rule itself. */
        if (!apy_is_iterator(it))
            return apy_fail2("TypeError",
                             "iter() returned non-iterator of type '%s'%s",
                             apy_kind_name(it), "");
    }
    if (!it) {
        /* No `__iter__`. THE OLDER PROTOCOL IS `__getitem__` ALONE, walked
           from 0 until it reports IndexError -- `__len__` is no part of it
           and CPython never consults it.

           Reading `__len__` as a BOUND made a class whose two disagree
           answer differently: one with a `__len__` of 2 and a `__getitem__`
           good for five stopped at two, and one whose `__getitem__` never
           raises is endless in CPython and was bounded here, so `a, b = obj`
           bound where CPython reports too many values. A `__len__` with no
           `__getitem__` does not make an object iterable at all, which is
           the other half of the same rule and is what the refusal below
           now covers. */
        if (!apy_class_find(O(v)->v.o.cls, apy_name("__getitem__")))
            return apy_fail2("TypeError", "'%s' object is not iterable%s",
                             apy_kind_name(v), "");
        out = apy_seq_new(APY_LIST_K, 8);
        for (guard = 0; guard < 1000000; guard++) {
            apy_value got = apy_getitem(v, apy_from_int(guard));
            if (!got) {
                if (apy_error_matches(apy_lit("IndexError"))) {
                    apy_error_clear();
                    break;
                }
                return 0;
            }
            apy_seq_push(out, got);
        }
        return out;
    }
    if (O(it)->kind != APY_INST_K
        || !apy_class_find(O(it)->v.o.cls, apy_name("__next__")))
        /* `__iter__` handed back something already walkable -- a list, or a
           real iterator. */
        return apy_iterable(it);
    out = apy_seq_new(APY_LIST_K, 8);
    for (guard = 0; guard < 1000000; guard++) {
        apy_value got = apy_unary_dunder(it, "__next__");
        if (!got) {
            if (apy_error_matches(apy_lit("StopIteration"))) {
                apy_error_clear();
                break;
            }
            return 0;
        }
        apy_seq_push(out, got);
    }
    return out;
}

/* `next(it)`, and `next(it, default)` when `has_default`.

   Exhaustion is a StopIteration, which is an ordinary exception here -- so a
   `try: next(it) except StopIteration:` works, and so does a `for` loop that
   never sees one because it counts instead. */
APY_API apy_value apy_next(apy_value it, apy_value fallback,
                           int64_t has_default) {
    /* ONE STEP, through the same protocol `for` uses. Anything with a
       position -- a generator, a cursor, a user iterator -- advances the same
       way, so `next(map(f, xs))` calls `f` once rather than draining.
       Exhaustion is a StopIteration here and a sentinel there, which is the
       only difference between the two spellings. */
    apy_value got;
    if (O(it)->kind != APY_GEN_K && O(it)->kind != APY_ITER_K
        && O(it)->kind != APY_INST_K)
        return apy_fail2("TypeError", "'%s' object is not an iterator%s",
                         apy_kind_name(it), "");
    got = apy_step(it);
    if (!got) return 0;
    if (got == apy_stop()) {
        if (has_default) return fallback;
        /* A GENERATOR CARRIES ITS RETURN VALUE OUT IN THE EXCEPTION:
           `return "done"` becomes `StopIteration("done")`, and `e.value` is
           how a program reads it. Raising a bare one here threw that away --
           `yield from` still saw the value, because it reads the object
           rather than catching, so the two spellings disagreed about the same
           generator. */
        if (O(it)->kind == APY_GEN_K) return apy_gen_stop(it);
        return apy_fail("StopIteration", "");
    }
    return got;
}

APY_API apy_value apy_sorted_by(apy_value seq, apy_value keyfn,
                                apy_value reverse) {
    /* `key=None` is "no key", which is what an omitted one lowers to. */
    apy_value fn = O(keyfn)->kind == APY_NONE_K ? 0 : keyfn;
    /* `sorted(xs, key=f)` ASKS TOO. The keyword does not change what the
       call does before it starts; see `apy_sorted`. */
    if (!apy_length_hint(seq)) return 0;
    return apy_sort_with(seq, fn, apy_truth(reverse) != 0);
}

/* `min(xs, key=f)` / `max(xs, key=f)`. The FIRST extreme wins for min and the
   LAST for max, which is CPython's tie-breaking and the reason the two
   comparisons below are not symmetric. */
APY_API apy_value apy_extreme_by_of(apy_value seq, apy_value keyfn,
                                   int64_t want_max) {
    int64_t n, i;
    apy_value best = 0, best_key = 0;
    /* Drained first, so a user iterator can be walked -- see `apy_sorted`. */
    seq = apy_iterable(seq);
    if (!seq) return 0;
    n = apy_raw_len(seq);
    if (apy_error_occurred()) return 0;
    for (i = 0; i < n; i++) {
        apy_value item = apy_key_at(seq, i), k;
        if (!item) return 0;
        k = keyfn ? apy_call_n(keyfn, &item, 1) : item;
        if (!k) return 0;
        if (!best) { best = item; best_key = k; continue; }
        {
            int c = apy_order_rich(k, best_key);
            if (c == 2) { apy_order_error((int)want_max, k, best_key); return 0; }
            if (want_max ? c > 0 : c < 0) { best = item; best_key = k; }
        }
    }
    if (!best)
        return apy_fail(want_max ? "ValueError" : "ValueError",
                        want_max ? "max() iterable argument is empty"
                                 : "min() iterable argument is empty");
    return best;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_extreme_by(apy_value seq, apy_value keyfn,
                                int want_max) {
    return apy_extreme_by_of(seq, keyfn, (int64_t)want_max);
}

APY_API apy_value apy_min_by(apy_value seq, apy_value keyfn) {
    return apy_extreme_by(seq, O(keyfn)->kind == APY_NONE_K ? 0 : keyfn, 0);
}

APY_API apy_value apy_max_by(apy_value seq, apy_value keyfn) {
    return apy_extreme_by(seq, O(keyfn)->kind == APY_NONE_K ? 0 : keyfn, 1);
}

APY_API apy_value apy_extreme_of(apy_value seq, int64_t want_max) {
    int64_t n, i;
    apy_value best;
    /* Drained first, so a user iterator can be walked -- see `apy_sorted`. */
    seq = apy_iterable(seq);
    if (!seq) return 0;
    n = apy_raw_len(seq);
    if (apy_error_occurred()) return 0;
    if (n == 0)
        return apy_fail(want_max ? "ValueError" : "ValueError",
                        want_max ? "max() iterable argument is empty"
                                 : "min() iterable argument is empty");
    best = apy_key_at(seq, 0);
    for (i = 1; i < n; i++) {
        apy_value item = apy_key_at(seq, i);
        int c = apy_order_rich(item, best);
        if (c == 2) { apy_order_error((int)want_max, item, best); return 0; }
        /* Strict, so that on a tie the EARLIER element wins -- which is what
           CPython does and is observable when the elements are equal but
           distinguishable. */
        if (want_max ? c > 0 : c < 0) best = item;
    }
    return best;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_extreme(apy_value seq, int want_max) {
    return apy_extreme_of(seq, (int64_t)want_max);
}

APY_API apy_value apy_min(apy_value seq) { return apy_extreme(seq, 0); }
APY_API apy_value apy_max(apy_value seq) { return apy_extreme(seq, 1); }

APY_API apy_value apy_sum(apy_value seq) {
    int64_t n, i;
    apy_value total;
    /* Drained first, so a user iterator can be walked -- see `apy_sorted`. */
    seq = apy_iterable(seq);
    if (!seq) return 0;
    n = apy_raw_len(seq);
    if (apy_error_occurred()) return 0;
    /* Starts at the INT zero, so `sum([])` is 0 and not 0.0, and so that a
       list of ints sums to an int. */
    total = apy_from_int(0);
    for (i = 0; i < n; i++) {
        total = apy_add(total, apy_key_at(seq, i));
        if (!total) return 0;
    }
    return total;
}

/* `sum(xs, start)`. The start is what an empty sequence returns and what sets
   the result's TYPE -- `sum([], 0.0)` is `0.0` and `sum([Vec()], Vec())` is a
   Vec, neither of which an int zero could produce. */
APY_API apy_value apy_sum_from(apy_value seq, apy_value start) {
    int64_t n, i;
    apy_value total = start;
    /* `sum` REFUSES STRINGS, and it is not an oversight in CPython: joining
       strings this way is quadratic, and `''.join(...)` is the answer. The
       concatenation works perfectly well, which is exactly why the refusal
       has to be explicit rather than emergent. */
    if (O(start)->kind == APY_STR_K)
        return apy_fail("TypeError",
                        "sum() can't sum strings [use ''.join(seq) instead]");
    if (O(start)->kind == APY_BYTES_K)
        return apy_fail("TypeError",
                        "sum() can't sum bytes [use b''.join(seq) instead]");
    /* Drained first, so a user iterator can be walked -- see `apy_sorted`. */
    seq = apy_iterable(seq);
    if (!seq) return 0;
    n = apy_raw_len(seq);
    if (apy_error_occurred()) return 0;
    for (i = 0; i < n; i++) {
        total = apy_add(total, apy_key_at(seq, i));
        if (!total) return 0;
    }
    return total;
}

/* `min(a, b, ...)` / `max(a, b, ...)` -- the MULTI-ARGUMENT form, which is a
   different function from the one-iterable form and not sugar for it:
   `min([3, 1])` is 1 and `min([3], [1])` is `[1]`. Two arguments already
   distinguish them, so the frontend picks by argument count. */
APY_API apy_value apy_extreme_n(apy_value buf, int64_t n, int64_t want_max) {
    apy_value *argv = (apy_value *)buf;
    apy_value best;
    int64_t i;
    if (n < 1) return apy_fail("TypeError", "min expected at least 1 argument");
    best = argv[0];
    for (i = 1; i < n; i++) {
        int c = apy_order_rich(argv[i], best);
        if (c == 2) { apy_order_error((int)want_max, argv[i], best); return 0; }
        if (want_max ? c > 0 : c < 0) best = argv[i];
    }
    return best;
}

/* `min(xs, default=v)` / `max(xs, default=v)`. Only an EMPTY iterable reaches
   the default; a non-empty one ignores it, which is why this cannot simply be
   "the answer or `v`" applied to the existing entry point -- that one has
   already reported a ValueError by then. */
APY_API apy_value apy_extreme_or(apy_value seq, apy_value keyfn,
                                 apy_value fallback, int64_t want_max) {
    int64_t n;
    /* Drained first, so a user iterator can be walked -- see `apy_sorted`. */
    seq = apy_iterable(seq);
    if (!seq) return 0;
    n = apy_raw_len(seq);
    if (apy_error_occurred()) return 0;
    if (n == 0) return fallback;
    if (O(keyfn)->kind == APY_NONE_K)
        return want_max ? apy_max(seq) : apy_min(seq);
    return want_max ? apy_max_by(seq, keyfn) : apy_min_by(seq, keyfn);
}

APY_API apy_value apy_reversed(apy_value seq) {
    int64_t n;
    apy_value out;
    int kind = O(seq)->kind;
    /* `__reversed__` WINS OVER THE INDEX WALK. A class may define both it and
       `__getitem__`, and they need not agree -- the hook is the answer the
       class chose, and walking indices backwards instead silently produced a
       different sequence from the one it asked for. */
    if (kind == APY_INST_K) {
        apy_value hook = apy_unary_dunder(seq, "__reversed__");
        if (apy_error_occurred()) return 0;
        if (hook) return apy_iterable(hook);
    }
    /* REVERSING NEEDS A SEQUENCE: a length and indexing, or a `__reversed__`.
       A set has no order, and a cursor or a generator has a POSITION rather
       than a length -- CPython refuses all three by name, in these words.
       The refusal has to be explicit because the walk below would otherwise
       have produced a confident answer to a question the type cannot be
       asked: a set in an arbitrary order, and an iterator drained FORWARDS
       by a walk that thinks it is going backwards. */
    if (!(kind == APY_LIST_K || kind == APY_TUPLE_K || kind == APY_STR_K
          || kind == APY_BYTES_K || kind == APY_DICT_K || kind == APY_RANGE_K
          || kind == APY_MVIEW_K || kind == APY_VIEW_K
          /* An instance with the older protocol -- or one extending a
             builtin, whose length and indexing are the builtin's. */
          || (kind == APY_INST_K
              && (O(seq)->v.o.held
                  || apy_class_find(O(seq)->v.o.cls,
                                    apy_name("__getitem__"))))))
        return apy_fail2("TypeError", "'%s' object is not reversible%s",
                         apy_kind_name(seq), "");
    /* A VIEW IS READ NOW, as it is for a forward walk: `reversed(d.keys())`
       reverses the keys the dict has at this point, and the cursor's source
       is the copy from here on -- so which view it was is recorded before it
       is gone. */
    if (kind == APY_VIEW_K) {
        apy_value items = apy_view_items(seq);
        if (!items) return 0;
        out = apy_cursor(items, 0, APY_IT_REV, O(items)->v.q.n - 1);
        if (out)
            O(out)->v.it.named =
                APY_IT_REVOF + APY_IT_VIEWED + O(seq)->v.vw.part;
        return out;
    }
    n = apy_raw_len(seq);
    if (apy_error_occurred()) return 0;
    /* A CURSOR COUNTING DOWN, not a list built backwards. `reversed(xs)` is
       an ITERATOR in CPython: it has no length, cannot be indexed and cannot
       be walked twice, and a list answered all three wrongly -- `type(...)`
       said `list`, `isinstance(..., list)` was True, and the whole sequence
       was copied before the first element was wanted. See `APY_IT_REV`. */
    out = apy_cursor(seq, 0, APY_IT_REV, n - 1);
    if (out) O(out)->v.it.named = APY_IT_REVOF + kind;
    return out;
}

APY_API apy_value apy_enumerate(apy_value seq, int64_t start) {
    apy_value src = apy_getiter(seq);
    if (!src) return 0;
    return apy_cursor(src, 0, APY_IT_ENUMERATE, start);
}

/* `zip(...)` for any number of iterables, including none -- `zip()` is empty,
   which is not the same as an error, and `zip(xs)` yields 1-tuples.

   `strict=1` makes an uneven zip a ValueError instead of a silent truncation,
   which is the whole point of PEP 618: the lossiness is useful and is also
   the bug, so the caller says which it meant. */
APY_API apy_value apy_zip_n(apy_value buf, int64_t argc, int64_t strict) {
    apy_value *argv = (apy_value *)buf;
    apy_value cursors = apy_seq_new(APY_LIST_K, argc + 1);
    int64_t k;
    for (k = 0; k < argc; k++) {
        apy_value got = apy_getiter(argv[k]);
        if (!got) return 0;
        apy_seq_push(cursors, got);
    }
    return apy_cursor(cursors, apy_from_bool(strict != 0), APY_IT_ZIP, 0);
}

/* `delattr(o, name)` and `del o.name`. Removing an attribute that is not
   there is an AttributeError, not a no-op: the two are different programs and
   only one of them is asking for something that exists. */
APY_API apy_value apy_default_delattr(apy_value obj, apy_value name);

APY_API apy_value apy_delattr(apy_value obj, apy_value name) {
    /* `__delattr__`, the same rule as `__setattr__`. */
    if (O(obj)->kind == APY_INST_K) {
        apy_value hook = apy_class_find(O(obj)->v.o.cls,
                                        apy_name("__delattr__"));
        if (hook) return apy_call_n(apy_bind(hook, obj), &name, 1);
        {
            /* A DATA DESCRIPTOR TAKES THE DELETE, exactly as it takes the
               write -- `__delete__` is the third of the three, and a property
               or a user descriptor that defines it never reaches the instance
               dict. Without this, `del c.d` on a descriptor attribute looked
               in the dict, found nothing, and reported an attribute the class
               plainly has. */
            apy_value found = apy_class_find(O(obj)->v.o.cls, name);
            if (found && apy_is_data_descriptor(found)) {
                apy_value m = 0;
                if (O(found)->kind == APY_PROP_K) {
                    if (!O(found)->v.p.del_)
                        return apy_fail("AttributeError",
                                        "can't delete attribute");
                    return apy_call_n(O(found)->v.p.del_, &obj, 1);
                }
                m = apy_class_find(O(found)->v.o.cls, apy_name("__delete__"));
                if (m) return apy_call_n(apy_bind(m, found), &obj, 1);
            }
        }
    }
    return apy_default_delattr(obj, name);
}

APY_API apy_value apy_default_delattr(apy_value obj, apy_value name) {
    if (O(obj)->kind != APY_INST_K)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute '%s'",
                         apy_kind_name(obj), APY_CSTR(name));
    if (apy_dict_find(O(obj)->v.o.dict, name) < 0)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute '%s'",
                         apy_kind_name(obj), APY_CSTR(name));
    return apy_delitem(O(obj)->v.o.dict, name);
}

APY_API apy_value apy_zip2(apy_value a, apy_value b) {
    apy_value pair[2];
    pair[0] = a;
    pair[1] = b;
    return apy_zip_n((apy_value)pair, 2, 0);
}

/* `range` as a VALUE, materialised as a list. Python's is a lazy sequence
   with its own type, so `type(range(3)).__name__` and the memory of
   `range(10**9)` both differ -- a stated divergence. A `for` header does NOT
   come here: it lowers to a counter loop with no allocation at all, which is
   the case that matters for cost. */
APY_API apy_value apy_range(int64_t start, int64_t stop, int64_t step) {
    apy_obj *o;
    if (step == 0)
        return apy_fail("ValueError", "range() arg 3 must not be zero");
    o = apy_alloc(APY_RANGE_K);
    o->v.rg.start = start;
    o->v.rg.stop = stop;
    o->v.rg.step = step;
    return V(o);
}

/* How many elements a range has. The formula rather than a walk: that is the
   whole reason a range is three numbers. */
static int64_t apy_range_len(apy_value r) {
    int64_t start = O(r)->v.rg.start, stop = O(r)->v.rg.stop;
    int64_t step = O(r)->v.rg.step, n;
    n = step > 0 ? (stop - start + step - 1) / step
                 : (start - stop - step - 1) / (-step);
    return n < 0 ? 0 : n;
}

static int64_t apy_range_at(apy_value r, int64_t i) {
    return O(r)->v.rg.start + i * O(r)->v.rg.step;
}

/* Is `want` in the range, and where? Arithmetic, so `10**11 in range(10**12)`
   is a division and not a search. Answers -1 for absent. */
static int64_t apy_range_find(apy_value r, int64_t want) {
    int64_t start = O(r)->v.rg.start, step = O(r)->v.rg.step, off, at;
    off = want - start;
    if (off % step) return -1;
    at = off / step;
    return (at >= 0 && at < apy_range_len(r)) ? at : -1;
}

APY_API apy_value apy_abs(apy_value v) {
    if (O(v)->kind == APY_INST_K) {
        apy_value r = apy_unary_dunder(v, "__abs__");
        if (r || apy_error_occurred()) return r;
    }
    if (O(v)->kind == APY_FLOAT_K) return apy_from_float(fabs(O(v)->v.f));
    if (apy_is_big(v))
        return O(v)->v.big.neg ? apy_neg(v) : v;
    if (apy_is_int_like(v))
        /* `abs(INT64_MIN)` does not fit an int64, so it goes through the
           negation that knows how to promote rather than through `-v`. */
        return O(v)->v.i < 0 ? apy_neg(v) : apy_from_int(O(v)->v.i);
    /* `abs(complex)` is its MODULUS, and a float -- the one kind for which
       abs changes the type rather than the sign. */
    if (O(v)->kind == APY_COMPLEX_K)
        return apy_from_float(sqrt(O(v)->v.z.re * O(v)->v.z.re
                                   + O(v)->v.z.im * O(v)->v.z.im));
    return apy_fail2("TypeError", "bad operand type for abs(): '%s'%s",
                     apy_kind_name(v), "");
}

/* `v.__index__()` called directly -- what `operator.index` and PEP 357's
   other two consumers (`hex`/`oct`/`bin`, a subscript) reach through
   already, but as the BOXED Python object those name rather than the
   machine word `apy_index` (mathints.py) unpacks for a subscript.

   Mirrors `apy_to_int`'s numeric branch: a big answers ITSELF -- it is
   already exactly the integer it names -- and a bool answers a genuine
   `int`, not itself, because `True.__index__()` is `1` and `type(1)` is not
   `bool`; `apy_is_int_like` covers both and `apy_from_int` does the
   normalising for the ones big skips. */
APY_API apy_value apy_index_obj(apy_value v) {
    if (O(v)->kind == APY_INST_K) {
        apy_value r = apy_unary_dunder(v, "__index__");
        if (r || apy_error_occurred()) return r;
    }
    if (apy_is_big(v)) return v;
    if (apy_is_int_like(v)) return apy_from_int(O(v)->v.i);
    return apy_fail2("TypeError",
                     "'%s' object cannot be interpreted as an integer%s",
                     apy_kind_name(v), "");
}

/* `weakref.ref` -- INTERPRETER-ONLY, same reason as `sys.getrefcount`
   above: this runtime keeps no reference count and frees nothing, so a
   compiled build has no moment at which an object becomes weakly
   UNREACHABLE and no honest answer to give here. Refusing BY NAME rather
   than one of the two plausible-looking wrong answers -- always claiming
   the target is alive (true in the sense that this runtime never frees
   it, false in the sense that CPython's `weakref` would already have
   said otherwise) or silently never firing a callback. */
APY_API apy_value apy_weakref_register(apy_value target, apy_value ref_self,
                                       apy_value callback) {
    (void)target; (void)ref_self; (void)callback;
    return apy_fail("NotImplementedError",
                    "weakref needs the reference interpreter -- a compiled "
                    "build keeps no reference count to weaken against");
}

APY_API apy_value apy_weakref_deref(apy_value target) {
    (void)target;
    return apy_fail("NotImplementedError",
                    "weakref needs the reference interpreter -- a compiled "
                    "build keeps no reference count to weaken against");
}

APY_API apy_value apy_weakref_count(apy_value target) {
    (void)target;
    return apy_fail("NotImplementedError",
                    "weakref needs the reference interpreter -- a compiled "
                    "build keeps no reference count to weaken against");
}

APY_API apy_value apy_weakref_existing(apy_value target) {
    (void)target;
    return apy_fail("NotImplementedError",
                    "weakref needs the reference interpreter -- a compiled "
                    "build keeps no reference count to weaken against");
}

/* `gc.collect()`. Same interpreter-only story: finding a cycle means
   subtracting the references its members make to each other from their
   reference COUNTS, and this build keeps none. Refusing rather than
   answering 0, which would read as "nothing was collectable" -- a
   plausible number, and a false one. */
/* `sys.getrefcount(obj)`. Interpreter-only for the same reason as the
   two above: there is no reference count in a compiled build to report. */
APY_API apy_value apy_sys_getrefcount(apy_value v) {
    (void)v;
    return apy_fail("NotImplementedError",
                    "sys.getrefcount() needs the reference interpreter -- a "
                    "compiled build keeps no reference count to report");
}

APY_API apy_value apy_gc_collect(void) {
    return apy_fail("NotImplementedError",
                    "gc.collect() needs the reference interpreter -- a "
                    "compiled build keeps no reference count to collect by");
}

/* `round` is round-HALF-TO-EVEN, which C's `round` is not: C rounds half away
   from zero, so it answers 3 for round(2.5) where Python answers 2. And
   `round(x)` with no digits returns an INT. */
APY_API apy_value apy_round(apy_value v) {
    /* `__round__` WITH NO DIGITS. A class defining it decides what rounding
       itself means, and answering from the numeric tower instead would round
       something the class never claimed was a number. */
    if (O(v)->kind == APY_INST_K) {
        apy_value got = apy_unary_dunder(v, "__round__");
        if (apy_error_occurred()) return 0;
        if (got) return got;
    }
    double x, down, frac;
    if (apy_is_big(v)) return v;      /* already whole, and already exact */
    if (apy_is_int_like(v)) return apy_from_int(O(v)->v.i);
    if (O(v)->kind != APY_FLOAT_K)
        return apy_fail2("TypeError",
                         "type '%s' doesn't define __round__ method%s",
                         apy_kind_name(v), "");
    x = O(v)->v.f;
    down = floor(x);
    frac = x - down;
    if (frac > 0.5) down += 1.0;
    else if (frac == 0.5 && fmod(down, 2.0) != 0.0) down += 1.0;
    /* `round(1e30)` is an integer with 100 bits, and casting it to int64 is
       undefined rather than merely wrong. */
    if (down >= 9223372036854775808.0 || down < -9223372036854775808.0)
        return apy_big_from_double(down);
    return apy_from_int((int64_t)down);
}

/* `round(x, n)`. A different function from `round(x)` in more than precision:
   the one-argument form returns an INT and this one returns a float, so
   `round(2.5)` is `2` and `round(2.5, 0)` is `2.0`.

   The rounding goes through the C library's decimal conversion and back,
   which is what CPython does (`_Py_dg_dtoa` in mode 3, then `strtod`) and the
   only way to get `round(2.675, 2) == 2.67` right: 2.675 is not 2.675 but
   2.67499999999999982..., so any scale-multiply-round-divide gets 2.68 and
   disagrees with Python on a number every tutorial uses as the example. */
APY_API apy_value apy_round_to(apy_value v, apy_value nd) {
    /* `__round__(ndigits)` -- the two-argument form, which is a different
       call into the same hook. */
    if (O(v)->kind == APY_INST_K) {
        apy_value hook = apy_class_find(O(v)->v.o.cls, apy_name("__round__"));
        if (hook) {
            apy_value arg = nd;
            return apy_call_n(apy_bind(hook, v), &arg, 1);
        }
    }
    int64_t n;
    double x, p, y, r;
    char buf[512];
    if (O(nd)->kind == APY_NONE_K) return apy_round(v);
    if (!apy_is_int_like(nd))
        return apy_fail2("TypeError",
                         "'%s' object cannot be interpreted as an integer%s",
                         apy_kind_name(nd), "");
    n = O(nd)->v.i;
    if (apy_is_big(v)) return v;
    if (apy_is_int_like(v)) {
        /* An int stays an int at ANY precision. A negative one rounds to a
           multiple of a power of ten, half to even like everything else. */
        int64_t i = O(v)->v.i, scale = 1, k, half, rem;
        if (n >= 0) return apy_from_int(i);
        for (k = 0; k < -n; k++) {
            if (scale > 922337203685477580LL) return apy_from_int(0);
            scale *= 10;
        }
        rem = i % scale;
        if (rem < 0) rem += scale;
        half = scale / 2;
        i -= rem;
        if (rem > half || (rem == half && ((i / scale) & 1))) i += scale;
        return apy_from_int(i);
    }
    if (O(v)->kind != APY_FLOAT_K)
        return apy_fail2("TypeError",
                         "type '%s' doesn't define __round__ method%s",
                         apy_kind_name(v), "");
    x = O(v)->v.f;
    if (x != x || x - x != 0.0) return apy_from_float(x);   /* nan, inf */
    if (n > 300) return apy_from_float(x);   /* finer than a double is */
    if (n >= 0) {
        snprintf(buf, sizeof buf, "%.*f", (int)n, x);
        return apy_from_float(strtod(buf, NULL));
    }
    if (n < -300) return apy_from_float(x < 0 ? -0.0 : 0.0);
    p = pow(10.0, (double)-n);
    y = x / p;
    r = floor(y);
    {
        double frac = y - r;
        if (frac > 0.5) r += 1.0;
        else if (frac == 0.5 && fmod(r, 2.0) != 0.0) r += 1.0;
    }
    return apy_from_float(r * p);
}

/* `issubclass(a, b)`. Only for user classes and only by the base chain, which
   is all single inheritance can be asked. A non-class first argument is a
   TypeError and not False -- `issubclass(1, int)` raises, where
   `isinstance(1, int)` answers. */
/* Is `v` the `object` type, however it was spelled? `object` in source
   lowers to the one cell `apy_object_class` hands out; a builtin type name
   held in a variable arrives as a function marked as standing for a type.
   Both carry the name, and the name is what settles it. */
APY_API int64_t apy_names_object(apy_value v) {
    apy_value name = 0;
    if (!v) return 0;
    if (O(v)->kind == APY_TYPE_K) name = O(v)->v.t.name;
    else if (O(v)->kind == APY_FUNC_K && O(v)->v.fn.is_type)
        name = O(v)->v.fn.name;
    if (!name || O(name)->kind != APY_STR_K) return 0;
    return strcmp(APY_CSTR(name), "object") == 0;
}

/* Is `v` something `issubclass` may be ASKED about? */
APY_API int64_t apy_is_classlike(apy_value v) {
    if (!v) return 0;
    if (O(v)->kind == APY_TYPE_K) return 1;
    return O(v)->kind == APY_FUNC_K && O(v)->v.fn.is_type;
}

APY_API apy_value apy_is_subclass(apy_value a, apy_value b) {
    /* THE METACLASS DECIDES, if it says so. `__subclasscheck__` is asked
       before the base chain is walked, which is what lets a metaclass claim
       a class it has no structural relationship to -- and is the whole of
       what `issubclass` means for an abstract base class. Asked FIRST, and
       of `b`, because it is the class being tested AGAINST. */
    if (b && O(b)->kind == APY_TYPE_K && O(b)->v.t.meta) {
        apy_value hook = apy_class_find(O(b)->v.t.meta,
                                        apy_name("__subclasscheck__"));
        if (hook) {
            apy_value args[2];
            apy_value got;
            args[0] = b;
            args[1] = a;
            got = apy_call_n(hook, args, 2);
            return got ? apy_from_bool(apy_truth(got) != 0) : got;
        }
    }
    /* BUILTIN TYPES REACHED AS VALUES: `issubclass(bool, int)`. Each side
       is a callable thunk carrying its name, so the question is asked of the
       names -- the same rule `isinstance` uses, and the only place the
       builtin subtype relation is written down. */
    /* EVERYTHING IS A SUBCLASS OF `object`, and nothing's base chain
       contains it: `object` lowers to one type cell that no class names as a
       base, so walking the chain answered False for every class -- and
       `issubclass(int, object)`, where the first argument is a builtin NAME
       and the second a type cell, matched neither shape below and was
       refused outright. Both are wrong about the most basic relation there
       is. */
    if (apy_names_object(b) && apy_is_classlike(a)) return apy_from_bool(1);
    if (a && O(a)->kind == APY_FUNC_K && O(a)->v.fn.is_type
        && b && O(b)->kind == APY_FUNC_K && O(b)->v.fn.is_type) {
        const char *have = APY_CSTR(O(a)->v.fn.name);
        const char *want = APY_CSTR(O(b)->v.fn.name);
        return apy_from_bool(strcmp(have, want) == 0
                             || strcmp(want, "object") == 0
                             || (strcmp(have, "bool") == 0
                                 && strcmp(want, "int") == 0));
    }
    if (O(a)->kind != APY_TYPE_K)
        return apy_fail("TypeError", "issubclass() arg 1 must be a class");
    /* A BUILTIN AS THE SECOND ARGUMENT, with a class extending it as the
       first: `issubclass(S, str)` for a `class S(str)` is True in CPython
       and was `issubclass() arg 2 must be a class or tuple of classes`
       here -- a builtin kind has no type cell, so the walk below had
       nothing to compare against. The relation is the one
       `apy_class_builtin_kind` records, and the name is how it travels,
       exactly as it does for `isinstance`. */
    if (O(b)->kind != APY_TYPE_K) {
        const char *want = 0;
        if (O(b)->kind == APY_FUNC_K && O(b)->v.fn.is_type)
            want = APY_CSTR(O(b)->v.fn.name);
        else if (O(b)->kind == APY_STR_K)
            want = APY_CSTR(b);
        if (want) {
            int64_t kind = apy_class_builtin_kind(a);
            const char *have =
                kind == APY_STR_K ? "str"
                : kind == APY_BYTES_K ? "bytes"
                : kind == APY_LIST_K ? "list"
                : kind == APY_TUPLE_K ? "tuple"
                : kind == APY_DICT_K ? "dict"
                : kind == APY_SET_K ? "set" : 0;
            return apy_from_bool(have && strcmp(have, want) == 0);
        }
    }
    if (O(b)->kind != APY_TYPE_K)
        return apy_fail("TypeError",
                        "issubclass() arg 2 must be a class or tuple of "
                        "classes");
    if (apy_type_is_sub(a, b)) return apy_from_bool(1);
    /* An EXCEPTION type has no base pointer -- the builtin hierarchy is a
       table of names, not a chain of type objects, because `raise` and
       `except` match on the name and never hold a class. So the same question
       is asked again, of that table: `issubclass(KeyError, LookupError)` is
       the hierarchy `except LookupError:` already walks. */
    {
        const char *have = APY_CSTR(O(a)->v.t.name);
        const char *want = APY_CSTR(O(b)->v.t.name);
        while (have) {
            if (strcmp(have, want) == 0) return apy_from_bool(1);
            have = apy_exc_parent(have);
        }
    }
    return apy_from_bool(0);
}

/* A builtin exception NAME used as a value -- `issubclass(KeyError, ...)`,
   `except (A, B)`, `e.__class__`. Interned by `apy_type_of`, so the same name
   is the same object and `type(e) is ValueError` holds. */
/* `E(a, b, ...)` -- an exception built from MORE THAN ONE argument. `e.args`
   is the whole tuple, and the OSError family reads the first two back as
   `errno` and `strerror`. */
/* Which OSError subclass an errno names, or 0 for one with no dedicated
   class. The numbers are the Linux/POSIX values CPython maps, and they are
   written out rather than taken from <errno.h> so the answer does not change
   with the platform the compiler happens to run on. */
static const char *apy_errno_class(int code) {
    switch (code) {
    case 1:   return "PermissionError";        /* EPERM */
    case 2:   return "FileNotFoundError";      /* ENOENT */
    case 3:   return "ProcessLookupError";     /* ESRCH */
    case 4:   return "InterruptedError";       /* EINTR */
    case 10:  return "ChildProcessError";      /* ECHILD */
    case 11:  return "BlockingIOError";        /* EAGAIN */
    case 13:  return "PermissionError";        /* EACCES */
    case 17:  return "FileExistsError";        /* EEXIST */
    case 20:  return "NotADirectoryError";     /* ENOTDIR */
    case 21:  return "IsADirectoryError";      /* EISDIR */
    case 32:  return "BrokenPipeError";        /* EPIPE */
    case 103: return "ConnectionAbortedError"; /* ECONNABORTED */
    case 104: return "ConnectionResetError";   /* ECONNRESET */
    case 110: return "TimeoutError";           /* ETIMEDOUT */
    case 111: return "ConnectionRefusedError"; /* ECONNREFUSED */
    case 115: return "BlockingIOError";        /* EINPROGRESS */
    default:  return 0;
    }
}

/* USER EXCEPTION CLASSES, BY NAME.

   The hierarchy is a table of names -- that is what makes `except
   LookupError:` catch a KeyError without either being a value -- so a class
   with a body has to be findable from the name alone, at the moment an
   exception of that name is made. Nothing else knows: `apy_make_exc` is
   handed a string, and the `class` statement that wrote the body may be in
   another function entirely. */
/* REACHED THROUGH ONE FUNCTION so it can move: both halves of this table are
   on their way to IR, and a macro naming the C's own variable would leave the
   two writing to different words. */
static apy_value apy_exc_class_table_c;
APY_API apy_value apy_exc_class_slot(void) {
    return (apy_value)&apy_exc_class_table_c;
}
#define apy_exc_class_table (*(apy_value *)apy_exc_class_slot())

APY_API apy_value apy_exc_class_bind(apy_value name, apy_value cls) {
    if (!apy_exc_class_table) apy_exc_class_table = apy_dict_new(8);
    if (!apy_dict_set(apy_exc_class_table, name, cls)) return 0;
    return apy_none();
}

APY_API apy_value apy_exc_class_named_of(apy_value name) {
    if (!apy_exc_class_table) return 0;
    return apy_dict_get_or(apy_exc_class_table,
                           apy_lit((const char *)name), 0);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_exc_class_named(const char *name) {
    return apy_exc_class_named_of((apy_value)(uintptr_t)name);
}

/* THE NAME AN EXCEPTION SHOWS, which is not always the name it MATCHES.

   A bundled module's classes are spliced under mangled names and the splice
   renames them back, so `copy.Error`'s cells still carry
   `_asmpy_bundled_4_copy_Error` -- and they must, because `except copy.Error`
   compiles to that spelling and `apy_error_matches` walks by name. Renaming
   the cell would fix the display and break the catch, so the mapping happens
   HERE: everything a program READS goes through this, everything that
   MATCHES keeps using `v.e.name`. */
APY_API apy_value apy_exc_shown_of(apy_value namev) {
    const char *name = (const char *)namev;
    apy_value cls = apy_exc_class_named(name);
    return cls ? (apy_value)(uintptr_t)APY_CSTR(O(cls)->v.t.name)
               : (apy_value)(uintptr_t)name;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static const char *apy_exc_shown(const char *name) {
    return (const char *)apy_exc_shown_of((apy_value)(uintptr_t)name);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */

/* Is this type object one that CONSTRUCTS AN EXCEPTION when it is called?

   `apy_exc_type` ends in `apy_type_of`, so `ValueError` as a VALUE is an
   ordinary `APY_TYPE_K` carrying the name and an empty dict -- nothing in the
   object itself says it stands for an exception, and the name is all there is
   left to ask. `apy_exc_parent` is the same table `except` matches through,
   so this cannot drift from the hierarchy: a name that is an exception to a
   handler is one here, and a class the program wrote by subclassing one is
   registered in that table too.

   THE CHEAP TEST FIRST. Every call of every class reaches this, and the walk
   of a table of names is a strcmp loop that allocates nothing; the two class
   lookups below run only for a name that is already an exception.

   A REGISTERED CLASS ANSWERS YES, and its `__init__` still runs --
   `apy_make_excn` ends in `apy_exc_construct`, which finds the class by name
   and calls it. Sending `AppError` through `apy_instantiate` instead built an
   ordinary instance: `str(e)` read `<AppError object at 0x...>` and `raise e`
   said `exceptions must derive from BaseException, not 'AppError'` -- about
   a class whose `class` statement names ValueError as its base.

   `user == f` and not merely `user`, because a program may define an
   ordinary class sharing the name; the object registered as the exception is
   the one this is true of. */
static int apy_type_is_exc(apy_value f) {
    const char *name;
    apy_value user;
    if (O(f)->kind != APY_TYPE_K || O(f)->v.t.meta) return 0;
    name = APY_CSTR(O(f)->v.t.name);
    if (strcmp(name, "BaseException") != 0 && apy_exc_parent(name) == NULL)
        return 0;
    user = apy_exc_class_named(name);
    if (user) return user == f;
    /* A BUILTIN NAME CARRYING A BODY is a class the program wrote over the
       top of one, and it means itself. */
    return !apy_class_find(f, apy_name("__init__"))
        && !apy_class_find(f, apy_name("__new__"));
}

/* Give a fresh exception its class and, where the class writes one, run its
   `__init__` over the arguments the `raise` supplied.

   AFTER the defaults are in place, not instead of them: CPython sets `args`
   in `BaseException.__new__` and only then calls `__init__`, so a class whose
   `__init__` never calls `super().__init__` still reads back what was passed.
   One that does call it overwrites them, which is the whole reason
   `AppError(404, "missing")` can report `('404: missing',)`. */
/* `args` IS THE ADDRESS OF AN ARRAY AND NOT A POINTER TO ONE, which reads
   like a distinction without a difference and is not: every crossing between
   the C and the IR carries a machine word, and the subset has no `apy_value *`
   to declare -- so a half taking one has a type gcc calls conflicting. See
   `runtime/calling.py`, where the same question decided where a split goes. */
APY_API apy_value apy_exc_construct_of(apy_value exc, apy_value args,
                                       int64_t n) {
    apy_value *raw = (apy_value *)args;
    apy_value cls = apy_exc_class_named(O(exc)->v.e.name), init, argv[9];
    int64_t i;
    if (!cls) return exc;
    O(exc)->v.e.cls = cls;
    init = apy_class_find(cls, apy_name("__init__"));
    if (!init || O(init)->kind != APY_FUNC_K) return exc;
    if (n > 8) n = 8;
    argv[0] = exc;
    for (i = 0; i < n; i++) argv[i + 1] = raw[i];
    if (!apy_call_n(init, argv, n + 1)) return 0;
    return exc;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_exc_construct(apy_value exc, apy_value *args,
                                   int64_t n) {
    return apy_exc_construct_of(exc, (apy_value)(uintptr_t)args, n);
}

APY_API apy_value apy_make_excn(apy_value name, apy_value argv, int64_t n) {
    apy_value *raw = (apy_value *)argv;
    apy_value tuple = apy_tuple_new(n > 0 ? n : 1);
    int64_t i;
    apy_obj *o;
    for (i = 0; i < n; i++) apy_seq_push(tuple, raw[i]);
    o = apy_alloc(APY_EXC_K);
    o->v.e.subs = 0;
    o->v.e.dict = 0;
    o->v.e.cls = 0;
    o->v.e.pos = -1;
    o->v.e.name = APY_CSTR(name);
    /* PEP 3151: `OSError(errno, ...)` BUILDS THE SPECIFIC SUBCLASS. The
       errno decides which -- `OSError(2, ...)` IS a FileNotFoundError -- so
       a program can catch the precise failure without inspecting `errno`,
       which is the whole point of the hierarchy. Only for the plain name: a
       subclass written out stays what it was written as. */
    if (n >= 1 && strcmp(o->v.e.name, "OSError") == 0
            && O(raw[0])->kind == APY_INT_K) {
        const char *want = apy_errno_class((int)O(raw[0])->v.i);
        if (want) o->v.e.name = want;
    }
    o->v.e.arg = n > 0 ? raw[0] : apy_none();
    o->v.e.has_arg = n > 0;
    o->v.e.argv = tuple;
    return apy_exc_construct(V(o), raw, n);
}

APY_API apy_value apy_exc_type(apy_value name) {
    apy_obj *o;
    /* THE CLASS THE PROGRAM WROTE, when it wrote one. `except AppError:`,
       `isinstance(e, AppError)` and `super()` inside its own method must all
       reach the SAME object, or a method found through one would be missing
       through another. */
    {
        apy_value user = apy_exc_class_named(APY_CSTR(name));
        if (user) return user;
    }
    o = apy_alloc(APY_EXC_K);
    /* NOT INHERITED FROM THE UNION: a fresh exception carries no
       sub-exceptions, and reading a stale pointer here would make
       every ordinary raise look like a group. */
    o->v.e.subs = 0;
    o->v.e.dict = 0;
    o->v.e.cls = 0;
    o->v.e.pos = -1;
    o->v.e.name = APY_CSTR(name);
    o->v.e.arg = apy_none();
    o->v.e.has_arg = 0;
    o->v.e.argv = 0;
    return apy_type_of(V(o));
}

/* `vars(obj)` -- the instance's own attribute dict, which is a VIEW in
   CPython and a copy here. The difference shows only when a program writes
   through the result, which the suite does not; a copy is honest about what
   this runtime can offer, where a silently-detached view would not be. */
APY_API apy_value apy_vars(apy_value obj) {
    /* A CLASS has a `__dict__` too, holding the names its body bound --
       methods and class attributes -- and `"x" in vars(C)` is how a program
       asks whether the class itself defines one. */
    if (O(obj)->kind == APY_TYPE_K) return apy_copy(O(obj)->v.t.dict);
    if (O(obj)->kind != APY_INST_K)
        return apy_fail2("TypeError",
                         "vars() argument must have __dict__ attribute%s%s",
                         "", "");
    return apy_copy(O(obj)->v.o.dict);
}

/* `iter(f, sentinel)` -- the CALLABLE form, which calls `f` until it answers
   `sentinel`.

   A CURSOR IN THE `APY_IT_CALL` MODE, which calls on every step. The calls
   USED TO ALL HAPPEN HERE, into a list, with a guard at a million to stop a
   callable that never answers the sentinel from running forever. Two things
   were wrong with that and both are what a program sees: the calls happened
   at CONSTRUCTION, so `iter(f, 4)` had already called `f` four times before
   the first `next()`, and a callable that never answers the sentinel made
   `for v in iter(f, s): break` -- which CPython leaves after ONE call -- run
   a million times and then stop short of what it was asked for.

   NAMED AFTER THE FORM: CPython calls this a `callable_iterator`, and
   nothing about the mode or the sentinel says so. */
APY_API apy_value apy_iter_until(apy_value fn, apy_value sentinel) {
    apy_value out;
    /* REFUSED WHEN IT IS MADE, not when it is walked. CPython checks the
       first argument here and says so in those words; leaving it to the walk
       reported `'int' object is not callable` from the first `next()`, which
       names the wrong moment and the wrong thing. */
    if (!apy_truth(apy_callable(fn)))
        return apy_fail2("TypeError", "iter(v, w): v must be callable%s%s",
                         "", "");
    out = apy_cursor(sentinel, fn, APY_IT_CALL, 0);
    if (out && O(out)->kind == APY_ITER_K)
        O(out)->v.it.named = APY_IT_CALLABLE;
    return out;
}

/* `isinstance(v, T)` where T is named by a string the frontend supplies. A
   real type object would be better and does not exist yet; the name is enough
   to answer every question the suite asks, including that `True` is an `int`
   -- bool is a SUBCLASS of int in Python, so this is not simply a name
   comparison. */
/* The second argument is EITHER a str naming a built-in kind OR a real type
   object, and the frontend picks per call site: a name it can see is a class
   travels as the class, anything else as its text. Two entry points would be
   tidier and would put the choice in the frontend twice -- once to pick the
   symbol and once to build the argument -- so the parameter carries it.

   Comparing NAMES would have been enough right up until user classes existed,
   and then two classes both called `Node` in one program would be instances
   of each other. */
"""
