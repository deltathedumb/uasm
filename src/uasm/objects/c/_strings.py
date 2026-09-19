"""The object runtime, in C: the string methods.

ONE PART OF ONE TRANSLATION UNIT. `c/__init__.py` concatenates
these in order and the result is the file it always was, so a
definition here may rely on anything in an earlier part and
nothing in a later one. Sections, in order:
  * string methods
  * case
  * predicates
  * strip
  * split and join
  * padding
"""

C = r"""/* --- string methods ----------------------------------------------------- */
/* Pure functions over the str cell: nothing here mutates, because a Python str
   cannot be mutated, so every one of them builds a new cell.

   TWO DIVERGENCES, BOTH STATED RATHER THAN LEFT TO BE FOUND:

   * INDICES AND LENGTHS ARE BYTES, not characters. That is the limitation the
     top of this file records for indexing and slicing, and every method here
     inherits it: `'café'.find('é')` is 3 and CPython says 3 as well only
     because the accent is the last character. A method that returns a
     POSITION is wrong for any string with a multi-byte character before that
     position. It is consistent -- a position from `find` can be fed back to
     the slicer -- and it is not what CPython reports.

   * CASE AND CLASSIFICATION ARE ASCII. `'ß'.upper()` is 'SS' in CPython and
     'ß' here, and `'²'.isdigit()` is True there and False here. Doing better
     needs Unicode's case-mapping and category tables, which are 30k of data
     this runtime does not carry. Every ASCII answer is exact.

   The bounds rules are Python's and they are not C's: a negative index counts
   from the end, `end` clamps down to the length, and `start` DOES NOT clamp
   -- `'abc'.find('', 9)` is -1 while `'abc'.find('', 3)` is 3, and an upper
   clamp on `start` would answer 3 to both. */
/* THE EXPORTED HALF, which `runtime/list_cell.py` replaces. The `static`
   below keeps the name its callers use and the cast they do not have to
   write; this body is what the runtime uses when nothing is ported. */
/* Declared here because the delegate is defined below its first use. */
static int apy_str_self(const char *name, apy_value v);
APY_API int64_t apy_str_self_of(apy_value name, apy_value v) {
    if (O(v)->kind == APY_STR_K || O(v)->kind == APY_BYTES_K) return 1;
    /* AN INSTANCE OF A CLASS EXTENDING str OR bytes IS ONE HERE TOO. The
       written `s.upper()` reaches this having been unwrapped by
       `apy_method_self`; `str.upper(s)` -- the unbound spelling -- does not,
       and reported that a str subclass had no attribute `upper`. */
    if (O(v)->kind == APY_INST_K && O(v)->v.o.held
            && (O(O(v)->v.o.held)->kind == APY_STR_K
                || O(O(v)->v.o.held)->kind == APY_BYTES_K))
        return 1;
    apy_fail2("AttributeError", "'%s' object has no attribute '%s'",
              apy_kind_name(v), (const char *)name);
    return 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above is what stands in when nothing is
   ported. The cast to a machine word happens here, once. */
static int apy_str_self(const char *name, apy_value v) {
    return (int)apy_str_self_of((apy_value)(uintptr_t)name, v);
}

/* `find() argument 1 must be str, not int`. TWO ODDITIES OF THIS FAMILY are
   reproduced rather than regularised, because the suite is generated from
   CPython and both are visible: the kind is written WITHOUT quotes, and
   NoneType is written as `None` -- while `startswith`, forty lines down, says
   `not NoneType` for the very same value. `argno` of 0 drops the number,
   which is how `removeprefix` words it. */
APY_API apy_value apy_arg_must_be_str_of(apy_value methv,
                                        int64_t argno, apy_value v) {
    const char *meth = (const char *)methv;
    char buf[160];
    const char *k = O(v)->kind == APY_NONE_K ? "None" : apy_kind_name(v);
    if (argno)
        snprintf(buf, sizeof buf, "%s() argument %d must be str, not %s",
                 meth, argno, k);
    else
        snprintf(buf, sizeof buf, "%s() argument must be str, not %s", meth, k);
    return apy_fail("TypeError", buf);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_arg_must_be_str(const char *meth, int argno,
                                     apy_value v) {
    return apy_arg_must_be_str_of((apy_value)(uintptr_t)meth,
                                  (int64_t)argno, v);
}

APY_API int64_t apy_str_other_of(apy_value methv, int64_t argno,
                                apy_value v) {
    const char *meth = (const char *)methv;
    /* BYTES TOO, for the same reason the receiver may be: `b.replace(b"l",
       b"L")` hands bytes to an operation that reads a pointer and a length.
       Mixing the two is what CPython rejects, and the RECEIVER is what
       decides -- see `apy_str_like`. */
    if (O(v)->kind == APY_STR_K || O(v)->kind == APY_BYTES_K) return 1;
    apy_arg_must_be_str(meth, argno, v);
    return 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_str_other(const char *meth, int argno, apy_value v) {
    return (int)apy_str_other_of((apy_value)(uintptr_t)meth,
                                 (int64_t)argno, v);
}

APY_API apy_value apy_mview_bytes(apy_value v);

/* AN INSTANCE OF A CLASS EXTENDING str OR bytes IS ONE, for anything that
   reads the buffer. `class S(str)` makes something CPython's `find`, `join`,
   `replace`, `split` and `in` all take without a second thought -- they read
   the C-level layout, which a subclass has. Every one of them refused it
   here, by kind.

   UNGATED, unlike `apy_as_builtin`: a class that writes `__str__` or `__eq__`
   changes none of this, because `str.find` never asks. `"abc".find(s)` is 1
   for an `s` whose `__str__` answers something else entirely.

   Identity for everything else, so this drops in FRONT of an existing test
   rather than beside it. */
static apy_value apy_text_like(apy_value v) {
    apy_value held = O(v)->kind == APY_INST_K ? O(v)->v.o.held : 0;
    if (held && (O(held)->kind == APY_STR_K || O(held)->kind == APY_BYTES_K))
        return held;
    return v;
}

/* A TEXT ARGUMENT AGAINST ITS RECEIVER: the value to use, or 0 having
   refused. The gate above asks only "is this str or bytes", which let either
   kind through for either receiver -- so `"abc".find(b"a")` answered 0 and
   `b"abc".find("a")` answered 0, two WRONG ANSWERS where CPython refuses and
   the interpreter already did.

   A MEMORYVIEW STANDS FOR THE BYTES IT VIEWS, but only for a bytes receiver:
   `b"xabcx".find(memoryview(b"abc"))` is 1 in CPython and
   `"abc".find(memoryview(b"a"))` is a TypeError. That is the whole of what
   "bytes-like" means here, and it is why the conversion belongs with the
   check rather than beside it.

   TWO WORDINGS FOR A BYTES RECEIVER, which is CPython's own split: the
   searches say `argument should be integer or bytes-like object` because an
   INTEGER is a legal needle for them, and everything else says `a bytes-like
   object is required`. */
static apy_value apy_text_arg(const char *meth, int argno, int indexy,
                              apy_value self, apy_value v) {
    int want_bytes = O(self)->kind == APY_BYTES_K;
    char buf[160];
    v = apy_text_like(v);
    if (want_bytes && O(v)->kind == APY_MVIEW_K) v = apy_mview_bytes(v);
    if (O(v)->kind == (want_bytes ? APY_BYTES_K : APY_STR_K)) return v;
    /* AN INTEGER IS A LEGAL NEEDLE FOR THE SEARCHES, which is what the
       wording below says and what this refused anyway: `b"abc".index(98)` is
       1 in Python. One byte, so anything outside a byte's range is a
       ValueError and not a wrong answer. */
    if (want_bytes && indexy && apy_is_int_like(v)) {
        int64_t byte = O(v)->v.i;
        char one[1];
        if (apy_is_big(v) || byte < 0 || byte > 255)
            { apy_fail("ValueError", "byte must be in range(0, 256)");
              return 0; }
        one[0] = (char)byte;
        return apy_bytes_copy(one, 1);
    }
    if (!want_bytes) {
        apy_arg_must_be_str(meth, argno, v);
        return 0;
    }
    if (indexy)
        snprintf(buf, sizeof buf, "argument should be integer or bytes-like "
                 "object, not '%s'", apy_kind_name(v));
    else
        snprintf(buf, sizeof buf, "a bytes-like object is required, not '%s'",
                 apy_kind_name(v));
    apy_fail("TypeError", buf);
    return 0;
}

APY_API int64_t apy_int_arg_of(apy_value v, apy_value out) {
    if (!apy_is_int_like(v)) {
        apy_fail2("TypeError",
                  "'%s' object cannot be interpreted as an integer%s",
                  apy_kind_name(v), "");
        return 0;
    }
    if (apy_is_big(v)) {
        apy_fail("OverflowError",
                 "Python int too large to convert to C ssize_t");
        return 0;
    }
    *(int64_t *)out = O(v)->v.i;
    return 1;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_int_arg(apy_value v, int64_t *out) {
    return (int)apy_int_arg_of(v, (apy_value)(uintptr_t)out);
}


APY_API int64_t apy_slice_arg_of(apy_value v, apy_value out) {
    if (O(v)->kind == APY_NONE_K) return 1;
    /* A BOUND IS A SLICE INDEX AND SAYS SO. `"abc".find("b", 1.0)` is
       `slice indices must be integers or None or have an __index__ method`
       in CPython, not the general `'float' object cannot be interpreted as
       an integer` that `apy_int_arg_of` gives every other integer argument.
       AN INSTANCE FALLS THROUGH, since one with `__index__` is a valid
       bound and the conversion below is what asks. */
    if (!apy_is_int_like(v) && O(v)->kind != APY_INST_K) {
        apy_fail("TypeError", "slice indices must be integers or None or "
                              "have an __index__ method");
        return 0;
    }
    if (apy_is_big(v)) {
        *(int64_t *)out = O(v)->v.big.neg
            ? -((int64_t)1 << 62) : ((int64_t)1 << 62);
        return 1;
    }
    return apy_int_arg(v, (int64_t *)out);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_slice_arg(apy_value v, int64_t *out) {
    return (int)apy_slice_arg_of(v, (apy_value)(uintptr_t)out);
}

APY_API void apy_clamp_range_of(int64_t n, apy_value lo,
                                apy_value hi) {
    int64_t *a = (int64_t *)lo, *b = (int64_t *)hi;
    if (*a < 0) { *a += n; if (*a < 0) *a = 0; }
    if (*b < 0) { *b += n; if (*b < 0) *b = 0; }
    if (*b > n) *b = n;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static void apy_clamp_range(int64_t n, int64_t *lo, int64_t *hi) {
    apy_clamp_range_of(n, (apy_value)(uintptr_t)lo,
                       (apy_value)(uintptr_t)hi);
}

/* The first occurrence of `sub` in `s[lo:hi]`, as an absolute index, or -1.
   An EMPTY needle matches at `lo` -- but only if `lo` is inside the window,
   which is the whole reason this takes `hi` rather than assuming the end. */
APY_API int64_t apy_find_at(apy_value s, apy_value sub,
                           int64_t lo, int64_t hi) {
    int64_t m = O(sub)->v.s.n, i;
    if (m == 0) return lo <= hi ? lo : -1;
    for (i = lo; i + m <= hi; i++)
        if (memcmp(O(s)->v.s.p + i, O(sub)->v.s.p, (size_t)m) == 0) return i;
    return -1;
}

APY_API int64_t apy_rfind_at(apy_value s, apy_value sub,
                            int64_t lo, int64_t hi) {
    int64_t m = O(sub)->v.s.n, i;
    if (m == 0) return lo <= hi ? hi : -1;
    for (i = hi - m; i >= lo; i--)
        if (memcmp(O(s)->v.s.p + i, O(sub)->v.s.p, (size_t)m) == 0) return i;
    return -1;
}

APY_API apy_value apy_str_slice_of(apy_value s, int64_t lo,
                                  int64_t hi) {
    if (hi < lo) hi = lo;
    return apy_str_copy(O(s)->v.s.p + lo, hi - lo);
}

/* find / rfind / index / rindex, all four from one place. `want_index` picks
   the -1-on-failure form from the raise-on-failure one; that is the only
   difference between `find` and `index`, and CPython's message for the second
   is `substring not found` with no mention of what was looked for. */
/* The byte offset of character `ci`, and the character index of byte offset
   `bo`. Every position a str method takes or answers is in CHARACTERS -- the
   search itself works in bytes, because that is what `memcmp` compares -- so
   these two are the boundary between the two counts. For an all-ASCII string
   both are the identity, which is why the distinction stayed invisible. */
static int64_t apy_char_to_byte(apy_value s, int64_t ci) {
    const unsigned char *p = (const unsigned char *)O(s)->v.s.p;
    int64_t bytes = O(s)->v.s.n, at = 0, seen = 0, used;
    while (seen < ci && at < bytes) {
        apy_utf8_at(p, bytes, at, &used);
        at += used;
        seen++;
    }
    return at;
}

static int64_t apy_byte_to_char(apy_value s, int64_t bo) {
    const unsigned char *p = (const unsigned char *)O(s)->v.s.p;
    int64_t bytes = O(s)->v.s.n, at = 0, seen = 0, used;
    while (at < bo && at < bytes) {
        apy_utf8_at(p, bytes, at, &used);
        at += used;
        seen++;
    }
    return seen;
}

static apy_value apy_str_search(apy_value s, apy_value sub, apy_value start,
                                apy_value end, int from_right, int want_index) {
    int64_t lo = 0, hi, at;
    const char *meth = want_index ? (from_right ? "rindex" : "index")
                                  : (from_right ? "rfind" : "find");
    if (!apy_str_self(meth, s)) return 0;
    sub = apy_text_arg(meth, 1, 1, s, sub);
    if (!sub) return 0;
    hi = apy_str_chars(s);
    if (start && !apy_slice_arg(start, &lo)) return 0;
    if (end && !apy_slice_arg(end, &hi)) return 0;
    /* THE BOUNDS ARRIVE IN CHARACTERS and the search runs in bytes, so they
       are clamped against the character count and then converted. Answering a
       byte offset made `"héllo".find("ll")` say 3 where CPython says 2.
    */
    apy_clamp_range(apy_str_chars(s), &lo, &hi);
    lo = apy_char_to_byte(s, lo);
    hi = apy_char_to_byte(s, hi);
    at = from_right ? apy_rfind_at(s, sub, lo, hi) : apy_find_at(s, sub, lo, hi);
    /* A BYTES RECEIVER HAS NO SUBSTRINGS: CPython says `subsection not
       found` for one, which is the same distinction the rest of this
       function now makes about units. */
    if (at < 0 && want_index)
        return apy_fail("ValueError", O(s)->kind == APY_STR_K
                        ? "substring not found" : "subsection not found");
    return apy_from_int(at < 0 ? at : apy_byte_to_char(s, at));
}

APY_API apy_value apy_str_find(apy_value s, apy_value sub) {
    return apy_str_search(s, sub, 0, 0, 0, 0);
}
APY_API apy_value apy_str_find2(apy_value s, apy_value sub, apy_value start) {
    return apy_str_search(s, sub, start, 0, 0, 0);
}
APY_API apy_value apy_str_find3(apy_value s, apy_value sub, apy_value start,
                                apy_value end) {
    return apy_str_search(s, sub, start, end, 0, 0);
}
APY_API apy_value apy_str_rfind(apy_value s, apy_value sub) {
    return apy_str_search(s, sub, 0, 0, 1, 0);
}
APY_API apy_value apy_str_rfind2(apy_value s, apy_value sub, apy_value start) {
    return apy_str_search(s, sub, start, 0, 1, 0);
}
APY_API apy_value apy_str_rfind3(apy_value s, apy_value sub, apy_value start,
                                 apy_value end) {
    return apy_str_search(s, sub, start, end, 1, 0);
}
APY_API apy_value apy_str_rindex(apy_value s, apy_value sub) {
    return apy_str_search(s, sub, 0, 0, 1, 1);
}
/* `index` AND `rindex` TAKE THE SAME BOUNDS `find` AND `rfind` DO, and did
   not: `"abcabc".index("c", 3)` fell off the arity table, reached the generic
   attribute lookup, and reported about a one-argument method. The search
   itself already took the window -- only the entry points were missing. */
APY_API apy_value apy_str_index2(apy_value s, apy_value sub, apy_value start) {
    return apy_str_search(s, sub, start, 0, 0, 1);
}
APY_API apy_value apy_str_index3(apy_value s, apy_value sub, apy_value start,
                                 apy_value end) {
    return apy_str_search(s, sub, start, end, 0, 1);
}
APY_API apy_value apy_str_rindex2(apy_value s, apy_value sub,
                                  apy_value start) {
    return apy_str_search(s, sub, start, 0, 1, 1);
}
APY_API apy_value apy_str_rindex3(apy_value s, apy_value sub, apy_value start,
                                  apy_value end) {
    return apy_str_search(s, sub, start, end, 1, 1);
}

/* Reached from `apy_index_of` and `apy_count_of` when the receiver is a str,
   so that `'abcabc'.index('bc')` looks for a SUBSTRING rather than for an
   element equal to it -- the sequence versions would answer only for
   single-character needles and would silently do so. */
static apy_value apy_str_index_of(apy_value s, apy_value sub) {
    return apy_str_search(s, sub, 0, 0, 0, 1);
}

/* NON-OVERLAPPING, which is what makes `'aaaa'.count('aa')` 2 and not 3, and
   an empty needle counts the gaps: `'abc'.count('')` is 4. */
APY_API apy_value apy_str_count_in_of(apy_value s, apy_value sub,
                                     apy_value start, apy_value end) {
    int64_t lo = 0, hi, m, i, n, hits = 0;
    /* BYTES COUNTS OCTETS AND STR COUNTS CHARACTERS. One function serves
       both receivers, so the unit is a fact about `s` rather than about the
       code: `"éàbcé".count("é", 1, 5)` names characters, and answering it
       over bytes found nothing. */
    int wide = O(s)->kind == APY_STR_K;
    sub = apy_text_arg("count", 1, 1, s, sub);
    if (!sub) return 0;
    n = wide ? apy_str_chars(s) : O(s)->v.s.n;
    hi = n;
    if (start && !apy_slice_arg(start, &lo)) return 0;
    if (end && !apy_slice_arg(end, &hi)) return 0;
    apy_clamp_range(n, &lo, &hi);
    m = O(sub)->v.s.n;
    /* AN EMPTY NEEDLE COUNTS POSITIONS, and a str's positions are BETWEEN
       CHARACTERS: `"café".count("")` is 5 and not 6. Counting byte
       boundaries put a position inside the two bytes of the `é`, which is a
       place Python says nothing can go. */
    if (m == 0) return apy_from_int(hi >= lo ? hi - lo + 1 : 0);
    /* THE WINDOW CROSSES INTO BYTES HERE, once. A non-empty needle can only
       match at a character boundary -- UTF-8 is prefix-free there -- so the
       walk itself needs no conversion. */
    if (wide) {
        lo = apy_char_to_byte(s, lo);
        hi = apy_char_to_byte(s, hi);
    }
    for (i = lo; i + m <= hi; ) {
        if (memcmp(O(s)->v.s.p + i, O(sub)->v.s.p, (size_t)m) == 0) {
            hits++;
            i += m;
        } else i++;
    }
    return apy_from_int(hits);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_str_count_in(apy_value s, apy_value sub,
                                  apy_value start, apy_value end) {
    return apy_str_count_in_of(s, sub, start, end);
}

APY_API apy_value apy_str_count2(apy_value s, apy_value sub, apy_value start) {
    if (!apy_str_self("count", s)) return 0;
    return apy_str_count_in(s, sub, start, 0);
}
APY_API apy_value apy_str_count3(apy_value s, apy_value sub, apy_value start,
                                 apy_value end) {
    if (!apy_str_self("count", s)) return 0;
    return apy_str_count_in(s, sub, start, end);
}

/* --- case ---------------------------------------------------------------
   ASCII rules, as the section header says. A byte outside ASCII is copied
   unchanged, which is at least stable and never corrupts UTF-8: every byte of
   a multi-byte sequence has its high bit set, so none of them can be mistaken
   for a letter to map. */
/* PROMOTED FROM `static` so the subset can name them. The five ASCII
   predicates are what every case transform and every `str.isalpha`
   family member rests on, and `runtime/str_cell.py` records the case
   transforms as blocked -- this is the half of that blockage which is
   not the Unicode table.

   `int64_t` RATHER THAN `unsigned char` AND `int`, because those are
   not types `signatures()` can describe: it knows `apy_value`,
   `int64_t`, `double` and `void`, and a runtime symbol the frontend
   cannot describe is one it cannot call. Every caller passes an
   `unsigned char`, which widens to `int64_t` on its own. */
APY_API int64_t apy_c_lower(int64_t c) { return c >= 'a' && c <= 'z'; }
APY_API int64_t apy_c_upper(int64_t c) { return c >= 'A' && c <= 'Z'; }
APY_API int64_t apy_c_alpha(int64_t c) { return apy_c_lower(c) || apy_c_upper(c); }
APY_API int64_t apy_c_digit(int64_t c) { return c >= '0' && c <= '9'; }
APY_API int64_t apy_c_space(int64_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f'
        || c == '\v';
}

enum { APY_UPPER, APY_LOWER, APY_TITLE, APY_CAPITAL, APY_SWAP, APY_FOLD };

/* THE OUTPUT CAN BE LONGER THAN THE INPUT, which is why the body sits below
   the case tables rather than here: `ß` uppercases to `SS` and `ﬃ` to `FFI`,
   and saying so needs the generated data. */
static apy_value apy_str_case(apy_value s, int mode);

APY_API apy_value apy_str_upper(apy_value s) {
    if (!apy_str_self("upper", s)) return 0;
    return apy_str_case(s, APY_UPPER);
}
APY_API apy_value apy_str_lower(apy_value s) {
    if (!apy_str_self("lower", s)) return 0;
    return apy_str_case(s, APY_LOWER);
}
APY_API apy_value apy_str_title(apy_value s) {
    if (!apy_str_self("title", s)) return 0;
    return apy_str_case(s, APY_TITLE);
}
APY_API apy_value apy_str_capitalize(apy_value s) {
    if (!apy_str_self("capitalize", s)) return 0;
    return apy_str_case(s, APY_CAPITAL);
}
APY_API apy_value apy_str_swapcase(apy_value s) {
    if (!apy_str_self("swapcase", s)) return 0;
    return apy_str_case(s, APY_SWAP);
}
/* `casefold` is aggressive lowercasing for caseless matching, and for ASCII
   it IS lowercasing. It differs from `lower` on the pair it exists for --
   'ß' folds to 'ss' where lowering leaves it alone -- so the two are not the
   same mode even though they agree on everything a plain program prints. */
APY_API apy_value apy_str_casefold(apy_value s) {
    if (!apy_str_self("casefold", s)) return 0;
    return apy_str_case(s, APY_FOLD);
}

/* --- predicates ---------------------------------------------------------
   All of them are False for the EMPTY string except `isascii`, which is True
   -- that is not an accident of the loop, it is Python's rule, and writing
   the loop so that "no character failed" means True would get every one of
   them wrong for ''. */
enum { APY_ISALPHA, APY_ISDIGIT, APY_ISALNUM, APY_ISSPACE, APY_ISLOWER,
       APY_ISUPPER, APY_ISTITLE, APY_ISPRINTABLE, APY_ISIDENT, APY_ISASCII,
       APY_ISDECIMAL, APY_ISNUMERIC };

/* The UTF-8 decoder lives with the codecs, far below the predicates that
   walk a string by code point. */
static int64_t apy_utf8_step(const unsigned char *p, int64_t n, int64_t i,
                             uint32_t *out);
/* And its inverse, which the case transforms need: a mapping can hand back a
   code point that was never in the input. */
static int apy_utf8_put(char *out, uint32_t cp);

/* @UNICODE_TABLE@ */

/* @UNICASE_TABLE@ */

/* --- case, the whole of it ----------------------------------------------

   THE SIX TRANSFORMS ARE ONE WALK over code points, differing in which
   mapping each character takes and in what the walk remembers between them.
   Written here rather than beside the other string methods because it stands
   on the generated tables above, and a forward declaration is what the
   callers up there see. The six mode names are the enum declared with
   them. */

/* Which mapping one character takes under one transform.

   `capitalize` AND `title` RAISE TO TITLECASE AND NOT TO UPPERCASE, which is
   a difference exactly one class of character can show: `ß` capitalizes to
   `Ss` where it uppercases to `SS`, and `ǆ` to `ǅ` where it uppercases to
   `Ǆ`. Both are single characters with a titlecase form of their own. */
static int apy_case_mode_for(int mode, int64_t at, int prev_cased) {
    if (mode == APY_UPPER) return 0;
    if (mode == APY_LOWER) return 1;
    if (mode == APY_FOLD) return 3;
    if (mode == APY_CAPITAL) return at == 0 ? 2 : 1;
    if (mode == APY_TITLE) return prev_cased ? 1 : 2;
    return -1;                          /* swapcase decides per character */
}

/* A GREEK CAPITAL SIGMA AT THE END OF A WORD LOWERCASES TO `ς` AND NOT `σ`.

   The rule is the only context-sensitive thing in Python's case mapping, and
   it is not optional: `"ΟΣ".lower()` is `"ος"` and `"ΣΟ".lower()` is `"σο"`,
   so a sigma-blind implementation is wrong about one of them whichever form
   it picks.

   FINAL MEANS: preceded by a cased character, with only case-ignorable
   characters in between, and NOT followed by the same. The full stop in
   `"ΟΣ."` is case-ignorable and does not stop the sigma being final; the
   space in `"Ο Σ"` is not, and does. */
static int apy_case_final_sigma(const unsigned char *p, int64_t n, int64_t at,
                                int64_t after) {
    int64_t i = at;
    uint32_t cp;
    int64_t used;
    int before = 0;
    /* BACKWARDS over what came before, which is a re-walk from the start:
       UTF-8 is scanned forwards here, and the alternative is remembering the
       last two states through the main loop -- state this rule is the only
       reader of. */
    for (i = 0; i < at; ) {
        used = apy_utf8_step(p, n, i, &cp);
        if (used <= 0) break;
        i += used;
        if (i > at) break;
        {
            unsigned f = apy_ucase_flags(cp);
            if (f & APY_CASED) before = 1;
            else if (!(f & APY_CASE_IGNORABLE)) before = 0;
        }
    }
    if (!before) return 0;
    for (i = after; i < n; ) {
        used = apy_utf8_step(p, n, i, &cp);
        if (used <= 0) break;
        i += used;
        {
            unsigned f = apy_ucase_flags(cp);
            if (f & APY_CASED) return 0;
            if (!(f & APY_CASE_IGNORABLE)) return 1;
        }
    }
    return 1;
}

/* Written below, beside the predicates that share them: one "character" of a
   receiver, and its class. Declared here because the case transforms sit
   above the character table they both rest on. */
static int64_t apy_text_step(int wide, const unsigned char *p, int64_t n,
                             int64_t i, uint32_t *cp);
static unsigned apy_text_class(int wide, uint32_t cp);

static apy_value apy_str_case(apy_value s, int mode) {
    const unsigned char *p = (const unsigned char *)O(s)->v.s.p;
    int64_t n = O(s)->v.s.n, i = 0, out_n = 0;
    int wide = O(s)->kind == APY_STR_K;
    /* THREE CODE POINTS IS THE WIDEST ANY MAPPING GROWS TO -- `ﬃ` becomes
       `FFI` -- and a code point is at most four UTF-8 bytes, so twelve bytes
       of output per byte of input cannot be exceeded. */
    char *buf = (char *)malloc((size_t)n * 12 + 1);
    int prev_cased = 0;
    if (!buf) { fputs("uasm: out of memory\n", stderr); exit(1); }
    while (i < n) {
        uint32_t cp, got[4];
        int64_t used = apy_text_step(wide, p, n, i, &cp);
        int which, k, count;
        unsigned flags;
        if (used <= 0) { buf[out_n++] = (char)p[i++]; continue; }
        /* ASCII-ONLY FOR BYTES. A byte above 0x7F has no case and is not
           cased, so it neither maps nor carries a word across -- which is
           why `b"\xc3\xa9ab".title()` is `b"\xc3\xa9Ab"`: the two bytes
           are uncased and the `a` after them starts a word. */
        if (!wide && cp >= 0x80) {
            buf[out_n++] = (char)cp;
            prev_cased = 0;
            i += used;
            continue;
        }
        flags = apy_ucase_flags(cp);
        which = apy_case_mode_for(mode, i, prev_cased);
        if (which < 0) {
            /* SWAPCASE ASKS WHAT THE CHARACTER IS, not what it maps to: a
               LOWERCASE one is raised and an UPPERCASE one is lowered, and
               everything else -- including a TITLECASE character like `ǅ`,
               which is neither -- is copied. Asking the mapping instead
               raised `ǅ` to `Ǆ`, where Python leaves it alone. */
            unsigned m = cp < 0x80
                ? ((cp >= 'a' && cp <= 'z') ? APY_UC_LOWER
                   : (cp >= 'A' && cp <= 'Z') ? APY_UC_UPPER : 0u)
                : apy_uc_mask(cp);
            if (m & APY_UC_LOWER) which = 0;
            else if (m & APY_UC_UPPER) which = 1;
            else {
                out_n += apy_utf8_put(buf + out_n, cp);
                prev_cased = (flags & APY_CASED) != 0;
                i += used;
                continue;
            }
        }
        count = apy_ucase_map(cp, which, got);
        /* U+03A3 IS THE ONE CHARACTER WHOSE ANSWER DEPENDS ON ITS
           NEIGHBOURS. `casefold` is exempt: it maps every sigma to `σ`, which
           is the point of a fold. */
        if (cp == 0x3A3 && which == 1
                && apy_case_final_sigma(p, n, i, i + used))
            got[0] = 0x3C2;
        for (k = 0; k < count; k++)
            out_n += apy_utf8_put(buf + out_n, got[k]);
        /* WHAT THE NEXT CHARACTER SEES. `title` needs to know whether a word
           is running, and an uncased character ends one -- which is why
           `"don't"` titles to `"Don'T"`: the apostrophe is uncased. */
        prev_cased = (flags & APY_CASED) != 0;
        i += used;
    }
    buf[out_n] = '\0';
    return apy_str_take(buf, out_n);
}

/* The classes ONE CODE POINT belongs to. ASCII is decided here -- it is the
   dense half of the range and the table starts past it -- and everything
   above is a lookup in the generated runs. */
APY_API int64_t apy_char_class_of(int64_t cp) {
    unsigned m = 0;
    if (cp >= 0x80) return (int64_t)apy_uc_mask((uint32_t)cp);
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z'))
        m |= APY_UC_ALPHA | APY_UC_XIDSTART | APY_UC_XIDCONT;
    if (cp >= '0' && cp <= '9')
        m |= APY_UC_DECIMAL | APY_UC_DIGIT | APY_UC_NUMERIC | APY_UC_XIDCONT;
    if (cp >= 'a' && cp <= 'z') m |= APY_UC_LOWER;
    if (cp >= 'A' && cp <= 'Z') m |= APY_UC_UPPER;
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\v' || cp == '\f'
        || cp == '\r') m |= APY_UC_SPACE;
    if (cp >= 0x20 && cp < 0x7f) m |= APY_UC_PRINTABLE;
    if (cp == '_') m |= APY_UC_XIDSTART | APY_UC_XIDCONT;
    return (int64_t)m;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now, and
   the exported half above stands in when nothing is ported. The exported
   half takes and returns a plain word because the subset has one integer
   width and `uint32_t` is not it. */
static unsigned apy_char_class(uint32_t cp) {
    return (unsigned)apy_char_class_of((int64_t)cp);
}

/* IS THIS CHARACTER PRINTABLE, for anyone outside this file.
   `repr` needs the answer and sits in an EARLIER part of the source, so it
   forward-declares this rather than the table it rests on -- which stays
   private to the part that generates it. The `' '` is the same exception
   `str.isprintable` makes: a space is printable and no other whitespace
   is. */
APY_API int64_t apy_cp_printable_of(int64_t cp) {
    return cp == ' ' || (apy_char_class_of(cp) & APY_UC_PRINTABLE) != 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate. */
static int apy_cp_printable(uint32_t cp) {
    return (int)apy_cp_printable_of((int64_t)cp);
}

/* ONE "CHARACTER" OF A RECEIVER -- which for BYTES is one BYTE and for a str
   is one code point -- and its class, which for bytes stops at ASCII.

   THE PREDICATES AND THE CASE TRANSFORMS SHARE ONE BODY between the two kinds
   and that body took the str view of both questions: it decoded
   `b"\xc3\xa9"` as one character and asked the Unicode table about it, so
   `b"\xc3\xa9".isalpha()` was True and `.upper()` answered `b"\xc3\x89"` --
   changing bytes the program never spelled as a character. Python's bytes
   methods are ASCII-ONLY: a byte above 0x7F is not a letter, has no case, is
   not whitespace, and is copied. */
static int64_t apy_text_step(int wide, const unsigned char *p, int64_t n,
                             int64_t i, uint32_t *cp) {
    if (!wide) { *cp = p[i]; return 1; }
    return apy_utf8_step(p, n, i, cp);
}
static unsigned apy_text_class(int wide, uint32_t cp) {
    if (!wide && cp >= 0x80) return 0u;
    return apy_char_class(cp);
}

static apy_value apy_str_is(apy_value s, int which) {
    int64_t n = O(s)->v.s.n, i;
    const unsigned char *p = (const unsigned char *)O(s)->v.s.p;
    int wide = O(s)->kind == APY_STR_K;
    int cased = 0, prev_cased = 0, ok = 1, first = 1, any = 0;
    if (which == APY_ISASCII) {
        for (i = 0; i < n; i++)
            if (p[i] > 0x7f) return apy_from_bool(0);
        return apy_from_bool(1);
    }
    if (which == APY_ISPRINTABLE && n == 0) return apy_from_bool(1);
    if (n == 0) return apy_from_bool(0);
    /* BY CODE POINT, not by byte. Every predicate below asks a question about
       a CHARACTER, and a multi-byte one walked as bytes was asked about its
       continuation bytes -- which belong to no class, so every non-ASCII
       string answered False. */
    for (i = 0; i < n; ) {
        uint32_t cp;
        int64_t used = apy_text_step(wide, p, n, i, &cp);
        unsigned m;
        if (!used) { cp = 0xFFFD; used = 1; }
        i += used;
        any = 1;
        m = apy_text_class(wide, cp);
        switch (which) {
        case APY_ISALPHA: if (!(m & APY_UC_ALPHA)) ok = 0; break;
        case APY_ISDIGIT: if (!(m & APY_UC_DIGIT)) ok = 0; break;
        case APY_ISDECIMAL: if (!(m & APY_UC_DECIMAL)) ok = 0; break;
        case APY_ISNUMERIC: if (!(m & APY_UC_NUMERIC)) ok = 0; break;
        case APY_ISALNUM:
            if (!(m & (APY_UC_ALPHA | APY_UC_NUMERIC | APY_UC_DIGIT
                       | APY_UC_DECIMAL))) ok = 0;
            break;
        case APY_ISSPACE: if (!(m & APY_UC_SPACE)) ok = 0; break;
        case APY_ISPRINTABLE:
            /* A SPACE IS PRINTABLE and no other whitespace is, which is the
               one place this differs from "not a control character". */
            if (cp != ' ' && !(m & APY_UC_PRINTABLE)) ok = 0;
            break;
        case APY_ISIDENT:
            if (first) {
                if (!(m & APY_UC_XIDSTART)) ok = 0;
            } else if (!(m & APY_UC_XIDCONT)) ok = 0;
            break;
        case APY_ISLOWER:
            /* "no uppercase AND at least one lowercase" -- `'ab1'.islower()`
               is True and `'123'.islower()` is False. A plain "every
               character is lowercase" answers the second one wrongly. */
            if (m & (APY_UC_UPPER | APY_UC_TITLE)) ok = 0;
            if (m & APY_UC_LOWER) cased = 1;
            break;
        case APY_ISUPPER:
            if (m & (APY_UC_LOWER | APY_UC_TITLE)) ok = 0;
            if (m & APY_UC_UPPER) cased = 1;
            break;
        default:
            /* `istitle`: an upper- or title-case character may only follow an
               uncased one, and a lowercase one may only follow a cased one. */
            if (m & (APY_UC_UPPER | APY_UC_TITLE)) {
                if (prev_cased) ok = 0;
                cased = 1;
            } else if (m & APY_UC_LOWER) {
                if (!prev_cased) ok = 0;
                cased = 1;
            }
            prev_cased = (m & (APY_UC_UPPER | APY_UC_LOWER
                               | APY_UC_TITLE)) != 0;
            break;
        }
        first = 0;
        if (!ok) return apy_from_bool(0);
    }
    if (!any) return apy_from_bool(0);
    if (which == APY_ISLOWER || which == APY_ISUPPER || which == APY_ISTITLE)
        return apy_from_bool(cased);
    return apy_from_bool(1);
}

APY_API apy_value apy_str_isalpha(apy_value s) {
    if (!apy_str_self("isalpha", s)) return 0;
    return apy_str_is(s, APY_ISALPHA);
}
APY_API apy_value apy_str_isdigit(apy_value s) {
    if (!apy_str_self("isdigit", s)) return 0;
    return apy_str_is(s, APY_ISDIGIT);
}
/* `isdecimal`, `isdigit` and `isnumeric` are three different questions
   outside ASCII: U+00B2 is a digit and numeric but not decimal, and U+2167 is
   numeric alone. They shared one test while there was no table to tell them
   apart; there is one now, so they do not. */
APY_API apy_value apy_str_isdecimal(apy_value s) {
    if (!apy_str_self("isdecimal", s)) return 0;
    return apy_str_is(s, APY_ISDECIMAL);
}
APY_API apy_value apy_str_isnumeric(apy_value s) {
    if (!apy_str_self("isnumeric", s)) return 0;
    return apy_str_is(s, APY_ISNUMERIC);
}
APY_API apy_value apy_str_isalnum(apy_value s) {
    if (!apy_str_self("isalnum", s)) return 0;
    return apy_str_is(s, APY_ISALNUM);
}
APY_API apy_value apy_str_isspace(apy_value s) {
    if (!apy_str_self("isspace", s)) return 0;
    return apy_str_is(s, APY_ISSPACE);
}
APY_API apy_value apy_str_islower(apy_value s) {
    if (!apy_str_self("islower", s)) return 0;
    return apy_str_is(s, APY_ISLOWER);
}
APY_API apy_value apy_str_isupper(apy_value s) {
    if (!apy_str_self("isupper", s)) return 0;
    return apy_str_is(s, APY_ISUPPER);
}
APY_API apy_value apy_str_istitle(apy_value s) {
    if (!apy_str_self("istitle", s)) return 0;
    return apy_str_is(s, APY_ISTITLE);
}
APY_API apy_value apy_str_isprintable(apy_value s) {
    if (!apy_str_self("isprintable", s)) return 0;
    return apy_str_is(s, APY_ISPRINTABLE);
}
APY_API apy_value apy_str_isidentifier(apy_value s) {
    if (!apy_str_self("isidentifier", s)) return 0;
    return apy_str_is(s, APY_ISIDENT);
}
APY_API apy_value apy_str_isascii(apy_value s) {
    if (!apy_str_self("isascii", s)) return 0;
    return apy_str_is(s, APY_ISASCII);
}

/* --- strip -------------------------------------------------------------- */
/* `chars` is a SET of characters to remove, not a prefix to match:
   `'xyabyx'.strip('xy')` is 'ab'. A null `chars` means whitespace. */
/* BY CODE POINT, not by byte, and both halves of that matter.
   WHITESPACE: `apy_c_space` knows the six ASCII bytes, so `strip()` left
   U+00A0 and U+2003 in place where Python removes them. The table already
   answers this question for `str.isspace`, and now it answers it here.
   THE SET: `chars` was compared a byte at a time, so a multi-byte character
   in it matched the HALVES of other characters -- `'ab'.strip('é')`
   would have eaten a 0xC3 lead byte and left a dangling continuation. */
/* AND FOR A BYTES RECEIVER, ASCII AGAIN. `b"\xc2\xa0a\xc2\xa0".strip()` is
   itself in Python: U+00A0 is whitespace, the two BYTES that spell it are
   not, and reading them as the character stripped bytes the program never
   spelled as one. `wide` says which receiver is asking; the set is walked in
   the same unit for the same reason. */
static int apy_in_chars(int wide, apy_value chars, uint32_t cp) {
    const unsigned char *p;
    int64_t n, i = 0;
    if (!chars) return (apy_text_class(wide, cp) & APY_UC_SPACE) != 0;
    p = (const unsigned char *)O(chars)->v.s.p;
    n = O(chars)->v.s.n;
    while (i < n) {
        uint32_t c2;
        int64_t used = apy_text_step(wide, p, n, i, &c2);
        if (!used) { c2 = 0xFFFD; used = 1; }
        if (c2 == cp) return 1;
        i += used;
    }
    return 0;
}

static apy_value apy_str_trim(apy_value s, apy_value chars, const char *meth,
                              int left, int right) {
    int64_t lo = 0, hi = O(s)->v.s.n;
    /* BYTES TOO, for a bytes receiver: `b'abc'.strip(b'a')` is Python and
       this refused it. The receiver and the argument still have to agree --
       `b'abc'.strip('a')` is a TypeError in CPython as well -- which the
       kind comparison below gets by asking whether they MATCH rather than
       whether the argument is a str. */
    if (chars) chars = apy_text_like(chars);
    if (chars && O(chars)->kind != APY_NONE_K && O(chars)->kind != O(s)->kind) {
        /* Its own wording, naming NEITHER the offending kind nor a position:
           `strip arg must be None or str`. */
        char buf[80];
        snprintf(buf, sizeof buf, "%s arg must be None or str", meth);
        return apy_fail("TypeError", buf);
    }
    /* ONE FORWARD PASS DECIDES BOTH ENDS. Walking in from the right would
       mean stepping UTF-8 backwards, which needs its own scan for a lead
       byte; remembering where the last non-stripped character ended costs
       one variable and no second way to be wrong. */
    {
        const unsigned char *p = (const unsigned char *)O(s)->v.s.p;
        int wide = O(s)->kind == APY_STR_K;
        int64_t i, last;
        if (left) {
            for (i = 0; i < hi; ) {
                uint32_t cp;
                int64_t used = apy_text_step(wide, p, hi, i, &cp);
                if (!used) { cp = 0xFFFD; used = 1; }
                if (!apy_in_chars(wide, chars, cp)) break;
                i += used;
            }
            lo = i;
        }
        if (right) {
            last = lo;
            for (i = lo; i < hi; ) {
                uint32_t cp;
                int64_t used = apy_text_step(wide, p, hi, i, &cp);
                if (!used) { cp = 0xFFFD; used = 1; }
                i += used;
                if (!apy_in_chars(wide, chars, cp)) last = i;
            }
            hi = last;
        }
    }
    return apy_str_slice_of(s, lo, hi);
}

APY_API apy_value apy_str_strip(apy_value s) {
    if (!apy_str_self("strip", s)) return 0;
    return apy_str_trim(s, 0, "strip", 1, 1);
}
APY_API apy_value apy_str_lstrip(apy_value s) {
    if (!apy_str_self("lstrip", s)) return 0;
    return apy_str_trim(s, 0, "lstrip", 1, 0);
}
APY_API apy_value apy_str_rstrip(apy_value s) {
    if (!apy_str_self("rstrip", s)) return 0;
    return apy_str_trim(s, 0, "rstrip", 0, 1);
}
APY_API apy_value apy_str_strip_chars(apy_value s, apy_value chars) {
    if (!apy_str_self("strip", s)) return 0;
    if (O(chars)->kind == APY_NONE_K) return apy_str_trim(s, 0, "strip", 1, 1);
    return apy_str_trim(s, chars, "strip", 1, 1);
}
APY_API apy_value apy_str_lstrip_chars(apy_value s, apy_value chars) {
    if (!apy_str_self("lstrip", s)) return 0;
    if (O(chars)->kind == APY_NONE_K) return apy_str_trim(s, 0, "lstrip", 1, 0);
    return apy_str_trim(s, chars, "lstrip", 1, 0);
}
APY_API apy_value apy_str_rstrip_chars(apy_value s, apy_value chars) {
    if (!apy_str_self("rstrip", s)) return 0;
    if (O(chars)->kind == APY_NONE_K) return apy_str_trim(s, 0, "rstrip", 0, 1);
    return apy_str_trim(s, chars, "rstrip", 0, 1);
}

APY_API apy_value apy_str_removeprefix(apy_value s, apy_value p) {
    if (!apy_str_self("removeprefix", s)) return 0;
    p = apy_text_arg("removeprefix", 0, 0, s, p);
    if (!p) return 0;
    if (O(p)->v.s.n && O(p)->v.s.n <= O(s)->v.s.n
        && memcmp(O(s)->v.s.p, O(p)->v.s.p, (size_t)O(p)->v.s.n) == 0)
        return apy_str_slice_of(s, O(p)->v.s.n, O(s)->v.s.n);
    return s;
}

APY_API apy_value apy_str_removesuffix(apy_value s, apy_value p) {
    if (!apy_str_self("removesuffix", s)) return 0;
    p = apy_text_arg("removesuffix", 0, 0, s, p);
    if (!p) return 0;
    if (O(p)->v.s.n && O(p)->v.s.n <= O(s)->v.s.n
        && memcmp(O(s)->v.s.p + O(s)->v.s.n - O(p)->v.s.n,
                  O(p)->v.s.p, (size_t)O(p)->v.s.n) == 0)
        return apy_str_slice_of(s, 0, O(s)->v.s.n - O(p)->v.s.n);
    return s;
}

/* --- split and join ------------------------------------------------------
   THE TWO SPLIT MODES ARE DIFFERENT ALGORITHMS, not one with a default
   separator, and the case that shows it is `'  a  b  '`: with no argument it
   splits on RUNS of whitespace and drops the empty pieces at both ends, giving
   ['a', 'b']; with `' '` it splits on each single space and keeps them, giving
   ['', '', 'a', '', 'b', '', '']. A default of `' '` would answer the second
   to both. */
/* The bytes of the whitespace CHARACTER starting at `i`, or 0.

   `apy_c_space` KNOWS THE SIX ASCII BYTES AND NOTHING ELSE, so
   `"\u00a0".split()` answered `["\u00a0"]` where Python answers `[]`: every
   Unicode space was an ordinary character to split around. `strip()` was
   fixed by asking the character table; this is the same fix for the other
   splitter. A BYTES RECEIVER KEEPS THE ASCII ANSWER.

   ANSWERING A WIDTH RATHER THAN A FLAG is what lets the caller advance: a
   space character may be two or three bytes, and stepping one at a time
   would ask about its continuation bytes. */
static int64_t apy_space_at(int wide, const unsigned char *p, int64_t n,
                            int64_t i) {
    uint32_t cp;
    int64_t used = apy_text_step(wide, p, n, i, &cp);
    if (used <= 0) return 0;
    return (apy_text_class(wide, cp) & APY_UC_SPACE) ? used : 0;
}

/* Where the character ENDING at `i` begins -- the backward step `rsplit`
   needs. A continuation byte is `10xxxxxx`, so the lead is the first byte at
   or before `i - 1` that is not one; four bytes is the most a character can
   take, which bounds the walk over malformed input. */
static int64_t apy_char_start(int wide, const unsigned char *p, int64_t i) {
    int64_t at = i - 1;
    if (!wide) return at;
    while (at > 0 && i - at < 4 && (p[at] & 0xC0) == 0x80) at--;
    return at;
}

APY_API apy_value apy_split_ws_of(apy_value s, int64_t maxsplit,
                                 int64_t from_right) {
    apy_value out = apy_seq_new(APY_LIST_K, 8);
    int64_t n = O(s)->v.s.n, i, j;
    const unsigned char *q = (const unsigned char *)O(s)->v.s.p;
    int wide = O(s)->kind == APY_STR_K;
    if (!from_right) {
        i = 0;
        while (i < n) {
            int64_t skip;
            while (i < n && (skip = apy_space_at(wide, q, n, i)) > 0) i += skip;
            if (i >= n) break;
            if (maxsplit >= 0 && O(out)->v.q.n == maxsplit) {
                /* The remainder goes in WHOLE, INCLUDING its trailing
                   whitespace: `'  a  b  '.split(None, 1)` is ['a', 'b  '].
                   Only the whitespace BEFORE a piece is skipped, and that
                   already happened above. Right-stripping the remainder as
                   well looks tidier and answers ['a', 'b'], which is wrong --
                   and invisible unless a case splits a string that has
                   trailing space. */
                apy_q_append(out, apy_str_slice_of(s, i, n));
                return out;
            }
            j = i;
            while (j < n && !apy_space_at(wide, q, n, j)) {
                uint32_t cp;
                int64_t used = apy_text_step(wide, q, n, j, &cp);
                j += used > 0 ? used : 1;
            }
            apy_q_append(out, apy_str_slice_of(s, i, j));
            i = j;
        }
        return out;
    }
    i = n;
    while (i > 0) {
        int64_t at;
        while (i > 0 && apy_space_at(wide, q, n,
                                     at = apy_char_start(wide, q, i))) i = at;
        if (i <= 0) break;
        if (maxsplit >= 0 && O(out)->v.q.n == maxsplit) {
            /* Mirror image of the forward case: the remainder keeps its
               LEADING whitespace. `'  a  b  '.rsplit(None, 1)` is
               ['  a', 'b']. */
            apy_q_append(out, apy_str_slice_of(s, 0, i));
            break;
        }
        j = i;
        while (j > 0 && !apy_space_at(wide, q, n,
                                      at = apy_char_start(wide, q, j))) j = at;
        apy_q_append(out, apy_str_slice_of(s, j, i));
        i = j;
    }
    /* Built back to front, so reverse it. */
    for (i = 0, j = O(out)->v.q.n - 1; i < j; i++, j--) {
        apy_value t = O(out)->v.q.items[i];
        O(out)->v.q.items[i] = O(out)->v.q.items[j];
        O(out)->v.q.items[j] = t;
    }
    return out;
}

APY_API apy_value apy_split_sep_of(apy_value s, apy_value sep,
                                  int64_t maxsplit, int64_t from_right) {
    apy_value out;
    int64_t n = O(s)->v.s.n, m = O(sep)->v.s.n, at, i, j;
    if (m == 0) return apy_fail("ValueError", "empty separator");
    out = apy_seq_new(APY_LIST_K, 8);
    if (!from_right) {
        i = 0;
        while (maxsplit < 0 || O(out)->v.q.n < maxsplit) {
            at = apy_find_at(s, sep, i, n);
            if (at < 0) break;
            apy_q_append(out, apy_str_slice_of(s, i, at));
            i = at + m;
        }
        apy_q_append(out, apy_str_slice_of(s, i, n));
        return out;
    }
    i = n;
    while (maxsplit < 0 || O(out)->v.q.n < maxsplit) {
        at = apy_rfind_at(s, sep, 0, i);
        if (at < 0) break;
        apy_q_append(out, apy_str_slice_of(s, at + m, i));
        i = at;
    }
    apy_q_append(out, apy_str_slice_of(s, 0, i));
    for (i = 0, j = O(out)->v.q.n - 1; i < j; i++, j--) {
        apy_value t = O(out)->v.q.items[i];
        O(out)->v.q.items[i] = O(out)->v.q.items[j];
        O(out)->v.q.items[j] = t;
    }
    return out;
}

APY_API apy_value apy_str_split_impl_of(apy_value s, apy_value sep,
                                       apy_value limit,
                                       int64_t from_right) {
    int64_t maxsplit = -1;
    if (limit && !apy_int_arg(limit, &maxsplit)) return 0;
    if (maxsplit < 0) maxsplit = -1;      /* any negative means "no limit" */
    if (!sep || O(sep)->kind == APY_NONE_K)
        return apy_split_ws_of(s, maxsplit, from_right);
    /* THE RECEIVER DECIDES. A bytes separator used to pass for a str
       receiver and back, so `b"a,b".split(",")` answered `[b'a', b'b']` -- a
       wrong answer where CPython refuses. */
    if (O(s)->kind == APY_BYTES_K) {
        sep = apy_text_arg("split", 0, 0, s, sep);
        if (!sep) return 0;
    } else if (O((sep = apy_text_like(sep)))->kind != APY_STR_K) {
        /* A STR RECEIVER HAS ITS OWN WORDING, and it mentions None because
           None is what a separator may also be. */
        return apy_fail2("TypeError", "must be str or None, not %s%s",
                         apy_kind_name(sep), "");
    }
    return apy_split_sep_of(s, sep, maxsplit, from_right);
}

/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_str_split_impl(apy_value s, apy_value sep,
                                    apy_value limit, int from_right) {
    return apy_str_split_impl_of(s, sep, limit, (int64_t)from_right);
}

APY_API apy_value apy_str_split_ws(apy_value s) {
    if (!apy_str_self("split", s)) return 0;
    return apy_str_split_impl(s, 0, 0, 0);
}
APY_API apy_value apy_str_split(apy_value s, apy_value sep) {
    if (!apy_str_self("split", s)) return 0;
    return apy_str_split_impl(s, sep, 0, 0);
}
APY_API apy_value apy_str_split_n(apy_value s, apy_value sep, apy_value limit) {
    if (!apy_str_self("split", s)) return 0;
    return apy_str_split_impl(s, sep, limit, 0);
}
APY_API apy_value apy_str_rsplit_ws(apy_value s) {
    if (!apy_str_self("rsplit", s)) return 0;
    return apy_str_split_impl(s, 0, 0, 1);
}
APY_API apy_value apy_str_rsplit(apy_value s, apy_value sep) {
    if (!apy_str_self("rsplit", s)) return 0;
    return apy_str_split_impl(s, sep, 0, 1);
}
APY_API apy_value apy_str_rsplit_n(apy_value s, apy_value sep, apy_value limit) {
    if (!apy_str_self("rsplit", s)) return 0;
    return apy_str_split_impl(s, sep, limit, 1);
}

/* `splitlines` breaks on \n, \r and \r\n. CPython also breaks on \v, \f,
   \x1c-\x1e and three Unicode separators; those are not here, and a text
   containing one comes back as a single line. Stated, not silent. */
APY_API apy_value apy_splitlines_impl_of(apy_value s, int64_t keepends) {
    apy_value out = apy_seq_new(APY_LIST_K, 8);
    int64_t n = O(s)->v.s.n, i = 0, start;
    while (i < n) {
        start = i;
        while (i < n && O(s)->v.s.p[i] != '\n' && O(s)->v.s.p[i] != '\r') i++;
        {
            int64_t stop = i;
            if (i < n) {
                if (O(s)->v.s.p[i] == '\r' && i + 1 < n
                    && O(s)->v.s.p[i + 1] == '\n') i += 2;
                else i++;
            }
            apy_q_append(out, apy_str_slice_of(s, start, keepends ? i : stop));
        }
    }
    return out;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_splitlines_impl(apy_value s, int keepends) {
    return apy_splitlines_impl_of(s, (int64_t)keepends);
}

APY_API apy_value apy_str_splitlines(apy_value s) {
    if (!apy_str_self("splitlines", s)) return 0;
    return apy_splitlines_impl(s, 0);
}
APY_API apy_value apy_str_splitlines_keep(apy_value s, apy_value keep) {
    if (!apy_str_self("splitlines", s)) return 0;
    return apy_splitlines_impl(s, apy_truth(keep));
}

/* `partition` returns three pieces ALWAYS. On a miss the original goes in the
   first slot and the other two are empty; `rpartition` puts it in the LAST,
   which is the only asymmetry between them and is easy to get backwards. */
static apy_value apy_partition_impl(apy_value s, apy_value sep, int from_right) {
    apy_value out = apy_seq_new(APY_TUPLE_K, 3);
    /* THE SEPARATOR COMES BACK AS THE OBJECT IT WAS HANDED, which for a str
       subclass means the INSTANCE and not the text inside it: CPython's
       `partition` increfs and returns `sep` itself, so
       `type("a-b".partition(S("-"))[1])` is `S`. The search below reads the
       buffer, which is what `apy_text_like` is for. */
    apy_value given = sep;
    int64_t n = O(s)->v.s.n, m, at;
    sep = apy_text_like(sep);
    /* `must be str, not int` -- no method name at all, which is how CPython
       words this one and unlike every other method in this file.
       BYTES TOO -- `b"abc".partition(b"b")` is the same operation -- and a
       BYTES RECEIVER WORDS IT DIFFERENTLY: `a bytes-like object is
       required, not 'int'`, quoting the type where the str form does not. */
    if (O(sep)->kind != APY_STR_K && O(sep)->kind != APY_BYTES_K) {
        if (O(s)->kind == APY_BYTES_K)
            return apy_fail2("TypeError",
                             "a bytes-like object is required, not '%s'%s",
                             apy_kind_name(sep), "");
        return apy_fail2("TypeError", "must be str, not %s%s",
                         apy_kind_name(sep), "");
    }
    m = O(sep)->v.s.n;
    if (m == 0) return apy_fail("ValueError", "empty separator");
    at = from_right ? apy_rfind_at(s, sep, 0, n) : apy_find_at(s, sep, 0, n);
    if (at < 0) {
        apy_q_append(out, from_right ? apy_lit("") : s);
        apy_q_append(out, apy_lit(""));
        apy_q_append(out, from_right ? s : apy_lit(""));
        return out;
    }
    apy_q_append(out, apy_str_slice_of(s, 0, at));
    apy_q_append(out, given);
    apy_q_append(out, apy_str_slice_of(s, at + m, n));
    return out;
}

APY_API apy_value apy_str_partition(apy_value s, apy_value sep) {
    if (!apy_str_self("partition", s)) return 0;
    return apy_partition_impl(s, sep, 0);
}
APY_API apy_value apy_str_rpartition(apy_value s, apy_value sep) {
    if (!apy_str_self("rpartition", s)) return 0;
    return apy_partition_impl(s, sep, 1);
}

/* `sep.join(parts)`. The receiver is the SEPARATOR, which reads backwards
   until you have written it once. Any iterable of str; a non-str element is
   reported with its position, because in a long list that is the only useful
   half of the message. */
APY_API apy_value apy_str_join(apy_value sep, apy_value parts) {
    int64_t n, i, len = 0, out = 0;
    apy_value *got;
    char *buf;
    if (!apy_str_self("join", sep)) return 0;
    /* THE CHECK COMES BEFORE THE FUNNEL, and is written out rather than left
       to `apy_iterable` or `apy_raw_len`: their message names the kind
       (`'int' object is not iterable`) where `join`'s names none (`can only
       join an iterable`). Checking afterwards would mean clearing a flag the
       funnel had already set in order to replace its text, which is exactly
       what the sticky-first-error rule forbids -- so the kinds below are the
       ARRIVING ones, before a generator has drained into a list. */
    if (O(parts)->kind != APY_STR_K && !apy_is_seq(parts)
        && !apy_is_set(parts) && O(parts)->kind != APY_DICT_K
        /* A CURSOR IS AN ITERABLE, and `apy_iterable` hands one straight
           back rather than draining it -- so `"".join(map(str, xs))`, about
           the commonest spelling of this whole function, was refused by the
           check meant for `"".join(5)`. `apy_raw_len` below drains it. The
           other three are iterable too and were refused for the same reason;
           each of them yields an int, so what they get now is CPython's
           per-item message about the element rather than a claim about the
           argument. */
        && O(parts)->kind != APY_ITER_K && O(parts)->kind != APY_RANGE_K
        && O(parts)->kind != APY_BYTES_K && O(parts)->kind != APY_MVIEW_K
        /* A GENERATOR AND A VIEW reach the walk as the list they drain into,
           so the old check never saw either kind; it does now. */
        && O(parts)->kind != APY_GEN_K && O(parts)->kind != APY_VIEW_K
        /* A CLASS IS JOINABLE WHEN IT CAN BE WALKED: `__iter__`, the older
           `__getitem__` protocol, or a builtin underneath. `for x in obj`
           walks all three, and this refused every one of them. */
        && !(O(parts)->kind == APY_INST_K
             && (O(parts)->v.o.held
                 || apy_class_find(O(parts)->v.o.cls, apy_name("__iter__"))
                 || apy_class_find(O(parts)->v.o.cls,
                                   apy_name("__getitem__"))))
        /* A CLASS OBJECT whose METACLASS says how to walk it -- what makes
           `",".join(Color)` an ordinary join over an enum's members. */
        && !(O(parts)->kind == APY_TYPE_K && O(parts)->v.t.meta
             && apy_class_find(O(parts)->v.t.meta, apy_name("__iter__"))))
        return apy_fail("TypeError", "can only join an iterable");
    /* AND WHAT `__iter__` GIVES BACK IS `join`'S QUESTION TOO. `apy_iterable`
       asks it as well, but its refusal names the kind -- `iter() returned
       non-iterator of type 'int'` -- where `join`'s names none. CPython
       reaches iteration through `PySequence_Fast(seq, "can only join an
       iterable")`, which REPLACES every TypeError that comes out of GETTING
       the iterator and lets the ones raised while WALKING through; the check
       above already covers the other way of failing to get one, and this is
       the last.

       NOT BY CLEARING THE FUNNEL'S FLAG afterwards, which is the other way
       to write this and is wrong: `apy_iterable` gets the iterator AND
       drains it, so a cleared TypeError would also swallow one raised inside
       a user's `__next__`, which CPython propagates.

       `__iter__` IS CALLED ONCE. What it answered is what goes on to the
       funnel, so a class whose `__iter__` has a side effect does not have it
       twice. */
    if (O(parts)->kind == APY_INST_K
        && apy_class_find(O(parts)->v.o.cls, apy_name("__iter__"))) {
        apy_value it = apy_unary_dunder(parts, "__iter__");
        if (apy_error_occurred()) return 0;
        if (it) {
            if (!apy_is_iterator(it))
                return apy_fail("TypeError", "can only join an iterable");
            parts = it;
        }
    }
    /* ANY iterable, not just an indexable one. The walk below is by index, so
       a generator has to be drained first -- and once generator expressions
       became real generators, `sep.join(f(x) for x in xs)` started arriving
       here as one. It reported "can only join an iterable" about something
       that plainly was one. */
    parts = apy_iterable(parts);
    if (!parts) return 0;
    n = apy_raw_len(parts);
    if (apy_error_occurred()) return 0;
    got = (apy_value *)malloc((size_t)(n ? n : 1) * sizeof(apy_value));
    for (i = 0; i < n; i++) {
        got[i] = apy_key_at(parts, i);
        if (!got[i]) { free(got); return 0; }
        /* THE RECEIVER DECIDES what the parts must be, and CPython words the
           two refusals differently: `expected str instance` for a str
           separator and `expected a bytes-like object` for a bytes one --
           which a MEMORYVIEW satisfies, and used to be refused. */
        got[i] = apy_text_like(got[i]);
        if (O(sep)->kind == APY_BYTES_K && O(got[i])->kind == APY_MVIEW_K)
            got[i] = apy_mview_bytes(got[i]);
        if (O(got[i])->kind != O(sep)->kind) {
            char msg[128];
            if (O(sep)->kind == APY_BYTES_K)
                snprintf(msg, sizeof msg,
                         "sequence item %lld: expected a bytes-like object, "
                         "%s found", (long long)i, apy_kind_name(got[i]));
            else
                snprintf(msg, sizeof msg,
                         "sequence item %lld: expected str instance, %s found",
                         (long long)i, apy_kind_name(got[i]));
            free(got);
            return apy_fail("TypeError", msg);
        }
        len += O(got[i])->v.s.n;
    }
    if (n > 1) len += O(sep)->v.s.n * (n - 1);
    buf = (char *)malloc((size_t)len + 1);
    for (i = 0; i < n; i++) {
        if (i) {
            memcpy(buf + out, O(sep)->v.s.p, (size_t)O(sep)->v.s.n);
            out += O(sep)->v.s.n;
        }
        memcpy(buf + out, O(got[i])->v.s.p, (size_t)O(got[i])->v.s.n);
        out += O(got[i])->v.s.n;
    }
    buf[out] = '\0';
    free(got);
    return apy_str_take(buf, out);
}

/* `replace`. An EMPTY `old` matches in every gap, so `'aaa'.replace('', '-')`
   is '-a-a-a-' -- four replacements in a three-character string. That is the
   case the obvious scan-for-a-match loop cannot express, which is why it is
   written as its own branch instead of falling out of the general one. */
static apy_value apy_replace_impl(apy_value s, apy_value old, apy_value new_,
                                  int64_t limit) {
    int64_t n = O(s)->v.s.n, m = O(old)->v.s.n, k = O(new_)->v.s.n;
    int64_t i, out = 0, hits = 0, cap;
    char *buf;
    cap = (n + 1) * (k + 1) + n + 1;
    buf = (char *)malloc((size_t)cap + 1);
    if (m == 0) {
        /* AN EMPTY NEEDLE MATCHES BETWEEN CHARACTERS and not between bytes:
           `"日".replace("", "-")` is `"-日-"`, and inserting at every byte
           boundary put a dash inside the three bytes the character is made
           of -- splitting it into rubbish. A bytes receiver counts octets
           and is left alone. */
        int wide = O(s)->kind == APY_STR_K;
        for (i = 0; i <= n; i++) {
            int starts = i == n || !wide
                || ((unsigned char)O(s)->v.s.p[i] & 0xC0) != 0x80;
            if (starts && (limit < 0 || hits < limit)) {
                memcpy(buf + out, O(new_)->v.s.p, (size_t)k);
                out += k;
                hits++;
            }
            if (i < n) buf[out++] = O(s)->v.s.p[i];
        }
        buf[out] = '\0';
        return apy_str_take(buf, out);
    }
    for (i = 0; i < n; ) {
        if ((limit < 0 || hits < limit) && i + m <= n
            && memcmp(O(s)->v.s.p + i, O(old)->v.s.p, (size_t)m) == 0) {
            memcpy(buf + out, O(new_)->v.s.p, (size_t)k);
            out += k;
            i += m;
            hits++;
        } else buf[out++] = O(s)->v.s.p[i++];
    }
    buf[out] = '\0';
    return apy_str_take(buf, out);
}

APY_API apy_value apy_str_replace(apy_value s, apy_value old, apy_value new_) {
    if (!apy_str_self("replace", s)) return 0;
    old = apy_text_arg("replace", 1, 0, s, old);
    if (!old) return 0;
    new_ = apy_text_arg("replace", 2, 0, s, new_);
    if (!new_) return 0;
    return apy_replace_impl(s, old, new_, -1);
}

APY_API apy_value apy_str_replace_n(apy_value s, apy_value old, apy_value new_,
                                    apy_value count) {
    int64_t limit;
    if (!apy_str_self("replace", s)) return 0;
    old = apy_text_arg("replace", 1, 0, s, old);
    if (!old) return 0;
    new_ = apy_text_arg("replace", 2, 0, s, new_);
    if (!new_) return 0;
    if (!apy_int_arg(count, &limit)) return 0;
    /* A NEGATIVE count means "all", not "none": `replace(a, b, -1)` replaces
       everything and `replace(a, b, 0)` replaces nothing. */
    return apy_replace_impl(s, old, new_, limit < 0 ? -1 : limit);
}

/* `startswith` / `endswith`, which accept a TUPLE of candidates and answer
   True if any of them matches -- `s.startswith(('a', 'file'))`. A tuple is
   the only container they accept; a list is a TypeError in CPython. */
APY_API int64_t apy_affix1_of(apy_value s, apy_value fix, int64_t lo,
                              int64_t hi, int64_t at_end) {
    int64_t m = O(fix)->v.s.n;
    if (m > hi - lo) return 0;
    return memcmp(O(s)->v.s.p + (at_end ? hi - m : lo),
                  O(fix)->v.s.p, (size_t)m) == 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_affix1(apy_value s, apy_value fix, int64_t lo, int64_t hi,
                      int at_end) {
    return (int)apy_affix1_of(s, fix, lo, hi, at_end);
}

APY_API apy_value apy_affix_of(apy_value s, apy_value fix, apy_value start,
                               apy_value end, int64_t at_end) {
    const char *meth = at_end ? "endswith" : "startswith";
    int64_t lo = 0, hi, i;
    if (!apy_str_self(meth, s)) return 0;
    hi = O(s)->v.s.n;
    if (start && !apy_slice_arg(start, &lo)) return 0;
    if (end && !apy_slice_arg(end, &hi)) return 0;
    apy_clamp_range(O(s)->v.s.n, &lo, &hi);
    if (O(fix)->kind == APY_TUPLE_K) {
        for (i = 0; i < O(fix)->v.q.n; i++) {
            apy_value one = O(fix)->v.q.items[i];
            /* A TUPLE ELEMENT IS HELD TO THE RECEIVER'S KIND TOO, and
               CPython words that one differently from the whole-argument
               refusal below it. */
            if (O(s)->kind == APY_BYTES_K) {
                one = apy_text_arg(meth, 0, 0, s, one);
                if (!one) return 0;
            } else if (O((one = apy_text_like(one)))->kind != APY_STR_K) {
                return apy_fail2("TypeError",
                                 "tuple for %s must only contain str, not %s",
                                 meth, apy_kind_name(one));
            }
            if (apy_affix1(s, one, lo, hi, at_end)) return apy_from_bool(1);
        }
        return apy_from_bool(0);
    }
    /* THE RECEIVER DECIDES, and so does the wording: a bytes receiver says
       `must be bytes or a tuple of bytes`. Either kind used to pass for
       either receiver, so `b"abc".startswith("a")` answered True. */
    if (O(s)->kind == APY_BYTES_K) {
        if (O(fix)->kind == APY_MVIEW_K) fix = apy_mview_bytes(fix);
        if (O(fix)->kind != APY_BYTES_K)
            return apy_fail2("TypeError",
                             "%s first arg must be bytes or a tuple of "
                             "bytes, not %s", meth, apy_kind_name(fix));
    } else if (O((fix = apy_text_like(fix)))->kind != APY_STR_K) {
        return apy_fail2("TypeError",
                         "%s first arg must be str or a tuple of str, not %s",
                         meth, apy_kind_name(fix));
    }
    return apy_from_bool(apy_affix1(s, fix, lo, hi, at_end));
}

/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now, and
   the exported half above stands in when nothing is ported. */
static apy_value apy_affix(apy_value s, apy_value fix, apy_value start,
                           apy_value end, int at_end) {
    return apy_affix_of(s, fix, start, end, (int64_t)at_end);
}

APY_API apy_value apy_str_startswith(apy_value s, apy_value fix) {
    return apy_affix(s, fix, 0, 0, 0);
}
APY_API apy_value apy_str_startswith2(apy_value s, apy_value fix, apy_value start) {
    return apy_affix(s, fix, start, 0, 0);
}
APY_API apy_value apy_str_startswith3(apy_value s, apy_value fix, apy_value start,
                                      apy_value end) {
    return apy_affix(s, fix, start, end, 0);
}
APY_API apy_value apy_str_endswith(apy_value s, apy_value fix) {
    return apy_affix(s, fix, 0, 0, 1);
}
APY_API apy_value apy_str_endswith2(apy_value s, apy_value fix, apy_value start) {
    return apy_affix(s, fix, start, 0, 1);
}
APY_API apy_value apy_str_endswith3(apy_value s, apy_value fix, apy_value start,
                                    apy_value end) {
    return apy_affix(s, fix, start, end, 1);
}

/* --- padding ------------------------------------------------------------ */
/* TWO different messages for two different mistakes: a fill that is not a str
   at all, and a str that is not exactly one character. Collapsing them reports
   `'ab'` as the wrong type and `1` as the wrong length, each of which sends
   the reader looking in the wrong place. */
/* ONE CHARACTER, WHICH MAY BE SEVERAL BYTES. This checked `v.s.n != 1` --
   a BYTE count -- so every fill character above U+007F was rejected as "not
   exactly one character long", which is a sentence about something the
   caller did not do. It answers the fill's bytes now, and the width its
   caller pads with. */
static int apy_fill_char(apy_value fill, const char **out, int64_t *nbytes) {
    /* BYTES TOO, because the receiver may be bytes: `b'ab'.ljust(4, b'*')`
       is Python and this refused it, naming the kind it had just been
       handed. A bytes fill is one ELEMENT when it is one byte, which is the
       same rule `apy_str_chars` applies to a str -- so only the counting
       differs, not the check. */
    fill = apy_text_like(fill);
    if (O(fill)->kind != APY_STR_K && O(fill)->kind != APY_BYTES_K) {
        apy_fail2("TypeError",
                  "The fill character must be a unicode character, not %s%s",
                  apy_kind_name(fill), "");
        return 0;
    }
    if ((O(fill)->kind == APY_BYTES_K ? O(fill)->v.s.n
                                      : apy_str_chars(fill)) != 1) {
        apy_fail("TypeError",
                 "The fill character must be exactly one character long");
        return 0;
    }
    *out = O(fill)->v.s.p;
    *nbytes = O(fill)->v.s.n;
    return 1;
}

enum { APY_LJUST, APY_RJUST, APY_CENTER };

static apy_value apy_pad(apy_value s, apy_value width, apy_value fill, int how) {
    /* A WIDTH IS COUNTED IN CHARACTERS, and this counted bytes: `'e'.ljust(3)`
       with an accented e produced three BYTES, which is one character of
       padding where Python gives two. The receiver's byte length is still
       needed -- it is what gets copied -- so both are kept. */
    int64_t n = apy_str_chars(s), nb = O(s)->v.s.n, w, pad, left;
    int64_t fb = 1, i, out = 0, total;
    const char *fp = " ";
    char *buf;
    if (!apy_int_arg(width, &w)) return 0;
    if (fill && !apy_fill_char(fill, &fp, &fb)) return 0;
    if (w <= n) return s;          /* already wide enough: Python returns it */
    pad = w - n;
    /* CPython's own split for `center`, which is NOT `pad / 2`: it biases the
       extra character to the RIGHT for an even width and to the LEFT for an
       odd one, so `'ab'.center(7, '*')` is '***ab**' and `'ab'.center(3)` is
       ' ab'. Halving alone gets both of those backwards. */
    left = how == APY_RJUST ? pad
         : how == APY_LJUST ? 0
         : pad / 2 + (pad & w & 1);
    /* THE RESULT IS `w` CHARACTERS AND NOT `w` BYTES, so the buffer is the
       receiver's bytes plus one fill character per pad position. `memset` is
       gone with the single-byte assumption it stood on. */
    total = nb + pad * fb;
    buf = (char *)malloc((size_t)total + 1);
    for (i = 0; i < left; i++) { memcpy(buf + out, fp, (size_t)fb); out += fb; }
    memcpy(buf + out, O(s)->v.s.p, (size_t)nb); out += nb;
    for (i = left; i < pad; i++) { memcpy(buf + out, fp, (size_t)fb); out += fb; }
    buf[out] = '\0';
    return apy_str_take(buf, out);
}

APY_API apy_value apy_str_ljust(apy_value s, apy_value w) {
    if (!apy_str_self("ljust", s)) return 0;
    return apy_pad(s, w, 0, APY_LJUST);
}
APY_API apy_value apy_str_ljust_fill(apy_value s, apy_value w, apy_value f) {
    if (!apy_str_self("ljust", s)) return 0;
    return apy_pad(s, w, f, APY_LJUST);
}
APY_API apy_value apy_str_rjust(apy_value s, apy_value w) {
    if (!apy_str_self("rjust", s)) return 0;
    return apy_pad(s, w, 0, APY_RJUST);
}
APY_API apy_value apy_str_rjust_fill(apy_value s, apy_value w, apy_value f) {
    if (!apy_str_self("rjust", s)) return 0;
    return apy_pad(s, w, f, APY_RJUST);
}
APY_API apy_value apy_str_center(apy_value s, apy_value w) {
    if (!apy_str_self("center", s)) return 0;
    return apy_pad(s, w, 0, APY_CENTER);
}
APY_API apy_value apy_str_center_fill(apy_value s, apy_value w, apy_value f) {
    if (!apy_str_self("center", s)) return 0;
    return apy_pad(s, w, f, APY_CENTER);
}

/* `zfill` is not `rjust(w, '0')`: a leading sign stays in FRONT of the zeros,
   so `'-5'.zfill(3)` is '-05' and not '0-5'. */
APY_API apy_value apy_str_zfill(apy_value s, apy_value width) {
    int64_t n, nb, w, pad;
    char *buf;
    int signed_ = 0;
    if (!apy_str_self("zfill", s)) return 0;
    if (!apy_int_arg(width, &w)) return 0;
    /* IN CHARACTERS TOO, for the same reason as `apy_pad` above. A zero is
       one byte, so only the receiver's two lengths can differ here. */
    n = apy_str_chars(s);
    nb = O(s)->v.s.n;
    if (w <= n) return s;
    signed_ = nb > 0 && (O(s)->v.s.p[0] == '-' || O(s)->v.s.p[0] == '+');
    pad = w - n;
    buf = (char *)malloc((size_t)(nb + pad) + 1);
    memset(buf, '0', (size_t)(nb + pad));
    if (signed_) {
        buf[0] = O(s)->v.s.p[0];
        memcpy(buf + 1 + pad, O(s)->v.s.p + 1, (size_t)(nb - 1));
    } else {
        memcpy(buf + pad, O(s)->v.s.p, (size_t)nb);
    }
    buf[nb + pad] = '\0';
    return apy_str_take(buf, nb + pad);
}

"""
