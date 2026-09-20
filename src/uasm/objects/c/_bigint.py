"""The object runtime, in C: arbitrary-precision integers.

ONE PART OF ONE TRANSLATION UNIT. `c/__init__.py` concatenates
these in order and the result is the file it always was, so a
definition here may rely on anything in an earlier part and
nothing in a later one. Sections, in order:
  * arbitrary precision integers
  * big: signed arithmetic
  * big: text
  * big: floats
  * big: bitwise
  * big: division to a float
  * big: comparison
  * big: base 2, 8 and 16
"""

C = r"""/* --- arbitrary precision integers --------------------------------------- */
/* Python has ONE integer type and it has no width. Until now this file had a
   64-bit one that wrapped, which the module docstring called the largest
   single divergence in it: `2 ** 64` was 0. This is the second integer kind
   that the `kind` field was left room for.

   THE ONE INVARIANT EVERYTHING ELSE RESTS ON: a big is NEVER a value that
   fits in an int64. Every constructor ends in `apy_big_done`, which trims the
   leading zero limbs and then, if what is left fits, throws the big away and
   returns `apy_from_int` instead. So each integer value has exactly ONE
   representation, and the cross-boundary properties the rest of the runtime
   would otherwise have to maintain by hand fall out for free:

     * `2 ** 100 // 2 ** 100 * 5` IS the small-int cell for 5 -- the same
       pointer `a = 5` gets, so even `is` agrees;
     * equality, ordering, `hash`, `repr` and dict-key identity between "a
       small 5" and "a 5 that came back from a big computation" cannot
       disagree, because there is no second 5 to disagree with;
     * `apy_eq_raw` needs no int-versus-big case at all: different kinds here
       mean different values, always.

   Maintaining that instead -- letting a big hold 5 and teaching six
   operations to compare across the boundary -- is the shape of bug this
   whole file was written to make unreachable, and it would be invisible
   until a program used one as a dict key.

   SIGN AND MAGNITUDE, base 2**32, limbs little-endian. Not two's complement:
   the magnitude algorithms are the ones written down in Knuth and are hard
   enough without a sign folded into every carry, and `&`/`|`/`^` -- the only
   operations Python defines in two's complement -- convert at the edge and
   back, which is 30 lines in one place.

   WHY 32-BIT LIMBS when the machine is 64: multiply and divide both need a
   product twice as wide as a limb, and `uint64_t` is that for a 32-bit limb
   on every C99 toolchain. 64-bit limbs would need `unsigned __int128`, which
   this file already refuses in `apy_int_quot` for the same reason -- it is
   not portable, and this source is compiled by whatever toolchain the target
   uses.

   COST. Multiplication and division are schoolbook, O(n*m); decimal
   conversion is repeated division by 10**9, so O(n**2). No Karatsuba, no
   divide-and-conquer base conversion. `2 ** 500` costs nothing measurable
   and a million-digit number would be slow; the suite's largest is
   `2 ** 1000`. Stated so the next person replaces it deliberately.

   MEMORY IS STILL NEVER FREED -- see the head of this file. Every temporary
   limb array here leaks, including the ones a single `a % b` allocates. */
typedef uint32_t apy_limb;
#define APY_LIMB_BITS 32
#define APY_LIMB_BASE ((uint64_t)1 << 32)

/* The cap on how big a big may get, in limbs -- about 1.2 million decimal
   digits. Python has no such limit and would simply take longer; the reason
   there is one here is that `10 ** 10 ** 9` should report something rather
   than allocate until the machine dies, and an OverflowError naming the
   operation is recoverable where an OOM kill is not. Nothing the conformance
   suite runs comes within four orders of magnitude of it. */
#define APY_BIG_MAX_LIMBS 131072

static apy_value apy_big_too_large(void) {
    return apy_fail("OverflowError",
                    "integer result too large for this implementation");
}

APY_API int64_t apy_is_big_of(apy_value v) {
    return O(v)->kind == APY_BIG_K;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now, and
   the exported half above stands in when nothing is ported. */
static int apy_is_big(apy_value v) {
    return (int)apy_is_big_of(v);
}

/* A `None` where a slice bound is expected means NOT GIVEN, not an error:
   `'ab'.find('a', None)` is 0. That is how CPython's own argument clinic
   spells an optional index, so every method taking start/end accepts it. */
/* An integer argument that has to fit a MACHINE INDEX. Widening
   `apy_is_int_like` to admit a big made every `O(v)->v.i` behind it a pointer
   read as an integer -- silently, and with a plausible-looking huge number
   coming out. There is no answer to give: a list cannot have 2**100 elements
   and a string cannot be padded to 2**100 columns, so CPython reports, and so
   does this.

   THREE REPORTS, and the pairing is not derivable from anything -- it is what
   CPython happens to raise at each of the three places it converts, so it is
   written out rather than reasoned about:
     APY_IDX_SUB     `[1, 2][2 ** 100]`   IndexError,    "index-sized"
     APY_IDX_REPEAT  `[1, 2] * (2 ** 100)` OverflowError, "index-sized"
     APY_IDX_SIZE    `'ab'.ljust(2 ** 100)` OverflowError, "C ssize_t" */
enum { APY_IDX_SUB, APY_IDX_REPEAT, APY_IDX_SIZE };

/* DECLARED AHEAD, because this part is second and both live far below it:
   `apy_index` is the one place a non-integer index is refused and a user
   object is asked for its `__index__`, and `apy_is_int_like` is what says
   whether either is needed. */
static int apy_is_int_like(apy_value v);
APY_API int64_t apy_index(apy_value v);

APY_API int64_t apy_index_arg_of(apy_value v, apy_value out, int64_t form) {
    if (apy_is_big(v)) {
        apy_fail(form == APY_IDX_SUB ? "IndexError" : "OverflowError",
                 form == APY_IDX_SIZE
                   ? "Python int too large to convert to C ssize_t"
                   : "cannot fit 'int' into an index-sized integer");
        return 0;
    }
    /* A NON-INTEGER IS REFUSED AND NOT READ. This used to take the payload of
       whatever it was handed, so `[1, 2].pop(1.0)` read the BITS OF A DOUBLE
       as an index. `apy_index` is what words the refusal and what asks a user
       object for its `__index__`, so the two are one implementation. */
    if (!apy_is_int_like(v)) {
        int64_t got = apy_index(v);
        if (apy_error_occurred()) return 0;
        *(int64_t *)(uintptr_t)out = got;
        return 1;
    }
    *(int64_t *)(uintptr_t)out = O(v)->v.i;
    return 1;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now, and the
   exported half above stands in when nothing is ported. THE DELEGATE CONVERTS
   rather than forwards -- the subset has one integer width, so the
   `int64_t *` out-parameter crosses as a plain word and is cast back here. */
static int apy_index_arg(apy_value v, int64_t *out, int form) {
    return (int)apy_index_arg_of(v, (apy_value)(uintptr_t)out, (int64_t)form);
}

APY_API apy_value apy_big_alloc_of(int64_t n) {
    apy_obj *o = apy_alloc(APY_BIG_K);
    if (n < 1) n = 1;
    o->v.big.limb = (apy_limb *)calloc((size_t)n, sizeof(apy_limb));
    if (!o->v.big.limb) { fputs("uasm: out of memory\n", stderr); exit(1); }
    o->v.big.n = n;
    o->v.big.neg = 0;
    return (apy_value)o;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now, and
   the exported half above stands in when nothing is ported. `apy_obj *`
   crosses as a plain word -- see `apy_mag_cmp_of`. */
static apy_obj *apy_big_alloc(int64_t n) {
    return (apy_obj *)(uintptr_t)apy_big_alloc_of(n);
}

/* Drop leading zero limbs, then DEMOTE if the value fits an int64 -- the
   invariant at the top of this section, enforced in the one place every
   result passes through. A zero-limb magnitude is the integer 0. */
APY_API apy_value apy_big_done_of(apy_value ov) {
    apy_obj *o = (apy_obj *)(uintptr_t)ov;
    int64_t n = o->v.big.n;
    while (n > 0 && o->v.big.limb[n - 1] == 0) n--;
    o->v.big.n = n;
    if (n == 0) return apy_from_int(0);
    if (n <= 2) {
        uint64_t m = o->v.big.limb[0];
        if (n == 2) m |= (uint64_t)o->v.big.limb[1] << 32;
        if (!o->v.big.neg) {
            if (m <= (uint64_t)9223372036854775807ULL)
                return apy_from_int((int64_t)m);
        } else if (m <= (uint64_t)9223372036854775808ULL) {
            /* -2**63 is representable and +2**63 is not, which is why the
               bound differs by one between the two branches. Negating
               through unsigned because negating INT64_MIN is undefined. */
            return apy_from_int((int64_t)(0u - m));
        }
    }
    return V(o);
}
/* THE NAME ITS CALLERS USE, kept as a delegate. */
static apy_value apy_big_done(apy_obj *o) {
    return apy_big_done_of((apy_value)o);
}

/* An int64 as a magnitude plus a sign, for feeding a mixed operation into the
   big path. Never normalised -- it is an operand, not a result. */
APY_API apy_value apy_big_of_i64_of(int64_t v) {
    apy_obj *o = apy_big_alloc(2);
    uint64_t m = apy_abs64(v);
    o->v.big.limb[0] = (apy_limb)(m & 0xffffffffu);
    o->v.big.limb[1] = (apy_limb)(m >> 32);
    o->v.big.neg = v < 0;
    if (o->v.big.limb[1] == 0) o->v.big.n = o->v.big.limb[0] ? 1 : 0;
    return (apy_value)o;
}
/* THE NAME ITS CALLERS USE, kept as a delegate. */
static apy_obj *apy_big_of_i64(int64_t v) {
    return (apy_obj *)(uintptr_t)apy_big_of_i64_of(v);
}

/* Either integer kind as a big object. A bool arrives here too -- `True` is
   1 for arithmetic -- which is why this reads `v.i` rather than checking for
   APY_INT_K alone. */
static apy_obj *apy_as_big(apy_value v) {
    if (O(v)->kind == APY_BIG_K) return O(v);
    return apy_big_of_i64(O(v)->v.i);
}

/* THE SAME INTEGER IN A CELL OF ITS OWN -- CPython's `_PyLong_Copy`, which is
   the whole of what `int.__getnewargs__` hands back (Objects/longobject.c).
   Only the big half needs one: `_PyLong_Copy` re-interns a value in the small
   range and `apy_from_int` already does exactly that, so the compact kinds
   copy by going through the ordinary constructor.

   THROUGH `apy_big_done` LIKE EVERY OTHER BIG RESULT. A copy of an already
   normal magnitude cannot demote, so the call changes nothing here; it is
   written because "every big leaves through `apy_big_done`" is the invariant
   that keeps a magnitude that FITS an int64 from ever reaching a program as
   a big, and one exception to it is how that invariant stops being true. */
static apy_value apy_big_dup(apy_value v) {
    apy_obj *o = apy_big_alloc(O(v)->v.big.n);
    int64_t i;
    o->v.big.neg = O(v)->v.big.neg;
    for (i = 0; i < O(v)->v.big.n; i++)
        o->v.big.limb[i] = O(v)->v.big.limb[i];
    return apy_big_done(o);
}

/* `apy_obj *` CROSSES AS A PLAIN WORD, because the subset has no
   pointer-to-struct to declare -- see `runtime/calling.py` for where
   this first bit. The locals below give the body its names back. */
APY_API int64_t apy_mag_cmp_of(apy_value av, apy_value bv) {
    const apy_obj *a = (const apy_obj *)av;
    const apy_obj *b = (const apy_obj *)bv;
    int64_t i;
    if (a->v.big.n != b->v.big.n) return a->v.big.n < b->v.big.n ? -1 : 1;
    for (i = a->v.big.n - 1; i >= 0; i--)
        if (a->v.big.limb[i] != b->v.big.limb[i])
            return a->v.big.limb[i] < b->v.big.limb[i] ? -1 : 1;
    return 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_mag_cmp(const apy_obj *a, const apy_obj *b) {
    return (int)apy_mag_cmp_of((apy_value)a, (apy_value)b);
}

static apy_obj *apy_mag_add(const apy_obj *a, const apy_obj *b) {
    int64_t na = a->v.big.n, nb = b->v.big.n, i;
    int64_t n = (na > nb ? na : nb) + 1;
    apy_obj *r = apy_big_alloc(n);
    uint64_t carry = 0;
    for (i = 0; i < n; i++) {
        uint64_t t = carry;
        if (i < na) t += a->v.big.limb[i];
        if (i < nb) t += b->v.big.limb[i];
        r->v.big.limb[i] = (apy_limb)t;
        carry = t >> APY_LIMB_BITS;
    }
    return r;
}

/* |a| - |b|, and the CALLER has established |a| >= |b|. A borrow out of the
   top would mean it had not. */
static apy_obj *apy_mag_sub(const apy_obj *a, const apy_obj *b) {
    int64_t na = a->v.big.n, nb = b->v.big.n, i;
    apy_obj *r = apy_big_alloc(na);
    int64_t borrow = 0;
    for (i = 0; i < na; i++) {
        int64_t t = (int64_t)a->v.big.limb[i] + borrow;
        if (i < nb) t -= (int64_t)b->v.big.limb[i];
        r->v.big.limb[i] = (apy_limb)t;
        /* An arithmetic shift, so this is 0 or -1. C leaves the sign of `>>`
           on a negative value implementation-defined and every toolchain this
           targets makes it arithmetic; `apy_intop` already depends on that
           for Python's `-1 >> 999`. */
        borrow = t >> APY_LIMB_BITS;
    }
    return r;
}

static apy_obj *apy_mag_mul(const apy_obj *a, const apy_obj *b) {
    int64_t na = a->v.big.n, nb = b->v.big.n, i, j;
    apy_obj *r;
    if (na == 0 || nb == 0) return apy_big_alloc(0);
    r = apy_big_alloc(na + nb);
    for (i = 0; i < na; i++) {
        uint64_t carry = 0, ai = a->v.big.limb[i];
        if (!ai) continue;
        for (j = 0; j < nb; j++) {
            uint64_t t = ai * b->v.big.limb[j] + r->v.big.limb[i + j] + carry;
            r->v.big.limb[i + j] = (apy_limb)t;
            carry = t >> APY_LIMB_BITS;
        }
        /* The carry cannot run past `na + nb` limbs: the product of an
           na-limb and an nb-limb number needs at most that many. */
        r->v.big.limb[i + nb] = (apy_limb)((uint64_t)r->v.big.limb[i + nb] + carry);
    }
    return r;
}

/* Drop leading zero limbs. EVERY magnitude that another magnitude routine
   will read has to come through here, not just the ones that become results:
   `apy_mag_cmp` compares limb COUNTS first, and Knuth's normalisation step
   spins forever on a top limb of zero. The shifts allocate a limb they may
   not need, so they are the two that must trim before returning -- found by
   `10 ** 30 / 7` hanging, where the shifted divisor had a zero on top. */
APY_API apy_value apy_mag_trim_of(apy_value ov) {
    apy_obj *o = (apy_obj *)(uintptr_t)ov;
    while (o->v.big.n > 0 && o->v.big.limb[o->v.big.n - 1] == 0) o->v.big.n--;
    return (apy_value)o;
}
/* THE NAME ITS CALLERS USE, kept as a delegate. */
static apy_obj *apy_mag_trim(apy_obj *o) {
    return (apy_obj *)(uintptr_t)apy_mag_trim_of((apy_value)o);
}

APY_API apy_value apy_mag_shl_of(apy_value av, int64_t bits) {
    const apy_obj *a = (const apy_obj *)(uintptr_t)av;
    int64_t words = bits / APY_LIMB_BITS, off = bits % APY_LIMB_BITS, i;
    apy_obj *r;
    if (a->v.big.n == 0) return (apy_value)apy_big_alloc(0);
    r = apy_big_alloc(a->v.big.n + words + 1);
    for (i = 0; i < a->v.big.n; i++) {
        uint64_t t = (uint64_t)a->v.big.limb[i] << off;
        r->v.big.limb[i + words] |= (apy_limb)t;
        /* `off` of 0 would make the second store a shift by 32, which is
           undefined in C -- the same trap `apy_intop` documents for `<< 64`.
           Skipping it is correct as well as safe: there is nothing to carry. */
        if (off) r->v.big.limb[i + words + 1] |= (apy_limb)(t >> APY_LIMB_BITS);
    }
    return (apy_value)apy_mag_trim(r);
}
/* THE NAME ITS CALLERS USE, kept as a delegate. */
static apy_obj *apy_mag_shl(const apy_obj *a, int64_t bits) {
    return (apy_obj *)(uintptr_t)apy_mag_shl_of((apy_value)a, bits);
}

/* A LOGICAL right shift of the magnitude. `lost` reports whether any 1 bit
   fell off the bottom, which is what the arithmetic `>>` needs to floor a
   negative value correctly. */
static apy_obj *apy_mag_shr(const apy_obj *a, int64_t bits, int *lost) {
    int64_t words = bits / APY_LIMB_BITS, off = bits % APY_LIMB_BITS, i;
    apy_obj *r;
    *lost = 0;
    for (i = 0; i < words && i < a->v.big.n; i++)
        if (a->v.big.limb[i]) *lost = 1;
    if (off && words < a->v.big.n
        && (a->v.big.limb[words] & (((apy_limb)1 << off) - 1))) *lost = 1;
    if (words >= a->v.big.n) return apy_big_alloc(0);
    r = apy_big_alloc(a->v.big.n - words);
    for (i = 0; i + words < a->v.big.n; i++) {
        uint64_t t = a->v.big.limb[i + words] >> off;
        if (off && i + words + 1 < a->v.big.n)
            t |= (uint64_t)a->v.big.limb[i + words + 1] << (APY_LIMB_BITS - off);
        r->v.big.limb[i] = (apy_limb)t;
    }
    return apy_mag_trim(r);
}

/* `apy_obj *` CROSSES AS A PLAIN WORD; the local below gives the body
   its name back. See `runtime/calling.py`. */
APY_API int64_t apy_mag_bits_of(apy_value av) {
    const apy_obj *a = (const apy_obj *)av;
    apy_limb top;
    int64_t bits;
    if (a->v.big.n == 0) return 0;
    top = a->v.big.limb[a->v.big.n - 1];
    bits = (a->v.big.n - 1) * (int64_t)APY_LIMB_BITS;
    while (top) { bits++; top >>= 1; }
    return bits;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int64_t apy_mag_bits(const apy_obj *a) {
    return apy_mag_bits_of((apy_value)a);
}

/* Knuth's Algorithm D, with the single-limb divisor split out because that
   case is most of the traffic (`% 97`, `// 7`, and the decimal conversion's
   `/ 10**9`) and Algorithm D would be pure overhead for it.

   Magnitudes only. Python's floor-toward-negative-infinity rules are applied
   by the callers, which already know how to do that for int64. */
static void apy_mag_divmod(const apy_obj *a, const apy_obj *b,
                           apy_obj **qo, apy_obj **ro) {
    int64_t n = b->v.big.n, m, i, j;
    int sh = 0, lost;
    apy_obj *u, *vv, *q;
    if (apy_mag_cmp(a, b) < 0) {
        /* The quotient is 0 and the remainder is the dividend, whole. */
        apy_obj *r = apy_big_alloc(a->v.big.n ? a->v.big.n : 1);
        for (i = 0; i < a->v.big.n; i++) r->v.big.limb[i] = a->v.big.limb[i];
        *qo = apy_big_alloc(0);
        *ro = r;
        return;
    }
    if (n == 1) {
        uint64_t d = b->v.big.limb[0], rem = 0;
        apy_obj *r;
        q = apy_big_alloc(a->v.big.n);
        for (i = a->v.big.n - 1; i >= 0; i--) {
            uint64_t cur = (rem << APY_LIMB_BITS) | a->v.big.limb[i];
            q->v.big.limb[i] = (apy_limb)(cur / d);
            rem = cur % d;
        }
        r = apy_big_alloc(1);
        r->v.big.limb[0] = (apy_limb)rem;
        *qo = q;
        *ro = r;
        return;
    }
    /* NORMALISE so the divisor's top limb has its high bit set. That is what
       bounds the trial quotient's error to at most 2, which is what makes the
       correction below a fixed two steps rather than a search. */
    {
        apy_limb top = b->v.big.limb[n - 1];
        while (!(top & 0x80000000u)) { top <<= 1; sh++; }
    }
    u = apy_mag_shl(a, sh);
    vv = apy_mag_shl(b, sh);
    vv->v.big.n = n;                 /* the shift cannot lengthen the divisor */
    while (vv->v.big.n > 0 && vv->v.big.limb[vv->v.big.n - 1] == 0) vv->v.big.n--;
    n = vv->v.big.n;
    m = a->v.big.n - n;
    /* `u` needs a->n + 1 limbs so that u[j+n] always exists; apy_mag_shl
       allocated a->n + 1 already. */
    u->v.big.n = a->v.big.n + 1;
    q = apy_big_alloc(m + 1);
    for (j = m; j >= 0; j--) {
        uint64_t num = ((uint64_t)u->v.big.limb[j + n] << APY_LIMB_BITS)
                     | u->v.big.limb[j + n - 1];
        uint64_t qhat = num / vv->v.big.limb[n - 1];
        uint64_t rhat = num % vv->v.big.limb[n - 1];
        int64_t borrow = 0;
        uint64_t carry = 0;
        while (qhat >= APY_LIMB_BASE
               || (n >= 2
                   && qhat * vv->v.big.limb[n - 2]
                      > ((rhat << APY_LIMB_BITS) | u->v.big.limb[j + n - 2]))) {
            qhat--;
            rhat += vv->v.big.limb[n - 1];
            if (rhat >= APY_LIMB_BASE) break;
        }
        for (i = 0; i < n; i++) {
            uint64_t p = qhat * vv->v.big.limb[i] + carry;
            int64_t t;
            carry = p >> APY_LIMB_BITS;
            t = (int64_t)u->v.big.limb[i + j] - (int64_t)(apy_limb)p + borrow;
            u->v.big.limb[i + j] = (apy_limb)t;
            borrow = t >> APY_LIMB_BITS;
        }
        {
            int64_t t = (int64_t)u->v.big.limb[j + n] - (int64_t)carry + borrow;
            u->v.big.limb[j + n] = (apy_limb)t;
            borrow = t >> APY_LIMB_BITS;
        }
        if (borrow) {
            /* The trial quotient was one too big after all -- which happens
               for about one divisor in 2**31, so it is nearly dead code and
               is exactly the branch a hand-written test would never reach.
               objects_diff runs enough random pairs to. */
            uint64_t c = 0;
            qhat--;
            for (i = 0; i < n; i++) {
                uint64_t t = (uint64_t)u->v.big.limb[i + j]
                           + vv->v.big.limb[i] + c;
                u->v.big.limb[i + j] = (apy_limb)t;
                c = t >> APY_LIMB_BITS;
            }
            u->v.big.limb[j + n] = (apy_limb)((uint64_t)u->v.big.limb[j + n] + c);
        }
        q->v.big.limb[j] = (apy_limb)qhat;
    }
    u->v.big.n = n;                  /* the remainder is what is left in u */
    *qo = q;
    *ro = apy_mag_shr(u, sh, &lost);
}

/* --- big: signed arithmetic --------------------------------------------- */
/* `bneg` is passed rather than read, so that subtraction can flip the sign of
   the right operand without mutating a value some other name still holds. */
static apy_value apy_big_addsub(apy_obj *a, apy_obj *b, int aneg, int bneg) {
    apy_obj *r;
    if (aneg == bneg) {
        r = apy_mag_add(a, b);
        r->v.big.neg = aneg;
    } else {
        int c = apy_mag_cmp(a, b);
        if (c == 0) return apy_from_int(0);
        if (c > 0) { r = apy_mag_sub(a, b); r->v.big.neg = aneg; }
        else       { r = apy_mag_sub(b, a); r->v.big.neg = bneg; }
    }
    if (r->v.big.n > APY_BIG_MAX_LIMBS) return apy_big_too_large();
    return apy_big_done(r);
}

static apy_value apy_big_mul(apy_obj *a, apy_obj *b) {
    apy_obj *r;
    if ((a->v.big.n + b->v.big.n) > APY_BIG_MAX_LIMBS)
        return apy_big_too_large();
    r = apy_mag_mul(a, b);
    r->v.big.neg = a->v.big.neg != b->v.big.neg;
    return apy_big_done(r);
}

/* `//` and `%` together, because Python's floor rule needs both: the quotient
   is decremented and the remainder shifted by the divisor exactly when the
   signs differ and the division was not exact. This is the same correction
   the int64 path makes, over magnitudes instead of over C's truncation. */
static void apy_big_floordivmod(apy_obj *a, apy_obj *b,
                                apy_value *qout, apy_value *rout) {
    apy_obj *q, *r;
    int neg = a->v.big.neg != b->v.big.neg;
    apy_mag_divmod(a, b, &q, &r);
    {
        int64_t i, rn = r->v.big.n;
        int nonzero = 0;
        for (i = 0; i < rn; i++) if (r->v.big.limb[i]) { nonzero = 1; break; }
        q->v.big.neg = neg;
        r->v.big.neg = a->v.big.neg;
        if (neg && nonzero) {
            /* floor(-x) is -(x) - 1 when the division left something over,
               and the remainder becomes divisor - |r| with the DIVISOR's
               sign. `-7 // 2` is -4 and `-7 % 2` is 1. */
            apy_obj *one = apy_big_of_i64(1);
            apy_obj *q2 = apy_mag_add(q, one);
            apy_obj *r2 = apy_mag_sub(b, r);
            q2->v.big.neg = 1;
            r2->v.big.neg = b->v.big.neg;
            q = q2;
            r = r2;
        }
    }
    *qout = apy_big_done(q);
    *rout = apy_big_done(r);
}

/* --- big: text ----------------------------------------------------------- */
/* Repeated division by 10**9, nine decimal digits at a time. O(n**2) and
   said so at the top of the section. Nine and not ten because 10**9 is the
   largest power of ten that fits a limb, which is what keeps the inner
   division single-limb. */
/* `apy_obj *` CROSSES AS A PLAIN WORD; the local gives the body its
   name back. See `runtime/calling.py`. */
APY_API apy_value apy_big_text(apy_value ov) {
    const apy_obj *o = (const apy_obj *)ov;
    int64_t n = o->v.big.n, cap, out = 0, i, nw = n;
    apy_limb *w;
    char *buf, *rev;
    if (n == 0) return apy_lit("0");
    /* A limb is 32 bits, so it is worth at most 9.633 decimal digits; ten per
       limb plus the sign and the NUL is always enough. */
    cap = n * 10 + 4;
    w = (apy_limb *)malloc((size_t)n * sizeof(apy_limb));
    buf = (char *)malloc((size_t)cap + 1);
    rev = (char *)malloc((size_t)cap + 1);
    for (i = 0; i < n; i++) w[i] = o->v.big.limb[i];
    while (nw > 0) {
        uint64_t rem = 0;
        int k;
        for (i = nw - 1; i >= 0; i--) {
            uint64_t cur = (rem << APY_LIMB_BITS) | w[i];
            w[i] = (apy_limb)(cur / 1000000000u);
            rem = cur % 1000000000u;
        }
        while (nw > 0 && w[nw - 1] == 0) nw--;
        /* Every chunk but the LAST is zero-padded to nine digits: the leading
           zeros are real digits in the middle of the number. Only the most
           significant chunk drops them, and it is the one produced last. */
        for (k = 0; k < 9; k++) {
            rev[out++] = (char)('0' + (int)(rem % 10));
            rem /= 10;
            if (nw == 0 && rem == 0) break;
        }
    }
    if (o->v.big.neg) rev[out++] = '-';
    for (i = 0; i < out; i++) buf[i] = rev[out - 1 - i];
    buf[out] = '\0';
    free(w);
    free(rev);
    return apy_str_take(buf, out);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */

/* `int('...')` for a decimal string of any length. Nine digits at a time for
   the same reason the other direction takes nine: one limb-sized multiply
   and add per chunk instead of one per digit. Returns 0 with no error set
   when a character is not a digit, so the caller can report the whole
   literal rather than the position. */
static apy_value apy_big_from_digits(const char *p, int64_t n, int neg) {
    apy_obj *acc;
    int64_t i = 0, used = 0, cap;
    if (n == 0) return 0;
    if (n > (int64_t)APY_BIG_MAX_LIMBS * 9) return apy_big_too_large();
    /* Sized ONCE, from the digit count. A decimal digit is under 3.33 bits,
       so n digits need at most n/9.63 limbs; n/9 + 2 is that with room to
       spare. The first version grew the accumulator by two limbs per chunk
       and copied it each time, which is quadratic in MEMORY as well as in
       time -- fine for the 31 digits of `2 ** 100` and several terabytes of
       leaked intermediates at the limb cap, on a runtime that frees
       nothing. */
    cap = n / 9 + 2;
    acc = apy_big_alloc(cap);
    while (i < n) {
        uint64_t chunk = 0, scale = 1, carry;
        int64_t j;
        for (j = 0; j < 9 && i < n; j++, i++) {
            if (p[i] < '0' || p[i] > '9') return 0;
            chunk = chunk * 10 + (uint64_t)(p[i] - '0');
            scale *= 10;
        }
        /* `limb * scale` is under 2**62 -- a 32-bit limb by a scale under
           2**30 -- so the running carry never leaves a uint64. */
        carry = chunk;
        for (j = 0; j < used; j++) {
            uint64_t t = (uint64_t)acc->v.big.limb[j] * scale + carry;
            acc->v.big.limb[j] = (apy_limb)t;
            carry = t >> APY_LIMB_BITS;
        }
        while (carry && used < cap) {
            acc->v.big.limb[used++] = (apy_limb)carry;
            carry >>= APY_LIMB_BITS;
        }
    }
    acc->v.big.n = used;
    acc->v.big.neg = neg;
    return apy_big_done(acc);
}

/* An integer LITERAL too large for a machine word.

   The frontend cannot emit one as a constant: `9223372036854775808` is one
   more than int64 holds, and the IR's `const` is a machine word. So a literal
   outside the 64-bit range travels as its DECIMAL TEXT and is parsed here --
   the same parser `int('...')` uses, so the two cannot disagree about a
   number that appears both ways in one program.

   `apy_big_from_digits` normalises: a value that fits a word comes back as an
   ordinary int, which is what keeps `5` and a promoted `5` indistinguishable.
   That property is the one most easily lost when big integers are added, and
   it is why this does not simply always build a big. */
APY_API apy_value apy_int_literal(apy_value digits, int64_t n, int64_t neg) {
    return apy_big_from_digits((const char *)(uintptr_t)digits, n, (int)neg);
}

/* --- big: floats --------------------------------------------------------- */
/* The magnitude as a double, correctly rounded, or an infinity the caller
   turns into an OverflowError. Built from the top 64 bits with a sticky bit
   for everything below, then rounded to 53 once -- the same shape as
   `apy_int_quot`, and for the same reason: rounding twice is how a
   last-digit disagreement gets in. */
/* The 64 bits starting at bit `from`, counting from the bottom of the
   magnitude, zero-filled past the top. Reading a bit RANGE rather than a run
   of limbs is the whole point: the top limb of a big carries between 1 and 32
   significant bits, so "the top two limbs" is between 33 and 64 bits of value
   and only sometimes the 64 that a correct rounding needs. Taking the top two
   limbs was the first version, and `float(10 ** 30)` came out as
   9.99999999994923e+29 -- a 4-bit top limb meant 28 bits of the number were
   dropped into the sticky flag instead of into the mantissa. */
static uint64_t apy_mag_window(const apy_obj *o, int64_t from) {
    int64_t w = from / APY_LIMB_BITS, off = from % APY_LIMB_BITS;
    uint64_t r;
#define APY_L(k) ((uint64_t)((k) < o->v.big.n && (k) >= 0 ? o->v.big.limb[k] : 0))
    r = APY_L(w) >> off;
    r |= APY_L(w + 1) << (APY_LIMB_BITS - off);
    /* Only when `off` is non-zero does a third limb reach into the window --
       and only then is the shift below 64, which is what makes it defined. */
    if (off) r |= APY_L(w + 2) << (2 * APY_LIMB_BITS - off);
    return r;
#undef APY_L
}

static double apy_big_double(const apy_obj *o) {
    int64_t nbits = apy_mag_bits(o), from, i, w, off;
    uint64_t head;
    int sticky = 0;
    if (nbits == 0) return 0.0;
    if (nbits <= 64) {
        /* Exact in a uint64, and C's uint64-to-double conversion rounds to
           nearest, so there is nothing left for this function to decide. */
        head = apy_mag_window(o, 0);
        return o->v.big.neg ? -(double)head : (double)head;
    }
    from = nbits - 64;
    head = apy_mag_window(o, from);        /* top bit set, so exactly 64 bits */
    w = from / APY_LIMB_BITS;
    off = from % APY_LIMB_BITS;
    for (i = 0; i < w; i++) if (o->v.big.limb[i]) { sticky = 1; break; }
    if (!sticky && off && (o->v.big.limb[w] & (((apy_limb)1 << off) - 1)))
        sticky = 1;
    {
        /* Round 64 bits down to 53, once, nearest-even -- and a tie is only a
           tie when nothing nonzero was dropped below it, which is what
           `sticky` records. Same rule as `apy_int_quot`. */
        int drop = 64 - 53;
        uint64_t mask = ((uint64_t)1 << drop) - 1;
        uint64_t low = head & mask, half = (uint64_t)1 << (drop - 1);
        head >>= drop;
        from += drop;
        if (low > half || (low == half && (sticky || (head & 1)))) head++;
    }
    return o->v.big.neg ? -ldexp((double)head, (int)from)
                        : ldexp((double)head, (int)from);
}

/* A double whose magnitude is at least 2**63, and therefore an exact integer,
   as a big. `frexp` gives the mantissa and exponent without any rounding, so
   nothing here can lose a bit. */
static apy_value apy_big_from_double(double f) {
    int e;
    double m = frexp(fabs(f), &e);
    uint64_t mant = (uint64_t)ldexp(m, 53);
    apy_obj *o;
    e -= 53;
    o = apy_big_of_i64((int64_t)mant);
    if (e > 0) {
        apy_obj *sh = apy_mag_shl(o, e);
        o = sh;
    }
    o->v.big.neg = f < 0;
    return apy_big_done(o);
}

/* --- big: bitwise -------------------------------------------------------- */
/* `&`, `|` and `^` are the only integer operations Python defines in INFINITE
   TWO'S COMPLEMENT rather than on the magnitude: `-1` is an endless run of
   1 bits, so `5 & -1` is 5 and `~0` is -1. Sign-magnitude cannot express
   that, so both operands are converted to a two's-complement limb array one
   limb longer than either needs -- which guarantees the top limb is pure sign
   and the result's sign bit is unambiguous -- and converted back after. */
static void apy_to_twos(const apy_obj *o, apy_limb *out, int64_t n) {
    int64_t i;
    uint64_t carry = 1;
    for (i = 0; i < n; i++) {
        apy_limb w = i < o->v.big.n ? o->v.big.limb[i] : 0;
        if (o->v.big.neg) {
            uint64_t t = (uint64_t)(apy_limb)~w + carry;
            out[i] = (apy_limb)t;
            carry = t >> APY_LIMB_BITS;
        } else {
            out[i] = w;
        }
    }
}

static apy_value apy_big_bitop(apy_obj *a, apy_obj *b, int which) {
    int64_t n = (a->v.big.n > b->v.big.n ? a->v.big.n : b->v.big.n) + 1, i;
    apy_limb *ua = (apy_limb *)malloc((size_t)n * sizeof(apy_limb));
    apy_limb *ub = (apy_limb *)malloc((size_t)n * sizeof(apy_limb));
    apy_obj *r = apy_big_alloc(n);
    int neg;
    apy_to_twos(a, ua, n);
    apy_to_twos(b, ub, n);
    for (i = 0; i < n; i++) {
        switch (which) {
        case 0: r->v.big.limb[i] = ua[i] & ub[i]; break;
        case 1: r->v.big.limb[i] = ua[i] | ub[i]; break;
        default: r->v.big.limb[i] = ua[i] ^ ub[i]; break;
        }
    }
    free(ua);
    free(ub);
    neg = (r->v.big.limb[n - 1] & 0x80000000u) != 0;
    if (neg) {
        /* Back out of two's complement: negate, which is complement plus one,
           and record the sign separately. */
        uint64_t carry = 1;
        for (i = 0; i < n; i++) {
            uint64_t t = (uint64_t)(apy_limb)~r->v.big.limb[i] + carry;
            r->v.big.limb[i] = (apy_limb)t;
            carry = t >> APY_LIMB_BITS;
        }
        r->v.big.neg = 1;
    }
    return apy_big_done(r);
}

/* `<<` is an exact multiply by a power of two, so sign-magnitude handles it
   untouched. `>>` FLOORS, which for a negative value is not the same as
   shifting the magnitude: `-1 >> 10` is -1, not 0, because flooring rounds
   away from zero. Hence the `lost` bit -- if anything fell off the bottom of
   a negative value, the magnitude gains one. */
static apy_value apy_big_shift(apy_obj *a, int64_t bits, int left) {
    apy_obj *r;
    int lost = 0;
    if (left) {
        if ((a->v.big.n + bits / APY_LIMB_BITS + 2) > APY_BIG_MAX_LIMBS)
            return apy_big_too_large();
        r = apy_mag_shl(a, bits);
    } else {
        r = apy_mag_shr(a, bits, &lost);
        if (a->v.big.neg && lost) {
            apy_obj *one = apy_big_of_i64(1);
            r = apy_mag_add(r, one);
        }
    }
    r->v.big.neg = a->v.big.neg;
    return apy_big_done(r);
}

/* --- big: division to a float -------------------------------------------- */
/* `a / b` for two integers, correctly rounded, when either is too big for
   `apy_int_quot`. Same shape as that function and for the same reason: the
   quotient of the two EXACT integers, rounded once. Dividing the two doubles
   instead would round three times, and past 2**53 the conversions alone are
   already wrong.

   The dividend is shifted left far enough that the quotient carries at least
   55 bits -- two more than a double's significand, which is what leaves a
   guard bit and a round bit to decide the last one with. */
static double apy_big_quot(apy_obj *a, apy_obj *b) {
    int64_t ba = apy_mag_bits(a), bb = apy_mag_bits(b);
    int64_t shift = 55 + bb - ba, e, i, qn;
    apy_obj *num, *den, *q, *r;
    uint64_t head = 0;
    int sticky = 0, drop, hb;
    if (ba == 0) return 0.0;
    if (shift > 0) { num = apy_mag_shl(a, shift); den = b; }
    else           { num = a; den = apy_mag_shl(b, -shift); }
    apy_mag_divmod(num, den, &q, &r);
    for (i = 0; i < r->v.big.n; i++) if (r->v.big.limb[i]) { sticky = 1; break; }
    while (q->v.big.n > 0 && q->v.big.limb[q->v.big.n - 1] == 0) q->v.big.n--;
    qn = q->v.big.n;
    if (qn == 0) return 0.0;
    e = -shift;
    if (qn == 1) {
        head = q->v.big.limb[0];
    } else {
        head = ((uint64_t)q->v.big.limb[qn - 1] << APY_LIMB_BITS)
             | q->v.big.limb[qn - 2];
        for (i = qn - 3; i >= 0; i--)
            if (q->v.big.limb[i]) { sticky = 1; break; }
        e += (qn - 2) * (int64_t)APY_LIMB_BITS;
    }
    hb = 0;
    { uint64_t t = head; while (t) { hb++; t >>= 1; } }
    drop = hb - 53;
    if (drop > 0) {
        uint64_t mask = ((uint64_t)1 << drop) - 1;
        uint64_t low = head & mask, half = (uint64_t)1 << (drop - 1);
        head >>= drop;
        e += drop;
        if (low > half || (low == half && (sticky || (head & 1)))) head++;
    }
    return ldexp((double)head, (int)e);
}

/* --- big: comparison ----------------------------------------------------- */
/* `apy_obj *` CROSSES AS A PLAIN WORD, because the subset has no
   pointer-to-struct to declare -- see `runtime/calling.py` for where
   this first bit. The locals below give the body its names back. */
APY_API int64_t apy_big_cmp_of(apy_value av, apy_value bv) {
    const apy_obj *a = (const apy_obj *)av;
    const apy_obj *b = (const apy_obj *)bv;
    int c;
    if (a->v.big.neg != b->v.big.neg) return a->v.big.neg ? -1 : 1;
    c = apy_mag_cmp(a, b);
    return a->v.big.neg ? -c : c;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_big_cmp(const apy_obj *a, const apy_obj *b) {
    return (int)apy_big_cmp_of((apy_value)a, (apy_value)b);
}

/* --- big: base 2, 8 and 16 ----------------------------------------------- */
/* A power-of-two base needs no division at all: each output digit is a fixed
   run of bits. That is why `bin`, `oct` and `hex` are cheap on a big where
   `str` is quadratic. */
/* HOW MANY DIGITS a magnitude takes in a power-of-two base. `bits_per` is 1,
   3 or 4 for base 2, 8 and 16. */
static int64_t apy_big_digit_count(const apy_obj *o, int64_t bits_per) {
    int64_t nbits = apy_mag_bits(o);
    return nbits == 0 ? 1 : (nbits + bits_per - 1) / bits_per;
}

/* THE MAGNITUDE'S DIGITS, and how many there were. NO SIGN AND NO PREFIX:
   what goes around them is the caller's, and it differs -- `hex()` writes
   `-0x...` while the format mini-language has its own sign rule, its own
   optional `0x`, and grouping between. Only this part is the same, so only
   this part is shared. */
static int64_t apy_big_digits(const apy_obj *o, int64_t bits_per, char *buf,
                              int upper) {
    const char *set = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    int64_t ndig = apy_big_digit_count(o, bits_per), i, out = 0;
    for (i = ndig - 1; i >= 0; i--) {
        int64_t bit = i * bits_per;
        int64_t w = bit / APY_LIMB_BITS, off = bit % APY_LIMB_BITS;
        uint64_t chunk = w < o->v.big.n ? (o->v.big.limb[w] >> off) : 0;
        if (off && w + 1 < o->v.big.n)
            chunk |= (uint64_t)o->v.big.limb[w + 1] << (APY_LIMB_BITS - off);
        buf[out++] = set[chunk & (((uint64_t)1 << bits_per) - 1)];
    }
    return out;
}

APY_API apy_value apy_big_base_text_of(apy_value ov, int64_t bits_per,
                                      apy_value prefixv) {
    const apy_obj *o = (const apy_obj *)ov;
    const char *prefix = (const char *)prefixv;
    int64_t out = 0;
    char *buf = (char *)malloc((size_t)apy_big_digit_count(o, bits_per) + 4);
    if (o->v.big.neg) buf[out++] = '-';
    buf[out++] = prefix[0];
    buf[out++] = prefix[1];
    out += apy_big_digits(o, bits_per, buf + out, 0);
    buf[out] = '\0';
    return apy_str_take(buf, out);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_big_base_text(const apy_obj *o, int bits_per,
                                  const char *prefix) {
    return apy_big_base_text_of((apy_value)o, (int64_t)bits_per,
                                (apy_value)(uintptr_t)prefix);
}

/* `apy_obj *` CROSSES AS A PLAIN WORD; the local gives the body its
   name back. See `runtime/calling.py`. */
APY_API int64_t apy_big_popcount(apy_value ov) {
    const apy_obj *o = (const apy_obj *)ov;
    int64_t i, n = 0;
    for (i = 0; i < o->v.big.n; i++) {
        apy_limb w = o->v.big.limb[i];
        while (w) { n += w & 1; w >>= 1; }
    }
    return n;
}

"""
