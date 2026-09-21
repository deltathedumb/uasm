# Two integer functions from `math`, and the dict views' contents.
#
# ALL THREE CAME READY the moment `apy_is_int_like` and `apy_seq_new` stopped
# being `static` in the C. None of them needed anything else.


def apy_int_not(v: ptr) -> ptr:
    """The TypeError both math functions raise for a non-integer.

    IT SAYS `float` WHATEVER IT WAS HANDED, which is the C's wording and is
    kept: a `math` function given a string says the same thing. Naming the
    real kind would need `apy_raise_fmt` and would be a better message than
    the one the two arrangements have to agree on.
    """
    return apy_raise_at(
        rodata(b"TypeError\0"),
        rodata(b"'float' object cannot be interpreted as an "
               b"integer\0"))


def apy_math_abs_int(v: ptr) -> ptr:
    """`abs(v)` for an int or a big.

    NOT `apy_abs`, which is the obvious call and is not one of the SPLIT
    functions: a ported module reaching for it would be a dependency on the C
    that nobody declared, which is exactly what
    `tests/uasm/integration/test_ported_int.py` exists to catch --
    "the allocator asks the floor and nothing else". `apy_lt` and `apy_neg`
    ARE split, and between them they are the whole of what abs means for an
    integer, big or not.

    THE FAILURE CHECK IS NOT DECORATION. `apy_lt` answers 0 for a comparison
    it could not make, and `apy_truth(0)` would read a null.
    """
    less: ptr = apy_lt(v, apy_from_int(0))
    if not less:
        return less
    if apy_truth(less):
        return apy_neg(v)
    return v


def apy_math_gcd(a: ptr, b: ptr) -> ptr:
    """`math.gcd(a, b)` by Euclid, for integers of any size.

    BOTH SIGNS ARE DROPPED FIRST, because a gcd is never negative and `%`
    keeps the sign of its left operand -- so `gcd(-12, 8)` would answer -4
    without this.

    TWO LOOPS, AND THE SECOND ONE IS THE FIX. This read `v.i` off both
    arguments and ran Euclid in machine words, which is a POINTER read as
    an integer when the argument is a big: `gcd(2**100, 2**60)` answered 8.
    The docstring said so and left it, on the grounds that "the fix belongs
    with the big integers" -- and it does, which is why it is here now that
    `bigmake.py` can make one.

    THE FAST PATH IS KEPT because it is most calls and it is much faster: a
    machine-word remainder is one instruction where `apy_mod` allocates a
    value per step. It runs only when NEITHER argument is a big, and the
    general loop below runs otherwise -- through `apy_math_abs_int`,
    `apy_mod` and `apy_truth`, each of which already knows what a big is.
    """
    if not apy_is_int_like_of(a):
        return apy_int_not(a)
    if not apy_is_int_like_of(b):
        return apy_int_not(b)
    if apy_is_big_of(a) or apy_is_big_of(b):
        u: ptr = apy_math_abs_int(a)
        if not u:
            return u
        v: ptr = apy_math_abs_int(b)
        if not v:
            return v
        while apy_truth(v):
            r: ptr = apy_mod(u, v)
            if not r:
                return r
            u = v
            v = r
        return u
    x: i64 = apy_int_payload(a)
    y: i64 = apy_int_payload(b)
    if x < 0:
        x = -x
    if y < 0:
        y = -y
    while y:
        t: i64 = x % y
        x = y
        y = t
    return apy_from_int(x)


def apy_math_factorial(v: ptr) -> ptr:
    """`math.factorial(n)`.

    THROUGH THE ORDINARY MULTIPLY, so a result past int64 promotes to a big
    the way `2 ** 100` does -- `factorial(30)` has 108 bits and would
    otherwise wrap silently.

    THE LOOP STARTS AT TWO, which makes `factorial(0)` and `factorial(1)`
    both answer 1 without either being tested for.
    """
    if not apy_is_int_like_of(v):
        return apy_int_not(v)
    n: i64 = apy_int_payload(v)
    if n < 0:
        return apy_raise_at(
            rodata(b"ValueError\0"),
            rodata(b"factorial() not defined for negative values\0"))
    acc: ptr = apy_from_int(1)
    i: i64 = 2
    while i <= n:
        acc = apy_mul(acc, apy_from_int(i))
        if not acc:
            return acc
        i = i + 1
    return acc


def apy_view_items(v: ptr) -> ptr:
    """What a dict view currently holds, as a list.

    NOT A VIEW ANY MORE. This is where the liveness ends: the view reads the
    dict when asked, and this is the asking, so what comes back is a snapshot
    of that moment.

    ANYTHING THAT IS NOT A VIEW COMES BACK UNCHANGED rather than raising,
    because the callers hand it whatever they are iterating -- a list stays a
    list and only a view needs unpacking.
    """
    if i64(load(i32, offset(v, 0))) != apy_view_kind():
        return v
    d: ptr = ptr(load(u64, offset(v, apy_vw_dict_offset())))
    part: i64 = i64(load(i32, offset(v, apy_vw_part_offset())))
    n: i64 = load(i64, offset(d, apy_d_n_offset()))
    room: i64 = n
    if room < 1:
        room = 1
    out: ptr = apy_list_new(room)
    if not out:
        return out
    keys: ptr = ptr(load(u64, offset(d, apy_d_keys_offset())))
    vals: ptr = ptr(load(u64, offset(d, apy_d_vals_offset())))
    i: i64 = 0
    while i < n:
        at: i64 = i * apy_value_size()
        if part == apy_part_keys():
            apy_seq_push(out, ptr(load(u64, offset(keys, at))))
        elif part == apy_part_values():
            apy_seq_push(out, ptr(load(u64, offset(vals, at))))
        else:
            pair: ptr = apy_tuple_new(2)
            if not pair:
                return pair
            apy_seq_push(pair, ptr(load(u64, offset(keys, at))))
            apy_seq_push(pair, ptr(load(u64, offset(vals, at))))
            apy_seq_push(out, pair)
        i = i + 1
    return out


def apy_math_perm(n: ptr, k: ptr) -> ptr:
    """`math.perm(n, k)` -- the ordered arrangements of `k` from `n`.

    THE FALLING FACTORIAL, computed directly rather than as
    `factorial(n) // factorial(n - k)`: `perm(1000, 2)` is 999000 and the
    quotient form would build a 2568-digit number to answer it.

    THROUGH THE ORDINARY MULTIPLY, so a result past int64 promotes to a big
    the way `2 ** 100` does -- `perm(25, 25)` has 84 bits.
    """
    if not apy_is_int_like_of(n):
        return apy_int_not(n)
    if not apy_is_int_like_of(k):
        return apy_int_not(k)
    top: i64 = apy_int_payload(n)
    want: i64 = apy_int_payload(k)
    if apy_is_big_of(n) or apy_is_big_of(k):
        # A COUNT THAT DOES NOT FIT A MACHINE WORD is a loop that would
        # never finish, so it is refused rather than started. CPython
        # answers it, eventually; nothing that could reach here would wait.
        return apy_raise_at(
            rodata(b"OverflowError\0"),
            rodata(b"comb()/perm() arguments must fit in a machine "
                   b"word here\0"))
    if top < 0 or want < 0:
        return apy_raise_at(
            rodata(b"ValueError\0"),
            rodata(b"perm() arguments must be non-negative\0"))
    if want > top:
        return apy_from_int(0)
    acc: ptr = apy_from_int(1)
    i: i64 = 0
    while i < want:
        acc = apy_mul(acc, apy_from_int(top - i))
        if not acc:
            return acc
        i = i + 1
    return acc


def apy_math_comb(n: ptr, k: ptr) -> ptr:
    """`math.comb(n, k)` -- the unordered choices of `k` from `n`.

    DIVIDING AS IT GOES, `acc = acc * (n - i) // (i + 1)`, which is exact at
    every step because the product of any `i + 1` consecutive integers is
    divisible by `(i + 1)!`. Multiplying the whole numerator first and
    dividing once would build a number far larger than the answer --
    `comb(1000, 500)` would need the 2568-digit `1000!` to produce a
    300-digit result.

    `k` IS MIRRORED to the smaller half, since `comb(n, k) == comb(n,
    n - k)`: `comb(1000, 999)` is 1000 loops or one.
    """
    if not apy_is_int_like_of(n):
        return apy_int_not(n)
    if not apy_is_int_like_of(k):
        return apy_int_not(k)
    if apy_is_big_of(n) or apy_is_big_of(k):
        return apy_raise_at(
            rodata(b"OverflowError\0"),
            rodata(b"comb()/perm() arguments must fit in a machine "
                   b"word here\0"))
    top: i64 = apy_int_payload(n)
    want: i64 = apy_int_payload(k)
    if top < 0 or want < 0:
        return apy_raise_at(
            rodata(b"ValueError\0"),
            rodata(b"comb() arguments must be non-negative\0"))
    if want > top:
        return apy_from_int(0)
    if want > top - want:
        want = top - want
    acc: ptr = apy_from_int(1)
    i: i64 = 0
    while i < want:
        acc = apy_mul(acc, apy_from_int(top - i))
        if not acc:
            return acc
        acc = apy_floordiv(acc, apy_from_int(i + 1))
        if not acc:
            return acc
        i = i + 1
    return acc


def apy_math_lcm(a: ptr, b: ptr) -> ptr:
    """`math.lcm(a, b)`.

    DIVIDE BEFORE MULTIPLYING, which is the only interesting line: `x * y`
    can overflow an int64 where `x / g * y` does not, and the division is
    exact because the gcd divides both. Writing it the obvious way round
    would wrap for operands a program could easily have.

    A ZERO GCD MEANS BOTH WERE ZERO, and `lcm(0, 0)` is 0 rather than a
    division by it.

    THE TYPE CHECK IS `apy_math_gcd`'s, reached by calling it -- so a
    non-integer gets the same message from both, which is what a program
    comparing them would expect.

    A BIG TAKES THE SAME ROUTE `apy_math_gcd` does, and for the same
    reason: reading `v.i` off one reads a pointer, so `lcm(10**29, 30)`
    answered a number with fifteen digits instead of thirty.
    """
    g: ptr = apy_math_gcd(a, b)
    if not g:
        return g
    if not apy_truth(g):
        return apy_from_int(0)
    if apy_is_big_of(a) or apy_is_big_of(b) or apy_is_big_of(g):
        u: ptr = apy_math_abs_int(a)
        if not u:
            return u
        v: ptr = apy_math_abs_int(b)
        if not v:
            return v
        q: ptr = apy_floordiv(u, g)
        if not q:
            return q
        return apy_mul(q, v)
    d: i64 = apy_int_payload(g)
    x: i64 = apy_int_payload(a)
    y: i64 = apy_int_payload(b)
    if x < 0:
        x = -x
    if y < 0:
        y = -y
    return apy_mul(apy_from_int(x // d), apy_from_int(y))


# ── the numeric walls ──────────────────────────────────────────────────────
#
# `apy_math_arg` IS THE BIGGEST BLOCKER LEFT in `uasm plugin port`: fifteen
# functions wait on it, and it is five lines. Like the three before it, what
# kept it in the C was `static` rather than difficulty.


def apy_is_big_of(v: ptr) -> i64:
    """Is `v` an integer that outgrew a machine word?

    ASKED BEFORE EVERY int64 FAST PATH, because `v.i` on a big is a POINTER
    read as an integer -- so this is the check that stands between the
    arithmetic and a number that is really an address.
    """
    if i64(load(i32, offset(v, 0))) == apy_big_kind():
        return 1
    return 0


def apy_num_f_of(v: ptr) -> f64:
    """`v` as a double, for a number that already is one.

    A BIG GOES TO THE SLOW HALF, which is the whole reason this is a split:
    converting one needs `apy_big_double`, and the big integers are still C.
    A float and an int are two loads and a widening.

    A BOOL FALLS THROUGH TO THE INT PATH and that is correct rather than
    lucky: `True` is stored as the integer 1, so `math.sqrt(True)` is
    `math.sqrt(1)`.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_float_kind():
        return load(f64, offset(v, apy_float_offset()))
    if k == apy_big_kind():
        return apy_num_f_of_slow(v)
    return f64(apy_int_payload(v))


def apy_math_arg_of(v: ptr, fn: ptr) -> f64:
    """A `math` function's argument as a double, or a TypeError.

    `fn` IS NEVER USED, and that is the C's shape rather than an oversight
    here: every caller passes its own name and the message names only the
    KIND it was given. Dropping the parameter would mean editing fifteen call
    sites for no change a program could see, so it stays -- and stays noted,
    because a parameter nothing reads is otherwise a puzzle for whoever finds
    it next.

    ZERO ON FAILURE, which the caller must not use: `apy_fail2` has already
    set the pending error and every caller checks it. A double has no null to
    return instead.
    """
    if apy_is_num_of(v):
        return apy_num_f_of(v)
    apy_raise_fmt(
        rodata(b"TypeError\0"),
        rodata(b"must be real number, not %s%s\0"),
        apy_kind_name_of(v), rodata(b"\0"))
    return f64(0)


# ── six that were waiting on `apy_math_arg` ────────────────────────────────
#
# EVERY ONE OF THEM OPENS THE SAME WAY: take the argument as a double, stop
# if that failed, answer. The error check is not optional -- `apy_math_arg`
# answers 0.0 on a bad argument, and 0.0 is a number these would happily
# report on.


def apy_math_pi() -> f64:
    """Pi, written the way the C writes it.

    FIFTY-ONE DIGITS FOR A FIFTY-THREE BIT NUMBER, which is not excess: it is
    the exact decimal value of the nearest double, so the compiler has
    nothing to round and the two arrangements cannot land on neighbouring
    representations.
    """
    return 3.141592653589793115997963468544185161590576171875


def apy_math_isnan(v: ptr) -> ptr:
    """`math.isnan(x)`.

    A NaN IS THE ONE VALUE UNEQUAL TO ITSELF, which is how this is asked
    without a library call: no other double fails `x == x`.
    """
    x: f64 = apy_math_arg_of(v, rodata(b"isnan\0"))
    if apy_error_occurred():
        return ptr(0)
    if x != x:
        return apy_from_bool(1)
    return apy_from_bool(0)


def apy_math_isinf(v: ptr) -> ptr:
    """`math.isinf(x)`.

    FINITE MINUS ITSELF IS ZERO and an infinity minus itself is NaN, so
    `x - x != 0` separates the two -- after `x == x` has ruled out a NaN,
    which would satisfy the second test as well.
    """
    x: f64 = apy_math_arg_of(v, rodata(b"isinf\0"))
    if apy_error_occurred():
        return ptr(0)
    if x == x:
        if x - x != f64(0):
            return apy_from_bool(1)
    return apy_from_bool(0)


def apy_math_isfinite(v: ptr) -> ptr:
    """`math.isfinite(x)` -- neither NaN nor an infinity."""
    x: f64 = apy_math_arg_of(v, rodata(b"isfinite\0"))
    if apy_error_occurred():
        return ptr(0)
    if x == x:
        if x - x == f64(0):
            return apy_from_bool(1)
    return apy_from_bool(0)


def apy_math_degrees(v: ptr) -> ptr:
    """`math.degrees(x)`."""
    x: f64 = apy_math_arg_of(v, rodata(b"degrees\0"))
    if apy_error_occurred():
        return ptr(0)
    return apy_from_float(x * (f64(180) / apy_math_pi()))


def apy_math_radians(v: ptr) -> ptr:
    """`math.radians(x)`.

    THE DIVISION IS INSIDE THE PARENTHESES in both directions, which matters:
    `x * (pi / 180)` and `x * pi / 180` can differ in the last bit, and the C
    groups it this way.
    """
    x: f64 = apy_math_arg_of(v, rodata(b"radians\0"))
    if apy_error_occurred():
        return ptr(0)
    return apy_from_float(x * (apy_math_pi() / f64(180)))


def apy_conjugate(v: ptr) -> ptr:
    """`x.conjugate()`.

    A REAL NUMBER IS ITS OWN CONJUGATE and comes back unchanged rather than
    copied -- which is what makes `(5).conjugate() is 5` for a small integer,
    the same as CPython.

    ONLY THE IMAGINARY PART IS NEGATED, and negating zero gives -0.0 for a
    complex whose imaginary part was 0.0. That is arithmetic rather than a
    choice, and it is what `complex(1, 0).conjugate()` shows.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_complex_kind():
        return apy_from_complex(
            load(f64, offset(v, apy_complex_re_offset())),
            -load(f64, offset(v, apy_complex_im_offset())))
    # A BOOL ANSWERS AN int, NOT ITSELF: `True.conjugate()` is `1` in
    # Python, because `bool` inherits the method from `int` and the method
    # is defined to answer an int. The C returned the receiver for every
    # real number alike, so a bool came back a bool.
    if i64(load(i32, offset(v, 0))) == apy_bool_kind():
        return apy_from_int(apy_int_payload(v))
    if apy_is_num_of(v):
        return v
    return apy_raise_fmt(
        rodata(b"AttributeError\0"),
        rodata(b"'%s' object has no attribute 'conjugate'%s\0"),
        apy_kind_name_of(v), rodata(b"\0"))


def apy_index(v: ptr) -> i64:
    """`v` as an index -- what `xs[v]`, `range(v)` and `chr(v)` all read.

    THE BIG TEST COMES FIRST, and it has to: `apy_is_int_like_of` is true for
    a big, so testing it first sent every big down the fast path and returned
    the LIMB POINTER read as an integer. An index that silently inverts is
    worse than one that refuses.

    `__index__` IS NOT `__int__`. A class that defines the first is saying it
    IS an integer, which is what an index position requires; one that defines
    only the second is saying it can be converted, which is not the same
    permission -- `xs[3.7]` is a TypeError for exactly that reason.
    """
    if apy_is_big_of(v):
        apy_raise_fmt(
            rodata(b"OverflowError\0"),
            rodata(b"cannot fit 'int' into an index-sized "
                   b"integer%s%s\0"),
            rodata(b"\0"), rodata(b"\0"))
        return 0
    if apy_is_int_like_of(v):
        return apy_int_payload(v)
    if i64(load(i32, offset(v, 0))) == apy_inst_kind():
        got: ptr = apy_unary_dunder_of(v, rodata(b"__index__\0"))
        if apy_error_occurred():
            return 0
        if got:
            if apy_is_big_of(got):
                apy_raise_fmt(
                    rodata(b"OverflowError\0"),
                    rodata(b"cannot fit 'int' into an index-sized "
                           b"integer%s%s\0"),
                    rodata(b"\0"), rodata(b"\0"))
                return 0
            if apy_is_int_like_of(got):
                return apy_int_payload(got)
    apy_raise_fmt(
        rodata(b"TypeError\0"),
        rodata(b"'%s' object cannot be interpreted as "
               b"an integer%s\0"),
        apy_kind_name_of(v), rodata(b"\0"))
    return 0


def apy_invert(a: ptr) -> ptr:
    """`~a`.

    A BIG INVERTS BY IDENTITY: `~n` is `-(n + 1)` for every integer, which is
    the whole of two's complement written in arithmetic the big path already
    has. Nothing here has to know how a limb is laid out.

    THE USER HOOK IS TRIED FIRST AND ITS 0 MEANS TWO THINGS -- no such method,
    or one that failed -- which the error flag tells apart. That is the
    protocol every operator dispatch in this runtime uses.
    """
    if i64(load(i32, offset(a, 0))) == apy_inst_kind():
        r: ptr = apy_unary_dunder_of(a, rodata(b"__invert__\0"))
        if r:
            return r
        if apy_error_occurred():
            return r
    if not apy_is_int_like_of(a):
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"bad operand type for unary ~: '%s'%s\0"),
            apy_kind_name_of(a), rodata(b"\0"))
    if apy_is_big_of(a):
        return apy_neg(apy_add(a, apy_from_int(1)))
    return apy_from_int(~apy_int_payload(a))


def apy_iter_result_ok(got: ptr) -> i64:
    """Is what `__iter__` answered actually an iterator?

    A GENERATOR OR A CURSOR IS ONE BY CONSTRUCTION. An INSTANCE is one only if
    its class defines `__next__` -- which is the check CPython makes and the
    reason `iter()` can report a non-iterator at all.
    """
    k: i64 = i64(load(i32, offset(got, 0)))
    if k == apy_gen_kind():
        return 1
    if k == apy_iter_kind():
        return 1
    if k == apy_inst_kind():
        if apy_class_find_of(ptr(load(u64, offset(got, apy_o_cls_offset()))),
                             apy_name_of(rodata(b"__next__\0"))):
            return 1
    return 0


def apy_not_an_iterator(got: ptr) -> ptr:
    """`iter()` was given something whose `__iter__` answered a non-iterator."""
    return apy_raise_fmt(
        rodata(b"TypeError\0"),
        rodata(b"iter() returned non-iterator of type '%s'%s\0"),
        apy_kind_name_of(got), rodata(b"\0"))


def apy_not_iterable(v: ptr) -> ptr:
    """`for x in 5:` -- there is nothing to walk."""
    return apy_raise_fmt(
        rodata(b"TypeError\0"),
        rodata(b"'%s' object is not iterable%s\0"),
        apy_kind_name_of(v), rodata(b"\0"))


def apy_getiter(v: ptr) -> ptr:
    """`iter(v)` -- what a `for` will step.

    A GENERATOR AND A CURSOR ARE ALREADY ITERATORS and come back untouched:
    `iter(it) is it` is a rule programs rely on, and wrapping one would make a
    partly-consumed iterator restart.

    A VIEW WALKS WHAT IT IS A VIEW OF, which is why this recurses rather than
    building a cursor over the view: `d.keys()` is a window onto the dict, and
    the window is not the thing with elements in it.

    A METACLASS `__iter__` MAKES THE CLASS ITSELF ITERABLE -- `for x in C` for
    `class C(metaclass=M)` where M defines one. Looked for before the instance
    path, because a TYPE_K is not an INST_K and would otherwise fall through
    to the not-iterable report.

    THE OLD PROTOCOL IS STILL HONOURED: a class with `__getitem__` and no
    `__iter__` is iterable, and is walked from 0 until it reports IndexError.
    A cursor over the object does both, since stepping one reads through
    `apy_getitem`.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_view_kind():
        return apy_getiter(apy_view_items(v))
    if k == apy_gen_kind():
        return v
    if k == apy_iter_kind():
        return v
    # AN ALIAS YIELDS ONE ITEM -- itself, starred. Here as well as in
    # `apy_iterable` because this is the LAZY entry a `for` uses; fixing only
    # the eager one left `for x in list[int]` reporting the alias as not
    # iterable while `list(list[int])` worked.
    if k == apy_alias_kind():
        one: ptr = apy_tuple_new(1)
        apy_q_append_of(one, apy_alias_unpack(v))
        return apy_getiter(one)
    if k == apy_type_kind():
        meta: ptr = ptr(load(u64, offset(v, apy_t_meta_offset())))
        if meta:
            hook: ptr = apy_class_find_of(meta, apy_name_of(rodata(b"__iter__\0")))
            if hook:
                got: ptr = apy_call(apy_bind_of(hook, v), ptr(0), 0)
                if not got:
                    return ptr(0)
                if apy_iter_result_ok(got):
                    return got
                return apy_not_an_iterator(got)
    if k == apy_inst_kind():
        made: ptr = apy_unary_dunder_of(v, rodata(b"__iter__\0"))
        if apy_error_occurred():
            return ptr(0)
        if made:
            if apy_iter_result_ok(made):
                return made
            return apy_not_an_iterator(made)
        held: ptr = ptr(load(u64, offset(v, apy_o_held_offset())))
        if held:
            return apy_getiter(held)
        if not apy_class_find_of(
                ptr(load(u64, offset(v, apy_o_cls_offset()))),
                apy_name_of(rodata(b"__getitem__\0"))):
            return apy_not_iterable(v)
    else:
        # A MEMORYVIEW IS WALKED BY INDEX, which is what it already answers
        # to everywhere else: `list(mv)`, `[*mv]`, `98 in mv` and
        # `reversed(mv)` all worked and `iter(mv)` alone said it was not
        # iterable. The cursor reads through `apy_getitem`, and `mv[i]` is
        # the int CPython yields.
        if not apy_is_seq_of(v) and not apy_is_set_of(v):
            if k != apy_str_kind() and k != apy_bytes_kind():
                if k != apy_dict_kind() and k != apy_range_kind():
                    if k != apy_mview_kind():
                        return apy_not_iterable(v)
    return apy_cursor_of(v, ptr(0), apy_it_plain(), 0)


def apy_slice_bound(v: ptr) -> i64:
    """A slice bound, where `apy_index` would refuse.

    A BOUND CLAMPS AND AN INDEX DOES NOT, which is the whole of the
    difference: `xs[:2 ** 100]` is the whole list in Python, and
    `xs[2 ** 100]` is an error. So a big becomes a huge bound of the right
    SIGN rather than an OverflowError.

    `1 << 62` AND NOT `INT64_MAX`, so that arithmetic on the result -- adding
    a length, negating it -- cannot overflow on the way to being clamped.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_none_kind():
        # AN EXPLICIT `None` IS NOT A BOUND, so there is no number to give --
        # and the caller pairs this with `apy_slice_given`, which says the
        # bound was not given at all. Zero is returned rather than refused
        # because refusing is what `xs[None:None]` used to do.
        return 0
    if apy_is_big_of(v):
        if load(i32, offset(v, apy_big_neg_offset())) != 0:
            return -(i64(1) << 62)
        return i64(1) << 62
    if apy_is_int_like_of(v):
        return apy_int_payload(v)
    if k == apy_inst_kind():
        got: ptr = apy_unary_dunder_of(v, rodata(b"__index__\0"))
        if apy_error_occurred():
            return 0
        if got:
            if apy_is_big_of(got):
                # A BIG FROM `__index__` CLAMPS TOO. The bound rule is about
                # the POSITION, not about where the number came from.
                if load(i32, offset(got, apy_big_neg_offset())) != 0:
                    return -(i64(1) << 62)
                return i64(1) << 62
            if apy_is_int_like_of(got):
                return apy_int_payload(got)
    # NOT `apy_index`'s MESSAGE. CPython words a bad slice bound differently
    # from a bad index, and now that None is accepted the difference matters:
    # its wording NAMES None as one of the things a bound may be.
    apy_raise_at(
        rodata(b"TypeError\0"),
        rodata(b"slice indices must be integers or None or have an "
               b"__index__ method\0"))
    return 0


def apy_slice_given(v: ptr) -> i64:
    """Whether this bound was GIVEN, in the sense a slice means.

    `xs[None:None]` IS `xs[:]`. Python treats an explicitly written `None`
    bound exactly as it treats an omitted one -- which is why
    `slice(None, None, None)` is the object a bare `[:]` builds, and why
    `xs[None::2]` steps from the start rather than raising.

    THE FRONTEND KNOWS WHICH BOUNDS WERE WRITTEN and cannot know which of
    them EVALUATE to None: `xs[a:b]` is a pair of expressions. So it answers
    the first half at compile time and this answers the second at run time,
    and the two are multiplied together into the `has_start`/`has_stop` flags
    `apy_slice` reads.
    """
    # THE KIND TEST IS WRITTEN OUT rather than behind a helper: a helper
    # would be a fourth symbol to define in the C, in the ported runtime and
    # in the interpreter's host table, for one comparison.
    if i64(load(i32, offset(v, 0))) == apy_none_kind():
        return 0
    return 1


def apy_slice_step(v: ptr) -> i64:
    """A slice's step, where `None` means ONE rather than nothing.

    Separate from `apy_slice_bound` because the default differs and because
    a step of zero is an error: `xs[::None]` is `xs[::1]`, and answering 0
    for the None would turn it into one.
    """
    if i64(load(i32, offset(v, 0))) == apy_none_kind():
        return 1
    return apy_slice_bound(v)


def apy_iadd(a: ptr, b: ptr) -> ptr:
    """`a += b`.

    A LIST EXTENDS IN PLACE, which is the whole reason `+=` is not `a = a + b`
    for one: `xs += ys` mutates the list every other name for it also sees,
    and CPython makes the same distinction.

    A CLASS MAY SAY WHAT `+=` MEANS, and `__iadd__` is asked first -- its 0
    means either "no such method" or "it failed", told apart by the error
    flag, as every operator dispatch here does.

    EVERYTHING ELSE IS ORDINARY ADDITION, so `n += 1` on an int is `n + 1`
    and nothing is mutated.
    """
    if i64(load(i32, offset(a, 0))) == apy_inst_kind():
        r: ptr = apy_method1_of(a, rodata(b"__iadd__\0"), b)
        if r:
            return r
        if apy_error_occurred():
            return r
    if i64(load(i32, offset(a, 0))) == apy_list_kind():
        # `xs += it` IS `list.__iadd__`, which is `list_extend` -- so it asks
        # the argument for a length hint, exactly as `xs.extend(it)` does.
        # ASKED HERE rather than through the C's `apy_extend_meth`, which
        # this side may not name: a ported function reaches only the floor
        # and the split halves.
        if not apy_length_hint(b):
            return ptr(0)
        if not apy_extend(a, b):
            return ptr(0)
        return a
    # A BYTEARRAY EXTENDS ITSELF TOO, and for the same reason -- it is
    # mutable, so `b += data` has to be visible through every other name for
    # it. Falling through to `apy_add` built a NEW bytearray and rebound the
    # one name that was written, which is the aliasing bug this function
    # exists to avoid, arrived at from the bytes side.
    #
    # ONLY FROM SOMETHING BYTES-LIKE. `apy_extend` would happily walk a list
    # of ints, and CPython refuses: `can't concat list to bytearray`.
    if apy_is_bytearray_of(a):
        src: ptr = b
        if i64(load(i32, offset(b, 0))) == apy_mview_kind():
            src = apy_mview_bytes(b)
        if i64(load(i32, offset(src, 0))) != apy_bytes_kind():
            return apy_raise_fmt(
                rodata(b"TypeError\0"),
                rodata(b"can't concat %s to bytearray%s\0"),
                apy_kind_name_of(b), rodata(b"\0"))
        if not apy_extend(a, src):
            return ptr(0)
        return a
    return apy_add(a, b)


def apy_big_n_offset() -> i64:
    return 16


def apy_big_limb_offset() -> i64:
    return 8


# -- ordering, from two limbs up to two user classes ------------------------
#
# THREE ANSWERS AND TWO REFUSALS. -1/0/1 order a pair; `apy_unord()` means a
# NaN or two sets that simply stand in no order -- all four comparisons answer
# False and none raises; and 2 means the pair is not orderable at all, which
# the caller turns into a TypeError naming the operator it knows and this does
# not.


def apy_unord() -> i64:
    return 3


def apy_mag_cmp_of(a: ptr, b: ptr) -> i64:
    """Compare two bigs by MAGNITUDE, ignoring both signs.

    THE LIMB COUNT SETTLES IT FIRST, because a big is kept normalised: the
    top limb is never zero, so more limbs is strictly bigger.

    FROM THE TOP DOWN, since the most significant limb that differs is the
    one that decides.
    """
    an: i64 = load(i64, offset(a, apy_big_n_offset()))
    bn: i64 = load(i64, offset(b, apy_big_n_offset()))
    if an != bn:
        if an < bn:
            return -1
        return 1
    alimb: ptr = ptr(load(u64, offset(a, apy_big_limb_offset())))
    blimb: ptr = ptr(load(u64, offset(b, apy_big_limb_offset())))
    i: i64 = an - 1
    while i >= 0:
        # A LIMB IS 32 BITS, not 64. Reading pairs of them as one word gives
        # the right ANSWER on little-endian for most values -- the pair
        # compares in the same order the two limbs do -- and reads twice as
        # far as the array holds, which is a fault waiting for an allocation
        # to land badly. `apy_limb_size` is what the C's `apy_limb` is.
        x: u64 = u64(load(u32, offset(alimb, i * apy_limb_size())))
        y: u64 = u64(load(u32, offset(blimb, i * apy_limb_size())))
        if x != y:
            if x < y:
                return -1
            return 1
        i = i - 1
    return 0


def apy_big_cmp_of(a: ptr, b: ptr) -> i64:
    """Compare two bigs, signs included.

    A DIFFERENCE IN SIGN ENDS IT, and otherwise the magnitude comparison is
    REVERSED for two negatives -- which is the whole of signed ordering.
    """
    an: i64 = i64(load(i32, offset(a, apy_big_neg_offset())))
    bn: i64 = i64(load(i32, offset(b, apy_big_neg_offset())))
    if an != bn:
        if an:
            return -1
        return 1
    c: i64 = apy_mag_cmp_of(a, b)
    if an:
        return -c
    return c


def apy_cmp_int_double_of(i: i64, f: f64) -> i64:
    """Compare an int64 with a double, EXACTLY.

    NOT BY CONVERTING THE INT TO A DOUBLE, which is the point: a double has
    53 bits of mantissa and an int64 has 63 bits of magnitude, so the
    conversion loses the low bits and calls unequal numbers equal.

    2**63 EXACTLY IS THE BOUND, and it is inclusive on one side only: no
    int64 reaches it, and -2**63 is INT64_MIN itself.

    THE FRACTION DECIDES A TIE. With the same integral part, a float with
    anything left over is the larger.
    """
    if apy_isnan_of(f):
        return apy_unord()
    limit: f64 = f64(4611686018427387904) * f64(2)
    if f >= limit:
        return -1
    if f < f64(0) - limit:
        return 1
    fl: f64 = apy_floor_of(f)
    t: i64 = i64(fl)
    if i != t:
        if i < t:
            return -1
        return 1
    if f > fl:
        return -1
    return 0


def apy_either_inst_of(a: ptr, b: ptr) -> i64:
    """Is either side an instance of a program-written class?"""
    if i64(load(i32, offset(a, 0))) == apy_inst_kind():
        return 1
    if i64(load(i32, offset(b, 0))) == apy_inst_kind():
        return 1
    return 0


def apy_binary_dunder_of(a: ptr, b: ptr, name: ptr, rname: ptr) -> ptr:
    """`a.__op__(b)` first, then `b.__rop__(a)`.

    THE REFLECTED FORM IS WHY `1 + v` CAN REACH A USER CLASS AT ALL: the int
    on the left has no idea what `v` is, so the right operand gets the second
    word.

    `NotImplemented` MEANS "ASK THE OTHER OPERAND", not "the answer is
    NotImplemented". Returning it as the result made `Left() == Right()`
    answer the sentinel instead of falling back to Right\'s `__eq__`, and a
    program printing it saw a word where its answer should have been.

    NEITHER SIDE ANSWERING IS A NULL WITH NO ERROR SET, which is how every
    caller here spells "fall back to the default" -- identity for `==`, a
    TypeError for arithmetic.
    """
    r: ptr = apy_method1_of(a, name, b)
    if apy_error_occurred():
        return r
    if r:
        if i64(load(i32, offset(r, 0))) != apy_notimpl_kind():
            return r
    other: ptr = apy_method1_of(b, rname, a)
    if apy_error_occurred():
        return other
    if other:
        if i64(load(i32, offset(other, 0))) != apy_notimpl_kind():
            return other
    return ptr(0)


def apy_order_held(v: ptr, name: ptr, mirror: ptr) -> ptr:
    """The builtin a class extends, when its own body writes neither ordering
    dunder -- and 0 for anything else, which is every caller's "no".

    A CLASS EXTENDING A BUILTIN ORDERS AS THE BUILTIN, the rule equality has
    always followed. `sorted([P(2, 1), P(1, 9)])` on a namedtuple compares two
    TUPLES; without this the walk gave up where the dunders did and reported
    that two of them could not be ordered at all, while `P(2, 1) ==
    P(1, 9)` answered.

    A CLASS THAT WROTE EITHER DUNDER IS LEFT ALONE, because it has already had
    its say and must not be read as its builtin behind its own back.
    """
    if i64(load(i32, offset(v, 0))) != apy_inst_kind():
        return ptr(0)
    held: ptr = ptr(load(u64, offset(v, apy_o_held_offset())))
    if not held:
        return ptr(0)
    cls: ptr = ptr(load(u64, offset(v, apy_o_cls_offset())))
    if apy_class_find_of(cls, apy_name_of(name)):
        return ptr(0)
    if apy_class_find_of(cls, apy_name_of(mirror)):
        return ptr(0)
    return held


def apy_order_error_of(greater: i64, a: ptr, b: ptr) -> ptr:
    """The refusal `sorted`, `min` and `max` report for a pair they could not
    order.

    THE OPERATOR ITSELF WORDS IT. `sorted` and `min` compare with `<` and
    `max` with `>`, and asking that very operator to compare the pair again is
    what keeps this message identical to the one the operator produces -- the
    ELEMENTS two sequences stopped on included, which only `apy_cmp` knows how
    to find. The comparison cannot succeed: `apy_order_rich_of` has already
    answered 2 for this very pair, so what it does is set the error.

    AN ERROR ALREADY SET IS LEFT ALONE. `apy_order_rich_of` answers 2 for a
    `__lt__` that RAISED as well as for one that is missing, and comparing
    again would run the raising method a second time over the first one's
    report.
    """
    if apy_error_occurred():
        return ptr(0)
    if greater:
        apy_gt(a, b)
    else:
        apy_lt(a, b)
    return ptr(0)


def apy_order_mirror_first_of(a: ptr, b: ptr, mirror: ptr) -> i64:
    """Does the RIGHT operand go first? 1 for yes, 0 for the ordinary order.

    CPYTHON'S ONE REORDERING RULE: a right operand whose type is a PROPER
    SUBCLASS of the left's and which overrides the mirror is asked before the
    left's own dunder, the subclass being presumed to know more about the pair
    than its base does. `[1] < OnlyGt([1, 2])` asks `OnlyGt.__gt__` before
    `list.__lt__`.

    WHEN THE LEFT IS NOT AN INSTANCE, "a proper subclass of type(a)" is asking
    whether b extends the very builtin a is one of -- which is exactly the
    case above, and the one the reordering exists for.
    """
    if i64(load(i32, offset(b, 0))) != apy_inst_kind():
        return 0
    bcls: ptr = ptr(load(u64, offset(b, apy_o_cls_offset())))
    if not apy_class_find_of(bcls, apy_name_of(mirror)):
        return 0
    if i64(load(i32, offset(a, 0))) == apy_inst_kind():
        acls: ptr = ptr(load(u64, offset(a, apy_o_cls_offset())))
        if acls == bcls:
            return 0
        return apy_type_is_sub_of(bcls, acls)
    held: ptr = ptr(load(u64, offset(b, apy_o_held_offset())))
    if not held:
        return 0
    if i64(load(i32, offset(held, 0))) == i64(load(i32, offset(a, 0))):
        return 1
    return 0


def apy_written_dunder_of(who: ptr, other: ptr, name: ptr) -> ptr:
    """ONE HALF OF `apy_binary_dunder_of`: the dunder THIS side wrote.

    0 for a class that wrote none and 0 for `NotImplemented`, which means
    "ask the other operand" rather than "the answer is NotImplemented" -- the
    same reading the paired form gives it. Split out so the builtin a class
    extends can be read BETWEEN the two halves rather than after both; see
    `apy_order_lt_of`.
    """
    r: ptr = apy_method1_of(who, name, other)
    if apy_error_occurred():
        return r
    if r:
        if i64(load(i32, offset(r, 0))) != apy_notimpl_kind():
            return r
    return ptr(0)


def apy_order_lt_held_of(a: ptr, b: ptr) -> i64:
    """`a < b` AS THE BUILTINS THE TWO EXTEND: 1, 0, or -1 for "not that
    either", which sends the caller on to the mirror.

    THE LEFT SIDE IS GATED ON THE DIRECT NAME ALONE, which is why
    `apy_order_held` is asked with `__lt__` twice: a class that wrote the
    MIRROR has not had its say about this comparison yet, and reading the
    builtin is exactly what CPython does before reaching that mirror. THE
    RIGHT SIDE IS NOT GATED AT ALL -- it is only an operand here, and
    CPython's `list.__lt__(a, b)` takes any list subclass for `b` without
    consulting its methods.
    """
    ha: ptr = apy_order_held(a, rodata(b"__lt__\0"), rodata(b"__lt__\0"))
    hb: ptr = ptr(0)
    if i64(load(i32, offset(b, 0))) == apy_inst_kind():
        hb = ptr(load(u64, offset(b, apy_o_held_offset())))
    left: ptr = a
    if ha:
        left = ha
    right: ptr = b
    if hb:
        right = hb
    # STILL AN INSTANCE MEANS NOT A BUILTIN COMPARISON. `list.__lt__` would
    # answer NotImplemented for an operand it knows nothing about, which is
    # when CPython goes on to the mirror AND gives it the operands the
    # PROGRAM wrote -- `Sub([1]) < Watcher()` owes `Watcher.__gt__` the Sub
    # and not a bare list. Subsumes the "neither side was unwrapped" check
    # this used to make: if nothing was unwrapped, both are still what they
    # were, and the outer guard only got here because one is an instance.
    if i64(load(i32, offset(left, 0))) == apy_inst_kind():
        return -1
    if i64(load(i32, offset(right, 0))) == apy_inst_kind():
        return -1
    c2: i64 = apy_order_rich_of(left, right)
    if c2 == apy_unord():
        return 0
    if c2 == 2:
        return -1
    if c2 < 0:
        return 1
    return 0


def apy_order_lt_of(a: ptr, b: ptr) -> i64:
    """Is `a < b`? 1 for yes, 0 for no, -1 for NEITHER SIDE ANSWERED -- which
    is distinct from "no", and the caller turns it into the refusal.

    THREE STEPS, AND THE BUILTIN IS THE MIDDLE ONE: the dunder a wrote, the
    builtin a extends, the mirror b wrote. `type(a).__lt__` is the written one
    or -- for a class extending a builtin that wrote none -- the INHERITED
    one, and in CPython an inherited slot wins OUTRIGHT over a `__gt__` the
    other side wrote. Asking both written dunders first and only then reading
    the builtin made `sorted` over a class extending list that wrote only
    `__gt__` order by that method, while `a < b` on the same two objects
    ordered as lists.

    THE C SAYS THIS TWICE, in `apy_cmp` and in its own `apy_order_lt`, and all
    three have to agree: `sorted` compares with the very `<` the operator
    spells. See `apy_order_mirror_first_of` for when the mirror goes first.
    """
    if apy_order_mirror_first_of(a, b, rodata(b"__gt__\0")):
        first: ptr = apy_written_dunder_of(b, a, rodata(b"__gt__\0"))
        if apy_error_occurred():
            return -1
        if first:
            if apy_truth(first):
                return 1
            return 0
        second: ptr = apy_written_dunder_of(a, b, rodata(b"__lt__\0"))
        if apy_error_occurred():
            return -1
        if second:
            if apy_truth(second):
                return 1
            return 0
        return apy_order_lt_held_of(a, b)
    direct: ptr = apy_written_dunder_of(a, b, rodata(b"__lt__\0"))
    # A FAILURE IS NOT A MISSING DUNDER: a `__lt__` that raised has already
    # reported, and going on would run a second comparison over its error.
    if apy_error_occurred():
        return -1
    if direct:
        if apy_truth(direct):
            return 1
        return 0
    held: i64 = apy_order_lt_held_of(a, b)
    if held >= 0:
        return held
    mirror: ptr = apy_written_dunder_of(b, a, rodata(b"__gt__\0"))
    if apy_error_occurred():
        return -1
    if mirror:
        if apy_truth(mirror):
            return 1
        return 0
    return -1


def apy_order_rich_of(a: ptr, b: ptr) -> i64:
    """`apy_order_of` with the program\'s own `__lt__` behind it.

    `apy_order_of` ANSWERS 2 FOR "NOT ORDERABLE TO ME", which is right for an
    int against a str and WRONG for two instances of a class that writes
    `__lt__`. The `<` operator already knew that; `sorted`, `min` and `max`
    called the plain one and reported `unsupported operand type(s) for <`. So
    `Num.THREE < Num.ONE` worked and `sorted([Num.THREE, Num.ONE])` did not,
    for the same two objects.

    THE MIRRORED OPERATOR IS THE REFLECTED NAME: `a < b` falls back to
    `b.__gt__(a)`, because what b is asked is the comparison from its side.
    """
    c: i64 = apy_order_of(a, b)
    if c != 2:
        return c
    if not apy_either_inst_of(a, b):
        return c
    lt: i64 = apy_order_lt_of(a, b)
    if lt < 0:
        return 2
    if lt:
        return -1
    # NOT LESS SAYS NOTHING ABOUT EQUAL VERSUS GREATER, so the other
    # direction is asked the same way rather than assumed.
    gt: i64 = apy_order_lt_of(b, a)
    if gt < 0:
        return 2
    if gt:
        return 1
    return 0


def apy_order_of(a: ptr, b: ptr) -> i64:
    """Order any two values, or say why they cannot be ordered.

    SET ORDERING IS CONTAINMENT, AND IT IS PARTIAL. `{1, 2} < {1, 3}` is
    False and so is `>`, and neither is an error -- the two sets simply stand
    in no order, which is the same outcome a NaN produces and reuses the same
    answer. A set against a NON-set is an ordinary un-orderable pair.

    COMPLEX HAS NO ORDERING, which is the rule that keeps it from being a
    third float: `1j < 2j` is a TypeError, and so is comparing one to a real
    number. Falling through to the numeric path would have compared the real
    parts and answered -- a wrong answer rather than a missing feature.

    SEQUENCES COMPARE LEXICOGRAPHICALLY and only against their OWN kind, so a
    list is never ordered against a tuple.
    """
    if apy_is_set_of(a) and apy_is_set_of(b):
        sub: i64 = apy_subset_of(a, b)
        sup: i64 = apy_subset_of(b, a)
        if sub and sup:
            return 0
        if sub:
            return -1
        if sup:
            return 1
        return apy_unord()
    ka: i64 = i64(load(i32, offset(a, 0)))
    kb: i64 = i64(load(i32, offset(b, 0)))
    if ka == apy_complex_kind() or kb == apy_complex_kind():
        return 2
    if ka == apy_bytes_kind() and kb == apy_bytes_kind():
        return apy_str_cmp_of(a, b)
    if ka == apy_str_kind() and kb == apy_str_kind():
        return apy_str_cmp_of(a, b)
    same_seq: i64 = 0
    if apy_is_seq_of(a) and apy_is_seq_of(b):
        if ka == kb:
            same_seq = 1
    if same_seq:
        an: i64 = load(i64, offset(a, apy_q_n_offset()))
        bn: i64 = load(i64, offset(b, apy_q_n_offset()))
        n: i64 = an
        if bn < an:
            n = bn
        aitems: ptr = ptr(load(u64, offset(a, apy_q_items_offset())))
        bitems: ptr = ptr(load(u64, offset(b, apy_q_items_offset())))
        i: i64 = 0
        while i < n:
            c: i64 = apy_order_rich_of(
                ptr(load(u64, offset(aitems, i * apy_value_size()))),
                ptr(load(u64, offset(bitems, i * apy_value_size()))))
            if c == 2:
                return 2
            if c:
                return c
            i = i + 1
        if an == bn:
            return 0
        if an < bn:
            return -1
        return 1
    if not apy_is_num_of(a) or not apy_is_num_of(b):
        return 2
    return apy_num_order_of(a, b)


def apy_num_order_of(a: ptr, b: ptr) -> i64:
    """Order two numbers.

    SPLIT. Anything involving a BIG against a FLOAT goes back to the C: the
    exact comparison there needs `frexp` and `ldexp` to take a double apart,
    and those are libm. Everything else -- two ints, two floats, an int
    against a float, and a big against an int -- is answered here.

    A BIG AGAINST AN INT64 IS SETTLED BY ITS SIGN, no digits compared: a big
    is outside int64 range by construction. That is also why the pair is
    never equal, and why equality needs no case for it at all.
    """
    biga: i64 = apy_is_big_of(a)
    bigb: i64 = apy_is_big_of(b)
    ka: i64 = i64(load(i32, offset(a, 0)))
    kb: i64 = i64(load(i32, offset(b, 0)))
    fa: i64 = 0
    if ka == apy_float_kind():
        fa = 1
    fb: i64 = 0
    if kb == apy_float_kind():
        fb = 1
    if biga or bigb:
        if biga and bigb:
            return apy_big_cmp_of(a, b)
        if fa or fb:
            return apy_num_order_of_slow(a, b)
        if biga:
            if load(i32, offset(a, apy_big_neg_offset())):
                return -1
            return 1
        if load(i32, offset(b, apy_big_neg_offset())):
            return 1
        return -1
    if fa and fb:
        x: f64 = load(f64, offset(a, apy_float_offset()))
        y: f64 = load(f64, offset(b, apy_float_offset()))
        if apy_isnan_of(x) or apy_isnan_of(y):
            return apy_unord()
        if x < y:
            return -1
        if x > y:
            return 1
        return 0
    if fa:
        c: i64 = apy_cmp_int_double_of(
            apy_int_payload(b), load(f64, offset(a, apy_float_offset())))
        if c == apy_unord():
            return c
        return -c
    if fb:
        return apy_cmp_int_double_of(
            apy_int_payload(a), load(f64, offset(b, apy_float_offset())))
    ia: i64 = apy_int_payload(a)
    ib: i64 = apy_int_payload(b)
    if ia < ib:
        return -1
    if ia > ib:
        return 1
    return 0


# -- max, min and sorted, which are all one comparison ----------------------


def apy_empty_extreme(want_max: i64) -> ptr:
    """`max([])` and `min([])`, which are the same refusal with two names."""
    if want_max:
        return apy_raise_at(
            rodata(b"ValueError\0"),
            rodata(b"max() iterable argument is empty\0"))
    return apy_raise_at(
        rodata(b"ValueError\0"),
        rodata(b"min() iterable argument is empty\0"))


def apy_extreme_n(buf: ptr, n: i64, want_max: i64) -> ptr:
    """`max(a, b, c)` -- the several-argument spelling.

    THE FIRST ARGUMENT IS THE STANDING ANSWER and every later one is compared
    against it, so a tie keeps the EARLIER, which is Python's rule and is what
    makes `max` stable for equal keys.
    """
    if n < 1:
        return apy_raise_at(
            rodata(b"TypeError\0"),
            rodata(b"min expected at least 1 argument\0"))
    best: ptr = ptr(load(u64, buf))
    i: i64 = 1
    while i < n:
        item: ptr = ptr(load(u64, offset(buf, i * apy_value_size())))
        c: i64 = apy_order_rich_of(item, best)
        if c == 2:
            apy_order_error_of(want_max, item, best)
            return ptr(0)
        if want_max:
            if c > 0:
                best = item
        else:
            if c < 0:
                best = item
        i = i + 1
    return best


def apy_extreme_of(seq: ptr, want_max: i64) -> ptr:
    """`max(xs)` and `min(xs)` over one sequence."""
    # A USER ITERATOR IS DRAINED BEFORE THE WALK. The walk below is by index
    # and a class whose `__iter__` answers an object with `__next__` has no
    # index; `apy_iterable` turns one into a list, as it does a generator.
    seq = apy_iterable(seq)
    if not seq:
        return ptr(0)
    n: i64 = apy_raw_len(seq)
    if apy_error_occurred():
        return ptr(0)
    if n == 0:
        return apy_empty_extreme(want_max)
    best: ptr = apy_key_at(seq, 0)
    if not best:
        return ptr(0)
    i: i64 = 1
    while i < n:
        item: ptr = apy_key_at(seq, i)
        if not item:
            return ptr(0)
        c: i64 = apy_order_rich_of(item, best)
        if c == 2:
            apy_order_error_of(want_max, item, best)
            return ptr(0)
        if want_max:
            if c > 0:
                best = item
        else:
            if c < 0:
                best = item
        i = i + 1
    return best


def apy_extreme_by_of(seq: ptr, keyfn: ptr,
                      want_max: i64) -> ptr:
    """`max(xs, key=f)` -- the KEY is compared and the ITEM is answered.

    TWO THINGS ARE CARRIED because they are not the same thing: the key
    decides and the item is what the caller wanted. Keeping only the item
    would mean recomputing `f` on every comparison, which a program can
    observe -- `f` may print, or count.
    """
    # Drained first, so a user iterator can be walked -- see `apy_sorted`.
    seq = apy_iterable(seq)
    if not seq:
        return ptr(0)
    n: i64 = apy_raw_len(seq)
    if apy_error_occurred():
        return ptr(0)
    best: ptr = ptr(0)
    best_key: ptr = ptr(0)
    i: i64 = 0
    while i < n:
        item: ptr = apy_key_at(seq, i)
        if not item:
            return ptr(0)
        k: ptr = item
        if keyfn:
            one: ptr = alloca(8)
            store(u64, u64(item), one)
            k = apy_call(keyfn, one, 1)
        if not k:
            return ptr(0)
        if not best:
            best = item
            best_key = k
        else:
            c: i64 = apy_order_rich_of(k, best_key)
            if c == 2:
                apy_order_error_of(want_max, k, best_key)
                return ptr(0)
            if want_max:
                if c > 0:
                    best = item
                    best_key = k
            else:
                if c < 0:
                    best = item
                    best_key = k
        i = i + 1
    if not best:
        return apy_empty_extreme(want_max)
    return best


def apy_length_hint(src: ptr) -> ptr:
    """`PyObject_LengthHint` -- what `list(x)` and `sorted(x)` ask.

    THE PORT IS HERE BECAUSE `apy_sorted` IS. The IR's `apy_sorted` replaces
    the C's, and a replaced function cannot reach back into the C for a call
    it made -- so the thing it asked for had to come across with it.
    `apy_iadd`, replaced for its own reasons, then had it to hand. The C's
    own callers reach this one, which is what replacing a name means.

    TWO VERBS, IN ORDER, AND ONLY ONE OF THEM: `__len__` first, and
    `__length_hint__` only when there is no `__len__`. A TypeError from
    `__len__` is swallowed and `__length_hint__` tried in its place; any
    other exception is the program's own and propagates.

    ANSWERS 0 ONLY TO PROPAGATE -- the hint itself is advisory and no caller
    reads it. See the C's `apy_length_hint` for why it is observable at all.
    """
    if i64(load(i32, offset(src, apy_kind_offset()))) != apy_inst_kind():
        return apy_none()
    cls: ptr = ptr(load(u64, offset(src, apy_o_cls_offset())))
    if apy_class_find_of(cls, apy_name_of(rodata(b"__len__\0"))):
        got: ptr = apy_unary_dunder_of(src, rodata(b"__len__\0"))
        if got:
            return apy_none()
        if not apy_error_matches(apy_from_cstr(rodata(b"TypeError\0"))):
            return ptr(0)
        apy_error_clear()
    if apy_class_find_of(cls, apy_name_of(rodata(b"__length_hint__\0"))):
        hint: ptr = apy_unary_dunder_of(src, rodata(b"__length_hint__\0"))
        if not hint:
            return ptr(0)
    return apy_none()


def apy_sorted(seq: ptr) -> ptr:
    """`sorted(xs)`, by insertion.

    STABLE, WHICH IS WHAT `c >= 0` SPELLS: an element equal to the one before
    it stops moving, so equal elements keep the order they arrived in. That is
    a promise Python makes and programs rely on -- `sorted` twice by different
    keys is how a secondary sort is written.

    INSERTION AND NOT SOMETHING FASTER because comparison here may run a
    program's `__lt__`, and the constant factor of the sort is small beside
    one of those. The C makes the same choice.
    """
    # ASKED FIRST, ON THE ARGUMENT, before anything unwraps or drains it:
    # CPython sizes the result from the hint and a class that writes one sees
    # the call. See `apy_length_hint`.
    if not apy_length_hint(seq):
        return ptr(0)
    # A USER ITERATOR IS DRAINED BEFORE THE WALK. The walk below is by index
    # and a class whose `__iter__` answers an object with `__next__` has no
    # index; `apy_iterable` turns one into a list, as it does a generator.
    seq = apy_iterable(seq)
    if not seq:
        return ptr(0)
    n: i64 = apy_raw_len(seq)
    if apy_error_occurred():
        return ptr(0)
    out: ptr = apy_seq_new_of(apy_list_kind(), n + 1)
    if not out:
        return out
    i: i64 = 0
    while i < n:
        apy_seq_push(out, apy_key_at(seq, i))
        i = i + 1
    items: ptr = ptr(load(u64, offset(out, apy_q_items_offset())))
    at: i64 = 1
    while at < n:
        key: ptr = ptr(load(u64, offset(items, at * apy_value_size())))
        j: i64 = at - 1
        going: i64 = 1
        while going:
            if j < 0:
                going = 0
            else:
                prev: ptr = ptr(load(u64, offset(
                    items, j * apy_value_size())))
                c: i64 = apy_order_rich_of(key, prev)
                if c == 2:
                    apy_order_error_of(0, key, prev)
                    return ptr(0)
                if c >= 0:
                    going = 0
                else:
                    store(u64, u64(prev),
                          offset(items, (j + 1) * apy_value_size()))
                    j = j - 1
        store(u64, u64(key), offset(items, (j + 1) * apy_value_size()))
        at = at + 1
    return out


def apy_max(seq: ptr) -> ptr:
    """`max(xs)`."""
    return apy_extreme_of(seq, 1)


def apy_min(seq: ptr) -> ptr:
    """`min(xs)`."""
    return apy_extreme_of(seq, 0)


def apy_key_or_none(keyfn: ptr) -> ptr:
    """`key=None` MEANS NO KEY, which is not the same as a key that answers
    None -- the default arrives as the None object and has to become a null
    before the comparison ever sees it."""
    if i64(load(i32, offset(keyfn, 0))) == apy_none_kind():
        return ptr(0)
    return keyfn


def apy_max_by(seq: ptr, keyfn: ptr) -> ptr:
    """`max(xs, key=f)`."""
    return apy_extreme_by_of(seq, apy_key_or_none(keyfn), 1)


def apy_min_by(seq: ptr, keyfn: ptr) -> ptr:
    """`min(xs, key=f)`."""
    return apy_extreme_by_of(seq, apy_key_or_none(keyfn), 0)


def apy_dict_of(args: ptr) -> ptr:
    """`dict()` and `dict(x)` -- the constructor, from a packed argument list.

    NO ARGUMENT IS AN EMPTY DICT and not an error, which is why the count is
    checked before anything is read.
    """
    if load(i64, offset(args, apy_q_n_offset())) == 0:
        return apy_dict_new(1)
    items: ptr = ptr(load(u64, offset(args, apy_q_items_offset())))
    return apy_to_dict(ptr(load(u64, items)))


def apy_dir_names(out: ptr, d: ptr) -> None:
    """Add every key of `d` to `out`, skipping the ones already there.

    THROUGH `apy_set_find_of` ON A LIST, which is a scan and not a hash
    lookup -- the result has to keep insertion order until it is sorted, and
    a name shadowed by a subclass must appear once rather than twice.
    """
    n: i64 = load(i64, offset(d, apy_d_n_offset()))
    keys: ptr = ptr(load(u64, offset(d, apy_d_keys_offset())))
    i: i64 = 0
    while i < n:
        key: ptr = ptr(load(u64, offset(keys, i * apy_value_size())))
        if apy_set_find_of(out, key) < 0:
            apy_q_append_of(out, key)
        i = i + 1


def apy_dir_chain(out: ptr, cls: ptr) -> None:
    """Add the names every class in `cls`\'s base chain defines.

    `__class__` NEEDS NO ARM OF ITS OWN. It is an entry in `object`'s dict --
    a getset descriptor put there by `apy_object_class` -- so the merge below
    lists it like any other name. It was a special push here while the dict
    had nothing to list.
    """
    here: ptr = cls
    going: i64 = 1
    while going:
        if not here:
            going = 0
        elif i64(load(i32, offset(here, 0))) != apy_type_kind():
            going = 0
        else:
            apy_dir_names(out, ptr(load(u64, offset(
                here, apy_t_dict_offset()))))
            here = ptr(load(u64, offset(here, apy_t_base_offset())))
    # AND WHAT THE CHAIN DOES NOT LINK TO. `t.base` runs out at 0: the
    # builtin a class extends is a KIND and not a class, and `object` is not
    # installed as a real base on anything. So the walk above saw a user
    # class's own dict and stopped, and `dir(P)` was the two names its body
    # left behind -- `__doc__` and `__module__` -- where CPython answers
    # twenty-nine.
    #
    # THE SAME TAIL SERVES THE INSTANCE ARM, which walks this same chain:
    # `dir(S("a"))` and `dir(S)` are one list in CPython, and both were two
    # names here.
    if cls:
        if i64(load(i32, offset(cls, 0))) == apy_type_kind():
            base: ptr = apy_builtin_base_name_of(apy_class_builtin_kind(cls))
            if base:
                names: ptr = apy_kind_dir_of(base)
                if names:
                    at: i64 = 0
                    more: i64 = 1
                    while more:
                        one: ptr = offset(names, at)
                        if load(u8, offset(one, 0)) == u8(0):
                            more = 0
                        else:
                            got: ptr = apy_name_of(one)
                            if apy_set_find_of(out, got) < 0:
                                apy_q_append_of(out, got)
                            at = at + apy_cstr_len(one) + 1
            root: ptr = apy_object_class()
            if cls != root:
                apy_dir_names(out, ptr(load(u64, offset(
                    root, apy_t_dict_offset()))))


def apy_builtin_base_name_of(kind: i64) -> ptr:
    """The name of the builtin a class extends, from its kind number.

    THE IR TWIN of the C's `apy_builtin_base_name`, and the list must agree
    with `_BUILTIN_BASE_KIND` in frontends/python/dynamic.py -- which is
    where a base NAME becomes one of these numbers in the first place.
    """
    if kind == apy_str_kind():
        return rodata(b"str\0")
    if kind == apy_bytes_kind():
        return rodata(b"bytes\0")
    if kind == apy_list_kind():
        return rodata(b"list\0")
    if kind == apy_tuple_kind():
        return rodata(b"tuple\0")
    if kind == apy_dict_kind():
        return rodata(b"dict\0")
    if kind == apy_set_kind():
        return rodata(b"set\0")
    return ptr(0)


def apy_type_is_minted(v: ptr) -> i64:
    """Is this TYPE cell one `apy_type_for` minted for a kind, rather than a
    class the program wrote?

    AN EMPTY DICT, NO BASE AND NO METACLASS is what identifies one. A class
    the program wrote is a TYPE cell too, and its body always leaves
    `__module__` behind -- so a class named `list` cannot be mistaken for the
    kind it shares a name with. The C draws the same line in `apy_dir`.
    """
    d: ptr = ptr(load(u64, offset(v, apy_t_dict_offset())))
    if not d:
        return 0
    n: i64 = load(i64, offset(d, apy_d_n_offset()))
    if n != 0:
        return 0
    base: ptr = ptr(load(u64, offset(v, apy_t_base_offset())))
    if base:
        return 0
    meta: ptr = ptr(load(u64, offset(v, apy_t_meta_offset())))
    if meta:
        return 0
    row: ptr = apy_kind_dir_of(apy_str_data(
        ptr(load(u64, offset(v, apy_t_name_offset())))))
    if not row:
        return 0
    return 1


def apy_dir_builtin(out: ptr, v: ptr) -> None:
    """Add every name a BUILT-IN value answers to, from the generated table.

    THE TABLE IS ONE STRING PER KIND, names NUL-separated and ended by an
    empty one -- so the walk is a `strlen` per name and stops on the empty.

    A BUILTIN TYPE IS A FUNC WEARING `is_type`, and `dir(str)` is the same
    list as `dir("")`, so the type's own name is the key rather than the
    word `type` its kind name would give.
    """
    kn: ptr = apy_kind_name_of(v)
    if i64(load(i32, offset(v, 0))) == apy_func_kind():
        if load(i32, offset(v, apy_fn_is_type_offset())):
            kn = apy_str_data(ptr(load(u64, offset(v, apy_fn_name_offset()))))
    # AND A CELL `apy_type_for` MINTED -- `type(iter([]))` -- reads the row
    # its KIND has, for the same reason: `dir(type(it))` and `dir(it)` are
    # one list in CPython. A class the program wrote never reaches here; the
    # caller walks its chain instead.
    if i64(load(i32, offset(v, 0))) == apy_type_kind():
        kn = apy_str_data(ptr(load(u64, offset(v, apy_t_name_offset()))))
    names: ptr = apy_kind_dir_of(kn)
    if not names:
        return
    at: i64 = 0
    going: i64 = 1
    while going:
        here: ptr = offset(names, at)
        if load(u8, offset(here, 0)) == u8(0):
            going = 0
        else:
            apy_q_append_of(out, apy_name_of(here))
            at = at + apy_cstr_len(here) + 1


def apy_dir(v: ptr) -> ptr:
    """`dir(v)` -- the names it answers to, sorted.

    A CLASS MAY SAY WHAT ITS NAMES ARE, and `__dir__` is asked first: that is
    how a proxy lists what it forwards rather than what it holds.

    THE INSTANCE FIRST AND THEN ITS CLASSES, so a name bound on the instance
    is the one that appears -- the same order a lookup takes.

    A BUILT-IN KIND READS A GENERATED TABLE -- `apy_kind_dir_of`, which is
    CPython's own `dir()` read at generation time. There is no class chain
    here to walk, so this answered an EMPTY LIST where CPython lists eighty
    names.
    """
    # EXCEPT FOR AN INSTANCE OF `object` ITSELF. `object`'s own `__dir__` IS
    # this computation and not an override of it; it sits in `object`'s dict
    # because `dir(object)` is that dict's keys, and `object()` finds it
    # there through its class -- so asking it is asking this function again,
    # with no base case. `object` is not installed as a real base on anything
    # (see `apy_object_class`), so a plain `object()` is the whole of what
    # this excludes. CPython draws the same line: `object.__dir__` is a
    # separate body that builds the list rather than calling `dir`.
    if i64(load(i32, offset(v, 0))) == apy_inst_kind():
        own: ptr = ptr(load(u64, offset(v, apy_o_cls_offset())))
        hook: ptr = ptr(0)
        if own != apy_object_class():
            hook = apy_class_find_of(own, apy_name_of(rodata(b"__dir__\0")))
        if hook:
            got: ptr = apy_call(apy_bind_of(hook, v), ptr(0), 0)
            if not got:
                return ptr(0)
            got = apy_iterable(got)
            if not got:
                return ptr(0)
            return apy_sorted(got)
    out: ptr = apy_seq_new_of(apy_list_kind(), 8)
    if not out:
        return out
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_inst_kind():
        apy_dir_names(out, ptr(load(u64, offset(v, apy_o_dict_offset()))))
        apy_dir_chain(out, ptr(load(u64, offset(v, apy_o_cls_offset()))))
    elif k == apy_type_kind():
        # A CELL `apy_type_for` MINTED reads its KIND's row rather than a
        # class chain it has none of -- see `apy_dir_builtin`.
        minted: i64 = apy_type_is_minted(v)
        if minted:
            apy_dir_builtin(out, v)
        else:
            apy_dir_chain(out, v)
    else:
        apy_dir_builtin(out, v)
    return apy_sorted(out)


def apy_extreme_or(seq: ptr, keyfn: ptr, fallback: ptr,
                   want_max: i64) -> ptr:
    """`max(xs, default=d)` and its three siblings.

    A DEFAULT TURNS THE EMPTY CASE INTO A VALUE rather than the ValueError
    `max([])` raises -- which is the only reason the length is measured before
    anything else happens.
    """
    # Drained first, so a user iterator can be walked -- see `apy_sorted`.
    seq = apy_iterable(seq)
    if not seq:
        return ptr(0)
    n: i64 = apy_raw_len(seq)
    if apy_error_occurred():
        return ptr(0)
    if n == 0:
        return fallback
    if i64(load(i32, offset(keyfn, 0))) == apy_none_kind():
        if want_max:
            return apy_max(seq)
        return apy_min(seq)
    if want_max:
        return apy_max_by(seq, keyfn)
    return apy_min_by(seq, keyfn)


def apy_limb_size() -> i64:
    return 4


# -- bin, oct and hex, which are one function three times -------------------


def apy_limb_bits() -> i64:
    """How many bits a limb holds. `apy_limb_size` bytes of them."""
    return apy_limb_size() * 8


def apy_base_digits() -> ptr:
    """The sixteen digits, which every power-of-two base is a prefix of."""
    return rodata(b"0123456789abcdef\0")


def apy_mag_bits_of(a: ptr) -> i64:
    """How many bits the magnitude of `a` actually uses.

    THE TOP LIMB IS NEVER ZERO in a normalised big, so counting its bits and
    adding a full limb for each one below is exact.
    """
    n: i64 = load(i64, offset(a, apy_big_n_offset()))
    if n == 0:
        return 0
    limb: ptr = ptr(load(u64, offset(a, apy_big_limb_offset())))
    top: u64 = u64(load(u32, offset(limb, (n - 1) * apy_limb_size())))
    bits: i64 = (n - 1) * apy_limb_bits()
    while top != u64(0):
        bits = bits + 1
        top = top >> u64(1)
    return bits


def apy_big_base_text_of(o: ptr, bits_per: i64, prefix: ptr) -> ptr:
    """A big in base 2, 8 or 16.

    A POWER-OF-TWO BASE NEEDS NO DIVISION AT ALL: each output digit is a
    fixed run of bits, which is why `bin`, `oct` and `hex` are cheap on a big
    where `str` is quadratic.

    A DIGIT MAY STRADDLE TWO LIMBS -- three bits do not divide thirty-two --
    so the run is read from one limb and topped up from the next whenever it
    starts partway through.
    """
    nbits: i64 = apy_mag_bits_of(o)
    ndig: i64 = 1
    if nbits != 0:
        ndig = (nbits + bits_per - 1) // bits_per
    buf: ptr = apy_alloc_bytes(ndig + 5)
    if not buf:
        return buf
    out: i64 = 0
    if load(i32, offset(o, apy_big_neg_offset())):
        store(u8, u8(45), buf)
        out = 1
    store(u8, load(u8, prefix), offset(buf, out))
    store(u8, load(u8, offset(prefix, 1)), offset(buf, out + 1))
    out = out + 2
    n: i64 = load(i64, offset(o, apy_big_n_offset()))
    limb: ptr = ptr(load(u64, offset(o, apy_big_limb_offset())))
    digits: ptr = apy_base_digits()
    mask: u64 = (u64(1) << u64(bits_per)) - u64(1)
    i: i64 = ndig - 1
    while i >= 0:
        bit: i64 = i * bits_per
        w: i64 = bit // apy_limb_bits()
        off: i64 = bit % apy_limb_bits()
        chunk: u64 = u64(0)
        if w < n:
            chunk = u64(load(u32, offset(limb, w * apy_limb_size()))) >> u64(off)
        if off:
            if w + 1 < n:
                chunk = chunk | (u64(load(u32, offset(
                    limb, (w + 1) * apy_limb_size())))
                    << u64(apy_limb_bits() - off))
        store(u8, load(u8, offset(digits, i64(chunk & mask))),
              offset(buf, out))
        out = out + 1
        i = i - 1
    store(u8, u8(0), offset(buf, out))
    return apy_from_bytes(buf, out)


def apy_base_scratch() -> ptr:
    """Where a machine-sized integer's digits are assembled.

    EIGHTY BYTES IS THE C'S NUMBER and is what base 2 of an int64 needs: 64
    digits, a sign, a two-character prefix and a terminator.
    """
    return reserve("apy_base_scratch_ir", 80)


def apy_base_text_of(v: ptr, bits_per: i64, prefix: ptr, fn: ptr) -> ptr:
    """`bin(v)`, `oct(v)` and `hex(v)` -- one body, three bases.

    `__index__` IS ASKED OF A CLASS, because these take an integer and a
    class that says it IS one should be accepted -- the same permission
    `xs[v]` needs.

    THE DIGITS COME OUT BACKWARDS and are reversed in place, which is what
    shifting right gives you. The PREFIX is written first and is not part of
    the reversal, which is why the run has its own start.
    """
    if i64(load(i32, offset(v, 0))) == apy_inst_kind():
        got: ptr = apy_unary_dunder_of(v, rodata(b"__index__\0"))
        if apy_error_occurred():
            return ptr(0)
        if got:
            if apy_is_int_like_of(got):
                v = got
    if not apy_is_int_like_of(v):
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"%s() argument can't be interpreted as "
                   b"an integer%s\0"),
            fn, rodata(b"\0"))
    if apy_is_big_of(v):
        return apy_big_base_text_of(v, bits_per, prefix)
    raw: i64 = apy_int_payload(v)
    m: u64 = u64(apy_abs64_of(raw))
    buf: ptr = apy_base_scratch()
    out: i64 = 0
    if raw < 0:
        store(u8, u8(45), buf)
        out = 1
    store(u8, load(u8, prefix), offset(buf, out))
    store(u8, load(u8, offset(prefix, 1)), offset(buf, out + 1))
    out = out + 2
    start: i64 = out
    digits: ptr = apy_base_digits()
    mask: u64 = (u64(1) << u64(bits_per)) - u64(1)
    if m == u64(0):
        store(u8, u8(48), offset(buf, out))
        out = out + 1
    while m != u64(0):
        store(u8, load(u8, offset(digits, i64(m & mask))), offset(buf, out))
        out = out + 1
        m = m >> u64(bits_per)
    i: i64 = 0
    half: i64 = (out - start) // 2
    while i < half:
        lo: ptr = offset(buf, start + i)
        hi: ptr = offset(buf, out - 1 - i)
        c: u8 = load(u8, lo)
        store(u8, load(u8, hi), lo)
        store(u8, c, hi)
        i = i + 1
    store(u8, u8(0), offset(buf, out))
    return apy_str_copy_bytes(buf, out)


def apy_bin(v: ptr) -> ptr:
    """`bin(v)`."""
    return apy_base_text_of(v, 1, rodata(b"0b\0"), rodata(b"bin\0"))


def apy_oct(v: ptr) -> ptr:
    """`oct(v)`."""
    return apy_base_text_of(v, 3, rodata(b"0o\0"), rodata(b"oct\0"))


def apy_hex(v: ptr) -> ptr:
    """`hex(v)`."""
    return apy_base_text_of(v, 4, rodata(b"0x\0"), rodata(b"hex\0"))


def apy_bit_length(v: ptr) -> ptr:
    """`n.bit_length()` -- how many bits the MAGNITUDE needs.

    THE SIGN IS NOT COUNTED, which is Python's definition: `(-5).bit_length()`
    is 3, the same as `(5).bit_length()`, because the question is about the
    number and not about how it is stored.

    ZERO IS ZERO BITS, which falls out of the loop rather than needing a case.
    """
    if not apy_is_int_like_of(v):
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute "
                   b"'bit_length'%s\0"),
            apy_kind_name_of(v), rodata(b"\0"))
    if apy_is_big_of(v):
        return apy_from_int(apy_mag_bits_of(v))
    m: u64 = u64(apy_abs64_of(apy_int_payload(v)))
    n: i64 = 0
    while m != u64(0):
        n = n + 1
        m = m >> u64(1)
    return apy_from_int(n)


def apy_big_popcount(o: ptr) -> i64:
    """How many one bits the magnitude of `o` has.

    THE SIGN IS NOT COUNTED, for the same reason `bit_length` does not count
    it: the question is about the number, and a big keeps its sign in a flag
    rather than in the limbs.
    """
    n: i64 = load(i64, offset(o, apy_big_n_offset()))
    limb: ptr = ptr(load(u64, offset(o, apy_big_limb_offset())))
    total: i64 = 0
    i: i64 = 0
    while i < n:
        w: u64 = u64(load(u32, offset(limb, i * apy_limb_size())))
        while w != u64(0):
            total = total + i64(w & u64(1))
            w = w >> u64(1)
        i = i + 1
    return total


def apy_bit_count(v: ptr) -> ptr:
    """`n.bit_count()` -- how many one bits the MAGNITUDE has.

    THE SIGN IS NOT COUNTED, the same rule `bit_length` follows: the question
    is about the number and not about how it is stored, so `(-7).bit_count()`
    is 3 like `(7).bit_count()` rather than counting a sign bit that only
    exists in two's complement.
    """
    if not apy_is_int_like_of(v):
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute "
                   b"'bit_count'%s\0"),
            apy_kind_name_of(v), rodata(b"\0"))
    if apy_is_big_of(v):
        return apy_from_int(apy_big_popcount(v))
    m: u64 = u64(apy_abs64_of(apy_int_payload(v)))
    n: i64 = 0
    while m != u64(0):
        n = n + i64(m & u64(1))
        m = m >> u64(1)
    return apy_from_int(n)
