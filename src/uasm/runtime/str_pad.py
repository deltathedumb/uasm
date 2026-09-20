# `replace`, and the justify family, in the machine subset.
#
# STAGE 5 OF docs/INERT-RUNTIME.md, and the last of the string families that
# `runtime/str_join.py` left. Both build a string whose length is not the
# receiver's, so both need the size known before a byte is written -- which is
# the one thing this file spends its care on.
#
# THEY SPLIT ON DIFFERENT TESTS, again, and again it is worth saying which:
#
#   `replace` NEEDS NO GATE. Its needle is a whole string, so finding it
#   byte-wise finds it exactly where it is character-wise, and every piece it
#   copies is cut at a character boundary. Same argument as `str_split.py`.
#
#   THE JUSTIFIES NEED ONE, because a width is counted in CHARACTERS.
#   `'é'.ljust(3)` is three characters, and a byte-wise pad would make it
#   three BYTES -- one character of padding instead of two. The C has this
#   wrong today; gating means the fast path is right about what it accepts and
#   the existing divergence stays where it already was.


def apy_str_pad_gate(s: ptr) -> i64:
    """The byte length of `s` if it is a pure-ASCII string, else -1.

    WHERE THE GATE EARNS ITS KEEP: below 0x80 the byte length IS the
    character count, so the `w <= n` test and the `w - n` padding are both in
    the unit Python means them to be in. Nothing else in these functions has
    to know that characters exist.
    """
    if not apy_is_str(s):
        return -1
    n: i64 = apy_str_byte_len(s)
    p: ptr = apy_str_data(s)
    i: i64 = 0
    while i < n:
        if load(u8, offset(p, i)) > u8(127):
            return -1
        i = i + 1
    return n


def apy_str_width_arg(w: ptr) -> i64:
    """A width argument as a machine integer, or -1 if it is not one.

    -1 IS SAFE AS A SENTINEL because a negative width is never wider than the
    receiver, so the C would answer the receiver unchanged -- the same thing
    declining does. A real negative width therefore reaches the C and gets
    the same answer it always did.
    """
    if not apy_is_int(w):
        return -1
    return apy_int_payload(w)


def apy_str_fill_byte(fill: ptr) -> i64:
    """The single ASCII byte of a fill argument, or -1.

    ONE BYTE, WHICH IS ONE CHARACTER ONLY BECAUSE OF THE `> 127` TEST. The C
    checks that the fill is one BYTE long, which quietly rejects a fill like
    'e' with an accent; that is its business, and this declines such a fill
    rather than deciding differently about it.
    """
    if not apy_is_str(fill):
        return -1
    if apy_str_byte_len(fill) != 1:
        return -1
    c: i64 = i64(load(u8, apy_str_data(fill)))
    if c > 127:
        return -1
    return c


def apy_str_pad_build(s: ptr, n: i64, w: i64, c: i64, left: i64) -> ptr:
    """`s` in a field of `w` bytes of `c`, starting at `left`."""
    buf: ptr = apy_alloc_bytes(w + 1)
    if not buf:
        return buf
    i: i64 = 0
    while i < w:
        store(u8, u8(c), offset(buf, i))
        i = i + 1
    apy_bytes_move(buf, left, apy_str_data(s), n)
    store(u8, u8(0), offset(buf, w))
    return apy_from_bytes(buf, w)


def apy_str_pad(s: ptr, w: i64, c: i64, how: i64) -> ptr:
    """The body all six justifies share. `how` is 0 left, 1 right, 2 centre.

    ALREADY WIDE ENOUGH RETURNS THE RECEIVER, not a copy of it -- which is
    what Python does and what the C does, and it is why `w <= n` is tested
    before anything is allocated.

    THE CENTRE SPLIT IS NOT `pad / 2`. CPython biases the extra character to
    the RIGHT for an even width and to the LEFT for an odd one, so
    `'ab'.center(7, '*')` is `'***ab**'` and `'ab'.center(3)` is `' ab'`.
    Halving alone gets both of those backwards, which is why the parity of
    the WIDTH appears in a calculation that looks like it should only care
    about the padding.
    """
    n: i64 = apy_str_byte_len(s)
    if w <= n:
        return s
    pad: i64 = w - n
    left: i64 = 0
    if how == 1:
        left = pad
    elif how == 2:
        left = (pad >> 1) + (pad & w & 1)
    return apy_str_pad_build(s, n, w, c, left)


def apy_str_ljust(s: ptr, w: ptr) -> ptr:
    """`s.ljust(width)`."""
    n: i64 = apy_str_pad_gate(s)
    width: i64 = apy_str_width_arg(w)
    if n < 0 or width < 0:
        return apy_str_ljust_slow(s, w)
    return apy_str_pad(s, width, 32, 0)


def apy_str_rjust(s: ptr, w: ptr) -> ptr:
    """`s.rjust(width)`."""
    n: i64 = apy_str_pad_gate(s)
    width: i64 = apy_str_width_arg(w)
    if n < 0 or width < 0:
        return apy_str_rjust_slow(s, w)
    return apy_str_pad(s, width, 32, 1)


def apy_str_center(s: ptr, w: ptr) -> ptr:
    """`s.center(width)`."""
    n: i64 = apy_str_pad_gate(s)
    width: i64 = apy_str_width_arg(w)
    if n < 0 or width < 0:
        return apy_str_center_slow(s, w)
    return apy_str_pad(s, width, 32, 2)


def apy_str_ljust_fill(s: ptr, w: ptr, f: ptr) -> ptr:
    """`s.ljust(width, fill)`."""
    n: i64 = apy_str_pad_gate(s)
    width: i64 = apy_str_width_arg(w)
    c: i64 = apy_str_fill_byte(f)
    if n < 0 or width < 0 or c < 0:
        return apy_str_ljust_fill_slow(s, w, f)
    return apy_str_pad(s, width, c, 0)


def apy_str_rjust_fill(s: ptr, w: ptr, f: ptr) -> ptr:
    """`s.rjust(width, fill)`."""
    n: i64 = apy_str_pad_gate(s)
    width: i64 = apy_str_width_arg(w)
    c: i64 = apy_str_fill_byte(f)
    if n < 0 or width < 0 or c < 0:
        return apy_str_rjust_fill_slow(s, w, f)
    return apy_str_pad(s, width, c, 1)


def apy_str_center_fill(s: ptr, w: ptr, f: ptr) -> ptr:
    """`s.center(width, fill)`."""
    n: i64 = apy_str_pad_gate(s)
    width: i64 = apy_str_width_arg(w)
    c: i64 = apy_str_fill_byte(f)
    if n < 0 or width < 0 or c < 0:
        return apy_str_center_fill_slow(s, w, f)
    return apy_str_pad(s, width, c, 2)


def apy_str_zfill(s: ptr, w: ptr) -> ptr:
    """`s.zfill(width)` -- zeros on the left, but AFTER any sign.

    THE SIGN IS WHY THIS IS NOT `rjust(width, '0')`. `'-5'.zfill(4)` is
    `'-005'` and not `'00-5'`: a leading `-` or `+` stays at the front and the
    zeros go behind it, because the result is meant to still read as the same
    number.
    """
    n: i64 = apy_str_pad_gate(s)
    width: i64 = apy_str_width_arg(w)
    if n < 0 or width < 0:
        return apy_str_zfill_slow(s, w)
    if width <= n:
        return s
    p: ptr = apy_str_data(s)
    signed_: i64 = 0
    if n > 0:
        c: i64 = i64(load(u8, p))
        if c == 45 or c == 43:
            signed_ = 1
    buf: ptr = apy_alloc_bytes(width + 1)
    if not buf:
        return apy_str_zfill_slow(s, w)
    i: i64 = 0
    while i < width:
        store(u8, u8(48), offset(buf, i))
        i = i + 1
    if signed_:
        store(u8, load(u8, p), offset(buf, 0))
        apy_bytes_move(buf, width - n + 1, offset(p, 1), n - 1)
    else:
        apy_bytes_move(buf, width - n, p, n)
    store(u8, u8(0), offset(buf, width))
    return apy_from_bytes(buf, width)


# ── replace ────────────────────────────────────────────────────────────────


def apy_str_replace_hits(s: ptr, old: ptr, n: i64, m: i64) -> i64:
    """How many non-overlapping occurrences of `old` are in `s`.

    COUNTED BEFORE ANYTHING IS BUILT, so the buffer is asked for exactly the
    bytes the answer needs. The C guesses `(n + 1) * (k + 1) + n + 1`, which
    is safe and can be many times the real size; here the count is already
    the walk the build does, so doing it twice costs less than over-asking
    once.

    NON-OVERLAPPING IS THE DEFINITION: `'aaa'.replace('aa', 'b')` is `'ba'`,
    not `'bb'` -- so a match advances by `m` and only a miss advances by one.
    """
    hits: i64 = 0
    p: ptr = apy_str_data(s)
    q: ptr = apy_str_data(old)
    i: i64 = 0
    while i + m <= n:
        if apy_bytes_equal_at(p, i, q, m):
            hits = hits + 1
            i = i + m
        else:
            i = i + 1
    return hits


def apy_str_replace(s: ptr, old: ptr, new_: ptr) -> ptr:
    """`s.replace(old, new)` with no limit, for three plain strings.

    AN EMPTY `old` IS DECLINED and it is the one case here that would need
    the gate: `''` matches BETWEEN CHARACTERS, so `'日'.replace('', '-')` is
    `'-日-'` and a byte-wise version would put a dash between the three bytes
    of the character as well. The C handles it in its own branch; this sends
    it there rather than growing a second rule.
    """
    if not apy_is_str(s):
        return apy_str_replace_slow(s, old, new_)
    if not apy_is_str(old):
        return apy_str_replace_slow(s, old, new_)
    if not apy_is_str(new_):
        return apy_str_replace_slow(s, old, new_)
    n: i64 = apy_str_byte_len(s)
    m: i64 = apy_str_byte_len(old)
    if m == 0:
        return apy_str_replace_slow(s, old, new_)
    # `old` AND `new` BEING THE SAME OBJECT REPLACES NOTHING VISIBLE, and
    # CPython shortcuts it for str and not for bytes: `unicode_replace`
    # opens with `if (str1 == str2) goto nothing;` and the stringlib every
    # bytes-like shares has no such test, so `"abc".replace("a", "a") is
    # "abc"` is True while `b"abc".replace(b"a", b"a") is b"abc"` is False,
    # with the same literal written on both sides. THAT ASYMMETRY IS
    # CPYTHON'S; do not 'fix' it. No kind test is needed to keep the two
    # apart here because the three gates above admit only str -- a bytes
    # receiver goes to the C, which is already right about it.
    #
    # THE TEST IS IDENTITY AND NOT EQUALITY, there as here: two equal
    # strings that are not the same object build the copy, which is why this
    # compares the cells rather than their bytes.
    if old == new_:
        return s
    hits: i64 = apy_str_replace_hits(s, old, n, m)
    # AND THE COMMONEST WAY OF ALL, which only the count can report: the
    # needle was not there. CPython reaches a `goto nothing` or a
    # `return_self` for that in both implementations, so it is True for str
    # and bytes alike.
    if hits == 0:
        return s
    k: i64 = apy_str_byte_len(new_)
    total: i64 = n - hits * m + hits * k
    buf: ptr = apy_alloc_bytes(total + 1)
    if not buf:
        return apy_str_replace_slow(s, old, new_)
    p: ptr = apy_str_data(s)
    q: ptr = apy_str_data(old)
    r: ptr = apy_str_data(new_)
    out: i64 = 0
    i: i64 = 0
    while i < n:
        if i + m <= n and apy_bytes_equal_at(p, i, q, m):
            out = apy_bytes_move(buf, out, r, k)
            i = i + m
        else:
            store(u8, load(u8, offset(p, i)), offset(buf, out))
            out = out + 1
            i = i + 1
    store(u8, u8(0), offset(buf, total))
    return apy_from_bytes(buf, total)
