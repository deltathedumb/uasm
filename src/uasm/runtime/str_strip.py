# The strips and the two removes, in the machine subset.
#
# STAGE 5 OF docs/INERT-RUNTIME.md. Two families here, split on two DIFFERENT
# tests, and the difference between them is the point of this file.
#
# THE STRIPS NEED THE ASCII GATE. `strip` walks in from an end while the
# character it is looking at is in a set, so it asks a question ABOUT EACH
# CHARACTER -- and byte-wise that question is asked of continuation bytes,
# which belong to no set the caller meant. Same gate as `runtime/str_is.py`
# and `runtime/str_case.py`.
#
# THE REMOVES DO NOT. `removeprefix` and `removesuffix` ask whether one whole
# string sits at an end of another, which is the `startswith` question -- and
# a prefix relation over UTF-8 bytes is the same relation as over characters,
# because a valid encoding is prefix-free at character boundaries. The cut
# they make is at that boundary, so the piece they keep is a whole string. No
# gate, and multi-byte input runs the fast path.
#
# THAT ASYMMETRY IS WHY THEY SHARE A FILE. Written apart, the next person to
# add a gate would add it to both.
#
# ── what the strips inherit, and what they do not ──────────────────────────
#
# THE C IS ALREADY BYTE-WISE HERE and the ASCII gate is what keeps that from
# spreading. `apy_in_chars` compares a byte against the bytes of `chars`, and
# with no `chars` it calls `apy_c_space` -- so the C's `strip()` does not
# remove U+00A0, which Python counts as whitespace. Gating rather than
# copying means the fast path is right about everything it accepts, and the
# existing divergence stays exactly where it already was instead of being
# reimplemented into a second place.
#
# A `chars` THAT IS NOT ASCII IS DECLINED for the same reason as the receiver:
# a set of BYTES drawn from a multi-byte character would match the halves of
# other characters.


def apy_strip_span_lo(p: ptr, n: i64, chars: ptr, cn: i64) -> i64:
    """The first index of `p` that is not in the strip set.

    `cn` OF -1 MEANS WHITESPACE, which is how the C spells "no `chars`" --
    it passes a null pointer, and the subset has no null to pass. A length
    cannot be negative, so the sentinel cannot collide with a real set.
    """
    lo: i64 = 0
    while lo < n:
        if not apy_strip_member(i64(load(u8, offset(p, lo))), chars, cn):
            return lo
        lo = lo + 1
    return lo


def apy_strip_span_hi(p: ptr, n: i64, lo: i64, chars: ptr, cn: i64) -> i64:
    """One past the last index of `p` that is not in the strip set.

    STOPS AT `lo` rather than at zero, so a string that is entirely strip
    characters gives an empty result instead of walking back over ground the
    other end already claimed.
    """
    hi: i64 = n
    while hi > lo:
        if not apy_strip_member(i64(load(u8, offset(p, hi - 1))), chars, cn):
            return hi
        hi = hi - 1
    return hi


def apy_strip_member(c: i64, chars: ptr, cn: i64) -> bool:
    """Is byte `c` in the strip set?"""
    if cn < 0:
        return apy_c_space(c) != 0
    i: i64 = 0
    while i < cn:
        if i64(load(u8, offset(chars, i))) == c:
            return True
        i = i + 1
    return False


def apy_str_data(s: ptr) -> ptr:
    """The bytes of a string cell."""
    return ptr(load(u64, offset(s, apy_str_ptr_offset())))


def apy_strip_gate(s: ptr) -> i64:
    """The byte length of `s` if it is a pure-ASCII string, else -1."""
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


def apy_str_slice_new(s: ptr, lo: i64, hi: i64) -> ptr:
    """`s[lo:hi]` as a new string, for indices already known to be sane.

    A COPY, NOT A VIEW, which is what the C does too: `apy_str_slice_of`
    reaches `apy_str_copy`. A view would have to keep the original alive and
    could not carry the NUL that two hundred places in the remaining C expect
    to find at the end of `v.s.p`.
    """
    return apy_str_copy_bytes(offset(apy_str_data(s), lo), hi - lo)


def apy_strip_run(s: ptr, chars: ptr, cn: i64, left: i64, right: i64) -> ptr:
    """The body all six strips share.

    NOTHING STRIPPED IS THE RECEIVER ITSELF, not a copy of it that compares
    equal. `lo` and `hi` still spanning the whole of `s` is what "there was
    nothing to do" MEANS for a strip, and CPython's `do_strip` ends on that
    same test -- `if (i == 0 && j == len && PyBytes_CheckExact(self))
    return self`, with `unicode_strip` carrying its twin -- so `s.strip() is
    s` and `b.strip() is b` are both True there, for all six spellings and
    with `chars` or without. The C half tests it at the tail of
    `apy_str_trim`, which is what the second rule of this runtime asks: a
    shortcut the fast half takes must hold on the slow side too, or the
    answer would depend on which half ran.

    THE TEST BELONGS HERE AND NOT IN `apy_str_slice_new`. That is the
    general slicer, and `split` and `partition` call it for pieces they are
    about to hand out separately; whether there was nothing to do is a
    question only the method can answer, so each method spells it at its own
    tail. The C says the same about `apy_str_slice_of`.

    THE PREDICATE IS SPELLED THOUGH THE GATE ABOVE ALREADY SETTLES IT: every
    caller here has passed `apy_strip_gate`, so the receiver is exactly a
    str and a bytearray cannot reach this. Asking anyway costs one compare
    and keeps the rule with the shortcut, where a later caller of this
    shared body will read it.
    """
    n: i64 = apy_str_byte_len(s)
    lo: i64 = 0
    hi: i64 = n
    p: ptr = apy_str_data(s)
    if left:
        lo = apy_strip_span_lo(p, n, chars, cn)
    if right:
        hi = apy_strip_span_hi(p, n, lo, chars, cn)
    if lo == 0 and hi == n and apy_str_may_return_self_of(s):
        return s
    return apy_str_slice_new(s, lo, hi)


def apy_strip_chars_len(chars: ptr) -> i64:
    """The byte length of a `chars` argument, or -1 for whitespace, or -2.

    -2 IS "SEND THIS BACK": `chars` is neither None nor a pure-ASCII string,
    so either it is the TypeError the C words itself, or it is a set of
    non-ASCII bytes this cannot reason about. Both go to the C, and the two
    negatives are kept apart so the caller's test reads as what it means.
    """
    if chars == apy_none():
        return -1
    n: i64 = apy_strip_gate(chars)
    if n < 0:
        return -2
    return n


def apy_str_strip(s: ptr) -> ptr:
    """`s.strip()` -- whitespace from both ends."""
    if apy_strip_gate(s) < 0:
        return apy_str_strip_slow(s)
    return apy_strip_run(s, s, -1, 1, 1)


def apy_str_lstrip(s: ptr) -> ptr:
    """`s.lstrip()`."""
    if apy_strip_gate(s) < 0:
        return apy_str_lstrip_slow(s)
    return apy_strip_run(s, s, -1, 1, 0)


def apy_str_rstrip(s: ptr) -> ptr:
    """`s.rstrip()`."""
    if apy_strip_gate(s) < 0:
        return apy_str_rstrip_slow(s)
    return apy_strip_run(s, s, -1, 0, 1)


def apy_str_strip_chars(s: ptr, chars: ptr) -> ptr:
    """`s.strip(chars)`, where `chars` may be None.

    `chars` IS A SET, NOT A PREFIX -- `'xyzzy'.strip('xy')` is 'zz', because
    every leading and trailing character that appears ANYWHERE in the
    argument is removed. Reading it as a prefix would leave 'zzy'.
    """
    if apy_strip_gate(s) < 0:
        return apy_str_strip_chars_slow(s, chars)
    cn: i64 = apy_strip_chars_len(chars)
    if cn == -2:
        return apy_str_strip_chars_slow(s, chars)
    return apy_strip_run(s, apy_str_data(chars), cn, 1, 1)


def apy_str_lstrip_chars(s: ptr, chars: ptr) -> ptr:
    """`s.lstrip(chars)`."""
    if apy_strip_gate(s) < 0:
        return apy_str_lstrip_chars_slow(s, chars)
    cn: i64 = apy_strip_chars_len(chars)
    if cn == -2:
        return apy_str_lstrip_chars_slow(s, chars)
    return apy_strip_run(s, apy_str_data(chars), cn, 1, 0)


def apy_str_rstrip_chars(s: ptr, chars: ptr) -> ptr:
    """`s.rstrip(chars)`."""
    if apy_strip_gate(s) < 0:
        return apy_str_rstrip_chars_slow(s, chars)
    cn: i64 = apy_strip_chars_len(chars)
    if cn == -2:
        return apy_str_rstrip_chars_slow(s, chars)
    return apy_strip_run(s, apy_str_data(chars), cn, 0, 1)


# ── the two removes, which need no gate ────────────────────────────────────


def apy_str_removeprefix(s: ptr, p: ptr) -> ptr:
    """`s.removeprefix(p)`.

    AN EMPTY PREFIX REMOVES NOTHING and the receiver comes back unchanged,
    which the C spells as `O(p)->v.s.n &&` ahead of the comparison. Without
    it an empty prefix would still take the copying path -- the same string,
    reallocated, for no reason.

    THE UNCHANGED CASE RETURNS `s` ITSELF, not a copy of it, and all three
    ways of reaching it do -- an empty prefix, one longer than the receiver,
    and one that simply is not there. Sharing the cell is what the C does and
    costs nothing, because the two gates below admit only an exactly-str
    receiver and a str cannot be written into.

    A BYTEARRAY WOULD BE A DIFFERENT MATTER, and it is the reason those gates
    are worth naming here: handing one back leaves the program holding two
    names for one writable buffer, so the answer it kept would change under
    it at the next `ba[0] = ...`. CPython copies for a mutable receiver --
    stringlib's `return_self` is a fresh object in the instantiation
    bytearray's methods come from -- so
    `bytearray(b"abc").removeprefix(b"z") is` it is False there. A bytearray
    declines to the C below, which copies for every receiver its own
    `apy_str_may_return_self` refuses.
    """
    if not apy_is_str(s):
        return apy_str_removeprefix_slow(s, p)
    if not apy_is_str(p):
        return apy_str_removeprefix_slow(s, p)
    n: i64 = apy_str_byte_len(s)
    m: i64 = apy_str_byte_len(p)
    if m == 0:
        return s
    if m > n:
        return s
    if not apy_bytes_equal_at(apy_str_data(s), 0, apy_str_data(p), m):
        return s
    return apy_str_slice_new(s, m, n)


def apy_str_removesuffix(s: ptr, p: ptr) -> ptr:
    """`s.removesuffix(p)` -- the mirror, cutting at `n - m` instead.

    ITS THREE `return s` PATHS ARE THE OTHER'S, and so is the reason they are
    allowed to share the cell rather than copy it: see `apy_str_removeprefix`
    above for why the gates, and not the method, are what make that safe.
    """
    if not apy_is_str(s):
        return apy_str_removesuffix_slow(s, p)
    if not apy_is_str(p):
        return apy_str_removesuffix_slow(s, p)
    n: i64 = apy_str_byte_len(s)
    m: i64 = apy_str_byte_len(p)
    if m == 0:
        return s
    if m > n:
        return s
    if not apy_bytes_equal_at(apy_str_data(s), n - m,
                              apy_str_data(p), m):
        return s
    return apy_str_slice_new(s, 0, n - m)
