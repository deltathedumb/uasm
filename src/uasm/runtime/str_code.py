# `ord` and `chr`, written in the machine subset.
#
# STAGE 5 OF docs/INERT-RUNTIME.md, the third str step and the first ported
# code that ALLOCATES A BUFFER and hands it to the rest of the runtime. Every
# earlier step either filled a cell the arena had already sized (`str_cell.py`)
# or only read one (`str_len.py`).
#
# WHY THAT MATTERS MORE THAN THE TWO FUNCTIONS DO. A bump-pointer arena cannot
# free, so the question every later kind runs into -- who owns the bytes, and
# what happens when nobody can give them back -- is answered here at the
# smallest possible scale: `chr` needs at most five bytes and they are
# immortal, which is exactly the case a bump pointer is right for.
#
# THE ERROR PATHS ARE DECLINED, NOT REIMPLEMENTED. `apy_fail` is not reachable
# from the subset, and that turns out to be the right shape anyway: the fast
# path handles the cases that ARE a character or ARE a code point, and hands
# everything else to the C, which already owns the wording of five different
# messages. A fast path that had to be total could not begin -- the same
# argument `int_arith.py` makes about `apy_add` being polymorphic over
# eighteen kinds.


def apy_utf8_width(lead: i64) -> i64:
    """How many bytes the sequence starting with `lead` occupies.

    ZERO IS NOT A WIDTH, it is "this is not a lead byte" -- a stray
    continuation, which the C treats as a one-byte character with the raw
    value. This answers 1 for it too, and the CALLER decides: `apy_ord`
    compares the width against the whole length, so a malformed string is
    simply not one character and declines.
    """
    if lead < 128:
        return 1
    if (lead & 224) == 192:
        return 2
    if (lead & 240) == 224:
        return 3
    if (lead & 248) == 240:
        return 4
    return 1


def apy_utf8_lead_bits(lead: i64, width: i64) -> i64:
    """The payload bits a lead byte contributes, by width."""
    if width == 1:
        return lead
    if width == 2:
        return lead & 31
    if width == 3:
        return lead & 15
    return lead & 7


def apy_ord(v: ptr) -> ptr:
    """`ord(s)` -- the code point of a one-CHARACTER string.

    ONE CHARACTER, NOT ONE BYTE, and the two stopped coinciding when `chr`
    learned to build a multi-byte one. The length test counts a sequence and
    the answer DECODES it; testing bytes made `ord(chr(233))` a TypeError
    about a string of length != 1, describing a string the program never
    wrote. The C carries that comment and this has to keep its bargain.
    """
    if apy_is_bytes(v):
        if apy_str_byte_len(v) == 1:
            p: ptr = ptr(load(u64, offset(v, apy_str_ptr_offset())))
            return apy_from_int(i64(load(u8, p)))
        return apy_ord_slow(v)
    if not apy_is_str(v):
        return apy_ord_slow(v)
    n: i64 = apy_str_byte_len(v)
    if n < 1:
        return apy_ord_slow(v)
    at: ptr = ptr(load(u64, offset(v, apy_str_ptr_offset())))
    lead: i64 = i64(load(u8, at))
    width: i64 = apy_utf8_width(lead)
    # NOT ONE CHARACTER, so it is not this function's business. A truncated
    # sequence, a string of two characters and an empty one all arrive here.
    if n != width:
        return apy_ord_slow(v)
    code: i64 = apy_utf8_lead_bits(lead, width)
    i: i64 = 1
    while i < width:
        code = (code << 6) | (i64(load(u8, offset(at, i))) & 63)
        i = i + 1
    return apy_from_int(code)


def apy_chr(v: ptr) -> ptr:
    """`chr(i)` -- the one-character string for a code point.

    UTF-8, because that is how a str is stored here, so a code point becomes
    one to four bytes and `len` counts characters by decoding them again.

    THE TRAILING NUL IS WRITTEN and is not part of the length. Two hundred
    places in the remaining C read `v.s.p` as a C string -- `APY_CSTR`,
    `strcmp`, `snprintf` -- so a cell built without a terminator is a cell the
    rest of the runtime reads off the end of.

    BOOL DECLINES. `chr(True)` is `chr(1)` in Python and a bool is a different
    kind here; letting it through would be right for the value and wrong for
    nothing else, which is the worst kind of nearly-right. The C knows the
    whole rule, so the fast path takes exact ints and lets the rest go.
    """
    if not apy_is_int(v):
        return apy_chr_slow(v)
    code: i64 = apy_int_payload(v)
    if code < 0 or code > 1114111:
        return apy_chr_slow(v)
    n: i64 = 1
    if code >= 128:
        n = 2
    if code >= 2048:
        n = 3
    if code >= 65536:
        n = 4
    buf: ptr = apy_alloc_bytes(n + 1)
    if not buf:
        return buf
    if n == 1:
        store(u8, u8(code), buf)
    elif n == 2:
        store(u8, u8(192 | (code >> 6)), buf)
        store(u8, u8(128 | (code & 63)), offset(buf, 1))
    elif n == 3:
        store(u8, u8(224 | (code >> 12)), buf)
        store(u8, u8(128 | ((code >> 6) & 63)), offset(buf, 1))
        store(u8, u8(128 | (code & 63)), offset(buf, 2))
    else:
        store(u8, u8(240 | (code >> 18)), buf)
        store(u8, u8(128 | ((code >> 12) & 63)), offset(buf, 1))
        store(u8, u8(128 | ((code >> 6) & 63)), offset(buf, 2))
        store(u8, u8(128 | (code & 63)), offset(buf, 3))
    store(u8, u8(0), offset(buf, n))
    return apy_from_bytes(buf, n)


# -- the pieces `repr` stands on --------------------------------------------


def apy_repr_state() -> ptr:
    """The recursion guard: a depth and sixty-four values being shown.

    A CONTAINER THAT HOLDS ITSELF would recur forever -- `xs = []; xs.append
    (xs); repr(xs)` -- and CPython prints `[...]` at the point it notices.
    This is what notices: a value already on the stack is reported as
    entered, and the caller prints the ellipsis instead of descending.

    SIXTY-FOUR AND THEN NOTHING. Past that depth the guard stops recording
    and a deeper structure simply is not protected -- the C's limit, kept,
    because a repr nested deeper than sixty-four is a different problem.
    """
    return reserve("apy_repr_state_ir", 520)


def apy_repr_entered(v: ptr) -> i64:
    """Is `v` already being shown? Records it if not."""
    depth: i64 = load(i64, apy_repr_state())
    active: ptr = offset(apy_repr_state(), 8)
    i: i64 = 0
    while i < depth:
        if ptr(load(u64, offset(active, i * apy_value_size()))) == v:
            return 1
        i = i + 1
    if depth < 64:
        store(u64, u64(v), offset(active, depth * apy_value_size()))
        store(i64, depth + 1, apy_repr_state())
    return 0


def apy_repr_left(v: ptr) -> None:
    """Done showing `v`.

    ONLY IF IT IS THE TOP one, which is what makes an unbalanced pair safe:
    a formatter that returned early without leaving would otherwise pop
    someone else's entry.
    """
    depth: i64 = load(i64, apy_repr_state())
    if depth > 0:
        active: ptr = offset(apy_repr_state(), 8)
        if ptr(load(u64, offset(active, (depth - 1) * apy_value_size()))) == v:
            store(i64, depth - 1, apy_repr_state())


def apy_special_form_slot() -> ptr:
    """Where the one `_SpecialForm` class lives."""
    return reserve("apy_special_form_ir", 8)


def apy_special_form_class() -> ptr:
    """The class every interned typing form is an instance of.

    ONE CLASS FOR ALL OF THEM, which is what `apy_is_special_form` tests
    against: `Literal`, `TypeGuard` and their siblings are told apart by
    identity, and what they have in common is that subscripting one
    PARAMETERISES it rather than looking anything up.
    """
    held: ptr = ptr(load(u64, apy_special_form_slot()))
    if held:
        return held
    cls: ptr = apy_type_new(apy_from_cstr(rodata(b"_SpecialForm\0")),
                            ptr(0))
    if cls:
        store(u64, u64(cls), apy_special_form_slot())
    return cls


def apy_is_special_form(v: ptr) -> i64:
    """Is `v` one of the interned typing forms?"""
    if i64(load(i32, offset(v, 0))) != apy_inst_kind():
        return 0
    if ptr(load(u64, offset(v, apy_o_cls_offset()))) == \
            apy_special_form_class():
        return 1
    return 0


def apy_exc_shown_of(name: ptr) -> ptr:
    """The name an exception SHOWS, which is not always the one it MATCHES.

    A BUNDLED MODULE'S CLASSES ARE SPLICED UNDER MANGLED NAMES and then
    restore `__name__`, precisely so the mangling stays invisible. The cell
    still carries the mangled spelling, so a repr built from it would print
    the splice's name -- which is why the class is asked instead when there
    is one.
    """
    cls: ptr = apy_exc_class_named_of(name)
    if not cls:
        return name
    return ptr(load(u64, offset(
        ptr(load(u64, offset(cls, apy_t_name_offset()))),
        apy_str_ptr_offset())))


def apy_text_result_of(r: ptr, which: ptr) -> ptr:
    """What a user `__repr__` or `__str__` answered, if it answered a string.

    A DUNDER THAT RETURNS A NON-STRING IS AN ERROR AND NOT A COERCION, which
    is CPython's rule: `__repr__` returning 5 is a TypeError naming the
    method, rather than `5` quietly becoming `"5"`.
    """
    if i64(load(i32, offset(r, 0))) == apy_str_kind():
        return r
    return apy_raise_fmt(
        rodata(b"TypeError\0"),
        rodata(b"%s returned non-string (type %s)\0"),
        which, apy_kind_name_of(r))


def apy_utf8_step_of(p: ptr, n: i64, i: i64,
                     out: ptr) -> i64:
    """One UTF-8 character at byte `i`. Its width, or 0 if malformed.

    OVERLONG FORMS ARE REJECTED, which is what each width's lower bound is
    for: `0xC0 0x80` encodes U+0000 in two bytes, and accepting it would let
    a NUL through a check that only looked for the one-byte spelling.

    A SURROGATE OR AN OUT-OF-RANGE CODE POINT IS MALFORMED TOO -- the upper
    bound on the four-byte form is what says so.
    """
    c: i64 = i64(load(u8, offset(p, i)))
    if c < 128:
        store(u32, u32(c), out)
        return 1
    if (c & 224) == 192 and i + 1 < n:
        b1: i64 = i64(load(u8, offset(p, i + 1)))
        if (b1 & 192) == 128:
            cp: i64 = ((c & 31) << 6) | (b1 & 63)
            store(u32, u32(cp), out)
            if cp >= 128:
                return 2
            return 0
    if (c & 240) == 224 and i + 2 < n:
        c1: i64 = i64(load(u8, offset(p, i + 1)))
        c2: i64 = i64(load(u8, offset(p, i + 2)))
        if (c1 & 192) == 128 and (c2 & 192) == 128:
            cp3: i64 = ((c & 15) << 12) | ((c1 & 63) << 6) | (c2 & 63)
            store(u32, u32(cp3), out)
            if cp3 >= 2048:
                return 3
            return 0
    if (c & 248) == 240 and i + 3 < n:
        d1: i64 = i64(load(u8, offset(p, i + 1)))
        d2: i64 = i64(load(u8, offset(p, i + 2)))
        d3: i64 = i64(load(u8, offset(p, i + 3)))
        if (d1 & 192) == 128 and (d2 & 192) == 128 and (d3 & 192) == 128:
            cp4: i64 = ((c & 7) << 18) | ((d1 & 63) << 12) | \
                       ((d2 & 63) << 6) | (d3 & 63)
            store(u32, u32(cp4), out)
            if cp4 >= 65536 and cp4 <= 1114111:
                return 4
            return 0
    return 0


# -- turning a value into text, from the bottom up --------------------------


def apy_big_text(o: ptr) -> ptr:
    """A big in base ten.

    NINE DIGITS AT A TIME, because 10**9 is the largest power of ten that
    fits a 32-bit limb's remainder without overflowing the 64-bit
    intermediate -- so one pass of long division yields nine characters
    instead of one, and the whole thing is O(n**2) in limbs rather than in
    digits.

    QUADRATIC AND KNOWN TO BE. Each pass divides the whole magnitude, so a
    number with n limbs takes n passes over n limbs. That is why `bin` and
    `hex` are cheap where this is not: a power-of-two base needs no division.

    THE DIGITS COME OUT BACKWARDS and are reversed at the end, which is what
    taking remainders gives you.

    THE MAGNITUDE IS COPIED because the division destroys it, and the value
    being printed must survive being printed.
    """
    n: i64 = load(i64, offset(o, apy_big_n_offset()))
    if n == 0:
        return apy_from_cstr(rodata(b"0\0"))
    cap: i64 = n * 10 + 4
    w: ptr = apy_alloc_bytes(n * apy_limb_size())
    if not w:
        return w
    rev: ptr = apy_alloc_bytes(cap + 1)
    if not rev:
        return rev
    limb: ptr = ptr(load(u64, offset(o, apy_big_limb_offset())))
    i: i64 = 0
    while i < n:
        store(u32, load(u32, offset(limb, i * apy_limb_size())),
              offset(w, i * apy_limb_size()))
        i = i + 1
    out: i64 = 0
    nw: i64 = n
    billion: u64 = u64(1000000000)
    while nw > 0:
        rem: u64 = u64(0)
        j: i64 = nw - 1
        while j >= 0:
            cur: u64 = (rem << u64(apy_limb_bits())) | u64(load(
                u32, offset(w, j * apy_limb_size())))
            store(u32, u32(i64(cur // billion)),
                  offset(w, j * apy_limb_size()))
            rem = cur % billion
            j = j - 1
        trimming: i64 = 1
        while trimming:
            if nw <= 0:
                trimming = 0
            elif load(u32, offset(w, (nw - 1) * apy_limb_size())) != u32(0):
                trimming = 0
            else:
                nw = nw - 1
        k: i64 = 0
        going: i64 = 1
        while going:
            if k >= 9:
                going = 0
            else:
                store(u8, u8(48 + i64(rem % u64(10))), offset(rev, out))
                out = out + 1
                rem = rem // u64(10)
                k = k + 1
                if nw == 0 and rem == u64(0):
                    going = 0
    if load(i32, offset(o, apy_big_neg_offset())):
        store(u8, u8(45), offset(rev, out))
        out = out + 1
    buf: ptr = apy_alloc_bytes(out + 1)
    if not buf:
        return buf
    m: i64 = 0
    while m < out:
        store(u8, load(u8, offset(rev, out - 1 - m)), offset(buf, m))
        m = m + 1
    store(u8, u8(0), offset(buf, out))
    return apy_from_bytes(buf, out)


def apy_bytes_quote(p: ptr, n: i64) -> i64:
    """Which quote character a bytes repr should use.

    PYTHON PREFERS SINGLE and switches to double only when the content has a
    single and no double -- so `b"it's"` avoids an escape and `b'he said "x"'`
    keeps the single it started with.
    """
    single: i64 = 0
    double: i64 = 0
    i: i64 = 0
    while i < n:
        c: i64 = i64(load(u8, offset(p, i)))
        if c == 39:
            single = 1
        if c == 34:
            double = 1
        i = i + 1
    if single:
        if not double:
            return 34
    return 39


def apy_bytes_repr(v: ptr) -> ptr:
    """`repr(b"...")`, and `bytearray(...)` around it for a mutable one.

    THE WRAPPER IS BUILT BY CLEARING THE FLAG ROUND A RECURSIVE CALL rather
    than by a second escaping loop, so the two spellings cannot drift apart.

    FOUR BYTES PER INPUT BYTE IS THE WORST CASE -- a backslash, an x and
    two hex digits -- plus the quotes and the `b`.
    """
    if load(i32, offset(v, apy_s_mut_offset())):
        store(i32, i32(0), offset(v, apy_s_mut_offset()))
        inner: ptr = apy_bytes_repr(v)
        store(i32, i32(1), offset(v, apy_s_mut_offset()))
        if not inner:
            return inner
        m: i64 = load(i64, offset(inner, apy_str_len_offset()))
        wrapped: ptr = apy_alloc_bytes(m + 12)
        if not wrapped:
            return wrapped
        at: i64 = apy_cstr_into(wrapped, 0, m + 11,
                                rodata(b"bytearray(\0"))
        src: ptr = ptr(load(u64, offset(inner, apy_str_ptr_offset())))
        j: i64 = 0
        while j < m:
            store(u8, load(u8, offset(src, j)), offset(wrapped, at + j))
            j = j + 1
        store(u8, u8(41), offset(wrapped, at + m))
        store(u8, u8(0), offset(wrapped, at + m + 1))
        return apy_from_bytes(wrapped, at + m + 1)
    n: i64 = load(i64, offset(v, apy_str_len_offset()))
    p: ptr = ptr(load(u64, offset(v, apy_str_ptr_offset())))
    quote: i64 = apy_bytes_quote(p, n)
    out: ptr = apy_alloc_bytes(n * 4 + 5)
    if not out:
        return out
    digits: ptr = apy_base_digits()
    k: i64 = 0
    store(u8, u8(98), offset(out, k))
    store(u8, u8(quote), offset(out, k + 1))
    k = 2
    i: i64 = 0
    while i < n:
        c: i64 = i64(load(u8, offset(p, i)))
        if c == quote or c == 92:
            store(u8, u8(92), offset(out, k))
            store(u8, u8(c), offset(out, k + 1))
            k = k + 2
        elif c == 9:
            store(u8, u8(92), offset(out, k))
            store(u8, u8(116), offset(out, k + 1))
            k = k + 2
        elif c == 10:
            store(u8, u8(92), offset(out, k))
            store(u8, u8(110), offset(out, k + 1))
            k = k + 2
        elif c == 13:
            store(u8, u8(92), offset(out, k))
            store(u8, u8(114), offset(out, k + 1))
            k = k + 2
        elif c >= 32 and c < 127:
            store(u8, u8(c), offset(out, k))
            k = k + 1
        else:
            store(u8, u8(92), offset(out, k))
            store(u8, u8(120), offset(out, k + 1))
            store(u8, load(u8, offset(digits, c >> 4)), offset(out, k + 2))
            store(u8, load(u8, offset(digits, c & 15)), offset(out, k + 3))
            k = k + 4
        i = i + 1
    store(u8, u8(quote), offset(out, k))
    k = k + 1
    store(u8, u8(0), offset(out, k))
    return apy_from_bytes(out, k)


# -- turning a value into text ----------------------------------------------
#
# SPLIT, and the line is drawn where the work is. `apy_text` answers eighteen
# kinds and two of them -- FLOAT and COMPLEX -- need `py_repr_double`, the
# shortest-round-trip loop that prints a double and reads it back to check.
# That is `snprintf` and `strtod` doing the arithmetic, and writing it here
# means writing Ryu.
#
# THE REPR OF A STRING USED TO NEED THE UNICODE TABLE and no longer does:
# "is this character printable" is a question about the character and not the
# byte -- U+00A0 and U+2003 are spaces Python escapes, and asked a byte at a
# time neither one is anything at all -- and `apy_uc_mask` is now a binary
# search in the subset like everything else. So a string's repr is here.
#
# SO THIS HALF TAKES THE SCALARS, `str()`, AND A STRING'S REPR. Everything a
# program prints without asking for a repr comes through here, so does every
# integer, and so does every quoted string in a message or a traceback.


def apy_text_of(v: ptr, quoted: i64) -> ptr:
    """`str(v)` when `quoted` is 0, `repr(v)` when it is 1.

    ONE FUNCTION FOR BOTH, because for most kinds they are the same text --
    only a string and the containers holding one differ, and the flag is what
    tells them apart.

    `str()` OF A STRING IS THE STRING, which is the early return below and the
    single most common call this runtime makes: every `print` of a str
    arrives here and leaves without allocating.
    """
    k: i64 = i64(load(i32, offset(v, 0)))
    if k == apy_none_kind():
        return apy_from_cstr(rodata(b"None\0"))
    if k == apy_bool_kind():
        if apy_int_payload(v):
            return apy_from_cstr(rodata(b"True\0"))
        return apy_from_cstr(rodata(b"False\0"))
    if k == apy_int_kind():
        return apy_from_cstr_copy(apy_decimal_of(apy_int_payload(v), 0))
    if k == apy_big_kind():
        return apy_big_text(v)
    if k == apy_bytes_kind():
        # A BYTES ALWAYS SHOWS ITS REPR, even under `str()` -- which is the
        # wart CPython emits a BytesWarning about under -b. Reproducing it
        # means `print(b"ab")` shows the prefix, and stripping it would
        # disagree with CPython on every line that printed one.
        return apy_bytes_repr(v)
    if k == apy_ellipsis_kind():
        return apy_from_cstr(rodata(b"Ellipsis\0"))
    if k == apy_notimpl_kind():
        return apy_from_cstr(rodata(b"NotImplemented\0"))
    if k == apy_str_kind():
        if not quoted:
            return v
        return apy_str_repr(v)
    # THE CONTAINERS IGNORE `quoted` and always show their elements quoted,
    # which is Python's rule and not an oversight: `print(['a'])` writes
    # `['a']` and not `[a]`, because a list of one string and a list of one
    # name would otherwise read the same.
    if k == apy_list_kind() or k == apy_tuple_kind():
        return apy_seq_text_of(v)
    if k == apy_dict_kind():
        inner: ptr = apy_dict_text_of(v)
        if not load(i32, offset(v, apy_d_ro_offset())):
            return inner
        if not inner:
            return inner
        # A READ-ONLY DICT WEARS ITS NAME: `repr(C.__dict__)` is
        # `mappingproxy({...})` -- the mapping's own repr inside the
        # wrapper's, which is what the wrapper is.
        room: i64 = load(i64, offset(inner, apy_str_len_offset())) + 16
        buf: ptr = apy_alloc_bytes(room)
        if not buf:
            return buf
        out: i64 = apy_cstr_into(buf, 0, room, rodata(b"mappingproxy(\0"))
        # THROUGH `apy_cstr_into` AND NOT A BYTE COPY: a str cell's bytes are
        # NUL-terminated, and a repr never holds a raw NUL -- a string with
        # one in it renders the ESCAPE, which is four ordinary characters.
        out = apy_cstr_into(buf, out, room,
                            ptr(load(u64, offset(inner,
                                                 apy_str_ptr_offset()))))
        store(u8, u8(41), offset(buf, out))
        out = out + 1
        store(u8, u8(0), offset(buf, out))
        return apy_from_bytes(buf, out)
    if k == apy_set_kind() or k == apy_frozen_kind():
        return apy_set_text_of(v)
    if k == apy_exc_kind():
        # AN EXCEPTION OF A CLASS THE PROGRAM WROTE goes to the C half, which
        # asks that class for its own `__str__`/`__repr__` before rendering
        # anything BaseException would. One place holds that rule; this one
        # only knows the runtime's own exceptions have no class to ask.
        if load(u64, offset(v, apy_e_cls_offset())):
            return apy_text_of_slow(v, quoted)
        return apy_exc_text_of(v, quoted)
    return apy_text_of_slow(v, quoted)


def apy_from_cstr_copy(p: ptr) -> ptr:
    """A str cell over a COPY of the bytes at `p`.

    THE SCRATCH IS REUSED. `apy_decimal_of` writes into a reserved slot that
    the next call overwrites, so a cell pointing at it would change under
    whoever held it -- which is exactly the bug a borrowed cell invites.
    """
    n: i64 = 0
    while load(u8, offset(p, n)) != u8(0):
        n = n + 1
    return apy_str_copy_bytes(p, n)


def apy_repr(v: ptr) -> ptr:
    """`repr(v)` -- the spelling that reads back."""
    return apy_text_of(v, 1)


def apy_str(v: ptr) -> ptr:
    """`str(v)` -- the spelling meant for a person."""
    return apy_text_of(v, 0)


def apy_repr_cp() -> ptr:
    """One word, for the code point `apy_utf8_step_of` writes back.

    RESERVED AND NOT `alloca`, because it is written and read on adjacent
    lines and nothing in between can reach this again.
    """
    return reserve("apy_repr_cp_ir", 8)


def apy_hex_into(dst: ptr, at: i64, cp: i64, width: i64) -> i64:
    r"""`cp` as exactly `width` hex digits, high digit first. The new end.

    FIXED WIDTH AND NOT SHORTEST, because these go into an escape that has to
    read back as what it was: `\x1` followed by a literal `2` would parse as
    `\x12`, so the leading zero is always written.
    """
    digits: ptr = apy_base_digits()
    i: i64 = width - 1
    k: i64 = at
    while i >= 0:
        store(u8, load(u8, offset(digits, (cp >> (i * 4)) & 15)),
              offset(dst, k))
        k = k + 1
        i = i - 1
    return k


def apy_str_quote(p: ptr, n: i64) -> i64:
    """Which quote character a string repr should use.

    PYTHON PREFERS SINGLE and switches to double only when the text holds a
    single and no double -- so `"it's"` avoids an escape and `'he said "x"'`
    keeps the single it started with.
    """
    single: i64 = 0
    double: i64 = 0
    i: i64 = 0
    while i < n:
        c: i64 = i64(load(u8, offset(p, i)))
        if c == 39:
            single = 1
        if c == 34:
            double = 1
        i = i + 1
    if single:
        if not double:
            return 34
    return 39


def apy_str_repr(v: ptr) -> ptr:
    r"""`repr(s)` for a string.

    BY CODE POINT ABOVE 0x7F, because whether a character is printable is a
    question about the CHARACTER: U+00A0 and U+2003 are spaces Python
    escapes, and asked a byte at a time neither one is anything at all.
    Below 0x80 a byte IS a character, and that half is written out.

    THE THREE WIDTHS ARE PYTHON'S: `\xNN` under 0x100, `\uNNNN` under
    0x10000, `\UNNNNNNNN` above. An unprintable character written out in its
    own bytes would come back from `eval` unchanged and LOOK like the space
    it is not.

    A BYTE THAT IS NOT VALID UTF-8 IS ESCAPED AS ITSELF rather than replaced:
    `repr` is what a person reads to find out what is ACTUALLY in a string,
    so a lie about its bytes is the one thing it must not tell.

    FOUR BYTES PER INPUT BYTE IS THE WORST CASE -- the widest escape is eight
    digits over four input bytes, and the narrowest input that escapes at all
    is one byte becoming four -- plus the two quotes and a terminator.
    """
    n: i64 = load(i64, offset(v, apy_str_len_offset()))
    p: ptr = ptr(load(u64, offset(v, apy_str_ptr_offset())))
    q: i64 = apy_str_quote(p, n)
    buf: ptr = apy_alloc_bytes(n * 4 + 4)
    if not buf:
        return buf
    slot: ptr = apy_repr_cp()
    used: i64 = 0
    cp: i64 = 0
    w: i64 = 0
    store(u8, u8(q), buf)
    out: i64 = 1
    i: i64 = 0
    while i < n:
        c: i64 = i64(load(u8, offset(p, i)))
        if c < 128:
            if c == q or c == 92:
                store(u8, u8(92), offset(buf, out))
                store(u8, u8(c), offset(buf, out + 1))
                out = out + 2
            elif c == 10:
                store(u8, u8(92), offset(buf, out))
                store(u8, u8(110), offset(buf, out + 1))
                out = out + 2
            elif c == 13:
                store(u8, u8(92), offset(buf, out))
                store(u8, u8(114), offset(buf, out + 1))
                out = out + 2
            elif c == 9:
                store(u8, u8(92), offset(buf, out))
                store(u8, u8(116), offset(buf, out + 1))
                out = out + 2
            elif c < 32 or c == 127:
                store(u8, u8(92), offset(buf, out))
                store(u8, u8(120), offset(buf, out + 1))
                out = apy_hex_into(buf, out + 2, c, 2)
            else:
                store(u8, u8(c), offset(buf, out))
                out = out + 1
            i = i + 1
        else:
            used = apy_utf8_step_of(p, n, i, slot)
            if not used:
                store(u8, u8(92), offset(buf, out))
                store(u8, u8(120), offset(buf, out + 1))
                out = apy_hex_into(buf, out + 2, c, 2)
                i = i + 1
            else:
                cp = i64(load(u32, slot))
                if apy_cp_printable_of(cp):
                    w = 0
                    while w < used:
                        store(u8, load(u8, offset(p, i + w)),
                              offset(buf, out))
                        out = out + 1
                        w = w + 1
                elif cp < 256:
                    store(u8, u8(92), offset(buf, out))
                    store(u8, u8(120), offset(buf, out + 1))
                    out = apy_hex_into(buf, out + 2, cp, 2)
                elif cp < 65536:
                    store(u8, u8(92), offset(buf, out))
                    store(u8, u8(117), offset(buf, out + 1))
                    out = apy_hex_into(buf, out + 2, cp, 4)
                else:
                    store(u8, u8(92), offset(buf, out))
                    store(u8, u8(85), offset(buf, out + 1))
                    out = apy_hex_into(buf, out + 2, cp, 8)
                i = i + used
    store(u8, u8(q), offset(buf, out))
    out = out + 1
    store(u8, u8(0), offset(buf, out))
    return apy_from_bytes(buf, out)
