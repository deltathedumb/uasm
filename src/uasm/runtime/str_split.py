# `split` and `rsplit` on a separator, in the machine subset.
#
# STAGE 5 OF docs/INERT-RUNTIME.md, and the first ported function that builds
# a LIST. Every string family before this one answered with a bool or with one
# string; this one needs `apy_list_new` and `apy_seq_push`, both of which are
# ported, which is the whole reason the sequence work came first.
#
# NO ASCII GATE, and for the same reason `removeprefix` needs none: the
# separator is a WHOLE STRING, so finding it byte-wise finds it exactly where
# it is character-wise -- a valid UTF-8 encoding cannot contain the bytes of
# one character starting part-way through another. Every piece the split hands
# back is cut at a separator boundary, so every piece is a whole string.
#
# That is what separates this file from `runtime/str_strip.py`, which needs
# the gate: `strip` asks a question about each character, and this asks one
# about a substring.
#
# ── rsplit IS split, without a limit ───────────────────────────────────────
#
# `'a,b,c'.rsplit(',')` and `'a,b,c'.split(',')` are the same list. Direction
# only matters once there is a `maxsplit` to run out of, and the two-argument
# forms have none -- so `apy_str_rsplit` below is not an approximation of the
# C's backwards walk, it is the same answer reached forwards.
#
# THE C DOES WALK BACKWARDS and then reverses the list it built, because it
# has a `maxsplit` to honour and one body serving all six wrappers. Written
# here, that walk would be code whose only observable effect is to undo
# itself.
#
# ── what is declined ───────────────────────────────────────────────────────
#
# A `maxsplit` -- `apy_str_split_n` and `apy_str_rsplit_n` are separate
# exported functions and simply are not split, so every call with a limit
# still reaches the C.
#
# `sep` OF None, which means "split on runs of whitespace" -- a different
# operation with its own worker, and one that WOULD need the ASCII gate.
# `apy_is_str` declines it along with a bytes separator and the TypeError.
#
# AN EMPTY SEPARATOR, which is a ValueError the C words itself.


def apy_str_split_all(s: ptr, sep: ptr, m: i64) -> ptr:
    """Every piece of `s` between occurrences of `sep`.

    THE TRAILING PUSH IS NOT AN EDGE CASE, it is the definition: a split on
    `k` separators has `k + 1` pieces, so the loop handles the ones that end
    at a separator and the line after it handles the one that ends at the
    string. `'a,'.split(',')` is `['a', '']` and `','.split(',')` is
    `['', '']`, both of which fall out of that rather than being tested for.

    THE CAPACITY IS A GUESS AND ONLY A GUESS. Eight is what the C asks for;
    `apy_seq_push` grows the block when it runs out, so the number costs a
    reallocation at worst and never an answer.
    """
    out: ptr = apy_list_new(8)
    if not out:
        return out
    n: i64 = apy_str_byte_len(s)
    i: i64 = 0
    at: i64 = apy_str_find_at(s, sep, i, n)
    while at >= 0:
        apy_seq_push(out, apy_str_slice_new(s, i, at))
        i = at + m
        at = apy_str_find_at(s, sep, i, n)
    # THE PIECE AFTER THE LAST SEPARATOR IS THE RECEIVER when there was no
    # separator to consume, which is the shortcut the slow half takes at the
    # same place -- see `apy_split_piece_of`. A fast half that skipped it
    # would answer `is` one way and the C the other for one call.
    apy_seq_push(out, apy_split_piece_of(s, i, n))
    return out


def apy_str_split_ok(s: ptr, sep: ptr) -> i64:
    """The separator's byte length if this file may do the split, else -1."""
    if not apy_is_str(s):
        return -1
    if not apy_is_str(sep):
        return -1
    m: i64 = apy_str_byte_len(sep)
    if m == 0:
        return -1
    return m


def apy_str_split(s: ptr, sep: ptr) -> ptr:
    """`s.split(sep)` for two plain strings."""
    m: i64 = apy_str_split_ok(s, sep)
    if m < 0:
        return apy_str_split_slow(s, sep)
    return apy_str_split_all(s, sep, m)


def apy_str_rsplit(s: ptr, sep: ptr) -> ptr:
    """`s.rsplit(sep)` -- the same list, reached forwards. See the header."""
    m: i64 = apy_str_split_ok(s, sep)
    if m < 0:
        return apy_str_rsplit_slow(s, sep)
    return apy_str_split_all(s, sep, m)


# -- split and rsplit, both modes ------------------------------------------
#
# THE TWO SPLIT MODES ARE DIFFERENT ALGORITHMS, not one with a default
# separator, and the case that shows it is `"  a  b  "`: with no argument it
# splits on RUNS of whitespace and drops the empty pieces at both ends, giving
# ["a", "b"]; with `" "` it splits on each single space and keeps them, giving
# ["", "", "a", "", "b", "", ""]. A default of `" "` would answer the second
# to both.


def apy_seq_reverse_of(out: ptr) -> ptr:
    """Turn a list back to front, in place.

    BOTH RIGHT-HAND SPLITS BUILD BACKWARDS, because walking from the end is
    the natural way to find the last separator first -- and both then need the
    pieces in reading order.
    """
    n: i64 = load(i64, offset(out, apy_q_n_offset()))
    items: ptr = ptr(load(u64, offset(out, apy_q_items_offset())))
    i: i64 = 0
    j: i64 = n - 1
    while i < j:
        a: u64 = load(u64, offset(items, i * apy_value_size()))
        b: u64 = load(u64, offset(items, j * apy_value_size()))
        store(u64, b, offset(items, i * apy_value_size()))
        store(u64, a, offset(items, j * apy_value_size()))
        i = i + 1
        j = j - 1
    return out


def apy_utf8_width_of(wide: i64, p: ptr, n: i64, i: i64) -> i64:
    """The bytes of the character starting at `i`, at least one.

    THE FORWARD STEP over a non-space run. `apy_utf8_at_of` writes the width
    back through a pointer and this is the only thing that wants it, so the
    code point it also answers is dropped.
    """
    if wide == 0:
        return 1
    slot: ptr = apy_space_slot()
    apy_utf8_at_of(p, n, i, slot)
    w: i64 = load(i64, slot)
    if w < 1:
        return 1
    return w


def apy_space_slot() -> ptr:
    """One word, for the width `apy_utf8_at_of` writes back."""
    return reserve("apy_space_width_ir", 8)


def apy_space_at_of(wide: i64, p: ptr, n: i64, i: i64) -> i64:
    """The bytes of the whitespace CHARACTER starting at `i`, or 0.

    `apy_c_space` KNOWS THE SIX ASCII BYTES AND NOTHING ELSE, so
    `"\u00a0".split()` answered `["\u00a0"]` where Python answers `[]`:
    every Unicode space was an ordinary character to split around. `strip()`
    was fixed by asking the character table; this is the same fix for the
    other splitter.

    A BYTES RECEIVER KEEPS THE ASCII ANSWER, which is the same split every
    shared text body now makes: `b"\xc2\xa0"` is two bytes and neither is
    whitespace.

    ANSWERING A WIDTH RATHER THAN A FLAG is what lets the caller advance: a
    space character may be two or three bytes, and stepping one at a time
    would ask about its continuation bytes.
    """
    if wide == 0:
        if apy_c_space(i64(load(u8, offset(p, i)))) != 0:
            return 1
        return 0
    slot: ptr = apy_space_slot()
    cp: i64 = apy_utf8_at_of(p, n, i, slot)
    if (apy_char_class_of(cp) & apy_uc_space()) != 0:
        return load(i64, slot)
    return 0


def apy_char_start_of(wide: i64, p: ptr, i: i64) -> i64:
    """Where the character ENDING at `i` begins.

    THE BACKWARD STEP `rsplit` NEEDS. A continuation byte is `10xxxxxx`, so
    the lead is the first byte at or before `i - 1` that is not one -- and
    four bytes is the most a character can take, which bounds the walk over
    malformed input rather than letting it run to the start.
    """
    if wide == 0:
        return i - 1
    at: i64 = i - 1
    while at > 0 and i - at < 4 and (load(u8, offset(p, at))
                                     & u8(192)) == u8(128):
        at = at - 1
    return at


def apy_split_piece_of(s: ptr, lo: i64, hi: i64) -> ptr:
    """One piece of a split, and the RECEIVER ITSELF when it spans the whole
    of it.

    CPython CARRIES THIS IN stringlib AND NOT AS AN OPTIMISATION.
    `STRINGLIB(split_whitespace)` and `STRINGLIB(rsplit_whitespace)` in
    `Objects/stringlib/split.h` end a piece with `Py_INCREF(str_obj);
    PyList_SET_ITEM(list, 0, str_obj)` under `#ifndef STRINGLIB_MUTABLE` and
    a `STRINGLIB_CHECK_EXACT` test, and the separator forms do the same for
    the piece after the last separator. Measured against CPython 3.14 in
    `scratchpad/probes/d175.py`: `"abc".split("z")[0] is "abc"` and
    `b"abc".split(b"z")[0] is b"abc"` are True, and so are `rsplit`, both
    whitespace forms, `splitlines` and the `maxsplit` of 0 spellings.

    A BYTEARRAY TAKES THE COPY, and a LIST SLOT is exactly why: a piece is a
    thing the program keeps, so handing a writable receiver into one gives
    it two live names for one buffer and the piece changes under it at the
    next `ba[0] = ...`. That is the same hazard the C's
    `apy_str_self_or_copy` answers for the eight methods that end in it, one
    level down -- and `bytearray(b"abc").split(b"z")[0] is` it is False in
    CPython, with the piece a fresh bytearray.

    THE COPY IS BUILT HERE AND NOT BY RE-TAGGING A SHARED CELL:
    `apy_bytes_made_of` with `mut` set allocates whatever the length, so an
    empty bytearray receiver cannot come back as the shared empty bytes with
    the writable flag written over it -- which would make every later `b""`
    in the program assignable.

    A PIECE SPANNING THE WHOLE RECEIVER IS THE ONLY PIECE, whichever split
    asks: a separator consumed, a run of whitespace dropped or a line break
    cut would each leave some byte out. So this can never hand the receiver
    into two slots, and no caller has to count.
    """
    if lo == 0 and hi == apy_str_byte_len(s):
        if apy_str_may_return_self_of(s):
            return s
        if apy_is_bytearray_of(s):
            return apy_bytes_made_of(apy_str_data(s), hi, 1)
    return apy_str_slice_of(s, lo, hi)


def apy_split_ws_of(s: ptr, maxsplit: i64, from_right: i64) -> ptr:
    """Split on RUNS of whitespace, dropping the empty ends.

    THE REMAINDER GOES IN WHOLE once the limit is reached, INCLUDING its
    trailing whitespace: `"  a  b  ".split(None, 1)` is ["a", "b  "]. Only the
    whitespace BEFORE a piece is skipped. Right-stripping the remainder as
    well looks tidier and answers ["a", "b"], which is wrong -- and invisible
    unless a case splits a string that has trailing space.

    A `going` FLAG RATHER THAN A `break`, which the subset has none of. Using
    the loop variable as the stop condition would work here and not in the
    right-hand walk below, so both are written the same way.
    """
    out: ptr = apy_seq_new_of(apy_list_kind(), 8)
    if not out:
        return out
    n: i64 = load(i64, offset(s, apy_str_len_offset()))
    p: ptr = ptr(load(u64, offset(s, apy_str_ptr_offset())))
    # BY CHARACTER, NOT BY BYTE, for a str receiver -- see `apy_space_at_of`.
    wide: i64 = 0
    if i64(load(i32, offset(s, 0))) == apy_str_kind():
        wide = 1
    if not from_right:
        i: i64 = 0
        going: i64 = 1
        while going:
            skip: i64 = 1
            while i < n and skip > 0:
                skip = apy_space_at_of(wide, p, n, i)
                i = i + skip
            if i >= n:
                going = 0
            else:
                if maxsplit >= 0 and load(
                        i64, offset(out, apy_q_n_offset())) == maxsplit:
                    # A FRESH PIECE EVEN WHEN IT SPANS THE WHOLE RECEIVER,
                    # which is the one place `apy_split_piece_of` must not
                    # be reached for. CPython's shortcut sits INSIDE the
                    # `while (maxcount-- > 0)` loop of
                    # `STRINGLIB(split_whitespace)` and the remainder is
                    # added after it by a plain `SPLIT_ADD`, so
                    # `"abc".split(None, 0)[0] is "abc"` is False there
                    # while `"abc".split(None, 1)[0] is "abc"` is True.
                    # Measured both ways before this was written.
                    apy_q_append_of(out, apy_str_slice_of(s, i, n))
                    going = 0
                else:
                    j: i64 = i
                    step: i64 = 0
                    while j < n and step == 0:
                        step = apy_space_at_of(wide, p, n, j)
                        if step == 0:
                            j = j + apy_utf8_width_of(wide, p, n, j)
                    apy_q_append_of(out, apy_split_piece_of(s, i, j))
                    i = j
        return out
    # THE MIRROR IMAGE: the remainder keeps its LEADING whitespace, so
    # `"  a  b  ".rsplit(None, 1)` is ["  a", "b"].
    k: i64 = n
    walking: i64 = 1
    while walking:
        back: i64 = 1
        while k > 0 and back > 0:
            at: i64 = apy_char_start_of(wide, p, k)
            back = apy_space_at_of(wide, p, n, at)
            if back > 0:
                k = at
        if k <= 0:
            walking = 0
        else:
            if maxsplit >= 0 and load(
                    i64, offset(out, apy_q_n_offset())) == maxsplit:
                # FRESH, for the reason the forward remainder is:
                # `"abc".rsplit(None, 0)[0] is "abc"` is False in CPython.
                apy_q_append_of(out, apy_str_slice_of(s, 0, k))
                walking = 0
            else:
                m: i64 = k
                held: i64 = 0
                while m > 0 and held == 0:
                    was: i64 = apy_char_start_of(wide, p, m)
                    held = apy_space_at_of(wide, p, n, was)
                    if held == 0:
                        m = was
                apy_q_append_of(out, apy_split_piece_of(s, m, k))
                k = m
    return apy_seq_reverse_of(out)


def apy_split_sep_of(s: ptr, sep: ptr, maxsplit: i64,
                     from_right: i64) -> ptr:
    """Split on each occurrence of `sep`, keeping the empty pieces.

    THE LAST PIECE IS APPENDED OUTSIDE THE LOOP, which is why `"a,".split(",")`
    is ["a", ""] and not ["a"]: the text after the final separator is a piece
    even when it is empty, and so is the whole string when there was no
    separator at all.

    AN EMPTY SEPARATOR IS REFUSED, because every position would match and the
    loop would not advance -- Python refuses it for the same reason.
    """
    n: i64 = load(i64, offset(s, apy_str_len_offset()))
    m: i64 = load(i64, offset(sep, apy_str_len_offset()))
    if m == 0:
        return apy_raise_fmt(rodata(b"ValueError\0"),
                             rodata(b"empty separator%s%s\0"),
                             rodata(b"\0"), rodata(b"\0"))
    out: ptr = apy_seq_new_of(apy_list_kind(), 8)
    if not out:
        return out
    if not from_right:
        i: i64 = 0
        going: i64 = 1
        while going:
            if maxsplit >= 0 and load(
                    i64, offset(out, apy_q_n_offset())) >= maxsplit:
                going = 0
            else:
                at: i64 = apy_find_at(s, sep, i, n)
                if at < 0:
                    going = 0
                else:
                    apy_q_append_of(out, apy_str_slice_of(s, i, at))
                    i = at + m
        # THE PIECE AFTER THE LAST SEPARATOR, which is the whole receiver
        # when there was no separator to consume -- or when `maxsplit` was 0,
        # which is the same thing measured from here. Every piece above ends
        # at a separator, so only this one can span the receiver.
        apy_q_append_of(out, apy_split_piece_of(s, i, n))
        return out
    k: i64 = n
    walking: i64 = 1
    while walking:
        if maxsplit >= 0 and load(
                i64, offset(out, apy_q_n_offset())) >= maxsplit:
            walking = 0
        else:
            back: i64 = apy_rfind_at(s, sep, 0, k)
            if back < 0:
                walking = 0
            else:
                apy_q_append_of(out, apy_str_slice_of(s, back + m, k))
                k = back
    # The mirror of the forward tail: the piece BEFORE the last separator
    # found, spanning the receiver when none was.
    apy_q_append_of(out, apy_split_piece_of(s, 0, k))
    return apy_seq_reverse_of(out)


def apy_split_limit() -> ptr:
    """One word: the `maxsplit` being read out of its argument."""
    return reserve("apy_split_limit_ir", 8)


def apy_str_split_impl_of(s: ptr, sep: ptr, limit: ptr,
                          from_right: i64) -> ptr:
    """Which of the two split algorithms this call means.

    NO SEPARATOR AND `None` ARE THE SAME THING and both mean whitespace, which
    is why the test admits either: `s.split()` and `s.split(None, 2)` are the
    same mode.

    ANY NEGATIVE LIMIT MEANS NO LIMIT, which is Python's rule -- `s.split(",",
    -3)` splits on every comma rather than refusing.

    BYTES TOO. `b"a,b".split(b",")` is the same operation on the same layout;
    the RECEIVER decides the result's kind, which `apy_str_like` settles above
    this.
    """
    maxsplit: i64 = -1
    if limit:
        bounds: ptr = apy_split_limit()
        store(i64, -1, bounds)
        if not apy_int_arg_of(limit, bounds):
            return ptr(0)
        maxsplit = load(i64, bounds)
    if maxsplit < 0:
        maxsplit = -1
    if not sep:
        return apy_split_ws_of(s, maxsplit, from_right)
    if i64(load(i32, offset(sep, 0))) == apy_none_kind():
        return apy_split_ws_of(s, maxsplit, from_right)
    # AN INSTANCE OF A CLASS EXTENDING str OR bytes IS ONE, for anything
    # that reads the buffer -- see the C's `apy_text_like`. Written out
    # rather than called, because the subset has no helper for it and three
    # sites need the same six lines.
    held: ptr = ptr(0)
    if i64(load(i32, offset(sep, 0))) == apy_inst_kind():
        held = ptr(load(u64, offset(sep, apy_o_held_offset())))
    if held:
        hk: i64 = i64(load(i32, offset(held, 0)))
        if hk == apy_str_kind():
            sep = held
        if hk == apy_bytes_kind():
            sep = held
    # THE RECEIVER DECIDES. A bytes separator used to pass for a str receiver
    # and back, so `b"a,b".split(",")` answered `[b'a', b'b']` -- a wrong
    # answer where CPython refuses.
    if i64(load(i32, offset(s, 0))) == apy_bytes_kind():
        sep = apy_text_arg_of(rodata(b"split\0"), 0, 0, s, sep)
        if not sep:
            return ptr(0)
    elif i64(load(i32, offset(sep, 0))) != apy_str_kind():
        # A STR RECEIVER HAS ITS OWN WORDING, and it mentions None because
        # None is what a separator may also be.
        return apy_raise_fmt(rodata(b"TypeError\0"),
                             rodata(b"must be str or None, not %s%s\0"),
                             apy_kind_name_of(sep), rodata(b"\0"))
    return apy_split_sep_of(s, sep, maxsplit, from_right)


def apy_str_split_ws(s: ptr) -> ptr:
    """`s.split()`."""
    if not apy_str_self_of(rodata(b"split\0"), s):
        return ptr(0)
    return apy_str_split_impl_of(s, ptr(0), ptr(0), 0)


def apy_str_split_n(s: ptr, sep: ptr, limit: ptr) -> ptr:
    """`s.split(sep, maxsplit)`."""
    if not apy_str_self_of(rodata(b"split\0"), s):
        return ptr(0)
    return apy_str_split_impl_of(s, sep, limit, 0)


def apy_str_rsplit_ws(s: ptr) -> ptr:
    """`s.rsplit()`."""
    if not apy_str_self_of(rodata(b"rsplit\0"), s):
        return ptr(0)
    return apy_str_split_impl_of(s, ptr(0), ptr(0), 1)


def apy_str_rsplit_n(s: ptr, sep: ptr, limit: ptr) -> ptr:
    """`s.rsplit(sep, maxsplit)`."""
    if not apy_str_self_of(rodata(b"rsplit\0"), s):
        return ptr(0)
    return apy_str_split_impl_of(s, sep, limit, 1)


def apy_linebreak_at_of(wide: i64, p: ptr, n: i64, i: i64) -> i64:
    """The bytes of the LINE BREAK starting at `i`, or 0.

    A str BREAKS ON TEN CHARACTERS AND A bytes ON TWO, which is the same
    split between the two kinds that `apy_space_at_of` above makes, and for
    the same reason: `STRINGLIB(splitlines)` in `Objects/stringlib/split.h`
    tests `STRINGLIB_ISLINEBREAK`, which is `Py_UNICODE_ISLINEBREAK`
    (`_PyUnicode_IsLinebreak`, ten code points) in the unicode instantiation
    and `(x == '\\n' || x == '\\r')` in the bytes one -- see
    `Objects/stringlib/stringdefs.h`. Measured against CPython 3.14 in
    `scratchpad/probes/d175f.py`: `"a\\x0bb".splitlines()` is `['a', 'b']`
    while `b"a\\x0bb".splitlines()` is `[b'a\\x0bb']`.

    THE SIX THIS WALKER USED TO MISS -- \\v \\f \\x1c \\x1d \\x1e \\x85
    \\u2028 \\u2029 -- were a wrong list of pieces, and once
    `apy_split_piece_of` arrived they became a wrong ANSWER TO `is` as well:
    that shortcut hands the receiver back when the one piece spans the whole
    of it, and "spans the whole of it" is decided here, so a text holding
    one of them answered `t.splitlines()[0] is t` True where CPython answers
    a different object AND a different list. The interpreter binds
    `splitlines` to Python's own and was right all along, so this was a
    two-against-two split.

    WRITTEN OUT AND NOT ASKED OF THE CHARACTER TABLE, because there is no
    line-break bit in it: `apy_char_class_of` carries alpha, the three digit
    kinds, the three cases, space, printable and the two identifier bits,
    and CPython's own answer is a generated fixed list rather than a
    category test.

    ANSWERING A WIDTH RATHER THAN A FLAG is what lets the caller advance
    past a break that is two or three bytes of UTF-8, exactly as
    `apy_space_at_of` does; stepping one byte at a time would ask about its
    continuation bytes.
    """
    if wide == 0:
        c: i64 = i64(load(u8, offset(p, i)))
        if c == 10 or c == 13:
            return 1
        return 0
    slot: ptr = apy_space_slot()
    cp: i64 = apy_utf8_at_of(p, n, i, slot)
    w: i64 = load(i64, slot)
    if w < 1:
        w = 1
    # IN DECIMAL BECAUSE THE SUBSET HAS NO CHARACTER LITERAL, and the C
    # twin spells the same ten as `'\n' '\r' '\v' '\f' 0x1c 0x1d 0x1e
    # 0x85 0x2028 0x2029`: 10 LF, 13 CR, 11 VT, 12 FF, 28-30 the file,
    # group and record separators, 133 NEL, 8232 LINE SEPARATOR, 8233
    # PARAGRAPH SEPARATOR. Read the two lists side by side when changing
    # either.
    if cp == 10 or cp == 13 or cp == 11 or cp == 12:
        return w
    if cp >= 28 and cp <= 30:
        return w
    if cp == 133 or cp == 8232 or cp == 8233:
        return w
    return 0


def apy_splitlines_impl_of(s: ptr, keepends: i64) -> ptr:
    """`s.splitlines()` -- split on line boundaries.

    `\\r\\n` IS ONE BREAK AND NOT TWO, which is the whole reason this is not
    `split("\\n")`: a file written on Windows would otherwise gain an empty
    line between every pair. It is also the ONLY pair -- a `\\r` before
    anything else, and every one of the other nine breaks
    `apy_linebreak_at_of` knows, stands alone.

    A TRAILING BREAK ADDS NO EMPTY PIECE -- `"a\\n".splitlines()` is `["a"]`
    where `"a\\n".split("\\n")` is `["a", ""]`. That falls out of the walk:
    the loop ends when the break is consumed, with nothing left to start a
    new piece.

    `keepends` PUTS THE BREAK BACK ON, which is what makes the pieces
    reassemble into the original.
    """
    out: ptr = apy_seq_new_of(apy_list_kind(), 8)
    if not out:
        return out
    n: i64 = load(i64, offset(s, apy_str_len_offset()))
    p: ptr = ptr(load(u64, offset(s, apy_str_ptr_offset())))
    wide: i64 = 0
    if apy_is_str(s):
        wide = 1
    i: i64 = 0
    while i < n:
        start: i64 = i
        brk: i64 = 0
        going: i64 = 1
        while going:
            if i >= n:
                going = 0
            else:
                brk = apy_linebreak_at_of(wide, p, n, i)
                if brk != 0:
                    going = 0
                else:
                    i = i + apy_utf8_width_of(wide, p, n, i)
        stop: i64 = i
        if i < n:
            two: i64 = 0
            if i64(load(u8, offset(p, i))) == 13:
                if i + 1 < n:
                    if i64(load(u8, offset(p, i + 1))) == 10:
                        two = 1
            if two:
                i = i + 2
            else:
                i = i + brk
        cut: i64 = stop
        if keepends:
            cut = i
        # A TEXT WITH NO LINE BREAK IN IT IS ONE LINE AND THAT LINE IS THE
        # RECEIVER, with or without `keepends`: the cut spans the whole of it
        # either way.
        apy_q_append_of(out, apy_split_piece_of(s, start, cut))
    return out


def apy_str_has_tab_of(p: ptr, n: i64) -> i64:
    """Whether any of the `n` bytes at `p` is a tab.

    THE MEASURING PASS, AND ONLY A TAB DECIDES IT. A newline resets the
    column and changes nothing else, so a string full of them still has
    nothing to expand; asking about byte 9 and nothing else is the whole
    question CPython's own pass asks.
    """
    i: i64 = 0
    while i < n:
        if i64(load(u8, offset(p, i))) == 9:
            return 1
        i = i + 1
    return 0


def apy_str_expandtabs(s: ptr, width: ptr) -> ptr:
    """`s.expandtabs(width)`.

    A TAB ADVANCES TO THE NEXT MULTIPLE OF `width`, which is not the same as
    inserting `width` spaces: the padding depends on the column reached so
    far, and that is what makes columns line up.

    THE COLUMN RESETS ON A LINE BREAK, so each line is tabulated from its own
    start rather than from the beginning of the string.

    A COLUMN IS A CHARACTER AND NOT A BYTE, which is the whole of what UTF-8
    changes here: counting bytes made `"é\tx"` reach column 2 after one
    character and the tab stop land one place early. A continuation byte is
    `10xxxxxx` and adds nothing to the column; every other byte starts a
    character and adds one.

    THE BUFFER IS SIZED FOR EVERY BYTE BEING A TAB, which is the worst case
    and cheap in a bump arena.
    """
    if not apy_str_self_of(rodata(b"expandtabs\0"), s):
        return ptr(0)
    # THE WIDTH IS REQUIRED AND MUST BE AN INTEGER. It arrives always -- the
    # frontend supplies 8 for the no-argument form -- so a value that is not
    # an integer is one the PROGRAM wrote, and `"a\tb".expandtabs(None)` is a
    # TypeError in Python. Reading a non-integer as the default made the
    # written None answer what the omitted argument answers, which is the
    # difference no padding scheme can see once it has been applied.
    if not apy_is_int_like_of(width):
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"'%s' object cannot be interpreted as an integer%s\0"),
            apy_kind_name_of(width), rodata(b"\0"))
    w: i64 = apy_int_payload(width)
    if apy_is_big_of(width):
        w = 8
    if w < 1:
        w = 1
    # A COLUMN IS A CHARACTER IN A STR AND A BYTE IN BYTES, which is the only
    # thing the bytes receiver changes: `b"\xc3\xa9\tx"` is two columns
    # before the tab where the str it decodes to is one.
    wide: i64 = 0
    if i64(load(i32, offset(s, 0))) == apy_str_kind():
        wide = 1
    n: i64 = load(i64, offset(s, apy_str_len_offset()))
    p: ptr = ptr(load(u64, offset(s, apy_str_ptr_offset())))
    # NO TAB, NOTHING TO EXPAND -- and a str then IS its own answer where
    # bytes builds the copy anyway. THAT ASYMMETRY IS CPYTHON'S and not a
    # slip to be tidied away: `unicode_expandtabs` ends its measuring pass
    # with `if (!found_tabs) return unicode_result_unchanged(self)`, and the
    # stringlib `expandtabs` that bytes and bytearray share has no such test
    # and hands back what it built. So `"abc".expandtabs() is "abc"` is True
    # while `b"abc".expandtabs() is b"abc"` is False, which is why this is
    # gated on `wide` and not on `apy_str_may_return_self_of`. The C half
    # gates on its own `wide` in exactly the same place.
    if wide:
        if not apy_str_has_tab_of(p, n):
            return s
    cap: i64 = n * w + 8
    buf: ptr = apy_alloc_bytes(cap + 1)
    if not buf:
        return buf
    col: i64 = 0
    out: i64 = 0
    i: i64 = 0
    while i < n:
        c: i64 = i64(load(u8, offset(p, i)))
        if c == 9:
            pad: i64 = w - (col % w)
            while pad > 0 and out < cap:
                store(u8, u8(32), offset(buf, out))
                out = out + 1
                col = col + 1
                pad = pad - 1
        else:
            if out < cap:
                store(u8, u8(c), offset(buf, out))
                out = out + 1
            if c == 13 or c == 10:
                col = 0
            elif wide == 0:
                col = col + 1
            elif (c & 192) != 128:
                col = col + 1
        i = i + 1
    store(u8, u8(0), offset(buf, out))
    return apy_from_bytes(buf, out)


def apy_int_mag_byte_of(v: ptr, k: i64) -> i64:
    """The `k`-th byte of this integer's MAGNITUDE, least significant first.

    ZERO BEYOND THE END, because a request for a wider byte is a request for
    the leading zeroes an integer conceptually has.

    THE WHOLE REASON THIS EXISTS. `apy_int_payload` is the i64 an int cell
    holds, and a BIG integer's cell does not hold its value there -- it holds
    a pointer to the limbs. Reading it as a number put the object's ADDRESS
    into the answer, so `(2**63).to_bytes(16, 'little')` returned a heap
    address formatted as data, on every path except the interpreter.
    """
    if apy_is_big_of(v):
        n: i64 = load(i64, offset(v, apy_big_n_offset()))
        which: i64 = k // apy_limb_size()
        if which >= n:
            return 0
        limb: ptr = ptr(load(u64, offset(v, apy_big_limb_offset())))
        w: u64 = u64(load(u32, offset(limb, which * apy_limb_size())))
        return i64((w >> u64((k - which * apy_limb_size()) * 8)) & u64(255))
    if k >= 8:
        return 0
    m: i64 = apy_int_payload(v)
    if m < 0:
        # NEGATION WRAPS FOR THE MOST NEGATIVE i64 and that is the right
        # answer: -(-2**63) is -2**63 again in two's complement, and read as
        # unsigned that is 2**63, which is exactly its magnitude.
        m = -m
    return i64((u64(m) >> u64(k * 8)) & u64(255))


def apy_int_mag_len_of(v: ptr) -> i64:
    """How many bytes the magnitude needs, with no leading zeroes."""
    top: i64 = 8
    if apy_is_big_of(v):
        top = load(i64, offset(v, apy_big_n_offset())) * apy_limb_size()
    while top > 0:
        if apy_int_mag_byte_of(v, top - 1) != 0:
            return top
        top = top - 1
    return 0


def apy_int_is_neg_of(v: ptr) -> i64:
    """Whether this integer is negative. A big keeps its sign in a flag."""
    if apy_is_big_of(v):
        return i64(load(i32, offset(v, apy_big_neg_offset())))
    if apy_int_payload(v) < 0:
        return 1
    return 0


def apy_to_bytes_n(v: ptr, length: ptr, order: ptr, signed: ptr) -> ptr:
    """`n.to_bytes(length, byteorder, signed)`.

    THE MAGNITUDE IS READ BYTE BY BYTE rather than shifted out of an i64, so
    a big integer answers its value instead of its address -- see
    `apy_int_mag_byte_of`, which is the bug this shape exists for.

    THE RANGE TEST IS ON THE MAGNITUDE'S LENGTH, not on what is left over
    after shifting. The two agree for a small int and only the first is
    available for a big, and having one rule for both is worth more than the
    shift being marginally cheaper.

    BIG-ENDIAN IS THE DEFAULT: anything that is not exactly "little" is big,
    which is the C's rule, kept.
    """
    if not apy_is_int_like_of(v):
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute "
                   b"'to_bytes'%s\0"),
            apy_kind_name_of(v), rodata(b"\0"))
    # CPYTHON'S OWN TWO REFUSALS, which are not one: the LENGTH is named by
    # the value that was handed over -- the wording every index-taking slot
    # uses -- and the BYTEORDER by the parameter, the arg clinic's form.
    if not apy_is_int_like_of(length):
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"'%s' object cannot be interpreted as an integer%s\0"),
            apy_kind_name_of(length), rodata(b"\0"))
    if i64(load(i32, offset(order, 0))) != apy_str_kind():
        named: ptr = apy_kind_name_of(order)
        if i64(load(i32, offset(order, 0))) == apy_none_kind():
            named = rodata(b"None\0")
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"to_bytes() argument 'byteorder' must be str, "
                   b"not %s%s\0"),
            named, rodata(b"\0"))
    n: i64 = apy_int_payload(length)
    if n < 0:
        return apy_raise_at(rodata(b"ValueError\0"),
                            rodata(b"length argument must be non-negative\0"))
    if n > 1024:
        return apy_raise_at(rodata(b"OverflowError\0"),
                            rodata(b"int too big to convert\0"))
    big: i64 = 1
    if i64(load(i32, offset(order, 0))) == apy_str_kind():
        if apy_cstr_eq(ptr(load(u64, offset(order, apy_str_ptr_offset()))),
                       rodata(b"little\0")):
            big = 0
    want_signed: i64 = apy_truth(signed)
    neg: i64 = apy_int_is_neg_of(v)
    # SPELLED AS TWO TESTS, not `neg and not want_signed`: in the machine
    # subset an i64 and a bool are different types and cannot meet in one
    # boolean expression.
    if neg != 0:
        if want_signed == 0:
            return apy_raise_at(
                rodata(b"OverflowError\0"),
                rodata(b"can't convert negative int to unsigned\0"))
    used: i64 = apy_int_mag_len_of(v)
    if apy_to_bytes_fits_of(v, n, used, neg, want_signed) == 0:
        return apy_raise_at(rodata(b"OverflowError\0"),
                            rodata(b"int too big to convert\0"))
    room: i64 = n
    if room == 0:
        room = 1
    buf: ptr = apy_alloc_bytes(room + 1)
    if not buf:
        return buf
    z: i64 = 0
    while z <= room:
        store(u8, u8(0), offset(buf, z))
        z = z + 1
    # LITTLE-ENDIAN FIRST, ALWAYS. The two's complement below carries from the
    # least significant byte upward, which is only simple in this order; the
    # bytes are reversed at the end if the caller asked for big-endian.
    i: i64 = 0
    while i < n:
        store(u8, u8(apy_int_mag_byte_of(v, i)), offset(buf, i))
        i = i + 1
    if neg != 0:
        carry: i64 = 1
        j: i64 = 0
        while j < n:
            b: i64 = (255 - i64(load(u8, offset(buf, j)))) + carry
            carry = 0
            if b > 255:
                b = b - 256
                carry = 1
            store(u8, u8(b), offset(buf, j))
            j = j + 1
    if big != 0:
        a: i64 = 0
        b2: i64 = n - 1
        while a < b2:
            t: i64 = i64(load(u8, offset(buf, a)))
            store(u8, load(u8, offset(buf, b2)), offset(buf, a))
            store(u8, u8(t), offset(buf, b2))
            a = a + 1
            b2 = b2 - 1
    return apy_bytes_literal(buf, n)


def apy_to_bytes_fits_of(v: ptr, n: i64, used: i64, neg: i64,
                         want_signed: i64) -> i64:
    """Whether the value fits `n` bytes, under the rule the caller asked for.

    THREE RULES, NOT ONE. Unsigned needs the magnitude to fit outright. Signed
    and positive loses the top bit to the sign, so 127 fits one byte and 128
    does not. Signed and NEGATIVE reaches one further in that direction -- -128
    fits one byte -- and that extra value is exactly the power of two, so it is
    the magnitude having a single one bit at the top that makes it fit.
    """
    if used > n:
        return 0
    if want_signed == 0:
        return 1
    if used < n:
        return 1
    top: i64 = apy_int_mag_byte_of(v, n - 1)
    if top < 128:
        return 1
    if neg == 0:
        return 0
    if top != 128:
        return 0
    # THE ONE EXTRA VALUE. -2**(8n-1) is the only magnitude with the top bit
    # set that fits, and it has no other bit set at all.
    k: i64 = 0
    while k < n - 1:
        if apy_int_mag_byte_of(v, k) != 0:
            return 0
        k = k + 1
    return 1


def apy_str_splitlines(s: ptr) -> ptr:
    """`s.splitlines()`."""
    if not apy_str_self_of(rodata(b"splitlines\0"), s):
        return ptr(0)
    return apy_splitlines_impl_of(s, 0)


def apy_str_splitlines_keep(s: ptr, keep: ptr) -> ptr:
    """`s.splitlines(keepends)`.

    ANY TRUTHY VALUE KEEPS THE ENDS, not just True -- `splitlines(1)` is what
    a program written against the C API spells, and Python accepts it.
    """
    if not apy_str_self_of(rodata(b"splitlines\0"), s):
        return ptr(0)
    return apy_splitlines_impl_of(s, apy_truth(keep))
