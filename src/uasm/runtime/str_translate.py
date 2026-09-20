# `s.translate(table)` -- a per-CHARACTER substitution.
#
# BY CODE POINT AND NOT BY BYTE, which is what the table is keyed on:
# `str.maketrans` builds `{ord(c): ...}`, so stepping a byte at a time would
# look up 'e' under each half of its encoding and match neither.
#
# A REPLACEMENT MAY BE A WHOLE STRING, not just one character --
# `{ord('&'): 'and'}` is an ordinary thing to write -- and it may be `None`,
# which DELETES. So the output length is not a function of the input length
# and cannot be guessed.
#
# TWO PASSES RATHER THAN A GROWING BUFFER. The C reallocs when a long
# replacement overruns its estimate; the arena has no realloc and does not
# want one, so the first pass adds up exactly how much room the answer needs
# and the second writes it. The lookups happen twice, which is the price, and
# it buys an exact allocation and no copying.


def apy_translate_slot() -> ptr:
    """One word, for the width `apy_utf8_at_of` writes back."""
    return reserve("apy_translate_slot_ir", 8)


def apy_translate_to(table: ptr, cp: i64) -> ptr:
    """What `cp` maps to, or null for "not in the table -- keep it"."""
    at: i64 = apy_dict_find_of(table, apy_from_int(cp))
    if at < 0:
        return ptr(0)
    return ptr(load(u64, offset(
        ptr(load(u64, offset(table, apy_d_vals_offset()))),
        at * apy_value_size())))


def apy_translate_width(to: ptr) -> i64:
    """How many bytes a replacement contributes. -1 if it is not a legal one.

    `None` IS ZERO AND NOT AN ERROR: that is how a table spells a deletion.
    """
    if i64(load(i32, offset(to, 0))) == apy_none_kind():
        return 0
    if apy_is_int_like_of(to):
        ch: ptr = apy_chr(to)
        if not ch:
            return -1
        return load(i64, offset(ch, apy_str_len_offset()))
    if i64(load(i32, offset(to, 0))) == apy_str_kind():
        return load(i64, offset(to, apy_str_len_offset()))
    return -1


def apy_str_translate(s: ptr, table: ptr) -> ptr:
    """`s.translate(table)`."""
    if not apy_str_self_of(rodata(b"translate\0"), s):
        return ptr(0)
    # A BYTES RECEIVER IS A DIFFERENT METHOD, mapping bytes through a
    # 256-byte table rather than code points through a dict. See the bytes
    # family at the foot of this file.
    #
    # AN EMPTY DELETE SET, NOT None. `b.translate(table)` deletes nothing and
    # `b.translate(table, None)` is a TypeError -- the one-argument form
    # reaches HERE and the two-argument one reaches `apy_bytes_translate`
    # directly, which is the only thing that tells them apart. `b""` is the
    # default CPython's own signature declares.
    if i64(load(i32, offset(s, 0))) == apy_bytes_kind():
        # THE SHARED EMPTY BYTES, asked for rather than built. It used to be
        # `apy_from_bytes` with the bytes kind written over the cell, which
        # stopped being safe the moment that constructor began answering the
        # SHARED empty string: re-tagging it turned every later `""` in the
        # program into `b""`, silently and all at once.
        return apy_bytes_translate(s, table, apy_shared_bytes(256))
    if i64(load(i32, offset(table, 0))) != apy_dict_kind():
        return apy_raise_fmt(
            rodata(b"TypeError\0"),
            rodata(b"'%s' object is not subscriptable%s\0"),
            apy_kind_name_of(table), rodata(b"\0"))
    n: i64 = load(i64, offset(s, apy_str_len_offset()))
    p: ptr = ptr(load(u64, offset(s, apy_str_ptr_offset())))
    slot: ptr = apy_translate_slot()
    room: i64 = 0
    i: i64 = 0
    while i < n:
        cp: i64 = apy_utf8_at_of(p, n, i, slot)
        used: i64 = load(i64, slot)
        to: ptr = apy_translate_to(table, cp)
        if not to:
            room = room + used
        else:
            w: i64 = apy_translate_width(to)
            if w < 0:
                return apy_raise_at(
                    rodata(b"TypeError\0"),
                    rodata(b"character mapping must be in range(0x110000)\0"))
            room = room + w
        i = i + used
    buf: ptr = apy_alloc_bytes(room + 1)
    if not buf:
        return buf
    out: i64 = 0
    i = 0
    while i < n:
        cp2: i64 = apy_utf8_at_of(p, n, i, slot)
        used2: i64 = load(i64, slot)
        to2: ptr = apy_translate_to(table, cp2)
        if not to2:
            k: i64 = 0
            while k < used2:
                store(u8, load(u8, offset(p, i + k)), offset(buf, out))
                out = out + 1
                k = k + 1
        elif i64(load(i32, offset(to2, 0))) == apy_none_kind():
            out = out + 0
        elif apy_is_int_like_of(to2):
            made: ptr = apy_chr(to2)
            if not made:
                return made
            out = apy_text_into(buf, out, made)
        else:
            out = apy_text_into(buf, out, to2)
        i = i + used2
    store(u8, u8(0), offset(buf, out))
    return apy_from_bytes(buf, out)


# ── the bytes family ────────────────────────────────────────────────────────
#
# `bytes.translate` IS A DIFFERENT METHOD WEARING THE SAME NAME. `str` maps by
# code point through a DICT and may replace one character with a whole string;
# `bytes` maps by BYTE through a 256-BYTE TABLE and cannot change the length
# except by deleting. The signatures differ too -- `bytes.translate(table, /,
# delete=b'')` takes a second argument and `str.translate(table)` does not --
# and the receiver is not known until run time, so the split is made here
# rather than in the frontend.
#
# WHY THREE ENTRY POINTS AND NOT ONE. CPython gives three different messages
# for the three ways a call can be wrong, and a program may test for any of
# them:
#
#     "abc".translate(t, b"c")        str.translate() takes exactly one
#                                     argument (2 given)
#     "abc".translate(t, delete=b"c") str.translate() takes no keyword
#                                     arguments
#     b"abc".translate(t, b"c")       the answer
#
# The first two are the SAME CALL by the time the arguments are lowered -- a
# keyword folded into its slot is indistinguishable from the positional it
# stands for -- so the spelling has to survive as far as the symbol, which is
# what `apy_translate_kw` is for.


def apy_bytes_like_of(v: ptr) -> ptr:
    """`v` as bytes if it is bytes-like, else `v` itself.

    A MEMORYVIEW IS BYTES-LIKE and every other bytes method here already
    takes one -- `b"abcd".find(memoryview(b"bc"))` answers 1. Flattening it
    is what makes the table and the delete set accept one too; a view has an
    offset and a step, so its bytes are not laid out where a plain read would
    look for them.
    """
    if i64(load(i32, offset(v, 0))) == apy_mview_kind():
        return apy_mview_bytes(v)
    # AND A bytes SUBCLASS IS BYTES-LIKE. Written out rather than through a
    # str-and-bytes unwrap: the callers' refusals name what the program
    # WROTE, which is why each keeps the original value for the message.
    if i64(load(i32, offset(v, 0))) == apy_inst_kind():
        held: ptr = ptr(load(u64, offset(v, apy_o_held_offset())))
        if held:
            if i64(load(i32, offset(held, 0))) == apy_bytes_kind():
                return held
    return v


def apy_bytes_like_bad_of(v: ptr) -> ptr:
    """CPython's refusal for a table, a delete set or a maketrans argument
    that is not bytes."""
    return apy_raise_fmt(
        rodata(b"TypeError\0"),
        rodata(b"a bytes-like object is required, not '%s'%s\0"),
        apy_kind_name_of(v), rodata(b"\0"))


def apy_bytes_maketrans(a: ptr, b: ptr) -> ptr:
    """`bytes.maketrans(frm, to)` -- the 256-byte table `translate` wants.

    IDENTITY EVERYWHERE ELSE: the table is a full mapping, not a sparse one,
    so every byte the caller did not name maps to itself. That is why it must
    be exactly 256 long and why `translate` can refuse any other length.
    """
    frm: ptr = apy_bytes_like_of(a)
    to: ptr = apy_bytes_like_of(b)
    if i64(load(i32, offset(frm, 0))) != apy_bytes_kind():
        return apy_bytes_like_bad_of(frm)
    if i64(load(i32, offset(to, 0))) != apy_bytes_kind():
        return apy_bytes_like_bad_of(to)
    n: i64 = apy_str_byte_len(frm)
    if n != apy_str_byte_len(to):
        return apy_raise_at(
            rodata(b"ValueError\0"),
            rodata(b"maketrans arguments must have same length\0"))
    buf: ptr = apy_alloc_bytes(257)
    if not buf:
        return buf
    i: i64 = 0
    while i < 256:
        store(u8, u8(i), offset(buf, i))
        i = i + 1
    ap: ptr = apy_str_data(frm)
    bp: ptr = apy_str_data(to)
    i = 0
    while i < n:
        store(u8, load(u8, offset(bp, i)),
              offset(buf, i64(load(u8, offset(ap, i)))))
        i = i + 1
    store(u8, u8(0), offset(buf, 256))
    return apy_bytes_literal(buf, 256)


def apy_bytes_translate(s: ptr, table: ptr, delete: ptr) -> ptr:
    """`b.translate(table)` and `b.translate(table, delete)`.

    A NONE TABLE IS THE IDENTITY and not an error: `b.translate(None, b"a")`
    is the ordinary way to spell "delete these bytes and change nothing
    else", and it is the only reason the two-argument form is worth having
    over building a table that deletes.

    THE DELETE SET IS CONSULTED BEFORE THE TABLE, which is the order CPython
    uses and is observable: a byte that is both mapped and deleted is
    deleted.
    """
    if not apy_str_self_of(rodata(b"translate\0"), s):
        return ptr(0)
    if i64(load(i32, offset(s, 0))) == apy_str_kind():
        return apy_raise_at(
            rodata(b"TypeError\0"),
            rodata(b"str.translate() takes exactly one argument (2 given)\0"))
    map256: ptr = alloca(256)
    i: i64 = 0
    while i < 256:
        store(u8, u8(i), offset(map256, i))
        i = i + 1
    tab: ptr = apy_bytes_like_of(table)
    if i64(load(i32, offset(tab, 0))) != apy_none_kind():
        if i64(load(i32, offset(tab, 0))) != apy_bytes_kind():
            return apy_bytes_like_bad_of(tab)
        if apy_str_byte_len(tab) != 256:
            return apy_raise_at(
                rodata(b"ValueError\0"),
                rodata(b"translation table must be 256 characters long\0"))
        tp: ptr = apy_str_data(tab)
        i = 0
        while i < 256:
            store(u8, load(u8, offset(tp, i)), offset(map256, i))
            i = i + 1
    drop: ptr = alloca(256)
    i = 0
    while i < 256:
        store(u8, u8(0), offset(drop, i))
        i = i + 1
    # NO None HERE. A delete set that was WRITTEN must be bytes-like --
    # `b"abc".translate(None, None)` is `a bytes-like object is required, not
    # 'NoneType'` in CPython -- and the one-argument form passes an empty one
    # rather than a None.
    gone: ptr = apy_bytes_like_of(delete)
    if i64(load(i32, offset(gone, 0))) != apy_bytes_kind():
        return apy_bytes_like_bad_of(gone)
    dn: i64 = apy_str_byte_len(gone)
    dp: ptr = apy_str_data(gone)
    i = 0
    while i < dn:
        store(u8, u8(1), offset(drop, i64(load(u8, offset(dp, i)))))
        i = i + 1
    n: i64 = apy_str_byte_len(s)
    p: ptr = apy_str_data(s)
    buf: ptr = apy_alloc_bytes(n + 1)
    if not buf:
        return buf
    out: i64 = 0
    changed: i64 = 0
    i = 0
    while i < n:
        c: i64 = i64(load(u8, offset(p, i)))
        if i64(load(u8, offset(drop, c))) == 0:
            to: i64 = i64(load(u8, offset(map256, c)))
            if to != c:
                changed = 1
            store(u8, u8(to), offset(buf, out))
            out = out + 1
        else:
            changed = 1
        i = i + 1
    store(u8, u8(0), offset(buf, out))
    # NOTHING MOVED, SO THE ANSWER IS THE RECEIVER: `bytes_translate` ends
    # both of its loops with `if (!changed && PyBytes_CheckExact(input_obj))
    # return input_obj`, which is what makes `b.translate(None) is b` True --
    # the ordinary spelling of "delete these and change nothing else" with
    # nothing to delete. A table that maps every byte to itself and a delete
    # set that matches nothing reach it the same way, so the test is what the
    # walk DID and not what it was handed, and that is why `changed` is set
    # in the walk rather than guessed from the arguments.
    #
    # AND str.translate HAS NO SUCH SHORTCUT, which is the other half of this
    # row's asymmetry and is CPYTHON'S: `"abc".translate({}) is "abc"` is
    # False there, because `_PyUnicode_TranslateCharmap` builds its result
    # through a writer and hands back whatever it copied. A str receiver
    # never reaches this line anyway -- it was refused at the top -- and
    # `apy_str_translate` above must be left building.
    #
    # THE PREDICATE IS WHAT KEEPS A BYTEARRAY OUT. `ba.translate(None)` must
    # answer a fresh bytearray, since handing the receiver back would give
    # the program two names for one writable buffer; `mut` is the whole of
    # what separates it from the bytes beside it. The buffer is abandoned
    # rather than freed because the arena never takes one back.
    if changed == 0 and apy_str_may_return_self_of(s):
        return s
    # THE RESULT IS BUILT AS bytes HERE AND NOT LEFT TO `apy_str_like`, and
    # the reason is IDENTITY rather than tidiness. This used to end in
    # `apy_from_bytes`, which is the STR funnel: the call site's
    # `apy_str_like` then saw a str under a bytes receiver and re-tagged it
    # through `apy_bytes_made_of`, and that funnel answers the 256-object
    # cache for a one-byte result. So `b"abc".translate(None, b"bc") is b"a"`
    # was True on both compiled paths.
    #
    # CPYTHON SAYS False, and the asymmetry is `_PyBytes_Resize`'s:
    # `bytes_translate` sizes its output buffer to the input and then resizes,
    # and that function special-cases a new size of ZERO -- handing back the
    # interned empty -- while every other size is a plain `realloc` that never
    # reaches the cache. Measured both ways round against CPython 3.14
    # (`scratchpad/probes/d172.py`): `b"abc".translate(None, b"abc") is b""`
    # is True and `b"abc".translate(None, b"bc") is b"a"` is False. The
    # interpreter was given exactly this rule in d60f3f3d; this is the same
    # rule for the two compiled paths, and the C's `apy_bytes_translate` says
    # the same beside the same lines.
    #
    # AND THE STR SIDE IS NOT THE SAME, so do not tidy the two into one:
    # `"abc".translate(str.maketrans("", "", "bc")) is "a"` IS True, because
    # the unicode writer consults the latin-1 cache. `apy_str_translate` above
    # keeps its `apy_from_bytes` for that reason.
    #
    # ONE-BYTE SHARING CANNOT BE SETTLED IN `apy_str_like` EITHER, which is
    # why the fix is here. That funnel serves every bytes method at once, and
    # CPython shares by method: `b" a ".strip()`, `b"a-b".partition(b"-")[0]`,
    # `b"a b".split()[0]` and `b"abc"[0:1]` all ARE `b"a"`, while `translate`,
    # `replace` and `upper` are not. Measured.
    #
    # NOT `apy_bytes_made_of`: that is the funnel that consults the cache.
    # `apy_bytes_cell` is the constructor underneath it, and the empty one is
    # asked for by name -- a cell decided BEFORE it is made, never a shared
    # one re-tagged afterwards.
    if out == 0:
        return apy_shared_bytes(256)
    return apy_bytes_cell(buf, out, 0)


def apy_translate_kw(s: ptr, table: ptr, delete: ptr) -> ptr:
    """`x.translate(table, delete=...)` -- the form written with the keyword.

    THE ONLY DIFFERENCE IS THE MESSAGE. `str.translate` takes no keyword at
    all, and CPython says so rather than complaining about the count; by the
    time the arguments reach a runtime symbol the keyword has been folded
    into its slot and nothing left in the call records how it was written, so
    the frontend picks this symbol instead and the wording survives.
    """
    if i64(load(i32, offset(s, 0))) == apy_str_kind():
        return apy_raise_at(
            rodata(b"TypeError\0"),
            rodata(b"str.translate() takes no keyword arguments\0"))
    return apy_bytes_translate(s, table, delete)
