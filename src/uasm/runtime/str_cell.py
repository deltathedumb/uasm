# The string cell, written in the machine subset.
#
# STAGE 5 OF docs/INERT-RUNTIME.md, and the second kind to leave C. Read
# `int_cell.py` first: it is the same idiom and it explains why a constant is a
# function returning a literal (the static path has no module-level storage)
# and why this file is not importable under CPython (`i64` is not a name Python
# has, which is the honest signal that it is not host code).
#
# WHAT THIS FILE IS AND IS NOT. It is the CELL and the three constructors that
# only fill one in. It is NOT the string library: the case transforms, the
# search family and the predicates stay in C for now, and each is blocked on
# something different -- see "what stage 5 actually is" below.
#
# THE ANALYSIS THAT PRECEDED IT found that the document's plan for this stage
# was wrong in a way worth writing down. It says "a string's bytes are immortal
# too, so `str` needs nothing new", orders the work `str -> the allocator
# upgrade -> list`, and concludes the allocator is what stands between them.
# The first clause is true and the conclusion is not: nothing in `str` needs
# the allocator upgrade, and the allocator is not what is stopping `str`.
# What stops it is three other things, and only functions clear of all three
# can move now.

# ── the layout, which is C's ────────────────────────────────────────────────
#
# `struct { const char *p; int64_t n; int mut; } s;` -- the str/bytes arm of
# the union in `objects/csource.py`. As with `int_cell.py` these numbers are NOT
# read off by eye: `tests/uasm/integration/test_ported_int.py` compiles
# the real C and asserts each one with `offsetof`, and asserts the two kind
# constants with the enum's own values.
#
# THAT THE PROBE COVERS THE KINDS IS NEW, and it is the reason it was extended
# before a line of this file was written. A wrong OFFSET produces a crash or
# obvious rubbish. A wrong KIND produces a perfectly formed cell of the wrong
# type -- every field in the right place, the object simply not a str -- and
# the enum numbers one member explicitly and positions the other twenty-eight,
# with `APY_BYTES_K` inserted in the middle of it.


def apy_str_kind() -> i64:
    return 4


def apy_bytes_kind() -> i64:
    return 17


def apy_str_ptr_offset() -> i64:
    return 8


def apy_str_len_offset() -> i64:
    return 16


#: `bytearray` IS THIS FLAG AND NOTHING ELSE: same cell, same thirty shared
#: paths, writable buffer. Every constructor here leaves it zero, which
#: `apy_obj_alloc` gives for free by zeroing the payload -- the same thing
#: `int_cell.py` relies on for its cache slots. A constructor that left it
#: anything else would make a literal in read-only memory assignable.
def apy_str_mut_offset() -> i64:
    return 24


def apy_is_bytearray_of(v: ptr) -> i64:
    """Whether `v` is a bytearray: the bytes cell that admits it is writable.

    THE FLAG IS THE WHOLE OF WHAT SEPARATES THE TWO, so it is what every
    mutating method tests -- `bytes` reaching one of them has no such method
    and must say so rather than quietly rewrite a literal.
    """
    if i64(load(i32, offset(v, 0))) != apy_bytes_kind():
        return 0
    return i64(load(i32, offset(v, apy_str_mut_offset())))


def apy_byte_arg_of(v: ptr) -> i64:
    """The octet a bytearray method's argument stands for, or -1 raising.

    A BYTEARRAY HOLDS NUMBERS AND NOT ONE-BYTE STRINGS -- `b.append(b"z")` is
    a TypeError and `b.append(300)` a ValueError -- and `append`, `insert`
    and `remove` all say it the same way, which is why it is said once.

    A BIG IS OUT OF RANGE AND NOT A CRASH: `2 ** 100` is an int like any
    other to the caller, and reading its payload as a machine word would
    answer a number that is not in it.
    """
    if not apy_is_int_like_of(v):
        return apy_raise_int_arg(v)
    if apy_is_big_of(v):
        return apy_raise_byte_range()
    byte: i64 = apy_int_payload(v)
    if byte < 0 or byte > 255:
        return apy_raise_byte_range()
    return byte


def apy_raise_int_arg(v: ptr) -> i64:
    """CPython's refusal for a non-integer where an integer belongs."""
    apy_raise_fmt(
        rodata(b"TypeError\0"),
        rodata(b"'%s' object cannot be interpreted as an integer%s\0"),
        apy_kind_name_of(v), rodata(b"\0"))
    return -1


def apy_raise_byte_range() -> i64:
    """And its refusal for an integer that is not an octet."""
    apy_raise_at(rodata(b"ValueError\0"),
                 rodata(b"byte must be in range(0, 256)\0"))
    return -1


# ── construction ────────────────────────────────────────────────────────────
#
# THE BYTES ARE BORROWED, NEVER COPIED. Nothing in the cell records whether
# `p` points at a read-only global or at a heap buffer, so nothing can free it
# and nothing does -- which is exactly why these three can be written against a
# bump-pointer arena that cannot free. The constructors that OWN their bytes
# (`apy_str_take`, `apy_str_copy`) are `static` in the C, so the subset cannot
# even name them yet; promoting them is a later step and a larger one.
#
# THE TRAILING NUL IS THE CALLER'S. Every producer NUL-terminates, and the
# remaining C reads `v.s.p` as a C string in two hundred places -- `APY_CSTR`,
# `strcmp`, `snprintf`, `strtod`. The terminator is load-bearing without being
# part of the value, so the LENGTH stored here never counts it.


def apy_str_cell(p: ptr, n: i64) -> ptr:
    """A str cell over `n` bytes at `p`, borrowed, WITHOUT the shared test.

    The whole constructor: allocate, tag, store two fields. `mut` is left zero
    by the arena rather than written, which is the difference between a str
    and a bytearray.

    SPLIT OUT OF `apy_from_bytes` so that `apy_shared_str` has something to
    build its cells with: the constructor asks the table first, so the one
    call that must not is the table's own.
    """
    cell: ptr = apy_obj_alloc(apy_str_kind())
    if not cell:
        return cell
    store(u64, u64(p), offset(cell, apy_str_ptr_offset()))
    store(i64, n, offset(cell, apy_str_len_offset()))
    return cell


def apy_bytes_cell(p: ptr, n: i64, mut: i64) -> ptr:
    """The same cell with the `bytes` tag and a chosen `mut`, no test either.

    `mut` IS A PARAMETER AND NOT SOMETHING THE CALLER WRITES AFTERWARDS: a
    bytearray is written into, so it can never be one of the shared cells,
    and asking for the flag here makes "a fresh cell" and "a bytearray" one
    decision. See `apy_bytes_own` in `objects/c/_core.py`, which is the same
    function on the other side.
    """
    cell: ptr = apy_obj_alloc(apy_bytes_kind())
    if not cell:
        return cell
    store(u64, u64(p), offset(cell, apy_str_ptr_offset()))
    store(i64, n, offset(cell, apy_str_len_offset()))
    store(i32, i32(mut), offset(cell, apy_str_mut_offset()))
    return cell


def apy_from_bytes(p: ptr, n: i64) -> ptr:
    """A str cell over `n` bytes at `p`, borrowed -- or the SHARED one.

    THE ONE PLACE THE SHARED STRINGS ARE ANSWERED FROM. Every string this
    runtime builds is a cell and every cell is made here, so putting the test
    in the constructor is what makes `"" is str()`, `"a" is chr(97)` and
    `"abc"[0:1] is "a"` all True without fifty call sites knowing about it.

    THE BUFFER IS ABANDONED when a shared cell answers. It is arena storage,
    which this runtime never releases anyway.
    """
    if n == 0:
        return apy_shared_str(256)
    cp: i64 = apy_shared_cp_of(p, n)
    if cp >= 0:
        return apy_shared_str(cp)
    return apy_str_cell(p, n)


def apy_bytes_literal(p: ptr, n: i64) -> ptr:
    """The same cell with the `bytes` tag.

    A SEPARATE FUNCTION RATHER THAN A FLAG, because that is what the C has and
    the two are reached from different places in the frontend. The only
    difference is the kind -- and the table it asks, since CPython keeps one
    empty bytes and one per octet as well.
    """
    if n == 0:
        return apy_shared_bytes(256)
    if n == 1:
        return apy_shared_bytes(i64(load(u8, p)))
    return apy_bytes_cell(p, n, 0)


# ── the shared empty and one-character strings ──────────────────────────────
#
# CPython keeps one empty string and one string per latin-1 character, and a
# program SEES it: `"" is str()`, `"a" is chr(97)` and `"abc"[0:1] is "a"` are
# all True, while a character above U+00FF is not cached, so
# `chr(256) is chr(256)` is False.
#
# 257 SLOTS, the last one for the empty string, plus three bytes per character
# for the UTF-8 it is spelled with and its terminator. `reserve` gives static
# storage that is ZEROED, so a null slot means "not built yet" and no separate
# flag is needed -- the same trick `int_cell.py` plays for the small integers.
#
# THIS DISPLACES THE C'S TABLE AND DOES NOT SIT BESIDE IT. A build must have
# exactly one: `apy_str_take` stays in C and `apy_str_copy_bytes` is this
# file's, and two tables would make a literal built through one and a slice
# built through the other two empty strings that compare unequal. The C's
# `apy_shared_str` is an `APY_API` for that reason and this name is in
# `REPLACES["str_cell.py"]`.


def apy_str_shared_slot(cp: i64) -> ptr:
    return offset(reserve("apy_str_shared_ir", 2056), cp * 8)


def apy_shared_utf8_slot(cp: i64) -> ptr:
    return offset(reserve("apy_shared_utf8_ir", 768), cp * 3)


def apy_shared_str(cp: i64) -> ptr:
    """The shared string for one code point, or for NOTHING at index 256."""
    slot: ptr = apy_str_shared_slot(cp)
    shared: ptr = ptr(load(u64, slot))
    if shared:
        return shared
    if cp == 256:
        shared = apy_str_cell(rodata(b"\0"), 0)
        store(u64, u64(shared), slot)
        return shared
    b: ptr = apy_shared_utf8_slot(cp)
    n: i64 = 1
    if cp < 128:
        store(u8, u8(cp), b)
    else:
        store(u8, u8(192 + (cp >> 6)), b)
        store(u8, u8(128 + (cp & 63)), offset(b, 1))
        n = 2
    store(u8, u8(0), offset(b, n))
    shared = apy_str_cell(b, n)
    store(u64, u64(shared), slot)
    return shared


def apy_bytes_shared_slot(b: i64) -> ptr:
    return offset(reserve("apy_bytes_shared_ir", 2056), b * 8)


def apy_shared_byte_slot(b: i64) -> ptr:
    return offset(reserve("apy_shared_byte_ir", 512), b * 2)


def apy_shared_bytes(b: i64) -> ptr:
    """The shared bytes for one octet, or for NOTHING at index 256."""
    slot: ptr = apy_bytes_shared_slot(b)
    shared: ptr = ptr(load(u64, slot))
    if shared:
        return shared
    if b == 256:
        shared = apy_bytes_cell(rodata(b"\0"), 0, 0)
        store(u64, u64(shared), slot)
        return shared
    one: ptr = apy_shared_byte_slot(b)
    store(u8, u8(b), one)
    store(u8, u8(0), offset(one, 1))
    shared = apy_bytes_cell(one, 1, 0)
    store(u64, u64(shared), slot)
    return shared


def apy_bytes_made_of(p: ptr, n: i64, mut: i64) -> ptr:
    """`n` bytes COPIED into a bytes cell, shared when it may be shared.

    WHAT THE TWO PLACES THAT USED TO RE-TAG A STRING NOW CALL. Building a str
    and writing the bytes kind over it was safe while every constructor
    answered a fresh cell; it stopped being safe the moment `apy_from_bytes`
    began answering shared ones, because re-tagging the shared empty STRING
    turns every later `""` into `b""` at once, everywhere.
    """
    if mut == 0:
        if n == 0:
            return apy_shared_bytes(256)
        if n == 1:
            return apy_shared_bytes(i64(load(u8, p)))
    buf: ptr = apy_alloc_bytes(n + 1)
    if not buf:
        return buf
    i: i64 = 0
    while i < n:
        store(u8, load(u8, offset(p, i)), offset(buf, i))
        i = i + 1
    store(u8, u8(0), offset(buf, n))
    return apy_bytes_cell(buf, n, mut)


def apy_shared_cp_of(p: ptr, n: i64) -> i64:
    """The code point `n` bytes spell when the runtime shares that string.

    -1 FOR EVERYTHING ELSE -- more than one character, or a character above
    U+00FF. One latin-1 character is one or two UTF-8 bytes, which is the
    whole of what the two arms decode.
    """
    if n == 1:
        lead: i64 = i64(load(u8, p))
        if lead < 128:
            return lead
        return -1
    if n == 2:
        b0: i64 = i64(load(u8, p))
        b1: i64 = i64(load(u8, offset(p, 1)))
        if (b0 & 224) == 192:
            if (b1 & 192) == 128:
                cp: i64 = ((b0 & 31) << 6) | (b1 & 63)
                if cp >= 128:
                    if cp < 256:
                        return cp
    return -1


def apy_str_copy_bytes(p: ptr, n: i64) -> ptr:
    """`n` bytes COPIED into storage the cell owns, plus a terminator.

    THE ONE THE RUNTIME BUILDS EVERY STRING WITH. Twenty-four places in the C
    reach it through the `apy_str_copy` shim -- every slice, join, case
    transform, repr and format -- so this is the function that decides where a
    compiled program's strings live. It was `malloc`; it is the arena now,
    which is what "the object runtime allocates from one place" finally means
    for a string's BYTES and not just for its cell.

    A BUMP POINTER CANNOT FREE, AND THAT COSTS NOTHING HERE -- checked rather
    than assumed. Nothing in the C frees `v.s.p`: all 51 `free()` calls
    release transient locals, never the buffer handed to `apy_str_take`. A
    string's bytes already lived until the program exited, so this changes the
    allocator and not the lifetime.

    THE TERMINATOR IS WRITTEN AND IS NOT PART OF THE LENGTH. Two hundred
    places in the remaining C read `v.s.p` as a C string, so a cell built
    without one is a cell the rest of the runtime reads off the end of. The
    arena is asked for `n + 1` for exactly that byte.
    """
    if n == 0:
        return apy_shared_str(256)
    cp: i64 = apy_shared_cp_of(p, n)
    if cp >= 0:
        return apy_shared_str(cp)
    buf: ptr = apy_alloc_bytes(n + 1)
    if not buf:
        return buf
    i: i64 = 0
    while i < n:
        store(u8, load(u8, offset(p, i)), offset(buf, i))
        i = i + 1
    store(u8, u8(0), offset(buf, n))
    return apy_from_bytes(buf, n)


def apy_from_cstr(p: ptr) -> ptr:
    """A str cell over a NUL-terminated C string.

    `strlen`, WRITTEN OUT, because it is the one thing the C had here that the
    IR does not. A byte at a time is what `strlen` is; a backend that has a
    faster one is free to recognise the loop.

    THE COMPILER NO LONGER CALLS THIS FOR ITS OWN LITERALS. It used to, and
    that was a live wrong answer: the length was known at compile time, thrown
    away, and re-derived by scanning to the first NUL -- so `len("a\\0b")`
    answered 1 where Python says 3. `dynamic._dyn_str_literal` passes the
    length it already counted. This stays because a C string arriving from
    outside the program is still a real case.
    """
    n: i64 = 0
    while load(u8, offset(p, n)) != 0:
        n = n + 1
    return apy_from_bytes(p, n)


# ── the two wrappers that CANNOT move, and why ─────────────────────────────
#
# `apy_str_copy` and `apy_bytes_copy` were written, declared, and taken back
# out. They look like the easiest ports left -- each is one call to
# `apy_str_copy_bytes`, which is already IR -- and the C explains in its own
# words why they are not:
#
#     The parameter is an `apy_value` and not a `const char *` because that is
#     what an IR `ptr` compiles to: a ported definition emits
#     `uintptr_t apy_str_copy_bytes(uintptr_t, int64_t)`, and a C prototype
#     spelling the first argument as a pointer is a CONFLICTING TYPE where gcc
#     sees both.
#
# THE SPLIT IS ALREADY THE ANSWER. `apy_str_copy_bytes` exists because the
# half that ALLOCATES could take an `apy_value` and move; `apy_str_copy` keeps
# `const char *` so that the twenty-four call sites in the C -- every slice,
# join, case transform and repr -- did not each need a cast. Porting the
# wrapper would undo the arrangement that made porting the body possible.
#
# HOW IT SURFACED, which is the part worth keeping: not as a wrong answer but
# as `apy_str_copy used but never defined` at link time, because the C omitted
# the body it was told was ported while gcc refused the mismatched prototype.
# A signature that disagrees across the boundary fails loudly, which is the
# one thing this whole arrangement gets for free.


# -- bytes.hex() and bytes.fromhex(), which the arena made sayable ----------
#
# NEITHER NEEDED LIBC. The C reaches `malloc` for a working buffer and
# `fputs`/`exit` for the out-of-memory abort beside it; the arena answers null
# and the caller propagates, which is the convention every ported function
# here already follows -- so the abort path simply does not exist.


def apy_hex_digit(d: i64) -> i64:
    """One lowercase hex digit, as a byte."""
    if d < 10:
        return 48 + d
    return 87 + d


def apy_bytes_hex(b: ptr, sep: ptr) -> ptr:
    """`b.hex()` and `b.hex(sep)`. ONE BYTE PER GROUP, which is the default
    CPython declares -- see `apy_bytes_hex_n`, which both spellings are. The
    count slot is EMPTY because this call did not write one."""
    return apy_bytes_hex_n(b, sep, ptr(0))


def apy_bytes_hex_n(b: ptr, sep: ptr, perv: ptr) -> ptr:
    """`b.hex()`, `b.hex(sep)` and `b.hex(sep, bytes_per_sep)`.

    THE GROUPING IS COUNTED FROM AN END AND WHICH END IS THE SIGN. A
    positive count groups from the RIGHT -- `bytes(range(1, 8)).hex(":", 2)`
    is `01:0203:0405:0607`, the leftover at the front -- and a negative one
    from the LEFT. That is CPython's rule and it is not the obvious one: the
    common use is a number written down, whose low end is the one that
    matters.

    THE BUFFER IS SIZED FOR THE WORST CASE -- three bytes per input byte,
    which is two digits and a separator -- and the terminator makes it
    `n * 3 + 2`. Over-allocating a few bytes in a bump arena costs a few
    bytes.
    """
    # A VIEW HEXES THE BYTES IT SHOWS, which is most of what a program makes
    # one to look at. The no-separator form is converted the same way.
    held: ptr = b
    if i64(load(i32, offset(held, 0))) == apy_mview_kind():
        held = apy_mview_bytes(held)
        if not held:
            return held
    if i64(load(i32, offset(held, 0))) != apy_bytes_kind():
        return apy_raise_fmt(
            rodata(b"AttributeError\0"),
            rodata(b"'%s' object has no attribute 'hex'%s\0"),
            apy_kind_name_of(b), rodata(b"\0"))
    # AN EMPTY SLOT IS ONE THE CALL DID NOT WRITE, and it is the only way to
    # tell `b.hex()` from `b.hex(None)` -- the first is the no-argument form
    # whose default this fills in, the second a separator CPython refuses.
    #
    # THE COUNT IS CONVERTED FIRST, which is CPython's order: `b.hex(None,
    # None)` complains about the integer and `b.hex(None)` about the
    # separator's length.
    per: i64 = 1
    if perv:
        pk: i64 = i64(load(i32, offset(perv, 0)))
        if pk != apy_int_kind():
            if pk != apy_big_kind():
                if pk != apy_bool_kind():
                    return apy_raise_fmt(
                        rodata(b"TypeError\0"),
                        rodata(b"'%s' object cannot be interpreted as an "
                               b"integer%s\0"),
                        apy_kind_name_of(perv), rodata(b"\0"))
        per = apy_as_int(perv)
        if apy_error_occurred():
            return ptr(0)
    s: i64 = 0
    if sep:
        # CPython ASKS THE SEPARATOR FOR ITS LENGTH, so anything without one
        # is a TypeError about `len` rather than about hex.
        sk: i64 = i64(load(i32, offset(sep, 0)))
        if sk != apy_str_kind():
            if sk != apy_bytes_kind():
                return apy_raise_fmt(
                    rodata(b"TypeError\0"),
                    rodata(b"object of type '%s' has no len()%s\0"),
                    apy_kind_name_of(sep), rodata(b"\0"))
        if load(i64, offset(sep, apy_str_len_offset())) != 1:
            return apy_raise_fmt(
                rodata(b"ValueError\0"), rodata(b"sep must be length 1.\0"),
                rodata(b"\0"), rodata(b"\0"))
        s = i64(load(u8, ptr(load(u64, offset(
            sep, apy_str_ptr_offset())))))
    # A NEGATIVE COUNT GROUPS FROM THE LEFT, a positive one from the right,
    # and zero -- or one no smaller than the length -- separates nothing.
    group: i64 = per
    if per < 0:
        group = -per
    n: i64 = load(i64, offset(held, apy_str_len_offset()))
    buf: ptr = apy_alloc_bytes(n * 3 + 2)
    if not buf:
        return buf
    src: ptr = ptr(load(u64, offset(held, apy_str_ptr_offset())))
    out: i64 = 0
    i: i64 = 0
    while i < n:
        c: i64 = i64(load(u8, offset(src, i)))
        if s:
            if i:
                if group:
                    at: i64 = (n - i) % group
                    if per < 0:
                        at = i % group
                    if at == 0:
                        store(u8, u8(s), offset(buf, out))
                        out = out + 1
        store(u8, u8(apy_hex_digit(c >> 4)), offset(buf, out))
        store(u8, u8(apy_hex_digit(c & 15)), offset(buf, out + 1))
        out = out + 2
        i = i + 1
    store(u8, u8(0), offset(buf, out))
    return apy_from_bytes(buf, out)


def apy_hex_value(c: i64) -> i64:
    """A hex digit's value, or -1 if it is not one."""
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 97 + 10
    if c >= 65 and c <= 70:
        return c - 65 + 10
    return -1


def apy_bytes_fromhex(self: ptr, text: ptr) -> ptr:
    """`bytes.fromhex(text)`.

    WHITESPACE BETWEEN PAIRS IS SKIPPED, which is Python\'s rule and is what
    lets a hex dump be pasted in. Whitespace INSIDE a pair is not special --
    it is skipped too, so `"a b"` is a single odd digit and refused.

    AN ODD NUMBER OF DIGITS IS AN ERROR, caught after the walk: a trailing
    high nibble with nothing to pair it with is exactly that.
    """
    # A FLOAT'S `fromhex` IS A DIFFERENT READING and is NOT SPLIT HERE.
    # The ported runtime has to DEFINE everything it calls -- see
    # `test_the_allocator_asks_the_floor_and_nothing_else` -- and the hex
    # float parser is the C's. `apy_any_fromhex` is where the receiver
    # decides, and it is reached from the call site rather than from here.
    if i64(load(i32, offset(text, 0))) != apy_str_kind():
        return apy_raise_at(
            rodata(b"TypeError\0"),
            rodata(b"fromhex() argument must be str\0"))
    n: i64 = load(i64, offset(text, apy_str_len_offset()))
    buf: ptr = apy_alloc_bytes(n // 2 + 2)
    if not buf:
        return buf
    src: ptr = ptr(load(u64, offset(text, apy_str_ptr_offset())))
    out: i64 = 0
    hi: i64 = -1
    i: i64 = 0
    while i < n:
        c: i64 = i64(load(u8, offset(src, i)))
        if c == 32 or c == 9 or c == 10:
            i = i + 1
        else:
            d: i64 = apy_hex_value(c)
            if d < 0:
                return apy_raise_at(
                    rodata(b"ValueError\0"),
                    rodata(b"non-hexadecimal number found in "
                           b"fromhex() arg\0"))
            if hi < 0:
                hi = d
            else:
                store(u8, u8((hi << 4) | d), offset(buf, out))
                out = out + 1
                hi = -1
            i = i + 1
    if hi >= 0:
        return apy_raise_at(
            rodata(b"ValueError\0"),
            rodata(b"non-hexadecimal number found in fromhex() arg\0"))
    store(u8, u8(0), offset(buf, out))
    # AND THE ANSWER IS THE KIND IT WAS REACHED THROUGH, which is what makes
    # `bytearray().fromhex(s)` answer a bytearray.
    #
    # DECIDED BEFORE THE CELL IS MADE AND NOT WRITTEN ONTO IT AFTERWARDS.
    # `apy_bytes_literal` answers a SHARED cell for the empty and one-byte
    # values, so setting `mut` on what it returned turned every later `b"A"`
    # in the program -- literals included -- into a bytearray. That is the
    # rule `apy_bytes_cell`'s own docstring states, and this was the one
    # place still breaking it.
    mut: i64 = 0
    if self:
        if i64(load(i32, offset(self, 0))) == apy_bytes_kind():
            if load(i32, offset(self, apy_s_mut_offset())):
                mut = 1
    if mut:
        return apy_bytes_cell(buf, out, 1)
    return apy_bytes_literal(buf, out)
