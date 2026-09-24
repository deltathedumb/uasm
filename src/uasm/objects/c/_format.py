"""The object runtime, in C: the format mini-language.

ONE PART OF ONE TRANSLATION UNIT. `c/__init__.py` concatenates
these in order and the result is the file it always was, so a
definition here may rely on anything in an earlier part and
nothing in a later one. Sections, in order:
  * format specs
"""

C = r"""/* --- format specs ------------------------------------------------------- */
/* `format(v, spec)`, `f"{v:spec}"` and `"{:spec}".format(v)` are ONE function
   because they are one language: the mini-language of PEP 3101, spelled

       [[fill]align][sign][#][0][width][grouping][.precision][type]

   Written out here rather than handed to `printf` because three parts of it
   have no printf equivalent -- `^` centring, `,` grouping, and the `=` align
   that puts padding between a sign and its digits -- and because a spec is
   USER INPUT, so translating it into a printf format string would be a way
   for a program to hand `%n` to the C library. */

typedef struct {
    /* THE FILL IS A CHARACTER AND NOT A BYTE. `"{:é>7}"` is an ordinary spec
       in Python and was refused here as invalid, because one `char` cannot
       hold `é` and the align was looked for at byte 1, which is the middle
       of it. */
    const char *fill;
    int filln;
    char align, sign, type, group;
    /* PEP 682's `z`: a negative zero formats as a POSITIVE one. It sits
       between the sign and the `#`, and it is about the VALUE rather than
       about the padding, which is why it is a flag of its own. */
    int alt, zero, width, precision, has_precision, coerce_zero;
} apy_spec;

static int apy_spec_parse(const char *p, int64_t n, apy_spec *out) {
    int64_t i = 0;
    int64_t fw;
    out->fill = " "; out->filln = 1;
    out->align = 0; out->sign = 0; out->type = 0;
    out->group = 0; out->alt = 0; out->zero = 0; out->width = 0;
    out->precision = 0; out->has_precision = 0; out->coerce_zero = 0;
    /* FILL is only a fill when an align follows it, which is why position 1 is
       examined before position 0: in `{:<5}` the `<` is the align and in
       `{:*<5}` the `*` is the fill. */
    /* HOW WIDE THE FIRST CHARACTER IS, from its lead byte alone -- which is
       all UTF-8 needs to say where the next one starts. Written out rather
       than borrowed: the decoder lives in a part that comes after this one. */
    fw = 0;
    if (n) {
        unsigned char c0 = (unsigned char)p[0];
        fw = c0 >= 0xF0 ? 4 : c0 >= 0xE0 ? 3 : c0 >= 0xC0 ? 2 : 1;
        if (fw > n) fw = n;
    }
    if (n > fw && (p[fw] == '<' || p[fw] == '>' || p[fw] == '^'
                   || p[fw] == '=')) {
        out->fill = p; out->filln = (int)fw; out->align = p[fw];
        i = fw + 1;
    } else if (n >= 1 && (p[0] == '<' || p[0] == '>' || p[0] == '^'
                          || p[0] == '=')) {
        out->align = p[0]; i = 1;
    }
    if (i < n && (p[i] == '+' || p[i] == '-' || p[i] == ' ')) out->sign = p[i++];
    /* PEP 682, between the sign and the `#`. */
    if (i < n && p[i] == 'z') { out->coerce_zero = 1; i++; }
    if (i < n && p[i] == '#') { out->alt = 1; i++; }
    if (i < n && p[i] == '0') {
        /* A leading zero means `0=` -- padding between the sign and the
           digits -- unless an explicit align already said otherwise. */
        out->zero = 1;
        if (!out->align) { out->align = '='; out->fill = "0"; out->filln = 1; }
        i++;
    }
    while (i < n && p[i] >= '0' && p[i] <= '9')
        out->width = out->width * 10 + (p[i++] - '0');
    if (i < n && (p[i] == ',' || p[i] == '_')) out->group = p[i++];
    if (i < n && p[i] == '.') {
        i++;
        out->has_precision = 1;
        while (i < n && p[i] >= '0' && p[i] <= '9')
            out->precision = out->precision * 10 + (p[i++] - '0');
    }
    if (i < n) out->type = p[i++];
    return i == n;
}

/* WHICH TYPES A GROUPING CHARACTER MAY GO WITH, worded as CPython words it,
   or 0 for a combination it allows.

   `,` IS DECIMAL ONLY and `_` is not: `format(x, "_x")` separates hex digits
   in fours while `format(x, ",x")` is refused outright, which is a rule
   about the SEPARATOR and not only about the base. And `n` takes neither,
   because its whole job is to group the way a locale says. All three were
   accepted here and quietly did something. */
static const char *apy_group_refusal(const apy_spec *sp) {
    if (!sp->group) return 0;
    if (sp->type == 'n')
        return sp->group == ',' ? "Cannot specify ',' with 'n'."
                                : "Cannot specify '_' with 'n'.";
    if (sp->group == ',' && (sp->type == 'b' || sp->type == 'o'
                             || sp->type == 'x' || sp->type == 'X'))
        return sp->type == 'b' ? "Cannot specify ',' with 'b'."
             : sp->type == 'o' ? "Cannot specify ',' with 'o'."
             : sp->type == 'x' ? "Cannot specify ',' with 'x'."
                               : "Cannot specify ',' with 'X'.";
    return 0;
}

/* Insert `group` every `every` digits of `body`, from the right, in place.
   The caller owns a buffer with room for the separators.

   EVERY FOUR IN A NON-DECIMAL BASE, which is CPython's rule and was not this
   one's: `format(0x123456789, "_x")` is `1_2345_6789` there and was
   `1_2345_6789`'s three-digit cousin here. Decimal groups by three, and the
   base is the only thing that decides, so the caller passes it.

   AND NO CAP. This kept a 160-byte scratch buffer and simply DID NOT GROUP a
   body longer than 120 digits, which is a wrong answer for every big integer
   past that -- `format(10 ** 301, ",d")` came back unseparated. The buffer
   the caller already sized for the separators is written backwards in place
   instead, so there is no second one to overflow. */
static int64_t apy_group_digits(char *body, int64_t n, char group, int every) {
    int64_t sep = (n - 1) / every, out = n + sep, i, at = out - 1, k = 0;
    if (sep <= 0) return n;
    for (i = n - 1; i >= 0; i--) {
        if (k == every) { body[at--] = group; k = 0; }
        body[at--] = body[i];
        k++;
    }
    return out;
}

/* How many digits a group holds for a spec's type: four for the non-decimal
   bases, three for everything else. */
static int apy_group_every(char type) {
    return (type == 'b' || type == 'o' || type == 'x' || type == 'X') ? 4 : 3;
}

/* Pad `body` to the spec's width under its align, and hand back a str.

   `=` splits: the sign stays at the front and the fill goes between it and the
   digits, which is what makes `{:08.2f}` of -1.5 come out `-0001.50` and not
   `000-1.50`. */
static apy_value apy_spec_pad(const char *body, int64_t n, const apy_spec *sp,
                         int numeric) {
    int64_t width = sp->width, pad, left, i, out = 0, signlen = 0, chars = 0;
    char align = sp->align;
    char *buf;
    if (!align) align = numeric ? '>' : '<';
    /* A WIDTH IS A CHARACTER COUNT. Comparing it against the BYTE length
       under-padded every non-ASCII body -- `"{:>7}".format("éàb")` came out
       unpadded, because three characters in five bytes looked wide enough. A
       continuation byte is `10xxxxxx` and every other byte starts a
       character. */
    for (i = 0; i < n; i++)
        if (((unsigned char)body[i] & 0xC0) != 0x80) chars++;
    if (width <= chars) return apy_str_copy(body, n);
    pad = width - chars;
    buf = (char *)malloc((size_t)(n + pad * sp->filln) + 1);
    if (!buf) { fputs("uasm: out of memory\n", stderr); exit(1); }
    if (align == '=') {
        if (n && (body[0] == '-' || body[0] == '+' || body[0] == ' '))
            signlen = 1;
        /* AND AFTER THE BASE PREFIX, which is the other thing that has to
           stay at the front. `format(255, "#030x")` is
           `0x00000000000000000000000000ff` in CPython and was
           `000000000000000000000000000xff` here -- the `0x` left stranded in
           the middle of the fill, which reads as neither a prefix nor a
           digit. `#` only ever writes `0` and then the type character, so
           that is exactly what is looked for. */
        if (n - signlen >= 2 && body[signlen] == '0'
                && (body[signlen + 1] == 'x' || body[signlen + 1] == 'X'
                    || body[signlen + 1] == 'o' || body[signlen + 1] == 'b'))
            signlen += 2;
        memcpy(buf, body, (size_t)signlen);
        out = signlen;
        for (i = 0; i < pad; i++) { memcpy(buf + out, sp->fill,
            (size_t)sp->filln); out += sp->filln; }
        memcpy(buf + out, body + signlen, (size_t)(n - signlen));
        out += n - signlen;
    } else if (align == '>') {
        for (i = 0; i < pad; i++) { memcpy(buf + out, sp->fill,
            (size_t)sp->filln); out += sp->filln; }
        memcpy(buf + out, body, (size_t)n);
        out += n;
    } else if (align == '^') {
        left = pad / 2;
        for (i = 0; i < left; i++) { memcpy(buf + out, sp->fill,
            (size_t)sp->filln); out += sp->filln; }
        memcpy(buf + out, body, (size_t)n);
        out += n;
        for (i = 0; i < pad - left; i++) { memcpy(buf + out, sp->fill,
            (size_t)sp->filln); out += sp->filln; }
    } else {
        memcpy(buf, body, (size_t)n);
        out = n;
        for (i = 0; i < pad; i++) { memcpy(buf + out, sp->fill,
            (size_t)sp->filln); out += sp->filln; }
    }
    buf[out] = 0;
    return apy_str_take(buf, out);
}

/* An unsigned integer in `base`, most significant digit first. */
static int64_t apy_int_digits(char *buf, uint64_t mag, int base, int upper) {
    const char *digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    char rev[80];
    int64_t n = 0, i;
    if (!mag) rev[n++] = '0';
    while (mag) {
        rev[n++] = digits[mag % (unsigned)base];
        mag /= (unsigned)base;
    }
    for (i = 0; i < n; i++) buf[i] = rev[n - 1 - i];
    return n;
}

static apy_value apy_bad_code(char code, apy_value v) {
    char c[2];
    c[0] = code ? code : 's';
    c[1] = 0;
    return apy_fail2("ValueError",
                     "Unknown format code '%s' for object of type '%s'",
                     c, apy_kind_name(v));
}

APY_API apy_value apy_format(apy_value v, apy_value spec) {
    apy_spec sp;
    const char *sptr = O(spec)->kind == APY_STR_K ? APY_CSTR(spec) : "";
    int64_t slen = O(spec)->kind == APY_STR_K ? O(spec)->v.s.n : 0;
    char body[600];
    int64_t n = 0;

    /* A user object formats ITSELF, given the spec, and is asked BEFORE the
       spec is parsed and before the empty-spec shortcut: `f"{obj}"` is
       `format(obj, "")`, which calls `__format__("")` -- not `str(obj)`, and
       a class defining both can tell the difference. */
    /* EXCEPT AN INSTANCE OF `object` ITSELF, whose class is the one that
       carries `object`'s own `__format__` -- which is this function. It is
       in that dict because `dir(object)` is the dict's keys, and asking it
       here would be asking this function again with no base case. `object`
       is not a real base on anything (see `apy_object_class`), so a plain
       `object()` is the whole of what this excludes, and for one of those
       `object.__format__` means exactly what the rest of this body does. */
    if (O(v)->kind == APY_INST_K && O(v)->v.o.cls != apy_object_class()) {
        apy_value r = apy_method1(v, "__format__", spec);
        if (r || apy_error_occurred()) return r;
    }
    /* AND `object.__format__` REFUSES A NON-EMPTY SPEC, which is the whole of
       what tells it apart from `str(self)`. `object___format___impl`
       (Objects/typeobject.c) is four lines: when the spec has any length at
       all it raises `TypeError: unsupported format string passed to
       <type>.__format__`, naming `Py_TYPE(self)->tp_name`, and otherwise it
       calls `PyObject_Str`. Measured against CPython 3.14 with
       `class C: pass` -- `format(c, ">30")`, `format(c, "s")`, `format(c,
       ".2")`, `f"{c:>300}"` and `"{:>300}".format(c)` are all that one
       TypeError, where this answered `ValueError: Unknown format code 's'
       for object of type 'C'` for most specs and a padded repr for `"s"`.

       AN INSTANCE HOLDING A BUILTIN IS NOT ONE OF THESE and is why `held`
       is asked: `class S(str)` reaches `str.__format__`, which takes the
       whole mini-language, and the branch below is that. A plain instance
       is the one that has `object`'s own.

       THE `object()` CASE IS WHY THIS IS NEEDED HERE RATHER THAN LEFT TO
       THE FALL-THROUGH. Since `object`'s dict was widened to the twenty-four
       names `dir(object)` answers, `object()` has a class that provides
       `__format__` -- so the hook above has to skip it or ask this function
       again, and what it skipped into was a body with no arm for an
       instance. The interpreter's twin of this trapped on a null. */
    /* `str` IS THE ONLY BUILTIN BASE WITH A `__format__` OF ITS OWN. bytes,
       bytearray, dict, list, tuple and set all leave `tp_format` at
       object's, so a non-empty spec on an instance of a class extending one
       of them is `object.__format__`'s refusal -- which names the CLASS, as
       `apy_kind_name` already does. Testing "holds nothing" let
       `format(B(b"Ab"), "5")` fall into str's mini-language below, where it
       reported `Unknown format code 's' for object of type 'B'`: a
       ValueError about the spec, where CPython raises a TypeError about the
       type not having a `__format__` at all. */
    if (O(v)->kind == APY_INST_K && slen
            && (!O(v)->v.o.held
                || O(O(v)->v.o.held)->kind != APY_STR_K))
        return apy_fail2("TypeError",
                         "unsupported format string passed to "
                         "%s.__format__%s", apy_kind_name(v), "");
    /* An EMPTY spec is `str(v)` and nothing else, which is the whole of the
       rule for it: `str.__format__` with an empty spec is
       `unicode_result_unchanged`, which hands an EXACT str back and copies
       for anything else -- and `apy_str` draws that same line for this
       runtime, through `apy_inst_text_result`. So `format(s) is s` is True,
       `format(x) is x` is False for a subclass instance `x`, and
       `format(x) is "abc"` is False too rather than answering the literal
       `x` holds. A NON-EMPTY spec parts company with all of that below. */
    if (!slen) return apy_str(v);
    if (!apy_spec_parse(sptr, slen, &sp))
        return apy_fail2("ValueError", "Invalid format specifier '%s'%s",
                         sptr, "");
    /* A GROUPING CHARACTER ITS TYPE WILL NOT TAKE is refused before anything
       is formatted, which is where CPython refuses it -- so the refusal does
       not depend on what the value turned out to be. */
    {
        const char *no = apy_group_refusal(&sp);
        if (no) return apy_fail("ValueError", no);
    }

    /* AN INSTANCE OF A CLASS EXTENDING str FORMATS AS TEXT. With no type
       character the spec takes its presentation from the VALUE, and asking
       only the kind sent `"{:>3}".format(S("b"))` past every branch to
       `Unknown format code 's' for object of type 'S'` -- about a value
       CPython pads as the string it is. */
    if (sp.type == 's' || (!sp.type && O(apy_text_like(v))->kind == APY_STR_K)) {
        apy_value s = apy_str(v);
        int64_t len;
        if (!s) return 0;
        len = O(s)->v.s.n;
        /* A PRECISION ON TEXT IS A MAXIMUM CHARACTER COUNT, so `"{:.2}"` of
           `"éàb"` is `"éà"`. Truncating bytes cut the `à` in half. */
        if (sp.has_precision && sp.precision < apy_str_chars(s))
            len = apy_char_to_byte(s, sp.precision);
        /* A SPEC THAT CHANGES NOTHING ANSWERS THE RECEIVER ITSELF, and for a
           str SUBCLASS that means the INSTANCE, subclass type and all. This
           is the asymmetry a non-empty spec has against the empty one above
           and it is CPython's rather than an economy taken here: an empty
           spec goes through `PyObject_Format`'s `PyUnicode_CheckExact` fast
           path and then copies, while a non-empty one reaches
           `_PyUnicode_FormatAdvancedWriter`, which has no such test -- it
           pads nothing, truncates nothing and hands the writer the object it
           was given, and an empty writer adopts that object as its result.
           Measured against CPython 3.14, with `x = S("abc")`:
               format(x, "s")   is x -> True, kind S
               format(x, ">3")  is x -> True, kind S   (three wide already)
               format(x, "*>3") is x -> True, kind S   (a fill changes nothing)
               format(x, ".3")  is x -> True, kind S   (nothing truncated)
               format(x, ">5")  is x -> False, kind str (it padded)
               format(x, ".2")  is x -> False, kind str (it truncated)
           NOTHING TRUNCATED IS `len` UNMOVED and nothing padded is a width
           the text already meets, which is exactly what `apy_spec_pad` would
           decide a line later -- the test is up here because only this
           function still knows which object the text came out of.

           THE TEXT MUST BE `v`'s OWN, AND IT IS COMPARED BY CONTENT rather
           than by pointer, which is not fussiness: `apy_str` of an instance
           answers a FRESH COPY of the held string rather than the held
           string itself -- `apy_inst_text_result` makes it one, so that
           `str(x) is "abc"` is False as CPython says -- so a pointer test
           here recognises no subclass at all and silently gives up the whole
           rule.

           A CLASS WHOSE `__str__` ANSWERS OTHER TEXT IS STILL WRONG HERE,
           and the content test is what LIMITS the damage rather than what
           settles the case. `_PyUnicode_FormatAdvancedWriter` never calls
           `__str__` at all -- it formats `self`'s own text -- so with
               class T(str):
                   def __str__(self): return "zzz"
           CPython 3.14 measures
               format(T("abc"), ">3") is T("abc") -> True, kind T
               format(T("abc"), ">5")             -> '  abc'
           while every path here formats what `apy_str` answered and says
               format(t, ">3") -> 'zzz', kind str;  format(t, ">5") -> '  zzz'
           The EMPTY spec really is `str(v)` and really does say `'zzz'`, so
           the two specs part company over more than identity. That is older
           than this rule and is not fixed by it: the body would have to come
           from `apy_text_like(v)` rather than from `apy_str(v)`, which is a
           change to what every path prints and belongs with the interpreter's
           half of it. Left as it is, on the measurement, and written down
           here so the next reader does not take the content test for a
           statement that this case is settled. The kind test carries the other
           half: a bytes reaching this branch through `sp.type == 's'` is
           formatted as the repr `apy_str` made of it and must not be handed
           back.

           AN EMPTY TEXT IS NEVER THE ANSWER, however inert the spec, and
           that is the one place the "pads nothing, truncates nothing" rule
           stops short of the writer. `_PyUnicodeWriter_WriteStr`
           (Objects/unicodeobject.c) opens with `if (len == 0) return 0;` --
           it writes nothing and adopts nothing -- so the writer reaches
           `_PyUnicodeWriter_Finish` with `pos == 0` and that returns
           `unicode_empty`, not the object it was handed. Measured against
           CPython 3.14 with `xe = S("")`:
               format(xe, ">0") is xe -> False, kind str
               format(xe, ">0") is "" -> True
           and the same for `"{:>0}".format(xe)`, `f"{xe:>0}"` and
           `"%0s" % xe`. WITHOUT THIS TEST THE RULE ANSWERS `xe`, which is a
           row these two runtimes had RIGHT before the rule was added: the
           fall-through below ends at `apy_str_copy(body, 0)`, and that is
           the shared empty cell already. */
        {
            apy_value own = apy_text_like(v);
            if (O(own)->kind == APY_STR_K && O(s)->v.s.n > 0
                    && len == O(s)->v.s.n
                    && sp.width <= apy_str_chars(s)
                    && O(own)->v.s.n == len
                    && (own == s || memcmp(APY_CSTR(own), APY_CSTR(s),
                                           (size_t)len) == 0))
                return v;
        }
        return apy_spec_pad(APY_CSTR(s), len, &sp, 0);
    }
    if (sp.type == 'b' || sp.type == 'o' || sp.type == 'x' || sp.type == 'X'
        || sp.type == 'd' || sp.type == 'c'
        /* `n` IS AN INTEGER TYPE ONLY FOR AN INTEGER. For a float it means
           `g`, locale-aware -- and this branch claimed every `n`, so
           `format(1.5, "n")` fell through to the integer rule and was
           refused as an unknown format code for a float. */
        || (sp.type == 'n' && (apy_is_int_like(v) || apy_is_big(v)))) {
        int64_t iv;
        uint64_t mag;
        int base = 10, upper = 0;
        if (!apy_is_int_like(v) && !apy_is_big(v)) return apy_bad_code(sp.type, v);
        if (sp.type == 'b') base = 2;
        else if (sp.type == 'o') base = 8;
        else if (sp.type == 'x') base = 16;
        else if (sp.type == 'X') { base = 16; upper = 1; }
        if (apy_is_big(v)) {
            /* `c` OF A BIG IS NO CHARACTER, and CPython says so in the words
               of the conversion that could not be made. Refused HERE rather
               than through `apy_chr`, which reads the machine word out of a
               value that has none -- for a big that is the limb pointer, and
               it only fails to be observable because a heap address is
               larger than 0x10FFFF. */
            if (sp.type == 'c')
                return apy_fail("OverflowError",
                                "Python int too large to convert to C long");
            {
                /* THE DIGITS IN THE BASE THE SPEC ASKED FOR. This answered
                   the DECIMAL text for every base, with nothing to say the
                   base had been ignored -- `format(2 ** 70, "x")` came back
                   as `1180591620717411303424`, a wrong answer rather than a
                   missing feature, and `%x` inherits it because `%` is
                   translated into this language. `hex()` has converted a big
                   integer all along; this path just never asked it.

                   THE SAME LAYOUT THE MACHINE-WORD PATH BELOW BUILDS: the
                   sign the spec asks for, then the optional prefix, then the
                   digits, then grouping. Only the digits differ, which is
                   why only the digits are shared. */
                int bits_per = base == 2 ? 1 : (base == 8 ? 3 : 4);
                int neg = O(v)->v.big.neg;
                apy_value dec = 0;
                const char *ds = 0;
                int64_t ndig, bn = 0, dn;
                char *big;
                apy_value out;
                if (base == 10) {
                    /* Its DECIMAL digits are what `apy_str` already makes;
                       the sign it writes is dropped here because the spec
                       decides which sign goes on. */
                    dec = apy_str(v);
                    if (!dec) return 0;
                    ds = APY_CSTR(dec) + (neg ? 1 : 0);
                    ndig = O(dec)->v.s.n - (neg ? 1 : 0);
                } else {
                    ndig = apy_big_digit_count(O(v), bits_per);
                }
                /* Room for the digits, a separator between every one of them
                   even though grouping never puts in that many, the sign,
                   the two prefix characters and the terminator. */
                big = (char *)malloc((size_t)(ndig * 2 + 8));
                if (!big) { fputs("uasm: out of memory\n", stderr); exit(1); }
                if (neg) big[bn++] = '-';
                else if (sp.sign == '+') big[bn++] = '+';
                else if (sp.sign == ' ') big[bn++] = ' ';
                if (sp.alt && base != 10) { big[bn++] = '0'; big[bn++] = sp.type; }
                if (base == 10) { memcpy(big + bn, ds, (size_t)ndig); dn = ndig; }
                else dn = apy_big_digits(O(v), bits_per, big + bn, upper);
                if (sp.group) dn = apy_group_digits(big + bn, dn, sp.group, apy_group_every(sp.type));
                bn += dn;
                big[bn] = 0;
                out = apy_spec_pad(big, bn, &sp, 1);
                free(big);
                return out;
            }
        }
        iv = O(v)->v.i;
        if (sp.type == 'c') {
            /* THE CHARACTER, encoded -- `format(255, 'c')` is `chr(255)`, and
               a str is stored as UTF-8, so that is two bytes and not one.
               Writing the low byte raw produced a string that compared
               unequal to `chr(255)` and was not valid UTF-8 either. */
            apy_value ch;
            /* OUT OF RANGE IS AN OverflowError HERE, whatever `chr` calls
               it: CPython words this one from `%c` -- the conversion that
               could not be made -- rather than from the builtin, and the
               TYPE is what a program catching it sees. */
            if (iv < 0 || iv > 0x10FFFF)
                return apy_fail("OverflowError",
                                "%c arg not in range(0x110000)");
            ch = apy_chr(v);
            if (!ch) return 0;
            return apy_spec_pad(APY_CSTR(ch), O(ch)->v.s.n, &sp, 0);
        }
        mag = iv < 0 ? (uint64_t)(-(iv + 1)) + 1u : (uint64_t)iv;
        if (iv < 0) body[n++] = '-';
        else if (sp.sign == '+') body[n++] = '+';
        else if (sp.sign == ' ') body[n++] = ' ';
        if (sp.alt && base != 10) {
            body[n++] = '0';
            body[n++] = sp.type;
        }
        {
            int64_t d = apy_int_digits(body + n, mag, base, upper);
            if (sp.group) d = apy_group_digits(body + n, d, sp.group, apy_group_every(sp.type));
            n += d;
        }
        body[n] = 0;
        return apy_spec_pad(body, n, &sp, 1);
    }
    {
        /* The float types. `printf` is the right decimal conversion -- the
           same one `repr` uses -- so only the sign, grouping and padding are
           added around it. */
        double d;
        char tmp[400];
        int prec = sp.has_precision ? sp.precision : 6;
        char type = sp.type;
        const char *src;
        int64_t len;
        if (!apy_is_num(v)) return apy_bad_code(type, v);
        d = apy_as_float(v);
        if (type == '%') { d *= 100.0; type = 'f'; }
        /* `n` IS `g` WITH THE LOCALE'S SEPARATORS, and the C locale is this
           runtime's -- so for a float the two are the same conversion, which
           is the whole of what `n` means here. */
        if (type == 'n') type = 'g';
        /* PEP 682: `z` turns a negative zero into a positive one -- and it is
           about the ROUNDED value, so `format(-0.001, 'z.1f')` is `0.0` too.
           Applied after the scaling above and before the conversion below,
           which is the only point at which both are true. */
        if (sp.coerce_zero) {
            double scale = 1.0;
            int k;
            for (k = 0; k < prec && k < 17; k++) scale *= 10.0;
            /* `signbit`, not `d < 0.0`: NEGATIVE ZERO is not less than
               zero, and it is the value the flag exists for. */
            if (signbit(d) && fabs(d) * scale < 0.5) d = 0.0;
        }
        if (!type && sp.has_precision) {
            /* A PRECISION WITH NO TYPE is `g`: `format(3.14159, '.3')` is
               '3.14', three SIGNIFICANT digits, not three decimal places and
               not the unrounded number. Without this the precision was
               dropped and the whole value printed. */
            snprintf(tmp, sizeof tmp, "%.*g", prec ? prec : 1, d);
        } else if (!type) {
            /* No type and no precision: `str(v)`, the shortest round-tripping
               form, and NOT `%g` -- `f"{0.1:>8}"` must still say `0.1`. */
            apy_value s = apy_str(v);
            if (!s) return 0;
            if (O(s)->v.s.n >= (int64_t)sizeof tmp)
                return apy_spec_pad(APY_CSTR(s), O(s)->v.s.n, &sp, 1);
            memcpy(tmp, APY_CSTR(s), (size_t)O(s)->v.s.n);
            tmp[O(s)->v.s.n] = 0;
        } else if (type == 'f' || type == 'F') {
            snprintf(tmp, sizeof tmp, "%.*f", prec, d);
        } else if (type == 'e' || type == 'E') {
            snprintf(tmp, sizeof tmp, type == 'e' ? "%.*e" : "%.*E", prec, d);
        } else if (type == 'g' || type == 'G') {
            snprintf(tmp, sizeof tmp, type == 'g' ? "%.*g" : "%.*G",
                     prec ? prec : 1, d);
        } else {
            return apy_bad_code(type, v);
        }
        src = tmp;
        len = (int64_t)strlen(tmp);
        n = 0;
        if (src[0] == '-') { body[n++] = '-'; src++; len--; }
        else if (sp.sign == '+') body[n++] = '+';
        else if (sp.sign == ' ') body[n++] = ' ';
        memcpy(body + n, src, (size_t)len);
        if (sp.group) {
            /* Group the INTEGER part only: the separator belongs to the left
               of the point, and grouping the fraction would produce a number
               that does not read back. */
            int64_t head = 0, grouped;
            char tail[400];
            while (head < len && body[n + head] != '.' && body[n + head] != 'e'
                   && body[n + head] != 'E') head++;
            memcpy(tail, body + n + head, (size_t)(len - head));
            /* A FLOAT ALWAYS GROUPS BY THREE: the four-digit rule is the
               non-decimal bases', and a float has no base to be in. */
            grouped = apy_group_digits(body + n, head, sp.group, 3);
            memcpy(body + n + grouped, tail, (size_t)(len - head));
            len = grouped + (len - head);
        }
        n += len;
        if (sp.type == '%') body[n++] = '%';
        body[n] = 0;
        return apy_spec_pad(body, n, &sp, 1);
    }
}

/* `"{} {:>5} {name!r}".format(...)` -- the OTHER half of PEP 3101: the
   replacement-field syntax around the spec `apy_format` reads.

   Auto-numbering and explicit numbering cannot be mixed, and CPython says so
   rather than guessing; that check is what `auto` below is for. A nested spec
   -- `{:{width}}` -- is one level deep, which is all CPython allows too. */
/* The AUTO-NUMBERING COUNTER IS SHARED with any nested spec: in
   `"{:>{}}".format('q', 5)` the field takes `'q'` and the `{}` inside the spec
   takes `5`. A recursive call with its own counter took `'q'` twice and then
   reported it as a bad format code -- so the state travels by pointer. */
static apy_value apy_format_at(apy_value fmt, apy_value args, apy_value kw,
                               int64_t *auto_at, int *auto_used,
                               int *explicit_used) {
    const char *p;
    int64_t n, i = 0, out_cap, out_n = 0;
    char *out;
    /* A str SUBCLASS IS THE FORMAT STRING TOO: `S("{}").format(1)` works in
       CPython, and this reaches the instance itself because `format` is
       compiled to a direct call rather than through `apy_method_self`. */
    fmt = apy_text_like(fmt);
    if (O(fmt)->kind != APY_STR_K)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'format'%s",
                         apy_kind_name(fmt), "");
    p = APY_CSTR(fmt);
    n = O(fmt)->v.s.n;
    out_cap = n + 64;
    out = (char *)malloc((size_t)out_cap + 1);
    if (!out) { fputs("uasm: out of memory\n", stderr); exit(1); }

    while (i < n) {
        if (p[i] == '{' && i + 1 < n && p[i + 1] == '{') {
            out[out_n++] = '{'; i += 2; continue;
        }
        if (p[i] == '}' && i + 1 < n && p[i + 1] == '}') {
            out[out_n++] = '}'; i += 2; continue;
        }
        if (p[i] != '{') {
            if (out_n + 1 >= out_cap) {
                out_cap *= 2;
                out = (char *)realloc(out, (size_t)out_cap + 1);
            }
            out[out_n++] = p[i++];
            continue;
        }
        {
            /* One replacement field: `{field!conv:spec}`. */
            int64_t field_at = i;
            int64_t start = ++i, colon = -1, bang = -1, depth = 0;
            char field[128], conv = 0;
            apy_value value, spec, shown, given;
            while (i < n && (p[i] != '}' || depth)) {
                if (p[i] == '{') depth++;
                else if (p[i] == '}') depth--;
                else if (p[i] == ':' && colon < 0 && !depth) colon = i;
                else if (p[i] == '!' && bang < 0 && colon < 0
                         && i + 1 < n && p[i + 1] != '=') bang = i;
                i++;
            }
            if (i >= n) {
                free(out);
                return apy_fail("ValueError",
                                "Single '{' encountered in format string");
            }
            {
                /* THE NAME STOPS AT WHICHEVER COMES FIRST, and a `!` always
                   comes before a `:` -- the scan above only records a bang
                   while `colon < 0`, so the two can never be out of order.
                   Asking the colon first meant a field wearing BOTH read its
                   name as everything up to the colon, conversion included:
                   `"{!s:>3}".format("abc")` looked up an argument called
                   `!s` and died as `KeyError: '!s'` on both compiled paths,
                   where CPython and the interpreter both answer `abc`.
                   Neither `{!s}` nor `{:>3}` alone shows it, which is why it
                   survived every probe until one wrote the pair. */
                int64_t fend = bang >= 0 ? bang : (colon >= 0 ? colon : i);
                int64_t flen = fend - start;
                int64_t blen;
                if (flen >= (int64_t)sizeof field) flen = sizeof field - 1;
                memcpy(field, p + start, (size_t)flen);
                field[flen] = 0;
                /* `{x[0]}` and `{a.real}`: the NAME stops at the first `.` or
                   `[`, and what follows is a chain of accessors applied to
                   whatever the name resolved to. Treating the whole thing as
                   one keyword looked for an argument called `x[0]`. */
                for (blen = 0; blen < flen; blen++)
                    if (field[blen] == '.' || field[blen] == '[') break;
                if (!flen) {
                    if (*explicit_used) {
                        free(out);
                        return apy_fail("ValueError",
                                        "cannot switch from manual field "
                                        "specification to automatic field "
                                        "numbering");
                    }
                    *auto_used = 1;
                    value = *auto_at < O(args)->v.q.n
                        ? O(args)->v.q.items[(*auto_at)++] : 0;
                    if (!value) {
                        /* WHICH INDEX, which is the part a reader needs:
                           `"{} {}".format(1)` names the 1 that is missing
                           and not merely that one is. */
                        char b[80];
                        snprintf(b, sizeof b, "Replacement index %lld out of "
                                 "range for positional args tuple",
                                 (long long)*auto_at);
                        free(out);
                        return apy_fail("IndexError", b);
                    }
                } else if (field[0] >= '0' && field[0] <= '9') {
                    int64_t at = 0, k;
                    if (*auto_used) {
                        free(out);
                        return apy_fail("ValueError",
                                        "cannot switch from automatic field "
                                        "numbering to manual field "
                                        "specification");
                    }
                    *explicit_used = 1;
                    for (k = 0; field[k] >= '0' && field[k] <= '9'; k++)
                        at = at * 10 + (field[k] - '0');
                    if (at >= O(args)->v.q.n) {
                        char b[80];
                        snprintf(b, sizeof b, "Replacement index %lld out of "
                                 "range for positional args tuple",
                                 (long long)at);
                        free(out);
                        return apy_fail("IndexError", b);
                    }
                    value = O(args)->v.q.items[at];
                } else {
                    char base[64];
                    apy_value key;
                    int64_t at;
                    int64_t n2 = blen < (int64_t)sizeof base - 1
                        ? blen : (int64_t)sizeof base - 1;
                    memcpy(base, field, (size_t)n2);
                    base[n2] = 0;
                    key = apy_lit(base);
                    /* CPYTHON SUBSCRIPTS RATHER THAN CHECKING, so a
                       non-mapping is a complaint only once a field ASKS for
                       a key: `"".format_map(None)` is `''` and
                       `"{}".format_map(None)` is an IndexError about the
                       positional args. These are the subscript's own two
                       messages, which is why they name no method. */
                    if (O(kw)->kind != APY_DICT_K) {
                        free(out);
                        return O(kw)->kind == APY_LIST_K
                                   || O(kw)->kind == APY_TUPLE_K
                               ? apy_fail2("TypeError",
                                           "%s indices must be integers or "
                                           "slices, not str%s",
                                           apy_kind_name(kw), "")
                               : apy_fail2("TypeError",
                                           "'%s' object is not "
                                           "subscriptable%s",
                                           apy_kind_name(kw), "");
                    }
                    at = apy_dict_find(kw, key);
                    if (at < 0) {
                        free(out);
                        return apy_fail2("KeyError", "'%s'%s", base, "");
                    }
                    value = O(kw)->v.d.vals[at];
                }
                /* THE ACCESSORS, left to right. `{a.b[0].c}` is ordinary
                   Python written inside a format field, and each step is the
                   operation it looks like. */
                {
                    int64_t k = blen;
                    while (k < flen && value) {
                        char part[64];
                        int64_t j = 0;
                        if (field[k] == '.') {
                            k++;
                            while (k < flen && field[k] != '.'
                                   && field[k] != '['
                                   && j < (int64_t)sizeof part - 1)
                                part[j++] = field[k++];
                            part[j] = 0;
                            value = apy_getattr(value, apy_lit(part));
                        } else if (field[k] == '[') {
                            int all_digits = 1;
                            k++;
                            while (k < flen && field[k] != ']'
                                   && j < (int64_t)sizeof part - 1) {
                                if (field[k] < '0' || field[k] > '9')
                                    all_digits = 0;
                                part[j++] = field[k++];
                            }
                            part[j] = 0;
                            if (k < flen && field[k] == ']') k++;
                            /* AN ALL-DIGIT KEY IS AN INDEX, as CPython reads
                               it -- `{x[0]}` indexes a list, and a mapping
                               with a numeric string key needs the quotes a
                               format field cannot carry. */
                            value = apy_getitem(
                                value, j && all_digits
                                    ? apy_from_int(strtoll(part, 0, 10))
                                    : apy_lit(part));
                        } else {
                            break;
                        }
                    }
                    if (!value) { free(out); return 0; }
                }
            }
            /* THE OBJECT THE FIELD NAMED, before any conversion runs over
               it: what the identity rule below is allowed to hand back. */
            given = value;
            if (bang >= 0) conv = p[bang + 1];
            if (colon >= 0) {
                /* A NESTED spec -- `{:{width}}` -- is itself formatted first,
                   with the same arguments. One level, which is CPython's
                   limit too. */
                apy_value inner = apy_str_copy(p + colon + 1,
                                               i - colon - 1);
                spec = memchr(p + colon + 1, '{', (size_t)(i - colon - 1))
                    ? apy_format_at(inner, args, kw, auto_at, auto_used,
                                    explicit_used)
                    : inner;
                if (!spec) { free(out); return 0; }
            } else {
                spec = apy_lit("");
            }
            /* `!a` WAS FOLDED INTO `!r` UNTIL THE ESCAPING EXISTED, and
               the two agree for every ASCII value -- so `"{!a}".format(x)`
               answered `'cafÃ©'`'s own bytes where CPython writes
               `'café'`, and nothing noticed until a case used a
               non-ASCII one. */
            if (conv == 'a') value = apy_ascii(value);
            else if (conv == 'r') value = apy_repr(value);
            else if (conv == 's') value = apy_str(value);
            if (!value) { free(out); return 0; }
            shown = apy_format(value, spec);
            if (!shown) { free(out); return 0; }
            /* ONE WHOLE FIELD AND NOTHING ELSE IS `format()` ITSELF, so the
               object it answered is the answer: `"{}".format(s)` IS `s`,
               `"{:>3}".format(x)` IS the subclass instance `x`, and
               `"{0.a}".format(obj)` is `obj.a`. CPython's writer is what
               makes that so -- a field is written with
               `_PyUnicodeWriter_WriteStr`, which for the FIRST thing written
               into a writer with no buffer yet adopts the object as the
               buffer and hands it back at the end. Joining anything to it,
               even one literal character, allocates and the result is a new
               string by construction, which is why this asks that the field
               begin at 0 and its `}` end the string.

               `shown == given` AND NOT MERELY "this was the only field":
               what the writer adopts is the object it was handed, so only an
               object that came THROUGH `apy_format` unchanged may be handed
               on. It rules out `{!r}` and `{!s}` of a subclass, where the
               conversion built a second string, and it rules out
               `"{}".format(5)`, whose text this runtime shares a cell for
               where CPython builds a fresh `"5"` every time. */
            if (field_at == 0 && i + 1 == n && shown == given) {
                free(out);
                return shown;
            }
            /* THE TEXT, NOT THE INSTANCE. `apy_format` answers the receiver
               itself for a str subclass under a spec that changes nothing,
               and that object's payload is an instance's and not a string's
               -- so reading it as one here would copy an object header into
               the result. */
            shown = apy_text_like(shown);
            while (out_n + O(shown)->v.s.n >= out_cap) {
                out_cap = out_cap * 2 + O(shown)->v.s.n;
                out = (char *)realloc(out, (size_t)out_cap + 1);
            }
            memcpy(out + out_n, APY_CSTR(shown), (size_t)O(shown)->v.s.n);
            out_n += O(shown)->v.s.n;
            i++;                        /* past the '}' */
        }
    }
    out[out_n] = 0;
    return apy_str_take(out, out_n);
}

/* `"%d %s" % (1, "a")` -- printf-style formatting.
   TRANSLATED INTO THE MINI-LANGUAGE, not reimplemented. `%05.2f` and
   `{:05.2f}` mean the same thing down to the zero padding and the rounding,
   so the padding, the precision and every presentation type are read from
   `apy_format` rather than written a second time here. What this function
   owns is the printf SPELLING: which flags mean what, where the arguments
   come from, and the two conversions (`%r`, `%s`) the mini-language has no
   type character for.

   WHAT IT DOES NOT DO: the mapping form, `"%(name)s" % {...}`. A dict on the
   right is currently one argument like any other, which is right for `%s` and
   wrong for the mapping form -- so that spelling is refused below rather than
   quietly formatting the dict. */
/* A USER OBJECT REACHES A NUMERIC `%` CONVERSION THROUGH ITS NUMBER and not
   through `__format__`: CPython's `%d` asks `__index__` and `%f` asks
   `__float__`. Handing the object straight to `apy_format` reached the
   mini-language, which reported an unknown FORMAT CODE for a class that
   defines exactly the method for it -- `"%d" % HasIndex()` is `'42'` there
   and was a ValueError here.

   THE OBJECT UNCHANGED when its class offers neither, so the refusal is
   `apy_percent_needs`'s below and a class is refused in the same words a
   builtin of the wrong kind is. Answers 0 with the error set for a dunder
   that raised. `objects/host.py`'s `_percent` is the same three lines. */
static apy_value apy_percent_number(char conv, apy_value value) {
    apy_value got;
    if (O(value)->kind != APY_INST_K) return value;
    if (conv == 'e' || conv == 'E' || conv == 'f' || conv == 'F'
            || conv == 'g' || conv == 'G') {
        got = apy_unary_dunder(value, "__float__");
        if (apy_error_occurred()) return 0;
        return got ? got : value;
    }
    got = apy_unary_dunder(value, "__index__");
    if (apy_error_occurred()) return 0;
    if (!got) {
        got = apy_unary_dunder(value, "__int__");
        if (apy_error_occurred()) return 0;
    }
    return got ? got : value;
}

/* CPython's refusal for an argument a `%` conversion cannot take: sets the
   error and answers 0, or answers 1 for one it can.

   THE WRONG EXCEPTION TYPE IS WHY THIS EXISTS. `%` is implemented by
   translating into the format MINI-LANGUAGE and handing the argument to
   `apy_format` -- which is what keeps `%05.2f` and `{:05.2f}` from being
   written twice -- and what that complains about is an unknown FORMAT CODE,
   a ValueError. What `%` complains about is the ARGUMENT, a TypeError. So
   `"%d" % "a"` raised `Unknown format code 'd' for object of type 'str'`
   where CPython raises `%d format: a real number is required, not str`, and
   a program catching TypeError around a `%` missed it entirely.

   THREE WORDINGS, and they are CPython's own rather than one generalised:
   `%d` takes any real number; `%x` takes an INTEGER and refuses a float that
   `%d` accepts; and the floating conversions do not name the conversion at
   all. */
static int apy_percent_needs(char conv, apy_value value) {
    char buf[256];
    if (conv == 'd' || conv == 'i' || conv == 'u') {
        if (apy_is_num(value)) return 1;
        snprintf(buf, sizeof buf,
                 "%%%c format: a real number is required, not %s",
                 conv, apy_kind_name(value));
    } else if (conv == 'x' || conv == 'X' || conv == 'o') {
        if (apy_is_int_like(value)) return 1;
        snprintf(buf, sizeof buf,
                 "%%%c format: an integer is required, not %s",
                 conv, apy_kind_name(value));
    } else if (conv == 'e' || conv == 'E' || conv == 'f' || conv == 'F'
               || conv == 'g' || conv == 'G') {
        if (apy_is_num(value)) return 1;
        snprintf(buf, sizeof buf, "must be real number, not %s",
                 apy_kind_name(value));
    } else {
        return 1;
    }
    apy_fail("TypeError", buf);
    return 0;
}

/* A run of decimal digits as a number, saturating rather than wrapping: the
   width or the precision of a `%` spec, read out of the format itself. */
static int64_t apy_percent_digit_run(const char *p, int64_t n) {
    int64_t k, got = 0;
    for (k = 0; k < n; k++) {
        if (got > (INT64_MAX - 9) / 10) return INT64_MAX;
        got = got * 10 + (p[k] - '0');
    }
    return got;
}

/* `%.5d` -- a precision on an INTEGER conversion, which is a minimum number
   of DIGITS.

   THE MINI-LANGUAGE HAS NO SUCH THING, and this runtime's own copy of it
   quietly ignored the precision it was handed, so `"%.5d" % 42` printed `42`
   where CPython prints `00042` -- a wrong answer that does not announce
   itself, while the interpreter refused the same line outright. printf's
   precision is a zero fill applied to the digits alone, AFTER the sign and
   AFTER any `0x`, so it is spelled here as the one thing the mini-language
   does have for that -- a `0`-filled width counting the sign and the prefix
   -- and the field width is laid over the result afterwards.

   THE `0` FLAG STILL APPLIES, which is where Python parts from C's printf:
   `"%08.5d" % 42` is `'00000042'`, so a zero-filled width simply widens the
   digit fill to the whole field. The interpreter's twin is
   `_percent_digits`. */
static apy_value apy_percent_digits(apy_value value, char conv, int plus,
                                    int space, int hash, int zero, int minus,
                                    int64_t width, int64_t prec) {
    char inner[48], outer[32];
    int64_t digits, n = 0;
    apy_value body;
    char kind = (conv == 'i' || conv == 'u') ? 'd' : conv;
    /* `%#d` IS ACCEPTED AND MEANS NOTHING, so the prefix is only counted
       where there is one to count. */
    int alt = hash && kind != 'd';
    int neg = O(value)->kind == APY_BIG_K ? O(value)->v.big.neg
            : (O(value)->kind == APY_INT_K && O(value)->v.i < 0);
    digits = prec + ((neg || plus || space) ? 1 : 0) + (alt ? 2 : 0);
    if (zero && !minus && width > digits) digits = width;
    if (plus) inner[n++] = '+';
    else if (space) inner[n++] = ' ';
    if (alt) inner[n++] = '#';
    if (digits > 0)
        n += snprintf(inner + n, sizeof inner - (size_t)n, "0%lld",
                      (long long)digits);
    inner[n++] = kind;
    body = apy_format(value, apy_str_copy(inner, n));
    if (!body || width <= 0) return body;
    n = snprintf(outer, sizeof outer, "%c%lld", minus ? '<' : '>',
                 (long long)width);
    return apy_format(body, apy_str_copy(outer, n));
}

/* The argument behind a `%*d` width or a `%.*f` precision: 1, with the number
   through `slot` and the cursor advanced, or 0 with the error set.

   IT IS FETCHED LIKE ANY OTHER POSITIONAL ARGUMENT, and that is what settles
   the mapping form without a case of its own: `"%(k)*s" % {...}` has no
   positional arguments to draw on, so the star is handed the MAPPING itself
   -- which is not an int, and the refusal below is exactly what CPython
   reports for it.

   REFUSED BY `PyLong_Check` AND NOTHING ELSE. A float is turned away here
   where `%d` of one truncates, and a class carrying `__index__` is turned
   away where `%d` of one is asked for it: the star reads a C integer out of
   the argument rather than converting anything. An int SUBCLASS passes,
   because `PyLong_Check` is what asks.

   TWO OVERFLOW WORDINGS, and they are not interchangeable: CPython reads the
   width through `PyLong_AsSsize_t` and the precision through `_PyLong_AsInt`,
   so one names `ssize_t` and the other names `int` -- and the precision is
   refused at a threshold four billion times lower. */
static int apy_percent_star(apy_value right, int many, int64_t supplied,
                            int64_t *at, int64_t *slot, int sized) {
    apy_value v;
    if (*at >= supplied) {
        apy_fail("TypeError", "not enough arguments for format string");
        return 0;
    }
    v = many ? O(right)->v.q.items[*at] : right;
    (*at)++;
    if (O(v)->kind == APY_INST_K && O(v)->v.o.held
            && apy_is_int_like(O(v)->v.o.held))
        v = O(v)->v.o.held;
    if (!apy_is_int_like(v)) {
        apy_fail("TypeError", "* wants int");
        return 0;
    }
    /* A BIG IS READ BEFORE `v.i` IS, because `v.i` on one is the limb
       POINTER: the overflow this reports is the reason the value never gets
       that far. */
    if (apy_is_big(v)
            || (!sized && (O(v)->v.i > 2147483647LL
                           || O(v)->v.i < -2147483647LL - 1))) {
        apy_fail("OverflowError",
                 sized ? "Python int too large to convert to C ssize_t"
                       : "Python int too large to convert to C int");
        return 0;
    }
    *slot = O(v)->v.i;
    return 1;
}

static apy_value apy_str_percent(apy_value fmt, apy_value right) {
    const char *p = APY_CSTR(fmt);
    int64_t n = O(fmt)->v.s.n, i = 0, out_cap = n + 64, out_n = 0, at = 0;
    int64_t supplied;
    char *out;
    /* A TUPLE ON THE RIGHT IS THE ARGUMENT LIST; anything else is one
       argument. That is the whole of the rule, and it is why `"%s" % (1, 2)`
       is an error while `"%s" % [1, 2]` prints the list. */
    int many;
    /* A MAPPING ON THE RIGHT supplies NAMED fields only -- `"%(x)s" % {...}`
       -- and nothing is consumed positionally, so an unused entry is not an
       error. `"ab" % {"ab": 1}` is just `"ab"`. */
    int mapping;
    /* THE ONE ARGUMENT THIS WHOLE FORMAT IS, or 0 -- see where it is set. */
    apy_value lone = 0;
    /* THE SINGLE ARGUMENT a non-tuple operand is, which a `%(key)` lookup
       REPLACES with the value it found -- CPython's model exactly. Every
       conversion and every `*` then takes it through the same `at`, which
       is why `"%(a)s %s" % d` has nothing left for its second conversion,
       why `"%s %(a)s" % d` hands the first one the whole mapping, and why
       `"%(a)*s"` reads its width out of the VALUE. The found value used to
       ride beside the arguments instead, so a bare `%s` after a key still
       saw the mapping and printed it. */
    apy_value cur;
    /* A TUPLE SUBCLASS IS THE ARGUMENT LIST, because CPython asks
       `PyTuple_Check` and a subclass passes it: `"%d-%d" % point` for a
       namedtuple formats its fields, where this took the whole point for
       one argument. */
    if (O(right)->kind == APY_INST_K && O(right)->v.o.held
            && O(O(right)->v.o.held)->kind == APY_TUPLE_K)
        right = O(right)->v.o.held;
    many = O(right)->kind == APY_TUPLE_K;
    mapping = O(right)->kind == APY_DICT_K;
    cur = right;
    supplied = many ? O(right)->v.q.n : 1;

    out = (char *)malloc((size_t)out_cap + 1);
    if (!out) { fputs("uasm: out of memory\n", stderr); exit(1); }

    while (i < n) {
        char spec[64];
        int64_t sn = 0, conv_at = i;
        apy_value value, shown, given;
        char conv;
        int minus = 0, zero = 0;

        if (p[i] != '%') {
            if (out_n + 1 >= out_cap) {
                out_cap = out_cap * 2 + 8;
                out = (char *)realloc(out, (size_t)out_cap + 1);
            }
            out[out_n++] = p[i++];
            continue;
        }
        i++;
        if (i < n && p[i] == '%') {      /* `%%` is a literal percent */
            if (out_n + 1 >= out_cap) {
                out_cap = out_cap * 2 + 8;
                out = (char *)realloc(out, (size_t)out_cap + 1);
            }
            out[out_n++] = '%'; i++; continue;
        }
        if (i < n && p[i] == '(') {
            /* `%(name)s` -- the MAPPING FORM. The key runs to the matching
               `)`; what follows is an ordinary spec. */
            char key[64];
            int64_t j = 0;
            apy_value found;
            if (!mapping) {
                free(out);
                return apy_fail("TypeError", "format requires a mapping");
            }
            i++;
            while (i < n && p[i] != ')' && j < (int64_t)sizeof key - 1)
                key[j++] = p[i++];
            key[j] = 0;
            if (i < n && p[i] == ')') i++;
            found = apy_dict_get_or(right, apy_lit(key), 0);
            if (!found) {
                free(out);
                return apy_fail2("KeyError", "'%s'%s", key, "");
            }
            cur = found;
            at = 0;
        }
        /* THE FLAGS ARE COLLECTED, NOT EMITTED, because two of them depend
           on the conversion that has not been read yet -- and because the
           mini-language fixes an order (align, sign, `#`, `0`, width) that
           printf does not. Emitting each flag where it was read produced
           `+<` for `%-+d`, which is not a spec at all. */
        int plus = 0, space = 0, hash = 0, is_text;
        int64_t wid_at, wid_n = 0, prec_at, prec_n = 0;
        /* THE STAR FORMS ARE READ, NOT COPIED, which is why they need slots
           of their own: the digits below are copied into the spec verbatim,
           and the mini-language has no `*` for them to land in. */
        int64_t star_w = 0, star_p = 0;
        int starred_w = 0, starred_p = 0;
        while (i < n && (p[i] == '-' || p[i] == '+' || p[i] == ' '
                         || p[i] == '0' || p[i] == '#')) {
            if (p[i] == '-') minus = 1;
            else if (p[i] == '0') zero = 1;
            else if (p[i] == '+') plus = 1;
            else if (p[i] == ' ') space = 1;
            else hash = 1;
            i++;
        }
        wid_at = i;
        if (i < n && p[i] == '*') {
            i++;
            if (!apy_percent_star(many ? right : cur, many, supplied, &at,
                                  &star_w, 1)) {
                free(out); return 0;
            }
            /* A NEGATIVE WIDTH IS THE `-` FLAG. printf's rule, and the flag
               is the only spelling the mini-language has for it, so
               `"%*s" % (-8, "hi")` left-aligns in a field of eight. */
            if (star_w < 0) { minus = 1; star_w = -star_w; }
            starred_w = 1;
        } else {
            while (i < n && p[i] >= '0' && p[i] <= '9') { i++; wid_n++; }
        }
        prec_at = i;
        if (i < n && p[i] == '.') {
            i++;
            if (i < n && p[i] == '*') {
                i++;
                if (!apy_percent_star(many ? right : cur, many, supplied,
                                      &at, &star_p, 0)) {
                    free(out); return 0;
                }
                /* A NEGATIVE PRECISION IS ZERO -- not an error, and not the
                   alignment flag its width twin turns into. */
                if (star_p < 0) star_p = 0;
                starred_p = 1;
            } else {
                prec_n++;
                while (i < n && p[i] >= '0' && p[i] <= '9') { i++; prec_n++; }
            }
        }
        if (i >= n) {
            free(out);
            return apy_fail("ValueError", "incomplete format");
        }
        conv = p[i++];
        is_text = conv == 's' || conv == 'r' || conv == 'a' || conv == 'c'
            || conv == 'b';
        /* PRINTF RIGHT-ALIGNS A STRING; the mini-language left-aligns one.
           The only difference between the two languages that is not a
           spelling, and `"%5s" % "ab"` is where it shows. */
        if (minus) spec[sn++] = '<';
        else if (is_text) spec[sn++] = '>';
        if (plus) spec[sn++] = '+';
        else if (space) spec[sn++] = ' ';
        if (hash) spec[sn++] = '#';
        /* A zero fill on TEXT is not a thing printf does either. */
        if (zero && !minus && !is_text) spec[sn++] = '0';
        { int64_t k;
          if (starred_w)
              sn += snprintf(spec + sn, sizeof spec - (size_t)sn,
                             "%lld", (long long)star_w);
          else
              for (k = 0; k < wid_n; k++) spec[sn++] = p[wid_at + k];
          if (starred_p)
              sn += snprintf(spec + sn, sizeof spec - (size_t)sn,
                             ".%lld", (long long)star_p);
          else
              for (k = 0; k < prec_n; k++) spec[sn++] = p[prec_at + k]; }
        if (at >= supplied) {
            free(out);
            return apy_fail("TypeError",
                            "not enough arguments for format string");
        }
        value = many ? O(right)->v.q.items[at] : cur;
        at++;
        /* THE ARGUMENT AS THE PROGRAM HANDED IT OVER, before `%s` turns it
           into text: what the identity rule at the end of this iteration is
           allowed to answer. */
        given = value;

        /* `%s` and `%r` have no mini-language type character: the value
           becomes text FIRST and the spec then pads that text. */
        if (conv == 's' || conv == 'b') {
            /* A bytes SUBCLASS IS A bytes HERE. `b"%s"` inserts what the
               argument HOLDS, and CPython's `format_obj` asks the buffer
               protocol -- which `class B(bytes)` answers like any other
               bytes. Without the unwrap the instance fell through to
               `apy_str` below and `b"%s" % B(b"ab")` printed the REPR,
               `b"b'ab'"`, which is a wrong answer and not a refusal. */
            if (O(fmt)->kind == APY_BYTES_K
                && O(value)->kind == APY_INST_K && O(value)->v.o.held
                && O(O(value)->v.o.held)->kind == APY_BYTES_K)
                value = O(value)->v.o.held;
            if (O(fmt)->kind == APY_BYTES_K
                && O(value)->kind == APY_BYTES_K) {
                /* `b"%s" % b"ab"` inserts THE BYTES, not their repr -- and
                   `%b` is PEP 461's spelling of the same thing. Re-tagged as
                   a str so the padding below stays one implementation: the
                   two kinds share a layout, and the result is stamped back
                   to bytes at the end. */
                value = apy_from_bytes(
                    (apy_value)(uintptr_t)O(value)->v.s.p, O(value)->v.s.n);
            } else {
                value = apy_str(value);
            }
        }
        else if (conv == 'r') { value = apy_repr(value); }
        else if (conv == 'a') { value = apy_ascii(value); }
        else if (conv == 'c') {
            /* ONE CHARACTER OR AN INT, AND NOTHING ELSE. `apy_str` on
               anything at all meant `"%c" % "ab"` answered `'ab'` and
               `"%c" % obj` answered its repr -- wrong answers rather than
               errors, which is the failure mode that does not announce
               itself. A string of the wrong length is refused BY ITS
               LENGTH, which is how CPython words it. */
            /* A str SUBCLASS IS A str HERE -- unlike `%s`, where a
               subclass is COPIED rather than handed back. `formatchar`
               asks `PyUnicode_Check`, which a subclass passes, so
               `"%c" % S("a")` writes its character like any other
               one-character string; this refused it by its class name. */
            if (O(value)->kind == APY_INST_K && O(value)->v.o.held
                    && O(O(value)->v.o.held)->kind == APY_STR_K)
                value = O(value)->v.o.held;
            if (apy_is_int_like(value)) {
                value = apy_chr(value);
            } else if (!(O(value)->kind == APY_STR_K
                         && apy_str_chars(value) == 1)) {
                char cbuf[256];
                if (O(value)->kind == APY_STR_K)
                    snprintf(cbuf, sizeof cbuf,
                             "%%c requires an int or a unicode character, "
                             "not a string of length %lld",
                             (long long)apy_str_chars(value));
                else
                    snprintf(cbuf, sizeof cbuf,
                             "%%c requires an int or a unicode character, "
                             "not %s", apy_kind_name(value));
                free(out);
                return apy_fail("TypeError", cbuf);
            }
        } else {
            /* A CLASS REACHES A NUMERIC CONVERSION THROUGH ITS NUMBER, and
               then anything still of the wrong kind is refused in CPython's
               words rather than the mini-language's. */
            value = apy_percent_number(conv, value);
            if (!value) { free(out); return 0; }
            if (!apy_percent_needs(conv, value)) { free(out); return 0; }
            /* `%d` OF A FLOAT TRUNCATES, which is why it takes a real number
               where `%x` takes an integer. The mini-language's `d` refuses a
               float outright, so the truncation happens before the handoff:
               `"%d" % 1.5` is `'1'`, not an unknown format code. */
            if ((conv == 'd' || conv == 'i' || conv == 'u')
                    && O(value)->kind == APY_FLOAT_K) {
                value = apy_to_int(value);
                if (!value) { free(out); return 0; }
            }
            /* `%i` and `%u` are both spelled `d` in the mini-language, and
               `%u` has meant `%d` since Python 2. */
            spec[sn++] = (conv == 'i' || conv == 'u') ? 'd' : conv;
        }
        if (!value) { free(out); return 0; }
        spec[sn] = 0;
        if ((prec_n || starred_p)
                && (conv == 'd' || conv == 'i' || conv == 'u' || conv == 'x'
                    || conv == 'X' || conv == 'o'))
            shown = apy_percent_digits(
                value, conv, plus, space, hash, zero, minus,
                starred_w ? star_w
                          : apy_percent_digit_run(p + wid_at, wid_n),
                starred_p ? star_p
                          : apy_percent_digit_run(p + prec_at + 1,
                                                  prec_n - 1));
        else
            shown = apy_format(value, apy_str_copy(spec, sn));
        if (!shown) { free(out); return 0; }
        /* ONE CONVERSION AND NOTHING ELSE HANDS THE ARGUMENT BACK: `"%s" % s`
           IS `s` in CPython, because `%s` calls `PyObject_Str` -- which
           answers an exact str unchanged -- and the format writer, with
           nothing written into it yet and nothing to follow, adopts that one
           object as its result. Measured against CPython 3.14 with
           `s = "abc"`:
               ("%s" % s)  is s -> True        ("%3s" % s) is s -> True
               ("%(a)s" % {"a": s}) is s -> True
               ("%.5s" % s) is s -> True       ("%.2s" % s) is s -> False
               ("%5s" % s) is s -> False       ("%s%s" % (s, "")) is s -> False
           `shown == given` carries the padding and the truncation rows
           without repeating them: `apy_format` above has just decided the
           same question, and it answers the object it was given only when
           the spec changed nothing.

           `%c` IS THE ONE CONVERSION THIS DOES NOT HOLD FOR, and it is not
           an oversight of CPython's: `%c` goes through `formatchar`, which
           writes a CHARACTER into the buffer rather than handing the writer
           an object, so `("%c" % w) is w` is False for a one-character `w`
           that is not latin-1 while `("%s" % w) is w` is True. For an ASCII
           one both are True there and here, by way of the shared
           one-character cell rather than by way of this rule.

           A BYTES FORMAT IS LEFT OUT because its result must be bytes and
           the loop's own `%s` re-tags its argument into a str to pad it;
           `(b"%s" % b"ab") is b"ab"` is False in CPython too. */
        if (conv_at == 0 && i == n && shown == given && conv != 'c'
                && O(fmt)->kind == APY_STR_K)
            lone = shown;
        while (out_n + O(shown)->v.s.n >= out_cap) {
            out_cap = out_cap * 2 + O(shown)->v.s.n;
            out = (char *)realloc(out, (size_t)out_cap + 1);
        }
        memcpy(out + out_n, APY_CSTR(shown), (size_t)O(shown)->v.s.n);
        out_n += O(shown)->v.s.n;
    }
    /* A MAPPING has nothing to leave unconsumed: its entries are reached by
       name, and an unused one is ordinary. */
    if (!mapping && at < supplied) {
        free(out);
        return apy_fail("TypeError",
                        "not all arguments converted during string "
                        "formatting");
    }
    out[out_n] = 0;
    /* THE LONE ARGUMENT, and only once the two checks above have had their
       say: `"%s" % (s, 1)` is "not all arguments converted" in CPython even
       though its one conversion did hand `s` straight through, so the
       shortcut is remembered during the loop and taken here rather than
       returned from inside it. */
    if (lone) { free(out); return lone; }
    /* `b"%d" % 3` IS BYTES. The whole of the difference is the kind: the
       format string's own bytes are ASCII either way, and every conversion
       above produced text. */
    if (O(fmt)->kind == APY_BYTES_K) return apy_bytes_take(out, out_n);
    return apy_str_take(out, out_n);
}

/* Re-tag a str METHOD'S RESULT to match its receiver.
   `b"a b".split()` answers a list of BYTES, not of str, and `b.find(...)`
   answers an int that must be left alone -- so this converts str results and
   the str elements of a sequence result, and nothing else. One place, at the
   call site, rather than a change to each of the fifty-odd methods. */
APY_API apy_value apy_str_like(apy_value recv, apy_value out) {
    /* AND A SUBCLASS'S NO-OP ANSWERS A FRESH PLAIN ONE. This is the call
       site's fixup and `recv` is the value the PROGRAM wrote, still wrapped
       -- the only point on the method route that can still see a `class
       S(str)`, because `apy_method_self` unwrapped it before the method ran.
       See `apy_inst_text_result`. */
    out = apy_inst_text_result(recv, out);
    if (!out) return out;
    /* AND THE RECEIVER'S KIND IS THE HELD ONE. The fixup above needed `recv`
       wrapped, and everything from here on is asking what tag the result
       should WEAR -- which for a `class B(bytes)` is the tag of the bytes it
       carries. Without this the test below saw APY_INST_K, returned, and
       `B(b"ab").upper()` answered a plain `'AB'`: a str where the program
       has a bytes, so the next `+` or `.decode()` on it fails. The str-held
       case needs nothing, since a str result already wears the right tag. */
    if (O(recv)->kind == APY_INST_K && O(recv)->v.o.held)
        recv = O(recv)->v.o.held;
    if (O(recv)->kind != APY_BYTES_K) return out;
    /* A RESULT THAT IS ALREADY BYTES STILL MAY NOT BE THE RIGHT ONE.
       `bytearray(b"a-b").partition(b"-")` hands back the SEPARATOR the
       caller passed, which is immutable bytes, and Python answers a
       bytearray for all three pieces. Copying is what makes that safe: the
       separator is the caller's object and setting `mut` on it in place
       would turn their `b"-"` into a bytearray. */
    if (O(out)->kind == APY_STR_K
            || (O(out)->kind == APY_BYTES_K
                && O(out)->v.s.mut != O(recv)->v.s.mut)) {
        /* A BYTEARRAY'S METHODS ANSWER A BYTEARRAY. `mut` is the whole of
           what separates the two kinds, so it has to travel with the tag:
           `bytearray(b"ab").upper()` is a bytearray in Python and came back
           as bytes here -- which a program then could not write into. */
        if (O(recv)->v.s.mut)
            return apy_bytearray_copy(O(out)->v.s.p, O(out)->v.s.n);
        return apy_bytes_copy(O(out)->v.s.p, O(out)->v.s.n);
    }
    if (apy_is_seq(out)) {
        int64_t i;
        for (i = 0; i < O(out)->v.q.n; i++)
            O(out)->v.q.items[i] = apy_str_like(recv, O(out)->v.q.items[i]);
    }
    return out;
}

APY_API apy_value apy_str_format(apy_value fmt, apy_value args, apy_value kw) {
    int64_t auto_at = 0;
    int auto_used = 0, explicit_used = 0;
    return apy_format_at(fmt, args, kw, &auto_at, &auto_used, &explicit_used);
}

/* `s.format_map(m)` -- `format` with the mapping handed over WHOLE rather
   than built from keywords. The difference CPython draws is that the mapping
   is not copied and may be any mapping, so `{k}` reaches `m[k]` and a missing
   key is the mapping's KeyError rather than a formatting error of its own. */
APY_API apy_value apy_str_format_map(apy_value fmt, apy_value mapping) {
    apy_value empty;
    fmt = apy_text_like(fmt);
    if (O(fmt)->kind != APY_STR_K)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'format_map'%s",
                         apy_kind_name(fmt), "");
    /* NO CHECK HERE. CPython SUBSCRIPTS the mapping, and only once a field
       asks it for a key -- so `"".format_map(None)` is `''` rather than a
       complaint about None. `apy_format_at` makes it where the subscript
       would have been. */
    empty = apy_list_new(1);
    if (!empty) return 0;
    return apy_str_format(fmt, empty, mapping);
}

"""
