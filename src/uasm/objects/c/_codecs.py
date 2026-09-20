"""The object runtime, in C: codecs.

ONE PART OF ONE TRANSLATION UNIT. `c/__init__.py` concatenates
these in order and the result is the file it always was, so a
definition here may rely on anything in an earlier part and
nothing in a later one. Sections, in order:
  * codecs
"""

C = r"""/* --- codecs ------------------------------------------------------------- */
/* Text is held as UTF-8, so `encode("utf-8")` is a re-tag and every other
   encoding is a real conversion. Both directions go through CODE POINTS: the
   internal form is decoded to them and the target built from them, which is
   one shared middle rather than a matrix of pairs. */

/* Which codec a name means. Canonicalised the way CPython does -- case and
   the `-`/`_` distinction do not matter -- so `UTF_8` and `utf-8` are one. */
enum { APY_ENC_UTF8 = 0, APY_ENC_ASCII, APY_ENC_LATIN1,
       APY_ENC_UTF16, APY_ENC_UTF16LE, APY_ENC_UTF16BE,
       APY_ENC_UTF32, APY_ENC_UTF32LE, APY_ENC_UTF32BE, APY_ENC_UNKNOWN };

/* Defined below, beside `apy_codec_arg`, which it is the strict half of. */
static apy_value apy_codec_given(const char *who, const char *slot,
                                 apy_value v);

static int apy_codec_of(apy_value name) {
    char buf[32];
    int64_t i, n;
    const char *p;
    if (!name || O(name)->kind != APY_STR_K) return APY_ENC_UTF8;
    p = O(name)->v.s.p;
    n = O(name)->v.s.n;
    if (n >= (int64_t)sizeof buf) return APY_ENC_UNKNOWN;
    for (i = 0; i < n; i++) {
        char c = p[i];
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        buf[i] = (c == '_') ? '-' : c;
    }
    buf[n] = 0;
    if (!strcmp(buf, "utf-8") || !strcmp(buf, "utf8")
        || !strcmp(buf, "u8")) return APY_ENC_UTF8;
    if (!strcmp(buf, "ascii") || !strcmp(buf, "us-ascii")
        || !strcmp(buf, "646")) return APY_ENC_ASCII;
    if (!strcmp(buf, "latin-1") || !strcmp(buf, "latin1")
        || !strcmp(buf, "iso-8859-1") || !strcmp(buf, "l1")
        || !strcmp(buf, "8859")) return APY_ENC_LATIN1;
    if (!strcmp(buf, "utf-16") || !strcmp(buf, "utf16")) return APY_ENC_UTF16;
    if (!strcmp(buf, "utf-16-le") || !strcmp(buf, "utf-16le"))
        return APY_ENC_UTF16LE;
    if (!strcmp(buf, "utf-16-be") || !strcmp(buf, "utf-16be"))
        return APY_ENC_UTF16BE;
    if (!strcmp(buf, "utf-32") || !strcmp(buf, "utf32")) return APY_ENC_UTF32;
    if (!strcmp(buf, "utf-32-le") || !strcmp(buf, "utf-32le"))
        return APY_ENC_UTF32LE;
    if (!strcmp(buf, "utf-32-be") || !strcmp(buf, "utf-32be"))
        return APY_ENC_UTF32BE;
    return APY_ENC_UNKNOWN;
}

/* THE ERROR HANDLER, as a small code. Every handler CPython answers without
   a registered callback -- `namereplace` is the one left out, because it
   spells a character by its UNICODE NAME and the name table is a bundled
   module rather than something the runtime carries.

   THREE OF THESE ARE WHY A SURROGATE HAD NOWHERE TO GO. The cell holds one
   as WTF-8 since the surrogate-literal round, so `"a\udcffb".encode(...)`
   finally has something real to work from -- and every compiled path used to
   hand the raw bytes back whatever the handler said, which is five wrong
   answers wearing one shape. */
enum { APY_ERR_STRICT = 0, APY_ERR_REPLACE, APY_ERR_IGNORE,
       APY_ERR_BACKSLASH, APY_ERR_XMLCHARREF, APY_ERR_SURROGATEESCAPE,
       APY_ERR_SURROGATEPASS };

static int apy_errors_of(apy_value name) {
    if (!name || O(name)->kind != APY_STR_K) return APY_ERR_STRICT;
    if (!strcmp(APY_CSTR(name), "replace")) return APY_ERR_REPLACE;
    if (!strcmp(APY_CSTR(name), "ignore")) return APY_ERR_IGNORE;
    if (!strcmp(APY_CSTR(name), "backslashreplace")) return APY_ERR_BACKSLASH;
    if (!strcmp(APY_CSTR(name), "xmlcharrefreplace"))
        return APY_ERR_XMLCHARREF;
    if (!strcmp(APY_CSTR(name), "surrogateescape"))
        return APY_ERR_SURROGATEESCAPE;
    if (!strcmp(APY_CSTR(name), "surrogatepass"))
        return APY_ERR_SURROGATEPASS;
    return APY_ERR_STRICT;
}

/* A CODE POINT AS CPYTHON WRITES IT IN AN ERROR MESSAGE: `'\xe9'`,
   `'\udcff'`, `'\U0001f600'` -- ASCII-safe whatever the terminal is, which
   is what the exception text uses rather than the terminal-aware repr. */
static int64_t apy_cp_shown(char *out, uint32_t cp) {
    if (cp >= 0x20 && cp < 0x7F) { out[0] = (char)cp; return 1; }
    if (cp < 0x100) return snprintf(out, 12, "\\x%02x", (unsigned)cp);
    if (cp < 0x10000) return snprintf(out, 12, "\\u%04x", (unsigned)cp);
    return snprintf(out, 12, "\\U%08x", (unsigned)cp);
}

/* `'utf-8' codec can't encode character '\udcff' in position 1: surrogates
   not allowed` -- the whole sentence, which used to stop after `character`.
   The POSITION IS COUNTED IN CHARACTERS, not in the bytes the cell holds.

   A RUN IS ONE COMPLAINT AND LOSES THE CHARACTER. CPython reports
   consecutive unencodable characters together -- `can't encode characters in
   position 0-1` -- and names none of them, which is why `run` decides the
   shape of the sentence rather than only its numbers. */
static apy_value apy_encode_failed(const char *codec, uint32_t cp,
                                   int64_t at, int64_t run,
                                   const char *why) {
    char shown[16], buf[200];
    int64_t used;
    if (run > 1) {
        snprintf(buf, sizeof buf, "'%s' codec can't encode characters in "
                 "position %lld-%lld: %s", codec, (long long)at,
                 (long long)(at + run - 1), why);
        return apy_fail("UnicodeEncodeError", buf);
    }
    used = apy_cp_shown(shown, cp);
    shown[used] = 0;
    snprintf(buf, sizeof buf, "'%s' codec can't encode character '%s' in "
             "position %lld: %s", codec, shown, (long long)at, why);
    return apy_fail("UnicodeEncodeError", buf);
}

/* `'utf-8' codec can't decode byte 0xed in position 1: invalid continuation
   byte` -- the same sentence from the other side, and the same split: a
   MAXIMAL SUBPART longer than one byte is `bytes in position 1-2` and names
   none of them. */
static apy_value apy_decode_failed(const char *codec, unsigned byte,
                                   int64_t at, int64_t run,
                                   const char *why) {
    char buf[200];
    if (run > 1)
        snprintf(buf, sizeof buf, "'%s' codec can't decode bytes in "
                 "position %lld-%lld: %s", codec, (long long)at,
                 (long long)(at + run - 1), why);
    else
        snprintf(buf, sizeof buf, "'%s' codec can't decode byte 0x%02x in "
                 "position %lld: %s", codec, byte, (long long)at, why);
    return apy_fail("UnicodeDecodeError", buf);
}

/* HOW MANY BYTES AT `i` ARE A PREFIX OF A SEQUENCE AND STILL WRONG -- the
   MAXIMAL SUBPART, which is what decides how many U+FFFD a `replace` puts
   and what range a refusal names. Never less than one.

   THE CONTINUATION RANGES ARE NOT ALL 80..BF, and that is the whole of the
   subtlety: `E0` needs `A0..BF`, `ED` needs `80..9F` -- which is what makes
   a WTF-8 surrogate three separate one-byte errors rather than one three-
   byte one -- `F0` needs `90..BF` and `F4` needs `80..8F`. */
static int64_t apy_utf8_subpart(const unsigned char *p, int64_t n,
                                int64_t i, const char **why) {
    unsigned char c = p[i], lo = 0x80, hi = 0xBF;
    int64_t need, k;
    *why = "invalid start byte";
    if (c < 0xC2 || c > 0xF4) return 1;
    need = c < 0xE0 ? 1 : c < 0xF0 ? 2 : 3;
    if (c == 0xE0) lo = 0xA0;
    else if (c == 0xED) hi = 0x9F;
    else if (c == 0xF0) lo = 0x90;
    else if (c == 0xF4) hi = 0x8F;
    for (k = 1; k <= need; k++) {
        unsigned char want_lo = k == 1 ? lo : 0x80;
        unsigned char want_hi = k == 1 ? hi : 0xBF;
        if (i + k >= n) { *why = "unexpected end of data"; return k; }
        if (p[i + k] < want_lo || p[i + k] > want_hi) {
            *why = "invalid continuation byte";
            return k;
        }
    }
    /* A COMPLETE AND VALID SEQUENCE, which only reaches here when the caller
       rejected it for what it MEANS rather than how it is spelled. */
    *why = "invalid continuation byte";
    return 1;
}

/* HOW MANY CHARACTERS FROM `i` THIS CODEC CANNOT ENCODE, in a row. CPython
   reports consecutive ones as ONE complaint naming a range, and stops at the
   first character it can encode. */
static int64_t apy_bad_run(const unsigned char *p, int64_t n, int64_t i,
                           int codec) {
    int64_t run = 0;
    while (i < n) {
        uint32_t cp;
        int64_t used = apy_utf8_step(p, n, i, &cp);
        if (!used) break;
        if (codec == APY_ENC_ASCII) { if (cp < 0x80) break; }
        else if (codec == APY_ENC_LATIN1) { if (cp < 0x100) break; }
        else if (cp < 0xD800 || cp > 0xDFFF) break;
        run++;
        i += used;
    }
    return run ? run : 1;
}

/* `\udcff` and `&#56575;` -- the two handlers that WRITE THE CHARACTER OUT
   rather than dropping it. Answers how many bytes it put. */
static int64_t apy_escape_put(char *out, int handler, uint32_t cp) {
    if (handler == APY_ERR_XMLCHARREF)
        return snprintf(out, 16, "&#%u;", (unsigned)cp);
    if (cp < 0x100) return snprintf(out, 16, "\\x%02x", (unsigned)cp);
    if (cp < 0x10000) return snprintf(out, 16, "\\u%04x", (unsigned)cp);
    return snprintf(out, 16, "\\U%08x", (unsigned)cp);
}

/* One code point out of UTF-8. Answers how many bytes it consumed, or 0 for
   a malformed sequence -- which is the whole of the validation the strict
   handler needs. */
APY_API int64_t apy_utf8_step_of(apy_value pv, int64_t n, int64_t i,
                             apy_value outv) {
    const unsigned char *p = (const unsigned char *)pv;
    uint32_t *out = (uint32_t *)outv;
    unsigned char c = p[i];
    if (c < 0x80) { *out = c; return 1; }
    if ((c & 0xE0) == 0xC0 && i + 1 < n && (p[i+1] & 0xC0) == 0x80) {
        *out = (uint32_t)((c & 0x1F) << 6) | (uint32_t)(p[i+1] & 0x3F);
        return *out >= 0x80 ? 2 : 0;      /* an overlong form is malformed */
    }
    if ((c & 0xF0) == 0xE0 && i + 2 < n && (p[i+1] & 0xC0) == 0x80
        && (p[i+2] & 0xC0) == 0x80) {
        *out = (uint32_t)((c & 0x0F) << 12)
             | (uint32_t)((p[i+1] & 0x3F) << 6) | (uint32_t)(p[i+2] & 0x3F);
        return *out >= 0x800 ? 3 : 0;
    }
    if ((c & 0xF8) == 0xF0 && i + 3 < n && (p[i+1] & 0xC0) == 0x80
        && (p[i+2] & 0xC0) == 0x80 && (p[i+3] & 0xC0) == 0x80) {
        *out = (uint32_t)((c & 0x07) << 18)
             | (uint32_t)((p[i+1] & 0x3F) << 12)
             | (uint32_t)((p[i+2] & 0x3F) << 6) | (uint32_t)(p[i+3] & 0x3F);
        return (*out >= 0x10000 && *out <= 0x10FFFF) ? 4 : 0;
    }
    return 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int64_t apy_utf8_step(const unsigned char *p, int64_t n,
                             int64_t i, uint32_t *out) {
    return apy_utf8_step_of((apy_value)(uintptr_t)p, n, i,
                            (apy_value)(uintptr_t)out);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */

/* One code point INTO UTF-8. Answers how many bytes it wrote. */
static int apy_utf8_put(char *out, uint32_t cp) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

APY_API apy_value apy_str_encode(apy_value s, apy_value encoding,
                                 apy_value errors) {
    int codec, handler;
    const unsigned char *p;
    int64_t n, i, at = 0, shown = 0;
    char *buf;
    apy_value out;
    if (O(s)->kind != APY_STR_K)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'encode'%s",
                         apy_kind_name(s), "");
    /* A str SUBCLASS NAMES A CODEC. `"ab".encode(S("utf-8"))` is ordinary
       Python -- see `apy_text_like`. */
    encoding = apy_text_like(encoding);
    errors = apy_text_like(errors);
    if (apy_codec_given("encode", "encoding", encoding)) return 0;
    if (apy_codec_given("encode", "errors", errors)) return 0;
    codec = apy_codec_of(encoding);
    handler = apy_errors_of(errors);
    if (codec == APY_ENC_UNKNOWN)
        return apy_fail2("LookupError", "unknown encoding: %s%s",
                         APY_CSTR(encoding), "");
    p = (const unsigned char *)O(s)->v.s.p;
    n = O(s)->v.s.n;
    if (codec == APY_ENC_UTF8) {
        /* ALMOST THE INTERNAL FORM: the cell is UTF-8 except where it holds a
           SURROGATE, which it keeps as WTF-8 so that `"a\udcffb"` can exist
           at all. UTF-8 has no such character, so the handler decides -- and
           a straight copy answered the raw bytes for every one of the seven,
           which is six wrong answers and one right one by accident. */
        int64_t seen = 0;
        for (i = 0; i < n; ) {
            uint32_t cp;
            int64_t used = apy_utf8_step(p, n, i, &cp);
            if (!used) { i++; continue; }
            if (cp >= 0xD800 && cp <= 0xDFFF) break;
            i += used;
            seen++;
        }
        if (i >= n) {
            return apy_bytes_copy(O(s)->v.s.p, n);
        }
        /* FOUR BYTES PER CHARACTER covers `\Uxxxxxxxx`, the widest thing any
           handler writes for one. */
        buf = (char *)malloc((size_t)(n * 10 + 16));
        if (!buf) { fputs("uasm: out of memory\n", stderr); exit(1); }
        memcpy(buf, p, (size_t)i);
        at = i;
        for (; i < n; ) {
            uint32_t cp;
            int64_t used = apy_utf8_step(p, n, i, &cp);
            if (!used) { buf[at++] = (char)p[i++]; continue; }
            if (cp < 0xD800 || cp > 0xDFFF) {
                memcpy(buf + at, p + i, (size_t)used);
                at += used;
                i += used;
                seen++;
                continue;
            }
            if (handler == APY_ERR_SURROGATEPASS) {
                /* THE WTF-8 BYTES, WHICH IS WHAT THE CELL ALREADY HOLDS.
                   `surrogatepass` is the handler that says "write it anyway",
                   and the internal form is exactly that encoding. */
                memcpy(buf + at, p + i, (size_t)used);
                at += used;
            } else if (handler == APY_ERR_SURROGATEESCAPE
                       && cp >= 0xDC80 && cp <= 0xDCFF) {
                /* THE LOW BYTE BACK. PEP 383: a byte that would not decode
                   was parked at U+DC80 + byte, and this is the way out. Only
                   that range -- a surrogate from anywhere else was never a
                   byte and CPython refuses it. */
                buf[at++] = (char)(cp - 0xDC00);
            } else if (handler == APY_ERR_IGNORE) {
                /* nothing */
            } else if (handler == APY_ERR_REPLACE) {
                buf[at++] = '?';
            } else if (handler == APY_ERR_BACKSLASH
                       || handler == APY_ERR_XMLCHARREF) {
                at += apy_escape_put(buf + at, handler, cp);
            } else {
                free(buf);
                return apy_encode_failed("utf-8", cp, seen,
                                         apy_bad_run(p, n, i, codec),
                                         "surrogates not allowed");
            }
            i += used;
            seen++;
        }
        out = apy_bytes_copy(buf, at);
        free(buf);
        return out;
    }
    /* Four bytes per code point covers every target, and a code point is at
       least one byte of the source -- so `4 * n` can never be short. */
    buf = (char *)malloc((size_t)(n * 4 + 8));
    if (!buf) { fputs("uasm: out of memory\n", stderr); exit(1); }
    /* THE BOM IS PART OF THE ENCODING for the unsuffixed spellings, which is
       what makes `len("a".encode("utf-16"))` 4 rather than 2. */
    if (codec == APY_ENC_UTF16) {
        buf[at++] = (char)0xFF; buf[at++] = (char)0xFE;
    } else if (codec == APY_ENC_UTF32) {
        buf[at++] = (char)0xFF; buf[at++] = (char)0xFE;
        buf[at++] = 0; buf[at++] = 0;
    }
    for (i = 0; i < n; ) {
        uint32_t cp;
        int64_t used = apy_utf8_step(p, n, i, &cp);
        if (!used) { cp = 0xFFFD; used = 1; }
        i += used;
        if (codec == APY_ENC_ASCII || codec == APY_ENC_LATIN1) {
            uint32_t limit = codec == APY_ENC_ASCII ? 0x80u : 0x100u;
            const char *named = codec == APY_ENC_ASCII ? "ascii" : "latin-1";
            if (cp >= limit) {
                if (handler == APY_ERR_IGNORE) { shown++; continue; }
                if (handler == APY_ERR_REPLACE) {
                    buf[at++] = '?';
                    shown++;
                    continue;
                }
                if (handler == APY_ERR_BACKSLASH
                        || handler == APY_ERR_XMLCHARREF) {
                    at += apy_escape_put(buf + at, handler, cp);
                    shown++;
                    continue;
                }
                /* PEP 383 AGAIN, and it reaches every narrow codec: a byte
                   parked at U+DC80 comes back out as that byte whatever the
                   encoding was, which is what makes a filename read from the
                   system writable again. */
                if (handler == APY_ERR_SURROGATEESCAPE
                        && cp >= 0xDC80 && cp <= 0xDCFF) {
                    buf[at++] = (char)(cp - 0xDC00);
                    shown++;
                    continue;
                }
                free(buf);
                return apy_encode_failed(
                    named, cp, shown, apy_bad_run(p, n, i - used, codec),
                    codec == APY_ENC_ASCII ? "ordinal not in range(128)"
                                           : "ordinal not in range(256)");
            }
            buf[at++] = (char)cp;
            shown++;
            continue;
        }
        shown++;
        if (codec == APY_ENC_UTF32 || codec == APY_ENC_UTF32LE
            || codec == APY_ENC_UTF32BE) {
            int be = codec == APY_ENC_UTF32BE;
            int k;
            for (k = 0; k < 4; k++) {
                int shift = be ? (24 - 8 * k) : (8 * k);
                buf[at++] = (char)((cp >> shift) & 0xFF);
            }
            continue;
        }
        {   /* UTF-16, with a surrogate pair above the BMP. */
            int be = codec == APY_ENC_UTF16BE;
            uint32_t units[2];
            int count = 1, k;
            if (cp >= 0x10000) {
                uint32_t v = cp - 0x10000;
                units[0] = 0xD800 + (v >> 10);
                units[1] = 0xDC00 + (v & 0x3FF);
                count = 2;
            } else units[0] = cp;
            for (k = 0; k < count; k++) {
                if (be) {
                    buf[at++] = (char)((units[k] >> 8) & 0xFF);
                    buf[at++] = (char)(units[k] & 0xFF);
                } else {
                    buf[at++] = (char)(units[k] & 0xFF);
                    buf[at++] = (char)((units[k] >> 8) & 0xFF);
                }
            }
        }
    }
    out = apy_bytes_copy(buf, at);
    free(buf);
    return out;
}

APY_API apy_value apy_bytes_decode(apy_value b, apy_value encoding,
                                   apy_value errors) {
    int codec, handler;
    const unsigned char *p;
    int64_t n, i, at = 0;
    char *buf;
    /* A VIEW IS NOT A RECEIVER FOR THIS. `memoryview(b"a").decode()` is an
       AttributeError in CPython -- a view has no `decode` -- and converting
       one here answered the text instead. The CONSTRUCTOR spelling does
       take a view, and `apy_str_ctor` converts it before calling. */
    if (O(b)->kind != APY_BYTES_K)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'decode'%s",
                         apy_kind_name(b), "");
    /* A str SUBCLASS NAMES A CODEC. `b"ab".decode(S("utf-8"))` is ordinary
       Python -- see `apy_text_like`. */
    encoding = apy_text_like(encoding);
    errors = apy_text_like(errors);
    if (apy_codec_given("decode", "encoding", encoding)) return 0;
    if (apy_codec_given("decode", "errors", errors)) return 0;
    codec = apy_codec_of(encoding);
    handler = apy_errors_of(errors);
    if (codec == APY_ENC_UNKNOWN)
        return apy_fail2("LookupError", "unknown encoding: %s%s",
                         APY_CSTR(encoding), "");
    p = (const unsigned char *)O(b)->v.s.p;
    n = O(b)->v.s.n;
    /* Three bytes of UTF-8 per input byte is the worst case for every codec
       here -- one latin-1 byte becomes at most two, one UTF-16 unit at most
       three, and a byte parked at U+DC80 by `surrogateescape` exactly three
       -- so this cannot be short. */
    buf = (char *)malloc((size_t)(n * 3 + 8));
    if (!buf) { fputs("uasm: out of memory\n", stderr); exit(1); }
    if (codec == APY_ENC_UTF8) {
        for (i = 0; i < n; ) {
            uint32_t cp;
            int64_t used = apy_utf8_step(p, n, i, &cp);
            /* A WTF-8 SURROGATE IS NOT UTF-8. The step accepts one because
               the cell stores text that way, but a byte string that spells
               ED B3 BF is malformed input -- CPython refuses it byte by byte
               and only `surrogatepass` lets it through. */
            int bad = !used || (cp >= 0xD800 && cp <= 0xDFFF
                                && handler != APY_ERR_SURROGATEPASS);
            if (bad) {
                /* THE MAXIMAL SUBPART, which is one U+FFFD however long it
                   is: a truncated four-byte sequence is ONE error and a
                   WTF-8 surrogate is three, because `ED` accepts only
                   `80..9F` and the bytes after it start nothing. */
                const char *why;
                int64_t part = apy_utf8_subpart(p, n, i, &why), k;
                if (handler == APY_ERR_IGNORE) { i += part; continue; }
                if (handler == APY_ERR_REPLACE) {
                    at += apy_utf8_put(buf + at, 0xFFFD);
                    i += part;
                    continue;
                }
                if (handler == APY_ERR_SURROGATEESCAPE) {
                    /* PEP 383: each byte is parked at U+DC80 + byte, so it
                       can be handed back unchanged when the text is encoded.
                       EVERY BYTE OF THE SUBPART, not one per subpart --
                       nothing could come back otherwise. */
                    for (k = 0; k < part; k++)
                        at += apy_utf8_put(buf + at, 0xDC00u + p[i + k]);
                    i += part;
                    continue;
                }
                free(buf);
                return apy_decode_failed("utf-8", p[i], i, part, why);
            }
            memcpy(buf + at, p + i, (size_t)used);
            at += used;
            i += used;
        }
        return apy_str_take(buf, at);
    }
    if (codec == APY_ENC_LATIN1 || codec == APY_ENC_ASCII) {
        for (i = 0; i < n; i++) {
            if (codec == APY_ENC_ASCII && p[i] >= 0x80) {
                if (handler == APY_ERR_IGNORE) continue;
                if (handler == APY_ERR_REPLACE) {
                    at += apy_utf8_put(buf + at, 0xFFFD);
                    continue;
                }
                if (handler == APY_ERR_SURROGATEESCAPE) {
                    at += apy_utf8_put(buf + at, 0xDC00u + p[i]);
                    continue;
                }
                free(buf);
                return apy_decode_failed("ascii", p[i], i, 1,
                                         "ordinal not in range(128)");
            }
            /* EVERY BYTE IS A CODE POINT in latin-1, which is what makes it
               the round-trip encoding for arbitrary octets. */
            at += apy_utf8_put(buf + at, p[i]);
        }
        return apy_str_take(buf, at);
    }
    {   /* UTF-16 and UTF-32, with the BOM consumed where one is allowed. */
        int wide = (codec == APY_ENC_UTF32 || codec == APY_ENC_UTF32LE
                    || codec == APY_ENC_UTF32BE) ? 4 : 2;
        int be = (codec == APY_ENC_UTF16BE || codec == APY_ENC_UTF32BE);
        i = 0;
        if (codec == APY_ENC_UTF16 && n >= 2) {
            if (p[0] == 0xFF && p[1] == 0xFE) { be = 0; i = 2; }
            else if (p[0] == 0xFE && p[1] == 0xFF) { be = 1; i = 2; }
        } else if (codec == APY_ENC_UTF32 && n >= 4) {
            if (p[0] == 0xFF && p[1] == 0xFE && !p[2] && !p[3]) {
                be = 0; i = 4;
            } else if (!p[0] && !p[1] && p[2] == 0xFE && p[3] == 0xFF) {
                be = 1; i = 4;
            }
        }
        for (; i + wide <= n; i += wide) {
            uint32_t cp = 0;
            int k;
            for (k = 0; k < wide; k++) {
                int shift = be ? (8 * (wide - 1 - k)) : (8 * k);
                cp |= (uint32_t)p[i + k] << shift;
            }
            /* A SURROGATE PAIR IS ONE CHARACTER. Only UTF-16 has them, and a
               lone half is as malformed as a truncated UTF-8 sequence. */
            if (wide == 2 && cp >= 0xD800 && cp <= 0xDBFF && i + 4 <= n) {
                uint32_t low = 0;
                for (k = 0; k < 2; k++) {
                    int shift = be ? (8 * (1 - k)) : (8 * k);
                    low |= (uint32_t)p[i + 2 + k] << shift;
                }
                if (low >= 0xDC00 && low <= 0xDFFF) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    i += 2;
                }
            }
            at += apy_utf8_put(buf + at, cp);
        }
        return apy_str_take(buf, at);
    }
}

/* `str() argument 'encoding' must be str, not int` -- THE CONSTRUCTOR'S OWN
   WORDING, which NAMES the parameter where the method family beside it
   NUMBERS it (`decode() argument 'encoding'` is the method's, and
   `find() argument 1` is `apy_arg_must_be_str`'s). A non-string encoding
   used to be passed straight through to the codec lookup, which answered
   `LookupError: unknown encoding: 5` in the interpreter and IGNORED it
   entirely when compiled -- a three-way split on one call.

   NONE IS NOT CHECKED HERE, because None is how "not given" travels to the
   codec pair: `str(b, errors="replace")` has no encoding to pass and CPython
   defaults it to UTF-8. */
static apy_value apy_codec_arg(const char *who, const char *slot,
                               apy_value v) {
    char buf[160];
    if (O(v)->kind == APY_NONE_K || O(v)->kind == APY_STR_K) return 0;
    snprintf(buf, sizeof buf, "%s() argument '%s' must be str, not %s",
             who, slot, apy_kind_name(v));
    return apy_fail("TypeError", buf);
}

/* THE SAME, WHERE NONE IS NOT A DEFAULT EITHER. `"a".encode(None)` is a
   TypeError in CPython and `"a".encode()` is `"a".encode("utf-8")` -- the
   two are told apart by the lowering, which pads a slot the call left out
   with the DEFAULT'S OWN TEXT rather than with None. The constructors are
   the exception and keep `apy_codec_arg`: a None encoding there means the
   slot was never written, which is `encoding without a string argument`
   and not a bad encoding.

   `not None` AND NOT `not NoneType`, because the arg clinic writes the
   VALUE for these two and the type for everything else. */
static apy_value apy_codec_given(const char *who, const char *slot,
                                 apy_value v) {
    char buf[160];
    if (O(v)->kind == APY_STR_K) return 0;
    snprintf(buf, sizeof buf, "%s() argument '%s' must be str, not %s",
             who, slot,
             O(v)->kind == APY_NONE_K ? "None" : apy_kind_name(v));
    return apy_fail("TypeError", buf);
}

/* `bytes(s, encoding)` and `bytearray(s, encoding, errors)` -- THE
   CONSTRUCTOR SPELLING OF `.encode()`, and a different constructor from the
   one-argument form beside it: `bytes(xs)` is a sequence of octets and
   `bytes(s, "utf-8")` is `s.encode("utf-8")`. What tells them apart is
   whether an ENCODING was given at all.

   A NON-STR WITH AN ENCODING IS THE OTHER HALF of the refusal the
   one-argument form already makes: `bytes("a")` is `string argument without
   an encoding` and `bytes(b"a", "utf-8")` is `encoding without a string
   argument`. */
APY_API apy_value apy_bytes_ctor(apy_value v, apy_value encoding,
                                 apy_value errors, int64_t mut) {
    apy_value made;
    const char *who = mut ? "bytearray" : "bytes";
    encoding = apy_text_like(encoding);
    errors = apy_text_like(errors);
    if (apy_codec_arg(who, "encoding", encoding)) return 0;
    if (apy_codec_arg(who, "errors", errors)) return 0;
    /* A str SUBCLASS IS A str HERE TOO: `bytes(S("ab"), "utf-8")` encodes
       the text it holds -- see `apy_text_like`. */
    v = apy_text_like(v);
    if (O(v)->kind != APY_STR_K)
        return apy_fail("TypeError", "encoding without a string argument");
    /* A NONE SLOT IS ONE THE CALL NEVER WROTE, which is the constructor's
       own reading of it -- so the default is filled in HERE rather than
       accepted by `apy_str_encode`, which refuses a None the way CPython
       does for the method spelling. */
    if (O(encoding)->kind == APY_NONE_K) encoding = apy_lit("utf-8");
    if (O(errors)->kind == APY_NONE_K) errors = apy_lit("strict");
    made = apy_str_encode(v, encoding, errors);
    if (!made) return 0;
    if (mut) {
        /* A BYTEARRAY IS A FRESH CELL and not a re-tagged one: `encode`
           answers bytes, and writing `mut` into it would make the caller's
           own value writable. */
        return apy_bytearray_copy(O(made)->v.s.p, O(made)->v.s.n);
    }
    return made;
}

/* `str(b, encoding)` -- the constructor spelling of `.decode()`, and the
   same split: `str(b)` is the REPR of the bytes and `str(b, "utf-8")` is the
   text they spell. */
APY_API apy_value apy_str_ctor(apy_value v, apy_value encoding,
                               apy_value errors) {
    if (apy_codec_arg("str", "encoding", encoding)) return 0;
    if (apy_codec_arg("str", "errors", errors)) return 0;
    /* DECODING A str IS ITS OWN SENTENCE, and CPython gives it for a
       SUBCLASS too: `str("ab", "utf-8")` and `str(S("ab"), "utf-8")` are
       both `decoding str is not supported`, while `str(5, "utf-8")` is the
       bytes-like message naming int. Both were the second message here. */
    if (O(apy_text_like(v))->kind == APY_STR_K)
        return apy_fail("TypeError", "decoding str is not supported");
    if (O(v)->kind == APY_MVIEW_K) v = apy_mview_bytes(v);
    /* A bytes SUBCLASS IS A BYTES-LIKE OBJECT. Written out rather than
       through `apy_text_like`, which also reaches past a `class S(str)`:
       the refusal below names what the program WROTE, and CPython names the
       subclass. */
    if (O(v)->kind == APY_INST_K && O(v)->v.o.held
            && O(O(v)->v.o.held)->kind == APY_BYTES_K)
        v = O(v)->v.o.held;
    if (O(v)->kind != APY_BYTES_K)
        return apy_fail2("TypeError",
                         "decoding to str: need a bytes-like object, %s "
                         "found%s", apy_kind_name(v), "");
    /* The same, and for the same reason -- see `apy_bytes_ctor`. */
    if (O(encoding)->kind == APY_NONE_K) encoding = apy_lit("utf-8");
    if (O(errors)->kind == APY_NONE_K) errors = apy_lit("strict");
    /* A VIEW IS A BYTES-LIKE OBJECT HERE, which `decode` itself no longer
       accepts as a receiver: `str(memoryview(b"a"), "utf-8")` is text and
       `memoryview(b"a").decode()` is an AttributeError. */
    if (O(v)->kind == APY_MVIEW_K) {
        v = apy_mview_bytes(v);
        if (!v) return 0;
    }
    return apy_bytes_decode(v, encoding, errors);
}

/* `b.hex()`, `b.hex(sep)` and `b.hex(sep, bytes_per_sep)` -- the octets as
   lowercase hex pairs.

   THE GROUPING IS COUNTED FROM AN END AND WHICH END IS THE SIGN. A positive
   count groups from the RIGHT -- `bytes(range(1, 8)).hex(":", 2)` is
   `01:0203:0405:0607`, the leftover at the front -- and a negative one from
   the LEFT. That is CPython's rule and it is not the obvious one: the
   common use is a number written down, whose low end is the one that
   matters. A count of zero, or one no smaller than the length, separates
   nothing. */
APY_API apy_value apy_bytes_hex_n(apy_value b, apy_value sep,
                                  apy_value perv) {
    static const char *D = "0123456789abcdef";
    int64_t n, i, out = 0, per = 1, group = 0;
    char *buf;
    char s = 0;
    /* A VIEW HEXES THE BYTES IT SHOWS, which is most of what a program
       makes one to look at. The no-separator form reaches `apy_hex_of` and
       is converted there; this is the same conversion for the other. */
    if (O(b)->kind == APY_MVIEW_K) {
        b = apy_mview_bytes(b);
        if (!b) return 0;
    }
    if (O(b)->kind != APY_BYTES_K)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'hex'%s",
                         apy_kind_name(b), "");
    /* A NULL SLOT IS ONE THE CALL DID NOT WRITE, and it is the only way to
       tell `b.hex()` from `b.hex(None)` -- the first is the no-argument form
       whose default this fills in, and the second is a separator CPython
       refuses. The two spellings reach different symbols (`apy_hex_of` and
       `apy_bytes_hex`), which is where the distinction is made.

       THE COUNT IS CONVERTED FIRST, which is CPython's order: `b.hex(None,
       None)` complains about the integer and `b.hex(None)` about the
       separator's length. */
    if (perv) {
        if (O(perv)->kind != APY_INT_K && O(perv)->kind != APY_BIG_K
                && O(perv)->kind != APY_BOOL_K)
            return apy_fail2("TypeError",
                             "'%s' object cannot be interpreted as an "
                             "integer%s", apy_kind_name(perv), "");
        per = apy_as_int(perv);
        if (apy_error_occurred()) return 0;
    }
    if (sep) {
        /* CPython ASKS THE SEPARATOR FOR ITS LENGTH, so anything without one
           is a TypeError about `len` rather than about hex. */
        if (O(sep)->kind != APY_STR_K && O(sep)->kind != APY_BYTES_K)
            return apy_fail2("TypeError",
                             "object of type '%s' has no len()%s",
                             apy_kind_name(sep), "");
        if (O(sep)->v.s.n != 1)
            return apy_fail("ValueError", "sep must be length 1.");
        s = APY_CSTR(sep)[0];
    }
    n = O(b)->v.s.n;
    if (per < 0) {
        /* FROM THE LEFT, which is what a negative count means. */
        group = -per;
    } else if (per > 0) {
        group = per;
    }
    buf = (char *)malloc((size_t)(n * 3 + 2));
    if (!buf) { fputs("uasm: out of memory\n", stderr); exit(1); }
    for (i = 0; i < n; i++) {
        unsigned char c = (unsigned char)O(b)->v.s.p[i];
        if (s && i && group
                && (per < 0 ? i % group == 0 : (n - i) % group == 0))
            buf[out++] = s;
        buf[out++] = D[c >> 4];
        buf[out++] = D[c & 15];
    }
    buf[out] = 0;
    return apy_str_take(buf, out);
}

/* The one-separator form. ONE BYTE PER GROUP is the default CPython
   declares, and the two spellings are one implementation. The count slot is
   NULL because this call did not write one. */
APY_API apy_value apy_bytes_hex(apy_value b, apy_value sep) {
    return apy_bytes_hex_n(b, sep, 0);
}

/* `bytes.fromhex(text)` -- the inverse, ignoring ASCII spaces between pairs
   the way CPython does. */
APY_API apy_value apy_bytes_fromhex(apy_value self, apy_value text) {
    /* The RECEIVER is ignored and present only so the shape matches the
       method table's -- `b.fromhex(s)` and `bytes.fromhex(s)` are the same
       call, and one signature means one implementation. */
    int64_t n, i, out = 0;
    char *buf;
    int hi = -1;
    if (O(text)->kind != APY_STR_K)
        return apy_fail("TypeError", "fromhex() argument must be str");
    n = O(text)->v.s.n;
    buf = (char *)malloc((size_t)(n / 2 + 2));
    if (!buf) { fputs("uasm: out of memory\n", stderr); exit(1); }
    for (i = 0; i < n; i++) {
        char c = APY_CSTR(text)[i];
        int d;
        if (c == ' ' || c == '\t' || c == '\n') continue;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else {
            free(buf);
            return apy_fail("ValueError",
                            "non-hexadecimal number found in fromhex() arg");
        }
        if (hi < 0) hi = d;
        else { buf[out++] = (char)((hi << 4) | d); hi = -1; }
    }
    if (hi >= 0) {
        free(buf);
        return apy_fail("ValueError",
                        "non-hexadecimal number found in fromhex() arg");
    }
    {
        if (self && O(self)->kind == APY_BYTES_K && O(self)->v.s.mut)
            return apy_bytes_own(buf, out, 1);
        return apy_bytes_take(buf, out);
    }
}

/* WHICHEVER `fromhex` THE RECEIVER MEANT. A float's is a different reading
   entirely -- `0x1.8p+0` is one number, not three bytes -- and the receiver
   is the only thing that says which was written. THE SPLIT LIVES HERE rather
   than inside the bytes body because that body is IR, and the ported runtime
   has to define everything it calls: the hex float parser is the C's. */
APY_API apy_value apy_any_fromhex(apy_value self, apy_value text) {
    if (self && O(self)->kind == APY_FLOAT_K) return apy_float_fromhex(text);
    return apy_bytes_fromhex(self, text);
}

/* `bytearray.fromhex(text)` -- the same reading, a MUTABLE answer. CPython
   gives back the kind the method was reached through, so this one is a
   bytearray where the shared body above would hand out bytes. */
APY_API apy_value apy_bytearray_fromhex(apy_value self, apy_value text) {
    apy_value got = apy_bytes_fromhex(self, text);
    if (!got) return 0;
    if (O(got)->v.s.mut) return got;
    /* A COPY AND NOT A FLAG WRITTEN ON WHAT CAME BACK. The receiver here is
       the TYPE, so the call above took the immutable path -- which answers a
       SHARED cell for the empty and the one-byte values, and `mut = 1` on
       one of those turned every later `b"A"` in the program into a
       bytearray, literals included. `apy_bytearray_copy` is the same answer
       `apy_bytearray()` reaches for, and for the same reason. */
    return apy_bytearray_copy(O(got)->v.s.p, O(got)->v.s.n);
}

/* The `k`-th byte of an integer's MAGNITUDE, least significant first, and
   zero beyond the end -- which is what asking for a wider byte means.

   THE WHOLE REASON THIS EXISTS. `O(v)->v.i` is the int64 an int cell holds,
   and a BIG integer's cell does not hold its value there: it holds a pointer
   to the limbs. Reading it as a number put the object's ADDRESS into the
   answer, so `(2**63).to_bytes(16, 'little')` returned a heap address
   formatted as data. */
static int64_t apy_int_mag_byte(apy_value v, int64_t k) {
    if (apy_is_big(v)) {
        int64_t which = k / (int64_t)sizeof(apy_limb);
        if (which >= O(v)->v.big.n) return 0;
        return (int64_t)((O(v)->v.big.limb[which]
                          >> ((k % (int64_t)sizeof(apy_limb)) * 8)) & 0xFF);
    }
    if (k >= 8) return 0;
    {   /* NEGATION WRAPS FOR THE MOST NEGATIVE int64 and that is the right
           answer: -(-2**63) is -2**63 again, and read unsigned that is 2**63,
           which is exactly its magnitude. */
        int64_t m = O(v)->v.i;
        uint64_t u = (uint64_t)(m < 0 ? -m : m);
        return (int64_t)((u >> (k * 8)) & 0xFF);
    }
}

/* How many bytes the magnitude needs, with no leading zeroes. */
static int64_t apy_int_mag_len(apy_value v) {
    int64_t top = apy_is_big(v)
        ? O(v)->v.big.n * (int64_t)sizeof(apy_limb) : 8;
    while (top > 0 && apy_int_mag_byte(v, top - 1) == 0) top--;
    return top;
}

static int apy_int_is_neg(apy_value v) {
    if (apy_is_big(v)) return O(v)->v.big.neg != 0;
    return O(v)->v.i < 0;
}

/* Whether the value fits `n` bytes under the rule the caller asked for.

   THREE RULES, NOT ONE. Unsigned needs the magnitude to fit outright. Signed
   and positive loses the top bit to the sign, so 127 fits one byte and 128
   does not. Signed and NEGATIVE reaches one further -- -128 fits one byte --
   and that extra value is exactly the power of two, so it is the magnitude
   having a single one bit at the top that makes it fit. */
static int apy_to_bytes_fits(apy_value v, int64_t n, int64_t used,
                             int neg, int want_signed) {
    int64_t top, k;
    if (used > n) return 0;
    if (!want_signed) return 1;
    if (used < n) return 1;
    top = apy_int_mag_byte(v, n - 1);
    if (top < 128) return 1;
    if (!neg || top != 128) return 0;
    for (k = 0; k < n - 1; k++)
        if (apy_int_mag_byte(v, k) != 0) return 0;
    return 1;
}

/* `n.to_bytes(length, byteorder, signed)`. Big-endian unless told otherwise,
   which is the C's rule kept: anything that is not exactly "little" is big.

   THE MAGNITUDE IS READ BYTE BY BYTE rather than shifted out of an int64, so
   a big integer answers its value instead of its address. */
APY_API apy_value apy_to_bytes_n(apy_value v, apy_value length,
                                 apy_value order, apy_value signed_) {
    int64_t n, i, used;
    char *buf;
    int big, neg, want_signed;
    if (!apy_is_int_like(v))
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'to_bytes'%s",
                         apy_kind_name(v), "");
    /* CPYTHON'S OWN TWO REFUSALS, which are not one: the LENGTH is named by
       the value that was handed over -- `'NoneType' object cannot be
       interpreted as an integer`, the wording every index-taking slot uses
       -- and the BYTEORDER by the parameter, the arg clinic's form. */
    if (!apy_is_int_like(length))
        return apy_fail2("TypeError",
                         "'%s' object cannot be interpreted as an integer%s",
                         apy_kind_name(length), "");
    if (O(order)->kind != APY_STR_K)
        return apy_fail2("TypeError",
                         "to_bytes() argument 'byteorder' must be str, not "
                         "%s%s",
                         O(order)->kind == APY_NONE_K
                             ? "None" : apy_kind_name(order), "");
    n = O(length)->v.i;
    if (n < 0)
        return apy_fail("ValueError", "length argument must be non-negative");
    if (n > 1024)
        return apy_fail("OverflowError", "int too big to convert");
    big = !(O(order)->kind == APY_STR_K
            && strcmp(APY_CSTR(order), "little") == 0);
    want_signed = apy_truth(signed_);
    neg = apy_int_is_neg(v);
    if (neg && !want_signed)
        return apy_fail("OverflowError",
                        "can't convert negative int to unsigned");
    used = apy_int_mag_len(v);
    if (!apy_to_bytes_fits(v, n, used, neg, want_signed))
        return apy_fail("OverflowError", "int too big to convert");
    buf = (char *)calloc((size_t)(n ? n : 1) + 1, 1);
    if (!buf) { fputs("uasm: out of memory\n", stderr); exit(1); }
    /* LITTLE-ENDIAN FIRST, ALWAYS. The two's complement below carries upward
       from the least significant byte, which is only simple in this order. */
    for (i = 0; i < n; i++)
        buf[i] = (char)apy_int_mag_byte(v, i);
    if (neg) {
        int carry = 1;
        for (i = 0; i < n; i++) {
            int b = (255 - (unsigned char)buf[i]) + carry;
            carry = b > 255;
            buf[i] = (char)(b & 0xFF);
        }
    }
    if (big) {
        int64_t a = 0, z = n - 1;
        while (a < z) {
            char t = buf[a]; buf[a] = buf[z]; buf[z] = t;
            a++; z--;
        }
    }
    {
        return apy_bytes_take(buf, n);
    }
}

/* `x.as_integer_ratio()` -- the EXACT fraction the double holds, in lowest
   terms. `0.1` is not one tenth, and this is the method that says so. */
APY_API apy_value apy_as_integer_ratio(apy_value v) {
    double d;
    int64_t num, den = 1;
    apy_value out;
    if (apy_is_int_like(v)) {
        out = apy_tuple_new(2);
        apy_seq_push(out, v);
        apy_seq_push(out, apy_from_int(1));
        return out;
    }
    if (O(v)->kind != APY_FLOAT_K)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute "
                         "'as_integer_ratio'%s", apy_kind_name(v), "");
    d = O(v)->v.f;
    if (d != d || d - d != 0.0)
        return apy_fail("OverflowError",
                        "cannot convert Infinity to integer ratio");
    while (d != floor(d) && den < (int64_t)1 << 60) { d *= 2.0; den *= 2; }
    num = (int64_t)d;
    out = apy_tuple_new(2);
    apy_seq_push(out, apy_from_int(num));
    apy_seq_push(out, apy_from_int(den));
    return out;
}

/* `s.expandtabs(n)` -- tabs to the next multiple of `n`, counting from the
   last newline. Not a fixed number of spaces per tab: the whole point is that
   columns line up. */
APY_API apy_value apy_str_expandtabs(apy_value s, apy_value width) {
    int64_t n, i, col = 0, out = 0, cap, w;
    /* A COLUMN IS A CHARACTER IN A STR AND A BYTE IN BYTES, which is the only
       thing the bytes receiver changes: `b"\xc3\xa9\tx"` is two columns
       before the tab where the str it decodes to is one. */
    int wide;
    char *buf;
    if (!apy_str_self("expandtabs", s)) return 0;
    wide = O(s)->kind == APY_STR_K;
    /* THE WIDTH IS REQUIRED AND MUST BE AN INTEGER. It arrives always -- the
       frontend supplies 8 for the no-argument form -- so a value that is not
       an integer is one the PROGRAM wrote, and `"a\tb".expandtabs(None)` is a
       TypeError in Python. Reading a non-integer as the default made the
       written None answer what the omitted argument answers, which is the
       difference no padding scheme can see once it has been applied. */
    if (!apy_is_int_like(width))
        return apy_fail2("TypeError",
                         "'%s' object cannot be interpreted as an integer%s",
                         apy_kind_name(width), "");
    w = apy_is_big(width) ? 8 : O(width)->v.i;
    if (w < 1) w = 1;
    n = O(s)->v.s.n;
    /* NO TAB, NOTHING TO EXPAND -- and a str then IS its own answer where
       bytes builds the copy anyway. That asymmetry is CPython's and not a
       slip to be tidied away: `unicode_expandtabs` ends its measuring pass
       with `if (!found_tabs) return unicode_result_unchanged(self)`, and the
       stringlib `expandtabs` that bytes and bytearray share has no such
       test and returns what it built. So `"abc".expandtabs() is "abc"` is
       True and `b"abc".expandtabs() is b"abc"` is False, which is why this
       is gated on `wide` rather than on `apy_str_may_return_self`.

       ONLY A TAB DECIDES IT. A newline resets the column and changes
       nothing else, so a string full of them still answers itself. */
    if (wide) {
        for (i = 0; i < n; i++) if (APY_CSTR(s)[i] == '\t') break;
        if (i == n) return s;
    }
    cap = n * (w > 1 ? w : 1) + 8;
    buf = (char *)malloc((size_t)cap + 1);
    if (!buf) { fputs("uasm: out of memory\n", stderr); exit(1); }
    for (i = 0; i < n; i++) {
        char c = APY_CSTR(s)[i];
        if (c == '\t') {
            int64_t pad = w - (col % w);
            while (pad-- > 0 && out < cap) { buf[out++] = ' '; col++; }
        } else {
            if (out < cap) buf[out++] = c;
            /* A COLUMN IS A CHARACTER AND NOT A BYTE. A continuation byte
               is `10xxxxxx` and adds nothing; counting it made `"é\tx"`
               reach column 2 after one character and the tab stop land one
               place early. */
            col = (c == '\n' || c == '\r') ? 0
                : (wide && (c & 0xC0) == 0x80 ? col : col + 1);
        }
    }
    buf[out] = 0;
    return apy_str_take(buf, out);
}

/* `x.is_integer()` -- a float method, and true for an int too, because
   `(5).is_integer()` is True in Python 3.12 and later. */
APY_API apy_value apy_is_integer(apy_value v) {
    if (apy_is_int_like(v) || apy_is_big(v)) return apy_from_bool(1);
    if (O(v)->kind != APY_FLOAT_K)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'is_integer'%s",
                         apy_kind_name(v), "");
    return apy_from_bool(O(v)->v.f == floor(O(v)->v.f)
                         && O(v)->v.f - O(v)->v.f == 0.0);
}

/* `z.conjugate()`. Defined on the whole numeric tower, not only on complex:
   `(5).conjugate()` is 5, which is what makes it usable without a kind test. */
APY_API apy_value apy_conjugate(apy_value v) {
    if (O(v)->kind == APY_COMPLEX_K)
        return apy_from_complex(O(v)->v.z.re, -O(v)->v.z.im);
    /* A BOOL ANSWERS AN int, not itself: `True.conjugate()` is `1`,
       because `bool` inherits the method from `int` and the method is
       defined to answer an int. */
    if (O(v)->kind == APY_BOOL_K) return apy_from_int(O(v)->v.i);
    if (apy_is_int_like(v) || apy_is_big(v) || O(v)->kind == APY_FLOAT_K)
        return v;
    return apy_fail2("AttributeError",
                     "'%s' object has no attribute 'conjugate'%s",
                     apy_kind_name(v), "");
}

"""
