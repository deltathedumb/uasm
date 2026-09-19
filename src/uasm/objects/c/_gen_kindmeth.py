"""Generate the builtin-method table, as C and as the machine subset.

Beside the tables it writes rather than in `tools/`, for the reason
`_gen_unicode.py` gives: a generated table nobody can regenerate is a table
nobody can check.

    python -m uasm.objects.c._gen_kindmeth

WHAT IT IS FOR. `"abc".upper` is lowered at the CALL SITE by the frontend, so
it exists as a call and never as an attribute -- and
`getattr("abc", "upper")` was an AttributeError about a method the object
plainly has. Anything dispatching by method NAME saw a builtin as having
none: `operator.methodcaller`, a plugin table, `getattr(x, name)(...)`.

TWO QUESTIONS AND TWO ANSWERS, and both are generated because both are
already written down somewhere else:

  WHICH KINDS OWN A NAME is CPython's own `hasattr`, asked over the twelve
  builtin types at generation time. Transcribing it by hand is how a table
  comes to claim that a list has `keys`.

  WHICH SYMBOL SERVES AN ARITY is `DYN_METHOD_TABLE`, which the frontend
  already uses to lower the written form. Reading it here is what makes the
  two spellings ONE implementation rather than two that can drift.

THE ARITY IS THE UNION over the owners, deliberately. `str.count` takes a
window and `list.count` does not, and the table has one entry per NAME -- so
the native admits the widest form and the symbol it reaches refuses a
receiver it cannot serve, exactly as the written form already does.

SEVEN CALL SHAPES ARE IRREGULAR and are mirrored from `_dyn_builtin_method`:
`keys`/`values`/`items` take an int selector, `pop` under two arguments takes
a had-index flag, `encode`/`decode` always take three, `hex`/`expandtabs`
with none take a filled second, `get`/`setdefault` with one take a None
default, `sort` takes its two keywords as values, and `update` takes a dict.
"""
import pathlib
import sys

#: The twelve builtin types a receiver can be, and the bit each gets. The
#: NAMES are what `apy_kind_attr_of` already computes for the receiver in
#: hand; the bits are private to the generated pair.
KINDS = ["str", "bytes", "bytearray", "list", "tuple", "dict", "set",
         "frozenset", "int", "float", "range", "complex", "memoryview"]

#: A view is in `KINDS` for the WRITTEN table alone -- which is about what
#: `type()` calls a bound method -- and not for the ARITY table, whose entries
#: say which runtime symbol serves a name. Its methods are hand-written in
#: `apy_kind_attr_of`, because several of them mean something a bytes receiver
#: would answer differently for.
NOT_IN_THE_METHOD_TABLE = {"memoryview"}

#: One value of each kind, to ask CPython about. Written out rather than
#: built from the name, because `range` and `complex` take arguments and an
#: empty one of either would answer differently about nothing.
SAMPLES = {"str": '""', "bytes": 'b""', "bytearray": "bytearray()",
           "list": "[]", "tuple": "()", "dict": "{}", "set": "set()",
           "frozenset": "frozenset()", "int": "5", "float": "1.5",
           "range": "range(3)", "complex": "1j",
           "memoryview": 'memoryview(b"ab")'}

#: Names the generated table must NOT claim, because something above it
#: answers them first and differently. `index` and `count` reach
#: `apy_index_of`/`apy_count_of`, which split on the receiver themselves;
#: the rest are answered by hand in `apy_kind_attr_of` for kinds the table
#: cannot see (a range's `start`, a generator's `send`).
SKIP = {"send", "throw", "close", "setter", "getter", "indices", "subgroup",
        "add_note", "fromhex"}


#: A name whose real arity range is NOT the one `DYN_METHOD_TABLE` shows.
#: `to_bytes` has ONE entry point taking all three parameters, because two of
#: them have defaults and `signed` is keyword-only -- so the table shows 3..3
#: where Python accepts nought to three, and the lowering pads. The dispatch
#: below pads the same way; see `call_shape`.
ARITY_OVERRIDE = {"to_bytes": (0, 3)}


def owners():
    """Every non-dunder method name, with the kinds that own it and the
    arities `DYN_METHOD_TABLE` serves."""
    from uasm.frontends.python.methods import DYN_METHOD_TABLE
    types = {"str": str, "bytes": bytes, "bytearray": bytearray,
             "list": list, "tuple": tuple, "dict": dict, "set": set,
             "frozenset": frozenset, "int": int, "float": float,
             "range": range, "complex": complex,
             "memoryview": memoryview}
    out = []
    for name in sorted(DYN_METHOD_TABLE):
        if name.startswith("__") or name in SKIP:
            continue
        kinds = [k for k in KINDS if k not in NOT_IN_THE_METHOD_TABLE
                 and hasattr(types[k], name)]
        if not kinds:
            continue
        syms = DYN_METHOD_TABLE[name]
        live = [i for i, s in enumerate(syms) if s is not None]
        if not live:
            continue
        lo, hi = ARITY_OVERRIDE.get(name, (min(live), max(live)))
        out.append((name, kinds, lo, hi,
                    [syms[i] if i < len(syms) else None
                     for i in range(max(live) + 1)]))
    return out


def mask(kinds) -> int:
    return sum(1 << KINDS.index(k) for k in kinds)


def emit_c(rows) -> str:
    """The C half: one arity lookup and one dispatch."""
    lines = [
        "/* WHICH BUILTIN METHODS A KIND HAS, and which runtime entry point",
        "   serves each arity.",
        "",
        "   GENERATED by `objects/c/_gen_kindmeth.py` from CPython's own",
        "   `hasattr` and from the frontend's `DYN_METHOD_TABLE`. See that",
        f"   file for why. {len(rows)} names.",
        "",
        "   THE BITS MATCH `KINDS` there and are private to this pair. */",
    ]
    for i, k in enumerate(KINDS):
        lines.append(f"#define APY_MK_{k.upper()} {1 << i}u")
    lines += [
        "",
        "/* The arity of `w` on a receiver of kind `bit`, packed as",
        "   `(most << 8) | nopt`, or 0 for a name that kind does not have. */",
        "static int64_t apy_kind_meth_arity(const char *w, unsigned bit) {",
    ]
    for name, kinds, lo, hi, _ in rows:
        lines.append(f'    if (strcmp(w, "{name}") == 0)')
        lines.append(f"        return (bit & {mask(kinds)}u) "
                     f"? (({hi + 1} << 8) | {hi - lo}) : 0;")
    lines += ["    return 0;", "}", ""]

    lines += emit_signatures(rows)
    lines += emit_written()

    lines += [
        "/* One builtin method call, by name and by the count that arrived.",
        "   The irregular shapes are mirrored from `_dyn_builtin_method`; see",
        "   the generator. Answers 0 having raised for an arity no entry",
        "   point serves, which is how a receiver's own kind refuses a form",
        "   another kind admits. */",
        "static apy_value apy_kind_meth_call(const char *w, apy_value *a,",
        "                                    int64_t n) {",
    ]
    for name, kinds, lo, hi, syms in rows:
        lines.append(f'    if (strcmp(w, "{name}") == 0) {{')
        for argc in range(hi, lo - 1, -1):
            sym = syms[argc] if argc < len(syms) else None
            if name in ARITY_OVERRIDE:
                sym = syms[hi]
            if sym is None:
                continue
            lines.append(f"        if (n >= {argc + 1}) return "
                         + call_shape(name, sym, argc) + ";")
        lines.append("    }")
    lines += [
        '    return apy_fail2("TypeError", "\'%s\' is not a builtin method%s",',
        '                     w, "");',
        "}",
    ]
    return "\n".join(lines) + "\n"


def call_shape(name, sym, argc) -> str:
    """The arguments one entry point takes, which is not always `a[0..argc]`.

    SEVEN OF THEM ARE IRREGULAR and every one is mirrored from the frontend's
    `_dyn_builtin_method`: the two spellings have to reach the same call.
    """
    if name in ("keys", "values", "items"):
        part = {"keys": 0, "values": 1, "items": 2}[name]
        return f"apy_dict_parts(a[0], {part})"
    if name == "pop" and argc < 2:
        if argc == 0:
            return f"{sym}(a[0], apy_none(), 0)"
        return f"{sym}(a[0], a[1], 1)"
    if name in ("encode", "decode"):
        # THE DEFAULTS THEMSELVES, NOT None. `"a".encode(None)` is a
        # TypeError in CPython and `"a".encode()` is `"a".encode("utf-8")`,
        # and once a slot has been padded nothing downstream can tell an
        # omission from a written value -- so the slot carries the default's
        # own text and the runtime refuses a None. The lowering pads the
        # written spelling the same way.
        pad = ['apy_lit("utf-8")', 'apy_lit("strict")']
        args = ", ".join(["a[0]"] + [f"a[{i + 1}]" if i < argc else pad[i]
                                     for i in range(2)])
        return f"{sym}({args})"
    if name in ("hex", "expandtabs") and argc == 0:
        filled = "apy_from_int(8)" if name == "expandtabs" else "apy_none()"
        return f"{sym}(a[0], {filled})"
    if name in ("get", "setdefault") and argc == 1:
        return f"{sym}(a[0], a[1], apy_none())"
    if name == "sort":
        # PADDED FROM THE DEFAULTS, so the no-argument form and the folded
        # two-keyword form reach one entry point -- the same arrangement
        # `to_bytes` has below.
        pad = ["apy_none()", "apy_from_bool(0)"]
        args = ["a[0]"] + [f"a[{i + 1}]" if i < argc else pad[i]
                           for i in range(2)]
        return f"{sym}({', '.join(args)})"
    if name == "update" and argc == 0:
        return f"{sym}(a[0], apy_dict_new(1))"
    if name == "to_bytes":
        # ONE ENTRY POINT TAKING ALL THREE, padded from the defaults
        # `METHOD_PARAMS` records -- which is what the lowering does for the
        # written form, so the two spellings reach the same call.
        pad = ["apy_from_int(1)", 'apy_lit("big")', "apy_from_bool(0)"]
        args = ["a[0]"] + [f"a[{i + 1}]" if i < argc else pad[i]
                           for i in range(3)]
        return f"{sym}({', '.join(args)})"
    return f"{sym}({', '.join(['a[0]'] + [f'a[{i + 1}]' for i in range(argc)])})"


def c_value(default) -> str:
    """One parameter default as the C that builds it."""
    if default is True or default is False:
        return f"apy_from_bool({1 if default else 0})"
    if default is None:
        return "apy_none()"
    if isinstance(default, int):
        return f"apy_from_int({default})"
    if isinstance(default, str):
        return f'apy_lit("{default}")'
    if default == b"":
        return 'apy_bytes_literal((apy_value)"", 0)'
    raise AssertionError(f"no C for the default {default!r}")


def emit_signatures(rows) -> list:
    """`apy_kind_meth_sign` -- the parameter NAMES a keyword may use.

    THE SAME TABLE THE WRITTEN FORM FOLDS AGAINST. `METHOD_PARAMS` says what
    each of these methods calls its parameters and what each defaults to, and
    the frontend arranges a written `x.split(",", maxsplit=1)` with it. A
    method reached as a VALUE -- `f = x.split` then `f(",", maxsplit=1)`, or
    the collision path handing an int's `to_bytes` to `apy_call_kw` -- has no
    frontend to arrange it, so the names travel on the callable itself and
    the ordinary keyword binder places them. Without this the bound method
    reported `got an unexpected keyword argument` for a call CPython answers.

    A PARAMETER CPYTHON MARKS POSITIONAL-ONLY HAS A NULL NAME, which is what
    keeps it unreachable by keyword while still leaving its slot in place.
    """
    from uasm.frontends.python.methods import (  # noqa: E402
        METHOD_PARAMS, POSITIONAL_ONLY, REQUIRED)
    arity = {name: hi + 1 for name, _, _, hi, _ in rows}
    lines = [
        "/* THE PARAMETER NAMES A KEYWORD MAY USE, and the value each slot",
        "   defaults to, for the methods CPython gives a keyword signature.",
        "",
        "   GENERATED from `METHOD_PARAMS`, which is the same table the",
        "   frontend folds the WRITTEN spelling against -- so `x.split(\",\",",
        "   maxsplit=1)` and `f = x.split` then `f(\",\", maxsplit=1)` are one",
        "   arrangement rather than two that can drift.",
        "",
        "   Answers the count, or 0 for a name with no keyword signature at",
        "   all. `names[i]` is 0 for a parameter CPython marks positional",
        "   only; `defs[i]` is 0 for one with no default. */",
        "static int64_t apy_kind_meth_sign(const char *w, apy_value *names,",
        "                                  apy_value *defs) {",
    ]
    for name, params in sorted(METHOD_PARAMS.items()):
        # ONLY WHERE THE TWO TABLES AGREE ON THE COUNT. A name whose dispatch
        # arity does not match its signature would put a keyword in the wrong
        # slot, which is worse than not placing it at all.
        if arity.get(name) != len(params) + 1:
            continue
        lines.append(f'    if (strcmp(w, "{name}") == 0) {{')
        for i, (pname, default) in enumerate(params):
            spelled = ("0" if pname is POSITIONAL_ONLY
                       else f'apy_lit("{pname}")')
            filled = "0" if default is REQUIRED else c_value(default)
            lines.append(f"        names[{i}] = {spelled}; "
                         f"defs[{i}] = {filled};")
        lines.append(f"        return {len(params)};")
        lines.append("    }")
    lines += ["    return 0;", "}", ""]
    return lines


def written_rows():
    """Which (kind, dunder) pairs CPython calls a `builtin_function_or_method`.

    THE REAL DISTINCTION IS NOT ARITY. A bound slot is a `method-wrapper` and
    a method the type WRITES OUT is a `builtin_function_or_method`, and
    nothing about the signature says which: `list.__getitem__` takes exactly
    one argument and is written out, while `tuple.__getitem__` fills the slot.
    The arity was standing in for it and got twenty-seven pairs wrong.

    ASKED OF CPYTHON, per kind and per name, because that is where the answer
    is -- and `__contains__` and `__getitem__` really do differ BY KIND, so a
    table keyed by name alone could not hold it.
    """
    out = {}
    for i, kind in enumerate(KINDS):
        try:
            sample = eval(SAMPLES[kind])
        except Exception:                        # pragma: no cover
            continue
        for name in dir(sample):
            if not name.startswith("__"):
                continue
            try:
                got = getattr(sample, name)
            except AttributeError:               # pragma: no cover
                continue
            if type(got).__name__ != "builtin_function_or_method":
                continue
            out.setdefault(name, 0)
            out[name] |= 1 << i
    return sorted(out.items())


def emit_written() -> list:
    """`apy_kind_meth_written` -- the C half of the table above."""
    lines = [
        "/* Does CPython WRITE this dunder out for a receiver of kind `bit`,",
        "   rather than filling a slot with it? That is the whole of what",
        "   tells `builtin_function_or_method` from `method-wrapper`, and it",
        "   is not derivable from the signature -- see `written_rows` in the",
        "   generator. */",
        "static int64_t apy_kind_meth_written(const char *w, unsigned bit) {",
    ]
    for name, mask_bits in written_rows():
        lines.append(f'    if (strcmp(w, "{name}") == 0)')
        lines.append(f"        return (bit & {mask_bits}u) ? 1 : 0;")
    lines += ["    return 0;", "}", ""]
    return lines


def emit_ir(rows) -> str:
    """The ported half: the arity lookup only.

    THE DISPATCH STAYS IN C. `apy_call_nk` is `static` there and serves both
    compiled runtimes, so the call never reaches the subset; only
    `apy_kind_attr_of` does, and all it needs is the arity.

    THE `_of` SUFFIX IS THE TREE'S CONVENTION for an IR twin of a C name: the
    two halves share one translation unit, so a subset function wearing a C
    static's name is a redefinition from gcc rather than a second copy.
    """
    lines = [
        '"""Which builtin methods a kind has, in the machine subset.',
        "",
        "GENERATED by `objects/c/_gen_kindmeth.py`. Its own module for the",
        "reason `unicode_table.py` is: it is a table, and the runtime is",
        "meant to be read.",
        '"""',
        "",
        "",
        "def apy_kind_meth_arity_of(w: ptr, bit: i64) -> i64:",
        '    """The arity of `w` on a receiver of kind `bit`, packed as',
        "    `(most << 8) | nopt`, or 0 for a name that kind does not have.",
        '    """',
    ]
    for name, kinds, lo, hi, _ in rows:
        lines.append(f'    if apy_cstr_eq(w, rodata(b"{name}\\0")):')
        lines.append(f"        if bit & {mask(kinds)}:")
        lines.append(f"            return {((hi + 1) << 8) | (hi - lo)}")
        lines.append("        return 0")
    lines.append("    return 0")
    lines += [
        "",
        "",
        "def apy_kind_meth_written_of(w: ptr, bit: i64) -> i64:",
        '    """Does CPython WRITE this dunder out for a receiver of kind',
        "    `bit`, rather than filling a slot with it? That is the whole of",
        "    what tells `builtin_function_or_method` from `method-wrapper`.",
        '    """',
    ]
    for name, mask_bits in written_rows():
        lines.append(f'    if apy_cstr_eq(w, rodata(b"{name}\\0")):')
        lines.append(f"        if bit & {mask_bits}:")
        lines.append("            return 1")
        lines.append("        return 0")
    lines.append("    return 0")
    return "\n".join(lines) + "\n"


def emit_dir_ir() -> str:
    """The ported half of `apy_kind_dir`.

    THE `_of` SUFFIX for the reason `emit_ir` gives: the C and the subset are
    one translation unit, so a subset function wearing the C static's name is
    a redefinition rather than a second copy.
    """
    lines = [
        "",
        "",
        "def apy_kind_dir_of(kind: ptr) -> ptr:",
        '    """Every name `dir(x)` answers for a value of this kind, by the',
        "    type's name: NUL-separated, ended by an empty name.",
        '    """',
    ]
    for kind, names in dir_rows():
        blob = "".join(n + "\\0" for n in names) + "\\0"
        lines.append(f'    if apy_cstr_eq(kind, rodata(b"{kind}\\0")):')
        lines.append(f'        return rodata(b"{blob}")')
    lines.append("    return ptr(0)")
    return "\n".join(lines) + "\n"


#: ONE VALUE OF EVERY KIND A CURSOR CAN BE, keyed by the name
#: `apy_kind_name` gives it -- `apy_cursor_name` for a plain or reversed
#: walk, the mode for the four lazy ones, and `v.g` for a generator.
#:
#: WRITTEN OUT AND NOT DERIVED, because the name is the interesting part:
#: `iter("")` is a `str_ascii_iterator` and `iter("\u00e9")` a `str_iterator`,
#: and only a sample with the right contents lands on the right one. The
#: table is held to that by `cursor_types`, which asks CPython what each
#: sample actually is and refuses to generate anything if one is misfiled.
#:
#: `iterator` IS DELIBERATELY ABSENT. It is this compiler's fallback name for
#: a cursor whose source it no longer knows, and CPython has no type by that
#: name to ask -- so there is nothing to transcribe and `dir()` over one
#: stays as empty as it was.
#:
#: `generator` IS HERE, and was not until the seven names that made a lying
#: list out of an honest one could all answer: `gi_code`, `gi_frame`,
#: `gi_running`, `gi_suspended`, `gi_yieldfrom`, `__del__` and
#: `__class_getitem__`. A generator is not a cursor -- it is a frame -- but
#: `dir()` and `__doc__` ask the same question of it, so it is filed here
#: with the rest of them.
#:
#: AND SO ARE ITS TWO SIBLINGS, which are the same cell in this runtime and
#: three different types in CPython: a coroutine names the same five facts
#: `cr_`, an async generator names them `ag_`, and each has its own three
#: methods. The samples are built by a helper because a coroutine cannot be
#: written as an expression -- and an unawaited one warns at collection, so
#: the sample is closed before it is dropped.
CURSOR_SAMPLES = {
    "generator":                 "(_ for _ in ())",
    "coroutine":                 "_sample_coroutine()",
    "async_generator":           "_sample_async_generator()",
    "coroutine_wrapper":         "_sample_coroutine_wrapper()",
    "list_iterator":             "iter([])",
    "list_reverseiterator":      "reversed([])",
    "tuple_iterator":            "iter(())",
    "reversed":                  "reversed(())",
    "str_iterator":              'iter("\u00e9")',
    "str_ascii_iterator":        'iter("")',
    "bytes_iterator":            'iter(b"")',
    "bytearray_iterator":        "iter(bytearray())",
    "range_iterator":            "iter(range(0))",
    "set_iterator":              "iter(set())",
    "dict_keyiterator":          "iter({}.keys())",
    "dict_valueiterator":        "iter({}.values())",
    "dict_itemiterator":         "iter({}.items())",
    "dict_reversekeyiterator":   "reversed({}.keys())",
    "dict_reversevalueiterator": "reversed({}.values())",
    "dict_reverseitemiterator":  "reversed({}.items())",
    "callable_iterator":         "iter(lambda: None, None)",
    "memory_iterator":           'iter(memoryview(b"a"))',
    "enumerate":                 "enumerate([])",
    "zip":                       "zip()",
    "map":                       "map(str, [])",
    "filter":                    "filter(None, [])",
}


def _sample_coroutine():
    """One coroutine, closed rather than awaited.

    `dir()` and `__doc__` are asked of the TYPE, so the object only has to
    exist -- and an unawaited coroutine that is collected prints a
    RuntimeWarning, which a generator step this file never takes would be a
    strange thing to emit while generating a table.
    """
    async def one():
        return None

    made = one()
    made.close()
    return made


def _sample_coroutine_wrapper():
    """What `c.__await__()` answers -- a second object, and not the
    coroutine. Closed rather than awaited, for the reason
    `_sample_coroutine` gives."""
    async def one():
        return None

    made = one()
    wrapped = made.__await__()
    made.close()
    return wrapped


def _sample_async_generator():
    """One async generator. Nothing has to drive it: see
    `_sample_coroutine`."""
    async def one():
        yield None

    return one()


def cursor_types() -> dict:
    """The TYPE of each sample, checked against the name it is filed under.

    A SAMPLE UNDER THE WRONG NAME would transcribe one type's `dir()` as
    another's, which is the one mistake this table can make and the one a
    reader cannot see. CPython is asked rather than trusted.
    """
    out = {}
    for name, expr in CURSOR_SAMPLES.items():
        got = type(eval(expr))
        if got.__name__ != name:
            raise SystemExit(
                f"{expr} is a {got.__name__}, filed under {name}")
        out[name] = got
    return out


#: The kinds a VALUE can be whose type carries a docstring. Wider than
#: `KINDS`, because `True.__doc__`, `None.__doc__` and a view's are the same
#: question and none of the three is in the method table.
#:
#: AND EVERY CURSOR, because `iter([]).__doc__` is a question too. Most of
#: them answer None, which is a row this table does not carry -- see
#: `apy_kind_doc`'s caller for how "a kind we model, with no docstring" is
#: told from "a kind we do not model".
DOC_KINDS = KINDS + ["bool", "NoneType"] + list(CURSOR_SAMPLES)


def kind_types() -> dict:
    """Every kind this file asks CPython about, by the name the runtime gives
    it. The builtin types longhand, and the cursors through their samples --
    one map, because `__doc__` and `dir()` are the same question asked twice
    and two copies of it would drift."""
    types = {"str": str, "bytes": bytes, "bytearray": bytearray, "list": list,
             "tuple": tuple, "dict": dict, "set": set, "frozenset": frozenset,
             "int": int, "float": float, "range": range, "complex": complex,
             "bool": bool, "NoneType": type(None), "memoryview": memoryview}
    types.update(cursor_types())
    return types


def c_string(text: str) -> str:
    """One C string literal per line of `text`, concatenated.

    SPLIT ON NEWLINES because a docstring is paragraphs and a single 600-byte
    literal on one line is unreadable in the generated file -- and adjacent
    literals are one string in C, so nothing about the result changes.
    """
    out = []
    for i, line in enumerate(text.split("\n")):
        body = (line.replace("\\", "\\\\").replace('"', '\\"')
                    .replace("\t", "\\t"))
        out.append(f'"{body}' + ('\\n"' if i < len(text.split("\n")) - 1
                                 else '"'))
    return out


def emit_doc() -> str:
    """`apy_kind_doc` -- what `str.__doc__` and `"".__doc__` answer.

    ASKED OF CPython, which is where the answer is: a type's docstring is
    text CPython ships and not a fact this compiler could derive. Every
    builtin value has one and reading it was an AttributeError about the
    attribute `help` is built on.
    """
    types = kind_types()
    lines = [
        "/* THE DOCSTRING OF THE TYPE A VALUE IS, by that type's name.",
        "",
        "   GENERATED by `objects/c/_gen_kindmeth.py` from CPython's own",
        "   `__doc__`, which is where the answer is: the text is CPython's",
        "   and not a fact this compiler could derive. Answers 0 for a kind",
        "   with none, which the caller reads as \"not this\". */",
        "static const char *apy_kind_doc(const char *kind) {",
    ]
    for name in DOC_KINDS:
        doc = types[name].__doc__
        if not doc:
            continue
        lines.append(f'    if (strcmp(kind, "{name}") == 0)')
        parts = c_string(doc)
        lines.append("        return " + parts[0])
        for part in parts[1:]:
            lines.append("               " + part)
        lines[-1] += ";"
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines) + "\n"


#: The kinds a VALUE can be whose names `dir()` should list. The same fifteen
#: `DOC_KINDS` covers, and for the same reason: `dir(True)` and `dir(None)`
#: are the question `dir(5)` is.
DIR_KINDS = DOC_KINDS


def dir_rows():
    """Every name `dir(x)` answers for a value of each kind, from CPython.

    ASKED OF CPYTHON rather than derived, because `dir()` is defined as what
    the type and its bases carry and this compiler has no such chain for a
    builtin -- the method table lives in the frontend and the dunder gates
    are strcmp chains, neither of which can be walked.

    THE LIST IS ONLY HONEST IF EVERY NAME ON IT ANSWERS. A `dir()` that
    advertises a name `getattr` then refuses is worse than the empty list it
    replaces, so the two are checked against each other -- on all four paths
    -- by the `dir_lists_every_name_getattr_answers` program in
    `test_dynamic_python.py`.
    """
    types = kind_types()
    return [(k, sorted(dir(types[k]))) for k in DIR_KINDS]


def name_block(names) -> list:
    r"""The names as ONE C string literal, NUL-separated, ended by an empty
    one -- so the reader walks it with `strlen` and stops on the empty name.

    A NAME NEVER STARTS WITH A DIGIT, which is what makes `"\0name"` safe:
    `\0` would swallow up to three octal digits and there are none to
    swallow.
    """
    out, line = [], ""
    for name in names:
        piece = f'"{name}\\0"'
        if len(line) + len(piece) > 66:
            out.append(line)
            line = ""
        line += piece
    if line:
        out.append(line)
    out.append('""')
    return out


def emit_dir() -> str:
    """`apy_kind_dir` -- what `dir(x)` answers for a builtin value."""
    lines = [
        "/* EVERY NAME `dir(x)` ANSWERS for a value of this kind, by the",
        "   type's name: NUL-separated, ended by an empty name.",
        "",
        "   GENERATED by `objects/c/_gen_kindmeth.py` from CPython's own",
        "   `dir()`. A builtin has no class chain here to walk -- the method",
        "   table is in the frontend and the dunder gates are strcmp chains",
        "   -- so `dir(5)` answered an EMPTY LIST where CPython lists",
        "   seventy names. Answers 0 for a kind with no list, which the",
        "   caller reads as \"nothing to add\". */",
        "static const char *apy_kind_dir(const char *kind) {",
    ]
    for kind, names in dir_rows():
        lines.append(f'    if (strcmp(kind, "{kind}") == 0)')
        parts = name_block(names)
        lines.append("        return " + parts[0])
        for part in parts[1:]:
            lines.append("               " + part)
        lines[-1] += ";"
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines) + "\n"


#: HOW CPYTHON WORDS A WRONG-ARITY CALL, one family per shape. Nine shapes
#: across 198 (name, kind) pairs, read out of CPython rather than guessed:
#: the differences between them are every visible thing -- whether the type
#: qualifies the name, whether the parentheses are there, whether the count
#: is called "at most" or "exactly" -- and a test tells them apart.
#:
#: THE NUMBERS ARE THE WIRE FORMAT. They are read by `apy_meth_arity_words`
#: in `_calling.py` and by its ported twin, and both switch on them, so a
#: number changing here is a number changing in three places.
FAMILIES = {
    # `list.append() takes exactly one argument (2 given)`
    "QUAL_EXACT": 1,
    # `int.bit_count() takes no arguments (1 given)`
    "QUAL_NONE": 2,
    # `encode() takes at most 2 arguments (3 given)`
    "ATMOST": 3,
    # `f() takes at most 2 positional arguments (3 given)`
    "ATMOST_POS": 4,
    # `replace() takes at least 2 positional arguments (1 given)`
    "ATLEAST_POS": 5,
    # `count expected at least 1 argument, got 0`
    "VA_ATLEAST": 6,
    # `lstrip expected at most 1 argument, got 2`
    "VA_ATMOST": 7,
    # `insert expected 2 arguments, got 1`
    "VA_EXACT": 8,
    # `sort() takes no positional arguments` -- NO COUNT AT ALL, which is
    # what a signature with only keyword-only parameters says.
    "NO_POS": 9,
}

#: The message shapes, as regular expressions over CPython's own text. The
#: captured group is the COUNT the message names, which is not always the
#: count that was accepted -- `int.to_bytes` says "at most 3" while refusing
#: a third positional -- so it is stored rather than derived.
_WORDINGS = [
    (r"^\w+\.\w+\(\) takes exactly one argument \(\d+ given\)$",
     "QUAL_EXACT", 1),
    (r"^\w+\.\w+\(\) takes no arguments \(\d+ given\)$", "QUAL_NONE", 0),
    (r"^\w+\(\) takes at most (\d+) positional arguments? \(\d+ given\)$",
     "ATMOST_POS", None),
    (r"^\w+\(\) takes at least (\d+) positional arguments? \(\d+ given\)$",
     "ATLEAST_POS", None),
    (r"^\w+\(\) takes at most (\d+) arguments? \(\d+ given\)$",
     "ATMOST", None),
    (r"^\w+ expected at least (\d+) arguments?, got \d+$", "VA_ATLEAST", None),
    (r"^\w+ expected at most (\d+) arguments?, got \d+$", "VA_ATMOST", None),
    (r"^\w+ expected (\d+) arguments?, got \d+$", "VA_EXACT", None),
    (r"^\w+\(\) takes no positional arguments$", "NO_POS", 0),
]

#: What a count of 15 means in the packed word: no upper bound at all.
#: `set.union` takes any number, and a four-bit field has to say so somehow.
UNBOUNDED = 15


def _wording(text: str):
    """`(family, count)` for one of CPython's refusals, or None for anything
    that is not about the number of arguments."""
    import re
    for pattern, family, fixed in _WORDINGS:
        hit = re.match(pattern, text)
        if hit:
            return family, fixed if fixed is not None else int(hit.group(1))
    return None


def _ask(sample, name, argc):
    """`(accepted, wording)` for calling `name` on `sample` with `argc`
    arguments -- `None` for the arguments, because what they ARE is not the
    question and a TypeError about a VALUE is an accepted count."""
    try:
        getattr(sample, name)(*([None] * argc))
        return True, None
    except TypeError as exc:
        said = _wording(str(exc))
        return (True, None) if said is None else (False, said)
    except Exception:
        return True, None


def word_rows():
    """Every `(name, kind)` pair's accepted range and its two refusals.

    ASKED OF CPYTHON BY CALLING IT, which is the only place the answer is:
    the range and the wording disagree for `int.to_bytes`, the wording
    differs BY KIND for seven names (`list.count` is qualified where
    `str.count` is not), and neither fact is derivable from a signature.
    """
    out = {}
    for name in sorted(DYN_METHOD_TABLE_NAMES()):
        if name.startswith("__") or name in SKIP:
            continue
        for kind in KINDS:
            try:
                sample = eval(SAMPLES[kind])
            except Exception:                        # pragma: no cover
                continue
            if not hasattr(sample, name):
                continue
            taken = [n for n in range(9) if _ask(sample, name, n)[0]]
            if not taken:                            # pragma: no cover
                continue
            least, most = min(taken), max(taken)
            low = _ask(sample, name, least - 1)[1] if least else None
            high = _ask(sample, name, most + 1)[1] if most < 8 else None
            # A SECOND UPPER WORDING. `int.to_bytes` says `takes at most 2
            # positional arguments` for a third POSITIONAL and `takes at
            # most 3 arguments` for a fourth argument of any kind -- its
            # `signed` is keyword-only, so the two bounds differ and so do
            # the sentences. Found by walking up until the wording changes
            # rather than special-cased, because a name that grows a
            # keyword-only parameter would need the same.
            over = None
            for more in range(most + 2, 9):
                said = _ask(sample, name, more)[1]
                if said is not None and said != high:
                    over = (more, said)
                    break
            out[(name, kind)] = (least, UNBOUNDED if most >= 8 else most,
                                 low, high, over)
    return out


def DYN_METHOD_TABLE_NAMES():
    from uasm.frontends.python.methods import DYN_METHOD_TABLE
    return DYN_METHOD_TABLE


def pack_word(row) -> int:
    """One `(least, most, low, high)` row as a single number.

    FOUR BITS EACH and a present bit between the two halves, because the
    reader is C and neither it nor a reader in the machine subset has a
    struct to hand back.
    """
    least, most, low, high, over = row
    lof, lon = (FAMILIES[low[0]], low[1]) if low else (0, 0)
    hif, hin = (FAMILIES[high[0]], high[1]) if high else (0, 0)
    at, (ovf, ovn) = ((over[0], (FAMILIES[over[1][0]], over[1][1]))
                      if over else (0, (0, 0)))
    return (least | (most << 4) | (lof << 8) | (lon << 12)
            | (hif << 16) | (hin << 20) | (1 << 24)
            | (at << 25) | (ovf << 29) | (ovn << 33))


def word_groups():
    """`[(name, [(mask, packed), ...])]` -- the rows folded by kind.

    ONE ARM PER DISTINCT ROW, so the seven names whose wording depends on the
    receiver get two arms and the other ninety-odd get one.
    """
    rows = word_rows()
    byname = {}
    for (name, kind), row in rows.items():
        byname.setdefault(name, {}).setdefault(row, []).append(kind)
    out = []
    for name in sorted(byname):
        arms = [(mask(kinds), pack_word(row))
                for row, kinds in sorted(byname[name].items(),
                                         key=lambda kv: sorted(kv[1]))]
        out.append((name, arms))
    return out


def emit_words_ir() -> str:
    """The ported half of `apy_kind_meth_words`.

    `apy_kind_attr_of` IS IR-REPLACED, so the subset builds the native for a
    builtin method on the `--object-runtime ir` path and needs this kind's
    real BOUNDS to declare it with -- see `apy_kind_method_ranged_of`. The
    wording above it stays in the C, which both compiled runtimes share.
    """
    lines = [
        "",
        "",
        "def apy_kind_meth_words_of(w: ptr, bit: i64) -> i64:",
        '    """What CPython says about a wrong number of arguments to this',
        "    builtin method on a receiver of kind `bit`, packed. See",
        "    `apy_kind_meth_words` in the C half.",
        '    """',
    ]
    for name, arms in word_groups():
        lines.append(f'    if apy_cstr_eq(w, rodata(b"{name}\\0")):')
        for mask_bits, packed in arms:
            lines.append(f"        if bit & {mask_bits}:")
            lines.append(f"            return {packed}")
        lines.append("        return 0")
    lines.append("    return 0")
    return "\n".join(lines) + "\n"


def emit_words() -> str:
    """`apy_kind_meth_words` -- the packed row for a `(name, kind)` pair."""
    lines = [
        "/* WHAT CPYTHON SAYS ABOUT A WRONG NUMBER OF ARGUMENTS to this",
        "   builtin method on a receiver of kind `bit`, packed: the accepted",
        "   range in the low byte and the two refusal wordings above it.",
        "   Zero for a name this kind does not have.",
        "",
        "   GENERATED by `objects/c/_gen_kindmeth.py`, which CALLS CPython to",
        "   find out. Neither half is derivable: the range and the wording",
        "   disagree for `int.to_bytes`, and seven names word it differently",
        "   depending on the receiver -- `list.count() takes exactly one",
        "   argument` against `count expected at most 3 arguments`. */",
        "static int64_t apy_kind_meth_words(const char *w, unsigned bit) {",
    ]
    for name, arms in word_groups():
        lines.append(f'    if (strcmp(w, "{name}") == 0) {{')
        for mask_bits, packed in arms:
            lines.append(f"        if (bit & {mask_bits}u) return {packed};")
        lines.append("        return 0;")
        lines.append("    }")
    lines += ["    return 0;", "}", ""]
    return "\n".join(lines) + "\n"


def emit_words_py() -> str:
    """The same rows as a Python dict, for the INTERPRETER.

    The host cannot read a C static, and a hand-kept second copy of a table
    is the thing that has drifted in this tree three times -- so the one
    generator writes both halves and `objects_host.py` imports this.
    """
    rows = word_rows()
    lines = [
        "#: The same rows as `apy_kind_meth_words`, for the interpreter:",
        "#: `(method name, type name)` to the packed word. See `emit_words`.",
        "KINDMETH_WORDS = {",
    ]
    for (name, kind) in sorted(rows):
        lines.append(f'    ("{name}", "{kind}"): {pack_word(rows[(name, kind)])},')
    lines += ["}", ""]
    return "\n".join(lines) + "\n"


def emit_written_cursor() -> str:
    """`apy_cursor_meth_written` -- is this name WRITTEN OUT on a cursor's
    type, or does the type fill a slot with it?

    ONE ROW SERVES EVERY CURSOR, which is measured rather than assumed: the
    two names a cursor adds of its own are the same two on all of them
    (`__length_hint__` and `__setstate__`, both written out), and everything
    else it has comes from `object` -- so `object`'s own answer is every
    cursor's. `__iter__` and `__next__` fill slots, which is why a cursor's
    are `slot wrapper`s where its `__length_hint__` is a `method`.

    THE ORDINARY TABLE CANNOT ANSWER THIS. `apy_kind_meth_written` is keyed
    by the thirteen builtin kinds' BITS and a cursor has no bit, so every
    name read off one came back "slotted" -- and `type(it).__length_hint__`
    printed as a slot wrapper where CPython prints a method.
    """
    it = iter([])
    names = sorted({n for n in dir(it)
                    if type(type(it).__dict__.get(n,
                                                  object.__dict__.get(n)))
                    .__name__ == "method_descriptor"})
    lines = [
        "/* Is this name WRITTEN OUT on a cursor's type, or a filled slot?",
        "",
        "   GENERATED by `objects/c/_gen_kindmeth.py` from CPython's own",
        "   descriptors. One row serves every cursor: the names a cursor",
        "   adds of its own are the same on all of them, and the rest come",
        "   from `object`. See `emit_written_cursor` for why the bit-keyed",
        "   table cannot answer it. */",
        "static int apy_cursor_meth_written(const char *w) {",
    ]
    for name in names:
        lines.append(f'    if (strcmp(w, "{name}") == 0) return 1;')
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def emit_tables_py() -> str:
    """`KIND_DIR` and `KIND_DOC`, for the host.

    THE THIRD READER OF THE SAME TABLE. The C reads `apy_kind_dir`, the
    subset reads `apy_kind_dir_of`, and the interpreter used to ask live
    CPython -- which worked for the fifteen builtin types, whose values it
    represents with real Python ones, and could not work for a cursor: its
    cursor is an object of this compiler's own and `dir()` over one says so.
    Reading the generated pair here is what keeps the three answers one
    answer.
    """
    lines = ["", "",
             "# THE THREE SAMPLES THAT ARE NOT EXPRESSIONS. A coroutine, its",
             "# await wrapper and an async generator cannot be written",
             "# inline, so the table's",
             "# entries for them name these -- and every reader that evals",
             "# the table needs them in scope. Copied out of",
             "# `objects/c/_gen_kindmeth.py`, which is where they are",
             "# written; see `_sample_coroutine` there for why the coroutine",
             "# is closed rather than awaited.",
             "",
             "",
             "def _sample_coroutine():",
             "    async def one():",
             "        return None",
             "",
             "    made = one()",
             "    made.close()",
             "    return made",
             "",
             "",
             "def _sample_coroutine_wrapper():",
             "    async def one():",
             "        return None",
             "",
             "    made = one()",
             "    wrapped = made.__await__()",
             "    made.close()",
             "    return wrapped",
             "",
             "",
             "def _sample_async_generator():",
             "    async def one():",
             "        yield None",
             "",
             "    return one()",
             "", "",
             "#: The sample expression behind every cursor row, so the host",
             "#: can ask CPython about the same types this file asked.",
             "CURSOR_SAMPLES = {"]
    for kind, expr in CURSOR_SAMPLES.items():
        lines.append(f"    {kind!r}: {expr!r},")
    lines.append("}")
    lines += ["", "", "#: What `dir(x)` answers for a value of each kind.",
              "KIND_DIR = {"]
    for kind, names in dir_rows():
        lines.append(f"    {kind!r}: {names!r},")
    lines.append("}")
    lines += ["", "", "#: The docstring of the type a value is, or None.",
              "KIND_DOC = {"]
    types = kind_types()
    for kind in DOC_KINDS:
        lines.append(f"    {kind!r}: {types[kind].__doc__!r},")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent.parent.parent))
    rows = owners()
    (here / "kindmeth_table.py").write_text(
        '"""The builtin-method table, as C.\n\n'
        "GENERATED by `objects/c/_gen_kindmeth.py`. Its own module for the\n"
        "reason `unicode_table.py` is: it is a table, and `objects.py` is\n"
        'meant to be read.\n"""\n\n'
        'KINDMETH_C = r"""' + emit_c(rows) + emit_doc() + emit_dir()
        + emit_written_cursor() + emit_words() + '"""\n\n'
        + emit_words_py() + emit_tables_py(),
        encoding="utf-8")
    (here.parent.parent / "runtime" / "kindmeth_table.py").write_text(
        emit_ir(rows) + emit_dir_ir() + emit_words_ir(),
        encoding="utf-8")
    print(f"{len(rows)} names")


if __name__ == "__main__":
    main()
