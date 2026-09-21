"""The dynamic object runtime: what a Python value IS at run time.

The Python frontend began as an annotated subset -- `int`, `float`, `bool`,
`None`, every parameter annotated, every expression's type known at compile
time. That is a real language and it compiles well, and it is not Python. A
Python file has unannotated functions, strings, lists, and values whose type is
decided by what flows into them, and none of that can be expressed by a static
type per expression.

So there are two representations, and the boundary between them is the single
most important thing in this file to understand:

  * A function whose parameters and return are ALL annotated stays on the
    static path. Its `int` is a machine word, its `float` an xmm register, and
    nothing here is involved.
  * Everything else -- the module's top-level statements, any function with an
    unannotated parameter -- is DYNAMIC. Every value is an `apy_value`, every
    operation is a call into this file, and the value's type is a field it
    carries rather than a fact the compiler knows.

WHY UNIFORM BOXING, and not NaN-boxing or tagged pointers, which are both
faster: the conformance suite's TAXONOMY.md names "the representation follows
the declared type of the slot the value is stored in, rather than the value"
as the dominant defect of the compiler this replaces -- one root cause that
surfaces as a dozen unrelated-looking bugs, because the symptom depends only on
how the result is read. A single representation for every value, with the type
inside it, makes that failure mode unreachable rather than unlikely. This
compiler is measured on agreeing with CPython, not on speed.

THE CELL. Every value is a pointer to one `struct apy_obj`. `NULL` is never a
valid value, so a null return is unambiguously "an error was set". `None`,
`True` and `False` are single shared cells: a program comparing `x is None`
compares pointers, and three statics cost nothing.

MEMORY IS NEVER FREED. Every constructor mallocs and nothing collects. That is
a real limitation and it is stated rather than left to be discovered: a program
that loops a million string concatenations will grow without bound. It is not a
correctness problem for anything the conformance suite runs, and a collector
needs a stack map the IR does not carry yet.

ERRORS are a sticky flag, not a longjmp. An operation that fails sets it and
returns NULL; the frontend checks it where it needs to. That keeps the policy
question -- how does an exception propagate, what does `try` do -- in the
frontend where it belongs, instead of this file inventing an answer that
`try`/`except` would then have to be built around. Until the frontend grows
real exception handling it calls `apy_fatal_if_error`, which writes
`TypeError: ...` to STDERR and exits 1. Deliberately not stdout: the suite
diffs stdout, and a traceback printed there turns a correctly-failing program
into a wrong answer.

THE FIRST ERROR WINS. `apy_fail` does not overwrite a flag that is already
set, so a frontend that lets a null value flow into a second operation still
reports the ORIGINAL failure rather than whatever the second one made of a
null. Every operation that can fail returns 0 (never a valid value) when it
does, so "did this fail" is answerable without the flag as well.

INTEGERS ARE ARBITRARY PRECISION. This entry used to head the list below --
"64-bit and wrap, `2 ** 64` is 0, the largest single divergence in the file"
-- and the second integer kind the `kind` field was left room for is now
there. A big is NEVER a value that fits an int64: every result demotes, so
each integer has exactly one representation and nothing downstream has to
maintain agreement between two of them. See the arbitrary-precision section.

WHAT IS DELIBERATELY NOT RIGHT YET, stated here rather than left to be found:

  * `len` counts characters, but INDEXING AND SLICING WOULD COUNT BYTES.
    Neither exists in v1; when they arrive they need the same UTF-8 walk
    `apy_str_chars` does.
  * A NEGATIVE BASE WITH A FRACTIONAL EXPONENT is a complex number in Python
    and there is no complex kind here, so `apy_pow` reports a ValueError
    instead of answering. It is the one place this file knowingly raises
    where CPython returns a value.
  * A SET ITERATES IN INSERTION ORDER and CPython's in hash-table slot order,
    so `print(set([3, 1, 2]))` differs. Reproducing CPython's would mean
    reproducing its table growth, its probe sequence and a str hash that is
    salted per process -- there is no fixed answer to match. See the set
    section for the full argument.
  * STRING CASE AND CLASSIFICATION ARE ASCII, and every method that returns a
    POSITION returns a byte offset. `'ß'.upper()` is 'SS' in CPython and 'ß'
    here. Both follow from the two entries above about `len` and indexing.

Everything else in here is differentially tested against CPython by
tools/objects_diff.py, which compiles this source with a C driver and diffs
it case by case. That tool is the reason to trust the sign rules and the
float formatting; the counts it reports are in its docstring.
"""
from __future__ import annotations

#: Storage class for every function the IR can call. Empty for the linked
#: runtime, `static` for the C backend, whose output is one self-contained
#: translation unit. One macro rather than a marker on sixty definitions --
#: the C is then byte-identical in both builds, so they cannot drift.
_API_TOKEN = "@APY_API@"

#: The generated Unicode class table, spliced into the C at its marker.
#: Sixty kilobytes of data, and generated, so it is neither written out
#: here nor concatenated with the parts -- it goes in at `@UNICODE_TABLE@`.
from .c.unicode_table import UNICODE_C

#: The generated Unicode CASE mappings, spliced in the same way and at
#: its own marker. A separate table from the classes because it answers
#: a different question: not what a character IS but what it BECOMES.
from .c.unicase_table import UNICASE_C

#: WHICH BUILTIN METHODS A KIND HAS, and which entry point serves each
#: arity. Generated from CPython's own `hasattr` and from the frontend's
#: `DYN_METHOD_TABLE`, so the written form and the looked-up form are ONE
#: implementation rather than two that can drift.
from .c.kindmeth_table import KINDMETH_C

#: THE RUNTIME'S C, whole. It was written out longhand right here, all
#: sixteen thousand lines of it, which made this module unopenable in an
#: editor and unreviewable in a diff. It is nineteen modules under `c/`
#: now, concatenated in source order; see `c/__init__.py` for why the
#: order is the whole contract. Nothing below can tell the difference --
#: every function here reads one string, exactly as it always did.
from .c import OBJECTS_C



#: Every symbol the IR may call. The frontend declares these as imports and
#: the link stage decides whether to pull the runtime in by looking at what is
#: declared, so a name missing here is a link error rather than a wrong answer.
OBJECT_NAMES = (
    #: Callables, classes and instances. `apy_call` is the only one of these
    #: that a program reaches for an ordinary call -- the rest build the values
    #: it dispatches on.
    "apy_cell_new", "apy_cell_get", "apy_cell_set",
    "apy_func_new", "apy_func_cell", "apy_func_default", "apy_env_cell",
    "apy_func_kwdefaults",
    "apy_call",
    "apy_type_new", "apy_type_set", "apy_instance_new",
    "apy_type_builtin", "apy_type_qual",
    "apy_getattr", "apy_getattr_default", "apy_setattr",
    "apy_super", "apy_type_object", "apy_type_class",
    "apy_prepare", "apy_type_make", "apy_object_class",
    "apy_class_build", "apy_class_build_kw", "apy_meta_for",
    "apy_mro_entries",
    "apy_check_slots",
    "apy_name_or", "apy_print_seq_with", "apy_unpack_check",
    "apy_call_spread_kw",
    #: `typing`, which is inert at run time -- a special form is an object a
    #: program names and never inspects, and the decorators mark and return.
    "apy_typing_form", "apy_typing_final", "apy_typing_override",
    "apy_type_alias", "apy_typevar", "apy_typevar_default",
    "apy_import",
    "apy_get_origin", "apy_get_args",
    "apy_typing_mark",
    #: asyncio. A coroutine is a generator, so these drive the same frame the
    #: generator entry points do.
    "apy_coro_mark", "apy_await_step", "apy_asyncio_run", "apy_asyncio_sleep",
    "apy_asyncio_gather", "apy_agen_mark", "apy_agen_step",
    "apy_suspend_value", "apy_func_coro",
    #: `inspect`, the three questions a program asks it about coroutines.
    "apy_inspect_iscoroutine", "apy_inspect_isgenerator",
    "apy_inspect_isasyncgen", "apy_inspect_iscoroutinefunction",
    "apy_is_instance", "apy_exc_register",
    "apy_method_is_builtin", "apy_method_self",
    "apy_range", "apy_sorted", "apy_min", "apy_max", "apy_sum", "apy_reversed", "apy_enumerate", "apy_zip2", "apy_abs", "apy_index_obj", "apy_round", "apy_isinstance", "apy_slice", "apy_list_pop", "apy_index_of", "apy_count_of", "apy_list_remove", "apy_dict_parts", "apy_dict_get_or",
    "apy_list_new", "apy_tuple_new", "apy_seq_push", "apy_getitem",
    "apy_dict_new", "apy_dict_set", "apy_key_at",
    "apy_setitem", "apy_raw_len",
    "apy_none", "apy_from_bool", "apy_from_int", "apy_from_float",
    "apy_from_cstr", "apy_from_bytes", "apy_as_int", "apy_as_float", "apy_as_bool",
    "apy_type_name", "apy_truth", "apy_len", "apy_repr", "apy_str",
    "apy_print",
    "apy_add", "apy_sub", "apy_mul", "apy_truediv", "apy_floordiv",
    "apy_mod", "apy_pow", "apy_neg", "apy_pos", "apy_invert",
    "apy_bitand", "apy_bitor", "apy_bitxor", "apy_lshift", "apy_rshift",
    "apy_eq", "apy_ne", "apy_is", "apy_contains", "apy_lt", "apy_le", "apy_gt", "apy_ge",
    "apy_to_int", "apy_to_float", "apy_to_bool",
    "apy_error_occurred", "apy_error_type", "apy_error_message",
    "apy_error_clear", "apy_fatal_if_error",
    "apy_make_exc", "apy_make_excn", "apy_raise", "apy_error_matches",
    "apy_error_value",
    # set and frozenset
    "apy_set_new", "apy_frozenset_new", "apy_set_push",
    "apy_to_set", "apy_to_frozenset",
    "apy_set_add", "apy_set_discard",
    "apy_set_union", "apy_set_intersection", "apy_set_difference",
    "apy_set_symdiff",
    "apy_set_inter_update", "apy_set_diff_update",
    "apy_set_symdiff_update",
    "apy_str_format_map", "apy_bytearray_resize",
    "apy_bytearray_fromhex",
    "apy_float_from_number", "apy_complex_from_number",
    "apy_any_fromhex",
    "apy_mview_live",
    "apy_mview_release", "apy_mview_readonly",
    "apy_mview_item", "apy_mview_cast",
    "apy_kind_method_var",
    "apy_meth_arity",
    "apy_bytes_hex_n",
    "apy_kw_put", "apy_kw_merge",
    "apy_mview_from_flags",
    "apy_set_issubset", "apy_set_issuperset", "apy_set_isdisjoint",
    "apy_update", "apy_clear", "apy_copy", "apy_hash",
    # str methods
    "apy_str_upper", "apy_str_lower", "apy_str_title", "apy_str_capitalize",
    "apy_str_swapcase", "apy_str_casefold",
    "apy_str_isalpha", "apy_str_isdigit", "apy_str_isdecimal",
    "apy_str_isnumeric", "apy_str_isalnum", "apy_str_isspace",
    "apy_str_islower", "apy_str_isupper", "apy_str_istitle",
    "apy_str_isprintable", "apy_str_isidentifier", "apy_str_isascii",
    "apy_str_strip", "apy_str_lstrip", "apy_str_rstrip",
    "apy_str_strip_chars", "apy_str_lstrip_chars", "apy_str_rstrip_chars",
    "apy_str_removeprefix", "apy_str_removesuffix",
    "apy_str_split_ws", "apy_str_split", "apy_str_split_n",
    "apy_str_rsplit_ws", "apy_str_rsplit", "apy_str_rsplit_n",
    "apy_str_splitlines", "apy_str_splitlines_keep",
    "apy_str_partition", "apy_str_rpartition", "apy_str_join",
    "apy_str_replace", "apy_str_replace_n",
    "apy_str_startswith", "apy_str_startswith2", "apy_str_startswith3",
    "apy_str_endswith", "apy_str_endswith2", "apy_str_endswith3",
    "apy_str_find", "apy_str_find2", "apy_str_find3",
    "apy_str_rfind", "apy_str_rfind2", "apy_str_rfind3", "apy_str_rindex",
    "apy_str_count2", "apy_str_count3",
    "apy_str_maketrans", "apy_str_translate", "apy_str_like",
    # The CONSTRUCTOR spellings of `.encode()` and `.decode()`.
    "apy_bytes_ctor", "apy_str_ctor",
    "apy_bytes_maketrans", "apy_bytes_translate", "apy_translate_kw",
    "apy_pop_or", "apy_dict_popitem",
    #: `match`: the predicates a `case` pattern needs and nothing else does.
    "apy_match_seq", "apy_match_map", "apy_match_args", "apy_match_rest",
    #: `slice` as an object, for a user `__getitem__` and for `c[1:2, 3]`.
    "apy_slice_new", "apy_slice_indices", "apy_matmul", "apy_alias_new",
    "apy_func_is_type", "apy_func_annotate", "apy_func_qualname",
    "apy_func_module", "apy_func_descr", "apy_func_builtin",
    "apy_ascii", "apy_notimplemented", "apy_id",
    "apy_hex_of", "apy_float_fromhex",
    #: `d.keys()` and friends -- a window on the dict, not a copy.
    "apy_dict_view", "apy_view_items",
    #: `async with` -- each answers a COROUTINE, which the caller awaits.
    "apy_aenter", "apy_aexit", "apy_dir",
    #: PEP 654 -- several exceptions raised as one.
    "apy_excgroup_new", "apy_group_split", "apy_group_subgroup",
    "apy_group_dispatch",
    "apy_template_new", "apy_interpolation_new", "apy_exc_class_bind",
    "apy_ns_get", "apy_aiter", "apy_asyncio_create_task",
    "apy_pos_add", "apy_at", "apy_code_of",
    "apy_task_cancel", "apy_task_result", "apy_task_done",
    "apy_task_cancelled", "apy_asyncio_wait_for",
    "apy_asyncio_taskgroup",
    "apy_init_subclass",
    "apy_split_of",
    #: descriptors: `property`, `classmethod`, `staticmethod`.
    "apy_descr_new", "apy_prop_setter", "apy_prop_getter",
    "apy_prop_deleter", "apy_set_names",
    "apy_str_ljust", "apy_str_ljust_fill", "apy_str_rjust",
    "apy_str_rjust_fill", "apy_str_center", "apy_str_center_fill",
    "apy_str_zfill",
    # arbitrary precision integers
    "apy_pow3", "apy_bit_length", "apy_bit_count",
    "apy_bin", "apy_oct", "apy_hex", "apy_to_int_base", "apy_divmod",
    "apy_ctor_call", "apy_kw_check", "apy_kw_none", "apy_kw_owner",
    "apy_sizeof",
)


#: C type -> the IR type a value of it travels in. `apy_value` is the IR's
#: `ptr`, which is what makes the object runtime's arguments and results
#: opaque to every backend.
_IR_TYPES = {"apy_value": "ptr", "int64_t": "i64", "double": "f64",
             "void": "void",
             # Every pointer is one machine word, whatever it points at.
             # `_param_type` folds `const char *` and `char *` to this.
             "ptr": "ptr"}


#: THE C A BACKEND ACTUALLY SEES: the runtime with the generated Unicode
#: table spliced in at its marker. One file, as it always was -- the split is
#: only so this module stays readable.
_WITH_TABLE = OBJECTS_C.replace("/* @UNICODE_TABLE@ */", UNICODE_C) \
                       .replace("/* @UNICASE_TABLE@ */", UNICASE_C) \
                       .replace("/* @KINDMETH_TABLE@ */", KINDMETH_C)


def _param_type(text: str) -> str:
    """The TYPE half of one C parameter, as a key for `_IR_TYPES`.

    A POINTER PARAMETER IS ONE IR WORD AND ITS SPELLING IS NOT ONE TOKEN. In
    C's grammar the `*` binds to the NAME, so `const char *p` splits on its
    last space into `const char` -- a type this table has never heard of, and
    the failure is an AssertionError naming a type nobody wrote.

    That mattered the moment `str` began moving: `apy_lit`, `apy_str_take`,
    `apy_str_copy` and `apy_str_copy`'s bytes twin are all `static`, so
    `_definition_of` cannot find them and the ported runtime cannot replace
    them -- and simply promoting them to `APY_API` broke this parser instead.
    The alternative was retyping four signatures to `apy_value` and casting at
    something like two hundred call sites, which is a lot of edits to avoid
    teaching one function that a star is a pointer.
    """
    text = text.strip()
    if "*" in text:
        return "ptr"
    return text.rsplit(" ", 1)[0].strip() if " " in text else text


def signatures() -> dict:
    """Every `APY_API` symbol, as (argument type names, result type name).

    READ OUT OF THE C rather than listed beside it. The frontend has to declare
    each of these as an import with the right signature, and a hand-kept list
    drifted three times in one afternoon -- each time the same way: a symbol
    added here, not declared there, and the failure arriving as `call to
    unknown function` from the IR verifier or as a link error, neither of which
    names the list that was not updated.

    The C is generated from a string in this module, so its shape is known: one
    `APY_API` per definition, arguments comma-separated, no function pointers
    and no structs by value.
    """
    import re
    out = {}
    for m in re.finditer(r"APY_API\s+([\w ]+?)\s+(apy_\w+)\(([^)]*)\)\s*\{",
                         _WITH_TABLE):
        ret, name, raw = m.group(1).strip(), m.group(2), m.group(3).strip()
        if name in out:
            continue
        args = [] if raw in ("void", "") else [
            _param_type(a) for a in raw.split(",")]
        try:
            out[name] = ([_IR_TYPES[a] for a in args], _IR_TYPES[_param_type(ret)])
        except KeyError as exc:
            raise AssertionError(
                f"{name}: no IR type for {exc.args[0]!r}. Add it to _IR_TYPES "
                f"-- a runtime symbol the frontend cannot describe is one it "
                f"cannot call.") from None
    return out


def split_c(text: str, names: tuple[str, ...], suffix: str = "_slow") -> str:
    """Rename a function's DEFINITION, leaving its name to something else.

    THE SHAPE EVERY PORTED KIND TAKES. `apy_add` is polymorphic over eighteen
    kinds, so it cannot be ported whole without porting all of them -- but its
    integer case is four instructions and is most of what a program does. So
    the subset defines `apy_add`, handles the case it knows, and hands
    everything else to what used to be here:

        APY_API apy_value apy_add(apy_value, apy_value);        <- IR
        APY_API apy_value apy_add_slow(apy_value a, apy_value b) { ...the C... }

    Only the DEFINITION is renamed. Every existing call inside the runtime --
    and there are many -- keeps saying `apy_add` and so reaches the fast path,
    which is the point: the port has to speed up the C's own work, not only
    the frontend's calls.

    Recursion still works and still means the right thing: a body that calls
    `apy_add` reaches the fast path, which falls back here if it must.
    """
    for name in names:
        at, line_start = _definition_of(text, name)
        open_brace = text.index("{", at)
        signature = text[line_start:open_brace].rstrip()
        # The declaration for the IR half goes ABOVE, so every later caller
        # has one in scope no matter where it sits.
        renamed = signature.replace(f" {name}(", f" {name}{suffix}(", 1)
        text = (text[:line_start] + signature + ";\n" + renamed
                + text[open_brace:])
    return text


def _definition_of(text: str, name: str) -> tuple[int, int]:
    """Where `name`'s APY_API definition starts: (name offset, line start)."""
    at = -1
    while True:
        at = text.find(f" {name}(", at + 1)
        if at < 0:
            raise KeyError(f"{name} is not defined in the object runtime")
        line_start = text.rfind("\n", 0, at) + 1
        # `APY_API`, the macro, is what marks a PUBLIC definition -- the token
        # `@APY_API@` is only its expansion at the `#define`.
        if text.startswith("APY_API ", line_start):
            semicolon = text.find(";", at)
            brace = text.find("{", at)
            if brace >= 0 and (semicolon < 0 or brace < semicolon):
                return at, line_start
    raise KeyError(f"{name} is declared but not defined")


def objects_c(*, static: bool = False, omit: tuple[str, ...] = (),
              split: tuple[str, ...] = ()) -> str:
    """The object runtime's C source, with its storage class chosen.

    `static` for the C backend, whose output is one self-contained translation
    unit; external for the linked runtime, where the IR's calls have to resolve
    across object files.

    `omit` names functions that are DEFINED IN IR INSTEAD -- the ported half of
    `docs/INERT-RUNTIME.md`. Keeping both definitions is a duplicate symbol at
    link time, so the C one gives way; see `_declare_only`.

    `split` names functions the IR takes the FRONT of: the C body is renamed
    `<name>_slow` and the IR defines `<name>`, which handles what it knows and
    calls back for the rest. See `split_c`.
    """
    text = _WITH_TABLE
    if split:
        text = split_c(text, split)
    if omit:
        text = _declare_only(text, omit)
    return text.replace(_API_TOKEN, "static" if static else "")


def _declare_only(text: str, names: tuple[str, ...]) -> str:
    """Replace each named function's DEFINITION with a declaration.

    Not a deletion. Fifteen thousand lines of C call these, and C wants a
    declaration in scope before a call -- so removing `apy_from_int` outright
    makes every caller an implicit declaration returning `int`, which on this
    ABI reads the wrong register and produces a plausible wrong number rather
    than an error.

    Brace-matched from the opening `{` rather than pattern-matched to the end,
    because the bodies are ordinary C with nested blocks and strings. Anything
    it cannot find is raised rather than skipped: a name silently not omitted
    is a duplicate symbol at LINK time, which names an object file instead of
    this list.
    """
    for name in names:
        # THROUGH `_definition_of`, which walks PAST a forward declaration.
        # `apy_cell_new` has one -- it must, because the `apy_alloc` wrapper
        # above it calls it -- and the older search here stopped at the first
        # `APY_API` line bearing the name and reported "declared but not
        # defined; nothing to omit" about a function defined thirty lines down.
        at, _ = _definition_of(text, name)
        open_brace = text.index("{", at)
        depth, i = 0, open_brace
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            raise KeyError(f"{name}'s body is not brace-balanced")
        text = (text[:open_brace] + ";    /* defined in IR: runtime/ */"
                + text[i + 1:])
    return text
