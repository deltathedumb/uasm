"""What `obj.method(...)` lowers to.

Its own module because it is a TABLE, not logic, and because both analysis and
lowering read it -- analysis to decide whether a call has a shape it can
accept, lowering to pick the symbol. Two copies of it would drift, and the way
they would drift is a method that type-checks and then fails to link.

There is no bound-method value here: `xs.append` on its own is not something
this can produce, only `xs.append(v)` as a whole. Method lookup on an instance
will need that to change.
"""
from __future__ import annotations


#: `obj.method(...)` -> the runtime symbol, indexed by ARGUMENT COUNT.
#:
#: A list per method, position `k` holding the symbol for `k` arguments and
#: `None` where that arity does not exist. Written out rather than derived from
#: the symbol names, even though the runtime's naming is regular
#: (`apy_str_find` / `find2` / `find3`), because the irregular ones are exactly
#: the ones worth being explicit about: `split()` with no argument splits on
#: RUNS of whitespace and `split(' ')` does not, so they are different
#: functions, not the same function with a default.
#:
#: A method missing from here is reported by analysis with its name. A method
#: present here whose symbol the runtime does not define is a link error, which
#: is why `_check_methods_exist` asserts the two agree at import.
#: THE BUILTIN METHODS THAT TAKE NO RECEIVER, by kind and by name.
#:
#: `dict.keys` is an unbound INSTANCE method: `dict.keys(d)` writes the
#: receiver out, and `str.upper(x)` is the same shape. `str.maketrans` is a
#: STATICMETHOD and `dict.fromkeys` a CLASSMETHOD, and neither takes one --
#: so both the written call and the read-as-a-value form ate the first
#: argument:
#:
#:     m = str.maketrans; m("ab", "xy")
#:     TypeError: if you give only one argument to maketrans it must be a dict
#:
#: with `"ab"` swallowed as a receiver that is not there.
#:
#: EVERY ROW READ OFF CPYTHON rather than guessed: it is every name in a
#: builtin type's `__dict__` whose value is a `staticmethod` or a
#: `classmethod_descriptor`. `__class_getitem__` is on almost all of them and
#: is tested separately by each reader, because it belongs to no particular
#: kind and because it binds the TYPE where these bind the prototype.
#:
#: KEYED BY THE PAIR AND NOT BY THE NAME, because a name alone does not say:
#: `fromhex` is a classmethod on bytes and another on float, and nothing
#: stops a later kind from having an instance method of that name.
#:
#: HERE BECAUSE THREE READERS NEED IT -- the lowering, to know not to move
#: the first argument into the receiver slot; the interpreter, to bind the
#: prototype rather than the first argument; and the same question again in
#: the two compiled runtimes, where it is written out in C
#: (`apy_kind_static`) and in the machine subset (`apy_kind_static_of`)
#: because neither can read this. Those two must agree with this one.
KIND_STATIC = frozenset({
    ("str", "maketrans"),
    ("bytes", "maketrans"), ("bytes", "fromhex"),
    ("bytearray", "maketrans"), ("bytearray", "fromhex"),
    ("dict", "fromkeys"),
    ("int", "from_bytes"),
    ("float", "from_number"), ("float", "fromhex"),
    ("float", "__getformat__"),
    ("complex", "from_number"),
    ("memoryview", "_from_flags"),
    ("type", "__prepare__"),
})


DYN_METHOD_TABLE = {
    # sequences
    "append":       [None, "apy_seq_push"],
    "add":          [None, "apy_set_add"],
    "discard":      [None, "apy_set_discard"],
    # `index` TAKES THE SAME BOUNDS `find` DOES, on a str, bytes, a list and
    # a tuple alike -- `xs.index(v, start)` is how a scan resumes past the
    # last hit. A range has no bounded form; it falls through to the
    # sequence refusal.
    "index":        [None, "apy_index_of", "apy_index_of2", "apy_index_of3"],
    # `count` on a LIST takes only the item; the start/end forms are str's
    # alone, which is why the wider arities name the string entry points
    # directly. A list reaching them is not a shape Python allows.
    "count":        [None, "apy_count_of", "apy_str_count2",
                     "apy_str_count3"],
    "remove":       [None, "apy_list_remove"],
    "insert":       [None, None, "apy_list_insert"],
    # `apy_extend_meth` AND NOT `apy_extend`: a written `.extend` asks the
    # argument for a length hint before it drains -- a list and a bytearray
    # both do. The bare drain is shared with `f(*xs)` and with the temporary
    # `set(x)` collects into, neither of which asks. See the C's
    # `apy_extend_meth`.
    "extend":       [None, "apy_extend_meth"],
    "reverse":      ["apy_list_reverse"],
    # 3.14's way of saying how long a bytearray is to be. Growing fills with
    # NUL and shrinking truncates, which is what makes it a buffer a program
    # can hand out and then fill.
    "resize":       [None, "apy_bytearray_resize"],
    "clear":        ["apy_clear"],
    "copy":         ["apy_copy"],
    # `sort` is IN PLACE and answers None; `sorted` is the other one. Both
    # arities take the key and reverse VALUES, which lowering supplies as
    # None/False when the call omitted them -- see `_dyn_builtin_method`.
    # TWO ENTRIES FOR ONE SYMBOL. `apy_list_sort` takes all three slots and
    # the lowering pads the ones a call left out, so the no-argument form and
    # the folded two-keyword form reach the same place -- which is what lets
    # `getattr(xs, "sort")(reverse=True)` arrive as a call the dispatch
    # recognises.
    "sort":         ["apy_list_sort", None, "apy_list_sort"],
    # dict. All three are `apy_dict_parts` with a selector, so the table
    # records the shape and lowering supplies the constant -- see
    # `DICT_PARTS`.
    "keys":         ["apy_dict_parts"],
    "values":       ["apy_dict_parts"],
    "items":        ["apy_dict_parts"],
    "get":          [None, "apy_dict_get_or", "apy_dict_get_or"],
    "setdefault":   [None, "apy_setdefault", "apy_setdefault"],
    # `d.update()` with no argument is a no-op, and `d.update(k=v)` is
    # all-keywords -- so the zero-argument shape is real. See
    # `_dyn_builtin_method`, which supplies the keyword dict.
    "update":       ["apy_update", "apy_update"],
    # `pop` reaches a LIST, a SET or a DICT, and the receiver is not known
    # until run time -- so the one-argument form dispatches inside
    # `apy_list_pop`, and the two-argument form is a dict's alone.
    "pop":          ["apy_list_pop", "apy_list_pop", "apy_pop_or"],
    "popitem":      ["apy_dict_popitem"],
    # set algebra, as methods
    "union":            [None, "apy_set_union"],
    "intersection":     [None, "apy_set_intersection"],
    "difference":       [None, "apy_set_difference"],
    "symmetric_difference": [None, "apy_set_symdiff"],
    # The same three IN PLACE, which a frozenset does not have: there is
    # nothing to update. `s.difference_update(t)` is `s -= t` by another
    # name, and like the algebra above it takes any iterable.
    "intersection_update": [None, "apy_set_inter_update"],
    "difference_update":   [None, "apy_set_diff_update"],
    "symmetric_difference_update": [None, "apy_set_symdiff_update"],
    "issubset":     [None, "apy_set_issubset"],
    "issuperset":   [None, "apy_set_issuperset"],
    "isdisjoint":   [None, "apy_set_isdisjoint"],
    # str: case and shape
    "upper":        ["apy_str_upper"],
    "lower":        ["apy_str_lower"],
    "title":        ["apy_str_title"],
    "capitalize":   ["apy_str_capitalize"],
    "swapcase":     ["apy_str_swapcase"],
    "casefold":     ["apy_str_casefold"],
    "zfill":        [None, "apy_str_zfill"],
    "center":       [None, "apy_str_center", "apy_str_center_fill"],
    "ljust":        [None, "apy_str_ljust", "apy_str_ljust_fill"],
    "rjust":        [None, "apy_str_rjust", "apy_str_rjust_fill"],
    # str: trimming. The no-argument form strips WHITESPACE; the one-argument
    # form strips any character in the set, which is not the same operation.
    "strip":        ["apy_str_strip", "apy_str_strip_chars"],
    "lstrip":       ["apy_str_lstrip", "apy_str_lstrip_chars"],
    "rstrip":       ["apy_str_rstrip", "apy_str_rstrip_chars"],
    "removeprefix": [None, "apy_str_removeprefix"],
    "removesuffix": [None, "apy_str_removesuffix"],
    # str: splitting and joining
    "split":        ["apy_str_split_ws", "apy_split_of", "apy_str_split_n"],
    "rsplit":       ["apy_str_rsplit_ws", "apy_str_rsplit", "apy_str_rsplit_n"],
    "splitlines":   ["apy_str_splitlines", "apy_str_splitlines_keep"],
    "partition":    [None, "apy_str_partition"],
    "rpartition":   [None, "apy_str_rpartition"],
    "join":         [None, "apy_str_join"],
    # `s.format_map(m)` is `format` with the mapping handed over whole. It
    # is in the table and `format` is not, because `format` takes keywords
    # and a variable count and is lowered at the call site instead.
    "format_map":   [None, "apy_str_format_map"],
    "replace":      [None, None, "apy_str_replace", "apy_str_replace_n"],
    # str: searching
    "find":         [None, "apy_str_find", "apy_str_find2", "apy_str_find3"],
    "rfind":        [None, "apy_str_rfind", "apy_str_rfind2", "apy_str_rfind3"],
    "rindex":       [None, "apy_str_rindex", "apy_str_rindex2",
                     "apy_str_rindex3"],
    "startswith":   [None, "apy_str_startswith", "apy_str_startswith2",
                     "apy_str_startswith3"],
    "endswith":     [None, "apy_str_endswith", "apy_str_endswith2",
                     "apy_str_endswith3"],
    # str: classification
    "isalpha":      ["apy_str_isalpha"],
    "isdigit":      ["apy_str_isdigit"],
    "isdecimal":    ["apy_str_isdecimal"],
    "isnumeric":    ["apy_str_isnumeric"],
    "isalnum":      ["apy_str_isalnum"],
    "isspace":      ["apy_str_isspace"],
    "islower":      ["apy_str_islower"],
    "isupper":      ["apy_str_isupper"],
    "istitle":      ["apy_str_istitle"],
    "isascii":      ["apy_str_isascii"],
    "isprintable":  ["apy_str_isprintable"],
    "isidentifier": ["apy_str_isidentifier"],
    # str <-> bytes, and the numeric methods
    # The ENCODING ARGUMENT is accepted and ignored: this runtime stores text
    # as UTF-8 already, so encoding is a change of kind and not of content.
    # That makes both exact for UTF-8 and wrong for every other codec -- and
    # refusing the argument would reject the spelling nearly all code uses.
    # THE ERROR HANDLER IS THE SECOND ARGUMENT, and every arity reaches the
    # same three-parameter entry -- lowering pads the ones the call omitted,
    # because `errors="replace"` is what decides whether a bad byte is a
    # refusal or a replacement character.
    "encode":       ["apy_str_encode", "apy_str_encode", "apy_str_encode"],
    "decode":       ["apy_bytes_decode", "apy_bytes_decode",
                     "apy_bytes_decode"],
    "bit_length":   ["apy_bit_length"],
    "bit_count":    ["apy_bit_count"],
    "is_integer":   ["apy_is_integer"],
    "conjugate":    ["apy_conjugate"],
    # `hex` with a separator, `to_bytes` and `expandtabs` all take an argument
    # the no-argument form supplies a default for -- see `_dyn_builtin_method`.
    # `hex` reaches BYTES or a FLOAT, and the two answer entirely different
    # things -- so the no-argument form dispatches on the receiver. The
    # separator form is bytes' alone.
    # `bytes_per_sep` IS THE THIRD SLOT. `b.hex(":", 2)` groups the pairs,
    # which is most of what makes a long fingerprint readable.
    "hex":          ["apy_hex_of", "apy_bytes_hex", "apy_bytes_hex_n"],
    # THE RECEIVER DECIDES WHICH READING: a float's `fromhex` parses one
    # number where a bytes one reads byte pairs, and `apy_any_fromhex` is
    # where that is settled.
    "fromhex":      [None, "apy_any_fromhex"],
    # `to_bytes` TAKES ITS THREE PARAMETERS ALWAYS. Lowering pads the ones the
    # call left out, because `signed` is keyword-only and the other two have
    # defaults -- so every arity reaches one entry point rather than three.
    "to_bytes":     [None, None, None, "apy_to_bytes_n"],
    "as_integer_ratio": ["apy_as_integer_ratio"],
    "expandtabs":   ["apy_str_expandtabs", "apy_str_expandtabs"],
    # `translate` IS TWO METHODS UNDER ONE NAME. A str maps code points
    # through a dict; bytes map BYTES through a 256-byte table and take a
    # second `delete` argument the str form does not have. The receiver
    # is a run-time question, so the split is made in the runtime and
    # only the arity is decided here.
    "translate":    [None, "apy_str_translate", "apy_bytes_translate"],
    # DUNDERS CALLED DIRECTLY ON A BUILTIN. `(-5).__abs__()` and
    # `(0.0).__bool__()` are ordinary Python, and every one of these is the
    # operation the runtime already performs for the operator form -- so they
    # are the same symbol reached by another spelling, not a second
    # implementation that could disagree with it.
    "__abs__":      ["apy_abs"],
    "__bool__":     ["apy_to_bool"],
    "__trunc__":    ["apy_math_trunc"],
    "__neg__":      ["apy_neg"],
    "__len__":      ["apy_len"],
    "__repr__":     ["apy_repr"],
    "__str__":      ["apy_str"],
    # `(5).__index__()` -- PEP 357, and what `operator.index` reaches
    # directly. See `apy_index_obj` (`objects/c/_builtins.py`): the boxed
    # twin of `apy_index`, which unpacks a machine word for a subscript
    # instead. Found writing `operator.index`, which the pure-Python spec
    # in `Lib/operator.py` calls exactly this way.
    "__index__":    ["apy_index_obj"],
    "add_note":     [None, "apy_add_note"],
    # PEP 654. `subgroup` answers one group or None; `split` is shared with
    # `str.split` and dispatches on the receiver -- see `apy_split_of`.
    "subgroup":     [None, "apy_group_subgroup"],
    # `s.indices(n)` -- the numbers a walk over a sequence of that length
    # would really use, with omitted and negative bounds resolved.
    "indices":      [None, "apy_slice_indices"],
    # `@v.setter` and `@v.getter` -- each answers a NEW property carrying the
    # other half, because the decorator's result is rebound afterwards.
    "setter":       [None, "apy_prop_setter"],
    "getter":       [None, "apy_prop_getter"],
    # generators
    "send":         [None, "apy_gen_send"],
    "throw":        [None, "apy_gen_throw"],
    "close":        ["apy_gen_close"],
}

#: `index` and `count` are on BOTH a sequence and a str, and the runtime has a
#: different symbol for each. The receiver's kind is not known at compile time,
#: so the sequence one is emitted and the runtime falls back to the string
#: behaviour when handed a str -- which is why `apy_index_of`/`apy_count_of`
#: accept one.
_SHAPES = frozenset(
    (name, k) for name, syms in DYN_METHOD_TABLE.items()
    for k, sym in enumerate(syms) if sym is not None
)


#: Which of a dict's three parts each method wants, matching the runtime's
#: `apy_dict_parts` selector.
DICT_PARTS = {"keys": 0, "values": 1, "items": 2}


def method_symbol(name: str, argc: int) -> str | None:
    """The runtime symbol for `name` called with `argc` arguments, or None."""
    syms = DYN_METHOD_TABLE.get(name)
    if syms is None or argc >= len(syms):
        return None
    return syms[argc]


# ── keyword arguments ───────────────────────────────────────────────────────
#
# WHY THIS TABLE EXISTS. `DYN_METHOD_TABLE` is indexed by ARGUMENT COUNT, and
# a keyword argument does not change the count -- so `"a,b,c".split(",",
# maxsplit=1)` looked like a one-argument call, dispatched to the
# one-argument symbol, and the limit was DROPPED. Not refused: dropped. The
# program printed `['a', 'b', 'c']` and nothing anywhere said why.
#
# Eight of the ten built-in methods that take a keyword were wrong that way,
# and an unknown keyword was ignored too -- `(5).to_bytes(2, 'little',
# nonsense=True)` answered a value where CPython raises TypeError.
#
# READ OUT OF CPYTHON'S OWN SIGNATURES rather than transcribed, the same way
# the `compile()` probes were: `inspect.signature` over every name in
# `DYN_METHOD_TABLE`, on every builtin type that has it.
#
# ONE TABLE PER METHOD IS SOUND ONLY WHILE THE OWNERS AGREE, and for eight of
# the nine here they do. `translate` is the one that does not --
# `str.translate(table)` takes no keyword and `bytes.translate(table,
# delete=b"")` takes one -- and the entry below describes the BYTES signature,
# because that is the only owner a keyword can be meant for. A str receiver
# reaching the folded call is refused at run time, where the receiver is
# finally known; see `METHOD_KW_SYMBOL` for how the two wordings survive.
# `test_method_keywords.py` asks CPython both questions again on every run.

#: A parameter that has no default: leaving it out is a TypeError, not a
#: substitution. Distinct from `None`, which is a real default for `split`.
REQUIRED = object()

#: A parameter that cannot be given by name -- CPython marks it positional
#: only, so `"aaa".replace(old="a", ...)` is an error there and here.
POSITIONAL_ONLY = None

#: method -> its parameters IN POSITIONAL ORDER, each `(name, default)`.
#:
#: `update` is absent ON PURPOSE: it already has a branch of its own in the
#: lowering, because its keywords ARE the value -- `d.update(a=1)` sets a key
#: called `a` -- and there is no slot to fold one into.
#:
#: `sort` IS HERE FOR THE BY-NAME SPELLING ONLY. The written form has a
#: branch of its own too (`_KEYWORDS_OF_THEIR_OWN` keeps the fold off it, so
#: nothing is lowered twice), but `getattr(xs, "sort")(reverse=True)` reaches
#: the runtime through `apy_call_kw`, which has no signature to match the
#: name against unless it is written here -- and answered `list.sort() takes
#: no keyword arguments` for a call CPython sorts.
#:
#: A method here takes its keywords by POSITION, which is the whole
#: mechanism -- the keyword is moved into the slot it names and the ordinary
#: arity dispatch then sees the call it should have seen all along.
METHOD_PARAMS: dict[str, tuple[tuple[str | None, object], ...]] = {
    "split":      (("sep", None), ("maxsplit", -1)),
    "rsplit":     (("sep", None), ("maxsplit", -1)),
    "splitlines": (("keepends", False),),
    "replace":    ((POSITIONAL_ONLY, REQUIRED), (POSITIONAL_ONLY, REQUIRED),
                   ("count", -1)),
    "expandtabs": (("tabsize", 8),),
    "encode":     (("encoding", "utf-8"), ("errors", "strict")),
    "decode":     (("encoding", "utf-8"), ("errors", "strict")),
    "to_bytes":   (("length", 1), ("byteorder", "big"), ("signed", False)),
    # `bytes.translate(table, /, delete=b"")` -- the BYTES signature, because
    # a keyword can only be meant for that owner. `str.translate` takes no
    # keyword at all and is refused at run time, where the receiver is finally
    # known; see `METHOD_KW_SYMBOL`.
    "translate":  ((POSITIONAL_ONLY, REQUIRED), ("delete", b"")),
    # KEYWORD-ONLY IN CPYTHON, and a positional is refused by the arity gate
    # rather than here: `[].sort(None)` is `sort() takes no positional
    # arguments`, which the generated words table knows -- see
    # `apy_meth_positional`.
    "sort":       (("key", None), ("reverse", False)),
}

#: A method whose KEYWORD spelling reaches a DIFFERENT runtime symbol than the
#: same call written positionally.
#:
#: ONLY `translate`, and only because CPython gives two different messages:
#:
#:     "abc".translate(t, b"c")         str.translate() takes exactly one
#:                                      argument (2 given)
#:     "abc".translate(t, delete=b"c")  str.translate() takes no keyword
#:                                      arguments
#:
#: Folding puts the keyword in its slot, and after that nothing in the lowered
#: call says how it was written -- so the spelling has to survive as far as
#: the symbol. The receiver decides which message is right and is not known
#: until run time, which is why this cannot be settled here.
METHOD_KW_SYMBOL = {"translate": "apy_translate_kw"}


class KeywordError(Exception):
    """A keyword this method cannot take. Carries CPython's own wording.

    `owner` MARKS THE ONE WORDING THAT NAMES THE RECEIVER'S TYPE. CPython
    writes `str.upper() takes no keyword arguments`, and which type that is
    cannot be known here -- the dispatch is by arity precisely because there
    is no static receiver. The lowering turns a marked refusal into a run-time
    call that does have it; see `apy_kw_owner`.
    """

    owner = False


#: What a substitution costs. CPython's `Python/suggestions.c` constants, and
#: the reason they are not 1: a CASE change is cheaper than a real one, so
#: `SEP` finds `sep` and a longer all-caps name does not find its lowercase
#: twin. Reproduced rather than approximated, because a suggestion this
#: compiler makes where CPython makes none is a new divergence, not a
#: kindness.
_MOVE_COST = 2
_CASE_COST = 1
_MAX_STRING_SIZE = 40


def _substitution_cost(a: str, b: str) -> int:
    if a == b:
        return 0
    if a.lower() == b.lower():
        return _CASE_COST
    return _MOVE_COST


def _levenshtein(a: str, b: str, max_cost: int) -> int:
    """CPython's own edit distance, ported from `Python/suggestions.c`.

    THE COMMON AFFIXES COME OFF FIRST, which is not an optimisation here but
    part of the answer: the trimmed lengths are what the row below is sized
    against, and the early exits depend on them.
    """
    if a == b:
        return 0
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    a, b = a[i:], b[i:]
    while a and b and a[-1] == b[-1]:
        a, b = a[:-1], b[:-1]
    if not a or not b:
        return (len(a) + len(b)) * _MOVE_COST
    if len(a) > _MAX_STRING_SIZE or len(b) > _MAX_STRING_SIZE:
        return max_cost + 1
    if len(b) < len(a):
        a, b = b, a
    if (len(b) - len(a)) * _MOVE_COST > max_cost:
        return max_cost + 1
    row = [(k + 1) * _MOVE_COST for k in range(len(a))]
    result = 0
    for b_index, code in enumerate(b):
        distance = result = b_index * _MOVE_COST
        minimum = None
        for k, ch in enumerate(a):
            substitute = distance + _substitution_cost(code, ch)
            distance = row[k]
            result = min(min(result, distance) + _MOVE_COST, substitute)
            row[k] = result
            if minimum is None or result < minimum:
                minimum = result
        if minimum is not None and minimum > max_cost:
            return max_cost + 1
    return result


def _suggest(wrong: str, options) -> str | None:
    """The parameter CPython would propose for a misspelling, or None.

    NEAREST WINS AND TIES KEEP THE FIRST, which is what the `<` does -- the
    same walk CPython makes over the names it has.
    """
    best, best_at = None, None
    for option in options:
        limit = (len(wrong) + len(option) + 3) * _MOVE_COST // 6
        far = _levenshtein(wrong, option, limit)
        if far > limit:
            continue
        if best_at is None or far < best_at:
            best, best_at = option, far
    return best


def fold_keywords(name: str, argc: int, given: list[str],
                  pad_to: int = 0) -> list | None:
    """How to arrange `argc` positional and these named arguments, or None.

    ANSWERS A PLAN RATHER THAN VALUES, because the caller has to LOWER each
    argument and the order it lowers them in is the order the program must
    evaluate them in -- which is source order, not the positional order the
    plan describes. Returning a plan lets the caller do both.

    Each entry is `("pos", i)`, `("kw", name)` or `("default", value)`.
    `None` means there were no keywords and the call is already positional.

    RAISES RATHER THAN DROPS. An unknown keyword, a duplicate, or a missing
    required parameter is a TypeError in CPython, so it is an error here --
    the alternative is the silent wrong answer this table was written for.
    """
    if not given and not pad_to:
        return None
    params = METHOD_PARAMS.get(name)
    if params is None:
        # CPython WRITES THE OWNER -- `str.upper() takes no keyword
        # arguments` -- and there is no static type for the receiver here,
        # which is the whole reason the dispatch is by arity. So the refusal
        # is MARKED rather than worded: the lowering has the receiver and
        # turns a marked one into `apy_kw_owner`, which names it.
        blame = KeywordError(f"{name}() takes no keyword arguments")
        blame.owner = True
        raise blame
    index = {p: i for i, (p, _) in enumerate(params) if p is not None}
    slots: dict[int, tuple] = {i: ("pos", i) for i in range(argc)}
    # THE ORDER OF THESE FOUR REFUSALS IS CPYTHON'S and is not the order they
    # occur to a reader. A MISSING REQUIRED POSITIONAL BEATS EVERYTHING:
    # `"aaa".replace(count=1)` reports the two positionals it did not get,
    # not the keyword it did -- so the count check comes before the names are
    # looked at all.
    required = sum(1 for _, d in params if d is REQUIRED)
    if argc < required:
        raise KeywordError(
            f"{name}() takes at least {required} positional "
            f"argument{'' if required == 1 else 's'} ({argc} given)")
    # THEN TOO MANY, counting the keywords in: `"a,b".split(",", None,
    # maxsplit=1)` is three arguments for two parameters, and CPython says so
    # rather than complaining that `maxsplit` was given twice. A call with no
    # positionals at all is worded as KEYWORD arguments.
    if argc + len(given) > len(params):
        if argc == 0:
            raise KeywordError(
                f"{name}() takes at most {len(params)} keyword "
                f"argument{'' if len(params) == 1 else 's'} "
                f"({len(given)} given)")
        raise KeywordError(
            f"{name}() takes at most {len(params)} "
            f"argument{'' if len(params) == 1 else 's'} "
            f"({argc + len(given)} given)")
    for kw in given:
        at = index.get(kw)
        if at is None:
            near = _suggest(kw, [p for p, _ in params if p is not None])
            raise KeywordError(
                f"{name}() got an unexpected keyword argument {kw!r}"
                + (f". Did you mean {near!r}?" if near else ""))
        if at in slots:
            # CPYTHON'S OWN WORDING, position counted from one -- it names the
            # slot the caller already filled, which is the useful half.
            raise KeywordError(f"argument for {name}() given by name "
                               f"({kw!r}) and position ({at + 1})")
        slots[at] = ("kw", kw)
    # FILL THE GAPS, so the arity dispatch sees a contiguous call. Every
    # default here is a value the runtime already accepts in that position --
    # `"a b c".split(None, 1)` is a call the runtime answers correctly today,
    # which is why folding can lean on it rather than needing new symbols.
    # `pad_to` FILLS THE TAIL AS WELL AS THE GAPS, for a method whose runtime
    # entry point takes every parameter and lets the defaults be supplied here.
    highest = max(max(slots) if slots else -1, pad_to - 1)
    plan = []
    for i in range(highest + 1):
        if i in slots:
            plan.append(slots[i])
            continue
        _, default = params[i]
        if default is REQUIRED:
            # UNREACHABLE while the count check above stands, and kept as the
            # backstop it is: the parameter may be POSITIONAL-ONLY, whose
            # name is None, and a message naming None is what this used to
            # say out loud.
            raise KeywordError(
                f"{name}() takes at least {required} positional "
                f"argument{'' if required == 1 else 's'} ({argc} given)")
        plan.append(("default", default))
    return plan


#: NOT GIVEN AT ALL -- a slot nobody filled, which SHORTENS the call rather
#: than filling it. `int()` is 0 and `int(x)` converts, and the two are
#: different runtime entry points; a default that made the short call long
#: would send `int()` down the conversion path with nothing to convert.
ABSENT = object()

#: The builtin TYPE CONSTRUCTORS, as `METHOD_PARAMS` holds the methods:
#: `(keyword name or POSITIONAL_ONLY, default)` per slot, read out of CPython.
#:
#: THE NAMES ARE THE POINT. `bytes(source="a", encoding="utf-8")` and
#: `int("ff", base=16)` are calls CPython answers, and every one of them used
#: to have its keywords DROPPED -- `int(x="1")` answered 0 and `list(x=1)`
#: answered `[]` rather than reporting anything at all.
#:
#: `dict` IS NOT HERE. It takes ARBITRARY keywords -- they become the
#: mapping's own keys -- so there is nothing to fold, only a count to check.
CTOR_PARAMS = {
    "int":        ((POSITIONAL_ONLY, ABSENT), ("base", ABSENT)),
    "bool":       ((POSITIONAL_ONLY, ABSENT),),
    "float":      ((POSITIONAL_ONLY, ABSENT),),
    "list":       ((POSITIONAL_ONLY, ABSENT),),
    "tuple":      ((POSITIONAL_ONLY, ABSENT),),
    "set":        ((POSITIONAL_ONLY, ABSENT),),
    "frozenset":  ((POSITIONAL_ONLY, ABSENT),),
    "range":      ((POSITIONAL_ONLY, REQUIRED), (POSITIONAL_ONLY, ABSENT),
                   (POSITIONAL_ONLY, ABSENT)),
    "slice":      ((POSITIONAL_ONLY, REQUIRED), (POSITIONAL_ONLY, ABSENT),
                   (POSITIONAL_ONLY, ABSENT)),
    "memoryview": (("object", REQUIRED),),
    # `complex(imag=2)` IS `2j`: the real part defaults to a real zero, which
    # is not the same as not being given -- `complex(x)` asks `x` through
    # `__complex__` and `complex(x, 0)` builds from parts.
    "complex":    (("real", 0), ("imag", ABSENT)),
    # NONE MEANS "THE DEFAULT" to the codec pair, which is the padding
    # `.encode()` and `.decode()` already take.
    "str":        (("object", ABSENT), ("encoding", None), ("errors", None)),
    "bytes":      (("source", ABSENT), ("encoding", None), ("errors", None)),
    "bytearray":  (("source", ABSENT), ("encoding", None), ("errors", None)),
}

#: The constructors that take ONE positional argument and any keyword at all,
#: because the keywords ARE the value: `dict(a=1)` is `{"a": 1}`.
CTOR_ANY_KEYWORD = {"dict": 1}

def _guarded_arities() -> frozenset:
    """The `(method name, argument count)` pairs that need a run-time guard.

    A SYMBOL SERVES THE COUNT BUT SOME KIND REFUSES IT. `x.pop()` reaches
    `apy_list_pop` because a list's `pop` takes none -- and a DICT's does not,
    so `{}.pop()` was `KeyError: None` from a symbol that should have refused
    the call, and `set().pop(1)` was `'set' object has no attribute 'pop'`
    about a method a set plainly has. CPython raises a TypeError about the
    COUNT for both.

    TEN PAIRS, computed rather than listed: the name and the count are both
    known at the call site, so the guard is emitted only where a kind
    disagrees and every other builtin method call pays nothing. See
    `apy_meth_arity`, which is the guard.
    """
    from ...objects.c.kindmeth_table import KINDMETH_WORDS
    out = set()
    for (name, _kind), packed in KINDMETH_WORDS.items():
        least, most = packed & 15, (packed >> 4) & 15
        for argc in range(5):
            if least <= argc <= most:
                continue
            if method_symbol(name, argc) is not None:
                out.add((name, argc))
    return frozenset(out)


#: Computed once at import, from the generated table -- see above.
METHOD_ARITY_GUARD = _guarded_arities()


#: THE SURPLUS-ARGUMENT WORDING IS NOT ONE WORDING. Most of the builtin types
#: are `list expected at most 1 argument, got 2` -- the bare name, no
#: parentheses -- and four of them are `bytes() takes at most 3 arguments (4
#: given)`. Read out of CPython rather than guessed, because the two forms
#: differ in every visible way and a test tells them apart.
_CTOR_PARENTHESISED = {"bytes", "bytearray", "complex", "memoryview"}

#: What a constructor says when its FIRST slot was left empty and a later one
#: was not. `str` is the odd one and says nothing at all: `str(encoding="x")`
#: is `''`, the zero-argument call, because there is nothing to decode.
_CTOR_HEADLESS = {
    "int": "int() missing string argument",
    "memoryview": "memoryview() missing required argument 'object' (pos 1)",
    "bytes": "encoding without a string argument",
    "bytearray": "encoding without a string argument",
}

#: WHAT A MISSING REQUIRED FIRST ARGUMENT IS CALLED. Two of the three count
#: their arguments and the third names the one it wanted, which is the arg
#: clinic's wording rather than the bare type's.
_CTOR_MISSING = {
    "range": "range expected at least 1 argument, got 0",
    "slice": "slice expected at least 1 argument, got 0",
    "memoryview": "memoryview() missing required argument 'object' (pos 1)",
}

#: `errors` WITHOUT AN `encoding` is its own refusal for the two byte
#: constructors, whatever the source is: `bytes(b"a", errors="replace")` is
#: `errors without a string argument` and not a decode with a default codec.
_CTOR_ERRORS_ALONE = {"bytes", "bytearray"}

#: The zero-argument call answers for a constructor whose first slot is empty
#: but whose later slots are not. Only `str`, and see `_CTOR_HEADLESS`.
CTOR_HEADLESS_IS_EMPTY = {"str"}


def _ctor_surplus(name: str, most: int, argc: int, kwc: int) -> str:
    """CPython's wording for too many arguments to a builtin constructor."""
    plural = "" if most == 1 else "s"
    if kwc or name in _CTOR_PARENTHESISED:
        # THE PARENTHESISED FORM, which is what every one of them uses once a
        # KEYWORD is in the count: `int("1", 10, base=2)` is `int() takes at
        # most 2 arguments (3 given)` where `int("1", 10, 3)` is not.
        return (f"{name}() takes at most {most} argument{plural} "
                f"({argc + kwc} given)")
    return f"{name} expected at most {most} argument{plural}, got {argc}"


def fold_ctor_keywords(name: str, argc: int, given: list[str]) -> list | None:
    """How to arrange a builtin CONSTRUCTOR's arguments, or None.

    THE SAME PLAN `fold_keywords` ANSWERS, and for the same reason: the
    caller lowers each argument in SOURCE order and then places it. What
    differs is the wording of every refusal -- a type constructor and a method
    disagree about all four of them -- and that a slot nobody filled makes the
    call SHORTER rather than taking a default. See `ABSENT`.

    `None` means the call is already positional and needs no rearranging.
    """
    params = CTOR_PARAMS.get(name)
    if params is None:
        most = CTOR_ANY_KEYWORD.get(name)
        if most is not None:
            # `dict`. THE KEYWORDS ARE THE VALUE, so only the count is ours.
            if argc > most:
                raise KeywordError(_ctor_surplus(name, most, argc, 0))
            return None
        return None
    most = len(params)
    index = {p: i for i, (p, _) in enumerate(params) if p is not POSITIONAL_ONLY}
    # THE ORDER OF THESE REFUSALS IS CPYTHON'S. A TYPE THAT TAKES NO KEYWORD
    # AT ALL says exactly that and names none of them, ahead of every other
    # complaint: `range(start=1)` is `range() takes no keyword arguments` and
    # not the missing first argument it also does not have.
    if given and not index:
        raise KeywordError(f"{name}() takes no keyword arguments")
    # THEN A MISSING REQUIRED FIRST ARGUMENT, in the type's own wording --
    # `range` counts them and `memoryview` names the one it wanted.
    required = sum(1 for _, d in params if d is REQUIRED)
    if argc < required and not any(g in index and index[g] < required
                                   for g in given):
        raise KeywordError(_CTOR_MISSING[name])
    if argc + len(given) > most:
        if argc == 0 and given:
            raise KeywordError(
                f"{name}() takes at most {most} keyword "
                f"argument{'' if most == 1 else 's'} ({len(given)} given)")
        raise KeywordError(_ctor_surplus(name, most, argc, len(given)))
    if not given:
        return None
    slots: dict[int, tuple] = {i: ("pos", i) for i in range(argc)}
    for kw in given:
        at = index.get(kw)
        if at is None:
            near = _suggest(kw, [p for p, _ in params
                                 if p is not POSITIONAL_ONLY])
            raise KeywordError(
                f"{name}() got an unexpected keyword argument {kw!r}"
                + (f". Did you mean {near!r}?" if near else ""))
        if at in slots:
            raise KeywordError(f"argument for {name}() given by name "
                               f"({kw!r}) and position ({at + 1})")
        slots[at] = ("kw", kw)
    if name in _CTOR_ERRORS_ALONE and "errors" in given and 1 not in slots:
        raise KeywordError("errors without a string argument")
    if 0 not in slots and params[0][1] in (ABSENT, REQUIRED):
        if name in CTOR_HEADLESS_IS_EMPTY:
            # `str(encoding="x")` IS `''`. There is nothing to decode, and
            # CPython answers the empty string rather than refusing.
            return []
        raise KeywordError(_CTOR_HEADLESS[name])
    plan = []
    for i in range(max(slots) + 1):
        if i in slots:
            plan.append(slots[i])
            continue
        _, default = params[i]
        if default is ABSENT or default is REQUIRED:
            # UNREACHABLE while the checks above stand: slot 0 is the only one
            # that can be empty under a filled one for these shapes, and it is
            # answered above. Kept as the backstop it is.
            raise KeywordError(_CTOR_HEADLESS.get(
                name, f"{name}() takes no keyword arguments"))
        plan.append(("default", default))
    return plan
