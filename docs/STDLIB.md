# The standard library, rebuilt

**Target: CPython 3.14, module by module, each one measured against it.**

The previous set is in `archived/stdlib-prerefactor/` -- 27 modules, 6,807
lines. It was not wrong, it was UNPLANNED: each module was written to the depth
some conformance case happened to need, so `typing` has the classes and not the
special forms, `sys` is half compiler constants, and nothing records which half
of anything is there. A library nobody can predict the coverage of is one every
user has to test for themselves.

This time the coverage is the deliverable, and it is stated per module.

## How a module is built

**It is ordinary Python, compiled by uasm.** That is the constraint that
makes the whole arrangement honest, and `bundled.py` has said so from the
start:

> a bundled module is compiled by this compiler, so it may only use what this
> compiler accepts. A construct one of them cannot use is a gap worth closing
> rather than a reason to drop back to C.

So a module that cannot be written is a compiler bug with a name, not a reason
to write C. That has already paid: the docstring gap, the width rules and the
static-path module storage were all found by writing library code.

**Nothing is loaded at run time.** `import functools` splices that module's
definitions into the program under mangled names -- see `bundled.py`. There is
no module object, no import system, and no cost to a program that imports
nothing.

## How a module is proved

`tests/stdlib/<module>.py` is a program that exercises it. The runner executes
it under **CPython** and under **uasm** and compares the output exactly.

That is the corpus argument applied to the library: CPython is the oracle, the
test is written against the SPECIFICATION rather than against what uasm
currently does, and a divergence is uasm's until proven otherwise. A test
that only asserts what already works tests nothing.

Each test file opens with a coverage line saying what of the module it claims.
That line is the module's contract and the thing this rebuild exists to make
true.

## The order, and why

Dependency order first, value second. A module may only import ones already
built.

    0  keyword  operator  types  abc  sys            no dependencies at all
    1  collections.abc  functools  itertools  enum   the protocol furniture
       numbers
    2  collections  dataclasses  typing  contextlib  what ordinary code uses
       copy  string  bisect  heapq  struct
    3  io  os  pathlib  json  re  textwrap           the ones with real
       traceback  warnings  inspect                  surface
    4  math  decimal  fractions  statistics          numeric and temporal
       random  datetime  zoneinfo
    5  time  threading  socket  select            THE FLOOR GREW -- below
       subprocess

**Tier 5 was a claim about the FLOOR, and it was mostly wrong.** Four of
the five modules came out of it, each for its own reason, and the reason
is worth recording because it is the same mistake a reader could make
again:

* `time` needed no new platform function at all. `objects/hostsvc.py`
  already declared a `time` group (`host_time_unix`,
  `host_time_monotonic`, `host_sleep`) and nobody had noticed.
* `socket` and `select` needed a group that was DECLARED AND IMPLEMENTED
  NOWHERE. `net` had been sitting in `hostsvc.py` with a comment saying
  streams only and blocking, and no backend answered any of it -- so the
  floor did not have to grow, it had to be BUILT OUT: the interpreter now
  calls CPython's `socket` and the C backend calls BSD sockets, and both
  run `tests/stdlib/socket.py` identically. Two operations were added to
  the group along the way (`host_net_port`, `host_net_ready`), because a
  contract that cannot report which ephemeral port it got, or whether a
  descriptor would block, cannot be used by the modules it exists for.
* `threading` still cannot have a second thread here, but the reason has
  moved: there IS a group now. `thread` was added to `hostsvc.py` for the
  C frontend's `<threads.h>`, the C backend answers it over pthreads and
  the reference interpreter over Python threads -- so the floor is not
  what stops this module any more. The object runtime's reference counts
  are: they are ordinary loads and stores, and two threads sharing a
  Python object would corrupt them. Everything in the module that is NOT
  the scheduler -- locks, semaphores, events, conditions, barriers,
  thread-local storage -- is a state machine that answers exactly what
  CPython answers on one thread, and only `Thread` itself diverges.

* `subprocess` is the one the claim was actually about: creating a
  process is the largest single addition to the floor of anything here,
  and there was no group waiting. `proc` is that addition, and it is ONE
  operation -- `host_proc_run` -- for a reason the module docstring gives
  at length: a live child with pipes needs a thread to drain them.
  `fork`/`execvp`/`waitpid` in the C backend, CPython's `subprocess` in
  the interpreter, and the same answers from both.

THE LESSON IS THAT THE DECISION IS PER MODULE AND NOT PER TIER. "Needs
the floor to grow" was false for `time`, false for `socket` and `select`
(the group was there, unimplemented), true for only part of `threading`,
and true for `subprocess` -- where the growth turned out to be one
operation rather than a redesign. The floor is still three MANDATORY
functions (`docs/INERT-RUNTIME.md`); everything added here is an OPTIONAL
group a backend declares, and a program using one a backend does not have
is still refused at compile time naming the group.

WHAT WRITING THESE FOUND, which is the argument for writing library code
in the language being compiled: a `@property` on an exception subclass
handed the program the descriptor OBJECT, and a `__str__` on one was
ignored in favour of the args. Both are fixed in `objects/host.py`.
`bytearray.append` and `.extend` did not exist at all -- `.append` on the
most ordinary way there is to build one up answered `'bytearray' object
has no attribute 'append'` -- and are now in `_apy_seq_push`/`_apy_extend`
beside the list's. Three bugs, all found by writing library code in the
language being compiled, which is the whole argument for doing it that
way.

`math` FOUND FOUR MORE, and they are the reason it is now complete rather
than merely larger. Writing `tests/stdlib/math.py` meant running the same
program under the interpreter and both compiled runtimes, which is where
three arrangements of one function stop agreeing:

* `gcd`, `lcm` and `isqrt` read a machine word off their arguments, which
  is a POINTER when the argument outgrew one -- `gcd(2**100, 2**60)`
  answered 8 in a compiled build and the right answer interpreted. Fixed
  in all three by detecting the big and going through `apy_mod`/`apy_abs`.
* `floor`/`ceil`/`trunc` never asked an object anything, so
  `math.floor(Fraction(7, 2))` was a TypeError. `bundled/fractions.py` had
  documented it and left it; PEP 3141's hooks are consulted now, and
  `bundled/decimal.py` gained `__floor__`/`__ceil__` to match.
* `math.log(x, base)` ignored the base and answered the natural log --
  a wrong number rather than an error.
* Every native default was emitted through `apy_from_float`, which was
  invisible while the only ones in the table were `isclose`'s two
  tolerances. `math.prod`'s `start=1` is an INT and decides the type of
  the whole product: `prod([1, 2, 3, 4])` answered 24.0.

## What is not the standard library

`bundled/` also holds `_pyast`, `_pycompile`, `_pylex`, `_pyparse`, `_pyrun`
and `_pyvalidate`. Those are the Python-in-Python compiler spliced into any
program naming `compile`, `eval` or `exec`. They are spliced by the same
machinery and that is all they have in common with a library module; they were
not archived and are not rebuilt.

`ctypes` is a compile-time feature of the frontend (`frontends/python/cffi.py`)
rather than a bundled module, and is unaffected.

## When an object dies, and how exactly that is known

`weakref` and `gc` are in the table below, and neither could be until the
runtime could answer a question it had never been asked: WHEN does an object
stop existing? The reference interpreter now keeps a SHADOW REFERENCE COUNT
(`objects/host.py`'s `incref`/`decref`, hooked from `ir/interpreter.py`'s
one register-write choke point) whose only job is to make what a Python
program can OBSERVE about lifetime match CPython: `__del__` at the right
statement, a `weakref` going dead at the right statement, and a reference
CYCLE surviving both until `gc.collect()` -- which falls out of TRUE
refcounting for free, with nothing added: a cycle's members hold each
other's counts above zero, so no amount of counting collects one, which is
exactly why CPython ships a second mechanism under that name and why this
does too.

It is a SHADOW count. Nothing is freed; `docs/INERT-RUNTIME.md`'s memory
story is untouched, and a compiled build keeps no count at all (which is why
`weakref`'s primitives refuse BY NAME there rather than answer a number they
cannot know).

**EXACT, measured against CPython as the oracle:** a name dropped by `del`,
by reassignment, or by its function returning; a value PASSED INTO A CALL
and dropped afterwards -- `f(x); del x` -- including a call through an
argument buffer, which is every method call, every class instantiation and
`weakref.ref(x)` itself; an object dropped by the container holding it
(`list`/`dict`/`set` element, instance attribute) being overwritten, removed
or cleared; `x is None` on a value nothing else holds; a reference CYCLE
broken by `gc.collect()`, and one NOT broken by it because something outside
still points at the cycle; `sys.getrefcount` for every reference a program
itself makes; and THE FINAL COLLECTION AT SHUTDOWN -- a `__del__` still
pending when the last statement finishes runs, after the program's own
output, exactly as CPython's does, with the module's globals dropped in the
order they were bound so a global list finalizes its elements and a closure
its captured variable. The ORDER several objects dying at the same moment are
finalized in is exact too, for the three shapes that have one: a frame's
locals go in reverse (CPython's own order, checked), a list's elements go in
reverse (CPython's `list_dealloc` walks its array backwards), and a dict's
entries go forward. So are the two orders `gc.collect()` has: a weakref
callback for a collected cycle fires BEFORE the cycle is torn down, and a
callback whose own `ref` is inside the same garbage does not fire at all --
both CPython's documented behaviour, both checked against it.

**A CAPTURED VALUE DIES WITH ITS CLOSURE**, which took three findings in a
row and is worth the space because each one hid the next. The cell counts
what is in it and a function counts its cells, so dropping the last
reference to a closure drops the box and what the box holds -- but reporting
those cells to `_held_by` fired EARLY twice before it fired correctly.
First, a closure reached `_invoke_obj` as the `env` argument with a count of
ZERO, because a dynamic call never retired its CALLEE slot, so nothing
counted the function at all and the unpin after each call finalized it
mid-call. Second, with the count right, a closure RETURNED from the function
that built it still died at that function's teardown -- and the reason was
not the returned-value handoff, which works: `_invoke_obj` minted a FRESH
HANDLE for the callee on every call, with its own count starting at zero, so
`_call`'s parameter incref and teardown decref took that second count from
one to zero and finalized the function while it was being called. The
interned handle is used for an unbound function now; a bound method still
needs its own, because the receiver travels in the value. The same shape of
bug the `Cell` interning fix found earlier, in the one place left holding
it.

**LATE, NEVER EARLY, and this is the direction the whole scheme is built to
be wrong in** -- an under-decref delays a finalizer, an over-decref runs one
while the object is still in use:

* a value passed into a HOST PRIMITIVE whose body has not been audited. A
  call retires its argument temporaries only when the callee is known to
  take its own counted reference to anything it keeps. For a call into
  compiled Python that is true BY CONSTRUCTION -- `_call` increfs every
  pointer parameter on entry and drops it at teardown, and every way a
  Python body can keep a value past its return counts it -- so
  `ir/interpreter.py`'s `_interpreted` covers all of those at once, and
  `f(x); del x` finalizes exactly where CPython does. For the two hundred
  `apy_*` primitives it is true one at a time, and `_NON_RETAINING` holds
  the ones actually read: `apy_is`, the three container stores, the two
  cell stores, and `apy_setattr`.

  THE PURE INSPECTORS ARE CERTIFIED TOO -- `apy_truth`, `apy_len`,
  `apy_raw_len`, `apy_hash`, `apy_repr`, `apy_str`, `apy_text_of`, the six
  comparisons, and the split halves of length, hash and equality. Each
  answers a fresh value or a machine word and stores none of its arguments;
  where one consults a user hook (`__bool__`, `__len__`, `__repr__`,
  `__eq__`) that is interpreted Python, which the `_interpreted` argument
  already covers. WHAT MADE THEM WORTH READING: `repr(g)` on a module-level
  GLOBAL minted a temporary the frame never retired, so `sys.getrefcount`
  read one high per call and anything held only that way waited for
  teardown. A LOCAL never showed it -- the register holding the binding IS
  the argument -- which is why the shape hid until a global was measured.

  THE CALLEE OF A DYNAMIC CALL IS RETIRED TOO, when the value in that slot
  can be certified. `apy_call` and its three siblings take the callable as
  their first argument, and leaving it alone made `alias()` on a
  module-level global read one high per CALL SITE. A PLAIN FUNCTION IS THE
  CASE THAT CAN BE ANSWERED: calling one runs interpreted Python, which
  counts its parameters by construction and keeps no reference to the
  function object itself. A CLASS CANNOT -- the instance it builds points
  back at it through `Instance.cls`, which is not counted -- and neither
  can a host `Native`, for the reason none of them can be certified one at
  a time. So `sys.getrefcount` on a function now matches CPython exactly,
  before and after any number of calls, through a global or a local,
  closure or not.

  `apy_setattr` WAS THE NOTABLE ABSENTEE AND IS NOW CERTIFIED, which is what
  makes `h.child = Noisy(); h.child = None` finalize at the second store
  rather than at frame teardown. Every branch was read: a `__setattr__`
  override and a data descriptor's `__set__` are interpreted Python, which
  the `_interpreted` argument already covers; an `Instance`, an `Exc`, a
  `Class` and a `Func` all store through `_attr_store`, which increfs the
  new value and decrefs the old; `C.__name__ = ...` writes a plain `str`,
  which is never tracked. THE ONE SHAPE NOT CERTIFIED is a property whose
  SETTER IS A NATIVE (`property(g).setter(some_builtin)`), which no program
  is known to write: `_pin_for_native` increfs the value instead of
  certifying the lambda, so that shape LEAKS rather than risking an early
  finalize.
* a temporary in a block that lies ON A CYCLE. A back edge means "last
  STATIC read" is not "last dynamic read", so retiring one is only sound
  where the value cannot survive an iteration -- which a block-local
  register cannot, since a block has no branches and the next pass writes a
  fresh value.

  THIS USED TO BE THE WHOLE FUNCTION and is now only the loop itself. A
  block off every cycle cannot be visited twice in one invocation --
  revisiting it would need a path from it back to itself, which is what
  being on a cycle means -- so a register written and read only in such
  blocks has a static read count that IS its dynamic one.
  `interpreter._looping_blocks` finds them with two ordinary depth-first
  walks. What it recovers is most of a loopy function, and the commonest
  shape is a MODULE BODY: one `for` anywhere in it used to make every
  temporary in the whole module wait for the end of the program, which
  turned up twice while writing the tests for this -- both times the loop
  was three lines from what was being measured and had nothing to do with
  it.

  WHAT IS LEFT is a value a LOOP-RESIDENT register still holds. An object
  built in the last iteration and kept in one finalizes at frame teardown
  rather than where the binding is cleared, because that register can be
  read again. It is usually visible as ORDER rather than lateness: `for it
  in items` leaves `it` holding the last element, so that one outlives the
  list and its `__del__` runs after the others' instead of first.
* the ORDER of a shutdown collection when several objects die together and
  one of them is also held by something the run over-held. Everything that
  should finalize does (see below); which of two `__del__`s prints first can
  differ.
* `sys.getrefcount(o)` asked by a function ABOUT ITS OWN PARAMETER reads one
  higher: that binding is a counted reference here and CPython's interpreter
  borrows it from the caller's frame. Every reference a PROGRAM makes -- a
  name, a list or set membership, a dict value, an attribute -- counts
  identically in both.

## Rebuilt so far

**48 modules, and every one of them has a differential test.** Nothing is in
the table below without a `tests/stdlib/<module>.py` that CPython 3.14 and
uasm both run to identical bytes, and nothing is in `bundled/` without
a row here -- the two are checked against each other rather than kept in
step by hand. The five tiers of the plan above are done; what is left is
DEPTH inside modules that are present, and each row says exactly which
depth it claims.

| module | coverage |
| --- | --- |
| `keyword` | complete |
| `warnings` | `warn`, filters, `catch_warnings`, `formatwarning`, `deprecated`; NOT the once/default registry, `stacklevel`, or the warning CPython issues when a class SUBCLASSES a deprecated one |
| `types` | `ModuleType`, `SimpleNamespace`, `new_class`. Also the runtime's own `GenericAlias`: `list[int]` compares and hashes by origin and arguments, and a union's arms are order-free and duplicate-free (`int \| str` is `str \| int` is `int \| str \| int`), so one is usable as a dict key or a set element. A builtin kind is named the way CPython names it in a message and in `<class '...'>` -- `types.GenericAlias`, `typing.Union` -- while `__name__` and `__qualname__` answer the last component alone. |
| `itertools` | `count`, `repeat`, `chain`, `islice`, `groupby`, `product`, `combinations` |
| `functools` | `reduce`, `wraps`, `total_ordering`, `partial`, `cached_property`, `lru_cache`, `cache`, `singledispatch`. `total_ordering` ROOTS ON WHICHEVER ordering operator the class wrote -- `__lt__` for preference, then `__le__`, `__gt__`, `__ge__`, and a ValueError when it wrote none -- and fills only the MISSING three, leaving an operator the class wrote alone. Each derived one reaches the root as a METHOD (`type(self).__lt__(self, other)`) so a NotImplemented comes back as a value: spelling it as the OPERATOR raises the moment the root declines, from inside the synthesised method and with that call's operands, so `5 < v` reported a refusal between `'Version'` and `'int'` in that order. No `__ne__`, which CPython's does not write either. |
| `re` | the whole ordinary language and surface; NOT lookbehind, conditional/atomic groups, possessive quantifiers, `\N{...}`, property escapes, `Scanner`, `bytes` patterns -- each refused BY NAME |
| `contextlib` | `contextmanager`, `suppress`, `ExitStack`, `nullcontext`; NOT the `async` half, `closing`, `redirect_*`, `chdir` |
| `inspect` | `signature`, `Signature`, `Parameter` and the five kinds, `isfunction`, `isclass`, `getdoc` of a FUNCTION; NOT source, frames, `bind`, `getfullargspec` |
| `__future__` | complete |
| `enum` | `Enum`, `IntEnum`, `StrEnum`, `Flag`, `IntFlag`, `auto`, `unique`, aliases, `__members__`; NOT `verify`/`EnumCheck`, `boundary=`, `global_enum`, `member`/`nonmember`, `_missing_`, functional creation |
| `copy` | `copy`, `deepcopy`, the hooks, the memo; NOT `__reduce__`, `__getstate__`, `copyreg`, `copy.replace` |
| `dataclasses` | `dataclass` with init/repr/eq/order/unsafe_hash/frozen/match_args/kw_only, `field` with all eight arguments, `Field`, `fields`, `is_dataclass`, `asdict`/`astuple` recursing, `replace`, `make_dataclass`, `InitVar`, `KW_ONLY`, `MISSING`, `FrozenInstanceError`, `__post_init__`, ClassVar exclusion, inheritance; NOT `slots`, `weakref_slot` (refused BY NAME), `field(doc=)`, `Field[int]` |
| `pathlib` | the whole PURE half -- `PurePosixPath`, `parts`, `name`, `stem`, `suffix`, `suffixes`, `parent`, `parents`, `root`, `anchor`, `is_absolute`, `joinpath`, `with_name`, `with_suffix`, `with_stem`, `relative_to`, `match`, `as_posix`, `/`, comparison, `__fspath__`; and a CONCRETE `Path` with `exists`, `is_file`, `is_dir`, `read_bytes`, `read_text`, `write_bytes`, `write_text`, `mkdir`, `touch`, `unlink`, `rmdir`. NOT `iterdir`, `glob`, `rglob`, `stat`, `resolve`, `absolute`, `open`, symlinks -- each refused BY NAME -- and `Path` is POSIX-flavoured, so `str()` renders `/` where CPython on Windows renders `\`. Its concrete half now goes through `objects/hostsvc.py` rather than `ctypes`, so it is no longer Windows-and-C-only -- see below. |
| `abc` | `ABCMeta`, `ABC`, `abstractmethod`, `register`, `__abstractmethods__`, `__subclasshook__`, `update_abstractmethods`, `get_cache_token`, `@property`/`@classmethod` stacked over `@abstractmethod`; the three deprecated decorators are FUNCTIONS rather than subclasses of `property`/`classmethod`/`staticmethod`, which this frontend cannot extend. NOT the per-class negative cache. |
| `collections` | `namedtuple` (positional and keyword construction, `defaults`, `rename`, `_make`/`_replace`/`_asdict`/`_fields`/`_field_defaults`), `deque` (both ends, `maxlen`, `rotate`, `extendleft`), `defaultdict`, `Counter` (all four operators, `most_common`, `elements`, `subtract`, `total`), `OrderedDict` (`move_to_end`, order-sensitive `__eq__`), `ChainMap`, `UserDict`, `UserList`, `UserString`. NOT `deque`'s O(1) ends -- it is a list inside, so the answers match and the costs do not -- and `namedtuple` has no `__slots__`. |
| `collections.abc` | all twenty-five ABCs, the mixin methods for `Sequence`/`MutableSequence`/`Set`/`MutableSet`/`Mapping`/`MutableMapping`, `__subclasshook__` for the structural protocols, and the builtin registrations. NOT `range`, `bytearray` or `memoryview`, which cannot be named as values in this frontend and so cannot be registered; `MappingView` and its three subclasses are names rather than working views. |
| `operator` | the whole public surface CPython 3.14 lists (`dir(operator)` minus the C-accelerated `_operator` re-exports) -- every arithmetic/comparison/bitwise/sequence/identity/in-place function, `attrgetter`/`itemgetter`/`methodcaller`, `countOf`/`indexOf`/`index`/`length_hint`/`call`/`truth`, and every dunder-named alias CPython assigns. Nothing refused BY NAME, but two narrow compiler-caused gaps remain: `methodcaller` cannot reach a BUILTIN type's own method (no bound-method value for one exists yet), and seven of the twelve in-place operators (all but `+ - * & \| ^`) do not consult a user class's own `__i*__` override. |
| `numbers` | the whole ABC tower -- `Number`, `Complex`, `Real`, `Rational`, `Integral` -- with CPython's own mixin methods on each; `bool`, `int` and `float` register at the same levels CPython does. NOT the per-class registration CPython gets by walking a registered class's real descendants: `bundled/abc.py`'s `ABCMeta` only walks a class's ANCESTORS on a registry miss, so `Integral.register(int)` alone answers `True` for `isinstance(3, Integral)` but `False` for `Rational`/`Real`/`Complex`/`Number`, where CPython answers `True` for all five -- `collections.abc` never hit this because it registers at every level individually. |
| `bisect` | complete -- `bisect_left`, `bisect_right`, `bisect` (an alias for `bisect_right`), `insort_left`, `insort_right`, `insort` (an alias for `insort_right`), each with `lo`, `hi` and `key=`. |
| `heapq` | `heappush`, `heappop`, `heappushpop`, `heapify`, `heapreplace`, `merge` (`key=`, `reverse=`, a genuine lazy k-way merge), `nlargest`, `nsmallest`. NOT the 3.14 max-heap family (`heapq_max`'s `heappush_max` and friends). |
| `struct` | `pack`, `unpack`, `pack_into`, `unpack_from`, `calcsize`, `iter_unpack`, `error`, and `Struct` with all six methods; byte-order prefixes `<`, `>`, `!`, `=`; type codes `x c b B h H i I l L q Q f d s ?` with repeat counts. NOT `@` (native byte order AND size/alignment -- refused BY NAME), a format with no prefix at all (CPython defaults to `@`, so it has the same problem), `e`, `p`, `n`/`N`/`P`. |
| `string` | the constants (`ascii_lowercase` through `whitespace`, in CPython's own order), `capwords`, `Template` (PEP 292, full subclass hooks), `Formatter` (every hook `str.format` uses). NOT `Template.pattern` as a live compiled-regex OBJECT -- recomputed on every call instead of cached, and refused as an `AttributeError` rather than returned as a different object each time -- or the exact wording of `Formatter.parse`'s error for a malformed field. |
| `textwrap` | `TextWrapper` with every real constructor parameter, `.wrap`/`.fill`; module functions `wrap`, `fill`, `shorten`, `dedent`, `indent`. NOT locale- or charset-aware sentence detection (CPython's own ASCII heuristic is reproduced, imperfect there too); the exact `TypeError` wording `dedent` raises for a non-`str` argument. |
| `json` | `dumps`, `loads`, `dump`, `load`, `JSONEncoder`, `JSONDecoder`, `JSONDecodeError` with real `.msg`/`.doc`/`.pos`/`.lineno`/`.colno`, and every keyword each documents. NOT `cls=` on the four top-level functions (not in this module's own signature, so it is an ordinary `TypeError` rather than a special refusal); BOM-sniffed UTF-16/32 input -- `bytes`/`bytearray` decode as UTF-8 only, which is everything this rebuild produces or a real JSON API sends. |
| `random` | `Random` with `seed`, `random`, `getrandbits`, `getstate`/`setstate`, `randrange`, `randint`, `choice`, `choices`, `shuffle`, `sample`, `uniform`, `gauss`, and the module-level bound functions off one shared instance. CPython's MT19937 core (seeding, twist, temper) is written out by hand and verified bit-for-bit against CPython's own `_random.Random`, so a seeded sequence matches exactly, call for call. |
| `fractions` | `Fraction` -- construction from `(numerator, denominator=1)`, an `int`, a `float` (exact, via `as_integer_ratio`), a `str` (CPython's own grammar), another `Fraction`, or anything with `numerator`/`denominator`; always reduced; arithmetic and comparison with `int`/`float`/`Fraction` on either side; `__hash__` agreeing with `hash(int)`/`hash(float)`. NOT `complex` arithmetic (cannot be named as a value in this frontend at all), `__format__`, `Decimal` interop, pickling hooks. `math.floor`/`ceil`/`trunc` do not consult `__floor__`/`__ceil__`/`__trunc__` for ANY class -- a pre-existing compiler gap this module found rather than caused. |
| `contextvars` | `ContextVar` (`default=`, `.get()` with and without a call-time override, `.set()`/`.reset(token)` as an exact round trip), `Token` (`.var`, `.old_value`, `Token.MISSING`, refusing a token used twice or from a different `Context`), `Context` as a full read-only mapping plus `.run()`, `copy_context()`. NOT automatic per-`asyncio.Task` context isolation -- this runtime's task scheduler steps a coroutine directly rather than copying/restoring a `Context` per step the way CPython's real `Task` does. |
| `annotationlib` | `Format` (`VALUE`/`FORWARDREF`/`STRING`, CPython's own integer values), `get_annotations` for `format=Format.VALUE` (functions, classes, plain objects, `eval_str` for any stringized annotation), `call_annotate_function`, `call_evaluate_function`, `get_annotate_from_class_namespace`, `ForwardRef`. A bare identifier resolves by lookup and anything else (`"list[int]"`, `"int \| None"`) through `eval()`. NOT `Format.FORWARDREF`/`Format.STRING` -- this runtime's `__annotate__` thunk has nothing to build them from. Writing this module found a compiler gap, since fixed: a bundled module calling `eval`, `exec` or `compile` from its own body never had `_pyrun`/`_pycompile` spliced in, because `_dependencies()` followed only `import` statements and `_Rename` did not rewrite the three names. |
| `tomllib` | `load`, `loads`, `TOMLDecodeError` -- a full TOML v1.0.0 document: dotted keys, `[table]`/`[[array of table]]` headers, inline tables, arrays, every string kind and escape, decimal/hex/octal/binary integers, floats, `inf`/`nan`, and the redefinition rules (a `[table]` cannot reopen a dotted key, an inline table or array cannot be mutated afterward, a key cannot be set twice). NOT date/time literals -- refused BY NAME with `TOMLDecodeError` rather than answered with a hand-rolled stand-in never checked against the real types. |
| `unicodedata` | `normalize` in all four forms, `category`, `combining`, `decomposition`, `is_normalized`, and the numeric properties (`decimal`, `digit`, `numeric`, including non-decimal/non-digit values like `½`). Tables generated from the reference implementation; Hangul is composed/decomposed arithmetically rather than tabulated. NOT `name`/`lookup` -- the name table is over a megabyte and answers a question about spelling, refused as a cost not worth every program paying. |
| `io` | `StringIO`/`BytesIO` (`write`, `read`, `readline`, `readlines`, `getvalue`, `seek`/`tell`, `truncate`, iteration, `close`/context manager), `open(file, mode=...)` as a real function over `objects/hostsvc.py` for `'r' 'w' 'a' 'rb' 'wb' 'ab' 'rb+'`, `SEEK_SET`/`SEEK_CUR`/`SEEK_END`; `IOBase`/`RawIOBase`/`BufferedIOBase`/`TextIOBase` as shallow markers. NOT `BytesIO.getbuffer` (needs a working `memoryview`), text update modes other than `'rb+'`, `open(newline=...)` past the default, `open(opener=...)`, `'x'` mode. One documented divergence: text-mode `open()` decodes the whole file eagerly, where CPython's `TextIOWrapper` decodes lazily on read. |
| `os` | `os.path.join`/`basename`/`dirname`/`splitext`/`normpath`/`isabs`/`exists`/`isfile`/`isdir`, `abspath` for an already-absolute path; `os.remove`/`unlink`/`rmdir`/`mkdir`/`makedirs`, `os.getenv`/`os.environ` (read-only), `os.linesep`/`sep`/`pathsep`; each mutating call raises CPython's own exception class per error code. NOT `os.getcwd`/`listdir`/`walk`/`stat`/`chdir`/`rename`/`replace` -- each refused BY NAME, since `objects/hostsvc.py` has no working directory, directory enumeration, `struct stat` or rename primitive at all. PEP 519's `fspath` and `PathLike` are both here, the second a real ABC so a class with `__fspath__` is path-like structurally and `register` works. |
| `decimal` | `Decimal` construction from `int`/`str`/`float` (exact)/a `(sign, digits, exponent)` tuple/another `Decimal`; `+ - * / // % **` computed exactly and rounded once at `getcontext().prec`; comparisons and `__hash__` agreeing with `hash(int)`/`hash(float)`; `Context`/`getcontext`/`setcontext`/`localcontext` with `prec` and all eight rounding modes; `__str__`/`__repr__` a field-for-field port of CPython's own; `quantize`, `to_integral_value`, `is_nan`/`is_infinite`/`is_zero`/`is_signed`, `as_tuple`, `as_integer_ratio`; `InvalidOperation`/`DivisionByZero` raised exactly as the default context does. NOT `Emin`/`Emax`/traps/flags/subnormals, `sqrt`/`ln`/`exp`, non-integer `**` exponents, `__format__`, `copy_sign` and friends. |
| `statistics` | `mean`, `fmean` (`weights=`), `geometric_mean`, `harmonic_mean` (`weights=`), `median`/`median_low`/`median_high`/`median_grouped`, `mode`, `multimode`, `variance`/`pvariance`, `stdev`/`pstdev`, `quantiles` (exclusive/inclusive), `covariance`, `correlation` (linear/ranked), `linear_regression`, `NormalDist` (full surface, including `pdf`/`cdf`/`inv_cdf`/`samples`/`from_samples` and arithmetic). `mean`/`variance`/`stdev`/`harmonic_mean` sum exactly through `Fraction` (CPython's own private `_sum`/`_ss`, ported) rather than a naive float loop, and `stdev`/`pstdev` finish with a correctly-rounded rational square root rather than double-rounding. NOT `Decimal` interop, `NormalDist.overlap`, `kde`/`kde_random` -- each needs a transcendental function this runtime does not have. |
| `datetime` | `timedelta` (all seven constructor keywords, CPython's exact normalization, full arithmetic/comparison), `date` (construction, `.today`/`.fromordinal`/`.fromtimestamp`, `.weekday`/`.isocalendar`, `.isoformat`/`.strftime` for the common directives, `.replace`, arithmetic, comparison including against `datetime`), `time` (construction incl. `tzinfo`/`fold`, `.isoformat` with every timespec, naive/aware comparison), `timezone` (fixed-offset only, a real `utc` singleton), `datetime` (`.now`/`.utcnow`/`.fromtimestamp`, `.combine`, `.timestamp`, `.astimezone`, `.strftime`/`.strptime`, arithmetic, comparison). NOT `zoneinfo`/real IANA timezones, `fromisoformat`, `ctime`, `__format__`, pickling; there is no host timezone lookup, so a naive `now()`/`fromtimestamp()` treats local time as UTC. |
| `traceback` | `format_exception_only`, `format_exception`, `print_exception`, `format_tb`/`print_tb`, `extract_tb`, `StackSummary`, `FrameSummary`, `TracebackException`/`.from_exception` -- exception chaining (`raise X from Y`, implicit `__context__`, `from None`), `__notes__`, and the ONE real frame `e.__traceback__` carries. NOT `format_exc`/`print_exc` -- there is no `sys.exc_info`; a bare `raise` resolves at COMPILE TIME against a lexical stack of enclosing `except` blocks rather than runtime thread state, so these are refused as fundamentally unimplementable rather than merely unwritten -- `walk_tb`/`walk_stack`/`extract_stack`/`format_stack`/`print_stack` (no call stack to walk), quoted source lines (`co_filename` is always `<compiled>`), `SyntaxError`'s multi-line format, `BaseExceptionGroup`'s tree format. |
| `time` | `time`/`time_ns`, `monotonic`/`monotonic_ns`, `perf_counter`/`perf_counter_ns`, `process_time`/`process_time_ns`, `sleep`; `struct_time` (indexed and named, comparable against a plain tuple, CPython's `repr`, and `tm_gmtoff`/`tm_zone` named but NOT indexed); `gmtime`, `localtime`, `mktime` (normalising an out-of-range field, and inverting `localtime` exactly); `asctime`, `ctime`; `strftime` over the C-locale directives including `%G`/`%V` ISO week numbers, with an unknown directive copied through as glibc does rather than raising; `strptime` including `asctime`'s default format, `%j`, the 12-hour family and the 69/68 two-digit-year pivot; `timezone`, `altzone`, `daylight`, `tzname`. THE FIRST TIER-5 MODULE, and it needed no new platform function: `objects/hostsvc.py` already declares a `time` group, so every backend answering it gets this. LOCAL TIME IS UTC -- there is no timezone database and no host call that would reach one -- so `localtime` is `gmtime`'s fields and the four constants say UTC; exact on a machine running UTC and off by a fixed offset elsewhere. NOT `thread_time`, `get_clock_info`, `clock_gettime`/`CLOCK_*`, `tzset`, or a non-C locale's `%c`/`%x`/`%X` -- each refused BY NAME. |
| `subprocess` | `run()` with `capture_output`, `stdout=PIPE`, `stderr=PIPE`, `stderr=STDOUT`, `text=`/`encoding=`, `check=`, and a list or a `str` with `shell=True`; `CompletedProcess` with `.args`/`.returncode`/`.stdout`/`.stderr`/`.check_returncode()`; `check_output`, `check_call`, `call`, `getoutput`, `getstatusoutput`; `CalledProcessError` (message copied from CPython's source, so a list command reads `Command '['false']'` exactly as there) and `SubprocessError`; `PIPE`, `STDOUT`, `DEVNULL`. A REAL CHILD PROCESS: the new `proc` host-service group is `fork`/`execvp`/`waitpid` over two pipes in the C backend and CPython's own `subprocess` in the interpreter, so `tests/stdlib/subprocess.py` passes interpreted AND as a linked binary. ONE OPERATION, not a `Popen`: a pipe you write to while the child writes back needs someone to drain the other end, and a runtime with one thread has nobody -- so the group promises the shape that is always safe (run to completion, then read) and `Popen` is refused BY NAME rather than offered as something that hangs. `shell=True` builds the shell's own argv (`["/bin/sh", "-c", cmd]`) here, where a reader can see it, so nothing below has to quote a list into a string. NOT `Popen` and everything needing a live child, `input=`, `timeout=`, `cwd=`, `env=`, `preexec_fn`; and NOT capture on Windows, where the group runs the child with inherited stdio because capturing there needs `CreateProcess` and two structs a hand-written prototype gets silently wrong. THE CAPTURE BUFFER IS 8 KiB and a child that writes more raises rather than half-reporting. |
| `math` | COMPLETE -- all 62 names CPython 3.14's `math` exports. The integer-preserving rounders (`floor`/`ceil`/`trunc`) consult PEP 3141's `__floor__`/`__ceil__`/`__trunc__`, so `Fraction` and `Decimal` round; `gcd`, `lcm`, `comb`, `perm`, `isqrt` and `factorial` are exact for integers of ANY size, promoting past a machine word rather than reading a pointer as one; `log(x)` and `log(x, base)`; the whole libm surface (`acos` through `tanh`, `erf`, `gamma`, `expm1`, `cbrt`, `nextafter`, `ulp`, `frexp`, `modf`, `ldexp`, `remainder`, `fma`); the sequence functions `fsum` (Neumaier-exact), `prod`, `dist` and `sumprod`, the last two integer-preserving; `isclose`; the constants. A BUILTIN module, not bundled -- it is `frontends/python/modules.py`'s table over `apy_math_*`, and every one of those exists three times (the C, `runtime/mathints.py`'s IR port, and `objects/host.py`), which is why `tests/stdlib/math.py` is run compiled as well as interpreted. NOT `math.nan` as a distinct NaN payload, and no `Decimal`-typed answers -- every float function answers a float. |
| `socket` | `socket(AF_INET, SOCK_STREAM)` -- `connect`, `bind`, `listen`, `accept`, `send`, `sendall`, `recv`, `recv_into`, `close`, `fileno`, `getsockname`, `connect_ex`, `detach`, `shutdown`, and the context-manager protocol; `create_connection`, `create_server`; `inet_aton`/`inet_ntoa`, `htons`/`ntohs`/`htonl`/`ntohl`, `gethostname`, `getaddrinfo` for a numeric address; `error` as an alias of `OSError` (as CPython's is), `timeout`, `gaierror`, `herror`, and the constants. A REAL TCP ROUND TRIP, over the `net` host-service group -- which was DECLARED in `objects/hostsvc.py` and implemented nowhere until now, and is implemented twice: `objects/hostsvc_host.py` calls CPython's `socket` and `C_SOURCE["net"]` calls BSD sockets (Winsock on Windows), so `tests/stdlib/socket.py` passes interpreted AND as a linked binary. Two operations were added to the group to make it usable: `host_net_port`, so a program that asks for an ephemeral port with `bind(("", 0))` can find out which one it got, and `host_net_ready`, which `select` is built from. STREAMS ONLY AND BLOCKING, which is the group's own contract: NOT `SOCK_DGRAM`/`sendto`/`recvfrom`, NOT `settimeout`/`setblocking(False)`, NOT `AF_INET6`/`AF_UNIX`/`ssl`/`makefile`, and NOT name resolution -- an address is numeric, and a name raises `gaierror` saying so, which is the exception CPython raises when a name does not resolve. |
| `select` | `select(rlist, wlist, xlist, timeout)` over sockets or bare descriptors, answering the OBJECTS that were passed; `poll()` with `register`/`modify`/`unregister`/`poll`, and `POLLIN`/`POLLOUT`/`POLLERR`/`POLLHUP`/`POLLNVAL`; `error` as `OSError`; `PIPE_BUF`. Built on the `net` group's `host_net_ready`, which asks about ONE descriptor -- a set would be a second ABI for every backend to agree on, for a loop the caller can write, and this module is that loop. A TIMEOUT OF 0 IS EXACT, which is the poll every event loop actually does; a POSITIVE timeout polls the whole set, then waits the remainder on the FIRST descriptor before polling once more, so a wait wakes on that one rather than on any of them. NOT the exceptional set (out-of-band data is a socket option the `net` group does not have), `epoll`/`kqueue`/`devpoll`, or `select` on a file -- `host_net_ready` knows about the sockets the group opened and nothing else. |
| `threading` | `Lock`, `RLock`, `Semaphore`, `BoundedSemaphore`, `Event`, `Condition`, `Barrier`, `local` and `Thread`; `current_thread`/`main_thread`/`active_count`/`get_ident`/`enumerate`, `TIMEOUT_MAX`, `BrokenBarrierError`. THE SYNCHRONIZATION PRIMITIVES ARE EXACT and are most of the module: they are state machines over a counter and a flag, so `acquire`/`release` pairing, the reentrancy count, `locked()`, the `RuntimeError` for releasing an un-acquired lock, the `ValueError` for over-releasing a bounded semaphore, `Event`'s set/clear/wait, `wait_for` on a predicate that already holds, and every one of them as a context manager all answer what CPython answers. THE SCHEDULER IS THE DIVERGENCE, and it is the part that genuinely needs the floor to grow: a `Thread` runs its target on the calling thread at `start()`, so `t.start(); t.is_alive()` is deterministically False here and a RACE in CPython, and two threads' output is sequential rather than interleaved. What holds either way -- the target running once with its arguments, `join`, `is_alive` before start and after join, an exception reported rather than propagated -- is what a program checks. An operation whose only single-threaded outcome is to wait for a thread that cannot exist (a blocking re-acquire of a `Lock`, `Condition.wait()`, a multi-party `Barrier.wait()`) raises `RuntimeError` naming the deadlock rather than hanging; CPython deadlocks there, so no correct program reaches one. NOT `Timer`, `settrace`/`setprofile`/`stack_size`, `excepthook` as a replaceable hook, or `get_native_id` -- each refused BY NAME. |
| `sys` | `exit` -- the exception it raises and the status it carries, with CPython's three cases: None or no argument is success, an int is the status, anything else is a MESSAGE printed to stderr with the status 1. `sys.exit(None)` and `sys.exit()` are one call (`args` is `()`), while `SystemExit(None)` keeps its None -- the one place the two spellings part company. An ESCAPING `SystemExit` sets the process status and prints nothing, on `uasm run` and on both compiled runtimes, which is the third way to set one beside `plat_exit` and a `main` returning an int. `stdout` and `stderr` as real writable streams on descriptors 1 and 2, replaceable by assignment -- `sys.stdout = io.StringIO()` captures what `print` writes, because `bundled.py` scans for the assignment and routes `print` through `sys._print` only in a program that makes one; `argv`, the command line the process was started with, as a real mutable list read once at import -- a compiled binary's `main` hands the words to the `env` group's statics on the way in and `uasm run` sets them from its own trailing arguments, so `./prog a b` and `uasm run prog.py a b` read the same list; NEVER EMPTY, since CPython's `argv[0]` is the empty string rather than absent when there was no script name; and DECODED WITH `surrogateescape`, which is what `os.fsdecode` is -- a command line is bytes and nothing promises they are UTF-8, so a word that is not survives as lone surrogates and re-encodes to exactly what came in rather than raising inside the module body and killing every program that imported `sys`; `breakpointhook`/`__breakpointhook__` (PEP 553), with the default refusing BY NAME since there is no debugger to enter; `audit`/`addaudithook` (PEP 578) and `monitoring` (PEP 669), both real bookkeeping over which NOTHING FIRES -- no audit event and no monitoring callback is raised by this runtime, only by the program itself, which is said in the module because silence is the failure that looks like success. Importing `sys` needs the `file` host-service group, since the two streams write through `host_file_write`, and the `env` group, since `argv` is built by module-level code that calls `host_arg_count`/`host_arg_get` whether the program reads it or not. Plus `sys.getrefcount(obj)`, answering the shadow count. EXACT against CPython 3.14 for every reference a PROGRAM makes -- a name bound and unbound, a list or set membership, a dict value, an instance attribute, and each of those removed again. ONE SHAPE READS ONE HIGHER: a function asking about its own PARAMETER, since that binding is a counted reference here and one CPython 3.14's interpreter borrows from the caller instead. INTERPRETER-ONLY. The rest of `sys` is unchanged and stays in `frontends/python/modules.py`'s native table. |
| `gc` | `collect()` -- a real cycle collector, CPython's own algorithm: subtract the references a candidate set makes to ITSELF from its members' counts, and whatever is left with nothing outside pointing at it is garbage. Finalizes it (`__del__`, weakref callbacks) and answers how many. `isenabled()`. INTERPRETER-ONLY, like `weakref` and for the same reason. NOT `enable`/`disable`, thresholds, `get_objects`/`get_referrers`/`get_referents`/`get_stats`, the debug flags, `garbage`, `freeze`/`unfreeze`, `is_tracked` -- there are no generations and no automatic runs here, so each would be a knob attached to nothing; refused BY NAME. |
| `weakref` | `ref(obj)` and `ref(obj, callback)`, calling a ref to get the referent or `None`, the interning of callback-free refs (`ref(o) is ref(o)`, as CPython interns them), `__eq__`/`__hash__`, `getweakrefcount`, and the `TypeError` for a target that cannot be weakly referenced. The callback fires when the referent's last reference drops and receives the REF, per CPython's convention. INTERPRETER-ONLY: it reads the shadow reference count below, which a compiled build does not keep, so its two primitives refuse BY NAME there. NOT `proxy`, `finalize`, `WeakValueDictionary`, `WeakKeyDictionary`, `WeakSet` -- each refused BY NAME, and each for its own reason (see the module docstring). |
| `typing` | `TypeVar`, `Generic` with a real `__class_getitem__` (`.__origin__`/`.__args__`), `NewType`, `cast`, `overload`, `Protocol` + `runtime_checkable` (structural `isinstance`), `get_type_hints` (built on `annotationlib`, with `include_extras=False` stripping `Annotated`), `ParamSpec` with `.args`/`.kwargs`, `TypeVarTuple`, `dataclass_transform`, `TypedDict` in BOTH forms, with PEP 655's `Required`/`NotRequired` and PEP 705's `ReadOnly` -- its four key sets are metaclass PROPERTIES, computed when a program asks and so after the class statement completed, which is what makes the class-based form work at all; `NamedTuple` in the FUNCTIONAL FORM only. Everything annotation-only (`Any`, `Union`, `Optional`, `Callable`, `Literal`, `Annotated`, `ParamSpec`, `final`, `override`, `get_origin`, `get_args`, ...) stays on the native `_TYPING` table, unchanged -- a bundled member always wins over the native table entry for the same name, so the two coexist without conflict. NOT class-based `NamedTuple` or attribute-only `Protocol` members, refused BY NAME: an annotation-only class-body statement never reaches a custom metaclass's `__new__` in this frontend, since the PEP 649 thunk is wired onto the class object only after the class statement completes -- and a namedtuple needs its field ORDER at construction rather than at first use, which is what `TypedDict` defers and it cannot. One divergence: a TypedDict inheriting from another answers `__annotations__` with its OWN declarations where CPython answers the merge; the key sets do merge. |
| `argparse` | `ArgumentParser` with every constructor parameter and every `add_argument` keyword; every action (`store`, `store_const`, `store_true`/`store_false`, `append`, `append_const`, `extend`, `count`, `help`, `version`, `BooleanOptionalAction`, a program's own `Action`); every `nargs` including the greedy give-one-back allocation of consecutive positionals; argument groups, mutually exclusive groups, subparsers (aliases, `deprecated`, a subparser's own defaults), `parents`, `conflict_handler='resolve'`; `parse_known_args` and both intermixed forms; abbreviations, `-xyz` clusters, `--opt=value`, negative numbers, `--`; `fromfile_prefix_chars`; `exit_on_error=False`; the five formatter classes, laid out byte for byte including the usage wrap; `FileType`; colour by CPython's `PYTHON_COLORS`/`NO_COLOR`/`FORCE_COLOR`/`TERM` rules. `suggest_on_error=True`, from `difflib`. NOT the terminal's own width -- `$COLUMNS` or 80, which is what CPython reads too whenever output is not a terminal; `FileType` given `-` for reading (`sys.stdin` is not bundled) or in a binary mode (`.buffer`). Written from the specification, not copied: positionals are allocated by a backtracking matcher over the same greedy runs CPython builds regexes from, so a program parsing its command line does not carry `re`. |
| `difflib` | `SequenceMatcher` (`isjunk`, `autojunk`, `set_seq1`/`set_seq2`, `find_longest_match` with bounds, `get_matching_blocks`, `get_opcodes`, `get_grouped_opcodes`, the three ratios, `b2j`/`bjunk`/`bpopular`), `Match`, `get_close_matches`, `Differ`/`ndiff` with intraline `?` marking (3.14's windowed synch search), `restore`, `unified_diff`, `context_diff`, `diff_bytes`, `IS_LINE_JUNK`, `IS_CHARACTER_JUNK` -- the search in CPython's order, so every ratio, opcode and `?` mark agrees, and the in-place trim `get_grouped_opcodes` makes to the cached opcodes is kept. NOT `HtmlDiff`, `SequenceMatcher[...]`. |

**Restoring is not free, and that is the point of stating coverage.** Three of
`itertools`'s seven functions were wrong in ways the old suite never asked
about, and each was a WRONG ANSWER rather than a missing one:

* `islice` accepted a step and ignored it -- `islice(xs, 1, 6, 2)` returned
  every element instead of every second
* `product` had no `repeat=`, the usual spelling of a fixed-width product
* `groupby` named its local `key`, shadowing the parameter, so passing a key
  function was accepted and dropped

None would fail a test that only asserted what the implementation already did.

## What writing one module found in the compiler

`warnings.deprecated` is one decorator. Writing it needed four compiler fixes,
and every one of them was a fault in code that had nothing to do with
`warnings` -- which is the argument for writing the library in the language
rather than in C, stated as a measurement rather than as a preference.

**A positional-only parameter was not visible to a nested function.** The
closure scope builder bound `args.args` and the two star-parameters, so a
parameter in `posonlyargs` or `kwonlyargs` was never bound in its own scope.
`def f(arg, /)` with a closure over `arg` reported `E0052: call to unknown
function 'arg'` and listed every function in the program except the one three
lines above. CPython writes `deprecated.__call__(self, arg, /)` deliberately,
so the module could not be written at all. See `analysis.parameter_names`.

**An exception class held in a VARIABLE could not be constructed on the
compiled path.** The interpreter grew that case and the C did not -- the C's
test read correctly and never fired, because `apy_exc_type` hands its
exception cell to `apy_type_of` and what a program holds for `ValueError` is
a plain type object. So `c = ValueError; c("v")` answered `ValueError() takes
no arguments` under the C backend and worked under the interpreter, for the
same source. `warnings.warn` does exactly this in `raise category(message)`.

**And a class the program wrote by subclassing an exception was worse, on
BOTH paths**: `a = AppError; a(7)` built an ordinary instance, so `str(e)`
read `<AppError object at 0x...>` and `raise e` said `exceptions must derive
from BaseException, not 'AppError'`. Written out as `AppError(7)` it had
always worked, because the frontend resolves that spelling at the call site
-- which is what kept it hidden.

**And the compiler contradicted the coverage line.** Stating what a module
covers is worth nothing if reaching past it reports something else, and it
reported one of two wrong things. `from warnings import deprecated` said
`E0083: no module named 'warnings' is available; there is no import path` --
flatly false, since the module is bundled and every other name in it worked.
`import warnings; warnings.deprecated(...)` said NOTHING at compile time and
raised `NameError: name 'warnings' is not defined` at run time. One sent the
reader after a missing module and the other after a broken import; the truth
in both cases was that the module is here and this member is not.

Both now report `E0084: module 'warnings' has no member 'deprecated'`, from
the splice -- the only pass that can, because afterwards the module is gone
and no module object is left for analysis to have an opinion about. The help
line lists what the module DOES provide, which is the coverage line restated
where someone has just run into its edge. See `bundled._no_member`.

The second and third are the divergence this project is arranged to catch:
two runtimes agreeing about the language and disagreeing about which object a
name holds. They were found by a library module and not by the conformance
suite, because the suite writes exception names out and library code takes
them as arguments.

## `pathlib` is not a module-writing problem, and this is what it is

The self-host probe ranks `pathlib` first at **30 of 79 files**, which reads
like the next module to write. It is not, and the reason is worth having
written down before someone spends a day on it.

**The compiler's own source needs the CONCRETE half.** Counted, not guessed:
`resolve` 14, `mkdir` 11, `write_text` 8, `read_text` 8, `exists` 8,
`absolute` 7, `glob` 5, `write_bytes` 4, `is_file`/`is_dir`/`rglob`/
`read_bytes` the rest. The archived module is 168 lines of the PURE half --
`PurePosixPath` and nothing that touches a disk -- and it is correct as far as
it goes.

**Shipping the pure half alone would LOWER THE PROBE'S NUMBER AND FIX
NOTHING.** Add a `Path = PurePosixPath` alias and those 30 files import
cleanly and compile; they then fail at run time on the first `read_text`. That
is precisely the "accepted and ignored" shape this rebuild exists to prevent
-- `frozen=True` in the archived `dataclasses`, `islice`'s ignored step -- and
it would be worse here because a green probe would say the constraint had
moved when it had not.

**`ctypes` LOOKED like the way in and is not, yet.** `docs/STDLIB.md` already
noted that a C library can be called with no new platform function, so
`fopen`/`fread`/`stat` through `ctypes` would give a concrete `Path` without
growing the floor. Measured instead of assumed:

    def static_call(x: f64) -> f64:
        return libm.sqrt(x)          # works, prints 3.0

    def dynamic_call(x):
        return libm.sqrt(x)          # NameError: name 'libm' is not defined

**`ctypes` resolves at COMPILE time on the STATICALLY TYPED path only.** A
bundled module is ordinary dynamic Python, so it cannot reach a C library at
all -- and `pathlib` has to be a bundled module. The same is true of any
module that would need the filesystem.

**THE FIRST HALF OF THAT IS NOW DONE.** `ctypes` is reachable from dynamic
code: `_dyn_call` recognises the shape and `dynamic._dyn_ctypes_call` emits the
same single `Op.CALL` the static path does, with an unbox before each argument
and a box after the result. The cause was narrow -- `ctypes_calls` was recorded
only by the static analyser and read only by the static lowerer, so a dynamic
call lowered as an ordinary attribute access on a name the splice had removed.

**POINTERS WORK NOW TOO.** `apy_str_bytes` hands a native call the bytes of a
`str` or `bytes`, checking the kind at run time because a dynamic value's kind
is a run-time fact -- handing a native function the address of an integer cell
is how a ctypes program corrupts memory rather than failing.

Both safety questions were CHECKED rather than assumed. NUL TERMINATION: every
producer writes one, including a slice, which reaches `apy_str_copy` like
everything else -- and the remaining C already depends on it in two hundred
places through `APY_CSTR`. LIFETIME: a literal lives in read-only program data
and an arena string is immortal, so nothing a callee keeps can be freed
underneath it. The bump-pointer arena that cannot free is exactly what makes
that answerable.

**A REAL FILESYSTEM QUERY NOW WORKS FROM DYNAMIC PYTHON**, byte-identical to
CPython:

    k32 = ctypes.CDLL("kernel32")
    k32.GetFileAttributesA.restype = ctypes.c_uint32
    k32.GetFileAttributesA.argtypes = [ctypes.c_char_p]

    def exists(path):
        return k32.GetFileAttributesA(path) != 4294967295

**BUT NOT THROUGH LIBC, AND THAT IS THE NEXT OBSTACLE.** `objects.py` includes
`<stdio.h>`, `<stdlib.h>`, `<string.h>`, `<math.h>` and `<errno.h>`, and the
backend emits `uint64_t strlen(uintptr_t)` for a symbol the IR declares --
which gcc rejects against `size_t strlen(const char *)`. Scalars are fine
(`double sqrt(double)` matches); pointers are not, because the IR has one
pointer type and C has many. So `sqrt` works and `fopen`, `strlen` and
`getenv` do not, and that is PRE-EXISTING -- the static path fails the same
way, checked.

So `pathlib` is written against the platform's own API rather than libc, or
the backend learns to CALL THROUGH A CAST instead of declaring an extern --
a change to `backends/c/emit.py`. The first works today; the second is the
general fix.

### It is written, and this is what it took

**THE FIRST ROUTE WAS TAKEN.** `bundled/pathlib.py` is the pure half restored
plus a concrete `Path`, and it passes `tests/stdlib/pathlib.py` against
CPython on the interpreter and compiled alike. The symbols are chosen for the
obstacle above rather than for tidiness: `_open`, `_read`, `_write`, `_close`
and `_lseek` from the C runtime, because `<fcntl.h>` and `<io.h>` are NOT
included and so nothing conflicts with them; and `GetFileAttributesA`,
`CreateDirectoryA`, `DeleteFileA` and `RemoveDirectoryA` from kernel32,
because `_unlink` and `_mkdir` ARE declared in MinGW's `<stdio.h>` and hit
exactly the conflict described above. `GetFileAttributesA` also answers
`is_dir` outright, where libc offers nothing short of `stat` and a struct
this frontend cannot receive. The general fix in `emit.py` is still worth
making; it is no longer what stands between here and a filesystem.

**READING NEEDED A WRITABLE BUFFER AND `bytearray` IS ONE.** Its cell is a
string's cell with the mutable flag set, so `apy_str_bytes` hands over the
object's own storage and `_read` fills it in place. Nothing new was added to
the object model to get it.

**A NULL POINTER ARGUMENT HAD TO BE ALLOWED.** `CreateDirectoryA(path, 0)` is
the ordinary spelling of "no security descriptor"; `apy_str_bytes` refused an
int, which made every native call with a NULL argument a TypeError. C's own
rule for a pointer parameter admits a null pointer constant, and now so does
this.

**THE INTERPRETER HAD TO LEARN THE SYMBOLS, and that is the part with real
weight.** A `ctypes` declaration is resolved by the LINKER, and the IR
interpreter has none -- so every concrete method trapped there with `call to
undefined function '_open'`. The interpreter is the ORACLE the C backend is
measured against, so a module it cannot execute is a module whose compiled
behaviour nothing checks. `objects/natives_host.py` binds the nine symbols through
`os`, MARSHALLING each pointer across the boundary between a host object and
interpreter memory. The write-back half of that marshalling is the subtle one:
without it a read fills a copy, and the program gets a buffer of the right
LENGTH full of zeroes with nothing raised.

**TEXT MODE TRANSLATES, so this does too.** CPython's `write_text` goes
through `open(mode="w")` and puts `\r\n` on the disk for every `\n` on
Windows; `read_text` turns them back. A program that writes with `write_text`
and reads with `read_bytes` can see the difference, so matching the oracle
means doing the translation rather than calling raw bytes "more honest".

**WHAT IS STILL OPEN is the flavour, and it is left open deliberately.** `Path`
here extends `PurePosixPath`, so `str(Path("a") / "b")` is `a/b` where CPython
on Windows answers `a\b`. Closing it means a real Windows flavour -- drive
letters, UNC names, case-insensitive comparison, both separators accepted --
and doing the SEPARATOR alone would be worse than not doing it at all:
`Path("C:/x").is_absolute()` would still answer False while the class looked
native, which is the plausible-wrong-answer this document exists to prevent.
`as_posix()` agrees on both today.

### It is off `ctypes` now, and that is what the layer was for

`objects/hostsvc.py` names the operations a backend provides -- open a file, read
the clock -- and each backend satisfies them however it can. `pathlib` was the
module it was designed around, and it is the first to use it.

**What that removed.** The `ctypes` block declared eight symbols in two
platform spellings: `_open`, `_read`, `_write`, `_close` and `_lseek` from the
MSVC C runtime, and `GetFileAttributesA`, `CreateDirectoryA`, `DeleteFileA`
and `RemoveDirectoryA` from kernel32. With them went every platform constant
-- `_O_BINARY` 32768, `_S_IWRITE` 128, `_INVALID` 4294967295 -- each of which
was this module knowing which operating system it was on.

**What it did not remove.** The separator is still `/`, because that is a
flavour question and not a host one; see above. And the concrete half is still
the same set of operations, so nothing about coverage changed.

**Why it could not have been done sooner.** The layer reached the statically
typed path first, and every bundled module is untyped Python -- so `pathlib`
could not call it until `_dyn_hostsvc_call` existed. That is the same gap
`ctypes` had before `_dyn_ctypes_call`, closed the same way.

**THE FLAVOUR IS THE NEXT THING TO WRITE, and it is worth more than it looks**,
because `cwd`, `home`, `absolute` and `resolve` are all waiting behind it and
those are the compiler's own most-wanted -- `resolve` 14 uses, `absolute` 7,
the two largest entries in the count at the top of this section. The CALL is
already reachable: `GetCurrentDirectoryA` takes a buffer and a `bytearray` is
one. It is the ANSWER that cannot be received -- `C:\Users\...` read by a
POSIX parser is a relative path whose first name is `C:`. So the four refuse
by name today and unblock together, which is one piece of work rather than
five.

What stays refused past that is `stat`, `iterdir`, `glob`, `rglob` and `walk`,
and they are blocked on something else entirely: a `struct stat` or a
`WIN32_FIND_DATA` has to come back, and a native call here returns one machine
word with no layout declared anywhere. That needs `ctypes.Structure` in
`cffi.py`, not more of this module. `open` needs an `io`.

## `re`, the keystone

Seven of the planned modules import it and three of the four things blocking
self-hosting are it. It landed with the tier-one subset complete: literals,
`.`, `[...]` with ranges and negation, every quantifier greedy and non-greedy
including `{m,n}`, alternation, capturing / non-capturing / named groups,
backreferences by number and name, lookahead both ways, `^ $ \b \B \A \Z`, the
character and category escapes, inline flags both global `(?i)` and scoped
`(?i:...)`, the flags `I M S X A U L`, and the surface `compile search match
fullmatch findall finditer sub subn split escape purge error` with `Match` and
`Pattern`.

**The implementation is a parse tree and a backtracking matcher**, not a
translation of CPython's `sre` bytecode. The bytecode form is faster and is
the wrong thing to copy here: it is an optimisation of an interpreter this
project does not have, and reading it back tells nobody what the module
means. Every node answers `match(ctx, pos, cont)` -- where the WHOLE pattern
ended, or -1 -- and that one decision is why greedy and non-greedy repetition
are four lines apart rather than two algorithms.

**Repetition has two implementations of one meaning**, and the second is not
an optimisation for its own sake. A repetition of something one character wide
that captures nothing -- `.*`, `\d+`, `[a-z]{2,4}`, which is most of them --
is scanned in a loop and backtracked by arithmetic. The general form recurses
once per repetition, so `.*` over a long subject would be as deep as the
subject is long, and this runtime's stack is not.

**What it refuses, it refuses BY NAME**: lookbehind, conditional groups,
atomic groups, possessive quantifiers, `\N{...}`, property escapes. An engine
that quietly matches the wrong thing is worse than one that says it cannot --
`(?<=a)b` read as an ordinary group matches a different language and reports
nothing. Those refusals are the one thing `tests/stdlib/re.py` structurally
cannot check, because CPython HAS those features and a differential test of
them could only ever fail; they are measured against uasm alone in
`test_stdlib.py::test_re_refuses_what_it_does_not_have`.

**Still outstanding: `warnings.filterwarnings` does not use it yet.** `_match`
there is a prefix test standing in for a regular expression, and says so. The
reason it stays is a dependency and not an oversight: `_pycompile` imports
`warnings` for one `SyntaxWarning`, so pointing `warnings` at `re` would
splice the whole engine into every program that calls `compile`, `eval` or
`exec`. The fix is for `_pycompile` to stop needing all of `warnings`, which
is its own piece of work.

## Held back, and released

`contextlib` was withheld: its `__exit__` called `gen.throw` correctly and the
cleanup still did not run when the block raised, while the same generator
throwing into the same `finally` worked in isolation. A context manager whose
cleanup silently does not run is worse than an absent one, so it stayed out
until the fault was understood rather than until it happened to pass.

**It was never a fault in this module.** Taking the shape apart one layer at a
time settled it -- a generator resumed past its last `yield`, then resumed
from a function, then from a method, then through the `with` protocol, then
through a factory, then thrown into with a `try` around the yield, then with a
`finally`. The ladder is written down in
`scratchpad/ctx-probe.py`-shaped form and every rung already agreed with
CPython, and so did the module itself once the compiler fixes above landed.

That is worth stating plainly: **withholding it was right and the diagnosis
was wrong.** The module was suspected because it was the new thing, and the
fault was in the runtime the whole time -- which is what a ladder of
one-difference-at-a-time cases is for, and what "a failure is uasm's
until shown otherwise" means applied to this suite's own output.

Nothing is held back now.

## What `enum` found in the compiler

A metaclass is the one thing an enum cannot be written without, and almost
nothing had exercised one. Seven faults, all fixed, none of them about enums:

* **`C()` ANSWERED None for any class with a metaclass.** Every metaclass
  inherits `type.__call__`, so one writing no `__call__` of its own still found
  a hook -- and `_invoke_obj` has no case for that Native the way `_invoke`
  does, so it called its body directly. That is every metaclass, `ABCMeta`
  included.
* **`len(C)`, `x in C` and `repr(C)` did not consult the metaclass.**
  `__iter__` had grown the case and the other three had not, so `for c in
  Colour` worked while `len(Colour)` said `object of type 'EnumMeta' has no
  len()` -- about a class whose metaclass plainly defines one.
* **The class statement bound a SECOND HANDLE to the class the metaclass
  returned.** `is` compares handles, so `C is Meta.seen` was False for a class
  the metaclass had just handed back, and every enum member was built against
  the class its metaclass saw rather than the one the statement bound. The same
  rule `_apy_exc_type` states for `OSError is OSError`, missed one call away.
* **`"%d" % obj` reached `object.__format__`** and reported `unsupported format
  string passed to Instance.__format__`, naming a dunder the user never wrote
  and `%` never consults -- CPython's `%d` asks `__index__`. `"{:03d}".format`
  failed separately, because Python's own formatting machinery asks the HOST
  object and `Instance` had no `__format__` at all.
* **`"the " + obj` never tried `__radd__`**, and **`a | b` on two instances
  never tried `__or__`**: both were refused by a kind rule that ran BEFORE the
  reflected dispatch that would have handled them. The dispatch was already
  correct; nothing reached it.
* **`sorted`, `min` and `max` bypassed the user's `__lt__`.** The `<` operator
  falls back to it and these called `apy_order` directly, so `Num.THREE <
  Num.ONE` worked and `sorted([Num.THREE, Num.ONE])` said `unsupported operand
  type(s) for <` -- for the same two objects.

## What `dataclasses` found in the compiler

Specified first, in five parallel slices measured against CPython, with an
adversarial critic behind them -- which is why the module was right on the
first compile rather than the fifth. What it then found was the compiler, and
three of the five are things no earlier module had reached:

* **`type(name, bases, {"__annotations__": ...})` LOST THE ANNOTATIONS**, and
  so did assigning `C.__annotations__` afterwards: both read back `{}`. The
  runtime's `__annotations__` read consulted ONLY the PEP 649 `__annotate__`
  thunk and ignored a stored entry, while `setattr` wrote one perfectly
  happily -- so the write appeared to succeed and the read never saw it. That
  is how every library that builds a class dynamically declares its fields.
  **Fixed on both runtimes** (what was stored wins, the thunk is the fallback),
  and `make_dataclass` went from refused to covered as a result. Refusing it
  was the right answer while the bug stood and the wrong thing to leave
  standing.
* **KEYWORD-ONLY PARAMETERS ARE NOT ENFORCED AT RUN TIME.** `field(3)` against
  `def field(*, ...)` is accepted and becomes the `default`; the compiler warns
  where it can see the call and lets it through. One permissive superset rather
  than a wrong answer, and stated in the module.
* **A comparison against a user object named the wrong class.** `a < b` where
  `__lt__` answered NotImplemented reported `'<' not supported between
  instances of 'Instance' and 'int'` -- `Instance` being the interpreter's own
  class, which every user object shares. The arithmetic path had rewritten that
  message and the comparison path had not, so `a + b` blamed the right types
  and `a < b` did not. Fixed.
* **`bytearray` cannot be used as a VALUE** (`E0056`), only inside an
  `isinstance` call, so a tuple of mutable types has to be written as separate
  tests.
* **`__class__` does not exist on an `int`.** CPython's generated `__eq__`
  tests `other.__class__ is self.__class__` deliberately -- it differs from
  `type()` for anything overriding `__class__` -- and comparing a dataclass to
  an integer raised `AttributeError` from inside `__eq__` where CPython answers
  NotImplemented. The module keeps CPython's spelling and falls back.

PEP 649 also caught the implementation out: `__annotations__` is never in
`cls.__dict__` under 3.14, so the `__dict__` lookup that used to be the careful
way to read a class's OWN annotations now answers `{}` for every class -- and
the first version produced dataclasses with no fields at all. The plain
attribute read is correct on both runtimes and was verified on both.

## What the adversarial critic caught that the specification missed

The five slices were commissioned to specify `dataclasses` before it was
written; a sixth agent was told to find what they had MISSED. It earned its
place three times over, and every one of the three is a WRONG ANSWER that no
happy-path test notices:

* **`asdict` deep-copies leaf values.** The recursion set is exactly
  dataclass / list / tuple / dict, and everything else goes through
  `deepcopy`. Passing leaves through gives a dict whose values are the SAME
  objects the instance holds, so mutating the snapshot mutates the original --
  which is precisely what somebody calling `asdict` is trying to avoid.
* **`replace()` must let an unknown keyword reach the constructor.** Building
  a fresh dict of known fields silently DROPPED it, so `replace(cfg,
  tiemout=5)` answered an unchanged object and nobody found out. CPython's
  message names the class, which only the constructor can do.
* **A namedtuple stays a namedtuple**, in both `asdict` and `astuple`, and is
  rebuilt SPLATTED (`NT(*[...])`) because its constructor takes one argument
  per field. Untestable here until `collections` is rebuilt, and said so in
  the module rather than left looking covered.

It also confirmed two things right by construction: a SET is not recursed into
(its members are deep-copied instances, and converting them to dicts would
make them unhashable), and `asdict`/`astuple`/`replace` take instances only,
never a class.

**And `dataclasses` then found a bug in `copy`.** `deepcopy` of a FROZEN
dataclass raised `FrozenInstanceError` from the class's own `__setattr__` --
while the copy was being CONSTRUCTED, not mutated. `_reconstruct` restores
state with `object.__setattr__` now, which is what CPython's `__dict__`
restore amounts to. Every program that freezes a dataclass and copies it hit
this.

## A bundled module's exception class leaks its mangled name

Found by restoring `copy`, which defines `class Error(Exception)`. Two
symptoms, one cause, and both are the compiler rather than the module:

    import copy
    issubclass(copy.Error, Exception)      # False; CPython says True
    try: raise copy.Error("x")
    except Exception as e: type(e).__name__   # `_asmpy_bundled_copy_Error`

A bundled module's classes are spliced under mangled names and the splice
restores `__name__` afterwards, precisely so the mangling stays invisible --
`bundled.py` says so where it emits that assignment. It did not stay invisible:
the runtime's exception hierarchy is keyed by NAME and was registered with the
mangled one, so the class sat outside the tree `issubclass` walks.

**THE `issubclass` HALF IS FIXED, on both runtimes.** The answer was not to
re-key the hierarchy -- that would conflate two bundled modules both defining
`Error`, and `copy.Error` and `re.error` are distinct classes in CPython. It
was to make the REGISTRATION FOLLOW THE RENAME: assigning `__name__` on a class
that is a registered exception re-registers it under the new spelling, keeping
BOTH, because generated code still raises through the mangled one. See
`objects_host._rename_exception` and the `__name__` arm of the C's
`apy_setattr`.

**THE PRINTED TYPE STILL LEAKS.** `type(e).__name__` for a bundled exception
answers `_asmpy_bundled_copy_Error` where CPython says `Error`, on both paths.
Catching it, `issubclass` and `isinstance` are all correct; what is wrong is
the name a traceback or a `print(type(e))` shows.

**An attempt at it was made and BACKED OUT**, and what it established is worth
more than the attempt was:

* The cell's name CANNOT simply be corrected. `except copy.Error` compiles to
  the mangled spelling and `apy_error_matches` walks the hierarchy by name, so
  renaming the cell fixes the display and breaks the catch. Any fix has to
  separate the name a program MATCHES on from the name it SHOWS.
* `type(e).__name__` is FUSED by the frontend into a single `apy_type_name`
  call (`dynamic.py`, the `ast.Attribute(attr="__name__")` case), which reads
  the cell's name string directly. That is why `t = type(e); t.__name__`
  answers `Error` and `type(e).__name__` does not, for the same object -- only
  one of them goes through the class.
* A display-name helper mapping the cell's name back through the registered
  class is the shape of the fix, and it did not work when tried: the lookup
  found nothing, which means a bundled module's exception class is not in
  `apy_exc_class_table` under the name its cells carry. THAT is the thing to
  establish first, and it was not established, so nothing was kept.

There are at least four display paths (`apy_type_name`, the `repr` and `str`
builders, `apy_kind_name`), so a partial fix makes the two runtimes disagree
with each other as well as with CPython -- which is why the attempt was
reverted whole rather than left half-applied.

`inspect` found a third, worked around rather than fixed: **`X = X` in a class
body reads the right-hand side as the class attribute being defined** instead
of the enclosing module's name, which is valid Python and how CPython's own
`inspect` writes `POSITIONAL_ONLY = POSITIONAL_ONLY`. The module uses
underscored module-level names instead and says so.

## What it cost to start

Recorded so each module restored is measured against a real baseline rather
than an impression.

| | conformance (spec+cpython) |
| --- | --- |
| before archiving | 1668/1668 (100%) |
| after archiving, with 8 modules rebuilt | 1627/1668 (97.5%) |
| with the library rebuilt | 1666/1668 (99.9%) |

**The two that remain are ABSENCES, not defects, and each is a scope
decision rather than a fix.** `pep/0749` wants `Format.STRING` from
`annotationlib`, which needs the annotation thunk to hand back SOURCE
TEXT: the compiler emits real expression code into it and keeps no
intermediate form, and there is no unparser bundled to rebuild one.
`pep/0615` wants `zoneinfo`, which needs the IANA database -- data
rather than code, and someone has to decide whether a freestanding
binary carries several hundred kilobytes of zone rules or reads them
from a host it may not have.

**The measurement before it found a bug the differential tests could not.** The
conformance shim COMPILES each case to a native binary; `tests/stdlib/` runs
them through the interpreter. `pep/0519-fspath` passed there and failed here,
because a class declaring `__slots__` answered a `member_descriptor` for every
name it did not define itself -- its METACLASS's methods included -- so
`isinstance` through any ABC died on one. `os.PathLike` has `__slots__ = ()`,
which is exactly that shape. The compiled runtime was asking whether an
INSTANCE may carry the attribute, which is true for every name the moment one
class in the chain omits `__slots__`; the question that belongs there is
whether the name is DECLARED in a `__slots__`. Fixing it exposed the mirror
image in the interpreter, which checked the class's OWN `__slots__` and missed
one inherited from a base. Both walk the chain now.

**41 cases, and not one of them is a wrong answer.** That is the number worth
having, because it says the clear cost coverage and not correctness:

| | | |
| --- | --- | --- |
| 35 | REFUSED | the program names a module that no longer exists |
| 5 | FAIL | four reach an archived module at RUN time (`sys.monitoring`, `sys.addaudithook`, `typing`, `inspect`); one was `__future__`, below |
| 1 | TIMEOUT | not a regression, see below |

The 35 refusals named what to write next, in order by how many cases each
unblocks. EVERY ONE OF THEM IS WRITTEN NOW -- `collections`,
`collections.abc`, `abc`, `dataclasses`, `enum`, `copy`, `pathlib`, `typing`,
`io`, `os`, `numbers`, `decimal`, `fractions`, `statistics`,
`datetime`/`zoneinfo`, `contextvars`, `unicodedata`, `tomllib` and
`annotationlib` -- along with the tiers that followed them.

**The TIMEOUT is a measurement artefact and was checked rather than assumed.**
`exceptions/type-raised-by-builtin-ops` calls `eval` in a loop; it passes in
isolation and takes ~18s, against a 30s default timeout, so under `-j 2` it
tipped over. The cost is the embedded compiler -- `eval` splices roughly 3,300
lines of `_pylex`, `_pyparse`, `_pyvalidate`, `_pycompile` and `_pyrun` into
the program. I suspected the `functools` import I had just added to
`warnings`, measured it at 17.4s with and 17.2s without, and it was not that.
The import is gone anyway (see `warnings._wraps`) on the smaller argument that
a module spliced into every `eval` program should not carry a dependency it
uses three lines of.

**One case found a file that had been LOST rather than archived.**
`pep/0236-future-statement` failed with `'int' object has no attribute
'optional'`, which turned out to be `bundled/__future__.py` deleted by the
clear and never copied into `archived/stdlib-prerefactor/` -- so unlike the
other twenty-seven it had no reference version, and the only copy was in git.
It is restored, with a coverage line and a test it never had. Every other
deleted file did reach the archive; that was checked, not assumed.

## `collections` was a compiler problem, and this is what it was

**A `dict` COULD NOT BE SUBCLASSED, and the failure was silent.** `class
D(dict)` produced an object that answered `isinstance(d, dict)` with True and
`len(d)` with 0 forever, had no `keys`, and printed as `<D object at 0x...>`.
The mechanism was half-built: an instance already carried a `held` value --
a real dict, list, tuple, set or str -- and `getitem`, `setitem` and `len`
consulted it. Nothing else did.

**THAT DECIDED THE SHAPE OF THE MODULE.** `defaultdict`, `Counter` and
`OrderedDict` ARE dicts in CPython: `isinstance(c, dict)` is True, `c == {...}`
compares by content, `dict(c)` copies. The archived module worked around the
gap with a `_Mapping` base and got all three wrong. Building on composition
again would have been a module that passed its own tests and disagreed with
the oracle on the three questions people actually ask, so the compiler was
fixed instead.

**WHAT THE FIX WAS**, in both runtimes and in the frontend:

| where | what |
| --- | --- |
| attribute lookup | a name the class does not define is asked of `held`, before `__getattr__` and after the class body, which is where the MRO would put it |
| method dispatch | the frontend chose between a user lookup and a direct builtin call by asking `apy_is_instance`; it now asks whether the CLASS defines the name, and hands the builtin branch the held value |
| iteration | four separate entry points -- the lazy cursor, the eager walk, the index-walk length and its element accessor -- and the last two had to move together or the walk read the right COUNT out of the wrong object |
| construction | a class extending a builtin inherits its constructor, and `super().__new__(cls, x)` past a builtin base builds one; a tuple's contents cannot be set afterwards, so that is the only place a `namedtuple` can be filled |
| operators | `+`, `==` and the orderings fall back to the builtin when neither side wrote the dunder -- and the HOST's own dunders needed it too, because `sorted` compares with Python's `<` rather than through the IR |
| `repr` and `hash` | a subclass with no `__repr__` prints its content, and two equal namedtuples hash alike |

**ONE OF THOSE FIXES FOUND A BUG THAT WAS ALREADY THERE.**
`_dyn_method_either` -- the two-way call shape -- DROPPED KEYWORD ARGUMENTS.
It was reached only on a name collision between a user method and a builtin
one, and no such call in the corpus passed a keyword, so nothing showed it.
Widening the test to cover builtin-extending classes brought ordinary calls
through it and `od.popitem(last=False)` popped the other end. A dropped
keyword is a wrong answer with nothing to mark it, which is why it survived.

**WHAT IS STILL A LIMIT**, found by writing the module against it:

- `super().__new__` past a builtin base works; `SomeClass.__new__` reached as
  an attribute does not, so `namedtuple` is built on a module-level base that
  every generated class extends rather than on a per-class `__new__`.
- A class NESTED IN A FUNCTION cannot use zero-argument `super()`: the
  frontend emits a reference to a `__class__` global it never creates, and the
  compile fails with `global_addr of unknown global`. That is a real bug with
  a name; it is why the base is at module level.
- A `str` does not answer its method names through `getattr`, so a starred
  forward (`self.data.replace(a, b, *rest)`) cannot reach them. `UserString`
  writes its optional parameters out instead.
- `range`, `bytearray`, `memoryview`, `complex` and `NoneType` are not values,
  so `collections.abc` cannot register them.
