"""The object runtime, in C: the preamble, errors, source positions and construction.

ONE PART OF ONE TRANSLATION UNIT. `c/__init__.py` concatenates
these in order and the result is the file it always was, so a
definition here may rely on anything in an earlier part and
nothing in a later one. Sections, in order:
  * errors
  * source positions
  * construction
"""

C = r"""/* uasm dynamic object runtime. Generated -- edit objects/csource.py. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#define APY_API @APY_API@

typedef struct apy_obj apy_obj;

/* The public type is `uintptr_t`, not `apy_obj *`, because that is what the
   IR's `ptr` is and what every backend emits for it -- the C backend types a
   pointer register `uintptr_t` and generates its `extern` declarations from
   the IR signature. A public signature taking `apy_obj *` would be
   ABI-identical and textually disagree with the declaration the backend
   writes, which newer C compilers reject outright. `O()` and `V()` cross the
   two, and nothing outside this file sees an `apy_obj`. */
typedef uintptr_t apy_value;
#define O(v) ((apy_obj *)(v))
#define V(p) ((apy_value)(p))

/* The runtime builtins a FUNC_K can stand for. `object`'s defaults, so a
   `super()` whose base chain has run out answers with the same behaviour
   CPython's `object` gives -- and `type`'s two, which are what a metaclass
   reaches through `super().__new__(mcls, name, bases, ns)`. */
enum {
    APY_NAT_NONE = 0,
    APY_NAT_INIT, APY_NAT_NEW, APY_NAT_REPR, APY_NAT_STR, APY_NAT_EQ,
    APY_NAT_NE, APY_NAT_HASH, APY_NAT_GETATTR, APY_NAT_SETATTR,
    APY_NAT_DELATTR,
    /* `type.__new__(mcls, name, bases, ns)` and `type.__init__`, which is a
       no-op -- the class is complete when `__new__` returns it. */
    APY_NAT_TYPE_NEW, APY_NAT_TYPE_INIT, APY_NAT_TYPE_CALL,
    APY_NAT_DESCR_GET, APY_NAT_DESCR_SET, APY_NAT_DESCR_DEL,
    APY_NAT_KIND, APY_NAT_HAS_DEFAULT,
    /* `object.__init_subclass__` -- a no-op that EXISTS, which is what
       `super().__init_subclass__(**kw)` at the end of a user hook needs. */
    APY_NAT_INIT_SUBCLASS,
    /* A generator's own three. They are dispatched by NAME at the call site,
       so nothing needed a VALUE for them -- until a program asked
       `hasattr(g, "close")`, which every duck-typed consumer does. */
    APY_NAT_GEN_SEND, APY_NAT_GEN_THROW, APY_NAT_EXC_INIT,
    APY_NAT_BUILTIN_INIT, APY_NAT_BUILTIN_NEW, APY_NAT_POSITIONS,
    APY_NAT_TASK_CANCEL, APY_NAT_TASK_RESULT, APY_NAT_TASK_DONE,
    APY_NAT_TASK_CANCELLED, APY_NAT_TG_ENTER, APY_NAT_TG_EXIT,
    APY_NAT_TG_CREATE,
    /* AND AN ASYNC GENERATOR'S THREE, which are not the same three: CPython
       gives `asend`, `athrow` and `aclose` to an async generator and `send`,
       `throw` and `close` to a coroutine and a plain one, and `dir()` over
       each says exactly that. Each answers an AWAITABLE rather than doing
       the work, which is what makes `await a.aclose()` the spelling. */
    APY_NAT_AGEN_SEND, APY_NAT_AGEN_THROW, APY_NAT_AGEN_CLOSE,
    APY_NAT_GEN_CLOSE,
    /* THE SIX `object` HANDS DOWN THAT ARE NOT THE RECEIVER'S. The four
       orderings and `__subclasshook__` answer NotImplemented whatever they
       are given -- `object` defines no order and vouches for no subclass --
       and `__format__` takes an EMPTY spec only. They cannot share
       `APY_NAT_KIND` with the rest, because that selector dispatches to the
       RECEIVER's kind: routed there, `object.__lt__(1, 2)` became int's `<`
       and answered True where CPython answers NotImplemented. See
       `apy_object_default`.

       LAST ON PURPOSE. `runtime/makers.py` writes these numbers out by hand
       -- `apy_nat_kind` is 17 there because it is 17 here -- so a member
       inserted in the middle silently renumbers every one after it, and
       `apy_nat_count` (38) sizes the IR's copy of the cache this enum sizes
       below. Appending costs one slot and moves nothing. */
    APY_NAT_OBJ_ONLY
};

enum {
    APY_NONE_K = 0, APY_BOOL_K, APY_INT_K, APY_FLOAT_K, APY_STR_K,
    APY_LIST_K, APY_TUPLE_K, APY_DICT_K, APY_EXC_K, APY_SET_K, APY_FROZEN_K,
    APY_FUNC_K, APY_CELL_K, APY_TYPE_K, APY_INST_K, APY_SUPER_K, APY_BIG_K,
    /* `bytes`. Shares the str layout -- a pointer and a length -- because
       that is exactly what it is; what differs is that its length and its
       indexing are in BYTES rather than characters, its repr wears a `b`
       and escapes non-printables, and indexing yields an int. Sharing the
       layout means `apy_str_take` and the slicing arithmetic work on it
       unchanged. */
    APY_BYTES_K,
    /* `complex`. Two doubles, and NOT part of the numeric tower's ordering:
       int/float/bool compare with `<` and complex does not, which is the one
       rule that keeps it from being just a third float. */
    APY_COMPLEX_K,
    /* An ITERATOR: what `iter(x)` returns and `next(it)` advances.

       A container and a position, because iteration here is by INDEX -- there
       is no iterator protocol underneath, so an iterator is a cursor over
       something already indexable rather than a thing with a `__next__`. That
       is a real limitation and a visible one: an iterator over a list sees
       later mutations of it, where CPython's does too, but a generator has no
       container to be a cursor over and is why `yield` needs more than this.
     */
    APY_ITER_K,
    /* `...`. A SINGLETON, like None and the two bools, because `... is
       Ellipsis` is the test programs write and a fresh cell per literal would
       answer False. It has no payload: being itself is all it does. */
    APY_ELLIPSIS_K,
    /* `NotImplemented`. A SINGLETON, because `x is NotImplemented` is the
       test programs write, and because what it means -- "I cannot answer
       this; ask the other operand" -- is a signal rather than a value. */
    APY_NOTIMPL_K,
    /* A SUSPENDED FUNCTION. Its locals live here rather than in registers,
       because a register does not survive the return a `yield` compiles to --
       see `apy_gen_new`. */
    APY_GEN_K,
    /* `a:b:c` AS A VALUE. Built only where one is needed as an object --
       a user `__getitem__` receives it, and `c[1:2, 3]` puts one in a tuple.
       Slicing a list or a str still goes straight through `apy_slice` without
       allocating anything, because that is the common case by a mile. */
    APY_SLICE_K,
    /* `list[int]`, `dict[str, int]` -- a PARAMETERISED type. It is not a type
       itself: nothing is instantiated from one here, and what a program does
       with it is print it or pass it to an annotation. Kept as the origin and
       the arguments so the repr can be rebuilt exactly. */
    APY_ALIAS_K,
    /* `d.keys()`, `d.values()`, `d.items()`. A WINDOW ON THE DICT, not a copy
       of it: `ks = d.keys()` then `d['b'] = 2` and `len(ks)` is 2. A snapshot
       is the obvious implementation and it is wrong in a way that only shows
       up after the dict changes, which is exactly when a program is relying
       on the view being live. */
    APY_VIEW_K,
    /* A DESCRIPTOR the runtime supplies: `property`, `classmethod`,
       `staticmethod`. One kind for the three because they differ only in what
       reading one through an instance does -- see `apy_descr_get`. A user
       class defining `__get__` is a descriptor too and is NOT this kind; it
       is an ordinary instance, recognised by having the method. */
    APY_PROP_K,
    /* `memoryview(b)`. A WINDOW, like the dict views and for the same reason:
       `mv[0] = 122` has to be seen by the bytearray it was taken from, and a
       copy would swallow the write. Kept as the buffer it looks at plus an
       offset, a length and a STRIDE -- the stride is what lets `mv[::-1]`
       stay a view rather than becoming the copy that would break the
       write-through. */
    APY_MVIEW_K,
    /* `range(0, 10, 2)`. A LAZY SEQUENCE: three numbers, not the elements.
       It was materialised into a list, so `type(range(3)).__name__` said
       `list` and `range(10**9)` would have allocated a billion of them.
       Everything a program does with one -- length, index, slice, membership,
       equality -- is arithmetic on the three. */
    APY_RANGE_K
};

/* The three views a dict offers. The numbers match `DICT_PARTS` in the
   frontend, which is where the selector is chosen -- two tables that must
   agree, and the agreement is what makes `d.items()` items rather than keys. */
enum { APY_PART_KEYS = 0, APY_PART_VALUES = 1, APY_PART_ITEMS = 2 };

/* Which of the three a descriptor cell is. `property` is the only DATA one --
   it defines `__set__` -- which is why it alone beats the instance dict. */
enum { APY_PROP_PROPERTY, APY_PROP_CLASSMETHOD, APY_PROP_STATICMETHOD };

/* WHAT A CURSOR DOES on the way. A plain `iter(x)` walks; the rest apply
   something as they go, which is what makes `map(f, xs)` lazy -- `f` runs
   when the result is walked, not when it is made. */
enum apy_iter_mode {
    APY_IT_PLAIN = 0,
    APY_IT_MAP,
    APY_IT_FILTER,
    APY_IT_ENUMERATE,
    APY_IT_ZIP,
    /* `reversed(x)`: the same index walk, counting DOWN. A mode rather than
       an eagerly built list, because CPython's is lazy -- `reversed(xs)` is
       an iterator, not a sequence, so it has no length, cannot be indexed
       and cannot be walked twice. Building the list answered all three
       wrongly and cost the whole copy. */
    APY_IT_REV,
    /* `iter(f, sentinel)`: call `f` on every step until it answers the
       sentinel. A MODE and not a list drained at construction, because
       CPython's is lazy -- the calls happen as the walk asks for them, which
       is the difference between `for v in iter(f, s): break` leaving after
       one call and never returning at all. `fn` is the callable and `src`
       the sentinel; `i` is 0 until the sentinel arrives and 1 after, because
       CPython drops the callable at that moment and a second `next()` must
       not call again. */
    APY_IT_CALL
};

/* WHAT A CURSOR IS NAMED AFTER, in `v.it.named`: the KIND of what it was
   made from, or `APY_IT_VIEWED + part` for a dict view -- whose items are
   materialised at construction, so the view itself is gone by the time
   anything asks. Recorded because CPython names an iterator after its
   source (`list_iterator`, `dict_valueiterator`) and neither the cursor's
   mode nor its `src` can still say which: a drained `map` becomes a plain
   cursor over a list, and so does `iter(d.values())`.

   Past every kind, so one field carries both without a second. */
enum { APY_IT_VIEWED = 64, APY_IT_CALLABLE = 68, APY_IT_REVOF = 128 };

struct apy_obj {
    int kind;
    union {
        int64_t i;          /* int, and bool as 0/1 */
        double  f;
        /* An integer too big for `i`. Sign and magnitude, base 2**32, limbs
           little-endian. A big NEVER holds a value that fits `i` -- see the
           arbitrary-precision section for why that invariant is the whole
           design and not an optimisation. */
        struct { uint32_t *limb; int64_t n; int neg; } big;
        /* str, and bytes. `mut` is the whole of `bytearray`: the same
           layout with the buffer writable, so every length, index, slice,
           comparison and method the bytes kind has is already the
           bytearray's -- which is also what CPython says, since
           `b"a" == bytearray(b"a")` is True. What differs is the repr, the
           name, that it can be assigned into, and that it cannot be hashed.
           A separate kind would have had to be taught all thirty of the
           shared paths to get the four that differ. */
        struct { const char *p; int64_t n; int mut; } s;
        struct { double re, im; } z;             /* complex */
        struct { int64_t start, stop, step; } rg;   /* range */
        /* A CURSOR. `src` is what it walks and `i` where it is; `fn` and
           `mode` are what it does on the way. `map`, `filter`, `enumerate`
           and `zip` are cursors rather than lists because they are LAZY --
           `map(f, xs)` calls `f` when the result is walked, not when it is
           made, and a program with a side-effecting `f` can tell. */
        struct {
            apy_value src, fn;
            int64_t i;
            /* THE SIZE THE WALK STARTED WITH, for a dict. Growing or
               shrinking one while iterating it rehashes the table and the
               walk would silently skip or repeat entries, so CPython refuses
               -- and refusing needs the original size to compare against. */
            int64_t n0;
            int mode;
            /* WHAT THIS CURSOR IS NAMED AFTER -- see `apy_iter_mode`. */
            int named;
        } it;
        /* A GENERATOR: the step function, the frame its locals live in, where
           to resume, and what `send` last passed in. */
        struct {
            apy_value step, sent, *slots;
            /* What a LENGTH QUERY drained. `sum(g)` and `sorted(g)` walk by
               index and an index walk needs a length, so the first one to ask
               consumes the generator into a list and every later index reads
               from it -- which keeps the two questions answering about the
               same elements. */
            apy_value cache;
            /* WHAT `return` GAVE, waiting to become `StopIteration.value`.
               A generator's return value is not the call's result -- the call
               already returned the generator -- so this is the only place it
               can live between the `return` and the exception that carries
               it. */
            apy_value result;
            /* An exception to raise AT THE SUSPENSION POINT, from `throw` or
               `close`. Delivered by the resume block rather than by the
               caller, so a `try` inside the generator body catches it -- which
               is the whole difference between `throw` and raising at the call
               site. */
            apy_value pending;
            int64_t n, state;
            int running;
            /* WHETHER THIS CAME FROM `async def`. The machinery is identical
               -- a coroutine is a generator with a frame and a step -- so
               this decides only what the object calls itself, which is how a
               program that awaits a generator or iterates a coroutine finds
               out it made a mistake. */
            int coro;
            /* WHICH BUILT-IN COROUTINE THIS IS, or 0 for an ordinary one
               lowered from an `async def`. `sleep` and `gather` have no
               Python body and so no step function to re-enter; they are
               driven directly by `apy_await_step`, which needs to know which
               of them it is holding. */
            int builtin;
            /* WHETHER THIS IS AN `async def` CONTAINING `yield` -- an async
               generator, which is neither a coroutine nor a plain generator.
               It is driven by `async for`, and awaiting one is an error. */
            int agen;
            /* WHEN A `sleep` WANTS TO WAKE, on the virtual clock, and
               what `wait_for` will give up at. Unused by everything else. */
            double deadline;
            /* CANCELLATION, for a task: 0 none, 1 asked for, 2 delivered.
               Three states and not two, because `cancel()` does not raise --
               it asks, and the exception is raised at the task's next
               suspension point, which is where a `try` around the `await`
               inside it can catch it. */
            int cancel;
            /* THE SUB-ITERATOR A `yield from` IS DELEGATING TO, or 0 when
               this generator is not inside one. `g.gi_yieldfrom` is the only
               way a program can see the delegation from outside -- an
               `asyncio`-shaped scheduler reads it to find what a coroutine is
               really waiting on -- and the delegate otherwise lives in a
               frame slot the outside cannot name. Set and cleared by the
               lowered loop; see `_dyn_yield_from`. */
            apy_value yieldfrom;
            /* THE `def`'S OWN SIGNATURE, as a FUNC that describes it and is
               never called. `g.gi_code` is the code of the function the
               generator came from -- in CPython literally the same object --
               and the STEP cannot carry it: the step is an ordinary compiled
               function of one argument and `apy_gen_step_of` reaches it
               through `apy_call`, so its arity is the number of arguments
               that call passes and not the number the `def` declared.

               ONE PER `def` AND SHARED, built by the module's literal filler
               and handed over here, so constructing a generator is a load
               rather than a second function object. See `_dyn_generator`. */
            apy_value sig;
            /* A COROUTINE_WRAPPER, which `c.__await__()` answers and nothing
               else builds. It has no body of its own: every `send`, `throw`
               and `close` goes to the coroutine in `yieldfrom`, which is
               what `yieldfrom` has always meant. CPython has a separate type
               for it and a program can see the difference -- `type(
               c.__await__()).__name__` is `coroutine_wrapper`, it is not the
               coroutine, and `dir()` of it holds three names. */
            int wrapper;
        } g;
        /* list, tuple, set and frozenset share ONE layout. They differ in
           what is allowed -- a tuple never grows, a set never holds two
           equal elements -- and in how they print, and in nothing else; two
           layouts would mean two copies of indexing, repr and equality. */
        struct { apy_value *items; int64_t n, cap; } q;
        /* `ro` MARKS A mappingproxy -- a dict a program may read and not
           write, which is what `C.__dict__` answers. A FLAG ON THE DICT
           rather than a kind of its own: CPython's proxy is LIVE over the
           mapping it wraps (`p = C.__dict__; C.x = 1` puts `x` in `p`), so
           the class's own dict is what has to be handed out, and everything
           that READS a dict must go on working unchanged. Only the writes,
           the kind name and the repr read this. */
        struct { apy_value *keys, *vals; int64_t n, cap; int ro; } d;
        /* An exception: the type's NAME (a static string from the
           hierarchy table, so comparing types is comparing text) and
           the single argument `str(e)` returns. */
        /* An exception: the type's NAME, the single argument `str(e)`
           returns, and WHETHER there was one. The flag is not redundant with
           `arg`: `E()` and `E(None)` both leave `arg` holding None, and
           `e.args` must be `()` for the first and `(None,)` for the second.
           Without it the second lost its argument. */
        struct {
            const char *name; apy_value arg; int has_arg;
            /* EVERY argument, when there is more than one. `OSError(2, "No
               such file")` carries both and reads them back as `errno` and
               `strerror`; one field could hold only the first, so a program
               that passed two silently lost one. 0 when a single argument (or
               none) was given, which `arg` already covers. */
            apy_value argv;
            /* CHAINING. `__context__` is whatever was being handled when this
               one was raised, set implicitly; `__cause__` is what `raise X
               from Y` said, set explicitly. They are separate because
               `raise ... from None` SUPPRESSES the context without having a
               cause, which one field could not express -- and that
               suppression is the whole of PEP 409.

               `notes` is what `add_note` appends, a list or 0. */
            apy_value context, cause, notes;
            /* THE CLASS THE PROGRAM WROTE, and this exception's own
               attributes -- both 0 for the exceptions the runtime raises
               itself, which have neither.

               `raise` and `except` match on the NAME and always did; what
               these add is everything a class body puts on an exception that
               a name alone cannot hold. `self.code = 404` goes in `dict` and
               `def summary(self)` is found through `cls`, so a user exception
               is an ordinary object in every way except how it is caught. */
            apy_value dict, cls;
            /* WHERE IT CAME FROM, as an index into the position table, or -1
               when nothing was recorded -- which is every program that never
               asks about a traceback. See `apy_pos_add`. */
            int64_t pos;
            /* THE EXCEPTIONS AN `ExceptionGroup` CARRIES, or 0 for an
               ordinary one. A group is an exception like any other -- it has
               a name and a message and propagates the same way -- and this is
               the only thing that distinguishes it. */
            apy_value subs;
            int suppress;
            /* WHETHER `arg` is the already-formatted message rather than the
               object the program raised. `apy_error_value` rebuilds an Exc
               from the message text when the error came from an OPERATION,
               and that text is already `'k'` for a KeyError -- so the
               KeyError repr rule below must not apply it a second time. */
            int rendered;
        } e;
        /* A CALLABLE. `code` is the compiled entry, from the IR's FUNC_ADDR;
           `arity` is the count of DECLARED Python parameters, which includes
           `self` for a method. `cells` are the closure boxes this particular
           function object captured -- two closures over one variable share
           the box, which is what makes them see each other's writes.
           `bound` is the receiver of a bound method, or 0.

           DEFAULTS AND `*rest` LIVE IN THE VALUE, not at the call site. A
           direct call resolves both at compile time and never looks here; a
           call through `apy_call` cannot, because it reaches a function whose
           definition the caller never saw -- and every method call is one of
           those. Without this a method could not have a default, which rules
           out `def __init__(self, v=0)`. */
        struct {
            uintptr_t code; int64_t arity; apy_value name;
            apy_value *cells; int64_t ncells; apy_value bound;
            apy_value *defaults; int64_t ndefaults; int vararg;
            /* How many of the trailing defaults are the keyword-only
               parameters'. See `apy_func_kwdefaults`. */
            int nkwdefault;
            /* WHETHER the last declared parameter is `**kw`. A flag and not a
               name, because the only thing any caller needs to know is where
               to put the keywords it could not place -- and that the slot is
               filled even when there are none. */
            int kwarg;
            /* How many TRAILING declared parameters are KEYWORD-ONLY. A
               position cannot reach one, so positional filling stops short of
               them and only a name arrives -- the whole of `*` in a
               signature. Zero for an ordinary function, so the arithmetic
               below is unchanged for every ordinary call. */
            int kwonly;
            /* How many LEADING declared parameters are POSITIONAL-ONLY. Their
               names ARE recorded -- so a call that passes one by keyword can
               be told which mistake it made -- but the matcher skips them,
               which is the whole of `/`. */
            int posonly;
            /* WHETHER THIS WAS REACHED OFF A TYPE RATHER THAN OFF A VALUE.
               `list.append` is a `method_descriptor` in CPython and
               `list.__len__` a `wrapper_descriptor`; both print as `<method
               'append' of 'list' objects>` rather than as a function, both
               qualify their name with the type, and neither has a
               `__self__`. What says so is not the SHAPE -- a user method
               reached as `C.m` is an ordinary function -- so the type that
               handed it over marks it, and the qualname carries which type.
               Sits in the padding after `posonly`, so no later field moves. */
            int descr;
            /* PARAMETER NAMES, in declaration order and including `self`.
               Only a KEYWORD argument needs them, and only a call through a
               value does -- a direct call matches names at compile time. They
               live here because that is the only place the name is still
               known: `apy_call` reaches a function whose `def` the caller
               never saw, so `C(1, swallow=True)` had nowhere to look and
               silently took the default instead. NULL when the frontend did
               not record them, which a keyword call reports rather than
               guesses at. */
            apy_value *pnames;
            /* The DOCSTRING, or 0. Recorded because `f.__doc__` is the one
               piece of a `def` a program routinely reads back, and the
               frontend would otherwise drop it as the bare string statement
               it is. */
            apy_value doc;
            /* WHICH RUNTIME BUILTIN THIS IS, or 0 for an ordinary
               compiled function. Every callable in this runtime used to be
               compiled code, which meant `super().__init__()` on a class with
               no base had nothing to hand back -- the default bodies exist
               (`apy_default_init` and friends) but no VALUE named them. A
               native carries a selector instead of a code pointer, and
               `apy_invoke` dispatches on it. */
            int native;
            /* WHETHER THIS IS A BUILTIN TYPE NAME used as a value --
               `int`, `str`, `list`. It stays callable, because that is what
               `map(str, xs)` needs; the flag is what makes `print(int)` say
               `<class 'int'>` rather than naming a function. */
            int is_type;
            /* WHETHER THIS IS A BUILTIN reached as a value -- `print`, `len`.
               A synthesised thunk is an ordinary compiled function here, so
               `type(print).__name__` said `function` where CPython says
               `builtin_function_or_method`. */
            int builtin;
            /* WHETHER CALLING THIS BUILDS A COROUTINE -- an `async def`.
               Recorded on the function and not only on what it returns,
               because `inspect.iscoroutinefunction(f)` asks before anything
               has been called. */
            int coro;
            /* PEP 3155: the QUALIFIED name -- `C.m`, `outer.<locals>.inner`
               -- or 0, in which case `__qualname__` is the plain name. The
               frontend's own key for a function is already in exactly this
               spelling, which is why it can simply be handed over. */
            apy_value qualname;
            /* PEP 649: the THUNK that builds `__annotations__`, or 0 for a
               function with none. Lazy rather than a dict evaluated at the
               `def`, because an annotation may name something that does not
               exist yet -- `def f(x: Undefined)` is a legal definition and
               only READING its annotations is an error. */
            apy_value annotate;
            /* ARBITRARY ATTRIBUTES SET ON THE FUNCTION ITSELF, or 0 until one
               is. A Python function is an object a program may hang anything
               on -- `f.__override__ = True` is a decorator doing exactly that
               -- and without somewhere to put it every such assignment was an
               AttributeError. Created on first write, so an ordinary `def`
               still allocates nothing extra. */
            apy_value dict;
            /* WHERE THE `def` WAS WRITTEN -- `fractions` for a spliced
               definition -- or 0, which reads as `__main__`. Only a spliced
               `def` is given one, because a program's own module has no
               other name and paying a call per `def` to say so would be a
               call per `def`. A builtin thunk answers `builtins` from its
               flag instead and never looks here. */
            apy_value module;
        } fn;
        /* A runtime descriptor -- `property`, `classmethod`, `staticmethod`.
           `get` holds the getter (or the wrapped function, for the other
           two) and `set` the setter, 0 when there is none. */
        struct { apy_value get, set, del_; int kind; } p;
        /* `slice(start, stop, step)`. Each is a VALUE and each may be None --
           an omitted bound is not the same as any number, which is the whole
           reason the three are kept rather than resolved to indices here. */
        struct { apy_value start, stop, step; } sl;
        /* `list[int]`: what was subscripted, and with what.

           `unpacked` IS PEP 646's `*list[int]`, and it exists because
           ITERATING an alias is what CPython does with one: `iter(list[int])`
           yields exactly one item, the unpacked form of itself. That is the
           whole reason `1 in list[int]` answers False rather than raising --
           membership walks that one item and does not match. Without the
           flag the unpacked form would be equal to the plain one, and
           `x in list[int]` would have had to answer False by fiat. */
        struct { apy_value origin, args; int unpacked; } ga;
        /* What a memoryview looks at, and where. `step` is signed: -1 is
           `mv[::-1]`, which reads the same bytes backwards. */
        /* `ro` IS NOT DERIVED FROM THE SOURCE. `m.toreadonly()` hands out
           a view that cannot be written over a buffer that can, so the
           answer to `m.readonly` has to live on the VIEW. A RELEASED view
           is one whose `src` is 0, which needs no field of its own and is
           what every reader tests. */
        /* `wide` AND `fmt` ARE WHAT `cast` CHANGES: one element is `wide`
           bytes and reads as the format character says. Both are 1 and 'B'
           for a fresh view, which is every source this runtime can wrap. */
        struct { apy_value src; int64_t off, n, step, ro, wide, fmt; } mv;
        /* The dict a view looks at, and WHICH of the three it is. */
        struct { apy_value dict; int part; } vw;
        /* One closure variable's box. A captured local lives HERE instead of
           in a register, so the enclosing function and every closure over it
           read and write the same storage. */
        struct { apy_value slot; } cell;
        /* A CLASS: its name as a str value (so `__name__` is just a field),
           its single base or 0, and a dict of everything its body bound --
           methods and class attributes alike. */
        /* A CLASS: its name as a str value (so `__name__` is just a
           field), its single base or 0, a dict of everything its body bound,
           and the METACLASS that made it -- 0 for an ordinary `class`, which
           reads as `type`. `type(C)` answers the metaclass, and a metaclass's
           `__instancecheck__` is reached through it. */
        /* `base` is the FIRST base and `bases` all of them; `mro` is the
           C3 linearisation attribute lookup walks, starting with the
           class itself. `mro` is 0 for a class built before it could be
           computed, and the base chain is the answer there. */
        /* `builtin` is the KIND a class extends -- `class D(dict)` --
           or 0. Recorded rather than derived, because the base chain holds
           only classes and a builtin is not one. */
        /* `qual` is PEP 3155's `__qualname__` -- `mk.<locals>.D` for a
           class written inside a function, `C.Inner` for one written inside
           another class -- or 0 for a class whose qualname IS its name,
           which is every class written at module level. A FIELD OF ITS OWN
           and not the name with dots in it: CPython's error messages say
           `'D' object has no attribute` for a nested class and only its
           reprs say `mk.<locals>.D`, so the two spellings have to be
           separable. See `apy_type_qualname`. */
        struct { apy_value name, base, dict, meta, bases, mro;
                 int builtin; apy_value qual; } t;
        /* An INSTANCE: what class made it, and its own attribute dict. */
        /* `held` is the BUILTIN VALUE an instance of a builtin-extending
           class carries -- a real list, dict or tuple. 0 for an ordinary
           instance. What a class defines wins; what it does not is answered
           by delegating here, which is the whole of `class D(dict)`. */
        struct { apy_value cls, dict, held; } o;
        /* What `super()` evaluates to: the class the calling method was
           DEFINED in, plus the receiver. Attribute lookup starts at that
           class's base, so a two-level hierarchy does not recurse forever. */
        struct { apy_value from, self; } sup;
    } v;
};

/* --- errors ----------------------------------------------------------- */
/* One slot, not a stack: an operation that fails is followed by a check
   before another can fail, because the frontend inserts the check. */
/* THE STORAGE IS IR'S NOW (`runtime/errstate.py`), and these three names
   stay so that the forty-odd places that read them do not have to change.
   `apy_err_type` and `apy_err_value` are lvalue macros over the two words the
   IR reserves; `apy_err_msg` is the buffer itself, which is why its capacity
   needs a name of its own -- `sizeof` on a pointer is 8, and the three
   `snprintf` calls below would have truncated every message to seven
   characters had they kept it. */
#define APY_ERR_MSG_CAP 256
/* THE STORAGE IS REACHED THROUGH TWO FUNCTIONS rather than named directly,
   which is what lets it move. Both are ordinary exports with C bodies, so the
   runtime still stands alone when nothing is ported; `runtime/errstate.py`
   replaces them, and then the same forty reads and writes land on the IR's
   reservation instead. Reaching for the statics by name -- as `apy_error_type`
   once did for the None cell -- is the mistake this shape prevents. */
static uintptr_t apy_err_slots_c[2];
static char apy_err_text_c[APY_ERR_MSG_CAP];
APY_API apy_value apy_err_slots(void) { return (apy_value)apy_err_slots_c; }
APY_API apy_value apy_err_text(void) { return (apy_value)apy_err_text_c; }
#define apy_err_type  (*(const char **)apy_err_slots())
#define apy_err_value (*(apy_value *)(apy_err_slots() + sizeof(uintptr_t)))
#define apy_err_msg   ((char *)apy_err_text())
/* The exception OBJECT, when one was raised rather than an operation failing.
   The type and the message text are enough to report an error and to match a
   handler, and they were all this kept -- so `except E as e` rebuilt an
   exception from them and `e.args[0]` came back as the STRING the payload had
   been rendered to. `raise E(42)` then caught an `E('42')`, which is a
   different value of a different type.

   Zero when the pending error came from an operation rather than a `raise`;
   `apy_error_value` builds one from the text in that case, which is right --
   there was no object. It is the second word of the IR reservation above. */

/* --- source positions -----------------------------------------------------

   PEP 657. A traceback names a FRAME and a frame names a CODE OBJECT, whose
   positions say where each of its operations was written. Neither existed
   here, so `e.__traceback__` was an empty tuple standing in for one.

   WHAT IS RECORDED IS ONE POSITION PER STATEMENT, not one per operation. The
   frontend already sets a span per statement and that is the granularity it
   has; `co_positions()` therefore answers one four-tuple per statement of the
   function, which is coarser than CPython's per-instruction table and is made
   of exactly the same kind of fact. Saying so is the whole of the difference.

   AND NOTHING IS RECORDED UNLESS THE PROGRAM ASKS. The cursor costs a call
   per statement, so the frontend emits none of it unless the source mentions
   `__traceback__` or one of the attributes that reads through it -- see
   `_wants_positions` in frontends/python/lower.py. */
typedef struct {
    apy_value fn;                       /* the function it was written in */
    int32_t line, end_line, col, end_col;
} apy_srcpos;                           /* NOT `apy_pos`, which is unary `+` */
/* REACHED THROUGH TWO FUNCTIONS, for the reason the error state above is:
   `runtime/errstate.py` replaces them and the table moves with them, while
   the bodies here keep the runtime standing alone when nothing is ported.
   `apy_pos_cap` stays a plain static because only `apy_pos_add` reads it, and
   `apy_pos_add` moves whole. */
static apy_srcpos *apy_pos_tab_c;
static int64_t apy_pos_n_c, apy_pos_cap;
APY_API apy_value apy_pos_rows(void) { return (apy_value)apy_pos_tab_c; }
APY_API int64_t apy_pos_count(void) { return apy_pos_n_c; }
#define apy_pos_tab ((apy_srcpos *)apy_pos_rows())
#define apy_pos_n   (apy_pos_count())
/* WHICH STATEMENT IS RUNNING, and where the last failure happened. Two cells
   because a handler's own statements move the first one, and what the
   traceback has to report is where the exception came from. */
static int64_t apy_pos_here = -1, apy_err_pos = -1;

APY_API void apy_pos_add(apy_value name, int64_t line, int64_t end_line,
                         int64_t col, int64_t end_col) {
    if (apy_pos_n_c == apy_pos_cap) {
        apy_pos_cap = apy_pos_cap ? apy_pos_cap * 2 : 128;
        apy_pos_tab_c = (apy_srcpos *)realloc(
            apy_pos_tab_c, (size_t)apy_pos_cap * sizeof *apy_pos_tab_c);
        if (!apy_pos_tab_c) return;
    }
    apy_pos_tab_c[apy_pos_n_c].fn = name;
    apy_pos_tab_c[apy_pos_n_c].line = (int32_t)line;
    apy_pos_tab_c[apy_pos_n_c].end_line = (int32_t)end_line;
    apy_pos_tab_c[apy_pos_n_c].col = (int32_t)col;
    apy_pos_tab_c[apy_pos_n_c].end_col = (int32_t)end_col;
    apy_pos_n_c++;
}

APY_API void apy_at(int64_t which) { apy_pos_here = which; }

/* THE SOURCE POSITION, THROUGH ACCESSORS. These four exist so the two
   variables above can move into the IR runtime: the subset cannot read a C
   static, and the remaining C that needs the current position now asks rather
   than reads. They are defined here so `signatures()` can type them, and then
   REPLACED -- see `runtime/errstate.py`, which owns the storage.

   `latch` IS ONE OPERATION AND NOT TWO. Both `apy_fail` and `apy_fail2` did
   `apy_err_pos = apy_pos_here`, which is "remember where the failure was
   raised" -- and a getter plus a setter would let a caller do half of it.
   The pair that must happen together is one function. */
APY_API int64_t apy_pos_now(void) { return apy_pos_here; }
APY_API int64_t apy_pos_latch(void) { apy_err_pos = apy_pos_here; return apy_err_pos; }
APY_API int64_t apy_pos_latched(void) { return apy_err_pos; }

APY_API apy_value apy_raise_at(apy_value type, apy_value msg) {
    if (!apy_err_type) {          /* first error wins, like a real traceback */
        /* WHERE IT HAPPENED, taken at the moment the flag goes up. By the
           time a handler asks, its own statements have moved the cursor --
           and this is the choke point every FAILED OPERATION comes through,
           which is most of them: `apy_raise_over` is only for a `raise`
           that overrides a pending error. */
        apy_pos_latch();
        apy_err_type = (const char *)type;
        apy_err_value = 0;
        snprintf(apy_err_msg, APY_ERR_MSG_CAP, "%s", (const char *)msg);
    }
    return 0;
}
/* THE NAME ITS CALL SITES USE, kept as a delegate so that moving the body to
   IR moves nothing else. Every caller passes a literal, so the cast to a
   machine word -- which is what an IR `ptr` compiles to -- happens here,
   once, rather than at each of them. */
static apy_value apy_fail(const char *type, const char *msg) {
    return apy_raise_at((apy_value)(uintptr_t)type,
                        (apy_value)(uintptr_t)msg);
}

/* An EXPLICIT `raise`, which replaces whatever was pending.
   `apy_fail` keeps the first writer, and that is right for an OPERATION
   failing: after one error the frontend may run more operations before it
   checks, and the second failure is a consequence of the first rather than
   the cause. A `raise` statement is not that -- reaching one means control
   deliberately got there, and Python's rule is that the new exception wins:

       try:
           raise ValueError("original")
       finally:
           raise KeyError("from-finally")   # this is what propagates

   Keeping the first here meant the ValueError escaped and the KeyError was
   discarded, which is the opposite of what the program says.

   The displaced exception becomes the new one's `__context__` in CPython.
   That is not recorded yet, so `e.__context__` is not available; the
   REPLACEMENT is what changes which exception a handler sees, and that part
   is now right. */
APY_API apy_value apy_raise_over(apy_value type, apy_value msg) {
    /* WHERE IT HAPPENED, taken at the moment the flag goes up. By the time a
       handler asks, its own statements have moved the cursor. */
    apy_pos_latch();
    apy_err_type = (const char *)type;
    apy_err_value = 0;
    snprintf(apy_err_msg, APY_ERR_MSG_CAP, "%s", (const char *)msg);
    return 0;
}
static apy_value apy_fail_replacing(const char *type, const char *msg) {
    return apy_raise_over((apy_value)(uintptr_t)type,
                          (apy_value)(uintptr_t)msg);
}

APY_API apy_value apy_raise_fmt(apy_value type, apy_value fmt,
                                apy_value a, apy_value b) {
    char buf[APY_ERR_MSG_CAP];
    snprintf(buf, sizeof buf, (const char *)fmt,
             (const char *)a, (const char *)b);
    return apy_raise_at(type, (apy_value)(uintptr_t)buf);
}
static apy_value apy_fail2(const char *type, const char *fmt,
                           const char *a, const char *b) {
    return apy_raise_fmt((apy_value)(uintptr_t)type, (apy_value)(uintptr_t)fmt,
                         (apy_value)(uintptr_t)a, (apy_value)(uintptr_t)b);
}

APY_API int64_t apy_error_occurred(void) { return apy_err_type != NULL; }
APY_API void apy_error_clear(void) {
    apy_err_type = NULL;
    apy_err_value = 0;
}

/* THE STATUS AN ESCAPING `SystemExit` ASKS FOR, or -1 for anything else.

   DECLARED HERE AND DEFINED LATER, which is the one shape this could take:
   `apy_fatal_if_error` is the choke point every escaping exception reaches
   and it belongs in this part, while deciding whether a type IS a SystemExit
   needs the hierarchy table and rendering a non-int status needs the text
   machinery -- and both of those are parts above this one. See
   `apy_exit_status` in the builtins part for the rule it applies.

   AND static, not APY_API, because it is this file's own plumbing rather
   than a runtime entry point: nothing a compiled PROGRAM emits calls it, so
   exporting it would oblige `objects/host.py` to carry a binding for a
   symbol no program can name. The forward declaration is the accepted way a
   part reaches below itself -- `apy_is_int_like` in the bigint part is
   declared the same way. */
static int64_t apy_exit_status(void);

APY_API void apy_fatal_if_error(void) {
    if (!apy_err_type) return;
    fflush(stdout);
    /* A `SystemExit` IS NOT A FAILURE TO REPORT. Every other exception
       arriving here is a program that went wrong and is worth a line on
       stderr; this one is a program that ASKED TO STOP, and printing
       `SystemExit: 3` while exiting 1 gets both halves wrong -- the status
       a caller reads is not the one the program chose, and `sys.exit(0)`
       would report failure. The interpreter's twin is `_exit_status_of`. */
    {
        int64_t status = apy_exit_status();
        if (status >= 0) exit((int)status);
    }
    fprintf(stderr, "%s: %s\n", apy_err_type, apy_err_msg);
    exit(1);
}

/* `apy_error_type` / `apy_error_message` are down in the construction section
   instead of here, because they return STR VALUES and so need the allocator.
   The state and the failure path stay up here where every operation that sets
   them can see them. */

/* --- construction ------------------------------------------------------ */
static apy_obj apy_none_cell = { APY_NONE_K, { 0 } };
static apy_obj apy_ellipsis_cell = { APY_ELLIPSIS_K, { 0 } };
static apy_obj apy_notimpl_cell = { APY_NOTIMPL_K, { 0 } };
static apy_obj apy_true_cell = { APY_BOOL_K, { 1 } };
static apy_obj apy_false_cell = { APY_BOOL_K, { 0 } };

/* THE ONLY PLACE OBJECTS COME FROM, split in two so that the allocation
   itself can be ported to the machine subset -- see `objects/ir.py` and
   stage 4 of docs/INERT-RUNTIME.md. `apy_cell_new` is the allocation and
   nothing else; the wrapper keeps the out-of-memory policy, because stopping
   the program is a decision rather than a way of getting memory, and a
   replacement allocator should not have to agree with it to be substituted. */
APY_API apy_value apy_obj_alloc(int64_t kind);

static apy_obj *apy_alloc(int kind) {
    apy_obj *o = (apy_obj *)apy_obj_alloc((int64_t)kind);
    if (!o) { fputs("uasm: out of memory\n", stderr); exit(1); }
    return o;
}

APY_API apy_value apy_obj_alloc(int64_t kind) {
    apy_obj *o = (apy_obj *)malloc(sizeof(apy_obj));
    if (!o) return 0;
    o->kind = (int)kind;
    /* ZEROED, not merely tagged. An exception carries `context`, `cause` and
       `notes` that most of them never set, and every one of the four places
       that builds one would otherwise have to remember all three -- which is
       the kind of thing that gets remembered in three places and forgotten in
       the fourth, where it reads uninitialised memory as a value. */
    memset(&o->v, 0, sizeof o->v);
    return V(o);
}

/* ── the BUFFERS ─────────────────────────────────────────────────────────
   A CELL IS IMMORTAL AND A BUFFER IS NOT, which is the whole reason these are
   three functions and not one. Stage 4's arena is a bump pointer and is
   correct for cells because nothing frees one -- checked, not assumed. A
   list's `v.q.items` is the other case: it DOUBLES on growth and it is the
   one allocation this runtime genuinely releases, so it needs an allocator
   that can hand the same memory out twice.

   NAMED SEPARATELY FROM `apy_alloc_bytes` rather than replacing it. Rounding
   every allocation up to a size class would cost a cell 40% -- 152 bytes into
   a 256-byte class -- to serve the one kind of allocation that is freed.
   Immortal things keep the exact-fit bump; things that come back get classes.

   THE SIZE TRAVELS WITH THE POINTER. `free(p)` needs no size because malloc
   keeps a header; a size-classed allocator over a bump arena has no header to
   read, so every caller passes the length it asked for. That is why
   `apy_free_block` and `apy_realloc_block` take one -- and it is checkable,
   because a wrong size puts the block on the wrong list and the corpus
   notices long before a user would.

   These three are the C's, over malloc, and `runtime/blocks.py` replaces them
   with the arena version. Both are supported, which is what
   `--object-runtime c` means. */
APY_API apy_value apy_alloc_block(int64_t n) {
    return (apy_value)(uintptr_t)malloc((size_t)(n > 0 ? n : 1));
}

APY_API apy_value apy_realloc_block(apy_value p, int64_t was, int64_t want) {
    (void)was;
    return (apy_value)(uintptr_t)realloc((void *)(uintptr_t)p,
                                         (size_t)(want > 0 ? want : 1));
}

/* Answers an int rather than nothing, because `signatures()` types every
   exported symbol as `ptr`/`i64`/`f64` and a void return is not among them. */
APY_API int64_t apy_free_block(apy_value p, int64_t was) {
    (void)was;
    free((void *)(uintptr_t)p);
    return 0;
}

APY_API apy_value apy_none(void) { return V(&apy_none_cell); }

/* `...` and the name `Ellipsis` -- one cell, so `is` answers True. */
APY_API apy_value apy_ellipsis(void) { return V(&apy_ellipsis_cell); }
APY_API apy_value apy_notimplemented(void) { return V(&apy_notimpl_cell); }

APY_API apy_value apy_from_bool(int64_t b) {
    return V(b ? &apy_true_cell : &apy_false_cell);
}

/* Small ints are SHARED cells, exactly as CPython caches -5..256, and for the
   same observable reason: `a = 1; b = 1; a is b` is True in CPython, and a
   fresh cell per literal would make it False. The range is CPython's own
   because the answer to `x is y` has to agree at the boundary too -- 256 is
   shared and 257 is not, and a program can see that.

   The table is filled lazily rather than at startup: there is no init hook to
   hang a loop on, and a NULL slot is a cheaper test than a "have I run yet"
   flag. */
#define APY_SMALL_LO (-5)
#define APY_SMALL_HI 256
static apy_obj *apy_small[APY_SMALL_HI - APY_SMALL_LO + 1];

APY_API apy_value apy_from_int(int64_t i) {
    apy_obj *o;
    if (i >= APY_SMALL_LO && i <= APY_SMALL_HI) {
        apy_obj **slot = &apy_small[i - APY_SMALL_LO];
        if (!*slot) { *slot = apy_alloc(APY_INT_K); (*slot)->v.i = i; }
        return V(*slot);
    }
    o = apy_alloc(APY_INT_K);
    o->v.i = i;
    return V(o);
}

APY_API apy_value apy_from_float(double f) {
    apy_obj *o = apy_alloc(APY_FLOAT_K);
    o->v.f = f;
    return V(o);
}

/* A string literal: static storage, so the bytes are borrowed rather than
   copied. Every str the runtime BUILDS mallocs instead; nothing mutates a str,
   so the two are indistinguishable to a reader. */
/* A C string literal, for the runtime's own use. `apy_from_cstr` is the same
   thing for the IR, which hands the address over as an integer because that is
   what a `global_addr` produces and how the C backend types a pointer. */
/* DEFINED BELOW, and declared here because the two delegates that follow sit
   above them in the source. */
APY_API apy_value apy_from_cstr(apy_value p);
APY_API apy_value apy_from_bytes(apy_value p, int64_t n);

/* THE SAME FUNCTION AS `apy_from_cstr`, WHICH IS PORTED. Both tagged a
   str cell, pointed it at bytes they did not own and measured with `strlen`;
   they differed in one cast. Two implementations of one fact is one of them
   waiting to be fixed alone, so this delegates and the IR keeps the body. */
APY_API apy_value apy_lit(const char *p) {
    return apy_from_cstr((apy_value)(uintptr_t)p);
}

APY_API apy_value apy_from_cstr(apy_value p) {
    return apy_from_bytes(p, (int64_t)strlen((const char *)(uintptr_t)p));
}

/* The same, with the length given rather than found. `apy_from_cstr` stops at
   the first NUL and a Python string may CONTAIN one -- `'\0a'` is a perfectly
   ordinary two-character string, and through `strlen` it becomes the empty
   string, silently and at every use. A literal whose bytes include a NUL has
   to come through here instead; the frontend knows the length at compile
   time, so this costs it nothing. */
/* A bytes LITERAL, from static storage -- the same borrowing `apy_from_cstr`
   does, and for the same reason: the bytes are in the binary's read-only data
   and copying them at every evaluation would allocate for a constant.

   The length is passed rather than measured. A bytes literal may contain a
   NUL (`b"a\\0b"` is three bytes) and `strlen` would silently truncate it,
   which is the one thing bytes must not do -- carrying arbitrary octets is
   what distinguishes it from str. */
/* One HALF of a complex, formatted.

   `repr(1j)` is `1j`, not `1.0j` -- a complex's parts are rendered without
   the trailing `.0` that a bare float gets. CPython does this by passing
   `PyOS_double_to_string` a different flag; here the suffix is removed after
   the fact, which reaches the same text and keeps ONE shortest-round-trip
   implementation rather than two.

   Only an exact `.0` at the end is removed, so `0.5` and `1e-05` are
   untouched and `inf`/`nan` never had one. */
static void apy_complex_part(char *out, size_t n, double v) {
    size_t len;
    py_repr_double(out, n, v);
    len = strlen(out);
    if (len > 2 && out[len - 2] == '.' && out[len - 1] == '0')
        out[len - 2] = 0;
}

APY_API apy_value apy_from_complex(double re, double im) {
    apy_obj *o = apy_alloc(APY_COMPLEX_K);
    o->v.z.re = re;
    o->v.z.im = im;
    return V(o);
}

/* THE TWO SHARED-CELL TABLES, declared ahead of the constructors that
   answer from them. Both are defined below, next to the storage they read;
   both are `APY_API` because a PORTED build replaces them (see
   `REPLACES["str_cell.py"]`) and a build must have exactly one of each. */
APY_API apy_value apy_shared_str(int64_t cp);
APY_API apy_value apy_shared_bytes(int64_t b);
/* THE CODE POINT A ONE-CHARACTER UTF-8 SEQUENCE SPELLS, or -1. Defined with
   the tables; declared here because `apy_from_bytes` is the first caller. */
static int apy_shared_cp(const char *p, int64_t n);

APY_API apy_value apy_bytes_literal(apy_value p, int64_t n) {
    apy_obj *o;
    /* THE SHARED ONES FIRST. CPython keeps one empty bytes and one per
       octet, and a program sees it: `b"" is bytes()` and
       `b"a" is bytes([97])` are True there. A LITERAL comes through here, so
       the module's own `b""` has to be the one every other path answers
       with -- a cell of its own would compare unequal to all of them. */
    if (n == 0) return apy_shared_bytes(256);
    if (n == 1)
        return apy_shared_bytes(
            (unsigned char)((const char *)(uintptr_t)p)[0]);
    o = apy_alloc(APY_BYTES_K);
    o->v.s.p = (const char *)(uintptr_t)p;
    o->v.s.n = n;
    return V(o);
}

/* THE BYTES OF A STRING, AS A POINTER, for a native call that wants one.

   SAFE ON BOTH COUNTS, and both were checked rather than assumed. NUL
   TERMINATION: every producer writes one and the remaining C already relies
   on it in two hundred places through `APY_CSTR`, including a SLICE, which
   reaches `apy_str_copy` like everything else. LIFETIME: a literal lives in
   the program's read-only data and an arena string is immortal, so nothing
   the callee keeps can be freed underneath it -- the bump-pointer arena that
   cannot free is exactly what makes this answerable.

   The kind IS checked, unlike the extraction helpers next door. Those are
   reached only where the frontend has proved the kind; this one is reached
   from `ctypes` with whatever the program passed, and handing a native
   function the address of an integer cell is how a ctypes program corrupts
   memory rather than failing. */
/* Declared here rather than waited for: `apy_kind_name` is defined a
   couple of thousand lines down, and the refusal below wants to name
   the kind it was handed. Same in-place idiom as `apy_obj_alloc`
   above. */
static const char *apy_kind_name(apy_value v);

APY_API apy_value apy_str_bytes(apy_value s) {
    /* AN INT IS ALLOWED AND MEANS THE ADDRESS ITSELF, because C's own rule
       for a pointer parameter is that a null pointer constant fits it -- and
       `CreateDirectoryA(path, 0)` is the ordinary way to pass "no security
       descriptor". Refusing it made every native call with a NULL argument a
       TypeError, which is not what the C being declared says. */
    if (O(s)->kind == APY_INT_K)
        return (apy_value)(uintptr_t)O(s)->v.i;
    /* THE MESSAGE NAMES THE KIND because the same refusal is worded by
       `objects_host._apy_str_bytes` for the interpreter, and a program that
       prints the exception would otherwise get different text from the two
       paths -- which the corpus compares. */
    if (O(s)->kind != APY_STR_K && O(s)->kind != APY_BYTES_K)
        return apy_fail2("TypeError",
                         "a pointer argument must be str, bytes or int, "
                         "not %s%s", apy_kind_name(s), "");
    return (apy_value)(uintptr_t)O(s)->v.s.p;
}

/* THE str CELL, AND THE ONE PLACE THE SHARED STRINGS ARE ANSWERED FROM.
   Every string this runtime builds is a cell, and every cell is made here --
   `apy_str_take` for the ones that own their bytes, `apy_str_copy_bytes` for
   the ones that copy, and the frontend's literals. Putting the test in the
   constructor is what makes `"" is str()`, `"a" is chr(97)` and
   `"abc"[0:1] is "a"` all True without fifty call sites knowing about it. */
APY_API apy_value apy_from_bytes(apy_value p, int64_t n) {
    apy_obj *o;
    int cp;
    if (n == 0) return apy_shared_str(256);
    cp = apy_shared_cp((const char *)(uintptr_t)p, n);
    if (cp >= 0) return apy_shared_str(cp);
    o = apy_alloc(APY_STR_K);
    o->v.s.p = (const char *)p;
    o->v.s.n = n;
    return V(o);
}

/* THE SHARED EMPTY STRING AND THE 256 ONE-CHARACTER ONES, and the same for
   bytes. CPython keeps exactly these -- `"" is str()`, `"a" is chr(97)` and
   `"abc"[0:1] is "a"` are all True, and a character ABOVE U+00FF is not
   cached, so `chr(256) is chr(256)` is False. A program sees the difference
   with `is` and with `id()`, which is the only reason a runtime would
   bother.

   BUILT ON FIRST ASK and never freed, which is what a singleton is. The
   bytes they point at are static storage here, so there is nothing to own
   and nothing to copy.

   INDEX 256 IS THE EMPTY ONE, which keeps the two tables one lookup each.

   THE str TABLE IS FILLED FROM `apy_str_take` AND THE bytes TABLE FROM
   `apy_bytes_take`, which are the two funnels every string and every bytes
   this runtime builds comes out of. Putting the test anywhere else would
   mean putting it in fifty places. */
static apy_value apy_str_shared[257];
static apy_value apy_bytes_shared[257];
static char apy_shared_utf8[256][3];
static char apy_shared_byte[256][2];

/* THE SHARED STRING FOR ONE CODE POINT, or for NOTHING at index 256.

   EXPORTED, AND PORTED, because a build must have exactly ONE of these
   tables. `apy_str_take` is C and stays C; `apy_str_copy_bytes` is replaced
   by IR on a ported build -- and if each kept its own table, a literal built
   through one and a slice built through the other would be two empty strings
   that compare unequal. Both call this instead, and `REPLACES["str_cell.py"]`
   names it, so the ported build has the IR's table and the plain build has
   this one. */
APY_API apy_value apy_shared_str(int64_t cp) {
    if (!apy_str_shared[cp]) {
        /* THE CELL IS BUILT HERE AND NOT THROUGH `apy_from_bytes`, which
           asks this function first: the constructor is where the sharing
           test lives, so the one call that must not take it is this one. */
        apy_obj *o = apy_alloc(APY_STR_K);
        int64_t n = 0;
        char *b = (char *)"";
        if (cp != 256) {
            b = apy_shared_utf8[cp];
            if (cp < 0x80) { b[0] = (char)cp; n = 1; }
            else {
                b[0] = (char)(0xC0 | (cp >> 6));
                b[1] = (char)(0x80 | (cp & 0x3F));
                n = 2;
            }
            b[n] = '\0';
        }
        o->v.s.p = b;
        o->v.s.n = n;
        apy_str_shared[cp] = V(o);
    }
    return apy_str_shared[cp];
}

/* THE CODE POINT A ONE-CHARACTER UTF-8 SEQUENCE SPELLS, or -1 for anything
   this shares nothing for -- more than one character, or a character above
   U+00FF. One latin-1 character is one or two bytes, which is the whole of
   what the two arms decode. */
static int apy_shared_cp(const char *p, int64_t n) {
    const unsigned char *b = (const unsigned char *)p;
    if (n == 1 && b[0] < 0x80) return (int)b[0];
    if (n == 2 && (b[0] & 0xE0) == 0xC0 && (b[1] & 0xC0) == 0x80) {
        int cp = ((b[0] & 0x1F) << 6) | (b[1] & 0x3F);
        if (cp >= 0x80 && cp < 256) return cp;
    }
    return -1;
}

/* AND THIS IS `apy_from_bytes`. See `apy_lit` above.

   THE BUFFER IS ABANDONED AND NOT FREED when the constructor answers a
   shared cell. Callers hand this `malloc`'d storage, arena storage and, in
   two places, a stack array; there is no one `free` that is right for all
   three, and this runtime releases no string's bytes anywhere --
   `apy_str_copy_bytes` says why at length. Two bytes per tiny string built
   is the price of not having to know which kind arrived. */
APY_API apy_value apy_str_take(char *p, int64_t n) {
    return apy_from_bytes((apy_value)(uintptr_t)p, n);
}

/* THE SAME BUFFER UNDER THE bytes TAG, and the funnel that replaced writing
   `O(x)->kind = APY_BYTES_K` over a cell somebody else might be holding.

   `mut` IS A PARAMETER AND NOT SOMETHING THE CALLER WRITES AFTERWARDS. A
   bytearray is written into, so it can never be one of the shared cells:
   `bytearray(b"")` made writable in place would be `b""` made writable, for
   every later use of it in the program. Asking for the flag here is what
   makes "a fresh cell" and "a bytearray" the same decision. */
static apy_value apy_bytes_own(char *p, int64_t n, int mut) {
    apy_obj *o = apy_alloc(APY_BYTES_K);
    o->v.s.p = p;
    o->v.s.n = n;
    o->v.s.mut = mut;
    return V(o);
}

/* THE SHARED BYTES FOR ONE OCTET, or for NOTHING at index 256 -- the bytes
   half of `apy_shared_str`, exported and ported for the same reason. */
APY_API apy_value apy_shared_bytes(int64_t b) {
    if (!apy_bytes_shared[b]) {
        char *s = (char *)"";
        int64_t n = 0;
        if (b != 256) {
            s = apy_shared_byte[b];
            s[0] = (char)b;
            s[1] = '\0';
            n = 1;
        }
        apy_bytes_shared[b] = apy_bytes_own(s, n, 0);
    }
    return apy_bytes_shared[b];
}

/* IMMUTABLE BYTES OVER A BUFFER, shared when CPython shares it: the one
   empty and the 256 one-byte values. The buffer is abandoned in that case
   for the reason `apy_str_take` gives. */
static apy_value apy_bytes_take(char *p, int64_t n) {
    if (n == 0) return apy_shared_bytes(256);
    if (n == 1) return apy_shared_bytes((unsigned char)p[0]);
    return apy_bytes_own(p, n, 0);
}

/* `n` BYTES COPIED INTO A CELL OF ITS OWN, which is what the bytearray
   constructors and every method that re-reads its receiver want. */
static apy_value apy_bytes_dup(const char *p, int64_t n, int mut) {
    char *buf;
    if (!mut) {
        /* The shared ones need no buffer at all. */
        if (n == 0) return apy_shared_bytes(256);
        if (n == 1) return apy_shared_bytes((unsigned char)p[0]);
    }
    buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fputs("uasm: out of memory\n", stderr); exit(1); }
    if (n) memcpy(buf, p, (size_t)n);
    buf[n] = '\0';
    return apy_bytes_own(buf, n, mut);
}

/* THE ONE THE RUNTIME BUILDS EVERY STRING WITH, split in two so the half
   that allocates can be replaced by IR.

   The parameter is an `apy_value` and not a `const char *` because that is
   what an IR `ptr` compiles to: a ported definition emits
   `uintptr_t apy_str_copy_bytes(uintptr_t, int64_t)`, and a C prototype
   spelling the first argument as a pointer is a CONFLICTING TYPE where gcc
   sees both. The three constructors ported before this one already took
   `apy_value`, so the question never arose.

   THE SHIM IS WHY THE 24 CALL SITES DID NOT HAVE TO CHANGE. Retyping
   `apy_str_copy` itself would have meant a cast at every slice, join, case
   transform and repr in this file -- a mechanical sweep through the one
   function every string operation flows through, where a wrong cast gives
   plausible wrong strings rather than a crash.

   THE SHARED ONES ARE ANSWERED BEFORE THE ALLOCATION rather than after it,
   so that a one-character string costs nothing. `apy_str_take` would answer
   them anyway; this only saves the `malloc`. The IR's copy of this function
   (`runtime/str_cell.py`) keeps its own table for the same reason -- on a
   ported build that one is live and this one is not, and a build has exactly
   one table either way. */
APY_API apy_value apy_str_copy_bytes(apy_value p, int64_t n) {
    char *buf;
    int cp;
    if (n == 0) return apy_shared_str(256);
    cp = apy_shared_cp((const char *)(uintptr_t)p, n);
    if (cp >= 0) return apy_shared_str(cp);
    buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fputs("uasm: out of memory\n", stderr); exit(1); }
    memcpy(buf, (const char *)(uintptr_t)p, (size_t)n);
    buf[n] = '\0';
    return apy_from_bytes((apy_value)(uintptr_t)buf, n);
}

APY_API apy_value apy_str_copy(const char *p, int64_t n) {
    return apy_str_copy_bytes((apy_value)(uintptr_t)p, n);
}

/* The same bytes under a different kind -- its OWN copy, and not
   `apy_str_copy` re-tagged: that funnel answers SHARED cells now, and
   re-tagging one would turn every later `""` into `b""` at once. */
APY_API apy_value apy_bytes_copy(const char *p, int64_t n) {
    return apy_bytes_dup(p, n, 0);
}

/* AND THE WRITABLE ONE. `bytearray(b)` must not share a cell with anything,
   least of all with the empty bytes every program holds. */
APY_API apy_value apy_bytearray_copy(const char *p, int64_t n) {
    return apy_bytes_dup(p, n, 1);
}

/* THE SAME TEXT IN A CELL OF ITS OWN, for the callers that must answer a
   DIFFERENT object holding it.

   THE EMPTY ONE IS STILL SHARED AND A ONE-CHARACTER ONE IS NOT, which is not
   this runtime's arrangement but CPython's: `_PyUnicode_Copy` builds its
   result with `PyUnicode_New(length, maxchar)`, and `PyUnicode_New`
   special-cases `size == 0` ALONE (Objects/unicodeobject.c) -- so `""
   .__getnewargs__()[0] is ""` is True there and `"a".__getnewargs__()[0] is
   "a"` is False. Measured both ways before this was written. `apy_str_copy`
   cannot serve: its funnel answers the shared cell for ANY latin-1 character,
   which is right for every other caller and wrong for this one.

   THE BYTES ARE COPIED rather than borrowed from the source cell. Borrowing
   would be safe -- no str is ever written into and no string's bytes are
   ever freed -- but two cells over one buffer is a thing a later reader has
   to hold in mind, and a copy costs one `malloc` on a path a program takes
   only when it pickles. */
static apy_value apy_str_fresh(const char *p, int64_t n) {
    apy_obj *o;
    char *buf;
    if (n == 0) return apy_shared_str(256);
    buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fputs("uasm: out of memory\n", stderr); exit(1); }
    memcpy(buf, p, (size_t)n);
    buf[n] = '\0';
    /* NOT THROUGH `apy_from_bytes`: that IS the sharing test, and this is the
       one caller that must not take it. `apy_shared_str` says the same of
       itself, for the same reason. */
    o = apy_alloc(APY_STR_K);
    o->v.s.p = buf;
    o->v.s.n = n;
    return V(o);
}

/* A BUILTIN OPERATION'S RESULT, MADE A DIFFERENT OBJECT when the receiver was
   an INSTANCE of a class extending str or bytes and the operation handed that
   instance's held text straight back.

   THE UNWRAP IS WHY THIS IS NEEDED AT ALL. `apy_method_self`, `apy_text_like`
   and the builtin step of `apy_binop_dispatch` all reach past a `class
   S(str)` to the str it carries BEFORE the operation runs, so by the time a
   method reaches its own tail the receiver it sees is an exact str and
   `apy_str_may_return_self` rightly says yes. The subclass is invisible
   there; it is visible only at the CALL SITE, which still holds the value the
   program wrote -- so the copy belongs there and not in a wider
   `may_return_self`.

   CPython COPIES FOR A SUBCLASS AT EVERY ONE OF THESE, and it is the same one
   test each time: `unicode_result_unchanged` (Objects/unicodeobject.c) hands
   the argument back only `if (PyUnicode_CheckExact(unicode))` and calls
   `_PyUnicode_Copy` otherwise, and stringlib's `return_self` is written the
   same way. So `S("ab").strip() is S("ab").strip()` is False in CPython.

   TWO INSTANCES SHARING ONE HELD CELL IS WHAT MAKES IT VISIBLE, and it is why
   this hid: `x.strip() is x` was already False -- the instance is not its own
   text -- and only `x.strip() is y.strip()` tells the truth. `S("ab")` twice
   over one literal gives two instances holding the SAME cell, since a literal
   is built once, so every no-op method answered that one cell for both.

   THE COPY IS OF THE HELD KIND, through the constructors that already know
   which cells CPython shares: `apy_str_fresh` above shares the empty string
   alone, and `apy_bytes_copy` shares the empty bytes and the 256 one-octet
   cells exactly as `PyBytes_FromStringAndSize` does. A BYTEARRAY IS LEFT
   ALONE -- it is written into, so a method that handed one back is a bug of a
   different shape and `apy_str_self_or_copy` is where that one is answered.

   AN ORDINARY INSTANCE AND A NON-TEXT ONE PASS THROUGH UNTOUCHED, which is
   what lets this drop in front of an existing tail rather than beside it. */
static apy_value apy_inst_text_result(apy_value self, apy_value out) {
    apy_value held;
    if (!out || O(self)->kind != APY_INST_K) return out;
    held = O(self)->v.o.held;
    if (!held || out != held) return out;
    if (O(held)->kind == APY_STR_K)
        return apy_str_fresh(O(held)->v.s.p, O(held)->v.s.n);
    if (O(held)->kind == APY_BYTES_K && !O(held)->v.s.mut)
        return apy_bytes_copy(O(held)->v.s.p, O(held)->v.s.n);
    return out;
}


/* The error accessors live down here rather than beside the error state,
   because they hand back STR VALUES and the allocator is only in scope from
   this point on. Both answer None when nothing is set, so a frontend that
   calls them unguarded gets a value rather than a null it would have to
   test -- and None is also what `sys.exc_info()` would say. */
APY_API apy_value apy_error_type(void) {
    /* THROUGH `apy_none()`, not `&apy_none_cell`. The None cell is defined in
       IR now (`runtime/singletons.py`), and this was one of exactly two places
       that reached the static directly. Left alone it would have handed back a
       SECOND None -- correct in every field, and unequal under `is` to the one
       every other path answers with. */
    return apy_err_type ? apy_lit(apy_err_type) : apy_none();
}

APY_API apy_value apy_error_message(void) {
    return apy_err_type
        ? apy_str_copy(apy_err_msg, (int64_t)strlen(apy_err_msg))
        /* See `apy_error_type` above: the cell is IR's, so this goes through
           the accessor rather than at the static. */
        : apy_none();
}

static uint64_t apy_abs64(int64_t v);
APY_API apy_value apy_lit(const char *p);
APY_API apy_value apy_str_take(char *p, int64_t n);

"""
