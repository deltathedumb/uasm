"""The object runtime, in C: generators, asyncio and tasks.

ONE PART OF ONE TRANSLATION UNIT. `c/__init__.py` concatenates
these in order and the result is the file it always was, so a
definition here may rely on anything in an earlier part and
nothing in a later one. Sections, in order:
  * generators
  * asyncio
  * tasks
"""

C = r"""/* --- generators --------------------------------------------------------- */
/* `def g(): yield 1` -- a function that SUSPENDS.

   The IR models a function as a block graph with one entry, so there is no
   way to stop in the middle of one and come back. A generator is therefore
   compiled as TWO functions and an object that carries what would otherwise
   be a live frame:

     * the CONSTRUCTOR keeps the name the `def` binds. Calling it allocates a
       generator, stores the arguments into its slots, and returns it having
       run none of the body -- which is why `g()` on a generator with a
       `print` at the top prints nothing.
     * the STEP function is the body, re-entered once per `next`. It begins
       with a dispatch on the saved state and jumps to the block after the
       `yield` that last returned.

   EVERY LOCAL LIVES IN THE OBJECT, not in a register, because a register does
   not survive the return that a `yield` compiles to. That costs a load per
   read and a store per write and it needs no liveness analysis to be correct
   -- and being correct without one is worth more here than the loads.

   `state` is 0 before the first step, k while suspended at yield k, and -1
   once the body has finished. The step sets it; nothing else needs to know
   what the numbers mean. */

APY_API apy_value apy_gen_new(apy_value step, int64_t nslots) {
    apy_obj *o = apy_alloc(APY_GEN_K);
    o->v.g.step = step;
    o->v.g.n = nslots;
    o->v.g.slots = nslots > 0
        ? (apy_value *)calloc((size_t)nslots, sizeof(apy_value)) : NULL;
    o->v.g.state = 0;
    o->v.g.sent = apy_none();
    o->v.g.running = 0;
    o->v.g.coro = 0;
    o->v.g.builtin = 0;
    o->v.g.agen = 0;
    o->v.g.deadline = 0.0;
    o->v.g.cancel = 0;
    o->v.g.yieldfrom = 0;
    o->v.g.sig = 0;
    o->v.g.wrapper = 0;
    return V(o);
}

/* `c.__await__()` -- THE OBJECT BETWEEN THE COROUTINE AND WHAT DRIVES IT.

   CPython answers a `coroutine_wrapper`, not the coroutine, and a program
   can see all three differences: it is not `c`, `type(...).__name__` is
   `coroutine_wrapper`, and `dir()` of it holds `close`, `send` and `throw`
   and nothing else -- none of the `cr_` introspection the coroutine carries.
   Handing the coroutine back made `w is c` True and `dir(w)` eight names
   longer.

   IT HAS NO BODY OF ITS OWN. Every `send`, `throw` and `close` goes to the
   coroutine, which is in `yieldfrom` -- which is what `yieldfrom` has always
   meant, the thing this frame is delegating to. */
APY_API apy_value apy_coro_wrapper(apy_value co) {
    apy_value w;
    if (O(co)->kind != APY_GEN_K) return co;
    w = apy_gen_new(0, 0);
    if (!w) return 0;
    O(w)->v.g.wrapper = 1;
    O(w)->v.g.yieldfrom = co;
    return w;
}

/* The coroutine a wrapper stands in front of, or the value itself. One test
   and one substitution, so it drops in front of an existing body rather than
   beside it -- see `apy_coro_wrapper`. */
APY_API apy_value apy_coro_wrapped(apy_value v) {
    if (O(v)->kind == APY_GEN_K && O(v)->v.g.wrapper && O(v)->v.g.yieldfrom)
        return O(v)->v.g.yieldfrom;
    return v;
}

/* Mark a freshly built frame as a COROUTINE. Separate from `apy_gen_new` so
   that the constructor an `async def` lowers to is the generator one plus a
   single call, rather than a second nearly identical entry point. */
APY_API apy_value apy_coro_mark(apy_value g) {
    if (O(g)->kind == APY_GEN_K) O(g)->v.g.coro = 1;
    return g;
}

/* EVERY ASYNC GENERATOR MADE DURING A RUN, so the loop can close the ones a
   program abandoned. `async for ...: break` leaves the generator suspended
   inside its own `try`, and its `finally` has not run; CPython closes those
   at loop shutdown -- `shutdown_asyncgens` -- rather than when the loop is
   left, and a program that breaks out of one and never touches it again
   still sees its cleanup. Nothing is ever removed: a run is short and the
   list is only walked once, at the end. */
/* REACHED THROUGH ONE FUNCTION so it can move -- see `apy_canonical_slot`. */
static apy_value apy_live_agens_c = 0;
APY_API apy_value apy_live_agens_slot(void) {
    return (apy_value)&apy_live_agens_c;
}
#define apy_live_agens (*(apy_value *)apy_live_agens_slot())

/* Mark a FUNCTION as one whose call builds a coroutine -- `async def`. */
/* Mark a thunk as standing for a builtin TYPE.

   WHAT THIS DOES NOT BUY IS IDENTITY. `type(1) is int` is still False: the
   frontend synthesises one thunk per builtin per module and `type` makes its
   own cells, so two different objects both claim to be `int`. A registry
   filled on first mention was tried and is worse -- it makes the answer
   depend on which of `type(1)` and `int` the program evaluates first.

   The real fix is one canonical cell per builtin type, callable, which means
   teaching the call path to construct from a type. That is the change this
   flag was chosen to avoid, and it is still the right one. */
/* Mark a thunk as a builtin TYPE, and hand back the CANONICAL one for that
   name. `int` mentioned twice built two thunks, so `int == int` was False and
   `{int, str} == {str, int}` was False -- a plainly wrong answer to a plainly
   ordinary question, and the reason the caller must use what this returns
   rather than the value it passed in.

   This is NOT the registry that was tried and reverted for `type(1) is int`.
   That one made `type(x)` answer whichever object had been built first, so
   the result depended on which of `type(1)` and `int` the program evaluated
   sooner. Interning by NAME has no such order to depend on: the first mention
   of `int` decides, every later one gets the same object, and `type()` is not
   involved. `type(1) is int` still needs a real type cell and still does not
   hold. */
/* REACHED THROUGH ONE FUNCTION so it can move: `apy_type_of` reads it and
   `apy_func_is_type` fills it, and both are on their way to IR. */
static apy_value apy_canonical_types_c = 0;
APY_API apy_value apy_canonical_slot(void) {
    return (apy_value)&apy_canonical_types_c;
}
/* THROUGH THE ACCESSOR, not at the static: when IR replaces
   `apy_canonical_slot` the storage MOVES, and a macro naming the
   C's own variable would leave the two halves writing to
   different words -- which is exactly what made `type(1) is int`
   False on the ported path while staying True on the C one. */
#define apy_canonical_types (*(apy_value *)apy_canonical_slot())

/* PEP 649: record the thunk that builds this function's annotations. */
/* PEP 3155: record a function's qualified name. */
APY_API apy_value apy_func_qualname(apy_value f, apy_value name) {
    if (O(f)->kind == APY_FUNC_K) O(f)->v.fn.qualname = name;
    return f;
}

/* REACHED OFF A TYPE: `list.append` and `str.upper` are DESCRIPTORS in
   CPython and not functions. The qualname carries which type; see the field. */
APY_API apy_value apy_func_descr(apy_value f) {
    if (O(f)->kind == APY_FUNC_K) O(f)->v.fn.descr = 1;
    return f;
}

/* WHERE THE `def` WAS WRITTEN, for `__module__`. Emitted only for a spliced
   definition: see the field. */
APY_API apy_value apy_func_module(apy_value f, apy_value name) {
    if (O(f)->kind == APY_FUNC_K) O(f)->v.fn.module = name;
    return f;
}

APY_API apy_value apy_func_annotate(apy_value f, apy_value thunk) {
    if (O(f)->kind == APY_FUNC_K) O(f)->v.fn.annotate = thunk;
    return f;
}

/* Mark a thunk as standing for a BUILTIN reached as a value. */
APY_API apy_value apy_func_builtin(apy_value f) {
    if (O(f)->kind == APY_FUNC_K) O(f)->v.fn.builtin = 1;
    return f;
}

APY_API apy_value apy_func_is_type(apy_value f) {
    apy_value found;
    if (O(f)->kind != APY_FUNC_K) return f;
    O(f)->v.fn.is_type = 1;
    if (!apy_canonical_types) apy_canonical_types = apy_dict_new(16);
    found = apy_dict_get_or(apy_canonical_types, O(f)->v.fn.name, 0);
    if (found) return found;
    apy_dict_set(apy_canonical_types, O(f)->v.fn.name, f);
    return f;
}

APY_API apy_value apy_func_coro(apy_value f) {
    if (O(f)->kind == APY_FUNC_K) O(f)->v.fn.coro = 1;
    return f;
}

/* `inspect.iscoroutine`, `isgenerator`, `iscoroutinefunction`. The three are
   the whole of what programs ask `inspect` about coroutines, and each is one
   flag -- an async generator is NEITHER a coroutine nor a generator, which is
   the distinction the case turns on. */
APY_API apy_value apy_inspect_iscoroutine(apy_value v) {
    return apy_from_bool(O(v)->kind == APY_GEN_K && O(v)->v.g.coro
                         && !O(v)->v.g.agen);
}

APY_API apy_value apy_inspect_isgenerator(apy_value v) {
    return apy_from_bool(O(v)->kind == APY_GEN_K && !O(v)->v.g.coro);
}

APY_API apy_value apy_inspect_isasyncgen(apy_value v) {
    return apy_from_bool(O(v)->kind == APY_GEN_K && O(v)->v.g.agen);
}

APY_API apy_value apy_inspect_iscoroutinefunction(apy_value v) {
    return apy_from_bool(O(v)->kind == APY_FUNC_K && O(v)->v.fn.coro);
}

/* Mark a frame as an ASYNC GENERATOR -- `async def` with `yield` in it. */
APY_API apy_value apy_agen_mark(apy_value g) {
    if (O(g)->kind == APY_GEN_K) {
        O(g)->v.g.coro = 1;
        O(g)->v.g.agen = 1;
        if (!apy_live_agens) apy_live_agens = apy_seq_new(APY_LIST_K, 4);
        apy_seq_push(apy_live_agens, g);
    }
    return g;
}

/* An UNSET slot reads as None rather than as a null. A local a `yield` has
   not reached yet is not an error to read here -- the frontend has already
   refused a genuine use-before-assignment -- and a null would be taken for a
   pending exception by the next operation that touched it. */
APY_API apy_value apy_gen_slot(apy_value g, int64_t i) {
    apy_value v;
    if (i < 0 || i >= O(g)->v.g.n) return apy_none();
    v = O(g)->v.g.slots[i];
    return v ? v : apy_none();
}

APY_API apy_value apy_gen_set(apy_value g, int64_t i, apy_value v) {
    if (i >= 0 && i < O(g)->v.g.n) O(g)->v.g.slots[i] = v;
    return apy_none();
}

APY_API int64_t apy_gen_state(apy_value g) { return O(g)->v.g.state; }

/* A slot holding a RAW machine word rather than a value.

   A `for` inside a generator keeps its index in the frame for the same reason
   the locals are there -- the register does not survive the return a `yield`
   compiles to -- but the index is a count, not an object, and boxing it would
   allocate once per iteration. Read only through these two, never through
   `apy_gen_slot`, which would take the bits for a pointer. */
APY_API int64_t apy_gen_iget(apy_value g, int64_t i) {
    if (i < 0 || i >= O(g)->v.g.n) return 0;
    return (int64_t)O(g)->v.g.slots[i];
}

APY_API apy_value apy_gen_iset(apy_value g, int64_t i, int64_t v) {
    if (i >= 0 && i < O(g)->v.g.n) O(g)->v.g.slots[i] = (apy_value)v;
    return apy_none();
}

/* `return v` inside the body. Held until the step that ran it reports
   exhaustion, which is when it becomes `StopIteration.value`. */
APY_API apy_value apy_gen_result(apy_value g, apy_value v) {
    O(g)->v.g.result = v;
    return apy_none();
}

/* What a delegated generator RETURNED, for `yield from` to answer with.

   Read off the object rather than caught as a StopIteration because the
   delegation drains rather than stepping -- see `_dyn_yield_from` -- so there
   is no exception to catch by the time the loop ends. Anything that is not a
   generator returned nothing, which is None. */
APY_API apy_value apy_gen_taken(apy_value g) {
    if (O(g)->kind != APY_GEN_K || !O(g)->v.g.result) return apy_none();
    return O(g)->v.g.result;
}

/* WHICH `def` THIS GENERATOR CAME FROM, as a FUNC that describes its
   signature and is never called. Set once by the constructor, from a value
   the module built once; see the `sig` field for why the step cannot be it. */
APY_API apy_value apy_gen_sig(apy_value g, apy_value f) {
    if (O(g)->kind == APY_GEN_K) O(g)->v.g.sig = f;
    return apy_none();
}

/* WHAT A `yield from` IS DELEGATING TO, recorded while the loop runs.

   `g.gi_yieldfrom` is the only way the delegation is visible from outside the
   generator -- the delegate otherwise lives in a frame slot nothing but the
   lowered loop can name -- and a scheduler reads it to find what a coroutine
   is really waiting on. Cleared with a null when the loop ends, which is why
   this takes a raw value rather than asking for a generator. */
APY_API apy_value apy_gen_delegate(apy_value g, apy_value it) {
    if (O(g)->kind == APY_GEN_K) O(g)->v.g.yieldfrom = it;
    return apy_none();
}

APY_API apy_value apy_gen_goto(apy_value g, int64_t k) {
    O(g)->v.g.state = k;
    return apy_none();
}

/* What `send` passed in -- the value a `yield` EXPRESSION evaluates to. None
   for a plain `next`, which is what `next(it)` and `it.send(None)` share. */
APY_API apy_value apy_gen_sent(apy_value g) { return O(g)->v.g.sent; }

/* Is an exception waiting to be raised here? Asked by every resume block, and
   CLEARING, so it fires once. */
APY_API int64_t apy_gen_throwing(apy_value g) {
    return O(g)->v.g.pending != 0;
}

APY_API apy_value apy_gen_pending(apy_value g) {
    apy_value v = O(g)->v.g.pending;
    O(g)->v.g.pending = 0;
    return v ? v : apy_none();
}

/* One step. `sent` is what a `send` supplied, ignored by the body unless it
   reads the yield expression's value.

   A generator ALREADY RUNNING cannot be re-entered: `next(g)` from inside `g`
   is a ValueError, and without the guard it would corrupt the slots it is
   halfway through writing. */
/* The exhaustion signal, carrying whatever `return` gave. A generator that
   returned nothing raises a bare StopIteration, which is not the same as one
   that returned None -- `e.value` is None either way, but only the second had
   a `return` statement, and `has_arg` keeps that distinction for `repr`. */
APY_API apy_value apy_gen_stop(apy_value g) {
    /* A WRAPPER HAS NO RESULT OF ITS OWN -- the coroutine it delegates to
       does, and `next(c.__await__())` carries what `c` returned. */
    apy_value carried = O(apy_coro_wrapped(g))->v.g.result;
    if (!carried || O(carried)->kind == APY_NONE_K)
        return apy_fail("StopIteration", "");
    return apy_raise(apy_make_exc(apy_lit("StopIteration"), carried));
}

/* `done` CROSSES AS A PLAIN WORD, not as an `int *`: the subset has no
   pointer-to-int to declare, and a half taking one has a type gcc calls
   conflicting. It is an int64 through the hole, which is what the delegate
   below converts. See `runtime/calling.py` for where this first bit. */
APY_API apy_value apy_gen_step_of(apy_value g, apy_value sent,
                                  apy_value done) {
    int64_t *out_done = (int64_t *)done;
    apy_value out, arg;
    *out_done = 0;
    /* A WRAPPER DELEGATES, having no body of its own -- see
       `apy_coro_wrapper`. Unwrapped HERE, which is the one funnel every
       driver goes through: `send`, `throw` and `close` each unwrap too, but
       `next(w)` and a `for` over one reach the step directly, and a step
       that is not there is a call through a null pointer. */
    g = apy_coro_wrapped(g);
    arg = g;
    if (O(g)->kind != APY_GEN_K) {
        apy_fail2("TypeError", "'%s' object is not a generator%s",
                  apy_kind_name(g), "");
        return 0;
    }
    if (O(g)->v.g.running) {
        apy_fail("ValueError", "generator already executing");
        return 0;
    }
    if (O(g)->v.g.state < 0) { *out_done = 1; return apy_none(); }
    O(g)->v.g.sent = sent;
    O(g)->v.g.running = 1;
    out = apy_invoke(O(g)->v.g.step, &arg, 1);
    O(g)->v.g.running = 0;
    if (!out) {
        O(g)->v.g.state = -1;
        /* PEP 479: a `StopIteration` that ESCAPES a generator body becomes a
           RuntimeError, with the original as its `__cause__`. Left alone it
           was indistinguishable from the generator finishing normally, so a
           bug inside the body read as a clean end of iteration -- which is
           the entire reason the PEP exists. */
        if (apy_error_occurred()
            && strcmp(apy_err_type, "StopIteration") == 0) {
            apy_value cause = apy_error_value();
            apy_value wrapped = apy_make_exc(
                apy_lit("RuntimeError"),
                apy_lit("generator raised StopIteration"));
            if (wrapped && cause) O(wrapped)->v.e.cause = cause;
            apy_error_clear();
            if (wrapped) apy_raise(wrapped);
        }
        return 0;
    }
    /* The body sets the state to -1 on its way out, so "did this call finish
       the generator" is a question about the state AFTER it, not about the
       value -- a generator may legitimately yield None. */
    if (O(g)->v.g.state < 0) *out_done = 1;
    return out;
}

/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_gen_step(apy_value g, apy_value sent, int *done) {
    int64_t wide = 0;
    apy_value out = apy_gen_step_of(g, sent, (apy_value)(uintptr_t)&wide);
    *done = (int)wide;
    return out;
}

APY_API apy_value apy_gen_next(apy_value g, apy_value fallback,
                               int64_t has_default) {
    int done;
    apy_value out = apy_gen_step(g, apy_none(), &done);
    if (!out) return 0;
    if (done) {
        if (has_default) return fallback;
        return apy_gen_stop(g);
    }
    return out;
}

APY_API apy_value apy_gen_send(apy_value g, apy_value v) {
    int done;
    apy_value out;
    /* A WRAPPER DELEGATES, having no body of its own -- see
       `apy_coro_wrapper`. */
    g = apy_coro_wrapped(g);
    if (O(g)->kind == APY_GEN_K && O(g)->v.g.state == 0
        && O(g)->kind == APY_GEN_K && O(v)->kind != APY_NONE_K)
        return apy_fail("TypeError",
                        "can't send non-None value to a just-started "
                        "generator");
    out = apy_gen_step(g, v, &done);
    if (!out) return 0;
    if (done) return apy_gen_stop(g);
    return out;
}

/* `g.close()`. There is no way to resume the body at its `yield` and raise
   there -- that needs the exception to enter a frame this design does not
   keep -- so the generator is simply marked finished, which is what `close`
   leaves behind and what every later `next` on it must see. */
/* `g.throw(exc)` -- raise AT the suspension point.

   The generator is resumed with the exception pending, so a `try` around the
   `yield` inside the body catches it. One that does not lets it out, and the
   generator is finished either way it does not yield again. */
APY_API apy_value apy_gen_throw(apy_value g, apy_value exc) {
    int done;
    apy_value out;
    /* A WRAPPER DELEGATES, having no body of its own -- see
       `apy_coro_wrapper`. */
    g = apy_coro_wrapped(g);
    if (O(g)->kind != APY_GEN_K)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'throw'%s",
                         apy_kind_name(g), "");
    /* NOT YET STARTED, or already finished: there is no suspension point to
       raise at, so it is raised here. */
    if (O(g)->v.g.state <= 0) {
        O(g)->v.g.state = -1;
        apy_raise(exc);
        return 0;
    }
    O(g)->v.g.pending = exc;
    out = apy_gen_step(g, apy_none(), &done);
    if (!out) return 0;
    if (done) return apy_fail("StopIteration", "");
    return out;
}

/* `g.close()` -- a GeneratorExit at the suspension point, so a `finally` in
   the body runs. The exception is SWALLOWED if it comes back out, which is
   what makes `close` quiet; anything else the body raised instead propagates,
   as CPython's does. */
APY_API apy_value apy_gen_close(apy_value g) {
    int done;
    /* A WRAPPER DELEGATES, having no body of its own -- see
       `apy_coro_wrapper`. */
    g = apy_coro_wrapped(g);
    if (O(g)->kind != APY_GEN_K)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'close'%s",
                         apy_kind_name(g), "");
    if (O(g)->v.g.state > 0) {
        apy_value exc = apy_make_exc(apy_lit("GeneratorExit"), apy_none());
        O(g)->v.g.pending = exc;
        apy_gen_step(g, apy_none(), &done);
        if (apy_error_occurred()) {
            if (apy_error_matches(apy_lit("GeneratorExit"))) apy_error_clear();
            else { O(g)->v.g.state = -1; return 0; }
        }
    }
    O(g)->v.g.state = -1;
    return apy_none();
}

/* Everything a generator will yield, as a list.

   EAGER, where CPython is lazy, and that is the visible limit of this design:
   a `for` over an infinite generator never starts rather than never ending.
   The lazy path exists -- `next(g)` steps once -- but `for` walks by index
   here, and an index walk needs a length. */
APY_API apy_value apy_gen_drain(apy_value g) {
    apy_value out = apy_seq_new(APY_LIST_K, 8);
    int64_t guard;
    for (guard = 0; guard < 1000000; guard++) {
        int done;
        apy_value v = apy_gen_step(g, apy_none(), &done);
        if (!v) return 0;
        if (done) break;
        apy_seq_push(out, v);
    }
    return out;
}

/* --- asyncio -------------------------------------------------------------
   A COROUTINE IS A GENERATOR, and this is the whole of what makes one run.

   `async def` lowers through the generator path: a frame, a step function
   re-entered per resume, and none of the body running until something drives
   it -- which is exactly what `await` and `asyncio.run` do here.

   WHAT THIS IS NOT. There is no clock and no I/O: `sleep` does not wait, it
   SUSPENDS, which is the only part of it a program can observe when nothing
   else is competing for the loop. A freestanding image has nothing to wait
   on, and pretending otherwise would mean a timer that never fires. */

enum { APY_CORO_SLEEP = 1, APY_CORO_GATHER = 2, APY_CORO_ANEXT = 3,
       APY_CORO_TASK = 4, APY_CORO_WAITFOR = 5, APY_CORO_VALUE = 6,
       APY_CORO_TGWAIT = 7,
       /* WHAT `a.asend(v)`, `a.athrow(e)` AND `a.aclose()` HAND BACK. An
          async generator's three methods answer an AWAITABLE rather than
          doing the work, which is the whole reason they are spelled with
          the `a` -- so each is a built-in coroutine holding the generator
          and what to do to it, and the work happens when it is awaited. */
       APY_CORO_ASEND = 8, APY_CORO_ATHROW = 9, APY_CORO_ACLOSE = 10 };

/* The steps for the kinds above. Declared here because `apy_await_step` --
   which dispatches on the kind -- comes before the task layer that defines
   them, and the object model each of them uses comes after both. */
static apy_value apy_task_step(apy_value t);
static apy_value apy_waitfor_step(apy_value w);
static apy_value apy_tgwait_step(apy_value w);
static apy_value apy_asend_step(apy_value g);
APY_API apy_value apy_type_set(apy_value cls, apy_value name, apy_value value);
static apy_value apy_native(int sel, int64_t arity, const char *name);
static apy_value apy_gather_step(apy_value g);

/* THE VIRTUAL CLOCK. Not a real one: nothing here waits, and `sleep(10)`
   returns as fast as `sleep(0)`. What it buys is ORDER -- two coroutines
   sleeping for different times must wake shortest-first, which is observable
   from the program and which round-robin gets wrong.

   Time only ever advances to the next moment something can happen: when
   everything is blocked, the driver jumps `now` to the earliest deadline. */
static double apy_now = 0.0;

/* WHERE A SUSPENSION LEAVES ITS WAKE TIME. Not in the value it hands back:
   an async generator yields values of its own through that same channel, and
   `yield 0.05` would have been read as a deadline. Only one coroutine steps
   at a time -- a step is synchronous from the driver's side -- so the time
   does not need to travel as a value at all.

   The driver CLEARS this, steps, and READS it: whatever suspended deepest has
   written the moment it wants back. */
static double apy_wake_at = 0.0;
static int apy_wake_set = 0;

static void apy_wake_clear(void) { apy_wake_set = 0; apy_wake_at = 0.0; }

/* The EARLIEST request wins, so a short sleep beside a long one decides how
   far the clock moves. */
static void apy_wake_note(double when) {
    if (!apy_wake_set || when < apy_wake_at) apy_wake_at = when;
    apy_wake_set = 1;
}

/* What a suspension hands back. An OPAQUE token carrying nothing, so that
   anything a program yields on purpose is unambiguous. */
static apy_value apy_suspend_token(void) {
    static apy_value tok = 0;
    if (!tok) tok = apy_str_copy("<suspend>", 9);
    return tok;
}

/* `await x`, one step of it.

   DELEGATION, NOT DRAINING. Each step of the awaited coroutine is handed back
   so the awaiting one can suspend too, which is what makes a whole chain of
   `await`s park together on one suspension point. Draining the inner one
   here would give the same answer for every case that awaits sequentially --
   and would have to be torn out the moment anything runs concurrently.

   FINISHING IS REPORTED AS `apy_stop()`, the same sentinel `apy_step` uses,
   rather than through an out-parameter -- the IR has no way to pass a pointer
   to a local. What the awaited coroutine returned is then read with
   `apy_gen_taken`, where its `return` already left it. */
APY_API apy_value apy_await_step(apy_value awaited, apy_value sent) {
    int fin = 0;
    apy_value out;
    if (O(awaited)->kind != APY_GEN_K)
        /* `await` on something that is not awaitable. Naming the kind is the
           whole diagnostic: it is nearly always a missing call, `await f`
           where `await f()` was meant. */
        return apy_fail2("TypeError", "object %s can't be used in 'await' "
                                      "expression%s", apy_kind_name(awaited), "");
    /* A BUILT-IN COROUTINE has no step function to re-enter, because it has
       no Python body. It is driven here instead. Handled before
       `apy_gen_step`, which would otherwise call through a null pointer. */
    if (!O(awaited)->v.g.step) {
        if (O(awaited)->v.g.builtin == APY_CORO_GATHER)
            return apy_gather_step(awaited);
        if (O(awaited)->v.g.builtin == APY_CORO_TASK)
            return apy_task_step(awaited);
        if (O(awaited)->v.g.builtin == APY_CORO_WAITFOR)
            return apy_waitfor_step(awaited);
        if (O(awaited)->v.g.builtin == APY_CORO_TGWAIT)
            return apy_tgwait_step(awaited);
        if (O(awaited)->v.g.builtin == APY_CORO_VALUE) {
            apy_gen_result(awaited, O(awaited)->v.g.slots[0]);
            return apy_stop();
        }
        if (O(awaited)->v.g.builtin == APY_CORO_ASEND
                || O(awaited)->v.g.builtin == APY_CORO_ATHROW
                || O(awaited)->v.g.builtin == APY_CORO_ACLOSE)
            return apy_asend_step(awaited);
        /* `sleep` ALWAYS SUSPENDS AT LEAST ONCE, before the clock is even
           consulted. `sleep(0)` is how a program hands control to the loop on
           purpose, and a version that returned immediately when the deadline
           had passed ran each coroutine straight to the end -- concurrency
           gone, and every conformance case still green because they only ever
           check the results.

           After that first suspension it is ready when the clock reaches its
           deadline, which is what wakes a short sleep before a long one. */
        if (O(awaited)->v.g.state == 0) {
            O(awaited)->v.g.state = 1;
            apy_wake_note(O(awaited)->v.g.deadline);
            return apy_suspend_token();
        }
        if (apy_now < O(awaited)->v.g.deadline) {
            apy_wake_note(O(awaited)->v.g.deadline);
            return apy_suspend_token();
        }
        return apy_stop();
    }
    out = apy_gen_step(awaited, sent, &fin);
    if (!out) return 0;
    return fin ? apy_stop() : out;
}

/* The suspend token as a value, so lowered code can compare against it.
   `async for` needs to tell "the generator suspended on an await" from "the
   generator produced an item", and those arrive through one channel. */
APY_API apy_value apy_suspend_value(void) { return apy_suspend_token(); }

/* One step of `async for v in agen`.

   AN ASYNC GENERATOR DOES TWO THINGS THROUGH ONE CHANNEL: `yield v` produces
   an item, and `await` inside it suspends. Both come back from the step as a
   value, and telling them apart is the whole of this function -- a suspension
   is the opaque token and nothing else can be, which is why the token stopped
   carrying the deadline.

   Answers the token when the generator suspended (pass it outward and come
   back), `apy_stop()` when it is exhausted, and otherwise the next item. */
/* What `async for` actually iterates: `__aiter__` of whatever was written.

   AN ASYNC GENERATOR IS ITS OWN ITERATOR, which is why this was skipped for
   so long -- every `async for` in the suite ran over one, and `__aiter__` on
   it answers itself. A CLASS is the other half of the protocol and the more
   common one in real code: `__aiter__` hands back the object that has
   `__anext__`, and `__anext__` is an `async def`, so each item arrives
   through a coroutine that may suspend before it produces one.

   That coroutine has to survive between steps -- the loop asks for one item
   at a time and a suspension means "no item yet, ask again" -- so the class
   is wrapped in a generator cell whose slots hold the iterator and whatever
   `__anext__` call is currently in flight. Nothing else in this runtime has
   somewhere to keep it. */
APY_API apy_value apy_aiter(apy_value src) {
    apy_value it, wrap;
    if (O(src)->kind == APY_GEN_K && O(src)->v.g.agen) return src;
    it = apy_dunder(src, "__aiter__");
    if (!it)
        return apy_fail2("TypeError",
                         "'async for' requires an object with "
                         "__aiter__ method, got %s%s",
                         apy_kind_name(src), "");
    it = apy_call_n(it, NULL, 0);
    if (!it) return 0;
    if (O(it)->kind == APY_GEN_K && O(it)->v.g.agen) return it;
    if (!apy_dunder(it, "__anext__")) {
        apy_error_clear();
        return apy_fail2("TypeError",
                         "'async for' requires an object with "
                         "__aiter__ method, got %s%s",
                         apy_kind_name(it), "");
    }
    wrap = apy_gen_new(0, 2);
    O(wrap)->v.g.agen = 1;
    O(wrap)->v.g.builtin = APY_CORO_ANEXT;
    O(wrap)->v.g.slots[0] = it;
    O(wrap)->v.g.slots[1] = 0;
    return wrap;
}

/* One step of an `async for` over a CLASS -- see `apy_aiter`.

   Answers the same three things every other step here does: the next item,
   the suspend token, or `apy_stop()`. `StopAsyncIteration` out of `__anext__`
   is exhaustion and not an error, which is the whole convention the protocol
   is built on. */
static apy_value apy_anext_step(apy_value g) {
    apy_value pending = O(g)->v.g.slots[1], v;
    if (!pending) {
        apy_value m = apy_dunder(O(g)->v.g.slots[0], "__anext__");
        if (!m) return 0;
        pending = apy_call_n(m, NULL, 0);
        if (!pending) {
            if (apy_error_matches(apy_lit("StopAsyncIteration"))) {
                apy_error_clear();
                return apy_stop();
            }
            return 0;
        }
        /* `__anext__` WRITTEN AS A PLAIN `def` answers the item itself rather
           than a coroutine. Ordinary Python -- `async for` awaits what it
           gets, and awaiting a value that is not awaitable is the error, so a
           class returning one directly is only an unusual way to write it. */
        if (O(pending)->kind != APY_GEN_K) return pending;
        O(g)->v.g.slots[1] = pending;
    }
    v = apy_await_step(pending, apy_none());
    if (!v) {
        O(g)->v.g.slots[1] = 0;
        if (apy_error_matches(apy_lit("StopAsyncIteration"))) {
            apy_error_clear();
            return apy_stop();
        }
        return 0;
    }
    if (v != apy_stop()) return v;              /* suspended: ask again */
    O(g)->v.g.slots[1] = 0;
    return apy_gen_taken(pending);
}

APY_API apy_value apy_agen_step(apy_value g) {
    int fin = 0;
    apy_value out;
    if (O(g)->kind != APY_GEN_K || !O(g)->v.g.agen)
        return apy_fail2("TypeError", "'%s' object does not support "
                                      "asynchronous iteration%s",
                         apy_kind_name(g), "");
    if (O(g)->v.g.builtin == APY_CORO_ANEXT) return apy_anext_step(g);
    if (O(g)->v.g.state < 0) return apy_stop();
    out = apy_gen_step(g, apy_none(), &fin);
    if (!out) return 0;
    return fin ? apy_stop() : out;
}

/* ONE STEP OF `asend`, `athrow` OR `aclose`, whichever built this awaitable.

   THE WORK HAPPENS HERE and not where the method was called, which is what
   makes `a.asend(1)` without an `await` do nothing -- as CPython's does. The
   generator is in slot 0 and the argument in slot 1; the builtin says which
   of the three this is.

   SUSPENSION PASSES THROUGH. An `await` inside the async generator's own
   body suspends it, and the token travels out to whoever is awaiting this --
   which is re-entered here and steps the generator again, exactly as
   `apy_anext_step` does for a class-based `__anext__`.

   EXHAUSTION IS `StopAsyncIteration`, which is the protocol's own end
   marker: `asend` on a finished generator raises it, and a `for` over the
   generator never sees it because `async for` reads `apy_agen_step`
   directly. */
static apy_value apy_asend_step(apy_value g) {
    apy_value agen = O(g)->v.g.slots[0];
    apy_value arg = O(g)->v.g.slots[1];
    apy_value out;
    int done = 0;
    if (O(g)->v.g.builtin == APY_CORO_ACLOSE) {
        if (!apy_gen_close(agen)) return 0;
        apy_gen_result(g, apy_none());
        return apy_stop();
    }
    if (O(g)->v.g.builtin == APY_CORO_ATHROW) {
        /* THE EXCEPTION IS DELIVERED ONCE. `apy_gen_throw` leaves it pending
           and steps, and a second entry after a suspension must not raise it
           again -- which is why the slot is cleared the moment it is used. */
        if (arg) {
            O(g)->v.g.slots[1] = 0;
            out = apy_gen_throw(agen, arg);
            if (!out) return 0;
            if (out == apy_suspend_token()) return out;
            apy_gen_result(g, out);
            return apy_stop();
        }
    }
    if (O(agen)->v.g.state < 0) return apy_fail("StopAsyncIteration", "");
    out = apy_gen_step(agen, arg ? arg : apy_none(), &done);
    if (!out) return 0;
    if (out == apy_suspend_token()) return out;
    if (done) return apy_fail("StopAsyncIteration", "");
    /* SENT ONCE. A resumption after a suspension carries None, exactly as a
       plain generator's does -- the value belongs to the `yield` that was
       waiting for it, not to every step of the same await. */
    O(g)->v.g.slots[1] = 0;
    apy_gen_result(g, out);
    return apy_stop();
}

/* THE THREE METHODS THEMSELVES, as the awaitables they answer. */
static apy_value apy_agen_await(apy_value agen, apy_value arg, int which) {
    apy_value g;
    if (O(agen)->kind != APY_GEN_K || !O(agen)->v.g.agen)
        return apy_fail2("TypeError", "'%s' object is not an async "
                                      "generator%s", apy_kind_name(agen), "");
    g = apy_gen_new(0, 2);
    if (!g) return 0;
    O(g)->v.g.coro = 1;
    O(g)->v.g.builtin = which;
    O(g)->v.g.slots[0] = agen;
    O(g)->v.g.slots[1] = arg;
    return g;
}

APY_API apy_value apy_agen_asend(apy_value agen, apy_value v) {
    return apy_agen_await(agen, v, APY_CORO_ASEND);
}

APY_API apy_value apy_agen_athrow(apy_value agen, apy_value exc) {
    return apy_agen_await(agen, exc, APY_CORO_ATHROW);
}

APY_API apy_value apy_agen_aclose(apy_value agen) {
    return apy_agen_await(agen, 0, APY_CORO_ACLOSE);
}

/* `asyncio.sleep(delay)`. A coroutine that suspends once and returns None.

   Built by hand rather than lowered from Python source, because there is no
   Python source in this compiler -- see `frontends/python/modules.py`. The
   frame is a generator with a NULL step function, which is how
   `apy_await_step` recognises it and drives it directly -- there is no Python
   body to re-enter. */
APY_API apy_value apy_asyncio_sleep(apy_value delay) {
    apy_value g = apy_gen_new(0, 0);
    double d = apy_is_num(delay) ? apy_as_float(delay) : 0.0;
    O(g)->v.g.coro = 1;
    O(g)->v.g.builtin = APY_CORO_SLEEP;
    /* THE DEADLINE IS TAKEN NOW, when `sleep` is called, not when it is first
       awaited -- exactly as a real loop does it, and the difference shows the
       moment a coroutine is created before the thing it races. */
    O(g)->v.g.deadline = apy_now + (d > 0.0 ? d : 0.0);
    return g;
}

/* `asyncio.gather(*coros)`. Runs them CONCURRENTLY and answers their results
   IN ARGUMENT ORDER -- which is the whole point of it, and not the order they
   happened to finish in.

   The arguments arrive as one tuple: the callable is variadic, so the frontend
   packs them. Slot 0 holds that tuple and slot 1 the result list, because a
   built-in coroutine has no registers that survive a suspension either. */
APY_API apy_value apy_asyncio_gather(apy_value coros) {
    apy_value g;
    int64_t i, n;
    if (!apy_is_seq(coros))
        return apy_fail2("TypeError", "gather() takes coroutines, not %s%s",
                         apy_kind_name(coros), "");
    n = O(coros)->v.q.n;
    g = apy_gen_new(0, 2);
    O(g)->v.g.coro = 1;
    O(g)->v.g.builtin = APY_CORO_GATHER;
    O(g)->v.g.slots[0] = coros;
    /* The results list is built FULL of None and filled in place, so that a
       coroutine finishing third still lands at its own index. Appending as
       they complete is what loses the ordering. */
    O(g)->v.g.slots[1] = apy_seq_new(APY_LIST_K, n ? n : 1);
    for (i = 0; i < n; i++) apy_seq_push(O(g)->v.g.slots[1], apy_none());
    return g;
}

/* One round of `gather`: advance every unfinished child once, then suspend.

   ROUND-ROBIN AND NOT ONE-AT-A-TIME, which is the difference between running
   concurrently and merely running. A child that suspends gives the round to
   the next one, so `gather` finishes in as many rounds as its slowest member
   needs rather than the sum of all of them -- and interleaving is observable,
   which is what makes this worth doing properly. */
static apy_value apy_gather_step(apy_value g) {
    apy_value coros = O(g)->v.g.slots[0];
    apy_value out = O(g)->v.g.slots[1];
    int64_t i, n = O(coros)->v.q.n, pending = 0;
    /* THE EARLIEST MOMENT ANY CHILD COULD MAKE PROGRESS. Suspending with the
       minimum rather than with the first one seen is what makes a short sleep
       wake before a long one when both are running here. */
    double soonest = 0.0;
    int have_soonest = 0;
    for (i = 0; i < n; i++) {
        apy_value child = O(coros)->v.q.items[i];
        apy_value v;
        if (O(child)->kind != APY_GEN_K) {
            /* Already finished, or never a coroutine. A non-coroutine is its
               own result -- `gather(3)` is not something to run. */
            O(out)->v.q.items[i] = child;
            continue;
        }
        /* A NEGATIVE STATE ALREADY MEANS FINISHED -- `apy_gen_step` sets it
           and reports done from it. Skipping those here is what stops a
           completed child being stepped again every round for as long as its
           slowest sibling runs. */
        if (O(child)->v.g.state < 0) continue;
        /* CLEARED BEFORE EACH CHILD, so what is read back afterwards is that
           child's request and not a sibling's left over from this round. */
        apy_wake_clear();
        v = apy_await_step(child, apy_none());
        if (!v) return 0;
        if (v == apy_stop()) {
            O(out)->v.q.items[i] = apy_gen_taken(child);
        } else {
            pending++;
            /* WHAT THIS CHILD ASKED FOR, read from where it left it. A child
               that suspended without naming a time is ready now, which keeps
               a plain `await` beside a sleep from stalling the clock past
               something that could already run. */
            if (!have_soonest || (apy_wake_set ? apy_wake_at : apy_now)
                                 < soonest) {
                soonest = apy_wake_set ? apy_wake_at : apy_now;
                have_soonest = 1;
            }
        }
    }
    if (pending) {
        apy_wake_note(have_soonest ? soonest : apy_now);
        return apy_suspend_token();
    }
    apy_gen_result(g, out);
    return apy_stop();
}

/* --- tasks ----------------------------------------------------------------

   A TASK IS A COROUTINE THE LOOP OWNS. `await coro` runs it inside the
   awaiting one; `create_task(coro)` hands it to the loop, which runs it
   whenever anything else suspends -- and that difference is the whole of what
   a task is for. Everything below exists to give the loop somewhere to keep
   them and something to do with one that has been cancelled. */

/* EVERY TASK THE PROGRAM HANDED OVER, so the loop can run them in the gaps.
   A list rather than a queue: they are stepped round-robin and a finished one
   is skipped, which is what `gather` already does for its children. */
/* REACHED THROUGH ONE FUNCTION so it can move -- see `apy_canonical_slot`. */
static apy_value apy_tasks_c;
APY_API apy_value apy_tasks_slot(void) {
    return (apy_value)&apy_tasks_c;
}
#define apy_tasks (*(apy_value *)apy_tasks_slot())

APY_API apy_value apy_asyncio_create_task(apy_value coro) {
    apy_value t;
    if (O(coro)->kind != APY_GEN_K || !O(coro)->v.g.coro)
        return apy_fail2("TypeError", "a coroutine was expected, got %s%s",
                         apy_kind_name(coro), "");
    t = apy_gen_new(0, 3);
    O(t)->v.g.coro = 1;
    O(t)->v.g.builtin = APY_CORO_TASK;
    O(t)->v.g.slots[0] = coro;      /* what it runs                        */
    O(t)->v.g.slots[1] = 0;         /* what it returned                    */
    O(t)->v.g.slots[2] = 0;         /* how it failed, if it did            */
    if (!apy_tasks) apy_tasks = apy_seq_new(APY_LIST_K, 4);
    apy_seq_push(apy_tasks, t);
    return t;
}

/* One step of a task, whether the loop is running it in a gap or a program is
   awaiting it. The two are the same act -- which is why `await task` after
   the loop has already finished it answers what it finished with rather than
   running it again. */
static apy_value apy_task_step(apy_value t) {
    apy_value child = O(t)->v.g.slots[0], v;
    if (O(t)->v.g.state < 0) {
        /* ALREADY FINISHED. A task that ended in an exception raises it
           again, which is what makes `await task` report a cancellation the
           loop delivered while the awaiter was elsewhere. */
        if (O(t)->v.g.slots[2]) {
            apy_raise(O(t)->v.g.slots[2]);
            return 0;
        }
        return apy_stop();
    }
    if (O(t)->v.g.cancel == 1) {
        O(t)->v.g.cancel = 2;
        if (O(child)->kind == APY_GEN_K && O(child)->v.g.state > 0) {
            /* AT THE SUSPENSION POINT, which is where CPython delivers it: a
               `try`/`except CancelledError` around the `await` inside the
               task catches it, and being catchable there is the whole of what
               cancellation means. */
            O(child)->v.g.pending = apy_make_exc0(apy_lit("CancelledError"));
        } else {
            /* NEVER STARTED, so there is no point to raise at and the task
               simply never runs. */
            O(t)->v.g.state = -1;
            O(t)->v.g.slots[2] = apy_make_exc0(apy_lit("CancelledError"));
            apy_raise(O(t)->v.g.slots[2]);
            return 0;
        }
    }
    v = apy_await_step(child, apy_none());
    if (!v) {
        /* HOW IT FAILED IS KEPT, because the loop may be the one that found
           out and the program may ask later. The flag stays set, so whoever
           was awaiting sees it now. */
        O(t)->v.g.state = -1;
        O(t)->v.g.slots[2] = apy_error_value();
        return 0;
    }
    if (v == apy_stop()) {
        O(t)->v.g.state = -1;
        O(t)->v.g.slots[1] = apy_gen_taken(child);
        /* AND WHERE `await` LOOKS FOR IT. `await task` reads what a finished
           coroutine returned off the cell it awaited, not off the child --
           so a task that answered 7 handed back None until this was here. */
        apy_gen_result(t, O(t)->v.g.slots[1]);
        return apy_stop();
    }
    return v;
}

/* Every task the loop owns, advanced once. Called wherever the thing being
   driven suspends -- that gap is exactly when a task may run. */
static apy_value apy_tasks_turn(void) {
    int64_t i;
    double soonest = 0.0;
    int have = 0;
    if (!apy_tasks) return apy_none();
    for (i = 0; i < O(apy_tasks)->v.q.n; i++) {
        apy_value t = O(apy_tasks)->v.q.items[i], v;
        if (O(t)->v.g.state < 0) continue;
        apy_wake_clear();
        v = apy_task_step(t);
        if (!v) {
            /* A TASK THAT FAILED IS NOT THE LOOP'S ERROR. It is recorded on
               the task and raised where the task is awaited -- which is what
               `asyncio` does, and why an un-awaited failing task is quiet. */
            apy_error_clear();
            continue;
        }
        if (v != apy_stop()) {
            if (!have || (apy_wake_set ? apy_wake_at : apy_now) < soonest) {
                soonest = apy_wake_set ? apy_wake_at : apy_now;
                have = 1;
            }
        }
    }
    apy_wake_clear();
    if (have) apy_wake_note(soonest);
    return apy_none();
}

/* `t.cancel()` -- ASK, do not raise. The exception arrives at the task's next
   suspension point; here it is only recorded. Answers True when there was a
   running task to ask, which is what CPython's does. */
APY_API apy_value apy_task_cancel(apy_value t) {
    if (O(t)->kind != APY_GEN_K || O(t)->v.g.builtin != APY_CORO_TASK)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'cancel'%s",
                         apy_kind_name(t), "");
    if (O(t)->v.g.state < 0) return apy_from_bool(0);
    if (!O(t)->v.g.cancel) O(t)->v.g.cancel = 1;
    return apy_from_bool(1);
}

/* `t.result()` -- what it returned, or the exception it ended with. */
APY_API apy_value apy_task_result(apy_value t) {
    if (O(t)->kind != APY_GEN_K || O(t)->v.g.builtin != APY_CORO_TASK)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'result'%s",
                         apy_kind_name(t), "");
    if (O(t)->v.g.state >= 0)
        return apy_fail("InvalidStateError", "Result is not set.");
    if (O(t)->v.g.slots[2]) { apy_raise(O(t)->v.g.slots[2]); return 0; }
    return O(t)->v.g.slots[1] ? O(t)->v.g.slots[1] : apy_none();
}

APY_API apy_value apy_task_done(apy_value t) {
    if (O(t)->kind != APY_GEN_K || O(t)->v.g.builtin != APY_CORO_TASK)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'done'%s",
                         apy_kind_name(t), "");
    return apy_from_bool(O(t)->v.g.state < 0);
}

APY_API apy_value apy_task_cancelled(apy_value t) {
    if (O(t)->kind != APY_GEN_K || O(t)->v.g.builtin != APY_CORO_TASK)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'cancelled'%s",
                         apy_kind_name(t), "");
    return apy_from_bool(O(t)->v.g.state < 0 && O(t)->v.g.slots[2]
                         && strcmp(O(O(t)->v.g.slots[2])->v.e.name,
                                   "CancelledError") == 0);
}

/* `asyncio.wait_for(coro, timeout)` -- run it, and give up at the deadline.

   THE CLOCK IS VIRTUAL and moves only to the next moment something can
   happen, so a timeout is not an approximation of waiting: it is a deadline
   among the others, and the loop reaching it first is exactly what "the
   coroutine took too long" means here. */
APY_API apy_value apy_asyncio_wait_for(apy_value coro, apy_value timeout) {
    apy_value w;
    double t = apy_is_num(timeout) ? apy_as_float(timeout) : -1.0;
    if (O(coro)->kind != APY_GEN_K)
        return apy_fail2("TypeError", "a coroutine was expected, got %s%s",
                         apy_kind_name(coro), "");
    w = apy_gen_new(0, 1);
    O(w)->v.g.coro = 1;
    O(w)->v.g.builtin = APY_CORO_WAITFOR;
    O(w)->v.g.slots[0] = coro;
    /* A negative or absent timeout is no deadline at all, which is what
       `wait_for(c, None)` means. */
    O(w)->v.g.deadline = t >= 0.0 ? apy_now + t : -1.0;
    return w;
}

static apy_value apy_waitfor_step(apy_value w) {
    apy_value child = O(w)->v.g.slots[0], v;
    double limit = O(w)->v.g.deadline;
    if (limit >= 0.0 && apy_now >= limit && O(child)->v.g.state >= 0) {
        /* THE CHILD IS STOPPED FIRST. `wait_for` promises not to leave it
           running, and a `finally` inside it runs on the way out. */
        if (O(child)->v.g.state > 0 && !apy_gen_close(child)) return 0;
        O(child)->v.g.state = -1;
        return apy_fail("TimeoutError", "");
    }
    apy_wake_clear();
    v = apy_await_step(child, apy_none());
    if (!v) return 0;
    if (v == apy_stop()) {
        apy_gen_result(w, apy_gen_taken(child));
        return apy_stop();
    }
    /* THE EARLIER OF the child's own wake and the deadline. Noting only the
       child's would let the clock jump past the moment this gives up. */
    if (limit >= 0.0) apy_wake_note(limit);
    return apy_suspend_token();
}

/* `asyncio.TaskGroup()`.

   AN OBJECT, not a coroutine: `async with` asks it for `__aenter__` and
   `__aexit__`, and `tg.create_task(...)` is an ordinary call between them.
   The class is built once and interned, so two groups share a type. */
APY_API apy_value apy_asyncio_taskgroup(void) {
    static apy_value cls;
    apy_value g;
    if (!cls) {
        cls = apy_type_new(apy_lit("TaskGroup"), 0);
        if (!cls) return 0;
        apy_type_set(cls, apy_lit("__aenter__"),
                     apy_native(APY_NAT_TG_ENTER, 1, "__aenter__"));
        apy_type_set(cls, apy_lit("__aexit__"),
                     apy_native(APY_NAT_TG_EXIT, 4, "__aexit__"));
        apy_type_set(cls, apy_lit("create_task"),
                     apy_native(APY_NAT_TG_CREATE, 2, "create_task"));
    }
    g = apy_instance_new(cls);
    if (!g) return 0;
    apy_setattr(g, apy_lit("_tasks"), apy_seq_new(APY_LIST_K, 4));
    if (apy_error_occurred()) return 0;
    return g;
}

/* A coroutine that is already finished, carrying one value. `__aenter__` has
   to answer an awaitable and has nothing to wait for. */
static apy_value apy_coro_value(apy_value v) {
    apy_value g = apy_gen_new(0, 1);
    O(g)->v.g.coro = 1;
    O(g)->v.g.builtin = APY_CORO_VALUE;
    O(g)->v.g.slots[0] = v;
    return g;
}

/* Leaving the `async with`: every task the group started runs to the end.

   THAT IS THE WHOLE PROMISE of a task group -- the block does not finish
   while its children are still going -- and it is why `t.result()` after the
   block is a question with an answer.

   WHAT IT DOES NOT DO is cancel the others when one of them fails, or when
   the block is left by an exception. CPython's group does both and collects
   what it cancelled into an `ExceptionGroup`; here every child runs to its
   end and a failing one is reported where it is awaited. The difference is
   visible only to a program whose tasks fail, and saying so is cheaper than
   a half-implementation of the cancellation cascade. */
static apy_value apy_tgwait_step(apy_value w) {
    apy_value tasks = O(w)->v.g.slots[0];
    int64_t i, pending = 0;
    double soonest = 0.0;
    int have = 0;
    for (i = 0; i < O(tasks)->v.q.n; i++) {
        apy_value t = O(tasks)->v.q.items[i], v;
        if (O(t)->v.g.state < 0) continue;
        apy_wake_clear();
        v = apy_task_step(t);
        if (!v) return 0;
        if (v != apy_stop()) {
            pending++;
            if (!have || (apy_wake_set ? apy_wake_at : apy_now) < soonest) {
                soonest = apy_wake_set ? apy_wake_at : apy_now;
                have = 1;
            }
        }
    }
    apy_wake_clear();
    if (pending) {
        apy_wake_note(have ? soonest : apy_now);
        return apy_suspend_token();
    }
    /* FALSE, not None: `__aexit__` answering truthy would swallow whatever
       exception was leaving the block. */
    apy_gen_result(w, apy_from_bool(0));
    return apy_stop();
}

/* Drive one coroutine to completion and answer what it returned.

   THE WHOLE EVENT LOOP for a single task: step until done, ignoring what it
   suspends with, because with one task there is never anything else to run
   in the gap. `gather` is where suspensions start to matter. */
APY_API apy_value apy_asyncio_run(apy_value coro) {
    int64_t guard;
    if (O(coro)->kind != APY_GEN_K || !O(coro)->v.g.coro)
        return apy_fail2("ValueError", "a coroutine was expected, got %s%s",
                         apy_kind_name(coro), "");
    for (guard = 0; guard < 100000000; guard++) {
        /* Through `apy_await_step` and not `apy_gen_step`, so that
           `asyncio.run(asyncio.sleep(0))` -- a coroutine with no Python body
           and so no step function to call -- is driven rather than jumped
           through as a null pointer. */
        apy_value v;
        apy_wake_clear();
        v = apy_await_step(coro, apy_none());
        if (!v) return 0;
        if (v == apy_stop()) {
            /* THE LOOP CLOSES WHAT THE PROGRAM ABANDONED, before answering.
               An `async for` left by `break` holds a generator suspended
               inside its own `try`, and its `finally` has not run yet. */
            apy_value result = apy_gen_taken(coro);
            if (apy_live_agens) {
                int64_t k;
                for (k = 0; k < O(apy_live_agens)->v.q.n; k++) {
                    apy_value ag = O(apy_live_agens)->v.q.items[k];
                    if (O(ag)->v.g.state > 0 && !apy_gen_close(ag)) return 0;
                }
                O(apy_live_agens)->v.q.n = 0;
            }
            return result;
        }
        /* THE CLOCK ONLY EVER MOVES FORWARD, and only to the next moment
           something can happen. Everything is blocked when this is reached,
           so jumping straight to the earliest deadline is not an
           approximation of waiting -- it is the whole of it, for a program
           that cannot observe duration. */
        /* THE GAP IS WHERE A TASK RUNS. What was being driven has
           suspended, so this is the moment anything the program handed to
           the loop can make progress -- and without it `create_task` would
           be an elaborate way of writing `await`. */
        {
            double mine = apy_wake_set ? apy_wake_at : apy_now;
            int had = apy_wake_set;
            apy_tasks_turn();
            if (apy_wake_set && had && mine < apy_wake_at) {
                apy_wake_at = mine;
            } else if (!apy_wake_set && had) {
                apy_wake_at = mine;
                apy_wake_set = 1;
            }
        }
        if (apy_wake_set && apy_wake_at > apy_now) apy_now = apy_wake_at;
    }
    return apy_fail("RuntimeError", "coroutine did not finish");
}

"""
