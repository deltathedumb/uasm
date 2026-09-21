"""The object runtime, in C: callables, cells, function objects, classes and attributes.

ONE PART OF ONE TRANSLATION UNIT. `c/__init__.py` concatenates
these in order and the result is the file it always was, so a
definition here may rely on anything in an earlier part and
nothing in a later one. Sections, in order:
  * callables, classes and instances
  * cells
  * function objects
  * classes and instances
  * attributes
"""

C = r"""/* --- callables, classes and instances ----------------------------------- */
/* Everything below exists to answer one question the rest of this file never
   had to: what is a value that can be CALLED, and what does calling it do.
   Until now a call was a fixed symbol the frontend picked at compile time, so
   `def` produced no value at all and a method was a name in a table here.
   `class` breaks both: a method is looked up on an object at run time, and
   `C(...)` calls something the compiler cannot name.

   THE CALLING CONVENTION, which every backend and the interpreter must agree
   on. A dynamic Python function compiles to an IR function

       f(env, p0, p1, ..., p[arity-1]) -> apy_value

   `env` is THE FUNCTION OBJECT THROUGH WHICH THE CALL WAS MADE, and it is
   first for one reason: a closure has to reach its captured boxes, and the
   only thing that knows which boxes THIS closure got is the closure itself.
   Two `def bump()` values made by two calls to the same enclosing function
   have the same `code` and different `cells`, so passing the code address
   alone would make them indistinguishable -- every closure would share one
   set of variables, which is the classic wrong answer here and one that
   passes every single-instance test.

   A function that captures nothing never reads `env`. That is not a special
   case in the ABI, just an unused parameter, and paying one register for
   uniformity is worth not having two calling conventions that could be
   confused for each other.

   WHY THE INDIRECT CALL LIVES HERE AND NOT IN THE IR. `apy_call` casts
   `code` to a function pointer and calls it. The IR has CALL_PTR and could
   express that directly, but then the arity switch below would be emitted by
   the frontend at every call site, once per possible argument count. One
   switch in C, reached by every backend through an ordinary call, is the same
   dispatch written once. */

APY_API apy_value apy_getattr(apy_value obj, apy_value name);
/* Declared here because the `typing` special forms below build an instance
   and stamp a name onto it, just above where both are defined. */
APY_API apy_value apy_setattr(apy_value obj, apy_value name, apy_value value);
APY_API apy_value apy_instance_new(apy_value cls);
static apy_value apy_call_n(apy_value f, apy_value *argv, int64_t argc);
static apy_value apy_type_of(apy_value v);

/* Interned dunder names. Building a str value per lookup would allocate on
   every `+` between instances; these are made once and compared by content
   like any other str. */
/* REACHED THROUGH TWO FUNCTIONS so the cache can move: both are ordinary
   exports with C bodies, and `runtime/containers.py` replaces them. The rows
   are PAIRS -- text then cell -- rather than two arrays, because the IR side
   reserves one block and a second reservation would be a second thing to keep
   in step. */
static apy_value apy_name_rows_c[48][2];
static int64_t apy_name_count_c;
APY_API apy_value apy_name_rows(void) { return (apy_value)apy_name_rows_c; }
APY_API apy_value apy_name_slot(void) { return (apy_value)&apy_name_count_c; }

APY_API apy_value apy_name_of(apy_value text) {
    int64_t i;
    for (i = 0; i < apy_name_count_c; i++)
        if (strcmp((const char *)apy_name_rows_c[i][0],
                   (const char *)text) == 0)
            return apy_name_rows_c[i][1];
    if (apy_name_count_c >= 48) return apy_lit((const char *)text);
    apy_name_rows_c[apy_name_count_c][0] = text;
    apy_name_rows_c[apy_name_count_c][1] = apy_lit((const char *)text);
    return apy_name_rows_c[apy_name_count_c++][1];
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_name(const char *text) {
    return apy_name_of((apy_value)(uintptr_t)text);
}

/* --- cells -------------------------------------------------------------- */
/* A captured variable's box. `functions/closure-cell-is-shared` is the case
   that decides this design: two closures made over one `n` must see each
   other's writes, so what they capture is the BOX and not the value in it.
   Copying the value at capture time passes every test where only one closure
   exists and fails that one. */

APY_API apy_value apy_cell_new(apy_value initial) {
    apy_obj *o = apy_alloc(APY_CELL_K);
    o->v.cell.slot = initial;
    return V(o);
}

APY_API apy_value apy_cell_get(apy_value c) { return O(c)->v.cell.slot; }

APY_API apy_value apy_cell_set(apy_value c, apy_value v) {
    O(c)->v.cell.slot = v;
    return apy_none();
}

/* --- function objects --------------------------------------------------- */

/* `code` is typed `apy_value` and is NOT one: it is the address the IR's
   FUNC_ADDR produced. Both are `uintptr_t`, and taking it as the IR's own
   pointer type is what lets the frontend pass a FUNC_ADDR register straight
   in -- an `int64_t` parameter would be the same bits and a type the backend
   would have to convert to, for no gain. */
APY_API apy_value apy_func_new(apy_value code, int64_t arity, apy_value name,
                               int64_t ncells, int64_t ndefaults,
                               int64_t vararg) {
    apy_obj *o = apy_alloc(APY_FUNC_K);
    o->v.fn.code = (uintptr_t)code;
    o->v.fn.arity = arity;
    o->v.fn.name = name;
    o->v.fn.ncells = ncells;
    o->v.fn.bound = 0;
    o->v.fn.ndefaults = ndefaults;
    o->v.fn.vararg = (int)vararg;
    o->v.fn.cells = ncells > 0
        ? (apy_value *)calloc((size_t)ncells, sizeof(apy_value)) : NULL;
    o->v.fn.defaults = ndefaults > 0
        ? (apy_value *)calloc((size_t)ndefaults, sizeof(apy_value)) : NULL;
    o->v.fn.pnames = NULL;
    o->v.fn.kwarg = 0;
    o->v.fn.kwonly = 0;
    o->v.fn.nkwdefault = 0;
    o->v.fn.posonly = 0;
    o->v.fn.doc = 0;
    o->v.fn.coro = 0;
    o->v.fn.is_type = 0;
    o->v.fn.dict = 0;
    o->v.fn.module = 0;
    o->v.fn.descr = 0;
    return V(o);
}

/* How many trailing parameters are keyword-only. Set after the object
   exists for the same reason the names and defaults are: the IR has no
   varargs, so each fact about a signature is its own call. */
APY_API apy_value apy_func_kwonly(apy_value f, int64_t n) {
    O(f)->v.fn.kwonly = (int)n;
    return f;
}

/* How many of the TRAILING DEFAULTS belong to keyword-only parameters.

   Not derivable from `kwonly`: a keyword-only parameter may be REQUIRED, and
   `def f(a, b=1, *args, c, **kw)` has one keyword-only parameter and one
   default that is not its. Splitting on `kwonly` alone therefore reported
   `b`'s default as `c`'s and left `__defaults__` empty. */
APY_API apy_value apy_func_kwdefaults(apy_value f, int64_t n) {
    O(f)->v.fn.nkwdefault = (int)n;
    return f;
}

/* The docstring. Set after the object exists, like the names and defaults. */
APY_API apy_value apy_func_doc(apy_value f, apy_value text) {
    O(f)->v.fn.doc = text;
    return f;
}

/* How many leading parameters are positional-only. */
APY_API apy_value apy_func_posonly(apy_value f, int64_t n) {
    O(f)->v.fn.posonly = (int)n;
    return f;
}

/* Mark the function as taking `**kw`. Set after the object exists for the
   same reason the parameter names and defaults are: the IR has no varargs. */
APY_API apy_value apy_func_kwarg(apy_value f, int64_t on) {
    O(f)->v.fn.kwarg = on != 0;
    return f;
}

/* One parameter NAME, at its declared index. Installed after the object
   exists for the same reason the cells and defaults are: the IR has no
   varargs, so each one is its own call. */
APY_API apy_value apy_func_param(apy_value f, int64_t i, apy_value name) {
    if (i < 0 || i >= O(f)->v.fn.arity) return f;
    if (!O(f)->v.fn.pnames)
        O(f)->v.fn.pnames = (apy_value *)calloc(
            (size_t)O(f)->v.fn.arity, sizeof(apy_value));
    if (O(f)->v.fn.pnames) O(f)->v.fn.pnames[i] = name;
    return f;
}

/* One default value, at index `i` among the LAST `ndefaults` parameters.
   Installed after the object exists for the same reason the cells are: the IR
   has no varargs. */
APY_API apy_value apy_func_default(apy_value f, int64_t i, apy_value value) {
    if (i >= 0 && i < O(f)->v.fn.ndefaults) O(f)->v.fn.defaults[i] = value;
    return f;
}

/* Install one captured box. Separate from `apy_func_new` because the IR has
   no varargs, and because a closure that captures itself -- a recursive inner
   `def` -- needs the function object to exist before the box naming it can be
   filled in. */
APY_API apy_value apy_func_cell(apy_value f, int64_t i, apy_value cell) {
    if (i >= 0 && i < O(f)->v.fn.ncells) O(f)->v.fn.cells[i] = cell;
    return f;
}

/* Read a captured box out of the env the callee was handed. */
APY_API apy_value apy_env_cell(apy_value env, int64_t i) {
    if (O(env)->kind != APY_FUNC_K || i < 0 || i >= O(env)->v.fn.ncells)
        return apy_fail("SystemError", "closure environment is not the one "
                                       "this function was compiled for");
    return O(env)->v.fn.cells[i];
}

/* A bound method: the same code and the same boxes, with a receiver attached.
   A FRESH cell every time, which is not waste -- CPython also builds a new
   method object per attribute access, and `datamodel/method-objects-are-
   created-per-access` measures exactly that. */
APY_API apy_value apy_bind_of(apy_value f, apy_value self) {
    apy_obj *o = apy_alloc(APY_FUNC_K);
    o->v.fn = O(f)->v.fn;
    o->v.fn.bound = self;
    return V(o);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_bind(apy_value f, apy_value self) {
    return apy_bind_of(f, self);
}

/* --- classes and instances ---------------------------------------------- */

APY_API apy_value apy_type_new(apy_value name, apy_value base) {
    apy_obj *o;
    if (base && O(base)->kind != APY_TYPE_K && O(base)->kind != APY_NONE_K)
        return apy_fail2("TypeError",
                         "bases must be types, not '%s'%s",
                         apy_kind_name(base), "");
    o = apy_alloc(APY_TYPE_K);
    o->v.t.name = name;
    o->v.t.base = (base && O(base)->kind == APY_TYPE_K) ? base : 0;
    o->v.t.dict = apy_dict_new(4);
    o->v.t.meta = 0;
    /* NOT INHERITED FROM THE UNION. A stale pointer here would give a fresh
       class somebody else's linearisation, which is a wrong answer that looks
       like a working program until one method resolves to the wrong body. */
    o->v.t.bases = 0;
    o->v.t.mro = 0;
    o->v.t.builtin = 0;
    /* NO QUALNAME UNTIL ONE IS GIVEN. A class qualifies as its own name
       unless a `class` statement nested it, and only the frontend knows
       that -- see `apy_type_qual`. */
    o->v.t.qual = 0;
    return V(o);
}

/* PEP 3155: the qualified name a NESTED `class` statement gives its class.

   WRITTEN BY THE FRONTEND, once, just after the class exists, and only when
   the qualname differs from the name -- which is to say only for a class
   inside a function or inside another class. `mk.<locals>.D` cannot be
   derived here: nothing the runtime holds says where the statement was
   written, and the `<locals>` marker is a fact about the source. */
APY_API apy_value apy_type_qual(apy_value cls, apy_value qual) {
    if (cls && O(cls)->kind == APY_TYPE_K)
        O(cls)->v.t.qual = qual;
    return cls;
}

/* `class D(dict)` -- which builtin kind this class extends. Set after the
   class exists, like its names and its metaclass. */
APY_API apy_value apy_type_builtin(apy_value cls, int64_t kind) {
    O(cls)->v.t.builtin = (int)kind;
    return cls;
}

/* The kind a class extends, looked up the whole chain: a subclass of a
   subclass of `dict` is still a dict. 0 when nothing in the chain does. */
static apy_value apy_inst_held(apy_value v);

APY_API int64_t apy_class_builtin_of(apy_value cls) {
    while (cls && O(cls)->kind == APY_TYPE_K) {
        if (O(cls)->v.t.builtin) return O(cls)->v.t.builtin;
        cls = O(cls)->v.t.base;
    }
    return 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_class_builtin(apy_value cls) {
    return (int)apy_class_builtin_of(cls);
}

/* MULTIPLE INHERITANCE, through the C3 linearisation in `apy_c3`. A class
   records both its bases and the order they linearise to, and every lookup
   walks that order; a class with one base has the same walk it always had,
   which is why the single-base path is left intact rather than replaced. */
APY_API apy_value apy_type_set(apy_value cls, apy_value name, apy_value value) {
    if (O(cls)->kind != APY_TYPE_K)
        return apy_fail2("TypeError", "'%s' object is not a class%s",
                         apy_kind_name(cls), "");
    if (!apy_dict_set(O(cls)->v.t.dict, name, value)) return 0;
    return apy_none();
}

/* Walk the base chain for `name`, without binding. Returns 0 and sets NO
   error when absent -- callers distinguish "not there" from "failed", and an
   attribute miss is only an AttributeError once every class in the chain has
   been asked. */
APY_API apy_value apy_class_find_of(apy_value cls, apy_value name) {
    /* THROUGH THE MRO WHEN THERE IS ONE. With a single base the two orders
       are the same walk; with several they are not, and the base chain finds
       the wrong body for `class D(B, C)` over a diamond. */
    if (cls && O(cls)->kind == APY_TYPE_K && O(cls)->v.t.mro) {
        apy_value order = O(cls)->v.t.mro;
        int64_t i;
        for (i = 0; i < O(order)->v.q.n; i++) {
            apy_value here = O(order)->v.q.items[i];
            int64_t at;
            if (O(here)->kind != APY_TYPE_K) continue;
            at = apy_dict_find(O(here)->v.t.dict, name);
            if (at >= 0)
                return O(here)->v.t.dict
                    ? O(O(here)->v.t.dict)->v.d.vals[at] : 0;
        }
        return 0;
    }
    while (cls && O(cls)->kind == APY_TYPE_K) {
        int64_t at = apy_dict_find(O(cls)->v.t.dict, name);
        if (at >= 0) return O(cls)->v.t.dict ? O(O(cls)->v.t.dict)->v.d.vals[at] : 0;
        cls = O(cls)->v.t.base;
    }
    return 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_class_find(apy_value cls, apy_value name) {
    return apy_class_find_of(cls, name);
}

/* The one class every `typing` special form is an instance of, made once
   and kept. CPython calls it `_SpecialForm`, and a program that prints
   `LiteralString.__class__.__name__` sees exactly that. */
APY_API apy_value apy_special_form_class(void) {
    static apy_value cls = 0;
    if (!cls) cls = apy_type_new(apy_lit("_SpecialForm"), 0);
    return cls;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */

/* `Final`, `LiteralString`, `Self`, ... A program may name one, annotate
   with it, and print its class; nothing else about it is observable, so one
   object kind carrying its own name covers the lot. */
APY_API apy_value apy_typing_form(apy_value name) {
    /* INTERNED BY NAME. `get_origin(Literal["a", "b"]) is Literal` is what a
       program tests, and it can only be True if the two mentions of `Literal`
       are one object. A fresh instance per mention is the obvious
       implementation and is wrong in exactly the way that test detects. */
    static apy_value seen = 0;
    apy_value found;
    if (!seen) seen = apy_dict_new(8);
    /* The NON-RAISING lookup: `apy_dict_get` reports a missing key as a
       KeyError, and a form not yet interned is the ordinary case here. */
    found = apy_dict_get_or(seen, name, 0);
    if (found) return found;
    {
    apy_value o = apy_instance_new(apy_special_form_class());
    if (!o) return 0;
    apy_setattr(o, apy_lit("_name"), name);
    if (apy_error_occurred()) return 0;
    apy_dict_set(seen, name, o);
    return o;
    }
}

/* PEP 695: `type Alias = list[int]`.

   A `TypeAliasType` is a NAME plus the thing it stands for, and neither means
   anything to the runtime beyond being readable back -- an alias annotates
   and never converts. Made an instance of an interned class so
   `type(Alias).__name__` answers `TypeAliasType`, which is what a program
   asks to tell an alias from the type it aliases. */
static apy_value apy_alias_class(void) {
    static apy_value cls = 0;
    if (!cls) cls = apy_type_new(apy_lit("TypeAliasType"), 0);
    return cls;
}

APY_API apy_value apy_type_alias(apy_value name, apy_value value,
                                 apy_value params) {
    apy_value o = apy_instance_new(apy_alias_class());
    if (!o) return 0;
    apy_dict_set(O(o)->v.o.dict, apy_lit("__name__"), name);
    apy_dict_set(O(o)->v.o.dict, apy_lit("__value__"), value);
    apy_dict_set(O(o)->v.o.dict, apy_lit("__type_params__"),
                 params ? params : apy_tuple_new(1));
    return o;
}

/* PEP 695's type PARAMETER, and PEP 484's `TypeVar` under one object: both are
   a name and nothing else at run time. */
/* `__import__(name)` -- a DYNAMIC import, which this compiler cannot do.

   There is no import machinery in a produced binary: every module a program
   uses is resolved and spliced where it is compiled, so a name computed at
   run time can never be one. Both answers below are honest and neither is
   silently wrong -- a module this build does not have is a
   ModuleNotFoundError exactly as in CPython, and one it does have is an
   ImportError saying the import cannot be performed dynamically. */
APY_API apy_value apy_import(apy_value name) {
    static const char *known[] = {
        "__future__", "_pyast", "_pycompile", "_pylex", "_pyparse",
        "_pyrun", "_pyvalidate", "abc", "annotationlib", "asyncio",
        "bisect", "collections", "collections.abc", "contextlib",
        "contextvars", "copy", "dataclasses", "datetime", "decimal",
        "enum", "fractions", "functools", "gc", "heapq", "inspect",
        "io", "itertools", "json", "keyword", "math", "numbers",
        "operator", "os", "pathlib", "random", "re", "select",
        "socket", "statistics", "string", "struct", "subprocess",
        "sys", "textwrap", "threading", "time", "tomllib", "traceback",
        "types", "typing", "unicodedata", "warnings", "weakref", 0};
    const char *want = O(name)->kind == APY_STR_K ? APY_CSTR(name) : "";
    int i;
    for (i = 0; known[i]; i++)
        if (strcmp(known[i], want) == 0)
            return apy_fail2("ImportError",
                             "cannot import '%s' dynamically: this build "
                             "resolves imports at compile time%s", want, "");
    return apy_fail2("ModuleNotFoundError", "No module named '%s'%s",
                     want, "");
}

APY_API apy_value apy_typevar(apy_value name) {
    static apy_value cls = 0;
    apy_value o;
    if (!cls) {
        cls = apy_type_new(apy_lit("TypeVar"), 0);
        /* PEP 696: `has_default()` is a METHOD and not an attribute, so the
           class needs one. Native, because the whole of it is "is the default
           slot filled" and there is no Python here to write it in. */
        apy_dict_set(O(cls)->v.t.dict, apy_name("has_default"),
                     apy_native(APY_NAT_HAS_DEFAULT, 1, "has_default"));
    }
    o = apy_instance_new(cls);
    if (!o) return 0;
    apy_dict_set(O(o)->v.o.dict, apy_lit("__name__"), name);
    apy_dict_set(O(o)->v.o.dict, apy_lit("__default__"), apy_none());
    return o;
}

/* PEP 696: the DEFAULT a type parameter was written with, or none. Set after
   the object exists because the default is an expression evaluated where the
   definition runs, exactly as a parameter's default is. */
APY_API apy_value apy_typevar_default(apy_value tv, apy_value value) {
    apy_dict_set(O(tv)->v.o.dict, apy_lit("__default__"), value);
    return tv;
}

/* Is this one of the interned typing forms? Subscripting one PARAMETERISES it
   -- `Literal["a", "b"]`, `TypeGuard[int]` -- rather than looking anything up,
   which is the same thing `list[int]` does to a builtin type. */
APY_API int64_t apy_is_special_form(apy_value v) {
    return O(v)->kind == APY_INST_K
        && O(v)->v.o.cls == apy_special_form_class();
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */

APY_API apy_value apy_getattr_default(apy_value obj, apy_value name,
                                      apy_value fallback);

/* `get_origin(x)` -- what was subscripted, or None.

   AN INSTANCE IS ASKED FOR ITS `__origin__`. `typing.Generic`'s own
   `__class_getitem__` is written in Python -- see `bundled/typing.py` -- so
   `Box[int]` is an ordinary instance rather than the alias kind `list[int]`
   builds, and both spellings carry the same two attributes. Reading the
   attribute covers the user generic without changing a single answer for the
   builtin one. A MISS IS NOT AN ERROR: `get_origin(3)` is None in CPython. */
APY_API apy_value apy_get_origin(apy_value v) {
    if (O(v)->kind == APY_ALIAS_K) return O(v)->v.ga.origin;
    if (O(v)->kind == APY_INST_K)
        return apy_getattr_default(v, apy_lit("__origin__"), apy_none());
    return apy_none();
}

/* `get_args(x)` -- what it was subscripted WITH, or the empty tuple. */
APY_API apy_value apy_get_args(apy_value v) {
    if (O(v)->kind == APY_ALIAS_K) return O(v)->v.ga.args;
    if (O(v)->kind == APY_INST_K)
        return apy_getattr_default(v, apy_lit("__args__"), apy_tuple_new(1));
    return apy_tuple_new(1);
}

/* `@final` on a class, `@override` on a method. BOTH RETURN THE ARGUMENT --
   a decorator that returned anything else would replace the thing it marks,
   and the marking is the whole of what they do. */
APY_API apy_value apy_typing_final(apy_value obj) {
    apy_setattr(obj, apy_lit("__final__"), apy_from_bool(1));
    if (apy_error_occurred()) return 0;
    return obj;
}

/* `@runtime_checkable`, `@no_type_check`. A decorator that marks a thing for
   a CHECKER and does nothing a running program can see -- so the honest
   implementation is to hand the argument back untouched. Distinct from
   `final`/`override` only in that no program reads an attribute afterwards;
   sharing their code would imply an attribute that is not there. */
APY_API apy_value apy_typing_mark(apy_value obj) { return obj; }

APY_API apy_value apy_typing_override(apy_value obj) {
    apy_setattr(obj, apy_lit("__override__"), apy_from_bool(1));
    if (apy_error_occurred()) return 0;
    return obj;
}

/* `__init_subclass__` -- the hook a base runs when a subclass is created.

   CALLED ON THE BASE, NOT THE NEW CLASS, and with the new class as its
   argument: a class does not announce its own creation to itself, which is
   why the lookup starts at the base. Implicitly a classmethod in CPython, so
   what it receives is the subclass and nothing else.

   Run after the class body has been filled, because the hook routinely reads
   what the body bound. */
APY_API apy_value apy_init_subclass(apy_value cls, apy_value kwd) {
    apy_value base, hook;
    if (O(cls)->kind != APY_TYPE_K) return apy_none();
    base = O(cls)->v.t.base;
    if (!base || O(base)->kind != APY_TYPE_K) return apy_none();
    hook = apy_class_find(base, apy_name("__init_subclass__"));
    if (!hook) {
        /* NOBODY WROTE ONE, SO `object`'s RUNS -- and it takes no keyword,
           so a class keyword with no hook to consume it is an error. This
           returned silently, and `class D(Plain, extra=1)` built a class
           where CPython raises. The message names the class BEING CREATED,
           as CPython's does: its classmethod is bound to that class.

           UNLESS A METACLASS TOOK THEM, which is the other place a class
           keyword can land and the reason the frontend hands the same dict
           to both. `class C(TypedDict, total=False)` gives `total` to
           `_TypedDictMeta.__new__`, which does not pass it on to
           `type.__new__` -- so CPython's `__init_subclass__` never sees it
           and there is nothing to refuse. The metaclass is asked FIRST here
           too, so by this point it has had them; refusing broke every
           metaclass that takes a class keyword, `typing`'s among them.

           THE LIMIT IS THAT A METACLASS FORWARDING THEM IS NOT TOLD APART.
           CPython refuses when `super().__new__(mcls, name, bases, ns,
           **kwds)` passes a keyword through to `type.__new__` and no hook
           consumes it; here the two spellings look the same, and the silence
           this restores is the one that stood before. */
        if (kwd && O(kwd)->kind == APY_DICT_K && O(kwd)->v.d.n
                && !O(cls)->v.t.meta) {
            char b[128];
            /* THE QUALNAME AND NOT THE NAME, which is this message and not
               the runtime's others: CPython words it
               `mk.<locals>.D.__init_subclass__()` for a class written
               inside a function while every refusal ABOUT an instance of
               one says plainly `'D' object`. */
            snprintf(b, sizeof b, "%s.__init_subclass__() takes no keyword "
                     "arguments", APY_CSTR(apy_type_qualname(cls)));
            return apy_fail("TypeError", b);
        }
        return apy_none();
    }
    {
        apy_value arg = cls;
        /* THE CLASS KEYWORDS TRAVEL WITH IT: `class A(Base, tag="a")` is how
           a program configures the hook, and dropping them left every
           subclass looking identically unconfigured -- the default, which is
           a wrong answer rather than a refusal.

           Whatever the hook answers is discarded: it is called for its
           effect, and CPython ignores the return too. */
        if (!apy_call_kw(hook, (apy_value)&arg, 1,
                         kwd ? kwd : apy_dict_new(1)))
            return 0;
    }
    return apy_none();
}

/* PEP 487: every descriptor a class body bound is TOLD ITS OWN NAME, once,
   after the body is complete. A descriptor cannot know it otherwise -- it is
   built by an expression that has no idea what it is about to be assigned to
   -- which is why the hook exists and why the class has to make the call. */
/* `__slots__ = ("v",)` and `v = 1` IN THE SAME BODY is a ValueError, at
   class creation. The slot and the class attribute would occupy the same
   name and the attribute would win silently, which is why CPython refuses it
   rather than picking one. Raised at run time, not refused at compile time:
   a program may catch it, and the case that measures this does. */
APY_API apy_value apy_check_slots(apy_value cls) {
    apy_value d, slots;
    int64_t i, n;
    if (O(cls)->kind != APY_TYPE_K) return apy_none();
    d = O(cls)->v.t.dict;
    slots = apy_dict_get_or(d, apy_name("__slots__"), 0);
    if (!slots) return apy_none();
    n = (O(slots)->kind == APY_STR_K) ? 1 : apy_raw_len(slots);
    if (apy_error_occurred()) { apy_error_clear(); return apy_none(); }
    for (i = 0; i < n; i++) {
        apy_value one = (O(slots)->kind == APY_STR_K) ? slots
                                                      : apy_key_at(slots, i);
        if (!one || O(one)->kind != APY_STR_K) continue;
        if (apy_dict_find(d, one) >= 0) {
            char buf[160];
            snprintf(buf, sizeof buf,
                     "'%s' in __slots__ conflicts with class variable",
                     APY_CSTR(one));
            return apy_fail("ValueError", buf);
        }
    }
    return apy_none();
}

/* A SLOT READ THROUGH THE CLASS -- `Ok.v` where `v` is in `__slots__` and the
   body bound nothing to it. CPython answers a `member_descriptor`, which is
   what tells a reader the name is a slot rather than a missing attribute. */
APY_API apy_value apy_member_descriptor(void) {
    static apy_value cls = 0;
    if (!cls) cls = apy_type_new(apy_lit("member_descriptor"), 0);
    return apy_instance_new(cls);
}

/* AND ITS SIBLING, WHICH IS WHAT A C-LEVEL ATTRIBUTE READS AS. A
   `member_descriptor` stands for a `__slots__` entry -- storage in the
   instance, at a fixed offset -- and a `getset_descriptor` stands for a pair
   of C functions called with whoever asked. CPython tells them apart by name
   and so does a program: `object.__dict__["__class__"]` is a
   getset_descriptor there, and `__class__` is the one name on `object` that
   is a rule rather than a slot.

   NOT CALLABLE, which is the point of using a descriptor cell here rather
   than a native: `object.__dict__["__class__"](x)` is a TypeError in CPython
   and a native would have answered `type(x)`. */
APY_API apy_value apy_getset_descriptor(void) {
    static apy_value cls = 0;
    if (!cls) cls = apy_type_new(apy_lit("getset_descriptor"), 0);
    return apy_instance_new(cls);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */

APY_API apy_value apy_set_names(apy_value cls) {
    int64_t i;
    apy_value d;
    if (O(cls)->kind != APY_TYPE_K) return apy_none();
    d = O(cls)->v.t.dict;
    for (i = 0; i < O(d)->v.d.n; i++) {
        apy_value member = O(d)->v.d.vals[i], hook, args[2];
        if (O(member)->kind != APY_INST_K) continue;
        hook = apy_class_find(O(member)->v.o.cls, apy_name("__set_name__"));
        if (!hook) continue;
        args[0] = cls;
        args[1] = O(d)->v.d.keys[i];
        if (!apy_call_n(apy_bind(hook, member), args, 2)) return 0;
    }
    return apy_none();
}

APY_API apy_value apy_instance_new(apy_value cls) {
    apy_obj *o;
    if (O(cls)->kind != APY_TYPE_K)
        return apy_fail2("TypeError", "'%s' object is not callable%s",
                         apy_kind_name(cls), "");
    o = apy_alloc(APY_INST_K);
    o->v.o.cls = cls;
    o->v.o.dict = apy_dict_new(4);
    /* AN INSTANCE OF A BUILTIN-EXTENDING CLASS CARRIES ONE. `class D(dict)`
       with only a `__missing__` in its body still has to BE a dict for
       everything it did not write, and this is the dict it is. */
    o->v.o.held = 0;
    {
        int kind = apy_class_builtin(cls);
        if (kind == APY_DICT_K) o->v.o.held = apy_dict_new(4);
        else if (kind == APY_LIST_K) o->v.o.held = apy_list_new(4);
        else if (kind == APY_SET_K) o->v.o.held = apy_set_new(4);
        else if (kind == APY_TUPLE_K) o->v.o.held = apy_tuple_new(1);
        else if (kind == APY_STR_K) o->v.o.held = apy_lit("");
        /* `class B(bytes)`. THE SHARED IMMUTABLE EMPTY, which is what
           `apy_bytes_copy` answers for a length of zero -- and the right
           cell precisely because a class can only extend `bytes` here and
           never `bytearray`: the two share this kind and are told apart by
           `mut`, which a kind number cannot carry. See `_BUILTIN_BASE_KIND`
           in frontends/python/dynamic.py, where that is decided. */
        else if (kind == APY_BYTES_K) o->v.o.held = apy_bytes_copy("", 0);
    }
    return V(o);
}

/* The builtin an instance carries, or 0. The one place anything asks. */
APY_API int64_t apy_class_builtin_kind(apy_value cls) {
    while (cls && O(cls)->kind == APY_TYPE_K) {
        if (O(cls)->v.t.builtin) return O(cls)->v.t.builtin;
        cls = O(cls)->v.t.base;
    }
    return 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */

APY_API apy_value apy_inst_held_of(apy_value v) {
    return O(v)->kind == APY_INST_K ? O(v)->v.o.held : 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_inst_held(apy_value v) {
    return apy_inst_held_of(v);
}

/* --- attributes --------------------------------------------------------- */

APY_API apy_value apy_kind_attr(apy_value obj, apy_value want);
APY_API apy_value apy_memoryview(apy_value src);
APY_API apy_value apy_kind_attr_of(apy_value obj, apy_value want,
                                  int64_t bind);
APY_API apy_value apy_kind_prototype(apy_value type_name);

/* WHAT A TYPE CARRIES, asked through a PROTOTYPE of its kind. `list.append`
   has no list to ask, so one is made -- empty, thrown away, and only ever
   used to decide which methods that kind has. 0 for a name nothing knows.

   REACHED OFF THE TYPE MAKES IT A DESCRIPTOR, which CPython names, reprs
   and qualifies differently from a function: `list.append` is a
   `method_descriptor`, prints as `<method 'append' of 'list' objects>` and
   answers `list.append` to `__qualname__`. This is the only place that knows
   WHICH type handed it over, so it is where both facts are recorded -- the
   cell `apy_kind_attr_of` builds is fresh per call, so writing to it cannot
   reach another type's copy.

   TWO CALLERS AND ONE BODY: a builtin type reached as a VALUE is a FUNC
   wearing `is_type`, and `type(iter([]))` is a TYPE cell `apy_type_for`
   minted from the kind's name. Both are asking what a kind carries. */
/* THE BUILTIN METHODS THAT TAKE NO RECEIVER, by kind and by name.

   `dict.keys` is an unbound INSTANCE method: `dict.keys(d)` writes the
   receiver out, which is what the stamp below is for. `str.maketrans` is a
   STATICMETHOD and `dict.fromkeys` a CLASSMETHOD, and neither takes one --
   so reading either off the type and calling it ate the first argument:
   `bm = bytes.maketrans; bm(b"ab", b"xy")` was `maketrans expected 3
   arguments, got 2`, the third being a receiver that is not there.

   EVERY ROW READ OFF CPYTHON rather than guessed: it is every name in a
   builtin type's `__dict__` whose value is a `staticmethod` or a
   `classmethod_descriptor`. `__class_getitem__` is on almost all of them and
   is tested separately, because it belongs to no particular kind and because
   it binds the TYPE where these bind the prototype.

   KEYED BY THE PAIR AND NOT BY THE NAME, because a name alone does not say:
   `fromhex` is a classmethod on bytes and another on float, and nothing
   stops a later kind from having an instance method of that name.

   BOUND TO THE PROTOTYPE, which is what carries the KIND. A classmethod
   needs it -- `bytes.fromhex` and `bytearray.fromhex` differ only in what
   they build -- and a staticmethod ignores the receiver, so one rule serves
   both. The interpreter's `_KIND_STATIC` is this list, and the two must
   agree. */
static int apy_kind_static(const char *owner, const char *name) {
    static const struct { const char *kind, *method; } rows[] = {
        {"str", "maketrans"},
        {"bytes", "maketrans"}, {"bytes", "fromhex"},
        {"bytearray", "maketrans"}, {"bytearray", "fromhex"},
        {"dict", "fromkeys"},
        {"int", "from_bytes"},
        {"float", "from_number"}, {"float", "fromhex"},
        {"float", "__getformat__"},
        {"complex", "from_number"},
        {"memoryview", "_from_flags"},
        {"type", "__prepare__"},
    };
    size_t i;
    for (i = 0; i < sizeof rows / sizeof rows[0]; i++)
        if (strcmp(owner, rows[i].kind) == 0
                && strcmp(name, rows[i].method) == 0)
            return 1;
    return 0;
}

/* STATIC, and declared in an earlier part so the descriptor path can reach
   it: the interpreter has its own `_kind_prototype` and does not need a
   binding for this one. */
static apy_value apy_type_kind_attr(apy_value type_obj, apy_value ownerv,
                                    apy_value name) {
    const char *owner = (const char *)ownerv;
    apy_value proto = apy_kind_prototype(ownerv);
    apy_value found;
    int bound;
    if (!proto) return 0;
    /* `__class_getitem__` IS A CLASSMETHOD and binds the TYPE, which is why
       `list.__class_getitem__(int)` takes only the key where
       `list.append(xs, 9)` takes a receiver first. Bound to the TYPE ITSELF
       and not to the prototype, because the receiver is what the repr names
       -- CPython's reads `of type object at ...` -- and because the body
       reads it too: the `__class_getitem__` arm of `apy_native_call` takes a
       type straight as the alias's origin.

       AND A BOUND ONE IS NOT A DESCRIPTOR: `list.append` is a
       `method_descriptor` and `list.__class_getitem__` a
       `builtin_function_or_method`, so the stamp below is not reached. */
    bound = strcmp(APY_CSTR(name), "__class_getitem__") == 0;
    found = apy_kind_attr_of(proto, (apy_value)(uintptr_t)APY_CSTR(name), 0);
    if (found && bound) return apy_bind(found, type_obj);
    /* AND THE STATIC AND CLASS METHODS BIND THE PROTOTYPE -- see
       `apy_kind_static`, which says which they are and why the prototype and
       not the type. */
    if (found && apy_kind_static(owner, APY_CSTR(name)))
        return apy_bind(found, proto);
    if (found && O(found)->kind == APY_FUNC_K) {
        char qual[128];
        snprintf(qual, sizeof qual, "%s.%s", owner, APY_CSTR(name));
        O(found)->v.fn.qualname = apy_str_copy(qual, (int64_t)strlen(qual));
        O(found)->v.fn.descr = 1;
    }
    return found;
}

/* Every builtin kind's lookup ends here, which is why the protocol table is
   consulted HERE rather than in each of a dozen branches: a kind that has
   `__iter__` reaches this line for it exactly as one that has nothing does. */
APY_API apy_value apy_no_attribute(apy_value obj, apy_value name) {
    apy_value found = apy_kind_attr(
        obj, (apy_value)(uintptr_t)APY_CSTR(name));
    if (found) return found;
    /* THE TYPE ITSELF answers for its instances: `issubclass(dict, Mapping)`
       is a question about what a dict CAN DO, asked of `dict` and never of
       one. Unbound, because `dict.keys` is unbound in CPython too. */
    if (O(obj)->kind == APY_FUNC_K && O(obj)->v.fn.is_type) {
        found = apy_type_kind_attr(
            obj, (apy_value)(uintptr_t)APY_CSTR(O(obj)->v.fn.name), name);
        if (found) return found;
    }
    /* A TYPE IS NAMED, NOT DESCRIBED. CPython says `type object 'list' has
       no attribute 'nope'` where a VALUE gets `'int' object has no attribute
       'nope'` -- the type's own name is what the reader needs, and
       `apy_kind_name` of one is only ever the word `type`. */
    if (O(obj)->kind == APY_FUNC_K && O(obj)->v.fn.is_type)
        return apy_fail2("AttributeError",
                         "type object '%s' has no attribute '%s'",
                         APY_CSTR(O(obj)->v.fn.name), APY_CSTR(name));
    return apy_fail2("AttributeError", "'%s' object has no attribute '%s'",
                     apy_kind_name(obj), APY_CSTR(name));
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */

/* The DEFAULT attribute lookup: instance dict, then class, then
   `__getattr__`. Named separately from `apy_getattr` because a class that
   overrides `__getattribute__` needs a way to do what it overrode -- and
   `object.__getattribute__(self, name)` is how Python spells that. */
APY_API apy_value apy_default_getattr(apy_value obj, apy_value name);

/* `property(fget)`, `classmethod(f)`, `staticmethod(f)`. One constructor for
   the three because they differ only in what reading one does. */
APY_API apy_value apy_descr_new(apy_value fn, int64_t kind) {
    apy_obj *o = apy_alloc(APY_PROP_K);
    o->v.p.get = fn;
    o->v.p.set = 0;
    o->v.p.del_ = 0;
    o->v.p.kind = (int)kind;
    return V(o);
}

/* `@v.setter` -- a NEW property carrying the original getter and this
   setter. New rather than mutated because the decorator's result is bound to
   the name afterwards, and a program that kept the old object around must not
   see it change under it. */
APY_API apy_value apy_prop_setter(apy_value prop, apy_value fn) {
    apy_value out;
    if (O(prop)->kind != APY_PROP_K)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'setter'%s",
                         apy_kind_name(prop), "");
    out = apy_descr_new(O(prop)->v.p.get, APY_PROP_PROPERTY);
    O(out)->v.p.set = fn;
    O(out)->v.p.del_ = O(prop)->v.p.del_;
    return out;
}

/* `@v.deleter`. The third of the three, and the one that was missing --
   `del obj.v` had a slot to read (`v.p.del_`) and no way to fill it. */
APY_API apy_value apy_prop_deleter(apy_value prop, apy_value fn) {
    apy_value out;
    if (O(prop)->kind != APY_PROP_K)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'deleter'%s",
                         apy_kind_name(prop), "");
    out = apy_descr_new(O(prop)->v.p.get, APY_PROP_PROPERTY);
    O(out)->v.p.set = O(prop)->v.p.set;
    O(out)->v.p.del_ = fn;
    return out;
}

/* `@v.getter`, the mirror of it. */
APY_API apy_value apy_prop_getter(apy_value prop, apy_value fn) {
    apy_value out;
    if (O(prop)->kind != APY_PROP_K)
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute 'getter'%s",
                         apy_kind_name(prop), "");
    out = apy_descr_new(fn, APY_PROP_PROPERTY);
    O(out)->v.p.set = O(prop)->v.p.set;
    O(out)->v.p.del_ = O(prop)->v.p.del_;
    return out;
}

"""
