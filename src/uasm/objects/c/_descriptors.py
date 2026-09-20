"""The object runtime, in C: the descriptor protocol.

ONE PART OF ONE TRANSLATION UNIT. `c/__init__.py` concatenates
these in order and the result is the file it always was, so a
definition here may rely on anything in an earlier part and
nothing in a later one. Sections, in order:
  * the descriptor protocol
"""

C = r"""/* --- the descriptor protocol -------------------------------------------
   An object on a CLASS that decides for itself what reading or writing it
   through an instance means. `property`, `classmethod` and `staticmethod` are
   all built on it, and a user class defining `__get__` is one too.

   TWO KINDS, and the difference is only which side of the instance dict they
   sit on. A DATA descriptor defines `__set__` (or `__delete__`) and wins over
   the instance dict; a NON-DATA one defines only `__get__` and loses to it.
   That is what lets `c.v = 4` reach a property's setter while an ordinary
   method can still be shadowed by an attribute of the same name. */
APY_API int64_t apy_is_descriptor_of(apy_value v) {
    if (O(v)->kind == APY_PROP_K) return 1;
    return O(v)->kind == APY_INST_K
        && apy_class_find(O(v)->v.o.cls, apy_name("__get__")) != 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_is_descriptor(apy_value v) {
    return (int)apy_is_descriptor_of(v);
}

APY_API int64_t apy_is_data_descriptor_of(apy_value v) {
    if (O(v)->kind == APY_PROP_K) return O(v)->v.p.kind == APY_PROP_PROPERTY;
    return O(v)->kind == APY_INST_K
        && (apy_class_find(O(v)->v.o.cls, apy_name("__set__")) != 0
            || apy_class_find(O(v)->v.o.cls, apy_name("__delete__")) != 0);
}

/* Read through a descriptor: `d.__get__(obj, type)`. */
APY_API apy_value apy_descr_get_of(apy_value d, apy_value obj,
                                  apy_value cls) {
    if (O(d)->kind == APY_PROP_K) {
        apy_value argv[2];
        switch (O(d)->v.p.kind) {
        case APY_PROP_PROPERTY:
            /* THROUGH THE CLASS, A PROPERTY IS ITSELF. `Base.v` is the
               property object -- which is how `Base.v.fget(self)` reaches the
               base getter from an override. Calling the getter with no
               instance instead returned whatever it computed from nothing,
               and the program then looked for `.fget` on a str. */
            if (!obj) return d;
            if (!O(d)->v.p.get)
                return apy_fail("AttributeError", "unreadable attribute");
            argv[0] = obj;
            return apy_call_n(O(d)->v.p.get, argv, 1);
        case APY_PROP_CLASSMETHOD:
            /* Bound to the CLASS, not the instance -- and to the class the
               lookup started from, so `D.make()` on a subclass sees `D`. */
            return apy_bind(O(d)->v.p.get, cls);
        default:
            /* `staticmethod`: no binding at all, which is its whole point. */
            return O(d)->v.p.get;
        }
    }
    {
        apy_value m = apy_class_find(O(d)->v.o.cls, apy_name("__get__"));
        apy_value argv[2];
        argv[0] = obj ? obj : apy_none();
        argv[1] = cls ? cls : apy_none();
        return apy_call_n(apy_bind(m, d), argv, 2);
    }
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_descr_get(apy_value d, apy_value obj,
                               apy_value cls) {
    return apy_descr_get_of(d, obj, cls);
}

/* Write through a descriptor: `d.__set__(obj, value)`. Answers 0 and leaves
   the flag set on failure, 1 when it handled the write, -1 when this is not
   a data descriptor and the caller should store normally. */
APY_API int64_t apy_descr_set_of(apy_value d, apy_value obj,
                                apy_value value) {
    if (!apy_is_data_descriptor(d)) return -1;
    if (O(d)->kind == APY_PROP_K) {
        apy_value argv[2];
        if (!O(d)->v.p.set)
            return apy_fail("AttributeError",
                            "can't set attribute") ? 1 : 0;
        argv[0] = obj;
        argv[1] = value;
        return apy_call_n(O(d)->v.p.set, argv, 2) ? 1 : 0;
    }
    {
        apy_value m = apy_class_find(O(d)->v.o.cls, apy_name("__set__"));
        apy_value argv[2];
        if (!m) return -1;
        argv[0] = obj;
        argv[1] = value;
        return apy_call_n(apy_bind(m, d), argv, 2) ? 1 : 0;
    }
}

/* `getattr(x, 'a', fallback)`. A MISS IS NOT AN ERROR HERE, which is the
   whole difference from the two-argument form -- so the pending AttributeError
   is cleared and the fallback answered. Only an AttributeError is swallowed:
   a `__getattr__` that raised something of its own is the program's error and
   has to survive, or this would turn every failure inside a property into a
   silent default. */
APY_API apy_value apy_getattr_default(apy_value obj, apy_value name,
                                      apy_value fallback) {
    apy_value got = apy_getattr(obj, name);
    if (got) return got;
    if (apy_error_matches(apy_lit("AttributeError"))) {
        apy_error_clear();
        return fallback;
    }
    return 0;
}

APY_API apy_value apy_getattr(apy_value obj, apy_value name) {
    /* A str SUBCLASS IS AN ATTRIBUTE NAME. `getattr("abc", S("upper"))` is
       ordinary Python; the instance reached the lookup as a name and the
       text read out of it was whatever lay at the instance's address. */
    name = apy_text_like(name);
    /* `__getattribute__` INTERCEPTS EVERYTHING, before the instance dict is
       even looked at -- that is what distinguishes it from `__getattr__`,
       which is consulted only after a miss. Asked here rather than inside the
       default lookup so that the default remains callable from inside the
       override without recursing. */
    if (O(obj)->kind == APY_INST_K) {
        apy_value hook = apy_class_find(O(obj)->v.o.cls,
                                        apy_name("__getattribute__"));
        if (hook) {
            apy_value argv[2];
            argv[0] = obj;
            argv[1] = name;
            return apy_call_n(apy_bind(hook, obj), argv + 1, 1);
        }
    }
    return apy_default_getattr(obj, name);
}

/* Is `name` DECLARED in this class's `__slots__`, or in an ancestor's?

   NOT `apy_slot_allows`, which answers a DIFFERENT QUESTION and was asked
   here by mistake: that one says whether an INSTANCE may carry this
   attribute at all, and it answers yes for every name the moment one class
   in the chain omits `__slots__` -- which is right for a write and wrong for
   this. The effect was that a class declaring `__slots__ = ()` and
   inheriting from one that does not answered a `member_descriptor` for EVERY
   name it did not define itself, including methods reached through its
   METACLASS: `WithSlots.__subclasscheck__` came back as a descriptor and
   `isinstance(x, WithSlots)` died with `'member_descriptor' object is not
   callable`. `os.PathLike` is exactly that shape, so PEP 519 passed in the
   interpreter and failed compiled. */
static int apy_slot_declares(apy_value cls, apy_value name) {
    apy_value here = cls;
    int64_t at, i, n;
    while (here && O(here)->kind == APY_TYPE_K) {
        at = apy_dict_find(O(here)->v.t.dict, apy_name("__slots__"));
        if (at >= 0) {
            apy_value names = O(O(here)->v.t.dict)->v.d.vals[at];
            /* A BARE STRING IS ONE SLOT, not a run of one-character ones. */
            if (O(names)->kind == APY_STR_K) {
                if (apy_eq_raw(names, name)) return 1;
            } else {
                n = apy_raw_len(names);
                if (apy_error_occurred()) { apy_error_clear(); n = 0; }
                for (i = 0; i < n; i++)
                    if (apy_eq_raw(apy_key_at(names, i), name)) return 1;
            }
        }
        here = O(here)->v.t.base;
    }
    return 0;
}

/* Generated, and spliced into a LATER part -- see `_gen_kindmeth.py`. */
static const char *apy_kind_doc(const char *kind);
/* AND THE `dir()` TABLE BESIDE IT, read here for ONE bit of information: is
   this a kind the generated pair knows about at all? A cursor's type has no
   docstring in CPython and `iter([]).__doc__` is None rather than an
   AttributeError, so "no text" and "no such kind" have to be told apart and
   only the dir table can tell them. */
static const char *apy_kind_dir(const char *kind);

/* WHICH OF A VIEW'S NAMES IS A FIELD rather than a method. Only these are
   read off the cell, and only these are refused once the view is released:
   CPython raises when a method is CALLED on a released view, not when it is
   looked up. */
static int apy_mview_field(const char *want) {
    return strcmp(want, "readonly") == 0 || strcmp(want, "nbytes") == 0
        || strcmp(want, "itemsize") == 0 || strcmp(want, "format") == 0
        || strcmp(want, "obj") == 0 || strcmp(want, "ndim") == 0
        || strcmp(want, "shape") == 0 || strcmp(want, "strides") == 0
        || strcmp(want, "suboffsets") == 0
        || strcmp(want, "c_contiguous") == 0
        || strcmp(want, "f_contiguous") == 0
        || strcmp(want, "contiguous") == 0;
}

/* `f.__code__` -- ENOUGH OF ONE to answer what a program asks a function
   about its own signature. Not a real code object: there is no bytecode here
   to describe, and `co_argcount` and `co_varnames` are what introspection
   actually reads.

   TWO CALLERS AND ONE BODY. A generator's `gi_code` is the code of the `def`
   it came from -- in CPython literally the same object -- and the generator
   carries a FUNC describing that `def`, so both questions are this one. */
static apy_value apy_func_code(apy_value fn) {
    static apy_value cls = 0;
    apy_value code, names;
    int64_t declared = O(fn)->v.fn.arity
        - (O(fn)->v.fn.vararg ? 1 : 0)
        - (O(fn)->v.fn.kwarg ? 1 : 0);
    int64_t i;
    if (!cls) cls = apy_type_new(apy_lit("code"), 0);
    code = apy_instance_new(cls);
    if (!code) return 0;
    names = apy_tuple_new(declared + 3);
    for (i = 0; i < declared; i++)
        if (O(fn)->v.fn.pnames && O(fn)->v.fn.pnames[i])
            apy_seq_push(names, O(fn)->v.fn.pnames[i]);
    /* `*rest` AND `**kw` COME LAST, after every declared parameter -- which
       is where CPython puts them and where `inspect` expects to find them.
       Omitted entirely before, so a signature rebuilt from `co_varnames` had
       no variadic parts at all. */
    for (i = declared; i < O(fn)->v.fn.arity; i++)
        if (O(fn)->v.fn.pnames && O(fn)->v.fn.pnames[i])
            apy_seq_push(names, O(fn)->v.fn.pnames[i]);
    apy_setattr(code, apy_lit("co_argcount"),
                apy_from_int(declared - O(fn)->v.fn.kwonly));
    apy_setattr(code, apy_lit("co_posonlyargcount"),
                apy_from_int(O(fn)->v.fn.posonly));
    apy_setattr(code, apy_lit("co_kwonlyargcount"),
                apy_from_int(O(fn)->v.fn.kwonly));
    /* THE FLAGS `inspect` READS: 0x04 is `*rest` and 0x08 is `**kw`, which is
       how a signature knows the variadic parts exist without a second field
       to carry them. */
    apy_setattr(code, apy_lit("co_flags"),
                apy_from_int((O(fn)->v.fn.vararg ? 4 : 0)
                             | (O(fn)->v.fn.kwarg ? 8 : 0)));
    apy_setattr(code, apy_lit("co_varnames"), names);
    apy_setattr(code, apy_lit("co_name"), O(fn)->v.fn.name);
    apy_setattr(code, apy_lit("co_qualname"),
                O(fn)->v.fn.qualname ? O(fn)->v.fn.qualname
                                     : O(fn)->v.fn.name);
    if (apy_error_occurred()) return 0;
    return code;
}

/* THE FRAME A SUSPENDED GENERATOR IS SITTING IN -- enough of one, for the
   reason the code object above is enough of one. `gi_frame` is read to ask
   where a generator is and what it can still see, and the two things a
   program reads off it are its code and whether it is there at all. */
static apy_value apy_gen_frame(apy_value g) {
    static apy_value cls = 0;
    apy_value frame, code, from;
    if (!cls) cls = apy_type_new(apy_lit("frame"), 0);
    from = O(g)->v.g.sig ? O(g)->v.g.sig : O(g)->v.g.step;
    code = from && O(from)->kind == APY_FUNC_K ? apy_func_code(from) : 0;
    if (!code) return 0;
    frame = apy_instance_new(cls);
    if (!frame) return 0;
    apy_setattr(frame, apy_lit("f_code"), code);
    apy_setattr(frame, apy_lit("f_back"), apy_none());
    apy_setattr(frame, apy_lit("f_lineno"), apy_from_int(0));
    /* WHERE THE BODY IS, as far as this runtime can say it: the resume state
       a `yield` left behind, and -1 before the first step -- which is the
       number CPython uses for "has not run an instruction yet". */
    apy_setattr(frame, apy_lit("f_lasti"),
                apy_from_int(O(g)->v.g.state > 0 ? O(g)->v.g.state : -1));
    apy_setattr(frame, apy_lit("f_globals"), apy_dict_new(1));
    apy_setattr(frame, apy_lit("f_locals"), apy_dict_new(1));
    apy_setattr(frame, apy_lit("f_builtins"), apy_dict_new(1));
    apy_setattr(frame, apy_lit("f_trace"), apy_none());
    if (apy_error_occurred()) return 0;
    return frame;
}

APY_API apy_value apy_default_getattr(apy_value obj, apy_value name) {
    const char *want = APY_CSTR(name);
    /* PEP 257 FOR THE BUILTINS. `"".__doc__` IS `str.__doc__` -- the text
       `help` prints and every docstring tool reads -- and it was an
       AttributeError on every builtin value there is. Answered before the
       switch because every one of those kinds has one and none of them
       stores it; the four that carry their own are excluded, because an
       instance's comes from its class and a function's from its `def`. */
    if (strcmp(want, "__doc__") == 0
            && O(obj)->kind != APY_INST_K && O(obj)->kind != APY_FUNC_K
            && O(obj)->kind != APY_TYPE_K && O(obj)->kind != APY_EXC_K) {
        const char *kn = apy_kind_name(obj);
        const char *doc = apy_kind_doc(kn);
        if (doc) return apy_lit(doc);
        /* A KIND WE MODEL WHOSE TYPE HAS NO DOCSTRING ANSWERS None, which is
           what CPython does: `iter([]).__doc__` is None and `__doc__` is on
           the list `dir()` gives, so refusing it would be a list that lies.
           Told apart from a kind nothing knows by the dir table. */
        if (apy_kind_dir(kn)) return apy_none();
    }
    /* AND A TYPE OBJECT REACHED THROUGH `type(x)` IS A CELL, not the thunk
       the program's own `bytes` names -- `apy_type_for` mints one keyed by
       the kind's name for anything the canonical table has no entry for. So
       `type(b"").__doc__` took this path and `bytes.__doc__` took the FUNC
       one below, and only the second answered. A class the PROGRAM wrote
       that happens to share a builtin's name is excluded by its own
       `__doc__`, which the body bound and which is found first. */
    if (strcmp(want, "__doc__") == 0 && O(obj)->kind == APY_TYPE_K
            && O(obj)->v.t.dict
            && !apy_dict_get_or(O(obj)->v.t.dict, name, 0)) {
        const char *tn = APY_CSTR(O(obj)->v.t.name);
        const char *doc = apy_kind_doc(tn);
        if (doc) return apy_lit(doc);
        if (apy_kind_dir(tn)) return apy_none();
    }
    switch (O(obj)->kind) {
    case APY_INST_K: {
        /* THE INSTANCE DICT WINS over the class. `self.x = 1` in `__init__`
           shadows a class attribute `x`, which is what makes a class body's
           `count = 0` a shared default that each instance can override.
           (CPython puts data DESCRIPTORS ahead of the instance dict; there
           are no descriptors here, so this order is the whole rule.)

           THIS IS THE DEFAULT LOOKUP -- what `object.__getattribute__` is.
           A class overriding `__getattribute__` is asked first by
           `apy_getattr`, and reaches this by calling the default explicitly,
           which is the only way out of the recursion. */
        int64_t at;
        apy_value found;
        /* A DATA DESCRIPTOR ON THE CLASS BEATS THE INSTANCE DICT. That is the
           one place the "instance wins" rule above does not hold, and it is
           what makes a `property` a property: `c.v = 4` runs its setter and
           the instance dict never gets a `v` to shadow it with. A NON-data
           descriptor -- `__get__` and no `__set__` -- loses to the instance
           dict instead, which is how a method can be shadowed by an
           attribute of the same name. */
        found = apy_class_find(O(obj)->v.o.cls, name);
        if (found && apy_is_data_descriptor(found))
            return apy_descr_get(found, obj, O(obj)->v.o.cls);
        at = apy_dict_find(O(obj)->v.o.dict, name);
        if (at >= 0) return O(O(obj)->v.o.dict)->v.d.vals[at];
        if (found) {
            /* A NON-DATA descriptor -- `staticmethod`, `classmethod`, or a
               user class with only `__get__` -- is asked here, after the
               instance dict has missed. */
            if (apy_is_descriptor(found))
                return apy_descr_get(found, obj, O(obj)->v.o.cls);
            /* A function found on the CLASS becomes a bound method; anything
               else -- an int, a str, a list -- is handed back as it is. That
               single test is the whole of the "methods take self" rule. */
            return O(found)->kind == APY_FUNC_K ? apy_bind(found, obj) : found;
        }
        if (strcmp(want, "__class__") == 0) return O(obj)->v.o.cls;
        /* THE INSTANCE'S OWN attributes, and the real dict rather than a copy:
           `obj.__dict__["x"] = 1` is how a program sets an attribute
           dynamically, and a copy would accept the write and lose it.

           ABSENT under `__slots__`, which is the point of declaring it --
           `hasattr(p, "__dict__")` is how a program checks. */
        if (strcmp(want, "__dict__") == 0) {
            if (!apy_slot_allows(O(obj)->v.o.cls, apy_lit("__dict__")))
                return apy_no_attribute(obj, name);
            return O(obj)->v.o.dict;
        }
        /* A CLASS THAT EXTENDS A BUILTIN answers with the builtin's own
           method for everything its body did not define. `class D(dict)` with
           only a `__missing__` still has `keys`, `items`, `get` and `update`,
           and this is where they come from -- asked of `held`, the real dict
           the instance carries.

           HERE AND NOT EARLIER, because the class body wins: a `Counter`
           defining `update` must shadow `dict.update` rather than be shadowed
           by it. And here and not LATER, because in CPython these arrive
           through the MRO, which is consulted before `__getattr__` -- a class
           extending a builtin AND defining `__getattr__` would otherwise
           route every inherited method through the fallback.

           THE MISS IS NOT THE ANSWER. A name neither the class nor the
           builtin has must still reach `__getattr__`, so the AttributeError
           the delegation raised is cleared rather than reported. */
        if (O(obj)->v.o.held) {
            apy_value got = apy_default_getattr(O(obj)->v.o.held, name);
            if (got) return got;
            if (apy_err_type && strcmp(apy_err_type, "AttributeError") == 0)
                apy_error_clear();
            else
                return got;
        }
        /* `__getattr__` -- the LAST resort, asked only after the instance
           dict and the class have both missed. That ordering is the whole
           protocol: `__getattribute__` intercepts everything and this
           intercepts nothing that already resolved, which is why a class can
           define it without shadowing its own attributes. */
        {
            apy_value hook = apy_class_find(O(obj)->v.o.cls,
                                            apy_name("__getattr__"));
            if (hook) return apy_call_n(apy_bind(hook, obj), &name, 1);
        }
        return apy_no_attribute(obj, name);
    }
    case APY_TYPE_K: {
        apy_value found;
        /* THE BARE NAME: a builtin kind is stored under the dotted spelling
           CPython puts in a message and in `<class '...'>`, and `__name__` is
           the last component of it. See `apy_bare_name`. */
        if (strcmp(want, "__name__") == 0)
            return apy_bare_name(O(obj)->v.t.name);
        /* PEP 3155. A class nested in another would qualify differently; only
           the top-level spelling is recorded, which is the same limit the
           frontend's own keys have for classes. */
        if (strcmp(want, "__qualname__") == 0)
            return apy_bare_name(O(obj)->v.t.name);
        /* `C.__class__` IS THE METACLASS, which is `type` unless the class
           named one. Every other kind answers this and a class did not, so
           `C.__class__` was an AttributeError about a class that plainly has
           one -- and `isinstance(x, C.__class__)` is how a program asks. */
        if (strcmp(want, "__class__") == 0
                && !apy_dict_get_or(O(obj)->v.t.dict, apy_name("__class__"),
                                    0))
            return apy_type_of(obj);
        /* PEP 649 for a CLASS: `C.__annotations__` is built on access by the
           thunk the body left in the dict, for the same reason a function's
           is -- an annotation may name something that does not exist yet. */
        if (strcmp(want, "__annotations__") == 0) {
            /* WHAT WAS STORED WINS OVER THE THUNK. A program may set
               `C.__annotations__` directly, or hand a class body to
               `type(name, bases, {"__annotations__": ...})`, and both write an
               ordinary dict entry -- which this read ignored entirely, so the
               write appeared to succeed and the read still answered `{}`.
               That is how every library building a class dynamically declares
               its fields, and it is why `dataclasses.make_dataclass` could not
               work here. The thunk stays the fallback, because for a class
               written out in source PEP 649 is what defers the annotations. */
            apy_value stored = apy_dict_get_or(O(obj)->v.t.dict,
                                               apy_name("__annotations__"), 0);
            apy_value thunk;
            if (stored) return stored;
            thunk = apy_dict_get_or(O(obj)->v.t.dict,
                                    apy_name("__annotate__"), 0);
            if (!thunk) return apy_dict_new(1);
            return apy_call_n(thunk, NULL, 0);
        }
        /* `C.__dict__` is what the class body bound, not what it inherited --
           which is the difference `"x" in vars(C)` asks about. A copy, because
           a type's dict is a mapping proxy in CPython and is not writable. */
        if (strcmp(want, "__dict__") == 0) return apy_copy(O(obj)->v.t.dict);
        /* A SLOT NAME reached through the class is a descriptor, not a
           missing attribute: `__slots__` declares storage, and the class dict
           holds nothing for it. */
        if (apy_dict_find(O(obj)->v.t.dict, name) < 0
                && apy_slot_declares(obj, name))
            return apy_member_descriptor();
        /* THE HIERARCHY, as a program reads it back. `object` is the root
           of every chain even though no class links to it -- see
           `apy_object_class` -- so a class with no written base still has one
           base, and only `object` itself has none. Answering the empty tuple
           there said the chain stopped at the class, which is what
           `C.__bases__` is asked to disprove. */
        /* AN EXCEPTION TYPE'S PARENT IS IN THE NAME TABLE, not in a base
           pointer -- the builtin hierarchy is a table because `raise` and
           `except` match on the name and never hold a class. So
           `Exception.__bases__` answered `object` where CPython says
           `BaseException`: the class had no base pointer and the walk stopped
           at the root without ever asking the table. */
        if ((strcmp(want, "__bases__") == 0 || strcmp(want, "__base__") == 0
             || strcmp(want, "__mro__") == 0)
                && !O(obj)->v.t.base) {
            const char *parent = apy_exc_parent(APY_CSTR(O(obj)->v.t.name));
            if (parent) {
                if (strcmp(want, "__base__") == 0)
                    return apy_exc_type(apy_lit(parent));
                {
                    apy_value out = apy_tuple_new(4);
                    const char *walk = parent;
                    /* `__mro__` STARTS WITH THE CLASS ITSELF; `__bases__` and
                       `__base__` start with its parent. */
                    if (strcmp(want, "__mro__") == 0)
                        apy_seq_push(out, obj);
                    while (walk) {
                        apy_seq_push(out, apy_exc_type(apy_lit(walk)));
                        if (strcmp(want, "__bases__") == 0) break;
                        walk = apy_exc_parent(walk);
                    }
                    if (strcmp(want, "__mro__") == 0)
                        apy_seq_push(out, apy_object_class());
                    return out;
                }
            }
        }
        /* `C.__mro__` -- the classes a lookup walks, in order, ending at
           `object`. Single inheritance makes it a chain rather than a
           linearisation. */
        if (strcmp(want, "__mro__") == 0) {
            apy_value out = apy_tuple_new(4);
            apy_value walk = obj, root = apy_object_class();
            int64_t i;
            /* THE RECORDED ORDER when the class has one -- that IS the answer,
               and rebuilding it from the base chain would give a different
               one for a class with several bases. */
            if (O(obj)->v.t.mro) {
                for (i = 0; i < O(O(obj)->v.t.mro)->v.q.n; i++)
                    apy_seq_push(out, O(O(obj)->v.t.mro)->v.q.items[i]);
                apy_seq_push(out, root);
                return out;
            }
            while (walk && O(walk)->kind == APY_TYPE_K && walk != root) {
                apy_seq_push(out, walk);
                walk = O(walk)->v.t.base;
            }
            apy_seq_push(out, root);
            return out;
        }
        if (strcmp(want, "__bases__") == 0) {
            apy_value root = apy_object_class();
            apy_value out = apy_tuple_new(2);
            /* ALL OF THEM, in the order written. `__base__` is the first;
               these are what the class statement actually said. */
            if (O(obj)->v.t.bases) return O(obj)->v.t.bases;
            if (obj != root)
                apy_seq_push(out, O(obj)->v.t.base ? O(obj)->v.t.base : root);
            return out;
        }
        if (strcmp(want, "__base__") == 0) {
            apy_value root = apy_object_class();
            if (obj == root) return apy_none();
            return O(obj)->v.t.base ? O(obj)->v.t.base : root;
        }
        {
            /* THROUGH THE CLASS, a descriptor is asked with no instance:
               `C.make()` binds the class and `C.plain` hands the plain
               function back. Without this the classmethod object itself came
               out and calling it failed. */
            apy_value d = apy_class_find(obj, name);
            if (d && apy_is_descriptor(d))
                return apy_descr_get(d, 0, obj);
        }
        /* DEFINING `__eq__` AND NOT `__hash__` SETS `__hash__` TO None, and a
           program reads it back: `C.__hash__ is None` is how it asks whether
           instances are hashable. The refusal in `apy_hash_raw` already
           follows this rule; without the attribute the two disagreed about
           the same class -- unhashable when hashed, no such attribute when
           asked. */
        if (strcmp(want, "__hash__") == 0
                && !apy_class_find(obj, apy_name("__hash__"))
                && apy_class_find(obj, apy_name("__eq__")))
            return apy_none();
        found = apy_class_find(obj, name);
        /* Reached through the CLASS, a method is not bound: `C.m` is a plain
           function and `C.m(x)` passes x as self. */
        if (found) return found;
        /* AN ATTRIBUTE OF THE CLASS'S OWN TYPE. `Quacks.register(Duck)` is a
           method the METACLASS defines, and the class is its receiver -- the
           same relationship an instance has to its class, one level up. Asked
           LAST, because a name the class itself binds wins over one its
           metaclass does. */
        if (O(obj)->v.t.meta) {
            apy_value m = apy_class_find(O(obj)->v.t.meta, name);
            if (m && apy_is_descriptor(m)) return apy_descr_get(m, obj, obj);
            if (m && O(m)->kind == APY_FUNC_K) return apy_bind(m, obj);
            if (m) return m;
        }
        /* A CELL `apy_type_for` MINTED ANSWERS WHAT ITS KIND CARRIES.
           `type(iter([]))` is not the canonical thunk a NAMED builtin gets
           -- no program writes `list_iterator` -- so `apy_type_for` mints a
           cell holding the kind's name and an empty dict, and every method
           that kind has was out of reach: `type(it).__next__` was an
           AttributeError about the one method an iterator is FOR.

           AN EMPTY DICT, NO BASE AND NO METACLASS is what identifies one. A
           class the program wrote is a TYPE cell too, and its body always
           leaves `__module__` behind -- so a class named `list` cannot fall
           in here and collect a list's methods by accident. */
        if (O(obj)->v.t.dict && !O(O(obj)->v.t.dict)->v.d.n
                && !O(obj)->v.t.base && !O(obj)->v.t.meta) {
            found = apy_type_kind_attr(
                obj, (apy_value)(uintptr_t)APY_CSTR(O(obj)->v.t.name), name);
            if (found) return found;
        }
        /* THE HIERARCHY, as a program reads it back. `object` is the root of
           every chain even though no class links to it -- see
           `apy_object_class`. */
        return apy_fail2("AttributeError", "type object '%s' has no "
                         "attribute '%s'", APY_CSTR(O(obj)->v.t.name), want);
    }
    case APY_SUPER_K: {
        /* Lookup starts at the BASE of the class the calling method was
           defined in -- not at the base of `type(self)`. With `B(A)` and
           `C(B)`, a `super().m()` inside B's `m` must find A's, and starting
           from `type(self)` would find B's own and loop forever. */
        apy_value from = O(obj)->v.sup.from;
        apy_value self = O(obj)->v.sup.self;
        apy_value found = 0;
        /* THE RECEIVER'S MRO, PAST THE DEFINING CLASS. This is what makes a
           diamond work: inside B's method, `super()` on a D instance must
           reach C and not A, and only the RECEIVER's order knows that C sits
           between them. With a single base the walk is the base chain again,
           which is what the fallback below still does. */
        {
            apy_value order = 0;
            apy_value host = self ? apy_type_of(self) : 0;
            if (host && O(host)->kind == APY_TYPE_K) order = O(host)->v.t.mro;
            if (order && apy_is_seq(order)) {
                int64_t i, at = -1;
                for (i = 0; i < O(order)->v.q.n; i++)
                    if (O(order)->v.q.items[i] == from) { at = i; break; }
                for (i = at + 1; at >= 0 && i < O(order)->v.q.n && !found;
                     i++) {
                    apy_value here = O(order)->v.q.items[i];
                    int64_t k;
                    if (O(here)->kind != APY_TYPE_K) continue;
                    k = apy_dict_find(O(here)->v.t.dict, name);
                    if (k >= 0) found = O(O(here)->v.t.dict)->v.d.vals[k];
                }
            }
        }
        if (!found) found = apy_class_find(O(from)->v.t.base, name);
        /* THE BASE CHAIN HAS RUN OUT OF PYTHON and the receiver is an
           exception, so what `super().__init__(msg)` means is
           `BaseException.__init__` -- which is not a function anywhere in
           that chain, because the hierarchy above a user exception class is a
           table of names rather than classes. AFTER the walk, not before: a
           subclass writing `super().__init__(...)` must reach its own base's
           `__init__` first, and intercepting early sent every one of them
           straight past it. Falling through instead would find `object`'s,
           which takes the message and does nothing with it. */
        if (!found && self && O(self)->kind == APY_EXC_K
                && strcmp(want, "__init__") == 0)
            return apy_bind(apy_native(APY_NAT_EXC_INIT, 2, "__init__"), self);
        /* THE SAME ARRANGEMENT FOR A BUILTIN BASE, and for the same reason:
           the chain above `class M(dict)` ends in a KIND rather than a class,
           so the walk finds nothing and `object.__init__` would accept the
           arguments and drop them. */
        if (!found && self && O(self)->kind == APY_INST_K
                && O(self)->v.o.held && strcmp(want, "__init__") == 0)
            return apy_bind(apy_native(APY_NAT_BUILTIN_INIT, 2, "__init__"),
                            self);
        /* AND `__new__`, for the same chain and a different reason -- see the
           native. `from` rather than `self` decides, because `__new__` is
           called with the class and there may be no instance yet. */
        if (!found && strcmp(want, "__new__") == 0 && from
                && O(from)->kind == APY_TYPE_K
                && apy_class_builtin_kind(from) != 0)
            return apy_native(APY_NAT_BUILTIN_NEW, 2, "__new__");
        if (!found) {
            /* THE BASE CHAIN HAS RUN OUT, which means `object` -- and every
               class has one. `super().__init__()` inside a class with no
               explicit base is ordinary Python and was an AttributeError
               here: the default bodies existed and no VALUE named them. */
            found = apy_object_default(
                (apy_value)(uintptr_t)want);
            if (!found)
                return apy_fail2("AttributeError", "'super' object has no "
                                 "attribute '%s'%s", want, "");
        }
        /* `__new__` IS AN IMPLICIT STATICMETHOD and is NEVER bound: it
           receives the class as an ordinary first argument, which the caller
           writes out -- `super().__new__(cls)`. Binding it put the receiver
           in front of that and every argument landed one place late. */
        if (strcmp(want, "__new__") == 0) return found;
        return O(found)->kind == APY_FUNC_K ? apy_bind(found, self) : found;
    }
    case APY_SLICE_K:
        /* `s.start`, `s.stop`, `s.step` -- what a `__getitem__` reads off the
           slice it was handed. None where the bound was omitted. */
        if (strcmp(want, "start") == 0) return O(obj)->v.sl.start;
        if (strcmp(want, "stop") == 0) return O(obj)->v.sl.stop;
        if (strcmp(want, "step") == 0) return O(obj)->v.sl.step;
        return apy_no_attribute(obj, name);
    case APY_PROP_K:
        /* `p.fget` and `p.fset` -- the functions a property was built from.
           A subclass overriding a property reaches the base one's getter
           through `fget`, which is the only way to extend rather than replace
           it. None where there is no such half, as CPython answers. */
        if (strcmp(want, "fget") == 0)
            return O(obj)->v.p.get ? O(obj)->v.p.get : apy_none();
        if (strcmp(want, "fset") == 0)
            return O(obj)->v.p.set ? O(obj)->v.p.set : apy_none();
        if (strcmp(want, "fdel") == 0)
            return O(obj)->v.p.del_ ? O(obj)->v.p.del_ : apy_none();
        /* `__func__` is what `classmethod` and `staticmethod` wrap. */
        if (strcmp(want, "__func__") == 0 && O(obj)->v.p.get)
            return O(obj)->v.p.get;
        /* THE PROTOCOL ITSELF, as values. `hasattr(p, "__get__")` is how a
           program asks whether something is a descriptor, and a property that
           answered False to it was reported as an ordinary attribute. */
        if (strcmp(want, "__get__") == 0)
            return apy_bind(apy_native(APY_NAT_DESCR_GET, 3, "__get__"), obj);
        if (strcmp(want, "__set__") == 0)
            return apy_bind(apy_native(APY_NAT_DESCR_SET, 3, "__set__"), obj);
        if (strcmp(want, "__delete__") == 0)
            return apy_bind(apy_native(APY_NAT_DESCR_DEL, 2, "__delete__"),
                            obj);
        return apy_no_attribute(obj, name);
    case APY_FUNC_K:
        /* WHAT A PROGRAM PUT THERE WINS over the built-in attributes below.
           A function's own dict is consulted first for the same reason an
           instance's is: `f.__name__ = 'other'` has to read back as it was
           written, not as the function was defined. */
        if (O(obj)->v.fn.dict) {
            int64_t at = apy_dict_find(O(obj)->v.fn.dict, name);
            if (at >= 0) return O(O(obj)->v.fn.dict)->v.d.vals[at];
        }
        if (strcmp(want, "__name__") == 0) return O(obj)->v.fn.name;
        if (strcmp(want, "__code__") == 0) return apy_func_code(obj);
        /* `f.__defaults__` is the POSITIONAL defaults as a tuple and
           `__kwdefaults__` the keyword-only ones as a dict -- and each is
           None rather than empty when there are none, which is how a program
           tells "no defaults" from "a default that is falsey".
           The two are stored as one trailing run, keyword-only last, so the
           split is the number of keyword-only parameters that have one. */
        if (strcmp(want, "__defaults__") == 0
                || strcmp(want, "__kwdefaults__") == 0) {
            int64_t nd = O(obj)->v.fn.ndefaults;
            /* THE RECORDED COUNT, not the number of keyword-only parameters:
               one of those may be required, and `def f(a, b=1, *args, c)` has
               one keyword-only parameter and one default that is not its. */
            int64_t kw = O(obj)->v.fn.nkwdefault;
            int64_t at_kw = kw < nd ? kw : nd;
            int64_t npos = nd - at_kw, i;
            int64_t declared = O(obj)->v.fn.arity
                - (O(obj)->v.fn.vararg ? 1 : 0)
                - (O(obj)->v.fn.kwarg ? 1 : 0);
            if (strcmp(want, "__defaults__") == 0) {
                apy_value out;
                if (npos <= 0) return apy_none();
                out = apy_tuple_new(npos + 1);
                for (i = 0; i < npos; i++)
                    apy_seq_push(out, O(obj)->v.fn.defaults[i]);
                return out;
            }
            if (at_kw <= 0 || !O(obj)->v.fn.pnames) return apy_none();
            {
                apy_value out = apy_dict_new(at_kw + 1);
                for (i = 0; i < at_kw; i++) {
                    apy_value pn = O(obj)->v.fn.pnames[declared - at_kw + i];
                    if (pn) apy_dict_set(out, pn,
                                         O(obj)->v.fn.defaults[npos + i]);
                }
                return out;
            }
        }
        /* PEP 3155. The frontend's own key for a function is already the
           qualified name -- `C.m`, `outer.<locals>.inner` -- so it is handed
           over at construction. A function built without one (a synthesised
           thunk) answers its plain name, which is what it is. */
        if (strcmp(want, "__qualname__") == 0) {
            /* A BOUND BUILTIN METHOD IS QUALIFIED BY ITS RECEIVER'S KIND.
               `[].append.__qualname__` is `list.append` in CPython, the same
               text the unbound descriptor answers -- binding does not change
               where a method was defined. Derived rather than recorded,
               because the receiver already says it. */
            if (O(obj)->v.fn.native && O(obj)->v.fn.bound
                    && !O(obj)->v.fn.qualname) {
                char qual[128];
                snprintf(qual, sizeof qual, "%s.%s",
                         apy_kind_name(O(obj)->v.fn.bound),
                         APY_CSTR(O(obj)->v.fn.name));
                return apy_str_copy(qual, (int64_t)strlen(qual));
            }
            return O(obj)->v.fn.qualname ? O(obj)->v.fn.qualname
                                         : O(obj)->v.fn.name;
        }
        /* `f.__module__` -- WHERE THE `def` WAS WRITTEN. Six shapes reach
           here and CPython answers each of them differently:

             `len.__module__`          `builtins`      a builtin thunk
             `int.__module__`          `builtins`      a builtin type name
             `[].append.__module__`    None            a BOUND native
             `list.append.__module__`  AttributeError  a method_descriptor
             `f.__module__`            `__main__`      a program's own `def`
             `Fraction.m.__module__`   `fractions`     a spliced one

           The last two are the reason the field exists; the rest are read
           off the flags. A missing field is `__main__` rather than absent,
           because every `def` a program writes is in a module and there is
           only one it can be in. */
        if (strcmp(want, "__module__") == 0) {
            if (O(obj)->v.fn.native)
                return O(obj)->v.fn.bound ? apy_none()
                                          : apy_no_attribute(obj, name);
            if (O(obj)->v.fn.builtin || O(obj)->v.fn.is_type)
                return apy_lit("builtins");
            return O(obj)->v.fn.module ? O(obj)->v.fn.module
                                       : apy_lit("__main__");
        }
        /* `list.append.__objclass__` is `list` -- the type a DESCRIPTOR came
           off, which is how `inspect` and every `functools` wrapper finds
           out what a method belongs to. Absent on anything else, as it is in
           CPython. The owner is the head of the qualname; see
           `apy_descr_owner`. */
        if (strcmp(want, "__objclass__") == 0) {
            apy_value proto = O(obj)->v.fn.descr ? apy_descr_owner(obj) : 0;
            if (!proto) return apy_no_attribute(obj, name);
            return apy_kind_class(proto);
        }
        /* `m.__self__` is the RECEIVER of a bound method, and its absence is
           how a program tells a bound method from a plain function. */
        if (strcmp(want, "__self__") == 0) {
            if (!O(obj)->v.fn.bound) return apy_no_attribute(obj, name);
            return O(obj)->v.fn.bound;
        }
        /* `m.__func__` is the UNDERLYING function of a bound method -- the
           one the class holds, without the receiver. `c.m.__func__ is C.m` is
           the identity that says so, which is why this hands back a value
           sharing the code rather than a fresh binding. */
        if (strcmp(want, "__func__") == 0) {
            apy_value receiver = O(obj)->v.fn.bound;
            apy_value found;
            if (!receiver) return apy_no_attribute(obj, name);
            /* THE OBJECT THE CLASS HOLDS, looked up by the method's own name
               -- not a copy of it. `c.m.__func__ is C.m` is the identity that
               says what `__func__` means, and a fresh binding would answer
               False to it. */
            if (O(receiver)->kind == APY_INST_K) {
                found = apy_class_find(O(receiver)->v.o.cls,
                                       O(obj)->v.fn.name);
                if (found) return found;
            }
            return apy_no_attribute(obj, name);
        }
        /* A DOCSTRING is not recorded -- the frontend drops it as the
           statement it is -- so this is None rather than absent: every
           function has `__doc__`, and one without a docstring has None. */
        if (strcmp(want, "__doc__") == 0) {
            /* A TYPE OBJECT'S IS ITS KIND'S. `str.__doc__` and `"".__doc__`
               are the same text in Python, and this is the half that reaches
               it through the name the type carries. */
            if (O(obj)->v.fn.doc) return O(obj)->v.fn.doc;
            if (O(obj)->v.fn.is_type) {
                const char *doc = apy_kind_doc(APY_CSTR(O(obj)->v.fn.name));
                if (doc) return apy_lit(doc);
            }
            return apy_none();
        }
        /* PEP 649: `__annotations__` is BUILT ON ACCESS, by the thunk the
           `def` recorded. Evaluating them at the `def` would make
           `def f(x: Undefined)` an error where Python accepts it -- only
           reading the annotations is. A function with none answers the empty
           dict, which is what every function has. */
        if (strcmp(want, "__annotate__") == 0)
            return O(obj)->v.fn.annotate ? O(obj)->v.fn.annotate
                                         : apy_none();
        if (strcmp(want, "__annotations__") == 0) {
            if (!O(obj)->v.fn.annotate) return apy_dict_new(1);
            return apy_call_n(O(obj)->v.fn.annotate, NULL, 0);
        }
        return apy_no_attribute(obj, name);
    case APY_ALIAS_K:
        /* `list[int].__origin__` is `list` and `.__args__` is `(int,)`. A
           program that inspects an annotation reads exactly these two, and
           `get_origin`/`get_args` are defined in terms of them. */
        if (strcmp(want, "__origin__") == 0) return O(obj)->v.ga.origin;
        if (strcmp(want, "__args__") == 0) return O(obj)->v.ga.args;
        /* NOT A BARE REFUSAL. `__class__` is answered for every kind that has
           no attributes of its own, and adding a case here cut this kind off
           from that -- which is the "a new case must be taught every generic
           path" trap, arrived at from the other direction. */
        if (strcmp(want, "__class__") == 0) return apy_kind_class(obj);
        return apy_no_attribute(obj, name);
    case APY_GEN_K:
        /* WHICH `def` IT CAME FROM. `g.__name__` is the plain name and
           `g.__qualname__` the dotted one -- `C.m` for a method's generator,
           `f.<locals>.<genexpr>` for an expression's -- which is also what
           the repr prints. Recorded on the STEP function, because the
           generator cell is a frame and the frame's code is the step. */
        if (strcmp(want, "__name__") == 0
                || strcmp(want, "__qualname__") == 0) {
            apy_value step = O(obj)->v.g.step;
            if (!step || O(step)->kind != APY_FUNC_K)
                return apy_no_attribute(obj, name);
            if (strcmp(want, "__qualname__") == 0 && O(step)->v.fn.qualname)
                return O(step)->v.fn.qualname;
            return O(step)->v.fn.name;
        }
        /* WHERE THE BODY IS, which is the whole of a generator's
           introspection and was missing entirely -- so `dir(g)` could not
           honestly list a single one of these and did not list anything.

           `gi_running` is True only WHILE the frame is executing, which a
           program outside it can only see through a callback the body made;
           `gi_suspended` is True between the first `next` and the last, and
           False both before it has started and after it has finished. The
           state is 0, k and -1 for those three, and the two questions are
           different halves of it.

           A PLAIN GENERATOR ONLY, which is why every one of the five is
           gated. A coroutine and an async generator are this same cell --
           see `apy_coro_mark` -- and CPython gives them the same facts under
           `cr_` and `ag_`, which are not these names: `c().gi_code` is an
           AttributeError where `c().cr_code` is the code. `!coro` is what
           `apy_inspect_isgenerator` already reads, because an async
           generator carries both flags. */
        if (strcmp(want, "gi_running") == 0 && !O(obj)->v.g.coro)
            return apy_from_bool(O(obj)->v.g.running);
        if (strcmp(want, "gi_suspended") == 0 && !O(obj)->v.g.coro)
            return apy_from_bool(O(obj)->v.g.state > 0);
        /* WHAT A `yield from` IS DELEGATING TO, or None. Recorded by the
           lowered loop, because the delegate otherwise lives in a frame slot
           nothing outside the generator can name. */
        if (strcmp(want, "gi_yieldfrom") == 0 && !O(obj)->v.g.coro)
            return O(obj)->v.g.yieldfrom ? O(obj)->v.g.yieldfrom
                                         : apy_none();
        /* THE CODE OF THE `def` IT CAME FROM. The step function IS that code
           here -- the generator cell is the frame and the step is what the
           frame runs -- so this is `step.__code__` under the name a
           generator gives it. */
        if (strcmp(want, "gi_code") == 0 && !O(obj)->v.g.coro) {
            apy_value from = O(obj)->v.g.sig ? O(obj)->v.g.sig
                                             : O(obj)->v.g.step;
            if (!from || O(from)->kind != APY_FUNC_K)
                return apy_no_attribute(obj, name);
            return apy_func_code(from);
        }
        /* AND THE FRAME ITSELF, which is None once the body has finished:
           CPython drops the frame at that point, and `g.gi_frame is None` is
           how a program asks whether a generator is spent. */
        if (strcmp(want, "gi_frame") == 0 && !O(obj)->v.g.coro) {
            if (O(obj)->v.g.state < 0) return apy_none();
            return apy_gen_frame(obj);
        }
        /* AND THE SAME FIVE UNDER TWO OTHER PREFIXES. A coroutine and an
           async generator are this same cell -- see `apy_coro_mark` -- and
           CPython gives each of the three its own spelling: `gi_` for a
           generator, `cr_` for a coroutine, `ag_` for an async generator,
           and `dir()` over each lists only its own.

           `cr_await` AND `ag_await` ARE THE DELEGATE, which is the same
           field a `yield from` records: what the frame is waiting on is what
           the frame is waiting on, whichever keyword put it there. See
           `_dyn_await`, which writes it.

           `cr_origin` IS ALWAYS None. It is filled by coroutine origin
           tracking, which CPython turns on with `sys.set_coroutine_origin_
           tracking_depth` and nothing here turns on -- so None is the
           answer, and the same one CPython gives by default. */
        if (O(obj)->v.g.coro) {
            const char *pre = O(obj)->v.g.agen ? "ag_" : "cr_";
            if (strncmp(want, pre, 3) == 0) {
                const char *rest = want + 3;
                if (strcmp(rest, "running") == 0)
                    return apy_from_bool(O(obj)->v.g.running);
                if (strcmp(rest, "suspended") == 0)
                    return apy_from_bool(O(obj)->v.g.state > 0);
                if (strcmp(rest, "await") == 0)
                    return O(obj)->v.g.yieldfrom ? O(obj)->v.g.yieldfrom
                                                 : apy_none();
                if (strcmp(rest, "origin") == 0 && !O(obj)->v.g.agen)
                    return apy_none();
                if (strcmp(rest, "code") == 0) {
                    apy_value from = O(obj)->v.g.sig ? O(obj)->v.g.sig
                                                     : O(obj)->v.g.step;
                    if (!from || O(from)->kind != APY_FUNC_K)
                        return apy_no_attribute(obj, name);
                    return apy_func_code(from);
                }
                if (strcmp(rest, "frame") == 0) {
                    if (O(obj)->v.g.state < 0) return apy_none();
                    return apy_gen_frame(obj);
                }
            }
        }
        /* THE THREE METHODS, AS VALUES. They are dispatched by name at the
           call site, so nothing needed a value for them -- until a program
           asked `hasattr(g, "close")`, which every duck-typed consumer does,
           and got False for a method it can plainly call.

           AN ASYNC GENERATOR HAS THE OTHER THREE. CPython gives `asend`,
           `athrow` and `aclose` to one and `send`, `throw` and `close` to a
           coroutine and a plain generator, and neither has the other's --
           `hasattr(a, "send")` is False. Each of these answers an AWAITABLE;
           see `apy_agen_await`. */
        if (O(obj)->v.g.agen) {
            if (strcmp(want, "asend") == 0)
                return apy_bind(apy_native(APY_NAT_AGEN_SEND, 2, "asend"),
                                obj);
            if (strcmp(want, "athrow") == 0)
                return apy_bind(apy_native(APY_NAT_AGEN_THROW, 2, "athrow"),
                                obj);
            if (strcmp(want, "aclose") == 0)
                return apy_bind(apy_native(APY_NAT_AGEN_CLOSE, 1, "aclose"),
                                obj);
            return apy_no_attribute(obj, name);
        }
        if (strcmp(want, "send") == 0)
            return apy_bind(apy_native(APY_NAT_GEN_SEND, 2, "send"), obj);
        if (strcmp(want, "throw") == 0)
            return apy_bind(apy_native(APY_NAT_GEN_THROW, 2, "throw"), obj);
        /* A TASK'S OWN METHODS. A task is a generator cell like any
           other here, so these sit beside `send` and `throw` rather than on a
           class of their own. */
        if (O(obj)->v.g.builtin == APY_CORO_TASK) {
            if (strcmp(want, "cancel") == 0)
                return apy_bind(apy_native(APY_NAT_TASK_CANCEL, 1, "cancel"),
                                obj);
            if (strcmp(want, "result") == 0)
                return apy_bind(apy_native(APY_NAT_TASK_RESULT, 1, "result"),
                                obj);
            if (strcmp(want, "done") == 0)
                return apy_bind(apy_native(APY_NAT_TASK_DONE, 1, "done"),
                                obj);
            if (strcmp(want, "cancelled") == 0)
                return apy_bind(apy_native(APY_NAT_TASK_CANCELLED, 1,
                                           "cancelled"), obj);
        }
        if (strcmp(want, "close") == 0)
            return apy_bind(apy_native(APY_NAT_GEN_CLOSE, 1, "close"), obj);
        return apy_no_attribute(obj, name);
    case APY_MVIEW_K:
        /* A RELEASED VIEW STILL ANSWERS ITS METHOD NAMES -- CPython raises
           when one is CALLED, not when it is looked up -- so the check
           guards the FIELDS alone, which really are read here. */
        if (apy_mview_field(want) && !apy_mview_live(obj)) return 0;
        /* THE FLAG IS THE VIEW'S, and falls back to the buffer's: a view
           handed out by `toreadonly` refuses writes over a source that
           accepts them. */
        if (strcmp(want, "readonly") == 0)
            return apy_from_bool(O(obj)->v.mv.ro
                                 || !O(O(obj)->v.mv.src)->v.s.mut);
        if (strcmp(want, "nbytes") == 0) return apy_from_int(O(obj)->v.mv.n);
        /* One byte per element, unsigned -- the only format a bytes-like
           source produces, and the only one this constructs. A view over an
           `array('i')` would report differently, and there is no such source
           here to report for. */
        if (strcmp(want, "itemsize") == 0)
            return apy_from_int(O(obj)->v.mv.wide ? O(obj)->v.mv.wide : 1);
        if (strcmp(want, "format") == 0) {
            char one[2];
            one[0] = O(obj)->v.mv.fmt ? (char)O(obj)->v.mv.fmt : 'B';
            one[1] = 0;
            return apy_str_copy(one, 1);
        }
        if (strcmp(want, "obj") == 0) return O(obj)->v.mv.src;
        /* THE BUFFER'S SHAPE, which is one dimension here because every
           source this runtime can wrap is flat. A program that asks is
           usually checking exactly that -- `m.ndim == 1` before it indexes
           -- and got an AttributeError about the answer it wanted. */
        if (strcmp(want, "ndim") == 0) return apy_from_int(1);
        if (strcmp(want, "shape") == 0 || strcmp(want, "strides") == 0) {
            apy_value out = apy_tuple_new(2);
            if (!out) return 0;
            /* IN ELEMENTS, NOT BYTES, which is what `cast` changes:
               eight bytes read as `i` are two elements four apart. */
            int64_t wide = O(obj)->v.mv.wide ? O(obj)->v.mv.wide : 1;
            if (!apy_seq_push(out, apy_from_int(
                    strcmp(want, "shape") == 0 ? O(obj)->v.mv.n / wide
                                               : O(obj)->v.mv.step * wide)))
                return 0;
            return out;
        }
        /* EMPTY FOR A FLAT BUFFER, and CPython answers the empty tuple
           rather than None for one. */
        if (strcmp(want, "suboffsets") == 0) return apy_tuple_new(1);
        /* A SLICE WITH A STEP IS NOT CONTIGUOUS: `m[::2]` skips bytes, so
           the three flags are one question asked three ways. Both orders
           agree for a single dimension, which is why C and Fortran answer
           alike here. */
        if (strcmp(want, "c_contiguous") == 0
                || strcmp(want, "f_contiguous") == 0
                || strcmp(want, "contiguous") == 0)
            return apy_from_bool(O(obj)->v.mv.step == 1);
        return apy_no_attribute(obj, name);
    case APY_COMPLEX_K:
        if (strcmp(want, "real") == 0) return apy_from_float(O(obj)->v.z.re);
        if (strcmp(want, "imag") == 0) return apy_from_float(O(obj)->v.z.im);
        return apy_no_attribute(obj, name);
    case APY_INT_K:
    case APY_BOOL_K:
    case APY_BIG_K:
        /* EVERY NUMBER HAS `real` AND `imag`, not only a complex one -- that
           is what makes the numeric tower uniform, and a program written
           against it reads `.real` off whatever it was handed. An int's
           imaginary part is the INT zero, not the float, which `type()` on it
           can tell apart. */
        /* A BOOL ANSWERS WITH THE INT, on every one of the four: `True.real`
           is `1` and not `True`, which `type()` on it can tell apart and a
           program printing it plainly can. Handing back the receiver was
           right for an int and wrong for the kind that is one. */
        if (strcmp(want, "real") == 0)
            return O(obj)->kind == APY_BOOL_K
                   ? apy_from_int(O(obj)->v.i) : obj;
        if (strcmp(want, "imag") == 0) return apy_from_int(0);
        /* AND A RATIONAL'S TWO HALVES, which an int carries because it IS
           one in lowest terms. `Fraction(x)` reads them off whatever it was
           handed, and `numbers.Rational` is the rung of the tower that names
           them -- so a program written against the tower asks an int for
           them as readily as it asks a Fraction. */
        if (strcmp(want, "numerator") == 0)
            return O(obj)->kind == APY_BOOL_K
                   ? apy_from_int(O(obj)->v.i) : obj;
        if (strcmp(want, "denominator") == 0) return apy_from_int(1);
        return apy_no_attribute(obj, name);
    case APY_FLOAT_K:
        if (strcmp(want, "real") == 0) return obj;
        if (strcmp(want, "imag") == 0) return apy_from_float(0.0);
        return apy_no_attribute(obj, name);
    case APY_EXC_K:
        /* WHAT THE PROGRAM STORED WINS over what the KIND offers. `value`,
           `message` and `exceptions` are answered for every exception here --
           one cell serves them all -- but in CPython they belong to
           StopIteration and ExceptionGroup alone, so a class of its own
           setting `self.value` owns that name and nothing should take it.
           It did: `raise _Returned(42)` then `caught.value` gave back the
           MESSAGE rather than the 42, which is a wrong answer and not a
           missing feature.
           THE DUNDERS AND `args` ARE NOT IN THIS, because those really are
           BaseException's and a program writing one means to write through
           it. */
        if (O(obj)->v.e.dict && want[0] != '_'
                && strcmp(want, "args") != 0) {
            int64_t at = apy_dict_find(O(obj)->v.e.dict, name);
            if (at >= 0) return O(O(obj)->v.e.dict)->v.d.vals[at];
        }
        /* `g.exceptions` -- what an `ExceptionGroup` carries. Absent on an
           ordinary exception, which is how a program tells the two apart
           without asking about the type. */
        if (strcmp(want, "exceptions") == 0) {
            if (!O(obj)->v.e.subs) return apy_no_attribute(obj, name);
            return O(obj)->v.e.subs;
        }
        /* `g.message` -- the text an ExceptionGroup was built with, which is
           its FIRST argument and separate from the exceptions it carries.
           Present only on a group, like `exceptions`. */
        if (strcmp(want, "message") == 0) {
            if (!O(obj)->v.e.subs) return apy_no_attribute(obj, name);
            return O(obj)->v.e.has_arg ? O(obj)->v.e.arg : apy_lit("");
        }
        /* `e.args` is the one attribute the suite reads off an exception. */
        if (strcmp(want, "args") == 0) {
            apy_value out;
            if (O(obj)->v.e.argv) return O(obj)->v.e.argv;
            out = apy_tuple_new(1);
            /* The FLAG, not "is the argument None": `E(None).args` is
               `(None,)` and `E().args` is `()`. */
            if (O(obj)->v.e.has_arg) apy_seq_push(out, O(obj)->v.e.arg);
            return out;
        }
        /* CHAINING. Both are None when unset, never absent: `e.__cause__` is
           an attribute every exception has, and code that reads it to decide
           whether to print "The above exception was the direct cause" would
           get an AttributeError instead of the answer. */
        /* `e.value` -- what a generator's `return` gave. Every exception has
           it in CPython (it is None unless something set it), so answering
           None rather than reporting keeps `except StopIteration as e:
           e.value` working for the bare form too. */
        /* PEP 3151: `OSError(2, "No such file")` READS BOTH BACK. They
           are positions in `args`, not fields, which is why they are answered
           from it rather than stored twice. */
        if (strcmp(want, "errno") == 0 || strcmp(want, "strerror") == 0) {
            int64_t at = strcmp(want, "errno") == 0 ? 0 : 1;
            apy_value all = O(obj)->v.e.argv;
            if (all && apy_is_seq(all) && O(all)->v.q.n > at)
                return O(all)->v.q.items[at];
            return apy_none();
        }
        if (strcmp(want, "value") == 0)
            return O(obj)->v.e.has_arg ? O(obj)->v.e.arg : apy_none();
        if (strcmp(want, "__context__") == 0)
            return O(obj)->v.e.context ? O(obj)->v.e.context : apy_none();
        if (strcmp(want, "__cause__") == 0)
            return O(obj)->v.e.cause ? O(obj)->v.e.cause : apy_none();
        if (strcmp(want, "__suppress_context__") == 0)
            return apy_from_bool(O(obj)->v.e.suppress);
        if (strcmp(want, "__notes__") == 0) {
            if (!O(obj)->v.e.notes)
                return apy_no_attribute(obj, name);
            return O(obj)->v.e.notes;
        }
        /* There is no traceback OBJECT -- no frames are recorded -- and None
           is what an exception that was never raised carries. A program that
           tests `e.__traceback__ is not None` sees the difference; one that
           merely reads the attribute does not. */
        /* There is no traceback OBJECT -- no frames are recorded -- but an
           exception that was raised HAS one, and `e.__traceback__ is not
           None` is the test programs actually write. An empty tuple is the
           least dishonest stand-in: it is not None, it is not a frame, and
           iterating it yields the nothing this runtime knows. */
        if (strcmp(want, "__traceback__") == 0) {
            /* A REAL TRACEBACK where the position table exists, and the old
               empty-tuple stand-in where it does not -- a program that never
               asks about positions gets none recorded, and `e.__traceback__
               is not None` still has to answer True. */
            if (O(obj)->v.e.pos >= 0) return apy_traceback_of(obj);
            /* NEVER RAISED, so there is nothing to point at -- which is what
               CPython answers, and how a program tells a caught exception
               from one it merely built. */
            if (apy_pos_n > 0) return apy_none();
            /* No positions recorded at all: the old stand-in, which is not
               None because an exception that WAS raised has a traceback and
               nothing here can tell the two apart without them. */
            return apy_tuple_new(1);
        }
        if (strcmp(want, "__class__") == 0) return apy_type_of(obj);
        /* THIS EXCEPTION'S OWN ATTRIBUTES, then its class's -- the ordinary
           two-step, arriving late because the fixed names above are what
           BaseException itself defines and a class body cannot shadow. */
        if (O(obj)->v.e.dict) {
            int64_t at = apy_dict_find(O(obj)->v.e.dict, name);
            if (at >= 0) return O(O(obj)->v.e.dict)->v.d.vals[at];
        }
        if (O(obj)->v.e.cls) {
            apy_value found = apy_class_find(O(obj)->v.e.cls, name);
            if (found)
                return O(found)->kind == APY_FUNC_K ? apy_bind(found, obj)
                                                    : found;
        }
        return apy_no_attribute(obj, name);
    default:
        /* `__class__` on a kind with no attributes of its own -- a generic
           alias, a slice, a view. INTERNED per name, because the object a
           program compares or reads `__name__` off has to be the same one
           each time it asks. */
        if (strcmp(want, "__class__") == 0) return apy_kind_class(obj);
        return apy_no_attribute(obj, name);
    }
}

/* `ord`, `chr`, `ascii`, `callable`, `hasattr`, `all`, `any` -- the small
   builtins, together because each is a few lines and separating them would be
   a section header per function.

   `ascii` differs from `repr` only for non-ASCII text, which this runtime does
   not represent yet, so it IS repr here. Saying so beats a second
   implementation that is the same code with a different name. */
/* `id(x)` -- the object's ADDRESS, which is what identity already means
   here: `is` compares these words, so the number a program prints and the
   comparison it writes agree by construction. */
APY_API apy_value apy_id(apy_value v) {
    return apy_from_int((int64_t)(uintptr_t)v);
}

APY_API apy_value apy_ord(apy_value v) {
    /* A str SUBCLASS IS A str HERE TOO -- see `apy_text_like`. */
    v = apy_text_like(v);
    if (O(v)->kind == APY_BYTES_K) {
        if (O(v)->v.s.n != 1)
            return apy_fail("TypeError",
                            "ord() expected a character, but string of "
                            "length != 1 found");
        return apy_from_int((int64_t)(unsigned char)O(v)->v.s.p[0]);
    }
    if (O(v)->kind != APY_STR_K)
        return apy_fail2("TypeError",
                         "ord() expected string of length 1, but %s found%s",
                         apy_kind_name(v), "");
    /* One CHARACTER, not one byte, and the two stopped coinciding when `chr`
       learned to build a multi-byte one: a str is stored as UTF-8, so the
       length test has to count characters and the answer has to DECODE the
       sequence. Testing bytes made `ord(chr(233))` a TypeError about a string
       of length != 1 -- describing a string the program never wrote. */
    {
        const unsigned char *p = (const unsigned char *)O(v)->v.s.p;
        int64_t n = O(v)->v.s.n, want;
        int64_t code;
        if (n < 1) want = 0;
        else if (p[0] < 0x80) { want = 1; code = p[0]; }
        else if ((p[0] & 0xE0) == 0xC0) { want = 2; code = p[0] & 0x1F; }
        else if ((p[0] & 0xF0) == 0xE0) { want = 3; code = p[0] & 0x0F; }
        else if ((p[0] & 0xF8) == 0xF0) { want = 4; code = p[0] & 0x07; }
        else { want = 1; code = p[0]; }        /* a stray continuation byte */
        if (n != want || want == 0)
            return apy_fail("TypeError",
                            "ord() expected a character, but string of "
                            "length != 1 found");
        {
            int64_t i;
            for (i = 1; i < want; i++) code = (code << 6) | (p[i] & 0x3F);
        }
        return apy_from_int(code);
    }
}

APY_API apy_value apy_chr(apy_value v) {
    int64_t code;
    char buf[5];
    int64_t n = 0;
    if (!apy_is_int_like(v))
        return apy_fail2("TypeError",
                         "an integer is required (got type %s)%s",
                         apy_kind_name(v), "");
    code = O(v)->v.i;
    if (code < 0 || code > 0x10FFFF)
        return apy_fail("ValueError", "chr() arg not in range(0x110000)");
    /* UTF-8, because that is how a str is stored here -- so a code point
       becomes one to four bytes and `len` counts characters by decoding
       them again. Refusing anything above ASCII, which is what this did,
       made `chr(233)` an error on a runtime that handles the byte fine. */
    if (code < 0x80) {
        buf[n++] = (char)code;
    } else if (code < 0x800) {
        buf[n++] = (char)(0xC0 | (code >> 6));
        buf[n++] = (char)(0x80 | (code & 0x3F));
    } else if (code < 0x10000) {
        buf[n++] = (char)(0xE0 | (code >> 12));
        buf[n++] = (char)(0x80 | ((code >> 6) & 0x3F));
        buf[n++] = (char)(0x80 | (code & 0x3F));
    } else {
        buf[n++] = (char)(0xF0 | (code >> 18));
        buf[n++] = (char)(0x80 | ((code >> 12) & 0x3F));
        buf[n++] = (char)(0x80 | ((code >> 6) & 0x3F));
        buf[n++] = (char)(0x80 | (code & 0x3F));
    }
    buf[n] = 0;
    return apy_str_copy(buf, n);
}

/* One code point starting at `p[i]`, with `*len` set to its byte count.
   Shared by `maketrans` and `translate`, which both have to walk a str by
   CHARACTER: a translation table is keyed by code point, so stepping a byte
   at a time would key 'é' under its two halves and match neither. */
APY_API int64_t apy_utf8_at_of(apy_value bytes, int64_t n, int64_t i,
                               apy_value len_out) {
    const unsigned char *p = (const unsigned char *)(uintptr_t)bytes;
    int64_t *len = (int64_t *)(uintptr_t)len_out;
    int64_t want, code;
    if (p[i] < 0x80) { want = 1; code = p[i]; }
    else if ((p[i] & 0xE0) == 0xC0) { want = 2; code = p[i] & 0x1F; }
    else if ((p[i] & 0xF0) == 0xE0) { want = 3; code = p[i] & 0x0F; }
    else if ((p[i] & 0xF8) == 0xF0) { want = 4; code = p[i] & 0x07; }
    else { *len = 1; return p[i]; }            /* a stray continuation byte */
    if (i + want > n) { *len = 1; return p[i]; }
    {
        int64_t k;
        for (k = 1; k < want; k++) code = (code << 6) | (p[i + k] & 0x3F);
    }
    *len = want;
    return code;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now, and
   the exported half above stands in when nothing is ported. THE DELEGATE
   CONVERTS rather than forwards -- the subset has one integer width, so the
   `int64_t *` out-parameter crosses as a plain word and is cast back here. */
static int64_t apy_utf8_at(const unsigned char *p, int64_t n, int64_t i,
                           int64_t *len) {
    return apy_utf8_at_of((apy_value)(uintptr_t)p, n, i,
                          (apy_value)(uintptr_t)len);
}

/* `str.maketrans(a, b)` and `str.maketrans(a, b, drop)`. The result is an
   ORDINARY DICT keyed by code point -- that is not an implementation detail,
   it is the documented shape, and a program may build the same dict by hand
   and pass it to `translate`. A None third argument is the two-argument form;
   the frontend supplies None rather than the table carrying two arities. */
APY_API apy_value apy_str_maketrans(apy_value a, apy_value b, apy_value drop) {
    apy_value out;
    int64_t i = 0, j = 0, alen, blen;
    const unsigned char *ap, *bp;
    /* THE ONE-ARGUMENT FORM IS PICKED BY THE ARGUMENTS THAT ARE NOT THERE,
       and not by the kind of the one that is. CPython's
       `unicode_maketrans_impl` (Objects/unicodeobject.c) branches on
       `y == NULL` and only then asks whether `x` is a dict, so
       `str.maketrans("ab")` is the MAPPING form refusing a non-dict and not
       the pairing form complaining about a second argument it was never
       given. Branching on `x` instead answered "maketrans() arguments must
       be strings" for each of `"ab"`, `[]`, `None` and `7`, where CPython
       names the dict it wanted. The frontend always passes three slots --
       see `dynamic.py` -- so an omitted argument arrives here as None. */
    if ((!b || O(b)->kind == APY_NONE_K)
            && (!drop || O(drop)->kind == APY_NONE_K)) {
        int64_t at;
        /* A DIFFERENT OPERATION WEARING THE SAME NAME: nothing is paired
           off, a table that is already a table is COPIED with its string
           keys turned into the code points `translate` looks up. CPython
           gives the wrong shape its own sentence rather than the pairing
           error, which is why that message is not reused here. */
        if (O(a)->kind != APY_DICT_K)
            return apy_fail("TypeError",
                            "if you give only one argument to maketrans it "
                            "must be a dict");
        out = apy_dict_new(O(a)->v.d.n + 1);
        if (!out) return 0;
        for (at = 0; at < O(a)->v.d.n; at++) {
            apy_value key = O(a)->v.d.keys[at];
            apy_value point = key;
            /* A str SUBCLASS IS A STRING KEY. CPython's loop runs
               `PyUnicode_Check` -- the SUBCLASS-ADMITTING check, not
               `PyUnicode_CheckExact` -- and then `PyUnicode_READ_CHAR`,
               which reads the same C-level layout a subclass has, so
               `str.maketrans({S("a"): "z"})` is `{97: 'z'}` for
               `class S(str)`. Reading the kind of the instance instead sent
               it to the TypeError below. Measured against CPython 3.14;
               `scratchpad/probes/d179adv.py`.

               THE UNWRAP FEEDS THE BUFFER READ AND NOTHING ELSE: `point`
               becomes the ordinal, so nothing downstream sees the held
               string, and an instance of a class extending BYTES unwraps to
               bytes -- not a string key, not an integer one -- and still
               lands on the TypeError. */
            apy_value ktext = apy_text_like(key);
            if (O(ktext)->kind == APY_STR_K) {
                int64_t used, cp;
                /* A ONE-CHARACTER STRING KEY IS THE ONLY KIND THAT HAS AN
                   ORDINAL, which is what the ValueError says. */
                if (apy_str_chars(ktext) != 1)
                    return apy_fail("ValueError",
                                    "string keys in translate table must be "
                                    "of length 1");
                cp = apy_utf8_at((const unsigned char *)O(ktext)->v.s.p,
                                 O(ktext)->v.s.n, 0, &used);
                point = apy_from_int(cp);
            } else if (!apy_is_int_like(key)) {
                /* AND NOTHING ELSE IS A KEY AT ALL. CPython's loop runs
                   `PyUnicode_Check` then `PyLong_Check` and raises on
                   anything else, so `{1.0: "z"}` is a TypeError there; this
                   copied such a key straight through and built a table whose
                   entries `translate` could never match.

                   A BOOL COUNTS AS AN INTEGER, because `PyLong_Check` says
                   so: `str.maketrans({True: "z"})` builds in CPython. */
                return apy_fail("TypeError",
                                "keys in translate table must be strings or "
                                "integers");
            }
            /* AN INTEGER KEY IS KEPT AS THE OBJECT IT IS, not narrowed to a
               plain int, because `unicode_maketrans_impl` stores it straight
               into the new dict: `str.maketrans({True: "z"})` is `{True: 'z'}`
               there and not `{1: 'z'}`. It translates the same either way,
               since True hashes as 1, and the difference is visible only to a
               program that prints the table.

               AND THE RANGE IS NOT CHECKED. `{0x110000: "z"}` and `{-1: "z"}`
               both build here, exactly as they do in CPython: a key outside
               the code points simply never matches, and refusing it would
               refuse a table CPython makes. */
            if (!apy_dict_set(out, point, O(a)->v.d.vals[at]))
                return 0;
        }
        return out;
    }
    /* A str SUBCLASS IS A str HERE TOO -- `str.maketrans(S("a"), S("z"))`
       builds the same table CPython's does. */
    a = apy_text_like(a);
    b = apy_text_like(b);
    /* A NON-str FIRST ARGUMENT WITH A SECOND ONE is neither form, and
       CPython names the FIRST argument because that is the one that decides
       which form this was going to be. The check lived inside the branch
       above while that branch was the one a dict took; the form is chosen by
       the absent arguments now, so it belongs on this side. */
    if (O(a)->kind != APY_STR_K && b && O(b)->kind != APY_NONE_K)
        return apy_fail("TypeError",
                        "first maketrans argument must be a string if "
                        "there is a second argument");
    if (O(a)->kind != APY_STR_K || O(b)->kind != APY_STR_K)
        return apy_fail("TypeError",
                        "maketrans() arguments must be strings");
    if (apy_str_chars(a) != apy_str_chars(b))
        return apy_fail("ValueError",
                        "the first two maketrans arguments must have equal "
                        "length");
    out = apy_dict_new(8);
    ap = (const unsigned char *)O(a)->v.s.p;
    bp = (const unsigned char *)O(b)->v.s.p;
    while (i < O(a)->v.s.n && j < O(b)->v.s.n) {
        int64_t ka, kb, from = apy_utf8_at(ap, O(a)->v.s.n, i, &ka);
        int64_t to = apy_utf8_at(bp, O(b)->v.s.n, j, &kb);
        if (!apy_dict_set(out, apy_from_int(from), apy_from_int(to))) return 0;
        i += ka;
        j += kb;
    }
    /* The third argument names characters to DELETE, which the table records
       as a None value -- the same thing a hand-written table uses to mean
       "drop this one". */
    if (drop && O(drop)->kind == APY_STR_K) {
        const unsigned char *dp = (const unsigned char *)O(drop)->v.s.p;
        int64_t k = 0;
        while (k < O(drop)->v.s.n) {
            int64_t used, cp = apy_utf8_at(dp, O(drop)->v.s.n, k, &used);
            if (!apy_dict_set(out, apy_from_int(cp), apy_none())) return 0;
            k += used;
        }
    }
    return out;
}

/* Declared here and written at the foot of this file: a bytes receiver takes
   a different method under this name, and `str.translate` hands it over. */
APY_API apy_value apy_bytes_translate(apy_value s, apy_value table,
                                      apy_value delete);

/* `s.translate(table)`. A character with no entry is KEPT -- translate maps
   what it knows and passes the rest through, which is what makes a table
   holding one key a useful thing to write. */
APY_API apy_value apy_str_translate(apy_value s, apy_value table) {
    int64_t i = 0, cap, out_n = 0;
    char *buf;
    const unsigned char *p;
    if (!apy_str_self("translate", s)) return 0;
    /* A BYTES RECEIVER IS A DIFFERENT METHOD -- bytes through a 256-byte
       table, not code points through a dict. None is the delete set the
       one-argument form has. */
    /* AN EMPTY DELETE SET, NOT None. `b.translate(table)` deletes nothing
       and `b.translate(table, None)` is a TypeError -- the one-argument form
       reaches HERE and the two-argument one reaches `apy_bytes_translate`
       directly, which is the only thing that tells them apart. `b""` is the
       default CPython's own signature declares. */
    if (O(s)->kind == APY_BYTES_K)
        return apy_bytes_translate(s, table, apy_bytes_copy("", 0));
    if (O(table)->kind != APY_DICT_K)
        return apy_fail2("TypeError", "'%s' object is not subscriptable%s",
                         apy_kind_name(table), "");
    /* Four bytes per input byte is the worst a replacement can cost: every
       character could map to a four-byte one. */
    cap = O(s)->v.s.n * 4 + 1;
    buf = (char *)malloc((size_t)cap);
    p = (const unsigned char *)O(s)->v.s.p;
    while (i < O(s)->v.s.n) {
        int64_t used, cp = apy_utf8_at(p, O(s)->v.s.n, i, &used);
        int64_t at = apy_dict_find(table, apy_from_int(cp));
        if (at < 0) {
            memcpy(buf + out_n, p + i, (size_t)used);
            out_n += used;
        } else {
            apy_value to = O(table)->v.d.vals[at];
            if (O(to)->kind == APY_NONE_K) {
                /* deleted */
            } else if (apy_is_int_like(to)) {
                apy_value ch = apy_chr(to);
                if (!ch) { free(buf); return 0; }
                memcpy(buf + out_n, O(ch)->v.s.p, (size_t)O(ch)->v.s.n);
                out_n += O(ch)->v.s.n;
            } else if (O(to)->kind == APY_STR_K) {
                /* A table may map to a whole STRING, not just one character
                   -- `{ord('&'): 'and'}` is a normal thing to write. */
                if (out_n + O(to)->v.s.n >= cap) {
                    cap = (out_n + O(to)->v.s.n) * 2 + 1;
                    buf = (char *)realloc(buf, (size_t)cap);
                }
                memcpy(buf + out_n, O(to)->v.s.p, (size_t)O(to)->v.s.n);
                out_n += O(to)->v.s.n;
            } else {
                free(buf);
                return apy_fail("TypeError",
                                "character mapping must be in range(0x110000)");
            }
        }
        i += used;
    }
    buf[out_n] = '\0';
    return apy_str_take(buf, out_n);
}

/* ── the bytes family ──────────────────────────────────────────────────────

   `bytes.translate` IS A DIFFERENT METHOD WEARING THE SAME NAME. `str` maps
   by code point through a DICT and may replace one character with a whole
   string; `bytes` maps by BYTE through a 256-BYTE TABLE and can only shorten,
   by deleting. The signatures differ too -- `bytes.translate(table, /,
   delete=b'')` takes a second argument and `str.translate(table)` does not --
   and the receiver is not known until run time, so the split is made here.

   WHY THREE ENTRY POINTS AND NOT ONE. CPython gives three different messages
   for the three ways the call can be wrong, and a program may test any:

       "abc".translate(t, b"c")         str.translate() takes exactly one
                                        argument (2 given)
       "abc".translate(t, delete=b"c")  str.translate() takes no keyword
                                        arguments
       b"abc".translate(t, b"c")        the answer

   The first two are the SAME CALL once the arguments are lowered -- a keyword
   folded into its slot is indistinguishable from the positional it stands for
   -- so the spelling has to survive as far as the symbol, which is what
   `apy_translate_kw` is for. */

static apy_value apy_bytes_like_bad(apy_value v) {
    return apy_fail2("TypeError", "a bytes-like object is required, not '%s'%s",
                     apy_kind_name(v), "");
}

/* `v` as bytes if it is bytes-like, else `v` itself. A MEMORYVIEW IS
   BYTES-LIKE and every other bytes method already takes one, so the table and
   the delete set take one too; a view has an offset and a step, so its bytes
   are not where a plain read would look for them. */
static apy_value apy_bytes_like(apy_value v) {
    if (O(v)->kind == APY_MVIEW_K) return apy_mview_bytes(v);
    return v;
}

/* `bytes.maketrans(frm, to)` -- the 256-byte table `translate` wants. The
   table is a FULL mapping and not a sparse one: every byte the caller did not
   name maps to itself, which is why it must be exactly 256 long and why
   `translate` can refuse any other length. */
APY_API apy_value apy_bytes_maketrans(apy_value a, apy_value b) {
    int64_t i, n;
    char *buf;
    const unsigned char *ap, *bp;
    a = apy_bytes_like(a);
    b = apy_bytes_like(b);
    if (O(a)->kind != APY_BYTES_K) return apy_bytes_like_bad(a);
    if (O(b)->kind != APY_BYTES_K) return apy_bytes_like_bad(b);
    n = O(a)->v.s.n;
    if (n != O(b)->v.s.n)
        return apy_fail("ValueError",
                        "maketrans arguments must have same length");
    buf = (char *)malloc(257);
    for (i = 0; i < 256; i++) buf[i] = (char)(unsigned char)i;
    ap = (const unsigned char *)O(a)->v.s.p;
    bp = (const unsigned char *)O(b)->v.s.p;
    for (i = 0; i < n; i++) buf[ap[i]] = (char)bp[i];
    buf[256] = '\0';
    {
        return apy_bytes_take(buf, 256);
    }
}

/* `b.translate(table)` and `b.translate(table, delete)`.

   A NONE TABLE IS THE IDENTITY and not an error: `b.translate(None, b"a")` is
   the ordinary way to spell "delete these bytes and change nothing else", and
   it is the only reason the two-argument form is worth having over a table
   that deletes.

   THE DELETE SET IS CONSULTED BEFORE THE TABLE, which is CPython's order and
   is observable: a byte that is both mapped and deleted is deleted. */
APY_API apy_value apy_bytes_translate(apy_value s, apy_value table,
                                      apy_value delete) {
    unsigned char map256[256], drop[256];
    int64_t i, n, out_n = 0;
    /* WHETHER ANY BYTE ACTUALLY MOVED -- see the foot of this function. */
    int changed = 0;
    char *buf;
    const unsigned char *p;
    if (!apy_str_self("translate", s)) return 0;
    if (O(s)->kind == APY_STR_K)
        return apy_fail("TypeError",
                        "str.translate() takes exactly one argument (2 given)");
    for (i = 0; i < 256; i++) { map256[i] = (unsigned char)i; drop[i] = 0; }
    table = apy_bytes_like(table);
    delete = apy_bytes_like(delete);
    if (O(table)->kind != APY_NONE_K) {
        if (O(table)->kind != APY_BYTES_K) return apy_bytes_like_bad(table);
        if (O(table)->v.s.n != 256)
            return apy_fail("ValueError",
                            "translation table must be 256 characters long");
        memcpy(map256, O(table)->v.s.p, 256);
    }
    /* NO None HERE. A delete set that was WRITTEN must be bytes-like --
       `b"abc".translate(None, None)` is `a bytes-like object is required,
       not 'NoneType'` in CPython -- and the one-argument form passes an
       empty one rather than a None. */
    if (O(delete)->kind != APY_BYTES_K) return apy_bytes_like_bad(delete);
    p = (const unsigned char *)O(delete)->v.s.p;
    for (i = 0; i < O(delete)->v.s.n; i++) drop[p[i]] = 1;
    n = O(s)->v.s.n;
    p = (const unsigned char *)O(s)->v.s.p;
    buf = (char *)malloc((size_t)n + 1);
    for (i = 0; i < n; i++) {
        if (drop[p[i]]) { changed = 1; continue; }
        if (map256[p[i]] != p[i]) changed = 1;
        buf[out_n++] = (char)map256[p[i]];
    }
    buf[out_n] = '\0';
    /* NOTHING MOVED, so the answer IS the receiver: `bytes_translate` ends
       both of its loops with `if (!changed && PyBytes_CheckExact(input_obj))
       return input_obj`, which is what makes `b.translate(None) is b` True
       -- the ordinary spelling of "delete these and change nothing else"
       with nothing to delete. A table that maps every byte to itself and a
       delete set that matches nothing reach it the same way, so the test is
       what the walk DID and not what it was handed.

       AND str.translate HAS NO SUCH SHORTCUT, which is the other half of
       this row's asymmetry: `"abc".translate({}) is "abc"` is False in
       CPython, because `_PyUnicode_TranslateCharmap` builds its result
       through a writer and hands it back whatever it copied. That function
       is above; leave it building. */
    if (!changed && apy_str_may_return_self(s)) { free(buf); return s; }
    /* THE RESULT IS BUILT AS bytes HERE AND NOT LEFT TO `apy_str_like`, and
       the reason is IDENTITY rather than tidiness. This used to end in
       `apy_str_take`, which is the STR funnel: the call site's
       `apy_str_like` then saw a str under a bytes receiver and re-tagged it
       through `apy_bytes_copy`, and that funnel answers the 256-object
       cache for a one-byte result. So `b"abc".translate(None, b"bc") is
       b"a"` was True on both compiled paths.

       CPYTHON SAYS False, and the asymmetry is `_PyBytes_Resize`'s:
       `bytes_translate` sizes its output buffer to the input and then
       resizes, and that function special-cases a new size of ZERO -- handing
       back the interned empty -- while every other size is a plain
       `realloc` that never reaches the cache. Measured both ways round
       against CPython 3.14 (`scratchpad/probes/d172.py`):
       `b"abc".translate(None, b"abc") is b""` is True and
       `b"abc".translate(None, b"bc") is b"a"` is False. The interpreter was
       given exactly this rule in d60f3f3d; this is the same rule for the
       two compiled paths.

       AND THE STR SIDE IS NOT THE SAME, so do not tidy the two into one:
       `"abc".translate(str.maketrans("", "", "bc")) is "a"` IS True, because
       the unicode writer consults the latin-1 cache. `apy_str_translate`
       above keeps its `apy_str_take` for that reason.

       ONE-BYTE SHARING CANNOT BE SETTLED IN `apy_str_like` EITHER, which is
       why the fix is here. That funnel serves every bytes method at once,
       and CPython shares by method: `b" a ".strip()`, `b"a-b".partition
       (b"-")[0]`, `b"a b".split()[0]` and `b"abc"[0:1]` all ARE `b"a"`,
       while `translate`, `replace` and `upper` are not. Measured.

       NOT `apy_bytes_take`: that is the funnel that consults the cache.
       `apy_bytes_own` is the cell constructor underneath it, and the empty
       one is asked for by name -- a cell decided BEFORE it is made, never a
       shared one re-tagged afterwards. */
    if (out_n == 0) { free(buf); return apy_shared_bytes(256); }
    return apy_bytes_own(buf, out_n, 0);
}

/* `x.translate(table, delete=...)` -- the form written with the keyword.
   THE ONLY DIFFERENCE IS THE MESSAGE; see the block above. */
APY_API apy_value apy_translate_kw(apy_value s, apy_value table,
                                   apy_value delete) {
    if (O(s)->kind == APY_STR_K)
        return apy_fail("TypeError",
                        "str.translate() takes no keyword arguments");
    return apy_bytes_translate(s, table, delete);
}

APY_API apy_value apy_callable(apy_value v) {
    if (O(v)->kind == APY_FUNC_K || O(v)->kind == APY_TYPE_K)
        return apy_from_bool(1);
    if (O(v)->kind == APY_INST_K)
        return apy_from_bool(apy_class_find(O(v)->v.o.cls,
                                            apy_name("__call__")) != 0);
    return apy_from_bool(0);
}

APY_API apy_value apy_hasattr(apy_value v, apy_value name) {
    apy_value got;
    name = apy_text_like(name);
    got = apy_getattr(v, name);
    if (got) return apy_from_bool(1);
    /* `hasattr` ANSWERS rather than propagating: a missing attribute is False,
       not the AttributeError the lookup raised. */
    /* ONLY AN AttributeError, which is Python's rule since 3.2: before
       that `hasattr` caught everything, and the change was made because
       a property raising a ValueError read as "no such attribute" and
       hid a real failure. This caught everything too. */
    if (!apy_error_matches(apy_lit("AttributeError"))) return 0;
    apy_error_clear();
    return apy_from_bool(0);
}

/* `all(xs)` and `any(xs)` -- and they SHORT-CIRCUIT, which is the whole
   reason they step rather than index.

   `any(expensive(x) for x in xs)` must stop at the first true one: with a
   generator argument that is observable, because the generator simply is not
   resumed again. Walking by index forced the argument to have a length, which
   drained the generator before the first test and ran `expensive` on every
   element. */
APY_API apy_value apy_every_of(apy_value v, int64_t want,
                              int64_t otherwise) {
    apy_value it = apy_getiter(v);
    if (!it) return 0;
    for (;;) {
        apy_value item = apy_step(it);
        if (!item) return 0;
        if (item == apy_stop()) return apy_from_bool(otherwise);
        if (apy_truth(item) == want) return apy_from_bool(want);
    }
}

/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_every(apy_value v, int want, int otherwise) {
    return apy_every_of(v, (int64_t)want, (int64_t)otherwise);
}

APY_API apy_value apy_all(apy_value v) { return apy_every(v, 0, 1); }
APY_API apy_value apy_any(apy_value v) { return apy_every(v, 1, 0); }

/* `object.__repr__(x)` -- the default, which a `__repr__` override needs in
   order to show what it overrode. */
APY_API apy_value apy_default_repr(apy_value v) {
    char buf[96];
    if (O(v)->kind != APY_INST_K) return apy_repr(v);
    snprintf(buf, sizeof buf, "<%s object at 0x%llx>",
             APY_CSTR(O(O(v)->v.o.cls)->v.t.name), (unsigned long long)v);
    /* COPIED. `apy_lit` keeps the pointer it is given, which is right for a
       string literal and catastrophic for a stack buffer -- the str outlives
       this frame and would be reading whatever came next. */
    return apy_str_copy(buf, (int64_t)strlen(buf));
}

/* `object.__eq__(a, b)` is IDENTITY, and `object.__hash__(x)` agrees with it.
   That pairing is the contract: two objects that compare equal must hash
   equally, and the default satisfies it by comparing nothing but address. */
APY_API apy_value apy_default_eq(apy_value a, apy_value b) {
    return apy_from_bool(a == b);
}

/* `object.__init__(self)` -- the default, which does nothing. A subclass
   calling it is saying "the base has no state to set up", and answering None
   is exactly that. */
APY_API apy_value apy_default_init(apy_value v) {
    (void)v;
    return apy_none();
}

APY_API apy_value apy_default_hash(apy_value v) {
    return apy_from_int((int64_t)(v >> 3));
}

/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_is_data_descriptor(apy_value v) {
    return (int)apy_is_data_descriptor_of(v);
}

/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_descr_set(apy_value d, apy_value obj,
                         apy_value value) {
    return (int)apy_descr_set_of(d, obj, value);
}

APY_API apy_value apy_default_setattr(apy_value obj, apy_value name,
                                      apy_value value);

APY_API apy_value apy_setattr(apy_value obj, apy_value name, apy_value value) {
    name = apy_text_like(name);
    /* `__setattr__` INTERCEPTS EVERY assignment, the mirror of
       `__getattribute__`. Asked here rather than inside the default so that
       the default stays callable from within the override -- which is what
       `object.__setattr__(self, name, value)` is for, and the only way an
       override can actually store anything. */
    /* `C.__name__ = ...` CHANGES WHAT THE CLASS IS CALLED. The name is a
       field on the type, not an entry in its dict, so storing it as an
       ordinary attribute left `__name__` reading the old one -- the write
       appeared to succeed and changed nothing. */
    if (O(obj)->kind == APY_TYPE_K && O(name)->kind == APY_STR_K
            && strcmp(APY_CSTR(name), "__name__") == 0) {
        /* AND THE EXCEPTION REGISTRATION FOLLOWS THE RENAME. The hierarchy is
           a table of NAMES, so a class renamed after it was registered leaves
           the two disagreeing. Not hypothetical: a BUNDLED module's classes
           are spliced under mangled names and the splice then restores
           `__name__` -- precisely so the mangling stays invisible -- so
           `copy.Error` registered as `_asmpy_bundled_4_copy_Error` and then
           started calling itself `Error`, and `issubclass(copy.Error,
           Exception)` asked the table for `Error`, found nothing, and answered
           False for a class whose `class` statement names Exception as its
           base. BOTH spellings are kept: generated code raises through the
           mangled one. */
        const char *was = APY_CSTR(O(obj)->v.t.name);
        const char *parent = apy_exc_parent(was);
        apy_value found = apy_exc_class_named(was);
        O(obj)->v.t.name = value;
        if (parent && strcmp(was, APY_CSTR(value)) != 0) {
            apy_value pname = apy_lit(parent);
            apy_exc_register(value, pname);
            /* BOTH SPELLINGS, and `!found` is the case that matters. An
               exception class with an EMPTY BODY is never handed to
               `apy_exc_class_bind` at all -- `_dyn_class` early-returns
               because there is nothing to build -- so nothing was registered
               under the mangled name, `apy_exc_construct` could not find a
               class, and every display fell back to the name the CELL
               carries. Right for a user's class, wrong for a bundled one
               whose cells carry the mangled spelling. */
            if (found == obj || !found) {
                apy_exc_class_bind(value, obj);
                apy_exc_class_bind(apy_lit(was), obj);
            }
        }
        return apy_none();
    }
    if (O(obj)->kind == APY_INST_K) {
        apy_value hook = apy_class_find(O(obj)->v.o.cls,
                                        apy_name("__setattr__"));
        if (hook) {
            apy_value argv[2];
            argv[0] = name;
            argv[1] = value;
            return apy_call_n(apy_bind(hook, obj), argv, 2);
        }
    }
    return apy_default_setattr(obj, name, value);
}

/* Is this instance restricted to a fixed set of attributes, and is `name`
   one of them? Answers 1 to allow and 0 to refuse.

   An instance is unrestricted unless EVERY class in its chain declares
   `__slots__` -- one that does not gives the dict back, and with it the
   freedom to set anything. */
APY_API int64_t apy_slot_allows_of(apy_value cls, apy_value name) {
    apy_value here = cls;
    int64_t at;
    while (here && O(here)->kind == APY_TYPE_K) {
        at = apy_dict_find(O(here)->v.t.dict, apy_name("__slots__"));
        if (at < 0) return 1;                  /* no `__slots__`: a dict */
        here = O(here)->v.t.base;
    }
    /* Every class declares one, so the name has to appear in one of them. */
    here = cls;
    while (here && O(here)->kind == APY_TYPE_K) {
        at = apy_dict_find(O(here)->v.t.dict, apy_name("__slots__"));
        if (at >= 0) {
            apy_value names = O(O(here)->v.t.dict)->v.d.vals[at];
            int64_t i, n = apy_raw_len(names);
            if (apy_error_occurred()) { apy_error_clear(); return 1; }
            if (O(names)->kind == APY_STR_K)
                return apy_eq_raw(names, name);
            for (i = 0; i < n; i++)
                if (apy_eq_raw(apy_key_at(names, i), name)) return 1;
        }
        here = O(here)->v.t.base;
    }
    return 0;
}

/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_slot_allows(apy_value cls, apy_value name) {
    return (int)apy_slot_allows_of(cls, name);
}


APY_API apy_value apy_default_setattr(apy_value obj, apy_value name,
                                      apy_value value) {
    if (O(obj)->kind == APY_INST_K
            && !apy_slot_allows(O(obj)->v.o.cls, name))
        return apy_fail2("AttributeError",
                         "'%s' object has no attribute '%s' and no __dict__ "
                         "for setting new attributes",
                         apy_kind_name(obj), APY_CSTR(name));
    if (O(obj)->kind == APY_INST_K) {
        /* A DATA DESCRIPTOR ON THE CLASS TAKES THE WRITE. `c.v = 4` where the
           class has a `property` runs its setter and stores nothing in the
           instance dict -- otherwise the next read would find the stored
           value and the property would never be consulted again. */
        apy_value found = apy_class_find(O(obj)->v.o.cls, name);
        if (found) {
            int handled = apy_descr_set(found, obj, value);
            if (handled == 0) return 0;
            if (handled == 1) return apy_none();
        }
        if (!apy_dict_set(O(obj)->v.o.dict, name, value)) return 0;
        return apy_none();
    }
    if (O(obj)->kind == APY_EXC_K) {
        /* `self.code = code` in a user exception's `__init__`. The dict is
           made on first write, so an exception the runtime raises itself --
           which is nearly all of them -- costs nothing for this. */
        if (!O(obj)->v.e.dict) O(obj)->v.e.dict = apy_dict_new(4);
        if (!apy_dict_set(O(obj)->v.e.dict, name, value)) return 0;
        return apy_none();
    }
    if (O(obj)->kind == APY_TYPE_K) return apy_type_set(obj, name, value);
    if (O(obj)->kind == APY_FUNC_K) {
        /* A function carries whatever a program hangs on it. The dict is
           made on first write so an ordinary `def` costs nothing. */
        if (!O(obj)->v.fn.dict) O(obj)->v.fn.dict = apy_dict_new(4);
        if (!apy_dict_set(O(obj)->v.fn.dict, name, value)) return 0;
        return apy_none();
    }
    return apy_fail2("AttributeError",
                     "'%s' object has no attribute '%s'",
                     apy_kind_name(obj), APY_CSTR(name));
}

/* `str.upper(5)` -- AN UNBOUND METHOD APPLIED TO THE WRONG KIND.

   CPython checks the receiver against the type the descriptor was found on
   before it runs, and words the refusal as the DESCRIPTOR's: `descriptor
   'upper' for 'str' objects doesn't apply to a 'int' object`. Without the
   check the call went straight through as `(5).upper()` and reported a
   missing attribute -- true of the int, and not what the program got wrong.

   THE RECEIVER COMES BACK so the check sits in the expression rather than
   beside it: the call site is `apy_descr_applies(recv, "str", "upper")` where
   it would otherwise be `recv`.

   `apy_isinstance` AND NOT A KIND COMPARISON, because a SUBCLASS passes:
   `int.bit_length(True)` is 1 in CPython and a bool is an int. */
APY_API apy_value apy_descr_applies(apy_value recv, apy_value type_name,
                                    apy_value method_name) {
    apy_value ok = apy_isinstance(recv, type_name);
    char buf[256];
    if (!ok) return 0;
    if (apy_truth(ok)) return recv;
    snprintf(buf, sizeof buf,
             "descriptor '%s' for '%s' objects doesn't apply to a "
             "'%s' object", APY_CSTR(method_name),
             APY_CSTR(type_name), apy_kind_name(recv));
    return apy_fail("TypeError", buf);
}

APY_API apy_value apy_super(apy_value from, apy_value self) {
    apy_obj *o;
    if (O(from)->kind != APY_TYPE_K)
        return apy_fail("TypeError", "super(type, obj): obj must be an "
                                     "instance or subtype of type");
    o = apy_alloc(APY_SUPER_K);
    o->v.sup.from = from;
    o->v.sup.self = self;
    return V(o);
}

"""
