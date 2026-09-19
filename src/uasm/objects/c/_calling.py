"""The object runtime, in C: calling, type objects and operator dispatch.

ONE PART OF ONE TRANSLATION UNIT. `c/__init__.py` concatenates
these in order and the result is the file it always was, so a
definition here may rely on anything in an earlier part and
nothing in a later one. Sections, in order:
  * calling
  * type objects
  * operator dispatch to user methods
"""

C = r"""/* --- calling ------------------------------------------------------------ */

/* The arity switch. Nine is the ceiling because `env` occupies one of the
   platform's argument registers and eight declared parameters is already far
   past anything the suite writes; a tenth would be another line here and no
   new idea. Every cast is to a function of `apy_value` arguments returning
   one, which is what every dynamic function compiles to. */
typedef apy_value (*apy_fn0)(apy_value);
typedef apy_value (*apy_fn1)(apy_value, apy_value);
typedef apy_value (*apy_fn2)(apy_value, apy_value, apy_value);
typedef apy_value (*apy_fn3)(apy_value, apy_value, apy_value, apy_value);
typedef apy_value (*apy_fn4)(apy_value, apy_value, apy_value, apy_value,
                             apy_value);
typedef apy_value (*apy_fn5)(apy_value, apy_value, apy_value, apy_value,
                             apy_value, apy_value);
typedef apy_value (*apy_fn6)(apy_value, apy_value, apy_value, apy_value,
                             apy_value, apy_value, apy_value);
typedef apy_value (*apy_fn7)(apy_value, apy_value, apy_value, apy_value,
                             apy_value, apy_value, apy_value, apy_value);
typedef apy_value (*apy_fn8)(apy_value, apy_value, apy_value, apy_value,
                             apy_value, apy_value, apy_value, apy_value,
                             apy_value);
/* PAST EIGHT. `datetime.replace` declares ten parameters and is ordinary
   Python, so the old ceiling was a limit of this dispatch rather than of the
   language. Written out because a call through a function pointer needs the
   exact arity at the call site -- there is no way to spell "n arguments" in C
   without varargs, and varargs would change the ABI every backend shares. */
typedef apy_value (*apy_fn9)(apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value);
typedef apy_value (*apy_fn10)(apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value);
typedef apy_value (*apy_fn11)(apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value);
typedef apy_value (*apy_fn12)(apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value);
typedef apy_value (*apy_fn13)(apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value);
typedef apy_value (*apy_fn14)(apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value);
typedef apy_value (*apy_fn15)(apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value);
typedef apy_value (*apy_fn16)(apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value, apy_value);

/* One value per selector, so `super().__init__ is super().__init__` holds
   the way it does for any other attribute reached twice -- and so the repr
   does not depend on how many times it was asked for. Binding copies it, as
   binding any function does. */
/* Declared here because the exported half calls it. */
static apy_value apy_native(int sel, int64_t arity, const char *name);
/* THE TYPE OBJECT FOR A VALUE'S KIND, declared here because
   `__class_getitem__` needs the origin of `list[int]` and the definition is
   in the type-objects section below. */
APY_API apy_value apy_type_for(apy_value v);
APY_API apy_value apy_native_of(int64_t sel, int64_t arity,
                                apy_value name) {
    return apy_native((int)sel, arity, (const char *)name);
}
static apy_value apy_native(int sel, int64_t arity, const char *name) {
    static apy_value made[APY_NAT_GEN_CLOSE + 1];
    apy_obj *o;
    /* INTERNED PER SELECTOR, so `super().__init__` reached twice is one
       object as any other attribute would be -- EXCEPT for the one selector
       whose name and arity vary, where interning handed back whichever was
       built first and every builtin protocol method then behaved as that one.
       A fresh object there is also what CPython answers: `[].append is
       [].append` is False. */
    if (sel != APY_NAT_KIND && made[sel]) return made[sel];
    o = apy_alloc(APY_FUNC_K);
    o->v.fn.code = 0;
    o->v.fn.native = sel;
    o->v.fn.arity = arity;
    o->v.fn.name = apy_lit(name);
    /* THE CONTENT ARGUMENT IS OPTIONAL for the three that can stand in for a
       builtin base's constructor, because `dict.__init__` takes nought or one
       and `super().__init__()` with nothing is the ordinary spelling -- as is
       `object.__new__(cls)` beside `object.__new__(cls, content)`. The arity
       check further down fills a missing trailing slot from `defaults` and
       then insists the count matches exactly, so an omitted argument was an
       arity error naming a method the program never declared. */
    if (sel == APY_NAT_BUILTIN_INIT || sel == APY_NAT_BUILTIN_NEW
            || sel == APY_NAT_NEW) {
        static apy_value absent[1];
        absent[0] = apy_none();
        o->v.fn.ndefaults = 1;
        o->v.fn.defaults = absent;
    }
    if (sel == APY_NAT_KIND) return V(o);
    made[sel] = V(o);
    return made[sel];
}

/* `type(name, bases, ns)` as an OBJECT: what `super().__new__` inside a
   metaclass's `__new__` answers. The class it builds records `mcls` as its
   metaclass, which is what makes `type(C)` say `Meta`. */
/* THE C3 LINEARISATION: the order attribute lookup walks.

   With one base it is the base chain and nothing is gained by computing it.
   With several it is the only order that keeps two promises at once -- a class
   comes before its bases, and the bases keep the order they were written in --
   and no simple walk can keep both. `class D(B, C)` over a diamond has to find
   B's method before C's and C's before A's, which depth-first does not.

   Answers 0 and reports a TypeError when no order satisfies both, which is
   what CPython does for `class Z(X, Y)` where X and Y disagree. */
static apy_value apy_c3(apy_value cls, apy_value bases) {
    apy_value out = apy_tuple_new(8);
    apy_value queues[9];
    int64_t heads[9], count = 0, i, j, k;
    int64_t nbases = (bases && apy_is_seq(bases)) ? O(bases)->v.q.n : 0;
    apy_seq_push(out, cls);
    if (nbases > 8) return apy_fail("TypeError", "too many bases");
    /* THE LISTS TO MERGE: each base's own linearisation, then the list of
       bases itself -- which is what makes the written order binding. */
    for (i = 0; i < nbases; i++) {
        apy_value b = O(bases)->v.q.items[i];
        if (O(b)->kind != APY_TYPE_K)
            return apy_fail("TypeError", "a base must be a class");
        queues[count] = O(b)->v.t.mro ? O(b)->v.t.mro : 0;
        if (!queues[count]) {
            /* A base with no recorded MRO is its own chain. */
            apy_value chain = apy_tuple_new(4);
            apy_value walk = b;
            while (walk && O(walk)->kind == APY_TYPE_K) {
                apy_seq_push(chain, walk);
                walk = O(walk)->v.t.base;
            }
            queues[count] = chain;
        }
        heads[count] = 0;
        count++;
    }
    queues[count] = bases ? bases : apy_tuple_new(1);
    heads[count] = 0;
    count++;
    for (;;) {
        apy_value chosen = 0;
        int64_t done = 1;
        for (i = 0; i < count; i++)
            if (heads[i] < O(queues[i])->v.q.n) { done = 0; break; }
        if (done) break;
        /* THE FIRST HEAD THAT IS IN NO TAIL. A candidate appearing in some
           other list's tail must wait for that list, or the result would put
           it before something that has to come first. */
        for (i = 0; i < count && !chosen; i++) {
            apy_value head;
            int blocked = 0;
            if (heads[i] >= O(queues[i])->v.q.n) continue;
            head = O(queues[i])->v.q.items[heads[i]];
            for (j = 0; j < count && !blocked; j++)
                for (k = heads[j] + 1; k < O(queues[j])->v.q.n; k++)
                    if (O(queues[j])->v.q.items[k] == head) { blocked = 1; break; }
            if (!blocked) chosen = head;
        }
        if (!chosen)
            return apy_fail("TypeError",
                            "Cannot create a consistent method resolution "
                            "order (MRO) for bases");
        {
            int64_t already = 0;
            for (i = 0; i < O(out)->v.q.n; i++)
                if (O(out)->v.q.items[i] == chosen) { already = 1; break; }
            if (!already) apy_seq_push(out, chosen);
        }
        for (i = 0; i < count; i++)
            if (heads[i] < O(queues[i])->v.q.n
                && O(queues[i])->v.q.items[heads[i]] == chosen)
                heads[i]++;
    }
    return out;
}

/* THE BUILTIN A `class` STATEMENT NAMED, waiting for the class to exist.

   `class Colour(str, Enum)` records its kind through `apy_type_builtin`, and
   that runs only once `apy_class_build` has ANSWERED -- which for a class
   with a metaclass is after the metaclass body has finished. An `EnumMeta`
   makes every member inside that body, so the members were built against a
   class that did not yet know it extended anything, and `isinstance(
   Colour.RED, str)` was False, `len(Colour.RED)` a TypeError, and
   `Colour.RED == "red"` False.

   THE NAME IS PART OF THE CELL so that a `class` statement inside a
   metaclass body cannot take the tag meant for the class being built: only a
   creation of the same name consumes it. The lowering sets it immediately
   before `apy_class_build` and that call clears it again, so nothing outlives
   one statement. */
static apy_value apy_builtin_pending_name = 0;
static int64_t apy_builtin_pending_kind = 0;

APY_API apy_value apy_type_builtin_pending(apy_value name, int64_t kind) {
    apy_builtin_pending_name = name;
    apy_builtin_pending_kind = kind;
    return apy_none();
}

static apy_value apy_type_from_ns(apy_value mcls, apy_value name,
                                  apy_value bases, apy_value ns) {
    apy_value base = 0, cls;
    int64_t i;
    if (bases && apy_is_seq(bases) && O(bases)->v.q.n > 0)
        base = O(bases)->v.q.items[0];
    cls = apy_type_new(name, base ? base : apy_none());
    if (!cls) return 0;
    /* THE KIND IS RECORDED HERE, where a metaclass's `super().__new__`
       reaches -- see `apy_type_builtin_pending`. */
    if (apy_builtin_pending_kind && name && apy_builtin_pending_name
            && O(name)->kind == APY_STR_K
            && O(apy_builtin_pending_name)->kind == APY_STR_K
            && O(name)->v.s.n == O(apy_builtin_pending_name)->v.s.n
            && memcmp(O(name)->v.s.p, O(apy_builtin_pending_name)->v.s.p,
                      (size_t)O(name)->v.s.n) == 0) {
        O(cls)->v.t.builtin = apy_builtin_pending_kind;
        apy_builtin_pending_kind = 0;
        apy_builtin_pending_name = 0;
    }
    if (mcls && O(mcls)->kind == APY_TYPE_K) O(cls)->v.t.meta = mcls;
    if (bases && apy_is_seq(bases) && O(bases)->v.q.n > 0) {
        apy_value order;
        O(cls)->v.t.bases = bases;
        order = apy_c3(cls, bases);
        if (!order) return 0;
        O(cls)->v.t.mro = order;
    }
    /* The namespace is COPIED IN, not adopted: `__prepare__` may hand back a
       mapping the program goes on using, and a class that shared it would see
       later writes to it. */
    if (ns && O(ns)->kind == APY_DICT_K)
        for (i = 0; i < O(ns)->v.d.n; i++)
            apy_dict_set(O(cls)->v.t.dict, O(ns)->v.d.keys[i],
                         O(ns)->v.d.vals[i]);
    /* `__module__` IS THE CONSTRUCTOR'S TO SUPPLY when the namespace carried
       none, exactly as CPython's `type_new` supplies it from the caller's
       globals. A class written out gets one from its body (see `_dyn_class`)
       and reaches here with it already in `ns`; one built by `type(name,
       bases, {})` -- which is what `enum`, `dataclasses` and `namedtuple`
       all do -- had none at all, so it printed as `<class 'X'>` where
       CPython says `<class '__main__.X'>`. There is no frame to ask, and a
       program's own module is the only one that can reach this. */
    if (!apy_dict_get_or(O(cls)->v.t.dict, apy_name("__module__"), 0))
        apy_dict_set(O(cls)->v.t.dict, apy_name("__module__"),
                     apy_lit("__main__"));
    return cls;
}

/* `type` AS A CLASS OBJECT, so `class Meta(type)` has a real base rather
   than a special case. Its dict holds the two natives a metaclass reaches
   through `super()`, which is what makes `super().__new__(mcls, name, bases,
   ns)` build a class instead of an instance. Interned: `Meta.__base__ is
   type` has to hold, and two of them would make it False. */
/* PEP 3115: the mapping a class body is executed into. A metaclass may
   supply one through `__prepare__`, which is how a body's bindings can be
   seen in order or pre-seeded; a metaclass without one gets a plain dict.

   Asked for even when there is no `__prepare__`, so the body always writes
   into a mapping rather than into the type -- one lowering for both. */
APY_API apy_value apy_prepare(apy_value meta, apy_value name, apy_value bases) {
    apy_value hook, args[2], got;
    if (!meta || O(meta)->kind != APY_TYPE_K) return apy_dict_new(8);
    hook = apy_class_find(meta, apy_name("__prepare__"));
    if (!hook) return apy_dict_new(8);
    /* AN IMPLICIT CLASSMETHOD, like `__init_subclass__`: it receives the
       metaclass, and a program writes `@classmethod` above it because
       CPython wants that spelling -- the descriptor is unwrapped here. */
    if (O(hook)->kind == APY_PROP_K && O(hook)->v.p.get) {
        /* THE METACLASS IS ITS FIRST ARGUMENT, which is what `@classmethod`
           on it means -- unwrapping the descriptor without supplying that
           called a three-parameter function with two. */
        apy_value three[3];
        three[0] = meta;
        three[1] = name;
        three[2] = bases;
        got = apy_call_n(O(hook)->v.p.get, three, 3);
    } else {
        args[0] = name;
        args[1] = bases;
        got = apy_call_n(hook, args, 2);
    }
    if (!got) return 0;
    if (O(got)->kind != APY_DICT_K)
        return apy_fail("TypeError",
                        "__prepare__() must return a mapping");
    return got;
}

/* `type(name, bases, ns)` -- the three-argument form, which is the `class`
   statement written out. The same builder a metaclass's `super().__new__`
   reaches, with no metaclass recorded: one made this way IS a plain `type`. */
APY_API apy_value apy_type_make(apy_value name, apy_value bases,
                                apy_value ns) {
    return apy_type_from_ns(0, name, bases, ns);
}

/* THE METACLASS A `class` STATEMENT SHOULD USE: the one written, or the one
   its base already has. A subclass of a class with a metaclass has the same
   metaclass -- that is what makes `class Shape(ABC)` collect its own abstract
   methods without repeating `metaclass=ABCMeta`. */
/* PEP 560: WHAT A NON-CLASS BASE CONTRIBUTES.

   `class C(Fake())` asks the object for `__mro_entries__(bases)` and inherits
   whatever it answers -- which is how a generic alias resolves to its origin
   and how a library builds a base at run time. A class contributes itself,
   which is the ordinary case and costs one test. */
APY_API apy_value apy_mro_entries(apy_value written, apy_value bases) {
    apy_value hook, got;
    if (O(written)->kind == APY_TYPE_K) return written;
    if (O(written)->kind != APY_INST_K)
        return apy_fail2("TypeError", "bases must be types, not '%s'%s",
                         apy_kind_name(written), "");
    hook = apy_class_find(O(written)->v.o.cls, apy_name("__mro_entries__"));
    if (!hook)
        return apy_fail2("TypeError", "bases must be types, not '%s'%s",
                         apy_kind_name(written), "");
    got = apy_call_n(apy_bind(hook, written), &bases, 1);
    if (!got) return 0;
    /* THE FIRST ENTRY. `__mro_entries__` answers a tuple because one object
       may contribute several bases; this runtime linearises from a flat list,
       so the rest would need splicing into the caller's tuple -- which is
       what the caller does, one entry at a time. */
    if (apy_is_seq(got) && O(got)->v.q.n > 0) return O(got)->v.q.items[0];
    return apy_object_class();
}

APY_API apy_value apy_meta_for(apy_value given, apy_value bases) {
    int64_t i;
    if (given && O(given)->kind == APY_TYPE_K) return given;
    if (bases && apy_is_seq(bases))
        for (i = 0; i < O(bases)->v.q.n; i++) {
            apy_value base = O(bases)->v.q.items[i];
            if (O(base)->kind == APY_TYPE_K && O(base)->v.t.meta)
                return O(base)->v.t.meta;
        }
    return apy_none();
}

/* Build the class a `class` statement describes.

   THROUGH THE METACLASS when there is one, which is what makes `ABCMeta`
   able to refuse an instantiation and `EnumMeta` able to rewrite the body.
   Without one this is the plain construction, and the two paths meet here so
   the lowering does not have to know which it is -- it cannot, because
   whether a base carries a metaclass is a run-time question. */
APY_API apy_value apy_class_build_kw(apy_value meta, apy_value name,
                                     apy_value bases, apy_value ns,
                                     apy_value kw) {
    apy_value use = apy_meta_for(meta, bases);
    if (use && O(use)->kind == APY_TYPE_K) {
        apy_value argv[3];
        apy_value built;
        argv[0] = name;
        argv[1] = bases;
        argv[2] = ns;
        /* THE CLASS KEYWORDS GO TO THE METACLASS -- `class C(metaclass=M,
           kind="x")` is `M(name, bases, ns, kind="x")`. Only when there IS
           one: without a metaclass they are for `__init_subclass__`, which
           the caller announces separately, and handing them to the plain
           construction would make them an arity error. */
        if (kw && O(kw)->kind == APY_DICT_K && O(kw)->v.d.n)
            built = apy_call_kw(use, (apy_value)(uintptr_t)argv, 3, kw);
        else
            built = apy_call_n(use, argv, 3);
        /* NOTHING PENDING OUTLIVES ONE STATEMENT. A metaclass that answers
           something it did not build with `type()` leaves the cell unread,
           and the next class of the same name would take it. */
        apy_builtin_pending_kind = 0;
        apy_builtin_pending_name = 0;
        return built;
    }
    {
        apy_value made = apy_type_from_ns(0, name, bases, ns);
        apy_builtin_pending_kind = 0;
        apy_builtin_pending_name = 0;
        return made;
    }
}

APY_API apy_value apy_class_build(apy_value meta, apy_value name,
                                  apy_value bases, apy_value ns) {
    return apy_class_build_kw(meta, name, bases, ns, 0);
}

/* An EMPTY value of a builtin kind: what `str.__new__(S)` puts inside. The
   same five `apy_instance_new` fills a held slot with. */
static apy_value apy_empty_of_kind(int kind) {
    if (kind == APY_DICT_K) return apy_dict_new(4);
    if (kind == APY_LIST_K) return apy_list_new(4);
    if (kind == APY_SET_K) return apy_set_new(4);
    if (kind == APY_TUPLE_K) return apy_tuple_new(1);
    if (kind == APY_STR_K) return apy_lit("");
    return apy_none();
}

/* `str.__new__(cls, content)` -- the builtin half of an instance, built
   through the type it extends. This is how CPython's own `enum` makes a
   mixin member: `member_type.__new__(enum_class, value)`.

   AN IMPLICIT STATICMETHOD, so the first argument is the CLASS TO BUILD and
   not a receiver of this type. The ordinary unbound-method check --
   `apy_descr_applies` -- therefore does not apply to it, and did: `str.__new__
   (S, "hi")` was `descriptor '__new__' for 'str' objects doesn't apply to a
   'type' object`, about a class that is exactly what the call meant to name.

   THE CONTENT IS TAKEN ONLY BY THE IMMUTABLE KINDS. `list.__new__(L, [1, 2])`
   is an EMPTY list in CPython and `tuple.__new__(T, [1, 2])` is `(1, 2)`: a
   mutable builtin fills in `__init__`, and an immutable one has nowhere else
   to do it.

   THE BUILTIN ITSELF ANSWERS A PLAIN ONE. `str.__new__(str, "ab")` is
   `"ab"` -- there is no class to put it in, and CPython answers the bare
   value. */
APY_API apy_value apy_builtin_new(apy_value type_name, int64_t kind,
                                  apy_value cls, apy_value content) {
    const char *want = APY_CSTR(type_name);
    const char *given = 0;
    char buf[192];
    apy_value made;
    int fills = kind == APY_STR_K || kind == APY_TUPLE_K;
    int has = content && O(content)->kind != APY_NONE_K;
    if (O(cls)->kind == APY_FUNC_K && O(cls)->v.fn.is_type) {
        given = APY_CSTR(O(cls)->v.fn.name);
        if (strcmp(given, want) == 0)
            return has ? apy_call_kind((int)kind, content)
                       : apy_empty_of_kind((int)kind);
    } else if (O(cls)->kind == APY_TYPE_K) {
        given = APY_CSTR(O(cls)->v.t.name);
        if (apy_class_builtin_kind(cls) == kind) {
            made = apy_instance_new(cls);
            if (!made) return 0;
            if (fills && has && O(made)->kind == APY_INST_K
                    && O(made)->v.o.held) {
                apy_value filled = apy_call_kind((int)kind, content);
                if (!filled) return 0;
                O(made)->v.o.held = filled;
            }
            return made;
        }
    } else {
        /* NOT A TYPE AT ALL, which CPython words differently from a type
           that is simply the wrong one -- and writes as a literal `X`. */
        snprintf(buf, sizeof buf, "%s.__new__(X): X is not a type object (%s)",
                 want, apy_kind_name(cls));
        return apy_fail("TypeError", buf);
    }
    snprintf(buf, sizeof buf, "%s.__new__(%s): %s is not a subtype of %s",
             want, given, given, want);
    return apy_fail("TypeError", buf);
}

/* `object` AS A CLASS OBJECT -- what `C.__base__` answers for a class with
   no written base, and what `C.__bases__` holds. Its dict carries the same
   defaults `super()` falls back to.

   NOT INSTALLED AS AN ACTUAL BASE POINTER on every class. It is the honest
   ANSWER to a question about the hierarchy; making it a real link would put
   `__eq__` and friends into every `apy_class_find` walk, which changes what
   `hasattr` says about classes that define none of them. */
APY_API apy_value apy_object_class(void) {
    static apy_value cls = 0;
    static const char *names[] = {"__init__", "__new__", "__repr__", "__str__",
                                  "__eq__", "__ne__", "__hash__",
                                  "__getattribute__", "__setattr__",
                                  "__delattr__"};
    size_t i;
    if (cls) return cls;
    cls = apy_type_new(apy_lit("object"), 0);
    for (i = 0; i < sizeof names / sizeof names[0]; i++)
        apy_dict_set(O(cls)->v.t.dict, apy_name(names[i]),
                     apy_object_default(
                         (apy_value)(uintptr_t)names[i]));
    return cls;
}

APY_API apy_value apy_type_class(void) {
    static apy_value cls = 0;
    if (cls) return cls;
    cls = apy_type_new(apy_lit("type"), 0);
    apy_dict_set(O(cls)->v.t.dict, apy_name("__new__"),
                 apy_native(APY_NAT_TYPE_NEW, 4, "__new__"));
    apy_dict_set(O(cls)->v.t.dict, apy_name("__init__"),
                 apy_native(APY_NAT_TYPE_INIT, 4, "__init__"));
    /* What `C(...)` MEANS, reachable by name so a metaclass's own `__call__`
       can delegate to it. */
    apy_dict_set(O(cls)->v.t.dict, apy_name("__call__"),
                 apy_native(APY_NAT_TYPE_CALL, 1, "__call__"));
    return cls;
}

/* `__class__` for a kind with no attributes of its own -- a generic alias,
   a slice, a view.

   THE SAME OBJECT `type(x)` ANSWERS, and it kept its OWN interning table
   until `__class__` reached the ordinary kinds: two tables, each interned
   correctly, each handing out a different type object for `list` -- so
   `[1].__class__ is type([1])` was False while `type([1]) is type([2])` was
   True. One concept has one table, and `apy_type_for`'s is the one that also
   honours the canonical registration, which is what makes `type(1) is int`
   hold. */
APY_API apy_value apy_kind_class(apy_value obj) {
    return apy_type_of(obj);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */

static apy_value apy_instantiate(apy_value f, apy_value *argv, int64_t argc,
                                 apy_value kwrest, int bound);
static apy_value apy_call_nk(apy_value f, apy_value *argv, int64_t argc,
                             apy_value kwrest, int bound);

/* A BUILTIN'S PROTOCOL METHODS, AS VALUES.

   `[].append` and `{}.keys` are lowered at the call site by the frontend,
   which means they exist as CALLS and never as attributes -- so
   `hasattr([1], "__iter__")` answered False for the most iterable object in
   the language, and every structural type test written against
   `collections.abc` said no.

   This does not make the whole method table reachable by name; it makes the
   PROTOCOL reachable, which is the part a program asks about rather than
   calls. Answers 0 for a name the kind does not have -- that is what keeps
   `hasattr` honest -- and `None` where CPython HAS the attribute and sets it
   to None, which is how a mutable container says it cannot be hashed. */
/* One protocol method as a value, bound to `obj` when there IS one. A TYPE
   asked the same question has no receiver -- `dict.keys` is unbound in
   CPython too, and `dict.keys(d)` is how it is called. */
APY_API apy_value apy_kind_method_of(apy_value obj, int64_t arity,
                                    apy_value namev, int64_t bind) {
    const char *name = (const char *)namev;
    apy_value fn = apy_native(APY_NAT_KIND, arity, name);
    return bind ? apy_bind(fn, obj) : fn;
}

/* THE SAME, WITH AN OPTIONAL TAIL: the method may be called with up to `nopt`
   fewer arguments than it declares.

   `x.find(sub)`, `x.find(sub, i)` and `x.find(sub, i, j)` are ONE METHOD, and
   a cell carrying one arity could not say so: `apy_call_nk` truncates a
   surplus argument (`take = byslot - n`) and then finds the count it expected,
   so `getattr([1], "__len__")(9)` answered 1 rather than refusing, and no
   method with an optional argument could be reached by name at all.

   RECORDED AS `ndefaults` WITH A NULL `defaults`, which no other cell has:
   there are no VALUES to fill a missing slot with -- the body dispatches on
   how many it actually got. A separate entry point rather than a parameter
   on the one above, because eighty-five call sites take a fixed arity and
   say so. */
APY_API apy_value apy_kind_method_opt(apy_value obj, int64_t arity,
                                      int64_t nopt, apy_value namev,
                                      int64_t bind) {
    const char *name = (const char *)namev;
    apy_value fn = apy_native(APY_NAT_KIND, arity, name);
    O(fn)->v.fn.ndefaults = nopt;
    O(fn)->v.fn.defaults = 0;
    return bind ? apy_bind(fn, obj) : fn;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
/* HOW MANY STEPS A CURSOR HAS LEFT, for `__length_hint__`.

   AN ESTIMATE AND NOT A PROMISE, which is what the hint is for: the source
   may grow or shrink under the walk, and CPython's own answer goes stale the
   same way. What it must never be is NEGATIVE -- `operator.length_hint`
   raises ValueError on one.

   A REVERSED CURSOR COUNTS DOWN from where `reversed` started it, and -1 is
   its exhaustion, so what it has left is the position plus one. */
static int64_t apy_cursor_left(apy_value it) {
    int64_t at = O(it)->v.it.i, n;
    if (O(it)->v.it.mode == APY_IT_REV) return at < 0 ? 0 : at + 1;
    n = apy_raw_len(O(it)->v.it.src);
    /* A SOURCE WITH NO LENGTH answers nothing rather than failing: the walk
       through `__getitem__` is the shape that has none, and a hint is
       allowed to say it does not know. */
    if (apy_error_occurred()) { apy_error_clear(); return 0; }
    return at >= n ? 0 : n - at;
}

/* Does this cursor carry `__setstate__`, and which ones do?

   READ OUT OF CPYTHON'S OWN `dir()` and transcribed here rather than
   reasoned about: the sequence walks have it -- a list, tuple, str, bytes or
   range, forward or reversed, and a reversed memoryview, which CPython calls
   a plain `reversed` -- and so do `zip` and `map`. A set, a dict's three
   walks, a `callable_iterator`, a forward `memory_iterator`, `enumerate` and
   `filter` do NOT, and a `dir()` that claimed otherwise would be a list that
   lies. The same line is drawn in `_gen_kindmeth.py`'s table, which is where
   the claim comes from. */
static int apy_cursor_setstate_p(apy_value it) {
    int rev, base;
    if (O(it)->kind != APY_ITER_K) return 0;
    if (O(it)->v.it.mode == APY_IT_MAP || O(it)->v.it.mode == APY_IT_ZIP)
        return 1;
    if (O(it)->v.it.mode != APY_IT_PLAIN && O(it)->v.it.mode != APY_IT_REV)
        return 0;
    rev = O(it)->v.it.named >= APY_IT_REVOF;
    base = O(it)->v.it.named - (rev ? APY_IT_REVOF : 0);
    if (rev && base == APY_MVIEW_K) return 1;
    return base == APY_LIST_K || base == APY_TUPLE_K || base == APY_STR_K
        || base == APY_BYTES_K || base == APY_RANGE_K;
}

/* `it.__setstate__(i)` -- WHERE THE WALK IS, written rather than read. The
   other half of `__reduce__`: pickle remakes a partly consumed iterator and
   then says how far it had got.

   THE CLAMPING IS CPYTHON'S, measured rather than derived. A forward cursor
   takes 0..len and anything outside that -- above OR BELOW -- leaves it
   exhausted, so `iter([1,2,3]).__setstate__(-1)` yields nothing. A reversed
   one counts down from `i` and -1 is already its exhaustion, so a position
   past the end is pulled back to the last element and a negative one stays
   the end of the walk.

   `zip` AND `map` KEEP NO POSITION OF THEIR OWN -- what CPython pickles for
   them is their sub-iterators -- so theirs takes the argument and changes
   nothing, which is what the same call does there. */
static apy_value apy_cursor_setstate(apy_value it, apy_value where) {
    int64_t at, n;
    if (!apy_is_int_like(where))
        return apy_fail("TypeError", "an integer is required");
    at = apy_index(where);
    if (apy_err_type) return 0;
    if (O(it)->v.it.mode == APY_IT_MAP || O(it)->v.it.mode == APY_IT_ZIP)
        return apy_none();
    n = apy_raw_len(O(it)->v.it.src);
    /* A SOURCE WITH NO LENGTH counts as empty rather than failing, which is
       the reading `apy_cursor_left` takes of the same question. */
    if (apy_error_occurred()) { apy_error_clear(); n = 0; }
    if (O(it)->v.it.mode == APY_IT_REV) {
        if (at >= n) at = n - 1;
        O(it)->v.it.i = at < 0 ? -1 : at;
        return apy_none();
    }
    O(it)->v.it.i = (at < 0 || at > n) ? n : at;
    return apy_none();
}

static apy_value apy_kind_method(apy_value obj, int64_t arity,
                                 const char *name, int bind) {
    return apy_kind_method_of(obj, arity,
                              (apy_value)(uintptr_t)name,
                              (int64_t)bind);
}

/* THE SAME, TAKING WHATEVER IT IS GIVEN. `"{} {}".format(a, b)` has no
   argument count to declare and keywords besides, so neither a fixed arity
   nor an optional tail can say what it accepts. DECLARED AS THREE WITH BOTH
   VARIADIC PARTS -- receiver, the surplus as a tuple, the keywords as a
   dict -- which is the shape the call machinery already packs for a
   `def f(self, *rest, **kw)`, and the body reads the three slots. */
APY_API apy_value apy_kind_method_var(apy_value obj, apy_value namev,
                                     int64_t bind) {
    apy_value fn = apy_native(APY_NAT_KIND, 3, (const char *)namev);
    O(fn)->v.fn.vararg = 1;
    O(fn)->v.fn.kwarg = 1;
    return bind ? apy_bind(fn, obj) : fn;
}

/* An EMPTY VALUE of the kind a builtin type names, so the table above can
   answer for the type without a second copy of it. Nothing is done with the
   prototype but ask its kind, and the natives it yields are unbound. */
/* A CURSOR PROTOTYPE IS A CURSOR WITH NO SOURCE, which is all
   `apy_kind_attr_of` reads of one: the kind it is, the mode it walks in and
   what it is named after. Nothing here is ever stepped -- a prototype exists
   to be asked which attributes its kind carries and is then thrown away --
   so there is nothing for a source to be.

   THE PAIRS ARE THE INVERSE of `apy_cursor_name`, which turns a mode and a
   `named` into one of these words. `str_iterator` and `str_ascii_iterator`
   are one row's worth of behaviour under two names, because the width of
   the string decides the name and neither changes what the type carries. */
static const struct { const char *name; int mode, named; }
apy_cursor_protos[] = {
    {"list_iterator",             APY_IT_PLAIN, APY_LIST_K},
    {"list_reverseiterator",      APY_IT_REV,   APY_LIST_K + APY_IT_REVOF},
    {"tuple_iterator",            APY_IT_PLAIN, APY_TUPLE_K},
    {"reversed",                  APY_IT_REV,   APY_TUPLE_K + APY_IT_REVOF},
    {"str_iterator",              APY_IT_PLAIN, APY_STR_K},
    {"str_ascii_iterator",        APY_IT_PLAIN, APY_STR_K},
    {"bytes_iterator",            APY_IT_PLAIN, APY_BYTES_K},
    {"bytearray_iterator",        APY_IT_PLAIN, APY_BYTES_K},
    {"range_iterator",            APY_IT_PLAIN, APY_RANGE_K},
    {"set_iterator",              APY_IT_PLAIN, APY_SET_K},
    {"memory_iterator",           APY_IT_PLAIN, APY_MVIEW_K},
    {"dict_keyiterator",          APY_IT_PLAIN, APY_DICT_K},
    {"dict_valueiterator",        APY_IT_PLAIN,
     APY_IT_VIEWED + APY_PART_VALUES},
    {"dict_itemiterator",         APY_IT_PLAIN,
     APY_IT_VIEWED + APY_PART_ITEMS},
    {"dict_reversekeyiterator",   APY_IT_REV,   APY_DICT_K + APY_IT_REVOF},
    {"dict_reversevalueiterator", APY_IT_REV,
     APY_IT_VIEWED + APY_PART_VALUES + APY_IT_REVOF},
    {"dict_reverseitemiterator",  APY_IT_REV,
     APY_IT_VIEWED + APY_PART_ITEMS + APY_IT_REVOF},
    {"callable_iterator",         APY_IT_CALL,  APY_IT_CALLABLE},
    {"enumerate",                 APY_IT_ENUMERATE, APY_LIST_K},
    {"zip",                       APY_IT_ZIP,   APY_LIST_K},
    {"map",                       APY_IT_MAP,   APY_LIST_K},
    {"filter",                    APY_IT_FILTER, APY_LIST_K},
};

APY_API apy_value apy_kind_prototype(apy_value type_namev) {
    const char *type_name = (const char *)type_namev;
    size_t ci;
    for (ci = 0; ci < sizeof apy_cursor_protos / sizeof *apy_cursor_protos;
         ci++)
        if (strcmp(apy_cursor_protos[ci].name, type_name) == 0) {
            apy_value it = apy_cursor_of(0, 0,
                                         apy_cursor_protos[ci].mode, 0);
            if (it) O(it)->v.it.named = apy_cursor_protos[ci].named;
            return it;
        }
    /* A GENERATOR PROTOTYPE IS A GENERATOR WITH NO STEP, for the reason a
       cursor prototype has no source: nothing is ever run, and what the type
       carries is all that is asked of it. */
    if (strcmp(type_name, "generator") == 0) return apy_gen_new(0, 0);
    if (strcmp(type_name, "coroutine") == 0)
        return apy_coro_mark(apy_gen_new(0, 0));
    if (strcmp(type_name, "async_generator") == 0)
        return apy_agen_mark(apy_gen_new(0, 0));
    if (strcmp(type_name, "list") == 0)  return apy_list_new(1);
    if (strcmp(type_name, "tuple") == 0) return apy_tuple_new(1);
    if (strcmp(type_name, "dict") == 0)  return apy_dict_new(1);
    if (strcmp(type_name, "set") == 0)   return apy_set_new(1);
    if (strcmp(type_name, "frozenset") == 0) return apy_frozenset_new(1);
    if (strcmp(type_name, "str") == 0)   return apy_lit("");
    if (strcmp(type_name, "bytes") == 0) return apy_bytes_copy("", 0);
    if (strcmp(type_name, "int") == 0 || strcmp(type_name, "bool") == 0)
        return apy_from_int(0);
    if (strcmp(type_name, "float") == 0) return apy_from_float(0.0);
    return 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */

/* The arity of a dunder EVERY object has, or 0 for anything else.

   WHAT `object` CARRIES, and the reason this is one table rather than a test
   per kind: `hasattr(x, "__eq__")` is True for every value in Python, a list
   and an int and a function alike, so the answer cannot depend on what `x`
   is. Comparison is the part programs actually read -- `functools`'
   `total_ordering` asks a class which orderings it already has, and `abc`
   asks the same question structurally.

   THE ORDERINGS ARE HERE EVEN THOUGH MOST KINDS REFUSE THEM, because that is
   what CPython does: `{}.__lt__({})` answers `NotImplemented` rather than
   raising, and it is the operator above it that turns that into the
   TypeError a program sees. */
APY_API int64_t apy_object_arity(apy_value wantv) {
    const char *want = (const char *)wantv;
    if (strcmp(want, "__eq__") == 0) return 2;
    if (strcmp(want, "__ne__") == 0) return 2;
    if (strcmp(want, "__lt__") == 0) return 2;
    if (strcmp(want, "__le__") == 0) return 2;
    if (strcmp(want, "__gt__") == 0) return 2;
    if (strcmp(want, "__ge__") == 0) return 2;
    if (strcmp(want, "__str__") == 0) return 1;
    if (strcmp(want, "__repr__") == 0) return 1;
    if (strcmp(want, "__format__") == 0) return 2;
    /* AND THE TWELVE `object` HANDS DOWN THAT NOTHING OVERRIDES. Every one
       was missing from every builtin value, which is 144 attributes a
       program can ask for and Python guarantees: `__reduce_ex__` is how
       pickle finds a value, `__dir__` is what `dir()` reads, and
       `__init__` and `__new__` are on everything there is. */
    if (strcmp(want, "__init__") == 0) return 1;
    if (strcmp(want, "__new__") == 0) return 2;
    if (strcmp(want, "__getattribute__") == 0) return 2;
    if (strcmp(want, "__setattr__") == 0) return 3;
    if (strcmp(want, "__delattr__") == 0) return 2;
    if (strcmp(want, "__init_subclass__") == 0) return 1;
    if (strcmp(want, "__subclasshook__") == 0) return 2;
    if (strcmp(want, "__dir__") == 0) return 1;
    if (strcmp(want, "__sizeof__") == 0) return 1;
    if (strcmp(want, "__reduce__") == 0) return 1;
    if (strcmp(want, "__reduce_ex__") == 0) return 2;
    if (strcmp(want, "__getstate__") == 0) return 1;
    return 0;
}

/* The arity of a NUMBER's dunder, or 0 where that kind has none.

   THREE OVERLAPPING SETS, and the overlaps are what make this one function.
   Every number adds, multiplies, divides, negates and has a truth; only an
   int and a float floor-divide, take a remainder and convert between the
   two; only an int has bits. A complex has neither of the last two groups --
   `(1j).__floordiv__` is an AttributeError in Python rather than a method
   that refuses -- which is the distinction a `numbers` registration reads.

   THE REFLECTED HALVES ARE REAL METHODS. `(1).__radd__(2)` is 3, and a class
   implementing a numeric tower calls them by name. */
APY_API int64_t apy_number_arity(apy_value wantv, int64_t is_int,
                                 int64_t is_complex) {
    const char *want = (const char *)wantv;
    if (strcmp(want, "__add__") == 0) return 2;
    if (strcmp(want, "__radd__") == 0) return 2;
    if (strcmp(want, "__sub__") == 0) return 2;
    if (strcmp(want, "__rsub__") == 0) return 2;
    if (strcmp(want, "__mul__") == 0) return 2;
    if (strcmp(want, "__rmul__") == 0) return 2;
    if (strcmp(want, "__truediv__") == 0) return 2;
    if (strcmp(want, "__rtruediv__") == 0) return 2;
    if (strcmp(want, "__pow__") == 0) return 2;
    if (strcmp(want, "__rpow__") == 0) return 2;
    if (strcmp(want, "__neg__") == 0) return 1;
    if (strcmp(want, "__pos__") == 0) return 1;
    if (strcmp(want, "__abs__") == 0) return 1;
    if (strcmp(want, "__bool__") == 0) return 1;
    if (is_complex) {
        if (strcmp(want, "__complex__") == 0) return 1;
        return 0;
    }
    /* AN INT AND A FLOAT, BOTH: the whole-number operations and the two
       conversions between them. A complex has left above. */
    if (strcmp(want, "__floordiv__") == 0) return 2;
    if (strcmp(want, "__rfloordiv__") == 0) return 2;
    if (strcmp(want, "__mod__") == 0) return 2;
    if (strcmp(want, "__rmod__") == 0) return 2;
    if (strcmp(want, "__divmod__") == 0) return 2;
    if (strcmp(want, "__rdivmod__") == 0) return 2;
    if (strcmp(want, "__int__") == 0) return 1;
    if (strcmp(want, "__float__") == 0) return 1;
    if (strcmp(want, "__trunc__") == 0) return 1;
    if (strcmp(want, "__floor__") == 0) return 1;
    if (strcmp(want, "__ceil__") == 0) return 1;
    /* `__round__` IS NOT HERE because it is not a fixed arity: it takes an
       OPTIONAL `ndigits`, which this table has no way to say. It is answered
       in `apy_kind_attr_of` through `apy_kind_method_opt` instead -- the
       entry point that exists for exactly this. */
    if (!is_int) return 0;
    /* AN INT'S OWN: the bit operations and `__index__`, which is the promise
       that this value may stand where a position is wanted. */
    if (strcmp(want, "__index__") == 0) return 1;
    if (strcmp(want, "__invert__") == 0) return 1;
    if (strcmp(want, "__and__") == 0) return 2;
    if (strcmp(want, "__rand__") == 0) return 2;
    if (strcmp(want, "__or__") == 0) return 2;
    if (strcmp(want, "__ror__") == 0) return 2;
    if (strcmp(want, "__xor__") == 0) return 2;
    if (strcmp(want, "__rxor__") == 0) return 2;
    if (strcmp(want, "__lshift__") == 0) return 2;
    if (strcmp(want, "__rlshift__") == 0) return 2;
    if (strcmp(want, "__rshift__") == 0) return 2;
    if (strcmp(want, "__rrshift__") == 0) return 2;
    return 0;
}

/* @KINDMETH_TABLE@ */

/* The bit the generated table wants for this receiver. One place that knows
   the pairing, so the enum and the table cannot drift apart silently. */
static unsigned apy_kind_bit(apy_value v) {
    switch (O(v)->kind) {
    case APY_STR_K:    return APY_MK_STR;
    /* A VIEW IS IN THE PAIRING FOR THE WRITTEN TABLE ALONE -- which is about
       what `type()` calls a bound method -- and has no row in the arity one:
       its methods are hand-written above, because several of them mean
       something a bytes receiver would answer differently for. */
    case APY_MVIEW_K:  return APY_MK_MEMORYVIEW;
    case APY_BYTES_K:  return O(v)->v.s.mut ? APY_MK_BYTEARRAY : APY_MK_BYTES;
    case APY_LIST_K:   return APY_MK_LIST;
    case APY_TUPLE_K:  return APY_MK_TUPLE;
    case APY_DICT_K:   return APY_MK_DICT;
    case APY_SET_K:    return APY_MK_SET;
    case APY_FROZEN_K: return APY_MK_FROZENSET;
    case APY_INT_K:
    case APY_BIG_K:
    case APY_BOOL_K:   return APY_MK_INT;
    case APY_FLOAT_K:  return APY_MK_FLOAT;
    case APY_RANGE_K:  return APY_MK_RANGE;
    case APY_COMPLEX_K: return APY_MK_COMPLEX;
    default: return 0;
    }
}

/* WHAT CPYTHON SAYS about the wrong number of arguments to a builtin
   method, worded for THIS receiver.

   NINE SHAPES and they disagree in every visible way: `list.append() takes
   exactly one argument (2 given)` qualifies the name and parenthesises it,
   `count expected at most 3 arguments, got 4` does neither, and which one a
   name uses can depend on the RECEIVER -- `list.count` and `str.count` word
   it differently. The shape and the counts come out of `apy_kind_meth_words`,
   which is generated by asking CPython; this only spells them.

   Raises and answers 0, so a caller can `return` it. */
static apy_value apy_meth_arity_words(apy_value recv, const char *w,
                                      int64_t packed, int64_t got) {
    char buf[192];
    int64_t least = packed & 15;
    int64_t fam, n;
    const char *s;
    /* WHICH END WAS MISSED decides the wording, not which is nearer: a
       method with a range says `at least` below it and `at most` above. */
    if (got < least) {
        fam = (packed >> 8) & 15;
        n = (packed >> 12) & 15;
    } else {
        int64_t over = (packed >> 25) & 15;
        fam = (packed >> 16) & 15;
        n = (packed >> 20) & 15;
        /* A SECOND UPPER WORDING, past a second bound. `(5).to_bytes` says
           `takes at most 2 positional arguments` for a third POSITIONAL and
           `takes at most 3 arguments` for a fourth argument -- its `signed`
           is keyword-only, so the two counts differ and so do the words. */
        if (over && got >= over) {
            fam = (packed >> 29) & 15;
            n = (packed >> 33) & 15;
        }
    }
    s = n == 1 ? "" : "s";
    switch (fam) {
    case 1:
        snprintf(buf, sizeof buf,
                 "%s.%s() takes exactly one argument (%lld given)",
                 apy_kind_name(recv), w, (long long)got);
        break;
    case 2:
        snprintf(buf, sizeof buf, "%s.%s() takes no arguments (%lld given)",
                 apy_kind_name(recv), w, (long long)got);
        break;
    case 3:
        snprintf(buf, sizeof buf,
                 "%s() takes at most %lld argument%s (%lld given)",
                 w, (long long)n, s, (long long)got);
        break;
    case 4:
        snprintf(buf, sizeof buf,
                 "%s() takes at most %lld positional argument%s (%lld given)",
                 w, (long long)n, s, (long long)got);
        break;
    case 5:
        snprintf(buf, sizeof buf,
                 "%s() takes at least %lld positional argument%s (%lld given)",
                 w, (long long)n, s, (long long)got);
        break;
    case 6:
        snprintf(buf, sizeof buf,
                 "%s expected at least %lld argument%s, got %lld",
                 w, (long long)n, s, (long long)got);
        break;
    case 7:
        snprintf(buf, sizeof buf,
                 "%s expected at most %lld argument%s, got %lld",
                 w, (long long)n, s, (long long)got);
        break;
    case 9:
        /* NO COUNT AT ALL, which is what a signature with nothing but
           keyword-only parameters says: `[].sort(None)`. */
        snprintf(buf, sizeof buf, "%s() takes no positional arguments", w);
        break;
    default:
        snprintf(buf, sizeof buf, "%s expected %lld argument%s, got %lld",
                 w, (long long)n, s, (long long)got);
        break;
    }
    return apy_fail("TypeError", buf);
}

/* Refuse this many POSITIONAL arguments to a bound builtin method, for the
   two whose positional bound is narrower than their total: `to_bytes` takes
   at most two positionals and three arguments, `sort` none and two, and the
   difference is a keyword-only parameter apiece.

   WHERE THE POSITIONAL COUNT IS STILL KNOWN, which is only here: the caller
   folds keywords into slots before the body sees them, so a call three
   slots wide may have been written with three positionals -- a TypeError --
   or with two names -- not one. `bound` is what says which.

   Answers 0 without raising for every other method, which is every row
   whose upper wording is about ARGUMENTS rather than positions. */
static apy_value apy_meth_positional(apy_value f, int64_t argc) {
    apy_value recv = O(f)->v.fn.bound;
    const char *w;
    int64_t said, family, allowed;
    if (!recv) return 0;
    w = APY_CSTR(O(f)->v.fn.name);
    said = apy_kind_meth_words(w, apy_kind_bit(recv));
    if (!said) return 0;
    family = (said >> 16) & 15;
    /* FAMILY 4 IS `takes at most N positional arguments` and family 9 is
       `takes no positional arguments`, whose N is nought. Nothing else in
       the table counts positions. */
    if (family == 4) allowed = (said >> 20) & 15;
    else if (family == 9) allowed = 0;
    else return 0;
    if (argc <= allowed) return 0;
    return apy_meth_arity_words(recv, w, said, argc);
}

/* Refuse `got` arguments to the builtin method `namev` on `recv` the way
   CPython does -- or answer None, meaning the count is one this receiver
   accepts and the call should go ahead.

   TWO CALLERS, ONE ANSWER. The lowering emits this ahead of a written call
   whose argument count SOME kind refuses, and `apy_arity_error` asks it for
   a bound native. Both reported the wrong thing before: `set().pop(1)` was
   `'set' object has no attribute 'pop'` about a method a set plainly has,
   and `{}.pop()` was `KeyError: None` from a symbol that took a call it
   should have refused.

   A NAME THIS KIND DOES NOT HAVE ANSWERS NONE TOO, because that really is an
   AttributeError and the caller's own path already words it. */
APY_API apy_value apy_meth_arity(apy_value recv, apy_value namev,
                                 int64_t got) {
    const char *w = APY_CSTR(namev);
    int64_t packed = apy_kind_meth_words(w, apy_kind_bit(recv));
    int64_t least, most;
    if (!packed) return apy_none();
    least = packed & 15;
    most = (packed >> 4) & 15;
    if (got >= least && got <= most) return apy_none();
    return apy_meth_arity_words(recv, w, packed, got);
}

/* THE SIX SET METHODS THAT TAKE ANY NUMBER OF OTHERS.

   `s.union()` is a copy, `s.union(a)` is one union and `s.union(a, b)` folds
   both in -- and no row in an arity table can say "any count", so these were
   declared as taking exactly one and refused every other call CPython
   answers. They are variadic natives now (see `apy_kind_method_var`) and are
   folded HERE, above the generated table, which would hand the whole `*rest`
   tuple to a two-argument symbol as though it were one other set. */
static int apy_set_folds(const char *w) {
    return strcmp(w, "union") == 0 || strcmp(w, "intersection") == 0
           || strcmp(w, "difference") == 0 || strcmp(w, "update") == 0
           || strcmp(w, "intersection_update") == 0
           || strcmp(w, "difference_update") == 0;
}

static apy_value apy_set_fold(const char *w, apy_value *a, int64_t n) {
    apy_value acc = a[0];
    apy_value rest = n > 1 ? a[1] : 0;
    apy_value kw = n > 2 ? a[2] : 0;
    apy_value *items = rest ? (apy_value *)O(rest)->v.q.items : 0;
    int64_t count = rest ? O(rest)->v.q.n : 0;
    int64_t i;
    /* NONE OF THE SIX TAKES A KEYWORD, and a variadic native declares the
       `**kw` slot whether or not the body wants one -- so a keyword would
       have been collected and dropped in silence. */
    if (kw && O(kw)->kind == APY_DICT_K && O(kw)->v.d.n) {
        /* NAMED BY ITS OWNER, which is the receiver's kind: CPython says
           `set.union() takes no keyword arguments`. */
        char buf[96];
        snprintf(buf, sizeof buf, "%s.%s() takes no keyword arguments",
                 apy_kind_name(acc), w);
        return apy_fail("TypeError", buf);
    }
    if (strcmp(w, "update") == 0 || strcmp(w, "intersection_update") == 0
            || strcmp(w, "difference_update") == 0) {
        /* IN PLACE, AND None COMES BACK. Nothing to do for no others at
           all, which is what `s.update()` is. */
        for (i = 0; i < count; i++) {
            apy_value step = strcmp(w, "update") == 0
                ? apy_update(acc, items[i])
                : strcmp(w, "intersection_update") == 0
                  ? apy_set_inter_update(acc, items[i])
                  : apy_set_diff_update(acc, items[i]);
            if (!step) return 0;
        }
        return apy_none();
    }
    /* A COPY IS WHAT NO OTHERS AT ALL ANSWERS -- `s.union()` is a new set
       equal to `s`, not `s` itself. With one or more, each fold already
       makes a new one. */
    if (!count) return apy_copy(acc);
    for (i = 0; i < count; i++) {
        acc = strcmp(w, "union") == 0
            ? apy_set_union(acc, items[i])
            : strcmp(w, "intersection") == 0
              ? apy_set_intersection(acc, items[i])
              : apy_set_difference(acc, items[i]);
        if (!acc) return 0;
    }
    return acc;
}

/* The native for a builtin method whose real bounds the generated words
   table knows -- or 0 for a name it does not have, and for the six set
   methods that take ANY number of others.

   THE BOUNDS ARE THIS KIND'S. The arity table holds one row per NAME:
   `str.count` takes three arguments and `list.count` takes one, and the
   union of the two said three for both -- so a method reached BY NAME
   accepted counts its receiver refuses and refused counts its receiver
   accepts. `getattr("ab", "count")("a", 0, 2)` was `count expected 1
   argument, got 3` for a call CPython answers, and `getattr({}, "pop")()`
   reached a symbol that answered `KeyError` for one CPython refuses. */
static apy_value apy_kind_method_ranged(apy_value obj, const char *want,
                                        int64_t bind) {
    int64_t said = apy_kind_meth_words(want, apy_kind_bit(obj));
    int64_t least, most;
    if (!said) return 0;
    most = (said >> 4) & 15;
    /* NO UPPER END AT ALL is the six set methods, and a declared arity
       cannot say so: they take whatever they are given. See `apy_set_fold`,
       which is what the call reaches. */
    if (most == 15)
        return apy_kind_method_var(obj, (apy_value)(uintptr_t)want, bind);
    least = said & 15;
    /* A SECOND BOUND MEANS THE DECLARED RANGE IS THE WIDER ONE. `to_bytes`
       takes at most TWO positionals and THREE arguments, and `sort` takes
       none and two, because each has a keyword-only parameter -- and once
       the keywords are folded into slots the call arrives at its full
       width, so a cell declaring the narrow bound refused
       `(258).to_bytes(**{"length": 4, "byteorder": "little"})`. The
       POSITIONAL bound is checked where the positional count is still
       known; see `apy_meth_positional`. */
    if (((said >> 25) & 15))
        most = ((said >> 25) & 15) - 1;
    return apy_kind_method_opt(obj, most + 1, most - least,
                               (apy_value)(uintptr_t)want, bind);
}

APY_API apy_value apy_kind_attr_of(apy_value obj, apy_value wantv,
                                  int64_t bind) {
    const char *want = (const char *)wantv;
    int k = O(obj)->kind;
    int seq = apy_is_seq(obj), set = apy_is_set(obj);
    int text = k == APY_STR_K || k == APY_BYTES_K;
    int dict = k == APY_DICT_K;
    int walks = seq || set || text || dict || k == APY_MVIEW_K
                || k == APY_VIEW_K;
    int mutable_ = k == APY_LIST_K || k == APY_DICT_K || k == APY_SET_K
                   || (k == APY_BYTES_K && O(obj)->v.s.mut);

    if (strcmp(want, "__hash__") == 0) {
        /* THE ATTRIBUTE EXISTS EITHER WAY. `[].__hash__ is None` is how a
           program asks whether a list can be a dict key, and answering "no
           such attribute" is a different claim from the one CPython makes. */
        if (mutable_) return apy_none();
        return apy_kind_method(obj, 1, "__hash__", bind);
    }
    /* `x.__class__` IS `type(x)`, for every value there is -- and it was
       missing from all of them. `obj.__class__.__name__` is an everyday
       idiom, and a program reaching for it got an AttributeError about the
       one attribute Python guarantees. NOT A METHOD but the type object
       itself, which is why it answers before `apy_kind_method` is reached;
       `apy_type_for` interns per kind, so `x.__class__ is type(x)` holds. */
    if (strcmp(want, "__class__") == 0) return apy_type_of(obj);
    /* `object` GIVES THESE TO EVERYTHING, which is why they are gated on no
       kind at all: `hasattr(x, "__eq__")` is True for every value in Python,
       a list and an int and a function alike. Comparison is the part
       programs read rather than call -- `functools.total_ordering` asks a
       class which orderings it already has, and `abc` asks structurally. */
    {
        int64_t common = apy_object_arity((apy_value)(uintptr_t)want);
        if (common)
            return apy_kind_method(obj, common, want, bind);
    }
    if (strcmp(want, "__len__") == 0 && walks)
        return apy_kind_method(obj, 1, "__len__", bind);
    /* ONLY THE CURSORS THAT WALK A SIZED SOURCE have a length hint, which is
       CPython's line: a list, tuple, str, bytes, range, dict, set or
       reversed iterator carries one, and `map`, `filter`, `enumerate`, `zip`
       and a `callable_iterator` do not. Those are the LAZY modes, and what
       they have left is not a question their source can answer. */
    /* AND A `memory_iterator` HAS NONE, which is the one sized walk without
       one: `iter(mv)` in CPython carries `__iter__` and `__next__` and
       nothing else, where `reversed(mv)` -- a plain `reversed` -- does have
       the hint. The generated `dir()` row says the same, and a name this
       answered that the row omits would be the lying list in reverse. */
    if (strcmp(want, "__length_hint__") == 0 && k == APY_ITER_K
            && (O(obj)->v.it.mode == APY_IT_PLAIN
                || O(obj)->v.it.mode == APY_IT_REV)
            && !(O(obj)->v.it.mode == APY_IT_PLAIN
                 && O(obj)->v.it.named == APY_MVIEW_K))
        return apy_kind_method(obj, 1, "__length_hint__", bind);
    if (strcmp(want, "__iter__") == 0
            && (walks || k == APY_GEN_K || k == APY_ITER_K))
        return apy_kind_method(obj, 1, "__iter__", bind);
    if (strcmp(want, "__next__") == 0
            && (k == APY_GEN_K || k == APY_ITER_K))
        return apy_kind_method(obj, 1, "__next__", bind);
    if (strcmp(want, "__contains__") == 0 && walks)
        return apy_kind_method(obj, 2, "__contains__", bind);
    if (strcmp(want, "__getitem__") == 0
            && (seq || text || dict || k == APY_MVIEW_K))
        return apy_kind_method(obj, 2, "__getitem__", bind);
    if (strcmp(want, "__setitem__") == 0
            && (k == APY_LIST_K || dict
                || (k == APY_BYTES_K && O(obj)->v.s.mut)))
        return apy_kind_method(obj, 3, "__setitem__", bind);
    /* WHATEVER `del x[k]` WOULD REACH -- exactly the three kinds that take a
       `__setitem__`, because a container that cannot be written cannot have
       a piece taken out of it either. */
    if (strcmp(want, "__delitem__") == 0
            && (k == APY_LIST_K || dict
                || (k == APY_BYTES_K && O(obj)->v.s.mut)))
        return apy_kind_method(obj, 2, "__delitem__", bind);
    /* `reversed(x)` WALKS ANYTHING INDEXABLE, but only three kinds carry the
       method that names it: a str or a tuple is reversed through `__len__`
       and `__getitem__`, and CPython gives neither a `__reversed__`. */
    if (strcmp(want, "__reversed__") == 0
            && (k == APY_LIST_K || dict || k == APY_RANGE_K))
        return apy_kind_method(obj, 1, "__reversed__", bind);
    /* CONCATENATION AND REPETITION, which a sequence has and a set, a dict
       and a range do not -- `range(3) * 2` is a TypeError in Python and the
       attribute is absent, not a method that refuses. */
    if ((seq || text) && (strcmp(want, "__add__") == 0
                          || strcmp(want, "__mul__") == 0
                          || strcmp(want, "__rmul__") == 0))
        return apy_kind_method(obj, 2, want, bind);
    if (apy_is_num(obj) || k == APY_COMPLEX_K) {
        int64_t arity;
        /* `__round__` TAKES AN OPTIONAL `ndigits`, which the arity table
           cannot say -- `(1.55).__round__()` and `(1.55).__round__(1)` are
           one method. A COMPLEX HAS NONE: there is no rounding of one. */
        if (k != APY_COMPLEX_K && strcmp(want, "__round__") == 0)
            return apy_kind_method_opt(obj, 2, 1,
                                       (apy_value)(uintptr_t)want, bind);
        arity = apy_number_arity(
            (apy_value)(uintptr_t)want, apy_is_int_like(obj),
            k == APY_COMPLEX_K);
        if (arity) return apy_kind_method(obj, arity, want, bind);
    }
    /* A RANGE IS FALSE WHEN IT IS EMPTY and says so with `__bool__` rather
       than through `__len__`, which is the one place it parts company with
       the other walkable kinds. */
    /* AND None SAYS SO TOO, which is the only other kind here that writes
       the method out rather than being read through `__len__`. */
    if ((k == APY_RANGE_K || k == APY_NONE_K)
            && strcmp(want, "__bool__") == 0)
        return apy_kind_method(obj, 1, "__bool__", bind);
    /* THE IN-PLACE OPERATORS, which belong to the MUTABLE kinds and to no
       other: `(1,).__iadd__` is an AttributeError in Python and `[1].__iadd__`
       is the method that makes `xs += ys` change the list every other name
       for it also sees. A tuple falls through to `+` and has nothing to
       name. A FROZENSET HAS NONE of them, which is why the set arm asks for
       APY_SET_K rather than `set`. */
    if (k == APY_LIST_K || (k == APY_BYTES_K && O(obj)->v.s.mut)) {
        if (strcmp(want, "__iadd__") == 0)
            return apy_kind_method(obj, 2, "__iadd__", bind);
        if (strcmp(want, "__imul__") == 0)
            return apy_kind_method(obj, 2, "__imul__", bind);
    }
    if ((dict || k == APY_SET_K) && strcmp(want, "__ior__") == 0)
        return apy_kind_method(obj, 2, "__ior__", bind);
    if (k == APY_SET_K) {
        if (strcmp(want, "__iand__") == 0)
            return apy_kind_method(obj, 2, "__iand__", bind);
        if (strcmp(want, "__isub__") == 0)
            return apy_kind_method(obj, 2, "__isub__", bind);
        if (strcmp(want, "__ixor__") == 0)
            return apy_kind_method(obj, 2, "__ixor__", bind);
    }
    /* `%` ON TEXT IS FORMATTING, not arithmetic -- which is why it belongs
       to str and bytes and to no other sequence. */
    if (text && strcmp(want, "__mod__") == 0)
        return apy_kind_method(obj, 2, "__mod__", bind);
    /* `d | e` MERGES TWO DICTS, and the same four spellings are a set's
       operations. A `collections.abc` mixin composes them by name, which is
       the reading that needed them to exist as values. */
    if (dict && (strcmp(want, "__or__") == 0
                 || strcmp(want, "__ror__") == 0))
        return apy_kind_method(obj, 2, want, bind);
    if (set && (strcmp(want, "__or__") == 0 || strcmp(want, "__and__") == 0
                || strcmp(want, "__sub__") == 0
                || strcmp(want, "__xor__") == 0
                || strcmp(want, "__ror__") == 0
                || strcmp(want, "__rand__") == 0
                || strcmp(want, "__rsub__") == 0
                || strcmp(want, "__rxor__") == 0))
        return apy_kind_method(obj, 2, want, bind);
    if (dict && (strcmp(want, "keys") == 0 || strcmp(want, "values") == 0
                 || strcmp(want, "items") == 0))
        return apy_kind_method(obj, 1, want, bind);
    if ((seq || text) && (strcmp(want, "index") == 0
                          || strcmp(want, "count") == 0)) {
        /* A WINDOW, ON THE KINDS THAT TAKE ONE. `"ab".count("a", 0, 2)` is
           a call CPython answers and `[1].count(1, 2)` is not, so the
           bounds come from the table rather than from this line. */
        apy_value ranged = apy_kind_method_ranged(obj, want, bind);
        if (ranged) return ranged;
        return apy_kind_method(obj, 2, want, bind);
    }
    if (k == APY_LIST_K && strcmp(want, "append") == 0)
        return apy_kind_method(obj, 2, "append", bind);
    if (k == APY_LIST_K && strcmp(want, "insert") == 0)
        return apy_kind_method(obj, 3, "insert", bind);
    if (k == APY_SET_K && (strcmp(want, "add") == 0
                           || strcmp(want, "discard") == 0))
        return apy_kind_method(obj, 2, want, bind);
    if (set && strcmp(want, "isdisjoint") == 0)
        return apy_kind_method(obj, 2, "isdisjoint", bind);
    /* PEP 688: whatever can be handed to `memoryview` HAS `__buffer__`. It is
       a protocol a program asks about far more often than it calls, and
       answering False for `bytes` said this runtime has no buffers at all. */
    /* THE STATICS A TYPE CARRIES AND A VALUE CARRIES TOO. `bytes.fromhex`
       and `b"".fromhex` are ONE method in Python -- an implicit
       staticmethod, so the receiver decides which body and is otherwise
       ignored -- and every one of these was reachable only as a written call
       on the type's own name. */
    /* `format` TAKES WHATEVER IT IS GIVEN, positionally and by keyword,
       which is why it is not in the method table: no row there can say "any
       count". The WRITTEN form is lowered at the call site; this is the same
       call reached by name. */
    if (strcmp(want, "format") == 0 && k == APY_STR_K)
        return apy_kind_method_var(obj, (apy_value)(uintptr_t)want, bind);
    if (strcmp(want, "maketrans") == 0) {
        if (k == APY_STR_K) return apy_kind_method_opt(
            obj, 4, 2, (apy_value)(uintptr_t)want, bind);
        if (k == APY_BYTES_K) return apy_kind_method(obj, 3, want, bind);
    }
    if (strcmp(want, "fromhex") == 0
            && (k == APY_BYTES_K || k == APY_FLOAT_K))
        return apy_kind_method(obj, 2, want, bind);
    if (strcmp(want, "from_number") == 0
            && (k == APY_FLOAT_K || k == APY_COMPLEX_K))
        return apy_kind_method(obj, 2, want, bind);
    if (strcmp(want, "fromkeys") == 0 && dict)
        return apy_kind_method_opt(obj, 3, 1, (apy_value)(uintptr_t)want,
                                   bind);
    if (strcmp(want, "from_bytes") == 0
            && (k == APY_INT_K || k == APY_BOOL_K || k == APY_BIG_K))
        return apy_kind_method_opt(obj, 3, 1, (apy_value)(uintptr_t)want,
                                   bind);
    if (strcmp(want, "__buffer__") == 0
            && (k == APY_BYTES_K || k == APY_MVIEW_K))
        return apy_kind_method(obj, 2, "__buffer__", bind);
    /* PEP 688's OTHER HALF, and the one bytearray internal a program can
       read. `__release_buffer__` is what a `with memoryview(...)` block
       calls on the way out, and only the buffer that can be RESIZED carries
       it -- bytes cannot move under a view and has nothing to be told. */
    if (k == APY_BYTES_K && O(obj)->v.s.mut) {
        if (strcmp(want, "__release_buffer__") == 0)
            return apy_kind_method(obj, 2, want, bind);
        if (strcmp(want, "__alloc__") == 0)
            return apy_kind_method(obj, 1, want, bind);
    }
    /* WHAT `copy` AND `pickle` REBUILD A VALUE FROM. Every immutable builtin
       answers `(self,)` -- a complex answers its two halves -- and a mutable
       one has none at all, because anything it handed back would be SHARED
       with the copy rather than rebuild it. */
    if (strcmp(want, "__getnewargs__") == 0
            && (k == APY_STR_K || k == APY_TUPLE_K || k == APY_INT_K
                || k == APY_BIG_K || k == APY_BOOL_K || k == APY_FLOAT_K
                || k == APY_COMPLEX_K
                || (k == APY_BYTES_K && !O(obj)->v.s.mut)))
        return apy_kind_method(obj, 1, want, bind);
    /* `bytes(x)` ASKS `x` FOR ITSELF FIRST, and bytes is the kind that
       answers -- a bytearray does not, which is why `bytes(ba)` copies. */
    if (strcmp(want, "__bytes__") == 0
            && k == APY_BYTES_K && !O(obj)->v.s.mut)
        return apy_kind_method(obj, 1, want, bind);
    /* `list[int]` REACHED BY NAME. The written form is a subscript the
       frontend lowers; this is the method behind it, which `typing` calls
       directly when it parameterises a container. TEXT HAS NONE: `str[int]`
       is a TypeError in Python and the attribute is absent. */
    if (strcmp(want, "__class_getitem__") == 0 && (seq || dict || set))
        return apy_kind_method(obj, 2, want, bind);
    /* AND `enumerate[int]`, which is the one CURSOR CPython gives one to:
       `zip`, `map` and `filter` have none, and `dir()` over each says so. */
    if (strcmp(want, "__class_getitem__") == 0
            && k == APY_ITER_K && O(obj)->v.it.mode == APY_IT_ENUMERATE)
        return apy_kind_method(obj, 2, want, bind);
    /* AND `generator[int]`, which a generic annotation on an `async def` or
       a `Generator[...]`-shaped alias reaches by name. */
    if (strcmp(want, "__class_getitem__") == 0 && k == APY_GEN_K)
        return apy_kind_method(obj, 2, want, bind);
    /* A GENERATOR HAS A FINALISER and is the only builtin value here that
       does: closing an abandoned one runs its `finally` blocks, which is why
       CPython gives the type a `__del__` where a list has none. Answerable
       rather than useful -- there is nothing for a program to do by calling
       it -- but `dir(g)` lists it and a list that lies is worse than the
       empty one this replaced. */
    if (strcmp(want, "__del__") == 0 && k == APY_GEN_K)
        return apy_kind_method(obj, 1, want, bind);
    /* AND THE PROTOCOL EACH OF THE THREE ANSWERS TO. A coroutine is what
       `await` walks and carries `__await__`; an async generator is what
       `async for` walks and carries the two halves of that protocol; a plain
       generator has neither, which is what `dir()` over each says. */
    if (k == APY_GEN_K && O(obj)->v.g.coro && !O(obj)->v.g.agen
            && strcmp(want, "__await__") == 0)
        return apy_kind_method(obj, 1, want, bind);
    if (k == APY_GEN_K && O(obj)->v.g.agen
            && (strcmp(want, "__aiter__") == 0
                || strcmp(want, "__anext__") == 0))
        return apy_kind_method(obj, 1, want, bind);
    /* WHERE A WALK IS, written. See `apy_cursor_setstate_p` for which
       cursors carry it. */
    if (strcmp(want, "__setstate__") == 0 && apy_cursor_setstate_p(obj))
        return apy_kind_method(obj, 2, want, bind);
    /* WHICH FLOATING-POINT FORMAT THIS BUILD USES. One answer, and a float
       is the only kind ever asked. */
    if (strcmp(want, "__getformat__") == 0 && k == APY_FLOAT_K)
        return apy_kind_method(obj, 2, want, bind);
    /* THE REFLECTED `%`, which text carries and always refuses -- `1 % "a"`
       is a TypeError the OPERATOR raises after this answers NotImplemented.
       A NUMBER'S IS ELSEWHERE: `apy_number_arity` already has it. */
    if (strcmp(want, "__rmod__") == 0 && text)
        return apy_kind_method(obj, 2, want, bind);
    /* THE TWO WAYS A VIEW HANDS ITS CONTENTS OVER, and the reason a program
       makes one at all: `mv.tobytes()` copies them out and `mv.tolist()`
       reads them as numbers. Neither existed, so a view could be indexed and
       sliced and never emptied. */
    if (k == APY_MVIEW_K
            && (strcmp(want, "tobytes") == 0 || strcmp(want, "tolist") == 0))
        return apy_kind_method(obj, 1, want, bind);
    /* WHAT A VIEW CARRIES BESIDE ITS TWO CONVERSIONS. `hex`, `count` and
       `index` read the bytes it shows -- a memoryview IS a sequence in
       Python -- and the three subscript dunders are the methods behind the
       `m[i]` a program writes. `__delitem__` exists and always refuses,
       which is not the same claim as having no such method. */
    if (k == APY_MVIEW_K) {
        if (strcmp(want, "hex") == 0 || strcmp(want, "index") == 0) {
            /* THE SAME BOUNDS A BYTES RECEIVER HAS, because a view's `hex`
               and `index` read the bytes it shows. */
            apy_value ranged = apy_kind_method_ranged(obj, want, bind);
            if (ranged) return ranged;
            return apy_kind_method_opt(obj, 2, 1,
                                       (apy_value)(uintptr_t)want, bind);
        }
        if (strcmp(want, "count") == 0
                || strcmp(want, "__delitem__") == 0
                || strcmp(want, "__class_getitem__") == 0
                || strcmp(want, "__release_buffer__") == 0)
            return apy_kind_method(obj, 2, want, bind);
        if (strcmp(want, "__setitem__") == 0)
            return apy_kind_method(obj, 3, want, bind);
        /* HANDING THE BUFFER BACK, and the `with` block that does it for a
           program. `__exit__` takes the three exception slots whether or
           not there was one, which is why it declares four. */
        if (strcmp(want, "release") == 0 || strcmp(want, "toreadonly") == 0
                || strcmp(want, "__enter__") == 0)
            return apy_kind_method(obj, 1, want, bind);
        if (strcmp(want, "cast") == 0)
            return apy_kind_method(obj, 2, want, bind);
        if (strcmp(want, "_from_flags") == 0)
            return apy_kind_method(obj, 3, want, bind);
        if (strcmp(want, "__exit__") == 0)
            return apy_kind_method(obj, 4, want, bind);
    }
    if (k == APY_RANGE_K) {
        /* THE THREE NUMBERS A RANGE IS, read back. */
        if (strcmp(want, "start") == 0)
            return apy_from_int(O(obj)->v.rg.start);
        if (strcmp(want, "stop") == 0)
            return apy_from_int(O(obj)->v.rg.stop);
        if (strcmp(want, "step") == 0)
            return apy_from_int(O(obj)->v.rg.step);
        if (strcmp(want, "index") == 0 || strcmp(want, "count") == 0)
            return apy_kind_method(obj, 2, want, bind);
        if (strcmp(want, "__len__") == 0)
            return apy_kind_method(obj, 1, "__len__", bind);
        if (strcmp(want, "__iter__") == 0)
            return apy_kind_method(obj, 1, "__iter__", bind);
        if (strcmp(want, "__contains__") == 0)
            return apy_kind_method(obj, 2, "__contains__", bind);
        if (strcmp(want, "__getitem__") == 0)
            return apy_kind_method(obj, 2, "__getitem__", bind);
    }
    {
        /* AND THE WHOLE METHOD TABLE, generated. Everything above answers a
           PROTOCOL name or a field; this answers the ORDINARY methods, which
           existed only as calls the frontend lowered and so could not be
           reached by NAME at all -- `getattr("abc", "upper")` was an
           AttributeError about a method the object plainly has, and anything
           dispatching by method name saw a builtin as having none.

           LAST, so every hand-written arm above still decides first: a few
           names mean different things to different kinds and are split there
           rather than in a table keyed by name. */
        /* THE BOUNDS ARE THIS KIND'S where the words table knows them --
           see `apy_kind_method_ranged`. The arity table decides for a name
           it does not carry, and for the six set methods whose range has no
           upper end at all. */
        int64_t packed;
        apy_value ranged = apy_kind_method_ranged(obj, want, bind);
        if (ranged) return ranged;
        packed = apy_kind_meth_arity(want, apy_kind_bit(obj));
        if (packed)
            return apy_kind_method_opt(obj, packed >> 8, packed & 0xFF,
                                       (apy_value)(uintptr_t)want, bind);
    }
    return 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */

APY_API apy_value apy_kind_attr(apy_value obj, apy_value wantv) {
    const char *want = (const char *)wantv;
    return apy_kind_attr_of(obj, wantv, 1);
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */

/* How a pair of values compares: 0 answers `NotImplemented`, 1 uses this
   runtime's own comparison, 2 uses `object`'s -- identity, or the sentinel.

   CPYTHON'S RICH COMPARISONS ANSWER A SENTINEL RATHER THAN RAISING:
   `(1).__eq__("a")` is NotImplemented, and it is `==` above the method that
   turns a pair of them into False and `<` that turns them into the TypeError
   a program sees. A method answering False directly would break
   `functools.total_ordering`, which reads the sentinel to decide whether to
   try the reflected operation.

   NUMBERS WIDEN ONE WAY AND ONLY ONE. `(1).__eq__(1.0)` is NotImplemented
   and `(1.0).__eq__(1)` is True -- the narrower type declines and the wider
   one answers, which is how Python arranges for exactly one of the two to
   decide. Nothing else here is asymmetric except the same rule for a
   bytearray against bytes.

   `ordering` DROPS WHAT HAS NO `<`: a complex, a dict and a range each
   answer equality and refuse an order. */
static int64_t apy_num_rank(apy_value v) {
    if (apy_is_int_like(v)) return 1;
    return O(v)->kind == APY_FLOAT_K ? 2 : 3;
}

/* A COMPLEX COUNTS HERE AND NOT IN `apy_is_num`, whose callers go on to ask
   for a `double` and a complex has two. What this asks is whether the value
   is a NUMBER, which a complex is. */
static int apy_is_number(apy_value v) {
    return apy_is_num(v) || O(v)->kind == APY_COMPLEX_K;
}

static int apy_compare_how(apy_value a, apy_value b, int ordering) {
    int ka = O(a)->kind, kb = O(b)->kind;
    if (apy_is_number(a)) {
        if (!apy_is_number(b)) return 0;
        if (ordering && ka == APY_COMPLEX_K) return 0;
        return apy_num_rank(a) >= apy_num_rank(b) ? 1 : 0;
    }
    if (apy_is_number(b)) return 0;
    /* ONE KIND, TWO TYPES: `mut` is what tells a bytearray from bytes, and
       a bytearray reaches for bytes where bytes does not reach back. */
    if (ka == APY_BYTES_K || kb == APY_BYTES_K) {
        if (ka != APY_BYTES_K || kb != APY_BYTES_K) return 0;
        return (O(a)->v.s.mut || !O(b)->v.s.mut) ? 1 : 0;
    }
    /* A SET AND A FROZENSET COMPARE EITHER WAY, and the order between them
       is the subset relation rather than a sequence's. */
    if (apy_is_set(a) || apy_is_set(b))
        return (apy_is_set(a) && apy_is_set(b)) ? 1 : 0;
    if (ka != kb) return 0;
    if (ka == APY_STR_K || ka == APY_LIST_K || ka == APY_TUPLE_K) return 1;
    if (!ordering && (ka == APY_DICT_K || ka == APY_RANGE_K)) return 1;
    /* EVERYTHING ELSE IS `object`'s, which is what a function, a generator
       and an instance with no `__eq__` of its own all get. */
    return 2;
}

/* `object`'s own comparison: identity, and the sentinel for the rest. Two
   objects of one user class are not equal in Python merely for being of one
   class, and nothing at all is ordered by it. */
static apy_value apy_object_compare(const char *w, apy_value a, apy_value b) {
    if (strcmp(w, "__eq__") == 0 && a == b) return apy_from_bool(1);
    if (strcmp(w, "__ne__") == 0 && a == b) return apy_from_bool(0);
    return apy_notimplemented();
}

/* One rich comparison, named. See `apy_compare_how` for the sentinel. */
static apy_value apy_kind_compare(const char *w, apy_value a, apy_value b) {
    int how;
    int ordering = strcmp(w, "__eq__") != 0 && strcmp(w, "__ne__") != 0;
    /* AN INSTANCE THAT REACHED HERE HAS NO COMPARISON OF ITS OWN -- the
       class chain was searched first -- so what it gets is `object`'s. */
    if (O(a)->kind == APY_INST_K || O(b)->kind == APY_INST_K)
        return apy_object_compare(w, a, b);
    how = apy_compare_how(a, b, ordering);
    if (how == 0) return apy_notimplemented();
    if (how == 2) return apy_object_compare(w, a, b);
    if (strcmp(w, "__eq__") == 0) return apy_eq(a, b);
    if (strcmp(w, "__ne__") == 0) return apy_ne(a, b);
    if (strcmp(w, "__lt__") == 0) return apy_lt(a, b);
    if (strcmp(w, "__le__") == 0) return apy_le(a, b);
    if (strcmp(w, "__gt__") == 0) return apy_gt(a, b);
    return apy_ge(a, b);
}

static apy_value apy_native_call(apy_value f, apy_value *a, int64_t n) {
    switch (O(f)->v.fn.native) {
    case APY_NAT_POSITIONS:
        /* `code.co_positions()` -- what `apy_code_of` recorded, handed back
           as it stands. A method rather than an attribute because that is how
           CPython spells it, and a program calls it. */
        if (n < 1) return apy_fail("TypeError", "unbound builtin method");
        return apy_getattr(a[0], apy_lit("_positions"));
    case APY_NAT_TASK_CANCEL:
        return n < 1 ? 0 : apy_task_cancel(a[0]);
    case APY_NAT_TASK_RESULT:
        return n < 1 ? 0 : apy_task_result(a[0]);
    case APY_NAT_TASK_DONE:
        return n < 1 ? 0 : apy_task_done(a[0]);
    case APY_NAT_TASK_CANCELLED:
        return n < 1 ? 0 : apy_task_cancelled(a[0]);
    case APY_NAT_TG_ENTER:
        /* THE GROUP ITSELF is what `async with ... as tg` binds. */
        return n < 1 ? 0 : apy_coro_value(a[0]);
    case APY_NAT_TG_CREATE: {
        apy_value t;
        if (n < 2) return apy_fail("TypeError",
                                   "create_task() takes a coroutine");
        t = apy_asyncio_create_task(a[1]);
        if (!t) return 0;
        apy_seq_push(apy_getattr(a[0], apy_lit("_tasks")), t);
        return t;
    }
    case APY_NAT_TG_EXIT: {
        apy_value g;
        if (n < 1) return 0;
        g = apy_gen_new(0, 1);
        O(g)->v.g.coro = 1;
        O(g)->v.g.builtin = APY_CORO_TGWAIT;
        O(g)->v.g.slots[0] = apy_getattr(a[0], apy_lit("_tasks"));
        if (!O(g)->v.g.slots[0]) return 0;
        return g;
    }
    case APY_NAT_INIT:     return apy_none();
    case APY_NAT_BUILTIN_NEW: {
        /* `super().__new__(cls, x)` PAST A BUILTIN BASE, which is the only
           way to build an immutable one: a tuple's contents cannot be set
           after it exists, so `class P(tuple)` fills it here or never.
           `apy_object_default("__new__")` would answer a bare instance with
           an EMPTY tuple inside -- a wrong value rather than an error, and
           the shape every `namedtuple` would have had.

           THE CLASS IS THE FIRST ARGUMENT and is not a bound receiver:
           `__new__` is an implicit staticmethod, so the caller writes the
           class out. */
        apy_value cls, made;
        int kind;
        if (n < 1) return apy_fail("TypeError", "unbound builtin method");
        cls = a[0];
        if (O(cls)->kind != APY_TYPE_K)
            return apy_fail("TypeError",
                            "__new__() argument 1 must be a type");
        made = apy_instance_new(cls);
        if (!made) return 0;
        if (n > 1 && O(a[1])->kind != APY_NONE_K
                && O(made)->kind == APY_INST_K && O(made)->v.o.held) {
            apy_value filled = apy_call_kind(O(O(made)->v.o.held)->kind, a[1]);
            if (!filled) return 0;
            O(made)->v.o.held = filled;
        }
        return made;
    }
    case APY_NAT_BUILTIN_INIT: {
        /* `super().__init__(...)` where the base chain ends at a BUILTIN.
           `class M(dict)` writing it means `dict.__init__`, which FILLS the
           instance -- and there is no Python above `M` to find that on, so
           the walk ran out, `object.__init__` answered, and the call quietly
           did nothing. An empty dict where the program asked for a full one
           is the failure this arrangement is worst at showing: no error, just
           a container that is not there.

           IN PLACE, not a new object: the lines after `super().__init__()` in
           the same body go on using the instance the caller already has. */
        apy_value self, made;
        if (n < 1) return apy_fail("TypeError", "unbound builtin method");
        self = a[0];
        if (O(self)->kind != APY_INST_K || !O(self)->v.o.held)
            return apy_none();
        /* NONE MEANS OMITTED, since the default above always fills the
           slot. `super().__init__(None)` written out is therefore the same as
           `super().__init__()` -- no builtin constructor takes None as
           content, so nothing is lost. */
        if (n < 2 || O(a[1])->kind == APY_NONE_K) return apy_none();
        made = apy_call_kind(O(O(self)->v.o.held)->kind, a[1]);
        if (!made) return 0;
        O(self)->v.o.held = made;
        return apy_none();
    }
    case APY_NAT_EXC_INIT: {
        /* `BaseException.__init__(*args)`: it SETS THE MESSAGE AND `args`,
           which is the whole of what it does and the reason a class writing
           `super().__init__(f"{code}: {message}")` prints that text. */
        apy_value exc, tuple;
        int64_t i;
        if (n < 1) return apy_fail("TypeError", "unbound builtin method");
        exc = a[0];
        if (O(exc)->kind != APY_EXC_K) return apy_none();
        tuple = apy_tuple_new(n > 1 ? n - 1 : 1);
        for (i = 1; i < n; i++) apy_seq_push(tuple, a[i]);
        O(exc)->v.e.arg = n > 1 ? a[1] : apy_none();
        O(exc)->v.e.has_arg = n > 1;
        O(exc)->v.e.argv = tuple;
        /* The text is the ARGUMENT again, not something already rendered --
           see `rendered` on the cell for what that flag stops twice-over. */
        O(exc)->v.e.rendered = 0;
        return apy_none();
    }
    case APY_NAT_NEW: {
        /* `object.__new__(cls)`. The CLASS is the argument, not an instance:
           it is an implicit staticmethod, which is why a bound one still
           receives the class in `a[0]`. */
        char buf[160];
        const char *who;
        if (n < 1 || O(a[0])->kind != APY_TYPE_K)
            return apy_fail("TypeError", "object.__new__(): not a type");
        who = APY_CSTR(O(a[0])->v.t.name);
        /* A CLASS EXTENDING A BUILTIN IS REFUSED, which is CPython's rule
           and not an arbitrary one: `object.__new__` would build the shell
           and leave the builtin half empty, where `str.__new__(S, value)`
           fills it. CPython walks up to the first base that is not a Python
           class and refuses when its `__new__` is not object's -- which for
           `class S(str)` is `str`'s. */
        if (apy_class_builtin_kind(a[0])) {
            snprintf(buf, sizeof buf,
                     "object.__new__(%s) is not safe, use %s.__new__()",
                     who, who);
            return apy_fail("TypeError", buf);
        }
        /* AN ARGUMENT BEYOND THE CLASS IS FOR `__init__` TO TAKE, and only
           when there is one to take it. CPython draws the line twice: a
           class that wrote `__new__` gets the argument count complained
           about, and a class with neither is told it takes none. */
        if (n > 1 && O(a[1])->kind != APY_NONE_K) {
            if (apy_class_find(a[0], apy_name("__new__")))
                return apy_fail("TypeError",
                                "object.__new__() takes exactly one argument "
                                "(the type to instantiate)");
            if (!apy_class_find(a[0], apy_name("__init__"))) {
                snprintf(buf, sizeof buf, "%s() takes no arguments", who);
                return apy_fail("TypeError", buf);
            }
        }
        return apy_instance_new(a[0]);
    }
    case APY_NAT_REPR:
    case APY_NAT_STR:      return n < 1 ? 0 : apy_default_repr(a[0]);
    case APY_NAT_EQ:       return n < 2 ? 0 : apy_default_eq(a[0], a[1]);
    case APY_NAT_NE: {
        apy_value r = n < 2 ? 0 : apy_default_eq(a[0], a[1]);
        return r ? apy_from_bool(!apy_truth(r)) : r;
    }
    case APY_NAT_HASH:     return n < 1 ? 0 : apy_default_hash(a[0]);
    case APY_NAT_GETATTR:  return n < 2 ? 0 : apy_default_getattr(a[0], a[1]);
    case APY_NAT_SETATTR:
        return n < 3 ? 0 : apy_default_setattr(a[0], a[1], a[2]);
    case APY_NAT_DELATTR:  return n < 2 ? 0 : apy_default_delattr(a[0], a[1]);
    case APY_NAT_TYPE_NEW:
        if (n < 4)
            return apy_fail("TypeError",
                            "type.__new__() takes 4 arguments");
        return apy_type_from_ns(a[0], a[1], a[2], a[3]);
    case APY_NAT_TYPE_INIT:
    case APY_NAT_INIT_SUBCLASS: return apy_none();
    /* THE DESCRIPTOR PROTOCOL AS VALUES. A property answers `hasattr(p,
       "__get__")` with True in CPython because the methods exist; here they
       existed as runtime behaviour with nothing naming them, so a program
       that ASKS -- and `enum` asks, to tell a member from a method -- got
       False for a descriptor. */
    case APY_NAT_KIND: {
        /* ONE SELECTOR FOR THE LOT, dispatched on the name it carries: the
           bodies are all one existing runtime entry point, and a selector per
           name would be twenty enum members that differ only in which. */
        const char *w = APY_CSTR(O(f)->v.fn.name);
        if (n < 1) return apy_fail("TypeError", "unbound builtin method");
        /* THE SIX SET METHODS THAT TAKE ANY NUMBER OF OTHERS, above the
           generated table, which knows these names and would hand the whole
           `*rest` tuple to a two-argument symbol. See `apy_set_fold`. */
        if (O(f)->v.fn.vararg && apy_set_folds(w))
            return apy_set_fold(w, a, n);
        /* THE GENERATED TABLE FIRST for a name it knows. Everything below
           answers a PROTOCOL name, and the arity guards between those arms
           would refuse the no-argument forms the table serves -- `upper` has
           none to give. A name the table knows means the same thing on every
           arm they share, because both read `DYN_METHOD_TABLE`. */
        if (apy_kind_meth_arity(w, 0xFFFFu)) return apy_kind_meth_call(w, a, n);
        /* THE STATICS, which a TYPE carries and a VALUE carries too --
           `bytes.fromhex` and `b"".fromhex` are one implicit staticmethod,
           so the receiver decides which body and is otherwise ignored. The
           frontend already lowers the written `bytes.fromhex(s)` form; this
           is the same call reached by name. */
        if (strcmp(w, "format") == 0) {
            /* THE THREE SLOTS THE PACKING FILLED: receiver, the positional
               surplus as a tuple, the keywords as a dict. */
            if (n < 3) return apy_fail("TypeError",
                                       "format() takes a string");
            return apy_str_format(a[0], a[1], a[2]);
        }
        if (strcmp(w, "maketrans") == 0) {
            if (O(a[0])->kind == APY_STR_K)
                return apy_str_maketrans(n > 1 ? a[1] : apy_none(),
                                         n > 2 ? a[2] : apy_none(),
                                         n > 3 ? a[3] : apy_none());
            return apy_bytes_maketrans(n > 1 ? a[1] : apy_none(),
                                       n > 2 ? a[2] : apy_none());
        }
        if (strcmp(w, "fromhex") == 0) {
            if (n < 2) return apy_fail("TypeError",
                                       "fromhex() takes exactly one "
                                       "argument (0 given)");
            /* THE KIND IT WAS REACHED THROUGH decides both which reading
               and, for bytes, the answer's mutability -- as it does for the
               written form, which reaches the same entry point. */
            return apy_any_fromhex(a[0], a[1]);
        }
        if (strcmp(w, "fromkeys") == 0) {
            if (n < 2) return apy_fail("TypeError",
                                       "fromkeys expected at least 1 "
                                       "argument, got 0");
            return apy_dict_fromkeys(a[1], n > 2 ? a[2] : apy_none());
        }
        if (strcmp(w, "from_bytes") == 0) {
            if (n < 2) return apy_fail("TypeError",
                                       "from_bytes() missing required "
                                       "argument 'bytes' (pos 1)");
            return apy_from_bytes_n(a[1], n > 2 ? a[2] : apy_lit("big"));
        }
        if (strcmp(w, "from_number") == 0) {
            if (n < 2) return apy_fail2("TypeError",
                                        "%s.from_number() takes exactly one "
                                        "argument (0 given)%s",
                                        apy_kind_name(a[0]), "");
            return O(a[0])->kind == APY_COMPLEX_K
                   ? apy_complex_from_number(a[0], a[1])
                   : apy_float_from_number(a[0], a[1]);
        }
        if (strcmp(w, "release") == 0) return apy_mview_release(a[0]);
        if (strcmp(w, "_from_flags") == 0) {
            if (n < 3) return apy_fail("TypeError",
                                       "_from_flags() missing required "
                                       "argument 'flags' (pos 2)");
            return apy_mview_from_flags(a[0], a[1], a[2]);
        }
        if (strcmp(w, "cast") == 0) {
            if (n < 2) return apy_fail("TypeError",
                                       "cast() takes exactly one argument "
                                       "(0 given)");
            return apy_mview_cast(a[0], a[1]);
        }
        if (strcmp(w, "toreadonly") == 0) return apy_mview_readonly(a[0]);
        if (strcmp(w, "__enter__") == 0) {
            /* THE VIEW ITSELF is what `with memoryview(b) as m` binds, and
               entering a released one is refused rather than silently
               handing back something every use will refuse. */
            if (!apy_mview_live(a[0])) return 0;
            return a[0];
        }
        if (strcmp(w, "__exit__") == 0) return apy_mview_release(a[0]);
        if (strcmp(w, "tobytes") == 0) return apy_mview_bytes(a[0]);
        if (strcmp(w, "tolist") == 0) {
            /* PER ELEMENT, DECODED AS THE FORMAT SAYS. A fresh view is one
               byte wide and this is a list of ints; a cast one answers
               whatever `cast` was given. */
            int64_t wide = O(a[0])->v.mv.wide ? O(a[0])->v.mv.wide : 1;
            int64_t i, many;
            apy_value out;
            if (!apy_mview_live(a[0])) return 0;
            many = O(a[0])->v.mv.n / wide;
            out = apy_list_new(many + 1);
            if (!out) return 0;
            for (i = 0; i < many; i++) {
                apy_value one = apy_mview_item(a[0], i);
                if (!one || !apy_seq_push(out, one)) return 0;
            }
            return out;
        }
        /* WHAT `object` HANDS DOWN, for a receiver that is a builtin value
           rather than an instance. Each is the answer CPython gives, and the
           four that do nothing really do nothing there too. */
        if (strcmp(w, "__init__") == 0) return apy_none();
        if (strcmp(w, "__init_subclass__") == 0) return apy_none();
        if (strcmp(w, "__getstate__") == 0) return apy_none();
        if (strcmp(w, "__release_buffer__") == 0) {
            /* PEP 688 SAYS WHAT IT IS HANDED: the view being closed, and one
               over THIS buffer. CPython checks both, and a `with
               memoryview(...)` block that ends on the wrong object is a
               mistake worth naming rather than a silent no-op. */
            if (n < 2 || O(a[1])->kind != APY_MVIEW_K)
                return apy_fail("TypeError",
                                "expected a memoryview object");
            if (O(a[1])->v.mv.src != a[0])
                return apy_fail("ValueError",
                                "memoryview's buffer is not this object");
            return apy_none();
        }
        if (strcmp(w, "__subclasshook__") == 0) return apy_notimplemented();
        if (strcmp(w, "__dir__") == 0) return apy_dir(a[0]);
        if (strcmp(w, "__sizeof__") == 0) return apy_sizeof(a[0]);
        if (strcmp(w, "__new__") == 0) {
            /* `[].__new__(list)` IS AN EMPTY LIST: an implicit staticmethod,
               so the receiver is ignored and the CLASS decides. */
            if (n < 2) return apy_fail("TypeError",
                                       "__new__() takes at least 1 argument");
            return apy_call_n(a[1], 0, 0);
        }
        if (strcmp(w, "__getattribute__") == 0) {
            if (n < 2) return apy_fail("TypeError",
                                       "__getattribute__() takes exactly one "
                                       "argument (0 given)");
            return apy_getattr(a[0], a[1]);
        }
        if (strcmp(w, "__setattr__") == 0 || strcmp(w, "__delattr__") == 0)
            /* A BUILTIN VALUE HAS NO `__dict__` to write into, which is the
               whole of what CPython says here. */
            return apy_fail2("AttributeError",
                             "'%s' object has no attribute '%s' and no "
                             "__dict__ for setting new attributes",
                             apy_kind_name(a[0]),
                             n > 1 && O(a[1])->kind == APY_STR_K
                                 ? APY_CSTR(a[1]) : "?");
        if (strcmp(w, "__reduce__") == 0 || strcmp(w, "__reduce_ex__") == 0)
            /* PICKLING IS NOT SERVED HERE, and CPython refuses most of these
               in exactly these words. The four it answers -- bytearray, set,
               frozenset and range -- build a tuple through `copyreg`, which
               this runtime has no counterpart for. */
            return apy_fail2("TypeError", "cannot pickle '%s' object%s",
                             apy_kind_name(a[0]), "");
        /* AND THE SEVEN A PARTICULAR KIND CARRIES, which `apy_kind_attr_of`
           gates -- so a name reaching here already belongs to the receiver
           and nothing below needs to ask which kind it is twice. */
        if (strcmp(w, "__getnewargs__") == 0) {
            /* WHAT `copy` AND `pickle` REBUILD A VALUE FROM. `(self,)` for
               every immutable kind, because handing the value back IS the
               argument that remakes it -- and a complex is the one that
               rebuilds from two numbers rather than from itself. */
            apy_value out = apy_tuple_new(2);
            if (!out) return 0;
            if (O(a[0])->kind == APY_COMPLEX_K) {
                if (!apy_seq_push(out, apy_from_float(O(a[0])->v.z.re)))
                    return 0;
                if (!apy_seq_push(out, apy_from_float(O(a[0])->v.z.im)))
                    return 0;
                return out;
            }
            if (!apy_seq_push(out, a[0])) return 0;
            return out;
        }
        if (strcmp(w, "__rmod__") == 0) return apy_notimplemented();
        if (strcmp(w, "__bytes__") == 0) return a[0];
        if (strcmp(w, "__alloc__") == 0)
            /* THE ONE BYTEARRAY INTERNAL A PROGRAM CAN READ, and CPython's
               answer for a freshly built one is its length plus the
               terminator -- nought for an empty one, which holds no buffer
               at all. */
            return apy_from_int(O(a[0])->v.s.n ? O(a[0])->v.s.n + 1 : 0);
        if (strcmp(w, "__getformat__") == 0) {
            /* THE ONLY TWO KEYS, and CPython names both in the refusal. */
            if (n < 2 || O(a[1])->kind != APY_STR_K
                || (strcmp(APY_CSTR(a[1]), "double") != 0
                    && strcmp(APY_CSTR(a[1]), "float") != 0))
                return apy_fail("ValueError",
                                "__getformat__() argument 1 must be "
                                "'double' or 'float'");
            return apy_lit("IEEE, little-endian");
        }
        if (strcmp(w, "__class_getitem__") == 0) {
            /* `list[int]` REACHED BY NAME. The written form is a subscript
               the frontend lowers; this is the method behind it, which
               `typing` calls directly to parameterise a container. */
            apy_value args;
            if (n < 2) return apy_fail("TypeError",
                                       "__class_getitem__() takes exactly "
                                       "one argument (0 given)");
            if (O(a[1])->kind == APY_TUPLE_K) args = a[1];
            else {
                args = apy_tuple_new(2);
                if (!args) return 0;
                if (!apy_seq_push(args, a[1])) return 0;
            }
            /* A TYPE IS ALREADY THE ORIGIN. Reached off `list` itself
               the receiver IS the type -- CPython binds the classmethod to
               it -- and `apy_type_for` of a type answers `type`, which
               would make `list.__class_getitem__(int)` a `type[int]`. */
            if ((O(a[0])->kind == APY_FUNC_K && O(a[0])->v.fn.is_type)
                    || O(a[0])->kind == APY_TYPE_K)
                return apy_alias_new(a[0], args);
            return apy_alias_new(apy_type_for(a[0]), args);
        }
        /* CALLING IT DOES NOTHING, which is what CPython's does for a
           generator that has already finished -- and one that has not is
           closed by the runtime rather than by the program. */
        if (strcmp(w, "__del__") == 0) return apy_none();
        if (strcmp(w, "__hash__") == 0) return apy_hash(a[0]);
        if (strcmp(w, "__len__") == 0) return apy_len(a[0]);
        if (strcmp(w, "__iter__") == 0) return apy_iter(a[0]);
        /* `await c` WALKS THE COROUTINE ITSELF, so `__await__` hands it
           back: there is no second object between the two here, and
           `iter(x)` answers the same way for a generator. */
        if (strcmp(w, "__await__") == 0 || strcmp(w, "__aiter__") == 0)
            return a[0];
        /* AND `__anext__` IS `asend(None)`, which is what CPython's is. */
        if (strcmp(w, "__anext__") == 0)
            return apy_agen_asend(a[0], apy_none());
        if (strcmp(w, "__next__") == 0) return apy_next(a[0], 0, 0);
        if (strcmp(w, "__length_hint__") == 0)
            return apy_from_int(apy_cursor_left(a[0]));
        if (strcmp(w, "__setstate__") == 0 && n >= 2)
            return apy_cursor_setstate(a[0], a[1]);
        if (strcmp(w, "keys") == 0)
            return apy_dict_parts(a[0], APY_PART_KEYS);
        if (strcmp(w, "values") == 0)
            return apy_dict_parts(a[0], APY_PART_VALUES);
        if (strcmp(w, "items") == 0)
            return apy_dict_parts(a[0], APY_PART_ITEMS);
        /* THE ONE-ARGUMENT HALF OF WHAT `object` AND THE NUMBERS CARRY. Each
           is an existing runtime entry point under the name Python gives it,
           which is the whole reason one selector serves the lot. */
        if (strcmp(w, "__str__") == 0) return apy_str(a[0]);
        if (strcmp(w, "__repr__") == 0) return apy_repr(a[0]);
        if (strcmp(w, "__bool__") == 0) return apy_from_bool(apy_truth(a[0]));
        if (strcmp(w, "__neg__") == 0) return apy_neg(a[0]);
        if (strcmp(w, "__pos__") == 0) return apy_pos(a[0]);
        if (strcmp(w, "__abs__") == 0) return apy_abs(a[0]);
        if (strcmp(w, "__invert__") == 0) return apy_invert(a[0]);
        if (strcmp(w, "__int__") == 0) return apy_to_int(a[0]);
        if (strcmp(w, "__float__") == 0) return apy_to_float(a[0]);
        /* A COMPLEX IS ALREADY ONE, and `__index__` on an int answers the
           int -- except for a bool, which becomes the 0 or 1 it stands for:
           `True.__index__()` is `1` and not `True`. */
        if (strcmp(w, "__complex__") == 0) return a[0];
        if (strcmp(w, "__index__") == 0)
            return apy_is_big(a[0]) ? a[0] : apy_from_int(O(a[0])->v.i);
        if (strcmp(w, "__trunc__") == 0) return apy_math_trunc(a[0]);
        if (strcmp(w, "__floor__") == 0) return apy_math_floor(a[0]);
        if (strcmp(w, "__ceil__") == 0) return apy_math_ceil(a[0]);
        if (strcmp(w, "__reversed__") == 0) return apy_reversed(a[0]);
        /* `x.__round__()` and `x.__round__(n)` are ONE method, told apart by
           the count that really arrived -- see `apy_kind_method_opt`. ABOVE
           the guard below, because the no-argument form is one of the two. */
        if (strcmp(w, "__round__") == 0)
            return n >= 2 ? apy_round_to(a[0], a[1]) : apy_round(a[0]);
        if (n < 2) return apy_fail("TypeError",
                                   "builtin method takes an argument");
        /* THE RICH COMPARISONS, which answer `NotImplemented` for a pair
           that does not compare rather than False. See `apy_kind_compare`. */
        if (w[0] == '_' && w[1] == '_' && apy_object_arity(
                (apy_value)(uintptr_t)w) == 2
                && strcmp(w, "__format__") != 0)
            return apy_kind_compare(w, a[0], a[1]);
        if (strcmp(w, "__format__") == 0) return apy_format(a[0], a[1]);
        /* THE BINARY OPERATORS AS METHODS. A reflected one is the same
           entry point with the operands the other way round, which is what
           `__radd__` means. */
        if (strcmp(w, "__add__") == 0) return apy_add(a[0], a[1]);
        if (strcmp(w, "__radd__") == 0) return apy_add(a[1], a[0]);
        if (strcmp(w, "__sub__") == 0) return apy_sub(a[0], a[1]);
        if (strcmp(w, "__rsub__") == 0) return apy_sub(a[1], a[0]);
        if (strcmp(w, "__mul__") == 0) return apy_mul(a[0], a[1]);
        if (strcmp(w, "__rmul__") == 0) return apy_mul(a[1], a[0]);
        if (strcmp(w, "__truediv__") == 0) return apy_truediv(a[0], a[1]);
        if (strcmp(w, "__rtruediv__") == 0) return apy_truediv(a[1], a[0]);
        if (strcmp(w, "__floordiv__") == 0) return apy_floordiv(a[0], a[1]);
        if (strcmp(w, "__rfloordiv__") == 0) return apy_floordiv(a[1], a[0]);
        if (strcmp(w, "__mod__") == 0) return apy_mod(a[0], a[1]);
        if (strcmp(w, "__rmod__") == 0) return apy_mod(a[1], a[0]);
        if (strcmp(w, "__pow__") == 0) return apy_pow(a[0], a[1]);
        if (strcmp(w, "__rpow__") == 0) return apy_pow(a[1], a[0]);
        if (strcmp(w, "__divmod__") == 0) return apy_divmod(a[0], a[1]);
        if (strcmp(w, "__rdivmod__") == 0) return apy_divmod(a[1], a[0]);
        if (strcmp(w, "__and__") == 0) return apy_bitand(a[0], a[1]);
        if (strcmp(w, "__rand__") == 0) return apy_bitand(a[1], a[0]);
        if (strcmp(w, "__or__") == 0) return apy_bitor(a[0], a[1]);
        if (strcmp(w, "__ror__") == 0) return apy_bitor(a[1], a[0]);
        if (strcmp(w, "__xor__") == 0) return apy_bitxor(a[0], a[1]);
        if (strcmp(w, "__rxor__") == 0) return apy_bitxor(a[1], a[0]);
        if (strcmp(w, "__lshift__") == 0) return apy_lshift(a[0], a[1]);
        if (strcmp(w, "__rlshift__") == 0) return apy_lshift(a[1], a[0]);
        if (strcmp(w, "__rshift__") == 0) return apy_rshift(a[0], a[1]);
        if (strcmp(w, "__rrshift__") == 0) return apy_rshift(a[1], a[0]);
        if (strcmp(w, "__delitem__") == 0)
            return apy_delitem(a[0], a[1]);
        /* `x in obj` -- the NEEDLE FIRST, which is the order `apy_contains`
           takes and the reverse of the method's. */
        if (strcmp(w, "__contains__") == 0) return apy_contains(a[1], a[0]);
        if (strcmp(w, "__getitem__") == 0) return apy_getitem(a[0], a[1]);
        if (O(a[0])->kind == APY_RANGE_K
                && (strcmp(w, "index") == 0 || strcmp(w, "count") == 0)) {
            int64_t want, at;
            if (!apy_is_int_like(a[1]))
                return strcmp(w, "count") == 0 ? apy_from_int(0)
                    : apy_fail("ValueError", "value is not in range");
            if (!apy_index_arg(a[1], &want, APY_IDX_SIZE)) return 0;
            at = apy_range_find(a[0], want);
            if (strcmp(w, "count") == 0) return apy_from_int(at >= 0 ? 1 : 0);
            if (at < 0) return apy_fail("ValueError",
                                        "value is not in range");
            return apy_from_int(at);
        }
        /* THE IN-PLACE OPERATORS, each the runtime entry point the operator
           form already reaches -- so `xs.__iadd__(ys)` and `xs += ys` are one
           implementation rather than two that could disagree. */
        if (strcmp(w, "__iadd__") == 0) return apy_iadd(a[0], a[1]);
        if (strcmp(w, "__imul__") == 0) return apy_iop(a[0], a[1], apy_lit("*"));
        if (strcmp(w, "__ior__") == 0) return apy_iop(a[0], a[1], apy_lit("|"));
        if (strcmp(w, "__iand__") == 0) return apy_iop(a[0], a[1], apy_lit("&"));
        if (strcmp(w, "__isub__") == 0) return apy_iop(a[0], a[1], apy_lit("-"));
        if (strcmp(w, "__ixor__") == 0) return apy_iop(a[0], a[1], apy_lit("^"));
        if (strcmp(w, "index") == 0) return apy_index_of(a[0], a[1]);
        if (strcmp(w, "count") == 0) return apy_count_of(a[0], a[1]);
        if (strcmp(w, "append") == 0) return apy_seq_push(a[0], a[1]);
        if (strcmp(w, "add") == 0) return apy_set_add(a[0], a[1]);
        if (strcmp(w, "discard") == 0) return apy_set_discard(a[0], a[1]);
        if (strcmp(w, "isdisjoint") == 0)
            return apy_set_isdisjoint(a[0], a[1]);
        /* `b.__buffer__(flags)` answers a memoryview over it, which is what
           the protocol is for and what `memoryview(b)` already does. */
        if (strcmp(w, "__buffer__") == 0) return apy_memoryview(a[0]);
        if (strcmp(w, "__setitem__") == 0 && n >= 3)
            return apy_setitem(a[0], a[1], a[2]);
        return apy_fail("TypeError", "not callable");
    }
    case APY_NAT_DESCR_GET:
        if (n < 2) return apy_fail("TypeError",
                                   "__get__() takes at least 2 arguments");
        return apy_descr_get(a[0], a[1], n > 2 ? a[2] : 0);
    case APY_NAT_DESCR_SET:
        if (n < 3) return apy_fail("TypeError",
                                   "__set__() takes 3 arguments");
        return apy_descr_set(a[0], a[1], a[2]) ? apy_none() : 0;
    case APY_NAT_DESCR_DEL:
        if (n < 2) return apy_fail("TypeError",
                                   "__delete__() takes 2 arguments");
        return apy_descr_set(a[0], a[1], 0) ? apy_none() : 0;
    case APY_NAT_HAS_DEFAULT: {
        /* `T.has_default()` -- whether a default was written. */
        apy_value held;
        if (n < 1) return apy_from_bool(0);
        held = apy_dict_get_or(O(a[0])->v.o.dict, apy_lit("__default__"), 0);
        return apy_from_bool(held && O(held)->kind != APY_NONE_K);
    }
    case APY_NAT_TYPE_CALL:
        /* Only reached for a native called with no keywords; the path that
           carries them intercepts this selector where the dict is in hand. */
        if (n < 1) return apy_fail("TypeError", "type.__call__() needs a type");
        return apy_instantiate(a[0], a + 1, n - 1, 0, 0);
    case APY_NAT_GEN_SEND:
        return n < 2 ? 0 : apy_gen_send(a[0], a[1]);
    case APY_NAT_GEN_THROW:
        return n < 2 ? 0 : apy_gen_throw(a[0], a[1]);
    case APY_NAT_GEN_CLOSE:
        return n < 1 ? 0 : apy_gen_close(a[0]);
    /* AND AN ASYNC GENERATOR'S THREE, each of which answers an AWAITABLE
       rather than doing the work -- see `apy_agen_await`. */
    case APY_NAT_AGEN_SEND:
        return n < 2 ? 0 : apy_agen_asend(a[0], a[1]);
    case APY_NAT_AGEN_THROW:
        return n < 2 ? 0 : apy_agen_athrow(a[0], a[1]);
    case APY_NAT_AGEN_CLOSE:
        return n < 1 ? 0 : apy_agen_aclose(a[0]);
    default:
        return apy_fail("TypeError", "not callable");
    }
}

/* The object defaults, by name. `super()` on a class whose base chain has run
   out looks here, which is what makes `super().__init__()` in a class with no
   base do what CPython's `object.__init__` does rather than fail. */
APY_API apy_value apy_object_default(apy_value wantv) {
    const char *want = (const char *)wantv;
    if (strcmp(want, "__init__") == 0)
        return apy_native(APY_NAT_INIT, 1, "__init__");
    if (strcmp(want, "__new__") == 0)
        /* TWO SLOTS, THE SECOND OPTIONAL. `object.__new__(cls)` is the
           ordinary spelling and `object.__new__(cls, content)` fills the
           builtin half of a class that extends one -- see the native. */
        return apy_native(APY_NAT_NEW, 2, "__new__");
    if (strcmp(want, "__repr__") == 0)
        return apy_native(APY_NAT_REPR, 1, "__repr__");
    if (strcmp(want, "__str__") == 0)
        return apy_native(APY_NAT_STR, 1, "__str__");
    if (strcmp(want, "__eq__") == 0)
        return apy_native(APY_NAT_EQ, 2, "__eq__");
    if (strcmp(want, "__ne__") == 0)
        return apy_native(APY_NAT_NE, 2, "__ne__");
    if (strcmp(want, "__hash__") == 0)
        return apy_native(APY_NAT_HASH, 1, "__hash__");
    if (strcmp(want, "__getattribute__") == 0)
        return apy_native(APY_NAT_GETATTR, 2, "__getattribute__");
    if (strcmp(want, "__setattr__") == 0)
        return apy_native(APY_NAT_SETATTR, 3, "__setattr__");
    if (strcmp(want, "__delattr__") == 0)
        return apy_native(APY_NAT_DELATTR, 2, "__delattr__");
    /* Every class has one, and a user hook ends by calling it: `object`'s is
       the no-op that terminates the chain. */
    if (strcmp(want, "__init_subclass__") == 0)
        return apy_native(APY_NAT_INIT_SUBCLASS, 1, "__init_subclass__");
    return 0;
}

static apy_value apy_invoke(apy_value f, apy_value *a, int64_t n) {
    uintptr_t c = O(f)->v.fn.code;
    /* A NATIVE has no code pointer to call; the selector is the whole of it.
       Tested first, because calling through a null `code` is the crash this
       replaced. */
    if (O(f)->v.fn.native) return apy_native_call(f, a, n);
    switch (n) {
    case 0: return ((apy_fn0)c)(f);
    case 1: return ((apy_fn1)c)(f, a[0]);
    case 2: return ((apy_fn2)c)(f, a[0], a[1]);
    case 3: return ((apy_fn3)c)(f, a[0], a[1], a[2]);
    case 4: return ((apy_fn4)c)(f, a[0], a[1], a[2], a[3]);
    case 5: return ((apy_fn5)c)(f, a[0], a[1], a[2], a[3], a[4]);
    case 6: return ((apy_fn6)c)(f, a[0], a[1], a[2], a[3], a[4], a[5]);
    case 7: return ((apy_fn7)c)(f, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    case 8: return ((apy_fn8)c)(f, a[0], a[1], a[2], a[3], a[4], a[5], a[6],
                                a[7]);
    case 9: return ((apy_fn9)c)(f, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
    case 10: return ((apy_fn10)c)(f, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
    case 11: return ((apy_fn11)c)(f, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10]);
    case 12: return ((apy_fn12)c)(f, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11]);
    case 13: return ((apy_fn13)c)(f, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12]);
    case 14: return ((apy_fn14)c)(f, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13]);
    case 15: return ((apy_fn15)c)(f, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14]);
    case 16: return ((apy_fn16)c)(f, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
    default:
        return apy_fail("TypeError", "a function of more than 16 parameters "
                                     "is not supported");
    }
}

/* THE DEFAULT THE `def` EVALUATED FOR DECLARED PARAMETER `idx`, or 0 where
   it has none.

   NOT SIMPLY THE TRAILING ONES. `defaults` holds the POSITIONAL defaults
   first and the KEYWORD-ONLY defaults after them, and a keyword-only
   parameter WITHOUT a default may follow a positional one WITH: `def f(a,
   b=1, *, k)` records one default, for `b`, while `k` has none. Indexing
   from the end of the whole declaration looked for `b`'s default two places
   before the start of the array, found nothing, and reported `b` as a hole
   -- so `f(1, k=2)` refused a call CPython answers.

   `nkwdefault` is on the function object for exactly this: it says where the
   first run ends and the second begins. */
static apy_value apy_fn_default(apy_value f, int64_t idx) {
    int64_t declared = O(f)->v.fn.arity - (O(f)->v.fn.vararg ? 1 : 0)
                                        - (O(f)->v.fn.kwarg ? 1 : 0);
    int64_t bypos = declared - O(f)->v.fn.kwonly;
    int64_t nkw = O(f)->v.fn.nkwdefault;
    int64_t npos = O(f)->v.fn.ndefaults - nkw;
    int64_t d;
    if (!O(f)->v.fn.defaults) return 0;
    if (npos < 0) npos = 0;
    if (nkw < 0) nkw = 0;
    if (idx < 0) return 0;
    if (idx < bypos) {
        /* The positional defaults cover the LAST `npos` positional
           parameters, and sit at the FRONT of the array. */
        d = idx - (bypos - npos);
        if (d < 0 || d >= npos) return 0;
        return O(f)->v.fn.defaults[d];
    }
    /* And the keyword-only defaults cover the last `nkw` of the keyword-only
       tail, sitting behind the positional ones. */
    if (idx < declared - nkw || idx >= declared) return 0;
    d = npos + (idx - (declared - nkw));
    if (d < 0 || d >= O(f)->v.fn.ndefaults) return 0;
    return O(f)->v.fn.defaults[d];
}

static apy_value apy_arity_error(apy_value f, int64_t got) {
    char buf[192];
    /* POSITIONS, not declared slots. A keyword-only parameter is declared and
       cannot be filled by position, so counting it here told the caller to
       pass more positional arguments than the function accepts. */
    int64_t want = O(f)->v.fn.arity - (O(f)->v.fn.bound ? 1 : 0)
                   - (O(f)->v.fn.vararg ? 1 : 0) - (O(f)->v.fn.kwarg ? 1 : 0)
                   - O(f)->v.fn.kwonly;
    if (want < 0) want = 0;
    /* CPYTHON WORDS A BUILTIN DIFFERENTLY from a function it compiled:
       `expected 0 arguments, got 2`, with no parentheses -- and it names the
       method only when the method has a name a program wrote. A bound SLOT
       (`[1].__len__`) is anonymous in the message, which is the same line
       `type()` draws between `method-wrapper` and
       `builtin_function_or_method`. */
    if (O(f)->v.fn.native) {
        const char *w = APY_CSTR(O(f)->v.fn.name);
        size_t len = strlen(w);
        /* A BOUND BUILTIN METHOD IS WORDED BY ITS RECEIVER, and there are
           nine wordings rather than the three below -- `[].append()` is
           `list.append() takes exactly one argument (0 given)` in CPython,
           not `append expected 1 argument, got 0`. See `apy_meth_arity`.
           Anything the generated table does not know -- a generator's
           `send`, a bound slot, a native this runtime invented -- falls
           through to the plainer wording. */
        if (O(f)->v.fn.bound) {
            apy_value recv = O(f)->v.fn.bound;
            int64_t packed = apy_kind_meth_words(w, apy_kind_bit(recv));
            if (packed)
                return apy_meth_arity_words(recv, w, packed, got);
        }
        /* A DUNDER WITH A FIXED ARITY IS A BOUND SLOT, and CPython leaves a
           slot anonymous in this message. One with a RANGE is a method
           CPython wrote out rather than a slot -- `(5).__round__` really is a
           `builtin_function_or_method` there -- and those are named. */
        int ranged = !O(f)->v.fn.defaults && O(f)->v.fn.ndefaults;
        int dunder = !ranged && len >= 5 && w[0] == '_' && w[1] == '_'
                     && w[len - 1] == '_' && w[len - 2] == '_';
        int64_t least = ranged ? want - O(f)->v.fn.ndefaults : want;
        if (least < 0) least = 0;
        /* A FIXED ARITY IS SAID PLAINLY whichever way the count is wrong;
           only a method with a RANGE says which end it missed. */
        if (least == want)
            snprintf(buf, sizeof buf, "%s%sexpected %lld argument%s, got %lld",
                     dunder ? "" : w, dunder ? "" : " ", (long long)want,
                     want == 1 ? "" : "s", (long long)got);
        else if (got < least)
            snprintf(buf, sizeof buf, "%s%sexpected at least %lld argument%s, "
                     "got %lld", dunder ? "" : w, dunder ? "" : " ",
                     (long long)least, least == 1 ? "" : "s", (long long)got);
        else
            snprintf(buf, sizeof buf, "%s%sexpected at most %lld argument%s, "
                     "got %lld", dunder ? "" : w, dunder ? "" : " ",
                     (long long)want, want == 1 ? "" : "s", (long long)got);
        return apy_fail("TypeError", buf);
    }
    snprintf(buf, sizeof buf,
             "%s() takes %lld positional argument%s but %lld %s given",
             APY_CSTR(O(f)->v.fn.name), (long long)want,
             want == 1 ? "" : "s", (long long)got,
             got == 1 ? "was" : "were");
    return apy_fail("TypeError", buf);
}

/* CPYTHON'S LIST OF NAMES -- `'a'`, then `'a' and 'b'`, then `'a', 'b', and
   'c'`. The Oxford comma appears only from three on, which is why the
   separator cannot be chosen from the position alone. Truncated rather than
   overrun: a signature wide enough to fill this is one no message was going
   to rescue anyway. */
static void apy_name_list(char *out, size_t cap, apy_value *names,
                          int64_t n) {
    int64_t i;
    size_t at = 0;
    out[0] = 0;
    for (i = 0; i < n; i++) {
        const char *nm = (names && names[i]) ? APY_CSTR(names[i]) : "?";
        const char *sep = "";
        if (i) sep = (i == n - 1) ? (n > 2 ? ", and " : " and ") : ", ";
        if (at + strlen(sep) + strlen(nm) + 3 >= cap) break;
        at += (size_t)snprintf(out + at, cap - at, "%s'%s'", sep, nm);
    }
}

/* THE ARITY OF A FUNCTION THE PROGRAM WROTE, worded as CPython words it.
   Four messages, and they are one family: which end the count fell off, and
   whether a default makes the low end a RANGE rather than a number.

       f() takes 2 positional arguments but 3 were given
       f() takes from 1 to 2 positional arguments but 3 were given
       f() missing 1 required positional argument: 'b'
       f() missing 2 required keyword-only arguments: 'k' and 'j'

   `given` COUNTS THE RECEIVER, because CPython does: `C().m(1, 2)` is
   `C.m() takes 2 positional arguments but 3 were given`, self included at
   both ends. The caller adds it, because only the caller knows whether the
   receiver is on the function (a bound method) or still ahead of it (a class
   being instantiated).

   `kwo` is how many of the keywords landed on KEYWORD-ONLY parameters, which
   CPython names in the surplus message and nowhere else. Zero from the plain
   positional path, which has no keywords to land.

   NAMED BY ITS QUALNAME. `C.m()` and `outer.<locals>.inner()` are what
   CPython prints, and the plain name is the fallback for a function that
   never got one. */
static apy_value apy_fn_arity_error(apy_value f, int64_t given, int64_t kwo) {
    char buf[352], names[224], lead[176];
    apy_value q = O(f)->v.fn.qualname;
    const char *who = APY_CSTR(q ? q : O(f)->v.fn.name);
    int64_t declared = O(f)->v.fn.arity - (O(f)->v.fn.vararg ? 1 : 0)
                                        - (O(f)->v.fn.kwarg ? 1 : 0);
    int64_t bypos = declared - O(f)->v.fn.kwonly;
    /* ONLY A POSITIONAL PARAMETER'S DEFAULT widens the low end. The
       keyword-only defaults sit past `bypos` and counting them said `takes
       from 0 to 1 positional arguments` for `def f(a, *, k=0)`. */
    int64_t ndef = O(f)->v.fn.ndefaults - O(f)->v.fn.nkwdefault;
    int64_t least;
    if (bypos < 0) bypos = 0;
    if (ndef < 0) ndef = 0;
    least = bypos - ndef;
    if (least < 0) least = 0;

    if (given > bypos && !O(f)->v.fn.vararg) {
        if (least == bypos)
            snprintf(lead, sizeof lead,
                     "%s() takes %lld positional argument%s", who,
                     (long long)bypos, bypos == 1 ? "" : "s");
        else
            snprintf(lead, sizeof lead,
                     "%s() takes from %lld to %lld positional arguments",
                     who, (long long)least, (long long)bypos);
        /* A KEYWORD-ONLY ARGUMENT IS COUNTED SEPARATELY OR NOT AT ALL --
           CPython switches the whole tail of the sentence on whether one was
           passed, rather than adding a clause to it. */
        if (kwo > 0)
            snprintf(buf, sizeof buf, "%s but %lld positional argument%s "
                     "(and %lld keyword-only argument%s) were given", lead,
                     (long long)given, given == 1 ? "" : "s",
                     (long long)kwo, kwo == 1 ? "" : "s");
        else
            snprintf(buf, sizeof buf, "%s but %lld %s given", lead,
                     (long long)given, given == 1 ? "was" : "were");
        return apy_fail("TypeError", buf);
    }
    /* TOO FEW IS SAID IN NAMES, not in a count of what the function takes.
       The names still wanted start where the arguments ran out -- and that
       index already skips a receiver, which is why `given` counts it. */
    if (given < least && O(f)->v.fn.pnames) {
        apy_name_list(names, sizeof names, O(f)->v.fn.pnames + given,
                      least - given);
        snprintf(buf, sizeof buf,
                 "%s() missing %lld required positional argument%s: %s", who,
                 (long long)(least - given), least - given == 1 ? "" : "s",
                 names);
        return apy_fail("TypeError", buf);
    }
    /* A REQUIRED KEYWORD-ONLY PARAMETER IS MISSED BY NAME and never by
       count: no position could have reached it, so a message about how many
       positional arguments the function takes sends the reader to the wrong
       half of the signature. They are the parameters past `bypos` that the
       trailing keyword defaults do not cover. */
    if (O(f)->v.fn.kwonly && O(f)->v.fn.pnames) {
        int64_t end = declared - O(f)->v.fn.nkwdefault;
        if (end > bypos) {
            apy_name_list(names, sizeof names, O(f)->v.fn.pnames + bypos,
                          end - bypos);
            snprintf(buf, sizeof buf,
                     "%s() missing %lld required keyword-only argument%s: %s",
                     who, (long long)(end - bypos),
                     end - bypos == 1 ? "" : "s", names);
            return apy_fail("TypeError", buf);
        }
    }
    /* NOTHING THE FAMILY ABOVE COVERS -- a function whose parameter names the
       frontend never recorded, or a count wrong in some way the signature
       does not explain. The plainer wording, with the receiver taken back out
       of the number because that wording never counted it. */
    return apy_arity_error(f, given - (O(f)->v.fn.bound ? 1 : 0));
}

/* One call, with the `**kw` dict the caller resolved (or 0 for none).

   Threaded as a parameter rather than appended to `argv` by the caller,
   because a callee with BOTH `*rest` and `**kw` packs the surplus positionals
   into `rest` FIRST -- a kw dict sitting in `argv` would be swallowed by that
   packing and arrive as one more element of `rest`. Only this function knows
   where the boundary is, so only it can put the dict past it. */
/* `C(...)` -- allocate, then run `__init__` if there is one. The instance is
   what the call yields whatever `__init__` returns, which is why its result is
   discarded rather than propagated.

   SEPARATE FROM `apy_call_nk` because `type.__call__` is exactly this and
   nothing else: a metaclass that overrides `__call__` and ends by delegating
   upward has to reach the default without re-entering the hook. */
/* `dict(x)`, `list(x)`, `tuple(x)`, `set(x)` or `str(x)`, chosen by KIND
   rather than by name -- the caller has an instance carrying one of these and
   wants another built from its argument, and the kind is what it knows.
   Assembled from the entry points the frontend already lowers those calls to,
   so there is no second opinion about what `dict(pairs)` means. */
static apy_value apy_call_kind(int kind, apy_value src) {
    apy_value out;
    /* ON THE ARGUMENT AS WRITTEN, ahead of the unwrap below: `list(d)` for a
       `class D(list)` asks D's own `__len__`, and unwrapping first would
       have asked the held list's instead -- which is not a method the
       program wrote and cannot print anything.

       ONLY FOR A LIST. `tuple(x)`, `set(x)` and `dict(x)` do not ask in
       CPython, and the difference is measurable from inside the program. */
    if (kind == APY_LIST_K && !apy_length_hint(src)) return 0;
    /* AN INSTANCE ARGUMENT MEANS ITS CONTENT. `OrderedDict(other)` reaches
       here with an instance, and every branch below reads `src` as a real
       container -- see `_content_of` in the host for the same unwrap. */
    if (O(src)->kind == APY_INST_K && O(src)->v.o.held)
        src = O(src)->v.o.held;
    if (kind == APY_DICT_K) {
        out = apy_dict_new(4);
        return (out && apy_update(out, src)) ? out : 0;
    }
    /* A SET IS FILLED ONE ELEMENT AT A TIME and not by `apy_extend`, which
       pushes through `apy_seq_push` -- a list operation that reports
       `'set' object has no attribute 'append'` when handed one. Adding is
       also the only way to get the DEDUPLICATION a set is for. */
    if (kind == APY_SET_K) {
        /* THROUGH A LIST, which is not a detour. `apy_extend` is the one
           thing here that already knows how to drain every kind of source --
           a generator, a dict (its KEYS), a str -- and it pushes through
           `apy_seq_push`, a LIST operation that refuses a set outright.
           Collect with it, then add, which is also where the deduplication a
           set is for happens.

           NOT `apy_key_at`: that answers what ITERATING yields for a dict, a
           set or a cursor, and a plain list falls past those branches into
           the cursor one, where the union's `it.i` overlaps the element
           count. It read off the end and the program died with its buffered
           output lost, which is why this printed nothing at all. */
        apy_value tmp = apy_list_new(4);
        int64_t i;
        if (!tmp || !apy_extend(tmp, src)) return 0;
        out = apy_set_new(4);
        if (!out) return 0;
        for (i = 0; i < O(tmp)->v.q.n; i++)
            if (!apy_set_add(out, O(tmp)->v.q.items[i])) return 0;
        return out;
    }
    if (kind == APY_LIST_K || kind == APY_TUPLE_K) {
        out = kind == APY_LIST_K ? apy_list_new(4) : apy_tuple_new(4);
        return (out && apy_extend(out, src)) ? out : 0;
    }
    if (kind == APY_STR_K) return apy_str(src);
    return 0;
}

/* THE BUILTIN TYPE CONSTRUCTORS, reached through a TYPE OBJECT rather than
   through the bare word: `type(5)("ff", 16)`, `x.__class__(v)`, and every
   `getattr(x, "__class__")(...)` a generic copy helper writes. The written
   form folds its keywords in the frontend against this same table -- see
   `CTOR_PARAMS` in `frontends/python/methods.py` -- and this is the half
   that cannot be folded there, because the NAME is not known until run time.

   ONE NARROWING against the written form, stated rather than hidden: an
   unknown keyword gets CPython's message without the `Did you mean` hint.
   The hint is an edit distance over the parameter names, and the runtime has
   no port of one. */
typedef struct {
    const char *name;
    signed char most, least, paren;
    const char *slot0, *slot1, *slot2;
    /* What a MISSING REQUIRED first argument says, and what an EMPTY first
       slot under a filled one says. A null `headless` means the type answers
       its ZERO-ARGUMENT form instead of refusing, which is `str`. */
    const char *missing, *headless;
} apy_ctor_shape;

static const apy_ctor_shape APY_CTORS[] = {
    {"int",        2, 0, 0, 0, "base", 0, 0, "int() missing string argument"},
    {"bool",       1, 0, 0, 0, 0, 0, 0, 0},
    {"float",      1, 0, 0, 0, 0, 0, 0, 0},
    {"list",       1, 0, 0, 0, 0, 0, 0, 0},
    {"tuple",      1, 0, 0, 0, 0, 0, 0, 0},
    {"set",        1, 0, 0, 0, 0, 0, 0, 0},
    {"frozenset",  1, 0, 0, 0, 0, 0, 0, 0},
    {"dict",       1, 0, 0, 0, 0, 0, 0, 0},
    {"range",      3, 1, 0, 0, 0, 0,
     "range expected at least 1 argument, got 0", 0},
    {"slice",      3, 1, 0, 0, 0, 0,
     "slice expected at least 1 argument, got 0", 0},
    {"memoryview", 1, 1, 1, "object", 0, 0,
     "memoryview() missing required argument 'object' (pos 1)",
     "memoryview() missing required argument 'object' (pos 1)"},
    {"complex",    2, 0, 1, "real", "imag", 0, 0, 0},
    {"str",        3, 0, 0, "object", "encoding", "errors", 0, 0},
    {"bytes",      3, 0, 1, "source", "encoding", "errors", 0,
     "encoding without a string argument"},
    {"bytearray",  3, 0, 1, "source", "encoding", "errors", 0,
     "encoding without a string argument"},
    {0, 0, 0, 0, 0, 0, 0, 0, 0}
};

/* CPYTHON'S OWN EDIT DISTANCE, ported from `Python/suggestions.c` -- the
   same port `frontends/python/methods.py` carries for the WRITTEN spelling,
   here so the run-time refusal says the same thing. A suggestion this
   compiler makes where CPython makes none is a new divergence, not a
   kindness, which is why the constants are reproduced rather than guessed:
   a CASE change is cheaper than a real one, so `SEP` finds `sep` and a
   longer all-caps name does not find its lowercase twin. */
#define APY_MOVE_COST 2
#define APY_CASE_COST 1
#define APY_MAX_SUGGEST 40

static int apy_sub_cost(char a, char b) {
    /* ASCII LOWERCASING WRITTEN OUT rather than through `tolower`, which is
       locale-sensitive and would need `<ctype.h>` -- a parameter name is
       ASCII by construction. */
    char la = a >= 'A' && a <= 'Z' ? (char)(a + 32) : a;
    char lb = b >= 'A' && b <= 'Z' ? (char)(b + 32) : b;
    if (a == b) return 0;
    if (la == lb) return APY_CASE_COST;
    return APY_MOVE_COST;
}

static int apy_edits(const char *a, const char *b, int max_cost) {
    int row[APY_MAX_SUGGEST], la, lb, i, k, result = 0;
    /* THE COMMON AFFIXES COME OFF FIRST, which is not an optimisation here
       but part of the answer: the trimmed lengths are what the row below is
       sized against, and the early exits depend on them. */
    while (*a && *b && *a == *b) { a++; b++; }
    la = (int)strlen(a); lb = (int)strlen(b);
    while (la && lb && a[la - 1] == b[lb - 1]) { la--; lb--; }
    if (!la || !lb) return (la + lb) * APY_MOVE_COST;
    if (la > APY_MAX_SUGGEST || lb > APY_MAX_SUGGEST) return max_cost + 1;
    if (lb < la) {
        const char *t = a; int tl = la;
        a = b; la = lb; b = t; lb = tl;
    }
    if ((lb - la) * APY_MOVE_COST > max_cost) return max_cost + 1;
    for (k = 0; k < la; k++) row[k] = (k + 1) * APY_MOVE_COST;
    for (i = 0; i < lb; i++) {
        int distance = result = i * APY_MOVE_COST, minimum = -1;
        for (k = 0; k < la; k++) {
            int substitute = distance + apy_sub_cost(b[i], a[k]);
            int best;
            distance = row[k];
            best = (result < distance ? result : distance) + APY_MOVE_COST;
            result = best < substitute ? best : substitute;
            row[k] = result;
            if (minimum < 0 || result < minimum) minimum = result;
        }
        if (minimum >= 0 && minimum > max_cost) return max_cost + 1;
    }
    return result;
}

/* The parameter CPython would propose for a misspelling, or null. NEAREST
   WINS AND TIES KEEP THE FIRST, which is the walk CPython makes. */
static const char *apy_suggest_n(const char *wrong, const char *const *names,
                                 int count) {
    const char *best = 0;
    int best_at = -1, i;
    for (i = 0; i < count; i++) {
        int limit, far;
        if (!names[i]) continue;
        limit = (int)((strlen(wrong) + strlen(names[i]) + 3)
                      * APY_MOVE_COST / 6);
        far = apy_edits(wrong, names[i], limit);
        if (far > limit) continue;
        if (best_at < 0 || far < best_at) { best = names[i]; best_at = far; }
    }
    return best;
}

static const char *apy_suggest(const char *wrong, const char *const *names) {
    return apy_suggest_n(wrong, names, 3);
}

static const apy_ctor_shape *apy_ctor_find(const char *tn) {
    int i;
    for (i = 0; APY_CTORS[i].name; i++)
        if (!strcmp(tn, APY_CTORS[i].name)) return &APY_CTORS[i];
    return 0;
}

/* `list expected at most 1 argument, got 2` -- and `bytes() takes at most 3
   arguments (4 given)`, which is the SAME complaint worded differently. Read
   out of CPython rather than regularised: the two forms differ in every
   visible way, and the parenthesised one is what every type uses once a
   KEYWORD is in the count. */
static apy_value apy_ctor_surplus(const apy_ctor_shape *sh, int64_t argc,
                                  int64_t kwc) {
    char buf[160];
    const char *plural = sh->most == 1 ? "" : "s";
    if (kwc || sh->paren)
        snprintf(buf, sizeof buf, "%s() takes at most %d argument%s "
                 "(%lld given)", sh->name, (int)sh->most, plural,
                 (long long)(argc + kwc));
    else
        snprintf(buf, sizeof buf, "%s expected at most %d argument%s, "
                 "got %lld", sh->name, (int)sh->most, plural,
                 (long long)argc);
    return apy_fail("TypeError", buf);
}

/* The positional forms, once the keywords have been folded into slots. Each
   arm is the same runtime entry point the WRITTEN spelling reaches, so
   `int("ff", 16)` and `type(5)("ff", 16)` are one implementation. */
static apy_value apy_ctor_make(const char *tn, apy_value *argv, int64_t argc) {
    if (argc == 0) {
        apy_value proto;
        if (!strcmp(tn, "bool")) return apy_from_bool(0);
        if (!strcmp(tn, "complex"))
            return apy_complex_of(apy_from_float(0.0), apy_from_float(0.0));
        /* `bytearray()` IS AN EMPTY SEQUENCE OF OCTETS, which is exactly what
           the written zero-argument form lowers to. There is no prototype for
           it -- a bytearray is a bytes cell with a flag -- so the conversion
           is named here rather than left to answer nothing. */
        if (!strcmp(tn, "bytearray")) return apy_to_bytearray(apy_list_new(1));
        proto = apy_kind_prototype((apy_value)(uintptr_t)tn);
        if (proto) return proto;
        return 0;
    }
    if (argc == 1) {
        /* EACH NAMED, because `apy_call_kind` serves five kinds and answers 0
           for the rest -- and a 0 with no error set is a null that segfaults
           downstream rather than reporting. */
        if (!strcmp(tn, "int")) return apy_to_int(argv[0]);
        if (!strcmp(tn, "bool")) return apy_from_bool(apy_truth(argv[0]));
        if (!strcmp(tn, "float")) return apy_to_float(argv[0]);
        if (!strcmp(tn, "str")) return apy_str(argv[0]);
        /* `bytes(xs)` ASKS FOR A LENGTH HINT and `bytearray(xs)` does not:
           CPython builds the immutable one from the iterator in a single
           allocation and grows the mutable one. See `apy_length_hint`. */
        if (!strcmp(tn, "bytes")) {
            if (!apy_length_hint(argv[0])) return 0;
            return apy_to_bytes(argv[0]);
        }
        if (!strcmp(tn, "bytearray")) return apy_to_bytearray(argv[0]);
        if (!strcmp(tn, "frozenset")) return apy_to_frozenset(argv[0]);
        if (!strcmp(tn, "list")) return apy_call_kind(APY_LIST_K, argv[0]);
        if (!strcmp(tn, "tuple")) {
            /* `tuple(t)` ON A TUPLE IS `t`, as the written form already
               answers -- see `_dyn_convert_sequence`. */
            apy_value same = apy_same_tuple(argv[0]);
            if (same) return same;
            return apy_call_kind(APY_TUPLE_K, argv[0]);
        }
        if (!strcmp(tn, "dict")) return apy_call_kind(APY_DICT_K, argv[0]);
        if (!strcmp(tn, "set")) return apy_call_kind(APY_SET_K, argv[0]);
        /* NONE FOR "NOT GIVEN", not the number 0. `complex(x)` asks the
           class through `__complex__`, parses a string and hands a complex
           straight back, and all three are the ONE-argument shape --
           `complex(x, 0)` is building from parts and has nothing to ask.
           The written form already passes None; this is the same
           constructor reached through the name as a value. */
        if (!strcmp(tn, "complex"))
            return apy_complex_of(argv[0], apy_none());
        if (!strcmp(tn, "memoryview")) return apy_memoryview(argv[0]);
    }
    if (argc >= 1 && argc <= 3
            && (!strcmp(tn, "range") || !strcmp(tn, "slice"))) {
        /* `range(stop)` AND `slice(stop)` BOTH PUT THE LONE ARGUMENT IN STOP,
           which is why neither can be a plain positional forward. */
        apy_value start = argc > 1 ? argv[0] : apy_none();
        apy_value stop = argc > 1 ? argv[1] : argv[0];
        apy_value step = argc > 2 ? argv[2] : apy_none();
        if (!strcmp(tn, "slice")) return apy_slice_new(start, stop, step);
        {
            int64_t a = argc > 1 ? apy_index(start) : 0;
            int64_t b = apy_index(stop);
            int64_t c = argc > 2 ? apy_index(step) : 1;
            if (apy_err_type) return 0;
            return apy_range(a, b, c);
        }
    }
    if (argc == 2 && !strcmp(tn, "int"))
        return apy_to_int_base(argv[0], argv[1]);
    if (argc == 2 && !strcmp(tn, "complex"))
        return apy_complex_of(argv[0], argv[1]);
    if (argc >= 2 && argc <= 3 && !strcmp(tn, "str"))
        return apy_str_ctor(argv[0], argv[1],
                            argc > 2 ? argv[2] : apy_none());
    if (argc >= 2 && argc <= 3
            && (!strcmp(tn, "bytes") || !strcmp(tn, "bytearray")))
        return apy_bytes_ctor(argv[0], argv[1],
                              argc > 2 ? argv[2] : apy_none(),
                              !strcmp(tn, "bytearray"));
    return 0;
}

/* The whole call: the keywords folded into slots, then `apy_ctor_make`.
   `*handled` says whether this was a builtin type at all -- 0 leaves the
   caller to go on allocating an instance, which is what a user class wants.
   A handled call answers a value, or 0 with the failure already reported. */
static apy_value apy_builtin_ctor(const char *tn, apy_value *argv,
                                  int64_t argc, apy_value kwrest,
                                  int *handled) {
    const apy_ctor_shape *sh = apy_ctor_find(tn);
    apy_value filled[3];
    int taken[3];
    int64_t kwc = 0, i, j, top;
    const char *names[3];
    *handled = 0;
    if (!sh) return 0;
    *handled = 1;
    kwc = kwrest ? O(kwrest)->v.d.n : 0;
    names[0] = sh->slot0; names[1] = sh->slot1; names[2] = sh->slot2;
    /* `dict` TAKES ANY KEYWORD AT ALL -- they become the mapping's own keys,
       so there is nothing to fold, only a count to check. */
    if (!strcmp(tn, "dict")) {
        apy_value made;
        if (argc > 1) return apy_ctor_surplus(sh, argc, 0);
        made = argc ? apy_call_kind(APY_DICT_K, argv[0]) : apy_dict_new(1);
        if (!made) return 0;
        if (kwc && !apy_update(made, kwrest)) return 0;
        return made;
    }
    /* THE ORDER OF THESE REFUSALS IS CPYTHON'S. A type that takes NO keyword
       at all says exactly that and names none of them, ahead of every other
       complaint. */
    if (kwc && !names[0] && !names[1] && !names[2]) {
        char buf[96];
        snprintf(buf, sizeof buf, "%s() takes no keyword arguments", tn);
        return apy_fail("TypeError", buf);
    }
    for (i = 0; i < 3; i++) { filled[i] = 0; taken[i] = 0; }
    for (i = 0; i < argc && i < 3; i++) { filled[i] = argv[i]; taken[i] = 1; }
    /* THEN A MISSING REQUIRED FIRST ARGUMENT, in the type's own wording. */
    if (argc < sh->least) {
        int named_first = 0;
        for (i = 0; i < kwc; i++) {
            apy_value k = O(kwrest)->v.d.keys[i];
            if (O(k)->kind == APY_STR_K && names[0]
                    && !strcmp(APY_CSTR(k), names[0])) named_first = 1;
        }
        if (!named_first) return apy_fail("TypeError", sh->missing);
    }
    if (argc + kwc > sh->most) {
        if (argc == 0 && kwc) {
            char buf[128];
            snprintf(buf, sizeof buf, "%s() takes at most %d keyword "
                     "argument%s (%lld given)", tn, (int)sh->most,
                     sh->most == 1 ? "" : "s", (long long)kwc);
            return apy_fail("TypeError", buf);
        }
        return apy_ctor_surplus(sh, argc, kwc);
    }
    for (i = 0; i < kwc; i++) {
        apy_value k = O(kwrest)->v.d.keys[i];
        int at = -1;
        if (O(k)->kind == APY_STR_K)
            for (j = 0; j < 3; j++)
                if (names[j] && !strcmp(APY_CSTR(k), names[j])) at = (int)j;
        if (at < 0) {
            char buf[200];
            const char *wrong = O(k)->kind == APY_STR_K ? APY_CSTR(k) : "?";
            const char *near = apy_suggest(wrong, names);
            if (near)
                snprintf(buf, sizeof buf, "%s() got an unexpected keyword "
                         "argument '%.60s'. Did you mean '%s'?",
                         tn, wrong, near);
            else
                snprintf(buf, sizeof buf, "%s() got an unexpected keyword "
                         "argument '%.60s'", tn, wrong);
            return apy_fail("TypeError", buf);
        }
        if (taken[at]) {
            char buf[160];
            snprintf(buf, sizeof buf, "argument for %s() given by name "
                     "('%s') and position (%d)", tn, names[at], at + 1);
            return apy_fail("TypeError", buf);
        }
        filled[at] = O(kwrest)->v.d.vals[i];
        taken[at] = 1;
    }
    /* `errors` WITHOUT AN `encoding` is its own refusal for the two byte
       constructors, whatever the source is. */
    if (taken[2] && !taken[1]
            && (!strcmp(tn, "bytes") || !strcmp(tn, "bytearray")))
        return apy_fail("TypeError", "errors without a string argument");
    if (!taken[0] && (taken[1] || taken[2])) {
        if (!strcmp(tn, "complex")) {
            filled[0] = apy_from_int(0);
            taken[0] = 1;
        } else if (sh->headless) {
            return apy_fail("TypeError", sh->headless);
        } else {
            /* `str(encoding="x")` IS `''`. There is nothing to decode, and
               CPython answers the empty string rather than refusing. */
            return apy_ctor_make(tn, filled, 0);
        }
    }
    /* NONE MEANS "THE DEFAULT" to the codec pair, which is the padding
       `.encode()` and `.decode()` already take. */
    top = 0;
    for (i = 0; i < 3; i++) if (taken[i]) top = i + 1;
    for (i = 0; i < top; i++) if (!taken[i]) filled[i] = apy_none();
    {
        apy_value out = apy_ctor_make(tn, filled, top);
        /* A SHAPE NOBODY SERVES is not a failure: it hands the call back to
           the caller, which goes on allocating an instance exactly as it did
           before any of this existed. A 0 with the flag UP is a real refusal
           and travels as one. */
        if (!out && !apy_err_type) *handled = 0;
        return out;
    }
}

/* `"aaa".replace("a", "b", **opts)` -- WHAT THE MAPPING HOLDS, checked where
   it is finally known.

   A `**` MAPPING CANNOT BE FOLDED AT COMPILE TIME: its keys are a run-time
   value, so the frontend used to refuse any mapping with something in it and
   answer only the empty case. That refused `f(*args, **kwargs)` forwarding
   through a wrapper, which is the shape the whole feature exists for.

   THE FOLD IS THE FRONTEND'S STILL -- it emits one `apy_dict_get_or` per
   parameter, against the names `METHOD_PARAMS` already records -- and this
   is the half that cannot be: whether the mapping holds a key that is NOT a
   parameter, or one naming a slot a positional already filled. The wordings
   are CPython's, the same ones `fold_keywords` raises for the written form.

   `names` IS THE PARAMETER LIST IN POSITIONAL ORDER, with an EMPTY STRING
   standing for a parameter CPython marks positional-only: a key can never
   name one of those, and leaving a hole in the tuple would misalign every
   slot after it. */
/* `"abc".upper(**opts)` -- a method that takes NO keyword at all, handed a
   mapping that may hold one. CPython names the OWNER here -- `str.upper()
   takes no keyword arguments` -- and the owner is the receiver's kind, which
   is not known until now: the frontend used to say "does not take a
   non-empty ** mapping in this compiler" instead, which is a limitation
   admitted out loud rather than an answer.

   THE EMPTY CASE IS THE COMMON ONE and passes straight through: a wrapper
   forwarding `*args, **kwargs` passes an empty mapping nearly always. */
APY_API apy_value apy_kw_owner(apy_value recv, apy_value methv) {
    char buf[160];
    snprintf(buf, sizeof buf, "%s.%s() takes no keyword arguments",
             apy_kind_name(recv), APY_CSTR(methv));
    return apy_fail("TypeError", buf);
}

APY_API apy_value apy_kw_none(apy_value recv, apy_value methv, apy_value bag) {
    if (!apy_truth(bag)) return apy_none();
    return apy_kw_owner(recv, methv);
}

/* `f(**{"k": v}, k=w)` -- THE SAME NAME TWICE, once through a mapping and
   once written out. CPython refuses it whichever order the two come in, and
   NAMES THE OWNER: `str.split() got multiple values for keyword argument
   'sep'`. Two WRITTEN names cannot collide -- that is a SyntaxError -- so any
   repeat in the bag-building order is this.
   OFFERED ONE KEY AT A TIME, because a merge cannot see it: the later key
   simply wins, and `"a,b".split(**{"sep": ","}, sep=";")` answered `['a,b']`
   with nothing to mark it. */
static apy_value apy_kw_twice(apy_value recv, apy_value methv,
                              apy_value key) {
    char buf[200];
    snprintf(buf, sizeof buf, "%s.%s() got multiple values for keyword "
             "argument '%.60s'", apy_kind_name(recv), APY_CSTR(methv),
             APY_CSTR(key));
    return apy_fail("TypeError", buf);
}

APY_API apy_value apy_kw_put(apy_value bag, apy_value key, apy_value val,
                             apy_value recv, apy_value methv) {
    if (apy_dict_find(bag, key) >= 0) return apy_kw_twice(recv, methv, key);
    if (!apy_dict_set(bag, key, val)) return 0;
    return apy_none();
}

APY_API apy_value apy_kw_merge(apy_value bag, apy_value src, apy_value recv,
                               apy_value methv) {
    int64_t i;
    if (O(src)->kind != APY_DICT_K) {
        /* NOT A MAPPING AT ALL, which `**` requires. Left to `apy_update`,
           which already words it -- this only has to not crash reading a
           `v.d` that is not there. */
        if (!apy_update(bag, src)) return 0;
        return apy_none();
    }
    for (i = 0; i < O(src)->v.d.n; i++) {
        apy_value k = O(src)->v.d.keys[i];
        if (O(k)->kind == APY_STR_K && apy_dict_find(bag, k) >= 0)
            return apy_kw_twice(recv, methv, k);
        if (!apy_dict_set(bag, k, O(src)->v.d.vals[i])) return 0;
    }
    return apy_none();
}

APY_API apy_value apy_kw_check(apy_value bag, apy_value names,
                               apy_value methv, int64_t argc) {
    const char *meth = APY_CSTR(methv);
    const char *known[8];
    int64_t count = O(names)->v.q.n, i, j;
    char buf[220];
    if (count > 8) count = 8;
    for (i = 0; i < count; i++) known[i] = APY_CSTR(O(names)->v.q.items[i]);
    /* TOO MANY BEATS EVERY OTHER COMPLAINT, counting the keywords in:
       `"a,b".split(",", 1, **{"maxsplit": 2})` is three arguments for two
       parameters, and CPython says so rather than reporting that `maxsplit`
       was given twice. A call with no positionals at all is worded as
       KEYWORD arguments -- the same split `fold_keywords` makes. */
    if (argc + O(bag)->v.d.n > count) {
        const char *plural = count == 1 ? "" : "s";
        if (argc == 0)
            snprintf(buf, sizeof buf, "%s() takes at most %lld keyword "
                     "argument%s (%lld given)", meth, (long long)count,
                     plural, (long long)O(bag)->v.d.n);
        else
            snprintf(buf, sizeof buf, "%s() takes at most %lld argument%s "
                     "(%lld given)", meth, (long long)count, plural,
                     (long long)(argc + O(bag)->v.d.n));
        return apy_fail("TypeError", buf);
    }
    for (i = 0; i < O(bag)->v.d.n; i++) {
        apy_value k = O(bag)->v.d.keys[i];
        const char *wrong;
        int64_t at = -1;
        if (O(k)->kind != APY_STR_K)
            return apy_fail("TypeError", "keywords must be strings");
        wrong = APY_CSTR(k);
        for (j = 0; j < count; j++)
            if (*known[j] && !strcmp(wrong, known[j])) at = j;
        if (at < 0) {
            const char *near = apy_suggest_n(wrong, known, (int)count);
            /* A POSITIONAL-ONLY NAME IS NOT A SUGGESTION, and cannot be one:
               its slot is the empty string above, so `apy_suggest_n` never
               answers it. */
            if (near && *near)
                snprintf(buf, sizeof buf, "%s() got an unexpected keyword "
                         "argument '%.60s'. Did you mean '%s'?",
                         meth, wrong, near);
            else
                snprintf(buf, sizeof buf, "%s() got an unexpected keyword "
                         "argument '%.60s'", meth, wrong);
            return apy_fail("TypeError", buf);
        }
        if (at < argc) {
            snprintf(buf, sizeof buf, "argument for %s() given by name "
                     "('%.60s') and position (%lld)", meth, wrong,
                     (long long)(at + 1));
            return apy_fail("TypeError", buf);
        }
    }
    return apy_none();
}

/* A BUILTIN TYPE REACHED AS A VALUE: `f = int` then `f("ff", 16)`,
   `map(int, xs)`, `defaultdict(list)`, `sorted(xs, key=str)`.

   THE THUNK `_dyn_builtin_value` SYNTHESISES CALLS EXACTLY THIS, so the
   value form, the written form and the type-object form are ONE
   implementation. The thunk used to take a single argument with a default,
   which meant `f("ff", 16)` silently dropped the base and `f(x=1)` silently
   dropped the keyword -- wrong answers with nothing to mark them. */
APY_API apy_value apy_ctor_call(apy_value namev, apy_value args,
                                apy_value kwrest) {
    const char *tn = APY_CSTR(namev);
    int handled = 0;
    apy_value out;
    /* AN EMPTY BAG IS NO BAG. The thunk always declares the `**kw` slot, so
       an ordinary `list(xs)` arrives with an empty dict in it and the fold
       must not read that as a keyword given. */
    if (kwrest && (O(kwrest)->kind != APY_DICT_K || !O(kwrest)->v.d.n))
        kwrest = 0;
    out = apy_builtin_ctor(tn, (apy_value *)O(args)->v.q.items,
                           O(args)->v.q.n, kwrest, &handled);
    if (handled) return out;
    /* UNREACHABLE while the thunk is only built for the names the table
       holds, and kept as the backstop it is. */
    {
        char buf[96];
        snprintf(buf, sizeof buf, "%s() takes no arguments", tn);
        return apy_fail("TypeError", buf);
    }
}

static apy_value apy_instantiate(apy_value f, apy_value *argv, int64_t argc,
                                 apy_value kwrest, int bound) {
    apy_value self;
    apy_value init;
    apy_value maker = apy_class_find(f, apy_name("__new__"));
    /* A BUILTIN TYPE OBJECT IS CALLABLE. `type(5)()` is `0` and
       `x.__class__()` is an empty one of whatever `x` is -- both are the
       ordinary way to make a fresh value of a type you were handed, and a
       generic copy or merge helper does exactly that.

       A TYPE INVENTED FOR A KIND THE PROGRAM NEVER NAMED had no constructor
       at all: it allocated an INSTANCE of a nameless class and answered
       `<int object at 0x...>` for `type(5)()`. Whether it worked depended on
       whether the module happened to write the bare word `int` somewhere,
       because only then is a canonical thunk registered for it.

       THROUGH THE PROTOTYPE, which already knows which kind each builtin
       type name stands for, and then through the same `apy_call_kind` a
       class extending a builtin uses. Only for a type with NOTHING OF ITS
       OWN -- no `__new__`, no `__init__`, no metaclass -- which is exactly
       the shape `apy_type_new` invents and nothing a program writes. */
    if (O(f)->kind == APY_TYPE_K && !maker && !O(f)->v.t.meta
            && !apy_class_find(f, apy_name("__init__"))) {
        /* THE KIND CHECK IS NOT REDUNDANT. A `class` statement lowers to a
           FUNC that carries `is_type`, and `type.__call__` hands one here --
           so reading `v.t.name` without asking would read a FUNC's code
           pointer as a string, which segfaults rather than answering. */
        const char *tn = APY_CSTR(O(f)->v.t.name);
        int handled = 0;
        apy_value made = apy_builtin_ctor(tn, argv, argc, kwrest, &handled);
        /* BY NAME AND NOT BY THE PROTOTYPE'S KIND. `bool` and `int` share a
           prototype -- the attribute question cannot tell them apart and does
           not need to -- and `type(True)()` is `False`, not `0`. */
        if (handled) return made;
    }
    if (maker) {
        /* `__new__` IS AN IMPLICIT STATICMETHOD: it receives the CLASS as
           its first argument, not an instance, so it is called unbound
           with the class pushed in front. It was ignored entirely before
           -- the instance was allocated and `__new__` never ran, which is
           a wrong answer rather than a missing feature. */
        apy_value pushed[17];
        int64_t j;
        pushed[0] = f;
        for (j = 0; j < argc && j + 1 < 17; j++) pushed[j + 1] = argv[j];
        self = apy_call_nk(maker, pushed,
                           argc + 1 < 17 ? argc + 1 : 17, kwrest, 0);
        if (!self) return 0;
        /* `__init__` RUNS ONLY IF `__new__` RETURNED ONE OF THESE.
           Returning something else is how a `__new__` deliberately
           bypasses initialisation, and CPython honours that. */
        /* `__init__` RUNS WHEN `__new__` ANSWERED AN INSTANCE OF THIS
           CLASS -- and for a METACLASS the thing it answered is a class
           whose metaclass is this one, which is the same test through
           `apy_type_of`. Comparing only against `v.o.cls` skipped a
           metaclass's `__init__` entirely. */
        if (apy_type_of(self) != f)
            return self;
    } else {
        self = apy_instance_new(f);
        if (!self) return 0;
    }
    init = apy_class_find(f, apy_name("__init__"));
    if (init) {
        apy_value bound_init = apy_bind(init, self);
        /* A NATIVE `__init__` TAKES NO KEYWORDS. `type.__init__` is the one
           that matters: `class C(metaclass=M, kind="x")` hands the keywords
           to `M.__new__`, and CPython's `type.__init__` ignores them rather
           than reporting one it does not declare. */
        if (O(init)->kind == APY_FUNC_K && O(init)->v.fn.native) kwrest = 0;
        if (!apy_call_nk(bound_init, argv, argc, kwrest, bound)) return 0;
    } else if ((argc != 0 || kwrest) && !maker) {
        /* A CLASS EXTENDING A BUILTIN INHERITS ITS CONSTRUCTOR. `class
           L(list): pass` then `L([1, 2, 3])` is a list of three, because
           `list.__init__` is what the empty body left in place -- and
           `L() takes no arguments` names the wrong thing entirely: the class
           HAS a constructor, inherited, and the arguments are what it wants.

           ONE ARGUMENT, which is every builtin constructor that fills from
           something: `dict(pairs)`, `list(it)`, `tuple(it)`, `set(it)`. The
           keyword forms (`dict(a=1)`) reach `__init__` on a class that wrote
           one; a class that did not gets the refusal it had before.

           NONE OF THIS WHEN THE CLASS WROTE `__new__` -- see the `!maker` on
           the branch above. That constructor has already decided what the
           instance holds (a `namedtuple` packs its arguments into one tuple),
           and CPython draws the same line: `object.__init__` complains about
           surplus arguments only when `__new__` is not overridden. */
        if (O(self)->kind == APY_INST_K && O(self)->v.o.held && argc == 1) {
            apy_value made = apy_call_kind(
                O(O(self)->v.o.held)->kind, argv[0]);
            if (!made) return 0;
            O(self)->v.o.held = made;
            if (kwrest && O(made)->kind == APY_DICT_K
                    && !apy_update(made, kwrest)) return 0;
            return self;
        }
        /* KEYWORDS ALONE, which `dict` is the whole reason for: `class
           C(dict): pass` then `C(a=1)` has NO positional argument, so the
           branch above never ran and the guard on this one used to send it
           straight past -- the instance came back with an EMPTY dict and no
           error at all, which is worse than the refusal the comment above
           promises. */
        if (O(self)->kind == APY_INST_K && O(self)->v.o.held && argc == 0
                && kwrest && O(O(self)->v.o.held)->kind == APY_DICT_K) {
            if (!apy_update(O(self)->v.o.held, kwrest)) return 0;
            return self;
        }
        char buf[128];
        snprintf(buf, sizeof buf,
                 "%s() takes no arguments", APY_CSTR(O(f)->v.t.name));
        return apy_fail("TypeError", buf);
    }
    return self;
}

static apy_value apy_call_nk(apy_value f, apy_value *argv, int64_t argc,
                             apy_value kwrest, int bound) {
    apy_value slots[17];
    int64_t i, n = 0;

    if (O(f)->kind == APY_TYPE_K && O(f)->v.t.meta) {
        /* THE METACLASS DECIDES WHAT CALLING THE CLASS DOES, if it says so:
           `type(C).__call__(C, ...)` is what `C(...)` means, and it is how
           `ABCMeta` refuses to instantiate a class with abstract methods.
           Only when a `__call__` is actually written -- the default is the
           allocate-and-init below, and routing every class through a lookup
           that almost never finds anything would cost every instantiation. */
        apy_value hook = apy_class_find(O(f)->v.t.meta, apy_name("__call__"));
        if (hook) {
            apy_value pushed[17];
            int64_t j;
            pushed[0] = f;
            for (j = 0; j < argc && j + 1 < 17; j++) pushed[j + 1] = argv[j];
            return apy_call_nk(hook, pushed, argc + 1 < 17 ? argc + 1 : 17,
                               kwrest, 0);
        }
    }
    /* AN EXCEPTION TYPE IS CALLABLE, and reaching it through a variable is
       the only way to notice that it was not. `ValueError("v")` is resolved
       at the CALL SITE by the frontend and never arrives here; `c =
       ValueError; c("v")` does, and answered `ValueError() takes no
       arguments` -- about a class every program constructs.

       `warnings.warn` is why this surfaced: `raise category(message)` holds
       the class in a parameter, so the entire module was unwritable. A type
       that can only be called by the spelling the compiler recognises is not
       a value, and every library that takes an exception class as an
       argument depends on it being one.

       A type carries no argument; an INSTANCE does. That is the whole
       distinction here, and it is what stops `e = ValueError("v"); e()` from
       being read as a second construction. */
    if (O(f)->kind == APY_EXC_K && !O(f)->v.e.has_arg && !O(f)->v.e.argv)
        return apy_make_excn(apy_lit(O(f)->v.e.name), (apy_value)argv, argc);
    /* AND THE TYPE OBJECT THE NAME ACTUALLY ANSWERS, which is not an
       `APY_EXC_K` at all. `apy_exc_type` builds one and immediately hands it
       to `apy_type_of`, so what a program holds when it writes `c =
       ValueError` is a plain `APY_TYPE_K` with an empty dict -- and the test
       above, which reads exactly right, never fired once.

       That is why this is a SECOND check and not a widening of the first: the
       interpreter's `objects_host` keeps the exception cell and matched on
       it, so the same source ran correctly there and failed here. Two paths
       agreeing on the language and disagreeing on which object a name holds
       is the failure this whole runtime is arranged to make visible, and it
       stayed invisible because nothing constructed an exception through a
       variable on the compiled path until `warnings.warn` did.

       GUARDED ON THE CLASS BEING EMPTY. A type with `__init__` or `__new__`
       is a class the program wrote, and `apy_exc_class_named` finds the ones
       it wrote by subclassing an exception; either way it means its own and
       `apy_instantiate` is right for it. */
    if (O(f)->kind == APY_TYPE_K && apy_type_is_exc(f))
        return apy_make_excn(O(f)->v.t.name, (apy_value)argv, argc);
    if (O(f)->kind == APY_TYPE_K)
        return apy_instantiate(f, argv, argc, kwrest, bound);
    if (O(f)->kind == APY_FUNC_K
            && O(f)->v.fn.native == APY_NAT_TYPE_CALL) {
        /* `type.__call__(cls, ...)` -- THE ORDINARY INSTANTIATION, with the
           metaclass hook deliberately skipped. This is what a metaclass's
           `__call__` ends with, and consulting the hook again from here
           would be that same `__call__` calling itself forever. */
        apy_value cls = O(f)->v.fn.bound;
        if (cls) return apy_instantiate(cls, argv, argc, kwrest, 0);
        if (argc < 1)
            return apy_fail("TypeError",
                            "type.__call__() takes at least 1 argument");
        return apy_instantiate(argv[0], argv + 1, argc - 1, kwrest, 0);
    }
    if (O(f)->kind == APY_INST_K) {
        /* A callable instance: `x(...)` is `type(x).__call__(x, ...)`. */
        apy_value m = apy_class_find(O(f)->v.o.cls, apy_name("__call__"));
        if (!m)
            return apy_fail2("TypeError", "'%s' object is not callable%s",
                             apy_kind_name(f), "");
        return apy_call_nk(apy_bind(m, f), argv, argc, kwrest, bound);
    }
    if (O(f)->kind != APY_FUNC_K)
        return apy_fail2("TypeError", "'%s' object is not callable%s",
                         apy_kind_name(f), "");

    if (O(f)->v.fn.bound) slots[n++] = O(f)->v.fn.bound;
    {   /* `*rest` collects everything past the declared parameters, and it
           occupies one argument slot of its own after them. Packed here
           rather than by the caller, which does not know the callee has
           one. */
        int64_t declared = O(f)->v.fn.arity - (O(f)->v.fn.vararg ? 1 : 0)
                                             - (O(f)->v.fn.kwarg ? 1 : 0);
        /* WHERE POSITIONS STOP. A keyword-only parameter is declared but not
           reachable by position, so a surplus argument belongs to `*rest` --
           or is an error -- rather than landing in it. */
        /* `bound` means the caller ALREADY matched names to slots, so every
           argument here belongs where it is. Re-applying the keyword-only
           limit would truncate a list `apy_call_kw` had just completed and
           then refill the tail from defaults, silently discarding the values
           the keywords supplied. */
        int64_t byslot = bound ? declared : declared - O(f)->v.fn.kwonly;
        int64_t take = argc;
        if (n + argc > byslot) take = byslot - n;
        if (take < 0) take = 0;
        for (i = 0; i < take && n < 17; i++) slots[n++] = argv[i];
        /* A missing trailing argument comes from the default the `def`
           evaluated, which lives in the function object -- see the comment on
           `fn` in `struct apy_obj`. */
        while (n < declared) {
            apy_value dv = apy_fn_default(f, n);
            if (!dv) break;
            slots[n++] = dv;
        }
        if (O(f)->v.fn.vararg) {
            apy_value rest = apy_tuple_new(argc - take + 1);
            for (i = take; i < argc; i++) apy_seq_push(rest, argv[i]);
            if (n < 17) slots[n++] = rest;
        }
        /* `**kw` is the LAST parameter and is passed even when empty: `def
           f(**kw)` called as `f()` binds `{}`, not nothing. */
        if (O(f)->v.fn.kwarg && n < 17)
            slots[n++] = kwrest ? kwrest : apy_dict_new(1);
    }
    if (O(f)->v.fn.native == APY_NAT_KIND
            || (O(f)->v.fn.native && !O(f)->v.fn.defaults
                && O(f)->v.fn.ndefaults)) {
        /* A BUILTIN METHOD TAKES A RANGE, and the body reads the count it
           really got -- which is how ONE selector tells `find(x)` from
           `find(x, i)`.

           COUNTED FROM `argc` AND NOT FROM `n`, because the packing above
           CAPS `n` at the declared arity: comparing the capped number is
           exactly how a surplus argument came to be dropped in silence, so
           `getattr([1], "__len__")(9)` answered 1 instead of refusing.

           THE OTHER SELECTORS ARE LEFT ALONE. `__get__` is declared with two
           and called with three, and the descriptor protocol has relied on
           the capping since it was written; only the kind methods and any
           native that DECLARES a tail are held to a count here. */
        int64_t most = O(f)->v.fn.arity;
        int64_t given = (O(f)->v.fn.bound ? 1 : 0) + argc;
        /* A VARIADIC ONE HAS NO COUNT TO CHECK: `"{}".format(*xs)` takes
           whatever it is handed, and the packing above has already put the
           surplus in `rest` and the keywords in `kw`. So the slots are the
           declared three and the count is `n`, not what arrived. */
        if (O(f)->v.fn.vararg) return apy_invoke(f, slots, n);
        /* THE POSITIONAL BOUND, for the two methods whose is narrower than
           their total. Only where the caller has NOT already matched names
           to slots, because after that the count says nothing about how the
           call was written. See `apy_meth_positional`. */
        if (!bound && !kwrest && O(f)->v.fn.native == APY_NAT_KIND) {
            apy_value refused = apy_meth_positional(f, argc);
            if (refused) return refused;
            if (apy_error_occurred()) return 0;
        }
        if (given < most - O(f)->v.fn.ndefaults || given > most)
            return apy_arity_error(f, argc);
        return apy_invoke(f, slots, given);
    }
    /* A SURPLUS POSITIONAL IS AN ERROR, not something to drop. The packing
       above CAPS what it copies at the positional capacity, so `n` below can
       only ever come out too SMALL -- which is exactly how `two(0, 1, 2)`
       answered `(0, 1)` instead of refusing, on every path there is. A
       `*rest` is what a surplus is FOR, and `bound` says the caller matched
       names to slots and counted them itself, so neither is one. */
    if (!bound && !O(f)->v.fn.vararg) {
        int64_t byslot = O(f)->v.fn.arity - (O(f)->v.fn.kwarg ? 1 : 0)
                         - O(f)->v.fn.kwonly;
        int64_t given = (O(f)->v.fn.bound ? 1 : 0) + argc;
        if (given > byslot) return apy_fn_arity_error(f, given, 0);
    }
    if (n != O(f)->v.fn.arity)
        return bound ? apy_arity_error(f, argc)
                     : apy_fn_arity_error(f, (O(f)->v.fn.bound ? 1 : 0) + argc,
                                          0);
    return apy_invoke(f, slots, n);
}


static apy_value apy_call_n(apy_value f, apy_value *argv,
                            int64_t argc) {
    return apy_call_nk(f, argv, argc, 0, 0);
}

/* The frontend's entry point: `argv` is the ADDRESS of an array of values in
   a stack slot, the same shape `apy_print` takes and for the same reason --
   the IR has no varargs. */
APY_API apy_value apy_call(apy_value f, apy_value argv, int64_t argc) {
    return apy_call_n(f, (apy_value *)argv, argc);
}

/* The FUNC_K object a call will actually ENTER, and how many of its declared
   parameters the caller does not supply -- 1 for the `self` of a method or of
   a class's `__init__`, 0 otherwise. Zero when there is nothing to enter.

   Keyword resolution needs this and a plain call does not: `C(n=1)` names a
   parameter of `C.__init__`, so the names have to be read off that function
   and matched against positions shifted past `self`. */
static apy_value apy_call_target(apy_value f, int64_t *skip) {
    *skip = 0;
    if (O(f)->kind == APY_TYPE_K) {
        /* `__new__` DECLARES THE KEYWORDS when a class writes one and leaves
           `__init__` to the default -- which is exactly a metaclass taking
           class keywords: `M.__new__(mcls, name, bases, ns, kind=None)` with
           `type.__init__` behind it. Reading the names off `__init__` there
           matched them against a native that declares none. */
        apy_value init = apy_class_find(f, apy_name("__init__"));
        apy_value maker = apy_class_find(f, apy_name("__new__"));
        if ((!init || O(init)->v.fn.native) && maker
                && O(maker)->kind == APY_FUNC_K && !O(maker)->v.fn.native) {
            *skip = 1;
            return maker;
        }
        if (!init || O(init)->kind != APY_FUNC_K || O(init)->v.fn.native)
            return 0;
        *skip = 1;
        return init;
    }
    if (O(f)->kind == APY_INST_K) {
        apy_value m = apy_class_find(O(f)->v.o.cls, apy_name("__call__"));
        if (!m || O(m)->kind != APY_FUNC_K) return 0;
        *skip = 1;
        return m;
    }
    if (O(f)->kind != APY_FUNC_K) return 0;
    if (O(f)->v.fn.bound) *skip = 1;
    return f;
}

/* `f(a, b, k=v, **d)`. The positional arguments are in `buf`; the keyword
   ones are a DICT, built at the call site, because `**d` merges a dict whose
   keys are not known until it exists -- a compile-time list of names could
   not express that and a second entry point for it would duplicate all of the
   resolution below.

   The keywords are placed into their PARAMETER POSITIONS here and a complete
   argument list is handed to `apy_call_n`, so everything downstream -- the
   arity check, `*rest`, a bound receiver -- stays the one implementation it
   already was. Defaults are filled here too, because a keyword can leave a
   HOLE in the middle (`f(1, c=3)` against `def f(a, b=2, c=3)`) and
   `apy_call_n` only knows how to fill a missing TAIL. */
APY_API apy_value apy_call_kw(apy_value f, apy_value buf, int64_t argc,
                              apy_value kwd) {
    apy_value *raw = (apy_value *)buf;
    apy_value slots[17], rest = 0;
    char filled[17];
    int64_t skip = 0, declared, want, bypos, i, k, kwn;
    int64_t kwonly_given = 0;
    const char *who;
    apy_value target = apy_call_target(f, &skip);

    if (!target)
        /* No signature to match against: a class with no `__init__`, or a
           value that is not callable at all. `apy_call_nk` words both of
           those, so let it.
           THROUGH `apy_call_nk` AND NOT `apy_call_n`, so the keywords SURVIVE.
           `class C(dict): pass` then `C(a=1)` has no `__init__` to match
           against and used to arrive here, drop `kwd` on the floor, and hand
           `apy_instantiate` nothing at all -- the instance came back with an
           empty dict and no error, which is a wrong answer where a refusal
           was intended. */
        return apy_call_nk(f, raw, argc, kwd, 0);
    kwn = apy_raw_len(kwd);
    /* A BUILTIN METHOD REACHED AS A VALUE, handed keywords. `f = x.split`
       then `f(",", maxsplit=1)` is a call CPython answers, and so is the one
       the COLLISION path makes when a module defines a class extending a
       builtin: `(5).to_bytes(2, byteorder="big")` arrives here rather than
       through the frontend's own fold. The binder below matches names
       against `pnames`, which a native has none of, so every one of them
       reported `got an unexpected keyword argument` for a call that works.

       THE SAME TABLE THE WRITTEN SPELLING FOLDS AGAINST. `apy_kind_meth_sign`
       is generated from `METHOD_PARAMS`, so the two arrangements cannot
       drift; the refusals below are CPython's and in CPython's order, the
       way `apy_kw_check` puts them for a `**` mapping.

       HERE RATHER THAN ON THE CALLABLE, because `apy_kind_method_opt` is
       replaced by its IR twin -- names installed there would reach the C
       build and not the ported one. */
    /* A VARIADIC BUILTIN COLLECTS ITS KEYWORDS rather than matching them
       against declared names: `"{a}".format(a=1)` has no parameter called
       `a`, and the dict is the argument. So it skips the matching below and
       lets `apy_call_nk`'s packing put the dict in the `**kw` slot. */
    if (kwn && O(f)->kind == APY_FUNC_K
            && O(f)->v.fn.native == APY_NAT_KIND
            && !O(f)->v.fn.vararg) {
        apy_value names[4], defs[4], slot[4];
        char taken[4];
        const char *meth = APY_CSTR(O(f)->v.fn.name);
        apy_value self = O(f)->v.fn.bound;
        int64_t np = apy_kind_meth_sign(meth, names, defs);
        int64_t at, k2, i2, top = 0;
        char b[200];
        /* `d.update(a=1)` -- THE KEYWORDS ARE THE VALUE, and no signature
           can say that: any name at all becomes a key. So the dict the
           caller built IS the argument, applied after a positional mapping
           if there was one. A SET'S `update` really does take no keyword,
           which is why this is the dict's alone. */
        if (strcmp(meth, "update") == 0 && self
                && O(self)->kind == APY_DICT_K) {
            if (argc && !apy_update(self, raw[0])) return 0;
            return apy_update(self, kwd);
        }
        if (!np || np != O(f)->v.fn.arity - 1)
            return apy_kw_owner(self ? self : (argc ? raw[0] : apy_none()),
                                O(f)->v.fn.name);
        for (i2 = 0; i2 < np; i2++) { slot[i2] = 0; taken[i2] = 0; }
        for (i2 = 0; i2 < argc && i2 < np; i2++) {
            slot[i2] = raw[i2];
            taken[i2] = 1;
        }
        /* TOO MANY BEATS EVERY OTHER COMPLAINT, counting the keywords in. */
        if (argc + kwn > np) {
            snprintf(b, sizeof b, "%s() takes at most %lld argument%s "
                     "(%lld given)", meth, (long long)np, np == 1 ? "" : "s",
                     (long long)(argc + kwn));
            return apy_fail("TypeError", b);
        }
        for (k2 = 0; k2 < kwn; k2++) {
            apy_value nm = O(kwd)->v.d.keys[k2];
            at = -1;
            if (O(nm)->kind != APY_STR_K)
                return apy_fail("TypeError", "keywords must be strings");
            for (i2 = 0; i2 < np; i2++)
                if (names[i2] && !strcmp(APY_CSTR(nm), APY_CSTR(names[i2])))
                    at = i2;
            if (at < 0) {
                snprintf(b, sizeof b, "%s() got an unexpected keyword "
                         "argument '%.60s'", meth, APY_CSTR(nm));
                return apy_fail("TypeError", b);
            }
            if (taken[at]) {
                snprintf(b, sizeof b, "argument for %s() given by name "
                         "('%.60s') and position (%lld)", meth, APY_CSTR(nm),
                         (long long)(at + 1));
                return apy_fail("TypeError", b);
            }
            slot[at] = O(kwd)->v.d.vals[k2];
            taken[at] = 1;
        }
        for (i2 = 0; i2 < np; i2++) if (taken[i2]) top = i2 + 1;
        for (i2 = 0; i2 < top; i2++)
            if (!taken[i2]) {
                if (!defs[i2]) {
                    snprintf(b, sizeof b, "%s() takes at least %lld "
                             "positional argument%s (%lld given)", meth,
                             (long long)(i2 + 1), i2 ? "s" : "",
                             (long long)argc);
                    return apy_fail("TypeError", b);
                }
                slot[i2] = defs[i2];
            }
        /* PADDED TO THE FULL ARITY, which is what the frontend's own fold
           does for the written spelling: the entry point takes every
           parameter and the defaults supply the rest. */
        for (i2 = top; i2 < np; i2++)
            slot[i2] = defs[i2] ? defs[i2] : apy_none();
        /* `bound` SAYS THE NAMES ARE ALREADY IN THEIR SLOTS, which is what
           keeps the POSITIONAL bound from being re-applied to a call that
           filled those slots by name: `(5).to_bytes(length=4)` arrives three
           slots wide and wrote no positional at all. See
           `apy_meth_positional`. */
        return apy_call_nk(f, slot, np, 0, 1);
    }
    declared = O(target)->v.fn.arity - (O(target)->v.fn.vararg ? 1 : 0)
                                     - (O(target)->v.fn.kwarg ? 1 : 0);
    want = declared - skip;
    if (want < 0) want = 0;
    if (want > 17) want = 17;
    /* Positions reach only as far as the keyword-only tail; names reach all
       of it, which is why `want` stays whole and only `bypos` shrinks. */
    bypos = want - O(target)->v.fn.kwonly;
    if (bypos < 0) bypos = 0;
    /* Extra positionals with a `*rest` never leave a hole, so they can go
       straight through -- and a keyword alongside them would name a parameter
       already filled, which the loop below reports. */
    for (i = 0; i < want; i++) filled[i] = 0;
    for (i = 0; i < argc && i < bypos; i++) { slots[i] = raw[i]; filled[i] = 1; }
    if (O(target)->v.fn.kwarg) rest = apy_dict_new(kwn + 1);
    /* NAMED BY ITS QUALNAME in every refusal below -- `K.m()`,
       `outer.<locals>.inner()` -- which is what CPython prints and what the
       arity messages already say. */
    who = APY_CSTR(O(target)->v.fn.qualname ? O(target)->v.fn.qualname
                                            : O(target)->v.fn.name);

    /* POSITIONAL-ONLY NAMES PASSED BY KEYWORD ARE GATHERED, all of them at
       once: CPython lists every offender in ONE refusal (`'a, b'`) and says
       it BEFORE any other keyword complaint -- `pos(a=1, zz=2)` names `a`
       and never mentions `zz`. Scanned in DECLARATION order, which is the
       order CPython prints and not the order the call wrote; refusing at the
       first match named one of them, and named whichever the CALL happened
       to put first.

       SKIPPED WHEN THERE IS A `**kw`. The name is not refused then, it lands
       in the collection: `def f(a, /, **kw)` called `f(1, a=2)` binds
       `kw = {'a': 2}`. */
    if (!O(target)->v.fn.kwarg && O(target)->v.fn.posonly
            && O(target)->v.fn.pnames && kwn) {
        char list[192];
        size_t at = 0;
        list[0] = 0;
        for (i = skip; i < O(target)->v.fn.posonly; i++) {
            apy_value pn = O(target)->v.fn.pnames[i];
            if (!pn) continue;
            for (k = 0; k < kwn; k++) {
                const char *nm = APY_CSTR(O(kwd)->v.d.keys[k]);
                if (strcmp(nm, APY_CSTR(pn)) != 0) continue;
                if (at + strlen(nm) + 3 < sizeof list)
                    at += (size_t)snprintf(list + at, sizeof list - at,
                                           "%s%s", at ? ", " : "", nm);
                break;
            }
        }
        if (at) {
            char b[256];
            snprintf(b, sizeof b, "%s() got some positional-only arguments "
                                  "passed as keyword arguments: '%s'",
                     who, list);
            return apy_fail("TypeError", b);
        }
    }

    for (k = 0; k < kwn; k++) {
        apy_value nm = O(kwd)->v.d.keys[k], val = O(kwd)->v.d.vals[k];
        int64_t at = -1;
        int posonly_hit = 0;
        if (O(target)->v.fn.pnames)
            for (i = 0; i < want; i++) {
                apy_value p = O(target)->v.fn.pnames[i + skip];
                if (!p || strcmp(APY_CSTR(p), APY_CSTR(nm)) != 0) continue;
                /* POSITIONAL-ONLY: the name is recorded so this message can
                   be specific, but it does not match. */
                if (i + skip < O(target)->v.fn.posonly) { posonly_hit = 1; break; }
                at = i;
                break;
            }
        /* NOT CONDITIONAL ON THERE BEING NO SURPLUS. Surplus positionals
           belong to `*rest`; they do not stop a keyword from naming a
           parameter, and least of all a keyword-only one, which no position
           could have filled. Requiring `argc <= bypos` here sent `c=3` in
           `d(1, 2, c=3, z=4)` into `**kw` and left `c` on its default. */
        if (at >= 0 && !filled[at]) {
            slots[at] = val;
            filled[at] = 1;
            /* COUNTED FOR THE REFUSAL AND NOTHING ELSE. A surplus positional
               alongside these is worded by CPython as `3 positional
               arguments (and 1 keyword-only argument)`, and this is the only
               place that can tell a keyword-only landing from any other. */
            if (at >= bypos) kwonly_given++;
            continue;
        }
        if (at >= 0) {
            char b[160];
            snprintf(b, sizeof b, "%s() got multiple values for argument '%s'",
                     who, APY_CSTR(nm));
            return apy_fail("TypeError", b);
        }
        /* Not a declared parameter. `**kw` collects it; without one it is the
           error CPython reports, naming the keyword rather than a count. */
        if (rest) { apy_dict_set(rest, nm, val); continue; }
        {
            char b[192];
            /* THE POSITIONAL-ONLY CASE NEVER REACHES HERE. The gather above
               owns it, and owns every name at once; by this point the only
               keyword still unplaced is one the signature does not declare
               at all. */
            snprintf(b, sizeof b,
                     "%s() got an unexpected keyword argument '%s'",
                     who, APY_CSTR(nm));
            return apy_fail("TypeError", b);
        }
    }

    /* EVERY HOLE AT ONCE, and the two kinds of hole kept apart. CPython
       names all the parameters a call left unfilled in one refusal -- `'a'
       and 'c'`, and the holes need not be adjacent -- where refusing at the
       first meant a caller who forgot two learned about one, fixed it, and
       came straight back. Keyword-only holes are a SEPARATE message, said
       only when no positional one is outstanding: no position could have
       filled one, so counting it among the positional arguments sends the
       reader to the wrong half of the signature.

       GATHERED BEFORE ANY DEFAULT IS WRITTEN, because the fill is what hides
       them: a hole a default covers stops being one. */
    {
        apy_value miss[18], kwmiss[18];
        int64_t nmiss = 0, nkwmiss = 0;
        for (i = 0; i < want; i++) {
            apy_value pn;
            if (filled[i]) continue;
            if (apy_fn_default(target, i + skip)) continue;
            pn = O(target)->v.fn.pnames ? O(target)->v.fn.pnames[i + skip] : 0;
            if (i >= bypos) {
                if (nkwmiss < 18) kwmiss[nkwmiss++] = pn;
            } else {
                if (nmiss < 18) miss[nmiss++] = pn;
            }
        }
        if (nmiss || nkwmiss) {
            char b[256], list[192];
            const char *kind = nmiss ? "positional" : "keyword-only";
            int64_t n = nmiss ? nmiss : nkwmiss;
            apy_name_list(list, sizeof list, nmiss ? miss : kwmiss, n);
            snprintf(b, sizeof b, "%s() missing %lld required %s argument%s: %s",
                     who, (long long)n, kind, n == 1 ? "" : "s", list);
            return apy_fail("TypeError", b);
        }
    }
    for (i = 0; i < want; i++) {
        apy_value dv;
        if (filled[i]) continue;
        dv = apy_fn_default(target, i + skip);
        if (dv) slots[i] = dv;
    }
    if (argc > bypos) {
        if (O(target)->v.fn.vararg) {
            /* AFTER the declared parameters, not over the keyword-only ones
               at the end of them -- those were just filled by name, and
               writing the surplus into their slots discarded what the
               keywords supplied. `apy_call_nk` with `bound` set takes the
               first `declared` as parameters and everything past them as
               `*rest`, which is exactly this layout. */
            int64_t j;
            for (j = 0; bypos + j < argc && want + j < 17; j++)
                slots[want + j] = raw[bypos + j];
            want += j;
        } else {
            /* NO `*rest` TO SWALLOW THEM, so this is an arity error -- and it
               is worded HERE rather than passed on, because past this point
               `want` is the WIDTH OF THE SLOT ARRAY the names filled and
               `apy_call_nk` would have to guess back from it how the call was
               written. `skip` is the receiver CPython counts. */
            return apy_fn_arity_error(target, skip + argc, kwonly_given);
        }
    }
    return apy_call_nk(f, slots, want, rest, 1);
}

/* `f(*xs)`, where the argument COUNT is a value rather than a constant.

   An ordinary call knows its arity at compile time and passes a stack array.
   A starred one cannot: `xs` decides how many arguments there are, and the
   frontend has no number to emit. So the arguments arrive as a list and are
   copied into an argv here, which is the one place that count exists.

   `apy_call_n` then binds them against the callee's own signature -- defaults,
   `*rest`, arity mismatch and all -- so a spread call reports a wrong count
   exactly as a direct one does, at run time rather than at compile time. */
/* `f(*xs, **kw)`. The keyword half travels SEPARATELY, for the reason it does
   everywhere else here: only the binder knows where the positional arguments
   stop, so a dict appended to the list would arrive as one more positional.
   Dropping it made `f(*xs, **kw)` ignore every keyword in silence. */
APY_API apy_value apy_call_spread_kw(apy_value f, apy_value args,
                                     apy_value kwd) {
    int64_t n = O(args)->v.q.n;
    apy_value *argv = (apy_value *)malloc(sizeof(apy_value)
                                          * (size_t)(n ? n : 1));
    apy_value r;
    if (!argv) { fputs("uasm: out of memory\n", stderr); exit(1); }
    memcpy(argv, O(args)->v.q.items, sizeof(apy_value) * (size_t)n);
    r = (kwd && O(kwd)->kind == APY_DICT_K && O(kwd)->v.d.n)
        ? apy_call_kw(f, (apy_value)argv, n, kwd)
        : apy_call_n(f, argv, n);
    free(argv);
    return r;
}

APY_API apy_value apy_call_spread(apy_value f, apy_value args) {
    int64_t n = O(args)->v.q.n;
    apy_value *argv = (apy_value *)malloc(sizeof(apy_value) * (size_t)(n ? n : 1));
    apy_value r;
    if (!argv) { fputs("uasm: out of memory\n", stderr); exit(1); }
    memcpy(argv, O(args)->v.q.items, sizeof(apy_value) * (size_t)n);
    r = apy_call_n(f, argv, n);
    free(argv);
    return r;
}

/* `xs.extend(other)` in all but name: every element of a sequence, appended.
   Used to flatten a starred argument into the list a spread call builds. */
APY_API apy_value apy_extend(apy_value seq, apy_value other) {
    int64_t i;
    int into_bytes = O(seq)->kind == APY_BYTES_K && O(seq)->v.s.mut;
    apy_value raw = other;
    /* A STR IS ITERABLE AND ITS ELEMENTS ARE NOT INTEGERS, which CPython
       says in its own words: `expected iterable of integers; got: 'str'`.
       That is NOT `can't extend bytearray with str`, which is what it says
       for something it cannot walk at all. */
    if (into_bytes && O(other)->kind == APY_STR_K)
        return apy_fail2("TypeError",
                         "expected iterable of integers; got: '%s'%s",
                         apy_kind_name(other), "");
    /* BYTES ON THE END OF A BYTEARRAY -- the other half of what
       `apy_seq_push` gained, and missing for the same reason. Handled
       before `apy_iterable` because a `bytes` argument is the common case
       and walking it element by element would push ints through the same
       one-byte reallocation each time.
       ONLY FOR SOMETHING THAT REALLY IS BYTES-LIKE: `apy_to_bytes` accepts
       an INT and makes that many zero bytes, which is `bytes(5)` and not
       what `extend(5)` means -- CPython refuses it as unwalkable. */
    if (into_bytes && (O(other)->kind == APY_BYTES_K
                       || O(other)->kind == APY_MVIEW_K)) {
        apy_value add = apy_to_bytes(other);
        int64_t n, m;
        char *buf;
        if (!add)
            return apy_fail2("TypeError",
                             "can't extend bytearray with %s%s",
                             apy_kind_name(other), "");
        n = O(seq)->v.s.n;
        m = O(add)->v.s.n;
        buf = (char *)malloc((size_t)(n + m) + 1);
        if (!buf) return apy_fail("MemoryError", "out of memory");
        if (n) memcpy(buf, O(seq)->v.s.p, (size_t)n);
        if (m) memcpy(buf + n, O(add)->v.s.p, (size_t)m);
        buf[n + m] = 0;
        O(seq)->v.s.p = buf;
        O(seq)->v.s.n = n + m;
        return apy_none();
    }
    /* Drain anything that is iterable but not indexable -- a generator, a
       user object with `__iter__` -- so `[*gen]` and `f(*gen)` work. Both
       walks below are by index and neither can step a cursor. */
    other = apy_iterable(other);
    if (!other) {
        /* A BYTEARRAY NAMES WHAT IT COULD NOT WALK rather than passing on
           the iteration's own complaint: `bytearray().extend(None)` is
           `can't extend bytearray with NoneType`. */
        if (into_bytes) {
            apy_error_clear();
            return apy_fail2("TypeError",
                             "can't extend bytearray with %s%s",
                             apy_kind_name(raw), "");
        }
        return 0;
    }
    /* AN INSTANCE `apy_iterable` LEFT ALONE is `__len__` plus `__getitem__`
       -- the older protocol, whose walk IS the index walk, and the one
       `apy_key_at` reads. Without this arm `[*obj]` and `a, b = obj` for such
       a class reported `'Seq' object is not iterable` about the very protocol
       that makes it iterable. */
    if (O(other)->kind == APY_INST_K) {
        int64_t n = apy_raw_len(other);
        if (apy_error_occurred()) return 0;
        for (i = 0; i < n; i++) {
            apy_value item = apy_key_at(other, i);
            if (!item) return 0;
            if (!apy_seq_push(seq, item)) return 0;
        }
        return apy_none();
    }
    /* A CURSOR IS STEPPED, NOT INDEXED. `apy_iterable` hands one straight
       back -- only a generator is drained there -- so both walks below saw a
       kind neither could read and `[*map(f, xs)]`, `f(*it)` and
       `(*reversed(xs),)` all reported a `map` as not iterable at all.
       Stepping is also what leaves a partly consumed cursor where it is,
       which is what `[*it]` after two `next`s has to see. */
    if (O(other)->kind == APY_ITER_K) {
        for (;;) {
            apy_value item = apy_step(other);
            if (!item) return 0;
            if (item == apy_stop()) break;
            if (!apy_seq_push(seq, item)) return 0;
        }
        return apy_none();
    }
    /* A MEMORYVIEW IS ITERABLE and yields ints, which is what `[*mv]` and
       `xs.extend(mv)` expect; it was refused here while `list(mv)` -- the
       same walk by another road -- worked. */
    if (O(other)->kind == APY_STR_K || O(other)->kind == APY_BYTES_K
        || O(other)->kind == APY_DICT_K || O(other)->kind == APY_RANGE_K
        || O(other)->kind == APY_MVIEW_K) {
        int64_t n = apy_raw_len(other);
        for (i = 0; i < n; i++) {
            apy_value item = O(other)->kind == APY_DICT_K
                ? O(other)->v.d.keys[i]
                : apy_getitem(other, apy_from_int(i));
            if (!item) return 0;
            apy_seq_push(seq, item);
        }
        return apy_none();
    }
    if (!apy_is_seq(other) && !apy_is_set(other)) {
        if (into_bytes)
            return apy_fail2("TypeError",
                             "can't extend bytearray with %s%s",
                             apy_kind_name(raw), "");
        return apy_fail2("TypeError", "'%s' object is not iterable%s",
                         apy_kind_name(other), "");
    }
    for (i = 0; i < O(other)->v.q.n; i++)
        apy_seq_push(seq, O(other)->v.q.items[i]);
    return apy_none();
}

/* `xs.extend(other)` AS THE PROGRAM WROTE IT -- `apy_extend` with the length
   hint CPython asks for in front of it.

   A SEPARATE ENTRY POINT AND NOT A LINE INSIDE `apy_extend`, because
   `apy_extend` is the shared drain: `f(*xs)` reaches it, and so does the
   temporary list `set(x)` collects into, and CPython asks in NEITHER. The
   ask belongs to the spellings that MEAN extend -- the method, `+=` and a
   starred display -- and each of those names this or asks for itself.

   NO KIND TEST, because both receivers `extend` has ask: `list_extend` and
   `bytearray.extend` each size their result from the hint. `b += data` on a
   bytearray reaches this too and asks nothing in practice -- it only accepts
   something bytes-like, which is never an instance. */
APY_API apy_value apy_extend_meth(apy_value seq, apy_value other) {
    if (!apy_length_hint(other)) return 0;
    return apy_extend(seq, other);
}

/* Is this a user object? The frontend branches on it before a built-in method
   whose NAME some class in the same program also defines -- `x.add(1)` is a
   set's `add` or the program's, and only the receiver knows which.

   A predicate rather than a check inside each method entry point, because the
   frontend already knows which names collide: the branch is emitted only
   where one actually might, so `xs.append(v)` in a program with no `append`
   method stays a single call and pays nothing. */
APY_API int64_t apy_is_instance(apy_value v) {
    return O(v)->kind == APY_INST_K;
}

/* WHICH SIDE OF A METHOD CALL A RECEIVER BELONGS ON, and what it should be
   when it lands there. `_dyn_method_either` emits both a user lookup and a
   direct builtin call and picks between them at run time; these two answer
   the picking.

   THE QUESTION IS ABOUT THE CLASS, NOT THE KIND. It used to be `is this an
   instance`, which is right for the collision it was written for -- a class
   defining `add` next to `set.add` -- and wrong for `class D(dict)`, which
   INHERITS `keys` without writing it. Such an instance is not a set and not a
   dict either, so both branches refused it: the user lookup found no `keys`
   on the class, and the builtin call was never reached.

   SO THE INSTANCE IS UNWRAPPED INSTEAD. A method the class did not define is
   the builtin's, and the builtin wants the value the instance CARRIES. That
   is one test and one substitution rather than teaching each of the hundred
   or so builtin methods what an instance is. */
/* An instance ACTING AS the builtin it carries, for one operation. The
   class body wins: a `Counter` writing `__eq__` means its own, so the dunder
   is asked for by name and a class that defines it is left alone. Identity
   for everything that is not a builtin-extending instance, which is what lets
   this be dropped in front of an existing test rather than beside it. */
static apy_value apy_as_builtin(apy_value v, const char *dunder) {
    if (O(v)->kind == APY_INST_K && O(v)->v.o.held
            && !apy_class_find(O(v)->v.o.cls, apy_name(dunder)))
        return O(v)->v.o.held;
    return v;
}

APY_API int64_t apy_method_is_builtin(apy_value obj, apy_value name) {
    if (O(obj)->kind != APY_INST_K)
        return 1;
    /* THE CLASS BODY WINS. A `Counter` defining `update` means its own, even
       though `dict` has one -- which is the whole reason this asks the class
       before it asks the kind. */
    if (apy_class_find(O(obj)->v.o.cls, name))
        return 0;
    /* AN ORDINARY INSTANCE STAYS ON THE USER SIDE even with nothing found,
       so a `__getattr__` still gets its chance -- the lookup there reports
       the AttributeError, and reporting it from a builtin that was handed the
       wrong kind would name the kind instead of the attribute. */
    return O(obj)->v.o.held != 0;
}

APY_API apy_value apy_method_self(apy_value obj, apy_value name) {
    if (O(obj)->kind == APY_INST_K && O(obj)->v.o.held
            && !apy_class_find(O(obj)->v.o.cls, name))
        return O(obj)->v.o.held;
    return obj;
}

/* --- type objects ------------------------------------------------------- */
/* `type(x)` has to be a VALUE now, not the string `apy_type_name` returns:
   `isinstance(p, Point)` names a class, and comparing its name to a string
   would make two different classes with the same name interchangeable.

   Built-in types are INTERNED by name, so `type(1) is type(2)` is True the
   way it is in CPython. Interning by name rather than by kind is what keeps
   each exception type a single object -- every `APY_EXC_K` cell shares one
   kind but names one of thirty types. */
/* REACHED THROUGH TWO FUNCTIONS so the table can move, the shape the name
   cache and the source positions use. Pairs rather than two arrays, because
   the IR side reserves one block. */
static apy_value apy_type_rows_c[64][2];
static int64_t apy_type_count_c;
APY_API apy_value apy_type_rows(void) { return (apy_value)apy_type_rows_c; }
APY_API apy_value apy_type_slot_count(void) {
    return (apy_value)&apy_type_count_c;
}
/* THROUGH THE ACCESSORS, for the reason `apy_canonical_types` gives. */
#define apy_type_rows_at(i) (((apy_value (*)[2])apy_type_rows())[i])
#define apy_type_names(i) (apy_type_rows_at(i)[1])
#define apy_type_keys(i)  ((const char *)apy_type_rows_at(i)[0])
#define apy_type_count    (*(int64_t *)apy_type_slot_count())

/* THE EXPORTED HALF, which `runtime/makers.py` replaces. The static below
   keeps the name its callers use; `_of` was already taken by this
   function's own spelling, so the export is `_for`. */
APY_API apy_value apy_type_for(apy_value v) {
    const char *key;
    int i;
    if (O(v)->kind == APY_INST_K) return O(v)->v.o.cls;
    /* AN EXCEPTION OF A CLASS THE PROGRAM WROTE answers that class, so
       `type(e).__name__` and `type(e) is AppError` say what the source does.
       Without a class it falls through to the name-keyed table below, which
       is what every exception the runtime raises itself has. */
    if (O(v)->kind == APY_EXC_K && O(v)->v.e.cls) return O(v)->v.e.cls;
    /* `type(C)` IS THE METACLASS when one made it. An ordinary class has no
       metaclass recorded and reads as `type`, which is what it is. */
    if (O(v)->kind == APY_TYPE_K && O(v)->v.t.meta) return O(v)->v.t.meta;
    if (O(v)->kind == APY_TYPE_K) return apy_type_class();
    /* A BUILTIN TYPE IS A CLASS TOO, and the canonical thunk standing for one
       is a FUNC carrying `is_type` rather than a TYPE cell. Without this it
       fell through to the name-keyed table below, which built a SECOND object
       named `type` -- so `type(int) is type` was False while `print(type(int))`
       said `<class 'type'>`, which is the worst pair of answers to have. */
    if (O(v)->kind == APY_FUNC_K && O(v)->v.fn.is_type)
        return apy_type_class();
    key = apy_kind_name(v);
    /* THE SAME OBJECT THE NAME ANSWERS, when the program names that builtin
       type anywhere -- so `type(1) is int` holds. The frontend registers each
       one at the top of the entry, before any statement, which is what takes
       the evaluation order out of it: registering lazily made the answer
       depend on whether `type(1)` or `int` was reached first. */
    if (apy_canonical_types) {
        apy_value found = apy_dict_get_or(apy_canonical_types,
                                          apy_lit(key), 0);
        if (found) return found;
    }
    for (i = 0; i < apy_type_count; i++)
        if (strcmp(apy_type_keys(i), key) == 0) return apy_type_names(i);
    if (apy_type_count >= 64) return apy_type_new(apy_lit(key), 0);
    apy_type_rows_at(apy_type_count)[0] = (apy_value)(uintptr_t)key;
    apy_type_rows_at(apy_type_count)[1] = apy_type_new(apy_lit(key), 0);
    return apy_type_rows_at(apy_type_count++)[1];
}
static apy_value apy_type_of(apy_value v) {
    return apy_type_for(v);
}

APY_API apy_value apy_type_object(apy_value v) { return apy_type_of(v); }

/* `with` -- the two halves of the context-manager protocol.

   Separate entry points rather than one `apy_method1` at each call site,
   because the error text is specific: a value with neither method is
   reported as not being a context manager, naming the one it lacks, which is
   what CPython says and what tells the reader which half to write. */
/* `__aenter__` / `__aexit__`. Each ANSWERS A COROUTINE rather than a value:
   `async with` awaits what these return, which is the whole difference from
   the synchronous pair and the reason they cannot share an entry point. */
APY_API apy_value apy_aenter(apy_value cm) {
    apy_value m = apy_dunder(cm, "__aenter__");
    if (!m)
        return apy_fail2("TypeError",
                         "'%s' object does not support the asynchronous "
                         "context manager protocol%s", apy_kind_name(cm), "");
    return apy_call_n(m, NULL, 0);
}

APY_API apy_value apy_aexit(apy_value cm, apy_value exc) {
    apy_value m = apy_dunder(cm, "__aexit__"), argv[3];
    if (!m)
        return apy_fail2("TypeError",
                         "'%s' object does not support the asynchronous "
                         "context manager protocol%s", apy_kind_name(cm), "");
    /* All three from the one value, as `apy_exit` does it: the TYPE is what
       `et.__name__` reads, the VALUE is the exception, the traceback is None
       because there are none here. */
    argv[0] = O(exc)->kind == APY_EXC_K ? apy_exc_type(exc) : apy_none();
    argv[1] = exc;
    argv[2] = apy_none();
    return apy_call_n(m, argv, 3);
}

APY_API apy_value apy_enter(apy_value cm) {
    apy_value m;
    /* A MEMORYVIEW IS THE ONE BUILTIN THAT IS A CONTEXT MANAGER, and its
       two halves are this runtime's own code reached BY KIND -- `apy_dunder`
       below serves an instance and would never find them. What the block
       binds is the view itself; what leaving it does is release. */
    if (O(cm)->kind == APY_MVIEW_K) {
        if (!apy_mview_live(cm)) return 0;
        return cm;
    }
    /* `__exit__` FIRST, which is CPython's order and shows in the message:
       `with 5:` reports the missing `__exit__` rather than the missing
       `__enter__`, even though both are absent. */
    if (!apy_dunder(cm, "__exit__")) {
        apy_error_clear();
        return apy_fail2("TypeError",
                         "'%s' object does not support the context manager "
                         "protocol (missed %s method)",
                         apy_kind_name(cm), "__exit__");
    }
    m = apy_dunder(cm, "__enter__");
    if (!m) {
        apy_error_clear();
        return apy_fail2("TypeError",
                         "'%s' object does not support the context manager "
                         "protocol (missed %s method)",
                         apy_kind_name(cm), "__enter__");
    }
    return apy_call_n(m, NULL, 0);
}

/* `__exit__(type, value, traceback)`. `exc` is the live exception or None.

   All three arguments come from the one value: the TYPE is what
   `et.__name__` reads, the VALUE is the exception itself, and the traceback
   is None because there are none here. Passing None for the type when there
   is an exception would make `et.__name__` fail in a manager that logs it. */
APY_API apy_value apy_exit(apy_value cm, apy_value exc) {
    apy_value argv[3];
    apy_value m;
    /* THE VIEW HANDS ITS BUFFER BACK on the way out, and answers a false
       `__exit__` -- it swallows nothing. */
    if (O(cm)->kind == APY_MVIEW_K) {
        (void)exc;
        return apy_mview_release(cm);
    }
    m = apy_dunder(cm, "__exit__");
    if (!m)
        return apy_fail2("TypeError",
                         "'%s' object does not support the context manager "
                         "protocol%s", apy_kind_name(cm), "");
    if (O(exc)->kind == APY_EXC_K) {
        argv[0] = apy_type_of(exc);
        argv[1] = exc;
    } else {
        argv[0] = apy_none();
        argv[1] = apy_none();
    }
    argv[2] = apy_none();
    return apy_call_n(m, argv, 3);
}


/* Is `cls` anywhere in `of`'s order? The `isinstance` rule for user classes.

   THROUGH THE MRO WHEN THERE IS ONE, for the same reason attribute lookup is:
   `isinstance(D(), C)` for `class D(B, C)` is True and the base chain from D
   reaches only B and A. */
APY_API int64_t apy_type_is_sub_of(apy_value of, apy_value cls) {
    if (of && O(of)->kind == APY_TYPE_K && O(of)->v.t.mro) {
        apy_value order = O(of)->v.t.mro;
        int64_t i;
        for (i = 0; i < O(order)->v.q.n; i++)
            if (O(order)->v.q.items[i] == cls) return 1;
        return 0;
    }
    while (of && O(of)->kind == APY_TYPE_K) {
        if (of == cls) return 1;
        of = O(of)->v.t.base;
    }
    return 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_type_is_sub(apy_value of, apy_value cls) {
    return (int)apy_type_is_sub_of(of, cls);
}

/* --- operator dispatch to user methods ---------------------------------- */
/* `apy_str`, `apy_eq`, `apy_add` and the rest were written against a closed
   set of kinds. An instance is not one of them, and the whole point of
   `class` is that the answer comes from the program rather than from here.

   THE PROTOCOL FOR RETURNING. These helpers answer 0 both for "the class
   defines no such method" and for "it does, and it failed". The two are told
   apart by the ERROR FLAG, which is exactly how every other fallible
   operation in this file already reports, so a caller reads

       r = apy_binary_dunder(...);
       if (r || apy_error_occurred()) return r;

   and otherwise falls through to the TypeError it would have raised anyway.
   A separate out-parameter would be one more thing for a call site to get
   wrong, and there are twenty call sites. */

APY_API apy_value apy_dunder_of(apy_value v, apy_value name) {
    apy_value m;
    if (O(v)->kind != APY_INST_K) return 0;
    m = apy_class_find(O(v)->v.o.cls, apy_name((const char *)name));
    return (m && O(m)->kind == APY_FUNC_K) ? apy_bind(m, v) : 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_dunder(apy_value v, const char *name) {
    return apy_dunder_of(v, (apy_value)(uintptr_t)name);
}


APY_API apy_value apy_unary_dunder_of(apy_value v, apy_value name) {
    apy_value m = apy_dunder_of(v, name);
    return m ? apy_call_n(m, NULL, 0) : 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_unary_dunder(apy_value v, const char *name) {
    return apy_unary_dunder_of(v, (apy_value)(uintptr_t)name);
}

APY_API apy_value apy_method1_of(apy_value v, apy_value name,
                                 apy_value arg) {
    apy_value m = apy_dunder_of(v, name);
    return m ? apy_call_n(m, &arg, 1) : 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_method1(apy_value v, const char *name, apy_value arg) {
    return apy_method1_of(v, (apy_value)(uintptr_t)name, arg);
}

/* `a + b` asks `a.__add__(b)` first and `b.__radd__(a)` second. The reflected
   form is why `1 + v` can reach a user class at all: the int on the left has
   no idea what `v` is, so the right operand gets the second word. */
APY_API apy_value apy_binary_dunder_of(apy_value a, apy_value b,
                                       apy_value name, apy_value rname) {
    apy_value r = apy_method1_of(a, name, b);
    if (apy_error_occurred()) return r;
    /* `NotImplemented` MEANS "ASK THE OTHER OPERAND", not "the answer is
       NotImplemented". Returning it as the result made `Left() == Right()`
       answer the sentinel instead of falling back to Right's `__eq__`, and a
       program printing it saw a word where its answer should have been. */
    if (r && O(r)->kind != APY_NOTIMPL_K) return r;
    {
        apy_value other = apy_method1_of(b, rname, a);
        if (apy_error_occurred()) return other;
        if (other && O(other)->kind != APY_NOTIMPL_K) return other;
        /* NEITHER SIDE ANSWERED. Nothing is returned and no error is set,
           which is how every caller here spells "fall back to the default" --
           identity for `==`, a TypeError for arithmetic. */
        return 0;
    }
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_binary_dunder(apy_value a, apy_value b,
                                   const char *name,
                                   const char *rname) {
    return apy_binary_dunder_of(a, b, (apy_value)(uintptr_t)name,
                                (apy_value)(uintptr_t)rname);
}

/* --- what a comparison asks, and in what order -------------------------- */
/* `a OP b` tries THREE things: the dunder A WROTE, the builtin a extends --
   which is its INHERITED dunder -- and the mirror B WROTE. The builtin sits
   in the MIDDLE and not after both, because `type(a).__lt__` is the written
   one or the inherited one, and in CPython an inherited slot wins OUTRIGHT
   over a `__gt__` the other side wrote: `OnlyGt([1]) < OnlyGt([1, 2])` is
   True because `list.__lt__` answers and the reflected `__gt__` is never
   reached. `apy_binary_dunder` runs both written halves back to back, so the
   two are split here and the call sites put the builtin between them.

   THE ONE REORDERING RULE IS CPYTHON'S OWN. A right operand whose type is a
   PROPER SUBCLASS of the left's and which overrides the mirror goes first --
   `[1] < OnlyGt([1, 2])` asks `OnlyGt.__gt__` before `list.__lt__`, the
   subclass being presumed to know more about the pair than its base does.
   Its own direct dunder still follows, because the mirror may answer
   NotImplemented. Answers 1 for that order and 0 for the ordinary one. */
APY_API int64_t apy_order_mirror_first_of(apy_value a, apy_value b,
                                          apy_value mirror) {
    apy_value held;
    if (O(b)->kind != APY_INST_K) return 0;
    if (!apy_class_find(O(b)->v.o.cls, apy_name((const char *)mirror)))
        return 0;
    if (O(a)->kind == APY_INST_K)
        return (O(a)->v.o.cls != O(b)->v.o.cls
                && apy_type_is_sub(O(b)->v.o.cls, O(a)->v.o.cls)) ? 1 : 0;
    /* A IS NOT AN INSTANCE, so "a proper subclass of type(a)" is asking
       whether b extends the very builtin a is one of. */
    held = O(b)->v.o.held;
    return (held && O(held)->kind == O(a)->kind) ? 1 : 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_order_mirror_first(apy_value a, apy_value b,
                                  const char *mirror) {
    return (int)apy_order_mirror_first_of(a, b, (apy_value)(uintptr_t)mirror);
}

/* ONE HALF OF `apy_binary_dunder`: the dunder THIS side wrote, and only that.
   0 for a class that wrote none and 0 for `NotImplemented`, which means "ask
   the other operand" rather than "the answer is NotImplemented" -- the same
   reading `apy_binary_dunder` gives it. A caller checks the error flag, as
   it does there. */
APY_API apy_value apy_written_dunder_of(apy_value who, apy_value other,
                                        apy_value name) {
    apy_value r = apy_method1_of(who, name, other);
    if (apy_error_occurred()) return r;
    return (r && O(r)->kind != APY_NOTIMPL_K) ? r : 0;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static apy_value apy_written_dunder(apy_value who, apy_value other,
                                    const char *name) {
    return apy_written_dunder_of(who, other, (apy_value)(uintptr_t)name);
}

/* The builtin an instance holds, WITHOUT asking what its class wrote. The
   right operand of an inherited comparison is only an OPERAND: CPython's
   `list.__lt__(a, b)` takes any list subclass for `b` and never consults its
   methods. `apy_order_held` is the gated form, and it is the left side that
   needs gating -- a class that wrote the direct dunder has already had its
   say by the time the builtin is read. */
static apy_value apy_held_value(apy_value v) {
    return O(v)->kind == APY_INST_K ? O(v)->v.o.held : 0;
}

/* True when either operand is an instance, which is the guard every operator
   below uses before paying for a lookup. */
APY_API int64_t apy_either_inst_of(apy_value a, apy_value b) {
    return O(a)->kind == APY_INST_K || O(b)->kind == APY_INST_K;
}
/* THE NAME ITS CALLERS USE, kept as a delegate: the body is IR's now,
   and the exported half above stands in when nothing is ported. */
static int apy_either_inst(apy_value a, apy_value b) {
    return (int)apy_either_inst_of(a, b);
}

"""
