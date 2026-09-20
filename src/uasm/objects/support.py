"""The runtime the Python frontend's programs link against.

The IR has no I/O opcodes -- `print` is a call to a named function like any
other, and something must define it. That something is here rather than in a
backend, because every backend needs the same one and a backend that carried
its own copy would drift from the others exactly as the two liveness analyses
in the tree this replaces did.

It is C, and it is deliberately tiny. A frontend that wants a different
runtime supplies its own; nothing in the IR, the backends or the linker knows
what these functions do.

FLOAT FORMATTING is `%f`, matching `Interpreter._host`. Python's repr would
print `32.0` where this prints `32.000000`. The divergence is real and
documented in docs/LANGUAGE.md; what matters is that the interpreter and every
compiled binary agree, because a reference implementation that disagrees with
the thing it is a reference for is worse than none.
"""
from __future__ import annotations

from pathlib import Path

# THE PLATFORM FLOOR is defined next door and travels with the host functions
# rather than beside them: everything that emits or links this C needs both,
# and a second wiring point is a second place to forget one. They are separate
# MODULES because they are separate ideas -- `put_bool` knows how Python spells
# a true value and `plat_write` knows nothing at all -- and stage 6 of
# docs/INERT-RUNTIME.md deletes the first group and keeps the second.
from .floor import NAMES as _FLOOR_NAMES, c_source as _floor_c

#: The host functions `frontends/python` emits calls to, as C source.
#:
#: ONE definition with TWO consumers: `write_runtime` below, which writes it to
#: a real translation unit for the machine backends to link against, and the C
#: backend, whose output is self-contained and inlines it `static`. They were
#: two hand-maintained copies and they drifted exactly as this module's
#: docstring warns two liveness analyses once did -- `put_bool` was added to
#: one of them, and every program that printed a bool failed at the LINKER.
#:
#: `@STATIC@` is the storage class and `@STRPTR@` the string parameter type;
#: the C backend needs `static` and passes pointers as `uintptr_t`.
#: `x ** n`, correctly rounded. Shared with the FREESTANDING runtime.
#:
#: Split out of `HOST_FUNCTIONS` because `link/baremetal.py` needs exactly
#: this and can use exactly this: the algorithm touches only `+`, `-` and
#: `*` on doubles, so it needs no libc and no <stdint.h> -- hence
#: `long long` rather than `int64_t`. The freestanding copy it replaced
#: squared in plain double and was a ulp off CPython often enough to fail
#: four cases in eighty; a second implementation is a second thing to get
#: wrong.
#:
#: Only `py_pow_int` itself is a host function. The three helpers are
#: always `static`: the IR cannot name them, and `py_two_prod` with
#: external linkage in every object would collide at link time.
POW_INT_C = r"""/* `x ** n` for a non-negative integral n, correctly rounded.

   NOT libm's `pow`, and this is measurable rather than theoretical: mingw's
   `pow` is one ulp low on 1.5682546124542256 ** 4, and CPython's (the UCRT's
   on Windows, glibc's on Linux, both correctly rounded) is not. Four such
   disagreements showed up in eighty random cases, so any program doing float
   `**` printed a different last digit from CPython on this platform.

   Exponentiation by squaring in DOUBLE-DOUBLE: each value is an unevaluated
   sum of two doubles, giving ~106 bits of significand, so every intermediate
   product keeps the bits an ordinary double would drop and only the final
   `hi + lo` rounds. Squaring in plain double instead rounds once per step, and
   those errors accumulate in the same direction.

   Two-product uses Veltkamp splitting rather than `fma`, which is not in the
   baseline ISA and would need a runtime check to use safely. The split
   overflows for |a| above about 1.34e300, and `py_two_prod` detects that and
   falls back to the plain product there -- see the note on it. Above that
   magnitude the only exponents with a finite answer are 0 and +-1, so what is
   lost is the extra precision on results that are already inf or 0.

   ARBITRARY-PRECISION INTEGERS MADE THAT BAND ORDINARY. `(2 ** 1000) ** -1`
   is a five-character expression and it reaches a base of 1.07e301; while
   integers were 64 bits the only way there was a float literal near the top
   of the range. The fallback exists because the failure was a nan, not
   because the precision matters out there. */
static void py_two_sum(double a, double b, double *s, double *e) {
    double S = a + b, bb;
    /* Same trap as in `py_two_prod`, one step later: once `S` is infinite,
       `S - a` is inf minus inf and the error term is a nan that then travels
       as the low half of the double-double. `(2 ** 1000) ** -2` overflows to
       infinity on the way -- which is the correct answer, 0.0, after the
       reciprocal -- and came back as nan because of it. */
    if (S - S != 0.0) { *s = S; *e = 0.0; return; }
    bb = S - a;
    *e = (a - (S - bb)) + (b - bb);
    *s = S;
}

static void py_two_prod(double a, double b, double *p, double *e) {
    const double SPLIT = 134217729.0;  /* 2^27 + 1 */
    double c, d, ah, al, bh, bl;
    double P = a * b;
    /* THE SPLIT OVERFLOWS FOR A LARGE OPERAND, and what came out was not the
       infinity the header used to claim -- it was a NAN. `SPLIT * a` is
       infinite for |a| past about 1.34e300, so `c - (c - a)` is inf minus
       inf, the error term is nan, and the nan then contaminates `rh` itself
       through `py_two_sum`. `1e301 ** 1` returned nan, and so did every
       `x ** -1` for an x that large.
       Out there the product has already overflowed or is one step from it, so
       there is no additional precision left to preserve: hand back the plain
       product with an exact-zero error and let the ordinary double answer
       stand. Nothing inside the representable band takes this branch, so the
       correctly-rounded results this function exists for are untouched. */
    if (a > 1.3e300 || a < -1.3e300 || b > 1.3e300 || b < -1.3e300
        || P - P != 0.0) {
        *p = P;
        *e = 0.0;
        return;
    }
    c = SPLIT * a; ah = c - (c - a); al = a - ah;
    d = SPLIT * b; bh = d - (d - b); bl = b - bh;
    *e = ((ah * bh - P) + ah * bl + al * bh) + al * bl;
    *p = P;
}

static void py_dd_mul(double ah, double al, double bh, double bl,
                      double *rh, double *rl) {
    double p, e;
    py_two_prod(ah, bh, &p, &e);
    /* Once the high product has overflowed there is no correction left to
       carry, and computing one anyway is how a nan gets in: the cross terms
       `ah * bl` and `al * bh` are `inf * 0` as soon as one half is infinite
       and the other's low word is an honest zero. `(2 ** 1000) ** -2` reached
       exactly that -- an infinite intermediate whose correct final answer is
       0.0 -- and returned nan. */
    if (p - p != 0.0) { *rh = p; *rl = 0.0; return; }
    e += ah * bl + al * bh;
    py_two_sum(p, e, rh, rl);
}

@STATIC@double py_pow_int(double base, long long n) {
    double rh = 1.0, rl = 0.0, bh = base, bl = 0.0;
    /* The magnitude as UNSIGNED: `-n` overflows for LLONG_MIN and signed
       overflow is undefined. Such an exponent saturates to 0 or infinity long
       before the loop ends anyway; what matters is that it not be UB. */
    unsigned long long m = n < 0 ? -(unsigned long long)n : (unsigned long long)n;
    /* `x ** 0` is 1.0 for every x including nan, which is what an empty loop
       gives -- CPython agrees, and it is the one case where nan is not
       contagious. */
    while (m > 0) {
        if (m & 1) py_dd_mul(rh, rl, bh, bl, &rh, &rl);
        m >>= 1;
        if (m) py_dd_mul(bh, bl, bh, bl, &bh, &bl);
    }
    if (n >= 0) return rh + rl;

    /* A NEGATIVE exponent is the reciprocal, and taking it as `1.0 / (rh+rl)`
       rounds twice -- once collapsing the double-double, once dividing. That
       second rounding was measurable: 348 of 4000 random `x ** -k` cases came
       out one ulp from CPython, which is the whole reason this function
       computes in double-double in the first place.

       So divide BEFORE collapsing. `q` is the ordinary quotient and the
       Newton step below corrects it using the exact residual of `rh * q`
       (which is what `py_two_prod` gives), plus the `rl` term the collapse
       would have thrown away. One rounding, on the final add.

       Guarded on a zero, infinite or nan `rh`, where `py_two_prod` would
       produce a nan out of an inf-minus-inf and turn a perfectly good 0.0 or
       infinity into garbage. `rh - rh != 0.0` is "not finite" without needing
       <math.h>, which this function must stay clear of -- link/baremetal.py
       compiles it freestanding.

       AND GUARDED ON A MERELY LARGE `rh`, which is a different failure with
       the same symptom. `py_two_prod` splits its argument by multiplying by
       2**27 + 1, and that product overflows for |rh| above about 1.34e300 --
       so the split yields inf, `c - (c - a)` yields nan, and a finite
       reciprocal that a plain divide computes exactly comes back as nan.
       The docstring at the top of this function has always recorded that the
       split overflows there; what it did not say is that the RECIPROCAL path
       reaches the same cliff at a base the forward path handles fine.
       `(2 ** 1000) ** -1` is the shortest case, and it was unreachable until
       integers stopped being 64 bits. Below the cliff nothing changes; above
       it, one rounding is lost and the answer is right instead of nan. */
    if (rh == 0.0 || rh - rh != 0.0 || rh > 1.3e300 || rh < -1.3e300)
        return 1.0 / (rh + rl);
    {
        double q = 1.0 / rh, p, e, r;
        py_two_prod(rh, q, &p, &e);
        r = ((1.0 - p) - e) - rl * q;
        return q + q * r;
    }
}
"""


HOST_FUNCTIONS = """\
/* `put_*` write a value and nothing else. The frontend supplies the space
   between printed arguments and the newline at the end with putchar, because
   a primitive that always appends a newline cannot express `print(1, 2)` --
   one line reading "1 2" -- or `print()`, an empty line. */
@STATIC@void put_int(int64_t v)     { printf("%lld", (long long)v); }

/* Python's float repr: the SHORTEST decimal string that reads back as the same
   double. Not `%f`, and not `%.17g` either.

   `%f` gave `3.500000` for 3.5 and `0.000000` for 1e-9, so every printed float
   disagreed with CPython and small ones lost all their information. `%.17g`
   round-trips but is not shortest -- 0.1 comes out `0.10000000000000001`.

   Shortest is found by asking for one significant digit, then two, until the
   result reads back equal. Seventeen always succeeds for a finite double, so
   the loop terminates.

   Choosing fixed vs exponent notation is then Python's rule, not C's, and this
   is where the obvious `%.*g` implementation is wrong: `%g` switches to
   exponent form at `exp >= precision`, which varies with how many digits the
   value happened to need, while Python switches at a FIXED `exp >= 16`. The
   two disagree for values like 12345678901234567.0, which Python prints as
   1.2345678901234568e+16 and `%.17g` prints in full.

   The exponent is normalised to at least two digits because Python writes
   `1e-05`, and to no more than the digits it needs because the MSVCRT that
   mingw links against writes three (`1e-005`).  */
@STATIC@void py_repr_double(char *out, size_t n, double v) {
    if (isnan(v)) { snprintf(out, n, "nan"); return; }
    if (isinf(v)) { snprintf(out, n, v < 0 ? "-inf" : "inf"); return; }

    char buf[64];
    int digits;
    /* `<= 17`, not `< 17`. Stopping at 16 left the loop with digits == 17 and
       `buf` still holding the SIXTEEN-digit attempt that had just failed to
       round-trip, so the one value in a thousand that genuinely needs all
       seventeen printed the rejected string: 12345678901234567.0 came out as
       1.234567890123457e+16, which reads back as a different double. */
    for (digits = 1; digits <= 17; digits++) {
        snprintf(buf, sizeof buf, "%.*e", digits - 1, v);
        if (strtod(buf, NULL) == v) break;
    }
    if (digits > 17) digits = 17;
    const char *epos = strchr(buf, 'e');
    int exp10 = (int)strtol(epos + 1, NULL, 10);

    if (exp10 < -4 || exp10 >= 16) {
        size_t mlen = (size_t)(epos - buf);
        char mant[48];
        if (mlen >= sizeof mant) mlen = sizeof mant - 1;
        memcpy(mant, buf, mlen);
        mant[mlen] = '\\0';
        snprintf(out, n, "%s%s%02d", mant, exp10 < 0 ? "e-" : "e+",
                 exp10 < 0 ? -exp10 : exp10);
        return;
    }

    int decimals = digits - 1 - exp10;
    if (decimals < 0) decimals = 0;
    snprintf(out, n, "%.*f", decimals, v);
    /* Python's repr always shows a float AS a float: `2.0`, never `2`. */
    if (!strchr(out, '.')) {
        size_t len = strlen(out);
        if (len + 2 < n) { out[len] = '.'; out[len+1] = '0'; out[len+2] = '\\0'; }
    }
}

@POW@

@STATIC@void put_float(double v)    { char b[64]; py_repr_double(b, sizeof b, v); fputs(b, stdout); }
/* bool and None are not integers to Python even though they are to the
   machine: `print(True)` is `True`, and printing 1 there is wrong in a way
   that no arithmetic fix can reach -- it is the rendering, not the value. */
@STATIC@void put_bool(int64_t v)    { fputs(v ? "True" : "False", stdout); }
@STATIC@void put_none(void)         { fputs("None", stdout); }
@STATIC@void print_int(int64_t v)   { printf("%lld\\n", (long long)v); }
@STATIC@void print_float(double v)  { char b[64]; py_repr_double(b, sizeof b, v); puts(b); }
@STATIC@void print_str(@STRPTR@ s) { fputs((const char *)s, stdout); }
"""


#: Every host function `HOST_FUNCTIONS` defines, by name. Written out rather
#: than parsed back out of the C, because "find the definitions in this source"
#: is a job for a C parser and a line-splitting approximation of one silently
#: MISSED `put_float` (its body mentions a helper, and the filter meant to skip
#: that helper's own definition skipped every line naming it).
#:
#: `putchar` is not here: it is libc's, not this runtime's.
HOST_NAMES = (
    "put_int", "put_float", "put_bool", "put_none",
    "print_int", "print_float", "print_str",
    "py_pow_int",
) + _FLOOR_NAMES


def host_functions(*, static: bool = False,
                   strptr: str = "const char *",
                   ptr: str = "void *", floor: bool = True) -> str:
    """`HOST_FUNCTIONS` with its substitutions made.

    `py_repr_double` is always `static`: it is an implementation detail of the
    writers, not something the IR can call, and giving it external linkage in
    the linked-runtime build would export a symbol no frontend asked for. The
    same goes for the three helpers inside `POW_INT_C`, which is why they are
    written `static` there rather than marked.

    `floor=False` LEAVES THE THREE FLOOR FUNCTIONS UNDEFINED, for the one
    consumer that must not have them: this runtime compiled by uasm's OWN C
    frontend. That frontend's `__uasm_base.h` already declares them --
    `extern long plat_write(long, const void *, long)` -- and its whole
    standard library is written against that declaration, so defining them
    here as `int64_t plat_write(int64_t, void *, int64_t)` is a conflicting
    type for the same symbol and the compile stops. The definitions come from
    `link/freestanding.py` instead, as syscalls, which is what makes the
    result need no libc.
    """
    text = HOST_FUNCTIONS.replace("@POW@", POW_INT_C).replace("@STRPTR@", strptr)
    text = text.replace("@STATIC@void py_repr_double",
                        "static void py_repr_double")
    # THE FLOOR FIRST. Nothing above it may depend on it yet, but everything
    # eventually will -- stage 6 replaces the writers with subset code that
    # calls `plat_write` -- and C wants a definition in scope before a use.
    if floor:
        text = _floor_c(static=static, ptr=ptr) + "\n" + text
    return text.replace("@STATIC@", "static " if static else "")


#: The linked runtime: the host functions, plus the C entry point.
#:
#: The IR's `main` is NOT C's `main`: it returns i64 and C requires int, and on
#: Windows the real entry point is further wrapped again. The backend renames
#: it and this calls it, so the rename lives in one place.
RUNTIME_C = """/* uasm runtime for the Python frontend. Generated -- edit objects/support.py. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

@HOST@

@OBJECTS@

extern int64_t @ENTRY@(void);

/* NO COMMAND LINE HERE, unlike the C backend's own wrapper, which takes
   `(int argc, char **argv)` and hands them to the `env` group's
   `apy_host_args_take`. This runtime is what a backend that emits MACHINE
   CODE links against, and `runtime_c` below substitutes the host functions
   and the object runtime only -- there is no `@HOSTSVC@` in it, so the
   statics `sys.argv` reads through do not exist in this translation unit at
   all. A program needing them is refused before it gets here, by
   `Backend.check_host_services` naming the `env` group; wiring this `main`
   would have to wait for those backends to provide it. */
int main(void) { return (int)@ENTRY@(); }
"""

#: Substituted by `write_runtime`. A literal token rather than %-formatting or
#: str.format: this text is C, it is full of `%lld` and `%f`, and both of those
#: mechanisms would try to interpret them. `%(entry)s` here raised
#: "unsupported format character 'l'" on the very first link.
_ENTRY_TOKEN = "@ENTRY@"

#: The C `main` shim, spelled once so that the two places that take it out
#: take out the same text.
MAIN_SHIM = "int main(void) { return (int)@ENTRY@(); }"

#: THE EXPORTED NAME OF THIS UNIT'S STATIC INITIALISER, and of the function
#: the floor calls in place of the entry when it is there.
#:
#: WHY THERE IS A SECOND ENTRY AT ALL. `static const char *D =
#: "0123456789abcdef";` -- `apy_bytes_hex_n` has one, `apy_bytes_repr` has
#: one, `apy_class_known` has one -- cannot be written into the global's
#: bytes: the literal's address belongs to the linker. The C frontend turns
#: each into a STORE in the unit's initialiser, and the only thing that calls
#: that initialiser is the entry point it builds around `main`. This
#: translation unit has no `main` on purpose (see `MAIN_SHIM` below), so
#: those globals stayed ZERO and the first read through one segfaulted --
#: silently, because the frontend's warning about it went to a sink nobody
#: rendered. `--c:init-symbol` gives the initialiser an exported name, this
#: shim calls it, and the floor calls this shim.
STATIC_INIT_SYMBOL = "uasm_static_init"
BOOT_SYMBOL = "uasm_boot"

#: What stands in the `main` shim's place for a freestanding link. Written
#: with `@ENTRY@` still in it, because every caller substitutes that after.
BOOT_SHIM = f"""/* The `main` shim is omitted: uasm's own C frontend renames C's `main` to
   the backend's entry symbol, which is the very name the program's own
   object defines, and two definitions of it is what the linker then reports
   about a function neither file appears to have written.

   THIS IS WHAT TAKES ITS PLACE, and it is not just a rename: `main` was also
   the one thing that ran this unit's static initialisers. See
   `STATIC_INIT_SYMBOL`. */
extern void {STATIC_INIT_SYMBOL}(void);
int64_t {BOOT_SYMBOL}(void) {{ {STATIC_INIT_SYMBOL}(); return @ENTRY@(); }}"""

# The name the backend gives the IR's `main`. Imported, not repeated: the
# backend writing the symbol and this runtime calling it must agree.
from ..backend.base import ENTRY_SYMBOL  # noqa: E402


def runtime_c(*, entry: str = ENTRY_SYMBOL, module=None,
              floor: bool = True, main_shim: bool = True) -> str:
    """The complete linked runtime, ready to compile.

    Every consumer goes through this rather than substituting the placeholders
    itself. `RUNTIME_C` has grown a placeholder twice now, and each time the
    hand-rolled copies -- two of them in the test suite -- kept compiling right
    up to `error: stray '@' in program`, which names neither the missing piece
    nor the file that should have supplied it.

    The object runtime comes AFTER the host functions: it calls
    `py_repr_double` and `py_pow_int`, and C wants a definition in scope.
    """
    from .csource import objects_c
    from .ir import omitted_by, split_by
    text = RUNTIME_C
    if not main_shim:
        # THE `main` SHIM IS NOT PART OF THE RUNTIME when this text is
        # compiled by uasm's OWN C frontend. `BOOT_SHIM` says why, and why
        # what replaces it is a function and not a comment.
        text = text.replace(MAIN_SHIM, BOOT_SHIM)
    return (text.replace("@HOST@", host_functions(floor=floor))
            .replace("@OBJECTS@", objects_c(omit=omitted_by(module),
                                            split=split_by(module)))
            .replace(_ENTRY_TOKEN, entry))


def write_runtime(directory: Path, *, entry: str = ENTRY_SYMBOL,
                  module=None, floor: bool = True) -> Path:
    """Write the runtime C file into `directory` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "uasm_runtime.c"
    path.write_text(runtime_c(entry=entry, module=module, floor=floor),
                    encoding="utf-8")
    return path


def needs_runtime(module) -> bool:
    """Whether `module` calls anything the runtime provides.

    A module that never prints does not need it linked in, and linking an
    unused object is how a "no dependencies" claim quietly stops being true.
    """
    from ..ir.opcodes import Op
    from .csource import OBJECT_NAMES
    provided = set(HOST_NAMES) | set(OBJECT_NAMES)
    return any(ins.sym in provided
               for f in module.functions for b in f.blocks
               for ins in b.instructions if ins.op is Op.CALL)
