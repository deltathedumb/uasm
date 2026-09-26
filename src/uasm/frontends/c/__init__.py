"""The C frontend: it compiles C.

    tokens.py        what a preprocessing token is, and why it is not a token
    lexer.py         phases 1-3: source text -> preprocessing tokens
    preprocess.py    phase 4: directives and macro expansion
    ppexpr.py        `#if` arithmetic, which is not the constant folder
    literals.py      decoding the four kinds of literal, for both of those
    ctype.py         the C type system: sizes, layout, conversions
    syntax.py        the tree, already typed
    parser.py        phase 7: the grammar, typed as it is parsed
    sema.py          the type rules the parser calls into
    fold.py          constant expressions, which C requires in six places
    lower.py         typed tree -> UIR
    lower_builtins.py  code for the `__builtin_*` names
    merge.py         several translation units, one module
    longjmp.py       `setjmp` and `longjmp`, without a machine frame
    builtins.py      which of them exist; `__has_builtin` reads this
    attributes.py    C23's `[[...]]`: which exist and what each one does
    builtin_check.py their type rules, including the three taking a TYPE
    ldouble.py       `long double` at compile time: exact, and encoded
    support.py       the VLA arena, the command line, complex and long double
    include/         the standard headers, written in C

WHAT IT ACCEPTS is C23, which is to say C: every version's syntax, every
version's semantics where they agree, and the newer one where they do not.

SIX PLACES THIS IMPLEMENTATION DIFFERS from a hosted one on x86-64 Linux,
and this is the whole list rather than the beginning of one. What the
LIBRARY needs from the TARGET -- a filesystem, a clock, an environment -- is a
different list, and it is in `include/README.md`: those are
`objects/hostsvc.py`'s optional groups, a backend declares the ones its target
has, and a program that calls into one it has not got is refused by name at
compile time rather than failing to link.

  * `long double` IS SOFTWARE. The format is x86-64's -- 80-bit extended,
    sixteen bytes with ten in use, a 64-bit significand -- so `sizeof`,
    `LDBL_*` and every printed digit agree with a hosted compiler. The
    arithmetic is `support.py`'s `ldouble` unit rather than the machine's,
    because the IR has `f32` and `f64` and a third width would have to be
    implemented by every backend, including the ones whose machine has no
    such thing. The `l` SERIES in `<math.h>` -- `sinl`, `expl`, `powl` and
    the rest -- compute in double and widen, so they answer to about a
    double's precision in the wider type. EVERYTHING THAT IS EXACT BY
    DEFINITION IS EXACT: `sqrtl`, the four operators, the conversions,
    `strtold` and `%Lf`, `frexpl`, `ilogbl`, `logbl`, `ldexpl`, `modfl`,
    the five roundings to an integer, `fmodl`, `remainderl`, `remquol`,
    `nextafterl` and the rest, all on the sixteen bytes rather than through
    a double that has neither the range nor the precision to hold them.
  * `_Imaginary` IS NOT THERE, which is conforming rather than missing:
    imaginary types are Annex G, supported only by an implementation that
    defines `__STDC_IEC_559_COMPLEX__`, and gcc has never had them either.
    `_Complex` is complete, including Annex G's multiply and divide.
  * A LOCAL THAT IS NOT `volatile` SURVIVES A `longjmp`. C says its value is
    indeterminate; here the frame never went away, so it keeps what it had.
    Stricter than the standard, which is the safe way to differ.
  * `localtime` IS `gmtime`. The host services can say what time it is and
    cannot say what the local offset from UTC is, so the calendar is UTC and
    `tm_isdst` is 0 -- not unknown, not in effect.
  * A STREAM'S ORIENTATION IS RECORDED AND NOT ENFORCED. C says a stream
    takes one on its first operation and leaves the other kind undefined
    afterwards; glibc makes the second call fail. Here there is one buffer
    under both faces, so a program that writes with `printf` and then with
    `fwprintf` gets both -- stricter than the standard in the safe
    direction, as the `longjmp` case is. `fwide` still answers truthfully.
  * THE EXECUTION CHARACTER SET IS UTF-8, ALWAYS, and so is the multibyte
    encoding. C leaves both to the implementation and this one chose the
    source's: `"caf\u00e9"` is six bytes, `'é'` is the multi-character
    constant its two make, `mbrtowc` decodes UTF-8 and `MB_CUR_MAX` is 4.
    glibc's `"C"` locale calls each byte a character and answers -1 for the
    same input; `<locale.h>` here has one locale, so there is nowhere to put
    the other answer even if it were wanted.

Everything else -- VLAs, flexible array members, bit-fields, anonymous
members, `_Generic`, designated initialisers, compound literals (including at
file scope, where one is a static object and its address is a constant),
`__VA_OPT__`, K&R definitions, statement expressions, `__attribute__` where it
is advisory, and GNU's `__typeof__`/`__restrict` spellings because real headers
use them -- is implemented.

AND C23's OWN: `_BitInt(N)`, a bit-precise integer that is NOT promoted --
`a + b` on two `_BitInt(4)`s is arithmetic in four bits and wraps there --
with the `wb` and `uwb` constant suffixes and a `BITINT_MAXWIDTH` of 64,
which is what C23 requires (`ULLONG_WIDTH`) and no more: a wider one would be
software arithmetic over several registers, for a type whose appeal is being
a machine integer with a narrower range. `constexpr` objects, whose NAME is a
constant expression --
which is the whole of what the specifier buys, since `const int n = 7;` has
always produced the same code and never been one; `[[...]]` attributes
everywhere C allows them, with `nodiscard` and `deprecated` the two that
warn and `attributes.py` saying why the rest do not; `nullptr` and
`nullptr_t`; `auto` with no type specifier, which takes the initialiser's
type after decay and lvalue conversion rather than being a storage class;
`typeof` and `typeof_unqual`; an `enum` with a fixed underlying type; a label
before a declaration and at the end of a block; `u8` character constants and
strings, binary literals and digit separators; `unreachable()`;
`mbrtoc8`/`c8rtomb` for the code units `char8_t` holds; and `%b`, `%ls` and
`%lc` in `printf` with `%ls`, `%lc` and `%l[` in `scanf`.

IN THE PREPROCESSOR: `#embed`, with `limit`, `prefix`, `suffix` and
`if_empty` and with `__has_embed` to ask first -- a file's bytes as integer
constants, which is a DIRECTIVE because the script that used to write that
array wrote a C file with a hundred thousand tokens in it that every build
then had to lex; `#elifdef` and `#elifndef`; `__VA_OPT__`; `#warning`;
`__has_include`, `__has_c_attribute` and `__has_builtin`; and `_Pragma`,
processed after macro replacement because that is where a header's
`_Pragma(STRINGIFY(x))` needs it to be.

ALL THIRTY-ONE HEADERS C23 REQUIRES are in `include/`, and all thirty-one
work. `<threads.h>` is `objects/hostsvc.py`'s `thread` group: threads are a
capability of the TARGET, so a backend that has them gets `thrd_create`, a
mutex, a condition variable and `tss_t`, one that has not got them refuses
the program by name, and `thrd_create` may still answer `thrd_error` --
which C allows and a portable program checks. `_Thread_local` is compiled
onto the same keys. `<stdatomic.h>` IS THE ONE TO READ CAREFULLY now that
there can be two threads: its operations are ordinary loads and stores with
the right names, which was right for one thread and is not a promise this
library can keep for two.

SEVERAL TRANSLATION UNITS IN ONE BUILD. `uasm build main.c --c:unit parse.c
--c:unit emit.c` compiles each file on its own and joins the modules, which is
what `cc main.c parse.c emit.c` does and means the same things by it: a
`static` in one file is not the one of that name in the next, a definition in
one is callable from another, and two definitions of one external name is an
error naming both files. `merge.py` is that step and says how.

ONE FLAG AND NOT SEVERAL POSITIONAL SOURCES, because the driver hands each
frontend ONE source -- it is what names the output, what the module search
path is relative to, and what every other frontend expects. A C build wanting
more says so with a flag of C's own rather than changing that for everybody.

THE STANDARD LIBRARY IS COMPILED FROM C, by this frontend, from `include/`.
It sits on the three platform-floor functions and nothing else, so a C program
built here runs on every backend and in the IR interpreter -- which is what
makes the differential suite able to compare all of them.
"""
from __future__ import annotations

from pathlib import Path

from ...diagnostics import DiagnosticSink, SourceFile, error, warning
from ...frontend import Frontend, register
from ...options import Option, OptionError
from ...ir import Module
from . import longjmp
from .lower import Lowerer
from .merge import merge
from .parser import parse
from .preprocess import Preprocessor, Search


class CFrontend(Frontend):
    name = "c"
    #: `.h` IS NOT HERE. A header is not a translation unit, and claiming the
    #: extension would make `uasm build foo.h` compile something that
    #: was never meant to stand alone. `.i` is preprocessed C and is still C.
    extensions = (".c", ".i")
    description = "C23, with the standard library compiled from C"
    language_versions = ("c23",)
    default_language_version = "c23"

    #: THE FLAGS A C COMPILER HAS ALWAYS HAD, declared here rather than on
    #: the driver's parser: they are this frontend's and nobody else's, and a
    #: Python build offered `--include-path` would be offered a flag that
    #: means nothing to it. `--c:include-path` is always spellable too.
    options = (
        Option("language-version", metavar="VERSION",
               help="C language version (currently c23)"),
        # `-I` AND `-D`, spelled as every C compiler spells them. They are
        # the two flags a C build is most likely to be handed by a Makefile
        # written elsewhere, and a compiler that only accepted the long form
        # would need that Makefile rewritten to use it.
        Option("include-path",
               "where #include <...> looks, before the bundled headers",
               metavar="DIR", repeat=True, short="I"),
        Option("define", "define a preprocessor macro; no value means 1",
               metavar="NAME[=VALUE]", repeat=True, short="D"),
        Option("trigraphs",
               "translate ??= and the rest; C23 deleted them", metavar="1|0"),
        Option("bundled-headers",
               "search the frontend's own standard headers",
               metavar="1|0"),
        Option("unit", "another translation unit to compile into this program",
               metavar="FILE", repeat=True),
        # A UNIT WITH NO `main` HAS NOTHING TO RUN ITS STATIC INITIALISERS.
        # `static const char *D = "0123456789abcdef";` cannot be written into
        # the global's bytes -- the literal's address is the linker's to
        # decide -- so it becomes a STORE in the unit's initialiser, and the
        # only thing that calls that initialiser is the entry point this
        # frontend builds around `main`. A unit compiled WITHOUT a `main` --
        # the object runtime is one, deliberately, since its `main` would be
        # a second definition of the backend's entry symbol -- therefore
        # starts with that global still zero, and reads through it at the
        # first use. Naming the initialiser here makes it EXPORTED and
        # callable, so the unit's own code, or whatever links it, can run it.
        Option("init-symbol",
               "name the unit's static initialiser and export it, for a "
               "unit with no `main` to run it from",
               metavar="NAME"),
    )

    def __init__(self, *, include_paths: tuple[Path, ...] = (),
                 defines: tuple[tuple[str, str], ...] = (),
                 trigraphs: bool = False, bundled: bool = True,
                 units: tuple[Path, ...] = (),
                 init_symbol: str = "", language_version: str = "c23") -> None:
        self.include_paths = include_paths
        self.defines = defines
        self.trigraphs = trigraphs
        self.bundled = bundled
        self.units = units
        self.init_symbol = init_symbol
        self.language_version = language_version

    def configure(self, values: dict, context, sink: DiagnosticSink
                  ) -> "CFrontend":
        """A frontend carrying this run's flags. See `Frontend.configure`.

        `context` IS UNUSED HERE, deliberately: `#include "foo.h"` resolves
        against the INCLUDING FILE's directory, which the preprocessor knows
        from the file it is reading, and `<foo.h>` against the search path.
        Neither needs the source the driver started from, and nothing this
        frontend does is scoped by the target platform.
        """
        version = values.get("language-version", self.default_language_version)
        if version.lower() not in self.language_versions:
            raise OptionError(
                f"unsupported C language version {version!r}; "
                f"supported: {', '.join(self.language_versions)}")
        return CFrontend(
            include_paths=tuple(Path(p) for p in
                                values.get("include-path", ())),
            defines=tuple(_split_define(d) for d in values.get("define", ())),
            trigraphs=_truth(values, "trigraphs", self.trigraphs),
            bundled=_truth(values, "bundled-headers", self.bundled),
            units=tuple(Path(u) for u in values.get("unit", ())),
            init_symbol=values.get("init-symbol", self.init_symbol) or "",
            language_version=version.lower())

    def compile(self, source: SourceFile, sink: DiagnosticSink) -> Module | None:
        """One module, from this source and any `--c:unit` beside it."""
        sources = [source]
        for path in self.units:
            try:
                sources.append(SourceFile.read(path))
            except OSError as exc:
                sink.report(error("E1602", f"cannot read {path}: "
                                           f"{exc.strerror}")
                            .help("--c:unit names another .c file to compile "
                                  "into the same program"))
                return None
        inits = tuple(f"c{i}.init" for i in range(len(sources))) \
            if len(sources) > 1 else ()
        modules = []
        for i, src in enumerate(sources):
            # ONE NAME CANNOT SERVE SEVERAL UNITS. `--c:init-symbol` gives
            # the initialiser an exported name, and exporting the same name
            # from every unit of a multi-unit build is two definitions of it.
            # A build with more than one unit already has `main` in one of
            # them and `inits` wiring the rest, which is the mechanism this
            # flag exists to stand in for.
            module = self._one(src, sink, prefix=f"c{i}.", inits=inits,
                               init_symbol=("" if len(sources) > 1
                                            else self.init_symbol))
            if module is None:
                return None
            modules.append(module)
        if len(modules) == 1:
            return self._finish(modules[0], sink)
        joined = merge(modules, [s.name for s in sources], sink)
        if sink.failed:
            return None
        if joined.function("main") is None:
            # NOT AN ERROR HERE. A library of several units with no `main` is
            # a reasonable thing to compile; the driver is what knows whether
            # a program was wanted, and it says so about a module with no
            # entry point. This only says the static initialisers nobody will
            # run, because THAT is invisible otherwise.
            if any(f.name.endswith(".init") for f in joined.functions):
                sink.report(
                    warning("W1500",
                            "these units have static initialisers that need "
                            "addresses, and no `main` to run them from")
                    .note("the entry point calls every unit's initialiser; "
                          "without one, none of them runs"))
        return self._finish(joined, sink)

    def _finish(self, module: Module, sink: DiagnosticSink) -> Module | None:
        """What is done to the whole program rather than to one unit.

        `setjmp` IS THE ONE THING, and it has to be here: the rewriting puts
        a check after every call in the PROGRAM, because a frame between the
        `longjmp` and the `setjmp` may be a library function in a unit that
        never mentioned either name.
        """
        if longjmp.used(module):
            longjmp.apply(module, sink)
            if sink.failed:
                return None
        return module

    def _one(self, source: SourceFile, sink: DiagnosticSink, *,
             prefix: str, inits: tuple[str, ...],
             init_symbol: str = "") -> Module | None:
        search = Search(angle=list(self.include_paths), bundled=self.bundled)
        pp = Preprocessor(sink, search, trigraphs=self.trigraphs,
                          defines=dict(self.defines))
        tokens = pp.run(source)
        if sink.failed:
            return None
        unit, parser = parse(tokens, sink, prefix)
        if sink.failed:
            # Lowering assumes the parser accepted everything it sees. Running
            # it anyway produces IR the verifier rejects, and the user gets an
            # internal-error report on top of their real diagnostics.
            return None
        try:
            return Lowerer(unit, parser, source, sink, inits,
                           init_symbol=init_symbol).run()
        except RecursionError:
            sink.report(
                error("E1599", "this program is too deeply nested to compile")
                .at(unit.span)
                .note("lowering walks the tree recursively, and this one is "
                      "deeper than the interpreter stack allows"))
            return None


def _split_define(item: str) -> tuple[str, str]:
    """`-D NAME=VALUE` and `-D NAME`, the latter defined as `1`.

    Split on the FIRST `=` only, because a macro body may contain one:
    `-D MAX(a,b)=((a)>(b)?(a):(b))` is a definition every build system
    writes, and splitting on all of them loses most of it.
    """
    name, sep, value = item.partition("=")
    return name, value if sep else "1"


def _truth(values: dict, name: str, default: bool) -> bool:
    """A `1|0` flag. The spelling `plugin add --cwd 1|0` already uses."""
    raw = values.get(name)
    if raw is None:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise OptionError(f"--{name} takes 1 or 0, not {raw!r}")


register(CFrontend())
