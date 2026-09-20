"""Typed C to UIR.

THE IR HAS NO AGGREGATES, NO VARARGS AND NO DYNAMIC `alloca`, and C has all
three. Each is met with a lowering convention, written down here because a
convention that is only in the code is one the next reader has to reverse
engineer from the instructions it emits.

A STRUCT IS ALWAYS AN ADDRESS. `_value` returns a register holding the value
of a scalar and, for a struct, union or array, a `ptr` to its storage. There is
no second representation, so a struct never has to be "materialised" and the
question "is this one in a register" never arises.

PASSING ONE BY VALUE IS PASSING A POINTER TO A FRESH COPY. The caller allocates
and copies; the callee receives a `ptr` and may write through it, exactly as C
says it may write to its own parameter. RETURNING ONE is the mirror: a function
whose C return type is a struct gets a hidden first parameter, the address to
build the result in, and returns `void`. Both are the classic lowerings and
both are self-consistent, which is what matters: every such function on both
sides of a call is compiled by this frontend.

VARARGS ARE AN ARGUMENT AREA. A variadic function gets one extra trailing
parameter -- a `ptr` to a block of 8-byte slots -- and a call to one builds
that block. Every slot is 8 bytes because the default argument promotions
leave only four shapes (`i64`, `u64`, `f64`, `ptr`) and a uniform stride is
what makes `va_arg` a load and an add rather than a table. A struct passed
through `...` occupies its slot as a pointer to a copy, exactly as a named
struct parameter does.

A VLA COMES FROM AN ARENA, because `Op.ALLOCA` takes a literal byte count and
the whole point of a VLA is that the count is not one. The arena is three
functions -- allocate, mark, release -- and they are WRITTEN IN C and compiled
by this frontend, at the end of this file, into the same module. A compiler
whose own support code is written in the language it compiles cannot have a
support routine that its own front end would reject.
"""
from __future__ import annotations

import struct as _struct

from ...diagnostics import DiagnosticSink, SourceFile, Span, warning
from ...ir import types as IR
from ...ir.builder import Builder
from ...ir.module import (
    Block, Function, Global, Instruction, Linkage as IRLinkage, Module,
)
from ...ir.opcodes import Op
from . import ctype as C
from . import syntax as S
from .ctype import CType
from .fold import Address, fold, wrap
from .sema import Linkage, Storage, Symbol

#: The byte width of one slot in a variadic argument area. See the docstring.
VA_SLOT = 8

#: The name the user's `main` is compiled under. The IR's `main` is a wrapper
#: this file generates: every backend's runtime declares the entry point as
#: `int64_t main(void)` and calls it with no arguments, so a C `main` that
#: returns `int` and takes `(int, char **)` cannot BE it.
USER_MAIN = "__c_main"

#: Copying more than this many bytes inline becomes a loop. Below it, a
#: straight run of loads and stores is smaller and has no branch.
_INLINE_COPY_LIMIT = 128


class Unsupported(Exception):
    """A construct with no lowering. Reported, then compilation continues."""


class Lowerer:
    def __init__(self, unit: S.Unit, parser, source: SourceFile,
                 sink: DiagnosticSink, inits: tuple[str, ...] = (), *,
                 init_symbol: str = "") -> None:
        #: THE NAME `--c:init-symbol` GAVE THE UNIT'S INITIALISER, or "".
        #: Named means exported, emitted even when it turns out to be empty,
        #: and nothing to warn about: the caller has said it will run it. See
        #: `_emit_init`.
        self.init_symbol = init_symbol
        #: EVERY UNIT'S INITIALISER, when the build has more than one, so
        #: that the entry point can call all of them. Empty for a single
        #: unit, which calls its own or has none. See `_emit_entry`.
        self.inits = inits
        self.unit = unit
        self.parser = parser
        self.source = source
        self.sink = sink
        self.module = Module(name=source.path.stem if source.path else "c")
        self.module.metadata = {"frontend": "c", "source": source.name}
        self.fn: Function | None = None
        self.b: Builder | None = None
        #: Labels for the innermost loop and switch.
        self.break_to: list[str] = []
        self.continue_to: list[str] = []
        #: `Case.label` -> the block it starts. Filled by `_switch` before the
        #: body is walked, because a `case` may sit anywhere inside it --
        #: including inside a nested block, which is what Duff's device is.
        self._case_blocks: dict[str, Block] = {}
        #: goto targets in the function being lowered: C name -> Block.
        self.labels: dict[str, Block] = {}
        #: Stores that a static initialiser could not put in the global's
        #: bytes -- an address is not known until the linker runs. See
        #: `_emit_init`.
        self.init_stores: list[tuple[str, int, CType, S.Expr]] = []
        self.declared: dict[str, Function] = {}
        self.globals: set[str] = set()
        self.needs_vla = False
        self.needs_args = False
        self.needs_complex = False
        self.needs_ldouble = False
        self.needs_tls = False
        #: `_Thread_local` objects: (the template's name, its size). Their
        #: keys are made by `_emit_init`.
        self.thread_locals: list[tuple[str, int]] = []
        #: Complex temporaries, by position in the expression being lowered.
        #: See `_cx_temp` for why they are pooled and where they live.
        self._cx_pool: list[dict[int, int]] = []
        self._cx_used = 0
        #: This unit's initialiser, named with its own prefix so that two
        #: units' initialisers are two functions.
        self._init_name = init_symbol or (parser.sema.prefix + "init")
        self._temp_n = 0
        #: The register holding this function's variadic argument area.
        self.va_area: int | None = None

    # ── entry ───────────────────────────────────────────────────────────────
    def run(self) -> Module:
        for (prefix, data), name in self.parser.strings.items():
            self._add_global(Global(name, len(data), data, readonly=True,
                                    align=_prefix_align(prefix)))
        for sym in self.parser.sema.file_scope.names.values():
            self._file_object(sym)
        for node in self.anon_literals():
            self._compound_global(node)
        for node in self.unit.decls:
            if isinstance(node, S.FunctionDef):
                self._function(node)
        self._declare_externals()
        self._emit_init()
        self._emit_entry()
        if (self.needs_vla or self.needs_args or self.needs_complex
                or self.needs_ldouble or self.needs_tls):
            self._splice_support()
        self._prune()
        return self.module

    def _prune(self) -> None:
        """Drop what nothing reaches. The walk itself is `prune` below."""
        prune(self.module)

    def anon_literals(self):
        return list(self.parser.anon)

    # ── globals ─────────────────────────────────────────────────────────────
    def _add_global(self, g: Global) -> None:
        if g.name in self.globals:
            return
        self.globals.add(g.name)
        self.module.globals.append(g)

    def _file_object(self, sym: Symbol) -> None:
        """Emit the storage for one file-scope object, if this unit owns it.

        ONE PASS OVER THE SYMBOLS, not over the declarations. A file-scope
        object may be declared more than once --

            extern int shared;
            int shared = 5;

        -- and walking the declarations emits the FIRST one, which has no
        initialiser, and then finds the name already taken and drops the
        second. `shared` came out zero. The symbol carries the composite type
        and the initialiser from whichever declaration supplied them, so it
        is the thing to ask.
        """
        if sym.storage in (Storage.TYPEDEF, Storage.ENUM_CONST):
            return
        if sym.type.is_function:
            return
        if sym.storage is Storage.EXTERN and not sym.defined:
            return                  # declared here, defined elsewhere
        try:
            size = sym.type.size
        except C.IncompleteType:
            return
        mark = len(self.init_stores)
        data = self._static_bytes(sym, sym.init, size)
        linkage = (IRLinkage.EXPORT if sym.linkage is Linkage.EXTERNAL
                   else IRLinkage.INTERNAL)
        name = sym.ir_name or sym.name
        self._add_global(Global(name, max(1, size), data,
                                readonly=self._readonly(sym, mark),
                                linkage=linkage,
                                align=sym.align or sym.type.align,
                                span=sym.span or self.unit.span))
        if sym.thread_local:
            # THE OBJECT ABOVE BECOMES THE TEMPLATE: what a thread's own
            # copy starts as. Beside it goes a key, made once before `main`
            # runs -- which is where the race a key would otherwise have is
            # avoided, because the program is one thread there.
            self._add_global(Global(name + ".key", 8, None, linkage=linkage,
                                    span=sym.span or self.unit.span))
            self.thread_locals.append((name, max(1, size)))

    def _compound_global(self, node: S.CompoundLiteral) -> None:
        size = node.type.size
        mark = len(self.init_stores)
        data = self._init_bytes(node.symbol, node.init, size)
        # READ-ONLY ONLY IF IT SAYS `const`, which is the same rule
        # `_readonly` applies to a named object and for the same reason.
        # A compound literal is a modifiable object unless its type is
        # qualified -- `int *p = (int[]){1, 2}; p[0] = 5;` is an ordinary
        # store -- and putting one in `.rodata` because its initialiser
        # happened to be constant is a program that SEGFAULTS on the store.
        # A per-thread one is never read-only either: `__c_tls_get` copies
        # out of it, and the symmetry with the named case is worth the page.
        self._add_global(Global(node.symbol, max(1, size), data,
                                readonly=(node.type.is_const
                                          and len(self.init_stores) == mark
                                          and not node.thread_local),
                                align=node.type.align))
        if node.thread_local:
            # THE SAME TWO PIECES A NAMED `thread_local` OBJECT GETS: the
            # object above is the template a thread's first use copies, and
            # the key beside it is made once before `main`. Not read-only,
            # whatever the initialiser looks like -- `__c_tls_get` copies
            # OUT of it and a backend that put it in `.rodata` would be
            # right to, but the symmetry with the named case is worth more
            # than the page.
            self._add_global(Global(node.symbol + ".key", 8, None,
                                    span=node.span))
            self.thread_locals.append((node.symbol, max(1, size)))

    def _readonly(self, sym: Symbol, mark: int) -> bool:
        """Whether a global may go in read-only storage.

        A `const` OBJECT WHOSE INITIALISER NEEDS AN ADDRESS IS NOT ONE.
        `Global.data` is bytes and holds no relocations, so

            static const struct Ops RECT = { rect, "rect" };

        is filled in by `__c_init` at run time -- and a backend that put it in
        `.rodata` because it is `const` produced a program that SEGFAULTED on
        the store. The interpreter does not enforce read-only storage, so only
        the compiled path found it; that is what the three-way comparison is
        for.
        """
        return sym.type.is_const and len(self.init_stores) == mark

    def _static_bytes(self, sym: Symbol, init, size: int) -> bytes | None:
        if init is None:
            return None             # zero-filled, which is what C promises
        return self._init_bytes(sym.ir_name or sym.name, init, size)

    def _init_bytes(self, owner: str, init: S.Init, size: int) -> bytes:
        """The bytes a static initialiser produces.

        An entry whose value is the address of something becomes a STORE in
        `__c_init` instead: `Global.data` is bytes and holds no relocations,
        and inventing a fake address here would produce a program that runs
        until it dereferences one.
        """
        out = bytearray(size)
        for e in init.entries:
            if e.data is not None:
                end = min(size, e.offset + len(e.data))
                out[e.offset:end] = e.data[:end - e.offset]
                continue
            if e.value is None:
                continue
            got = fold(e.value)
            if isinstance(got, Address):
                self.init_stores.append((owner, e.offset, e.type, e.value))
                continue
            if got is None:
                continue            # already reported by `_check_static_init`
            self._pack(out, e.offset, e.type, got, e.bits, e.bit_offset)
        return bytes(out)

    def _pack(self, out: bytearray, offset: int, ty: CType, value,
              bits: int | None, bit_offset: int) -> None:
        size = ty.size
        if bits is not None:
            unit = int.from_bytes(out[offset:offset + size], "little")
            mask = ((1 << bits) - 1) << bit_offset
            unit = (unit & ~mask) | ((int(value) << bit_offset) & mask)
            out[offset:offset + size] = unit.to_bytes(size, "little")
            return
        if ty.is_ldouble:
            # THE SIXTEEN BYTES THE FORMAT SAYS, which `ldouble.py` writes
            # and `support.py`'s arithmetic reads.
            from .ldouble import encode
            out[offset:offset + 16] = encode(value)
            return
        if ty.is_complex:
            # TWO NUMBERS IN THE BYTES, real part first, which is the layout
            # every rule about complex assumes and what `creal` reads.
            v = complex(value)
            self._pack(out, offset, ty.of, v.real, None, 0)
            self._pack(out, offset + ty.of.size, ty.of, v.imag, None, 0)
            return
        if ty.is_float:
            raw = _struct.pack("<f" if size == 4 else "<d", float(value))
        else:
            raw = (wrap(int(value), ty) & ((1 << (size * 8)) - 1)).to_bytes(
                size, "little")
        out[offset:offset + size] = raw

    # ── externals ───────────────────────────────────────────────────────────
    def _declare_externals(self) -> None:
        """Every function the module calls and does not define.

        The verifier requires a `call` to name a function in the module, so a
        declaration has to exist for each -- and the C backend generates its
        `extern` from the IR signature, which is why the parameter types are
        written out rather than left blank: without them C assumes `int f()`,
        and a call to something returning `double` reads the wrong register.
        """
        for sym in self.parser.sema.file_scope.names.values():
            if not sym.type.is_function or sym.has_body:
                continue
            if sym.storage is Storage.TYPEDEF:
                continue
            self._declare_function(sym.ir_name or sym.name, sym.type,
                                   external=True, span=sym.span)

    def _ensure_extern(self, name: str, ret: IR.Type,
                       params: list[IR.Type]) -> None:
        """Declare a library function a builtin lowers to a call of.

        The verifier requires every `call` to name a function in the module,
        and a program may reach `memcpy` through `__builtin_memcpy` without
        ever including `<string.h>`. A declaration built from the signature
        the call actually uses is what keeps the C backend's generated
        `extern` right.
        """
        if self.module.function(name) is not None:
            return
        fn = Function(name, ret, external=True, linkage=IRLinkage.IMPORT,
                      span=self.unit.span)
        for p in params:
            fn.params.append(fn.new_register(p))
        self.module.functions.append(fn)

    def _declare_function(self, name: str, ty: CType, *, external: bool,
                          span: Span | None = None) -> Function:
        got = self.module.function(name)
        if got is not None:
            return got
        ret, params, _ = _signature(ty)
        fn = Function(name, ret, external=external,
                      linkage=IRLinkage.IMPORT if external else IRLinkage.EXPORT,
                      span=span or self.unit.span)
        for p in params:
            fn.params.append(fn.new_register(p))
        self.module.functions.append(fn)
        self.declared[name] = fn
        return fn

    # ── functions ───────────────────────────────────────────────────────────
    def _function(self, node: S.FunctionDef) -> None:
        sym = node.sym
        name = sym.ir_name or sym.name
        if name == "main":
            name = USER_MAIN
        ty = node.type
        ret_ir, param_ir, sret = _signature(ty)
        existing = self.module.function(name)
        if existing is not None and existing.external:
            self.module.functions.remove(existing)
        fn = Function(name, ret_ir,
                      linkage=(IRLinkage.INTERNAL if sym.linkage is Linkage.INTERNAL
                               else IRLinkage.EXPORT),
                      span=node.span)
        self.module.functions.append(fn)
        self.declared[name] = fn
        self.fn = fn
        # THE TEMPORARY POOL IS PER FUNCTION: its slots are allocas in THIS
        # function's entry block, and a register from the last one means
        # nothing here.
        self._cx_pool = []
        self._cx_used = 0
        entry = Block("entry")
        fn.blocks.append(entry)
        self.b = Builder(fn)
        self.b.span = node.span
        self.labels = {}
        self.break_to = []
        self.continue_to = []
        self.va_area = None

        self.sret = fn.new_register(IR.PTR) if sret else None
        if self.sret is not None:
            fn.params.append(self.sret)
        for psym in node.params:
            reg = fn.new_register(_param_ir(psym.type))
            fn.params.append(reg)
            # AN AGGREGATE PARAMETER ARRIVES AS A POINTER, so its register
            # holds an ADDRESS and not a value -- which is precisely what a
            # non-register symbol's `slot` means. Marking it as living in a
            # register would make `p.a` ask for the address of a register.
            psym.in_register = not psym.type.in_memory
            psym.slot = reg
        if node.variadic:
            self.va_area = fn.new_register(IR.PTR)
            fn.params.append(self.va_area)

        # A PARAMETER THAT IS ADDRESSED NEEDS A SLOT, and the value is copied
        # into it on entry. An aggregate parameter already IS an address, so
        # it keeps the register it arrived in.
        for psym in node.params:
            if not psym.in_register:
                continue
            if psym.addressed:
                slot = self.b.alloca(max(1, psym.type.size))
                self.b.store(C.to_ir(psym.type), psym.slot, slot)
                psym.slot = slot
                psym.in_register = False
        for local in node.locals:
            self._allocate_local(local)
        for name_, label in node.labels.items():
            self.labels[name_] = self.b.new_block(f"L{len(self.labels)}")

        self._block(node.body)
        self._finish_function(node)
        self.fn = None
        self.b = None

    def _allocate_local(self, sym: Symbol) -> None:
        if sym.storage is Storage.EXTERN:
            # `extern int x;` inside a function names a global defined
            # elsewhere. Emitting one here would define it twice.
            sym.in_register = False
            sym.slot = None
            return
        if sym.is_global:
            # A block-scope `static`: an ordinary global with a private name.
            size = sym.type.size
            mark = len(self.init_stores)
            data = self._static_bytes(sym, sym.init, size)
            self._add_global(Global(sym.ir_name, max(1, size), data,
                                    readonly=self._readonly(sym, mark),
                                    align=sym.align or sym.type.align,
                                    span=sym.span or self.unit.span))
            sym.in_register = False
            sym.slot = None
            return
        if sym.type.is_vla:
            sym.in_register = False
            sym.slot = self.b.alloca(8)     # holds the arena address
            self.needs_vla = True
            return
        if sym.type.in_memory or sym.addressed:
            sym.in_register = False
            sym.slot = self._aligned_slot(sym.type.size,
                                          sym.align or sym.type.align)
            return
        # EVERY REGISTER LOCAL IS ZEROED IN THE ENTRY BLOCK, not where it is
        # declared. `goto L; int x; L: use(x);` jumps over the declaration and
        # is legal C, and the IR verifier requires a register to be written on
        # every path that reads it -- so a write at the declaration would make
        # a legal program fail to verify. Reading an uninitialised object is
        # undefined behaviour, and zero is a permitted value for it.
        sym.in_register = True
        ir = C.to_ir(sym.type)
        sym.slot = self.b.reg(ir)
        zero = self.b.const(ir, 0.0 if ir.is_float else 0)
        self.b.copy(sym.slot, zero)

    def _finish_function(self, node: S.FunctionDef) -> None:
        blk = self.b.current
        if blk.terminator is not None:
            self._drop_unreachable()
            return
        ret = node.type.ret
        if ret.is_void or self.sret is not None:
            self.b.emit(Instruction(Op.RET, IR.VOID))
        else:
            # FALLING OFF THE END of a non-void function is undefined
            # behaviour in C, and the IR requires every path to end in a
            # `ret` of the right type. Zero is the value that makes the
            # program verifiable without inventing a branch.
            ir = C.to_ir(ret)
            self.b.emit(Instruction(Op.RET, ir,
                                    args=[self.b.const(ir, 0.0 if ir.is_float
                                                       else 0)]))
        self._drop_unreachable()

    def _drop_unreachable(self) -> None:
        """Terminate any block nothing terminated, and drop empty ones.

        A `return` in the middle of a block leaves the statements after it in
        a new block that nothing jumps to. The IR requires every block to end
        in a terminator, so an empty tail gets `unreachable` -- and an
        entirely empty one is removed, because a block with no instructions
        is not legal IR at all.
        """
        kept = []
        for blk in self.fn.blocks:
            if not blk.instructions:
                if blk is self.fn.blocks[0]:
                    blk.instructions.append(Instruction(Op.UNREACHABLE, IR.VOID))
                    kept.append(blk)
                continue
            if blk.terminator is None:
                blk.instructions.append(Instruction(Op.UNREACHABLE, IR.VOID))
            kept.append(blk)
        live = {b.label for b in kept}
        for blk in kept:
            t = blk.terminator
            if t is None:
                continue
            for i, lbl in enumerate(t.labels):
                if lbl not in live:
                    t.labels[i] = kept[0].label
            t.cases = [(v, lbl) for v, lbl in t.cases if lbl in live]
        self.fn.blocks = kept

    # ── the entry point and the static initialiser ──────────────────────────
    def _emit_init(self) -> None:
        """The stores a static initialiser could not put in a global's bytes.

        ALWAYS EMITTED IN A MULTI-UNIT BUILD, even when it is empty, and
        EXPORTED. The entry point is in whichever unit defines `main`, and it
        cannot know whether another unit needed one -- so every unit has one
        and the entry calls them all. An empty one costs a `ret`, which the
        alternative (working out afterwards which units produced one and
        editing the entry block) does not come close to.
        """
        if (not self.init_stores and not self.inits
                and not self.thread_locals and not self.init_symbol):
            return
        # A NAMED INITIALISER IS EXPORTED AND IS EMITTED EMPTY. The unit that
        # asked for the name calls it by that name, in C, through an `extern`
        # declaration -- and a declaration with nothing behind it is an
        # undefined symbol at link time. An empty one costs a `ret`.
        fn = Function(self._init_name, IR.VOID, span=self.unit.span,
                      linkage=(IRLinkage.EXPORT
                               if (self.inits or self.init_symbol)
                               else IRLinkage.INTERNAL))
        # THE C SIDE MAY HAVE DECLARED IT FIRST, which is how it calls it.
        # That declaration is an IMPORT function of the same name, and two
        # functions of one name is what the verifier reports. `_function`
        # drops the same declaration when a definition arrives.
        prior = self.module.function(self._init_name)
        if prior is not None and prior.external:
            self.module.functions.remove(prior)
        self.module.functions.append(fn)
        self.fn = fn
        fn.blocks.append(Block("entry"))
        self.b = Builder(fn)
        self.sret = None
        # THE KEYS FIRST, and before `main`: a key made here is made by one
        # thread, which is what makes `__c_tls_get` need no lock.
        for name, _size in self.thread_locals:
            self._ensure_extern("host_tss_new", IR.I64, [IR.PTR])
            key = self.b.call(IR.I64, "host_tss_new",
                              [self.b.const(IR.PTR, 0)])
            self.b.store(IR.I64, key, self._global_addr(name + ".key"))
        for owner, offset, ty, value in self.init_stores:
            base = self._global_addr(owner)
            addr = self._offset(base, offset)
            self._store_value(ty, value, addr)
        self.b.emit(Instruction(Op.RET, IR.VOID))
        self.fn = None
        self.b = None

    def _emit_entry(self) -> None:
        """The IR's `main`, which is not the program's.

        Every backend's runtime declares the entry point as `int64_t
        main(void)` and calls it with no arguments. A C `main` returns `int`
        and may take `(int, char **)`, so it is compiled under another name
        and this calls it -- which is also the one place a static initialiser
        can be run before the program starts.
        """
        user = self.module.function(USER_MAIN)
        if user is None:
            # IN A MULTI-UNIT BUILD THIS IS THE ORDINARY CASE -- only one
            # unit has `main` -- and the frontend says so once, after the
            # merge, if no unit had one at all.
            if ((self.init_stores or self.thread_locals)
                    and not self.inits and not self.init_symbol):
                self.sink.report(
                    warning("W1500",
                            "this unit has static initialisers that need "
                            "addresses, and no `main` to run them from")
                    .at(self.unit.span)
                    .note("they are applied by `__c_init`, which the program "
                          "entry point calls")
                    .help("compile the unit that defines `main` together "
                          "with this one"))
            return
        fn = Function("main", IR.I64, linkage=IRLinkage.EXPORT,
                      span=user.span)
        self.module.functions.append(fn)
        self.fn = fn
        fn.blocks.append(Block("entry"))
        self.b = Builder(fn)
        self.sret = None
        # EVERY UNIT'S, IN UNIT ORDER. C does not say in what order the
        # translation units of a program are initialised, only that it
        # happens before `main` runs; unit order is the one a reader can
        # predict.
        for name in (self.inits or
                     ((self._init_name,)
                      if (self.init_stores or self.thread_locals) else ())):
            self.b.call(IR.VOID, name, [])
        args = self._main_args(user)
        got = self.b.call(user.ret, USER_MAIN, args)
        if user.ret.is_void:
            self.b.emit(Instruction(Op.RET, IR.I64,
                                    args=[self.b.const(IR.I64, 0)]))
        else:
            self.b.emit(Instruction(Op.RET, IR.I64,
                                    args=[self._ir_convert(got, user.ret, IR.I64)]))
        self.fn = None
        self.b = None

    def _main_args(self, user: Function) -> list[int]:
        """What to hand `main`, which is `argc` and `argv` when it asks.

        THE COMMAND LINE IS `objects/hostsvc.py`'s `env` GROUP, reached
        through the two functions `support.py`'s `args` unit is written
        around. A `main(void)` never mentions them, so a program that does
        not ask still runs on a backend whose target has no command line --
        and one that does ask is refused there by name, at compile time,
        rather than being handed zero arguments and left to wonder.

        THE THIRD PARAMETER IS NULL. `main(argc, argv, envp)` is a common
        extension and not C; the host services can look a variable UP by
        name and cannot enumerate them, so there is no array to point at. A
        null pointer is what a program that tests before walking expects,
        and W1501 says so for one that would not have.
        """
        args: list[int] = []
        for i, reg in enumerate(user.params):
            ty = user.registers[reg]
            if i == 0 and not ty.is_ptr:
                self.needs_args = True
                got = self.b.call(IR.I64, "__c_args_count", [])
                args.append(self._ir_convert(got, IR.I64, ty))
            elif i == 1 and ty.is_ptr:
                self.needs_args = True
                args.append(self.b.call(IR.PTR, "__c_args_build", []))
            else:
                if ty.is_ptr:
                    self.sink.report(
                        warning("W1501",
                                "`main` declares a third parameter, and the "
                                "environment is not an array here")
                        .at(user.span)
                        .note("the `env` host service looks a name up; it "
                              "cannot list what is set")
                        .help("this parameter is a null pointer; `getenv` "
                              "reads the environment"))
                    args.append(self._null_argv())
                else:
                    args.append(self.b.const(ty, 0))
        return args

    def _null_argv(self) -> int:
        self._add_global(Global("__c_argv", 8, bytes(8), readonly=True))
        return self._global_addr("__c_argv")

    # ── statements ──────────────────────────────────────────────────────────
    def _block(self, node: S.Compound) -> None:
        mark = None
        if node.has_vla:
            mark = self.b.call(IR.PTR, "__c_vla_mark", [])
            self.needs_vla = True
        for item in node.items:
            if isinstance(item, S.Decl):
                self._local_decl(item)
            else:
                self._stmt(item)
        if mark is not None and self.b.current.terminator is None:
            self.b.call(IR.VOID, "__c_vla_release", [mark])

    def _local_decl(self, node: S.Decl) -> None:
        sym = node.sym
        if sym is None or sym.storage is Storage.TYPEDEF or sym.is_global:
            return
        if sym.type.is_function:
            return
        self.b.span = node.span
        if node.vla_size is not None:
            size = self._value(node.vla_size)
            addr = self.b.call(IR.PTR, "__c_vla_alloc",
                               [self._ir_convert(size,
                                                 C.to_ir(node.vla_size.type),
                                                 IR.I64)])
            self.b.store(IR.PTR, addr, sym.slot)
            self.needs_vla = True
            return
        if node.init is None:
            return
        if sym.in_register:
            entry = node.init.entries[0] if node.init.entries else None
            if entry is not None and entry.value is not None:
                self.b.copy(sym.slot, self._rvalue(entry.value, sym.type))
            return
        self._emit_init_into(sym.slot, sym.type, node.init)

    def _emit_init_into(self, addr: int, ty: CType, init: S.Init) -> None:
        """Run an initialiser into storage at `addr`.

        THE WHOLE OBJECT IS ZEROED FIRST when the initialiser does not fill
        it. C says any member not initialised is as if it were a static zero,
        and finding which bytes those are means walking the layout -- which
        the zero already did, faster and with no edge cases around padding.
        """
        size = ty.size
        covered = sum(len(e.data) if e.data is not None else e.type.size
                      for e in init.entries)
        if covered < size or any(e.bits is not None for e in init.entries):
            self._zero(addr, size)
        for e in init.entries:
            target = self._offset(addr, e.offset)
            if e.data is not None:
                blob = self._blob(e.data)
                self._copy(target, blob, len(e.data), 1)
                continue
            if e.value is None:
                continue
            if e.bits is not None:
                self._store_bitfield(target, e.type, e.bits, e.bit_offset,
                                     self._value(e.value))
                continue
            self._store_value(e.type, e.value, target)

    def _store_value(self, ty: CType, value: S.Expr, addr: int) -> None:
        if ty.in_memory:
            src = self._value(value)
            self._copy(addr, src, ty.size, ty.align)
            return
        self.b.store(C.to_ir(ty), self._rvalue(value, ty), addr)

    def _blob(self, data: bytes) -> int:
        name = f".blob.{len(self.module.globals)}"
        self._add_global(Global(name, len(data), data, readonly=True))
        return self._global_addr(name)

    def _stmt(self, node) -> None:
        if node is None:
            return
        # THE COMPLEX TEMPORARIES ARE DEAD AT A STATEMENT BOUNDARY, so the
        # pool starts again here and a loop reuses one set rather than
        # allocating a set per turn. See `_cx_temp`.
        self._cx_used = 0
        self.b.span = node.span
        match node:
            case S.Compound():
                self._block(node)
            case S.ExprStmt():
                if node.expr is not None:
                    self._discard(node.expr)
            case S.Empty():
                pass
            case S.If():
                self._if(node)
            case S.While():
                self._while(node)
            case S.DoWhile():
                self._do(node)
            case S.For():
                self._for(node)
            case S.Switch():
                self._switch(node)
            case S.Case() | S.Default():
                self._case(node)
            case S.Label():
                self._label(node)
            case S.Goto():
                self._goto(node)
            case S.Break():
                if self.break_to:
                    self._jump(self.break_to[-1])
            case S.Continue():
                if self.continue_to:
                    self._jump(self.continue_to[-1])
            case S.Return():
                self._return(node)
            case _:
                raise AssertionError(f"no lowering for {type(node).__name__}")

    def _jump(self, label: str) -> None:
        if self.b.current.terminator is None:
            self.b.jump(label)
        self._open(self.b.new_block("dead"))

    def _open(self, block: Block) -> None:
        self.b.switch_to(block)

    def _goto_block(self, block: Block) -> None:
        if self.b.current.terminator is None:
            self.b.jump(block)
        self._open(block)

    def _if(self, node: S.If) -> None:
        then = self.b.new_block("then")
        after = self.b.new_block("endif")
        other = self.b.new_block("else") if node.otherwise is not None else after
        self._branch(node.cond, then, other)
        self._open(then)
        self._stmt(node.then)
        if self.b.current.terminator is None:
            self.b.jump(after)
        if node.otherwise is not None:
            self._open(other)
            self._stmt(node.otherwise)
            if self.b.current.terminator is None:
                self.b.jump(after)
        self._open(after)

    def _while(self, node: S.While) -> None:
        head = self.b.new_block("while")
        body = self.b.new_block("body")
        after = self.b.new_block("endwhile")
        self._goto_block(head)
        self._branch(node.cond, body, after)
        self._open(body)
        self.break_to.append(after.label)
        self.continue_to.append(head.label)
        self._stmt(node.body)
        self.break_to.pop()
        self.continue_to.pop()
        if self.b.current.terminator is None:
            self.b.jump(head)
        self._open(after)

    def _do(self, node: S.DoWhile) -> None:
        body = self.b.new_block("do")
        test = self.b.new_block("dotest")
        after = self.b.new_block("enddo")
        self._goto_block(body)
        self.break_to.append(after.label)
        self.continue_to.append(test.label)
        self._stmt(node.body)
        self.break_to.pop()
        self.continue_to.pop()
        if self.b.current.terminator is None:
            self.b.jump(test)
        self._open(test)
        self._branch(node.cond, body, after)
        self._open(after)

    def _for(self, node: S.For) -> None:
        if isinstance(node.init, list):
            for d in node.init:
                self._local_decl(d)
        elif isinstance(node.init, S.ExprStmt):
            self._discard(node.init.expr)
        head = self.b.new_block("for")
        body = self.b.new_block("forbody")
        step = self.b.new_block("forstep")
        after = self.b.new_block("endfor")
        self._goto_block(head)
        if node.cond is None:
            self.b.jump(body)
        else:
            self._branch(node.cond, body, after)
        self._open(body)
        self.break_to.append(after.label)
        self.continue_to.append(step.label)
        self._stmt(node.body)
        self.break_to.pop()
        self.continue_to.pop()
        if self.b.current.terminator is None:
            self.b.jump(step)
        self._open(step)
        if node.step is not None:
            self._discard(node.step)
        if self.b.current.terminator is None:
            self.b.jump(head)
        self._open(after)

    def _switch(self, node: S.Switch) -> None:
        value = self._value(node.expr)
        after = self.b.new_block("endswitch")
        blocks: dict[str, Block] = {}
        for _, label in node.cases:
            blocks.setdefault(label, self.b.new_block("case"))
        if node.default_label:
            blocks.setdefault(node.default_label, self.b.new_block("default"))
        self._case_blocks.update(blocks)
        default = blocks.get(node.default_label) if node.default_label else after
        cases = [(v, blocks[label].label) for v, label in node.cases]
        self.b.emit(Instruction(Op.SWITCH, C.to_ir(node.expr.type),
                                args=[value], labels=[default.label],
                                cases=cases))
        self._open(self.b.new_block("swbody"))
        self.break_to.append(after.label)
        self._stmt(node.body)
        self.break_to.pop()
        if self.b.current.terminator is None:
            self.b.jump(after)
        self._open(after)

    def _case(self, node) -> None:
        block = self._case_blocks.get(node.label)
        if block is None:
            self._stmt(node.body)
            return
        self._goto_block(block)
        self._stmt(node.body)

    def _label(self, node: S.Label) -> None:
        block = self.labels.get(node.name)
        if block is None:
            self._stmt(node.body)
            return
        self._goto_block(block)
        self._stmt(node.body)

    def _goto(self, node: S.Goto) -> None:
        block = self.labels.get(node.name)
        if block is None:
            return
        if self.b.current.terminator is None:
            self.b.jump(block)
        self._open(self.b.new_block("dead"))

    def _return(self, node: S.Return) -> None:
        if self.sret is not None:
            if node.value is not None:
                src = self._value(node.value)
                self._copy(self.sret, src, node.value.type.size,
                           node.value.type.align)
            self.b.emit(Instruction(Op.RET, IR.VOID))
        elif node.value is None:
            self.b.emit(Instruction(Op.RET, IR.VOID))
        else:
            ir = C.to_ir(node.value.type)
            self.b.emit(Instruction(Op.RET, ir, args=[self._value(node.value)]))
        self._open(self.b.new_block("dead"))

    # ── expressions ─────────────────────────────────────────────────────────
    def _discard(self, e: S.Expr) -> None:
        """Evaluate for effect. The value is dropped; `dce` removes the rest."""
        if isinstance(e, S.Comma):
            self._discard(e.left)
            self._discard(e.right)
            return
        if isinstance(e, S.Logical):
            self._value(e)
            return
        if isinstance(e, (S.Conv, S.Cast)) and e.type.is_void:
            self._discard(e.operand)
            return
        self._value(e)

    def _rvalue(self, e: S.Expr, want: CType) -> int:
        """A value converted to `want`. Only the scalar case needs anything."""
        if want.is_ldouble or e.type.is_ldouble:
            return self._ld_convert(e, want)
        if want.is_complex:
            # A REAL VALUE WIDENS INTO A COMPLEX ONE with a zero imaginary
            # part. Sema usually inserts the conversion node, but an
            # initialiser reaches here with the operand as written.
            return self._cx_convert(e, want)
        got = self._value(e)
        if want.in_memory:
            return got
        return self._ir_convert(got, e.type, C.to_ir(want))

    def _value(self, e: S.Expr) -> int:
        self.b.span = e.span
        match e:
            case S.IntLit():
                return self.b.const(C.to_ir(e.type), wrap(e.value, e.type)
                                    if e.type.is_integer else e.value)
            case S.FloatLit():
                if e.type.is_ldouble:
                    return self._ld_const(e.value)
                if e.type.is_complex:
                    elem = C.to_ir(e.type.of)
                    return self._cx_make(
                        e.type, self.b.const(elem, e.value.real),
                        self.b.const(elem, e.value.imag))
                return self.b.const(C.to_ir(e.type), e.value)
            case S.StringLit():
                return self._global_addr(e.symbol)
            case S.Ident():
                return self._load_lvalue(e)
            case S.SizeofType():
                if e.dynamic is not None:
                    return self._value(e.dynamic)
                return self.b.const(IR.U64,
                                    e.operand_type.align if e.alignment
                                    else e.operand_type.size)
            case S.Conv() | S.Cast():
                return self._conversion(e)
            case S.Unary():
                return self._unary(e)
            case S.Binary():
                return self._binary(e)
            case S.Logical():
                return self._logical(e)
            case S.Comma():
                self._discard(e.left)
                return self._value(e.right)
            case S.Conditional():
                return self._conditional(e)
            case S.Assign():
                return self._assign(e)
            case S.Index() | S.MemberAccess():
                return self._load_lvalue(e)
            case S.Call():
                return self._call(e)
            case S.CompoundLiteral():
                # AN LVALUE LIKE ANY OTHER. `(int){5}` has type `int` and a
                # location; `_compound_literal` yields the location, so a
                # scalar one still has to be read through. Returning the
                # address made `return (int){5}` return a pointer, and the
                # verifier caught it -- which is what it is for.
                return self._load_lvalue(e)
            case S.VaArg():
                return self._va_arg(e)
            case S.BuiltinCall():
                return self._builtin(e)
            case S.StmtExpr():
                return self._stmt_expr(e)
        raise AssertionError(f"no lowering for {type(e).__name__}")

    def _load_lvalue(self, e: S.Expr) -> int:
        """Read an lvalue. An aggregate reads as its own address."""
        if isinstance(e, S.Ident) and e.sym is not None:
            sym = e.sym
            if sym.type.is_function:
                d = self.b.reg(IR.PTR)
                self.b.emit(Instruction(Op.FUNC_ADDR, IR.PTR, dst=d,
                                        sym=sym.ir_name or sym.name))
                return d
            if sym.in_register:
                return sym.slot
        if e.type.in_memory:
            return self._address(e)
        if isinstance(e, S.MemberAccess) and e.path and e.path[-1].is_bitfield:
            member = e.path[-1]
            addr = self._address(e)
            return self._load_bitfield(addr, member.type, member.bits,
                                       member.bit_offset)
        return self.b.load(C.to_ir(e.type), self._address(e))

    def _address(self, e: S.Expr) -> int:
        """The address of an lvalue."""
        self.b.span = e.span
        match e:
            case S.Ident():
                sym = e.sym
                if sym is None:
                    raise Unsupported("address of an unresolved name")
                if sym.type.is_function:
                    d = self.b.reg(IR.PTR)
                    self.b.emit(Instruction(Op.FUNC_ADDR, IR.PTR, dst=d,
                                            sym=sym.ir_name or sym.name))
                    return d
                if sym.is_global:
                    if sym.thread_local:
                        return self._tls_address(sym)
                    return self._global_addr(sym.ir_name or sym.name)
                if sym.type.is_vla:
                    return self.b.load(IR.PTR, sym.slot)
                if sym.in_register:
                    raise Unsupported(f"address of the register local {sym.name}")
                return sym.slot
            case S.StringLit():
                return self._global_addr(e.symbol)
            case S.Unary() if e.op == "*":
                return self._value(e.operand)
            case S.Index():
                base = self._value(e.base)
                index = self._value(e.index)
                return self._scaled(base, index, e.index.type, e.scale)
            case S.MemberAccess():
                base = self._value(e.base) if e.arrow else self._address(e.base)
                return self._offset(base, sum(m.offset for m in e.path))
            case S.Conv() | S.Cast():
                return self._address(e.operand)
            case S.CompoundLiteral():
                return self._compound_literal(e)
            case S.Conditional():
                return self._conditional(e, address=True)
            case S.Comma():
                self._discard(e.left)
                return self._address(e.right)
            case S.Assign():
                self._assign(e)
                return self._address(e.target)
            case S.Call():
                return self._call(e)
            case S.StmtExpr():
                return self._stmt_expr(e)
        raise Unsupported(f"address of {type(e).__name__}")

    def _tls_address(self, sym: Symbol) -> int:
        """Where THIS thread's copy of a `_Thread_local` object lives.

        A LOOKUP AND NOT AN ADDRESS, which is the whole difference from
        every other global: the answer depends on which thread is asking.
        `__c_tls_get` reads the key, asks the host service for this thread's
        pointer, and makes one from the template the first time -- so the
        first use in a thread is an allocation and every later one is a
        load.

        WHAT IT COSTS is `objects/hostsvc.py`'s `thread` group: a program
        with a `_Thread_local` object is refused on a backend whose target
        has no threads, by name. That is the honest reading of what the
        program asked for.
        """
        return self._tls_lookup(sym.ir_name or sym.name,
                                max(1, sym.type.size))

    def _tls_lookup(self, name: str, size: int) -> int:
        """The lookup itself, by name and size, so that a `thread_local`
        COMPOUND LITERAL -- which has a global but no `Symbol` -- reaches the
        same three-argument call rather than a copy of it."""
        self.needs_tls = True
        self._ensure_extern("__c_tls_get", IR.PTR, [IR.PTR, IR.I64, IR.PTR])
        return self.b.call(IR.PTR, "__c_tls_get",
                           [self._global_addr(name + ".key"),
                            self.b.const(IR.I64, size),
                            self._global_addr(name)])

    def _global_addr(self, name: str) -> int:
        d = self.b.reg(IR.PTR)
        self.b.emit(Instruction(Op.GLOBAL_ADDR, IR.PTR, dst=d, sym=name))
        return d

    def _offset(self, base: int, byte_offset: int) -> int:
        if byte_offset == 0:
            return base
        return self.b.offset(base, self.b.const(IR.I64, byte_offset))

    def _scaled(self, base: int, index: int, index_ty: CType, scale) -> int:
        idx = self._ir_convert(index, index_ty, IR.I64)
        if isinstance(scale, int):
            if scale != 1:
                idx = self.b.mul(IR.I64, idx, self.b.const(IR.I64, scale))
        else:
            # A VARIABLE-LENGTH ELEMENT: the stride is an expression, so it is
            # computed here and multiplied in. See `sema.scale_of`.
            width = self._ir_convert(self._value(scale), scale.type, IR.I64)
            idx = self.b.mul(IR.I64, idx, width)
        return self.b.offset(base, idx)

    # ── conversions ─────────────────────────────────────────────────────────
    def _conversion(self, e) -> int:
        src = e.operand
        if e.type.is_void:
            self._discard(src)
            return self.b.const(IR.I64, 0)
        if src.type.is_array or src.type.is_function:
            # Decay: the address IS the value, and an array lvalue already
            # evaluates to its address.
            return self._value(src) if src.type.is_function else \
                self._address(src) if _addressable(src) else self._value(src)
        if e.type.is_ldouble or src.type.is_ldouble:
            return self._ld_convert(src, e.type)
        if e.type.is_complex or src.type.is_complex:
            return self._cx_convert(src, e.type)
        if e.type.is_record or e.type.is_array:
            return self._value(src)
        return self._ir_convert(self._value(src), src.type, C.to_ir(e.type),
                                target=e.type)

    def _narrow_bitint(self, reg: int, ty: CType) -> int:
        """Put back the bits a `_BitInt(N)` does not have.

        THE CONTAINER IS WIDER THAN THE TYPE. `_BitInt(13)` lives in an i16,
        and an addition that carried into bit 13 has produced a value the
        type cannot hold -- so every operation whose RESULT is bit-precise
        ends here and the value is normalised again: sign-extended from bit
        N-1 if it is signed, masked if it is not. Do it once, at the end of
        the operation, and every later read of the object is already right.

        THE SHIFT PAIR AND NOT A MASK for the signed case, for the same
        reason `_load_bitfield` uses one: `SHR` is arithmetic on a signed IR
        type and logical on an unsigned one, so shifting up and back down is
        the sign extension, with no second constant and no branch.
        """
        if not ty.is_bitint:
            return reg
        ir = C.to_ir(ty)
        spare = ir.bits - ty.width
        if spare == 0:
            return reg              # the width IS the container's
        if not ty.signed:
            d = self.b.reg(ir)
            self.b.emit(Instruction(Op.AND, ir, dst=d,
                                    args=[reg, self.b.const(
                                        ir, (1 << ty.width) - 1)]))
            return d
        up = self.b.reg(ir)
        self.b.emit(Instruction(Op.SHL, ir, dst=up,
                                args=[reg, self.b.const(ir, spare)]))
        down = self.b.reg(ir)
        self.b.emit(Instruction(Op.SHR, ir, dst=down,
                                args=[up, self.b.const(ir, spare)]))
        return down

    def _ir_convert(self, reg: int, src: CType | IR.Type, want: IR.Type,
                    target: CType | None = None) -> int:
        """Convert a value between IR types, following C's rules.

        THE NARROWING IS HERE AND NOT IN THE CALLER because a conversion INTO
        a bit-precise type is one of the two ways a value of one is made --
        `_binary` is the other -- and because the IR conversion alone is not
        enough even when the two land in the same container: `_BitInt(13)` to
        `_BitInt(9)` is i16 to i16, and the four bits in between have to go.
        """
        got = self._ir_convert_to(reg, src, want, target)
        if target is not None and target.is_bitint:
            return self._narrow_bitint(got, target)
        return got

    def _ir_convert_to(self, reg: int, src: CType | IR.Type, want: IR.Type,
                       target: CType | None = None) -> int:
        have = src if isinstance(src, IR.Type) else C.to_ir(src)
        if have is want:
            return reg
        if target is not None and target.is_bool:
            return self._truthy(reg, have)
        if have.is_ptr and want.is_int:
            reg = self._bitcast(reg, IR.U64)
            have = IR.U64
        elif have.is_int and want.is_ptr:
            reg = self._int_convert(reg, have, IR.U64)
            return self._bitcast(reg, IR.PTR)
        if have.is_float and want.is_float:
            return self._emit_conv(Op.FTOF, want, reg)
        if have.is_float and want.is_int:
            return self._emit_conv(Op.FTOI, want, reg)
        if have.is_int and want.is_float:
            return self._emit_conv(Op.ITOF, want, reg)
        if have.is_int and want.is_int:
            return self._int_convert(reg, have, want)
        if have is want:
            return reg
        raise Unsupported(f"conversion {have} -> {want}")

    def _int_convert(self, reg: int, have: IR.Type, want: IR.Type) -> int:
        if have is want:
            return reg
        if have is IR.I1:
            # An i1 holds 0 or 1 and widening it must not sign-extend it to
            # -1. `EXTEND` reads the SOURCE's signedness and `i1` is spelled
            # with an `i`, so the bits go through a same-width unsigned type
            # first -- except that there is no `u1`, and every backend and the
            # interpreter already produce 0 or 1 for a comparison. So the
            # widening is a plain EXTEND and the value is right by
            # construction; this comment exists so the next reader does not
            # have to re-derive that.
            return self._emit_conv(Op.EXTEND, want, reg)
        if have.bits == want.bits:
            return self._bitcast(reg, want)
        if want.bits < have.bits:
            return self._emit_conv(Op.TRUNC, want, reg)
        return self._emit_conv(Op.EXTEND, want, reg)

    def _bitcast(self, reg: int, want: IR.Type) -> int:
        if self.fn.register_type(reg) is want:
            return reg
        return self._emit_conv(Op.BITCAST, want, reg)

    def _emit_conv(self, op: Op, want: IR.Type, reg: int) -> int:
        d = self.b.reg(want)
        self.b.emit(Instruction(op, want, dst=d, args=[reg]))
        return d

    def _truthy(self, reg: int, have: IR.Type) -> int:
        """`x != 0` as an i1, whatever `x` is."""
        if have is IR.I1:
            return reg
        if have.is_ptr:
            reg = self._bitcast(reg, IR.U64)
            have = IR.U64
        zero = self.b.const(have, 0.0 if have.is_float else 0)
        return self.b.cmp(Op.NE, have, reg, zero)

    # ── operators ───────────────────────────────────────────────────────────
    def _unary(self, e: S.Unary) -> int:
        op = e.op
        if op == "&":
            return self._address(e.operand)
        if op == "*":
            return self._load_lvalue(e)
        if op in ("++", "--"):
            return self._incdec(e)
        if op in ("__real__", "__imag__"):
            return self._cx_part(e.operand, op == "__imag__")
        if op == "!":
            # BEFORE THE OPERAND IS EVALUATED BELOW, because `_truth_of`
            # evaluates it itself -- and evaluating it twice would call
            # `!f()`'s function twice.
            got = self._truth_of(e.operand)
            one = self.b.const(IR.I1, 1)
            flipped = self.b.reg(IR.I1)
            self.b.emit(Instruction(Op.XOR, IR.I1, dst=flipped,
                                    args=[got, one]))
            return self._int_convert(flipped, IR.I1, C.to_ir(e.type))
        if e.type.is_ldouble:
            return self._ld_unary(e)
        if e.type.is_complex:
            return self._cx_unary(e)
        value = self._value(e.operand)
        ir = C.to_ir(e.type)
        if op == "+":
            return value
        if op == "-":
            d = self.b.reg(ir)
            self.b.emit(Instruction(Op.NEG, ir, dst=d, args=[value]))
            return self._narrow_bitint(d, e.type)
        if op == "~":
            d = self.b.reg(ir)
            self.b.emit(Instruction(Op.NOT, ir, dst=d, args=[value]))
            return self._narrow_bitint(d, e.type)
        raise AssertionError(op)

    def _incdec(self, e: S.Unary) -> int:
        target = e.operand
        ty = target.type.unqualified()
        ir = C.to_ir(ty)
        old = self._load_lvalue(target)
        if e.postfix and _in_register(target):
            # A REGISTER LOCAL READS AS ITS OWN REGISTER, so the value this
            # returns has to be a SNAPSHOT: the store below assigns that same
            # register, and without the copy `o[j++] = c` writes at the
            # incremented index -- leaving `o[0]` untouched and running one
            # past the end. That is exactly what it did, and the symptom was
            # a decimal number printed with its first digit replaced by a NUL.
            snapshot = self.b.reg(ir)
            self.b.copy(snapshot, old)
            old = snapshot
        if ty.is_pointer:
            if isinstance(e.scale, int):
                step = e.scale if e.op == "++" else -e.scale
                delta = self.b.const(IR.I64, step)
            else:
                delta = self._ir_convert(self._value(e.scale), e.scale.type,
                                         IR.I64)
                if e.op == "--":
                    neg = self.b.reg(IR.I64)
                    self.b.emit(Instruction(Op.NEG, IR.I64, dst=neg,
                                            args=[delta]))
                    delta = neg
            new = self.b.offset(old, delta)
        else:
            one = self.b.const(ir, 1.0 if ir.is_float else 1)
            new = self.b.reg(ir)
            self.b.emit(Instruction(Op.ADD if e.op == "++" else Op.SUB, ir,
                                    dst=new, args=[old, one]))
            new = self._narrow_bitint(new, ty)
        self._store_into(target, new)
        return old if e.postfix else new

    def _store_into(self, target: S.Expr, value: int) -> None:
        if isinstance(target, S.Ident) and target.sym is not None and \
                target.sym.in_register:
            self.b.copy(target.sym.slot, value)
            return
        if isinstance(target, S.MemberAccess) and target.path and \
                target.path[-1].is_bitfield:
            member = target.path[-1]
            self._store_bitfield(self._address(target), member.type,
                                 member.bits, member.bit_offset, value)
            return
        self.b.store(C.to_ir(target.type), value, self._address(target))

    def _binary(self, e: S.Binary) -> int:
        op = e.op
        if e.type.is_ldouble:
            return self._ld_binary(e)
        if e.left.type.is_ldouble and op in _CMP:
            return self._ld_compare(e)
        if e.type.is_complex or e.left.type.is_complex \
                or e.right.type.is_complex:
            return self._cx_binary(e)
        if op == "-p":
            left = self._value(e.left)
            right = self._value(e.right)
            a = self._bitcast(left, IR.I64) if self.fn.register_type(left).is_ptr \
                else left
            b = self._bitcast(right, IR.I64) if self.fn.register_type(right).is_ptr \
                else right
            diff = self.b.sub(IR.I64, a, b)
            if isinstance(e.scale, int):
                if e.scale != 1:
                    diff = self.b.div(IR.I64, diff,
                                      self.b.const(IR.I64, e.scale))
            else:
                width = self._ir_convert(self._value(e.scale), e.scale.type,
                                         IR.I64)
                diff = self.b.div(IR.I64, diff, width)
            return diff
        left = self._value(e.left)
        if e.left.type.is_pointer and op in ("+", "-"):
            index = self._value(e.right)
            if op == "-":
                neg = self.b.reg(IR.I64)
                index = self._ir_convert(index, e.right.type, IR.I64)
                self.b.emit(Instruction(Op.NEG, IR.I64, dst=neg, args=[index]))
                index = neg
                return self._scaled(left, index, C.LONG, e.scale)
            return self._scaled(left, index, e.right.type, e.scale)
        right = self._value(e.right)
        if op in ("==", "!=", "<", ">", "<=", ">="):
            operand_ty = e.left.type
            ir = C.to_ir(operand_ty)
            if ir.is_ptr:
                left = self._bitcast(left, IR.U64)
                right = self._bitcast(right, IR.U64)
                ir = IR.U64
            got = self.b.cmp(_CMP[op], ir, left, right)
            return self._int_convert(got, IR.I1, C.to_ir(e.type))
        ir = C.to_ir(e.type)
        if op in ("<<", ">>"):
            # The IR requires both operands of a shift to have the same type;
            # C promotes them separately, so the count is converted here.
            right = self._ir_convert(right, e.right.type, ir)
        d = self.b.reg(ir)
        self.b.emit(Instruction(_ARITH[op], ir, dst=d, args=[left, right]))
        return self._narrow_bitint(d, e.type)

    def _logical(self, e: S.Logical) -> int:
        result = self.b.reg(IR.I1)
        other = self.b.new_block("sc")
        after = self.b.new_block("scend")
        left = self._truth_of(e.left)
        self.b.copy(result, left)
        if e.op == "&&":
            self.b.branch(left, other, after)
        else:
            self.b.branch(left, after, other)
        self._open(other)
        right = self._truth_of(e.right)
        self.b.copy(result, right)
        self.b.jump(after)
        self._open(after)
        return self._int_convert(result, IR.I1, C.to_ir(e.type))

    def _branch(self, cond: S.Expr | None, then: Block, other: Block) -> None:
        """Emit a branch on `cond` without materialising a 0/1 first."""
        if cond is None:
            self.b.jump(then)
            return
        if isinstance(cond, S.Logical):
            middle = self.b.new_block("sc")
            if cond.op == "&&":
                self._branch(cond.left, middle, other)
            else:
                self._branch(cond.left, then, middle)
            self._open(middle)
            self._branch(cond.right, then, other)
            return
        if isinstance(cond, S.Unary) and cond.op == "!":
            self._branch(cond.operand, other, then)
            return
        self.b.branch(self._truth_of(cond), then, other)
        self._open(self.b.new_block("dead"))

    def _conditional(self, e: S.Conditional, *, address: bool = False) -> int:
        ir = IR.PTR if address or e.type.in_memory else C.to_ir(e.type)
        result = None if e.type.is_void and not address else self.b.reg(ir)
        then = self.b.new_block("cond")
        other = self.b.new_block("condelse")
        after = self.b.new_block("condend")
        self._branch(e.cond, then, other)
        self._open(then)
        got = self._address(e.then) if address else self._value(e.then)
        if result is not None:
            self.b.copy(result, got)
        if self.b.current.terminator is None:
            self.b.jump(after)
        self._open(other)
        got = self._address(e.otherwise) if address else self._value(e.otherwise)
        if result is not None:
            self.b.copy(result, got)
        if self.b.current.terminator is None:
            self.b.jump(after)
        self._open(after)
        return result if result is not None else self.b.const(IR.I64, 0)

    def _assign(self, e: S.Assign) -> int:
        target = e.target
        ty = target.type.unqualified()
        if e.op == "=":
            if ty.in_memory:
                dst = self._address(target)
                src = self._value(e.value)
                self._copy(dst, src, ty.size, ty.align)
                return dst
            value = self._rvalue(e.value, ty)
            self._store_into(target, value)
            return value
        # A COMPOUND ASSIGNMENT EVALUATES ITS TARGET ONCE. `*p++ += 1` must
        # advance `p` once, so the address is computed here and both the read
        # and the write go through it.
        binop = e.op[:-1]
        if ty.is_pointer:
            addr = None if _in_register(target) else self._address(target)
            old = self._read_through(target, addr)
            index = self._value(e.value)
            if binop == "-":
                index = self._ir_convert(index, e.value.type, IR.I64)
                neg = self.b.reg(IR.I64)
                self.b.emit(Instruction(Op.NEG, IR.I64, dst=neg, args=[index]))
                new = self._scaled(old, neg, C.LONG, e.scale)
            else:
                new = self._scaled(old, index, e.value.type, e.scale)
            self._write_through(target, addr, new)
            return new
        compute = e.compute or ty
        if compute.is_ldouble or ty.is_ldouble:
            # `x *= y` ON A LONG DOUBLE. The target always has an address --
            # `in_memory` says so -- so there is no register case here.
            wide = self._ld_binary(S.Binary(e.span, compute, False, binop,
                                            target, e.value))
            dst = self._address(target)
            if ty.is_ldouble:
                self._copy(dst, wide, 16, 16)
                return dst
            self.b.store(C.to_ir(ty), self._ld_to(wide, ty), dst)
            return self._ld_to(wide, ty)
        if compute.is_complex or ty.is_complex:
            # `z *= w` COMPUTES IN THE COMMON COMPLEX TYPE and converts back,
            # exactly as `c += 1` computes in `int`. The target of a complex
            # assignment always has an address -- `in_memory` says so -- so
            # there is no register case to keep in step.
            got = self._cx_arith(e.op[:-1], compute,
                                 self._cx_pair(target, compute),
                                 self._cx_pair(e.value, compute))
            dst = self._address(target)
            self._cx_assign(dst, ty, got, compute)
            return dst
        cir = C.to_ir(compute)
        addr = None if _in_register(target) else self._address(target)
        old = self._read_through(target, addr)
        old = self._ir_convert(old, ty, cir)
        right = self._value(e.value)
        if binop in ("<<", ">>"):
            right = self._ir_convert(right, e.value.type, cir)
        result = self.b.reg(cir)
        self.b.emit(Instruction(_ARITH[binop], cir, dst=result,
                                args=[old, right]))
        result = self._narrow_bitint(result, compute)
        narrowed = self._ir_convert(result, compute, C.to_ir(ty), target=ty)
        self._write_through(target, addr, narrowed)
        return narrowed

    def _read_through(self, target: S.Expr, addr: int | None) -> int:
        if addr is None:
            return target.sym.slot
        member = _bitfield_of(target)
        if member is not None:
            return self._load_bitfield(addr, member.type, member.bits,
                                       member.bit_offset)
        return self.b.load(C.to_ir(target.type), addr)

    def _write_through(self, target: S.Expr, addr: int | None,
                       value: int) -> None:
        if addr is None:
            self.b.copy(target.sym.slot, value)
            return
        member = _bitfield_of(target)
        if member is not None:
            self._store_bitfield(addr, member.type, member.bits,
                                 member.bit_offset, value)
            return
        self.b.store(C.to_ir(target.type), value, addr)

    # ── complex ─────────────────────────────────────────────────────────────
    #
    # TWO FLOATS SIDE BY SIDE, real part first, and nothing else: that is what
    # C says a complex value is (6.2.5p13, "an array of two elements"), what
    # every ABI does, and what `<complex.h>`'s `creal` reads. The IR has no
    # complex type and does not need one -- a value in memory with an address
    # is exactly what an aggregate already is here, so `in_memory` covers the
    # passing, returning, copying and storing, and what is left is the
    # arithmetic.
    #
    # THE TEMPORARIES COME FROM THE ENTRY BLOCK, pooled and reused, because
    # `Op.ALLOCA` inside a loop allocates once per turn -- and a complex
    # expression makes one per operator. A statement's temporaries are dead
    # at its end (nothing can hold the address of one past that), so the pool
    # resets there and a function ends up with as many as its deepest
    # expression needed rather than as many as it evaluated.

    def _cx_temp(self, ty: CType) -> int:
        want = ty.size
        index = self._cx_used
        self._cx_used += 1
        while len(self._cx_pool) <= index:
            self._cx_pool.append({})
        slot = self._cx_pool[index]
        if want not in slot:
            slot[want] = self._entry_alloca(want)
        return slot[want]

    #: WHAT EVERY BACKEND'S `alloca` GUARANTEES. The reference interpreter
    #: aligns one to 8 and the C backend's `alloca` to 16, so 8 is the number
    #: a program may rely on and anything stricter has to be arranged here.
    ALLOCA_ALIGN = 8

    def _aligned_slot(self, size: int, align: int) -> int:
        """An `alloca` whose address is a multiple of `align`.

        `Op.ALLOCA` TAKES A SIZE AND NOT AN ALIGNMENT, and giving it one
        would be a change to the IR that every backend has to implement for
        a case most programs never reach. Over-allocating and rounding the
        address up is the same answer, arrived at in the frontend: it costs
        `align - 1` bytes of a frame and four instructions, and only for an
        object that asked for more than the floor above.

        `_Alignas(32) char buf[64]` IS THE WHOLE REASON. Without this the
        address is whatever the backend's `alloca` happened to give, which
        is 8 in the interpreter -- and a program that checks, or a machine
        instruction that requires it, is not wrong to have believed the
        declaration.
        """
        size = max(1, size)
        if align <= self.ALLOCA_ALIGN:
            return self.b.alloca(size)
        raw = self.b.alloca(size + align - 1)
        addr = self._bitcast(raw, IR.U64)
        addr = self.b.add(IR.U64, addr, self.b.const(IR.U64, align - 1))
        mask = (~(align - 1)) & 0xFFFFFFFFFFFFFFFF
        d = self.b.reg(IR.U64)
        self.b.emit(Instruction(Op.AND, IR.U64, dst=d,
                                args=[addr, self.b.const(IR.U64, mask)]))
        return self._bitcast(d, IR.PTR)

    def _entry_alloca(self, size: int) -> int:
        """An ALLOCA in the entry block, wherever we are now.

        The entry block has no predecessors (the verifier's rule 9), so a
        register defined there is written on every path -- which is what
        makes a temporary allocated once usable from inside a loop.
        """
        d = self.b.reg(IR.PTR)
        entry = self.fn.blocks[0]
        at = len(entry.instructions) - (1 if entry.terminator is not None
                                        else 0)
        entry.instructions.insert(
            at, Instruction(Op.ALLOCA, IR.PTR, dst=d, imm=max(1, size),
                            span=self.b.span))
        return d

    def _cx_parts(self, addr: int, ty: CType) -> tuple[int, int]:
        """The two halves of the complex value at `addr`.

        AN ELEMENT THAT LIVES IN MEMORY -- `long double` -- IS ITS ADDRESS,
        which is the same convention the whole file uses for a value with no
        IR type. So the halves of a `long double _Complex` come back as two
        pointers and everything below reads them the same way.
        """
        if ty.of.in_memory:
            return addr, self._offset(addr, ty.of.size)
        elem = C.to_ir(ty.of)
        return (self.b.load(elem, addr),
                self.b.load(elem, self._offset(addr, ty.of.size)))

    def _cx_make(self, ty: CType, re: int, im: int) -> int:
        addr = self._cx_temp(ty)
        if ty.of.in_memory:
            self._copy(addr, re, ty.of.size, ty.of.align)
            self._copy(self._offset(addr, ty.of.size), im, ty.of.size,
                       ty.of.align)
            return addr
        elem = C.to_ir(ty.of)
        self.b.store(elem, re, addr)
        self.b.store(elem, im, self._offset(addr, ty.of.size))
        return addr

    def _cx_pair(self, e: S.Expr, ty: CType) -> tuple[int, int]:
        """`e` as (real, imaginary) in `ty`'s element type, whatever `e` is."""
        if ty.of.in_memory:
            if e.type.is_complex:
                re, im = self._cx_parts(self._value(e), e.type)
                return (self._ld_from(re, e.type.of),
                        self._ld_from(im, e.type.of))
            zero = self._ld_temp()
            self._ld_call("__c_ldfromi", [self.b.const(IR.I64, 0), zero])
            return self._ld_from(self._value(e), e.type), zero
        elem = C.to_ir(ty.of)
        if e.type.is_complex:
            re, im = self._cx_parts(self._value(e), e.type)
            if e.type.of.in_memory:
                # NARROWING FROM A LONG DOUBLE COMPLEX: each half is an
                # address and `_ld_to` is what reads one.
                return self._ld_to(re, ty.of), self._ld_to(im, ty.of)
            return (self._ir_convert(re, e.type.of, elem),
                    self._ir_convert(im, e.type.of, elem))
        if e.type.is_ldouble:
            return self._ld_to(self._value(e), ty.of), self.b.const(elem, 0.0)
        got = self._ir_convert(self._value(e), e.type, elem)
        return got, self.b.const(elem, 0.0)

    def _cx_part(self, e: S.Expr, imaginary: bool) -> int:
        """`__imag__ z` and `__real__ z`, which real headers use.

        A REAL OPERAND HAS AN IMAGINARY PART OF ZERO, which is gcc's rule and
        what makes `__real__ x` work on an ordinary double rather than being
        an error a macro has to avoid.
        """
        if not e.type.is_complex:
            got = self._value(e)
            if not imaginary:
                return got
            if e.type.is_ldouble:
                zero = self._ld_temp()
                self._ld_call("__c_ldfromi", [self.b.const(IR.I64, 0), zero])
                return zero
            return self.b.const(C.to_ir(e.type), 0.0)
        re, im = self._cx_parts(self._value(e), e.type)
        return im if imaginary else re

    def _cx_convert(self, src: S.Expr, want: CType) -> int:
        """Between complex and everything else, in either direction."""
        if want.is_complex:
            re, im = self._cx_pair(src, want)
            return self._cx_make(want, re, im)
        if src.type.of.in_memory:
            # A COMPLEX LONG DOUBLE NARROWING: the real half is an address
            # and `_ld_to` is what reads one.
            re, im = self._cx_parts(self._value(src), src.type)
            if want.is_bool:
                return self._int_convert(
                    self._cx_nonzero(re, im, src.type), IR.I1, C.to_ir(want))
            return self._ld_to(re, want)
        # COMPLEX TO REAL DISCARDS THE IMAGINARY PART, 6.3.1.7p2 -- except
        # for `_Bool`, where the question is whether the value is zero and
        # BOTH parts answer it.
        re, im = self._cx_parts(self._value(src), src.type)
        if want.is_bool:
            return self._int_convert(self._cx_nonzero(re, im, src.type),
                                     IR.I1, C.to_ir(want))
        return self._ir_convert(re, src.type.of, C.to_ir(want), target=want)

    def _cx_nonzero(self, re: int, im: int, ty: CType) -> int:
        if ty.of.in_memory:
            a = self._ld_nonzero(re)
            b = self._ld_nonzero(im)
            d = self.b.reg(IR.I1)
            self.b.emit(Instruction(Op.OR, IR.I1, dst=d, args=[a, b]))
            return d
        elem = C.to_ir(ty.of)
        zero = self.b.const(elem, 0.0)
        a = self.b.cmp(Op.NE, elem, re, zero)
        b = self.b.cmp(Op.NE, elem, im, zero)
        d = self.b.reg(IR.I1)
        self.b.emit(Instruction(Op.OR, IR.I1, dst=d, args=[a, b]))
        return d

    def _truth_of(self, e: S.Expr) -> int:
        """`e != 0` as an i1. The one place that knows a complex has two."""
        if e.type.is_ldouble:
            return self._ld_nonzero(self._value(e))
        if e.type.is_complex:
            re, im = self._cx_parts(self._value(e), e.type)
            return self._cx_nonzero(re, im, e.type)
        return self._truthy(self._value(e), C.to_ir(e.type))

    def _cx_unary(self, e: S.Unary) -> int:
        op = e.op
        if op == "+":
            return self._value(e.operand)
        if op == "-":
            ty = e.type
            re, im = self._cx_pair(e.operand, ty)
            if ty.of.in_memory:
                out = []
                for half in (re, im):
                    d = self._ld_temp()
                    self._ld_call("__c_ldneg", [half, d])
                    out.append(d)
                return self._cx_make(ty, out[0], out[1])
            elem = C.to_ir(ty.of)
            out = []
            for half in (re, im):
                d = self.b.reg(elem)
                self.b.emit(Instruction(Op.NEG, elem, dst=d, args=[half]))
                out.append(d)
            return self._cx_make(ty, out[0], out[1])
        if op == "~":
            # gcc's CONJUGATE, which `<complex.h>`'s `conj` is written with.
            ty = e.type
            re, im = self._cx_pair(e.operand, ty)
            if ty.of.in_memory:
                d = self._ld_temp()
                self._ld_call("__c_ldneg", [im, d])
                return self._cx_make(ty, re, d)
            elem = C.to_ir(ty.of)
            d = self.b.reg(elem)
            self.b.emit(Instruction(Op.NEG, elem, dst=d, args=[im]))
            return self._cx_make(ty, re, d)
        raise Unsupported(f"complex unary {op}")

    def _cx_binary(self, e: S.Binary) -> int:
        op = e.op
        if op in ("==", "!="):
            ty = C.usual_arithmetic(e.left.type, e.right.type)
            a = self._cx_pair(e.left, ty)
            b = self._cx_pair(e.right, ty)
            want = Op.EQ if op == "==" else Op.NE
            if ty.of.in_memory:
                halves = []
                for x, y in ((a[0], b[0]), (a[1], b[1])):
                    got = self._ld_call("__c_ldcmp", [x, y])
                    halves.append(self.b.cmp(want, IR.I32, got,
                                             self.b.const(IR.I32, 0)))
                re, im = halves
            else:
                elem = C.to_ir(ty.of)
                re = self.b.cmp(want, elem, a[0], b[0])
                im = self.b.cmp(want, elem, a[1], b[1])
            d = self.b.reg(IR.I1)
            # BOTH HALVES FOR `==`, EITHER FOR `!=`, which is the same test
            # written the two ways round.
            self.b.emit(Instruction(Op.AND if op == "==" else Op.OR, IR.I1,
                                    dst=d, args=[re, im]))
            return self._int_convert(d, IR.I1, C.to_ir(e.type))
        got = self._cx_arith(op, e.type, self._cx_pair(e.left, e.type),
                             self._cx_pair(e.right, e.type))
        return self._cx_make(e.type, got[0], got[1])

    def _cx_arith(self, op: str, ty: CType, a: tuple[int, int],
                  b: tuple[int, int]) -> tuple[int, int]:
        """(re, im) of `a op b`, both already in `ty`'s element type.

        ADDITION AND SUBTRACTION ARE INLINE and multiplication and division
        are CALLS, and the split is not about how long the formula is.
        `(a+bi)(c+di)` is four multiplies and two adds -- but Annex G says
        what an infinity times a zero has to produce, and getting that right
        takes the recovery step libgcc's `__muldc3` has. Division needs
        scaling to avoid an overflow that the mathematics does not have.
        Both are in `support.py`'s `complex` unit, written in C, which is
        also what makes them agree with gcc's answers rather than nearly.
        """
        if ty.of.in_memory:
            # AN ELEMENT WITH NO IR TYPE: every half is an address and every
            # operation on one is a call, which is what `long double` is.
            if op in ("+", "-"):
                out = []
                for x, y in ((a[0], b[0]), (a[1], b[1])):
                    d = self._ld_temp()
                    self._ld_call("__c_ldadd" if op == "+" else "__c_ldsub",
                                  [x, y, d])
                    out.append(d)
                return out[0], out[1]
            if op not in ("*", "/"):
                raise Unsupported(f"complex operator {op}")
            self.needs_complex = True
            name = "__c_cmull" if op == "*" else "__c_cdivl"
            self._ensure_extern(name, IR.VOID, [IR.PTR] * 5)
            out = self._cx_temp(ty)
            self.b.call(IR.VOID, name, [a[0], a[1], b[0], b[1], out])
            return self._cx_parts(out, ty)
        elem = C.to_ir(ty.of)
        if op in ("+", "-"):
            out = []
            for x, y in ((a[0], b[0]), (a[1], b[1])):
                d = self.b.reg(elem)
                self.b.emit(Instruction(Op.ADD if op == "+" else Op.SUB, elem,
                                        dst=d, args=[x, y]))
                out.append(d)
            return out[0], out[1]
        if op not in ("*", "/"):
            raise Unsupported(f"complex operator {op}")
        self.needs_complex = True
        wide = elem is IR.F64
        name = ("__c_cmul" if op == "*" else "__c_cdiv") + ("" if wide else "f")
        self._ensure_extern(name, IR.VOID,
                            [elem, elem, elem, elem, IR.PTR])
        out = self._cx_temp(ty)
        self.b.call(IR.VOID, name, [a[0], a[1], b[0], b[1], out])
        return self._cx_parts(out, ty)

    def _cx_assign(self, dst: int, ty: CType, got: tuple[int, int],
                   compute: CType) -> None:
        """Store a computed pair into a complex object, converting if the
        object is narrower than the type the arithmetic happened in."""
        if ty.of.in_memory:
            self._copy(dst, got[0], ty.of.size, ty.of.align)
            self._copy(self._offset(dst, ty.of.size), got[1], ty.of.size,
                       ty.of.align)
            return
        elem = C.to_ir(ty.of)
        re = self._ir_convert(got[0], compute.of, elem)
        im = self._ir_convert(got[1], compute.of, elem)
        self.b.store(elem, re, dst)
        self.b.store(elem, im, self._offset(dst, ty.of.size))

    # ── long double ─────────────────────────────────────────────────────────
    #
    # SIXTEEN BYTES WITH AN ADDRESS, like an aggregate, because 80-bit
    # extended is wider than anything the IR has -- and a third floating
    # width would have to be implemented by every backend, including the
    # ones whose machine has no such thing. So the value lives in memory and
    # every operation on it is a call into `support.py`'s `ldouble` unit,
    # which is the format written out in C. `ctype.CType.is_ldouble` says
    # the same thing from the other end.

    #: name -> (argument IR types, result). The declarations the calls need,
    #: kept here rather than spelled at each site so that a signature cannot
    #: be right in one place and wrong in another.
    _LD_CALLS = {
        "__c_ldadd": ((IR.PTR, IR.PTR, IR.PTR), IR.VOID),
        "__c_ldsub": ((IR.PTR, IR.PTR, IR.PTR), IR.VOID),
        "__c_ldmul": ((IR.PTR, IR.PTR, IR.PTR), IR.VOID),
        "__c_lddiv": ((IR.PTR, IR.PTR, IR.PTR), IR.VOID),
        "__c_ldneg": ((IR.PTR, IR.PTR), IR.VOID),
        "__c_ldcmp": ((IR.PTR, IR.PTR), IR.I32),
        "__c_ldnz": ((IR.PTR,), IR.I32),
        "__c_ldtod": ((IR.PTR,), IR.F64),
        "__c_ldfromd": ((IR.F64, IR.PTR), IR.VOID),
        "__c_ldtoi": ((IR.PTR,), IR.I64),
        "__c_ldtou": ((IR.PTR,), IR.U64),
        "__c_ldfromi": ((IR.I64, IR.PTR), IR.VOID),
        "__c_ldfromu": ((IR.U64, IR.PTR), IR.VOID),
    }

    def _ld_call(self, name: str, args: list[int]) -> int | None:
        params, ret = self._LD_CALLS[name]
        self.needs_ldouble = True
        self._ensure_extern(name, ret, list(params))
        return self.b.call(ret, name, args)

    def _ld_temp(self) -> int:
        return self._cx_temp(C.LDOUBLE)

    def _ld_binary(self, e: S.Binary) -> int:
        """`+ - * /`, with both operands already this type."""
        name = {"+": "__c_ldadd", "-": "__c_ldsub",
                "*": "__c_ldmul", "/": "__c_lddiv"}.get(e.op)
        if name is None:
            raise Unsupported(f"long double operator {e.op}")
        left = self._value(e.left)
        right = self._value(e.right)
        out = self._ld_temp()
        self._ld_call(name, [left, right, out])
        return out

    def _ld_compare(self, e: S.Binary) -> int:
        """The six orderings, out of one three-way answer.

        `__c_ldcmp` answers -1, 0 or 1, and 2 for UNORDERED -- which is not
        a value an ordering can take, so one call decides every operator and
        a NaN comes out false for all of them but `!=`.
        """
        left = self._value(e.left)
        right = self._value(e.right)
        got = self._ld_call("__c_ldcmp", [left, right])
        zero = self.b.const(IR.I32, 0)
        one = self.b.const(IR.I32, 1)
        if e.op == "==":
            r = self.b.cmp(Op.EQ, IR.I32, got, zero)
        elif e.op == "!=":
            r = self.b.cmp(Op.NE, IR.I32, got, zero)
        elif e.op == "<":
            r = self.b.cmp(Op.LT, IR.I32, got, zero)
        elif e.op == "<=":
            r = self.b.cmp(Op.LE, IR.I32, got, zero)
        elif e.op == ">":
            r = self.b.cmp(Op.EQ, IR.I32, got, one)
        else:                                       # ">="
            # UNSIGNED, WHICH IS THE WHOLE TRICK: 0 and 1 are both at most
            # one, -1 is enormous and 2 is two, so one comparison rules out
            # both "less" and "unordered".
            wide = self._bitcast(got, IR.U32)
            r = self.b.cmp(Op.LE, IR.U32, wide, self.b.const(IR.U32, 1))
        return self._int_convert(r, IR.I1, C.to_ir(e.type))

    def _ld_unary(self, e: S.Unary) -> int:
        if e.op == "+":
            return self._value(e.operand)
        if e.op == "-":
            out = self._ld_temp()
            self._ld_call("__c_ldneg", [self._value(e.operand), out])
            return out
        raise Unsupported(f"long double unary {e.op}")

    def _ld_nonzero(self, addr: int) -> int:
        """`x != 0` as an i1. A NaN is not zero, which is what C wants."""
        got = self._ld_call("__c_ldnz", [addr])
        return self.b.cmp(Op.NE, IR.I32, got, self.b.const(IR.I32, 0))

    def _ld_from(self, reg: int, have: CType) -> int:
        """A scalar value of `have`, widened into a fresh long double."""
        out = self._ld_temp()
        if have.is_ldouble:
            self._copy(out, reg, 16, 16)
            return out
        if have.is_float:
            self._ld_call("__c_ldfromd",
                          [self._ir_convert(reg, have, IR.F64), out])
            return out
        if have.is_pointer:
            reg = self._bitcast(reg, IR.U64)
            self._ld_call("__c_ldfromu", [reg, out])
            return out
        # AN INTEGER, SIGNED OR NOT, and which one matters: the same 64 bits
        # are two different numbers, and the conversion has to know.
        want = IR.U64 if not have.signed else IR.I64
        self._ld_call("__c_ldfromu" if want is IR.U64 else "__c_ldfromi",
                      [self._ir_convert(reg, have, want), out])
        return out

    def _ld_to(self, addr: int, want: CType) -> int:
        """A long double as `want`, which is usually narrower."""
        if want.is_ldouble:
            return addr                 # already this type: nothing to do
        if want.is_bool:
            return self._int_convert(self._ld_nonzero(addr), IR.I1,
                                     C.to_ir(want))
        if want.is_float:
            return self._ir_convert(self._ld_call("__c_ldtod", [addr]),
                                    C.DOUBLE, C.to_ir(want))
        if want.is_complex:
            re = self._ld_to(addr, want.of)
            if want.of.in_memory:
                zero = self._ld_temp()
                self._ld_call("__c_ldfromi", [self.b.const(IR.I64, 0), zero])
                return self._cx_make(want, re, zero)
            return self._cx_make(want, re, self.b.const(C.to_ir(want.of), 0.0))
        # UNSIGNED GOES THROUGH THE UNSIGNED CONVERSION, because a value
        # above 2^63 is not the same number read as signed.
        name = "__c_ldtoi" if want.signed else "__c_ldtou"
        got = self._ld_call(name, [addr])
        return self._ir_convert(got, C.LONG if want.signed else C.ULONG,
                                C.to_ir(want))

    def _ld_convert(self, src: S.Expr, want: CType) -> int:
        if want.is_ldouble:
            if src.type.is_complex:
                # A COMPLEX NARROWS TO ITS REAL PART, and then widens.
                re, _ = self._cx_parts(self._value(src), src.type)
                return self._ld_from(re, src.type.of)
            return self._ld_from(self._value(src), src.type)
        return self._ld_to(self._value(src), want)

    def _ld_const(self, value) -> int:
        """A literal, as the sixteen bytes it is.

        THROUGH A GLOBAL AND NOT THROUGH ARITHMETIC, because the value has
        64 significant bits and the IR's widest constant has 53. The bytes
        are what `ldouble.py` encodes, which is what the format says.
        """
        from .ldouble import encode
        data = encode(value)
        name = f"{self.parser.sema.prefix}ld.{len(self.module.globals)}"
        have = next((g for g in self.module.globals
                     if g.data == data and g.readonly and g.size == 16), None)
        if have is not None:
            return self._global_addr(have.name)
        self._add_global(Global(name, 16, data, readonly=True, align=16))
        return self._global_addr(name)

    # ── bit-fields ──────────────────────────────────────────────────────────
    def _load_bitfield(self, addr: int, ty: CType, bits: int,
                       bit_offset: int) -> int:
        ir = C.to_ir(ty)
        unit = self.b.load(ir, addr)
        width = ir.bits
        left = width - bits - bit_offset
        if left:
            shifted = self.b.reg(ir)
            self.b.emit(Instruction(Op.SHL, ir, dst=shifted,
                                    args=[unit, self.b.const(ir, left)]))
            unit = shifted
        # SHR IS ARITHMETIC ON A SIGNED TYPE AND LOGICAL ON AN UNSIGNED ONE,
        # which is exactly the difference between a signed and an unsigned
        # bit-field -- so the shift pair needs no mask and no sign-extension
        # step. That is the whole reason the value is shifted up first.
        down = width - bits
        if down:
            shifted = self.b.reg(ir)
            self.b.emit(Instruction(Op.SHR, ir, dst=shifted,
                                    args=[unit, self.b.const(ir, down)]))
            unit = shifted
        return unit

    def _store_bitfield(self, addr: int, ty: CType, bits: int, bit_offset: int,
                        value: int) -> None:
        ir = C.to_ir(ty)
        value = self._ir_convert(value, self.fn.register_type(value), ir)
        mask = ((1 << bits) - 1) << bit_offset
        full = (1 << ir.bits) - 1
        old = self.b.load(ir, addr)
        cleared = self.b.reg(ir)
        self.b.emit(Instruction(Op.AND, ir, dst=cleared,
                                args=[old, self.b.const(ir, _as(~mask & full, ir))]))
        shifted = value
        if bit_offset:
            shifted = self.b.reg(ir)
            self.b.emit(Instruction(Op.SHL, ir, dst=shifted,
                                    args=[value, self.b.const(ir, bit_offset)]))
        kept = self.b.reg(ir)
        self.b.emit(Instruction(Op.AND, ir, dst=kept,
                                args=[shifted, self.b.const(ir, _as(mask, ir))]))
        merged = self.b.reg(ir)
        self.b.emit(Instruction(Op.OR, ir, dst=merged, args=[cleared, kept]))
        self.b.store(ir, merged, addr)

    # ── calls ───────────────────────────────────────────────────────────────
    def _call(self, e: S.Call) -> int:
        sig = e.func.type.of if e.func.type.is_pointer else e.func.type
        ret = sig.ret
        sret = None
        args: list[int] = []
        if ret.is_record or ret.is_complex or ret.is_ldouble:
            sret = self.b.alloca(max(1, ret.size))
            args.append(sret)
        for arg in e.args:
            args.append(self._argument(arg))
        if e.varargs or sig.variadic:
            args.append(self._va_pack(e.varargs))
        target = _direct_target(e.func)
        ret_ir = IR.VOID if (ret.is_void or sret is not None) else C.to_ir(ret)
        if target is not None:
            got = self.b.call(ret_ir, target, args)
        else:
            addr = self._value(e.func)
            d = None if ret_ir.is_void else self.b.reg(ret_ir)
            self.b.emit(Instruction(Op.CALL_PTR, ret_ir, dst=d,
                                    args=[addr] + args))
            got = d
        if sret is not None:
            return sret
        return got if got is not None else self.b.const(IR.I64, 0)

    def _argument(self, arg: S.Expr) -> int:
        if arg.type.is_record or arg.type.is_complex or arg.type.is_ldouble:
            # BY VALUE MEANS A FRESH COPY. The callee receives a pointer and
            # is entitled to write through it, so handing it the caller's own
            # object would make `f(s)` able to change `s`.
            slot = self.b.alloca(max(1, arg.type.size))
            self._copy(slot, self._value(arg), arg.type.size, arg.type.align)
            return slot
        return self._value(arg)

    def _va_pack(self, varargs: list[S.Expr]) -> int:
        """Build the argument area for one call. See the module docstring."""
        if not varargs:
            return self.b.alloca(VA_SLOT)
        area = self.b.alloca(VA_SLOT * len(varargs))
        for i, arg in enumerate(varargs):
            slot = self._offset(area, i * VA_SLOT)
            if (arg.type.is_record or arg.type.is_complex
                    or arg.type.is_ldouble):
                copy = self.b.alloca(max(1, arg.type.size))
                self._copy(copy, self._value(arg), arg.type.size,
                           arg.type.align)
                self.b.store(IR.PTR, copy, slot)
                continue
            value = self._value(arg)
            ir = C.to_ir(arg.type)
            if ir.is_float:
                self.b.store(IR.F64, self._ir_convert(value, arg.type, IR.F64),
                             slot)
            elif ir.is_ptr:
                self.b.store(IR.PTR, value, slot)
            else:
                want = IR.I64 if arg.type.signed else IR.U64
                self.b.store(want, self._ir_convert(value, arg.type, want), slot)
        return area

    def _va_arg(self, e: S.VaArg) -> int:
        ap = e.ap
        addr = self._address(_unconvert(ap))
        cursor = self.b.load(IR.PTR, addr)
        nxt = self.b.offset(cursor, self.b.const(IR.I64, VA_SLOT))
        self.b.store(IR.PTR, nxt, addr)
        ty = e.type
        if ty.is_record or ty.is_complex or ty.is_ldouble:
            return self.b.load(IR.PTR, cursor)
        ir = C.to_ir(ty)
        if ir.is_float:
            got = self.b.load(IR.F64, cursor)
            return self._ir_convert(got, C.DOUBLE, ir)
        if ir.is_ptr:
            return self.b.load(IR.PTR, cursor)
        want = IR.I64 if ty.signed else IR.U64
        got = self.b.load(want, cursor)
        return self._ir_convert(got, C.LONG if ty.signed else C.ULONG, ir,
                                target=ty)

    def _compound_literal(self, e: S.CompoundLiteral) -> int:
        if e.symbol is not None and e.thread_local:
            return self._tls_lookup(e.symbol, max(1, e.type.size))
        if e.symbol is not None:
            return self._global_addr(e.symbol)
        slot = self.b.alloca(max(1, e.type.size))
        self._emit_init_into(slot, e.type, e.init)
        return slot

    def _stmt_expr(self, e: S.StmtExpr) -> int:
        body = e.body
        result = None
        for i, item in enumerate(body.items):
            last = i == len(body.items) - 1
            if last and isinstance(item, S.ExprStmt) and item.expr is not None \
                    and not e.type.is_void:
                result = self._value(item.expr)
            elif isinstance(item, S.Decl):
                self._local_decl(item)
            else:
                self._stmt(item)
        if result is not None:
            return result
        return self.b.const(IR.I64, 0)

    # ── builtins ────────────────────────────────────────────────────────────
    def _builtin(self, e: S.BuiltinCall) -> int:
        from .lower_builtins import lower_builtin
        return lower_builtin(self, e)

    # ── memory helpers ──────────────────────────────────────────────────────
    def _zero(self, addr: int, size: int) -> None:
        self._fill(addr, size, None)

    def _copy(self, dst: int, src: int, size: int, align: int) -> None:
        if dst == src or size <= 0:
            return
        if size > _INLINE_COPY_LIMIT:
            self._copy_loop(dst, src, size)
            return
        for offset, ty in _chunks(size, align):
            value = self.b.load(ty, self._offset(src, offset))
            self.b.store(ty, value, self._offset(dst, offset))

    def _fill(self, addr: int, size: int, value: int | None) -> None:
        if size <= 0:
            return
        if size > _INLINE_COPY_LIMIT:
            self._fill_loop(addr, size)
            return
        for offset, ty in _chunks(size, 8):
            zero = value if value is not None else self.b.const(ty, 0)
            self.b.store(ty, zero, self._offset(addr, offset))

    def _byte_loop(self, count: int, body) -> int:
        """`for (i = 0; i < count; i++) body(i)`, as blocks. Returns `i`.

        The shape every one of the memory builtins needs, written once: a
        mutable cursor, a head that tests it, a body the caller fills, and a
        latch. Four of them written out separately is four places for the
        cursor to be incremented on the wrong side of the test.
        """
        cursor = self.b.reg(IR.I64)
        self.b.copy(cursor, self.b.const(IR.I64, 0))
        head = self.b.new_block("loop")
        inner = self.b.new_block("loopbody")
        after = self.b.new_block("loopend")
        self.b.jump(head)
        self._open(head)
        test = self.b.cmp(Op.LT, IR.I64, cursor, count)
        self.b.branch(test, inner, after)
        self._open(inner)
        body(cursor, after)
        if self.b.current.terminator is None:
            self.b.copy(cursor, self.b.add(IR.I64, cursor,
                                           self.b.const(IR.I64, 1)))
            self.b.jump(head)
        self._open(after)
        return cursor

    def _mem_inline(self, name: str, args: list[int]) -> int:
        """`memcpy` and friends with no library behind them.

        A BUILTIN MUST NOT NEED A LIBRARY. `__builtin_memcmp(a, b, n)` appears
        in programs that never include `<string.h>` -- gcc emits a call and
        lets the link fail -- and every one of these is a loop over memory,
        which is what the IR expresses directly. So when nothing in the module
        defines the function, the loop is emitted here instead of a call to
        something that is not there.
        """
        b = self.b
        if name == "strlen":
            base = args[0]
            n = b.reg(IR.I64)
            b.copy(n, b.const(IR.I64, 0))
            head = b.new_block("slen")
            body = b.new_block("slenbody")
            after = b.new_block("slenend")
            b.jump(head)
            self._open(head)
            byte = b.load(IR.U8, b.offset(base, n))
            more = b.cmp(Op.NE, IR.U8, byte, b.const(IR.U8, 0))
            b.branch(more, body, after)
            self._open(body)
            b.copy(n, b.add(IR.I64, n, b.const(IR.I64, 1)))
            b.jump(head)
            self._open(after)
            return n
        count = self._ir_convert(args[-1], self.fn.register_type(args[-1]),
                                 IR.I64)
        if name == "memset":
            value = self._ir_convert(args[1], self.fn.register_type(args[1]),
                                     IR.U8)
            self._byte_loop(count, lambda i, _end:
                            b.store(IR.U8, value, b.offset(args[0], i)))
            return args[0]
        if name == "memcmp":
            result = b.reg(IR.I32)
            b.copy(result, b.const(IR.I32, 0))

            def compare(i, end):
                x = b.load(IR.U8, b.offset(args[0], i))
                y = b.load(IR.U8, b.offset(args[1], i))
                same = b.cmp(Op.EQ, IR.U8, x, y)
                differs = b.new_block("cmpdiff")
                nxt = b.new_block("cmpnext")
                b.branch(same, nxt, differs)
                self._open(differs)
                below = b.cmp(Op.LT, IR.U8, x, y)
                b.copy(result, self._select(below, b.const(IR.I32, -1),
                                            b.const(IR.I32, 1), IR.I32))
                b.jump(end)
                self._open(nxt)

            self._byte_loop(count, compare)
            return result
        if name == "memmove":
            # BACKWARDS WHEN THE DESTINATION IS ABOVE THE SOURCE. This is the
            # whole difference from `memcpy`, and getting it wrong is
            # invisible until two objects overlap.
            dst_i = self._bitcast(args[0], IR.U64)
            src_i = self._bitcast(args[1], IR.U64)
            above = b.cmp(Op.GT, IR.U64, dst_i, src_i)
            back = b.new_block("mmback")
            fwd = b.new_block("mmfwd")
            done = b.new_block("mmend")
            b.branch(above, back, fwd)
            self._open(back)
            self._byte_loop(count, lambda i, _e: b.store(
                IR.U8,
                b.load(IR.U8, b.offset(args[1], b.sub(
                    IR.I64, b.sub(IR.I64, count, i),
                    b.const(IR.I64, 1)))),
                b.offset(args[0], b.sub(IR.I64, b.sub(IR.I64, count, i),
                                        b.const(IR.I64, 1)))))
            b.jump(done)
            self._open(fwd)
            self._byte_loop(count, lambda i, _e: b.store(
                IR.U8, b.load(IR.U8, b.offset(args[1], i)),
                b.offset(args[0], i)))
            b.jump(done)
            self._open(done)
            return args[0]
        self._byte_loop(count, lambda i, _e: b.store(
            IR.U8, b.load(IR.U8, b.offset(args[1], i)), b.offset(args[0], i)))
        return args[0]

    def _select(self, cond: int, a: int, c: int, ty: IR.Type) -> int:
        """`cond ? a : c` without a call. Two blocks and a shared register."""
        out = self.b.reg(ty)
        yes = self.b.new_block("sel")
        no = self.b.new_block("selelse")
        after = self.b.new_block("selend")
        self.b.branch(cond, yes, no)
        self._open(yes)
        self.b.copy(out, a)
        self.b.jump(after)
        self._open(no)
        self.b.copy(out, c)
        self.b.jump(after)
        self._open(after)
        return out

    def _copy_loop(self, dst: int, src: int, size: int) -> None:
        cursor = self.b.reg(IR.I64)
        self.b.copy(cursor, self.b.const(IR.I64, 0))
        head = self.b.new_block("copy")
        body = self.b.new_block("copybody")
        after = self.b.new_block("copyend")
        self.b.jump(head)
        self._open(head)
        limit = self.b.const(IR.I64, size)
        test = self.b.cmp(Op.LT, IR.I64, cursor, limit)
        self.b.branch(test, body, after)
        self._open(body)
        byte = self.b.load(IR.U8, self.b.offset(src, cursor))
        self.b.store(IR.U8, byte, self.b.offset(dst, cursor))
        self.b.copy(cursor, self.b.add(IR.I64, cursor, self.b.const(IR.I64, 1)))
        self.b.jump(head)
        self._open(after)

    def _fill_loop(self, addr: int, size: int) -> None:
        cursor = self.b.reg(IR.I64)
        self.b.copy(cursor, self.b.const(IR.I64, 0))
        head = self.b.new_block("fill")
        body = self.b.new_block("fillbody")
        after = self.b.new_block("fillend")
        self.b.jump(head)
        self._open(head)
        limit = self.b.const(IR.I64, size)
        test = self.b.cmp(Op.LT, IR.I64, cursor, limit)
        self.b.branch(test, body, after)
        self._open(body)
        self.b.store(IR.U8, self.b.const(IR.U8, 0), self.b.offset(addr, cursor))
        self.b.copy(cursor, self.b.add(IR.I64, cursor, self.b.const(IR.I64, 1)))
        self.b.jump(head)
        self._open(after)

    # ── the support code, written in C ──────────────────────────────────────
    def _splice_support(self) -> None:
        """Compile `support.c` into this module. See the module docstring."""
        from .support import compile_support
        units = ((("vla",) if self.needs_vla else ())
                 + (("args",) if self.needs_args else ())
                 + (("complex",) if self.needs_complex else ())
                 + (("ldouble",) if self.needs_ldouble else ())
                 + (("tls",) if self.needs_tls else ()))
        functions, globals_ = compile_support(self.sink, units)
        for fn in functions:
            existing = self.module.function(fn.name)
            if existing is not None:
                # The user's own definition wins, and so does their own
                # declaration of `plat_heap` -- a second one would be a
                # duplicate symbol rather than a missing one.
                if not existing.external or fn.external:
                    continue
                self.module.functions.remove(existing)
            self.module.functions.append(fn)
        for g in globals_:
            self._add_global(g)



def prune(module: Module) -> None:
    """Drop what nothing reaches.

    THE STANDARD LIBRARY IS A HEADER, so `#include <stdio.h>` brings in
    `printf` AND `fprintf` AND `snprintf` AND the whole formatter, as
    `static` definitions in this one translation unit. Every one of them
    would reach the backend and be emitted: the C backend writes a function
    per IR function, and a program that prints one line would carry a `qsort`
    it never calls.

    A linker drops those, and this frontend produces a module rather than an
    object file, so nothing downstream would. Reachability from the exported
    functions is exactly the question a linker asks, and the answer here
    costs one walk of the instruction stream.

    ONLY INTERNAL ONES GO. Anything with external linkage is part of the
    artifact's interface whether or not this unit calls it, and an external
    DECLARATION is kept only while something still calls it -- which is what
    stops `extern` lines for functions the pruning just removed the calls to.

    A FREE FUNCTION AND NOT A METHOD because it runs twice when a build has
    several translation units: once per unit, where it is most of what makes
    a unit's own copy of the library small, and once over the merged module,
    where it removes what the merge made unreachable -- a unit's string
    literals, say, whose only user was its copy of a library function that
    the merge kept one of.
    """
    by_name = {f.name: f for f in module.functions}
    roots = [f for f in module.functions
             if f.linkage is not IRLinkage.INTERNAL and not f.external]
    seen: set[str] = set()
    globals_used: set[str] = set()
    stack = [f.name for f in roots]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        fn = by_name.get(name)
        if fn is None:
            continue
        for _, ins in fn.instructions():
            if ins.op in (Op.CALL, Op.FUNC_ADDR) and ins.sym:
                if ins.sym not in seen:
                    stack.append(ins.sym)
            elif ins.op is Op.GLOBAL_ADDR and ins.sym:
                globals_used.add(ins.sym)
    module.functions = [
        f for f in module.functions
        if f.name in seen or (f.linkage is not IRLinkage.INTERNAL
                              and not f.external)]
    # A global an initialiser points AT is used even when no instruction
    # names it: `static char *p = msg;` becomes a store in the unit's
    # initialiser, which does name it -- but a string only ever reached
    # through another global's bytes would not be there. Nothing produces
    # that today; the assertion is the comment.
    module.globals = [
        g for g in module.globals
        if g.name in globals_used or g.linkage is IRLinkage.EXPORT]

#: The IR opcode each C operator becomes.
_ARITH = {
    "+": Op.ADD, "-": Op.SUB, "*": Op.MUL, "/": Op.DIV, "%": Op.REM,
    "&": Op.AND, "|": Op.OR, "^": Op.XOR, "<<": Op.SHL, ">>": Op.SHR,
}
_CMP = {"==": Op.EQ, "!=": Op.NE, "<": Op.LT, "<=": Op.LE, ">": Op.GT,
        ">=": Op.GE}


def _as(value: int, ty: IR.Type) -> int:
    """A mask as the IR type spells it: signed types want a signed constant."""
    bits = ty.bits
    v = value & ((1 << bits) - 1)
    if ty.is_signed and bits > 1 and v >= (1 << (bits - 1)):
        v -= 1 << bits
    return v


def _chunks(size: int, align: int):
    """(offset, type) pairs covering `size` bytes, widest first."""
    out = []
    offset = 0
    for width, ty in ((8, IR.U64), (4, IR.U32), (2, IR.U16), (1, IR.U8)):
        if width > align and width > 1 and offset % width:
            continue
        while size - offset >= width:
            if offset % width == 0 or width == 1:
                out.append((offset, ty))
                offset += width
            else:
                break
    while offset < size:
        out.append((offset, IR.U8))
        offset += 1
    return out


def _signature(ty: CType) -> tuple[IR.Type, list[IR.Type], bool]:
    """The IR signature of a C function type. See the module docstring."""
    sret = ty.ret.is_record or ty.ret.is_complex or ty.ret.is_ldouble
    ret = IR.VOID if (sret or ty.ret.is_void) else C.to_ir(ty.ret)
    params: list[IR.Type] = [IR.PTR] if sret else []
    for p in (ty.params or ()):
        params.append(_param_ir(p.type))
    if ty.variadic or ty.params is None:
        params.append(IR.PTR)
    return ret, params, sret


def _param_ir(ty: CType) -> IR.Type:
    return IR.PTR if ty.in_memory else C.to_ir(ty)


def _direct_target(func: S.Expr) -> str | None:
    """The symbol a call names directly, or None for a call through a value."""
    e = func
    while isinstance(e, (S.Conv, S.Cast)):
        e = e.operand
    if isinstance(e, S.Unary) and e.op == "&":
        e = e.operand
    if isinstance(e, S.Ident) and e.sym is not None and e.sym.type.is_function:
        name = e.sym.ir_name or e.sym.name
        return USER_MAIN if name == "main" else name
    return None


def _addressable(e: S.Expr) -> bool:
    return e.lvalue or isinstance(e, (S.StringLit, S.CompoundLiteral))


def _in_register(e: S.Expr) -> bool:
    return (isinstance(e, S.Ident) and e.sym is not None
            and getattr(e.sym, "in_register", False))


def _bitfield_of(e: S.Expr):
    if isinstance(e, S.MemberAccess) and e.path and e.path[-1].is_bitfield:
        return e.path[-1]
    return None


def _unconvert(e: S.Expr) -> S.Expr:
    while isinstance(e, (S.Conv, S.Cast)):
        e = e.operand
    return e


def _prefix_align(prefix: str) -> int:
    from .literals import PREFIXES
    return PREFIXES[prefix][0]
