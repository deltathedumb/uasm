"""A binary backend emits bytes; a language backend emits source. Nothing else.

THE RULE. A backend is one of two things and the difference decides whether
text may come out of it:

  * a LANGUAGE backend emits source in another language -- C, LLVM IR -- and
    something downstream compiles it. Text is the artifact.
  * a BINARY backend emits machine code or bytecode. The output is bytes a
    loader takes directly.

ASSEMBLY IS NEITHER, and that is the case this file exists to catch. A backend
emitting `.s` looks finished from the outside -- the file appears, `cc` links
it, the program runs -- while it has never encoded an instruction: the last
stage was handed to `as`. Nothing else in the suite can tell the difference,
because every test asks whether the program produced the right answer and the
answer is right either way.

So the check is on the ARTIFACT, not on the backend's own description of
itself. `KNOWN_TEXT_EMITTERS` was the list of backend-and-target pairs that
still emitted assembly. IT IS NOW EMPTY -- every binary backend encodes its
own instructions and writes its own object file for every target it serves --
and the assertion is that it stays empty.
"""
from __future__ import annotations

from tests import harness

from uasm import backend as backend_registry
from uasm import target as target_registry
from uasm.diagnostics import DiagnosticSink
from uasm.driver import Options, compile_source

backend_registry.load_builtin()
BACKENDS = sorted(backend_registry.available())

#: BACKENDS THAT CLAIM TO WORK. An unfinished one declares `ready = False` and
#: refuses to emit, so asking it for artifacts tests nothing about its output
#: -- it has none. What it owes instead is a clear refusal, which
#: `TestAnUnfinishedBackendRefuses` checks.
READY = [b for b in BACKENDS if backend_registry.get(b).ready]

SOURCE = """\
def add(a: int, b: int) -> int:
    return a + b


def main() -> int:
    return add(3, 4)
"""

#: BINARY BACKEND AND TARGET PAIRS THAT STILL EMIT ASSEMBLY TEXT. Each is a
#: combination with no instruction encoder or no object writer yet, so it stops
#: at `.s` and lets `as` finish. THIS LIST MUST ONLY EVER SHRINK.
#:
#: KEYED BY TARGET AND NOT BY BACKEND, because an encoder arrives before the
#: object writers do: x86-64 encodes its own instructions and writes ELF, and
#: still has no COFF or Mach-O writer to put them in. Keyed by backend alone,
#: removing it would claim a completeness it does not have on Windows, and
#: keeping it would deny the ELF path that works.
#: IT IS EMPTY, which is what it was written to become. Every binary backend
#: now encodes its own instructions and writes its own object for every target
#: it serves. A pair added back here needs the reason written down with it.
KNOWN_TEXT_EMITTERS: set[tuple[str, str]] = set()


def _machine_targets(backend: str) -> list[str]:
    """Every registered target this binary backend is expected to serve.

    MATCHED ON ARCHITECTURE, which is what decides whether a code generator
    applies at all: the x86-64 backend serves each x86_64 target and no other.
    A source target is excluded even when the arch matches -- `cpyext` emits C
    through a different backend entirely.
    """
    arch = {"x86-64": "x86_64", "x86-32": "x86", "arm64": "aarch64",
            "arm32": "arm", "jvm": "jvm", "pybc": "cpython"}.get(backend)
    if arch is None:
        return []
    return sorted(n for n in target_registry.available()
                  if target_registry.get(n).arch == arch
                  and not target_registry.get(n).is_source)


#: What a LANGUAGE backend is allowed to name its output.
TEXT_SUFFIXES = (".c", ".h", ".ll", ".wat")

#: Source or assembly. A binary backend may emit NONE of these -- each one
#: means a stage was left to an external assembler or compiler.
CODE_SUFFIXES = (".c", ".h", ".ll", ".wat", ".s", ".asm", ".S")


def _artifacts(backend: str, target: str | None = None) -> dict[str, bytes]:
    be = backend_registry.get(backend)
    chosen = target_registry.get(target or be.default_target)
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "prog.py"
        path.write_text(SOURCE, encoding="utf-8")
        result = compile_source(
            Options(source=path, backend=backend, target=chosen),
            DiagnosticSink())
        assert result.ok, f"{backend} did not compile for {chosen.name}"
        return dict(result.artifacts)


#: EVERY BINARY BACKEND AND TARGET IT SERVES, one case each. Built here rather
#: than in the decorator so the two tests below partition the same list and
#: cannot disagree about what the whole set is.
BINARY_PAIRS = [(b, t) for b in READY
                if backend_registry.get(b).kind == "binary"
                for t in _machine_targets(b)]


def _looks_like_text(data: bytes) -> bool:
    """Whether these bytes are source a human wrote conventions for.

    Decoding as UTF-8 is not the test on its own: a class file decodes often
    enough by accident. What settles it is a NUL byte, which no source file
    has and nearly every binary format does in its first few words.
    """
    if b"\x00" in data[:512]:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


class TestEveryBackendDeclaresWhatItIs:
    @harness.cases("backend", BACKENDS)
    def test_the_kind_is_one_of_the_two(self, backend):
        kind = backend_registry.get(backend).kind
        assert kind in ("language", "binary"), f"{backend}: kind={kind!r}"


class TestLanguageBackendsEmitSource:
    @harness.cases("backend", [b for b in READY
                               if backend_registry.get(b).kind == "language"])
    def test_the_artifact_is_text(self, backend):
        for name, data in _artifacts(backend).items():
            assert _looks_like_text(data), f"{backend}/{name} is not source"
            assert name.endswith(TEXT_SUFFIXES), f"{backend}/{name}: odd suffix"


class TestBinaryBackendsEmitBytes:
    @harness.cases("backend, target",
                   [p for p in BINARY_PAIRS if p not in KNOWN_TEXT_EMITTERS])
    def test_the_artifact_is_not_text(self, backend, target):
        """The claim `kind = "binary"` makes, checked against the bytes.

        TWO CHECKS, because "no text at all" is the wrong rule. A jar carries
        a `META-INF/MANIFEST.MF`, which is text and is not code -- packaging
        metadata that has to be readable. What a binary backend must not emit
        is SOURCE OR ASSEMBLY, and it must emit at least one thing that really
        is bytes; a backend passing only the first check could emit nothing.
        """
        artifacts = _artifacts(backend, target)
        for name in artifacts:
            assert not name.endswith(CODE_SUFFIXES), (
                f"{backend}/{target}/{name} is source or assembly; a binary "
                f"backend must encode its own output rather than leave it "
                f"to `as`")
        assert any(not _looks_like_text(d) for d in artifacts.values()), (
            f"{backend} on {target} emitted nothing binary: "
            f"{sorted(artifacts)}")

    @harness.cases("backend, target", sorted(KNOWN_TEXT_EMITTERS))
    def test_the_gap_list_is_not_stale(self, backend, target):
        """A pair that has grown an encoder must LEAVE the list.

        Otherwise the list stops describing anything and starts being a place
        where exemptions accumulate -- which is how a known gap becomes a
        permanent one.
        """
        emitted = _artifacts(backend, target)
        assert any(n.endswith(CODE_SUFFIXES) for n in emitted), (
            f"{backend} on {target} no longer emits text -- remove the pair "
            f"from KNOWN_TEXT_EMITTERS")

    def test_no_binary_backend_emits_text_any_more(self):
        """THE FILE'S WHOLE POINT, now that the answer is none.

        This started as a list of backends that stopped at `.s` and let `as`
        finish. It is empty; the assertion is that it stays that way, and a
        pair added back has to edit this test and say why.
        """
        assert KNOWN_TEXT_EMITTERS == set(), (
            f"a binary backend emits text again: "
            f"{sorted(KNOWN_TEXT_EMITTERS)}")

    @harness.cases("backend,target",
                   [p for p in BINARY_PAIRS
                    if p[0] in ("x86-64", "arm64")])
    def test_the_pair_writes_its_own_object(self, backend, target):
        """Every machine pair, asserted rather than merely absent from a list.

        A pair silently dropped from `BINARY_PAIRS` -- by a target being
        renamed, say -- would leave nothing testing it, and the suite would go
        quiet about a path that works. Now that the gap list is empty this is
        the only place the machine backends' output is named.
        """
        import struct
        (name, data), = _artifacts(backend, target).items()
        assert name.endswith((".o", ".obj")), name
        expected = target_registry.get(target).object_format
        if expected == "elf":
            assert data[:4] == b"\x7fELF", "not an ELF object"
        elif expected == "macho":
            assert struct.unpack_from("<I", data, 0)[0] == 0xFEEDFACF, (
                "not a 64-bit Mach-O object")
        else:
            # COFF HAS NO MAGIC. The first field is the machine number, which
            # is the only thing at a fixed offset that says what this is --
            # and it is the ARCHITECTURE'S, so a Windows-on-ARM object says
            # 0xAA64 where an x86-64 one says 0x8664.
            wanted = {"x86_64": 0x8664, "aarch64": 0xAA64}[
                target_registry.get(target).arch]
            assert struct.unpack_from("<H", data, 0)[0] == wanted, (
                f"not a COFF object for {target_registry.get(target).arch}")


class TestAnUnfinishedBackendRefuses:
    """A registered name that cannot emit must SAY so, not crash.

    The six stubs exist so `uasm plugin backends` shows the whole matrix rather
    than four names, with the unfinished half marked. The price of that is a
    name a user can select, so the refusal is part of the contract: it names
    the backend and what is missing, and it arrives as a diagnostic rather
    than a traceback.
    """

    UNFINISHED = sorted(b for b in BACKENDS
                        if not backend_registry.get(b).ready)

    def test_there_are_stubs_and_they_are_marked(self):
        assert self.UNFINISHED, "no unfinished backends; update this file"
        for name in self.UNFINISHED:
            assert not backend_registry.get(name).ready, (
                f"{name} is in UNFINISHED but declares itself ready")

    @harness.cases("backend", UNFINISHED)
    def test_it_refuses_with_a_reason(self, backend):
        from uasm.backend.base import BackendUnsupported
        be = backend_registry.get(backend)
        try:
            be.emit(None, None)
        except BackendUnsupported as exc:
            assert "not written yet" in str(exc) or "not identified" in str(exc)
            assert len(str(exc)) > 40, "the refusal does not say what is missing"
        else:
            raise AssertionError(f"{backend} emitted something")

    @harness.cases("backend", UNFINISHED)
    def test_it_still_declares_a_kind(self, backend):
        """Known before it is written, because it decides the whole design."""
        assert backend_registry.get(backend).kind in ("language", "binary")


class TestCSymbolsAreMangled:
    """A Python name that is a C keyword must not reach the C backend raw.

    FOUND BY `funcaddr`. The C backend has three places that write a symbol --
    `GLOBAL_ADDR`, `CALL` and `FUNC_ADDR` -- and two of them went through
    `_cname`. The third did not, and survived because NOTHING EMITTED
    `Op.FUNC_ADDR` from the static path until the subset grew an indirect
    call. It was reachable the whole time from ordinary Python: a program
    defining `def double(...)` and putting it in a list produced C reading
    `(uintptr_t)&double`, which gcc rejects with "expected expression".

    So this is not a test of the new intrinsic. It is a test that a user's
    choice of function name cannot break the backend, which is what the bug
    actually was.
    """

    KEYWORDS = ["double", "int", "float", "long", "register", "static",
                "signed", "union", "switch"]

    @harness.cases("name", KEYWORDS)
    def test_a_function_named_after_a_c_keyword_compiles(self, name, tmp_path):
        import pathlib
        source = pathlib.Path(tmp_path) / "prog.py"
        source.write_text(
            f"def {name}(x):\n"
            f"    return x * 2\n"
            f"\n"
            f"fs = [{name}]\n"
            f"print(fs[0](21))\n", encoding="utf-8")
        result = compile_source(
            Options(source=source, backend="c",
                    target=target_registry.get("c")), DiagnosticSink())
        assert result.ok, f"a function named {name!r} did not compile"
        (text,) = [d.decode("utf-8") for d in result.artifacts.values()]
        # THE ASSERTION IS ON THE C, not on the program running: taking the
        # address is what was wrong, and it appears whether or not the
        # toolchain is present to build it.
        assert f"&{name};" not in text, (
            f"the C takes the address of {name!r} unmangled")
