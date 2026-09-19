"""The builtin linker, exercised by running what it produces.

THE CLAIM THIS FILE CHECKS is a negative one -- that nothing outside this
repository is involved -- and a negative claim cannot be checked by looking
at the code. So every test here builds a real program, runs it, and compares
the output; and the two that matter most run it with `env -i`, so a `PATH`
pointing at a toolchain cannot be what made it work.

WHY THE PROGRAMS ARE THE ONES THEY ARE, smallest first:

  * `FLOOR_PROGRAM` uses `plat_write`, `plat_exit` and `plat_heap` and
    nothing else, so it fails if the syscall stubs in `link/freestanding.py`
    are wrong and passes with no object runtime at all. It is the same
    program `test_platform_floor.py` builds, for the same reason.
  * `print("hello")` is the whole Python runtime: the object runtime is
    compiled by uasm's own C frontend, linked, and run. It fails if any of
    the four pieces is wrong.

AND THEN THERE IS WINDOWS, which is not run, because there is no Windows on
the machine this was written on and no emulator either. `TestWindows` builds
real programs and takes the image apart again -- the headers, the section
table and the import directory -- with its own struct walk rather than with
the writer that produced them, and it checks the relocation ARITHMETIC by
reading the displacement out of a `call` and adding it up. That is a weaker
claim than "it printed hello", and saying which tests make which claim is
the point of writing this down.

THESE TESTS ARE SLOW THE FIRST TIME -- the object runtime is 22k lines of C
and takes about twelve seconds to compile -- and fast afterwards, because
`link/runtimeobj.py` caches it under the workdir by the hash of its own
source. The cache is per workdir, and `tmp_path` gives each test its own, so
`test_the_runtime_is_cached` is the one that measures the reuse.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from tests import harness
from tests.harness import snapshot

SRC = snapshot.current(Path(__file__).resolve().parents[3])

#: The floor and nothing else. See `test_platform_floor.py`, whose copy of
#: this is the oracle for what it should print.
FLOOR_PROGRAM = """\
def write_i64(v: i64) -> None:
    buf: ptr = alloca(24)
    at: i64 = 24
    neg: bool = v < 0
    m: u64 = u64(v)
    if neg:
        m = 0 - m
    at = at - 1
    store(u8, u8(48 + m % 10), offset(buf, at))
    m = m // 10
    while m > 0:
        at = at - 1
        store(u8, u8(48 + m % 10), offset(buf, at))
        m = m // 10
    if neg:
        at = at - 1
        store(u8, 45, offset(buf, at))
    plat_write(1, offset(buf, at), 24 - at)


def newline() -> None:
    buf: ptr = alloca(1)
    store(u8, 10, buf)
    plat_write(1, buf, 1)


def main() -> int:
    write_i64(0)
    newline()
    write_i64(-9223372036854775808)
    newline()
    p: ptr = plat_heap(64)
    store(i64, 4242, p)
    write_i64(load(i64, p))
    newline()
    return 0
"""

FLOOR_EXPECTED = "0\n-9223372036854775808\n4242\n"

#: `plat_exit` ENDS THE PROCESS, so the second write must never happen and
#: the status must be the one asked for. A floor whose exit merely returned
#: would pass a test that only looked at stdout.
EXIT_PROGRAM = """\
def main() -> int:
    buf: ptr = alloca(3)
    store(u8, 104, buf)
    store(u8, 105, offset(buf, 1))
    store(u8, 10, offset(buf, 2))
    plat_write(1, buf, 3)
    plat_exit(7)
    plat_write(1, buf, 3)
    return 0
"""


def _cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    base = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run([sys.executable, "-m", "uasm", *args],
                          capture_output=True, text=True, env=env or base)


def _write(tmp_path: Path, source: str, name: str = "prog.py") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def _build(tmp_path: Path, source: Path, *extra: str) -> Path:
    """Build and link `source` with the builtin linker. Returns the program."""
    out = tmp_path / "prog"
    got = _cli("build", str(source), "--link", "-ln", "builtin",
               "-o", str(out), "--workdir", str(tmp_path / ".uasm"), *extra)
    assert got.returncode == 0, got.stderr[-3000:]
    assert out.exists(), got.stdout + got.stderr
    return out


def _run(program: Path, *, clean_env: bool = False) -> \
        subprocess.CompletedProcess:
    """Run it. `clean_env` empties the environment, PATH included.

    THAT IS THE WHOLE POINT OF THE FLAG. A statically linked program needs
    no loader, no library path and no PATH; running one with `env -i` is how
    this file's central claim is checked rather than asserted.
    """
    return subprocess.run([str(program)], capture_output=True,
                          env={} if clean_env else None, timeout=120)


class TestTheFloorAlone:
    """A program that uses only the three platform functions."""

    def test_it_runs_and_prints_what_the_interpreter_does(self, tmp_path):
        source = _write(tmp_path, FLOOR_PROGRAM)
        oracle = _cli("run", str(source))
        assert oracle.returncode == 0, oracle.stderr[-2000:]
        assert oracle.stdout == FLOOR_EXPECTED, oracle.stdout

        got = _run(_build(tmp_path, source))
        assert got.returncode == 0, got.stderr[-2000:]
        assert got.stdout.decode() == FLOOR_EXPECTED

    def test_it_runs_with_no_environment_at_all(self, tmp_path):
        program = _build(tmp_path, _write(tmp_path, FLOOR_PROGRAM))
        got = _run(program, clean_env=True)
        assert got.returncode == 0, got.stderr[-2000:]
        assert got.stdout.decode() == FLOOR_EXPECTED

    def test_plat_exit_ends_the_process_with_the_status(self, tmp_path):
        program = _build(tmp_path, _write(tmp_path, EXIT_PROGRAM))
        got = _run(program)
        assert got.returncode == 7, (got.returncode, got.stderr[-2000:])
        assert got.stdout == b"hi\n", got.stdout

    def test_the_image_is_static_and_has_no_interpreter(self, tmp_path):
        import struct
        program = _build(tmp_path, _write(tmp_path, FLOOR_PROGRAM))
        blob = program.read_bytes()
        assert blob[:4] == b"\x7fELF"
        e_type, = struct.unpack_from("<H", blob, 16)
        assert e_type == 2, "the image should be ET_EXEC"
        phoff, = struct.unpack_from("<Q", blob, 0x20)
        phentsize, phnum = struct.unpack_from("<HH", blob, 0x36)
        kinds = {struct.unpack_from("<I", blob, phoff + i * phentsize)[0]
                 for i in range(phnum)}
        # PT_INTERP is 3, and its absence is what "static" means to a loader.
        assert 3 not in kinds, "a static image must name no interpreter"
        # PT_DYNAMIC is 2: there is nothing for a dynamic loader to do.
        assert 2 not in kinds, "a static image must have no dynamic section"


class TestAWholePythonProgram:
    """`print("hello")`: the object runtime, compiled by uasm and linked."""

    def test_it_prints_what_the_interpreter_does(self, tmp_path):
        source = _write(tmp_path, 'print("hello")\nprint(2 + 3)\n')
        oracle = _cli("run", str(source))
        assert oracle.returncode == 0, oracle.stderr[-2000:]
        want = oracle.stdout

        got = _run(_build(tmp_path, source))
        assert got.returncode == 0, got.stderr[-2000:]
        assert got.stdout.decode() == want, (got.stdout, want)

    def test_it_runs_with_no_environment_at_all(self, tmp_path):
        source = _write(tmp_path, 'print("hello")\n')
        got = _run(_build(tmp_path, source), clean_env=True)
        assert got.returncode == 0, got.stderr[-2000:]
        assert got.stdout == b"hello\n"

    def test_the_runtime_is_cached_between_links(self, tmp_path):
        """The second link reuses the compiled runtime rather than rebuilding.

        MEASURED BY THE FILE AND NOT BY THE CLOCK. A timing assertion would
        be a flake on a loaded machine; the cache is a file whose name is the
        hash of its own source, so "was it reused" is "is there still exactly
        one of them, and is it older than the second program".
        """
        source = _write(tmp_path, 'print("hello")\n')
        _build(tmp_path, source)
        cache = tmp_path / ".uasm" / "runtime"
        objects = sorted(cache.glob("*.o"))
        assert len(objects) == 1, objects
        stamp = objects[0].stat().st_mtime_ns

        _build(tmp_path, source)
        again = sorted(cache.glob("*.o"))
        assert [p.name for p in again] == [p.name for p in objects]
        assert again[0].stat().st_mtime_ns == stamp, \
            "the runtime was recompiled when it should have been reused"


class TestBuildAndLinkAreSeparateVerbs:
    def test_build_writes_an_object_and_link_makes_the_program(self, tmp_path):
        source = _write(tmp_path, 'print("hello")\n')
        obj = tmp_path / "prog.o"
        built = _cli("build", str(source), "-o", str(obj),
                     "--workdir", str(tmp_path / ".uasm"))
        assert built.returncode == 0, built.stderr[-3000:]
        assert obj.exists(), built.stdout + built.stderr
        assert obj.read_bytes()[:4] == b"\x7fELF"
        # ET_REL: it is an object, not a program.
        import struct
        assert struct.unpack_from("<H", obj.read_bytes(), 16)[0] == 1

        out = tmp_path / "prog"
        linked = _cli("link", str(obj), "-o", str(out),
                      "--workdir", str(tmp_path / ".uasm"))
        assert linked.returncode == 0, linked.stderr[-3000:]
        got = _run(out)
        assert got.returncode == 0, got.stderr[-2000:]
        assert got.stdout == b"hello\n"


class TestItSaysWhatItCannotDo:
    def test_an_undefined_symbol_is_named(self, tmp_path):
        """A missing symbol is a diagnostic naming it, not a traceback."""
        from uasm.backend.objfile import (
            EM_X86_64, SHF_ALLOC, SHF_EXECINSTR, STB_GLOBAL, STT_FUNC,
            ElfObject, Relocation, Symbol,
        )
        from uasm.link.staticlink import LinkFailed, link

        obj = ElfObject(EM_X86_64)
        obj.section(".text", b"\xe8\x00\x00\x00\x00\xc3",
                    flags=SHF_ALLOC | SHF_EXECINSTR, align=16)
        obj.symbol(Symbol("_start", ".text", 0, 6, STB_GLOBAL, STT_FUNC))
        obj.symbol(Symbol("nowhere_at_all", "", binding=STB_GLOBAL))
        obj.relocate(".text", Relocation(1, "nowhere_at_all", 4, -4))
        try:
            link([("made-up.o", obj.to_bytes())])
        except LinkFailed as exc:
            assert "1 undefined symbol" in exc.message, exc.message
            assert "nowhere_at_all" in exc.detail, exc.detail
            assert "made-up.o" in exc.detail, exc.detail
        else:
            harness.fail("linking an undefined symbol should have failed")

    def test_two_definitions_of_one_name_are_refused(self, tmp_path):
        from uasm.backend.objfile import (
            EM_X86_64, SHF_ALLOC, SHF_EXECINSTR, STB_GLOBAL, STT_FUNC,
            ElfObject, Symbol,
        )
        from uasm.link.staticlink import LinkFailed, link

        def one(name: str) -> bytes:
            obj = ElfObject(EM_X86_64)
            obj.section(".text", b"\xc3", flags=SHF_ALLOC | SHF_EXECINSTR,
                        align=16)
            obj.symbol(Symbol("_start", ".text", 0, 1, STB_GLOBAL, STT_FUNC))
            obj.symbol(Symbol("twice", ".text", 0, 1, STB_GLOBAL, STT_FUNC))
            return obj.to_bytes()

        try:
            link([("a.o", one("a")), ("b.o", one("b"))])
        except LinkFailed as exc:
            assert "defined twice" in exc.message, exc.message
        else:
            harness.fail("two definitions should have been refused")


# ── Windows ─────────────────────────────────────────────────────────────────
#
# NOT RUN, AND THE FILE SAYS SO. Everything below takes a linked PE apart and
# checks it against what the format requires. Where a claim can be made
# arithmetically -- a relocation reaches the slot it names -- it is, because
# that is the part a structural test can still get right.

def _pe(blob: bytes) -> dict:
    """A linked PE, taken apart by hand.

    BY HAND AND NOT WITH `pewrite`. A test that parsed the image with the
    module that wrote it would agree with itself about a field in the wrong
    place; this walks the format from `e_lfanew` the way a loader does.
    """
    import struct
    assert blob[:2] == b"MZ", blob[:8]
    at, = struct.unpack_from("<I", blob, 0x3C)
    assert blob[at:at + 4] == b"PE\0\0", (at, blob[at:at + 8])
    machine, nsections, _, _, _, opt_size, _ = struct.unpack_from(
        "<HHIIIHH", blob, at + 4)
    opt = at + 24
    magic, = struct.unpack_from("<H", blob, opt)
    entry, = struct.unpack_from("<I", blob, opt + 16)
    base, = struct.unpack_from("<Q", blob, opt + 24)
    salign, falign = struct.unpack_from("<II", blob, opt + 32)
    image_size, headers_size = struct.unpack_from("<II", blob, opt + 56)
    subsystem, = struct.unpack_from("<H", blob, opt + 68)
    ndirs, = struct.unpack_from("<I", blob, opt + 108)
    dirs = [struct.unpack_from("<II", blob, opt + 112 + 8 * i)
            for i in range(ndirs)]
    sections = []
    table = at + 24 + opt_size
    for i in range(nsections):
        name, vsize, vaddr, rawsize, rawptr = struct.unpack_from(
            "<8sIIII", blob, table + 40 * i)
        chars, = struct.unpack_from("<I", blob, table + 40 * i + 36)
        sections.append({"name": name.rstrip(b"\0").decode(),
                         "vsize": vsize, "vaddr": vaddr,
                         "rawsize": rawsize, "rawptr": rawptr,
                         "characteristics": chars})
    return {"machine": machine, "magic": magic, "entry": entry, "base": base,
            "salign": salign, "falign": falign, "image_size": image_size,
            "headers_size": headers_size, "subsystem": subsystem,
            "dirs": dirs, "sections": sections,
            "table_end": table + 40 * nsections}


def _imports(blob: bytes, image: dict) -> dict[str, list[str]]:
    """The import directory, walked the way the Windows loader walks it."""
    import struct

    def at_rva(rva: int) -> int:
        for sec in image["sections"]:
            if sec["vaddr"] <= rva < sec["vaddr"] + max(sec["vsize"],
                                                        sec["rawsize"]):
                return sec["rawptr"] + (rva - sec["vaddr"])
        raise AssertionError(f"rva {rva:#x} is in no section")

    def name_at(rva: int) -> str:
        start = at_rva(rva)
        return blob[start:blob.index(b"\0", start)].decode()

    rva, size = image["dirs"][1]
    if not size:
        return {}
    out: dict[str, list[str]] = {}
    at = at_rva(rva)
    while True:
        ilt, _, _, name_rva, iat = struct.unpack_from("<IIIII", blob, at)
        if not (ilt or name_rva or iat):
            break
        funcs = []
        walk = at_rva(ilt)
        while True:
            thunk, = struct.unpack_from("<Q", blob, walk)
            if not thunk:
                break
            # A hint/name pair: two bytes of hint, then the name.
            funcs.append(name_at(thunk + 2))
            walk += 8
        out[name_at(name_rva)] = funcs
        at += 20
    return out


class TestWindows:
    """A PE, linked with no toolchain, taken apart again. NOT RUN."""

    def _build(self, tmp_path: Path, source: str) -> tuple[bytes, dict]:
        path = _write(tmp_path, source)
        out = tmp_path / "prog.exe"
        got = _cli("build", str(path), "--target", "x86_64-windows",
                   "--link", "-ln", "builtin", "-o", str(out),
                   "--workdir", str(tmp_path / ".uasm"))
        assert got.returncode == 0, got.stderr[-3000:]
        blob = out.read_bytes()
        return blob, _pe(blob)

    def test_the_floor_program_links_into_a_pe(self, tmp_path):
        blob, image = self._build(tmp_path, FLOOR_PROGRAM)
        assert image["machine"] == 0x8664
        assert image["magic"] == 0x20B, "PE32+, not PE32"
        assert image["subsystem"] == 3, "a console program"
        assert image["base"] == 0x400000
        assert image["salign"] == image["falign"] == 0x1000

    def test_the_headers_do_not_overlap_the_first_section(self, tmp_path):
        """The bug this test exists for wrote the headers over `.text`.

        The layout gives every byte `vaddr = base + file offset`, so the
        first section has to start AFTER the headers -- and the first
        version of the PE writer reserved the ELF header's worth of room,
        which is a quarter of what a PE needs.
        """
        blob, image = self._build(tmp_path, FLOOR_PROGRAM)
        first = min(s["vaddr"] for s in image["sections"])
        assert image["headers_size"] <= first, (image["headers_size"], first)
        assert image["table_end"] <= first, "the section table overlaps it"

    def test_every_section_starts_on_a_page(self, tmp_path):
        """Which PE requires of `VirtualAddress`, where ELF does not."""
        blob, image = self._build(tmp_path, 'print("hello")\n')
        for sec in image["sections"]:
            assert sec["vaddr"] % image["salign"] == 0, sec
            if sec["rawptr"]:
                assert sec["rawptr"] % image["falign"] == 0, sec
                assert sec["rawptr"] == sec["vaddr"], \
                    "the layout makes the file offset and the RVA equal"

    def test_the_entry_point_is_inside_the_code(self, tmp_path):
        blob, image = self._build(tmp_path, FLOOR_PROGRAM)
        text = next(s for s in image["sections"] if s["name"] == ".text")
        assert text["vaddr"] <= image["entry"] < text["vaddr"] + text["vsize"]
        # IMAGE_SCN_MEM_EXECUTE and not IMAGE_SCN_MEM_WRITE.
        assert text["characteristics"] & 0x20000000
        assert not text["characteristics"] & 0x80000000

    def test_the_import_table_names_what_the_floor_calls(self, tmp_path):
        blob, image = self._build(tmp_path, FLOOR_PROGRAM)
        got = _imports(blob, image)
        assert list(got) == ["kernel32.dll"], got
        assert set(got["kernel32.dll"]) == {
            "ExitProcess", "GetStdHandle", "VirtualAlloc", "WriteFile"}, got
        iat_rva, iat_size = image["dirs"][12]
        # FOUR SLOTS AND THE ZERO THAT ENDS THEM. The directory covers the
        # whole table, terminator included, because the loader is told to
        # make that RANGE writable.
        assert iat_size == 8 * 5, iat_size
        table = next(s for s in image["sections"] if s["name"] == ".idata")
        assert table["vaddr"] <= iat_rva < table["vaddr"] + table["vsize"]

    def test_an_indirect_call_reaches_the_slot_it_names(self, tmp_path):
        """The relocation arithmetic, checked by doing it in reverse.

        THIS IS THE ONE TEST HERE THAT IS NOT STRUCTURAL. `_start` ends with
        `call *__imp_ExitProcess(%rip)`; the displacement in it is four bytes
        that only add up to the right address if the linker patched them
        against a VIRTUAL ADDRESS. The first version of this seeded the
        import slots as RVAs, so every such call landed four megabytes
        short -- and nothing structural would have noticed.
        """
        import struct
        blob, image = self._build(tmp_path, FLOOR_PROGRAM)
        text = next(s for s in image["sections"] if s["name"] == ".text")
        code = blob[text["rawptr"]:text["rawptr"] + text["vsize"]]
        iat_rva, iat_size = image["dirs"][12]
        slots = set(range(iat_rva, iat_rva + iat_size - 8, 8))

        seen = 0
        for at in range(len(code) - 6):
            if code[at:at + 2] != b"\xff\x15":
                continue
            disp, = struct.unpack_from("<i", code, at + 2)
            target = text["vaddr"] + at + 6 + disp
            assert target in slots, \
                f"the call at {at:#x} reaches {target:#x}, which is no slot"
            seen += 1
        assert seen >= 4, f"only {seen} indirect calls found"

    def test_a_whole_python_program_links(self, tmp_path):
        """`print("hello")`, object runtime and all, into a PE."""
        blob, image = self._build(tmp_path, 'print("hello")\n')
        names = [s["name"] for s in image["sections"]]
        assert ".text" in names and ".idata" in names, names
        assert _imports(blob, image)["kernel32.dll"], "the floor is linked in"
        # A `.bss` occupies memory and no file. THE FILE IS SMALLER THAN THE
        # IMAGE, which is the whole reason to have one.
        assert image["image_size"] > len(blob)


class TestItRefusesWhatItCannotDo:
    def test_an_import_with_no_known_dll_is_named(self):
        from uasm.link.pewrite import PeError, imports_needed
        try:
            imports_needed({"__imp_NoSuchFunction"})
        except PeError as exc:
            assert "NoSuchFunction" in str(exc), str(exc)
        else:
            harness.fail("an unknown import should have been refused")

    def test_objects_in_two_formats_are_refused(self):
        """An ELF and a COFF object have nothing to say to each other."""
        from uasm.backend.objfile import (
            EM_X86_64, IMAGE_FILE_MACHINE_AMD64, IMAGE_SCN_CNT_CODE,
            IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ, SHF_ALLOC,
            SHF_EXECINSTR, STB_GLOBAL, STT_FUNC, CoffObject, CoffSymbol,
            ElfObject, Symbol,
        )
        from uasm.link.staticlink import LinkFailed, link

        elf = ElfObject(EM_X86_64)
        elf.section(".text", b"\xc3", flags=SHF_ALLOC | SHF_EXECINSTR)
        elf.symbol(Symbol("_start", ".text", 0, 1, STB_GLOBAL, STT_FUNC))

        coff = CoffObject(IMAGE_FILE_MACHINE_AMD64)
        coff.section(".text", b"\xc3",
                     characteristics=IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_READ
                     | IMAGE_SCN_MEM_EXECUTE)
        coff.symbol(CoffSymbol(name="other", section=".text", value=0, size=1,
                               binding=1, kind=2))
        try:
            link([("a.o", elf.to_bytes()), ("b.obj", coff.to_bytes())])
        except LinkFailed as exc:
            assert "different object formats" in exc.message, exc.message
        else:
            harness.fail("two containers in one link should be refused")
