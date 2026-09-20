"""The JVM backend, executed.

A class file that a JVM refuses is not a partial success, and nothing short of
running one finds out: the format has a verifier with opinions about stack
maps, local types and operand depths, and a writer can satisfy `javap` and
still produce something `java` will not load. So these build a jar and run it,
and the assertion is the program's output.

THE ORACLE IS THE INTERPRETER. `uasm run` executes the same IR and defines
what the program means, so a disagreement is always the backend's -- which is
the property that makes a differential test worth more than a golden file
somebody would eventually update to match a bug.

Every test here declares `java`. Building a jar needs no JDK at all -- the
class writer and the jar packager are both Python -- so the unit tests still
cover the backend on a machine with no Java, and only these are blocked.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests import harness
from tests.harness import snapshot

SRC = snapshot.current(Path(__file__).resolve().parents[3])

PROGRAM = """\
def double(n: int) -> int:
    return n * 2

def main() -> int:
    total: int = 0
    for i in range(5):
        total = total + double(i)
    print(total)
    return 0
"""


def run_cli(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run([sys.executable, "-m", "uasm", *args],
                          capture_output=True, text=True, env=env)


#: A JVM sizes its initial heap from physical memory and reserves it up front
#: -- 192 MB on an ordinary machine. The suite runs one JVM per worker, and on
#: a four-core box that reservation failed with "the paging file is too small",
#: which reaches a test as an empty stdout and reads exactly like a miscompile.
#: These programs allocate one byte array; sixty-four megabytes is generous.
JAVA_FLAGS = ["-Xmx64m", "-XX:-UsePerfData"]


def java(jar: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["java", *JAVA_FLAGS, "-jar", str(jar), *args],
                          capture_output=True, text=True)


def interpret(source: Path) -> str:
    """What the program means, according to the reference implementation.

    A NON-ZERO STATUS IS NOT A FAILURE HERE. `uasm run` reports the
    entry's return value as the process exit code, exactly as every compiled
    backend does -- so a program ending `return 45` exits 45, and asserting 0
    would reject the reference implementation for being right. It did: this
    read `assert r.returncode == 0` and started failing the moment the
    interpreter stopped discarding the status.

    What still has to be zero is the COMPILER's own trouble, which is a
    different thing arriving on a different channel. A trap prints `trap:` and
    a diagnostic prints `error[`, and neither is a program's answer.
    """
    r = run_cli("run", str(source))
    assert "trap:" not in r.stderr and "error[" not in r.stderr, r.stderr
    return r.stdout


def compile_ir(tmp_path: Path, text: str, *flags: str) -> Path:
    """Hand-written IR to a jar.

    IR text goes through a Python file rather than `build prog.ir` because the
    build command takes source and the IR is not a frontend's language -- so
    the module is parsed and handed to the backend directly, which is also how
    `docs/BACKENDS.md` tells a backend author to test one.
    """
    source = tmp_path / "prog.ir"
    source.write_text(text, encoding="utf-8")
    jar = tmp_path / "prog.jar"
    script = tmp_path / "build.py"
    script.write_text(f"""
import sys, zipfile
from pathlib import Path
from uasm import target as target_registry
from uasm.backend import load_builtin
from uasm.backends.jvm import resolve
from uasm.backends.jvm.emit import JvmBackend
from uasm.ir import verify
from uasm.ir.printer import parse_module

load_builtin()
module = parse_module(Path(sys.argv[1]).read_text(encoding="utf-8"))
verify(module)
be = JvmBackend(resolve(**dict(a.split("=", 1) for a in sys.argv[3:])))
artifacts = be.emit(module, target_registry.get("jvm"))
with zipfile.ZipFile(sys.argv[2], "w") as jar:
    for name, data in artifacts.items():
        jar.writestr(name, data)
""", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    r = subprocess.run([sys.executable, str(script), str(source), str(jar),
                        *flags], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    return jar


def agrees(tmp_path: Path, ir_text: str, *flags: str) -> str:
    """Run `ir_text` both ways and assert they agree. Returns the output."""
    source = tmp_path / "prog.ir"
    source.write_text(ir_text, encoding="utf-8")
    expected = interpret(source)
    got = java(compile_ir(tmp_path, ir_text, *flags))
    assert got.returncode in (0, *range(1, 256)), got.stderr
    assert "Exception" not in got.stderr, got.stderr
    assert got.stdout == expected, (
        f"the JVM disagrees with the interpreter\n"
        f"  jvm: {got.stdout!r}\n  ref: {expected!r}\n{got.stderr}")
    return got.stdout


def ir(body: str, *, externals=("put_int",), prelude: str = "") -> str:
    declarations = "\n".join(
        f"func {name}({args}) -> {ret} external"
        for name, args, ret in [
            ("put_int", "%0: i64", "void"),
            ("putchar", "%0: i64", "i64"),
            ("print_str", "%0: ptr", "void"),
            ("put_float", "%0: f64", "void"),
            ("put_bool", "%0: i64", "void"),
            ("put_none", "", "void"),
            ("py_pow_int", "%0: f64, %1: i64", "f64"),
        ] if name in externals)
    return f"module prog\n\n{prelude}\n{declarations}\n\n{body}\n"


@harness.needs("java")
class TestItRuns:
    # EVERY BUILD IN THIS CLASS PASSES `--link` BECAUSE `build` STOPS AT AN
    # UNLINKED OBJECT. The jar is the JVM toolchain's product, not the
    # backend's: without the link step `java -jar` is handed a file no
    # packager ever wrote. What each test here checks is a program that runs,
    # so each one has to ask for a program.
    def test_a_compiled_python_program_prints_what_cpython_would(
            self, tmp_path):
        source = tmp_path / "prog.py"
        source.write_text(PROGRAM, encoding="utf-8")
        jar = tmp_path / "prog.jar"
        r = run_cli("build", str(source), "--backend", "jvm", "--link",
                    "--java-version", "21", "-o", str(jar))
        assert r.returncode == 0, r.stderr
        assert jar.exists()
        got = java(jar)
        assert got.stdout.splitlines() == ["20"], got.stderr

    def test_the_exit_status_is_what_main_returned(self, tmp_path):
        source = tmp_path / "prog.py"
        source.write_text("def main() -> int:\n    return 3\n",
                          encoding="utf-8")
        jar = tmp_path / "prog.jar"
        assert run_cli("build", str(source), "--backend", "jvm", "--link",
                       "-o", str(jar)).returncode == 0
        assert java(jar).returncode == 3

    def test_output_with_no_trailing_newline_is_not_lost(self, tmp_path):
        """`System.exit` does not flush, and `System.out` is buffered."""
        source = tmp_path / "prog.py"
        source.write_text('def main() -> int:\n    print(41 + 1)\n'
                          '    return 0\n', encoding="utf-8")
        jar = tmp_path / "prog.jar"
        assert run_cli("build", str(source), "--backend", "jvm", "--link",
                       "-o", str(jar)).returncode == 0
        assert java(jar).stdout.strip() == "42"

    @harness.cases("release", ["8", "11", "17", "21"])
    def test_it_runs_at_every_class_version_a_modern_jvm_accepts(
            self, tmp_path, release):
        source = tmp_path / "prog.py"
        source.write_text(PROGRAM, encoding="utf-8")
        jar = tmp_path / f"prog{release}.jar"
        r = run_cli("build", str(source), "--backend", "jvm", "--link",
                    "--java-version", release, "-o", str(jar))
        assert r.returncode == 0, r.stderr
        got = java(jar)
        assert got.stdout.splitlines() == ["20"], got.stderr

    def test_a_version_the_jvm_is_too_old_for_is_refused_by_the_jvm(
            self, tmp_path):
        """`--class-version` is an escape hatch, and this is what it opens:
        the compiler writes what was asked for, and the JVM has the last
        word."""
        source = tmp_path / "prog.py"
        source.write_text(PROGRAM, encoding="utf-8")
        jar = tmp_path / "prog.jar"
        r = run_cli("build", str(source), "--backend", "jvm", "--link",
                    "--class-version", "75", "-o", str(jar))
        assert r.returncode == 0, r.stderr
        got = java(jar)
        assert got.returncode != 0
        assert "UnsupportedClassVersionError" in got.stderr


@harness.needs("java")
class TestItAgreesWithTheInterpreter:
    def test_integer_arithmetic(self, tmp_path):
        agrees(tmp_path, ir("""\
export func main() -> i64 {
entry:
    %0 = i64.const 1000000007
    %1 = i64.const -3
    %2 = i64.mul %0, %1
    call @put_int(%2)
    %3 = i64.div %2, %1
    call @put_int(%3)
    %4 = i64.rem %2, %0
    call @put_int(%4)
    %5 = i64.neg %2
    call @put_int(%5)
    %6 = i64.not %5
    call @put_int(%6)
    %7 = i64.const 0
    ret %7
}"""))

    def test_narrow_types_wrap_at_their_own_width(self, tmp_path):
        agrees(tmp_path, ir("""\
export func main() -> i64 {
entry:
    %0 = i8.const 127
    %1 = i8.const 1
    %2 = i8.add %0, %1
    %3 = i64.extend %2
    call @put_int(%3)
    %4 = u8.const 255
    %5 = u8.const 1
    %6 = u8.add %4, %5
    %7 = i64.extend %6
    call @put_int(%7)
    %8 = u8.const 200
    %9 = i64.extend %8
    call @put_int(%9)
    %10 = i16.const -32768
    %11 = i16.const 1
    %12 = i16.sub %10, %11
    %13 = i64.extend %12
    call @put_int(%13)
    %14 = i64.const 0
    ret %14
}"""))

    def test_unsigned_division_and_comparison(self, tmp_path):
        agrees(tmp_path, ir("""\
export func main() -> i64 {
entry:
    %0 = u32.const 4294967295
    %1 = u32.const 7
    %2 = u32.div %0, %1
    %3 = i64.extend %2
    call @put_int(%3)
    %4 = u32.rem %0, %1
    %5 = i64.extend %4
    call @put_int(%5)
    %6 = u32.lt %1, %0
    %7 = i64.extend %6
    call @put_int(%7)
    %8 = u32.shr %0, %1
    %9 = i64.extend %8
    call @put_int(%9)
    %10 = i64.const 0
    ret %10
}"""))

    def test_floats_and_their_comparisons(self, tmp_path):
        agrees(tmp_path, ir("""\
export func main() -> i64 {
entry:
    %0 = f64.const 3.5
    %1 = f64.const 0.7
    %2 = f64.div %0, %1
    call @put_float(%2)
    %3 = f64.rem %0, %1
    call @put_float(%3)
    %4 = i64.ftoi %2
    call @put_int(%4)
    %5 = f64.itof %4
    call @put_float(%5)
    %6 = f64.gt %0, %1
    %7 = i64.extend %6
    call @put_int(%7)
    %8 = i64.const 0
    ret %8
}""", externals=("put_int", "put_float")))

    def test_a_float_too_large_for_the_integer_it_converts_to(self, tmp_path):
        """`d2i` SATURATES -- 1e10 to i32 is 2147483647 -- and the reference
        interpreter wraps, giving 1410065408. The IR calls the out-of-range
        case undefined, so the tie is broken by what `uasm run` prints:
        going through a `long` and narrowing agrees with it."""
        agrees(tmp_path, ir("""\
export func main() -> i64 {
entry:
    %0 = f64.const 10000000000.0
    %1 = i32.ftoi %0
    %2 = i64.extend %1
    call @put_int(%2)
    %3 = i16.ftoi %0
    %4 = i64.extend %3
    call @put_int(%4)
    %5 = u16.ftoi %0
    %6 = i64.extend %5
    call @put_int(%6)
    %7 = u8.ftoi %0
    %8 = i64.extend %7
    call @put_int(%8)
    %9 = i64.ftoi %0
    call @put_int(%9)
    %10 = i64.const 0
    ret %10
}"""))

    def test_memory_at_every_width(self, tmp_path):
        agrees(tmp_path, ir("""\
export func main() -> i64 {
entry:
    %0 = ptr.alloca 64
    %1 = i8.const -3
    i8.store %1, %0
    %2 = i8.load %0
    %3 = i64.extend %2
    call @put_int(%3)
    %4 = i64.const 8
    %5 = ptr.offset %0, %4
    %6 = i32.const -70000
    i32.store %6, %5
    %7 = i32.load %5
    %8 = i64.extend %7
    call @put_int(%8)
    %9 = f64.const 2.5
    f64.store %9, %5
    %10 = f64.load %5
    %11 = i64.ftoi %10
    call @put_int(%11)
    %12 = ptr.global_addr @scratch
    %13 = i64.const 987654321
    i64.store %13, %12
    %14 = i64.load %12
    call @put_int(%14)
    %15 = i64.const 0
    ret %15
}""", prelude="global scratch: 64 bytes\n"))

    def test_a_global_larger_than_one_constant_pool_string(self, tmp_path):
        """Global data travels as constant-pool strings, chunked at 16 KiB
        because a character above 127 costs two bytes in the class file's
        modified UTF-8 and the format caps a string at 65535. Forty thousand
        bytes is three chunks, and the checksum catches one landing at the
        wrong offset."""
        data = bytes(((i * 7 + 13) % 251) + 1 for i in range(40000))
        literal = "".join("\\%02x" % b for b in data + b"\x00")
        agrees(tmp_path, ir(f"""\
export func main() -> i64 {{
entry:
    %0 = ptr.global_addr @big
    %1 = i64.const 0
    %2 = i64.const 0
    %3 = i64.const {len(data)}
    %4 = i64.const 1
    jump head
head:
    %5 = i64.lt %2, %3
    branch %5, body, done
body:
    %6 = ptr.offset %0, %2
    %7 = u8.load %6
    %8 = i64.extend %7
    %1 = i64.add %1, %8
    %2 = i64.add %2, %4
    jump head
done:
    call @put_int(%1)
    %9 = i64.const 0
    ret %9
}}""", prelude=f'global big = "{literal}" readonly\n'))

    def test_global_data_survives_zero_and_high_bytes(self, tmp_path):
        """The class file's string encoding is modified UTF-8, not UTF-8: a
        zero byte is written as two, and Python's own `encode` writes a NUL
        that terminates the string early."""
        raw = b"ab\x00cd\x00\xff\xfe\x00"
        literal = "".join("\\%02x" % b for b in raw)
        agrees(tmp_path, ir(f"""\
export func main() -> i64 {{
entry:
    %0 = ptr.global_addr @holes
    %1 = i64.const 0
    %2 = i64.const {len(raw)}
    %3 = i64.const 1
    %4 = i64.const 32
    jump head
head:
    %5 = i64.lt %1, %2
    branch %5, body, done
body:
    %6 = ptr.offset %0, %1
    %7 = u8.load %6
    %8 = i64.extend %7
    call @put_int(%8)
    %9 = i64.call @putchar(%4)
    %1 = i64.add %1, %3
    jump head
done:
    %10 = i64.const 0
    ret %10
}}""", externals=("put_int", "putchar"),
            prelude=f'global holes = "{literal}" readonly\n'))

    def test_the_remaining_host_functions(self, tmp_path):
        agrees(tmp_path, ir("""\
export func main() -> i64 {
entry:
    %0 = i64.const 10
    %1 = i64.const 1
    call @put_bool(%1)
    %2 = i64.call @putchar(%0)
    %3 = i64.const 0
    call @put_bool(%3)
    %4 = i64.call @putchar(%0)
    call @put_none()
    %5 = i64.call @putchar(%0)
    %6 = f64.const 0.1
    call @put_float(%6)
    %7 = i64.call @putchar(%0)
    ret %3
}""", externals=("put_bool", "put_none", "put_float", "putchar")))

    def test_a_string_global_reaches_stdout_unchanged(self, tmp_path):
        agrees(tmp_path, ir("""\
export func main() -> i64 {
entry:
    %0 = ptr.global_addr @greeting
    call @print_str(%0)
    %1 = i64.const 0
    ret %1
}""", externals=("print_str",),
            prelude='global greeting = "hi\\0a\\00" readonly\n'))

    def test_a_loop_with_a_value_live_across_the_back_edge(self, tmp_path):
        agrees(tmp_path, ir("""\
export func main() -> i64 {
entry:
    %0 = i64.const 0
    %1 = i64.const 0
    %2 = i64.const 10
    %3 = i64.const 1
    jump head
head:
    %4 = i64.lt %1, %2
    branch %4, body, done
body:
    %0 = i64.add %0, %1
    %1 = i64.add %1, %3
    jump head
done:
    call @put_int(%0)
    ret %0
}"""))

    def test_a_call_with_more_arguments_than_the_stack_allowance(
            self, tmp_path):
        """`max_stack` was a constant 16, and eight `i64` arguments is sixteen
        slots. The ninth produced a class the verifier refused to load, in a
        program with nothing unusual about it -- so the bound is computed from
        the widest call rather than assumed."""
        n = 12
        params = ", ".join(f"%{i}: i64" for i in range(n))
        adds = "\n".join(f"    %{n + i} = i64.add %{n + i - 1}, %{i}"
                         for i in range(1, n))
        consts = "\n".join(f"    %{i} = i64.const {i + 1}" for i in range(n))
        args = ", ".join(f"%{i}" for i in range(n))
        assert agrees(tmp_path, ir(f"""\
func total({params}) -> i64 {{
entry:
    %{n} = i64.copy %0
{adds}
    ret %{n + n - 1}
}}

export func main() -> i64 {{
entry:
{consts}
    %{n} = i64.call @total({args})
    call @put_int(%{n})
    %{n + 1} = i64.const 0
    ret %{n + 1}
}}""")).strip() == "78"

    def test_calling_through_a_function_pointer(self, tmp_path):
        """The JVM has no addressable code, so `func_addr` is an integer and
        `call_ptr` is a switch over the functions of that signature.

        Covered here: two pointers of one shape, a pointer chosen at run time
        so nothing can be folded, a second signature entirely, and a pointer
        stored to memory and loaded back -- which only works because an
        address is an ordinary value."""
        agrees(tmp_path, ir("""\
func twice(%0: i64) -> i64 {
entry:
    %1 = i64.const 2
    %2 = i64.mul %0, %1
    ret %2
}

func square(%0: i64) -> i64 {
entry:
    %1 = i64.mul %0, %0
    ret %1
}

func addf(%0: f64, %1: f64) -> f64 {
entry:
    %2 = f64.add %0, %1
    ret %2
}

export func main() -> i64 {
entry:
    %20 = i64.const 10
    %0 = ptr.func_addr @twice
    %1 = ptr.func_addr @square
    %2 = i64.const 7
    %3 = i64.call_ptr %0(%2)
    call @put_int(%3)
    %4 = i64.call @putchar(%20)
    %5 = i64.call_ptr %1(%2)
    call @put_int(%5)
    %6 = i64.call @putchar(%20)
    %7 = i64.const 0
    %8 = i64.gt %2, %7
    branch %8, pickone, picktwo
pickone:
    %9 = ptr.copy %1
    jump go
picktwo:
    %9 = ptr.copy %0
    jump go
go:
    %10 = i64.call_ptr %9(%2)
    call @put_int(%10)
    %11 = i64.call @putchar(%20)
    %12 = ptr.func_addr @addf
    %13 = f64.const 1.5
    %14 = f64.const 2.25
    %15 = f64.call_ptr %12(%13, %14)
    %16 = i64.ftoi %15
    call @put_int(%16)
    %17 = i64.call @putchar(%20)
    %18 = ptr.alloca 8
    ptr.store %0, %18
    %21 = ptr.load %18
    %22 = i64.call_ptr %21(%2)
    call @put_int(%22)
    %23 = i64.call @putchar(%20)
    ret %7
}""", externals=("put_int", "putchar")))

    def test_a_function_address_is_never_zero(self, tmp_path):
        """Zero is the null pointer, here as in memory. A function numbered
        from zero would be indistinguishable from one."""
        assert agrees(tmp_path, ir("""\
func target() -> i64 {
entry:
    %0 = i64.const 5
    ret %0
}

export func main() -> i64 {
entry:
    %0 = ptr.func_addr @target
    %1 = ptr.const 0
    %2 = ptr.ne %0, %1
    %3 = i64.extend %2
    call @put_int(%3)
    %4 = i64.const 0
    ret %4
}""")).strip() == "1"

    def test_integer_powers_are_correctly_rounded(self, tmp_path):
        """`Math.pow` is allowed a whole ulp and CPython's is not, so this is
        the double-double algorithm from the C runtime rather than a library
        call. `1.5682546124542256 ** 4` is the case that first showed the
        difference."""
        cases = [(2.0, 10), (1.5, 4), (10.0, 0), (0.1, 3), (2.0, -1),
                 (1.5682546124542256, 4), (3.0, 40), (0.0, 5), (-2.0, 3),
                 (-2.0, 4), (1e150, 2), (1e150, -2), (2.0, 1000),
                 (2.0, -1000), (7.0, -7), (1e300, -1), (1.0000001, 100)]
        body, r = [], 1
        for base, n in cases:
            body.append(f"    %{r} = f64.const {base!r}")
            body.append(f"    %{r + 1} = i64.const {n}")
            body.append(f"    %{r + 2} = f64.call @py_pow_int(%{r}, %{r + 1})")
            body.append(f"    call @put_float(%{r + 2})")
            body.append(f"    %{r + 3} = i64.call @putchar(%0)")
            r += 4
        source = ("export func main() -> i64 {\nentry:\n"
                  "    %0 = i64.const 10\n" + "\n".join(body)
                  + f"\n    %{r} = i64.const 0\n    ret %{r}\n}}")
        got = agrees(tmp_path, ir(
            source, externals=("py_pow_int", "put_float", "putchar")))
        # And the same numbers CPython prints, which is the actual contract.
        assert got.split() == [repr(base ** n) for base, n in cases]

    def test_a_function_too_large_for_one_jvm_method(self, tmp_path):
        """A JVM method holds 65535 bytes of bytecode. Past that the function
        is compiled as several, with its registers in an array and an entry
        method driving them -- so the answer must not change."""
        n = 9000
        body = ["    %0 = i64.const 0", "    %1 = i64.const 0"]
        for i in range(n):
            body.append(f"    %{i + 2} = i64.const {i}")
            body.append(f"    %0 = i64.add %0, %{i + 2}")
        body.append("    call @put_int(%0)")
        source = ("export func main() -> i64 {\nentry:\n" + "\n".join(body)
                  + "\n    ret %1\n}")
        assert agrees(tmp_path, ir(source)).strip() == str(n * (n - 1) // 2)

    def test_a_switch(self, tmp_path):
        agrees(tmp_path, ir("""\
export func main() -> i64 {
entry:
    %0 = i32.const 2
    i32.switch %0, default other [0 -> zero] [1 -> one] [2 -> two]
zero:
    %1 = i64.const 100
    call @put_int(%1)
    jump done
one:
    %2 = i64.const 200
    call @put_int(%2)
    jump done
two:
    %3 = i64.const 300
    call @put_int(%3)
    jump done
other:
    %4 = i64.const 999
    call @put_int(%4)
    jump done
done:
    %5 = i64.const 0
    ret %5
}"""))

    def test_a_returning_function_frees_its_alloca(self, tmp_path):
        """"Freed when the function returns" is what the IR promises, and the
        JVM backend keeps it by restoring a saved stack pointer.

        Two thousand calls of four kilobytes each is eight megabytes, against a
        one-megabyte alloca region -- so a frame that leaks its allocation does
        not merely waste space, it runs off the end of memory. The loop is in
        the caller rather than the callee because the reference interpreter
        recurses on the Python stack and would hit its own limit first."""
        agrees(tmp_path, ir("""\
func scratch(%0: i64) -> i64 {
entry:
    %1 = ptr.alloca 4096
    i64.store %0, %1
    %2 = i64.load %1
    ret %2
}

export func main() -> i64 {
entry:
    %0 = i64.const 0
    %1 = i64.const 0
    %2 = i64.const 2000
    %3 = i64.const 1
    jump head
head:
    %4 = i64.lt %1, %2
    branch %4, body, done
body:
    %5 = i64.call @scratch(%1)
    %0 = i64.add %0, %5
    %1 = i64.add %1, %3
    jump head
done:
    call @put_int(%0)
    %6 = i64.const 0
    ret %6
}"""))

    def test_recursion_unwinds_the_alloca_stack_too(self, tmp_path):
        agrees(tmp_path, ir("""\
func depth(%0: i64) -> i64 {
entry:
    %1 = ptr.alloca 256
    i64.store %0, %1
    %2 = i64.const 0
    %3 = i64.le %0, %2
    branch %3, base, recurse
base:
    %4 = i64.load %1
    ret %4
recurse:
    %5 = i64.const 1
    %6 = i64.sub %0, %5
    %7 = i64.call @depth(%6)
    %8 = i64.load %1
    %9 = i64.add %7, %8
    ret %9
}

export func main() -> i64 {
entry:
    %0 = i64.const 100
    %1 = i64.call @depth(%0)
    call @put_int(%1)
    %2 = i64.const 0
    ret %2
}"""))
