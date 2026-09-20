"""Java packages, as modules a Python program imports.

    import com.minecraft.block as block
    import jvm.com.minecraft.block

    my_block = block.Block()
    my_block.setName("granite")

THE API IS BUILT WITH `javac`, into a real jar. That is the point: the class
path reader is exercised against what a Java toolchain actually emits, rather
than against a class this compiler wrote and would read back by construction.
Nothing in uasm calls `javac` -- only these tests do, to have something
worth importing.

And every test RUNS the result. A class path that type-checks and then fails to
load is exactly the failure this exists to catch, and nothing short of a JVM
finds it.
"""
from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

from tests import harness

from .test_jvm import JAVA_FLAGS, run_cli

#: A two-class API with the shapes that matter: two constructors, instance
#: methods over `int`, `double` and `String`, a static method, a static
#: factory returning an object, and a method taking one.
API_SOURCE = {
    "com/minecraft/block/Block.java": """\
package com.minecraft.block;
public class Block {
    private String name = "air";
    private int hardness = 1;
    public Block() {}
    public Block(int hardness) { this.hardness = hardness; }
    public void setName(String name) { this.name = name; }
    public String getName() { return name; }
    public int getHardness() { return hardness; }
    public void setHardness(int h) { this.hardness = h; }
    public double scaled(double by) { return hardness * by; }
    public static int count() { return 42; }
    public String toString() { return "Block(" + name + "," + hardness + ")"; }
}
""",
    "com/minecraft/level/Level.java": """\
package com.minecraft.level;
import com.minecraft.block.Block;
public class Level {
    private int placed = 0;
    public Level() {}
    public void place(Block b, int x, int y, int z) {
        placed++;
        System.out.println("placed " + b + " at " + x + "," + y + "," + z);
    }
    public int placedCount() { return placed; }
    public static Level create() { return new Level(); }
}
""",
}


@harness.fixture
def api_jar(tmp_path) -> Path:
    src = tmp_path / "api"
    for name, text in API_SOURCE.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    classes = tmp_path / "classes"
    built = subprocess.run(
        ["javac", "-d", str(classes)] + [str(src / n) for n in API_SOURCE],
        capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    jar = tmp_path / "api.jar"
    with zipfile.ZipFile(jar, "w") as archive:
        for cls in sorted(classes.rglob("*.class")):
            archive.write(cls, str(cls.relative_to(classes)).replace("\\", "/"))
    return jar


@harness.needs("java", "javac")
class TestImportingJava:
    def compile(self, tmp_path, api_jar, source, *extra):
        path = tmp_path / "mod.py"
        path.write_text(source, encoding="utf-8")
        jar = tmp_path / "mod.jar"
        # `-o` with `--emit-ir` writes the IR TO THAT FILE rather than to
        # stdout, so a test reading the IR must not ask for both.
        #
        # `--link` BECAUSE `build` STOPS AT AN UNLINKED OBJECT, and what
        # every test on this path checks is a program a JVM loads and runs
        # -- `java -cp mod.jar Mod` needs the packaged jar, not the artifact
        # the build would otherwise leave behind. The flag belongs with `-o`
        # and not in the `--emit-ir` branch: a caller asking for the IR is
        # asking the build to stop at it, and there is nothing to link.
        output = [] if "--emit-ir" in extra else ["-o", str(jar), "--link"]
        r = run_cli("build", str(path), "--backend", "jvm",
                    "--java-version", "21", "--classpath", str(api_jar),
                    *output, *extra)
        return r, jar

    def run(self, tmp_path, api_jar, source) -> str:
        r, jar = self.compile(tmp_path, api_jar, source)
        assert r.returncode == 0, r.stdout + r.stderr
        entry = os.pathsep.join([str(jar), str(api_jar)])
        got = subprocess.run(["java", *JAVA_FLAGS, "-cp", entry, "Mod"],
                             capture_output=True, text=True)
        assert not got.returncode, got.stderr
        return got.stdout

    def test_the_shape_the_feature_was_asked_for(self, tmp_path, api_jar):
        assert self.run(tmp_path, api_jar, """\
import com.minecraft.block as block

def main() -> int:
    my_block = block.Block()
    my_block.setHardness(7)
    print(my_block.getHardness())
    return 0
""").split() == ["7"]

    def test_the_prefixed_form_always_works(self, tmp_path, api_jar):
        """`jvm.<name>` is not a fallback for a collision. It is the real name,
        and it resolves whether or not anything else wants the bare one."""
        assert self.run(tmp_path, api_jar, """\
import jvm.com.minecraft.block

def main() -> int:
    b = jvm.com.minecraft.block.Block(3)
    print(b.getHardness())
    return 0
""").split() == ["3"]

    def test_both_spellings_reach_the_same_class(self, tmp_path, api_jar):
        assert self.run(tmp_path, api_jar, """\
import com.minecraft.block as bare
import jvm.com.minecraft.block

def main() -> int:
    a = bare.Block(5)
    b = jvm.com.minecraft.block.Block(5)
    print(a.getHardness() + b.getHardness())
    return 0
""").split() == ["10"]

    def test_constructors_statics_and_objects_as_arguments(self, tmp_path,
                                                           api_jar):
        assert self.run(tmp_path, api_jar, """\
import com.minecraft.block as block
import jvm.com.minecraft.level

def main() -> int:
    stone = block.Block()
    stone.setHardness(7)
    stone.setName("granite")
    print(block.Block.count())
    world = jvm.com.minecraft.level.Level.create()
    for x in range(2):
        world.place(stone, x, 64, 0)
    print(world.placedCount())
    return 0
""").splitlines() == ["42",
                      "placed Block(granite,7) at 0,64,0",
                      "placed Block(granite,7) at 1,64,0",
                      "2"]

    def test_a_float_crosses_in_both_directions(self, tmp_path, api_jar):
        assert self.run(tmp_path, api_jar, """\
import com.minecraft.block as block

def main() -> int:
    print(block.Block(4).scaled(2.5))
    return 0
""").split() == ["10.0"]

    def test_an_overload_is_chosen_by_the_arguments(self, tmp_path, api_jar):
        assert self.run(tmp_path, api_jar, """\
import com.minecraft.block as block

def main() -> int:
    print(block.Block().getHardness())
    print(block.Block(9).getHardness())
    return 0
""").split() == ["1", "9"]

    def test_a_namespace_import_costs_no_object_runtime(self, tmp_path,
                                                        api_jar):
        """An `import` of a Java package emits nothing, so it must not make the
        module body the entry -- a module body is dynamic, and a dynamic
        program needs the `apy_*` runtime the JVM backend does not have."""
        r, _ = self.compile(tmp_path, api_jar, """\
import com.minecraft.block as block

def main() -> int:
    print(block.Block.count())
    return 0
""", "--emit-ir")
        assert r.returncode == 0, r.stderr
        assert "apy_" not in r.stdout

    def test_ir_text_names_the_java_call_so_it_can_be_rebuilt(self, tmp_path,
                                                              api_jar):
        r, _ = self.compile(tmp_path, api_jar, """\
import com.minecraft.block as block

def main() -> int:
    print(block.Block(6).getHardness())
    return 0
""", "--emit-ir")
        assert r.returncode == 0, r.stderr
        assert "jvm$new$com.minecraft.block.Block" in r.stdout

    @harness.cases("bad,message", [
        ("b = block.Block()\n    b.noSuchMethod()", "has no method"),
        ("b = block.Block(1, 2, 3)", "argument"),
        ("b = block.Nothing()", "no member"),
    ])
    def test_a_call_the_class_path_lacks_is_a_diagnostic(self, tmp_path,
                                                         api_jar, bad, message):
        """Refused against the jar at compile time, rather than as a
        NoSuchMethodError on the machine that runs it."""
        r, _ = self.compile(tmp_path, api_jar, f"""\
import com.minecraft.block as block

def main() -> int:
    {bad}
    return 0
""", "--emit-ir")
        assert r.returncode != 0
        assert "Traceback" not in r.stderr
        assert message in (r.stdout + r.stderr)

    def test_a_java_type_can_be_written_as_an_annotation(self, tmp_path,
                                                         api_jar):
        """Without this a handle could be made and used inside one function and
        never leave it, because there was no way to spell the parameter that
        would receive it."""
        assert self.run(tmp_path, api_jar, """\
import com.minecraft.block as block
import jvm.com.minecraft.level


def make(hardness: int) -> block.Block:
    made = block.Block(hardness)
    made.setName("granite")
    return made


def place_it(world: jvm.com.minecraft.level.Level, b: block.Block) -> None:
    world.place(b, 0, 64, 0)


def main() -> int:
    stone = make(7)
    print(stone.getHardness())
    world = jvm.com.minecraft.level.Level.create()
    place_it(world, stone)
    print(world.placedCount())
    return 0
""").splitlines() == ["7", "placed Block(granite,7) at 0,64,0", "1"]

    def test_a_java_annotation_keeps_the_function_static(self, tmp_path,
                                                         api_jar):
        """An annotation the static path cannot represent makes a function
        dynamic -- which for a Java type would have made every parameter an
        `object`, a representation this backend has no runtime for."""
        r, _ = self.compile(tmp_path, api_jar, """\
import com.minecraft.block as block


def hardness_of(b: block.Block) -> int:
    return b.getHardness()


def main() -> int:
    print(hardness_of(block.Block(2)))
    return 0
""", "--emit-ir")
        assert r.returncode == 0, r.stderr
        assert "apy_" not in r.stdout
