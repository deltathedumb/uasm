"""A third party's backend, target and toolchain, reached from the CLI.

The four registries always accepted an outside registration -- `register()`
is deliberately the same call the built-ins make. What did not work was
GETTING THERE: registration is a side effect of importing a module, and
nothing ever imported anyone else's, so `uasm build --backend mine`
answered `unknown backend 'mine'` for a backend whose registration was
perfectly correct. The extension point existed and could not be used, which
is worse than not having one, because the documentation says it works.

So the test that matters is this whole file's shape: a module written OUTSIDE
the package, loaded the way a stranger would load it, driving a build to
completion. Testing `register()` in-process would have passed the entire time
the feature was unusable.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from tests import harness
from tests.harness import snapshot

ROOT = Path(__file__).resolve().parents[3]
#: THE TREE THIS RUN IS MEASURING, not `src/` unconditionally: the runner
#: works from a snapshot so `src/` can be edited mid-run, and a subprocess
#: started here has to compile with the same code the rest of the run did.
#: See `tests/harness/snapshot.py`.
SRC = snapshot.current(ROOT)

#: A plugin registering one of everything. Deliberately not importing any
#: private module: if this needs an underscore name, the public surface is
#: incomplete and that is the finding.
PLUGIN = '''
from uasm.backend import Backend, register as reg_backend
from uasm.target import Target, register as reg_target
from uasm.link import Toolchain, register as reg_link


class CountingBackend(Backend):
    name = "counting"
    description = "counts instructions; emits a report"
    default_target = "counting-machine"
    self_contained = True

    def emit(self, module, target):
        n = sum(len(b.instructions) for f in module.defined_functions()
                for b in f.blocks)
        return {"report.txt": f"{n} instructions for {target.name}\\n".encode()}


class ReportToolchain(Toolchain):
    name = "counting-link"
    description = "writes the report and calls it a program"

    def supports(self, target):
        return target.arch == "counting"

    def link(self, request):
        request.workdir.mkdir(parents=True, exist_ok=True)
        for name, data in request.artifacts.items():
            (request.workdir / name).write_bytes(data)
        request.output.write_bytes(b"counted\\n")
        return request.output


reg_target(Target("counting-machine", arch="counting", os="linux",
                  abi="none", object_format="source"), aliases=("cm",))
reg_backend(CountingBackend())
reg_link(ReportToolchain())
'''

PROGRAM = "def main() -> int:\n    print(1 + 1)\n    return 0\n"


@harness.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "mypack.py").write_text(textwrap.dedent(PLUGIN),
                                        encoding="utf-8")
    (tmp_path / "prog.py").write_text(PROGRAM, encoding="utf-8")
    return tmp_path


def run(workspace: Path, *args: str, env_plugins: str = "") -> subprocess.CompletedProcess:
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), str(workspace)])
    if env_plugins:
        env["UASM_PLUGINS"] = env_plugins
    return subprocess.run([sys.executable, "-m", "uasm", *args],
                          capture_output=True, text=True, env=env, cwd=workspace)


class TestItIsReachableFromTheCommandLine:
    def test_without_a_plugin_it_is_unknown(self, workspace):
        """The baseline. If this ever passes, the test below proves nothing."""
        done = run(workspace, "plugin", "backends")
        assert "counting" not in done.stdout

    def test_the_flag_works_before_the_command(self, workspace):
        done = run(workspace, "plugin", "--plugin", "mypack", "backends")
        assert "counting" in done.stdout, done.stderr

    def test_the_flag_works_after_the_command(self, workspace):
        """Where people actually type it.

        argparse accepts a parser-level flag only BEFORE the subcommand, so
        the natural spelling needs the subparsers to take it too -- and with
        a shared `dest` the subparser silently OVERWRITES the global list
        rather than appending, which loses plugins named in both positions.
        """
        done = run(workspace, "build", "prog.py", "--plugin", "mypack",
                   "--backend", "counting", "--emit", "-o", "out.txt")
        assert done.returncode == 0, done.stderr
        assert "9 instructions" in (workspace / "out.txt").read_text()

    def test_both_positions_at_once(self, workspace):
        (workspace / "second.py").write_text(
            "from uasm.target import Target, register\n"
            "register(Target('second-machine', arch='second'))\n",
            encoding="utf-8")
        done = run(workspace, "--plugin", "mypack", "plugin", "targets",
                   "--plugin", "second")
        assert "counting-machine" in done.stdout, done.stderr
        assert "second-machine" in done.stdout, done.stderr

    def test_the_environment_variable_works(self, workspace):
        done = run(workspace, "plugin", "targets", env_plugins="mypack")
        assert "counting-machine" in done.stdout, done.stderr

    def test_a_plugin_loads_only_once(self, workspace):
        """Registries refuse a duplicate backend name, so this is not tidiness.

        Naming the same module twice -- easy, when a flag and the environment
        variable disagree -- would otherwise be a crash.
        """
        done = run(workspace, "--plugin", "mypack", "plugin", "backends",
                   "--plugin", "mypack", env_plugins="mypack")
        assert done.returncode == 0, done.stderr
        assert done.stdout.count("counting") == 1


class TestItBuildsWithThirdPartyPartsOnly:
    def test_backend_target_and_toolchain_together(self, workspace):
        """`--link` BECAUSE `build` STOPS AT AN UNLINKED OBJECT.

        A third party's toolchain is only exercised once something asks for
        a program, so without the flag `counting-link` would never be called
        and `out.bin` would hold the backend's artifact instead.
        """
        done = run(workspace, "build", "prog.py", "--plugin", "mypack",
                   "--backend", "counting", "--toolchain", "counting-link",
                   "--link", "-o", "out.bin")
        assert done.returncode == 0, done.stderr + done.stdout
        assert (workspace / "out.bin").read_bytes() == b"counted\n"

    def test_a_target_alias_resolves(self, workspace):
        done = run(workspace, "build", "prog.py", "--plugin", "mypack",
                   "--backend", "counting", "--target", "cm", "--emit",
                   "-o", "out.txt")
        assert done.returncode == 0, done.stderr
        assert "counting-machine" in (workspace / "out.txt").read_text()


class TestFailuresAreReported:
    def test_a_missing_plugin_is_an_error(self, workspace):
        """Not a traceback, and not silence.

        Carrying on without it produces `unknown backend` for a backend the
        user is looking at in their own file, which sends them to debug the
        wrong thing entirely.
        """
        done = run(workspace, "plugin", "--plugin", "nosuchthing", "backends")
        assert done.returncode == 2
        assert "nosuchthing" in done.stderr
        assert "Traceback" not in done.stderr

    def test_a_plugin_that_raises_names_itself(self, workspace):
        (workspace / "broken.py").write_text("raise ValueError('boom')\n",
                                             encoding="utf-8")
        done = run(workspace, "plugin", "--plugin", "broken", "backends")
        assert done.returncode == 2
        assert "broken" in done.stderr and "boom" in done.stderr


class TestTheListingsSurviveAThirdParty:
    def test_a_long_name_does_not_break_alignment(self, workspace):
        """`counting-link` is longer than every built-in toolchain name.

        The column width was fixed at ten, which is right for exactly the
        names that shipped -- the one class of bug an extension point has
        that nobody in-tree can see.

        THE COMPONENT ROWS AND NOT EVERY LINE. A toolchain that declares
        flags lists them under its own line, indented past the name column
        -- `--link-input` is one -- and those lines are deliberately not in
        this column. They are told apart by their indent: a component row
        begins at column two and nothing else does.
        """
        done = run(workspace, "plugin", "--plugin", "mypack", "linkers")
        assert done.returncode == 0, done.stderr
        rows = [l for l in done.stdout.splitlines()
                if l.strip() and l.startswith("  ") and not l.startswith("   ")]
        assert len(rows) > 1, f"no component rows at all: {done.stdout}"
        starts = {l.index(l.strip().split()[1]) for l in rows
                  if len(l.strip().split()) > 1}
        assert len(starts) == 1, f"descriptions are not aligned: {done.stdout}"
