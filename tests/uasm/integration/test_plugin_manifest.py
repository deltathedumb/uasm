"""`uasm plugin add`, and the manifest a module declares.

The shape the user writes:

    from uasm.plugins import Plugin, Backend, Target, Frontend, Linker

    plugin = Plugin("mypack")
    plugin.backends.append(MyBackend())
    __uasm_plugin__ = plugin

then `uasm plugin add my_plugin_module`, and from then on the backend is
there with no flag at all.

Everything here runs the real CLI in a subprocess against a real config file
in a temporary directory. An in-process test of `install()` would pass while
the persisted half was broken, and persistence IS the feature -- `add` that
does not outlive the process is a flag with extra steps.
"""
from __future__ import annotations

import json
import os
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

PLUGIN = '''
from uasm.plugins import Plugin, Backend, Target, Frontend, Linker


class MyBackend(Backend):
    name = "my-backend"
    description = "demo backend"
    default_target = "my-machine"
    self_contained = True

    def emit(self, module, target):
        return {"out.txt": f"built for {target.name}\\n".encode()}


class MyLinker(Linker):
    name = "my-linker"
    description = "demo linker"

    def supports(self, target):
        return True

    def link(self, request):
        request.output.write_bytes(b"linked\\n")
        return request.output


plugin = Plugin("mypack", version="1.0", description="a demo plugin")
plugin.backends.append(MyBackend())
plugin.linkers.append(MyLinker())
plugin.add_target(Target("my-machine", arch="my", os="linux"), aliases=("mm",))

__uasm_plugin__ = plugin
'''

PROGRAM = "def main() -> int:\n    print(1)\n    return 0\n"


@harness.fixture
def ws(tmp_path: Path) -> Path:
    (tmp_path / "my_plugin_module.py").write_text(textwrap.dedent(PLUGIN),
                                                  encoding="utf-8")
    (tmp_path / "prog.py").write_text(PROGRAM, encoding="utf-8")
    (tmp_path / "cfg").mkdir()
    return tmp_path


def run(ws: Path, *args: str, path_extra: str = "") -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC)] + ([path_extra] if path_extra else []))
    env["UASM_CONFIG_DIR"] = str(ws / "cfg")
    env.pop("UASM_PLUGINS", None)
    return subprocess.run([sys.executable, "-m", "uasm", *args],
                          capture_output=True, text=True, env=env, cwd=ws)


class TestTheManifest:
    def test_show_lists_contents_without_registering(self, ws):
        """`show` must not have the side effect it is describing.

        This is the whole reason a manifest exists: with import-time
        `register()` there is no way to ask what a module provides without
        letting it change the compiler.
        """
        done = run(ws, "plugin", "show", "my_plugin_module")
        assert done.returncode == 0, done.stderr
        for expected in ("my-backend", "my-machine", "my-linker", "mypack 1.0"):
            assert expected in done.stdout
        # Nothing was installed by showing.
        assert "my-backend" not in run(ws, "plugin", "backends").stdout

    def test_a_module_without_a_manifest_still_works(self, ws):
        """The older style -- register() at import -- is not broken by this."""
        (ws / "old_style.py").write_text(
            "from uasm.target import Target, register\n"
            "register(Target('old-machine', arch='old'))\n", encoding="utf-8")
        done = run(ws, "plugin", "show", "old_style")
        assert done.returncode == 0, done.stderr
        assert "registers on import" in done.stdout
        assert "old-machine" in run(ws, "plugin", "--plugin", "old_style", "targets").stdout


class TestAddIsRemembered:
    def test_add_then_no_flag_at_all(self, ws):
        assert run(ws, "plugin", "add", "my_plugin_module").returncode == 0
        listing = run(ws, "plugin", "backends")
        assert "my-backend" in listing.stdout, listing.stderr

    def test_it_survives_into_a_build(self, ws):
        run(ws, "plugin", "add", "my_plugin_module")
        done = run(ws, "build", "prog.py", "--backend", "my-backend",
                   "--toolchain", "my-linker", "--link", "-o", "out.bin")
        assert done.returncode == 0, done.stderr + done.stdout
        assert (ws / "out.bin").read_bytes() == b"linked\n"

    def test_the_alias_is_installed_too(self, ws):
        run(ws, "plugin", "add", "my_plugin_module")
        done = run(ws, "build", "prog.py", "--backend", "my-backend",
                   "--target", "mm", "--emit", "-o", "out.txt")
        assert done.returncode == 0, done.stderr
        assert "my-machine" in (ws / "out.txt").read_text()

    def test_list_says_what_it_provides(self, ws):
        run(ws, "plugin", "add", "my_plugin_module")
        done = run(ws, "plugin", "list")
        assert "my_plugin_module" in done.stdout
        assert "my-backend" in done.stdout

    def test_remove_undoes_it(self, ws):
        run(ws, "plugin", "add", "my_plugin_module")
        assert run(ws, "plugin", "remove", "my_plugin_module").returncode == 0
        assert "my-backend" not in run(ws, "plugin", "backends").stdout
        assert "no plugins installed" in run(ws, "plugin", "list").stdout

    def test_removing_something_absent_is_an_error(self, ws):
        done = run(ws, "plugin", "remove", "nothing")
        assert done.returncode == 2 and "not installed" in done.stderr

    def test_adding_twice_does_not_duplicate(self, ws):
        """Registries refuse a duplicate name, so this would be a crash.

        Re-adding now REPLACES rather than adding again, and asks first --
        `--yes` is the non-interactive answer.
        """
        run(ws, "plugin", "add", "my_plugin_module")
        second = run(ws, "plugin", "add", "my_plugin_module", "--yes")
        assert second.returncode == 0, second.stderr
        stored = json.loads((ws / "cfg" / "plugins.json").read_text())
        assert len(stored["plugins"]) == 1

    def test_the_config_records_how_it_was_resolved(self, ws):
        """A cwd plugin must not later resolve to a same-named package.

        Without the recorded sources, `plugin add ./thing.py` today and a
        PyPI package called `thing` tomorrow silently become the same name.
        """
        run(ws, "plugin", "add", "my_plugin_module")
        stored = json.loads((ws / "cfg" / "plugins.json").read_text())
        entry = stored["plugins"][0]
        assert entry["source"] == "cwd"
        assert entry["sources"] == {"cwd": True, "pypath": True, "pip": False}


class TestTheSourceSwitches:
    def test_cwd_1_finds_a_file_beside_you(self, ws):
        done = run(ws, "plugin", "show", "my_plugin_module", "--cwd", "1")
        assert done.returncode == 0 and "(cwd:" in done.stdout

    def test_cwd_0_really_excludes_the_directory(self, ws):
        """`python -m` puts the working directory on sys.path.

        So `--cwd 0` that only skips the file search still found the file,
        reported it as `pypath`, and did exactly what the user said not to.
        """
        done = run(ws, "plugin", "show", "my_plugin_module", "--cwd", "0")
        assert done.returncode == 2
        assert "cannot find plugin" in done.stderr

    def test_pypath_0_excludes_an_installed_module(self, ws):
        done = run(ws, "plugin", "show", "json", "--cwd", "0", "--pypath", "0")
        assert done.returncode == 2

    def test_no_sources_says_so(self, ws):
        done = run(ws, "plugin", "show", "my_plugin_module",
                   "--cwd", "0", "--pypath", "0", "--pip", "0")
        assert done.returncode == 2
        assert "no sources were enabled" in done.stderr

    def test_pip_is_off_by_default(self, ws):
        """Adding a name must not reach the network on its own.

        The check is that a nonexistent name fails FAST and suggests the
        flag, rather than attempting an install nobody asked for.
        """
        done = run(ws, "plugin", "add", "uasm-no-such-plugin-xyz")
        assert done.returncode == 2
        assert "--pip 1" in done.stderr
        assert "pip install failed" not in done.stderr


class TestBrokenPluginsDoNotTrapYou:
    def test_an_installed_plugin_that_vanishes_is_a_warning(self, ws):
        """And crucially, not fatal.

        If loading an installed plugin were fatal, deleting its file would
        break every uasm command -- including `plugin remove`, the only
        one that fixes it.
        """
        run(ws, "plugin", "add", "my_plugin_module")
        # Deleting the SOURCE is no longer enough to break it -- that is what
        # the cache is for -- so the cache has to go too for this to be the
        # "installed but unloadable" state the warning exists for.
        (ws / "my_plugin_module.py").unlink()
        import shutil
        shutil.rmtree(ws / "cfg" / "cache", ignore_errors=True)
        done = run(ws, "plugin", "backends")
        assert done.returncode == 0, done.stderr
        assert "warning" in done.stderr and "my_plugin_module" in done.stderr
        assert "c" in done.stdout          # built-ins still work

    def test_and_it_can_still_be_removed(self, ws):
        run(ws, "plugin", "add", "my_plugin_module")
        (ws / "my_plugin_module.py").unlink()
        done = run(ws, "plugin", "remove", "my_plugin_module")
        assert done.returncode == 0, done.stderr

    def test_a_plugin_that_raises_is_named(self, ws):
        (ws / "boom.py").write_text("raise ValueError('nope')\n",
                                    encoding="utf-8")
        done = run(ws, "plugin", "add", "boom")
        assert done.returncode != 0
        assert "boom" in done.stderr and "Traceback" not in done.stderr

    def test_a_broken_plugin_is_not_recorded(self, ws):
        """Loaded before it is written down.

        The other order installs a plugin that cannot load, which is the
        trap the two tests above exist to prevent.
        """
        (ws / "boom.py").write_text("raise ValueError('nope')\n",
                                    encoding="utf-8")
        run(ws, "plugin", "add", "boom")
        assert "boom" not in run(ws, "plugin", "list").stdout


PATCHING_PLUGIN = '''
from uasm.plugins import Plugin, CompilerPatch

def note(original, module, show_spans=False):
    return "; patched" + chr(10) + original(module, show_spans=show_spans)

plugin = Plugin("patcher", description="rewrites printed IR")
# NOT uasm.ir.printer.print_module: pipeline.py did
# `from ..ir import print_module`, so it holds the function directly and
# would never see a patch to the printer's own module attribute. Patch
# what the CALLER reaches. See patch.py's docstring.
plugin.patches.append(CompilerPatch("uasm.driver.pipeline.print_module",
                                    wrap=note, reason="mark the output"))

__uasm_plugin__ = plugin
'''

SEALED_PLUGIN = '''
from uasm.plugins import Plugin, CompilerPatch

plugin = Plugin("bad")
plugin.patches.append(CompilerPatch("uasm.plugins.store.write",
                                    replace=lambda *a, **k: None))

__uasm_plugin__ = plugin
'''


class TestPatchesArriveThroughAPlugin:
    def test_a_patch_takes_effect_after_add(self, ws):
        """The whole point: installed once, it changes the compiler."""
        (ws / "patcher.py").write_text(textwrap.dedent(PATCHING_PLUGIN),
                                       encoding="utf-8")
        assert run(ws, "plugin", "add", "patcher").returncode == 0
        done = run(ws, "build", "prog.py", "--emit-ir")
        assert done.returncode == 0, done.stderr
        assert done.stdout.startswith("; patched")

    def test_show_lists_the_patch_without_applying_it(self, ws):
        (ws / "patcher.py").write_text(textwrap.dedent(PATCHING_PLUGIN),
                                       encoding="utf-8")
        done = run(ws, "plugin", "show", "patcher")
        assert done.returncode == 0, done.stderr
        assert "print_module" in done.stdout and "mark the output" in done.stdout
        # Showing did not patch anything: a plain build is untouched.
        assert not run(ws, "build", "prog.py", "--emit-ir").stdout.startswith(
            "; patched")

    def test_a_sealed_patch_stops_the_install(self, ws):
        """And leaves nothing installed, so the next run is clean."""
        (ws / "sealed.py").write_text(textwrap.dedent(SEALED_PLUGIN),
                                      encoding="utf-8")
        done = run(ws, "plugin", "add", "sealed")
        assert done.returncode != 0
        assert "sealed" in done.stderr
        # Checked against the JSON, not the listing: the temporary directory
        # is itself named after this test and contains the word "sealed", so
        # a substring check on the output passes for the wrong reason.
        assert not json.loads(
            (ws / "cfg" / "plugins.json").read_text())["plugins"] \
            if (ws / "cfg" / "plugins.json").exists() else True

    def test_the_from_import_caveat_is_real(self, ws):
        """Patching the definition does not reach a caller that imported it.

        Documented in patch.py and asserted here, because it is the reason a
        patch appears to do nothing and there is no way to fix it from this
        side -- it is how names work in Python. The test exists so the
        docstring cannot quietly stop being true.
        """
        (ws / "wrong.py").write_text(textwrap.dedent('''
            from uasm.plugins import Plugin, CompilerPatch

            def note(original, module, show_spans=False):
                return "; patched" + chr(10) + original(module, show_spans=show_spans)

            plugin = Plugin("wrong")
            plugin.patches.append(CompilerPatch(
                "uasm.ir.printer.print_module", wrap=note))

            __uasm_plugin__ = plugin
        '''), encoding="utf-8")
        assert run(ws, "plugin", "add", "wrong").returncode == 0
        done = run(ws, "build", "prog.py", "--emit-ir")
        assert done.returncode == 0, done.stderr
        assert not done.stdout.startswith("; patched")


class TestAddIsCachedAndReplaceable:
    """`add` caches, and re-adding an installed id asks first."""

    def test_the_plugin_is_copied_into_a_cache(self, ws):
        run(ws, "plugin", "add", "my_plugin_module")
        cached = list((ws / "cfg" / "cache").rglob("*.py"))
        assert cached, "nothing was cached"

    def test_it_loads_after_the_source_is_deleted(self, ws):
        """The whole point of caching.

        An install that stops working because the file it came from moved is
        not an install, it is a bookmark.
        """
        run(ws, "plugin", "add", "my_plugin_module")
        (ws / "my_plugin_module.py").unlink()
        done = run(ws, "plugin", "backends")
        assert done.returncode == 0, done.stderr
        assert "my-backend" in done.stdout

    def test_re_adding_refuses_when_not_interactive(self, ws):
        """Refuses rather than PROMPTING with no tty.

        A build that blocks forever on a hidden question is worse than one
        that stops and names the flag.
        """
        run(ws, "plugin", "add", "my_plugin_module")
        done = run(ws, "plugin", "add", "my_plugin_module")
        assert done.returncode == 1
        assert "--yes" in done.stderr
        assert "cancelled" in done.stdout

    def test_no_declines(self, ws):
        run(ws, "plugin", "add", "my_plugin_module")
        done = run(ws, "plugin", "add", "my_plugin_module", "--no")
        assert done.returncode == 1 and "cancelled" in done.stdout

    def test_yes_replaces(self, ws):
        run(ws, "plugin", "add", "my_plugin_module")
        done = run(ws, "plugin", "add", "my_plugin_module", "--yes")
        assert done.returncode == 0, done.stderr
        stored = json.loads((ws / "cfg" / "plugins.json").read_text())
        assert len(stored["plugins"]) == 1


class TestInvalidate:
    def _edit(self, ws):
        src = ws / "my_plugin_module.py"
        src.write_text(src.read_text(encoding="utf-8")
                       .replace('"my-backend"', '"my-backend-v2"'),
                       encoding="utf-8")

    def test_it_picks_up_an_edited_source(self, ws):
        run(ws, "plugin", "add", "my_plugin_module")
        self._edit(ws)
        # Until invalidated, the CACHED copy is what loads.
        assert "my-backend-v2" not in run(ws, "plugin", "backends").stdout
        done = run(ws, "plugin", "invalidate", "my_plugin_module")
        assert done.returncode == 0, done.stderr
        assert "my-backend-v2" in run(ws, "plugin", "backends").stdout

    @harness.cases("form", [
        ["my_plugin_module"],                      # bare
        ["my_plugin_module,second"],               # comma-separated
        ["my_plugin_module", "second"],            # repeated
        ["--all"],
    ])
    def test_every_input_form(self, ws, form):
        (ws / "second.py").write_text(textwrap.dedent("""
            from uasm.plugins import Plugin, Target
            plugin = Plugin('second')
            plugin.targets.append(Target('second-machine', arch='second'))
            __uasm_plugin__ = plugin
        """), encoding="utf-8")
        run(ws, "plugin", "add", "my_plugin_module")
        run(ws, "plugin", "add", "second")
        done = run(ws, "plugin", "invalidate", *form)
        assert done.returncode == 0, done.stderr
        assert "invalidated" in done.stdout

    def test_a_vanished_source_is_a_clean_error(self, ws):
        """Not a traceback -- and the cached copy must survive it.

        `resolve` finds the module in sys.modules if it was already loaded
        this process, so re-resolving has to drop that record first or a
        deleted source reports success.
        """
        run(ws, "plugin", "add", "my_plugin_module")
        (ws / "my_plugin_module.py").unlink()
        done = run(ws, "plugin", "invalidate", "my_plugin_module")
        assert done.returncode == 1
        assert "Traceback" not in done.stderr
        # Still loadable from cache afterwards.
        assert "my-backend" in run(ws, "plugin", "backends").stdout

    def test_an_unknown_id_is_reported(self, ws):
        done = run(ws, "plugin", "invalidate", "nosuch")
        assert done.returncode == 1
        assert "not installed" in done.stderr

    def test_invalidate_with_nothing_installed(self, ws):
        done = run(ws, "plugin", "invalidate", "--all")
        assert done.returncode == 0
        assert "no plugins installed" in done.stdout
