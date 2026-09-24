"""A bundled module that imports another bundled module INSIDE A FUNCTION.

The program's side has spliced `def f(): import heapq` for a long time. A
bundled module's own body did not, and in two ways at once:

  * `_dependencies` read only the module's TOP-LEVEL statements, so a module
    imported nowhere else was never spliced at all;
  * and when it was spliced for some other reason, the nested statement
    still bound a LOCAL, which `_bound_locally` reports faithfully -- so
    `lib_a.g()` inside the function was the function's own business and was
    never repointed at the spliced definition.

argparse is why it matters: it imports `copy`, `textwrap` and `warnings`
where they are used rather than at the top, to keep its own import cheap.

THE LIBRARY HERE IS A DIRECTORY OF TWO FILES standing in for `bundled/`,
because the real one has no module doing this yet, and a test that waits
for argparse to exist is a test of argparse.
"""
from __future__ import annotations

import textwrap
from io import StringIO

from tests import harness

from uasm.diagnostics import DiagnosticSink
from uasm.driver import Options, compile_source
from uasm.frontends.python import bundled
from uasm.ir.interpreter import Interpreter

LIB_A = '''
TABLE = {"k": "from the table"}


def g():
    return "a.g"
'''

LIB_B = '''
def through_the_module():
    import lib_a
    return lib_a.g()


def through_a_from_import():
    from lib_a import g as renamed
    return renamed() + "!"


class Holder:
    def method(self):
        import lib_a
        return lib_a.TABLE["k"]


def nothing_but_the_import():
    import lib_a


def in_a_nested_block(flag):
    if flag:
        import lib_a
        return lib_a.g().upper()
    return "not taken"
'''


@harness.fixture
def library(tmp_path):
    """`bundled/` pointed at a directory holding `lib_a` and `lib_b`, for as
    long as the test runs -- `available()`, `_dependencies` and `splice` all
    read the directory through `bundled._HERE`."""
    lib = tmp_path / "bundled"
    lib.mkdir()
    (lib / "lib_a.py").write_text(LIB_A, encoding="utf-8")
    (lib / "lib_b.py").write_text(LIB_B, encoding="utf-8")
    saved = bundled._HERE
    bundled._HERE = lib
    yield tmp_path
    bundled._HERE = saved


def _run(where, program: str) -> list[str]:
    path = where / "prog.py"
    path.write_text(textwrap.dedent(program).lstrip(), encoding="utf-8")
    sink = DiagnosticSink()
    result = compile_source(Options(source=path), sink)
    assert result.ok, [d.message for d in sink.diagnostics]
    out = StringIO()
    Interpreter(result.module, out=out).run("main")
    return out.getvalue().split("\n")[:-1]


class TestANestedImportIsTheSameImport:
    def test_the_module_it_names_is_spliced(self, library):
        """Found at ANY depth: the only import of `lib_a` anywhere is inside
        `lib_b`'s functions, and it has to come first in the prelude."""
        order = bundled._dependencies(["lib_b"], bundled.available())
        assert order == ["lib_a", "lib_b"]

    def test_every_spelling_reaches_the_spliced_definition(self, library):
        assert _run(library, """
            import lib_b
            print(lib_b.through_the_module())
            print(lib_b.through_a_from_import())
            print(lib_b.Holder().method())
            print(lib_b.nothing_but_the_import())
            print(lib_b.in_a_nested_block(True))
            print(lib_b.in_a_nested_block(False))
        """) == ["a.g", "a.g!", "from the table", "None", "A.G",
                 "not taken"]
