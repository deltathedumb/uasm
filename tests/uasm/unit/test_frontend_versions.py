"""Language-version selection belongs to each frontend."""
from __future__ import annotations

from uasm.diagnostics import DiagnosticSink, SourceFile
from uasm.frontend import BuildContext
from uasm.frontends.c import CFrontend
from uasm.frontends.python import PythonFrontend
from uasm.options import OptionError


class TestLanguageVersions:
    def test_python_declares_and_accepts_314(self):
        frontend = PythonFrontend().configure(
            {"language-version": "3.14"},
            BuildContext(source=SourceFile("x.py", "").path),
            DiagnosticSink())
        assert frontend.language_version == "3.14"
        assert frontend.default_language_version == "3.14"

    def test_python_rejects_an_unimplemented_release(self):
        try:
            PythonFrontend().configure(
                {"language-version": "3.13"},
                BuildContext(source=SourceFile("x.py", "").path),
                DiagnosticSink())
        except OptionError as exc:
            assert "3.13" in str(exc)
        else:
            raise AssertionError("expected OptionError")

    def test_c_declares_and_accepts_c23(self):
        frontend = CFrontend().configure(
            {"language-version": "C23"},
            None, DiagnosticSink())
        assert frontend.language_version == "c23"
        assert frontend.default_language_version == "c23"

    def test_c_rejects_an_unimplemented_standard(self):
        try:
            CFrontend().configure({"language-version": "c17"}, None,
                                  DiagnosticSink())
        except OptionError as exc:
            assert "c17" in str(exc)
        else:
            raise AssertionError("expected OptionError")
