"""What each backend and linker writes, declared so the driver can read it.

THE RULE, AND WHY IT IS A TEST RATHER THAN A CONVENTION. A component's
`artifacts` is what lets `-o thing.wasm` choose the wasm backend without the
user naming it. A component that forgets to declare one is not broken and
says nothing: it compiles, it emits, it links -- it simply becomes
unreachable by spelling, and the only symptom is that the driver picks
something else and builds the wrong thing. Nothing else in the suite can see
that, because every other test names its backend explicitly.

SO THE DECLARATION IS CHECKED AGAINST WHAT THE COMPONENT REALLY WRITES rather
than against a list kept here. A list here would be a second place to forget.

EMPTY IS A REAL ANSWER and not an omission, which is why it is spelled out
per component below: a native executable on a Unix has no extension, and the
`none` toolchain writes whatever the backend called its files. Both are
honest and neither may be chosen by spelling, so both are listed as the
exceptions they are rather than tolerated by a blanket rule.
"""
from __future__ import annotations

from tests import harness

from uasm import backend as backend_registry
from uasm import link as link_registry

backend_registry.load_builtin()
link_registry.load_builtin()

#: THE COMPONENTS WITH NOTHING TO CLAIM, each for a stated reason. Anything
#: else declaring no artifacts is the omission this file exists to catch.
NO_ARTIFACTS_BY_DESIGN = {
    # A native executable on a Unix has no extension at all, so `-o thing`
    # is what this means and nothing can be told apart from it by spelling.
    "cc",
    # The same answer, for the same reason: it produces a native executable
    # too, and the difference between the two is what they need on the
    # machine, not what the result is called.
    "builtin",
    # Writes the BACKEND's artifacts under the backend's own names, so the
    # spelling belongs to the backend and never to this.
    "none",
}


class TestEveryBackendSaysWhatItWrites:
    def test_each_backend_declares_its_artifacts(self):
        missing = [name for name, be in backend_registry.available().items()
                   if not be.artifacts]
        assert missing == [], (
            "these backends cannot be chosen by output spelling because they "
            f"declare no artifacts: {sorted(missing)}")

    def test_an_extension_starts_with_a_dot(self):
        wrong = [(name, ext)
                 for name, be in backend_registry.available().items()
                 for ext in be.artifacts if not ext.startswith(".")]
        assert wrong == [], f"artifacts must be suffixes: {wrong}"

    def test_no_backend_claims_the_same_extension_twice(self):
        for name, be in backend_registry.available().items():
            assert len(set(be.artifacts)) == len(be.artifacts), (
                f"{name} lists an extension twice: {be.artifacts}")


class TestEveryLinkerSaysWhatItProduces:
    def test_each_toolchain_declares_its_artifacts(self):
        missing = [name for name, tc in link_registry.available().items()
                   if not tc.artifacts and name not in NO_ARTIFACTS_BY_DESIGN]
        assert missing == [], (
            "these linkers cannot be chosen by output spelling because they "
            f"declare no artifacts: {sorted(missing)}")

    def test_the_ones_with_nothing_to_claim_are_exactly_the_stated_ones(self):
        # BOTH DIRECTIONS, because each failure is silent in its own way. One
        # that forgets to declare and is not listed here is unreachable by
        # spelling; one that gains an artifact and stays listed here is an
        # exception nobody needs, and the next reader believes it.
        bare = {name for name, tc in link_registry.available().items()
                if not tc.artifacts}
        assert bare == NO_ARTIFACTS_BY_DESIGN, (
            f"undeclared and not listed as deliberate: "
            f"{sorted(bare - NO_ARTIFACTS_BY_DESIGN)}; "
            f"listed but now declares one: "
            f"{sorted(NO_ARTIFACTS_BY_DESIGN - bare)}")

    def test_an_extension_starts_with_a_dot(self):
        wrong = [(name, ext)
                 for name, tc in link_registry.available().items()
                 for ext in tc.artifacts if not ext.startswith(".")]
        assert wrong == [], f"artifacts must be suffixes: {wrong}"
