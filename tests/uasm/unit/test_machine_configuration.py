"""Native backend ISA configuration is explicit and never silently ignored."""
from __future__ import annotations

from tests import harness

from uasm.backends.arm64.emit import Arm64Backend
from uasm.backends.x86_64.emit import X86_64Backend
from uasm.backend.base import BackendUnsupported
from uasm.options import OptionError
from uasm.target import Target


class TestMachineConfiguration:
    def test_x86_cpu_and_required_feature_are_published(self):
        backend = X86_64Backend().configure(
            {"cpu": "x86-64", "feature": ["+sse2"]}, None)
        assert backend.machine.cpu == "x86-64"
        assert backend.machine.features == frozenset(("sse2",))

    def test_required_x86_feature_cannot_be_disabled(self):
        with harness.raises(OptionError, match="requires feature 'sse2'"):
            X86_64Backend().configure({"feature": ["-sse2"]}, None)

    def test_unknown_feature_is_rejected(self):
        with harness.raises(OptionError, match="cannot emit feature 'avx'"):
            X86_64Backend().configure({"feature": ["+avx"]}, None)

    def test_arm_cpu_baseline_is_selected(self):
        backend = Arm64Backend().configure({"cpu": "armv8-a"}, None)
        assert backend.machine.cpu == "armv8-a"

    def test_target_architecture_must_match_encoder(self):
        target = Target("wrong", arch="aarch64", pointer_size=8)
        with harness.raises(BackendUnsupported, match="architecture 'aarch64'"):
            X86_64Backend().validate_target(target)

    def test_target_width_must_match_encoder(self):
        target = Target("wrong", arch="x86_64", pointer_size=4)
        with harness.raises(BackendUnsupported, match="32-bit pointers"):
            X86_64Backend().validate_target(target)
