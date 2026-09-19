"""The toolchain registry."""
from __future__ import annotations

from .base import Toolchain

_REGISTRY: dict[str, Toolchain] = {}


def register(tc: Toolchain) -> Toolchain:
    if not tc.name:
        raise ValueError(f"{type(tc).__name__} has no name")
    _REGISTRY[tc.name] = tc
    return tc


def unregister(name: str) -> None:
    """Remove a toolchain. Registration is process-wide, so anything that adds
    one temporarily -- a test, a plugin being reloaded -- needs a way to take
    it back out. Removing one that is not there is not an error: the caller is
    asking for a state, not performing a transaction."""
    _REGISTRY.pop(name, None)


def get(name: str) -> Toolchain:
    load_builtin()
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise LookupError(
            f"unknown toolchain {name!r}\navailable: {known}") from None


def available() -> dict[str, Toolchain]:
    load_builtin()
    return dict(_REGISTRY)


_loaded = False


def load_builtin() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    from .toolchains import load_builtin as _load
    _load()
    from .baremetal import load_builtin as _load_bare
    _load_bare()
    from .builtin import load_builtin as _load_own
    _load_own()
