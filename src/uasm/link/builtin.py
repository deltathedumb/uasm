"""The toolchain that links with no toolchain.

`staticlink.py` is the linker; this is the registration that lets the driver
choose it, and the thin layer that turns `LinkRequest` -- artifacts in memory,
paths on disk, a target -- into a call to it.

WHY IT IS A SEPARATE MODULE FROM THE LINKER. `staticlink` knows about object
containers and nothing about this compiler's driver: it takes bytes and gives
bytes back, so it can be tested by handing it hand-built objects and running
the result, with no `Target`, no workdir and no options. Everything that knows
what a `LinkRequest` is lives here.

WHAT IT CAN AND CANNOT LINK. ELF64, PE/COFF and Mach-O relocatables, for
x86-64 and AArch64, into a STATIC executable with no libc and no dynamic
loader. That is exactly what a program built on the platform floor needs --
`plat_write`, `plat_exit` and `plat_heap` are three syscalls on Linux and on
macOS, and three calls into `kernel32.dll` on Windows -- and it is not enough
for a program that calls `printf`. A missing symbol is reported by name,
which is the honest failure: the alternative is finding libc and becoming
`cc` with extra steps.

THE CONTAINER IS READ OFF THE OBJECTS, not taken from the target. `uasm link
a.obj` says nothing about Windows and the object says everything; a target
that disagrees with its own inputs would be a worse answer than the inputs.
"""
from __future__ import annotations

from pathlib import Path

from ..backend.objfile import EM_AARCH64, EM_X86_64
from ..backend.objfile.coffread import (
    IMAGE_FILE_MACHINE_AMD64, IMAGE_FILE_MACHINE_ARM64,
)
from ..backend.objfile.macho import CPU_TYPE_ARM64, CPU_TYPE_X86_64
from ..target import Target
from .base import LinkError, LinkRequest, Toolchain
from .freestanding import floor_object, provides
from .pewrite import IMP_PREFIX
from .registry import register
from .runtimeobj import RuntimeBuildFailed, runtime_object
from . import machowrite
from .staticlink import LinkFailed, executable, link, read_object

#: WHICH BACKEND BUILT AN OBJECT, read back off the object. The runtime has
#: to be compiled for the same machine and into the same container as the
#: program it is linked with, and the program is the only thing here that
#: knows which those are -- the `LinkRequest`'s target may be the host's
#: default when `uasm link a.obj` is all that was said.
_BACKEND_OF = {
    ("elf", EM_X86_64): "x86-64",
    ("elf", EM_AARCH64): "arm64",
    ("coff", IMAGE_FILE_MACHINE_AMD64): "x86-64",
    ("coff", IMAGE_FILE_MACHINE_ARM64): "arm64",
    ("macho", CPU_TYPE_X86_64): "x86-64",
    ("macho", CPU_TYPE_ARM64): "arm64",
}

#: The architecture names that go with each, for deciding whether a target
#: the caller already named is the right one after all.
_ARCH_OF = {
    ("elf", EM_X86_64): ("x86_64",),
    ("elf", EM_AARCH64): ("aarch64",),
    ("coff", IMAGE_FILE_MACHINE_AMD64): ("x86_64",),
    ("coff", IMAGE_FILE_MACHINE_ARM64): ("aarch64",),
    ("macho", CPU_TYPE_X86_64): ("x86_64",),
    ("macho", CPU_TYPE_ARM64): ("aarch64",),
}

#: And which TARGET to compile it for, since the container is part of the
#: answer and a backend name is not.
_TARGET_OF = {
    ("elf", EM_X86_64): "x86_64-linux",
    ("elf", EM_AARCH64): "aarch64-linux",
    ("coff", IMAGE_FILE_MACHINE_AMD64): "x86_64-windows",
    ("coff", IMAGE_FILE_MACHINE_ARM64): "aarch64-windows",
    ("macho", CPU_TYPE_X86_64): "x86_64-macos",
    ("macho", CPU_TYPE_ARM64): "aarch64-macos",
}

#: The symbol the program starts at. `_start` and not `main`: there is no
#: crt1.o in a static image built this way, so the ENTRY POINT is what the
#: kernel jumps to with the stack laid out as the ABI describes, and nothing
#: has run before it.
ENTRY = "_start"


class BuiltinToolchain(Toolchain):
    """Link relocatables into a static executable, using no external tool."""

    name = "builtin"
    #: A native executable, which on a Unix has no extension -- the same
    #: answer `cc` gives, and for the same reason.
    artifacts = ()
    #: THE MACHINE BACKENDS AND NOT `c`. This links objects; the `c` backend
    #: writes C, which needs a compiler, which is the whole thing being
    #: avoided. A `c` artifact handed here is a file this cannot read, and
    #: saying so in the declaration is better than finding out at link time.
    backends = ("x86-64", "arm64")
    description = "link objects into a static executable, with no " \
                  "external tools"

    def supports(self, target: Target) -> bool:
        """Whether a static link is a thing this target wants.

        ASKED OF THE FORMAT AND NOT THE OPERATING SYSTEM. A bare-metal ELF
        target is served exactly as a Linux one is -- the difference is the
        container, not whether there is a kernel -- and Mach-O is the one
        that is not served yet.
        """
        fmt = getattr(target, "object_format", None)
        return fmt in (None, "elf", "coff", "macho")

    def link(self, request: LinkRequest) -> Path:
        inputs: list[tuple[str, bytes]] = []
        # THE ARTIFACTS ARE ALREADY IN MEMORY, so they are linked from there
        # rather than written out and read back. The workdir copy is still
        # made when `--keep-intermediates` asks, because the point of that
        # flag is to be able to look at what went in.
        for name, blob in request.artifacts.items():
            if not name.endswith((".o", ".obj")):
                raise LinkError(
                    f"the builtin linker cannot read {name}",
                    detail="it links relocatable objects; a backend that "
                           "emits source or assembly needs a toolchain that "
                           "can compile it",
                    help="use `-ln cc`, or a backend that emits objects "
                         "(-bk x86-64)")
            inputs.append((name, blob))
            if request.keep_intermediates:
                (request.workdir / name).write_bytes(blob)
        for path in request.input_paths:
            try:
                inputs.append((str(path), path.read_bytes()))
            except OSError as exc:
                raise LinkError(f"cannot read {path}",
                                detail=exc.strerror or "") from None
        if request.extra_inputs:
            raise LinkError(
                "the builtin linker takes no library inputs",
                detail=f"given: {', '.join(request.extra_inputs)}",
                help="a static link here resolves every symbol from the "
                     "objects it is handed; there is no library search path "
                     "and no `-l` handling")

        inputs = _with_runtime(inputs, request)
        inputs = _with_floor(inputs)
        try:
            # NO BASE GIVEN. Where an image is loaded is the container's
            # to say -- 0x400000 for ELF and PE, 0x100000000 for Mach-O,
            # which is where `__PAGEZERO` ends -- and the container is not
            # known until the objects have been read.
            image = link(inputs, entry=ENTRY)
            blob = executable(image)
            # A CORRECT IMAGE THE PLATFORM WILL NOT START is still worth
            # saying out loud. `machowrite` knows the one case there is.
            if image.fmt == "macho":
                said = machowrite.arm64_static_will_not_load(image.machine)
                if said:
                    request.notes.append(said)
        except LinkFailed as exc:
            raise LinkError(exc.message, detail=exc.detail) from None

        request.output.parent.mkdir(parents=True, exist_ok=True)
        request.output.write_bytes(blob)
        request.output.chmod(0o755)
        # RECORDED AS IF IT WERE A COMMAND, so `--verbose` and the tests that
        # read `commands` see the link step at all. It is not a command and
        # saying so plainly is better than printing nothing: a build that
        # shows every step except the one that produced the file reads as if
        # the file appeared by itself.
        request.commands.append(
            ["<builtin linker>", "-o", str(request.output),
             *(name for name, _ in inputs)])
        return request.output


def _survey(inputs: list[tuple[str, bytes]]) -> tuple[
        set[str], set[str], tuple[str, int] | None]:
    """What the inputs define, what they still need, and what they are for.

    The third answer is `(container, machine)` -- the key everything else
    here is looked up by -- or None when the inputs could not all be read.
    NOT AN ERROR TO REPORT FROM HERE: the link is about to read the same
    bytes and say so properly, and two messages for one bad file is worse
    than one.
    """
    defined: set[str] = set()
    wanted: set[str] = set()
    kind: tuple[str, int] | None = None
    for origin, blob in inputs:
        try:
            got = read_object(blob, origin)
        except Exception:                          # noqa: BLE001
            return set(), set(), None
        if kind is None:
            kind = (getattr(got, "fmt", "elf"), got.machine)
        for sym in got.symbols:
            if not sym.name:
                continue
            (wanted if sym.is_undefined else defined).add(sym.name)
    return defined, wanted - defined, kind


def _c_names(names: set[str], fmt: str) -> set[str]:
    """The same symbols, spelled the way the C they came from spells them.

    MACH-O PUTS AN UNDERSCORE IN FRONT OF EVERY C NAME, so a program object
    for macOS defines `_apy_err_slots` where the C it was compiled from says
    `apy_err_slots`. `runtimeobj.runtime_source` decides what to leave OUT of
    the runtime by matching those names -- so handing it the platform's
    spelling left everything in, and the link then had every runtime symbol
    defined twice.
    """
    if fmt != "macho":
        return names
    return {name[1:] if name.startswith("_") else name for name in names}


def _imports(wanted: set[str], kind: tuple[str, int] | None) -> set[str]:
    """The names the CONTAINER answers rather than an object.

    A PE reaches the kernel through a DLL, so `__imp_WriteFile` is not a
    missing symbol but a slot the linker builds an import table for. On ELF
    there is no such thing and this is empty, which is why the caller can
    subtract it unconditionally.
    """
    if kind is None or kind[0] != "coff":
        return set()
    return {name for name in wanted if name.startswith(IMP_PREFIX)}


def _with_runtime(inputs: list[tuple[str, bytes]],
                  request: LinkRequest) -> list[tuple[str, bytes]]:
    """The inputs, plus the object runtime when the program still needs it.

    COMPILED BY UASM, from C generated to match what these objects already
    define -- see `runtimeobj.py`. Added only when something is missing that
    is not the floor: an object that needs nothing needs no runtime, and a
    program already linked against one must not get a second.
    """
    defined, wanted, kind = _survey(inputs)
    fmt = kind[0] if kind else "elf"
    wanted = wanted - set(provides(fmt)) - _imports(wanted, kind)
    if not wanted:
        return inputs
    backend = _BACKEND_OF.get(kind)
    if backend is None:
        return inputs
    try:
        blob = runtime_object(_c_names(defined, fmt), backend=backend,
                              target=_target_for(kind, request.target),
                              workdir=request.workdir,
                              verbose=request.verbose)
    except RuntimeBuildFailed as exc:
        raise LinkError(exc.message, detail=exc.detail,
                        help="the object runtime is compiled by uasm itself; "
                             "this is a compiler failure, not a missing "
                             "tool") from None
    request.commands.append(["<uasm>", "--object-runtime", "c",
                             "the object runtime"])
    return [*inputs, ("<runtime>", blob)]


def _target_for(kind: tuple[str, int] | None, given):
    """The target to compile the object runtime for.

    THE OBJECTS DECIDE, not the request. `uasm link hello.obj` names no
    target, so the request carries the host's -- and compiling a Windows
    program's runtime for Linux would fail at the link with a hundred
    undefined names rather than with the one sentence that is true. The
    request's own target is kept when it already agrees, because it may be a
    more specific member of the same family than the table's default.
    """
    want = _TARGET_OF.get(kind)
    if want is None:
        return given
    if getattr(given, "object_format", None) == kind[0] and \
            getattr(given, "arch", None) in _ARCH_OF.get(kind, ()):
        return given
    from .. import target as target_registry
    try:
        return target_registry.get(want)
    except Exception:                              # noqa: BLE001
        return given


def _with_floor(inputs: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    """The inputs, plus the freestanding floor when nothing else has one.

    ADDED ONLY WHEN IT IS NEEDED, and "needed" is decided by looking rather
    than by a flag: a caller that already links its own `_start` -- a
    bare-metal image, an object built against a different floor -- must not
    get a second one, and a duplicate definition is an error rather than a
    silent choice.
    """
    if not inputs:
        return inputs
    defined, _, kind = _survey(inputs)
    if kind is None or not defined.isdisjoint(provides(kind[0])):
        return inputs
    fmt, machine = kind
    try:
        return [("<floor>", floor_object(machine, fmt=fmt)), *inputs]
    except KeyError:
        # No floor for this container and machine. The link will fail on the
        # undefined symbols, which names them, which is the better message.
        return inputs


def load_builtin() -> None:
    register(BuiltinToolchain())


__all__ = ["BuiltinToolchain", "ENTRY", "load_builtin"]
