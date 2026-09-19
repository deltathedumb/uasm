"""The toolchain that links with no toolchain.

`staticlink.py` is the linker; this is the registration that lets the driver
choose it, and the thin layer that turns `LinkRequest` -- artifacts in memory,
paths on disk, a target -- into a call to it.

WHY IT IS A SEPARATE MODULE FROM THE LINKER. `staticlink` knows about ELF and
nothing about this compiler's driver: it takes bytes and gives bytes back, so
it can be tested by handing it hand-built objects and running the result, with
no `Target`, no workdir and no options. Everything that knows what a
`LinkRequest` is lives here.

WHAT IT CAN AND CANNOT LINK. ELF64 relocatables for x86-64 and AArch64, into a
STATIC executable with no libc and no dynamic loader. That is exactly what a
program built on the platform floor needs -- `plat_write`, `plat_exit` and
`plat_heap` are syscalls -- and it is not enough for a program that calls
`printf`. A missing symbol is reported by name, which is the honest failure:
the alternative is finding libc and becoming `cc` with extra steps.
"""
from __future__ import annotations

from pathlib import Path

from ..backend.objfile.elfread import ElfError, read
from ..target import Target
from .base import LinkError, LinkRequest, Toolchain
from .freestanding import PROVIDES, floor_object
from .registry import register
from .staticlink import DEFAULT_BASE, LinkFailed, executable, link

#: The symbol the program starts at. `_start` and not `main`: there is no
#: crt1.o in a static image built this way, so the ENTRY POINT is what the
#: kernel jumps to with the stack laid out as the ABI describes, and nothing
#: has run before it.
ENTRY = "_start"


class BuiltinToolchain(Toolchain):
    """Link ELF objects into a static executable, using no external tool."""

    name = "builtin"
    #: A native executable, which on a Unix has no extension -- the same
    #: answer `cc` gives, and for the same reason.
    artifacts = ()
    #: THE MACHINE BACKENDS AND NOT `c`. This links objects; the `c` backend
    #: writes C, which needs a compiler, which is the whole thing being
    #: avoided. A `c` artifact handed here is a file this cannot read, and
    #: saying so in the declaration is better than finding out at link time.
    backends = ("x86-64", "arm64")
    description = "link ELF objects into a static executable, with no " \
                  "external tools"

    def supports(self, target: Target) -> bool:
        """Whether an ELF static link is a thing this target wants.

        ASKED OF THE FORMAT AND NOT THE OPERATING SYSTEM. A target that wants
        Mach-O or PE is not served by this yet, and a bare-metal ELF one is --
        the difference is the container, not whether there is a kernel.
        """
        fmt = getattr(target, "object_format", None)
        return fmt in (None, "elf")

    def link(self, request: LinkRequest) -> Path:
        inputs: list[tuple[str, bytes]] = []
        # THE ARTIFACTS ARE ALREADY IN MEMORY, so they are linked from there
        # rather than written out and read back. The workdir copy is still
        # made when `--keep-intermediates` asks, because the point of that
        # flag is to be able to look at what went in.
        for name, blob in request.artifacts.items():
            if not name.endswith(".o"):
                raise LinkError(
                    f"the builtin linker cannot read {name}",
                    detail="it links ELF objects; a backend that emits "
                           "source or assembly needs a toolchain that can "
                           "compile it",
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

        inputs = _with_floor(inputs)
        try:
            image = link(inputs, entry=ENTRY, base=DEFAULT_BASE)
            blob = executable(image, base=DEFAULT_BASE)
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


def _with_floor(inputs: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    """The inputs, plus the freestanding floor when nothing else has one.

    ADDED ONLY WHEN IT IS NEEDED, and "needed" is decided by looking rather
    than by a flag: a caller that already links its own `_start` -- a
    bare-metal image, an object built against a different floor -- must not
    get a second one, and a duplicate definition is an error rather than a
    silent choice.

    The machine comes from the first input, because a link of objects for two
    machines is refused a moment later anyway and reading one header here is
    cheaper than reading all of them twice.
    """
    if not inputs:
        return inputs
    defined: set[str] = set()
    machine = None
    for origin, blob in inputs:
        try:
            got = read(blob, origin)
        except ElfError:
            # NOT THIS FUNCTION'S ERROR TO REPORT. The link is about to read
            # the same bytes and say so properly; guessing here would mean
            # two messages for one bad file.
            return inputs
        if machine is None:
            machine = got.machine
        defined.update(s.name for s in got.symbols
                       if s.name and not s.is_undefined)
    if not defined.isdisjoint(PROVIDES):
        return inputs
    try:
        return [("<floor>", floor_object(machine)), *inputs]
    except KeyError:
        # No floor for this machine. The link will fail on the undefined
        # symbols, which names them, which is the better message.
        return inputs


def load_builtin() -> None:
    register(BuiltinToolchain())


__all__ = ["BuiltinToolchain", "ENTRY", "load_builtin"]
