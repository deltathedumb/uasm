"""The targets that ship with uasm.

Every one is an ordinary `register()` call -- there is no privileged path for
built-ins. If this file were deleted, uasm would still compile; it would simply
have no platforms until something registered one, which is the property that
makes the extension point real rather than decorative.

`docs/TARGETS.md` explains adding your own.
"""
from __future__ import annotations

from ..target.base import Target
from ..target.registry import register

#: The reference target for the C backend. Not a machine: the "object format"
#: is source, and the fields describing a machine are the ones a C compiler
#: will decide for itself later.
PORTABLE_C = register(Target(
    "c", arch="any", os="any", abi="none", object_format="source",
    object_suffix=".c",
), aliases=("portable", "source"))

X86_64_LINUX = register(Target(
    "x86_64-linux", arch="x86_64", os="linux", abi="sysv",
    object_format="elf", object_suffix=".o", executable_suffix="",
), aliases=("linux", "x86_64-unknown-linux-gnu"))

X86_64_WINDOWS = register(Target(
    "x86_64-windows", arch="x86_64", os="windows", abi="win64",
    object_format="coff", object_suffix=".obj", executable_suffix=".exe",
), aliases=("windows", "win64", "x86_64-pc-windows-msvc"))

X86_64_MACOS = register(Target(
    "x86_64-macos", arch="x86_64", os="macos", abi="sysv",
    object_format="macho", object_suffix=".o", executable_suffix="",
), aliases=("macos", "darwin"))

#: Bare-metal AArch64: no operating system, no libc entry point. This is the
#: one that can be EXECUTED on a developer machine without an ARM box --
#: qemu-system-aarch64 boots it directly with `-M virt -kernel`, which is why
#: the arm64 backend defaults to it and the tests use it.
AARCH64_NONE = register(Target(
    "aarch64-none", arch="aarch64", os="none", abi="aapcs64",
    object_format="elf", object_suffix=".o", executable_suffix=".elf",
    cc_names=("aarch64-none-elf-gcc", "aarch64-elf-gcc"),
), aliases=("arm64-bare", "aarch64-elf"))

AARCH64_LINUX = register(Target(
    "aarch64-linux", arch="aarch64", os="linux", abi="aapcs64",
    object_format="elf", object_suffix=".o", executable_suffix="",
    cc_names=("aarch64-linux-gnu-gcc", "aarch64-none-linux-gnu-gcc"),
), aliases=("arm64", "arm64-linux"))

#: WINDOWS ON ARM. The ABI is AAPCS64 with Microsoft's variations, and the
#: one that matters to a code generator is that `x18` is the thread
#: environment block's -- which this backend already leaves alone, because
#: the AAPCS calls it the platform register and says not to touch it.
AARCH64_WINDOWS = register(Target(
    "aarch64-windows", arch="aarch64", os="windows", abi="aapcs64",
    object_format="coff", object_suffix=".obj", executable_suffix=".exe",
), aliases=("arm64-windows", "aarch64-pc-windows-msvc"))

AARCH64_MACOS = register(Target(
    "aarch64-macos", arch="aarch64", os="macos", abi="aapcs64",
    object_format="macho", object_suffix=".o", executable_suffix="",
    cc_names=("clang",),
), aliases=("arm64-macos", "apple-silicon"))

#: The Java virtual machine. Not a machine anyone builds: the architecture is
#: the bytecode, the "object format" is a class file, and there is no calling
#: convention to get wrong because the JVM's own is the only one.
#:
#: `pointer_size` is 8 and the byte order little-endian because the IR's memory
#: is a byte array this target's backend indexes itself -- those two fields
#: describe that array, not a JVM, which has no addressable memory at all.
JVM = register(Target(
    "jvm", arch="jvm", os="jvm", abi="jvm",
    object_format="class", object_suffix=".class", executable_suffix=".jar",
    stack_alignment=8, default_toolchain="jar",
), aliases=("java", "jar"))

#: The host CPython interpreter, as a bytecode target. Not a machine either:
#: like `jvm`, the "architecture" is a virtual machine's own instruction set
#: (CPython's, this release's), and the "object format" is a `.pyc` --
#: written whole by the `pybc` backend, not assembled from parts, which is
#: why `default_toolchain` just names the file rather than invoking one.
PYBC = register(Target(
    "pybc", arch="cpython", os="cpython", abi="cpython",
    object_format="pyc", object_suffix=".pyc", executable_suffix=".pyc",
    stack_alignment=8, default_toolchain="pyc",
), aliases=("pyc",))

#: A REAL CPYTHON EXTENSION MODULE, not a program: `object_format="source"`
#: because the `cpyext` backend emits C, same as `c` -- what makes these
#: targets distinct is `abi="cpyext"`, which is what tells
#: `CPyExtToolchain.supports()` to link with `-shared -fPIC` and the Python
#: headers instead of building an executable. `executable_suffix` is the
#: DEFAULT name only: CPython's own import machinery looks for the fuller
#: ABI-tagged suffix (`importlib.machinery.EXTENSION_SUFFIXES[0]`, e.g.
#: `.cpython-314-x86_64-linux-gnu.so`) to find a module by bare `import
#: name` on `sys.path` -- `-o name.cpython-...-gnu.so` gets that; a plain
#: `.so`/`.pyd` still loads correctly through
#: `importlib.util.spec_from_file_location`, which is what every extension
#: module in `tests/uasm/unit/test_cpyext_backend.py` uses to avoid
#: hard-coding the host's own tag into a test that runs on every host.
X86_64_LINUX_CPYEXT = register(Target(
    "x86_64-linux-cpyext", arch="x86_64", os="linux", abi="cpyext",
    object_format="source", object_suffix=".c", executable_suffix=".so",
    default_toolchain="cpyext",
), aliases=("cpyext-linux", "cpyext"))

#: THE ONE THIS SESSION CANNOT LINK: it needs a Windows-targeting C
#: compiler (`x86_64-w64-mingw32-gcc`, MinGW-w64's cross toolchain) on
#: PATH, which is not installed everywhere this compiler is. Registered
#: and code-complete regardless -- `uasm plugin targets`/`uasm plugin backends`
#: should show the whole matrix, and `CPyExtToolchain.link()` refuses with
#: a named, installable tool rather than a traceback when it is missing,
#: exactly like every other cross target here (`AARCH64_LINUX` et al.).
X86_64_WINDOWS_CPYEXT = register(Target(
    "x86_64-windows-cpyext", arch="x86_64", os="windows", abi="cpyext",
    object_format="source", object_suffix=".c", executable_suffix=".pyd",
    cc_names=("x86_64-w64-mingw32-gcc",), default_toolchain="cpyext",
), aliases=("cpyext-windows", "pyd"))

__all__ = ["PORTABLE_C", "X86_64_LINUX", "X86_64_WINDOWS", "X86_64_MACOS",
           "AARCH64_NONE", "AARCH64_LINUX", "AARCH64_MACOS", "JVM", "PYBC",
           "X86_64_LINUX_CPYEXT", "X86_64_WINDOWS_CPYEXT"]
