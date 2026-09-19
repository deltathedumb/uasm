# Adding a toolchain

A backend produces **artifacts** — assembly, C, object bytes. A **toolchain**
turns those into a program you can run.

That is not the backend's job. The same x86-64 assembly is assembled by gas on
Linux, by gas-for-COFF on Windows, and linked by ld, lld, or a C driver
standing in for either; making the backend own it means every backend growing
a copy of the same toolchain search.

## The whole thing

```python
from pathlib import Path

from uasm.link import (
    LinkError, LinkRequest, Toolchain, find_tool, register, run,
)

class MyLinker(Toolchain):
    name = "my-linker"
    description = "link with a linker script and no libc"

    def supports(self, target):
        return target.object_format == "elf"

    def link(self, request: LinkRequest) -> Path:
        ld = find_tool(("ld.lld", "ld"), what="linker")
        objs = []
        for name, data in request.artifacts.items():
            path = request.workdir / name
            path.write_bytes(data)
            objs.append(str(path))
        run(request, [ld, "-T", "link.ld", *objs, "-o", str(request.output)],
            what="linking")
        return request.output

register(MyLinker())
```

Then:

```bash
uasm build prog.py --toolchain my-linker
uasm toolchains                    # lists it
```

## What ships

| name | what it does |
|------|--------------|
| `builtin` | links ELF objects into a static executable with no external tool at all: uasm's own linker, its own `_start`, and the platform floor as syscalls. The object runtime is compiled by uasm's own C frontend. The default for a machine backend. |
| `cc` | hands everything to `gcc`/`clang`. Assembles `.s`, compiles `.c`, links, and finds the system libraries. What the `c` backend needs, and what a program linking against real libraries wants. |
| `jar` | packages class files into a runnable jar. Chosen automatically by the `jvm` target, which names it in `default_toolchain`. |
| `baremetal` | a freestanding image: no libc, no start files, a linker script, and a runtime built from source. Chosen automatically when the target's `os` is `"none"`. |
| `cpyext` | compiles and links a CPython extension module — a `.so` or a `.pyd` that `import` finds. Chosen automatically by the `cpyext` target. |
| `pyc` | writes the `pybc` backend's `.pyc` where `-o` says, rather than under an artifact name in a directory. Chosen automatically by a target whose object format is `pyc`. |
| `none` | writes the artifacts and stops. What `--emit` means. |

`jar` needs no JDK. A jar is a zip with a manifest and Python has both, so
building for the JVM asks nothing of the machine doing the building — only
*running* the result needs a JVM. It does not write the manifest either: the
backend emits `META-INF/MANIFEST.MF` as an artifact, because the backend is
what knows which class holds the entry point, and a toolchain reconstructing
that would be a second place deciding it.

`cc` uses a C compiler driver rather than calling `as` and `ld` directly
because the driver knows where `crt1.o`, libc and the dynamic loader live on
*this* machine. Reproducing that search is both the hardest part of linking and
the part with no portable answer — hand-written `ld` invocations work on the
machine they were written on.

`baremetal` is the case where that reasoning inverts, and it is a useful
contrast. There is no crt1.o, no libc and no loader to find, so the driver's
search buys nothing — while the memory map, the entry point and the runtime
are all things only the person writing the toolchain knows. Everything `cc`
delegates, it has to state. That is why it is a separate toolchain rather than
a flag: almost nothing is shared, and the one line they do share
(`find_tool`) is already shared.

Which one runs is decided by the target, not the command line — `--toolchain`
defaults to `cc`, and the driver substitutes `baremetal` when `target.os ==
"none"`. A default that is wrong for a whole class of targets is a default
people learn to override, and then override in the other direction by mistake.

## The request

```python
request.artifacts        # {filename: bytes} from the backend
request.target           # object_format, object_suffix, executable_suffix
request.output           # where the program goes
request.workdir          # for intermediates
request.extra_inputs     # --link-input: objects, archives, -l names
request.runtime_sources  # runtime the frontend needs (see below)
request.commands         # append what you run; --verbose prints it
```

Use `run(request, argv, what=...)` rather than `subprocess` directly. It
records the command line whether it succeeds or fails — when a build breaks,
the exact invocation is the first thing anyone wants, and a command
*described* in an error message drifts from the one that ran.

## Failures are diagnostics

Raise `LinkError`, not a bare exception:

```python
raise LinkError("no assembler found",
                detail="looked for: as, gas, clang",
                help="install binutils, or pass --toolchain none")
```

A missing assembler is an ordinary state for a machine to be in, not an
internal error. `find_tool` already does this, and names everything it tried —
"gcc not found" is unhelpful when `cc` or `clang` would have worked equally.

## The runtime

The IR has no I/O opcodes: `print` is a call to a named function, and
something has to define it. `uasm.link.runtime` supplies a small C file
for the Python frontend, and the driver adds it when the backend is not
`self_contained`.

```python
class MyBackend(Backend):
    self_contained = False   # my artifacts need the runtime linked in
```

The C backend sets `True` — it emits its own `main` and its own `print_int`.
A machine backend sets `False`. Getting it wrong produces a duplicate-symbol
error or an undefined one, both at link time and both clear.

A freestanding toolchain supplies its own runtime instead, because the shipped
one is C that expects a libc. `link/baremetal.py` has the AArch64 one, and the
part worth reading is `put_float`: sixty lines because there is no `printf`,
and correct because it is diff-tested against the host's `printf` on 200,000
values rather than eyeballed. Formatting looks like the trivial part of a
runtime and contains the ties — 0.5, 1.5, 2.5 — that a naive round-half sends
the wrong way, in output nobody double-checks.

The IR's `main` is emitted under `ENTRY_SYMBOL` (`uasm_main`), because
it is not C's `main` — it returns i64 where C requires int, and would
collide with the runtime's entry point. That constant lives in
`uasm.backend.base` so the backend writing the symbol and the runtime
calling it cannot disagree.

## Checklist

- [ ] `name` and `description` set; `register()` called
- [ ] `supports(target)` returns False for platforms you cannot produce
- [ ] every external command goes through `run(request, ...)`
- [ ] every failure a user can act on is a `LinkError` with `help`
- [ ] the produced program runs, and its output matches `uasm run` on
      the same source

## Getting it loaded

`register()` runs when your module is imported, and nothing imports it for
you. Declare what you provide and install it once:

```python
from uasm.plugins import Plugin

plugin = Plugin("mypack")
plugin.backends.append(MyBackend)         # a class or an instance, either
plugin.linkers.append(MyLinker)            # .frontends and .targets too

__uasm_plugin__ = plugin
```

```bash
uasm plugin add mypack        # remembered; loaded on every run afterwards
uasm plugin show mypack       # what it provides, registering none of it
uasm plugin list | remove
```

`add` looks in the working directory, then the Python path, then pip --
`--cwd 1|0`, `--pypath 1|0`, `--pip 1|0`, with pip off unless asked.

### Replacing and refreshing

`plugin add` on a name that is already installed ASKS before replacing it,
because replacing clears a cached copy that may be the only one left -- the
origin it came from is not guaranteed to still exist. `--yes` and `--no`
answer without asking, and a non-interactive run declines rather than
prompting: a build that blocks forever on a hidden question is worse than one
that stops and names the flag.

An installed plugin is CACHED and loaded from the cache, not from wherever it
was found, so an install keeps working when the original file moves or the
compiler is run from another directory. `origin` stays recorded for exactly
one purpose:

```bash
uasm plugin invalidate mypack           # one id
uasm plugin invalidate a,b              # comma-separated
uasm plugin invalidate a b              # or repeated
uasm plugin invalidate --all
```

`invalidate` goes back to the origin, re-resolves, and refreshes the cache --
which is how an edited plugin under development is picked up. If the origin is
gone it fails and says so, leaving the cached copy in place: a cache that no
longer matches any real source is exactly the state worth being told about.

Without installing: `--plugin mypack` for one invocation, `UASM_PLUGINS`
for a CI job, or an `uasm.plugins` entry point if you ship a
distribution. Declaring a manifest is better than calling `register()` at
import time, which also works: a manifest can be READ, so `plugin show` and
the install-time report can say what a module provides without letting it
change the compiler's state first.
