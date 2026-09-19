"""Writing a PE executable, import table and all.

`staticlink.py` lays out, resolves and patches; this is the container it
writes the result into when the inputs were COFF. The ELF half of that job is
a hundred lines and lives in `staticlink.executable`; this is longer for one
reason, and it is not the headers.

WINDOWS HAS NO SYSCALL ABI A PROGRAM MAY USE. On Linux the floor is three
`syscall` instructions and the image needs nothing from anybody; `int 0x2e`
and the `syscall` numbers in `ntdll` are unstable between builds of the
operating system, and Microsoft says so. So a Windows program reaches the
kernel through `kernel32.dll` -- and calling a DLL means an IMPORT TABLE,
which is a linker's job and the reason this file exists at all. A "static"
PE is static in the sense that it has no C runtime; it still imports.

WHAT AN IMPORT TABLE IS, laid out here so the code below reads as arithmetic
rather than as magic:

    IMAGE_IMPORT_DESCRIPTOR[]   one per DLL, then a zeroed one to end it
      .OriginalFirstThunk  -> the Import Lookup Table
      .Name                -> "kernel32.dll"
      .FirstThunk          -> the Import Address Table
    ILT: uint64[]               one per function, then a zero
      each -> IMAGE_IMPORT_BY_NAME { uint16 hint; char name[]; }
    IAT: uint64[]               the SAME contents as the ILT on disk
    names                       the hint/name pairs, and the DLL names

The loader walks the ILT, resolves each name, and writes the function's
address over the corresponding IAT slot. So an indirect call through the IAT
slot reaches the function, and the slot's address is what `__imp_<Name>`
means -- which is MSVC's convention and the one the freestanding object here
uses.

THE IAT MUST BE WRITABLE, because the loader writes into it. It is put in
`.data` for that reason and not in `.rdata`, where a real linker puts it
after marking the range in the data directory; the simpler placement costs a
few bytes of a page and cannot be got subtly wrong.
"""
from __future__ import annotations

import struct

#: The DLL each imported name comes from.
#:
#: A FIXED TABLE, and honest about being one. Nothing but the freestanding
#: floor in `link/freestanding.py` imports anything, and it imports these
#: four; a name outside this table is refused by name rather than guessed at,
#: because guessing which DLL exports a symbol is what an import library is
#: for and there is none here.
WINDOWS_IMPORTS = {
    "ExitProcess": "kernel32.dll",
    "WriteFile": "kernel32.dll",
    "GetStdHandle": "kernel32.dll",
    "VirtualAlloc": "kernel32.dll",
}

#: The prefix that makes a name an import. MSVC's convention: `__imp_WriteFile`
#: is the ADDRESS OF THE IAT SLOT, so `call *__imp_WriteFile(%rip)` is an
#: indirect call through it.
IMP_PREFIX = "__imp_"

# ── PE constants ────────────────────────────────────────────────────────────
IMAGE_FILE_EXECUTABLE_IMAGE = 0x0002
IMAGE_FILE_LARGE_ADDRESS_AWARE = 0x0020
PE32PLUS_MAGIC = 0x20B
IMAGE_SUBSYSTEM_WINDOWS_CUI = 3
#: `IMAGE_DLLCHARACTERISTICS_NX_COMPAT` alone. Not `DYNAMIC_BASE`: this image
#: has no relocation directory, so it cannot be moved and must not claim it
#: can -- a loader that believed the claim and moved it would corrupt every
#: absolute address in it.
DLL_CHARACTERISTICS = 0x0100

IMAGE_SCN_CNT_CODE = 0x00000020
IMAGE_SCN_CNT_INITIALIZED_DATA = 0x00000040
IMAGE_SCN_CNT_UNINITIALIZED_DATA = 0x00000080
IMAGE_SCN_MEM_EXECUTE = 0x20000000
IMAGE_SCN_MEM_READ = 0x40000000
IMAGE_SCN_MEM_WRITE = 0x80000000

#: The little DOS program every PE still begins with, and the only reason it
#: exists: run on DOS, it prints a line and exits. `e_lfanew` at offset 0x3c
#: is the field that actually matters -- it says where the real header is,
#: and everything before it is this fossil.
_DOS_PROGRAM = (
    b"\x0e\x1f\xba\x0e\x00\xb4\x09\xcd\x21\xb8\x01\x4c\xcd\x21"
    b"This program cannot be run in DOS mode.\r\r\n$")


def _dos_header() -> bytes:
    """The 64-byte `IMAGE_DOS_HEADER` and the stub after it.

    WRITTEN FIELD BY FIELD rather than as a copied blob, because the one
    field that is read is computed: `e_lfanew` is the offset of `PE\0\0`,
    and a blob with it baked in is a blob that is wrong the moment the stub's
    length changes. That is how this file's first attempt put `e_lfanew` at
    the wrong offset entirely and produced a file `objdump` would not name.
    """
    stub = _DOS_PROGRAM + b"\0" * (-len(_DOS_PROGRAM) % 8)
    head = bytearray(64)
    head[0:2] = b"MZ"
    struct.pack_into("<13H", head, 2,
                     len(stub) % 512,        # e_cblp: bytes on the last page
                     1,                      # e_cp: pages in the file
                     0,                      # e_crlc: relocations
                     4,                      # e_cparhdr: header paragraphs
                     0x0000, 0xFFFF,         # e_minalloc, e_maxalloc
                     0, 0xB8,                # e_ss, e_sp
                     0, 0, 0,                # e_csum, e_ip, e_cs
                     0x40, 0)                # e_lfarlc, e_ovno
    struct.pack_into("<I", head, 0x3C, 64 + len(stub))
    return bytes(head) + stub


_MACHINE_OF = {0x8664: 0x8664, 0xAA64: 0xAA64}


class PeError(Exception):
    """A PE image could not be written. Carries a user-facing reason."""


def imports_needed(wanted: set[str]) -> dict[str, list[str]]:
    """The imports these undefined names ask for, grouped by DLL.

    ONLY THE `__imp_` ONES. Everything else is an ordinary undefined symbol
    and the linker reports it as such; turning any missing name into an
    import would make a typo into a program that fails at load time on
    somebody else's machine.
    """
    by_dll: dict[str, list[str]] = {}
    for name in sorted(wanted):
        if not name.startswith(IMP_PREFIX):
            continue
        bare = name[len(IMP_PREFIX):]
        dll = WINDOWS_IMPORTS.get(bare)
        if dll is None:
            raise PeError(
                f"{name!r} asks for an import and no DLL is known for "
                f"{bare!r}",
                )
        by_dll.setdefault(dll, []).append(bare)
    return by_dll


def import_section(by_dll: dict[str, list[str]], rva: int) -> \
        tuple[bytes, dict[str, int], tuple[int, int]]:
    """The import directory's bytes, the `__imp_` slots, and the IAT.

    `rva` is where these bytes will live, as a virtual address -- every
    pointer inside an import table is one, so the layout has to be decided
    before the bytes can be written.

    Returns the blob, `{"__imp_WriteFile": RVA of its IAT slot}`, and the
    `(rva, size)` of the Import Address Table, which goes in a data
    directory of its own so that a loader can make just that range writable.
    """
    if not by_dll:
        return b"", {}, (0, 0)

    descriptors = 20 * (len(by_dll) + 1)      # one each, plus the terminator
    # THE ILT AND THE IAT ARE THE SAME CONTENTS AT TWO ADDRESSES. The loader
    # reads one and overwrites the other, so they cannot be shared -- a
    # linker that pointed both at one array would have the loader destroy
    # the names it was still reading.
    thunks = sum(8 * (len(names) + 1) for names in by_dll.values())
    ilt_at = descriptors
    iat_at = descriptors + thunks
    names_at = descriptors + 2 * thunks

    blob = bytearray(names_at)
    names = bytearray()

    def put_name(text: str, hint: bool) -> int:
        """Append a name and answer its RVA. Hint/name pairs are 2-aligned."""
        at = names_at + len(names)
        if hint:
            names.extend(struct.pack("<H", 0))
            names.extend(text.encode("ascii") + b"\0")
            if len(names) & 1:
                names.append(0)
        else:
            names.extend(text.encode("ascii") + b"\0")
        return rva + at

    slots: dict[str, int] = {}
    ilt_cursor, iat_cursor = ilt_at, iat_at
    for i, (dll, funcs) in enumerate(sorted(by_dll.items())):
        struct.pack_into("<IIIII", blob, 20 * i,
                         rva + ilt_cursor,     # OriginalFirstThunk
                         0, 0,                 # TimeDateStamp, ForwarderChain
                         put_name(dll, hint=False),
                         rva + iat_cursor)     # FirstThunk
        for func in funcs:
            where = put_name(func, hint=True)
            struct.pack_into("<Q", blob, ilt_cursor, where)
            struct.pack_into("<Q", blob, iat_cursor, where)
            slots[IMP_PREFIX + func] = rva + iat_cursor
            ilt_cursor += 8
            iat_cursor += 8
        # THE ZERO THAT ENDS EACH LIST, which `bytearray` already holds.
        ilt_cursor += 8
        iat_cursor += 8
    return (bytes(blob) + bytes(names), slots,
            (rva + iat_at, iat_cursor - iat_at))


def header_space(page: int, sections: int = 8) -> int:
    """How many bytes the headers need in front of the first section.

    ASKED BEFORE THE LAYOUT, which is why it takes a section COUNT and not
    the sections: the answer decides where the first one goes, so it cannot
    be computed from where they went. Eight is more than this linker's four
    buckets plus the import table, and a page is a page either way.
    """
    return _round_up(len(_dos_header()) + 4 + 20 + 240 + 40 * sections, page)


def _characteristics(flags: int, nobits: bool) -> int:
    """PE section characteristics, from the linker's ELF-shaped flags."""
    out = IMAGE_SCN_MEM_READ
    if flags & 0x4:                                   # SHF_EXECINSTR
        out |= IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE
    elif nobits:
        out |= IMAGE_SCN_CNT_UNINITIALIZED_DATA
    else:
        out |= IMAGE_SCN_CNT_INITIALIZED_DATA
    if flags & 0x1:                                   # SHF_WRITE
        out |= IMAGE_SCN_MEM_WRITE
    return out


def executable(image, *, base: int, page: int, import_rva: int = 0,
               import_size: int = 0, iat: tuple[int, int] = (0, 0)) -> bytes:
    """Wrap a laid-out image in a PE executable.

    `image` is `staticlink.Image`; `base` is where it was laid out, which is
    the PE's `ImageBase`. The layout invariant `vaddr = base + file offset`
    means `SectionAlignment` and `FileAlignment` are the same number -- which
    PE permits, and which is why no section here needs a second offset.
    """
    machine = _MACHINE_OF.get(image.machine)
    if machine is None:
        raise PeError(f"no PE support for COFF machine {image.machine:#x}")

    sections = [s for s in image.sections if s.pieces]
    sizeof_headers = header_space(page, len(sections))
    first = min(s.addr - base for s in sections)
    if sizeof_headers > first:
        # NOT AN ASSERTION, because the way to get here is a layout that did
        # not reserve the headers any room -- and the symptom of that is the
        # headers written over the start of `.text`, which is a program that
        # jumps into a DOS stub. Saying so is better than shipping it.
        raise PeError(
            "the PE headers do not fit in front of the first section",
            )

    table = bytearray()
    size_code = size_data = size_bss = 0
    for sec in sections:
        raw = 0 if sec.nobits else _round_up(sec.size, page)
        table += struct.pack(
            "<8sIIIIIIHHI",
            sec.name.encode("ascii")[:8].ljust(8, b"\0"),
            sec.size,                       # VirtualSize
            sec.addr - base,                # VirtualAddress
            raw,                            # SizeOfRawData
            0 if sec.nobits else sec.addr - base,   # PointerToRawData
            0, 0, 0, 0,
            _characteristics(sec.flags, sec.nobits))
        if sec.flags & 0x4:
            size_code += sec.size
        elif sec.nobits:
            size_bss += sec.size
        else:
            size_data += sec.size

    end = max((s.addr - base) + s.size for s in sections)
    image_size = _round_up(end, page)

    coff = struct.pack("<HHIIIHH", machine, len(sections), 0, 0, 0, 240,
                       IMAGE_FILE_EXECUTABLE_IMAGE
                       | IMAGE_FILE_LARGE_ADDRESS_AWARE)

    # THE DATA DIRECTORIES, of which exactly two are filled. Import is the
    # one that matters; IAT is advisory and some loaders use it to make the
    # range writable before resolving, which costs nothing to state.
    directories = [(0, 0)] * 16
    if import_size:
        directories[1] = (import_rva, import_size)
        directories[12] = iat

    optional = struct.pack(
        "<HBBIIIIIQ", PE32PLUS_MAGIC, 14, 0, size_code, size_data, size_bss,
        image.entry - base, sections[0].addr - base, base)
    optional += struct.pack(
        "<IIHHHHHHIIIIHHQQQQII",
        page, page,                 # SectionAlignment, FileAlignment
        6, 0, 0, 0, 6, 0,           # OS / image / subsystem versions
        0, image_size, sizeof_headers, 0,
        IMAGE_SUBSYSTEM_WINDOWS_CUI, DLL_CHARACTERISTICS,
        0x100000, 0x1000,           # stack reserve / commit
        0x100000, 0x1000,           # heap reserve / commit
        0, len(directories))
    for rva, size in directories:
        optional += struct.pack("<II", rva, size)

    dos = _dos_header()
    head = bytearray(dos + b"PE\0\0" + coff + optional + bytes(table))
    if len(head) > sizeof_headers:
        raise PeError("the PE headers do not fit in the space reserved "
                      "for them")

    out = bytearray(image.data)
    if len(out) < len(head):
        out.extend(b"\0" * (len(head) - len(out)))
    out[0:len(head)] = head
    # EVERY SECTION IS A WHOLE NUMBER OF PAGES ON DISK, because
    # `SizeOfRawData` was rounded up and a loader reads that many bytes.
    want = max((s.addr - base) + _round_up(s.size, page)
               for s in sections if not s.nobits)
    if len(out) < want:
        out.extend(b"\0" * (want - len(out)))
    return bytes(out)


def _round_up(value: int, to: int) -> int:
    return (value + to - 1) & ~(to - 1) if to > 1 else value


__all__ = ["IMP_PREFIX", "PeError", "WINDOWS_IMPORTS", "executable",
           "header_space", "import_section", "imports_needed"]
