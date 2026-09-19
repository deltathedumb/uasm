"""One PE/COFF object from a whole module, for AArch64.

THE FOURTH OF THESE, and the pair to `x86_64/coffemit.py`: the sections and
symbols are the same shape as ELF's, and what changes is the relocation
vocabulary.

COFF'S ARM64 NUMBERS ARE NOT COFF'S x86-64 NUMBERS, and that is the thing to
get right here because they look like they should be. `1` is `ADDR32` on
AArch64 where it is `ADDR64` on x86-64, and AArch64's `ADDR64` is FOURTEEN.
A writer that assumed one table for the container would produce a file whose
every absolute reference is patched at the wrong width. The numbers below
were read back off objects `llvm-mc -triple aarch64-pc-windows-msvc`
assembled, which is available here and settles it without a Windows machine.

THE ADDEND IS WHERE ELF'S IS NOT. AArch64 packs its operands into
instruction bitfields, so COFF has nowhere to put an addend and neither do
the fields -- every relocation this encoder produces carries zero, as it
does for Mach-O, and a non-zero one is refused rather than dropped. See
`machoemit.py`, which says the same and for the same reason.
"""
from __future__ import annotations

from ...backend.objfile import (
    IMAGE_FILE_MACHINE_ARM64, IMAGE_SCN_CNT_CODE,
    IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_CNT_UNINITIALIZED_DATA,
    IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_READ, IMAGE_SCN_MEM_WRITE,
    CoffObject, CoffRelocation, CoffSymbol,
)
from ...ir import Module
from ...ir.module import Linkage
from .encode import (
    R_AARCH64_ABS64, R_AARCH64_ADD_ABS_LO12_NC, R_AARCH64_ADR_PREL_PG_HI21,
    R_AARCH64_CALL26, R_AARCH64_JUMP26, encode_function,
)
from .objemit import _is_directive

#: COFF relocation numbers for AArch64, from the PE specification and
#: confirmed against llvm-mc.
IMAGE_REL_ARM64_ADDR32 = 0x0001
IMAGE_REL_ARM64_ADDR32NB = 0x0002
IMAGE_REL_ARM64_BRANCH26 = 0x0003
IMAGE_REL_ARM64_PAGEBASE_REL21 = 0x0004
IMAGE_REL_ARM64_PAGEOFFSET_12A = 0x0006
IMAGE_REL_ARM64_PAGEOFFSET_12L = 0x0007
IMAGE_REL_ARM64_ADDR64 = 0x000E

#: What each of the encoder's relocations becomes. One number to one number,
#: with no addend arithmetic: see the module docstring.
_TRANSLATION = {
    R_AARCH64_CALL26: IMAGE_REL_ARM64_BRANCH26,
    R_AARCH64_JUMP26: IMAGE_REL_ARM64_BRANCH26,
    R_AARCH64_ADR_PREL_PG_HI21: IMAGE_REL_ARM64_PAGEBASE_REL21,
    R_AARCH64_ADD_ABS_LO12_NC: IMAGE_REL_ARM64_PAGEOFFSET_12A,
    R_AARCH64_ABS64: IMAGE_REL_ARM64_ADDR64,
}

_CODE = IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ
_DATA = (IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ
         | IMAGE_SCN_MEM_WRITE)
_BSS = (IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_MEM_READ
        | IMAGE_SCN_MEM_WRITE)


def _translate(kind: int, addend: int, symbol: str) -> int:
    """One ELF relocation as COFF spells it on AArch64."""
    try:
        number = _TRANSLATION[kind]
    except KeyError:
        raise NotImplementedError(
            f"no COFF spelling for ELF relocation {kind}; the AArch64 "
            f"encoder produced one this file has not been taught") from None
    if addend:
        # SEE THE MODULE DOCSTRING. There is nowhere to put it, and dropping
        # it would reach the right symbol at the wrong offset.
        raise NotImplementedError(
            f"relocation against {symbol!r} carries an addend of {addend}, "
            f"which COFF has no field for on AArch64")
    return number


def object_bytes(backend, module: Module, abi, dialect) -> bytes:
    """The module as one relocatable PE/COFF object."""
    obj = CoffObject(IMAGE_FILE_MACHINE_ARM64)
    text = bytearray()
    text_relocs: list[CoffRelocation] = []
    symbols: list[CoffSymbol] = []

    for fn in module.defined_functions():
        lines = backend._function(fn, abi, dialect)
        enc = encode_function([one for one in lines if not _is_directive(one)])
        base = len(text)
        text += enc.code
        symbols.append(CoffSymbol(
            name=backend.symbol(fn.name, dialect), section=".text",
            value=base, size=len(enc.code),
            binding=(1 if fn.linkage is Linkage.EXPORT else 0), kind=2))
        for at, sym, kind, addend in enc.relocs:
            text_relocs.append(CoffRelocation(
                offset=base + at, symbol=sym,
                kind=_translate(kind, addend, sym), addend=0))

    obj.section(".text", bytes(text), characteristics=_CODE, align=4)

    data = bytearray()
    bss = 0
    for g in module.globals:
        name = backend.global_symbol(g.name, dialect)
        binding = 1 if g.linkage is Linkage.EXPORT else 0
        align = g.align or 8
        if g.data is None:
            bss = (bss + align - 1) & ~(align - 1)
            symbols.append(CoffSymbol(name=name, section=".bss", value=bss,
                                      size=max(1, g.size), binding=binding))
            bss += max(1, g.size)
        else:
            at = (len(data) + align - 1) & ~(align - 1)
            data += b"\0" * (at - len(data))
            symbols.append(CoffSymbol(name=name, section=".data", value=at,
                                      size=len(g.data), binding=binding))
            data += bytes(g.data)
    if data:
        obj.section(".data", bytes(data), characteristics=_DATA, align=8)
    if bss:
        obj.section(".bss", b"", characteristics=_BSS, align=8,
                    size_override=bss)

    # UNDEFINED SYMBOLS LAST, and only what nothing here defines. COFF names
    # a relocation's symbol by INDEX, so one with no record at all is not a
    # missing-symbol message but an index into whatever happens to be there.
    defined = {s.name for s in symbols}
    for rel in text_relocs:
        if rel.symbol not in defined:
            defined.add(rel.symbol)
            symbols.append(CoffSymbol(name=rel.symbol, section="", binding=1))

    # THE SYMBOLS BEFORE THE RELOCATIONS, because the writer resolves a
    # relocation's symbol to its index and refuses one it cannot find.
    for sym in symbols:
        obj.symbol(sym)
    for rel in text_relocs:
        obj.relocate(".text", rel)
    return obj.to_bytes()
