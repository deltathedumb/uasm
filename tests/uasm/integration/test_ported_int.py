"""The integer cell, ported to the machine subset.

STAGE 3 OF `docs/INERT-RUNTIME.md`: the first piece of uasm's object
runtime that is not C. `src/uasm/runtime/int_cell.py` is compiled by
uasm's own frontend and spliced into every program that reaches the
runtime, and the C definitions it replaces are turned into declarations so the
two cannot both be linked.

WHAT THESE TESTS ARE FOR, in order of how much they matter:

1. THE LAYOUT CANNOT DRIFT. A cell the subset builds is read by 15,000 lines of
   C that were not told anything changed. The offsets in `int_cell.py` are
   therefore compiled out of that C and compared, rather than trusted.
2. THE SWITCH IS COHERENT. A name declared ported that is not defined, or
   defined and not omitted from the C, is a duplicate symbol at LINK time --
   which names an object file rather than the list that was wrong.
3. THE BEHAVIOUR IS CPYTHON'S. Small-integer sharing is observable, and the
   ported cache has to share exactly what CPython shares, including at the
   boundary: 256 is shared and 257 is not, and a program can see which.
4. NOTHING ELSE PAYS. A statically typed program must not acquire the object
   runtime because the splice exists.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from tests import harness
from tests.harness import snapshot

SRC = snapshot.current(Path(__file__).resolve().parents[3])


def _cli(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run([sys.executable, "-m", "uasm", *args],
                          capture_output=True, text=True, env=env)


def write(tmp_path: Path, source: str, name: str = "prog.py") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def build_and_run(tmp_path: Path, source: Path) -> subprocess.CompletedProcess:
    # `--link` BECAUSE `build` STOPS AT AN UNLINKED OBJECT. Every test that
    # comes through here observes the ported runtime by RUNNING a program and
    # reading what it printed, so the object has to be linked before the
    # exec below has anything it is allowed to run.
    out = tmp_path / "prog.exe"
    built = _cli("build", str(source), "--backend", "c", "--link",
                 "-o", str(out), "--workdir", str(tmp_path / "wd"))
    assert built.returncode == 0, built.stdout + built.stderr
    return subprocess.run([str(out)], capture_output=True, text=True)


#: probe name -> the C expression that computes it. Everything the ported
#: runtime believes about the object cell's layout is asked for here.
_ASK = {
    "obj_size": "sizeof(apy_obj)",
    "value_size": "sizeof(apy_value)",
    "kind_offset": "offsetof(apy_obj, kind)",
    "payload_offset": "offsetof(apy_obj, v.i)",
    "str_ptr": "offsetof(apy_obj, v.s.p)",
    "str_len": "offsetof(apy_obj, v.s.n)",
    "str_mut": "offsetof(apy_obj, v.s.mut)",
    "q_items": "offsetof(apy_obj, v.q.items)",
    "q_n": "offsetof(apy_obj, v.q.n)",
    "q_cap": "offsetof(apy_obj, v.q.cap)",
    "cell_slot": "offsetof(apy_obj, v.cell.slot)",
    "slice_start": "offsetof(apy_obj, v.sl.start)",
    "slice_stop": "offsetof(apy_obj, v.sl.stop)",
    "slice_step": "offsetof(apy_obj, v.sl.step)",
    "prop_get": "offsetof(apy_obj, v.p.get)",
    "prop_set": "offsetof(apy_obj, v.p.set)",
    "prop_del": "offsetof(apy_obj, v.p.del_)",
    "prop_kind": "offsetof(apy_obj, v.p.kind)",
    "float_at": "offsetof(apy_obj, v.f)",
    "complex_re": "offsetof(apy_obj, v.z.re)",
    "complex_im": "offsetof(apy_obj, v.z.im)",
    "fn_code": "offsetof(apy_obj, v.fn.code)",
    "fn_name": "offsetof(apy_obj, v.fn.name)",
    "fn_vararg": "offsetof(apy_obj, v.fn.vararg)",
    "g_step": "offsetof(apy_obj, v.g.step)",
    "g_cache": "offsetof(apy_obj, v.g.cache)",
    "pos_row": "sizeof(apy_srcpos)",
    "pos_fn": "offsetof(apy_srcpos, fn)",
    "pos_line": "offsetof(apy_srcpos, line)",
    "pos_end_line": "offsetof(apy_srcpos, end_line)",
    "pos_col": "offsetof(apy_srcpos, col)",
    "pos_end_col": "offsetof(apy_srcpos, end_col)",
    "d_keys": "offsetof(apy_obj, v.d.keys)",
    "d_vals": "offsetof(apy_obj, v.d.vals)",
    "d_n": "offsetof(apy_obj, v.d.n)",
    "d_cap": "offsetof(apy_obj, v.d.cap)",
    "d_ro": "offsetof(apy_obj, v.d.ro)",
    "o_cls": "offsetof(apy_obj, v.o.cls)",
    "o_held": "offsetof(apy_obj, v.o.held)",
    "o_dict": "offsetof(apy_obj, v.o.dict)",
    "t_name": "offsetof(apy_obj, v.t.name)",
    "e_name": "offsetof(apy_obj, v.e.name)",
    "it_mode": "offsetof(apy_obj, v.it.mode)",
    "ga_origin": "offsetof(apy_obj, v.ga.origin)",
    "vw_part": "offsetof(apy_obj, v.vw.part)",
    "vw_dict": "offsetof(apy_obj, v.vw.dict)",
    "rg_start": "offsetof(apy_obj, v.rg.start)",
    "rg_stop": "offsetof(apy_obj, v.rg.stop)",
    "rg_step": "offsetof(apy_obj, v.rg.step)",
    "sup_from": "offsetof(apy_obj, v.sup.from)",
    "sup_self": "offsetof(apy_obj, v.sup.self)",
    "mv_src": "offsetof(apy_obj, v.mv.src)",
    "mv_off": "offsetof(apy_obj, v.mv.off)",
    "mv_n": "offsetof(apy_obj, v.mv.n)",
    "mv_step": "offsetof(apy_obj, v.mv.step)",
    "t_base": "offsetof(apy_obj, v.t.base)",
    "t_dict": "offsetof(apy_obj, v.t.dict)",
    "t_mro": "offsetof(apy_obj, v.t.mro)",
    "t_qual": "offsetof(apy_obj, v.t.qual)",
    "fn_native": "offsetof(apy_obj, v.fn.native)",
    "nat_kind": "(size_t)APY_NAT_KIND",
    "nat_builtin_init": "(size_t)APY_NAT_BUILTIN_INIT",
    "nat_builtin_new": "(size_t)APY_NAT_BUILTIN_NEW",
    "nat_type_new": "(size_t)APY_NAT_TYPE_NEW",
    "nat_type_init": "(size_t)APY_NAT_TYPE_INIT",
    "nat_type_call": "(size_t)APY_NAT_TYPE_CALL",
    # THE ENUM'S LAST MEMBER PLUS ONE, which is how the C sizes the cache
    # itself -- `made[APY_NAT_OBJ_ONLY + 1]` in `apy_native`. It has to be
    # spelled with whatever is last, so appending a selector means changing
    # it here too; that is the point of the check, since the IR writes the
    # same count out by hand in `makers.py`.
    "nat_count": "(size_t)(APY_NAT_OBJ_ONLY + 1)",
    "nat_has_default": "(size_t)APY_NAT_HAS_DEFAULT",
    "nat_tg_enter": "(size_t)APY_NAT_TG_ENTER",
    "nat_tg_exit": "(size_t)APY_NAT_TG_EXIT",
    "nat_tg_create": "(size_t)APY_NAT_TG_CREATE",
    "nat_positions": "(size_t)APY_NAT_POSITIONS",
    "uc_alpha": "(size_t)APY_UC_ALPHA",
    "uc_decimal": "(size_t)APY_UC_DECIMAL",
    "uc_digit": "(size_t)APY_UC_DIGIT",
    "uc_numeric": "(size_t)APY_UC_NUMERIC",
    "uc_lower": "(size_t)APY_UC_LOWER",
    "uc_upper": "(size_t)APY_UC_UPPER",
    "uc_title": "(size_t)APY_UC_TITLE",
    "uc_space": "(size_t)APY_UC_SPACE",
    "uc_printable": "(size_t)APY_UC_PRINTABLE",
    "uc_xidstart": "(size_t)APY_UC_XIDSTART",
    "uc_xidcont": "(size_t)APY_UC_XIDCONT",
    "nat_init": "(size_t)APY_NAT_INIT",
    "nat_new": "(size_t)APY_NAT_NEW",
    "nat_repr": "(size_t)APY_NAT_REPR",
    "nat_str": "(size_t)APY_NAT_STR",
    "nat_eq": "(size_t)APY_NAT_EQ",
    "nat_ne": "(size_t)APY_NAT_NE",
    "nat_hash": "(size_t)APY_NAT_HASH",
    "nat_getattr": "(size_t)APY_NAT_GETATTR",
    "nat_setattr": "(size_t)APY_NAT_SETATTR",
    "nat_delattr": "(size_t)APY_NAT_DELATTR",
    "nat_init_subclass": "(size_t)APY_NAT_INIT_SUBCLASS",
    "g_builtin": "offsetof(apy_obj, v.g.builtin)",
    "g_cancel": "offsetof(apy_obj, v.g.cancel)",
    "coro_task": "(size_t)APY_CORO_TASK",
    "coro_gather": "(size_t)APY_CORO_GATHER",
    "coro_anext": "(size_t)APY_CORO_ANEXT",
    "it_src": "offsetof(apy_obj, v.it.src)",
    "it_fn": "offsetof(apy_obj, v.it.fn)",
    "it_i": "offsetof(apy_obj, v.it.i)",
    "it_n0": "offsetof(apy_obj, v.it.n0)",
    "it_plain": "(size_t)APY_IT_PLAIN",
    "fn_is_type": "offsetof(apy_obj, v.fn.is_type)",
    "fn_dict": "offsetof(apy_obj, v.fn.dict)",
    "fn_bound": "offsetof(apy_obj, v.fn.bound)",
    "big_neg": "offsetof(apy_obj, v.big.neg)",
    "big_n": "offsetof(apy_obj, v.big.n)",
    "limb_size": "sizeof(apy_limb)",
    "big_limb": "offsetof(apy_obj, v.big.limb)",
    "fn_span": "sizeof(((apy_obj *)0)->v.fn)",
    "g_coro": "offsetof(apy_obj, v.g.coro)",
    "g_agen": "offsetof(apy_obj, v.g.agen)",
    "s_mut": "offsetof(apy_obj, v.s.mut)",
    "e_dict": "offsetof(apy_obj, v.e.dict)",
    "e_cls": "offsetof(apy_obj, v.e.cls)",
    "e_pos": "offsetof(apy_obj, v.e.pos)",
    "e_subs": "offsetof(apy_obj, v.e.subs)",
    "e_rendered": "offsetof(apy_obj, v.e.rendered)",
    "e_arg": "offsetof(apy_obj, v.e.arg)",
    "e_argv": "offsetof(apy_obj, v.e.argv)",
    "sl_start": "offsetof(apy_obj, v.sl.start)",
    "sl_stop": "offsetof(apy_obj, v.sl.stop)",
    "sl_step": "offsetof(apy_obj, v.sl.step)",
    "e_has_arg": "offsetof(apy_obj, v.e.has_arg)",
    "e_context": "offsetof(apy_obj, v.e.context)",
    "e_cause": "offsetof(apy_obj, v.e.cause)",
    "e_suppress": "offsetof(apy_obj, v.e.suppress)",
    "g_running": "offsetof(apy_obj, v.g.running)",
    "e_notes": "offsetof(apy_obj, v.e.notes)",
    "t_meta": "offsetof(apy_obj, v.t.meta)",
    "it_map": "(size_t)APY_IT_MAP",
    "it_filter": "(size_t)APY_IT_FILTER",
    "it_enumerate": "(size_t)APY_IT_ENUMERATE",
    "it_zip": "(size_t)APY_IT_ZIP",
    "part_keys": "(size_t)APY_PART_KEYS",
    "part_values": "(size_t)APY_PART_VALUES",
    "part_items": "(size_t)APY_PART_ITEMS",
    "it_rev": "(size_t)APY_IT_REV",
    "it_viewed": "(size_t)APY_IT_VIEWED",
    "it_callable": "(size_t)APY_IT_CALLABLE",
    "it_revof": "(size_t)APY_IT_REVOF",
    "it_named": "offsetof(apy_obj, v.it.named)",
    "prop_classmethod": "(size_t)APY_PROP_CLASSMETHOD",
    "prop_staticmethod": "(size_t)APY_PROP_STATICMETHOD",
    "fn_cells": "offsetof(apy_obj, v.fn.cells)",
    "fn_ncells": "offsetof(apy_obj, v.fn.ncells)",
    "fn_defaults": "offsetof(apy_obj, v.fn.defaults)",
    "fn_ndefaults": "offsetof(apy_obj, v.fn.ndefaults)",
    "fn_nkwdefault": "offsetof(apy_obj, v.fn.nkwdefault)",
    "fn_kwarg": "offsetof(apy_obj, v.fn.kwarg)",
    "fn_kwonly": "offsetof(apy_obj, v.fn.kwonly)",
    "fn_posonly": "offsetof(apy_obj, v.fn.posonly)",
    "fn_doc": "offsetof(apy_obj, v.fn.doc)",
    "fn_builtin": "offsetof(apy_obj, v.fn.builtin)",
    "fn_coro": "offsetof(apy_obj, v.fn.coro)",
    "fn_qualname": "offsetof(apy_obj, v.fn.qualname)",
    "fn_annotate": "offsetof(apy_obj, v.fn.annotate)",
    "fn_module": "offsetof(apy_obj, v.fn.module)",
    "fn_descr": "offsetof(apy_obj, v.fn.descr)",
    "g_sent": "offsetof(apy_obj, v.g.sent)",
    "g_slots": "offsetof(apy_obj, v.g.slots)",
    "g_result": "offsetof(apy_obj, v.g.result)",
    "g_pending": "offsetof(apy_obj, v.g.pending)",
    "g_n": "offsetof(apy_obj, v.g.n)",
    "g_state": "offsetof(apy_obj, v.g.state)",
    "g_coro": "offsetof(apy_obj, v.g.coro)",
    "g_agen": "offsetof(apy_obj, v.g.agen)",
    "ga_origin": "offsetof(apy_obj, v.ga.origin)",
    "ga_args": "offsetof(apy_obj, v.ga.args)",
    "t_builtin": "offsetof(apy_obj, v.t.builtin)",
    "fn_arity": "offsetof(apy_obj, v.fn.arity)",
    "fn_pnames": "offsetof(apy_obj, v.fn.pnames)",
}

#: probe name -> the enum member, which only the C compiler knows. The enum
#: numbers ONE of its twenty-nine members explicitly and positions the rest,
#: so reading these off by eye is exactly how they go wrong -- and a wrong
#: kind builds a perfectly formed cell of the wrong TYPE, which nothing
#: crashes on.
_KINDS = {
    "int_k": "INT", "str_k": "STR", "bytes_k": "BYTES", "list_k": "LIST",
    "tuple_k": "TUPLE", "none_k": "NONE", "bool_k": "BOOL",
    "ellipsis_k": "ELLIPSIS", "notimpl_k": "NOTIMPL", "cell_k": "CELL",
    "dict_k": "DICT", "inst_k": "INST", "func_k": "FUNC", "gen_k": "GEN",
    "alias_k": "ALIAS", "type_k": "TYPE",
    "slice_k": "SLICE", "prop_k": "PROP",
    "float_k": "FLOAT", "complex_k": "COMPLEX",
    "set_k": "SET", "frozen_k": "FROZEN",
    "big_k": "BIG", "iter_k": "ITER", "mview_k": "MVIEW",
    "range_k": "RANGE", "super_k": "SUPER", "view_k": "VIEW",
}

#: probe name -> the (module, one-line constant) in `runtime/` that must
#: agree with it. ADDING A PORT MEANS ADDING A LINE HERE, and nothing else:
#: the probe, the comparison and the failure message all follow from this.
WANT = {
    "obj_size": ("int_cell.py", "apy_obj_size"),
    "kind_offset": ("int_cell.py", "apy_kind_offset"),
    "payload_offset": ("int_cell.py", "apy_payload_offset"),
    "int_k": ("int_cell.py", "apy_int_kind"),
    "str_k": ("str_cell.py", "apy_str_kind"),
    "bytes_k": ("str_cell.py", "apy_bytes_kind"),
    "str_ptr": ("str_cell.py", "apy_str_ptr_offset"),
    "str_len": ("str_cell.py", "apy_str_len_offset"),
    "str_mut": ("str_cell.py", "apy_str_mut_offset"),
    "list_k": ("list_cell.py", "apy_list_kind"),
    "tuple_k": ("list_cell.py", "apy_tuple_kind"),
    "q_items": ("list_cell.py", "apy_q_items_offset"),
    "q_n": ("list_cell.py", "apy_q_n_offset"),
    "q_cap": ("list_cell.py", "apy_q_cap_offset"),
    "slice_k": ("slots.py", "apy_slice_kind"),
    "prop_k": ("slots.py", "apy_prop_kind"),
    "slice_start": ("slots.py", "apy_slice_start_offset"),
    "slice_stop": ("slots.py", "apy_slice_stop_offset"),
    "slice_step": ("slots.py", "apy_slice_step_offset"),
    "prop_get": ("slots.py", "apy_prop_get_offset"),
    "prop_set": ("slots.py", "apy_prop_set_offset"),
    "prop_del": ("slots.py", "apy_prop_del_offset"),
    "prop_kind": ("slots.py", "apy_prop_kind_offset"),
    "float_k": ("floats.py", "apy_float_kind"),
    "complex_k": ("floats.py", "apy_complex_kind"),
    "float_at": ("floats.py", "apy_float_offset"),
    "complex_re": ("floats.py", "apy_complex_re_offset"),
    "complex_im": ("floats.py", "apy_complex_im_offset"),
    "fn_code": ("makers.py", "apy_fn_code_offset"),
    "fn_name": ("makers.py", "apy_fn_name_offset"),
    "fn_vararg": ("makers.py", "apy_fn_vararg_offset"),
    "g_step": ("makers.py", "apy_g_step_offset"),
    "g_cache": ("makers.py", "apy_g_cache_offset"),
    "pos_row": ("errstate.py", "apy_pos_row_size"),
    "pos_fn": ("errstate.py", "apy_pos_fn_offset"),
    "pos_line": ("errstate.py", "apy_pos_line_offset"),
    "pos_end_line": ("errstate.py", "apy_pos_end_line_offset"),
    "pos_col": ("errstate.py", "apy_pos_col_offset"),
    "pos_end_col": ("errstate.py", "apy_pos_end_col_offset"),
    "set_k": ("containers.py", "apy_set_kind"),
    "frozen_k": ("containers.py", "apy_frozen_kind"),
    "d_keys": ("containers.py", "apy_d_keys_offset"),
    "d_vals": ("containers.py", "apy_d_vals_offset"),
    "d_n": ("containers.py", "apy_d_n_offset"),
    "d_cap": ("containers.py", "apy_d_cap_offset"),
    "big_k": ("kindname.py", "apy_big_kind"),
    "iter_k": ("kindname.py", "apy_iter_kind"),
    "mview_k": ("kindname.py", "apy_mview_kind"),
    "range_k": ("kindname.py", "apy_range_kind"),
    "super_k": ("kindname.py", "apy_super_kind"),
    "view_k": ("kindname.py", "apy_view_kind"),
    "o_cls": ("kindname.py", "apy_o_cls_offset"),
    "o_held": ("kindname.py", "apy_o_held_offset"),
    "o_dict": ("kindname.py", "apy_o_dict_offset"),
    "t_name": ("kindname.py", "apy_t_name_offset"),
    "e_name": ("kindname.py", "apy_e_name_offset"),
    "it_mode": ("kindname.py", "apy_it_mode_offset"),
    "ga_origin": ("kindname.py", "apy_ga_origin_offset"),
    "vw_part": ("kindname.py", "apy_vw_part_offset"),
    "vw_dict": ("kindname.py", "apy_vw_dict_offset"),
    "rg_start": ("makers.py", "apy_rg_start_offset"),
    "rg_stop": ("makers.py", "apy_rg_stop_offset"),
    "rg_step": ("makers.py", "apy_rg_step_offset"),
    "sup_from": ("makers.py", "apy_sup_from_offset"),
    "sup_self": ("makers.py", "apy_sup_self_offset"),
    "mv_src": ("makers.py", "apy_mv_src_offset"),
    "mv_off": ("makers.py", "apy_mv_off_offset"),
    "mv_n": ("makers.py", "apy_mv_n_offset"),
    "mv_step": ("makers.py", "apy_mv_step_offset"),
    "t_base": ("makers.py", "apy_t_base_offset"),
    "t_dict": ("makers.py", "apy_t_dict_offset"),
    "t_mro": ("makers.py", "apy_t_mro_offset"),
    "d_ro": ("containers.py", "apy_d_ro_offset"),
    "t_qual": ("makers.py", "apy_t_qual_offset"),
    "fn_native": ("makers.py", "apy_fn_native_offset"),
    "nat_kind": ("makers.py", "apy_nat_kind"),
    "nat_builtin_init": ("makers.py", "apy_nat_builtin_init"),
    "nat_builtin_new": ("makers.py", "apy_nat_builtin_new"),
    "nat_type_new": ("makers.py", "apy_nat_type_new"),
    "nat_type_init": ("makers.py", "apy_nat_type_init"),
    "nat_type_call": ("makers.py", "apy_nat_type_call"),
    "nat_count": ("makers.py", "apy_nat_count"),
    "nat_has_default": ("makers.py", "apy_nat_has_default"),
    "nat_tg_enter": ("tasks.py", "apy_nat_tg_enter"),
    "nat_tg_exit": ("tasks.py", "apy_nat_tg_exit"),
    "nat_tg_create": ("tasks.py", "apy_nat_tg_create"),
    "nat_positions": ("errstate.py", "apy_nat_positions"),
    "uc_alpha": ("charclass.py", "apy_uc_alpha"),
    "uc_decimal": ("charclass.py", "apy_uc_decimal"),
    "uc_digit": ("charclass.py", "apy_uc_digit"),
    "uc_numeric": ("charclass.py", "apy_uc_numeric"),
    "uc_lower": ("charclass.py", "apy_uc_lower"),
    "uc_upper": ("charclass.py", "apy_uc_upper"),
    "uc_title": ("charclass.py", "apy_uc_title"),
    "uc_space": ("charclass.py", "apy_uc_space"),
    "uc_printable": ("charclass.py", "apy_uc_printable"),
    "uc_xidstart": ("charclass.py", "apy_uc_xidstart"),
    "uc_xidcont": ("charclass.py", "apy_uc_xidcont"),
    "nat_init": ("slots.py", "apy_nat_init"),
    "nat_new": ("slots.py", "apy_nat_new"),
    "nat_repr": ("slots.py", "apy_nat_repr"),
    "nat_str": ("slots.py", "apy_nat_str"),
    "nat_eq": ("slots.py", "apy_nat_eq"),
    "nat_ne": ("slots.py", "apy_nat_ne"),
    "nat_hash": ("slots.py", "apy_nat_hash"),
    "nat_getattr": ("slots.py", "apy_nat_getattr"),
    "nat_setattr": ("slots.py", "apy_nat_setattr"),
    "nat_delattr": ("slots.py", "apy_nat_delattr"),
    "nat_init_subclass": ("slots.py", "apy_nat_init_subclass"),
    "g_builtin": ("tasks.py", "apy_g_builtin_offset"),
    "g_cancel": ("tasks.py", "apy_g_cancel_offset"),
    "coro_task": ("tasks.py", "apy_coro_task"),
    "coro_gather": ("tasks.py", "apy_coro_gather"),
    "coro_anext": ("tasks.py", "apy_coro_anext"),
    "it_src": ("cursor.py", "apy_it_src_offset"),
    "it_fn": ("cursor.py", "apy_it_fn_offset"),
    "it_i": ("cursor.py", "apy_it_i_offset"),
    "it_plain": ("cursor.py", "apy_it_plain"),
    "it_n0": ("cursor.py", "apy_it_n0_offset"),
    "fn_is_type": ("kindname.py", "apy_fn_is_type_offset"),
    "fn_dict": ("slots.py", "apy_fn_dict_offset"),
    "fn_bound": ("kindname.py", "apy_fn_bound_offset"),
    "big_neg": ("kindname.py", "apy_big_neg_offset"),
    "big_n": ("mathints.py", "apy_big_n_offset"),
    "limb_size": ("mathints.py", "apy_limb_size"),
    "big_limb": ("mathints.py", "apy_big_limb_offset"),
    "fn_span": ("kindname.py", "apy_fn_span"),
    "g_coro": ("kindname.py", "apy_g_coro_offset2"),
    "g_agen": ("kindname.py", "apy_g_agen_offset2"),
    "s_mut": ("kindname.py", "apy_s_mut_offset"),
    "it_map": ("kindname.py", "apy_it_map"),
    "it_filter": ("kindname.py", "apy_it_filter"),
    "it_enumerate": ("kindname.py", "apy_it_enumerate"),
    "it_zip": ("kindname.py", "apy_it_zip"),
    "part_keys": ("kindname.py", "apy_part_keys"),
    "part_values": ("kindname.py", "apy_part_values"),
    "part_items": ("kindname.py", "apy_part_items"),
    "it_rev": ("kindname.py", "apy_it_rev"),
    "it_viewed": ("kindname.py", "apy_it_viewed"),
    "it_callable": ("kindname.py", "apy_it_callable"),
    "it_revof": ("kindname.py", "apy_it_revof"),
    "it_named": ("kindname.py", "apy_it_named_offset"),
    "prop_classmethod": ("kindname.py", "apy_prop_classmethod"),
    "prop_staticmethod": ("kindname.py", "apy_prop_staticmethod"),
    "e_dict": ("excval.py", "apy_e_dict_offset"),
    "e_cls": ("excval.py", "apy_e_cls_offset"),
    "e_pos": ("excval.py", "apy_e_pos_offset"),
    "e_subs": ("excval.py", "apy_e_subs_offset"),
    "e_rendered": ("excval.py", "apy_e_rendered_offset"),
    "e_arg": ("excval.py", "apy_e_arg_offset"),
    "e_argv": ("text_exc.py", "apy_e_argv_offset"),
    "sl_start": ("slicing.py", "apy_sl_start_offset"),
    "sl_stop": ("slicing.py", "apy_sl_stop_offset"),
    "sl_step": ("slicing.py", "apy_sl_step_offset"),
    "e_has_arg": ("excval.py", "apy_e_has_arg_offset"),
    "e_context": ("excval.py", "apy_e_context_offset"),
    "e_cause": ("excval.py", "apy_e_cause_offset"),
    "e_suppress": ("excval.py", "apy_e_suppress_offset"),
    "g_running": ("gens.py", "apy_g_running_offset"),
    "e_notes": ("excval.py", "apy_e_notes_offset"),
    "t_meta": ("excval.py", "apy_t_meta_offset"),
    "value_size": ("list_cell.py", "apy_value_size"),
    "none_k": ("singletons.py", "apy_none_kind"),
    "bool_k": ("singletons.py", "apy_bool_kind"),
    "ellipsis_k": ("singletons.py", "apy_ellipsis_kind"),
    "notimpl_k": ("singletons.py", "apy_notimpl_kind"),
    "cell_k": ("cells.py", "apy_cell_kind"),
    "cell_slot": ("cells.py", "apy_cell_slot_offset"),
    "dict_k": ("kinds.py", "apy_dict_kind"),
    "inst_k": ("kinds.py", "apy_inst_kind"),
    "func_k": ("funcs.py", "apy_func_kind"),
    "fn_cells": ("funcs.py", "apy_fn_cells_offset"),
    "fn_ncells": ("funcs.py", "apy_fn_ncells_offset"),
    "fn_defaults": ("funcs.py", "apy_fn_defaults_offset"),
    "fn_ndefaults": ("funcs.py", "apy_fn_ndefaults_offset"),
    "fn_nkwdefault": ("funcs.py", "apy_fn_nkwdefault_offset"),
    "fn_kwarg": ("funcs.py", "apy_fn_kwarg_offset"),
    "fn_kwonly": ("funcs.py", "apy_fn_kwonly_offset"),
    "fn_posonly": ("funcs.py", "apy_fn_posonly_offset"),
    "fn_doc": ("funcs.py", "apy_fn_doc_offset"),
    "fn_builtin": ("funcs.py", "apy_fn_builtin_offset"),
    "fn_coro": ("funcs.py", "apy_fn_coro_offset"),
    "fn_qualname": ("funcs.py", "apy_fn_qualname_offset"),
    "fn_annotate": ("funcs.py", "apy_fn_annotate_offset"),
    "fn_module": ("funcs.py", "apy_fn_module_offset"),
    "fn_descr": ("funcs.py", "apy_fn_descr_offset"),
    "gen_k": ("gens.py", "apy_gen_kind"),
    "g_sent": ("gens.py", "apy_g_sent_offset"),
    "g_slots": ("gens.py", "apy_g_slots_offset"),
    "g_result": ("gens.py", "apy_g_result_offset"),
    "g_pending": ("gens.py", "apy_g_pending_offset"),
    "g_n": ("gens.py", "apy_g_n_offset"),
    "g_state": ("gens.py", "apy_g_state_offset"),
    "g_coro": ("gens.py", "apy_g_coro_offset"),
    "g_agen": ("gens.py", "apy_g_agen_offset"),
    "alias_k": ("alias.py", "apy_alias_kind"),
    "type_k": ("alias.py", "apy_type_kind"),
    "ga_origin": ("alias.py", "apy_ga_origin_offset"),
    "ga_args": ("alias.py", "apy_ga_args_offset"),
    "t_builtin": ("alias.py", "apy_t_builtin_offset"),
    "fn_arity": ("funcs.py", "apy_fn_arity_offset"),
    "fn_pnames": ("funcs.py", "apy_fn_pnames_offset"),
}


class TestTheLayoutIsTheCs:
    """Compiled out of the C, not written down twice."""

    @harness.needs("gcc")
    def test_the_offsets_match_the_struct(self, tmp_path):
        """Every layout constant the ported runtime assumes, against the C.

        A PROBE RATHER THAN A PARSE. The union has twenty arms and its size
        follows from padding rules no reader should have to apply; asking the
        compiler is the only answer that is right by construction.

        NAME-KEYED, AND IT DID NOT USED TO BE. The probe printed a row of bare
        numbers and this side unpacked them positionally, so adding
        `v.cell.slot` in the wrong group shifted everything after it and the
        test failed complaining about `APY_INT_K` -- an edit to the closure
        cell reported as a problem with integers. There are forty-six
        constants now; one row per name costs nothing, and a new one is a line
        in `WANT` that cannot disturb its neighbours.

        WHAT A WRONG VALUE DOES, which is why this machinery earns its place:
        a wrong OFFSET crashes or returns rubbish, and someone investigates. A
        wrong KIND builds a perfectly formed cell of the wrong type. A wrong
        WIDTH writes over the field next door -- `builtin` and `coro` are
        adjacent in the function arm, so eight bytes into a four-byte flag
        would make every builtin a coroutine.
        """
        sys.path.insert(0, str(SRC))
        try:
            from uasm.objects.support import runtime_c
        finally:
            del sys.path[0]

        # `esc` is the two characters a C string needs for a newline; the
        # joins use a real one. Built with `chr` rather than written out,
        # because a `\n` here has to survive being a Python literal on its
        # way into a C literal, and getting that wrong puts a real newline
        # inside a C string -- which fails as "missing terminating quote"
        # several thousand lines from the edit that caused it.
        esc = chr(92) + "n"
        lines = [f'    printf("{name} %zu{esc}", (size_t)({expr}));'
                 for name, expr in _ASK.items()]
        lines += [f'    printf("{name} %d{esc}", (int)APY_{k}_K);'
                  for name, k in _KINDS.items()]
        body = "\n".join(lines)

        probe = runtime_c(entry="probe_entry")
        probe = probe.replace("extern int64_t probe_entry(void);", "")
        probe = probe.replace(
            "int main(void) { return (int)probe_entry(); }",
            "#include <stddef.h>\nint main(void) {\n" + body
            + "\n    return 0;\n}\n")
        # `runtime_c()` with no module keeps every C definition -- the switch
        # is per-build and this build has no IR to stand aside for -- so the
        # probe compiles the runtime exactly as an unported program would.
        c = tmp_path / "probe.c"
        c.write_text(probe, encoding="utf-8")
        exe = tmp_path / "probe.exe"
        # -lm -ldl: the runtime probe calls libm (floor, fmod, ...) and libdl
        # unconditionally, needed on every ELF host -- see toolchains.py.
        system_libs = [] if sys.platform == "win32" else ["-lm", "-ldl"]
        built = subprocess.run(["gcc", "-w", str(c), "-o", str(exe), *system_libs],
                               capture_output=True, text=True)
        assert built.returncode == 0, built.stderr[-3000:]
        out = subprocess.run([str(exe)], capture_output=True, text=True).stdout
        got = {ln.split()[0]: int(ln.split()[1])
               for ln in out.splitlines() if ln.strip()}

        where = Path(SRC) / "uasm" / "runtime"

        def literal(module: str, fn: str) -> int:
            source = (where / module).read_text(encoding="utf-8")
            m = re.search(rf"def {fn}\(\) -> i64:\n    return (-?\d+)",
                          source)
            assert m, f"{fn} is not a one-line constant in {module} any more"
            return int(m.group(1))

        for key, (module, fn) in WANT.items():
            assert key in got, f"the probe did not report {key!r}"
            assert literal(module, fn) == got[key], (
                f"{module}:{fn}() says {literal(module, fn)}, "
                f"the C says {key} is {got[key]}")

        # THE RESERVED SINGLETONS, which `reserve()` forces to be literals
        # (E0018: a reservation is laid out before any code runs). They
        # therefore duplicate `apy_obj_size`, and this is the only thing
        # keeping the two in step -- a cell reserved too small overlaps the
        # tag of the one after it.
        reserved = re.findall(
            r'reserve\("apy_\w+_cell_ir", (\d+)\)',
            (where / "singletons.py").read_text(encoding="utf-8"))
        assert reserved, "the singleton reservations changed shape"
        assert {int(x) for x in reserved} == {got["obj_size"]}, (
            f"a singleton is reserved {set(reserved)} bytes and the C struct "
            f"is {got['obj_size']}")


class TestTheSwitchIsCoherent:
    """A half-applied port is a link error naming the wrong thing."""

    def test_every_runtime_file_is_named_by_a_table(self):
        """A file in `runtime/` that no table names is not compiled at all.

        `sources()` IS THE TABLES -- `set(REPLACES) | set(SPLITS)` -- so an
        unkeyed module is never handed to the frontend, never checked by
        `check()`, and never reaches a build. It just sits in the directory
        looking ported, and the first thing that calls into it fails with
        `call to unknown function`.

        THIS IS NOT HYPOTHETICAL: `unicode_table.py` landed with a generator,
        a drift ratchet comparing it to the C row for row, and no key -- so
        every one of those passed while nothing compiled the file. A ratchet
        over the CONTENTS cannot notice that the contents are unreachable.

        AN EMPTY TUPLE IS A FINE ANSWER, and means the module displaces no C
        definition. The key is what matters; the names are a separate claim.
        """
        from uasm.objects.ir import REPLACES, SPLITS, SOURCE_DIR

        keyed = set(REPLACES) | set(SPLITS)
        on_disk = {p.name for p in SOURCE_DIR.glob("*.py")
                   if p.name != "__init__.py"}
        assert not (on_disk - keyed), (
            "these runtime modules are not named by REPLACES or SPLITS, so "
            f"nothing compiles them: {sorted(on_disk - keyed)}")
        assert not (keyed - on_disk), (
            f"these keys name no file: {sorted(keyed - on_disk)}")

    def test_the_generated_table_holds_only_generated_things(self):
        """`unicode_table.py` says GENERATED at the top and means it.

        ANYTHING HAND-WRITTEN IN THERE IS ONE REGENERATION AWAY FROM BEING
        GONE, with nothing to say it ever existed. That is not a worry: the
        reading of the table -- the masks, `apy_char_class_of`,
        `apy_cp_printable_of` -- was written straight into the generated file
        and sat there until this test was written, at which point it moved to
        `charclass.py` where a regeneration cannot reach it.

        SO THE FILE MAY DEFINE EXACTLY WHAT THE GENERATOR WRITES and nothing
        else. A new name here means someone appended to a generated file.
        """
        import ast

        ir_src = (Path(SRC) / "uasm" / "runtime"
                  / "unicode_table.py").read_text(encoding="utf-8")
        defined = {n.name for n in ast.parse(ir_src).body
                   if isinstance(n, ast.FunctionDef)}
        assert defined == {"apy_uc_rows", "apy_uc_stride", "apy_uc_table",
                           "apy_uc_lookup"}, (
            "unicode_table.py is generated and defines something the "
            f"generator does not write: {sorted(defined)}")

    def test_the_unicode_table_agrees_with_the_c(self):
        """The IR's character-class table is the C's, row for row.

        TWO COPIES OF ONE FACT, which is what this exists to stop drifting.
        The C stores `struct { unsigned lo, hi, m; }` and the IR stores the
        same twelve bytes little-endian; both are generated from the
        reference implementation's data, and `tools/gen_unicode_ir.py`
        derives the second FROM the first so they cannot disagree at birth.
        This is what catches them disagreeing later.

        READ FROM THE SOURCE rather than by running either one: the question
        is whether the two tables hold the same numbers, and that is a
        question about the text.
        """
        import ast
        import struct

        c_src = (Path(SRC) / "uasm" / "objects" / "c"
                 / "unicode_table.py").read_text(encoding="utf-8")
        rows = [(int(a), int(b), int(m)) for a, b, m in
                re.findall(r"\{(\d+),(\d+),(\d+)\}", c_src)]
        assert rows, "no ranges found in the C table"

        ir_src = (Path(SRC) / "uasm" / "runtime"
                  / "unicode_table.py").read_text(encoding="utf-8")
        blob = None
        for node in ast.walk(ast.parse(ir_src)):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "rodata"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, bytes)):
                blob = node.args[0].value
                break
        assert blob is not None, "no rodata blob in the IR table"
        assert len(blob) == len(rows) * 12, (
            f"the IR blob holds {len(blob) // 12} rows and the C holds "
            f"{len(rows)}")
        got = [struct.unpack_from("<III", blob, i * 12)
               for i in range(len(rows))]
        assert got == rows, "the two tables hold different ranges"

        # AND THE ROW COUNT THE SEARCH USES, which is a third copy: a binary
        # search over the wrong length reads past the blob or stops early.
        m = re.search(r"def apy_uc_rows\(\) -> i64:.*?return (\d+)",
                      ir_src, re.S)
        assert m, "apy_uc_rows does not end in a plain integer"
        assert int(m.group(1)) == len(rows), (
            f"apy_uc_rows says {m.group(1)}, the table holds {len(rows)}")

    def test_no_table_names_a_file_twice(self):
        """A duplicate key in REPLACES or SPLITS silently drops the first one.

        Python takes the LAST value for a repeated dict key and says nothing,
        so a second `"mathints.py":` entry added at the bottom of SPLITS threw
        away every name the first one listed -- and the failure was not a
        duplicate-key complaint but `call to unknown function
        'apy_num_order_of_slow'`, which points at the runtime rather than at
        the table.

        READ FROM THE SOURCE rather than from the imported dict, because by
        the time it is a dict the evidence is gone.
        """
        import ast
        source = (Path(SRC) / "uasm" / "objects" / "ir.py").read_text(
            encoding="utf-8")
        for node in ast.parse(source).body:
            target = getattr(getattr(node, "target", None), "id", "")
            if target not in ("REPLACES", "SPLITS"):
                continue
            keys = [k.value for k in node.value.keys]
            twice = sorted({k for k in keys if keys.count(k) > 1})
            assert not twice, (
                f"{target} names {twice} more than once; the later entry "
                f"silently replaces the earlier one")

    def test_every_replaced_name_is_actually_defined(self):
        sys.path.insert(0, str(SRC))
        try:
            from uasm.objects import ir as objects_ir
            assert objects_ir.check() == []
        finally:
            del sys.path[0]

    def test_the_c_no_longer_defines_what_the_ir_does(self):
        """The C keeps a DECLARATION, which its own hundred-odd callers need,
        and loses the body. Removing it outright would make every caller an
        implicit declaration returning `int`."""
        sys.path.insert(0, str(SRC))
        try:
            from uasm.objects.csource import objects_c
            from uasm.objects.ir import REPLACES
            replaced = tuple(sorted(n for ns in REPLACES.values() for n in ns))
            text = objects_c(omit=replaced)
        finally:
            del sys.path[0]
        # A SIGNATURE MAY SPAN LINES, which this used to assume it did not:
        # `apy_slice_new` takes three parameters and wraps, so the `;` and the
        # marker comment land on the CONTINUATION line -- which does not start
        # with `APY_API`, so the marker was looked for on the wrong line and a
        # perfectly good declaration was reported as a missing one. Each head
        # is joined to what follows it up to the first `;` or `{`.
        lines = text.splitlines()
        for name in replaced:
            heads = []
            for i, l in enumerate(lines):
                if not (l.startswith("APY_API") and f" {name}(" in l):
                    continue
                whole = l
                j = i
                while ";" not in whole and "{" not in whole and j + 1 < len(lines):
                    j += 1
                    whole += " " + lines[j].strip()
                heads.append(whole)
            # A FORWARD DECLARATION MAY ALSO EXIST and is not a definition:
            # `apy_obj_alloc` has one, because the `apy_alloc` wrapper above it
            # calls it. What must be gone is the BODY.
            assert heads, name
            assert not [h for h in heads if "{" in h],                 f"{name} still has a C body: {heads}"
            assert any(h.rstrip().endswith("*/") for h in heads), heads

    def test_the_runtime_source_compiles_on_the_static_path(self):
        """It must emit no `apy_*` it did not name, and no object at all.

        The static path is the one that stands on the machine rather than on
        the runtime -- that is the whole reason the port is written in it --
        so a runtime module that had drifted onto the dynamic path would be
        depending on the thing it is replacing.
        """
        sys.path.insert(0, str(SRC))
        try:
            from uasm.objects import ir as objects_ir
            module = objects_ir.compile_runtime()
        finally:
            del sys.path[0]
        defined = {f.name for f in module.functions if not f.external}
        assert "apy_from_int" in defined and "apy_as_int" in defined
        # THE FLOOR, and the `_slow` halves of the split functions -- which
        # are the C, deliberately, and the only `apy_*` a ported file may
        # reach that it does not define. Anything else here would mean the
        # runtime had grown a dependency nobody declared.
        # WHAT THE RUNTIME CALLS, not what it declares. An intrinsic is
        # declared as an external in every module lowered -- see
        # `lower._intrinsic_externals`, which says why it cannot wait until a
        # body mentions one -- so an intrinsic's symbol appears in the
        # runtime's declarations and is called by nothing in it. Counting
        # declarations made this fail the day a backend first lowered one,
        # over symbols the runtime does not reach for in any sense the test
        # means.
        #
        # A CALL IS THE THING THE TEST IS ABOUT: "the allocator asks the floor
        # and nothing else" is a statement about what it REACHES, and an
        # unused declaration costs no bytes and creates no link dependency.
        called = {ins.sym for fn in module.functions
                  for block in fn.blocks for ins in block.instructions
                  if ins.sym and ins.op.name in ("CALL", "FUNC_ADDR")}
        external = {f.name for f in module.functions
                    if f.external and f.name in called}
        from uasm.objects.ir import SPLIT
        assert external <= ({"plat_heap", "plat_write", "plat_exit"}
                            | {n + "_slow" for n in SPLIT}), external


class TestItBehavesLikeCPython:
    """Small-integer sharing is observable, so it has to be exact."""

    @harness.needs("gcc")
    def test_the_cache_shares_exactly_what_cpython_shares(self, tmp_path):
        """-5..256 shared, 257 not. The boundary is the point: a cache that
        was one wider or one narrower passes every test that does not look at
        it and fails `a = 257; b = 257; a is b`."""
        program = """\
            for n in (-6, -5, 0, 1, 256, 257):
                a = n + 0
                b = n + 0
                print(n, a is b)
        """
        got = build_and_run(tmp_path, write(tmp_path, program))
        assert got.returncode == 0, got.stderr
        assert got.stdout.split() == [
            "-6", "False", "-5", "True", "0", "True", "1", "True",
            "256", "True", "257", "False"], got.stdout

    @harness.needs("gcc")
    def test_arithmetic_through_the_ported_constructor(self, tmp_path):
        """Everything the runtime builds an int with goes through it -- every
        literal, every length, every loop counter -- so a wrong cell is not a
        subtle failure."""
        program = """\
            print(sum(range(100)))
            print(len([1, 2, 3]) * 7)
            print((2 ** 70) // 3)
            print(-9223372036854775808 + 1)
            print(int("12345") + 1)
        """
        got = build_and_run(tmp_path, write(tmp_path, program))
        assert got.returncode == 0, got.stderr
        assert got.stdout.split("\n")[:5] == [
            "4950", "21", "393530540239137101141", "-9223372036854775807",
            "12346"]


class TestNothingElsePays:
    """The splice must not give the object runtime to programs that had none."""

    def test_a_program_with_no_runtime_at_all_gets_nothing(self, tmp_path):
        """A program that links none of the C gets none of the IR either.

        The condition is `objects.support.needs_runtime`, which is exactly
        "will the object runtime's C be in this build". A statically typed
        program that PRINTS does link it -- for `put_int` -- and `objects_c()`
        comes with it, so it needs the ported definitions even though its own
        code never names one. That is not the splice being greedy; it is what
        the C runtime being included whole already meant.
        """
        source = write(tmp_path, """\
            def main() -> int:
                return 6 * 7
        """)
        emitted = _cli("build", str(source), "--emit-ir",
                       "-o", str(tmp_path / "prog.ir"),
                       "--workdir", str(tmp_path / "wd"))
        assert emitted.returncode == 0, emitted.stdout + emitted.stderr
        text = (tmp_path / "prog.ir").read_text(encoding="utf-8")
        assert "apy_" not in text, text

    def test_a_dynamic_program_gets_the_ported_definitions(self, tmp_path):
        source = write(tmp_path, "print(1 + 1)\n")
        emitted = _cli("build", str(source), "--emit-ir",
                       "-o", str(tmp_path / "prog.ir"),
                       "--workdir", str(tmp_path / "wd"))
        assert emitted.returncode == 0, emitted.stdout + emitted.stderr
        text = (tmp_path / "prog.ir").read_text(encoding="utf-8")
        # DEFINED, not declared: `func apy_from_int(...) -> ptr {`, with a body.
        assert re.search(r"func apy_from_int\([^)]*\) -> ptr \{", text), \
            "apy_from_int is not defined in the spliced IR"
        assert "apy_small_ir" in text, "the small-int cache global is missing"


class TestTheArithmeticSplit:
    """Stage 3's other half: a fast path in IR over a C remainder.

    `apy_add` is polymorphic over eighteen kinds, so it cannot be ported whole
    without porting all of them. The subset defines it, answers int x int, and
    calls `apy_add_slow` -- the C body under a new name -- for everything else.
    THIS IS THE SHAPE EVERY REMAINING KIND TAKES, so it is tested for the two
    ways it can go wrong: answering a case it should have declined, and
    declining one it should have answered.
    """

    @harness.needs("gcc")
    def test_the_fast_path_and_the_fallback_agree_with_cpython(self, tmp_path):
        """Each line crosses the boundary somewhere.

        The overflow lines are the ones that matter: Python's integers are
        arbitrary precision, so an overflow is not an error but a promotion to
        the big-integer path -- which is entirely in the C. A fast path that
        answered them would be wrong by a factor of 2**64.
        """
        program = """\
            print(2 + 3, 10 - 4, 6 * 7)
            print(9223372036854775807 + 1)
            print(-9223372036854775808 - 1)
            print(-9223372036854775808 - 0)
            print(3037000500 * 3037000500)
            print(True + True, 1.5 + 2, "a" + "b", [1] + [2])
            print((2 ** 70) + 1, (2 ** 70) * 3)
            print((-5) - (-5), 0 - (-9223372036854775808))
        """
        got = build_and_run(tmp_path, write(tmp_path, program))
        assert got.returncode == 0, got.stderr
        assert got.stdout.split("\n")[:8] == [
            "5 6 42",
            "9223372036854775808",
            "-9223372036854775809",
            "-9223372036854775808",
            "9223372037000250000",
            "2 3.5 ab [1, 2]",
            "1180591620717411303425 3541774862152233910272",
            "0 9223372036854775808",
        ], got.stdout

    def test_the_c_keeps_its_body_under_the_new_name(self):
        """A split is not an omission: the body is what the fast path calls.
        Confusing the two is a link error naming `apy_add_slow`."""
        sys.path.insert(0, str(SRC))
        try:
            from uasm.objects.csource import objects_c
            from uasm.objects.ir import SPLIT
            text = objects_c(split=SPLIT)
        finally:
            del sys.path[0]
        # A SIGNATURE MAY WRAP, and until `str` began moving none of them did.
        # `apy_str_find3(apy_value s, apy_value sub, apy_value start,` carries
        # its `apy_value end);` on the next line, so asking whether THIS line
        # ends with `;` or `{` put it in neither bucket -- and the test failed
        # for a split the generator had performed perfectly. Each head is
        # joined to its terminator before anything is decided about it.
        # A ONE-LINE BODY ENDS WITH `}`, WHICH IS ALSO COMPLETE. The
        # comparison family is written `APY_API ... apy_lt_slow(a, b){ return
        # apy_cmp("<", ...); }` all on one line, and a joiner that stopped
        # only at `;` or `{` treated that as an unfinished signature and glued
        # the NEXT declaration onto it -- so `apy_lt_slow` had no definition
        # and `apy_le` appeared to have two. The split was performed
        # perfectly; the reading of it was wrong.
        lines = text.splitlines()
        joined = []
        for i, first in enumerate(lines):
            if not first.startswith("APY_API"):
                continue
            head, j = first, i
            while (not head.rstrip().endswith((";", "{", "}"))
                   and j + 1 < len(lines)):
                j += 1
                head = head + " " + lines[j].strip()
            joined.append(head)
        for name in SPLIT:
            heads = [l for l in joined
                     if f" {name}(" in l or f" {name}_slow(" in l]
            # CLASSIFIED BY WHETHER A BODY OPENS, not by the last character:
            # a definition may end with `{` (multi-line) or with `}` (one
            # line), and both contain a brace that a declaration never does.
            declared = [l for l in heads if "{" not in l]
            defined = [l for l in heads if "{" in l]
            assert any(f" {name}(" in l for l in declared), (name, heads)
            assert [l for l in defined if f" {name}_slow(" in l], (name, heads)
            assert not [l for l in defined if f" {name}(" in l], \
                f"{name} still has a C body: {defined}"

    def test_a_split_name_is_never_also_omitted(self):
        """The two lists ask the C for opposite things."""
        sys.path.insert(0, str(SRC))
        try:
            from uasm.objects.ir import PORTED, SPLIT, REPLACES
            replaced = {n for names in REPLACES.values() for n in names}
            assert not (replaced & set(SPLIT)), replaced & set(SPLIT)
            assert set(SPLIT) <= set(PORTED)
        finally:
            del sys.path[0]


class TestTheArena:
    """Stage 4: one allocator, in the subset, over the floor.

    `apy_alloc` in the C is now a wrapper around `apy_obj_alloc`, which this
    port defines -- so every object in a compiled program, the C runtime's and
    the ported code's alike, comes from a bump pointer in the machine subset
    rather than from a `malloc` per cell.

    A BUMP ALLOCATOR IS ONLY CORRECT BECAUSE NOTHING FREES A CELL. That was
    checked rather than assumed: all 51 `free()` calls in the C release
    buffers, never an `apy_obj`. If that ever stops being true, this is where
    it breaks, and the test below is what notices.
    """

    @harness.needs("gcc")
    def test_it_survives_tens_of_thousands_of_objects(self, tmp_path):
        """Past one chunk, so the refill path runs and the abandoned tail of
        the previous chunk is exercised. 27,000 cells is several megabytes."""
        program = """\
            xs = []
            for i in range(20000):
                xs.append(i * 7 + 1)
            print(len(xs), xs[0], xs[-1], sum(xs))
            d = {}
            for i in range(5000):
                d[i] = str(i) + "!"
            print(len(d), d[4999])
            class P:
                def __init__(self, n):
                    self.n = n
            print(sum(P(i).n for i in range(2000)))
        """
        got = build_and_run(tmp_path, write(tmp_path, program))
        assert got.returncode == 0, got.stderr
        assert got.stdout.split("\n")[:3] == [
            "20000 1 139994 1399950000", "5000 4999!", "1999000"], got.stdout

    @harness.needs("gcc")
    def test_cells_do_not_overlap(self, tmp_path):
        """The failure a bump allocator has: handing the same bytes out twice.

        Held simultaneously and compared at the end, so an overlap shows as a
        value that changed without being assigned -- which no amount of
        sequential allocation would reveal.
        """
        program = """\
            cells = [[i, i * 3] for i in range(4000)]
            bad = 0
            for i in range(4000):
                if cells[i][0] != i or cells[i][1] != i * 3:
                    bad = bad + 1
            print(bad)
        """
        got = build_and_run(tmp_path, write(tmp_path, program))
        assert got.returncode == 0, got.stderr
        assert got.stdout.strip() == "0", got.stdout

    def test_the_allocator_asks_the_floor_and_nothing_else(self):
        """It stands on stage 2 and on nothing that is not there.

        An allocator that reached for anything else would be the point at
        which "three functions per backend" stopped being true.
        """
        sys.path.insert(0, str(SRC))
        try:
            from uasm.objects import ir as objects_ir
            module = objects_ir.compile_runtime()
        finally:
            del sys.path[0]
        # WHAT THE RUNTIME CALLS, not what it declares. An intrinsic is
        # declared as an external in every module lowered -- see
        # `lower._intrinsic_externals`, which says why it cannot wait until a
        # body mentions one -- so an intrinsic's symbol appears in the
        # runtime's declarations and is called by nothing in it. Counting
        # declarations made this fail the day a backend first lowered one,
        # over symbols the runtime does not reach for in any sense the test
        # means.
        #
        # A CALL IS THE THING THE TEST IS ABOUT: "the allocator asks the floor
        # and nothing else" is a statement about what it REACHES, and an
        # unused declaration costs no bytes and creates no link dependency.
        called = {ins.sym for fn in module.functions
                  for block in fn.blocks for ins in block.instructions
                  if ins.sym and ins.op.name in ("CALL", "FUNC_ADDR")}
        external = {f.name for f in module.functions
                    if f.external and f.name in called}
        floor = {"plat_heap", "plat_write", "plat_exit"}
        # THE `_slow` HALF OF EVERY DECLARED SPLIT, DERIVED rather than listed.
        # A split's whole point is that the C body survives under a new name,
        # so each one legitimately adds an external -- and a hand-written list
        # meant every new split failed here on arrival and was "fixed" by
        # editing the expectation. That is the one edit that can silently
        # widen this assertion into meaninglessness, so the expectation now
        # follows the declaration and the test keeps asking the real question:
        # the runtime reaches for the floor and its own fallbacks, and nothing
        # else.
        allowed = floor | {f"{name}_slow" for name in objects_ir.SPLIT}
        assert external <= allowed, external - allowed
        assert "apy_obj_alloc" in {f.name for f in module.functions
                                   if not f.external}


class TestTheCRuntimeIsStillSupported:
    """The port REMOVES an obligation; it must not create one.

    The reason to write the object runtime in uasm's own subset is that a
    backend should not HAVE to define 229 functions. That is an argument for
    making the C unnecessary -- not for making it unavailable. So the whole
    hand-written C runtime remains a supported arrangement, reachable two ways:

        --object-runtime c          the whole build
        Backend.object_runtime      one function, for a backend that has its own

    Both are tested here rather than left as a flag nobody exercises, because
    an untested opt-out is one that stops working the first time the ported set
    grows -- and it will grow every stage from here.
    """

    def test_the_default_is_the_ported_runtime(self, tmp_path):
        sys.path.insert(0, str(SRC))
        try:
            from uasm.driver import Options
            assert Options(source=tmp_path / "x.py").object_runtime == "ir"
        finally:
            del sys.path[0]

    @harness.needs("gcc")
    def test_both_arrangements_agree(self, tmp_path):
        """The point of the switch: the same program, either runtime, one
        answer. A divergence here is a defect in the port, and this is the
        cheapest place it can be caught."""
        program = """\
            a = 1
            b = 1
            print(a is b, a + b)
            x = 257
            y = 257
            print(x is y, x + y)
            print(sum(range(20)), len("hello") * 3)
        """
        source = write(tmp_path, program)
        outputs = {}
        for mode in ("ir", "c"):
            out = tmp_path / f"prog_{mode}.exe"
            built = _cli("build", str(source), "--backend", "c", "--link",
                         "--object-runtime", mode, "-o", str(out),
                         "--workdir", str(tmp_path / f"wd_{mode}"))
            assert built.returncode == 0, built.stdout + built.stderr
            ran = subprocess.run([str(out)], capture_output=True, text=True)
            assert ran.returncode == 0, ran.stderr
            outputs[mode] = ran.stdout
        assert outputs["ir"] == outputs["c"], outputs
        assert outputs["ir"].split("\n")[0] == "True 2"

    def test_c_mode_splices_nothing_and_keeps_every_c_definition(self, tmp_path):
        source = write(tmp_path, "print(1 + 1)\n")
        emitted = _cli("build", str(source), "--emit-ir", "--object-runtime", "c",
                       "-o", str(tmp_path / "prog.ir"),
                       "--workdir", str(tmp_path / "wd"))
        assert emitted.returncode == 0, emitted.stdout + emitted.stderr
        text = (tmp_path / "prog.ir").read_text(encoding="utf-8")
        assert not re.search(r"func apy_from_int\([^)]*\) -> ptr \{", text), \
            "apy_from_int was spliced despite --object-runtime c"
        assert "apy_small_ir" not in text, "the cache global was spliced anyway"
        # And with nothing spliced, the C keeps its bodies -- which is what
        # makes the two halves of the switch impossible to get out of step:
        # the C omits exactly what the module defines, so no splice is no
        # omission.
        sys.path.insert(0, str(SRC))
        try:
            from uasm.ir.printer import parse_module
            from uasm.objects.csource import objects_c
            from uasm.objects.ir import omitted_by
            module = parse_module(text)
            assert omitted_by(module) == ()
            c = objects_c(omit=omitted_by(module))
            assert "APY_API apy_value apy_from_int(int64_t i) {" in c
        finally:
            del sys.path[0]

    def test_a_backend_may_define_one_itself(self, tmp_path):
        """`Backend.object_runtime` is per FUNCTION, so a backend keeps the one
        it has an opinion about and takes the rest.

        Checked against the splice directly rather than through a real backend:
        no built-in one claims anything today -- that is what makes the shipped
        arrangement the ported one -- and a test that needed one to exist would
        be testing a fixture.
        """
        sys.path.insert(0, str(SRC))
        try:
            from uasm.diagnostics import DiagnosticSink
            from uasm.driver import Options, compile_source
            from uasm.objects import ir as objects_ir
            path = tmp_path / "prog.py"
            path.write_text("print(1 + 1)\n", encoding="utf-8")
            sink = DiagnosticSink()
            # Compiled with the splice OFF, then spliced by hand, so that this
            # asks about `splice` rather than about the driver's plumbing.
            result = compile_source(
                Options(source=path, emit_ir=True, object_runtime="c"), sink)
            assert result.ok, [d.message for d in sink.diagnostics]
            module = result.module
            objects_ir.splice(module, provided=frozenset({"apy_from_int"}))
            defined = {f.name for f in module.functions if not f.external}
            assert "apy_from_int" not in defined, \
                "the backend said it defines this one"
            assert "apy_as_int" in defined, \
                "the rest of the runtime should still arrive"
            # And the C is asked to stand aside for exactly that rest -- the
            # REPLACED names only. A SPLIT name appears in neither list here:
            # its body stays, under the other name, because the ported half
            # calls it.
            omitted = objects_ir.omitted_by(module)
            assert "apy_from_int" not in omitted, omitted
            assert "apy_as_int" in omitted, omitted
            assert set(omitted).isdisjoint(objects_ir.SPLIT), omitted
        finally:
            del sys.path[0]


class TestTheInterpreterKeepsItsOwn:
    """Documented because it is a finding, not a preference.

    `objects_host.py` represents an `apy_value` as a HANDLE and the ported code
    represents one as an ADDRESS, so the two cannot be mixed -- the port is
    all-or-nothing on the interpreter path. Asserted so that a later change
    which quietly starts running spliced IR there fails here, with this
    explanation, rather than as `2248 is not a runtime value handle`.
    """

    def test_the_host_runtime_still_answers(self, tmp_path):
        source = write(tmp_path, "a = 1\nb = 1\nprint(a is b, a + b)\n")
        got = _cli("run", str(source))
        assert got.returncode == 0, got.stderr
        assert got.stdout.strip().endswith("True 2"), got.stdout
