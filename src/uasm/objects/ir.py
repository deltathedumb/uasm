"""The part of the object runtime that is IR, and how it gets into a program.

STAGE 3 OF `docs/INERT-RUNTIME.md`. The object runtime is 15,560 lines of C
(`objects/csource.py`) plus an 8,583-line Python re-implementation for the IR
interpreter (`objects/host.py`), and a backend that wants dynamic Python has
to find 229 `apy_*` symbols -- which is why exactly one backend has them. The
way out is to write the runtime in the machine subset and let it be IR, so that
every backend gets it for free and the two copies become one.

This is the mechanism, and one kind has gone through it.

## What happens

`src/uasm/runtime/*.py` is uasm source in the machine subset. It is
compiled BY THE PYTHON FRONTEND, as a library -- definitions only, no entry --
and the resulting functions and globals are merged into the program's module
before any backend sees it. No artifact is checked in and nothing is generated
ahead of time: the runtime is compiled by the compiler it is part of, every
time, which is the only arrangement in which it cannot go stale.

## Why this replaces rather than adds

A ported function keeps the C's NAME, so a program calls `apy_from_int` and
does not know or care which one it got. That means the C definition has to go,
or the two collide at link time -- so `objects_c()` takes the set of ported
names and `#if`s those definitions out. One switch, and it is the same switch
that makes a regression one flag away from being isolated.

## Why the IR interpreter does NOT get it, and what that tells us

It was going to. The interpreter calls `_host` for a symbol the module does not
DEFINE, so splicing a definition would have made it run the same IR the C
backend compiles -- one runtime, both paths, which is the design document's
central promise.

It cannot, and the reason is worth more than the feature would have been:
**the two runtimes disagree about what an `apy_value` IS.**
`objects/host.py` represents one as a HANDLE into a Python-side table; the
ported code represents one as an ADDRESS in the interpreter's flat memory. So a
ported `apy_from_int` hands back something every unported function rejects --
observed as `apy_is: 2248 is not a runtime value handle`, a message naming
neither runtime.

**The port is therefore not incremental on the interpreter path.** It is
all-or-nothing there, and `Interpreter._call` says so: the host runtime owns
every `apy_*` name it claims, whatever the module defines.

That is not a setback for stage 3, it is what stage 3 is for. The interpreter
stays the ORACLE while the C path runs the ported code, so the corpus compares
a ported `apy_from_int` against an unported one on every program it runs --
which is a sharper test than having both paths run the same code would be.
What it changes is stage 6: `objects_host.py` cannot be retired function by
function, only in one step, once every kind it claims has been ported.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

#: Where the subset-written runtime lives.
SOURCE_DIR = Path(__file__).resolve().parent.parent / "runtime"

#: The C definitions a ported module replaces, per source file.
#:
#: WRITTEN OUT rather than derived from what the compiled module exports,
#: because most of what a runtime file defines is its own helpers --
#: `apy_int_alloc`, `apy_small_slot` -- and those must NOT displace anything.
#: Only the names listed here are `#if`d out of the C, and a name listed here
#: that the module does not define is caught by `check()`.
REPLACES: dict[str, tuple[str, ...]] = {
    "arena.py": ("apy_obj_alloc",),
    # THE CHARACTER CLASSES. `unicode_table.py` DISPLACES NOTHING and the
    # empty tuple says so -- it is generated, it is the ranges, and its lookup
    # is named `apy_uc_lookup` rather than the C's `apy_uc_mask` for the
    # reason both halves of this pair exist: THE C AND THE IR ARE ONE
    # TRANSLATION UNIT. A subset function keeping a C static's name is not a
    # second copy sitting harmlessly beside it, it is `conflicting types for
    # 'apy_char_class'` from gcc -- so a name is either replaced properly or
    # it is a different name.
    #
    # AND A KEY IS WHAT MAKES A FILE A MODULE: `sources()` is these tables, so
    # an unkeyed file in `runtime/` is not compiled, not checked, and not
    # anything -- it just sits there looking ported.
    #
    # `unicode_table.py` IS GENERATED and holds the ranges; `charclass.py` is
    # hand-written and holds the reading of them.
    "unicode_table.py": (),
    "charclass.py": ("apy_char_class_of", "apy_cp_printable_of"),
    # THE THREE CONTAINER FORMATTERS, which recurse through `apy_text_of` and
    # therefore through the split -- so each one is portable on its own and
    # anything they hold that is not ported yet goes back to the C and
    # returns. See the file's own header.
    "text_containers.py": ("apy_seq_text_of", "apy_dict_text_of",
                           "apy_set_text_of"),
    # THE FOURTH FORMATTER, and the one that is not a container: an exception
    # renders its ARGUMENT, and which of `str` and `repr` it uses for that
    # depends on the exception's NAME. See the file.
    "text_exc.py": ("apy_exc_text_of",),
    # `ascii()`, and the PERMISSIVE utf-8 step it walks with -- which is a
    # different function from the validating one `repr` uses, for a reason the
    # file explains. `apy_utf8_at` was named as a blocker by three others.
    "str_ascii.py": ("apy_utf8_at_of", "apy_ascii",
                     "apy_default_repr"),
    # A WALL AND NOT A LEAF: `apy_index_arg` is eight lines and four exported
    # functions were waiting behind it. See the file's own header.
    "slicing.py": ("apy_index_arg_of", "apy_slice_indices"),
    # AND WHAT WAS BEHIND THE WALL. Two containers, four failure modes, and
    # CPython's own wording for each -- see the file.
    "delitem.py": ("apy_delitem", "apy_default_delattr",
                    "apy_delattr",
                    # THE TWO HALVES OF A SLICE DELETE. They are exported --
                    # rather than private the way `apy_del_span` beside them
                    # is -- because the C has them under the same names, and
                    # the C and the IR are ONE translation unit: a name is
                    # either replaced properly or it is a different name.
                    "apy_del_run", "apy_del_bytes"),
    # A PER-CHARACTER SUBSTITUTION whose output length is not a function of
    # its input length -- a replacement may be a whole string or a deletion.
    # AND THE BYTES FAMILY THAT SHARES THE NAME: bytes map by BYTE
    # through a 256-byte table, which is a different method under one
    # spelling. Exported rather than private because the C has them
    # under the same names and the two are ONE translation unit.
    # GENERATED, and it DISPLACES NOTHING: the C keeps its own copy under a
    # `static` name and this is the `_of` twin, so the key is here to satisfy
    # "every runtime module is named" rather than to claim a C definition.
    "kindmeth_table.py": (),
    "str_translate.py": ("apy_str_translate", "apy_bytes_maketrans",
                         "apy_bytes_translate", "apy_translate_kw"),
    # MAKING a big, where every other big function in IR could only READ one
    # -- which is what kept the whole float-to-integer boundary in C. See the
    # file's own header.
    "bigmake.py": ("apy_big_alloc_of", "apy_mag_trim_of", "apy_big_done_of",
                   "apy_big_of_i64_of", "apy_mag_shl_of",
                   "apy_math_floor", "apy_math_ceil", "apy_math_trunc"),
    "int_cell.py": ("apy_from_int", "apy_as_int"),
    # THE THREE STRING CONSTRUCTORS THAT BORROW THEIR BYTES. The ones that OWN
    # them -- `apy_str_take`, `apy_str_copy` -- were `static` when this was
    # written and are exported now, so the wall this described is gone. What
    # keeps them here is no longer visibility but their SIGNATURES: both take
    # a `char *`, and see `_WONT` below for why that is not a thing to fix by
    # porting them.
    # `apy_str_copy_bytes` IS THE ONE THE RUNTIME BUILDS EVERY STRING WITH --
    # 24 call sites reach it through the `apy_str_copy` shim -- so replacing it
    # moves a compiled program's string BYTES off malloc and onto the arena.
    # It could only be named once the promotion made it visible, and it needed
    # an `apy_value` parameter because that is what an IR `ptr` compiles to.
    "str_cell.py": ("apy_from_cstr", "apy_from_bytes", "apy_bytes_literal",
                    "apy_str_copy_bytes", "apy_shared_str",
                    "apy_shared_bytes"),
    # THE BUFFERS, which is the prerequisite docs/INERT-RUNTIME.md names ahead
    # of `list` and `dict`: those two grow by DOUBLING and release the block
    # they grew out of, and stage 4's bump arena can neither resize nor
    # reclaim. Size classes and a free list over the same arena, so the
    # platform floor is still three functions.
    "blocks.py": ("apy_alloc_block", "apy_realloc_block", "apy_free_block"),
    # THE ASCII PREDICATES, and the smallest thing in the runtime that can be
    # ported: they take a number and answer a number, so there is no cell
    # layout to know and no allocation to get right. The four not listed here
    # are written out as specifications at the foot of `runtime/ascii.py`; a
    # name joins this tuple when its function exists, and until then the C's
    # version is what a program uses.
    "ascii.py": ("apy_c_lower", "apy_c_upper", "apy_c_digit", "apy_c_alpha",
                 "apy_c_space"),
    # THE SLICE AND THE DESCRIPTOR, found by asking a better question: not
    # "what is blocked by a static" but "what does this call that IR does not
    # already define". Both are cell construction and nothing else, so both
    # move whole -- there is no case they cannot handle and so nothing for a
    # slow half to do.
    "slots.py": ("apy_slice_new", "apy_descr_new",
                 # THE PROPERTY DECORATORS AND `list.reverse`, the first
                 # things the fixed survey turned up: its stripper had been
                 # deleting most of the C, so these had never appeared.
                 "apy_prop_getter", "apy_prop_setter", "apy_prop_deleter",
                 "apy_list_reverse"),
    # THE FLOAT AND COMPLEX CELLS, and the first port to carry an `f64`. Found
    # the same way the two above were.
    "floats.py": ("apy_from_float", "apy_from_complex"),
    # THE FUNCTION AND GENERATOR CELLS. `funcs.py` and `gens.py` could already
    # read and write these two arms; this is what MAKES one.
    "makers.py": ("apy_func_new", "apy_gen_new",
                  # FOUR MORE CELLS the fixed survey turned up.
                  "apy_range", "apy_super", "apy_memoryview",
                  # AND WHAT A VIEW'S BORROW IS. `release` drops it and every
                  # reader has to ask whether it still holds, so the test and
                  # the two methods that change it are ported together with
                  # the cell they read.
                  "apy_mview_live", "apy_mview_release",
                  "apy_mview_readonly",
                  # AND `cast`, which changes only the cell -- the element
                  # DECODE stays in the C, because `apy_mview_item` reads a
                  # float out of eight bytes and the subset has no way to
                  # spell that reinterpretation.
                  "apy_mview_cast",
                  # WHAT A VIEW SHOWS, AS BYTES. Ported because the ported
                  # runtime has to DEFINE everything it calls -- see
                  # `test_the_allocator_asks_the_floor_and_nothing_else` --
                  # and `bytes.translate` takes a bytes-like table.
                  "apy_mview_bytes",
                  "apy_type_new",
                  # THE NATIVE CACHE AND THE `type` CLASS.
                  "apy_native_of", "apy_type_class", "apy_abs64_of",
                  # THE SEVEN SET METHODS, AND WHAT A VALUE'S TYPE IS.
                  "apy_set_union", "apy_set_intersection",
                  "apy_set_difference", "apy_set_symdiff",
                  "apy_set_issubset", "apy_set_issuperset",
                  "apy_set_isdisjoint", "apy_type_for", "apy_type_object",
                  "apy_type_rows", "apy_type_slot_count",
                  "apy_canonical_slot"),
    # THE SET, FROZENSET AND DICT CONSTRUCTORS, which `apy_seq_alloc` in
    # `list_cell.py` said it was waiting for.
    # THE EXCEPTION OBJECT, which `errstate.py` owning the pending error was
    # the prerequisite for.
    "excval.py": ("apy_error_type", "apy_error_value", "apy_add_note"),
    # TWO LOOKUPS THAT ANSWER WITH A POINTER, both of which needed the kind
    # names first.
    "kindname.py": ("apy_str_bytes", "apy_type_name"),
    "containers.py": ("apy_set_new", "apy_frozenset_new", "apy_dict_new",
                      "apy_clear", "apy_dict_view", "apy_from_bytes_n",
                      "apy_env_cell", "apy_dict_popitem",
                      # TWO LOOKUPS THE EQUALITY SPLIT UNBLOCKED.
                      "apy_dict_find_of", "apy_class_find_of",
                      # AND THE THREE THOSE TWO MADE READY.
                      "apy_dict_get_or", "apy_method_is_builtin",
                      "apy_method_self",
                      # THE NAME CACHE AND THE HASHABILITY CHECK.
                      "apy_name_of", "apy_name_rows", "apy_name_slot",
                      "apy_unhashable_of", "apy_unhashable_key_of",
                      "apy_dict_set",
                      # AND THE SEVEN `apy_dict_set` MADE READY.
                      "apy_callable", "apy_locals_put", "apy_setdefault",
                      "apy_type_set", "apy_typevar_default",
                      "apy_match_args", "apy_match_rest",
                      # THE INSTANCE CHAIN.
                      "apy_class_builtin_of", "apy_instance_new",
                      "apy_key_at",
                      # AND THREE WALKS `apy_key_at` MADE READY.
                      "apy_dict_fromkeys", "apy_sum", "apy_sum_from",
                      # BINDING, AND READING AN INTEGER ARGUMENT.
                      "apy_bind_of", "apy_dunder_of", "apy_clamp_range_of",
                      "apy_int_arg_of", "apy_slice_arg_of",
                      "apy_affix1_of",
                      # AND THE WORKER ABOVE IT, WHICH THE SPLITS DECLINED:
                      # a tuple of prefixes, a start and an end.
                      "apy_affix_of", "apy_str_startswith2",
                      "apy_str_startswith3", "apy_str_endswith2",
                      "apy_str_endswith3",
                      # THE SEARCH PRIMITIVES, AND THE SPLIT FAMILY ON THEM.
                      "apy_find_at", "apy_rfind_at", "apy_str_slice_of",
                      "apy_split_ws_of",
                      "apy_split_sep_of", "apy_str_split_impl_of",
                      "apy_str_split_ws", "apy_str_split_n",
                      "apy_str_rsplit_ws", "apy_str_rsplit_n",
                      # AND WHAT THE CALLING SPLIT MADE SAYABLE.
                      "apy_enter", "apy_exit", "apy_aenter",
                      "apy_index", "apy_invert", "apy_slice_bound",
                      "apy_getiter", "apy_enumerate", "apy_filter",
                      "apy_map", "apy_zip_n", "apy_zip2", "apy_prepare",
                      # BUILDING AN EXCEPTION, now that raising one is split.
                      "apy_exc_class_named_of", "apy_exc_construct_of",
                      "apy_make_exc",
                      # AND RESUMING A GENERATOR, which needed both.
                      "apy_gen_step_of",
                      "apy_make_exc0", "apy_exc_type", "apy_raise_from",
                      "apy_gen_drain", "apy_gen_close", "apy_gen_throw",
                      "apy_task_result",
                      "apy_step", "apy_aexit",
                      "apy_iterable",
                      "apy_iter", "apy_reversed", "apy_extend",
                      "apy_delegate_step", "apy_excgroup_new",
                      "apy_every_of", "apy_all", "apy_any",
                      "apy_iter_until", "apy_iadd", "apy_gen_stop",
                      # SETTING AN ATTRIBUTE, six deep.
                      "apy_is_data_descriptor_of", "apy_descr_set_of",
                      "apy_slot_allows_of", "apy_default_setattr",
                      "apy_setattr",
                      # AND EIGHT THAT SETTING ONE MADE READY.
                      "apy_gen_next", "apy_gen_send", "apy_next",
                      "apy_typing_final", "apy_typing_override",
                      "apy_interpolation_new", "apy_template_new",
                      "apy_asyncio_taskgroup",
                      # AND THREE WHOSE ONLY LIBC WAS A MESSAGE.
                      "apy_hash", "apy_ns_get", "apy_unpack_check",
                      "apy_check_slots", "apy_to_dict", "apy_update",
                      # AND FOUR WHOSE ONLY LIBM WAS A BIT TEST.
                      "apy_math_fabs", "apy_math_copysign",
                      "apy_is_integer", "apy_math_isclose",
                      # ORDERING, from two limbs up to two user classes.
                      "apy_mag_cmp_of", "apy_big_cmp_of",
                      "apy_cmp_int_double_of", "apy_either_inst_of",
                      "apy_binary_dunder_of", "apy_order_rich_of",
                      "apy_order_of",
                      "apy_extreme_n", "apy_extreme_of",
                      "apy_extreme_by_of", "apy_sorted",
                      # AND THE HINT `apy_sorted` ASKS FOR, which had to come
                      # with it: a replaced function cannot call back into the
                      # C half of the one translation unit for its own helper.
                      "apy_length_hint",
                      "apy_max", "apy_min", "apy_max_by", "apy_min_by",
                      "apy_dict_of", "apy_dir",
                      # AND TWO WHOSE ONLY LIBC WAS AN ABORT THE ARENA
                      # DOES NOT NEED.
                      "apy_bytes_hex", "apy_bytes_hex_n", "apy_bytes_fromhex",
                      "apy_extreme_or", "apy_splitlines_impl_of",
                      "apy_str_expandtabs", "apy_to_bytes_n",
                      "apy_call_spread",
                      # bin, oct and hex, which are one function three times.
                      "apy_mag_bits_of", "apy_big_base_text_of",
                      "apy_base_text_of", "apy_bin", "apy_oct", "apy_hex",
                      "apy_bit_length", "apy_str_splitlines",
                      "apy_str_splitlines_keep",
                      "apy_code_of",
                      "apy_arg_must_be_str_of", "apy_str_other_of",
                      "apy_str_count_in_of",
                      # isinstance, and the two questions under it.
                      "apy_inst_held_of", "apy_type_is_sub_of",
                      "apy_isinstance",
                      "apy_is_subclass", "apy_str_count2",
                      "apy_str_count3",
                      "apy_group_select_of",
                      "apy_names_object", "apy_is_classlike",
                      "apy_group_split", "apy_group_subgroup",
                      "apy_group_dispatch",
                      # THE PIECES ATTRIBUTE LOOKUP STANDS ON.
                      "apy_class_builtin_kind", "apy_is_descriptor_of",
                      "apy_member_descriptor", "apy_getset_descriptor",
                      "apy_kind_class",
                      "apy_object_default", "apy_object_class",
                      "apy_descr_get_of", "apy_split_of",
                      "apy_kind_method_of", "apy_kind_method_opt",
                      "apy_kind_attr_of",
                      # WHICH DUNDERS A VALUE HAS, as two tables of names.
                      # They are `apy_kind_attr_of`'s and live beside it: a
                      # builtin has no class dict to search, so the list of
                      # what it carries has to be written somewhere, and
                      # arity is the only thing the caller needs back.
                      "apy_object_arity", "apy_number_arity",
                      # AND THE VARIADIC BUILTIN METHOD, beside the fixed
                      # and optional-tail ones: `format` takes whatever it
                      # is given and neither of those can say so.
                      "apy_kind_method_var",
                      "apy_kind_attr", "apy_kind_prototype",
                      "apy_no_attribute", "apy_mro_entries",
                      "apy_traceback_of",
                      "apy_getattr", "apy_big_popcount",
                      "apy_bit_count", "apy_getattr_default",
                      "apy_hasattr",
                      # THE PIECES `repr` STANDS ON.
                      "apy_repr_entered", "apy_repr_left",
                      "apy_special_form_class", "apy_is_special_form",
                      "apy_exc_shown_of", "apy_text_result_of",
                      "apy_utf8_step_of",
                      "apy_big_text", "apy_bytes_repr", "apy_typing_form",
                      "apy_repr", "apy_str",
                      "apy_set_names",
                      "apy_aiter", "apy_unary_dunder_of", "apy_method1_of",
                      # THE SET CORE, which stands on hashing and touches the
                      # call machinery nowhere.
                      "apy_set_mask_of", "apy_q_append_of",
                      "apy_set_find_of", "apy_set_reorder_of",
                      # AND THE LAYER ABOVE THE CORE.
                      "apy_unhashable_elem_of", "apy_subset_of",
                      "apy_mutable_set_of", "apy_set_insert_of",
                      "apy_set_from_of",
                      # AND SEVEN THE SET LAYER MADE READY.
                      "apy_set_add", "apy_set_push", "apy_set_discard",
                      "apy_to_set", "apy_to_frozenset", "apy_copy",
                      "apy_list_insert",
                      # THE SET ALGEBRA AND THE RELATIONS.
                      "apy_binop_error_of", "apy_set_algebra_of",
                      "apy_set_method_of", "apy_set_relate_of",
                      # AND FOUR THE TYPE AND SET LAYERS MADE READY.
                      "apy_set_update", "apy_vars", "apy_func_is_type",
                      "apy_typevar",
                      # AND THREE WHOSE STORAGE MOVED WITH THEM.
                      "apy_exc_class_slot", "apy_exc_class_bind",
                      "apy_live_agens_slot", "apy_agen_mark",
                      "apy_tasks_slot", "apy_asyncio_create_task"),
    # THE TASK PREDICATES. A task is a generator with a flag.
    # TWO `math` FUNCTIONS AND THE VIEWS' CONTENTS, all three of which came
    # ready the moment `apy_is_int_like` and `apy_seq_new` stopped being
    # static in the C.
    # THE CURSOR AND THE STRING COMPARE, two more statics the rest of the
    # runtime could not name.
    "cursor.py": ("apy_str_cmp_of", "apy_cursor_of",
                  # THE RULE `__iter__`'S ANSWER HAS TO SATISFY, which three
                  # funnels here ask and which used to be written out at
                  # each. Ported rather than called in the C for the reason
                  # the ordering's helpers were: the ported runtime reaches
                  # for the floor and its own, and an unported C name called
                  # from here is neither.
                  "apy_is_iterator_of"),
    "mathints.py": ("apy_math_gcd", "apy_math_factorial", "apy_view_items",
                    "apy_math_lcm",
                    # `comb` AND `perm` came with the big-integer fix to
                    # `gcd`: all four multiply through `apy_mul`, so all
                    # four promote past a machine word the way `2 ** 100`
                    # does. Listed here so the C's own definitions stand
                    # aside -- two bodies for one symbol is a link error.
                    "apy_math_comb", "apy_math_perm",
                    # THE NUMERIC WALLS. `apy_math_arg` is the biggest blocker
                    # `uasm plugin port` reports and is five lines; what kept it
                    # in the C was `static`, not difficulty.
                    "apy_is_big_of", "apy_math_arg_of",
                    # AND THE SIX THAT WERE WAITING ON IT.
                    "apy_math_isnan", "apy_math_isinf", "apy_math_isfinite",
                    "apy_math_degrees", "apy_math_radians",
                    "apy_conjugate",
                    # THE ORDERING'S TWO. Both had to be PORTED rather than
                    # left in the C and called: the ported runtime reaches
                    # for the floor and its own fallbacks and nothing else,
                    # which `test_the_allocator_asks_the_floor_and_nothing_
                    # else` is there to keep true -- and an unported C name
                    # called from here is neither.
                    "apy_order_held", "apy_order_error_of",
                    # AND THE ORDERING'S THREE STEPS, split out of
                    # `apy_binary_dunder_of` so the builtin a class extends
                    # can be read BETWEEN the two written halves rather than
                    # after both -- see `apy_order_lt_of`. Ported for the
                    # same reason the two above were: `apy_order_lt_of` is
                    # the subset's own and calls these, and an unported C
                    # name called from here is not the floor.
                    "apy_order_mirror_first_of", "apy_written_dunder_of"),
    "tasks.py": ("apy_str_like", "apy_meta_for", "apy_asyncio_gather",
                 "apy_task_done", "apy_task_cancelled",
                 "apy_task_cancel"),
    # THE SEQUENCE CELL, and the first caller `blocks.py` was written for.
    # Both constructors only fill a cell in, so both move whole; `apy_seq_push`
    # cannot, and is a split below.
    "list_cell.py": ("apy_list_new", "apy_tuple_new",
                     # THREE PREDICATES THE REST OF THE RUNTIME WAITS ON.
                     # None is interesting; all three are `static` in the C,
                     # and twenty-one functions have one of them as their last
                     # blocker.
                     "apy_is_int_like_of", "apy_is_num_of",
                     "apy_str_self_of", "apy_seq_new_of",
                     "apy_is_seq_of", "apy_is_set_of"),
    # THE SINGLETON CELLS -- stage 5b, which was not in the plan and was found
    # by `list_cell.py` walking into it. These four move TOGETHER and cannot
    # move one at a time: a singleton's whole meaning is that there is one of
    # it, so an IR copy beside the C's static is a second None and every `is`
    # spanning the halves answers False. The C's own two direct uses of
    # `&apy_none_cell` were redirected through `apy_none()` for the same
    # reason -- see `objects/c/_core.py`.
    "singletons.py": ("apy_none", "apy_from_bool", "apy_ellipsis",
                      "apy_notimplemented", "apy_stop", "apy_is_stop",
                      "apy_is", "apy_id"),
    # THE CLOSURE CELL: one slot, one allocation, nothing else. It needed
    # nothing that was not already ported, which is what made it the next
    # cheapest thing in the runtime.
    "cells.py": ("apy_cell_new", "apy_cell_get", "apy_cell_set"),
    # ASKING WHAT KIND A VALUE IS: the tag at offset 0, and nothing else.
    # These move ahead of the kinds they ask about -- `apy_match_map` can say
    # "this is a dict" long before `dict` itself leaves the C.
    "kinds.py": ("apy_is_instance", "apy_match_seq", "apy_match_map",
                 "apy_as_bool", "apy_to_bool"),
    # `object`'S OWN BEHAVIOUR, which every class inherits and a `super()`
    # whose base chain has run out lands on.
    "defaults.py": ("apy_default_eq", "apy_default_hash", "apy_default_init",
                    "apy_typing_mark", "apy_name_or"),
    # THE FUNCTION OBJECT'S SETTERS: one per property a `def` can have. They
    # only write into a cell `apy_func_new` already laid out, so they need
    # the offsets and nothing else -- the larger half by call count and the
    # smaller half by risk.
    "funcs.py": ("apy_func_cell", "apy_func_default", "apy_func_kwdefaults",
                 "apy_func_kwarg", "apy_func_kwonly", "apy_func_posonly",
                 "apy_func_qualname", "apy_func_module", "apy_func_descr",
                 "apy_func_annotate", "apy_func_builtin",
                 "apy_func_coro", "apy_func_doc", "apy_func_param"),
    # THE GENERATOR FRAME'S accessors -- slots, resume point, what crossed
    # the suspension -- and the four questions `inspect` asks about them.
    "gens.py": ("apy_gen_state", "apy_gen_goto", "apy_gen_set", "apy_gen_iset",
                "apy_gen_iget", "apy_gen_sent", "apy_gen_result",
                "apy_gen_taken", "apy_gen_throwing", "apy_coro_mark",
                "apy_inspect_isgenerator", "apy_inspect_isasyncgen",
                "apy_inspect_iscoroutine", "apy_inspect_iscoroutinefunction",
                "apy_gen_pending", "apy_gen_slot"),
    # PARAMETERISED TYPES -- `list[int]` as a value a program holds, prints
    # and passes to an annotation, and nothing else. Two fields.
    "alias.py": ("apy_alias_new", "apy_alias_unpack", "apy_get_origin",
                 "apy_get_args", "apy_type_builtin"),
    # THE ERROR PATH'S OWN STATE -- the first shared state to move since the
    # singleton cells, and the same argument: the subset cannot read a C
    # static, so the storage crosses before anything that reads it can. The
    # four accessors were added to the C for exactly this and are replaced
    # here; `apy_err_type`/`apy_err_msg` are the harder half and stay put.
    "errstate.py": ("apy_at", "apy_pos_now", "apy_pos_latch",
                    "apy_pos_latched", "apy_handling_now",
                    "apy_error_handling",
                    # THE PENDING ERROR, whose one-line flag ten other
                    # functions are closed over. It could not move without the
                    # STATE moving, which is why these arrive together.
                    "apy_error_occurred", "apy_error_clear",
                    "apy_err_slots", "apy_err_text",
                    # THE SOURCE POSITIONS, moved the same way and for the
                    # same reason. It needs `blocks.py` to double, which is
                    # why it could not have gone at stage 4.
                    "apy_pos_add", "apy_pos_rows", "apy_pos_count",
                    # RAISING. `apy_fail` and `apy_fail2` are the two biggest
                    # walls the survey reports and neither was what it looked
                    # like: one is a bounded string copy and the other is a
                    # template with two strings in it. All three keep their C
                    # names as delegates so 153 call sites do not move.
                    "apy_raise_at", "apy_raise_over", "apy_raise_fmt",
                    # THE FIRST PORTED FUNCTION THAT NAMES A STRING. `rodata`
                    # was declared and unused; without it every caller of
                    # `apy_fail2` stayed in the C however much else moved,
                    # because all 153 of those sites pass literals.
                    "apy_check_bound",
                    # THE USER EXCEPTION TABLE. `apy_exc_register` called
                    # nothing the IR lacked and was still stuck, because it
                    # names a C static -- the third kind of dependency the
                    # survey reports.
                    "apy_exc_register", "apy_user_exc_rows",
                    "apy_user_exc_slot",
                    # THE EXCEPTION HIERARCHY, packed into `rodata` because a
                    # table of POINTERS is the one thing it cannot hold.
                    "apy_exc_parent_of",
                    # HANDLER MATCHING AND THE MESSAGE, which the tree and
                    # the error state between them were the last pieces of.
                    "apy_error_matches", "apy_error_message",
                    # THE BUNDLED MODULE NAMES, packed the way the exception
                    # tree is and compared against the C by the same test.
                    "apy_import"),
}

#: The C functions a ported module takes the FRONT of, per source file.
#:
#: Different from `REPLACES` in exactly one way and it is the important one:
#: the C body is not dropped, it is renamed `<name>_slow` and the ported code
#: calls it for everything it does not handle. `apy_add` is polymorphic over
#: eighteen kinds and its integer case is four instructions, so a port that had
#: to be total could not begin. See `objects/csource.split_c`.
SPLITS: dict[str, tuple[str, ...]] = {
    # THE LAST WALL. Thirteen exported functions were blocked on this one and
    # nothing else; see `runtime/calling.py` for what the fast path takes.
    "calling.py": ("apy_call",),
    # `raise` -- the fast half takes an exception whose argument is already a
    # string, which is what a `raise` statement writes. Rendering any other
    # argument is `apy_str`, polymorphic over every kind there is.
    "excval.py": ("apy_raise",),
    "int_arith.py": ("apy_add", "apy_sub", "apy_mul", "apy_eq",
                     "apy_ne", "apy_lt", "apy_le", "apy_gt",
                     "apy_ge", "apy_bitand", "apy_bitor", "apy_bitxor",
                     "apy_neg", "apy_floordiv", "apy_mod",
                     "apy_rshift", "apy_lshift"),
    # THE FIRST PORTED CODE THAT READS A CELL RATHER THAN BUILDING ONE. All
    # three are polymorphic over every kind and all three DECLINE everything
    # that is not a str or bytes -- which matters most for `apy_truth`, whose
    # default is TRUE: a fast path that read `v.s.n` for a type would read its
    # base pointer instead.
    "str_len.py": ("apy_raw_len", "apy_len", "apy_truth"),
    # THE FIRST PORTED CODE THAT ALLOCATES A BUFFER. `apy_chr` asks the arena
    # for five bytes and hands them to a cell the rest of the runtime reads --
    # which is the question every later kind runs into (who owns the bytes,
    # and what happens when nobody can give them back) asked at the smallest
    # scale there is.
    # TURNING A VALUE INTO TEXT. The fast half takes the scalars and the
    # `str()` of a string; floats need the round-trip printer and a string's
    # repr needs the Unicode table, and both stay in the C.
    "str_code.py": ("apy_ord", "apy_chr", "apy_text_of"),
    # THE LARGEST GROUP SO FAR, and the safest: six entry points over one
    # shared search, and not one of them allocates anything but the integer it
    # answers with. The whole group is verifiable by comparing values, with no
    # question about who owns a buffer.
    "str_find.py": ("apy_str_find", "apy_str_find2", "apy_str_find3",
                    "apy_str_rfind", "apy_str_rfind2", "apy_str_rfind3"),
    # `startswith`/`endswith` in their TWO-ARGUMENT form. The three-argument
    # ones clamp a start and an end through static helpers, so they are left
    # whole -- being separate exported functions, declining them costs a line
    # here rather than a branch there.
    "str_affix.py": ("apy_str_startswith", "apy_str_endswith"),
    "str_is.py": ("apy_str_isalpha", "apy_str_isdigit", "apy_str_isdecimal",
                  "apy_str_isnumeric", "apy_str_isalnum", "apy_str_isspace",
                  "apy_str_isprintable", "apy_str_isascii",
                  "apy_str_islower", "apy_str_isupper", "apy_str_istitle",
                  "apy_str_isidentifier"),
    "str_case.py": ("apy_str_upper", "apy_str_lower", "apy_str_title",
                    "apy_str_capitalize", "apy_str_swapcase",
                    "apy_str_casefold"),
    "str_strip.py": ("apy_str_strip", "apy_str_lstrip", "apy_str_rstrip",
                     "apy_str_strip_chars", "apy_str_lstrip_chars",
                     "apy_str_rstrip_chars", "apy_str_removeprefix",
                     "apy_str_removesuffix"),
    "str_split.py": ("apy_str_split", "apy_str_rsplit"),
    "str_join.py": ("apy_str_join", "apy_str_partition",
                    "apy_str_rpartition"),
    # THE KIND NAMES, which a hundred messages rest on and which `uasm
    # port` has reported at the top since the first survey. A SPLIT because of
    # one case: an exception's DISPLAYED name is a class lookup, not a
    # literal, so it goes back to the C and every other kind stays here.
    # EQUALITY, whose fast half is two integers or two strings and whose
    # slow half is every mixed pair and every container.
    "containers.py": ("apy_hash_raw_of",),
    "cursor.py": ("apy_eq_raw_of",),
    "kindname.py": ("apy_kind_name_of",),
    # ORDERING TWO NUMBERS. The fast half answers everything except a big
    # against a float -- that comparison takes a double apart with `frexp`
    # and `ldexp`, and those are libm.
    "mathints.py": ("apy_num_f_of", "apy_num_order_of",
                    # AN EXPLICITLY WRITTEN `None` BOUND IS NO BOUND:
                    # `xs[None:None]` is `xs[:]`. Which bounds were written
                    # is the frontend's to know and which EVALUATE to None is
                    # not, so the second half is asked at run time.
                    "apy_slice_given", "apy_slice_step"),
    # READING AN ATTRIBUTE. The fast half answers an INSTANCE, which is where
    # a program spends its time; the other nine kinds are a page of field
    # names each and stay in the C.
    "slots.py": ("apy_default_getattr",),
    "str_pad.py": ("apy_str_ljust", "apy_str_rjust", "apy_str_center",
                   "apy_str_ljust_fill", "apy_str_rjust_fill",
                   "apy_str_center_fill", "apy_str_zfill",
                   "apy_str_replace"),
    # THE FIRST PORTED CODE THAT GROWS A BUFFER, which is what stage 5a was
    # built for -- and the first that answers `None`, which is what stage 5b
    # was. It was written before both, refused by the purity test over one
    # call to `apy_none`, and is here because that call is now IR.
    #
    # A SPLIT rather than a replacement because the failure path builds an
    # AttributeError through `apy_fail2` and `apy_kind_name`, both `static` in
    # the C and so unnameable from the subset.
    "list_cell.py": ("apy_seq_push", "apy_getitem", "apy_setitem",
                     "apy_contains"),
}

#: Every C symbol currently provided by IR instead. What `objects_c()` guards.
PORTED = tuple(sorted(
    [n for names in REPLACES.values() for n in names]
    + [n for names in SPLITS.values() for n in names]))

#: Every C symbol that keeps its body under a new name.
SPLIT = tuple(sorted(n for names in SPLITS.values() for n in names))


def sources() -> list[Path]:
    """The runtime modules, in a stable order."""
    return [SOURCE_DIR / name
            for name in sorted(set(REPLACES) | set(SPLITS))]


def compile_runtime(sink=None):
    """Compile every runtime module and return one merged IR module.

    A FRESH SINK BY DEFAULT, and diagnostics from it are a compiler bug rather
    than a user's: this source is shipped, not written by whoever is compiling.
    A caller that passes one wants to see them -- the tests do.
    """
    from ..diagnostics import DiagnosticSink, SourceFile
    from ..frontends.python import PythonFrontend
    from ..ir.module import Module

    own = sink if sink is not None else DiagnosticSink()
    frontend = PythonFrontend()
    # ONE COMPILATION UNIT, not one per file. The runtime is a single library
    # and its pieces call each other -- `int_arith.py` reads the cell layout
    # `int_cell.py` defines -- and the static path resolves a call against the
    # functions of the module being compiled. Compiling separately made every
    # cross-file call `call to unknown function`, and the obvious fix
    # (duplicate the helpers) made the merge refuse two definitions of
    # `apy_obj_size`, which is the same problem wearing a different error.
    #
    # The files stay separate because they are separate SUBJECTS, and the
    # order is `sources()`'s so a build is reproducible.
    text = "\n\n".join(
        f"# ── {path.name} " + "─" * 40 + "\n"
        + path.read_text(encoding="utf-8")
        for path in sources())
    module = frontend.compile(
        SourceFile(text, SOURCE_DIR / "<runtime>"), own, library=True)
    if module is None:
        raise RuntimeError(
            "the shipped object runtime did not compile:\n"
            + "\n".join(f"  {d.code}: {d.message}" for d in own.diagnostics))
    # A RUNTIME FILE CALLING ANOTHER'S FUNCTION gets it declared as an external
    # too: the frontend declares the whole `apy_*` runtime and keeps whatever
    # is called. So `apy_from_int` arrives both DECLARED and DEFINED, which is
    # one name and two functions -- rejected by the verifier, after the splice,
    # against the symbol rather than against this.
    defined = {f.name for f in module.functions if not f.external}
    module.functions = [f for f in module.functions
                        if not (f.external and f.name in defined)]
    return module


def _merge(into, module) -> None:
    """Add `module`'s definitions to `into`, dropping duplicate declarations.

    Two runtime files both calling `apy_alloc` each declare it, and a module
    with the symbol declared twice is rejected by the verifier -- which reports
    it against the second declaration rather than against the merge that made
    it.
    """
    have = {f.name for f in into.functions}
    for fn in module.functions:
        if fn.name in have:
            if not fn.external:
                raise RuntimeError(
                    f"the runtime defines {fn.name} more than once")
            continue
        have.add(fn.name)
        into.functions.append(fn)
    names = {g.name for g in into.globals}
    for g in module.globals:
        if g.name not in names:
            names.add(g.name)
            into.globals.append(g)


def wants_runtime(module) -> bool:
    """Whether this program will have the object runtime's C in it.

    NOT "does the program call a ported function", and not even "does it call
    `apy_*`". The C object runtime calls `apy_from_int` in about 120 places,
    and it is included whole -- so the moment ANY of it is compiled in, the
    ported definition has to be there or the reference is unresolved.

    Both narrower questions were tried and both are wrong in the same way.
    Asking about ported names left it undefined for every dynamic program that
    did not build an integer directly. Asking about `apy_*` left it undefined
    for every STATIC program, which links the runtime for `put_int` and gets
    `objects_c()` along with it -- and that one fails only on the machine
    backends, where the runtime is a separate object file and the linker
    cannot drop what a compiler in one translation unit would have:

        rt.c:(.text+0x248f): undefined reference to `apy_from_int'

    So the question is exactly `objects.support.needs_runtime`, which is what
    decides whether that C is there at all.

    AND THAT WAS ALSO WRONG, for a different reason: it fires for every program
    that merely PRINTS, so ten runtime functions and a two-kilobyte global went
    into programs that never build an object -- which the x86-64 and AArch64
    backends and the lifter all noticed, 101 tests' worth.

    The resolution is that the switch is per-BUILD and reads the module: the C
    omits exactly the definitions this module supplies, so a program with no
    splice keeps the C's. See `omitted_by`.
    """
    from ..ir.opcodes import Op
    return any(ins.sym and ins.sym.startswith("apy_")
               for f in module.functions for b in f.blocks
               for ins in b.instructions if ins.op is Op.CALL)


def omitted_by(module) -> tuple[str, ...]:
    """The ported names THIS module defines, so the C can stand aside.

    ASKED OF THE MODULE rather than threaded through from the splice, because
    the module is what both C emitters already have and because it cannot get
    out of step with itself: the C omits a definition exactly when the IR
    provides one, whatever decided that.
    """
    if module is None:
        return ()
    have = {f.name for f in module.functions if not f.external}
    return tuple(n for n in PORTED if n in have and n not in SPLIT)


def split_by(module) -> tuple[str, ...]:
    """The SPLIT names this module supplies a front half for.

    Separate from `omitted_by` because the two ask the C for opposite things:
    omit drops the body, split keeps it under another name. A function in both
    lists would be a body renamed and then deleted, which is a link error
    naming `apy_add_slow`.
    """
    if module is None:
        return ()
    have = {f.name for f in module.functions if not f.external}
    return tuple(n for n in SPLIT if n in have)


def splice(module, sink=None, *, provided=frozenset(), enabled: bool = True) -> None:
    """Merge the IR runtime into `module`, in place. A no-op if unneeded.

    Called once per compilation, after the frontend and before any backend.
    A program that never touches the object runtime gets nothing -- not even a
    declaration -- so "a statically typed program has no runtime dependencies"
    stays true, which is the property `objects/support.needs_runtime` exists to
    protect and the one a silent splice would quietly end.

    `enabled=False` is the whole-build opt-out (`--object-runtime c`) and
    `provided` is the per-backend one (`Backend.object_runtime`). BOTH EXIST
    ON PURPOSE. The reason to write the runtime in IR is that a backend should
    not have to define 229 functions -- which is an argument for making it
    unnecessary, not for making it compulsory. A backend with its own
    implementation keeps it, and a build that wants the C runtime exactly as
    it was before any of this gets it, unchanged and still tested.

    Whatever is left after both is what the C stands aside for, because the C
    omits exactly what the module defines. So there is one rule and no way for
    the two halves to disagree: if it is not spliced, the C still has it.
    """
    if not enabled or not wants_runtime(module):
        return
    runtime = compile_runtime(sink)
    if provided:
        # DROP WHAT THE BACKEND ALREADY HAS -- and only the replaced names, not
        # the helpers behind them: `apy_int_alloc` is this file's own and a
        # backend claiming `apy_from_int` has no opinion about it. An unreached
        # helper is dropped downstream like any other.
        keep = set(provided) & set(PORTED)
        runtime.functions = [f for f in runtime.functions
                             if f.name not in keep]
    #: A DEFINITION BEATS A DECLARATION. The program declared `apy_from_int`
    #: as an external because the frontend declares the whole runtime that
    #: way; the merged module DEFINES it. Keeping both is two functions of one
    #: name, which the verifier rejects -- correctly, and unhelpfully, because
    #: it names the symbol rather than the splice.
    defined = {f.name for f in runtime.functions if not f.external}
    module.functions = [f for f in module.functions
                        if not (f.external and f.name in defined)]
    _merge(module, runtime)


def check() -> list[str]:
    """Problems with the port, as a list of messages. Empty is good.

    Read by a test rather than by the compiler: this asks whether the tree is
    consistent, which is a question about the tree and not about any one
    program.
    """
    problems: list[str] = []
    module = compile_runtime()
    defined = {f.name for f in module.functions if not f.external}
    for source, names in list(REPLACES.items()) + list(SPLITS.items()):
        for name in names:
            if name not in defined:
                problems.append(
                    f"{source} is declared to replace {name}, and does not "
                    f"define it")
    from .csource import signatures
    known = signatures()
    for name in PORTED:
        if name not in known:
            problems.append(f"{name} is not an APY_API symbol in the C")
    return problems


# ── what to port next ──────────────────────────────────────────────────────
#
# THE QUESTION THIS ANSWERS, and the two wrong ones it replaces.
#
# The first wrong question was "what is blocked by a `static` C helper". It
# gives a big number -- most of the runtime -- and points at `apy_fail2` and
# `apy_kind_name`, and it is wrong because a function that uses no static at
# all can still be unportable: `apy_hasattr` calls `apy_getattr`, and porting
# it would open a hole in the one invariant that matters.
#
# THAT INVARIANT IS CLOSURE: the ported runtime calls nothing it does not
# define. It is what lets a backend owe three functions and no more, and it is
# what `check()` above would have to grow to enforce if anything ever broke it.
#
# The second wrong question was closure ALONE. `apy_fatal_if_error` calls no
# `apy_*` function at all and so looked ready -- but its body is `fflush`,
# `fprintf` and `exit`, and the flush is load-bearing: it orders buffered
# stdout ahead of the message. The floor is `plat_write`, `plat_exit` and
# `plat_heap`, with no flush in it, so ported code writing stderr directly
# would reorder a program's last words against everything it had printed.
#
# SO A CALL TO libc COUNTS AS A CALLEE TOO, and the table below says which
# ones the subset can answer for.

#: libc functions the machine subset can express, and how.
#:
#: WRITTEN OUT, NOT LINKED. `strcmp` is a byte loop and `memcpy` is a byte
#: loop; `runtime/str_find.py` and `runtime/str_join.py` already contain both
#: under other names. A backend with a faster one is free to recognise the
#: shape.
_LIBC_OK = {
    "strcmp": "a byte loop",
    "strlen": "a byte loop",
    "memcmp": "a byte loop",
    "memcpy": "a byte loop",
    "memset": "a store loop",
    "malloc": "apy_alloc_block",
    "calloc": "apy_alloc_block, then cleared",
    "realloc": "apy_realloc_block",
    "free": "apy_free_block",
    "exit": "plat_exit",
}

#: libc functions it cannot, and why. A function reaching any of these stays
#: in the C until the reason is dealt with rather than until someone tries.
_LIBC_NO = {
    "snprintf": "formatting; nothing in the subset builds a string from a "
                "format",
    "sprintf": "formatting",
    "fprintf": "formatting, and stdio ordering",
    "printf": "formatting, and stdio ordering",
    "fputs": "stdio ordering: the floor has no flush",
    "fputc": "stdio ordering: the floor has no flush",
    "fwrite": "stdio ordering: the floor has no flush",
    "fflush": "the floor has no flush",
    "strtod": "parsing a float is its own algorithm",
    "pow": "libm", "sqrt": "libm", "floor": "libm", "ceil": "libm",
    "fabs": "libm", "fmod": "libm", "signbit": "libm", "isnan": "libm",
    "isinf": "libm", "log": "libm", "exp": "libm", "sin": "libm",
    "cos": "libm", "tan": "libm", "atan2": "libm", "frexp": "libm",
    "log10": "libm", "log2": "libm", "log1p": "libm", "expm1": "libm",
    "sinh": "libm", "cosh": "libm", "tanh": "libm", "asin": "libm",
    "acos": "libm", "atan": "libm", "hypot": "libm", "cbrt": "libm",
    "trunc": "libm", "nextafter": "libm", "erf": "libm", "tgamma": "libm",
    "lgamma": "libm", "remainder": "libm",
    "ldexp": "libm", "modf": "libm", "copysign": "libm", "round": "libm",
    "qsort": "a comparison callback the subset cannot describe",
}

_C_KEYWORDS = frozenset((
    "if", "for", "while", "switch", "return", "sizeof", "do", "else",
    "case", "default", "goto", "break", "continue"))

#: MACROS, WHICH ARE NOT CALLS. `O(v)` is the cast from a value to the object
#: it points at and `V(o)` is the cast back; both vanish at compile time, and
#: the subset spells the same thing with `ptr` and `offset`. Counting them as
#: callees made five perfectly portable functions look blocked on a symbol
#: that does not exist.
_C_MACROS = frozenset(("O", "V", "APY_CSTR", "APY_API"))

#: A C static the port has an equivalent for, and what to call instead.
#:
#: DERIVED WHERE IT CAN BE and declared where it cannot. The delegate rules in
#: `survey` spot a static that is one call to something ported; these are the
#: ones whose equivalence is a JUDGEMENT rather than a line of C, so it is
#: written down with its reason instead of inferred.
_STANDS_FOR = {
    "apy_alloc": ("apy_obj_alloc",
                  "the same allocation; `apy_alloc` adds an abort on failure "
                  "where the ported runtime answers null and lets the caller "
                  "propagate, which is the convention `arena.py` set"),
    "apy_gen_step": ("apy_gen_step_of",
                     "the same resume; the delegate WIDENS its `int *done` to "
                     "the `int64_t *` the subset can address, which is why "
                     "the automatic alias check does not see it -- that check "
                     "wants a pure forwarder and this one converts"),
    "apy_exc_construct": ("apy_exc_construct_of",
                          "the same construction; the delegate casts an "
                          "`apy_value *` to the plain word the subset "
                          "declares, for the reason the C comment beside it "
                          "gives"),
}

#: RULED OUT, and why. A name here is closed and expressible and STILL should
#: not move, so the survey stops proposing it -- otherwise the same candidate
#: comes back every time the list is regenerated and someone tries it again.
#:
#: This exists because someone did try it: `apy_str_copy` was ported once and
#: backed out at the first compile.
_WONT = {
    "apy_str_copy":
        "a one-line shim whose whole purpose is the `const char *` to "
        "`apy_value` cast -- 24 call sites pass a `char *`, and IR would "
        "declare it taking a machine word, which is a conflicting type. Its "
        "body is already one call to the ported `apy_str_copy_bytes`, so "
        "moving it would buy nothing and cost the signature.",
    "apy_bytes_copy": "the same shim for bytes; see `apy_str_copy`.",
    "apy_lit":
        "already one line over the ported `apy_from_cstr`, which is the same "
        "function -- porting the delegate too would move a cast and nothing "
        "else, and its `const char *` parameter is what the delegate exists "
        "to absorb.",
    "apy_str_take": "the same, over `apy_from_bytes`; see `apy_lit`.",
    "apy_call_n":
        "two lines over `apy_call_nk`, and promoting that worker to an export "
        "does NOT make this portable: IR calling an unported C function is "
        "exactly the hole the closure invariant forbids, and "
        "`test_the_allocator_asks_the_floor_and_nothing_else` catches it. The "
        "sanctioned way to reach back into C is a SPLIT, whose `_slow` half "
        "is allowed -- so this moves when someone writes a real fast path for "
        "it (a plain function, arity matching, no defaults and no keywords) "
        "and lets everything else go slow. Fourteen functions wait on it.",
    "apy_math_1":
        "takes a `double (*)(double)` and reads `errno`, and the machine "
        "subset has neither: a function POINTER is not one of the types "
        "`signatures()` can describe, and `errno` is a C global no `reserve` "
        "can stand in for. Five `math` functions wait on it and will keep "
        "waiting until libm reaches the platform floor -- which is a decision "
        "about the floor, not a port.",
    "apy_fatal_if_error":
        "its `fflush(stdout)` orders buffered output ahead of the message it "
        "prints, and the platform floor has no flush. Ported code writing "
        "stderr directly would reorder a program's last words against "
        "everything it had already printed.",
}


def _strip_c(text: str) -> str:
    """C with its comments and literals emptied out.

    WITHOUT THIS THE SCAN READS PROSE. Every function in this runtime carries
    a comment, and those comments talk about Python -- so `isinstance(`,
    `range(` and `dict(` all show up as callees, and `type` outranks `memcpy`.
    The first version of this survey reported eleven calls to `complex`.

    ── A SCANNER, AND NOT A REGULAR EXPRESSION ────────────────────────────

    THIS WAS THREE REGULAR EXPRESSIONS AND EVERY ARRANGEMENT OF THEM WAS
    WRONG, because each of the four things being skipped can contain the
    opening of another:

      * a block comment can contain a quote, and does;
      * a line comment can contain an apostrophe, and does, constantly;
      * a string literal can contain `//` and an apostrophe -- dozens do,
        every `"'%s' object has no attribute"` among them;
      * a char literal can contain a double quote, and four do.

    Run any of the four matchers over the whole file and it pairs an opener
    inside something else with a closer far away, deleting everything
    between. The measured damage was not marginal: 753k of C came out as
    114k, 133k or 118k depending on the order, and the survey read the
    remainder while reporting on the whole runtime. An alternation of all
    four in one pass fixes the ORDER but not the anchoring -- a comment
    replaced by a space joins the line before it to the `static` after it,
    which hid forty-eight more functions.

    A left-to-right scan has none of these problems because it is what C's
    own tokeniser is: at every position exactly one thing is being read, and
    what it may contain follows from that.

    WHAT COMES OUT: a comment becomes its own newlines, so line anchoring
    survives; a literal becomes an empty one of its own kind, so nothing
    either side of it runs together; everything else is untouched, so every
    offset a reader might want still means what it meant.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * text.count("\n", i, j))
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                # A BACKSLASH TAKES THE NEXT CHARACTER WITH IT, whatever it
                # is -- that is what makes `'\\''` one literal and not two.
                j += 2 if text[j] == "\\" else 1
            out.append(c * 2)
            i = min(j + 1, n)
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _params(decl: str) -> list[str]:
    """The parameter NAMES of a C declaration, in order.

    THE NAME IS THE LAST WORD OF EACH PARAMETER, after any `*` -- C's grammar
    binds the star to the name, so `const char *msg` is `msg` and not `*msg`.
    A `void` parameter list has none.
    """
    inner = decl[decl.index("(") + 1:decl.rindex(")")].strip()
    if not inner or inner == "void":
        return []
    out = []
    for piece in inner.split(","):
        words = piece.replace("*", " ").split()
        if words:
            out.append(words[-1])
    return out


def _bodies() -> dict[str, tuple[str, list[str]]]:
    """Every function in the C, by name: its body and its parameter names."""
    from .csource import OBJECTS_C

    text = _strip_c(OBJECTS_C)
    out: dict[str, str] = {}
    for m in re.finditer(
            r"^(?:APY_API |static )[^\n{;]*?\b(apy_\w+)\s*\([^;{]*\)\s*\{",
            text, re.M):
        i, depth = m.end(), 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        # WITHOUT THE CLOSING BRACE. `i` lands past it, and a body that
        # carries its own `}` cannot be matched as a single statement --
        # which is how the delegate detection below missed every
        # delegate there was.
        out.setdefault(m.group(1),
                       (text[m.end():i - 1], _params(m.group(0))))
    return out


def _static_data() -> set[str]:
    """The C's file-scope STATIC VARIABLES.

    THE THIRD KIND OF DEPENDENCY, after calls and libc, and the one that has
    caught this survey out twice. A function reading `apy_err_type` cannot
    move however closed its call graph is, because the IR cannot name a C
    static -- and `apy_error_occurred`, whose whole body is one comparison
    against that variable, looked ready for weeks.

    THE WAY PAST IT IS NOT TO PORT THE FUNCTION but to move the STORAGE: an
    exported accessor with a C body, replaced from IR, so both arrangements
    reach the same words. `runtime/errstate.py` did it for the pending error
    and the source positions.

    A DECLARATION INSIDE A FUNCTION IS NOT ONE OF THESE. The pattern anchors
    at the start of a line, and every static local in this runtime is
    indented.
    """
    from .csource import OBJECTS_C

    found = set()
    for m in re.finditer(r"^static [^(){};]*?(\w+)\s*(?:\[[^\]]*\])*\s*(?:=[^;]*)?;",
                         _strip_c(OBJECTS_C), re.M):
        found.add(m.group(1))
    return found


def _forwards(stmt: str, callee: str, args: list[str]) -> bool:
    """Does this one-line body pass its own parameters straight through?

    CASTS ARE IGNORED AND ORDER IS NOT. Every crossing between the C and the
    IR casts a pointer to a machine word, so the text differs from the
    parameter name while the VALUE does not; what must hold is that argument
    `i` mentions parameter `i` and that there are as many of one as the other.
    """
    # FROM THE CALLEE'S OWN PAREN. Searching from `return` missed a
    # `void` delegate, which has none; searching for the last `apy_`
    # found one inside a cast, since `apy_value` starts the same way.
    inside = stmt[stmt.index("(", stmt.index(callee)) + 1:stmt.rindex(")")]
    depth, piece, pieces = 0, [], []
    for ch in inside:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            pieces.append("".join(piece))
            piece = []
        else:
            piece.append(ch)
    if "".join(piece).strip():
        pieces.append("".join(piece))
    if len(pieces) != len(args):
        return False
    return all(re.search(r"\b" + re.escape(a) + r"\b", p)
               for p, a in zip(pieces, args))


def survey() -> dict:
    """What is ported, what could be next, and what stands in the way.

    READY means every callee is already in IR and every libc call is one the
    subset can write out. It is a short list on purpose: a name on it can be
    ported without deciding anything new.

    NEAR IS WHERE THE WORK ACTUALLY IS, and leaving it out was a mistake worth
    recording: the report said "nothing is ready" while three constructors sat
    one small wall away, and finding them meant querying this by hand. A wall
    is only worth calling a wall once you can see what is behind it, so each
    near candidate carries the names it is waiting on.
    """
    from .csource import signatures

    exported = set(signatures())
    # EVERYTHING THE RUNTIME DEFINES, splits included. A split name has a C
    # half too, but that is about which body a LINK uses -- the IR defines the
    # front half, so ported code calling it is calling itself. Excluding
    # splits here said `apy_seq_push` was unavailable to a runtime file that
    # sits next to the one defining it.
    done = set(PORTED)
    bodies = _bodies()

    def callees(name: str) -> set[str]:
        return {c for c in re.findall(r"\b(\w+)\s*\(", bodies.get(name, ("", []))[0])
                if c not in _C_KEYWORDS
                and c not in _C_MACROS} - {name}

    # A STATIC THAT IS ONE CALL IS NOT A WALL, IT IS AN ALIAS. When a
    # body moves to IR the C keeps its old name as a delegate --
    # `apy_fail` is one line calling `apy_raise_at` -- so a function
    # reaching for the static can be ported by reaching for what it
    # delegates to. Counting the delegate as a blocker kept `apy_fail` at
    # the top of this report after it had already moved, which is the
    # report describing its own history rather than the runtime.
    alias: dict[str, str] = {}
    for _name, (_body, _args) in bodies.items():
        # AN EXPORTED DELEGATE IS JUST AS TRANSPARENT. `apy_exc_parent` is
        # exported and one line over `apy_exc_parent_of`; a caller being
        # ported reaches for the second, so the first is no more a wall than
        # a static one is.
        #
        # A NAME ALREADY IN IR IS NOT SKIPPED, which it used to be: its body
        # is where the INVERSE rule below reads from, and skipping it meant
        # `apy_kind_name_of` never taught the survey about `apy_kind_name`.
        # A CAST PREFIX IS STILL A DELEGATE:
        # `return (const char *)(uintptr_t)f(x);`
        # is `f` with the word-vs-pointer seam in it.
        # A `void` DELEGATE HAS NO `return`, and one that insisted on it
        # missed `apy_clamp_range`, whose whole body is a call. The
        # keyword is optional here for the same reason the casts are:
        # what matters is that one call forwards the parameters.
        m = re.fullmatch(r'\s*(?:return\s+)?(?:\([^()]*\)\s*)*(apy_\w+)\s*\(.*?\)\s*;\s*',
                         _body, re.S)
        # IT MUST FORWARD ITS PARAMETERS, which is what makes it an alias
        # rather than one particular call. `apy_list_new(cap)` is one line
        # calling `apy_seq_new(APY_LIST_K, cap)` -- and reading that as
        # "`apy_seq_new` is available, call `apy_list_new`" would lose the
        # kind, quietly turning every tuple and set into a list.
        forwards = m is not None and _forwards(m.group(0), m.group(1), _args)
        if forwards and m.group(1) in done and _name not in done:
            alias[_name] = m.group(1)
        # AND THE OTHER DIRECTION. `apy_kind_name_of` is exported and split,
        # and its C body is one call to the static `apy_kind_name` -- so a
        # function reaching for the static can be ported by reaching for the
        # export. The first rule reads a delegate ONTO ported code; this one
        # reads a delegate FROM it, and both mean the static is not a wall.
        # ONLY FOR A NAME THAT IS NOT ALREADY IN IR. `apy_id` is exported
        # and forwards to `apy_from_int`, which is ported -- recording that as
        # an alias says "reach `apy_from_int` through `apy_id`" about a
        # function nothing is blocked on, and the two do not even take the
        # same type. Every useful inverse alias points at a STATIC.
        if (forwards and _name in exported and _name in PORTED
                and m.group(1) not in done):
            alias.setdefault(m.group(1), _name)

    data = _static_data() - {n for n in exported}
    ready, near, blockers = [], [], Counter()
    reasons: dict[str, str] = {}
    for name in sorted(exported - set(PORTED)):
        if name not in bodies:
            continue
        missing = set()
        # THE STATICS THIS BODY NAMES, which are dependencies as real as its
        # calls: see `_static_data`.
        for word in set(re.findall(r"\w+", bodies.get(name, ("", []))[0])) & data:
            missing.add(word)
            reasons.setdefault(
                word, "a C file-scope static; move the storage behind an "
                      "exported accessor first")
        stands = {k: v for k, (v, _) in _STANDS_FOR.items()}
        for c in {stands.get(c, alias.get(c, c))
                  for c in callees(name)}:
            if c.startswith("apy_"):
                if c not in done:
                    missing.add(c)
            elif c in _LIBC_NO:
                missing.add(c)
                reasons[c] = _LIBC_NO[c]
            elif c not in _LIBC_OK and not c.startswith("py_"):
                missing.add(c)
                reasons.setdefault(c, "unclassified: add it to _LIBC_OK or "
                                      "_LIBC_NO")
        if name in _WONT:
            continue
        if not missing:
            ready.append(name)
        elif len(missing) <= 2:
            near.append((name, sorted(missing)))
            for c in missing:
                blockers[c] += 1
    return {
        "replaced": len(done),
        "split": len(SPLIT),
        "untouched": len(exported - set(PORTED)),
        "ready": ready,
        "near": near,
        "blockers": blockers.most_common(15),
        "reasons": reasons,
        "wont": dict(_WONT),
        "aliases": alias,
    }
