# SPDX-License-Identifier: Apache-2.0
"""Tests for the AST format-reader audit scanner and pytest CI helper."""

from __future__ import annotations

import ast
import json
import os
import pathlib
import subprocess
import sys

import pytest

from omlx.formats.audit import (
    INVALID_PRAGMA_RULE,
    RULES,
    AllowEntry,
    AuditError,
    assert_no_unguarded_readers,
    format_report,
    main,
    scan_paths,
    scan_source,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
QWEN4_COMPAT_DIR = REPO_ROOT / "omlx/patches/mlx_vlm_qwen4_exp_compat"
HC_PREFILL_NAX = QWEN4_COMPAT_DIR / "vendor/mlx_vlm/models/qwen4_exp/hc_prefill_nax.py"


def _rules(source: str, **kwargs) -> set[str]:
    report = scan_source(source, **kwargs)
    return {f.rule for f in report.findings}


def test_detects_each_unguarded_rule():
    assert len(RULES) == 7

    assert "unguarded-weight" in _rules("def f(layer):\n    return layer.weight\n")
    assert "unguarded-scales" in _rules("def f(layer):\n    return layer.scales\n")
    assert "unguarded-biases" in _rules("def f(layer):\n    return layer.biases\n")
    assert "unguarded-weight" in _rules('def f(layer):\n    return layer["weight"]\n')
    assert "unguarded-scales" in _rules('def f(layer):\n    return layer["scales"]\n')
    assert "unguarded-biases" in _rules('def f(layer):\n    return layer["biases"]\n')
    assert "unguarded-weight" in _rules(
        'def f(layer):\n    return getattr(layer, "weight")\n'
    )
    assert "unguarded-scales" in _rules(
        'def f(layer):\n    return getattr(layer, "scales")\n'
    )
    assert "unguarded-biases" in _rules(
        'def f(layer):\n    return getattr(layer, "biases")\n'
    )
    assert "unguarded-quantized-matmul" in _rules(
        "import mlx.core as mx\n"
        "def f(x, w, s, b):\n"
        "    return mx.quantized_matmul(x, w, s, b)\n"
    )
    # Literal mode="affine" with module .weight must still flag unguarded-dequantize.
    deq_rules = _rules(
        "import mlx.core as mx\n"
        "def f(layer):\n"
        '    return mx.dequantize(layer.weight, layer.scales, layer.biases, mode="affine")\n'
    )
    assert "unguarded-dequantize" in deq_rules
    assert "unguarded-weight" in deq_rules
    assert "unguarded-dequantize" in _rules(
        "from mlx.core import dequantize as deq\n"
        "def f(w, s, b):\n"
        "    return deq(w, s, b)\n"
    )
    assert "unguarded-gather-qmm" in _rules(
        "import mlx.core as mx\n"
        "def f(x, w, s, b):\n"
        "    return mx.gather_qmm(x, w, s, b)\n"
    )
    assert "exact-type-check" in _rules(
        "import mlx.nn as nn\n"
        "def f(x):\n"
        "    return type(x) is nn.QuantizedLinear\n"
    )
    assert "exact-type-check" in _rules(
        "def f(x):\n    return type(x) is not QuantizedSwitchLinear\n"
    )
    assert "exact-type-check" in _rules(
        'def f(x):\n    return type(x).__name__ == "QuantizedEmbedding"\n'
    )
    assert "exact-type-check" in _rules(
        "from mlx.nn import QuantizedLinear as QL\n"
        "def f(x):\n"
        "    return x.__class__ is QL\n"
    )


def test_guarded_patterns_produce_zero_findings():
    source = """
import mlx.core as mx
import mlx.nn as nn
from omlx.formats.linear import FormatLinear

class MyRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = mx.ones((dim,))

    def __call__(self, x):
        return x * self.weight

def guarded_isinstance(m, x):
    if isinstance(m, nn.QuantizedLinear):
        return mx.quantized_matmul(x, m.weight, m.scales, m.biases)
    return None

def guarded_early_return(m):
    if not isinstance(m, nn.QuantizedLinear):
        return None
    return m.weight, m.scales, m.biases

def guarded_and_chain(p, dt):
    return (
        isinstance(getattr(p, "weight", None), mx.array)
        and p.weight.dtype == mx.uint32
        and isinstance(getattr(p, "scales", None), mx.array)
        and isinstance(getattr(p, "biases", None), mx.array)
        and p.scales.dtype == dt
        and p.biases.dtype == dt
    )

def guarded_or_chain(m):
    if not hasattr(m, "scales") or m.scales.ndim != 2:
        return None
    return m.scales

def guarded_mode_affine(m, x):
    if getattr(m, "mode", None) == "affine":
        return mx.dequantize(m.weight, m.scales, m.biases)
    return x

def guarded_non_format(m):
    if not isinstance(m, FormatLinear):
        return m.weight
    return None

def safe_probes_and_norms(m, norm, conv1d, weights, tensors):
    _ = getattr(m, "scales", None)
    _ = getattr(m, "biases", None)
    _ = norm.weight
    _ = conv1d.weight
    _ = weights["weight"]
    _ = tensors["scales"]

def allowed_type_checks(proj, x):
    a = type(proj) is FormatLinear
    b = type(proj) is not FormatLinear
    c = type(x) is int
    d = type(x) in (nn.QuantizedLinear, FormatLinear)
    return a, b, c, d

def kv_cache_dequantize(keys_tuple, values_tuple, group_size, bits):
    k = mx.dequantize(keys_tuple[0], keys_tuple[1], keys_tuple[2], group_size=group_size, bits=bits)
    v = mx.dequantize(values_tuple[0], values_tuple[1], values_tuple[2], group_size=group_size, bits=bits)
    return k, v

def raw_mmap_dequantize(buf, scales, biases, group_size, bits):
    return mx.dequantize(buf, scales, biases, group_size=group_size, bits=bits, mode="affine")

def _not_format_layer(s):
    return not isinstance(s, FormatLinear)

class NonFormatSharded:
    def _non_format_layout(self):
        shards = self.shards
        if not shards or not all(_not_format_layer(s) for s in shards):
            return False
        return (1, 1)

    def run(self, x):
        layout = self._non_format_layout()
        if layout:
            first = self.shards[0]
            return mx.quantized_matmul(x, first.weight, first.scales, first.biases)
        return x

    def run_lazy_not(self, x, layout=None):
        if not layout:
            layout = self._non_format_layout()
        if layout:
            first = self.shards[0]
            return mx.quantized_matmul(x, first.weight, first.scales, first.biases)
        return x

    def run_lazy_eq_none(self, x, layout=None):
        if layout == None:
            layout = self._non_format_layout()
        if layout:
            first = self.shards[0]
            return mx.quantized_matmul(x, first.weight, first.scales, first.biases)
        return x
"""
    report = scan_source(source)
    assert report.ok
    assert report.findings == ()


def test_soundness_else_branch_shadowing_and_reassignment():
    # 1. else: branch of isinstance(x, QuantizedLinear) is NOT guarded (FormatLinear lands there).
    else_src = """
import mlx.nn as nn
def fold(lm_head):
    if isinstance(lm_head, nn.QuantizedLinear):
        w = lm_head.weight
    else:
        w = lm_head.weight
    return w
"""
    else_report = scan_source(else_src)
    assert [(f.rule, f.line) for f in else_report.findings] == [("unguarded-weight", 7)]

    # 2. Post-block after `if isinstance(gu, QuantizedSwitchLinear): return ...` is NOT guarded.
    post_src = """
def half(gu):
    if isinstance(gu, QuantizedSwitchLinear):
        return gu["weight"]
    return gu["weight"]
"""
    post_report = scan_source(post_src)
    assert [(f.rule, f.line) for f in post_report.findings] == [("unguarded-weight", 5)]

    # 3. Nested def resets local guards (parameter shadowing / closure soundness).
    nested_src = """
import mlx.nn as nn
def outer(layer):
    if isinstance(layer, nn.QuantizedLinear):
        def inner(layer):
            return layer.weight
        return inner
    return None
"""
    nested_report = scan_source(nested_src)
    assert [(f.rule, f.line, f.scope) for f in nested_report.findings] == [
        ("unguarded-weight", 6, "outer.inner")
    ]

    # 4. Local reassignment invalidates prior guards on that variable.
    reassign_src = """
import mlx.nn as nn
def f(m, other):
    if isinstance(m, nn.QuantizedLinear):
        m = other
        return m.scales
    return None
"""
    reassign_report = scan_source(reassign_src)
    assert [(f.rule, f.line) for f in reassign_report.findings] == [
        ("unguarded-scales", 6)
    ]

    # 5. Attribute reassignment / delete / aug-assign invalidates prior attr guards.
    attr_reassign_src = """
import mlx.nn as nn
class C:
    def f(self, other):
        if isinstance(self.proj, nn.QuantizedLinear):
            self.proj = other
            return self.proj.weight
        return None

    def g(self, other):
        if isinstance(self.proj, nn.QuantizedLinear):
            del self.proj
            return self.proj.weight
        return None

    def h(self, other):
        if isinstance(self.proj, nn.QuantizedLinear):
            self.proj += other
            return self.proj.weight
        return None
"""
    attr_report = scan_source(attr_reassign_src)
    assert [(f.rule, f.line) for f in attr_report.findings] == [
        ("unguarded-weight", 7),
        ("unguarded-weight", 13),
        ("unguarded-weight", 19),
    ]

    # 6. Local reassignment clears op guards (no sticky scope_ops_guarded).
    op_reassign_src = """
import mlx.core as mx
import mlx.nn as nn
def f(m, other, x):
    if isinstance(m, nn.QuantizedLinear):
        m = other
        return mx.quantized_matmul(x, m.weight, m.scales, m.biases)
    return x
"""
    op_reassign_rules = _rules(op_reassign_src)
    assert "unguarded-quantized-matmul" in op_reassign_rules
    assert "unguarded-weight" in op_reassign_rules

    # 7. Guarding receiver `a` must not silence ops on unguarded receiver `b`.
    cross_recv_src = """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, x, s, bi):
    if not isinstance(a, nn.QuantizedLinear):
        return None
    w = getattr(b, "weight", None)
    return mx.quantized_matmul(x, w, s, bi)
"""
    cross_report = scan_source(cross_recv_src)
    assert [(f.rule, f.line) for f in cross_report.findings] == [
        ("unguarded-quantized-matmul", 8)
    ]


def test_scans_match_and_trystar_bodies():
    match_src = """
def f(k, m):
    match k:
        case 1:
            return m.weight
    return None
"""
    assert "unguarded-weight" in _rules(match_src)

    trystar_src = """
def f(m):
    try:
        return m.scales
    except* ValueError:
        return m.biases
"""
    trystar_rules = _rules(trystar_src)
    assert "unguarded-scales" in trystar_rules
    assert "unguarded-biases" in trystar_rules

    shadow_src = """
import mlx.nn as nn
def f(m, x):
    if isinstance(m, nn.QuantizedLinear):
        match x:
            case m:
                return m.weight
    return None
"""
    assert "unguarded-weight" in _rules(shadow_src)

    # Guarded case via match guard stays clean; post-match binding invalidates guard.
    match_guard_src = """
import mlx.nn as nn
def f(m, x):
    match x:
        case y if isinstance(y, nn.QuantizedLinear):
            _ = y.weight
    if isinstance(m, nn.QuantizedLinear):
        match x:
            case [*m]:
                pass
        return m.scales
    return None
"""
    mg_report = scan_source(match_guard_src)
    assert [(f.rule, f.line) for f in mg_report.findings] == [("unguarded-scales", 11)]


def test_early_truthy_return_does_not_guard():
    pred_src = """
import mlx.core as mx
import mlx.nn as nn
from omlx.formats.linear import FormatLinear

def is_qf(m):
    if isinstance(m, FormatLinear):
        return True
    return isinstance(m, nn.QuantizedLinear)

def f(m, x):
    if is_qf(m):
        return mx.quantized_matmul(x, m.weight, m.scales, m.biases)
    return x
"""
    pred_rules = _rules(pred_src)
    assert "unguarded-weight" in pred_rules
    assert "unguarded-quantized-matmul" in pred_rules

    layout_src = """
import mlx.core as mx

def _is_aff(s):
    return getattr(s, "mode", None) == "affine"

class BadSharded:
    def _uniform_affine_layout(self):
        shards = self.shards
        if self.fast:
            return (1, 1)
        if not shards or not all(_is_aff(s) for s in shards):
            return False
        return (2, 2)

    def run(self, x):
        layout = self._uniform_affine_layout()
        if layout:
            first = self.shards[0]
            return mx.quantized_matmul(x, first.weight, first.scales, first.biases)
        return x
"""
    layout_rules = _rules(layout_src)
    assert "unguarded-weight" in layout_rules
    assert "unguarded-quantized-matmul" in layout_rules

    reassign_pred_src = """
import mlx.nn as nn

def bad_reassign(m, other):
    if not isinstance(m, nn.QuantizedLinear):
        return False
    m = other
    return True

def f(m, other):
    if bad_reassign(m, other):
        return m.weight
    return None
"""
    assert "unguarded-weight" in _rules(reassign_pred_src)

    nested_pred_src = """
import mlx.nn as nn

def is_nested_qf(m, flag):
    if not isinstance(m, nn.QuantizedLinear):
        if flag:
            return True
        else:
            return False
    return True

def f(m, flag):
    if is_nested_qf(m, flag):
        return m.weight
    return None
"""
    assert "unguarded-weight" in _rules(nested_pred_src)


def test_getattr_dequantize_and_loop_backedge_reassignment():
    getattr_deq_src = """
import mlx.core as mx
def f(layer, s, b):
    w = getattr(layer, "weight", None)
    return mx.dequantize(w, s, b, mode="affine")
"""
    assert "unguarded-dequantize" in _rules(getattr_deq_src)

    for_backedge_src = """
import mlx.nn as nn
def f(m, xs):
    if isinstance(m, nn.QuantizedLinear):
        for o in xs:
            _ = m.weight
            m = o
"""
    assert "unguarded-weight" in _rules(for_backedge_src)

    while_backedge_src = """
import mlx.nn as nn
def f(m, other, cond):
    if isinstance(m, nn.QuantizedLinear):
        while cond:
            _ = m.weight
            m = other
"""
    assert "unguarded-weight" in _rules(while_backedge_src)

    loop_else_src = """
import mlx.nn as nn
def f(m, xs):
    if isinstance(m, nn.QuantizedLinear):
        for o in xs:
            m = o
        else:
            return m.weight
    return None
"""
    assert "unguarded-weight" in _rules(loop_else_src)


def test_layout_method_param_rebind_and_mutation_do_not_guard():
    base_header = """
import mlx.core as mx

def _is_aff(s):
    return getattr(s, "mode", None) == "affine"

class BadSharded:
    def _uniform_affine_layout(self, flag=False, bad=None):
"""
    base_footer = """
    def run(self, x):
        layout = self._uniform_affine_layout()
        if layout:
            first = self.shards[0]
            return mx.quantized_matmul(x, first.weight, first.scales, first.biases)
        return x
"""
    variants = [
        # (a) Rebinding `self` before reading `self.shards`.
        """        self = None
        shards = self.shards
        if not shards or not all(_is_aff(s) for s in shards):
            return False
        return (2, 2)
""",
        # (b) AugAssign mutation of `self.shards` after guard.
        """        shards = self.shards
        if not shards or not all(_is_aff(s) for s in shards):
            return False
        self.shards += []
        return (2, 2)
""",
        # (c) Delete of `self.shards` after guard.
        """        shards = self.shards
        if not shards or not all(_is_aff(s) for s in shards):
            return False
        del self.shards
        return (2, 2)
""",
        # (d) Multi-target assignment mutating `self.shards` after guard.
        """        shards = self.shards
        if not shards or not all(_is_aff(s) for s in shards):
            return False
        self.shards = a = bad
        return (2, 2)
""",
        # (e) Conditional mutation + truthy return inside `if` after guard.
        """        shards = self.shards
        if not shards or not all(_is_aff(s) for s in shards):
            return False
        if flag:
            self.shards = bad
            return (1, 1)
        return (2, 2)
""",
        # (f) Mutating `self.shards` after aliasing `shards = self.shards`.
        """        shards = self.shards
        self.shards = bad
        if not shards or not all(_is_aff(s) for s in shards):
            return False
        return (2, 2)
""",
    ]
    for idx, body in enumerate(variants):
        rules = _rules(base_header + body + base_footer)
        assert "unguarded-weight" in rules, f"variant {idx} missed unguarded-weight"
        assert (
            "unguarded-quantized-matmul" in rules
        ), f"variant {idx} missed unguarded-quantized-matmul"


def test_taint_survives_control_flow_merges():
    # (a) If/else merge preserves var_unguarded_attr under an unrelated op guard.
    if_qmm_src = """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, c, cond, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        if cond:
            w = getattr(b, "weight", None)
        else:
            w = getattr(c, "weight", None)
        return mx.quantized_matmul(x, w, s, bi)
    return x
"""
    assert "unguarded-quantized-matmul" in _rules(if_qmm_src)

    # (b) If/else merge preserves var_reads_module_attr for literal mode="affine".
    if_deq_src = """
import mlx.core as mx
def f(layer, cond, s, b):
    if cond:
        w = getattr(layer, "weight", None)
    else:
        w = getattr(layer, "weight", None)
    return mx.dequantize(w, s, b, mode="affine")
"""
    assert "unguarded-dequantize" in _rules(if_deq_src)

    # (c) For-loop merge preserves taint into post-loop op and loop orelse.
    for_qmm_src = """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, xs, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = None
        for _ in xs:
            w = getattr(b, "weight", None)
        return mx.quantized_matmul(x, w, s, bi)
    return x
"""
    assert "unguarded-quantized-matmul" in _rules(for_qmm_src)

    for_orelse_deq_src = """
import mlx.core as mx
def f(layer, xs, s, b):
    w = None
    for _ in xs:
        w = getattr(layer, "weight", None)
    else:
        return mx.dequantize(w, s, b, mode="affine")
"""
    assert "unguarded-dequantize" in _rules(for_orelse_deq_src)

    # (d) While-loop merge and orelse preserve taint.
    while_qmm_src = """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, cond, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = None
        while cond:
            w = getattr(b, "weight", None)
        return mx.quantized_matmul(x, w, s, bi)
    return x
"""
    assert "unguarded-quantized-matmul" in _rules(while_qmm_src)

    while_orelse_deq_src = """
import mlx.core as mx
def f(layer, cond, s, b):
    w = None
    while cond:
        w = getattr(layer, "weight", None)
    else:
        return mx.dequantize(w, s, b, mode="affine")
"""
    assert "unguarded-dequantize" in _rules(while_orelse_deq_src)

    # (e) Try and Match merges preserve taint.
    try_qmm_src = """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = None
        try:
            w = getattr(b, "weight", None)
        except ValueError:
            pass
        return mx.quantized_matmul(x, w, s, bi)
    return x
"""
    assert "unguarded-quantized-matmul" in _rules(try_qmm_src)

    match_qmm_src = """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, k, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = None
        match k:
            case 1:
                w = getattr(b, "weight", None)
            case _:
                w = 0
        return mx.quantized_matmul(x, w, s, bi)
    return x
"""
    assert "unguarded-quantized-matmul" in _rules(match_qmm_src)

    # (f) Loop back-edge taint and pre-loop taint before body overwrite.
    loop_backedge_taint_src = """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, xs, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = 0
        for _ in xs:
            mx.quantized_matmul(x, w, s, bi)
            w = getattr(b, "weight", None)
"""
    assert "unguarded-quantized-matmul" in _rules(loop_backedge_taint_src)

    loop_pre_taint_src = """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, xs, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        for _ in xs:
            mx.quantized_matmul(x, w, s, bi)
            w = 0
"""
    assert "unguarded-quantized-matmul" in _rules(loop_pre_taint_src)

    # (g) Mid-block taint before exception in try survives into except.
    try_mid_taint_src = """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, x, s, bi, work):
    if isinstance(a, nn.QuantizedLinear):
        w = 0
        try:
            w = getattr(b, "weight", None)
            work()
            w = 0
        except ValueError:
            return mx.quantized_matmul(x, w, s, bi)
    return x
"""
    assert "unguarded-quantized-matmul" in _rules(try_mid_taint_src)


def test_try_reassignment_invalidates_except_else_finally():
    # (a) Reassignment in try invalidates guard in except.
    try_except_src = """
import mlx.nn as nn
def f(m, other):
    if isinstance(m, nn.QuantizedLinear):
        try:
            m = other
            raise ValueError()
        except ValueError:
            return m.weight
    return None
"""
    assert "unguarded-weight" in _rules(try_except_src)

    # (b) Same with except* (TryStar).
    trystar_except_src = """
import mlx.nn as nn
def f(m, other):
    if isinstance(m, nn.QuantizedLinear):
        try:
            m = other
            raise ValueError()
        except* ValueError:
            return m.weight
    return None
"""
    assert "unguarded-weight" in _rules(trystar_except_src)

    # (c) Reassignment in try invalidates guard in else.
    try_else_src = """
import mlx.nn as nn
def f(m, other):
    if isinstance(m, nn.QuantizedLinear):
        try:
            m = other
        except ValueError:
            return None
        else:
            return m.weight
    return None
"""
    assert "unguarded-weight" in _rules(try_else_src)

    # (d) Reassignment in try invalidates guard in finally.
    try_finally_src = """
import mlx.nn as nn
def f(m, other):
    if isinstance(m, nn.QuantizedLinear):
        try:
            m = other
        finally:
            _ = m.weight
    return None
"""
    assert "unguarded-weight" in _rules(try_finally_src)

    # (e) ExceptHandler.name invalidates guard after try.
    except_as_post_src = """
import mlx.nn as nn
def f(m):
    if isinstance(m, nn.QuantizedLinear):
        try:
            pass
        except ValueError as m:
            pass
        return m.weight
    return None
"""
    assert "unguarded-weight" in _rules(except_as_post_src)

    # (f) handler.type sees try body invalidation.
    handler_type_src = """
import mlx.nn as nn
def f(m, other):
    if isinstance(m, nn.QuantizedLinear):
        try:
            m = other
        except (ValueError, m.weight):
            pass
"""
    assert "unguarded-weight" in _rules(handler_type_src)

    # (g) finally sees except body, except `as`, and else reassignments.
    for snippet in (
        """
import mlx.nn as nn
def f(m, other):
    if isinstance(m, nn.QuantizedLinear):
        try:
            pass
        except ValueError:
            m = other
        finally:
            _ = m.weight
""",
        """
import mlx.nn as nn
def f(m):
    if isinstance(m, nn.QuantizedLinear):
        try:
            pass
        except ValueError as m:
            pass
        finally:
            _ = m.weight
""",
        """
import mlx.nn as nn
def f(m, other):
    if isinstance(m, nn.QuantizedLinear):
        try:
            pass
        except ValueError:
            pass
        else:
            m = other
        finally:
            _ = m.weight
""",
    ):
        assert "unguarded-weight" in _rules(snippet)

    # (h) TryStar sequential except* handlers propagate invalidation & taint.
    trystar_seq_src = """
import mlx.nn as nn
def f(m, other, work):
    if isinstance(m, nn.QuantizedLinear):
        try:
            work()
        except* ValueError:
            m = other
        except* TypeError:
            return m.weight
    return None
"""
    assert "unguarded-weight" in _rules(trystar_seq_src)


def test_match_pattern_expressions_are_scanned():
    src = """
def f(layer, x):
    match x:
        case layer.weight:
            pass
        case {layer.scales: v}:
            pass
        case layer.biases():
            pass
"""
    rules = _rules(src)
    assert "unguarded-weight" in rules
    assert "unguarded-scales" in rules
    assert "unguarded-biases" in rules

    nested_src = """
def f(layer, x):
    match x:
        case [1, {layer.scales: layer.biases}]:
            pass
"""
    nested_rules = _rules(nested_src)
    assert "unguarded-scales" in nested_rules
    assert "unguarded-biases" in nested_rules


def test_match_guarded_case_binding_invalidates_following_cases():
    guarded_src = """
import mlx.nn as nn
def f(m, x):
    if isinstance(m, nn.QuantizedLinear):
        match x:
            case m if False:
                pass
            case _:
                return m.weight
    return None
"""
    assert "unguarded-weight" in _rules(guarded_src)

    # Guarded case binding also invalidates later case pattern expressions.
    guarded_pattern_src = """
import mlx.nn as nn
def f(m, x):
    if isinstance(m, nn.QuantizedLinear):
        match x:
            case m if False:
                pass
            case m.scales:
                pass
"""
    assert "unguarded-scales" in _rules(guarded_pattern_src)

    # MatchMapping rest (`{**m}`) invalidates in-case, across guarded cases, and post-match.
    mapping_rest_src = """
import mlx.nn as nn
def f(m, x):
    if isinstance(m, nn.QuantizedLinear):
        match x:
            case {**m} if False:
                pass
            case _:
                return m.weight
    return None

def g(m, x):
    if isinstance(m, nn.QuantizedLinear):
        match x:
            case {**m}:
                return m.weight
    return None

def h(m, x):
    if isinstance(m, nn.QuantizedLinear):
        match x:
            case {**m}:
                pass
        return m.scales
    return None
"""
    mr_report = scan_source(mapping_rest_src)
    assert [(f.scope, f.rule) for f in mr_report.findings] == [
        ("f", "unguarded-weight"),
        ("g", "unguarded-weight"),
        ("h", "unguarded-scales"),
    ]


def test_with_and_async_with_soundness():
    # (a) Exception-suppressing `with` / `async with` must not leak guards established inside.
    with_guard_leak_src = """
import mlx.nn as nn
def f(m, ctx):
    with ctx():
        if not isinstance(m, nn.QuantizedLinear):
            return None
    return m.weight

async def g(m, ctx):
    async with ctx():
        if not isinstance(m, nn.QuantizedLinear):
            return None
    return m.weight
"""
    wg_report = scan_source(with_guard_leak_src)
    assert [(f.scope, f.rule) for f in wg_report.findings] == [
        ("f", "unguarded-weight"),
        ("g", "unguarded-weight"),
    ]

    # (b) Taint from context_expr flows into `as` target (quantized_matmul & dequantize).
    with_as_taint_src = """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, x, s, bi, ctx):
    if isinstance(a, nn.QuantizedLinear):
        with ctx(getattr(b, "weight", None)) as w:
            return mx.quantized_matmul(x, w, s, bi)
    return x

def g(layer, s, b, ctx):
    with ctx(getattr(layer, "weight", None)) as w:
        return mx.dequantize(w, s, b, mode="affine")
"""
    wat_rules = _rules(with_as_taint_src)
    assert "unguarded-quantized-matmul" in wat_rules
    assert "unguarded-dequantize" in wat_rules

    # (c) Mid-block taint inside `with` survives after `with` even if later overwritten.
    with_mid_taint_src = """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, x, s, bi, ctx, work):
    if isinstance(a, nn.QuantizedLinear):
        w = 0
        with ctx():
            w = getattr(b, "weight", None)
            work()
            w = 0
        return mx.quantized_matmul(x, w, s, bi)
    return x
"""
    assert "unguarded-quantized-matmul" in _rules(with_mid_taint_src)

    # (d) `with ctx() as m:` invalidates prior guard inside and after `with`.
    with_as_shadow_src = """
import mlx.nn as nn
def f(m, ctx):
    if isinstance(m, nn.QuantizedLinear):
        with ctx() as m:
            _ = m.weight
        return m.scales
    return None
"""
    was_rules = _rules(with_as_shadow_src)
    assert "unguarded-weight" in was_rules
    assert "unguarded-scales" in was_rules


def test_compound_neighbours_walrus_augassign_match_subject_and_defs():
    # (a) Walrus operator invalidates guarded variable and propagates taint.
    walrus_guard_src = """
import mlx.nn as nn
def f(m, other):
    if isinstance(m, nn.QuantizedLinear):
        if (m := other):
            return m.weight
    return None
"""
    assert "unguarded-weight" in _rules(walrus_guard_src)

    walrus_taint_src = """
import mlx.core as mx
def f(layer, s, b):
    if (w := getattr(layer, "weight", None)) is not None:
        return mx.dequantize(w, s, b, mode="affine")
    return None
"""
    assert "unguarded-dequantize" in _rules(walrus_taint_src)

    # (b) AugAssign reads target tensor attr and preserves/acquires taint.
    aug_src = """
import mlx.core as mx
import mlx.nn as nn
def f(layer):
    layer.weight += 1

def g(a, b, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        w += 0
        return mx.quantized_matmul(x, w, s, bi)
    return x
"""
    aug_rules = _rules(aug_src)
    assert "unguarded-weight" in aug_rules
    assert "unguarded-quantized-matmul" in aug_rules

    # (c) Match subject taint flows into pattern binding.
    match_subj_taint_src = """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        match getattr(b, "weight", None):
            case w:
                return mx.quantized_matmul(x, w, s, bi)
    return x
"""
    assert "unguarded-quantized-matmul" in _rules(match_subj_taint_src)

    # (d) For-loop and comprehension target taint from iterable.
    iter_taint_src = """
import mlx.core as mx
def f(layer, s, b):
    for w in [getattr(layer, "weight", None)]:
        return mx.dequantize(w, s, b, mode="affine")

def g(layer, s, b):
    return [
        mx.dequantize(w, s, b, mode="affine")
        for w in [getattr(layer, "weight", None)]
    ]
"""
    it_report = scan_source(iter_taint_src)
    assert [(f.scope, f.rule) for f in it_report.findings] == [
        ("f", "unguarded-dequantize"),
        ("g", "unguarded-dequantize"),
    ]

    # (e) Nested def/class/import rebinds guarded variable.
    rebind_def_src = """
import mlx.nn as nn
def f(m):
    if isinstance(m, nn.QuantizedLinear):
        def m():
            pass
        return m.weight
    return None

def g(m):
    if isinstance(m, nn.QuantizedLinear):
        import math as m
        return m.weight
    return None
"""
    rd_report = scan_source(rebind_def_src)
    assert [(f.scope, f.rule) for f in rd_report.findings] == [
        ("f", "unguarded-weight"),
        ("g", "unguarded-weight"),
    ]

    # (f) Layout variable assigned in only one branch of non-`is None` if must not guard.
    layout_branch_src = """
import mlx.core as mx

def _is_aff(s):
    return getattr(s, "mode", None) == "affine"

class Sharded:
    def _uniform_affine_layout(self):
        shards = self.shards
        if not shards or not all(_is_aff(s) for s in shards):
            return False
        return (2, 2)

    def run(self, x, cond):
        if cond:
            layout = self._uniform_affine_layout()
        else:
            layout = True
        if layout:
            first = self.shards[0]
            return mx.quantized_matmul(x, first.weight, first.scales, first.biases)
        return x

    def run_one_sided(self, x, cond):
        layout = True
        if cond:
            layout = self._uniform_affine_layout()
        if layout:
            first = self.shards[0]
            return mx.quantized_matmul(x, first.weight, first.scales, first.biases)
        return x
"""
    lb_report = scan_source(layout_branch_src)
    lb_by_scope = {(f.scope, f.rule) for f in lb_report.findings}
    assert ("Sharded.run", "unguarded-weight") in lb_by_scope
    assert ("Sharded.run", "unguarded-quantized-matmul") in lb_by_scope
    assert ("Sharded.run_one_sided", "unguarded-weight") in lb_by_scope
    assert ("Sharded.run_one_sided", "unguarded-quantized-matmul") in lb_by_scope

    # (g) Nested combination: with -> for -> try reassignment + taint.
    nested_combo_src = """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, m, other, xs, x, s, bi, ctx):
    if isinstance(a, nn.QuantizedLinear) and isinstance(m, nn.QuantizedLinear):
        w = 0
        with ctx():
            for _ in xs:
                try:
                    m = other
                    w = getattr(b, "weight", None)
                except ValueError:
                    _ = m.weight
        return mx.quantized_matmul(x, w, s, bi)
    return x
"""
    nc_rules = _rules(nested_combo_src)
    assert "unguarded-weight" in nc_rules
    assert "unguarded-quantized-matmul" in nc_rules


def test_walrus_after_guard_in_boolop_does_not_leak():
    # HIGH-1: walrus rebinding after a guard in ast.And leaks stale guard if not invalidated.
    and_leak_src = """
import mlx.nn as nn
def f(m, other):
    if isinstance(m, nn.QuantizedLinear) and (m := other):
        return m.weight
    return None
"""
    assert "unguarded-weight" in _rules(and_leak_src)

    # ast.Or equivalent: early-return neg path must not keep stale guard on rebound m.
    or_leak_src = """
import mlx.nn as nn
def f(m, other):
    if not isinstance(m, nn.QuantizedLinear) or not (m := other):
        return None
    return m.weight
"""
    assert "unguarded-weight" in _rules(or_leak_src)

    # Guard AFTER walrus in ast.And / ast.Or remains clean (0 findings).
    safe_src = """
import mlx.nn as nn
def f_and(m, other):
    if (m := other) and isinstance(m, nn.QuantizedLinear):
        return m.weight
    return None

def f_or(m, other):
    if not (m := other) or not isinstance(m, nn.QuantizedLinear):
        return None
    return m.weight
"""
    safe_report = scan_source(safe_src)
    assert safe_report.ok
    assert safe_report.findings == ()

    # Dotted receiver (self.proj) and seq_guards (self.shards) invalidated by `(self := other)`.
    dotted_and_seq_src = """
import mlx.core as mx
import mlx.nn as nn

def _is_aff(s):
    return getattr(s, "mode", None) == "affine"

class C:
    def _uniform_affine_layout(self):
        shards = self.shards
        if not shards or not all(_is_aff(s) for s in shards):
            return False
        return (1, 1)

    def f_attr_and(self, other):
        if isinstance(self.proj, nn.QuantizedLinear) and (self := other):
            return self.proj.weight
        return None

    def f_attr_or(self, other):
        if not isinstance(self.proj, nn.QuantizedLinear) or not (self := other):
            return None
        return self.proj.weight

    def f_seq_and(self, other):
        layout = self._uniform_affine_layout()
        if layout and (self := other):
            first = self.shards[0]
            return first.weight
        return None

    def f_seq_or(self, other):
        layout = self._uniform_affine_layout()
        if not layout or not (self := other):
            return None
        first = self.shards[0]
        return first.weight

    def f_all_seq_and(self, shards, other):
        if all(_is_aff(s) for s in shards) and (shards := other):
            first = shards[0]
            return first.weight
        return None

    def f_all_self_seq_or(self, other):
        if not all(_is_aff(s) for s in self.shards) or not (self := other):
            return None
        first = self.shards[0]
        return first.weight
"""
    ds_report = scan_source(dotted_and_seq_src)
    assert [(f.scope, f.rule) for f in ds_report.findings] == [
        ("C.f_attr_and", "unguarded-weight"),
        ("C.f_attr_or", "unguarded-weight"),
        ("C.f_seq_and", "unguarded-weight"),
        ("C.f_seq_or", "unguarded-weight"),
        ("C.f_all_seq_and", "unguarded-weight"),
        ("C.f_all_self_seq_or", "unguarded-weight"),
    ]

    # Leaf expression containing walrus (evaluated after checked target) must not guard.
    leaf_walrus_src = """
import mlx.nn as nn
def f_isinstance(m, other):
    if isinstance(m, (nn.QuantizedLinear, (m := other))):
        return m.weight
    return None

def f_getattr_mode(m, other):
    if getattr(m, "mode", (m := other)) == "affine":
        return m.weight
    return None
"""
    lw_report = scan_source(leaf_walrus_src)
    assert [(f.scope, f.rule) for f in lw_report.findings] == [
        ("f_isinstance", "unguarded-weight"),
        ("f_getattr_mode", "unguarded-weight"),
    ]


def test_walrus_boolop_neighbour_positions():
    # Every test/guard position that calls _extract_facts: While, Assert, IfExp,
    # Match guard, Comprehension if/back-edge, and nested BoolOp.
    flagged_src = """
import mlx.nn as nn

def f_while(m, other):
    while isinstance(m, nn.QuantizedLinear) and (m := other):
        return m.weight
    return None

def f_assert(m, other):
    assert isinstance(m, nn.QuantizedLinear) and (m := other)
    return m.weight

def f_ifexp_and(m, other):
    return m.weight if (isinstance(m, nn.QuantizedLinear) and (m := other)) else None

def f_ifexp_or(m, other):
    return None if (not isinstance(m, nn.QuantizedLinear) or not (m := other)) else m.weight

def f_match_guard(x, other):
    match x:
        case y if isinstance(y, nn.QuantizedLinear) and (y := other):
            return y.weight
    return None

def f_comp_if(m, other, xs):
    return [m.weight for _ in xs if isinstance(m, nn.QuantizedLinear) and (m := other)]

def f_comp_backedge(m, other, xs):
    if isinstance(m, nn.QuantizedLinear):
        return [(m.weight, (m := other)) for _ in xs]
    return []

def f_nested_boolop(m, other):
    return (isinstance(m, nn.QuantizedLinear) and (m := other)) and m.weight
"""
    flagged_report = scan_source(flagged_src)
    assert [(f.scope, f.rule) for f in flagged_report.findings] == [
        ("f_while", "unguarded-weight"),
        ("f_assert", "unguarded-weight"),
        ("f_ifexp_and", "unguarded-weight"),
        ("f_ifexp_or", "unguarded-weight"),
        ("f_match_guard", "unguarded-weight"),
        ("f_comp_if", "unguarded-weight"),
        ("f_comp_backedge", "unguarded-weight"),
        ("f_nested_boolop", "unguarded-weight"),
    ]

    # Filtered all(...) generator must not certify the whole sequence.
    filtered_all_src = """
import mlx.core as mx
def _is_aff(s):
    return getattr(s, "mode", None) == "affine"

def f_filtered_all(shards, flag):
    if all(_is_aff(s) for s in shards if flag):
        first = shards[0]
        return first.weight
    return None
"""
    assert "unguarded-weight" in _rules(filtered_all_src)

    # Safe walrus-before-guard equivalents stay clean (0 findings).
    safe_neighbours_src = """
import mlx.nn as nn

def s_while(m, other):
    while (m := other) and isinstance(m, nn.QuantizedLinear):
        return m.weight
    return None

def s_assert(m, other):
    assert (m := other) and isinstance(m, nn.QuantizedLinear)
    return m.weight

def s_ifexp(m, other):
    return m.weight if ((m := other) and isinstance(m, nn.QuantizedLinear)) else None

def s_match_guard(x, other):
    match x:
        case _ if (m := other) and isinstance(m, nn.QuantizedLinear):
            return m.weight
    return None

def s_comp_if(m, other, xs):
    return [m.weight for _ in xs if (m := other) and isinstance(m, nn.QuantizedLinear)]
"""
    sn_report = scan_source(safe_neighbours_src)
    assert sn_report.ok
    assert sn_report.findings == ()


def test_conditional_rebind_preserves_prior_taint():
    # HIGH-2: all four arbiter snippets + BoolOp(Or), TryStar body, Match guard walrus,
    # dequantize(mode="affine"), and For/While/With/Try/IfExp conditional rebinds.
    snippets = {
        "boolop_and": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, cond, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        if cond and (w := 0):
            pass
        return mx.quantized_matmul(x, w, s, bi)
    return x
""",
        "boolop_or": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, cond, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        if cond or (w := 0):
            pass
        return mx.quantized_matmul(x, w, s, bi)
    return x
""",
        "comprehension": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, xs, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        _ = [(w := 0) for _ in xs]
        return mx.quantized_matmul(x, w, s, bi)
    return x
""",
        "trystar_as": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, x, s, bi, work):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        try:
            work()
        except* ValueError as w:
            pass
        except* TypeError:
            return mx.quantized_matmul(x, w, s, bi)
    return x
""",
        "trystar_body": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, x, s, bi, work):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        try:
            work()
        except* ValueError as w:
            w = 0
        except* TypeError:
            return mx.quantized_matmul(x, w, s, bi)
    return x
""",
        "comp_backedge_taint": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, xs, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        _ = [(mx.quantized_matmul(x, w, s, bi), (w := 0)) for _ in xs]
    return x
""",
        "try_assert_msg_taint": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = 0
        try:
            assert False, (w := getattr(b, "weight", None))
        except AssertionError:
            return mx.quantized_matmul(x, w, s, bi)
    return x
""",
        "match_pattern": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, k, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        match k:
            case [w] if False:
                pass
            case _:
                return mx.quantized_matmul(x, w, s, bi)
    return x
""",
        "match_guard_walrus": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, k, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        match k:
            case 1 if (w := 0):
                pass
            case _:
                return mx.quantized_matmul(x, w, s, bi)
    return x
""",
        "for_rebind": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, xs, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        for _ in xs:
            w = 0
        return mx.quantized_matmul(x, w, s, bi)
    return x
""",
        "while_or_walrus_in_body": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, cond, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        while cond or (w := 0):
            return mx.quantized_matmul(x, w, s, bi)
    return x
""",
        "with_rebind": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, ctx, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        with ctx():
            w = 0
        return mx.quantized_matmul(x, w, s, bi)
    return x
""",
        "try_rebind": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, work, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        try:
            work()
            w = 0
        except ValueError:
            return mx.quantized_matmul(x, w, s, bi)
    return x
""",
        "ifexp_rebind": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, cond, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        _ = (w := 0) if cond else 1
        return mx.quantized_matmul(x, w, s, bi)
    return x
""",
    }
    for name, src in snippets.items():
        assert "unguarded-quantized-matmul" in _rules(
            src
        ), f"{name} lost var_unguarded_attr taint"

    # Also verify var_reads_module_attr survives conditional rebind for mode="affine".
    deq_snippets = {
        "boolop_and_deq": """
import mlx.core as mx
def f(layer, cond, s, b):
    w = getattr(layer, "weight", None)
    if cond and (w := 0):
        pass
    return mx.dequantize(w, s, b, mode="affine")
""",
        "comp_deq": """
import mlx.core as mx
def f(layer, xs, s, b):
    w = getattr(layer, "weight", None)
    _ = [(w := 0) for _ in xs]
    return mx.dequantize(w, s, b, mode="affine")
""",
    }
    for name, src in deq_snippets.items():
        assert "unguarded-dequantize" in _rules(
            src
        ), f"{name} lost var_reads_module_attr taint"


def test_module_star_import_invalidates_guards():
    # LOW-1: straight-line module-level `from q import *` invalidates guards.
    straight_src = """
import mlx.nn as nn
if isinstance(m, nn.QuantizedLinear):
    from q import *
    _ = m.weight
"""
    assert "unguarded-weight" in _rules(straight_src)

    # Conditional module-level `from q import *` invalidates guards after `if`.
    cond_src = """
import mlx.nn as nn
if isinstance(m, nn.QuantizedLinear):
    if cond:
        from q import *
    _ = m.weight
"""
    assert "unguarded-weight" in _rules(cond_src)

    # Star import must not erase pre-existing taint on local/module vars.
    taint_src = """
import mlx.core as mx
import mlx.nn as nn
if isinstance(a, nn.QuantizedLinear):
    w = getattr(b, "weight", None)
    from q import *
    _ = mx.dequantize(w, s, bi, mode="affine")
"""
    assert "unguarded-dequantize" in _rules(taint_src)

    # LOW-2: module-level `from q import *` invalidates `seq_guards`.
    seq_src = """
def _is_aff(s):
    return getattr(s, "mode", None) == "affine"

if all(_is_aff(s) for s in shards):
    from q import *
    first = shards[0]
    _ = first.weight
"""
    assert "unguarded-weight" in _rules(seq_src)


def test_comprehension_backedge_walrus_taint():
    # MEDIUM-1: taint created by a walrus in elt or an if-clause must flow
    # across comprehension iterations via the seed taint fixpoint.
    snippets = {
        "elt_before_walrus": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, xs, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = 0
        _ = [
            (mx.quantized_matmul(x, w, s, bi), (w := getattr(b, "weight", None)))
            for _ in xs
        ]
    return x
""",
        "if_before_elt_walrus": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, xs, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = 0
        _ = [
            (w := getattr(b, "weight", None))
            for _ in xs
            if mx.quantized_matmul(x, w, s, bi) is not None
        ]
    return x
""",
        "if_walrus_overwritten_by_elt": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, xs, x, s, bi, cond):
    if isinstance(a, nn.QuantizedLinear):
        w = 0
        _ = [
            (w := 0)
            for _ in xs
            if (mx.quantized_matmul(x, w, s, bi) or (w := getattr(b, "weight", None)))
            if cond
        ]
    return x
""",
        "post_comp_failing_filter_qmm": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, xs, x, s, bi, cond):
    if isinstance(a, nn.QuantizedLinear):
        w = 0
        _ = [(w := 0) for _ in xs if (w := getattr(b, "weight", None)) if cond]
        return mx.quantized_matmul(x, w, s, bi)
    return x
""",
    }
    for name, src in snippets.items():
        assert "unguarded-quantized-matmul" in _rules(
            src
        ), f"{name} missed unguarded-quantized-matmul"

    post_comp_deq = """
import mlx.core as mx
def f(b, xs, s, bi, cond):
    w = 0
    _ = [(w := 0) for _ in xs if (w := getattr(b, "weight", None)) if cond]
    return mx.dequantize(w, s, bi, mode="affine")
"""
    assert "unguarded-dequantize" in _rules(post_comp_deq)

    # Multi-pass fixpoint must discard intermediate findings (no duplicates).
    dedup_src = """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, xs, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = 0
        _ = [
            (b.weight, mx.quantized_matmul(x, w, s, bi), (w := getattr(b, "weight", None)))
            for _ in xs
        ]
    return x
"""
    dedup_findings = scan_source(dedup_src).findings
    assert [f.rule for f in dedup_findings] == [
        "unguarded-weight",
        "unguarded-quantized-matmul",
    ]

    # Safe counterpart: untainted walrus rebind stays clean.
    clean_src = """
import mlx.core as mx
import mlx.nn as nn
def f(a, xs, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = 0
        _ = [(mx.quantized_matmul(x, w, s, bi), (w := 0)) for _ in xs]
    return x
"""
    assert scan_source(clean_src).findings == ()


def test_boolop_first_operand_walrus_clears_taint():
    # LOW-1: the first operand of `and` / `or` runs unconditionally, so a
    # top-level walrus clearing `w` there clears stale taint (no false positive).
    clean_snippets = {
        "and_first_operand": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, cond, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        if (w := 0) and cond:
            pass
        return mx.quantized_matmul(x, w, s, bi)
    return x
""",
        "or_first_operand": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, cond, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        if (w := 0) or cond:
            pass
        return mx.quantized_matmul(x, w, s, bi)
    return x
""",
        "straight_line_clear": """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        w = 0
        return mx.quantized_matmul(x, w, s, bi)
    return x
""",
    }
    for name, src in clean_snippets.items():
        assert "unguarded-quantized-matmul" not in _rules(
            src
        ), f"{name} falsely reported unguarded-quantized-matmul"

    # Nested conditional walrus inside values[0] must still preserve taint.
    nested_cond_src = """
import mlx.core as mx
import mlx.nn as nn
def f(a, b, cond1, cond2, x, s, bi):
    if isinstance(a, nn.QuantizedLinear):
        w = getattr(b, "weight", None)
        if (cond1 and (w := 0)) and cond2:
            pass
        return mx.quantized_matmul(x, w, s, bi)
    return x
"""
    assert "unguarded-quantized-matmul" in _rules(nested_cond_src)


def test_nested_walrus_then_guard_is_clean():
    # NIT-1 (Option A): facts established after a walrus inside a nested BoolOp
    # must survive into the outer BoolOp.
    clean_and = """
import mlx.nn as nn
def f(cond, other):
    if cond and ((m := other) and isinstance(m, nn.QuantizedLinear)):
        return m.weight
    return None
"""
    assert scan_source(clean_and).findings == ()

    clean_or = """
from omlx.formats.linear import FormatLinear
def f(cond, other):
    if cond or ((m := other) or isinstance(m, FormatLinear)):
        return None
    return m.weight
"""
    assert scan_source(clean_or).findings == ()

    # Guard-then-walrus inside a nested BoolOp must still be invalidated.
    unsafe_and = """
import mlx.nn as nn
def f(cond, m, other):
    if cond and (isinstance(m, nn.QuantizedLinear) and (m := other)):
        return m.weight
    return None
"""
    assert "unguarded-weight" in _rules(unsafe_and)

    unsafe_or = """
from omlx.formats.linear import FormatLinear
def f(cond, m, other):
    if cond or (isinstance(m, FormatLinear) or (m := other)):
        return None
    return m.weight
"""
    assert "unguarded-weight" in _rules(unsafe_or)


def test_pragmas_allow_and_fail_closed(tmp_path):
    # Same-line pragma suppresses matching rule only; non-matching rule stays reported.
    same_line = """
def f(layer):
    a = layer.weight  # omlx-audit: allow[unguarded-weight] -- verified dense
    b = layer.scales  # omlx-audit: allow[unguarded-weight] -- wrong rule
    return a, b
"""
    r1 = scan_source(same_line)
    assert r1.suppressed == 1
    assert [(f.rule, f.line) for f in r1.findings] == [("unguarded-scales", 4)]

    # Preceding-line and closing-line pragmas on multi-line call.
    multi = """
import mlx.core as mx
def f(layer, x):
    # omlx-audit: allow[unguarded-weight] -- preceding line ok
    w = layer.weight
    y = mx.quantized_matmul(
        x,
        w,
        w,
        w,
    )  # omlx-audit: allow[unguarded-quantized-matmul] -- closing line ok
    return y
"""
    r2 = scan_source(multi, check_unused_pragmas=True)
    assert r2.ok
    assert r2.suppressed == 2
    assert r2.unused_pragmas == ()

    # Valid file pragma in leading header suppresses matching rule across file.
    file_pragma = '''"""Module docstring."""
# omlx-audit: allow-file[unguarded-weight] -- whole file uses dense layers only
def f(layer):
    return layer.weight, layer.scales
'''
    r3 = scan_source(file_pragma)
    assert r3.suppressed == 1
    assert [(f.rule, f.line) for f in r3.findings] == [("unguarded-scales", 4)]

    # Malformed pragma, missing reason, blank reason, unknown rule, late allow-file
    # all emit invalid-pragma and CANNOT be suppressed.
    bad_pragmas = """
x = 1
# omlx-audit: allow-file -- too late after first statement
# omlx-audit: bogus syntax
# omlx-audit: allow[unguarded-weight]
# omlx-audit: allow[unguarded-weight] --
# omlx-audit: allow[not-a-real-rule] -- reason
# omlx-audit: allow[invalid-pragma] -- cannot suppress invalid-pragma
"""
    r4 = scan_source(
        bad_pragmas,
        allowlist=(AllowEntry(path="<string>", rule="*", reason="try suppress all"),),
    )
    assert not r4.ok
    assert len(r4.findings) == 6
    assert all(f.rule == INVALID_PRAGMA_RULE for f in r4.findings)

    # String literal containing pragma text must NOT suppress anything.
    in_string = """
def f(layer):
    s = "# omlx-audit: allow -- fake pragma inside string"
    return s, layer.weight
"""
    r5 = scan_source(in_string)
    assert r5.suppressed == 0
    assert [(f.rule, f.line) for f in r5.findings] == [("unguarded-weight", 4)]

    # Unused pragma flagged when check_unused_pragmas=True -> assert raises AuditError.
    unused_file = tmp_path / "unused.py"
    unused_file.write_text(
        "def f(x):\n"
        "    # omlx-audit: allow[unguarded-weight] -- never triggered\n"
        "    return x\n"
    )
    r6 = scan_paths([unused_file], check_unused_pragmas=True)
    assert r6.ok
    assert len(r6.unused_pragmas) == 1
    assert r6.unused_pragmas[0][1] == 2
    with pytest.raises(AuditError) as exc_info:
        assert_no_unguarded_readers([unused_file], check_unused_pragmas=True)
    assert exc_info.value.report.unused_pragmas == r6.unused_pragmas


def test_allow_entry_validation_and_scoping():
    with pytest.raises(ValueError):
        AllowEntry(path="foo.py", reason="")
    with pytest.raises(ValueError):
        AllowEntry(path="foo.py", reason="   ")
    with pytest.raises(ValueError):
        AllowEntry(path="", reason="ok")
    with pytest.raises(ValueError):
        AllowEntry(path="foo.py", scope="", reason="ok")
    with pytest.raises(ValueError):
        AllowEntry(path="foo.py", rule="bogus-rule", reason="ok")
    with pytest.raises(ValueError):
        AllowEntry(path="foo.py", rule=INVALID_PRAGMA_RULE, reason="ok")

    source = """
def allowed_fn(layer):
    return layer.weight, layer.scales

def other_fn(layer):
    return layer.weight
"""
    entry = AllowEntry(
        path="pkg/mod.py",
        rule="unguarded-weight",
        scope="allowed_fn",
        reason="scoped allow",
    )
    report = scan_source(source, path="a/pkg/mod.py", allowlist=(entry,))
    assert report.suppressed == 1
    assert [(f.scope, f.rule) for f in report.findings] == [
        ("allowed_fn", "unguarded-scales"),
        ("other_fn", "unguarded-weight"),
    ]


def test_fail_closed_on_syntax_error_and_missing_path(tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("def broken(:\n    pass\n")

    report = scan_paths([broken])
    assert not report.ok
    assert len(report.unparsable) == 1
    assert report.unparsable[0][0] == broken.as_posix()

    with pytest.raises(AuditError) as exc_info:
        assert_no_unguarded_readers([broken])
    assert exc_info.value.report.unparsable == report.unparsable

    with pytest.raises(FileNotFoundError):
        scan_paths([tmp_path / "does_not_exist.py"])

    # Symlink loop must terminate cleanly.
    loop_dir = tmp_path / "loop"
    loop_dir.mkdir()
    (loop_dir / "clean.py").write_text("x = 1\n")
    os.symlink(loop_dir, loop_dir / "self_link")
    loop_report = scan_paths([loop_dir])
    assert loop_report.ok
    assert loop_report.files_scanned == ((loop_dir / "clean.py").as_posix(),)


def test_zero_false_positives_on_ported_qwen4_exp():
    allow = (
        AllowEntry(
            path="vendor/mlx_vlm/models/qwen4_exp/hc_prefill_nax.py",
            rule="*",
            reason="DEFERRED 59 / U29: gated off by hc_fused._nax_projection_eligible() -> False",
        ),
    )
    report = assert_no_unguarded_readers([QWEN4_COMPAT_DIR], allowlist=allow)
    assert report.ok
    assert len(report.findings) == 0
    assert len(report.files_scanned) >= 17
    assert report.suppressed > 0


def test_catches_real_unported_misreaders():
    # 1. hc_prefill_nax.py (unallowlisted)
    hc_report = scan_paths([HC_PREFILL_NAX])
    hc_lines = {f.line for f in hc_report.findings}
    assert {661, 702, 728}.issubset(hc_lines)

    # 2. inkling2/dspark.py
    dspark = REPO_ROOT / "omlx/patches/inkling2/dspark.py"
    dspark_report = scan_paths([dspark])
    dspark_lines = {f.line for f in dspark_report.findings}
    assert {205, 206}.issubset(dspark_lines)

    # 3. raon1_mini/raon1_mini_model.py
    raon = REPO_ROOT / "omlx/patches/raon1_mini/raon1_mini_model.py"
    raon_report = scan_paths([raon])
    raon_lines = {f.line for f in raon_report.findings}
    assert {948, 961, 1618}.issubset(raon_lines)

    # 4. deepseek_v4/verify_qmv.py
    verify_qmv = REPO_ROOT / "omlx/patches/deepseek_v4/verify_qmv.py"
    vq_report = scan_paths([verify_qmv])
    vq_lines = {f.line for f in vq_report.findings}
    assert {550, 594}.issubset(vq_lines)


def test_audit_module_has_zero_mlx_imports():
    audit_path = REPO_ROOT / "omlx/formats/audit.py"
    source = audit_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(audit_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(
                    "mlx"
                ), f"forbidden import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "audit.py must not use relative imports"
            mod = node.module or ""
            assert not mod.startswith("mlx"), f"forbidden import from {mod}"
            assert not mod.startswith("omlx"), f"forbidden import from {mod}"
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
        ):
            assert node.attr not in {
                "environ",
                "getenv",
            }, f"Rule 30: forbidden os.{node.attr} in audit.py"


def _run_module_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "omlx.formats.audit", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_cli_clean_file_exits_0_text_and_json(tmp_path, capsys):
    clean = tmp_path / "clean.py"
    clean.write_text(
        "import mlx.core as mx\n"
        "import mlx.nn as nn\n"
        "\n"
        "def f(m, x):\n"
        "    if isinstance(m, nn.QuantizedLinear):\n"
        "        return mx.quantized_matmul(x, m.weight, m.scales, m.biases)\n"
        "    return x\n"
    )

    assert main([str(clean)]) == 0
    text_out = capsys.readouterr().out
    assert "1 file(s) scanned, 0 finding(s)" in text_out
    assert "[unguarded" not in text_out

    assert main([str(clean), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["findings"] == []
    assert payload["files_scanned"] == [clean.as_posix()]
    assert payload["suppressed"] == 0
    assert payload["unparsable"] == []
    assert payload["unused_pragmas"] == []

    with pytest.raises(ValueError):
        format_report(scan_paths([clean]), fmt="yaml")

    proc = _run_module_cli([str(clean)])
    assert proc.returncode == 0, proc.stderr
    assert "0 finding(s)" in proc.stdout

    proc_json = _run_module_cli([str(clean), "--format", "json"])
    assert proc_json.returncode == 0, proc_json.stderr
    assert json.loads(proc_json.stdout)["ok"] is True


def test_cli_finding_exits_1_with_structured_json(tmp_path, capsys):
    bad = tmp_path / "bad.py"
    bad.write_text("def f(layer):\n    return layer.weight\n")

    assert main([str(bad), "--format", "json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert len(payload["findings"]) == 1
    finding = payload["findings"][0]
    assert finding["rule"] == "unguarded-weight"
    assert finding["line"] == 2
    assert finding["col"] == 11
    assert finding["scope"] == "f"
    assert finding["path"] == bad.as_posix()
    assert "weight" in finding["message"]

    assert main([str(bad)]) == 1
    text_out = capsys.readouterr().out
    assert f"{bad.as_posix()}:2:11: [unguarded-weight] (f)" in text_out
    assert "1 finding(s)" in text_out

    proc = _run_module_cli([str(bad), "--format", "json"])
    assert proc.returncode == 1, proc.stderr
    assert json.loads(proc.stdout)["findings"][0]["rule"] == "unguarded-weight"


def test_cli_allow_and_exclude_flags_suppress_cleanly(capsys):
    allow_spec = "vendor/mlx_vlm/models/qwen4_exp/hc_prefill_nax.py::DEFERRED 59 / U29"

    assert main([str(QWEN4_COMPAT_DIR), "--allow", allow_spec]) == 0
    text_out = capsys.readouterr().out
    assert "0 finding(s)" in text_out

    assert main([str(QWEN4_COMPAT_DIR)]) == 1
    assert "unguarded-" in capsys.readouterr().out

    assert main([str(QWEN4_COMPAT_DIR), "--exclude", "hc_prefill_nax.py"]) == 0
    assert "0 finding(s)" in capsys.readouterr().out

    proc = _run_module_cli([str(QWEN4_COMPAT_DIR), "--allow", allow_spec])
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_cli_missing_path_and_bad_allow_exit_2(tmp_path, capsys):
    assert main([str(tmp_path / "does_not_exist.py")]) == 2
    assert "does not exist" in capsys.readouterr().err

    (tmp_path / "clean.py").write_text("x = 1\n")

    assert main([str(tmp_path), "--allow", "foo.py"]) == 2
    assert "--allow" in capsys.readouterr().err

    assert main([str(tmp_path), "--allow", "foo.py::"]) == 2
    assert "--allow" in capsys.readouterr().err

    assert main([str(tmp_path), "--allow", "foo.py=bogus-rule::reason"]) == 2
    assert "--allow" in capsys.readouterr().err

    for bad_spec in (
        "clean.py=::reason",
        "clean.py:=unguarded-weight::reason",
        "clean.py: ::reason",
    ):
        assert main([str(tmp_path), "--allow", bad_spec]) == 2
        assert "--allow" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc_info:
        main([str(tmp_path), "--format", "yaml"])
    assert exc_info.value.code == 2


def test_cli_syntax_error_and_unused_pragma_exit_1(tmp_path, capsys):
    broken = tmp_path / "broken.py"
    broken.write_text("def broken(:\n    pass\n")
    assert main([str(broken)]) == 1
    assert "unparsable" in capsys.readouterr().out

    unused = tmp_path / "unused.py"
    unused.write_text(
        "def f(x):\n"
        "    # omlx-audit: allow[unguarded-weight] -- never fires\n"
        "    return x\n"
    )
    assert main([str(unused)]) == 0
    capsys.readouterr()
    assert main([str(unused), "--check-unused-pragmas"]) == 1
    out = capsys.readouterr().out
    assert "unused pragma" in out
    assert "1 unused pragma(s)" in out


def test_cli_standalone_script_invocation(tmp_path):
    script = REPO_ROOT / "omlx/formats/audit.py"
    clean = tmp_path / "clean.py"
    clean.write_text("x = 1\n")
    proc = subprocess.run(
        [sys.executable, str(script), str(clean)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "1 file(s) scanned, 0 finding(s)" in proc.stdout

    bad = tmp_path / "bad.py"
    bad.write_text("def f(layer):\n    return layer.weight\n")
    proc_bad = subprocess.run(
        [sys.executable, str(script), str(bad)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc_bad.returncode == 1, proc_bad.stderr
    assert f"{bad.as_posix()}:2:11: [unguarded-weight] (f)" in proc_bad.stdout


def test_plugin_overview_docs_cover_audit_cli_and_pragmas():
    page = (REPO_ROOT / "docs/quant-format-plugins.md").read_text(encoding="utf-8")
    assert "## Developer tooling: audit CLI" in page
    assert "audit and debug CLIs" not in page
    for token in (
        "python -m omlx.formats.audit",
        "assert_no_unguarded_readers",
        "omlx-audit: allow",
        "--check-unused-pragmas",
        "AuditReport",
        "AllowEntry",
    ):
        assert token in page, f"missing {token!r} in docs/quant-format-plugins.md"
