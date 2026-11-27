# SPDX-License-Identifier: Apache-2.0
"""Tests for the AST format-reader audit scanner and pytest CI helper."""

from __future__ import annotations

import ast
import os
import pathlib

import pytest

from omlx.formats.audit import (
    INVALID_PRAGMA_RULE,
    RULES,
    AllowEntry,
    AuditError,
    assert_no_unguarded_readers,
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
