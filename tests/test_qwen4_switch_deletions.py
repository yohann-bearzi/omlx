# SPDX-License-Identifier: Apache-2.0
"""The Qwen4 switch removes its private MLX routes and their last consumers."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
Q4 = ROOT / "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp"
LANGUAGE = Q4 / "language.py"
ADAPTER = ROOT / "omlx/prefill_helper/adapters/qwen4_exp.py"


def _function_nodes(path, function):
    if not path.exists():
        return []
    tree = ast.parse(path.read_text())
    return [
        node
        for definition in ast.walk(tree)
        if isinstance(definition, ast.FunctionDef) and definition.name == function
        for node in ast.walk(definition)
    ]


@pytest.mark.parametrize(
    "path,function,symbol",
    [
        (LANGUAGE, "_fused_plan", "QuantizedLinear"),
        (Q4 / "hc_fused.py", "_projection_face", "QuantizedLinear"),
        (Q4 / "hc_fused.py", "_nax_projection_eligible", "QuantizedLinear"),
        (LANGUAGE, "_uniform_affine_layout", "QuantizedEmbedding"),
        (ADAPTER, "_ple_reader", "QuantizedEmbedding"),
    ],
)
def test_qwen_private_route_releases_its_mlx_class_branch(path, function, symbol):
    nodes = _function_nodes(path, function)
    assert nodes, f"{function} not found"
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == symbol for node in nodes
    )


def test_hc_fused_keeps_no_mlx_quantized_linear_or_inject_hasattr():
    """U21 ported hc_fused to FormatLinear faces and DEL3-3's hook returns
    False, so no QuantizedLinear name/attribute and no dead
    input_inject_weight hasattr remains anywhere in the file (the plan
    reads the member, not a kind)."""
    tree = ast.parse((Q4 / "hc_fused.py").read_text())
    assert not any(
        (isinstance(node, ast.Attribute) and node.attr == "QuantizedLinear")
        or (isinstance(node, ast.Name) and node.id == "QuantizedLinear")
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "hasattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "input_inject_weight"
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize("operation", ["construct", "exact_type"])
def test_ple_join_releases_the_mlx_constructor_and_source_test(operation):
    nodes = _function_nodes(LANGUAGE, "fuse_quantized_shards")
    assert nodes, "fuse_quantized_shards not found"
    for node in nodes:
        if operation == "construct" and isinstance(node, ast.Call):
            assert not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "QuantizedEmbedding"
            )
        elif operation == "exact_type" and isinstance(node, ast.Compare):
            assert not any(
                isinstance(n, ast.Attribute) and n.attr == "QuantizedEmbedding"
                for n in ast.walk(node)
            )


@pytest.mark.parametrize(
    "function,legacy",
    [
        ("_forward", "hybrid_projection"),
        ("compile_hyper_connections", "_can_prepare_exact_hybrid"),
    ],
)
def test_hyper_connection_releases_the_hybrid_call_and_its_preparation(
    function, legacy
):
    nodes = _function_nodes(LANGUAGE, function)
    assert nodes, f"{function} not found"
    assert not any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == legacy
        for n in nodes
    )


def test_hybrid_projection_module_and_its_private_header_are_removed():
    assert not (Q4 / "hc_projection.py").exists()


def test_host_ple_uses_its_reader_for_quantized_rows():
    tree = ast.parse(ADAPTER.read_text())
    host = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_HostPLERows"
    )
    assert not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "dequantize"
        for n in ast.walk(host)
    )


def test_hyper_connection_releases_the_split_indices_and_inject_lookup():
    text = LANGUAGE.read_text()
    assert "_HYPER_SPLIT_INDICES" not in text
    assert "input_inject_weight" not in text
    tree = ast.parse(text)
    assert not any(
        isinstance(node, ast.Name) and node.id == "_HYPER_SPLIT_INDICES"
        for node in ast.walk(tree)
    )
    nodes = _function_nodes(LANGUAGE, "_forward")
    assert nodes
    assert not any(
        (isinstance(node, ast.Constant) and node.value == "input_inject_weight")
        or (isinstance(node, ast.Attribute) and node.attr == "input_inject_weight")
        or (isinstance(node, ast.Name) and node.id == "input_inject_weight")
        for node in nodes
    )


def test_the_qwen4_adapter_wires_the_qwen35_q4_patches_in_load_first():
    calls = {
        node.func.id
        for node in _function_nodes(ADAPTER, "load_first")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert calls
    assert "apply_qwen35_q4_mlp_patch" in calls
    assert "apply_qwen35_q4_prefill_linear_patch" in calls


GDN = ROOT / "omlx/patches/qwen35_gdn_prework.py"
ROW_EXACT = ROOT / "omlx/patches/row_exact_qmv.py"


def test_row_exact_qmv_releases_the_one_row_launch():
    text = ROW_EXACT.read_text()
    assert "OneRowQmv" not in text
    assert "one_row_qmv" not in text
    tree = ast.parse(text)
    assert not any(
        isinstance(node, ast.Name) and node.id in ("OneRowQmv", "one_row_qmv")
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ("OneRowQmv", "one_row_qmv")
        for node in ast.walk(tree)
    )
    exported = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    values = {
        node.value
        for node in ast.walk(exported.value)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "OneRowQmv" not in values and "one_row_qmv" not in values


def test_gdn_prework_releases_the_legacy_quantized_paths():
    tree = ast.parse(GDN.read_text())
    imported = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.endswith("row_exact_qmv")
        ):
            imported.update(alias.name for alias in node.names)
    assert "one_row_qmv" not in imported and "rows_qmv" not in imported
    functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert "_one_row_fused_projections" not in functions
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "_ALLOWED_BITS" not in names and "_ALLOWED_GROUPS" not in names
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in ("quantized_matmul", "QuantizedLinear")
        for node in ast.walk(tree)
    )


def test_batch_decode_docstring_drops_stock_matmul():
    tree = ast.parse(GDN.read_text())
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_qwen4_batch_decode"
    )
    doc = ast.get_docstring(fn)
    assert doc
    assert "quantized matmul" not in doc and "stock layer" not in doc
    assert "format layer's own forward" in doc
