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
        (LANGUAGE, "_fused_projections_ready", "QuantizedLinear"),
        (LANGUAGE, "_can_fuse_hyper_connection", "QuantizedLinear"),
        (Q4 / "hc_fused.py", "_quantized_ok", "QuantizedLinear"),
        (LANGUAGE, "_uniform_affine_layout", "QuantizedEmbedding"),
        (ADAPTER, "_ple_reader", "QuantizedEmbedding"),
    ],
)
def test_qwen_private_route_releases_its_mlx_class_branch(path, function, symbol):
    nodes = _function_nodes(path, function)
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == symbol for node in nodes
    )


@pytest.mark.parametrize("operation", ["construct", "exact_type"])
def test_ple_join_releases_the_mlx_constructor_and_source_test(operation):
    nodes = _function_nodes(LANGUAGE, "fuse_quantized_shards")
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
        ("fuse_hyper_connection_projections", "_can_prepare_exact_hybrid"),
    ],
)
def test_hyper_connection_releases_the_hybrid_call_and_its_preparation(
    function, legacy
):
    nodes = _function_nodes(LANGUAGE, function)
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
