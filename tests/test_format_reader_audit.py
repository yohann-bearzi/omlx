"""D10: every reader of MLX's quantized layer classes is tabled.

Every site in ``omlx/``, the vendored ``qwen4_exp`` model, and the installed
``mlx_vlm`` verifier/linear helpers that detects or references a quantized
layer -- class references (including import and assignment aliases, class
bases, and exact class-name strings, excluding docstrings), ``isinstance``
(including tuples, PEP 604 unions, and aliases), ``type(x) is`` checks,
``getattr``/``hasattr`` on ``.bits``/``.group_size``/``.scales``, direct
attribute reads of those names on layer-tied receivers, and
``quantize``/``to_quantized``/``class_predicate`` sites -- must appear below
with a decision and the test that proves it. A new, removed or changed reader
fails the test, as do unparsable/missing audited files and unused table rows.
Decisions that call a path "not served" are backed by a second test: a fresh
process importing ``omlx.engine.vlm`` and building a synthetic stock
``qwen4_exp`` model must not import any of those modules.
"""

from __future__ import annotations

import ast
import collections
import importlib.util
import pathlib
import subprocess
import sys

NAMES = {"QuantizedLinear", "QuantizedSwitchLinear", "QuantizedEmbedding"}
ATTRS = {"bits", "group_size", "scales"}
LAYERISH = {
    "linear",
    "layer",
    "module",
    "proj",
    "projection",
    "shard",
    "gate",
    "up",
    "down",
    "w",
    "weight",
    "scales",
    "embed",
    "table",
    "router",
    "qkv",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "index_qk_proj",
    "fc",
    "fc1",
    "fc2",
    "shards",
    "layers",
    "m",
    "x",
    "head",
}
BASE = pathlib.Path(__file__).parents[1]
ROOTS = (
    BASE / "omlx",
    BASE / "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp",
)


def _installed_reader_files():
    """The installed mlx_vlm readers the compat layer reaches. They decide a
    layer's type (isinstance nn.QuantizedLinear) for the row-invariant,
    batch-invariant and exact projections; an unparsed or missing file is an
    unaudited reader, so the audit fails closed."""
    spec = importlib.util.find_spec("mlx_vlm")
    if spec is None or spec.origin is None:
        return ()
    root = pathlib.Path(spec.origin).parent
    return (
        (root, root / "models/quantized_verifier.py"),
        (root, root / "models/linear.py"),
    )


def _tail(node):
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _class_names(node, aliases):
    """Quantized class names referenced by an expression (Name, Attribute,
    Tuple/List/Set, PEP 604 BitOr union, or import/assignment alias)."""
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        names = set()
        for element in node.elts:
            names |= _class_names(element, aliases)
        return names
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _class_names(node.left, aliases) | _class_names(node.right, aliases)
    if isinstance(node, ast.Attribute) and node.attr in NAMES:
        return {node.attr}
    if isinstance(node, ast.Name):
        if node.id in NAMES:
            return {node.id}
        if node.id in aliases:
            return set(aliases[node.id])
    return set()


def _layer_receiver(node):
    """A receiver we can tie to a layer by name (the module or ``self.x``)."""
    if isinstance(node, ast.Attribute):
        return node.attr in LAYERISH
    return isinstance(node, ast.Name) and node.id in LAYERISH | {"self"}


def _is_docstring(stmt):
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _collect_aliases(tree):
    """Collect import aliases and pure class/tuple/union assignment aliases of
    ``NAMES`` across ``tree``."""
    aliases: dict[str, set[str]] = collections.defaultdict(set)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                tail = alias.name.rsplit(".", 1)[-1]
                if alias.asname and tail in NAMES:
                    aliases[alias.asname].add(tail)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            targets = ()
            value = None
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = (node.target,)
                value = node.value
            if value is None:
                continue
            resolved = _class_names(value, aliases)
            if not resolved:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    before = len(aliases[target.id])
                    aliases[target.id] |= resolved
                    if len(aliases[target.id]) > before:
                        changed = True
    return aliases


def _scan_source(relative, source):
    tree = ast.parse(source)
    aliases = _collect_aliases(tree)
    found: dict[tuple[str, str], set[str]] = collections.defaultdict(set)

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scope: list[str] = []

        def add(self, signal):
            key = (relative, ".".join(self.scope) or "<module>")
            found[key].add(signal)

        def _visit_body(self, body):
            start = 1 if body and _is_docstring(body[0]) else 0
            for stmt in body[start:]:
                self.visit(stmt)

        def visit_Module(self, node):
            self._visit_body(node.body)

        def visit_ClassDef(self, node):
            for dec in node.decorator_list:
                self.visit(dec)
            self.scope.append(node.name)
            for base in node.bases:
                self.visit(base)
            for kw in node.keywords:
                self.visit(kw)
            self._visit_body(node.body)
            self.scope.pop()

        def visit_FunctionDef(self, node):
            for dec in node.decorator_list:
                self.visit(dec)
            self.scope.append(node.name)
            all_args = (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            )
            if node.name == "class_predicate" or any(
                arg.arg == "class_predicate" for arg in all_args
            ):
                self.add("class_predicate")
            self.visit(node.args)
            if node.returns is not None:
                self.visit(node.returns)
            self._visit_body(node.body)
            self.scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                if node.id in NAMES:
                    self.add(node.id)
                elif node.id in aliases:
                    for target in aliases[node.id]:
                        self.add(target)

        def visit_Attribute(self, node):
            if node.attr in NAMES:
                self.add(node.attr)
            if (
                isinstance(node.ctx, ast.Load)
                and node.attr in ATTRS
                and _layer_receiver(node.value)
            ):
                self.add(f"read {node.attr}")
            self.generic_visit(node)

        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in NAMES:
                self.add(node.value)

        def visit_Call(self, node):
            name = _tail(node.func)
            if name == "isinstance" and len(node.args) == 2:
                for cls in _class_names(node.args[1], aliases):
                    self.add(f"isinstance {cls}")
            elif name == "getattr" and len(node.args) >= 2:
                second = node.args[1]
                if isinstance(second, ast.Constant) and second.value in ATTRS:
                    self.add(f"getattr {second.value}")
            elif name == "hasattr" and len(node.args) >= 2:
                second = node.args[1]
                if isinstance(second, ast.Constant) and second.value in ATTRS:
                    self.add(f"hasattr {second.value}")
            elif name in {"quantize", "to_quantized"}:
                self.add(f"quantize-call {name}")
            elif name == "class_predicate":
                self.add("class_predicate")
            for kw in node.keywords:
                if kw.arg == "class_predicate":
                    self.add("class_predicate")
            self.generic_visit(node)

        def visit_Compare(self, node):
            if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Is, ast.IsNot)):
                for side, other in (
                    (node.left, node.comparators[0]),
                    (node.comparators[0], node.left),
                ):
                    if isinstance(other, ast.Call) and _tail(other.func) == "type":
                        for cls in _class_names(side, aliases):
                            self.add(f"type-is {cls}")
            self.generic_visit(node)

    Visitor().visit(tree)
    return {key: tuple(sorted(signals)) for key, signals in found.items()}


def _readers():
    found = {}
    unparsable = []
    extra = _installed_reader_files()
    if not extra:
        unparsable.append("installed/mlx_vlm (missing)")
    else:
        unparsable.extend(str(path) for _, path in extra if not path.exists())
    paths = [(root, path) for root in ROOTS for path in sorted(root.rglob("*.py"))]
    paths.extend(extra)
    for root, path in paths:
        if root in ROOTS and "vendor" in path.parts and root is not ROOTS[1]:
            continue
        if root in ROOTS:
            relative = path.relative_to(BASE).as_posix()
        else:
            relative = "installed/" + path.relative_to(root).as_posix()
        try:
            source = path.read_text()
            found.update(_scan_source(relative, source))
        except (OSError, SyntaxError):
            unparsable.append(relative)
            continue
    return found, sorted(set(unparsable))


FACE = "takes the layer's own format face (falls back to its call)"
OWN = "reads the format layer's own tensors (the format implementation)"
KV = "quantized KV-cache state, not a layer reader"
WRITER = "writer/quantizer side, not the served read path"
UNSERVED = "not a served reader for a stock qwen4_exp"
GATED = (
    "imported by engine/vlm.py at module level, gated by model_type or opt-in "
    "(never invoked for a stock qwen4_exp)"
)
PENDING = "pending - hc_prefill_nax DEFERRED 59 (owner's ruling)"
REFUSE = "declines loudly at load"
A8FACE = "takes the built-in affine FormatLinear face"

PATH_DECISIONS = {
    "omlx/models/vlm.py": (
        "declines (rebinds exact OneBitLinear/OneBitEmbedding only; no-op on FormatLinear/FormatEmbedding)",
        "omlx/models/vlm.py:66-76",
    ),
    "omlx/patches/deepseek_v4/decode_consistency.py": (UNSERVED, "family not ported"),
    "omlx/patches/deepseek_v4/moe_offload.py": (UNSERVED, "family not ported"),
    "omlx/patches/glm_moe_dsa/moe_offload.py": (UNSERVED, "family not ported"),
    "omlx/patches/mlx_vlm_mtp/qwen35_verify_linear.py": (
        FACE,
        "tests/test_mlx_vlm_qwen4_exp_compat.py (_QUANTIZED_LINEAR_TYPES includes FormatLinear)",
    ),
    "omlx/affine4.py": (OWN, "the KV codec's own format state"),
    "omlx/admin/ane_tuning.py": (
        UNSERVED,
        "admin ANE calibration; not the serving path",
    ),
    "omlx/cluster/tensor_strategies.py": (
        OWN,
        "shard loading of a format layer's own tensors",
    ),
    "omlx/formats/__init__.py": (OWN, "the format face builder itself"),
    "omlx/formats/affine.py": (OWN, "the affine format's own implementation"),
    "omlx/formats/oqh.py": (OWN, "the built-in oQh formats' own implementation"),
    "omlx/formats/kernels.py": (OWN, "the format kernel builders"),
    "omlx/formats/tile_dense.py": (
        OWN,
        "tests/test_formats_tile_dense.py and tests/test_formats_tile_dense_parity.py: the dense skeleton reads the layer's own tensors and the tuning entry",
    ),
    "omlx/formats/linear.py": (OWN, "the format layer's own implementation"),
    "omlx/lora_adapters.py": (
        GATED,
        "LoRA adapter opt-in; not invoked for a stock qwen4_exp load without adapters",
    ),
    "omlx/oq.py": (WRITER, "quantizer-side measurement, not the served read path"),
    "omlx/oqh/modules.py": (WRITER, "oQ writer modules"),
    "omlx/oqh/runtime.py": (WRITER, "oQ writer runtime"),
    "omlx/oqh/writer.py": (WRITER, "oQ writer"),
    "omlx/prefill_helper/adapters/qwen4_exp.py": (
        OWN,
        "the PLE reader reads the format layer's own shards",
    ),
    "omlx/scheduler.py": (
        "reads embed_tokens.scales (returns None dtype when absent on plugin FormatEmbedding)",
        "omlx/scheduler.py:1626-1635",
    ),
    "omlx/speculative/dflash/__init__.py": (
        GATED,
        "DFlash drafter opt-in; not invoked for stock qwen4_exp",
    ),
    "omlx/speculative/dflash/drafter.py": (
        GATED,
        "DFlash drafter opt-in; not invoked for stock qwen4_exp",
    ),
    "omlx/speculative/dflash/selection.py": (
        GATED,
        "DFlash drafter opt-in; not invoked for stock qwen4_exp",
    ),
    "omlx/turboquant_kv.py": (KV, "quantized KV cache"),
    "omlx/patches/batched_decode_qmm.py": (
        OWN,
        "reads MLX's own quantized layers in the batched qmm route",
    ),
    "omlx/patches/m5_gather_qmm.py": (
        OWN,
        "reads MLX's own quantized layers in the gather qmm route",
    ),
    "omlx/patches/m5_gather_qmm_a8.py": (
        OWN,
        "reads MLX's own quantized layers in the A8 gather route",
    ),
    "omlx/patches/m5_gather_qmm_nax.py": (
        OWN,
        "reads MLX's own quantized layers in the NAX gather route",
    ),
    "omlx/patches/moe_gate_up_fusion.py": (
        "declines (a FormatLinear fails its QuantizedSwitchLinear class check; the format layer keeps its own call)",
        "tests/test_qwen35_moe_gate_up.py",
    ),
    "omlx/patches/mlx_lm_mtp/batch_generator.py": (
        GATED,
        "mlx-lm MTP batch generator; not invoked for VLM stock qwen4_exp",
    ),
    "omlx/patches/qwen38_modelopt_mixed.py": (
        WRITER,
        "ModelOpt checkpoint conversion, writer side",
    ),
    "omlx/patches/qwen35_ane_prefill.py": (
        "declines (opt-in ANE route; only MLX quantized layers are eligible, so a format checkpoint keeps the GPU route)",
        "tests/test_qwen35_ane_prefill.py and the qwen35_ane_prefill_enabled setting gate",
    ),
    "omlx/patches/qwen35_fa256_attention.py": (
        KV,
        "quantized KV state in the FA-256 attention route",
    ),
    "omlx/patches/qwen35_gdn_prework.py": (
        FACE,
        "tests/test_qwen35_gdn_prework.py (stock affine plan vs serial bitwise)",
    ),
    "omlx/models/decision.py": (
        FACE,
        "tests/test_formats_stock_dense.py::test_decision_backbone_format_linear_head",
    ),
    "omlx/patches/moe_expert_offload.py": (
        "declines (a FormatLinear is not MLX's QuantizedSwitchLinear)",
        "tests/test_qwen35_moe_gate_up.py",
    ),
    "omlx/patches/qwen35_moe_gate_up.py": (
        "declines (expert stacks stay MLX's QuantizedSwitchLinear; the stock switch never builds FormatSwitchLinear)",
        "tests/test_qwen35_moe_gate_up.py",
    ),
    "omlx/patches/qwen35_moe_routed_decode.py": (
        A8FACE,
        "tests/test_qwen35_moe_verify_window.py::test_stock_affine_format_router_takes_the_fused_window",
    ),
    "omlx/patches/qwen35_oq_a8.py": (
        A8FACE,
        "tests/test_qwen35_oq_a8.py::test_stock_affine_format_layer_classifies_like_mlxs_layer",
    ),
    "omlx/patches/qwen35_packed_linear.py": (
        "declines (PackedLinear-only probe)",
        "tests/test_formats_stock_dense.py",
    ),
    "omlx/patches/qwen35_padded_row_attention.py": (
        KV,
        "quantized KV state in the padded-row attention route",
    ),
    "omlx/patches/qwen35_q4_mlp.py": (
        FACE,
        "tests/test_qwen35_q4_mlp.py::test_vlm_format_prefill_linear_routes_affine_language_layers",
    ),
    "omlx/patches/qwen35_verify_qmm.py": (
        FACE,
        "tests/test_qwen35_moe_verify_window.py (window rows equal one-token decode)",
    ),
    "omlx/patches/qwen35_verify_sdpa_split.py": (
        KV,
        "quantized KV state in the split SDPA verify route",
    ),
    "omlx/patches/row_exact_qmv.py": (
        OWN,
        "the plan builder reads the layer's own face",
    ),
    "omlx/patches/sdpa256_attention.py": (
        KV,
        "quantized KV state in the SDPA-256 route",
    ),
    "omlx/patches/turboquant_attention.py": (KV, "quantized KV state"),
    "omlx/patches/deepseek_v4/switch_layers.py": (GATED, "DeepSeek family not ported"),
    "omlx/patches/deepseek_v4/utils_patch.py": (UNSERVED, "family not ported"),
    "omlx/patches/deepseek_v4/verify_qmv.py": (UNSERVED, "family not ported"),
    "omlx/patches/deepseek_v41/a8.py": (GATED, "DeepSeek V4.1 family not ported"),
    "omlx/patches/deepseek_v41/decode_fused.py": (
        GATED,
        "DeepSeek V4.1 family not ported",
    ),
    "omlx/patches/deepseek_v41/language.py": (GATED, "DeepSeek V4.1 family not ported"),
    "omlx/patches/deepseek_v41/oq.py": (UNSERVED, "family not ported"),
    "omlx/patches/deepseek_v41/oqh.py": (UNSERVED, "family not ported"),
    "omlx/patches/deepseek_v41/quantization.py": (
        GATED,
        "DeepSeek V4.1 family not ported",
    ),
    "omlx/patches/deepseek_v41/sensitivity.py": (UNSERVED, "family not ported"),
    "omlx/patches/glm_moe_dsa/deepseek_v32.py": (UNSERVED, "family not ported"),
    "omlx/patches/glm_moe_dsa/sparse_mla.py": (UNSERVED, "family not ported"),
    "omlx/patches/glm_moe_dsa/switch_layers.py": (UNSERVED, "family not ported"),
    "omlx/patches/inkling2/dspark.py": (UNSERVED, "family not ported"),
    "omlx/patches/inkling2/model.py": (GATED, "Inkling2 family not ported"),
    "omlx/patches/k2_horizon/ane_prefill.py": (UNSERVED, "family not ported"),
    "omlx/patches/laguna/laguna_model.py": (UNSERVED, "family not ported"),
    "omlx/patches/mimo_v2/decode_fast.py": (UNSERVED, "family not ported"),
    "omlx/patches/mimo_v2/mimo_v2_model.py": (UNSERVED, "family not ported"),
    "omlx/patches/mimo_v2/moe_decode.py": (UNSERVED, "family not ported"),
    "omlx/patches/minimax_m3_dspark.py": (UNSERVED, "family not ported"),
    "omlx/patches/muse_spark/model.py": (GATED, "Muse Spark family not ported"),
    "omlx/patches/bailing_hybrid/bailing_hybrid_model.py": (
        UNSERVED,
        "family not ported",
    ),
    "omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py": (
        UNSERVED,
        "family not ported",
    ),
    "omlx/patches/mlx_vlm_minimax_m3_compat/__init__.py": (
        GATED,
        "MiniMax M3 family not ported",
    ),
    "omlx/patches/mlx_vlm_minimax_m3_compat/a8.py": (
        GATED,
        "MiniMax M3 family not ported",
    ),
    "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/cache.py": (
        KV,
        "the model's own quantized KV caches",
    ),
    "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/hc_fused.py": (
        FACE,
        "U21 ported hc_fused to FormatLinear faces and set _nax_projection_eligible=False; DEL3-3 pins them (hc_prefill_nax stays DEFERRED 59)",
    ),
    "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/hc_prefill_nax.py": (
        PENDING,
        "DEFERRED 59 / U29 (tensor-unit prefill not yet ported to faces)",
    ),
    "installed/models/quantized_verifier.py": (
        "declines a FormatLinear (upstream helper; returns None); the row-invariant design adoption is DEFERRED and the narrow-N >16 route choice is PFX/tilebf under C2",
        "the audit pins every installed reader the compat layer reaches; tests/test_formats_stock_dense.py pins the format routes",
    ),
    "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py": (
        OWN,
        "the vendored model reads the format layer's own tensors; `_affine_rows` documents the stock switch's layer",
    ),
    "omlx/patches/raon1_mini/raon1_mini_model.py": (UNSERVED, "family not ported"),
}

SCOPE_DECISIONS = {
    ("omlx/patches/qwen35_oq_a8.py", "_is_q8_linear"): (
        A8FACE,
        "tests/test_qwen35_oq_a8.py::test_q8_stock_affine_format_mtp_head_not_tagged",
    ),
    ("omlx/patches/qwen35_oq_a8.py", "_refuse_unclassified"): (
        REFUSE,
        "tests/test_qwen35_oq_a8.py::test_refuses_a_dense_projection_the_a8_kernels_cannot_take",
    ),
    ("omlx/patches/qwen35_oq_a8.py", "_classify_uncached"): (
        A8FACE,
        "tests/test_qwen35_oq_a8.py::test_stock_affine_format_layer_classifies_like_mlxs_layer",
    ),
    ("omlx/patches/qwen35_oq_a8.py", "OqA8Plan.kernel"): (
        A8FACE,
        "tests/test_qwen35_oq_a8.py::test_stock_affine_format_layer_classifies_like_mlxs_layer",
    ),
    ("omlx/patches/qwen35_oq_a8.py", "OqA8Plan.layout"): (
        A8FACE,
        "tests/test_qwen35_oq_a8.py::test_stock_affine_format_layer_classifies_like_mlxs_layer",
    ),
    ("omlx/patches/qwen35_oq_a8.py", "_prepared_weights"): (
        A8FACE,
        "tests/test_qwen35_oq_a8.py::test_a8_plan_on_the_format_layer_is_bitwise_the_mlx_layer_plan",
    ),
    ("omlx/patches/qwen35_moe_routed_decode.py", "_window_fast"): (
        A8FACE,
        "tests/test_qwen35_moe_verify_window.py::test_stock_affine_format_router_takes_the_fused_window",
    ),
    ("omlx/patches/qwen35_moe_routed_decode.py", "_window_coalesce_modes"): (
        A8FACE,
        "tests/test_qwen35_moe_verify_window.py::test_routed_window_coalescing_uses_per_chip_data_not_gpu_generation",
    ),
    ("omlx/patches/qwen35_moe_routed_decode.py", "_window_down_policies"): (
        A8FACE,
        "tests/test_qwen35_moe_verify_window.py::test_routed_down_split_and_fused_rps8_bitwise_and_dispatch",
    ),
    ("omlx/patches/qwen35_moe_routed_decode.py", "_window_router_ok"): (
        A8FACE,
        "tests/test_qwen35_moe_verify_window.py::test_stock_affine_format_router_takes_the_fused_window",
    ),
    ("omlx/patches/qwen35_moe_routed_decode.py", "warm_moe_model"): (
        OWN,
        "tests/test_release3_warm.py::test_zero_new_proofs_on_first_served_request_after_load",
    ),
    ("omlx/patches/qwen35_gdn_prework.py", "_format_projection"): (
        FACE,
        "tests/test_qwen35_gdn_prework.py (stock affine plan vs serial bitwise)",
    ),
    ("omlx/patches/qwen35_gdn_prework.py", "_stack_format_inputs"): (
        FACE,
        "tests/test_qwen35_gdn_prework.py (the format inputs stack through the face)",
    ),
    ("omlx/patches/qwen35_verify_qmm.py", "sg8_swiglu_eligible"): (
        FACE,
        "tests/test_qwen35_moe_verify_window.py (window rows equal one-token decode)",
    ),
    ("omlx/patches/qwen35_verify_qmm.py", "sg8_group_eligible"): (
        FACE,
        "tests/test_qwen35_moe_verify_window.py (window rows equal one-token decode)",
    ),
    ("omlx/patches/row_exact_qmv.py", "<module>"): (
        OWN,
        "tests/test_release3_warm.py::test_warm_row_exact_qmv_zero_compiles_on_first_request",
    ),
    ("omlx/patches/row_exact_qmv.py", "_linear_in_out"): (
        OWN,
        "tests/test_release3_warm.py::test_warm_row_exact_qmv_zero_compiles_on_first_request",
    ),
    ("omlx/patches/row_exact_qmv.py", "_linear_sig"): (
        OWN,
        "tests/test_release3_warm.py::test_warm_row_exact_qmv_zero_compiles_on_first_request",
    ),
    ("omlx/patches/row_exact_qmv.py", "warm_model"): (
        OWN,
        "tests/test_release3_warm.py::test_warm_row_exact_qmv_zero_compiles_on_first_request",
    ),
    ("omlx/patches/row_exact_qmv.py", "warm_model_dispatch"): (
        OWN,
        "tests/test_release3_warm.py::test_warm_row_exact_qmv_zero_compiles_on_first_request",
    ),
}


EXPECTED = {
    ("installed/models/quantized_verifier.py", "_can_optimized_affine_linear"): (
        "read " "bits",
        "read " "scales",
    ),
    ("installed/models/quantized_verifier.py", "_exact_time_batch"): (
        "QuantizedLinear",
    ),
    ("installed/models/quantized_verifier.py", "_fused_quantized_moe_hc_expand"): (
        "QuantizedSwitchLinear",
        "isinstance " "QuantizedSwitchLinear",
        "read " "bits",
        "read " "group_size",
        "read " "scales",
    ),
    ("installed/models/quantized_verifier.py", "_input_dims"): (
        "read group_size",
        "read scales",
    ),
    ("installed/models/quantized_verifier.py", "_stable_nvfp4_argmax"): (
        "QuantizedLinear",
        "isinstance " "QuantizedLinear",
    ),
    ("installed/models/quantized_verifier.py", "_stable_nvfp4_linear"): (
        "QuantizedLinear",
        "isinstance " "QuantizedLinear",
    ),
    ("installed/models/quantized_verifier.py", "exact_quantized_linear"): (
        "QuantizedLinear",
        "isinstance " "QuantizedLinear",
    ),
    ("installed/models/quantized_verifier.py", "exact_quantized_selected_linear"): (
        "QuantizedSwitchLinear",
        "isinstance " "QuantizedSwitchLinear",
    ),
    ("installed/models/quantized_verifier.py", "exact_quantized_switch_linear"): (
        "QuantizedSwitchLinear",
        "isinstance " "QuantizedSwitchLinear",
    ),
    ("installed/models/quantized_verifier.py", "optimized_affine_argmax"): (
        "read bits",
        "read " "group_size",
        "read scales",
    ),
    ("installed/models/quantized_verifier.py", "optimized_affine_linear"): (
        "read bits",
        "read " "group_size",
        "read scales",
    ),
    ("installed/models/quantized_verifier.py", "optimized_affine_linears"): (
        "QuantizedLinear",
        "getattr " "bits",
        "isinstance " "QuantizedLinear",
        "read bits",
        "read " "group_size",
        "read " "scales",
    ),
    ("installed/models/quantized_verifier.py", "optimized_nvfp4_argmax"): (
        "QuantizedLinear",
        "isinstance " "QuantizedLinear",
        "read bits",
        "read " "group_size",
        "read scales",
    ),
    ("installed/models/quantized_verifier.py", "singleton_quantized_linear"): (
        "QuantizedLinear",
        "isinstance " "QuantizedLinear",
        "read bits",
        "read " "group_size",
        "read " "scales",
    ),
    ("installed/models/quantized_verifier.py", "supports_optimized_affine_head"): (
        "QuantizedLinear",
        "isinstance " "QuantizedLinear",
        "read " "bits",
        "read " "scales",
    ),
    ("installed/models/quantized_verifier.py", "supports_quantization"): (
        "QuantizedLinear",
        "QuantizedSwitchLinear",
        "isinstance " "QuantizedLinear",
        "isinstance " "QuantizedSwitchLinear",
        "read bits",
        "read " "group_size",
    ),
    ("omlx/admin/ane_tuning.py", "_calibrate_components_sync"): (
        "getattr bits",
        "getattr group_size",
        "read bits",
        "read scales",
    ),
    ("omlx/admin/ane_tuning.py", "_calibrate_fused_components_sync"): (
        "read bits",
        "read scales",
    ),
    ("omlx/admin/ane_tuning.py", "_validate_k2_tuning_model"): (
        "QuantizedLinear",
        "isinstance " "QuantizedLinear",
        "read bits",
    ),
    ("omlx/affine4.py", "Affine4Codec.__init__"): ("read bits",),
    ("omlx/affine4.py", "Affine4Codec._codes"): ("read bits",),
    ("omlx/affine4.py", "Affine4Codec.dequantize_rotated"): ("read bits",),
    ("omlx/affine4.py", "Affine4Codec.maximum_code"): ("read bits",),
    ("omlx/affine4.py", "Affine4Codec.minimum_code"): ("read bits",),
    ("omlx/affine4.py", "Affine4Codec.quantize"): ("read bits",),
    ("omlx/affine4.py", "Affine4Codec.values_per_word"): ("read bits",),
    ("omlx/affine4.py", "Affine4KVCache.__init__"): ("read bits",),
    ("omlx/affine4.py", "Affine4KVCache._try_fused_kv_quantize"): ("read bits",),
    ("omlx/affine4.py", "Affine4KVCache.meta_state"): ("read bits",),
    ("omlx/affine4.py", "Affine4KVCache.rebuild_codecs"): ("read bits",),
    ("omlx/affine4.py", "BatchAffine4KVCache.__init__"): ("read bits",),
    ("omlx/affine4.py", "BatchAffine4KVCache._new_single_cache"): ("read bits",),
    ("omlx/cluster/tensor_strategies.py", "_shard_nemotron_h"): ("hasattr scales",),
    ("omlx/cluster/tensor_strategies.py", "_shard_switch_mlp_uneven"): (
        "hasattr scales",
        "read scales",
    ),
    ("omlx/formats/__init__.py", "_face"): ("getattr bits", "getattr group_size"),
    ("omlx/formats/__init__.py", "_probe_mlx_mode"): ("quantize-call quantize",),
    ("omlx/formats/affine.py", "AffineFormat.codes"): ("read bits", "read group_size"),
    ("omlx/formats/affine.py", "AffineFormat.dot"): ("read bits", "read group_size"),
    ("omlx/formats/affine.py", "AffineFormat.tile"): ("read bits", "read group_size"),
    ("omlx/formats/kernels.py", "Tables.add"): ("read bits",),
    ("omlx/formats/kernels.py", "Tables.header"): ("read bits",),
    ("omlx/formats/linear.py", "FormatLinear._bind_one_row_mlx_fast"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/formats/linear.py", "FormatLinear._stock_matmul"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/formats/linear.py", "_FormatLayer._decoded"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/formats/linear.py", "_FormatLayer._extra_repr"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/formats/linear.py", "_FormatLayer._stacked_outputs"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/formats/linear.py", "_plan_existing_dense_warm"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/formats/oqh.py", "OqhAffineFormat.codes"): ("read bits", "read group_size"),
    ("omlx/formats/oqh.py", "OqhAffineFormat.dot"): ("read bits", "read group_size"),
    ("omlx/formats/oqh.py", "OqhAffineFormat.tile"): ("read bits", "read group_size"),
    ("omlx/formats/oqh.py", "OqhFormat.codes"): ("read bits",),
    ("omlx/formats/oqh.py", "OqhFormat.dot"): ("read bits",),
    ("omlx/formats/oqh.py", "OqhFormat.tile"): ("read bits",),
    ("omlx/formats/tile_dense.py", "_resolve_cell"): ("read bits", "read group_size"),
    ("omlx/formats/tile_dense.py", "build_plan"): ("read bits", "read group_size"),
    ("omlx/lora_adapters.py", "_linear_dims"): (
        "QuantizedLinear",
        "isinstance QuantizedLinear",
        "read bits",
    ),
    ("omlx/models/decision.py", "DecisionBackbone.logits"): (
        "QuantizedEmbedding",
        "isinstance " "QuantizedEmbedding",
    ),
    ("omlx/models/decision.py", "DecisionBackbone.output_rows"): (
        "QuantizedEmbedding",
        "QuantizedLinear",
        "isinstance " "QuantizedEmbedding",
        "isinstance " "QuantizedLinear",
        "read bits",
        "read group_size",
        "read scales",
    ),
    ("omlx/models/vlm.py", "restore_bonsai_quantized_modules"): (
        "QuantizedEmbedding",
        "QuantizedLinear",
    ),
    ("omlx/oq.py", "<module>"): ("QuantizedSwitchLinear",),
    ("omlx/oq.py", "OQImatrixCollector._is_capture_module"): ("QuantizedLinear",),
    ("omlx/oq.py", "OQImatrixCollector._module_in_dim"): ("getattr bits",),
    ("omlx/oq.py", "OQImatrixCollector.collect_multi"): ("getattr bits",),
    ("omlx/oq.py", "_measure_sensitivity_from_quantized_model"): (
        "getattr bits",
        "getattr group_size",
        "hasattr scales",
        "quantize-call quantize",
        "read scales",
    ),
    ("omlx/oq.py", "_qdq_weight_only"): ("quantize-call quantize",),
    ("omlx/oq.py", "_quantize_chunked"): ("quantize-call quantize",),
    ("omlx/oqh/modules.py", "OqhAffineLinear._affine"): (
        "QuantizedLinear",
        "read bits",
        "read group_size",
    ),
    ("omlx/oqh/modules.py", "OqhAffineSwitchLinear.matmul_rotated"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/oqh/modules.py", "OqhLinear._affine"): ("QuantizedLinear", "read bits"),
    ("omlx/oqh/modules.py", "OqhLinear.matmul_rotated"): ("read bits",),
    ("omlx/oqh/modules.py", "OqhLinear.pair_rotated"): ("read bits",),
    ("omlx/oqh/modules.py", "OqhSwitchLinear._gemv"): ("read bits",),
    ("omlx/oqh/modules.py", "OqhSwitchLinear.matmul_rotated"): ("read bits",),
    ("omlx/oqh/runtime.py", "_pairs"): ("read bits",),
    ("omlx/oqh/writer.py", "OqhWriter._emit"): ("read group_size",),
    ("omlx/oqh/writer.py", "OqhWriter._quantize"): (
        "quantize-call quantize",
        "read group_size",
    ),
    ("omlx/oqh/writer.py", "_float_weight"): (
        "read bits",
        "read group_size",
        "read scales",
    ),
    (
        "omlx/patches/bailing_hybrid/bailing_hybrid_model.py",
        "Model._convert_fp8_block_weights",
    ): ("quantize-call " "quantize",),
    ("omlx/patches/batched_decode_qmm.py", "apply_batched_decode_qmm_patch"): (
        "QuantizedLinear",
    ),
    (
        "omlx/patches/batched_decode_qmm.py",
        "apply_batched_decode_qmm_patch.patched_call",
    ): ("read " "bits", "read " "group_size", "read " "scales"),
    ("omlx/patches/deepseek_v4/decode_consistency.py", "apply"): ("QuantizedLinear",),
    ("omlx/patches/deepseek_v4/moe_offload.py", "_is_quantized"): (
        "QuantizedSwitchLinear",
    ),
    ("omlx/patches/deepseek_v4/switch_layers.py", "QuantizedSwitchLinear.__call__"): (
        "read " "bits",
        "read " "group_size",
    ),
    ("omlx/patches/deepseek_v4/switch_layers.py", "QuantizedSwitchLinear.__init__"): (
        "quantize-call " "quantize",
    ),
    (
        "omlx/patches/deepseek_v4/switch_layers.py",
        "QuantizedSwitchLinear._can_use_affine_blocks",
    ): ("read " "bits", "read " "group_size"),
    (
        "omlx/patches/deepseek_v4/switch_layers.py",
        "QuantizedSwitchLinear._can_use_mxfp4_blocks",
    ): ("read " "bits", "read " "group_size"),
    ("omlx/patches/deepseek_v4/switch_layers.py", "QuantizedSwitchLinear.input_dims"): (
        "read " "group_size",
        "read " "scales",
    ),
    ("omlx/patches/deepseek_v4/switch_layers.py", "SwitchGLU.__call__"): (
        "QuantizedSwitchLinear",
        "isinstance " "QuantizedSwitchLinear",
        "read bits",
        "read " "group_size",
    ),
    ("omlx/patches/deepseek_v4/switch_layers.py", "SwitchLinear.to_quantized"): (
        "QuantizedSwitchLinear",
        "quantize-call " "quantize",
    ),
    ("omlx/patches/deepseek_v4/switch_layers.py", "_sort_threshold"): (
        "QuantizedSwitchLinear",
        "isinstance " "QuantizedSwitchLinear",
    ),
    ("omlx/patches/deepseek_v4/switch_layers.py", "has_native_block_kernels"): (
        "QuantizedSwitchLinear",
        "isinstance " "QuantizedSwitchLinear",
        "read " "bits",
        "read " "group_size",
    ),
    ("omlx/patches/deepseek_v4/switch_layers.py", "install_nvfp4_global_scales"): (
        "QuantizedSwitchLinear",
        "isinstance " "QuantizedSwitchLinear",
    ),
    (
        "omlx/patches/deepseek_v4/utils_patch.py",
        "_build_patched_load_model._load_deepseek_v4._maybe_qq",
    ): (
        "QuantizedLinear",
        "isinstance " "QuantizedLinear",
        "read " "bits",
        "read " "group_size",
    ),
    (
        "omlx/patches/deepseek_v4/utils_patch.py",
        "_build_patched_load_model._load_deepseek_v4._quantize",
    ): ("class_predicate", "quantize-call " "quantize"),
    (
        "omlx/patches/deepseek_v4/utils_patch.py",
        "_build_patched_load_model._load_deepseek_v4._quantize.class_predicate",
    ): ("class_predicate",),
    ("omlx/patches/deepseek_v4/verify_qmv.py", "eligible"): (
        "read bits",
        "read group_size",
        "read scales",
    ),
    ("omlx/patches/deepseek_v4/verify_qmv.py", "exact_verify_multi_qmv"): (
        "read " "group_size",
        "read scales",
    ),
    ("omlx/patches/deepseek_v4/verify_qmv.py", "exact_verify_qmv"): (
        "read group_size",
        "read scales",
    ),
    ("omlx/patches/deepseek_v4/verify_qmv.py", "multi_eligible"): (
        "hasattr scales",
        "read bits",
        "read group_size",
        "read scales",
    ),
    ("omlx/patches/deepseek_v41/a8.py", "_affine_gs64"): (
        "getattr bits",
        "getattr group_size",
        "read bits",
    ),
    ("omlx/patches/deepseek_v41/a8.py", "_mx"): ("getattr group_size", "read bits"),
    ("omlx/patches/deepseek_v41/a8.py", "_plan"): ("read scales",),
    ("omlx/patches/deepseek_v41/a8.py", "tag_dense_a8"): ("read bits", "read scales"),
    ("omlx/patches/deepseek_v41/decode_fused.py", "_woa_mxfp8"): (
        "quantize-call " "quantize",
    ),
    ("omlx/patches/deepseek_v41/language.py", "Expert.__call__.gather"): (
        "read bits",
        "read " "group_size",
        "read scales",
    ),
    ("omlx/patches/deepseek_v41/oq.py", "quantize_engram"): ("quantize-call quantize",),
    ("omlx/patches/deepseek_v41/oqh.py", "_quantize_module"): (
        "quantize-call quantize",
    ),
    ("omlx/patches/deepseek_v41/oqh.py", "float_experts"): (
        "read bits",
        "read group_size",
        "read scales",
    ),
    ("omlx/patches/deepseek_v41/quantization.py", "QuantizedProjection"): (
        "QuantizedSwitchLinear",
    ),
    (
        "omlx/patches/deepseek_v41/quantization.py",
        "QuantizedProjection._can_use_affine_blocks",
    ): ("read " "bits", "read " "group_size"),
    ("omlx/patches/deepseek_v41/quantization.py", "QuantizedProjection.input_dims"): (
        "read " "bits",
    ),
    (
        "omlx/patches/deepseek_v41/quantization.py",
        "QuantizedProjection.project_quantized",
    ): ("read " "bits", "read " "group_size", "read " "scales"),
    ("omlx/patches/deepseek_v41/sensitivity.py", "_quantized_block"): (
        "read bits",
        "read group_size",
        "read scales",
    ),
    ("omlx/patches/glm_moe_dsa/deepseek_v32.py", "Indexer.__init__"): (
        "QuantizedLinear",
    ),
    ("omlx/patches/glm_moe_dsa/deepseek_v32.py", "Model.sanitize"): (
        "quantize-call " "quantize",
    ),
    ("omlx/patches/glm_moe_dsa/moe_offload.py", "_is_quantized"): (
        "QuantizedSwitchLinear",
    ),
    ("omlx/patches/glm_moe_dsa/sparse_mla.py", "q8_vup_flat"): (
        "getattr bits",
        "getattr group_size",
        "hasattr scales",
    ),
    ("omlx/patches/glm_moe_dsa/switch_layers.py", "QuantizedSwitchLinear.__call__"): (
        "read " "bits",
        "read " "group_size",
    ),
    ("omlx/patches/glm_moe_dsa/switch_layers.py", "QuantizedSwitchLinear.__init__"): (
        "quantize-call " "quantize",
    ),
    ("omlx/patches/glm_moe_dsa/switch_layers.py", "QuantizedSwitchLinear.input_dims"): (
        "read " "group_size",
        "read " "scales",
    ),
    ("omlx/patches/glm_moe_dsa/switch_layers.py", "SwitchLinear.to_quantized"): (
        "QuantizedSwitchLinear",
        "quantize-call " "quantize",
    ),
    ("omlx/patches/inkling2/dspark.py", "Inkling2MarkovHead.fold"): (
        "QuantizedLinear",
        "isinstance " "QuantizedLinear",
    ),
    ("omlx/patches/inkling2/dspark.py", "load_inkling2_dspark_model"): (
        "class_predicate",
        "quantize-call " "quantize",
    ),
    ("omlx/patches/inkling2/model.py", "ExpertProj.__call__"): (
        "read bits",
        "read group_size",
        "read scales",
    ),
    ("omlx/patches/inkling2/model.py", "ExpertProj.__init__"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/patches/k2_horizon/ane_prefill.py", "PrefillMLP.__init__"): (
        "QuantizedLinear",
        "isinstance " "QuantizedLinear",
    ),
    ("omlx/patches/k2_horizon/ane_prefill.py", "_projection_part"): (
        "QuantizedLinear",
        "isinstance " "QuantizedLinear",
        "read bits",
        "read group_size",
        "read scales",
    ),
    ("omlx/patches/k2_horizon/ane_prefill.py", "enable_ane_prefill"): (
        "QuantizedLinear",
        "getattr " "group_size",
        "isinstance " "QuantizedLinear",
        "read scales",
    ),
    ("omlx/patches/laguna/laguna_model.py", "Model._dequantize_fp8_block_weights"): (
        "quantize-call " "quantize",
    ),
    ("omlx/patches/m5_gather_qmm.py", "_sorted_gather_qmm_defective"): (
        "quantize-call " "quantize",
    ),
    ("omlx/patches/m5_gather_qmm.py", "fused_gate_up_activation"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/patches/m5_gather_qmm.py", "split_gate_up_swiglu_oai"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/patches/m5_gather_qmm_a8.py", "_self_test"): ("quantize-call quantize",),
    ("omlx/patches/m5_gather_qmm_a8.py", "_self_test_extended.quant"): (
        "quantize-call " "quantize",
    ),
    ("omlx/patches/m5_gather_qmm_nax.py", "_canary_problem"): (
        "quantize-call quantize",
    ),
    ("omlx/patches/mimo_v2/decode_fast.py", "_experts"): ("read scales",),
    ("omlx/patches/mimo_v2/decode_fast.py", "_fuse_qkv"): (
        "QuantizedLinear",
        "isinstance QuantizedLinear",
    ),
    ("omlx/patches/mimo_v2/decode_fast.py", "_vector_attention"): ("hasattr bits",),
    ("omlx/patches/mimo_v2/decode_fast.py", "run_layers"): ("hasattr bits",),
    ("omlx/patches/mimo_v2/mimo_v2_model.py", "Attention.__call__"): ("hasattr bits",),
    ("omlx/patches/mimo_v2/moe_decode.py", "supported"): (
        "getattr bits",
        "getattr group_size",
        "read scales",
    ),
    ("omlx/patches/minimax_m3_dspark.py", "load_minimax_dspark_model"): (
        "class_predicate",
        "quantize-call " "quantize",
    ),
    ("omlx/patches/mlx_lm_mtp/batch_generator.py", "_CoarseDraftHead.candidates"): (
        "read " "bits",
        "read " "group_size",
    ),
    ("omlx/patches/mlx_lm_mtp/batch_generator.py", "_coarse_draft_head"): (
        "QuantizedLinear",
        "isinstance " "QuantizedLinear",
        "quantize-call " "quantize",
        "read bits",
        "read " "group_size",
    ),
    ("omlx/patches/mlx_lm_mtp/batch_generator.py", "_head_rows"): ("read scales",),
    ("omlx/patches/mlx_lm_mtp/batch_generator.py", "_quantize_mtp_fc"): (
        "QuantizedLinear",
    ),
    ("omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py", "_affine_parts"): (
        "read " "bits",
        "read " "group_size",
    ),
    ("omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py", "_oqh_group_parts"): (
        "read " "bits",
        "read " "group_size",
    ),
    ("omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py", "_oqh_parts"): (
        "read " "bits",
    ),
    ("omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py", "mla_head_qmv"): (
        "read " "bits",
        "read " "group_size",
    ),
    (
        "omlx/patches/mlx_vlm_minimax_m3_compat/__init__.py",
        "_patch_nn_quantize_for_ignored_layers.patched_quantize.combined_predicate",
    ): ("class_predicate",),
    ("omlx/patches/mlx_vlm_minimax_m3_compat/a8.py", "_affine"): ("read scales",),
    ("omlx/patches/mlx_vlm_minimax_m3_compat/a8.py", "routed"): (
        "read bits",
        "read scales",
    ),
    ("omlx/patches/mlx_vlm_mtp/qwen35_verify_linear.py", "<module>"): (
        "QuantizedLinear",
    ),
    (
        "omlx/patches/mlx_vlm_mtp/qwen35_verify_linear.py",
        "apply.target_verify_linears",
    ): ("QuantizedLinear", "isinstance " "QuantizedLinear"),
    ("omlx/patches/mlx_vlm_mtp/qwen35_verify_linear.py", "apply.use_verify_dense"): (
        "QuantizedLinear",
        "isinstance " "QuantizedLinear",
    ),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/cache.py",
        "BatchQuantizedKVCache.dequantize_for_apc",
    ): ("read " "bits", "read " "group_size"),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/cache.py",
        "BatchQuantizedKVCache.extract",
    ): ("read " "bits", "read " "group_size"),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/cache.py",
        "BatchQuantizedKVCache.meta_state",
    ): ("read " "bits", "read " "group_size"),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/cache.py",
        "BatchQuantizedKVCache.update_and_fetch",
    ): ("quantize-call " "quantize", "read " "bits", "read " "group_size"),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/cache.py",
        "BatchQuantizedKVCache.update_and_fetch._init",
    ): ("read " "group_size",),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/cache.py",
        "KVCache.to_quantized",
    ): ("quantize-call " "quantize",),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/cache.py",
        "QuantizedKVCache.dequantize_for_apc",
    ): ("read " "bits", "read " "group_size"),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/cache.py",
        "QuantizedKVCache.meta_state",
    ): ("read " "bits", "read " "group_size"),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/cache.py",
        "QuantizedKVCache.update_and_fetch",
    ): ("quantize-call " "quantize", "read " "bits", "read " "group_size"),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/cache.py",
        "QuantizedKVCache.update_and_fetch.init_quant",
    ): ("read " "group_size",),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/hc_fused.py",
        "_affine_tensors_ok",
    ): (
        "getattr " "bits",
        "getattr " "group_size",
        "getattr " "scales",
        "read " "scales",
    ),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/hc_fused.py",
        "_nax_prefill",
    ): ("read " "bits",),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/hc_fused.py",
        "_nax_prefill_ok",
    ): ("read " "group_size",),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/hc_fused.py",
        "_projection_face",
    ): ("read " "bits", "read " "group_size"),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/hc_prefill_nax.py",
        "down_silu",
    ): ("read " "bits", "read " "scales"),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/hc_prefill_nax.py",
        "up_tail",
    ): ("read " "bits", "read " "scales"),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py",
        "QSAKVCache.to_quantized",
    ): ("quantize-call " "to_quantized",),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py",
        "QSAQuantizedKVCache.extract",
    ): ("read " "bits", "read " "group_size"),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py",
        "Qwen4ExpRMSNorm.__call__",
    ): ("read " "group_size",),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py",
        "ShardedEmbedding._fuse_format_shards",
    ): ("read " "bits", "read " "group_size"),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py",
        "ShardedEmbedding._uniform_affine_layout",
    ): ("read " "bits", "read " "group_size", "read " "scales"),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py",
        "ShardedEmbedding._windowed_lookup",
    ): ("read " "scales",),
    (
        "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py",
        "_affine_rows",
    ): ("getattr " "scales", "hasattr " "bits", "hasattr " "group_size"),
    ("omlx/patches/moe_expert_offload.py", "_is_quantized_switch_linear"): (
        "QuantizedSwitchLinear",
    ),
    ("omlx/patches/moe_gate_up_fusion.py", "_is_quantized"): ("QuantizedSwitchLinear",),
    ("omlx/patches/moe_gate_up_fusion.py", "can_fuse"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/patches/muse_spark/model.py", "_unify_attention"): (
        "quantize-call quantize",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_affine_spec"): (
        "getattr bits",
        "getattr group_size",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_compile_gdn"): ("read scales",),
    ("omlx/patches/qwen35_ane_prefill.py", "_compile_gdn.dense_logical_slice"): (
        "read " "scales",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_compile_pair"): (
        "read bits",
        "read group_size",
        "read scales",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_compile_pair.dense_slice"): (
        "read scales",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_eligible_affine_linear"): (
        "QuantizedLinear",
        "getattr bits",
        "getattr " "group_size",
        "getattr scales",
        "isinstance " "QuantizedLinear",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_eligible_gdn"): (
        "getattr scales",
        "read bits",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_eligible_pair"): (
        "getattr bits",
        "getattr scales",
        "read scales",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_enable_dual_procedure_banks"): (
        "getattr " "bits",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_gate_up_gpu_suffix"): ("read scales",),
    ("omlx/patches/qwen35_ane_prefill.py", "_pack_affine_gdn_suffix"): ("read scales",),
    ("omlx/patches/qwen35_ane_prefill.py", "_post_ane_linear"): ("getattr bits",),
    ("omlx/patches/qwen35_ane_prefill.py", "_prepare_cpu_linear"): (
        "getattr scales",
        "read scales",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_prepare_down_for_bank"): (
        "getattr scales",
        "read scales",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_prepare_down_for_bank.dense_rows"): (
        "read " "scales",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_prepare_fused_down_for_bank"): (
        "getattr " "bits",
        "getattr " "group_size",
        "read " "scales",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_prepare_fused_down_for_bank.dense_rows"): (
        "read " "scales",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_prepare_gdn_for_bank"): ("read scales",),
    (
        "omlx/patches/qwen35_ane_prefill.py",
        "_prepare_gdn_for_bank.dense_logical_slice",
    ): ("read " "scales",),
    ("omlx/patches/qwen35_ane_prefill.py", "_prepare_gdn_runtime_state"): (
        "read scales",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_prepare_pair_for_bank"): (
        "read bits",
        "read group_size",
        "read scales",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_prepare_pair_for_bank.dense_slice"): (
        "read " "scales",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_prepare_pair_runtime_state"): (
        "read bits",
        "read " "group_size",
        "read scales",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "_tail_qmm_or_linear"): ("getattr bits",),
    ("omlx/patches/qwen35_ane_prefill.py", "_warm_cpu_sharing_path._warm_input"): (
        "read " "bits",
        "read " "scales",
    ),
    ("omlx/patches/qwen35_ane_prefill.py", "enable_qwen35_ane_prefill"): (
        "getattr " "scales",
    ),
    ("omlx/patches/qwen35_fa256_attention.py", "_has_quantized_cache"): (
        "hasattr bits",
    ),
    ("omlx/patches/qwen35_gdn_prework.py", "_canonical_projection"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/patches/qwen35_gdn_prework.py", "_format_projection"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/patches/qwen35_gdn_prework.py", "_stack_format_inputs"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/patches/qwen35_moe_gate_up.py", "_can_fuse"): (
        "QuantizedSwitchLinear",
        "isinstance " "QuantizedSwitchLinear",
        "read bits",
        "read group_size",
    ),
    ("omlx/patches/qwen35_moe_gate_up.py", "_fuse_one"): (
        "QuantizedSwitchLinear",
        "isinstance " "QuantizedSwitchLinear",
    ),
    ("omlx/patches/qwen35_moe_gate_up.py", "_quantized_operands"): (
        "QuantizedSwitchLinear",
        "read bits",
        "read group_size",
        "type-is " "QuantizedSwitchLinear",
    ),
    ("omlx/patches/qwen35_moe_routed_decode.py", "_build_plan"): (
        "QuantizedSwitchLinear",
    ),
    ("omlx/patches/qwen35_moe_routed_decode.py", "_shared_formats"): (
        "QuantizedLinear",
    ),
    ("omlx/patches/qwen35_moe_routed_decode.py", "_window_fast"): ("getattr bits",),
    ("omlx/patches/qwen35_moe_routed_decode.py", "_window_coalesce_modes"): (
        "getattr bits",
        "getattr group_size",
    ),
    ("omlx/patches/qwen35_moe_routed_decode.py", "_window_down_policies"): (
        "getattr bits",
        "getattr group_size",
    ),
    ("omlx/patches/qwen35_moe_routed_decode.py", "_window_router_ok"): (
        "QuantizedLinear",
        "read bits",
        "read group_size",
    ),
    ("omlx/patches/qwen35_moe_routed_decode.py", "warm_moe_model"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/patches/qwen35_oq_a8.py", "OqA8Plan.kernel"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/patches/qwen35_oq_a8.py", "OqA8Plan.layout"): ("read bits",),
    ("omlx/patches/qwen35_oq_a8.py", "_classify_uncached"): (
        "QuantizedLinear",
        "getattr bits",
        "getattr group_size",
        "getattr scales",
        "isinstance QuantizedLinear",
    ),
    ("omlx/patches/qwen35_oq_a8.py", "_is_q8_linear"): (
        "QuantizedLinear",
        "getattr bits",
        "isinstance QuantizedLinear",
    ),
    ("omlx/patches/qwen35_oq_a8.py", "_prepared_weights"): ("read bits", "read scales"),
    ("omlx/patches/qwen35_oq_a8.py", "_refuse_unclassified"): (
        "QuantizedLinear",
        "getattr bits",
        "isinstance " "QuantizedLinear",
    ),
    ("omlx/patches/qwen35_packed_linear.py", "eligible"): (
        "QuantizedLinear",
        "read bits",
        "read group_size",
        "read scales",
        "type-is QuantizedLinear",
    ),
    ("omlx/patches/qwen35_padded_row_attention.py", "_plain_padded_cache"): (
        "hasattr " "bits",
    ),
    ("omlx/patches/qwen35_q4_mlp.py", "_VLMQuantizedPrefillLinear"): (
        "QuantizedLinear",
    ),
    ("omlx/patches/qwen35_q4_mlp.py", "_can_route_affine_linear"): ("getattr bits",),
    ("omlx/patches/qwen35_q4_mlp.py", "_can_route_affine_linear_shape"): (
        "getattr bits",
    ),
    ("omlx/patches/qwen35_q4_mlp.py", "_is_supported_affine_linear_shape"): (
        "QuantizedLinear",
        "getattr " "bits",
        "getattr " "group_size",
        "getattr " "scales",
        "isinstance " "QuantizedLinear",
    ),
    ("omlx/patches/qwen35_q4_mlp.py", "_linear_qmm"): (
        "QuantizedLinear",
        "getattr bits",
        "getattr group_size",
        "read scales",
    ),
    ("omlx/patches/qwen35_q4_mlp.py", "_post_ane_qmm_or_linear"): ("getattr bits",),
    ("omlx/patches/qwen35_q4_mlp.py", "_q8_backend_candidate"): ("getattr bits",),
    ("omlx/patches/qwen35_q4_mlp.py", "apply_qwen35_q4_prefill_linear_patch"): (
        "QuantizedLinear",
        "type-is " "QuantizedLinear",
    ),
    ("omlx/patches/qwen35_verify_qmm.py", "_patch_verify_swiglu._feed_forward"): (
        "read " "bits",
        "read " "group_size",
        "read " "scales",
    ),
    ("omlx/patches/qwen35_verify_qmm.py", "apply_verify_qmm_patch"): (
        "QuantizedLinear",
    ),
    ("omlx/patches/qwen35_verify_qmm.py", "apply_verify_qmm_patch.patched_call"): (
        "read " "bits",
        "read " "group_size",
        "read " "scales",
    ),
    ("omlx/patches/qwen35_verify_qmm.py", "run_route"): (
        "read bits",
        "read group_size",
        "read scales",
    ),
    ("omlx/patches/qwen35_verify_qmm.py", "sg8_group_eligible"): (
        "QuantizedLinear",
        "isinstance " "QuantizedLinear",
        "read bits",
        "read group_size",
        "read scales",
    ),
    ("omlx/patches/qwen35_verify_qmm.py", "sg8_swiglu_eligible"): (
        "QuantizedLinear",
        "isinstance " "QuantizedLinear",
        "read bits",
        "read group_size",
        "read scales",
    ),
    ("omlx/patches/qwen35_verify_qmm.py", "takes_verify_route"): (
        "read bits",
        "read group_size",
        "read scales",
    ),
    ("omlx/patches/qwen35_verify_qmm.py", "vk_group_sg8"): ("read bits", "read scales"),
    ("omlx/patches/qwen35_verify_qmm.py", "vk_swiglu_sg8"): (
        "read bits",
        "read group_size",
        "read scales",
    ),
    ("omlx/patches/qwen35_verify_sdpa_split.py", "_eligible"): ("hasattr bits",),
    ("omlx/patches/qwen38_modelopt_mixed.py", "ScaledQuantizedLinear.__call__"): (
        "read " "bits",
        "read " "group_size",
    ),
    ("omlx/patches/raon1_mini/raon1_mini_model.py", "MLA.__call__"): ("hasattr bits",),
    ("omlx/patches/raon1_mini/raon1_mini_model.py", "MergedSwitchGLU._half"): (
        "QuantizedSwitchLinear",
        "isinstance " "QuantizedSwitchLinear",
    ),
    ("omlx/patches/raon1_mini/raon1_mini_model.py", "_qparams"): (
        "QuantizedLinear",
        "isinstance " "QuantizedLinear",
        "read bits",
        "read group_size",
    ),
    ("omlx/patches/raon1_mini/raon1_mini_model.py", "_share_parts"): ("read scales",),
    ("omlx/patches/row_exact_qmv.py", "<module>"): ("QuantizedLinear",),
    ("omlx/patches/row_exact_qmv.py", "_face"): (
        "QuantizedLinear",
        "isinstance QuantizedLinear",
        "read bits",
    ),
    ("omlx/patches/row_exact_qmv.py", "_format_plans"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/patches/row_exact_qmv.py", "_linear_in_out"): (
        "QuantizedLinear",
        "isinstance QuantizedLinear",
        "read bits",
    ),
    ("omlx/patches/row_exact_qmv.py", "_linear_sig"): (
        "read bits",
        "read group_size",
    ),
    ("omlx/patches/row_exact_qmv.py", "warm_model"): (
        "QuantizedLinear",
        "isinstance QuantizedLinear",
    ),
    ("omlx/patches/row_exact_qmv.py", "warm_model_dispatch"): (
        "QuantizedLinear",
        "isinstance QuantizedLinear",
    ),
    ("omlx/patches/sdpa256_attention.py", "_should_route"): ("hasattr bits",),
    (
        "omlx/patches/turboquant_attention.py",
        "_patch_update_eval_policy.update_and_fetch",
    ): ("quantize-call " "quantize",),
    ("omlx/prefill_helper/adapters/qwen4_exp.py", "_ple_reader"): ("hasattr scales",),
    ("omlx/scheduler.py", "_dense_kv_template"): ("getattr scales",),
    ("omlx/speculative/dflash/__init__.py", "quantize_drafter"): (
        "class_predicate",
        "quantize-call " "quantize",
    ),
    ("omlx/speculative/dflash/drafter.py", "DFlashDrafter._draft_batched"): (
        "hasattr " "scales",
        "read " "scales",
    ),
    ("omlx/speculative/dflash/selection.py", "fused_select_eligible"): (
        "hasattr scales",
    ),
    ("omlx/turboquant_kv.py", "BatchTurboQuantKVCache._new_single_cache"): (
        "read bits",
    ),
    ("omlx/turboquant_kv.py", "BatchTurboQuantKVCache._update_rows"): (
        "quantize-call " "quantize",
    ),
}


def _decide(key):
    if key in SCOPE_DECISIONS:
        return SCOPE_DECISIONS[key]
    return PATH_DECISIONS.get(key[0])


def test_every_quantized_layer_reader_is_tabled():
    found, unparsable = _readers()
    assert not unparsable, f"unparsable files are unaudited files: {unparsable}"
    unlisted = sorted(set(found) - set(EXPECTED))
    stale = sorted(set(EXPECTED) - set(found))
    assert not unlisted, f"new readers are not tabled: {unlisted}"
    assert not stale, f"tabled readers are gone: {stale}"
    found_paths = {path for path, _ in found}
    unused_paths = sorted(set(PATH_DECISIONS) - found_paths)
    unused_scopes = sorted(set(SCOPE_DECISIONS) - set(found))
    assert not unused_paths, f"unused PATH_DECISIONS entries: {unused_paths}"
    assert not unused_scopes, f"unused SCOPE_DECISIONS entries: {unused_scopes}"
    for key, signals in found.items():
        assert signals == EXPECTED[key], (key, EXPECTED[key], signals)
        decision, evidence = _decide(key)
        assert decision and evidence, key
    pending = sorted(key for key in found if _decide(key)[0].startswith("pending"))
    assert pending, "hc_prefill_nax stays pending (DEFERRED 59)"
    assert all(
        path
        == "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/hc_prefill_nax.py"
        for path, _ in pending
    ), pending


PROOF = "\n".join(
    [
        "import importlib",
        "import sys",
        "import omlx.engine.vlm",
        "import omlx.utils.model_loading",
        "from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config",
        "from omlx.patches import mlx_vlm_qwen4_exp_compat as compat",
        "compat.apply_mlx_vlm_qwen4_exp_compat_patch()",
        "L = importlib.import_module('mlx_vlm.models.qwen4_exp.language')",
        "config = _tiny_config()",
        "L.LanguageModel(config.text_config, config)",
        "print(chr(10).join(sorted(n for n in sys.modules if n == 'omlx' or n.startswith('omlx.'))))",
    ]
)


def test_a_stock_qwen4_load_imports_no_not_served_reader():
    """The 'not served' decisions are a check, not a claim: importing
    ``omlx.engine.vlm`` and building a synthetic two-layer stock ``qwen4_exp``
    model in a fresh process must not import any ``UNSERVED`` module."""
    not_served = {
        path for path, (decision, _) in PATH_DECISIONS.items() if decision == UNSERVED
    }
    assert not_served
    result = subprocess.run(
        [sys.executable, "-c", PROOF],
        cwd=BASE,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    imported = set(result.stdout.split())
    hits = set()
    for path in not_served:
        module = path.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        if module in imported:
            hits.add(module)
    assert not hits, f"not-served readers imported by a stock load: {sorted(hits)}"


def test_audit_detects_union_alias_name_string_and_class_arg():
    snippets = (
        "def f(x):\n    return isinstance(x, int | QuantizedLinear)\n",
        "from mlx.nn import QuantizedLinear as Q\n"
        "def f(x):\n    return type(x) is Q\n",
        "def f(x):\n" "    return type(x).__qualname__ in ('QuantizedLinear',)\n",
        "def f(x):\n    return helper(x, QuantizedLinear)\n",
    )
    for snippet in snippets:
        found = _scan_source("snippet.py", snippet)
        assert any("QuantizedLinear" in signals for signals in found.values()), snippet


def test_audit_module_binding_and_short_alias():
    bound = _scan_source(
        "s.py",
        "_T = (nn.QuantizedLinear,)\ndef f(x):\n    return isinstance(x, _T)\n",
    )
    assert ("s.py", "<module>") in bound
    assert ("s.py", "f") in bound
    assert "QuantizedLinear" in bound[("s.py", "f")]
    assert "isinstance QuantizedLinear" in bound[("s.py", "f")]

    # Short alias must not match unrelated string literals like 'Q4'.
    assert (
        _scan_source(
            "s.py",
            "from mlx.nn import QuantizedLinear as Q\ndef g():\n    return 'Q4'\n",
        )
        == {}
    )

    # Docstrings alone do not make a function a quantized reader.
    assert (
        _scan_source(
            "s.py",
            'def h():\n    """Mentions QuantizedLinear only."""\n    return 1\n',
        )
        == {}
    )

    # Class bases and class_predicate / type-is are detected.
    subclass = _scan_source("s.py", "class C(nn.QuantizedLinear):\n    pass\n")
    assert ("s.py", "C") in subclass
    type_is = _scan_source(
        "s.py", "def k(x):\n    return type(x) is nn.QuantizedLinear\n"
    )
    assert "type-is QuantizedLinear" in type_is[("s.py", "k")]
    pred = _scan_source(
        "s.py", "def p(m):\n    return quantize(m, class_predicate=fn)\n"
    )
    assert "class_predicate" in pred[("s.py", "p")]


def test_hc_audit_evidence_names_u21_and_deferred_59():
    """The evidence text says what happened: U21 ported hc_fused and set the
    hook False; hc_prefill_nax is DEFERRED 59, not a live U21 item."""
    hc = "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/hc_fused.py"
    nax = "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/hc_prefill_nax.py"
    assert "DEL3-3 closes" not in PATH_DECISIONS[hc][1]
    assert "U21 ported hc_fused" in PATH_DECISIONS[hc][1]
    assert "DEFERRED 59" in PATH_DECISIONS[nax][1]


def test_release3_warm_audit_citations_exist():
    """NIT 8: every tests/test_release3_warm.py::* citation in SCOPE_DECISIONS exists."""
    import tests.test_release3_warm as warm_tests

    prefix = "tests/test_release3_warm.py::"
    checked = 0
    for _key, (_decision, proof) in SCOPE_DECISIONS.items():
        if proof.startswith(prefix):
            name = proof.removeprefix(prefix).split()[0]
            assert hasattr(
                warm_tests, name
            ), f"missing cited test {name!r} in tests/test_release3_warm.py"
            checked += 1
    assert checked > 0
