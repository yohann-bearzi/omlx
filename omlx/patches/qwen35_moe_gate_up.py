# SPDX-License-Identifier: Apache-2.0
"""Fuse supported MoE routed gate/up projections into one gather_qmm.

At single-token decode the routed-expert path runs three tiny
``gather_qmm`` launches per MoE layer (gate, up, down). Affine
quantization packs each output row independently, so concatenating the
gate and up expert weights along the output axis and issuing one
``gather_qmm`` over ``[E, 2*inter, hidden]`` is bit-identical to the two
separate calls while removing one launch per MoE layer per token
(issue #2238). The same fused weights serve prefill and batched decode,
also bit-exact.

The fusion runs once post-load and rewrites stock ``SwitchGLU``
instances in place: gate/up weights are concatenated, ``gate_up_proj``
replaces ``gate_proj``/``up_proj`` (mirroring the vendored GLM DSA
switch layers), and the class ``__call__`` gains a fused branch.
Instances without ``gate_up_proj`` keep the original code path.

mlx-vlm verification uses the same fused projection without sorting routes.
Separate gate/up views remain available to other upstream callers.
"""

from __future__ import annotations

import copy
import logging
import math
from functools import wraps
from typing import Any

import mlx.core as mx
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
    _scatter_unsort,
)
from mlx_vlm.models.switch_layers import (
    DECODE_BLOCK_SIZE,
)
from mlx_vlm.models.switch_layers import (
    QuantizedSwitchLinear as VLMQuantizedSwitchLinear,
)
from mlx_vlm.models.switch_layers import (
    SwitchGLU as VLMSwitchGLU,
)
from mlx_vlm.models.switch_layers import (
    SwitchLinear as VLMSwitchLinear,
)

from .. import formats
from ..formats import transforms
from ..formats.linear import FormatSwitchLinear
from ..scheduler import _sync_and_clear_cache
from . import m5_gather_qmm as _m5
from . import moe_verify_gather
from .m5_gather_qmm import fused_gate_up_activation
from .m5_gather_qmm_a8 import try_routed_a8
from .module_cache import cached_per_module
from .moe_routes import sort_routes

logger = logging.getLogger(__name__)

_CALL_PATCHED = False

# Loaded model classes whose module path marks a supported SwitchGLU family:
# mlx-lm Qwen3.5/3.6 and HyV3, Qwen4-Exp's inherited SwitchGLU, the oMLX
# single-checkpoint MTP wrapper, the vendored Laguna module, Muse Spark and
# mlx-vlm Flint.
_FAMILY_TOKENS = (
    "qwen3_5",
    "qwen3_6",
    "qwen35",
    "qwen4_exp",
    "laguna",
    "hy_v3",
    "muse_spark",
    "flint",
)


def _is_supported_family(model: Any) -> bool:
    module = type(model).__module__ or ""
    return any(token in module for token in _FAMILY_TOKENS)


def _can_fuse(switch_mlp: Any) -> bool:
    if hasattr(switch_mlp, "gate_up_proj"):
        return False
    if not (
        hasattr(switch_mlp, "gate_proj")
        and hasattr(switch_mlp, "up_proj")
        and hasattr(switch_mlp, "down_proj")
    ):
        return False
    gate, up = switch_mlp.gate_proj, switch_mlp.up_proj
    if type(gate) is not type(up):
        return False
    if any("global_scale" in p for p in (gate, up, switch_mlp.down_proj)):
        # NVFP4 tensor scales stay per projection, and the fused forward does
        # not apply the down projection's scale.
        return False
    if isinstance(gate, (QuantizedSwitchLinear, VLMQuantizedSwitchLinear)):
        if (gate.group_size, gate.bits, gate.mode) != (
            up.group_size,
            up.bits,
            up.mode,
        ):
            return False
        if (gate.get("biases") is None) != (up.get("biases") is None):
            return False
    elif isinstance(gate, FormatSwitchLinear):
        fmt = gate.fmt
        k, n = gate.input_dims, gate.output_dims
        if (
            fmt is not up.fmt
            or (k, gate.bits, gate.group_size)
            != (up.input_dims, up.bits, up.group_size)
            or not formats.stackable(fmt, k, n, gate.bits, gate.group_size)
        ):
            return False
        # The declaration covers one matrix. Both expert stacks must actually
        # store it, with the same concrete dtype even for a floating placeholder.
        layout = fmt.layout(k, n, gate.bits, gate.group_size)
        if layout.keys() != set(formats.all_tensor_names(fmt)):
            return False
        shared = set(fmt.shared_tensors)
        for name, (shape, dtype) in layout.items():
            a, b = gate.get(name), up.get(name)
            held = tuple(shape) if name in shared else (gate.num_experts, *shape)
            if (
                not isinstance(a, mx.array)
                or not isinstance(b, mx.array)
                or a.shape != held
                or a.shape != b.shape
                or a.dtype != b.dtype
                or (
                    not mx.issubdtype(a.dtype, mx.floating)
                    if dtype is None
                    else a.dtype != dtype
                )
            ):
                return False
            if name in shared and not mx.array_equal(a, b).item():
                return False
    elif not isinstance(gate, (SwitchLinear, VLMSwitchLinear)):
        return False
    if ("bias" in gate) != ("bias" in up):
        return False
    gate_w = gate[formats.tensor_names(gate)[0]]
    up_w = up[formats.tensor_names(up)[0]]
    if gate_w.shape != up_w.shape or gate_w.dtype != up_w.dtype:
        return False
    if isinstance(gate, FormatSwitchLinear):
        # Check every target before installing hooks or rewriting any block.
        gate.refuse_placeholders()
        up.refuse_placeholders()
    return True


def _fuse_one(switch_mlp: Any) -> None:
    gate, up = switch_mlp.gate_proj, switch_mlp.up_proj
    # Concat order is [gate, up] along the output axis, matching the GLM
    # DSA fused layout and the HF gate_up_proj checkpoint convention.
    keep_views = type(switch_mlp) is VLMSwitchGLU
    if isinstance(gate, FormatSwitchLinear):
        gate_up = gate._stacked_outputs(up)
        fused = gate_up.parameters()
    else:
        names = ["weight"]
        if isinstance(gate, (QuantizedSwitchLinear, VLMQuantizedSwitchLinear)):
            names.append("scales")
            if gate.get("biases") is not None:
                names.append("biases")
        if "bias" in gate:
            names.append("bias")
        fused = {
            name: mx.concatenate(
                [gate[name], up[name]], axis=-1 if name == "bias" else 1
            )
            for name in names
        }
        gate_up = copy.copy(gate) if keep_views else gate
    mx.eval(list(fused.values()))
    if not isinstance(gate, FormatSwitchLinear):
        for name, array in fused.items():
            setattr(gate_up, name, array)

    if keep_views:
        shared = (
            set(gate.fmt.shared_tensors)
            if isinstance(gate, FormatSwitchLinear)
            else set()
        )
        for name, array in fused.items():
            if name in shared:
                # One layer keeps one of it: the views read it whole.
                gate[name] = up[name] = array
                continue
            gate[name], up[name] = mx.split(array, 2, axis=-1 if name == "bias" else 1)
        # Verify uses these views on the engine thread after loading.
        mx.eval([gate[name] for name in fused], [up[name] for name in fused])
    switch_mlp.gate_up_proj = gate_up
    if not keep_views:
        del switch_mlp.gate_proj
        del switch_mlp.up_proj


def _bind_with_key(layer):
    bound = formats.input_transform(layer)
    return None if bound is None else (*bound, transforms.key(*bound))


def _stack_transforms(switch_mlp):
    """The gate/up and down input transforms of a fused stack, bound once."""
    return (
        _bind_with_key(getattr(switch_mlp, "gate_up_proj", None)),
        _bind_with_key(getattr(switch_mlp, "down_proj", None)),
    )


def _make_patched_call(orig_call):
    def patched(self, x: mx.array, indices: mx.array) -> mx.array:
        gate_up = getattr(self, "gate_up_proj", None)
        if gate_up is None:
            return orig_call(self, x, indices)

        pair = cached_per_module(self, "_omlx_stack_transforms", _stack_transforms)
        gu, dn = (None, None) if pair is None else pair
        down = self.down_proj
        prepared = gu is not None
        if prepared and x.size:
            # Rotate the token rows once, before sorting expands them.
            x = transforms.prepare_bound(gu, x)
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        token_rows = None
        if do_sort:
            # mlx-lm's _gather_sort with the replicated rows left lazy.
            x_tok, row_map, idx, inv_order = sort_routes(x, indices)
            x = x_tok[row_map]
            token_rows = (x_tok, row_map)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_act = None
        rotated = False
        if do_sort and not self.training and gu is None and dn is None:
            # Routed A8 (opt-in per model, see m5_gather_qmm_a8): the Gate+Up
            # on INT8 operands and the A16 Down; None keeps the A16 path.
            routed = try_routed_a8(
                self,
                token_rows,
                idx,
                # [B, L, k] routes: the sequence length, not B * L
                seq_len=int(indices.shape[-2]) if indices.ndim >= 3 else None,
            )
            if routed is not None:
                x = _scatter_unsort(routed, inv_order, indices.shape)
                return x.squeeze(-2)
        if do_sort and not self.training:
            # Sorted prefill on M5: the activation in the [gate; up]
            # matmul's epilogue, token rows read in place (bit-identical;
            # None keeps this path). When ``prepared`` is true the rows are
            # already rotated.
            x_act = fused_gate_up_activation(
                gate_up,
                x,
                idx,
                self.activation,
                token_rows=token_rows,
                **({"prepared": True} if prepared else {}),
            )
            # The epilogue's activation is in the plain basis: the down
            # projection applies its own transform when it is called.
        if x_act is None:
            x_gate_up = (
                gate_up._call_prepared(x, idx, sorted_indices=do_sort)
                if prepared
                else gate_up(x, idx, sorted_indices=do_sort)
            )
            limit = _m5._swiglu_limit(self.activation) if dn is not None else None
            if (
                dn is not None
                and dn[0].apply_split_swiglu is not None
                and x_gate_up.size
                and limit is not _m5._UNSUPPORTED
                and (limit is None or math.isfinite(limit))
            ):
                # The M3 steel route: split, SwiGLU and rotation in one pass.
                x_act = dn[0].apply_split_swiglu(x_gate_up, *dn[1], limit=limit)
                rotated = x_act is not None
            if x_act is None:
                x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
                x_act = self.activation(x_up, x_gate)
                if dn is not None and x_act.size:
                    x_act = transforms.prepare_bound(dn, x_act)
                    rotated = True
        x = (
            down._call_prepared(x_act, idx, sorted_indices=do_sort)
            if rotated
            else down(x_act, idx, sorted_indices=do_sort)
        )
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)

    return patched


def _quantized_operands(linear) -> tuple | None:
    """``mx.gather_qmm`` operands of an mlx-vlm QuantizedSwitchLinear without bias."""
    if type(linear) is not VLMQuantizedSwitchLinear or "bias" in linear:
        return None
    return (
        linear["weight"],
        linear["scales"],
        linear.get("biases"),
        linear.group_size,
        linear.bits,
        linear.mode,
    )


def _build_decode_plan(switch_mlp) -> tuple | None:
    # Training decode stops gradients through the routes; keep the stock body.
    if switch_mlp.training:
        return None
    gate_up = _quantized_operands(switch_mlp.get("gate_up_proj"))
    down = _quantized_operands(switch_mlp.get("down_proj"))
    activation = switch_mlp.get("activation")
    if gate_up is None or down is None or activation is None:
        return None
    return gate_up, down, activation


def _unsorted_switch(plan, x, indices):
    """The unsorted branch of the fused call on resolved operands.

    Same ops as ``gate_up(x, idx, sorted_indices=False)``, the split, the
    activation and ``down_proj(..., sorted_indices=False)`` through
    mlx-vlm's ``QuantizedSwitchLinear.__call__``.
    """
    (
        (gw, gs, gb, g_group, g_bits, g_mode),
        (dw, ds, db, d_group, d_bits, d_mode),
        act,
    ) = plan
    x_gate_up = mx.gather_qmm(
        mx.expand_dims(x, (-2, -3)),
        gw,
        gs,
        gb,
        rhs_indices=indices,
        transpose=True,
        group_size=g_group,
        bits=g_bits,
        mode=g_mode,
        sorted_indices=False,
    )
    x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
    x = mx.gather_qmm(
        act(x_up, x_gate),
        dw,
        ds,
        db,
        rhs_indices=indices,
        transpose=True,
        group_size=d_group,
        bits=d_bits,
        mode=d_mode,
        sorted_indices=False,
    )
    return x.squeeze(-2)


def _make_vlm_patched_call(original):
    fused_call = _make_patched_call(original)

    def patched(self, x, indices, weights=None, shared=None, residual=None):
        if getattr(self, "gate_up_proj", None) is None:
            return original(self, x, indices, weights, shared, residual)
        if not self.training and x.ndim == 3 and 1 < x.shape[1] <= DECODE_BLOCK_SIZE:
            # original() runs the gate and up views, which a format stack's
            # rows kernel would copy whole at every launch.
            fused_format = type(self.gate_up_proj) is FormatSwitchLinear
            if x.shape[0] * indices.shape[-1] >= 64 and not fused_format:
                return original(self, x, indices, weights, shared, residual)
            # The upstream kernel would copy the strided gate/up views.
            routed = _fused_verify_switch(self, x, indices)
            return self._combine(routed, weights, shared, residual)
        if indices.size < 64:
            # fused_call's unsorted branch (decode rows) with cached operands.
            plan = cached_per_module(
                self, "_omlx_gate_up_decode_plan", _build_decode_plan
            )
            if plan is not None:
                routed = _unsorted_switch(plan, x, indices)
                return self._combine(routed, weights, shared, residual)
        routed = fused_call(self, x, indices)
        return self._combine(routed, weights, shared, residual)

    return patched


def _fused_verify_switch(switch_mlp, x, indices):
    """Keep unsorted routing for the verifier's per-position reductions."""
    batch, length, width = x.shape
    top_k = indices.shape[-1]
    routed = moe_verify_gather.fused_switch(switch_mlp, x, indices)
    if routed is not None:
        return routed
    flat_x = mx.expand_dims(x.reshape(batch * length, width), (-2, -3))
    flat_indices = indices.reshape(batch * length, top_k)
    gate_up = switch_mlp.gate_up_proj(flat_x, flat_indices, sorted_indices=False)
    gate, up = mx.split(gate_up, 2, axis=-1)
    out = switch_mlp.down_proj(
        switch_mlp.activation(up, gate), flat_indices, sorted_indices=False
    )
    return out.squeeze(-2).reshape(batch, length, top_k, -1)


def _ensure_vlm_verify_patch() -> None:
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    original = Qwen3_5BatchInvariantForward._switch_glu
    if getattr(original, "_omlx_gate_up_fused_verify", False):
        return

    @wraps(original)
    def fused_verify(self, switch_mlp, x, indices):
        if x.ndim == 3 and getattr(switch_mlp, "gate_up_proj", None) is not None:
            return _fused_verify_switch(switch_mlp, x, indices)
        return original(self, switch_mlp, x, indices)

    fused_verify._omlx_gate_up_fused_verify = True
    Qwen3_5BatchInvariantForward._switch_glu = fused_verify


def _ensure_call_patch() -> None:
    global _CALL_PATCHED
    for cls, wrap in (
        (SwitchGLU, _make_patched_call),
        (VLMSwitchGLU, _make_vlm_patched_call),
    ):
        if getattr(cls, "_omlx_gate_up_fused_call", False):
            continue
        original = cls.__call__
        cls.__call__ = wrap(original)
        cls._omlx_gate_up_fused_call = True
        cls._omlx_gate_up_original_call = original
    _CALL_PATCHED = True


def apply_qwen35_moe_gate_up_fusion(model: Any) -> int:
    """Fuse gate+up expert projections on a supported loaded MoE model.

    Returns the number of fused ``SwitchGLU`` instances (0 when the model
    family is unsupported or there is nothing to fuse).
    """
    if not _is_supported_family(model):
        return 0
    # Subclasses opt in when their forward defers to the patched SwitchGLU
    # forward once gate_up_proj exists.
    targets = [
        m
        for _, m in model.named_modules()
        if (
            type(m) in (SwitchGLU, VLMSwitchGLU)
            or getattr(type(m), "_omlx_gate_up_fusion", False)
        )
        and _can_fuse(m)
    ]
    if not targets:
        return 0
    # Read the weight formats with the model rather than in its first verify
    # block. A registry that cannot load leaves the model unfused. The engine
    # applies its patches under a guard that logs at debug level.
    try:
        formats.load()
    except Exception:
        logger.exception("quant formats did not load; MoE gate+up fusion not applied")
        raise
    _ensure_call_patch()
    _ensure_vlm_verify_patch()
    for switch_mlp in targets:
        _fuse_one(switch_mlp)
        # The freed gate/up buffers land in the MLX buffer pool, which the
        # server pins to total RAM (#300), so nothing releases them during
        # load and the transient grows by the whole routed gate/up set,
        # ~2/3 of the expert bytes (#2304). Drain per fused layer to bound
        # the transient to a single layer's worth.
        _sync_and_clear_cache()
    logger.info("MoE gate+up fusion applied: %d layers", len(targets))
    return len(targets)


__all__ = ["apply_qwen35_moe_gate_up_fusion"]
