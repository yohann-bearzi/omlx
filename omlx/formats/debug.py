# SPDX-License-Identifier: Apache-2.0
"""Debug, inspect, and profile tools for oMLX quant formats.

This module is strictly opt-in (Rule 28: never imported on hot paths), reads
zero environment variables (Rule 30), and holds zero mutable module-global
caches (Rule 14).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import statistics
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .. import formats
from ..patches import m5_gather_qmm_nax as nax
from . import (
    Codes,
    Dot,
    QuantFormat,
    Table,
    Tile,
    affine,
    kernels,
    linear,
    testing,
    tile_dense,
    tile_gather,
    tile_wide,
    transforms,
    tuning,
    verify_skeleton,
)
from .linear import FormatEmbedding, FormatLinear, FormatSwitchLinear
from .tuning import DenseTilePolicy, MmaTilePolicy, MsgTilePolicy

DEFAULT_EXPLAIN_ROWS: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    8,
    16,
    32,
    64,
    128,
    512,
    2048,
)

DUMP_ROUTES: tuple[str, ...] = (
    "dot",
    "dot_shared",
    "steel",
    "nax",
    "wide",
    "mma",
    "msg",
    "switch_dot",
    "switch_tile_steel",
    "switch_tile_nax",
)

_DTYPE_ALIASES: dict[str, mx.Dtype] = {
    "bf16": mx.bfloat16,
    "bfloat16": mx.bfloat16,
    "mlx.core.bfloat16": mx.bfloat16,
    "fp16": mx.float16,
    "float16": mx.float16,
    "mlx.core.float16": mx.float16,
    "fp32": mx.float32,
    "float32": mx.float32,
    "mlx.core.float32": mx.float32,
}

_DTYPE_LABELS: dict[mx.Dtype, str] = {
    mx.bfloat16: "bf16",
    mx.float16: "fp16",
    mx.float32: "fp32",
}

_CHIP_PARSE_RE = re.compile(r"^applegpu_g(\d+)([a-z]?)(?:_c(\d+))?$")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _dtype_label(dtype: mx.Dtype) -> str:
    return _DTYPE_LABELS.get(dtype, str(dtype).removeprefix("mlx.core."))


def parse_dtype(value: str | mx.Dtype) -> mx.Dtype:
    """Parse a floating activation dtype string or ``mx.Dtype``."""
    if isinstance(value, mx.Dtype):
        if value in _DTYPE_LABELS:
            return value
        raise ValueError(
            f"unsupported activation dtype {value!r}; expected bf16, fp16, or fp32"
        )
    if not isinstance(value, str):
        raise TypeError(
            f"dtype must be a string or mx.Dtype, not {type(value).__name__}"
        )
    key = value.strip().lower()
    dtype = _DTYPE_ALIASES.get(key)
    if dtype is None:
        raise ValueError(
            f"unknown activation dtype {value!r}; expected bf16, fp16, or fp32"
        )
    return dtype


def _check_positive_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive int, not {value!r}")
    return value


def _load_plugin_format(plugin: str) -> QuantFormat:
    if not isinstance(plugin, str) or ":" not in plugin:
        raise ValueError(f"plugin must have 'pkg.mod:ATTR' syntax, not {plugin!r}")
    mod_name, attr_path = plugin.split(":", 1)
    if (
        not mod_name
        or not attr_path
        or not all(part.isidentifier() for part in mod_name.split("."))
        or not all(part.isidentifier() for part in attr_path.split("."))
    ):
        raise ValueError(f"plugin must have 'pkg.mod:ATTR' syntax, not {plugin!r}")
    module = importlib.import_module(mod_name)
    obj: Any = module
    for part in attr_path.split("."):
        if not hasattr(obj, part):
            raise ValueError(f"plugin {plugin!r}: {part!r} not found on {mod_name!r}")
        obj = getattr(obj, part)
    if not isinstance(obj, QuantFormat):
        raise TypeError(
            f"plugin {plugin!r} resolved to {type(obj).__name__}, not QuantFormat"
        )
    return obj


def load_format(
    mode_or_fmt: str | QuantFormat, *, plugin: str | None = None
) -> QuantFormat:
    """Resolve a ``QuantFormat`` from a mode name, instance, or plugin path.

    Loading via ``plugin='pkg.mod:ATTR'`` never mutates ``formats._FORMATS``.
    """
    if plugin is not None:
        fmt = _load_plugin_format(plugin)
        if isinstance(mode_or_fmt, str) and mode_or_fmt and fmt.mode != mode_or_fmt:
            raise ValueError(
                f"plugin {plugin!r} provides mode {fmt.mode!r}, "
                f"not requested mode {mode_or_fmt!r}"
            )
        return fmt
    if isinstance(mode_or_fmt, QuantFormat):
        return mode_or_fmt
    if not isinstance(mode_or_fmt, str) or not mode_or_fmt.strip():
        raise ValueError(
            f"mode_or_fmt must be a mode name or QuantFormat, not {mode_or_fmt!r}"
        )
    fmt = formats.get(mode_or_fmt)
    if fmt is None:
        raise ValueError(f"unknown quant format mode {mode_or_fmt!r}")
    return fmt


def _make_synthetic_tensors(
    fmt: QuantFormat,
    layout: dict[str, tuple[tuple[int, ...], mx.Dtype | None]],
    *,
    bits: int,
    group_size: int,
    k: int,
    n: int,
    dtype: mx.Dtype,
    experts: int | None,
    seed: int,
    realistic_weights: bool,
) -> dict[str, mx.array]:
    shared = set(getattr(fmt, "shared_tensors", ()))
    prefix = (experts,) if experts is not None else ()
    if realistic_weights and fmt.mode == "affine":
        w_shape = (*prefix, n, k)
        w = (mx.random.normal(w_shape, key=mx.random.key(seed)) * 0.25).astype(dtype)
        q, scales, biases = mx.quantize(
            w, group_size=group_size, bits=bits, mode="affine"
        )
        return {"weight": q, "scales": scales, "biases": biases}
    if realistic_weights and fmt.mode in ("oqh_affine", "oqh"):
        signs_shape = layout["oqh_signs"][0]
        block = signs_shape[-1]
        key_w, key_s = mx.random.split(mx.random.key(seed))
        signs_bits = mx.random.randint(0, 2, shape=signs_shape, key=key_s)
        signs = mx.where(signs_bits == 0, -1.0, 1.0).astype(dtype)
        w_plain = (mx.random.normal((*prefix, n, k), key=key_w) * 0.25).astype(
            mx.float32
        )
        w_rot = transforms._rotate_mlx(w_plain, signs, block).astype(dtype)
        if fmt.mode == "oqh_affine":
            q, scales, biases = mx.quantize(
                w_rot, group_size=group_size, bits=bits, mode="affine"
            )
            return {
                "weight": q,
                "oqh_scales": scales,
                "oqh_biases": biases,
                "oqh_signs": signs,
            }
        # Random uint32 words hold uniform codes in [0, 2**bits - 1] with
        # mean = center; scale each row by a small positive rscale.
        q_shape = (*prefix, *layout["weight"][0])
        r_shape = (*prefix, *layout["rscale"][0])
        k_lo, k_hi, k_rs = mx.random.split(key_w, 3)
        lo = mx.random.randint(0, 1 << 16, shape=q_shape, key=k_lo, dtype=mx.uint32)
        hi = mx.random.randint(0, 1 << 16, shape=q_shape, key=k_hi, dtype=mx.uint32)
        q_packed = lo | (hi << 16)
        rscale = mx.random.uniform(0.02, 0.12, shape=r_shape, key=k_rs).astype(dtype)
        return {"weight": q_packed, "rscale": rscale, "oqh_signs": signs}

    tensors: dict[str, mx.array] = {}
    key = mx.random.key(seed)
    for idx, (name, (shape, spec_dt)) in enumerate(layout.items()):
        full_shape = shape if name in shared else (*prefix, *shape)
        arr_dt = dtype if spec_dt is None else spec_dt
        if name == "oqh_signs":
            if realistic_weights:
                sub = mx.random.key(seed + 1000 + idx)
                bits_arr = mx.random.randint(0, 2, shape=full_shape, key=sub)
                tensors[name] = mx.where(bits_arr == 0, -1.0, 1.0).astype(arr_dt)
            else:
                tensors[name] = mx.ones(full_shape, dtype=arr_dt)
        elif realistic_weights:
            sub = mx.random.key(seed + 100 + idx)
            if mx.issubdtype(arr_dt, mx.floating):
                if "scale" in name:
                    val = mx.random.uniform(
                        0.02, 0.15, shape=full_shape, key=sub
                    ).astype(arr_dt)
                else:
                    val = (mx.random.normal(full_shape, key=sub) * 0.1).astype(arr_dt)
                tensors[name] = val
            elif arr_dt == mx.uint32:
                lo = mx.random.randint(
                    0, 1 << 16, shape=full_shape, key=sub, dtype=mx.uint32
                )
                hi = mx.random.randint(
                    0,
                    1 << 16,
                    shape=full_shape,
                    key=mx.random.key(seed + 200 + idx),
                    dtype=mx.uint32,
                )
                tensors[name] = lo | (hi << 16)
            elif mx.issubdtype(arr_dt, mx.unsignedinteger):
                tensors[name] = mx.random.randint(
                    0, 16, shape=full_shape, key=sub
                ).astype(arr_dt)
            else:
                tensors[name] = mx.random.randint(
                    -8, 8, shape=full_shape, key=sub
                ).astype(arr_dt)
        else:
            if name in ("scales", "oqh_scales", "rscale") and mx.issubdtype(
                arr_dt, mx.floating
            ):
                tensors[name] = mx.ones(full_shape, dtype=arr_dt)
            else:
                tensors[name] = mx.zeros(full_shape, dtype=arr_dt)
    _ = key
    return tensors


def make_layer(
    mode_or_fmt: str | QuantFormat,
    *,
    bits: int,
    group_size: int,
    k: int,
    n: int,
    dtype: str | mx.Dtype = mx.bfloat16,
    experts: int | None = None,
    bias: bool = False,
    seed: int = 0,
    realistic_weights: bool = False,
    plugin: str | None = None,
) -> FormatLinear | FormatSwitchLinear:
    """Build a synthetic ``FormatLinear`` or ``FormatSwitchLinear`` safely.

    Fails closed if a checkpoint load is active on this thread/context
    (``formats._LOADING.get() is not None``) and scopes
    ``testing.verified("qwen4_exp")`` strictly around layer instantiation.
    """
    if formats._LOADING.get() is not None:
        raise RuntimeError(
            "debug.make_layer cannot construct synthetic layers while a "
            "checkpoint load is active (formats._LOADING is set)"
        )
    _check_positive_int("bits", bits)
    _check_positive_int("group_size", group_size)
    _check_positive_int("k", k)
    _check_positive_int("n", n)
    if experts is not None:
        _check_positive_int("experts", experts)
    if type(seed) is not int:
        raise ValueError(f"seed must be an int, not {seed!r}")
    if not isinstance(bias, bool):
        raise TypeError(f"bias must be a bool, not {type(bias).__name__}")
    if not isinstance(realistic_weights, bool):
        raise TypeError(
            f"realistic_weights must be a bool, not {type(realistic_weights).__name__}"
        )
    dt = parse_dtype(dtype)
    fmt = load_format(mode_or_fmt, plugin=plugin)
    layout = fmt.layout(k, n, bits, group_size)
    if layout is None:
        raise ValueError(
            f"quant format {fmt.mode!r} has no layout for "
            f"(k={k}, n={n}, bits={bits}, group_size={group_size})"
        )
    formats.check_layout(fmt, n, k, layout)
    tensors = _make_synthetic_tensors(
        fmt,
        layout,
        bits=bits,
        group_size=group_size,
        k=k,
        n=n,
        dtype=dt,
        experts=experts,
        seed=seed,
        realistic_weights=realistic_weights,
    )
    bias_arr: mx.array | None = None
    if bias:
        b_shape = (experts, n) if experts is not None else (n,)
        if realistic_weights:
            bias_arr = (
                mx.random.normal(b_shape, key=mx.random.key(seed + 9999)) * 0.05
            ).astype(dt)
        else:
            bias_arr = mx.zeros(b_shape, dtype=dt)
    cls = FormatSwitchLinear if experts is not None else FormatLinear
    with testing.verified("qwen4_exp"):
        return cls(fmt, bits, group_size, k, tensors, bias=bias_arr)


# ---------------------------------------------------------------------------
# Chip & target helpers (read-only)
# ---------------------------------------------------------------------------


def _resolve_chip(
    chip: str | None, tensor_units: bool | None
) -> tuple[str, str, int, int, bool]:
    """Return ``(chip_id, architecture, gpu_cores, gpu_generation, tensor_units)``."""
    if chip is None:
        chip_id = tuning.current_chip()
        match = _CHIP_PARSE_RE.match(chip_id.lower())
        if match is not None:
            gen = int(match.group(1))
            suffix = match.group(2)
            arch = f"applegpu_g{gen}{suffix}"
            cores = int(match.group(3)) if match.group(3) else kernels.gpu_cores()
        else:
            arch = chip_id.rsplit("_c", 1)[0] if "_c" in chip_id else chip_id
            gen = kernels.gpu_generation()
            cores = kernels.gpu_cores()
        tu = linear._nax_host() if tensor_units is None else bool(tensor_units)
        return chip_id, arch, cores, gen, tu

    if not isinstance(chip, str) or not chip.strip():
        raise ValueError(f"chip must be a non-empty string, not {chip!r}")
    chip_id = chip.strip()
    match = _CHIP_PARSE_RE.match(chip_id.lower())
    if match is not None:
        gen = int(match.group(1))
        suffix = match.group(2)
        arch = f"applegpu_g{gen}{suffix}"
        cores = int(match.group(3)) if match.group(3) else 0
        default_tu = gen >= (18 if suffix == "p" else 17)
    else:
        arch = chip_id.rsplit("_c", 1)[0] if "_c" in chip_id else chip_id
        gen = 0
        cores = 0
        if "_c" in chip_id:
            tail = chip_id.rsplit("_c", 1)[1]
            if tail.isdigit():
                cores = int(tail)
        default_tu = False
    tu = default_tu if tensor_units is None else bool(tensor_units)
    return chip_id, arch, cores, gen, tu


def _group_for_chip(
    rows: int, vpt: int, n: int, gpu_generation: int, gpu_cores: int
) -> int:
    """Pure mirror of ``linear._group`` parameterized by target chip."""
    if vpt < 1:
        return 1
    limit = kernels.ROW_VALUES_BUDGET // vpt
    groups = [g for g in (4, 3, 2) if g <= limit and rows % g == 0]
    if not groups:
        return 1
    table = linear.OCCUPANCY.get(gpu_generation, linear._UNMEASURED)
    occ = table.get(vpt, max(table.values()))
    floor = occ * gpu_cores
    tiles = -(-n // linear._SHARED_TILE)
    return next((g for g in groups if rows // g * tiles >= floor), 1)


def _is_stackable_pure(
    fmt: QuantFormat, k: int, n: int, bits: int, group_size: int
) -> bool:
    """Pure mirror of ``formats.stackable`` that never touches ``_DOT_DEFECTS``."""
    if n <= 0:
        return False
    shared = set(getattr(fmt, "shared_tensors", ()))
    try:
        one = fmt.layout(k, n, bits, group_size)
        expected_keys = {fmt.codes_name, *fmt.tensors, *shared}
        if not isinstance(one, dict) or one.keys() != expected_keys:
            return False
        two = fmt.layout(k, 2 * n, bits, group_size)
        return (
            isinstance(two, dict)
            and two.keys() == one.keys()
            and all(
                (
                    tuple(two[name][0]) == tuple(shape)
                    if name in shared
                    else (
                        len(shape) >= 1
                        and shape[0] == n
                        and tuple(two[name][0]) == (2 * n, *shape[1:])
                    )
                )
                and two[name][1] == dt
                for name, (shape, dt) in one.items()
            )
        )
    except Exception:  # noqa: BLE001
        return False


def _layer_type_name(layer: Any) -> str:
    if isinstance(layer, FormatSwitchLinear):
        return "FormatSwitchLinear"
    if isinstance(layer, FormatEmbedding):
        return "FormatEmbedding"
    if isinstance(layer, FormatLinear):
        return "FormatLinear"
    return type(layer).__name__


# ---------------------------------------------------------------------------
# Explain dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaceTrace:
    kind: str
    available: bool
    key: str | None
    type_name: str | None
    exact_for_dtype: bool
    dtypes: tuple[str, ...]
    tensors: tuple[str, ...]
    vpt: int | None
    table_profile: str | None
    table_bytes: int
    face_digest: str | None
    decline_reason: str | None
    defective: bool

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class InputTransformTrace:
    declared: bool
    active: bool
    kind: str | None
    version: int | None
    block: int | None
    tensors: tuple[str, ...]
    producer_spec: dict[str, Any] | None
    native_fast_disabled: bool

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class TuningCandidateTrace:
    route: str
    face_id: str
    rows_bucket: str
    shape_scope: str
    match_rank: tuple[int, int]
    params_repr: str
    accepted: bool
    selected: bool
    rejected_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class RowRouteTrace:
    rows: int
    schedule_bucket: tuple[int, int | None] | None
    selected_route: str
    selected_backend: str
    kernel_name: str | None
    policy_repr: str | None
    effective_policy_repr: str | None
    split_k: int | None
    shared_group: int | None
    unrolled: bool | None
    native_fast_eligible: bool
    decision_source: str
    tuning_probes: tuple[TuningCandidateTrace, ...]
    decline_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class ExplainResult:
    layer_type: str
    mode: str
    origin: str | None
    contract: int
    bits: int
    group_size: int
    k: int
    n: int
    experts: int | None
    has_bias: bool
    is_stock: bool
    stackable: bool
    dtype: str
    chip: str
    architecture: str
    tensor_units: bool
    gpu_cores: int
    gpu_generation: int
    table_mode: str
    codes_name: str
    tensors: tuple[str, ...]
    shared_tensors: tuple[str, ...]
    tensor_specs: tuple[dict[str, Any], ...]
    faces: tuple[FaceTrace, ...]
    input_transform: InputTransformTrace
    schedule_summary: tuple[dict[str, Any], ...]
    row_traces: tuple[RowRouteTrace, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


# ---------------------------------------------------------------------------
# Pure face / transform / policy diagnosis helpers
# ---------------------------------------------------------------------------


def _diagnose_builtin_face_decline(
    fmt: QuantFormat, layer: Any, k: int, n: int, name: str
) -> str:
    mode = fmt.mode
    if mode == "affine":
        bits, gs = layer.bits, layer.group_size
        w = layer.get("weight")
        s = layer.get("scales")
        b = layer.get("biases")
        if name == "dot":
            if bits not in affine._BITS:
                return f"affine dot: unsupported bits={bits}"
            if gs not in affine._GROUP_SIZES:
                return f"affine dot: unsupported group_size={gs}"
            if not isinstance(s, mx.array) or not isinstance(b, mx.array):
                return "affine dot: missing scales or biases array"
            if s.dtype not in (mx.bfloat16, mx.float16):
                return f"affine dot: scales.dtype={s.dtype} not bf16/fp16"
            if b.dtype != s.dtype:
                return f"affine dot: biases.dtype={b.dtype} != scales.dtype={s.dtype}"
            if not isinstance(w, mx.array) or w.ndim not in (2, 3):
                return "affine dot: weight must be 2-D or 3-D"
            if w.shape[-2:] != (n, k * bits // 32):
                return (
                    f"affine dot: weight shape {w.shape[-2:]} != "
                    f"({n}, {k * bits // 32})"
                )
            if k % gs:
                return f"affine dot: k={k} not divisible by group_size={gs}"
            if n * k // gs < 8:
                return f"affine dot: scales.size={n * k // gs} < 8 (MLX const bind)"
            if w.ndim == 2 and k in (64, 128) and bits in (2, 4, 8):
                return (
                    f"affine dot: qmv_quad shape (2-D k={k}, bits={bits}); "
                    "MLX uses qmv_quad so no dot face is provided"
                )
        elif name == "tile":
            if bits not in affine._TILE_FACE_BITS:
                return f"affine tile: unsupported bits={bits}"
            if gs not in affine._GROUP_SIZES:
                return f"affine tile: unsupported group_size={gs}"
            if not all(isinstance(x, mx.array) for x in (w, s, b)):
                return "affine tile: missing weight, scales, or biases"
            if k % gs:
                return f"affine tile: k={k} not divisible by group_size={gs}"
            if s.dtype not in affine._METAL_TYPES:
                return f"affine tile: scales.dtype={s.dtype} not bf16/fp16"
            if b.dtype != s.dtype:
                return f"affine tile: biases.dtype={b.dtype} != scales.dtype={s.dtype}"
            if s.size < 8:
                return f"affine tile: scales.size={s.size} < 8"
        elif name == "codes":
            if bits not in affine._CODE_BITS:
                return f"affine codes: unsupported bits={bits} (allowed 2,3,4,5,6)"
            if gs not in affine._GROUP_SIZES:
                return f"affine codes: unsupported group_size={gs}"
            if k <= 0 or k % gs:
                return f"affine codes: k={k} not divisible by group_size={gs}"
            if (
                isinstance(w, mx.array)
                and isinstance(s, mx.array)
                and min(w.size, s.size) < 8
            ):
                return "affine codes: tensor size < 8"
    return f"format {mode!r}.{name}() returned None for (k={k}, n={n})"


def _probe_faces(
    layer: Any, dtype: mx.Dtype
) -> tuple[dict[str, Any], tuple[FaceTrace, ...]]:
    fmt = layer.fmt
    k, n = layer.input_dims, layer.output_dims
    objs: dict[str, Any] = {}
    traces: list[FaceTrace] = []
    for kind_name, kind_cls in (("dot", Dot), ("tile", Tile), ("codes", Codes)):
        try:
            face = formats._checked_face(fmt, layer, k, n, kind_cls, kind_name)
        except Exception as exc:  # noqa: BLE001
            objs[kind_name] = None
            traces.append(
                FaceTrace(
                    kind=kind_name,
                    available=False,
                    key=None,
                    type_name=None,
                    exact_for_dtype=False,
                    dtypes=(),
                    tensors=(),
                    vpt=None,
                    table_profile=None,
                    table_bytes=0,
                    face_digest=None,
                    decline_reason=f"defective face: {type(exc).__name__}: {exc}",
                    defective=True,
                )
            )
            continue
        objs[kind_name] = face
        if face is None:
            traces.append(
                FaceTrace(
                    kind=kind_name,
                    available=False,
                    key=None,
                    type_name=None,
                    exact_for_dtype=False,
                    dtypes=(),
                    tensors=(),
                    vpt=None,
                    table_profile=None,
                    table_bytes=0,
                    face_digest=None,
                    decline_reason=_diagnose_builtin_face_decline(
                        fmt, layer, k, n, kind_name
                    ),
                    defective=False,
                )
            )
        else:
            traces.append(
                FaceTrace(
                    kind=kind_name,
                    available=True,
                    key=face.key,
                    type_name=face.type,
                    exact_for_dtype=bool(face.exact_for(dtype)),
                    dtypes=tuple(_dtype_label(dt) for dt in face.dtypes),
                    tensors=tuple(face.tensors),
                    vpt=face.vpt if isinstance(face, Dot) else None,
                    table_profile=tuning.table_profile(face),
                    table_bytes=kernels.table_bytes(face),
                    face_digest=tuning.face_digest(face),
                    decline_reason=None,
                    defective=False,
                )
            )
    return objs, tuple(traces)


def _probe_input_transform(layer: Any) -> InputTransformTrace:
    fmt = layer.fmt
    declared = formats._declares_input_transform(fmt)
    if not declared:
        return InputTransformTrace(
            declared=False,
            active=False,
            kind=None,
            version=None,
            block=None,
            tensors=(),
            producer_spec=None,
            native_fast_disabled=False,
        )
    # Use _bound_transform directly so layer.__dict__ is never mutated (Rule 14).
    # Malformed InputTransform propagates TypeError / ValueError (fail closed).
    bound = formats._bound_transform(fmt, layer, transforms)
    if bound is None:
        return InputTransformTrace(
            declared=True,
            active=False,
            kind=None,
            version=None,
            block=None,
            tensors=(),
            producer_spec=None,
            native_fast_disabled=False,
        )
    tx, _ = bound
    prod_dict: dict[str, Any] | None = None
    if tx.producer_spec is not None:
        spec = tx.producer_spec(layer)
        if spec is not None:
            prod_dict = {
                "rmsnorm": spec.rmsnorm is not None,
                "rmsnorm_block": int(spec.rmsnorm_block),
                "rmsnorm_max_width": int(spec.rmsnorm_max_width),
                "gated_norm": spec.gated_norm is not None,
                "gated_norm_head_dim": int(spec.gated_norm_head_dim),
                "act_pair": spec.act_pair is not None,
            }
    return InputTransformTrace(
        declared=True,
        active=True,
        kind=tx.kind,
        version=tx.version,
        block=tx.block,
        tensors=tuple(tx.tensors),
        producer_spec=prod_dict,
        native_fast_disabled=True,
    )


def _policy_illegality_reason(
    params: Any,
    face: Tile,
    k: int,
    n: int,
    m: int,
    chip: str,
    dtype: mx.Dtype,
) -> str:
    if isinstance(params, DenseTilePolicy):
        if params.backend == "wide":
            if k % 8 != 0:
                return f"is_legal failed: wide requires k % 8 == 0 (k={k})"
            return "is_legal failed: wide table_bytes > 32768"
        if params.backend == "dot":
            return f"is_legal failed: dot requires m <= 16 (m={m})"
        if params.backend == "nax":
            match = tuning.schema._CHIP_GEN_RE.match((chip or "").lower())
            if match is None:
                return f"is_legal failed: nax unrecognized chip {chip!r}"
            gen = int(match.group(1))
            req = 18 if match.group(2) == "p" else 17
            if gen < req:
                return f"is_legal failed: nax requires GPU gen >= {req} (chip={chip})"
            if params.wm * params.wn * 32 < 2 * params.bn:
                return (
                    f"is_legal failed: nax threads={params.wm * params.wn * 32} "
                    f"< 2 * bn={2 * params.bn}"
                )
        if params.align_n and n % params.bn:
            return f"is_legal failed: align_n=True but n={n} % bn={params.bn} != 0"
        if params.bk and k % (params.split_k * params.bk):
            return (
                f"is_legal failed: k={k} not divisible by "
                f"split_k*bk={params.split_k * params.bk}"
            )
        return f"is_legal failed for {params!r} on (k={k}, n={n}, m={m}, chip={chip})"
    if isinstance(params, MmaTilePolicy):
        if k % (params.bk * params.nsg):
            return (
                f"is_legal failed: mma requires k % (bk*nsg) == 0 "
                f"(k={k}, bk*nsg={params.bk * params.nsg})"
            )
        if n % params.bn:
            return f"is_legal failed: mma requires n % bn == 0 (n={n}, bn={params.bn})"
        return f"is_legal failed for {params!r} (threadgroup staging or constraints)"
    if isinstance(params, MsgTilePolicy):
        if not 1 <= int(m) <= 8:
            return f"is_legal failed: msg requires 1 <= m <= 8 (m={m})"
        if int(n) < 4 * params.nsg or int(n) % (4 * params.nsg):
            return (
                f"is_legal failed: msg requires n % (4*nsg) == 0 "
                f"(n={n}, 4*nsg={4 * params.nsg})"
            )
        return f"is_legal failed for {params!r}"
    return f"is_legal failed for {params!r}"


def _trace_route_candidates(
    route_name: str,
    table_face: Tile,
    actual_face: Tile,
    *,
    layer: Any,
    dtype: mx.Dtype,
    sample: int,
    bound: int,
    chip: str,
    tensor_units: bool,
    dot_ok: bool,
) -> tuple[
    list[TuningCandidateTrace],
    tuple[Any, tuple[int, int]] | None,
]:
    """Mirror ``tuning._lookup_best`` + ``tile_dense._resolve_cell`` filters."""
    k, n = layer.input_dims, layer.output_dims
    table = tuning._index_for_chip(chip)
    skeleton = tuning.skeleton_digest(route_name)
    dtype_name = tuning._dtype_name(dtype)
    entries = (
        table.by_key.get((route_name, dtype_name, layer.bits, layer.group_size), ())
        if table.by_key
        else table.entries
    )
    query = tuning._Query(
        route=route_name,
        face_key=table_face.key,
        profile=tuning.table_profile(table_face),
        dtype=dtype_name,
        bits=layer.bits,
        group=layer.group_size,
        n=n,
        k=k,
        rows=sample,
        rows_bucket=tuning.rows_bucket(route_name, sample),
        n_bucket=tuning._n_bucket(n),
        k_bucket=tuning._k_bucket(k),
    )
    digest: str | None = None
    best_rank = (0, 0)
    best_params = None
    best_idx = -1
    raw_records: list[dict[str, Any]] = []

    for entry in entries:
        rank = tuning._matches(entry, query)
        if rank[0] == 0:
            continue
        ek = entry.key
        scope = (
            f"exact({ek['n']}x{ek['k']})"
            if "n" in ek and "k" in ek
            else f"bucket({ek['n_bucket']},{ek['k_bucket']})"
        )
        reasons: list[str] = []
        if skeleton is None or entry.skeleton_sha256 != skeleton:
            reasons.append(
                f"skeleton_sha256 stale (entry={entry.skeleton_sha256}, "
                f"current={skeleton})"
            )
        if ek.get("face_id") == table_face.key:
            if digest is None:
                digest = tuning.face_digest(table_face)
            if entry.face_sha256 != digest:
                reasons.append(
                    f"face_sha256 stale (entry={entry.face_sha256}, current={digest})"
                )
        if not entry.params.is_legal(table_face, k, n, sample, chip, dtype=dtype):
            reasons.append(
                _policy_illegality_reason(
                    entry.params, table_face, k, n, sample, chip, dtype
                )
            )
        # Post-lookup filters from tile_dense._resolve_cell.
        if not reasons:
            pol = entry.params
            if route_name == "dense_tile" and isinstance(pol, DenseTilePolicy):
                if pol.backend == "nax" and (
                    not tensor_units or pol.wm * pol.wn * 32 < 2 * pol.bn
                ):
                    reasons.append(
                        "post-filter: nax requires tensor_units and " "wm*wn*32 >= 2*bn"
                    )
                elif pol.backend == "mlx" and layer.mode != "affine":
                    reasons.append("post-filter: mlx backend is affine-only")
                elif pol.backend == "wide" and not tile_dense._wide_legal(
                    actual_face, k, chip
                ):
                    reasons.append("post-filter: wide route not legal for face/shape")
                elif pol.backend == "dot" and (
                    not dot_ok or bound > tuning.schema._DOT_MAX_ROWS
                ):
                    reasons.append(
                        f"post-filter: dot unavailable or bound={bound} > 16"
                    )
            elif route_name in ("mma_tile", "msg_tile") and not pol.is_legal(
                actual_face, k, n, bound, chip, dtype
            ):
                reasons.append(
                    _policy_illegality_reason(
                        pol, actual_face, k, n, bound, chip, dtype
                    )
                )

        accepted = not reasons
        rec = {
            "route": route_name,
            "face_id": ek["face_id"],
            "rows_bucket": ek["rows_bucket"],
            "shape_scope": scope,
            "match_rank": rank,
            "params": entry.params,
            "accepted": accepted,
            "reasons": reasons,
        }
        raw_records.append(rec)
        # Track the exact winner of tuning._lookup_best (note: tuning._lookup_best
        # checks validity against table_face at rows=sample, then _resolve_cell
        # checks post-filters; we match that exact ordering).
        lookup_valid = (
            skeleton is not None
            and entry.skeleton_sha256 == skeleton
            and (ek.get("face_id") != table_face.key or entry.face_sha256 == digest)
            and entry.params.is_legal(table_face, k, n, sample, chip, dtype=dtype)
        )
        if lookup_valid and rank > best_rank:
            best_rank = rank
            best_params = entry.params
            best_idx = len(raw_records) - 1

    # Confirm whether the lookup winner passed post-filters.
    winner: tuple[Any, tuple[int, int]] | None = None
    if best_idx >= 0 and raw_records[best_idx]["accepted"]:
        winner = (best_params, best_rank)

    traces: list[TuningCandidateTrace] = []
    for idx, rec in enumerate(raw_records):
        reasons = list(rec["reasons"])
        is_route_winner = idx == best_idx and winner is not None
        if rec["accepted"] and not is_route_winner:
            reasons.append("outranked by higher specificity/span candidate")
        traces.append(
            TuningCandidateTrace(
                route=rec["route"],
                face_id=rec["face_id"],
                rows_bucket=rec["rows_bucket"],
                shape_scope=rec["shape_scope"],
                match_rank=rec["match_rank"],
                params_repr=repr(rec["params"]),
                accepted=rec["accepted"],
                selected=is_route_winner,
                rejected_reasons=tuple(reasons),
            )
        )
    return traces, winner


def _trace_resolve_cell(
    layer: Any,
    face: Tile,
    dtype: mx.Dtype,
    lo: int,
    hi: int | None,
    sample: int,
    *,
    tensor_units: bool,
    chip: str,
    dot_ok: bool,
) -> tuple[tile_dense.DenseRoute, str, tuple[TuningCandidateTrace, ...], list[str]]:
    """Pure mirror of ``tile_dense._resolve_cell`` returning full trace metadata."""
    k, n = layer.input_dims, layer.output_dims
    bound = sample if hi is None else hi
    table_face = (
        affine.tile_face(layer.bits, layer.group_size, dtype)
        if layer.mode == "oqh_affine"
        else face
    )

    all_probes: list[TuningCandidateTrace] = []
    candidates: list[tuple[int, str, str, object, tuple[int, int]]] = []

    specs = (
        (2, "dense_tile", None, DenseTilePolicy),
        (1, "mma_tile", "mma", MmaTilePolicy),
        (0, "msg_tile", "msg", MsgTilePolicy),
    )
    for prio, route_name, fixed_backend, cls in specs:
        probes, winner = _trace_route_candidates(
            route_name,
            table_face,
            face,
            layer=layer,
            dtype=dtype,
            sample=sample,
            bound=bound,
            chip=chip,
            tensor_units=tensor_units,
            dot_ok=dot_ok,
        )
        all_probes.extend(probes)
        if winner is not None:
            pol, rank = winner
            if isinstance(pol, cls):
                backend = pol.backend if fixed_backend is None else fixed_backend
                candidates.append((prio, route_name, backend, pol, rank))

    declines: list[str] = []
    if candidates:
        win = max(candidates, key=lambda item: (item[4], item[0]))
        _, win_route, win_backend, win_pol, _ = win
        # Mark only the overall winning probe as selected=True.
        final_probes: list[TuningCandidateTrace] = []
        for p in all_probes:
            if p.selected and (p.route != win_route or p.params_repr != repr(win_pol)):
                final_probes.append(
                    TuningCandidateTrace(
                        route=p.route,
                        face_id=p.face_id,
                        rows_bucket=p.rows_bucket,
                        shape_scope=p.shape_scope,
                        match_rank=p.match_rank,
                        params_repr=p.params_repr,
                        accepted=p.accepted,
                        selected=False,
                        rejected_reasons=(
                            *p.rejected_reasons,
                            f"outranked by {win_route} ({win_backend})",
                        ),
                    )
                )
            else:
                final_probes.append(p)
        return (
            tile_dense.DenseRoute(hi, win_backend, win_pol),
            "tuning_table",
            tuple(final_probes),
            declines,
        )

    if tensor_units and 4 <= lo <= bound <= 16:
        default = tile_dense._default_nax_policy(layer.bits, sample, n, k)
        if default.sched == "register":
            if default.is_legal(face, k, n, bound, chip, dtype):
                return (
                    tile_dense.DenseRoute(hi, "nax", default),
                    "parity_default",
                    tuple(all_probes),
                    declines,
                )
            declines.append(
                f"short-row nax default {default!r} illegal: "
                + _policy_illegality_reason(default, face, k, n, bound, chip, dtype)
            )

    if hi is not None and hi <= 16:
        if layer.mode == "affine":
            return (
                tile_dense.DenseRoute(hi, "mlx", DenseTilePolicy("mlx")),
                "unmeasured_short_fallback",
                tuple(all_probes),
                declines,
            )
        if tile_dense._wide_legal(face, k, chip):
            return (
                tile_dense.DenseRoute(hi, "wide", DenseTilePolicy("wide")),
                "unmeasured_short_fallback",
                tuple(all_probes),
                declines,
            )
        declines.append(
            f"wide route illegal for (k={k}, chip={chip}, "
            f"table_bytes={kernels.table_bytes(face)})"
        )
        return (
            tile_dense.DenseRoute(hi, "dot", None),
            "dot_fallback",
            tuple(all_probes),
            declines,
        )

    default = tile_dense._default_policy(tensor_units, layer.bits, sample, n, k)
    if default.is_legal(face, k, n, bound, chip, dtype, tail=True):
        return (
            tile_dense.DenseRoute(hi, default.backend, default),
            "parity_default",
            tuple(all_probes),
            declines,
        )
    declines.append(
        f"parity default {default!r} illegal: "
        + _policy_illegality_reason(default, face, k, n, bound, chip, dtype)
    )
    if layer.mode == "affine":
        return (
            tile_dense.DenseRoute(hi, "mlx", DenseTilePolicy("mlx")),
            "parity_default",
            tuple(all_probes),
            declines,
        )
    return (
        tile_dense.DenseRoute(hi, "dot", None),
        "dot_fallback",
        tuple(all_probes),
        declines,
    )


def _build_traced_schedule(
    layer: Any,
    face: Tile,
    dtype: mx.Dtype,
    *,
    tensor_units: bool,
    chip: str,
    dot_ok: bool,
) -> tuple[
    list[
        tuple[
            int,
            int | None,
            tile_dense.DenseRoute,
            str,
            tuple[TuningCandidateTrace, ...],
            list[str],
        ]
    ],
    tuple[dict[str, Any], ...],
]:
    cuts = tuple(
        sorted(
            {
                1,
                16,
                *tuning.row_bounds("dense_tile", chip=chip),
                *tuning.row_bounds("mma_tile", chip=chip),
                *tuning.row_bounds("msg_tile", chip=chip),
            }
        )
    )
    cells: list[
        tuple[
            int,
            int | None,
            tile_dense.DenseRoute,
            str,
            tuple[TuningCandidateTrace, ...],
            list[str],
        ]
    ] = [
        (1, 1, tile_dense.DenseRoute(1, "dot", None), "parity_default", (), []),
    ]
    lo = 2
    for top in cuts:
        if top < 2:
            continue
        route, source, probes, declines = _trace_resolve_cell(
            layer,
            face,
            dtype,
            lo,
            top,
            top,
            tensor_units=tensor_units,
            chip=chip,
            dot_ok=dot_ok,
        )
        cells.append((lo, top, route, source, probes, declines))
        lo = top + 1
    tail_sample = max(4096, (cuts[-1] + 1) if cuts else 4096)
    tail_route, tail_src, tail_probes, tail_declines = _trace_resolve_cell(
        layer,
        face,
        dtype,
        lo,
        None,
        tail_sample,
        tensor_units=tensor_units,
        chip=chip,
        dot_ok=dot_ok,
    )
    cells.append((lo, None, tail_route, tail_src, tail_probes, tail_declines))

    # Coalesce adjacent identical (backend, policy) cells for schedule_summary.
    coalesced: list[tuple[int, int | None, tile_dense.DenseRoute]] = []
    for c_lo, c_hi, route, _, _, _ in cells:
        if coalesced:
            prev_lo, _, prev_route = coalesced[-1]
            if (
                prev_route.backend == route.backend
                and prev_route.policy == route.policy
            ):
                coalesced[-1] = (prev_lo, c_hi, route)
                continue
        coalesced.append((c_lo, c_hi, route))

    k, n = layer.input_dims, layer.output_dims
    summary: list[dict[str, Any]] = []
    for s_lo, s_hi, route in coalesced:
        eff_repr: str | None = None
        if isinstance(route.policy, DenseTilePolicy) and route.backend in (
            "steel",
            "nax",
        ):
            eff_repr = repr(tile_dense._effective_policy(route.policy, k, n))
        summary.append(
            {
                "lo": s_lo,
                "hi": s_hi,
                "backend": route.backend,
                "policy_repr": repr(route.policy) if route.policy is not None else None,
                "effective_policy_repr": eff_repr,
            }
        )
    return cells, tuple(summary)


def _explain_dense_row(
    layer: FormatLinear,
    r: int,
    *,
    dtype: mx.Dtype,
    architecture: str,
    tensor_units: bool,
    gpu_cores: int,
    gpu_generation: int,
    input_ndim: int,
    dot_face: Dot | None,
    dot_plan_ok: bool,
    dot_exact: bool,
    tile_face: Tile | None,
    tile_exact: bool,
    tx_active: bool,
    cells: list[tuple[Any, ...]] | None,
) -> RowRouteTrace:
    k, n = layer.input_dims, layer.output_dims
    is_stock = bool(getattr(layer, "_stock", False))
    declines: list[str] = []
    if dot_face is None:
        declines.append("dot face unavailable")
    elif not dot_plan_ok:
        if k < 8:
            declines.append(f"dot plan declined: k={k} < 8")
        elif n % 8 != 0 and dot_face.vpt < 1:
            declines.append(f"dot plan declined: n={n} % 8 != 0 and vpt < 1")
    elif not dot_exact:
        declines.append(f"dot face not exact for {_dtype_label(dtype)}")

    # Row 1: M3 C7 override (3-D only), else Dot / dot_shared, else stock mlx / decode.
    if r == 1:
        c7_rows = (
            linear._decode_mlx_rows(architecture, n, k) if is_stock else frozenset()
        )
        if 1 in c7_rows and input_ndim == 3 and dot_plan_ok:
            return RowRouteTrace(
                rows=1,
                schedule_bucket=(1, 1),
                selected_route="mlx",
                selected_backend="mlx",
                kernel_name=None,
                policy_repr=None,
                effective_policy_repr=None,
                split_k=None,
                shared_group=None,
                unrolled=None,
                native_fast_eligible=(
                    not tx_active
                    and dtype in (mx.float16, mx.bfloat16)
                    and tile_face is not None
                ),
                decision_source="m3_c7_override",
                tuning_probes=(),
                decline_reasons=(),
            )
        if 1 in c7_rows and input_ndim == 2:
            declines.append(
                "m3_c7_override applies only to 3-D (1, 1, K) inputs; "
                "2-D (1, K) uses dot"
            )
        if dot_exact:
            if n % 8 != 0:
                unrolled = kernels.unrolled_tile_ok(dot_face.vpt, linear._SHARED_RPS, 1)
                return RowRouteTrace(
                    rows=1,
                    schedule_bucket=(1, 1),
                    selected_route="dot_shared",
                    selected_backend="dot",
                    kernel_name=f"omlx_format_linear_rows_{dot_face.key}",
                    policy_repr=None,
                    effective_policy_repr=None,
                    split_k=None,
                    shared_group=1,
                    unrolled=unrolled,
                    native_fast_eligible=not tx_active,
                    decision_source="parity_default",
                    tuning_probes=(),
                    decline_reasons=tuple(declines),
                )
            return RowRouteTrace(
                rows=1,
                schedule_bucket=(1, 1),
                selected_route="dot",
                selected_backend="dot",
                kernel_name=f"omlx_format_linear_{dot_face.key}",
                policy_repr=None,
                effective_policy_repr=None,
                split_k=None,
                shared_group=1,
                unrolled=None,
                native_fast_eligible=not tx_active,
                decision_source="parity_default",
                tuning_probes=(),
                decline_reasons=tuple(declines),
            )
        if is_stock:
            return RowRouteTrace(
                rows=1,
                schedule_bucket=(1, 1),
                selected_route="mlx",
                selected_backend="mlx",
                kernel_name=None,
                policy_repr=None,
                effective_policy_repr=None,
                split_k=None,
                shared_group=None,
                unrolled=None,
                native_fast_eligible=False,
                decision_source="dot_fallback",
                tuning_probes=(),
                decline_reasons=tuple(declines),
            )
        return RowRouteTrace(
            rows=1,
            schedule_bucket=(1, 1),
            selected_route="decode",
            selected_backend="decode",
            kernel_name=None,
            policy_repr=None,
            effective_policy_repr=None,
            split_k=None,
            shared_group=None,
            unrolled=None,
            native_fast_eligible=False,
            decision_source="decode_fallback",
            tuning_probes=(),
            decline_reasons=tuple(declines),
        )

    # Rows > 1: check tile_dense schedule if bf16/fp16 and tile face exists.
    row_limit = (2**31) if tensor_units else (2**30)
    if dtype not in (mx.float16, mx.bfloat16):
        declines.append(
            f"tile_dense skipped: dtype={_dtype_label(dtype)} is not bf16/fp16"
        )
    elif tile_face is None:
        declines.append("tile_dense skipped: tile face unavailable")
    elif not tile_exact:
        declines.append(
            f"tile_dense skipped: tile face not exact for {_dtype_label(dtype)}"
        )
    elif r >= row_limit:
        declines.append(f"tile_dense skipped: rows={r} >= row_limit={row_limit}")
    elif cells is not None:
        cell = next(
            (c for c in cells if c[0] <= r and (c[1] is None or r <= c[1])),
            cells[-1],
        )
        c_lo, c_hi, route, source, probes, cell_declines = cell
        all_declines = [*declines, *cell_declines]
        bucket = (c_lo, c_hi)
        if route.backend == "dot":
            if dot_exact and r <= linear.MAX_ROWS:
                group = (
                    _group_for_chip(r, dot_face.vpt, n, gpu_generation, gpu_cores)
                    if dot_face.vpt >= 1
                    else 1
                )
                use_shared = group > 1 or n % 8 != 0
                return RowRouteTrace(
                    rows=r,
                    schedule_bucket=bucket,
                    selected_route="dot_shared" if use_shared else "dot",
                    selected_backend="dot",
                    kernel_name=(
                        f"omlx_format_linear_rows_{dot_face.key}"
                        if use_shared
                        else f"omlx_format_linear_{dot_face.key}"
                    ),
                    policy_repr=repr(route.policy) if route.policy else None,
                    effective_policy_repr=None,
                    split_k=None,
                    shared_group=group,
                    unrolled=(
                        kernels.unrolled_tile_ok(dot_face.vpt, linear._SHARED_RPS, 1)
                        if use_shared
                        else None
                    ),
                    native_fast_eligible=not tx_active,
                    decision_source=source,
                    tuning_probes=probes,
                    decline_reasons=tuple(all_declines),
                )
            if r > linear.MAX_ROWS:
                all_declines.append(f"dot declined: rows={r} > MAX_ROWS=16")
            if is_stock and r <= linear.MAX_ROWS:
                return RowRouteTrace(
                    rows=r,
                    schedule_bucket=bucket,
                    selected_route="mlx",
                    selected_backend="mlx",
                    kernel_name=None,
                    policy_repr=None,
                    effective_policy_repr=None,
                    split_k=None,
                    shared_group=None,
                    unrolled=None,
                    native_fast_eligible=False,
                    decision_source="dot_fallback",
                    tuning_probes=probes,
                    decline_reasons=tuple(all_declines),
                )
            return RowRouteTrace(
                rows=r,
                schedule_bucket=bucket,
                selected_route="decode",
                selected_backend="decode",
                kernel_name=None,
                policy_repr=None,
                effective_policy_repr=None,
                split_k=None,
                shared_group=None,
                unrolled=None,
                native_fast_eligible=False,
                decision_source="decode_fallback",
                tuning_probes=probes,
                decline_reasons=tuple(all_declines),
            )

        if route.backend in ("steel", "nax"):
            eff = tile_dense._effective_policy(route.policy, k, n)
            split_k, _ = tile_dense._split_k_geometry(k, eff.bk, eff.split_k)
            prefix = (
                tile_dense._nax_body(eff.interleaved_weights)[0]
                if route.backend == "nax"
                else "omlx_steel_dense"
            )
            return RowRouteTrace(
                rows=r,
                schedule_bucket=bucket,
                selected_route=route.backend,
                selected_backend=route.backend,
                kernel_name=f"{prefix}_{tile_face.key}",
                policy_repr=repr(route.policy),
                effective_policy_repr=repr(eff),
                split_k=split_k,
                shared_group=None,
                unrolled=None,
                native_fast_eligible=not tx_active,
                decision_source=source,
                tuning_probes=probes,
                decline_reasons=tuple(all_declines),
            )

        if route.backend == "wide":
            return RowRouteTrace(
                rows=r,
                schedule_bucket=bucket,
                selected_route="wide",
                selected_backend="wide",
                kernel_name=f"omlx_format_linear_wide_{tile_face.key}",
                policy_repr=repr(route.policy),
                effective_policy_repr=None,
                split_k=None,
                shared_group=None,
                unrolled=None,
                native_fast_eligible=not tx_active,
                decision_source=source,
                tuning_probes=probes,
                decline_reasons=tuple(all_declines),
            )

        if route.backend == "mma":
            m_pad = ((r + 7) // 8) * 8
            return RowRouteTrace(
                rows=r,
                schedule_bucket=bucket,
                selected_route="mma",
                selected_backend="mma",
                kernel_name=f"omlx_fl_mma_rt{m_pad // 8}_k{k}_{tile_face.key}",
                policy_repr=repr(route.policy),
                effective_policy_repr=None,
                split_k=None,
                shared_group=None,
                unrolled=None,
                native_fast_eligible=not tx_active,
                decision_source=source,
                tuning_probes=probes,
                decline_reasons=tuple(all_declines),
            )

        if route.backend == "msg":
            m_pad = 4 if r <= 4 else 6 if r <= 6 else 8
            pol = route.policy
            lt = getattr(pol.load_lifetime, "value", pol.load_lifetime) or "batch"
            return RowRouteTrace(
                rows=r,
                schedule_bucket=bucket,
                selected_route="msg",
                selected_backend="msg",
                kernel_name=f"omlx_vk_m{m_pad}_nsg{pol.nsg}_{lt}_{tile_face.key}",
                policy_repr=repr(pol),
                effective_policy_repr=None,
                split_k=None,
                shared_group=None,
                unrolled=None,
                native_fast_eligible=not tx_active,
                decision_source=source,
                tuning_probes=probes,
                decline_reasons=tuple(all_declines),
            )

        if route.backend == "mlx":
            return RowRouteTrace(
                rows=r,
                schedule_bucket=bucket,
                selected_route="mlx",
                selected_backend="mlx",
                kernel_name=None,
                policy_repr=repr(route.policy),
                effective_policy_repr=None,
                split_k=None,
                shared_group=None,
                unrolled=None,
                native_fast_eligible=not tx_active,
                decision_source=source,
                tuning_probes=probes,
                decline_reasons=tuple(all_declines),
            )

    # Fallback when tile_dense was skipped (fp32, no tile face, or row_limit).
    if dot_exact and r <= linear.MAX_ROWS:
        group = (
            _group_for_chip(r, dot_face.vpt, n, gpu_generation, gpu_cores)
            if dot_face.vpt >= 1
            else 1
        )
        use_shared = group > 1 or n % 8 != 0
        return RowRouteTrace(
            rows=r,
            schedule_bucket=None,
            selected_route="dot_shared" if use_shared else "dot",
            selected_backend="dot",
            kernel_name=(
                f"omlx_format_linear_rows_{dot_face.key}"
                if use_shared
                else f"omlx_format_linear_{dot_face.key}"
            ),
            policy_repr=None,
            effective_policy_repr=None,
            split_k=None,
            shared_group=group,
            unrolled=(
                kernels.unrolled_tile_ok(dot_face.vpt, linear._SHARED_RPS, 1)
                if use_shared
                else None
            ),
            native_fast_eligible=not tx_active,
            decision_source="dot_fallback",
            tuning_probes=(),
            decline_reasons=tuple(declines),
        )
    if r > linear.MAX_ROWS:
        declines.append(f"dot declined: rows={r} > MAX_ROWS=16")
    if is_stock and r <= linear.MAX_ROWS:
        return RowRouteTrace(
            rows=r,
            schedule_bucket=None,
            selected_route="mlx",
            selected_backend="mlx",
            kernel_name=None,
            policy_repr=None,
            effective_policy_repr=None,
            split_k=None,
            shared_group=None,
            unrolled=None,
            native_fast_eligible=False,
            decision_source="dot_fallback",
            tuning_probes=(),
            decline_reasons=tuple(declines),
        )
    return RowRouteTrace(
        rows=r,
        schedule_bucket=None,
        selected_route="decode",
        selected_backend="decode",
        kernel_name=None,
        policy_repr=None,
        effective_policy_repr=None,
        split_k=None,
        shared_group=None,
        unrolled=None,
        native_fast_eligible=False,
        decision_source="decode_fallback",
        tuning_probes=(),
        decline_reasons=tuple(declines),
    )


def _explain_switch_row(
    layer: FormatSwitchLinear,
    r: int,
    *,
    dtype: mx.Dtype,
    tensor_units: bool,
    sorted_indices: bool,
    dot_face: Dot | None,
    dot_plan_ok: bool,
    dot_exact: bool,
    tile_face: Tile | None,
    tile_exact: bool,
) -> RowRouteTrace:
    k, n, experts = layer.input_dims, layer.output_dims, layer.num_experts
    declines: list[str] = []
    if not sorted_indices:
        declines.append("sorted tile declined: sorted_indices=False")
    if not dot_plan_ok:
        declines.append("sorted tile declined: dot plan unavailable")
    if r < 16:
        declines.append(f"sorted tile declined: rows={r} < 16")
    if r < 4 * experts:
        declines.append(f"sorted tile declined: rows={r} < 4 * experts={4 * experts}")
    if dtype not in (mx.float16, mx.bfloat16):
        declines.append(
            f"sorted tile declined: dtype={_dtype_label(dtype)} not bf16/fp16"
        )
    if k % 32 != 0:
        declines.append(f"sorted tile declined: k={k} % 32 != 0")
    if tensor_units:
        if experts > nax._MAX_EXPERTS:
            declines.append(
                f"sorted tile declined: experts={experts} > {nax._MAX_EXPERTS}"
            )
        if n % 32 != 0:
            declines.append(f"sorted tile declined: nax requires n={n} % 32 == 0")
    elif r >= (1 << 30):
        declines.append(f"sorted tile declined: rows={r} >= 2**30")
    if tile_face is None:
        declines.append("sorted tile declined: tile face unavailable")
    elif not tile_exact:
        declines.append(
            f"sorted tile declined: tile face not exact for {_dtype_label(dtype)}"
        )

    if not declines:
        if tensor_units:
            plan = nax.face_plan(r, experts, k, n)
            return RowRouteTrace(
                rows=r,
                schedule_bucket=None,
                selected_route="switch_tile_nax",
                selected_backend="nax",
                kernel_name=f"omlx_gqmm_{tile_face.key}",
                policy_repr=repr(plan),
                effective_policy_repr=None,
                split_k=None,
                shared_group=None,
                unrolled=None,
                native_fast_eligible=False,
                decision_source="sorted_tile",
                tuning_probes=(),
                decline_reasons=(),
            )
        return RowRouteTrace(
            rows=r,
            schedule_bucket=None,
            selected_route="switch_tile_steel",
            selected_backend="steel",
            kernel_name=f"omlx_steel_gather_{tile_face.key}",
            policy_repr="SteelGather(BM=16, BN=32, BK=32)",
            effective_policy_repr=None,
            split_k=None,
            shared_group=None,
            unrolled=None,
            native_fast_eligible=False,
            decision_source="sorted_tile",
            tuning_probes=(),
            decline_reasons=(),
        )

    if dot_exact and r < (1 << 30):
        return RowRouteTrace(
            rows=r,
            schedule_bucket=None,
            selected_route="switch_dot",
            selected_backend="dot",
            kernel_name=f"omlx_format_switch_linear_{dot_face.key}",
            policy_repr=None,
            effective_policy_repr=None,
            split_k=None,
            shared_group=None,
            unrolled=None,
            native_fast_eligible=False,
            decision_source="switch_dot",
            tuning_probes=(),
            decline_reasons=tuple(declines),
        )

    if dot_face is None:
        declines.append("switch_dot declined: dot face unavailable")
    elif not dot_plan_ok:
        if k < 8:
            declines.append(f"switch_dot declined: k={k} < 8")
        elif n % 8 != 0:
            declines.append(f"switch_dot declined: n={n} % 8 != 0")
    elif not dot_exact:
        declines.append(
            f"switch_dot declined: dot face not exact for {_dtype_label(dtype)}"
        )
    elif r >= (1 << 30):
        declines.append(f"switch_dot declined: rows={r} >= 2**30")

    return RowRouteTrace(
        rows=r,
        schedule_bucket=None,
        selected_route="switch_decode",
        selected_backend="decode",
        kernel_name=None,
        policy_repr=None,
        effective_policy_repr=None,
        split_k=None,
        shared_group=None,
        unrolled=None,
        native_fast_eligible=False,
        decision_source="switch_decode",
        tuning_probes=(),
        decline_reasons=tuple(declines),
    )


def explain(
    layer: FormatLinear | FormatSwitchLinear | FormatEmbedding,
    *,
    rows: Sequence[int] = DEFAULT_EXPLAIN_ROWS,
    dtype: str | mx.Dtype = mx.bfloat16,
    chip: str | None = None,
    tensor_units: bool | None = None,
    sorted_indices: bool = True,
    input_ndim: int = 3,
) -> ExplainResult:
    """Explain face availability, tuning probes, and route choice for ``layer``.

    Pure and read-only: never mutates ``layer.__dict__``, ``formats._DOT_DEFECTS``,
    ``kernels._PROOFS``, or ``kernels._FACE_KERNELS``.
    """
    if not isinstance(layer, (FormatLinear, FormatSwitchLinear, FormatEmbedding)):
        raise TypeError(
            "explain expects FormatLinear, FormatSwitchLinear, or FormatEmbedding, "
            f"not {type(layer).__name__}"
        )
    layer.refuse_placeholders()
    if type(input_ndim) is not int or input_ndim not in (2, 3):
        raise ValueError(f"input_ndim must be 2 or 3, not {input_ndim!r}")
    if not isinstance(sorted_indices, bool):
        raise TypeError(
            f"sorted_indices must be a bool, not {type(sorted_indices).__name__}"
        )
    row_list = tuple(rows)
    if not row_list:
        raise ValueError("rows must be a non-empty sequence of positive integers")
    for r in row_list:
        _check_positive_int("rows entry", r)

    dt = parse_dtype(dtype)
    chip_id, arch, gpu_cores, gpu_gen, tu = _resolve_chip(chip, tensor_units)
    table_mode = "selects" if kernels._selects_for(arch) else "staged"

    fmt = layer.fmt
    k, n = layer.input_dims, layer.output_dims
    is_switch = isinstance(layer, FormatSwitchLinear)
    is_embedding = isinstance(layer, FormatEmbedding)
    experts = layer.num_experts if is_switch else None
    shared_names = tuple(getattr(fmt, "shared_tensors", ()))
    all_names = (fmt.codes_name, *fmt.tensors, *shared_names)

    tensor_specs: list[dict[str, Any]] = []
    for name in all_names:
        arr = layer.get(name)
        if isinstance(arr, mx.array):
            tensor_specs.append(
                {
                    "name": name,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype).removeprefix("mlx.core."),
                    "nbytes": int(arr.nbytes),
                    "is_codes": name == fmt.codes_name,
                    "shared": name in shared_names,
                }
            )
    if "bias" in layer and isinstance(layer["bias"], mx.array):
        b_arr = layer["bias"]
        tensor_specs.append(
            {
                "name": "bias",
                "shape": list(b_arr.shape),
                "dtype": str(b_arr.dtype).removeprefix("mlx.core."),
                "nbytes": int(b_arr.nbytes),
                "is_codes": False,
                "shared": False,
            }
        )

    face_objs, face_traces = _probe_faces(layer, dt)
    tx_trace = _probe_input_transform(layer)

    dot_face: Dot | None = face_objs["dot"]
    tile_face: Tile | None = face_objs["tile"]
    stacked = is_switch
    dot_plan_ok = (
        dot_face is not None
        and k >= 8
        and not (n % 8 != 0 and (stacked or dot_face.vpt < 1))
    )
    dot_exact = dot_plan_ok and dot_face.exact_for(dt)
    tile_exact = tile_face is not None and tile_face.exact_for(dt)

    schedule_summary: tuple[dict[str, Any], ...] = ()
    row_traces: list[RowRouteTrace] = []

    if is_embedding:
        for r in row_list:
            row_traces.append(
                RowRouteTrace(
                    rows=r,
                    schedule_bucket=None,
                    selected_route="embedding_lookup",
                    selected_backend="decode",
                    kernel_name=None,
                    policy_repr=None,
                    effective_policy_repr=None,
                    split_k=None,
                    shared_group=None,
                    unrolled=None,
                    native_fast_eligible=False,
                    decision_source="decode_fallback",
                    tuning_probes=(),
                    decline_reasons=(
                        "FormatEmbedding.__call__ decodes selected rows directly; "
                        "as_linear() uses FormatLinear routes",
                    ),
                )
            )
    elif is_switch:
        for r in row_list:
            row_traces.append(
                _explain_switch_row(
                    layer,
                    r,
                    dtype=dt,
                    tensor_units=tu,
                    sorted_indices=sorted_indices,
                    dot_face=dot_face,
                    dot_plan_ok=dot_plan_ok,
                    dot_exact=dot_exact,
                    tile_face=tile_face,
                    tile_exact=tile_exact,
                )
            )
    else:
        cells = None
        if tile_face is not None and tile_exact and dt in (mx.float16, mx.bfloat16):
            cells, schedule_summary = _build_traced_schedule(
                layer,
                tile_face,
                dt,
                tensor_units=tu,
                chip=chip_id,
                dot_ok=dot_exact,
            )
        for r in row_list:
            row_traces.append(
                _explain_dense_row(
                    layer,
                    r,
                    dtype=dt,
                    architecture=arch,
                    tensor_units=tu,
                    gpu_cores=gpu_cores,
                    gpu_generation=gpu_gen,
                    input_ndim=input_ndim,
                    dot_face=dot_face,
                    dot_plan_ok=dot_plan_ok,
                    dot_exact=dot_exact,
                    tile_face=tile_face,
                    tile_exact=tile_exact,
                    tx_active=tx_trace.active,
                    cells=cells,
                )
            )

    return ExplainResult(
        layer_type=_layer_type_name(layer),
        mode=layer.mode,
        origin=formats.origin(layer.mode),
        contract=int(fmt.contract),
        bits=int(layer.bits),
        group_size=int(layer.group_size),
        k=int(k),
        n=int(n),
        experts=experts,
        has_bias="bias" in layer,
        is_stock=bool(getattr(layer, "_stock", False)),
        stackable=_is_stackable_pure(fmt, k, n, layer.bits, layer.group_size),
        dtype=_dtype_label(dt),
        chip=chip_id,
        architecture=arch,
        tensor_units=tu,
        gpu_cores=gpu_cores,
        gpu_generation=gpu_gen,
        table_mode=table_mode,
        codes_name=fmt.codes_name,
        tensors=tuple(fmt.tensors),
        shared_tensors=shared_names,
        tensor_specs=tuple(tensor_specs),
        faces=face_traces,
        input_transform=tx_trace,
        schedule_summary=schedule_summary,
        row_traces=tuple(row_traces),
    )


def format_explain(result: ExplainResult, *, as_json: bool = False) -> str:
    """Format an ``ExplainResult`` as human-readable text or JSON."""
    if as_json:
        return json.dumps(result.to_dict(), indent=2, sort_keys=True)
    lines = [
        f"Layer: {result.layer_type} mode={result.mode} "
        f"(bits={result.bits}, group_size={result.group_size}, "
        f"k={result.k}, n={result.n}"
        + (f", experts={result.experts}" if result.experts is not None else "")
        + ")",
        f"Target: chip={result.chip} arch={result.architecture} "
        f"tensor_units={result.tensor_units} cores={result.gpu_cores} "
        f"dtype={result.dtype} tables={result.table_mode}",
        f"Format: contract={result.contract} stock={result.is_stock} "
        f"stackable={result.stackable} origin={result.origin or 'builtin/local'}",
    ]
    tx = result.input_transform
    if tx.declared:
        lines.append(
            f"InputTransform: active={tx.active} kind={tx.kind} "
            f"version={tx.version} block={tx.block} tensors={tx.tensors} "
            f"native_fast_disabled={tx.native_fast_disabled}"
        )
    lines.append("Faces:")
    for f in result.faces:
        if f.available:
            lines.append(
                f"  - {f.kind}: key={f.key} exact={f.exact_for_dtype} "
                f"profile={f.table_profile} table_bytes={f.table_bytes} "
                f"digest={f.face_digest}"
            )
        else:
            lines.append(f"  - {f.kind}: unavailable ({f.decline_reason})")
    lines.append("Routes:")
    for rt in result.row_traces:
        extra = f" kernel={rt.kernel_name}" if rt.kernel_name else ""
        if rt.policy_repr:
            extra += f" policy={rt.policy_repr}"
        lines.append(
            f"  rows={rt.rows:<5} -> {rt.selected_route:<18} "
            f"[{rt.decision_source}]{extra}"
        )
        for reason in rt.decline_reasons:
            lines.append(f"      decline: {reason}")
        for probe in rt.tuning_probes:
            status = "SELECTED" if probe.selected else "rejected"
            why = "; ".join(probe.rejected_reasons)
            lines.append(
                f"      probe({probe.route}, {probe.face_id}, "
                f"{probe.shape_scope}, {probe.rows_bucket}): "
                f"{status}" + (f" ({why})" if why else "")
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Milestone 2 — dump_metal standalone Metal emitter
# ---------------------------------------------------------------------------

_METAL_TYPE_NAMES: dict[mx.Dtype, str] = {
    mx.bfloat16: "bfloat16_t",
    mx.float16: "half",
    mx.float32: "float",
    mx.uint32: "uint32_t",
    mx.int32: "int32_t",
    mx.uint16: "uint16_t",
    mx.int16: "int16_t",
    mx.uint8: "uint8_t",
    mx.int8: "int8_t",
    mx.uint64: "uint64_t",
    mx.int64: "int64_t",
    mx.bool_: "bool",
}

_THREAD_ATTR_SPECS: tuple[tuple[str, str], ...] = (
    ("thread_position_in_grid", "uint3"),
    ("thread_position_in_threadgroup", "uint3"),
    ("threadgroup_position_in_grid", "uint3"),
    ("threadgroups_per_grid", "uint3"),
    ("threads_per_threadgroup", "uint3"),
    ("threads_per_grid", "uint3"),
    ("thread_index_in_threadgroup", "uint"),
    ("thread_index_in_simdgroup", "uint"),
    ("simdgroup_index_in_threadgroup", "uint"),
    ("simdgroups_per_threadgroup", "uint"),
)


@dataclass(frozen=True)
class MetalSection:
    name: str
    start_line: int
    end_line: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class LaunchGeometry:
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]
    threadgroups: tuple[int, int, int]
    template: tuple[tuple[str, str | int | bool], ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    input_specs: tuple[tuple[str, str, tuple[int, ...], str], ...]
    output_specs: tuple[tuple[str, str, tuple[int, ...]], ...]
    staged_table_bytes: int
    split_k: int | None
    k_part: int | None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class DumpMetalResult:
    kernel_name: str
    route: str
    backend: str
    mode: str
    bits: int
    group_size: int
    k: int
    n: int
    rows: int
    experts: int | None
    dtype: str
    chip: str
    tensor_units: bool
    table_mode: str
    face_key: str
    face_digest: str
    skeleton_digest: str | None
    policy_repr: str | None
    launch: LaunchGeometry
    raw_header: str
    raw_source: str
    raw_digest: str
    metal: str
    metal_digest: str
    sections: tuple[MetalSection, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


class _TargetTables(kernels.Tables):
    """Local mirror of ``kernels.Tables`` with explicit target ``selects`` flag."""

    def __init__(self, *, selects: bool) -> None:
        super().__init__()
        self._selects = selects

    def add(self, table: Table) -> str:
        if not isinstance(table, Table):
            raise TypeError(f"expected a Table, got {type(table).__name__}")
        at = self._at.get(table)
        if at is None:
            size = kernels._bytes((*self._at, table))
            if size > formats.TABLE_BYTES:
                raise ValueError(
                    f"table {table.name!r}: the kernel's tables would hold {size} "
                    f"bytes; a kernel holds at most {formats.TABLE_BYTES}"
                )
            metal, bits = kernels._METAL[table.dtype]
            n = len(table.values)
            if n <= kernels._SELECT_VALUES and self._selects:
                self._values = True
                past = kernels._PAST.get(table.dtype, 0)
                words = [*table.bits, *(past,) * (kernels._SELECT_VALUES - n)]
                values = ", ".join(f"as_type<{metal}>({bits}({w:#x}u))" for w in words)
            else:
                run = self._by_type.setdefault(metal, [])
                values = f"omlx_tg_{metal} + {sum(len(t.values) for t in run)}"
                run.append(table)
            at = f"omlx_table<{metal}, {n}>{{{values}}}"
            self._at[table] = at
        return at


def _make_matrices(
    face: Dot | Tile,
    *,
    target_selects: bool,
    unrolled: bool = False,
    unroll_pairs: bool = False,
) -> kernels.Matrices:
    """Build a local ``kernels.Matrices`` without mutating host globals."""
    matrices = kernels.Matrices(unrolled=unrolled, unroll_pairs=unroll_pairs)
    has_small = any(
        len(t.values) <= kernels._SELECT_VALUES for t in getattr(face, "tables", ())
    )
    if has_small and target_selects != kernels._small_tables_as_selects():
        matrices._tables = _TargetTables(selects=target_selects)
    return matrices


def _metal_type(dtype: mx.Dtype) -> str:
    name = _METAL_TYPE_NAMES.get(dtype)
    if name is None:
        raise ValueError(f"unsupported Metal element dtype {dtype!r}")
    return name


def _format_template_arg(value: Any) -> str:
    # bool must be checked strictly before int (bool is a subclass of int).
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, mx.Dtype):
        return _metal_type(value)
    raise TypeError(f"unsupported Metal template value {value!r}")


def _template_param_decl(name: str, value: Any) -> str:
    if isinstance(value, mx.Dtype):
        return f"typename {name}"
    if isinstance(value, bool):
        return f"bool {name}"
    if isinstance(value, int):
        return f"int {name}"
    raise TypeError(f"unsupported Metal template value {value!r}")


def _template_record_value(value: Any) -> str | int | bool:
    if isinstance(value, mx.Dtype):
        return _metal_type(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    return str(value)


def _addr_space(shape: tuple[int, ...]) -> str:
    if len(shape) == 0:
        return "scalar"
    if math.prod(shape) < 8:
        return "constant"
    return "device"


def _inline_mlx_utils_preamble() -> str:
    """Flatten MLX's ``utils.h`` preamble for standalone Metal compilation."""
    root = Path(mx.__file__).parent / "include"
    utils_rel = "mlx/backend/metal/kernels/utils.h"
    if not (root / utils_rel).is_file():
        raise RuntimeError(f"MLX Metal headers not found under {root}")
    seen: set[Path] = set()

    def expand(rel: str) -> str:
        path = root / rel
        if path in seen:
            return ""
        seen.add(path)
        out: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith('#include "mlx/') and stripped.endswith('"'):
                out.append(expand(stripped[len('#include "') : -1]))
            elif stripped != "#pragma once":
                out.append(line)
        return "\n".join(out)

    head = (
        "#include <metal_stdlib>\n"
        "#include <metal_simdgroup_matrix>\n"
        "#ifndef METAL_FUNC\n"
        "#define METAL_FUNC inline\n"
        "#endif\n"
        "using namespace metal;\n"
    )
    return head + expand(utils_rel)


def _assemble_standalone_metal(
    *,
    kernel_name: str,
    route: str,
    mode: str,
    bits: int,
    group_size: int,
    k: int,
    n: int,
    rows: int,
    chip: str,
    table_mode: str,
    policy_repr: str | None,
    raw_digest: str,
    raw_source: str,
    mlx_route_headers: str,
    face_header: str,
    skeleton_header: str,
    matrix_declarations: str,
    skeleton_body: str,
    template: list[tuple[str, Any]],
    launch: LaunchGeometry,
    include_mlx_preamble: bool,
) -> tuple[str, tuple[MetalSection, ...]]:
    body_sha = hashlib.sha256(raw_source.encode()).hexdigest()
    banner_lines = [
        f"// oMLX Standalone Metal Dump — {kernel_name}",
        f"// route={route} mode={mode} bits={bits} group_size={group_size} "
        f"k={k} n={n} rows={rows} chip={chip} tables={table_mode}",
        f"// policy={policy_repr or 'none'}",
        f"// emitted_sha256={raw_digest}",
        f"// body_sha256={body_sha}",
        f"// grid={launch.grid} threadgroup={launch.threadgroup} "
        f"threadgroups={launch.threadgroups}",
        f"// template={list(launch.template)}",
        f"// inputs={list(launch.input_names)} outputs={list(launch.output_names)}",
        f"// staged_tables={launch.staged_table_bytes}B "
        f"split_k={launch.split_k} k_part={launch.k_part}",
    ]

    lines: list[str] = [*banner_lines, ""]
    sections: list[MetalSection] = []

    def add_section(sec_name: str, content: str) -> None:
        start_line = len(lines) + 1
        lines.append(f"// >>> OMLX_SECTION: {sec_name} >>>")
        sec_lines = content.splitlines()
        lines.extend(sec_lines)
        end_line = len(lines) + 1
        lines.append(f"// <<< OMLX_SECTION: {sec_name} <<<")
        sec_sha = hashlib.sha256("\n".join(sec_lines).encode()).hexdigest()
        sections.append(MetalSection(sec_name, start_line, end_line, sec_sha))

    if include_mlx_preamble:
        add_section("mlx_preamble", _inline_mlx_utils_preamble())
    if mlx_route_headers:
        add_section("mlx_route_headers", mlx_route_headers)
    add_section("face", face_header)
    add_section("skeleton_header", skeleton_header)

    # Build templated [[kernel]] entry point matching MLX buffer binding rules.
    params: list[str] = []
    buf_idx = 0
    for in_name, mtype, shape, space in launch.input_specs:
        if space == "scalar":
            params.append(
                f"    const constant {mtype}& {in_name} [[buffer({buf_idx})]]"
            )
        elif space == "constant":
            params.append(
                f"    const constant {mtype}* {in_name} [[buffer({buf_idx})]]"
            )
        else:
            params.append(f"    const device {mtype}* {in_name} [[buffer({buf_idx})]]")
        buf_idx += 1
        if shape:
            if re.search(rf"\b{re.escape(in_name)}_shape\b", raw_source):
                params.append(
                    f"    const constant int* {in_name}_shape [[buffer({buf_idx})]]"
                )
                buf_idx += 1
            if re.search(rf"\b{re.escape(in_name)}_strides\b", raw_source):
                params.append(
                    f"    const constant int64_t* {in_name}_strides "
                    f"[[buffer({buf_idx})]]"
                )
                buf_idx += 1
            if re.search(rf"\b{re.escape(in_name)}_ndim\b", raw_source):
                params.append(
                    f"    const constant int& {in_name}_ndim [[buffer({buf_idx})]]"
                )
                buf_idx += 1
    for out_name, mtype, _shape in launch.output_specs:
        params.append(f"    device {mtype}* {out_name} [[buffer({buf_idx})]]")
        buf_idx += 1

    for attr_name, attr_type in _THREAD_ATTR_SPECS:
        if re.search(rf"\b{attr_name}\b", raw_source):
            params.append(f"    {attr_type} {attr_name} [[{attr_name}]]")

    t_decls = ", ".join(_template_param_decl(tn, tv) for tn, tv in template)
    t_args = ", ".join(_format_template_arg(tv) for _, tv in template)
    tmpl_fn = f"{kernel_name}_tmpl"

    entry_start = len(lines) + 1
    lines.append("// >>> OMLX_SECTION: kernel_entry >>>")
    if t_decls:
        lines.append(f"template <{t_decls}>")
    lines.append(f"[[kernel]] void {tmpl_fn}(")
    for idx, param in enumerate(params):
        comma = "," if idx < len(params) - 1 else ""
        lines.append(f"{param}{comma}")
    lines.append(") {")

    add_section("matrix_declarations", matrix_declarations)
    add_section("skeleton_body", skeleton_body)

    lines.append("}")
    if t_args:
        lines.append(
            f'template [[host_name("{kernel_name}")]] [[kernel]] '
            f"decltype({tmpl_fn}<{t_args}>) {tmpl_fn}<{t_args}>;"
        )
    entry_end = len(lines) + 1
    lines.append("// <<< OMLX_SECTION: kernel_entry <<<")

    entry_body = "\n".join(lines[entry_start : entry_end - 1])
    entry_sec = MetalSection(
        "kernel_entry",
        entry_start,
        entry_end,
        hashlib.sha256(entry_body.encode()).hexdigest(),
    )
    # Place kernel_entry ahead of its nested sections for predictable order.
    insert_at = len(sections) - 2
    sections.insert(insert_at, entry_sec)

    return "\n".join(lines) + "\n", tuple(sections)


def _operand_specs(
    layer: Any, prefix: str, face: Dot | Tile
) -> list[tuple[str, str, tuple[int, ...], str]]:
    packed_key = linear.packed_name(layer)
    arr_names = (packed_key, *face.tensors)
    in_names = (f"{prefix}_weight", *(f"{prefix}_{t}" for t in face.tensors))
    specs: list[tuple[str, str, tuple[int, ...], str]] = []
    for in_name, layer_key in zip(in_names, arr_names, strict=True):
        arr = layer[layer_key]
        shape = tuple(int(d) for d in arr.shape)
        specs.append((in_name, _metal_type(arr.dtype), shape, _addr_space(shape)))
    return specs


def _lookup_forced_dense_policy(
    table_face: Tile,
    actual_face: Tile,
    *,
    backend: str,
    dtype: mx.Dtype,
    bits: int,
    group: int,
    n: int,
    k: int,
    rows: int,
    chip: str,
) -> DenseTilePolicy | None:
    """Find the highest-rank legal ``DenseTilePolicy`` for a forced backend."""
    table = tuning._index_for_chip(chip)
    if not table.entries:
        return None
    skeleton = tuning.skeleton_digest("dense_tile")
    if skeleton is None:
        return None
    dtype_name = tuning._dtype_name(dtype)
    candidates = (
        table.by_key.get(("dense_tile", dtype_name, bits, group), ())
        if table.by_key
        else table.entries
    )
    if not candidates:
        return None
    query = tuning._Query(
        route="dense_tile",
        face_key=table_face.key,
        profile=tuning.table_profile(table_face),
        dtype=dtype_name,
        bits=bits,
        group=group,
        n=n,
        k=k,
        rows=rows,
        rows_bucket=tuning.rows_bucket("dense_tile", rows),
        n_bucket=tuning._n_bucket(n),
        k_bucket=tuning._k_bucket(k),
    )
    digest: str | None = None
    best_rank = (0, 0)
    best_params: DenseTilePolicy | None = None
    for entry in candidates:
        params = entry.params
        if not isinstance(params, DenseTilePolicy) or params.backend != backend:
            continue
        rank = tuning._matches(entry, query)
        if rank[0] == 0 or rank <= best_rank:
            continue
        if entry.skeleton_sha256 != skeleton:
            continue
        if entry.key.get("face_id") == table_face.key:
            if digest is None:
                digest = tuning.face_digest(table_face)
            if entry.face_sha256 != digest:
                continue
        if not params.is_legal(actual_face, k, n, rows, chip, dtype, tail=True):
            continue
        best_rank = rank
        best_params = params
    return best_params


def dump_metal(
    layer: FormatLinear | FormatSwitchLinear,
    *,
    rows: int = 1,
    dtype: str | mx.Dtype = mx.bfloat16,
    chip: str | None = None,
    tensor_units: bool | None = None,
    route: str | None = None,
    sorted_indices: bool = True,
    per_row: int = 1,
    input_ndim: int = 3,
    include_mlx_preamble: bool = True,
) -> DumpMetalResult:
    """Emit exact, self-contained Metal source and launch geometry for ``layer``.

    Pure and read-only (Rule 14): never mutates ``layer.__dict__``,
    ``kernels._FACE_KERNELS``, ``kernels._PROOFS``, or host table-mode globals.
    """
    if not isinstance(layer, (FormatLinear, FormatSwitchLinear)):
        raise TypeError(
            "dump_metal expects FormatLinear or FormatSwitchLinear, "
            f"not {type(layer).__name__}"
        )
    layer.refuse_placeholders()
    _check_positive_int("rows", rows)
    _check_positive_int("per_row", per_row)
    if type(input_ndim) is not int or input_ndim not in (2, 3):
        raise ValueError(f"input_ndim must be 2 or 3, not {input_ndim!r}")
    if not isinstance(sorted_indices, bool):
        raise TypeError(
            f"sorted_indices must be a bool, not {type(sorted_indices).__name__}"
        )
    if not isinstance(include_mlx_preamble, bool):
        raise TypeError(
            "include_mlx_preamble must be a bool, "
            f"not {type(include_mlx_preamble).__name__}"
        )

    dt = parse_dtype(dtype)
    chip_id, arch, gpu_cores, gpu_gen, tu = _resolve_chip(chip, tensor_units)
    target_selects = kernels._selects_for(arch)
    table_mode = "selects" if target_selects else "staged"

    is_switch = isinstance(layer, FormatSwitchLinear)
    k, n = layer.input_dims, layer.output_dims
    experts = layer.num_experts if is_switch else None

    # Resolve route (auto via explain if route is None, or validate forced route).
    scheduled_policy: Any = None
    if route is not None:
        if not isinstance(route, str) or route not in DUMP_ROUTES:
            raise ValueError(
                f"unknown dump route {route!r}; valid routes are {DUMP_ROUTES}"
            )
        chosen_route = route
    else:
        exp = explain(
            layer,
            rows=(rows,),
            dtype=dt,
            chip=chip_id,
            tensor_units=tu,
            sorted_indices=sorted_indices,
            input_ndim=input_ndim,
        )
        rt = exp.row_traces[0]
        chosen_route = rt.selected_route
        if chosen_route in ("mlx", "decode", "switch_decode"):
            valid = (
                ("switch_dot", "switch_tile_steel", "switch_tile_nax")
                if is_switch
                else ("dot", "dot_shared", "steel", "nax", "wide", "mma", "msg")
            )
            raise ValueError(
                f"layer selected non-custom-Metal route {chosen_route!r} for "
                f"rows={rows} dtype={_dtype_label(dt)} chip={chip_id}; "
                f"pass an explicit route override from {valid}"
            )

    # Validate layer kind matches route kind.
    switch_routes = ("switch_dot", "switch_tile_steel", "switch_tile_nax")
    if is_switch and chosen_route not in switch_routes:
        raise ValueError(
            f"route {chosen_route!r} is a dense route and cannot be used with "
            f"FormatSwitchLinear (valid: {switch_routes})"
        )
    if not is_switch and chosen_route in switch_routes:
        raise ValueError(
            f"route {chosen_route!r} is a switch route and cannot be used with "
            "FormatLinear"
        )

    face_objs, _ = _probe_faces(layer, dt)
    dot_face: Dot | None = face_objs["dot"]
    tile_face: Tile | None = face_objs["tile"]

    # Recover the exact scheduled policy on dense layers (auto or forced route).
    forced_table_face: Tile | None = None
    if not is_switch and tile_face is not None and dt in (mx.float16, mx.bfloat16):
        dot_ok = (
            dot_face is not None
            and k >= 8
            and not (n % 8 != 0 and dot_face.vpt < 1)
            and dot_face.exact_for(dt)
        )
        sched = tile_dense._build_schedule(
            layer,
            tile_face,
            dt,
            tensor_units=tu,
            chip=chip_id,
            dot_ok=dot_ok,
        )
        scheduled_policy = tile_dense._route_for_rows(sched, rows).policy
        forced_table_face = (
            affine.tile_face(layer.bits, layer.group_size, dt)
            if layer.mode == "oqh_affine"
            else tile_face
        )

    x_shape = (
        (rows, 1, k) if is_switch else ((1, rows, k) if input_ndim == 3 else (rows, k))
    )
    out_shape_default = (*x_shape[:-1], n)
    mtype_x = _metal_type(dt)

    mlx_route_headers = ""
    skeleton_header = ""
    skeleton_body = ""
    skeleton_digest: str | None = None
    policy_repr: str | None = None
    split_k_val: int | None = None
    k_part_val: int | None = None
    output_names: tuple[str, ...] = ("y",)
    out_specs: list[tuple[str, str, tuple[int, ...]]] = [
        ("y", mtype_x, out_shape_default)
    ]

    if chosen_route == "dot":
        if dot_face is None or not dot_face.exact_for(dt) or k < 8 or n % 8 != 0:
            raise ValueError(
                f"route 'dot' is illegal for {layer.mode} "
                f"(k={k}, n={n}, dtype={_dtype_label(dt)})"
            )
        face: Dot | Tile = dot_face
        backend = "dot"
        matrices = _make_matrices(face, target_selects=target_selects)
        input_names = tuple(["x", *matrices.add("m", face)])
        kernel_name = f"omlx_format_linear_{face.key}"
        skeleton_header = kernels.STORE_ROWS
        skeleton_body = linear._SOURCE.replace("__MATRIX__", "0").replace(
            "__X_ROW__", "pair"
        )
        template: list[tuple[str, Any]] = [("T", dt), ("K_SIZE", k), ("N_SIZE", n)]
        grid = (32, 2 * rows, n // 8)
        threadgroup = (32, 2, 1)
        input_specs = [
            ("x", mtype_x, x_shape, _addr_space(x_shape)),
            *_operand_specs(layer, "m", face),
        ]

    elif chosen_route == "dot_shared":
        if dot_face is None or not dot_face.exact_for(dt) or k < 8 or dot_face.vpt < 1:
            raise ValueError(
                f"route 'dot_shared' is illegal for {layer.mode} "
                f"(k={k}, n={n}, dtype={_dtype_label(dt)})"
            )
        face = dot_face
        backend = "dot"
        unrolled = kernels.unrolled_tile_ok(face.vpt, linear._SHARED_RPS, 1)
        matrices = _make_matrices(
            face, target_selects=target_selects, unrolled=unrolled
        )
        input_names = tuple(["x", *matrices.add("m", face, shared=True)])
        kernel_name = f"omlx_format_linear_rows_{face.key}"
        skeleton_header = kernels.STORE_ROWS
        skeleton_body = linear._SHARED_SOURCE
        group = _group_for_chip(rows, face.vpt, n, gpu_gen, gpu_cores)
        template = [
            ("T", dt),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("ROWS", group),
            ("VPT", face.vpt),
        ]
        grid = (32, 2 * (rows // group), -(-n // linear._SHARED_TILE))
        threadgroup = (32, 2, 1)
        input_specs = [
            ("x", mtype_x, x_shape, _addr_space(x_shape)),
            *_operand_specs(layer, "m", face),
        ]

    elif chosen_route in ("steel", "nax"):
        if chosen_route == "nax" and not tu:
            raise ValueError(
                f"route 'nax' requires tensor units (unavailable on chip {chip_id})"
            )
        if tile_face is None or not tile_face.exact_for(dt):
            raise ValueError(
                f"route {chosen_route!r} requires a tile face exact for "
                f"{_dtype_label(dt)}"
            )
        face = tile_face
        backend = chosen_route
        if (
            isinstance(scheduled_policy, DenseTilePolicy)
            and scheduled_policy.backend == chosen_route
        ):
            pol = scheduled_policy
        else:
            cand = (
                _lookup_forced_dense_policy(
                    forced_table_face,
                    face,
                    backend=chosen_route,
                    dtype=dt,
                    bits=layer.bits,
                    group=layer.group_size,
                    n=n,
                    k=k,
                    rows=rows,
                    chip=chip_id,
                )
                if forced_table_face is not None
                else None
            )
            if cand is not None:
                pol = cand
            elif chosen_route == "nax":
                pol = tile_dense._default_nax_policy(layer.bits, rows, n, k)
            else:
                pol = tile_dense._default_steel_policy(rows, n, k)
        if not pol.is_legal(face, k, n, rows, chip_id, dt, tail=True):
            raise ValueError(
                f"route {chosen_route!r} policy {pol!r} is illegal for "
                f"(k={k}, n={n}, rows={rows}, chip={chip_id})"
            )
        eff = tile_dense._effective_policy(pol, k, n)
        policy_repr = repr(eff)
        mlx_hdr = (
            tile_dense._nax_headers()
            if chosen_route == "nax"
            else tile_dense._steel_headers()
        )
        if mlx_hdr is None:
            raise RuntimeError(f"MLX {chosen_route} headers unavailable")
        mlx_route_headers = mlx_hdr
        matrices = _make_matrices(face, target_selects=target_selects)
        input_names = tuple(["x", *matrices.add("w", face)])
        skeleton_digest = tuning.skeleton_digest("dense_tile")
        split_k_val, k_part_val = tile_dense._split_k_geometry(k, eff.bk, eff.split_k)
        n_cols = (n + eff.bn - 1) // eff.bn
        n_rows = (rows + eff.bm - 1) // eff.bm
        out_shape = (split_k_val, rows, n) if split_k_val > 1 else out_shape_default
        out_dtype = mx.float32 if split_k_val > 1 else dt
        out_specs = [("y", _metal_type(out_dtype), out_shape)]

        if chosen_route == "nax":
            prefix, skeleton_header = tile_dense._nax_body(eff.interleaved_weights)
            kernel_name = f"{prefix}_{face.key}"
            skeleton_body = tile_dense._NAX_SOURCE
            staged = matrices.staged_bytes
            if staged:
                skeleton_body = skeleton_body.replace(
                    "PAD / sizeof(T)];",
                    f"(PAD > {staged} ? PAD - {staged} : 0) / sizeof(T)];",
                )
            template, grid, threadgroup = tile_dense._nax_spec(eff, k, n, rows, dt)
        else:
            kernel_name = f"omlx_steel_dense_{face.key}"
            skeleton_header = tile_dense._STEEL_HEADER
            skeleton_body = tile_dense._STEEL_SOURCE
            tgp = eff.wm * eff.wn * 32
            cache_p = eff.param_staging is tuning.ParamStaging.GROUP_STEPS_COUNTER
            template = [
                ("T", dt),
                ("K", k),
                ("BM", eff.bm),
                ("BN", eff.bn),
                ("BK", eff.bk),
                ("WM", eff.wm),
                ("WN", eff.wn),
                ("ALIGN_N", eff.align_n),
                ("VEC_STORE", eff.vec_store),
                ("CACHE_P", cache_p),
                ("X_NREADS", eff.x_nreads),
                ("SPLIT_K", split_k_val),
                ("K_PART", k_part_val),
            ]
            grid = (tgp * n_cols, n_rows, split_k_val)
            threadgroup = (tgp, 1, 1)
        input_specs = [
            ("x", mtype_x, x_shape, _addr_space(x_shape)),
            *_operand_specs(layer, "w", face),
        ]

    elif chosen_route == "wide":
        if (
            tile_face is None
            or not tile_face.exact_for(dt)
            or k % 8 != 0
            or kernels.table_bytes(tile_face) > 32768
        ):
            raise ValueError(
                f"route 'wide' is illegal for {layer.mode} "
                f"(k={k}, n={n}, dtype={_dtype_label(dt)})"
            )
        face = tile_face
        backend = "wide"
        matrices = _make_matrices(face, target_selects=target_selects)
        input_names = tuple(["x", *matrices.add("w", face)])
        kernel_name = f"omlx_format_linear_wide_{face.key}"
        skeleton_header = tile_wide._HEADER
        skeleton_body = tile_wide._SOURCE
        tiles = (rows + 4) // 5
        # Mirror the live builder's template (single source of truth).
        template = tile_wide.template(dt, k, n, *tile_wide.layout(rows))
        grid = (32 * tiles, 2 * ((n + 7) // 8), 1)
        threadgroup = (32, 2, 1)
        input_specs = [
            ("x", mtype_x, x_shape, _addr_space(x_shape)),
            *_operand_specs(layer, "w", face),
        ]

    elif chosen_route == "mma":
        if tile_face is None or not tile_face.exact_for(dt):
            raise ValueError(
                f"route 'mma' requires a tile face exact for {_dtype_label(dt)}"
            )
        face = tile_face
        backend = "mma"
        m_pad = ((rows + 7) // 8) * 8
        if isinstance(scheduled_policy, MmaTilePolicy):
            pol = scheduled_policy
        else:
            cand = (
                tuning.lookup(
                    "mma_tile",
                    forced_table_face,
                    dtype=dt,
                    bits=layer.bits,
                    group=layer.group_size,
                    n=n,
                    k=k,
                    rows=rows,
                    chip=chip_id,
                )
                if forced_table_face is not None
                else None
            )
            pol = (
                cand
                if isinstance(cand, MmaTilePolicy)
                and cand.is_legal(face, k, n, m_pad, chip_id, dt)
                else MmaTilePolicy()
            )
        if not pol.is_legal(face, k, n, m_pad, chip_id, dt):
            raise ValueError(
                f"route 'mma' policy {pol!r} is illegal for "
                f"(k={k}, n={n}, m={m_pad}, chip={chip_id})"
            )
        policy_repr = repr(pol)
        skeleton_digest = tuning.skeleton_digest("mma_tile")
        ops = verify_skeleton.tile_mma_ops(face, policy=pol)
        matrices = _make_matrices(face, target_selects=target_selects)
        input_names = tuple(["x", *matrices.add("w", face), "N_size"])
        kernel_name = f"omlx_fl_mma_rt{m_pad // 8}_k{k}_{face.key}"
        skeleton_header = ops.header
        skeleton_body = verify_skeleton.emit_mma_source(ops, m_pad // 8, k)
        template = [("T", dt)]
        grid = (32 * pol.nsg, n // pol.bn, 1)
        threadgroup = (32 * pol.nsg, 1, 1)
        input_specs = [
            ("x", mtype_x, (m_pad, k), _addr_space((m_pad, k))),
            *_operand_specs(layer, "w", face),
            ("N_size", "uint32_t", (), "scalar"),
        ]
        out_specs = [("y", mtype_x, (m_pad, n))]

    elif chosen_route == "msg":
        if tile_face is None or not tile_face.exact_for(dt) or not (1 <= rows <= 8):
            raise ValueError(
                f"route 'msg' requires 1 <= rows <= 8 and tile face exact for "
                f"{_dtype_label(dt)}"
            )
        face = tile_face
        backend = "msg"
        m_pad = 4 if rows <= 4 else 6 if rows <= 6 else 8
        if isinstance(scheduled_policy, MsgTilePolicy):
            pol = scheduled_policy
        else:
            cand = (
                tuning.lookup(
                    "msg_tile",
                    forced_table_face,
                    dtype=dt,
                    bits=layer.bits,
                    group=layer.group_size,
                    n=n,
                    k=k,
                    rows=rows,
                    chip=chip_id,
                )
                if forced_table_face is not None
                else None
            )
            pol = (
                cand
                if isinstance(cand, MsgTilePolicy)
                and cand.is_legal(face, k, n, m_pad, chip_id, dt)
                else MsgTilePolicy()
            )
        if not pol.is_legal(face, k, n, m_pad, chip_id, dt):
            raise ValueError(
                f"route 'msg' policy {pol!r} is illegal for "
                f"(k={k}, n={n}, m={m_pad}, chip={chip_id})"
            )
        policy_repr = repr(pol)
        skeleton_digest = tuning.skeleton_digest("msg_tile")
        lt = getattr(pol.load_lifetime, "value", pol.load_lifetime) or "batch"
        ops = verify_skeleton.msg_tile_ops(face, m_pad, pol.nsg, lt)
        matrices = _make_matrices(face, target_selects=target_selects)
        input_names = tuple(["x", *matrices.add("w", face), "N_size"])
        kernel_name = f"omlx_vk_m{m_pad}_nsg{pol.nsg}_{lt}_{face.key}"
        skeleton_header = ""
        skeleton_body = verify_skeleton.emit_msg_source(ops)
        template = [("T", dt), ("K", k)]
        cols = 4 * pol.nsg
        grid = (32 * pol.nsg, (n + cols - 1) // cols, 1)
        threadgroup = (32 * pol.nsg, 1, 1)
        input_specs = [
            ("x", mtype_x, (m_pad, k), _addr_space((m_pad, k))),
            *_operand_specs(layer, "w", face),
            ("N_size", "uint32_t", (), "scalar"),
        ]
        out_specs = [("y", mtype_x, (m_pad, n))]

    elif chosen_route == "switch_dot":
        if dot_face is None or not dot_face.exact_for(dt) or k < 8 or n % 8 != 0:
            raise ValueError(
                f"route 'switch_dot' is illegal for {layer.mode} "
                f"(k={k}, n={n}, dtype={_dtype_label(dt)})"
            )
        face = dot_face
        backend = "dot"
        matrices = _make_matrices(face, target_selects=target_selects)
        input_names = tuple(["x", *matrices.add("m", face), "rhs"])
        kernel_name = f"omlx_format_switch_linear_{face.key}"
        skeleton_header = kernels.STORE_ROWS
        skeleton_body = linear._STACKED + linear._SOURCE.replace(
            "__MATRIX__", "rhs[pair]"
        ).replace("__X_ROW__", "pair / uint(PAIRS_PER_ROW)")
        pairs = rows * per_row
        template = [
            ("T", dt),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("PAIRS_PER_ROW", per_row),
        ]
        grid = (32, 2 * pairs, n // 8)
        threadgroup = (32, 2, 1)
        input_specs = [
            ("x", mtype_x, x_shape, _addr_space(x_shape)),
            *_operand_specs(layer, "m", face),
            ("rhs", "uint32_t", (pairs,), _addr_space((pairs,))),
        ]
        out_specs = [("y", mtype_x, (pairs, n))]

    elif chosen_route == "switch_tile_steel":
        if (
            tile_face is None
            or not tile_face.exact_for(dt)
            or k % 32 != 0
            or n <= 0
            or not experts
        ):
            raise ValueError(
                f"route 'switch_tile_steel' is illegal for {layer.mode} "
                f"(k={k}, n={n}, experts={experts}, dtype={_dtype_label(dt)})"
            )
        face = tile_face
        backend = "steel"
        mlx_hdr = tile_gather._headers()
        if mlx_hdr is None:
            raise RuntimeError("MLX steel headers unavailable")
        mlx_route_headers = mlx_hdr
        matrices = _make_matrices(face, target_selects=target_selects)
        input_names = tuple(["x", *matrices.add("w", face), "offsets", "meta"])
        kernel_name = f"omlx_steel_gather_{face.key}"
        skeleton_header = tile_gather._HEADER
        skeleton_body = tile_gather._SOURCE
        policy_repr = "SteelGather(BM=16, BN=32, BK=32)"
        template = [("T", dt), ("K", k), ("ALIGN_N", n % 32 == 0)]
        grid = (
            64 * ((n + 31) // 32),
            (rows + 15) // 16 + min(experts, rows),
            1,
        )
        threadgroup = (64, 1, 1)
        offsets_shape = (max(8, experts),)
        input_specs = [
            ("x", mtype_x, x_shape, _addr_space(x_shape)),
            *_operand_specs(layer, "w", face),
            ("offsets", "int32_t", offsets_shape, _addr_space(offsets_shape)),
            ("meta", "int32_t", (3,), "constant"),
        ]
        out_specs = [("y", mtype_x, (rows, 1, n))]

    else:  # switch_tile_nax
        if not tu:
            raise ValueError(
                f"route 'switch_tile_nax' requires tensor units "
                f"(unavailable on chip {chip_id})"
            )
        if (
            tile_face is None
            or not tile_face.exact_for(dt)
            or not experts
            or experts > nax._MAX_EXPERTS
            or n % 32 != 0
            or k % 32 != 0
        ):
            raise ValueError(
                f"route 'switch_tile_nax' is illegal for {layer.mode} "
                f"(k={k}, n={n}, experts={experts}, dtype={_dtype_label(dt)})"
            )
        face = tile_face
        backend = "nax"
        mlx_hdr = nax._read_mlx_headers(nax._MLX_MM_HEADERS)
        if mlx_hdr is None:
            raise RuntimeError("MLX NAX headers unavailable")
        mlx_route_headers = mlx_hdr
        suffix, body_src, extra = nax._FACE_KINDS["plain"]
        matrices = _make_matrices(face, target_selects=target_selects)
        input_names = tuple(
            ["x", *matrices.add("w", face), "tiles", "tile_count", "params", *extra]
        )
        kernel_name = f"omlx_gqmm_{face.key}{suffix}"
        skeleton_header = nax._MM_HEADER + nax._FACE_HEADER
        skeleton_body = body_src
        staged = matrices.staged_bytes
        if staged:
            skeleton_body = skeleton_body.replace(
                "PAD / sizeof(T)];",
                f"(PAD > {staged} ? PAD - {staged} : 0) / sizeof(T)];",
            )
        plan = nax.face_plan(rows, experts, k, n)
        policy_repr = repr(plan)
        max_tiles = (rows + plan.bm - 1) // plan.bm + min(experts, rows)
        g = nax._grid(n, max_tiles, plan)
        grid = g["grid"]
        threadgroup = g["threadgroup"]
        template = [
            ("T", dt),
            ("K", k),
            ("SCHED", int(plan.sched)),
            ("ALIGN_N", n % nax._BN == 0),
            ("BM", plan.bm),
            ("BK", plan.bk),
            ("GX", plan.gx),
            ("PAD", plan.pad),
        ]
        tiles_shape = (max_tiles * 4,)
        input_specs = [
            ("x", mtype_x, x_shape, _addr_space(x_shape)),
            *_operand_specs(layer, "w", face),
            ("tiles", "uint32_t", tiles_shape, _addr_space(tiles_shape)),
            ("tile_count", "uint32_t", (1,), "constant"),
            ("params", "int32_t", (1,), "constant"),
        ]
        out_specs = [("y", mtype_x, (rows, 1, n))]

    face_header = matrices.header
    matrix_declarations = matrices.declarations
    raw_header = mlx_route_headers + face_header + skeleton_header
    raw_source = matrix_declarations + skeleton_body
    raw_digest = hashlib.sha256(
        f"{kernel_name}\0{raw_header}\0{raw_source}".encode()
    ).hexdigest()

    threadgroups = (
        (grid[0] + threadgroup[0] - 1) // threadgroup[0],
        (grid[1] + threadgroup[1] - 1) // threadgroup[1],
        (grid[2] + threadgroup[2] - 1) // threadgroup[2],
    )
    launch = LaunchGeometry(
        grid=grid,
        threadgroup=threadgroup,
        threadgroups=threadgroups,
        template=tuple((tn, _template_record_value(tv)) for tn, tv in template),
        input_names=input_names,
        output_names=output_names,
        input_specs=tuple(input_specs),
        output_specs=tuple(out_specs),
        staged_table_bytes=int(matrices.staged_bytes),
        split_k=split_k_val,
        k_part=k_part_val,
    )

    metal, sections = _assemble_standalone_metal(
        kernel_name=kernel_name,
        route=chosen_route,
        mode=layer.mode,
        bits=int(layer.bits),
        group_size=int(layer.group_size),
        k=int(k),
        n=int(n),
        rows=int(rows),
        chip=chip_id,
        table_mode=table_mode,
        policy_repr=policy_repr,
        raw_digest=raw_digest,
        raw_source=raw_source,
        mlx_route_headers=mlx_route_headers,
        face_header=face_header,
        skeleton_header=skeleton_header,
        matrix_declarations=matrix_declarations,
        skeleton_body=skeleton_body,
        template=template,
        launch=launch,
        include_mlx_preamble=include_mlx_preamble,
    )
    metal_digest = hashlib.sha256(metal.encode()).hexdigest()

    return DumpMetalResult(
        kernel_name=kernel_name,
        route=chosen_route,
        backend=backend,
        mode=layer.mode,
        bits=int(layer.bits),
        group_size=int(layer.group_size),
        k=int(k),
        n=int(n),
        rows=int(rows),
        experts=experts,
        dtype=_dtype_label(dt),
        chip=chip_id,
        tensor_units=tu,
        table_mode=table_mode,
        face_key=face.key,
        face_digest=tuning.face_digest(face),
        skeleton_digest=skeleton_digest,
        policy_repr=policy_repr,
        launch=launch,
        raw_header=raw_header,
        raw_source=raw_source,
        raw_digest=raw_digest,
        metal=metal,
        metal_digest=metal_digest,
        sections=sections,
    )


def format_dump_metal_summary(result: DumpMetalResult, *, as_json: bool = False) -> str:
    """Format a ``DumpMetalResult`` summary (metadata, launch geometry, digests)."""
    if as_json:
        summary = {k: v for k, v in result.to_dict().items() if k != "metal"}
        return json.dumps(summary, indent=2, sort_keys=True)
    lg = result.launch
    lines = [
        f"Kernel: {result.kernel_name} (route={result.route}, "
        f"backend={result.backend}, face={result.face_key})",
        f"Layer: mode={result.mode} bits={result.bits} gs={result.group_size} "
        f"k={result.k} n={result.n} rows={result.rows}"
        + (f" experts={result.experts}" if result.experts is not None else ""),
        f"Target: chip={result.chip} tensor_units={result.tensor_units} "
        f"dtype={result.dtype} tables={result.table_mode}",
        f"Launch: grid={lg.grid} threadgroup={lg.threadgroup} "
        f"threadgroups={lg.threadgroups} staged_tables={lg.staged_table_bytes}B",
        f"Template: {list(lg.template)}",
        f"Digests: raw_digest={result.raw_digest}",
        f"         metal_digest={result.metal_digest}",
        f"         face_digest={result.face_digest} "
        f"skeleton_digest={result.skeleton_digest}",
        "Sections:",
    ]
    for sec in result.sections:
        lines.append(
            f"  - {sec.name}: lines {sec.start_line}-{sec.end_line} "
            f"sha256={sec.sha256[:12]}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Milestone 3 Commit 3 — bench (Rule-31 chained + C2 fp32_ref + memory delta)
# ---------------------------------------------------------------------------

_MLX_QUANT_BITS = (2, 3, 4, 5, 6, 8)
_MLX_QUANT_GROUPS = (32, 64, 128)


@dataclass(frozen=True)
class BenchResult:
    mode: str
    bits: int
    group_size: int
    k: int
    n: int
    rows: int
    dtype: str
    chip: str
    route: str
    backend: str
    chain_len: int
    units: int
    rounds: int
    warmup: int
    layer_median_us: float
    layer_p25_us: float
    layer_p75_us: float
    baseline_available: bool
    baseline_bits: int | None
    baseline_group_size: int | None
    baseline_note: str | None
    baseline_median_us: float | None
    baseline_p25_us: float | None
    baseline_p75_us: float | None
    ratio_of_medians: float | None
    paired_balanced_ratio: float | None
    position_factor: float | None
    layer_finite: bool
    layer_rel_l2_fp32: float
    layer_max_abs_fp32: float
    baseline_rel_l2_fp32: float | None
    c2_threshold: float | None
    c2_pass: bool
    bitwise_equal_baseline: bool | None
    weight_bytes_layer: int
    weight_bytes_baseline: int | None
    weight_bytes_delta: int | None
    peak_memory_layer_bytes: int
    peak_memory_baseline_bytes: int | None
    peak_memory_delta_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def _fit_chain(y: mx.array, k: int, norm_w: mx.array) -> mx.array:
    """Project output width ``N`` back to input width ``K`` + RMS norm (Rule 31)."""
    n = y.shape[-1]
    if n >= k:
        z = y[..., :k]
    else:
        reps = -(-k // n)
        z = mx.concatenate([y] * reps, axis=-1)[..., :k]
    return mx.fast.rms_norm(z.astype(norm_w.dtype), norm_w, 1e-6)


def _both_orders(
    base: Sequence[float],
    new: Sequence[float],
    new_first: Sequence[bool],
    units: int,
) -> tuple[list[float], list[float]]:
    """Balanced paired geometric-mean ratios and position factors (ab_common)."""
    ratios: list[float] = []
    positions: list[float] = []
    for start in range(0, len(base) - units + 1, 2 * units):
        for u in range(units):
            i, j = start + u, start + u + units
            if j >= len(base):
                break
            if new_first[i] == new_first[j]:
                raise ValueError(
                    f"samples {i} and {j} ran the same order: "
                    f"expected alternating orders across {units} units"
                )
            ratios.append(((new[i] / base[i]) * (new[j] / base[j])) ** 0.5)
            k_idx, m_idx = (i, j) if new_first[i] else (j, i)
            positions.append(
                ((base[k_idx] / new[k_idx]) * (new[m_idx] / base[m_idx])) ** 0.5
            )
    return ratios, positions


def _rel_l2(y: mx.array, ref: mx.array) -> float:
    """``||y - ref||_2 / max(||ref||_2, 1e-30)`` in float32."""
    y32 = y.astype(mx.float32).reshape(-1)
    ref32 = ref.astype(mx.float32).reshape(-1)
    num = mx.linalg.norm(y32 - ref32)
    den = mx.maximum(mx.linalg.norm(ref32), 1e-30)
    return float((num / den).item())


def _finite(y: mx.array) -> bool:
    return bool(mx.all(mx.isfinite(y.astype(mx.float32))).item())


def _c2_check(
    layer_out: mx.array, base_out: mx.array | None, ref: mx.array
) -> tuple[bool, float, float, float | None, float | None, bool]:
    """Contract C2 FP32 check: finite and ``rel_l2 <= 1.05 * base + 1e-6``."""
    is_fin = _finite(layer_out)
    rl = _rel_l2(layer_out, ref)
    max_abs = float(
        mx.max(mx.abs(layer_out.astype(mx.float32) - ref.astype(mx.float32))).item()
    )
    if base_out is None:
        return is_fin, rl, max_abs, None, None, bool(is_fin and math.isfinite(rl))
    rb = _rel_l2(base_out, ref)
    threshold = 1.05 * rb + 1e-6
    passed = bool(is_fin and math.isfinite(rl) and rl <= threshold)
    return is_fin, rl, max_abs, rb, threshold, passed


def _plain_weight_fp32(layer: FormatLinear) -> tuple[mx.array | None, Any]:
    """Decode ``layer`` weights in float32 plain basis.

    Returns ``(w_plain_fp32, fallback_tx_bound)`` where ``w_plain_fp32`` is
    ``None`` only when an ``InputTransform`` has no ``inverse_weight`` callable.
    """
    if layer.mode == "affine":
        w = mx.dequantize(
            layer["weight"],
            layer["scales"].astype(mx.float32),
            layer["biases"].astype(mx.float32),
            group_size=layer.group_size,
            bits=layer.bits,
            mode="affine",
        ).astype(mx.float32)
    else:
        w = layer._decoded().astype(mx.float32)

    if formats._declares_input_transform(layer.fmt):
        bound = formats._bound_transform(layer.fmt, layer, transforms)
        if bound is not None:
            tx, tx_tensors = bound
            if tx.inverse_weight is not None:
                return tx.inverse_weight(w, *tx_tensors).astype(mx.float32), None
            return None, (w, tx, tx_tensors)
    return w, None


def _lacks_inverse_weight(layer: FormatLinear) -> bool:
    if not formats._declares_input_transform(layer.fmt):
        return False
    bound = formats._bound_transform(layer.fmt, layer, transforms)
    return bound is not None and bound[0].inverse_weight is None


def _fp32_reference(layer: FormatLinear, x: mx.array) -> mx.array:
    w_plain, fallback = _plain_weight_fp32(layer)
    x32 = x.astype(mx.float32)
    if w_plain is not None:
        ref = x32 @ w_plain.T
    else:
        w_rot, tx, tx_tensors = fallback
        x_rot = tx.apply(x32, *tx_tensors).astype(mx.float32)
        ref = x_rot @ w_rot.T
    if "bias" in layer:
        ref = ref + layer["bias"].astype(mx.float32)
    return ref


def _raw_bits_equal(a: mx.array, b: mx.array) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype in (mx.bfloat16, mx.float16):
        return bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())
    return bool(mx.array_equal(a, b).item())


def _layer_weight_bytes(layer: FormatLinear) -> int:
    total = 0
    for val in dict.values(layer):
        if isinstance(val, mx.array):
            total += int(val.nbytes)
    return total


def _baseline_weight_bytes(base: nn.QuantizedLinear) -> int:
    total = 0
    for name in ("weight", "scales", "biases", "bias"):
        arr = base.get(name)
        if isinstance(arr, mx.array):
            total += int(arr.nbytes)
    return total


def _make_mlx_baseline(
    layer: FormatLinear,
    *,
    dtype: mx.Dtype,
    baseline_gs: int,
    exact_affine: bool,
) -> nn.QuantizedLinear:
    k, n = layer.input_dims, layer.output_dims
    has_bias = "bias" in layer
    ql = nn.QuantizedLinear(
        k,
        n,
        bias=has_bias,
        group_size=baseline_gs,
        bits=layer.bits,
        mode="affine",
    )
    if exact_affine:
        ql["weight"] = layer["weight"]
        ql["scales"] = layer["scales"]
        ql["biases"] = layer["biases"]
    else:
        w_plain, _ = _plain_weight_fp32(layer)
        if w_plain is None:
            raise ValueError(
                "input transform has no inverse_weight; plain-basis MLX "
                "baseline unavailable"
            )
        q, scales, biases = mx.quantize(
            w_plain.astype(dtype),
            group_size=baseline_gs,
            bits=layer.bits,
            mode="affine",
        )
        ql["weight"] = q
        ql["scales"] = scales
        ql["biases"] = biases
    if has_bias:
        ql["bias"] = layer["bias"]
    return ql


def _resolve_baselines(
    layers: tuple[FormatLinear, ...],
    explicit: nn.QuantizedLinear | Sequence[nn.QuantizedLinear] | None,
    dtype: mx.Dtype,
) -> tuple[tuple[nn.QuantizedLinear, ...] | None, int | None, int | None, str | None]:
    first = layers[0]
    k = first.input_dims
    if explicit is not None:
        base_list = (
            (explicit,) if isinstance(explicit, nn.QuantizedLinear) else tuple(explicit)
        )
        if len(base_list) != len(layers):
            raise ValueError(
                f"baselines length ({len(base_list)}) must match layers length "
                f"({len(layers)})"
            )
        for idx, b in enumerate(base_list):
            if not isinstance(b, nn.QuantizedLinear):
                raise TypeError(
                    f"baselines[{idx}] must be nn.QuantizedLinear, "
                    f"not {type(b).__name__}"
                )
        b0 = base_list[0]
        return (
            base_list,
            int(b0.bits),
            int(b0.group_size),
            "caller-supplied nn.QuantizedLinear baseline",
        )

    if first.bits not in _MLX_QUANT_BITS:
        return (
            None,
            None,
            None,
            f"MLX nn.QuantizedLinear does not support bits={first.bits}",
        )

    if (
        first.mode == "affine"
        and first.group_size in _MLX_QUANT_GROUPS
        and k % first.group_size == 0
    ):
        bases = tuple(
            _make_mlx_baseline(
                layer, dtype=dtype, baseline_gs=first.group_size, exact_affine=True
            )
            for layer in layers
        )
        return bases, int(first.bits), int(first.group_size), "exact affine weights"

    # Nearest legal MLX group_size: prefer layer.group_size if legal, else 64->32->128.
    candidates = [first.group_size] if first.group_size in _MLX_QUANT_GROUPS else []
    for gs in (64, 32, 128):
        if gs not in candidates:
            candidates.append(gs)
    chosen_gs = next((gs for gs in candidates if k % gs == 0), None)
    if chosen_gs is None:
        return (
            None,
            None,
            None,
            f"no legal MLX affine group_size in (64, 32, 128) divides k={k}",
        )

    if any(_lacks_inverse_weight(layer) for layer in layers):
        return (
            None,
            None,
            None,
            "input transform has no inverse_weight; plain-basis MLX baseline "
            "unavailable",
        )

    bases = tuple(
        _make_mlx_baseline(
            layer, dtype=dtype, baseline_gs=chosen_gs, exact_affine=False
        )
        for layer in layers
    )
    if chosen_gs != first.group_size:
        note = (
            f"non-MLX group_size={first.group_size}; compared against MLX affine "
            f"bits={first.bits} group_size={chosen_gs}"
        )
    else:
        note = (
            f"re-quantized {first.mode} plain weights to MLX affine "
            f"bits={first.bits} group_size={chosen_gs}"
        )
    return bases, int(first.bits), int(chosen_gs), note


def _percentile(values: list[float], frac: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = frac * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(len(ordered) - 1, lo + 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _measure_peak_working_bytes(fn, x: mx.array) -> int:
    mx.clear_cache()
    mx.reset_peak_memory()
    active = int(mx.get_active_memory())
    out = fn(x)
    mx.eval(out)
    peak = int(mx.get_peak_memory())
    del out
    mx.clear_cache()
    return max(0, peak - active)


def bench(
    layers: FormatLinear | Sequence[FormatLinear],
    *,
    rows: int = 1,
    dtype: str | mx.Dtype = mx.bfloat16,
    chain_len: int = 16,
    rounds: int = 40,
    warmup: int = 5,
    seed: int = 0,
    baselines: nn.QuantizedLinear | Sequence[nn.QuantizedLinear] | None = None,
    input_ndim: int = 3,
) -> BenchResult:
    """Rule-31 chained micro-benchmark with C2 FP32 check and memory delta."""
    if not mx.metal.is_available():
        raise RuntimeError("debug.bench requires an available Metal GPU")

    layer_list = (layers,) if isinstance(layers, FormatLinear) else tuple(layers)
    if not layer_list:
        raise ValueError("layers must be a FormatLinear or non-empty sequence")
    for idx, layer in enumerate(layer_list):
        if not isinstance(layer, FormatLinear):
            raise TypeError(
                f"layers[{idx}] must be FormatLinear, not {type(layer).__name__}"
            )
        layer.refuse_placeholders()

    first = layer_list[0]
    for idx, layer in enumerate(layer_list[1:], start=1):
        if (
            layer.input_dims != first.input_dims
            or layer.output_dims != first.output_dims
            or layer.mode != first.mode
            or layer.bits != first.bits
            or layer.group_size != first.group_size
        ):
            raise ValueError(
                f"layers[{idx}] shape/format mismatch with layers[0]: "
                f"({layer.mode}, {layer.bits}b, g{layer.group_size}, "
                f"k={layer.input_dims}, n={layer.output_dims}) != "
                f"({first.mode}, {first.bits}b, g{first.group_size}, "
                f"k={first.input_dims}, n={first.output_dims})"
            )

    _check_positive_int("rows", rows)
    _check_positive_int("chain_len", chain_len)
    if type(rounds) is not int or rounds < 2 or rounds % 2 != 0:
        raise ValueError(f"rounds must be an even int >= 2, not {rounds!r}")
    if type(warmup) is not int or warmup < 0:
        raise ValueError(f"warmup must be a non-negative int, not {warmup!r}")
    if type(seed) is not int:
        raise ValueError(f"seed must be an int, not {seed!r}")
    if type(input_ndim) is not int or input_ndim not in (2, 3):
        raise ValueError(f"input_ndim must be 2 or 3, not {input_ndim!r}")

    dt = parse_dtype(dtype)
    k, n = int(first.input_dims), int(first.output_dims)
    units = len(layer_list)

    exp = explain(first, rows=(rows,), dtype=dt, input_ndim=input_ndim)
    rt = exp.row_traces[0]
    chip_id = exp.chip

    base_list, base_bits, base_gs, base_note = _resolve_baselines(
        layer_list, baselines, dt
    )
    baseline_available = base_list is not None

    x_shape = (1, rows, k) if input_ndim == 3 else (rows, k)
    norm_w = mx.ones((k,), dtype=dt)
    xs = tuple(
        (mx.random.normal(x_shape, key=mx.random.key(seed + u)) * 0.5).astype(dt)
        for u in range(units)
    )
    mx.eval(norm_w, *xs)

    def run_chain(side_layers, start_u: int, x0: mx.array) -> mx.array:
        cur = x0
        for j in range(chain_len):
            y = side_layers[(start_u + j) % units](cur)
            cur = _fit_chain(y, k, norm_w) if j + 1 < chain_len else y
        mx.eval(cur)
        return cur

    # C2 accuracy + bitwise check on unit 0 before timing loop.
    layer_y0 = layer_list[0](xs[0])
    mx.eval(layer_y0)
    ref_y0 = _fp32_reference(layer_list[0], xs[0])
    mx.eval(ref_y0)
    base_y0: mx.array | None = None
    bitwise_eq: bool | None = None
    if base_list is not None:
        base_y0 = base_list[0](xs[0])
        mx.eval(base_y0)
        if first.mode == "affine" and base_gs == first.group_size:
            bitwise_eq = _raw_bits_equal(layer_y0, base_y0)

    (
        layer_finite,
        layer_rel_l2,
        layer_max_abs,
        base_rel_l2,
        c2_threshold,
        c2_pass,
    ) = _c2_check(layer_y0, base_y0, ref_y0)

    # Paired alternating-order chained timing (Rule 31).
    layer_samples: list[float] = []
    base_samples: list[float] = []
    order_flags: list[bool] = []

    for r in range(warmup + rounds):
        new_first = r % 2 == 0
        # One untimed pre-call per side per round to stabilize command queue.
        if new_first:
            run_chain(layer_list, 0, xs[0])
            if base_list is not None:
                run_chain(base_list, 0, xs[0])
        else:
            if base_list is not None:
                run_chain(base_list, 0, xs[0])
            run_chain(layer_list, 0, xs[0])

        for u in range(units):
            if new_first:
                t0 = time.perf_counter()
                run_chain(layer_list, u, xs[u])
                t_layer = (time.perf_counter() - t0) * 1e6 / chain_len
                if base_list is not None:
                    t1 = time.perf_counter()
                    run_chain(base_list, u, xs[u])
                    t_base = (time.perf_counter() - t1) * 1e6 / chain_len
            else:
                if base_list is not None:
                    t1 = time.perf_counter()
                    run_chain(base_list, u, xs[u])
                    t_base = (time.perf_counter() - t1) * 1e6 / chain_len
                t0 = time.perf_counter()
                run_chain(layer_list, u, xs[u])
                t_layer = (time.perf_counter() - t0) * 1e6 / chain_len

            if r >= warmup:
                layer_samples.append(t_layer)
                order_flags.append(new_first)
                if base_list is not None:
                    base_samples.append(t_base)

    layer_med = float(statistics.median(layer_samples))
    layer_p25 = _percentile(layer_samples, 0.25)
    layer_p75 = _percentile(layer_samples, 0.75)

    base_med: float | None = None
    base_p25: float | None = None
    base_p75: float | None = None
    ratio_med: float | None = None
    balanced_ratio: float | None = None
    pos_factor: float | None = None
    if base_list is not None:
        base_med = float(statistics.median(base_samples))
        base_p25 = _percentile(base_samples, 0.25)
        base_p75 = _percentile(base_samples, 0.75)
        ratio_med = layer_med / base_med if base_med > 0 else None
        ratios, positions = _both_orders(
            base_samples, layer_samples, order_flags, units
        )
        if ratios:
            balanced_ratio = float(statistics.median(ratios))
            pos_factor = float(statistics.median(positions))

    w_bytes_layer = _layer_weight_bytes(layer_list[0])
    w_bytes_base = (
        _baseline_weight_bytes(base_list[0]) if base_list is not None else None
    )
    w_bytes_delta = w_bytes_layer - w_bytes_base if w_bytes_base is not None else None

    peak_layer = _measure_peak_working_bytes(layer_list[0], xs[0])
    peak_base = (
        _measure_peak_working_bytes(base_list[0], xs[0])
        if base_list is not None
        else None
    )
    peak_delta = peak_layer - peak_base if peak_base is not None else None

    return BenchResult(
        mode=first.mode,
        bits=int(first.bits),
        group_size=int(first.group_size),
        k=k,
        n=n,
        rows=rows,
        dtype=_dtype_label(dt),
        chip=chip_id,
        route=rt.selected_route,
        backend=rt.selected_backend,
        chain_len=chain_len,
        units=units,
        rounds=rounds,
        warmup=warmup,
        layer_median_us=layer_med,
        layer_p25_us=layer_p25,
        layer_p75_us=layer_p75,
        baseline_available=baseline_available,
        baseline_bits=base_bits,
        baseline_group_size=base_gs,
        baseline_note=base_note,
        baseline_median_us=base_med,
        baseline_p25_us=base_p25,
        baseline_p75_us=base_p75,
        ratio_of_medians=ratio_med,
        paired_balanced_ratio=balanced_ratio,
        position_factor=pos_factor,
        layer_finite=layer_finite,
        layer_rel_l2_fp32=layer_rel_l2,
        layer_max_abs_fp32=layer_max_abs,
        baseline_rel_l2_fp32=base_rel_l2,
        c2_threshold=c2_threshold,
        c2_pass=c2_pass,
        bitwise_equal_baseline=bitwise_eq,
        weight_bytes_layer=w_bytes_layer,
        weight_bytes_baseline=w_bytes_base,
        weight_bytes_delta=w_bytes_delta,
        peak_memory_layer_bytes=peak_layer,
        peak_memory_baseline_bytes=peak_base,
        peak_memory_delta_bytes=peak_delta,
    )


def format_bench(result: BenchResult, *, as_json: bool = False) -> str:
    """Format a ``BenchResult`` as human-readable text or JSON."""
    if as_json:
        return json.dumps(result.to_dict(), indent=2, sort_keys=True)
    lines = [
        f"Bench: mode={result.mode} bits={result.bits} gs={result.group_size} "
        f"k={result.k} n={result.n} rows={result.rows} dtype={result.dtype}",
        f"Target: chip={result.chip} route={result.route} ({result.backend}) "
        f"chain_len={result.chain_len} units={result.units} "
        f"rounds={result.rounds} warmup={result.warmup}",
        f"Timing (layer): median={result.layer_median_us:.2f} us "
        f"(p25={result.layer_p25_us:.2f}, p75={result.layer_p75_us:.2f})",
    ]
    if result.baseline_available and result.baseline_median_us is not None:
        lines.append(
            f"Timing (base):  median={result.baseline_median_us:.2f} us "
            f"(bits={result.baseline_bits}, gs={result.baseline_group_size}, "
            f"{result.baseline_note})"
        )
        lines.append(
            f"Ratio: paired_balanced={result.paired_balanced_ratio:.4f} "
            f"median_ratio={result.ratio_of_medians:.4f} "
            f"position_factor={result.position_factor:.4f}"
        )
    else:
        lines.append(f"Baseline: unavailable ({result.baseline_note})")
    c2_status = "PASS" if result.c2_pass else "FAIL"
    lines.append(
        f"C2 FP32 ({c2_status}): finite={result.layer_finite} "
        f"layer_rel_l2={result.layer_rel_l2_fp32:.6g} "
        f"max_abs={result.layer_max_abs_fp32:.6g} "
        f"base_rel_l2={result.baseline_rel_l2_fp32} "
        f"threshold={result.c2_threshold}"
    )
    lines.append(
        f"Memory: weight={result.weight_bytes_layer} B "
        f"(delta={result.weight_bytes_delta} B), "
        f"peak_working={result.peak_memory_layer_bytes} B "
        f"(delta={result.peak_memory_delta_bytes} B)"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Milestone 3 Commit 4 — warm_report pipeline dedup & compile timing reporter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarmPipelineInfo:
    kind: str
    kernel_name: str
    template: tuple[tuple[str, str | int | bool], ...]
    requested_count: int
    already_built: bool
    compiled_now: bool

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class WarmReport:
    chip: str
    tensor_units: bool
    compile_executed: bool
    layers_scanned: int
    dense_layers: int
    switch_layers: int
    total_requested: int
    unique_pipelines: int
    deduped_count: int
    dedup_ratio: float
    already_built: int
    compiled_now: int
    plan_ms: float
    compile_ms: float
    dispatch_warm_ms: float
    total_ms: float
    runtime_proof_count_before: int
    runtime_proof_count_after: int
    runtime_proof_delta: int
    pipelines: tuple[WarmPipelineInfo, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def _resolve_warm_kernel_name(inst: Any, raw_name_map: dict[int, str]) -> str:
    """Resolve an ``_Instance``'s kernel name without reading ``.native.name``."""
    raw_name = raw_name_map.get(id(inst.kernel.native))
    if raw_name is not None:
        return raw_name
    for fk_key, fk_obj in tuple(kernels._FACE_KERNELS.items()):
        if fk_obj is inst.kernel:
            base_name, fk_inputs, fk_mats_frozen, *_ = fk_key
            fk_mats = dict(fk_mats_frozen)
            return "_".join(
                (
                    base_name,
                    *(fk_mats[name].key for name in fk_inputs if name in fk_mats),
                )
            )
    return "omlx_unknown_kernel"


def _register_raw_names_for_layer(
    layer: FormatLinear | FormatSwitchLinear,
    *,
    tensor_units: bool,
    raw_name_map: dict[int, str],
) -> None:
    k, n = layer.input_dims, layer.output_dims
    is_switch = isinstance(layer, FormatSwitchLinear)
    dot_face = formats.dot(layer, k, n) if k >= 8 else None
    if dot_face is not None:
        if is_switch:
            if n % 8 == 0:
                raw_name_map[id(linear._kernel(dot_face, True))] = (
                    f"omlx_format_switch_linear_{dot_face.key}"
                )
        else:
            if n % 8 == 0:
                raw_name_map[id(linear._kernel(dot_face, False))] = (
                    f"omlx_format_linear_{dot_face.key}"
                )
            if dot_face.vpt >= 1:
                raw_name_map[id(linear._shared_kernel(dot_face))] = (
                    f"omlx_format_linear_rows_{dot_face.key}"
                )
    if not is_switch:
        tile_face = formats.tile(layer, k, n)
        if tile_face is not None:
            variants: tuple[tuple[str, bool, str], ...] = (
                (
                    ("nax", False, tile_dense._nax_body(False)[0]),
                    ("nax", True, tile_dense._nax_body(True)[0]),
                    ("steel", False, "omlx_steel_dense"),
                )
                if tensor_units
                else (("steel", False, "omlx_steel_dense"),)
            )
            for backend, interleaved, prefix in variants:
                with tile_dense._lock:
                    mm = (
                        tile_dense._build(tile_face, backend, True)
                        if interleaved
                        else tile_dense._build(tile_face, backend)
                    )
                if mm is not None:
                    raw_name_map[id(mm)] = f"{prefix}_{tile_face.key}"
            raw_name_map[id(tile_dense._splitk_accum_kernel())] = (
                "omlx_dense_splitk_accum"
            )


def _format_proof_task_template(metal: Any) -> tuple[tuple[str, str | int | bool], ...]:
    if isinstance(metal, tuple):
        out: list[tuple[str, str | int | bool]] = []
        for idx, item in enumerate(metal):
            if isinstance(item, mx.Dtype):
                out.append((f"arg{idx}", _metal_type(item)))
            elif isinstance(item, bool):
                out.append((f"arg{idx}", item))
            elif isinstance(item, int):
                out.append((f"arg{idx}", int(item)))
            else:
                out.append((f"arg{idx}", repr(item)))
        return tuple(out)
    return (("metal", repr(metal)),)


def warm_report(
    target: nn.Module | Sequence[FormatLinear | FormatSwitchLinear],
    *,
    dtypes: Sequence[str | mx.Dtype] | None = None,
    compile: bool = True,
    warm_dispatch: bool = False,
    per_rows: Sequence[int] = (1,),
    prefill_lengths: Sequence[int] = (2047, 2048),
    decode_batch_sizes: Sequence[int] = (1, 2, 4),
    mtp_enabled: bool = False,
    verify_rows: Sequence[int] = (2, 3, 4),
    isolated_proofs: bool = False,
) -> WarmReport:
    """Plan, deduplicate, and optionally compile warm-up Metal pipelines."""
    if not isinstance(compile, bool):
        raise TypeError(f"compile must be a bool, not {type(compile).__name__}")
    if not isinstance(warm_dispatch, bool):
        raise TypeError(
            f"warm_dispatch must be a bool, not {type(warm_dispatch).__name__}"
        )
    if not isinstance(mtp_enabled, bool):
        raise TypeError(f"mtp_enabled must be a bool, not {type(mtp_enabled).__name__}")
    if not isinstance(isolated_proofs, bool):
        raise TypeError(
            f"isolated_proofs must be a bool, not {type(isolated_proofs).__name__}"
        )

    per_rows_tuple = tuple(per_rows)
    if not per_rows_tuple:
        raise ValueError("per_rows must be a non-empty sequence of positive integers")
    for pr in per_rows_tuple:
        _check_positive_int("per_rows entry", pr)

    explicit_dtypes: tuple[mx.Dtype, ...] | None = None
    if dtypes is not None:
        parsed = tuple(dict.fromkeys(parse_dtype(dt) for dt in dtypes))
        if not parsed or any(dt not in linear._WARM_DTYPES for dt in parsed):
            raise ValueError(
                "warm_report dtypes must be non-empty and contain only bf16/fp16"
            )
        explicit_dtypes = parsed

    if compile and not mx.metal.is_available():
        raise RuntimeError(
            "debug.warm_report(compile=True) requires an available Metal GPU"
        )

    # Scan target for FormatLinear / FormatSwitchLinear layers.
    model: nn.Module | None = None
    layers: list[FormatLinear | FormatSwitchLinear] = []
    seen_ids: set[int] = set()
    if isinstance(target, (FormatLinear, FormatSwitchLinear)):
        model = target
        layers.append(target)
    elif isinstance(target, nn.Module):
        model = target
        for _, mod in target.named_modules():
            if isinstance(mod, (FormatLinear, FormatSwitchLinear)) and (
                id(mod) not in seen_ids
            ):
                seen_ids.add(id(mod))
                layers.append(mod)
    elif isinstance(target, Sequence) and not isinstance(target, (str, bytes)):
        for idx, item in enumerate(target):
            if not isinstance(item, (FormatLinear, FormatSwitchLinear)):
                raise TypeError(
                    f"target[{idx}] must be FormatLinear or FormatSwitchLinear, "
                    f"not {type(item).__name__}"
                )
            layers.append(item)
    else:
        raise TypeError(
            "warm_report target must be an nn.Module, format layer, or sequence "
            f"of format layers, not {type(target).__name__}"
        )

    if not layers:
        raise ValueError(
            "warm_report target contains no FormatLinear or FormatSwitchLinear layers"
        )
    for layer in layers:
        layer.refuse_placeholders()

    chip_id = tuning.current_chip()
    tensor_units = linear._nax_host()
    dense_count = sum(1 for item in layers if isinstance(item, FormatLinear))
    switch_count = sum(1 for item in layers if isinstance(item, FormatSwitchLinear))

    fp16_recast = bool(getattr(model, "_omlx_fp16_recast", False)) if model else False
    model_dtype = (
        None
        if explicit_dtypes is not None or fp16_recast or model is None
        else linear._infer_model_dtype(model)
    )

    t_total0 = time.perf_counter()
    proof_before = kernels.proof_count()
    ctx = testing.fresh_proofs() if isolated_proofs else nullcontext()

    with ctx:
        t_plan0 = time.perf_counter()
        raw_name_map: dict[int, str] = {}
        instances: dict[Any, Any] = {}
        proofs: dict[Any, list[tuple[Any, Any]]] = {}
        inst_counts: dict[Any, int] = {}

        unique_tasks: dict[tuple[int, Any], tuple[Any, Any, Any, str]] = {}
        task_counts: dict[tuple[int, Any], int] = {}
        total_requested = 0

        for layer in layers:
            k, n = layer.input_dims, layer.output_dims
            if k < 8 or n <= 0:
                continue
            target_dtypes = linear._layer_warm_dtypes(
                layer, model_dtype, explicit_dtypes, fp16_recast=fp16_recast
            )
            if not target_dtypes:
                continue
            _register_raw_names_for_layer(
                layer, tensor_units=tensor_units, raw_name_map=raw_name_map
            )
            layer_inst: dict[Any, Any] = {}
            layer_proofs: dict[Any, list[tuple[Any, Any]]] = {}
            layer_tasks: list[tuple[Any, Any, Any]] = []
            if isinstance(layer, FormatSwitchLinear):
                tile_face = formats.tile(layer, k, n)
                task_kname = (
                    (
                        f"omlx_gqmm_{tile_face.key}"
                        if tensor_units
                        else f"omlx_steel_gather_{tile_face.key}"
                    )
                    if tile_face is not None
                    else "omlx_switch_tile"
                )
                linear._plan_switch_warm(
                    layer,
                    k,
                    target_dtypes,
                    per_rows_tuple,
                    linear._NO_EPILOGUE,
                    layer_inst,
                    layer_proofs,
                    layer_tasks,
                )
            else:
                task_kname = "omlx_dense_tile"
                linear._plan_dense_warm(
                    layer,
                    layer.fmt,
                    layer.bits,
                    layer.group_size,
                    k,
                    target_dtypes,
                    layer_inst,
                    layer_proofs,
                )

            for inst, recipe in layer_inst.items():
                total_requested += 1
                inst_counts[inst] = inst_counts.get(inst, 0) + 1
                instances.setdefault(inst, recipe)
                for proof_obj, metal in layer_proofs.get(inst, ()):
                    linear._record_proof(proofs, inst, proof_obj, metal)

            for proof_obj, metal, build in layer_tasks:
                total_requested += 1
                task_key = (id(proof_obj), metal)
                task_counts[task_key] = task_counts.get(task_key, 0) + 1
                unique_tasks.setdefault(task_key, (proof_obj, metal, build, task_kname))

        # Snapshot already_built BEFORE submitting any warm futures.
        inst_built_before = {inst: (inst in inst.kernel._built) for inst in instances}
        task_built_before = {
            task_key: (metal in proof_obj.ran)
            for task_key, (proof_obj, metal, _b, _kn) in unique_tasks.items()
        }
        plan_ms = (time.perf_counter() - t_plan0) * 1000.0

        compile_ms = 0.0
        dispatch_warm_ms = 0.0
        if compile:
            t_comp0 = time.perf_counter()
            futures: list[Any] = []
            if instances:
                futures.extend(kernels.warm(linear._attach_proofs(instances, proofs)))
            if unique_tasks:
                futures.extend(
                    kernels.warm_proofs(
                        [(p, m, b) for p, m, b, _ in unique_tasks.values()]
                    )
                )
            # Await EVERY future before re-raising the first build error (Rule 14).
            first_err: BaseException | None = None
            for fut in futures:
                try:
                    fut.result()
                except BaseException as exc:  # noqa: BLE001
                    if first_err is None:
                        first_err = exc
            if first_err is not None:
                raise first_err
            compile_ms = (time.perf_counter() - t_comp0) * 1000.0

            if warm_dispatch:
                dispatch_model = model
                if dispatch_model is None:
                    dispatch_model = nn.Module()
                    for idx, lyr in enumerate(layers):
                        setattr(dispatch_model, f"layer_{idx}", lyr)
                t_disp0 = time.perf_counter()
                linear.warm_model_dispatch(
                    dispatch_model,
                    dtypes=explicit_dtypes,
                    prefill_lengths=tuple(prefill_lengths),
                    decode_batch_sizes=tuple(decode_batch_sizes),
                    mtp_enabled=mtp_enabled,
                    verify_rows=tuple(verify_rows),
                )
                dispatch_warm_ms = (time.perf_counter() - t_disp0) * 1000.0

        proof_after = kernels.proof_count()

    total_ms = (time.perf_counter() - t_total0) * 1000.0

    pipelines: list[WarmPipelineInfo] = []
    for inst in instances:
        was_built = inst_built_before[inst]
        tmpl = tuple(
            (name, _template_record_value(val)) for name, _kind, val in inst.template
        )
        pipelines.append(
            WarmPipelineInfo(
                kind="instance",
                kernel_name=_resolve_warm_kernel_name(inst, raw_name_map),
                template=tmpl,
                requested_count=inst_counts[inst],
                already_built=was_built,
                compiled_now=bool(compile and not was_built),
            )
        )
    for task_key, (_p, metal, _b, task_kname) in unique_tasks.items():
        was_built = task_built_before[task_key]
        pipelines.append(
            WarmPipelineInfo(
                kind="proof_task",
                kernel_name=task_kname,
                template=_format_proof_task_template(metal),
                requested_count=task_counts[task_key],
                already_built=was_built,
                compiled_now=bool(compile and not was_built),
            )
        )

    unique_count = len(pipelines)
    deduped = total_requested - unique_count
    dedup_ratio = (deduped / total_requested) if total_requested > 0 else 0.0
    already_built_total = sum(1 for p in pipelines if p.already_built)
    compiled_now_total = sum(1 for p in pipelines if p.compiled_now)

    return WarmReport(
        chip=chip_id,
        tensor_units=tensor_units,
        compile_executed=compile,
        layers_scanned=len(layers),
        dense_layers=dense_count,
        switch_layers=switch_count,
        total_requested=total_requested,
        unique_pipelines=unique_count,
        deduped_count=deduped,
        dedup_ratio=dedup_ratio,
        already_built=already_built_total,
        compiled_now=compiled_now_total,
        plan_ms=plan_ms,
        compile_ms=compile_ms,
        dispatch_warm_ms=dispatch_warm_ms,
        total_ms=total_ms,
        runtime_proof_count_before=proof_before,
        runtime_proof_count_after=proof_after,
        runtime_proof_delta=proof_after - proof_before,
        pipelines=tuple(pipelines),
    )


def format_warm_report(report: WarmReport, *, as_json: bool = False) -> str:
    """Format a ``WarmReport`` as human-readable text or JSON."""
    if as_json:
        return json.dumps(report.to_dict(), indent=2, sort_keys=True)
    lines = [
        f"WarmReport: chip={report.chip} tensor_units={report.tensor_units} "
        f"compile_executed={report.compile_executed}",
        f"Layers: scanned={report.layers_scanned} dense={report.dense_layers} "
        f"switch={report.switch_layers}",
        f"Pipelines: total_requested={report.total_requested} "
        f"unique={report.unique_pipelines} deduped={report.deduped_count} "
        f"({report.dedup_ratio * 100:.1f}%) "
        f"already_built={report.already_built} compiled_now={report.compiled_now}",
        f"Timing: plan={report.plan_ms:.2f} ms compile={report.compile_ms:.2f} ms "
        f"dispatch={report.dispatch_warm_ms:.2f} ms total={report.total_ms:.2f} ms",
        f"Runtime proofs: before={report.runtime_proof_count_before} "
        f"after={report.runtime_proof_count_after} "
        f"delta={report.runtime_proof_delta}",
    ]
    for p in report.pipelines:
        lines.append(
            f"  - [{p.kind}] {p.kernel_name} req={p.requested_count} "
            f"built={p.already_built} compiled_now={p.compiled_now} "
            f"tmpl={list(p.template)}"
        )
    return "\n".join(lines)
