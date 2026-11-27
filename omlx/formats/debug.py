# SPDX-License-Identifier: Apache-2.0
"""Debug, inspect, and profile tools for oMLX quant formats.

This module is strictly opt-in (Rule 28: never imported on hot paths), reads
zero environment variables (Rule 30), and holds zero mutable module-global
caches (Rule 14).
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import mlx.core as mx

from .. import formats
from ..patches import m5_gather_qmm_nax as nax
from . import (
    Codes,
    Dot,
    QuantFormat,
    Tile,
    affine,
    kernels,
    linear,
    testing,
    tile_dense,
    transforms,
    tuning,
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
        # Uniform row-scale quantization matching OqhFormat.decode.
        max_level = (1 << bits) - 1
        center = max_level / 2.0
        w_f32 = w_rot.astype(mx.float32)
        row_max = mx.maximum(mx.max(mx.abs(w_f32), axis=-1), 1e-6)
        rscale = (row_max / center).astype(dtype)
        levels = mx.clip(
            mx.round(w_f32 / rscale[..., None].astype(mx.float32) + center),
            0,
            max_level,
        ).astype(mx.uint32)
        # Pack via mx.quantize on affine with scale=1, bias=0 -> exact packed codes.
        # Use manual bit packing into uint32 words of k * bits // 32.
        q_packed, _, _ = mx.quantize(
            levels.astype(mx.float32), group_size=32, bits=bits, mode="affine"
        )
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
            if isinstance(w, mx.array) and isinstance(s, mx.array):
                if min(w.size, s.size) < 8:
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
            elif route_name in ("mma_tile", "msg_tile"):
                if not pol.is_legal(actual_face, k, n, bound, chip, dtype):
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
            return RowRouteTrace(
                rows=r,
                schedule_bucket=bucket,
                selected_route=route.backend,
                selected_backend=route.backend,
                kernel_name=f"omlx_{route.backend}_dense_{tile_face.key}",
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
        if tile_face is not None and dt in (mx.float16, mx.bfloat16):
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
