"""Load oQh checkpoints through the regular mlx-lm / mlx-vlm loaders.

oQh tensors carry no ``.scales``, so the stock loaders leave their modules as
plain ``Linear`` / ``SwitchLinear``. ``nn.Module.load_weights`` is wrapped once:
when the weights hold ``.oqh_signs`` tensors, the matching modules are swapped
for oQh modules before the weights load, then producer fusion is installed.
Weights without oQh tensors take the original path unchanged.
"""

from __future__ import annotations

import logging
import threading
import types

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from ..custom_kernels.oqh import fast
from . import format as fmt
from . import runtime
from .modules import from_tensors

logger = logging.getLogger(__name__)

_SIGNS = ".oqh_signs"
_installed = False
_INSTALL_LOCK = threading.Lock()


def _format_module(
    prefix: str,
    weights: dict[str, mx.array],
    readers: dict[tuple[str, int], object] | None = None,
) -> nn.Module:
    """The checkpoint's oQh tensors as a built-in format layer."""
    from ..formats import oqh as oqh_format
    from ..formats.linear import FormatLinear, FormatSwitchLinear

    signs = weights[f"{prefix}.oqh_signs"]
    if signs.ndim < 1 or signs.size == 0:
        raise ValueError(f"oQh tensor {prefix}: invalid oqh_signs shape {signs.shape}")
    block = int(signs.shape[-1])
    in_dims = int(signs.size)
    weight = weights.get(f"{prefix}.weight")
    if weight is None:
        raise ValueError(f"oQh tensor {prefix}: missing weight")
    if (int(weight.shape[-1]) * 32) % in_dims != 0:
        raise ValueError(
            f"oQh tensor {prefix}: weight shape {weight.shape} does not match "
            f"oqh_signs {signs.shape}"
        )
    bits = int(weight.shape[-1]) * 32 // in_dims
    bias = weights.get(f"{prefix}.bias")
    scales = weights.get(f"{prefix}.oqh_scales")
    mode = "oqh_affine" if scales is not None else "oqh"
    reader = None if readers is None else readers.get((mode, block))
    if reader is None:
        reader = (
            oqh_format.OqhAffineFormat()
            if scales is not None
            else oqh_format.OqhFormat()
        )
        reader.block = block
        if readers is not None:
            readers[(mode, block)] = reader
    if scales is not None:
        biases = weights.get(f"{prefix}.oqh_biases")
        if biases is None:
            raise ValueError(f"oQh tensor {prefix}: missing oqh_biases")
        if scales.shape[-1] == 0 or in_dims % int(scales.shape[-1]) != 0:
            raise ValueError(
                f"oQh tensor {prefix}: invalid oqh_scales shape {scales.shape} "
                f"for input_dims {in_dims}"
            )
        gs = in_dims // int(scales.shape[-1])
        tensors = {
            "weight": weight,
            "oqh_scales": scales,
            "oqh_biases": biases,
            "oqh_signs": signs,
        }
    else:
        rscale = weights.get(f"{prefix}.rscale")
        if rscale is None:
            raise ValueError(f"oQh tensor {prefix}: missing rscale")
        gs = 0
        tensors = {
            "weight": weight,
            "rscale": rscale,
            "oqh_signs": signs,
        }
    layer_cls = FormatSwitchLinear if weight.ndim == 3 else FormatLinear
    return layer_cls(reader, bits, gs, in_dims, tensors, bias)


def _verified_format_family(config: dict) -> bool:
    """Whether this checkpoint's family is D10-verified and excludes in-place TP."""
    if not isinstance(config, dict):
        return False
    text = config.get("text_config")
    name = str(config.get("model_type") or "").strip().lower().replace("-", "_")
    nested = (
        str((text or {}).get("model_type") or "").strip().lower().replace("-", "_")
        if isinstance(text, dict)
        else ""
    )
    family = name or nested
    if family.endswith("_text"):
        family = family[: -len("_text")]
    return family not in fmt.TENSOR_PARALLEL_TYPES and family in (
        "qwen4_exp",
        "qwen3_5",
        "qwen3_5_moe",
    )


def _swap_modules(
    model: nn.Module, weights: dict[str, mx.array]
) -> tuple[int, bool, list]:
    prefixes = [k[: -len(_SIGNS)] for k in weights if k.endswith(_SIGNS)]
    if not prefixes:
        return 0, False, []
    from ..formats.linear import FormatLinear, FormatSwitchLinear
    from ..patches import format_load

    leaves = dict(
        tree_flatten(model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module))
    )
    pending = []
    for prefix in prefixes:
        if prefix not in leaves:
            raise ValueError(f"oQh tensor {prefix} has no matching module")
        # A format-written checkpoint already built FormatLinear/FormatSwitchLinear
        # layers in format_load; leave them intact so checkpoint.built stays valid.
        if isinstance(leaves[prefix], (FormatLinear, FormatSwitchLinear)):
            continue
        pending.append(prefix)
    if not pending:
        return 0, False, []

    checkpoint = format_load._current()
    use_format = False
    if (
        checkpoint is not None
        and checkpoint.active
        and getattr(checkpoint, "_allow_stock_switch", True)
        and getattr(format_load._SHARDED, "entry", None) is None
    ):
        try:
            config = checkpoint._read_config()
        except (OSError, ValueError):
            config = None
        if isinstance(config, dict) and _verified_format_family(config):
            from .. import formats, oq

            if oq._unverified_family(config) is None:
                require_native(config)
                formats.load()
                use_format = True
                # This stock oQh checkpoint's family passed D10: the format
                # layers the bridge builds are the verified path.
                checkpoint.verified = True
                checkpoint.verified_modes = checkpoint.verified_modes | frozenset(
                    {"oqh", "oqh_affine"}
                )
    readers: dict[tuple[str, int], object] = {}
    swapped = {}
    built = []
    for prefix in pending:
        mod = (
            _format_module(prefix, weights, readers)
            if use_format
            else from_tensors(prefix, weights)
        )
        swapped[prefix] = mod
        if use_format:
            built.append(mod)
            layout = mod.fmt.layout(
                mod.input_dims, mod.output_dims, mod.bits, mod.group_size
            )
            format_load._warm_layer(
                mod,
                mod.fmt,
                layout,
                mod.input_dims,
                {n: mod[n] for n in formats.all_tensor_names(mod.fmt)},
                checkpoint,
            )
    model.update_modules(tree_unflatten(list(swapped.items())))
    return len(swapped), use_format, built


def _through_load_weights(*args, _omlx_oqh_load, **kwargs):
    """In-place wrapper code for ``nn.Module.load_weights`` so subclasses that
    captured ``cls.load_weights`` earlier still reach the oQh load hook."""
    return _omlx_oqh_load(*args, **kwargs)


def install_load_hook() -> None:
    """Make load_weights understand oQh tensors (idempotent, process wide)."""
    global _installed
    with _INSTALL_LOCK:
        if _installed:
            return
        target = nn.Module.load_weights
        if getattr(target, "_omlx_oqh_load_hook", False):
            _installed = True
            return
        if target.__closure__ is None:
            original = types.FunctionType(
                target.__code__,
                target.__globals__,
                target.__name__,
                target.__defaults__,
                None,
            )
            original.__kwdefaults__ = target.__kwdefaults__
        else:
            original = target

        def load_weights(self, file_or_weights, strict: bool = True):
            weights = file_or_weights
            if isinstance(weights, str):
                weights = list(mx.load(weights).items())
            as_dict = dict(weights) if not isinstance(weights, dict) else weights
            if not any(k.endswith(_SIGNS) for k in as_dict):
                return original(self, weights, strict=strict)
            count, use_format, built = _swap_modules(self, as_dict)
            result = original(self, list(as_dict.items()), strict=strict)
            if not count:
                return result
            if use_format:
                for layer in built:
                    layer.refuse_placeholders()
                from ..formats import transforms

                layers = transforms.install_producers(self)
                logger.info(
                    "oQh (format): %d modules loaded, producer fusion in %d layers",
                    count,
                    layers,
                )
            else:
                layers = runtime.install(self)
                logger.info(
                    "oQh: %d modules loaded, producer fusion in %d layers",
                    count,
                    layers,
                )
            return result

        if target.__closure__ is None:
            target.__wrapped__ = original
            target.__kwdefaults__ = {
                **(target.__kwdefaults__ or {}),
                "_omlx_oqh_load": load_weights,
            }
            target.__code__ = _through_load_weights.__code__
            target._omlx_oqh_load_hook = True
        else:
            load_weights._omlx_oqh_load_hook = True
            nn.Module.load_weights = load_weights
        _installed = True


def require_native(config: dict) -> None:
    """Refuse an oQh checkpoint up front when this build cannot serve it."""
    if not fmt.is_oqh_config(config):
        return
    block = config[fmt.CONFIG_KEY]
    version = block.get("version") if isinstance(block, dict) else None
    # ValueError: engines pass it through as a load refusal.
    if version != fmt.FORMAT_VERSION:
        raise ValueError(
            f"This checkpoint uses oQh format version {version}; this oMLX reads "
            f"version {fmt.FORMAT_VERSION}"
        )
    if not fast.has_native():
        raise ValueError(
            "This oQh checkpoint needs the oQh native kernels, which failed to load: "
            f"{fast.import_error()}"
        )
