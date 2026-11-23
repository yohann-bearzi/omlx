"""The verification of a quantization in a quant format: what it checks the
written checkpoint against.

The reference is the source's own: its next-token distribution at each
position of a fixed prompt, kept as the top tokens and the mass left out,
computed by the family's own forward over the lazily loaded source, one
decoder layer resident at a time.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

logger = logging.getLogger(__name__)

# The prompt's length, and the tokens kept of each position's distribution.
REFERENCE_TOKENS = 64
REFERENCE_TOP = 16
# How a reference is made: a reference of another version is not reused.
REFERENCE_VERSION = 1
# The files a checkpoint's tokenizer is read from.
_TOKENIZER_FILES = frozenset(
    (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
    )
)


_STREAMED_CALLS = threading.RLock()


@contextmanager
def layer_by_layer(layers) -> Iterator[None]:
    """While the block runs, each of ``layers`` evaluates its output and lets
    its weights go once it has run: a forward over a lazily loaded model
    holds one decoder layer's weights at a time. A layer that runs again in
    the block is refused (its weights are gone). Other modules of the same
    classes run as they do."""
    from .formats.linear import _FormatLayer
    from .patches.module_cache import clear_cached

    ours = {id(layer): layer for layer in layers}
    ran: set[int] = set()
    classes = {type(layer) for layer in ours.values()}

    def streamed(cls, original):
        def call(self, *args, **kwargs):
            # A subclass may still need its weights after calling super().
            if type(self) is not cls or id(self) not in ours:
                return original(self, *args, **kwargs)
            if id(self) in ran:
                raise RuntimeError(
                    f"decoder layer {type(self).__name__} ran twice in one "
                    "streamed forward: its weights were let go after its first run"
                )
            out = original(self, *args, **kwargs)
            mx.eval(
                [leaf for _, leaf in tree_flatten(out) if isinstance(leaf, mx.array)]
            )
            ran.add(id(self))
            empty = mx.zeros((0,))
            self.update(tree_map(lambda _: empty, self.parameters()))
            # Format tensors reject a change of shape while the original is
            # present, and per-module caches hold strong references to them.
            for module in self.modules():
                clear_cached(module)
                if isinstance(module, _FormatLayer):
                    for name in module.keys() & module._tensors:
                        del module[name]
                        module[name] = empty
            return out

        return call

    # The methods belong to the classes. Each pass takes its snapshot after
    # the previous pass has restored them, including inherited methods.
    with _STREAMED_CALLS:
        own = {}
        try:
            for cls in classes:
                saved = cls.__dict__.get("__call__")
                cls.__call__ = streamed(cls, cls.__call__)
                own[cls] = saved
            yield
        finally:
            for cls, call in own.items():
                if call is None:
                    del cls.__call__
                else:
                    cls.__call__ = call


def reference_of(model, prompt: mx.array, layers) -> dict[str, np.ndarray]:
    """``model``'s reference over ``prompt`` (one sequence of token ids), its
    ``layers`` streamed: per position the ``REFERENCE_TOP`` most likely next
    tokens (``ids``), their log-probabilities (``logprobs``) and the
    log-probability of all others (``tail``), with the ``prompt``."""
    with layer_by_layer(layers):
        out = model(prompt[None])
    logits = getattr(out, "logits", out)[0].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    order = mx.argsort(-logprobs, axis=-1)
    ranked = mx.take_along_axis(logprobs, order, axis=-1)
    reference = {
        "prompt": prompt.astype(mx.int32),
        "ids": order[:, :REFERENCE_TOP].astype(mx.int32),
        "logprobs": ranked[:, :REFERENCE_TOP],
        "tail": mx.logsumexp(ranked[:, REFERENCE_TOP:], axis=-1),
    }
    return {name: np.array(value) for name, value in reference.items()}


def _take_reference(
    path: Path, signature: dict, verify_dir: Path | None, staged: Path | None = None
) -> bool:
    """Reuse a matching archive or publish a completed one. Pin the same pair
    into the writer's output while other publishers are excluded."""
    from .oq import _atomic_write_json, _file_sha256

    signed = path.with_suffix(".json")
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(directory, fcntl.LOCK_EX)
        try:
            metadata = json.loads(signed.read_text())
            matches = metadata == {**signature, "archive_sha256": _file_sha256(path)}
        except (OSError, ValueError):
            matches = False
        if not matches:
            if staged is None:
                return False
            metadata = {**signature, "archive_sha256": _file_sha256(staged)}
            staged.replace(path)
            # An interruption between the replacements leaves a digest mismatch.
            _atomic_write_json(signed, metadata)
        if verify_dir is not None:
            verify_dir.mkdir()
            shutil.copy2(path, verify_dir / "reference.npz")
            shutil.copy2(signed, verify_dir / "reference.json")
        return True
    finally:
        os.close(directory)


def source_reference(
    model_path,
    config: dict,
    cache_dir: Path,
    *,
    trust_remote_code: bool = False,
    verify_dir: Path | None = None,
    name: str | None = None,
) -> Path:
    """The file of the source's reference: the one in ``cache_dir`` when it
    is of this source (its config and weight files), calibration corpus,
    tokenizer and version, found without loading any of them; else made by
    one streamed pass over the source (its time logged). A writer supplies
    its private verification directory to keep a snapshot of this generation.
    ``name`` replaces ``<source>`` in ``<name>.reference.{npz,json}``; the
    collector keeps the default, so a writer's pair cannot be rewritten by a
    calibration beside it."""
    if name is not None and (
        not name
        or name in (".", "..")
        or "\x00" in name
        or Path(name).name != name
    ):
        raise ValueError(f"reference name must be a single file stem, not {name!r}")
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    from . import oq

    source = Path(model_path)
    # Of what the reference is made, read without loading anything.
    made_of = oq._source_imatrix_signature(
        source,
        config,
        num_samples=1,
        seq_length=REFERENCE_TOKENS,
        calib_dataset=oq._OQE_CALIB_DATASET,
    )
    tokenizer = hashlib.sha256()
    for tokenizer_file in sorted(
        path.name for path in source.iterdir() if path.name in _TOKENIZER_FILES
    ):
        tokenizer.update(
            tokenizer_file.encode() + (source / tokenizer_file).read_bytes()
        )
    signature = {
        "reference_version": REFERENCE_VERSION,
        "source_hash": made_of["source_hash"],
        "calib_data_hash": made_of["calib_data_hash"],
        "tokenizer_hash": tokenizer.hexdigest(),
    }
    path = cache_dir / f"{name or source.name}.reference.npz"
    if cache_dir.is_dir() and _take_reference(path, signature, verify_dir):
        return path
    prompt = oq._load_calibration_data(
        load_tokenizer(source),
        dataset=oq._OQE_CALIB_DATASET,
        num_samples=1,
        seq_length=REFERENCE_TOKENS,
    )[0]
    started = time.monotonic()
    model, _ = oq._calibration_model(
        str(source), config, trust_remote_code=trust_remote_code
    )
    _, layers = oq._find_model_layers(model)
    if layers is None:
        raise ValueError(
            f"oQ finds no decoder layers in {type(model).__name__}: no reference "
            "of the source to verify the output against"
        )
    reference = reference_of(model, prompt, layers)
    del model, layers
    mx.clear_cache()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache_dir) as tmp:
        archive = Path(tmp) / path.name
        np.savez(archive, **reference)
        _take_reference(path, signature, verify_dir, archive)
    logger.info(
        "oQ verify: the source's reference over %d tokens, streamed in %.0f s",
        REFERENCE_TOKENS,
        time.monotonic() - started,
    )
    return path


def _writer_failure(
    output: Path, exc: BaseException, *, mode: str | None = None
) -> None:
    """Record a failed writer verification in the staged checkpoint.

    The report is written first and every write is guarded: recording the
    failure must not replace the error that failed the check."""
    detail = str(exc).strip() or type(exc).__name__
    try:
        (output / "VERIFY-FAILED.txt").write_text(f"{detail}\n", encoding="utf-8")
    except Exception:
        logger.exception("oQ verify: could not write VERIFY-FAILED.txt")
    from .oq import _atomic_write_json

    config_path = output / "config.json"
    try:
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
    except Exception:
        config = {}
    if not isinstance(config, dict):
        config = {}
    record = config.get("omlx_verification")
    if not isinstance(record, dict):
        record = {}
    record["verified"] = False
    if mode is not None:
        record["mode"] = mode
    record["reason"] = f"writer verification failed: {detail}"
    config["omlx_verification"] = record
    try:
        _atomic_write_json(config_path, config)
    except Exception:
        logger.exception("oQ verify: could not record the failure in config.json")


def _verify_attached_mtp(model, prompt_mx, config) -> None:
    """Run an attached MTP head of format layers once, before publishing.

    The trunk forward is not the whole model: a checkpoint of a family with
    an embedded Lightning MTP head (``qwen4_exp`` with ``mtp_enabled``) also
    serves that head, and a head whose forward fails or produces empty or
    non-finite logits still passes a trunk-only forward. The head is run on
    the checkpoint's first token and a zero hidden state of the width it
    reads (``hc_count * hidden_size`` for Qwen4-Exp), at the head's float
    dtype (bfloat16 when it declares none); its logits must be a non-empty
    array of finite values. A model with no attached head, or a head holding
    no format layer, is left as it was.
    """
    from .formats.linear import FormatEmbedding, FormatLinear, FormatSwitchLinear

    owner = getattr(model, "language_model", model)
    mtp_forward = getattr(owner, "mtp_forward", None)
    cache_fn = getattr(owner, "make_mtp_cache", None)
    mtp = getattr(model, "mtp", None)
    if mtp is None:
        mtp = getattr(owner, "mtp", None)
    if mtp_forward is None or cache_fn is None or mtp is None:
        return
    # Several families attach their head as a list of blocks (or a dict);
    # a Qwen4-Exp head is one module. Walk both, modules before dict (an
    # nn.Module is itself a dict), and once per object: a self-referencing
    # container must not recurse forever.
    seen: set[int] = set()

    def head_modules(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        if callable(getattr(node, "named_modules", None)):
            for _, module in node.named_modules():
                yield module
        elif isinstance(node, (list, tuple)):
            for item in node:
                yield from head_modules(item)
        elif isinstance(node, dict):
            for item in node.values():
                yield from head_modules(item)

    modules = list(head_modules(mtp))
    kinds = (FormatLinear, FormatSwitchLinear, FormatEmbedding)
    format_layers = [m for m in modules if isinstance(m, kinds)]
    if not format_layers:
        return
    hidden_size = getattr(mtp, "hidden_size", None)
    if hidden_size is None and isinstance(config, dict):
        for source in (config.get("text_config"), config):
            if isinstance(source, dict) and source.get("hidden_size") is not None:
                hidden_size = source["hidden_size"]
                break
    if hidden_size is None:
        raise RuntimeError(
            "the written checkpoint's attached MTP head names no hidden size "
            "to verify it with"
        )
    hc_count = getattr(mtp, "hc_count", None)
    width = (1 if hc_count is None else int(hc_count)) * int(hidden_size)
    # Probe at the head's own float dtype: an fp16 head's affine faces are
    # built for fp16 and decline bf16, which would fall back to an fp32
    # promotion instead of the served kernels. A plugin format may declare
    # fixed-dtype scales (no _stored_floats, no "scales" key), so scan every
    # format layer's tensors first and then the head's own float weights.
    half_dtypes = (mx.bfloat16, mx.float16)
    probe_dtype = None
    for layer in format_layers:
        for name in (
            *getattr(layer, "_stored_floats", ()),
            *getattr(layer, "_tensors", ()),
            "scales",
        ):
            value = layer.get(name)
            if isinstance(value, mx.array) and value.dtype in half_dtypes:
                probe_dtype = value.dtype
                break
        if probe_dtype is not None:
            break
    if probe_dtype is None:
        for module in modules:
            if isinstance(module, dict):
                for value in module.values():
                    if isinstance(value, mx.array) and value.dtype in half_dtypes:
                        probe_dtype = value.dtype
                        break
            if probe_dtype is not None:
                break
    if probe_dtype is None:
        probe_dtype = mx.bfloat16
    tokens = prompt_mx[:1, :1]
    hidden = mtp_cache = mtp_out = logits = None
    try:
        hidden = mx.zeros((1, 1, width), dtype=probe_dtype)
        mtp_cache = cache_fn()
        mtp_out = mtp_forward(hidden, tokens, mtp_cache)
        logits = getattr(mtp_out, "logits", mtp_out)
        if not isinstance(logits, mx.array):
            raise RuntimeError(
                "the written checkpoint's attached MTP head produced "
                f"{type(logits).__name__}, not logits"
            )
        mx.eval(logits)
        if logits.size == 0:
            raise RuntimeError(
                "the written checkpoint's attached MTP head produced empty logits"
            )
        if not bool(mx.all(mx.isfinite(logits))):
            raise RuntimeError(
                "the written checkpoint's attached MTP head produced non-finite "
                "logits"
            )
    finally:
        hidden = mtp_cache = mtp_out = logits = None


def verify_written_format_checkpoint(
    output,
    *,
    mode: str | None = None,
    prompt: np.ndarray | mx.array | None = None,
    trust_remote_code: bool = False,
) -> None:
    """Load the staged checkpoint the way a server does and run a forward on
    it before the run publishes it.

    The load runs in ``verifying()``, which lifts the unverified refusal for
    this checkpoint alone; the family gate still applies. It runs on the MLX
    executor thread the engines load on, with the process-wide state a load
    sets restored after. On any failure of the staged run the staged folder
    keeps ``verified: false`` with the reason and a ``VERIFY-FAILED.txt``
    report, and the error is raised as a ``ValueError``/``RuntimeError``; on
    success the caller may mark it verified and rename it into place. An
    invalid caller-supplied ``prompt`` is refused before the staged folder is
    read or written.
    """
    from . import oq
    from .patches import format_load

    output = Path(output)
    if prompt is not None:
        if not isinstance(prompt, mx.array):
            prompt = np.asarray(prompt, dtype=np.int32)
        if prompt.ndim not in (1, 2):
            raise ValueError(
                f"the verification prompt must be 1-D or 2-D, not {prompt.ndim}-D"
            )
        if prompt.size == 0:
            raise ValueError("the verification prompt must contain at least one token")

    def run() -> None:
        config = json.loads((output / "config.json").read_text())
        chosen = prompt
        if chosen is None:
            chosen = np.arange(REFERENCE_TOKENS, dtype=np.int32) % 63 + 1
            snapshot = output / "verify" / "reference.npz"
            if snapshot.is_file():
                with np.load(snapshot) as reference:
                    if "prompt" in reference:
                        chosen = reference["prompt"]
        if isinstance(chosen, mx.array):
            prompt_mx = chosen.astype(mx.int32)
            if prompt_mx.ndim == 1:
                prompt_mx = prompt_mx[None]
        else:
            chosen = np.asarray(chosen, dtype=np.int32)
            if chosen.ndim == 1:
                chosen = chosen[None]
            if chosen.ndim not in (1, 2):
                raise ValueError(
                    f"the verification prompt must be 1-D or 2-D, not {chosen.ndim}-D"
                )
            if chosen.size == 0:
                raise ValueError(
                    "the verification prompt must contain at least one token"
                )
            prompt_mx = mx.array(chosen)
        model = layers = out = logits = None
        with (
            oq._load_globals_kept(config),
            oq._calibration_patches(
                str(output), config, serving_routing=True
            ) as is_vlm,
        ):
            try:
                with format_load.verifying(output):
                    if is_vlm:
                        import mlx_vlm.utils as vlm_utils

                        from .engine.vlm import (
                            _derive_gemma4_global_kv_on_load,
                            _drop_gemma4_mlx_shared_kv_extras_on_load,
                            _force_minimax_m3_moe_sanitize_on_load,
                            _force_qwen4_exp_sanitize_on_load,
                            _load_optiq_vision_sidecar_on_load,
                            _preserve_gemma4_unified_image_processor_on_load,
                            _remap_nested_visual_on_load,
                            _strip_audio_config_if_orphaned,
                            _strip_vision_config_if_orphaned,
                            _transpose_qwen35_mlx_vision_patch_embed_on_load,
                        )

                        with (
                            _strip_audio_config_if_orphaned(output),
                            _strip_vision_config_if_orphaned(output),
                            _drop_gemma4_mlx_shared_kv_extras_on_load(output),
                            _derive_gemma4_global_kv_on_load(output),
                            _preserve_gemma4_unified_image_processor_on_load(output),
                            _force_minimax_m3_moe_sanitize_on_load(output),
                            _force_qwen4_exp_sanitize_on_load(output),
                            _remap_nested_visual_on_load(output),
                            _transpose_qwen35_mlx_vision_patch_embed_on_load(output),
                        ):
                            with _load_optiq_vision_sidecar_on_load(output):
                                model = vlm_utils.load_model(
                                    output,
                                    lazy=True,
                                    trust_remote_code=trust_remote_code,
                                    strict=True,
                                )
                                oq.check_loaded(model)
                    else:
                        from omlx.utils.model_loading import lm_load_compat as lm_load

                        model, _ = lm_load(
                            str(output),
                            lazy=True,
                            trust_remote_code=trust_remote_code,
                            model_config=oq._sensitivity_lm_config_override(
                                config, str(output)
                            ),
                        )
                _, layers = oq._find_model_layers(model)
                if layers is None:
                    out = model(prompt_mx)
                else:
                    with layer_by_layer(layers):
                        out = model(prompt_mx)
                logits = getattr(out, "logits", out)
                if not isinstance(logits, mx.array):
                    raise RuntimeError(
                        "the written checkpoint's forward produced "
                        f"{type(logits).__name__}, not logits"
                    )
                mx.eval(logits)
                if logits.size == 0:
                    raise RuntimeError(
                        "the written checkpoint's forward produced empty logits"
                    )
                if not bool(mx.all(mx.isfinite(logits))):
                    raise RuntimeError(
                        "the written checkpoint's forward produced non-finite logits"
                    )
                _verify_attached_mtp(model, prompt_mx, config)
            finally:
                if model is not None:
                    close = getattr(model, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            logger.exception("oQ verify: model.close() failed")
                model = layers = out = logits = None
                mx.clear_cache()

    try:
        oq._on_mlx_thread(run)
    except Exception as exc:
        try:
            _writer_failure(output, exc, mode=mode)
        except Exception:
            logger.exception("oQ verify: could not record the verification failure")
        if isinstance(exc, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"the written checkpoint did not verify: {exc}") from exc
