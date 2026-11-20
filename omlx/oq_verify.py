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
) -> Path:
    """The file of the source's reference: the one in ``cache_dir`` when it
    is of this source (its config and weight files), calibration corpus,
    tokenizer and version, found without loading any of them; else made by
    one streamed pass over the source (its time logged). A writer supplies
    its private verification directory to keep a snapshot of this generation."""
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
    for name in sorted(
        path.name for path in source.iterdir() if path.name in _TOKENIZER_FILES
    ):
        tokenizer.update(name.encode() + (source / name).read_bytes())
    signature = {
        "reference_version": REFERENCE_VERSION,
        "source_hash": made_of["source_hash"],
        "calib_data_hash": made_of["calib_data_hash"],
        "tokenizer_hash": tokenizer.hexdigest(),
    }
    path = cache_dir / f"{source.name}.reference.npz"
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
    try:
        (output / "VERIFY-FAILED.txt").write_text(f"{exc}\n", encoding="utf-8")
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
    record["reason"] = f"writer verification failed: {exc}"
    config["omlx_verification"] = record
    try:
        _atomic_write_json(config_path, config)
    except Exception:
        logger.exception("oQ verify: could not record the failure in config.json")


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
    sets restored after. On any failure the staged folder keeps
    ``verified: false`` with the reason and a ``VERIFY-FAILED.txt`` report, and
    the error is raised as a ``ValueError``/``RuntimeError``; on success the
    caller may mark it verified and rename it into place.
    """
    from . import oq
    from .patches import format_load

    output = Path(output)

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
        chosen = np.asarray(chosen, dtype=np.int32)
        if chosen.ndim == 1:
            chosen = chosen[None]
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
                mx.eval(logits)
                if not bool(mx.all(mx.isfinite(logits))):
                    raise RuntimeError(
                        "the written checkpoint's forward produced non-finite logits"
                    )
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
