"""The verification of a quantization in a quant format: what it checks the
written checkpoint against.

The reference is the source's own: its next-token distribution at each
position of a fixed prompt, kept as the top tokens and the mass left out,
computed by the family's own forward over the lazily loaded source, one
decoder layer resident at a time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

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
            self.update(tree_map(lambda _: mx.zeros((0,)), self.parameters()))
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


def source_reference(
    model_path, config: dict, cache_dir: Path, *, trust_remote_code: bool = False
) -> Path:
    """The file of the source's reference: the one in ``cache_dir`` when it
    is of this source (its config and weight files), calibration corpus,
    tokenizer and version, found without loading any of them; else made by
    one streamed pass over the source (its time logged)."""
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
    signed = cache_dir / f"{source.name}.reference.json"
    if (
        path.is_file()
        and signed.is_file()
        and json.loads(signed.read_text()) == signature
    ):
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
    np.savez(path, **reference)
    signed.write_text(json.dumps(signature, indent=2))
    logger.info(
        "oQ verify: the source's reference over %d tokens, streamed in %.0f s",
        REFERENCE_TOKENS,
        time.monotonic() - started,
    )
    return path
