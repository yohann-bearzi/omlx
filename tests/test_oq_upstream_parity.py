# SPDX-License-Identifier: Apache-2.0
"""A stock oQe build of the merged tree is upstream's, byte for byte.

The merge of upstream main e1a402cf ported upstream's changes of the
monolithic quantize_oq_streaming (oQh #5317, the oQe code refinement #5292,
streamed Qwen3.5 calibration #5332/#5459, per-round embedding #5269, the
discovery lock #5447) into this branch's split writer (_open_oq_run, the run
object, the format writer). A build without a quant format must come out as
upstream's own: the same tensors, config and report, from the same source and
the same calibration tokens.

Upstream's tree is ``OMLX_UPSTREAM_TREE`` (a directory holding upstream's
``omlx`` package), or else the package archived from this repository's
history at e1a402cf. Each build runs in a fresh interpreter with only its
tree on the path.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

UPSTREAM = "e1a402cf7b8e36599a3c1689b9d12cec5da79ade"
_REPO = Path(__file__).resolve().parents[1]

# The build both trees run: the calibration tokens replace the corpus (the
# imatrix and the refinement read the same rows), the sensitivity is given.
_DRIVER = r"""
import sys
from pathlib import Path

tree, source, output, level = sys.argv[1:5]
import mlx.core as mx
import omlx.oq as oq

where = Path(oq.__file__).resolve()
assert where.is_relative_to(Path(tree).resolve()), where
tokens = (mx.arange(64).reshape(4, 16) * 7 % 120 + 1).astype(mx.int32)
oq._load_calibration_data = lambda *args, **kwargs: tokens
out = Path(output)
oq.quantize_oq_streaming(
    source,
    output,
    int(level),
    enhanced=True,
    imatrix_num_samples=4,
    imatrix_seq_length=16,
    imatrix_cache_path=str(out.parent / f"{out.name}.npz"),
    sensitivity_map_override={0: 0.1, 1: 0.2},
)
"""

# Volatile fields of oq_imatrix_report.json: where the cache is, how long the
# refinement took, the free memory the calibration's batch plan read (its
# decisions, the micro-batch size and budgets, are compared).
_VOLATILE = ("cache_path",)
_VOLATILE_REFINE = ("seconds",)
_VOLATILE_PLAN = (
    "live_available_bytes",
    "remaining_available_bytes",
    "system_available_bytes",
)


@pytest.fixture(scope="module")
def upstream_tree(tmp_path_factory) -> Path:
    given = os.environ.get("OMLX_UPSTREAM_TREE")
    if given:
        tree = Path(given)
    else:
        try:
            archived = subprocess.run(
                ["git", "-C", str(_REPO), "archive", "--format=tar", UPSTREAM, "omlx"],
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip(
                f"upstream {UPSTREAM[:8]} is not in this tree's history: set "
                "OMLX_UPSTREAM_TREE to a directory holding its omlx package"
            )
        tree = tmp_path_factory.mktemp("upstream")
        with tarfile.open(fileobj=io.BytesIO(archived)) as tar:
            tar.extractall(tree, filter="data")
    oq_source = (tree / "omlx" / "oq.py").read_text()
    # Upstream's own writer: the refinement and oQh, and no quant format.
    assert "def _refine_oqe_output(" in oq_source
    assert "def _write_oqh_output(" in oq_source
    assert "quant_format" not in oq_source
    return tree


def _llama_source(path: Path) -> None:
    """A tiny BF16 Llama with an untied head, wide enough to be quantized
    and refined, with its tokenizer."""
    from mlx_lm.models import llama
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
    }
    path.mkdir()
    mx.random.seed(11)
    model = llama.Model(llama.ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    mx.save_safetensors(
        str(path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    (path / "config.json").write_text(json.dumps(config))
    vocab = {f"t{i}": i for i in range(config["vocab_size"])}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="t0"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0").save_pretrained(
        path
    )


def _qwen3_5_moe_source(path: Path, monkeypatch) -> None:
    """Upstream's tiny Qwen3.5 MoE (tests/test_oq.py), widened as upstream's
    refinement test widens it: routed experts the refinement re-picks."""
    from tests import test_oq

    monkeypatch.setitem(test_oq._QWEN3_5_STREAM_TEXT, "hidden_size", 256)
    monkeypatch.setitem(test_oq._QWEN3_5_STREAM_TEXT, "intermediate_size", 512)
    for key in ("moe_intermediate_size", "shared_expert_intermediate_size"):
        monkeypatch.setitem(test_oq._QWEN3_5_STREAM_EXPERTS, key, 256)
    test_oq._qwen3_5_stream_source(path, "qwen3_5_moe", vlm=False, mtp=False)


def _build(tree: Path, source: Path, output: Path, level: int, workdir: Path) -> str:
    env = {
        **os.environ,
        "PYTHONPATH": str(tree),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
    }
    args = [str(tree), str(source), str(output), str(level)]
    done = subprocess.run(
        [sys.executable, "-c", _DRIVER, *args],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, f"build under {tree} failed:\n{done.stderr[-6000:]}"
    return done.stderr


def _report(path: Path) -> dict:
    report = json.loads(path.read_text())
    for key in _VOLATILE:
        report.pop(key, None)
    for key in _VOLATILE_REFINE:
        (report.get("refine") or {}).pop(key, None)
    plan = (report.get("collection") or {}).get("batch_plan") or {}
    for key in _VOLATILE_PLAN:
        plan.pop(key, None)
    return report


@pytest.mark.parametrize("level", [4, 3], ids=["oQ4e", "oQ3e"])
@pytest.mark.parametrize("family", ["llama", "qwen3_5_moe"])
def test_a_stock_oqe_build_is_upstream_s_byte_for_byte(
    upstream_tree, tmp_path, monkeypatch, family, level
):
    source = tmp_path / "source"
    if family == "llama":
        _llama_source(source)
    else:
        _qwen3_5_moe_source(source, monkeypatch)
    work = tmp_path / "work"
    work.mkdir()
    theirs = tmp_path / "upstream" / "out"
    ours = tmp_path / "merged" / "out"
    _build(upstream_tree, source, theirs, level, work)
    _build(_REPO, source, ours, level, work)

    their_files = sorted(p.name for p in theirs.iterdir())
    assert sorted(p.name for p in ours.iterdir()) == their_files
    report = "oq_imatrix_report.json"
    for name in their_files:
        if name == report:
            continue
        assert (ours / name).read_bytes() == (theirs / name).read_bytes(), name
    # The report too, but where its cache is and how long the refinement took.
    mine, upstream = _report(ours / report), _report(theirs / report)
    differing = {
        (key, sub)
        for key in mine.keys() | upstream.keys()
        for sub in (
            (mine.get(key) or {}).keys() | (upstream.get(key) or {}).keys()
            if isinstance(mine.get(key), dict) and isinstance(upstream.get(key), dict)
            else [None]
        )
        if (mine.get(key) if sub is None else mine[key].get(sub))
        != (upstream.get(key) if sub is None else upstream[key].get(sub))
    }
    assert not differing, {
        f"{key}.{sub}": (
            (mine.get(key) if sub is None else mine[key].get(sub)),
            (upstream.get(key) if sub is None else upstream[key].get(sub)),
        )
        for key, sub in differing
    }
    # Upstream refined this build: the parity covers the refinement's codes.
    refine = json.loads((theirs / report).read_text())["refine"]
    assert refine["status"] == "done" and refine["tensors"] > 0, refine
