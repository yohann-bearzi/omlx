# SPDX-License-Identifier: Apache-2.0
"""A stock oQe/oQh build of this tree is upstream e1a402cf's, byte for byte.

The merge of upstream main e1a402cf ported upstream's changes of the
monolithic quantize_oq_streaming (oQh #5317, the oQe code refinement #5292,
streamed Qwen3.5 calibration #5332/#5459, per-round embedding #5269, the
discovery lock #5447) into this branch's split writer (_open_oq_run, the run
object, the format writer). A build without a quant format must come out as
upstream's own: the same stored tensor bytes (dtype, shape, sha256), files,
config and report, from the same source and the same calibration tokens.

Goldens are generated from upstream e1a402cf archived from this repository's
git history (tools/gen_oq_upstream_parity.py) with MLX_ENABLE_TF32=0 (matching
tests/conftest.py) and keyed by GPU architecture, because upstream e1a402cf
itself differs on near-tie #5292 refinement codes across Apple GPU generations
(steel on M3 vs NAX on M5). On an unlisted architecture in a git checkout, the
test archives upstream e1a402cf directly and compares live.
"""

from __future__ import annotations

import copy
import io
import json
import struct
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from tests.safetensors_bytes import tensor_bytes

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    _HAS_METAL = mx.metal.is_available()
except ImportError:
    mx = None
    tree_flatten = None
    _HAS_METAL = False

_REQUIRES_METAL = pytest.mark.skipif(not _HAS_METAL, reason="requires Metal")

UPSTREAM = "e1a402cf7b8e36599a3c1689b9d12cec5da79ade"
_REPO = Path(__file__).resolve().parents[1]

# The build both trees run: the calibration tokens replace the corpus (the
# imatrix and the refinement read the same rows), the sensitivity is given.
_DRIVER = r"""
import sys
from pathlib import Path

tree, source, output, level, hadamard, preserve_mtp = sys.argv[1:7]
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
    hadamard=hadamard == "1",
    preserve_mtp=preserve_mtp == "1",
)
"""

# Volatile fields of oq_imatrix_report.json: where the cache is, how long the
# refinement took, the free memory the calibration's batch plan read (its
# decisions and the micro-batch size are compared).
_VOLATILE = ("cache_path",)
_VOLATILE_REFINE = ("seconds",)
_VOLATILE_PLAN = (
    "live_available_bytes",
    "remaining_available_bytes",
    "system_available_bytes",
    # Runtime facts of the Mac, not the writer's output.
    "metal_available_bytes",
    "capture_budget_bytes",
)


def _child_env(tree: Path) -> dict[str, str]:
    """Deterministic child-process environment shared by test and generator.

    Pins MLX_ENABLE_TF32=0 to match tests/conftest.py so M5-class tensor units
    run fp32 matmuls identically in both golden generation and pytest.
    """
    return {
        "PYTHONPATH": str(tree),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "PATH": "/usr/bin:/bin",
        "MLX_ENABLE_TF32": "0",
    }


def _goldens() -> dict:
    return json.loads(
        (Path(__file__).parent / "data" / "oq_upstream_parity.json").read_text()
    )


def _tensor_hashes(output: Path) -> dict[str, dict]:
    hashes = {}
    for shard in sorted(output.glob("*.safetensors")):
        for name, entry in tensor_bytes(shard).items():
            hashes[f"{shard.name}:{name}"] = entry
    return hashes


def _snapshot(output: Path) -> dict:
    index = output / "model.safetensors.index.json"
    return {
        "files": sorted(path.name for path in output.iterdir()),
        "tensors": _tensor_hashes(output),
        "config": json.loads((output / "config.json").read_text()),
        "report": _report(output / "oq_imatrix_report.json"),
        "index": json.loads(index.read_text()) if index.exists() else None,
    }


def _assert_parity(output: Path, case: dict) -> None:
    assert sorted(path.name for path in output.iterdir()) == case["files"]
    assert _tensor_hashes(output) == case["tensors"]
    assert json.loads((output / "config.json").read_text()) == case["config"]
    assert _report(output / "oq_imatrix_report.json") == _stripped(case["report"])
    index = output / "model.safetensors.index.json"
    got = json.loads(index.read_text()) if index.exists() else None
    assert got == case["index"]


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


def _qwen3_5_moe_source(path: Path, monkeypatch, *, mtp: bool = False) -> None:
    """Upstream's tiny Qwen3.5 MoE (tests/test_oq.py), widened as upstream's
    refinement test widens it: routed experts, shared expert, and optional MTP."""
    from tests import test_oq

    monkeypatch.setitem(test_oq._QWEN3_5_STREAM_TEXT, "hidden_size", 256)
    monkeypatch.setitem(test_oq._QWEN3_5_STREAM_TEXT, "intermediate_size", 512)
    for key in ("moe_intermediate_size", "shared_expert_intermediate_size"):
        monkeypatch.setitem(test_oq._QWEN3_5_STREAM_EXPERTS, key, 256)
    test_oq._qwen3_5_stream_source(path, "qwen3_5_moe", vlm=False, mtp=mtp)


def _build(
    tree: Path,
    source: Path,
    output: Path,
    level: int,
    workdir: Path,
    *,
    hadamard: bool = False,
    preserve_mtp: bool = False,
) -> str:
    args = [
        str(tree),
        str(source),
        str(output),
        str(level),
        "1" if hadamard else "0",
        "1" if preserve_mtp else "0",
    ]
    done = subprocess.run(
        [sys.executable, "-c", _DRIVER, *args],
        cwd=workdir,
        env=_child_env(tree),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, f"build under {tree} failed:\n{done.stderr[-6000:]}"
    return done.stderr


def _stripped(report: dict) -> dict:
    """``report`` without the run's volatile facts (both sides of a compare)."""
    report = copy.deepcopy(report)
    for key in _VOLATILE:
        report.pop(key, None)
    for key in _VOLATILE_REFINE:
        (report.get("refine") or {}).pop(key, None)
    (report.get("oqh") or {}).pop("seconds", None)
    plan = (report.get("collection") or {}).get("batch_plan") or {}
    for key in _VOLATILE_PLAN:
        plan.pop(key, None)
    return report


def _report(path: Path) -> dict:
    return _stripped(json.loads(path.read_text()))


def _archive_upstream(destination: Path) -> bool:
    try:
        archived = subprocess.run(
            ["git", "-C", str(_REPO), "archive", "--format=tar", UPSTREAM, "omlx"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return False
    with tarfile.open(fileobj=io.BytesIO(archived)) as tar:
        tar.extractall(destination, filter="data")
    oq_source = (destination / "omlx" / "oq.py").read_text()
    assert "def _refine_oqe_output(" in oq_source
    assert "def _write_oqh_output(" in oq_source
    assert "quant_format" not in oq_source
    return True


def _expected_case(
    case_name: str,
    source: Path,
    level: int,
    tmp_path: Path,
    *,
    hadamard: bool = False,
    preserve_mtp: bool = False,
) -> dict:
    goldens = _goldens()
    assert goldens["upstream"] == UPSTREAM
    assert goldens["numerics"]["tf32"] == "off"
    arch = str(mx.device_info().get("architecture", ""))
    by_arch = goldens.get("architectures") or {}
    if arch in by_arch and case_name in by_arch[arch]:
        assert goldens["numerics"]["mlx"] == mx.__version__
        return by_arch[arch][case_name]
    upstream_tree = tmp_path / "upstream_tree"
    upstream_tree.mkdir()
    if not _archive_upstream(upstream_tree):
        pytest.fail(
            f"no golden for GPU architecture {arch!r} case {case_name!r} and git "
            f"archive of {UPSTREAM[:8]} is unavailable; run "
            "tools/gen_oq_upstream_parity.py"
        )
    up_work = tmp_path / "upstream_work"
    up_work.mkdir()
    theirs = tmp_path / "upstream_out"
    _build(
        upstream_tree,
        source,
        theirs,
        level,
        up_work,
        hadamard=hadamard,
        preserve_mtp=preserve_mtp,
    )
    return _snapshot(theirs)


def _write_raw_safetensors(
    path: Path, tensors: dict[str, tuple[str, list[int], bytes]]
) -> None:
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        lo = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [lo, len(payload)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(
        struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(payload)
    )


def test_safetensors_bytes_detects_flipped_byte_dtype_and_shape(tmp_path):
    """CPU-only negative control: a single flipped byte, dtype or shape change
    in stored safetensors bytes fails parity."""
    k_bytes = bytes(range(32))
    s_bytes = bytes(range(16))
    base_tensors = {
        "model.layers.0.self_attn.k_proj.weight": ("U32", [2, 4], k_bytes),
        "model.layers.0.self_attn.k_proj.scales": ("BF16", [2, 4], s_bytes),
    }

    def write_case(directory: Path, tensors: dict) -> None:
        directory.mkdir()
        _write_raw_safetensors(directory / "model.safetensors", tensors)
        (directory / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (directory / "oq_imatrix_report.json").write_text(
            json.dumps({"refine": {"status": "done", "tensors": 1}})
        )

    good = tmp_path / "good"
    write_case(good, base_tensors)
    expected = _snapshot(good)
    _assert_parity(good, expected)

    # Flipped-byte negative control: flip one bit in k_proj.weight's payload.
    flipped = tmp_path / "flipped"
    bad_bytes = bytearray(k_bytes)
    bad_bytes[7] ^= 0x01
    write_case(
        flipped,
        {
            **base_tensors,
            "model.layers.0.self_attn.k_proj.weight": ("U32", [2, 4], bytes(bad_bytes)),
        },
    )
    with pytest.raises(AssertionError):
        _assert_parity(flipped, expected)

    # Dtype mismatch control: same stored bytes, different header dtype.
    wrong_dtype = tmp_path / "wrong_dtype"
    write_case(
        wrong_dtype,
        {
            **base_tensors,
            "model.layers.0.self_attn.k_proj.weight": ("I32", [2, 4], k_bytes),
        },
    )
    with pytest.raises(AssertionError):
        _assert_parity(wrong_dtype, expected)

    # Shape mismatch control: same stored bytes and dtype, different shape.
    wrong_shape = tmp_path / "wrong_shape"
    write_case(
        wrong_shape,
        {
            **base_tensors,
            "model.layers.0.self_attn.k_proj.weight": ("U32", [4, 2], k_bytes),
        },
    )
    with pytest.raises(AssertionError):
        _assert_parity(wrong_shape, expected)

    # Corrupt byte length in header fails closed.
    bad_len = tmp_path / "bad_len.safetensors"
    _write_raw_safetensors(
        bad_len,
        {"model.layers.0.self_attn.k_proj.weight": ("U32", [2, 4], k_bytes[:-4])},
    )
    with pytest.raises(ValueError, match="byte length"):
        tensor_bytes(bad_len)


def _write_raw(path: Path, header: dict, payload: bytes) -> None:
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def test_safetensors_bytes_refuses_out_of_bounds_truncated_and_unknown(tmp_path):
    # data_offsets past the payload
    bad = tmp_path / "oob.safetensors"
    _write_raw(bad, {"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}}, b"\x01")
    with pytest.raises(ValueError, match="out of bounds"):
        tensor_bytes(bad)
    # truncated header length
    trunc = tmp_path / "trunc.safetensors"
    trunc.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="truncated"):
        tensor_bytes(trunc)
    # header length past the file's end
    short = tmp_path / "eof.safetensors"
    short.write_bytes(struct.pack("<Q", 100) + b"{}")
    with pytest.raises(ValueError, match="extends past EOF"):
        tensor_bytes(short)
    # an unknown dtype is refused, not guessed
    unknown = tmp_path / "dtype.safetensors"
    _write_raw(unknown, {"a": {"dtype": "Q99", "shape": [1], "data_offsets": [0, 1]}}, b"\x00")
    with pytest.raises(ValueError, match="unknown safetensors dtype"):
        tensor_bytes(unknown)
    # FP8 and C64 are known names with their stored widths
    fp8 = tmp_path / "fp8.safetensors"
    _write_raw(fp8, {"a": {"dtype": "F8_E4M3", "shape": [2], "data_offsets": [0, 2]}}, b"\x00\x00")
    assert tensor_bytes(fp8)["a"]["dtype"] == "F8_E4M3"
    c64 = tmp_path / "c64.safetensors"
    _write_raw(c64, {"a": {"dtype": "C64", "shape": [1], "data_offsets": [0, 8]}}, b"\x00" * 8)
    assert tensor_bytes(c64)["a"]["dtype"] == "C64"
    # offsets are two non-bool integers and shapes are lists
    for name, entry, said in (
        ("bool_offsets", {"dtype": "U8", "shape": [1], "data_offsets": [False, True]}, "data_offsets"),
        ("float_offsets", {"dtype": "U8", "shape": [1], "data_offsets": [0.0, 1.0]}, "data_offsets"),
        ("str_shape", {"dtype": "U8", "shape": "", "data_offsets": [0, 1]}, "shape"),
        ("dict_shape", {"dtype": "U8", "shape": {}, "data_offsets": [0, 1]}, "shape"),
    ):
        bad = tmp_path / f"{name}.safetensors"
        _write_raw(bad, {"a": entry}, b"\x00")
        with pytest.raises(ValueError, match=said):
            tensor_bytes(bad)
    # negative shapes, a non-dict header, a non-dict entry
    # a boolean is not a shape integer
    boolean = tmp_path / "boolean.safetensors"
    _write_raw(
        boolean, {"a": {"dtype": "U8", "shape": [True], "data_offsets": [0, 1]}}, b"\x00"
    )
    with pytest.raises(ValueError, match="negative or non-integer shape"):
        tensor_bytes(boolean)
    neg = tmp_path / "neg.safetensors"
    _write_raw(neg, {"a": {"dtype": "U8", "shape": [-1], "data_offsets": [0, 0]}}, b"")
    with pytest.raises(ValueError, match="negative"):
        tensor_bytes(neg)
    nondict = tmp_path / "nondict.safetensors"
    _write_raw(nondict, [1, 2], b"")
    with pytest.raises(ValueError, match="not an object"):
        tensor_bytes(nondict)
    entry = tmp_path / "entry.safetensors"
    _write_raw(entry, {"a": 7}, b"")
    with pytest.raises(ValueError, match="not an object"):
        tensor_bytes(entry)


def test_parity_child_env_pins_tf32_off_and_uses_no_custom_env_vars():
    """Both test and generator pin MLX_ENABLE_TF32=0 (matching conftest.py)
    and never read environment variables."""
    env = _child_env(_REPO)
    assert env["MLX_ENABLE_TF32"] == "0"
    assert env["PYTHONPATH"] == str(_REPO)
    for rel in ("tests/test_oq_upstream_parity.py", "tools/gen_oq_upstream_parity.py"):
        text = (_REPO / rel).read_text()
        assert "os." + "environ" not in text, rel
        assert "os." + "getenv" not in text, rel


@_REQUIRES_METAL
@pytest.mark.parametrize("level", [4, 3], ids=["oQ4e", "oQ3e"])
@pytest.mark.parametrize("family", ["llama", "qwen3_5_moe"])
def test_a_stock_oqe_build_is_upstream_s_byte_for_byte(
    tmp_path, monkeypatch, family, level
):
    source = tmp_path / "source"
    if family == "llama":
        _llama_source(source)
    else:
        _qwen3_5_moe_source(source, monkeypatch, mtp=False)
    work = tmp_path / "work"
    work.mkdir()
    ours = tmp_path / "merged" / "out"
    _build(_REPO, source, ours, level, work)
    case = _expected_case(f"{family}-oQ{level}e", source, level, tmp_path)
    _assert_parity(ours, case)
    # Upstream refined this build: the parity covers the refinement's codes.
    refine = case["report"]["refine"]
    assert refine["status"] == "done" and refine["tensors"] > 0, refine


@_REQUIRES_METAL
@pytest.mark.parametrize("level", [4, 3, 2], ids=["oQ4e", "oQ3e", "oQ2e"])
def test_a_stock_moe_mtp_oqe_build_is_upstream_s_byte_for_byte(
    tmp_path, monkeypatch, level
):
    """A tiny MoE + MTP build (routed experts, shared expert, preserved MTP
    layer) at oQ4e/oQ3e/oQ2e matches upstream e1a402cf on all stored tensors."""
    source = tmp_path / "source"
    _qwen3_5_moe_source(source, monkeypatch, mtp=True)
    work = tmp_path / "work"
    work.mkdir()
    ours = tmp_path / "merged" / "out"
    _build(_REPO, source, ours, level, work, preserve_mtp=True)
    case = _expected_case(
        f"qwen3_5_moe_mtp-oQ{level}e",
        source,
        level,
        tmp_path,
        preserve_mtp=True,
    )
    _assert_parity(ours, case)
    tensor_names = case["tensors"]
    assert any(".switch_mlp." in name for name in tensor_names)
    assert any(".shared_expert." in name for name in tensor_names)
    assert any(".mtp." in name for name in tensor_names)
    refine = case["report"]["refine"]
    assert refine["status"] == "done" and refine["tensors"] > 0, refine


@_REQUIRES_METAL
@pytest.mark.parametrize("level", [4, 3], ids=["oQ4h", "oQ3h"])
def test_a_stock_oqh_build_is_upstream_s_byte_for_byte(tmp_path, monkeypatch, level):
    source = tmp_path / "source"
    _qwen3_5_moe_source(source, monkeypatch, mtp=False)
    work = tmp_path / "work"
    work.mkdir()
    ours = tmp_path / "merged" / "out"
    _build(_REPO, source, ours, level, work, hadamard=True)
    case = _expected_case(
        f"qwen3_5_moe-oQ{level}h", source, level, tmp_path, hadamard=True
    )
    _assert_parity(ours, case)
    assert "refine" not in case["report"]
    assert case["report"]["oqh"]["status"] == "done"
    assert case["report"]["oqh"]["tensors"] > 0


@pytest.mark.skipif(mx is None, reason="requires mlx")
def test_expected_case_fails_loudly_on_unlisted_arch_without_git(tmp_path, monkeypatch):
    """An architecture with no golden must fail, never skip, git or not."""
    import sys
    from types import SimpleNamespace

    monkeypatch.setattr(
        sys.modules[__name__],
        "mx",
        SimpleNamespace(
            device_info=lambda: {"architecture": "applegpu_g99z"},
            __version__="0.0.0",
        ),
    )
    monkeypatch.setattr(
        "tests.test_oq_upstream_parity._archive_upstream", lambda destination: False
    )
    with pytest.raises(pytest.fail.Exception, match=r"applegpu_g99z.*llama-oQ4e"):
        _expected_case("llama-oQ4e", tmp_path, 4, tmp_path)


@pytest.mark.skipif(mx is None, reason="requires mlx")
def test_gen_drops_stale_arch_when_mlx_version_changes():
    """Preserved architectures are reused only under the same numerics."""
    from tools.gen_oq_upstream_parity import _merge_architectures

    loaded = {
        "upstream": UPSTREAM,
        "numerics": {"tf32": "off", "mlx": "0.0.0"},
        "architectures": {"applegpu_g17s": {"llama-oQ4e": {"report": {}}}},
    }
    merged = _merge_architectures(
        loaded, "applegpu_g15d", {"llama-oQ4e": {"report": {}}}, "0.32.4"
    )
    assert "applegpu_g17s" not in merged
    assert "applegpu_g15d" in merged


@pytest.mark.skipif(mx is None, reason="requires mlx")
def test_gen_preserves_matching_arch_and_strips_volatile_report():
    """A preserved arch under the same numerics is kept, with its report stripped."""
    from tools.gen_oq_upstream_parity import _merge_architectures

    report = {
        "collection": {
            "batch_plan": {
                "metal_available_bytes": 1,
                "capture_budget_bytes": 2,
                "micro_batch_size": 4,
            }
        }
    }
    loaded = {
        "upstream": UPSTREAM,
        "numerics": {"tf32": "off", "mlx": "0.32.4"},
        "architectures": {"applegpu_g17s": {"llama-oQ4e": {"report": report}}},
    }
    merged = _merge_architectures(
        loaded, "applegpu_g15d", {"llama-oQ4e": {"report": {}}}, "0.32.4"
    )
    assert "applegpu_g17s" in merged and "applegpu_g15d" in merged
    plan = merged["applegpu_g17s"]["llama-oQ4e"]["report"]["collection"]["batch_plan"]
    assert "metal_available_bytes" not in plan
    assert "capture_budget_bytes" not in plan
    dropped = _merge_architectures(
        {**loaded, "upstream": "0" * 40}, "applegpu_g15d", {}, "0.32.4"
    )
    assert "applegpu_g17s" not in dropped


def _parity_directory(directory: Path) -> Path:
    directory.mkdir()
    _write_raw_safetensors(
        directory / "model.safetensors",
        {"model.layers.0.self_attn.k_proj.weight": ("U32", [2, 4], bytes(range(32)))},
    )
    (directory / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (directory / "oq_imatrix_report.json").write_text(
        json.dumps(
            {
                "refine": {"status": "done", "tensors": 1},
                "collection": {
                    "batch_plan": {
                        "metal_available_bytes": 524287998672,
                        "capture_budget_bytes": 805306368,
                        "micro_batch_size": 4,
                    }
                },
            }
        )
    )
    return directory


def test_assert_parity_detects_corrupt_index(tmp_path):
    """A multi-shard index's weight_map is part of the golden, not just names."""
    good = _parity_directory(tmp_path / "good")
    (good / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 32},
                "weight_map": {
                    "model.layers.0.self_attn.k_proj.weight": "model.safetensors"
                },
            }
        )
    )
    expected = _snapshot(good)
    _assert_parity(good, expected)
    expected.pop("index")
    with pytest.raises(KeyError):
        _assert_parity(good, expected)
    expected = _snapshot(good)
    expected["index"]["weight_map"][
        "model.layers.0.self_attn.k_proj.weight"
    ] = "other.safetensors"
    with pytest.raises(AssertionError):
        _assert_parity(good, expected)


def test_assert_parity_strips_volatile_plan_keys_from_golden(tmp_path):
    """The golden side is stripped of host facts too (M5 goldens lack them)."""
    good = _parity_directory(tmp_path / "good")
    expected = _snapshot(good)
    plan = expected["report"].setdefault("collection", {}).setdefault(
        "batch_plan", {}
    )
    plan["metal_available_bytes"] = 524287998672
    plan["capture_budget_bytes"] = 805306368
    _assert_parity(good, expected)
