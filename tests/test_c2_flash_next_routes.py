"""Development run in a fresh process on the two-layer Flash-Next checkpoint."""

import asyncio
import json
import shutil
from dataclasses import asdict, replace
from string import ascii_letters
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from omlx import formats, oq
from omlx.cache.type_registry import CacheTypeRegistry
from omlx.engine.vlm import VLMBatchedEngine
from omlx.engine_core import get_mlx_executor
from omlx.engine_pool import EnginePool
from omlx.formats import kernels
from omlx.formats.linear import FormatLinear, FormatSwitchLinear
from omlx.model_settings import ModelSettings
from omlx.patches import qwen35_verify_qmm as routing
from omlx.patches.mlx_lm_mtp.batch_generator import _call_backbone
from omlx.scheduler import _patched_merge_caches
from omlx.settings import init_settings
from tests.test_formats_kernels import _toy
from tests.test_formats_oq import _Codec, _in_format, _quantize
from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config


class ExactCodec(_Codec):
    """One signed power-of-two coefficient per row, chosen from the source.

    Every matmul has exactly one nonzero summand per output, so the face
    and the independent dense checkpoint can be compared bit for bit.
    """

    face_for_bf16 = replace(_toy("flash_next"), dtypes=(mx.bfloat16,))

    def dot(self, layer, k, n):
        # A face reads a tensor of 8 elements or more: oMLX binds a smaller one
        # as a constant, and the layer keeps its composed path.
        return self.face_for_bf16 if k % 8 == 0 and n >= 8 else None

    def encode(self, w, importance, bits, group_size):
        self.encoded.append((w.shape, bits, group_size))
        col = mx.argmax(mx.abs(w), axis=-1, keepdims=True)
        peak = mx.take_along_axis(w, col, axis=-1).astype(mx.float32)
        # Quantize the sign at the selected column. The gain is a nonzero
        # power of two and varies by row, including across the experts.
        rows = mx.arange(peak.size).reshape(peak.shape[:-1])
        gain = mx.power(2.0, (rows % 3) - 2).astype(mx.float32)
        code = mx.where(peak >= 0, 129, 127)
        packed = mx.where(mx.arange(w.shape[-1]) == col, code, 128)
        return {"codes": packed.astype(mx.uint8), "group_scales": gain}


def tiny_processor(path, cfg):
    from transformers import Qwen3_5Tokenizer

    words = [f"<t{i}>" for i in range(cfg.vocab_size)]
    words[2 : 2 + len(ascii_letters)] = ascii_letters
    words[0] = "<|endoftext|>"
    words[cfg.text_config.eos_token_id] = "<|im_end|>"
    specials = {
        cfg.image_token_id: "<|image_pad|>",
        cfg.video_token_id: "<|video_pad|>",
        cfg.vision_start_token_id: "<|vision_start|>",
        cfg.vision_end_token_id: "<|vision_end|>",
    }
    for index, token in specials.items():
        words[index] = token
    tokenizer = Qwen3_5Tokenizer(
        vocab={word: index for index, word in enumerate(words)},
        merges=[],
        eos_token="<|im_end|>",
        additional_special_tokens=list(specials.values()),
        chat_template="{% for message in messages %}{{ message.content }}{% endfor %}",
    )
    tokenizer.save_pretrained(path)
    (path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Qwen3VLProcessor",
                "image_processor_type": "Qwen2VLImageProcessorFast",
                "patch_size": cfg.vision_config.patch_size,
                "merge_size": cfg.vision_config.spatial_merge_size,
                "temporal_patch_size": cfg.vision_config.temporal_patch_size,
            }
        )
    )


def tiny_source(path, *, fused_experts=False):
    """A local processor and synthetic BF16 weights share the same vocabulary."""
    from mlx_vlm.models.qwen4_exp.language import (
        configure_mtp_runtime,
        configure_ple_runtime,
    )
    from mlx_vlm.models.qwen4_exp.qwen4_exp import Model

    path.mkdir()
    cfg = _tiny_config()
    tiny_processor(path, cfg)
    # The fused shared gate needs eight g64 scales in its single row.
    # The router also requires hidden < 16 * experts.
    hidden = 512 if fused_experts else 128
    cfg.text_config.hidden_size = hidden
    cfg.vision_config.out_hidden_size = hidden
    cfg.text_config.linear_key_head_dim = 32
    cfg.text_config.linear_value_head_dim = 32
    # The 128-wide dense shared down projection uses MLX's qmv_quad.
    cfg.text_config.shared_expert_intermediate_size = 256 if fused_experts else 128
    cfg.text_config.num_experts = 64 if fused_experts else 32
    cfg.text_config.num_experts_per_tok = 10
    cfg.text_config.moe_intermediate_size = 128
    cfg.text_config.tie_word_embeddings = False
    cfg.text_config.mtp_num_hidden_layers = 1
    (path / "config.json").write_text(json.dumps(asdict(cfg)))
    index = path / "model.safetensors.index.json"
    filename = "model.safetensors"
    index.write_text(json.dumps({"weight_map": {"mtp.fc_hidden.weight": filename}}))
    configure_ple_runtime(path, mode="resident")
    assert configure_mtp_runtime(path, enabled=True).enabled
    try:
        mx.random.seed(7)
        model = Model(cfg)
        weights = {
            name: value.astype(mx.bfloat16) if value.dtype == mx.float32 else value
            for name, value in tree_flatten(model.parameters())
        }
        assert any("mtp.fc_hidden.weight" in name for name in weights)
        mx.save_safetensors(str(path / filename), weights, metadata={"format": "mlx"})
        index.write_text(json.dumps({"weight_map": dict.fromkeys(weights, filename)}))
    finally:
        configure_mtp_runtime(path, enabled=False)


def write_sharded(source, output, mode):
    # Flash-Next on the Studios is sharded. Use the same oQ writer on a
    # smaller shard budget so its own weight index reaches the mmap loader.
    with patch.object(oq, "_MAX_SHARD_BYTES", 1 << 20):
        tensors, config, _ = _quantize(
            source, output, quant_format=mode, preserve_mtp=True, group_size=32
        )
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(tensors)
    assert len(set(index["weight_map"].values())) >= 2
    return tensors, config


def decoded_checkpoint(source, target, tensors, config, codec):
    """Decode only the recorded toy matrices. Serve every other tensor as written."""
    target.mkdir()
    for part in source.iterdir():
        if part.is_file() and part.suffix not in (".json", ".safetensors"):
            (target / part.name).symlink_to(part.resolve())
        elif part.suffix == ".json" and part.name not in (
            "config.json",
            "model.safetensors.index.json",
        ):
            shutil.copyfile(part, target / part.name)
    entries = _in_format(config)
    assert entries
    weights = tensors.copy()
    for name, entry in entries.items():
        stored = {
            field: weights.pop(f"{name}.{field}")
            for field in (codec.codes_name, *codec.tensors)
        }
        weights[f"{name}.weight"] = codec.decode(
            stored, entry["bits"], entry["group_size"]
        ).astype(mx.bfloat16)
    cfg = json.loads(json.dumps(config))
    del cfg["omlx_quant_formats"]
    for key in ("quantization", "quantization_config"):
        for name in entries:
            del cfg[key][name]
    (target / "config.json").write_text(json.dumps(cfg))
    filename = "model.safetensors"
    mx.save_safetensors(str(target / filename), weights, metadata={"format": "mlx"})
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(weights, filename)})
    )


def verify_logits(adapter, inputs, cache):
    observed = []
    forward = type(adapter).__call__

    def observe(module, *args, **kwargs):
        if module is adapter:
            observed.append((routing.is_row_exact_armed(), routing._is_armed()))
        return forward(module, *args, **kwargs)

    with patch.object(type(adapter), "__call__", observe):
        logits, _, _ = _call_backbone(adapter, inputs, cache, n_confirmed=1)
    assert observed == [(inputs.shape[0] == 1, inputs.shape[0] > 1)], inputs.shape
    return logits


def run_routes(engine, mode, expected_layers):
    from mlx_vlm.models.qwen4_exp.language import get_ple_runtime_mode

    assert get_ple_runtime_mode() == mode
    adapter = engine._adapter
    assert adapter.mtp is not None
    table = adapter.layers[0].ple.ple_embedding.ngram_embedding
    assert type(table).__name__ == (
        "DiskBackedShardedEmbedding" if mode == "mmap" else "ShardedEmbedding"
    )
    layers = {
        name: layer
        for name, layer in engine.vlm_model.named_modules()
        if isinstance(layer, (FormatLinear, FormatSwitchLinear))
    }
    assert written_names(layers.keys()) == expected_layers, (
        sorted(layers),
        sorted(expected_layers),
    )
    ids = ((mx.arange(48) * 5 + 7) % 58).astype(mx.int32)
    outputs = {}

    def record(name, logits, rows, width):
        assert logits.shape == (rows, width, 64), (name, logits.shape)
        values = np.asarray(logits.astype(mx.float32))
        assert np.isfinite(values).all(), name
        outputs[name] = values

    def prefix(length):
        cache = adapter.make_cache()
        adapter.set_text_prefill_rope_delta(0.0)
        logits = adapter(ids[None, :length], cache=cache)
        mx.eval(logits)
        return cache, logits

    cache, logits = prefix(24)
    record("prefill", logits, 1, 24)
    # Take a separate prefix through serving's cache handlers. The live
    # decode can then update its recurrent buffers without touching this prefix.
    prefix_to_save, _ = prefix(24)
    reconstructed = []
    for layer_cache in prefix_to_save:
        kind = type(layer_cache).__name__
        handler = CacheTypeRegistry.get_handler_by_class_name(kind)
        state = handler.extract_state(layer_cache)
        kwargs = {"token_count": 24} if CacheTypeRegistry.is_arrays_family(kind) else {}
        reconstructed.append(handler.reconstruct_cache(state, **kwargs))
    restored = adapter.restore_cache(reconstructed)

    def positions(n):
        uids = list(range(n))
        for uid in uids:
            adapter.mark_text_positions(uid)
        adapter.set_step_rope_deltas(mx.zeros((n,)), uids)

    for name, live in (("decode", cache), ("restore", restored)):
        positions(1)
        out, hidden, _ = _call_backbone(adapter, ids[None, 24:25], live)
        record(name, out, 1, 1)
        if name == "decode":
            proposal = adapter.mtp_forward(
                hidden[:, -1:], ids[None, 25:26], adapter.make_mtp_cache()
            )
            record("draft", proposal, 1, 1)
    np.testing.assert_array_equal(outputs["restore"], outputs["decode"])
    for width in range(1, 17):
        for starts in ((12,), (12, 17)):
            caches = [prefix(pos)[0] for pos in starts]
            batched = _patched_merge_caches(caches)
            positions(len(starts))
            inputs = mx.stack([ids[pos : pos + width] for pos in starts])
            if width == 1:
                logits = adapter(inputs, cache=batched)
            else:
                logits = verify_logits(adapter, inputs, batched)
            record(f"batch_{len(starts)}_width_{width}", logits, len(starts), width)
    return outputs


def wrong_draft(engine, mode, expected_layers):
    # Feed the draft projection a wrong column order; target logits use
    # exactly the same weights, prompts and route matrix as the clean run.
    layer = engine._adapter.mtp.fc_hidden
    assert type(layer) is FormatLinear and "mtp.fc_hidden" in expected_layers
    name = formats.tensor_names(layer)[0]
    original = layer[name]
    try:
        layer[name] = mx.concatenate([original[..., 1:], original[..., :1]], axis=-1)
        return run_routes(engine, mode, expected_layers)
    finally:
        layer[name] = original


async def serve(path, scratch, ssd, expected_layers, measure=run_routes):
    init_settings(base_path=scratch / "settings")
    pool = EnginePool()
    try:
        pool.discover_models(str(path.parent))
        settings = ModelSettings(
            qwen4_ple_ssd_offload=ssd, mtp_enabled=True, mtp_fixed_depth=1
        )
        engine = await pool.get_engine(path.name, runtime_settings=settings)
        assert isinstance(engine, VLMBatchedEngine)
        return await asyncio.get_running_loop().run_in_executor(
            get_mlx_executor(),
            measure,
            engine,
            "mmap" if ssd else "resident",
            expected_layers,
        )
    finally:
        await pool.shutdown()


def written_names(names):
    """The written matrices behind the layers a model holds: a fused
    ``gate_up_proj`` is the written gate and up projections of its switch
    block (the engine's legal expert fusion), every other layer its own
    matrix. A tree without the fusion keeps the two, either way."""
    written = set()
    for name in names:
        if name.endswith(".gate_up_proj"):
            base = name[: -len(".gate_up_proj")]
            written.add(f"{base}.gate_proj")
            written.add(f"{base}.up_proj")
        else:
            written.add(name)
    return written


def test_format_write_and_serve_against_its_decoded_checkpoint(
    tmp_path, monkeypatch, caplog
):
    source = tmp_path / "source"
    tiny_source(source)
    models = tmp_path / "models"
    models.mkdir()
    written, reference = models / "encoded", models / "decoded"
    formats.load()
    codec = ExactCodec()
    monkeypatch.setitem(formats._FORMATS, codec.mode, codec)
    with caplog.at_level("INFO", logger=oq.logger.name):
        tensors, config = write_sharded(source, written, codec.mode)
    entries = _in_format(config)
    assert len(entries) == len(codec.encoded) > 0
    reported = f"oQ4: {len(entries)} layers in quant format {codec.mode}"
    assert reported in caplog.messages, reported
    assert any(".switch_mlp." in name for name in entries)
    assert any(tensors[f"{name}.{codec.codes_name}"].ndim == 2 for name in entries)
    decoded_checkpoint(written, reference, tensors, config, codec)
    for ssd in (False, True):
        expected = asyncio.run(serve(reference, tmp_path, ssd, set()))
        actual = asyncio.run(serve(written, tmp_path, ssd, written_names(entries)))
        assert actual.keys() == expected.keys()
        for route, values in actual.items():
            np.testing.assert_array_equal(
                values.view(np.uint32),
                expected[route].view(np.uint32),
                err_msg=route,
            )
        print(f"{ssd=} layers={len(entries)} routes={len(actual)}", flush=True)
    damaged = asyncio.run(
        serve(written, tmp_path, True, written_names(entries), wrong_draft)
    )
    changed = [
        name
        for name in actual
        if not np.array_equal(
            damaged[name].view(np.uint32), actual[name].view(np.uint32)
        )
    ]
    assert changed == ["draft"], changed
    print(
        "wrong mtp.fc_hidden column order changed only the draft logits", flush=True
    )
    for kind in ("format_linear", "format_switch_linear"):
        assert any(
            key[0] == kind and key[1] == codec.face_for_bf16 and proof.ran
            for key, proof in kernels._PROOFS.items()
        ), kind
