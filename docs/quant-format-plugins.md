# Quant-format plugins: design overview

This page summarises the quant-format plugin interface on this fork's `quant-plugins-poc` branch. The contract itself, with every rule and error message, is [quant-formats.md](quant-formats.md). The interface is not part of upstream oMLX. Its contract revision 1 is unreleased and can still change ([Versions](quant-formats.md#versions)).

## Why

A quant format is how a quantized linear's tensors decode into weights. Without a plugin interface, a new format means changing oMLX itself: its loader, its writer and each fused kernel that should run it. The interface moves that work into an installed package:

- **No fork.** A package registers a format through the `omlx.quant_formats` entry point. oMLX finds it at the registry's first use. Nothing in oMLX changes.
- **One kernel family per route.** A fused kernel is a skeleton: it owns the launch shape, the walk over the inputs and what is fused around the product. A format supplies a *face*, a small Metal struct with the loads and arithmetic of one run of weights. Each kernel is compiled once per face, with the face inlined. A kernel tuned once runs every format that has the face, and a format costs what its own arithmetic costs ([Quant formats](quant-formats.md#quant-formats)).
- **Stock formats on the same path.** MLX's affine quantization (oQ and oQe checkpoints) registers through the same `register`, as a built-in. The fused kernels read affine weights through the affine face, which is MLX 0.32.4's `quantized.h` code, so each row has MLX's bits ([Affine](quant-formats.md#affine)).

## How a format plugs in

```mermaid
flowchart LR
    P["plugin package<br/>entry point omlx.quant_formats"] --> R["registry (omlx.formats)<br/>built-in affine, oqh, oqh_affine first, then plugins"]
    R --> F["QuantFormat<br/>layout · decode · encode_layer · record/accept · input_transform"]
    F --> D["Dot face<br/>mat-vec"]
    F --> T["Tile face<br/>weight by weight"]
    D --> K["kernels over faces<br/>skeletons, compiled once per face"]
    T --> K
    K --> L["model layers<br/>FormatLinear, FormatSwitchLinear,<br/>routed decode, verify gather, sorted prefill"]
    F --> W["oQ writer<br/>quantize_oq_streaming(quant_format=...)"]
    W --> C["verified checkpoint<br/>config.json: omlx_quant_formats"]
    C -->|"load: accept(record)"| L
```

A format is a `QuantFormat` with a `mode` (the per-layer `mode` in the quantization config), a codes tensor, side tensors and the contract revision it targets ([Register a format](quant-formats.md#register-a-format)). Its methods:

- `layout(k, n, bits, group_size)`: shape and dtype of each tensor of one `(n, k)` matrix, or `None` for a matrix it does not keep.
- `decode`: the float reference, run where no kernel takes the rows and used by tests ([Decode](quant-formats.md#decode)).
- `encode_layer`: the tensors oQ writes, given oQe's per-input importance and an `EncodeContext`. It is the only writer hook; a format may also declare `shared_tensors` (layer-wide unstacked tensors) and an `InputTransform` (one rotation every route applies, with optional fused producer specs), documented in [quant-formats.md](quant-formats.md#shared-tensors-and-the-input-transform). Built-in `oqh` and `oqh_affine` register beside `affine` using the same contract.
- `record` / `accept`: a checkpoint's own parameters (a fitted level table, say), and a reader configured from them at load.
- `dot` / `tile`: the faces for a layer, or `None`.

A `Dot` face is a per-lane cursor (`Lanes<K>`: `load`, `madd`, `finish`, and `decode`/`apply` to share a block's weights across input vectors). A `Tile` face is a per-row cursor (`Runs<K>`: `params`, `fetch`, `weight`) that decodes runs of 8 codes into threadgroup tiles. Lookup tables are declared as `Table` data, and oMLX stages them for the GPU it runs on ([Faces](quant-formats.md#faces), [Tables](quant-formats.md#tables)).

Routes built on faces ([Kernels built on faces](quant-formats.md#kernels-built-on-faces)):

| Route | Face |
|---|---|
| Format layers: decode steps and short blocks (1 to 16 rows), switch-layer pairs | dot |
| Qwen3.5-family one-token MoE decode, routed and shared experts | dot |
| Qwen3.5-family verify blocks of 2 to 8 rows, routed experts | dot |
| Sorted prefill of format switch layers, tensor-unit GPUs (M5, NAX), with fused SwiGLU | tile |
| Sorted prefill of format switch layers, M3 (steel gather) | tile |

oQ writes a format with `quantize_oq_streaming(..., enhanced=True, quant_format=...)`, given a mode or a configured format object. It offers the matrices oQe has importance for, held by `nn.Linear` or the mlx-lm/mlx-vlm `SwitchLinear`. Declined matrices stay oQ's affine. oQ checks each `encode_layer` result against `layout` and asks for both faces of each encoded layer. It writes to `<output>.unverified`, loads the staged checkpoint, runs a forward, records `omlx_verification` and `omlx_quant_formats` (plugin, contract, record) in `config.json`, and renames the folder ([Quantize a checkpoint](quant-formats.md#quantize-a-checkpoint)).

## What oMLX guarantees, and what it refuses

A format that cannot be shown right is refused, or its layers fall back to a slower composed path.

- **Registration**: one format per mode. Built-ins register first and cannot be replaced. MLX's own modes (`affine`, `mxfp4`, `mxfp8`, `nvfp4`, ...) are reserved for built-ins. A plugin's tensors may not be named `weight`, `scales`, `biases` or `bias`, so code that reads MLX's tensors by hand fails loudly. A plugin that fails to import is logged and skipped, and the server starts ([Register a format](quant-formats.md#register-a-format)).
- **Contract**: a format states its revision in `contract`, each face in `CONTRACT`. A revision outside `OLDEST_CONTRACT..CONTRACT` is refused at registration. A face of another revision fails one `static_assert`. A new revision is an addition, and earlier formats run as before, to the bit ([Versions](quant-formats.md#versions)).
- **Load**: every refusal is one `FormatRefusedError` naming the mode, the checkpoint, the plugin that wrote it and the one installed. Refused: a format not installed, a record `accept` rejects, a layer filled in part, a checkpoint not recorded as verified, a family outside the verified allowlist in `omlx/oq.py` (which oQ applies too). A layer never computes on a placeholder ([Load a checkpoint](quant-formats.md#load-a-checkpoint)).
- **Runtime**: a `dot()` or `tile()` that raises or returns an invalid face counts as `None`, logged once. Each kernel proves its Metal on a zero stand-in before its first real launch. A face that fails to compile is logged, and its layers keep the composed path or `decode` ([Kernels over faces](quant-formats.md#kernels-over-faces)).
- **Compile time**: face sources are C++17 Metal without derived classes. Skeleton `static_assert`s name the broken rule. A face source with a static-storage array is refused: tables are `Table` data, because per-lane reads from `constant` or `device` memory varied 2 to 7% in speed between processes on an M3 Ultra ([Compile-time rules](quant-formats.md#compile-time-rules), [Tables](quant-formats.md#tables)).
- **Exactness**: kernels compile with fast math off. A dot row is a function of the row and the input alone, and `apply(decode(r))` equals `madd(r)` bit for bit, so each vector of a shared-rows launch has the bits of its one-vector call ([The dot face](quant-formats.md#the-dot-face)). A tile `weight` is the format's `decode` in float32, bit for bit, rounded once to the activation type ([The weight](quant-formats.md#the-weight)). The documented exception: an ordinary Qwen dense call of two or more rows through the wide mat-vec kernel can change low bits against each row alone. Row-exact calls, MTP verification included, keep single-row bits ([Layers](quant-formats.md#layers)).

## What a format author writes

The reference's own example, 8-bit codes with one float scale per row:

```python
import mlx.core as mx
from omlx.formats import (
    ApplyRead,
    Dot,
    DotReads,
    LoadRead,
    MaddRead,
    QuantFormat,
    TypeParam,
)

class MyFormat(QuantFormat):
    mode = "myformat"
    codes_name = "codes"
    tensors = ("group_scales",)
    contract = 1

    def layout(self, k, n, bits, group_size):
        if bits != 8:
            return None
        return {"codes": ((n, k), mx.uint8), "group_scales": ((n,), mx.float32)}

    def decode(self, tensors, bits, group_size): ...
    def encode_layer(self, w, importance, context): ...

    def dot(self, layer, k, n):
        if k % 8:  # K must be whole runs of the face's VPT (8)
            return None
        return Dot(
            key="myformat_b8",
            source=SOURCE,
            type="omlx_myformat_b8::Dot",
            tensors=("group_scales",),
            reads=DotReads(
                load=LoadRead(input=TypeParam("XP")),
                madd=MaddRead(params=("acc", "r", "xt", "")),
                apply=ApplyRead(aux=TypeParam("A")),
            ),
        )

FORMAT = MyFormat()
```

```toml
[project.entry-points."omlx.quant_formats"]
myformat = "my_package:FORMAT"
```

`SOURCE` is a Metal struct of about 50 lines ([An example](quant-formats.md#an-example), [An example of a tile face](quant-formats.md#an-example-of-a-tile-face)). `omlx.formats.testing` checks the face rules that can be tested at runtime (`dot_rows`, `shared_rows`, `lane_parts`, `finish_rows`, `tile_rows`, `codes_rows`, `check_constructor_reads`, `staged_tables`, `rows_stack`) ([Test a format](quant-formats.md#test-a-format)).

[omlx-oqel](https://github.com/yohann-bearzi/omlx-oqel) is a complete plugin: Lloyd-codebook formats at 2, 3 and 4 bits, with a scale byte per group of 16 and a bfloat16 row scale. Its level tables are `Table`s, fitted tables travel in the checkpoint record, and `accept` builds a reader from that record. It gives dot and tile faces.

## Scope and status today

On the interface now:

- oQ writes, and oMLX loads, format checkpoints for one family: `qwen4_exp` (Qwen3.8 Flash-Next). Tests can mark tiny checkpoints of other families as verified with `testing.verified`.
- The first real checkpoint is Qwen3.8 Flash-Next with its routed experts in oQeL and the rest oQ4e: written by oQ, loaded, and measured for quality and speed (figures in the omlx-oqel README).
- The routes in the table above. The fused routed decode takes a format's experts only when the gate+up face is exact for exactly one activation dtype; otherwise the block keeps its composed path.

Not yet:

- Other families. GLM-5.2, GLM-5.3, DeepSeek V4, Inkling 2, Raon-1 mini and Clef are refused by name for hand-written readers. DeepSeek V4.1 and Muse Spark use writers of their own.
- Layer types other than `nn.Linear` and `SwitchLinear` (MLA's `MultiLinear`, a family's own switch layer). oQ writes them affine.
- Dense prefill: a dense format linear's calls above 16 rows that no tile or row-exact route takes run `decode`, then MLX's matmul. Correct, and slow.
- Distributed serving (cluster, tensor and pipeline parallel), custom `model_file` models, external drafters, and embedding, reranking and audio readers refuse format checkpoints. MoE expert offload keeps every expert resident.
- Formats whose product is not row by row over blocks of inputs need a new kind of face ([Add to the contract](quant-formats.md#add-to-the-contract)).

## Where to read next

- [Register a format](quant-formats.md#register-a-format), [Versions](quant-formats.md#versions): the package-level contract.
- [The dot face](quant-formats.md#the-dot-face), [The tile face](quant-formats.md#the-tile-face), [Tables](quant-formats.md#tables): what a face must satisfy.
- [Kernels over faces](quant-formats.md#kernels-over-faces), [Kernels built on faces](quant-formats.md#kernels-built-on-faces): how kernels are built and proven.
- [Load a checkpoint](quant-formats.md#load-a-checkpoint), [Quantize a checkpoint](quant-formats.md#quantize-a-checkpoint): the checkpoint lifecycle.
- [Test a format](quant-formats.md#test-a-format), [Affine](quant-formats.md#affine): the checks, and the built-in format as a worked case.
