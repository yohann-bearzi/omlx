# Quant formats

A quant format is how a quantized linear's tensors decode into weights. An installed package can add one: it registers a `QuantFormat` under the `mode` its linears carry in a checkpoint's quantization config.

oMLX's fused kernels are not written for one format. A kernel is a skeleton: it owns the launch shape, the walk over the inputs and what is fused around the product. A format gives a *face*: a small Metal struct with the loads and the arithmetic of one run of weights. A kernel is compiled once per face, with the face inlined. So:

- a kernel tuned for one format runs every format that has the face,
- a format written once runs in every kernel built on that face,
- a format costs what its own arithmetic costs.

MLX's affine quantization (the oQ and oQe checkpoints) is registered the same way as a plugin, so the plugin path is the path those checkpoints take. See [Affine](#affine).

This page is the contract between oMLX and a format. It has a revision number, and a format states the revision it is written for, so that formats keep running as the contract grows. See [Versions](#versions).

| File | What it holds |
|---|---|
| `omlx/formats/__init__.py` | the registry, `QuantFormat`, `Dot` |
| `omlx/formats/kernels.py` | the skeletons and `Matrices` |
| `omlx/formats/linear.py` | the layers that hold a format's tensors |
| `omlx/patches/format_load.py` | what loads a format's checkpoint into them |
| `omlx/formats/testing.py` | checks for a format's own tests |
| `omlx/formats/affine.py` | affine, the built-in format |

## Register a format

```python
# my_package/__init__.py
from omlx.formats import Dot, QuantFormat


class MyFormat(QuantFormat):
    mode = "myformat"      # a linear's "mode" in the quantization config
    tensors = ("scales",)  # its parameters beside "weight" and "bias"
    contract = 1           # the revision of this page it is written for

    def decode(self, tensors, bits, group_size):
        ...                # the weights as floats

    def layout(self, k, n, bits, group_size):
        ...                # what a checkpoint keeps of one matrix

    def dot(self, layer, k, n) -> Dot | None:
        ...                # the face that oMLX's kernels run them through


FORMAT = MyFormat()
```

```toml
# pyproject.toml
[project.entry-points."omlx.quant_formats"]
myformat = "my_package:FORMAT"
```

Installing the package is enough. oMLX reads the `omlx.quant_formats` entry points the first time the registry is used (`omlx.formats.load`, `get`, `dot` or `register`). A kernel patch calls `load` when it is applied, so that a model's first token does not wait for the plugins' imports.

- A mode has one format. A second format for a registered mode is refused.
- A format states the revision of this page it is written for (`contract`). One that states none, or a revision this oMLX does not run, is refused. See [Versions](#versions).
- oMLX's own formats (`BUILTIN`) register first, through the same `register`, so a plugin cannot replace one.
- Plugins register in the order of their entry points' names, then of their values (`my_package:FORMAT`). Of two that claim one mode, the first is kept.
- A plugin is logged when it is loaded, with the distribution that installed it and the modes it registered: `quant format plugin myformat (my-package 0.1.0) registered myformat`.
- A plugin that fails to import or to register is logged (`quant format plugin NAME (VALUE) failed to load`) and skipped, with every format registered while it loaded. The server still starts.
- A plugin's module does not use the registry when it is imported: the entry point registers its format. A format that builds on another looks it up when it is asked for a face (`omlx.formats.get` in `dot()`). At import the registry may be loading: another plugin may not be registered yet, a module still half imported when its own entry point is read can fail to load (it is then skipped), and an import in one thread can deadlock with the first lookup in another.

## Decode

`QuantFormat.decode(tensors, bits, group_size)` returns a matrix's weights as floats, of shape `(..., n, k)`. It is the format's reference: what a layer runs where no kernel takes its rows, and what a format's tests compare a face with.

- `tensors` holds `weight` and the format's `tensors`, by name. They are one matrix's, or those of several: a switch layer passes the experts a call asks for, gathered tensor by tensor (`tensor[indices]`), or its whole stack. Each tensor then has the same axes ahead of its own, and so do the weights returned.
- `bits` and `group_size` are the layer's, from the checkpoint's quantization config.
- The floats are of any float type. A layer casts them to the activations' type ahead of MLX's matmul, so the product has `x`'s dtype.

## Layers

`omlx.formats.linear` has the layers that hold a format's tensors. `FormatLinear` is to a format what `nn.QuantizedLinear` is to MLX's modes, and `FormatSwitchLinear` what mlx-vlm's `QuantizedSwitchLinear` is: a stack of matrices (a switch layer's experts), called with the indices of the ones to run.

```python
from omlx.formats.linear import FormatLinear, FormatSwitchLinear

linear = FormatLinear("myformat", bits, group_size, k, {"weight": codes, "scales": scales})
y = linear(x)  # x @ W.T, plus the bias if the layer is given one

experts = FormatSwitchLinear("myformat", bits, group_size, k, stacked, bias=bias)
y = experts(x, indices)  # as mlx-vlm calls a switch layer
```

A layer keeps the tensors under their names, frozen, with the attributes of MLX's quantized layers (`mode`, `bits`, `group_size`): the registry finds a layer's format by its `mode`, and the format's `dot` reads the rest. Of the [kernels built on faces](#kernels-built-on-faces), the verify gather asks any layer for its face; the routed decode plans over MLX's quantized layers only. A layer is given every tensor the format names, and no other:

- the weight of a matrix has two axes, the first of which is N (the layer's `output_dims`); the second is the format's own. K is given (`input_dims`);
- in a stack every tensor has the stack's axis first, and the weight then two more. What follows the stack's axis in another tensor is the format's own.

| Call | What runs |
|---|---|
| `FormatLinear`, 1 to 16 rows of `x` (`linear.MAX_ROWS`: a decode step and the short blocks around one) | the dot face in the mat-vec skeleton, in one launch. Each row has the bits it has alone. |
| `FormatLinear`, more rows | `decode`, then MLX's matmul |
| `FormatSwitchLinear` | each pair (a row of `x` and its expert) through the dot face, in one launch, as MLX's `gather_qmm` runs a decode step's pairs. Each pair has the bits it has alone. |
| either, with no dot face that is exact for `x`'s dtype, an N that is not a multiple of 8, or a K under 8 | `decode`, then MLX's matmul. A switch layer decodes the matrices its indices name or, from as many indices as its stack has matrices, the stack once (then MLX's `gather_mm`). The bits are those of the MLX kernel the call lands on, not a pair's own. |

A format without a dot face is correct and slow: every call decodes, a switch layer its stack at most. A switch layer's kernel takes any number of pairs, one mat-vec each. Over affine weights on an M5 Max, up to 256 pairs it takes 0.75 to 1.05 of the time of MLX's `gather_qmm`, and at a prefill's 1024 to 8192 pairs MLX's sorted `gather_qmm` is 1.6 to 6.2 times faster. The rows kernel is proved as every kernel in a model's forward (see [Kernels over faces](#kernels-over-faces)): a face that does not compile is logged with the compiler's messages (`quant format face <key>: rows kernel failed; decode fallback`), and the layers of the same face, tensor dtypes and shape run through `decode`.

## Load a checkpoint

mlx-lm and mlx-vlm rebuild a checkpoint's quantized layers with `nn.quantize`: a layer the quantization config names is replaced by its `to_quantized(group_size, bits, mode)`, and the checkpoint's tensors are then loaded over the model's, strictly. oMLX has `nn.Linear` and the `SwitchLinear` of mlx-lm and of mlx-vlm build a `FormatLinear` or a `FormatSwitchLinear` for a mode whose format lays the matrix out:

```python
    def layout(self, k, n, bits, group_size):
        if bits != 8:
            return None
        return {"weight": ((n, k), mx.uint8), "scales": ((n,), mx.float32)}
```

`layout` gives the shape and the dtype of each tensor a checkpoint keeps of one `(n, k)` matrix: its `weight` and the format's `tensors`, by name. A switch layer's stack has its axis ahead of each. The layer is built over placeholders of those shapes, with the bias of the layer it replaces, and the load fills them: a checkpoint whose tensors have other names or other shapes is refused there, by MLX.

`layout` returns None for a matrix the format does not keep (bits it has no codes for, a K its groups do not divide). The load then fails in words: `quant format 'myformat' lays out no (16, 64) matrix of 4 bits in groups of 2`. A format of one of MLX's own modes returns None for the matrices that stay MLX's layers; affine does for all.

A checkpoint names each layer of the format in its quantization config:

```json
"quantization": {
    "group_size": 64, "bits": 4, "mode": "affine",
    "model.layers.0.mlp.gate_proj": {"group_size": 1, "bits": 8, "mode": "myformat"}
}
```

- A layer's entry holds `group_size`, `bits` and `mode`, and nothing else: the loader calls the layer's `to_quantized` with it.
- The top-level `mode` stays one of MLX's. A layer the config does not name takes the top-level values, and the loaders find such a layer by its `scales` tensor.
- A checkpoint of a format that is not installed fails to load: `no quant format is registered for mode 'myformat': install the plugin the checkpoint was written with`.
- A layer of a format is built from its bits and group size. MLX's other arguments (`quantize_input`, mlx-lm's `global_scale`) are refused with a format's mode.
- A model whose layers have a `to_quantized` of their own loads a format once `omlx.patches.format_load` wraps it as it wraps those three.

## Faces

A face is a frozen dataclass. A kernel builder is cached on the faces it is instantiated with, so a format returns equal faces for layers of the same layout (cache them) and a kernel is compiled once per layout.

`QuantFormat.dot(layer, k, n)` returns the mat-vec face of `layer`'s `(..., n, k)` matrix, a `Dot`, or `None`. With `None` the kernel is not used for that layer and the model keeps its composed path. Return `None` for:

- a layout the face does not cover,
- a layer with a tensor of fewer than 8 elements. MLX binds such an input as constants, not as the device pointer a face holds.

Kernels ask through `omlx.formats.dot(layer, k, n)`, inside a model's forward, and keep the answer with what they build for the layer (until one of its tensors is replaced). A `dot()` that raises an exception there, or that returns anything but `None` or a `Dot` of arrays the layer has (`weight` and each of the face's `tensors`, of 8 elements or more each), counts as `None`: the layer keeps its composed path. Each such defect of a format is logged once (`quant format MODE gave no usable dot face ...`, with the traceback). A format's own tests call `FORMAT.dot` directly and take the operands from the layer by the face's `tensors` (see [Test a format](#test-a-format)), so that these defects fail them. Tests that run a model do it inside `omlx.formats.testing.no_dot_defects()`, which fails on leaving the block when a `dot()` was defective in it (an exception of the block's body is raised in its place):

```python
import pytest
from omlx.formats import testing


@pytest.fixture(autouse=True)
def dot_defects():
    with testing.no_dot_defects() as defects:
        yield defects
```

The block's value is the set of the defects met so far, each as `(mode, repr(error))`; a test that expects one removes it. A kernel keeps the answer with what it builds for the layer, so only the first forward that asks for a layer's face meets its defect: a model that a module or session fixture runs in its setup (or a function fixture set up ahead of this one) meets it outside this block.

| `Dot` field | Meaning |
|---|---|
| `key` | An identifier. Names the face in kernel names and compiler messages. |
| `source` | Metal at global scope that declares the face's struct, in its own namespace (`omlx_<key>`). A kernel includes each distinct source once. |
| `type` | The struct, as a type expression for a kernel body, where `T` names the activations' float type. For example `omlx_myformat_b8::Dot`. |
| `tensors` | The layer's parameters beside `weight` that the face reads, as a tuple. They become kernel inputs in this order. Identifiers, each named once. |
| `dtypes` | The activation dtypes the face is exact for, as a tuple of distinct `mx.Dtype`; empty (the default) for any. Their order is part of the face: a format lists them in one order. Ask with `face.exact_for(dtype)`. |

## The dot face

The struct named by `type` is the matrix. It is an aggregate: the weight as bytes, then one device pointer per name in `tensors`, in order. Each pointer has its tensor's element type (`float` for float32, `T` for a tensor in the activations' dtype). Ahead of them it states the revision of this page the face is written for, as the constant `CONTRACT` (see [Versions](#versions)).

```cpp
struct Dot {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device float* scales;

  template <int K>
  struct Lanes { ... };
};
```

`Lanes<K>` is one simdgroup lane's cursor over rows of K inputs. A simdgroup's 32 lanes split each block of inputs: lane `l` reads inputs `l * VPT` to `l * VPT + VPT - 1` of the block.

| Member of `Lanes<K>` | Role |
|---|---|
| `static constant constexpr int VPT` | Inputs a lane reads per block. K is a multiple of it. |
| `static constant constexpr int BLOCK` | Inputs the simdgroup covers per block: `32 * VPT`. |
| `Lanes(Dot m, size_t row, uint lane)` | Lane `lane` at the first block of row `row`. |
| `struct Row` and `Row row(int r) const` | What `madd` needs of the current block of row `row + r`: pointers, the group's scale. |
| `void step()` | Move to the next block. |
| `template <typename X> static A load(const device X* x, thread float* xt)` | Read the lane's `VPT` inputs into `xt` as floats. Its result is passed to every `madd` of the block as `aux`. `A` is the face's choice: a `float` (affine returns the sum of the inputs, for the bias term) or a struct of its own. |
| `static float madd(float acc, Row r, thread const float* xt, A aux)` | `acc` plus the lane's share of the row's product for the block. |
| `float finish(int r, float sum) const` | Row `row + r`'s result from the sum of `acc` over the lanes (for example times a row scale). |

A kernel runs a face in this order, in every lane, on the rows `r` it computes together:

```
acc[r] = 0                                   for each r
for each whole block of K:
    aux = load(x, xt)                        once
    acc[r] = madd(acc[r], row(r), xt, aux)   for each r
    step()                                   once
if K ends inside a block, in the lanes that have inputs in it:
    aux = load(x, xt)                        once
    acc[r] = madd(acc[r], row(r), xt, aux)   for each r
result[r] = finish(r, simd_sum(acc[r]))      for each r
```

Rows of two matrices (the skeleton's two packs) have a cursor each, and the block's one `load`.

K need not be a multiple of `BLOCK`. Where it is not, the last block is partial: its first `K % BLOCK / VPT` lanes have their `VPT` inputs, the other lanes have none and skip it. A face only ever sees whole runs of `VPT` inputs, and the lanes of the last block are known when the kernel is compiled.

Rules:

- `BLOCK == 32 * VPT`, and K is a multiple of `VPT`. A kernel asserts both when it is compiled, so a format's `dot()` returns `None` for a K that is not.
- A row's result is a function of the row and the input only: not of the kernel, of the rows computed with it or of its place in a launch. MLX compiles kernels with fast math off, so the float operations run as written; keep `madd` and `finish` pure and every kernel computes the same bits for a row. oMLX relies on it wherever a fused kernel and another path must agree on a token.
- A face never writes the matrix: its members store to nothing but the cursor's own state and `xt`.
- The constructor and `step()` only move pointers and read nothing. A kernel may build a cursor on a row past the matrix (for an empty pack of rows) and step it into a block it then skips. `row`, `load`, `madd` and `finish` run only for rows and blocks the kernel computes; `row` is `const`.
- No member uses a threadgroup operation (a barrier) or threadgroup memory: the simdgroups of a kernel need not all run the skeleton, or run it over the same K.
- `row`, `load` and `madd` use a simdgroup operation (`simd_sum`, `simd_shuffle`) only in a face that takes whole blocks alone (its format's `dot()` never returns it for a K that ends inside a block): in a last, partial block only some of the lanes run them.
- `finish` runs where the walk left the cursor, which is not one place: past the last whole block, or on a last, partial one. It reads only what `step()` does not move, and nothing that depends on the lane: after it every lane holds the row's result.
- `r` is a constant at every call, an element of the kernel's pack of rows. `row` and `finish` do arithmetic on it that folds (`r * K`), not a loop over the rows.
- The source is C++17 Metal without derived classes. See [Compile-time rules](#compile-time-rules).

### An example

The format `tests/test_formats_kernels.py` runs: 8-bit codes around 128, one float scale per row, K a multiple of 8.

```cpp
namespace omlx_myformat_b8 {
struct Dot {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device float* scales;

  template <int K>
  struct Lanes {
    static constant constexpr int VPT = 8;
    static constant constexpr int BLOCK = 32 * VPT;

    const device uint8_t* w;
    const device float* s;

    Lanes(Dot m, size_t row, uint lane)
        : w(m.codes + row * K + lane * VPT), s(m.scales + row) {}

    struct Row {
      const device uint8_t* w;
    };
    Row row(int r) const { return {w + r * K}; }
    void step() { w += BLOCK; }

    template <typename X, int... I>
    static void copy(omlx_ints<I...>, const device X* x, thread float* xt) {
      ((xt[I] = float(x[I])), ...);
    }
    template <int... I>
    static float dot(omlx_ints<I...>, const device uint8_t* w, thread const float* xt) {
      return (... + ((float(w[I]) - 128.0f) * xt[I]));
    }

    template <typename X>
    static float load(const device X* x, thread float* xt) {
      copy(omlx_iota<VPT>(), x, xt);
      return 0;
    }
    static float madd(float acc, Row r, thread const float* xt, float) {
      return acc + dot(omlx_iota<VPT>(), r.w, xt);
    }

    float finish(int r, float sum) const { return sum * s[r]; }
  };
};
}  // namespace omlx_myformat_b8
```

```python
Dot(
    key="myformat_b8",
    source=SOURCE,
    type="omlx_myformat_b8::Dot",
    tensors=("scales",),
)
```

## Kernels over faces

A kernel names each matrix it reads in a `Matrices`. That gives the matrix's kernel inputs, the Metal to include and the statements that declare the matrices in the kernel body. A matrix's inputs are `<name>_weight` and `<name>_<tensor>`. `add` refuses a name that is not an identifier, and a matrix whose name or inputs repeat those of a matrix added before. The kernel's other names (`x`, its outputs, its body's locals) are the kernel's to keep apart from them.

```python
import mlx.core as mx
from omlx.formats import kernels

matrices = kernels.Matrices()
inputs = ["x", *matrices.add("m", face)]  # ["x", "m_weight", "m_scales"]

kernel = mx.fast.metal_kernel(
    name=f"two_rows_{face.key}",
    input_names=inputs,
    output_names=["y"],
    header=matrices.header,
    source=matrices.declarations + r"""
    const uint lane = thread_index_in_simdgroup;
    const size_t row = size_t(threadgroup_position_in_grid.y) * 2;
    float result[2];
    qmv_rows<T, K, 2, 0>(m, row, m, row, x, lane, result);
    if (lane == 0) {
      y[row] = static_cast<T>(result[0]);
      y[row + 1] = static_cast<T>(result[1]);
    }
""",
)

y = kernel(
    inputs=[x, weight, scales],
    template=[("T", x.dtype), ("K", k)],
    grid=(32, n // 2, 1),
    threadgroup=(32, 1, 1),
    output_shapes=[(n,)],
    output_dtypes=[x.dtype],
)[0]
```

`qmv_rows<T, K, NA, NB>(ma, row_a, mb, row_b, x, lane, result)` is one simdgroup's work: the K inputs at `x` against rows `row_a` to `row_a + NA - 1` of `ma` (`result[0]` to `result[NA - 1]`) and rows `row_b` to `row_b + NB - 1` of `mb` (the next `NB` results). Every lane of the simdgroup calls it; after the call every lane holds the rows' results. `ma` and `mb` have the same face. Adjacent rows of one matrix go in one pack, as above. The second pack is for rows elsewhere: another matrix's (a gate and an up projection), or a run of the same matrix that does not follow the first. It is empty (`NB = 0`) when there are none. A kernel that reads two formats calls it once per format.

What the kernel does around the call is its own: which rows a simdgroup takes, the activation, the combine, the stores. For the plain store, a kernel adds `kernels.STORE_ROWS` to its header: `store_rows(omlx_iota<N>(), result, y + row)` rounds a simdgroup's `N` results to the output's type. The kernel calls it from one lane (the example above stores its two rows by hand).

MLX compiles a kernel when it first evaluates it, which in a model is wherever the graph is evaluated: too late to fall back. So a kernel in a model's forward evaluates a launch inside the call that builds it first, and keeps what it learns in a `kernels.Proof`: the launches that compiled (`ran`) and whether one failed (`failed`). `kernels.proof(kernel, ...)` gives the one proof of a key: the kernel's name, its faces and what else its source is instantiated with (the tensors' dtypes, the shapes), so that layers of the same layout share it. Once a launch has failed, whatever the failure of its first evaluation was, what shares the proof keeps its composed path until the process ends.

## Compile-time rules

Kernels are compiled on the user's Mac by the Metal compiler when they are first run. Metal 4.1 is C++17. It has fold expressions, `if constexpr`, `constexpr` functions, `static_assert`, alias templates and packs of integers. It has no `template for`, pack indexing, concepts, `consteval`, `auto` parameters or derived classes.

- What is known when a kernel is compiled is a template parameter or a `constexpr`: K, the bit width, the group size, the rows of a launch.
- A choice between code paths is `if constexpr`: a kernel whose K is whole blocks has no code for a last, partial one.
- A source's constants are `constexpr`. It defines no macro, or `#undef`s it before it ends: a macro reaches past the source's namespace, into every source, skeleton and kernel body after it.
- A run over a fixed count is a fold over an index pack, unless the loop measures faster. `omlx_iota<N>()` is `omlx_ints<0, ..., N - 1>`; a format's source may use both:

```cpp
template <int... I>
static float dot(omlx_ints<I...>, const device uint8_t* w, thread const float* xt) {
  return (... + ((float(w[I]) - 128.0f) * xt[I]));
}
// dot(omlx_iota<VPT>(), w, xt)
```

The walk over the blocks of K in `qmv_rows` is a loop: unrolled into a fold, the fused Qwen decode ran 3 to 8% slower on an M5 Max. The rows of a block are folds.

## Reading compile errors

MLX compiles a kernel at the first `mx.eval` of its output. A failure is a `RuntimeError`, `[metal::Device] Unable to build metal library from source`, followed by the compiler's messages. Each message cites one of:

| Cited as | What it is |
|---|---|
| `<quant format KEY>` | The `source` of the face with that key, with its own line numbers. |
| `<omlx skeletons, docs/quant-formats.md>` | The skeletons in `omlx/formats/kernels.py`. The line that fails is a use of the face: read the message, then this page. |
| `<kernel header>` | What a kernel adds to its header after `Matrices.header`, MLX's generated signature and the matrix declarations. |
| `<kernel>` | The kernel body, with its own line numbers. |

The messages quoted below are the compiler's for the example above with one mistake each, in the kernel of `dot_rows` ([Test a format](#test-a-format)) with `packs=(1, 1)`: hence `qmv_rows<float, 512, 1, 1, ...>`.

A face that breaks a rule the compiler cannot see fails with the rule:

```
<omlx skeletons, docs/quant-formats.md>:96:5: error: static_assert failed due to requirement
'Lanes<512>::BLOCK == 32 * Lanes<512>::VPT' "dot face: Lanes<K>::BLOCK must be VPT inputs for
each of a simdgroup's 32 lanes"
<kernel>:5:5: note: in instantiation of function template specialization
'qmv_rows<float, 512, 1, 1, omlx_myformat_b8::Dot>' requested here
```

```
<omlx skeletons, docs/quant-formats.md>:100:5: error: static_assert failed due to requirement
'60 % Lanes<60>::VPT == 0' "dot face: K must be a whole number of a lane's VPT inputs"
```

A mistake inside the format is cited in the format:

```
<quant format myformat_b8>:21:36: error: use of undeclared identifier 'weights'
```

A missing or mismatched member is cited where the skeleton uses it, with a note on the format's line when there is one:

```
<omlx skeletons, docs/quant-formats.md>:67:20: error: no member named 'finish' in
'omlx_myformat_b8::Dot::Lanes<512>'
```

```
<omlx skeletons, docs/quant-formats.md>:28:59: error: too many arguments to function call,
expected 3, have 4
<quant format myformat_b8>:38:18: note: 'madd' declared here
```

The last one is a `madd` declared without its `aux` parameter.

## Test a format

`omlx.formats.testing.dot_rows` runs a face in the row skeleton, outside any model:

```python
import mlx.core as mx
from omlx.formats import testing


def test_rows_are_the_dequantized_products():
    face = FORMAT.dot(layer, k, n)
    operands = [layer[name] for name in ("weight", *face.tensors)]
    y = testing.dot_rows(face, operands, x, n, packs=(2, 2))
    assert mx.array_equal(y, dequantize(layer) @ x)
```

`operands` are the weight, then the face's tensors in order. `x` holds K activations. `packs=(NA, NB)` are the rows a simdgroup computes: `NA` rows as the first pack, the `NB` after them as the second. A kernel puts such adjacent rows in one pack; `dot_rows` splits them to run both. `n` is one or more times `NA + NB`, and no more than the matrix's rows (which `dot_rows` does not check). `x` and each operand have 8 elements or more: MLX binds a smaller input as constants.

Worth testing for every layout a format gives a face for:

- rows against the format's reference, bit for bit when the format has an exact composed path,
- the same rows with different `packs`: they must not change by a bit,
- a K of whole blocks and a K that ends in a partial block, if the face takes both,
- the rows in fp32 too, if the reference can be computed in fp32: rounded to bf16 or fp16, a sum in another order than the reference's shows in one row in a thousand or fewer, in fp32 in most rows. `dot_rows` compiles the skeleton for `x.dtype` and refuses a dtype the face is not exact for. For a face whose tensors are typed `T`, `dataclasses.replace(face, dtypes=(mx.float32,))` runs the same traversal in fp32, on fp32 activations and fp32 copies of those tensors.

## Versions

This page is a contract: a format written to it keeps running as oMLX changes. The contract is what a format uses and what oMLX asks of it:

- `QuantFormat`, `Dot`, `register`, `load`, `get`, `dot`, `CONTRACT` and `OLDEST_CONTRACT` in `omlx.formats`, `dot_rows` and `no_dot_defects` in `omlx.formats.testing`, and the `omlx.quant_formats` entry-point group,
- the members of a face, the order a kernel runs them in and the rules a face follows,
- the names of oMLX's Metal that a format's source may use: `omlx_ints` and `omlx_iota`.

The contract has a revision, a whole number that goes up by one with each change. `omlx.formats.CONTRACT` is the revision this oMLX implements. `omlx.formats.OLDEST_CONTRACT` is the oldest revision whose formats it still runs.

| Revision | In oMLX since | What it adds |
|---|---|---|
| 1 | not released yet | The registry, the dot face, `omlx.formats.testing`. |

A format states the revision it is written for in two places: `contract` on its `QuantFormat`, and `CONTRACT` in the struct of each face. oMLX reads the first when the format registers. A kernel reads the second when it is compiled, and calls on the face the members of that revision and of none after it. A format written for one revision has the same number in both, and no face states a later revision than its format.

```python
class MyFormat(QuantFormat):
    contract = 1
```

```cpp
struct Dot {
  static constant constexpr int CONTRACT = 1;
```

oMLX runs the formats of revisions `OLDEST_CONTRACT` to `CONTRACT`. A format of another revision is refused when it registers. For a plugin that means it is logged and skipped, and oMLX runs as it does without the plugin:

```
quant format 'myformat' is written for revision 2 of the format contract; this oMLX runs
revision 1: update oMLX
```

A face of another revision does not compile. The requirement that fails is the face's revision against oMLX's, here a face of revision 2 on an oMLX of revision 1. It is the only error: a kernel checks the revision first, and asks nothing else of a face whose revision it does not run.

```
<omlx skeletons, docs/quant-formats.md>:75:3: error: static_assert failed due to requirement
'2 <= 1' "dot face: written for a later revision of the contract than this oMLX runs"
<omlx skeletons, docs/quant-formats.md>:93:9: note: in instantiation of template class
'omlx_revision<2>' requested here
<kernel>:5:5: note: in instantiation of function template specialization
'qmv_rows<float, 512, 1, 1, omlx_myformat_b8::Dot>' requested here
```

A registered format meets this only with a face that does not state the format's revision. The kernel's launch then fails as it does for any face that does not compile: the routed decode logs the message and the layer keeps its composed path (see [Kernels built on faces](#kernels-built-on-faces)).

What a revision can change:

- **An addition** is a new revision: a member of a face, a kind of face, a name in `omlx.formats`. Formats of earlier revisions run as before, to the bit. A kernel calls a member added in revision N only on a face that states N or later, so a format moves to a later revision when it wants what that revision adds, and not before. Revisions add up: a face that states N has every member that the revisions up to N give its kind of face.
- **A change to something formats rely on** (a member's signature, the run order, a rule) is a new revision that also raises `OLDEST_CONTRACT`: every format of an earlier revision is refused until it is updated. Where an addition can do the job, it is done as one: a new member beside the old one.
- **oMLX's own side** is not in the contract and changes without a revision: which kernels are built on a face, their launch shapes, `Matrices`, `qmv_rows` and the rest of `omlx.formats.kernels` (see [Kernels over faces](#kernels-over-faces)), speed.

A format states the oldest revision that has everything it uses: it then runs on every oMLX from that revision's first release until `OLDEST_CONTRACT` passes it. A format that can use a later revision's additions and still run without them reads `omlx.formats.CONTRACT` and states the newest revision it knows that this oMLX runs. It never states `omlx.formats.CONTRACT` itself: a later oMLX would then call members the format does not have. oMLX's revision is not a name that a format's Metal can read: the format writes the revision it picks into each face's `CONTRACT`, and its source branches on that with `if constexpr`.

A format imports a name that a later revision added where it uses the name, not at the top of its module. At the top, an oMLX without the name fails the plugin's load with an `ImportError`, in place of the refusal above and its advice.

A revision that no release of oMLX has shipped can still change. Revision 1 is one.

### Add to the contract

The contract holds what a format uses, and grows when a format needs more. To add a member to a face:

1. In the skeleton, call the member behind the revision that adds it. Faces of earlier revisions compile as before, without it:

   ```cpp
   if constexpr (D::CONTRACT >= 2) {
     ...  // the call, and everything that uses its result
   }
   ```

   The gate discards the call where the face `D` is a template parameter: in a skeleton, never in a kernel body. `omlx_qmv_block`, which calls `load`, `row` and `madd`, has the lanes' type and not the face's: `D` becomes a template parameter of it with the first member gated there.

2. Raise `CONTRACT` in `omlx/formats/__init__.py`, add a row to the table above, update the messages this page quotes (their revisions and, where the skeleton's lines moved, the lines they cite: the tests reproduce each one), and document the member on this page with its revision.
3. Test it with a face of the new revision that uses the member, with one that states the new revision and lacks the member (it must fail to compile, with a message that names the member), and with faces of earlier revisions: their rows must not change by a bit. Affine's rows stay MLX's.
4. Measure the kernels built on the face against the commit before, in pairs, on the formats that do not have the member. It must cost them nothing.

No member is added ahead of a format that uses it.

A format whose product is not computed row by row over blocks of the inputs (one that shares partial products between rows through a codebook, for example) does not fit the dot face at any revision. It needs a kind of face of its own, and skeletons built on it: a larger change to oMLX, and still an addition.

## Affine

`omlx/formats/affine.py` registers MLX's affine quantization (`mode="affine"`) as a format, written for revision 1 of this page. Its face is the MLX 0.32.4 `quantized.h` code: a row computed through it has the bits MLX gives one row of activations, through `mx.quantized_matmul` or, for stacked expert weights, `mx.gather_qmm`. Its `decode` is `mx.dequantize`.

An affine checkpoint's linears stay MLX's layers (`nn.QuantizedLinear`, `QuantizedSwitchLinear`), which the kernels below read through the affine face. oMLX's tests run the affine format through [the format layers](#layers) as well, against MLX's layers bit for bit: the layers are tested on a format oMLX ships.

| | Covered |
|---|---|
| Bits | 4, 5, 6, 8 |
| Group size | 32, 64, 128 |
| Scales and biases | both bf16 or both fp16. The face is exact for activations of that dtype (its `dtypes`); no kernel is built on it for another. |
| Traversal | MLX's `qmv_fast` when K is a multiple of its block (512 at 4 and 5 bits, 256 at 6 and 8) and, for stacked expert weights, N is a multiple of 8. MLX's `qmv` otherwise. |
| Face key | `affine_b{bits}g{group_size}f` (`qmv_fast`) or `affine_b{bits}g{group_size}s` |

Any other affine layer has no face and runs MLX's ops:

- another bit width or group size, fp32 scales, biases of another dtype than the scales, no scales or no biases, a K that is not whole groups, weights that are neither 2-D nor 3-D or that are not the N by K asked for,
- a matrix of fewer than 8 groups, counted in one expert for stacked weights (MLX binds an input of fewer than 8 elements as constants, not as a buffer),
- a 2-D matrix of K 64 or 128 at 4 or 8 bits: MLX runs one row of it through `qmv_quad`, which sums in another order than `qmv`.

The `qmv_fast` face does not compile for a K that is not whole blocks: `AffineFormat.dot` is what gives a matrix its traversal.

## Kernels built on faces

| Kernel | Face | Code |
|---|---|---|
| Qwen3.5-family one-token MoE decode: routed and shared expert rows | dot | `omlx/patches/qwen35_moe_routed_decode.py` |
| Qwen3.5-family verify blocks of 2 to 8 rows: routed expert rows, grouped by expert | dot | `omlx/patches/moe_verify_gather.py` |
| A format's own layers: the rows of a decode step or of a short block, a switch layer's pairs | dot | `omlx/formats/linear.py` |

The routed decode runs a block in the one dtype its gate+up face is exact for. With a gate+up face exact for any dtype, or for several, the block keeps its composed path, as it does with a down face that is not exact for that dtype. Its launches take rows in whole groups: a block whose expert width or hidden size is not a multiple of 4 keeps its composed path too (every affine layout is whole groups), and a shared expert whose width is not is left out of the fused launches.

A face that does not compile does not fail the model. The decode evaluates a launch inside the call the first time blocks of the same faces, tensor dtypes and shapes build it, once the block's input is evaluated (a failure ahead of the block is raised as it is). If a one-token launch fails, the compiler's messages are logged (`fused routed-expert decode failed; composed fallback`) and those blocks keep their composed path. If the launch of a verify window or of batched rows fails (`fused verify-window experts failed; verifier fallback`), those blocks keep the composed path for windows and batched rows, and the fused one for one token. Blocks of other faces or shapes are not affected.

The verify gather runs a block's fused gate+up projection, then its down projection, when both are stacked expert weights without a bias, of an N that is whole tiles of 8 outputs, each with a dot face that is exact for the block's dtype, and the block's activation returns that dtype in the shape of the up's rows. It proves its launches as the routed decode does: a block that fails (`expert-ordered verify gather failed; composed fallback`) keeps the projections of the same faces, tensor dtypes and shapes on their composed path.
