# Quant formats

A quant format is how a quantized linear's tensors decode into weights. An installed package can add one: it registers a `QuantFormat` under the `mode` its linears carry in a checkpoint's quantization config.

oMLX's fused kernels are not written for one format. A kernel is a skeleton: it owns the launch shape, the walk over the inputs and what is fused around the product. A format gives a *face*: a small Metal struct with the loads and the arithmetic of one run of weights. A kernel is compiled once per face, with the face inlined. So:

- a kernel tuned for one format runs every format that has the face,
- a format written once runs in every kernel built on that face,
- a format costs what its own arithmetic costs.

MLX's affine quantization (the oQ and oQe checkpoints) is registered the same way as a plugin, so the plugin path is the path those checkpoints take. See [Affine](#affine).

| File | What it holds |
|---|---|
| `omlx/formats/__init__.py` | the registry, `QuantFormat`, `Dot` |
| `omlx/formats/kernels.py` | the skeletons and `Matrices` |
| `omlx/formats/testing.py` | checks for a format's own tests |
| `omlx/formats/affine.py` | affine, the built-in format |

## Register a format

```python
# my_package/__init__.py
from omlx.formats import Dot, QuantFormat


class MyFormat(QuantFormat):
    mode = "myformat"      # a linear's "mode" in the quantization config
    tensors = ("scales",)  # its parameters beside "weight" and "bias"

    def dot(self, layer, k, n) -> Dot | None:
        ...


FORMAT = MyFormat()
```

```toml
# pyproject.toml
[project.entry-points."omlx.quant_formats"]
myformat = "my_package:FORMAT"
```

Installing the package is enough. oMLX reads the `omlx.quant_formats` entry points the first time the registry is used (`omlx.formats.get`, `dot` or `register`).

- A mode has one format. A second format for a registered mode is refused.
- oMLX's own formats (`BUILTIN`) register first, through the same `register`, so a plugin cannot replace one.
- Plugins register in the order of their entry points' names, then of their values (`my_package:FORMAT`). Of two that claim one mode, the first is kept.
- A plugin that fails to import or to register is logged (`quant format plugin NAME (VALUE) failed to load`) and skipped, with any format its module registered by itself. The server still starts.
- A plugin's module does not use the registry when it is imported: the entry point registers its format. A format that builds on another looks it up when it is asked for a face (`omlx.formats.get` in `dot()`). At import the registry may be loading, and another plugin may not be registered yet.

## Faces

A face is a frozen dataclass. A kernel builder is cached on the faces it is instantiated with, so a format returns equal faces for layers of the same layout (cache them) and a kernel is compiled once per layout.

`QuantFormat.dot(layer, k, n)` returns the mat-vec face of `layer`'s `(..., n, k)` matrix, a `Dot`, or `None`. With `None` the kernel is not used for that layer and the model keeps its composed path. Return `None` for:

- a layout the face does not cover,
- a layer with a tensor of fewer than 8 elements. MLX binds such an input as constants, not as the device pointer a face holds.

| `Dot` field | Meaning |
|---|---|
| `key` | An identifier. Names the face in kernel names and compiler messages. |
| `source` | Metal at global scope that declares the face's struct, in its own namespace (`omlx_<key>`). A kernel includes each distinct source once. |
| `type` | The struct, as a type expression for a kernel body, where `T` names the activations' float type. For example `omlx_myformat_b8::Dot`. |
| `tensors` | The layer's parameters beside `weight` that the face reads, as a tuple. They become kernel inputs in this order. Identifiers, each named once. |
| `dtypes` | The activation dtypes the face is exact for, as a tuple of distinct `mx.Dtype`; empty (the default) for any. Their order is part of the face: a format lists them in one order. Ask with `face.exact_for(dtype)`. |

## The dot face

The struct named by `type` is the matrix. It is an aggregate: the weight as bytes, then one device pointer per name in `tensors`, in order. Each pointer has its tensor's element type (`float` for float32, `T` for a tensor in the activations' dtype).

```cpp
struct Dot {
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
- No member uses a threadgroup operation or threadgroup memory (a barrier): the simdgroups of a kernel need not all run the skeleton, or run it over the same K.
- `row`, `load` and `madd` use a simdgroup operation (`simd_sum`, `simd_shuffle`) only in a face whose `dot()` returns `None` for every K that ends inside a block: in a last, partial block only some of the lanes run them.
- `finish` runs where the walk left the cursor, which is not one place: past the last whole block, or on a last, partial one. It reads only what `step()` does not move, and nothing that depends on the lane: after it every lane holds the row's result.
- `r` is a constant at every call, an element of the kernel's pack of rows. `row` and `finish` do arithmetic on it that folds (`r * K`), not a loop over the rows.
- The source is C++17 Metal without derived classes. See [Compile-time rules](#compile-time-rules).

### An example

The format `tests/test_formats_kernels.py` runs: 8-bit codes around 128, one float scale per row, K a multiple of 8.

```cpp
namespace omlx_myformat_b8 {
struct Dot {
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

A kernel names each matrix it reads in a `Matrices`. That gives the matrix's kernel inputs, the Metal to include and the statements that declare the matrices in the kernel body. A matrix's inputs are `<name>_w` and `<name>_<tensor>`; `add` refuses a matrix that would repeat an input.

```python
import mlx.core as mx
from omlx.formats import kernels

matrices = kernels.Matrices()
inputs = ["x", *matrices.add("m", face)]  # ["x", "m_w", "m_scales"]

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

`qmv_rows<T, K, NA, NB>(ma, row_a, mb, row_b, x, lane, result)` is one simdgroup's work: the K inputs at `x` against rows `row_a` to `row_a + NA - 1` of `ma` (`result[0]` to `result[NA - 1]`) and rows `row_b` to `row_b + NB - 1` of `mb` (the next `NB` results). Every lane of the simdgroup calls it; after the call every lane holds the rows' results. `ma` and `mb` have the same face. Rows of one matrix go in one pack, as above: the second pack is for the rows of another matrix (a gate and an up projection), and is empty (`NB = 0`) without one. A kernel that reads two formats calls it once per format.

What the kernel does around the call is its own: which rows a simdgroup takes, the activation, the combine, the stores.

## Compile-time rules

Kernels are compiled on the user's Mac by the Metal compiler when they are first run. Metal 4.1 is C++17. It has fold expressions, `if constexpr`, `constexpr` functions, `static_assert`, alias templates and packs of integers. It has no `template for`, pack indexing, concepts, `consteval`, `auto` parameters or derived classes.

- What is known when a kernel is compiled is a template parameter or a `constexpr`: K, the bit width, the group size, the rows of a launch.
- A choice between code paths is `if constexpr`: a kernel whose K is whole blocks has no code for a last, partial one.
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
| `<kernel header>` | What a kernel adds to its header, MLX's generated signature and the matrix declarations. |
| `<kernel>` | The kernel body, with its own line numbers. |

A face that breaks a rule the compiler cannot see fails with the rule:

```
<omlx skeletons, docs/quant-formats.md>:80:3: error: static_assert failed due to requirement
'Lanes<512>::BLOCK == 32 * Lanes<512>::VPT' "dot face: Lanes<K>::BLOCK must be VPT inputs for
each of a simdgroup's 32 lanes"
<kernel>:5:5: note: in instantiation of function template specialization
'qmv_rows<float, 512, 1, 1, omlx_myformat_b8::Dot>' requested here
```

```
<omlx skeletons, docs/quant-formats.md>:83:3: error: static_assert failed due to requirement
'60 % Lanes<60>::VPT == 0' "dot face: K must be a whole number of a lane's VPT inputs"
```

A mistake inside the format is cited in the format:

```
<quant format myformat_b8>:20:36: error: use of undeclared identifier 'weights'
```

A missing or mismatched member is cited where the skeleton uses it, with a note on the format's line when there is one:

```
<omlx skeletons, docs/quant-formats.md>:66:20: error: no member named 'finish' in
'omlx_myformat_b8::Dot::Lanes<512>'
```

```
<omlx skeletons, docs/quant-formats.md>:27:59: error: too many arguments to function call,
expected 3, have 4
<quant format myformat_b8>:37:18: note: 'madd' declared here
```

The last one is a `madd` declared without its `aux` parameter.

## Test a format

`omlx.formats.testing.dot_rows` runs a face in the row skeleton, outside any model:

```python
import mlx.core as mx
from omlx.formats import testing


def test_rows_are_the_dequantized_products():
    face = FORMAT.dot(layer, k, n)
    y = testing.dot_rows(face, (layer["weight"], layer["scales"]), x, n, packs=(2, 2))
    assert mx.array_equal(y, dequantize(layer) @ x)
```

`operands` are the weight, then the face's tensors in order. `x` holds K activations. `packs=(NA, NB)` are the rows a simdgroup computes; `n` is a multiple of `NA + NB`.

Worth testing for every layout a format gives a face for:

- rows against the format's reference, bit for bit when the format has an exact composed path,
- the same rows with different `packs`: they must not change by a bit,
- a K of whole blocks and a K that ends in a partial block, if the face takes both.

## Affine

`omlx/formats/affine.py` registers MLX's affine quantization (`mode="affine"`) as a format. Its face is the MLX 0.32.4 `quantized.h` code: a row computed through it has the bits of `mx.quantized_matmul`.

| | Covered |
|---|---|
| Bits | 4, 5, 6, 8 |
| Group size | 32, 64, 128 |
| Scales and biases | both bf16 or both fp16. The face is exact for activations of that dtype (its `dtypes`); no kernel is built on it for another. |
| Traversal | MLX's `qmv_fast` when K is a multiple of its block (512 at 4 and 5 bits, 256 at 6 and 8) and, for stacked expert weights, N is a multiple of 8. MLX's `qmv` otherwise. |
| Face key | `affine_b{bits}g{group_size}f` (`qmv_fast`) or `affine_b{bits}g{group_size}s` |

Any other affine layer has no face and runs MLX's ops: 2 or 3 bits, fp32 scales, no biases, a K that is not whole groups, or a matrix of fewer than 8 groups (MLX binds an input of fewer than 8 elements as constants, not as a buffer).

The `qmv_fast` face does not compile for a K that is not whole blocks: `AffineFormat.dot` is what gives a matrix its traversal.

## Kernels built on faces

| Kernel | Face | Code |
|---|---|---|
| Qwen3.5-family one-token MoE decode: routed and shared expert rows | dot | `omlx/patches/qwen35_moe_routed_decode.py` |

The routed decode runs a block in the one dtype its gate+up face is exact for. With a gate+up face exact for any dtype, or for several, the block keeps its composed path.
