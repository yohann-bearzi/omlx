# Quant formats

A quant format is how a quantized linear's tensors decode into weights. An installed package can add one: it registers a `QuantFormat` under the `mode` its linears carry in a checkpoint's quantization config.

oMLX's fused kernels are not written for one format. A kernel is a skeleton: it owns the launch shape, the walk over the inputs and what is fused around the product. A format gives a *face*: a small Metal struct with the loads and the arithmetic of one run of weights. A kernel is compiled once per face, with the face inlined. So:

- a kernel tuned for one format runs every format that has the face,
- a format written once runs in every kernel built on that face,
- a format costs what its own arithmetic costs.

| File | What it holds |
|---|---|
| `omlx/formats/__init__.py` | the registry, `QuantFormat`, `Dot` |

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

Installing the package is enough. oMLX reads the `omlx.quant_formats` entry points the first time a format is looked up (`omlx.formats.get(mode)`).

- A mode has one format. A second format for a registered mode is refused.
- oMLX's own formats (`BUILTIN`) register first, through the same `register`, so a plugin cannot replace one.
- A plugin that fails to import or to register is logged (`quant format plugin NAME (VALUE) failed to load`) and skipped. The server still starts.

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
| `tensors` | The layer's parameters beside `weight` that the face reads. They become kernel inputs in this order. Identifiers. |
| `tail` | The face's `Lanes<K>::TAIL` (below). A kernel without a tail path does not take a face that has one. |
| `dtype` | The only activation dtype the face is exact for, or `None` for any. Ask with `face.exact_for(dtype)`: an `mx.Dtype` does not compare with `None`. |

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
| `static constant constexpr int VPT` | Inputs a lane reads per block. |
| `static constant constexpr int BLOCK` | Inputs the simdgroup covers per block: `32 * VPT`. |
| `static constant constexpr bool TAIL` | Whether the walk ends with a block of `n` forms (below). `false` needs K to be a multiple of `BLOCK`. |
| `Lanes(Dot m, size_t row, uint lane)` | Lane `lane` at the first block of row `row`. |
| `struct Row` and `Row row(int r) const` | What `madd` needs of the current block of row `row + r`: pointers, the group's scale. |
| `void step()` | Move to the next block. |
| `template <typename X> static float load(const device X* x, thread float* xt)` | Read the lane's `VPT` inputs into `xt` as floats. Its result is passed to every `madd` of the block as `aux` (affine returns the sum of the inputs, for the bias term). |
| `static float madd(float acc, Row r, thread const float* xt, float aux)` | `acc` plus the lane's share of the row's product for the block. |
| `load(x, xt, int n)` and `madd(acc, r, xt, aux, int n)` | Only with `TAIL`: the same over the lane's first `n` inputs, `0 < n <= VPT`. |
| `float finish(int r, float sum) const` | Row `row + r`'s result from the sum of `acc` over the lanes (for example times a row scale). |

A kernel runs a face in this order, in every lane, for each of its rows `r`:

```
acc[r] = 0
for each whole block:           (all blocks but the last with TAIL)
    aux = load(x, xt)
    acc[r] = madd(acc[r], row(r), xt, aux)
    step()
with TAIL, for the last block, whole or partial, with n > 0 inputs in this lane:
    aux = load(x, xt, n)
    acc[r] = madd(acc[r], row(r), xt, aux, n)
result[r] = finish(r, simd_sum(acc[r]))
```

Rules:

- `BLOCK == 32 * VPT`, and K is a multiple of `BLOCK` unless `TAIL`. A kernel asserts both when it is compiled.
- A row's result is a function of the row and the input only: not of the kernel, of the rows computed with it or of its place in a launch. MLX compiles kernels with fast math off, so the float operations run as written; keep `madd` and `finish` pure and every kernel computes the same bits for a row. oMLX relies on it wherever a fused kernel and another path must agree on a token.
