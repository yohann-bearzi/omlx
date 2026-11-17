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
| `omlx/formats/__init__.py` | the registry, `QuantFormat`, `Dot`, `Tile`, `FormatRefusedError` |
| `omlx/formats/kernels.py` | the skeletons and `Matrices` |
| `omlx/formats/linear.py` | the layers that hold a format's tensors |
| `omlx/patches/format_load.py` | what loads a format's checkpoint into them |
| `omlx/oq.py` | oQ, which writes a checkpoint in a format |
| `omlx/formats/testing.py` | checks for a format's own tests |
| `omlx/formats/affine.py` | affine, the built-in format |

## Register a format

```python
# my_package/__init__.py
from omlx.formats import Dot, QuantFormat, Tile


class MyFormat(QuantFormat):
    mode = "myformat"      # a linear's "mode" in the quantization config
    tensors = ("scales",)  # its parameters beside "weight" and "bias"
    contract = 1           # the revision of this page it is written for

    def decode(self, tensors, bits, group_size):
        ...                # the weights as floats

    def layout(self, k, n, bits, group_size):
        ...                # what a checkpoint keeps of one matrix

    def encode(self, w, importance, bits, group_size):
        ...                # the tensors oQ writes of one matrix

    def dot(self, layer, k, n) -> Dot | None:
        ...                # the face that oMLX's kernels run them through

    def tile(self, layer, k, n) -> Tile | None:
        ...                # the face that oMLX's tile kernels decode them through

    def record(self):
        ...                # what a checkpoint keeps of the format ({})

    def accept(self, record):
        ...                # the format that reads a checkpoint (itself)


FORMAT = MyFormat()
```

```toml
# pyproject.toml
[project.entry-points."omlx.quant_formats"]
myformat = "my_package:FORMAT"
```

Installing the package is enough. oMLX reads the `omlx.quant_formats` entry points the first time the registry is used (`omlx.formats.load`, `get`, `dot`, `tile` or `register`). A kernel patch calls `load` when it is applied, so that a model's first token does not wait for the plugins' imports.

- A mode has one format. A second format for a registered mode is refused.
- A format states the revision of this page it is written for (`contract`). One that states none, or a revision this oMLX does not run, is refused. See [Versions](#versions).
- oMLX's own formats (`BUILTIN`) register first, through the same `register`, so a plugin cannot replace one.
- MLX's own modes (`affine`, `mxfp4`, `mxfp8`, `nvfp4`, any mode `mx.quantize` takes) are oMLX's built-in formats' only: a checkpoint of one of them loads as MLX's layers wherever no built-in format lays its matrix out. Another format under one is refused: `quant format 'mxfp4': 'mxfp4' is one of MLX's own modes, which only oMLX's built-in formats take: register the format under a mode of its own`.
- Plugins register in the order of their entry points' names, then of their values (`my_package:FORMAT`). Of two that claim one mode, the first is kept.
- A plugin is logged when it is loaded, with the distribution that installed it and the modes it registered: `quant format plugin myformat (my-package 0.1.0) registered myformat`.
- A plugin that fails to import or to register is logged (`quant format plugin NAME (VALUE) failed to load`) and skipped, with every format registered while it loaded. The server still starts. A checkpoint of a mode it registered, or handed to `register` (which refused it), before it failed is then refused with what failed (the first such plugin's, in the order plugins register; none once a later plugin registers the mode; a plugin that fails before it hands a format to `register`, its import for one, names no mode, and its checkpoints are told to install the plugin they record): `quant format 'myformat' of the checkpoint at /models/my-model (written by my-package 0.2.0; installed: none): my-package 0.2.0 is installed but did not load: quant format 'myformat' is written for revision 2 of the format contract; this oMLX runs revision 1: update oMLX`.
- A plugin's module does not use the registry when it is imported: the entry point registers its format. A format that builds on another looks it up when it is asked for a face (`omlx.formats.get` in `dot()`). At import the registry may be loading: another plugin may not be registered yet, a module still half imported when its own entry point is read can fail to load (it is then skipped), and an import in one thread can deadlock with the first lookup in another.

## Decode

`QuantFormat.decode(tensors, bits, group_size)` returns a matrix's weights as floats, of shape `(..., n, k)`. It is the format's reference: what a layer runs where no kernel takes its rows, and what a format's tests compare a face with.

- `tensors` holds `weight` and the format's `tensors`, by name. They are one matrix's, or those of several: a switch layer passes the experts a call asks for, gathered tensor by tensor (`tensor[indices]`), or its whole stack. Each tensor then has the same axes ahead of its own, and so do the weights returned.
- `bits` and `group_size` are the layer's, from the checkpoint's quantization config.
- The floats are of any float type. A layer casts them to the activations' type ahead of MLX's matmul, so the product has `x`'s dtype.

## Layers

`omlx.formats.linear` has the layers that hold a format's tensors. `FormatLinear` is to a format what `nn.QuantizedLinear` is to MLX's modes, and `FormatSwitchLinear` what mlx-vlm's `QuantizedSwitchLinear` is: a stack of matrices (a switch layer's experts), called with the indices of the ones to run.

```python
from omlx.formats import testing
from omlx.formats.linear import FormatLinear, FormatSwitchLinear

with testing.verified("llama"):
    linear = FormatLinear(
        "myformat", bits, group_size, k, {"weight": codes, "scales": scales}
    )
    y = linear(x)

    experts = FormatSwitchLinear("myformat", bits, group_size, k, stacked, bias=bias)
    y = experts(x, indices)
```

A layer keeps the tensors under their names, frozen, with the attributes of MLX's quantized layers (`mode`, `bits`, `group_size`), and the format that reads them (`fmt`: what the checkpoint's [`accept`](#load-a-checkpoint) returned, or the registered format of a mode the layer is built with by hand), whose `dot` reads the rest. Of the [kernels built on faces](#kernels-built-on-faces), the verify gather asks any layer for its face; the routed decode plans over MLX's quantized layers only. A layer is given every tensor the format names, and no other:

- the weight of a matrix has two axes, the first of which is N (the layer's `output_dims`); the second is the format's own. K is given (`input_dims`);
- in a stack every tensor has the stack's axis first, and the weight then two more. What follows the stack's axis in another tensor is the format's own.

A layout that breaks these rules for a matrix is refused, by oQ before it encodes the matrix and by a layer when it is built: tensors of other names than the weight and the format's `tensors`, a weight of other than two axes, or a weight whose first axis is not N (a format packs along K, never along N): `quant format 'myformat' lays out the weight of a (16, 64) matrix as (8, 128): its first axis is N (16); a format packs along K, never along N`.

A format whose tensors are all row-major keeps a matrix's rows apart: where every tensor of `layout(k, n, ...)` has N as its first axis, and `layout(k, 2n, ...)` is the same with 2N, row `r` of the decoded matrix depends on row `r` of each tensor alone. Kernels then stack two matrices of one K along N (a gate and an up projection, fused) and read the stack as one. `formats.stackable(fmt, k, n, bits, group_size)` answers whether a format's matrices may be stacked. It requires a positive N and a layout declaring exactly the weight and the tensors named by `fmt.tensors`. What the layout cannot show is the format's promise, and its tests check it with `omlx.formats.testing.rows_stack(fmt, first, second, k, bits, group_size, x)`: the decode of two stacked matrices is their two decodes, bit for bit, and with `x` so are the dot rows (see [Test a format](#test-a-format)). A format that a layout does not describe whole (a tensor shared by every row, a row's parameters that depend on others) is not stackable: its layout must say so, with a tensor whose first axis is not N.

A switch layer's indices are integers of any integer dtype (mlx-vlm's are `uint32`); others are refused. A negative index counts from the end of the stack, as MLX's indexing does, on every route. A call of 2^30 pairs or more runs through `decode`: MLX's grid is int32.

A layer keeps the format's tensors in the shapes and dtypes it is built with, which the face reads: an array of another shape or dtype assigned to one of them is not taken, and raises nothing, so MLX's `set_dtype` and oMLX's float16 activations recast a format layer's bias alone.

| Call | What runs |
|---|---|
| `FormatLinear`, 1 to 16 rows of `x` (`linear.MAX_ROWS`: a decode step and the short blocks around one) | the dot face, in one launch. Rows in groups of 2, 3 or 4 (the most that divide them with a lane's inputs of the group within `kernels.ROW_VALUES_BUDGET`, 32 values: groups of 4 at 8 values a lane, of 2 at 16) run in the shared-rows skeleton, each block of the matrix decoded once for its group (`decode`, then `apply`); a count that none divides, and one row, run in the mat-vec skeleton, one row a simdgroup pair. Each row has the bits it has alone. A matrix whose N is not whole tiles of 8 runs every count in the shared-rows skeleton, its last tile holding the rows that are left, as MLX's `qmv` does. A face of no `vpt` runs every count in the mat-vec skeleton. |
| `FormatLinear`, more rows | `decode`, then MLX's matmul |
| `FormatSwitchLinear` | each pair (a row of `x` and its expert) through the dot face, in one launch, as MLX's `gather_qmm` runs a decode step's pairs. Each pair has the bits it has alone. |
| either, with no dot face that is exact for `x`'s dtype, or a K under 8; a switch layer, or a layer of a face of no `vpt`, with an N that is not a multiple of 8 | `decode`, then MLX's matmul. A switch layer decodes the matrices its indices name or, from as many indices as its stack has matrices, the stack once (then MLX's `gather_mm`). The bits are those of the MLX kernel the call lands on, not a pair's own. |

A format without a dot face is correct and slow: every call decodes, a switch layer its stack at most. A switch layer's decode outside the rows kernel decodes its whole stack at once for a call of at least 20 pairs per matrix (`linear._AT_ONCE_PAIRS`) when the stack is within 2 GiB (`linear._DECODE_AT_ONCE_BYTES`), counted in the decode's dtype and, when `x`'s differs, in `x`'s too: on Flash-Next's stack (512 matrices of 640 × 2560) with affine's decode and bfloat16 `x`, 40,960 sorted pairs peak at 1.56 GiB above the inputs, in 33 ms on an M5 Max and 43 ms on an M3 Ultra, against 63 and 144 ms in pieces. Below 20 pairs per matrix pieces are faster (at 5,120 pairs a decode at once takes 1.16 times their time on an M5 Max, 1.06 on an M3 Ultra; at 10,240 the two tie on an M5 Max), so the matrices are decoded a piece at a time there, and above the budget: 256 MiB of them (`linear._DECODE_BYTES`) counted as float32 and in `x`'s dtype; over the matrices of the indices (fewer indices than the stack has matrices) always. MLX keeps every piece of one evaluation in flight and frees a buffer when its command buffer completes, so only many small pieces, which split its command buffers, hold less than a decode at once (seven pieces of 81 matrices peaked at 1.86 GiB there): a float32 decode of the same stack peaks at 1.40 GiB instead of 4.69, in 104 ms instead of 119 on an M5 Max and 172 instead of 142 on an M3 Ultra. Over the stack each piece's `gather_mm` takes every pair, so a call costs pieces × the gather's work; each pair keeps the product of the piece that holds its matrix. With a decode that works element by element, as affine's does, a pair keeps the bits of a decode at once: over the stack, and over the matrices of the indices when `x` has one row for each index, as a model calls a switch layer; with more rows MLX's matmul may choose another kernel for a piece's smaller batch. A switch layer's kernel takes any number of pairs, one mat-vec each. Over affine weights on an M5 Max, up to 256 pairs it takes 0.8 to 1.1 of the time of MLX's `gather_qmm`, and at a prefill's 1024 to 8192 pairs MLX's sorted `gather_qmm` is 1.6 to 6.2 times faster. The rows kernel is proved as every kernel in a model's forward (see [Kernels over faces](#kernels-over-faces)): a face that does not compile is logged with the compiler's messages (`quant format face <key>: rows kernel failed; decode fallback`), and the layers of the same face, tensor dtypes and shape run through `decode`. A layer's first launch is safe inside a function that `mx.compile` traces, where `x`, the indices and, in a function compiled over the model's state (`inputs=`), the layer's own tensors cannot be evaluated. The layer proves the launch on a stand-in: the smallest launch that MLX generates the same Metal for, over zeros in the place of `x`, of the indices and of the tensors. It evaluates nothing of the call, and what fails in `x`, in the indices or in a tensor is raised where the caller evaluates the result. The stand-in's zeros are made on the GPU's stream and evaluated by the proof ahead of the launch, and the stream is synchronized: what fails in making them, or what an earlier evaluation left pending on the GPU, is raised by the call and fails no proof. Inside a function compiled over the model's state, such an error leaves the state's arrays as MLX's trace left them, placeholders without a value (MLX restores a compiled function's state only when the function returns): the model is loaded again. The call raises it all the same: an error left pending is an earlier evaluation's, which nothing else would report. A matrix's stand-in is one vector over zeros of each tensor's whole shape, once for a layout and a dtype of `x`: what follows the rows in a matrix's tensors is the format's own, so no part of one is known to be all that a row reads. A stack's stand-in has the fewest matrices that leave each tensor 8 elements (one, unless a tensor has fewer than 8 elements a matrix), and of `x` one row with its pairs where the call has fewer than 8 pairs, or the fewest rows that have 8 pairs between them: MLX binds fewer than 8 indices another way.

A layer builds a launch once for the shapes and dtypes of its arrays, in each thread that calls it, and replays it at every call after (see [Kernels over faces](#kernels-over-faces)): a decode step is a lookup, MLX's replay of the launch and the bias. The layer works out its plan (its face, its proof and the tensors a launch reads) and finds its bias at its first call, and keeps them until one of its arrays or attributes is assigned or removed, as a load, `update`, `set_dtype` or an attribute's write does: a call reads them back in one step. Over affine weights on an M5 Max, a one-row call takes 1.01 of the time of `nn.QuantizedLinear`'s, for the same bits: 0.9 µs more of Python around a call of 133 to 147 µs, the launch itself taking the time of MLX's.

## Load a checkpoint

mlx-lm and mlx-vlm rebuild a checkpoint's quantized layers with `nn.quantize`: a layer the quantization config names is replaced by its `to_quantized(group_size, bits, mode)`, and the checkpoint's tensors are then loaded over the model's, strictly. oMLX has `nn.Linear` and the `SwitchLinear` of mlx-lm and of mlx-vlm build a `FormatLinear` or a `FormatSwitchLinear` for a mode whose format lays the matrix out:

```python
    def layout(self, k, n, bits, group_size):
        if bits != 8:
            return None
        return {"weight": ((n, k), mx.uint8), "scales": ((n,), mx.float32)}
```

`layout` gives the shape and the dtype of each tensor a checkpoint keeps of one `(n, k)` matrix: its `weight` and the format's `tensors`, by name. A switch layer's stack has its axis ahead of each. The layer is built over placeholders of those shapes, with the bias of the layer it replaces, and the load fills them: a checkpoint whose tensors have other names or other shapes is refused there (by MLX in a strict load, by oMLX when the load returns, whatever its `strict`, and a layer the load filled none of at the end of oMLX's own load: see [Load a checkpoint](#load-a-checkpoint)). MLX's load does not check a tensor's dtype, nor its shape when it is not strict: one of another shape or dtype than the layout's is not taken, and leaves the placeholder. A layer that still holds a placeholder when it is first called (that tensor's, or one a load that is not strict lacked) refuses to compute, naming itself, the tensor and the shape and dtype it was offered or `never loaded`: it never computes on a placeholder, and neither its decode nor `formats.dot` reads one. Only a load fills a placeholder: an array assigned while no checkpoint loads on that thread (a recast, `set_dtype` before or after the load, code of one's own) is stored, and leaves each tensor loaded or not as it was; and a load's array of another shape or dtype, not taken, leaves that tensor unloaded even in a layer loaded before, which then refuses to compute until a load gives it one.

`layout` returns None for a matrix the format does not keep (bits it has no codes for, a K its groups do not divide). The load then fails in words: `quant format 'myformat' of the checkpoint at /models/my-model (written by my-package 0.1.0; installed: my-package 0.1.0): it lays out no (16, 64) matrix of 4 bits in groups of 2`. A built-in format of one of MLX's own modes returns None for the matrices that stay MLX's layers; affine does for all.

A checkpoint names each layer of the format in its quantization config:

```json
"quantization": {
    "group_size": 64, "bits": 4, "mode": "affine",
    "model.layers.0.mlp.gate_proj": {"group_size": 1, "bits": 8, "mode": "myformat"}
}
```

- A layer's entry holds `group_size`, `bits` and `mode`, and nothing else: the loader calls the layer's `to_quantized` with it.
- The top-level `mode` stays one of MLX's. A layer the config does not name takes the top-level values, and the loaders find such a layer by its `scales` tensor.
- A checkpoint of a format that is not installed fails to load, naming the plugin its record says wrote it: `quant format 'myformat' of the checkpoint at /models/my-model (written by my-package 0.1.0; installed: none): no format of this mode is installed: install my-package 0.1.0`.
- A layer of a format is built from its bits and group size. MLX's other arguments (`quantize_input`, mlx-lm's `global_scale`) are refused with a format's mode.
- A model whose layers have a `to_quantized` of their own loads a format once `omlx.patches.format_load` wraps it as it wraps those three. Until then oQ writes their matrices in its affine (see [Quantize a checkpoint](#quantize-a-checkpoint)).

A checkpoint's layers of a format are read with the record oQ stored for it (`omlx_quant_formats` in `config.json`, see [Quantize a checkpoint](#quantize-a-checkpoint)). When the registered format lays out the first of them, the load hands the record to the format's `accept`, which returns the format that reads the checkpoint, or refuses it:

```python
    def accept(self, record):
        if record.get("layout", 1) != 1:
            raise ValueError(f"written in layout {record['layout']}; this plugin reads layout 1")
        return self
```

- Each load hands the record to `accept` of its own copy of the registered format (`copy.deepcopy`), and every layer of the checkpoint is built with the object `accept` returns, and keeps it (`layer.fmt`): its `layout` gives the layer's tensors, and its `decode` and `dot` run them. `accept` returns the format itself (the copy) by default; None means the copy too.
- A format whose checkpoints differ by what their record holds (a table of levels, say) configures its copy from it, or returns a new object configured from it: its faces can hold the record's values as constants. The object is a `QuantFormat` of the same `mode`, `tensors` and `contract`; anything else is refused: `quant format 'myformat' of the checkpoint at /models/my-model (written by my-package 0.1.0; installed: my-package 0.1.0): accept returned 'myformat', not a format of its mode, tensors and contract: report it to my-package 0.1.0`. Make it immutable once accepted, with no unevaluated arrays: the layers of several threads read it.
- Nothing `accept` does reaches the registered format, so checkpoints of several records load and run side by side, each layer with its own checkpoint's object. The registered format is copied once per load and checkpoint, so it holds nothing `copy.deepcopy` refuses (a lock, an MLX dtype; define `__deepcopy__` for anything else) and nothing large; a format that cannot be copied is refused: `quant format 'myformat' of the checkpoint at /models/my-model (written by my-package 0.1.0; installed: my-package 0.1.0): the format cannot be copied for the load (cannot pickle '_thread.lock' object): accept runs on a copy of the registered format, which holds no lock, dtype or other object that copy.deepcopy refuses; report it to my-package 0.1.0`. A format keeps what it caches (its faces) outside the object: `functools.cache` on a function (or a `staticmethod`), as the built-in affine format does (on a method it would key on, and keep, the format object).
- `accept` raises a ValueError in the format's own words to refuse a checkpoint. The load raises them as its reason: `quant format 'myformat' of the checkpoint at /models/my-model (written by my-package 0.2.0; installed: my-package 0.1.0): written in layout 2; this plugin reads layout 1`.
- A checkpoint written before formats had records, or one that names the format in its quantization config alone, hands an empty record. A record that is not a JSON object is refused: `quant format 'myformat' of the checkpoint at /models/my-model (written by my-package 0.1.0; installed: my-package 0.1.0): config.json['omlx_quant_formats']['myformat']['record'] is null, where oQ writes an object: the checkpoint's config is damaged`.

The record is that of the checkpoint the load reads. oMLX wraps the libraries' functions that build a model from a checkpoint's directory (mlx-lm's, mlx-vlm's and mlx-embeddings' `load_model`, and the encoder loader of mlx-vlm's embedding models), each the first time it finds it, imported or as it is imported, and in place: a script that imported one before oMLX's patches were applied (`from mlx_lm.utils import load_model`) calls the wrapped function too:

- A format checkpoint enters the checked model loader in mlx-lm or mlx-vlm. Embedding,
  reranking, decision, audio, external engines and custom weight readers refuse it
  before reading its tensors. The encoder and embedding library hooks also check the
  resolved model directory when a caller supplies a Hub ID.
- Stock MLX checkpoints use the library's own quantization layouts. These include
  MiniMax NVFP4 expert global scales and a VLM's nested quantization metadata.
  A plugin record anywhere in the config, or a plugin mode under quantization,
  requires a checked reader. A checked reader accounts for the top-level and text
  model declarations together and refuses a declaration in an unsupported location.
  Fields inside the plugin's own record belong to the plugin.
- Bundled MiniMax and Inkling DSpark readers reject a format before opening draft
  weights. The engine pool releases the target and surfaces the typed refusal.
  A cached retry receives the same refusal details.
- In place means the function object itself: its `__code__` becomes that of a short function that calls oMLX's loader, which runs a copy of the function (its own code, globals and defaults). The object, its name, module, docstring, annotations and `inspect.signature` (through `__wrapped__`, the copy) are those of the library's function; `functools.wraps` over it copies them. It is wrapped once, under a lock: two threads applying the patch at once, or a module found again, leave it wrapped once. A call on another thread meanwhile runs the function whole, its own code or the wrapped one. A library version whose loader is not a plain Python function makes every load fail with `omlx.formats.LoadHookError` (a TypeError) that names it, which no engine or fallback load swallows (a test runs each installed library's loader through the hook); a family's loader that took the module's name first, a closure, is wrapped by that name, and the library's function under it is refused by name when it builds a format's layer: `mlx_lm.utils.load_model was imported before oMLX's patches were applied and runs unwrapped (a family's loader took its name in its module first): call mlx_lm.utils.load_model through its module, or apply the patches first (maybe_apply_pre_load_patches)`.
- A checkpoint that records a quant format, or names one in its quantization config (its text model's too), and is of a family oQ writes in none (one that is not verified, see [Quantize a checkpoint](#quantize-a-checkpoint)) is refused before the library reads a weight, whoever wrote it (a plugin's own writer, a config edited by hand): `quant format 'myformat' of the checkpoint at /models/my-model (written by my-package 0.1.0; installed: my-package 0.1.0): oMLX loads no raon1_mini checkpoint in a quant format: its merged gate and up projection is sliced by hand on a prefill, as MLX's quantized layer or as floats; have the family verified`. The family is the model type its config.json names (and its text model's), wherever the load finds it: a directory's name never decides. A decision model (Clef, OpenJev, d1, d3) is refused as its decision family by the loader that serves it as one, whatever its backbone's type: `quant format 'myformat' of the checkpoint at /models/my-model (written by my-package 0.1.0; installed: my-package 0.1.0): oMLX loads no clef checkpoint in a quant format: it reads rows of its output head by hand, as MLX's quantized layers or as floats; have the family verified`. A record of formats that is no JSON object is refused the same way, as a damaged config.
- A format's layer (`FormatLinear`, `FormatSwitchLinear`) is built only in the load of a
  checkpoint the family gate passed, on its thread or on one that copied its context
  while the load runs, or in a block of `omlx.formats.testing.verified` (see
  [Test a format](#test-a-format)): built anywhere else (a family's own loader, a script
  by hand), it is refused: `quant format 'myformat': a format's layer is built only
  while oMLX loads a checkpoint of a family verified with quant formats, and this one is
  built outside such a load (by a family's own loader, or by hand): load the checkpoint
  with mlx-lm's or mlx-vlm's load_model`.
- MoE expert offload offloads no expert of a model with any format's layer (one its config records or names, or that its load built), and says so once: the model loads with every expert resident, and admission counts them in full.
- The format layers built on a thread while one runs (or on a thread that copied its context, while it runs) belong to its checkpoint, and a load nested in another reads its own; a load of the same checkpoint (a view of links to it) on the same thread inside a library's loader is that loader's load, which checks it when it returns.
- A format's layer built anywhere else is refused: `quant format 'myformat': a layer is
  built outside a checkpoint's load, which gives it the record to be read with: load the
  checkpoint with mlx-lm's or mlx-vlm's load_model`. So is a load that returns a model
  without a format layer built under it (another checkpoint's, built inside this one's
  load).
- A family's loader that replaces one of them afterwards runs around it: DeepSeek V4
  opens a checkpoint load around its own reader. The family gate checks the checkpoint.
  The native reader refuses a format in either the disk config or a load override before
  opening weights.
- When the load returns, whatever its `strict`, a layer it filled in part is refused (a
  tensor of the checkpoint renamed, or missing): `quant format 'myformat' of the
  checkpoint at /models/my-model (written by my-package 0.1.0; installed: my-package
  0.1.0): the load filled model.layers.0.mlp.gate_proj but its scales: the checkpoint
  holds no such tensor, or holds it under another name`. So is a tensor of another shape
  or dtype than the layout's, or no array at all (tensors nested under its name), and an
  entry of the quantization config (its own and its text model's, under either key) in a
  format's mode (one that is not MLX's, or one the checkpoint records; an entry without
  a mode is MLX's affine) where the load built no layer of that mode, one layer for each
  entry, at the entry's whole path (or under `language_model.`, where mlx-vlm builds a
  language model's checkpoint, or without it, where mlx-lm builds a VLM's text model; a
  Hugging Face name under `model.language_model.` at `language_model.model.` or
  `model.`): a family's loader that renames the module, an entry that names the end of a
  path. An entry in a part of the model the load leaves out (an MTP head left off, a
  text-only load's vision or audio tower) binds nothing. So is the quantization config's
  own mode, a format's, when the load built no layer of it, and a record of formats that
  is no object. A layer the load left wholly unfilled is one a partial load does not
  hold (a pipeline rank's, a prefill helper's), and is not refused there: oMLX refuses
  it at the end of its own load, once what the process does not hold is dropped
  (`format_load.check_loaded`: the LLM and VLM engines' loads, the cluster's (a rank's,
  a pipeline stage's, the progressive ones'), a prefill helper's right after it keeps
  its first layers (or, for a family that keeps them before the load, after it), the
  drafters' and SpecPrefill's, the embedding and reranker models' before their compile's
  warm-up, the decision models', oQ's calibration and sensitivity loads; a refusal there
  is never a drafter or draft skipped): `quant format 'myformat' of the checkpoint at
  /models/my-model (written by my-package 0.1.0; installed: my-package 0.1.0): the
  loaded model holds model.layers.0.mlp.gate_proj without its weight, scales: the
  checkpoint holds no such tensor, or holds it under another name`.

When a caller supplies `model_config` to mlx-lm, oMLX checks both the
on-disk family and the family selected by the override before opening
weights. A format record supplied by the caller also has to account for
layers in the loaded model. When the checkpoint already names a format,
the override must preserve its record and quantization settings. It may
copy a layer's settings to the runtime spelling of that same layer.

The quantization config of a format checkpoint uses a flat map. Its defaults
are `bits`, `group_size` and `mode`. Each per-layer value is an object with
fields from that same set. A refusal identifies the field containing an
unknown layout. Each recorded mode must account for a built layer or name
layers that all belong to omitted heads or towers. A head stored as a list
or dictionary of blocks counts as present.

oQ writes `omlx_quant_formats` at the top level of `config.json`, with an
entry for each format it used. Each entry may name the `plugin` (a string,
or null for a format registered by hand), the supported integer `contract`
revision, and a `record` object whose contents belong to the plugin. An
older entry can omit these fields. An empty format map, one nested under
`text_config`, or an unknown entry field raises a refusal that names its
location.

Every refusal of a format's checkpoint at load is one exception, `omlx.formats.FormatRefusedError` (a ValueError), which names the mode, the checkpoint, the plugin its record says wrote it and the one installed (`mode`, `path`, `written`, `installed`, and the `reason`). oMLX lets it through as it is: a model that mlx-vlm refuses is not loaded again with mlx-lm, nor the other way, since either would build the same layers.

## Quantize a checkpoint

oQ writes a checkpoint's matrices in a format that encodes them:

```python
    def encode(self, w, importance, bits, group_size):
        if bits != 8:
            return None
        rows = w.astype(mx.float32)
        scales = mx.maximum(mx.abs(rows).max(axis=-1), 1e-6) / 127.0
        codes = mx.round(rows / mx.expand_dims(scales, -1)) + 128.0
        return {"weight": codes.astype(mx.uint8), "scales": scales}
```

```python
from omlx.oq import quantize_oq_streaming

quantize_oq_streaming(source, output, 4, enhanced=True, quant_format="myformat")
```

`quant_format` is the format's mode, or an object of the class registered for its mode: a plugin's command that fits its format to the model first (a table of levels, say) hands oQ the object it configured. oQ encodes with that object and stores its record; the load reads the record back through the registered format's `accept`. An object of another class, a subclass included, is refused: `quant format 'myformat' is registered as MyFormat, not OtherFormat: pass an object of the registered class`.

A format fitted to the model reads oQe's importance before oQ asks for its record. `calibrate_oq_streaming` runs oQe's calibration alone, into the imatrix cache, by the code of the run that will reuse it, and returns the cache's entries and metadata; that run then reads the cache instead of calibrating again:

```python
from omlx.oq import calibrate_oq_streaming, quantize_oq_streaming

calibrate_oq_streaming(source, cache, 4)
fitted = fit(source, cache)  # the plugin's own: a configured MyFormat
quantize_oq_streaming(
    source, output, 4, enhanced=True, quant_format=fitted, imatrix_cache_path=cache
)
```

Its keyword arguments are that run's own (`group_size`, `text_only`, `dtype`, `preserve_mtp`, the `imatrix_*` settings, `stream_calibration`, ...): a run with other settings does not match the cache and calibrates again. It writes nothing but the cache (`<cache>` and `<cache>.json`); a calibration proxy that a model above the calibration budget needs is built beside the cache and removed. DeepSeek V4.1 calibrates inside its own writer and is refused.

`encode` returns the tensors that keep `w`, by name, in the shapes and dtypes that `layout(k, n, bits, group_size)` gives, or None for a matrix it leaves to oQ.

- `w` is a matrix `(n, k)`, or some of a stack's matrices, `(e, n, k)`, in the floats of the source checkpoint. The tensors of a stack's matrices have its axis first.
- `importance` weighs the squared error of each of the K inputs, as oQe measured them over its calibration samples. It is `(k,)`, or `(e, k)` when each matrix of a stack has its own, in fp32.
- `bits` and `group_size` are what oQ gives the layer at the level it writes. The layer's quantization entry carries them, so `layout`, `decode` and `dot` get the same values when the checkpoint loads. A format with groups of its own ignores `group_size`.

`quant_format` is the mode of a registered format that is not one of MLX's own: oQ writes those modes itself, so one of them is refused before a weight is read: `quant format 'mxfp4' is one of MLX's own modes: oQ writes it without a quant format`.

What oQ offers a format:

- The matrices that oQe has an importance for: the linears and the switch layers its calibration ran through. A format is therefore written with oQe only (`enhanced=True`), and a tensor oQe has no importance for (a table of embeddings, for one) is written as without the format.
- A matrix that loads as a format's layer: one held, at its path and shape, by a module
  of exactly one of the classes the [load hook](#load-a-checkpoint) wraps (`nn.Linear`,
  the `SwitchLinear` of mlx-lm and of mlx-vlm; a subclass of one is not one of them). At
  every run with a format oQ builds the model the checkpoint loads as from the config
  the run writes (its own changes to the source's included, such as MiniMax's
  shared-expert layout), with the library that serves the written config and the
  source's pre-load patches, reading no weight and declaring every tensor the source
  holds (so that mlx-vlm keeps a vision tower or an MTP head); so one imatrix cache
  serves runs with and without a format. A matrix of another class (a family's own
  switch layer, MLA's `MultiLinear`) is written in oQ's affine, and a warning counts
  them: `oQ4: 6 matrices stay oQ's affine: a quant format's layer is not built for their
  class (omlx.patches.deepseek_v4.switch_layers.SwitchLinear)`. So is a matrix that no
  module holds at its path and shape (a family whose loader renames, joins or splits the
  checkpoint's tensors), and the warning names them: `paths that no module of the model
  it loads as holds at their shape: model.layers.0.mlp.experts.0.gate_proj`, the first
  twenty and then how many more.
- A matrix, or a stack of matrices, that `layout` returns a layout for at the layer's bits and group size. `layout` is asked first, and `encode` is not called for a matrix it returns None for. That includes a matrix whose K the group size does not divide, which oQ's affine keeps in floats.
- A matrix that fits one launch: up to 2^30 weights, fewer on a Mac whose largest Metal buffer is small, and a stack whose tensors, joined, each fit a Metal buffer. A stack that does not fit one launch is offered in runs of whole matrices, in several calls, and oQ joins their tensors along the stack's axis. A format therefore encodes each matrix by itself: its tensors do not depend on the matrices beside it in a call. A format that encodes a stack's first runs and returns None for a later one is refused.
- Nothing that oQ does not quantize itself: a tensor the source already keeps at the level's bits or fewer (experts their lab ships in 4 bits, at a level of 4 bits or more) is written as the source has it, and so is a table kept with `preserve_ngram_table`.

What oQ writes:

- The tensors, as `<layer>.<name>`, and the layer's entry in the quantization config: its `bits`, its `group_size` and the format's `mode`. That is what [loads](#load-a-checkpoint).
- For a matrix the format returns None for, what it writes without the format: its own affine tensors at the same bits, or its floats for a matrix whose K oQ's groups do not divide. Such a matrix's importance is looked up for the format alone: a strict run (`imatrix_strict`) does not stop for its missing entry, the imatrix report counts it only when the format takes it, and the warning does not name it.
- In `config.json`, each format the checkpoint has layers of, with the plugin that wrote them (`omlx.formats.origin(mode)`: the distribution's name and version, or null for a format registered by hand), the revision the format states and the format's record:

  ```json
  "omlx_quant_formats": {"myformat": {"plugin": "my-package 0.1.0", "contract": 1, "record": {"layout": 1}}}
  ```

  `record()` gives what a checkpoint keeps of the format beside its layers: the revision of its layout, what the format needs to read its own checkpoints. It is a dict that JSON keeps as it is (string keys; strings of UTF-8, finite numbers, lists, dicts; no tuple, array, NaN or infinity), asked once for the whole checkpoint, and empty by default. oQ stores it as it checked it: a record the format changes afterwards (while it encodes) is not what the checkpoint keeps. A source written in a format carries its own record in its config; oQ leaves it out of an output with no layer in a format. oQ refuses another before it reads a weight: `quant format 'myformat' gives the record {'levels': (1, 2)}, not a dict that JSON keeps as it is: report it to the plugin`. The load hands it back to [`accept`](#load-a-checkpoint).

- A line in the log, `oQ4: 23 layers in quant format myformat`, or a warning when the format took none: `oQ4: quant format myformat took no layer: the output is as without it`.

oQ checks each answer against the layout, which is what the checkpoint loads by. Tensors of other names, shapes or dtypes stop the run:

```
quant format 'myformat' encoded model.layers.0.mlp.gate_proj.weight as {'weight': ((16, 64),
mlx.core.uint8)}, and lays it out as {'weight': ((16, 64), mlx.core.uint8), 'scales': ((16,),
mlx.core.float32)}: the checkpoint would not load
```

oQ gives each layer its bits within a budget of bits per weight, which it counts in what affine costs at `(bits, group_size)`. A format that costs about as much keeps the level's size. One that costs more writes a larger checkpoint than the level states.

oQ refuses a format before it reads a weight:

- without `enhanced=True`: `quant format 'myformat' encodes with oQe's importance: pass enhanced=True`,
- for a mode that is not registered: `no quant format is registered for mode 'myformat': install its plugin`,
- for a DeepSeek V4.1 or a Muse Spark checkpoint: `oQ writes a deepseek_v41 checkpoint through a writer of its own, which takes no quant format; write it without a quant format`,
- for a family oQ does not know to read a format's layers right. The gate fails closed:
  oQ writes a format only for a family in `_QUANT_FORMAT_VERIFIED` (`omlx/oq.py`), under
  any spelling of its model type (lower case, `-` for `_`), the checkpoint's and its
  text model's: families quant formats have been verified with end to end. A family
  enters with the run that verifies it, and none has yet; families mlx-lm's own code
  serves come back after the proof of concept. The family is read from the config alone,
  never from a directory's name; a decision model identified by its declared family,
  architecture or sidecar is refused before the format walk, as are MiMo v2 and MiMo v2
  Flash. Stock writes use their usual writer. Any other is refused with
  `omlx.formats.FormatRefusedError`: `quant format 'myformat': oQ writes no step3p7
  checkpoint in it: it is no family verified with quant formats end to end; write it
  without a quant format, or have the family verified`. For a family whose own code the
  audits found reading a layer's quantized tensors by hand, which would take a format's
  layer for MLX's or for floats, the refusal names that reader: GLM-5.3, `quant format
  'myformat': oQ writes no glm5_next checkpoint in it: its linear attention fuses its
  input projections by hand, as MLX's affine layers or as floats; write it without a
  quant format, or have the family verified`; Inkling 2, `quant format 'myformat': oQ
  writes no inkling2 checkpoint in it: its DSpark draft folds the output head into its
  own projection by hand, as MLX's quantized layer or as floats; write it without a
  quant format, or have the family verified`; Raon-1 mini, `quant format 'myformat': oQ
  writes no raon1_mini checkpoint in it: its merged gate and up projection is sliced by
  hand on a prefill, as MLX's quantized layer or as floats; write it without a quant
  format, or have the family verified`; DeepSeek V4, `quant format 'myformat': oQ writes
  no deepseek_v4 checkpoint in it: its DSpark draft reads the output head's weight by
  hand, as floats; write it without a quant format, or have the family verified`; and
  GLM-5.2, `quant format 'myformat': oQ writes no glm_moe_dsa checkpoint in it: its
  indexer fuses its projections' 8-bit affine layers by hand, and refuses a load where
  one of them is a format's; write it without a quant format, or have the family
  verified`. `_QUANT_FORMAT_MISREAD` in `omlx/oq.py` names each reader (Clef's too,
  which its loader names). A family leaves the table when its readers decline a format's
  layer and call it.

## Faces

A face is a frozen dataclass: a `Dot` for the mat-vec kernels, a `Tile` for the kernels that decode tiles of weights (see [The tile face](#the-tile-face)). A kernel builder is cached on the faces it is instantiated with, so a format returns equal faces for layers of the same layout (cache them) and a kernel is compiled once per layout.

`QuantFormat.dot(layer, k, n)` returns the mat-vec face of `layer`'s `(..., n, k)` matrix, a `Dot`, or `None`. With `None` the kernel is not used for that layer and the model keeps its composed path. Return `None` for:

- a layout the face does not cover,
- a layer with a tensor of fewer than 8 elements. MLX binds such an input as constants, not as the device pointer a face holds.

Kernels ask through `omlx.formats.dot(layer, k, n)`, inside a model's forward, and keep the answer with what they build for the layer (until one of its tensors is replaced). A forward runs on one of oMLX's threads (each loaded model's engine has its own, and the loads, the embeddings and the rerankers share another): `dot()` is called from any of them, and what a format keeps between calls (its faces) is shared by all (see [Kernels over faces](#kernels-over-faces)). A `dot()` that raises an exception there, or that returns anything but `None` or a `Dot` of arrays the layer has (`weight` and each of the face's `tensors`, among its format's `tensors`, of 8 elements or more each and none of float64, which the GPU does not take), counts as `None`: the layer keeps its composed path. Each such defect of a format is logged once (`quant format MODE gave no usable dot face ...`, with the traceback). A format's own tests call `FORMAT.dot` directly and take the operands from the layer by the face's `tensors` (see [Test a format](#test-a-format)), so that these defects fail them. Tests that run a model do it inside `omlx.formats.testing.no_dot_defects()`, which fails on leaving the block when a `dot()` or a `tile()` was defective in it (an exception of the block's body is raised in its place):

```python
import pytest
from omlx.formats import testing


@pytest.fixture(autouse=True)
def dot_defects():
    with testing.no_dot_defects() as defects:
        yield defects
```

The block's value is the set of the defects met so far, each as `(mode, repr(error))`; a test that expects one removes it. A kernel keeps the answer with what it builds for the layer, so only the first forward that asks for a layer's face meets its defect: a model that a module or session fixture runs in its setup (or a function fixture set up ahead of this one) meets it outside this block.

`QuantFormat.tile(layer, k, n)` returns the tile face of the same matrix, a `Tile`, or `None`, and kernels ask through `omlx.formats.tile(layer, k, n)` with the same answers to the same defects (`quant format MODE gave no usable tile face ...`). A defect is a format's mode and the error: the same error from its `dot()` and its `tile()` is one defect, logged once.

| `Dot` and `Tile` field | Meaning |
|---|---|
| `key` | An identifier. Names the face in kernel names and compiler messages. |
| `source` | Metal at global scope that declares the face's struct, in its own namespace (`omlx_<key>`, or one that the faces sharing the source share: the affine tile face's bf16 and fp16 keys share `omlx_affine_tile_b{bits}g{group_size}`). A kernel includes each distinct source once. |
| `type` | The struct, as a type expression for a kernel body, where `T` names the activations' float type. For example `omlx_myformat_b8::Dot`. A `Tile`'s struct is in a namespace of its own, apart from its format's `Dot`: a kernel includes the source of the faces it reads, and no other. |
| `tensors` | The layer's parameters beside `weight` and `bias` that the face reads, as a tuple, among its format's `tensors`. They become kernel inputs in this order. Identifiers, each named once, as a format's `tensors` are. |
| `vpt` | `Dot` only. `Lanes<K>::VPT`, the inputs a lane reads a block: a shared-rows kernel sizes its launch by it and asserts that the struct agrees. 0 (the default) for a face that no such kernel reads; a kernel refuses to share one. |
| `shared` | `Dot` only. Metal that a kernel which applies one block's weights to several inputs includes after `source`, and no other kernel does: where a face defines `decode` and `apply` out of its struct (its struct declares them), so that the kernels that never call them do not carry them. MLX hashes a kernel's whole source at every launch, about 0.05 µs a KB. Empty (the default) for a face whose struct defines them. |
| `dtypes` | The activation dtypes the face is exact for, as a tuple of distinct `mx.Dtype` among `float16`, `bfloat16` and `float32`; empty (the default) for those three (`formats.FLOATS`). Their order is part of the face: a format lists them in one order. Ask with `face.exact_for(dtype)`. |
| `tables` | The lookup tables the face reads, as a tuple of `Table`, given by keyword; empty (the default) for none. See [Tables](#tables). |

## The dot face

The struct named by `type` holds the matrix. Its aggregate members are the weight as bytes, one device pointer per name in `tensors`, then one `omlx_table<T, N>` per entry in `tables`. Both tuples set the order of their members. Each tensor pointer has its tensor's element type (`float` for float32, `T` for a tensor in the activations' dtype). A layer's tensors keep their dtypes when the activations are recast (see [Layers](#layers)), so a pointer typed `T` reads its tensor right only for the activation dtypes the face lists in `dtypes`: a face whose pointers are `T` lists its tensors' dtype there, or names their type, `bfloat16_t` for one. Ahead of them it states the revision of this page the face is written for, as the constant `CONTRACT` (see [Versions](#versions)).

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
| `static constant constexpr int VPT` | Inputs a lane reads per block: one or more. K is a multiple of it. |
| `static constant constexpr int BLOCK` | Inputs the simdgroup covers per block: `32 * VPT`. |
| `Lanes(Dot m, size_t row, uint lane)` | Lane `lane` at the first block of row `row`. |
| `struct Row` and `Row row(int r) const` | What `madd` needs of the current block of row `row + r`: pointers, the group's scale. |
| `void step()` | Move to the next block. |
| `template <typename XP> static A load(XP x, thread float* xt)` | Read the lane's `VPT` inputs into `xt` as floats. `x` points at them in device memory, or in threadgroup memory where a kernel stages its inputs: `XP` is `const device X*` or `const threadgroup X*`, for the activations' type `X`. Its result is passed to every `madd` of the block as `aux`. `A` is the face's choice: a `float` (affine returns the sum of the inputs, for the bias term) or a struct of its own. |
| `static float madd(float acc, Row r, thread const float* xt, A aux)` | `acc` plus the lane's share of the row's product for the block. |
| `float finish(int r, float sum) const` | Row `row + r`'s result from the sum of `acc` over the lanes (for example times a row scale). |
| `struct Decoded` and `static Decoded decode(Row r)` | One row's weights of the current block as `apply` takes them, read once for several inputs. A face with nothing to share writes `using Decoded = Row;` and returns `r`. A face whose `decode` and `apply` are long declares them in its struct and defines them in `shared` (affine does). |
| `static float apply(float acc, thread const Decoded& d, thread const float* xt, A aux)` | What `madd` gives, from `decode`'s weights: `apply(acc, decode(r), xt, aux)` equals `madd(acc, r, xt, aux)` bit for bit. A kernel that applies one block's weights to several inputs decodes them once and calls `apply` for each. |

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

- `VPT` is one or more, `BLOCK == 32 * VPT`, and K is a multiple of `VPT`. A kernel asserts the three when it is compiled, so a format's `dot()` returns `None` for a K that is not.
- A row's result is a function of the row and the input only: not of the kernel, of the rows computed with it or of its place in a launch. MLX compiles kernels with fast math off, so the float operations run as written; keep `madd` and `finish` pure and every kernel computes the same bits for a row. oMLX relies on it wherever a fused kernel and another path must agree on a token.
- A face never writes the matrix: its members store to nothing but the cursor's own state and `xt`.
- A face is safe over tensors and activations of any value, zeros included, on which a kernel proves its Metal: every address that `row`, `load`, `madd` or `finish` reads stays inside the arrays the face was given (an index read from a tensor, such as a code that picks an entry of a level table, is masked to the size of what it indexes; the constructor and `step()` form addresses and read none, next rule), and no loop bound or exit depends on a value that it reads.
- A cursor may be placed at any lane index: for every `j` of 0 or more, `Lanes(m, row, j)` is the lane over inputs `j * VPT` to `j * VPT + VPT - 1` of the row, with the parameters of the group those inputs lie in. So `j = 32 * b + l` is lane `l` placed at block `b`, and a cursor may start inside a group. A kernel that splits K across threadgroups, or walks a row in one thread, builds its cursors there instead of stepping one. Affine places its scale pointer at `lane / SCALE_STEP_PER_THREAD`, an integer division: it qualifies as it is. `omlx.formats.testing.lane_parts` checks the rule ([Test a format](#test-a-format)).
- The constructor and `step()` only move pointers and read nothing. A kernel may build a cursor on a row past the matrix (for an empty pack of rows) and step it into a block it then skips. `row`, `load`, `madd` and `finish` run only for rows and blocks the kernel computes; `row` is `const`.
- No member uses a threadgroup operation (a barrier) or threadgroup memory of its own: the simdgroups of a kernel need not all run the skeleton, or run it over the same K. A face reads its tables through the members it is given (see [Tables](#tables)).
- `row`, `load` and `madd` use a simdgroup operation (`simd_sum`, `simd_shuffle`) only in a face that takes whole blocks alone (its format's `dot()` never returns it for a K that ends inside a block): in a last, partial block only some of the lanes run them.
- `finish` is a scale: `finish(r, sum)` is `sum` times a value that depends on the row alone. A kernel applies it once to a row's whole sum, or once to each part's sum where it splits the row's K across threadgroups or launches and adds the parts afterwards (the hyper-connection decode adds four parts of a down row in another launch, which binds no down operand). Affine's `finish` returns the sum; a row scale (NVFP4's global scale, oQeL's row scale) qualifies. `omlx.formats.testing.finish_rows` checks it ([Test a format](#test-a-format)).
- `finish` runs where the walk left the cursor, which is not one place: past the last whole block, or on a last, partial one. It reads only what `step()` does not move, and nothing that depends on the lane: after it every lane holds the row's result.
- `apply(acc, decode(r), xt, aux)` and `madd(acc, r, xt, aux)` are equal bit for bit: kernels take either. `decode` reads the block's weights and no input. `omlx.formats.testing.shared_rows` runs a face's rows through `decode` and `apply` for several input vectors ([Test a format](#test-a-format)).
- `r` is a constant or a loop counter with constant bounds: the row skeleton calls `row` and `finish` with an element of its pack of rows, the shared-rows skeleton in a loop or a fold over its `RPS` rows. `row` and `finish` do arithmetic on it (`r * K`), not a loop over the rows.
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

    template <typename XP, int... I>
    static void copy(omlx_ints<I...>, XP x, thread float* xt) {
      ((xt[I] = float(x[I])), ...);
    }
    template <int... I>
    static float dot(omlx_ints<I...>, const device uint8_t* w, thread const float* xt) {
      return (... + ((float(w[I]) - 128.0f) * xt[I]));
    }

    template <typename XP>
    static float load(XP x, thread float* xt) {
      copy(omlx_iota<VPT>(), x, xt);
      return 0;
    }
    static float madd(float acc, Row r, thread const float* xt, float) {
      return acc + dot(omlx_iota<VPT>(), r.w, xt);
    }

    float finish(int r, float sum) const { return sum * s[r]; }

    using Decoded = Row;
    static Decoded decode(Row r) { return r; }
    template <typename A>
    static float apply(float acc, thread const Decoded& d, thread const float* xt, A aux) {
      return madd(acc, d, xt, aux);
    }
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

## The tile face

A kernel that multiplies many rows of activations by a matrix (a prompt's prefill, a switch layer's experts over a prompt) decodes a tile of the weights, some rows by some inputs, into threadgroup memory once, and every row of activations it computes reads them there. The tile face is how it decodes them: the matrix of the dot face, read in runs of 8 codes and given weight by weight. A format gives it with `QuantFormat.tile(layer, k, n)` (see [Faces](#faces)).

The tile struct declares the static constant `CONTRACT`. Its aggregate members follow the dot face's order. The weight as bytes comes first, followed by the device pointers in `tensors` order and the `omlx_table<T, N>` members in `tables` order. Its member template `Runs<K>` is the cursor of one row of K inputs.

```cpp
struct Tile {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device float* group_scales;
  const device float* row_scales;

  template <int K>
  struct Runs { ... };
};
```

A kernel builds a cursor for each row a thread decodes, and reads the row through it in runs of 8 codes, and in groups of codes that share their parameters. K is known when the kernel is compiled, as a dot face's `Lanes<K>` has it: a row's strides are `constexpr`, and a face that takes only some K states which in a `static_assert` in its constructor (the affine face: whole groups). An assert that fails in the body of `Runs<K>` itself leaves it without members for the kernel's checks, each of which then fails too.

| Member of `Runs<K>` | Role |
|---|---|
| `static constant constexpr int RUN` | 8, the codes of a run. A run of 8 codes of b bits is b whole bytes. |
| `static constant constexpr int GROUP` | The codes that share one `params()`: 8 times a power of two (8, 16, 32, ..., 2^30). A row has `(K - 1) / GROUP + 1` groups, the last of which may be partial. |
| `struct P` | A group's parameters, as a kernel holds them in registers. |
| `Codes` | A run's codes, as a kernel holds them in registers: a `uint`, a `uint2`, ... |
| `Runs(Tile m, size_t row)` | The cursor of row `row` of `m`. In a stack of matrices of N rows each, row `n` of matrix `e` is row `e * N + n`. |
| `P params(int g) const` | The parameters of group `g` of the row, `g` in `[0, (K - 1) / GROUP + 1)`. |
| `Codes fetch(int run) const` | The codes `8 * run` to `8 * run + 7` of the row, `run` in `[0, K / 8)`. |
| `static float weight(thread const P& p, Codes c, int i)` | Weight `i` (0 to 7) of a run of codes `c` in a group of parameters `p`, as a float32. |

`P` and `Codes` are types that are default-constructible and copy-assignable: a kernel holds arrays of them and assigns their elements. K is a whole number of runs. A kernel runs a face in this order, in each thread that decodes values of a row, for each block of K that it decodes, with `T` the activations' type:

```
runs = Runs<K>(m, row)                            once for the row
c[r] = fetch(first_run + r)                       for each run r of the thread's values
p[g] = params(first_group + g)                    for each group g they lie in
w[8 r + i] = T(weight(p[group of run r], c[r], i))  for each run r, i from 0 to 7
```

The values a thread decodes are whole groups or lie in one group: hence a `GROUP` of 8 times a power of two. A kernel calls `params` and `fetch` on one cursor as often as it needs, for any group and run, in any order.

Three things to know when writing one:

- A run of eight b-bit codes occupies b bytes. For 2-, 4- and 8-bit codes,
  read a run as a `ushort`, `uint` or `packed_uint2` when the base is aligned to
  2, 4 or 4 bytes respectively. Byte loads also work on unaligned storage.
  A 3-bit run takes three byte loads, or one byte and an aligned `ushort`. A
  four-byte load at the last run of the last matrix reads past the array (rule 1).
- A face's source is a function of its `key`: whatever the source is built from goes in the key (the bit width, the group size). A table of values that a checkpoint carries is no part of the source: it is the face's `tables` (see [Tables](#tables)), and faces of two checkpoints with different tables are two kernels whatever their keys.
- In a face's source, MLX names a bf16 tensor's element type `bfloat16_t` and an fp16 one's `float16_t`, as in `T` and in the `type` of a face whose struct is a template of it (`omlx_affine_tile_b4g64::Tile<bfloat16_t>`).

### The weight

`weight` gives the weight that the format's `decode` gives, computed in float32, bit for bit; the kernel rounds it once to `T` where it stores it. The weights a tile kernel multiplies are therefore the decoded weights rounded to `T`, whatever the kernel, and a format's tests check the face weight by weight (see [Test a format](#test-a-format)).

Where a weight has several scales, a group's and a row's, `params` multiplies them, once for the group, `P` holds the product, and `weight` multiplies the code's level by it: `level * (group_scale * row_scale)`. The format's `decode` computes the same products in the same order. Multiplication of floats is not associative: `(level * group_scale) * row_scale` can differ from it in the last bit. MLX compiles kernels with fast math off, so the operations run as written.

### Rules

1. **Every address a member forms lies inside the arrays of its aggregate**, for tensors of any value: an index read from a tensor is masked to its table's size.
2. **A face is safe on zeros**: tensors of zeros give finite weights. A kernel proves its Metal on zeros.
3. **No loop bound or exit comes from a value read**: a face has no loop over data.
4. **The constructor reads nothing.** A kernel builds a cursor in each thread of its tile, also for rows past the matrix, and calls `params` and `fetch` only for the rows, groups and runs it computes. A constant of the row (a row scale) is read again at each `params()`: that is the price of this rule. The constructor computes pointers from `row` and K, and nothing else.
5. **No threadgroup memory of its own, no barrier, no simdgroup operation**: not every thread of a kernel decodes. A face reads its tables through the members it is given: the kernel prepares them before any of its own code (see [Tables](#tables)).
6. **A face never writes.**
7. **`weight` is pure and exact**: a function of `p`, `c` and `i` alone, which reads no memory but `constant` data and the face's tables (through `p`), and gives the format's decode computed in float32. `i` is not always known when the kernel is compiled.
8. **The members keep no state but the cursor's**, and `params` and `fetch` are `const`: kernels run on several threads, and a format returns equal `Tile`s for the layers of one layout.
9. **The source is C++17 Metal without derived classes.** See [Compile-time rules](#compile-time-rules).
10. **Use templates for branches that depend on an instance's types.** Outside
    a template, the compiler checks both branches of `if constexpr`, even when
    the condition uses namespace constants for the bit width and group size.
    Use a type overload, a dependent template parameter, or generate the source
    for each width separately (as the affine face does in `_tile_source`).

Rules 1, 2 and 7 are what `tile_rows` checks, on codes and scales at their largest values, on random ones and on zeros (see [Test a format](#test-a-format)). It also calls `params` and `fetch` twice on one cursor, another group and run first (where the row has more than one: K above `GROUP` and above 8), and keeps the second answers, so a member that keeps state between calls (rule 8) shows in its weights. Nothing checks rules 3, 4, 5 and 6: a face is read against them. (A constructor that reads cannot be caught by building a cursor past the matrix: the compiler moves the read to where the cursor is used, which for such a row is nowhere.)

The compiler checks the members a kernel uses when it is compiled, the face's revision first. A face that lacks one, or has it with another signature, fails with the rule, cited in `<omlx skeletons, docs/quant-formats.md>`. The compiler reports the first rule a face breaks, in the order below, and no error after it: fix that one, and the next is reported. A face of another revision is told that and nothing else, as a dot face is (see [Versions](#versions)).

- `tile face: written for a later revision of the contract than this oMLX runs`
- `tile face: written for a revision of the contract this oMLX no longer runs`
- `tile face: the face needs a static constant constexpr int CONTRACT: the revision of the contract it is written for`
- `tile face: the face needs a member template Runs<int K>: the cursor of a row of K inputs`
- `tile face: Runs<K>::RUN must be 8: every tile kernel reads runs of 8 codes`
- `tile face: Runs<K>::GROUP must be 8 times a power of two (8, 16, 32, ..., 2^30): the values a kernel's thread decodes are then whole groups or in one`
- `tile face: Runs<K>::P and Runs<K>::Codes must be types that are default-constructible and copy-assignable: a kernel holds arrays of them`
- `tile face: Runs<K> must be constructible from the face and a size_t row`
- `tile face: Runs<K> needs a const member params(int g) that returns a P`
- `tile face: Runs<K> needs a const member fetch(int run) that returns a Codes`
- `tile face: Runs<K> needs a static member weight(P, Codes, int i) that returns a float`

### An example of a tile face

A format of 8-bit codes around 128, with a float32 scale for each group of 32 codes of a row and one for the row: `w = (code - 128) * (group_scale * row_scale)`.

```cpp
namespace omlx_myformat_b8g32 {
struct Tile {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device float* group_scales;
  const device float* row_scales;

  template <int K>
  struct Runs {
    static constant constexpr int RUN = 8;
    static constant constexpr int GROUP = 32;
    static constant constexpr int GROUPS = (K - 1) / GROUP + 1;
    struct P {
      float s;  // the group's scale times the row's
    };
    using Codes = uint2;  // 8 codes of 8 bits

    const device packed_uint2* c;
    const device float* g;
    const device float* r;

    Runs(Tile m, size_t row)
        : c((const device packed_uint2*)(m.codes + row * K)),
          g(m.group_scales + row * GROUPS),
          r(m.row_scales + row) {}

    P params(int group) const {
      return {g[group] * r[0]};
    }
    Codes fetch(int run) const {
      return Codes(c[run]);
    }
    static float weight(thread const P& p, Codes q, int i) {
      const uint code = ((i < 4 ? q.x : q.y) >> (8 * (i & 3))) & 0xFFu;
      return (float(code) - 128.0f) * p.s;
    }
  };
};
}  // namespace omlx_myformat_b8g32
```

```python
Tile(
    key="myformat_b8g32",
    source=METAL,
    type="omlx_myformat_b8g32::Tile",
    tensors=("group_scales", "row_scales"),
)
```

Its `decode`, the same products in the same order:

```python
    def decode(self, tensors, bits, group_size):
        levels = tensors["weight"].astype(mx.float32) - 128.0
        scales = tensors["group_scales"] * tensors["row_scales"][..., None]
        return levels * mx.repeat(scales, 32, axis=-1)[..., : levels.shape[-1]]
```

## Tables

A table that a face reads at indices that differ from lane to lane (a level for each code, a codebook) is data, not Metal. The face gives it in Python, and every kernel built on the face holds it in the form this GPU reads fastest: a copy in threadgroup memory, made when the kernel starts, or for a small table its values themselves. A table read per lane from `constant` or `device` memory instead makes the kernel's speed depend on where the GPU loads the kernel: 2 to 7% on an M3 Ultra from one process to the next, for the same source. The staged copy is the fastest form measured for tables of more than 4 values and has no such spread.

```python
from omlx.formats import Dot, Table

Dot(
    key="mytable_b2",
    source=metal_source,
    type="omlx_mytable_b2::Dot",
    tensors=("scales",),
    tables=(Table("levels", mx.float32, (-1.5, -0.5, 0.5, 1.5)),),
)
```

`Table(name, dtype, values)`: `name` is an identifier, `dtype` one of float32, float16, bfloat16, int8, uint8, int16, uint16, int32 and uint32, and `values` a tuple of 1 to 1024 numbers. A float table rounds each value to its type, to nearest even, when it is made; a value that is NaN or out of the type's range is refused. An integer table takes whole numbers in its range. A face's tables hold at most 4096 bytes, and so do a kernel's.

The face's struct declares one member for each table, after the pointers of its tensors, in the order of `tables`: an `omlx_table<T, N>`, `T` the table's type (`float`, `half`, `bfloat16_t`, `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`) and `N` its number of values. It reads like an array, `levels[code]`, and is copied like a pointer. A dot face's `Lanes` copies it, and `Row` carries it to `madd`, which reads it; a tile face's `Runs` copies it, and `P` carries it to `weight`:

```cpp
struct Dot {
  static constant constexpr int CONTRACT = 1;
  const device uint8_t* codes;
  const device float* scales;
  omlx_table<float, 4> levels;

  template <int K>
  struct Lanes {
    ...
    omlx_table<float, 4> lv;
    Lanes(Dot m, size_t row, uint lane) : ..., lv(m.levels) {}
    struct Row {
      const device uint8_t* w;
      omlx_table<float, 4> lv;
    };
    Row row(int r) const { return {w + r * K, lv}; }
    static float madd(float acc, Row r, thread const float* xt, float) {
      ...  // r.lv[code]
    }
  };
};
```

oMLX defines `omlx_table` when it builds a kernel, for the GPU it runs on, so one source of the face serves every GPU. On a GPU without tensor units (the M3 family), a table of up to 4 values holds the values themselves and reads them by selects on the index's bits: measured up to 15% faster there than a staged copy, on the 2-bit levels of a mixture of experts' layers. On a GPU with tensor units (the M5 family) every table is staged: there a staged copy measured 0.6 to 3% faster than selects. A family not measured follows its GPU's tensor units (the GPU's own, whatever MLX's kernels use), and a GPU oMLX cannot name stages. Larger tables are always staged. The face cannot tell which form it has, and needs not. A face must index its table within `0 <= i < N`. The select form reads slot `i % 4`. Unused slots from `N` through 3 hold NaN for floats and zero for integers. For example, index 4 selects slot 0.

A face's tables are part of the face: two faces are equal when their tables' names, dtypes and bits are, and a kernel is compiled once for each. A format whose tables come from a checkpoint (a level table fitted to it) makes the checkpoint's faces with its tables, and puts what tells them apart in the faces' keys, so that a log line names the face that ran.

The copy runs once in every threadgroup, before the kernel's own code, and ends with one barrier for all of a kernel's staged tables. It costs about 0.3 to 0.5 µs of a one-token launch, which a face without tables does not pay: its kernels are the same Metal as before tables existed. `omlx.formats.testing.staged_tables` reads a face's tables back as a kernel stages them (see [Test a format](#test-a-format)).

## Kernels over faces

A kernel names each matrix it reads in a `Matrices`, with the face it reads the matrix through, a `Dot` or a `Tile`. That gives the matrix's kernel inputs, the Metal to include (the faces' sources, then the skeletons the kernel calls, and no other: MLX hashes a kernel's whole source at every launch, so the skeletons are also carried without indentation or comment lines) and the statements that declare the matrices in the kernel body. A matrix's inputs are `<name>_weight` and `<name>_<tensor>`. `add` refuses a name that is not an identifier, and a matrix whose name or inputs repeat those of a matrix added before. The kernel's other names (`x`, its outputs, its body's locals) are the kernel's to keep apart from them.

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

A kernel that applies each matrix block to several input vectors (verify windows, batched decode) adds the matrix with `matrices.add(name, face, shared=True)`: its header then holds the face's `shared` source and the shared-rows skeleton, `qmv_shared<T, K, ROWS, RPS, VPT, PARTIAL>(m, row, valid, x, lane, result)`. It is one simdgroup's work over `ROWS` vectors (`x`, `x + K`, ...) and `RPS` rows of `m` from `row`: per block, each row's weights are decoded once (`decode`) and applied to every vector (`apply`), and `result[i * RPS + r]` is vector `i`'s product with row `row + r`. With `PARTIAL`, the pack holds `valid` rows (the last of an N that is not whole packs). `VPT` is `face.vpt`, which the skeleton asserts against the struct. `kernels.Matrices(unrolled=True)` folds each block over its vectors and its rows and unrolls the skeleton's other loops in full, which keeps its arrays in registers; `kernels.unrolled_tile_ok(vpt, rps, rows)` is the envelope inside which the compiler was measured to give unrolled tiles their bits, and `kernels.shared_geometry(vpt, n, rows, pad)` picks a launch's vectors per threadgroup and rows per simdgroup. Each vector's row has the bits of its one-vector call: the rule `apply(decode(r)) == madd(r)` and the run order make it so.

What the kernel does around the call is its own: which rows a simdgroup takes, the activation, the combine, the stores. For the plain store, a kernel adds `kernels.STORE_ROWS` to its header: `store_rows(omlx_iota<N>(), result, y + row)` rounds a simdgroup's `N` results to the output's type. The kernel calls it from one lane (the example above stores its two rows by hand).

`kernels.face_kernel(name, inputs, matrices, outputs, header, source)` makes that kernel and keeps it: `inputs` are the kernel's inputs in order, a name of `matrices` standing for that matrix's (`Matrices.add`); its header is the faces' Metal, then `header`; its source opens with the matrices' declarations, then `source`; it is named `<name>`, then each face's key, `_`-joined in the order of `inputs`. `inputs` and `outputs` are sequences of names (a string is refused). Its source is written and MLX's kernel made once for each name, inputs, faces, outputs, header and source, so a kernel may ask for it at every call. So may it ask `formats.dot(layer, k, n)` and `formats.tile(layer, k, n)`: a layer (an `nn.Module`) keeps its format's answer for each kind and (k, n), and asks again only when one of its arrays is another or is kept under another name, its bits or group size change (in value or in type), or its mode is another format's. A defective answer is not kept: the format is asked again at the next call. So a format's `dot()` and `tile()` answer from those alone: the layer's arrays (their names, shapes and dtypes), its mode, bits and group size, and k and n. An answer that depends on another attribute of the layer is kept past that attribute's change.

`declarations` opens the kernel's body, ahead of its own code. When a face has tables it starts with their staging: the copy into threadgroup memory by the threadgroup's threads and one barrier, which every thread of a threadgroup must meet, so nothing of the body may come before it (a `return` ahead of it would leave the others waiting). The tables' bits are in `header`. A kernel that reads a table of its own, without a face, takes it with `kernels.Tables`: `add(table)` gives the Metal expression of an `omlx_table` that reads it, `header` goes in the kernel's header and `prologue` first in its body. Both are empty without a table, and `prologue` when every table is held by value. The staging copies every entry whatever the threadgroup's size, a threadgroup at the grid's edge included. A kernel holds at most 4096 bytes of tables, and a kernel over several faces that would hold more declines (the routed decode keeps the shared expert's launches apart).

MLX compiles a kernel when it first evaluates it, which in a model is wherever the graph is evaluated: too late to fall back. So a kernel in a model's forward evaluates a launch inside the call that builds it first, and keeps what it learns in a `kernels.Proof`: the launches that compiled (`ran`) and whether one failed (`failed`). `kernels.proof(kernel, ...)` gives the one proof of a key: the kernel's name, its faces and what else its source is instantiated with (the tensors' dtypes, the shapes), so that layers of the same layout share it. Once a launch has failed, whatever the failure of its first evaluation was, what shares the proof keeps its composed path until the process ends.

`proof.prove(metal, build, log, engaged, failed)` is that step, for one Metal that the kernel is instantiated as (`metal`: what its source is instantiated with beside the proof's key, such as `x`'s dtype). It answers at once for a Metal that ran, and for a proof that failed. Otherwise `build()` gives a stand-in: its inputs, not evaluated, and a function that launches it over them. The proof evaluates the inputs and synchronizes the GPU's stream, then evaluates the launch: one that runs is marked in `ran` and logged (`engaged`); one that fails, with an error of any type, fails the proof and is logged with the error (`failed`). What `build()`, the inputs' evaluation and the function raise is raised: what fails in making the inputs (an allocation, an error that an earlier evaluation left pending on the GPU) is the call's, not taken for the Metal's, and the next call proves the Metal again. The kernel answers for three things. The launch reads the inputs alone, zeros that the kernel makes for it (`kernels.zeros`, on the GPU's stream), and no array of the call or tensor of a layer: those may belong to a function that MLX is tracing, where they cannot be evaluated, what fails in them is their caller's, and their size is the call's. MLX generates the Metal of the call's launch for it. MLX writes a launch's Metal from the kernel's header and body, the template's values, the outputs' dtypes and, input by input, the dtype, whether the input has fewer than 8 elements and whether it has no axis; the other shapes, the grid and the output's shape are no part of it. And the stand-in has the call's threadgroup, which MLX checks against the pipeline's limit at every evaluation (the grid it does not check). So the smallest stand-in of the same Metal and threadgroup proves every launch of it, and a face is safe over tensors and activations of any value, zeros included (see [The dot face](#the-dot-face)).

Building a launch has MLX write the kernel's source out, each time. A launch's graph depends on the shapes and dtypes of its arrays alone, so a kernel can build it once for each, as a function of the arrays that it hands to `mx.compile`, and keep the result in the proof: `proof.keep(signature, graph)`, then `proof.graphs.get(signature)` at the calls after, which replay it. The signature covers everything that is computed outside the function and built into it: the pairs, the grid, the output's shape, whatever the kernel works out of the arrays' shapes and dtypes before it writes the function (a layer's signature is the shape of `x`, its dtype and the shape and dtype of the indices). MLX's compile tells apart only what the function reads from its arguments: given arguments of other shapes or dtypes it traces the function again, with everything else as the first call left it. Under a signature of the dtype alone, a call of three rows would replay the function of a call of one, with one row's grid and output. The function closes over no array: the proof lasts as long as the process, and the layers that share it pass their own tensors as arguments. A thread keeps `kernels.MAX_GRAPHS` graphs of a proof (a decode step's, those of the short blocks around it, a few prompts' chunks); one more drops them all, and they are built again as they are called. A kernel reports a failed launch with `proof.fail()`, which also drops the graphs, every thread's: nothing that shares the proof launches again, and `keep` keeps nothing from then on. Setting `proof.failed` by hand would leave the kept graphs replaying: a call that finds its graph does not read `failed`.

Several threads run kernels: each loaded model's engine has a thread of its own, and the loads, the embeddings and the rerankers share another. A proof is the process's, with what it learned: MLX builds a kernel's Metal once for the device, so a launch that one thread proved (`ran`) is built by the others without a proof, and a launch that failed on one (`failed`) stops them all. Nothing is held while a stand-in is evaluated: two threads that meet a Metal unproven at the same moment each prove it. The graphs are each thread's: `proof.graphs` is the calling thread's own, and `proof.keep` keeps a graph for the thread that calls it, which is the one that built it. MLX keeps the launch that it traces a compiled function into in the thread that calls the function, and when the function ends it forgets that launch in one thread only, the last that called it; in another thread the launch would stay, for the next function that lands on the same address to replay. So a kernel builds, calls and keeps a graph in one call, and never hands one to another thread. It keeps a graph nowhere but in the proof: what it builds for a layer (a plan) is read by every thread that calls the layer. A thread's graphs end with the thread: a model that is loaded again builds its own. What a kernel or a format keeps for itself between calls (a cache of kernels, of faces, of sources) is shared by every thread, and must hold when two reach it at once.

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

The walk over the blocks of K in `qmv_rows` is a loop: unrolled into a fold, the fused Qwen decode ran 3 to 8% slower on an M5 Max. The rows of a block are folds, and so are the vectors and rows of a shared block in an unrolled kernel: there the folds measured as fast as the unrolled loops or faster on the M5 Max and the M3 Ultra (a MiniMax shape and the Qwen GDN input projection, every bit equal).

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
<omlx skeletons, docs/quant-formats.md>:104:1: error: static_assert failed due to requirement
'Lanes<512>::BLOCK == 32 * Lanes<512>::VPT' "dot face: Lanes<K>::BLOCK must be VPT inputs for
each of a simdgroup's 32 lanes"
<kernel>:5:5: note: in instantiation of function template specialization
'qmv_rows<float, 512, 1, 1, omlx_myformat_b8::Dot, const device float *>' requested here
```

```
<omlx skeletons, docs/quant-formats.md>:108:1: error: static_assert failed due to requirement
'60 % Lanes<60>::VPT == 0' "dot face: K must be a whole number of a lane's VPT inputs"
```

A mistake inside the format is cited in the format:

```
<quant format myformat_b8>:21:36: error: use of undeclared identifier 'weights'
```

A missing or mismatched member is cited where the skeleton uses it, with a note on the format's line when there is one:

```
<omlx skeletons, docs/quant-formats.md>:82:18: error: no member named 'finish' in
'omlx_myformat_b8::Dot::Lanes<512>'
```

```
<omlx skeletons, docs/quant-formats.md>:43:57: error: too many arguments to function call,
expected 3, have 4
<quant format myformat_b8>:38:18: note: 'madd' declared here
```

The last one is a `madd` declared without its `aux` parameter.

## Test a format

`omlx.formats.testing.dot_rows` runs a dot face in the row skeleton, outside any model:

```python
import mlx.core as mx
from omlx.formats import testing


def test_rows_are_the_dequantized_products():
    face = FORMAT.dot(layer, k, n)
    operands = [layer[name] for name in ("weight", *face.tensors)]
    y = testing.dot_rows(face, operands, x, n, packs=(2, 2))
    assert mx.array_equal(y, dequantize(layer) @ x)
```

`operands` are the weight, then the face's tensors in order. `x` holds K activations. `packs=(NA, NB)` are the rows a simdgroup computes: `NA` rows as the first pack, the `NB` after them as the second. A kernel puts such adjacent rows in one pack; `dot_rows` splits them to run both. `n` is one or more times `NA + NB`, and no more than the matrix's rows (which `dot_rows` does not check). `x` and each operand have 8 elements or more: MLX binds a smaller input as constants. With `staged=True` the kernel copies `x` into threadgroup memory first and the face's `load` reads it there, as in a kernel that stages its inputs; `x` then fits in a threadgroup's 32 KiB. A `load` that takes only a device pointer fails to compile there: `no matching function for call to 'load'`.

oQ writes, and oMLX loads, a checkpoint in a format only for a family verified with
formats (see [Quantize a checkpoint](#quantize-a-checkpoint)). A test that writes or
loads a tiny checkpoint of any family (Llama's too), to test a format with, takes it as
verified for the test with `omlx.formats.testing.verified(*model_types)` (include the
top-level backbone type and its nested text model type). Tests that build layers by hand
use the same block. Dedicated decision and embedding readers still refuse format
checkpoints. During verification, while one of its blocks is open, in any thread, oQ
writes and oMLX loads the family as a verified one. Blocks of several threads open and
close in any order. It decorates a test, async too, and a fixture opens a block of it (a
bare `@verified` is refused):

```python
import pytest
from omlx.formats import testing


@testing.verified("qwen3_5_moe_text")
def test_a_tiny_moe_is_written_in_the_format(tmp_path):
    ...


@pytest.fixture(autouse=True)
def tiny_families_verified():
    with testing.verified("qwen3_5_moe_text"):
        yield
```

Worth testing for every layout a format gives a face for:

- rows against the format's reference, bit for bit when the format has an exact composed path,
- each lane's part of the rows through cursors stepped and placed: `stepped, placed = testing.lane_parts(face, operands, x, n)` reads `madd(0, row(0), xt, aux)` over each lane index's `VPT` inputs, by lane `l`'s cursor stepped `b` blocks and by a cursor built at `32 * b + l`; the two are equal bit for bit (column `j` of each is lane index `j`'s part, zeros past `K / VPT`),
- `finish` as a scale: `got = testing.finish_rows(face, operands, k, n, sums, dtype)` gives `finish(0, s)` of each row for each float32 sum `s`, in each lane (shape `(n, 32, len(sums))`); with `sums = [0, 1, -1, 2, 4, 0.5, 2**-20, 1024]`, every lane equals lane 0, `finish(r, 0) == 0`, `finish(r, -1) == -finish(r, 1)` and `finish(r, 2**j) == 2**j * finish(r, 1)`, bit for bit,
- the same rows with different `packs`, with `staged=True`, and through `testing.shared_rows(face, operands, xs, n, rps=...)` (1 to 16 vectors `xs`, each block's weights by `decode` once, then `apply` for each vector; `rps` rows a simdgroup, `unrolled` the unrolled skeleton): no vector's row may change by a bit,
- a K of whole blocks and a K that ends in a partial block, if the face takes both,
- the face's tables, with `testing.staged_tables(face, operands, dtype)`: one launch over the face's operands, compiled for activations of `dtype`, that reads each table back through the face's member of its name, as a kernel stages it. Each array it returns has the table's dtype, length and bits (`table.bits`) when the members are declared as [Tables](#tables) says,
- for a stackable format, compare two matrices with `testing.rows_stack` (for checkpoint tensors, pass `layer.fmt`). With `x`, the helper builds three format layers to check dot rows. In a format test, run that check inside the family's verification block. Omitting `x` checks the decoded rows directly.

  ```python
  with testing.verified("qwen3_5_moe_text"):
      testing.rows_stack(layer.fmt, first, second, k, bits, group_size, x)
  ```

- the rows in fp32 too, if the reference can be computed in fp32: rounded to bf16 or fp16, a sum in another order than the reference's shows in one row in a thousand or fewer, in fp32 in most rows. `dot_rows` compiles the skeleton for `x.dtype` and refuses a dtype the face is not exact for. For a face whose tensors are typed `T`, `dataclasses.replace(face, dtypes=(mx.float32,))` runs the same traversal in fp32, on fp32 activations and fp32 copies of those tensors.

`omlx.formats.testing.tile_rows` reads every weight of some rows through a tile face, as a tile kernel reads it (`weight(params(g), fetch(run), i)`, with `i` not known when the kernel is compiled), and gives them twice: as `weight` gives them, in float32, and rounded once to a dtype the face is exact for, as a kernel stores them for activations of that dtype:

```python
def test_the_tile_face_is_the_decode_in_float32():
    face = FORMAT.tile(layer, k, n)
    operands = [layer[name] for name in ("weight", *face.tensors)]
    exact, rounded = testing.tile_rows(face, operands, k, range(n), mx.bfloat16)
    want = decode_in_float32(layer)  # (n, k)
    assert mx.array_equal(exact, want).item()
    assert mx.array_equal(rounded, want.astype(mx.bfloat16)).item()
```

`rows` are the rows' indices (in a stack, row `n` of matrix `e` is `e * N + n`), none past the matrix's, which is not checked; K is a whole number of runs of 8; the operands have 8 elements or more. The reference is the format's decode computed in float32: for a format whose `decode` returns another type (affine returns the scales' type), compute it in float32 for the test. Worth testing for every layout a format gives a tile face for:

- every weight of rows of each matrix of a stack, its last rows included, against the reference, in float32 and rounded to each dtype the face is exact for,
- the codes and the scales at their largest values and at random ones, with float fields finite, and at zero: rules 1, 2 and 7 of [The tile face](#the-tile-face). A NaN the decode gives compares as a NaN, not by its bits,
- a K whose last group is partial, if the face takes one.

A layer runs a call's rows through its face, or through the format's `decode` (see [Layers](#layers)), and when the face is exact the two give the same rows: comparing a layer's rows with the face's does not tell which ran. `omlx.formats.testing.layer_calls()` records, in its block's thread, where the rows of format layers ran, each where it ran and tagged with its layer, as a `LayerCall(layer, face=None, metal=None)`:

- a launch of the face, or the replay of the launch a layer kept: `face` is that face, and `metal` what the launch's Metal is built for beside it. For a `FormatLinear` that is the activations' dtype; for a `FormatSwitchLinear` the tuple `(dtype, per_row, few)`: the pairs of a row of `x` (as many as its indices where each row of `x` meets its own run of experts, 1 where `x` is broadcast against the indices) and whether the call has fewer than 8 pairs, which MLX binds as constants;
- the format's `decode`: `face` and `metal` are None. A decode is recorded for the layer whose decode it is, whatever runs it.

```python
with testing.layer_calls() as calls:
    y = layer(x)
assert calls == [testing.LayerCall(layer, FORMAT.dot(layer, k, n), x.dtype)]
```

Two records are equal when they hold the same layer, the very object (`is`: a layer is a dict of arrays, which `==` does not compare), with equal `face` and `metal`. A record has no hash, like the layer it holds: compare lists of records, not sets. A call's rows are recorded when the call returns, in order with those of the calls it made.

A face that does not compile leaves its layers to `decode`, so a call recorded with its face ran that face's launch. The layers are wrapped while a block is open in any thread, and rows in a thread without one are not recorded; with no block open, a call is the layer's own and costs nothing more. Not recorded either: a call that raises and the calls it made, rows a layer's subclass computes its own way, a call that a function compiled by `mx.compile` replays (no Python of the layer runs: the call is recorded once, when the function is traced), and a fused kernel's launch over a layer's tensors.

A launch that fails keeps the layers and kernels of its proof's key on their composed path for the life of the process (see [Kernels over faces](#kernels-over-faces)): a test that fails a face would leave the tests after it on `decode`. `omlx.formats.testing.fresh_proofs()` gives its block a registry of proofs of its own and gives the old one back when the block ends. oMLX runs each of its tests in one, and a plugin can too, from an autouse fixture:

```python
import pytest
from omlx.formats import testing


@pytest.fixture(autouse=True)
def kernel_proofs():
    with testing.fresh_proofs():
        yield
```

A test's `monkeypatch` is undone after the fixture's block ends, unless `monkeypatch` depends on the fixture: a test that patches the registry would then leave its patch as the process's registry. oMLX's `tests/conftest.py` overrides `monkeypatch` to request the fixture (`def monkeypatch(kernel_proofs, monkeypatch): return monkeypatch`); a plugin that patches the registry in its tests does the same. The block's value is its registry, a dict of `kernels.Proof`, which is not part of the contract. The registry is the process's while the block is open: every thread's kernels prove their launches in it, and blocks may end in any order, in any thread. `omlx.formats.testing.no_dot_defects()` holds the process's set of defects the same way: the latest of its blocks still open takes the defects that any thread meets. A layer, and a kernel that a layer of an MoE block feeds, keeps the plan and proof of whichever call ran it first: a layer or a model shared between tests (built by a module or session fixture, or kept by a test for later) keeps the proofs of the first test that ran it, and a face that failed there stays failed for it in every later test. Build the layers a test runs inside the test, or run a shared one first inside the block.

## Versions

This page is a contract: a format written to it keeps running as oMLX changes. The contract is what a format uses and what oMLX asks of it:

- `QuantFormat`, `Dot`, `Tile`, `Table`, `FormatRefusedError`, `register`, `load`, `get`, `origin`, `dot`, `tile`, `stackable`, `FLOATS`, `CONTRACT` and `OLDEST_CONTRACT` in `omlx.formats`, `dot_rows`, `shared_rows`, `lane_parts`, `finish_rows`, `rows_stack`, `tile_rows`, `staged_tables`, `no_dot_defects`, `fresh_proofs`, `layer_calls`, `LayerCall` and `verified` in `omlx.formats.testing`, and the `omlx.quant_formats` entry-point group,
- the members of a face, the order a kernel runs them in and the rules a face follows,
- the names of oMLX's Metal that a format's source may use: `omlx_ints`, `omlx_iota` and `omlx_table`.

The contract has a revision, a whole number that goes up by one with each change. `omlx.formats.CONTRACT` is the revision this oMLX implements. `omlx.formats.OLDEST_CONTRACT` is the oldest revision whose formats it still runs.

| Revision | In oMLX since | What it adds |
|---|---|---|
| 1 | not released yet | The registry, the dot face and the tile face, a face's tables, what loads and what writes a checkpoint (`layout`, `encode`, and its record: `record`, `accept`), `omlx.formats.testing`. |

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
<omlx skeletons, docs/quant-formats.md>:18:1: error: static_assert failed due to requirement
'2 <= 1' "dot face: written for a later revision of the contract than this oMLX runs"
<omlx skeletons, docs/quant-formats.md>:97:7: note: in instantiation of template class
'omlx_revision<2>' requested here
<kernel>:5:5: note: in instantiation of function template specialization
'qmv_rows<float, 512, 1, 1, omlx_myformat_b8::Dot, const device float *>' requested here
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
| Bits | 2, 3, 4, 5, 6, 8 |
| Group size | 32, 64, 128 |
| Scales and biases | both bf16 or both fp16. The face is exact for activations of that dtype (its `dtypes`); no kernel is built on it for another. |
| Traversal | MLX's `qmv_fast` when K is a multiple of its block (512 at 2 to 5 bits, 256 at 6 and 8) and, for stacked expert weights, N is a multiple of 8. MLX's `qmv` otherwise. |
| Face key | `affine_b{bits}g{group_size}f` (`qmv_fast`) or `affine_b{bits}g{group_size}s` |

Any other affine layer has no dot face and runs MLX's ops:

- another bit width or group size, fp32 scales, biases of another dtype than the scales, no scales or no biases, a K that is not whole groups, weights that are neither 2-D nor 3-D or that are not the N by K asked for,
- a matrix of fewer than 8 groups, counted in one expert for stacked weights (MLX binds an input of fewer than 8 elements as constants, not as a buffer),
- a 2-D matrix of K 64 or 128 at 2, 4 or 8 bits: MLX runs one row of it through `qmv_quad`, which sums in another order than `qmv`.

The `qmv_fast` face does not compile for a K that is not whole blocks: `AffineFormat.dot` is what gives a matrix its traversal.

Its tile face gives each weight as `scale * q + bias` computed in float32, which a kernel rounds once to the activations' dtype: MLX's `dequantize` of the weight over float32 copies of the scales and the biases (`scale * q` is exact in float32 where it is finite), and what MLX's sorted `gather_qmm` loads into its tiles. MLX's `dequantize` over bf16 or fp16 scales computes in their type, and can differ from it.

| | Covered |
|---|---|
| Bits | 4, 8: a run of 8 codes is one 32-bit word, or two. 2- and 3-bit runs are 2 and 3 bytes, which the face's `fetch` does not read: `affine.tile_face` refuses them. |
| Group size | 32, 64, 128 |
| Scales and biases | both bf16 or both fp16, one of each for each group of a row: exact for activations of that dtype. |
| Weights | 32-bit words, a matrix or a stack of them, K whole groups (its `Runs<K>` asserts it), scales of 8 elements or more |
| Face key | `affine_tile_b{bits}g{group_size}_bf16`, or `_fp16`, by the scales' dtype |

## Kernels built on faces

| Kernel | Face | Code |
|---|---|---|
| Qwen3.5-family one-token MoE decode: routed and shared expert rows | dot | `omlx/patches/qwen35_moe_routed_decode.py` |
| Qwen3.5-family verify blocks of 2 to 8 rows: routed expert rows, grouped by expert | dot | `omlx/patches/moe_verify_gather.py` |
| A format's own layers: the rows of a decode step or of a short block, a switch layer's pairs | dot | `omlx/formats/linear.py` |
| M5 sorted prefill and fused SwiGLU | tile | `omlx/patches/m5_gather_qmm_nax.py` |

The routed decode runs a block in the one dtype its gate+up face is exact for. With a gate+up face that names no dtype (exact for the three of `formats.FLOATS`), or several, the block keeps its composed path, as it does with a down face that is not exact for that dtype. Its launches take rows in whole groups: a block whose expert width or hidden size is not a multiple of 4 keeps its composed path too (every affine layout is whole groups), and a shared expert whose width is not is left out of the fused launches.

A face that does not compile does not fail the model. One token's launches (the router's, gate+up and down) are one graph of the token and the block's arrays: built once on a thread for the token's shape and dtype and the router's width, kept in the proof of the block's faces, tensor dtypes and shapes (see [Kernels over faces](#kernels-over-faces)), and replayed by every block that shares the proof, over its own arrays. Before the graph is first built its Metal is proven on a stand-in, as a format layer's is: zeros for the token, the router's weight, each shared matrix and what the block computes outside the launches, as many matrices of each stack as a token selects, and router logits that rank the experts, so that it selects those, all made on the GPU's stream and evaluated by the proof ahead of the launch (what fails there is raised by the call and fails no proof). Nothing of the call is evaluated (once, as a block's plan is built outside a trace, a view of each stack's first matrix is, to learn whether it shares the stack's buffer): a failure in the block's input or tensors is raised where the caller evaluates the result, and a block's first call is safe inside a function that `mx.compile` traces, over the model's state too (the stacks are then bound whole). If the stand-in fails, the compiler's messages are logged (`fused routed-expert decode failed; composed fallback`) and those blocks keep their composed path, for one token, verify windows and batched rows. A verify window or batched rows of 2 rows or more are a graph of the same kind, kept in the window's own proof for the window's shape and dtype and the router's width, and proven on a stand-in of the window's rows; a window of one row is one token's call. If a window's stand-in fails (`fused verify-window experts failed; verifier fallback`), those blocks keep the composed path for windows and batched rows, and the fused one for one token. Blocks of other faces or shapes are not affected.

On the M5 Max and later, MLX's affine switch layers use the affine tile face
for sorted prompt rows. The gather can apply SwiGLU to fused gate/up outputs
using sorted rows or a row map. Before use, each configuration checks the
plain result against MLX's sorted `gather_qmm` at the call's K. The fused
variants check the result of splitting the plain output and applying SwiGLU.
A format's own switch layers use their rows kernel.

The verify gather runs a block's fused gate+up projection, then its down projection, when both are stacked expert weights without a bias, of an N that is whole tiles of 8 outputs, each with a dot face that is exact for the block's dtype, and the block's activation returns that dtype in the shape of the up's rows. From 24 pairs (rows times top-k) one launch puts a block's pairs in expert order ahead of its two; below, each threadgroup searches for its own. A block of 2 to 8 rows takes it at any bit width, but one of 16 pairs or fewer over 64 experts or more, which keeps the composed path (measured faster than the composed path everywhere else; 2 rows of a 256-expert top-8 block read 1% slower). The block's indices are integers, top-k of them (one or more) for each row of `x`, and `x` is float16, bfloat16 or float32 (as `gather_qmm` takes) with 8 values or more (MLX binds fewer as constants). The gather proves each Metal of a block's two launches (and, from 24 pairs, its order's) as a format layer does, on a stand-in: zeros of the block's rows and pairs, every pair with expert 0 of stacks of the fewest experts that leave each tensor 8 elements, views of one buffer made on the GPU's stream and evaluated by the proof ahead of the launch (what fails in making them is raised by the call and fails no proof). It evaluates nothing of the call, so a block may be built inside a function that `mx.compile` traces, and what fails in `x`, in the indices, in a tensor or in the activation's arrays is raised where the caller evaluates the result; the activation itself runs in the call, so what it raises, the call raises. A block whose stand-in fails (`expert-ordered verify gather failed; composed fallback`) keeps the projections of the same faces, tensor dtypes and shapes on their composed path, on every thread.
