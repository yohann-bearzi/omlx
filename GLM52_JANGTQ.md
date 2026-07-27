# GLM-5.2 JANGTQ support for oMLX

This branch adds two things to oMLX:

1. **JANGTQ checkpoint loading for `glm_moe_dsa`** - GLM-5.2 bundles quantized
   with JANGTQ ship flat per-expert weights that stock `mlx_lm` sanitize cannot
   handle. They are routed to `load_jangtq_model`, which does the flat-to-nested
   key bridging.
2. **A steel-tiled TurboQuant MoE GEMM** - a `TurboQuantBlockLoader` that feeds
   `mlx::steel`'s tiled GEMM from codebook weights, so prefill amortizes each
   expert's weight read across a block of rows instead of re-reading it per
   token.

## Why the kernel

The per-row TQ gather kernel reads an expert's entire packed weight matrix once
per token routed to it. With top-8 routing and a 512-token chunk, each of ~160
experts is read ~25 times - a single projection of a single layer moves ~13 GB,
and prefill is bandwidth-bound on pure redundancy.

Sorted indices already group rows by expert. The steel kernel exploits that:
one threadgroup per (expert, output tile, row block), weight tile unpacked once,
accumulated against every row in the block via simdgroup fragments.

Isolated, on GLM-5.2 expert shapes (6144->2048, 2-bit, 160 experts, M3 Ultra):

| rows | per-row kernel | steel | speedup |
|-----:|---------------:|------:|--------:|
|  512 |         3.1 ms | 2.5 ms | 1.2x |
| 4096 |        22.3 ms | 5.9 ms | 3.8x |

Decode (batch 1) keeps the per-row kernel - at one row a BM x BN tile wastes most
of every fragment, and steel measures ~40% slower there.

## Measured end-to-end

`GLM-5.2-JANGTQ_K-mxfp4attn` (236 GB) on an M3 Ultra / 512 GB:

| | before | after |
|---|---:|---:|
| decode | 10.6 tok/s | **18.8 tok/s** |
| prefill (1024 tok) | 79 tok/s | **177 tok/s** |

For reference, the same model as `GLM-5.2-mxfp4` (386 GB) runs 19.2 tok/s decode
and 121 tok/s prefill - so the JANGTQ bundle is within 2% on decode, ~45% faster
on prefill, and 150 GB smaller.

## Building

The kernel needs the Metal toolchain (full Xcode, not just Command Line Tools)
and Python 3.11-3.13:

    python3.13 -m venv ~/omlx-venv313
    OMLX_WITH_CUSTOM_KERNEL=1 \
      CMAKE_ARGS="-DPython_EXECUTABLE=$HOME/omlx-venv313/bin/python3.13" \
      ~/omlx-venv313/bin/pip install -e . --no-build-isolation

Verify the kernels registered:

    python -c "from omlx.custom_kernels.glm_moe_dsa import fast; print(fast._ext is not None)"

`/api/status` also reports `custom_kernels.glm_moe_dsa.available`.

## Environment overrides

None are required. For debugging:

- `JANGTQ_STEEL=0` - disable the steel path, fall back to the per-row kernel
- `JANGTQ_STEEL_STRICT=1` - raise instead of silently falling back
- `OMLX_GLM_FUSED_GATE_UP=0|1` - force the split/fused expert layout

## Upstreamability

The kernel is additive: a new `.metal` source, loader header, primitive, and
binding. It does not modify existing paths, and the JANGTQ routing is one entry
in an existing model-type list.
