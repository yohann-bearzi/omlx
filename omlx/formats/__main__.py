# SPDX-License-Identifier: Apache-2.0
"""``python -m omlx.formats``: debug, inspect and profile format layers.

Subcommands map one-to-one onto the pure ``omlx.formats.debug`` API
(``explain``, ``dump-metal``, ``bench``, ``warm-report``); ``audit`` is
dispatched lazily to the ``omlx.formats.audit`` module so a build without it
still exposes the rest of the CLI. Neither this module nor ``debug`` is
imported by any runtime ``omlx.formats.*`` module (Rule 28), and neither reads
environment variables (Rule 30).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import mlx.nn as nn

from omlx.formats import debug

_ROWS_DEFAULT = ",".join(str(rows) for rows in debug.DEFAULT_EXPLAIN_ROWS)


def _positive_int(text: str) -> int:
    value = int(text, 10)
    if value < 1:
        raise ValueError(f"expected a positive integer, got {text!r}")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text, 10)
    if value < 0:
        raise ValueError(f"expected a non-negative integer, got {text!r}")
    return value


def _parse_int_list(text: str, name: str) -> tuple[int, ...]:
    parts = [part.strip() for part in text.split(",")]
    if any(not part for part in parts):
        raise ValueError(
            f"{name} must be a comma-separated list of integers, not {text!r}"
        )
    return tuple(_positive_int(part) for part in parts)


def _parse_str_list(text: str, name: str) -> tuple[str, ...]:
    parts = [part.strip() for part in text.split(",")]
    if any(not part for part in parts):
        raise ValueError(f"{name} must be a comma-separated list, not {text!r}")
    return tuple(parts)


def _parse_shapes(text: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for token in text.split(","):
        token = token.strip()
        lowered = token.lower()
        if lowered.count("x") != 1:
            raise ValueError(f"--shapes entries must be NxK, not {token!r}")
        n_text, k_text = lowered.split("x")
        shapes.append((_positive_int(n_text), _positive_int(k_text)))
    return shapes


def _add_layer_options(parser: argparse.ArgumentParser, *, shape: bool) -> None:
    parser.add_argument("--mode", required=True, help="quant format mode name")
    parser.add_argument("--bits", required=True, type=_positive_int)
    parser.add_argument("--group-size", required=True, type=_positive_int)
    if shape:
        parser.add_argument("--k", required=True, type=_positive_int, help="input")
        parser.add_argument("--n", required=True, type=_positive_int, help="output")
    parser.add_argument(
        "--dtype", default="bf16", help="activation dtype (bf16, fp16 or fp32)"
    )
    parser.add_argument(
        "--plugin", default=None, metavar="MOD:ATTR", help="plugin format path"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omlx.formats",
        description=(
            "Inspect, explain, dump and benchmark oMLX quant format layers "
            "without coupling the runtime to this tooling."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    explain = sub.add_parser(
        "explain", help="trace faces, tuning probes and per-row route choice"
    )
    _add_layer_options(explain, shape=True)
    explain.add_argument(
        "--rows", default=_ROWS_DEFAULT, metavar="R1,R2,...", help="row counts"
    )
    explain.add_argument("--experts", type=_positive_int, default=None)
    explain.add_argument("--bias", action="store_true")
    explain.add_argument("--unsorted", action="store_true")
    explain.add_argument("--chip", default=None, help="target chip ID")
    explain.add_argument("--input-ndim", type=int, default=3, choices=(2, 3))
    explain.add_argument("--json", action="store_true")

    dump = sub.add_parser(
        "dump-metal", help="emit the exact standalone Metal source and launch geometry"
    )
    _add_layer_options(dump, shape=True)
    dump.add_argument("--rows", type=_positive_int, default=1)
    dump.add_argument("--experts", type=_positive_int, default=None)
    dump.add_argument("--bias", action="store_true")
    dump.add_argument("--unsorted", action="store_true")
    dump.add_argument("--chip", default=None, help="target chip ID")
    dump.add_argument("--route", default=None, help="force a kernel route")
    dump.add_argument("--per-row", type=_positive_int, default=1)
    dump.add_argument("--input-ndim", type=int, default=3, choices=(2, 3))
    dump.add_argument("--no-preamble", action="store_true")
    dump.add_argument(
        "--out", default=None, metavar="PATH", help="write .metal to PATH"
    )
    dump.add_argument("--json", action="store_true")

    bench = sub.add_parser(
        "bench", help="Rule-31 chained micro-bench, C2 FP32 check and memory delta"
    )
    _add_layer_options(bench, shape=True)
    bench.add_argument("--rows", type=_positive_int, default=1)
    bench.add_argument("--chain-len", type=_positive_int, default=16)
    bench.add_argument("--units", type=_positive_int, default=4)
    bench.add_argument("--rounds", type=_positive_int, default=40)
    bench.add_argument("--warmup", type=_non_negative_int, default=5)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--bias", action="store_true")
    bench.add_argument("--input-ndim", type=int, default=3, choices=(2, 3))
    bench.add_argument("--json", action="store_true")

    warm = sub.add_parser(
        "warm-report", help="report warm-up pipelines, dedup and compile timing"
    )
    _add_layer_options(warm, shape=False)
    warm.add_argument("--k", type=_positive_int, default=None)
    warm.add_argument("--n", type=_positive_int, default=None)
    warm.add_argument("--repeat", type=_positive_int, default=1)
    warm.add_argument("--shapes", default=None, metavar="NxK,...")
    warm.add_argument("--dtypes", default=None, metavar="bf16,fp16")
    warm.add_argument("--experts", type=_positive_int, default=None)
    warm.add_argument("--per-rows", default="1", metavar="R1,R2,...")
    warm.add_argument("--dry-run", action="store_true")
    warm.add_argument("--warm-dispatch", action="store_true")
    warm.add_argument("--json", action="store_true")

    sub.add_parser(
        "audit",
        help="run the omlx.formats.audit scanner (delegated to omlx.formats.audit)",
        add_help=False,
        description="Arguments are forwarded verbatim to omlx.formats.audit.main.",
    )
    return parser


def _dispatch_audit(rest: Sequence[str]) -> int:
    try:
        module = importlib.import_module("omlx.formats.audit")
    except ModuleNotFoundError as exc:
        if exc.name in ("omlx.formats.audit", "audit"):
            print(
                "omlx.formats audit: omlx.formats.audit is not available in "
                "this build",
                file=sys.stderr,
            )
            return 2
        raise
    entry = getattr(module, "main", None)
    if not callable(entry):
        print(
            "omlx.formats audit: omlx.formats.audit has no callable main",
            file=sys.stderr,
        )
        return 2
    try:
        result = entry(list(rest))
    except SystemExit as exc:
        code = exc.code
        return 0 if code is None else (code if isinstance(code, int) else 1)
    return 0 if result is None else int(result)


def _run(args: argparse.Namespace) -> int:
    if args.command == "explain":
        layer = debug.make_layer(
            args.mode,
            bits=args.bits,
            group_size=args.group_size,
            k=args.k,
            n=args.n,
            dtype=args.dtype,
            experts=args.experts,
            bias=args.bias,
            plugin=args.plugin,
        )
        result = debug.explain(
            layer,
            rows=_parse_int_list(args.rows, "--rows"),
            dtype=args.dtype,
            chip=args.chip,
            sorted_indices=not args.unsorted,
            input_ndim=args.input_ndim,
        )
        print(debug.format_explain(result, as_json=args.json))
        return 0

    if args.command == "dump-metal":
        layer = debug.make_layer(
            args.mode,
            bits=args.bits,
            group_size=args.group_size,
            k=args.k,
            n=args.n,
            dtype=args.dtype,
            experts=args.experts,
            bias=args.bias,
            plugin=args.plugin,
        )
        result = debug.dump_metal(
            layer,
            rows=args.rows,
            dtype=args.dtype,
            chip=args.chip,
            route=args.route,
            sorted_indices=not args.unsorted,
            per_row=args.per_row,
            input_ndim=args.input_ndim,
            include_mlx_preamble=not args.no_preamble,
        )
        if args.out is not None:
            Path(args.out).write_text(result.metal, encoding="utf-8")
            print(debug.format_dump_metal_summary(result, as_json=args.json))
        elif args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            sys.stdout.write(result.metal)
        return 0

    if args.command == "bench":
        layers = [
            debug.make_layer(
                args.mode,
                bits=args.bits,
                group_size=args.group_size,
                k=args.k,
                n=args.n,
                dtype=args.dtype,
                bias=args.bias,
                seed=args.seed + unit,
                plugin=args.plugin,
            )
            for unit in range(args.units)
        ]
        result = debug.bench(
            layers,
            rows=args.rows,
            dtype=args.dtype,
            chain_len=args.chain_len,
            rounds=args.rounds,
            warmup=args.warmup,
            seed=args.seed,
            input_ndim=args.input_ndim,
        )
        print(debug.format_bench(result, as_json=args.json))
        return 0

    if args.command == "warm-report":
        if args.shapes is None:
            if args.k is None or args.n is None:
                raise ValueError(
                    "warm-report requires --k and --n, or --shapes NxK,..."
                )
            shapes = [(args.n, args.k)] * args.repeat
        else:
            if args.k is not None or args.n is not None or args.repeat != 1:
                raise ValueError("--shapes cannot be combined with --k/--n/--repeat")
            shapes = _parse_shapes(args.shapes)
        layers = [
            debug.make_layer(
                args.mode,
                bits=args.bits,
                group_size=args.group_size,
                k=k,
                n=n,
                dtype=args.dtype,
                experts=args.experts,
                seed=index,
                plugin=args.plugin,
            )
            for index, (n, k) in enumerate(shapes)
        ]
        target: Sequence | nn.Module = layers
        if args.warm_dispatch:
            bag = nn.Module()
            for index, layer in enumerate(layers):
                setattr(bag, f"layer_{index}", layer)
            target = bag
        report = debug.warm_report(
            target,
            dtypes=(
                _parse_str_list(args.dtypes, "--dtypes")
                if args.dtypes is not None
                else None
            ),
            compile=not args.dry_run,
            warm_dispatch=args.warm_dispatch,
            per_rows=_parse_int_list(args.per_rows, "--per-rows"),
        )
        print(debug.format_warm_report(report, as_json=args.json))
        return 0

    raise ValueError(f"unknown command {args.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code (0 ok, 1 runtime, 2 usage)."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    # Dispatch ``audit`` before argparse so every flag (including --help)
    # reaches the audit module verbatim.
    if args_list and args_list[0] == "audit":
        return _dispatch_audit(args_list[1:])
    parser = _build_parser()
    try:
        args = parser.parse_args(args_list)
    except SystemExit as exc:
        code = exc.code
        return 0 if code is None else (code if isinstance(code, int) else 2)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        return _run(args)
    except (ValueError, TypeError) as exc:
        print(f"python -m omlx.formats: {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, OSError) as exc:
        print(f"python -m omlx.formats: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
