# SPDX-License-Identifier: Apache-2.0
"""AST format-reader audit scanner and pytest CI helper.

Pure Python stdlib (zero ``mlx`` imports, zero environment variables) so it runs
on any host, including CPU-only CI and MLX-free hosts.
"""

from __future__ import annotations

import argparse
import ast
import collections
import fnmatch
import io
import json
import os
import pathlib
import re
import sys
import tokenize
from collections.abc import Sequence
from dataclasses import dataclass

RULES: frozenset[str] = frozenset(
    {
        "unguarded-weight",
        "unguarded-scales",
        "unguarded-biases",
        "unguarded-quantized-matmul",
        "unguarded-dequantize",
        "unguarded-gather-qmm",
        "exact-type-check",
    }
)
INVALID_PRAGMA_RULE: str = "invalid-pragma"

_TENSOR_ATTR_TO_RULE: dict[str, str] = {
    "weight": "unguarded-weight",
    "scales": "unguarded-scales",
    "biases": "unguarded-biases",
}
_MLX_OP_TO_RULE: dict[str, str] = {
    "quantized_matmul": "unguarded-quantized-matmul",
    "dequantize": "unguarded-dequantize",
    "gather_qmm": "unguarded-gather-qmm",
}
_QUANTIZED_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "QuantizedLinear",
        "QuantizedSwitchLinear",
        "QuantizedEmbedding",
    }
)
_FORMAT_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "FormatLinear",
        "FormatSwitchLinear",
        "FormatEmbedding",
        "_FormatLayer",
    }
)
_DENSE_LAYER_NAMES: frozenset[str] = frozenset(
    {
        "Linear",
        "SwitchLinear",
        "Embedding",
    }
)
_PLAIN_DICT_NAMES: frozenset[str] = frozenset(
    {
        "weights",
        "tensors",
        "state_dict",
        "encoded",
        "layout",
        "record",
        "config",
        "values",
    }
)
_MODULE_TENSOR_ATTRS: frozenset[str] = frozenset(
    {
        "weight",
        "scales",
        "biases",
        "bits",
        "group_size",
        "mode",
    }
)
_PRAGMA_PREFIX_RE = re.compile(r"^#\s*omlx-audit\s*:")
_PRAGMA_FULL_RE = re.compile(
    r"^#\s*omlx-audit:\s*(?P<kind>allow-file|allow)"
    r"(?:\[(?P<rules>[^\]]*)\])?"
    r"\s*(?:--\s*(?P<reason>.*))?$"
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    col: int
    rule: str
    scope: str
    symbol: str
    message: str


@dataclass(frozen=True)
class AllowEntry:
    path: str
    rule: str = "*"
    scope: str = "*"
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError(f"AllowEntry.path must be non-empty, not {self.path!r}")
        if self.rule != "*" and self.rule not in RULES:
            raise ValueError(
                f"AllowEntry.rule must be '*' or one of {sorted(RULES)}, "
                f"not {self.rule!r}"
            )
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError(f"AllowEntry.scope must be non-empty, not {self.scope!r}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("AllowEntry.reason must be a non-empty string")


@dataclass(frozen=True)
class AuditReport:
    files_scanned: tuple[str, ...]
    findings: tuple[Finding, ...]
    suppressed: int
    unparsable: tuple[tuple[str, str], ...]
    unused_pragmas: tuple[tuple[str, int, str], ...]

    @property
    def ok(self) -> bool:
        return not self.findings and not self.unparsable


class AuditError(AssertionError):
    """Raised by ``assert_no_unguarded_readers`` when findings or errors exist."""

    def __init__(self, report: AuditReport, message: str | None = None) -> None:
        self.report = report
        if message is None:
            parts: list[str] = []
            if report.findings:
                lines = [
                    f"  {f.path}:{f.line}:{f.col}: [{f.rule}] ({f.scope}) {f.message}"
                    for f in report.findings[:20]
                ]
                if len(report.findings) > 20:
                    lines.append(f"  ... ({len(report.findings) - 20} more)")
                parts.append(
                    f"{len(report.findings)} unguarded format-reader finding(s):\n"
                    + "\n".join(lines)
                )
            if report.unparsable:
                parts.append(
                    f"{len(report.unparsable)} unparsable file(s): "
                    + ", ".join(f"{p} ({err})" for p, err in report.unparsable)
                )
            if report.unused_pragmas:
                parts.append(
                    f"{len(report.unused_pragmas)} unused pragma(s): "
                    + ", ".join(f"{p}:{ln}" for p, ln, _ in report.unused_pragmas[:10])
                )
            message = "; ".join(parts) or "format-reader audit failed"
        super().__init__(message)


@dataclass(frozen=True)
class _Pragma:
    line: int
    col: int
    kind: str  # "allow" | "allow-file"
    rules: frozenset[str]
    reason: str
    raw: str


@dataclass(frozen=True)
class _RawFinding:
    finding: Finding
    end_line: int


def _is_docstring_stmt(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _first_code_line(tree: ast.Module) -> int | None:
    body = tree.body
    start = 1 if body and _is_docstring_stmt(body[0]) else 0
    if start < len(body):
        return body[start].lineno
    return None


def _parse_pragmas(
    source: str, path: str, first_code_line: int | None
) -> tuple[list[_Pragma], list[Finding]]:
    pragmas: list[_Pragma] = []
    invalid: list[Finding] = []
    reader = io.StringIO(source).readline
    for tok in tokenize.generate_tokens(reader):
        if tok.type != tokenize.COMMENT:
            continue
        raw = tok.string
        if not _PRAGMA_PREFIX_RE.match(raw):
            continue
        line, col = tok.start
        match = _PRAGMA_FULL_RE.match(raw)
        if not match:
            invalid.append(
                Finding(
                    path=path,
                    line=line,
                    col=col,
                    rule=INVALID_PRAGMA_RULE,
                    scope="<module>",
                    symbol="omlx-audit",
                    message=f"malformed omlx-audit pragma: {raw!r}",
                )
            )
            continue
        kind = match.group("kind")
        rules_raw = match.group("rules")
        reason = match.group("reason")
        if reason is None or not reason.strip():
            invalid.append(
                Finding(
                    path=path,
                    line=line,
                    col=col,
                    rule=INVALID_PRAGMA_RULE,
                    scope="<module>",
                    symbol="omlx-audit",
                    message="omlx-audit pragma requires a non-empty '-- <reason>'",
                )
            )
            continue
        if rules_raw is not None:
            parsed_rules = [item.strip() for item in rules_raw.split(",")]
            bad = [item for item in parsed_rules if not item or item not in RULES]
            if not parsed_rules or bad:
                invalid.append(
                    Finding(
                        path=path,
                        line=line,
                        col=col,
                        rule=INVALID_PRAGMA_RULE,
                        scope="<module>",
                        symbol="omlx-audit",
                        message=(
                            f"omlx-audit pragma names unknown rule(s): "
                            f"{', '.join(repr(item) for item in bad)}"
                        ),
                    )
                )
                continue
            rule_set = frozenset(parsed_rules)
        else:
            rule_set = RULES
        if (
            kind == "allow-file"
            and first_code_line is not None
            and line >= first_code_line
        ):
            invalid.append(
                Finding(
                    path=path,
                    line=line,
                    col=col,
                    rule=INVALID_PRAGMA_RULE,
                    scope="<module>",
                    symbol="omlx-audit",
                    message=(
                        "allow-file pragma must appear in the file header "
                        "before the first non-docstring statement"
                    ),
                )
            )
            continue
        pragmas.append(
            _Pragma(
                line=line,
                col=col,
                kind=kind,
                rules=rule_set,
                reason=reason.strip(),
                raw=raw,
            )
        )
    return pragmas, invalid


def _expr_key(node: ast.AST) -> str | None:
    """Canonical dotted key for Name / Attribute chains (e.g. ``self.gate_up``)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expr_key(node.value)
        if parent is not None:
            return f"{parent}.{node.attr}"
    return None


def _tail_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _is_norm_or_conv_name(name: str) -> bool:
    low = name.lower()
    return (
        low.endswith("norm") or "layernorm" in low or "rmsnorm" in low or "conv" in low
    )


def _is_kv_cache_name(name: str) -> bool:
    low = name.lower()
    return (
        low.startswith("keys")
        or low.startswith("values")
        or low.startswith("k_cache")
        or low.startswith("v_cache")
        or low.startswith("kv_cache")
        or "cache" in low
    )


def _resolve_class_set(
    node: ast.AST,
    aliases: dict[str, set[str]],
    known: frozenset[str],
) -> set[str]:
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        out: set[str] = set()
        for elt in node.elts:
            out |= _resolve_class_set(elt, aliases, known)
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _resolve_class_set(node.left, aliases, known) | _resolve_class_set(
            node.right, aliases, known
        )
    if isinstance(node, ast.Attribute) and node.attr in known:
        return {node.attr}
    if isinstance(node, ast.Name):
        if node.id in known:
            return {node.id}
        if node.id in aliases:
            return set(aliases[node.id])
    return set()


def _resolve_string_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        out: set[str] = set()
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.add(elt.value)
        return out
    return set()


def _collect_class_aliases(tree: ast.AST, known: frozenset[str]) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = collections.defaultdict(set)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                tail = alias.name.rsplit(".", 1)[-1]
                if alias.asname and tail in known:
                    aliases[alias.asname].add(tail)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            targets: Sequence[ast.AST] = ()
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = (node.target,)
                value = node.value
            if value is None:
                continue
            resolved = _resolve_class_set(value, aliases, known)
            if not resolved:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    before = len(aliases[target.id])
                    aliases[target.id] |= resolved
                    if len(aliases[target.id]) > before:
                        changed = True
    return aliases


@dataclass(frozen=True)
class _FileInfo:
    mx_modules: frozenset[str]
    mlx_roots: frozenset[str]
    op_aliases: dict[str, str]
    quant_aliases: dict[str, set[str]]
    format_aliases: dict[str, set[str]]
    dense_aliases: dict[str, set[str]]
    self_weight_classes: frozenset[str]
    pred_facts: dict[str, tuple[int, frozenset[str]]]
    class_layout_guards: dict[tuple[str, str], tuple[str, frozenset[str]]]


def _mlx_op_name(
    func: ast.AST,
    mx_modules: frozenset[str],
    mlx_roots: frozenset[str],
    op_aliases: dict[str, str],
) -> str | None:
    if isinstance(func, ast.Name):
        return op_aliases.get(func.id)
    if isinstance(func, ast.Attribute) and func.attr in _MLX_OP_TO_RULE:
        recv = _expr_key(func.value)
        if recv is None:
            return None
        if recv in mx_modules:
            return func.attr
        for root in mlx_roots:
            if recv == f"{root}.core":
                return func.attr
    return None


@dataclass
class _FactsDelta:
    recv_facts: dict[str, set[str]]
    seq_guards: dict[str, set[str]]

    @classmethod
    def empty(cls) -> _FactsDelta:
        return cls({}, {})

    def merge(self, other: _FactsDelta) -> _FactsDelta:
        recv: dict[str, set[str]] = {k: set(v) for k, v in self.recv_facts.items()}
        for k, v in other.recv_facts.items():
            recv.setdefault(k, set()).update(v)
        seq: dict[str, set[str]] = {k: set(v) for k, v in self.seq_guards.items()}
        for k, v in other.seq_guards.items():
            seq.setdefault(k, set()).update(v)
        return _FactsDelta(recv, seq)


@dataclass
class _GuardEnv:
    recv_facts: dict[str, set[str]]
    seq_guards: dict[str, set[str]]
    layout_vars: dict[str, tuple[str, frozenset[str]]]
    seq_aliases: dict[str, str]
    var_reads_module_attr: dict[str, bool]
    var_unguarded_attr: dict[str, bool]

    @classmethod
    def empty(cls) -> _GuardEnv:
        return cls({}, {}, {}, {}, {}, {})

    def copy(self) -> _GuardEnv:
        return _GuardEnv(
            recv_facts={k: set(v) for k, v in self.recv_facts.items()},
            seq_guards={k: set(v) for k, v in self.seq_guards.items()},
            layout_vars=dict(self.layout_vars),
            seq_aliases=dict(self.seq_aliases),
            var_reads_module_attr=dict(self.var_reads_module_attr),
            var_unguarded_attr=dict(self.var_unguarded_attr),
        )

    def apply(self, delta: _FactsDelta) -> _GuardEnv:
        out = self.copy()
        for k, v in delta.recv_facts.items():
            out.recv_facts.setdefault(k, set()).update(v)
        for k, v in delta.seq_guards.items():
            out.seq_guards.setdefault(k, set()).update(v)
            aliased = out.seq_aliases.get(k)
            if aliased is not None:
                out.seq_guards.setdefault(aliased, set()).update(v)
        return out

    def invalidate(self, var_name: str) -> None:
        prefix = f"{var_name}."
        for key in list(self.recv_facts):
            if key == var_name or key.startswith(prefix):
                del self.recv_facts[key]
        for key in list(self.seq_guards):
            if key == var_name or key.startswith(prefix):
                del self.seq_guards[key]
        for key, (seq_attr, _) in list(self.layout_vars.items()):
            seq_key = f"self.{seq_attr}"
            if (
                key == var_name
                or key.startswith(prefix)
                or seq_key == var_name
                or seq_key.startswith(prefix)
            ):
                del self.layout_vars[key]
        for key, target in list(self.seq_aliases.items()):
            if (
                key == var_name
                or key.startswith(prefix)
                or target == var_name
                or target.startswith(prefix)
            ):
                del self.seq_aliases[key]
        self.var_reads_module_attr.pop(var_name, None)
        self.var_unguarded_attr.pop(var_name, None)

    def merge_taint_from(self, other: _GuardEnv) -> None:
        for var_name, flag in other.var_unguarded_attr.items():
            if flag:
                self.var_unguarded_attr[var_name] = True
        for var_name, flag in other.var_reads_module_attr.items():
            if flag:
                self.var_reads_module_attr[var_name] = True

    def has_attr_guard(self, recv_key: str, attr: str) -> bool:
        facts = self.recv_facts.get(recv_key)
        if not facts:
            return False
        if attr in facts:
            return True
        if facts & {"affine", "quantized", "non_format"}:
            return True
        return attr == "weight" and "dense" in facts

    def has_op_guard(self, recv_key: str | None = None) -> bool:
        if recv_key is not None:
            facts = self.recv_facts.get(recv_key)
            return bool(facts and (facts & {"affine", "quantized", "non_format"}))
        return any(
            f & {"affine", "quantized", "non_format"}
            for f in (*self.recv_facts.values(), *self.seq_guards.values())
        )


def _is_none_const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_none_or_falsy_var_test(test: ast.expr, var_name: str) -> bool:
    if (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and isinstance(test.operand, ast.Name)
        and test.operand.id == var_name
    ):
        return True
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], (ast.Is, ast.Eq))
    ):
        left, right = test.left, test.comparators[0]
        for a, b in ((left, right), (right, left)):
            if isinstance(a, ast.Name) and a.id == var_name and _is_none_const(b):
                return True
    return False


def _taint_snapshot(env: _GuardEnv) -> tuple[frozenset[str], frozenset[str]]:
    return (
        frozenset(k for k, v in env.var_unguarded_attr.items() if v),
        frozenset(k for k, v in env.var_reads_module_attr.items() if v),
    )


def _is_str_const(node: ast.AST, value: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


def _getattr_target(node: ast.AST) -> tuple[str | None, str | None, bool]:
    """If ``node`` is ``getattr(recv, "attr"[, default])``, return
    ``(recv_key, attr, has_default)``."""
    if not isinstance(node, ast.Call) or _tail_name(node.func) != "getattr":
        return None, None, False
    if len(node.args) < 2:
        return None, None, False
    attr_node = node.args[1]
    if not (isinstance(attr_node, ast.Constant) and isinstance(attr_node.value, str)):
        return None, None, False
    return _expr_key(node.args[0]), attr_node.value, len(node.args) >= 3


def _extract_facts(
    expr: ast.AST,
    info: _FileInfo,
    env: _GuardEnv,
) -> tuple[_FactsDelta, _FactsDelta]:
    """Return ``(pos_delta, neg_delta)`` established when ``expr`` is True / False."""
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        pos, neg = _extract_facts(expr.operand, info, env)
        return neg, pos

    if isinstance(expr, ast.BoolOp):
        if isinstance(expr.op, ast.And):
            pos = _FactsDelta.empty()
            cur = env
            for val in expr.values:
                v_pos, _ = _extract_facts(val, info, cur)
                pos = pos.merge(v_pos)
                cur = cur.apply(v_pos)
            return pos, _FactsDelta.empty()
        if isinstance(expr.op, ast.Or):
            neg = _FactsDelta.empty()
            cur = env
            for val in expr.values:
                _, v_neg = _extract_facts(val, info, cur)
                neg = neg.merge(v_neg)
                cur = cur.apply(v_neg)
            return _FactsDelta.empty(), neg

    if isinstance(expr, ast.Name):
        # Local layout guard variable (e.g. `layout = self._uniform_affine_layout()`).
        layout_info = env.layout_vars.get(expr.id)
        if layout_info is not None:
            seq_attr, facts = layout_info
            seq_key = f"self.{seq_attr}"
            return (
                _FactsDelta({}, {seq_key: set(facts)}),
                _FactsDelta.empty(),
            )
        return _FactsDelta.empty(), _FactsDelta.empty()

    if isinstance(expr, ast.Attribute) and expr.attr == "_stock":
        recv = _expr_key(expr.value)
        if recv is not None:
            return (
                _FactsDelta({recv: {"affine"}}, {}),
                _FactsDelta.empty(),
            )
        return _FactsDelta.empty(), _FactsDelta.empty()

    if isinstance(expr, ast.Call):
        fname = _tail_name(expr.func)
        if fname == "hasattr" and len(expr.args) == 2:
            recv = _expr_key(expr.args[0])
            attr_node = expr.args[1]
            if (
                recv is not None
                and isinstance(attr_node, ast.Constant)
                and isinstance(attr_node.value, str)
            ):
                if attr_node.value in _TENSOR_ATTR_TO_RULE:
                    return (
                        _FactsDelta({recv: {attr_node.value}}, {}),
                        _FactsDelta.empty(),
                    )
                if attr_node.value == "fmt":
                    return (
                        _FactsDelta.empty(),
                        _FactsDelta({recv: {"non_format"}}, {}),
                    )
        elif fname == "isinstance" and len(expr.args) == 2:
            arg0, arg1 = expr.args
            # isinstance(getattr(r, "weight", None), ...)
            g_recv, g_attr, _ = _getattr_target(arg0)
            if g_recv is not None and g_attr in _TENSOR_ATTR_TO_RULE:
                return _FactsDelta({g_recv: {g_attr}}, {}), _FactsDelta.empty()
            recv = _expr_key(arg0)
            if recv is not None:
                q_cls = _resolve_class_set(
                    arg1, info.quant_aliases, _QUANTIZED_CLASS_NAMES
                )
                f_cls = _resolve_class_set(
                    arg1, info.format_aliases, _FORMAT_CLASS_NAMES
                )
                d_cls = _resolve_class_set(arg1, info.dense_aliases, _DENSE_LAYER_NAMES)
                if q_cls and not f_cls:
                    # Soundness: neg is EMPTY (FormatLinear lands in else).
                    return (
                        _FactsDelta({recv: {"quantized"}}, {}),
                        _FactsDelta.empty(),
                    )
                if f_cls and not q_cls:
                    return (
                        _FactsDelta.empty(),
                        _FactsDelta({recv: {"non_format"}}, {}),
                    )
                if d_cls and not q_cls and not f_cls:
                    return (
                        _FactsDelta({recv: {"dense"}}, {}),
                        _FactsDelta.empty(),
                    )
        elif fname == "all" and len(expr.args) == 1:
            gen = expr.args[0]
            if isinstance(gen, ast.GeneratorExp) and len(gen.generators) == 1:
                comp = gen.generators[0]
                if (
                    isinstance(comp.target, ast.Name)
                    and isinstance(gen.elt, ast.Call)
                    and isinstance(gen.elt.func, ast.Name)
                ):
                    pred = info.pred_facts.get(gen.elt.func.id)
                    seq_key = _expr_key(comp.iter)
                    if pred is not None and seq_key is not None:
                        param_idx, facts = pred
                        if (
                            param_idx < len(gen.elt.args)
                            and isinstance(gen.elt.args[param_idx], ast.Name)
                            and gen.elt.args[param_idx].id == comp.target.id
                        ):
                            return (
                                _FactsDelta({}, {seq_key: set(facts)}),
                                _FactsDelta.empty(),
                            )
        elif isinstance(expr.func, ast.Name) and expr.func.id in info.pred_facts:
            param_idx, facts = info.pred_facts[expr.func.id]
            if param_idx < len(expr.args):
                recv = _expr_key(expr.args[param_idx])
                if recv is not None:
                    return (
                        _FactsDelta({recv: set(facts)}, {}),
                        _FactsDelta.empty(),
                    )
        return _FactsDelta.empty(), _FactsDelta.empty()

    if isinstance(expr, ast.Compare) and len(expr.ops) == 1:
        op = expr.ops[0]
        left, right = expr.left, expr.comparators[0]

        # "attr" in r / "attr" not in r
        if isinstance(op, (ast.In, ast.NotIn)):
            if (
                isinstance(left, ast.Constant)
                and isinstance(left.value, str)
                and left.value in _TENSOR_ATTR_TO_RULE
            ):
                recv = _expr_key(right)
                if recv is not None:
                    delta = _FactsDelta({recv: {left.value}}, {})
                    if isinstance(op, ast.In):
                        return delta, _FactsDelta.empty()
                    return _FactsDelta.empty(), delta
            return _FactsDelta.empty(), _FactsDelta.empty()

        # Check both (a op b) and (b op a) for symmetric ops.
        if isinstance(op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
            is_pos = isinstance(op, (ast.Is, ast.Eq))
            for a, b in ((left, right), (right, left)):
                # getattr(r, attr, None) is [not] None
                g_recv, g_attr, _ = _getattr_target(a)
                if g_recv is not None and _is_none_const(b):
                    if g_attr in _TENSOR_ATTR_TO_RULE:
                        delta = _FactsDelta({g_recv: {g_attr}}, {})
                        return (
                            (_FactsDelta.empty(), delta)
                            if is_pos
                            else (delta, _FactsDelta.empty())
                        )
                    if g_attr == "fmt":
                        delta = _FactsDelta({g_recv: {"non_format"}}, {})
                        return (
                            (delta, _FactsDelta.empty())
                            if is_pos
                            else (_FactsDelta.empty(), delta)
                        )
                # getattr(r, "mode", None) == "affine" or r.mode == "affine"
                if _is_str_const(b, "affine"):
                    mode_recv: str | None = None
                    if g_recv is not None and g_attr == "mode":
                        mode_recv = g_recv
                    elif isinstance(a, ast.Attribute) and a.attr == "mode":
                        mode_recv = _expr_key(a.value)
                    if mode_recv is not None:
                        delta = _FactsDelta({mode_recv: {"affine"}}, {})
                        return (
                            (delta, _FactsDelta.empty())
                            if is_pos
                            else (_FactsDelta.empty(), delta)
                        )
                # type(r) is nn.Linear
                if (
                    isinstance(a, ast.Call)
                    and _tail_name(a.func) == "type"
                    and len(a.args) == 1
                ):
                    recv = _expr_key(a.args[0])
                    if recv is not None:
                        d_cls = _resolve_class_set(
                            b, info.dense_aliases, _DENSE_LAYER_NAMES
                        )
                        q_cls = _resolve_class_set(
                            b, info.quant_aliases, _QUANTIZED_CLASS_NAMES
                        )
                        f_cls = _resolve_class_set(
                            b, info.format_aliases, _FORMAT_CLASS_NAMES
                        )
                        if d_cls and not q_cls and not f_cls and is_pos:
                            return (
                                _FactsDelta({recv: {"dense"}}, {}),
                                _FactsDelta.empty(),
                            )

    return _FactsDelta.empty(), _FactsDelta.empty()


def _always_exits(stmts: Sequence[ast.stmt]) -> bool:
    if not stmts:
        return False
    last = stmts[-1]
    if isinstance(last, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
        return True
    if isinstance(last, ast.If) and last.orelse:
        return _always_exits(last.body) and _always_exits(last.orelse)
    return False


def _is_falsy_return(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Return):
        return False
    if stmt.value is None:
        return True
    return isinstance(stmt.value, ast.Constant) and not bool(stmt.value.value)


def _has_non_falsy_return(node: ast.AST) -> bool:
    return any(
        isinstance(s, ast.Return) and not _is_falsy_return(s) for s in ast.walk(node)
    )


def _qualifying_seq(env: _GuardEnv) -> tuple[str, frozenset[str]] | None:
    for seq_key, facts in env.seq_guards.items():
        if seq_key.startswith("self.") and (
            facts & {"affine", "quantized", "non_format"}
            or {"weight", "scales", "biases"}.issubset(facts)
        ):
            return seq_key.split(".", 1)[1], frozenset(facts)
    return None


def _analyze_predicate_fn(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    base_info: _FileInfo,
) -> tuple[int, frozenset[str]] | None:
    params = [a.arg for a in fn.args.posonlyargs] + [a.arg for a in fn.args.args]
    if not params:
        return None
    param_set = set(params)
    env = _GuardEnv.empty()
    body = fn.body
    start = 1 if body and _is_docstring_stmt(body[0]) else 0
    for stmt in body[start:]:
        assigned = _assigned_names([stmt])
        if any(
            name in param_set or any(name.startswith(f"{p}.") for p in param_set)
            for name in assigned
        ):
            return None
        if isinstance(stmt, ast.Return):
            if _is_falsy_return(stmt) or stmt.value is None:
                return None
            pos, _ = _extract_facts(stmt.value, base_info, env)
            env = env.apply(pos)
            for idx, pname in enumerate(params):
                facts = env.recv_facts.get(pname)
                if facts:
                    return idx, frozenset(facts)
            return None
        if (
            isinstance(stmt, ast.If)
            and _always_exits(stmt.body)
            and not stmt.orelse
            and not _has_non_falsy_return(stmt)
        ):
            _, neg = _extract_facts(stmt.test, base_info, env)
            env = env.apply(neg)
            continue
        if _has_non_falsy_return(stmt):
            return None
        for name in assigned:
            env.invalidate(name)
    return None


def _analyze_layout_method(
    cls_name: str,
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    info: _FileInfo,
) -> tuple[str, frozenset[str]] | None:
    """Detect class methods like ``ShardedEmbedding._uniform_affine_layout``
    that return False/None unless every element of ``self.<seq>`` passes an
    affine guard predicate."""
    params = [a.arg for a in fn.args.posonlyargs] + [a.arg for a in fn.args.args]
    if not params or params[0] != "self":
        return None
    param_set = set(params)
    env = _GuardEnv.empty()
    body = fn.body
    start = 1 if body and _is_docstring_stmt(body[0]) else 0
    saw_truthy_return = False
    for stmt in body[start:]:
        assigned = _assigned_names([stmt])
        if assigned & param_set:
            return None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            t = stmt.targets[0]
            if (
                isinstance(t, ast.Name)
                and isinstance(stmt.value, ast.Attribute)
                and isinstance(stmt.value.value, ast.Name)
                and stmt.value.value.id == "self"
            ):
                env.invalidate(t.id)
                env.seq_aliases[t.id] = f"self.{stmt.value.attr}"
            else:
                for name in assigned:
                    env.invalidate(name)
        elif (
            isinstance(stmt, ast.If)
            and _always_exits(stmt.body)
            and not stmt.orelse
            and not _has_non_falsy_return(stmt)
        ):
            for name in _named_expr_targets(stmt.test):
                env.invalidate(name)
            _, neg = _extract_facts(stmt.test, info, env)
            env = env.apply(neg)
        elif isinstance(stmt, ast.Return):
            if not _is_falsy_return(stmt):
                if _qualifying_seq(env) is None:
                    return None
                saw_truthy_return = True
        else:
            for name in assigned:
                env.invalidate(name)
            if _has_non_falsy_return(stmt):
                if _qualifying_seq(env) is None:
                    return None
                saw_truthy_return = True
    if not saw_truthy_return:
        return None
    return _qualifying_seq(env)


def _collect_file_info(tree: ast.Module) -> _FileInfo:
    mx_modules: set[str] = set()
    mlx_roots: set[str] = set()
    op_aliases: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                if alias.name == "mlx.core":
                    mx_modules.add(local)
                elif alias.name == "mlx":
                    mlx_roots.add(local)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "mlx":
                for alias in node.names:
                    if alias.name == "core":
                        mx_modules.add(alias.asname or "core")
            elif mod == "mlx.core":
                for alias in node.names:
                    if alias.name in _MLX_OP_TO_RULE:
                        op_aliases[alias.asname or alias.name] = alias.name

    quant_aliases = _collect_class_aliases(tree, _QUANTIZED_CLASS_NAMES)
    format_aliases = _collect_class_aliases(tree, _FORMAT_CLASS_NAMES)
    dense_aliases = _collect_class_aliases(tree, _DENSE_LAYER_NAMES)

    # Identify classes that own plain self.weight without self.scales or quantized ops.
    self_weight_classes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        assigns_self_weight = False
        assigns_self_scales = False
        calls_quant_op = False
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Store):
                if isinstance(sub.value, ast.Name) and sub.value.id == "self":
                    if sub.attr == "weight":
                        assigns_self_weight = True
                    elif sub.attr == "scales":
                        assigns_self_scales = True
            elif isinstance(sub, ast.Call) and _mlx_op_name(
                sub.func, frozenset(mx_modules), frozenset(mlx_roots), op_aliases
            ):
                calls_quant_op = True
        if assigns_self_weight and not assigns_self_scales and not calls_quant_op:
            self_weight_classes.add(node.name)

    base_info = _FileInfo(
        mx_modules=frozenset(mx_modules),
        mlx_roots=frozenset(mlx_roots),
        op_aliases=op_aliases,
        quant_aliases=quant_aliases,
        format_aliases=format_aliases,
        dense_aliases=dense_aliases,
        self_weight_classes=frozenset(self_weight_classes),
        pred_facts={},
        class_layout_guards={},
    )

    pred_facts: dict[str, tuple[int, frozenset[str]]] = {}
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parsed = _analyze_predicate_fn(stmt, base_info)
            if parsed is not None:
                pred_facts[stmt.name] = parsed

    mid_info = _FileInfo(
        mx_modules=base_info.mx_modules,
        mlx_roots=base_info.mlx_roots,
        op_aliases=base_info.op_aliases,
        quant_aliases=base_info.quant_aliases,
        format_aliases=base_info.format_aliases,
        dense_aliases=base_info.dense_aliases,
        self_weight_classes=base_info.self_weight_classes,
        pred_facts=pred_facts,
        class_layout_guards={},
    )

    class_layout_guards: dict[tuple[str, str], tuple[str, frozenset[str]]] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef):
            for item in stmt.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    guard = _analyze_layout_method(stmt.name, item, mid_info)
                    if guard is not None:
                        class_layout_guards[(stmt.name, item.name)] = guard

    return _FileInfo(
        mx_modules=mid_info.mx_modules,
        mlx_roots=mid_info.mlx_roots,
        op_aliases=mid_info.op_aliases,
        quant_aliases=mid_info.quant_aliases,
        format_aliases=mid_info.format_aliases,
        dense_aliases=mid_info.dense_aliases,
        self_weight_classes=mid_info.self_weight_classes,
        pred_facts=pred_facts,
        class_layout_guards=class_layout_guards,
    )


def _node_reads_module_attr(node: ast.AST, var_reads: dict[str, bool]) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr in _MODULE_TENSOR_ATTRS:
            return True
        if isinstance(sub, ast.Subscript):
            sl = sub.slice
            if isinstance(sl, ast.Constant) and sl.value in _MODULE_TENSOR_ATTRS:
                return True
        _, g_attr, _ = _getattr_target(sub)
        if g_attr in _MODULE_TENSOR_ATTRS:
            return True
        if isinstance(sub, ast.Name) and var_reads.get(sub.id, False):
            return True
    return False


def _node_reads_unguarded_attr(node: ast.AST, env: _GuardEnv) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr in _TENSOR_ATTR_TO_RULE:
            recv = _expr_key(sub.value)
            if recv is None or not env.has_op_guard(recv):
                return True
        elif isinstance(sub, ast.Subscript):
            sl = sub.slice
            if (
                isinstance(sl, ast.Constant)
                and isinstance(sl.value, str)
                and sl.value in _TENSOR_ATTR_TO_RULE
                and _tail_name(sub.value) not in _PLAIN_DICT_NAMES
            ):
                recv = _expr_key(sub.value)
                if recv is None or not env.has_op_guard(recv):
                    return True
        elif isinstance(sub, ast.Call):
            g_recv, g_attr, _ = _getattr_target(sub)
            if g_attr in _TENSOR_ATTR_TO_RULE and (
                g_recv is None or not env.has_op_guard(g_recv)
            ):
                return True
        elif isinstance(sub, ast.Name) and env.var_unguarded_attr.get(sub.id, False):
            return True
    return False


def _call_has_unguarded_arg(call: ast.Call, env: _GuardEnv) -> bool:
    return any(
        _node_reads_unguarded_attr(arg, env)
        for arg in [*call.args, *(kw.value for kw in call.keywords)]
    )


def _call_references_kv_cache(call: ast.Call) -> bool:
    for arg in [*call.args, *(kw.value for kw in call.keywords)]:
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Name) and _is_kv_cache_name(sub.id):
                return True
            if isinstance(sub, ast.Attribute) and _is_kv_cache_name(sub.attr):
                return True
    return False


def _call_has_literal_affine_mode(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "mode" and _is_str_const(kw.value, "affine"):
            return True
    return False


def _call_has_guarded_arg(call: ast.Call, env: _GuardEnv) -> bool:
    for arg in [*call.args, *(kw.value for kw in call.keywords)]:
        for sub in ast.walk(arg):
            if (
                isinstance(sub, ast.Attribute) and sub.attr in _MODULE_TENSOR_ATTRS
            ) or isinstance(sub, ast.Subscript):
                recv = _expr_key(sub.value)
                if recv is not None and env.has_op_guard(recv):
                    return True
    return False


def _is_type_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and _tail_name(node.func) == "type"
        and len(node.args) == 1
    )


def _is_class_attr(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "__class__"


def _is_type_name_expr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr in {"__name__", "__qualname__"}
        and (_is_type_call(node.value) or _is_class_attr(node.value))
    )


class _AuditVisitor:
    """Flow-sensitive AST scanner implementing the 7 format-reader rules."""

    def __init__(self, path: str, info: _FileInfo) -> None:
        self.path = path
        self.info = info
        self.scope_stack: list[str] = []
        self.class_stack: list[str] = []
        self.raw_findings: list[_RawFinding] = []
        self._taint_sinks: list[_GuardEnv] = []

    def _scope(self) -> str:
        return ".".join(self.scope_stack) if self.scope_stack else "<module>"

    def _emit(self, node: ast.AST, rule: str, symbol: str, message: str) -> None:
        line = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        end_line = getattr(node, "end_lineno", line) or line
        self.raw_findings.append(
            _RawFinding(
                finding=Finding(
                    path=self.path,
                    line=line,
                    col=col,
                    rule=rule,
                    scope=self._scope(),
                    symbol=symbol,
                    message=message,
                ),
                end_line=end_line,
            )
        )

    def scan(self, tree: ast.Module) -> list[_RawFinding]:
        self._visit_block(tree.body, _GuardEnv.empty(), skip_docstring=True)
        return self.raw_findings

    def _visit_block(
        self,
        stmts: Sequence[ast.stmt],
        env: _GuardEnv,
        *,
        skip_docstring: bool = False,
    ) -> _GuardEnv:
        start = 1 if skip_docstring and stmts and _is_docstring_stmt(stmts[0]) else 0
        cur = env
        for sink in self._taint_sinks:
            sink.merge_taint_from(cur)
        for stmt in stmts[start:]:
            cur = self._visit_stmt(stmt, cur)
            for sink in self._taint_sinks:
                sink.merge_taint_from(cur)
        return cur

    def _layout_promise(
        self, value: ast.AST | None
    ) -> tuple[str, frozenset[str]] | None:
        if (
            value is not None
            and self.class_stack
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "self"
        ):
            return self.info.class_layout_guards.get(
                (self.class_stack[-1], value.func.attr)
            )
        return None

    def _subscript_elem_facts(
        self, value: ast.AST | None, env: _GuardEnv
    ) -> set[str] | None:
        if value is not None and isinstance(value, ast.Subscript):
            seq_key = _expr_key(value.value)
            if seq_key is not None:
                facts = env.seq_guards.get(seq_key)
                if facts is None and seq_key in env.seq_aliases:
                    facts = env.seq_guards.get(env.seq_aliases[seq_key])
                if facts:
                    return set(facts)
        return None

    def _bind_targets(
        self,
        targets: Sequence[ast.AST],
        value: ast.AST | None,
        env: _GuardEnv,
    ) -> _GuardEnv:
        out = env.copy()
        rhs_reads_attr = (
            _node_reads_module_attr(value, out.var_reads_module_attr)
            if value is not None
            else False
        )
        rhs_reads_unguarded_attr = (
            _node_reads_unguarded_attr(value, out) if value is not None else False
        )
        layout_promise = self._layout_promise(value)
        elem_facts = self._subscript_elem_facts(value, out)
        rhs_key = _expr_key(value) if value is not None else None

        for target in targets:
            self._bind_one_target(
                target,
                out,
                rhs_reads_attr=rhs_reads_attr,
                rhs_reads_unguarded_attr=rhs_reads_unguarded_attr,
                layout_promise=layout_promise,
                elem_facts=elem_facts,
                rhs_key=rhs_key,
            )
        return out

    def _bind_one_target(
        self,
        target: ast.AST,
        env: _GuardEnv,
        *,
        rhs_reads_attr: bool,
        rhs_reads_unguarded_attr: bool,
        layout_promise: tuple[str, frozenset[str]] | None,
        elem_facts: set[str] | None,
        rhs_key: str | None,
    ) -> None:
        if isinstance(target, ast.Name):
            env.invalidate(target.id)
            if rhs_reads_attr:
                env.var_reads_module_attr[target.id] = True
            if rhs_reads_unguarded_attr:
                env.var_unguarded_attr[target.id] = True
            if layout_promise is not None:
                env.layout_vars[target.id] = layout_promise
            if elem_facts:
                env.recv_facts[target.id] = set(elem_facts)
            if rhs_key is not None:
                if rhs_key in env.seq_guards:
                    env.seq_guards[target.id] = set(env.seq_guards[rhs_key])
                if rhs_key.startswith("self."):
                    env.seq_aliases[target.id] = rhs_key
                if rhs_key in env.recv_facts:
                    env.recv_facts[target.id] = set(env.recv_facts[rhs_key])
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._bind_one_target(
                    elt,
                    env,
                    rhs_reads_attr=rhs_reads_attr,
                    rhs_reads_unguarded_attr=rhs_reads_unguarded_attr,
                    layout_promise=None,
                    elem_facts=None,
                    rhs_key=None,
                )
        elif isinstance(target, ast.Starred):
            self._bind_one_target(
                target.value,
                env,
                rhs_reads_attr=rhs_reads_attr,
                rhs_reads_unguarded_attr=rhs_reads_unguarded_attr,
                layout_promise=None,
                elem_facts=None,
                rhs_key=None,
            )
        elif isinstance(target, ast.Attribute):
            # Non-self store to .scales / .biases (e.g. raon1_mini_model.py:947).
            self._visit_expr(target.value, env)
            if target.attr in {"scales", "biases"}:
                recv_key = _expr_key(target.value)
                if recv_key != "self" and not (
                    recv_key is not None and env.has_attr_guard(recv_key, target.attr)
                ):
                    self._emit(
                        target,
                        _TENSOR_ATTR_TO_RULE[target.attr],
                        f"{recv_key or '<expr>'}.{target.attr}",
                        f"unguarded write to '.{target.attr}' on possible format layer",
                    )
            key = _expr_key(target)
            if key is not None:
                env.invalidate(key)
        elif isinstance(target, ast.Subscript):
            self._visit_expr(target.value, env)
            self._visit_expr(target.slice, env)

    def _visit_stmt(self, stmt: ast.stmt, env: _GuardEnv) -> _GuardEnv:
        if isinstance(stmt, ast.ClassDef):
            cur = env.copy()
            for dec in stmt.decorator_list:
                self._visit_expr(dec, cur)
            for base in stmt.bases:
                self._visit_expr(base, cur)
            for kw in stmt.keywords:
                self._visit_expr(kw.value, cur)
            saved_sinks = self._taint_sinks
            self._taint_sinks = []
            self.scope_stack.append(stmt.name)
            self.class_stack.append(stmt.name)
            try:
                self._visit_block(stmt.body, _GuardEnv.empty(), skip_docstring=True)
            finally:
                self.class_stack.pop()
                self.scope_stack.pop()
                self._taint_sinks = saved_sinks
            cur.invalidate(stmt.name)
            return cur

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cur = env.copy()
            for dec in stmt.decorator_list:
                self._visit_expr(dec, cur)
            for default in stmt.args.defaults:
                self._visit_expr(default, cur)
            for kw_default in stmt.args.kw_defaults:
                if kw_default is not None:
                    self._visit_expr(kw_default, cur)
            saved_sinks = self._taint_sinks
            self._taint_sinks = []
            self.scope_stack.append(stmt.name)
            # Soundness: entering a nested def resets local variable guards.
            try:
                self._visit_block(stmt.body, _GuardEnv.empty(), skip_docstring=True)
            finally:
                self.scope_stack.pop()
                self._taint_sinks = saved_sinks
            cur.invalidate(stmt.name)
            return cur

        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            out = env.copy()
            for alias in stmt.names:
                if alias.asname:
                    out.invalidate(alias.asname)
                elif alias.name != "*":
                    out.invalidate(alias.name.split(".", 1)[0])
            return out

        if isinstance(stmt, ast.Assign):
            cur = env.copy()
            self._visit_expr(stmt.value, cur)
            return self._bind_targets(stmt.targets, stmt.value, cur)

        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                cur = env.copy()
                self._visit_expr(stmt.value, cur)
                return self._bind_targets((stmt.target,), stmt.value, cur)
            return env

        if isinstance(stmt, ast.AugAssign):
            cur = env.copy()
            self._visit_expr(stmt.value, cur)
            if (
                isinstance(stmt.target, ast.Attribute)
                and stmt.target.attr in _TENSOR_ATTR_TO_RULE
            ):
                self._check_attr_read(stmt.target, cur)
            elif isinstance(stmt.target, ast.Subscript):
                self._check_subscript_read(stmt.target, cur)
            self._visit_expr(stmt.target, cur)
            reads_attr = _node_reads_module_attr(
                stmt.value, cur.var_reads_module_attr
            ) or _node_reads_module_attr(stmt.target, cur.var_reads_module_attr)
            reads_unguarded = _node_reads_unguarded_attr(
                stmt.value, cur
            ) or _node_reads_unguarded_attr(stmt.target, cur)
            key = _expr_key(stmt.target)
            if key is not None:
                cur.invalidate(key)
                if isinstance(stmt.target, ast.Name):
                    if reads_attr:
                        cur.var_reads_module_attr[stmt.target.id] = True
                    if reads_unguarded:
                        cur.var_unguarded_attr[stmt.target.id] = True
            return cur

        if isinstance(stmt, ast.Assert):
            cur = env.copy()
            self._visit_expr(stmt.test, cur)
            if stmt.msg is not None:
                self._visit_expr(stmt.msg, cur.copy())
            pos, _ = _extract_facts(stmt.test, self.info, cur)
            return cur.apply(pos)

        if isinstance(stmt, ast.If):
            test_env = env.copy()
            self._visit_expr(stmt.test, test_env)
            pos, neg = _extract_facts(stmt.test, self.info, test_env)
            then_env = self._visit_block(stmt.body, test_env.apply(pos))
            else_env = self._visit_block(stmt.orelse, test_env.apply(neg))
            then_exits = _always_exits(stmt.body)
            else_exits = _always_exits(stmt.orelse)
            if then_exits and not stmt.orelse:
                return test_env.apply(neg)
            if then_exits and not else_exits:
                return else_env
            if else_exits and not then_exits:
                return then_env
            # Neither side guaranteed to exit: conservatively invalidate vars
            # reassigned inside either branch while preserving taint.
            merged = test_env.copy()
            for name in _assigned_names(stmt.body) | _assigned_names(stmt.orelse):
                merged.invalidate(name)
            merged.merge_taint_from(test_env)
            merged.merge_taint_from(then_env)
            merged.merge_taint_from(else_env)
            # Preserve layout_vars newly initialized in an `if layout is None:`
            # block (or established identically on both branches).
            for var_name, promise in then_env.layout_vars.items():
                if var_name not in test_env.layout_vars and (
                    (
                        var_name in else_env.layout_vars
                        and else_env.layout_vars[var_name] == promise
                    )
                    or (
                        not stmt.orelse
                        and _is_none_or_falsy_var_test(stmt.test, var_name)
                    )
                ):
                    merged.layout_vars[var_name] = promise
            return merged

        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            iter_env = env.copy()
            self._visit_expr(stmt.iter, iter_env)
            iter_reads_attr = _node_reads_module_attr(
                stmt.iter, iter_env.var_reads_module_attr
            )
            iter_reads_unguarded = _node_reads_unguarded_attr(stmt.iter, iter_env)
            loop_env = iter_env.copy()
            iter_key = _expr_key(stmt.iter)
            elem_facts: set[str] | None = None
            if iter_key is not None:
                facts = loop_env.seq_guards.get(iter_key)
                if facts is None and iter_key in loop_env.seq_aliases:
                    facts = loop_env.seq_guards.get(loop_env.seq_aliases[iter_key])
                if facts:
                    elem_facts = set(facts)
            body_assigned = _assigned_names(stmt.body)
            for name in body_assigned:
                loop_env.invalidate(name)
            loop_env.merge_taint_from(iter_env)
            self._bind_one_target(
                stmt.target,
                loop_env,
                rhs_reads_attr=iter_reads_attr,
                rhs_reads_unguarded_attr=iter_reads_unguarded,
                layout_promise=None,
                elem_facts=elem_facts,
                rhs_key=None,
            )
            while True:
                mark = len(self.raw_findings)
                before_taint = _taint_snapshot(loop_env)
                body_taint = _GuardEnv.empty()
                self._taint_sinks.append(body_taint)
                try:
                    body_env = self._visit_block(stmt.body, loop_env.copy())
                finally:
                    self._taint_sinks.pop()
                body_taint.merge_taint_from(body_env)
                loop_env.merge_taint_from(body_taint)
                if _taint_snapshot(loop_env) == before_taint:
                    break
                del self.raw_findings[mark:]
            orelse_base = iter_env.copy()
            for name in body_assigned | _node_assigned_names(stmt.target):
                orelse_base.invalidate(name)
            orelse_base.merge_taint_from(iter_env)
            orelse_base.merge_taint_from(loop_env)
            orelse_base.merge_taint_from(body_taint)
            orelse_env: _GuardEnv | None = None
            if stmt.orelse:
                orelse_env = self._visit_block(stmt.orelse, orelse_base)
            out = iter_env.copy()
            for name in _assigned_names([stmt]):
                out.invalidate(name)
            out.merge_taint_from(iter_env)
            out.merge_taint_from(loop_env)
            out.merge_taint_from(body_taint)
            if orelse_env is not None:
                out.merge_taint_from(orelse_env)
            return out

        if isinstance(stmt, ast.While):
            while_env = env.copy()
            body_and_test_assigned = _assigned_names(stmt.body) | _node_assigned_names(
                stmt.test
            )
            for name in body_and_test_assigned:
                while_env.invalidate(name)
            while_env.merge_taint_from(env)
            while True:
                mark = len(self.raw_findings)
                before_taint = _taint_snapshot(while_env)
                test_env = while_env.copy()
                self._visit_expr(stmt.test, test_env)
                pos, _ = _extract_facts(stmt.test, self.info, test_env)
                body_taint = _GuardEnv.empty()
                body_taint.merge_taint_from(test_env)
                self._taint_sinks.append(body_taint)
                try:
                    body_env = self._visit_block(stmt.body, test_env.apply(pos))
                finally:
                    self._taint_sinks.pop()
                body_taint.merge_taint_from(body_env)
                while_env.merge_taint_from(body_taint)
                if _taint_snapshot(while_env) == before_taint:
                    break
                del self.raw_findings[mark:]
            orelse_base = env.copy()
            for name in body_and_test_assigned:
                orelse_base.invalidate(name)
            orelse_base.merge_taint_from(env)
            orelse_base.merge_taint_from(while_env)
            orelse_base.merge_taint_from(body_taint)
            orelse_env: _GuardEnv | None = None
            if stmt.orelse:
                orelse_env = self._visit_block(stmt.orelse, orelse_base)
            out = env.copy()
            for name in _assigned_names([stmt]):
                out.invalidate(name)
            out.merge_taint_from(env)
            out.merge_taint_from(while_env)
            out.merge_taint_from(body_taint)
            if orelse_env is not None:
                out.merge_taint_from(orelse_env)
            return out

        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            with_env = env.copy()
            for item in stmt.items:
                self._visit_expr(item.context_expr, with_env)
                if item.optional_vars is not None:
                    self._bind_one_target(
                        item.optional_vars,
                        with_env,
                        rhs_reads_attr=_node_reads_module_attr(
                            item.context_expr, with_env.var_reads_module_attr
                        ),
                        rhs_reads_unguarded_attr=_node_reads_unguarded_attr(
                            item.context_expr, with_env
                        ),
                        layout_promise=None,
                        elem_facts=None,
                        rhs_key=None,
                    )
            body_taint = _GuardEnv.empty()
            self._taint_sinks.append(body_taint)
            try:
                body_env = self._visit_block(stmt.body, with_env)
            finally:
                self._taint_sinks.pop()
            body_taint.merge_taint_from(body_env)
            # Context managers may suppress exceptions raised mid-body, so guards
            # established inside `with` cannot be trusted after exit and taint
            # from any point in the block must survive.
            out = env.copy()
            for name in _assigned_names([stmt]):
                out.invalidate(name)
            out.merge_taint_from(env)
            out.merge_taint_from(with_env)
            out.merge_taint_from(body_taint)
            return out

        if isinstance(stmt, (ast.Try, ast.TryStar)):
            body_taint = _GuardEnv.empty()
            self._taint_sinks.append(body_taint)
            try:
                body_env = self._visit_block(stmt.body, env.copy())
            finally:
                self._taint_sinks.pop()
            body_taint.merge_taint_from(body_env)
            body_assigned = _assigned_names(stmt.body)

            handler_base = env.copy()
            for name in body_assigned:
                handler_base.invalidate(name)
            handler_base.merge_taint_from(env)
            handler_base.merge_taint_from(body_taint)

            h_taints: list[_GuardEnv] = []
            h_assigned: set[str] = set()
            is_trystar = isinstance(stmt, ast.TryStar)
            for handler in stmt.handlers:
                if handler.type is not None:
                    self._visit_expr(handler.type, handler_base)
                    h_assigned |= _node_assigned_names(handler.type)
                h_env = handler_base.copy()
                if handler.name:
                    h_env.invalidate(handler.name)
                    h_assigned.add(handler.name)
                h_body_assigned = _assigned_names(handler.body)
                h_assigned |= h_body_assigned
                h_taint = _GuardEnv.empty()
                self._taint_sinks.append(h_taint)
                try:
                    h_out = self._visit_block(handler.body, h_env)
                finally:
                    self._taint_sinks.pop()
                h_taint.merge_taint_from(h_out)
                h_taints.append(h_taint)
                if is_trystar:
                    # PEP 654: multiple except* handlers may run in sequence.
                    if handler.name:
                        handler_base.invalidate(handler.name)
                    for name in h_body_assigned:
                        handler_base.invalidate(name)
                    handler_base.merge_taint_from(h_taint)

            orelse_taint: _GuardEnv | None = None
            if stmt.orelse:
                orelse_taint = _GuardEnv.empty()
                self._taint_sinks.append(orelse_taint)
                try:
                    orelse_env = self._visit_block(stmt.orelse, body_env.copy())
                finally:
                    self._taint_sinks.pop()
                orelse_taint.merge_taint_from(orelse_env)

            final_taint: _GuardEnv | None = None
            if stmt.finalbody:
                final_base = env.copy()
                for name in body_assigned | h_assigned | _assigned_names(stmt.orelse):
                    final_base.invalidate(name)
                final_base.merge_taint_from(env)
                final_base.merge_taint_from(body_taint)
                for h_t in h_taints:
                    final_base.merge_taint_from(h_t)
                if orelse_taint is not None:
                    final_base.merge_taint_from(orelse_taint)
                final_taint = _GuardEnv.empty()
                self._taint_sinks.append(final_taint)
                try:
                    final_out = self._visit_block(stmt.finalbody, final_base)
                finally:
                    self._taint_sinks.pop()
                final_taint.merge_taint_from(final_out)

            out = env.copy()
            for name in _assigned_names([stmt]):
                out.invalidate(name)
            out.merge_taint_from(env)
            out.merge_taint_from(body_taint)
            for h_t in h_taints:
                out.merge_taint_from(h_t)
            if orelse_taint is not None:
                out.merge_taint_from(orelse_taint)
            if final_taint is not None:
                out.merge_taint_from(final_taint)
            return out

        if isinstance(stmt, ast.Return):
            cur = env.copy()
            if stmt.value is not None:
                self._visit_expr(stmt.value, cur)
            return cur

        if isinstance(stmt, ast.Expr):
            cur = env.copy()
            self._visit_expr(stmt.value, cur)
            return cur

        if isinstance(stmt, ast.Raise):
            cur = env.copy()
            if stmt.exc is not None:
                self._visit_expr(stmt.exc, cur)
            if stmt.cause is not None:
                self._visit_expr(stmt.cause, cur)
            return cur

        if isinstance(stmt, ast.Delete):
            out = env.copy()
            for target in stmt.targets:
                if not isinstance(target, ast.Name):
                    self._visit_expr(target, out)
                key = _expr_key(target)
                if key is not None:
                    out.invalidate(key)
            return out

        if isinstance(stmt, ast.Match):
            subj_env = env.copy()
            self._visit_expr(stmt.subject, subj_env)
            subj_reads_attr = _node_reads_module_attr(
                stmt.subject, subj_env.var_reads_module_attr
            )
            subj_reads_unguarded = _node_reads_unguarded_attr(stmt.subject, subj_env)
            running_env = subj_env.copy()
            case_outs: list[_GuardEnv] = []
            for case in stmt.cases:
                # PEP 634: evaluate pattern sub-expressions before binding names.
                for pnode in ast.walk(case.pattern):
                    if isinstance(pnode, ast.MatchValue):
                        self._visit_expr(pnode.value, running_env)
                    elif isinstance(pnode, ast.MatchClass):
                        self._visit_expr(pnode.cls, running_env)
                    elif isinstance(pnode, ast.MatchMapping):
                        for key in pnode.keys:
                            self._visit_expr(key, running_env)
                bound: set[str] = set()
                for pnode in ast.walk(case.pattern):
                    if isinstance(pnode, (ast.MatchAs, ast.MatchStar)) and pnode.name:
                        bound.add(pnode.name)
                    elif isinstance(pnode, ast.MatchMapping) and pnode.rest:
                        bound.add(pnode.rest)
                c_env = running_env.copy()
                for name in bound:
                    c_env.invalidate(name)
                    if subj_reads_attr:
                        c_env.var_reads_module_attr[name] = True
                    if subj_reads_unguarded:
                        c_env.var_unguarded_attr[name] = True
                if case.guard is not None:
                    # PEP 634 commits pattern bindings before evaluating the guard,
                    # so a failing guard still leaves those names rebound for later cases.
                    for name in bound:
                        running_env.invalidate(name)
                        if subj_reads_attr:
                            running_env.var_reads_module_attr[name] = True
                        if subj_reads_unguarded:
                            running_env.var_unguarded_attr[name] = True
                    self._visit_expr(case.guard, c_env)
                    for name in _node_assigned_names(case.guard):
                        running_env.invalidate(name)
                    running_env.merge_taint_from(c_env)
                    pos, _ = _extract_facts(case.guard, self.info, c_env)
                    c_env = c_env.apply(pos)
                case_outs.append(self._visit_block(case.body, c_env))
            out = subj_env.copy()
            for name in _assigned_names([stmt]):
                out.invalidate(name)
            out.merge_taint_from(subj_env)
            out.merge_taint_from(running_env)
            for c_out in case_outs:
                out.merge_taint_from(c_out)
            return out

        cur = env.copy()
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.expr):
                self._visit_expr(child, cur)
        return cur

    def _visit_expr(self, expr: ast.expr, env: _GuardEnv) -> None:
        if isinstance(expr, ast.NamedExpr):
            self._visit_expr(expr.value, env)
            rhs_reads_attr = _node_reads_module_attr(
                expr.value, env.var_reads_module_attr
            )
            rhs_reads_unguarded_attr = _node_reads_unguarded_attr(expr.value, env)
            self._bind_one_target(
                expr.target,
                env,
                rhs_reads_attr=rhs_reads_attr,
                rhs_reads_unguarded_attr=rhs_reads_unguarded_attr,
                layout_promise=self._layout_promise(expr.value),
                elem_facts=self._subscript_elem_facts(expr.value, env),
                rhs_key=_expr_key(expr.value),
            )
            return

        if isinstance(expr, ast.BoolOp):
            if isinstance(expr.op, ast.And):
                cur = env.copy()
                for val in expr.values:
                    self._visit_expr(val, cur)
                    for name in _named_expr_targets(val):
                        env.invalidate(name)
                    env.merge_taint_from(cur)
                    pos, _ = _extract_facts(val, self.info, cur)
                    cur = cur.apply(pos)
                return
            if isinstance(expr.op, ast.Or):
                cur = env.copy()
                for val in expr.values:
                    self._visit_expr(val, cur)
                    for name in _named_expr_targets(val):
                        env.invalidate(name)
                    env.merge_taint_from(cur)
                    _, neg = _extract_facts(val, self.info, cur)
                    cur = cur.apply(neg)
                return

        if isinstance(expr, ast.IfExp):
            self._visit_expr(expr.test, env)
            pos, neg = _extract_facts(expr.test, self.info, env)
            body_env = env.apply(pos)
            self._visit_expr(expr.body, body_env)
            else_env = env.apply(neg)
            self._visit_expr(expr.orelse, else_env)
            for name in _named_expr_targets(expr.body) | _named_expr_targets(
                expr.orelse
            ):
                env.invalidate(name)
            env.merge_taint_from(body_env)
            env.merge_taint_from(else_env)
            return

        if isinstance(expr, ast.Lambda):
            # Nested lambda resets local guards (soundness against parameter shadowing).
            for default in expr.args.defaults:
                self._visit_expr(default, env)
            for kw_default in expr.args.kw_defaults:
                if kw_default is not None:
                    self._visit_expr(kw_default, env)
            self._visit_expr(expr.body, _GuardEnv.empty())
            return

        if isinstance(expr, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            self._visit_comprehension(expr, expr.generators, (expr.elt,), env)
            return

        if isinstance(expr, ast.DictComp):
            self._visit_comprehension(
                expr, expr.generators, (expr.key, expr.value), env
            )
            return

        if isinstance(expr, ast.Compare):
            self._check_exact_type_compare(expr)
            self._visit_expr(expr.left, env)
            for comp in expr.comparators:
                self._visit_expr(comp, env)
            return

        if isinstance(expr, ast.Call):
            self._check_call(expr, env)
            self._visit_expr(expr.func, env)
            for arg in expr.args:
                self._visit_expr(arg, env)
            for kw in expr.keywords:
                self._visit_expr(kw.value, env)
            return

        if isinstance(expr, ast.Attribute):
            if isinstance(expr.ctx, ast.Load) and expr.attr in _TENSOR_ATTR_TO_RULE:
                self._check_attr_read(expr, env)
            self._visit_expr(expr.value, env)
            return

        if isinstance(expr, ast.Subscript):
            if isinstance(expr.ctx, ast.Load):
                self._check_subscript_read(expr, env)
            self._visit_expr(expr.value, env)
            self._visit_expr(expr.slice, env)
            return

        for child in ast.iter_child_nodes(expr):
            if isinstance(child, ast.expr):
                self._visit_expr(child, env)

    def _visit_comprehension(
        self,
        comp_node: ast.AST,
        generators: Sequence[ast.comprehension],
        elts: Sequence[ast.expr],
        env: _GuardEnv,
    ) -> None:
        cur = env.copy()
        for gen in generators:
            self._visit_expr(gen.iter, cur)
            iter_reads_attr = _node_reads_module_attr(
                gen.iter, cur.var_reads_module_attr
            )
            iter_reads_unguarded = _node_reads_unguarded_attr(gen.iter, cur)
            iter_key = _expr_key(gen.iter)
            elem_facts: set[str] | None = None
            if iter_key is not None:
                facts = cur.seq_guards.get(iter_key)
                if facts is None and iter_key in cur.seq_aliases:
                    facts = cur.seq_guards.get(cur.seq_aliases[iter_key])
                if facts:
                    elem_facts = set(facts)
            self._bind_one_target(
                gen.target,
                cur,
                rhs_reads_attr=iter_reads_attr,
                rhs_reads_unguarded_attr=iter_reads_unguarded,
                layout_promise=None,
                elem_facts=elem_facts,
                rhs_key=None,
            )
            for if_clause in gen.ifs:
                self._visit_expr(if_clause, cur)
                pos, _ = _extract_facts(if_clause, self.info, cur)
                cur = cur.apply(pos)
        for elt in elts:
            self._visit_expr(elt, cur)
        # PEP 572: walrus targets inside a comprehension bind in the enclosing scope.
        for name in _named_expr_targets(comp_node):
            env.invalidate(name)
            if cur.var_unguarded_attr.get(name):
                env.var_unguarded_attr[name] = True
            if cur.var_reads_module_attr.get(name):
                env.var_reads_module_attr[name] = True

    def _check_attr_read(self, node: ast.Attribute, env: _GuardEnv) -> None:
        attr = node.attr
        recv_key = _expr_key(node.value)
        tail = _tail_name(node.value)
        if attr == "weight":
            if tail and _is_norm_or_conv_name(tail):
                return
            if (
                recv_key == "self"
                and self.class_stack
                and self.class_stack[-1] in self.info.self_weight_classes
            ):
                return
        if recv_key is not None and env.has_attr_guard(recv_key, attr):
            return
        self._emit(
            node,
            _TENSOR_ATTR_TO_RULE[attr],
            f"{recv_key or '<expr>'}.{attr}",
            f"unguarded read of '.{attr}' on possible format layer",
        )

    def _check_subscript_read(self, node: ast.Subscript, env: _GuardEnv) -> None:
        sl = node.slice
        if not (isinstance(sl, ast.Constant) and isinstance(sl.value, str)):
            return
        attr = sl.value
        if attr not in _TENSOR_ATTR_TO_RULE:
            return
        tail = _tail_name(node.value)
        if tail in _PLAIN_DICT_NAMES:
            return
        recv_key = _expr_key(node.value)
        if recv_key is not None and env.has_attr_guard(recv_key, attr):
            return
        self._emit(
            node,
            _TENSOR_ATTR_TO_RULE[attr],
            f"{recv_key or '<expr>'}[{attr!r}]",
            f"unguarded subscript [{attr!r}] on possible format layer",
        )

    def _check_call(self, node: ast.Call, env: _GuardEnv) -> None:
        # 2-arg getattr(recv, "weight"|"scales"|"biases") without default.
        if (
            _tail_name(node.func) == "getattr"
            and len(node.args) == 2
            and not node.keywords
        ):
            attr_node = node.args[1]
            if (
                isinstance(attr_node, ast.Constant)
                and isinstance(attr_node.value, str)
                and attr_node.value in _TENSOR_ATTR_TO_RULE
            ):
                attr = attr_node.value
                recv_key = _expr_key(node.args[0])
                tail = _tail_name(node.args[0])
                if attr == "weight" and tail and _is_norm_or_conv_name(tail):
                    return
                if recv_key is not None and env.has_attr_guard(recv_key, attr):
                    return
                self._emit(
                    node,
                    _TENSOR_ATTR_TO_RULE[attr],
                    f"getattr({recv_key or '<expr>'}, {attr!r})",
                    f"unguarded 2-arg getattr(..., {attr!r}) on possible format layer",
                )
            return

        op = _mlx_op_name(
            node.func,
            self.info.mx_modules,
            self.info.mlx_roots,
            self.info.op_aliases,
        )
        if op is None:
            return
        if not _call_has_unguarded_arg(node, env) and (
            env.has_op_guard() or _call_has_guarded_arg(node, env)
        ):
            return
        has_mod_attr = any(
            _node_reads_module_attr(arg, env.var_reads_module_attr)
            for arg in [*node.args, *(kw.value for kw in node.keywords)]
        )
        if op == "dequantize":
            # KV-cache tuple dequantization (no module tensor attrs).
            if not has_mod_attr and _call_references_kv_cache(node):
                return
            # Raw disk/mmap affine dequantization: literal mode="affine" with
            # zero module tensor attr reads in args or their defining RHS.
            if not has_mod_attr and _call_has_literal_affine_mode(node):
                return
        elif (
            op in {"quantized_matmul", "gather_qmm"}
            and not has_mod_attr
            and _call_references_kv_cache(node)
        ):
            return
        self._emit(
            node,
            _MLX_OP_TO_RULE[op],
            op,
            f"unguarded MLX quantized op '{op}' without format-layer guard",
        )

    def _check_exact_type_compare(self, node: ast.Compare) -> None:
        operands = [node.left, *node.comparators]
        for idx, op in enumerate(node.ops):
            left, right = operands[idx], operands[idx + 1]
            if isinstance(op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
                for a, b in ((left, right), (right, left)):
                    if _is_type_call(a) or _is_class_attr(a):
                        q = _resolve_class_set(
                            b, self.info.quant_aliases, _QUANTIZED_CLASS_NAMES
                        )
                        f = _resolve_class_set(
                            b, self.info.format_aliases, _FORMAT_CLASS_NAMES
                        )
                        if q and not f:
                            self._emit(
                                node,
                                "exact-type-check",
                                ",".join(sorted(q)),
                                (
                                    f"exact type comparison against {', '.join(sorted(q))} "
                                    "excludes FormatLinear/FormatSwitchLinear/FormatEmbedding"
                                ),
                            )
                            return
                    elif _is_type_name_expr(a):
                        names = _resolve_string_names(b)
                        q = names & _QUANTIZED_CLASS_NAMES
                        f = names & _FORMAT_CLASS_NAMES
                        if q and not f:
                            self._emit(
                                node,
                                "exact-type-check",
                                ",".join(sorted(q)),
                                (
                                    f"exact type name comparison against {', '.join(sorted(q))} "
                                    "excludes format layers"
                                ),
                            )
                            return
            elif isinstance(op, (ast.In, ast.NotIn)):
                if _is_type_call(left) or _is_class_attr(left):
                    q = _resolve_class_set(
                        right, self.info.quant_aliases, _QUANTIZED_CLASS_NAMES
                    )
                    f = _resolve_class_set(
                        right, self.info.format_aliases, _FORMAT_CLASS_NAMES
                    )
                    if q and not f:
                        self._emit(
                            node,
                            "exact-type-check",
                            ",".join(sorted(q)),
                            (
                                f"exact type membership check against {', '.join(sorted(q))} "
                                "excludes format layers"
                            ),
                        )
                        return
                elif _is_type_name_expr(left):
                    names = _resolve_string_names(right)
                    q = names & _QUANTIZED_CLASS_NAMES
                    f = names & _FORMAT_CLASS_NAMES
                    if q and not f:
                        self._emit(
                            node,
                            "exact-type-check",
                            ",".join(sorted(q)),
                            (
                                f"exact type name membership check against {', '.join(sorted(q))} "
                                "excludes format layers"
                            ),
                        )
                        return


def _walk_assigned_names(root: ast.AST, names: set[str]) -> None:
    for node in ast.walk(root):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            key = _expr_key(node)
            if key is not None:
                names.add(key)
        elif (
            isinstance(
                node,
                (
                    ast.MatchAs,
                    ast.MatchStar,
                    ast.ExceptHandler,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            )
            and node.name
        ):
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.asname:
                    names.add(alias.asname)
                elif alias.name != "*":
                    names.add(alias.name.split(".", 1)[0])


def _assigned_names(stmts: Sequence[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for stmt in stmts:
        _walk_assigned_names(stmt, names)
    return names


def _node_assigned_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    _walk_assigned_names(node, names)
    return names


def _named_expr_targets(node: ast.AST) -> set[str]:
    return {
        sub.target.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name)
    }


def _allow_entry_matches(entry: AllowEntry, finding: Finding) -> bool:
    if finding.rule == INVALID_PRAGMA_RULE:
        return False
    if entry.rule != "*" and entry.rule != finding.rule:
        return False
    if entry.scope != "*" and entry.scope != finding.scope:
        return False
    norm_path = finding.path.replace("\\", "/")
    ep = entry.path.replace("\\", "/").strip()
    if ep.startswith("./"):
        ep = ep[2:]
    if any(ch in ep for ch in "*?["):
        return fnmatch.fnmatchcase(norm_path, ep) or fnmatch.fnmatchcase(
            norm_path, f"*/{ep}"
        )
    return norm_path == ep or norm_path.endswith(f"/{ep}")


def _scan_single_source(
    source: str,
    path: str,
    allowlist: Sequence[AllowEntry],
    check_unused_pragmas: bool,
) -> tuple[list[Finding], int, list[tuple[str, str]], list[tuple[str, int, str]]]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        detail = f"SyntaxError: {exc.msg} (line {exc.lineno})"
        return [], 0, [(path, detail)], []

    first_code = _first_code_line(tree)
    try:
        pragmas, invalid_pragmas = _parse_pragmas(source, path, first_code)
    except tokenize.TokenError as exc:
        return [], 0, [(path, f"TokenError: {exc}")], []

    info = _collect_file_info(tree)
    visitor = _AuditVisitor(path, info)
    raw_findings = visitor.scan(tree)

    file_pragmas = [p for p in pragmas if p.kind == "allow-file"]
    line_pragmas_by_line: dict[int, list[_Pragma]] = collections.defaultdict(list)
    for p in pragmas:
        if p.kind == "allow":
            line_pragmas_by_line[p.line].append(p)

    used_pragmas: set[int] = set()
    kept: list[Finding] = list(invalid_pragmas)
    suppressed = 0

    for raw in raw_findings:
        finding = raw.finding
        matched_pragma: _Pragma | None = None

        # Check line pragmas: same line, preceding line, or closing line of multi-line call.
        candidate_lines = (finding.line, finding.line - 1, raw.end_line)
        for cand_line in candidate_lines:
            for p in line_pragmas_by_line.get(cand_line, ()):
                if finding.rule in p.rules:
                    matched_pragma = p
                    break
            if matched_pragma is not None:
                break

        if matched_pragma is None:
            for p in file_pragmas:
                if finding.rule in p.rules:
                    matched_pragma = p
                    break

        if matched_pragma is not None:
            used_pragmas.add(matched_pragma.line)
            suppressed += 1
            continue

        if any(_allow_entry_matches(entry, finding) for entry in allowlist):
            suppressed += 1
            continue

        kept.append(finding)

    kept.sort(key=lambda f: (f.path, f.line, f.col, f.rule, f.symbol))

    unused: list[tuple[str, int, str]] = []
    if check_unused_pragmas:
        for p in pragmas:
            if p.line not in used_pragmas:
                unused.append((path, p.line, p.raw))
        unused.sort(key=lambda item: (item[0], item[1]))

    return kept, suppressed, [], unused


def scan_source(
    source: str,
    path: str = "<string>",
    *,
    allowlist: Sequence[AllowEntry] = (),
    check_unused_pragmas: bool = False,
) -> AuditReport:
    """Scan Python ``source`` text for unguarded quant-format layer readers."""
    norm_path = path.replace("\\", "/")
    findings, suppressed, unparsable, unused = _scan_single_source(
        source, norm_path, allowlist, check_unused_pragmas
    )
    return AuditReport(
        files_scanned=(norm_path,),
        findings=tuple(findings),
        suppressed=suppressed,
        unparsable=tuple(unparsable),
        unused_pragmas=tuple(unused),
    )


def _is_excluded(posix_path: str, exclude: Sequence[str]) -> bool:
    for pat in exclude:
        norm_pat = pat.replace("\\", "/")
        if fnmatch.fnmatchcase(posix_path, norm_pat) or fnmatch.fnmatchcase(
            posix_path, f"*/{norm_pat}"
        ):
            return True
    return False


def _collect_python_files(
    paths: Sequence[str | pathlib.Path], exclude: Sequence[str]
) -> list[pathlib.Path]:
    collected: list[pathlib.Path] = []
    seen_files: set[str] = set()
    for raw in paths:
        root = pathlib.Path(raw)
        if not root.exists():
            raise FileNotFoundError(f"audit path does not exist: {raw}")
        if root.is_file():
            posix = root.as_posix()
            if posix not in seen_files and not _is_excluded(posix, exclude):
                seen_files.add(posix)
                collected.append(root)
            continue
        visited_dirs: set[tuple[int, int]] = set()
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            try:
                st = os.stat(dirpath)
            except OSError:
                dirnames.clear()
                continue
            dir_id = (st.st_dev, st.st_ino)
            if dir_id in visited_dirs:
                dirnames.clear()
                continue
            visited_dirs.add(dir_id)
            dirnames.sort()
            # Prune symlinked directories already visited and __pycache__.
            kept_dirs: list[str] = []
            for dname in dirnames:
                if dname == "__pycache__":
                    continue
                child = pathlib.Path(dirpath) / dname
                if _is_excluded(child.as_posix(), exclude):
                    continue
                try:
                    cst = os.stat(child)
                except OSError:
                    continue
                if (cst.st_dev, cst.st_ino) in visited_dirs:
                    continue
                kept_dirs.append(dname)
            dirnames[:] = kept_dirs

            for fname in sorted(filenames):
                if not fname.endswith(".py"):
                    continue
                fpath = pathlib.Path(dirpath) / fname
                posix = fpath.as_posix()
                if posix not in seen_files and not _is_excluded(posix, exclude):
                    seen_files.add(posix)
                    collected.append(fpath)
    collected.sort(key=lambda p: p.as_posix())
    return collected


def scan_paths(
    paths: Sequence[str | pathlib.Path],
    *,
    exclude: Sequence[str] = (),
    allowlist: Sequence[AllowEntry] = (),
    check_unused_pragmas: bool = False,
) -> AuditReport:
    """Scan Python files/directories deterministically for unguarded readers."""
    files = _collect_python_files(paths, exclude)
    scanned: list[str] = []
    all_findings: list[Finding] = []
    total_suppressed = 0
    all_unparsable: list[tuple[str, str]] = []
    all_unused: list[tuple[str, int, str]] = []

    for fpath in files:
        posix = fpath.as_posix()
        scanned.append(posix)
        try:
            source = fpath.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            all_unparsable.append((posix, f"{type(exc).__name__}: {exc}"))
            continue
        findings, suppressed, unparsable, unused = _scan_single_source(
            source, posix, allowlist, check_unused_pragmas
        )
        all_findings.extend(findings)
        total_suppressed += suppressed
        all_unparsable.extend(unparsable)
        all_unused.extend(unused)

    all_findings.sort(key=lambda f: (f.path, f.line, f.col, f.rule, f.symbol))
    all_unparsable.sort()
    all_unused.sort(key=lambda item: (item[0], item[1]))

    return AuditReport(
        files_scanned=tuple(scanned),
        findings=tuple(all_findings),
        suppressed=total_suppressed,
        unparsable=tuple(all_unparsable),
        unused_pragmas=tuple(all_unused),
    )


def assert_no_unguarded_readers(
    paths: Sequence[str | pathlib.Path],
    *,
    exclude: Sequence[str] = (),
    allowlist: Sequence[AllowEntry] = (),
    check_unused_pragmas: bool = False,
) -> AuditReport:
    """Pytest / CI helper: raise ``AuditError`` on any finding, parse error,
    or (when requested) unused pragma."""
    report = scan_paths(
        paths,
        exclude=exclude,
        allowlist=allowlist,
        check_unused_pragmas=check_unused_pragmas,
    )
    if not report.ok or (check_unused_pragmas and report.unused_pragmas):
        raise AuditError(report)
    return report


def format_report(report: AuditReport, *, fmt: str = "text") -> str:
    """Render ``report`` as human-readable text or deterministic JSON."""
    if fmt == "text":
        lines: list[str] = []
        for f in report.findings:
            lines.append(
                f"{f.path}:{f.line}:{f.col}: [{f.rule}] ({f.scope}) {f.message}"
            )
        for path, err in report.unparsable:
            lines.append(f"{path}: unparsable: {err}")
        for path, line, raw in report.unused_pragmas:
            lines.append(f"{path}:{line}: unused pragma: {raw.strip()}")
        lines.append(
            f"{len(report.files_scanned)} file(s) scanned, "
            f"{len(report.findings)} finding(s), {report.suppressed} suppressed, "
            f"{len(report.unparsable)} unparsable, "
            f"{len(report.unused_pragmas)} unused pragma(s)"
        )
        return "\n".join(lines)
    if fmt == "json":
        payload = {
            "files_scanned": list(report.files_scanned),
            "findings": [
                {
                    "path": f.path,
                    "line": f.line,
                    "col": f.col,
                    "rule": f.rule,
                    "scope": f.scope,
                    "symbol": f.symbol,
                    "message": f.message,
                }
                for f in report.findings
            ],
            "suppressed": report.suppressed,
            "unparsable": [
                {"path": path, "error": err} for path, err in report.unparsable
            ],
            "unused_pragmas": [
                {"path": path, "line": line, "pragma": raw}
                for path, line, raw in report.unused_pragmas
            ],
            "ok": report.ok,
        }
        return json.dumps(payload, sort_keys=True, indent=2)
    raise ValueError(f"unsupported report format {fmt!r}; expected 'text' or 'json'")


def _parse_allow_arg(spec: str) -> AllowEntry:
    """Parse ``PATH[:SCOPE][=RULE]::REASON`` into an :class:`AllowEntry`."""
    if not isinstance(spec, str) or "::" not in spec:
        raise ValueError(f"--allow expects PATH[:SCOPE][=RULE]::REASON, not {spec!r}")
    head, reason = spec.split("::", 1)
    rule = "*"
    if "=" in head:
        head, rule = head.rsplit("=", 1)
        rule = rule.strip()
    scope = "*"
    if ":" in head:
        head, scope = head.rsplit(":", 1)
        scope = scope.strip()
    try:
        return AllowEntry(
            path=head.strip(), rule=rule, scope=scope, reason=reason.strip()
        )
    except ValueError as exc:
        raise ValueError(f"invalid --allow {spec!r}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Scan paths from the command line; return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m omlx.formats.audit",
        description=(
            "Scan Python sources for unguarded quant-format layer reads "
            "(.weight/.scales/.biases, quantized matmul, dequantize, gather_qmm, "
            "exact type checks)."
        ),
    )
    parser.add_argument(
        "paths", nargs="+", metavar="PATH", help="file or directory to scan"
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="skip paths matching GLOB (repeatable)",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="SPEC",
        help="suppress findings: PATH[:SCOPE][=RULE]::REASON (repeatable)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="report format (default: text)",
    )
    parser.add_argument(
        "--check-unused-pragmas",
        action="store_true",
        help="fail when an omlx-audit pragma suppresses nothing",
    )
    args = parser.parse_args(argv)

    try:
        allowlist = tuple(_parse_allow_arg(spec) for spec in args.allow)
    except ValueError as exc:
        print(f"omlx.formats.audit: {exc}", file=sys.stderr)
        return 2

    try:
        report = scan_paths(
            args.paths,
            exclude=args.exclude,
            allowlist=allowlist,
            check_unused_pragmas=args.check_unused_pragmas,
        )
    except (FileNotFoundError, OSError) as exc:
        print(f"omlx.formats.audit: {exc}", file=sys.stderr)
        return 2

    print(format_report(report, fmt=args.format))
    if not report.ok:
        return 1
    if args.check_unused_pragmas and report.unused_pragmas:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
