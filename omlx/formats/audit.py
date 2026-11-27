# SPDX-License-Identifier: Apache-2.0
"""AST format-reader audit scanner and pytest CI helper.

Pure Python stdlib (zero ``mlx`` imports, zero environment variables) so it runs
on any host, including CPU-only CI and MLX-free hosts.
"""

from __future__ import annotations

import ast
import collections
import fnmatch
import io
import os
import pathlib
import re
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
    scope_ops: bool = False

    @classmethod
    def empty(cls) -> _FactsDelta:
        return cls({}, {}, False)

    def merge(self, other: _FactsDelta) -> _FactsDelta:
        recv: dict[str, set[str]] = {k: set(v) for k, v in self.recv_facts.items()}
        for k, v in other.recv_facts.items():
            recv.setdefault(k, set()).update(v)
        seq: dict[str, set[str]] = {k: set(v) for k, v in self.seq_guards.items()}
        for k, v in other.seq_guards.items():
            seq.setdefault(k, set()).update(v)
        return _FactsDelta(recv, seq, self.scope_ops or other.scope_ops)


@dataclass
class _GuardEnv:
    recv_facts: dict[str, set[str]]
    seq_guards: dict[str, set[str]]
    layout_vars: dict[str, tuple[str, frozenset[str]]]
    seq_aliases: dict[str, str]
    var_reads_module_attr: dict[str, bool]
    scope_ops_guarded: bool = False

    @classmethod
    def empty(cls) -> _GuardEnv:
        return cls({}, {}, {}, {}, {}, False)

    def copy(self) -> _GuardEnv:
        return _GuardEnv(
            recv_facts={k: set(v) for k, v in self.recv_facts.items()},
            seq_guards={k: set(v) for k, v in self.seq_guards.items()},
            layout_vars=dict(self.layout_vars),
            seq_aliases=dict(self.seq_aliases),
            var_reads_module_attr=dict(self.var_reads_module_attr),
            scope_ops_guarded=self.scope_ops_guarded,
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
        if delta.scope_ops:
            out.scope_ops_guarded = True
        return out

    def invalidate(self, var_name: str) -> None:
        prefix = f"{var_name}."
        for key in list(self.recv_facts):
            if key == var_name or key.startswith(prefix):
                del self.recv_facts[key]
        for key in list(self.seq_guards):
            if key == var_name or key.startswith(prefix):
                del self.seq_guards[key]
        self.layout_vars.pop(var_name, None)
        self.seq_aliases.pop(var_name, None)
        self.var_reads_module_attr.pop(var_name, None)

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
        if self.scope_ops_guarded:
            return True
        if recv_key is not None:
            facts = self.recv_facts.get(recv_key)
            if facts and (facts & {"affine", "quantized", "non_format"}):
                return True
        return False


def _is_none_const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


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
                _FactsDelta({}, {seq_key: set(facts)}, scope_ops=True),
                _FactsDelta.empty(),
            )
        return _FactsDelta.empty(), _FactsDelta.empty()

    if isinstance(expr, ast.Attribute) and expr.attr == "_stock":
        recv = _expr_key(expr.value)
        if recv is not None:
            return (
                _FactsDelta({recv: {"affine"}}, {}, scope_ops=True),
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
                        _FactsDelta({recv: {"non_format"}}, {}, scope_ops=True),
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
                        _FactsDelta({recv: {"quantized"}}, {}, scope_ops=True),
                        _FactsDelta.empty(),
                    )
                if f_cls and not q_cls:
                    return (
                        _FactsDelta.empty(),
                        _FactsDelta({recv: {"non_format"}}, {}, scope_ops=True),
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
                            ops = bool(facts & {"affine", "quantized", "non_format"})
                            return (
                                _FactsDelta({}, {seq_key: set(facts)}, scope_ops=ops),
                                _FactsDelta.empty(),
                            )
        elif isinstance(expr.func, ast.Name) and expr.func.id in info.pred_facts:
            param_idx, facts = info.pred_facts[expr.func.id]
            if param_idx < len(expr.args):
                recv = _expr_key(expr.args[param_idx])
                if recv is not None:
                    ops = bool(facts & {"affine", "quantized", "non_format"})
                    return (
                        _FactsDelta({recv: set(facts)}, {}, scope_ops=ops),
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
                        delta = _FactsDelta(
                            {g_recv: {"non_format"}}, {}, scope_ops=True
                        )
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
                        delta = _FactsDelta({mode_recv: {"affine"}}, {}, scope_ops=True)
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


def _analyze_predicate_fn(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    base_info: _FileInfo,
) -> tuple[int, frozenset[str]] | None:
    params = [a.arg for a in fn.args.posonlyargs] + [a.arg for a in fn.args.args]
    if not params:
        return None
    env = _GuardEnv.empty()
    body = fn.body
    start = 1 if body and _is_docstring_stmt(body[0]) else 0
    for stmt in body[start:]:
        if isinstance(stmt, ast.Return) and stmt.value is not None:
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
            and all(_is_falsy_return(s) for s in stmt.body if isinstance(s, ast.Return))
        ):
            _, neg = _extract_facts(stmt.test, base_info, env)
            env = env.apply(neg)
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
    env = _GuardEnv.empty()
    body = fn.body
    start = 1 if body and _is_docstring_stmt(body[0]) else 0
    saw_truthy_return = False
    for stmt in body[start:]:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            t = stmt.targets[0]
            if (
                isinstance(t, ast.Name)
                and isinstance(stmt.value, ast.Attribute)
                and isinstance(stmt.value.value, ast.Name)
                and stmt.value.value.id == "self"
            ):
                env.seq_aliases[t.id] = f"self.{stmt.value.attr}"
        elif isinstance(stmt, ast.If) and _always_exits(stmt.body) and not stmt.orelse:
            if any(_is_falsy_return(s) for s in stmt.body):
                _, neg = _extract_facts(stmt.test, info, env)
                env = env.apply(neg)
        elif isinstance(stmt, ast.Return) and not _is_falsy_return(stmt):
            saw_truthy_return = True
    if not saw_truthy_return:
        return None
    for seq_key, facts in env.seq_guards.items():
        if seq_key.startswith("self.") and (
            facts & {"affine", "quantized", "non_format"}
            or {"weight", "scales", "biases"}.issubset(facts)
        ):
            return seq_key.split(".", 1)[1], frozenset(facts)
    return None


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
        if isinstance(sub, ast.Name) and var_reads.get(sub.id, False):
            return True
    return False


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
        for stmt in stmts[start:]:
            cur = self._visit_stmt(stmt, cur)
        return cur

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
        # Check if RHS is a layout guard call `self.<method>(...)`.
        layout_promise: tuple[str, frozenset[str]] | None = None
        if (
            value is not None
            and self.class_stack
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "self"
        ):
            layout_promise = self.info.class_layout_guards.get(
                (self.class_stack[-1], value.func.attr)
            )
        # Check if RHS is indexing a guarded sequence `seq[i]`.
        elem_facts: set[str] | None = None
        if value is not None and isinstance(value, ast.Subscript):
            seq_key = _expr_key(value.value)
            if seq_key is not None:
                facts = out.seq_guards.get(seq_key)
                if facts is None and seq_key in out.seq_aliases:
                    facts = out.seq_guards.get(out.seq_aliases[seq_key])
                if facts:
                    elem_facts = set(facts)
        # Check if RHS aliases a sequence (`shards = self.shards`).
        rhs_key = _expr_key(value) if value is not None else None

        for target in targets:
            self._bind_one_target(
                target,
                out,
                rhs_reads_attr=rhs_reads_attr,
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
        layout_promise: tuple[str, frozenset[str]] | None,
        elem_facts: set[str] | None,
        rhs_key: str | None,
    ) -> None:
        if isinstance(target, ast.Name):
            env.invalidate(target.id)
            if rhs_reads_attr:
                env.var_reads_module_attr[target.id] = True
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
                    layout_promise=None,
                    elem_facts=None,
                    rhs_key=None,
                )
        elif isinstance(target, ast.Starred):
            self._bind_one_target(
                target.value,
                env,
                rhs_reads_attr=rhs_reads_attr,
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
        elif isinstance(target, ast.Subscript):
            self._visit_expr(target.value, env)
            self._visit_expr(target.slice, env)

    def _visit_stmt(self, stmt: ast.stmt, env: _GuardEnv) -> _GuardEnv:
        if isinstance(stmt, ast.ClassDef):
            for dec in stmt.decorator_list:
                self._visit_expr(dec, env)
            for base in stmt.bases:
                self._visit_expr(base, env)
            for kw in stmt.keywords:
                self._visit_expr(kw.value, env)
            self.scope_stack.append(stmt.name)
            self.class_stack.append(stmt.name)
            self._visit_block(stmt.body, _GuardEnv.empty(), skip_docstring=True)
            self.class_stack.pop()
            self.scope_stack.pop()
            return env

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in stmt.decorator_list:
                self._visit_expr(dec, env)
            for default in stmt.args.defaults:
                self._visit_expr(default, env)
            for kw_default in stmt.args.kw_defaults:
                if kw_default is not None:
                    self._visit_expr(kw_default, env)
            self.scope_stack.append(stmt.name)
            # Soundness: entering a nested def resets local variable guards.
            self._visit_block(stmt.body, _GuardEnv.empty(), skip_docstring=True)
            self.scope_stack.pop()
            return env

        if isinstance(stmt, ast.Assign):
            self._visit_expr(stmt.value, env)
            return self._bind_targets(stmt.targets, stmt.value, env)

        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                self._visit_expr(stmt.value, env)
                return self._bind_targets((stmt.target,), stmt.value, env)
            return env

        if isinstance(stmt, ast.AugAssign):
            self._visit_expr(stmt.value, env)
            self._visit_expr(stmt.target, env)
            if isinstance(stmt.target, ast.Name):
                out = env.copy()
                out.invalidate(stmt.target.id)
                return out
            return env

        if isinstance(stmt, ast.Assert):
            self._visit_expr(stmt.test, env)
            if stmt.msg is not None:
                self._visit_expr(stmt.msg, env)
            pos, _ = _extract_facts(stmt.test, self.info, env)
            return env.apply(pos)

        if isinstance(stmt, ast.If):
            self._visit_expr(stmt.test, env)
            pos, neg = _extract_facts(stmt.test, self.info, env)
            then_env = self._visit_block(stmt.body, env.apply(pos))
            else_env = self._visit_block(stmt.orelse, env.apply(neg))
            then_exits = _always_exits(stmt.body)
            else_exits = _always_exits(stmt.orelse)
            if then_exits and not stmt.orelse:
                return env.apply(neg)
            if then_exits and not else_exits:
                return else_env
            if else_exits and not then_exits:
                return then_env
            # Neither side guaranteed to exit: conservatively invalidate vars
            # reassigned inside either branch.
            merged = env.copy()
            for name in _assigned_names(stmt.body) | _assigned_names(stmt.orelse):
                merged.invalidate(name)
            # Preserve layout_vars newly initialized in an `if layout is None:` block.
            for var_name, promise in then_env.layout_vars.items():
                if var_name not in env.layout_vars:
                    merged.layout_vars[var_name] = promise
            return merged

        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            self._visit_expr(stmt.iter, env)
            loop_env = env.copy()
            iter_key = _expr_key(stmt.iter)
            elem_facts: set[str] | None = None
            if iter_key is not None:
                facts = loop_env.seq_guards.get(iter_key)
                if facts is None and iter_key in loop_env.seq_aliases:
                    facts = loop_env.seq_guards.get(loop_env.seq_aliases[iter_key])
                if facts:
                    elem_facts = set(facts)
            self._bind_one_target(
                stmt.target,
                loop_env,
                rhs_reads_attr=False,
                layout_promise=None,
                elem_facts=elem_facts,
                rhs_key=None,
            )
            self._visit_block(stmt.body, loop_env)
            if stmt.orelse:
                self._visit_block(stmt.orelse, env.copy())
            out = env.copy()
            for name in _assigned_names([stmt]):
                out.invalidate(name)
            return out

        if isinstance(stmt, ast.While):
            self._visit_expr(stmt.test, env)
            pos, _ = _extract_facts(stmt.test, self.info, env)
            self._visit_block(stmt.body, env.apply(pos))
            if stmt.orelse:
                self._visit_block(stmt.orelse, env.copy())
            out = env.copy()
            for name in _assigned_names([stmt]):
                out.invalidate(name)
            return out

        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            with_env = env.copy()
            for item in stmt.items:
                self._visit_expr(item.context_expr, with_env)
                if item.optional_vars is not None:
                    self._bind_one_target(
                        item.optional_vars,
                        with_env,
                        rhs_reads_attr=False,
                        layout_promise=None,
                        elem_facts=None,
                        rhs_key=None,
                    )
            return self._visit_block(stmt.body, with_env)

        if isinstance(stmt, ast.Try):
            self._visit_block(stmt.body, env.copy())
            for handler in stmt.handlers:
                if handler.type is not None:
                    self._visit_expr(handler.type, env)
                h_env = env.copy()
                if handler.name:
                    h_env.invalidate(handler.name)
                self._visit_block(handler.body, h_env)
            if stmt.orelse:
                self._visit_block(stmt.orelse, env.copy())
            if stmt.finalbody:
                self._visit_block(stmt.finalbody, env.copy())
            out = env.copy()
            for name in _assigned_names([stmt]):
                out.invalidate(name)
            return out

        if isinstance(stmt, ast.Return):
            if stmt.value is not None:
                self._visit_expr(stmt.value, env)
            return env

        if isinstance(stmt, ast.Expr):
            self._visit_expr(stmt.value, env)
            return env

        if isinstance(stmt, ast.Raise):
            if stmt.exc is not None:
                self._visit_expr(stmt.exc, env)
            if stmt.cause is not None:
                self._visit_expr(stmt.cause, env)
            return env

        if isinstance(stmt, ast.Delete):
            out = env.copy()
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    out.invalidate(target.id)
                else:
                    self._visit_expr(target, out)
            return out

        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.expr):
                self._visit_expr(child, env)
        return env

    def _visit_expr(self, expr: ast.expr, env: _GuardEnv) -> None:
        if isinstance(expr, ast.BoolOp):
            if isinstance(expr.op, ast.And):
                cur = env
                for val in expr.values:
                    self._visit_expr(val, cur)
                    pos, _ = _extract_facts(val, self.info, cur)
                    cur = cur.apply(pos)
                return
            if isinstance(expr.op, ast.Or):
                cur = env
                for val in expr.values:
                    self._visit_expr(val, cur)
                    _, neg = _extract_facts(val, self.info, cur)
                    cur = cur.apply(neg)
                return

        if isinstance(expr, ast.IfExp):
            self._visit_expr(expr.test, env)
            pos, neg = _extract_facts(expr.test, self.info, env)
            self._visit_expr(expr.body, env.apply(pos))
            self._visit_expr(expr.orelse, env.apply(neg))
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
            self._visit_comprehension(expr.generators, (expr.elt,), env)
            return

        if isinstance(expr, ast.DictComp):
            self._visit_comprehension(expr.generators, (expr.key, expr.value), env)
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
        generators: Sequence[ast.comprehension],
        elts: Sequence[ast.expr],
        env: _GuardEnv,
    ) -> None:
        cur = env.copy()
        for gen in generators:
            self._visit_expr(gen.iter, cur)
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
                rhs_reads_attr=False,
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
        if env.has_op_guard() or _call_has_guarded_arg(node, env):
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
        elif op in {"quantized_matmul", "gather_qmm"}:
            if not has_mod_attr and _call_references_kv_cache(node):
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


def _assigned_names(stmts: Sequence[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
    return names


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
