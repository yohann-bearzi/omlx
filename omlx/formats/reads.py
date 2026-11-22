# SPDX-License-Identifier: Apache-2.0
"""Declared read signatures for quant-format faces, and C++ signature emission.

A face declares its read members' signatures in Python (``DotReads``,
``TileReads``, ``CodesReads``) and oMLX emits the exact C++ signatures into
the face's Metal source at explicit ``__OMLX_READ_<SLOT>__`` markers when the
face is constructed.

Nothing parses C++: every unsafe form (mutable activation or scratch pointer,
wrong address space, pointer-const without pointee-const, narrow or reference
index, template ``weight`` or ``pair``, template defaults or SFINAE, wrong
return or row type, reference/pointer or mismatched ``aux``) is refused at
declaration time with a clear ``ValueError``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_KEYWORDS = frozenset(
    {
        "alignas",
        "alignof",
        "auto",
        "bfloat16_t",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "const_cast",
        "constant",
        "constexpr",
        "continue",
        "decltype",
        "default",
        "delete",
        "device",
        "do",
        "double",
        "else",
        "enum",
        "explicit",
        "extern",
        "false",
        "float",
        "float16_t",
        "for",
        "friend",
        "goto",
        "half",
        "if",
        "inline",
        "int",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
        "long",
        "METAL_FUNC",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "nullptr",
        "operator",
        "private",
        "protected",
        "public",
        "register",
        "reinterpret_cast",
        "return",
        "short",
        "signed",
        "size_t",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "template",
        "this",
        "thread",
        "threadgroup",
        "throw",
        "true",
        "try",
        "typedef",
        "typename",
        "uchar",
        "uchar2",
        "uchar4",
        "uint",
        "uint2",
        "uint3",
        "uint4",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "ulong",
        "union",
        "unsigned",
        "ushort",
        "ushort2",
        "ushort4",
        "using",
        "virtual",
        "void",
        "volatile",
        "while",
    }
)

_CONTRACT_TYPES = frozenset(
    {"Row", "Decoded", "P", "Codes", "Lanes", "Runs", "Dot", "Tile"}
)
_RESERVED = _KEYWORDS | _CONTRACT_TYPES

_POINTER_SPACES = frozenset({"device", "threadgroup", "thread"})
_ACTIVATION_SPACES = frozenset({"device", "threadgroup"})
_FLOAT_ELEMENTS = frozenset({"float", "half", "bfloat16_t"})
_DECODED_TYPES = frozenset({"thread const Decoded&", "const thread Decoded&"})
_WEIGHT_PARAMS_TYPES = frozenset({"thread const P&", "const thread P&"})
_WEIGHT_CODES_TYPES = frozenset({"Codes", "thread const Codes&", "const thread Codes&"})
_PACK_BUILTINS = frozenset(
    {
        "uchar",
        "uchar2",
        "uchar4",
        "uint",
        "uint2",
        "uint3",
        "uint4",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "ulong",
        "ushort",
        "ushort2",
        "ushort4",
    }
)
_INLINE_QUALIFIERS = frozenset({"METAL_FUNC", "static", "inline", "constexpr"})
_OUT_OF_LINE_QUALIFIERS = frozenset({"METAL_FUNC", "inline", "constexpr"})

_SLOT_PREFIX = "__OMLX_READ_"
_SLOT_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SHARED_TEMPLATE_RE = re.compile(r"^(typename|class|int) ([A-Za-z_][A-Za-z0-9_]*)$")
_OWNER_SEG_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:<([A-Za-z_][A-Za-z0-9_]*)>)?$")


def slot_marker(slot: str) -> str:
    """The Metal source placeholder expanded by a declared read member."""
    _check_slot(slot, "slot")
    return f"{_SLOT_PREFIX}{slot.upper()}__"


def _check_bool(value, label: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a bool, not {value!r}")


def _check_ident(name, label: str, *, allow_contract: bool = False) -> None:
    if not (isinstance(name, str) and name.isascii() and name.isidentifier()):
        raise ValueError(f"{label} must be a single ASCII identifier, not {name!r}")
    forbidden = _KEYWORDS if allow_contract else _RESERVED
    if name in forbidden:
        raise ValueError(f"{label} cannot use reserved C++/Metal name {name!r}")


def _check_slot(slot, label: str) -> None:
    if not (isinstance(slot, str) and _SLOT_RE.fullmatch(slot)):
        raise ValueError(
            f"{label} must be a lowercase ASCII identifier ([a-z][a-z0-9_]*), "
            f"not {slot!r}"
        )


def _check_params(
    params, expected: int, label: str, *, forbidden: frozenset[str] = frozenset()
) -> None:
    if not isinstance(params, tuple) or len(params) != expected:
        raise ValueError(
            f"{label} must have {expected} parameter names as a tuple, not {params!r}"
        )
    seen: set[str] = set()
    for item in params:
        if item == "":
            continue
        _check_ident(item, f"{label} parameter name")
        if item in seen:
            raise ValueError(
                f"{label} parameter names must be distinct, repeated {item!r}"
            )
        if item in forbidden:
            raise ValueError(
                f"{label} parameter name {item!r} cannot collide with type name"
            )
        seen.add(item)


def _check_qualifiers(
    qualifiers, label: str, *, allowed: frozenset[str], require_static: bool
) -> None:
    if not isinstance(qualifiers, tuple) or not all(
        isinstance(q, str) for q in qualifiers
    ):
        raise ValueError(f"{label} must be a tuple of strings, not {qualifiers!r}")
    seen: set[str] = set()
    for q in qualifiers:
        if q not in allowed:
            raise ValueError(
                f"{label} contains unsupported qualifier {q!r}; "
                f"allowed: {', '.join(sorted(allowed))}"
            )
        if q in seen:
            raise ValueError(f"{label} contains duplicate qualifier {q!r}")
        seen.add(q)
    if require_static and "static" not in seen:
        raise ValueError(f"{label} must include 'static', not {qualifiers!r}")


def _param(type_cpp: str, name: str) -> str:
    return f"{type_cpp} {name}" if name else type_cpp


@dataclass(frozen=True)
class TypeParam:
    """A member's own single template type parameter (``template <typename NAME>``).

    Cannot carry defaults, non-type parameters, parameter packs, or SFINAE.
    """

    name: str

    def __post_init__(self) -> None:
        _check_ident(self.name, "TypeParam name")


@dataclass(frozen=True)
class AuxType:
    """A face-declared auxiliary type returned by ``load`` and passed by value
    to ``madd`` and ``apply``."""

    name: str

    def __post_init__(self) -> None:
        _check_ident(self.name, "AuxType name")


@dataclass(frozen=True)
class Pointer:
    """A Metal address-space pointer type.

    ``const=True`` qualifies the pointed-to element (``thread const float*`` or
    ``const thread float*`` when ``const_first=True``). Top-level pointer
    constness (``T* const``) and references to pointers (``T* const&``) cannot
    be declared.
    """

    space: str
    element: str | TypeParam
    const: bool = True
    const_first: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.space, str) or self.space not in _POINTER_SPACES:
            raise ValueError(
                f"Pointer address space must be one of "
                f"{', '.join(sorted(_POINTER_SPACES))}, not {self.space!r}"
            )
        _check_bool(self.const, "Pointer.const")
        _check_bool(self.const_first, "Pointer.const_first")
        if self.const_first and not self.const:
            raise ValueError("Pointer.const_first=True requires const=True")
        if isinstance(self.element, TypeParam):
            return
        if not isinstance(self.element, str) or self.element not in _FLOAT_ELEMENTS:
            raise ValueError(
                f"Pointer element type must be one of "
                f"{', '.join(sorted(_FLOAT_ELEMENTS))} or TypeParam(...), "
                f"not {self.element!r}"
            )

    def render(self) -> str:
        elem = (
            self.element.name if isinstance(self.element, TypeParam) else self.element
        )
        if not self.const:
            return f"{self.space} {elem}*"
        if self.const_first:
            return f"const {self.space} {elem}*"
        return f"{self.space} const {elem}*"


@dataclass(frozen=True)
class LoadRead:
    """Declared signature of a ``Dot`` face's ``load`` member."""

    input: TypeParam | Pointer = TypeParam("XP")
    scratch: Pointer = Pointer("thread", "float", const=False)
    returns: str | AuxType = "float"
    params: tuple[str, str] = ("x", "xt")
    qualifiers: tuple[str, ...] = ("static",)
    slot: str = "load"

    def __post_init__(self) -> None:
        forbidden: set[str] = set()
        input_tparam: str | None = None
        if isinstance(self.input, TypeParam):
            forbidden.add(self.input.name)
            input_tparam = self.input.name
        elif isinstance(self.input, Pointer):
            if self.input.space not in _ACTIVATION_SPACES:
                raise ValueError(
                    "load input pointer must be in 'device' or 'threadgroup' "
                    f"address space, not {self.input.space!r}"
                )
            if not self.input.const:
                raise ValueError(
                    "load input pointer must point to const data (const=True), "
                    f"not {self.input!r}"
                )
            if isinstance(self.input.element, TypeParam):
                forbidden.add(self.input.element.name)
                input_tparam = self.input.element.name
        else:
            raise ValueError(
                "load input must be TypeParam(...) or a const device/threadgroup "
                f"Pointer(...), not {self.input!r}"
            )
        if (
            not isinstance(self.scratch, Pointer)
            or self.scratch.space != "thread"
            or self.scratch.element != "float"
            or self.scratch.const is not False
        ):
            raise ValueError(
                "load scratch must be Pointer('thread', 'float', const=False), "
                f"not {self.scratch!r}"
            )
        if self.returns == "float":
            pass
        elif isinstance(self.returns, AuxType):
            if input_tparam is not None and self.returns.name == input_tparam:
                raise ValueError(
                    f"load return AuxType {self.returns.name!r} cannot collide "
                    "with input TypeParam"
                )
            forbidden.add(self.returns.name)
        else:
            raise ValueError(
                "load return type must be 'float' or AuxType(...), "
                f"not {self.returns!r}"
            )
        _check_params(self.params, 2, "load", forbidden=frozenset(forbidden))
        _check_qualifiers(
            self.qualifiers,
            "load qualifiers",
            allowed=_INLINE_QUALIFIERS,
            require_static=True,
        )
        _check_slot(self.slot, "load slot")

    def _overload_key(self) -> tuple:
        if isinstance(self.input, TypeParam):
            return ("generic",)
        elem = (
            ("tparam",)
            if isinstance(self.input.element, TypeParam)
            else ("concrete", self.input.element)
        )
        return ("ptr", self.input.space, elem)

    def render(self, indent: str = "") -> str:
        tparam: str | None = None
        if isinstance(self.input, TypeParam):
            tparam = self.input.name
            input_cpp = self.input.name
        else:
            if isinstance(self.input.element, TypeParam):
                tparam = self.input.element.name
            input_cpp = self.input.render()
        ret_cpp = (
            self.returns.name if isinstance(self.returns, AuxType) else self.returns
        )
        quals = " ".join(self.qualifiers)
        sig = (
            f"{quals} {ret_cpp} load("
            f"{_param(input_cpp, self.params[0])}, "
            f"{_param(self.scratch.render(), self.params[1])})"
        )
        if tparam is not None:
            return f"template <typename {tparam}>\n{indent}{sig}"
        return sig


def _check_const_scratch(scratch, member: str) -> None:
    if (
        not isinstance(scratch, Pointer)
        or scratch.space != "thread"
        or scratch.element != "float"
        or scratch.const is not True
    ):
        raise ValueError(
            f"{member} scratch must be a const thread float Pointer, not {scratch!r}"
        )


def _check_aux(aux, member: str) -> tuple[str, str | None, str | None]:
    """Return (aux_cpp, tparam_name, forbidden_name) for madd/apply aux."""
    if aux == "float":
        return "float", None, None
    if isinstance(aux, AuxType):
        return aux.name, None, aux.name
    if isinstance(aux, TypeParam):
        return aux.name, aux.name, aux.name
    raise ValueError(
        f"{member} aux must be 'float', AuxType(...), or TypeParam(...), not {aux!r}"
    )


@dataclass(frozen=True)
class MaddRead:
    """Declared signature of a ``Dot`` face's ``madd`` member."""

    returns: str = "float"
    acc: str = "float"
    row: str = "Row"
    scratch: Pointer = Pointer("thread", "float", const=True)
    aux: str | AuxType | TypeParam = "float"
    params: tuple[str, str, str, str] = ("acc", "r", "xt", "aux")
    qualifiers: tuple[str, ...] = ("static",)
    slot: str = "madd"

    def __post_init__(self) -> None:
        if self.returns != "float":
            raise ValueError(f"madd must return 'float', not {self.returns!r}")
        if self.acc != "float":
            raise ValueError(f"madd accumulator must be 'float', not {self.acc!r}")
        if self.row != "Row":
            raise ValueError(f"madd row must be 'Row', not {self.row!r}")
        _check_const_scratch(self.scratch, "madd")
        _, _, aux_name = _check_aux(self.aux, "madd")
        forbidden = frozenset({aux_name}) if aux_name else frozenset()
        _check_params(self.params, 4, "madd", forbidden=forbidden)
        _check_qualifiers(
            self.qualifiers,
            "madd qualifiers",
            allowed=_INLINE_QUALIFIERS,
            require_static=True,
        )
        _check_slot(self.slot, "madd slot")

    def render(self, indent: str = "") -> str:
        aux_cpp, tparam, _ = _check_aux(self.aux, "madd")
        quals = " ".join(self.qualifiers)
        sig = (
            f"{quals} float madd("
            f"{_param('float', self.params[0])}, "
            f"{_param('Row', self.params[1])}, "
            f"{_param(self.scratch.render(), self.params[2])}, "
            f"{_param(aux_cpp, self.params[3])})"
        )
        if tparam is not None:
            return f"template <typename {tparam}>\n{indent}{sig}"
        return sig


@dataclass(frozen=True)
class ApplyRead:
    """Declared signature of a ``Dot`` face's ``apply`` member, inline or split
    between an in-class declaration and an out-of-line definition in ``shared``."""

    returns: str = "float"
    acc: str = "float"
    decoded: str = "thread const Decoded&"
    scratch: Pointer = Pointer("thread", "float", const=True)
    aux: str | AuxType | TypeParam = "float"
    params: tuple[str, str, str, str] = ("acc", "d", "xt", "aux")
    qualifiers: tuple[str, ...] = ("static",)
    slot: str = "apply"
    shared_slot: str = ""
    shared_owner: str = ""
    shared_templates: tuple[str, ...] = ()
    shared_qualifiers: tuple[str, ...] = ("METAL_FUNC",)
    wrap: bool = False
    wrap_indent: str = "    "

    def __post_init__(self) -> None:
        if self.returns != "float":
            raise ValueError(f"apply must return 'float', not {self.returns!r}")
        if self.acc != "float":
            raise ValueError(f"apply accumulator must be 'float', not {self.acc!r}")
        if not isinstance(self.decoded, str) or self.decoded not in _DECODED_TYPES:
            raise ValueError(
                "apply decoded row must be 'thread const Decoded&' or "
                f"'const thread Decoded&', not {self.decoded!r}"
            )
        _check_const_scratch(self.scratch, "apply")
        _, _, aux_name = _check_aux(self.aux, "apply")
        forbidden = frozenset({aux_name}) if aux_name else frozenset()
        _check_params(self.params, 4, "apply", forbidden=forbidden)
        _check_qualifiers(
            self.qualifiers,
            "apply qualifiers",
            allowed=_INLINE_QUALIFIERS,
            require_static=True,
        )
        _check_slot(self.slot, "apply slot")
        _check_bool(self.wrap, "apply wrap")
        if (
            not isinstance(self.wrap_indent, str)
            or len(self.wrap_indent) > 16
            or not all(ch == " " for ch in self.wrap_indent)
        ):
            raise ValueError(
                f"apply wrap_indent must be up to 16 spaces, not {self.wrap_indent!r}"
            )
        if not isinstance(self.shared_slot, str):
            raise ValueError(
                f"apply shared_slot must be a str, not {self.shared_slot!r}"
            )
        if self.shared_slot == "":
            if (
                self.shared_owner
                or self.shared_templates
                or self.shared_qualifiers != ("METAL_FUNC",)
            ):
                raise ValueError(
                    "apply shared_owner, shared_templates, and shared_qualifiers "
                    "require shared_slot"
                )
            return
        _check_slot(self.shared_slot, "apply shared_slot")
        if self.shared_slot == self.slot:
            raise ValueError(
                f"apply shared_slot {self.shared_slot!r} must differ from slot"
            )
        if not isinstance(self.shared_templates, tuple):
            raise ValueError(
                "apply shared_templates must be a tuple of strings, "
                f"not {self.shared_templates!r}"
            )
        template_names: set[str] = set()
        for item in self.shared_templates:
            if not isinstance(item, str):
                raise ValueError(
                    f"apply shared_templates entry must be str, not {item!r}"
                )
            match = _SHARED_TEMPLATE_RE.fullmatch(item)
            if not match:
                raise ValueError(
                    "apply shared_templates entries must be "
                    "'typename <Ident>', 'class <Ident>', or 'int <Ident>', "
                    f"not {item!r}"
                )
            tname = match.group(2)
            _check_ident(tname, "apply shared_templates parameter")
            if tname in template_names:
                raise ValueError(
                    f"apply shared_templates has duplicate parameter {tname!r}"
                )
            if aux_name is not None and tname == aux_name:
                raise ValueError(
                    f"apply shared_templates parameter {tname!r} cannot "
                    "collide with aux"
                )
            if tname in self.params:
                raise ValueError(
                    f"apply shared_templates parameter {tname!r} cannot "
                    "collide with parameter name"
                )
            template_names.add(tname)
        if not isinstance(self.shared_owner, str) or not self.shared_owner:
            raise ValueError(
                "apply shared_owner must name the enclosing struct when "
                f"shared_slot is set, not {self.shared_owner!r}"
            )
        segments = self.shared_owner.split("::")
        for seg in segments:
            match = _OWNER_SEG_RE.fullmatch(seg)
            if not match:
                raise ValueError(
                    f"apply shared_owner {self.shared_owner!r} is not a valid "
                    "qualified struct name"
                )
            cls_name, arg = match.group(1), match.group(2)
            _check_ident(cls_name, "apply shared_owner segment", allow_contract=True)
            if arg is not None and arg not in template_names:
                raise ValueError(
                    f"apply shared_owner template argument {arg!r} is not "
                    "declared in shared_templates"
                )
        _check_qualifiers(
            self.shared_qualifiers,
            "apply shared_qualifiers",
            allowed=_OUT_OF_LINE_QUALIFIERS,
            require_static=False,
        )

    def _args(self) -> str:
        aux_cpp, _, _ = _check_aux(self.aux, "apply")
        return (
            f"{_param('float', self.params[0])}, "
            f"{_param(self.decoded, self.params[1])}, "
            f"{_param(self.scratch.render(), self.params[2])}, "
            f"{_param(aux_cpp, self.params[3])})"
        )

    def render(self, indent: str = "") -> str:
        _, tparam, _ = _check_aux(self.aux, "apply")
        quals = " ".join(self.qualifiers)
        head = f"{quals} float apply("
        args = self._args()
        body = f"{head}\n{indent}{self.wrap_indent}{args}" if self.wrap else head + args
        if tparam is not None:
            return f"template <typename {tparam}>\n{indent}{body}"
        return body

    def render_shared(self, indent: str = "") -> str:
        _, tparam, _ = _check_aux(self.aux, "apply")
        lines = [f"template <{item}>" for item in self.shared_templates]
        if tparam is not None:
            lines.append(f"template <typename {tparam}>")
        prefix = " ".join(self.shared_qualifiers)
        head = (
            f"{prefix} float {self.shared_owner}::apply("
            if prefix
            else f"float {self.shared_owner}::apply("
        )
        args = self._args()
        if self.wrap:
            lines.append(head)
            lines.append(f"{self.wrap_indent}{args}")
        else:
            lines.append(head + args)
        return f"\n{indent}".join(lines)


@dataclass(frozen=True)
class DotReads:
    """Declared read-member signatures of a ``Dot`` face."""

    load: LoadRead | tuple[LoadRead, ...] = LoadRead()
    madd: MaddRead = MaddRead()
    apply: ApplyRead | None = None

    def __post_init__(self) -> None:
        if isinstance(self.load, LoadRead):
            loads: tuple[LoadRead, ...] = (self.load,)
        elif (
            isinstance(self.load, tuple)
            and self.load
            and all(isinstance(item, LoadRead) for item in self.load)
        ):
            loads = self.load
        else:
            raise ValueError(
                "DotReads.load must be a LoadRead or a non-empty tuple with at "
                f"least one LoadRead, not {self.load!r}"
            )
        object.__setattr__(self, "load", loads)
        if not isinstance(self.madd, MaddRead):
            raise ValueError(f"DotReads.madd must be a MaddRead, not {self.madd!r}")
        if self.apply is not None and not isinstance(self.apply, ApplyRead):
            raise ValueError(
                f"DotReads.apply must be an ApplyRead or None, not {self.apply!r}"
            )
        seen_slots: set[str] = set()
        seen_overloads: set[tuple] = set()
        for ld in loads:
            if ld.slot in seen_slots:
                raise ValueError(f"DotReads has duplicate slot {ld.slot!r}")
            seen_slots.add(ld.slot)
            key = ld._overload_key()
            if key in seen_overloads:
                raise ValueError(
                    f"DotReads has duplicate load input signature ({ld.input!r})"
                )
            seen_overloads.add(key)
        for member in (self.madd, self.apply):
            if member is None:
                continue
            if member.slot in seen_slots:
                raise ValueError(f"DotReads has duplicate slot {member.slot!r}")
            seen_slots.add(member.slot)
        returns = {ld.returns for ld in loads}
        if len(returns) != 1:
            raise ValueError(
                "all load overloads of a DotReads must declare the same return "
                f"type, not {sorted(repr(r) for r in returns)}"
            )
        (load_returns,) = returns
        for name, member in (("madd", self.madd), ("apply", self.apply)):
            if member is None or isinstance(member.aux, TypeParam):
                continue
            if member.aux != load_returns:
                raise ValueError(
                    f"{name} takes {member.aux!r}, which load does not return "
                    f"(load returns {load_returns!r})"
                )


@dataclass(frozen=True)
class WeightRead:
    """Declared signature of a ``Tile`` face's ``weight`` member."""

    returns: str = "float"
    params_type: str = "thread const P&"
    codes: str = "Codes"
    index: str = "int"
    params: tuple[str, str, str] = ("p", "q", "i")
    qualifiers: tuple[str, ...] = ("static",)
    slot: str = "weight"

    def __post_init__(self) -> None:
        if self.returns != "float":
            raise ValueError(f"weight must return 'float', not {self.returns!r}")
        if (
            not isinstance(self.params_type, str)
            or self.params_type not in _WEIGHT_PARAMS_TYPES
        ):
            raise ValueError(
                "weight params must be 'thread const P&' or 'const thread P&', "
                f"not {self.params_type!r}"
            )
        if not isinstance(self.codes, str) or self.codes not in _WEIGHT_CODES_TYPES:
            raise ValueError(
                "weight codes must be 'Codes', 'thread const Codes&', or "
                f"'const thread Codes&', not {self.codes!r}"
            )
        if self.index != "int":
            raise ValueError(f"weight index must be 'int', not {self.index!r}")
        _check_params(self.params, 3, "weight")
        _check_qualifiers(
            self.qualifiers,
            "weight qualifiers",
            allowed=_INLINE_QUALIFIERS,
            require_static=True,
        )
        _check_slot(self.slot, "weight slot")

    def render(self, indent: str = "") -> str:
        quals = " ".join(self.qualifiers)
        return (
            f"{quals} float weight("
            f"{_param(self.params_type, self.params[0])}, "
            f"{_param(self.codes, self.params[1])}, "
            f"{_param('int', self.params[2])})"
        )


@dataclass(frozen=True)
class TileReads:
    """Declared read-member signature of a ``Tile`` face."""

    weight: WeightRead = WeightRead()

    def __post_init__(self) -> None:
        if not isinstance(self.weight, WeightRead):
            raise ValueError(
                f"TileReads.weight must be a WeightRead, not {self.weight!r}"
            )


@dataclass(frozen=True)
class PairRead:
    """Declared signature of a ``Codes`` face's ``pair`` member."""

    returns: str = "uint"
    pack: str | AuxType = "uint"
    index: str = "int"
    params: tuple[str, str] = ("q", "j")
    qualifiers: tuple[str, ...] = ("static",)
    slot: str = "pair"

    def __post_init__(self) -> None:
        if self.returns != "uint":
            raise ValueError(f"pair must return 'uint', not {self.returns!r}")
        forbidden: set[str] = set()
        if isinstance(self.pack, AuxType):
            forbidden.add(self.pack.name)
        elif isinstance(self.pack, str) and self.pack in _PACK_BUILTINS:
            pass
        elif (
            isinstance(self.pack, str)
            and self.pack.isascii()
            and self.pack.isidentifier()
            and self.pack not in _RESERVED
        ):
            forbidden.add(self.pack)
        else:
            raise ValueError(
                "pair pack must be an unsigned Metal integer/vector type or a "
                f"by-value struct identifier, not {self.pack!r}"
            )
        if self.index != "int":
            raise ValueError(f"pair index must be 'int', not {self.index!r}")
        _check_params(self.params, 2, "pair", forbidden=frozenset(forbidden))
        _check_qualifiers(
            self.qualifiers,
            "pair qualifiers",
            allowed=_INLINE_QUALIFIERS,
            require_static=True,
        )
        _check_slot(self.slot, "pair slot")

    def render(self, indent: str = "") -> str:
        pack_cpp = self.pack.name if isinstance(self.pack, AuxType) else self.pack
        quals = " ".join(self.qualifiers)
        return (
            f"{quals} uint pair("
            f"{_param(pack_cpp, self.params[0])}, "
            f"{_param('int', self.params[1])})"
        )


@dataclass(frozen=True)
class CodesReads:
    """Declared read-member signature of a ``Codes`` face."""

    pair: PairRead = PairRead()

    def __post_init__(self) -> None:
        if not isinstance(self.pair, PairRead):
            raise ValueError(f"CodesReads.pair must be a PairRead, not {self.pair!r}")


Reads = DotReads | TileReads | CodesReads


def _expand_slot(
    owner: str, part: str, text: str, marker: str, render, *, decl_only: bool
) -> str:
    """Replace ``marker`` in ``text`` once, requiring line-start placement and
    a legal declaration/definition suffix on its line."""
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(marker)}(?![A-Za-z0-9_])"
    matches = list(re.finditer(pattern, text))
    if not matches:
        raise ValueError(f"{owner}: {part} is missing read slot {marker}")
    if len(matches) > 1:
        raise ValueError(f"{owner}: {part} repeats read slot {marker}")
    at = matches[0].start()
    line_start = text.rfind("\n", 0, at) + 1
    prefix = text[line_start:at]
    if not all(ch in " \t" for ch in prefix):
        raise ValueError(
            f"{owner}: read slot {marker} in {part} must start its line "
            f"(only leading whitespace may precede it, found {prefix!r})"
        )
    end = at + len(marker)
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    suffix = text[end:line_end].strip()
    if decl_only:
        if suffix != ";":
            raise ValueError(
                f"{owner}: read slot {marker} in {part} declares an out-of-line "
                f"member and must end with ';', not {suffix!r}"
            )
    elif suffix != "" and not suffix.startswith("{"):
        raise ValueError(
            f"{owner}: read slot {marker} in {part} must end with '{{' or a "
            f"line break, not {suffix!r}"
        )
    elif not text[end:].lstrip().startswith("{"):
        raise ValueError(
            f"{owner}: read slot {marker} in {part} must be followed by '{{'"
        )
    emitted = render(prefix)
    return text[:at] + emitted + text[end:]


def emit_parts(
    face, raw_parts: tuple[tuple[str, str, str], ...]
) -> tuple[tuple[str, str, str], ...]:
    """Validate ``face.reads`` and emit its declared C++ signatures into
    ``raw_parts`` once at face construction.

    A face must declare its read members: a face without ``reads`` is refused
    at construction, never compiled.
    """
    from . import Codes, Dot, Tile

    owner = f"quant format face {face.key!r}"
    if isinstance(face, Dot):
        expected = DotReads
    elif isinstance(face, Tile):
        expected = TileReads
    elif isinstance(face, Codes):
        expected = CodesReads
    else:
        return raw_parts

    reads = face.reads
    if reads is None:
        raise ValueError(
            f"{owner}: a face declares its read members: pass "
            f"reads={expected.__name__}(...) so oMLX emits its read signatures "
            "(docs/quant-formats.md)"
        )

    if not isinstance(reads, expected):
        raise ValueError(
            f"{owner}: reads must be {expected.__name__}, "
            f"not {type(reads).__name__}"
        )

    by_name = {part: text for part, _, text in raw_parts}
    source = by_name["source"]

    if isinstance(reads, DotReads):
        for ld in reads.load:
            source = _expand_slot(
                owner,
                "source",
                source,
                slot_marker(ld.slot),
                ld.render,
                decl_only=False,
            )
        source = _expand_slot(
            owner,
            "source",
            source,
            slot_marker(reads.madd.slot),
            reads.madd.render,
            decl_only=False,
        )
        if reads.apply is None:
            if (type(face.vpt) is int and face.vpt > 0) or by_name.get("shared"):
                raise ValueError(
                    f"{owner}: Dot face with vpt > 0 or shared source must "
                    "declare DotReads.apply"
                )
        if reads.apply is not None:
            has_shared = bool(reads.apply.shared_slot)
            source = _expand_slot(
                owner,
                "source",
                source,
                slot_marker(reads.apply.slot),
                reads.apply.render,
                decl_only=has_shared,
            )
            if has_shared:
                shared = by_name.get("shared", "")
                if not shared:
                    raise ValueError(
                        f"{owner}: apply declares shared_slot "
                        f"{reads.apply.shared_slot!r}, but face.shared is empty"
                    )
                by_name["shared"] = _expand_slot(
                    owner,
                    "shared",
                    shared,
                    slot_marker(reads.apply.shared_slot),
                    reads.apply.render_shared,
                    decl_only=False,
                )
    elif isinstance(reads, TileReads):
        source = _expand_slot(
            owner,
            "source",
            source,
            slot_marker(reads.weight.slot),
            reads.weight.render,
            decl_only=False,
        )
    elif isinstance(reads, CodesReads):
        source = _expand_slot(
            owner,
            "source",
            source,
            slot_marker(reads.pair.slot),
            reads.pair.render,
            decl_only=False,
        )

    by_name["source"] = source
    emitted = tuple((part, phase, by_name[part]) for part, phase, _ in raw_parts)
    for part, _, text in emitted:
        if _SLOT_PREFIX in text:
            raise ValueError(
                f"{owner}: unexpanded read slot remaining in {part} "
                f"({_SLOT_PREFIX}...)"
            )
    return emitted
