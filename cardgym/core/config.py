"""TOML configuration: loading, typed overrides, strict dataclass sections.

A run configuration is a TOML document of flat sections. Each section is
parsed into a frozen dataclass by :func:`parse_section`, which rejects unknown
keys and coerces values to the annotated field types, so a typo fails at load
time instead of silently selecting a default. The raw document (after
overrides) is what a run records and what resume compares against.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import tomllib
import types
import typing
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args, get_origin

T = TypeVar("T")


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Return a copy of ``raw`` with ``section.key=value`` overrides applied.

    Values are parsed as TOML (``3``, ``1e-4``, ``true``, ``"text"``,
    ``[3, 4]``); anything that is not valid TOML is taken as a bare string.
    Only existing keys can be overridden, so a misspelt override fails.
    """

    raw = copy.deepcopy(raw)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be section.key=value: {override}")
        dotted, source = override.split("=", 1)
        parts = dotted.strip().split(".")
        if len(parts) != 2 or parts[0] not in raw or parts[1] not in raw[parts[0]]:
            raise ValueError(f"Unknown configuration key: {dotted}")
        try:
            value = tomllib.loads(f"value = {source}")["value"]
        except tomllib.TOMLDecodeError:
            value = source
        raw[parts[0]][parts[1]] = value
    return raw


def config_diff(recorded: dict[str, Any], requested: dict[str, Any]) -> list[str]:
    left = _flatten(recorded)
    right = _flatten(requested)
    return [
        f"{key}: recorded={left.get(key)!r} requested={right.get(key)!r}"
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    ]


def dump_toml(raw: dict[str, Any]) -> str:
    """Serialize the flat-section configuration shape deterministically."""

    lines: list[str] = []
    top = {key: value for key, value in raw.items() if not isinstance(value, dict)}
    for key, value in top.items():
        lines.append(f"{key} = {_toml_value(value)}")
    if top:
        lines.append("")
    for section, values in raw.items():
        if not isinstance(values, dict):
            continue
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def parse_section(cls: type[T], data: dict[str, Any] | None, section: str) -> T:
    """Build dataclass ``cls`` from one TOML section, strictly.

    Unknown keys raise. Missing keys take the dataclass default. Values are
    coerced to the annotated type (int, float, bool, str, Literal, tuples,
    ``dict[str, float]``, Optional); a value that does not fit raises with the
    section-qualified key in the message. ``validate()`` runs if defined.
    """

    data = dict(data or {})
    fields = {field.name: field for field in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(f"Unknown keys in [{section}]: {unknown}")
    hints = typing.get_type_hints(cls)
    values = {}
    for name, value in data.items():
        try:
            values[name] = _coerce(value, hints[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"[{section}] {name}: {error}") from None
    instance = cls(**values)
    validate = getattr(instance, "validate", None)
    if callable(validate):
        validate()
    return instance


def section_dict(instance: Any) -> dict[str, Any]:
    """The TOML-ready mapping of a section dataclass (tuples become lists)."""

    def plain(value: Any) -> Any:
        if isinstance(value, tuple):
            return [plain(item) for item in value]
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        return value

    return {
        field.name: plain(getattr(instance, field.name))
        for field in dataclasses.fields(instance)
    }


def _coerce(value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for option in args:
            if option is type(None):
                continue
            try:
                return _coerce(value, option)
            except (TypeError, ValueError) as error:
                errors.append(str(error))
        raise ValueError("; ".join(errors) or f"cannot interpret {value!r}")
    if origin is Literal:
        if value not in args:
            raise ValueError(f"{value!r} is not one of {list(args)}")
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"expected a list, got {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(item, args[0]) for item in value)
        if len(args) != len(value):
            raise ValueError(f"expected {len(args)} items, got {len(value)}")
        return tuple(_coerce(item, kind) for item, kind in zip(value, args))
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"expected a table, got {value!r}")
        key_type, value_type = args
        return {
            _coerce(key, key_type): _coerce(item, value_type)
            for key, item in value.items()
        }
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError(f"expected true/false, got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"expected an integer, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"expected a number, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError(f"expected a string, got {value!r}")
        return value
    raise TypeError(f"unsupported annotation {annotation!r}")


def _flatten(raw: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for section, values in raw.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat[f"{section}.{key}"] = value
        else:
            flat[section] = values
    return flat


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{key} = {_toml_value(item)}" for key, item in value.items())
        return "{ " + inner + " }" if inner else "{}"
    raise TypeError(f"Unsupported TOML value: {value!r}")
