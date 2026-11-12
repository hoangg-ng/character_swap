"""What a provider can do, advertised by the worker and consumed by the web app.

This is the piece that lets the Research repo gain a model or a generation parameter
without a single edit in the web app. Workers publish a
:class:`ProviderCapabilities` document on start-up and refresh it on a heartbeat; the web
app reads it to build its form and to validate submitted overrides.

The web app therefore holds **no hardcoded list of generation parameters**. Adding one in
Research means adding a :class:`ParameterSpec` next to the provider that understands it
and restarting the worker; the field appears in the UI on the next page load.

Capabilities are advisory. If no worker has published any (none running, or an older
worker that predates this), the web app falls back to passing overrides through with only
generic sanity checks, so generation never blocks on a missing heartbeat.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Self

from .messages import utcnow
from .streams import CONTRACT_VERSION, DEFAULT_TASK

#: Parameter types a client can render a control for. Anything else should be treated as
#: "string" by a reader that does not recognise it.
PARAMETER_TYPES = ("string", "integer", "number", "boolean", "enum")


def _known(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


@dataclass(slots=True)
class ParameterSpec:
    """One generation parameter a client may override.

    Carries enough for a UI to render a control and for a server to validate a value,
    so neither has to know the parameter exists ahead of time.
    """

    name: str
    type: str = "string"
    #: Human label for the control. Falls back to a prettified ``name``.
    label: str | None = None
    #: One-line explanation shown next to the control.
    help: str | None = None
    #: Numeric bounds, for ``integer`` and ``number``.
    minimum: float | None = None
    maximum: float | None = None
    #: Character limit, for ``string``.
    max_length: int | None = None
    #: Allowed values, for ``enum``.
    values: tuple[str, ...] = ()
    #: Render a textarea rather than a single-line input.
    multiline: bool = False
    #: Which form the control belongs in, when one provider serves several (e.g. the
    #: character-swap "create" options versus its per-shot "shot" edits). ``None`` means
    #: the provider's only form.
    group: str | None = None
    #: Suggested starting value. ``None`` means "take it from the shot file".
    default: Any = None

    def display_label(self) -> str:
        return self.label or self.name.replace("_", " ").capitalize()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        known = _known(cls, data)
        if "values" in known and known["values"] is not None:
            known["values"] = tuple(known["values"])
        return cls(**known)


@dataclass(slots=True)
class ModelSpec:
    """One model a provider can render with."""

    id: str
    label: str | None = None
    #: Preselected in a UI when nothing else is chosen.
    default: bool = False
    #: Parameter specs that apply only to this model. Merged over the provider-level
    #: list by ``name``, so a model can narrow a shared parameter (a shorter duration
    #: range, a different resolution set) without restating the rest.
    parameters: tuple[ParameterSpec, ...] = ()

    def display_label(self) -> str:
        return self.label or self.id

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        known = _known(cls, data)
        known["parameters"] = tuple(
            ParameterSpec.from_dict(p) for p in (known.get("parameters") or []) if isinstance(p, dict)
        )
        return cls(**known)


@dataclass(slots=True)
class ProviderCapabilities:
    """Everything one provider advertises about itself."""

    provider: str
    models: tuple[ModelSpec, ...] = ()
    #: Parameters accepted regardless of model.
    parameters: tuple[ParameterSpec, ...] = ()
    #: Free-form note for the UI, e.g. that this provider is a simulator.
    note: str | None = None
    #: The job task this provider serves (``streams.Task``). Overrides of a job are
    #: validated only against providers of the job's task, so one pipeline's
    #: parameters never leak into another's form.
    task: str = DEFAULT_TASK.value
    updated_at: str = field(default_factory=utcnow)
    contract_version: str = CONTRACT_VERSION

    def default_model(self) -> ModelSpec | None:
        return next(
            (m for m in self.models if m.default), self.models[0] if self.models else None
        )

    def parameters_for(self, model_id: str | None) -> tuple[ParameterSpec, ...]:
        """Provider parameters with the chosen model's narrower specs merged in by name.

        The merge is field-level, not wholesale: a model that restates a parameter only
        to tighten a bound keeps the shared label, help text and type rather than
        silently dropping them.
        """
        model = next((m for m in self.models if m.id == model_id), None)
        if model is None or not model.parameters:
            return self.parameters

        overrides = {spec.name: spec for spec in model.parameters}
        merged = []
        for spec in self.parameters:
            override = overrides.pop(spec.name, None)
            merged.append(merge_specs(spec, override) if override else spec)
        # Anything the model declares that the provider level does not.
        return tuple(merged + list(overrides.values()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "task": self.task,
            "note": self.note,
            "updated_at": self.updated_at,
            "contract_version": self.contract_version,
            "models": [
                {
                    "id": m.id,
                    "label": m.label,
                    "default": m.default,
                    "parameters": [_spec_dict(p) for p in m.parameters],
                }
                for m in self.models
            ],
            "parameters": [_spec_dict(p) for p in self.parameters],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        known = _known(cls, data)
        known["models"] = tuple(
            ModelSpec.from_dict(m) for m in (known.get("models") or []) if isinstance(m, dict)
        )
        known["parameters"] = tuple(
            ParameterSpec.from_dict(p)
            for p in (known.get("parameters") or [])
            if isinstance(p, dict)
        )
        return cls(**known)


def merge_specs(base: ParameterSpec, override: ParameterSpec) -> ParameterSpec:
    """Overlay ``override`` onto ``base``, field by field.

    Works through the serialised form because that is what already encodes "set" versus
    "unset": :func:`_spec_dict` omits a field the author never supplied, so a dict merge
    lets the override contribute exactly what it set and nothing more.
    """
    return ParameterSpec.from_dict({**_spec_dict(base), **_spec_dict(override)})


def _spec_dict(spec: ParameterSpec) -> dict[str, Any]:
    """Serialise a spec, dropping the fields that were never set."""
    out: dict[str, Any] = {"name": spec.name, "type": spec.type}
    if spec.label is not None:
        out["label"] = spec.label
    if spec.help is not None:
        out["help"] = spec.help
    if spec.minimum is not None:
        out["minimum"] = spec.minimum
    if spec.maximum is not None:
        out["maximum"] = spec.maximum
    if spec.max_length is not None:
        out["max_length"] = spec.max_length
    if spec.values:
        out["values"] = list(spec.values)
    if spec.multiline:
        out["multiline"] = True
    if spec.group is not None:
        out["group"] = spec.group
    if spec.default is not None:
        out["default"] = spec.default
    return out


# ------------------------------------------------------------------ validation


class ParameterError(ValueError):
    """A submitted override does not satisfy its advertised spec."""


def validate_override(spec: ParameterSpec, value: Any) -> Any:
    """Check one value against its spec, returning the value to send.

    Lives in the contract rather than in either app so both sides agree on what a spec
    means — the web app rejects a bad value up front and the worker can re-check it.
    """
    label = spec.display_label()

    if spec.type == "boolean":
        if not isinstance(value, bool):
            raise ParameterError(f"'{spec.name}' ({label}) must be true or false")
        return value

    if spec.type in ("integer", "number"):
        # bool is a subclass of int, so it has to be excluded explicitly.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ParameterError(f"'{spec.name}' ({label}) must be a number")
        if spec.type == "integer" and not float(value).is_integer():
            raise ParameterError(f"'{spec.name}' ({label}) must be a whole number")
        number = int(value) if spec.type == "integer" else float(value)
        if spec.minimum is not None and number < spec.minimum:
            raise ParameterError(f"'{spec.name}' ({label}) must be at least {_num(spec.minimum)}")
        if spec.maximum is not None and number > spec.maximum:
            raise ParameterError(f"'{spec.name}' ({label}) must be at most {_num(spec.maximum)}")
        return number

    if spec.type == "enum":
        if value not in spec.values:
            allowed = ", ".join(spec.values)
            raise ParameterError(f"'{spec.name}' ({label}) must be one of: {allowed}")
        return value

    # Unknown types are treated as strings, so a reader on an older contract still works.
    if not isinstance(value, str):
        raise ParameterError(f"'{spec.name}' ({label}) must be a string")
    if spec.max_length is not None and len(value) > spec.max_length:
        raise ParameterError(
            f"'{spec.name}' ({label}) is longer than the {spec.max_length} character limit"
        )
    return value


def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)
