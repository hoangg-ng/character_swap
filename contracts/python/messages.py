"""Typed messages, mirroring ``messages.schema.json``.

Plain dataclasses rather than pydantic: the only runtime dependency either repo needs
for the broker is ``redis``. The schema is still the source of truth and the test suite
validates round-trips against it with ``jsonschema``.

Readers follow the compatibility rule in ../README.md: unknown fields are dropped rather
than raising, so a writer on a newer minor version cannot break an older reader.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar, Self, TypeVar

from .streams import CONTRACT_VERSION


def utcnow() -> str:
    """RFC 3339 timestamp in UTC with a trailing Z, as the schema requires."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_job_id() -> str:
    return str(uuid.uuid4())


class Stage(StrEnum):
    QUEUED = "queued"
    RESOLVING = "resolving"
    SUBMITTED = "submitted"
    RENDERING = "rendering"
    DOWNLOADING = "downloading"
    FINALIZING = "finalizing"


class Status(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorCode(StrEnum):
    SHOT_NOT_FOUND = "shot_not_found"
    INVALID_SHOT = "invalid_shot"
    PROVIDER_REJECTED = "provider_rejected"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    QUOTA_EXCEEDED = "quota_exceeded"
    CONTENT_REJECTED = "content_rejected"
    DOWNLOAD_FAILED = "download_failed"
    INTERNAL_ERROR = "internal_error"
    CANCELLED = "cancelled"

    @property
    def retryable(self) -> bool:
        return self in _RETRYABLE


_RETRYABLE = frozenset(
    {
        ErrorCode.PROVIDER_TIMEOUT,
        ErrorCode.PROVIDER_UNAVAILABLE,
        ErrorCode.DOWNLOAD_FAILED,
    }
)

T = TypeVar("T", bound="Message")


@dataclass(slots=True)
class _Nested:
    """Base for the nested value objects, with the same tolerant parsing as Message."""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Self | None:
        if data is None:
            return None
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class Output(_Nested):
    """Where the finished clip is.

    ``extra`` holds any field this version does not know about — a thumbnail URL, a
    second variant — so a provider can report new artifacts without a contract change.
    It is flattened into the payload on the wire, never nested under an "extra" key.
    """

    filename: str
    url: str | None = None
    path: str | None = None
    bytes: int | None = None
    duration_seconds: float | None = None
    resolution: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Output | None:
        """Unlike the other nested types, unknown keys are kept rather than dropped."""
        if data is None:
            return None
        known = {f.name for f in fields(cls)} - {"extra"}
        return cls(
            **{k: v for k, v in data.items() if k in known},
            extra={k: v for k, v in data.items() if k not in known and k != "extra"},
        )


@dataclass(slots=True)
class ProviderInfo(_Nested):
    name: str
    model: str | None = None
    request_id: str | None = None


@dataclass(slots=True)
class ErrorInfo(_Nested):
    code: ErrorCode
    message: str
    retryable: bool = False

    @classmethod
    def of(cls, code: ErrorCode, message: str) -> ErrorInfo:
        """Build an error with ``retryable`` derived from the code."""
        return cls(code=code, message=message, retryable=code.retryable)


@dataclass(slots=True)
class Message:
    """Common envelope behaviour. Subclasses declare ``KIND``."""

    KIND: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        """Schema-shaped dict: enums flattened to strings, unset optionals omitted.

        An unset optional is left out rather than sent as an explicit null, so the
        payload stays small and a reader distinguishes "not provided" from a real null
        only when the writer means it.
        """
        return _plain(asdict(self))

    def index_fields(self) -> dict[str, str]:
        """Flat fields stored next to the payload so a tail can filter without parsing JSON."""
        return {
            "job_id": self.job_id,
            "kind": self.KIND,
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**cls._coerce(kwargs))

    @classmethod
    def _coerce(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        return kwargs


def _plain(value: Any) -> Any:
    """asdict leaves StrEnum members in place; JSON wants their string value.

    Also drops keys whose value is None: ``asdict`` materialises every optional field,
    and the schema's nested objects do not accept an explicit null for an absent value.
    """
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


@dataclass(slots=True)
class JobRequest(Message):
    """web-app -> Research. See ``$defs.JobRequest``."""

    KIND: ClassVar[str] = "job.generate"

    project_id: str
    shot_id: str
    job_id: str = field(default_factory=new_job_id)
    requested_at: str = field(default_factory=utcnow)
    requested_by: str | None = None
    idempotency_key: str | None = None
    priority: int = 5
    #: Shallow overrides on the shot's ``generation`` block. An open mapping on purpose:
    #: a provider may accept parameters this contract has never heard of, and advertises
    #: them through ProviderCapabilities instead of through a schema change.
    overrides: dict[str, Any] | None = None
    #: Which worker pool the job is for (``streams.Task``); decides the job stream.
    #: Unset means ``generate``, and is left off the wire so a generate job is
    #: byte-for-byte what a 1.1 reader expects.
    task: str | None = None
    kind: str = KIND
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        """Normalise overrides however the job was built.

        Nulls are dropped rather than forwarded: a null means "not provided", and
        passing it on would erase the shot file's value downstream.
        """
        if self.overrides is None:
            return
        if not isinstance(self.overrides, dict):
            raise ValueError("overrides must be an object")
        self.overrides = {k: v for k, v in self.overrides.items() if v is not None}


@dataclass(slots=True)
class ProgressEvent(Message):
    """Research -> web-app. Advisory; safe to drop."""

    KIND: ClassVar[str] = "event.progress"

    job_id: str
    stage: Stage
    percent: int
    project_id: str | None = None
    shot_id: str | None = None
    message: str | None = None
    attempt: int = 1
    emitted_at: str = field(default_factory=utcnow)
    kind: str = KIND
    contract_version: str = CONTRACT_VERSION

    @classmethod
    def _coerce(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        if "stage" in kwargs:
            kwargs["stage"] = Stage(kwargs["stage"])
        return kwargs


@dataclass(slots=True)
class CompletedEvent(Message):
    """Research -> web-app. Terminal and authoritative."""

    KIND: ClassVar[str] = "event.completed"

    job_id: str
    output: Output
    project_id: str | None = None
    shot_id: str | None = None
    provider: ProviderInfo | None = None
    started_at: str | None = None
    finished_at: str | None = None
    emitted_at: str = field(default_factory=utcnow)
    kind: str = KIND
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Flatten ``output.extra`` up into ``output``, so unrecognised artifact fields
        travel as ordinary members rather than nested under a bookkeeping key."""
        # Not super(): @dataclass(slots=True) recreates the class, so the zero-arg
        # super()'s __class__ cell points at the pre-slots class and raises.
        payload = Message.to_dict(self)
        output = payload.get("output")
        if isinstance(output, dict):
            extra = output.pop("extra", None) or {}
            payload["output"] = {**output, **extra}
        return payload

    @classmethod
    def _coerce(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        if isinstance(kwargs.get("output"), dict):
            kwargs["output"] = Output.from_dict(kwargs["output"])
        if isinstance(kwargs.get("provider"), dict):
            kwargs["provider"] = ProviderInfo.from_dict(kwargs["provider"])
        return kwargs


@dataclass(slots=True)
class FailedEvent(Message):
    """Research -> web-app. Terminal; published after retries are exhausted."""

    KIND: ClassVar[str] = "event.failed"

    job_id: str
    error: ErrorInfo
    project_id: str | None = None
    shot_id: str | None = None
    attempts: int = 1
    emitted_at: str = field(default_factory=utcnow)
    kind: str = KIND
    contract_version: str = CONTRACT_VERSION

    @classmethod
    def _coerce(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        if isinstance(kwargs.get("error"), dict):
            error = dict(kwargs["error"])
            if "code" in error:
                error["code"] = ErrorCode(error["code"])
            kwargs["error"] = ErrorInfo.from_dict(error)
        return kwargs


EVENT_TYPES: dict[str, type[Message]] = {
    ProgressEvent.KIND: ProgressEvent,
    CompletedEvent.KIND: CompletedEvent,
    FailedEvent.KIND: FailedEvent,
}

MESSAGE_TYPES: dict[str, type[Message]] = {JobRequest.KIND: JobRequest, **EVENT_TYPES}


def parse_event(data: dict[str, Any]) -> ProgressEvent | CompletedEvent | FailedEvent:
    """Dispatch on ``kind``. Raises ValueError for a kind this reader does not know."""
    kind = data.get("kind")
    cls = EVENT_TYPES.get(kind or "")
    if cls is None:
        raise ValueError(f"not an event kind: {kind!r}")
    return cls.from_dict(data)  # type: ignore[return-value]
