"""Stream and consumer-group names.

Keys are namespaced by a prefix (default ``video``) so one Redis can host several
environments: ``STREAM_PREFIX=video.staging`` keeps staging traffic off production
streams without any code change.
"""

import os
from enum import StrEnum
from pathlib import Path

CONTRACT_VERSION = (Path(__file__).resolve().parent.parent / "VERSION").read_text().strip()

DEFAULT_PREFIX = "video"


class Stream(StrEnum):
    """Logical stream names, unprefixed. Resolve through :class:`StreamNames`."""

    JOBS_GENERATE = "jobs.generate"
    #: Character swap renders (a whole video, or one shot of it), consumed by the
    #: character_swap repo's render worker. Jobs here run for up to an hour.
    JOBS_CHARACTER_SWAP = "jobs.character_swap"
    #: Short character-swap work that must not wait behind a render: planning a shot
    #: edit, preparing a new character from photos.
    JOBS_CHARACTER_SWAP_UTIL = "jobs.character_swap_util"
    EVENTS_PROGRESS = "events.progress"
    EVENTS_COMPLETED = "events.completed"
    EVENTS_FAILED = "events.failed"


class Task(StrEnum):
    """What a job asks for. Each task has its own job stream and worker pool; the event
    streams are shared, since every event already carries its ``job_id``."""

    #: Render one authored shot (Research). A job without a task is this one.
    GENERATE = "generate"
    #: Replace a person in an uploaded video (character_swap render worker).
    CHARACTER_SWAP = "character_swap"
    #: Character-swap utilities: ``overrides.action`` names the work (character_swap
    #: util worker).
    CHARACTER_SWAP_UTIL = "character_swap_util"


DEFAULT_TASK = Task.GENERATE

#: Where a job of each task is enqueued.
TASK_STREAMS: dict[Task, Stream] = {
    Task.GENERATE: Stream.JOBS_GENERATE,
    Task.CHARACTER_SWAP: Stream.JOBS_CHARACTER_SWAP,
    Task.CHARACTER_SWAP_UTIL: Stream.JOBS_CHARACTER_SWAP_UTIL,
}


def job_stream(task: str | None) -> Stream:
    """The job stream for ``task`` (``None`` means :data:`DEFAULT_TASK`).

    Raises ValueError for a task this contract does not know, so a typo is refused at
    enqueue time rather than sitting on a stream no worker reads.
    """
    try:
        return TASK_STREAMS[Task(task or DEFAULT_TASK)]
    except ValueError:
        known = ", ".join(t.value for t in Task)
        raise ValueError(f"unknown task {task!r}; expected one of: {known}") from None


class Group(StrEnum):
    """Consumer groups. One per role, so adding a reader never steals another's messages."""

    #: Research workers competing for generation jobs.
    RESEARCH_WORKERS = "research-workers"
    #: character_swap render workers (jobs.character_swap).
    SWAP_WORKERS = "swap-workers"
    #: character_swap utility workers (jobs.character_swap_util).
    SWAP_UTIL_WORKERS = "swap-util-workers"
    #: web-app reader that folds events into the job-status snapshot.
    WEBAPP_STATUS = "webapp-status"


#: Order matters. A reader of several event streams must drain them in this order so a
#: progress event is never delivered after the terminal event of the same job within one
#: batch; Redis preserves order inside a stream but gives no cross-stream guarantee.
EVENT_STREAMS: tuple[Stream, ...] = (
    Stream.EVENTS_PROGRESS,
    Stream.EVENTS_COMPLETED,
    Stream.EVENTS_FAILED,
)

TERMINAL_STREAMS: frozenset[Stream] = frozenset(
    {Stream.EVENTS_COMPLETED, Stream.EVENTS_FAILED}
)


class StreamNames:
    """Resolves logical names to real Redis keys under a prefix."""

    def __init__(self, prefix: str | None = None):
        self.prefix = (prefix or os.environ.get("STREAM_PREFIX") or DEFAULT_PREFIX).rstrip(".")

    def __call__(self, stream: Stream) -> str:
        return f"{self.prefix}.{stream}"

    def events(self) -> list[str]:
        """Event stream keys in the mandated drain order."""
        return [self(s) for s in EVENT_STREAMS]

    def job_status(self, job_id: str) -> str:
        """Hash holding the job-status snapshot (``JobStatus`` in messages.schema.json)."""
        return f"{self.prefix}.job:{job_id}"

    def job_index(self) -> str:
        """Sorted set of job ids scored by creation time, newest last."""
        return f"{self.prefix}.jobs.recent"

    def idempotency(self, key: str) -> str:
        """Maps an idempotency key to the job id that first claimed it."""
        return f"{self.prefix}.idem:{key}"

    def capabilities(self, provider: str) -> str:
        """Document advertising what one provider supports. Keyed by provider rather
        than by worker, so N workers of the same provider do not multiply entries."""
        return f"{self.prefix}.capabilities:{provider}"

    def capabilities_pattern(self) -> str:
        """Matches every provider's capability document, for discovery."""
        return f"{self.prefix}.capabilities:*"

    def cancel(self, job_id: str) -> str:
        """Flag set by the producer to ask the worker running ``job_id`` to stop. The
        worker polls it; a stopped job ends with a FailedEvent of code ``cancelled``."""
        return f"{self.prefix}.cancel:{job_id}"


# ------------------------------------------------------------------ SSE cursors
#
# The SSE endpoint tails three streams but the browser may only hand back a single
# ``Last-Event-ID`` on reconnect, so the three Redis ids travel as one opaque token.
# Format: ``progress=<id>;completed=<id>;failed=<id>``. Clients must treat it as opaque
# and echo it back unchanged; only the server parses it.

_CURSOR_KEYS: dict[str, Stream] = {
    "progress": Stream.EVENTS_PROGRESS,
    "completed": Stream.EVENTS_COMPLETED,
    "failed": Stream.EVENTS_FAILED,
}


def encode_cursor(cursors: dict[str, str], names: StreamNames) -> str:
    """Pack per-stream Redis ids into one ``Last-Event-ID`` token."""
    return ";".join(f"{short}={cursors.get(names(stream), '0')}" for short, stream in _CURSOR_KEYS.items())


def decode_cursor(token: str | None, names: StreamNames, default: str = "$") -> dict[str, str]:
    """Unpack a token back into stream-keyed cursors.

    Anything unparsable falls back to ``default`` for that stream rather than raising: a
    mangled or truncated header should cost the client some replay, not the connection.
    """
    cursors = {names(stream): default for stream in _CURSOR_KEYS.values()}
    for part in (token or "").split(";"):
        short, _, value = part.partition("=")
        stream = _CURSOR_KEYS.get(short.strip())
        if stream is not None and _is_stream_id(value.strip()):
            cursors[names(stream)] = value.strip()
    return cursors


def _is_stream_id(value: str) -> bool:
    """Redis stream ids are ``<ms>-<seq>``; ``0`` and ``$`` are also valid cursors."""
    if value in ("0", "$"):
        return True
    ms, _, seq = value.partition("-")
    return ms.isdigit() and seq.isdigit()
