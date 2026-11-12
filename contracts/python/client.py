"""Redis Streams client shared by both repos.

Delivery model
--------------
Producers ``XADD``. Workers read through a consumer group (``XREADGROUP``) and ``XACK``
only after the work is done, so a worker that dies mid-render leaves its message pending
and :meth:`BrokerClient.claim_stale` hands it to another worker. That is **at-least-once**:
a job can be delivered twice, which is why :class:`~.messages.JobRequest` carries an
``idempotency_key``.

The SSE tail is deliberately *not* a consumer group. Every browser must see every event
for its job, so it uses plain ``XREAD`` from a caller-held cursor, which also makes
reconnection a matter of replaying from ``Last-Event-ID``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import redis
from redis.exceptions import ResponseError

from .capabilities import ProviderCapabilities
from .messages import (
    CompletedEvent,
    FailedEvent,
    JobRequest,
    Message,
    ProgressEvent,
    Status,
    utcnow,
)
from .streams import DEFAULT_TASK, EVENT_STREAMS, Group, Stream, StreamNames, job_stream

logger = logging.getLogger(__name__)

#: Approximate cap per stream. Streams are a transport, not the system of record: the job
#: status hash outlives them, so trimming only costs replay depth for a reconnecting UI.
DEFAULT_MAXLEN = 10_000

#: How long a job status hash sticks around after the job ends.
STATUS_TTL_SECONDS = 7 * 24 * 3600

#: Fields of the status hash whose values are JSON rather than plain strings.
_JSON_STATUS_FIELDS = frozenset({"output", "error", "provider"})

#: How long an advertised capability document survives without a refresh. Must exceed
#: the worker's heartbeat interval comfortably, so a brief hiccup does not make a
#: provider look gone.
CAPABILITIES_TTL_SECONDS = 120

#: How long a cancel request is kept. Long enough to outlive any job it could apply to.
CANCEL_TTL_SECONDS = 24 * 3600


@dataclass(frozen=True, slots=True)
class ConsumedMessage:
    """One stream entry, with the stream it came from and its id for acking or resuming."""

    stream: str
    id: str
    payload: dict[str, Any]
    kind: str
    job_id: str


@dataclass(frozen=True, slots=True)
class Enqueued:
    job_id: str
    stream_id: str | None
    #: True when ``idempotency_key`` had already been claimed and nothing new was written.
    deduplicated: bool


class BrokerClient:
    """Thin, synchronous wrapper. One instance per process is enough; it is thread-safe
    to the same degree as the underlying ``redis.Redis`` connection pool.
    """

    def __init__(
        self,
        url: str | None = None,
        prefix: str | None = None,
        client: redis.Redis | None = None,
        maxlen: int | None = None,
    ):
        self.names = StreamNames(prefix)
        self.maxlen = maxlen or int(os.environ.get("STREAM_MAXLEN") or DEFAULT_MAXLEN)
        if client is not None:
            self.redis = client
        else:
            self.redis = redis.Redis.from_url(
                url or os.environ.get("REDIS_URL") or "redis://127.0.0.1:6379/0",
                decode_responses=True,
                socket_keepalive=True,
                health_check_interval=30,
            )

    # ------------------------------------------------------------------ producing

    def publish_job(self, job: JobRequest) -> Enqueued:
        """Enqueue a generation job and open its status snapshot in one round trip.

        With an ``idempotency_key``, the first caller to claim the key wins and later
        callers get that job id back without a second message being written.

        The job goes to the stream of its ``task`` (``jobs.generate`` when unset); an
        unknown task raises ValueError before anything is written.
        """
        stream = job_stream(job.task)
        if job.idempotency_key:
            key = self.names.idempotency(job.idempotency_key)
            if not self.redis.set(key, job.job_id, nx=True, ex=STATUS_TTL_SECONDS):
                existing = self.redis.get(key)
                logger.info("job %s deduplicated onto %s", job.job_id, existing)
                return Enqueued(job_id=existing or job.job_id, stream_id=None, deduplicated=True)

        now = utcnow()
        pipe = self.redis.pipeline(transaction=True)
        pipe.hset(
            self.names.job_status(job.job_id),
            mapping={
                "job_id": job.job_id,
                "status": Status.QUEUED.value,
                "project_id": job.project_id,
                "shot_id": job.shot_id,
                "task": job.task or DEFAULT_TASK.value,
                "stage": "queued",
                "percent": "0",
                "attempt": "1",
                "requested_by": job.requested_by or "",
                "created_at": now,
                "updated_at": now,
            },
        )
        pipe.expire(self.names.job_status(job.job_id), STATUS_TTL_SECONDS)
        pipe.zadd(self.names.job_index(), {job.job_id: _epoch(now)})
        pipe.xadd(
            self.names(stream),
            self._entry(job),
            maxlen=self.maxlen,
            approximate=True,
        )
        stream_id = pipe.execute()[-1]
        logger.info("enqueued job %s (%s/%s) as %s", job.job_id, job.project_id, job.shot_id, stream_id)
        return Enqueued(job_id=job.job_id, stream_id=stream_id, deduplicated=False)

    def publish_progress(self, event: ProgressEvent) -> str:
        return self._publish_event(
            Stream.EVENTS_PROGRESS,
            event,
            {
                "status": Status.RUNNING.value,
                "stage": event.stage.value,
                "percent": str(event.percent),
                "message": event.message or "",
                "attempt": str(event.attempt),
            },
        )

    def publish_completed(self, event: CompletedEvent) -> str:
        payload = event.to_dict()
        return self._publish_event(
            Stream.EVENTS_COMPLETED,
            event,
            {
                "status": Status.COMPLETED.value,
                "stage": "finalizing",
                "percent": "100",
                "message": "",
                "output": json.dumps(payload["output"]),
                # Recorded so a reader can report what actually ran, which may differ
                # from what the shot file asked for.
                "provider": json.dumps(payload.get("provider") or {}),
            },
        )

    def publish_failed(self, event: FailedEvent) -> str:
        return self._publish_event(
            Stream.EVENTS_FAILED,
            event,
            {
                "status": Status.FAILED.value,
                "attempt": str(event.attempts),
                "error": json.dumps(event.to_dict()["error"]),
            },
        )

    def _publish_event(
        self, stream: Stream, event: Message, status_patch: dict[str, str]
    ) -> str:
        """XADD the event, then fold it into the job's status hash.

        The event is written first: a crash between the two leaves a reader that tails
        events correct and only the snapshot stale, which is the recoverable direction.
        """
        stream_id = self.redis.xadd(
            self.names(stream), self._entry(event), maxlen=self.maxlen, approximate=True
        )
        key = self.names.job_status(event.job_id)
        pipe = self.redis.pipeline(transaction=True)
        pipe.hset(key, mapping={**status_patch, "updated_at": utcnow(), "last_event_id": stream_id})
        pipe.expire(key, STATUS_TTL_SECONDS)
        pipe.execute()
        return stream_id

    def _entry(self, message: Message) -> dict[str, str]:
        """Stream entry: the payload as one JSON field plus flat fields for filtering."""
        return {"data": json.dumps(message.to_dict(), separators=(",", ":")), **message.index_fields()}

    # ------------------------------------------------------------------ consuming

    def ensure_groups(self, *streams: Stream, group: Group) -> None:
        """Create the group (and the stream) if absent. Safe to call on every start-up."""
        for stream in streams:
            try:
                self.redis.xgroup_create(self.names(stream), group.value, id="0", mkstream=True)
                logger.info("created group %s on %s", group.value, self.names(stream))
            except ResponseError as error:
                if "BUSYGROUP" not in str(error):
                    raise

    def consume(
        self,
        *streams: Stream,
        group: Group,
        consumer: str,
        count: int = 10,
        block_ms: int = 5000,
    ) -> list[ConsumedMessage]:
        """Read undelivered entries for this group. Returns [] when the block expires.

        Streams are passed to Redis in the given order, which for the event streams must
        be :data:`~.streams.EVENT_STREAMS` so progress is drained ahead of terminal events.
        """
        try:
            response = self.redis.xreadgroup(
                group.value,
                consumer,
                {self.names(s): ">" for s in streams},
                count=count,
                block=block_ms,
            )
        except ResponseError as error:
            if "NOGROUP" in str(error):
                self.ensure_groups(*streams, group=group)
                return []
            raise
        return _decode(response)

    def claim_stale(
        self,
        stream: Stream,
        *,
        group: Group,
        consumer: str,
        min_idle_ms: int = 60_000,
        count: int = 10,
    ) -> list[ConsumedMessage]:
        """Take over entries another consumer read but never acked (it crashed or hung).

        Call this periodically from the worker loop; without it a dead worker's jobs sit
        pending forever.
        """
        _cursor, entries, _deleted = self.redis.xautoclaim(
            self.names(stream),
            group.value,
            consumer,
            min_idle_time=min_idle_ms,
            count=count,
        )
        return _decode([(self.names(stream), entries)])

    def ack(self, stream: str, group: Group, *ids: str) -> None:
        if ids:
            self.redis.xack(stream, group.value, *ids)

    def tail_events(
        self, cursors: dict[str, str], count: int = 100, block_ms: int = 15_000
    ) -> list[ConsumedMessage]:
        """Group-less read used by the SSE endpoint, so every browser sees every event.

        ``cursors`` maps stream key to the last id that reader has seen; pass ``"$"`` to
        start from now. Returns [] when the block expires, which the caller uses as its
        heartbeat tick.
        """
        response = self.redis.xread(cursors, count=count, block=block_ms)
        return _decode(response)

    def event_cursors(self, start: str = "$") -> dict[str, str]:
        """Starting cursors for :meth:`tail_events` over all three event streams."""
        return {self.names(s): start for s in EVENT_STREAMS}

    def event_tips(self) -> dict[str, str]:
        """Cursors pointing at the newest entry currently in each event stream.

        Used to open a tail with no gap: capture the tips *before* reading a job's
        snapshot, and every event that lands from then on is delivered. An event written
        between the two reads is delivered twice rather than lost, which is the
        recoverable direction for a UI that folds events idempotently.
        """
        pipe = self.redis.pipeline(transaction=False)
        keys = [self.names(stream) for stream in EVENT_STREAMS]
        for key in keys:
            pipe.xrevrange(key, count=1)
        tips = {}
        for key, entries in zip(keys, pipe.execute(), strict=True):
            tips[key] = entries[0][0] if entries else "0-0"
        return tips

    # ------------------------------------------------------------------ status reads

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        """The ``JobStatus`` snapshot, or None once it has expired or never existed."""
        raw = self.redis.hgetall(self.names.job_status(job_id))
        return _decode_status(raw) if raw else None

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Newest jobs first. Ids whose hash has expired are skipped."""
        job_ids = self.redis.zrevrange(self.names.job_index(), 0, max(limit - 1, 0))
        if not job_ids:
            return []
        pipe = self.redis.pipeline(transaction=False)
        for job_id in job_ids:
            pipe.hgetall(self.names.job_status(job_id))
        return [_decode_status(raw) for raw in pipe.execute() if raw]

    # ------------------------------------------------------------------ capabilities

    def publish_capabilities(
        self, capabilities: ProviderCapabilities, ttl_seconds: int | None = None
    ) -> None:
        """Advertise what one provider supports, with a TTL.

        Called by workers on start-up and on a heartbeat. Keyed by provider name, so
        several workers of the same provider keep refreshing one document rather than
        each publishing its own.
        """
        self.redis.set(
            self.names.capabilities(capabilities.provider),
            json.dumps(capabilities.to_dict(), separators=(",", ":")),
            ex=ttl_seconds or CAPABILITIES_TTL_SECONDS,
        )

    def read_capabilities(self) -> list[ProviderCapabilities]:
        """Every provider currently advertising, oldest refresh last.

        Returns [] when no worker has published — the caller must treat capabilities as
        advisory and keep working, since a stopped worker should not make the API reject
        otherwise valid requests.
        """
        keys = sorted(self.redis.scan_iter(match=self.names.capabilities_pattern(), count=100))
        if not keys:
            return []
        documents = []
        for raw in self.redis.mget(keys):
            if not raw:
                continue  # expired between the scan and the read
            try:
                documents.append(ProviderCapabilities.from_dict(json.loads(raw)))
            except (TypeError, ValueError):
                logger.warning("ignoring malformed capability document")
        return sorted(documents, key=lambda d: d.updated_at, reverse=True)

    # ------------------------------------------------------------------ cancellation

    def request_cancel(self, job_id: str, ttl_seconds: int | None = None) -> None:
        """Ask whichever worker runs ``job_id`` to stop. Advisory: a worker that does not
        poll the flag finishes the job normally, and a job not yet picked up is dropped
        by a polling worker as soon as it is read."""
        self.redis.set(self.names.cancel(job_id), utcnow(), ex=ttl_seconds or CANCEL_TTL_SECONDS)

    def cancel_requested(self, job_id: str) -> bool:
        return bool(self.redis.exists(self.names.cancel(job_id)))

    # ------------------------------------------------------------------ health

    def ping(self) -> bool:
        try:
            return bool(self.redis.ping())
        except redis.RedisError:
            return False

    def close(self) -> None:
        self.redis.close()


# ------------------------------------------------------------------ decoding helpers


def _decode(response: Any) -> list[ConsumedMessage]:
    """Flatten the XREAD/XREADGROUP reply, skipping entries this reader cannot parse."""
    messages: list[ConsumedMessage] = []
    for stream, entries in response or []:
        for entry_id, fields in entries or []:
            raw = fields.get("data")
            if raw is None:
                logger.warning("entry %s on %s has no data field", entry_id, stream)
                continue
            try:
                payload = json.loads(raw)
            except ValueError:
                logger.warning("entry %s on %s is not valid JSON; skipping", entry_id, stream)
                continue
            messages.append(
                ConsumedMessage(
                    stream=stream,
                    id=entry_id,
                    payload=payload,
                    kind=fields.get("kind") or payload.get("kind", ""),
                    job_id=fields.get("job_id") or payload.get("job_id", ""),
                )
            )
    return messages


def _decode_status(raw: dict[str, str]) -> dict[str, Any]:
    """Hash values are strings; restore the numbers and JSON blobs the schema declares."""
    status: dict[str, Any] = dict(raw)
    for field_name in _JSON_STATUS_FIELDS:
        if status.get(field_name):
            try:
                status[field_name] = json.loads(status[field_name])
            except ValueError:
                status[field_name] = None
    for field_name in ("percent", "attempt"):
        if status.get(field_name) not in (None, ""):
            try:
                status[field_name] = int(status[field_name])
            except ValueError:
                status[field_name] = None
    for field_name in ("message", "requested_by"):
        if status.get(field_name) == "":
            status[field_name] = None
    # Hashes written by a 1.1 producer have no task; those jobs were all "generate".
    status.setdefault("task", DEFAULT_TASK.value)
    return status


def _epoch(timestamp: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
