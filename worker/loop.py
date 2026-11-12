"""The job consumer shared by both roles: reads jobs of its stream, runs the role's handler, reports back.

Modelled on GoldenAI-Video/Research pipeline/worker.py:
  - one consumer group per role (render: jobs.character_swap / swap-workers, util: jobs.character_swap_util /
    swap-util-workers), so the two never take each other's jobs;
  - a job is acked only after its terminal event is published (at-least-once); a job that already completed is
    re-published instead of run again; jobs left pending by a dead worker are claimed after claim_idle_ms;
  - a job is cancelled before it starts if the web-app already asked (BrokerClient.request_cancel).
Differences: one attempt per job (a failed swap is reported, not re-run for another hour), and the capabilities are
re-advertised from a background thread, so they do not lapse while a render blocks the loop for an hour.
"""
from __future__ import annotations

import logging
import signal
import threading
import time
from types import FrameType
from typing import Protocol

from contracts.python import BrokerClient, CompletedEvent, FailedEvent, Group, ProgressEvent, Stream
from contracts.python.client import ConsumedMessage
from contracts.python.messages import ErrorCode, ErrorInfo, JobRequest, Output, ProviderInfo, Stage, Status, utcnow
from utils import swap_project as sp

from . import capabilities as caps
from .config import Config

logger = logging.getLogger("worker")

ROLE_STREAMS: dict[str, tuple[Stream, Group]] = {
    "render": (Stream.JOBS_CHARACTER_SWAP, Group.SWAP_WORKERS),
    "util": (Stream.JOBS_CHARACTER_SWAP_UTIL, Group.SWAP_UTIL_WORKERS),
}

#: Don't call XAUTOCLAIM on every pass; it is a scan.
_CLAIM_EVERY_SECONDS = 30

_STAGE_ORDER = list(Stage)

#: Schema limits of ProgressEvent.message and ErrorInfo.message.
_MAX_PROGRESS_MESSAGE = 500
_MAX_ERROR_MESSAGE = 2000


class JobFailed(Exception):
    """A job ended without a result, for a reason to report rather than crash on."""

    def __init__(self, code: ErrorCode, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class Handler(Protocol):
    def handle(self, job: JobRequest, ctx: JobContext) -> Output: ...


class JobContext:
    """What a handler may do while it runs: report progress (throttled, monotonic) and ask whether the job was
    cancelled."""

    def __init__(self, worker: Worker, job: JobRequest):
        self.worker = worker
        self.job = job
        self._stage = Stage.QUEUED
        self._percent = 0
        self._last: tuple | None = None
        self._sent_at = 0.0

    def report(self, stage: Stage, percent: int, message: str | None = None, force: bool = False) -> None:
        stage = Stage(stage)
        if _STAGE_ORDER.index(stage) < _STAGE_ORDER.index(self._stage):
            stage = self._stage                      # stages are monotonic
        percent = max(self._percent, min(100, max(0, int(percent))))
        message = (message or "")[:_MAX_PROGRESS_MESSAGE] or None
        key = (stage, percent, message)
        if key == self._last:
            return
        now = time.monotonic()
        if not force and stage == self._stage and now - self._sent_at < self.worker.config.progress_interval:
            return                                   # throttled: the next report after the interval carries it
        self.worker.publish_progress(self.job, stage, percent, message)
        self._stage, self._percent, self._last, self._sent_at = stage, percent, key, now

    def cancelled(self) -> bool:
        try:
            return self.worker.broker.cancel_requested(self.job.job_id)
        except Exception:
            logger.warning("job %s: could not read the cancel flag", self.job.job_id, exc_info=True)
            return False


class Worker:
    def __init__(self, config: Config, broker: BrokerClient | None = None, handler: Handler | None = None):
        self.config = config
        sp.configure(config.cases_dir, config.chars_dir)
        self.broker = broker or BrokerClient(url=config.redis_url, prefix=config.stream_prefix)
        self.stream, self.group = ROLE_STREAMS[config.role]
        self.handler = handler or build_handler(config)
        self.capabilities = caps.for_role(config.role)
        self.shutdown = threading.Event()

    # ------------------------------------------------------------------ loop

    def run(self) -> int:
        """Block consuming jobs until a shutdown signal arrives. Returns an exit code."""
        if not self.broker.ping():
            logger.error("cannot reach Redis at %s", self.config.redis_url)
            return 1
        self.broker.ensure_groups(self.stream, group=self.group)
        signal.signal(signal.SIGINT, self._request_shutdown)
        signal.signal(signal.SIGTERM, self._request_shutdown)
        logger.info("worker %s up: role=%s stream=%s cases=%s characters=%s prefix=%s", self.config.worker_name,
                    self.config.role, self.broker.names(self.stream), self.config.cases_dir, self.config.chars_dir,
                    self.config.stream_prefix)
        self.advertise()
        beat = threading.Thread(target=self._heartbeat, name="advertise", daemon=True)
        beat.start()

        next_claim = 0.0
        while not self.shutdown.is_set():
            if time.monotonic() >= next_claim:
                next_claim = time.monotonic() + _CLAIM_EVERY_SECONDS
                for message in self._claim_abandoned():
                    self.handle(message)
                    if self.shutdown.is_set():
                        break
            if self.shutdown.is_set():
                break
            for message in self._read():
                self.handle(message)
                if self.shutdown.is_set():
                    break

        self.broker.close()
        logger.info("worker %s stopped", self.config.worker_name)
        return 0

    def _request_shutdown(self, signum: int, _frame: FrameType | None) -> None:
        logger.info("signal %s received; finishing the current job then exiting", signal.Signals(signum).name)
        self.shutdown.set()

    def _heartbeat(self) -> None:
        while not self.shutdown.wait(self.config.advertise_seconds):
            self.advertise()

    def advertise(self) -> None:
        """Publish what this role accepts. Failures are logged and swallowed: without it the web-app only falls back
        to unchecked overrides."""
        try:
            self.broker.publish_capabilities(self.capabilities)
        except Exception:
            logger.warning("could not publish capabilities", exc_info=True)

    def _read(self) -> list[ConsumedMessage]:
        try:
            return self.broker.consume(self.stream, group=self.group, consumer=self.config.worker_name,
                                       count=1, block_ms=self.config.block_ms)
        except Exception:
            logger.exception("read failed; retrying")
            time.sleep(2)
            return []

    def _claim_abandoned(self) -> list[ConsumedMessage]:
        try:
            claimed = self.broker.claim_stale(self.stream, group=self.group, consumer=self.config.worker_name,
                                              min_idle_ms=self.config.claim_idle_ms, count=1)
        except Exception:
            logger.exception("claim failed; continuing")
            return []
        if claimed:
            logger.warning("claimed %d job(s) abandoned by another worker", len(claimed))
        return claimed

    # ------------------------------------------------------------------ one job

    def handle(self, message: ConsumedMessage) -> None:
        """Process one job to a terminal event, then ack it (every exit path acks)."""
        try:
            job = JobRequest.from_dict(message.payload)
        except (TypeError, ValueError):
            logger.exception("dropping malformed job at %s: %r", message.id, message.payload)
            self._ack(message)
            return

        if self._replay_completed(job):
            self._ack(message)
            return
        ctx = JobContext(self, job)
        if ctx.cancelled():
            self._fail(job, ErrorCode.CANCELLED, "Cancelled before it started.")
            self._ack(message)
            return

        logger.info("job %s: %s %s/%s %s", job.job_id, job.task, job.project_id, job.shot_id,
                    (job.overrides or {}).get("action", ""))
        started_at = utcnow()
        try:
            output = self.handler.handle(job, ctx)
        except JobFailed as failure:
            self._fail(job, failure.code, failure.message)
        except Exception as error:
            logger.exception("job %s: unexpected error", job.job_id)
            self._fail(job, ErrorCode.INTERNAL_ERROR, f"{type(error).__name__}: {error}")
        else:
            self._complete(job, output, started_at)
        self._ack(message)

    def _replay_completed(self, job: JobRequest) -> bool:
        """Re-publish a previous result instead of running again (the ack of a finished job was lost)."""
        status = self.broker.get_status(job.job_id)
        if not status or status.get("status") != Status.COMPLETED.value:
            return False
        output = status.get("output")
        if not isinstance(output, dict) or not output.get("filename"):
            return False
        logger.info("job %s already completed; re-publishing stored result", job.job_id)
        self.broker.publish_completed(CompletedEvent(
            job_id=job.job_id, project_id=job.project_id, shot_id=job.shot_id, output=Output.from_dict(output),
            provider=ProviderInfo(name=self.capabilities.provider), finished_at=status.get("updated_at")))
        return True

    # ------------------------------------------------------------------ publishing

    def publish_progress(self, job: JobRequest, stage: Stage, percent: int, message: str | None) -> None:
        """Advisory: a publish failure is logged and swallowed, never aborting the job."""
        try:
            self.broker.publish_progress(ProgressEvent(job_id=job.job_id, project_id=job.project_id,
                                                       shot_id=job.shot_id, stage=stage, percent=percent,
                                                       message=message))
        except Exception:
            logger.warning("job %s: could not publish progress", job.job_id, exc_info=True)

    def _complete(self, job: JobRequest, output: Output, started_at: str) -> None:
        self.broker.publish_completed(CompletedEvent(
            job_id=job.job_id, project_id=job.project_id, shot_id=job.shot_id, output=output,
            provider=ProviderInfo(name=self.capabilities.provider), started_at=started_at, finished_at=utcnow()))
        logger.info("job %s completed -> %s", job.job_id, output.path or output.filename)

    def _fail(self, job: JobRequest, code: ErrorCode, message: str) -> None:
        message = message if len(message) <= _MAX_ERROR_MESSAGE else "…" + message[-(_MAX_ERROR_MESSAGE - 1):]
        logger.error("job %s failed: %s - %s", job.job_id, code.value, message.splitlines()[0] if message else "")
        try:
            self.broker.publish_failed(FailedEvent(job_id=job.job_id, project_id=job.project_id, shot_id=job.shot_id,
                                                   error=ErrorInfo.of(code, message), attempts=1))
        except Exception:
            logger.exception("job %s: could not publish failure event", job.job_id)

    def _ack(self, message: ConsumedMessage) -> None:
        try:
            self.broker.ack(message.stream, self.group, message.id)
        except Exception:
            logger.exception("could not ack %s; it may be redelivered", message.id)


def build_handler(config: Config) -> Handler:
    if config.role == "render":
        from .render import RenderHandler
        return RenderHandler(config)
    from .util import UtilHandler
    return UtilHandler(config)
