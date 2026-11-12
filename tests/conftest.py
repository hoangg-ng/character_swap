"""Fixtures for the swap_project / worker tests: temporary project and character folders, a fake Redis, and the
fake pipeline (tests/fake_run_swap.sh). No GPU, ComfyUI or real Redis needed; ffmpeg and bash must be on PATH."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import fakeredis  # noqa: E402

from contracts.python import BrokerClient, Stream  # noqa: E402
from contracts.python.client import ConsumedMessage  # noqa: E402
from contracts.python.messages import JobRequest  # noqa: E402
from contracts.python.streams import EVENT_STREAMS, job_stream  # noqa: E402
from utils import swap_project as sp  # noqa: E402
from worker.config import Config  # noqa: E402
from worker.loop import ROLE_STREAMS, Worker  # noqa: E402

BASH = shutil.which("bash")
FFMPEG = shutil.which("ffmpeg")
FAKE_RUN = REPO / "tests" / "fake_run_swap.sh"


@pytest.fixture(scope="session")
def source_video(tmp_path_factory) -> Path:
    """72 frames at 24 fps, 160x120 (the fake pipeline cuts it at 24 and 48)."""
    if not FFMPEG:
        pytest.skip("ffmpeg not on PATH")
    out = tmp_path_factory.mktemp("media") / "video.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=24",
                    "-frames:v", "72", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


@pytest.fixture
def dirs(tmp_path, monkeypatch) -> tuple[Path, Path]:
    cases, chars = tmp_path / "cases", tmp_path / "characters"
    cases.mkdir(); chars.mkdir()
    monkeypatch.setattr(sp, "CASES", cases)
    monkeypatch.setattr(sp, "CHARS", chars)
    return cases, chars


@pytest.fixture
def broker() -> BrokerClient:
    return BrokerClient(client=fakeredis.FakeRedis(decode_responses=True), prefix="test")


def make_character(chars: Path, name: str = "giang", details: str = "Female, long black hair") -> Path:
    d = chars / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "cutout.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (d / "face.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (d / "details.txt").write_text(details + "\n", encoding="utf-8")
    return d


def make_project(cases: Path, video: Path, proj: str = "demo1", char: str = "giang", **meta) -> Path:
    """A project as the web-app creates it: input/video.mp4 + project.json (spec section 1)."""
    d = cases / proj
    (d / "input").mkdir(parents=True)
    shutil.copy2(video, d / "input" / "video.mp4")
    m = {"char": char, "details": "Female, long black hair", "who": 'the woman in the "red" dress', "keep": False,
         "q": "480", "relight": True, "protect": "", "seed": 7, "created": "2026-11-12 14:03", "source": "web-app",
         "jobs": []}
    m.update(meta)
    (d / "project.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return d


def make_config(dirs: tuple[Path, Path], role: str, **kw) -> Config:
    if not BASH:
        pytest.skip("bash not on PATH")
    cases, chars = dirs
    defaults = dict(role=role, cases_dir=cases, chars_dir=chars, run_cmd=[BASH, str(FAKE_RUN)], worker_name="test",
                    cancel_poll_seconds=0.2, progress_interval=0.0, block_ms=10)
    defaults.update(kw)
    return Config(**defaults)


def enqueue(broker: BrokerClient, task: str, project_id: str, shot_id: str,
            overrides: dict | None = None) -> tuple[str, ConsumedMessage]:
    """Publish a job the way the web-app does and read it back as the worker's consumer group would."""
    job = JobRequest(project_id=project_id, shot_id=shot_id, task=task, overrides=overrides)
    broker.publish_job(job)
    role = "render" if job_stream(task) == Stream.JOBS_CHARACTER_SWAP else "util"
    stream, group = ROLE_STREAMS[role]
    broker.ensure_groups(stream, group=group)
    messages = broker.consume(stream, group=group, consumer="test", count=1, block_ms=10)
    assert len(messages) == 1
    return job.job_id, messages[0]


def events(broker: BrokerClient, job_id: str) -> list[dict]:
    """Every event published for job_id, progress first, in stream order."""
    out = []
    for stream in EVENT_STREAMS:
        for _id, fields in broker.redis.xrange(broker.names(stream)):
            data = json.loads(fields["data"])
            if data.get("job_id") == job_id:
                out.append(data)
    return out


def run_job(worker: Worker, broker: BrokerClient, task: str, project_id: str, shot_id: str,
            overrides: dict | None = None) -> tuple[str, dict]:
    job_id, msg = enqueue(broker, task, project_id, shot_id, overrides)
    worker.handle(msg)
    return job_id, broker.get_status(job_id)
