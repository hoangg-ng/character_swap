"""Render role: jobs.character_swap run through the fake pipeline (tests/fake_run_swap.sh)."""
from __future__ import annotations

import json
import threading
import time

import pytest
from conftest import enqueue, events, make_character, make_config, make_project, run_job

from contracts.python.capabilities import ProviderCapabilities
from contracts.python.messages import Stage
from utils import swap_project as sp
from worker.loop import Worker
from worker.render import RenderHandler

TASK = "character_swap"


@pytest.fixture
def worker(dirs, broker, source_video) -> Worker:
    cases, chars = dirs
    make_character(chars)
    make_project(cases, source_video)
    return Worker(make_config(dirs, "render"), broker=broker)


def test_whole_video_job(worker, broker, dirs):
    cases, _ = dirs
    job_id, status = run_job(worker, broker, TASK, "demo1", "all", {"quality": "480p"})

    assert status["status"] == "completed", status
    assert status["task"] == "character_swap"
    out = status["output"]
    assert out["filename"] == "swapped.mp4" and out["path"] == "demo1/output/final/swapped.mp4"
    assert out["manifest"] == "demo1/manifest.json" and out["shots"] == 3
    assert out["resolution"] == "160x120" and out["bytes"] > 0 and out["duration_seconds"] == pytest.approx(3, 0.1)

    d = cases / "demo1"
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert man["project_id"] == "demo1" and man["job_id"] == job_id and man["final"] == "output/final/swapped.mp4"
    assert [s["swapped"] for s in man["shots"]] == [True, False, True]
    assert [(s["start"], s["end"]) for s in man["shots"]] == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    for s in man["shots"]:
        assert (d / s["clip"]).exists() and (d / s["thumb"]).exists() and s["edit"] == {}

    assert (d / "case.env").exists() and (d / "input" / "giang_cutout.png").exists()
    log = (d / "output" / "run.log").read_text(encoding="utf-8")
    assert "ONLY=all" in log and "[done]" in log
    assert "SEED=7 ONLY= KEEP_OUTFIT=0 UPSCALE=0 RELIGHT=1" in log     # options came from project.json

    progress = [e for e in events(broker, job_id) if e["kind"] == "event.progress"]
    order = list(Stage)
    assert [order.index(Stage(e["stage"])) for e in progress] == sorted(order.index(Stage(e["stage"])) for e in progress)
    assert [e["percent"] for e in progress] == sorted(e["percent"] for e in progress)
    messages = [e.get("message") for e in progress]
    assert messages[0] == "Preparing the project" and progress[0]["stage"] == "resolving"
    # lines that arrive together are folded into one event, so not every stage shows up with the fake pipeline
    assert "Generating shots (1/2)" in messages and "Generating shots (2/2)" in messages
    assert messages[-1] == "Cutting the shots" and progress[-1]["percent"] == 99


def test_shot_job_applies_the_edit(worker, broker, dirs):
    cases, _ = dirs
    run_job(worker, broker, TASK, "demo1", "all")
    edit = {"note": "she smiles softly", "identity": "stronger", "change_light": True, "relight_strength": 0.5,
            "clothes": "video", "new_take": True}
    job_id, status = run_job(worker, broker, TASK, "demo1", "3", edit)

    assert status["status"] == "completed", status
    d = cases / "demo1"
    expected = {"note": "she smiles softly", "soft_face_scale": 0.1, "soft_face_px": 14, "relight": 0.5,
                "keep_outfit": True, "take": 1}
    assert json.loads((d / "output" / "gen" / "shot_edits.json").read_text(encoding="utf-8")) == {"3": expected}
    log = sp.last_run_log("demo1")
    assert log.startswith("===== ") and "ONLY=3" in log
    assert "shot 1: generating" not in log
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert man["job_id"] == job_id and man["shots"][2]["edit"] == expected


def test_shot_job_on_a_kept_shot_is_refused(worker, broker):
    _, status = run_job(worker, broker, TASK, "demo1", "2")          # no run yet: unknown
    assert status["status"] == "failed" and status["error"]["code"] == "invalid_shot"
    run_job(worker, broker, TASK, "demo1", "all")
    _, status = run_job(worker, broker, TASK, "demo1", "2")          # shot 2 is kept from the source
    assert status["error"]["code"] == "invalid_shot" and "kept from the original video" in status["error"]["message"]
    _, status = run_job(worker, broker, TASK, "demo1", "x")
    assert status["error"]["code"] == "invalid_shot"


def test_missing_project_or_character(worker, broker, dirs):
    _, status = run_job(worker, broker, TASK, "demo9", "all")
    assert status["status"] == "failed" and status["error"]["code"] == "shot_not_found"
    cases, chars = dirs
    (chars / "giang" / "cutout.png").unlink()
    _, status = run_job(worker, broker, TASK, "demo1", "all")
    assert status["error"]["code"] == "shot_not_found" and "giang" in status["error"]["message"]


def test_pipeline_failure_reports_the_log_tail(worker, broker, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "1")
    _, status = run_job(worker, broker, TASK, "demo1", "all")
    assert status["status"] == "failed"
    assert status["error"]["code"] == "internal_error" and status["error"]["retryable"] is False
    assert "exit 1" in status["error"]["message"] and "boom: the sheet model ran out of memory" in status["error"]["message"]


def test_cancel_stops_the_pipeline(worker, broker, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "30")
    stopped = []
    monkeypatch.setattr(sp, "stop_comfy_jobs", lambda proj: stopped.append(proj))
    job_id, msg = enqueue(broker, TASK, "demo1", "all")
    t = threading.Thread(target=worker.handle, args=(msg,))
    started = time.monotonic()
    t.start()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        st = broker.get_status(job_id) or {}
        if st.get("stage") == "rendering":
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"never reached rendering: {broker.get_status(job_id)}")
    broker.request_cancel(job_id)
    t.join(timeout=30)

    assert not t.is_alive()
    assert time.monotonic() - started < 25
    status = broker.get_status(job_id)
    assert status["status"] == "failed" and status["error"]["code"] == "cancelled"
    handler: RenderHandler = worker.handler      # type: ignore[assignment]
    assert handler.last_proc is not None and handler.last_proc.poll() is not None
    assert stopped == ["demo1"]
    assert "[cancelled]" in sp.last_run_log("demo1")


def test_cancel_before_start(worker, broker, dirs):
    cases, _ = dirs
    job_id, msg = enqueue(broker, TASK, "demo1", "all")
    broker.request_cancel(job_id)
    worker.handle(msg)
    assert broker.get_status(job_id)["error"]["code"] == "cancelled"
    assert not (cases / "demo1" / "output" / "run.log").exists()


def test_completed_job_is_replayed_not_rerun(worker, broker, dirs):
    cases, _ = dirs
    job_id, msg = enqueue(broker, TASK, "demo1", "all")
    worker.handle(msg)
    worker.handle(msg)                           # redelivered after a lost ack
    log = (cases / "demo1" / "output" / "run.log").read_text(encoding="utf-8")
    assert log.count("=====") == 1
    completed = [e for e in events(broker, job_id) if e["kind"] == "event.completed"]
    assert len(completed) == 2 and completed[1]["output"]["path"] == "demo1/output/final/swapped.mp4"


def test_render_capabilities(worker, broker):
    worker.advertise()
    (doc,) = broker.read_capabilities()
    assert isinstance(doc, ProviderCapabilities)
    assert doc.provider == "character_swap" and doc.task == "character_swap"
    groups = {}
    for spec in doc.parameters:
        groups.setdefault(spec.group, []).append(spec.name)
    assert groups == {"create": ["quality", "keep_outfit", "relight", "protect", "seed"],
                      "shot": ["note", "identity", "change_light", "relight_strength", "clothes", "new_take"]}
