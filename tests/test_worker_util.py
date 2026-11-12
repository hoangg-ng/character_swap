"""Util role: plan_edit and add_character on jobs.character_swap_util (planner and cut-out stubbed)."""
from __future__ import annotations

import json

import pytest
from conftest import make_character, make_config, make_project, run_job

from utils import swap_project as sp
from worker.loop import Worker
from worker.util import UtilHandler

TASK = "character_swap_util"

PLAN = {"possible": True, "explanation": "Giảm hiệu ứng ánh sáng và làm nụ cười nhẹ hơn.",
        "note": "she has a gentle, subtle smile", "new_take": True, "identity": "same", "relight": 0.5,
        "keep_outfit": None}


@pytest.fixture
def calls() -> dict:
    return {"planner": [], "process": []}


@pytest.fixture
def worker(dirs, broker, source_video, calls, monkeypatch) -> Worker:
    cases, chars = dirs
    make_character(chars)
    d = make_project(cases, source_video)
    gen = d / "output" / "gen"
    gen.mkdir(parents=True)
    (gen / "target_check.json").write_text(json.dumps({"1": {"generate": True}, "2": {"generate": False}}))
    monkeypatch.setattr(sp, "plan_inputs", lambda proj, shot: (["f0.jpg", "f1.jpg"], {"shot": shot}))

    def planner(request, frames, info):
        calls["planner"].append((request, frames, info))
        return dict(PLAN)

    def process_character(name, photo, face, details):
        calls["process"].append((name, photo.name, face.name if face else None, details))
        if "nobody" in photo.read_text():
            raise sp.NoPersonFound("No person found in that photo.")
        (chars / name / "cutout.png").write_bytes(b"png")
        return chars / name

    config = make_config(dirs, "util")
    return Worker(config, broker=broker,
                  handler=UtilHandler(config, planner=planner, process_character=process_character))


def test_plan_edit(worker, broker, dirs, calls):
    cases, _ = dirs
    request = "cô ấy trông hơi tái, và cười nhẹ hơn"
    _, status = run_job(worker, broker, TASK, "demo1", "1", {"action": "plan_edit", "request": f"  {request} "})

    assert status["status"] == "completed", status
    assert status["task"] == "character_swap_util"
    out = status["output"]
    assert out["filename"] == "plan_01.json" and out["path"] == "demo1/output/gen/plan_01.json"
    assert out["possible"] is True and out["explanation"] == PLAN["explanation"]
    # the contract's serialisation drops null members at every level: absent means null in output.plan
    assert out["plan"] == {k: v for k, v in PLAN.items() if v is not None}
    assert out["fields"] == {"note": "she has a gentle, subtle smile", "identity": "same", "change_light": True,
                             "relight_strength": 0.5, "clothes": "unchanged", "new_take": True}
    assert calls["planner"] == [(request, ["f0.jpg", "f1.jpg"], {"shot": 1})]
    saved = json.loads((cases / "demo1" / "output" / "gen" / "plan_01.json").read_text(encoding="utf-8"))
    assert saved["request"] == request and saved["fields"] == out["fields"]


def test_plan_edit_refusals(worker, broker, calls):
    _, st = run_job(worker, broker, TASK, "demo1", "2", {"action": "plan_edit", "request": "smile"})
    assert st["error"]["code"] == "invalid_shot"
    _, st = run_job(worker, broker, TASK, "demo1", "1", {"action": "plan_edit", "request": "   "})
    assert st["error"]["code"] == "invalid_shot" and "Describe the change" in st["error"]["message"]
    _, st = run_job(worker, broker, TASK, "demo7", "1", {"action": "plan_edit", "request": "smile"})
    assert st["error"]["code"] == "shot_not_found"
    _, st = run_job(worker, broker, TASK, "demo1", "1", {"action": "dance"})
    assert st["error"]["code"] == "invalid_shot"
    assert calls["planner"] == []


def test_add_character(worker, broker, dirs, calls):
    _, chars = dirs
    up = chars / "linh" / "_upload"
    up.mkdir(parents=True)
    (up / "photo.jpg").write_text("person")
    (up / "face.png").write_text("face")
    (chars / "linh" / "details.txt").write_text("Female, short hair\n")

    _, status = run_job(worker, broker, TASK, "characters", "linh", {"action": "add_character"})

    assert status["status"] == "completed", status
    assert status["output"] == {"filename": "cutout.png", "path": "linh/cutout.png", "character": "linh"}
    assert calls["process"] == [("linh", "photo.jpg", "face.png", None)]
    assert (chars / "linh" / "cutout.png").exists() and not up.exists()
    assert (chars / "linh" / "details.txt").read_text() == "Female, short hair\n"


def test_add_character_without_a_person_removes_it(worker, broker, dirs):
    _, chars = dirs
    up = chars / "empty_room" / "_upload"
    up.mkdir(parents=True)
    (up / "photo.png").write_text("nobody")

    _, status = run_job(worker, broker, TASK, "characters", "empty_room", {"action": "add_character"})

    assert status["status"] == "failed"
    assert status["error"] == {"code": "invalid_shot", "message": "No person found in that photo.", "retryable": False}
    assert not (chars / "empty_room").exists()


def test_add_character_refusals(worker, broker):
    _, st = run_job(worker, broker, TASK, "characters", "ghost", {"action": "add_character"})
    assert st["error"]["code"] == "shot_not_found"
    _, st = run_job(worker, broker, TASK, "characters", "Bad.Name", {"action": "add_character"})
    assert st["error"]["code"] == "invalid_shot"
    _, st = run_job(worker, broker, TASK, "demo1", "linh", {"action": "add_character"})
    assert st["error"]["code"] == "invalid_shot"


def test_util_capabilities(worker, broker):
    worker.advertise()
    (doc,) = broker.read_capabilities()
    assert doc.provider == "character_swap_util" and doc.task == "character_swap_util"
    specs = {s.name: s for s in doc.parameters}
    assert specs["action"].values == ("plan_edit", "add_character") and specs["request"].max_length == 2000
    assert all(s.group is None for s in doc.parameters)
