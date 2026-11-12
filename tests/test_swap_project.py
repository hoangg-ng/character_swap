"""utils/swap_project.py: the logic shared by webui.py and the worker."""
from __future__ import annotations

import itertools
import json

import pytest
from conftest import make_character, make_project

from utils import swap_project as sp


# ---------- apply_shot_edit: same result as webui.rerender_shot before the refactor ----------
def webui_rerender_edit(e: dict, note, identity, change_light, light, keep, new_take) -> dict:
    """The edit-merging lines of webui.rerender_shot as they were before utils/swap_project.py existed."""
    IDENT = {"same": (None, None), "stronger": (0.10, 14), "strongest": (0.07, 10)}
    KEEP = ["unchanged", "the video's clothes", "her own clothes"]
    e = dict(e) if isinstance(e, dict) else {"note": str(e)}
    if (note or "").strip():
        e["note"] = note.strip()
    sc, px = IDENT.get(identity or "same", (None, None))
    if sc:
        e["soft_face_scale"], e["soft_face_px"] = sc, px
    if change_light:
        e["relight"] = float(light)
    if keep == KEEP[1]:
        e["keep_outfit"] = True
    elif keep == KEEP[2]:
        e["keep_outfit"] = False
    if new_take or not (note or "").strip():
        e["take"] = int(e.get("take", 0)) + 1
    return e


CASES_EDIT = list(itertools.product(
    ["", "  she smiles softly  ", None],
    ["same", "stronger", "strongest", None],
    [False, True],
    [0.5],
    ["unchanged", "the video's clothes", "her own clothes"],
    [False, True],
))


@pytest.mark.parametrize("note,identity,change_light,light,keep,new_take", CASES_EDIT)
def test_apply_shot_edit_matches_webui(dirs, note, identity, change_light, light, keep, new_take):
    cases, _ = dirs
    previous = {"note": "old", "take": 2, "relight": 1.0}
    gen = cases / "demo1" / "output" / "gen"
    gen.mkdir(parents=True)
    (gen / "shot_edits.json").write_text(json.dumps({"3": previous, "1": {"take": 1}}))

    e = sp.apply_shot_edit("demo1", 3, note, identity, change_light, light, keep, new_take)

    assert e == webui_rerender_edit(previous, note, identity, change_light, light, keep, new_take)
    saved = json.loads((gen / "shot_edits.json").read_text(encoding="utf-8"))
    assert saved["3"] == e and saved["1"] == {"take": 1}


@pytest.mark.parametrize("clothes,expected", [("video", True), ("own", False), ("unchanged", None)])
def test_apply_shot_edit_web_clothes_values(dirs, clothes, expected):
    e = sp.apply_shot_edit("demo1", 2, "", "same", False, 1.0, clothes, False)
    assert e.get("keep_outfit") == expected
    assert e["take"] == 1                      # empty note: always a new take


def test_apply_shot_edit_creates_the_file(dirs):
    cases, _ = dirs
    e = sp.apply_shot_edit("demo1", 1, "nụ cười nhẹ", "stronger", True, 0.5, "video", True)
    assert e == {"note": "nụ cười nhẹ", "soft_face_scale": 0.1, "soft_face_px": 14, "relight": 0.5,
                 "keep_outfit": True, "take": 1}
    text = (cases / "demo1" / "output" / "gen" / "shot_edits.json").read_text(encoding="utf-8")
    assert "nụ cười nhẹ" in text               # ensure_ascii=False, like webui


# ---------- progress ----------
LOG = """===== 14:03:01 ONLY=all
[case] /x/cases/demo1  video=video.mp4  gen=832x480  job=gen  out=final
[sheet] character sheet of giang in her own clothes -> input/sheet_giang_own
auto cuts: 33,92
target check shot 1: target visible in 90% of frames -> generate
target check shot 2: target visible in 0% of frames -> KEEP SOURCE
target check shot 3: target visible in 80% of frames -> generate
shot 1: generating on http://127.0.0.1:8188
shot 3: generating on http://127.0.0.1:8189
shot 3: done
"""


def test_stage_of_follows_webui_stages():
    assert sp.stage_of("") == (None, 0)
    assert sp.stage_of(LOG) == ("Generating shots", 4)
    assert sp.stage_of(LOG + "[assembled] x\n[facerefine] ref=a\n[done] x") == ("Done", 7)


def test_parse_progress_stages_and_counts():
    assert sp.parse_progress("===== 14:03:01 ONLY=all\n") == ("resolving", 2, "Starting")
    head = LOG.split("auto cuts")[0]
    assert sp.parse_progress(head) == ("resolving", 5, "Building the character sheet")
    assert sp.parse_progress(LOG.split("shot 1: generating")[0])[:2] == ("resolving", 15)
    assert sp.parse_progress(LOG) == ("rendering", 50, "Generating shots (1/2)")
    done = LOG + "shot 1: done\n"
    assert sp.parse_progress(done) == ("rendering", 80, "Generating shots (2/2)")
    assert sp.parse_progress(done + "[assembled] x\n") == ("finalizing", 85, "Assembling the video")
    assert sp.parse_progress(done + "[assembled] x\n[facerefine] r\n") == ("finalizing", 92, "Refining faces")


def test_parse_progress_counts_only_the_requested_shots():
    log = LOG.replace("shot 1: generating on http://127.0.0.1:8188\n", "").replace("shot 3: done\n", "")
    assert sp.parse_progress(log, only="3") == ("rendering", 20, "Generating shots (0/1)")
    assert sp.parse_progress(log + "shot 3: done\n", only="3") == ("rendering", 80, "Generating shots (1/1)")


def test_clean_log_lines_drops_noise():
    lines = sp.clean_log_lines("a\n  running... 12s\r\nb\n\n\nc", keep=2)
    assert lines == ["b", "c"]


# ---------- planner answer -> settings ----------
def test_plan_to_fields():
    assert sp.plan_to_fields({"possible": True, "note": "", "new_take": False, "identity": "same", "relight": None,
                              "keep_outfit": None}) == {
        "note": "", "identity": "same", "change_light": False, "relight_strength": 1.0, "clothes": "unchanged",
        "new_take": False}
    assert sp.plan_to_fields({"note": "she smiles", "identity": "stronger", "relight": 0.5,
                              "keep_outfit": True}) == {
        "note": "she smiles", "identity": "stronger", "change_light": True, "relight_strength": 0.5,
        "clothes": "video", "new_take": True}
    assert sp.plan_to_fields({"keep_outfit": False, "new_take": True})["clothes"] == "own"
    assert sp.plan_to_fields({"keep_outfit": False, "new_take": True})["new_take"] is True


# ---------- case preparation ----------
def test_prepare_case_and_pipeline_env(dirs, source_video, monkeypatch):
    cases, chars = dirs
    c = make_character(chars)
    (c / "_upload").mkdir()                    # a pending upload folder is not copied
    make_project(cases, source_video, keep=True, q="720", protect="the man", seed=11)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    m = sp.prepare_case("demo1")

    d = cases / "demo1"
    assert sorted(p.name for p in (d / "input").iterdir()) == [
        "giang_cutout.png", "giang_details.txt", "giang_face.png", "video.mp4"]
    assert (d / "case.env").read_text() == (
        "# Swap character web UI project\nVIDEO=${VIDEO:-input/video.mp4}\nCUTS=${CUTS:-auto}\n"
        "CROP=${CROP-auto}\nTARGET_TEXT=${TARGET_TEXT:-\"the woman in the 'red' dress\"}\nPRESENCE=${PRESENCE:-head}\n")
    env = sp.pipeline_env("demo1", m, only="3")
    assert env["CASE"] == str(d) and env["JOB"] == "gen" and env["OUT"] == "final"
    assert env["SHEET_PERSON"] == "input/giang_cutout.png"
    assert env["SHEET_FACE"] == env["FACE_REFINE_REF"] == "input/giang_face.png"
    assert (env["KEEP_OUTFIT"], env["UPSCALE"], env["RELIGHT"], env["PROTECT"], env["PROTECT_TEXT"]) == (
        "1", "1.5", "1", "the man", "the man")
    assert env["SEED"] == "11" and env["ONLY"] == "3"
    assert env["CHAR_REF"] == env["CHAR_REF_EXTRA"] == env["CHAR_REF_FACE"] == ""
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_pipeline_env_webui_project_without_seed(dirs, source_video, monkeypatch):
    cases, chars = dirs
    make_character(chars)
    make_project(cases, source_video)
    monkeypatch.delenv("SEED", raising=False)
    m = sp.meta("demo1")
    del m["seed"]                              # projects made by webui.py have no seed: run_swap.sh's default
    env = sp.pipeline_env("demo1", m)
    assert "SEED" not in env
    assert env["UPSCALE"] == "0" and env["ONLY"] == ""


def test_project_ids_and_names(dirs):
    cases, _ = dirs
    (cases / "demo2").mkdir(); (cases / "demo10").mkdir()
    assert sp.next_project() == "demo11"
    assert sp.project_dir("demo2") == (cases / "demo2").resolve()
    assert sp.project_dir("../demo2") is None and sp.project_dir("") is None and sp.project_dir("..") is None
    assert sp.normalize_name("  Giang Nguyễn!! ") == "giang_nguyen"
    assert sp.normalize_name("Đào Lê") == "dao_le"
