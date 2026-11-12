#!/usr/bin/env python3
"""Character-swap projects: the logic shared by the Gradio UI (webui.py) and the web-app worker (worker/).

A project is a case folder `<CASES>/demoN/` made from an uploaded video and a character of the library
`<CHARS>/<name>/` (cutout.png, face.png|jpg, details.txt). Files of a project:
  project.json      what the user chose (char, details, who, keep, q, relight, protect, seed, created) - written by
                    the UI that created the project (webui.py or the web-app), only read here
  input/            the video (video.<ext>) and the character files copied as <char>_<file>
  case.env          the case settings for scripts/run_swap.sh
  output/           gen/ (per-shot generations, cuts.txt, target_check.json, shot_edits.json, _ui_shots/),
                    final/swapped.mp4, run.log
  manifest.json     the result for the web-app: final video and the shots (start/end, swapped, clip, thumbnail)

The library folders come from SWAP_CASES_DIR / SWAP_CHARS_DIR (default cases/ and characters/ of this repo), so the
Gradio UI and the web-app see the same projects. Heavy imports (cv2, numpy, insightface, the planner) are made inside
the functions that need them: importing this module needs only the standard library.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _dir_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


#: Project and character libraries. Callers that set the folders at run time (the worker, tests) use configure().
CASES = _dir_env("SWAP_CASES_DIR", ROOT / "cases")
CHARS = _dir_env("SWAP_CHARS_DIR", ROOT / "characters")
COMFY = ["http://127.0.0.1:8188", "http://127.0.0.1:8189", "http://127.0.0.1:8190"]

#: Log markers of scripts/run_swap.sh and the stage each one starts (in pipeline order).
STAGES = [("[sheet]", "Building the character sheet"), ("auto cuts", "Finding scene cuts"),
          ("target check", "Checking which shots show the person"), ("generating on", "Generating shots"),
          ("[assembled]", "Assembling the video"), ("[facerefine]", "Refining faces"), ("[done]", "Done")]

#: Identity strength -> how strongly the original face is softened in the guide (soft_face_scale, soft_face_px).
IDENT = {"same": (None, None), "stronger": (0.10, 14), "strongest": (0.07, 10)}
#: Clothes choices as the Gradio UI labels them (the web-app uses "unchanged" / "video" / "own").
KEEP = ["unchanged", "the video's clothes", "her own clothes"]

SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def configure(cases_dir: str | Path | None = None, chars_dir: str | Path | None = None) -> None:
    """Point the module at other project / character folders (worker config, tests)."""
    global CASES, CHARS
    if cases_dir:
        CASES = Path(cases_dir).expanduser().resolve()
    if chars_dir:
        CHARS = Path(chars_dir).expanduser().resolve()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_json(path: Path, data) -> None:
    """Atomic write (temp file + replace), so a reader never sees half a file."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# ---------- projects ----------
def num(p: Path | str) -> int:
    return int(re.sub(r"\D", "", Path(p).name) or 0)


def project_dir(proj: str) -> Path | None:
    """<CASES>/<proj>, or None for an id that is unsafe or resolves outside CASES."""
    if not proj or not SAFE_ID.match(proj):
        return None
    d = (CASES / proj).resolve()
    return d if d.parent == CASES.resolve() else None


def projects() -> list[tuple[str, str]]:
    ds = sorted((p for p in CASES.glob("demo*") if (p / "case.env").exists()), key=num, reverse=True)
    return [(f"Demo {num(p)}", p.name) for p in ds]


def next_project() -> str:
    return f"demo{max((num(p) for p in CASES.glob('demo*')), default=0) + 1}"


def meta(proj: str) -> dict:
    f = CASES / proj / "project.json"
    return json.loads(f.read_text(encoding="utf-8")) if proj and f.exists() else {}


def shot_edits(proj: str) -> dict:
    f = CASES / proj / "output" / "gen" / "shot_edits.json"
    return json.loads(f.read_text(encoding="utf-8")) if proj and f.exists() else {}


def target_check(proj: str) -> dict:
    f = CASES / proj / "output" / "gen" / "target_check.json"
    return json.loads(f.read_text(encoding="utf-8")) if proj and f.exists() else {}


def is_swapped(proj: str, shot: int | str) -> bool:
    """True if the pipeline generated this shot (False: kept from the source, unknown, or not run yet)."""
    return bool(target_check(proj).get(str(shot), {}).get("generate"))


def source_video(proj: str) -> Path | None:
    return next(iter(sorted((CASES / proj / "input").glob("video.*"))), None)


# ---------- characters ----------
def normalize_name(name: str | None) -> str:
    s = (name or "").strip().replace("đ", "d").replace("Đ", "D")
    # Vietnamese and other accented names keep their letters ("Lê" -> "le", not "l").
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")


def characters() -> list[str]:
    return sorted(p.name for p in CHARS.iterdir() if (p / "cutout.png").exists()) if CHARS.exists() else []


def char_face(name: str) -> Path | None:
    return next((p for p in (CHARS / name / "face.png", CHARS / name / "face.jpg") if p.exists()), None)


class NoPersonFound(ValueError):
    """The character photo shows no person to cut out."""


def process_character(name: str, photo: str | Path, face_photo: str | Path | None = None,
                      details: str | None = None) -> Path:
    """Character from photos: <CHARS>/<name>/cutout.png (the person on white), face.png (the largest face, from
    face_photo or the photo) and details.txt. details=None keeps an existing details.txt ("Female" if none).
    Raises NoPersonFound without creating anything when the photo shows no person."""
    import cv2
    import numpy as np

    from utils import tone_match as tm
    from utils.face_refine import _face_app
    name = normalize_name(name)
    if not (name and photo):
        raise ValueError("Give the character a name and a full-body photo.")
    img = cv2.imread(str(photo))
    if img is None:
        raise NoPersonFound("That photo cannot be read.")
    mask, box = tm.person_mask(img)
    if mask is None:
        raise NoPersonFound("No person found in that photo.")
    d = CHARS / name; d.mkdir(parents=True, exist_ok=True)
    out = np.full_like(img, 255); out[mask] = img[mask]
    x0, y0, x1, y1 = box; m = int(0.04 * max(img.shape[:2]))
    cv2.imwrite(str(d / "cutout.png"), out[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m])
    src = cv2.imread(str(face_photo)) if face_photo else img
    if src is None:
        src = img
    fs = _face_app().get(src)
    if fs:
        f = max(fs, key=lambda x: x.bbox[2] - x.bbox[0]); x0, y0, x1, y1 = f.bbox.astype(int); w, h = x1 - x0, y1 - y0
        cv2.imwrite(str(d / "face.png"), src[max(0, y0 - h // 2):y1 + h // 3, max(0, x0 - w // 2):x1 + w // 2])
    if details is not None or not (d / "details.txt").exists():
        (d / "details.txt").write_text((details or "Female").strip() + "\n", encoding="utf-8")
    return d


# ---------- running ----------
def prepare_case(proj: str) -> dict:
    """Make the case folder runnable from project.json: input/ (character files copied as <char>_<file>),
    output/ and case.env. Returns the project's meta."""
    d = CASES / proj
    m = meta(proj)
    char = m["char"]
    (d / "input").mkdir(parents=True, exist_ok=True); (d / "output").mkdir(exist_ok=True)
    video = source_video(proj)
    ext = video.suffix if video else ".mp4"
    for f in (CHARS / char).iterdir():
        if f.is_file():
            shutil.copy2(f, d / "input" / f"{char}_{f.name}")
    who = (m.get("who") or "the woman").strip().replace('"', "'")
    (d / "case.env").write_text(f"# Swap character web UI project\nVIDEO=${{VIDEO:-input/video{ext}}}\nCUTS=${{CUTS:-auto}}\n"
                                f"CROP=${{CROP-auto}}\nTARGET_TEXT=${{TARGET_TEXT:-\"{who}\"}}\nPRESENCE=${{PRESENCE:-head}}\n")
    return m


def pipeline_env(proj: str, m: dict, only: str = "") -> dict:
    """Environment of scripts/run_swap.sh for this project (only = shot numbers to re-render, "" = all)."""
    inp = CASES / proj / "input"
    env = dict(os.environ, CASE=str(CASES / proj), JOB="gen", OUT="final", SHEET_PERSON=f"input/{m['char']}_cutout.png",
               CHAR_DETAILS=m["details"], KEEP_OUTFIT="1" if m.get("keep") else "0",
               UPSCALE="1.5" if str(m.get("q", "480")) == "720" else "0", RELIGHT="1" if m.get("relight") else "0",
               PROTECT=m.get("protect", "") or "", PROTECT_TEXT=m.get("protect", "") or "",
               CHAR_REF="", CHAR_REF_EXTRA="", CHAR_REF_FACE="", ONLY=only)
    if m.get("seed") is not None:
        env["SEED"] = str(int(m["seed"]))
    env.pop("CUDA_VISIBLE_DEVICES", None)
    face = next((f"input/{m['char']}_face{e}" for e in (".png", ".jpg") if (inp / f"{m['char']}_face{e}").exists()), "")
    if face:
        env["SHEET_FACE"] = face; env["FACE_REFINE_REF"] = face
    return env


def open_run_log(proj: str, only: str = ""):
    """output/run.log opened for appending, with the header that starts one run (status parsing reads from the
    last header on)."""
    log = CASES / proj / "output" / "run.log"; log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "a", encoding="utf-8")
    fh.write(f"\n===== {time.strftime('%H:%M:%S')} ONLY={only or 'all'}\n"); fh.flush()
    return fh


def last_run_log(proj: str) -> str:
    log = CASES / proj / "output" / "run.log" if proj else None
    t = log.read_text(encoding="utf-8", errors="replace") if log and log.exists() else ""
    return t[t.rfind("====="):] if "=====" in t else t


def clean_log_lines(text: str, keep: int = 40) -> list[str]:
    """The last `keep` non-empty lines, without the "running... Ns" progress noise."""
    t = re.sub(r"\s*running\.\.\.\s+\d+s", "", text.replace("\r", "\n"))
    return [l for l in t.splitlines() if l.strip()][-keep:]


def apply_shot_edit(proj: str, shot: int | str, note: str | None = None, identity: str | None = "same",
                    change_light: bool = False, light: float | None = 1.0, clothes: str | None = "unchanged",
                    new_take: bool = False) -> dict:
    """Merge one edit into output/gen/shot_edits.json (read by the next run with ONLY=<shot>); returns the shot's
    settings. clothes: "unchanged", "video" / "the video's clothes", "own" / "her own clothes"."""
    f = CASES / proj / "output" / "gen" / "shot_edits.json"
    edits = shot_edits(proj); e = edits.get(str(shot), {})
    e = e if isinstance(e, dict) else {"note": str(e)}
    if (note or "").strip():
        e["note"] = note.strip()
    sc, px = IDENT.get(identity or "same", (None, None))
    if sc:
        e["soft_face_scale"], e["soft_face_px"] = sc, px
    if change_light:
        e["relight"] = float(light if light is not None else 1.0)
    if clothes in (KEEP[1], "video"):
        e["keep_outfit"] = True
    elif clothes in (KEEP[2], "own"):
        e["keep_outfit"] = False
    if new_take or not (note or "").strip():
        e["take"] = int(e.get("take", 0)) + 1
    edits[str(shot)] = e
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(edits, indent=1, ensure_ascii=False), encoding="utf-8")
    return e


# ---------- progress ----------
def stage_of(txt: str) -> tuple[str | None, int]:
    """(label of the latest stage reached, its 1-based index in STAGES); (None, 0) before the first marker."""
    stage, k = None, 0
    for i, (key, label) in enumerate(STAGES):
        if key in txt:
            stage, k = label, i + 1
    return stage, k


#: Contract stage and percent at the start of each STAGES entry; "generating" spreads 20..80 over the shots.
_STAGE_PROGRESS = {1: ("resolving", 5), 2: ("resolving", 10), 3: ("resolving", 15), 4: ("rendering", 20),
                   5: ("finalizing", 85), 6: ("finalizing", 92), 7: ("finalizing", 99)}


def parse_progress(txt: str, only: str = "") -> tuple[str, int, str]:
    """(contract stage, percent, message) of one run's log (the text from its "=====" header on)."""
    label, k = stage_of(txt)
    if not k:
        return "resolving", 2, "Starting"
    stage, pct = _STAGE_PROGRESS[k]
    if k == 4:
        done = len(set(re.findall(r"shot (\d+): done", txt)))
        if only:
            total = len([s for s in only.split(",") if s.strip()])
        else:
            total = len(re.findall(r"target check shot \d+:[^\n]*-> generate\b", txt))
            total = total or len(set(re.findall(r"shot (\d+): generating on", txt)))
        total = max(1, total, done)
        pct = 20 + int(60 * done / total)
        label = f"{label} ({done}/{total})"
    return stage, pct, label


# ---------- results ----------
def cut_shots(proj: str) -> list[dict]:
    """The shots of the final video, cut at the pipeline's shot boundaries into output/gen/_ui_shots/sNN.mp4 (+ a
    320 px .jpg thumbnail from the middle frame). Clips older than the final video are cut again. Each entry:
    n, start, end (seconds), swapped, clip, thumb (Paths, None when missing)."""
    import cv2
    d = CASES / proj
    job, final = d / "output" / "gen", d / "output" / "final" / "swapped.mp4"
    if not (proj and final.exists() and (job / "cuts.txt").exists()):
        return []
    src = next((d / "output").glob("_source_cropped*.mp4"), None) or next((d / "input").glob("video.*"))
    cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 24; n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cuts = [0] + [int(c) for c in (job / "cuts.txt").read_text().strip().split(",") if c] + [n_src]
    tc = target_check(proj)
    sd = job / "_ui_shots"; sd.mkdir(exist_ok=True)
    stamp = final.stat().st_mtime
    out = []
    for i in range(len(cuts) - 1):
        a, b = cuts[i] / fps, cuts[i + 1] / fps
        thumb, clip = sd / f"s{i + 1:02d}.jpg", sd / f"s{i + 1:02d}.mp4"
        if not clip.exists() or clip.stat().st_mtime < stamp:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", str(final),
                            "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-an", str(clip)])
            c2 = cv2.VideoCapture(str(clip)); c2.set(cv2.CAP_PROP_POS_FRAMES, c2.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
            ok, f = c2.read(); c2.release()
            if ok:
                cv2.imwrite(str(thumb), cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1]))))
        out.append(dict(n=i + 1, start=a, end=b, swapped=bool(tc.get(str(i + 1), {}).get("generate")),
                        clip=clip if clip.exists() else None, thumb=thumb if thumb.exists() else None))
    return out


def video_info(path: Path) -> dict:
    """width, height, duration_seconds of a video (cv2), {} if it cannot be read."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 24, cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return {"width": w, "height": h, "duration_seconds": round(n / fps, 3) if n > 0 else None}


def build_manifest(proj: str, job_id: str) -> dict:
    """Cut the shots of the final video and write <project>/manifest.json (the result the web-app shows). Paths are
    relative to the project folder."""
    d = CASES / proj
    final = d / "output" / "final" / "swapped.mp4"
    edits = shot_edits(proj)

    def rel(p: Path | None) -> str | None:
        return p.relative_to(d).as_posix() if p else None

    shots = [dict(n=s["n"], start=round(s["start"], 3), end=round(s["end"], 3), swapped=s["swapped"],
                  clip=rel(s["clip"]), thumb=rel(s["thumb"]),
                  edit=(edits.get(str(s["n"])) if isinstance(edits.get(str(s["n"])), dict) else {}))
             for s in cut_shots(proj)]
    man = {"project_id": proj, "job_id": job_id, "updated_at": utcnow(),
           "final": rel(final) if final.exists() else None, **(video_info(final) if final.exists() else {}),
           "shots": shots}
    _write_json(d / "manifest.json", man)
    return man


# ---------- planning an edit ----------
def plan_inputs(proj: str, shot: int) -> tuple[list[str], dict]:
    """(three frames of the shot's clip as .jpg paths, shot information) for utils.edit_planner.plan."""
    import cv2
    clip = CASES / proj / "output" / "gen" / "_ui_shots" / f"s{shot:02d}.mp4"
    if not clip.exists():
        cut_shots(proj)
    cap = cv2.VideoCapture(str(clip)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); frames = []
    for j, k in enumerate((n // 6, n // 2, 5 * n // 6)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if ok:
            fp = clip.with_name(f"plan_{shot:02d}_{j}.jpg"); cv2.imwrite(str(fp), f); frames.append(str(fp))
    cap.release()
    m = meta(proj)
    info = {"shot": shot, "status": "swapped", "character": m.get("details", ""),
            "current_settings": shot_edits(proj).get(str(shot), {})}
    return frames, info


def plan_to_fields(p: dict) -> dict:
    """A planner answer as the shot-edit settings (note, identity, change_light, relight_strength, clothes,
    new_take)."""
    rl = p.get("relight")
    ko = p.get("keep_outfit")
    return {"note": p.get("note") or "", "identity": p.get("identity") or "same", "change_light": rl is not None,
            "relight_strength": float(rl if rl is not None else 1.0),
            "clothes": "unchanged" if ko is None else ("video" if ko else "own"),
            "new_take": bool(p.get("new_take") or p.get("note"))}


# ---------- stopping ----------
def stop_comfy_jobs(proj: str) -> None:
    """Remove this project's jobs from the ComfyUI queues and interrupt the one running."""
    for u in COMFY:
        try:
            q = json.load(urllib.request.urlopen(u + "/queue", timeout=2))
            mine = lambda it: f"/{proj}/" in json.dumps(it[2]) or f"{proj}_" in json.dumps(it[2])  # noqa: E731
            pend = [it[1] for it in q["queue_pending"] if mine(it)]
            if pend:
                urllib.request.urlopen(urllib.request.Request(u + "/queue", data=json.dumps({"delete": pend}).encode(),
                                                              headers={"Content-Type": "application/json"}))
            if any(mine(it) for it in q["queue_running"]):
                urllib.request.urlopen(urllib.request.Request(u + "/interrupt", data=b"{}",
                                                              headers={"Content-Type": "application/json"}))
        except Exception:
            pass
