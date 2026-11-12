#!/usr/bin/env python3
"""Swap character - web UI for the pipeline (Gradio).

Workflow: upload a source video, say who to replace, pick a character -> "Swap character" runs the standard
pipeline (scripts/run_swap.sh) in a new project (Demo 1, Demo 2, ... = cases/demoN). The result is the final video
and the list of its shots; any swapped shot can be re-rendered with an extra instruction ("edit this shot"), after
which the final video is rebuilt.

Characters live in characters/<name>/ (cutout.png, face.png|jpg, details.txt); new ones can be added from photos.
Start: bash scripts/start_ui.sh
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import cv2
import gradio as gr

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
CHARS = ROOT / "characters"
COMFY = ["http://127.0.0.1:8188", "http://127.0.0.1:8189", "http://127.0.0.1:8190"]
RUNS: dict[str, subprocess.Popen] = {}          # project -> running pipeline
SEEN: dict[str, float] = {}                     # project -> final video mtime shown

CSS = """
:root, .dark { --accent: #3b82f6; --bg: #121212; --panel: #1c1c1e; --panel2: #232326; --line: #2e2e32;
  --text: #f2f2f2; --muted: #9a9aa2; }
body, .gradio-container { background: var(--bg) !important; color: var(--text); }
.gradio-container { max-width: 1440px !important; padding: 0 20px !important; }
#topbar { display: flex; align-items: center; gap: 12px; padding: 14px 4px 10px; border-bottom: 1px solid var(--line);
  margin-bottom: 14px; }
#topbar .logo { width: 30px; height: 30px; border-radius: 8px; background: linear-gradient(135deg, #3b82f6, #a855f7); }
#topbar .name { font-size: 18px; font-weight: 600; letter-spacing: .2px; }
#topbar .sub { color: var(--muted); font-size: 13px; margin-left: 6px; }
.panel { background: var(--panel) !important; border: 1px solid var(--line) !important; border-radius: 12px !important;
  padding: 14px !important; }
.section-title { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; color: var(--muted);
  margin: 4px 0 6px; }
#swap-btn { background: var(--accent) !important; color: white !important; font-weight: 600; font-size: 15px;
  border-radius: 10px !important; min-height: 46px; }
#status { background: var(--panel2); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; }
#status .stage { font-weight: 600; } #status .meta { color: var(--muted); font-size: 13px; }
.bar { height: 6px; background: #2c2c30; border-radius: 4px; margin-top: 8px; overflow: hidden; }
.bar > div { height: 100%; background: var(--accent); transition: width .5s; }
footer { display: none !important; }
"""
STAGES = [("[sheet]", "Building the character sheet"), ("auto cuts", "Finding scene cuts"),
          ("target check", "Checking which shots show the person"), ("generating on", "Generating shots"),
          ("[assembled]", "Assembling the video"), ("[facerefine]", "Refining faces"), ("[done]", "Done")]


# ---------- projects ----------
def _num(p: Path) -> int:
    return int(re.sub(r"\D", "", p.name) or 0)


def projects() -> list[tuple[str, str]]:
    ds = sorted((p for p in CASES.glob("demo*") if (p / "case.env").exists()), key=_num, reverse=True)
    return [(f"Demo {_num(p)}", p.name) for p in ds]


def first_project():
    ps = projects()
    return ps[0][1] if ps else None


def next_project() -> str:
    return f"demo{max((_num(p) for p in CASES.glob('demo*')), default=0) + 1}"


def meta(proj: str) -> dict:
    f = CASES / proj / "project.json"
    return json.loads(f.read_text()) if proj and f.exists() else {}


# ---------- characters ----------
def characters() -> list[str]:
    return sorted(p.name for p in CHARS.iterdir() if (p / "cutout.png").exists()) if CHARS.exists() else []


def char_face(name: str) -> Path | None:
    return next((p for p in (CHARS / name / "face.png", CHARS / name / "face.jpg") if p.exists()), None)


def char_gallery():
    return [(str(char_face(n) or CHARS / n / "cutout.png"), n) for n in characters()]


def add_character(name, photo, face_photo, details):
    import numpy as np
    from utils import tone_match as tm
    from utils.face_refine import _face_app
    name = re.sub(r"[^a-z0-9_]+", "_", (name or "").strip().lower()).strip("_")
    if not (name and photo):
        return "Give the character a name and a full-body photo.", gr.update()
    img = cv2.imread(photo)
    mask, box = tm.person_mask(img)
    if mask is None:
        return "No person found in that photo.", gr.update()
    d = CHARS / name; d.mkdir(parents=True, exist_ok=True)
    out = np.full_like(img, 255); out[mask] = img[mask]
    x0, y0, x1, y1 = box; m = int(0.04 * max(img.shape[:2]))
    cv2.imwrite(str(d / "cutout.png"), out[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m])
    src = cv2.imread(face_photo) if face_photo else img
    fs = _face_app().get(src)
    if fs:
        f = max(fs, key=lambda x: x.bbox[2] - x.bbox[0]); x0, y0, x1, y1 = f.bbox.astype(int); w, h = x1 - x0, y1 - y0
        cv2.imwrite(str(d / "face.png"), src[max(0, y0 - h // 2):y1 + h // 3, max(0, x0 - w // 2):x1 + w // 2])
    (d / "details.txt").write_text((details or "Female").strip() + "\n")
    return f"Added **{name}**.", gr.update(value=char_gallery())


# ---------- running ----------
def pipeline_env(proj: str, m: dict, only: str = "") -> dict:
    inp = CASES / proj / "input"
    env = dict(os.environ, CASE=f"cases/{proj}", JOB="gen", OUT="final", SHEET_PERSON=f"input/{m['char']}_cutout.png",
               CHAR_DETAILS=m["details"], KEEP_OUTFIT="1" if m["keep"] else "0",
               UPSCALE="1.5" if m["q"] == "720" else "0", RELIGHT="1" if m["relight"] else "0",
               PROTECT=m.get("protect", ""), PROTECT_TEXT=m.get("protect", ""),
               CHAR_REF="", CHAR_REF_EXTRA="", CHAR_REF_FACE="", ONLY=only)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    face = next((f"input/{m['char']}_face{e}" for e in (".png", ".jpg") if (inp / f"{m['char']}_face{e}").exists()), "")
    if face:
        env["SHEET_FACE"] = face; env["FACE_REFINE_REF"] = face
    return env


def launch(proj: str, env: dict) -> None:
    log = CASES / proj / "output" / "run.log"; log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "a"); fh.write(f"\n===== {time.strftime('%H:%M:%S')} ONLY={env.get('ONLY') or 'all'}\n"); fh.flush()
    RUNS[proj] = subprocess.Popen(["bash", "scripts/run_swap.sh"], cwd=ROOT, env=env, stdout=fh,
                                  stderr=subprocess.STDOUT, start_new_session=True)


def start_swap(video, who, sel, keep, quality, relight, protect):
    if not video:
        return "Upload a source video.", gr.update()
    if not sel:
        return "Pick a character.", gr.update()
    proj = next_project(); d = CASES / proj
    (d / "input").mkdir(parents=True); (d / "output").mkdir()
    ext = Path(video).suffix or ".mp4"; shutil.copy2(video, d / "input" / f"video{ext}")
    for f in (CHARS / sel).iterdir():
        shutil.copy2(f, d / "input" / f"{sel}_{f.name}")
    who = (who or "the woman").strip().replace('"', "'")
    (d / "case.env").write_text(f"# Swap character web UI project\nVIDEO=${{VIDEO:-input/video{ext}}}\nCUTS=${{CUTS:-auto}}\n"
                                f"CROP=${{CROP-auto}}\nTARGET_TEXT=${{TARGET_TEXT:-\"{who}\"}}\nPRESENCE=${{PRESENCE:-head}}\n")
    m = dict(char=sel, details=(CHARS / sel / "details.txt").read_text().strip(), keep=bool(keep),
             q="720" if quality.startswith("720") else "480", relight=bool(relight), protect=(protect or "").strip(),
             who=who, created=time.strftime("%Y-%m-%d %H:%M"))
    (d / "project.json").write_text(json.dumps(m, indent=1))
    launch(proj, pipeline_env(proj, m))
    return f"Started **Demo {_num(d)}**.", gr.update(choices=projects(), value=proj)


IDENT = {"same": (None, None), "stronger": (0.10, 14), "strongest": (0.07, 10)}
KEEP = ["unchanged", "the video's clothes", "her own clothes"]


def shot_edits(proj: str) -> dict:
    f = CASES / proj / "output" / "gen" / "shot_edits.json"
    return json.loads(f.read_text()) if proj and f.exists() else {}


def plan_edit(proj, shot, request):
    """Free-form request -> editable plan (explanation, note, identity, light, clothes, new take)."""
    nothing = ("", "", "same", False, 1.0, "unchanged", True)
    if not (proj and shot):
        return ("Select a shot first.",) + nothing[1:]
    if not (request or "").strip():
        return ("Describe the change first.",) + nothing[1:]
    tc = json.loads((CASES / proj / "output" / "gen" / "target_check.json").read_text())
    if not tc.get(str(shot), {}).get("generate"):
        return (f"Shot {shot} is kept from the original video (nobody to swap), so it cannot be edited.",) + nothing[1:]
    from utils.edit_planner import plan
    clip = CASES / proj / "output" / "gen" / "_ui_shots" / f"s{shot:02d}.mp4"
    cap = cv2.VideoCapture(str(clip)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); frames = []
    for j, k in enumerate((n // 6, n // 2, 5 * n // 6)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if ok:
            fp = clip.with_name(f"plan_{shot:02d}_{j}.jpg"); cv2.imwrite(str(fp), f); frames.append(str(fp))
    m = meta(proj)
    info = {"shot": shot, "status": "swapped", "character": m.get("details", ""),
            "current_settings": shot_edits(proj).get(str(shot), {})}
    p = plan(request.strip(), frames, info)
    if not p.get("possible", True):
        return (f"**Not possible:** {p.get('explanation', '')}",) + nothing[1:]
    keep = "unchanged" if p.get("keep_outfit") is None else (KEEP[1] if p["keep_outfit"] else KEEP[2])
    rl = p.get("relight")
    return (f"**Plan:** {p.get('explanation', '')}  \nCheck the settings below, adjust if needed, then re-render.",
            p.get("note", ""), p.get("identity", "same"), rl is not None, float(rl if rl is not None else 1.0), keep,
            bool(p.get("new_take") or p.get("note")))


def rerender_shot(proj, shot, note, identity, change_light, light, keep, new_take):
    if not (proj and shot):
        return "Select a shot first."
    if proj in RUNS and RUNS[proj].poll() is None:
        return "This project is still running."
    job = CASES / proj / "output" / "gen"
    tc = json.loads((job / "target_check.json").read_text())
    if not tc.get(str(shot), {}).get("generate"):
        return f"Shot {shot} is kept from the original video (nobody to swap), so there is nothing to re-render."
    edits = shot_edits(proj); e = edits.get(str(shot), {})
    e = e if isinstance(e, dict) else {"note": str(e)}
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
    edits[str(shot)] = e
    (job / "shot_edits.json").write_text(json.dumps(edits, indent=1, ensure_ascii=False))
    launch(proj, pipeline_env(proj, meta(proj), only=str(shot)))
    return f"Re-rendering shot {shot} with {json.dumps(e, ensure_ascii=False)}. The final video is rebuilt afterwards."


def stop(proj):
    p = RUNS.get(proj)
    if not p or p.poll() is not None:
        return "Nothing is running."
    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    for u in COMFY:                                   # remove this project's jobs from the GPU queues
        try:
            q = json.load(urllib.request.urlopen(u + "/queue", timeout=2))
            mine = lambda it: f"/{proj}/" in json.dumps(it[2]) or f"{proj}_" in json.dumps(it[2])
            pend = [it[1] for it in q["queue_pending"] if mine(it)]
            if pend:
                urllib.request.urlopen(urllib.request.Request(u + "/queue", data=json.dumps({"delete": pend}).encode(),
                                                              headers={"Content-Type": "application/json"}))
            if any(mine(it) for it in q["queue_running"]):
                urllib.request.urlopen(urllib.request.Request(u + "/interrupt", data=b"{}",
                                                              headers={"Content-Type": "application/json"}))
        except Exception:
            pass
    return "Stopped."


# ---------- status & results ----------
def last_run_log(proj: str) -> str:
    log = CASES / proj / "output" / "run.log" if proj else None
    t = log.read_text(errors="replace") if log and log.exists() else ""
    return t[t.rfind("====="):] if "=====" in t else t


def status_html(proj: str) -> str:
    if not proj:
        return "<div id='status'><span class='meta'>No project yet. Upload a video and press Swap character.</span></div>"
    txt = last_run_log(proj)
    running = proj in RUNS and RUNS[proj].poll() is None
    stage, k = ("Starting" if running else "Ready"), 0
    for i, (key, label) in enumerate(STAGES):
        if key in txt:
            stage, k = label, i + 1
    if stage == "Generating shots":
        stage += f" ({len(re.findall(r'shot \d+: done', txt))}/{max(1, len(re.findall(r'generating on', txt)))})"
    if not running and txt and "[done]" not in txt:
        stage = "Stopped"
    pct = int(100 * k / len(STAGES))
    m = meta(proj)
    info = (f"{m.get('char', '?')} · replaces “{m.get('who', '')}” · {m.get('q', '480')}p"
            + (" · keeps the video's clothes" if m.get("keep") else "") + (" · running" if running else ""))
    return (f"<div id='status'><div class='stage'>{stage}</div><div class='meta'>{info}</div>"
            f"<div class='bar'><div style='width:{pct}%'></div></div></div>")


def shots(proj: str):
    """([(thumbnail, caption)], [shot numbers]) cut from the final video at the pipeline's shot boundaries."""
    job, final = CASES / proj / "output" / "gen", CASES / proj / "output" / "final" / "swapped.mp4"
    if not (proj and final.exists() and (job / "cuts.txt").exists()):
        return [], []
    src = next((CASES / proj / "output").glob("_source_cropped*.mp4"), None) or next((CASES / proj / "input").glob("video.*"))
    cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 24; n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cuts = [0] + [int(c) for c in (job / "cuts.txt").read_text().strip().split(",") if c] + [n_src]
    tc = json.loads((job / "target_check.json").read_text()) if (job / "target_check.json").exists() else {}
    sd = job / "_ui_shots"; sd.mkdir(exist_ok=True)
    stamp = final.stat().st_mtime
    items, nums = [], []
    for i in range(len(cuts) - 1):
        a, b = cuts[i] / fps, cuts[i + 1] / fps
        thumb, clip = sd / f"s{i + 1:02d}.jpg", sd / f"s{i + 1:02d}.mp4"
        if not clip.exists() or clip.stat().st_mtime < stamp:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", str(final),
                            "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-an", str(clip)])
            c2 = cv2.VideoCapture(str(clip)); c2.set(cv2.CAP_PROP_POS_FRAMES, c2.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
            ok, f = c2.read()
            if ok:
                cv2.imwrite(str(thumb), cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1]))))
        if not thumb.exists():
            continue
        gen = tc.get(str(i + 1), {}).get("generate")
        items.append((str(thumb), f"Shot {i + 1} · {a:.1f}–{b:.1f}s · {'swapped' if gen else 'original'}"))
        nums.append(i + 1)
    return items, nums


def refresh(proj):
    final = CASES / proj / "output" / "final" / "swapped.mp4" if proj else None
    if final and final.exists():
        SEEN[proj] = final.stat().st_mtime
    items, nums = shots(proj) if proj else ([], [])
    return status_html(proj), (str(final) if final and final.exists() else None), items, nums


def tick(proj):
    t = re.sub(r"\s*running\.\.\.\s+\d+s", "", last_run_log(proj).replace("\r", "\n")) if proj else ""
    log = "\n".join([l for l in t.splitlines() if l.strip()][-40:])
    final = CASES / proj / "output" / "final" / "swapped.mp4" if proj else None
    if final and final.exists() and final.stat().st_mtime > SEEN.get(proj, 0) and not (proj in RUNS and RUNS[proj].poll() is None):
        s, f, items, nums = refresh(proj)                  # a run just finished: show the new final and shots
        return s, log, f, items, nums
    return status_html(proj), log, gr.update(), gr.update(), gr.update()


def pick_shot(evt: gr.SelectData, proj, nums):
    i = nums[evt.index] if nums and evt.index < len(nums) else None
    if not i:
        return None, None, ""
    clip = CASES / proj / "output" / "gen" / "_ui_shots" / f"s{i:02d}.mp4"
    e = shot_edits(proj).get(str(i), {})
    note = e.get("note", "") if isinstance(e, dict) else str(e)
    return (str(clip) if clip.exists() else None), i, note


# ---------- layout ----------
def build() -> gr.Blocks:
    theme = gr.themes.Base(primary_hue="blue", neutral_hue="zinc", font=[gr.themes.GoogleFont("Inter"), "system-ui"])
    with gr.Blocks(title="Swap character", theme=theme, css=CSS,
                   js="() => { document.body.classList.add('dark'); }") as ui:
        gr.HTML("<div id='topbar'><div class='logo'></div><div class='name'>Swap character</div>"
                "<div class='sub'>replace a person in a video with your character</div></div>")
        sel_char, shot_nums, cur_shot = gr.State(None), gr.State([]), gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_classes="panel"):
                gr.HTML("<div class='section-title'>1 · Source video</div>")
                video = gr.Video(sources=["upload"], show_label=False, height=240)
                gr.HTML("<div class='section-title'>2 · Who to replace</div>")
                who = gr.Textbox(show_label=False, placeholder="e.g. the woman in the red dress", lines=1)
                gr.HTML("<div class='section-title'>3 · Character</div>")
                gal = gr.Gallery(value=char_gallery(), show_label=False, columns=4, height=160, allow_preview=False,
                                 object_fit="cover")
                chosen = gr.Markdown("*Pick a character above.*")
                with gr.Accordion("Add a character", open=False):
                    cn = gr.Textbox(label="Name")
                    with gr.Row():
                        cp = gr.Image(type="filepath", label="Full-body photo", height=160)
                        cf = gr.Image(type="filepath", label="Face photo (optional)", height=160)
                    cd = gr.Textbox(label="Description (head to toe)", lines=2,
                                    placeholder="Female, slim, long black hair, ... wearing ...")
                    cb = gr.Button("Add character"); cm = gr.Markdown()
                with gr.Accordion("Options", open=False):
                    keep = gr.Checkbox(label="Keep the video's clothes", value=False)
                    quality = gr.Radio(["480p · quick test", "720p · final"], value="480p · quick test", label="Quality")
                    relight = gr.Checkbox(label="Re-light to the scene", value=True)
                    protect = gr.Textbox(label="People to keep unchanged (optional)", placeholder="e.g. the man")
                go = gr.Button("Swap character", elem_id="swap-btn")
                msg = gr.Markdown()
            with gr.Column(scale=7, elem_classes="panel"):
                with gr.Row():
                    proj = gr.Dropdown(projects(), value=first_project(), label="Project", scale=4)
                    rb = gr.Button("Refresh", scale=1); sb = gr.Button("Stop", variant="stop", scale=1)
                status = gr.HTML(status_html(first_project()))
                final = gr.Video(label="Final video", height=380, interactive=False)
                gr.HTML("<div class='section-title'>Shots · select one to edit</div>")
                shotgal = gr.Gallery(show_label=False, columns=5, height=230, allow_preview=False, object_fit="cover")
                with gr.Row(equal_height=False):
                    shotvid = gr.Video(label="Selected shot", height=220, interactive=False, scale=5)
                    with gr.Column(scale=5):
                        req = gr.Textbox(label="Edit this shot (any language)", lines=2,
                                         placeholder="e.g. she looks too pale and not like Giang, softer smile")
                        pb = gr.Button("Plan edit")
                        pmsg = gr.Markdown()
                        with gr.Accordion("Edit settings", open=True):
                            note = gr.Textbox(label="Appearance note (English, added to this shot's prompt)", lines=2)
                            ident = gr.Radio(list(IDENT), value="same", label="Identity strength")
                            with gr.Row():
                                chl = gr.Checkbox(label="Change light", value=False)
                                light = gr.Slider(0, 1, value=1.0, step=0.1, label="Re-light strength")
                            keepc = gr.Radio(KEEP, value="unchanged", label="Clothes")
                            take = gr.Checkbox(label="New take (another variation)", value=True)
                        rr = gr.Button("Re-render shot", variant="primary"); rmsg = gr.Markdown()
                with gr.Accordion("Log", open=False):
                    logbox = gr.Textbox(show_label=False, lines=14, autoscroll=True)

        def on_char(evt: gr.SelectData):
            n = characters()[evt.index]
            return n, f"Character: **{n}**"
        gal.select(on_char, None, [sel_char, chosen])
        cb.click(add_character, [cn, cp, cf, cd], [cm, gal])
        go.click(start_swap, [video, who, sel_char, keep, quality, relight, protect], [msg, proj])
        outs = [status, final, shotgal, shot_nums]
        proj.change(refresh, proj, outs); rb.click(refresh, proj, outs); ui.load(refresh, proj, outs)
        shotgal.select(pick_shot, [proj, shot_nums], [shotvid, cur_shot, note])
        pb.click(plan_edit, [proj, cur_shot, req], [pmsg, note, ident, chl, light, keepc, take])
        rr.click(rerender_shot, [proj, cur_shot, note, ident, chl, light, keepc, take], rmsg)
        sb.click(stop, proj, msg)
        gr.Timer(4).tick(tick, proj, [status, logbox, final, shotgal, shot_nums])
    return ui


def main() -> None:
    os.chdir(ROOT)
    pf = ROOT / ".ui_password"
    pw = os.environ.get("UI_PASSWORD") or (pf.read_text().strip() if pf.exists() else "swap")
    build().queue().launch(server_name=os.environ.get("UI_HOST", "127.0.0.1"),
                           server_port=int(os.environ.get("UI_PORT", "7860")),
                           auth=(os.environ.get("UI_USER", "swap"), pw), allowed_paths=[str(CASES), str(CHARS)],
                           share=False)


if __name__ == "__main__":
    main()
