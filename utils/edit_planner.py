#!/usr/bin/env python3
"""Edit planner: turns a free-form request for one shot ("she looks too pale and not like Giang, smile less") into
per-shot pipeline settings, using a local vision-language model (Qwen3-VL-4B-Instruct, on the free GPU).

The model sees a few frames of the shot and its identity scores and answers with JSON:
  possible     can this pipeline do it (the swap model follows the guide's motion: no new actions/objects/camera)
  explanation  one or two sentences for the user, in the user's language
  note         appearance instruction in English for the prompt ("she smiles softly, eyes looking down"), or ""
  new_take     render a new variation
  identity     "same" | "stronger" | "strongest"  (stronger = soften the original face more in the guide)
  relight      null or 0-1 (re-light strength for this shot)
  keep_outfit  null | true | false
apply_plan() merges the result into the shot's entry of output/gen/shot_edits.json.

  python utils/edit_planner.py --frames a.jpg b.jpg --request "smile less" --info '{"shot": 4}'   (test)
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

MODEL = os.environ.get("EDIT_PLANNER_MODEL",
                       "/common/users/hn315/checkpoints/char_swap_models/llm/Qwen3-VL-4B-Instruct")
_M = None

SYSTEM = """You plan edits for ONE shot of a character-swap video. A diffusion model has replaced a person in the
original video with a new character (identity from reference photos). Motion, camera, pose and timing always follow
the original video and cannot be changed. Translate the user's request into these levers, and change ONLY what the
user asks for; everything not mentioned stays at its neutral value.
- note: English appearance instruction added to the prompt, describing ONLY the requested change of expression,
  gaze, mouth, hair, makeup or skin tone, e.g. "she smiles softly with closed lips". Never invent other changes.
  Use "" if the request has no appearance change.
- new_take: true only if the user asks for another version / just dislikes it, or when a note is given.
- identity: "stronger" or "strongest" only if the user says the face does not look like the character or looks like
  the original person; otherwise "same".
- relight: how strongly the scene's light is applied to the person. Too bright, glowing, too lit -> 1.0.
  Too dark, too pale, washed out, grey, lifeless -> 0.5. Not mentioned -> null.
- keep_outfit: true/false only if the user talks about clothes; otherwise null.
Impossible (possible=false): new actions or movements, adding/removing objects or people, camera changes, background
changes, timing changes. If only part is impossible, set possible=true and explain which part cannot be done.
"explanation": one or two short sentences IN THE SAME LANGUAGE AS THE USER'S REQUEST (Vietnamese request ->
Vietnamese explanation), saying what will be changed.
Answer ONLY with a JSON object with keys: possible, explanation, note, new_take, identity, relight, keep_outfit.

Examples:
Request: "her face looks like the original woman, not Giang"
{"possible": true, "explanation": "The original face is softened more strongly in this shot so the character's face takes over.", "note": "", "new_take": true, "identity": "stronger", "relight": null, "keep_outfit": null}
Request: "cô ấy trông hơi tái, và cười nhẹ hơn"
{"possible": true, "explanation": "Giảm hiệu ứng ánh sáng để da cô ấy bớt tái và làm nụ cười nhẹ hơn.", "note": "she has a warm, healthy skin tone and a gentle, subtle smile", "new_take": true, "identity": "same", "relight": 0.5, "keep_outfit": null}
Request: "too bright for the room"
{"possible": true, "explanation": "The scene's light is applied at full strength so she matches the room.", "note": "", "new_take": false, "identity": "same", "relight": 1.0, "keep_outfit": null}
Request: "make her dance faster and add a hat"
{"possible": false, "explanation": "Movements and new objects cannot be changed: the swap always follows the original video.", "note": "", "new_take": false, "identity": "same", "relight": null, "keep_outfit": null}"""


def _load():
    global _M
    if _M is None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        proc = AutoProcessor.from_pretrained(MODEL)
        model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda:0").eval()
        _M = (proc, model)
    return _M


def plan(request: str, frames: list[str], info: dict) -> dict:
    import torch
    from PIL import Image
    proc, model = _load()
    content = [{"type": "image", "image": Image.open(f).convert("RGB")} for f in frames[:3]]
    content.append({"type": "text", "text": f"Shot information: {json.dumps(info, ensure_ascii=False)}\n"
                                             f"User request: {request}"})
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": content}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"\{.*\}", text, re.S)
    try:
        p = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        p = {}
    p.setdefault("possible", True); p.setdefault("explanation", text.strip()[:300] if not m else "")
    p.setdefault("note", ""); p.setdefault("new_take", False); p.setdefault("identity", "same")
    p.setdefault("relight", None); p.setdefault("keep_outfit", None)
    return sanity(request, p, proc, model)


BRIGHT = ("too bright", "glow", "overexposed", "too lit", "quá sáng", "sáng quá", "chói")
DARK = ("too dark", "pale", "washed", "grey", "gray", "lifeless", "quá tối", "tối quá", "tái", "nhợt")
OWN = ("her own", "own clothes", "own top", "own outfit", "own dress", "của cô ấy", "đồ riêng", "quần áo riêng")
ORIGINAL_FACE = ("original woman", "original person", "original face", "old face", "the old one", "previous person",
                 "not like", "doesn't look like", "does not look like", "người cũ", "người gốc", "cô gái cũ",
                 "mặt cũ", "không giống", "chưa giống", "vẫn giống")
VIDEO_CLOTHES = ("video's clothes", "original clothes", "same clothes as the video", "giữ quần áo", "đồ trong video")


def sanity(request: str, p: dict, proc=None, model=None) -> dict:
    """Rule layer over the small model's answer for the levers it tends to confuse."""
    r = request.lower()
    if any(k in r for k in BRIGHT):
        p["relight"] = 1.0
    elif any(k in r for k in DARK):
        p["relight"] = 0.5
    elif not any(k in r for k in ("light", "bright", "dark", "sáng", "tối", "màu", "colour", "color")):
        p["relight"] = None
    if any(k in r for k in OWN):
        p["keep_outfit"] = False
    elif any(k in r for k in VIDEO_CLOTHES):
        p["keep_outfit"] = True
    elif not any(k in r for k in ("cloth", "outfit", "dress", "top", "wear", "kimono", "quần", "áo", "váy", "mặc")):
        p["keep_outfit"] = None
    if any(k in r for k in ORIGINAL_FACE):
        if p.get("identity", "same") == "same":
            p["identity"] = "stronger"
        p["new_take"] = True
    elif not any(k in r for k in ("look like", "looks like", "giống", "identity", "not her", "face is wrong", "khuôn mặt")):
        p["identity"] = "same"
    note = str(p.get("note") or "")
    if note and not note.isascii() and proc is not None:            # the prompt must be English
        import torch
        msgs = [{"role": "user", "content": [{"type": "text", "text":
                 f"Translate into short English, keep the meaning exactly, output only the translation: {note}"}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=80, do_sample=False)
        p["note"] = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    if p.get("note"):
        p["new_take"] = True
    return p


def apply_plan(edits_file: Path, shot: int, p: dict) -> dict:
    """Merge a plan into shot_edits.json; returns the shot's new settings."""
    edits = json.loads(edits_file.read_text()) if edits_file.exists() else {}
    e = edits.get(str(shot), {})
    e = e if isinstance(e, dict) else {"note": str(e)}
    if p.get("note"):
        e["note"] = p["note"]
    if p.get("new_take"):
        e["take"] = int(e.get("take", 0)) + 1
    ident = p.get("identity", "same")
    if ident == "stronger":
        e["soft_face_scale"], e["soft_face_px"] = 0.10, 14
    elif ident == "strongest":
        e["soft_face_scale"], e["soft_face_px"] = 0.07, 10
    if isinstance(p.get("relight"), (int, float)):
        e["relight"] = max(0.0, min(1.0, float(p["relight"])))
    if isinstance(p.get("keep_outfit"), bool):
        e["keep_outfit"] = p["keep_outfit"]
    if not any(k in e for k in ("take",)) and not p.get("new_take") and e == edits.get(str(shot), {}):
        e["take"] = int(e.get("take", 0)) + 1          # nothing changed: at least render a new variation
    edits[str(shot)] = e
    edits_file.write_text(json.dumps(edits, indent=1, ensure_ascii=False))
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--request", required=True)
    ap.add_argument("--info", default="{}")
    a = ap.parse_args()
    print(json.dumps(plan(a.request, a.frames, json.loads(a.info)), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
