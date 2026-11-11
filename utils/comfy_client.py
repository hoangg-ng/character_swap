#!/usr/bin/env python3
"""Minimal ComfyUI HTTP client used by the pipeline scripts: submit an API-format workflow, wait for it, and
download its outputs."""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


class Comfy:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def _req(self, path: str, data: dict | None = None):
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(self.url + path, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    def alive(self) -> bool:
        try:
            self._req("/system_stats")
            return True
        except Exception:
            return False

    def submit(self, wf: dict) -> str:
        try:
            res = json.loads(self._req("/prompt", {"prompt": wf, "client_id": uuid.uuid4().hex}))
        except urllib.error.HTTPError as e:
            sys.exit(f"ComfyUI rejected the workflow:\n{e.read().decode()[:4000]}")
        if res.get("node_errors"):
            sys.exit(f"node errors: {json.dumps(res['node_errors'], indent=2)}")
        return res["prompt_id"]

    def wait(self, pid: str, poll: float = 10.0) -> dict:
        t0 = time.time()
        while True:
            hist = json.loads(self._req(f"/history/{pid}"))
            if pid in hist:
                h = hist[pid]
                st = h.get("status", {})
                if st.get("status_str") == "error":
                    msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                    sys.exit(f"execution error:\n{json.dumps(msgs, indent=2)[:4000]}")
                if st.get("completed", True):
                    return h
            print(f"\r  running... {time.time() - t0:6.0f}s", end="", flush=True)
            time.sleep(poll)

    def download_outputs(self, hist: dict, out_dir: Path) -> list[Path]:
        saved = []
        for nid, outs in hist.get("outputs", {}).items():
            for items in outs.values():
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not (isinstance(it, dict) and "filename" in it):
                        continue
                    q = urllib.parse.urlencode({"filename": it["filename"],
                                                "subfolder": it.get("subfolder", ""),
                                                "type": it.get("type", "output")})
                    dst = out_dir / f"{nid}_{Path(it['filename']).name}"
                    dst.write_bytes(self._req(f"/view?{q}"))
                    saved.append(dst)
        return saved
