"""What the character-swap workers accept, advertised to the web-app.

The web-app builds its forms from these specs and validates overrides against them, so a new option is added here
(and handled in render.py / util.py) without a change in the web-app. `group` splits the render provider's
parameters into the "create" options of a new swap and the per-"shot" edits of a re-render.
"""
from __future__ import annotations

from contracts.python.capabilities import ParameterSpec, ProviderCapabilities
from contracts.python.streams import Task

RENDER_PROVIDER = "character_swap"
UTIL_PROVIDER = "character_swap_util"

CREATE = "create"
SHOT = "shot"

RENDER_PARAMETERS = (
    # options of a new swap (project.json)
    ParameterSpec(name="quality", type="enum", values=("480p", "720p"), default="480p", label="Quality",
                  help="480p is a quick test (~25 min for 20 s); 720p is the final quality (~45-60 min).",
                  group=CREATE),
    ParameterSpec(name="keep_outfit", type="boolean", default=False, label="Keep the video's clothes", group=CREATE),
    ParameterSpec(name="relight", type="boolean", default=True, label="Re-light to the scene", group=CREATE),
    ParameterSpec(name="protect", type="string", max_length=200, label="People to keep unchanged",
                  help="e.g. the man", group=CREATE),
    ParameterSpec(name="seed", type="integer", minimum=0, maximum=2147483647, default=42, label="Seed", group=CREATE),
    # edits of one shot (output/gen/shot_edits.json)
    ParameterSpec(name="note", type="string", multiline=True, max_length=1000, label="Appearance note",
                  help="English, added to this shot's prompt", group=SHOT),
    ParameterSpec(name="identity", type="enum", values=("same", "stronger", "strongest"), default="same",
                  label="Identity strength",
                  help="Stronger softens the original face more in the guide, so the character's face takes over.",
                  group=SHOT),
    ParameterSpec(name="change_light", type="boolean", default=False, label="Change light", group=SHOT),
    ParameterSpec(name="relight_strength", type="number", minimum=0, maximum=1, default=1.0,
                  label="Re-light strength", help="Used with Change light: 1 = the scene's light at full strength.",
                  group=SHOT),
    ParameterSpec(name="clothes", type="enum", values=("unchanged", "video", "own"), default="unchanged",
                  label="Clothes", help="video = the video's clothes, own = her own clothes", group=SHOT),
    ParameterSpec(name="new_take", type="boolean", default=True, label="New take (another variation)", group=SHOT),
)

UTIL_PARAMETERS = (
    ParameterSpec(name="action", type="enum", values=("plan_edit", "add_character"), label="Action"),
    ParameterSpec(name="request", type="string", multiline=True, max_length=2000, label="Request",
                  help="What to change in the shot, in any language"),
)


def render_capabilities() -> ProviderCapabilities:
    return ProviderCapabilities(provider=RENDER_PROVIDER, task=Task.CHARACTER_SWAP.value, parameters=RENDER_PARAMETERS,
                                note="Character swap: MiniMax H3 + BFS on ComfyUI (scripts/run_swap.sh).")


def util_capabilities() -> ProviderCapabilities:
    return ProviderCapabilities(provider=UTIL_PROVIDER, task=Task.CHARACTER_SWAP_UTIL.value,
                                parameters=UTIL_PARAMETERS,
                                note="Character swap utilities: edit planner (Qwen3-VL), character preparation.")


def for_role(role: str) -> ProviderCapabilities:
    return render_capabilities() if role == "render" else util_capabilities()
