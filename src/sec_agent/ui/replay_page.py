"""The replay audit page (T7): `replay.html` with a run's replay data filled in.

The data is JSON inside a `<script type="application/json">` block, read with `JSON.parse`, and the
page sets every string from it as text. Run text can still hold `</script>`, so every `<`, `>`
and `&` in the JSON is written as a Unicode escape: the block cannot be closed early, and the
parsed strings are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

PAGE = Path(__file__).with_name("replay.html")
SLOT = "/*__REPLAY_DATA__*/"


def page_json(data: dict) -> str:
    """The data as JSON that is safe inside a script block."""
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render(data: dict, *, dark: bool = False) -> str:
    """The replay page for `build_replay`'s data, in the app's light or dark theme."""
    template = PAGE.read_text(encoding="utf-8")
    if template.count(SLOT) != 1:
        raise ValueError(f"{PAGE.name} must hold the data slot {SLOT} exactly once")
    return template.replace(SLOT, page_json({**data, "theme": "dark" if dark else "light"}))
