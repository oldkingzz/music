"""The route table: which blocks a feature goes through, in order. One feature = one entry in
host/routes.json. A route's chain is a list of block names; the clip that comes out of one block goes into
the next. Every step's result is stored as its own clip, so nothing is lost when a later block fails."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from .blocks import BLOCKS, Block, BlockError, BlockUnavailable, Context
from .clips import ClipStore

ROUTES_FILE = Path(__file__).resolve().parents[1] / "routes.json"


def load_routes(path: Optional[Path] = None) -> List[dict]:
    routes = json.loads(Path(path or ROUTES_FILE).read_text())
    for r in routes:
        missing = [b for b in r["chain"] if b not in BLOCKS]
        if missing:
            raise ValueError(f"route {r['name']!r} names unknown blocks {missing}")
        blocked = [BLOCKS[b] for b in r["chain"] if BLOCKS[b].status != "ready"]
        r["ready"] = not blocked
        r["why_not"] = "；".join(f"{b.title}：{b.note}" for b in blocked)
    return routes


def run_route(route: dict, clip_id: str, params: dict, store: ClipStore, ctx: Context) -> dict:
    """params: {block name: {param: value}}. Returns {'clip': meta of the last clip, 'steps': [...]}."""
    x, meta = store.load(clip_id)
    cid, steps = clip_id, []
    for name in route["chain"]:
        block: Block = BLOCKS[name]
        if block.status != "ready":
            raise BlockUnavailable(f"{block.title}：{block.note}")
        p = block.clean(params.get(name, {}))
        t0 = time.perf_counter()
        res = block.run(ctx, x, p)
        secs = time.perf_counter() - t0
        extra = {k: v for k, v in res.items() if k not in ("audio", "summary")}
        out = store.add_array(res["audio"], name=f"{meta['name']} → {block.title}", source="block", parent=cid,
                              made_by=name, params=p, extra={"summary": res["summary"], **extra})
        steps.append({"block": name, "title": block.title, "seconds": round(secs, 3), "summary": res["summary"],
                      "clip": out["id"]})
        x, cid, meta = res["audio"], out["id"], out
    return {"clip": meta, "steps": steps}


__all__ = ["load_routes", "run_route", "BlockError", "BlockUnavailable"]
