"""Ingesta LLM: sesiones raw_json → stream canónico por token.

Canónico: session_id, model, pos, surprisal, entropy, gap, finish_reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_sessions(root: str) -> pd.DataFrame:
    rows = []
    files = sorted(Path(root).rglob("*.json"))
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "turns" not in d:
            continue
        sid = d.get("session_id", f.stem)
        model = d.get("model", "")
        for turn in d["turns"]:
            toks = turn.get("tokens") or []
            fr = turn.get("finish_reason", "")
            for i, t in enumerate(toks):
                rows.append({
                    "session_id": sid,
                    "model": model,
                    "pos": i,
                    "surprisal": -t["top1_logprob"],
                    "entropy": t.get("entropy", 0.0),
                    "gap": t.get("gap", 0.0),
                    "finish_reason": fr,
                })
    return pd.DataFrame(rows)
