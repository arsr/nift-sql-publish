"""Test double for laya 0.4.8: same call signatures, no model. Scores = word overlap between question and table card."""
import json
import re
from pathlib import Path

__version__ = "0.4.8"


def _w(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s).replace("_", " ").lower()
    return set(re.findall(r"[a-z]{3,}", s))


class _Agent:
    def predict_batch(self, states, questions, batch_size=128):
        out = []
        for s in states:
            q, t = _w(s["question"]), _w(s["table"])
            p = min(0.98, 0.02 + 0.3 * len(q & t))
            out.append({"answers": {k: {"noul": p} for k in questions}})
        return out


def load(path, device="cpu"):
    assert (Path(path) / "rl_agent_config.json").exists(), f"no checkpoint at {path}"
    return _Agent()
