"""Load the frozen config and set up one run folder per (person, seed)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import subprocess
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
RUN_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,30}$")


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or REPO / "configs" / "default.yaml")
    cfg = yaml.safe_load(path.read_text())
    cfg["_config_sha"] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True, timeout=10).stdout.strip()
        return (out.stdout.strip() or "unknown") + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def run_dir(out_root: str | os.PathLike, run_name: str, seed: int) -> Path:
    """results/<run_name>_seed<seed>/ ; the cache for this run (rows, checkpoints, scores) lives inside it."""
    if not RUN_NAME_RE.match(run_name):
        raise ValueError(f"RUN_NAME must be lowercase letters, digits, _ or -, e.g. 'arjun' (got {run_name!r})")
    d = Path(out_root) / "results" / f"{run_name}_seed{seed}"
    for sub in ("rows", "checkpoints", "scores"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def write_manifest(rd: Path, cfg: dict, seed: int, run_name: str, extra: dict | None = None) -> dict:
    import sklearn
    man = dict(run_name=run_name, seed=seed, git_commit=git_commit(), config_sha=cfg["_config_sha"],
               python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__)
    try:
        import torch
        man["torch"] = torch.__version__
        man["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        man["gpu"] = "cpu"
    try:
        import laya
        man["laya"] = laya.__version__
    except ImportError:
        man["laya"] = "not installed"
    man.update(extra or {})
    (rd / "manifest.json").write_text(json.dumps(man, indent=1))
    (rd / "config_used.yaml").write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=False))
    return man
