import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TrainConfig:
    epochs: int = 4
    loss: str = "soft-ce"
    micro_batch: int = 16
    grad_accum: int = 4
    calib_max: int = 2000
    seed: int = 0


def finetune(rows_path, base, out_dir, cfg, device="cpu"):
    assert cfg.loss in ("soft-ce", "bce"), f"loss must be one of ['soft-ce', 'bce'], got {cfg.loss!r}"
    rows = [json.loads(l) for l in open(rows_path)]
    for r in rows:
        assert set(r) == {"state", "questions", "expected"}, "row keys must be state / questions / expected"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "rl_agent_config.json").write_text(json.dumps({"base": base, **asdict(cfg)}))
    return {"rows": len(rows)}
