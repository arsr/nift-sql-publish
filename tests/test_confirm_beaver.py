"""experiments/confirm_beaver_v3.py end to end on a tiny synthetic BEAVER folder (hash embedder, tiny cross-encoder)."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from test_v2_beaver import _folder

REPO = Path(__file__).resolve().parents[1]


def test_confirm_end_to_end(tmp_path):
    src = _folder(tmp_path / "beaver")
    keys = tmp_path / "schema_keys.json"
    keys.write_text(json.dumps({"nova": {"T1": {"fk": [[["ID"], "T0", ["ID"]]], "pk": ["ID"], "cols": []}}}))
    out = tmp_path / "out"
    cmd = [sys.executable, str(REPO / "experiments" / "confirm_beaver_v3.py"), "--beaver", str(src), "--keys", str(keys),
           "--out", str(out), "--fake"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-4000:]
    txt = (out / "CONFIRM.txt").read_text()
    assert "(H1)" in txt and "(H2)" in txt and "== real" in txt
    sys.path.insert(0, str(REPO))
    from nift import data
    D = data.load_subset(out / "beaver_v3_suite", "beaver_v3", "beaver_v3")       # the paper's loader reads the declared key
    t1 = D.ids.index("nova__T1")
    assert D.joins.get(t1) == ["T1.ID = T0.ID"] and D.cards[t1] == "T1: ID, C1"
    R = pd.read_csv(out / "results" / "confirm_v3_seed1" / "results.csv")
    assert set(R.part) == {"test", "real"} and R.method.nunique() == 6
    assert (out / "results" / "confirm_v3_seed1" / "PREREGISTRATION.txt").exists()
    p2 = subprocess.run(cmd, capture_output=True, text=True)                          # test is scored once
    assert p2.returncode != 0 and "already scored once" in p2.stderr
