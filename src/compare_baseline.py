"""Сравнение текущего baseline с прежним (парный bootstrap на одних и тех же строках, модели не переобучаются).

Запуск из корня проекта (после train_baseline.py):
    venv\\Scripts\\python src\\compare_baseline.py --old artifacts\\baseline_c96 --name c96

Срезы: noR2 — без служебных оценок (основной, постановка задачи с 2026-11-03), full — все строки.
Δ = новый − прежний; для RMSLE и MdAPE меньше — лучше, для доли в ±25% — больше.
Результат: reports/baseline/compare_<name>.csv (части × срезы) и compare_<name>_by_group.csv (test, noR2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import stats as st
from config import ARTIFACTS_DIR, BOOTSTRAP_B, PROJECT_ROOT, REPORTS_DIR, SEED
from data import load_dataset, service_mask
from train_baseline import PRED_PATH

METRICS = (("d_rmsle", st.rmsle), ("d_mdape_pp", st.mdape), ("d_within25", st.within25))


def _paired(y, pn, po, metrics=METRICS) -> dict:
    row = {}
    for name, fn in metrics:
        ci = st.paired_bootstrap_ci(fn, y, pn, po, vectorized=True, B=BOOTSTRAP_B, seed=SEED)
        row |= {name: ci.estimate, f"{name}_low": ci.low, f"{name}_high": ci.high, f"{name}_method": ci.method}
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default=str(ARTIFACTS_DIR / "baseline_c96"))
    ap.add_argument("--name", default="c96")
    args = ap.parse_args()
    old_dir = Path(args.old)
    if not old_dir.is_absolute():
        old_dir = PROJECT_ROOT / old_dir
    new = pd.read_parquet(PRED_PATH)
    old = pd.read_parquet(old_dir / PRED_PATH.name)
    d = new[["row_id", "split", "y_log", "pred_log"]].merge(
        old[["row_id", "y_log", "pred_log"]], on="row_id", suffixes=("", "_old"), validate="one_to_one")
    assert np.array_equal(d["y_log"].to_numpy(), d["y_log_old"].to_numpy())
    meta = load_dataset()[["row_id", "issue_date", "PERMIT_TYPE", "PERMIT_SUB_TYPE"]]
    d = d.merge(meta, on="row_id", how="left", validate="one_to_one")
    d["service"] = service_mask(np.expm1(d["y_log"].to_numpy()))

    rows = []
    for part in ("val", "test"):
        for sl in ("noR2", "full"):
            x = d[(d["split"] == part) & (~d["service"] if sl == "noR2" else True)]
            y, pn, po = x["y_log"].to_numpy(), x["pred_log"].to_numpy(), x["pred_log_old"].to_numpy()
            rows.append({"part": part, "slice": sl, "n": len(x), "rmsle_new": float(st.rmsle(y, pn)),
                         "rmsle_old": float(st.rmsle(y, po)), **_paired(y, pn, po)})
            print(f"{part}/{sl}: RMSLE {rows[-1]['rmsle_new']:.4f} против {rows[-1]['rmsle_old']:.4f}, "
                  f"Δ {rows[-1]['d_rmsle']:+.4f} [{rows[-1]['d_rmsle_low']:+.4f}; {rows[-1]['d_rmsle_high']:+.4f}]")
    out = REPORTS_DIR / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"compare_{args.name}.csv", index=False)

    x = d[(d["split"] == "test") & ~d["service"]]
    groups = {"year=" + str(g): (x["issue_date"].dt.year == g).to_numpy() for g in sorted(x["issue_date"].dt.year.unique())}
    for col in ("PERMIT_TYPE", "PERMIT_SUB_TYPE"):
        groups |= {f"{col}={g}": (x[col] == g).to_numpy() for g in x[col].dropna().unique()}
    y, pn, po = x["y_log"].to_numpy(), x["pred_log"].to_numpy(), x["pred_log_old"].to_numpy()
    by = []
    for g, m in groups.items():
        if m.sum() < 30:
            continue
        by.append({"group": g, "n": int(m.sum()), "rmsle_new": float(st.rmsle(y[m], pn[m])),
                   "rmsle_old": float(st.rmsle(y[m], po[m])), **_paired(y[m], pn[m], po[m], METRICS[:1])})
    pd.DataFrame(by).to_csv(out / f"compare_{args.name}_by_group.csv", index=False)
    print(json.dumps({"old": str(old_dir), "rows": len(rows), "groups": len(by)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
