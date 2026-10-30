"""Короткие тесты src/stats.py на синтетике.

Запуск:  venv\\Scripts\\python tests\\test_stats.py   (или pytest, если установлен)
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import stats as st  # noqa: E402


def test_wilson_matches_scipy():
    for k, n in [(0, 50), (3, 50), (25, 50), (50, 50), (1234, 100000)]:
        ci = st.wilson_ci(k, n)
        ref = sps.binomtest(k, n).proportion_ci(method="wilson")
        assert abs(ci.low - ref.low) < 1e-7 and abs(ci.high - ref.high) < 1e-7, (k, n, ci, ref)
        assert ci.low <= ci.estimate <= ci.high


def test_bootstrap_mean_covers_truth():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 2.0, 2000)
    ci = st.bootstrap_ci(st.mean_stat, x, vectorized=True)
    assert ci.method == "BCa"
    assert ci.low < 5.0 < ci.high
    # ширина близка к нормальной аппроксимации 2·1,96·σ/√n
    assert abs((ci.high - ci.low) - 2 * 1.96 * 2.0 / np.sqrt(2000)) < 0.05


def test_vectorized_equals_loop():
    rng = np.random.default_rng(1)
    y = rng.normal(9, 1.5, 800)
    p = y + rng.normal(0, 0.7, 800)
    a = st.bootstrap_ci(st.rmsle, y, p, vectorized=True)
    b = st.bootstrap_ci(lambda u, v: st.rmsle(u, v), y, p, vectorized=False)
    assert abs(a.low - b.low) < 1e-9 and abs(a.high - b.high) < 1e-9


def test_degenerate_falls_back_to_percentile():
    x = np.ones(300)
    ci = st.bootstrap_ci(st.mean_stat, x, vectorized=True)
    assert ci.method == "percentile" and ci.low == ci.high == 1.0


def test_paired_detects_difference():
    rng = np.random.default_rng(2)
    y = rng.normal(9, 1.5, 3000)
    clean = y + rng.normal(0, 0.5, 3000)
    noisy = clean + rng.normal(0, 0.5, 3000)
    d = st.paired_bootstrap_ci(st.rmsle, y, noisy, clean, vectorized=True)
    assert d.low > 0  # шум ухудшает RMSLE значимо
    same = st.paired_bootstrap_ci(st.rmsle, y, clean, clean, vectorized=True)
    assert same.estimate == 0 and same.low == same.high == 0


def test_stratified_keeps_group_sizes():
    rng = np.random.default_rng(3)
    g = np.r_[np.zeros(900, int), np.ones(100, int)]
    x = np.where(g == 1, 10.0, 0.0) + rng.normal(0, 1, 1000)
    # доля группы 1 в перевыборке фиксирована → дисперсия среднего меньше, чем у обычного bootstrap
    strat = st.stratified_bootstrap_ci(st.mean_stat, x, strata=g, vectorized=True)
    plain = st.bootstrap_ci(st.mean_stat, x, vectorized=True)
    assert strat.low < x.mean() < strat.high
    assert (strat.high - strat.low) < 0.6 * (plain.high - plain.low)


def test_stratified_auc_like():
    # ROC-AUC members vs non-members, стратифицировано по группе (сценарий ПР4)
    rng = np.random.default_rng(4)
    s = np.r_[rng.normal(1, 1, 500), rng.normal(0, 1, 500)]
    lab = np.r_[np.ones(500, int), np.zeros(500, int)]

    def auc(score, label):
        return sps.mannwhitneyu(score[label == 1], score[label == 0]).statistic / (
            (label == 1).sum() * (label == 0).sum())

    ci = st.stratified_bootstrap_ci(auc, s, lab, strata=lab, B=500)
    true_auc = sps.norm.cdf(1 / np.sqrt(2))
    assert ci.low < true_auc < ci.high


def test_regression_metrics():
    y = np.log1p(np.array([100.0, 200.0, 1000.0]))
    p = np.log1p(np.array([110.0, 300.0, 1000.0]))
    assert abs(st.mdape(y, p) - 10.0) < 1e-6
    assert abs(st.within25(y, p) - 2 / 3) < 1e-12


def test_large_n_uses_percentile():
    x = np.random.default_rng(5).normal(size=st.BCA_MAX_N + 1)
    t0 = time.perf_counter()
    ci = st.bootstrap_ci(st.mean_stat, x, vectorized=True, B=200)
    assert ci.method == "percentile" and time.perf_counter() - t0 < 30


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.perf_counter()
            try:
                fn()
                print(f"[OK]   {name} ({time.perf_counter() - t0:.1f} с)")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
