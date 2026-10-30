"""Доверительные интервалы, общие для ПР1–ПР4 (правила — docs/boot.docx).

* Доли — интервал Уилсона (`wilson_ci`).
* Остальные метрики — bootstrap по объектам, B = 2000, seed = 42 (`bootstrap_ci`).
  Основной метод — BCa (`scipy.stats.bootstrap`); если BCa неприменим (вырожденное
  распределение, NaN) или слишком дорог (jackknife для очень больших n), берётся
  percentile, и это отражается в поле `method` результата.
* Сравнение clean/perturbed на одних объектах — парный bootstrap (`paired_bootstrap_ci`).
* Bootstrap внутри групп (Permit Type в ПР1, members/non-members в ПР4) —
  `stratified_bootstrap_ci`.

Модель при bootstrap не переобучается: перевыбираются только объекты.

Статистика передаётся как функция от одномерных массивов, возвращающая число:
`statistic(*arrays) -> float`. Если функция векторизована (принимает `axis` и
считает по последней оси), укажите `vectorized=True` — так BCa (jackknife) работает
на порядки быстрее. Готовые векторизованные метрики регрессии — ниже (`rmsle` и др.).
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats as sps

from config import BOOTSTRAP_B, SEED

# Выше этого n BCa по умолчанию заменяется на percentile: jackknife требует n пересчётов
# метрики, и для сотен тысяч объектов это слишком дорого (допускается docs/boot.docx).
BCA_MAX_N = 60_000


@dataclass
class CI:
    """Точечная оценка и 95% доверительный интервал."""

    estimate: float
    low: float
    high: float
    method: str  # wilson | BCa | percentile
    n: int
    B: int | None = None
    confidence: float = 0.95

    def __str__(self) -> str:
        return f"{self.estimate:.4g}; 95% CI = [{self.low:.4g}; {self.high:.4g}] ({self.method})"

    def fmt(self, digits: int = 4, scale: float = 1.0, suffix: str = "") -> str:
        e, lo, hi = (v * scale for v in (self.estimate, self.low, self.high))
        return f"{e:.{digits}g}{suffix}; 95% CI = [{lo:.{digits}g}{suffix}; {hi:.{digits}g}{suffix}]"

    def as_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


# ---------------------------------------------------------------- доли
def wilson_ci(k: int, n: int, confidence: float = 0.95) -> CI:
    """Интервал Уилсона для доли k/n."""
    k, n = int(k), int(n)
    if n <= 0:
        return CI(float("nan"), float("nan"), float("nan"), "wilson", n)
    z = sps.norm.ppf(0.5 + confidence / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    low = 0.0 if k == 0 else max(0.0, center - half)
    high = 1.0 if k == n else min(1.0, center + half)
    return CI(p, low, high, "wilson", n, None, confidence)


def proportion_ci(mask: np.ndarray, confidence: float = 0.95) -> CI:
    """Интервал Уилсона для доли True в булевом массиве (NaN не допускаются)."""
    mask = np.asarray(mask, dtype=bool)
    return wilson_ci(int(mask.sum()), mask.size, confidence)


# ---------------------------------------------------------------- bootstrap
def _as_arrays(arrays: Sequence) -> list[np.ndarray]:
    out = [np.asarray(a) for a in arrays]
    n = len(out[0])
    if any(len(a) != n for a in out):
        raise ValueError("все массивы должны иметь одинаковую длину (по объектам)")
    return out


def bootstrap_ci(
    statistic: Callable[..., float],
    *arrays,
    strata: np.ndarray | None = None,
    B: int = BOOTSTRAP_B,
    seed: int = SEED,
    method: str = "BCa",
    confidence: float = 0.95,
    vectorized: bool = False,
    bca_max_n: int = BCA_MAX_N,
    batch: int | None = None,
) -> CI:
    """Bootstrap-CI для statistic(*arrays) с перевыборкой объектов (строк).

    Все массивы индексируются одними и теми же перевыбранными индексами, поэтому
    связи между ними (y_true ↔ y_pred, clean ↔ perturbed) сохраняются.
    При заданном `strata` перевыборка выполняется независимо внутри каждой группы
    (размеры групп фиксированы) — стратифицированный bootstrap.
    """
    data = _as_arrays(arrays)
    n = len(data[0])
    est = float(statistic(*data))

    if strata is None:
        groups = [np.arange(n)]
    else:
        strata = np.asarray(strata)
        if len(strata) != n:
            raise ValueError("strata должна иметь длину n")
        groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]

    def stat_idx(*idx_groups, axis=-1):
        # idx_groups: индексы по каждой страте; в векторизованном режиме у них
        # ведущие оси (batch), которые могут не совпадать — приводим к общей форме.
        idx_groups = [np.asarray(g, dtype=np.int64) for g in idx_groups]
        lead = np.broadcast_shapes(*[g.shape[:-1] for g in idx_groups])
        idx = np.concatenate([np.broadcast_to(g, lead + g.shape[-1:]) for g in idx_groups], axis=-1)
        sub = [a[idx] for a in data]
        if vectorized:
            return statistic(*sub, axis=-1)
        return statistic(*sub)

    use = method
    if use == "BCa" and n > bca_max_n:
        use = "percentile"
    if batch is None:
        batch = max(1, min(B, 2_000_000 // max(n, 1)))

    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sps.bootstrap(
            tuple(groups), stat_idx, n_resamples=B, batch=batch, vectorized=vectorized,
            paired=False, axis=-1, confidence_level=confidence, method=use, rng=rng,
        )
    low, high = float(res.confidence_interval.low), float(res.confidence_interval.high)
    dist = np.asarray(res.bootstrap_distribution, dtype=float)
    if not (np.isfinite(low) and np.isfinite(high)):
        # BCa не определён (например, все перевыборки дали одно значение) → percentile
        alpha = (1 - confidence) / 2
        low, high = (float(q) for q in np.nanquantile(dist, [alpha, 1 - alpha]))
        use = "percentile"
    return CI(est, low, high, use, n, B, confidence)


def paired_bootstrap_ci(
    metric: Callable[..., float],
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    **kwargs,
) -> CI:
    """CI для ΔM = metric(y, pred_a) − metric(y, pred_b) на одних и тех же объектах.

    На каждой итерации один набор индексов применяется к обеим версиям (clean и
    perturbed). Если CI не содержит 0, различие значимо на уровне ≈ 0,05.
    `kwargs` передаются в `bootstrap_ci` (B, seed, method, vectorized, strata…).
    """
    vectorized = kwargs.get("vectorized", False)

    def delta(y, a, b, axis=None):
        if vectorized:
            return metric(y, a, axis=axis) - metric(y, b, axis=axis)
        return metric(y, a) - metric(y, b)

    if vectorized:
        stat = lambda y, a, b, axis=-1: delta(y, a, b, axis)  # noqa: E731
    else:
        stat = lambda y, a, b: delta(y, a, b)  # noqa: E731
    return bootstrap_ci(stat, y_true, pred_a, pred_b, **kwargs)


def stratified_bootstrap_ci(statistic: Callable[..., float], *arrays, strata: np.ndarray, **kwargs) -> CI:
    """Стратифицированный bootstrap: объекты перевыбираются внутри каждой группы `strata`.

    Для MIA (ПР4) strata — признак member/non-member, для ПР1 — Permit Type.
    """
    return bootstrap_ci(statistic, *arrays, strata=strata, **kwargs)


# ---------------------------------------------------------------- метрики регрессии
# Аргументы: y_log — log1p(истинной стоимости), p_log — прогноз в той же шкале.
# Все функции векторизованы по последней оси (axis=-1 при вызове из bootstrap).
def rmsle(y_log, p_log, axis=None):
    return np.sqrt(np.mean((np.asarray(p_log) - np.asarray(y_log)) ** 2, axis=axis))


def _rel_err(y_log, p_log):
    y, p = np.expm1(np.asarray(y_log, dtype=float)), np.expm1(np.asarray(p_log, dtype=float))
    return np.abs(p - y) / y


def mdape(y_log, p_log, axis=None):
    """Медиана абсолютной процентной ошибки, %."""
    return 100.0 * np.median(_rel_err(y_log, p_log), axis=axis)


def within_frac(y_log, p_log, tol: float = 0.25, axis=None):
    """Доля прогнозов в пределах ±tol от истинной стоимости."""
    return np.mean(_rel_err(y_log, p_log) <= tol, axis=axis)


def within25(y_log, p_log, axis=None):
    return within_frac(y_log, p_log, 0.25, axis=axis)


def mean_stat(x, axis=None):
    return np.mean(x, axis=axis)


def median_stat(x, axis=None):
    return np.median(x, axis=axis)
