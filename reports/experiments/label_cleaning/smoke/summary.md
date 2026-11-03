# Эксперимент label_cleaning: очистка служебных оценок (варианты A и C96, val_hold)

План — `docs/context/PLAN_LABEL_CLEANING.md`. Правило R2: VALUATION ∈ {$501, $301} ∪ {≤ $10}; R3 отброшен на шаге 0 (`step0_summary.json`, `r3_precision.csv`).
val_sel 2022-07-16 … 2023-08-16 — ранняя остановка и выбор эпохи (у *_noR2 без R2); val_hold 2023-08-17 … 2024-09-04 — только оценка. Test не используется.
Срезы val_hold: full — все строки, noR2 — без служебных (критерий для *_noR2), onlyR2 — только служебные.
Парный bootstrap против эталона того же семейства (A_noR2 → A_full, C96_noR2 → C96_full) с тем же seed, B = 2000: ΔRMSLE на срезе критерия — BCa, остальное — percentile (метод — в столбцах `*_method`).
Эталоны A_full (seed 43, 44) и C96_full взяты из text_tokens_holdout (`adopted_from` в metrics.json).

## Запуски и метрики по срезам

| variant | seed | slice | n | rmsle | mdape_pct | within25 | best_epoch | epochs_run | sel_rmsle | n_train | train_y_mean | train_y_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_full | 42 | full | 53268 | 1.3552 | 68.1264 | 0.1848 | 2 | 2 | 1.3451 | 5000 | 9.3774 | 1.8277 |
| A_full | 42 | noR2 | 48166 | 1.1854 | 61.4173 | 0.2044 | 2 | 2 | 1.3451 | 5000 | 9.3774 | 1.8277 |
| A_full | 42 | onlyR2 | 5102 | 2.4310 | 498.7800 | 0.0000 | 2 | 2 | 1.3451 | 5000 | 9.3774 | 1.8277 |
| A_noR2 | 42 | full | 53268 | 1.5840 | 72.5891 | 0.1786 | 1 | 2 | 1.2359 | 5000 | 9.7077 | 1.6782 |
| A_noR2 | 42 | noR2 | 48166 | 1.2243 | 64.7705 | 0.1975 | 1 | 2 | 1.2359 | 5000 | 9.7077 | 1.6782 |
| A_noR2 | 42 | onlyR2 | 5102 | 3.4706 | 2182.3394 | 0.0000 | 1 | 2 | 1.2359 | 5000 | 9.7077 | 1.6782 |
| C96_full | 42 | full | 53268 | 1.2475 | 65.1355 | 0.1849 | 2 | 2 | 1.2175 | 5000 | 9.3774 | 1.8277 |
| C96_full | 42 | noR2 | 48166 | 1.1477 | 59.6092 | 0.2044 | 2 | 2 | 1.2175 | 5000 | 9.3774 | 1.8277 |
| C96_full | 42 | onlyR2 | 5102 | 1.9526 | 293.6272 | 0.0016 | 2 | 2 | 1.2175 | 5000 | 9.3774 | 1.8277 |
| C96_noR2 | 42 | full | 53268 | 1.3689 | 69.3319 | 0.1619 | 2 | 2 | 1.1218 | 5000 | 9.7077 | 1.6782 |
| C96_noR2 | 42 | noR2 | 48166 | 1.1249 | 63.4240 | 0.1790 | 2 | 2 | 1.1218 | 5000 | 9.7077 | 1.6782 |
| C96_noR2 | 42 | onlyR2 | 5102 | 2.7601 | 1096.3676 | 0.0000 | 2 | 2 | 1.1218 | 5000 | 9.7077 | 1.6782 |

## Парные сравнения с эталоном

| variant | reference | seed | slice | criterion | n | d_rmsle | d_rmsle_low | d_rmsle_high | d_rmsle_method | d_mdape_pp | d_within25 | ci_excludes_0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_noR2 | A_full | 42 | full | False | 53268 | 0.2288 | 0.2214 | 0.2363 | percentile | 4.4627 | -0.0062 | True |
| A_noR2 | A_full | 42 | noR2 | True | 48166 | 0.0390 | 0.0328 | 0.0453 | BCa | 3.3532 | -0.0069 | True |
| A_noR2 | A_full | 42 | onlyR2 | False | 5102 | 1.0396 | 1.0172 | 1.0607 | percentile | 1683.5595 | 0.0000 | True |
| C96_noR2 | C96_full | 42 | full | False | 53268 | 0.1214 | 0.1156 | 0.1267 | percentile | 4.1964 | -0.0230 | True |
| C96_noR2 | C96_full | 42 | noR2 | True | 48166 | -0.0228 | -0.0272 | -0.0182 | BCa | 3.8148 | -0.0253 | True |
| C96_noR2 | C96_full | 42 | onlyR2 | False | 5102 | 0.8075 | 0.7913 | 0.8244 | percentile | 802.7405 | -0.0016 | True |

## Правило решения

Средний ΔRMSLE по seed ≤ -0.02 и CI не содержит 0 при каждом seed — на собственном очищенном срезе варианта; минимум seed: A — 3, C96 — 2 (`min_seeds`).

| variant | slice | n_seeds | d_rmsle_mean | d_rmsle_min | d_rmsle_max | d_rmsle_high_max | d_rmsle_low_min | d_mdape_mean | d_within25_mean | min_seeds | passes_rule |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_noR2 | noR2 | 1 | 0.0390 | 0.0390 | 0.0390 | 0.0453 | 0.0328 | 3.3532 | -0.0069 | 3 | False |
| C96_noR2 | noR2 | 1 | -0.0228 | -0.0228 | -0.0228 | -0.0182 | -0.0272 | 3.8148 | -0.0253 | 2 | False |

- A_noR2 (срез noR2): средний ΔRMSLE +0.0390 (min–max +0.0390…+0.0390) — не проходит.
- C96_noR2 (срез noR2): средний ΔRMSLE -0.0228 (min–max -0.0228…-0.0228) — не проходит.

## По типу разрешения (seed 42, срез критерия)

| variant | reference | slice | PERMIT_TYPE | n | rmsle_reference | rmsle_variant | d_rmsle | d_rmsle_low | d_rmsle_high | d_rmsle_method |
|---|---|---|---|---|---|---|---|---|---|---|
| A_noR2 | A_full | noR2 | Bldg-Addition | 5310 | 1.0328 | 1.3483 | 0.3156 | 0.2953 | 0.3351 | percentile |
| A_noR2 | A_full | noR2 | Bldg-Alter/Repair | 39155 | 1.2145 | 1.1993 | -0.0151 | -0.0219 | -0.0085 | percentile |
| A_noR2 | A_full | noR2 | Bldg-New | 3701 | 1.0734 | 1.2971 | 0.2237 | 0.1956 | 0.2511 | percentile |
| C96_noR2 | C96_full | noR2 | Bldg-Addition | 5310 | 0.9413 | 0.9821 | 0.0408 | 0.0270 | 0.0544 | percentile |
| C96_noR2 | C96_full | noR2 | Bldg-Alter/Repair | 39155 | 1.1766 | 1.1488 | -0.0278 | -0.0329 | -0.0230 | percentile |
| C96_noR2 | C96_full | noR2 | Bldg-New | 3701 | 1.1068 | 1.0596 | -0.0473 | -0.0603 | -0.0337 | percentile |
