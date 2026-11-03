# Эксперимент time_clip: время без обрезки и доменные правила

План — `docs/context/PLAN_TIME_CLIP.md`. Модель C96, меняются только числовые признаки: A — `t` без обрезки + sin/cos месяца и дня года; Aplus — A + доменные правила вместо квантильной обрезки, предохранитель ±50σ.
val_sel 2022-07-16 … 2023-08-16 — ранняя остановка и выбор эпохи; val_hold 2023-08-17 … 2024-09-04 — только оценка. Test не используется.
Фильтр — обучающая выборка (full или noR2, как у принятого baseline); решающий срез val_hold совпадает с фильтром. Парный bootstrap против эталона с тем же seed, B = 2000, percentile. bias — средний (прогноз − факт) в log-шкале.


## Запуски и метрики по срезам

| filter | variant | seed | slice | n | rmsle | mdape_pct | within25 | bias | best_epoch | epochs_run | sel_rmsle | n_tokens | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noR2 | A | 42 | full | 53268 | 1.1612 | 53.4426 | 0.2545 | 0.3483 | 2 | 8 | 0.7805 | 128.0000 | time_clip |
| noR2 | A | 42 | noR2 | 48166 | 0.8251 | 47.9060 | 0.2807 | 0.1206 | 2 | 8 | 0.7805 | 128.0000 | time_clip |
| noR2 | Aplus | 42 | full | 53268 | 1.1748 | 53.3042 | 0.2536 | 0.3713 | 2 | 8 | 0.7739 | 128.0000 | time_clip |
| noR2 | Aplus | 42 | noR2 | 48166 | 0.8190 | 48.1803 | 0.2796 | 0.1377 | 2 | 8 | 0.7739 | 128.0000 | time_clip |
| noR2 | C96 | 42 | full | 53268 | 1.1368 | 55.7648 | 0.2535 | 0.2975 | 3 | 9 | 0.7842 | nan | artifacts\experiments\label_cleaning\C96_noR2_s42 |
| noR2 | C96 | 42 | noR2 | 48166 | 0.8230 | 49.6867 | 0.2779 | 0.0736 | 3 | 9 | 0.7842 | nan | artifacts\experiments\label_cleaning\C96_noR2_s42 |

## Парные сравнения

| filter | variant | reference | seed | slice | decision_slice | n | d_rmsle | d_rmsle_low | d_rmsle_high | d_mdape_pp | d_within25 | ci_excludes_0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noR2 | A | C96 | 42 | full | False | 53268 | 0.0244 | 0.0209 | 0.0282 | -2.3221 | 0.0010 | True |
| noR2 | A | C96 | 42 | noR2 | True | 48166 | 0.0020 | -0.0015 | 0.0055 | -1.7808 | 0.0028 | False |
| noR2 | Aplus | C96 | 42 | full | False | 53268 | 0.0380 | 0.0344 | 0.0420 | -2.4606 | 0.0000 | True |
| noR2 | Aplus | C96 | 42 | noR2 | True | 48166 | -0.0040 | -0.0078 | -0.0003 | -1.5064 | 0.0017 | True |
| noR2 | Aplus | A | 42 | full | False | 53268 | 0.0136 | 0.0122 | 0.0149 | -0.1385 | -0.0009 | True |
| noR2 | Aplus | A | 42 | noR2 | True | 48166 | -0.0060 | -0.0074 | -0.0047 | 0.2744 | -0.0011 | True |

## Решение (на решающем срезе)

Правило принятия: средний ΔRMSLE против C96 по seed ≤ -0.02 и CI без 0 при каждом seed (не меньше 3 seed). Промежуточное решение после этапа 1: на этап 2 — средний ΔRMSLE ≤ -0.01. Aplus против A — только для понимания.

| filter | variant | reference | slice | n_seeds | d_rmsle_mean | d_rmsle_min | d_rmsle_max | d_rmsle_high_max | d_mdape_mean | d_within25_mean | promising | passes_rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noR2 | A | C96 | noR2 | 1 | 0.0020 | 0.0020 | 0.0020 | 0.0055 | -1.7808 | 0.0028 | False | False |
| noR2 | Aplus | A | noR2 | 1 | -0.0060 | -0.0060 | -0.0060 | -0.0047 | 0.2744 | -0.0011 | False | False |
| noR2 | Aplus | C96 | noR2 | 1 | -0.0040 | -0.0040 | -0.0040 | -0.0003 | -1.5064 | 0.0017 | False | False |

- noR2/A: средний ΔRMSLE +0.0020 (1 seed) — не перспективный, правило не проходит.
- noR2/Aplus: средний ΔRMSLE -0.0040 (1 seed) — не перспективный, правило не проходит.

## По кварталам val_hold (среднее по seed, решающий срез)

| filter | variant | quarter | rmsle | bias |
|---|---|---|---|---|
| noR2 | A | 2023Q3 | 0.8112 | 0.1264 |
| noR2 | A | 2023Q4 | 0.8129 | 0.1233 |
| noR2 | A | 2024Q1 | 0.8131 | 0.1080 |
| noR2 | A | 2024Q2 | 0.8261 | 0.0853 |
| noR2 | A | 2024Q3 | 0.8628 | 0.1777 |
| noR2 | Aplus | 2023Q3 | 0.7976 | 0.1329 |
| noR2 | Aplus | 2023Q4 | 0.8061 | 0.1358 |
| noR2 | Aplus | 2024Q1 | 0.8097 | 0.1328 |
| noR2 | Aplus | 2024Q2 | 0.8228 | 0.1074 |
| noR2 | Aplus | 2024Q3 | 0.8558 | 0.1919 |
| noR2 | C96 | 2023Q3 | 0.8146 | 0.0925 |
| noR2 | C96 | 2023Q4 | 0.8111 | 0.0961 |
| noR2 | C96 | 2024Q1 | 0.8146 | 0.0443 |
| noR2 | C96 | 2024Q2 | 0.8182 | 0.0337 |
| noR2 | C96 | 2024Q3 | 0.8603 | 0.1229 |

## По группам (seed 42, решающий срез)

| filter | variant | reference | group | n | rmsle_reference | rmsle_variant | d_rmsle | d_rmsle_low | d_rmsle_high |
|---|---|---|---|---|---|---|---|---|---|
| noR2 | A | C96 | all | 48166 | 0.8230 | 0.8251 | 0.0020 | -0.0015 | 0.0055 |
| noR2 | A | C96 | valuation>=1M | 529 | 1.6457 | 1.5764 | -0.0694 | -0.1227 | -0.0170 |
| noR2 | A | C96 | PERMIT_TYPE=Bldg-Alter/Repair | 39155 | 0.8647 | 0.8666 | 0.0019 | -0.0019 | 0.0059 |
| noR2 | A | C96 | PERMIT_TYPE=Bldg-Addition | 5310 | 0.6704 | 0.6780 | 0.0076 | -0.0025 | 0.0162 |
| noR2 | A | C96 | PERMIT_TYPE=Bldg-New | 3701 | 0.5104 | 0.5046 | -0.0059 | -0.0129 | 0.0005 |
| noR2 | A | C96 | PERMIT_SUB_TYPE=1 or 2 Family Dwelling | 36093 | 0.7153 | 0.7150 | -0.0003 | -0.0032 | 0.0026 |
| noR2 | A | C96 | PERMIT_SUB_TYPE=Apartment | 7069 | 0.9352 | 0.9510 | 0.0159 | 0.0063 | 0.0248 |
| noR2 | A | C96 | PERMIT_SUB_TYPE=Commercial | 5004 | 1.2627 | 1.2600 | -0.0027 | -0.0176 | 0.0130 |
| noR2 | Aplus | C96 | all | 48166 | 0.8230 | 0.8190 | -0.0040 | -0.0078 | -0.0003 |
| noR2 | Aplus | C96 | valuation>=1M | 529 | 1.6457 | 1.5386 | -0.1071 | -0.1589 | -0.0545 |
| noR2 | Aplus | C96 | PERMIT_TYPE=Bldg-Alter/Repair | 39155 | 0.8647 | 0.8589 | -0.0058 | -0.0100 | -0.0018 |
| noR2 | Aplus | C96 | PERMIT_TYPE=Bldg-Addition | 5310 | 0.6704 | 0.6805 | 0.0101 | 0.0000 | 0.0190 |
| noR2 | Aplus | C96 | PERMIT_TYPE=Bldg-New | 3701 | 0.5104 | 0.5118 | 0.0013 | -0.0069 | 0.0092 |
| noR2 | Aplus | C96 | PERMIT_SUB_TYPE=1 or 2 Family Dwelling | 36093 | 0.7153 | 0.7075 | -0.0078 | -0.0109 | -0.0048 |
| noR2 | Aplus | C96 | PERMIT_SUB_TYPE=Apartment | 7069 | 0.9352 | 0.9451 | 0.0099 | -0.0001 | 0.0192 |
| noR2 | Aplus | C96 | PERMIT_SUB_TYPE=Commercial | 5004 | 1.2627 | 1.2591 | -0.0036 | -0.0192 | 0.0123 |
| noR2 | Aplus | A | all | 48166 | 0.8251 | 0.8190 | -0.0060 | -0.0074 | -0.0047 |
| noR2 | Aplus | A | valuation>=1M | 529 | 1.5764 | 1.5386 | -0.0378 | -0.0565 | -0.0187 |
| noR2 | Aplus | A | PERMIT_TYPE=Bldg-Alter/Repair | 39155 | 0.8666 | 0.8589 | -0.0077 | -0.0093 | -0.0062 |
| noR2 | Aplus | A | PERMIT_TYPE=Bldg-Addition | 5310 | 0.6780 | 0.6805 | 0.0024 | -0.0014 | 0.0061 |
| noR2 | Aplus | A | PERMIT_TYPE=Bldg-New | 3701 | 0.5046 | 0.5118 | 0.0072 | 0.0026 | 0.0117 |
| noR2 | Aplus | A | PERMIT_SUB_TYPE=1 or 2 Family Dwelling | 36093 | 0.7150 | 0.7075 | -0.0075 | -0.0088 | -0.0062 |
| noR2 | Aplus | A | PERMIT_SUB_TYPE=Apartment | 7069 | 0.9510 | 0.9451 | -0.0060 | -0.0107 | -0.0015 |
| noR2 | Aplus | A | PERMIT_SUB_TYPE=Commercial | 5004 | 1.2600 | 1.2591 | -0.0009 | -0.0056 | 0.0035 |

## Строки, затронутые правилами Aplus или обрезкой C96 (seed 42, решающий срез)

| filter | rows | slice | n | rmsle_C96 | rmsle_Aplus | d_rmsle | d_rmsle_low | d_rmsle_high | d_rmsle_method |
|---|---|---|---|---|---|---|---|---|---|
| noR2 | any_rule | noR2 | 1843 | 1.0640 | 1.0669 | 0.0029 | -0.0216 | 0.0269 | percentile |
| noR2 | height_le0 | noR2 | 0 | nan | nan | nan | nan | nan | nan |
| noR2 | height_gt_max | noR2 | 3 | 0.0948 | 0.1794 | nan | nan | nan | nan |
| noR2 | height_family | noR2 | 5 | 0.7144 | 0.9000 | nan | nan | nan | nan |
| noR2 | height_eq_sqft | noR2 | 2 | 0.0962 | 0.0308 | nan | nan | nan | nan |
| noR2 | sqft_le0 | noR2 | 1836 | 1.0654 | 1.0678 | 0.0025 | -0.0249 | 0.0266 | percentile |
| noR2 | latlon_out_bbox | noR2 | 0 | nan | nan | nan | nan | nan | nan |
| noR2 | adu_abs_gt_max | noR2 | 2 | 0.2876 | 0.4732 | nan | nan | nan | nan |
| noR2 | adu_family | noR2 | 2 | 0.2876 | 0.4732 | nan | nan | nan | nan |
| noR2 | c96_clipped | noR2 | 840 | 0.7755 | 0.7931 | 0.0176 | -0.0042 | 0.0410 | percentile |

## Вывод (2026-11-03)

Этап 1 (seed 42, `--filter noR2`, решающий срез noR2, эталон C96_noR2):
A ΔRMSLE +0,0020 [−0,0015; +0,0055], A+ −0,0040 [−0,0078; −0,0003]. Ни один вариант не достиг порога
«перспективный» (ΔRMSLE ≤ −0,01) → этап 2 (seed 43, 44) не запускался, правило решения не проверяется.

- MdAPE лучше у обоих (−1,8 и −1,5 п.п.), но решение по RMSLE.
- Смещение (прогноз − факт) по кварталам val_hold выросло: C96 +0,03…+0,12, A/A+ +0,09…+0,19 —
  непрерывное время экстраполирует рост стоимости; риск для test (2–4 года после train).
- Выигрыш A+ сосредоточен в хвосте ≥ $1M (n = 529, −0,107); у Apartment и Addition хуже.
- На строках, изменённых правилами A+, эффекта нет.

Решение пользователя 2026-11-03: числовые признаки baseline остаются как в C96 (year_clip / quantile);
baseline переобучается только без R2.
