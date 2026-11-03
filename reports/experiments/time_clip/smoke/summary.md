# Эксперимент time_clip: время без обрезки и доменные правила (smoke)

План — `docs/context/PLAN_TIME_CLIP.md`. Модель C96, меняются только числовые признаки: A — `t` без обрезки + sin/cos месяца и дня года; Aplus — A + доменные правила вместо квантильной обрезки, предохранитель ±50σ.
val_sel 2022-07-16 … 2023-08-16 — ранняя остановка и выбор эпохи; val_hold 2023-08-17 … 2024-09-04 — только оценка. Test не используется.
Фильтр — обучающая выборка (full или noR2, как у принятого baseline); решающий срез val_hold совпадает с фильтром. Парный bootstrap против эталона с тем же seed, B = 2000, percentile. bias — средний (прогноз − факт) в log-шкале.

Проверка пути обучения (full, seed 42): свой C96 против `artifacts\experiments\label_cleaning\smoke\C96_full_s42` — бит-в-бит: True (max |Δ| = 0).
Проверка пути обучения (noR2, seed 42): свой C96 против `artifacts\experiments\label_cleaning\smoke\C96_noR2_s42` — бит-в-бит: True (max |Δ| = 0).

## Запуски и метрики по срезам

| filter | variant | seed | slice | n | rmsle | mdape_pct | within25 | bias | best_epoch | epochs_run | sel_rmsle | n_tokens | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | A | 42 | full | 53268 | 1.2343 | 57.6383 | 0.2283 | -0.1045 | 2 | 2 | 1.2085 | 128 | time_clip |
| full | A | 42 | noR2 | 48166 | 1.1343 | 53.8608 | 0.2423 | -0.2727 | 2 | 2 | 1.2085 | 128 | time_clip |
| full | Aplus | 42 | full | 53268 | 1.3076 | 73.4453 | 0.1728 | 0.4187 | 2 | 2 | 1.2722 | 128 | time_clip |
| full | Aplus | 42 | noR2 | 48166 | 1.0920 | 65.7310 | 0.1912 | 0.2200 | 2 | 2 | 1.2722 | 128 | time_clip |
| full | C96 | 42 | full | 53268 | 1.2475 | 65.1355 | 0.1849 | -0.0312 | 2 | 2 | 1.2175 | 126 | time_clip |
| full | C96 | 42 | noR2 | 48166 | 1.1477 | 59.6092 | 0.2044 | -0.2103 | 2 | 2 | 1.2175 | 126 | time_clip |
| noR2 | A | 42 | full | 53268 | 1.3601 | 60.5225 | 0.2133 | 0.1394 | 2 | 2 | 1.0656 | 128 | time_clip |
| noR2 | A | 42 | noR2 | 48166 | 1.0605 | 54.6356 | 0.2359 | -0.1448 | 2 | 2 | 1.0656 | 128 | time_clip |
| noR2 | Aplus | 42 | full | 53268 | 1.3569 | 59.7120 | 0.2181 | 0.1207 | 2 | 2 | 1.0656 | 128 | time_clip |
| noR2 | Aplus | 42 | noR2 | 48166 | 1.0596 | 53.7071 | 0.2412 | -0.1643 | 2 | 2 | 1.0656 | 128 | time_clip |
| noR2 | C96 | 42 | full | 53268 | 1.3689 | 69.3319 | 0.1619 | 0.1385 | 2 | 2 | 1.1218 | 126 | time_clip |
| noR2 | C96 | 42 | noR2 | 48166 | 1.1249 | 63.4240 | 0.1790 | -0.1255 | 2 | 2 | 1.1218 | 126 | time_clip |

## Парные сравнения

| filter | variant | reference | seed | slice | decision_slice | n | d_rmsle | d_rmsle_low | d_rmsle_high | d_mdape_pp | d_within25 | ci_excludes_0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | A | C96 | 42 | full | True | 53268 | -0.0132 | -0.0190 | -0.0071 | -7.4972 | 0.0433 | True |
| full | A | C96 | 42 | noR2 | False | 48166 | -0.0134 | -0.0193 | -0.0071 | -5.7484 | 0.0380 | True |
| full | Aplus | C96 | 42 | full | True | 53268 | 0.0602 | 0.0530 | 0.0673 | 8.3098 | -0.0121 | True |
| full | Aplus | C96 | 42 | noR2 | False | 48166 | -0.0557 | -0.0626 | -0.0491 | 6.1218 | -0.0132 | True |
| full | Aplus | A | 42 | full | True | 53268 | 0.0734 | 0.0676 | 0.0787 | 15.8070 | -0.0554 | True |
| full | Aplus | A | 42 | noR2 | False | 48166 | -0.0423 | -0.0478 | -0.0373 | 11.8702 | -0.0512 | True |
| noR2 | A | C96 | 42 | full | False | 53268 | -0.0088 | -0.0134 | -0.0041 | -8.8093 | 0.0514 | True |
| noR2 | A | C96 | 42 | noR2 | True | 48166 | -0.0644 | -0.0688 | -0.0595 | -8.7885 | 0.0569 | True |
| noR2 | Aplus | C96 | 42 | full | False | 53268 | -0.0120 | -0.0167 | -0.0072 | -9.6199 | 0.0562 | True |
| noR2 | Aplus | C96 | 42 | noR2 | True | 48166 | -0.0653 | -0.0699 | -0.0605 | -9.7169 | 0.0622 | True |
| noR2 | Aplus | A | 42 | full | False | 53268 | -0.0032 | -0.0036 | -0.0028 | -0.8106 | 0.0048 | True |
| noR2 | Aplus | A | 42 | noR2 | True | 48166 | -0.0010 | -0.0015 | -0.0005 | -0.9284 | 0.0053 | True |

## Решение (на решающем срезе)

Правило принятия: средний ΔRMSLE против C96 по seed ≤ -0.02 и CI без 0 при каждом seed (не меньше 3 seed). Промежуточное решение после этапа 1: на этап 2 — средний ΔRMSLE ≤ -0.01. Aplus против A — только для понимания.

| filter | variant | reference | slice | n_seeds | d_rmsle_mean | d_rmsle_min | d_rmsle_max | d_rmsle_high_max | d_mdape_mean | d_within25_mean | promising | passes_rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | A | C96 | full | 1 | -0.0132 | -0.0132 | -0.0132 | -0.0071 | -7.4972 | 0.0433 | True | False |
| full | Aplus | A | full | 1 | 0.0734 | 0.0734 | 0.0734 | 0.0787 | 15.8070 | -0.0554 | False | False |
| full | Aplus | C96 | full | 1 | 0.0602 | 0.0602 | 0.0602 | 0.0673 | 8.3098 | -0.0121 | False | False |
| noR2 | A | C96 | noR2 | 1 | -0.0644 | -0.0644 | -0.0644 | -0.0595 | -8.7885 | 0.0569 | True | False |
| noR2 | Aplus | A | noR2 | 1 | -0.0010 | -0.0010 | -0.0010 | -0.0005 | -0.9284 | 0.0053 | False | False |
| noR2 | Aplus | C96 | noR2 | 1 | -0.0653 | -0.0653 | -0.0653 | -0.0605 | -9.7169 | 0.0622 | True | False |

- full/A: средний ΔRMSLE -0.0132 (1 seed) — перспективный, правило не проходит.
- full/Aplus: средний ΔRMSLE +0.0602 (1 seed) — не перспективный, правило не проходит.
- noR2/A: средний ΔRMSLE -0.0644 (1 seed) — перспективный, правило не проходит.
- noR2/Aplus: средний ΔRMSLE -0.0653 (1 seed) — перспективный, правило не проходит.

## По кварталам val_hold (среднее по seed, решающий срез)

| filter | variant | quarter | rmsle | bias |
|---|---|---|---|---|
| full | A | 2023Q3 | 1.2230 | -0.1066 |
| full | A | 2023Q4 | 1.2394 | -0.0879 |
| full | A | 2024Q1 | 1.2323 | -0.0481 |
| full | A | 2024Q2 | 1.2384 | -0.1781 |
| full | A | 2024Q3 | 1.2323 | -0.0925 |
| full | Aplus | 2023Q3 | 1.2886 | 0.3856 |
| full | Aplus | 2023Q4 | 1.3071 | 0.4312 |
| full | Aplus | 2024Q1 | 1.3372 | 0.5076 |
| full | Aplus | 2024Q2 | 1.2895 | 0.3708 |
| full | Aplus | 2024Q3 | 1.3095 | 0.3829 |
| full | C96 | 2023Q3 | 1.2575 | 0.0768 |
| full | C96 | 2023Q4 | 1.2478 | 0.0304 |
| full | C96 | 2024Q1 | 1.2522 | -0.1419 |
| full | C96 | 2024Q2 | 1.2459 | -0.0990 |
| full | C96 | 2024Q3 | 1.2362 | 0.0429 |
| noR2 | A | 2023Q3 | 1.0449 | -0.0862 |
| noR2 | A | 2023Q4 | 1.0471 | -0.0916 |
| noR2 | A | 2024Q1 | 1.0585 | -0.1459 |
| noR2 | A | 2024Q2 | 1.0812 | -0.2241 |
| noR2 | A | 2024Q3 | 1.0623 | -0.1432 |
| noR2 | Aplus | 2023Q3 | 1.0419 | -0.1082 |
| noR2 | Aplus | 2023Q4 | 1.0451 | -0.1090 |
| noR2 | Aplus | 2024Q1 | 1.0574 | -0.1616 |
| noR2 | Aplus | 2024Q2 | 1.0825 | -0.2474 |
| noR2 | Aplus | 2024Q3 | 1.0610 | -0.1627 |
| noR2 | C96 | 2023Q3 | 1.1139 | -0.0324 |
| noR2 | C96 | 2023Q4 | 1.1101 | -0.0441 |
| noR2 | C96 | 2024Q1 | 1.1451 | -0.2166 |
| noR2 | C96 | 2024Q2 | 1.1300 | -0.1998 |
| noR2 | C96 | 2024Q3 | 1.1191 | -0.0792 |

## По группам (seed 42, решающий срез)

| filter | variant | reference | group | n | rmsle_reference | rmsle_variant | d_rmsle | d_rmsle_low | d_rmsle_high |
|---|---|---|---|---|---|---|---|---|---|
| full | A | C96 | all | 53268 | 1.2475 | 1.2343 | -0.0132 | -0.0190 | -0.0071 |
| full | A | C96 | valuation>=1M | 529 | 3.3795 | 2.9454 | -0.4340 | -0.4945 | -0.3737 |
| full | A | C96 | PERMIT_TYPE=Bldg-Alter/Repair | 44154 | 1.2808 | 1.2712 | -0.0096 | -0.0162 | -0.0032 |
| full | A | C96 | PERMIT_TYPE=Bldg-Addition | 5385 | 1.0097 | 1.0307 | 0.0210 | 0.0000 | 0.0419 |
| full | A | C96 | PERMIT_TYPE=Bldg-New | 3729 | 1.1547 | 1.0461 | -0.1087 | -0.1319 | -0.0856 |
| full | A | C96 | PERMIT_SUB_TYPE=1 or 2 Family Dwelling | 39924 | 1.0896 | 1.0513 | -0.0383 | -0.0446 | -0.0318 |
| full | A | C96 | PERMIT_SUB_TYPE=Apartment | 7600 | 1.4195 | 1.4627 | 0.0431 | 0.0276 | 0.0589 |
| full | A | C96 | PERMIT_SUB_TYPE=Commercial | 5744 | 1.8743 | 1.9013 | 0.0269 | 0.0056 | 0.0484 |
| full | Aplus | C96 | all | 53268 | 1.2475 | 1.3076 | 0.0602 | 0.0530 | 0.0673 |
| full | Aplus | C96 | valuation>=1M | 529 | 3.3795 | 2.9194 | -0.4600 | -0.5156 | -0.3993 |
| full | Aplus | C96 | PERMIT_TYPE=Bldg-Alter/Repair | 44154 | 1.2808 | 1.3597 | 0.0789 | 0.0707 | 0.0873 |
| full | Aplus | C96 | PERMIT_TYPE=Bldg-Addition | 5385 | 1.0097 | 0.9891 | -0.0206 | -0.0415 | -0.0006 |
| full | Aplus | C96 | PERMIT_TYPE=Bldg-New | 3729 | 1.1547 | 1.0596 | -0.0951 | -0.1154 | -0.0753 |
| full | Aplus | C96 | PERMIT_SUB_TYPE=1 or 2 Family Dwelling | 39924 | 1.0896 | 1.1528 | 0.0631 | 0.0557 | 0.0710 |
| full | Aplus | C96 | PERMIT_SUB_TYPE=Apartment | 7600 | 1.4195 | 1.4813 | 0.0618 | 0.0410 | 0.0814 |
| full | Aplus | C96 | PERMIT_SUB_TYPE=Commercial | 5744 | 1.8743 | 1.9281 | 0.0538 | 0.0282 | 0.0779 |
| full | Aplus | A | all | 53268 | 1.2343 | 1.3076 | 0.0734 | 0.0676 | 0.0787 |
| full | Aplus | A | valuation>=1M | 529 | 2.9454 | 2.9194 | -0.0260 | -0.0591 | 0.0081 |
| full | Aplus | A | PERMIT_TYPE=Bldg-Alter/Repair | 44154 | 1.2712 | 1.3597 | 0.0885 | 0.0821 | 0.0949 |
| full | Aplus | A | PERMIT_TYPE=Bldg-Addition | 5385 | 1.0307 | 0.9891 | -0.0415 | -0.0500 | -0.0338 |
| full | Aplus | A | PERMIT_TYPE=Bldg-New | 3729 | 1.0461 | 1.0596 | 0.0136 | 0.0053 | 0.0210 |
| full | Aplus | A | PERMIT_SUB_TYPE=1 or 2 Family Dwelling | 39924 | 1.0513 | 1.1528 | 0.1014 | 0.0951 | 0.1079 |
| full | Aplus | A | PERMIT_SUB_TYPE=Apartment | 7600 | 1.4627 | 1.4813 | 0.0186 | 0.0040 | 0.0334 |
| full | Aplus | A | PERMIT_SUB_TYPE=Commercial | 5744 | 1.9013 | 1.9281 | 0.0268 | 0.0115 | 0.0422 |
| noR2 | A | C96 | all | 48166 | 1.1249 | 1.0605 | -0.0644 | -0.0688 | -0.0595 |
| noR2 | A | C96 | valuation>=1M | 529 | 3.2550 | 2.8788 | -0.3762 | -0.4344 | -0.3193 |
| noR2 | A | C96 | PERMIT_TYPE=Bldg-Alter/Repair | 39155 | 1.1488 | 1.0759 | -0.0729 | -0.0779 | -0.0683 |
| noR2 | A | C96 | PERMIT_TYPE=Bldg-Addition | 5310 | 0.9821 | 0.9735 | -0.0086 | -0.0252 | 0.0074 |
| noR2 | A | C96 | PERMIT_TYPE=Bldg-New | 3701 | 1.0596 | 1.0160 | -0.0436 | -0.0654 | -0.0230 |
| noR2 | A | C96 | PERMIT_SUB_TYPE=1 or 2 Family Dwelling | 36093 | 1.0008 | 0.9159 | -0.0849 | -0.0896 | -0.0802 |
| noR2 | A | C96 | PERMIT_SUB_TYPE=Apartment | 7069 | 1.2263 | 1.2258 | -0.0005 | -0.0122 | 0.0107 |
| noR2 | A | C96 | PERMIT_SUB_TYPE=Commercial | 5004 | 1.6827 | 1.6287 | -0.0540 | -0.0714 | -0.0364 |
| noR2 | Aplus | C96 | all | 48166 | 1.1249 | 1.0596 | -0.0653 | -0.0699 | -0.0605 |
| noR2 | Aplus | C96 | valuation>=1M | 529 | 3.2550 | 2.8591 | -0.3958 | -0.4561 | -0.3379 |
| noR2 | Aplus | C96 | PERMIT_TYPE=Bldg-Alter/Repair | 39155 | 1.1488 | 1.0774 | -0.0714 | -0.0763 | -0.0667 |
| noR2 | Aplus | C96 | PERMIT_TYPE=Bldg-Addition | 5310 | 0.9821 | 0.9756 | -0.0065 | -0.0229 | 0.0096 |
| noR2 | Aplus | C96 | PERMIT_TYPE=Bldg-New | 3701 | 1.0596 | 0.9825 | -0.0770 | -0.0987 | -0.0572 |
| noR2 | Aplus | C96 | PERMIT_SUB_TYPE=1 or 2 Family Dwelling | 36093 | 1.0008 | 0.9112 | -0.0896 | -0.0945 | -0.0849 |
| noR2 | Aplus | C96 | PERMIT_SUB_TYPE=Apartment | 7069 | 1.2263 | 1.2322 | 0.0060 | -0.0060 | 0.0176 |
| noR2 | Aplus | C96 | PERMIT_SUB_TYPE=Commercial | 5004 | 1.6827 | 1.6350 | -0.0476 | -0.0653 | -0.0300 |
| noR2 | Aplus | A | all | 48166 | 1.0605 | 1.0596 | -0.0010 | -0.0015 | -0.0005 |
| noR2 | Aplus | A | valuation>=1M | 529 | 2.8788 | 2.8591 | -0.0197 | -0.0253 | -0.0144 |
| noR2 | Aplus | A | PERMIT_TYPE=Bldg-Alter/Repair | 39155 | 1.0759 | 1.0774 | 0.0015 | 0.0011 | 0.0019 |
| noR2 | Aplus | A | PERMIT_TYPE=Bldg-Addition | 5310 | 0.9735 | 0.9756 | 0.0021 | -0.0002 | 0.0044 |
| noR2 | Aplus | A | PERMIT_TYPE=Bldg-New | 3701 | 1.0160 | 0.9825 | -0.0335 | -0.0361 | -0.0308 |
| noR2 | Aplus | A | PERMIT_SUB_TYPE=1 or 2 Family Dwelling | 36093 | 0.9159 | 0.9112 | -0.0048 | -0.0054 | -0.0042 |
| noR2 | Aplus | A | PERMIT_SUB_TYPE=Apartment | 7069 | 1.2258 | 1.2322 | 0.0064 | 0.0054 | 0.0075 |
| noR2 | Aplus | A | PERMIT_SUB_TYPE=Commercial | 5004 | 1.6287 | 1.6350 | 0.0064 | 0.0047 | 0.0079 |

## Строки, затронутые правилами Aplus или обрезкой C96 (seed 42, решающий срез)

| filter | rows | slice | n | rmsle_C96 | rmsle_Aplus | d_rmsle | d_rmsle_low | d_rmsle_high | d_rmsle_method |
|---|---|---|---|---|---|---|---|---|---|
| full | any_rule | full | 2097 | 1.5522 | 1.6089 | 0.0566 | 0.0206 | 0.0937 | percentile |
| full | height_le0 | full | 0 | nan | nan | nan | nan | nan | nan |
| full | height_gt_max | full | 3 | 0.4357 | 0.5801 | nan | nan | nan | nan |
| full | height_family | full | 5 | 0.9874 | 0.8977 | nan | nan | nan | nan |
| full | height_eq_sqft | full | 2 | 0.4080 | 0.6685 | nan | nan | nan | nan |
| full | sqft_le0 | full | 2090 | 1.5539 | 1.6108 | 0.0569 | 0.0198 | 0.0945 | percentile |
| full | latlon_out_bbox | full | 0 | nan | nan | nan | nan | nan | nan |
| full | adu_abs_gt_max | full | 2 | 0.8566 | 0.7021 | nan | nan | nan | nan |
| full | adu_family | full | 2 | 0.8566 | 0.7021 | nan | nan | nan | nan |
| full | c96_clipped | full | 858 | 1.3159 | 1.2765 | -0.0394 | -0.0814 | 0.0030 | percentile |
| noR2 | any_rule | noR2 | 1843 | 1.3947 | 1.3368 | -0.0579 | -0.0858 | -0.0292 | percentile |
| noR2 | height_le0 | noR2 | 0 | nan | nan | nan | nan | nan | nan |
| noR2 | height_gt_max | noR2 | 3 | 0.1665 | 0.7408 | nan | nan | nan | nan |
| noR2 | height_family | noR2 | 5 | 0.9044 | 0.9731 | nan | nan | nan | nan |
| noR2 | height_eq_sqft | noR2 | 2 | 0.1903 | 0.6525 | nan | nan | nan | nan |
| noR2 | sqft_le0 | noR2 | 1836 | 1.3962 | 1.3378 | -0.0584 | -0.0844 | -0.0298 | percentile |
| noR2 | latlon_out_bbox | noR2 | 0 | nan | nan | nan | nan | nan | nan |
| noR2 | adu_abs_gt_max | noR2 | 2 | 0.8996 | 1.1944 | nan | nan | nan | nan |
| noR2 | adu_family | noR2 | 2 | 0.8996 | 1.1944 | nan | nan | nan | nan |
| noR2 | c96_clipped | noR2 | 840 | 1.3125 | 1.2820 | -0.0305 | -0.0788 | 0.0162 | percentile |
