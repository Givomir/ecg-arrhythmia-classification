# S ablation (DS2 test set)

Protocol: S threshold chosen on out-of-fold scores (5-fold GroupKFold by record over DS1, criterion `f1`), final model on all of DS1, tested on DS2. Seeds: 0,1,2. Values are mean ± std. Macro F1 is over N, S, V and F (Q excluded). "w/o 232" excludes record 232, which holds ~75% of the DS2 S beats. Stage 1 is a Random Forest; the cascade's S/N stage model is named in the Features column. For cascades, "argmax" means stage-2 threshold 0.5.

## DS2 with the tuned threshold

| Experiment | Features | Balance | threshold | S rec | S prec | S F1 | N→S | S→N | V rec | macro F1 | S rec w/o 232 | S prec w/o 232 | S F1 w/o 232 |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| full_cw | morph+rr+rrn+p | cw | 0.30 ± 0.01 | 0.280 ± 0.003 | 0.308 ± 0.013 | 0.293 ± 0.007 | 1117 ± 49 | 1281 ± 7 | 0.951 ± 0.001 | 0.524 ± 0.008 | 0.684 ± 0.013 | 0.212 ± 0.005 | 0.323 ± 0.004 |
| base_rrlong_cw | morph+rr+rrn+rrlong | cw | 0.26 ± 0.02 | 0.819 ± 0.040 | 0.450 ± 0.018 | 0.580 ± 0.006 | 1779 ± 204 | 304 ± 73 | 0.946 ± 0.002 | 0.608 ± 0.005 | 0.770 ± 0.039 | 0.160 ± 0.010 | 0.265 ± 0.012 |
| base_rrrec_cw | morph+rr+rrn+rrrec | cw | 0.24 ± 0.01 | 0.802 ± 0.019 | 0.448 ± 0.015 | 0.574 ± 0.013 | 1735 ± 112 | 335 ± 34 | 0.943 ± 0.002 | 0.599 ± 0.010 | 0.762 ± 0.012 | 0.160 ± 0.006 | 0.265 ± 0.008 |
| base_rrlong_irr_cw | morph+rr+rrn+rrlong+irr | cw | 0.28 ± 0.01 | 0.530 ± 0.025 | 0.397 ± 0.023 | 0.454 ± 0.024 | 1418 ± 89 | 821 ± 50 | 0.949 ± 0.001 | 0.576 ± 0.004 | 0.728 ± 0.004 | 0.183 ± 0.008 | 0.293 ± 0.010 |
| base_rrrec_irr_cw | morph+rr+rrn+rrrec+irr | cw | 0.28 ± 0.00 | 0.418 ± 0.076 | 0.351 ± 0.042 | 0.381 ± 0.056 | 1333 ± 35 | 1029 ± 139 | 0.949 ± 0.001 | 0.550 ± 0.015 | 0.730 ± 0.006 | 0.192 ± 0.005 | 0.304 ± 0.006 |
| base_rrlong_irr_none | morph+rr+rrn+rrlong+irr | none | 0.22 ± 0.01 | 0.445 ± 0.055 | 0.449 ± 0.013 | 0.446 ± 0.035 | 940 ± 75 | 985 ± 100 | 0.906 ± 0.002 | 0.577 ± 0.008 | 0.664 ± 0.024 | 0.233 ± 0.010 | 0.345 ± 0.009 |
| cascade_rf | morph+rr+rrn+rrlong+irr → S/N rf: pca+rr+rrn+rrlong+irr | cw | 0.38 ± 0.01 | 0.225 ± 0.018 | 0.392 ± 0.014 | 0.286 ± 0.018 | 606 ± 22 | 1356 ± 33 | 0.958 ± 0.002 | 0.534 ± 0.006 | 0.626 ± 0.010 | 0.308 ± 0.004 | 0.413 ± 0.002 |
| cascade_hgb | morph+rr+rrn+rrlong+irr → S/N hgb: pca+rr+rrn+rrlong+irr | cw | 0.34 ± 0.20 | 0.345 ± 0.118 | 0.239 ± 0.078 | 0.260 ± 0.035 | 2343 ± 1160 | 1135 ± 222 | 0.958 ± 0.002 | 0.523 ± 0.008 | 0.725 ± 0.080 | 0.155 ± 0.077 | 0.241 ± 0.091 |
| cascade_rf_rhythm_only | morph+rr+rrn+rrlong+irr → S/N rf: rr+rrn+rrlong+irr | cw | 0.10 ± 0.02 | 0.581 ± 0.061 | 0.222 ± 0.006 | 0.321 ± 0.016 | 3627 ± 293 | 702 ± 115 | 0.958 ± 0.002 | 0.536 ± 0.005 | 0.812 ± 0.008 | 0.091 ± 0.006 | 0.163 ± 0.009 |

## DS2 with argmax (no threshold tuning)

| Experiment | S rec | S prec | S F1 | V rec | macro F1 |
|---|--:|--:|--:|--:|--:|
| full_cw | 0.153 ± 0.010 | 0.338 ± 0.015 | 0.211 ± 0.012 | 0.955 ± 0.001 | 0.504 ± 0.010 |
| base_rrlong_cw | 0.303 ± 0.047 | 0.420 ± 0.028 | 0.351 ± 0.041 | 0.956 ± 0.002 | 0.552 ± 0.014 |
| base_rrrec_cw | 0.232 ± 0.013 | 0.383 ± 0.006 | 0.289 ± 0.012 | 0.956 ± 0.002 | 0.529 ± 0.007 |
| base_rrlong_irr_cw | 0.200 ± 0.002 | 0.363 ± 0.003 | 0.258 ± 0.002 | 0.958 ± 0.002 | 0.527 ± 0.004 |
| base_rrrec_irr_cw | 0.161 ± 0.010 | 0.337 ± 0.012 | 0.217 ± 0.012 | 0.960 ± 0.002 | 0.510 ± 0.005 |
| base_rrlong_irr_none | 0.067 ± 0.008 | 0.305 ± 0.025 | 0.109 ± 0.012 | 0.915 ± 0.002 | 0.492 ± 0.009 |
| cascade_rf | 0.131 ± 0.002 | 0.496 ± 0.014 | 0.207 ± 0.002 | 0.958 ± 0.002 | 0.515 ± 0.005 |
| cascade_hgb | 0.269 ± 0.027 | 0.250 ± 0.053 | 0.255 ± 0.028 | 0.958 ± 0.002 | 0.524 ± 0.007 |
| cascade_rf_rhythm_only | 0.137 ± 0.007 | 0.248 ± 0.014 | 0.176 ± 0.009 | 0.958 ± 0.002 | 0.506 ± 0.003 |

## DS1 out-of-fold (with threshold) - for comparison with DS2

| Experiment | S rec | S prec | S F1 |
|---|--:|--:|--:|
| full_cw | 0.316 ± 0.024 | 0.307 ± 0.023 | 0.311 ± 0.022 |
| base_rrlong_cw | 0.425 ± 0.043 | 0.284 ± 0.019 | 0.339 ± 0.005 |
| base_rrrec_cw | 0.450 ± 0.018 | 0.266 ± 0.007 | 0.334 ± 0.008 |
| base_rrlong_irr_cw | 0.388 ± 0.008 | 0.322 ± 0.020 | 0.352 ± 0.010 |
| base_rrrec_irr_cw | 0.396 ± 0.022 | 0.304 ± 0.010 | 0.343 ± 0.001 |
| base_rrlong_irr_none | 0.433 ± 0.033 | 0.268 ± 0.010 | 0.331 ± 0.017 |
| cascade_rf | 0.362 ± 0.009 | 0.324 ± 0.018 | 0.341 ± 0.014 |
| cascade_hgb | 0.413 ± 0.055 | 0.266 ± 0.026 | 0.320 ± 0.010 |
| cascade_rf_rhythm_only | 0.606 ± 0.016 | 0.236 ± 0.005 | 0.340 ± 0.004 |
