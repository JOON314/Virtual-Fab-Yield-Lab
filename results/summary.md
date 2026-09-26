# Test results (lot-split test set unless noted)

R = recall

| model | macro-F1 | defect-only macro-F1 | accuracy | Center R | Donut R | Edge-Loc R | Edge-Ring R | Loc R | Near-full R | Random R | Scratch R | none R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RF (random split) | 0.827 | 0.807 | 0.965 | 0.87 | 0.72 | 0.69 | 0.96 | 0.50 | 1.00 | 0.82 | 0.51 | 0.99 |
| RF (lot split) | 0.840 | 0.822 | 0.964 | 0.86 | 0.89 | 0.67 | 0.95 | 0.50 | 0.96 | 0.85 | 0.50 | 0.99 |
| CNN v1 raw | 0.847 | 0.830 | 0.963 | 0.96 | 0.98 | 0.88 | 0.97 | 0.84 | 0.96 | 0.92 | 0.88 | 0.97 |
| CNN v2 calibrated | 0.878 | 0.864 | 0.976 | 0.91 | 0.98 | 0.80 | 0.95 | 0.78 | 0.96 | 0.92 | 0.84 | 0.99 |
| CNN v3 max-pool resize | 0.881 | 0.867 | 0.977 | 0.91 | 0.96 | 0.82 | 0.96 | 0.77 | 0.96 | 0.94 | 0.87 | 0.99 |
