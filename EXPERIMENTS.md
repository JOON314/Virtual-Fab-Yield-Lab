# Experiments (test = unseen lots unless noted)

| date | exp | change | val macro-F1 | test macro-F1 | acc |
|---|---|---|---|---|---|
| 2026-09-26 | rf_randomsplit | 특징 48 + RF300, 웨이퍼 단위 분할 | 0.836 | 0.827 | 0.965 |
| 2026-09-26 | rf_lotsplit | 동일, lot 단위 분할 (기본) | 0.829 | 0.840 | 0.964 |
| 2026-09-26 | cnn v1 | CNN 4블록, none 2만장 다운샘플, 가중 CE, 12ep | 0.853 | 0.847 | 0.963 |
| 2026-09-26 | cnn v2 | v1 + none logit bias +1.75 (val 탐색) | — | **0.878** | 0.976 |
| 2026-09-26 | virtual_fab v0 | 가상 맵 9메커니즘 × 300장, 실데이터 CNN으로 분류 | — | 일치율 91.7% | — |
