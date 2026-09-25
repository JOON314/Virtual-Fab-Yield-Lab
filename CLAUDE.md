# Virtual Fab — Yield Lab

## 목표
WM-811K 웨이퍼 맵 결함 패턴 분류 + 패턴→공정 원인 매핑. SK하이닉스 AI 해커톤 2026 포트폴리오용 (제출 마감 2026-09-28).
장기 목표: 공정 파라미터 → 가상 웨이퍼 맵을 생성하는 Virtual Fab 시뮬레이터 (sim-to-real 검증).

## 역할 분담
- 사람(Joon): 문제 정의, 지표·분할 설계, 오분류 판정, 공정 원인 가설 검증
- Claude Code: 구현, 실험 실행, 결과 정리

## 원칙
- 주 지표: macro-F1 + 클래스별 recall. accuracy 단독 보고 금지 ('none'이 85%).
- 기본 분할은 **lot 단위 그룹 분할** (같은 lot이 train/test에 섞이면 성능이 부풀려짐).
- seed=42 고정, 결과는 results/<exp_name>/metrics.json + png.
- 테스트셋은 최종 평가에만 사용. 튜닝(모델 선택, bias 보정)은 validation으로.
- 노트북 CPU 기준, 학습 1회 30~60분 이내.
- 새 실험마다 EXPERIMENTS.md에 한 줄 기록. AI 제안 → 사람 검증 내역은 AI_LOG.md에 기록.

## 구조
- data/ (gitignore) — LSWMD.pkl, 전처리 npz
- src/load.py → eda.py → features.py → train_rf.py → train_cnn.py → analyze.py
- results/ — 지표·그래프 (커밋 대상)
