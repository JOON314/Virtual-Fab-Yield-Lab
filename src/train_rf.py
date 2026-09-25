"""베이스라인: 도메인 특징 + RandomForest.
두 가지 분할로 학습해 'lot 누수'가 성능을 얼마나 부풀리는지 확인한다.
실행: python -m src.train_rf
"""
import json
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.common import load_npz, split, evaluate, OUT, SEED

FEAT = os.path.join("data", "features_64.npy")


def run(mode):
    X, y, classes, lots = load_npz()
    if not os.path.exists(FEAT):
        raise FileNotFoundError("먼저 `python -m src.features` 실행")
    F = np.load(FEAT)
    tr, va, te = split(y, lots, mode=mode)
    t = time.time()
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                class_weight="balanced_subsample",
                                n_jobs=-1, random_state=SEED)
    rf.fit(F[tr], y[tr])
    name = f"rf_{mode}split"
    out = os.path.join(OUT, name)
    mv = evaluate(y[va], rf.predict(F[va]), classes, os.path.join(out, "val"), name + " (val)")
    mt = evaluate(y[te], rf.predict(F[te]), classes, out, name + " (test)",
                  extra={"split": mode, "train_n": int(len(tr)), "fit_sec": round(time.time() - t, 1),
                         "val_macro_f1": mv["macro_f1"]})
    np.save(os.path.join(out, "test_pred.npy"), rf.predict(F[te]))
    np.save(os.path.join(out, "test_idx.npy"), te)
    print(f"{name}: test macro-F1={mt['macro_f1']:.4f} acc={mt['accuracy']:.4f} (val {mv['macro_f1']:.4f})")
    return mt


if __name__ == "__main__":
    res = {m: run(m) for m in ["lot", "random"]}
    json.dump({m: {"macro_f1": r["macro_f1"], "accuracy": r["accuracy"]} for m, r in res.items()},
              open(os.path.join(OUT, "rf_split_comparison.json"), "w"), indent=2)
