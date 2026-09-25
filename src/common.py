"""공통 유틸: 데이터 로드, 분할, 지표 저장."""
import json
import os

import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             accuracy_score)

DATA = os.path.join("data", "wm811k_64.npz")
OUT = "results"
SEED = 42
CMAP = ListedColormap(["white", "#d9d9d9", "#d62728"])


def load_npz(path: str = DATA):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 없음 -> 먼저 `python -m src.load` 실행")
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"], list(d["classes"]), d["lots"]


def split(y, lots, mode="lot", seed=SEED):
    """70/15/15 분할.
    mode='random': 웨이퍼 단위 stratified (같은 lot이 train/test에 섞임 -> 낙관적)
    mode='lot'   : lot 단위 그룹 분할 (실제 운영처럼 '처음 보는 lot'에서 평가)
    """
    from sklearn.model_selection import train_test_split, StratifiedGroupKFold
    idx = np.arange(len(y))
    if mode == "random":
        tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
        va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
        return tr, va, te
    # lot 그룹 분할: 20-fold 중 3개 fold=test, 3개=val (≈15%/15%)
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    folds = [f for _, f in sgkf.split(idx, y, groups=lots)]
    te = np.concatenate(folds[:3]); va = np.concatenate(folds[3:6])
    tr = np.concatenate(folds[6:])
    assert not (set(lots[tr]) & set(lots[te])), "lot 누수 발생"
    return tr, va, te


def evaluate(y_true, y_pred, classes, out_dir, name, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    labels = list(range(len(classes)))
    rep = classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                output_dict=True, zero_division=0)
    defect = [i for i, c in enumerate(classes) if c != "none"]
    m = {
        "name": name,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1_defect_only": float(f1_score(y_true, y_pred, labels=defect, average="macro", zero_division=0)),
        "per_class_recall": {c: round(rep[c]["recall"], 4) for c in classes},
        "per_class_f1": {c: round(rep[c]["f1-score"], 4) for c in classes},
    }
    if extra: m.update(extra)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.save(os.path.join(out_dir, "confusion.npy"), cm)
    json.dump(m, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2, ensure_ascii=False)
    plot_confusion(cm, classes, os.path.join(out_dir, "confusion.png"),
                   f"{name}  macro-F1={m['macro_f1']:.3f}")
    return m


def plot_confusion(cm, classes, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true (row-normalized recall)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
