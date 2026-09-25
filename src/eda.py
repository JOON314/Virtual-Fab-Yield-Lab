"""EDA: 클래스 분포, 패턴별 예시 그리드, lot 구조 확인.
실행: python -m src.eda
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.common import load_npz, OUT, CMAP

def main():
    X, y, classes, lots = load_npz()
    out = os.path.join(OUT, "eda"); os.makedirs(out, exist_ok=True)
    counts = np.bincount(y, minlength=len(classes))

    # 1) 클래스 분포 (로그 스케일)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    order = np.argsort(counts)[::-1]
    bars = ax.bar([classes[i] for i in order], counts[order],
                  color=["#9aa0a6" if classes[i] == "none" else "#2f6fde" for i in order])
    ax.set_yscale("log"); ax.set_ylabel("wafers (log)")
    ax.set_title(f"WM-811K labeled wafers: {len(y):,}  —  'none' = {counts[classes.index('none')]/len(y):.1%}")
    for b, c in zip(bars, counts[order]):
        ax.text(b.get_x() + b.get_width()/2, c*1.08, f"{c:,}", ha="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "class_distribution.png"), dpi=160); plt.close(fig)

    # 2) 패턴별 예시 (클래스당 4장)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(4, len(classes), figsize=(len(classes)*1.5, 6.4))
    for j, c in enumerate(classes):
        idx = rng.choice(np.where(y == j)[0], 4, replace=False)
        for i, k in enumerate(idx):
            axes[i, j].imshow(X[k], cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
            axes[i, j].axis("off")
        axes[0, j].set_title(c, fontsize=9)
    fig.suptitle("Defect pattern examples (grey=good die, red=failed die)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "pattern_examples.png"), dpi=160); plt.close(fig)

    stats = {
        "n_labeled": int(len(y)),
        "class_counts": {c: int(n) for c, n in zip(classes, counts)},
        "none_ratio": float(counts[classes.index("none")] / len(y)),
        "imbalance_max_over_min": float(counts.max() / counts.min()),
        "n_lots": int(len(np.unique(lots))),
    }
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
