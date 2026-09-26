"""모델 비교표 + 오분류 분석.
- results/summary.md, results/model_comparison.png
- 최종 모델의 상위 혼동 쌍 3개 × 샘플 6장 그리드 -> results/errors/
- results/errors/review_sheet.csv : 사람이 직접 판정(라벨 노이즈 / 모델 한계 / 애매)할 시트
실행: python -m src.analyze
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common import load_npz, OUT, CMAP

MODELS = [("RF (random split)", "rf_randomsplit"), ("RF (lot split)", "rf_lotsplit"),
          ("CNN v1 raw", "cnn_lotsplit/v1_raw"), ("CNN v2 calibrated", "cnn_lotsplit/v2_calibrated"),
          ("CNN v3 max-pool resize", "cnn_lotsplit_maxpool/v2_calibrated")]


def load_metrics():
    rows = []
    for label, d in MODELS:
        p = os.path.join(OUT, d, "metrics.json")
        if os.path.exists(p):
            rows.append((label, json.load(open(p))))
    return rows


def comparison(rows, classes):
    lines = ["| model | macro-F1 | defect-only macro-F1 | accuracy | " +
             " | ".join(f"{c} R" for c in classes) + " |",
             "|" + "---|" * (4 + len(classes))]
    for label, m in rows:
        lines.append(f"| {label} | {m['macro_f1']:.3f} | {m['macro_f1_defect_only']:.3f} | {m['accuracy']:.3f} | " +
                     " | ".join(f"{m['per_class_recall'][c]:.2f}" for c in classes) + " |")
    open(os.path.join(OUT, "summary.md"), "w").write(
        "# Test results (lot-split test set unless noted)\n\nR = recall\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))

    lot_rows = [(l, m) for l, m in rows if "random" not in l]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(lot_rows); xs = np.arange(len(classes))
    colors = ["#9aa0a6", "#c6d7f5", "#8fb3f0", "#1f4fae"]
    for k, (l, m) in enumerate(lot_rows):
        ax.bar(xs + k * w, [m["per_class_f1"][c] for c in classes], w,
               label=f"{l} (macro-F1 {m['macro_f1']:.3f})", color=colors[k % len(colors)])
    ax.set_xticks(xs + w * (len(lot_rows) - 1) / 2, classes, fontsize=9)
    ax.set_ylabel("F1 (test, unseen lots)"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "model_comparison.png"), dpi=160); plt.close(fig)


def errors(classes, X, y):
    d = os.path.join(OUT, "cnn_lotsplit")
    cal = json.load(open(os.path.join(d, "calibration.json")))
    lt = np.load(os.path.join(d, "test_logits.npy")); te = np.load(os.path.join(d, "test_idx.npy"))
    none = classes.index("none")
    lt[:, none] += cal["best_bias"]
    p = np.exp(lt - lt.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    pred = p.argmax(1); yt = y[te]
    cm = np.zeros((len(classes),) * 2, int)
    np.add.at(cm, (yt, pred), 1)
    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = [np.unravel_index(i, off.shape) for i in np.argsort(off.ravel())[::-1][:3]]
    out = os.path.join(OUT, "errors"); os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0); sheet = []
    fig, axes = plt.subplots(3, 6, figsize=(11, 6.2))
    for r, (t, pr) in enumerate(pairs):
        cand = np.where((yt == t) & (pred == pr))[0]
        pick = rng.choice(cand, min(6, len(cand)), replace=False)
        for c, k in enumerate(pick):
            axes[r, c].imshow(X[te[k]], cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
            axes[r, c].set_title(f"#{te[k]}  p={p[k, pr]:.2f}", fontsize=7); axes[r, c].axis("off")
            sheet.append({"wafer_idx": int(te[k]), "true": classes[t], "pred": classes[pr],
                          "confidence": round(float(p[k, pr]), 3), "verdict(label_noise/model_limit/ambiguous)": "", "note": ""})
        for c in range(len(pick), 6): axes[r, c].axis("off")
        axes[r, 0].text(-0.15, 0.5, f"true {classes[t]}\n→ pred {classes[pr]}\n(n={off[t, pr]})",
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=9)
    fig.suptitle("Top-3 confusions of final CNN on unseen lots")
    fig.tight_layout(); fig.savefig(os.path.join(out, "top_confusions.png"), dpi=160, bbox_inches="tight"); plt.close(fig)
    # 확신도 높은 오류 = 라벨 노이즈 후보
    wrong = np.where(pred != yt)[0]
    top = wrong[np.argsort(-p[wrong, pred[wrong]])[:12]]
    fig, axes = plt.subplots(2, 6, figsize=(11, 4.4))
    for a, k in zip(axes.ravel(), top):
        a.imshow(X[te[k]], cmap=CMAP, vmin=0, vmax=2, interpolation="nearest"); a.axis("off")
        a.set_title(f"label {classes[yt[k]]}\npred {classes[pred[k]]} ({p[k, pred[k]]:.2f})", fontsize=7)
        sheet.append({"wafer_idx": int(te[k]), "true": classes[yt[k]], "pred": classes[pred[k]],
                      "confidence": round(float(p[k, pred[k]]), 3), "verdict(label_noise/model_limit/ambiguous)": "", "note": "high-confidence error"})
    fig.suptitle("Most confident errors — label-noise candidates (human review)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "confident_errors.png"), dpi=160); plt.close(fig)
    sheet_path = os.path.join(out, "review_sheet.csv")
    if os.path.exists(sheet_path):   # 사람 판정이 담긴 시트는 절대 덮어쓰지 않는다
        sheet_path = os.path.join(out, "review_sheet_new.csv")
    with open(sheet_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0].keys())); w.writeheader(); w.writerows(sheet)
    print("top confusions:", [(classes[t], classes[pr], int(off[t, pr])) for t, pr in pairs])


if __name__ == "__main__":
    X, y, classes, lots = load_npz()
    comparison(load_metrics(), classes)
    if os.path.exists(os.path.join(OUT, "cnn_lotsplit", "test_logits.npy")):
        errors(classes, X, y)
