"""Virtual Fab v0 — 공정 이상 파라미터 → 가상 웨이퍼 맵 생성기 (씨앗 버전).

각 다이의 불량 확률 p(r, θ) = p0 + Σ(공정 이상 메커니즘의 기여) 로 모델링하고
베르누이 샘플링으로 웨이퍼 맵을 만든다. 메커니즘-공정 대응은 '가설'이며 README에 근거를 적는다.

검증: 실데이터로만 학습한 CNN이 가상 맵을 '의도한 패턴'으로 분류하는가? (sim-to-real 일관성)
  -> 생성기가 실제 패턴의 핵심 형태를 담았는지, 모델이 형태를 학습했는지를 동시에 점검.
실행: python -m src.virtual_fab   (results/cnn_lotsplit/model.pt 필요)
"""
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common import OUT, CMAP, SEED, plot_confusion
from src.train_cnn import Net, logits_of

S = 64
yy, xx = np.mgrid[0:S, 0:S]
R = np.hypot(yy - (S - 1) / 2, xx - (S - 1) / 2) / (S / 2)
TH = np.arctan2(yy - S / 2, xx - S / 2)
WAFER = R <= 1.0

# 메커니즘: (이름, 의도한 WM-811K 패턴, 공정 가설)
MECHS = {
    "cmp_center_pressure": ("Center", "CMP 헤드 중심 압력 과다 / 스핀코팅 중심부 두께 이상"),
    "annular_thermal":     ("Donut", "RTP·베이크 링 형태 온도 불균일"),
    "edge_etch_nonunif":   ("Edge-Ring", "식각·증착 반경 방향 불균일 (가장자리 플라즈마 밀도 변화)"),
    "edge_clamp_contact":  ("Edge-Loc", "척/클램프 접촉·가장자리 국부 오염"),
    "local_particle":      ("Loc", "국부 파티클 클러스터·국부 공정 이상"),
    "handling_scratch":    ("Scratch", "웨이퍼 이송·CMP 중 기계적 스크래치"),
    "chamber_particle":    ("Random", "챔버 파티클 오염 (공간적으로 무작위)"),
    "tool_excursion":      ("Near-full", "장비 이상으로 웨이퍼 전반 불량"),
    "healthy":             ("none", "정상 공정 (배경 결함만)"),
}


def p_map(mech, rng):
    p0 = rng.uniform(0.01, 0.06)                      # 배경 결함 밀도
    p = np.full((S, S), p0)
    if mech == "cmp_center_pressure":
        s = rng.uniform(0.15, 0.3); p += rng.uniform(0.6, 0.9) * np.exp(-(R / s) ** 2)
    elif mech == "annular_thermal":
        r0, w = rng.uniform(0.4, 0.6), rng.uniform(0.08, 0.14)
        p += rng.uniform(0.6, 0.9) * np.exp(-((R - r0) / w) ** 2)
    elif mech == "edge_etch_nonunif":
        w = rng.uniform(0.02, 0.05); p += rng.uniform(0.6, 0.95) / (1 + np.exp(-(R - 0.93) / w))
    elif mech == "edge_clamp_contact":
        t0, dt = rng.uniform(-np.pi, np.pi), rng.uniform(0.3, 0.8)
        dth = np.angle(np.exp(1j * (TH - t0)))
        p += rng.uniform(0.6, 0.9) * np.exp(-(dth / dt) ** 2) / (1 + np.exp(-(R - 0.8) / 0.04))
    elif mech == "local_particle":
        rc, tc, s = rng.uniform(0.2, 0.65), rng.uniform(-np.pi, np.pi), rng.uniform(0.08, 0.16)
        cy, cx = S / 2 + rc * S / 2 * np.sin(tc), S / 2 + rc * S / 2 * np.cos(tc)
        d = np.hypot(yy - cy, xx - cx) / (S / 2)
        p += rng.uniform(0.6, 0.9) * np.exp(-(d / s) ** 2)
    elif mech == "handling_scratch":
        a = rng.uniform(0, np.pi); c = rng.uniform(-0.4, 0.4) * S / 2; L = rng.uniform(0.5, 1.2) * S / 2
        u = (xx - S / 2) * np.cos(a) + (yy - S / 2) * np.sin(a)
        v = -(xx - S / 2) * np.sin(a) + (yy - S / 2) * np.cos(a) - c
        u0 = rng.uniform(-0.3, 0.3) * S / 2
        p += 0.9 * ((np.abs(v) < 0.8) & (np.abs(u - u0) < L / 2))
    elif mech == "chamber_particle":
        p += rng.uniform(0.2, 0.4)
    elif mech == "tool_excursion":
        p += rng.uniform(0.7, 0.9)
    return np.clip(p, 0, 1)


def sample(mech, rng):
    fail = rng.random((S, S)) < p_map(mech, rng)
    m = np.where(WAFER, 1, 0).astype(np.uint8); m[WAFER & fail] = 2
    return m


def main(n_per=300):
    rng = np.random.default_rng(SEED)
    classes = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
    mechs = list(MECHS)
    X = np.stack([sample(m, rng) for m in mechs for _ in range(n_per)])
    y = np.array([classes.index(MECHS[m][0]) for m in mechs for _ in range(n_per)])
    out = os.path.join(OUT, "virtual_fab"); os.makedirs(out, exist_ok=True)

    fig, axes = plt.subplots(3, len(mechs), figsize=(len(mechs) * 1.5, 5))
    for j, m in enumerate(mechs):
        for i in range(3):
            axes[i, j].imshow(X[j * n_per + i], cmap=CMAP, vmin=0, vmax=2, interpolation="nearest"); axes[i, j].axis("off")
        axes[0, j].set_title(f"{m}\n→ {MECHS[m][0]}", fontsize=7)
    fig.suptitle("Virtual Fab v0: process-anomaly parameters → synthetic wafer maps")
    fig.tight_layout(); fig.savefig(os.path.join(out, "synthetic_examples.png"), dpi=160); plt.close(fig)

    pt = os.path.join(OUT, "cnn_lotsplit", "model.pt")
    if not os.path.exists(pt):
        print("CNN 모델 없음: 생성 예시만 저장"); return
    model = Net(len(classes)); model.load_state_dict(torch.load(pt)); model.eval()
    bias = json.load(open(os.path.join(OUT, "cnn_lotsplit", "calibration.json")))["best_bias"]
    z = logits_of(model, X); z[:, classes.index("none")] += bias
    pred = z.argmax(1)
    cm = np.zeros((len(classes),) * 2, int); np.add.at(cm, (y, pred), 1)
    plot_confusion(cm, classes, os.path.join(out, "sim2real_confusion.png"),
                   f"Real-trained CNN on synthetic maps  (agreement {np.mean(pred == y):.1%})")
    res = {"n_per_mechanism": n_per, "overall_agreement": float(np.mean(pred == y)),
           "per_mechanism": {m: {"intended": MECHS[m][0], "hypothesis": MECHS[m][1],
                                 "agreement": float(np.mean(pred[y == classes.index(MECHS[m][0])] == classes.index(MECHS[m][0]))),
                                 "most_common_pred": classes[int(np.bincount(pred[y == classes.index(MECHS[m][0])], minlength=9).argmax())]}
                             for m in mechs}}
    json.dump(res, open(os.path.join(out, "sim2real.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
