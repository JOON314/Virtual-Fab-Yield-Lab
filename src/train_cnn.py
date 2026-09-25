"""소형 CNN (64x64, 2채널: wafer 영역 / 불량 다이).

- 분할: lot 단위 (RF와 동일한 test 세트)
- 불균형 처리: train에서 'none'을 N_NONE장으로 다운샘플 + 클래스 가중 CE(역빈도^0.5)
- 증강: 90도 회전 + 좌우/상하 반전 (웨이퍼 패턴은 회전 불변)
- 모델 선택: val macro-F1 최고 epoch
- 교정(v2): 다운샘플로 생긴 'none' 과소예측 편향을 logit에 상수 bias로 보정, bias는 val에서만 탐색
실행: python -m src.train_cnn [--epochs 12]
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_
from sklearn.metrics import f1_score

from src.common import load_npz, split, evaluate, OUT, SEED

torch.manual_seed(SEED); np.random.seed(SEED)


class Net(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(2, 16), blk(16, 32), blk(32, 64), blk(64, 96))  # 64->4
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(96, n_cls))

    def forward(self, x):
        return self.head(self.f(x))


def to_tensor(Xu8):
    x = torch.from_numpy(Xu8)
    return torch.stack([(x > 0).float(), (x == 2).float()], 1)


def augment(x):
    k = np.random.randint(4)
    x = torch.rot90(x, k, (2, 3))
    if np.random.rand() < .5: x = torch.flip(x, (3,))
    return x


@torch.no_grad()
def logits_of(model, X, bs=1024):
    model.eval()
    return torch.cat([model(to_tensor(X[i:i + bs])) for i in range(0, len(X), bs)]).numpy()


def main(epochs=12, n_none=20000, bs=128):
    X, y, classes, lots = load_npz()
    none = classes.index("none"); C = len(classes)
    tr, va, te = split(y, lots, mode="lot")
    rng = np.random.default_rng(SEED)
    tr_none = tr[y[tr] == none]
    tr_bal = np.concatenate([tr[y[tr] != none], rng.choice(tr_none, min(n_none, len(tr_none)), replace=False)])
    cnt = np.bincount(y[tr_bal], minlength=C)
    w = torch.tensor((cnt.max() / cnt) ** 0.5, dtype=torch.float32)
    print("train(balanced):", len(tr_bal), dict(zip(classes, cnt.tolist())))

    model = Net(C)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, epochs=epochs,
                                                steps_per_epoch=int(np.ceil(len(tr_bal) / bs)))
    out = os.path.join(OUT, "cnn_lotsplit"); os.makedirs(out, exist_ok=True)
    hist, best, best_state = [], -1, None
    for ep in range(epochs):
        model.train(); t = time.time(); perm = rng.permutation(tr_bal); tot = 0.
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            xb = augment(to_tensor(X[b])); yb = torch.from_numpy(y[b])
            loss = F_.cross_entropy(model(xb), yb, weight=w, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(b)
        lv = logits_of(model, X[va])
        f1 = f1_score(y[va], lv.argmax(1), average="macro")
        hist.append({"epoch": ep + 1, "train_loss": tot / len(perm), "val_macro_f1": float(f1),
                     "sec": round(time.time() - t, 1)})
        print(hist[-1], flush=True)
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(out, "model.pt"))
    json.dump(hist, open(os.path.join(out, "history.json"), "w"), indent=2)

    # --- 교정: 'none' logit bias를 val에서 탐색 ---
    lv, lt = logits_of(model, X[va]), logits_of(model, X[te])
    grid = np.round(np.arange(-1, 6.01, 0.25), 2); scores = []
    for b in grid:
        z = lv.copy(); z[:, none] += b
        scores.append(f1_score(y[va], z.argmax(1), average="macro"))
    b_star = float(grid[int(np.argmax(scores))])
    json.dump({"grid": grid.tolist(), "val_macro_f1": scores, "best_bias": b_star},
              open(os.path.join(out, "calibration.json"), "w"), indent=2)

    m1 = evaluate(y[te], lt.argmax(1), classes, os.path.join(out, "v1_raw"), "CNN v1 (raw)")
    zt = lt.copy(); zt[:, none] += b_star
    m2 = evaluate(y[te], zt.argmax(1), classes, os.path.join(out, "v2_calibrated"),
                  f"CNN v2 (none-bias +{b_star})", extra={"none_bias": b_star})
    np.save(os.path.join(out, "test_logits.npy"), lt); np.save(os.path.join(out, "test_idx.npy"), te)
    print(f"v1 test macro-F1={m1['macro_f1']:.4f} acc={m1['accuracy']:.4f}")
    print(f"v2 test macro-F1={m2['macro_f1']:.4f} acc={m2['accuracy']:.4f} (bias {b_star})")

    # 학습 곡선
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], "o-"); ax[0].set_title("train loss")
    ax[1].plot([h["epoch"] for h in hist], [h["val_macro_f1"] for h in hist], "o-", color="#2f6fde")
    ax[1].set_title("val macro-F1")
    for a in ax: a.set_xlabel("epoch"); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "learning_curve.png"), dpi=160)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=12)
    main(ap.parse_args().epochs)
