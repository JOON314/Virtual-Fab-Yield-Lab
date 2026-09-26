"""사람 판정(review_sheet.csv) 집계 -> results/errors/review_summary.png / .json
실행: python -m src.review
"""
import json
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common import OUT

COLORS = {"label_noise": "#d62728", "model_limit": "#2f6fde", "ambiguous": "#9aa0a6"}
NAMES = {"label_noise": "label noise (label is wrong)", "model_limit": "model limit (model missed)",
         "ambiguous": "ambiguous"}


def main():
    p = os.path.join(OUT, "errors", "review_sheet.csv")
    d = pd.read_csv(p)
    d["verdict"] = d["verdict"].str.strip().str.lower()
    if d["verdict"].isna().any():
        raise ValueError("verdict가 비어 있는 행이 있습니다")
    d["pair"] = d["true"] + " → " + d["pred"]
    t = d.groupby(["pair", "verdict"]).size().unstack(fill_value=0)
    t = t.reindex(columns=[c for c in COLORS if c in t.columns])
    t = t.loc[t.sum(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    left = None
    for v in t.columns:
        ax.barh(t.index, t[v], left=left, color=COLORS[v], label=NAMES[v])
        left = t[v] if left is None else left + t[v]
    ax.set_xlabel("wafers (human-reviewed)")
    total = d["verdict"].value_counts()
    ax.set_title(f"Human review of {len(d)} errors — label noise {total.get('label_noise', 0)}, "
                 f"model limit {total.get('model_limit', 0)}, ambiguous {total.get('ambiguous', 0)}", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "errors", "review_summary.png"), dpi=160)
    out = {"n": int(len(d)), "total": {k: int(v) for k, v in total.items()},
           "by_pair": {k: {c: int(n) for c, n in r.items()} for k, r in t.iterrows()}}
    json.dump(out, open(os.path.join(OUT, "errors", "review_summary.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
