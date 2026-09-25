"""WM-811K (LSWMD.pkl) 로더 + 전처리.

- LSWMD.pkl은 2019년 이전 pandas로 저장되어 최신 pandas에서 모듈 경로 오류가 난다.
  -> sys.modules 별칭으로 구 경로(pandas.indexes)를 현재 경로에 연결해 해결.
- 라벨(failureType)이 있는 웨이퍼만 추출하고, 64x64로 리사이즈(nearest)한 uint8 배열을
  data/wm811k_64.npz 로 저장한다. (0=칩 없음, 1=양품 다이, 2=불량 다이)

실행: python -m src.load  [--size 64]
"""
import argparse
import os
import sys
import pickle

import numpy as np
import pandas as pd

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Near-full", "Random", "Scratch", "none"]
RAW = os.path.join("data", "LSWMD.pkl")


def _install_pandas_compat():
    """구버전 pandas pickle 경로 호환 처리."""
    import pandas.core.indexes as idx
    sys.modules.setdefault("pandas.indexes", idx)
    sys.modules.setdefault("pandas.indexes.base", idx.base)
    sys.modules.setdefault("pandas.indexes.numeric", idx.base)
    sys.modules.setdefault("pandas.indexes.range", idx.range)


def load_raw(path: str = RAW) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 없음. Kaggle WM-811K zip을 풀어 data/에 넣으세요.")
    _install_pandas_compat()
    try:
        return pd.read_pickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")


def _label(v):
    """failureType은 [['Center']] 같은 중첩 배열이거나 빈 배열."""
    a = np.asarray(v)
    return str(a.ravel()[0]) if a.size > 0 else None


def resize_nearest(m: np.ndarray, size: int) -> np.ndarray:
    h, w = m.shape
    ri = (np.arange(size) * h / size).astype(int)
    ci = (np.arange(size) * w / size).astype(int)
    return m[np.ix_(ri, ci)]


def build(size: int = 64, out: str = None):
    out = out or os.path.join("data", f"wm811k_{size}.npz")
    df = load_raw()
    print(f"전체 웨이퍼: {len(df):,}")
    labels = df["failureType"].map(_label)
    df = df[labels.isin(CLASSES)].copy()
    df["label"] = labels[labels.isin(CLASSES)]
    print(f"라벨 있는 웨이퍼: {len(df):,}")
    X = np.stack([resize_nearest(np.asarray(m, dtype=np.uint8), size)
                  for m in df["waferMap"]])
    y = df["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy(np.int64)
    dims = np.array([np.asarray(m).shape for m in df["waferMap"]])
    lots = df["lotName"].astype(str).to_numpy()
    np.savez_compressed(out, X=X, y=y, dims=dims, lots=lots, classes=np.array(CLASSES))
    print(f"저장: {out}  X={X.shape}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    build(ap.parse_args().size)
