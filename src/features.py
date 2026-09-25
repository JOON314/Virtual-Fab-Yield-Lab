"""도메인 특징 추출 (Wu et al. 2015 계열의 수작업 특징을 단순화).

- 영역 밀도(13): 중앙 3x3 격자 9개 + 가장자리 4사분면
- 반경 프로파일(8): 중심에서 바깥으로 링 8개의 불량률  <- Center/Donut/Edge-Ring 구분
- Radon 투영 통계(20): 선형 패턴(Scratch) 감지
- 최대 불량 영역 기하(6): 면적, 둘레, 이심률, solidity, 중심거리, 장축
- 전체 불량률(1)
실행: python -m src.features   -> data/features_64.npy
"""
import os
import numpy as np
from multiprocessing import Pool
from skimage.transform import radon
from skimage.measure import label, regionprops

from src.common import load_npz

S = 64
yy, xx = np.mgrid[0:S, 0:S]
R = np.hypot(yy - (S - 1) / 2, xx - (S - 1) / 2) / (S / 2)   # 0(중심)~1(가장자리)
THETA = np.linspace(0., 180., 36, endpoint=False)


def _interp(v, n=10):
    return np.interp(np.linspace(0, len(v) - 1, n), np.arange(len(v)), v)


def extract(m):
    wafer = m > 0
    fail = (m == 2)
    nw = max(wafer.sum(), 1)
    f = [fail.sum() / nw]
    # 반경 링 8개
    for k in range(8):
        hi = (k + 1) / 8 if k < 7 else np.inf   # 마지막 링은 가장자리 다이까지 포함
        ring = wafer & (R >= k / 8) & (R < hi)
        f.append(fail[ring].mean() if ring.any() else 0.)
    # 영역 밀도 13
    inner = (R < 0.66)
    q = S // 3
    for i in range(3):
        for j in range(3):
            cell = wafer & inner & (yy >= i * q) & (yy < (i + 1) * q) & (xx >= j * q) & (xx < (j + 1) * q)
            f.append(fail[cell].mean() if cell.any() else 0.)
    ang = np.arctan2(yy - S / 2, xx - S / 2)
    for a in range(4):
        sec = wafer & ~inner & (ang >= -np.pi + a * np.pi / 2) & (ang < -np.pi + (a + 1) * np.pi / 2)
        f.append(fail[sec].mean() if sec.any() else 0.)
    # Radon
    sino = radon(fail.astype(float), theta=THETA, circle=False)
    f += list(_interp(sino.mean(0))) + list(_interp(sino.std(0)))
    # 최대 불량 연결영역
    lab = label(fail, connectivity=2)
    if lab.max() > 0:
        rp = max(regionprops(lab), key=lambda r: r.area)
        cy, cx = rp.centroid
        f += [rp.area / nw, rp.perimeter / S, rp.eccentricity, rp.solidity,
              np.hypot(cy - S / 2, cx - S / 2) / (S / 2), rp.axis_major_length / S]
    else:
        f += [0.] * 6
    return np.asarray(f, dtype=np.float32)


def main(out=os.path.join("data", "features_64.npy")):
    X, y, classes, lots = load_npz()
    with Pool(os.cpu_count()) as p:
        F = np.stack(p.map(extract, X, chunksize=2000))
    np.save(out, F)
    print("features:", F.shape, "->", out)


if __name__ == "__main__":
    main()
