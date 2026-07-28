# -*- coding: utf-8 -*-
"""도면 ↔ 제품사진 정합. 구멍(원) 배치를 지문처럼 써서 두 좌표계를 맞춘다.

[왜 원인가]
색이나 질감으로 도면과 사진을 비교할 수는 없다. 대신 양쪽을 같은 기하로
환원한다 — 사진에도 도면과 똑같은 파이프라인(LSD -> G0 -> 원 추출)을 돌린다.
그중 원이 특히 좋다. 구멍은 도면에도 원이고 사진에도 원이며, 여러 개가 이루는
배치는 지문처럼 유일하다. 플랜지의 볼트홀 16개면 사실상 오정합이 불가능하다.

[여기서는 RANSAC이 맞다]
원 피팅에는 RANSAC이 쓸모없었다(사슬에 이상치가 없어서 측정상 오히려 악화).
그러나 대응 문제는 정반대다 — 도면의 어느 원이 사진의 어느 원인지 모르고,
한쪽에만 있는 원도 많다. 이상치투성이라 무작위 표집이 제 역할을 한다.

[스케일이 덤으로 나온다]
사진은 ArUco 덕분에 mm 단위인데 도면은 픽셀이다. 두 대응쌍이 정해지면
거리비가 곧 도면의 px/mm이므로, 정합이 성공하면 도면 축척도 함께 확정된다.
치수값에서 축척을 따로 추정할 필요가 없어진다.
"""
import numpy as np

# 인라이어 판정 거리(mm). 측정 정확도(±0.3~1mm)를 감안한 값.
INLIER_MM = 2.5
# 반지름이 이 비율 넘게 어긋나면 같은 구멍으로 보지 않는다.
RADIUS_TOL = 0.25
# 두 원이 이보다 가까우면 방향을 정하기 어려워 표본으로 쓰지 않는다(px).
MIN_PAIR_DIST = 20.0


def extract_geometry(rect_bgr, stroke_gap=None):
    """보정된 사진에서 선분과 원을 뽑는다. 도면과 똑같은 파이프라인을 재사용한다."""
    import cv2
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.join(root, "pipeline") not in sys.path:
        sys.path.insert(0, os.path.join(root, "pipeline"))
    from line_detect import backend_lsd, refine, edge_pair, arc_detect

    gray = rect_bgr if rect_bgr.ndim == 2 else cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    raw = backend_lsd.detect(gray)
    pre = refine.merge_collinear(refine.length_nms(raw, 5.0), 1.5, 1.0, 15.0)
    merged, meta = edge_pair.merge_edge_pairs(pre, gray)
    gap = stroke_gap if stroke_gap is not None else meta["stats"].get("gap_mode")
    rest, arcs, st = arc_detect.extract_arcs(merged, gray, stroke_gap=gap)
    return {"lines": rest, "arcs": arcs, "stats": st,
            "circles": np.array([[a["cx"], a["cy"], a["r"]] for a in arcs], float)
                       if arcs else np.zeros((0, 3))}


def _similarity(p_src, p_dst):
    """두 점쌍 -> 닮음변환 (회전+등방스케일+이동). 반환 (s, R, t) 또는 None."""
    a0, a1 = p_src
    b0, b1 = p_dst
    va, vb = a1 - a0, b1 - b0
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-6 or nb < 1e-6:
        return None
    s = nb / na
    ang = np.arctan2(vb[1], vb[0]) - np.arctan2(va[1], va[0])
    c, sn = np.cos(ang), np.sin(ang)
    R = np.array([[c, -sn], [sn, c]])
    t = b0 - s * (R @ a0)
    return s, R, t


def apply_transform(pts, s, R, t):
    pts = np.asarray(pts, float).reshape(-1, 2)
    return (s * (R @ pts.T).T) + t


def register_circles(src_circles, dst_circles, inlier_mm=INLIER_MM,
                     radius_tol=RADIUS_TOL, max_trials=4000, seed=0,
                     allow_flip=True):
    """도면 원 -> 사진 원 대응을 찾는다.

    src_circles : (N,3) [x, y, r]  도면 (픽셀)
    dst_circles : (M,3) [x, y, r]  사진 (mm 단위 격자의 픽셀 = px_per_mm 곱해진 값)
    반환 dict 또는 None
      scale, R, t : src -> dst 변환.  scale은 곧 도면 픽셀당 사진 픽셀 수
      inliers     : [(src_idx, dst_idx), ...]
      rms_mm      : 인라이어 잔차

    거울상(뒤집힘)도 시도한다. 부품을 뒤집어 놓고 찍는 일이 실제로 있다.
    """
    S = np.asarray(src_circles, float).reshape(-1, 3)
    D = np.asarray(dst_circles, float).reshape(-1, 3)
    if len(S) < 2 or len(D) < 2:
        return None
    rng = np.random.default_rng(seed)
    best = None

    variants = [1.0, -1.0] if allow_flip else [1.0]
    for flip in variants:
        Sx = S.copy()
        Sx[:, 0] *= flip                      # x 반전 = 거울상
        for _ in range(max_trials):
            i, j = rng.choice(len(Sx), 2, replace=False)
            k, l = rng.choice(len(D), 2, replace=False)
            if (np.linalg.norm(Sx[i, :2] - Sx[j, :2]) < MIN_PAIR_DIST
                    or np.linalg.norm(D[k, :2] - D[l, :2]) < MIN_PAIR_DIST):
                continue
            m = _similarity((Sx[i, :2], Sx[j, :2]), (D[k, :2], D[l, :2]))
            if m is None:
                continue
            s, R, t = m
            # 반지름도 같은 배율로 커져야 한다 — 아니면 다른 구멍끼리 짝지은 것
            if (abs(s * Sx[i, 2] - D[k, 2]) > radius_tol * D[k, 2]
                    or abs(s * Sx[j, 2] - D[l, 2]) > radius_tol * D[l, 2]):
                continue

            moved = apply_transform(Sx[:, :2], s, R, t)
            pairs, used, resid = [], set(), []
            for a in range(len(Sx)):
                d = np.linalg.norm(D[:, :2] - moved[a], axis=1)
                rok = np.abs(s * Sx[a, 2] - D[:, 2]) <= radius_tol * np.maximum(D[:, 2], 1e-6)
                d = np.where(rok, d, np.inf)
                b = int(np.argmin(d))
                if d[b] <= inlier_mm and b not in used:
                    used.add(b)
                    pairs.append((a, b))
                    resid.append(d[b])
            if len(pairs) < 2:
                continue
            score = (len(pairs), -float(np.mean(resid)))
            if best is None or score > best[0]:
                best = (score, dict(scale=s, R=R, t=t, flip=flip,
                                    inliers=pairs,
                                    rms_mm=float(np.sqrt(np.mean(np.square(resid))))))
    if best is None:
        return None

    out = best[1]
    # 인라이어 전체로 다시 풀어 정밀도를 올린다(2점 표본은 표본일 뿐이다)
    ref = _refit(S, D, out)
    return ref or out


def _refit(S, D, sol):
    """인라이어 전체에 대한 최소자승 닮음변환(Umeyama)."""
    pairs = sol["inliers"]
    if len(pairs) < 2:
        return None
    A = S[[p[0] for p in pairs], :2].copy()
    A[:, 0] *= sol["flip"]
    B = D[[p[1] for p in pairs], :2]
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - ca, B - cb
    H = A0.T @ B0 / len(A)
    U, Sg, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, d]) @ U.T
    var = float(np.mean(np.sum(A0 ** 2, axis=1)))
    s = float(np.sum(Sg * [1, d]) / var) if var > 1e-9 else sol["scale"]
    t = cb - s * (R @ ca)
    resid = np.linalg.norm(apply_transform(A, s, R, t) - B, axis=1)
    return dict(scale=s, R=R, t=t, flip=sol["flip"], inliers=pairs,
                rms_mm=float(np.sqrt(np.mean(resid ** 2))))


def register_from_docs(drawing_arcs, photo_arcs, px_per_mm, **kw):
    """review 문서의 arcs와 사진 arcs로 정합. 반환에 도면 축척(px/mm)을 덧붙인다."""
    S = np.array([[a["center"][0], a["center"][1], a["r"]] for a in drawing_arcs], float) \
        if drawing_arcs else np.zeros((0, 3))
    D = np.array([[a["cx"], a["cy"], a["r"]] for a in photo_arcs], float) \
        if photo_arcs else np.zeros((0, 3))
    r = register_circles(S, D, **kw)
    if r is None:
        return None
    # scale = 사진픽셀 / 도면픽셀,  사진픽셀 = mm * px_per_mm
    # => 도면 1픽셀 = scale / px_per_mm  mm  => 도면 px/mm = px_per_mm / scale
    r["drawing_px_per_mm"] = float(px_per_mm / r["scale"]) if r["scale"] > 1e-9 else None
    return r
