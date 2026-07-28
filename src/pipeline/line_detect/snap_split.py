# -*- coding: utf-8 -*-
"""교차점 강제 분할 + 끝점 스내핑 — 선분 집합을 위상이 연결된 그래프로 만든다.

[왜 필요한가]
LSD 선분은 서로 닿아 있지 않다. 직교하는 모서리에서도 1~3px 벌어져 끝나는 일이
흔해서, '이어져 있는가'를 접촉으로 판정할 수 없다. 수학적 교차점을 계산해
거기서 강제로 쪼개고 끝점을 끌어당기면, 이후 모든 탐색이 그래프 순회가 된다.

[전부 쪼개면 안 된다 — 해칭]
모든 교차점에서 쪼개는 순진한 구현은 해칭에서 터진다. 해칭선 수십 개가 외형선
하나를 가로지르면 그 외형선이 수십 조각으로 부서지고, 노드가 폭발해서 오히려
그래프가 쓸모없어진다. 그래서 두 가지를 둔다:

  1. 분할 자격을 각도로 제한 — 거의 나란히 만나는 교차는 쪼개지 않는다.
     해칭은 외형선과 45도쯤에서 만나므로 각도만으로는 못 걸러지는데,
  2. 한 선분이 받을 수 있는 분할 수에 상한을 둔다. 상한을 넘는 선분은
     '해칭에 뒤덮인 것'으로 보고 통째로 남긴다(쪼개지 않는다).

[스냅 허용치는 상대값]
1~2px 같은 절대값을 쓰면 안 된다. 도면 축척이 2.7배 차이나므로(글자 높이
13~35px 실측) 스냅 허용치도 함께 커져야 한다.
"""
import numpy as np
from scipy.spatial import cKDTree

# 교차로 인정할 최소 교차각. 이보다 나란하면 '스치는' 것으로 보고 무시한다.
MIN_CROSS_ANGLE_DEG = 20.0
# 한 선분이 받을 수 있는 분할 수 상한. 넘으면 해칭에 뒤덮인 선으로 보고 보존한다.
MAX_SPLITS_PER_LINE = 6
# 선분 끝을 넘어서 이만큼(글자높이 비율)까지는 연장해서 교차를 인정한다 —
# LSD가 모서리에서 짧게 끝나는 것을 메우는 몫이다.
EXTEND_RATIO = 0.30
# 끝점을 교차점으로 끌어당기는 허용치(글자높이 비율).
SNAP_RATIO = 0.20


def _intersect(a, b):
    """두 선분이 놓인 무한직선의 교차점과 각 선분에서의 매개변수 t, u."""
    p, r = a[:2], a[2:] - a[:2]
    q, s = b[:2], b[2:] - b[:2]
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return None
    d = q - p
    t = (d[0] * s[1] - d[1] * s[0]) / den
    u = (d[0] * r[1] - d[1] * r[0]) / den
    return p + t * r, float(t), float(u)


def find_intersections(lines, text_h, min_angle_deg=MIN_CROSS_ANGLE_DEG):
    """쪼갤 만한 교차점을 찾는다. 반환 {line_idx: [t, ...]} (t는 0~1 매개변수)."""
    L = np.asarray(lines, float)
    n = len(L)
    if n < 2:
        return {}
    vec = L[:, 2:] - L[:, :2]
    length = np.hypot(vec[:, 0], vec[:, 1])
    ext = np.maximum(text_h * EXTEND_RATIO, 2.0) / np.maximum(length, 1e-6)
    mid = (L[:, :2] + L[:, 2:]) / 2
    ang = np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) % 180.0

    # 후보쌍은 '바운딩박스가 겹치는 것'만 본다.
    #
    # 처음에는 중점 KD-tree에 최대 선분길이를 반경으로 줬는데, 도면 테두리처럼
    # 아주 긴 선이 하나 있으면 반경이 도면 폭만큼 커져 사실상 전수비교가 된다
    # (실측: 30-1-2에서 48.5초). 균일 격자에 선분의 박스를 등록하고 같은 칸을
    # 공유하는 쌍만 꺼내면, 긴 선은 여러 칸에 걸치되 비교 대상은 그 칸 안으로
    # 한정된다.
    pad = np.maximum(text_h * EXTEND_RATIO, 2.0)
    x0 = np.minimum(L[:, 0], L[:, 2]) - pad
    x1 = np.maximum(L[:, 0], L[:, 2]) + pad
    y0 = np.minimum(L[:, 1], L[:, 3]) - pad
    y1 = np.maximum(L[:, 1], L[:, 3]) + pad
    cell = max(float(np.median(length)) * 2.0, 32.0)
    buckets = {}
    for i in range(n):
        for gx in range(int(x0[i] // cell), int(x1[i] // cell) + 1):
            for gy in range(int(y0[i] // cell), int(y1[i] // cell) + 1):
                buckets.setdefault((gx, gy), []).append(i)
    pairs = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a_ in range(len(members)):
            for b_ in range(a_ + 1, len(members)):
                i, j = members[a_], members[b_]
                if x1[i] < x0[j] or x1[j] < x0[i] or y1[i] < y0[j] or y1[j] < y0[i]:
                    continue          # 박스가 안 겹치면 교차 불가
                pairs.add((i, j) if i < j else (j, i))

    cuts = {}
    for i, j in pairs:
        da = abs(ang[i] - ang[j])
        da = min(da, 180 - da)
        if da < min_angle_deg:
            continue                       # 거의 나란함 — 스치는 교차는 무시
        r = _intersect(L[i], L[j])
        if r is None:
            continue
        _, t, u = r
        # 양쪽 선분 '안'이거나, 끝에서 조금 연장한 범위 안이어야 한다
        if not (-ext[i] <= t <= 1 + ext[i] and -ext[j] <= u <= 1 + ext[j]):
            continue
        # 실제 몸통을 자르는 쪽만 분할 대상으로 삼는다(끝 근처는 스냅으로 처리)
        if 0.02 < t < 0.98:
            cuts.setdefault(i, []).append(t)
        if 0.02 < u < 0.98:
            cuts.setdefault(j, []).append(u)
    return cuts


def split_lines(lines, text_h, max_splits=MAX_SPLITS_PER_LINE):
    """교차점에서 선분을 쪼갠다. 반환 (new_lines, origin_idx, stats).

    origin_idx[k] = 새 선분 k가 원래 어느 선분에서 나왔는지 — 기존 id 참조를
    유지하려면 이 대응이 필요하다.
    """
    L = np.asarray(lines, float)
    cuts = find_intersections(L, text_h)
    out, origin = [], []
    n_split = n_skipped = 0
    for i in range(len(L)):
        ts = sorted(set(round(t, 4) for t in cuts.get(i, [])))
        if not ts:
            out.append(L[i]); origin.append(i); continue
        if len(ts) > max_splits:
            # 해칭에 뒤덮인 선분 — 쪼개면 조각만 늘고 쓸 데가 없다
            out.append(L[i]); origin.append(i); n_skipped += 1; continue
        p, q = L[i, :2], L[i, 2:]
        prev = 0.0
        for t in ts + [1.0]:
            a = p + (q - p) * prev
            b = p + (q - p) * t
            if np.linalg.norm(b - a) >= 2.0:
                out.append(np.concatenate([a, b])); origin.append(i)
            prev = t
        n_split += 1
    return (np.array(out, float) if out else np.zeros((0, 4)), origin,
            {"n_in": len(L), "n_out": len(out), "n_split": n_split,
             "n_skipped_dense": n_skipped})


def snap_endpoints(lines, text_h, tol_ratio=SNAP_RATIO):
    """가까운 끝점끼리 한 점으로 모아 붙인다(자석). 반환 (lines, n_moved)."""
    L = np.asarray(lines, float).copy()
    if len(L) == 0:
        return L, 0
    tol = max(1.5, text_h * tol_ratio)
    pts = np.vstack([L[:, :2], L[:, 2:]])
    tree = cKDTree(pts)
    groups = tree.query_ball_tree(tree, tol)
    moved = 0
    seen = np.zeros(len(pts), bool)
    for k, grp in enumerate(groups):
        if seen[k] or len(grp) < 2:
            continue
        idx = [g for g in grp if not seen[g]]
        if len(idx) < 2:
            continue
        c = pts[idx].mean(axis=0)
        for g in idx:
            seen[g] = True
            if np.linalg.norm(pts[g] - c) > 1e-9:
                moved += 1
            if g < len(L):
                L[g, :2] = c
            else:
                L[g - len(L), 2:] = c
    return L, moved


def snap_and_split(lines, text_h):
    """스냅 먼저, 그다음 분할. 순서가 중요하다 — 끝점을 먼저 붙여야 교차점이
    실제 모서리와 일치하고, 어긋난 상태로 쪼개면 헛노드가 생긴다."""
    snapped, n_moved = snap_endpoints(lines, text_h)
    out, origin, st = split_lines(snapped, text_h)
    st["n_snapped_pts"] = n_moved
    return out, origin, st
