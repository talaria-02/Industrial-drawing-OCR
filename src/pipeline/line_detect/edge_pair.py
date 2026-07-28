# -*- coding: utf-8 -*-
"""엣지쌍 병합 (G0) — LSD가 선 하나를 경계 2개로 검출하는 문제의 해소.

[문제]
LSD는 "선"이 아니라 "밝기가 변하는 경계"를 검출한다. 그래서 도면에 그어진
선 하나가 선분 2개(위쪽 경계 + 아래쪽 경계)로 나온다. 실측:

    1px 선 -> 2개 검출, 간격 2.1px
    5px 선 -> 2개 검출, 간격 6.9px

즉 우리가 "선 812개"라고 불러온 것은 실제 선 ~400개를 두 번 센 것이고,
치수 매칭에서는 모든 치수선이 자기 복제본과 경쟁하고 있었다.

[판별 — 무엇이 획쌍이고 무엇이 아닌가]
나란히 붙은 두 선분이 항상 한 획의 양 경계는 아니다. 두 획 '사이의 여백'도
기하학적으로 똑같이 생겼다(평행, 근접, 방향 반대). 구분하려면 사이를 봐야 한다.

    한 획      : 사이가 검다(잉크), 바깥이 희다
    두 획 사이 : 사이가 희다(여백), 바깥이 검다      <- 정확히 반대

합성영상 검증에서 완전히 분리됐고(한 획 사이 0/바깥 255, 여백 사이 255/바깥 0),
실제 도면 3장에서도 간격 4~5px를 경계로 부호가 뒤집혔다.

[버린 가설 두 개 — 다시 시도하지 말 것]
  (1) LSD 방향(극성)으로 구분: 한 획도 여백도 둘 다 180°였다. 실제 도면에서
      잉크검사와 일치율 65.8% — 쓸 수 없다.
  (2) 선 굵기로 외형선/치수선 구분: 실제 도면 굵기 분포가 2.2~3.0px 단봉으로
      뭉쳐 전혀 갈리지 않았다(별개 실험). 굵기는 여기서 '기록'만 하고
      분류 근거로는 쓰지 않는다.

[고정 임계값 금지]
도면마다 선 진하기가 다르다(실측: 어떤 도면은 획 사이 밝기 179, 다른 도면은 2).
arrowhead.py가 도면 1장에 맞춘 절대 픽셀값(MIN_ARM_LENGTH=6.0 등)을 박아둬서
148장/12출처에서 깨질 위험을 안고 있는데, 같은 실수를 반복하지 않는다.
여기서는 간격 범위와 대비 임계를 모두 도면에서 자가보정한다.
"""
import numpy as np
from scipy.spatial import cKDTree

# 각도 허용치만 고정값 — 이건 도면 스케일과 무관한 기하 조건이라 안전하다.
ANGLE_THRESH_DEG = 1.5
# 자가보정 전, 후보를 모을 때 쓰는 넉넉한 간격 범위(px). 보정의 입력일 뿐
# 판정 기준이 아니다.
SCAN_GAP_MIN = 0.8
SCAN_GAP_MAX = 16.0
# 겹침이 이보다 짧으면 "나란히 놓였다"고 보지 않는다(짧은 쪽 길이 대비 비율).
MIN_OVERLAP_RATIO = 0.30
# 잉크검사 샘플 수(겹침 구간을 몇 등분해 훑을지).
N_SAMPLES = 15
# 바깥 밝기를 잴 때 경계에서 얼마나 더 나가서 잴지(px).
OUTSIDE_MARGIN = 3.0
# 사이 밝기에 쓸 백분위. 중앙값(50)을 쓰면 파선(숨은선)에서 잉크와 여백이
# 번갈아 나와 값이 뜨는데, 낮은 백분위를 쓰면 "가끔이라도 잉크가 있었나"를
# 보게 되어 파선이 살아남는다.
INSIDE_PCTL = 30


def estimate_ink_paper(gray):
    """Otsu로 그 도면의 잉크/바탕 대표 밝기를 추정. 반환 (ink, paper).

    [어두운 배경 도면 주의]
    "잉크는 어둡다"고 못 박으면 안 된다. CAD를 다크 테마로 캡처한 도면은
    검은 바탕에 흰 선이라 정반대다(실측: monami_clear는 91%가 밝기 32~64이고,
    그 상태에서 짝지음률이 4%로 무너졌다 — 일반 도면은 44%).

    그래서 밝기로 정하지 않고 '넓이'로 정한다. 도면은 어느 쪽이든 바탕이
    압도적으로 넓으므로, 화소가 많은 쪽이 바탕이고 적은 쪽이 잉크다.
    이렇게 하면 ink > paper 인 경우도 자연스럽게 나온다.
    """
    thr, _ = cv2_threshold_otsu(gray)
    dark = gray[gray <= thr]
    light = gray[gray > thr]
    if dark.size == 0 or light.size == 0:
        return 0.0, 255.0
    d_med, l_med = float(np.median(dark)), float(np.median(light))
    if dark.size >= light.size:
        return l_med, d_med       # 어두운 쪽이 넓다 -> 그쪽이 바탕(흰 선 도면)
    return d_med, l_med


def ink_polarity(gray):
    """잉크가 바탕보다 어두우면 +1, 밝으면 -1. 대비 부호를 맞추는 데 쓴다."""
    ink, paper = estimate_ink_paper(gray)
    return 1.0 if ink <= paper else -1.0


def cv2_threshold_otsu(gray):
    """cv2 의존을 이 함수 하나로 가둬둔다(테스트에서 갈아끼우기 쉽게)."""
    import cv2
    thr, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thr, binimg


def _geometry(lines):
    """선분 배열에서 중점/단위방향/법선/길이/각도(mod180)를 한번에 계산."""
    p1, p2 = lines[:, :2], lines[:, 2:]
    vec = p2 - p1
    length = np.hypot(vec[:, 0], vec[:, 1])
    safe = np.maximum(length, 1e-9)
    dirs = vec / safe[:, None]
    normals = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1)
    mid = (p1 + p2) / 2
    ang = np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) % 180.0
    return mid, dirs, normals, length, ang


def _candidate_pairs(lines, mid, radius):
    """끝점과 중점을 함께 색인해 '근처에 나란히 있을 수 있는' 선분쌍만 추린다.

    끝점만 색인하면 긴 획의 한쪽 경계가 조각나 있을 때 중간 조각을 놓친다.
    중점을 같이 넣어 그 경우를 일부 건진다(완전한 해결은 상위 파이프라인에서
    '조각 잇기'를 먼저 돌리는 것).
    """
    n = len(lines)
    pts = np.concatenate([lines[:, :2], lines[:, 2:], mid], axis=0)
    owner = np.tile(np.arange(n), 3)
    pairs = cKDTree(pts).query_pairs(r=radius, output_type='ndarray')
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    i, j = owner[pairs[:, 0]], owner[pairs[:, 1]]
    keep = i != j
    i, j = i[keep], j[keep]
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    return np.unique(np.stack([lo, hi], axis=1), axis=0)


def _pair_geometry(cand, mid, dirs, normals, length):
    """후보쌍마다 부호있는 수직거리(gap)와 겹침 구간을 계산.

    반환: gap(부호있음), overlap_len, overlap_center(i 기준 축방향 좌표),
          union_lo, union_hi(i 기준 축방향 — 출력 중심선의 범위로 씀)
    """
    i, j = cand[:, 0], cand[:, 1]
    delta = mid[j] - mid[i]
    gap = np.einsum('ij,ij->i', delta, normals[i])      # 부호있는 수직거리
    along = np.einsum('ij,ij->i', delta, dirs[i])       # i 중점 기준 j 중점의 축방향 위치
    hi_half, hj_half = length[i] / 2, length[j] / 2

    j_lo, j_hi = along - hj_half, along + hj_half
    ov_lo = np.maximum(-hi_half, j_lo)
    ov_hi = np.minimum(hi_half, j_hi)
    overlap_len = ov_hi - ov_lo
    overlap_center = (ov_lo + ov_hi) / 2

    union_lo = np.minimum(-hi_half, j_lo)
    union_hi = np.maximum(hi_half, j_hi)
    return gap, overlap_len, overlap_center, union_lo, union_hi


def _sample_contrast(gray, cand, mid, dirs, normals, gap, overlap_len, overlap_center,
                     n_samples=N_SAMPLES, outside_margin=OUTSIDE_MARGIN,
                     inside_pctl=INSIDE_PCTL):
    """후보쌍마다 '사이 밝기'와 '바깥 밝기'를 재서 대비를 반환.

    두 선분의 정확히 가운데를 지나는 선(중심선 후보)을 겹침 구간만큼 훑고,
    같은 구간을 법선 방향으로 양쪽 바깥으로 밀어서 한 번씩 더 훑는다.
    """
    H, W = gray.shape
    i = cand[:, 0]
    # 중심선 위의 기준점: i의 중점에서 축방향으로 겹침중심만큼, 법선방향으로 gap 절반만큼
    base = mid[i] + dirs[i] * overlap_center[:, None] + normals[i] * (gap[:, None] / 2)
    half = (overlap_len * 0.85 / 2)[:, None]
    ts = np.linspace(-1.0, 1.0, n_samples)[None, :]                  # (1,S)
    # (K,S,2) 샘플점
    axis_off = dirs[i][:, None, :] * (half[:, :, None] * ts[:, :, None])
    center_pts = base[:, None, :] + axis_off

    def grab(offset_px):
        pts = center_pts + normals[i][:, None, :] * offset_px[:, None, None]
        xs = np.clip(np.rint(pts[:, :, 0]).astype(np.int32), 0, W - 1)
        ys = np.clip(np.rint(pts[:, :, 1]).astype(np.int32), 0, H - 1)
        return gray[ys, xs]

    out_off = np.abs(gap) / 2 + outside_margin
    # 흰 선 도면에서는 '사이가 밝고 바깥이 어둡다'. 극성을 곱해 부호를 통일하면
    # 아래 판정식(대비 > 임계)을 두 경우 모두에 그대로 쓸 수 있다.
    pol = ink_polarity(gray)
    if pol < 0:
        g2 = 255 - gray.astype(np.int16)
        gray = np.clip(g2, 0, 255).astype(np.uint8)
        H, W = gray.shape

        def grab(offset_px):
            pts = center_pts + normals[i][:, None, :] * offset_px[:, None, None]
            xs = np.clip(np.rint(pts[:, :, 0]).astype(np.int32), 0, W - 1)
            ys = np.clip(np.rint(pts[:, :, 1]).astype(np.int32), 0, H - 1)
            return gray[ys, xs]

    inside = np.percentile(grab(np.zeros(len(cand))), inside_pctl, axis=1)
    out_a = np.median(grab(out_off), axis=1)
    out_b = np.median(grab(-out_off), axis=1)
    outside = np.minimum(out_a, out_b)
    return outside - inside, inside, outside


def _auto_gap_range(gaps, contrast):
    """그 도면의 대표 선 간격 g*를 후보 분포에서 직접 찾고 허용범위를 만든다.

    대비가 양수인(=사이가 바깥보다 어두운) 후보만 모아 히스토그램 최빈값을
    취한다. 이게 그 도면의 대표 획 간격이다. 실측 3장에서 2.6~3.3px로 안정적.
    """
    pos = gaps[contrast > 0]
    if len(pos) < 20:
        return SCAN_GAP_MIN, 6.0, float('nan')     # 표본 부족 시 보수적 기본값
    bins = np.arange(SCAN_GAP_MIN, SCAN_GAP_MAX + 0.25, 0.25)
    hist, edges = np.histogram(pos, bins=bins)
    mode = float(edges[int(np.argmax(hist))] + 0.125)
    return max(SCAN_GAP_MIN, mode * 0.5), min(SCAN_GAP_MAX, mode * 2.0), mode


def _greedy_assign(cand, score, n_lines):
    """한 선분이 여러 후보와 짝날 수 있으므로(해칭 등) 1:1로 확정한다.
    점수(대비) 높은 순 그리디 — assign_greedy와 같은 방식."""
    order = np.argsort(-score)
    used = np.zeros(n_lines, dtype=bool)
    chosen = []
    for k in order:
        a, b = cand[k]
        if used[a] or used[b]:
            continue
        used[a] = used[b] = True
        chosen.append(k)
    return np.array(chosen, dtype=np.int64), used


def _on_ink(lines, gray, pctl=INSIDE_PCTL, n_samples=13):
    """각 선분이 잉크 위에 놓였는지 bool 배열로 반환. 낮은 백분위를 쓰는 이유는
    파선(숨은선/중심선)이 '가끔이라도 잉크가 있으면' 통과하게 하기 위해서다."""
    H, W = gray.shape
    ink, paper = estimate_ink_paper(gray)
    cut = (ink + paper) / 2
    p1, p2 = lines[:, :2], lines[:, 2:]
    ts = np.linspace(0.08, 0.92, n_samples)[None, :, None]
    pts = p1[:, None, :] + (p2 - p1)[:, None, :] * ts
    xs = np.clip(np.rint(pts[:, :, 0]).astype(np.int32), 0, W - 1)
    ys = np.clip(np.rint(pts[:, :, 1]).astype(np.int32), 0, H - 1)
    v = gray[ys, xs]
    # 흰 선 도면이면 '잉크에 가깝다'가 밝은 쪽이다
    if ink <= paper:
        return np.percentile(v, pctl, axis=1) < cut
    return np.percentile(v, 100 - pctl, axis=1) > cut


def merge_edge_pairs(lines, gray, angle_thresh_deg=ANGLE_THRESH_DEG,
                     min_overlap_ratio=MIN_OVERLAP_RATIO,
                     gap_range=None, contrast_thresh=None,
                     verify_ink=True, ink_cut_pctl=INSIDE_PCTL):
    """LSD 선분에서 '한 획의 양쪽 경계'를 찾아 중심선 하나로 합친다.

    lines : (N,4) float — LSD 출력(조각 잇기를 먼저 돌린 것을 권장)
    gray  : 원본 그레이스케일. 잉크검사에 쓴다.
    gap_range, contrast_thresh : None이면 도면에서 자가보정(권장).
    verify_ink : 출력 직전 "중심선이 잉크 위에 있는가"를 재확인해 실패분을
                 미짝으로 되돌린다. 끄면 오병합이 그대로 나간다.

    반환 (out_lines, meta)
      out_lines : (M,4) — 병합된 중심선 + 짝 못 찾은 원본 선분
      meta      : dict — thickness(M,), paired(M,) bool, stats(dict)
    """
    lines = np.asarray(lines, dtype=np.float64)
    n = len(lines)
    if n < 2:
        return lines, {"thickness": np.zeros(n), "paired": np.zeros(n, bool),
                       "stats": {"n_in": n, "n_out": n, "n_pairs": 0}}

    mid, dirs, normals, length, ang = _geometry(lines)

    # 1) 후보 추리기 — 근접 + 평행
    cand = _candidate_pairs(lines, mid, radius=SCAN_GAP_MAX)
    if len(cand) == 0:
        return lines, {"thickness": np.zeros(n), "paired": np.zeros(n, bool),
                       "stats": {"n_in": n, "n_out": n, "n_pairs": 0}}
    da = np.abs(ang[cand[:, 0]] - ang[cand[:, 1]])
    da = np.minimum(da, 180 - da)
    cand = cand[da <= angle_thresh_deg]
    if len(cand) == 0:
        return lines, {"thickness": np.zeros(n), "paired": np.zeros(n, bool),
                       "stats": {"n_in": n, "n_out": n, "n_pairs": 0}}

    # 2) 간격/겹침 계산 후 넉넉한 범위로 1차 선별
    gap, ov_len, ov_c, u_lo, u_hi = _pair_geometry(cand, mid, dirs, normals, length)
    agap = np.abs(gap)
    shorter = np.minimum(length[cand[:, 0]], length[cand[:, 1]])
    ok = ((agap >= SCAN_GAP_MIN) & (agap <= SCAN_GAP_MAX) &
          (ov_len > 0) & (ov_len >= min_overlap_ratio * shorter))
    cand, gap, agap = cand[ok], gap[ok], agap[ok]
    ov_len, ov_c, u_lo, u_hi = ov_len[ok], ov_c[ok], u_lo[ok], u_hi[ok]
    if len(cand) == 0:
        return lines, {"thickness": np.zeros(n), "paired": np.zeros(n, bool),
                       "stats": {"n_in": n, "n_out": n, "n_pairs": 0}}

    # 3) 잉크검사
    contrast, inside, outside = _sample_contrast(
        gray, cand, mid, dirs, normals, gap, ov_len, ov_c)

    # 4) 자가보정 — 간격 범위와 대비 임계를 이 도면에서 직접 정한다
    if gap_range is None:
        g_lo, g_hi, g_mode = _auto_gap_range(agap, contrast)
    else:
        g_lo, g_hi = gap_range
        g_mode = float('nan')
    if contrast_thresh is None:
        ink, paper = estimate_ink_paper(gray)
        contrast_thresh = 0.25 * (paper - ink)
    else:
        ink, paper = float('nan'), float('nan')

    # 5) 최종 판정
    is_pair = (agap >= g_lo) & (agap <= g_hi) & (contrast > contrast_thresh)
    sel = np.where(is_pair)[0]
    if len(sel) == 0:
        return lines, {"thickness": np.zeros(n), "paired": np.zeros(n, bool),
                       "stats": {"n_in": n, "n_out": n, "n_pairs": 0,
                                 "gap_mode": g_mode, "contrast_thresh": contrast_thresh}}

    # 6) 1:1 배정
    chosen, used = _greedy_assign(cand[sel], contrast[sel], n)
    pick = sel[chosen]

    # 7) 중심선 생성 — 두 경계의 정확히 가운데, 축방향으로는 합집합 구간
    i = cand[pick, 0]
    base = mid[i] + normals[i] * (gap[pick, None] / 2)
    p0 = base + dirs[i] * u_lo[pick, None]
    p1 = base + dirs[i] * u_hi[pick, None]
    merged = np.concatenate([p0, p1], axis=1)

    # 8) 출력 검증 — 잉크 위에 놓이지 않은 중심선은 되돌린다.
    #
    # 대비검사는 '겹침 구간'만 15점 훑는데, 출력 중심선은 합집합 구간이라
    # 검사하지 않은 구간이 섞일 수 있다. 또 그리디 배정이 엉뚱한 짝을 고르면
    # (실측: 테두리선 옆 흰 공간에 중심선이 생기는 사례) 대비검사를 통과하고도
    # 결과가 틀린다. 어느 쪽이든 "중심선은 잉크 위에 있어야 한다"는 최종 조건을
    # 출력 직전에 한 번 더 강제하는 게 확실하다.
    #
    # 실패한 짝은 버리지 않고 원본 선분 두 개를 미짝으로 되돌린다 — 오병합이
    # '틀린 선'이 아니라 '병합 안 됨'으로 떨어져야 하류가 안전하다.
    if verify_ink and len(merged):
        keep = _on_ink(merged, gray, ink_cut_pctl)
        if not np.all(keep):
            used[cand[pick[~keep], 0]] = False
            used[cand[pick[~keep], 1]] = False
            pick, merged = pick[keep], merged[keep]
            i = cand[pick, 0]
        n_rejected = int(np.sum(~keep))
    else:
        n_rejected = 0

    leftover = np.where(~used)[0]
    out_lines = np.concatenate([merged, lines[leftover]], axis=0)
    thickness = np.concatenate([agap[pick], np.zeros(len(leftover))])
    paired = np.concatenate([np.ones(len(pick), bool), np.zeros(len(leftover), bool)])

    # 길이 불일치 — 부분겹침 분할 로직이 필요한지 판단할 근거로 기록
    if len(pick):
        li, lj = length[cand[pick, 0]], length[cand[pick, 1]]
        mismatch = np.abs(li - lj) / np.maximum(li, lj)
        thick_med = float(np.median(agap[pick]))
        mm_med, mm_p90 = float(np.median(mismatch)), float(np.percentile(mismatch, 90))
    else:
        thick_med = mm_med = mm_p90 = float("nan")

    stats = {
        "n_in": int(n),
        "n_out": int(len(out_lines)),
        "n_pairs": int(len(pick)),
        "n_rejected_by_ink": n_rejected,
        "n_unpaired": int(len(leftover)),
        "reduction_pct": float((1 - len(out_lines) / n) * 100),
        "gap_mode": float(g_mode),
        "gap_range": [float(g_lo), float(g_hi)],
        "contrast_thresh": float(contrast_thresh),
        "ink_level": float(ink),
        "paper_level": float(paper),
        "thickness_median": thick_med,
        "length_mismatch_median": mm_med,
        "length_mismatch_p90": mm_p90,
    }
    return out_lines, {"thickness": thickness, "paired": paired, "stats": stats}


def centerline_ink_rate(lines, gray, n_samples=11, pctl=50):
    """검증용 — 출력 중심선이 실제로 잉크 위에 놓였는지.

    여백을 잘못 병합하면 중심선이 흰 바탕에 뜬다. 오병합을 직접 잡아내는
    지표라 라벨 없이 쓸 수 있다. 반환: (잉크 위 비율, 중심선별 밝기 배열)
    """
    if len(lines) == 0:
        return float('nan'), np.array([])
    H, W = gray.shape
    ink, paper = estimate_ink_paper(gray)
    cut = (ink + paper) / 2
    p1, p2 = lines[:, :2], lines[:, 2:]
    ts = np.linspace(0.1, 0.9, n_samples)[None, :, None]
    pts = p1[:, None, :] + (p2 - p1)[:, None, :] * ts
    xs = np.clip(np.rint(pts[:, :, 0]).astype(np.int32), 0, W - 1)
    ys = np.clip(np.rint(pts[:, :, 1]).astype(np.int32), 0, H - 1)
    # 흰 선 도면이면 '잉크 위'가 밝은 쪽이다(estimate_ink_paper가 극성을 알려준다)
    if ink <= paper:
        vals = np.percentile(gray[ys, xs], pctl, axis=1)
        return float(np.mean(vals < cut)), vals
    vals = np.percentile(gray[ys, xs], 100 - pctl, axis=1)
    return float(np.mean(vals > cut)), vals
