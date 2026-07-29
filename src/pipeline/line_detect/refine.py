# -*- coding: utf-8 -*-
"""LSD가 뱉은 원시 선분(N,4)을 다듬는 기하학적 후처리.

순서: length_nms(약한 필터) -> merge_collinear -> length_nms(최종 필터) -> orthogonal_snap

이 파일이 우리가 겪은 두 실패의 직접 해법이다:
  - 65/85/100 스택형 치수사슬이 같은 선에 몰렸던 문제
    -> merge_collinear의 "각도차 AND 수직오프셋 AND 끝점거리" 3중 조건이
       tick 경계를 우연이 아니라 명시적 규칙으로 통제한다.
  - 45도 사선 리더가 엉뚱한 선에 매칭됐던 문제
    -> 이 파일 자체는 각도에 축 가정을 두지 않는다(직교 스냅도 90/0 근처만
       건드리고 나머지 각도는 그대로 둠). 사선은 LSD가 그냥 검출하고
       여기서도 그대로 살아남는다.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def _angle_len(lines):
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    # 선분은 방향이 없다(A->B나 B->A나 같은 선) -> mod 180으로 뒤집힘을 없앤다
    ang = np.degrees(np.arctan2(dy, dx)) % 180.0
    length = np.hypot(dx, dy)
    return ang, length


def length_nms(lines, min_length=10.0):
    """너무 짧은 선분(주로 글자 획 조각 노이즈)을 제거."""
    if len(lines) == 0:
        return lines
    _, length = _angle_len(lines)
    return lines[length >= min_length]


def _candidate_pairs_by_proximity(lines, endpoint_gap_px):
    """KD-tree로 "끝점이 서로 가까운 선분쌍"만 먼저 추린다.

    예전 버전은 n개 선분의 모든 쌍(n²)에 대해 각도/수직거리/끝점거리를 다
    계산했음 — n=6000이면 3600만 쌍짜리 행렬을 여러 개 만드는 꼴이라 복잡한
    도면(해칭 많은)에서 10초 넘게 걸림(실측: n_raw 7082 -> merge 13.1초).

    실제로 합쳐질 수 있는 쌍은 "끝점끼리 endpoint_gap_px(기본 15px) 이내로
    가까운 쌍"뿐이다 — 화면 반대편의 무관한 선분끼리는애초에 비교할 필요가
    없다. KD-tree의 query_pairs(r=반경)는 "반경 이내 점쌍"을 O(n log n)급
    으로 바로 찾아주므로, 그 결과를 선분쌍으로 변환하면 전수비교 없이
    후보를 좁힐 수 있다.

    반환: (K,2) 선분 인덱스쌍 배열 (i<j, 중복없음). 이 후보들은 "끝점거리
    조건"을 이미 만족한 상태이므로, 호출부에서는 각도/수직거리만 추가로
    검사하면 된다.
    """
    n = len(lines)
    pts = np.empty((2 * n, 2), dtype=np.float64)
    pts[0::2] = lines[:, :2]
    pts[1::2] = lines[:, 2:]
    tree = cKDTree(pts)
    point_pairs = tree.query_pairs(r=endpoint_gap_px, output_type='ndarray')
    if len(point_pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)

    li = point_pairs[:, 0] // 2
    lj = point_pairs[:, 1] // 2
    mask = li != lj  # 같은 선분의 두 끝점끼리는 제외
    li, lj = li[mask], lj[mask]
    lo, hi = np.minimum(li, lj), np.maximum(li, lj)
    cand = np.unique(np.stack([lo, hi], axis=1), axis=0)
    return cand


def _filter_by_angle_and_perp(lines, cand, angle_thresh_deg, perp_thresh_px):
    """proximity로 추린 후보쌍 중, 각도가 비슷하고 서로가 놓인 무한직선에서
    수직으로 멀지 않은 것만 남긴다 (평행하지만 다른 위치의 별개 선분,
    또는 우연히 끝점만 가까운 직교선 등을 걸러냄)."""
    if len(cand) == 0:
        return cand
    ang, length = _angle_len(lines)
    i_idx, j_idx = cand[:, 0], cand[:, 1]

    d = np.abs(ang[i_idx] - ang[j_idx])
    d = np.minimum(d, 180 - d)
    angle_ok = d <= angle_thresh_deg

    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
    mxi, myi = (x1[i_idx] + x2[i_idx]) / 2, (y1[i_idx] + y2[i_idx]) / 2
    mxj, myj = (x1[j_idx] + x2[j_idx]) / 2, (y1[j_idx] + y2[j_idx]) / 2

    vxi, vyi = (x2 - x1)[i_idx], (y2 - y1)[i_idx]
    perp_j_on_i = np.abs((mxj - x1[i_idx]) * vyi - (myj - y1[i_idx]) * vxi) / np.maximum(length[i_idx], 1e-6)
    vxj, vyj = (x2 - x1)[j_idx], (y2 - y1)[j_idx]
    perp_i_on_j = np.abs((mxi - x1[j_idx]) * vyj - (myi - y1[j_idx]) * vxj) / np.maximum(length[j_idx], 1e-6)
    perp_ok = np.maximum(perp_j_on_i, perp_i_on_j) <= perp_thresh_px

    return cand[angle_ok & perp_ok]


def merge_collinear(lines, angle_thresh_deg=1.5, perp_thresh_px=3.0,
                     endpoint_gap_px=15.0, max_iters=3):
    """공선(같은 직선 위) 선분들을 하나로 합친다. 조각난 tick 구간이나
    글자/기호에 가려 끊긴 치수선을 다시 잇는 단계."""
    lines = np.asarray(lines, dtype=np.float64)
    for _ in range(max_iters):
        n = len(lines)
        if n < 2:
            break

        cand = _candidate_pairs_by_proximity(lines, endpoint_gap_px)
        cand = _filter_by_angle_and_perp(lines, cand, angle_thresh_deg, perp_thresh_px)
        if len(cand) == 0:
            break  # 더 이상 합쳐질 게 없음

        rows = np.concatenate([cand[:, 0], cand[:, 1]])
        cols = np.concatenate([cand[:, 1], cand[:, 0]])
        data = np.ones(len(rows), dtype=np.uint8)
        adj_sparse = coo_matrix((data, (rows, cols)), shape=(n, n))
        n_comp, labels = connected_components(adj_sparse, directed=False)
        if n_comp == n:
            break  # 더 이상 합쳐질 게 없음

        ang, _ = _angle_len(lines)
        new_lines = []
        for comp_id in range(n_comp):
            idxs = np.where(labels == comp_id)[0]
            if len(idxs) == 1:
                new_lines.append(lines[idxs[0]])
                continue
            pts = np.concatenate([lines[idxs, :2], lines[idxs, 2:]], axis=0)
            # 선분은 방향이 없어서 각도를 그냥 평균내면 0도/180도가 서로 상쇄돼버림
            # (예: 0도와 179도는 사실 거의 같은 방향인데 산술평균은 90도가 됨)
            # -> 각도를 2배로 키워서 평균낸 뒤 다시 반으로 나누는 표준 기법으로 우회
            a2 = np.radians(ang[idxs] * 2)
            avg_ang = np.degrees(np.arctan2(np.mean(np.sin(a2)), np.mean(np.cos(a2)))) / 2
            dirv = np.array([np.cos(np.radians(avg_ang)), np.sin(np.radians(avg_ang))])
            center = pts.mean(axis=0)
            t = (pts - center) @ dirv
            p0 = center + t.min() * dirv
            p1 = center + t.max() * dirv
            new_lines.append([p0[0], p0[1], p1[0], p1[1]])
        lines = np.array(new_lines, dtype=np.float64)
    return lines


def _collinear_groups(lines, perp_px, radius):
    """끝점이 radius 이내이고 네 끝점이 공통직선에서 perp_px 이내인 선분들을
    union-find로 묶는다. 묶음은 넉넉하게 만들고, 실제 판단은 호출부에서
    1차원으로 투영해 '연속한 간격'을 보고 한다."""
    n = len(lines)
    pts = np.empty((2 * n, 2))
    pts[0::2] = lines[:, :2]
    pts[1::2] = lines[:, 2:]
    tree = cKDTree(pts)
    pp = tree.query_pairs(r=radius, output_type='ndarray')
    par = list(range(n))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    if len(pp):
        li, lj = pp[:, 0] // 2, pp[:, 1] // 2
        m = li != lj
        for i, j in zip(li[m], lj[m]):
            i, j = int(i), int(j)
            a, b = find(i), find(j)
            if a == b:
                continue
            if _endpoint_perp(lines, i, j) <= perp_px:
                par[a] = b
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


def _endpoint_perp(lines, i, j):
    """i가 놓인 무한직선에서 j의 두 끝점까지 수직거리(양방향 최대).

    각도차 비교를 쓰지 않는 이유: 길이 6px 조각의 각도 불확실도는 atan(1/6)
    = 9.5도다. 고정 1.5도 기준을 대면 같은 직선 위의 짧은 대시 둘이 탈락한다.
    끝점 수직거리는 길이에 자동으로 적응한다 — 긴 선분에는 각도보다 엄격하고,
    짧은 조각에는 관대하되 위치는 그대로 요구한다.
    """
    def one(a, b):
        vx, vy = a[2] - a[0], a[3] - a[1]
        ln = np.hypot(vx, vy)
        if ln < 1e-9:
            return np.inf
        d1 = abs((b[0] - a[0]) * vy - (b[1] - a[1]) * vx) / ln
        d2 = abs((b[2] - a[0]) * vy - (b[3] - a[1]) * vx) / ln
        return max(d1, d2)
    return max(one(lines[i], lines[j]), one(lines[j], lines[i]))


def _text_spans_gap(text_boxes, p, q):
    """p~q 구간(선분 위의 빈 틈)이 텍스트 상자에 덮여 있는지."""
    if text_boxes is None:
        return False
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    for bx0, by0, bx1, by1 in text_boxes:
        if bx0 <= mx <= bx1 and by0 <= my <= by1:
            return True
    return False


def merge_dash_trains(lines, perp_px=1.2, min_dashes=3, gap_over_dash_max=1.5,
                      gap_spread_max=2.5, radius=None, text_boxes=None):
    """점선(숨은선·중심선)과 글자에 끊긴 치수선을 하나의 선분으로 잇는다.

    [간격 크기로는 못 가른다 — 실측]
    도면 8장에서 '공선(수직 1px 이내)인데 안 합쳐진 선분쌍' 862개의 간격 분포를
    재보니 중앙값 66px, 대시 길이의 3.2배였고 작은 간격에 봉우리가 없었다. 즉
    간격 상한을 15 -> 60px으로 올리면 점선도 합쳐지지만 '우연히 같은 직선 위에
    있는 별개의 외형선'까지 합쳐진다. 임계값 조정으로 풀 문제가 아니다.

    [주기성으로 가른다]
    점선은 대시와 간격이 반복된다. 실측한 중심선은 대시 97px / 간격 52px로
    간격이 대시보다 짧고 고른데, 별개의 선들은 간격이 들쑥날쑥하다. 그래서
    공선 묶음을 1차원으로 투영해 '연속한 틈'만 보고
      (a) 틈이 대시보다 짧다        gap <= gap_over_dash_max * 대시중앙값
      (b) 틈끼리 고르다             gap <= gap_spread_max * 최소틈
      (c) 대시가 min_dashes개 이상
    셋을 모두 만족하는 구간만 잇는다. 하나라도 깨지면 거기서 끊어 별개로 둔다.

    [글자에 끊긴 선은 예외]
    치수선이 숫자에 가려 둘로 끊긴 경우는 주기성을 확인할 대시가 없다. 대신
    틈 가운데가 OCR 글자상자 안이면 잇는다 — 원래 한 선이었다는 직접 증거다.

    반환: (lines, stats)
    """
    L = np.asarray(lines, dtype=np.float64)
    if len(L) < 2:
        return L, {"n_trains": 0, "n_absorbed": 0, "n_text_bridged": 0}
    vec = L[:, 2:] - L[:, :2]
    length = np.hypot(vec[:, 0], vec[:, 1])
    if radius is None:
        # 대시 간격은 대시 길이 수준이므로 대표 길이에서 잡는다. 고정 px을
        # 쓰면 축척이 다른 도면에서 다시 어긋난다.
        radius = float(np.clip(np.median(length) * 4.0, 30.0, 400.0))

    groups = _collinear_groups(L, perp_px, radius)
    used = np.zeros(len(L), bool)
    merged, n_trains, n_bridged = [], 0, 0

    for g in groups:
        idx = np.array(g)
        # 묶음의 공통 방향 — 길이로 가중해서 짧은 조각의 흔들림을 억제한다
        a2 = np.radians((np.degrees(np.arctan2(vec[idx, 1], vec[idx, 0])) % 180.0) * 2)
        w = length[idx]
        av = np.arctan2(float((np.sin(a2) * w).sum()), float((np.cos(a2) * w).sum())) / 2
        dv = np.array([np.cos(av), np.sin(av)])
        origin = L[idx[0], :2]
        t1 = (L[idx, :2] - origin) @ dv
        t2 = (L[idx, 2:] - origin) @ dv
        lo, hi = np.minimum(t1, t2), np.maximum(t1, t2)
        order = np.argsort(lo)
        idx, lo, hi = idx[order], lo[order], hi[order]

        run = [0]
        gaps = []

        def flush(run, gaps):
            nonlocal n_trains, n_bridged
            if len(run) < 2:
                return
            members = idx[run]
            bridged = bool(gaps) and all(
                _text_spans_gap(text_boxes,
                                origin + hi[run[k]] * dv,
                                origin + lo[run[k + 1]] * dv)
                for k in range(len(run) - 1))
            if len(run) < min_dashes and not bridged:
                return
            a = origin + lo[run[0]] * dv
            b = origin + hi[run[-1]] * dv
            merged.append([a[0], a[1], b[0], b[1]])
            used[members] = True
            n_trains += 1
            if bridged:
                n_bridged += 1

        for k in range(1, len(idx)):
            gap = float(lo[k] - hi[run[-1]])
            if gap < 0.0:
                gap = 0.0
            dash_med = float(np.median(hi[run] - lo[run]))
            ok = gap <= gap_over_dash_max * max(dash_med, 1.0)
            if ok and gaps:
                ok = gap <= gap_spread_max * max(min(gaps), 1.0)
            if not ok:
                # 글자에 가려 끊긴 경우만 예외로 잇는다
                ok = _text_spans_gap(text_boxes, origin + hi[run[-1]] * dv,
                                     origin + lo[k] * dv)
            if ok:
                gaps.append(gap)
                run.append(k)
            else:
                flush(run, gaps)
                run, gaps = [k], []
        flush(run, gaps)

    out = L[~used]
    if merged:
        out = np.vstack([out, np.array(merged, dtype=np.float64)])
    return out, {"n_trains": n_trains, "n_absorbed": int(used.sum()),
                 "n_text_bridged": n_bridged}


def orthogonal_snap(lines, tol_deg=1.0):
    """0도/90도에 아주 가까운(±tol_deg 이내) 선분만 정확히 0/90도로 강제.

    허용치를 좁게(기본 1도) 잡는 이유: 우리 도면엔 45도 챔퍼(2x45°)나
    사선 인출선처럼 "삐뚤어진 게 아니라 원래 그런" 선이 실제로 존재한다.
    허용치를 넓히면 그런 의미있는 사선까지 90도로 끌려가 죽는다.

    반환: (스냅된 lines, 조정 로그 리스트[(index, before_deg, after_deg), ...])
    """
    lines = lines.copy()
    log = []
    ang, length = _angle_len(lines)
    for i in range(len(lines)):
        a = ang[i]
        target = None
        if a <= tol_deg or a >= 180 - tol_deg:
            target = 0.0
        elif abs(a - 90) <= tol_deg:
            target = 90.0
        if target is None:
            continue

        x1, y1, x2, y2 = lines[i]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = length[i] / 2
        rad = np.radians(target)
        dx, dy = np.cos(rad) * half, np.sin(rad) * half
        lines[i] = [cx - dx, cy - dy, cx + dx, cy + dy]
        log.append((i, float(a), float(target)))
    return lines, log


def refine_pipeline_g0(raw_lines, gray, min_length_pre=5.0, min_length_post=15.0,
                        angle_thresh_deg=1.5, endpoint_gap_px=15.0,
                        frag_perp_px=1.0, snap_tol_deg=1.0,
                        gap_range=None, contrast_thresh=None,
                        min_length_short=5.0):
    """엣지쌍 병합(G0)을 포함한 후처리. refine_pipeline의 후속 버전.

    [기존 refine_pipeline과 무엇이 다른가]
    기존은 merge_collinear(perp_thresh=3.0) 하나로 끝냈는데, 실측 선 굵기가
    2.6px라 3.0 > 2.6 이 되어 '한 획의 양쪽 경계'가 공선조각으로 오인돼 우발적
    으로 합쳐지고 있었다. 굵은 선(3px 초과)은 안 합쳐지므로 일관성이 없었다.

    그래서 두 가지를 분리한다:
      A. 조각 잇기   perp <= frag_perp_px(1.0)  — 굵기(2.6px)를 못 건너뛴다.
                     같은 경계의 끊긴 조각만 잇는다.
      B. 엣지쌍 병합 edge_pair.merge_edge_pairs — 잉크검사로 획쌍만 골라 병합.
      C. 중심선끼리 다시 조각 잇기
      D. 직교 스냅

    A를 B보다 먼저 두는 이유: 한 획의 위 경계가 3조각, 아래가 1조각으로 검출
    되면 B의 겹침 조건이 실패한다. 조각을 먼저 이어야 짝이 맞는다.
    frag_perp_px가 굵기보다 작아야 A가 엣지쌍을 건드리지 않는다.
    """
    from . import edge_pair

    lines = length_nms(raw_lines, min_length_pre)
    n_after_pre_filter = len(lines)

    # A. 같은 경계의 조각만 잇기 (굵기를 건너뛰지 않는 좁은 허용치)
    lines = merge_collinear(lines, angle_thresh_deg, frag_perp_px, endpoint_gap_px)
    n_after_defrag = len(lines)

    # B. 엣지쌍 -> 중심선
    lines, pair_meta = edge_pair.merge_edge_pairs(
        lines, gray, angle_thresh_deg=angle_thresh_deg,
        gap_range=gap_range, contrast_thresh=contrast_thresh)
    n_after_pairs = len(lines)

    # B'. 짧은 선분을 길이만으로 자르지 않는다.
    #
    # 예전에는 마지막에 15px 미만을 통째로 버렸는데, 도면의 짧은 실선(모따기,
    # tick, 짧은 치수선)이 같이 죽었다. 그렇다고 임계값만 낮추면 글자획이
    # 쏟아진다(실측: 잉크 조건만 걸면 XYZ 도면이 986 -> 1896개로 두 배).
    #
    # 대신 '짝을 찾았는가'로 가른다. G0에서 짝지어진 선분은 두 경계 사이가
    # 잉크임을 확인하고 출력검증까지 통과한 것이라, 6~15px 구간에서도 잉크
    # 적중률이 도면 4장 모두 100%였다. 반대로 짝 없는 짧은 선분은 21~61%로
    # 대부분 노이즈다. 그래서 짧아도 짝이 있으면 남기고, 짝이 없으면 기존
    # 길이 기준으로 자른다.
    #
    # 여기에 '자기 중심선이 잉크 위'인 경우를 더한다. 얇은 선(1~2px)은 LSD가
    # scale 0.8로 흐리는 과정에서 두 경계가 하나로 뭉쳐 애초에 짝이 생기지
    # 않는다 — 노이즈가 아니라 이미 중심선인데 짝이 없어 길이로 재단됐다.
    # 짝 여부와 달리 잉크검사는 선분 하나하나를 개별로 판정하므로 이 경우를
    # 건져낸다(실측 도면 6장 전부 recall 상승, Slide13은 +11.4%p, precis 유지).
    paired = pair_meta["paired"]
    if len(lines):
        seg_len = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
        lines = lines[paired | (seg_len >= min_length_post)
                      | edge_pair._on_ink(lines, gray)]
    n_after_short = len(lines)

    # C. 중심선끼리 조각 잇기
    lines = merge_collinear(lines, angle_thresh_deg, frag_perp_px, endpoint_gap_px)
    # 노이즈는 위에서 이미 걸렀으므로 여기서는 아주 짧은 잔재만 턴다
    lines = length_nms(lines, min_length_short)
    lines, snap_log = orthogonal_snap(lines, snap_tol_deg)

    stats = {
        "n_raw": int(len(raw_lines)),
        "n_after_pre_filter": int(n_after_pre_filter),
        "n_after_defrag": int(n_after_defrag),
        "n_after_pairs": int(n_after_pairs),
        "n_after_short_filter": int(n_after_short),
        "n_final": int(len(lines)),
        "n_snapped": len(snap_log),
        **{f"pair_{k}": v for k, v in pair_meta["stats"].items()},
    }
    return lines, stats


def refine_pipeline(raw_lines, min_length_pre=8.0, min_length_post=15.0,
                     angle_thresh_deg=1.5, perp_thresh_px=3.0, endpoint_gap_px=15.0,
                     snap_tol_deg=1.0):
    """전체 후처리 순서: 약한 길이필터 -> 공선병합 -> 최종 길이필터 -> 직교스냅.

    약한 필터를 병합 전에 먼저 하는 이유: median 16px짜리 글자획 노이즈를
    미리 쳐내야 병합 단계의 O(n^2) 연산량이 감당 가능한 수준으로 줄어든다
    (실측: 필터 전 2000+개 -> 필터 후 수백 개 수준)."""
    lines = length_nms(raw_lines, min_length_pre)
    n_after_pre_filter = len(lines)
    lines = merge_collinear(lines, angle_thresh_deg, perp_thresh_px, endpoint_gap_px)
    n_after_merge = len(lines)
    lines = length_nms(lines, min_length_post)
    lines, snap_log = orthogonal_snap(lines, snap_tol_deg)
    stats = {
        "n_raw": int(len(raw_lines)),
        "n_after_pre_filter": int(n_after_pre_filter),
        "n_after_merge": int(n_after_merge),
        "n_final": int(len(lines)),
        "n_snapped": len(snap_log),
    }
    return lines, stats
