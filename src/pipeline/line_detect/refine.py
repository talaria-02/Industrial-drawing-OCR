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


def refine_pipeline_g0(raw_lines, gray, min_length_pre=8.0, min_length_post=15.0,
                        angle_thresh_deg=1.5, endpoint_gap_px=15.0,
                        frag_perp_px=1.0, snap_tol_deg=1.0,
                        gap_range=None, contrast_thresh=None):
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

    # C. 중심선끼리 조각 잇기
    lines = merge_collinear(lines, angle_thresh_deg, frag_perp_px, endpoint_gap_px)
    lines = length_nms(lines, min_length_post)
    lines, snap_log = orthogonal_snap(lines, snap_tol_deg)

    stats = {
        "n_raw": int(len(raw_lines)),
        "n_after_pre_filter": int(n_after_pre_filter),
        "n_after_defrag": int(n_after_defrag),
        "n_after_pairs": int(n_after_pairs),
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
