# -*- coding: utf-8 -*-
"""숫자(치수값) <-> 선분 DB 매칭.

기존 anchor_all_test.py 방식(숫자마다 근처만 crop해서 그때그때 Hough 실행)과
다르게, 이건 Stage A(run_line_detect.process_image_lsd)가 미리 만들어둔
"도면 전체 선분 DB"를 입력으로 받는다. 그래서:
  - 숫자마다 선 검출을 새로 안 돌림 (DB 하나를 전체 숫자가 공유)
  - "한 선분은 한 숫자에만" 같은 전역 제약을 걸 수 있음 (Step 5)

설계 근거(실측으로 확인한 4가지 사실)는 이 파일 함수 docstring에 각각 남김.

가중치 관련 주의: score_candidates()가 계산하는 개별 점수(거리/각도/중심성)는
그대로 신뢰할만하지만, 이를 합친 combined_score의 가중치(WEIGHTS)는 아직
데이터로 검증되지 않은 잠정값(각 1/3 동일가중)이다. 배치 실행 후 top-5
점수표를 보고 조정할 것 — 지금은 "전역배정이 되는지" 자체를 확인하는 단계.
"""
import numpy as np
from scipy.spatial import cKDTree

# 잠정 가중치 — 데이터 검증 전. 셋 다 동일가중으로 시작.
WEIGHTS = {"dist": 1 / 3, "angle": 1 / 3, "centrality": 1 / 3}

MIN_LINE_LENGTH = 15.0     # 이보다 짧으면 글자획 잔재로 보고 후보에서 제외
CANDIDATE_RADIUS = 150.0   # 텍스트 중심에서 이 거리(px) 이내 선분만 후보로
TEXT_BBOX_MARGIN = 2.0     # 텍스트 bbox 내부 판정 시 여유(px)


def text_angle_from_poly(poly):
    """OCR poly(회전 quad, 4개 꼭짓점)의 긴 변 방향 = 텍스트 진행방향.

    근거(실측): '45°' quad 긴변 -14.3도, '85'(세로쓰기) quad 긴변 87도 —
    지금까지는 poly를 min/max로 뭉개서 이 각도 정보를 그냥 버리고 있었음.
    이 각도를 살리면 "가로/세로만 보는 축 필터"를 버리고 실제 각도로
    비교할 수 있게 됨(45도 사선 케이스가 이걸로 풀릴 가능성 확인 — 사실④).

    반환: (angle_deg[0~180 mod], uncertain(bool), aspect_ratio)
    uncertain=True면 글자 하나짜리처럼 정사각형에 가까워 방향을 못 믿는다는 뜻
    (aspect_ratio < 1.2).
    """
    pts = np.asarray(poly, dtype=np.float64)
    edges = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
    lens = [float(np.hypot(*e)) for e in edges]

    pair0_len = (lens[0] + lens[2]) / 2
    pair1_len = (lens[1] + lens[3]) / 2
    if pair0_len >= pair1_len:
        long_edges, long_len, short_len = [edges[0], edges[2]], pair0_len, pair1_len
    else:
        long_edges, long_len, short_len = [edges[1], edges[3]], pair1_len, pair0_len

    angs = np.array([np.degrees(np.arctan2(e[1], e[0])) % 180 for e in long_edges])
    # mod-180 각도의 평균은 산술평균이 아니라 2배각 트릭으로 (0도와 179도가 사실 거의 같은 방향인데
    # 그냥 평균내면 90도가 되어버리는 것 방지 — refine.py의 merge_collinear와 같은 이유)
    a2 = np.radians(angs * 2)
    avg_ang = float(np.degrees(np.arctan2(np.mean(np.sin(a2)), np.mean(np.cos(a2)))) / 2 % 180)

    aspect = long_len / max(short_len, 1e-6)
    return avg_ang, aspect < 1.2, aspect


def filter_lines_in_text_regions(lines, text_bboxes, margin=TEXT_BBOX_MARGIN):
    """선분 DB에서, 어떤 텍스트 bbox 안에 두 끝점이 다 들어있는 선분을 제외.

    근거(실측): '85' bbox 안에서 (696,776)-(696,804) 길이29 선분이 검출됨
    — 이건 치수선이 아니라 숫자 '8'/'5' 글자 획 자체가 LSD에 잡힌 것.
    부분적으로만 겹치는 선분(치수선이 텍스트를 가로지르는 경우)은 남긴다."""
    if len(lines) == 0 or len(text_bboxes) == 0:
        return lines
    keep = np.ones(len(lines), dtype=bool)
    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
    for (bx0, by0, bx1, by1) in text_bboxes:
        bx0, by0, bx1, by1 = bx0 - margin, by0 - margin, bx1 + margin, by1 + margin
        inside = ((x1 >= bx0) & (x1 <= bx1) & (y1 >= by0) & (y1 <= by1) &
                  (x2 >= bx0) & (x2 <= bx1) & (y2 >= by0) & (y2 <= by1))
        keep &= ~inside
    return lines[keep]


def _dist_and_t(px, py, lines):
    """텍스트 중심(px,py)에서 각 선분까지의 최단거리 + 투영비율(raw, 클램프 안 함).

    t가 0~1 밖이면 텍스트 중심이 선분의 "연장선상"에 있다는 뜻 — 그 선분의
    실제 몸통(span) 근처가 아니라 옆으로 비껴서 우연히 가까운 경우일 수 있음
    (사실① 판별자: 진짜 매칭은 t≈0.5에 몰려있었음)."""
    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
    vx, vy = x2 - x1, y2 - y1
    L2 = np.maximum(vx ** 2 + vy ** 2, 1e-9)
    t_raw = ((px - x1) * vx + (py - y1) * vy) / L2
    t_c = np.clip(t_raw, 0, 1)
    cx, cy = x1 + t_c * vx, y1 + t_c * vy
    dist = np.hypot(px - cx, py - cy)
    return dist, t_raw


def score_candidates(target, lines, radius=CANDIDATE_RADIUS, min_length=MIN_LINE_LENGTH):
    """숫자 하나(target)에 대해, 반경 내 선분 후보를 모으고 개별 점수를 매긴다.

    target: dict with keys cx, cy, angle_deg, angle_uncertain
    반환: 후보 리스트(각 원소: dict — line_idx, dist, angle_diff, t_raw,
          centrality, angle_score, combined) — combined_score 내림차순 정렬.
    """
    if len(lines) == 0:
        return []

    length = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    dist, t_raw = _dist_and_t(target["cx"], target["cy"], lines)

    cand_mask = (dist <= radius) & (length >= min_length)
    idxs = np.where(cand_mask)[0]
    if len(idxs) == 0:
        return []

    line_ang = np.degrees(np.arctan2(
        lines[idxs, 3] - lines[idxs, 1], lines[idxs, 2] - lines[idxs, 0])) % 180

    if target["angle_uncertain"]:
        # 글자 하나짜리처럼 텍스트 방향을 못 믿는 경우 — 각도점수를 중립(0.5)으로.
        # 근거: 방향을 모르는데 억지로 맞다/틀리다 점수를 매기면 오히려 왜곡됨.
        angle_diff = np.full(len(idxs), np.nan)
        angle_score = np.full(len(idxs), 0.5)
    else:
        d = np.abs(line_ang - target["angle_deg"])
        angle_diff = np.minimum(d, 180 - d)
        angle_score = np.clip(1 - angle_diff / 90, 0, 1)

    t = t_raw[idxs]
    # t가 0.5(선분 정중앙)에 가까울수록 1점, 끝점이면 0점, 선분 밖(연장선상)이면
    # 슬랙(±0.25)까지만 봐주고 그 밖은 0점 처리 (근거①: 정답 4/4가 t≈0.5였음)
    within_slack = (t >= -0.25) & (t <= 1.25)
    centrality = np.where(within_slack, np.clip(1 - 2 * np.abs(t - 0.5), 0, 1), 0.0)

    dist_score = 1 - dist[idxs] / radius  # 0~1, 가까울수록 1

    combined = (WEIGHTS["dist"] * dist_score +
                WEIGHTS["angle"] * angle_score +
                WEIGHTS["centrality"] * centrality)

    out = []
    for k, li in enumerate(idxs):
        out.append({
            "line_idx": int(li),
            "line": lines[li].tolist(),
            "dist": float(dist[li]),
            "dist_score": float(dist_score[k]),
            "angle_diff": None if np.isnan(angle_diff[k]) else float(angle_diff[k]),
            "angle_score": float(angle_score[k]),
            "t_raw": float(t[k]),
            "centrality": float(centrality[k]),
            "length": float(length[li]),
            "combined": float(combined[k]),
        })
    out.sort(key=lambda r: -r["combined"])
    return out


def assign_greedy(all_candidates_by_target):
    """전역 배정: 점수 높은 순으로 (target, line) 쌍을 훑으며,
    target과 line이 둘 다 아직 안 쓰였으면 확정. 이미 쓰였으면 충돌로 로그.

    all_candidates_by_target: {target_idx: [candidate,...]} (score_candidates 출력)
    반환: (assignment: {target_idx: candidate}, conflicts: [로그 dict, ...])
    """
    pool = []
    for ti, cands in all_candidates_by_target.items():
        for c in cands:
            pool.append((c["combined"], ti, c))
    pool.sort(key=lambda x: -x[0])

    assigned_target = {}
    used_lines = {}  # line_idx -> target_idx that took it
    conflicts = []

    for score, ti, cand in pool:
        if ti in assigned_target:
            continue  # 이 숫자는 이미 더 높은 점수로 배정됨
        li = cand["line_idx"]
        if li in used_lines:
            conflicts.append({
                "target_idx": ti, "line_idx": li, "score": score,
                "lost_to_target_idx": used_lines[li],
            })
            continue
        assigned_target[ti] = cand
        used_lines[li] = ti

    return assigned_target, conflicts
