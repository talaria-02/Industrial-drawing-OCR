# -*- coding: utf-8 -*-
"""화살촉(arrowhead) 검출 — 선분 DB만으로, 규칙 기반.

[왜 딥러닝을 안 쓰는가]
P&ID 논문(Moon et al. 2026)은 화살표를 Sparse R-CNN으로 검출한다 —
9종류(fat/standard/slim × sharp/round) 분류에 화살표 32,590개 라벨셋을
썼다. 우리는 그런 데이터도, GPU도 없다(torch/paddle CPU-only 확인됨).

그런데 우리 문제는 더 좁다. "이 치수선 끝점에 화살촉이 있는가" 이진 판정만
필요하고, 어디를 봐야 하는지도 이미 안다(치수선 후보의 끝점 근처만). 이건
전역 탐색이 아니라 국소 기하 패턴 매칭이라, Stage A(refine.py의 공선성
병합)와 같은 성격의 문제 — 규칙으로 푼다.

[근거: 실측된 화살촉 패턴]
'65' 치수선(세로, ~90도) 끝점 근처에서 실제로 검출된 것:
  (699,592)-(702,577)  len 15.6  각도 100.0도
  (703,577)-(706,592)  len 15.6  각도  79.9도
공유점(702,577) 근처에서 두 짧은 선분이 만나고, 그 방향이 "치수선 본체
쪽으로 되꺾인" 방향(치수선이 아래로 뻗어있으니, 화살촉 팔은 그 반대 —
위에서 아래로 되짚어가는 방향) 기준으로 좌우 대칭(약 ±10~11도)이다.
이게 실제 CAD 화살촉(">" 나 "<" 모양)의 기하학적 서명이다.
"""
import numpy as np
from scipy.spatial import cKDTree

# 실측(65/85/100/400 치수선 끝점)에 맞춘 기본값. 다른 스케일 도면에서는
# 재조정 필요할 수 있음 — 아직 10장 배치로 검증 전.
MIN_ARM_LENGTH = 6.0     # 이보다 짧으면 노이즈
MAX_ARM_LENGTH = 45.0    # 이보다 길면 화살촉 팔이 아니라 별개의 선
# "끝점 근처"로 볼 반경(px). 실측: '65' 치수선 끝점(594)과 화살촉 팁(577)의
# 거리가 17.4px — 화살촉이 선 몸통보다 한 발 더 튀어나온 위치에 있어서
# 예상보다 여유를 더 둬야 함(15px로는 놓침, 실측 확인 후 25로 조정).
SEARCH_RADIUS = 25.0
MAX_ARM_ANGLE_DEG = 45.0 # 팔이 "화살촉 팁이 있어야 할 연장방향"에서 이 각도 이내여야 후보


def _angle_deg(vx, vy):
    return np.degrees(np.arctan2(vy, vx))


def _signed_angle_diff(a, b):
    """a-b를 -180~180 범위로 정규화한 부호있는 각도차."""
    return ((a - b + 180) % 360) - 180


class ArrowIndex:
    """선분 DB 하나에 대해 KD-tree를 한 번만 만들어두고 여러 끝점 질의에 재사용.
    (refine.py의 _candidate_pairs_by_proximity와 같은 이유 — 매번 전수비교하면 느림)"""

    def __init__(self, lines):
        self.lines = np.asarray(lines, dtype=np.float64)
        n = len(self.lines)
        self.pts = np.empty((2 * n, 2), dtype=np.float64)
        self.pts[0::2] = self.lines[:, :2]
        self.pts[1::2] = self.lines[:, 2:]
        self.tree = cKDTree(self.pts) if n > 0 else None
        self.length = np.hypot(self.lines[:, 2] - self.lines[:, 0],
                                self.lines[:, 3] - self.lines[:, 1])

    def detect_at_endpoint(self, line_idx, endpoint,
                            search_radius=SEARCH_RADIUS,
                            min_arm_len=MIN_ARM_LENGTH,
                            max_arm_len=MAX_ARM_LENGTH,
                            max_arm_angle_deg=MAX_ARM_ANGLE_DEG):
        """line_idx번 선분의 endpoint('start' 또는 'end') 쪽에 화살촉이 있는지 판정.

        원리:
          1) 그 끝점 P 근처(search_radius 이내)에 자기 끝점 하나가 있는 "다른" 짧은
             선분들을 모음 (길이가 min~max_arm_len 범위인 것만 — 화살촉 팔 크기대).
             이 후보들의 "P에 가까운 쪽 끝점"이 화살촉의 밑변, "먼 쪽 끝점"이 화살촉의
             뾰족한 끝(tip)이다 — 즉 화살촉은 P보다 한 발 더 나간 위치에 있다
             (실측: '65' 치수선은 P=(702,594)에서 끝나지만 화살촉 팁은 (702,577)까지
             17.4px 더 뻗어있음 — 화살촉이 선분 몸통 뒤로 접히는 게 아니라, 선이
             원래 뻗어오던 방향으로 좀 더 튀어나온 뒤 좌우로 갈라지는 모양이기 때문).
          2) 팔의 방향(arm_dir)은 그 팔 자기 자신의 두 끝점(가까운 쪽->먼 쪽)으로
             계산한다. P를 기준으로 계산하면 안 됨 — P가 팔의 실제 밑변과 몇 px
             어긋나 있을 수 있는데(위 예시 4.2px), 팔 길이가 15px 안팎이라 그
             몇 px 오차가 각도로 크게 증폭돼서 완전히 다른 값이 나온다(실측:
             P기준 -92.9도 vs 팔 자체 기준 -80.0도, 13도 차이 — 대칭판정을 망가뜨림).
          3) 기준 방향은 "P에서 선분(L) 반대쪽 끝점 Q로 되돌아가는 방향"이 아니라
             "Q에서 P로 다가와 그대로 연장되는 방향"(= 화살촉 팁이 있어야 할 방향)
             이다. 이 두 방향은 정확히 반대(180도 차이)라서 반대로 잡으면 모든
             화살촉이 "180도 어긋남"으로 걸러져 하나도 검출이 안 된다.
          4) 각도차가 양수인 후보 하나 + 음수인 후보 하나가 함께 있으면(팔이 좌우로
             벌어진 V자) 화살촉으로 판정. 신뢰도는 좌우 대칭성 + 팔 길이 유사성으로 계산.

        반환: dict(found, confidence, arm_pos, arm_neg, n_candidates)
        """
        if self.tree is None:
            return {"found": False, "confidence": 0.0, "n_candidates": 0}

        L = self.lines[line_idx]
        if endpoint == "start":
            P, Q = L[:2], L[2:]
        else:
            P, Q = L[2:], L[:2]

        # 화살촉 팁은 "Q->P로 다가온 방향을 그대로 연장한 자리"에 있다(사실 확인됨).
        approach_dir = _angle_deg(P[0] - Q[0], P[1] - Q[1])

        near_pt_idxs = self.tree.query_ball_point(P, r=search_radius)
        candidates = []
        for pi in near_pt_idxs:
            j = pi // 2
            if j == line_idx:
                continue
            if not (min_arm_len <= self.length[j] <= max_arm_len):
                continue
            near_pt = self.pts[pi]
            far_pi = pi + 1 if pi % 2 == 0 else pi - 1
            far_pt = self.pts[far_pi]
            # 팔 자신의 두 끝점(근접점->원접점)으로 방향 계산 — P를 대신 쓰지 않음
            arm_dir = _angle_deg(far_pt[0] - near_pt[0], far_pt[1] - near_pt[1])
            diff = _signed_angle_diff(arm_dir, approach_dir)
            if abs(diff) <= max_arm_angle_deg:
                candidates.append({"line_idx": j, "diff": float(diff), "length": float(self.length[j])})

        pos = [c for c in candidates if c["diff"] > 0]
        neg = [c for c in candidates if c["diff"] < 0]
        if not pos or not neg:
            return {"found": False, "confidence": 0.0, "n_candidates": len(candidates)}

        best_score, best_pair = -1.0, None
        for p in pos:
            for ng in neg:
                len_ratio = min(p["length"], ng["length"]) / max(p["length"], ng["length"])
                angle_sym = 1 - abs(abs(p["diff"]) - abs(ng["diff"])) / max_arm_angle_deg
                score = 0.5 * len_ratio + 0.5 * max(angle_sym, 0.0)
                if score > best_score:
                    best_score, best_pair = score, (p, ng)

        return {
            "found": True,
            "confidence": float(best_score),
            "arm_pos": best_pair[0],
            "arm_neg": best_pair[1],
            "n_candidates": len(candidates),
        }

    def classify_line(self, line_idx, **kwargs):
        """선분 하나에 대해 양끝 화살촉 여부를 판정하고, 치수선/리더선 후보 여부를 함께 반환.

        판정 기준(P&ID 논문 Algorithm 1과 같은 논리):
          - 양끝 다 화살촉 있음 -> 치수선(dimension line) 후보
          - 한쪽만 있음         -> 리더선(leader line) 후보 (화살촉 쪽이 지시 대상)
          - 둘 다 없음          -> 화살표 무관 선(테두리/외곽선 등일 가능성)
        """
        start = self.detect_at_endpoint(line_idx, "start", **kwargs)
        end = self.detect_at_endpoint(line_idx, "end", **kwargs)
        if start["found"] and end["found"]:
            role = "dimension"
        elif start["found"] or end["found"]:
            role = "leader"
        else:
            role = "none"
        return {"start": start, "end": end, "role": role}
