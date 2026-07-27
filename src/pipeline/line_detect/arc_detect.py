# -*- coding: utf-8 -*-
"""원/호 추출 — 곡선을 선분으로 다루지 않고 원으로 되돌린다.

[문제]
LSD는 직선 검출기라 원과 호를 짧은 현(chord) 수십 개로 쪼갠다. 실측(도면 6장):
전체 선분의 0.5~19%가 사실은 원 조각이었고, 볼트홀이 많은 플랜지 도면에서는
5분의 1에 달했다.

그리고 이건 단순한 노이즈 청소가 아니다. 도면의 ø/R 치수는 '원'을 가리키는데
match_numbers는 텍스트를 '선분'에만 매칭한다. 후보 풀에 원이 없으니 ø275 같은
치수는 엉뚱한 현 조각에 붙는다 — 원을 복원해야 비로소 매칭이 가능해진다.

[방법]
짧은 선분을 "끝점이 이어지고 방향이 같은 쪽으로 계속 도는" 사슬로 엮은 뒤
원을 피팅한다. 직선은 회전이 없어 사슬이 만들어지지 않으므로 자동으로 걸러진다.
실측 피팅 잔차 0.20~0.92px.

[하지 말 것 — 직접 확인한 실패]
  - cv2.HoughCircles: 도면 1장당 259~1280개 검출. 전부 오검이라 못 쓴다.
  - 컨투어 원형도: 글자 О/О/0/6을 원으로 잡는다("КОНФИДЕНЦИАЛЬНО"의 О들이
    전부 검출됐다). 텍스트 제외가 선행되지 않으면 무의미하다.
  - 현 길이로 후보를 제한(max_chord): 반지름이 크면 현도 길다. 40px로 자르면
    큰 원을 통째로 놓친다(실측: 최대 반지름 182px -> 1041px). 회전 일관성과
    피팅 잔차가 이미 직선을 걸러주므로 길이 제한 자체가 불필요했다.

[G0과의 관계]
G0(엣지쌍 병합)은 호에 듣지 않는다. 안쪽/바깥쪽 링의 현 분할이 서로 달라
겹침 조건이 실패하기 때문이다(오버레이에서 원이 미짝으로 남는 것을 확인).
대신 여기서 같은 원리를 극좌표로 적용한다 — 중심이 같고 반지름이 획 두께만큼
차이나며 사이가 잉크면 한 획의 안팎 경계다.
"""
import numpy as np
from scipy.spatial import cKDTree

# 사슬 연결 조건 — 전부 기하 조건이라 도면 스케일과 무관하다.
CHAIN_GAP_PX = 4.0        # 끝점이 이만큼 가까우면 이어진 것으로 본다
TURN_MIN_DEG = 1.0        # 이보다 안 꺾이면 직선(사슬 안 만듦)
TURN_MAX_DEG = 40.0       # 이보다 꺾이면 모서리(사슬 끊음)
MIN_SEGMENTS = 4          # 최소 현 개수
MIN_TOTAL_TURN = 45.0     # 총 회전각. 너무 낮으면 등각뷰의 타원 조각이 원으로 오인됨
MAX_FIT_RMS = 1.2         # 원 피팅 잔차 상한(px)
MIN_INK_FRACTION = 0.70   # 피팅된 원호를 훑어 잉크 위에 있는 점의 최소 비율
FRAGMENT_OVERLAP_MAX = 0.25   # 각도가 이보다 겹치면 같은 링의 조각으로 보지 않는다


def _fit_circle(pts):
    """대수적 최소자승 원 피팅. 반환 (cx, cy, r, rms)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0] / 2, sol[1] / 2
    inner = sol[2] + cx ** 2 + cy ** 2
    if inner <= 0:
        return None
    r = float(np.sqrt(inner))
    res = np.abs(np.hypot(x - cx, y - cy) - r)
    return float(cx), float(cy), r, float(np.sqrt(np.mean(res ** 2)))


def _angular_span(cx, cy, pts):
    """점들이 원 위에서 차지하는 각도 구간. 반환 (start_deg, span_deg).

    가장 큰 빈 구간을 찾아 그 반대쪽을 실제 구간으로 본다(0/360 경계 처리).
    """
    a = np.sort(np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360.0)
    if len(a) < 2:
        return 0.0, 0.0
    gaps = np.diff(np.concatenate([a, [a[0] + 360.0]]))
    k = int(np.argmax(gaps))
    return float(a[(k + 1) % len(a)]), float(360.0 - gaps[k])


def _build_chains(lines):
    """끝점 인접 + 일관된 회전방향으로 선분을 엮어 사슬 후보를 만든다."""
    n = len(lines)
    ang = np.degrees(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0]))
    ends = np.vstack([lines[:, :2], lines[:, 2:]])
    adj = {i: [] for i in range(n)}
    for a, b in cKDTree(ends).query_pairs(CHAIN_GAP_PX):
        i, j = a % n, b % n
        if i == j:
            continue
        d = (ang[j] - ang[i] + 180.0) % 360.0 - 180.0
        if TURN_MIN_DEG <= abs(d) <= TURN_MAX_DEG:
            s = 1.0 if d > 0 else -1.0
            adj[i].append((j, s))
            adj[j].append((i, s))

    used, chains = set(), []
    # 이웃이 많은 선분부터 시작하면 긴 사슬이 먼저 잡혀 조각남이 줄어든다
    for s in sorted(range(n), key=lambda k: -len(adj[k])):
        if s in used or not adj[s]:
            continue
        for sense in (1.0, -1.0):
            if s in used:
                break
            chain, cur = [s], s
            while True:
                nxt = [j for j, sg in adj[cur]
                       if sg == sense and j not in chain and j not in used]
                if not nxt:
                    break
                # 가장 완만하게 이어지는 쪽으로 진행
                cur = min(nxt, key=lambda j: abs((ang[j] - ang[cur] + 180) % 360 - 180))
                chain.append(cur)
                if len(chain) > 400:
                    break
            if len(chain) >= MIN_SEGMENTS:
                total_turn = abs(sum((ang[chain[k + 1]] - ang[chain[k]] + 180) % 360 - 180
                                     for k in range(len(chain) - 1)))
                if total_turn >= MIN_TOTAL_TURN:
                    used.update(chain)
                    chains.append((chain, total_turn))
    return chains


def _ink_fraction(cx, cy, r, start, span, gray, n=48):
    """원호를 훑어 잉크 위에 놓인 표본의 비율. G0의 출력검증과 같은 원리."""
    from .edge_pair import estimate_ink_paper
    ink, paper = estimate_ink_paper(gray)
    cut = (ink + paper) / 2
    H, W = gray.shape
    k = max(8, int(n * min(span, 360.0) / 360.0))
    t = np.radians(start + np.linspace(0, span, k))
    xs = np.clip(np.rint(cx + r * np.cos(t)).astype(np.int32), 0, W - 1)
    ys = np.clip(np.rint(cy + r * np.sin(t)).astype(np.int32), 0, H - 1)
    # 반지름 방향으로 ±1px까지는 같은 획으로 인정(피팅 오차 흡수)
    best = gray[ys, xs].astype(np.int32)
    for dr in (-1.0, 1.0):
        xs2 = np.clip(np.rint(cx + (r + dr) * np.cos(t)).astype(np.int32), 0, W - 1)
        ys2 = np.clip(np.rint(cy + (r + dr) * np.sin(t)).astype(np.int32), 0, H - 1)
        best = np.minimum(best, gray[ys2, xs2])
    return float(np.mean(best < cut))


def _radius_sigma(arc):
    """짧은 호일수록 반지름 추정이 부정확하다. 그 불확실도를 근사한다.

    호가 짧으면 원의 곡률을 거의 못 보므로 반지름이 크게 흔들린다. 실측: 보어
    원(r≈340)이 60° 조각들로 쪼개졌을 때 조각별 반지름이 336.9~344.3px로
    7.4px 흩어졌다(개별 피팅 잔차는 0.3~0.5px에 불과한데도).
    기하적으로 sagitta 오차가 반지름 오차로 증폭되는 비율이 1/(1-cos(폭/2))이다.
    """
    half = np.radians(min(arc["span"], 180.0) / 2.0)
    amp = 1.0 / max(0.02, 1.0 - np.cos(half))
    return float(np.clip(max(arc["rms"], 0.3) * amp, 0.5, arc["r"] * 0.15))


def _angular_overlap(a, b):
    """두 호의 각도 구간이 겹치는 비율(짧은 쪽 기준, 0~1)."""
    if a["span"] >= 360.0 or b["span"] >= 360.0:
        return 1.0
    d = (b["start"] - a["start"]) % 360.0
    # a는 [0, sp_a], b는 [d, d+sp_b] — b를 한 바퀴 앞뒤로도 대보고 최대 겹침을 취한다
    best = 0.0
    for shift in (d - 360.0, d, d + 360.0):
        best = max(best, min(a["span"], shift + b["span"]) - max(0.0, shift))
    return float(np.clip(best, 0.0, min(a["span"], b["span"]))
                 / max(1e-6, min(a["span"], b["span"])))


def _merge_fragments(arcs, lines, max_iter=4):
    """같은 링이 여러 조각으로 검출된 것을 합치고 다시 피팅한다.

    [핵심 판별 — 각도 겹침]
    반지름만으로는 '같은 링의 조각'과 '안팎 두 링'을 못 가른다. 조각들의 반지름
    산포(7.4px)가 링 간격(2.6px)보다 크기 때문이다. 대신 각도를 본다:
      같은 링의 조각  -> 원 둘레의 다른 구간에 있다 (각도가 안 겹침)
      안팎 두 링      -> 같은 구간을 위아래로 덮는다 (각도가 겹침)
    그래서 '각도가 거의 안 겹치고 원이 호환되면' 같은 링으로 합친다.

    합칠 때마다 멤버 점 전체로 재피팅한다 — 각도폭이 넓어질수록 반지름 추정이
    정확해지므로, 반복하면 조각들이 눈덩이처럼 하나의 원으로 모인다.
    """
    arcs = [dict(a) for a in arcs]
    for _ in range(max_iter):
        arcs.sort(key=lambda d: -d["span"] * d["r"])
        merged_any = False
        out = []
        for a in arcs:
            hit = None
            for b in out:
                if _angular_overlap(a, b) > FRAGMENT_OVERLAP_MAX:
                    continue                      # 겹치면 안팎 두 링일 수 있다 — 건드리지 않음
                tol = 2.0 * (_radius_sigma(a) + _radius_sigma(b))
                if (abs(a["r"] - b["r"]) <= tol
                        and np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) <= tol):
                    hit = b
                    break
            if hit is None:
                out.append(a)
                continue
            hit["members"] = hit["members"] + a["members"]
            pts = np.vstack([lines[hit["members"]][:, :2], lines[hit["members"]][:, 2:]])
            fit = _fit_circle(pts)
            if fit is not None:
                hit["cx"], hit["cy"], hit["r"], hit["rms"] = fit
                hit["start"], hit["span"] = _angular_span(hit["cx"], hit["cy"], pts)
            merged_any = True
        arcs = out
        if not merged_any:
            break
    return arcs


def _union_span(s1, sp1, s2, sp2):
    """두 각도 구간의 합집합을 하나의 (시작, 길이)로 근사. 원을 넘으면 360으로."""
    if sp1 >= 360.0 or sp2 >= 360.0:
        return 0.0, 360.0
    # s1을 기준으로 s2를 상대 각도로 옮겨놓고 바깥 경계를 취한다
    d = (s2 - s1) % 360.0
    end = max(sp1, d + sp2)
    if end >= 360.0:
        return 0.0, 360.0
    return s1, end


def _merge_concentric(arcs, gray, stroke_gap):
    """한 획의 안팎 경계에 해당하는 동심 호 두 개를 중심 반지름 하나로 합친다.

    G0의 엣지쌍 병합과 같은 판정을 극좌표에서 한다 — 중심이 거의 같고 반지름
    차이가 획 두께쯤이며, 두 반지름 사이가 잉크면 한 획이다.
    """
    if stroke_gap is None or not np.isfinite(stroke_gap):
        return arcs
    lo, hi = stroke_gap * 0.5, stroke_gap * 2.0
    arcs = sorted(arcs, key=lambda d: d["r"])
    used, out = set(), []
    for i, a in enumerate(arcs):
        if i in used:
            continue
        partner = None
        for j in range(i + 1, len(arcs)):
            if j in used:
                continue
            b = arcs[j]
            dr = b["r"] - a["r"]
            if dr > hi:
                break
            if dr < lo:
                continue
            if np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) > max(1.5, a["r"] * 0.05):
                continue
            mid_r = (a["r"] + b["r"]) / 2
            start = a["start"] if a["span"] >= b["span"] else b["start"]
            span = max(a["span"], b["span"])
            if _ink_fraction(a["cx"], a["cy"], mid_r, start, span, gray) >= MIN_INK_FRACTION:
                partner = (j, mid_r, start, span)
                break
        if partner is None:
            out.append(a)
            continue
        j, mid_r, start, span = partner
        b = arcs[j]
        used.add(j)
        out.append({"cx": (a["cx"] + b["cx"]) / 2, "cy": (a["cy"] + b["cy"]) / 2,
                    "r": mid_r, "start": start, "span": span,
                    "rms": max(a["rms"], b["rms"]),
                    "thickness": float(b["r"] - a["r"]),
                    "members": a["members"] + b["members"], "paired": True})
    return out


def extract_arcs(lines, gray, stroke_gap=None, min_radius=None):
    """선분 배열에서 원/호를 추출하고, 흡수된 선분을 제거한 나머지를 함께 반환.

    lines       : (N,4) — G0(엣지쌍 병합)을 먼저 돌린 결과를 넣는 것을 권장
    gray        : 원본 그레이스케일 (잉크검사용)
    stroke_gap  : G0가 자가보정한 대표 획 간격. 주면 동심 호 병합에 쓴다.
    min_radius  : 이보다 작은 원은 버린다. None이면 획 두께의 3배.

    반환 (remaining_lines, arcs, stats)
      arcs 원소: {cx, cy, r, start, span, rms, thickness, paired, members}
    """
    lines = np.asarray(lines, dtype=np.float64)
    stats = {"n_in": int(len(lines)), "n_chains": 0, "n_arcs": 0,
             "n_absorbed": 0, "n_rejected_by_ink": 0}
    if len(lines) < MIN_SEGMENTS:
        return lines, [], stats

    if min_radius is None:
        min_radius = 3.0 * stroke_gap if stroke_gap and np.isfinite(stroke_gap) else 6.0
    max_radius = 2.0 * max(gray.shape)

    chains = _build_chains(lines)
    stats["n_chains"] = len(chains)

    cands = []
    for chain, total_turn in chains:
        pts = np.vstack([lines[chain][:, :2], lines[chain][:, 2:]])
        fit = _fit_circle(pts)
        if fit is None:
            continue
        cx, cy, r, rms = fit
        if rms > MAX_FIT_RMS or not (min_radius <= r <= max_radius):
            continue
        start, span = _angular_span(cx, cy, pts)
        cands.append({"cx": cx, "cy": cy, "r": r, "start": start, "span": span,
                      "rms": rms, "thickness": float("nan"), "paired": False,
                      "members": list(chain)})

    cands = _merge_fragments(cands, lines)
    cands = _merge_concentric(cands, gray, stroke_gap)

    # 출력 검증 — 잉크 위에 없는 원은 버린다. G0에서 오병합을 잡아낸 것과 같은
    # 안전장치로, 등각뷰의 타원 조각에서 나오는 유령 원이 여기서 걸러진다.
    arcs, rejected = [], 0
    for a in cands:
        if _ink_fraction(a["cx"], a["cy"], a["r"], a["start"], a["span"], gray) >= MIN_INK_FRACTION:
            arcs.append(a)
        else:
            rejected += 1

    absorbed = set()
    for a in arcs:
        absorbed.update(a["members"])
    keep = np.ones(len(lines), dtype=bool)
    if absorbed:
        keep[np.fromiter(absorbed, dtype=np.int64)] = False

    stats.update({"n_arcs": len(arcs), "n_absorbed": int(len(absorbed)),
                  "n_rejected_by_ink": rejected,
                  "absorbed_pct": float(len(absorbed) / max(1, len(lines)) * 100),
                  "n_full_circles": int(sum(1 for a in arcs if a["span"] >= 300))})
    return lines[keep], arcs, stats


def draw_arcs(vis, arcs, color=(0, 0, 255), thickness=2, mark_center=True):
    """검증용 오버레이."""
    import cv2
    for a in arcs:
        cv2.ellipse(vis, (int(round(a["cx"])), int(round(a["cy"]))),
                    (int(round(a["r"])), int(round(a["r"]))), 0,
                    a["start"], a["start"] + a["span"], color, thickness)
        if mark_center:
            cv2.drawMarker(vis, (int(round(a["cx"])), int(round(a["cy"]))),
                           (255, 0, 0), cv2.MARKER_CROSS, 12, 2)
    return vis
