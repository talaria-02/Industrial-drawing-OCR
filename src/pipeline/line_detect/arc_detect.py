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
CHAIN_GAP_PX = 6.0        # 끝점이 이만큼 가까우면 이어진 것으로 본다
TURN_MIN_DEG = 1.0        # 이보다 안 꺾이면 직선(사슬 안 만듦)
TURN_MAX_DEG = 40.0       # 이보다 꺾이면 모서리(사슬 끊음)
# 사슬 수용 기준을 느슨하게 잡는다. 처음에는 (현 4개, 총회전 45도)로 조였는데
# 작은 원을 대량으로 놓쳤다 — 샤프트 도면에서 2개만 검출됐고, 같은 도면에서
# EDCircles는 유효한 원 25개를 찾았다. 조건을 (2개, 25도)로 풀면 29개가 나와
# EDCircles와 맞고, 8장 합계로는 135개 -> 440개가 된다.
#
# 느슨하게 해도 되는 이유는 아래 출력검증(_ink_fraction) 때문이다. 걸러야 할
# 것은 여기서가 아니라 거기서 걸린다(실측: 거부 수가 1~22개에서 24~96개로 늘고,
# 살아남은 원의 잉크 적중률 중앙값은 88~100%로 유지됨).
# 대가는 속도다 — 사슬이 많아져 도면당 최대 7.7초에서 19.3초로 늘었다.
MIN_SEGMENTS = 2          # 최소 현 개수
MIN_TOTAL_TURN = 25.0     # 총 회전각
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

    # members만으로는 부족하다 — 체인 조건(간격 6px, 회전 일관성)을 만족하지
    # 못한 현들은 체인에 못 들어갔을 뿐 여전히 그 원 위에 놓여 있다. 실측:
    # 도면 8장에서 채택된 호 84개 위에 선분 71개가 그대로 남아 있었다(C-074는
    # 호 5개에 잔존 16개). 화면에서는 원과 조각이 겹쳐 보이고, 매칭 후보 풀도
    # 그만큼 오염된다. 그래서 기하 조건으로 한 번 더 훑는다.
    #
    # 훑기 전에 호의 각도구간을 먼저 늘린다. 남아 있던 조각을 조건별로 세어보니
    # 60%(216/358)가 '원 위에 있고 기울기도 맞는데 호의 구간 밖'이었다. 구간이
    # 짧게 잡힌 것이 주원인이었으므로, 그걸 지우면 그 자리의 잉크가 사라진다.
    tol = max(2.0, 1.25 * float(stroke_gap)
              if stroke_gap and np.isfinite(stroke_gap) else 2.0)
    n_extended = 0
    for a in arcs:
        if _extend_span_with_fragments(a, lines, gray, tol):
            n_extended += 1

    n_swept = 0
    for a in arcs:
        on = lines_on_arc(lines, a, tol_px=tol)
        for i in np.where(on)[0]:
            if i not in absorbed:
                absorbed.add(int(i))
                n_swept += 1

    keep = np.ones(len(lines), dtype=bool)
    if absorbed:
        keep[np.fromiter(absorbed, dtype=np.int64)] = False

    stats.update({"n_swept_on_arc": n_swept, "n_span_extended": n_extended,
                  "n_arcs": len(arcs), "n_absorbed": int(len(absorbed)),
                  "n_rejected_by_ink": rejected,
                  "absorbed_pct": float(len(absorbed) / max(1, len(lines)) * 100),
                  "n_full_circles": int(sum(1 for a in arcs if a["span"] >= 300))})
    return lines[keep], arcs, stats


TANGENT_TOL_DEG = 20.0


def _extend_span_with_fragments(arc, lines, gray, tol_px):
    """호의 각도구간을 같은 원 위의 조각까지 넓힌다. 넓혔으면 True.

    [왜 필요한가]
    구간을 짧게 잡으면 나머지 조각이 '구간 밖'이라 흡수되지 않고 남는다. 실측:
    남은 조각 358개 중 216개가 원 위에 있고 기울기도 맞는데 구간 밖이었다.
    그걸 그냥 지우면 호가 안 그리는 자리의 잉크가 사라지므로, 지우기 전에
    호를 그 자리까지 늘리는 것이 맞다.

    [늘려도 되는지는 잉크가 정한다]
    구간을 함부로 늘리면 아무것도 그려지지 않은 자리에 호가 생긴다. 그래서
    늘릴 구간마다 _ink_fraction으로 잉크를 확인하고, 통과한 것만 반영한다 —
    호 채택에 쓰던 것과 같은 안전장치다.
    """
    if len(lines) == 0 or arc["span"] >= 354.0:
        return False
    on = lines_on_arc(lines, arc, tol_px=tol_px, angle_margin_deg=360.0)
    if not on.any():
        return False

    covered = [(0.0, arc["span"])]          # arc["start"] 기준 상대각
    for x1, y1, x2, y2 in lines[on]:
        a1 = (np.degrees(np.arctan2(y1 - arc["cy"], x1 - arc["cx"]))
              - arc["start"]) % 360.0
        a2 = (np.degrees(np.arctan2(y2 - arc["cy"], x2 - arc["cx"]))
              - arc["start"]) % 360.0
        lo, hi = min(a1, a2), max(a1, a2)
        if hi - lo > 180.0:                 # 0/360 경계를 넘은 조각
            lo, hi = hi, lo + 360.0
        covered.append((lo, hi))

    covered.sort()
    span = arc["span"]
    changed = False
    for lo, hi in covered:
        if hi <= span:
            continue
        gap_lo = max(lo, span)
        frac = _ink_fraction(arc["cx"], arc["cy"], arc["r"],
                             arc["start"] + gap_lo, max(hi - gap_lo, 1.0), gray)
        if frac < MIN_INK_FRACTION:
            continue
        span = min(hi, 360.0)
        changed = True
    if changed:
        arc["span"] = float(span)
        arc["closed"] = bool(span >= 300.0)
    return changed


def lines_on_arc(lines, arc, tol_px=2.0, n_samples=5, angle_margin_deg=6.0,
                 tangent_tol_deg=TANGENT_TOL_DEG):
    """호 위에 놓인 선분 마스크. 원이 덮은 자리의 조각을 지우는 데 쓴다.

    조건 두 개다.
      ① 가까이 있다   — 선분의 두 끝점이 원에서 tol_px 이내
      ② 기울기가 같다 — 선분 방향이 그 자리 접선 방향과 tangent_tol_deg 이내

    [왜 끝점만 보는가]
    현은 중앙이 원 안쪽으로 처진다(r=100, 30도 현이면 3.4px). 모든 샘플을 같은
    허용치로 재면 이 처짐이 '링 두께만큼 떨어진 오프셋'과 섞여, 처짐을 담으려
    허용치를 키우면 별개의 동심원까지 먹는다. 처짐은 끝점에서 0이므로 끝점만
    재면 두 가지가 분리된다. 중간이 얼마나 처졌는지는 ②가 대신 본다.

    [tol_px를 획두께에서 잡는 이유]
    링은 두 경계로 검출되므로 반대편 경계에서 나온 조각은 링 두께만큼(실측
    2.67px) 원에서 떨어져 앉는다. 2px 고정으로는 그게 하나도 안 걸렸다.
    반대로 획두께보다 확실히 먼 동심원은 별개의 형상이라 남아야 한다.

    [기울기 조건이 길이 상한을 대신한다]
    구멍을 가로지르는 지름선이나 긴 현은 끝점이 원 위에 있어도 방향이 접선과
    크게 다르다. 예전에는 '60도 넘는 현은 제외' 같은 상한을 따로 뒀는데,
    기울기를 보면 그 상한이 저절로 나온다 — 현이 벌어질수록 접선과의 각도차가
    그 절반씩 커지므로, 기본값 15도는 30도보다 벌어진 현을 남긴다는 뜻이다.
    조건을 하나 줄이면서 지름선도 그대로 지킨다.

    [접선은 원리상 구별되지 않는다]
    원에 접하는 짧은 직선은 위 두 조건을 똑같이 만족한다. 링의 조각인지
    의도적으로 그린 접선인지는 이 정보만으로 가를 수 없다. 잘못 지워도 그
    자리의 잉크는 호가 대신 표현하고, 사람이 그린 원에는 Ctrl+Z가 있다.

    arc: {cx, cy, r, start, span} 또는 {center, r, start_deg, span_deg}
    """
    L = np.asarray(lines, dtype=np.float64)
    if len(L) == 0:
        return np.zeros(0, dtype=bool)
    cx = arc.get("cx", arc.get("center", [0, 0])[0])
    cy = arc.get("cy", arc.get("center", [0, 0])[1])
    r = float(arc["r"])
    start = float(arc.get("start", arc.get("start_deg", 0.0)))
    span = float(arc.get("span", arc.get("span_deg", 360.0)))
    if r <= 0:
        return np.zeros(len(L), dtype=bool)

    ts = np.linspace(0.0, 1.0, n_samples)[None, :, None]
    pts = L[:, None, :2] + (L[:, None, 2:] - L[:, None, :2]) * ts
    dx, dy = pts[:, :, 0] - cx, pts[:, :, 1] - cy
    rad = np.hypot(dx, dy)

    tol = max(float(tol_px), r * 0.01)
    on = (np.abs(rad[:, 0] - r) <= tol) & (np.abs(rad[:, -1] - r) <= tol)

    # 선분 방향 vs 접선 방향. 반드시 '끝점에서' 재야 한다 — 현의 방향은 중점
    # 에서의 접선 방향과 항상 정확히 같아서(중심에서 내린 수선이 현을 이등분
    # 하므로) 중점에서 재면 어떤 현이든 통과해 조건이 무의미해진다. 끝점에서는
    # 벌어진 각의 절반만큼 차이가 나므로 이것이 곧 판별이 된다.
    seg = np.degrees(np.arctan2(L[:, 3] - L[:, 1], L[:, 2] - L[:, 0])) % 180.0
    for k in (0, -1):
        tangent = (np.degrees(np.arctan2(dy[:, k], dx[:, k])) + 90.0) % 180.0
        d = np.abs(seg - tangent)
        on &= np.minimum(d, 180.0 - d) <= tangent_tol_deg

    if span < 360.0 - angle_margin_deg:
        ang = np.degrees(np.arctan2(dy, dx)) % 360.0
        rel = (ang - start) % 360.0
        on &= (rel <= span + angle_margin_deg).all(axis=1)
    return on


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
