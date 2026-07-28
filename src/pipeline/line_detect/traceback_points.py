# -*- coding: utf-8 -*-
"""치수 -> 측정점 역추적. "이 치수가 실제로 어느 두 모서리를 가리키는가".

[왜 필요한가]
match_numbers는 텍스트를 '치수선'에 붙인다. 그런데 치수선은 주석이라 제품에
존재하지 않는다. 사진에서 재려면 치수선이 아니라 그것이 가리키는 '외형선'을
알아야 한다. 도면의 구조가 그 답을 갖고 있다:

     ┌──────────────┐   <- 외형선 (재야 할 것)
     ╎              ╎   <- 치수보조선 (외형선에서 뻗어나옴)
     ╎<---- 65 ---->╎   <- 치수선 (매칭이 찾아준 것)

치수선 끝점에서 수직으로 뻗은 선을 따라가면 외형선에 닿는다. 그 두 점이
'사진에서 클릭해야 할 자리'다.

[스케일을 글자 높이로 잡는 이유]
arrowhead.py가 절대 픽셀값(MIN_ARM_LENGTH=6.0 등)을 쓰고 있는데, 실측해보니
도면별 OCR 글자 높이 중앙값이 13~35px로 2.7배 차이난다. 절대값은 반드시 깨진다.
글자 높이는 도면마다 OCR이 공짜로 주고 축척을 따라가므로 좋은 기준자다.
G0에서 획 간격을 자가보정한 것과 같은 접근이다.
"""
import numpy as np
from scipy.spatial import cKDTree

# 아래 비율은 기존 절대값을 '글자 높이 27px 도면'에서 튜닝된 것으로 보고
# 환산한 값이다(그 도면들의 글자 높이 중앙값이 26.5px이었다).
REF_TEXT_H = 27.0
RATIO_MIN_ARM = 6.0 / REF_TEXT_H
RATIO_MAX_ARM = 45.0 / REF_TEXT_H
RATIO_SEARCH_RADIUS = 25.0 / REF_TEXT_H
# 치수보조선으로 인정할 각도(치수선과 수직에서 이만큼 벗어나도 허용)
PERP_TOL_DEG = 20.0
# 측정점이 외형선 위에 있다고 볼 거리 — 글자 높이 대비
RATIO_SNAP = 0.5


def text_scale(polys):
    """OCR 폴리곤들에서 글자 높이 중앙값(px). 도면마다의 '기준자'."""
    hs = []
    for p in polys:
        q = np.asarray(p, float)
        if len(q) < 4:
            continue
        hs.append(min(np.linalg.norm(q[1] - q[2]), np.linalg.norm(q[0] - q[3])))
    return float(np.median(hs)) if hs else REF_TEXT_H


def scaled_params(text_h):
    """글자 높이로부터 화살촉/역추적 임계값을 만든다."""
    h = max(float(text_h), 4.0)
    return {
        "min_arm_len": h * RATIO_MIN_ARM,
        "max_arm_len": h * RATIO_MAX_ARM,
        "search_radius": h * RATIO_SEARCH_RADIUS,
        "snap_px": h * RATIO_SNAP,
        "text_h": h,
    }


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _seg_dist_to_ray(p1, p2, origin, direction, lateral_tol):
    """반직선(origin, direction) 위에서 선분까지의 전진거리. 벗어나면 None.

    LSD 선분은 서로 닿아 있지 않다. 끊긴 채로 같은 방향에 늘어서 있으므로,
    '접촉'이 아니라 '진행 방향 위에 있는가'로 판단해야 한다.
    """
    n = np.array([-direction[1], direction[0]])
    best = None
    for q in (p1, p2, (p1 + p2) / 2):
        d = q - origin
        fwd = float(d @ direction)
        lat = abs(float(d @ n))
        if fwd < -lateral_tol or lat > lateral_tol:
            continue
        if best is None or fwd < best:
            best = fwd
    return best


def trace_measure_points(dim_line, lines, params, exclude_idx=()):
    """치수선 하나에서 측정점 두 개를 역추적한다.

    [닿아 있는지로 판정할 수 없다]
    LSD가 뱉는 선분은 조각조각 끊겨 있어서, 치수선 끝점에 치수보조선이 딱
    붙어 있는 경우가 오히려 드물다. 그래서 '끝점이 가까운가'가 아니라
    '치수선에 수직인 방향으로 나아가면 그 위에 있는가'로 찾는다. 조금 짧고
    떨어져 있어도 같은 진행선 위에 있으면 후보로 받는다.

    [경로를 남긴다]
    어느 선분을 밟고 갔는지 path에 기록한다. 자동 결과가 논리적으로 말이
    되는지는 사람이 그 경로를 보고 판단해야 하고, 곧바로 이은 직선만 보여주면
    검수가 불가능하다.

    반환 dict:
      points   : [(x,y), (x,y)]  외형선 위의 측정점
      path     : [[line_idx...], [line_idx...]]  양쪽에서 밟은 선분들
      span_px  : 두 점 사이 거리(치수선 축 방향)
      quality  : 'traced' | 'partial' | 'fallback'
    """
    d = np.asarray(dim_line, float)
    a, b = d[:2], d[2:]
    axis = _unit(b - a)
    perp = np.array([-axis[1], axis[0]])
    lines = np.asarray(lines, float)
    if len(lines) == 0:
        return {"points": [tuple(a), tuple(b)], "via": [None, None], "path": [[], []],
                "span_px": float(np.linalg.norm(b - a)), "quality": "fallback"}

    h = params["text_h"]
    lat_tol = max(2.5, h * 0.25)          # 진행선에서 벗어나도 되는 폭
    reach = max(params["search_radius"], h * 4.0)   # 한 번에 내다볼 거리
    max_steps = 4                          # 끊긴 조각을 몇 번까지 이어 갈지

    out, via, paths = [], [], []
    for tip in (a, b):
        # 치수선 바깥쪽(부품 쪽)으로 뻗는 두 방향을 모두 시도한다
        best = None
        for sgn in (1.0, -1.0):
            dirv = perp * sgn
            cur, walked, total = tip.copy(), [], 0.0
            for _ in range(max_steps):
                cand = None
                for li in range(len(lines)):
                    if li in exclude_idx or li in walked:
                        continue
                    p1, p2 = lines[li, :2], lines[li, 2:]
                    u = _unit(p2 - p1)
                    if abs(float(u @ dirv)) < np.cos(np.radians(PERP_TOL_DEG)):
                        continue          # 진행 방향과 나란하지 않으면 보조선이 아니다
                    fwd = _seg_dist_to_ray(p1, p2, cur, dirv, lat_tol)
                    if fwd is None or fwd > reach:
                        continue
                    if cand is None or fwd < cand[0]:
                        cand = (fwd, li, p1, p2)
                if cand is None:
                    break
                _, li, p1, p2 = cand
                far = p2 if float((p2 - cur) @ dirv) > float((p1 - cur) @ dirv) else p1
                total += float((far - cur) @ dirv)
                cur = far
                walked.append(li)
            if walked and (best is None or total > best[0]):
                best = (total, cur, walked)
        if best is None:
            out.append(tuple(tip)); via.append(None); paths.append([])
        else:
            out.append(tuple(best[1])); via.append(best[2][0]); paths.append(best[2])

    got = sum(1 for v in via if v is not None)
    quality = "traced" if got == 2 else ("partial" if got == 1 else "fallback")
    span = abs(float((np.array(out[1]) - np.array(out[0])) @ axis))
    return {"points": out, "via": via, "path": paths,
            "span_px": span, "quality": quality}


def build_measure_points(doc_links, texts, lines, text_polys):
    """검수 문서의 링크들을 훑어 각 치수의 측정점을 채운다.

    doc_links : review 문서의 links 배열 (line_ids / arc_ids 보유)
    반환: {text_id: measure dict}
    """
    params = scaled_params(text_scale(text_polys))
    lines = np.asarray(lines, float)
    by_id = {t["id"]: t for t in texts}
    idx_of = {}
    out = {}
    for li, l in enumerate(lines):
        idx_of[li] = li

    for link in doc_links:
        tid = link["text_id"]
        if tid not in by_id:
            continue
        lids = link.get("line_ids", [])
        if not lids:
            continue
        # 링크가 여러 선분이면 가장 긴 것을 치수선 본체로 본다
        best_i, best_len = None, -1.0
        for lid in lids:
            try:
                i = int(str(lid).lstrip("l")) - 1
            except ValueError:
                continue
            if not (0 <= i < len(lines)):
                continue
            L = float(np.hypot(lines[i, 2] - lines[i, 0], lines[i, 3] - lines[i, 1]))
            if L > best_len:
                best_i, best_len = i, L
        if best_i is None:
            continue
        r = trace_measure_points(lines[best_i], lines, params, exclude_idx={best_i})
        r["params"] = {k: round(v, 2) for k, v in params.items()}
        out[tid] = r
    return out
