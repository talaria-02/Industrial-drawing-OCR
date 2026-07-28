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


def trace_measure_points(dim_line, lines, params, exclude_idx=()):
    """치수선 하나에서 측정점 두 개를 역추적한다.

    dim_line : [x1,y1,x2,y2]  매칭이 찾아준 치수선
    lines    : (N,4) 선분 DB 전체
    반환 dict 또는 None
      points   : [(x,y), (x,y)]  외형선 위의 측정점
      via      : 사용한 치수보조선 인덱스 (없으면 None)
      span_px  : 두 점 사이 거리(도면 px)
      quality  : 'traced'(양쪽 다 추적) | 'partial' | 'fallback'(치수선 끝점 그대로)

    양쪽 다 못 찾으면 치수선 끝점을 그대로 쓴다(fallback). 치수선 길이는 원래
    측정 거리와 같으므로 값 자체는 맞고, 다만 '사진의 어디'인지가 부정확해진다.
    """
    d = np.asarray(dim_line, float)
    a, b = d[:2], d[2:]
    axis = _unit(b - a)
    lines = np.asarray(lines, float)
    if len(lines) == 0:
        return {"points": [tuple(a), tuple(b)], "via": [None, None],
                "span_px": float(np.linalg.norm(b - a)), "quality": "fallback"}

    ends = np.vstack([lines[:, :2], lines[:, 2:]])
    tree = cKDTree(ends)
    n = len(lines)
    R = params["search_radius"]

    out, via = [], []
    for tip in (a, b):
        best = None
        for k in tree.query_ball_point(tip, R):
            li = k % n
            if li in exclude_idx:
                continue
            p1, p2 = lines[li, :2], lines[li, 2:]
            u = _unit(p2 - p1)
            # 치수보조선은 치수선과 수직이다
            cosang = abs(float(u @ axis))
            if cosang > np.cos(np.radians(90 - PERP_TOL_DEG)):
                continue
            near, far = (p1, p2) if np.linalg.norm(p1 - tip) <= np.linalg.norm(p2 - tip) else (p2, p1)
            dist = float(np.linalg.norm(near - tip))
            if best is None or dist < best[0]:
                best = (dist, far, li)
        if best is None:
            out.append(tuple(tip)); via.append(None)
        else:
            out.append(tuple(best[1])); via.append(int(best[2]))

    got = sum(v is not None for v in via)
    quality = "traced" if got == 2 else ("partial" if got == 1 else "fallback")
    # 측정 거리는 치수선 축 방향 성분으로 낸다. 치수보조선 길이가 서로 달라도
    # (외형선이 계단이면 흔하다) 축 방향 거리는 원래 치수와 일치한다.
    span = abs(float((np.array(out[1]) - np.array(out[0])) @ axis))
    return {"points": out, "via": via, "span_px": span, "quality": quality}


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
