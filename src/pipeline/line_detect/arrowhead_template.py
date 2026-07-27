# -*- coding: utf-8 -*-
"""화살촉 검출 — 템플릿 매칭(정규화 상호상관) 방식. 현재 채택된 방법.

[왜 이 방식인가]
CNN이 필요한 이유는 보통 "대상의 외형 변동이 커서"인데, CAD 화살촉은 같은
렌더러가 그린 동일 도형이라 변동성이 거의 없다. 남는 변동은 회전과 크기뿐이고,
회전은 선분 방향으로 이미 알고 있어서 탐색이 불필요하다(360번 -> 1번).
그래서 학습 없이 템플릿 매칭만으로 충분하다.

[앞선 두 방식과의 비교 — 실측(데모 도면, 매칭선 15개 x 양끝)]
  arrowhead.py        (선분 끝점 V자 기하): 양끝확인 3.5/15. LSD가 화살촉을
                       선분으로 못 잡으면 구조적으로 놓침(`76`이 그 예).
  arrowhead_width.py  (폭 프로파일):        양끝확인 9/15. 짧은 선/해칭 근처에서
                       기준 두께가 오염됨(`49`, `ø80`에서 18~22px로 측정).
  이 파일 (템플릿):                          양끝확인 13/15. 파라미터도 훨씬 적음.

[템플릿을 합성으로 만드는 이유]
도면에서 오려낸 실물 템플릿이 절대 점수는 높지만(0.86 vs 0.75), 그러면 모듈이
특정 도면에 묶이고 배포도 어렵다. 합성 삼각형은 점수가 낮아도 정답/오답 간격이
오히려 더 넓었다(+0.220 vs +0.172, 전체배율 기준). ISO 표준 화살촉 비율
(길이:폭 = 약 3:1)로 그리면 충분하다.

[배율 자동 보정이 필요한 이유]
같은 도면 안에서도 화살촉 크기가 다를 수 있다(예: (2:1) 확대 상세뷰).
그런데 배율 탐색 범위를 넓히면 오답도 같이 점수가 오른다 — 실측으로 오답들이
전부 최소 배율에서 이겼다(작은 템플릿일수록 우연히 맞기 쉬움).
그래서 2단계로 한다:
  1차) 넓은 배율로 전체 끝점 훑어서, 고득점 후보들이 어느 배율에 몰리는지 봄
  2차) 그 우세 배율 ± 한 칸으로 범위를 좁혀 재채점
이러면 도면별 축척을 자동으로 맞추면서 오답 유입은 억제된다.
"""
import cv2
import numpy as np

# 실측 기반 기본값(데모 도면 화살촉 길이 ~15px, 정답/오답 간격이 최대였던 설정)
DEFAULT_ARROW_LENGTH = 15      # 합성 템플릿의 기준 길이(px)
DEFAULT_RATIO = 3.0            # 길이:폭 비 (ISO 표준 화살촉 형태)
SEARCH_RADIUS = 30             # 끝점 주변 이 반경(px) 안에서 템플릿을 찾음
WIDE_SCALES = (0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0)
SCORE_THRESHOLD = 0.65         # 이 값 이상이면 화살촉으로 판정
CALIB_MIN_SAMPLES = 4          # 배율 보정에 쓸 최소 고득점 표본 수


def make_synthetic_template(length=DEFAULT_ARROW_LENGTH, ratio=DEFAULT_RATIO, pad=3):
    """위(-y 방향)를 향한, 속이 꽉 찬 삼각형 화살촉 템플릿 생성.
    배경 흰색(255), 화살촉 검정(0). pad는 회전 시 잘림 방지용 여백."""
    w = int(round(length / ratio * 2))
    h, wd = length + 2 * pad, w + 2 * pad
    img = np.full((h, wd), 255, np.uint8)
    tip = (wd // 2, pad)
    bl = (pad, pad + length)
    br = (wd - pad - 1, pad + length)
    cv2.fillPoly(img, [np.array([tip, bl, br])], 0)
    return img


def _rotate(tpl, deg):
    """템플릿을 정사각 캔버스에 넣고 회전(모서리 잘림 방지, 배경 흰색 유지)."""
    h, w = tpl.shape
    s = int(np.hypot(h, w)) + 2
    canvas = np.full((s, s), 255, np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = tpl
    M = cv2.getRotationMatrix2D((s / 2, s / 2), deg, 1.0)
    return cv2.warpAffine(canvas, M, (s, s), borderValue=255)


def score_endpoint(img_gray, tpl, P, approach_deg, scales=WIDE_SCALES,
                    search_radius=SEARCH_RADIUS):
    """끝점 P에서 화살촉 유사도를 잰다.

    approach_deg: 선이 P로 다가와 그대로 연장되는 방향(= 화살촉 팁이 있어야 할 방향).
                  끝점 P, 반대쪽 끝점 Q일 때 angle(P - Q).
                  (P->Q 방향으로 잡으면 180도 어긋나서 전부 놓친다)
    반환: (최고점수, 그 점수를 낸 배율)
    """
    # 템플릿이 위(-90도)를 향하므로, 목표 방향과의 차이만큼 회전시킨다
    rot_deg = -(approach_deg + 90)
    best_score, best_scale = 0.0, None

    x0 = max(0, int(P[0] - search_radius))
    y0 = max(0, int(P[1] - search_radius))
    roi = img_gray[y0:int(P[1] + search_radius), x0:int(P[0] + search_radius)]

    for s in scales:
        t = cv2.resize(tpl, None, fx=s, fy=s,
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        t = _rotate(t, rot_deg)
        if roi.shape[0] < t.shape[0] or roi.shape[1] < t.shape[1]:
            continue
        v = float(cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED).max())
        if v > best_score:
            best_score, best_scale = v, s
    return best_score, best_scale


def approach_direction(near_pt, far_pt):
    """near_pt 쪽 끝점에서 화살촉 팁이 있어야 할 방향(도)."""
    return float(np.degrees(np.arctan2(near_pt[1] - far_pt[1], near_pt[0] - far_pt[0])))


def _narrow_scales(dominant, scales=WIDE_SCALES):
    """우세 배율 기준 ±1칸으로 좁힌 배율 목록."""
    if dominant not in scales:
        return scales
    i = scales.index(dominant)
    return tuple(scales[max(0, i - 1):i + 2])


def detect_on_lines(img_gray, lines, tpl=None, scales=WIDE_SCALES,
                     threshold=SCORE_THRESHOLD, search_radius=SEARCH_RADIUS,
                     calibrate_scale=True):
    """선분 여러 개의 양 끝점에 대해 화살촉을 검출.

    calibrate_scale=True면 2단계로 동작한다(위 docstring의 '배율 자동 보정').

    lines: (N,4) 배열 [x1,y1,x2,y2]
    반환: results 리스트. 각 원소:
      {"line_idx", "start": {"score","scale","found"}, "end": {...},
       "role": "dimension"|"leader"|"none"}
      + 두 번째 반환값으로 진단 정보 dict(우세 배율, 사용한 배율 목록)
    """
    if tpl is None:
        tpl = make_synthetic_template()
    lines = np.asarray(lines, dtype=np.float64)

    def pass_over(use_scales):
        out = []
        for i, L in enumerate(lines):
            P, Q = L[:2], L[2:]
            s_sc, s_scale = score_endpoint(img_gray, tpl, P, approach_direction(P, Q),
                                            use_scales, search_radius)
            e_sc, e_scale = score_endpoint(img_gray, tpl, Q, approach_direction(Q, P),
                                            use_scales, search_radius)
            out.append((i, (s_sc, s_scale), (e_sc, e_scale)))
        return out

    used_scales = scales
    dominant = None
    if calibrate_scale and len(lines) > 0:
        first = pass_over(scales)
        hits = [sc for _, (s, sc), (e, ec) in first if s >= threshold and sc is not None]
        hits += [ec for _, (s, sc), (e, ec) in first if e >= threshold and ec is not None]
        if len(hits) >= CALIB_MIN_SAMPLES:
            vals, counts = np.unique(hits, return_counts=True)
            dominant = float(vals[int(np.argmax(counts))])
            used_scales = _narrow_scales(dominant, scales)
            raw = pass_over(used_scales)
        else:
            raw = first  # 표본이 적으면 보정 없이 1차 결과 사용
    else:
        raw = pass_over(scales)

    results = []
    for i, (s_sc, s_scale), (e_sc, e_scale) in raw:
        s_found, e_found = s_sc >= threshold, e_sc >= threshold
        role = "dimension" if (s_found and e_found) else ("leader" if (s_found or e_found) else "none")
        results.append({
            "line_idx": i,
            "start": {"score": s_sc, "scale": s_scale, "found": s_found},
            "end": {"score": e_sc, "scale": e_scale, "found": e_found},
            "role": role,
        })
    return results, {"dominant_scale": dominant, "used_scales": used_scales}
