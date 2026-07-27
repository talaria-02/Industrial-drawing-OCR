# -*- coding: utf-8 -*-
"""화살촉 검출 — 픽셀 폭(stroke width) 프로파일 기반.

[배경]
Intwala et al.(2016)의 Black-Hat/White-Hat 모폴로지 방식을 먼저 시도했으나,
우리 도면에서는 화살촉이 다른 선(치수선 자체, 교차하는 연장선)과 맞닿아
있어서 "고립된 작은 점"이라는 전제가 깨져 깔끔하게 분리되지 않았다(실측
확인: kernel 크기 무관하게 화살촉 삼각형과 교차선이 뭉쳐서 하나의 컨투어로
잡히거나, 삼각형 자체가 아예 안 잡힘).

대신 실측으로 확인한 더 단순하고 강건한 사실을 이용한다: **화살촉은 결국
선의 국소적인 두께 뭉침이다.** '65' 치수선 끝점 근처를 행(row) 단위로
훑어보면:
  줄기 부분(화살촉 아래): 폭 2px 고정
  화살촉 구간(끝점 근처 15px 정도): 폭이 2 -> 4 -> 6 -> 4 -> 2 로 부풀었다 줄어듦
  (교차하는 가로 연장선 위치엔 폭이 순간적으로 25px로 튐 — 이건 화살촉이
   아니라 다른 선과의 교차이므로 제외해야 함)

이 방식은 LSD 선분 DB에 의존하지 않고 원본 픽셀을 직접 본다 — 그래서
arrowhead.py(선분 끝점 기하 방식)가 "애초에 후보 선분이 DB에 없어서" 놓친
경우(`76` 등)에도 이론적으로 검출 가능하다.
"""
import cv2
import numpy as np

# '65' 치수선 실측(줄기폭 2px, 화살촉 피크폭 6px, 화살촉 길이 ~15px)에 맞춘 기본값.
# 다른 스케일 도면에서는 재조정 필요할 수 있음 — 10장 배치로 검증 전.
BASELINE_SAMPLE_RANGE = (5, 20)   # 줄기 폭 측정 구간(끝점에서 이 정도 뒤로 들어간 곳, px)
SEARCH_RANGE = (0, 45)            # 화살촉을 찾을 구간(끝점 기준 -5~+45px)
BULGE_RATIO = 1.8                 # 줄기 폭의 이 배수 이상이면 "부풀었다"로 판정
MIN_BULGE_LEN = 6.0                # 부푼 구간이 이 정도 길이는 돼야 화살촉(노이즈 제외)
MAX_SPIKE_RATIO = 8.0              # 이 배수 넘는 순간 스파이크는 교차선으로 보고 무시
CORRIDOR_HALF_WIDTH = 25           # 선 방향에 수직으로 얼마나 넓게 볼지(px)
SMOOTH_WINDOW = 3                  # 폭 프로파일 스무딩(교차선 스파이크 억제용)


def _width_profile(img_gray, P, direction_deg, length_back=20, length_fwd=45,
                    half_width=CORRIDOR_HALF_WIDTH):
    """P를 원점으로, direction_deg 방향을 +x축으로 삼아 회전시킨 좁고 긴 복도(corridor)
    이미지를 만들고, 그 복도의 각 x열(원래 선 방향을 따라가는 축)마다 전경(어두운 선)
    픽셀이 y방향으로 몇 px에 걸쳐 있는지(국소 폭)를 잰다.

    반환: (profile, x_offsets) — profile[i]는 x_offsets[i] 위치(P 기준, direction_deg
    방향으로 +)에서의 국소 폭(px). 전경이 없으면 0.
    """
    H, W = img_gray.shape[:2]
    total_len = length_back + length_fwd
    # P가 복도 왼쪽에서 length_back만큼 들어간 위치에 오도록 회전+이동
    M = cv2.getRotationMatrix2D(tuple(P), direction_deg, 1.0)
    M[0, 2] += length_back - P[0]
    M[1, 2] += half_width - P[1]
    corridor = cv2.warpAffine(img_gray, M, (total_len, half_width * 2),
                               flags=cv2.INTER_LINEAR, borderValue=255)

    _, th = cv2.threshold(corridor, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    profile = np.zeros(total_len, dtype=np.float64)
    for x in range(total_len):
        col = th[:, x]
        ys = np.where(col > 0)[0]
        profile[x] = (ys.max() - ys.min() + 1) if len(ys) > 0 else 0.0

    x_offsets = np.arange(total_len) - length_back
    return profile, x_offsets, corridor, th


def _smooth(profile, window=SMOOTH_WINDOW):
    """교차선 때문에 생기는 단일 열짜리 폭 스파이크를 누르기 위한 중앙값 스무딩."""
    if window <= 1:
        return profile.copy()
    pad = window // 2
    padded = np.pad(profile, pad, mode='edge')
    out = np.empty_like(profile)
    for i in range(len(profile)):
        out[i] = np.median(padded[i:i + window])
    return out


def detect_arrowhead_by_width(img_gray, P, direction_deg,
                               baseline_range=BASELINE_SAMPLE_RANGE,
                               search_range=SEARCH_RANGE,
                               bulge_ratio=BULGE_RATIO,
                               min_bulge_len=MIN_BULGE_LEN,
                               max_spike_ratio=MAX_SPIKE_RATIO,
                               half_width=CORRIDOR_HALF_WIDTH,
                               return_debug=False):
    """선분 끝점 P에서, direction_deg 방향(화살촉 팁이 있어야 할 연장방향)으로
    폭 프로파일을 재서 화살촉(폭이 부풀었다 줄어드는 구간)이 있는지 판정.

    반환: dict(found, confidence, bulge_start, bulge_end, peak_width, baseline_width)
    """
    length_back = baseline_range[1] + 5
    length_fwd = search_range[1] + 5
    profile, x_off, corridor, th = _width_profile(
        img_gray, P, direction_deg, length_back, length_fwd, half_width)

    smoothed = _smooth(profile)

    back_mask = (x_off >= -baseline_range[1]) & (x_off <= -baseline_range[0])
    baseline_vals = smoothed[back_mask]
    baseline_vals = baseline_vals[baseline_vals > 0]
    baseline = float(np.median(baseline_vals)) if len(baseline_vals) > 0 else 0.0

    result = {"found": False, "confidence": 0.0, "baseline_width": baseline}
    if baseline <= 0:
        if return_debug:
            result["debug"] = (profile, smoothed, x_off, corridor, th)
        return result

    search_mask = (x_off >= search_range[0]) & (x_off <= search_range[1])
    search_profile = smoothed[search_mask]
    search_x = x_off[search_mask]

    # 교차선 스파이크(줄기폭의 max_spike_ratio배 넘는 순간값)는 화살촉이 아니므로
    # 부풀음 판정에서 상한으로 눌러버림(그 지점의 진짜 폭이 아니라 "부풀었다"로만 취급)
    clipped = np.minimum(search_profile, baseline * max_spike_ratio)

    is_bulge = clipped >= baseline * bulge_ratio
    # 가장 긴 연속 부풀음 구간 찾기
    best_run = None
    i = 0
    while i < len(is_bulge):
        if is_bulge[i]:
            j = i
            while j < len(is_bulge) and is_bulge[j]:
                j += 1
            run_len_px = search_x[j - 1] - search_x[i] + 1
            if best_run is None or run_len_px > best_run[2]:
                best_run = (i, j, run_len_px)
            i = j
        else:
            i += 1

    if best_run is not None and best_run[2] >= min_bulge_len:
        i, j, run_len = best_run
        peak = float(np.max(search_profile[i:j]))
        result.update({
            "found": True,
            "confidence": float(min(1.0, (peak / baseline - bulge_ratio) / bulge_ratio + 0.5)),
            "bulge_start": float(search_x[i]),
            "bulge_end": float(search_x[j - 1]),
            "bulge_len": float(run_len),
            "peak_width": peak,
        })

    if return_debug:
        result["debug"] = (profile, smoothed, x_off, corridor, th)
    return result
