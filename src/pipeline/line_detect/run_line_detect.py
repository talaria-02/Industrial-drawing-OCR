# -*- coding: utf-8 -*-
"""이미지 1장 -> 선분 검출 파이프라인 실행 (Stage A: 고전 LSD 백엔드).

process_image_lsd()   : 전처리 -> (선택)타일링 -> LSD -> 후처리 -> JSON/통계
process_image_hough()  : 기존 HoughLinesP 전역(이미지 전체) 베이스라인
                          (anchor_all_test.py의 "bbox 근처 crop" 버전과 다르게,
                          여기선 LSD와 공정 비교를 위해 이미지 전체에 대해 돌림)

두 함수 모두 같은 형태의 stats dict를 반환해서 나란히 비교하기 쉽게 한다.
"""
import json
import os
import time

import cv2
import numpy as np

from . import tiling, backend_lsd, refine


def _angle_stats(lines):
    if len(lines) == 0:
        return 0, 0, 0
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    ang = np.degrees(np.arctan2(dy, dx))
    near_vert = np.sum(np.minimum(np.abs(np.abs(ang) - 90), np.abs(np.abs(ang) - 270)) < 5)
    near_horiz = np.sum((np.abs(ang) < 5) | (np.abs(np.abs(ang) - 180) < 5))
    diagonal = len(lines) - near_vert - near_horiz
    return int(near_vert), int(near_horiz), int(diagonal)


def load_gray_color(img_path):
    color = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    return color, gray


def process_image_lsd(img_path, use_tiling=False, tile=512, overlap=64,
                       refine_kwargs=None):
    """Stage A 5단계(1,3,4,5 — 2단계는 DL 생략) 전체 실행."""
    color, gray = load_gray_color(img_path)
    H, W = gray.shape
    refine_kwargs = refine_kwargs or {}

    t0 = time.time()
    if use_tiling:
        boxes = tiling.make_tiles(H, W, tile=tile, overlap=overlap)
        all_lines = []
        for (x0, y0, x1, y1) in boxes:
            patch_gray = gray[y0:y1, x0:x1]
            local_lines = backend_lsd.detect(patch_gray)
            all_lines.append(tiling.to_global(local_lines, x0, y0))
        raw_lines = np.concatenate(all_lines, axis=0) if all_lines else np.empty((0, 4))
    else:
        raw_lines = backend_lsd.detect(gray)
    t_detect = time.time() - t0

    t1 = time.time()
    final_lines, refine_stats = refine.refine_pipeline(raw_lines, **refine_kwargs)
    t_refine = time.time() - t1

    elapsed = time.time() - t0
    n_vert, n_horiz, n_diag = _angle_stats(final_lines)

    stats = {
        "method": "lsd_tiled" if use_tiling else "lsd",
        "elapsed_sec": elapsed,
        "detect_sec": t_detect,
        "refine_sec": t_refine,
        "n_lines_final": int(len(final_lines)),
        "n_vertical": n_vert,
        "n_horizontal": n_horiz,
        "n_diagonal": n_diag,
        **refine_stats,
    }
    return color, final_lines, stats


def process_image_hough(img_path, canny_lo=50, canny_hi=150,
                         hough_threshold=80, min_len_ratio=0.02, max_gap=5):
    """기존 HoughLinesP를 이미지 전체(global)에 그대로 적용한 베이스라인.
    (anchor_all_test.py는 숫자 bbox 근처 crop에만 적용 — 여기선 LSD와
    같은 "전역 선분 검출" 과제로 공정 비교하기 위해 전체 이미지에 돌림)"""
    color, gray = load_gray_color(img_path)
    H, W = gray.shape

    t0 = time.time()
    edges = cv2.Canny(gray, canny_lo, canny_hi, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_threshold,
                             minLineLength=int(W * min_len_ratio), maxLineGap=max_gap)
    elapsed = time.time() - t0

    final_lines = np.empty((0, 4)) if lines is None else lines.reshape(-1, 4).astype(np.float64)
    n_vert, n_horiz, n_diag = _angle_stats(final_lines)

    stats = {
        "method": "hough",
        "elapsed_sec": elapsed,
        "n_lines_final": int(len(final_lines)),
        "n_vertical": n_vert,
        "n_horizontal": n_horiz,
        "n_diagonal": n_diag,
    }
    return color, final_lines, stats


def draw_lines(color_img, lines, ortho_color=(0, 0, 255), diag_color=(255, 140, 0), thickness=1):
    """직교(수직/수평) 선은 빨강, 사선은 주황으로 구분해서 그림."""
    vis = color_img.copy()
    for (x1, y1, x2, y2) in lines:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        is_diag = not (np.abs(ang) < 5 or np.abs(np.abs(ang) - 180) < 5
                       or np.abs(np.abs(ang) - 90) < 5 or np.abs(np.abs(ang) - 270) < 5)
        color = diag_color if is_diag else ortho_color
        cv2.line(vis, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, thickness)
    return vis


def save_png(path, image):
    cv2.imencode('.png', image)[1].tofile(path)


def save_lines_json(path, lines, stats, image_name):
    payload = {
        "image": image_name,
        "stats": stats,
        "lines": [{"x1": float(l[0]), "y1": float(l[1]),
                   "x2": float(l[2]), "y2": float(l[3])} for l in lines],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
