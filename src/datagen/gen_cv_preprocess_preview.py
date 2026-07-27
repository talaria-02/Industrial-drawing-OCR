# -*- coding: utf-8 -*-
"""고전 CV 전처리 검증 — Canny / Hough(HoughLinesP) / RANSAC 직선피팅을
data/real/val 이미지 전체에 적용해서 결과 저장.

3개 다 원본이 아니라 Canny 경계맵을 입력으로 받는 파이프라인(원본→Canny→Hough/RANSAC).
출력: data/generated/cv_preprocess_preview/{stem}_1_canny.png / _2_hough.png / _3_ransac.png
"""
import glob
import os

import cv2
import numpy as np
from skimage.measure import LineModelND, ransac

SRC_DIR = 'data/real/val'
OUT_DIR = 'data/generated/cv_preprocess_preview'

RANSAC_MAX_LINES = 40
RANSAC_MIN_INLIERS = 150
RANSAC_RESIDUAL = 2.0


def process(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f'로드 실패: {path}')
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # 1) Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imencode('.png', edges)[1].tofile(f'{OUT_DIR}/{stem}_1_canny.png')

    # 2) Hough (확률적, Canny 경계맵 입력)
    vis_hough = img.copy()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=int(W * 0.02), maxLineGap=5)
    n_lines = 0
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l.ravel()
            cv2.line(vis_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)
            n_lines += 1
    cv2.imencode('.png', vis_hough)[1].tofile(f'{OUT_DIR}/{stem}_2_hough.png')

    # 3) RANSAC 직선 피팅 (Canny 경계픽셀 점군에서 반복적으로 최대 inlier 직선 추출)
    ys, xs = np.nonzero(edges)
    pts = np.column_stack([xs, ys]).astype(float)
    vis_ransac = img.copy()
    remaining = pts.copy()
    n_ransac = 0
    rng = np.random.default_rng(0)
    for _ in range(RANSAC_MAX_LINES):
        if len(remaining) < 200:
            break
        model, inliers = ransac(remaining, LineModelND, min_samples=2,
                                 residual_threshold=RANSAC_RESIDUAL, max_trials=200, rng=rng)
        if inliers is None or inliers.sum() < RANSAC_MIN_INLIERS:
            break
        in_pts = remaining[inliers]
        origin, direction = model.params
        direction = direction / np.linalg.norm(direction)
        t = (in_pts - origin) @ direction
        p0 = origin + t.min() * direction
        p1 = origin + t.max() * direction
        cv2.line(vis_ransac, (int(round(p0[0])), int(round(p0[1]))),
                 (int(round(p1[0])), int(round(p1[1]))), (255, 0, 0), 2)
        n_ransac += 1
        remaining = remaining[~inliers]
    cv2.imencode('.png', vis_ransac)[1].tofile(f'{OUT_DIR}/{stem}_3_ransac.png')

    print(f'{stem}: Hough {n_lines}개, RANSAC {n_ransac}개')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(f'{SRC_DIR}/*.jpg')) + sorted(glob.glob(f'{SRC_DIR}/*.png'))
    print(f'대상: {len(paths)}장')
    for p in paths:
        process(p)
    print(f'완료 -> {OUT_DIR}')


if __name__ == '__main__':
    main()
