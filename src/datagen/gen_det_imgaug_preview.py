# -*- coding: utf-8 -*-
"""
det 이미지 증강 프리뷰 — ±45도 회전 + TIA 왜곡(공간 휘어짐)
============================================================
텍스트 증강(gen_det_overlay) 후 추가로 적용할 '이미지 레벨' 증강 검수용.
박스도 같은 변환으로 이동시켜서 정합 확인.
  - 회전: affine 행렬로 이미지+박스 동시 회전(±45도), 흰 배경 확장
  - 왜곡: WarpMLS(공간 휘어짐, tia_distort 방식). 박스 4점은 동일한 제어점
    대응(src_pts→dst_pts)의 forward affine-MLS로 이동 (WarpMLS 내부와 같은 수식).

출력: results/imgaug_preview/{...}_rot+45 / _rot-45 / _warp .png (박스 오버레이)
"""

import json
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_det_overlay as g
from zone_utils import split_zone
from warp_mls import WarpMLS

random.seed(5)
np.random.seed(5)

BG = 'Gripper.jpg'
OUT = 'results/imgaug_preview'


def load_aug():
    """Gripper에 텍스트증강 적용 → (PIL이미지, 전체 박스 리스트)."""
    for line in open(g.REAL_LABEL, encoding='utf-8'):
        path, js = line.split('\t', 1)
        if path.endswith(BG):
            raw = json.loads(js)
            break
    _z, dr, meta = split_zone(raw)
    items = dr + meta
    img = Image.open(g.REAL_DIR / BG).convert('RGB')
    aug, extra = g.augment_image(img, items)
    boxes = [it['points'] for it in (g.clean_items(items) + extra)]
    return aug, boxes


# ── 회전 (이미지+박스) ────────────────────────────────────────
def rotate(img, boxes, deg):
    arr = np.array(img)
    h, w = arr.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - cx
    M[1, 2] += nh / 2 - cy
    out = cv2.warpAffine(arr, M, (nw, nh), borderValue=(255, 255, 255))
    new_boxes = []
    for pts in boxes:
        p = np.array([[x, y, 1] for x, y in pts]).T  # 3xN
        q = (M @ p).T
        new_boxes.append([[float(x), float(y)] for x, y in q])
    return Image.fromarray(out), new_boxes


# ── forward affine-MLS 점변환 (WarpMLS와 동일 수식) ───────────
def mls_affine_points(query, src_pts, dst_pts):
    """query 점들을 (src_pts→dst_pts) 대응 기반 affine MLS로 이동."""
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    out = []
    for v in np.asarray(query, dtype=np.float64):
        d2 = np.sum((src - v) ** 2, axis=1)
        hit = np.where(d2 < 1e-8)[0]
        if len(hit):
            out.append(dst[hit[0]].tolist())
            continue
        w = 1.0 / d2
        pstar = (w[:, None] * src).sum(0) / w.sum()
        qstar = (w[:, None] * dst).sum(0) / w.sum()
        phat = src - pstar
        qhat = dst - qstar
        # M = (sum w phat^T phat)^-1 ; f(v) = (v-pstar) M (sum w phat^T qhat) + qstar
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))
        for i in range(len(src)):
            A += w[i] * np.outer(phat[i], phat[i])
            B += w[i] * np.outer(phat[i], qhat[i])
        try:
            fv = (v - pstar) @ np.linalg.inv(A) @ B + qstar
        except np.linalg.LinAlgError:
            fv = v - pstar + qstar
        out.append(fv.tolist())
    return out


def warp(img, boxes, segment=4, thresh_ratio=0.03):
    arr = np.array(img)
    h, w = arr.shape[:2]
    cut = w // segment
    thresh = int(min(h, w) * thresh_ratio)
    src_pts, dst_pts = [], []
    ri = lambda: np.random.randint(thresh)
    src_pts += [[0, 0], [w, 0], [w, h], [0, h]]
    dst_pts += [[ri(), ri()], [w - ri(), ri()], [w - ri(), h - ri()], [ri(), h - ri()]]
    for c in range(1, segment):
        src_pts += [[cut * c, 0], [cut * c, h]]
        dst_pts += [[cut * c + ri() - thresh // 2, ri()],
                    [cut * c + ri() - thresh // 2, h - ri()]]
    trans = WarpMLS(arr, src_pts, dst_pts, w, h)
    out = trans.generate()
    new_boxes = [mls_affine_points(pts, src_pts, dst_pts) for pts in boxes]
    return Image.fromarray(out.astype(np.uint8)), new_boxes


def draw(img, boxes, path):
    vis = img.copy()
    d = ImageDraw.Draw(vis)
    for pts in boxes:
        d.polygon([tuple(p) for p in pts], outline=(255, 0, 0), width=2)
    vis.save(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    aug, boxes = load_aug()
    stem = os.path.splitext(BG)[0]

    ri, rb = rotate(aug, boxes, 45)
    draw(ri, rb, f'{OUT}/{stem}_rot+45.png')
    ri, rb = rotate(aug, boxes, -45)
    draw(ri, rb, f'{OUT}/{stem}_rot-45.png')
    wi, wb = warp(aug, boxes)
    draw(wi, wb, f'{OUT}/{stem}_warp.png')
    print(f'출력: {OUT} (rot+45 / rot-45 / warp)')


if __name__ == '__main__':
    main()
