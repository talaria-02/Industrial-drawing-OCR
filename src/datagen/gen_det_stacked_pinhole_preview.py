# -*- coding: utf-8 -*-
"""
det 증강 프리뷰 — 적층공차(공칭/상한/하한 별도박스) + 핀홀 하드네거티브
========================================================================
목적 2가지:
  1. 적층공차: 공칭값 + 상한(+0.05) + 하한(-0.02)을 실도면 배경 위에 얹되,
     셋을 '명확히 떨어진 별도 박스' 3개로 라벨 → det이 통짜로 묶지 않고
     각각 분리 검출하도록 학습.
  2. 핀홀/센터마크: 구멍단면·중심마크 그래픽을 얹되 박스 0개(하드네거티브)
     → det이 이런 도형을 텍스트로 오검출하지 않도록.

이 스크립트는 검수용 프리뷰 5장만 생성. GT박스를 색으로 그려서(공칭=빨강,
상한=파랑, 하한=초록, 핀홀=박스없음) 눈으로 확인.
출력: results/stacked_pinhole_preview/
"""

import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_det_overlay import compose_word  # 글자별 왜곡 조립단어
from zone_utils import split_zone

random.seed(11)

REAL_LABEL = Path('data/real/train/Label.txt')
REAL_DIR = Path('data/real/train')
OUT_DIR = Path('results/stacked_pinhole_preview')

# 프리뷰용 배경 (치수 여백 있는 실도면 몇 장)
BG_IMAGES = ['Gripper.jpg', 'BM_part.jpg', 'Candle_holder.jpg',
             'LIU0010.jpg', 'Dessin-ind-Omnifab-vis-page-001-1-2.jpg']

N_STACKS_PER_IMG = (5, 8)
N_PINHOLES_PER_IMG = (3, 6)


def paste_multiply(img, word_rgba, x, y):
    """조립단어를 (x,y) 좌상단에 multiply 블렌드. 반환: 실제 박스 (x0,y0,x1,y1)."""
    w, h = word_rgba.width, word_rgba.height
    W, H = img.size
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    region = img.crop((x0, y0, x1, y1)).convert('RGB')
    crop = word_rgba.crop((x0 - x, y0 - y, x1 - x, y1 - y))
    _, _, _, a = crop.split()
    black = Image.new('RGB', region.size, (12, 12, 12))
    region = Image.composite(black, region, a)
    img.paste(region, (x0, y0))
    return (x0, y0, x1, y1)


def rand_nominal():
    dia = random.random() < 0.5
    v = random.choice([random.randint(5, 200),
                       round(random.uniform(5, 200), 1)])
    fit = random.choice(['', '', 'H7', 'g6', 'f7', 'h6'])
    return f'{"ø" if dia else ""}{v}{fit}'


def rand_tol():
    return random.choice(['+0.05', '+0.1', '+0.02', '+0.15', '+0.2',
                          '-0.05', '-0.1', '-0.02', '-0.15', '0'])


def place_stacked(img, x, y, base_h, boxes_out):
    """공칭 + 상한 + 하한을 '명확히 떨어진' 별도 박스 3개로 배치.
    공칭은 크게, 상/하한은 그 오른쪽에 위/아래로 — 상한과 하한 사이에 공칭 세로중심을
    기준으로 뚜렷한 세로 gap을 둬서 det이 3개를 확실히 분리 학습하도록."""
    vgap = max(3, int(base_h * 0.2))        # 상/하한 세로 간격(좁게)
    hgap = max(3, int(base_h * 0.25))       # 공칭↔공차 가로 간격(기존의 절반)

    nominal = compose_word(rand_nominal(), base_h)
    if nominal is None:
        return
    nb = paste_multiply(img, nominal, x, y)
    if nb is None:
        return
    boxes_out.append((nb, 'nominal'))

    small_h = max(12, int(base_h * 0.7))   # 상/하한도 읽힐 만큼 키움
    up = compose_word(rand_tol(), small_h)
    lo = compose_word(rand_tol(), small_h)
    if up is None or lo is None:
        return

    tx = nb[2] + hgap                       # 공칭 오른쪽
    ctr = (nb[1] + nb[3]) // 2              # 공칭 세로 중심
    # 상한: 박스 하단이 중심-vgap 위
    uy = ctr - vgap - up.height
    ub = paste_multiply(img, up, tx, uy)
    if ub:
        boxes_out.append((ub, 'upper'))
    # 하한: 박스 상단이 중심+vgap 아래 (상한과 세로 gap = 2*vgap)
    ly = ctr + vgap
    lb = paste_multiply(img, lo, tx, ly)
    if lb:
        boxes_out.append((lb, 'lower'))


def draw_pinhole(img, cx, cy, r):
    """중심마크(원+십자 점선) 또는 구멍단면(동심원+해칭). 박스 없음(하드네거티브)."""
    d = ImageDraw.Draw(img)
    kind = random.choice(['centermark', 'hole_section', 'concentric'])
    c = (20, 20, 20)
    if kind == 'centermark':
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=c, width=2)
        ext = int(r * 1.6)
        for i in range(-ext, ext, 6):  # 점선 느낌
            pass
        d.line([cx - ext, cy, cx + ext, cy], fill=c, width=1)
        d.line([cx, cy - ext, cx, cy + ext], fill=c, width=1)
    elif kind == 'concentric':
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=c, width=2)
        r2 = int(r * 0.6)
        d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], outline=c, width=2)
        d.line([cx - r * 1.4, cy, cx + r * 1.4, cy], fill=c, width=1)
        d.line([cx, cy - r * 1.4, cx, cy + r * 1.4], fill=c, width=1)
    else:  # hole_section: 원 + 해칭선
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=c, width=2)
        for off in range(-r, r, 5):
            d.line([cx + off, cy - r, cx + off + r, cy + r], fill=c, width=1)


def rects_overlap(a, b, margin=10):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + margin < bx0 or bx1 + margin < ax0
                or ay1 + margin < by0 or by1 + margin < ay0)


def gen_one(bg_name):
    line = None
    for L in open(REAL_LABEL, encoding='utf-8'):
        if L.split('\t', 1)[0].endswith(bg_name):
            line = L
            break
    if line is None:
        return None
    items = json.loads(line.split('\t', 1)[1])
    _z, drawing, meta = split_zone(items)
    real_boxes = []
    for it in drawing + meta:
        xs = [p[0] for p in it['points']]
        ys = [p[1] for p in it['points']]
        real_boxes.append((min(xs), min(ys), max(xs), max(ys)))

    img = Image.open(REAL_DIR / bg_name).convert('RGB')
    W, H = img.size
    placed = list(real_boxes)
    new_boxes = []

    # 적층공차
    n_stack = random.randint(*N_STACKS_PER_IMG)
    tries = 0
    while len([b for b in new_boxes]) < n_stack * 2 and tries < n_stack * 30:
        tries += 1
        base_h = random.randint(int(H * 0.012), int(H * 0.025))
        x = random.randint(int(W * 0.05), int(W * 0.8))
        y = random.randint(int(H * 0.05), int(H * 0.85))
        probe = (x, y, x + base_h * 5, y + base_h * 2)
        if any(rects_overlap(probe, p) for p in placed):
            continue
        before = len(new_boxes)
        place_stacked(img, x, y, base_h, new_boxes)
        for (bb, _kind) in new_boxes[before:]:
            placed.append(bb)

    # 핀홀 하드네거티브 (박스 없음)
    n_pin = random.randint(*N_PINHOLES_PER_IMG)
    tries = 0
    done = 0
    while done < n_pin and tries < n_pin * 30:
        tries += 1
        r = random.randint(int(H * 0.01), int(H * 0.025))
        cx = random.randint(int(W * 0.05), int(W * 0.9))
        cy = random.randint(int(H * 0.05), int(H * 0.9))
        probe = (cx - r * 1.6, cy - r * 1.6, cx + r * 1.6, cy + r * 1.6)
        if any(rects_overlap(probe, p) for p in placed):
            continue
        draw_pinhole(img, cx, cy, r)
        placed.append(probe)
        done += 1

    return img, new_boxes


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    colors = {'nominal': (255, 0, 0), 'upper': (0, 90, 255), 'lower': (0, 160, 0)}
    for name in BG_IMAGES:
        res = gen_one(name)
        if res is None:
            print(f'스킵(라벨없음): {name}')
            continue
        img, boxes = res
        vis = img.copy()
        d = ImageDraw.Draw(vis)
        for (bb, kind) in boxes:
            d.rectangle(bb, outline=colors[kind], width=3)
        stem = Path(name).stem
        vis.save(OUT_DIR / f'{stem}_preview.png')
        print(f'{name}: 적층박스 {len(boxes)}개 (핀홀은 박스없음) -> {stem}_preview.png')
    print(f'출력: {OUT_DIR}')


if __name__ == '__main__':
    main()
