"""
합성 도면 데이터 생성 v5 — 실제 도면 복잡도 미러링
=====================================================
output_v4/real_images 수준의 복잡성을 목표로 한다.

실제 도면 특징 반영(4대 요구):
  1. 한 장에 3~5개 도형(뷰) 배치        → 2×3 그리드 레이아웃
  2. 글씨체 노이즈/분산                  → 폰트 3종·크기지터·굵기·기울기·미세회전 + 후처리 왜곡
  3. 숫자 외 기호(공차/끼워맞춤/지름/각도/
     표면기호/단면기호) + 외곽 타이틀정보 → type 태깅으로 구분
  4. 지운 흔적·선 겹침 등 큰 노이즈       → 후처리(밝은획/어두운획/블러/노이즈/회전/얼룩)

라벨(JSON)은 도면의 모든 텍스트를 아래 형태로 기록한다:
  {"raw": "50±0,1", "value": "50", "type": "dimension_tol",
   "x": .., "y": .., "orient": "h"}
→ 이후 OCR 채점 시 type이 dimension* 인 것만 recall 대상으로 삼는다.

사용법:
  python create_data_v5.py            # 기본 60장
  python create_data_v5.py 120        # 120장
"""

import sys
import json
import csv
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFilter

OUTPUT_DIR = Path('./output_v5')
NUM_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 60
TRAIN_RATIO, VAL_RATIO = 0.7, 0.2
random.seed(7)
np.random.seed(7)

# ── 시트/그리드 좌표계 (A4 세로 비율) ─────────────────────────
SHEET_W, SHEET_H = 620, 820
TB_H = 110                       # 하단 타이틀블록 높이
GRID_COLS, GRID_ROWS = 2, 3
MARGIN = 12
CELL_W = (SHEET_W - 2 * MARGIN) / GRID_COLS
CELL_H = (SHEET_H - TB_H - 2 * MARGIN) / GRID_ROWS

# ── 스타일 풀 (글씨체 분산용) ─────────────────────────────────
FONTS = ['DejaVu Sans', 'DejaVu Serif', 'DejaVu Sans Mono']  # matplotlib 기본 탑재
WEIGHTS = ['normal', 'bold']
STYLES = ['normal', 'normal', 'italic']
BG_COLORS = ['#FAFAFA', '#FFFDE7', '#F1F8E9', '#E8F4FD', '#FFF3E0', '#FFFFFF']
LINE_WIDTHS = [1.0, 1.4, 1.8]

SHAPE_TYPES = [
    'rectangle', 'rect_hole', 'rect_2holes',
    'l_shape', 'stepped', 't_shape', 'hexagon',
]

MATERIALS = ['SS400', 'S45C', 'AL6061', 'SUS304', 'FC250', 'SCM440']
SCALES = ['1:1', '1:2', '2:1', '1:5']
FINISHES = ['▽', '▽▽', '▽▽▽']


# ── 숫자/기호 포매팅 ─────────────────────────────────────────
def numfmt(v):
    """표시용 숫자. 절반 확률로 유럽식 쉼표 소수점 사용(분산)."""
    s = f'{v:g}'
    if random.random() < 0.5 and '.' in s:
        s = s.replace('.', ',')
    return s


def canon(v):
    """정답 비교용 표준 숫자 문자열(항상 마침표)."""
    return f'{v:g}'


def deco(base, allow_symbol=True, is_diameter=False):
    """
    치수 기본값 base를 실제 도면풍 표기로 장식.
    반환: (display, type, value)
      type ∈ dimension / dimension_tol / dimension_fit / diameter
    value = 채점용 표준 숫자(canon)

    규칙:
      - is_diameter=True(원형 피처: 홀/외접원)에만 Ø 접두.
      - 직선 변 치수엔 Ø 금지(평치수/공차/끼워맞춤만).
    """
    prefix = 'Ø' if is_diameter else ''
    base_type = 'diameter' if is_diameter else 'dimension'

    if not allow_symbol or random.random() < 0.4:
        return f'{prefix}{numfmt(base)}', base_type, canon(base)

    style = random.choice(['sym', 'sym', 'fit', 'asym'])
    if style == 'sym':
        tol = random.choice(['0,1', '0,05', '0,2', '0.1', '0,15'])
        return f'{prefix}{numfmt(base)}±{tol}', \
               ('diameter' if is_diameter else 'dimension_tol'), canon(base)
    if style == 'fit':
        # 끼워맞춤 공차는 지름/변 모두 실제로 붙음
        fit = random.choice(['H7', 'H9', 'f7', 'g6', 'h6', 'k6', 'j6'])
        return f'{prefix}{numfmt(base)} {fit}', 'dimension_fit', canon(base)
    if style == 'asym':
        return f'{prefix}{numfmt(base)}+0,1', \
               ('diameter' if is_diameter else 'dimension_tol'), canon(base)
    return f'{prefix}{numfmt(base)}', base_type, canon(base)


# ── 텍스트 렌더 + 정답 기록 ──────────────────────────────────
def put_text(ax, rec, x, y, s, kind, value=None, orient='h', **kw):
    """
    글씨체 분산을 적용해 텍스트를 그리고 정답(rec)에 기록.
    kind: dimension* / diameter / symbol / annotation / info
    """
    font = random.choice(FONTS)
    size = kw.pop('fontsize', random.uniform(9, 13))
    weight = random.choice(WEIGHTS)
    style = random.choice(STYLES)
    rot = kw.pop('rotation', 0) + random.uniform(-2.5, 2.5)  # 미세 기울기

    ax.text(x, y, s, fontfamily=font, fontsize=size,
            fontweight=weight, fontstyle=style, rotation=rot,
            color='#111', **kw)

    rec.append({
        'raw': s,
        'value': value if value is not None else s,
        'type': kind,
        'x': round(float(x), 1),
        'y': round(float(y), 1),
        'orient': orient,
    })


# ── 치수선 헬퍼 (정답 기록형) ────────────────────────────────
def dim_h(ax, rec, x1, x2, y, base, off=20, allow_symbol=True, is_diameter=False):
    disp, kind, val = deco(base, allow_symbol, is_diameter)
    yl = y - off
    ax.annotate('', xy=(x2, yl), xytext=(x1, yl),
                arrowprops=dict(arrowstyle='<->', color='#333', lw=0.8))
    ax.plot([x1, x1], [y, yl], '#666', lw=0.5, ls='--')
    ax.plot([x2, x2], [y, yl], '#666', lw=0.5, ls='--')
    put_text(ax, rec, (x1 + x2) / 2, yl - 3, disp, kind, value=val,
             orient='h', ha='center', va='top',
             bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.85))


def dim_v(ax, rec, x, y1, y2, base, off=22, allow_symbol=True, is_diameter=False):
    disp, kind, val = deco(base, allow_symbol, is_diameter)
    xl = x - off
    ax.annotate('', xy=(xl, y2), xytext=(xl, y1),
                arrowprops=dict(arrowstyle='<->', color='#333', lw=0.8))
    ax.plot([x, xl], [y1, y1], '#666', lw=0.5, ls='--')
    ax.plot([x, xl], [y2, y2], '#666', lw=0.5, ls='--')
    put_text(ax, rec, xl - 4, (y1 + y2) / 2, disp, kind, value=val,
             orient='v', ha='right', va='center', rotation=90,
             bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.85))


def leader(ax, rec, xy, xytext, base):
    # 지시선은 홀(원형)에만 쓰므로 항상 지름
    disp, kind, val = deco(base, allow_symbol=True, is_diameter=True)
    ax.annotate('', xy=xy, xytext=xytext,
                arrowprops=dict(arrowstyle='->', color='#444', lw=0.7))
    put_text(ax, rec, xytext[0], xytext[1], disp, kind, value=val,
             orient='h', ha='left', va='bottom',
             bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.85))


# ── 도형별 렌더 (셀 안에 하나) ────────────────────────────────
def rand_wh(cat):
    ranges = {'small': (40, 70), 'medium': (70, 110), 'large': (110, 145)}
    lo, hi = ranges[cat]
    W = round(min(145, random.uniform(lo, hi)), 1)
    H = round(min(125, random.uniform(lo * 0.7, hi)), 1)
    return W, H


def render_shape(ax, rec, shape, x0, y0, bg, lw):
    """셀 로컬 원점(x0,y0)에 도형 1개 렌더. 치수는 정답 기록."""
    cat = random.choice(['small', 'medium', 'medium', 'large'])
    W, H = rand_wh(cat)

    if shape == 'rectangle':
        ax.add_patch(patches.Rectangle((x0, y0), W, H, lw=lw,
                     edgecolor='#111', facecolor=bg))
        dim_h(ax, rec, x0, x0 + W, y0, W)
        dim_v(ax, rec, x0, y0, y0 + H, H)

    elif shape == 'rect_hole':
        ax.add_patch(patches.Rectangle((x0, y0), W, H, lw=lw,
                     edgecolor='#111', facecolor=bg))
        d = round(random.uniform(4, max(5, min(W, H) * 0.3)), 1)
        cx, cy = x0 + W / 2, y0 + H / 2
        ax.add_patch(plt.Circle((cx, cy), d / 2, lw=lw * 0.7,
                     edgecolor='#111', facecolor='white'))
        ax.plot([cx - d, cx + d], [cy, cy], 'r--', lw=0.5)
        ax.plot([cx, cx], [cy - d, cy + d], 'r--', lw=0.5)
        dim_h(ax, rec, x0, x0 + W, y0, W)
        dim_v(ax, rec, x0, y0, y0 + H, H)
        leader(ax, rec, (cx + d * 0.7, cy + d * 0.7),
               (cx + d + 14, cy + d + 12), d)

    elif shape == 'rect_2holes':
        ax.add_patch(patches.Rectangle((x0, y0), W, H, lw=lw,
                     edgecolor='#111', facecolor=bg))
        d = round(random.uniform(3, max(4, min(W, H) * 0.22)), 1)
        h1 = round(W * random.uniform(0.2, 0.35), 1)
        h2 = round(W * random.uniform(0.65, 0.8), 1)
        cy = y0 + H / 2
        for hx in (h1, h2):
            ax.add_patch(plt.Circle((x0 + hx, cy), d / 2, lw=lw * 0.7,
                         edgecolor='#111', facecolor='white'))
        dim_h(ax, rec, x0 + h1, x0 + h2, cy, round(h2 - h1, 1), off=d + 12)
        dim_h(ax, rec, x0, x0 + W, y0, W)
        dim_v(ax, rec, x0, y0, y0 + H, H)
        leader(ax, rec, (x0 + h2 + d * 0.6, cy + d * 0.6),
               (x0 + h2 + d + 12, cy + d + 10), d)

    elif shape == 'l_shape':
        cw = round(random.uniform(W * 0.3, W * 0.5), 1)
        ch = round(random.uniform(H * 0.3, H * 0.5), 1)
        ax.add_patch(patches.Polygon(
            [(x0, y0), (x0 + W, y0), (x0 + W, y0 + H - ch),
             (x0 + W - cw, y0 + H - ch), (x0 + W - cw, y0 + H), (x0, y0 + H)],
            closed=True, lw=lw, edgecolor='#111', facecolor=bg))
        dim_h(ax, rec, x0, x0 + W, y0, W)
        dim_v(ax, rec, x0, y0, y0 + H, H)
        dim_h(ax, rec, x0 + W - cw, x0 + W, y0 + H, cw, off=14)

    elif shape == 'stepped':
        sw = round(random.uniform(W * 0.3, W * 0.6), 1)
        sh = round(random.uniform(H * 0.3, H * 0.6), 1)
        ax.add_patch(patches.Polygon(
            [(x0, y0), (x0 + W, y0), (x0 + W, y0 + sh),
             (x0 + sw, y0 + sh), (x0 + sw, y0 + H), (x0, y0 + H)],
            closed=True, lw=lw, edgecolor='#111', facecolor=bg))
        dim_h(ax, rec, x0, x0 + W, y0, W)
        dim_v(ax, rec, x0, y0, y0 + H, H)
        dim_v(ax, rec, x0 + sw, y0, y0 + sh, sh, off=16)

    elif shape == 't_shape':
        th = round(H * random.uniform(0.25, 0.45), 1)
        stw = round(W * random.uniform(0.3, 0.5), 1)
        sth = round(H - th, 1)
        sx = x0 + (W - stw) / 2
        ax.add_patch(patches.Rectangle((x0, y0 + sth), W, th, lw=lw,
                     edgecolor='#111', facecolor=bg))
        ax.add_patch(patches.Rectangle((sx, y0), stw, sth, lw=lw,
                     edgecolor='#111', facecolor=bg))
        dim_h(ax, rec, x0, x0 + W, y0 + sth, W)
        dim_v(ax, rec, x0, y0, y0 + H, H)
        dim_h(ax, rec, sx, sx + stw, y0, stw, off=12)

    elif shape == 'hexagon':
        R = round(min(W, H) / 2 * random.uniform(0.7, 0.95), 1)
        cx, cy = x0 + W / 2, y0 + H / 2
        ang = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
        pts = [(cx + R * np.cos(a), cy + R * np.sin(a)) for a in ang]
        ax.add_patch(patches.Polygon(pts, closed=True, lw=lw,
                     edgecolor='#111', facecolor=bg))
        dia = round(R * 2, 1)
        ax.plot([cx - R, cx + R], [cy, cy], '#999', lw=0.5, ls='--')
        dim_h(ax, rec, cx - R, cx + R, cy, dia, is_diameter=True)
        flat = round(R * np.sqrt(3), 1)
        dim_v(ax, rec, cx - R, cy - flat / 2, cy + flat / 2, flat, off=24)

    # 셀에 부가 기호(치수 아님) — 요구3: 숫자/기호 구분 학습용
    add_cell_symbols(ax, rec, x0, y0, W, H)


def add_cell_symbols(ax, rec, x0, y0, W, H):
    """단면기호(A-A), 표면기호(▽), 각도/모따기 등 비치수 기호."""
    # 단면 지시 화살표 + 문자
    if random.random() < 0.5:
        letter = random.choice(['A', 'B', 'C'])
        sx = x0 + W * random.uniform(0.3, 0.7)
        ax.annotate('', xy=(sx, y0 + H + 8), xytext=(sx, y0 + H + 20),
                    arrowprops=dict(arrowstyle='->', color='#111', lw=1.0))
        put_text(ax, rec, sx + 3, y0 + H + 18, letter, 'symbol',
                 fontsize=11, ha='left', va='center')
    # 표면 거칠기 기호
    if random.random() < 0.4:
        put_text(ax, rec, x0 + W + 6, y0 + H * random.uniform(0.3, 0.8),
                 random.choice(FINISHES), 'symbol', fontsize=12,
                 ha='left', va='center')
    # 모따기 2x45°
    if random.random() < 0.3:
        a = random.choice(['2x45°', '1x45°', '0,5x45°', '3x45°'])
        put_text(ax, rec, x0 + W * 0.5, y0 - 6, a, 'annotation',
                 fontsize=9, ha='center', va='top')
    # 원문자 뷰 번호 ①②③ (원 + 숫자)
    if random.random() < 0.5:
        num = random.randint(1, 4)
        cx, cy = x0 - 2, y0 + H + 16
        ax.add_patch(plt.Circle((cx, cy), 8, lw=1.0,
                     edgecolor='#111', facecolor='white'))
        put_text(ax, rec, cx, cy, str(num), 'symbol', fontsize=10,
                 ha='center', va='center')


# ── 타이틀블록 (외곽 정보 — 치수 아님) ───────────────────────
def render_titleblock(ax, rec, params):
    y_top = TB_H
    # 블록 테두리
    ax.add_patch(patches.Rectangle((MARGIN, MARGIN), SHEET_W - 2 * MARGIN,
                 TB_H - MARGIN, lw=1.5, edgecolor='#222', facecolor='#F2F5F8'))
    # 세로 구분선
    for fx in (0.28, 0.55, 0.78):
        x = MARGIN + (SHEET_W - 2 * MARGIN) * fx
        ax.plot([x, x], [MARGIN, TB_H], '#222', lw=1.0)

    info = [
        (MARGIN + 6, TB_H - 18, f"PART No. {params['id']}", 'info'),
        (MARGIN + 6, TB_H - 42, f"TITLE  {params['title']}", 'info'),
        (MARGIN + 6, TB_H - 66, f"MAT  {params['material']}", 'info'),
        ((MARGIN + (SHEET_W - 2 * MARGIN) * 0.28) + 6, TB_H - 18,
         f"SCALE {params['scale']}", 'info'),
        ((MARGIN + (SHEET_W - 2 * MARGIN) * 0.28) + 6, TB_H - 42,
         f"UNIT mm", 'info'),
        ((MARGIN + (SHEET_W - 2 * MARGIN) * 0.55) + 6, TB_H - 18,
         f"DATE {params['date']}", 'info'),
        ((MARGIN + (SHEET_W - 2 * MARGIN) * 0.55) + 6, TB_H - 42,
         f"DRAWN {params['drawn']}", 'info'),
        ((MARGIN + (SHEET_W - 2 * MARGIN) * 0.78) + 6, TB_H - 30,
         f"REV {params['rev']}", 'info'),
    ]
    for x, y, s, kind in info:
        put_text(ax, rec, x, y, s, kind, fontsize=8.5, ha='left', va='center')


# ── 시트 1장 렌더 ────────────────────────────────────────────
def render_sheet(params, save_path):
    fig, ax = plt.subplots(figsize=(6.2, 8.2), facecolor='white')
    ax.set_facecolor('#FCFCFC')
    rec = []

    # 셀 위치 목록 (하단 타이틀블록 위쪽 영역)
    cells = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cx = MARGIN + c * CELL_W
            cy = TB_H + MARGIN + r * CELL_H
            cells.append((cx, cy))
    random.shuffle(cells)

    # 도형 4~6개 배치 (밀도 ↑)
    n_shapes = random.randint(4, 6)
    shapes_used = []
    for (cx, cy) in cells[:n_shapes]:
        shape = random.choice(SHAPE_TYPES)
        shapes_used.append(shape)
        # 셀 내부 여백: 치수선이 왼쪽/아래로 뻗으므로 padding 확보
        x0 = cx + 50
        y0 = cy + 42
        bg = random.choice(BG_COLORS)
        lw = random.choice(LINE_WIDTHS)
        render_shape(ax, rec, shape, x0, y0, bg, lw)

    # 외곽 시트 테두리
    ax.add_patch(patches.Rectangle((4, 4), SHEET_W - 8, SHEET_H - 8,
                 lw=2, edgecolor='#222', facecolor='none'))
    # 타이틀블록
    render_titleblock(ax, rec, params)

    ax.set_xlim(0, SHEET_W)
    ax.set_ylim(0, SHEET_H)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(save_path, dpi=150, facecolor='white')
    plt.close()

    return rec, shapes_used


# ── 후처리 노이즈 (요구4: 지운자국·선겹침·대노이즈) ──────────
NOISE_CFG = {
    #            dark  light blur  sigma  rot    smudge
    'clean':    (0,    0,    0.0,  0,     0.0,   0),
    'slight':   (2,    1,    0.4,  4,     0.5,   0),
    'noisy':    (5,    3,    0.7,  8,     1.0,   2),
    'heavy':    (11,   7,    1.1,  15,    1.8,   4),
}


def degrade(path, level, rng):
    if level == 'clean':
        return
    n_dark, n_light, blur, sigma, rot_max, n_smudge = NOISE_CFG[level]
    im = Image.open(path).convert('RGB')
    w, h = im.size

    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # 어두운 획 = 선 겹침 / 잘못 그은 선
    for _ in range(n_dark):
        x1, y1 = rng.integers(0, w), rng.integers(0, h)
        x2 = x1 + rng.integers(-w // 3, w // 3)
        y2 = y1 + rng.integers(-h // 6, h // 6)
        gray = int(rng.integers(40, 110))
        a = int(rng.integers(90, 180))
        d.line([(x1, y1), (x2, y2)], fill=(gray, gray, gray, a),
               width=int(rng.integers(1, 4)))

    # 밝은 두꺼운 획 = 지운 흔적(지우개 자국)
    for _ in range(n_light):
        x1, y1 = rng.integers(0, w), rng.integers(0, h)
        x2 = x1 + rng.integers(-w // 4, w // 4)
        y2 = y1 + rng.integers(-h // 8, h // 8)
        val = int(rng.integers(215, 245))
        a = int(rng.integers(120, 210))
        d.line([(x1, y1), (x2, y2)], fill=(val, val, val, a),
               width=int(rng.integers(6, 16)))

    # 얼룩 반점
    for _ in range(n_smudge):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        rad = int(rng.integers(min(w, h) // 20, min(w, h) // 7))
        tone = int(rng.integers(180, 220))
        a = int(rng.integers(40, 90))
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                  fill=(tone, tone, tone - 10, a))

    im = Image.alpha_composite(im.convert('RGBA'), overlay).convert('RGB')

    if blur > 0:
        im = im.filter(ImageFilter.GaussianBlur(blur))

    # 가우시안 픽셀 노이즈
    arr = np.asarray(im).astype(np.int16)
    arr += rng.normal(0, sigma, arr.shape).astype(np.int16)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # 미세 회전(스캔 기울기)
    if rot_max > 0:
        ang = float(rng.uniform(-rot_max, rot_max))
        im = im.rotate(ang, expand=False, fillcolor=(255, 255, 255),
                       resample=Image.BICUBIC)

    im.save(path)


# ── 파라미터 ─────────────────────────────────────────────────
def gen_params(idx):
    return {
        'id': f'{idx:04d}',
        'title': f"{random.choice(['BRACKET','HOUSING','PLATE','SHAFT','COVER','FLANGE'])}"
                 f"-{random.randint(100,999)}",
        'material': random.choice(MATERIALS),
        'scale': random.choice(SCALES),
        'date': f"2026-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        'drawn': random.choice(['KS.LEE', 'JH.KIM', 'SW.PARK', 'MJ.CHOI']),
        'rev': random.choice(['A', 'B', 'C', '0', '1']),
        'noise': random.choices(
            ['clean', 'slight', 'noisy', 'heavy'],
            weights=[0.15, 0.3, 0.3, 0.25])[0],
    }


# ── 실행 ─────────────────────────────────────────────────────
def run():
    (OUTPUT_DIR / 'images').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'labels').mkdir(parents=True, exist_ok=True)

    records = []
    noise_counts = {'clean': 0, 'slight': 0, 'noisy': 0, 'heavy': 0}
    print(f"도면 생성 중... (총 {NUM_SAMPLES}장, 실제 도면 복잡도)")

    for i in range(NUM_SAMPLES):
        p = gen_params(i)
        ip = OUTPUT_DIR / 'images' / f'drawing_{i:04d}.png'
        lp = OUTPUT_DIR / 'labels' / f'drawing_{i:04d}.json'

        rec, shapes = render_sheet(p, ip)

        # 후처리 노이즈 (도면마다 독립 시드로 재현성 확보)
        rng = np.random.default_rng(1000 + i)
        degrade(ip, p['noise'], rng)

        # 정답 라벨 저장
        dim_values = [t['value'] for t in rec
                      if t['type'].startswith('dimension') or t['type'] == 'diameter']
        label = {
            'id': p['id'],
            'image': str(ip),
            'shapes': shapes,
            'n_shapes': len(shapes),
            'noise': p['noise'],
            'titleblock': {k: p[k] for k in
                           ('title', 'material', 'scale', 'date', 'drawn', 'rev')},
            'texts': rec,
            'dimension_values': dim_values,   # ← OCR recall 채점 대상
            'n_dimensions': len(dim_values),
        }
        with open(lp, 'w', encoding='utf-8') as f:
            json.dump(label, f, indent=2, ensure_ascii=False)

        records.append({
            'id': p['id'], 'n_shapes': len(shapes),
            'n_dimensions': len(dim_values), 'noise': p['noise'],
            'image': str(ip), 'label': str(lp),
        })
        noise_counts[p['noise']] += 1

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{NUM_SAMPLES} 완료")

    # split
    random.shuffle(records)
    n = len(records)
    for i, r in enumerate(records):
        r['split'] = ('train' if i < int(n * TRAIN_RATIO)
                      else 'val' if i < int(n * (TRAIN_RATIO + VAL_RATIO))
                      else 'test')

    with open(OUTPUT_DIR / 'dataset.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)

    total_dims = sum(r['n_dimensions'] for r in records)
    print(f"\n{'='*55}")
    print(f"  생성 완료: {len(records)}장")
    print(f"  노이즈 분포: {noise_counts}")
    print(f"  총 치수 텍스트: {total_dims}개 (장당 평균 {total_dims/max(1,n):.1f})")
    print(f"  저장 위치: {OUTPUT_DIR.resolve()}")
    print(f"{'='*55}")


if __name__ == '__main__':
    run()
