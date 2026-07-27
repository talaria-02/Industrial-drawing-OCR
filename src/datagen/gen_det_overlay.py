# -*- coding: utf-8 -*-
"""
det 학습 데이터 — 실도면 위 '글자 갈아끼우기' 방식
====================================================
컴퓨터 폰트로 통짜 합성한 도면은 실제와 괴리가 큼(폰트 stroke/레이아웃/기호형태).
대신 실도면을 배경으로 그대로 쓰고, 기존 텍스트박스의 일부(40%)만
 (1) 박스영역을 흰색쪽으로 80% 페이드(원글자 잔상만 남김)
 (2) 그 자리에 '글자별 왜곡+합성'한 조립단어를 quad각도 맞춰 얹음(multiply 블렌드)
하는 방식. 배경/노이즈/해상도/텍스트 위치분포가 전부 진짜.

글자별 왜곡(단순 폰트의 "너무 완벽함"을 깸):
  - 글자마다 폰트 랜덤(한 단어 안에서도 섞음 — det은 글리프정확도 무관)
  - 미세회전 / 베이스라인 지터 / 스케일 지터 / 굵기 morphology(팽창·침식)
  - baseline 정렬(소수점 '.'이 바닥에 붙도록 — 중앙정렬 금지)
특수기호(원문자 ①~⑳, GD&T ⊥∥⏥, 거칠기 √▽▽ₒ, 카운터보어 ⌴)는 폰트 tofu 문제라
벡터로 직접 그림.

라벨: 실도면의 기존 박스 좌표를 전부 유지(det은 좌표만 학습).
      갈아끼운 박스도 좌표는 원래 박스 그대로 — 조립단어를 그 박스에 맞춰 넣기 때문.

출력: data/generated/det_overlay_data/{imgs/, train_list.txt, val_list.txt}
"""

import json
import math
import os
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from fontTools.ttLib import TTFont

import numpy as np

from zone_utils import split_zone
from warp_mls import WarpMLS

random.seed(2025)

# ── 경로 ──────────────────────────────────────────────────────
REAL_LABEL = Path('data/real/train/Label.txt')
REAL_DIR = Path('data/real/train')
OUT_DIR = Path('data/generated/det_overlay_data')
IMG_OUT = OUT_DIR / 'imgs'

# ── 파라미터 ──────────────────────────────────────────────────
SWAP_RATIO = 0.4          # 한 이미지에서 갈아끼울 박스 비율
FADE_LEVEL = 0.95         # 원글자 흰색 페이드 강도(잔상 거의 안 보이게)
N_AUG_VARIANTS = 2        # 이미지당 증강본 개수(원본 1장 + 증강 N장)
RARE_SYMBOL_RATIO = 0.45  # 갈아끼울 때 희귀기호 넣을 확률(나머진 치수풍 숫자)
ROT_SIGN = -1             # PIL회전 부호(이미지좌표 y-down 보정 — 시각검증으로 확정)

# 증강본에 얹을 적층공차(공칭/상한/하한 별도박스) 그룹 수, 핀홀 하드네거티브 수
N_STACK_GROUPS = (3, 7)   # det이 상/하한을 별도 박스로 분리 학습하도록
N_PINHOLES = (2, 6)       # 구멍/중심마크 — 박스 없음(det가 텍스트로 오검출 안 하게)
WARP_PROB = 0.4           # 증강본에 TIA 공간왜곡(사진 렌즈/촬영 왜곡 모사) 적용 확률
# (회전은 학습중 config IaaAugment rotate:[-50,50]에 맡김 — 여기선 안 구움)

# ── 폰트 (글자별 랜덤) ────────────────────────────────────────
_CAND_FONTS = [
    r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\arialbd.ttf',
    r'C:\Windows\Fonts\ariali.ttf', r'C:\Windows\Fonts\times.ttf',
    r'C:\Windows\Fonts\timesbd.ttf', r'C:\Windows\Fonts\cour.ttf',
    r'C:\Windows\Fonts\consola.ttf', r'C:\Windows\Fonts\tahoma.ttf',
    r'C:\Windows\Fonts\calibri.ttf', r'C:\Windows\Fonts\verdana.ttf',
]
FONT_FILES = [f for f in _CAND_FONTS if os.path.exists(f)]

# 폰트별 글리프 커버리지 캐시 — 없는 글자는 그 폰트로 렌더 안 함
_cmap_cache = {}


def _cmap(font_path):
    if font_path not in _cmap_cache:
        try:
            _cmap_cache[font_path] = set(TTFont(font_path).getBestCmap().keys())
        except Exception:
            _cmap_cache[font_path] = set()
    return _cmap_cache[font_path]


def _font_for(ch):
    """해당 글자 글리프가 있는 폰트 중 랜덤. 없으면 None."""
    ok = [f for f in FONT_FILES if ord(ch) in _cmap(f)]
    return random.choice(ok) if ok else None


# ── 글자 하나 렌더 (baseline 정보 포함) ───────────────────────
def render_glyph(ch, px):
    """글자 하나 → (RGBA, baseline_y). 굵기·회전 지터 적용.
    baseline_y: 이 글리프 이미지 안에서 폰트 baseline의 y좌표(정렬 기준)."""
    fp = _font_for(ch)
    if fp is None:
        return None
    f = ImageFont.truetype(fp, px)
    ascent, descent = f.getmetrics()
    canvas_h = ascent + descent
    tmp = Image.new('L', (px * 2, canvas_h + px), 0)
    d = ImageDraw.Draw(tmp)
    d.text((px // 2, 0), ch, font=f, fill=255)  # (x, top=0) → baseline = ascent
    bbox = tmp.getbbox()
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    g = tmp.crop((x0, y0, x1, y1))
    baseline_in_crop = ascent - y0  # baseline y within cropped glyph

    r = random.random()
    if r < 0.22:
        g = g.filter(ImageFilter.MaxFilter(3))   # 굵게
    elif r < 0.4:
        g = g.filter(ImageFilter.MinFilter(3))   # 얇게

    ang = random.uniform(-4, 4)
    before_h = g.height
    g = g.rotate(ang, expand=True, resample=Image.BICUBIC)
    # 회전으로 이미지 높이 늘어난 만큼 baseline도 중심기준 보정
    baseline_in_crop += (g.height - before_h) / 2

    out = Image.new('RGBA', g.size, (0, 0, 0, 0))
    out.putalpha(g)
    return out, baseline_in_crop


# ── 조립단어 (글자별 왜곡 → baseline 정렬로 이어붙임) ──────────
def compose_word(text, target_h):
    px = max(14, int(target_h * random.uniform(0.9, 1.1)))
    glyphs = []  # (RGBA, baseline_y) or ('space', width)
    for ch in text:
        if ch == ' ':
            glyphs.append(('space', int(px * 0.35)))
            continue
        r = render_glyph(ch, px)
        if r is None:
            continue
        g, base = r
        s = random.uniform(0.88, 1.12)  # 글자별 스케일 지터
        nw, nh = max(1, int(g.width * s)), max(1, int(g.height * s))
        g = g.resize((nw, nh), Image.BICUBIC)
        base *= s
        glyphs.append((g, base))
    real = [(g, b) for g, b in ((x, y) for t in glyphs for x, y in [t]) if not isinstance(g, str)]
    real = [t for t in glyphs if t[0] != 'space']
    if not real:
        return None

    # baseline 정렬: 각 글자 baseline을 공통선에 맞추고, 위/아래로 지터
    max_above = max(b for _, b in real)                       # baseline 위 최대높이
    max_below = max(g.height - b for g, b in real)            # baseline 아래 최대깊이
    jitter_range = int(px * 0.1)
    pad = int(px * 0.4)
    baseline_y = pad + max_above + jitter_range

    total_w = pad * 2
    for t in glyphs:
        total_w += (t[1] if t[0] == 'space' else t[0].width + int(px * 0.12))
    canvas_h = int(baseline_y + max_below + jitter_range + pad)
    canvas = Image.new('RGBA', (total_w, canvas_h), (0, 0, 0, 0))

    x = pad
    for t in glyphs:
        if t[0] == 'space':
            x += t[1]
            continue
        g, base = t
        jit = random.randint(-jitter_range, jitter_range)
        y = int(baseline_y - base + jit)
        canvas.alpha_composite(g, (x, y))
        x += g.width + random.randint(int(px * 0.0), int(px * 0.16))
    bb = canvas.getbbox()
    return canvas.crop(bb) if bb else None


# ── 벡터 특수기호 (폰트 tofu 회피) ────────────────────────────
def _rgba_from_draw(size):
    im = Image.new('RGBA', size, (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def vec_circled_number(target_h):
    n = random.randint(1, 20)
    px = max(18, int(target_h))
    im, d = _rgba_from_draw((px + 6, px + 6))
    lw = max(1, px // 18)
    d.ellipse([2, 2, px + 2, px + 2], outline=(10, 10, 10, 255), width=lw)
    fp = _font_for(str(n)[0]) or FONT_FILES[0]
    f = ImageFont.truetype(fp, int(px * 0.6))
    tb = d.textbbox((0, 0), str(n), font=f)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    d.text(((px - tw) / 2 - tb[0] + 2, (px - th) / 2 - tb[1] + 2), str(n),
           font=f, fill=(10, 10, 10, 255))
    return im


def vec_roughness(target_h, kind):
    """kind: solo(√) / removal(▽) / no_removal(▽ₒ). ISO 1302 검증형태."""
    s = max(16, int(target_h))
    im, d = _rgba_from_draw((int(s * 1.6), int(s * 1.5)))
    lw = max(1, s // 16)
    bx, by = s * 0.3, s * 1.1
    p_short = (bx - s * 0.32, by - s * 0.5)
    p_long = (bx + s * 0.55, by - s * 1.0)
    d.line([bx, by, p_short[0], p_short[1]], fill=(10, 10, 10, 255), width=lw)
    d.line([bx, by, p_long[0], p_long[1]], fill=(10, 10, 10, 255), width=lw)
    if kind in ('removal', 'no_removal'):
        d.line([p_short[0], p_short[1], p_long[0], p_long[1]],
               fill=(10, 10, 10, 255), width=lw)
    if kind == 'no_removal':
        r = s * 0.2
        d.ellipse([bx - r, by - r, bx + r, by + r], outline=(10, 10, 10, 255), width=lw)
    bb = im.getbbox()
    return im.crop(bb) if bb else im


def vec_gdt_symbol(target_h):
    """⊥(직각) / ∥(평행) / ⏥(평면) 중 하나를 벡터로."""
    kind = random.choice(['perp', 'parallel', 'flat'])
    s = max(16, int(target_h))
    im, d = _rgba_from_draw((s + 6, s + 6))
    lw = max(1, s // 14)
    c = (10, 10, 10, 255)
    if kind == 'perp':
        d.line([s * 0.5, s * 0.1, s * 0.5, s * 0.9], fill=c, width=lw)
        d.line([s * 0.15, s * 0.9, s * 0.85, s * 0.9], fill=c, width=lw)
    elif kind == 'parallel':
        d.line([s * 0.35, s * 0.15, s * 0.55, s * 0.85], fill=c, width=lw)
        d.line([s * 0.6, s * 0.15, s * 0.8, s * 0.85], fill=c, width=lw)
    else:  # flat = 평행사변형
        d.polygon([(s * 0.2, s * 0.7), (s * 0.45, s * 0.3),
                   (s * 0.85, s * 0.3), (s * 0.6, s * 0.7)], outline=c, width=lw)
    bb = im.getbbox()
    return im.crop(bb) if bb else im


def vec_counterbore(target_h):
    """⌴ 브래킷 + øXX 숫자."""
    s = max(16, int(target_h))
    w, h = int(s * 0.55), int(s * 0.5)
    d_val = f'ø{random.choice([6,8,10,12,16,20])}'
    word = compose_word(d_val, int(target_h * 0.9))
    ww = word.width if word else 0
    im, d = _rgba_from_draw((w + 6 + ww + 6, max(h, word.height if word else h) + 6))
    lw = max(1, s // 16)
    c = (10, 10, 10, 255)
    d.line([2, 2, 2, h], fill=c, width=lw)
    d.line([2, 2, w, 2], fill=c, width=lw)
    d.line([w, 2, w, h], fill=c, width=lw)
    if word:
        im.alpha_composite(word, (w + 6, 2))
    bb = im.getbbox()
    return im.crop(bb) if bb else im


# ── 갈아끼울 내용 생성 ────────────────────────────────────────
def _rand_dim_text(aspect=1.0):
    """aspect = oriented_w / oriented_h (긴 박스일수록 긴 문자열 선택).
    박스를 대략 채워서 det이 헐렁한 박스를 학습하지 않도록 함."""
    short = [
        f'{random.randint(1, 99)}',
        f'ø{random.randint(3, 90)}',
        f'R{random.choice([1,2,3,5,8])}',
        f'M{random.choice([3,4,5,6,8,10,12])}',
    ]
    mid = [
        f'{random.randint(1, 200)}.{random.randint(0, 9)}',
        f'ø{random.randint(10,120)}',
        f'{random.randint(5, 150)}±0.{random.randint(1,3)}'[:6],
        f'{random.randint(2,12)}/{random.randint(2,16)}',
        f'{random.choice([1,2,3])}x45°',
    ]
    long = [
        f'{random.randint(5, 150)}±0.{random.randint(1,3)}',
        f'ø{random.randint(10,90)}{random.choice(["H7","g6","f7","h6","k6","js6"])}',
        f'{random.randint(10,300)}.{random.randint(0,9)}±0.{random.randint(1,5)}',
        f'ø{random.randint(10,90)} {random.choice(["H7","g6","f7"])}',
    ]
    # 세로텍스트(aspect<1)는 회전 후 세로로 길어지므로 aspect를 뒤집어 길이판단
    a = aspect if aspect >= 1 else 1 / max(aspect, 1e-3)
    if a >= 3.2:
        return random.choice(long)
    if a >= 1.8:
        return random.choice(mid)
    return random.choice(short)


def make_content(target_h, aspect=1.0):
    """(RGBA, kind). kind는 로깅용. aspect=oriented_w/oriented_h."""
    if random.random() < RARE_SYMBOL_RATIO:
        pick = random.choices(
            ['circled', 'roughness', 'gdt', 'counterbore'],
            weights=[0.3, 0.35, 0.2, 0.15])[0]
        if pick == 'circled':
            return vec_circled_number(target_h), 'circled'
        if pick == 'roughness':
            k = random.choices(['solo', 'removal', 'no_removal'],
                               weights=[0.3, 0.5, 0.2])[0]
            img = vec_roughness(target_h, k)
            if random.random() < 0.5:  # Ra값 붙이기
                ra = compose_word(f'Ra{random.choice([0.8,1.6,3.2,6.3])}',
                                  int(target_h * 0.8))
                if ra:
                    merged = Image.new('RGBA', (img.width + 4 + ra.width,
                                                max(img.height, ra.height)), (0, 0, 0, 0))
                    merged.alpha_composite(img, (0, 0))
                    merged.alpha_composite(ra, (img.width + 4,
                                                (merged.height - ra.height) // 2))
                    img = merged
            return img, 'roughness'
        if pick == 'gdt':
            return vec_gdt_symbol(target_h), 'gdt'
        return vec_counterbore(target_h), 'counterbore'
    w = compose_word(_rand_dim_text(aspect), target_h)
    return w, 'dim'


# ── quad 기하 ─────────────────────────────────────────────────
def quad_geom(pts):
    """[TL,TR,BR,BL] → (cx,cy, oriented_w, oriented_h, angle_deg).
    4점 아닌 폴리곤은 axis-aligned bbox(각도0)로 폴백."""
    if len(pts) != 4:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.0
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts
    w = (math.hypot(x1 - x0, y1 - y0) + math.hypot(x2 - x3, y2 - y3)) / 2
    h = (math.hypot(x3 - x0, y3 - y0) + math.hypot(x2 - x1, y2 - y1)) / 2
    ang = math.degrees(math.atan2(y1 - y0, x1 - x0))  # top edge 방향(이미지좌표)
    cx = (x0 + x1 + x2 + x3) / 4
    cy = (y0 + y1 + y2 + y3) / 4
    return cx, cy, w, h, ang


def fade_polygon(img, pts, level):
    """quad 다각형 내부만 흰색쪽으로 level 블렌드(잔상 남김)."""
    mask = Image.new('L', img.size, 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in pts], fill=int(255 * level))
    white = Image.new('RGB', img.size, (255, 255, 255))
    img.paste(Image.composite(white, img, mask), (0, 0))


def overlay_content(img, pts, content):
    """조립단어/기호를 quad중심에 각도 맞춰 multiply 블렌드.
    반환: 실제 그려진 글자의 tight oriented quad([[x,y]*4]) 또는 None.
    (원본 박스는 새 내용보다 클 수 있으므로, 라벨을 이 tight quad로 갱신해야
    det이 헐렁한 박스를 학습하지 않음)."""
    cx, cy, ow, oh, ang = quad_geom(pts)
    if ow < 4 or oh < 4 or content is None:
        return None
    # 가로쓰기 기준으로 만든 content를 oriented 박스 안에 맞춤(aspect 유지, 약간 여백)
    scale = min(ow / content.width, oh / content.height) * 0.92
    nw, nh = max(1, int(content.width * scale)), max(1, int(content.height * scale))
    c = content.resize((nw, nh), Image.BICUBIC)
    c = c.rotate(ROT_SIGN * ang, expand=True, resample=Image.BICUBIC)
    px = int(cx - c.width / 2)
    py = int(cy - c.height / 2)
    # multiply 블렌드: 알파 있는 곳만 어둡게(검은 stroke)
    x0, y0 = max(0, px), max(0, py)
    x1, y1 = min(img.width, px + c.width), min(img.height, py + c.height)
    if x1 <= x0 or y1 <= y0:
        return None
    region = img.crop((x0, y0, x1, y1)).convert('RGB')
    cc = c.crop((x0 - px, y0 - py, x1 - px, y1 - py))
    _, _, _, a = cc.split()
    black = Image.new('RGB', region.size, (12, 12, 12))
    region = Image.composite(black, region, a)
    img.paste(region, (x0, y0))

    # 실제 그려진 글자(nw×nh)를 원본 박스 축(ang)에 정렬한 tight oriented quad
    ar = math.radians(ang)
    ux, uy = math.cos(ar), math.sin(ar)          # 박스 가로축
    vx, vy = -math.sin(ar), math.cos(ar)         # 박스 세로축
    hw, hh = nw / 2, nh / 2
    corners = []
    for sx, sy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        corners.append([round(cx + sx * ux + sy * vx, 1),
                        round(cy + sx * uy + sy * vy, 1)])
    return corners


# ── 적층공차(3분리) + 핀홀(하드네거티브) 얹기 ──────────────────
def _rand_nominal():
    dia = random.random() < 0.5
    v = random.choice([random.randint(5, 200), round(random.uniform(5, 200), 1)])
    fit = random.choice(['', '', 'H7', 'g6', 'f7', 'h6'])
    return f'{"ø" if dia else ""}{v}{fit}'


def _rand_tol():
    return random.choice(['+0.05', '+0.1', '+0.02', '+0.15', '+0.2',
                          '-0.05', '-0.1', '-0.02', '-0.15', '0'])


def _paste_multiply(img, word_rgba, x, y):
    """조립단어를 (x,y)에 multiply 블렌드. 반환: 실제 박스 (x0,y0,x1,y1) 또는 None."""
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


def _box_item(bb, text):
    x0, y0, x1, y1 = bb
    return {'transcription': text,
            'points': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            'difficult': False}


def place_stacked(img, x, y, base_h):
    """공칭+상한+하한을 '촘촘하되 분리된' 별도 박스 3개로 배치.
    반환: [box_item, ...] (라벨에 추가할 것). 실패시 부분/빈 리스트."""
    vgap = max(3, int(base_h * 0.2))
    hgap = max(3, int(base_h * 0.25))
    out = []
    nominal = compose_word(_rand_nominal(), base_h)
    if nominal is None:
        return out
    nb = _paste_multiply(img, nominal, x, y)
    if nb is None:
        return out
    out.append(_box_item(nb, 'nominal'))

    small_h = max(12, int(base_h * 0.7))
    up = compose_word(_rand_tol(), small_h)
    lo = compose_word(_rand_tol(), small_h)
    tx = nb[2] + hgap
    ctr = (nb[1] + nb[3]) // 2
    if up is not None:
        ub = _paste_multiply(img, up, tx, ctr - vgap - up.height)
        if ub:
            out.append(_box_item(ub, 'upper'))
    if lo is not None:
        lb = _paste_multiply(img, lo, tx, ctr + vgap)
        if lb:
            out.append(_box_item(lb, 'lower'))
    return out


def draw_pinhole(img, cx, cy, r):
    """중심마크/동심원/구멍단면 그래픽 — 박스 없음(하드네거티브)."""
    d = ImageDraw.Draw(img)
    kind = random.choice(['centermark', 'concentric', 'hole_section'])
    c = (20, 20, 20)
    if kind == 'centermark':
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=c, width=2)
        ext = int(r * 1.6)
        d.line([cx - ext, cy, cx + ext, cy], fill=c, width=1)
        d.line([cx, cy - ext, cx, cy + ext], fill=c, width=1)
    elif kind == 'concentric':
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=c, width=2)
        r2 = int(r * 0.6)
        d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], outline=c, width=2)
        d.line([cx - int(r * 1.4), cy, cx + int(r * 1.4), cy], fill=c, width=1)
        d.line([cx, cy - int(r * 1.4), cx, cy + int(r * 1.4)], fill=c, width=1)
    else:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=c, width=2)
        for off in range(-r, r, 5):
            d.line([cx + off, cy - r, cx + off + r, cy + r], fill=c, width=1)


def _rects_overlap(a, b, margin=10):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + margin < bx0 or bx1 + margin < ax0
                or ay1 + margin < by0 or by1 + margin < ay0)


def _aabb(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


# ── TIA 공간왜곡 (사진 렌즈/촬영 왜곡 모사) + 박스 동시 이동 ────
def _mls_affine_points(query, src_pts, dst_pts):
    """query 점들을 (src_pts→dst_pts) 대응 기반 forward affine-MLS로 이동
    (WarpMLS 내부와 동일 수식)."""
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    out = []
    for v in np.asarray(query, dtype=np.float64):
        d2 = np.sum((src - v) ** 2, axis=1)
        hit = np.where(d2 < 1e-8)[0]
        if len(hit):
            out.append([float(dst[hit[0]][0]), float(dst[hit[0]][1])])
            continue
        w = 1.0 / d2
        pstar = (w[:, None] * src).sum(0) / w.sum()
        qstar = (w[:, None] * dst).sum(0) / w.sum()
        phat, qhat = src - pstar, dst - qstar
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))
        for i in range(len(src)):
            A += w[i] * np.outer(phat[i], phat[i])
            B += w[i] * np.outer(phat[i], qhat[i])
        try:
            fv = (v - pstar) @ np.linalg.inv(A) @ B + qstar
        except np.linalg.LinAlgError:
            fv = v - pstar + qstar
        out.append([float(fv[0]), float(fv[1])])
    return out


def warp_image(img, box_items, segment=4, thresh_ratio=0.03):
    """이미지에 TIA 공간왜곡, 박스 4점도 동일 MLS로 이동. 반환 (img, box_items)."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    cut = w // segment
    thresh = max(2, int(min(h, w) * thresh_ratio))
    ri = lambda: random.randint(0, thresh - 1)
    src_pts = [[0, 0], [w, 0], [w, h], [0, h]]
    dst_pts = [[ri(), ri()], [w - ri(), ri()], [w - ri(), h - ri()], [ri(), h - ri()]]
    for c in range(1, segment):
        src_pts += [[cut * c, 0], [cut * c, h]]
        dst_pts += [[cut * c + ri() - thresh // 2, ri()],
                    [cut * c + ri() - thresh // 2, h - ri()]]
    out = WarpMLS(arr, src_pts, dst_pts, w, h).generate()
    new_items = []
    for it in box_items:
        moved = _mls_affine_points(it['points'], src_pts, dst_pts)
        new_items.append({'transcription': it.get('transcription', ''),
                          'points': moved, 'difficult': False})
    return Image.fromarray(out.astype(np.uint8)), new_items


# ── 이미지 1장 증강 ──────────────────────────────────────────
def augment_image(img, items):
    """items의 40%를 갈아끼우고, 빈 공간에 적층공차(3분리)+핀홀(박스없음)을 얹음.
    반환: (증강이미지, 전체_라벨_아이템_리스트).
    라벨 = 기존 박스(갈아끼운 것은 실제 그려진 글자 tight quad로 갱신) + 적층공차 박스.
    핀홀은 박스 없음. → main은 이 리스트를 그대로 라벨로 쓰면 됨."""
    out = img.copy()
    W, H = out.size

    # 1) 기존 박스 40% 갈아끼우기 (갈아끼운 것은 라벨을 실제 글자범위로 갱신)
    idxs = list(range(len(items)))
    random.shuffle(idxs)
    swap_set = set(idxs[:int(len(items) * SWAP_RATIO)])
    label_items = []   # 최종 라벨(갈아끼운 박스는 실제 그려진 tight quad로 갱신)
    for i, it in enumerate(items):
        base = {'transcription': it.get('transcription', ''),
                'points': it['points'], 'difficult': False}
        if i not in swap_set:
            label_items.append(base)
            continue
        pts = it['points']
        _, _, ow, oh, _ = quad_geom(pts)
        target_h = min(ow, oh)
        if target_h < 8:
            label_items.append(base)
            continue
        aspect = ow / oh if oh > 1e-3 else 1.0
        content, _kind = make_content(int(target_h), aspect)
        if content is None:
            label_items.append(base)
            continue
        fade_polygon(out, pts, FADE_LEVEL)
        tight = overlay_content(out, pts, content)
        # 새 내용이 원본박스보다 작을 수 있으므로 라벨을 실제 글자범위로 갱신
        label_items.append({'transcription': it.get('transcription', ''),
                            'points': tight if tight else it['points'],
                            'difficult': False})

    # 겹침회피용 점유영역: 갱신된 라벨 기준
    placed = [_aabb(it['points']) for it in label_items]
    extra = []

    # 2) 적층공차 3분리 씬
    n_stack = random.randint(*N_STACK_GROUPS)
    tries = 0
    while len(extra) < n_stack * 3 and tries < n_stack * 30:
        tries += 1
        base_h = random.randint(max(12, int(H * 0.012)), max(16, int(H * 0.025)))
        x = random.randint(int(W * 0.05), int(W * 0.8))
        y = random.randint(int(H * 0.05), int(H * 0.85))
        probe = (x, y, x + base_h * 5, y + base_h * 2)
        if any(_rects_overlap(probe, p) for p in placed):
            continue
        boxes = place_stacked(out, x, y, base_h)
        for it in boxes:
            placed.append(_aabb(it['points']))
        extra.extend(boxes)

    # 3) 핀홀 하드네거티브 (박스 없음)
    n_pin = random.randint(*N_PINHOLES)
    tries = 0
    done = 0
    while done < n_pin and tries < n_pin * 30:
        tries += 1
        r = random.randint(max(8, int(H * 0.01)), max(12, int(H * 0.025)))
        cx = random.randint(int(W * 0.05), int(W * 0.9))
        cy = random.randint(int(H * 0.05), int(H * 0.9))
        probe = (cx - int(r * 1.6), cy - int(r * 1.6), cx + int(r * 1.6), cy + int(r * 1.6))
        if any(_rects_overlap(probe, p) for p in placed):
            continue
        draw_pinhole(out, cx, cy, r)
        placed.append(probe)
        done += 1

    return out, label_items + extra


def clean_items(items):
    return [{'transcription': it.get('transcription', ''),
             'points': it['points'], 'difficult': False} for it in items]


# ── 실행 ─────────────────────────────────────────────────────
def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    IMG_OUT.mkdir(parents=True, exist_ok=True)

    entries = []  # (img_name, items)
    n_real = 0
    with open(REAL_LABEL, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            path, js = line.split('\t', 1)
            raw_items = json.loads(js)
            # zone 태그 박스 자체는 학습/갈아끼우기에서 제외(도면영역 마커일 뿐).
            # drawing_items+metadata_items = zone박스 뺀 실제 텍스트 박스 전부.
            _zone, drawing_items, metadata_items = split_zone(raw_items)
            items = drawing_items + metadata_items
            src = REAL_DIR / Path(path).name
            if not src.exists() or not items:
                continue
            n_real += 1
            base_img = Image.open(src).convert('RGB')

            # 원본 1장 (실글자 그대로)
            name = f'{src.stem}_orig{src.suffix}'
            base_img.save(IMG_OUT / name)
            entries.append((name, clean_items(items)))

            # 증강 N장 (40% 갈아끼움 + 적층공차 3분리 + 핀홀)
            # 라벨 = 기존 실박스 전부 + 새로 얹은 적층공차 박스(핀홀은 박스 없음)
            # + 확률적 TIA 공간왜곡(사진 촬영 왜곡 강건성). 회전은 학습중 config에 맡김.
            for v in range(N_AUG_VARIANTS):
                aug, box_items = augment_image(base_img, items)
                if random.random() < WARP_PROB:
                    aug, box_items = warp_image(aug, box_items)
                aname = f'{src.stem}_aug{v}{src.suffix}'
                aug.save(IMG_OUT / aname)
                entries.append((aname, box_items))

    random.shuffle(entries)
    n_val = max(1, int(len(entries) * 0.1))
    val, train = entries[:n_val], entries[n_val:]

    def write(split, rows):
        with open(OUT_DIR / f'{split}_list.txt', 'w', encoding='utf-8') as f:
            for name, items in rows:
                f.write(f'imgs/{name}\t{json.dumps(items, ensure_ascii=False)}\n')

    write('train', train)
    write('val', val)
    print(f'실도면 {n_real}장 -> 출력 {len(entries)}장 (원본+증강 {1+N_AUG_VARIANTS}배)')
    print(f'train {len(train)} / val {len(val)}')
    print(f'출력: {OUT_DIR}')


if __name__ == '__main__':
    main()
