"""
실라벨 이미지 기반 증강 프리뷰 (text-swap + 회전 + 포토메트릭)
================================================================
1번 증강(실라벨 15장에 증강 적용) — ###ZONE:DRAWING### 태그로 구분된
도면영역(drawing_items)에서만 text-swap 대상 뽑음. 메타데이터(타이틀
블록)는 이미 인식 잘 되는 영역이라 증강 대상에서 제외([[zone_utils]]).

이번 실행: 15장 중 2장 랜덤으로 뽑아 시험 — 스케일업 전 검증용.

text-swap: 원본 crop에서 글자 지우고(흰색 채움) 같은 카테고리의
새 텍스트를 비슷한 크기로 다시 그림 — 실제 배경/레이아웃은 유지한
채 내용만 다양화(예전에 합의한 SynthText 스타일 기법).
원근증강은 제외(사용자 확인: 실제 각도사진 없음 -> 근거없는 왜곡).

출력: finetune/real_augment_preview/preview.png
"""

import json
import random
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

from zone_utils import split_zone
from gen_line_noise_rec import draw_through_line, draw_stub

random.seed(505)

LABEL_FILE = Path('finetune/labeling/train/Label.txt')
IMG_DIR = Path('finetune/labeling/train')
OUT_DIR = Path('finetune/real_augment_preview')

FONT_PATHS = [r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\calibri.ttf']
SEGUISYM = r'C:\Windows\Fonts\seguisym.ttf'
SEGUISYM_ONLY = set('⏤⏥⌭∥⊥▽⌀∅①②③④⑤⑥Ⓐ↧')
DATUM_LETTERS = list('ABCDEF')
CIRCLED = list('①②③④⑤⑥')


def get_font(size, text=''):
    if any(ch in SEGUISYM_ONLY for ch in text):
        return ImageFont.truetype(SEGUISYM, size)
    return ImageFont.truetype(random.choice(FONT_PATHS), size)


# (판별 정규식, swap 함수) 순서대로 검사 — 먼저 매치되는 카테고리로 처리.
# 매치 안 되면 스킵(제네릭 영어 텍스트/타이틀블록 문구 등은 대상 아님).
def _rand_num(lo=1, hi=200):
    return str(random.randint(lo, hi))


def _rand_decimal_like(sample):
    """sample과 같은 소수부 자릿수를 유지하며 완전히 새로운 소수값 생성
    (원본 소수부를 그대로 남기고 정수부만 바꾸는 건 비현실적인 값이 나와서 버그)."""
    m = re.search(r'(\d+)([.,])(\d+)', sample)
    whole, sep, frac = m.groups()
    new_whole = str(random.randint(1, 200))
    new_frac = ''.join(random.choice('0123456789') for _ in frac)
    return f'{new_whole}{sep}{new_frac}'


def _swap_tolerance(t):
    """공차(±) 표기 — 공칭값(왼쪽)과 공차량(오른쪽)을 따로 생성.
    공차량은 실제 도면 관례상 작은 값(0.05~0.2대)이어야 함 — 왼쪽이랑
    같은 방식으로 1~99 통크게 새로 생성하면 '±90' 같은 비현실적 값 나옴."""
    idx = t.index('±')
    left, right = t[:idx], t[idx + 1:]

    if re.search(r'[.,]', left):
        new_left = _rand_decimal_like(left)
    else:
        new_left = re.sub(r'\d+', lambda m: _rand_num(1, 200), left, count=1)

    tol_sep = ',' if ',' in right else '.'
    tol_digits = random.choice(['05', '1', '15', '2'])
    new_right = re.sub(r'\d+[.,]?\d*', f'0{tol_sep}{tol_digits}', right, count=1)
    return f'{new_left}±{new_right}'


def _normalize_diam(t):
    """Φ/φ/Ø를 ø로 통일 — 사람 눈엔 사실상 동일 모양이라(확인함) 구분해서
    학습시킬 이유 없음. 매칭 자체는 원본에 실제 있을 수 있는 이 4개 다
    넓게 잡되(실라벨 원문 인식용), swap 출력은 항상 ø로 정규화."""
    return re.sub(r'[ØΦφ]', 'ø', t)


CATEGORIES = [
    (re.compile(r'^\d+x[øØΦφ]\d+$|^\d+-[øØΦφ]\d+$'),
     lambda t: re.sub(r'\d+', lambda m: _rand_num(2, 30), _normalize_diam(t), count=2)),
    (re.compile(r'^[øØΦφ]\d+([.,]\d+)?( H\d)?$'),
     lambda t: re.sub(r'\d+([.,]\d+)?', lambda m: _rand_num(1, 200), _normalize_diam(t), count=1)),
    (re.compile(r'^R\d+( REF)?$'),
     lambda t: f'R{_rand_num(1, 99)}' + (' REF' if t.endswith('REF') else '')),
    (re.compile(r'^\d+ REF$'), lambda t: f'{_rand_num(1, 99)} REF'),
    (re.compile(r'^M\d+(x[\d.]+)?$'),
     lambda t: re.sub(r'^M\d+', f'M{random.choice([3,4,5,6,8,10,12])}', t)),
    (re.compile(r'±'), _swap_tolerance),
    (re.compile(r'^\d+[.,]\d+$'), _rand_decimal_like),
    # 단순 정수만 있는 패턴은 실측 baseline이 이미 거의 100%라 증강 대상에서 제외
    # (사용자 확인: "단순 숫자만 있는건 학습할 이유가 없어"). 기호 붙은 패턴들에
    # 슬롯을 몰아주려고 의도적으로 뺌 — 위쪽 카테고리(ø/R/M/±/데시멀 등)에서
    # 이미 잡히지 않은 순수 정수(예: '35', '50')는 스왑 대상에서 제외됨.
    (re.compile(r'^\d+/\d+$'), lambda t: f'{random.randint(1,63)}/{random.choice([4,8,16,32,64])}'),
    (re.compile(r'^\d+°$|^\d+\.\d+°$'), lambda t: f'{_rand_num(1, 90)}°'),
    (re.compile(r'^\d+:\d+$'), lambda t: f'{random.randint(1,5)}:{random.randint(1,5)}'),
    (re.compile(r'^[A-F]$'), lambda t: random.choice(DATUM_LETTERS)),
    (re.compile(r'^[①②③④⑤⑥]$'), lambda t: random.choice(CIRCLED)),
    (re.compile(r'^CH \d+$'), lambda t: f'CH {_rand_num(1, 20)}'),
    (re.compile(r'^\(\d+[.,]?\d*\)$'), lambda t: f'({_rand_num(1, 200)})'),
]


def swap_text(original):
    for pat, fn in CATEGORIES:
        if pat.search(original):
            try:
                return fn(original)
            except Exception:
                return original
    return None  # 매치 카테고리 없음 -> 스킵 대상


def add_adjacent_fragment(draw, w, h):
    """det bbox가 살짝 넓게 잡혀서 옆 박스 글자 일부가 딸려온 상황 흉내.
    crop 가장자리에 랜덤 문자를 대부분 밖으로 나가게 배치해 일부만 보이게 함
    (제아무리 det이 잘해도 실제로 이런 크롭 오차는 생김 — 사용자 확인)."""
    frag_char = random.choice('0123456789ABCDXYZ.,±ø')
    size = random.randint(int(h * 0.5), max(int(h * 0.9), int(h * 0.5) + 1))
    font = get_font(size, frag_char)
    side = random.choice(['left', 'right'])
    if side == 'left':
        x = random.randint(-size, -int(size * 0.25))
    else:
        x = random.randint(w - int(size * 0.75), w - int(size * 0.25))
    y = random.randint(0, max(0, h - size))
    draw.text((x, y), frag_char, font=font, fill=(0, 0, 0))


def erase_and_redraw(crop, new_text, add_noise=True):
    """crop 전체를 배경색(흰색)으로 지우고 같은 자리에 새 텍스트를 원본과 비슷한
    크기로 다시 그림. 실제 도면은 흰 배경이라 단순 흰색 채움으로 지움(1차 시도).

    add_noise=True면 확률적으로 미세 선분(치수선/중심선 잔재)이나 인접박스
    문자 파편을 같이 그려서 det crop이 완벽하지 않은 실제 상황에 대한
    robust를 강화함 — 대부분은 깨끗하게 두고 일부만 노이즈 추가."""
    w, h = crop.size
    out = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(out)

    lo, hi = 6, max(h, 10)
    best_size = lo
    for _ in range(12):
        mid = (lo + hi) // 2
        font = get_font(mid, new_text)
        bbox = draw.textbbox((0, 0), new_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= w * 0.95 and th <= h * 0.95:
            best_size = mid
            lo = mid + 1
        else:
            hi = mid - 1
    font = get_font(best_size, new_text)
    bbox = draw.textbbox((0, 0), new_text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) / 2 - bbox[0], (h - th) / 2 - bbox[1]), new_text, font=font, fill=(0, 0, 0))

    if add_noise:
        r = random.random()
        if r < 0.2:
            draw_through_line(draw, w, h)
        elif r < 0.35:
            y_mid = h / 2
            draw_stub(draw, 4, y_mid, 'before')
        elif r < 0.5:
            y_mid = h / 2
            draw_stub(draw, w - 4, y_mid, 'after')
        if random.random() < 0.2:
            add_adjacent_fragment(draw, w, h)

    return out


def apply_rotation_photometric(img):
    """회전(스캔기울기~완전회전 섞음) + 포토메트릭(밝기/대비/블러) 적용."""
    angle = random.choices([None, 90, 270], weights=[0.4, 0.3, 0.3])[0]
    if angle is None:
        angle = random.uniform(-6, 6)
    rotated = img.rotate(angle, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC)

    if random.random() < 0.5:
        rotated = rotated.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    rotated = ImageEnhance.Brightness(rotated).enhance(random.uniform(0.85, 1.15))
    rotated = ImageEnhance.Contrast(rotated).enhance(random.uniform(0.85, 1.2))
    return rotated, angle


def get_bbox(points, pad=4):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(0, min(xs) - pad), max(0, min(ys) - pad),
            max(xs) + pad, max(ys) + pad)


def main(target_total=50):
    OUT_DIR.mkdir(exist_ok=True)

    with open(LABEL_FILE, encoding='utf-8') as f:
        lines = f.readlines()

    pool = []  # (path, item) — 전체 15장 통틀어서 뽑음, 특정 이미지에 몰리지 않게
    for line in lines:
        path, js = line.split('\t', 1)
        items = json.loads(js)
        zone_bbox, drawing_items, _ = split_zone(items)
        if zone_bbox is None:
            continue
        for it in drawing_items:
            if swap_text(it['transcription']) is not None:
                pool.append((path, it))

    picked = random.sample(pool, min(target_total, len(pool)))
    print(f'{len(pool)}개 후보 중 {len(picked)}개 뽑음 (이미지 {len(set(p for p,_ in picked))}종)')

    img_cache = {}
    rows = []
    for path, item in picked:
        fname = Path(path).name
        if fname not in img_cache:
            img_cache[fname] = Image.open(IMG_DIR / fname).convert('RGB')
        img = img_cache[fname]

        target = item['transcription']
        box = get_bbox(item['points'])
        orig_crop = img.crop(box)
        new_text = swap_text(target)
        swapped = erase_and_redraw(orig_crop, new_text)
        augmented, angle = apply_rotation_photometric(swapped)
        rows.append((fname, orig_crop, target, swapped, new_text, augmented, angle))

    row_h = 160
    preview = Image.new('RGB', (800, row_h * len(rows)), (255, 255, 255))
    pd = ImageDraw.Draw(preview)
    label_font = ImageFont.truetype(r'C:\Windows\Fonts\malgun.ttf', 14)

    for i, (fname, orig, otext, swapped, ntext, aug, angle) in enumerate(rows):
        y = i * row_h
        preview.paste(orig, (10, y + 10))
        preview.paste(swapped, (150, y + 10))
        aug_thumb = aug.copy()
        aug_thumb.thumbnail((260, 120))
        preview.paste(aug_thumb, (450, y + 10))
        pd.text((10, y + row_h - 25),
                 f"#{i:02d} [{fname}] 원본='{otext}' -> swap='{ntext}'  angle={angle:.1f}",
                 font=label_font, fill=(0, 0, 150))
        pd.line([(0, y + row_h), (preview.width, y + row_h)], fill=(220, 220, 220))

    preview.save(OUT_DIR / 'preview.png')
    print(f'{len(rows)}개 생성 -> {OUT_DIR / "preview.png"}')


if __name__ == '__main__':
    main()
