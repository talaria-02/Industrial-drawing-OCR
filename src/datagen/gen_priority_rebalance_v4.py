"""
실측 기반 우선순위 재조정 (v4) — 순수 숫자 제외, 기호결합 패턴 집중
======================================================================
사용자 실측 피드백 반영:
  - 순수 숫자(예: "35","50")는 baseline이 이미 거의 100% -> 학습가치 없음, 제외
  - ①~⑳ 원문자 전체 필요, 근데 실제론 작은 숫자(①②③)가 훨씬 흔함
    -> 균등분포 아니라 작은 값에 가중치
  - ▽(표면거칠기) 노출 부족
  - "20°"류 각도표시는 실제로 대부분 삐딱하게(회전된 채로) 쓰여있는데
    이 조합(각도표시+회전) 비율이 거의 없음
  - "ø58.00 +0.30" 같은 기호+숫자+공차 결합 부족
  - "50± 0,1" 플마 패턴은 실제로 제일 흔한데 비중 낮음

비중(총 샘플 기준): ± 30% / ø+공차 20% / 각도(회전) 20% / 거칠기 20% / 원문자 10%

출력: data/generated/priority_v4_preview/preview.png
"""

import math
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

random.seed(707)

OUT_DIR = Path('data/generated/priority_v4_preview')
IMG_DIR = OUT_DIR / 'images'

FONT_PATHS = [r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\calibri.ttf']
SEGUISYM = r'C:\Windows\Fonts\seguisym.ttf'
SEGUISYM_ONLY = set('⏤⏥⌭∥⊥▽⌀∅Ⓐ↧') | {chr(0x2460 + i) for i in range(20)}  # ①..⑳ 포함
# Φ/φ/Ø/ø는 사람 눈엔 사실상 동일 모양(확인함) — 인코딩만 다른 중복이라
# ▽/∇ 케이스처럼 학습데이터로 구분시킬 이유 없음. ø 하나로 통일.
DIAM = ['ø']
CIRCLED = [chr(0x2460 + i) for i in range(20)]  # ①..⑳
# 작은 숫자일수록 실제 등장빈도 높음(BOM 항목 대부분 5개 이내) -> 역수 가중치
CIRCLED_WEIGHTS = [1 / (i + 1) for i in range(20)]


def get_font(size, text=''):
    if any(ch in SEGUISYM_ONLY for ch in text):
        return ImageFont.truetype(SEGUISYM, size)
    return ImageFont.truetype(random.choice(FONT_PATHS), size)


def rand_tol_frac():
    sep = random.choice(['.', ','])
    val = random.choice(['05', '1', '15', '2'])
    return f'0{sep}{val}'


def gen_plusminus():
    """±패턴 — 실제로 제일 흔한 형태. 공백유무/붙여쓰기 다양화."""
    whole = random.randint(1, 200)
    dec = random.choice(['', f'{random.choice([".", ","])}{random.choice(["5","25","75"])}'])
    nominal = f'{whole}{dec}'
    sep = random.choice([' ± ', '± ', '±'])
    return f'{nominal}{sep}{rand_tol_frac()}'


def gen_diam_tolerance():
    """ø+숫자+공차 결합 — 괄호 없이 상한/하한 기호로 바로 붙는 형태
    ("ø58.00 +0.30" 스타일). 별도 crop 원칙(줄별분리)에 따라 nominal과
    tolerance를 별도 텍스트로 반환(한 crop에 몰아넣지 않음)."""
    sym = random.choice(DIAM)
    whole = random.randint(3, 120)
    dec = random.choice(['00', '5', '25'])
    nominal = f'{sym}{whole}.{dec}'
    upper = f'+{rand_tol_frac()}'
    lower = random.choice(['-0.00', f'-{rand_tol_frac()}'])
    return nominal, upper, lower


def gen_angle():
    """각도표시 — 실제로 대부분 삐딱하게(각도 자체와 비슷한 기울기로) 쓰임.
    회전각을 0~360 넓은 범위로 잡아서 각도표시+회전 조합 노출을 늘림."""
    val = random.choice([random.randint(1, 90), round(random.uniform(1, 90), 1)])
    text = f'{val}°'
    tilt = random.uniform(-75, 75)  # 넓은 범위 — 실제 각도선 따라 다양하게 기울어짐
    return text, tilt


def gen_roughness():
    kind = random.choice(['solo', 'double', 'triple', 'checkra', 'ra_space', 'ra_tight'])
    val = random.choice(['0.8', '1.6', '3.2', '6.3', '12.5', '25'])
    if kind == 'solo':
        return '▽'
    if kind == 'double':
        return '▽▽'
    if kind == 'triple':
        return '▽▽▽'
    if kind == 'checkra':
        return f'√Ra{val}'
    if kind == 'ra_space':
        return f'√ Ra {val}'
    return f'▽Ra{val}'


def gen_circled():
    return random.choices(CIRCLED, weights=CIRCLED_WEIGHTS, k=1)[0]


def render_crop(text, size=None):
    size = size or random.randint(28, 40)
    font = get_font(size, text)
    tmp = Image.new('RGB', (10, 10), (255, 255, 255))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0] + 24, bbox[3] - bbox[1] + 24
    img = Image.new('RGB', (max(w, 20), max(h, 20)), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((12 - bbox[0], 12 - bbox[1]), text, font=font, fill=(0, 0, 0))
    return img


def main(n=30):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    labels, rows = [], []
    counts = {'plusminus': 0, 'diam_tol': 0, 'angle': 0, 'rough': 0, 'circled': 0}

    for i in range(n):
        r = random.random()
        if r < 0.30:
            text = gen_plusminus()
            img = render_crop(text)
            counts['plusminus'] += 1
            entries = [(text, img)]
        elif r < 0.50:
            nominal, upper, lower = gen_diam_tolerance()
            entries = [(nominal, render_crop(nominal)),
                       (upper, render_crop(upper, size=random.randint(18, 26))),
                       (lower, render_crop(lower, size=random.randint(18, 26)))]
            counts['diam_tol'] += 1
        elif r < 0.70:
            text, tilt = gen_angle()
            img = render_crop(text)
            img = img.rotate(tilt, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
            counts['angle'] += 1
            entries = [(f'{text} (tilt={tilt:.0f})', img)]
        elif r < 0.90:
            text = gen_roughness()
            img = render_crop(text)
            counts['rough'] += 1
            entries = [(text, img)]
        else:
            text = gen_circled()
            img = render_crop(text)
            counts['circled'] += 1
            entries = [(text, img)]

        for text, img in entries:
            name = f'v4_{len(rows):03d}.png'
            img.save(IMG_DIR / name)
            labels.append(f'{IMG_DIR / name}\t{text}')
            rows.append((img, text))

    with open(OUT_DIR / 'rec_gt.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels) + '\n')

    row_h = 100
    preview = Image.new('RGB', (500, row_h * len(rows)), (255, 255, 255))
    pd = ImageDraw.Draw(preview)
    label_font = ImageFont.truetype(r'C:\Windows\Fonts\malgun.ttf', 15)
    for i, (img, text) in enumerate(rows):
        y = i * row_h
        thumb = img.copy()
        thumb.thumbnail((260, 80))
        preview.paste(thumb, (10, y + 10))
        pd.text((280, y + 35), f"#{i:02d} '{text}'", font=label_font, fill=(0, 0, 150))
        pd.line([(0, y + row_h), (preview.width, y + row_h)], fill=(220, 220, 220))
    preview.save(OUT_DIR / 'preview.png')
    print(f'생성 완료: {len(rows)}장, 분포={counts}')


if __name__ == '__main__':
    main()
