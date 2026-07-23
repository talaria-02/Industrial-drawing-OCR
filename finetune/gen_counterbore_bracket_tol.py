"""
카운터보어/카운터싱크(⌴/↧)+분수 조합, 괄호묶인 상한/하한 조합 (검수용 프리뷰)
================================================================================
실라벨 데이터(finetune/labeling/train/Label.txt)에서 실사용 확인된 두 패턴:
  1) ⌴7/16↧1/4, ⌴ø25/64↧ 13/64 처럼 카운터보어지름+카운터싱크깊이가
     인치분수와 붙어 나오는 케이스. ↧는 사전에 있음(노출부족 문제),
     ⌴는 사전에 없음(별도 결정 필요 — 이 스크립트에선 ↧ 조합만 생성,
     ⌴ 들어간 조합은 사전확장 결정 나기 전까진 라벨에서 제외).
  2) "0.792(+0.005/-0.000)", "[ø2.283(+0.012/-0.000)]" 처럼 상한/하한을
     괄호로 묶어 한 줄에 표기하는 케이스 — 괄호는 원본에 실제로 있는
     문자(우리가 추가하는 게 아님, 실라벨로 확인됨).

출력: finetune/counterbore_bracket_preview/preview.png
"""

import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

random.seed(404)

OUT_DIR = Path('finetune/counterbore_bracket_preview')
IMG_DIR = OUT_DIR / 'images'

FONT_PATHS = [r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\calibri.ttf']
SEGUISYM = r'C:\Windows\Fonts\seguisym.ttf'
SEGUISYM_ONLY = set('⏤⏥⌭∥⊥▽⌀∅①②③Ⓐ↧')  # ↧ 추가 — 이 문자도 seguisym 전용
DIAM = ['ø']  # Φ/φ/Ø는 사람 눈엔 사실상 동일 모양이라 ø로 통일
INCH_MARKS = ['"', '″', '”']  # 직선/더블프라임(사실상 동일 취급) + 겹따옴표


def get_font(size, text=''):
    if any(ch in SEGUISYM_ONLY for ch in text):
        return ImageFont.truetype(SEGUISYM, size)
    return ImageFont.truetype(random.choice(FONT_PATHS), size)


def rand_fraction():
    num = random.randint(1, 63)
    den = random.choice([4, 8, 16, 32, 64])
    return f'{num}/{den}'


def gen_counterbore():
    """⌴(카운터보어)는 사전에 없어서 제외, ↧(카운터싱크깊이)만 사용.
    실사용 예시 형태: 지름값 + ↧ + 깊이값 (분수, 가끔 ø 붙음)."""
    d1 = rand_fraction()
    if random.random() < 0.3:
        d1 = f'{random.choice(DIAM)}{d1}'
    d2 = rand_fraction()
    sep = random.choice([' ', ''])
    return f'{d1}↧{sep}{d2}'


def gen_bracket_tol():
    """괄호로 묶은 상한/하한. 지름기호+인치값+괄호쌍, 대괄호로 전체 감싸는 것도 섞음."""
    nominal = f'{random.randint(0, 3)}.{random.randint(100, 999)}'
    upper = f'0.{random.randint(1, 99):03d}'
    lower = random.choice(['0.000', f'0.{random.randint(1, 99):03d}'])
    core = f'{nominal}(+{upper}/-{lower})'
    if random.random() < 0.4:
        sym = random.choice(DIAM)
        core = f'{sym}{core}'
    if random.random() < 0.5:
        core = f'[{core}]'
    return core


def render_crop(text):
    size = random.randint(24, 34)
    font = get_font(size, text)
    tmp = Image.new('RGB', (10, 10), (255, 255, 255))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0] + 24, bbox[3] - bbox[1] + 24
    img = Image.new('RGB', (max(w, 20), max(h, 20)), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((12 - bbox[0], 12 - bbox[1]), text, font=font, fill=(0, 0, 0))
    return img


def main(n=25):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    labels, rows = [], []
    for i in range(n):
        kind = random.choice(['counterbore', 'counterbore', 'bracket_tol'])
        text = gen_counterbore() if kind == 'counterbore' else gen_bracket_tol()
        img = render_crop(text)
        name = f'cb_{i:03d}.png'
        img.save(IMG_DIR / name)
        labels.append(f'{IMG_DIR / name}\t{text}')
        rows.append((img, text, kind))

    with open(OUT_DIR / 'rec_gt.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels) + '\n')

    row_h = 90
    preview = Image.new('RGB', (700, row_h * n), (255, 255, 255))
    pd = ImageDraw.Draw(preview)
    label_font = ImageFont.truetype(r'C:\Windows\Fonts\malgun.ttf', 15)
    for i, (img, text, kind) in enumerate(rows):
        y = i * row_h
        preview.paste(img, (10, y + 10))
        pd.text((350, y + 35), f"#{i:02d} [{kind}] 정답='{text}'", font=label_font, fill=(0, 0, 150))
        pd.line([(0, y + row_h), (preview.width, y + row_h)], fill=(220, 220, 220))
    preview.save(OUT_DIR / 'preview.png')
    print(f'생성 완료: {n}장 -> {IMG_DIR}')


if __name__ == '__main__':
    main()
