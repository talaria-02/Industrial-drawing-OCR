"""
3차 기호조합 생성 — 실측 심각도 순으로 비중 재배분
=====================================================
v2 실측 결과(200장 샘플):
  GD&T 프레임   2% (2/82)   ← 거의 전멸, 최우선
  표면거칠기   16% (12/75)  ← 심각, 2순위
  반복구멍     79% (34/43)  ← 이미 어느정도 됨, 최소비중

그래서 비중을 50% / 35% / 15%로 재배분.
GD&T는 간격(공백 있음/없음) 변수도 추가 — v2는 전부 공백 있는
버전만 썼는데, 간격이 정확도에 영향을 주는지 확인하기 위함.

출력: finetune/synth_symbols_v3/images/*.png + rec_gt.txt
"""

import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

random.seed(23)

OUT_DIR = Path('finetune/synth_symbols_v3')
IMG_DIR = OUT_DIR / 'images'

FONT_PATHS = [
    r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\seguisym.ttf',
    r'C:\Windows\Fonts\calibri.ttf', r'C:\Windows\Fonts\times.ttf',
]
SEGUISYM = r'C:\Windows\Fonts\seguisym.ttf'
# fontTools cmap으로 실측 확인됨: 이 문자들은 seguisym에만 있음
SEGUISYM_ONLY = set('⏤⏥⌭∥⊥▽⌀∅①②③Ⓐ')

def get_font(size, text=''):
    """텍스트에 seguisym 전용 문자가 있으면 강제로 seguisym 사용.
    (arial/calibri/times는 이 문자들을 tofu box로 깨뜨림 — fontTools로 실측 확인)"""
    if any(ch in SEGUISYM_ONLY for ch in text):
        return ImageFont.truetype(SEGUISYM, size)
    p = random.choice(FONT_PATHS)
    if not Path(p).exists():
        p = FONT_PATHS[0]
    return ImageFont.truetype(p, size)


GDT_SYMS = ['⏤', '⏥', '⌭', '∥', '⊥']
DIAM = ['ø']  # Φ/φ/Ø는 사람 눈엔 사실상 동일 모양이라 ø로 통일
DATUMS = ['A', 'B', 'C']


def rand_tol():
    whole = random.choice(['0', '0', '1'])
    frac = random.choice(['02', '05', '1', '2'])
    sep = random.choice(['.', ','])
    return f'{whole}{sep}{frac}'


def gen_gdt():
    """GD&T 프레임 — 데이텀 개수·Ø유무·구획 간격을 다 섞음.
    기호-공차 사이는 항상 공백(안 그러면 렌더가 겹쳐 깨짐).
    데이텀 문자 사이 간격만 있음/없음을 변수로 둠."""
    sym = random.choice(GDT_SYMS)
    tol = rand_tol()
    if random.random() < 0.35:
        tol = f'{random.choice(DIAM)}{tol}'          # Ø 붙는 경우

    n_datum = random.choice([0, 1, 1, 2, 2, 3])
    datums = DATUMS[:] if n_datum == 3 else random.sample(DATUMS, n_datum)

    datum_sep = ' ' if random.random() < 0.6 else ''
    return f'{sym} {tol}' + (' ' + datum_sep.join(datums) if datums else '')


def gen_rough():
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
    if kind == 'ra_tight':
        return f'▽Ra{val}'
    return '▽'


def gen_hole():
    n = random.randint(2, 8)
    d = random.randint(3, 30)
    sep = random.choice(['×', '-'])
    return f'{n}{sep}{random.choice(DIAM)}{d}'


def gen_pattern():
    r = random.random()
    if r < 0.50:
        return gen_gdt(), 'gdt'
    elif r < 0.85:
        return gen_rough(), 'rough'
    else:
        return gen_hole(), 'hole'


def render_crop(text):
    size = random.randint(26, 44)
    font = get_font(size, text)
    tmp = Image.new('RGB', (10, 10), (255, 255, 255))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0] + 24, bbox[3] - bbox[1] + 24
    img = Image.new('RGB', (max(w, 20), max(h, 20)), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((12 - bbox[0], 12 - bbox[1]), text, font=font, fill=(0, 0, 0))
    if random.random() < 0.3:
        ang = random.uniform(-4, 4)
        img = img.rotate(ang, expand=True, fillcolor=(255, 255, 255),
                         resample=Image.BICUBIC)
    return img


def main(n_samples=3000):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    labels = []
    counts = {'gdt': 0, 'rough': 0, 'hole': 0}
    for i in range(n_samples):
        text, cat = gen_pattern()
        counts[cat] += 1
        img = render_crop(text)
        name = f'v3_{i:05d}.png'
        img.save(IMG_DIR / name)
        labels.append(f'{IMG_DIR / name}\t{text}')
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{n_samples}')
    with open(OUT_DIR / 'rec_gt.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels) + '\n')
    print(f'생성 완료: {n_samples}장 -> {IMG_DIR}')
    print(f'분포: {counts}')


if __name__ == '__main__':
    main()
