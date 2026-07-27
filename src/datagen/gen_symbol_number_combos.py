"""
기호+숫자 인접패턴 합성 생성기 (rec 학습용)
==============================================
문제 진단: Ø10 같은 "기호+숫자가 붙어있는 시퀀스"를 모델이 학습 중
충분히 못 봐서, 010처럼 기호를 숫자로 오인식함. 기호 자체는 사전에
있고 단독으로는 문제없음 — 부족한 건 "기호 바로 옆 숫자" 조합의 노출량.

그래서 이 스크립트는 실제 도면에 나오는 조합 패턴대로, 다양한 숫자값·
폰트·크기·간격·회전으로 대량의 crop을 찍어낸다. 사용 기호는 전부
PP-OCRv6_small_rec 사전에 실제 등록된 것만 사용(등록 자체가 안 된
Ø·GD&T 전용기호는 이 스크립트의 대상이 아님 — 그건 다른 문제).

출력: data/generated/synth_symbols/images/*.png + rec_gt.txt (경로\t정답텍스트)
"""

import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

random.seed(7)

OUT_DIR = Path('data/generated/synth_symbols')
IMG_DIR = OUT_DIR / 'images'

FONT_PATHS = [
    r'C:\Windows\Fonts\arial.ttf',
    r'C:\Windows\Fonts\seguisym.ttf',
    r'C:\Windows\Fonts\calibri.ttf',
    r'C:\Windows\Fonts\times.ttf',
]
SEGUISYM = r'C:\Windows\Fonts\seguisym.ttf'
# fontTools cmap 실측: ⌀·∅는 arial/calibri/times엔 없음 (tofu box로 깨짐)
SEGUISYM_ONLY = set('⏤⏥⌭∥⊥▽⌀∅①②③Ⓐ')

def get_font(size, text=''):
    if any(ch in SEGUISYM_ONLY for ch in text):
        return ImageFont.truetype(SEGUISYM, size)
    p = random.choice(FONT_PATHS)
    if not Path(p).exists():
        p = FONT_PATHS[0]
    return ImageFont.truetype(p, size)


def rand_num(kind=None):
    """치수값다운 숫자 문자열 생성. kind로 소수점 표기 다양화."""
    kind = kind or random.choice(['int', 'int', 'dec_dot', 'dec_comma'])
    whole = random.randint(1, 200)
    if kind == 'int':
        return str(whole)
    frac = random.choice(['1', '5', '05', '25'])
    sep = '.' if kind == 'dec_dot' else ','
    return f'{whole}{sep}{frac}'


# ∅/⌀은 폰트 미지원 폰트에서 소실 확인됨(위험), Φ/φ/Ø는 사람 눈엔 ø와
# 사실상 동일 모양이라 구분 학습시킬 이유 없음(확인 후 통일) -> ø만 사용
DIAMETER_SYMS = ['ø']
FIT_CODES = ['H7', 'H9', 'f7', 'g6', 'h6', 'k6']
TOL_VALS = ['0.1', '0,1', '0.05', '0,05', '0.2']


def gen_pattern():
    """(표시텍스트, 패턴이름) 하나 생성 — 실제 도면 표기 관례 반영."""
    p = random.choice([
        'diam_tight', 'diam_space', 'radius', 'square',
        'tol_dot', 'tol_comma', 'angle', 'chamfer', 'fit',
    ])
    n = rand_num('int')

    if p == 'diam_tight':               # Ø10, ø10, ⌀10  (붙여쓰기 — 제일 흔함)
        sym = random.choice(DIAMETER_SYMS)
        return f'{sym}{n}'
    if p == 'diam_space':               # Ø 10  (기호-숫자 사이 살짝 띄움)
        sym = random.choice(DIAMETER_SYMS)
        return f'{sym} {n}'
    if p == 'radius':                   # R5, R70
        return f'R{n}'
    if p == 'square':                   # □15
        return f'□{n}'
    if p == 'tol_dot':                  # 50±0.1
        return f'{n}±{random.choice(TOL_VALS)}'
    if p == 'tol_comma':                # 50±0,1 (쉼표 소수점)
        return f'{rand_num("int")}±{random.choice(["0,1","0,05","0,2"])}'
    if p == 'angle':                    # 45°
        return f'{n}°'
    if p == 'chamfer':                  # 2×45°
        return f'{random.randint(1,3)}×{n}°'
    if p == 'fit':                      # 30 H9, Ø10 H7
        base = f'{random.choice(DIAMETER_SYMS)}{n}' if random.random() < 0.4 else n
        return f'{base} {random.choice(FIT_CODES)}'
    return n


def render_crop(text, idx):
    """텍스트 하나를 crop 이미지로 렌더링. 폰트/크기/약간의 회전 다양화."""
    size = random.randint(26, 44)
    font = get_font(size, text)

    tmp = Image.new('RGB', (10, 10), (255, 255, 255))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0] + 20, bbox[3] - bbox[1] + 20

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10 - bbox[0], 10 - bbox[1]), text, font=font, fill=(0, 0, 0))

    # 미세 회전(스캔 기울기 흉내) — 배경 흰색으로 보정
    if random.random() < 0.3:
        ang = random.uniform(-4, 4)
        img = img.rotate(ang, expand=True, fillcolor=(255, 255, 255),
                         resample=Image.BICUBIC)

    return img


def main(n_samples=1500):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    labels = []

    for i in range(n_samples):
        text = gen_pattern()
        img = render_crop(text, i)
        name = f'sym_{i:05d}.png'
        img.save(IMG_DIR / name)
        labels.append(f'{IMG_DIR / name}\t{text}')

        if (i + 1) % 300 == 0:
            print(f'  {i+1}/{n_samples}')

    with open(OUT_DIR / 'rec_gt.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels) + '\n')

    print(f'\n생성 완료: {n_samples}장 -> {IMG_DIR}')
    print(f'라벨: {OUT_DIR / "rec_gt.txt"}')


if __name__ == '__main__':
    main()
