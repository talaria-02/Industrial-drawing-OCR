"""
선/점선 겹침 노이즈 rec 데이터 생성 (검수용 프리뷰)
====================================================
문제: 실제 도면에서 치수선/중심선/숨은선이 숫자 bbox를 가로질러
겹치는 경우, rec이 진짜 숫자에 선을 "-"로 잘못 붙여 읽음(rec 단계
확인됨 — det이 만든 bbox 자체는 정상, bbox 안에서 텍스트+선이 같이
읽히는 케이스).

정답 라벨은 원래 텍스트 그대로(선 관련 문자 없음) — 선이 겹쳐도
텍스트만 읽어야 한다는 걸 hard negative로 학습시킴.

출력: data/generated/line_noise_preview/preview.png   (검수용 대조표)
      data/generated/line_noise_preview/images/*.png  (개별 crop)
      data/generated/line_noise_preview/rec_gt.txt
"""

import math
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

random.seed(101)

OUT_DIR = Path('data/generated/line_noise_preview')
IMG_DIR = OUT_DIR / 'images'

FONT_PATHS = [
    r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\calibri.ttf',
    r'C:\Windows\Fonts\times.ttf',
]
SEGUISYM = r'C:\Windows\Fonts\seguisym.ttf'
SEGUISYM_ONLY = set('⏤⏥⌭∥⊥▽⌀∅①②③Ⓐ')


def get_font(size, text=''):
    if any(ch in SEGUISYM_ONLY for ch in text):
        return ImageFont.truetype(SEGUISYM, size)
    return ImageFont.truetype(random.choice(FONT_PATHS), size)


DIAM = ['ø']  # Φ/φ/Ø는 사람 눈엔 사실상 동일 모양이라 ø로 통일


def rand_text():
    """실제 도면에서 치수선/중심선이 자주 겹치는 대상 — 치수숫자/공차/지름/반지름."""
    kind = random.choice(['int', 'dec', 'tol', 'diam', 'radius'])
    whole = random.randint(1, 200)
    if kind == 'int':
        return str(whole)
    if kind == 'dec':
        frac = random.choice(['5', '25', '75', '1'])
        sep = random.choice(['.', ','])
        return f'{whole}{sep}{frac}'
    if kind == 'tol':
        sep = random.choice(['.', ','])
        tv = random.choice(['05', '1', '2'])
        return f'{whole}±0{sep}{tv}'
    if kind == 'diam':
        return f'{random.choice(DIAM)}{whole}'
    if kind == 'radius':
        return f'R{whole}'
    return str(whole)


def draw_through_line(draw, w, h):
    """텍스트 전체를 가로지르는 선 — 실선/점선/1점쇄선.
    실제 사례 대비 빈도는 낮춤(사용자 피드백: 짧은 스텁이 더 흔함)."""
    kind = random.choice(['solid', 'dashed', 'dashdot'])
    angle = random.uniform(-8, 8)
    cx, cy = w / 2, h / 2
    length = max(w, h) * 1.5
    dx = math.cos(math.radians(angle)) * length / 2
    dy = math.sin(math.radians(angle)) * length / 2
    x0, y0 = cx - dx, cy - dy
    x1, y1 = cx + dx, cy + dy
    thickness = random.choice([1, 1, 2])

    if kind == 'solid':
        draw.line([(x0, y0), (x1, y1)], fill=(0, 0, 0), width=thickness)
        return

    total = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / total, (y1 - y0) / total
    # 폰트 크기 대비 촘촘하게 — 이전 버전은 세그먼트가 너무 커서 실선처럼 보였음
    pattern = [6, 4] if kind == 'dashed' else [9, 3, 2, 3]
    pos, pi = 0, 0
    while pos < total:
        seg = pattern[pi % len(pattern)]
        if pi % 2 == 0:
            sx, sy = x0 + ux * pos, y0 + uy * pos
            ex, ey = x0 + ux * min(pos + seg, total), y0 + uy * min(pos + seg, total)
            draw.line([(sx, sy), (ex, ey)], fill=(0, 0, 0), width=thickness)
        pos += seg
        pi += 1


def draw_stub(draw, x_edge, y_mid, side):
    """숫자 바로 앞/뒤에 붙는 짧은 선분 — 실제 불만사항의 핵심 케이스
    ("-50"처럼 숫자 앞에 짧은 선이 붙어서 마이너스로 오인식).
    side: 'before'면 x_edge 왼쪽으로, 'after'면 오른쪽으로 뻗음."""
    length = random.randint(10, 22)
    gap = random.randint(2, 6)
    thickness = random.choice([1, 1, 2])
    tilt = random.uniform(-6, 6)  # 완전 수평보다 살짝 삐뚤어진 경우가 많음
    dy = math.tan(math.radians(tilt)) * length

    if side == 'before':
        x1 = x_edge - gap
        x0 = x1 - length
    else:
        x0 = x_edge + gap
        x1 = x0 + length
    draw.line([(x0, y_mid), (x1, y_mid + dy)], fill=(0, 0, 0), width=thickness)


def render_crop(text):
    size = random.randint(28, 40)
    font = get_font(size, text)
    tmp = Image.new('RGB', (10, 10), (255, 255, 255))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # 스텁이 앞/뒤에 들어갈 수 있게 좌우 여백을 넉넉히 둠
    pad_side = 34
    pad_v = 12
    img = Image.new('RGB', (tw + pad_side * 2, th + pad_v * 2), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    text_x = pad_side - bbox[0]
    text_y = pad_v - bbox[1]
    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

    y_mid = pad_v + th / 2
    text_left = pad_side
    text_right = pad_side + tw

    # 짧은 스텁(앞/뒤)을 주력으로, 관통선은 일부만 유지
    kind = random.choices(
        ['stub_before', 'stub_after', 'stub_both', 'through'],
        weights=[0.35, 0.30, 0.15, 0.20],
    )[0]

    if kind == 'stub_before':
        draw_stub(draw, text_left, y_mid, 'before')
    elif kind == 'stub_after':
        draw_stub(draw, text_right, y_mid, 'after')
    elif kind == 'stub_both':
        draw_stub(draw, text_left, y_mid, 'before')
        draw_stub(draw, text_right, y_mid, 'after')
    else:
        draw_through_line(draw, img.width, img.height)

    return img


def main(n_sets=25):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    labels = []
    rows = []
    for i in range(n_sets):
        text = rand_text()
        img = render_crop(text)
        name = f'ln_{i:03d}.png'
        img.save(IMG_DIR / name)
        labels.append(f'{IMG_DIR / name}\t{text}')
        rows.append((img, text))

    with open(OUT_DIR / 'rec_gt.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels) + '\n')

    row_h = 90
    col_w = 340
    preview = Image.new('RGB', (col_w + 300, row_h * n_sets), (255, 255, 255))
    pd = ImageDraw.Draw(preview)
    label_font = ImageFont.truetype(r'C:\Windows\Fonts\malgun.ttf', 16)
    for i, (img, text) in enumerate(rows):
        y = i * row_h
        preview.paste(img, (10, y + 10))
        pd.text((col_w + 20, y + 35), f"#{i:02d}  정답='{text}'", font=label_font, fill=(0, 0, 150))
        pd.line([(0, y + row_h), (preview.width, y + row_h)], fill=(220, 220, 220))
    preview.save(OUT_DIR / 'preview.png')
    print(f'생성 완료: {n_sets}장 -> {IMG_DIR}')
    print(f'검수용 대조표: {OUT_DIR / "preview.png"}')


if __name__ == '__main__':
    main()
