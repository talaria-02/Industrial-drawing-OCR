"""
특수기호 인식 성능 테스트 이미지 생성
======================================
DOMAIN.md/사전 실측에서 정리한 기호들을 격자로 띄엄띄엄 배치.
등록됨(동작 예상) / 미등록(실패 예상) 두 줄로 나눠 그린다.
각 칸에 인덱스 번호를 붙여 OCR 결과와 위치로 대조 가능하게 함.

출력: data/generated/symbol_test.png
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SYMBOL_FONTS = [r'C:\Windows\Fonts\seguisym.ttf', r'C:\Windows\Fonts\arial.ttf']
CAPTION_FONTS = [r'C:\Windows\Fonts\malgun.ttf', r'C:\Windows\Fonts\arial.ttf']

def _pick_font(candidates, size):
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def get_font(size):
    return _pick_font(SYMBOL_FONTS, size)

def get_caption_font(size):
    return _pick_font(CAPTION_FONTS, size)

# (기호, 설명, 등록여부)
REGISTERED = [
    ('ø10', '지름(소문자)'), ('⌀10', '지름(전용기호)'), ('∅10', '지름(공집합대용)'),
    ('Φ10', '파이(대문자)'), ('φ10', '파이(소문자)'),
    ('±0.1', '공차'), ('45°', '각도'), ('2×45°', '모따기'),
    ('▽▽', '표면거칠기(구식)'), ('√Ra', '표면거칠기(신식)'),
    ('①②③', '원문자(부품번호)'),
    ('⏤', 'GDT 진직도'), ('⏥', 'GDT 평면도'), ('⌭', 'GDT 원통도'),
    ('∥', 'GDT 평행도'), ('⊥', 'GDT 직각도'),
]

UNREGISTERED = [
    ('Ø10', '지름(대문자, ISO표준)'), ('Ⓐ', '데이텀 원문자'), ('⌴', '카운터보어'),
    ('○', 'GDT 진원도'), ('⌖', 'GDT 위치도'), ('◎', 'GDT 동심도'),
    ('⌰', 'GDT 원주흔들림'), ('⌱', 'GDT 온흔들림'),
]


ROW_H = 140
COL_W = 180
COLS = 8

def draw_row(draw, items, y0, font_sym, font_cap, idx_start):
    x = 40
    y = y0
    for i, (sym, desc) in enumerate(items):
        idx = idx_start + i
        if i > 0 and i % COLS == 0:
            x = 40
            y += ROW_H
        draw.text((x, y), f'[{idx}]', font=font_cap, fill=(120, 120, 120))
        draw.text((x, y + 22), sym, font=font_sym, fill=(0, 0, 0))
        draw.text((x, y + 80), desc, font=font_cap, fill=(80, 80, 80))
        x += COL_W
    n_rows = (len(items) - 1) // COLS + 1
    return y0 + n_rows * ROW_H


def main():
    W = 40 + COL_W * COLS
    img = Image.new('RGB', (W, 900), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_sym = get_font(34)
    font_cap = get_caption_font(14)
    font_title = get_caption_font(20)

    draw.text((40, 15), '=== 등록됨 (동작 예상) ===', font=font_title, fill=(0, 100, 0))
    next_y = draw_row(draw, REGISTERED, 55, font_sym, font_cap, 1)

    title_y = next_y + 20
    draw.text((40, title_y), '=== 미등록 (실패 예상) ===', font=font_title, fill=(150, 0, 0))
    final_y = draw_row(draw, UNREGISTERED, title_y + 40, font_sym, font_cap, 100)

    img = img.crop((0, 0, W, final_y + 20))
    out = Path('data/generated/symbol_test.png')
    img.save(out)
    print(f'저장: {out}, 크기: {img.size}')


if __name__ == '__main__':
    main()
