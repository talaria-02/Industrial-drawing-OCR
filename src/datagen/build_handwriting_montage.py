"""
실라벨 이미지에서 crop한 진짜 도면 텍스트로 격자 몽타주 생성
(이미지생성모델에 한번에 "손글씨로 다시 그려줘" 요청용)
================================================================================================
합성 렌더링이 아니라 실제 도면(data/real/train)에서 크롭한 진짜 텍스트를 씀 —
실제 폰트/스캔노이즈/주변맥락이 있는 진짜배기라야 손글씨 증강 테스트 의미가 있음.

칸 사이 검은 테두리선 + 여백을 넉넉히 둬서 생성모델이 격자구조를 최대한 유지하게 유도.
생성 결과를 다시 원래 좌표로 잘라내 rec으로 검증할 수 있도록, 몽타주 배치 좌표를
grid_layout.json에 같이 저장함(각 칸의 (row,col)->원본이미지/정답라벨 매핑).

사용:
  python data/generated/build_handwriting_montage.py --n 12 --cols 4 --out data/generated/handwriting_test
"""

import argparse
import json
import random
import re
from pathlib import Path
from PIL import Image, ImageDraw

from zone_utils import split_zone

CELL_PAD = 20       # 칸 내부 여백(이미지 주변)
BORDER = 4          # 칸 테두리선 굵기
GAP = 30            # 칸 사이 간격(오염 방지용 여유)

LABEL_FILE = Path('data/real/train/Label.txt')
IMG_DIR = Path('data/real/train')
V6_DICT = r'C:\Users\zxc20\OneDrive\바탕 화면\ppocr\PaddleOCR\ppocr\utils\dict\ppocrv6_dict.txt'

# 기호+숫자 조합 위주로 뽑기 — 순수 숫자만 있는 건 이미 baseline 잘 되니 우선순위 낮음
INTERESTING_PATTERNS = [
    r'[øØ⌀∅Φφ]', r'±', r'[⏤⏥⌭∥⊥]', r'[▽√]', r'[①②③④⑤⑥⑦⑧⑨⑩]',
    r'[↧⌴]', r'^R\d', r'°', r'H\d|g\d|f\d|js\d',
]


def load_dict_chars():
    with open(V6_DICT, encoding='utf-8') as f:
        return set(f.read().replace('\n', ''))


def is_interesting(text, dict_chars):
    if not all(ch in dict_chars or ch.isspace() for ch in text):
        return False  # 사전 밖 문자(⌴/⌵ 등) 포함된 라벨은 제외
    return any(re.search(p, text) for p in INTERESTING_PATTERNS)


def get_bbox(points, pad=3):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(0, min(xs) - pad), max(0, min(ys) - pad), max(xs) + pad, max(ys) + pad)


def collect_real_symbol_crops():
    """실라벨 전체를 훑어서 zone 안쪽(도면영역) + 기호포함 항목만 (crop, label) 리스트로."""
    dict_chars = load_dict_chars()
    candidates = []  # (image_path, points, label)

    with open(LABEL_FILE, encoding='utf-8') as f:
        for line in f:
            path, js = line.split('\t', 1)
            items = json.loads(js)
            zone_bbox, drawing_items, _ = split_zone(items)
            if zone_bbox is None:
                continue
            for it in drawing_items:
                text = it['transcription']
                if is_interesting(text, dict_chars):
                    candidates.append((path, it['points'], text))
    return candidates


def build_montage(samples, cols, out_dir):
    """samples: [(PIL.Image, label, source_desc), ...]"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = [s[0] for s in samples]
    rows = (len(imgs) + cols - 1) // cols

    cell_w = max(im.width for im in imgs) + CELL_PAD * 2
    cell_h = max(im.height for im in imgs) + CELL_PAD * 2

    canvas_w = cols * cell_w + (cols + 1) * GAP
    canvas_h = rows * cell_h + (rows + 1) * GAP
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    layout = []
    for i, (im, label, source_desc) in enumerate(samples):
        r, c = divmod(i, cols)
        cx0 = GAP + c * (cell_w + GAP)
        cy0 = GAP + r * (cell_h + GAP)
        cx1, cy1 = cx0 + cell_w, cy0 + cell_h

        draw.rectangle([cx0, cy0, cx1, cy1], outline=(0, 0, 0), width=BORDER)

        px = cx0 + (cell_w - im.width) // 2
        py = cy0 + (cell_h - im.height) // 2
        canvas.paste(im, (px, py))

        layout.append({
            'row': r, 'col': c,
            'cell_box': [cx0, cy0, cx1, cy1],
            'content_box': [px, py, px + im.width, py + im.height],
            'source': source_desc, 'label': label,
        })

    canvas.save(out_dir / 'montage_input.png')
    with open(out_dir / 'grid_layout.json', 'w', encoding='utf-8') as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)
    return out_dir / 'montage_input.png', out_dir / 'grid_layout.json'


PROMPT_TEMPLATE = """\
Redraw every text/symbol inside this grid as if quickly hand-written in the field
by a tired machinist marking up a shop-floor engineering drawing — messy, rushed,
imperfect handwriting, NOT a clean font.

Make it genuinely messy, not stylized:
- Inconsistent letter size and slant within the same cell; baselines wander up/down.
- Uneven, shaky stroke width — pen pressure varies, some strokes are thin/faint,
  others thick/blotchy where the pen paused.
- Imperfect closures on loops/circles (Ø, 0, digits) — slightly open or overlapping.
- Slightly irregular spacing between characters, occasional touching/overlapping
  strokes between adjacent characters (but still legible).
- Add light physical noise: faint paper grain/texture, a little pencil smudging
  or eraser ghosting nearby, mild ink bleed at stroke ends. Keep it subtle enough
  that the text is still readable, not destroyed.
- This should look like a real worn field annotation, not a "handwriting font" —
  avoid uniform, evenly-spaced, calligraphic, or decorative styles.

CRITICAL constraints (do not break these):
- This image has a grid of {n} separate cells, each bordered by a black rectangle.
  Preserve the exact grid layout, cell positions, and cell border lines. Do not
  merge, resize, move, or reorder cells.
- Inside each cell, preserve the EXACT characters, digits, and symbols shown —
  do not add, remove, or change any character. Content must stay identical to
  the input, cell by cell, even though the style is messy.
- Use the SAME messy handwriting/pen across all cells (as if one rushed person
  wrote the whole sheet), but each cell's own noise/wobble should differ slightly.
- Background stays plain white/off-white outside the ink strokes. Single ink
  color per cell (black, blue, or graphite gray), no color gradients unless
  that's what light smudging naturally produces.
- Do not let content from one cell bleed into a neighboring cell.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--cols', type=int, default=4)
    ap.add_argument('--out', default='data/generated/handwriting_test')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    candidates = collect_real_symbol_crops()
    print(f'기호포함 실라벨 후보: {len(candidates)}개')
    picked = random.sample(candidates, min(args.n, len(candidates)))

    img_cache = {}
    samples = []
    for path, points, label in picked:
        fname = Path(path).name
        if fname not in img_cache:
            img_cache[fname] = Image.open(IMG_DIR / fname).convert('RGB')
        full_img = img_cache[fname]
        box = get_bbox(points)
        crop = full_img.crop(box)
        samples.append((crop, label, f'{fname}@{box}'))

    montage_path, layout_path = build_montage(samples, args.cols, args.out)

    prompt = PROMPT_TEMPLATE.format(n=len(samples))
    with open(Path(args.out) / 'prompt.txt', 'w', encoding='utf-8') as f:
        f.write(prompt)

    print(f'몽타주: {montage_path}')
    print(f'격자 좌표(검증용): {layout_path}')
    print(f'프롬프트: {Path(args.out) / "prompt.txt"}')
    print(f'{len(samples)}개 셀, {args.cols}열')


if __name__ == '__main__':
    main()
