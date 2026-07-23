"""
적층 공차 det 통짜박스 -> rec 전 분리 (옵션 a 구현)
==================================================
det은 nominal+upper+lower 3줄을 통짜 박스 하나로 잡는 게 자연스러움
(실제 라벨링 관행과 일치, [[stacked-tolerance-split-approach]] 참고).
근데 rec(CTC)은 한 줄만 읽으므로, rec에 넣기 전에 이 crop을 쪼개는
후처리 단계가 필요함.

레이아웃 특성: nominal(왼쪽, 전체높이) + upper/lower(오른쪽, 위아래
2단). 그래서 단순 가로전체 행(row) 투영만으론 안 됨 — nominal 잉크가
전체 높이를 채워서 상한/하한 사이 여백이 안 보임. 그래서:
  1단계: 컬럼(열) 투영으로 좌(nominal)/우(upper+lower) 블록 분리
  2단계: 우측 블록 안에서만 행(row) 투영으로 상한/하한 분리

merge_for_test(): 검증용 — 기존 3-crop(nominal/upper/lower)을 합쳐
실제 det 통짜박스처럼 재구성.
split_stacked(): 실제 분리 로직.
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def _ink_mask(img, thresh=200):
    arr = np.array(img.convert('L'))
    return arr < thresh


def _runs(has_ink, min_gap):
    """bool 배열에서 True-run들을 뽑되, min_gap보다 좁은 False갭은 같은 run으로 병합."""
    n = len(has_ink)
    raw = []
    i = 0
    while i < n and not has_ink[i]:
        i += 1
    while i < n:
        start = i
        while i < n and has_ink[i]:
            i += 1
        raw.append([start, i])
        while i < n and not has_ink[i]:
            i += 1
    if not raw:
        return []
    merged = [raw[0]]
    for r in raw[1:]:
        if r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged


def _widest_gap_center(has_ink, min_gap):
    """False(빈칸) 구간 중 가장 넓은 것 하나의 중간 지점을 분리기준으로 반환.
    맨 앞/뒤 여백(텍스트 시작 전/끝난 후)은 후보에서 제외 — 실제 분리 지점이 아님.
    run 개수로 좌/우를 나누면 문자 내부의 우연한 sub-gap에 오작동하기 쉬워서,
    "가장 넓은 gap = 진짜 블록 경계"라는 가정으로 안정성을 높임."""
    n = len(has_ink)
    gaps = []
    i = 0
    while i < n:
        if not has_ink[i]:
            start = i
            while i < n and not has_ink[i]:
                i += 1
            if start > 0 and i < n:  # 시작 전/끝난 후 여백 제외
                gaps.append((start, i))
        else:
            i += 1
    if not gaps:
        return None
    widest = max(gaps, key=lambda g: g[1] - g[0])
    if widest[1] - widest[0] < min_gap:
        return None
    return (widest[0] + widest[1]) // 2


def _tight_crop(img, pad=2, thresh=200):
    mask = _ink_mask(img, thresh)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    y0, y1 = max(0, rows[0] - pad), min(img.height, rows[-1] + 1 + pad)
    x0, x1 = max(0, cols[0] - pad), min(img.width, cols[-1] + 1 + pad)
    return img.crop((x0, y0, x1, y1))


def split_stacked(img, col_min_gap=5, row_min_gap=3, pad=2):
    """1단계: 가장 넓은 컬럼 여백으로 좌(nominal)/우(upper+lower) 분리.
    2단계: 우측 블록 안에서 행 투영으로 상한/하한 분리.
    좌우 분리 지점을 못 찾으면(뚜렷한 넓은 여백 없음) 원본 그대로 1장 반환(순수 1줄 케이스)."""
    mask = _ink_mask(img)
    col_has_ink = mask.any(axis=0)
    split_x = _widest_gap_center(col_has_ink, min_gap=col_min_gap)
    if split_x is None:
        return [img]

    nominal_crop = _tight_crop(img.crop((0, 0, split_x, img.height)), pad)
    right_crop = img.crop((split_x, 0, img.width, img.height))

    right_mask = _ink_mask(right_crop)
    row_has_ink = right_mask.any(axis=1)
    row_runs = _runs(row_has_ink, row_min_gap)

    right_subs = []
    for r in row_runs:
        sub = right_crop.crop((0, max(0, r[0] - pad), right_crop.width,
                                min(right_crop.height, r[1] + pad)))
        right_subs.append(_tight_crop(sub, pad))
    if not right_subs:
        right_subs = [_tight_crop(right_crop, pad)]

    return [nominal_crop] + right_subs


def merge_for_test(nom_img, up_img, lo_img, gap=14):
    """검증용 — 실제 det이 잡을 법한 통짜박스 재구성(nominal 왼쪽/upper·lower 오른쪽 위아래)."""
    right_w = max(up_img.width, lo_img.width)
    right_h = up_img.height + lo_img.height + 2
    total_w = nom_img.width + gap + right_w
    total_h = max(nom_img.height, right_h)

    img = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    ny = (total_h - nom_img.height) // 2
    img.paste(nom_img, (0, ny))
    rx = nom_img.width + gap
    ry = (total_h - right_h) // 2
    img.paste(up_img, (rx, ry))
    img.paste(lo_img, (rx, ry + up_img.height + 2))
    return img


def main():
    src = Path('finetune/stacked_tol_preview/images')
    with open('finetune/stacked_tol_preview/rec_gt.txt', encoding='utf-8') as f:
        gt = dict(line.strip().split('\t') for line in f if line.strip())

    n_sets = 20
    rows = []
    for i in range(n_sets):
        nom = Image.open(src / f'set{i:02d}_nom.png')
        up = Image.open(src / f'set{i:02d}_up.png')
        lo = Image.open(src / f'set{i:02d}_lo.png')
        nom_text = gt[str(src / f'set{i:02d}_nom.png')]
        up_text = gt[str(src / f'set{i:02d}_up.png')]
        lo_text = gt[str(src / f'set{i:02d}_lo.png')]

        merged = merge_for_test(nom, up, lo)
        parts = split_stacked(merged)
        ok = len(parts) == 3
        rows.append((merged, parts, (nom_text, up_text, lo_text), ok))

    out_dir = Path('finetune/split_stacked_preview')
    out_dir.mkdir(exist_ok=True)
    row_h = 130
    preview = Image.new('RGB', (900, row_h * n_sets), (255, 255, 255))
    pd = ImageDraw.Draw(preview)
    font = ImageFont.truetype(r'C:\Windows\Fonts\malgun.ttf', 15)

    n_ok = 0
    for i, (merged, parts, texts, ok) in enumerate(rows):
        y = i * row_h
        preview.paste(merged, (10, y + 10))
        x = merged.width + 40
        for p in parts:
            preview.paste(p, (x, y + 10))
            x += p.width + 10
        status = 'OK(3분리)' if ok else f'FAIL({len(parts)}개로 분리됨)'
        pd.text((10, y + merged.height + 15),
                 f"#{i:02d} 정답=nom:{texts[0]!r} up:{texts[1]!r} lo:{texts[2]!r}  ->  {status}",
                 font=font, fill=(0, 0, 150) if ok else (200, 0, 0))
        pd.line([(0, y + row_h), (preview.width, y + row_h)], fill=(220, 220, 220))
        n_ok += ok

    preview.save(out_dir / 'preview.png')
    print(f'{n_ok}/{n_sets} 성공적으로 3분리됨')
    print(f'검수용: {out_dir / "preview.png"}')


if __name__ == '__main__':
    main()
