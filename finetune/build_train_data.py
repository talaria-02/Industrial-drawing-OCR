"""
rec 학습셋 통합
================
지금까지 만든 모든 rec 데이터(합성 + 실라벨crop)를 PaddleOCR가 바로
쓸 수 있는 train_data/ 구조로 합침.

소스:
  - synth_symbols, synth_symbols_v2, synth_symbols_v3 (기호+숫자 조합)
  - stacked_tol_preview (적층 비대칭공차)
  - line_noise_preview (선/화살표 노이즈 hard negative)
  - counterbore_bracket_preview (카운터싱크+분수, 괄호 상하한)
  - real_augment_preview (실라벨 text-swap+회전+노이즈)
  - priority_v4_preview (실측 우선순위 재조정판)
  - 실라벨 원본 crop (finetune/labeling/train, zone 안쪽만, OOV 제외)

출력: finetune/train_data_consolidated/{imgs/, train_list.txt, val_list.txt}
      -> 이후 PaddleOCR repo의 train_data/ 로 복사해서 사용
"""

import json
import random
import shutil
from pathlib import Path
from PIL import Image

from zone_utils import split_zone

random.seed(2024)

OUT_DIR = Path('finetune/train_data_consolidated')
IMG_OUT = OUT_DIR / 'imgs'

SYNTH_SOURCES = [
    'finetune/synth_symbols/rec_gt.txt',
    'finetune/synth_symbols_v2/rec_gt.txt',
    'finetune/synth_symbols_v3/rec_gt.txt',
    'finetune/stacked_tol_preview/rec_gt.txt',
    'finetune/line_noise_preview/rec_gt.txt',
    'finetune/counterbore_bracket_preview/rec_gt.txt',
    'finetune/real_augment_preview/rec_gt.txt' if Path('finetune/real_augment_preview/rec_gt.txt').exists() else None,
    'finetune/priority_v4_preview/rec_gt.txt',
]

V6_DICT = r'C:\Users\zxc20\OneDrive\바탕 화면\ppocr\PaddleOCR\ppocr\utils\dict\ppocrv6_dict.txt'
REAL_LABEL_FILE = Path('finetune/labeling/train/Label.txt')
REAL_IMG_DIR = Path('finetune/labeling/train')


def load_dict_chars():
    with open(V6_DICT, encoding='utf-8') as f:
        return set(f.read().replace('\n', ''))


def is_in_dict(text, dict_chars):
    return all(ch in dict_chars or ch.isspace() for ch in text)


def collect_synth(dict_chars):
    pairs = []
    for src in SYNTH_SOURCES:
        if not src or not Path(src).exists():
            continue
        with open(src, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                path, label = line.split('\t', 1)
                if not is_in_dict(label, dict_chars):
                    continue
                pairs.append((Path(path), label))
    return pairs


def collect_real_crops(dict_chars):
    """실라벨 이미지에서 zone 안쪽 item들 crop 떠서 (원본 그대로) 학습쌍으로 추가.
    메타데이터(zone 밖)도 포함 — 일반텍스트 감지력 유지 목적([[project-scope-korean-cancelled]] 참고 논의와
    같은 맥락: 기존에 잘 되는 영역은 학습셋에 그대로 두되 증강만 안 함)."""
    pairs = []
    with open(REAL_LABEL_FILE, encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        path, js = line.split('\t', 1)
        items = json.loads(js)
        zone_bbox, drawing_items, metadata_items = split_zone(items)
        img = None
        for it in drawing_items + metadata_items:
            text = it['transcription']
            if not is_in_dict(text, dict_chars):
                continue
            if img is None:
                img = Image.open(REAL_IMG_DIR / Path(path).name).convert('RGB')
            xs = [p[0] for p in it['points']]
            ys = [p[1] for p in it['points']]
            box = (max(0, min(xs) - 3), max(0, min(ys) - 3), max(xs) + 3, max(ys) + 3)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crop = img.crop(box)
            pairs.append((crop, text))
    return pairs


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    IMG_OUT.mkdir(parents=True, exist_ok=True)

    dict_chars = load_dict_chars()

    all_pairs = []  # (image_source, label) where image_source is Path(파일경로) or PIL.Image

    synth_pairs = collect_synth(dict_chars)
    print(f'합성데이터: {len(synth_pairs)}쌍')
    all_pairs.extend(synth_pairs)

    real_pairs = collect_real_crops(dict_chars)
    print(f'실라벨 crop: {len(real_pairs)}쌍')
    all_pairs.extend(real_pairs)

    print(f'총합: {len(all_pairs)}쌍 (사전밖 문자 포함 라벨은 이미 제외됨)')

    random.shuffle(all_pairs)
    n_val = max(1, int(len(all_pairs) * 0.05))
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]

    def write_split(pairs, list_name):
        lines = []
        for i, (src, label) in enumerate(pairs):
            name = f'{list_name}_{i:06d}.png'
            dst = IMG_OUT / name
            if isinstance(src, Path):
                img = Image.open(src).convert('RGB')
            else:
                img = src
            img.save(dst)
            lines.append(f'imgs/{name}\t{label}')
        with open(OUT_DIR / f'{list_name}_list.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

    write_split(train_pairs, 'train')
    write_split(val_pairs, 'val')

    print(f'train: {len(train_pairs)} / val: {len(val_pairs)}')
    print(f'출력: {OUT_DIR}')


if __name__ == '__main__':
    main()
