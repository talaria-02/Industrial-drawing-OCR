"""
PaddleOCR v2(3패스 회전-병합 + 영역분리) 성능 채점
====================================================
비교 대상:
  baseline (1패스):        recall 52.8% / precision 44.4%
  v2 (3패스 병합):         recall ? / precision ?
  v2 + region(meta 제외):  precision ?

동일 정답·동일 매칭 기준(eval_ocr.py) 사용.

사용법: python eval_paddle_v2.py
"""

import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

import cv2

# paddle_rotate_merge / region_split 는 src/pipeline 에 있음 — 그쪽을 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))

from eval_ocr import to_num, gt_numbers, match_multiset
from paddle_rotate_merge import build_ocr, extract, center_of
from region_split import split_regions, classify_point


def main():
    ocr = build_ocr()
    img_dir = Path('data/synth/images')
    lbl_dir = Path('data/synth/labels')
    pairs = [(ip, lbl_dir / f'{ip.stem}.json')
             for ip in sorted(img_dir.glob('drawing_*.png'))
             if (lbl_dir / f'{ip.stem}.json').exists()]
    print(f"채점 대상: {len(pairs)}장 (3패스 + 영역분리)", flush=True)

    agg = Counter()
    by_noise = defaultdict(Counter)

    for i, (ip, lp) in enumerate(pairs):
        try:
            with open(lp, encoding='utf-8') as f:
                label = json.load(f)
            gt = gt_numbers(label)

            dets = extract(ocr, ip)

            # 영역 분리 → drawing/meta 태깅
            _, frame, _, titleblock = split_regions(ip)
            det_all, det_drawing = [], []
            for d in dets:
                n = to_num(d['text'])
                if n is None:
                    continue
                det_all.append(n)
                cx, cy = center_of(d['poly'])
                if classify_point(cx, cy, frame, titleblock) == 'drawing':
                    det_drawing.append(n)

            hit_all = match_multiset(gt, det_all)
            hit_drw = match_multiset(gt, det_drawing)

            agg['n_gt'] += len(gt)
            agg['n_det_all'] += len(det_all)
            agg['n_det_drw'] += len(det_drawing)
            agg['hit_all'] += hit_all
            agg['hit_drw'] += hit_drw

            nb = by_noise[label.get('noise', '?')]
            nb['n_gt'] += len(gt)
            nb['hit_all'] += hit_all

            print(f"  [{i+1}/{len(pairs)}] {ip.name} gt={len(gt)} "
                  f"det={len(det_all)}/{len(det_drawing)} "
                  f"hit={hit_all}/{hit_drw}", flush=True)
        except Exception as e:
            print(f"  [ERROR] {ip.name}: {e}", flush=True)

    def pct(a, b):
        return f'{100*a/b:5.1f}%' if b else '  n/a'

    print(f"\n{'='*62}")
    print("  PaddleOCR v2 (3패스 회전-병합) 결과")
    print(f"{'='*62}")
    print(f"  정답 치수 총계     : {agg['n_gt']}")
    print(f"  recall (전체검출)  : {pct(agg['hit_all'], agg['n_gt'])}  "
          f"({agg['hit_all']}/{agg['n_gt']})")
    print(f"  precision (전체)   : {pct(agg['hit_all'], agg['n_det_all'])}  "
          f"({agg['hit_all']}/{agg['n_det_all']})")
    print(f"  precision (drawing만): {pct(agg['hit_drw'], agg['n_det_drw'])}  "
          f"({agg['hit_drw']}/{agg['n_det_drw']})")
    print(f"  recall (drawing만) : {pct(agg['hit_drw'], agg['n_gt'])}  "
          f"({agg['hit_drw']}/{agg['n_gt']})")
    print(f"\n  [노이즈 레벨별 recall(전체)]")
    for lvl in ('clean', 'slight', 'noisy', 'heavy'):
        nb = by_noise.get(lvl)
        if nb and nb['n_gt']:
            print(f"    {lvl:>7}: {pct(nb['hit_all'], nb['n_gt'])}  "
                  f"({nb['hit_all']}/{nb['n_gt']})")
    print(f"{'='*62}")


if __name__ == '__main__':
    main()
