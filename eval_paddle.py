"""
PaddleOCR 치수 인식 성능 채점 (recall / precision)
===================================================
eval_ocr.py와 동일 지표·동일 정답으로 PaddleOCR을 평가.
Tesseract baseline(recall 35.1%)과 직접 비교용.

사용법:
  python eval_paddle.py                  # output_v5 전체
  python eval_paddle.py output_v5/images/drawing_0003.png
"""

import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

from paddleocr import PaddleOCR
from eval_ocr import to_num, gt_numbers, match_multiset


def build_ocr():
    # enable_mkldnn=False: paddle 3.3.1 CPU oneDNN 버그 우회
    # mobile 모델: CPU에서 medium보다 5~10배 빠름 (정확도 소폭↓)
    return PaddleOCR(
        text_detection_model_name='PP-OCRv5_mobile_det',
        text_recognition_model_name='PP-OCRv5_mobile_rec',
        use_textline_orientation=True, lang='en',
        enable_mkldnn=False)


def detected_numbers(ocr, img_path):
    r = ocr.predict(str(img_path))
    res = r[0]
    texts = res.get('rec_texts', [])
    scores = res.get('rec_scores', [])
    nums = []
    for t, s in zip(texts, scores):
        if float(s) < 0.5:           # 저신뢰 버림
            continue
        n = to_num(t)
        if n is not None:
            nums.append(n)
    return nums


def main():
    ocr = build_ocr()
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('output_v5/images')

    if arg.is_file():
        pairs = [(arg, Path('output_v5/labels') / f'{arg.stem}.json')]
    else:
        lbl_dir = arg.parent / 'labels'
        pairs = [(ip, lbl_dir / f'{ip.stem}.json')
                 for ip in sorted(arg.glob('drawing_*.png'))
                 if (lbl_dir / f'{ip.stem}.json').exists()]

    print(f"채점 대상: {len(pairs)}장\n")

    agg = Counter()
    by_noise = defaultdict(Counter)

    for i, (ip, lp) in enumerate(pairs):
        try:
            with open(lp, encoding='utf-8') as f:
                label = json.load(f)
            gt = gt_numbers(label)
            det = detected_numbers(ocr, ip)
            hit = match_multiset(gt, det)

            agg['n_gt'] += len(gt)
            agg['n_det'] += len(det)
            agg['hit'] += hit
            nb = by_noise[label.get('noise', '?')]
            nb['n_gt'] += len(gt)
            nb['hit'] += hit
        except Exception as e:
            print(f"  [ERROR] {ip.name}: {e}")
            continue

        print(f"  [{i+1}/{len(pairs)}] {ip.name}  gt={len(gt)} det={len(det)} hit={hit}",
              flush=True)

    def pct(a, b):
        return f'{100*a/b:5.1f}%' if b else '  n/a'

    print(f"\n{'='*60}")
    print("  PaddleOCR 결과")
    print(f"{'='*60}")
    print(f"  정답 치수 총계 : {agg['n_gt']}")
    print(f"  recall         : {pct(agg['hit'], agg['n_gt'])}  "
          f"({agg['hit']}/{agg['n_gt']})")
    print(f"  precision      : {pct(agg['hit'], agg['n_det'])}  "
          f"({agg['hit']}/{agg['n_det']} 검출 숫자)")
    print(f"\n  [노이즈 레벨별 recall]")
    for lvl in ('clean', 'slight', 'noisy', 'heavy'):
        nb = by_noise.get(lvl)
        if nb and nb['n_gt']:
            print(f"    {lvl:>7}: {pct(nb['hit'], nb['n_gt'])}  "
                  f"({nb['hit']}/{nb['n_gt']})")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
