"""
OCR 치수 인식 성능 채점 (recall / precision)
=============================================
라벨된 합성 도면(data/synth)에 현재 OCR을 돌려 정량 평가한다.

지표:
  recall    = 정답 치수값 중 OCR이 찾아낸 비율   (놓친 숫자 파악 = 핵심)
  precision = OCR이 뽑은 치수 숫자 중 정답인 비율 (헛것 파악)

정답: 라벨 JSON의 dimension_values (type이 dimension*/diameter인 것만)
비교: 쉼표→마침표 정규화 후 소수1자리 반올림하여 값 매칭(오차 0.05)

사용법 (repo 루트에서 실행):
  python pipeline/eval_ocr.py                     # data/synth 전체
  python pipeline/eval_ocr.py data/synth/images/drawing_0003.png
"""

import sys
import re
import json
from pathlib import Path
from collections import Counter, defaultdict

import pytesseract

# ocr_real_drawing.py는 legacy/ 폴더로 이동됨 (baseline 재현용 보존)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'legacy'))
from ocr_real_drawing import find_tesseract, extract_real_drawing


NUM_RE = re.compile(r'\d+\.?\d*')


def to_num(s):
    """텍스트에서 첫 숫자 추출 → 소수1자리 float. 없으면 None."""
    s = s.replace(',', '.')
    m = NUM_RE.search(s)
    if not m:
        return None
    try:
        return round(float(m.group()), 1)
    except ValueError:
        return None


def gt_numbers(label):
    """정답 치수값(숫자) 멀티셋."""
    nums = []
    for v in label.get('dimension_values', []):
        n = to_num(v)
        if n is not None:
            nums.append(n)
    return nums


def detected_numbers(result, dims_only=False):
    """
    OCR 검출에서 숫자 멀티셋 추출.
    dims_only=True: 분류가 치수인 것만(precision용, 공정 비교)
    False: 모든 검출(recall 상한 측정)
    """
    src = result['dimensions'] if dims_only else result['all_detections']
    nums = []
    for d in src:
        n = to_num(d['text'])
        if n is not None:
            nums.append(n)
    return nums


def match_multiset(gt, det, tol=0.05):
    """
    gt의 각 값을 det에서 하나씩 소거하며 매칭 수 계산.
    반환: 매칭된 개수
    """
    remaining = list(det)
    matched = 0
    for g in gt:
        for i, d in enumerate(remaining):
            if abs(g - d) <= tol:
                matched += 1
                remaining.pop(i)
                break
    return matched


def eval_one(img_path, label_path):
    with open(label_path, encoding='utf-8') as f:
        label = json.load(f)

    result = extract_real_drawing(img_path)

    gt = gt_numbers(label)
    det_all = detected_numbers(result, dims_only=False)
    det_dim = detected_numbers(result, dims_only=True)

    recall_hit = match_multiset(gt, det_all)          # 관대한 recall(전체 검출 대상)
    prec_hit = match_multiset(det_dim, gt)            # 치수분류 중 정답 수

    return {
        'noise': label.get('noise', '?'),
        'n_gt': len(gt),
        'n_det_all': len(det_all),
        'n_det_dim': len(det_dim),
        'recall_hit': recall_hit,
        'prec_hit': prec_hit,
    }


def main():
    tess = find_tesseract()
    if not tess:
        print("[!] Tesseract 못 찾음")
        sys.exit(1)
    pytesseract.pytesseract.tesseract_cmd = tess

    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/synth/images')

    if arg.is_file():
        pairs = [(arg, Path('data/synth/labels') / f'{arg.stem}.json')]
    else:
        img_dir = arg
        lbl_dir = arg.parent / 'labels'
        pairs = []
        for ip in sorted(img_dir.glob('drawing_*.png')):
            lp = lbl_dir / f'{ip.stem}.json'
            if lp.exists():
                pairs.append((ip, lp))

    print(f"채점 대상: {len(pairs)}장\n")

    agg = Counter()
    by_noise = defaultdict(Counter)

    for i, (ip, lp) in enumerate(pairs):
        try:
            r = eval_one(ip, lp)
        except Exception as e:
            print(f"  [ERROR] {ip.name}: {e}")
            continue

        agg['n_gt'] += r['n_gt']
        agg['recall_hit'] += r['recall_hit']
        agg['prec_hit'] += r['prec_hit']
        agg['n_det_dim'] += r['n_det_dim']

        nb = by_noise[r['noise']]
        nb['n_gt'] += r['n_gt']
        nb['recall_hit'] += r['recall_hit']
        nb['prec_hit'] += r['prec_hit']
        nb['n_det_dim'] += r['n_det_dim']

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(pairs)} 처리...")

    def pct(a, b):
        return f'{100*a/b:5.1f}%' if b else '  n/a'

    print(f"\n{'='*60}")
    print("  전체 결과")
    print(f"{'='*60}")
    print(f"  정답 치수 총계 : {agg['n_gt']}")
    print(f"  recall         : {pct(agg['recall_hit'], agg['n_gt'])}  "
          f"({agg['recall_hit']}/{agg['n_gt']})")
    print(f"  precision      : {pct(agg['prec_hit'], agg['n_det_dim'])}  "
          f"({agg['prec_hit']}/{agg['n_det_dim']} 치수분류 검출)")

    print(f"\n  [노이즈 레벨별 recall]")
    for lvl in ('clean', 'slight', 'noisy', 'heavy'):
        nb = by_noise.get(lvl)
        if nb and nb['n_gt']:
            print(f"    {lvl:>7}: {pct(nb['recall_hit'], nb['n_gt'])}  "
                  f"({nb['recall_hit']}/{nb['n_gt']})")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
