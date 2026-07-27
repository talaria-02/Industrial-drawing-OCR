"""
PaddleOCR 회전-병합 OCR v2
===========================
세로 치수 대응: 원본 + 시계90° + 반시계90° 3패스 OCR 후
회전 좌표를 원본 기준으로 역변환하여 병합, 중복 제거.

도면 세로 텍스트는 보통 아래→위(반시계로 누움)라 시계 회전으로 세워지지만,
반대 방향도 있어 양쪽 모두 돌린다.

사용법:
  python pipeline/paddle_rotate_merge.py data/real/scanA.png   (repo 루트에서 실행)
  python pipeline/paddle_rotate_merge.py data/real/            (폴더 전체)

출력: results/ocr/<이름>_v2.png / .json
"""

import re
import sys
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

OUTPUT_DIR = Path('results/ocr')
MIN_SCORE = 0.5

FONT_PATHS = [r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\malgun.ttf']


def get_font(size):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def build_ocr():
    return PaddleOCR(
        text_detection_model_name='PP-OCRv5_mobile_det',
        text_recognition_model_name='PP-OCRv6_small_rec',
        use_textline_orientation=True, lang='en',
        # 문서방향 자동보정 끔: 회전 패스 좌표가 입력 기준으로 나와야 역변환 가능
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        enable_mkldnn=False)


# ── 좌표 역변환 ──────────────────────────────────────────────
def rotate_expand(img, angle_deg):
    """
    임의각 회전 (캔버스 확장, 잘림 없음).
    반환: (회전 이미지, 역변환 행렬 2x3) — 회전좌표 → 원본좌표 복원용
    """
    h, w = img.shape[:2]
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    rotated = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC,
                             borderValue=(255, 255, 255))
    inv = cv2.invertAffineTransform(M)
    return rotated, inv


def map_point_cw(xp, yp, orig_h):
    """시계90° 회전 이미지의 점 → 원본 좌표. x=y', y=h-1-x'"""
    return yp, orig_h - 1 - xp


def map_point_ccw(xp, yp, orig_w):
    """반시계90° 회전 이미지의 점 → 원본 좌표. x=w-1-y', y=x'"""
    return orig_w - 1 - yp, xp


# 획 문자만으로 구성된 텍스트 (치수선/점선 오인식 후보)
STROKE_ONLY_RE = re.compile(r'^[1lI|/\\\-_—–·.,:;\'"`~ ]+$')
# 점선 낱개 대시가 단일 문자로 읽힌 경우 (T 포함 — 단독 T 치수는 없음)
SINGLE_STROKE_RE = re.compile(r'^[1lIT|/\\\-_—–+~·^°]$')
HAS_ALNUM_RE = re.compile(r'[A-Za-z0-9]')


def is_line_artifact(text, poly):
    """
    점선/치수선 오인식 판정.
    1) 단일 획-문자 ('|','1','T' 등 단독) — 점선 대시 하나가 읽힌 것
    2) 영숫자가 전혀 없는 텍스트 ('-->+-' 등)
    3) 획 문자 3자 이상 반복 ('111','---')
    4) 획 문자 조합 + 길쭉한 박스
    """
    if SINGLE_STROKE_RE.match(text):
        return True
    if not HAS_ALNUM_RE.search(text):
        return True
    if not STROKE_ONLY_RE.match(text):
        return False
    if len(text.replace(' ', '')) >= 3:      # '111', '---' 등 반복
        return True
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    aspect = max(w, h) / max(1.0, min(w, h))
    return aspect > 4                         # 진짜 글자보다 훨씬 길쭉


def run_pass(ocr, img, mapper, orient):
    """한 방향 OCR 후 (원본좌표 poly, text, score, orient) 리스트."""
    res = ocr.predict(img)[0]
    dets = []
    n_dropped = 0
    for t, s, poly in zip(res['rec_texts'], res['rec_scores'],
                          res['rec_polys']):
        s = float(s)
        if s < MIN_SCORE or not t.strip():
            continue
        if is_line_artifact(t.strip(), np.array(poly)):
            n_dropped += 1
            continue
        pts = [mapper(float(x), float(y)) for x, y in np.array(poly)]
        dets.append({'text': t.strip(), 'score': round(s, 3),
                     'poly': [[round(x, 1), round(y, 1)] for x, y in pts],
                     'orient': orient})
    if n_dropped:
        print(f"      (선 오인식 {n_dropped}개 제거)", flush=True)
    return dets


# ── 병합/중복 제거 ───────────────────────────────────────────
def bbox_of(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def center_of(poly):
    x1, y1, x2, y2 = bbox_of(poly)
    return (x1 + x2) / 2, (y1 + y2) / 2


def overlap_ratio(a, b):
    """작은 박스 기준 교집합 비율."""
    ax1, ay1, ax2, ay2 = bbox_of(a)
    bx1, by1, bx2, by2 = bbox_of(b)
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    small = max(1e-6, min(area_a, area_b))
    return inter / small


def merge_dets(all_dets):
    """신뢰도 내림차순으로 훑으며 많이 겹치는 검출 제거."""
    all_dets.sort(key=lambda d: d['score'], reverse=True)
    kept = []
    for d in all_dets:
        dup = False
        for k in kept:
            if overlap_ratio(d['poly'], k['poly']) > 0.6:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


# ── 메인 추출 ────────────────────────────────────────────────
def extract(ocr, image_path, clean_lines=False, extra_angles=()):
    img = cv2.imread(str(image_path))
    if img is None:
        # 한글 경로 대응
        arr = np.fromfile(str(image_path), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if clean_lines == 'fk':
        # 전처리 v3: F-K식 공선 체인 점선 제거 (각도 무관)
        from remove_lines import remove_dashed_lines_fk
        img = remove_dashed_lines_fk(img)
    elif clean_lines:
        # 전처리 v2: morphology 점선/치수선 제거 (숫자 보호)
        from remove_lines import remove_dashed_lines
        img = remove_dashed_lines(img)

    h, w = img.shape[:2]

    passes = [
        (img, lambda x, y: (x, y), 'h'),
        (cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
         lambda x, y: map_point_cw(x, y, h), 'v_cw'),
        (cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
         lambda x, y: map_point_ccw(x, y, w), 'v_ccw'),
    ]

    # 추가 각도 패스 (대각선 치수용, --angles)
    for ang in extra_angles:
        rot, inv = rotate_expand(img, ang)

        def make_mapper(inv_m):
            def mp(x, y):
                return (inv_m[0, 0] * x + inv_m[0, 1] * y + inv_m[0, 2],
                        inv_m[1, 0] * x + inv_m[1, 1] * y + inv_m[1, 2])
            return mp

        passes.append((rot, make_mapper(inv), f'a{ang:+d}'))

    all_dets = []
    for im, mapper, orient in passes:
        dets = run_pass(ocr, im, mapper, orient)
        print(f"    pass {orient}: {len(dets)}개", flush=True)
        all_dets.extend(dets)

    merged = merge_dets(all_dets)
    print(f"    병합 후: {len(merged)}개", flush=True)
    return merged


# ── 시각화 ───────────────────────────────────────────────────
COLORS = {'h': (255, 0, 0), 'v_cw': (0, 130, 0), 'v_ccw': (0, 0, 220)}


def visualize(image_path, dets, out_path):
    im = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(im)
    font = get_font(max(14, im.width // 60))

    for d in dets:
        color = COLORS.get(d['orient'], (128, 128, 128))
        pts = [tuple(p) for p in d['poly']]
        draw.polygon(pts, outline=color, width=2)
        x1, y1, _, _ = bbox_of(d['poly'])
        label = f"{d['text']} ({d['score']:.2f})"
        tb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2],
                       fill=(255, 255, 160))
        draw.text((x1, y1), label, fill=color, font=font)

    im.save(out_path)


def main():
    global OUTPUT_DIR
    argv = sys.argv[1:]
    clean_lines = 'fk' if '--fk' in argv else ('--clean' in argv)
    flag_vals = set()
    if '--out' in argv:
        OUTPUT_DIR = Path(argv[argv.index('--out') + 1])
        flag_vals.add(argv[argv.index('--out') + 1])
    extra_angles = ()
    if '--angles' in argv:
        s = argv[argv.index('--angles') + 1]
        flag_vals.add(s)
        extra_angles = tuple(int(a) for a in s.split(','))
    args = [a for a in argv
            if a not in ('--clean', '--fk', '--out', '--angles')
            and a not in flag_vals]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = Path(args[0]) if args else Path('data/real')
    if clean_lines:
        print("[전처리] 점선 제거 활성 (--clean)")

    if target.is_file():
        images = [target]
    else:
        images = sorted(p for p in target.glob('*.*')
                        if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))

    ocr = build_ocr()
    print(f"대상: {len(images)}장 (3패스 회전-병합)")

    for ip in images:
        print(f"\n--- {ip.name} ---", flush=True)
        dets = extract(ocr, ip, clean_lines=clean_lines,
                       extra_angles=extra_angles)

        out_img = OUTPUT_DIR / f'{ip.stem}_v2.png'
        visualize(ip, dets, out_img)

        out_json = OUTPUT_DIR / f'{ip.stem}_v2.json'
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({'image': ip.name, 'n': len(dets),
                       'detections': dets}, f, indent=2, ensure_ascii=False)
        print(f"    저장: {out_img.name}")

    print("\n완료.")


if __name__ == '__main__':
    main()
