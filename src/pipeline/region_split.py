"""
도면 영역 분리: 프레임 / 타이틀블록(메타데이터) / 그림 영역
=============================================================
eDOCr 방식 차용:
  1) 프레임: 이미지 크기 대비 70%+ 길이의 수평/수직 장선 → 외곽 테두리
  2) 타이틀블록: 프레임 내부 하단부의 선 밀집(표) 영역
  3) OCR 검출 분류: 타이틀블록 내부/프레임 외부 = meta, 나머지 = drawing

사용법:
  python pipeline/region_split.py data/real/scanA.png   (repo 루트에서 실행)
  python pipeline/region_split.py data/real/            (폴더 전체)
  # results/ocr/<stem>_v2.json 있으면 검출 분류까지 수행

출력: results/region/<이름>_region.png / _region.json
"""

import sys
import json
from pathlib import Path

import cv2
import numpy as np

OUTPUT_DIR = Path('results/region')
DET_DIR = Path('results/ocr')


def imread_kr(path):
    """한글 경로 대응 이미지 로드"""
    arr = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def find_long_lines(binary, min_len, max_gap=8):
    """
    HoughLinesP로 장선 검출 (미세 기울기 허용 — 스캔 기울기 대응).
    반환: (h_segs, v_segs)  각 원소 (x1,y1,x2,y2,length)
    """
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=100,
                            minLineLength=int(min_len), maxLineGap=max_gap)
    h_segs, v_segs = [], []
    if lines is None:
        return h_segs, v_segs
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        length = float(np.hypot(x2 - x1, y2 - y1))
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if ang < 5 or ang > 175:          # 수평 (±5도)
            h_segs.append((int(x1), int(y1), int(x2), int(y2), length))
        elif 85 < ang < 95:               # 수직 (±5도)
            v_segs.append((int(x1), int(y1), int(x2), int(y2), length))
    return h_segs, v_segs


def detect_frame(binary, w, h):
    """
    외곽 프레임 검출: 이미지 60%+ 길이의 수평/수직 장선 중
    최외곽 위치의 것들로 테두리 사각형 추정.
    실패 시 이미지 전체 반환.
    """
    # 블러/노이즈로 약해진 테두리 보강
    fat = cv2.dilate(binary, np.ones((3, 3), np.uint8))
    h_segs, v_segs = find_long_lines(fat, min_len=0.6 * w)
    _, v_segs2 = find_long_lines(fat, min_len=0.6 * h)
    v_segs = v_segs + v_segs2

    h_long = [s for s in h_segs if s[4] > 0.6 * w]
    v_long = [s for s in v_segs if s[4] > 0.6 * h]

    if len(h_long) >= 2 and len(v_long) >= 2:
        ys = [(s[1] + s[3]) / 2 for s in h_long]
        xs = [(s[0] + s[2]) / 2 for s in v_long]
        top, bottom = int(min(ys)), int(max(ys))
        left, right = int(min(xs)), int(max(xs))
        if (right - left) > w * 0.5 and (bottom - top) > h * 0.5:
            return (left, top, right, bottom), True
    return (0, 0, w - 1, h - 1), False


def detect_titleblock(binary, frame, w, h):
    """
    타이틀블록 검출: 프레임 내부 하단 40% 영역에서
    수평 장선(프레임 폭 25%+)이 밀집한 최하단 밴드를 표로 간주.
    반환: (x1,y1,x2,y2) 또는 None
    """
    fx1, fy1, fx2, fy2 = frame
    fw = fx2 - fx1
    fh = fy2 - fy1

    y_start = fy1 + int(fh * 0.75)   # 타이틀블록은 통상 하단 25% 이내
    roi = binary[y_start:fy2, fx1:fx2]
    if roi.size == 0:
        return None

    # ROI 내 수평 장선 (프레임 폭의 25% 이상, 기울기 허용)
    h_segs, _ = find_long_lines(roi, min_len=0.25 * fw)
    # 프레임 하단선 자체(바닥 2% 이내) 제외
    cands = [s for s in h_segs
             if s[4] > 0.25 * fw
             and (fy2 - y_start) - (s[1] + s[3]) / 2 > fh * 0.02
             # 표 상단선은 프레임 우측 경계까지 닿음 (우하단 코너 관례)
             and max(s[0], s[2]) > fw * 0.9]
    if not cands:
        return None

    # 타이틀블록 = 하단 영역 최상단 장수평선 ~ 프레임 바닥
    top_seg = min(cands, key=lambda s: (s[1] + s[3]) / 2)
    tb_top = y_start + int((top_seg[1] + top_seg[3]) / 2)
    tb_bottom = fy2

    # 높이 sanity: 프레임의 3%~25% 사이여야 표로 인정
    tb_h = tb_bottom - tb_top
    if not (fh * 0.03 < tb_h < fh * 0.25):
        return None

    # 표의 좌우 범위 = 상단 경계선의 x범위
    # (우측 코너형 표는 우측만, 전폭형 표는 전체가 자연히 잡힘)
    tb_left = fx1 + min(top_seg[0], top_seg[2])
    tb_right = fx1 + max(top_seg[0], top_seg[2])
    if tb_right - tb_left < fw * 0.3:      # 너무 좁으면 잡선 → 전체 폭
        tb_left, tb_right = fx1, fx2

    return (int(tb_left), int(tb_top), int(tb_right), int(tb_bottom))


def detect_titleblock_right(binary, frame, w, h):
    """
    우측 기둥형 타이틀블록 검출.
    프레임 우측 40% 구역에서 프레임 높이 85%+ 수직 장선(표 좌측 경계)을 찾음.
    반환: (x1,y1,x2,y2) 또는 None
    """
    fx1, fy1, fx2, fy2 = frame
    fw = fx2 - fx1
    fh = fy2 - fy1

    x_start = fx1 + int(fw * 0.6)
    roi = binary[fy1:fy2, x_start:fx2]
    if roi.size == 0:
        return None

    _, v_segs = find_long_lines(roi, min_len=0.5 * fh, max_gap=12)
    # 프레임 우측 경계 자체(우측 2% 이내)는 제외, 높이 85% 이상만
    cands = [s for s in v_segs
             if s[4] > 0.85 * fh
             and (fx2 - x_start) - (s[0] + s[2]) / 2 > fw * 0.02]
    if not cands:
        return None

    left_seg = min(cands, key=lambda s: (s[0] + s[2]) / 2)
    tb_left = x_start + int((left_seg[0] + left_seg[2]) / 2)

    # 폭 sanity: 프레임의 5%~30%
    tb_w = fx2 - tb_left
    if not (fw * 0.05 < tb_w < fw * 0.30):
        return None

    return (int(tb_left), int(fy1), int(fx2), int(fy2))


def split_regions(image_path):
    img = imread_kr(image_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    frame, frame_found = detect_frame(binary, w, h)
    # 타이틀블록은 프레임 있는 도면에만 존재 — 오검출 방지
    # 하단형 우선, 없으면 우측 기둥형 시도
    titleblock = None
    if frame_found:
        titleblock = detect_titleblock(binary, frame, w, h)
        if titleblock is None:
            titleblock = detect_titleblock_right(binary, frame, w, h)

    return img, frame, frame_found, titleblock


def classify_point(cx, cy, frame, titleblock):
    """검출 중심점 → 'meta' / 'drawing'"""
    fx1, fy1, fx2, fy2 = frame
    if not (fx1 <= cx <= fx2 and fy1 <= cy <= fy2):
        return 'meta'          # 프레임 밖
    if titleblock:
        tx1, ty1, tx2, ty2 = titleblock
        if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
            return 'meta'      # 타이틀블록 안
    return 'drawing'


def process(image_path):
    img, frame, frame_found, titleblock = split_regions(image_path)
    stem = Path(image_path).stem

    vis = img.copy()
    fx1, fy1, fx2, fy2 = frame
    cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), (0, 180, 0), 3)
    cv2.putText(vis, 'FRAME', (fx1 + 5, fy1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 0), 2)
    if titleblock:
        tx1, ty1, tx2, ty2 = titleblock
        cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), (255, 0, 0), 3)
        cv2.putText(vis, 'TITLEBLOCK', (tx1 + 5, ty1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    # OCR 검출 결과 있으면 분류 표시
    det_json = DET_DIR / f'{stem}_v2.json'
    n_meta = n_draw = 0
    classified = []
    if det_json.exists():
        with open(det_json, encoding='utf-8') as f:
            data = json.load(f)
        for d in data['detections']:
            xs = [p[0] for p in d['poly']]
            ys = [p[1] for p in d['poly']]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            region = classify_point(cx, cy, frame, titleblock)
            d['region'] = region
            classified.append(d)
            color = (0, 0, 255) if region == 'drawing' else (255, 0, 255)
            if region == 'drawing':
                n_draw += 1
            else:
                n_meta += 1
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    out_img = OUTPUT_DIR / f'{stem}_region.png'
    cv2.imencode('.png', vis)[1].tofile(str(out_img))

    result = {
        'image': Path(image_path).name,
        'frame': [int(v) for v in frame], 'frame_found': frame_found,
        'titleblock': [int(v) for v in titleblock] if titleblock else None,
        'n_meta': n_meta, 'n_drawing': n_draw,
        'detections': classified,
    }
    with open(OUTPUT_DIR / f'{stem}_region.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=int)

    tb_str = 'O' if titleblock else 'X'
    print(f"  {Path(image_path).name}: frame={'O' if frame_found else 'X(전체)'} "
          f"titleblock={tb_str}  meta={n_meta} drawing={n_draw}")
    return result


def main():
    global OUTPUT_DIR, DET_DIR
    argv = sys.argv[1:]
    consumed = set()
    if '--det' in argv:
        i = argv.index('--det')
        DET_DIR = Path(argv[i + 1])
        consumed |= {argv[i], argv[i + 1]}
    if '--out' in argv:
        i = argv.index('--out')
        OUTPUT_DIR = Path(argv[i + 1])
        consumed |= {argv[i], argv[i + 1]}
    args = [a for a in argv if a not in consumed]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = Path(args[0]) if args else Path('data/real')

    if target.is_file():
        images = [target]
    else:
        images = sorted(p for p in target.glob('*.*')
                        if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))

    print(f"영역 분리: {len(images)}장")
    for ip in images:
        try:
            process(ip)
        except Exception as e:
            print(f"  [ERROR] {ip.name}: {e}")

    print("완료.")


if __name__ == '__main__':
    main()
