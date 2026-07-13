"""
실제 도면 OCR 추출기 — OpenCV 전처리 + Tesseract OCR
====================================================
합성 도면과 다른 실제 도면의 특성을 반영:
  - 세로(90도 회전) 치수 텍스트 처리
  - 공차 표기(±0.1, H7 등) 인식
  - 스캔 노이즈/해칭 패턴 대응
  - 복잡한 타이틀블록 파싱
  - 다중 뷰(정면/측면/등각) 도면 지원

실행:
  python ocr_real_drawing.py [이미지경로]
  python ocr_real_drawing.py output_v4/real_images/  (폴더 전체)
"""

import sys
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract


# ── Tesseract 설정 ───────────────────────────────────────────

TESSERACT_PATHS = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
]


def find_tesseract():
    import shutil
    path = shutil.which('tesseract')
    if path:
        return path
    import os
    username = os.getenv('USERNAME', '')
    for p in TESSERACT_PATHS:
        if Path(p).exists():
            return p
    user_path = rf'C:\Users\{username}\AppData\Local\Tesseract-OCR\tesseract.exe'
    if Path(user_path).exists():
        return user_path
    return None


# ── 전처리 파이프라인 ────────────────────────────────────────

def preprocess_real_drawing(img, save_dir=None, stem=''):
    """
    실제 도면용 전처리.
    스캔 노이즈, 불균일 조명, 해칭 패턴 등에 대응.
    save_dir이 지정되면 각 단계별 중간 이미지를 저장.
    """
    def _save(name, image):
        if save_dir:
            path = save_dir / f'{stem}_step_{name}.png'
            cv2.imwrite(str(path), image)
            print(f"    [save] {path.name}")

    # 0) 원본
    _save('0_original', img)

    # 1) 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _save('1_gray', gray)

    # 사용자의 요청에 따라 전처리 과정을 생략하고 바로 반환합니다.
    return gray


def preprocess_for_rotated_text(img_gray):
    """세로 텍스트용 전처리 (90도 회전 후 OCR)"""
    # 반시계 90도 회전 → 세로 텍스트가 수평이 됨
    rotated = cv2.rotate(img_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated


# ── OCR 엔진 ─────────────────────────────────────────────────

def ocr_full_image(binary, config=None):
    """
    전체 이미지에 Tesseract image_to_data 적용.
    바운딩박스 + 텍스트 + 신뢰도 반환.
    """
    if config is None:
        # 실제 도면용: whitelist를 넓혀 영문/특수문자도 포함
        config = '--psm 11 --oem 3'

    data = pytesseract.image_to_data(
        binary, config=config,
        output_type=pytesseract.Output.DICT
    )

    detections = []
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        if not text or conf < 15:
            continue

        detections.append({
            'text': text,
            'confidence': conf,
            'bbox': {
                'x': data['left'][i],
                'y': data['top'][i],
                'w': data['width'][i],
                'h': data['height'][i],
            },
            'orientation': 'horizontal',
        })

    return detections


def ocr_rotated(binary):
    """
    이미지를 90도 회전하여 세로 텍스트 인식.
    검출된 좌표를 원본 기준으로 역변환.
    """
    h, w = binary.shape[:2]

    # 반시계 90도 회전
    rotated = cv2.rotate(binary, cv2.ROTATE_90_COUNTERCLOCKWISE)

    config = '--psm 11 --oem 3'
    data = pytesseract.image_to_data(
        rotated, config=config,
        output_type=pytesseract.Output.DICT
    )

    detections = []
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        if not text or conf < 30:
            continue

        # 회전 좌표 → 원본 좌표 역변환
        # 반시계 90도 회전 시: (rx, ry) → 원본 (ry, w - rx - rw)
        rx = data['left'][i]
        ry = data['top'][i]
        rw = data['width'][i]
        rh = data['height'][i]

        orig_x = ry
        orig_y = w - rx - rw
        orig_w = rh
        orig_h = rw

        detections.append({
            'text': text,
            'confidence': conf,
            'bbox': {
                'x': orig_x,
                'y': orig_y,
                'w': orig_w,
                'h': orig_h,
            },
            'orientation': 'vertical',
        })

    return detections


# ── 치수 텍스트 분류 ─────────────────────────────────────────

DIMENSION_PATTERNS = [
    # 숫자 + 공차: "50±0.1", "30+0.1", "20±0.1"
    (r'^(\d+\.?\d*)\s*[±+\-]\s*\d+\.?\d*$', 'dimension_toleranced'),
    # 숫자 + 끼워맞춤: "30H9", "20f7", "10H7"
    (r'^(\d+\.?\d*)\s*[A-Za-z]\d+$', 'dimension_fit'),
    # 지름 기호: "Ø10", "φ3.6"
    (r'^[OoØø0Dd]\s*(\d+\.?\d*)$', 'diameter'),
    # NxM 패턴: "3.6x2", "2x45°"
    (r'^(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)', 'multiplied'),
    # 분수: "3/8"
    (r'^(\d+)/(\d+)$', 'fraction'),
    # 순수 숫자: "50", "14", "28"
    (r'^(\d+\.?\d*)$', 'dimension'),
    # 각도: "90°", "45°"
    (r'^(\d+\.?\d*)\s*[°o]$', 'angle'),
]


def classify_text(text):
    """
    추출된 텍스트를 도면 요소 유형으로 분류.
    반환: {'type': ..., 'value': ..., 'raw': ...} 또는 None
    """
    text = text.strip()
    if not text:
        return None

    # 치수 패턴 순서대로 매칭
    for pattern, dim_type in DIMENSION_PATTERNS:
        m = re.match(pattern, text)
        if m:
            return {
                'type': dim_type,
                'raw': text,
                'value': m.group(1),
            }

    # 뷰 라벨: "Front", "Left", "Top", "Corte AB"
    if re.match(r'^(Front|Left|Top|Right|Bottom|Isometric|Corte|Section|Detail)\b',
                text, re.IGNORECASE):
        return {'type': 'view_label', 'raw': text, 'value': text}

    # 스케일: "2:1", "1:2"
    if re.match(r'^\d+:\d+$', text):
        return {'type': 'scale', 'raw': text, 'value': text}

    # 기타 영문 텍스트 (주석, 라벨 등) — 3글자 이상만
    if re.match(r'^[A-Za-z]', text) and len(text) >= 3:
        return {'type': 'annotation', 'raw': text, 'value': text}

    # 단일 문자 (참조 기호: A, B, C, D 등)
    if re.match(r'^[A-D]$', text):
        return {'type': 'reference', 'raw': text, 'value': text}

    # 2글자 이하 영문은 노이즈일 가능성이 높으므로 무시
    if re.match(r'^[A-Za-z]{1,2}$', text):
        return None

    return {'type': 'unknown', 'raw': text, 'value': text}


def is_dimension(cls):
    """치수 관련 유형인지 확인"""
    if cls is None:
        return False
    return cls['type'] in (
        'dimension', 'dimension_toleranced', 'dimension_fit',
        'diameter', 'multiplied', 'fraction', 'angle'
    )


# ── 중복 제거 ────────────────────────────────────────────────

def deduplicate_detections(horiz, vert):
    """
    수평/수직 OCR 결과를 병합하고 중복 제거.
    같은 영역에서 같은 텍스트가 검출되면 신뢰도가 높은 것만 남김.
    """
    all_dets = horiz + vert
    if not all_dets:
        return []

    # 신뢰도 내림차순 정렬
    all_dets.sort(key=lambda d: d['confidence'], reverse=True)

    kept = []
    for det in all_dets:
        bbox = det['bbox']
        cx = bbox['x'] + bbox['w'] / 2
        cy = bbox['y'] + bbox['h'] / 2

        is_dup = False
        for k in kept:
            kb = k['bbox']
            kcx = kb['x'] + kb['w'] / 2
            kcy = kb['y'] + kb['h'] / 2

            # 중심점 거리 기반 중복 판정
            dist = ((cx - kcx) ** 2 + (cy - kcy) ** 2) ** 0.5
            # 같은 텍스트이고 근접하면 중복
            if det['text'] == k['text'] and dist < max(bbox['w'], bbox['h'], kb['w'], kb['h']):
                is_dup = True
                break
            # 다른 텍스트라도 바운딩박스가 크게 겹치면 중복
            if dist < 20 and abs(bbox['w'] - kb['w']) < 10:
                is_dup = True
                break

        if not is_dup:
            kept.append(det)

    return kept


# ── 타이틀블록 검출 ──────────────────────────────────────────

def detect_titleblock(img):
    """
    실제 도면의 타이틀블록(표 형태) 검출.
    주로 이미지 우하단에 위치.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 하단 30%, 우측 70% 영역 탐색
    y_start = int(h * 0.65)
    x_start = int(w * 0.3)
    roi = gray[y_start:, x_start:]

    # 엣지 검출
    edges = cv2.Canny(roi, 50, 150)

    # 수평/수직 선분 검출 (HoughLinesP)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                            minLineLength=50, maxLineGap=10)

    if lines is None:
        return None

    # 수평/수직 선분이 많이 모여 있는 영역 = 표(타이틀블록)
    # OpenCV 5: lines shape may be (N,4) or (N,1,4)
    lines_2d = lines.reshape(-1, 4)

    h_lines = []
    v_lines = []
    for seg in lines_2d:
        x1, y1, x2, y2 = seg
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 10:  # 수평
            h_lines.append(seg)
        elif angle > 80:  # 수직
            v_lines.append(seg)

    if len(h_lines) < 3 or len(v_lines) < 2:
        return None

    # 선분들의 바운딩 영역 계산
    all_pts = np.array(h_lines + v_lines).reshape(-1, 2)
    min_x = all_pts[:, 0].min() + x_start
    min_y = all_pts[:, 1].min() + y_start
    max_x = all_pts[:, 0].max() + x_start
    max_y = all_pts[:, 1].max() + y_start

    return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))


def ocr_titleblock(img, box):
    """타이틀블록 영역에서 OCR"""
    x, y, w, h = box
    pad = 5
    img_h, img_w = img.shape[:2]
    roi = img[max(0, y - pad):min(img_h, y + h + pad),
              max(0, x - pad):min(img_w, x + w + pad)]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 업스케일
    scale = max(2, 400 // max(gray.shape[0], 1))
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(upscaled, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = '--psm 6 --oem 3'  # PSM 6: 균일한 텍스트 블록
    text = pytesseract.image_to_string(binary, config=config).strip()
    return text


def parse_titleblock(text):
    """타이틀블록 텍스트에서 메타데이터 추출"""
    result = {}

    # Title
    title_match = re.search(r'(?:Title|TITLE)[:\s]*(.+)', text, re.IGNORECASE)
    if title_match:
        result['title'] = title_match.group(1).strip()

    # 도면 제목 (대문자로 된 이름)
    ejercicio_match = re.search(r'(EJERCICIO\s+\d+\s+\w+)', text, re.IGNORECASE)
    if ejercicio_match:
        result['title'] = ejercicio_match.group(1).strip()

    # Scale
    scale_match = re.search(r'(?:SCALE|Scale)[:\s]*(\d+:\d+)', text, re.IGNORECASE)
    if scale_match:
        result['scale'] = scale_match.group(1)

    # Size
    size_match = re.search(r'(?:SIZE)\s*([A-Z]\d?)', text, re.IGNORECASE)
    if size_match:
        result['size'] = size_match.group(1)

    # Date
    date_match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})', text)
    if date_match:
        result['date'] = date_match.group(1)

    # Drawn by
    drawn_match = re.search(r'(?:DRAWN|Drawn)[:\s]*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if drawn_match:
        result['drawn_by'] = drawn_match.group(1).strip()

    # Material
    mat_match = re.search(r'(?:MATERIAL|Material)[:\s]*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if mat_match:
        result['material'] = mat_match.group(1).strip()

    # Modeling Practice
    practice_match = re.search(r'Modeling\s+Practice\s+Drawings?\s+(\d+)', text, re.IGNORECASE)
    if practice_match:
        result['title'] = f'Modeling Practice Drawings {practice_match.group(1)}'

    return result


# ── 전체 OCR 파이프라인 ──────────────────────────────────────

def extract_real_drawing(image_path, output_dir=None):
    """
    실제 도면 이미지에서 OCR 추출.
    수평 + 수직 텍스트 모두 인식.
    output_dir이 지정되면 전처리 단계별 이미지를 해당 폴더에 저장.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    h, w = img.shape[:2]
    stem = Path(image_path).stem
    print(f"  Image: {w}x{h}")

    # 1) 전처리 (단계별 이미지 저장)
    print("  Preprocessing...")
    binary = preprocess_real_drawing(img, save_dir=output_dir, stem=stem)

    # 2) 수평 텍스트 OCR
    print("  Horizontal OCR...")
    horiz_dets = ocr_full_image(binary)
    print(f"    -> {len(horiz_dets)} detections")

    # 3) 수직 텍스트 OCR (90도 회전)
    print("  Vertical OCR...")
    vert_dets = ocr_rotated(binary)
    print(f"    -> {len(vert_dets)} detections")

    # 회전 이미지도 저장
    if output_dir:
        rotated_img = cv2.rotate(binary, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rot_path = output_dir / f'{stem}_step_6_rotated.png'
        cv2.imwrite(str(rot_path), rotated_img)
        print(f"    [save] {rot_path.name}")

    # 4) 중복 제거 및 병합
    all_dets = deduplicate_detections(horiz_dets, vert_dets)
    print(f"  After dedup: {len(all_dets)} detections")

    # 5) 분류
    dimensions = []
    annotations = []
    for det in all_dets:
        cls = classify_text(det['text'])
        if cls:
            det['classification'] = cls
            if is_dimension(cls):
                dimensions.append(det)
            elif cls['type'] in ('view_label', 'annotation', 'scale'):
                annotations.append(det)

    # 6) 타이틀블록 검출
    titleblock_box = detect_titleblock(img)
    titleblock_info = {}
    titleblock_raw = ''
    if titleblock_box:
        print(f"  Titleblock: {titleblock_box}")
        titleblock_raw = ocr_titleblock(img, titleblock_box)
        titleblock_info = parse_titleblock(titleblock_raw)
    else:
        print("  [!] No titleblock found")

    result = {
        'image': Path(image_path).name,
        'image_size': {'width': w, 'height': h},
        'titleblock': {
            'bbox': titleblock_box,
            'raw_text': titleblock_raw,
            'parsed': titleblock_info,
        },
        'dimensions': dimensions,
        'annotations': annotations,
        'all_detections': all_dets,
    }

    return result


# ── 결과 출력 ────────────────────────────────────────────────

def print_results(result):
    print("\n" + "=" * 65)
    print(f"  Real Drawing OCR: {result['image']}")
    print(f"  Size: {result['image_size']['width']}x{result['image_size']['height']}")
    print("=" * 65)

    # 타이틀블록
    tb = result['titleblock']
    print("\n[Titleblock]")
    if tb['parsed']:
        for k, v in tb['parsed'].items():
            print(f"  {k:>12}: {v}")
    elif tb['raw_text']:
        # 첫 3줄만 표시
        lines = tb['raw_text'].split('\n')[:3]
        for line in lines:
            print(f"  {line}")
    else:
        print("  (not detected)")

    # 치수
    print(f"\n[Dimensions] ({len(result['dimensions'])})")
    for d in result['dimensions']:
        cls = d.get('classification', {})
        bbox = d['bbox']
        orient = d.get('orientation', '?')[0].upper()
        print(f"  {cls.get('raw', d['text']):>15}  "
              f"type={cls.get('type', '?'):>22}  "
              f"conf={d['confidence']:>3}%  "
              f"orient={orient}  "
              f"pos=({bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']})")

    # 주석
    if result['annotations']:
        print(f"\n[Annotations] ({len(result['annotations'])})")
        for d in result['annotations']:
            cls = d.get('classification', {})
            print(f"  {cls.get('raw', d['text']):>20}  type={cls.get('type', '?')}")

    # 전체 검출 수
    print(f"\n[Total detections: {len(result['all_detections'])}]")
    print()


def draw_result_image(image_path, result, output_path=None):
    """시각화 — 치수/주석/타이틀블록 바운딩박스 표시"""
    img = cv2.imread(str(image_path))

    # 색상 정의
    COLOR_DIM = (0, 0, 255)       # 빨강 — 치수
    COLOR_DIM_V = (0, 100, 255)   # 주황 — 세로 치수
    COLOR_ANNOT = (255, 150, 0)   # 파랑 — 주석
    COLOR_TB = (255, 0, 0)        # 파랑 — 타이틀블록
    COLOR_OTHER = (160, 160, 160) # 회색 — 기타

    # 타이틀블록
    tb = result['titleblock']
    if tb['bbox']:
        x, y, w, h = tb['bbox']
        cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_TB, 2)
        cv2.putText(img, 'TITLEBLOCK', (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TB, 2)

    # 치수 바운딩박스
    for d in result['dimensions']:
        bbox = d['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        color = COLOR_DIM_V if d.get('orientation') == 'vertical' else COLOR_DIM
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = d.get('classification', {}).get('raw', d['text'])
        cv2.putText(img, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 주석
    for d in result['annotations']:
        bbox = d['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_ANNOT, 1)

    if output_path is None:
        stem = Path(image_path).stem
        output_path = Path(image_path).parent / f'{stem}_ocr_result.png'

    cv2.imwrite(str(output_path), img)
    print(f"  Result image: {output_path}")
    return output_path


# ── 메인 ─────────────────────────────────────────────────────

def main():
    # Tesseract 설정
    tess_path = find_tesseract()
    if tess_path:
        pytesseract.pytesseract.tesseract_cmd = tess_path
        print(f"Tesseract: {tess_path}")
    else:
        print("[!] Tesseract not found.")
        print("  Install: winget install UB-Mannheim.TesseractOCR")
        sys.exit(1)

    # 경로 결정
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = Path('./output_v4/real_images/')

    # 출력 디렉터리: real_images_result
    output_dir = target.parent / 'real_images_result' if target.is_dir() else target.parent
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # 이미지 목록 (_ocr_result, _step_ 파일 제외)
    if target.is_dir():
        all_imgs = sorted(target.glob('*.png')) + sorted(target.glob('*.jpg'))
        images = [p for p in all_imgs
                  if '_ocr_result' not in p.stem
                  and '_ocr' not in p.stem
                  and '_step_' not in p.stem]
    elif target.is_file():
        images = [target]
    else:
        print(f"Not found: {target}")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  Real Drawing OCR — {len(images)} images")
    print(f"  Output: {output_dir}")
    print(f"{'='*65}\n")

    all_results = []

    for img_path in images:
        print(f"\n--- Processing: {img_path.name} ---")
        try:
            result = extract_real_drawing(img_path, output_dir=output_dir)
            print_results(result)

            # 시각화 이미지 저장 → output_dir에
            result_img_path = output_dir / f'{img_path.stem}_ocr_result.png'
            draw_result_image(img_path, result, output_path=result_img_path)

            # JSON 저장 → output_dir에
            json_path = output_dir / f'{img_path.stem}_ocr.json'
            result_json = json.loads(json.dumps(result, default=str))
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, indent=2, ensure_ascii=False)
            print(f"  Result JSON: {json_path}")

            all_results.append(result)

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    # 요약
    print(f"\n{'='*65}")
    print(f"  Summary: {len(all_results)}/{len(images)} processed")
    total_dims = sum(len(r['dimensions']) for r in all_results)
    print(f"  Total dimensions extracted: {total_dims}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
