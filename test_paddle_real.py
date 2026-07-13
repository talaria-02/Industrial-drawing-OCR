"""
실제 도면 5장에 PaddleOCR 적용 → 시각화
==========================================
검출 박스 + 인식 텍스트 + 신뢰도를 원본 위에 그려 저장.

출력: paddle_real_results/
  <이름>_paddle.png   시각화 이미지
  <이름>_paddle.json  검출 결과(텍스트/신뢰도/박스)
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

INPUT_DIR = Path('real_images')
OUTPUT_DIR = Path('paddle_real_results')
OUTPUT_DIR.mkdir(exist_ok=True)

# ± Ø ° 렌더 가능한 폰트
FONT_PATHS = [
    r'C:\Windows\Fonts\arial.ttf',
    r'C:\Windows\Fonts\malgun.ttf',
]


def get_font(size):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ocr = PaddleOCR(
        text_detection_model_name='PP-OCRv5_mobile_det',
        text_recognition_model_name='PP-OCRv5_mobile_rec',
        use_textline_orientation=True, lang='en',
        # 문서방향 보정/왜곡펴기 끔: 반환 좌표가 원본 기준이어야 시각화 정확
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        enable_mkldnn=False)

    images = sorted(p for p in INPUT_DIR.glob('*.*')
                    if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    print(f"대상: {len(images)}장")

    for ip in images:
        print(f"\n--- {ip.name} ---", flush=True)
        res = ocr.predict(str(ip))[0]

        texts = res['rec_texts']
        scores = [float(s) for s in res['rec_scores']]
        polys = [np.array(p).tolist() for p in res['rec_polys']]

        # 시각화
        im = Image.open(ip).convert('RGB')
        draw = ImageDraw.Draw(im)
        font = get_font(max(14, im.width // 60))

        n_shown = 0
        for t, s, poly in zip(texts, scores, polys):
            if s < 0.5:
                continue
            n_shown += 1
            pts = [tuple(pt) for pt in poly]
            draw.polygon(pts, outline=(255, 0, 0), width=2)
            # 라벨: 박스 위에 텍스트+신뢰도
            x = min(p[0] for p in pts)
            y = min(p[1] for p in pts)
            label = f'{t} ({s:.2f})'
            tb = draw.textbbox((x, y), label, font=font)
            draw.rectangle([tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2],
                           fill=(255, 255, 160))
            draw.text((x, y), label, fill=(200, 0, 0), font=font)

        out_img = OUTPUT_DIR / f'{ip.stem}_paddle.png'
        im.save(out_img)

        # JSON 저장
        out_json = OUTPUT_DIR / f'{ip.stem}_paddle.json'
        dets = [{'text': t, 'score': round(s, 3), 'poly': poly}
                for t, s, poly in zip(texts, scores, polys)]
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({'image': ip.name, 'n_detections': len(dets),
                       'detections': dets}, f, indent=2, ensure_ascii=False)

        print(f"  검출 {len(texts)}개 (신뢰도 0.5+ 표시 {n_shown}개)")
        print(f"  저장: {out_img.name}", flush=True)

    print("\n완료.")


if __name__ == '__main__':
    main()
