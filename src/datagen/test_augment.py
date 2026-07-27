# [파일 설명]
# augment.py에 구현된 3가지 이미지 변형(왜곡, 늘리기, 원근법) 기능들이 잘 작동하는지 
# 눈으로 확인하고 테스트해 보기 위해 샘플 이미지를 변형하여 저장하는 테스트 스크립트입니다.

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from __init__ import tia_distort, tia_stretch, tia_perspective

SAMPLES_DIR = Path(__file__).resolve().parent / 'samples'
SAMPLES_DIR.mkdir(exist_ok=True)

# 명령어 인자(Argument)를 받을 수 있도록 설정
parser = argparse.ArgumentParser(description="Test TIA Augmentation")
parser.add_argument("--image", type=str,
                    default=str(Path(__file__).resolve().parent.parent
                                / 'data' / 'real' / 'scanA.png'),
                    help="Path to the input image")
args = parser.parse_args()

img_path = args.image

if not os.path.exists(img_path):
    print(f"이미지가 없습니다: {img_path}")
    # 테스트를 위한 더미 이미지 생성 (흰색 바탕에 검은색 텍스트)
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, 'Test TIA Augmentation', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.imwrite(img_path, img)
else:
    print(f"이미지 불러오기: {img_path}")
    # 한글 경로 대응: cv2.imread는 non-ASCII 경로를 못 읽음
    img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)

if img is None:
    print("이미지를 불러오는데 실패했습니다.")
else:
    print("1. tia_distort 실행 중...")
    distort_img = tia_distort(img.copy(), segment=4)
    cv2.imwrite(str(SAMPLES_DIR / "test_distort.png"), distort_img)

    print("2. tia_stretch 실행 중...")
    stretch_img = tia_stretch(img.copy(), segment=4)
    cv2.imwrite(str(SAMPLES_DIR / "test_stretch.png"), stretch_img)

    print("3. tia_perspective 실행 중...")
    perspective_img = tia_perspective(img.copy())
    cv2.imwrite(str(SAMPLES_DIR / "test_perspective.png"), perspective_img)

    print(f"테스트가 완료되었습니다! 결과 이미지 확인: {SAMPLES_DIR}/")
