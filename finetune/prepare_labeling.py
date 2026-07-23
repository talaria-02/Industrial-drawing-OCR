# [파일 설명]
# 이 파일은 data/real 폴더에 있는 원본 이미지들을 학습용(Train)과 테스트용(Test)으로 나누어
# PPOCRLabel 프로그램이 바로 읽을 수 있는 폴더(finetune/labeling/train, test)로
#  복사해 주는 역할을 합니다.

"""
PPOCRLabel 라벨링 작업공간 준비
================================
data/real(29장)에서 학습 12장 / 테스트 5장을 뽑아
PPOCRLabel이 바로 열 수 있는 폴더로 복사한다.

선택 기준 (다양성 확보): data/real 현재 17장 전부를 12/5로 분배.
  train 12 = 숫자 변형 묶음(04,10,11,17) + 신규 PDF 도면 +
             eDOCr 검증 도면 4장(BM_part 등) + CAD시트/손글씨스캔(cadA,scanA,scanB)
  test  5  = train과 겹치지 않는 숫자 변형(05,12,16) + eDOCr 도면(halter) + 사진(photoA)

PPOCRLabel 사용법 (이 스크립트 실행 후):
  1) PPOCRLabel --lang en           (GUI 실행, 폴더 열기 다이얼로그에서
                                      finetune/labeling/train 선택)
  2) 박스 그리고 텍스트 입력 (자동 인식 결과가 뜨면 수정만 하면 됨)
  3) Ctrl+S 저장 → 폴더 안에 Label.txt 자동 생성 (det 학습용 포맷)
  4) 메뉴 File > Export Recognition Result
     → crop_img/ 폴더 + rec_gt.txt 생성 (rec 학습용 포맷, det 라벨에서 자동 파생)
  5) test 폴더도 동일하게 반복 (검증셋)

즉 한 번의 수작업 라벨링으로 det(Label.txt)·rec(crop_img/+rec_gt.txt)
라벨을 동시에 확보한다 — 두 형식을 따로 만들 필요 없음.
"""

import shutil
from pathlib import Path

SRC_DIR = Path('data/real')
OUT_DIR = Path('finetune/labeling')

# data/real 현재 17장 전부를 12(train)/5(test)로 분배
# (이전엔 29장이었으나 이후 추가 정리로 17장만 남음 — 아래는 그 17장 기준)
TRAIN_IMAGES = [
    '04.png', '10.png', '11.png', '17.png',           # 숫자 변형 묶음
    '2d-drw-Model.pdf_page_1.jpg',                     # 신규 PDF 추출 실도면
    'BM_part.jpg', 'Candle_holder.jpg', 'Gripper.jpg', 'LIU0010.jpg',  # eDOCr 검증 도면
    'cadA.png', 'scanA.png', 'scanB.png',              # CAD 시트 / 손그림 스캔
]

TEST_IMAGES = [
    '05.png', '12.png', '16.png',   # 숫자 변형 묶음 (train과 겹치지 않는 것)
    'halter.jpg', 'photoA.jpg',     # eDOCr 도면 + 사진 스타일
]


def copy_set(names, dest):
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in names:
        src = SRC_DIR / name
        if not src.exists():
            print(f"  [경고] 없음: {src}")
            continue
        shutil.copy2(src, dest / name)
        n += 1
    return n


def main():
    assert not set(TRAIN_IMAGES) & set(TEST_IMAGES), "train/test 중복 있음"

    train_dir = OUT_DIR / 'train'
    test_dir = OUT_DIR / 'test'

    n_train = copy_set(TRAIN_IMAGES, train_dir)
    n_test = copy_set(TEST_IMAGES, test_dir)

    print(f"\n준비 완료: train {n_train}장 → {train_dir}")
    print(f"          test  {n_test}장 → {test_dir}")
    print(f"\n다음 단계:")
    print(f"  PPOCRLabel --lang en")
    print(f"  → 폴더 열기: {train_dir.resolve()}")
    print(f"  → 라벨링 후 Ctrl+S, 그 다음 test 폴더도 동일하게 반복")
    print(f"  → 완료 후 File > Export Recognition Result 로 rec 라벨까지 생성")


if __name__ == '__main__':
    main()
