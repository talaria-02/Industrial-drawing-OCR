"""
로컬 실행 런처 — infer_raw_pipeline (det+rec+회전판단)을 윈도우에서 돌림
==========================================================================
infer_raw_pipeline.py 자체는 Colab(PaddleOCR/tools/ 안)용이라 그대로는 로컬에서
ppocr import/torch DLL 충돌이 남. 이 런처가:
  1) torch를 먼저 import(PyQt5/paddle DLL 선점 — test_det_only와 같은 패턴)
  2) 로컬 PaddleOCR 레포를 sys.path에 추가
  3) argv 세팅 후 infer_raw_pipeline.main() 호출
하는 얇은 래퍼.

가중치 경로만 아래 상수에서 바꾸면 됨. load_model이 .pdparams 를 자동으로 붙이므로
가중치는 확장자 빼고 지정.
"""

import os
import sys

# 1) torch 선점 (DLL 충돌 회피)
try:
    import torch  # noqa: F401
except ImportError:
    pass

# 2) 로컬 PaddleOCR 레포 + infer_raw_pipeline 위치를 경로에 추가
PPOCR_REPO = r'C:\Users\zxc20\OneDrive\바탕 화면\ppocr\PaddleOCR'
sys.path.insert(0, PPOCR_REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 설정 (여기만 바꾸면 됨) ──────────────────────────────────
PARAM_DIR = r'C:\Users\zxc20\OneDrive\바탕 화면\현장미러형\src\parameter'

DET_CONFIG = os.path.join(PPOCR_REPO, r'configs\det\PP-OCRv6\PP-OCRv6_small_det.yml')
DET_WEIGHTS = os.path.join(PARAM_DIR, 'det_model2')          # 재학습 det (확장자 빼고)

REC_CONFIG = os.path.join(PPOCR_REPO, r'configs\rec\PP-OCRv6\PP-OCRv6_small_rec.yml')
REC_WEIGHTS = os.path.join(PARAM_DIR, 'rec_model')           # 파인튜닝 rec (확장자 빼고)

IMG_DIR = r'C:\Users\zxc20\OneDrive\바탕 화면\현장미러형\data\real\test'
OUT_DIR = r'C:\Users\zxc20\OneDrive\바탕 화면\현장미러형\results\pipeline_test_v2'
SCORE_THRESH = 0.5
FLIP_VOTE_THRESH = 0.5


def main():
    os.environ["FLAGS_allocator_strategy"] = "auto_growth"
    # rec config의 character_dict_path가 PaddleOCR 레포 기준 상대경로라 거기로 이동
    # (우리 config/weights/img/out 경로는 전부 절대경로라 chdir 영향 없음)
    os.chdir(PPOCR_REPO)
    sys.argv = [
        'infer_raw_pipeline',
        '--det_config', DET_CONFIG,
        '--det_weights', DET_WEIGHTS,
        '--rec_config', REC_CONFIG,
        '--rec_weights', REC_WEIGHTS,
        '--img_dir', IMG_DIR,
        '--out_dir', OUT_DIR,
        '--score_thresh', str(SCORE_THRESH),
        '--flip_vote_thresh', str(FLIP_VOTE_THRESH),
    ]
    import infer_raw_pipeline
    infer_raw_pipeline.main()


if __name__ == '__main__':
    main()
