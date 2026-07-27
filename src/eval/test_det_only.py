"""
det(stock, 파인튜닝 안 함)이 실제 도면 특수기호를 얼마나 놓치는지 bbox만 확인
==================================================================================
rec 없이 det 박스만 그려서 저장 — 특수기호(GD&T, 거칠기 등) 있는 위치에
박스가 아예 안 쳐지는지(= det이 그 자체를 못 찾음) 눈으로 확인하는 용도.

det 파인튜닝을 실제로 할지 결정하기 전에, "진짜 문제가 있는지"부터 검증.
"""

import sys
sys.path.insert(0, r'C:\Users\zxc20\OneDrive\바탕 화면\ppocr\PaddleOCR')

try:
    import torch  # noqa: F401  (DLL 선점 목적 — PyQt5/torch 충돌때와 같은 패턴)
except ImportError:
    pass

import os
import cv2
import numpy as np
import paddle
import yaml
from PIL import Image, ImageDraw

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

# 파인튜닝된 v6_small_det (best_accuracy, Colab에서 받은 model.pdparams)
# load_model이 경로 뒤에 .pdparams 를 자동으로 붙이므로 확장자 빼고 지정.
DET_CONFIG = r'C:\Users\zxc20\OneDrive\바탕 화면\ppocr\PaddleOCR\configs\det\PP-OCRv6\PP-OCRv6_small_det.yml'
DET_WEIGHTS = r'C:\Users\zxc20\Downloads\model'

# 특수기호 많은 실이미지 위주로 선정 (√, ø, GD&T 등 이미 있다고 확인된 것들)
TARGET_IMAGES = [
    r'data\real\train\BM_part.jpg',
    r'data\real\train\Gripper.jpg',
    r'data\real\train\Dessin-ind-Omnifab-vis-page-001-1-2.jpg',
]

OUT_DIR = 'results/det_only_check'


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_det(det_config_path, det_weights):
    config = load_yaml(det_config_path)
    config["Global"]["pretrained_model"] = det_weights
    global_config = config["Global"]
    model = build_model(config["Architecture"])
    load_model(config, model)
    model.eval()
    post_process_class = build_post_process(config["PostProcess"])
    transforms = []
    for op in config["Eval"]["dataset"]["transforms"]:
        op_name = list(op)[0]
        if "Label" in op_name:
            continue
        elif op_name == "KeepKeys":
            op[op_name]["keep_keys"] = ["image", "shape"]
        transforms.append(op)
    ops = create_operators(transforms, global_config)
    return model, post_process_class, ops


@paddle.no_grad()
def run_det(model, post_process_class, ops, img_bytes):
    data = {"image": img_bytes}
    batch = transform(data, ops)
    images = np.expand_dims(batch[0], axis=0)
    shape_list = np.expand_dims(batch[1], axis=0)
    images = paddle.to_tensor(images)
    preds = model(images)
    post_result = post_process_class(preds, shape_list)
    return post_result[0]["points"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("det 모델 로드 중...")
    det_model, det_post, det_ops = build_det(DET_CONFIG, DET_WEIGHTS)

    for path in TARGET_IMAGES:
        if not os.path.exists(path):
            print(f'없음: {path}')
            continue
        with open(path, "rb") as f:
            img_bytes = f.read()
        boxes = run_det(det_model, det_post, det_ops, img_bytes)

        pil_img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(pil_img)
        for box in boxes:
            pts = [tuple(p) for p in box]
            draw.polygon(pts, outline=(255, 0, 0), width=3)

        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(OUT_DIR, f'{stem}_bbox_only.png')
        pil_img.save(out_path)
        print(f'{os.path.basename(path)}: 박스 {len(boxes)}개 -> {out_path}')

    print('완료')


if __name__ == '__main__':
    main()
