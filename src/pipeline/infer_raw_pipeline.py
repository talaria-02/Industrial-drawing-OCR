"""
export 없이 raw checkpoint로 det->crop->rec 파이프라인
==========================================================
paddlex/paddleocr pip패키지, tools/export_model.py 둘 다 필요없음.
학습(tools/train.py)때 쓰는 것과 똑같은 ppocr 패키지 API(build_model/
load_model/build_post_process)로 det+rec을 직접 체이닝함.

이렇게 하는 이유: export 경로는 Paddle Inference라는 별도의 버전에
민감한 엔진을 타서(PIR assert, set_optimization_level 등) 자꾸 버전
충돌이 났음([[stacked-tolerance-split-approach]]와 무관, 별개 이슈).
raw checkpoint 그대로 쓰면 학습 때 이미 검증된 API만 쓰니까 그 문제
자체가 안 생김. 대신 crop 자르는 로직(get_rotate_crop_image, PaddleOCR
tools/infer/utility.py에서 그대로 가져옴)은 직접 챙겨야 함.

주의: PaddleOCR repo의 tools/ 폴더 안에 이 파일을 복사해서 실행해야
함(같은 폴더에 있는 ppocr 패키지/program.py를 상대경로로 import하는
구조라서) — repo 루트에서 `python tools/infer_raw_pipeline.py`로 실행.

사용:
  python tools/infer_raw_pipeline.py \
    --det_config configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml \
    --det_weights pretrain_weights/PP-OCRv5_mobile_det_pretrained \
    --rec_config configs/rec/PP-OCRv6/PP-OCRv6_small_rec_finetune.yml \
    --rec_weights output/PP-OCRv6_small_rec_finetune/best_accuracy \
    --img_dir test_imgs \
    --out_dir vis_out_raw
"""

import argparse
import json
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
import numpy as np
import paddle
import yaml
from PIL import Image, ImageDraw, ImageFont

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list


def get_rotate_crop_image(img, points):
    """PaddleOCR tools/infer/utility.py 기반 — 회전/기운 quad를 perspective warp으로
    똑바로 편 crop 만듦.

    ※ 원본과 달리 "세로로 길면 90도 돌림"(np.rot90) 단계를 제거함. 그 고정회전은
    세로쓰기 crop을 항상 한 방향으로 미리 돌려버려서, 뒤이은 resolve_orientation의
    0/90/270 판단과 270 편향이 '이미 돌아간 것' 기준이 되어 어긋남. 여기선 warp만
    하고 방향은 rec 신뢰도(resolve_orientation)에 온전히 맡긴다."""
    points = points.astype(np.float32)
    img_crop_width = int(max(
        np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(
        np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([
        [0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst = cv2.warpPerspective(
        img, M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    return dst


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


def build_rec(rec_config_path, rec_weights):
    """tools/infer_rec.py의 MultiHead out_channels_list 설정 로직 그대로 재사용
    (CTCHead+NRTRHead 조합이라 이 설정 없으면 모델 출력 채널 크기가 안 맞음)."""
    config = load_yaml(rec_config_path)
    config["Global"]["pretrained_model"] = rec_weights
    global_config = config["Global"]
    post_process_class = build_post_process(config["PostProcess"], global_config)

    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["Head"]["name"] == "MultiHead":
            out_channels_list = {
                "CTCLabelDecode": char_num,
                "SARLabelDecode": char_num + 2,
                "NRTRLabelDecode": char_num + 3,
            }
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        else:
            config["Architecture"]["Head"]["out_channels"] = char_num

    model = build_model(config["Architecture"])
    load_model(config, model)
    model.eval()

    transforms = []
    for op in config["Eval"]["dataset"]["transforms"]:
        op_name = list(op)[0]
        if "Label" in op_name:
            continue
        elif op_name == "RecResizeImg":
            op[op_name]["infer_mode"] = True
        elif op_name == "KeepKeys":
            op[op_name]["keep_keys"] = ["image"]
        transforms.append(op)
    global_config["infer_mode"] = True
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


@paddle.no_grad()
def run_rec(model, post_process_class, ops, crop_bgr):
    ok, buf = cv2.imencode(".png", crop_bgr)
    data = {"image": buf.tobytes()}
    batch = transform(data, ops)
    images = np.expand_dims(batch[0], axis=0)
    images = paddle.to_tensor(images)
    preds = model(images)
    post_result = post_process_class(preds)
    return post_result[0][0], float(post_result[0][1])


def rotate_crop(img, angle):
    """angle: 0/90/180/270(시계방향). 별도 방향분류 모델 없이 cv2.rotate만 씀."""
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(angle)


def resolve_orientation(rec_model, rec_post, rec_ops, crops, flip_vote_thresh=0.5):
    """방향분류 모델 없이 rec 자체 신뢰도로 회전 판단(사용자 아이디어).

    1단계 - 페이지 단위 180도 판단: crop 하나하나의 0/180 비교는 노이즈가 커서
    믿기 어려움(문자 몇 개짜리 crop은 뒤집어도 rec이 그럴듯하게 읽어버릴 수 있음).
    실제로 스캔이 통째로 뒤집히는 경우 "페이지 안의 모든 crop"이 다 뒤집혀 있다는
    전제가 있으므로, crop 전체에 대해 0도/180도 중 뭐가 더 신뢰도 높은지 투표해서
    다수(threshold 이상)가 180을 선호할 때만 페이지 전체를 180도 보정한다.

    2단계 - crop별 0/90/270 판단: 페이지 보정 후에도 개별 crop이 세로쓰기(90도
    회전)일 수 있음(같은 도면 안에 가로/세로 텍스트 섞여있는 게 실제로 흔함).
    이건 페이지 단위가 아니라 crop별로 따로 판단해야 해서, 각 crop을 0/90/270도
    회전시켜보고 rec 신뢰도가 가장 높은 방향을 채택한다.

    반환: (보정된 crop 리스트, 각 crop의 최종 (text, score) 리스트)
    """
    if not crops:
        return [], [], False

    # 1단계: 페이지 단위 180도 투표
    flip_votes = 0
    zero_results = []
    for crop in crops:
        t0, s0 = run_rec(rec_model, rec_post, rec_ops, crop)
        t180, s180 = run_rec(rec_model, rec_post, rec_ops, rotate_crop(crop, 180))
        zero_results.append((t0, s0))
        if s180 > s0:
            flip_votes += 1
    page_flip = (flip_votes / len(crops)) >= flip_vote_thresh

    # 2단계: crop별 0/90/270 중 rec 신뢰도 최고 방향 채택
    # (0도는 1단계에서 이미 계산한 값 재사용해서 낭비 안 함)
    #
    # ※ 세로쓰기 방향은 rec 신뢰도로만 판단(고정 편향 없음). ISO 270도 관례를
    # 편향으로 강제해봤더니, 실제로는 도면마다 세로쓰기 방향이 달라서 90도가 정답인
    # 케이스에서 90(conf 1.0, '100')을 억누르고 270(반전 오독 '00 1')을 강제하는
    # 역효과가 확인됨. get_rotate_crop_image의 고정 90도 회전을 제거한 뒤로는
    # 신뢰도-최대가 정방향을 정확히 가리킴.
    final_crops, final_results = [], []
    for crop, (t0, s0) in zip(crops, zero_results):
        base = rotate_crop(crop, 180) if page_flip else crop
        if page_flip:
            best_text, best_score, best_crop = None, -1, None
            candidates = [0, 90, 270]
        else:
            best_text, best_score, best_crop = t0, s0, base
            candidates = [90, 270]
        for ang in candidates:
            c = rotate_crop(base, ang)
            t, s = run_rec(rec_model, rec_post, rec_ops, c)
            if s > best_score:
                best_text, best_score, best_crop = t, s, c
        final_crops.append(best_crop)
        final_results.append((best_text, best_score))

    return final_crops, final_results, page_flip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_config", required=True)
    ap.add_argument("--det_weights", required=True)
    ap.add_argument("--rec_config", required=True)
    ap.add_argument("--rec_weights", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", default="vis_out_raw")
    ap.add_argument("--score_thresh", type=float, default=0.5)
    ap.add_argument("--flip_vote_thresh", type=float, default=0.5,
                     help="페이지 전체 180도 보정을 적용할 최소 득표비율")
    args = ap.parse_args()

    print("det 모델 로드 중...")
    det_model, det_post, det_ops = build_det(args.det_config, args.det_weights)
    print("rec 모델 로드 중...")
    rec_model, rec_post, rec_ops = build_rec(args.rec_config, args.rec_weights)

    os.makedirs(args.out_dir, exist_ok=True)
    images = get_image_file_list(args.img_dir)
    print(f"대상: {len(images)}장")

    font_path = None
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if os.path.exists(cand):
            font_path = cand
            break

    for path in images:
        with open(path, "rb") as f:
            img_bytes = f.read()
        boxes = run_det(det_model, det_post, det_ops, img_bytes)

        # cv2.imread는 한글경로(바탕 화면 등) 못 읽음 → imdecode 패턴
        src_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        pil_img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(pil_img)
        size = max(9, pil_img.width // 130)
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()

        raw_crops = []
        pts_list = []
        for box in boxes:
            pts = np.array(box, dtype=np.float32)
            crop = get_rotate_crop_image(src_img, pts)
            if crop.size == 0:
                continue
            raw_crops.append(crop)
            pts_list.append(pts)

        _, results, page_flip = resolve_orientation(
            rec_model, rec_post, rec_ops, raw_crops, args.flip_vote_thresh)

        dets = []
        for pts, (text, score) in zip(pts_list, results):
            dets.append({"text": text, "score": round(score, 3), "poly": pts.tolist()})
            if score < args.score_thresh:
                continue

            pts_tuple = [tuple(p) for p in pts]
            draw.polygon(pts_tuple, outline=(255, 0, 0), width=2)
            x, y = min(p[0] for p in pts_tuple), min(p[1] for p in pts_tuple)
            label = f"{text} ({score:.2f})"
            tb = draw.textbbox((x, y), label, font=font)
            draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=(255, 255, 160))
            draw.text((x, y), label, fill=(200, 0, 0), font=font)

        stem = os.path.splitext(os.path.basename(path))[0]
        pil_img.save(os.path.join(args.out_dir, f"{stem}_vis.png"))
        with open(os.path.join(args.out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
            json.dump({"image": os.path.basename(path), "page_flip": page_flip,
                      "detections": dets}, f, indent=2, ensure_ascii=False)
        print(f"  {os.path.basename(path)}: {len(dets)}개 검출 (page_flip={page_flip})")

    print("완료")


if __name__ == "__main__":
    main()
