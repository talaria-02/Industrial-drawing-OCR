# -*- coding: utf-8 -*-
"""자동 파이프라인 실행 → review.json 초기 상태 생성.

UI에서 도면을 열면 이 모듈이 OCR(det+rec) → LSD 선분검출 → 매칭 → 화살촉검출을
순서대로 돌려서 사람이 검수할 초안을 만든다.

[모델을 전역 캐시하는 이유]
det/rec 모델 로딩만 2~3초 걸리고, 도면마다 다시 로드하면 낭비다. 한 번 로드해서
프로세스 수명 동안 재사용한다. (UI가 계속 켜져 있는 사용 패턴이므로 안전)

[chdir이 필요한 이유]
rec config의 character_dict_path가 PaddleOCR 레포 기준 상대경로라서, 모델을
만드는 동안에는 그 폴더에 있어야 한다. 끝나면 반드시 원위치로 돌린다.
"""
import os
import re
import sys

import cv2
import numpy as np

from . import model as M

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PPOCR_REPO = os.path.join(os.path.dirname(PROJECT_ROOT), "ppocr", "PaddleOCR")
PARAM_DIR = os.path.join(PROJECT_ROOT, "src", "parameter")
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "src", "pipeline")

DET_CONFIG = os.path.join(PPOCR_REPO, "configs", "det", "PP-OCRv6", "PP-OCRv6_small_det.yml")
DET_WEIGHTS = os.path.join(PARAM_DIR, "det_model2")
REC_CONFIG = os.path.join(PPOCR_REPO, "configs", "rec", "PP-OCRv6", "PP-OCRv6_small_rec.yml")
REC_WEIGHTS = os.path.join(PARAM_DIR, "rec_model")
FLIP_VOTE_THRESH = 0.5

REVIEW_ROOT = os.path.join(PROJECT_ROOT, "results", "review")

_models = None   # (pipe, det_model, det_post, det_ops, rec_model, rec_post, rec_ops)


def review_dir(img_path):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(REVIEW_ROOT, stem)


def review_json_path(img_path):
    return os.path.join(review_dir(img_path), "review.json")


def _ensure_dygraph():
    """이 스레드에서 paddle 동적그래프(dygraph) 모드를 보장한다.

    [왜 필요한가]
    paddle의 정적/동적 그래프 모드는 '스레드별'로 관리된다. 우리 UI는 OCR을
    워커 스레드(QThread)에서 돌리므로 모델이 그 스레드에서 만들어지는데, 이후
    메인 스레드에서 같은 모델로 추론을 시도하면 그 스레드는 여전히 정적 모드라서
    아래 오류가 난다:
      TypeError: conv2d(): argument (position 2) must be Value, but got EagerParamBase
    (Value=정적그래프 값, EagerParamBase=동적그래프 파라미터 — 둘이 안 맞음)

    실측: 워커에서 모델 로드 후 메인 스레드는 in_dynamic_mode()==False 였고,
    disable_static() 한 번 부르면 True가 되어 정상 동작했다.
    그래서 모델을 쓰는 모든 진입점에서 이걸 먼저 부른다.
    """
    import paddle
    if not paddle.in_dynamic_mode():
        paddle.disable_static()


def _load_models(progress=None):
    global _models
    if _models is not None:
        _ensure_dygraph()      # 캐시된 모델을 다른 스레드에서 쓰는 경우 대비
        return _models
    # torch 선점. 정상 경로에서는 main.py 최상단(PyQt5 import 전)에서 이미 로드돼
    # 있으므로 여기서는 사실상 no-op이다. 이 모듈을 단독으로 쓸 때를 위한 보험.
    # ImportError만 잡으면 안 된다 — PyQt5가 먼저 로드된 상태면 OSError(WinError 1114)가
    # 나므로 Exception으로 넓게 잡아야 한다(그 경우 아래 ppocr import에서 제대로 실패함).
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    if PPOCR_REPO not in sys.path:
        sys.path.insert(0, PPOCR_REPO)
    if PIPELINE_DIR not in sys.path:
        sys.path.insert(0, PIPELINE_DIR)
    os.environ["FLAGS_allocator_strategy"] = "auto_growth"
    _ensure_dygraph()      # 모델을 만들 스레드도 동적그래프 모드여야 한다

    prev = os.getcwd()
    os.chdir(PPOCR_REPO)
    try:
        import infer_raw_pipeline as pipe
        if progress:
            progress("det 모델 로딩...")
        det = pipe.build_det(DET_CONFIG, DET_WEIGHTS)
        if progress:
            progress("rec 모델 로딩...")
        rec = pipe.build_rec(REC_CONFIG, REC_WEIGHTS)
    finally:
        os.chdir(prev)
    _models = (pipe,) + det + rec
    return _models


def guess_category(text):
    """텍스트 모양만 보고 카테고리를 추정. 사람이 UI에서 고칠 수 있으므로
    틀려도 되지만, 맞으면 검수 클릭 수가 줄어든다."""
    s = text.strip()
    if re.search(r'[Rr][Aa]|√|▽|∇', s):
        return '거칠기'
    if '°' in s:
        return '각도'
    if any(c in s for c in 'øØ⌀φΦ'):
        return '지름'
    if '±' in s:
        return '공차'
    if re.match(r'^M\s*\d', s):
        return '나사'
    core = s.replace('.', '').replace(',', '').replace('-', '').replace(' ', '')
    if core.isdigit() and core:
        return '치수'
    return '메타데이터' if re.search(r'[A-Za-z]{3,}', s) else '기타'


def recognize_box(img_bgr, poly, progress=None):
    """사람이 그린 bbox 한 개에 대해 rec(문자인식)만 돌린다.

    det가 놓친 텍스트를 사람이 박스로 표시해주면, 그 안의 글자만 읽는 용도.
    det를 다시 돌리지 않으므로 1초 안에 끝난다.

    회전 판단(0/90/270)도 함께 한다 — 세로쓰기 치수가 흔하고, 방향을 틀리면
    '100'이 '00 1'처럼 읽히기 때문. resolve_orientation이 rec 신뢰도를 비교해서
    가장 그럴듯한 방향을 고른다.

    반환: (text, score). 실패하면 ('', 0.0).
    """
    pipe, det_model, det_post, det_ops, rec_model, rec_post, rec_ops = _load_models(progress)
    pts = np.array(poly, dtype=np.float32)
    crop = pipe.get_rotate_crop_image(img_bgr, pts)
    if crop is None or crop.size == 0:
        return '', 0.0
    # flip_vote_thresh=1.1 : 1개짜리 crop에서는 페이지 단위 180도 보정을 아예 끈다
    # (표본 1개로 페이지 전체가 뒤집혔다고 판단하면 위험)
    _, results, _ = pipe.resolve_orientation(rec_model, rec_post, rec_ops, [crop], 1.1)
    if not results:
        return '', 0.0
    text, score = results[0]
    return text, float(score)


def build_review(img_path, progress=None):
    """자동 파이프라인 전체 실행 → review 문서(dict) 반환."""
    if PIPELINE_DIR not in sys.path:
        sys.path.insert(0, PIPELINE_DIR)
    from line_detect import run_line_detect as rld
    from line_detect import match_numbers as mn
    from line_detect import arrowhead_template as at
    from line_detect import targets as TG

    pipe, det_model, det_post, det_ops, rec_model, rec_post, rec_ops = _load_models(progress)

    # ── 1) OCR (det → crop → rec + 회전판단) ──────────────
    if progress:
        progress("텍스트 검출/인식 중... (수십 초)")
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    boxes = pipe.run_det(det_model, det_post, det_ops, img_bytes)
    src = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    H, W = src.shape[:2]

    raw_crops, pts_list = [], []
    for box in boxes:
        pts = np.array(box, dtype=np.float32)
        crop = pipe.get_rotate_crop_image(src, pts)
        if crop.size == 0:
            continue
        raw_crops.append(crop)
        pts_list.append(pts)
    _, rec_results, _ = pipe.resolve_orientation(
        rec_model, rec_post, rec_ops, raw_crops, FLIP_VOTE_THRESH)

    doc = M.ReviewDoc(M.empty_doc(os.path.basename(img_path), (W, H)))
    ocr_json = {"detections": []}
    for pts, (text, score) in zip(pts_list, rec_results):
        poly = pts.tolist()
        ocr_json["detections"].append({"text": text, "score": round(float(score), 3),
                                        "poly": poly})
        doc.data["texts"].append({
            "id": f't{len(doc.data["texts"]) + 1}',
            "poly": [[float(x), float(y)] for x, y in poly],
            "text": text, "score": round(float(score), 3),
            "category": guess_category(text), "source": "auto", "verified": False,
        })

    # ── 2) 선분 검출 (LSD + 정리) ─────────────────────────
    if progress:
        progress("선분 검출 중...")
    _, raw_lines, _ = rld.process_image_lsd(img_path)

    # 텍스트 영역 안에 완전히 들어간 선분은 글자 획이므로 제외
    _, text_bboxes = TG.build_targets_and_text_bboxes(ocr_json)
    lines = mn.filter_lines_in_text_regions(raw_lines, text_bboxes)
    for i, L in enumerate(lines):
        doc.data["lines"].append({
            "id": f'l{i + 1}', "p1": [float(L[0]), float(L[1])],
            "p2": [float(L[2]), float(L[3])], "source": "auto",
        })

    # ── 3) 매칭 (숫자 ↔ 선분) ─────────────────────────────
    if progress:
        progress("숫자-선 매칭 중...")
    targets, _ = TG.build_targets_and_text_bboxes(ocr_json)
    # targets는 숫자류만 걸러진 목록이라, 원본 detections 순서와 맞추기 위해
    # bbox로 대응되는 text id를 찾는다
    def find_text_id(bbox):
        for t in doc.data["texts"]:
            tb = doc.text_bbox(t)
            if all(abs(a - b) < 1.0 for a, b in zip(tb, bbox)):
                return t["id"]
        return None

    cands = {ti: mn.score_candidates(t, lines) for ti, t in enumerate(targets)}
    assignment, _ = mn.assign_greedy(cands)
    for ti, cand in assignment.items():
        tid = find_text_id(targets[ti]["bbox"])
        if tid is None:
            continue
        doc.data["links"].append({
            "text_id": tid, "line_ids": [f'l{cand["line_idx"] + 1}'],
            "source": "auto", "confidence": round(float(cand["combined"]), 3),
            "verified": False,
        })

    # ── 4) 화살촉 검출 ────────────────────────────────────
    if progress:
        progress("화살촉 검출 중...")
    if len(lines) > 0:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # 연결된 선분만 검사 — 전체(수백~수천 개)를 다 돌리면 느리고, 검수에 필요한 건
        # 실제로 숫자와 연결된 선의 화살촉 여부다
        linked_ids = sorted({lid for l in doc.data["links"] for lid in l["line_ids"]})
        idx_of = {f'l{i + 1}': i for i in range(len(lines))}
        sub_idx = [idx_of[x] for x in linked_ids if x in idx_of]
        if sub_idx:
            sub = lines[sub_idx]
            results, _diag = at.detect_on_lines(gray, sub)
            for res, lid in zip(results, [linked_ids[k] for k in range(len(sub_idx))]):
                for end in ("start", "end"):
                    doc.data["arrows"].append({
                        "id": f'a{len(doc.data["arrows"]) + 1}',
                        "line_id": lid, "end": end,
                        "present": bool(res[end]["found"]),
                        "score": round(float(res[end]["score"]), 3),
                        "source": "auto",
                    })

    if progress:
        progress("완료")
    return doc
