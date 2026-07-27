# -*- coding: utf-8 -*-
"""이미지 1장 -> 선분 검출 파이프라인 실행 (Stage A: 고전 LSD 백엔드).

process_image_lsd()   : 전처리 -> (선택)타일링 -> LSD -> 후처리 -> JSON/통계
process_image_hough()  : 기존 HoughLinesP 전역(이미지 전체) 베이스라인
                          (anchor_all_test.py의 "bbox 근처 crop" 버전과 다르게,
                          여기선 LSD와 공정 비교를 위해 이미지 전체에 대해 돌림)

두 함수 모두 같은 형태의 stats dict를 반환해서 나란히 비교하기 쉽게 한다.
"""
import json
import os
import time

import cv2
import numpy as np

from . import tiling, backend_lsd, refine


def _angle_stats(lines):
    """주어진 선분 배열에서 수직, 수평, 사선(대각선)의 개수를 각각 카운트하여 반환합니다."""
    # 선분이 하나도 없으면 모두 0 반환
    if len(lines) == 0:
        return 0, 0, 0
    # 각 선분의 x, y 변화량 계산 (x2 - x1, y2 - y1)
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    # 아크탄젠트(arctan2)를 이용해 선분의 절대 각도를 구하고(라디안), degree(도) 단위로 변환
    ang = np.degrees(np.arctan2(dy, dx))
    # 수직선 판별: 각도가 90도 또는 270도에서 ±5도 이내인 선분의 개수
    near_vert = np.sum(np.minimum(np.abs(np.abs(ang) - 90), np.abs(np.abs(ang) - 270)) < 5)
    # 수평선 판별: 각도가 0도 또는 180도에서 ±5도 이내인 선분의 개수
    near_horiz = np.sum((np.abs(ang) < 5) | (np.abs(np.abs(ang) - 180) < 5))
    # 사선 판별: 전체 선분 개수에서 수직선과 수평선을 뺀 나머지
    diagonal = len(lines) - near_vert - near_horiz
    # 카운트된 개수를 정수형으로 반환
    return int(near_vert), int(near_horiz), int(diagonal)


def load_gray_color(img_path):
    """한글 경로명 등에서도 문제없이 이미지를 읽어오기 위해 numpy로 읽은 뒤 디코딩합니다."""
    # np.fromfile을 이용해 바이트 스트림으로 읽고, cv2.imdecode로 BGR 컬러 이미지로 변환 (한글 경로 지원)
    color = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    # 컬러 이미지를 흑백(Grayscale) 이미지로 변환 (선분 검출은 주로 흑백에서 수행됨)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    # 원본 컬러 이미지와 흑백 이미지를 함께 반환
    return color, gray


def process_image_lsd(img_path, use_tiling=False, tile=512, overlap=64,
                       refine_kwargs=None, use_g0=True):
    """LSD 알고리즘을 사용한 메인 선분 검출 파이프라인 (이미지 로드 -> 검출 -> 후처리 -> 통계).

    use_g0=True(기본): 엣지쌍 병합을 포함한 refine_pipeline_g0을 쓴다.
      LSD는 선이 아니라 '경계'를 검출하므로 도면의 선 하나가 선분 2개로 나온다.
      G0은 그 둘을 잉크검사로 확인해 중심선 하나로 합친다(층화 24장 검증:
      중심선 잉크 적중률 중앙 99.4%).
    use_g0=False: 기존 refine_pipeline. A/B 비교용으로 남겨둠."""
    # 1. 이미지 로드: 원본 컬러 이미지와 분석에 쓰일 흑백 이미지를 가져옵니다.
    color, gray = load_gray_color(img_path)
    H, W = gray.shape # 이미지의 높이(H)와 너비(W) 추출
    refine_kwargs = refine_kwargs or {} # 후처리 인자가 없으면 빈 딕셔너리로 초기화

    t0 = time.time() # 전체 실행 시간 측정을 위한 시작 시간 저장
    
    # 2. 선분 검출 (LSD)
    if use_tiling:
        # 초대형 이미지 처리를 위해 타일링(쪼개기) 옵션이 켜진 경우
        boxes = tiling.make_tiles(H, W, tile=tile, overlap=overlap) # 겹치는 타일 박스들을 생성
        all_lines = []
        for (x0, y0, x1, y1) in boxes:
            patch_gray = gray[y0:y1, x0:x1] # 원본에서 타일 크기만큼 이미지 부분 추출(Crop)
            local_lines = backend_lsd.detect(patch_gray) # 잘라낸 부분 이미지에서 선분 검출
            all_lines.append(tiling.to_global(local_lines, x0, y0)) # 검출된 지역 좌표를 전체 이미지 기준 전역 좌표로 변환
        # 모든 타일에서 찾은 선분들을 하나의 배열로 합침
        raw_lines = np.concatenate(all_lines, axis=0) if all_lines else np.empty((0, 4))
    else:
        # 타일링 없이 전체 이미지를 한 번에 LSD로 선분 검출 (기본값)
        raw_lines = backend_lsd.detect(gray)
    t_detect = time.time() - t0 # 검출 단계에 걸린 시간 계산

    t1 = time.time() # 후처리 시간 측정을 위한 시간 저장
    # 3. 후처리 (Refinement): 잔챙이 제거, 엣지쌍 병합, 끊어진 선 잇기, 직교 스냅
    if use_g0:
        # gray를 넘기는 이유: 엣지쌍 판정이 '두 선분 사이가 잉크인가'를 원본
        # 픽셀에서 직접 확인하기 때문. 기하 정보만으로는 한 획의 양쪽 경계와
        # 두 획 사이의 여백을 구분할 수 없다(둘 다 평행/근접/역방향).
        final_lines, refine_stats = refine.refine_pipeline_g0(
            raw_lines, gray, **refine_kwargs)
    else:
        final_lines, refine_stats = refine.refine_pipeline(raw_lines, **refine_kwargs)
    t_refine = time.time() - t1 # 후처리에 걸린 시간 계산

    elapsed = time.time() - t0 # 전체(검출+후처리) 소요 시간 계산
    # 4. 각도 통계 추출: 최종 선분들 중 수직/수평/사선의 개수를 셉니다.
    n_vert, n_horiz, n_diag = _angle_stats(final_lines)

    # 5. 결과 통계를 딕셔너리 형태로 묶어서 정리
    stats = {
        "method": ("lsd_tiled" if use_tiling else "lsd") + ("_g0" if use_g0 else ""),
        "elapsed_sec": elapsed, # 전체 소요 시간
        "detect_sec": t_detect, # 검출 소요 시간
        "refine_sec": t_refine, # 후처리 소요 시간
        "n_lines_final": int(len(final_lines)), # 최종 살아남은 선분의 총 개수
        "n_vertical": n_vert, # 수직선 개수
        "n_horizontal": n_horiz, # 수평선 개수
        "n_diagonal": n_diag, # 사선 개수
        **refine_stats, # 후처리 모듈에서 나온 상세 통계(필터링된 개수 등)를 병합
    }
    # 원본 컬러 이미지, 최종 완성된 선분 좌표 리스트, 통계 딕셔너리를 반환
    return color, final_lines, stats


def process_image_hough(img_path, canny_lo=50, canny_hi=150,
                         hough_threshold=80, min_len_ratio=0.02, max_gap=5):
    """비교용 베이스라인: 전통적인 허프 변환(HoughLinesP)을 사용한 선분 검출.
    (LSD의 성능과 비교하기 위해 덧붙여진 과거의 레거시 방식입니다.)"""
    # 1. 이미지 로드
    color, gray = load_gray_color(img_path)
    H, W = gray.shape

    t0 = time.time() # 시작 시간 기록
    # 2. Canny Edge Detection: 픽셀의 밝기 차이가 큰 윤곽선(Edge)만 흰색으로 추출
    edges = cv2.Canny(gray, canny_lo, canny_hi, apertureSize=3)
    # 3. Probabilistic Hough Transform: 추출된 윤곽선을 기반으로 파라미터 공간에 투표하여 직선 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_threshold,
                             minLineLength=int(W * min_len_ratio), maxLineGap=max_gap)
    elapsed = time.time() - t0 # 소요 시간 기록

    # 4. 검출된 선분이 없을 경우 빈 배열을, 있으면 [N, 4] 형태의 numpy 배열로 모양을 맞춤
    final_lines = np.empty((0, 4)) if lines is None else lines.reshape(-1, 4).astype(np.float64)
    
    # 5. 각도 통계 추출
    n_vert, n_horiz, n_diag = _angle_stats(final_lines)

    # 6. 통계 정리 후 반환
    stats = {
        "method": "hough",
        "elapsed_sec": elapsed,
        "n_lines_final": int(len(final_lines)),
        "n_vertical": n_vert,
        "n_horizontal": n_horiz,
        "n_diagonal": n_diag,
    }
    return color, final_lines, stats


def draw_lines(color_img, lines, ortho_color=(0, 0, 255), diag_color=(255, 140, 0), thickness=1):
    """검출된 선분들을 사람이 볼 수 있게 이미지 위에 그려주는 함수 (디버깅/결과 확인용).
    직교(수직/수평) 선은 빨강, 사선은 주황으로 구분해서 그립니다."""
    vis = color_img.copy() # 원본 이미지를 훼손하지 않기 위해 복사본 생성
    for (x1, y1, x2, y2) in lines:
        # 각 선분의 각도를 계산
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # 수평(0, 180) 혹은 수직(90, 270)에서 ±5도 이내가 아니면 사선(Diagonal)으로 판별
        is_diag = not (np.abs(ang) < 5 or np.abs(np.abs(ang) - 180) < 5
                       or np.abs(np.abs(ang) - 90) < 5 or np.abs(np.abs(ang) - 270) < 5)
        # 사선이면 주황색, 직교선이면 빨간색 지정
        color = diag_color if is_diag else ortho_color
        # 원본 복사 이미지 위에 실제로 선을 그림 (좌표는 정수형으로 변환)
        cv2.line(vis, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, thickness)
    return vis # 선이 그려진 최종 이미지 반환


def save_png(path, image):
    """결과 이미지를 PNG 파일로 저장합니다. (한글 경로 지원을 위해 np.tofile 사용)"""
    cv2.imencode('.png', image)[1].tofile(path)


def save_lines_json(path, lines, stats, image_name):
    """검출된 선분 좌표들과 통계 데이터를 다른 프로그램(웹이나 다른 분석 툴)에서 쓰기 쉽게 JSON 파일로 저장합니다."""
    # JSON 파일에 들어갈 데이터 구조(Payload) 만들기
    payload = {
        "image": image_name, # 처리된 원본 이미지 파일명
        "stats": stats, # 걸린 시간, 선 개수 등 딕셔너리 형태의 통계
        # 리스트 안의 모든 선분 배열을 순회하며 x1, y1, x2, y2 키를 가진 객체 리스트로 변환
        "lines": [{"x1": float(l[0]), "y1": float(l[1]),
                   "x2": float(l[2]), "y2": float(l[3])} for l in lines],
    }
    # utf-8 인코딩으로 JSON 파일 쓰기 (한글이나 특수문자 깨짐 방지)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
