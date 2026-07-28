# -*- coding: utf-8 -*-
"""고전 LSD(Line Segment Detector) 백엔드.

Stage B(DeepLSD)로 교체될 때도 인터페이스(gray 이미지 -> (N,4) 선분 배열)는
그대로 유지되도록, 이 파일의 detect() 시그니처를 backend_deeplsd.py도
동일하게 맞출 것.

LSD 원리(Hough와 다른 점): 픽셀별 그래디언트 방향을 재서 방향이 비슷한
이웃끼리 영역 성장(region growing)으로 뭉친 뒤, 그 뭉친 영역을 감싸는
직사각형의 긴 축을 선분으로 채택한다. Hough처럼 "투표 후 픽셀을 소비하며
걸어나가는" 과정이 없어서 두 선이 교차하는 지점에서 픽셀을 서로 빼앗는
문제(PPHT의 교차점 끊김)가 원리적으로 발생하지 않는다.
"""
import cv2
import numpy as np


def detect(gray):
    """gray: 2D uint8 배열. 반환: (N,4) float64 배열 [x1,y1,x2,y2]."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    result = lsd.detect(gray)
    lines = result[0]
    if lines is None:
        return np.empty((0, 4), dtype=np.float64)
    return lines.reshape(-1, 4).astype(np.float64)

