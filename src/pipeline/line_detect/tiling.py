# -*- coding: utf-8 -*-
"""이미지를 겹치는(overlap) 타일로 나누고, 타일 안 좌표를 전역 좌표로 되돌리는 유틸.

Stage A(고전 LSD)는 실측상 타일링 없이도 전체 이미지를 0.256초에 처리하므로
기본은 타일링 OFF. Stage B(DeepLSD, CPU 추론)에서 메모리/속도 때문에 필수가
될 것을 대비해 인터페이스를 미리 갖춰둔다.
"""
import numpy as np


def make_tiles(H, W, tile=512, overlap=64):
    """(x0,y0,x1,y1) 타일 박스 리스트 생성. 이미지 전체를 겹치게 덮는다.

    overlap이 필요한 이유: 타일 경계에 걸친 선분은 자르는 순간 반쪽씩
    두 조각으로 나뉘어, 각 조각이 최소길이 미달로 버려질 수 있다.
    겹치는 영역에는 선분이 안 잘린 채로 들어있는 타일이 최소 하나는 있게 된다.
    """
    step = tile - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than tile")

    xs = list(range(0, max(W - tile, 0) + 1, step))
    ys = list(range(0, max(H - tile, 0) + 1, step))
    if not xs or xs[-1] + tile < W:
        xs.append(max(W - tile, 0))
    if not ys or ys[-1] + tile < H:
        ys.append(max(H - tile, 0))

    boxes = []
    for y0 in sorted(set(ys)):
        for x0 in sorted(set(xs)):
            x1, y1 = min(x0 + tile, W), min(y0 + tile, H)
            boxes.append((x0, y0, x1, y1))
    return boxes


def to_global(lines_local, x0, y0):
    """타일 내부 좌표(x1,y1,x2,y2) 배열에 타일 오프셋(x0,y0)을 더해 전역 좌표로 변환."""
    if len(lines_local) == 0:
        return lines_local
    out = lines_local.copy()
    out[:, [0, 2]] += x0
    out[:, [1, 3]] += y0
    return out
