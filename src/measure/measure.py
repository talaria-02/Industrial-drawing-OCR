# -*- coding: utf-8 -*-
"""보정된 이미지 위에서 실제 길이를 잰다.

[전제]
입력은 calibration.rectify_board()가 돌려준 정면 mm 격자 이미지다. 여기서는
(x2-x1)/px_per_mm 가 그대로 mm이므로, 남은 일은 '어디를 잴 것인가'를 정확히
집는 것뿐이다.

[사람이 찍은 점을 그대로 쓰지 않는 이유]
마우스 클릭은 ±3px쯤 부정확하다. 8px/mm에서 3px면 0.375mm — 목표 정확도
(±0.3~1mm)의 절반을 클릭 오차로 날리는 셈이다. 그래서 클릭 지점 주변에서
실제 밝기 경계를 찾아 붙인다(스냅). 측정 방향으로만 훑는 것이 요령인데,
경계가 측정 방향과 수직이라 그쪽으로 가장 가파르게 변하기 때문이다.

[자동 검출의 한계 — 미리 알 것]
배경과 대비되는 물체만 잡힌다. 흰 종이 위의 흰 물체는 임계값으로 못 가른다.
그런 경우는 클릭 측정을 쓰거나, 대비되는 배경지를 깔고 다시 찍어야 한다.
알고리즘으로 풀 문제를 촬영으로 치환하는 편이 훨씬 싸다.
"""
import numpy as np
import cv2


def _to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def snap_to_edge(img, point, direction, search_px=12, blur=1.0):
    """측정 방향을 따라 훑어 가장 가파른 밝기 변화 지점으로 점을 옮긴다.

    point     : (x, y) 클릭 좌표
    direction : 측정 방향 단위벡터. 이 축을 따라서만 탐색한다.
    반환      : (스냅된 점, 이동거리 px, 엣지 강도)  — 강도가 낮으면 못 믿는다.

    포물선 보간으로 서브픽셀까지 낸다. 정수 픽셀에서 멈추면 8px/mm에서 0.125mm의
    양자화 오차가 그대로 남는다.
    """
    g = _to_gray(img).astype(np.float32)
    if blur:
        g = cv2.GaussianBlur(g, (0, 0), blur)
    H, W = g.shape
    d = np.asarray(direction, float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.asarray(point, float), 0.0, 0.0
    d = d / n

    ts = np.arange(-search_px, search_px + 1, 0.5)
    pts = np.asarray(point, float)[None, :] + d[None, :] * ts[:, None]
    xs = np.clip(pts[:, 0], 0, W - 1).astype(np.float32)
    ys = np.clip(pts[:, 1], 0, H - 1).astype(np.float32)
    prof = cv2.remap(g, xs.reshape(-1, 1), ys.reshape(-1, 1),
                     cv2.INTER_LINEAR).ravel()
    grad = np.abs(np.gradient(prof))
    k = int(np.argmax(grad))
    if k == 0 or k == len(grad) - 1:
        return np.asarray(point, float) + d * ts[k], float(ts[k]), float(grad[k])

    # 포물선 정점으로 서브픽셀 보정
    y0, y1, y2 = grad[k - 1], grad[k], grad[k + 1]
    denom = (y0 - 2 * y1 + y2)
    off = 0.0 if abs(denom) < 1e-9 else 0.5 * (y0 - y2) / denom
    t = ts[k] + off * (ts[1] - ts[0])
    return np.asarray(point, float) + d * t, float(t), float(y1)


def measure_two_points(rectified, p1, p2, px_per_mm, snap=True, search_px=12):
    """두 점 사이 거리(mm). snap=True면 양 끝을 측정축 방향의 엣지에 붙인다."""
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    v = p2 - p1
    L = np.linalg.norm(v)
    if L < 1e-9:
        return {"mm": 0.0, "p1": p1, "p2": p2, "snapped": False}
    d = v / L
    info = {"snapped": bool(snap)}
    if snap:
        # 각 끝점은 '안쪽에서 바깥쪽으로' 훑는다. 물체 끝을 재는 상황이 대부분이라
        # 바깥 방향으로 나가면서 만나는 첫 경계가 우리가 원하는 지점이다.
        q1, s1, g1 = snap_to_edge(rectified, p1, -d, search_px)
        q2, s2, g2 = snap_to_edge(rectified, p2, d, search_px)
        info.update({"shift1_px": s1, "shift2_px": s2,
                     "edge1": g1, "edge2": g2})
        p1, p2 = q1, q2
    mm = float(np.linalg.norm(p2 - p1)) / px_per_mm
    info.update({"mm": mm, "p1": p1, "p2": p2})
    return info


def paper_region(rectified, shrink_mm=3.0, px_per_mm=8.0, valid_mask=None):
    """촬영 배경지(밝은 큰 영역)만 남기는 마스크.

    보정 이미지에는 종이 바깥의 책상까지 들어온다. 나뭇결은 대비가 뚜렷해서
    그대로 두면 물체로 잡힌다(실측: 299mm짜리 '물체' 두 개가 검출됐다).
    가장 큰 밝은 덩어리를 종이로 보고 그 안쪽만 남긴다. 가장자리를 조금
    깎는 것은 종이 경계선 자체가 물체로 잡히는 것을 막기 위해서다.
    """
    g = _to_gray(rectified)
    _, bright = cv2.threshold(cv2.GaussianBlur(g, (0, 0), 2.0), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if valid_mask is not None:
        # 촬영 범위 밖(흰색 채움)을 먼저 잘라내야 종이만 남는다
        bright = cv2.bitwise_and(bright, valid_mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k, iterations=3)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    if n <= 1:
        return np.full(g.shape, 255, np.uint8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(lab == big, 255, 0).astype(np.uint8)

    # 구멍을 메운다. 종이 위에 놓인 '어두운' 물체는 밝은 덩어리에 속하지 않아
    # 종이 영역에 구멍을 뚫는데, 그대로 두면 정작 재려던 물체가 마스크에서
    # 지워진다(실측: 검은 클립과 심이 통째로 사라졌다).
    # 바깥 윤곽만 채우면 물체가 다시 종이 안쪽으로 들어온다.
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, cv2.FILLED)

    e = int(max(1, shrink_mm * px_per_mm))
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))


def detect_objects(rectified, px_per_mm, exclude_boxes=None,
                   min_area_mm2=20.0, margin_px=4, restrict_to_paper=True,
                   drop_nested=True, drop_border=True, valid_mask=None):
    """배경에서 물체를 분리해 각각의 길이/폭을 잰다.

    exclude_boxes : [(x,y,w,h), ...] 제외 영역(마커 등). 마커를 안 빼면 물체로 잡힌다.
    반환: [{contour, length_mm, width_mm, area_mm2, angle, box}] 길이 내림차순

    최소외접 회전사각형을 쓰는 이유: 물체가 비스듬히 놓여 있어도 '긴 쪽/짧은 쪽'을
    바로 얻는다. 축정렬 bbox를 쓰면 기울어진 물체의 길이가 실제보다 커진다.
    """
    g = _to_gray(rectified)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    # 조명 불균일에 강하도록 적응 임계 + Otsu를 함께 쓰고 둘의 합집합을 취한다.
    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 51, 8)
    mask = cv2.bitwise_or(otsu, adap)

    if restrict_to_paper:
        mask = cv2.bitwise_and(mask, paper_region(rectified, px_per_mm=px_per_mm,
                                                  valid_mask=valid_mask))

    if exclude_boxes:
        for (x, y, w, h) in exclude_boxes:
            x0, y0 = max(0, int(x - margin_px)), max(0, int(y - margin_px))
            x1, y1 = int(x + w + margin_px), int(y + h + margin_px)
            mask[y0:y1, x0:x1] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area_px = min_area_mm2 * px_per_mm ** 2
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area_px]

    # 배경지 경계에 닿는 덩어리는 물체가 아니라 종이 가장자리/그림자다
    # (실측: 181mm짜리 '물체'가 왼쪽 종이 경계였다). 재려는 부품은 안쪽에 있다.
    if drop_border and restrict_to_paper:
        inner = cv2.erode(paper_region(rectified, px_per_mm=px_per_mm,
                                       valid_mask=valid_mask),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                          iterations=2)
        H, W = mask.shape
        kept = []
        for c in cnts:
            pts = c.reshape(-1, 2)[::max(1, len(c) // 40)]
            xs = np.clip(pts[:, 0], 0, W - 1)
            ys = np.clip(pts[:, 1], 0, H - 1)
            if np.all(inner[ys, xs] > 0):
                kept.append(c)
        cnts = kept

    # 몸통에 인쇄된 글자/표시는 물체가 아니라 물체 '안'의 무늬다. 다른 물체의
    # 회전사각형 안에 완전히 들어가면 부품으로 세지 않는다.
    if drop_nested and len(cnts) > 1:
        rects = [cv2.minAreaRect(c) for c in cnts]
        areas = [r[1][0] * r[1][1] for r in rects]
        keep = []
        for i, c in enumerate(cnts):
            inside = False
            for j, rj in enumerate(rects):
                if i == j or areas[j] <= areas[i]:
                    continue
                box = cv2.boxPoints(rj).astype(np.float32)
                if all(cv2.pointPolygonTest(box, (float(p[0][0]), float(p[0][1])), False) >= 0
                       for p in c[::max(1, len(c) // 12)]):
                    inside = True
                    break
            if not inside:
                keep.append(c)
        cnts = keep

    out = []
    for c in cnts:
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        L, W = max(w, h), min(w, h)
        out.append({
            "contour": c,
            "center": (cx / px_per_mm, cy / px_per_mm),
            "length_mm": L / px_per_mm,
            "width_mm": W / px_per_mm,
            "area_mm2": cv2.contourArea(c) / px_per_mm ** 2,
            "angle": float(ang),
            "box": cv2.boxPoints(((cx, cy), (w, h), ang)),
        })
    out.sort(key=lambda d: -d["length_mm"])
    return out, mask


def marker_boxes(rectified, dict_id=None):
    """보정 이미지에서 마커를 찾아 제외용 bbox 리스트를 만든다."""
    from . import calibration as cal
    g = _to_gray(rectified)
    corners, ids, _ = cal.make_detector(dict_id or cal.DEFAULT_DICT).detectMarkers(g)
    boxes = []
    if ids is not None:
        for c in corners:
            x, y, w, h = cv2.boundingRect(c.reshape(4, 2).astype(np.float32))
            boxes.append((x, y, w, h))
    return boxes


def draw_objects(rectified, objects, px_per_mm):
    vis = rectified.copy() if rectified.ndim == 3 else cv2.cvtColor(rectified, cv2.COLOR_GRAY2BGR)
    for i, o in enumerate(objects):
        box = np.int32(o["box"])
        cv2.drawContours(vis, [box], 0, (0, 0, 255), 3)
        cx, cy = np.int32(np.array(o["center"]) * px_per_mm)
        cv2.putText(vis, f'{i+1}: {o["length_mm"]:.1f}x{o["width_mm"]:.1f}mm',
                    (cx - 90, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 5)
        cv2.putText(vis, f'{i+1}: {o["length_mm"]:.1f}x{o["width_mm"]:.1f}mm',
                    (cx - 90, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    return vis
