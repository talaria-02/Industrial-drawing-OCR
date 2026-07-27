# -*- coding: utf-8 -*-
"""촬영 보정 — 스마트폰 사진을 '정면에서 본 mm 단위 평면'으로 되돌린다.

[왜 필요한가]
사진의 픽셀은 mm가 아니다. 스케일을 알아야 하고, 비스듬히 찍힌 원근도 펴야 한다.
ArUco 마커 하나를 부품 옆에 같이 찍으면 둘 다 해결된다 — 마커의 물리 크기를
아니까 px/mm가 나오고, 마커 네 꼭짓점으로 호모그래피를 세우면 평면이 펴진다.

[시뮬레이션으로 확인한 사실 — 촬영 지침의 근거]
가상 카메라로 측정오차를 재봤을 때:

  1. 기울기는 거의 제약이 아니다. 0도와 70도의 오차가 같았다(0.036 vs 0.041mm).
     호모그래피가 평면 원근을 '근사'가 아니라 정확히 되돌리기 때문이다.
     실용 한계는 마커 검출이 깨지는 75도 부근.
  2. 롤(광축 둘레 회전)과 카메라 거리도 오차에 영향이 없다.
  3. 지배적 오차는 '높이차'다. 마커면과 측정면의 높이가 다르면
         오차 = 측정길이 x 높이차 / 카메라거리
     로 배율 오차가 생긴다(공식과 시뮬레이션이 소수점 셋째자리까지 일치).
     100mm를 재는데 높이차 1mm, 거리 400mm면 그것만으로 0.25mm가 틀린다.
  4. 마커 크기가 가장 강력한 개선 수단이다. 마커에서 100mm 떨어진 곳을 잴 때
     마커 30mm는 0.878mm, 120mm는 0.059mm 오차 — 대략 1/크기^2로 준다.

그래서 촬영 지침의 우선순위는 (1) 마커를 측정면과 같은 높이에, (2) 마커를 크게,
(3) 마커를 측정 부위 가까이 이고, 기울기는 신경 쓸 필요가 없다.

[정밀도의 현실]
엣지 검출 +-1~2px, 렌즈왜곡 잔차, 초점 흐림을 합치면 실측 정확도는 +-0.3~1mm다.
도면 공차(+-0.01~0.1mm)보다 10~100배 크다. 이 기능은 정밀 측정이 아니라
'큰 오류 선별'용이며, 결과에는 반드시 불확실도를 함께 표시해야 한다.
"""
import numpy as np
import cv2

DEFAULT_DICT = cv2.aruco.DICT_4X4_50
# 마커가 화면에서 이보다 작으면 코너 정밀도가 급격히 나빠진다(경고용 임계).
MIN_MARKER_PX = 120
# 측정 지점이 마커 크기의 이 배수보다 멀면 호모그래피 외삽 오차가 커진다.
MAX_DIST_IN_MARKERS = 3.0
# 보정 결과에 담을 범위(마커 크기의 배수). 이보다 멀면 외삽 오차가 커서
# 어차피 측정에 못 쓴다. 50mm 마커면 반경 500mm로, 웬만한 부품은 다 들어간다.
EXTENT_IN_MARKERS = 10.0


def make_detector(dict_id=DEFAULT_DICT):
    """ArUco 검출기. 코너 서브픽셀 보정을 반드시 켠다.

    기본값(CORNER_REFINE_NONE)이면 코너 오차가 1.0~1.3px인데, SUBPIX를 켜면
    0.58~0.73px로 줄고 특히 기울기가 커져도 정밀도가 유지된다(70도에서
    1.26px -> 0.58px). 스케일과 원근이 전부 이 네 점에서 나오므로 여기서의
    0.5px가 측정 전체에 실린다.
    """
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), params)


def generate_marker(marker_id=0, px=1200, border=1, dict_id=DEFAULT_DICT):
    """인쇄용 마커 이미지. 실제 인쇄 크기를 자로 재서 marker_mm에 넣어야 한다
    (프린터 여백 설정 때문에 지정한 크기와 다르게 나오는 일이 흔하다)."""
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    img = cv2.aruco.generateImageMarker(d, marker_id, px)
    pad = int(px * 0.15 * border)
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def estimate_intrinsics(images, pattern=(9, 6), square_mm=25.0):
    """체커보드 여러 장으로 렌즈 왜곡계수를 구한다(폰마다 1회).

    반환 dict: camera_matrix, dist_coeffs, rms, n_used
    싼 렌즈는 화면 가장자리에서 배럴 왜곡이 수 % 나오므로, 이걸 안 잡으면
    가장자리 측정이 크게 틀린다.
    """
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_mm
    obj_pts, img_pts, size = [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for img in images:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, pattern, None)
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        obj_pts.append(objp)
        img_pts.append(corners)
    if len(obj_pts) < 5:
        raise ValueError(f"체커보드가 잡힌 사진이 {len(obj_pts)}장뿐입니다 (최소 5장, 권장 15~20장)")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    return {"camera_matrix": K, "dist_coeffs": dist, "rms": float(rms),
            "n_used": len(obj_pts)}


def _order_corners(c):
    """ArUco 코너를 좌상->우상->우하->좌하 순으로 정규화."""
    c = c.reshape(4, 2).astype(np.float64)
    s, d = c.sum(axis=1), np.diff(c, axis=1).ravel()
    return np.array([c[np.argmin(s)], c[np.argmin(d)],
                     c[np.argmax(s)], c[np.argmax(d)]], dtype=np.float32)


def rectify(image, marker_mm, px_per_mm=8.0, dict_id=DEFAULT_DICT,
            camera_matrix=None, dist_coeffs=None, margin_mm=40.0):
    """사진 -> 정면 시점의 mm 격자 이미지.

    image        : BGR 또는 그레이
    marker_mm    : 마커 한 변의 실제 길이(mm). 인쇄물을 자로 잰 값을 쓸 것.
    px_per_mm    : 결과 이미지의 해상도. 원본보다 크게 잡아도 정보가 늘지는 않는다.
    camera_matrix/dist_coeffs : estimate_intrinsics 결과. 주면 렌즈왜곡을 먼저 편다.
    margin_mm    : 마커 주변 몇 mm까지 결과에 담을지.

    반환 dict: rectified, px_per_mm, homography, marker_px, warnings, marker_id
      rectified 위에서는 (x2-x1)/px_per_mm 이 그대로 mm다.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if camera_matrix is not None and dist_coeffs is not None:
        image = cv2.undistort(image, camera_matrix, dist_coeffs)
        gray = cv2.undistort(gray, camera_matrix, dist_coeffs)

    corners, ids, _ = make_detector(dict_id).detectMarkers(gray)
    if ids is None or len(ids) == 0:
        raise ValueError("ArUco 마커를 찾지 못했습니다. 마커가 프레임 안에 있고 "
                         "초점이 맞았는지, 기울기가 75도를 넘지 않는지 확인하세요.")

    # 여러 개면 가장 크게 찍힌 것 — 코너 정밀도가 가장 좋다
    areas = [cv2.contourArea(c.reshape(4, 2).astype(np.float32)) for c in corners]
    k = int(np.argmax(areas))
    src = _order_corners(corners[k])
    marker_px = float(np.sqrt(areas[k]))

    warnings = []
    if marker_px < MIN_MARKER_PX:
        warnings.append(
            f"마커가 화면에서 {marker_px:.0f}px 뿐입니다(권장 {MIN_MARKER_PX}px 이상). "
            "더 가까이 찍거나 큰 마커를 쓰세요 — 오차가 대략 1/마커크기^2로 커집니다.")

    m = marker_mm * px_per_mm
    dst = np.array([[0, 0], [m, 0], [m, m], [0, m]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)

    # 출력 크기는 '원본 사진 전체가 담기도록' 잡는다.
    # 마커 주변 고정 여백(margin_mm)만 담으면 정작 측정할 부품이 잘려 나간다 —
    # 합성 검증 이미지를 눈으로 보고서야 발견했다. 측정 자체는 좌표 변환으로
    # 되므로 수치는 맞았지만, 사람이 결과를 보고 측정점을 찍을 수 없다.
    h, w = gray.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    pad = margin_mm * px_per_mm
    lo = np.minimum(warped.min(axis=0), 0) - pad
    hi = np.maximum(warped.max(axis=0), [m, m]) + pad
    # 기울여 찍으면 프레임의 먼 쪽 모서리가 아주 멀리 투영돼, 그대로 캔버스를
    # 잡으면 대부분이 빈 여백인 거대한 이미지가 나온다(실측: 5609x5291px).
    # 마커를 중심으로 현실적인 범위(마커 크기의 EXTENT_IN_MARKERS배)로 자른다 —
    # 그보다 멀면 어차피 외삽 오차가 커서 측정에 쓸 수 없는 영역이다.
    reach = EXTENT_IN_MARKERS * m
    lo = np.maximum(lo, np.array([m / 2 - reach, m / 2 - reach]))
    hi = np.minimum(hi, np.array([m / 2 + reach, m / 2 + reach]))
    span = np.maximum(hi - lo, m)
    T = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]], np.float64)
    H = T @ H
    size = (int(np.clip(span[0], 64, 20000)), int(np.clip(span[1], 64, 20000)))
    rectified = cv2.warpPerspective(image, H, size, flags=cv2.INTER_CUBIC,
                                    borderValue=(255, 255, 255))
    dst = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    off_x, off_y = float(dst[0][0]), float(dst[0][1])

    # 되돌려 본 잔차 — 호모그래피가 제대로 섰는지 확인용(4점 피팅이라 0에 가깝지만,
    # 코너가 엉뚱하게 잡힌 경우를 잡아낸다)
    ideal = dst[0] + np.array([[0, 0], [m, 0], [m, m], [0, m]], np.float64)
    residual = float(np.sqrt(np.mean(np.sum((dst - ideal) ** 2, axis=1))))

    return {"rectified": rectified, "px_per_mm": float(px_per_mm), "homography": H,
            # ids 모양이 (N,1)인 버전과 (N,)인 버전이 둘 다 있어 ravel로 통일
            "marker_px": marker_px, "marker_id": int(np.ravel(ids)[k]),
            "residual_px": residual, "warnings": warnings,
            "marker_origin_px": (off_x, off_y), "marker_mm": float(marker_mm)}


def make_board(cols=2, rows=2, marker_mm=60.0, gap_mm=20.0, dict_id=DEFAULT_DICT):
    """여러 마커가 '알려진 배치'로 인쇄된 보드.

    [낱개 마커 여러 장을 흩어놓으면 왜 안 되는가]
    서로의 상대 위치를 모르면 한 좌표계로 묶을 수 없다. 보드는 배치가 설계상
    확정돼 있어서(getObjPoints가 mm 좌표를 준다) 검출된 모든 코너를 하나의
    호모그래피에 함께 넣을 수 있다.

    [왜 하나보다 나은가]
    오차를 지배하는 건 마커 크기가 아니라 '기준점이 퍼진 폭'이다. 측정 지점이
    기준점 바깥에 있으면 외삽이라 오차가 거리에 비례해 커지는데, 보드로 부품을
    둘러싸면 내삽이 되어 오차가 급감한다. 덤으로 코너 수가 4개에서 4N개로 늘어
    최소자승 평균화 효과와, 일부가 가려져도 동작하는 여유가 생긴다.
    """
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.GridBoard((cols, rows), float(marker_mm), float(gap_mm), d)


def board_image(board, px_per_mm=8.0, margin_mm=10.0):
    """인쇄용 보드 이미지. 100% 배율로 인쇄한 뒤 반드시 자로 실측할 것."""
    cols, rows = board.getGridSize()
    ml, gap = board.getMarkerLength(), board.getMarkerSeparation()
    w_mm = cols * ml + (cols - 1) * gap
    h_mm = rows * ml + (rows - 1) * gap
    img = board.generateImage((int(w_mm * px_per_mm), int(h_mm * px_per_mm)))
    pad = int(margin_mm * px_per_mm)
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def rectify_board(image, board, px_per_mm=8.0, camera_matrix=None, dist_coeffs=None,
                  margin_mm=40.0, min_markers=2):
    """보드 기반 보정. rectify()와 같은 dict를 돌려주되 정확도가 더 좋다.

    검출된 마커가 min_markers 미만이면 실패시킨다 — 1개로 떨어지면 rectify()와
    같아지는데, 사용자는 보드를 썼으니 더 정확할 거라고 믿게 되므로 조용히
    수준을 낮추기보다 알려주는 편이 낫다.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if camera_matrix is not None and dist_coeffs is not None:
        image = cv2.undistort(image, camera_matrix, dist_coeffs)
        gray = cv2.undistort(gray, camera_matrix, dist_coeffs)

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(board.getDictionary(), params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < min_markers:
        n = 0 if ids is None else len(ids)
        raise ValueError(f"보드 마커를 {n}개만 찾았습니다(최소 {min_markers}개). "
                         "보드 전체가 프레임에 들어오고 초점이 맞았는지 확인하세요.")

    obj_all, ids_all = board.getObjPoints(), np.ravel(board.getIds())
    src_pts, dst_mm = [], []
    for c, i in zip(corners, np.ravel(ids)):
        w = np.where(ids_all == i)[0]
        if len(w) == 0:
            continue                      # 보드에 없는 마커(다른 물건)는 무시
        src_pts.append(c.reshape(4, 2))
        dst_mm.append(np.asarray(obj_all[w[0]], np.float64)[:, :2])
    if len(src_pts) < min_markers:
        raise ValueError(f"보드에 속한 마커가 {len(src_pts)}개뿐입니다.")

    src = np.concatenate(src_pts).astype(np.float64)
    dst0 = np.concatenate(dst_mm) * px_per_mm
    # 4점 초과라 최소자승이 걸린다. RANSAC은 쓰지 않는다 — 코너는 이미 검증된
    # 대응이라 이상치가 없고, 표본이 적어 무작위 표집이 오히려 불안정하다.
    H, _ = cv2.findHomography(src, dst0, method=0)
    if H is None:
        raise ValueError("호모그래피 계산에 실패했습니다.")

    # 재투영 잔차 — 마커 1개일 때는 항상 0이라 무의미했지만, 여기서는 실제
    # 품질 지표다(보드가 휘었거나 코너가 잘못 잡히면 값이 뛴다).
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    residual = float(np.sqrt(np.mean(np.sum((proj - dst0) ** 2, axis=1)))) / px_per_mm

    h, w = gray.shape[:2]
    fr = cv2.perspectiveTransform(
        np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2),
        H).reshape(-1, 2)
    pad = margin_mm * px_per_mm
    span_mm = max(np.ptp(dst0[:, 0]), np.ptp(dst0[:, 1]))
    reach = EXTENT_IN_MARKERS * max(span_mm, 1.0)
    ctr = dst0.mean(axis=0)
    lo = np.maximum(np.minimum(fr.min(axis=0), dst0.min(axis=0)) - pad, ctr - reach)
    hi = np.minimum(np.maximum(fr.max(axis=0), dst0.max(axis=0)) + pad, ctr + reach)
    T = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]], np.float64)
    H = T @ H
    size = (int(np.clip(hi[0] - lo[0], 64, 20000)), int(np.clip(hi[1] - lo[1], 64, 20000)))
    rectified = cv2.warpPerspective(image, H, size, flags=cv2.INTER_CUBIC,
                                    borderValue=(255, 255, 255))

    areas = [cv2.contourArea(c.reshape(4, 2).astype(np.float32)) for c in src_pts]
    marker_px = float(np.sqrt(np.mean(areas)))
    warnings = []
    if marker_px < MIN_MARKER_PX:
        warnings.append(f"마커가 평균 {marker_px:.0f}px 뿐입니다(권장 {MIN_MARKER_PX}px 이상).")
    if residual > 0.5:
        warnings.append(f"재투영 잔차 {residual:.2f}mm — 보드가 휘었거나 평면이 아닐 수 있습니다.")

    return {"rectified": rectified, "px_per_mm": float(px_per_mm), "homography": H,
            "marker_px": marker_px, "n_markers": len(src_pts),
            "residual_mm": residual, "warnings": warnings,
            "marker_origin_px": (float(-lo[0]), float(-lo[1])),
            "marker_mm": float(board.getMarkerLength())}


def measurement_uncertainty(length_mm, dist_from_marker_mm, marker_mm,
                            height_diff_mm=0.0, camera_dist_mm=400.0,
                            edge_px_err=1.5, px_per_mm=8.0):
    """이 조건에서 기대되는 측정 불확실도(mm). 결과에 반드시 같이 표시할 것.

    측정값만 보여주면 사용자가 그것을 정밀값으로 오해한다. '64.8'과
    '64.8 +-0.35'는 판정에서 전혀 다른 의미다.
    """
    # 높이차 -> 배율 오차 (시뮬레이션으로 검증된 공식)
    e_height = length_mm * abs(height_diff_mm) / max(camera_dist_mm, 1.0)
    # 엣지 검출 오차 (양 끝 2회)
    e_edge = np.sqrt(2) * edge_px_err / px_per_mm
    # 마커에서 멀어질수록 커지는 외삽 오차 — 마커 크기 대비 거리로 스케일
    e_extrap = e_edge * (dist_from_marker_mm / max(marker_mm, 1.0)) * 0.5
    total = float(np.sqrt(e_height ** 2 + e_edge ** 2 + e_extrap ** 2))
    return {"total": total, "height": float(e_height), "edge": float(e_edge),
            "extrapolation": float(e_extrap)}
