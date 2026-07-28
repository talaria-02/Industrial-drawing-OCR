# -*- coding: utf-8 -*-
"""폰 사진 1장 검증 — 보드가 잡히는지, 보정이 제대로 되는지, 스케일이 맞는지.

측정 UI를 만들기 전에 '촬영이 쓸 만한가'부터 확인하는 용도다. 촬영 조건이
나쁘면 뒷단을 아무리 잘 만들어도 소용없으므로, 이 단계에서 걸러야 한다.

실행:
    python src/measure/check_photo.py 사진.jpg
    python src/measure/check_photo.py 사진.jpg --p1 120,340 --p2 480,344

보드 규격 기본값은 사용자가 인쇄한 A4 보드에 맞춰져 있다
(마커 50mm, 중심간 150mm -> 간격 100mm). 인쇄물을 자로 재서 다르면 옵션으로 준다.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measure import calibration as cal  # noqa: E402


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def draw_mm_grid(img, px_per_mm, step_mm=10, major=50):
    """보정 결과에 mm 격자를 얹는다. 눈으로 스케일을 검증할 수 있어야 한다 —
    자를 같이 찍었다면 격자와 눈금이 맞는지 바로 보인다."""
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    for mm in range(0, int(w / px_per_mm) + 1, step_mm):
        x = int(mm * px_per_mm)
        big = (mm % major == 0)
        cv2.line(out, (x, 0), (x, h), (0, 200, 255) if big else (200, 230, 230), 2 if big else 1)
        if big:
            cv2.putText(out, str(mm), (x + 4, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 140, 200), 2)
    for mm in range(0, int(h / px_per_mm) + 1, step_mm):
        y = int(mm * px_per_mm)
        big = (mm % major == 0)
        cv2.line(out, (0, y), (w, y), (0, 200, 255) if big else (200, 230, 230), 2 if big else 1)
        if big:
            cv2.putText(out, str(mm), (4, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 140, 200), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--marker-mm", type=float, default=50.0, help="마커 한 변 실측값")
    ap.add_argument("--pitch-mm", type=float, default=150.0, help="마커 중심간 거리 실측값")
    ap.add_argument("--px-per-mm", type=float, default=8.0)
    ap.add_argument("--p1", type=str, default=None, help="측정점1 'x,y' (보정 이미지 px)")
    ap.add_argument("--p2", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    img = imread_unicode(a.photo)
    if img is None:
        print(f"[실패] 사진을 못 읽었습니다: {a.photo}")
        return 1
    h, w = img.shape[:2]
    print(f"사진: {w}x{h} ({w*h/1e6:.1f}MP)")

    # 중심간 거리(pitch) = 마커변 + 간격 이므로 간격을 역산한다.
    gap = a.pitch_mm - a.marker_mm
    board = cal.make_board(2, 2, marker_mm=a.marker_mm, gap_mm=gap)
    print(f"보드: 마커 {a.marker_mm}mm, 중심간 {a.pitch_mm}mm (간격 {gap}mm)")

    try:
        r = cal.rectify_board(img, board, px_per_mm=a.px_per_mm)
    except ValueError as e:
        print(f"\n[실패] {e}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c, ids, _ = cal.make_detector().detectMarkers(gray)
        n = 0 if ids is None else len(ids)
        print(f"  참고: 마커 {n}개 검출"
              + (f" (ID {sorted(np.ravel(ids).tolist())})" if n else ""))
        print("  확인할 것: 보드 전체가 프레임 안에 있는가 / 초점 / 그림자·반사 /"
              " 기울기 75도 미만 / 인쇄가 100% 배율인가")
        return 1

    print(f"\n검출 마커 {r['n_markers']}/4개, 평균 {r['marker_px']:.0f}px")
    print(f"재투영 잔차 {r['residual_mm']:.3f}mm  "
          + ("(양호)" if r['residual_mm'] < 0.3 else "(보드가 휘었거나 평면이 아닐 수 있음)"))
    print(f"보정 결과 {r['rectified'].shape[1]}x{r['rectified'].shape[0]}px "
          f"@ {r['px_per_mm']}px/mm")
    for wmsg in r["warnings"]:
        print(f"  [경고] {wmsg}")

    stem = os.path.splitext(os.path.basename(a.photo))[0]
    out = a.out or os.path.join("results", f"{stem}_rectified.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    vis = draw_mm_grid(r["rectified"], r["px_per_mm"])

    if a.p1 and a.p2:
        p1 = np.array([float(v) for v in a.p1.split(",")])
        p2 = np.array([float(v) for v in a.p2.split(",")])
        mm = float(np.hypot(*(p2 - p1))) / r["px_per_mm"]
        # 마커 중심에서 측정 중점까지의 거리로 외삽 오차를 추정
        ctr = np.array(r["marker_origin_px"]) + a.pitch_mm * r["px_per_mm"] / 2
        dist_mm = float(np.hypot(*((p1 + p2) / 2 - ctr))) / r["px_per_mm"]
        u = cal.measurement_uncertainty(mm, dist_mm, a.pitch_mm,
                                        px_per_mm=r["px_per_mm"])
        print(f"\n측정: {mm:.2f} ±{u['total']:.2f} mm")
        print(f"  (엣지 {u['edge']:.2f} / 외삽 {u['extrapolation']:.2f} / "
              f"높이차 {u['height']:.2f} — 높이차는 0 가정)")
        cv2.line(vis, tuple(np.int32(p1)), tuple(np.int32(p2)), (0, 0, 255), 3)
        for q in (p1, p2):
            cv2.drawMarker(vis, tuple(np.int32(q)), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 30, 3)

    cv2.imencode(".png", vis)[1].tofile(out)
    print(f"\n저장: {out}")
    print("  격자는 10mm 간격(굵은 선 50mm). 자를 같이 찍었다면 눈금과 맞는지 보세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
