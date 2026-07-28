# -*- coding: utf-8 -*-
"""폰 렌즈 왜곡 보정값 구하기 (폰마다 1회). 체커보드 사진 -> JSON.

[왜 이걸 먼저 해야 하는가]
스마트폰 렌즈는 화각을 넓게 잡느라 배럴(볼록) 왜곡이 있다. 화면 중앙은
멀쩡한데 가장자리로 갈수록 바깥으로 늘어난다. 볼펜처럼 긴 물체를 화면에
꽉 채워 찍으면 양 끝이 정확히 그 늘어나는 영역에 놓여, 길이가 실제보다
크게 나온다.

ArUco 호모그래피는 '평면의 원근'만 되돌린다. 렌즈 왜곡은 원근이 아니라
광학적 곡률이므로 호모그래피로는 절대 펴지지 않는다. 그래서 순서가 고정된다:

    ① 렌즈왜곡 보정(undistort)  ->  ② 원근 보정(호모그래피)  ->  ③ 측정

거꾸로 하면 이미 휘어진 좌표에 평면 변환을 씌우는 셈이라, 마커 근처는
맞고 멀어질수록 틀리는 오차가 남는다.

실행:
    # 체커보드로 (정확도 우선)
    python src/measure/calibrate_camera.py data/calib/내폰 --square-mm 25

    # 측정용 ArUco 보드로 (인쇄물 하나로 끝내기 — 장수를 더 찍어야 함)
    python src/measure/calibrate_camera.py data/calib/내폰 --aruco \
        --marker-mm 50 --pitch-mm 150

    -> results/camera_calib/내폰.json  저장

촬영 요령:
  - 체커보드를 A4에 100% 배율로 인쇄하고 딱딱한 판에 붙인다(휘면 값이 틀어진다)
  - 15~20장. 정면 몇 장 + 좌우/상하로 기울인 것 + 회전시킨 것
  - 보드가 화면의 1/3 이상을 채우고, 특히 '화면 가장자리에 보드가 걸친 컷'을
    반드시 포함한다 — 왜곡이 가장 큰 곳의 표본이 없으면 그 영역이 안 펴진다
  - 초점 고정, 디지털 줌 금지(줌 배율이 바뀌면 다른 렌즈나 다름없다)
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measure import calibration as cal  # noqa: E402

CALIB_DIR = os.path.join("results", "camera_calib")


def save_calib(name, data, out_dir=CALIB_DIR):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "camera_matrix": np.asarray(data["camera_matrix"]).tolist(),
            "dist_coeffs": np.asarray(data["dist_coeffs"]).ravel().tolist(),
            "rms": data["rms"], "n_used": data["n_used"],
            "image_size": data.get("image_size"),
            "pattern": data.get("pattern"), "square_mm": data.get("square_mm"),
        }, f, indent=2)
    return path


def load_calib(path):
    """저장된 JSON -> rectify_board에 넘길 수 있는 dict."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {"camera_matrix": np.array(d["camera_matrix"], float),
            "dist_coeffs": np.array(d["dist_coeffs"], float).reshape(1, -1),
            "rms": d.get("rms"), "n_used": d.get("n_used"),
            "image_size": d.get("image_size")}


def latest_calib(out_dir=CALIB_DIR):
    """가장 최근에 저장된 보정값. UI가 자동으로 집어 쓰게 하려는 용도."""
    files = sorted(glob.glob(os.path.join(out_dir, "*.json")),
                   key=os.path.getmtime, reverse=True)
    if not files:
        return None, None
    return load_calib(files[0]), files[0]


def make_checkerboard(cols=9, rows=6, square_mm=25.0, dpi=300, margin_mm=10.0):
    """인쇄용 체커보드. cols/rows는 '내부 코너 수'이므로 칸은 +1개씩 그린다.

    인쇄 후 한 칸을 자로 재서 --square-mm에 넣어야 한다. 프린터 배율 때문에
    지정값과 다르게 나오는 일이 흔하고, 이 값이 틀리면 초점거리 추정이
    그만큼 틀어진다.
    """
    ppmm = dpi / 25.4
    sq = int(round(square_mm * ppmm))
    w, h = sq * (cols + 1), sq * (rows + 1)
    img = np.zeros((h, w), np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                img[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 255
    pad = int(round(margin_mm * ppmm))
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def intrinsics_from_aruco(images, marker_mm=50.0, pitch_mm=150.0):
    """ArUco 보드 사진들로 렌즈 왜곡을 구한다. 체커보드를 따로 인쇄하지 않는 길.

    [체커보드와 무엇이 다른가]
    9x6 체커보드는 한 장에서 코너 54점을 준다. 2x2 ArUco 보드는 16점뿐이다.
    점이 적으면 왜곡계수(특히 k2, k3) 추정이 흔들리므로, 장수로 메워야 한다 —
    체커보드 15~20장 자리에 25~30장을 권한다.

    [그래도 이 길을 두는 이유]
    측정에 쓸 보드는 이미 인쇄돼 있다. 종이를 하나 더 만들고 그 칸을 다시
    재는 일이 늘면 실제로는 캘리브레이션을 건너뛰게 된다. 정확도가 조금
    낮아도 실행되는 절차가 낫다.

    marker_mm / pitch_mm 은 인쇄물을 자로 잰 값을 넣어야 한다 — 이 값이 틀리면
    초점거리가 그만큼 통째로 틀어진다.
    """
    board = cal.make_board(2, 2, marker_mm=marker_mm, gap_mm=pitch_mm - marker_mm)
    det = cal.make_detector()
    obj_pts, img_pts, size = [], [], None
    for im in images:
        gray = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None or len(ids) < 3:
            continue          # 3개 미만이면 그 컷은 자세를 못 믿는다
        o, i = board.matchImagePoints(corners, ids)
        if o is None or len(o) < 8:
            continue
        obj_pts.append(o.astype(np.float32))
        img_pts.append(i.astype(np.float32))
    if len(obj_pts) < 8:
        raise ValueError(
            f"보드가 잡힌 사진이 {len(obj_pts)}장뿐입니다. "
            "ArUco 보드로 왜곡을 구하려면 최소 8장(권장 25~30장) 필요합니다 — "
            "코너 수가 체커보드의 1/3이라 장수로 메워야 합니다.")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    return {"camera_matrix": K, "dist_coeffs": dist, "rms": float(rms),
            "n_used": len(obj_pts), "image_size": size,
            "pattern": "aruco2x2", "square_mm": marker_mm}


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=".",
                    help="체커보드 사진이 든 폴더")
    ap.add_argument("--name", default=None, help="저장 이름(기본: 폴더명)")
    ap.add_argument("--pattern", default="9x6", help="내부 코너 수 (열x행)")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--make-board", action="store_true",
                    help="인쇄용 체커보드만 만들고 끝낸다")
    ap.add_argument("--aruco", action="store_true",
                    help="체커보드 대신 측정용 ArUco 보드 사진으로 구한다")
    ap.add_argument("--marker-mm", type=float, default=50.0)
    ap.add_argument("--pitch-mm", type=float, default=150.0)
    a = ap.parse_args()

    if a.make_board:
        img = make_checkerboard(*[int(v) for v in a.pattern.lower().split("x")],
                                square_mm=a.square_mm)
        os.makedirs(CALIB_DIR, exist_ok=True)
        out = os.path.join(CALIB_DIR, f"checkerboard_{a.pattern}_{a.square_mm:g}mm.png")
        cv2.imencode(".png", img)[1].tofile(out)
        print(f"저장: {out}  ({img.shape[1]}x{img.shape[0]}px @300dpi)")
        print("  100% 배율로 인쇄하고, 딱딱한 판에 붙인 뒤 한 칸을 자로 재세요.")
        return 0

    pat = tuple(int(v) for v in a.pattern.lower().split("x"))
    files = sorted(sum([glob.glob(os.path.join(a.folder, e))
                        for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG")], []))
    if not files:
        print(f"[실패] 사진이 없습니다: {a.folder}")
        return 1
    print(f"사진 {len(files)}장, 패턴 {pat[0]}x{pat[1]}, 한 칸 {a.square_mm}mm")

    imgs, size = [], None
    for f in files:
        im = imread_unicode(f)
        if im is None:
            continue
        imgs.append(im)
        size = (im.shape[1], im.shape[0])

    try:
        if a.aruco:
            print(f"ArUco 보드 모드 (마커 {a.marker_mm}mm, 중심간 {a.pitch_mm}mm)")
            r = intrinsics_from_aruco(imgs, a.marker_mm, a.pitch_mm)
        else:
            r = cal.estimate_intrinsics(imgs, pattern=pat, square_mm=a.square_mm)
            r["image_size"] = size
            r["pattern"] = list(pat)
            r["square_mm"] = a.square_mm
    except ValueError as e:
        print(f"\n[실패] {e}")
        print("  보드가 잘려 있거나 흐리면 코너 검출이 안 됩니다. "
              "보드 전체가 프레임에 들어오게 다시 찍으세요.")
        return 1
    K, D = r["camera_matrix"], np.asarray(r["dist_coeffs"]).ravel()
    print(f"\n사용된 사진 {r['n_used']}/{len(imgs)}장,  재투영 RMS {r['rms']:.3f}px "
          + ("(양호)" if r["rms"] < 1.0 else "(높음 — 보드가 휘었거나 흐린 컷이 섞였을 수 있음)"))
    print(f"초점거리 fx={K[0,0]:.1f} fy={K[1,1]:.1f}  주점 ({K[0,2]:.1f}, {K[1,2]:.1f})")
    print(f"왜곡계수 k1={D[0]:+.4f} k2={D[1]:+.4f} p1={D[2]:+.4f} p2={D[3]:+.4f}"
          + (f" k3={D[4]:+.4f}" if len(D) > 4 else ""))

    # 왜곡이 실제로 얼마나 큰지 화면 모서리에서 재본다 — 이 값이 작으면
    # 애초에 보정이 필요 없다는 뜻이고, 크면 반드시 적용해야 한다.
    if size:
        w, h = size
        pts = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]],
                        [[w / 2, h / 2]]], np.float32)
        und = cv2.undistortPoints(pts, K, r["dist_coeffs"], P=K).reshape(-1, 2)
        shift = np.linalg.norm(und - pts.reshape(-1, 2), axis=1)
        print(f"보정으로 움직이는 거리: 네 모서리 {shift[:4].max():.1f}px, "
              f"중앙 {shift[4]:.1f}px")
        print("  -> 모서리 이동량이 크면, 긴 물체를 화면에 꽉 채워 찍었을 때 "
              "그만큼 길이가 부풀어 있었다는 뜻입니다.")

    name = a.name or os.path.basename(os.path.normpath(a.folder))
    path = save_calib(name, r)
    print(f"\n저장: {path}")
    print("  이후 측정 모드에서 사진을 열면 이 값이 자동으로 적용됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
