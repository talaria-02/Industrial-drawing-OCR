# -*- coding: utf-8 -*-
"""도면 1장 + 제품사진 N장 -> 치수별 합/불 보고서. 전체를 잇는 얇은 층.

[파일 이름 주의]
inspect.py로 두면 안 된다. 이 폴더가 sys.path에 올라가는 순간 파이썬 표준
라이브러리 inspect를 가려서, scipy가 "from inspect import signature"에 실패한다
(실제로 겪었다). 표준 모듈과 이름이 겹치는 파일은 만들지 않는다.

    도면 ─ OCR ─ 선분(G0) ─ 원/호 ─ 매칭 ─ 역추적(측정점) ─┐
                                                            ├─ 비교 ─ 판정
    사진 ─ ArUco 보정 ─ 기하추출 ─ 원배치 정합 ─ 국소측정 ──┘

[설계 원칙 — 어느 단계가 막혀도 사람이 이어받는다]
자동 정합이 실패하는 경우가 실제로 있다(원이 없는 부품, 대칭이라 방향이 모호한
플랜지). 그럴 때 전체를 포기하는 대신, 무엇이 왜 막혔는지 남기고 사람이 검수
UI에서 이어서 하도록 한다. 지금 만들어둔 측정 모드가 그 수동 경로다.

[정합은 위치만, 값은 국소 측정으로]
정합 변환으로 옮긴 측정점은 '사진의 이쯤'을 알려줄 뿐이다. 최종 수치는 그
주변에서 실제 엣지를 찾아 낸다. 그래서 정합이 2mm쯤 어긋나도 측정값은 정확하다.
"""
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE), os.path.join(os.path.dirname(_HERE), "pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from measure import calibration as cal, measure as ms, compare as cp, registration as reg  # noqa: E402
from line_detect import traceback_points as tp  # noqa: E402

# 국소 재탐색 반경(mm). 정합 오차를 흡수할 만큼 넓되, 옆 형상까지 먹지 않을 만큼 좁게.
LOCAL_SEARCH_MM = 2.5


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def measure_points_from_doc(doc_data, lines=None):
    """검수 문서에서 치수별 측정점을 만든다. 이미 저장돼 있으면 그것을 쓴다.

    사람이 UI에서 고친 측정점이 있으면 자동 역추적보다 우선한다 — 사람이 손댄
    것을 자동 결과로 덮어쓰면 검수한 의미가 없다.
    """
    out = {}
    for link in doc_data.get("links", []):
        m = link.get("measure")
        if m and m.get("points"):
            out[link["text_id"]] = {"points": [tuple(p) for p in m["points"]],
                                    "quality": m.get("quality", "human"),
                                    "source": m.get("source", "human")}
    if lines is None or len(out) == len(doc_data.get("links", [])):
        return out

    L = np.asarray(lines, float)
    polys = [t["poly"] for t in doc_data.get("texts", [])]
    params = tp.scaled_params(tp.text_scale(polys))
    id2idx = {f"l{i+1}": i for i in range(len(L))}
    for link in doc_data.get("links", []):
        tid = link["text_id"]
        if tid in out:
            continue
        idxs = [id2idx[x] for x in link.get("line_ids", []) if x in id2idx]
        if not idxs:
            continue
        i = max(idxs, key=lambda k: np.hypot(L[k, 2] - L[k, 0], L[k, 3] - L[k, 1]))
        r = tp.trace_measure_points(L[i], L, params, exclude_idx={i})
        out[tid] = {"points": r["points"], "quality": r["quality"], "source": "auto"}
    return out


def inspect_photo(doc_data, photo_path, board, px_per_mm=8.0,
                  drawing_lines=None, camera=None):
    """사진 1장에 대해 정합 + 측정 + 판정. 반환 dict(성공/실패 사유 포함)."""
    res = {"photo": photo_path, "ok": False, "reason": None, "results": []}
    img = imread_unicode(photo_path)
    if img is None:
        res["reason"] = "사진을 읽지 못했습니다"
        return res

    # ① 보정
    try:
        r = cal.rectify_board(img, board, px_per_mm=px_per_mm,
                              camera_matrix=(camera or {}).get("camera_matrix"),
                              dist_coeffs=(camera or {}).get("dist_coeffs"))
    except ValueError as e:
        res["reason"] = f"보정 실패: {e}"
        return res
    res["calibration"] = {"n_markers": r["n_markers"],
                          "residual_mm": round(r["residual_mm"], 3),
                          "warnings": r["warnings"]}

    # ② 사진 기하
    geom = reg.extract_geometry(r["rectified"])
    res["photo_geometry"] = {"n_lines": int(len(geom["lines"])),
                             "n_circles": int(len(geom["circles"]))}

    # ③ 정합 — 도면 원 ↔ 사진 원
    T = reg.register_from_docs(doc_data.get("arcs", []), geom["arcs"], r["px_per_mm"])
    if T is None:
        res["reason"] = ("원 배치 정합 실패 — 도면 또는 사진에 원이 부족합니다. "
                         "검수 UI 측정 모드에서 수동으로 재세요.")
        res["rectified_shape"] = list(r["rectified"].shape[:2])
        return res
    res["registration"] = {"scale": round(T["scale"], 5),
                           "rms_mm": round(T["rms_mm"], 3),
                           "n_inliers": len(T["inliers"]),
                           "flipped": bool(T["flip"] < 0),
                           "drawing_px_per_mm": (None if T["drawing_px_per_mm"] is None
                                                 else round(T["drawing_px_per_mm"], 4))}

    # ④ 치수별 측정
    mp = measure_points_from_doc(doc_data, drawing_lines)
    texts = {t["id"]: t for t in doc_data.get("texts", [])}
    search_px = LOCAL_SEARCH_MM * r["px_per_mm"]
    for tid, info in mp.items():
        t = texts.get(tid)
        if t is None:
            continue
        parsed = cp.parse_dimension(t.get("text", ""))
        src = np.array(info["points"], float)
        if T["flip"] < 0:
            src = src.copy(); src[:, 0] *= -1
        q = reg.apply_transform(src, T["scale"], T["R"], T["t"])
        m = ms.measure_two_points(r["rectified"], q[0], q[1], r["px_per_mm"],
                                  snap=True, search_px=int(search_px))
        ctr = np.array(r["marker_origin_px"])
        dist_mm = float(np.hypot(*((q[0] + q[1]) / 2 - ctr))) / r["px_per_mm"]
        u = cal.measurement_uncertainty(m["mm"], dist_mm,
                                        board.getMarkerLength() * 3,
                                        px_per_mm=r["px_per_mm"])
        j = cp.judge(parsed, m["mm"], u["total"])
        res["results"].append({
            "text_id": tid, "text": t.get("text", ""),
            "nominal": parsed["nominal"], "upper": parsed["upper"],
            "lower": parsed["lower"],
            "measured_mm": round(float(m["mm"]), 3),
            "uncertainty_mm": round(float(u["total"]), 3),
            "deviation_mm": None if j["deviation"] is None else round(j["deviation"], 3),
            "verdict": j["verdict"], "reason": j["reason"],
            "point_quality": info["quality"], "point_source": info["source"],
            "points": [[float(v) for v in q[0]], [float(v) for v in q[1]]],
            "source": "auto", "verified": False,
        })
    res["ok"] = True
    return res


def run_inspection(doc_data, photo_paths, marker_mm=50.0, pitch_mm=150.0,
                   px_per_mm=8.0, drawing_lines=None, camera=None):
    """도면 문서 + 사진 여러 장(3면도 등) -> 통합 결과.

    같은 치수가 여러 사진에서 측정되면 불확실도가 가장 작은 것을 채택한다.
    """
    board = cal.make_board(2, 2, marker_mm=marker_mm, gap_mm=pitch_mm - marker_mm)
    per_photo = [inspect_photo(doc_data, p, board, px_per_mm, drawing_lines, camera)
                 for p in photo_paths]

    best = {}
    for pr in per_photo:
        for row in pr.get("results", []):
            cur = best.get(row["text_id"])
            if cur is None or row["uncertainty_mm"] < cur["uncertainty_mm"]:
                row = dict(row, photo=pr["photo"])
                best[row["text_id"]] = row

    counts = {}
    for row in best.values():
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    return {"photos": per_photo, "results": list(best.values()), "summary": counts}


def format_report(out):
    """사람이 읽을 요약. 판정 못 한 것과 그 이유를 반드시 같이 낸다."""
    lines = []
    for pr in out["photos"]:
        tag = "OK" if pr["ok"] else "실패"
        lines.append(f"[{tag}] {os.path.basename(pr['photo'])}"
                     + (f"  — {pr['reason']}" if pr.get("reason") else ""))
        if pr.get("registration"):
            g = pr["registration"]
            lines.append(f"     정합 인라이어 {g['n_inliers']}개, RMS {g['rms_mm']}mm"
                         + (", 거울상" if g["flipped"] else "")
                         + f", 도면 {g['drawing_px_per_mm']}px/mm")
    lines.append("")
    lines.append(f"{'치수':<16}{'공칭':>8}{'실측':>18}{'편차':>9}  판정")
    for r in sorted(out["results"], key=lambda x: x["text"]):
        dev = "-" if r["deviation_mm"] is None else f"{r['deviation_mm']:+.2f}"
        nom = "-" if r["nominal"] is None else f"{r['nominal']:g}"
        v = {"pass": "합격", "fail": "불합격", "borderline": "경계",
             "inconclusive": "판정불가", "unknown": "판정불가"}[r["verdict"]]
        lines.append(f"{r['text'][:15]:<16}{nom:>8}"
                     f"{r['measured_mm']:>12.2f} ±{r['uncertainty_mm']:.2f}{dev:>9}  {v}"
                     + (f"   ({r['reason']})" if r.get("reason") else ""))
    lines.append("")
    lines.append("요약: " + "  ".join(f"{k} {v}" for k, v in out["summary"].items()))
    return "\n".join(lines)
