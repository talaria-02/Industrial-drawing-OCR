# -*- coding: utf-8 -*-
"""원/호 추출 검증 — 출처별 층화 표본. 라벨 없이 품질을 잰다.

[지표]
  1. 잉크 적중률   : 피팅된 원호를 훑어 잉크 위에 놓인 표본 비율. 유령 원 탐지.
  2. 각도폭        : 조각남 정도. 낮으면 원 하나가 여러 조각으로 남았다는 뜻.
  3. 완전원 수     : 각도폭 300도 이상.
  4. 흡수 비율     : 선분 풀에서 빠진 비율. 곧 매칭 후보 노이즈 감소분.
  5. 잉크 거부 수  : 출력검증에서 걸러낸 유령 원(등각뷰 타원 조각 등).

실행: python src/eval/eval_arc_detect.py [표본수]
"""
import os
import sys
import glob
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.line_detect import backend_lsd, refine, edge_pair, arc_detect  # noqa: E402
from eval_edge_pair import stratified_sample, source_key, imread_unicode, DATA_DIR  # noqa: E402


def evaluate(path):
    gray = imread_unicode(path)
    if gray is None:
        return None
    raw = backend_lsd.detect(gray)
    if len(raw) < 20:
        return None
    pre = refine.merge_collinear(refine.length_nms(raw, 8.0), 1.5, 1.0, 15.0)
    merged, meta = edge_pair.merge_edge_pairs(pre, gray)
    gap = meta["stats"].get("gap_mode")

    t0 = time.time()
    rem, arcs, st = arc_detect.extract_arcs(merged, gray, stroke_gap=gap)
    elapsed = time.time() - t0

    # 원이 하나도 없는 도면(샤프트 측면도 등)이 정상적으로 존재하므로,
    # 통계값은 NaN 대신 명시적으로 비워두고 집계에서 제외한다.
    if arcs:
        spans = np.array([a["span"] for a in arcs])
        inks = np.array([arc_detect._ink_fraction(a["cx"], a["cy"], a["r"],
                                                  a["start"], a["span"], gray)
                         for a in arcs])
        radii = np.array([a["r"] for a in arcs])
        n_paired = sum(1 for a in arcs if a["paired"])
        span_med, ink_med = float(np.median(spans)), float(np.median(inks)) * 100
        ink_min, r_med = float(inks.min()) * 100, float(np.median(radii))
    else:
        span_med = ink_med = ink_min = r_med = float("nan")
        n_paired = 0

    return {
        "src": source_key(path),
        "n_lines": len(merged),
        "n_arcs": st["n_arcs"],
        "n_full": st["n_full_circles"],
        "n_paired": n_paired,
        "absorbed": st["absorbed_pct"],
        "rejected": st["n_rejected_by_ink"],
        "span_med": span_med,
        "ink_med": ink_med,
        "ink_min": ink_min,
        "r_med": r_med,
        "sec": elapsed,
    }


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")))
    sample = stratified_sample(files, per_source=2, limit=limit)
    print(f"층화 표본 {len(sample)}장")
    print("=" * 104)
    print(f"{'출처':<24}{'선분':>7}{'원':>5}{'완전원':>7}{'동심병합':>9}{'흡수':>7}"
          f"{'잉크거부':>9}{'각도폭중앙':>11}{'잉크중앙':>9}{'잉크최소':>9}{'반지름중앙':>11}{'초':>6}")
    print("-" * 104)
    rows = []
    for f in sample:
        r = evaluate(f)
        if r is None:
            continue
        rows.append(r)
        print(f"{r['src'][:23]:<24}{r['n_lines']:>7}{r['n_arcs']:>5}{r['n_full']:>7}"
              f"{r['n_paired']:>9}{r['absorbed']:>6.1f}%{r['rejected']:>9}"
              f"{r['span_med']:>10.0f}°{r['ink_med']:>8.0f}%{r['ink_min']:>8.0f}%"
              f"{r['r_med']:>11.0f}{r['sec']:>6.1f}")
    if not rows:
        print("표본 없음")
        return
    print("-" * 104)
    agg = lambda k: np.array([r[k] for r in rows], dtype=float)
    with_arcs = [r for r in rows if r["n_arcs"] > 0]
    nz = lambda k: (lambda v: v[~np.isnan(v)])(agg(k))
    print(f"{'중앙값':<24}{np.median(agg('n_lines')):>7.0f}{np.median(agg('n_arcs')):>5.0f}"
          f"{np.median(agg('n_full')):>7.0f}{np.median(agg('n_paired')):>9.0f}"
          f"{np.median(agg('absorbed')):>6.1f}%{np.median(agg('rejected')):>9.0f}"
          f"{np.median(nz('span_med')):>10.0f}°{np.median(nz('ink_med')):>8.0f}%"
          f"{np.median(nz('ink_min')):>8.0f}%{np.median(nz('r_med')):>11.0f}"
          f"{np.median(agg('sec')):>6.1f}")
    print()
    print(f"원이 검출된 도면: {len(with_arcs)}/{len(rows)}장  "
          f"(원이 없는 도면에서 0개인 것은 정상 — 샤프트 측면도 등)")
    ink = agg("ink_med")
    ink = ink[~np.isnan(ink)]
    print(f"[게이트] 원호 잉크 적중률 중앙값의 최소: {ink.min():.0f}%  "
          f"— 낮으면 유령 원이 섞였다는 뜻")
    tot_abs = agg("absorbed")
    print(f"[효과 ] 선분 풀에서 빠진 비율: 중앙 {np.median(tot_abs):.1f}%  "
          f"최대 {tot_abs.max():.1f}%  — 그만큼 매칭 후보 노이즈가 준다")
    print(f"[비용 ] 도면당 {np.median(agg('sec')):.1f}초 (OCR 25~30초 대비 무시 가능)")


if __name__ == "__main__":
    main()
