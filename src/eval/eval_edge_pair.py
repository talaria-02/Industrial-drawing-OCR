# -*- coding: utf-8 -*-
"""G0(엣지쌍 병합) 검증 — 출처별 층화 표본에서 라벨 없이 품질을 잰다.

[왜 층화 표본인가]
data/real/train 148장은 최소 12개 출처에서 왔다(V01-055, UVW, XYZ, W502, LMN ...).
회사마다 제도 관행(선 굵기, 화살촉 모양, 해칭 각도)이 달라서 한 출처에 몰아
검증하면 과적합을 놓친다. 출처별로 고르게 뽑는다.

[지표 — 전부 라벨 없이 계산됨]
  1. 중심선 잉크 적중률(병합분)  : 병합 결과가 실제 잉크 위에 놓였는가.
                                   여백을 잘못 병합하면 흰 바탕에 뜬다 -> 오병합 직접 탐지.
  2. 짝지음률                    : 선분 중 몇 %가 짝을 찾았는가.
  3. 두께/간격 자가보정값        : 도면마다 제대로 잡히는가(고정값 회귀 방지).
  4. 길이 불일치                 : 부분겹침 분할 로직이 필요한지 판단 근거.
  5. 기존 파이프라인과 최종 개수 비교.

실행: python src/eval/eval_edge_pair.py [표본수]
"""
import os
import re
import sys
import glob
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.line_detect import backend_lsd, refine, edge_pair  # noqa: E402

DATA_DIR = os.path.join("data", "real", "train")
OUT_DIR = os.path.join("results", "line_detect", "edge_pair")


def source_key(path):
    """파일명에서 출처를 뽑는다 (숫자 일련번호를 뭉개서 묶음)."""
    name = os.path.basename(path)
    name = re.sub(r"_page.*$", "", name)
    name = re.sub(r"\.rf\..*$", "", name)
    return re.sub(r"\d{3,}", "#", name)


def stratified_sample(files, per_source=2, limit=24):
    """출처별로 per_source장씩 뽑되 전체 limit장을 넘지 않게."""
    groups = defaultdict(list)
    for f in files:
        groups[source_key(f)].append(f)
    # 출처가 많은 순으로 돌면서 라운드로빈 — 소수 출처도 반드시 포함되게
    ordered = sorted(groups.values(), key=len, reverse=True)
    picked = []
    for r in range(per_source):
        for g in ordered:
            if r < len(g) and len(picked) < limit:
                picked.append(g[r])
    return picked


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)


def evaluate(path, save_overlay=False):
    gray = imread_unicode(path)
    if gray is None:
        return None
    raw = backend_lsd.detect(gray)
    if len(raw) < 10:
        return None

    # 기존 파이프라인 (비교군)
    old_lines, _ = refine.refine_pipeline(raw)
    old_ink, _ = edge_pair.centerline_ink_rate(old_lines, gray)

    # G0 — 조각잇기까지만 돌린 뒤 엣지쌍 단계를 따로 재서 지표를 분리한다
    pre = refine.merge_collinear(refine.length_nms(raw, 8.0), 1.5, 1.0, 15.0)
    merged, meta = edge_pair.merge_edge_pairs(pre, gray)
    paired = meta["paired"]
    st = meta["stats"]

    ink_pair, _ = edge_pair.centerline_ink_rate(merged[paired], gray)
    ink_unpair, _ = edge_pair.centerline_ink_rate(merged[~paired], gray)

    new_lines, new_stats = refine.refine_pipeline_g0(raw, gray)
    new_ink, _ = edge_pair.centerline_ink_rate(new_lines, gray)

    if save_overlay:
        os.makedirs(OUT_DIR, exist_ok=True)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in merged[~paired]:
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 120, 0), 2)
        for x1, y1, x2, y2 in merged[paired]:
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        name = os.path.splitext(os.path.basename(path))[0][:40] + "_g0.png"
        cv2.imencode(".png", vis)[1].tofile(os.path.join(OUT_DIR, name))

    return {
        "src": source_key(path),
        "n_raw": len(raw),
        "n_pre": len(pre),
        "n_pairs": st["n_pairs"],
        "pair_rate": st["n_pairs"] * 2 / max(1, len(pre)) * 100,
        "gap_mode": st.get("gap_mode", float("nan")),
        "thick_med": st.get("thickness_median", float("nan")),
        "contrast_T": st.get("contrast_thresh", float("nan")),
        "ink_pair": ink_pair * 100,
        "ink_unpair": ink_unpair * 100,
        "mismatch_med": st.get("length_mismatch_median", float("nan")) * 100,
        "mismatch_p90": st.get("length_mismatch_p90", float("nan")) * 100,
        "old_final": len(old_lines),
        "new_final": len(new_lines),
        "old_ink": old_ink * 100,
        "new_ink": new_ink * 100,
    }


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")))
    sample = stratified_sample(files, per_source=2, limit=limit)
    print(f"전체 {len(files)}장 중 층화 표본 {len(sample)}장 "
          f"(출처 {len(set(source_key(f) for f in sample))}종)")
    print("=" * 118)
    hdr = (f"{'출처':<22}{'raw':>6}{'조각후':>7}{'짝':>6}{'짝률':>7}"
           f"{'간격g*':>7}{'두께':>6}{'T':>5}{'잉크(병합)':>10}{'잉크(미짝)':>10}"
           f"{'불일치50/90':>12}{'최종 기존→G0':>14}")
    print(hdr)
    print("-" * 118)

    rows = []
    for i, f in enumerate(sample):
        r = evaluate(f, save_overlay=(i < 6))
        if r is None:
            continue
        rows.append(r)
        print(f"{r['src'][:21]:<22}{r['n_raw']:>6}{r['n_pre']:>7}{r['n_pairs']:>6}"
              f"{r['pair_rate']:>6.0f}%{r['gap_mode']:>7.2f}{r['thick_med']:>6.2f}"
              f"{r['contrast_T']:>5.0f}{r['ink_pair']:>9.1f}%{r['ink_unpair']:>9.1f}%"
              f"{r['mismatch_med']:>6.0f}/{r['mismatch_p90']:<5.0f}"
              f"{r['old_final']:>6}→{r['new_final']:<7}")

    if not rows:
        print("표본 없음")
        return
    print("-" * 118)
    agg = lambda k: np.array([r[k] for r in rows], dtype=float)
    print(f"{'중앙값':<22}{np.median(agg('n_raw')):>6.0f}{np.median(agg('n_pre')):>7.0f}"
          f"{np.median(agg('n_pairs')):>6.0f}{np.median(agg('pair_rate')):>6.0f}%"
          f"{np.median(agg('gap_mode')):>7.2f}{np.median(agg('thick_med')):>6.2f}"
          f"{np.median(agg('contrast_T')):>5.0f}{np.median(agg('ink_pair')):>9.1f}%"
          f"{np.median(agg('ink_unpair')):>9.1f}%"
          f"{np.median(agg('mismatch_med')):>6.0f}/{np.median(agg('mismatch_p90')):<5.0f}"
          f"{np.median(agg('old_final')):>6.0f}→{np.median(agg('new_final')):<7.0f}")
    print()
    ip = agg("ink_pair")
    print(f"[G0 게이트] 병합 중심선 잉크 적중률: "
          f"최소 {ip.min():.1f}%  중앙 {np.median(ip):.1f}%  최대 {ip.max():.1f}%")
    print(f"            90% 미만인 도면: {int(np.sum(ip < 90))}/{len(ip)}장")
    gm = agg("gap_mode")
    print(f"[자가보정 ] 간격 g* 범위 {gm.min():.2f}~{gm.max():.2f}px "
          f"(도면마다 달라지는 게 정상 — 고정값이었다면 여기서 깨졌을 것)")
    mm = agg("mismatch_med")
    print(f"[분할필요?] 길이 불일치 중앙값의 중앙 {np.median(mm):.0f}% "
          f"— 낮으면 부분겹침 분할 로직 불필요")
    print(f"\n오버레이 저장: {OUT_DIR}  (빨강=병합 중심선, 파랑=짝없는 선분)")


if __name__ == "__main__":
    main()
