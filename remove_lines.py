"""
점선/치수선 제거 전처리
========================
OCR 전에 도면의 점선(치수선·중심선)을 지워 오인식과 박스 뭉침을 막는다.

원리:
  1) 대시 잇기      — 수평/수직 방향 closing으로 점선을 연속선으로 연결
  2) 장선 추출      — 긴 커널 opening: 글자 획(짧음)은 사라지고 선만 남음
  3) 숫자 보호      — 원본의 글자급 연결요소 영역을 마스크에서 제외
                      (선이 숫자를 관통해도 숫자 획이 깎이지 않게)
  4) 선 제거        — 원본에서 마스크 픽셀을 배경색으로 덮음

사용:
  from remove_lines import remove_dashed_lines
  cleaned = remove_dashed_lines(img_bgr)          # BGR in → BGR out

  python remove_lines.py real_images/01.png       # 단독 실행: 전후 이미지 저장
"""

import sys
from pathlib import Path

import cv2
import numpy as np


def remove_dashed_lines(img,
                        bridge_gap=25,      # 대시 사이 갭 잇기 폭(px)
                        min_line_len=80,    # 이 길이 이상만 '선'으로 인정(px)
                        text_max_h=40,      # 글자급 blob 최대 높이(px)
                        text_pad=3):        # 글자 보호 팽창 반경(px)
    """
    도면 이미지에서 수평/수직 점선·실선을 제거.
    글자(숫자) 픽셀은 보호. BGR 입력 → BGR 출력.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── 1) 대시 잇기 + 2) 장선 추출 ──────────────────────────
    # 수평: 가로로 closing(갭 연결) 후 긴 가로 opening(선만 생존)
    close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_gap, 1))
    open_h = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_h)
    h_lines = cv2.morphologyEx(h_lines, cv2.MORPH_OPEN, open_h)

    # 수직
    close_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge_gap))
    open_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_v)
    v_lines = cv2.morphologyEx(v_lines, cv2.MORPH_OPEN, open_v)

    line_mask = cv2.bitwise_or(h_lines, v_lines)
    # 닫기로 두꺼워진 부분 원본 잉크로 한정 + 경계 여유 1px
    line_mask = cv2.bitwise_and(line_mask, cv2.dilate(binary, np.ones((3, 3), np.uint8)))

    # ── 3) 숫자(글자급 blob) 보호 ────────────────────────────
    def collect_text_blobs(src):
        """글자급 blob의 bbox 영역 마스크. 양방향 종횡비 제한으로 대시 배제."""
        mask = np.zeros_like(binary)
        n, _, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
        for i in range(1, n):
            x, y, w_, h_, area = stats[i]
            if (10 <= h_ <= text_max_h and w_ <= text_max_h * 4
                    and w_ / max(1, h_) <= 4 and h_ / max(1, w_) <= 4
                    and area >= 20):
                mask[y:y + h_, x:x + w_] = 255
        return mask

    # 1차: 원본 blob 기준 글자 보호
    text_mask = collect_text_blobs(binary)
    # 2차: 선을 뺀 잔여물에서 다시 글자 탐색
    #      (선이 숫자를 관통해 한 덩어리가 된 경우, 선 제거 후 분리된
    #       숫자가 여기서 발견됨 — '20'이 대시와 붙어 '2'만 남는 문제 방지)
    residual = cv2.subtract(binary, line_mask)
    text_mask = cv2.bitwise_or(text_mask, collect_text_blobs(residual))
    text_mask = cv2.dilate(text_mask,
                           np.ones((text_pad * 2 + 1, text_pad * 2 + 1), np.uint8))

    removal = cv2.bitwise_and(line_mask, cv2.bitwise_not(text_mask))

    # ── 4) 선 제거: 배경색(주변 중앙값)으로 덮음 ─────────────
    bg = int(np.median(gray[binary == 0])) if (binary == 0).any() else 255
    out = img.copy()
    out[removal > 0] = (bg, bg, bg)
    return out


# ════════════════════════════════════════════════════════════
# v3: Fletcher-Kasturi식 공선 체인 판정
#   대시/글자를 blob "모양"이 아니라 "배열 패턴"으로 구분:
#   일직선 + 등간격 + 가늘고 균일 = 점선 체인 (각도 무관 → 대각선 OK)
#   체인 양 끝의 작은 blob(화살촉)도 흡수해 함께 제거.
# ════════════════════════════════════════════════════════════

def _find_chain(pts, eligible, start, second, used, r_max, perp_tol):
    """
    start→second 방향으로 등간격 이웃을 양방향 탐색해 체인 인덱스 반환.
    eligible[k]=False인 노드(굵은 blob=숫자 등)는 체인에 편입 불가 —
    선 위에 박힌 숫자에서 체인이 끊기고 숫자는 생존한다.
    """
    p0, p1 = pts[start], pts[second]
    d = p1 - p0
    norm = np.hypot(*d)
    if norm < 4:
        return [start, second]
    u = d / norm                       # 진행 방향 단위벡터

    chain = [start, second]
    # 앞으로 확장
    cur = second
    while True:
        best, best_dist = -1, r_max
        for k in range(len(pts)):
            if k in used or k in chain or not eligible[k]:
                continue
            v = pts[k] - pts[cur]
            along = v @ u
            if not (4 < along < best_dist):
                continue
            perp = abs(v[0] * u[1] - v[1] * u[0])
            if perp <= perp_tol:
                best, best_dist = k, along
        if best < 0:
            break
        chain.append(best)
        cur = best
    # 뒤로 확장
    cur = start
    while True:
        best, best_dist = -1, r_max
        for k in range(len(pts)):
            if k in used or k in chain or not eligible[k]:
                continue
            v = pts[cur] - pts[k]
            along = v @ u
            if not (4 < along < best_dist):
                continue
            perp = abs(v[0] * u[1] - v[1] * u[0])
            if perp <= perp_tol:
                best, best_dist = k, along
        if best < 0:
            break
        chain.insert(0, best)
        cur = best
    return chain


def remove_dashed_lines_fk(img,
                           max_dash=45,     # 대시 후보 blob 최대 변(px)
                           min_members=3,   # 체인 최소 멤버 수 (짧은 대시 런도 인정)
                           r_max=110,       # 이웃 탐색 반경(px) — 화살촉+숫자 건너뛸 수 있게
                           perp_tol=5,      # 직선 이탈 허용(px)
                           thin_max=5,      # 대시 굵기(min dim) 상한 — '1'(약 6px)보다 얇게
                           arrow_area=400,  # 끝단 흡수(화살촉) 최대 면적
                           return_mask=False):
    """
    F-K식: 작은 blob 중심점의 공선·등간격 체인 = 점선으로 제거.
    글자는 모양 규칙 없이도 생존 (체인 패턴에 안 맞으므로).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 대시 후보: 작은 blob 전부 (글자도 포함 — 체인 검사가 구분)
    cand = [i for i in range(1, n)
            if stats[i, cv2.CC_STAT_WIDTH] <= max_dash
            and stats[i, cv2.CC_STAT_HEIGHT] <= max_dash
            and stats[i, cv2.CC_STAT_AREA] >= 3]
    if len(cand) < min_members:
        return (img.copy(), np.zeros_like(binary)) if return_mask else img.copy()

    pts = np.array([cents[i] for i in cand])
    sizes = np.array([[stats[i, cv2.CC_STAT_WIDTH],
                       stats[i, cv2.CC_STAT_HEIGHT]] for i in cand])
    thin = np.minimum(sizes[:, 0], sizes[:, 1])   # blob 굵기
    # 대시 자격: 가늘어야 함 (숫자/글자는 굵어서 탈락 → 체인이 숫자에서 끊김)
    eligible = thin <= thin_max

    used = set()
    dash_members = set()          # cand 인덱스
    chains = []                   # (멤버 리스트, 방향 단위벡터)

    for a in np.argsort(thin):    # 가는 blob부터 시드
        if a in used or not eligible[a]:
            continue
        # 가까운 이웃을 시드 파트너로
        d2 = np.hypot(pts[:, 0] - pts[a, 0], pts[:, 1] - pts[a, 1])
        for b in np.argsort(d2)[1:6]:
            if b in used or d2[b] > r_max or not eligible[b]:
                continue
            chain = _find_chain(pts, eligible, a, int(b), used, r_max, perp_tol)
            if len(chain) < min_members:
                continue
            # 등간격 검사
            cpts = pts[chain]
            gaps = np.hypot(np.diff(cpts[:, 0]), np.diff(cpts[:, 1]))
            if gaps.std() / max(1e-6, gaps.mean()) > 1.2:
                continue
            # 굵기 균일·가늘음 검사 (글자열 배제: 글자는 굵음)
            if np.median(thin[chain]) > thin_max:
                continue
            used.update(chain)
            dash_members.update(chain)
            u = (cpts[-1] - cpts[0]) / max(1e-6, np.hypot(*(cpts[-1] - cpts[0])))
            chains.append((chain, u))
            break

    # 화살촉 흡수: 체인 양 끝 근처 + 체인 수직방향으로 납작한 blob만
    # (선 안에 박힌 숫자는 수직으로 두꺼워 제외됨)
    for chain, u in chains:
        ends = [pts[chain[0]], pts[chain[-1]]]
        for k in range(len(cand)):
            if k in dash_members:
                continue
            if stats[cand[k], cv2.CC_STAT_AREA] > arrow_area:
                continue
            # 체인 진행방향 기준 수직 폭 (화살촉: 납작 / 숫자: 두꺼움)
            w_k, h_k = sizes[k]
            perp_size = w_k * abs(u[1]) + h_k * abs(u[0])
            if perp_size > 14:
                continue
            # 화살촉은 짧다 — 긴 blob('1' 등 글자)은 흡수 금지
            if max(w_k, h_k) > 16:
                continue
            dmin = min(np.hypot(*(pts[k] - e)) for e in ends)
            if dmin <= r_max:
                dash_members.add(k)

    # 제거: 멤버 blob의 실제 픽셀만 배경색으로
    removal = np.zeros_like(binary)
    for k in dash_members:
        removal[labels == cand[k]] = 255

    bg = int(np.median(gray[binary == 0])) if (binary == 0).any() else 255
    out = img.copy()
    out[removal > 0] = (bg, bg, bg)
    if return_mask:
        return out, removal
    return out


def main():
    args = [a for a in sys.argv[1:] if a != '--fk']
    use_fk = '--fk' in sys.argv[1:]
    path = Path(args[0])
    arr = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    cleaned = remove_dashed_lines_fk(img) if use_fk else remove_dashed_lines(img)

    suffix = '_fk' if use_fk else '_nolines'
    out = path.parent / f'{path.stem}{suffix}.png'
    cv2.imencode('.png', cleaned)[1].tofile(str(out))
    print(f"저장: {out}")


if __name__ == '__main__':
    main()
