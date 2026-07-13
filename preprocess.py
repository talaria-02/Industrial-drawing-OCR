"""
도면 치수 추출을 위한 전처리 파이프라인 (딥러닝 X, 순수 OpenCV)

파이프라인 순서:
  로드/확대 → 이진화 → 노이즈 제거 → 기울기 보정(deskew)
  → 레이어 분리(선/텍스트) → 텍스트 ROI 추출

각 단계 결과를 output 폴더에 저장하여 파라미터 튜닝을 눈으로 확인할 수 있게 함.

사용법:
  python preprocess.py 도면.png
  python preprocess.py 도면.png --scan      # 스캔/촬영본 (adaptive threshold)
  python preprocess.py 도면.png --scale 3   # 글자가 작으면 확대 배율 증가
"""

import cv2
import numpy as np
import argparse
import os


class DrawingPreprocessor:
    def __init__(self, scale=2, is_scan=False, debug_dir="debug"):
        self.scale = scale            # 확대 배율 (치수 글자가 작을 때 키움)
        self.is_scan = is_scan        # 스캔/촬영본이면 adaptive threshold 사용
        self.debug_dir = debug_dir    # 중간 결과 저장 폴더
        os.makedirs(debug_dir, exist_ok=True)
        self.step = 0

    def _save(self, name, img):
        """중간 결과 저장 (튜닝 확인용)"""
        self.step += 1
        path = os.path.join(self.debug_dir, f"{self.step:02d}_{name}.png")
        cv2.imwrite(path, img)
        print(f"  [저장] {path}")

    # ------------------------------------------------------------------
    # 0. 로드 & 정규화
    # ------------------------------------------------------------------
    def load(self, path):
        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"이미지를 열 수 없습니다: {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 치수 글자가 작아 OCR이 놓치는 경우가 많으므로 확대
        if self.scale != 1:
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_CUBIC)
        self._save("gray", gray)
        return gray

    # ------------------------------------------------------------------
    # 1. 이진화
    # ------------------------------------------------------------------
    def binarize(self, gray):
        if self.is_scan:
            # 조명 불균일한 스캔/촬영본
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 8)
        else:
            # 깨끗한 CAD 출력물
            _, binary = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # 선/글자가 흰색(255), 배경이 검은색(0) 상태
        self._save("binary", binary)
        return binary

    # ------------------------------------------------------------------
    # 2. 노이즈 제거
    # ------------------------------------------------------------------
    def denoise(self, binary):
        # (a) 작은 점 노이즈 제거: 연결요소 면적 필터
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = binary.copy()
        removed = 0
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 10 * self.scale:  # 배율 보정
                cleaned[labels == i] = 0
                removed += 1
        print(f"  [노이즈] 작은 성분 {removed}개 제거")

        # (b) 끊긴 선 살짝 잇기 (글자 뭉치지 않게 커널 작게)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        self._save("denoised", cleaned)
        return cleaned

    # ------------------------------------------------------------------
    # 3. 기울기 보정 (deskew)
    # ------------------------------------------------------------------
    def deskew(self, binary, gray):
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=10)
        if lines is None:
            print("  [deskew] 직선 미검출 → 보정 생략")
            return binary, gray

        angles = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(a) < 45:            # 수평에 가까운 선만 사용
                angles.append(a)
        if not angles:
            print("  [deskew] 수평선 미검출 → 보정 생략")
            return binary, gray

        skew = float(np.median(angles))
        print(f"  [deskew] 기울기 {skew:.2f}° 보정")
        if abs(skew) < 0.1:            # 거의 수평이면 생략
            return binary, gray

        h, w = binary.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
        binary = cv2.warpAffine(binary, M, (w, h),
                                flags=cv2.INTER_NEAREST, borderValue=0)
        gray = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC, borderValue=255)
        self._save("deskewed", binary)
        return binary, gray

    # ------------------------------------------------------------------
    # 4. 레이어 분리 (선 / 텍스트)  ← 도면 전처리의 핵심
    # ------------------------------------------------------------------
    def separate_layers(self, binary):
        # 선(치수선/외곽선)의 최소 길이 기준. 배율에 비례.
        line_len = int(40 * self.scale)

        # 긴 수평선
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        # 긴 수직선
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        lines_layer = cv2.bitwise_or(h_lines, v_lines)

        # 텍스트 레이어 = 전체 - 선
        text_layer = cv2.subtract(binary, lines_layer)
        # 선 제거 후 남은 자잘한 잔선 정리
        text_layer = cv2.morphologyEx(
            text_layer, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

        self._save("layer_lines", lines_layer)
        self._save("layer_text", text_layer)
        return lines_layer, text_layer

    # ------------------------------------------------------------------
    # 5. 텍스트 ROI 추출 (OCR 입력 영역)
    # ------------------------------------------------------------------
    def extract_text_rois(self, text_layer, gray):
        # 인접한 글자를 하나의 치수 텍스트 덩어리로 뭉침
        merge_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (int(15 * self.scale), int(5 * self.scale)))
        merged = cv2.morphologyEx(text_layer, cv2.MORPH_CLOSE, merge_kernel)

        contours, _ = cv2.findContours(
            merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rois = []
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            ratio = h / w if w > 0 else 999
            # 너무 작거나 비율이 이상한 덩어리 제외
            if area > 50 * self.scale and 0.1 < ratio < 10:
                rois.append((x, y, w, h))
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

        print(f"  [ROI] 텍스트 후보 {len(rois)}개 검출")
        self._save("text_rois", vis)
        # 좌→우, 위→아래 순 정렬
        rois.sort(key=lambda r: (r[1] // 20, r[0]))
        return rois

    # ------------------------------------------------------------------
    # 전체 파이프라인 실행
    # ------------------------------------------------------------------
    def run(self, path):
        print(f"\n=== 전처리 시작: {path} ===")
        gray = self.load(path)
        binary = self.binarize(gray)
        binary = self.denoise(binary)
        binary, gray = self.deskew(binary, gray)
        lines_layer, text_layer = self.separate_layers(binary)
        rois = self.extract_text_rois(text_layer, gray)
        print(f"=== 완료. 중간 결과는 '{self.debug_dir}/' 폴더 확인 ===\n")

        return {
            "gray": gray,                # 원본(확대·보정) 그레이스케일
            "binary": binary,            # 이진화·정제본
            "lines_layer": lines_layer,  # 선 레이어 → Hough 치수선 검출용
            "text_layer": text_layer,    # 텍스트 레이어
            "text_rois": rois,           # (x,y,w,h) 리스트 → OCR 입력 영역
        }


def main():
    ap = argparse.ArgumentParser(description="도면 치수 추출용 전처리")
    ap.add_argument("image", help="입력 도면 이미지 경로")
    ap.add_argument("--scan", action="store_true",
                    help="스캔/촬영본이면 지정 (adaptive threshold)")
    ap.add_argument("--scale", type=int, default=2,
                    help="확대 배율 (기본 2, 글자 작으면 3)")
    ap.add_argument("--debug-dir", default="debug",
                    help="중간 결과 저장 폴더")
    args = ap.parse_args()

    pre = DrawingPreprocessor(scale=args.scale, is_scan=args.scan,
                              debug_dir=args.debug_dir)
    result = pre.run(args.image)

    # OCR 단계로 넘길 ROI들을 잘라서 crops 폴더에 저장 (다음 단계 연결용)
    crop_dir = os.path.join(args.debug_dir, "crops")
    os.makedirs(crop_dir, exist_ok=True)
    gray = result["gray"]
    for idx, (x, y, w, h) in enumerate(result["text_rois"]):
        pad = 3
        crop = gray[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        cv2.imwrite(os.path.join(crop_dir, f"roi_{idx:03d}.png"), crop)
    print(f"OCR용 crop {len(result['text_rois'])}개 저장 → {crop_dir}/")


if __name__ == "__main__":
    main()