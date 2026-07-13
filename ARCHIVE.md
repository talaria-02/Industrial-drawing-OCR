# 폐기 코드 기록 (ARCHIVE)

프로젝트 진행 중 작성되었으나 방향 전환/대체로 더 이상 사용하지 않아 삭제한
코드들의 기록. 파일 자체는 삭제됨 (2026-07-11 정리).

| 파일 | 역할 | 폐기 사유 |
|---|---|---|
| `create_3d.py` | Blender로 테스트 큐브 생성 (.blend/.glb 내보내기) | 초기 "3D 모델링 → 2D 투영" 방향 탐색용. 멘토 가이드로 치수 중심 접근으로 전환하며 3D 경로 폐기 |
| `create_data.py` | 합성 도면 생성기 v4 — 단일 도형(7종) 1장당 1개, 100장 | v5(`create_data_v5.py`)로 대체. 다중 뷰·글씨체 분산·기호·대노이즈 미지원. 산출물 `output_v4/`도 삭제됨 |
| `augment_data.py` | 이미지 증강 — 얼룩/회전(90·180·270°)/뒤집기 | 회전·뒤집기는 치수 검증 목적과 무관(도면 방향 파괴). 노이즈 증강은 v5 생성기 내부 `degrade()`로 흡수 |
| `ocr_drawing.py` | 합성 도면 v4 전용 Tesseract OCR 추출기 | 실도면 대응판(`ocr_real_drawing.py`)에 흡수 후 엔진 자체를 PaddleOCR로 교체 |
| `code.md` | `ocr_real_drawing.py`의 초보자용 해설 문서 (과거 LLM 작성) | 해설 대상 코드가 baseline 보존용으로만 남고 파이프라인이 PaddleOCR 기반으로 재편되어 내용 실효 |
| `test_rtdetr.py` | RT-DETR 단일 이미지 테스트 | 폴더 일괄 처리판 `test_rtdetr_all.py`의 부분집합이라 중복 |
| `debug/` (9.2MB) | `preprocess.py` 중간 단계 이미지 (1.1~5.5, 단계별 PNG + ROI crop) | 파라미터 튜닝 확인용 일회성 산출물. `preprocess.py` 재실행으로 언제든 재생성 가능 |
| `region_results/` 내 합성 시각화 대부분 (~150MB) | 합성 100장 영역분리 시각화 PNG/JSON | 재생성 가능(`python region_split.py output_v5/images/`). 대표 1장(`drawing_0000`)과 실도면 5장만 보존 |

## 참고: 보존 이유가 있는 레거시

| 파일 | 상태 | 보존 이유 |
|---|---|---|
| `ocr_real_drawing.py` | 레거시지만 보존 | Tesseract baseline(recall 35.1%) 재현용. `eval_ocr.py`가 import |
| `preprocess.py` | 미연결이지만 보존 | 선/텍스트 레이어 분리 — 향후 치수선-형상 연결 작업의 기반. 멘토 강조점(전처리) 증빙 |
| `paddle_real_results/` | 개선 전 결과물 | v1(1패스) vs v2(3패스) before/after 비교 발표 자료 |
| `test_rtdetr_all.py` + `eng_dwg_v1.pt` | 실험 보류 | 향후 "검출+인식" 2단계 구조의 검출기 후보 |
