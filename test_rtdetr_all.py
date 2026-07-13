from ultralytics import RTDETR
import cv2
import supervision as sv 
import os
from pathlib import Path

print("모델 로드 중...")
model_file = 'eng_dwg_v1.pt'
model = RTDETR(model_file)

image_dir = Path('real_images')
output_dir = Path('rtdetr_results')
output_dir.mkdir(exist_ok=True)
crops_dir = output_dir / 'crops'
crops_dir.mkdir(exist_ok=True)

bounding_box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_scale=2, text_thickness=2)

count = 0
for image_path in image_dir.glob('*.*'):
    if image_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
        continue
    
    print(f"--- 처리 중: {image_path.name} ---")
    img = cv2.imread(str(image_path))
    
    if img is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        continue

    # verbose=False를 통해 로그를 깔끔하게 유지
    results = model.predict(img, imgsz=1024, verbose=False) 
    detections = sv.Detections.from_ultralytics(results[0])
    
    if len(detections) == 0:
        print(f"  -> 발견된 객체가 없습니다.")
        continue

    # 클래스 이름 매핑
    labels = [
        f"{results[0].names[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    # 시각화 박스 그리기
    annotated_image = bounding_box_annotator.annotate(
        scene=img.copy(),
        detections=detections
    )
    annotated_image = label_annotator.annotate(annotated_image, detections=detections, labels=labels)

    # 원본 이미지에 박스가 그려진 전체 결과 저장
    cv2.imwrite(str(output_dir / f'annotated_{image_path.name}'), annotated_image)

    # Bounding Box 단위로 이미지 자르기 (Crop)
    for xyxy, class_id in zip(detections.xyxy, detections.class_id):
        cropped_image = sv.crop_image(image=img, xyxy=xyxy)
        count += 1
        class_name = results[0].names[class_id]
        cv2.imwrite(str(crops_dir / f'{image_path.stem}_{class_name}_{count}.png'), cropped_image)
        
    print(f"  -> {len(detections)}개의 영역을 찾아 크롭했습니다.")

print(f"\n모든 이미지 처리 완료! 전체 결과는 '{output_dir}', 크롭된 조각 {count}개는 '{crops_dir}'에 저장되었습니다.")
