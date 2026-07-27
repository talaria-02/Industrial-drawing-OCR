"""
###ZONE:DRAWING### 태그 박스 처리 유틸
========================================
사용자가 PPOCRLabel에서 도면영역 전체를 감싸는 박스 하나를 그리고
transcription을 "###ZONE:DRAWING###"으로 표시함. 증강(text-swap/
회전/포토메트릭)은 이 zone 안에 들어가는 item에만 적용하고, 밖(메타
데이터/타이틀블록)은 그대로 둠 — 이미 인식 잘 되는 영역이라 증강
투자할 필요 없음(단, 학습셋엔 그대로 포함시켜서 일반텍스트 감지력은
유지).

ZONE_TAG: 예약 transcription 문자열. items 리스트에서 이걸로 zone
박스를 찾고 나머지 실제 콘텐츠 item과 구분한다.
"""

ZONE_TAG = '###ZONE:DRAWING###'


def get_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def point_in_bbox(cx, cy, bbox):
    x0, y0, x1, y1 = bbox
    return x0 <= cx <= x1 and y0 <= cy <= y1


def split_zone(items):
    """items(한 이미지의 라벨 리스트)를 (zone_bbox, drawing_items, metadata_items)로 분리.
    zone 태그가 없으면 zone_bbox=None, 전부 metadata_items로 반환(아직 zone 라벨 안 된 이미지)."""
    zone_bbox = None
    for it in items:
        if it['transcription'] == ZONE_TAG:
            zone_bbox = get_bbox(it['points'])
            break

    if zone_bbox is None:
        return None, [], items

    drawing_items, metadata_items = [], []
    for it in items:
        if it['transcription'] == ZONE_TAG:
            continue
        bx0, by0, bx1, by1 = get_bbox(it['points'])
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        if point_in_bbox(cx, cy, zone_bbox):
            drawing_items.append(it)
        else:
            metadata_items.append(it)

    return zone_bbox, drawing_items, metadata_items


if __name__ == '__main__':
    import json
    with open('data/real/train/Label.txt', encoding='utf-8') as f:
        lines = f.readlines()
    tagged = 0
    for line in lines:
        path, js = line.split('\t', 1)
        items = json.loads(js)
        zone_bbox, drawing_items, metadata_items = split_zone(items)
        if zone_bbox is not None:
            tagged += 1
            print(f'{path}: zone={zone_bbox}  drawing={len(drawing_items)}  metadata={len(metadata_items)}')
    print(f'\n{tagged}/{len(lines)} 이미지에 zone 태그 있음')
