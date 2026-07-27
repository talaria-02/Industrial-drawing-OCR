# -*- coding: utf-8 -*-
"""숫자류(치수값) 판별 + OCR json -> 매칭용 target/텍스트bbox 변환. 여러 배치
스크립트(match_batch10, arrowhead_check 등)가 공유하는 헬퍼 — results/ 안
스크립트가 서로 import하던 것을 여기로 모음."""
import re

from . import match_numbers as mn

PALETTE = [
    (0, 140, 255), (255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 0, 255),
    (255, 140, 0), (0, 200, 200), (128, 0, 128), (0, 100, 0), (255, 0, 128),
    (128, 128, 0), (0, 0, 128),
]

_DECORATION_CHARS = ('ø', 'Ø', '⌀', 'φ', 'Φ', '±', '°')
# ISO 286 끼워맞춤 등급 문자 화이트리스트 (63RA 같은 거칠기값 오인 방지)
_ISO_FIT_TOKENS = ['js', 'JS'] + list('ABCDEFGHJKMNPRSTUVXYZ') + list('abcdefghjkmnprstuvxyz')
_TOLERANCE_SUFFIX_RE = re.compile(
    r'^(\d+)(?:' + '|'.join(sorted(_ISO_FIT_TOKENS, key=len, reverse=True)) + r')\d{0,2}$')


def is_numeric_target(text):
    s = text.strip()
    for ch in _DECORATION_CHARS:
        s = s.replace(ch, '')
    s = s.replace(' ', '').replace(',', '')
    if s.endswith('.'):
        s = s[:-1]
    plain_digits = s.replace('.', '').replace('-', '')
    if plain_digits.isdigit() and len(plain_digits) > 0:
        return True
    return bool(_TOLERANCE_SUFFIX_RE.match(s))


def sanitize_folder_name(name):
    bad_chars = '<>:"/\\|?*°'
    cleaned = ''.join(c for c in name if c not in bad_chars)
    return cleaned if cleaned else 'x'


def build_targets_and_text_bboxes(ocr_json):
    """반환: (targets 리스트, 전체 텍스트bbox 리스트(제외영역용, score>=0.5 전부))"""
    targets = []
    text_bboxes = []
    for d in ocr_json["detections"]:
        xs = [p[0] for p in d["poly"]]
        ys = [p[1] for p in d["poly"]]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        if d["score"] >= 0.5:
            text_bboxes.append(bbox)

        t = d["text"].strip()
        if d["score"] < 0.85 or not is_numeric_target(t):
            continue

        angle_deg, uncertain, aspect = mn.text_angle_from_poly(d["poly"])
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        targets.append({
            "label": t, "bbox": bbox, "cx": cx, "cy": cy,
            "angle_deg": angle_deg, "angle_uncertain": uncertain, "aspect": aspect,
        })
    return targets, text_bboxes
