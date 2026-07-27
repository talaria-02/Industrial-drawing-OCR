# -*- coding: utf-8 -*-
"""치수 텍스트 -> 공칭값/공차 파싱, 그리고 측정값과의 판정.

[왜 파싱이 까다로운가]
OCR이 읽어온 문자열은 표기가 제각각이다. 실제 도면에서 나온 형태:

    15 +0.01 -0.01     상하한 분리 표기
    42±0,1             유럽식 쉼표 소수점
    ø110H7             ISO 끼워맞춤 등급 (숫자 공차가 아예 없음)
    20 +0.1 0          한쪽만 0
    Ø 236              지름 기호 + 공백
    R4 typ.            주기(typical) 표기
    33°45'±0°30'       각도, 분 단위

여기에 OCR 오독까지 겹친다(0<->O, 1<->l, .<->,). 그래서 파서는 '읽어낸 것만
확실히 처리하고 나머지는 미상으로 남기는' 방향으로 만든다 — 억지로 숫자를
만들어내면 판정이 조용히 틀린다.

[공차가 없으면 판정하지 않는다]
공차 미기재 치수(예: '46')는 도면의 일반공차(ISO 2768 등)를 따르는데, 그 등급은
표제란에 있고 우리는 아직 안 읽는다. 임의로 ±0.1을 가정하면 오판정이 나오므로,
등급을 모르면 '판정 불가'로 남긴다.
"""
import re
import unicodedata

# 지름/반지름/나사 기호. OCR이 ø를 0이나 O로 읽는 경우가 있어 넉넉히 잡는다.
DIA_CHARS = "øØ⌀φΦ∅"
GENERAL_TOLERANCE = {          # ISO 2768 (mm). 표제란에서 등급을 읽으면 쓸 수 있다.
    "f": [(3, 0.05), (6, 0.05), (30, 0.1), (120, 0.15), (400, 0.2), (1000, 0.3)],
    "m": [(3, 0.1), (6, 0.1), (30, 0.2), (120, 0.3), (400, 0.5), (1000, 0.8)],
    "c": [(3, 0.2), (6, 0.3), (30, 0.5), (120, 0.8), (400, 1.2), (1000, 2.0)],
    "v": [(3, None), (6, 0.5), (30, 1.0), (120, 1.5), (400, 2.5), (1000, 4.0)],
}


def _norm(s):
    """전각/유사문자 정리 + 소수점 쉼표 통일."""
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("＋", "+").replace("㎜", "mm")
    # 숫자 사이의 쉼표는 유럽식 소수점 (1,5 -> 1.5). 천단위 쉼표는 도면에 거의 없다.
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    return s


def parse_dimension(text):
    """치수 텍스트 -> dict.

    반환 키:
      kind      : 'linear' | 'diameter' | 'radius' | 'thread' | 'angle' | None
      nominal   : 공칭값(mm 또는 도). 못 읽으면 None
      upper/lower : 공차(mm). 못 읽으면 None
      fit       : 'H7' 같은 끼워맞춤 기호 (있을 때)
      judgeable : 측정값과 비교해 합/불 판정이 가능한가
      raw       : 원문
    """
    raw = text
    s = _norm(text)
    out = {"kind": None, "nominal": None, "upper": None, "lower": None,
           "fit": None, "judgeable": False, "raw": raw, "note": None}
    if not s:
        return out

    # 종류 판정
    if re.match(r"^M\s*\d", s, re.I):
        out["kind"] = "thread"
    elif any(c in s for c in DIA_CHARS):
        out["kind"] = "diameter"
    elif re.match(r"^R\s*\d", s, re.I):
        out["kind"] = "radius"
    elif "°" in s or "'" in s:
        out["kind"] = "angle"
    else:
        out["kind"] = "linear"

    # 각도는 도/분 표기가 섞여 별도 처리
    if out["kind"] == "angle":
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*°(?:\s*(\d+(?:\.\d+)?)\s*')?", s)
        if m:
            out["nominal"] = float(m.group(1)) + (float(m.group(2)) / 60 if m.group(2) else 0.0)
        t = re.search(r"±\s*(\d+(?:\.\d+)?)\s*°(?:\s*(\d+(?:\.\d+)?)\s*')?", s)
        if t:
            v = float(t.group(1)) + (float(t.group(2)) / 60 if t.group(2) else 0.0)
            out["upper"], out["lower"] = v, -v
        out["judgeable"] = out["nominal"] is not None and out["upper"] is not None
        return out

    # 공칭값 — 기호/접두를 걷어내고 첫 숫자
    body = s
    for c in DIA_CHARS:
        body = body.replace(c, " ")
    body = re.sub(r"^\s*[MR]\s*", " ", body, flags=re.I)
    m = re.search(r"(-?\d+(?:\.\d+)?)", body)
    if m:
        out["nominal"] = float(m.group(1))
    rest = body[m.end():] if m else ""

    # 끼워맞춤 등급 (H7, g6, JS9 ...) — 등급표가 있어야 수치가 나온다
    f = re.search(r"\b([A-Za-z]{1,2}\s?\d{1,2})\b", rest)
    if f and not re.match(r"^\s*typ", f.group(1), re.I):
        out["fit"] = f.group(1).replace(" ", "")
        out["note"] = "끼워맞춤 등급은 등급표가 필요해 수치 공차를 못 냈습니다"

    # ± 표기
    pm = re.search(r"±\s*(\d+(?:\.\d+)?)", rest)
    if pm:
        v = float(pm.group(1))
        out["upper"], out["lower"] = v, -v
    else:
        # 상/하한 분리 표기: '+0.01 -0.01', '+0.1 0', '0 -0.2'
        ups = re.findall(r"([+-]\s*\d+(?:\.\d+)?|(?<![\d.])0(?![\d.]))", rest)
        vals = [float(u.replace(" ", "")) for u in ups]
        if len(vals) >= 2:
            hi, lo = max(vals[0], vals[1]), min(vals[0], vals[1])
            out["upper"], out["lower"] = hi, lo

    out["judgeable"] = (out["nominal"] is not None and out["upper"] is not None
                        and out["lower"] is not None)
    if out["nominal"] is not None and not out["judgeable"] and out["note"] is None:
        out["note"] = "공차 미기재 — 표제란의 일반공차 등급을 알아야 판정 가능"
    return out


def general_tolerance(nominal_mm, grade="m"):
    """ISO 2768 일반공차. 표제란에서 등급을 읽었을 때만 쓸 것."""
    table = GENERAL_TOLERANCE.get(grade.lower())
    if table is None or nominal_mm is None:
        return None
    for upper_bound, tol in table:
        if abs(nominal_mm) <= upper_bound:
            return tol
    return table[-1][1]


def judge(parsed, measured_mm, uncertainty_mm=0.0):
    """측정값 판정. 불확실도를 고려해 '판정 보류'를 명시적으로 낸다.

    측정 불확실도가 공차보다 크면 합/불을 단정할 수 없다. 이때 pass/fail을
    찍어버리면 사용자는 그것을 신뢰해버린다 — 카메라 측정의 정확도(+-0.3~1mm)가
    도면 공차(+-0.01~0.1mm)보다 10~100배 크므로 이 경우가 오히려 흔하다.
    """
    res = {"verdict": "unknown", "deviation": None, "reason": None}
    if not parsed.get("judgeable") or measured_mm is None:
        res["reason"] = parsed.get("note") or "공칭값 또는 공차를 읽지 못했습니다"
        return res

    nominal = parsed["nominal"]
    lo = nominal + parsed["lower"]
    hi = nominal + parsed["upper"]
    res["deviation"] = measured_mm - nominal

    tol_width = hi - lo
    if uncertainty_mm > 0 and uncertainty_mm * 2 > tol_width:
        res["verdict"] = "inconclusive"
        res["reason"] = (f"측정 불확실도 ±{uncertainty_mm:.2f}mm가 공차 폭 "
                         f"{tol_width:.2f}mm보다 큽니다 — 카메라로는 판정 불가")
        return res

    if lo - uncertainty_mm <= measured_mm <= hi + uncertainty_mm:
        # 불확실도를 감안하면 경계에 걸치는 경우
        res["verdict"] = "pass" if lo <= measured_mm <= hi else "borderline"
        if res["verdict"] == "borderline":
            res["reason"] = "공차 경계에 불확실도 범위가 걸칩니다 — 재측정 권장"
    else:
        res["verdict"] = "fail"
    return res


def format_result(parsed, measured_mm, uncertainty_mm=0.0):
    """사람이 읽을 한 줄. 측정값은 반드시 불확실도와 함께 낸다."""
    j = judge(parsed, measured_mm, uncertainty_mm)
    label = {"pass": "합격", "fail": "불합격", "borderline": "경계",
             "inconclusive": "판정불가", "unknown": "판정불가"}[j["verdict"]]
    if measured_mm is None:
        return f"{parsed['raw']} → 미측정 ({label})"
    dev = f"{j['deviation']:+.2f}" if j["deviation"] is not None else "?"
    tail = f"  · {j['reason']}" if j["reason"] else ""
    return (f"{parsed['raw']} → {measured_mm:.2f} ±{uncertainty_mm:.2f}mm "
            f"({dev}) {label}{tail}")
