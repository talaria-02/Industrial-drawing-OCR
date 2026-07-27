# -*- coding: utf-8 -*-
"""검수 문서(review.json) 데이터 모델 + 편집 연산 + 재학습용 내보내기.

[구조를 평평하게 잡은 이유]
설계문서(docs/pipeline_architecture.md)의 스키마는 dimensions[].links[].review
처럼 깊게 중첩돼 있는데, 그건 최종 산출물 형태로는 맞지만 UI에서 편집/되돌리기
하기엔 불편하다. 그래서 texts / lines / arrows / links 를 각각 평평한 배열로
두고 id로 참조한다. 설계문서 형태가 필요하면 나중에 변환해서 내보낸다.

[source / verified 필드가 중요한 이유]
자동 생성인지 사람이 만들거나 고친 것인지 구분해야, 나중에 재학습할 때
"사람이 검수한 것만" 골라서 학습 데이터로 쓸 수 있다. 자동 결과를 그대로
학습에 넣으면 모델이 자기 실수를 다시 배우는 자기강화가 일어난다.
"""
import copy
import json
import os

# 나중에 늘리기 쉽게 상수로 둔다
CATEGORIES = ['치수', '지름', '각도', '거칠기', '공차', '나사', '메타데이터', '기타']

SCHEMA_VERSION = 1

# 되돌리기 보관 개수. 스냅샷 방식이라 문서 하나가 통째로 복사되지만,
# 우리 문서는 수십 KB~1MB 수준이라 이 정도는 부담이 없다.
MAX_UNDO = 60


def empty_doc(image_name, image_size):
    return {
        "schema_version": SCHEMA_VERSION,
        "image": image_name,
        "image_size": list(image_size),   # [width, height]
        "texts": [],
        "lines": [],
        "arrows": [],
        "links": [],
        # ⑦ 제품사진 비교용 예약 공간. 도면은 mm, 사진은 px이므로 환산(캘리브레이션)이
        # 반드시 필요해서 자리를 미리 잡아둔다. 지금은 UI에서 비활성.
        "measure": {"photo_path": None, "calibration": None, "results": []},
        "history": [],
    }


class ReviewDoc:
    def __init__(self, data):
        self.data = data
        self.dirty = False
        self._undo = []
        self._redo = []

    # ── 되돌리기 (스냅샷 방식) ──────────────────────────────
    # 편집 종류마다 역연산을 만드는 방식(커맨드 패턴)은 종류가 많아질수록 버그가
    # 생기기 쉽다. 문서를 통째로 복사해두는 편이 단순하고 확실하다.
    #
    # 주의: 드래그처럼 마우스 이동마다 값이 바뀌는 편집은 '시작할 때 한 번만'
    # push_undo()를 불러야 한다. 매 이동마다 부르면 스냅샷이 수십 개 쌓여서
    # Ctrl+Z를 여러 번 눌러야 드래그 하나가 취소된다.
    def push_undo(self):
        self._undo.append(copy.deepcopy(self.data))
        if len(self._undo) > MAX_UNDO:
            self._undo.pop(0)
        self._redo.clear()      # 새 편집이 생기면 다시실행 이력은 무효

    def undo(self):
        if not self._undo:
            return False
        self._redo.append(copy.deepcopy(self.data))
        self.data = self._undo.pop()
        self.dirty = True
        return True

    def redo(self):
        if not self._redo:
            return False
        self._undo.append(copy.deepcopy(self.data))
        self.data = self._redo.pop()
        self.dirty = True
        return True

    def can_undo(self):
        return bool(self._undo)

    def can_redo(self):
        return bool(self._redo)

    # ── 파일 I/O ────────────────────────────────────────────
    @staticmethod
    def load(path):
        with open(path, encoding='utf-8') as f:
            return ReviewDoc(json.load(f))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        self.dirty = False

    # ── 공통 헬퍼 ───────────────────────────────────────────
    def _next_id(self, key, prefix):
        """기존 id 중 최대 번호 + 1. 삭제 후 재사용을 피해서 history 참조가 깨지지 않게 한다."""
        mx = 0
        for o in self.data[key]:
            try:
                mx = max(mx, int(o["id"][len(prefix):]))
            except (ValueError, KeyError):
                pass
        return f'{prefix}{mx + 1}'

    def _log(self, action, **kw):
        self.data["history"].append({"action": action, **kw})
        self.dirty = True

    def find(self, key, obj_id):
        for o in self.data[key]:
            if o["id"] == obj_id:
                return o
        return None

    # ── 텍스트 ─────────────────────────────────────────────
    def add_text(self, poly, text="", category="치수", score=None, source="human"):
        self.push_undo()
        tid = self._next_id("texts", "t")
        self.data["texts"].append({
            "id": tid, "poly": [[float(x), float(y)] for x, y in poly],
            "text": text, "score": score, "category": category,
            "source": source, "verified": (source == "human"),
        })
        self._log("text_add", text_id=tid)
        return tid

    def update_text(self, tid, **kw):
        self.push_undo()
        t = self.find("texts", tid)
        if t is None:
            return
        for k, v in kw.items():
            t[k] = v
        t["verified"] = True          # 사람이 손댄 것은 검수 완료로 표시
        self._log("text_edit", text_id=tid, fields=list(kw))

    def delete_text(self, tid):
        self.push_undo()
        self.data["texts"] = [t for t in self.data["texts"] if t["id"] != tid]
        self.data["links"] = [l for l in self.data["links"] if l["text_id"] != tid]
        self._log("text_delete", text_id=tid)

    def text_center(self, t):
        xs = [p[0] for p in t["poly"]]
        ys = [p[1] for p in t["poly"]]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def text_bbox(self, t):
        xs = [p[0] for p in t["poly"]]
        ys = [p[1] for p in t["poly"]]
        return (min(xs), min(ys), max(xs), max(ys))

    # ── 선분 ───────────────────────────────────────────────
    def add_line(self, p1, p2, source="human"):
        self.push_undo()
        lid = self._next_id("lines", "l")
        self.data["lines"].append({
            "id": lid, "p1": [float(p1[0]), float(p1[1])],
            "p2": [float(p2[0]), float(p2[1])], "source": source,
        })
        self._log("line_add", line_id=lid)
        return lid

    def update_line(self, lid, p1=None, p2=None):
        self.push_undo()
        l = self.find("lines", lid)
        if l is None:
            return
        if p1 is not None:
            l["p1"] = [float(p1[0]), float(p1[1])]
        if p2 is not None:
            l["p2"] = [float(p2[0]), float(p2[1])]
        l["source"] = "human"
        self._log("line_edit", line_id=lid)

    def delete_line(self, lid):
        self.push_undo()
        self.data["lines"] = [l for l in self.data["lines"] if l["id"] != lid]
        self.data["arrows"] = [a for a in self.data["arrows"] if a["line_id"] != lid]
        for link in self.data["links"]:
            if lid in link["line_ids"]:
                link["line_ids"] = [x for x in link["line_ids"] if x != lid]
                link["source"] = "human"
        self._log("line_delete", line_id=lid)

    # ── 화살촉 ─────────────────────────────────────────────
    def get_arrow(self, lid, end):
        for a in self.data["arrows"]:
            if a["line_id"] == lid and a["end"] == end:
                return a
        return None

    def toggle_arrow(self, lid, end):
        """화살촉 상태를 3단계로 순환시킨다:  미검사(회색) → 있음(초록) → 없음(빨강) → 미검사

        2단계(있음↔없음)로만 돌리면 한번 만든 뒤 '미검사'로 되돌릴 수 없어서,
        실수로 만든 것을 취소할 방법이 없다. 그래서 한 바퀴 돌면 항목 자체를
        삭제해 원래의 미검사 상태로 복귀시킨다.

        '미검사'와 '없음'은 뜻이 다르다 — 전자는 판단하지 않았다는 것이고,
        후자는 확인했고 화살촉이 없다는 것이다. 이 구분이 있어야 검수 진행률을
        셀 수 있고, 나중에 재학습 데이터로 쓸 때도 의미가 유지된다."""
        self.push_undo()
        a = self.get_arrow(lid, end)
        if a is None:                       # 미검사 → 있음
            aid = self._next_id("arrows", "a")
            self.data["arrows"].append({
                "id": aid, "line_id": lid, "end": end,
                "present": True, "score": None, "source": "human",
            })
            state = 'present'
        elif a["present"]:                  # 있음 → 없음
            a["present"] = False
            a["source"] = "human"
            state = 'absent'
        else:                               # 없음 → 미검사(항목 제거)
            self.data["arrows"] = [x for x in self.data["arrows"]
                                    if not (x["line_id"] == lid and x["end"] == end)]
            state = 'unchecked'
        self._log("arrow_toggle", line_id=lid, end=end, state=state)
        return state

    # ── 연결(매칭) ─────────────────────────────────────────
    def get_link(self, tid):
        for l in self.data["links"]:
            if l["text_id"] == tid:
                return l
        return None

    def toggle_link_line(self, tid, lid):
        """이미 연결된 선이면 해제, 아니면 추가.
        line_ids가 배열인 이유: 하나의 치수가 여러 선분을 가리키는 경우가 실제로 있다."""
        self.push_undo()
        link = self.get_link(tid)
        if link is None:
            self.data["links"].append({
                "text_id": tid, "line_ids": [lid],
                "source": "human", "confidence": None, "verified": True,
            })
        else:
            before = list(link["line_ids"])
            if lid in link["line_ids"]:
                link["line_ids"] = [x for x in link["line_ids"] if x != lid]
            else:
                link["line_ids"].append(lid)
            link["source"] = "human"
            link["verified"] = True
            self._log("link_edit", text_id=tid, before=before, after=list(link["line_ids"]))
            return
        self._log("link_add", text_id=tid, line_id=lid)

    def clear_link(self, tid):
        self.push_undo()
        self.data["links"] = [l for l in self.data["links"] if l["text_id"] != tid]
        self._log("link_clear", text_id=tid)

    def mark_link_verified(self, tid):
        link = self.get_link(tid)
        if link is not None:
            self.push_undo()
            link["verified"] = True
            self.dirty = True

    # ── 통계(상태바 표시용) ─────────────────────────────────
    def stats(self):
        n_text = len(self.data["texts"])
        n_line = len(self.data["lines"])
        links = self.data["links"]
        n_link = len(links)
        n_link_ok = sum(1 for l in links if l.get("verified"))
        n_arrow_present = sum(1 for a in self.data["arrows"] if a.get("present"))
        n_text_ok = sum(1 for t in self.data["texts"] if t.get("verified"))
        return {
            "texts": n_text, "texts_verified": n_text_ok,
            "lines": n_line,
            "links": n_link, "links_verified": n_link_ok,
            "arrows_present": n_arrow_present, "arrows": len(self.data["arrows"]),
        }

    # ── 재학습용 내보내기 ───────────────────────────────────
    def export_label_txt(self, path, image_rel_path):
        """det/rec 재학습용 PaddleOCR 라벨 포맷.
        사람이 검수한 텍스트만 내보낸다 — 자동 결과를 그대로 학습에 넣으면
        모델이 자기 오류를 다시 배우는 자기강화가 일어난다."""
        items = []
        for t in self.data["texts"]:
            if not t.get("verified"):
                continue
            if not t.get("text"):
                continue
            items.append({
                "transcription": t["text"],
                "points": [[int(round(x)), int(round(y))] for x, y in t["poly"]],
                "difficult": False,
            })
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'{image_rel_path}\t{json.dumps(items, ensure_ascii=False)}\n')
        return len(items)

    def export_matching_gt(self, path):
        """매칭 정답(ground truth). 지금까지 정확도를 측정할 수단이 없었던 문제를
        이 파일이 해결한다 — 검수된 연결만 정답으로 취급."""
        gt = []
        for link in self.data["links"]:
            if not link.get("verified"):
                continue
            t = self.find("texts", link["text_id"])
            if t is None:
                continue
            segs = []
            for lid in link["line_ids"]:
                l = self.find("lines", lid)
                if l is not None:
                    segs.append({"id": lid, "p1": l["p1"], "p2": l["p2"]})
            gt.append({
                "text_id": link["text_id"], "text": t["text"],
                "category": t.get("category"), "bbox": self.text_bbox(t),
                "lines": segs,
            })
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"image": self.data["image"], "image_size": self.data["image_size"],
                        "links": gt}, f, indent=2, ensure_ascii=False)
        return len(gt)
