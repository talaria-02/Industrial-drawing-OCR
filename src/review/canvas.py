# -*- coding: utf-8 -*-
"""도면 표시 + 확대/이동 + 객체 선택/편집 캔버스.

[좌표계가 두 개인 점 주의]
  이미지 좌표: 원본 픽셀 기준. review.json에 저장되는 값은 항상 이쪽.
  화면 좌표: 위젯 픽셀 기준. 마우스 이벤트가 주는 값은 이쪽.
  변환:  화면 = 이미지 * scale + offset
        이미지 = (화면 - offset) / scale
편집 결과를 저장할 때는 반드시 이미지 좌표로 되돌려야 한다(안 그러면 확대 배율에
따라 좌표가 달라지는 버그가 생긴다).

[히트테스트 임계값을 scale로 나누는 이유]
"화면에서 10px 이내 클릭"을 이미지 좌표로 환산하면 10/scale 이다. 확대해서
보고 있으면 이미지 기준으로는 더 정밀하게 찍는 셈이므로 이렇게 해야 체감이 일정하다.
"""
import numpy as np
from PyQt5.QtCore import Qt, QPoint, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import (QPainter, QPen, QColor, QBrush, QImage, QPixmap,
                          QFont, QPolygonF)
from PyQt5.QtWidgets import QWidget

MODE_TEXT = 'text'
MODE_LINE = 'line'
MODE_ARC = 'arc'
MODE_ARROW = 'arrow'
MODE_MATCH = 'match'
MODE_CATEGORY = 'category'   # 미사용(호환용)
MODE_NODE = 'node'
MODE_MEASURE = 'measure'

HIT_PX = 10          # 화면 기준 클릭 허용 오차
HANDLE_PX = 6        # 끝점 핸들 반지름(화면 기준)
# 클릭 허용 오차의 이미지좌표 상한(px). 축소해서 볼 때 HIT_PX/scale이 수십~수백 px로
# 커지면서 멀리 있는 객체가 잡히는 것을 막는다.
MAX_TOL_IMG_PX = 14.0

CATEGORY_COLORS = {
    '치수': (0, 160, 0), '지름': (0, 130, 200), '각도': (200, 120, 0),
    '거칠기': (160, 0, 160), '공차': (0, 150, 150), '나사': (120, 90, 40),
    '메타데이터': (130, 130, 130), '기타': (90, 90, 90),
}


def bgr_to_qpixmap(bgr):
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w, _ = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())


def _pt_seg_dist(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    L2 = vx * vx + vy * vy
    if L2 < 1e-9:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - x1) * vx + (py - y1) * vy) / L2))
    cx, cy = x1 + t * vx, y1 + t * vy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def _point_in_poly(px, py, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > py) != (y2 > py):
            xin = x1 + (py - y1) / (y2 - y1) * (x2 - x1)
            if px < xin:
                inside = not inside
    return inside


class Canvas(QWidget):
    selectionChanged = pyqtSignal()
    docChanged = pyqtSignal()
    statusMessage = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.doc = None
        self.pixmap = None
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.mode = MODE_MATCH
        self.sel_kind = None      # 'text' | 'line'
        self.sel_id = None
        self.pending_text_id = None   # 매칭 모드에서 숫자를 먼저 클릭한 상태
        self._panning = False
        self._pan_start = None
        self._drag = None          # 편집 중인 대상 정보
        self._draw_start = None    # 선 그리기 첫 클릭 지점(이미지 좌표)
        self._space_held = False   # Space = 임시 이동 모드(그래픽 도구 표준)
        self.show_unlinked_lines = True

    # ── 문서 설정 ──────────────────────────────────────────
    def set_document(self, doc, bgr_image):
        self.doc = doc
        self.pixmap = bgr_to_qpixmap(bgr_image)
        self.sel_kind = self.sel_id = self.pending_text_id = None
        self.fit_to_window()
        self.update()

    def fit_to_window(self):
        if self.pixmap is None:
            return
        pw, ph = self.pixmap.width(), self.pixmap.height()
        if pw == 0 or ph == 0:
            return
        s = min(self.width() / pw, self.height() / ph) * 0.98
        self.scale = max(s, 0.01)
        self.offset = QPointF((self.width() - pw * self.scale) / 2,
                               (self.height() - ph * self.scale) / 2)

    # ── 좌표 변환 ──────────────────────────────────────────
    def to_screen(self, x, y):
        return QPointF(x * self.scale + self.offset.x(), y * self.scale + self.offset.y())

    def to_image(self, pos):
        return ((pos.x() - self.offset.x()) / self.scale,
                (pos.y() - self.offset.y()) / self.scale)

    def visibility(self):
        """모드별로 화면에 그릴(그리고 클릭으로 선택할) 요소를 정한다.
        paintEvent와 히트테스트가 같은 규칙을 쓰게 하려고 메서드로 뺐다 —
        화면에 없는 것이 클릭되면 사용자가 혼란스럽다."""
        m = self.mode
        show = {
            # 텍스트/카테고리 작업 중엔 선분·연결선이 방해만 된다
            MODE_TEXT:     dict(texts=True, all_lines=False, linked_lines=False,
                                links=False, arrows=False),
            MODE_CATEGORY: dict(texts=True, all_lines=False, linked_lines=False,
                                links=False, arrows=False),
            # 노드 단계 — 분할 결과를 눈으로 확인하고 고치는 자리. 선분이 전부
            # 보이고 클릭돼야 하며, 텍스트는 접점을 가리므로 끈다.
            MODE_NODE:     dict(texts=False, all_lines=True, linked_lines=True,
                                links=False, arrows=False),
            # 선분 편집은 선분이 주인공. 텍스트는 위치 파악용으로만 옅게 필요
            MODE_LINE:     dict(texts=True, all_lines=True, linked_lines=True,
                                links=False, arrows=False),
            # 원/호 편집은 선분을 다 끈다. 원 둘레와 선분이 겹쳐 보이면 어느 쪽을
            # 잡는 건지 알 수 없고, 히트테스트도 서로 먹는다.
            MODE_ARC:      dict(texts=False, all_lines=False, linked_lines=False,
                                links=False, arrows=False),
            # 화살촉 모드도 모든 선분을 보여야 한다. 연결된 선만 보이게 했더니
            # '연결 안 된 선에는 화살촉을 만들 수 없다'는 문제가 생겼다
            # (게다가 안 보이는 선은 클릭도 안 되니 화면이동으로 처리돼 버렸음).
            MODE_ARROW:    dict(texts=False, all_lines=True, linked_lines=True,
                                links=False, arrows=True),
            # 매칭은 숫자·선분·연결선이 다 필요
            MODE_MATCH:    dict(texts=True, all_lines=True, linked_lines=True,
                                links=True, arrows=True),
            # 검증 단계에서는 텍스트 박스를 끈다. 이 모드에서 볼 것은 '측정점이
            # 외곽선 위에 제대로 놓였는가'인데, 글자 박스가 깔리면 그게 가려진다.
            # 치수 선택은 오른쪽 목록에서 한다.
            # 사람이 체인을 짚어야 하므로 모든 선분이 보이고 클릭돼야 한다.
            # 텍스트 박스만 끈다(측정점을 가리므로).
            MODE_MEASURE:  dict(texts=False, all_lines=True, linked_lines=True,
                                links=False, arrows=False),
        }.get(m, dict(texts=True, all_lines=True, linked_lines=True,
                      links=True, arrows=True))
        # 사용자가 체크박스로 미연결 선을 끈 경우는 그 뜻을 우선한다
        if not self.show_unlinked_lines:
            show = dict(show, all_lines=False)
        return show

    # ── 렌더링 ────────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(245, 245, 245))
        if self.pixmap is None:
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), Qt.AlignCenter, '도면을 열어주세요 (열기 버튼)')
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 1.0)
        p.drawPixmap(int(self.offset.x()), int(self.offset.y()),
                     int(self.pixmap.width() * self.scale),
                     int(self.pixmap.height() * self.scale), self.pixmap)
        if self.doc is None:
            return

        # ── 모드별 표시 범위 ──────────────────────────────
        # 모든 요소를 항상 그리면 도면이 뒤덮여 정작 지금 다루는 것이 안 보인다.
        # 그래서 모드마다 필요한 것만 남긴다.
        #   texts        : 텍스트 박스
        #   all_lines    : 연결 안 된 선분까지 전부
        #   linked_lines : 숫자와 연결된 선분만
        #   links        : 숫자-선분 연결선(보라 화살표)
        #   arrows       : 선 끝점의 화살촉 표시(초록/빨강 원)
        show = self.visibility()

        linked = {lid for l in self.doc.data['links'] for lid in l['line_ids']}
        linked_arcs = {cid for l in self.doc.data['links']
                       for cid in l.get('arc_ids', [])}

        # 원/호 — 선분보다 먼저 그려서 아래에 깔리게 한다(선분 선택 표시가 가려지지 않게).
        # Qt의 drawArc는 1/16도 단위이고 3시 방향에서 '반시계'로 잰다. 우리 각도는
        # 이미지 좌표계(y가 아래로) 기준이라 화면상 '시계' 방향이므로 부호를 뒤집는다.
        if show.get('arcs', True):
            for c in self.doc.data.get('arcs', []):
                sel = (self.sel_kind == 'arc' and self.sel_id == c['id'])
                if sel:
                    acol, awd = QColor(255, 0, 255), 3
                elif c['id'] in linked_arcs:
                    acol, awd = QColor(30, 90, 220), 3      # 선분과 같은 '연결됨' 색
                else:
                    acol, awd = QColor(0, 150, 80), 2
                p.setPen(QPen(acol, awd))
                cx, cy = c['center']
                r = c['r']
                tl = self.to_screen(cx - r, cy - r)
                br = self.to_screen(cx + r, cy + r)
                rect = QRectF(tl.x(), tl.y(), br.x() - tl.x(), br.y() - tl.y())
                if c.get('closed'):
                    p.drawEllipse(rect)
                else:
                    p.drawArc(rect, int(-c['start_deg'] * 16), int(-c['span_deg'] * 16))
                # 원 모드이거나 선택 중일 때만 조작 핸들을 띄운다(평소엔 지저분해진다)
                if self.mode == MODE_ARC or sel:
                    for pt, col in self._arc_handles(c):
                        s = self.to_screen(*pt)
                        p.setBrush(QBrush(col))
                        p.setPen(QPen(QColor(40, 40, 40), 1))
                        p.drawEllipse(s, HANDLE_PX // 2, HANDLE_PX // 2)
                    p.setBrush(Qt.NoBrush)

        # 측정 모드에서는 '지금 고른 치수'만 옅은 테두리로 표시한다. 전부 그리면
        # 측정점이 가려지고, 아무것도 안 그리면 무엇을 재는 중인지 알 수 없다.
        if self.mode == MODE_MEASURE and self.sel_kind == 'text' and self.sel_id:
            t = self.doc.find('texts', self.sel_id)
            if t:
                poly = QPolygonF([self.to_screen(x, y) for x, y in t['poly']])
                p.setPen(QPen(QColor(230, 0, 130), 2, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawPolygon(poly)

        # 외곽선 그래프 — 노드(칼자국)와 추론된 치수. 측정 모드에서만.
        if self.mode == MODE_MEASURE:
            G = self.doc.data.get('graph', {})
            for e in G.get('edges', []):
                nd = G['nodes']
                if e['start'] < len(nd) and e['end'] < len(nd):
                    p.setPen(QPen(QColor(0, 170, 120), 2))
                    p.drawLine(self.to_screen(*nd[e['start']]),
                               self.to_screen(*nd[e['end']]))
            for k, (nx, ny) in enumerate(G.get('nodes', [])):
                sp = self.to_screen(nx, ny)
                p.setPen(QPen(QColor(255, 255, 255), 1.5))
                p.setBrush(QBrush(QColor(0, 150, 100)))
                p.drawEllipse(sp, HANDLE_PX / 2 + 1, HANDLE_PX / 2 + 1)
            p.setBrush(Qt.NoBrush)
            # 도면에 안 적힌 값은 다른 색으로 — OCR로 읽은 것과 섞이면 안 된다
            f = QFont(); f.setPointSize(9); f.setBold(True); p.setFont(f)
            for d in G.get('dimensions', []):
                if d.get('is_explicit') or d.get('value') is None:
                    continue
                a, b = d['ref_nodes']
                nd = G['nodes']
                if a >= len(nd) or b >= len(nd):
                    continue
                sa, sb = self.to_screen(*nd[a]), self.to_screen(*nd[b])
                mid = QPointF((sa.x() + sb.x()) / 2, (sa.y() + sb.y()) / 2 - 8)
                txt = f"({d['value']:g})"
                p.setPen(QPen(QColor(255, 255, 255), 4)); p.drawText(mid, txt)
                p.setPen(QPen(QColor(200, 90, 0), 1)); p.drawText(mid, txt)

        # 노드 단계: 끝점이 겹치는 접점을 찍는다. 분할·스냅이 실제로 이어졌는지
        # 여기서 보인다(숫자는 그 점에서 만나는 선분 수).
        if self.mode == MODE_NODE:
            fnt = QFont()
            fnt.setPointSize(8)
            p.setFont(fnt)
            for (jx, jy), cnt in self.doc.junctions():
                sp = self.to_screen(jx, jy)
                col = QColor(0, 150, 100) if cnt >= 3 else QColor(70, 120, 220)
                p.setPen(QPen(QColor(255, 255, 255), 1.5))
                p.setBrush(QBrush(col))
                p.drawEllipse(sp, HANDLE_PX / 2 + 1, HANDLE_PX / 2 + 1)
                if cnt >= 3:
                    p.setPen(QPen(col.darker(150), 1))
                    p.drawText(QPointF(sp.x() + 6, sp.y() - 4), str(cnt))
            p.setBrush(Qt.NoBrush)

        # 사람이 지정한 체인 — 순서 번호를 선 위에 찍는다. 마지막 번호가 외곽선.
        if self.mode == MODE_MEASURE:
            fnt = QFont()
            fnt.setPointSize(10)
            fnt.setBold(True)
            p.setFont(fnt)
            for tid, chain in self.doc.measure_chains().items():
                hot = (self.sel_kind == 'text' and self.sel_id == tid)
                for k, lid in enumerate(chain):
                    l = self.doc.find('lines', lid)
                    if l is None:
                        continue
                    last = (k == len(chain) - 1)
                    if hot:
                        col = QColor(0, 160, 90) if last else QColor(255, 140, 0)
                    else:
                        col = QColor(150, 200, 170) if last else QColor(255, 205, 150)
                    wdt = 6 if (hot and last) else (4 if hot else 3)
                    p.setPen(QPen(col, wdt))
                    a, b = self.to_screen(*l['p1']), self.to_screen(*l['p2'])
                    p.drawLine(a, b)
                    mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2 - 5)
                    txt = str(k + 1) + ('(외곽)' if last and hot else '')
                    p.setPen(QPen(QColor(255, 255, 255), 4))
                    p.drawText(mid, txt)
                    p.setPen(QPen(col.darker(140), 1))
                    p.drawText(mid, txt)

        # 측정점 — 치수가 가리키는 실제 두 모서리. 측정 모드에서만 띄운다
        # (다른 모드에서는 화면만 어지럽힌다).
        if self.mode == MODE_MEASURE:
            # 역추적이 밟고 간 선분을 굵게 덧그린다. 결과 두 점만 찍어두면
            # 그 연결이 말이 되는지 사람이 검수할 방법이 없다.
            for link in self.doc.data['links']:
                m = link.get('measure') or {}
                hot = (self.sel_kind == 'text' and self.sel_id == link['text_id'])
                for side in m.get('path', []):
                    for lid in side:
                        l = self.doc.find('lines', lid)
                        if l is None:
                            continue
                        p.setPen(QPen(QColor(255, 150, 0) if hot else QColor(255, 200, 130),
                                      5 if hot else 3))
                        p.drawLine(self.to_screen(*l['p1']), self.to_screen(*l['p2']))
            for tid, pts in self.doc.measure_points().items():
                hot = (self.sel_kind == 'text' and self.sel_id == tid)
                col = QColor(230, 0, 130) if hot else QColor(150, 90, 190)
                a, b = self.to_screen(*pts[0]), self.to_screen(*pts[1])
                p.setPen(QPen(QColor(255, 255, 255, 200), 5 if hot else 3))
                p.drawLine(a, b)
                p.setPen(QPen(col, 2.5 if hot else 1.5, Qt.DashLine))
                p.drawLine(a, b)
                for k, s_ in enumerate((a, b)):
                    act = self._drag and self._drag[0] == 'mpoint'                         and self._drag[1] == tid and self._drag[2] == k
                    p.setPen(QPen(QColor(255, 255, 255), 1.5))
                    p.setBrush(QBrush(QColor(255, 0, 255) if act else col))
                    r = HANDLE_PX / 2 + (1 if hot else 0)
                    p.drawEllipse(s_, r, r)
                p.setBrush(Qt.NoBrush)

        # 선분
        for l in self.doc.data['lines']:
            is_linked = l['id'] in linked
            if is_linked and not show['linked_lines']:
                continue
            if not is_linked and not show['all_lines']:
                continue
            sel = (self.sel_kind == 'line' and self.sel_id == l['id'])
            if sel:
                col, wdt = QColor(255, 0, 255), 3
            elif is_linked:
                col, wdt = QColor(30, 90, 220), (4 if self.mode == MODE_MEASURE else 3)
            else:
                col, wdt = QColor(190, 190, 190), 1
            p.setPen(QPen(col, wdt))
            a = self.to_screen(*l['p1'])
            b = self.to_screen(*l['p2'])
            p.drawLine(a, b)
            # 화살촉 모드에서는 모든 선의 끝점 상태를 봐야 하므로 조건을 풀어준다.
            # (다른 모드에서는 화면이 지저분해지지 않게 연결된/선택된 선만)
            if show['arrows'] and (is_linked or sel or self.mode == MODE_ARROW):
                for end, pt in (('start', a), ('end', b)):
                    ar = self.doc.get_arrow(l['id'], end)
                    if ar is None:
                        c = QColor(150, 150, 150)
                    else:
                        c = QColor(0, 190, 0) if ar['present'] else QColor(230, 0, 0)
                    p.setBrush(QBrush(c))
                    p.setPen(QPen(c, 1))
                    p.drawEllipse(pt, HANDLE_PX, HANDLE_PX)

        # 연결선(텍스트 중심 → 선분 중점).
        # 예전엔 얇은 연파랑 점선 1px이라 도면 위에서 거의 안 보였다. 그래서
        #  (1) 굵기를 올리고 색을 진하게, (2) 흰색 외곽선을 먼저 깔아 도면 선과 대비를 주고,
        #  (3) 선분 쪽 끝에 화살표 머리를 붙여 방향을 알 수 있게,
        #  (4) 지금 선택/대기 중인 숫자의 연결은 더 굵고 진하게 강조한다.
        for link in (self.doc.data['links'] if show['links'] else []):
            t = self.doc.find('texts', link['text_id'])
            if t is None:
                continue
            tid = link['text_id']
            hot = (tid == self.pending_text_id) or \
                  (self.sel_kind == 'text' and self.sel_id == tid)
            tc = self.to_screen(*self.doc.text_center(t))
            # 선분은 중점으로, 원은 '텍스트에서 가장 가까운 둘레 위 점'으로 잇는다.
            # 원 중심으로 이으면 큰 원에서 연결선이 도면을 가로질러 버린다.
            targets = []
            for lid in link['line_ids']:
                l = self.doc.find('lines', lid)
                if l is not None:
                    targets.append(((l['p1'][0] + l['p2'][0]) / 2,
                                    (l['p1'][1] + l['p2'][1]) / 2))
            tcx, tcy = self.doc.text_center(t)
            for cid in link.get('arc_ids', []):
                c = self.doc.find('arcs', cid)
                if c is None:
                    continue
                cx, cy = c['center']
                d = np.hypot(tcx - cx, tcy - cy)
                if d < 1e-6:
                    targets.append((cx + c['r'], cy))
                else:
                    targets.append((cx + (tcx - cx) / d * c['r'],
                                    cy + (tcy - cy) / d * c['r']))
            for tgt in targets:
                mid = self.to_screen(*tgt)
                col = QColor(255, 60, 0) if hot else QColor(90, 40, 220)
                wdt = 4 if hot else 2
                # 흰색 밑선을 먼저 깔면 검은 도면선 위에서도 연결선이 뚜렷하게 보인다
                p.setPen(QPen(QColor(255, 255, 255, 200), wdt + 3))
                p.drawLine(tc, mid)
                p.setPen(QPen(col, wdt, Qt.SolidLine if hot else Qt.DashLine))
                p.drawLine(tc, mid)
                self._draw_arrow_head(p, tc, mid, col, 9 if hot else 7)

        # 텍스트
        f = QFont()
        f.setPointSize(9)
        p.setFont(f)
        for t in (self.doc.data['texts'] if show['texts'] else []):
            cat = t.get('category', '기타')
            r, g, b = CATEGORY_COLORS.get(cat, (90, 90, 90))
            sel = (self.sel_kind == 'text' and self.sel_id == t['id'])
            pending = (self.pending_text_id == t['id'])
            if pending:
                pen = QPen(QColor(255, 0, 0), 3)
            elif sel:
                pen = QPen(QColor(255, 0, 255), 3)
            else:
                pen = QPen(QColor(r, g, b), 2 if t.get('verified') else 1,
                           Qt.SolidLine if t.get('verified') else Qt.DotLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            pts = [self.to_screen(x, y) for x, y in t['poly']]
            for i in range(len(pts)):
                p.drawLine(pts[i], pts[(i + 1) % len(pts)])
            if self.scale > 0.35:
                p.setPen(QPen(QColor(r, g, b)))
                p.drawText(pts[0] + QPointF(0, -3), t.get('text', ''))

        # 그리기 중 미리보기 — 텍스트 모드는 사각형, 선분 모드는 직선.
        # (예전엔 둘 다 직선으로 그려서, 텍스트 박스를 만드는 중인데 선이 보여 헷갈렸음)
        if self._draw_start is not None:
            cur = self.mapFromGlobal(self.cursor().pos())
            a = self.to_screen(*self._draw_start)
            b = QPointF(cur)
            if self.mode == MODE_TEXT:
                rect = QRectF(a, b).normalized()
                p.setPen(QPen(QColor(255, 0, 255), 2, Qt.DashLine))
                p.setBrush(QBrush(QColor(255, 0, 255, 40)))    # 옅은 채움으로 영역을 명확히
                p.drawRect(rect)
                p.setBrush(Qt.NoBrush)
                # 시작 모서리를 점으로 표시(어디서 시작했는지 헷갈리지 않게)
                p.setBrush(QBrush(QColor(255, 0, 255)))
                p.drawEllipse(a, 4, 4)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(180, 0, 180)))
                p.drawText(rect.topLeft() + QPointF(2, -4),
                           f'{abs(rect.width()/self.scale):.0f}×{abs(rect.height()/self.scale):.0f}')
            else:
                p.setPen(QPen(QColor(255, 0, 255), 2, Qt.DashLine))
                p.drawLine(a, b)
                p.setBrush(QBrush(QColor(255, 0, 255)))
                p.drawEllipse(a, 4, 4)
                p.setBrush(Qt.NoBrush)

    def _draw_arrow_head(self, p, frm, to, color, size):
        """연결선의 '선분 쪽' 끝에 삼각형 머리를 그린다 — 숫자에서 선으로 향하는
        방향을 한눈에 알 수 있게 하려는 것."""
        import math
        dx, dy = to.x() - frm.x(), to.y() - frm.y()
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return
        ux, uy = dx / d, dy / d
        # 머리가 선분 중점을 살짝 덮도록 끝에서 조금 물러난 지점을 밑변 중심으로 삼는다
        bx, by = to.x() - ux * size, to.y() - uy * size
        nx, ny = -uy, ux
        half = size * 0.5
        poly = QPolygonF([QPointF(to.x(), to.y()),
                          QPointF(bx + nx * half, by + ny * half),
                          QPointF(bx - nx * half, by - ny * half)])
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(color))
        p.drawPolygon(poly)
        p.setBrush(Qt.NoBrush)

    # ── 히트테스트 ────────────────────────────────────────
    def _tol(self, base_px=HIT_PX):
        """클릭 허용 오차를 이미지 좌표로 환산. 축소해서 볼 때 오차가 무한정
        커지지 않도록 상한을 둔다.

        예전엔 base/scale 만 썼는데, 전체보기(scale≈0.3)에서 33px, 더 축소하면
        수백 px까지 벌어져서 '한참 떨어진 박스가 클릭되어 새 박스를 그릴 수 없는'
        문제가 있었다. MAX_TOL_IMG_PX로 상한을 건다."""
        return min(base_px / self.scale, MAX_TOL_IMG_PX)

    def hit_text(self, ix, iy):
        # 측정 모드는 예외 — 글자 박스를 그리지 않지만 클릭으로는 고를 수 있어야 한다.
        # 박스를 다 그리면 정작 확인해야 할 측정점이 가려지는데, 클릭까지 막으면
        # 도면에서 치수를 고를 방법이 사라진다(오른쪽 목록만 남아 불편했다).
        if not self.visibility()['texts'] and self.mode != MODE_MEASURE:
            return None          # 화면에 안 보이는 건 클릭으로도 잡히지 않게
        # 박스 안을 찍었으면 그게 우선(겹쳐 있으면 더 작은 박스를 고른다 —
        # 큰 박스에 갇혀서 안쪽 작은 박스를 못 고르는 일을 막으려는 것)
        inside = []
        for t in self.doc.data['texts']:
            if _point_in_poly(ix, iy, t['poly']):
                x0, y0, x1, y1 = self.doc.text_bbox(t)
                inside.append(((x1 - x0) * (y1 - y0), t['id']))
        if inside:
            return min(inside)[1]

        # 박스 밖이면 '가장 가까운' 것을 고른다. 예전엔 먼저 발견된 것을 그냥
        # 채택해서(best = best or ...), 목록 앞쪽의 먼 박스가 잡히곤 했다.
        tol = self._tol()
        best, bd = None, tol
        for t in self.doc.data['texts']:
            x0, y0, x1, y1 = self.doc.text_bbox(t)
            dx = max(x0 - ix, 0, ix - x1)
            dy = max(y0 - iy, 0, iy - y1)
            d = (dx * dx + dy * dy) ** 0.5
            if d <= bd:
                best, bd = t['id'], d
        return best

    def hit_line(self, ix, iy):
        show = self.visibility()
        if not (show['all_lines'] or show['linked_lines']):
            return None
        linked = {lid for l in self.doc.data['links'] for lid in l['line_ids']}
        tol = self._tol()
        best, bd = None, tol
        for l in self.doc.data['lines']:
            is_linked = l['id'] in linked
            if is_linked and not show['linked_lines']:
                continue
            if not is_linked and not show['all_lines']:
                continue
            d = _pt_seg_dist(ix, iy, l['p1'][0], l['p1'][1], l['p2'][0], l['p2'][1])
            if d <= bd:
                best, bd = l['id'], d
        return best

    def hit_line_endpoint(self, ix, iy):
        tol = self._tol(HANDLE_PX + 3)
        for l in self.doc.data['lines']:
            for end, pt in (('p1', l['p1']), ('p2', l['p2'])):
                if ((ix - pt[0]) ** 2 + (iy - pt[1]) ** 2) ** 0.5 <= tol:
                    return l['id'], end
        return None, None

    # ── 원/호 조작 ────────────────────────────────────────
    # 핸들 세 종류: 중심(이동) / 시작각 / 끝각. 반지름은 원 둘레를 잡아끈다.
    # 완전 원일 때는 각도 핸들이 의미가 없어 중심만 띄운다.
    def _arc_handles(self, c):
        cx, cy = c['center']
        r = c['r']
        out = [((cx, cy), QColor(60, 120, 255))]
        if not c.get('closed'):
            for ang, col in ((c['start_deg'], QColor(0, 200, 90)),
                             (c['start_deg'] + c['span_deg'], QColor(255, 140, 0))):
                t = np.radians(ang)
                out.append(((cx + r * np.cos(t), cy + r * np.sin(t)), col))
        return out

    def hit_arc_handle(self, ix, iy):
        """반환 ('center'|'start'|'end', arc_id) 또는 (None, None)."""
        tol = self._tol(HANDLE_PX + 3)
        names = ('center', 'start', 'end')
        for c in self.doc.data.get('arcs', []):
            for (pt, _), nm in zip(self._arc_handles(c), names):
                if np.hypot(ix - pt[0], iy - pt[1]) <= tol:
                    return nm, c['id']
        return None, None

    def hit_arc(self, ix, iy):
        """원 둘레를 클릭했는지. 부분 호는 각도 구간 안일 때만 잡는다."""
        tol = self._tol()
        best, bd = None, tol
        for c in self.doc.data.get('arcs', []):
            cx, cy = c['center']
            d = abs(np.hypot(ix - cx, iy - cy) - c['r'])
            if d > bd:
                continue
            if not c.get('closed'):
                a = (np.degrees(np.arctan2(iy - cy, ix - cx)) - c['start_deg']) % 360.0
                if a > c['span_deg']:
                    continue
            best, bd = c['id'], d
        return best

    def seed_measure_chain(self, tid):
        """매칭에서 이미 연결된 선분을 체인 1번으로 채워둔다.

        치수선은 매칭 단계에서 이미 확정돼 있다. 측정 단계에서 그것부터 다시
        클릭하게 하면 같은 일을 두 번 시키는 셈이고, 사용자는 '연결한 선이
        체인에 안 들어간다'고 느낀다. 비어 있을 때만 채우고, 사람이 손댄
        체인은 건드리지 않는다.
        """
        link = self.doc.get_link(tid)
        if link is None or link.get('measure_chain'):
            return
        seed = [x for x in link.get('line_ids', []) if self.doc.find('lines', x)]
        if not seed:
            self.statusMessage.emit(
                '이 치수는 매칭된 선분이 없습니다 — 치수선부터 순서대로 클릭하세요')
            return
        self.doc.set_measure_chain(tid, seed)
        self.docChanged.emit()
        self.statusMessage.emit(
            f'매칭된 치수선 {len(seed)}개를 1번으로 넣었습니다 — '
            '이어서 치수보조선, 마지막에 외곽선을 클릭하세요')

    def hit_measure_point(self, ix, iy):
        """반환 (text_id, 0|1) 또는 (None, None). 선택된 치수의 것만 잡는다."""
        tol = self._tol(HANDLE_PX + 3)
        for tid, pts in self.doc.measure_points().items():
            if tid != self.sel_id:
                continue
            for k, q in enumerate(pts):
                if np.hypot(ix - q[0], iy - q[1]) <= tol:
                    return tid, k
        return None, None

    def _press_measure(self, ix, iy):
        tid, k = self.hit_measure_point(ix, iy)
        if tid is not None:
            self._drag = ('mpoint', tid, k)
            self.sel_kind, self.sel_id = 'text', tid
            self.selectionChanged.emit()
            return
        t = self.hit_text(ix, iy)
        if t:
            self.sel_kind, self.sel_id = 'text', t
            self.selectionChanged.emit()
            self.seed_measure_chain(t)
            return
        # 선분 클릭 = 체인에 추가. 자동 역추적이 자주 실패하므로 사람이 직접
        # 경로를 짚는다. 순서가 곧 의미이고, 마지막이 외곽선이다.
        if self.sel_kind != 'text' or not self.sel_id:
            return
        lid = self.hit_line(ix, iy)
        if lid is None:
            return
        link = self.doc.get_link(self.sel_id)
        chain = list((link or {}).get('measure_chain') or [])
        if lid in chain:
            chain.remove(lid)          # 같은 선 재클릭 = 취소
        else:
            chain.append(lid)
        pts = self.doc.set_measure_chain(self.sel_id, chain)
        self.docChanged.emit()
        self.statusMessage.emit(
            f'{len(chain)}개 지정' + (' -> 측정점 확정' if pts else ' (2개 이상 필요)'))

    def _press_arc(self, ix, iy):
        nm, cid = self.hit_arc_handle(ix, iy)
        if nm is not None:
            self._drag = ('arc_' + nm, cid, None)
            self.sel_kind, self.sel_id = 'arc', cid
            self.selectionChanged.emit()
            return
        cid = self.hit_arc(ix, iy)
        if cid and self._draw_start is None:
            # 둘레를 잡으면 반지름 조절
            self._drag = ('arc_radius', cid, None)
            self.sel_kind, self.sel_id = 'arc', cid
            self.selectionChanged.emit()
            return
        if self._draw_start is None:
            self._draw_start = (ix, iy)
            self.statusMessage.emit('원의 둘레가 지날 점을 클릭하세요 (우클릭 취소)')
        else:
            r = float(np.hypot(ix - self._draw_start[0], iy - self._draw_start[1]))
            cid = self.doc.add_arc(self._draw_start, max(2.0, r))
            self._draw_start = None
            self.sel_kind, self.sel_id = 'arc', cid
            self.docChanged.emit()
            self.selectionChanged.emit()

    def hit_text_corner(self, ix, iy):
        tol = self._tol(HANDLE_PX + 3)
        for t in self.doc.data['texts']:
            for i, (x, y) in enumerate(t['poly']):
                if ((ix - x) ** 2 + (iy - y) ** 2) ** 0.5 <= tol:
                    return t['id'], i
        return None, None

    def _start_pan(self, e):
        self._panning = True
        self._pan_start = (e.pos(), QPointF(self.offset))
        self.setCursor(Qt.ClosedHandCursor)

    # ── 마우스 ────────────────────────────────────────────
    def wheelEvent(self, e):
        if self.pixmap is None:
            return
        ix, iy = self.to_image(e.pos())
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale = max(0.02, min(40.0, self.scale * factor))
        # 커서 아래 이미지 지점이 제자리에 있도록 offset 보정
        self.offset = QPointF(e.pos().x() - ix * self.scale,
                               e.pos().y() - iy * self.scale)
        self.update()

    def mousePressEvent(self, e):
        if self.pixmap is None or self.doc is None:
            return
        ix, iy = self.to_image(e.pos())

        # 화면 이동: 가운데 버튼 / Shift+좌클릭 / Space누른상태+좌클릭
        if e.button() == Qt.MiddleButton or (
                e.button() == Qt.LeftButton and
                (self._space_held or e.modifiers() & Qt.ShiftModifier)):
            self._start_pan(e)
            return

        # 편의: 빈 공간(객체 없는 곳)에서 좌클릭 드래그도 화면 이동으로 처리한다.
        # 도면을 훑어보는 게 가장 빈번한 동작인데 매번 Shift를 누르는 건 번거롭다.
        # 단, "빈 곳 클릭"이 새 객체 그리기 시작인 모드(텍스트/선분)와 이미 그리는
        # 중일 때는 제외해야 그리기 동작을 막지 않는다.
        if (e.button() == Qt.LeftButton and self._draw_start is None
                and self.mode not in (MODE_TEXT, MODE_LINE, MODE_ARC, MODE_ARROW,
                                      MODE_NODE)
                and not (self.mode == MODE_MEASURE and self.hit_measure_point(ix, iy)[0])
                and self.hit_text(ix, iy) is None and self.hit_line(ix, iy) is None
                and self.hit_arc(ix, iy) is None
                and self.hit_arc_handle(ix, iy)[0] is None):
            self._start_pan(e)
            return

        if e.button() == Qt.RightButton and self.mode == MODE_MEASURE \
                and self.sel_kind == 'text' and self.sel_id:
            link = self.doc.get_link(self.sel_id)
            chain = list((link or {}).get('measure_chain') or [])
            if chain:
                chain.pop()
                self.doc.set_measure_chain(self.sel_id, chain)
                self.docChanged.emit()
                self.statusMessage.emit('마지막 취소 — %d개 남음' % len(chain))
                self.update()
                return
        if e.button() == Qt.RightButton:
            # 우클릭: 선 그리기 취소 / 매칭 대기 해제
            self._draw_start = None
            self.pending_text_id = None
            self.update()
            return

        if e.button() != Qt.LeftButton:
            return

        if self.mode == MODE_MATCH:
            self._press_match(ix, iy)
        elif self.mode == MODE_TEXT:
            self._press_text(ix, iy)
        elif self.mode == MODE_LINE:
            self._press_line(ix, iy)
        elif self.mode == MODE_ARC:
            self._press_arc(ix, iy)
        elif self.mode == MODE_NODE:
            self._press_line(ix, iy)      # 선분 모드와 같은 편집을 허용한다
        elif self.mode == MODE_MEASURE:
            self._press_measure(ix, iy)
        elif self.mode == MODE_ARROW:
            self._press_arrow(ix, iy)
        elif self.mode == MODE_CATEGORY:
            tid = self.hit_text(ix, iy)
            if tid:
                self.sel_kind, self.sel_id = 'text', tid
                self.selectionChanged.emit()
        self.update()

    def _press_match(self, ix, iy):
        """숫자 클릭 → 선 클릭(2클릭). 여러 선이면 계속 클릭, Enter로 종료."""
        if self.pending_text_id is None:
            tid = self.hit_text(ix, iy)
            if tid:
                self.pending_text_id = tid
                self.sel_kind, self.sel_id = 'text', tid
                self.selectionChanged.emit()
                self.statusMessage.emit('연결할 선을 클릭하세요 (여러 개 가능, Enter로 종료)')
            return
        # 원을 선분보다 먼저 본다. ø/R 치수는 원을 가리키는데, 지름 치수선이
        # 원을 가로질러 그려져 있으면 선분이 먼저 잡혀서 원을 못 고른다.
        cid = self.hit_arc(ix, iy)
        lid = None if cid else self.hit_line(ix, iy)
        if cid or lid:
            if cid:
                self.doc.toggle_link_arc(self.pending_text_id, cid)
            else:
                self.doc.toggle_link_line(self.pending_text_id, lid)
            self.docChanged.emit()
            link = self.doc.get_link(self.pending_text_id)
            nl = len(link['line_ids']) if link else 0
            nc = len(link.get('arc_ids', [])) if link else 0
            self.statusMessage.emit(
                f'선 {nl}개 · 원 {nc}개 연결됨 — 더 클릭하거나 Enter로 종료')
        else:
            # 빈 곳 클릭 = 다른 숫자 선택으로 전환
            tid = self.hit_text(ix, iy)
            self.pending_text_id = tid
            if tid:
                self.sel_kind, self.sel_id = 'text', tid
                self.selectionChanged.emit()

    def _press_text(self, ix, iy):
        tid, ci = self.hit_text_corner(ix, iy)
        if tid is not None:
            self._drag = ('text_corner', tid, ci)
            self.sel_kind, self.sel_id = 'text', tid
            self.selectionChanged.emit()
            return
        tid = self.hit_text(ix, iy)
        if tid:
            self.sel_kind, self.sel_id = 'text', tid
            self._drag = ('text_move', tid, (ix, iy))
            self.selectionChanged.emit()
        else:
            # 빈 곳 클릭 → 새 텍스트 박스 그리기 시작
            if self._draw_start is None:
                self._draw_start = (ix, iy)
                self.statusMessage.emit('반대쪽 모서리를 클릭하세요 (우클릭 취소)')
            else:
                x0, y0 = self._draw_start
                poly = [[min(x0, ix), min(y0, iy)], [max(x0, ix), min(y0, iy)],
                        [max(x0, ix), max(y0, iy)], [min(x0, ix), max(y0, iy)]]
                tid = self.doc.add_text(poly, text='', category='치수')
                self._draw_start = None
                self.sel_kind, self.sel_id = 'text', tid
                self.docChanged.emit()
                self.selectionChanged.emit()
                self.statusMessage.emit('텍스트 내용을 오른쪽 패널에서 입력하세요')

    def _press_line(self, ix, iy):
        lid, end = self.hit_line_endpoint(ix, iy)
        if lid is not None:
            self._drag = ('line_end', lid, end)
            self.sel_kind, self.sel_id = 'line', lid
            self.selectionChanged.emit()
            return
        lid = self.hit_line(ix, iy)
        if lid and self._draw_start is None:
            self.sel_kind, self.sel_id = 'line', lid
            self.selectionChanged.emit()
            return
        if self._draw_start is None:
            self._draw_start = (ix, iy)
            self.statusMessage.emit('선의 끝점을 클릭하세요 (우클릭 취소)')
        else:
            lid = self.doc.add_line(self._draw_start, (ix, iy))
            self._draw_start = None
            self.sel_kind, self.sel_id = 'line', lid
            self.docChanged.emit()
            self.selectionChanged.emit()

    def _press_arrow(self, ix, iy):
        lid, end = self.hit_line_endpoint(ix, iy)
        if lid is None:
            lid = self.hit_line(ix, iy)
            if lid is None:
                return
            # 선 몸통을 클릭했으면 더 가까운 끝점을 고른다
            l = self.doc.find('lines', lid)
            d1 = (ix - l['p1'][0]) ** 2 + (iy - l['p1'][1]) ** 2
            d2 = (ix - l['p2'][0]) ** 2 + (iy - l['p2'][1]) ** 2
            end = 'p1' if d1 <= d2 else 'p2'
        side = 'start' if end == 'p1' else 'end'
        state = self.doc.toggle_arrow(lid, side)
        self.sel_kind, self.sel_id = 'line', lid
        self.docChanged.emit()
        self.selectionChanged.emit()
        label = {'present': '화살촉 있음(초록)', 'absent': '화살촉 없음(빨강)',
                 'unchecked': '미검사로 되돌림(회색)'}.get(state, '')
        self.statusMessage.emit(f'{lid} {side} → {label}')

    def mouseMoveEvent(self, e):
        if self._panning and self._pan_start is not None:
            start_pos, start_off = self._pan_start
            d = e.pos() - start_pos
            self.offset = QPointF(start_off.x() + d.x(), start_off.y() + d.y())
            self.update()
            return
        if self._drag is not None and self.doc is not None:
            ix, iy = self.to_image(e.pos())
            kind = self._drag[0]
            if kind == 'text_corner':
                _, tid, ci = self._drag
                t = self.doc.find('texts', tid)
                if t:
                    t['poly'][ci] = [ix, iy]
                    t['source'] = 'human'
                    t['verified'] = True
            elif kind == 'text_move':
                _, tid, (lx, ly) = self._drag
                t = self.doc.find('texts', tid)
                if t:
                    dx, dy = ix - lx, iy - ly
                    t['poly'] = [[x + dx, y + dy] for x, y in t['poly']]
                    t['source'] = 'human'
                    t['verified'] = True
                    self._drag = ('text_move', tid, (ix, iy))
            elif kind == 'line_end':
                _, lid, end = self._drag
                l = self.doc.find('lines', lid)
                if l:
                    l[end] = [ix, iy]
                    l['source'] = 'human'
            elif kind == 'mpoint':
                _, tid, k = self._drag
                link = self.doc.get_link(tid)
                if link and link.get('measure'):
                    link['measure']['points'][k] = [ix, iy]
                    link['measure']['source'] = 'human'
                    link['measure']['quality'] = 'human'
            elif kind.startswith('arc_'):
                _, cid, _ = self._drag
                c = self.doc.find('arcs', cid)
                if c:
                    cx, cy = c['center']
                    if kind == 'arc_center':
                        c['center'] = [ix, iy]
                    elif kind == 'arc_radius':
                        c['r'] = max(1.0, float(np.hypot(ix - cx, iy - cy)))
                    else:
                        ang = float(np.degrees(np.arctan2(iy - cy, ix - cx))) % 360.0
                        if kind == 'arc_start':
                            # 끝각을 붙잡아둔 채 시작각만 옮긴다
                            end = c['start_deg'] + c['span_deg']
                            c['start_deg'] = ang
                            c['span_deg'] = max(1.0, (end - ang) % 360.0)
                        else:
                            c['span_deg'] = max(1.0, (ang - c['start_deg']) % 360.0)
                        c['closed'] = c['span_deg'] >= 300.0
                    c['source'], c['verified'] = 'human', True
            self.update()
            return
        if self._draw_start is not None:
            self.update()

    def mouseReleaseEvent(self, e):
        if self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.OpenHandCursor if self._space_held else Qt.ArrowCursor)
            return
        if self._drag is not None:
            self._drag = None
            self.doc.dirty = True
            self.docChanged.emit()

    # ── 키보드 ────────────────────────────────────────────
    def keyPressEvent(self, e):
        k = e.key()
        # Space = 누르고 있는 동안 이동 모드 (문서 없어도 동작해야 하므로 위에서 처리)
        if k == Qt.Key_Space and not e.isAutoRepeat():
            self._space_held = True
            self.setCursor(Qt.OpenHandCursor)
            return
        if self.doc is None:
            return
        if k in (Qt.Key_Return, Qt.Key_Enter):
            if self.pending_text_id:
                self.doc.mark_link_verified(self.pending_text_id)
                self.pending_text_id = None
                self.docChanged.emit()
                self.statusMessage.emit('연결 확정')
            self.update()
        elif k == Qt.Key_Escape:
            self.pending_text_id = None
            self._draw_start = None
            self.update()
        elif k == Qt.Key_C and self.sel_kind == 'arc' and self.sel_id:
            closed = self.doc.toggle_arc_closed(self.sel_id)
            self.docChanged.emit()
            self.statusMessage.emit(
                f"{self.sel_id} → {'완전 원' if closed else '부분 호(180도)'}")
        elif k == Qt.Key_Delete:
            # 매칭 모드에서는 '연결만' 지운다. 이 모드에서 Del로 숫자 자체가
            # 지워지면 사고에 가깝다(연결을 지우려던 것일 뿐인데 텍스트가 사라짐).
            # 숫자/선분 자체를 지우는 건 각각 텍스트/선분 모드에서 한다.
            if self.mode == MODE_MATCH:
                tid = self.pending_text_id or (self.sel_id if self.sel_kind == 'text' else None)
                if tid:
                    self.doc.clear_link(tid)
                    self.pending_text_id = None
                    self.docChanged.emit()
                    self.selectionChanged.emit()
                    self.statusMessage.emit('연결을 모두 해제했습니다 (Ctrl+Z로 복구 가능)')
                else:
                    self.statusMessage.emit('먼저 숫자를 클릭하세요')
            elif self.mode == MODE_MEASURE and self.sel_kind == 'text' and self.sel_id:
                self.doc.clear_measure_chain(self.sel_id)
                self.docChanged.emit()
                self.statusMessage.emit('측정 체인 전체 취소')
            elif self.sel_kind == 'text' and self.sel_id:
                self.doc.delete_text(self.sel_id)
                self.sel_kind = self.sel_id = None
                self.docChanged.emit()
                self.selectionChanged.emit()
            elif self.sel_kind == 'line' and self.sel_id:
                self.doc.delete_line(self.sel_id)
                self.sel_kind = self.sel_id = None
                self.docChanged.emit()
                self.selectionChanged.emit()
            elif self.sel_kind == 'arc' and self.sel_id:
                self.doc.delete_arc(self.sel_id)
                self.sel_kind = self.sel_id = None
                self.docChanged.emit()
                self.selectionChanged.emit()
            self.update()
        elif k == Qt.Key_F:
            self.fit_to_window()
            self.update()
        elif k == Qt.Key_H:
            self.show_unlinked_lines = not self.show_unlinked_lines
            self.update()
        elif k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            # 방향키로도 조금씩 이동 (정밀 조정용)
            step = 60
            dx = (-step if k == Qt.Key_Right else step if k == Qt.Key_Left else 0)
            dy = (-step if k == Qt.Key_Down else step if k == Qt.Key_Up else 0)
            self.offset = QPointF(self.offset.x() + dx, self.offset.y() + dy)
            self.update()

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space_held = False
            if not self._panning:
                self.setCursor(Qt.ArrowCursor)

    def focus_on(self, kind, obj_id):
        """오른쪽 목록에서 항목을 고르면 그 위치로 화면을 옮긴다."""
        if self.doc is None:
            return
        if kind == 'text':
            t = self.doc.find('texts', obj_id)
            if t is None:
                return
            cx, cy = self.doc.text_center(t)
        else:
            l = self.doc.find('lines', obj_id)
            if l is None:
                return
            cx, cy = (l['p1'][0] + l['p2'][0]) / 2, (l['p1'][1] + l['p2'][1]) / 2
        self.sel_kind, self.sel_id = kind, obj_id
        if self.scale < 1.0:
            self.scale = 1.5
        self.offset = QPointF(self.width() / 2 - cx * self.scale,
                               self.height() / 2 - cy * self.scale)
        self.update()
