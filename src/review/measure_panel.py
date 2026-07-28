# -*- coding: utf-8 -*-
"""측정 모드 — 도면 치수와 실제 제품 사진을 나란히 놓고 비교한다.

[화면 구성]
왼쪽은 기존 도면 캔버스, 오른쪽에 이 패널이 붙는다. 도면에서 치수를 고르고
사진에서 그 부위를 두 번 클릭하면 실측값과 판정이 기록된다.

[왜 도면 쪽에서 '연결된' 치수만 쓰는가]
치수 텍스트만으로는 무엇을 재야 할지 모른다. 사람이 검수 단계에서 그 치수를
선분이나 원에 연결해두었을 때에야 "이 치수는 저 외곽선을 가리킨다"가 확정되고,
그제서야 사진에서 같은 부위를 찾아 잴 수 있다. 연결이 없는 치수는 목록에
회색으로 남겨 사용자가 먼저 연결하도록 유도한다.

[판정을 함부로 내리지 않는다]
카메라 측정 정확도(±0.3~1mm)가 도면 공차(±0.01~0.1mm)보다 10~100배 크다.
불확실도가 공차보다 크면 합/불 대신 '판정불가'를 낸다 — compare.judge가
그 판단을 하고, 여기서는 결과를 그대로 표시만 한다.
"""
import os
import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import (QPainter, QPen, QBrush, QColor, QPixmap, QImage,
                          QFont)
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
                              QFileDialog, QMessageBox, QHeaderView, QGroupBox,
                              QSplitter, QSizePolicy)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from measure import calibration as cal, measure as ms, compare as cp  # noqa: E402

# 도면 캔버스와 같은 값 — 같은 창에서 번갈아 쓰므로 클릭 감도가 같아야 한다
HIT_PX = 10
HANDLE_PX = 6
MAX_TOL_IMG_PX = 14.0

VERDICT_COLOR = {
    "pass": QColor(215, 245, 215), "fail": QColor(255, 215, 215),
    "borderline": QColor(255, 240, 200), "inconclusive": QColor(230, 230, 230),
    "unknown": QColor(240, 240, 240),
}
VERDICT_TEXT = {"pass": "합격", "fail": "불합격", "borderline": "경계",
                "inconclusive": "판정불가", "unknown": "판정불가"}


def bgr_to_pixmap(bgr):
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w, _ = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())


class PhotoCanvas(QWidget):
    """보정된 제품 사진 위에서 두 점을 찍어 재는 캔버스.

    조작 규약을 도면 캔버스(canvas.Canvas)와 똑같이 맞춘다 — 같은 창에서 번갈아
    쓰는 도구인데 마우스 습관이 다르면 계속 헛클릭하게 된다.
      빈 곳 드래그 / Space+드래그 / 가운데버튼 = 이동
      휠 = 확대,  F = 전체보기,  우클릭/Esc = 취소,  Del = 측정 지우기
      끝점 드래그 = 측정점 조정 (놓는 순간 다시 스냅되고 값이 갱신된다)

    처음에는 2클릭으로 확정하고 끝이었는데, 한 픽셀 어긋나면 처음부터 다시 찍어야
    해서 불편했다. 그래서 도면 쪽 선분 편집과 같은 방식 — 만든 뒤 끝점을 끌어서
    다듬는다 — 으로 바꿨다. 표시도 가늘게 줄였다(선 1.6px, 핸들 반지름 3px).

    도면 캔버스를 그대로 재사용하지 않는 이유는 그쪽이 review 문서의
    texts/lines/arcs 구조에 묶여 있기 때문이다. 여기 필요한 건 사진 한 장과
    측정선 하나뿐이다.
    """
    measured = pyqtSignal(float, object, object)   # mm, p1, p2 (보정이미지 좌표)
    activeChanged = pyqtSignal(str)                # 사진에서 다른 측정선을 잡았을 때
    measureDeleted = pyqtSignal(str)
    statusMessage = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumHeight(120)
        self.pixmap = None
        self.rect_img = None       # 보정된 BGR (엣지 스냅용 원본 화소)
        self.px_per_mm = 8.0
        self.scale, self.offset = 1.0, QPointF(0, 0)
        # 치수 하나당 측정선 하나를 들고 있는다. {text_id: (p1, p2, mm)}
        # 단일 측정만 쥐고 있던 초기 버전은 다음 치수를 재는 순간 앞의 것이
        # 화면에서 사라져, 여러 치수를 훑는 검수 흐름에 맞지 않았다.
        self.items = {}
        self.active = None         # 지금 편집 대상 text_id
        self._draw_start = None    # 새로 그리는 중인 첫 점
        self._hover = None
        self._drag_end = None      # 'p1' | 'p2'
        self._pan = None
        self._space = False

    # ── 이미지 설정 / 좌표 변환 ───────────────────────────
    def set_photo(self, rect_bgr, px_per_mm):
        self.rect_img = rect_bgr
        self.px_per_mm = px_per_mm
        self.pixmap = bgr_to_pixmap(rect_bgr)
        self.items = {}
        self._draw_start = None
        self.fit()
        self.update()

    def fit(self):
        if self.pixmap is None:
            return
        s = min(self.width() / self.pixmap.width(), self.height() / self.pixmap.height())
        self.scale = max(s * 0.98, 0.01)
        self.offset = QPointF((self.width() - self.pixmap.width() * self.scale) / 2,
                               (self.height() - self.pixmap.height() * self.scale) / 2)

    def to_img(self, pos):
        return ((pos.x() - self.offset.x()) / self.scale,
                (pos.y() - self.offset.y()) / self.scale)

    def to_scr(self, x, y):
        return QPointF(x * self.scale + self.offset.x(), y * self.scale + self.offset.y())

    def _tol(self, base_px=HIT_PX):
        """클릭 허용 오차를 이미지 좌표로. 축소했을 때 무한정 커지지 않게 상한."""
        return min(base_px / max(self.scale, 1e-6), MAX_TOL_IMG_PX)

    def set_items(self, items, active=None):
        """{text_id: (p1, p2, mm)} 전체 교체. 문서의 기록과 화면을 맞출 때 쓴다."""
        self.items = dict(items)
        self.active = active
        self.update()

    def set_active(self, tid):
        self.active = tid
        self._draw_start = None
        self.update()

    def hit_endpoint(self, ix, iy):
        """반환 (text_id, 'p1'|'p2') 또는 (None, None). 활성 치수를 먼저 본다."""
        tol = self._tol(HANDLE_PX + 3)
        order = ([self.active] if self.active in self.items else []) +                 [k for k in self.items if k != self.active]
        for tid in order:
            p1, p2, _ = self.items[tid]
            for name, q in (("p1", p1), ("p2", p2)):
                if np.hypot(ix - q[0], iy - q[1]) <= tol:
                    return tid, name
        return None, None

    # ── 렌더링 ────────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(238, 238, 238))
        if self.pixmap is None:
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "제품 사진을 열어주세요 (ArUco 보드가 함께 찍힌 사진)")
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 1.0)
        p.drawPixmap(int(self.offset.x()), int(self.offset.y()),
                     int(self.pixmap.width() * self.scale),
                     int(self.pixmap.height() * self.scale), self.pixmap)

        # 비활성 측정은 옅게 깔고 활성만 진하게 — 지금 무엇을 만지는지가 보여야 한다
        for tid, (q1, q2, mm) in self.items.items():
            act = (tid == self.active)
            col = QColor(20, 70, 230) if act else QColor(120, 140, 175)
            self._draw_line(p, q1, q2, col, True, thin=not act)
            if act:
                for name, q in (("p1", q1), ("p2", q2)):
                    self._draw_handle(p, q, self._drag_end == (tid, name))
            if mm is not None:
                self._draw_label(p, "%.2f mm" % mm, q1, q2, col, bold=act)
        if self._draw_start is not None and self._hover is not None:
            self._draw_line(p, self._draw_start, self._hover, QColor(255, 130, 0), False)

    def _draw_line(self, p, a, b, col, solid, thin=False):
        sa, sb = self.to_scr(*a), self.to_scr(*b)
        # 흰 밑선을 먼저 깔아야 사진 위 어떤 색에서도 보인다(도면 연결선과 같은 처리)
        p.setPen(QPen(QColor(255, 255, 255, 190), 2.5 if thin else 3))
        p.drawLine(sa, sb)
        p.setPen(QPen(col, 1.0 if thin else 1.6,
                      Qt.SolidLine if solid else Qt.DashLine))
        p.drawLine(sa, sb)

    def _draw_handle(self, p, q, active):
        s = self.to_scr(*q)
        r = HANDLE_PX / 2 + (1 if active else 0)
        p.setPen(QPen(QColor(255, 255, 255), 1.5))
        p.setBrush(QBrush(QColor(255, 0, 255) if active else QColor(20, 70, 230)))
        p.drawEllipse(s, r, r)
        p.setBrush(Qt.NoBrush)

    def _draw_label(self, p, text, a, b, col, bold=True):
        sa, sb = self.to_scr(*a), self.to_scr(*b)
        mid = QPointF((sa.x() + sb.x()) / 2 + 8, (sa.y() + sb.y()) / 2 - 6)
        f = QFont()
        f.setPointSize(9 if bold else 8)
        f.setBold(bold)
        p.setFont(f)
        p.setPen(QPen(QColor(255, 255, 255), 3))
        p.drawText(mid, text)
        p.setPen(QPen(col, 1))
        p.drawText(mid, text)

    # ── 측정 확정 ─────────────────────────────────────────
    def _commit(self, p1, p2, snap=True):
        if self.active is None:
            self.statusMessage.emit('먼저 도면에서 비교할 치수를 클릭하세요')
            return
        res = ms.measure_two_points(self.rect_img, p1, p2, self.px_per_mm, snap=snap)
        self.items[self.active] = (tuple(res["p1"]), tuple(res["p2"]), res["mm"])
        self.measured.emit(res["mm"], res["p1"], res["p2"])
        if snap:
            self.statusMessage.emit(
                "%.2f mm  (스냅 %+.1f/%+.1fpx · 끝점을 끌어 조정)"
                % (res["mm"], res.get("shift1_px", 0), res.get("shift2_px", 0)))
        self.update()

    # ── 마우스 ────────────────────────────────────────────
    def mousePressEvent(self, e):
        if self.pixmap is None:
            return
        ix, iy = self.to_img(e.pos())

        if e.button() == Qt.MiddleButton or (
                e.button() == Qt.LeftButton and
                (self._space or e.modifiers() & Qt.ShiftModifier)):
            self._start_pan(e)
            return
        if e.button() == Qt.RightButton:
            self._draw_start = None
            self.statusMessage.emit("취소")
            self.update()
            return
        if e.button() != Qt.LeftButton:
            return

        tid, end = self.hit_endpoint(ix, iy)
        if tid is not None:
            self._drag_end = (tid, end)
            if tid != self.active:
                self.active = tid
                self.activeChanged.emit(tid)
            self.update()
            return
        # 도면 캔버스와 같은 편의: 활성 치수에 이미 측정이 있고 빈 곳을 눌렀으면
        # 화면 이동. (다시 재려면 Del로 지우고 찍는다 — 실수로 날아가지 않게)
        if self._draw_start is None and self.active in self.items:
            self._start_pan(e)
            return
        if self._draw_start is None:
            self._draw_start = (ix, iy)
            self.statusMessage.emit("반대쪽 끝을 클릭하세요 (우클릭 취소)")
        else:
            start, self._draw_start = self._draw_start, None
            self._commit(start, (ix, iy))
        self.update()

    def _start_pan(self, e):
        self._pan = (e.pos(), QPointF(self.offset))
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            st, off = self._pan
            d = e.pos() - st
            self.offset = QPointF(off.x() + d.x(), off.y() + d.y())
            self.update()
            return
        ix, iy = self.to_img(e.pos())
        if self._drag_end is not None:
            tid, name = self._drag_end
            q1, q2, _ = self.items[tid]
            if name == "p1":
                q1 = (ix, iy)
            else:
                q2 = (ix, iy)
            mm = float(np.hypot(q2[0] - q1[0], q2[1] - q1[1])) / self.px_per_mm
            self.items[tid] = (q1, q2, mm)
            self.update()
            return
        if self._draw_start is not None:
            self._hover = (ix, iy)
            self.update()
            return
        self.setCursor(Qt.SizeAllCursor if self.hit_endpoint(ix, iy)
                       else (Qt.OpenHandCursor if self._space else Qt.CrossCursor))

    def mouseReleaseEvent(self, e):
        if self._pan is not None:
            self._pan = None
            self.setCursor(Qt.OpenHandCursor if self._space else Qt.CrossCursor)
            return
        if self._drag_end is not None:
            tid, _ = self._drag_end
            self._drag_end = None
            # 놓는 순간 다시 스냅해서 값을 확정한다. 끄는 동안 매번 스냅하면
            # 커서가 엣지에 끌려다녀 오히려 조준이 어렵다.
            if tid in self.items:
                prev, self.active = self.active, tid
                q1, q2, _ = self.items[tid]
                self._commit(q1, q2)
                self.active = prev if prev is not None else tid

    def wheelEvent(self, e):
        if self.pixmap is None:
            return
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        bx, by = self.to_img(e.pos())
        self.scale = float(np.clip(self.scale * f, 0.02, 40))
        self.offset += e.pos() - self.to_scr(bx, by)
        self.update()

    # ── 키보드 (도면 캔버스와 같은 키) ────────────────────
    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self.setCursor(Qt.OpenHandCursor)
        elif k == Qt.Key_F:
            self.fit()
            self.update()
        elif k == Qt.Key_Escape:
            self._draw_start = None
            self.update()
        elif k == Qt.Key_Delete:
            self._draw_start = None
            if self.active in self.items:
                del self.items[self.active]
                self.measureDeleted.emit(self.active)
                self.statusMessage.emit("이 치수의 측정을 지웠습니다")
            self.update()

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self.setCursor(Qt.CrossCursor)


class MeasurePanel(QWidget):
    """사진 불러오기 + 보정 + 치수별 비교 결과 표."""
    docChanged = pyqtSignal()
    statusMessage = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc = None
        self.calib = None          # rectify_board 결과
        self.active_text = None    # 지금 비교 중인 도면 치수 id

        self.canvas = PhotoCanvas()
        self.canvas.measured.connect(self.on_measured)
        self.canvas.statusMessage.connect(self.statusMessage)
        self.canvas.activeChanged.connect(self._on_canvas_active)
        self.canvas.measureDeleted.connect(self._on_canvas_deleted)

        # ── 보드 설정 + 사진 열기 ──
        bar = QHBoxLayout()
        self.btn_open = QPushButton('제품 사진 열기')
        self.btn_open.clicked.connect(self.open_photo)
        bar.addWidget(self.btn_open)
        bar.addWidget(QLabel('마커'))
        self.sp_marker = QDoubleSpinBox(); self.sp_marker.setRange(5, 500)
        self.sp_marker.setValue(50.0); self.sp_marker.setSuffix(' mm')
        self.sp_marker.setToolTip('인쇄물을 자로 잰 값. 3% 틀리면 모든 측정이 3% 틀립니다')
        bar.addWidget(self.sp_marker)
        bar.addWidget(QLabel('중심간'))
        self.sp_pitch = QDoubleSpinBox(); self.sp_pitch.setRange(10, 1000)
        self.sp_pitch.setValue(150.0); self.sp_pitch.setSuffix(' mm')
        bar.addWidget(self.sp_pitch)
        self.btn_trace = QPushButton('측정점 자동 추출')
        self.btn_trace.setToolTip(
            '연결된 치수마다 치수보조선을 따라가 실제 외곽선 위의 두 점을 찾습니다.\n'
            '도면 캔버스에 보라 점선으로 표시되며 끌어서 고칠 수 있습니다.')
        self.btn_trace.clicked.connect(self.auto_trace)
        bar.addWidget(self.btn_trace)
        bar.addStretch(1)
        self.lbl_calib = QLabel('사진 없음')
        self.lbl_calib.setStyleSheet('color:#555;')
        bar.addWidget(self.lbl_calib)

        # ── 지금 비교 중인 치수 ──
        self.lbl_target = QLabel('도면에서 치수를 클릭하세요')
        self.lbl_target.setStyleSheet(
            'background:#eef4ff; border:1px solid #aac; padding:6px; font-weight:bold;')

        # ── 결과 표 ──
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ['치수', '공칭', '공차', '실측', '편차', '판정'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setMinimumHeight(80)

        btns = QHBoxLayout()
        self.btn_del = QPushButton('선택 행 삭제')
        self.btn_del.clicked.connect(self.delete_selected)
        btns.addWidget(self.btn_del)
        btns.addStretch(1)
        self.lbl_summary = QLabel('')
        btns.addWidget(self.lbl_summary)

        # 사진과 결과표의 비율을 사용자가 정하게 한다. 고정 비율로 두면 도면마다
        # (치수 3개짜리 vs 40개짜리) 어느 한쪽이 늘 답답하다.
        g = QGroupBox('비교 결과')
        gl = QVBoxLayout(g)
        gl.setContentsMargins(4, 4, 4, 4)
        gl.addWidget(self.table, 1)
        gl.addLayout(btns)

        top_w = QWidget()
        tl = QVBoxLayout(top_w)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.canvas, 1)
        tl.addWidget(self.lbl_target)

        self.vsplit = QSplitter(Qt.Vertical)
        self.vsplit.addWidget(top_w)
        self.vsplit.addWidget(g)
        self.vsplit.setStretchFactor(0, 3)
        self.vsplit.setStretchFactor(1, 2)
        self.vsplit.setChildrenCollapsible(False)
        self.vsplit.setHandleWidth(6)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addLayout(bar)
        lay.addWidget(self.vsplit, 1)
        # 패널 자체도 바깥 스플리터로 자유롭게 줄였다 늘렸다 할 수 있어야 한다
        self.setMinimumWidth(280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ── 문서 연동 ─────────────────────────────────────────
    def set_document(self, doc):
        self.doc = doc
        self.calib = None
        self.active_text = None
        self.lbl_calib.setText('사진 없음')
        self.refresh_table()
        self._sync_canvas()

    def _sync_canvas(self):
        """문서의 기록을 사진 캔버스에 반영. 여러 치수의 측정선이 동시에 보인다."""
        items = {}
        for r in self.doc.data.get('measure', {}).get('results', []) if self.doc else []:
            pts = r.get('points')
            if pts and len(pts) == 2:
                items[r['text_id']] = (tuple(pts[0]), tuple(pts[1]), r['measured_mm'])
        self.canvas.set_items(items, self.active_text)

    def _on_canvas_active(self, tid):
        self.active_text = tid
        self.set_active_text(tid)

    def _on_canvas_deleted(self, tid):
        if self.doc is None:
            return
        self.doc.push_undo()
        res = self.doc.data['measure'].get('results', [])
        self.doc.data['measure']['results'] = [x for x in res if x['text_id'] != tid]
        self.doc.dirty = True
        self.refresh_table()
        self.docChanged.emit()

    def set_active_text(self, tid):
        """도면 캔버스에서 치수를 고르면 호출된다."""
        self.active_text = tid
        self.canvas.set_active(tid)
        if self.doc is None or tid is None:
            self.lbl_target.setText('도면에서 치수를 클릭하세요')
            return
        t = self.doc.find('texts', tid)
        if t is None:
            return
        parsed = cp.parse_dimension(t.get('text', ''))
        link = self.doc.get_link(tid)
        n = (len(link['line_ids']) + len(link.get('arc_ids', []))) if link else 0
        tol = ('공차 없음' if parsed['upper'] is None
               else f"+{parsed['upper']:g} / {parsed['lower']:g}")
        warn = '' if n else '   ⚠ 이 치수는 아직 도면 외곽선에 연결되지 않았습니다'
        self.lbl_target.setText(
            f"측정 대상: {t.get('text','')}  ·  공칭 {parsed['nominal']}  ·  {tol}"
            f"  ·  연결 {n}개{warn}")

    def auto_trace(self):
        """연결된 치수마다 측정점을 역추적한다. 사람이 고친 것은 건드리지 않는다."""
        if self.doc is None:
            return
        import numpy as _np
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'pipeline'))
        from line_detect import traceback_points as tp
        L = _np.array([[l['p1'][0], l['p1'][1], l['p2'][0], l['p2'][1]]
                       for l in self.doc.data['lines']], float)
        if len(L) == 0:
            self.statusMessage.emit('선분이 없습니다')
            return
        polys = [t['poly'] for t in self.doc.data['texts']]
        params = tp.scaled_params(tp.text_scale(polys))
        id2i = {l['id']: i for i, l in enumerate(self.doc.data['lines'])}
        self.doc.push_undo()
        n = {'traced': 0, 'partial': 0, 'fallback': 0, 'skip': 0}
        for link in self.doc.data['links']:
            m = link.get('measure')
            if m and m.get('source') == 'human':
                n['skip'] += 1          # 사람이 고친 것은 덮지 않는다
                continue
            idxs = [id2i[x] for x in link.get('line_ids', []) if x in id2i]
            if not idxs:
                continue
            i = max(idxs, key=lambda k: _np.hypot(L[k, 2] - L[k, 0], L[k, 3] - L[k, 1]))
            r = tp.trace_measure_points(L[i], L, params, exclude_idx={i})
            link['measure'] = {'points': [[float(a), float(b)] for a, b in r['points']],
                               'quality': r['quality'], 'source': 'auto'}
            n[r['quality']] += 1
        self.doc.dirty = True
        self.statusMessage.emit(
            f"측정점 추출: 추적 {n['traced']} · 부분 {n['partial']} · "
            f"치수선대체 {n['fallback']} · 사람수정 유지 {n['skip']}  "
            f"(글자높이 {params['text_h']:.0f}px 기준)")
        self.docChanged.emit()

    # ── 사진 ──────────────────────────────────────────────
    def open_photo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '제품 사진 열기', '', 'Images (*.jpg *.jpeg *.png *.bmp)')
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, '실패', '사진을 읽지 못했습니다.')
            return
        marker = self.sp_marker.value()
        board = cal.make_board(2, 2, marker_mm=marker,
                               gap_mm=self.sp_pitch.value() - marker)
        try:
            r = cal.rectify_board(img, board, px_per_mm=8.0)
        except ValueError as e:
            QMessageBox.warning(self, 'ArUco 보드를 못 찾음', str(e))
            return
        self.calib = r
        self.canvas.set_photo(r['rectified'], r['px_per_mm'])
        self._sync_canvas()
        msg = (f"마커 {r['n_markers']}/4 · 평균 {r['marker_px']:.0f}px · "
               f"잔차 {r['residual_mm']:.2f}mm")
        self.lbl_calib.setText(msg)
        self.lbl_calib.setStyleSheet(
            'color:#a00;' if r['warnings'] or r['residual_mm'] > 0.3 else 'color:#080;')
        if self.doc is not None:
            self.doc.data['measure']['photo_path'] = path
            self.doc.data['measure']['calibration'] = {
                'marker_mm': marker, 'pitch_mm': self.sp_pitch.value(),
                'px_per_mm': r['px_per_mm'], 'n_markers': r['n_markers'],
                'residual_mm': round(r['residual_mm'], 4),
            }
            self.doc.dirty = True
        self.statusMessage.emit(msg + ('  ' + ' / '.join(r['warnings']) if r['warnings'] else ''))

    # ── 측정 기록 ─────────────────────────────────────────
    def on_measured(self, mm, p1, p2):
        if self.doc is None or self.calib is None:
            return
        if not self.active_text:
            self.statusMessage.emit('먼저 도면에서 비교할 치수를 클릭하세요')
            return
        t = self.doc.find('texts', self.active_text)
        if t is None:
            return
        parsed = cp.parse_dimension(t.get('text', ''))

        pitch = self.sp_pitch.value()
        ctr = np.array(self.calib['marker_origin_px']) + pitch * self.calib['px_per_mm'] / 2
        dist_mm = float(np.hypot(*((np.array(p1) + np.array(p2)) / 2 - ctr))) \
            / self.calib['px_per_mm']
        u = cal.measurement_uncertainty(mm, dist_mm, pitch,
                                        px_per_mm=self.calib['px_per_mm'])
        j = cp.judge(parsed, mm, u['total'])

        self.doc.push_undo()
        results = self.doc.data['measure'].setdefault('results', [])
        results[:] = [x for x in results if x['text_id'] != self.active_text]
        results.append({
            'text_id': self.active_text, 'text': t.get('text', ''),
            'nominal': parsed['nominal'], 'upper': parsed['upper'],
            'lower': parsed['lower'], 'kind': parsed['kind'],
            'measured_mm': round(float(mm), 3),
            'uncertainty_mm': round(float(u['total']), 3),
            'deviation_mm': None if j['deviation'] is None else round(j['deviation'], 3),
            'verdict': j['verdict'], 'reason': j['reason'],
            'points': [[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]],
            'source': 'human', 'verified': True,
        })
        self.doc.data['history'].append(
            {'action': 'measure_add', 'text_id': self.active_text})
        self.doc.dirty = True
        self.refresh_table()
        self._sync_canvas()
        self.docChanged.emit()

    def delete_selected(self):
        if self.doc is None:
            return
        rows = {i.row() for i in self.table.selectedIndexes()}
        if not rows:
            return
        res = self.doc.data['measure'].get('results', [])
        self.doc.push_undo()
        self.doc.data['measure']['results'] = [
            r for i, r in enumerate(res) if i not in rows]
        self.doc.dirty = True
        self.refresh_table()
        self._sync_canvas()
        self.docChanged.emit()

    def refresh_table(self):
        self.table.setRowCount(0)
        if self.doc is None:
            self.lbl_summary.setText('')
            return
        res = self.doc.data.get('measure', {}).get('results', [])
        counts = {}
        for r in res:
            row = self.table.rowCount()
            self.table.insertRow(row)
            tol = ('-' if r['upper'] is None
                   else f"+{r['upper']:g}/{r['lower']:g}")
            dev = '-' if r['deviation_mm'] is None else f"{r['deviation_mm']:+.2f}"
            cells = [r['text'], '-' if r['nominal'] is None else f"{r['nominal']:g}",
                     tol, f"{r['measured_mm']:.2f} ±{r['uncertainty_mm']:.2f}",
                     dev, VERDICT_TEXT.get(r['verdict'], r['verdict'])]
            for c, v in enumerate(cells):
                it = QTableWidgetItem(v)
                it.setBackground(VERDICT_COLOR.get(r['verdict'], QColor(255, 255, 255)))
                if r.get('reason'):
                    it.setToolTip(r['reason'])
                self.table.setItem(row, c, it)
            counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
        if res:
            self.lbl_summary.setText('  '.join(
                f"{VERDICT_TEXT.get(k, k)} {v}" for k, v in sorted(counts.items())))
        else:
            self.lbl_summary.setText('')
