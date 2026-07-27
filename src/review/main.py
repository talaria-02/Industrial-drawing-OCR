# -*- coding: utf-8 -*-
"""도면 검수 UI (PyQt5) — 자동 처리 결과를 사람이 확인/수정하고, 그 수정을
재학습 데이터로 내보내는 도구.

실행:
  python -m src.review.main          (프로젝트 루트에서)
  또는  python src/review/main.py

[전체 루프에서의 위치]
  자동 파이프라인(OCR→선분→매칭→화살촉)  →  [이 도구로 사람이 검수]  →
  export/ (Label.txt, matching_gt.json)  →  정확도 측정 / 모델 재학습

100% 자동화가 어렵다는 전제 위에 설계됐다. 모든 레이어(텍스트/선분/화살촉/매칭/
카테고리)를 사람이 고칠 수 있고, 고친 것에는 source='human', verified=True가
붙어서 내보낼 때 자동 결과와 구분된다.

[OCR을 별도 스레드에서 돌리는 이유]
도면 한 장 OCR이 25~30초 걸린다. 메인 스레드에서 돌리면 창이 응답 없음 상태가
되므로 QThread로 분리하고 진행 상황만 상태바에 표시한다.
"""
import os
import sys

# ── torch를 PyQt5보다 먼저 import해야 한다 (중요) ──────────────────────────
# ppocr.data -> imaug -> iaa_augment -> albumentations -> torch 경로로 torch가
# 반드시 로드된다(회피 불가). 그런데 PyQt5가 먼저 로드된 상태에서 torch를 import하면
# c10.dll 초기화가 실패한다(WinError 1114). 둘이 같은 런타임 DLL을 다투기 때문.
# 반대 순서(torch 먼저)는 문제가 없으므로, 이 파일 최상단에서 선점해 둔다.
# 여기서 torch를 직접 쓰지는 않는다 — 순서 확보가 목적.
try:
    import torch  # noqa: F401
except Exception as _torch_err:      # noqa: F841  (torch가 아예 없는 환경도 허용)
    pass

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QFileDialog, QLabel,
                             QVBoxLayout, QHBoxLayout, QPushButton, QRadioButton,
                             QButtonGroup, QListWidget, QListWidgetItem, QLineEdit,
                             QComboBox, QGroupBox, QFormLayout, QMessageBox, QSplitter,
                             QCheckBox, QShortcut)

if __package__ in (None, ''):   # 파일 직접 실행 지원
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.review import bootstrap, canvas as C
    from src.review.model import ReviewDoc, CATEGORIES
else:
    from . import bootstrap, canvas as C
    from .model import ReviewDoc, CATEGORIES

MODES = [
    ('텍스트', C.MODE_TEXT),
    ('선분', C.MODE_LINE),
    ('원/호', C.MODE_ARC),
    ('화살촉', C.MODE_ARROW),
    ('매칭', C.MODE_MATCH),
    ('카테고리', C.MODE_CATEGORY),
    ('측정(⑦)', C.MODE_MEASURE),
]

MODE_HELP = {
    C.MODE_TEXT: '빈곳 2클릭=새 박스 → F2로 인식 · 클릭=선택 · 모서리드래그=크기 · 안쪽드래그=이동 · Del=삭제',
    C.MODE_LINE: '끝점드래그=수정 · 빈곳 2클릭=새 선 · 클릭=선택 · Del=삭제',
    C.MODE_ARC: '빈곳 2클릭(중심→둘레)=새 원 · 파란점=중심이동 · 둘레드래그=반지름 · 초록/주황점=각도 · C=원↔호 전환 · Del=삭제',
    C.MODE_ARROW: '끝점(또는 선 몸통) 클릭마다 순환: 회색(미검사) → 초록(있음) → 빨강(없음) → 회색',
    C.MODE_MATCH: '숫자 클릭 → 선 클릭(여러 개 가능) → Enter 확정 · 연결된 선 재클릭=그 선만 해제 · Del=전체 해제 · 우클릭=취소',
    C.MODE_CATEGORY: '숫자 클릭 후 오른쪽 패널에서 카테고리 변경',
    C.MODE_MEASURE: '제품사진 비교 기능은 향후 추가 예정 (자리만 예약)',
}


class BootstrapWorker(QThread):
    progress = pyqtSignal(str)
    done = pyqtSignal(object, object)     # (ReviewDoc, error_or_None)

    def __init__(self, img_path):
        super().__init__()
        self.img_path = img_path

    def run(self):
        try:
            doc = bootstrap.build_review(self.img_path, progress=self.progress.emit)
            self.done.emit(doc, None)
        except Exception as e:      # UI가 죽지 않도록 예외를 넘겨서 표시만 한다
            import traceback
            self.done.emit(None, traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('현장미러형 — 도면 검수')
        self.resize(1500, 950)
        self.img_path = None
        self.doc = None
        self.worker = None
        self.bgr = None       # 원본 이미지(rec 재실행용). 매번 디스크에서 읽지 않게 캐시

        self.canvas = C.Canvas()
        self.canvas.selectionChanged.connect(self.on_selection)
        self.canvas.docChanged.connect(self.on_doc_changed)
        self.canvas.statusMessage.connect(lambda s: self.statusBar().showMessage(s, 4000))

        # ── 상단 툴바 ──────────────────────────────────────
        top = QHBoxLayout()
        for label, fn in (('열기', self.open_image), ('저장', self.save_doc),
                           ('내보내기', self.export_all), ('자동 재실행', self.rerun_auto)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            top.addWidget(b)
        top.addSpacing(10)
        self.btn_undo = QPushButton('↶ 되돌리기')
        self.btn_undo.setToolTip('Ctrl+Z')
        self.btn_undo.clicked.connect(self.do_undo)
        self.btn_redo = QPushButton('↷ 다시실행')
        self.btn_redo.setToolTip('Ctrl+Shift+Z 또는 Ctrl+Y')
        self.btn_redo.clicked.connect(self.do_redo)
        top.addWidget(self.btn_undo)
        top.addWidget(self.btn_redo)
        top.addSpacing(20)
        top.addWidget(QLabel('모드:'))
        self.mode_group = QButtonGroup(self)
        for i, (label, mode) in enumerate(MODES):
            rb = QRadioButton(label)
            rb.setProperty('mode', mode)
            if mode == C.MODE_MEASURE:
                rb.setEnabled(False)          # ⑦ 자리만 예약
            if mode == C.MODE_MATCH:
                rb.setChecked(True)
            self.mode_group.addButton(rb, i)
            top.addWidget(rb)
        self.mode_group.buttonClicked.connect(self.on_mode)
        self.chk_unlinked = QCheckBox('미연결 선 표시')
        self.chk_unlinked.setChecked(True)
        self.chk_unlinked.stateChanged.connect(self.on_toggle_unlinked)
        top.addSpacing(20)
        top.addWidget(self.chk_unlinked)
        top.addStretch(1)

        self.help_label = QLabel(MODE_HELP[C.MODE_MATCH])
        self.help_label.setStyleSheet('color:#555;')

        # ── 오른쪽 패널 ────────────────────────────────────
        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self.on_list_select)

        prop = QGroupBox('속성')
        form = QFormLayout()
        self.edit_text = QLineEdit()
        self.edit_text.editingFinished.connect(self.apply_text_edit)
        self.combo_cat = QComboBox()
        self.combo_cat.addItems(CATEGORIES)
        self.combo_cat.currentIndexChanged.connect(self.apply_category)
        self.lbl_info = QLabel('-')
        self.lbl_info.setWordWrap(True)
        form.addRow('텍스트', self.edit_text)
        form.addRow('카테고리', self.combo_cat)
        form.addRow('정보', self.lbl_info)
        self.btn_recognize = QPushButton('선택한 박스 인식 (F2)')
        self.btn_recognize.setToolTip(
            '텍스트 모드에서 박스를 그린 뒤 누르면 rec 모델이 그 안의 글자를 읽습니다')
        self.btn_recognize.clicked.connect(self.recognize_selected)
        form.addRow(self.btn_recognize)
        self.btn_clear_link = QPushButton('이 숫자의 연결 모두 해제')
        self.btn_clear_link.clicked.connect(self.clear_selected_link)
        form.addRow(self.btn_clear_link)
        prop.setLayout(form)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel('객체 목록 (클릭 = 해당 위치로 이동)'))
        rl.addWidget(self.list_widget, 1)
        rl.addWidget(prop)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.canvas, 1)
        ll.addWidget(self.help_label)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.addLayout(top)
        cl.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self._install_shortcuts()
        self.update_undo_buttons()
        self.statusBar().showMessage(
            '도면을 열어주세요.  |  확대=휠 · 이동=빈곳 드래그 / Space+드래그 / 가운데버튼 / 방향키 · '
            'F=전체보기 · H=미연결선 숨기기')

    # ── 되돌리기 / 다시실행 ────────────────────────────────
    def _install_shortcuts(self):
        # 캔버스가 키 입력을 먼저 먹지 않도록 창(window) 범위 단축키로 등록한다.
        for keys, fn in ((('Ctrl+Z',), self.do_undo),
                          (('Ctrl+Shift+Z', 'Ctrl+Y'), self.do_redo),
                          (('F2',), self.recognize_selected)):
            for k in keys:
                sc = QShortcut(QKeySequence(k), self)
                sc.setContext(Qt.ApplicationShortcut)
                sc.activated.connect(fn)

    def do_undo(self):
        if self.doc is None or not self.doc.undo():
            self.statusBar().showMessage('되돌릴 내용이 없습니다', 2000)
            return
        self._after_history_move('되돌렸습니다')

    def do_redo(self):
        if self.doc is None or not self.doc.redo():
            self.statusBar().showMessage('다시실행할 내용이 없습니다', 2000)
            return
        self._after_history_move('다시실행했습니다')

    def _after_history_move(self, msg):
        # 되돌리면서 객체가 사라졌을 수 있으므로 선택 상태를 검증한다
        c = self.canvas
        if c.sel_kind == 'text' and self.doc.find('texts', c.sel_id) is None:
            c.sel_kind = c.sel_id = None
        elif c.sel_kind == 'line' and self.doc.find('lines', c.sel_id) is None:
            c.sel_kind = c.sel_id = None
        if c.pending_text_id and self.doc.find('texts', c.pending_text_id) is None:
            c.pending_text_id = None
        self.refresh_list()
        self.on_selection()
        self.canvas.update()
        self.update_undo_buttons()
        self.statusBar().showMessage(msg, 2000)

    def update_undo_buttons(self):
        ok = self.doc is not None
        self.btn_undo.setEnabled(ok and self.doc.can_undo())
        self.btn_redo.setEnabled(ok and self.doc.can_redo())

    # ── 파일 ──────────────────────────────────────────────
    def open_image(self):
        start = os.path.join(bootstrap.PROJECT_ROOT, 'data', 'real', 'train')
        path, _ = QFileDialog.getOpenFileName(self, '도면 열기', start,
                                               '이미지 (*.jpg *.jpeg *.png)')
        if not path:
            return
        self.img_path = path
        rp = bootstrap.review_json_path(path)
        if os.path.exists(rp):
            ret = QMessageBox.question(
                self, '기존 검수 결과', '이 도면의 검수 파일이 있습니다. 불러올까요?\n'
                '(아니오 = 자동 파이프라인을 새로 실행)',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ret == QMessageBox.Yes:
                self.doc = ReviewDoc.load(rp)
                self._show_doc()
                return
        self.run_auto()

    def rerun_auto(self):
        if not self.img_path:
            return
        ret = QMessageBox.question(self, '자동 재실행',
                                    '현재 검수 내용을 버리고 자동 파이프라인을 다시 실행합니다.',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret == QMessageBox.Yes:
            self.run_auto()

    def run_auto(self):
        self.statusBar().showMessage('자동 파이프라인 시작...')
        self.worker = BootstrapWorker(self.img_path)
        self.worker.progress.connect(lambda s: self.statusBar().showMessage(s))
        self.worker.done.connect(self.on_auto_done)
        self.worker.start()

    def on_auto_done(self, doc, err):
        if err:
            QMessageBox.critical(self, '자동 처리 실패', str(err)[-3000:])
            self.statusBar().showMessage('자동 처리 실패')
            return
        self.doc = doc
        self._show_doc()

    def _show_doc(self):
        self.bgr = cv2.imdecode(np.fromfile(self.img_path, np.uint8), cv2.IMREAD_COLOR)
        self.canvas.set_document(self.doc, self.bgr)
        self.doc._undo.clear(); self.doc._redo.clear()   # 새 문서는 이력 초기화
        self.update_undo_buttons()
        self.refresh_list()
        self.update_status()

    def save_doc(self):
        if self.doc is None or not self.img_path:
            return
        p = bootstrap.review_json_path(self.img_path)
        self.doc.save(p)
        self.statusBar().showMessage(f'저장: {p}', 5000)

    def export_all(self):
        if self.doc is None or not self.img_path:
            return
        d = os.path.join(bootstrap.review_dir(self.img_path), 'export')
        rel = os.path.join('train', os.path.basename(self.img_path)).replace('\\', '/')
        n_lbl = self.doc.export_label_txt(os.path.join(d, 'Label.txt'), rel)
        n_gt = self.doc.export_matching_gt(os.path.join(d, 'matching_gt.json'))
        self.doc.save(bootstrap.review_json_path(self.img_path))
        QMessageBox.information(
            self, '내보내기 완료',
            f'검수된 텍스트 {n_lbl}건 → Label.txt\n검수된 연결 {n_gt}건 → matching_gt.json\n\n{d}')

    # ── 모드/목록/속성 ────────────────────────────────────
    def on_mode(self, btn):
        m = btn.property('mode')
        self.canvas.mode = m
        self.canvas.pending_text_id = None
        self.canvas.update()
        self.help_label.setText(MODE_HELP.get(m, ''))

    def on_toggle_unlinked(self):
        self.canvas.show_unlinked_lines = self.chk_unlinked.isChecked()
        self.canvas.update()

    def refresh_list(self):
        # 목록을 다시 만들면 현재 선택이 풀리므로, 갱신 후 캔버스 선택과 다시 맞춘다.
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        if self.doc is not None:
            for t in self.doc.data['texts']:
                link = self.doc.get_link(t['id'])
                n = len(link['line_ids']) if link else 0
                mark = '✓' if (link and link.get('verified')) else ('·' if n else '✗')
                item = QListWidgetItem(
                    f'{mark} {t.get("text","")[:14]:<14} [{t.get("category","")}] 선{n}')
                item.setData(Qt.UserRole, ('text', t['id']))
                self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)
        self.sync_list_to_canvas()

    def sync_list_to_canvas(self):
        """캔버스에서 선택된 객체를 오른쪽 목록에서도 선택 상태로 만들고 보이게 스크롤.

        blockSignals로 신호를 막는 이유: 목록 선택이 바뀌면 on_list_select가 불려서
        캔버스를 다시 그 위치로 이동시킨다. 캔버스에서 선택한 것 때문에 목록을 맞추는
        중인데 그게 또 캔버스를 움직이면 화면이 튀고(사용자가 보던 위치가 바뀜),
        무한 왕복이 될 수 있다."""
        c = self.canvas
        target = None
        if c.sel_kind == 'text' and c.sel_id:
            target = ('text', c.sel_id)
        elif c.sel_kind == 'line' and c.sel_id:
            # 선분을 골랐으면, 그 선과 연결된 숫자를 목록에서 짚어준다
            for link in (self.doc.data['links'] if self.doc else []):
                if c.sel_id in link['line_ids']:
                    target = ('text', link['text_id'])
                    break
        self.list_widget.blockSignals(True)
        if target is None:
            self.list_widget.setCurrentRow(-1)
        else:
            for i in range(self.list_widget.count()):
                if self.list_widget.item(i).data(Qt.UserRole) == target:
                    self.list_widget.setCurrentRow(i)
                    self.list_widget.scrollToItem(self.list_widget.item(i))
                    break
        self.list_widget.blockSignals(False)

    def on_list_select(self, cur, _prev):
        if cur is None:
            return
        kind, oid = cur.data(Qt.UserRole)
        self.canvas.focus_on(kind, oid)
        self.on_selection()

    def on_selection(self):
        self.sync_list_to_canvas()
        k, oid = self.canvas.sel_kind, self.canvas.sel_id
        self.edit_text.blockSignals(True)
        self.combo_cat.blockSignals(True)
        if k == 'text' and self.doc is not None:
            t = self.doc.find('texts', oid)
            if t:
                self.edit_text.setEnabled(True)
                self.combo_cat.setEnabled(True)
                self.edit_text.setText(t.get('text', ''))
                cat = t.get('category', '기타')
                self.combo_cat.setCurrentIndex(
                    CATEGORIES.index(cat) if cat in CATEGORIES else len(CATEGORIES) - 1)
                link = self.doc.get_link(oid)
                lids = ', '.join(link['line_ids']) if link else '없음'
                self.lbl_info.setText(
                    f'id={oid} · score={t.get("score")} · {"검수됨" if t.get("verified") else "자동"}\n'
                    f'연결: {lids}')
        elif k == 'line' and self.doc is not None:
            l = self.doc.find('lines', oid)
            self.edit_text.setEnabled(False)
            self.combo_cat.setEnabled(False)
            self.btn_clear_link.setEnabled(False)
            self.edit_text.setText('')
            if l:
                a1 = self.doc.get_arrow(oid, 'start')
                a2 = self.doc.get_arrow(oid, 'end')
                def fmt(a):
                    if a is None:
                        return '미검사'
                    return ('있음' if a['present'] else '없음') + \
                           (f'({a["score"]:.2f})' if a.get('score') is not None else '')
                ln = ((l['p2'][0] - l['p1'][0]) ** 2 + (l['p2'][1] - l['p1'][1]) ** 2) ** 0.5
                self.lbl_info.setText(f'id={oid} · 길이 {ln:.1f}px\n'
                                       f'화살촉 start={fmt(a1)} end={fmt(a2)}')
        else:
            self.edit_text.setEnabled(False)
            self.combo_cat.setEnabled(False)
            self.btn_clear_link.setEnabled(False)
            self.btn_recognize.setEnabled(False)
            self.lbl_info.setText('-')
        self.edit_text.blockSignals(False)
        self.combo_cat.blockSignals(False)

    def apply_text_edit(self):
        if self.doc and self.canvas.sel_kind == 'text' and self.canvas.sel_id:
            t = self.doc.find('texts', self.canvas.sel_id)
            new = self.edit_text.text()
            if t is None or t.get('text', '') == new:
                return          # 값이 그대로면 되돌리기 이력을 더럽히지 않는다
            self.doc.update_text(self.canvas.sel_id, text=new)
            self.on_doc_changed()

    def apply_category(self):
        if self.doc and self.canvas.sel_kind == 'text' and self.canvas.sel_id:
            t = self.doc.find('texts', self.canvas.sel_id)
            new = self.combo_cat.currentText()
            if t is None or t.get('category') == new:
                return
            self.doc.update_text(self.canvas.sel_id, category=new)
            self.on_doc_changed()

    def recognize_selected(self):
        """선택된 텍스트 박스 안의 글자를 rec 모델로 읽어서 채워넣는다.
        det가 놓친 텍스트를 사람이 박스로 표시해준 경우에 쓴다."""
        c = self.canvas
        if self.doc is None or self.bgr is None or c.sel_kind != 'text' or not c.sel_id:
            self.statusBar().showMessage('먼저 텍스트 박스를 선택하세요 (텍스트 모드)', 3000)
            return
        t = self.doc.find('texts', c.sel_id)
        if t is None:
            return
        self.statusBar().showMessage('인식 중...')
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            text, score = bootstrap.recognize_box(self.bgr, t['poly'])
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, '인식 실패', str(e)[-2000:])
            return
        QApplication.restoreOverrideCursor()
        if not text:
            self.statusBar().showMessage('글자를 읽지 못했습니다 — 박스를 조금 넓혀보세요', 4000)
            return
        self.doc.update_text(c.sel_id, text=text, score=round(score, 3),
                              category=bootstrap.guess_category(text))
        self.on_doc_changed()
        self.on_selection()
        self.statusBar().showMessage(f'인식 결과: "{text}" (신뢰도 {score:.2f})', 5000)

    def clear_selected_link(self):
        c = self.canvas
        tid = c.pending_text_id or (c.sel_id if c.sel_kind == 'text' else None)
        if self.doc is None or not tid:
            self.statusBar().showMessage('먼저 숫자를 선택하세요', 2000)
            return
        if self.doc.get_link(tid) is None:
            self.statusBar().showMessage('해제할 연결이 없습니다', 2000)
            return
        self.doc.clear_link(tid)
        c.pending_text_id = None
        self.on_doc_changed()
        self.on_selection()
        self.statusBar().showMessage('연결을 해제했습니다 (Ctrl+Z로 복구 가능)', 3000)

    def on_doc_changed(self):
        self.update_undo_buttons()
        self.refresh_list()
        self.update_status()
        self.canvas.update()

    def update_status(self):
        if self.doc is None:
            return
        s = self.doc.stats()
        self.statusBar().showMessage(
            f'텍스트 {s["texts_verified"]}/{s["texts"]} 검수 · '
            f'선분 {s["lines"]} · 연결 {s["links_verified"]}/{s["links"]} 검수 · '
            f'화살촉 있음 {s["arrows_present"]}/{s["arrows"]}'
            + ('   [저장 안 됨]' if self.doc.dirty else ''))


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
