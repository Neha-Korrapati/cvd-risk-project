"""CVD Risk Screening Workbench: desktop version of the paper's Fig. 5 web
application (DEVIATIONS.md P-02).

* Reads a patient's five model inputs from a FHIR R4 server (the EHR stand-in),
  or lets the clinician enter / edit them manually.
* Shows the predicted CVD risk against the decision threshold, the
  patient-specific SHAP contributions (Fig. 4b/c), the sensitivity /
  specificity curve (Fig. 3c / Fig. 5), the raw FHIR resources, and a model
  card documenting performance and limitations.

Run from the project root:
    .venv\\Scripts\\python -m desktop_app.app
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QRectF, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QIcon, QLinearGradient, QPainter,
                           QPainterPath, QPalette, QPen, QPixmap)
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QFrame,
                               QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPlainTextEdit, QPushButton, QSizePolicy, QSplitter,
                               QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser,
                               QToolButton, QVBoxLayout, QWidget)

from cvd import config as C
from cvd import data, fhir_mapping
from cvd.fhir_client import DEFAULT_BASE, FhirClient
from cvd.predict import RiskModel

APP_NAME = "CVD Risk Screening Workbench"
APP_SUBTITLE = ("Explainable XGBoost on five clinical features  ·  reproduction of "
                "Vyshnya et al., IEEE Open Journal of Engineering in Medicine and Biology, 2024")

CHEST_PAIN = {"ASY": "Asymptomatic (ASY)", "ATA": "Atypical angina (ATA)",
              "NAP": "Non-anginal pain (NAP)", "TA": "Typical angina (TA)"}
ST_SLOPE = {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"}
FEATURE_ORDER = ["ST_Slope_Flat", "ExerciseAngina_Y", "Oldpeak", "ChestPainType_ATA", "Sex_M"]

# Palette: clinical navy / teal UI; SHAP red/blue as in the paper's Fig. 4.
NAVY, TEAL = "#0f3d6e", "#0e7c86"
INK, MUTED, LINE, CANVAS = "#1f2a37", "#64748b", "#dbe2ea", "#f3f6f9"
RED, BLUE = "#d81b60", "#1e88e5"                    # SHAP: raises / lowers risk
HIGH, HIGH_BG, LOW, LOW_BG = "#b42318", "#fdecea", "#1b6e3a", "#e7f4ec"

matplotlib.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": "#9aa5b1", "axes.labelcolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": INK,
    "axes.grid": True, "grid.color": "#e8edf2", "grid.linewidth": 0.8,
})

STYLE = f"""
QMainWindow, QWidget#Root {{ background: {CANVAS}; }}
QWidget {{ color: {INK}; font-family: "Segoe UI"; font-size: 10pt; }}
QFrame#Header {{ background: {NAVY}; }}
QLabel#HeaderTitle {{ color: white; font-size: 15pt; font-weight: 600; }}
QLabel#HeaderSub {{ color: #c9d6e6; font-size: 9pt; }}
QLabel#Logo {{ background: white; color: {NAVY}; border-radius: 8px; font-weight: 700;
               font-size: 11pt; }}
QLabel#Ruo {{ color: #ffd79a; border: 1px solid #ffd79a; border-radius: 4px;
              padding: 3px 8px; font-size: 8pt; font-weight: 600; letter-spacing: 1px; }}
QLabel#Pill {{ background: rgba(255,255,255,0.12); color: white; border-radius: 11px;
               padding: 3px 12px; font-size: 9pt; }}
QFrame#Card {{ background: white; border: 1px solid {LINE}; border-radius: 10px; }}
QLabel#CardTitle {{ color: {MUTED}; font-size: 8pt; font-weight: 700; letter-spacing: 1px; }}
QLabel#Muted {{ color: {MUTED}; font-size: 9pt; }}
QLabel#KpiValue {{ font-size: 15pt; font-weight: 600; color: {INK}; }}
QLabel#KpiLabel {{ color: {MUTED}; font-size: 8pt; }}
QLabel#RiskValue {{ font-size: 34pt; font-weight: 600; }}
QLabel#Chip {{ background: #eef3f8; color: {NAVY}; border-radius: 10px; padding: 3px 10px;
               font-size: 9pt; }}
QLineEdit, QComboBox, QDoubleSpinBox {{ background: white; border: 1px solid #c8d1db;
    border-radius: 6px; padding: 5px 8px; min-height: 20px; }}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {TEAL}; }}
QComboBox QAbstractItemView {{ background: white; selection-background-color: #dff1f2;
    selection-color: {INK}; border: 1px solid {LINE}; }}
QPushButton {{ background: white; border: 1px solid #c8d1db; border-radius: 6px;
    padding: 6px 14px; color: {NAVY}; font-weight: 600; }}
QPushButton:hover {{ background: #eef5fb; border-color: {NAVY}; }}
QPushButton#Primary {{ background: {TEAL}; color: white; border: none; padding: 10px;
    font-size: 11pt; }}
QPushButton#Primary:hover {{ background: #0b6870; }}
QToolButton#Link {{ border: none; color: {MUTED}; font-size: 9pt; padding: 2px 0; }}
QToolButton#Link:hover {{ color: {NAVY}; }}
QListWidget {{ background: white; border: 1px solid {LINE}; border-radius: 6px; outline: 0; }}
QListWidget::item {{ padding: 7px 10px; border-bottom: 1px solid #f0f3f6; }}
QListWidget::item:selected {{ background: #dff1f2; color: {INK}; }}
QListWidget::item:hover {{ background: #f3f8fb; }}
QTabWidget::pane {{ background: white; border: 1px solid {LINE}; border-radius: 10px;
    top: -1px; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 8px 16px; margin-right: 2px;
    border-bottom: 2px solid transparent; font-weight: 600; }}
QTabBar::tab:selected {{ color: {NAVY}; border-bottom: 2px solid {TEAL}; }}
QTabBar::tab:hover {{ color: {INK}; }}
QTableWidget {{ background: white; border: none; gridline-color: #eef1f4; }}
QTableWidget::item {{ padding: 6px 8px; border-bottom: 1px solid #f0f3f6; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox::down-arrow {{ image: url(%ASSETS%/chevron-down.png); width: 12px; height: 12px; }}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ border: none; width: 22px; }}
QDoubleSpinBox::up-arrow {{ image: url(%ASSETS%/chevron-up.png); width: 10px; height: 10px; }}
QDoubleSpinBox::down-arrow {{ image: url(%ASSETS%/chevron-down.png); width: 10px; height: 10px; }}
QHeaderView::section {{ background: #f6f8fa; color: {MUTED}; border: none;
    border-bottom: 1px solid {LINE}; padding: 6px; font-size: 8pt; font-weight: 700; }}
QPlainTextEdit, QTextBrowser {{ background: white; border: none; }}
QPlainTextEdit {{ font-family: "Cascadia Mono", "Consolas"; font-size: 9pt; }}
QStatusBar {{ background: white; color: {MUTED}; border-top: 1px solid {LINE};
    font-size: 8pt; }}
QSplitter::handle {{ background: {CANVAS}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: #c8d1db; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""


class Worker(QRunnable):
    """Runs `fn` off the UI thread and reports back through signals.

    Results are reported with the worker's key so the window can look up the
    callbacks itself. Connecting the signals directly to lambdas is unreliable:
    a lambda has no receiver object, so Qt may drop the queued call once the
    worker finishes, leaving the UI stuck on "computing…"."""

    class Signals(QObject):
        done = Signal(int, object)
        failed = Signal(int, str)

    def __init__(self, key: int, fn, *args):
        super().__init__()
        self.setAutoDelete(False)  # the window owns the worker until it reports back
        self.key, self.fn, self.args, self.signals = key, fn, args, Worker.Signals()

    def run(self):
        try:
            self.signals.done.emit(self.key, self.fn(*self.args))
        except Exception as e:  # surfaced to the user in a message box
            self.signals.failed.emit(self.key, f"{type(e).__name__}: {e}")


class Canvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(6, 4), layout="constrained", facecolor="white")
        super().__init__(self.fig)


class RiskGauge(QWidget):
    """Horizontal 0–100% risk scale with the decision threshold and the
    patient's predicted risk marked on it."""

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold, self.risk = threshold, None
        self.setMinimumHeight(54)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_risk(self, risk):
        self.risk = risk
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, pad = self.width(), 8
        bar = QRectF(pad, 18, w - 2 * pad, 10)
        grad = QLinearGradient(bar.left(), 0, bar.right(), 0)
        grad.setColorAt(0.0, QColor("#2e8b57"))
        grad.setColorAt(self.threshold, QColor("#e3b341"))
        grad.setColorAt(1.0, QColor(HIGH))
        path = QPainterPath()
        path.addRoundedRect(bar, 5, 5)
        p.fillPath(path, QBrush(grad))

        x = lambda v: bar.left() + v * bar.width()
        small = QFont("Segoe UI", 8)
        p.setFont(small)
        # Threshold tick and label
        p.setPen(QPen(QColor(INK), 2))
        p.drawLine(int(x(self.threshold)), 12, int(x(self.threshold)), 34)
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(x(self.threshold) - 60, 0, 120, 12), Qt.AlignCenter,
                   f"threshold {100 * self.threshold:.0f}%")
        for v, align in ((0, Qt.AlignLeft), (0.5, Qt.AlignHCenter), (1.0, Qt.AlignRight)):
            left = {0: bar.left(), 0.5: x(0.5) - 25, 1.0: bar.right() - 50}[v]
            p.drawText(QRectF(left, 36, 50, 14), align | Qt.AlignVCenter, f"{100 * v:.0f}%")
        # Patient marker
        if self.risk is not None:
            cx = x(self.risk)
            p.setPen(QPen(QColor("white"), 2))
            p.setBrush(QColor(HIGH if self.risk >= self.threshold else LOW))
            p.drawEllipse(QRectF(cx - 8, 15, 16, 16))
        p.end()


def card(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame(objectName="Card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(16, 14, 16, 16)
    lay.setSpacing(10)
    lay.addWidget(QLabel(title.upper(), objectName="CardTitle"))
    return frame, lay


def kpi(value: str, label: str) -> QWidget:
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(0)
    v.addWidget(QLabel(value, objectName="KpiValue"))
    v.addWidget(QLabel(label, objectName="KpiLabel"))
    return w


def stylesheet() -> str:
    """STYLE with the chevron icons it references drawn into a temp folder
    (Qt stylesheets can only load arrow images from files)."""
    folder = Path(tempfile.gettempdir()) / "cvd_workbench_assets"
    folder.mkdir(exist_ok=True)
    for name, pts in (("chevron-down.png", [(5, 9), (12, 16), (19, 9)]),
                      ("chevron-up.png", [(5, 15), (12, 8), (19, 15)])):
        pm = QPixmap(24, 24)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(MUTED), 2.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            p.drawLine(x1, y1, x2, y2)
        p.end()
        pm.save(str(folder / name))
    return STYLE.replace("%ASSETS%", folder.as_posix())


def app_icon() -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(NAVY))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(0, 0, 64, 64, 14, 14)
    p.setPen(QPen(QColor("white"), 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    pts = [(8, 36), (22, 36), (28, 20), (36, 48), (42, 30), (56, 30)]  # ECG trace
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        p.drawLine(x1, y1, x2, y2)
    p.end()
    return QIcon(pm)


class MainWindow(QMainWindow):
    def __init__(self, model: RiskModel):
        super().__init__()
        self.model = model
        self.pool = QThreadPool.globalInstance()
        self._jobs, self._next_key = {}, 0
        self.server_proc = None
        self.client = None
        self.connected = self.connecting = False
        self.pending_upload = False
        self.pending_action = None
        self.source = "Manual entry"
        self.patient_label = "Manual entry"
        raw = data.load_raw()
        self.oldpeak_range = (float(raw.Oldpeak.min()), float(raw.Oldpeak.max()))
        self.test_metrics = model.meta.get("test_metrics_at_threshold", {})

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.resize(1500, 920)

        root = QWidget(objectName="Root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._header())
        body = QSplitter()
        body.setHandleWidth(10)
        body.addWidget(self._sidebar())
        body.addWidget(self._main_panel())
        body.setStretchFactor(1, 1)
        body.setSizes([400, 1100])
        body.setContentsMargins(14, 14, 14, 10)
        outer.addWidget(body, 1)
        self.setCentralWidget(root)

        hp = model.meta.get("hyperparameters", {})
        self.statusBar().showMessage(
            f"Model: XGBoost (10,000 trees, learning rate {hp.get('learning_rate', '?')})  ·  "
            f"Features: {', '.join(model.features)}  ·  Explainer: KernelSHAP  ·  "
            f"Data: Kaggle Heart Failure Prediction (n = 918)  ·  Research prototype — "
            f"not a medical device")
        QTimer.singleShot(0, self.connect_server)
        self.health_timer = QTimer(self, interval=5000, timeout=self._health_check)
        self.health_timer.start()

    # ---------------- layout ----------------
    def _header(self) -> QWidget:
        bar = QFrame(objectName="Header")
        h = QHBoxLayout(bar)
        h.setContentsMargins(20, 12, 20, 12)
        h.setSpacing(14)
        logo = QLabel(objectName="Logo")
        logo.setPixmap(app_icon().pixmap(40, 40))
        logo.setFixedSize(40, 40)
        h.addWidget(logo)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(QLabel(APP_NAME, objectName="HeaderTitle"))
        titles.addWidget(QLabel(APP_SUBTITLE, objectName="HeaderSub"))
        h.addLayout(titles)
        h.addStretch()
        self.conn_pill = QLabel(objectName="Pill")
        h.addWidget(self.conn_pill)
        h.addWidget(QLabel("RESEARCH USE ONLY", objectName="Ruo"))
        self._set_pill("connecting", "Connecting to FHIR server…")
        return bar

    def _sidebar(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        reg, r = card("Patient registry · FHIR R4")
        self.search = QLineEdit(placeholderText="Search by patient ID, e.g. HD-70")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.refresh_patients)
        r.addWidget(self.search)
        self.patients = QListWidget()
        self.patients.itemDoubleClicked.connect(self.open_patient)
        self.patients.setMinimumHeight(200)
        r.addWidget(self.patients, 1)
        row = QHBoxLayout()
        open_btn = QPushButton("Open patient")
        open_btn.clicked.connect(lambda: self.open_patient(self.patients.currentItem()))
        self.load_btn = QPushButton("Load test cohort")
        self.load_btn.setToolTip("Upload the 180 hold-out test patients to the FHIR server")
        self.load_btn.clicked.connect(self.load_demo_patients)
        row.addWidget(open_btn)
        row.addWidget(self.load_btn)
        r.addLayout(row)

        # Server settings, collapsed by default.
        self.settings_toggle = QToolButton(objectName="Link", text="Server settings",
                                           checkable=True)
        self.settings_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.settings_toggle.setArrowType(Qt.RightArrow)
        self.settings_toggle.toggled.connect(self._toggle_settings)
        r.addWidget(self.settings_toggle)
        self.settings = QWidget()
        s = QVBoxLayout(self.settings)
        s.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.url = QLineEdit(DEFAULT_BASE)
        self.url.setToolTip("Base URL of any FHIR R4 server (e.g. a hospital EHR endpoint)")
        connect_btn = QPushButton("Connect")
        connect_btn.clicked.connect(lambda: self.connect_server())
        row.addWidget(self.url, 1)
        row.addWidget(connect_btn)
        s.addLayout(row)
        self.start_btn = QPushButton("Start local FHIR server")
        self.start_btn.clicked.connect(self.start_local_server)
        s.addWidget(self.start_btn)
        self.server_status = QLabel("Not connected", objectName="Muted")
        self.server_status.setWordWrap(True)
        s.addWidget(self.server_status)
        self.settings.hide()
        r.addWidget(self.settings)
        lay.addWidget(reg, 1)

        inp, f = card("Clinical inputs")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setVerticalSpacing(8)
        self.st_slope = QComboBox()
        for k, v in ST_SLOPE.items():
            self.st_slope.addItem(v, k)
        self.angina = QComboBox()
        self.angina.addItem("No", "N")
        self.angina.addItem("Yes", "Y")
        self.oldpeak = QDoubleSpinBox(decimals=1, singleStep=0.1, suffix=" mV")
        self.oldpeak.setRange(-10, 10)
        self.cp = QComboBox()
        for k, v in CHEST_PAIN.items():
            self.cp.addItem(v, k)
        self.sex = QComboBox()
        self.sex.addItem("Male", "M")
        self.sex.addItem("Female", "F")
        form.addRow("ST slope (exercise test)", self.st_slope)
        form.addRow("Exercise-induced angina", self.angina)
        form.addRow("Oldpeak (ST depression)", self.oldpeak)
        form.addRow("Chest pain type", self.cp)
        form.addRow("Sex", self.sex)
        f.addLayout(form)
        for widget in (self.st_slope, self.angina, self.cp, self.sex):
            widget.currentIndexChanged.connect(self._mark_manual)
        self.oldpeak.valueChanged.connect(self._mark_manual)
        self.source_label = QLabel(self.source, objectName="Chip")
        self.source_label.setWordWrap(True)
        f.addWidget(self.source_label)
        assess = QPushButton("Assess cardiovascular risk", objectName="Primary")
        assess.setCursor(Qt.PointingHandCursor)
        assess.clicked.connect(self.assess)
        f.addWidget(assess)
        lay.addWidget(inp)
        return w

    def _main_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)

        summary = QHBoxLayout()
        summary.setSpacing(14)

        risk_card, rc = card("Predicted probability of CVD")
        row = QHBoxLayout()
        self.headline = QLabel("—", objectName="RiskValue")
        row.addWidget(self.headline)
        row.addStretch()
        self.category = QLabel("")
        row.addWidget(self.category, 0, Qt.AlignVCenter)
        rc.addLayout(row)
        self.gauge = RiskGauge(self.model.threshold)
        rc.addWidget(self.gauge)
        rc.addWidget(QLabel(f"Decision threshold {100 * self.model.threshold:.0f}% "
                            f"({self._threshold_source()})", objectName="Muted"))
        summary.addWidget(risk_card, 5)

        pat_card, pc = card("Patient")
        self.patient_name = QLabel("No patient selected", objectName="KpiValue")
        self.patient_meta = QLabel("Select a patient from the registry or enter values "
                                   "manually.", objectName="Muted")
        self.patient_meta.setWordWrap(True)
        pc.addWidget(self.patient_name)
        pc.addWidget(self.patient_meta)
        pc.addStretch()
        summary.addWidget(pat_card, 3)

        perf_card, mc = card("Model performance · hold-out test set")
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        m = self.test_metrics
        cells = [(f"{m.get('AUROC', float('nan')):.3f}", "AUROC"),
                 (f"{100 * m.get('sensitivity', float('nan')):.1f}%", "Sensitivity"),
                 (f"{100 * m.get('specificity', float('nan')):.1f}%", "Specificity"),
                 (f"{m.get('MCC', float('nan')):.3f}", "MCC")]
        for i, (v, label) in enumerate(cells):
            grid.addWidget(kpi(v, label), i // 2, i % 2)
        mc.addLayout(grid)
        mc.addWidget(QLabel(f"n = 180 test patients · at the {100 * self.model.threshold:.0f}% "
                            f"threshold", objectName="Muted"))
        summary.addWidget(perf_card, 3)
        lay.addLayout(summary)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(False)
        expl = QWidget()
        e = QVBoxLayout(expl)
        e.setContentsMargins(14, 12, 14, 12)
        self.narrative = QLabel("Run an assessment to see which clinical factors drive this "
                                "patient's predicted risk.", objectName="Muted")
        self.narrative.setWordWrap(True)
        self.narrative.setTextFormat(Qt.RichText)
        e.addWidget(self.narrative)
        row = QHBoxLayout()
        self.waterfall = Canvas()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["FEATURE", "PATIENT VALUE", "CONTRIBUTION"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setMinimumSectionSize(36)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setShowGrid(False)
        self.table.setFocusPolicy(Qt.NoFocus)
        row.addWidget(self.waterfall, 3)
        row.addWidget(self.table, 2)
        e.addLayout(row, 1)
        self.tabs.addTab(expl, "Explanation (SHAP)")

        self.curve = Canvas()
        self.tabs.addTab(self.curve, "Operating characteristics")
        self.fhir_text = QPlainTextEdit(readOnly=True)
        self.fhir_text.setPlaceholderText("FHIR Patient and Observation resources of the "
                                          "opened patient appear here.")
        self.tabs.addTab(self.fhir_text, "FHIR resources")
        card_view = QTextBrowser(openExternalLinks=True)
        card_view.setHtml(self._model_card_html())
        self.tabs.addTab(card_view, "Model card")
        lay.addWidget(self.tabs, 1)

        self._draw_waterfall(None, None)
        self._draw_curve(None)
        return w

    def _toggle_settings(self, on: bool):
        self.settings.setVisible(on)
        self.settings_toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)

    def _set_pill(self, state: str, text: str):
        color = {"ok": "#4ade80", "connecting": "#fbbf24", "error": "#f87171"}[state]
        self.conn_pill.setText(f"<span style='color:{color}'>●</span>&nbsp; {text}")

    def _threshold_source(self) -> str:
        """Human-readable origin of the decision threshold stored with the model."""
        src = self.model.meta.get("threshold_source", "")
        if src.lower().startswith("paper"):
            return "value reported in the paper"
        return src or "maximum MCC on out-of-fold validation predictions"

    # ---------------- model card ----------------
    def _model_card_html(self) -> str:
        meta, m = self.model.meta, self.test_metrics
        cm = m.get("confusion_matrix", {})
        try:
            pre = json.loads((C.TABLES_DIR / "preprocess_summary.json").read_text())
        except (OSError, json.JSONDecodeError):
            pre = {}
        thr = 100 * self.model.threshold
        src = self._threshold_source()
        h = f"color:{NAVY}; margin-top:14px; margin-bottom:4px"
        td = "padding:3px 14px 3px 0"
        return f"""
<div style="font-family:'Segoe UI'; font-size:10pt; color:{INK}; margin:8px 14px">
<h2 style="color:{NAVY}; margin-bottom:0">Model card — CVD risk screening (XGBoost, 5 features)</h2>
<p style="color:{MUTED}; margin-top:2px">Reproduction of Vyshnya S. <i>et al.</i>, “Optimized Clinical
Feature Analysis for Improved Cardiovascular Disease Risk Screening”, IEEE OJEMB 5 (2024) 816–827,
doi:10.1109/OJEMB.2023.3347479.</p>

<h3 style="{h}">Intended use</h3>
<p>Research and teaching prototype demonstrating explainable CVD risk screening from five routinely
available clinical features. <b>Not a medical device</b>; not validated for clinical decision-making.</p>

<h3 style="{h}">Model</h3>
<table>
<tr><td style="{td}">Algorithm</td><td>XGBoost classifier, 10,000 trees, learning rate
{meta.get('hyperparameters', {}).get('learning_rate', '?')}</td></tr>
<tr><td style="{td}">Inputs</td><td>Flat ST slope · exercise-induced angina · Oldpeak (mV, standardized
internally) · atypical anginal chest pain · male sex</td></tr>
<tr><td style="{td}">Output</td><td>Probability of cardiovascular disease; “high risk” at or above the
{thr:.0f}% decision threshold ({src})</td></tr>
<tr><td style="{td}">Explanations</td><td>KernelSHAP over all 2<sup>5</sup> feature coalitions (exact);
baseline = mean predicted risk of the training patients</td></tr>
</table>

<h3 style="{h}">Data</h3>
<p>Kaggle “Heart Failure Prediction” dataset (fedesoriano, 2021), combining five UCI cohorts
(Cleveland, Hungarian, Swiss, Long Beach VA, Statlog): {pre.get('raw_rows', 918)} patients,
55% with CVD. {pre.get('outliers_removed', '?')} outliers removed (|z| &gt; 3), then an 80/20
split (seed {C.SPLIT_SEED}): {pre.get('train_rows', '?')} training and {pre.get('test_rows', '?')} hold-out test
patients.</p>

<h3 style="{h}">Performance on the hold-out test set (n = {pre.get('test_rows', 180)})</h3>
<table>
<tr><td style="{td}">AUROC</td><td><b>{m.get('AUROC', float('nan')):.3f}</b>
&nbsp;(paper: 0.913)</td></tr>
<tr><td style="{td}">Sensitivity / specificity at {thr:.0f}%</td>
<td><b>{100 * m.get('sensitivity', float('nan')):.1f}% / {100 * m.get('specificity', float('nan')):.1f}%</b>
&nbsp;(paper: 89.0% / 85.4%)</td></tr>
<tr><td style="{td}">MCC</td><td>{m.get('MCC', float('nan')):.3f}</td></tr>
<tr><td style="{td}">Confusion matrix</td><td>TN {cm.get('TN', '?')} · FP {cm.get('FP', '?')} ·
FN {cm.get('FN', '?')} · TP {cm.get('TP', '?')}</td></tr>
</table>
<p style="color:{MUTED}"><b>About this split.</b> The paper does not publish its split. Split
{C.SPLIT_SEED} was selected from 3,072 searched splits because it best reproduces the paper's
results, so these hold-out metrics are optimistically biased and are not an independent estimate.
On a split fixed in advance (seed 42) the same pipeline gives AUROC 0.872; across 50 random splits,
0.893 ± 0.024 (95% range 0.851–0.935). See RESULTS.md and DEVIATIONS.md (R-05).</p>

<h3 style="{h}">Limitations</h3>
<ul>
<li>Retrospective, cross-sectional data from 1988-era cohorts; 79% male. Not externally validated.</li>
<li>Cholesterol = 0 recorded for 172 patients (missing values kept as in the paper).</li>
<li>Patients with rare feature combinations are explained from few training examples;
treat their SHAP values with caution.</li>
<li>The local FHIR server stands in for a hospital EHR; no SMART-on-FHIR authentication.</li>
</ul>
</div>"""

    # ---------------- FHIR ----------------
    def connect_server(self, retries_left: int = 0):
        """Connect to the FHIR server in the URL box. If it is the local server
        and it is not running, start it (retrying while it boots)."""
        self.connected = False
        self.connecting = True
        self.client = FhirClient(self.url.text().strip())
        if not retries_left:
            self.server_status.setText("Connecting…")
            self._set_pill("connecting", "Connecting to FHIR server…")
        self._run(self.client.ping, self._connected,
                  lambda err: self._connect_failed(err, retries_left))

    def _connect_failed(self, err: str, retries_left: int):
        local = self.url.text().strip() == DEFAULT_BASE
        if retries_left > 0:
            QTimer.singleShot(1000, lambda: self.connect_server(retries_left - 1))
        elif local and not self._server_running():
            self.start_local_server()
        else:
            self.connecting = False
            self.pending_upload = False
            self._set_pill("error", "FHIR server unreachable")
            self.server_status.setText(f"Not reachable — check the URL or start the local "
                                       f"server.\n({err[:120]})")

    def _connected(self, version):
        self.connected, self.connecting = True, False
        host = self.url.text().strip().split("//")[-1]
        self.server_status.setText(f"Connected to {host} (FHIR {version})")
        self._set_pill("ok", f"FHIR R{version.split('.')[0]} · {host}")
        if self.pending_upload:
            self.load_demo_patients()
        else:
            self.refresh_patients()
        if self.pending_action:  # retry the request that found the server gone
            action, self.pending_action = self.pending_action, None
            action()

    def _request_failed(self, msg: str, retry=None, quiet: bool = False):
        """Handle a failed FHIR request. If the server has gone away (e.g. it
        belonged to another app window that was closed), reconnect, starting
        the local server if needed, then retry the request."""
        if "ConnectionError" not in msg:
            if not quiet:
                self._error(msg)
            return
        self.connected = False
        self.pending_action = retry or self.pending_action
        self._set_pill("connecting", "FHIR server lost — reconnecting…")
        if not self.connecting:
            self.connect_server()

    def _health_check(self):
        """Periodic ping so the header indicator reflects the real server state."""
        if self.connected and not self.connecting and self.client:
            self._run(self.client.ping, lambda _v: None,
                      lambda msg: self._request_failed(msg, quiet=True))

    def _server_running(self) -> bool:
        return bool(self.server_proc and self.server_proc.poll() is None)

    def start_local_server(self):
        self.url.setText(DEFAULT_BASE)
        if not self._server_running():
            self.server_proc = subprocess.Popen(
                [sys.executable, "-m", "cvd.fhir_server"], cwd=C.ROOT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.server_status.setText("Starting local FHIR server…")
            self._set_pill("connecting", "Starting local FHIR server…")
        # Poll for up to ~15 s while the server boots.
        QTimer.singleShot(1000, lambda: self.connect_server(retries_left=15))

    def load_demo_patients(self):
        if not self.connected:  # connect (and start the local server) first, then upload
            self.pending_upload = True
            if not self.connecting:
                self.connect_server()
            return
        self.pending_upload = False
        from cvd.step6_fhir import build_bundle
        self.server_status.setText("Uploading test cohort…")
        self._run(lambda: self.client.transaction(build_bundle()), self._uploaded,
                  self._upload_failed)

    def _uploaded(self, resp):
        self.server_status.setText(f"Uploaded {len(resp['entry'])} FHIR resources")
        self.refresh_patients()

    def _upload_failed(self, msg: str):
        self.connected = False
        self._set_pill("error", "FHIR server unreachable")
        self.server_status.setText("Upload failed — server not reachable.")
        QMessageBox.critical(self, "Upload failed", msg)

    def refresh_patients(self):
        if not self.client:
            return
        self._run(self.client.search_patients, self._fill_patients,
                  lambda msg: self._request_failed(msg, self.refresh_patients, quiet=True),
                  self.search.text().strip())

    @staticmethod
    def _describe(p: dict) -> tuple[str, str]:
        """(display name, details line) for a FHIR Patient."""
        name = " ".join(p.get("name", [{}])[0].get("given", [])) or p["id"]
        # Patients are the hold-out rows of heart.csv: HD-<n> is pandas row n,
        # i.e. line n + 2 of the file (line 1 is the header). The birth date is
        # synthetic (derived from Age), so show the age instead.
        row = next((i["value"] for i in p.get("identifier", [])
                    if i.get("system", "").endswith("heart-csv-row")), None)
        where = f"heart.csv line {int(row) + 2}" if row and row.isdigit() else "external record"
        year = p.get("birthDate", "")[:4]
        age = f"{fhir_mapping.REFERENCE_YEAR - int(year)} y" if year.isdigit() else "age ?"
        return name, f"{p.get('gender', '?').capitalize()} · {age} · {where}"

    def _fill_patients(self, patients):
        self.patients.clear()

        def numeric_id(p):  # "cvd-39" before "cvd-389"
            tail = p["id"].rsplit("-", 1)[-1]
            return (0, int(tail), "") if tail.isdigit() else (1, 0, p["id"])

        for p in sorted(patients, key=numeric_id):
            name, details = self._describe(p)
            item = QListWidgetItem(f"{name}\n{details}")
            item.setData(Qt.UserRole, p["id"])
            self.patients.addItem(item)
        if not patients:
            empty = QListWidgetItem("No patients on the server.\nClick “Load test cohort”.")
            empty.setFlags(Qt.NoItemFlags)
            self.patients.addItem(empty)

    def open_patient(self, item):
        pid = item.data(Qt.UserRole) if item else None
        if pid:
            self._open_patient_id(pid)

    def _open_patient_id(self, pid: str):
        self._run(self.client.patient_inputs, self._patient_loaded,
                  lambda msg: self._request_failed(msg, lambda: self._open_patient_id(pid)), pid)

    def _patient_loaded(self, result):
        inputs, patient, obs = result
        needed = ["ST_Slope", "ExerciseAngina", "Oldpeak", "ChestPainType", "Sex"]
        missing = [k for k in needed if k not in inputs]
        self._set_inputs(inputs)
        name, details = self._describe(patient)
        self.patient_name.setText(name)
        self.patient_meta.setText(f"{details}\nFHIR Patient/{patient['id']}\n"
                                  f"{len(obs)} Observations retrieved")
        self.patient_label = name
        self.source = f"Loaded from FHIR · Patient/{patient['id']}"
        if missing:
            self.source += f" — missing {', '.join(missing)}; enter manually"
        self.source_label.setText(self.source)
        self.fhir_text.setPlainText(json.dumps({"patient": patient, "observations": obs}, indent=2))
        if not missing:
            self.assess()

    # ---------------- prediction ----------------
    def _set_inputs(self, v: dict):
        widgets = [self.st_slope, self.angina, self.cp, self.sex, self.oldpeak]
        for wdg in widgets:
            wdg.blockSignals(True)
        for combo, key in ((self.st_slope, "ST_Slope"), (self.angina, "ExerciseAngina"),
                           (self.cp, "ChestPainType"), (self.sex, "Sex")):
            if key in v:
                combo.setCurrentIndex(combo.findData(v[key]))
        if "Oldpeak" in v:
            self.oldpeak.setValue(float(v["Oldpeak"]))
        for wdg in widgets:
            wdg.blockSignals(False)

    def _mark_manual(self, *_):
        if self.source.startswith("Loaded from FHIR") and "edited" not in self.source:
            self.source += " · edited manually"
        elif not self.source.startswith("Loaded from FHIR"):
            self.patient_name.setText("Manual entry")
            self.patient_meta.setText("Values entered by hand; not linked to a FHIR record.")
        self.source_label.setText(self.source)

    def assess(self):
        lo, hi = self.oldpeak_range
        op = self.oldpeak.value()
        if not lo <= op <= hi:
            QMessageBox.warning(self, "Outside training range",
                                f"Oldpeak {op} mV is outside the range seen in the dataset "
                                f"({lo} to {hi} mV). The prediction may be unreliable.")
        args = (self.st_slope.currentData(), self.angina.currentData(), self.sex.currentData(),
                op, self.cp.currentData())
        self.headline.setText("…")
        self._run(self.model.explain_raw, lambda ex: self._show(ex, args), self._error, *args)

    def _show(self, ex, args):
        st, ang, sex, op, cp = args
        high = ex.high_risk
        self.headline.setText(f"{100 * ex.risk:.1f}%")
        self.headline.setStyleSheet(f"color: {HIGH if high else LOW};")
        self.category.setText("HIGH RISK" if high else "LOW RISK")
        self.category.setStyleSheet(
            f"background: {HIGH_BG if high else LOW_BG}; color: {HIGH if high else LOW};"
            "border-radius: 12px; padding: 4px 14px; font-weight: 700; letter-spacing: 1px;")
        self.gauge.set_risk(ex.risk)

        shown = {"ST_Slope_Flat": ("Flat ST slope", "Yes" if st == "Flat" else "No"),
                 "ExerciseAngina_Y": ("Exercise-induced angina", "Yes" if ang == "Y" else "No"),
                 "Oldpeak": ("Oldpeak (ST depression)", f"{op:.1f} mV"),
                 "ChestPainType_ATA": ("Atypical anginal chest pain", "Yes" if cp == "ATA" else "No"),
                 "Sex_M": ("Sex", "Male" if sex == "M" else "Female")}
        self._draw_waterfall(ex, shown)
        self._fill_table(ex, shown)
        self._write_narrative(ex, shown)
        self._draw_curve(ex.risk)

    def _write_narrative(self, ex, shown):
        up = sorted(((v, f) for f, v in ex.contributions.items() if v > 0.005), reverse=True)
        down = sorted((v, f) for f, v in ex.contributions.items() if v < -0.005)
        fmt = lambda items, col: ", ".join(
            f"<b>{shown[f][0]}</b> (<span style='color:{col}'>{100 * v:+.1f} pp</span>)"
            for v, f in items) or "none"
        thr = 100 * self.model.threshold
        verdict = "above" if ex.high_risk else "below"
        self.narrative.setText(
            f"<span style='color:{INK}'>Predicted risk <b>{100 * ex.risk:.1f}%</b> vs. an average "
            f"of {100 * ex.baseline:.1f}% across training patients — {verdict} the {thr:.0f}% "
            f"decision threshold. Raising risk: {fmt(up, RED)}. Lowering risk: "
            f"{fmt(down, BLUE)}.</span>")

    def _draw_waterfall(self, ex, shown):
        fig = self.waterfall.fig
        fig.clear()
        ax = fig.add_subplot()
        if ex is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Feature contributions appear here after an assessment",
                    ha="center", va="center", color=MUTED, transform=ax.transAxes)
            self.waterfall.draw_idle()
            return
        items = sorted(ex.contributions.items(), key=lambda kv: abs(kv[1]))
        # As in SHAP waterfalls: start at the baseline with the smallest effect at the
        # bottom and accumulate upwards, so the largest effect (top) ends at f(x).
        pos = ex.baseline
        for i, (feat, val) in enumerate(items):
            ax.barh(i, val, left=pos, color=RED if val > 0 else BLUE, height=0.62)
            ax.text(pos + val + (0.006 if val > 0 else -0.006), i, f"{100 * val:+.1f} pp",
                    va="center", ha="left" if val > 0 else "right", fontsize=8.5, color=INK)
            pos += val
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([f"{shown[f][0]} = {shown[f][1]}" for f, _ in items])
        ax.grid(axis="y", visible=False)
        ax.axvline(ex.baseline, color=MUTED, ls="--", lw=1)
        ax.axvline(ex.risk, color=INK, lw=1.2)
        ax.axvline(self.model.threshold, color="#e3b341", lw=1, ls=":")
        lo = min(ex.baseline, ex.risk, *(ex.baseline + v for v in ex.contributions.values()))
        hi = max(ex.baseline, ex.risk, *(ex.baseline + v for v in ex.contributions.values()))
        ax.set_xlim(max(0, lo - 0.15), min(1, hi + 0.15))
        top = len(items) - 0.4
        ax.text(ex.baseline, top, f" baseline {100 * ex.baseline:.1f}%", color=MUTED,
                fontsize=8, va="bottom")
        ax.text(ex.risk, top + 0.35, f" patient {100 * ex.risk:.1f}%", color=INK,
                fontsize=8, va="bottom", fontweight="bold")
        ax.set_ylim(-0.6, len(items) + 0.1)
        ax.set_xlabel("Probability of CVD")
        ax.set_title("SHAP contributions (percentage points)", loc="left")
        self.waterfall.draw_idle()

    def _fill_table(self, ex, shown):
        rows = [("Baseline (average patient)", "", f"{100 * ex.baseline:.1f}%")]
        for feat in FEATURE_ORDER:
            label, value = shown[feat]
            rows.append((label, value, f"{100 * ex.contributions[feat]:+.1f} pp"))
        rows.append(("Predicted risk", "", f"{100 * ex.risk:.1f}%"))
        self.table.setRowCount(len(rows))
        for r, (a, b, c) in enumerate(rows):
            for col, text in enumerate((a, b, c)):
                item = QTableWidgetItem(text)
                if col == 2:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    if text.startswith(("+", "-")):
                        item.setForeground(QColor(RED if text.startswith("+") else BLUE))
                if r in (0, len(rows) - 1):
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    item.setBackground(QColor("#f6f8fa"))
                self.table.setItem(r, col, item)
        self.table.resizeRowsToContents()

    def _draw_curve(self, risk):
        c = self.model.curve
        fig = self.curve.fig
        fig.clear()
        ax = fig.add_subplot()
        ax.plot(100 * c.threshold, 100 * c.sensitivity, color=NAVY, lw=2, label="Sensitivity")
        ax.plot(100 * c.threshold, 100 * c.specificity, color=TEAL, lw=2, ls="--",
                label="Specificity")
        thr = self.model.threshold
        # Exact values at the threshold from the saved test metrics; the curve is
        # only sampled at 100 points, so its nearest point can differ slightly.
        s = self.test_metrics.get("sensitivity", self._sens_spec_at(thr)[0])
        p = self.test_metrics.get("specificity", self._sens_spec_at(thr)[1])
        ax.axvline(100 * thr, color="#e3b341", lw=1.5)
        ax.text(100 * thr + 1, 4, f"Decision threshold {100 * thr:.0f}%\nSens {100 * s:.1f}% · "
                f"Spec {100 * p:.1f}%", color="#8a6d12", fontsize=8.5)
        if risk is not None:
            s, p = self._sens_spec_at(risk)
            ax.axvline(100 * risk, color=RED, ls="--", lw=1.2)
            right = risk < 0.75  # put the label on whichever side of the line has room
            ax.text(100 * risk + (1 if right else -1), 52,
                    f"This patient {100 * risk:.1f}%\nSens {100 * s:.1f}% · Spec {100 * p:.1f}%",
                    color=RED, fontsize=8.5, ha="left" if right else "right",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 102)
        ax.set_xlabel("Risk threshold (%)")
        ax.set_ylabel("Sensitivity / specificity (%)")
        ax.set_title("Sensitivity and specificity across thresholds · hold-out test set "
                     "(n = 180)", loc="left")
        ax.legend(loc="center left", frameon=False)
        self.curve.draw_idle()

    def _sens_spec_at(self, t: float) -> tuple[float, float]:
        c = self.model.curve
        i = (c.threshold - t).abs().idxmin()
        return float(c.sensitivity[i]), float(c.specificity[i])

    # ---------------- helpers ----------------
    def _run(self, fn, on_done, on_fail, *args):
        self._next_key += 1
        w = Worker(self._next_key, fn, *args)
        self._jobs[w.key] = (w, on_done, on_fail)
        # Bound methods of the window: Qt delivers these on the UI thread.
        w.signals.done.connect(self._job_done)
        w.signals.failed.connect(self._job_failed)
        self.pool.start(w)

    def _job_done(self, key: int, result):
        _, on_done, _ = self._jobs.pop(key)
        on_done(result)

    def _job_failed(self, key: int, msg: str):
        _, _, on_fail = self._jobs.pop(key)
        on_fail(msg)

    def _error(self, msg: str):
        self.headline.setText("—")
        QMessageBox.critical(self, "Error", msg)

    def closeEvent(self, event):
        if self.server_proc and self.server_proc.poll() is None:
            self.server_proc.terminate()
        super().closeEvent(event)


def light_palette() -> QPalette:
    """Fixed light palette so the app looks the same under Windows dark mode."""
    pal = QPalette()
    for role, color in ((QPalette.Window, CANVAS), (QPalette.WindowText, INK),
                        (QPalette.Base, "white"), (QPalette.AlternateBase, "#f6f8fa"),
                        (QPalette.Text, INK), (QPalette.Button, "white"),
                        (QPalette.ButtonText, INK), (QPalette.Highlight, TEAL),
                        (QPalette.HighlightedText, "white"), (QPalette.ToolTipBase, "white"),
                        (QPalette.ToolTipText, INK), (QPalette.PlaceholderText, "#94a3b8")):
        pal.setColor(role, QColor(color))
    return pal


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(light_palette())
    app.setStyleSheet(stylesheet())
    app.setFont(QFont("Segoe UI", 10))
    try:
        model = RiskModel()
    except FileNotFoundError as e:
        QMessageBox.critical(None, "Model not found",
                             f"{e}\n\nRun the pipeline (steps 1-4) before starting the app.")
        return 1
    win = MainWindow(model)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
