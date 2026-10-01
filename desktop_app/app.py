"""CVD risk screening desktop application (desktop version of the paper's
Fig. 5 web application; DEVIATIONS.md P-02).

* Reads a patient's five model inputs from a FHIR R4 server (the EHR stand-in),
  or lets the clinician enter / edit them manually.
* Shows the predicted CVD risk, the patient-specific SHAP contributions
  (Fig. 4b/c) and the sensitivity / specificity curve (Fig. 3c / Fig. 5).

Run from the project root:
    .venv\\Scripts\\python -m desktop_app.app
"""
import json
import subprocess
import sys

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPlainTextEdit, QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from cvd import config as C
from cvd import data
from cvd.fhir_client import DEFAULT_BASE, FhirClient
from cvd.predict import RiskModel

CHEST_PAIN = {"ASY": "Asymptomatic (ASY)", "ATA": "Atypical angina (ATA)",
              "NAP": "Non-anginal pain (NAP)", "TA": "Typical angina (TA)"}
ST_SLOPE = {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"}
RED, BLUE = "#e8175d", "#1e88e5"


class Worker(QRunnable):
    """Runs `fn` off the UI thread and reports back through signals."""

    class Signals(QObject):
        done = Signal(object)
        failed = Signal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args, self.signals = fn, args, Worker.Signals()

    def run(self):
        try:
            self.signals.done.emit(self.fn(*self.args))
        except Exception as e:  # surfaced to the user in a message box
            self.signals.failed.emit(f"{type(e).__name__}: {e}")


class Canvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(6, 4), tight_layout=True)
        super().__init__(self.fig)


class MainWindow(QMainWindow):
    def __init__(self, model: RiskModel):
        super().__init__()
        self.model = model
        self.pool = QThreadPool.globalInstance()
        self.server_proc = None
        self.client = None
        self.source = "Entered manually"
        raw = data.load_raw()
        self.oldpeak_range = (float(raw.Oldpeak.min()), float(raw.Oldpeak.max()))

        self.setWindowTitle("CVD Risk Screening - Explainable XGBoost (5 features)")
        self.resize(1400, 860)
        splitter = QSplitter()
        splitter.addWidget(self._left_panel())
        splitter.addWidget(self._right_panel())
        splitter.setSizes([430, 970])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage(
            f"Model: XGBoost on {', '.join(model.features)} | decision threshold "
            f"{100 * model.threshold:.0f}%  —  research prototype, not for clinical use")
        QTimer.singleShot(0, self.connect_server)

    # ---------------- layout ----------------
    def _left_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        ehr = QGroupBox("EHR — FHIR R4 server")
        e = QVBoxLayout(ehr)
        row = QHBoxLayout()
        self.url = QLineEdit(DEFAULT_BASE)
        btn = QPushButton("Connect")
        btn.clicked.connect(self.connect_server)
        row.addWidget(self.url)
        row.addWidget(btn)
        e.addLayout(row)
        row = QHBoxLayout()
        self.start_btn = QPushButton("Start local server")
        self.start_btn.clicked.connect(self.start_local_server)
        self.load_btn = QPushButton("Load test patients")
        self.load_btn.setToolTip("Upload the 20% hold-out patients to the server as FHIR resources")
        self.load_btn.clicked.connect(self.load_demo_patients)
        row.addWidget(self.start_btn)
        row.addWidget(self.load_btn)
        e.addLayout(row)
        self.server_status = QLabel("Not connected")
        e.addWidget(self.server_status)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search patients (name or id)")
        self.search.textChanged.connect(self.refresh_patients)
        e.addWidget(self.search)
        self.patients = QListWidget()
        self.patients.itemDoubleClicked.connect(self.open_patient)
        e.addWidget(self.patients, 1)
        open_btn = QPushButton("Open selected patient")
        open_btn.clicked.connect(lambda: self.open_patient(self.patients.currentItem()))
        e.addWidget(open_btn)
        lay.addWidget(ehr, 1)

        form_box = QGroupBox("Patient features")
        form = QFormLayout(form_box)
        self.st_slope = QComboBox()
        for k, v in ST_SLOPE.items():
            self.st_slope.addItem(v, k)
        self.angina = QComboBox()
        self.angina.addItem("No", "N")
        self.angina.addItem("Yes", "Y")
        self.oldpeak = QDoubleSpinBox()
        self.oldpeak.setRange(-10, 10)
        self.oldpeak.setDecimals(1)
        self.oldpeak.setSingleStep(0.1)
        self.oldpeak.setSuffix(" mV")
        self.cp = QComboBox()
        for k, v in CHEST_PAIN.items():
            self.cp.addItem(v, k)
        self.sex = QComboBox()
        self.sex.addItem("Male", "M")
        self.sex.addItem("Female", "F")
        form.addRow("ST slope (exercise test):", self.st_slope)
        form.addRow("Exercise-induced angina:", self.angina)
        form.addRow("Oldpeak (ST depression):", self.oldpeak)
        form.addRow("Chest pain type:", self.cp)
        form.addRow("Sex:", self.sex)
        for widget in (self.st_slope, self.angina, self.cp, self.sex):
            widget.currentIndexChanged.connect(self._mark_manual)
        self.oldpeak.valueChanged.connect(self._mark_manual)
        self.source_label = QLabel(self.source)
        self.source_label.setWordWrap(True)
        form.addRow("Source:", self.source_label)
        assess = QPushButton("Assess CVD risk")
        assess.setMinimumHeight(38)
        assess.clicked.connect(self.assess)
        form.addRow(assess)
        lay.addWidget(form_box)
        return w

    def _right_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.headline = QLabel("Cardiovascular Risk Assessment: —")
        f = QFont()
        f.setPointSize(18)
        f.setBold(True)
        self.headline.setFont(f)
        self.category = QLabel("")
        self.category.setFont(QFont(f.family(), 12))
        lay.addWidget(self.headline)
        lay.addWidget(self.category)

        self.tabs = QTabWidget()
        contrib = QWidget()
        c = QHBoxLayout(contrib)
        self.waterfall = Canvas()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Feature", "Value", "Effect on risk"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        c.addWidget(self.waterfall, 3)
        c.addWidget(self.table, 2)
        self.tabs.addTab(contrib, "Feature contributions (SHAP)")
        self.curve = Canvas()
        self.tabs.addTab(self.curve, "Sensitivity / specificity")
        self.fhir_text = QPlainTextEdit()
        self.fhir_text.setReadOnly(True)
        self.tabs.addTab(self.fhir_text, "FHIR record")
        lay.addWidget(self.tabs, 1)
        self._draw_curve(None)
        return w

    # ---------------- FHIR ----------------
    def connect_server(self):
        self.client = FhirClient(self.url.text().strip())
        self.server_status.setText("Connecting…")
        self._run(self.client.ping, self._connected,
                  lambda err: self.server_status.setText(f"Not reachable — start the local server "
                                                         f"or check the URL.\n({err[:120]})"))

    def _connected(self, version):
        self.server_status.setText(f"Connected — FHIR {version}")
        self.refresh_patients()

    def start_local_server(self):
        if self.server_proc and self.server_proc.poll() is None:
            self.connect_server()
            return
        self.server_proc = subprocess.Popen([sys.executable, "-m", "cvd.fhir_server"], cwd=C.ROOT,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.url.setText(DEFAULT_BASE)
        self.server_status.setText("Starting local server…")
        QTimer.singleShot(2500, self.connect_server)

    def load_demo_patients(self):
        if not self.client:
            return
        from cvd.step6_fhir import build_bundle
        self.server_status.setText("Uploading test patients…")
        self._run(lambda: self.client.transaction(build_bundle()),
                  lambda resp: (self.server_status.setText(
                      f"Uploaded {len(resp['entry'])} FHIR resources"), self.refresh_patients()),
                  self._error)

    def refresh_patients(self):
        if not self.client:
            return
        self._run(self.client.search_patients, self._fill_patients, lambda _e: None,
                  self.search.text().strip())

    def _fill_patients(self, patients):
        self.patients.clear()

        def numeric_id(p):  # "cvd-39" before "cvd-389"
            tail = p["id"].rsplit("-", 1)[-1]
            return (0, int(tail), "") if tail.isdigit() else (1, 0, p["id"])

        for p in sorted(patients, key=numeric_id):
            name = " ".join(p.get("name", [{}])[0].get("given", []))
            item = QListWidgetItem(f"{name}   ({p.get('gender', '?')}, born {p.get('birthDate', '?')})")
            item.setData(Qt.UserRole, p["id"])
            self.patients.addItem(item)
        if not patients:
            self.patients.addItem("No patients on server — click “Load test patients”.")

    def open_patient(self, item):
        pid = item.data(Qt.UserRole) if item else None
        if not pid:
            return
        self._run(self.client.patient_inputs, self._patient_loaded, self._error, pid)

    def _patient_loaded(self, result):
        inputs, patient, obs = result
        needed = ["ST_Slope", "ExerciseAngina", "Oldpeak", "ChestPainType", "Sex"]
        missing = [k for k in needed if k not in inputs]
        self._set_inputs(inputs)
        self.source = f"FHIR Patient/{patient['id']}"
        if missing:
            self.source += f" — missing: {', '.join(missing)} (please enter manually)"
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
        if self.source.startswith("FHIR") and "edited" not in self.source:
            self.source += " (edited manually)"
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
        self.headline.setText("Cardiovascular Risk Assessment: computing…")
        self._run(self.model.explain_raw, lambda ex: self._show(ex, args), self._error, *args)

    def _show(self, ex, args):
        st, ang, sex, op, cp = args
        self.headline.setText(f"Cardiovascular Risk Assessment: {100 * ex.risk:.1f}%")
        thr = 100 * self.model.threshold
        if ex.high_risk:
            self.category.setText(f"HIGH RISK — at or above the {thr:.0f}% decision threshold")
            self.category.setStyleSheet(f"color: {RED};")
        else:
            self.category.setText(f"Low risk — below the {thr:.0f}% decision threshold")
            self.category.setStyleSheet(f"color: {BLUE};")

        shown = {"ST_Slope_Flat": ("Flat ST slope?", "Y" if st == "Flat" else "N"),
                 "ExerciseAngina_Y": ("Exercise angina?", ang),
                 "Oldpeak": ("Oldpeak", f"{op:.1f} mV"),
                 "ChestPainType_ATA": ("Atypical anginal chest pain?", "Y" if cp == "ATA" else "N"),
                 "Sex_M": ("Sex", sex)}
        self._draw_waterfall(ex, shown)
        self._fill_table(ex, shown)
        self._draw_curve(ex.risk)

    def _draw_waterfall(self, ex, shown):
        fig = self.waterfall.fig
        fig.clear()
        ax = fig.add_subplot()
        items = sorted(ex.contributions.items(), key=lambda kv: abs(kv[1]))
        # As in SHAP waterfalls: start at the baseline with the smallest effect at the
        # bottom and accumulate upwards, so the largest effect (top) ends at f(x).
        pos = ex.baseline
        for i, (feat, val) in enumerate(items):
            ax.barh(i, val, left=pos, color=RED if val > 0 else BLUE)
            ax.text(pos + val + (0.005 if val > 0 else -0.005), i, f"{100 * val:+.1f}%",
                    va="center", ha="left" if val > 0 else "right", fontsize=9)
            pos += val
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([f"{shown[f][0]} {shown[f][1]}" for f, _ in items])
        ax.axvline(ex.baseline, color="grey", ls="--", lw=1)
        ax.axvline(ex.risk, color="black", lw=1)
        ax.set_xlabel("Probability of CVD")
        ax.set_title(f"Baseline E[f(x)] = {ex.baseline:.3f}  →  patient f(x) = {ex.risk:.3f}",
                     fontsize=10)
        lo = min(ex.baseline, ex.risk, *(ex.baseline + v for v in ex.contributions.values()))
        hi = max(ex.baseline, ex.risk, *(ex.baseline + v for v in ex.contributions.values()))
        ax.set_xlim(max(0, lo - 0.15), min(1, hi + 0.15))
        self.waterfall.draw_idle()

    def _fill_table(self, ex, shown):
        rows = [("Baseline CVD risk", "", f"{100 * ex.baseline:.1f}%")]
        for feat in ["ST_Slope_Flat", "ExerciseAngina_Y", "Oldpeak", "ChestPainType_ATA", "Sex_M"]:
            label, value = shown[feat]
            rows.append((label, value, f"{100 * ex.contributions[feat]:+.1f}%"))
        rows.append(("Patient's CVD risk", "", f"{100 * ex.risk:.1f}%"))
        self.table.setRowCount(len(rows))
        for r, (a, b, c) in enumerate(rows):
            for col, text in enumerate((a, b, c)):
                item = QTableWidgetItem(text)
                if col == 2 and text.startswith(("+", "-")):
                    item.setForeground(Qt.red if text.startswith("+") else Qt.blue)
                if r in (0, len(rows) - 1):
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(r, col, item)

    def _draw_curve(self, risk):
        c = self.model.curve
        fig = self.curve.fig
        fig.clear()
        ax = fig.add_subplot()
        ax.plot(100 * c.threshold, 100 * c.sensitivity, color="#1b5e63", label="Sensitivity (Sen)")
        ax.plot(100 * c.threshold, 100 * c.specificity, color="#b5a93a", label="Specificity (Spe)")
        thr = self.model.threshold
        s, p = self._sens_spec_at(thr)
        ax.axvline(100 * thr, color="green")
        ax.text(100 * thr + 1, 5, f"Decision threshold {100 * thr:.0f}%\nSen = {100 * s:.1f}%\n"
                f"Spe = {100 * p:.1f}%", color="green", fontsize=8)
        if risk is not None:
            s, p = self._sens_spec_at(risk)
            ax.axvline(100 * risk, color=RED, ls="--")
            # Put the label on whichever side of the line has room.
            right = risk < 0.75
            ax.text(100 * risk + (1 if right else -1), 55,
                    f"This patient {100 * risk:.1f}%\nSen = {100 * s:.1f}%\nSpe = {100 * p:.1f}%",
                    color=RED, fontsize=8, ha="left" if right else "right")
        ax.set_xlabel("Risk of CVD (%)")
        ax.set_ylabel("Sensitivity or Specificity (%)")
        ax.set_title("Sensitivity and Specificity for CVD Risk Score (hold-out test set)")
        ax.legend(loc="center left")
        self.curve.draw_idle()

    def _sens_spec_at(self, t: float) -> tuple[float, float]:
        c = self.model.curve
        i = (c.threshold - t).abs().idxmin()
        return float(c.sensitivity[i]), float(c.specificity[i])

    # ---------------- helpers ----------------
    def _run(self, fn, on_done, on_fail, *args):
        w = Worker(fn, *args)
        w.signals.done.connect(on_done)
        w.signals.failed.connect(on_fail)
        self.pool.start(w)

    def _error(self, msg: str):
        self.headline.setText("Cardiovascular Risk Assessment: —")
        QMessageBox.critical(self, "Error", msg)

    def closeEvent(self, event):
        if self.server_proc and self.server_proc.poll() is None:
            self.server_proc.terminate()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
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
