import sys
import cv2
import time
import os
import openpyxl
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QSlider, QMessageBox,
    QTableWidget, QTableWidgetItem, QDialog, QFrame, QApplication
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt

from constants.vehicle_classes import VEHICLE_CLASSES

# --- detector import ---
try:
    from core.detector_yolo import YOLODetector
    DETECTOR_AVAILABLE = True
except Exception as e:
    print("Detector import failed:", e)
    YOLODetector = None
    DETECTOR_AVAILABLE = False


class DetailWindow(QDialog):
    def __init__(self, vehicle_counts, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detail Data Kendaraan")
        self.resize(400, 300)
        self.table = QTableWidget(self)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Kelas Kendaraan", "Jumlah"])
        self.table.setRowCount(len(VEHICLE_CLASSES))
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.update_table(vehicle_counts)

    def update_table(self, vehicle_counts):
        for i, cls in enumerate(VEHICLE_CLASSES):
            count = vehicle_counts.get(cls, 0)
            self.table.setItem(i, 0, QTableWidgetItem(str(cls)))
            self.table.setItem(i, 1, QTableWidgetItem(str(count)))


class TrafficVisionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Vision 🚦")
        self.setGeometry(100, 100, 1800, 1100)

        # --- Video setup ---
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = None
        self.total_frames = 0

        # --- Detector setup ---
        self.detector = None
        if DETECTOR_AVAILABLE:
            model_path = os.path.join("models", "yolov8n.pt")
            try:
                self.detector = YOLODetector(model_path=model_path)
            except Exception as e:
                print("❌ Gagal load YOLO model:", e)
                self.detector = None

        self.capture_dir = "captures"
        os.makedirs(self.capture_dir, exist_ok=True)

        self.vehicle_counts_total = {cls: 0 for cls in VEHICLE_CLASSES}
        self.vehicle_counts_live = {cls: 0 for cls in VEHICLE_CLASSES}

        # ====================== UI ======================
        # --- VIDEO PREVIEW ---
        self.video_label = QLabel("Video Preview\n16:9")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #222; color: white; font-size: 24px;")
        self.video_label.setFixedSize(1280, 720)

        # --- SLIDER BAR ---
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderReleased.connect(self.set_video_position)
        self.slider.setStyleSheet("margin-top: 15px; margin-bottom: 5px;")

        slider_label = QLabel("Slider bar (menit)")
        slider_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_label.setStyleSheet("color: gray; font-size: 12px;")

        # --- BUTTON STYLE ---
        btn_style = """
            QPushButton {
                background-color: #2e2e2e;
                color: white;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3c3c3c;
            }
        """

        # --- Tombol baris pertama ---
        self.btn_input = QPushButton("Input Video")
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_capture = QPushButton("Capture")

        for btn in [self.btn_input, self.btn_play, self.btn_pause, self.btn_capture]:
            btn.setStyleSheet(btn_style)

        row1 = QHBoxLayout()
        row1.addWidget(self.btn_input)
        row1.addWidget(self.btn_play)
        row1.addWidget(self.btn_pause)
        row1.addWidget(self.btn_capture)

        # --- Tombol baris kedua (3 tombol aja) ---
        self.btn_export = QPushButton("Data Report")
        self.btn_detail = QPushButton("Detail Data")
        self.btn_filter = QPushButton("Filter")

        for btn in [self.btn_export, self.btn_detail, self.btn_filter]:
            btn.setStyleSheet(btn_style)

        row2 = QHBoxLayout()
        row2.addWidget(self.btn_export)
        row2.addWidget(self.btn_detail)
        row2.addWidget(self.btn_filter)

        # --- Kumpulan semua tombol ---
        control_layout = QVBoxLayout()
        control_layout.addLayout(row1)
        control_layout.addLayout(row2)

        # --- LEFT PANEL (video + slider + tombol) ---
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_label)
        left_layout.addWidget(self.slider)
        left_layout.addWidget(slider_label)
        left_layout.addLayout(control_layout)

        # --- RIGHT PANEL (live detail count) ---
        self.side_frame = QFrame()
        self.side_frame.setStyleSheet("background-color: #1e1e1e; color: white; border-radius: 10px;")
        side_layout = QVBoxLayout(self.side_frame)
        title = QLabel("Live Detail Count Kendaraan")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        side_layout.addWidget(title)

        self.vehicle_labels = {}
        for cls in VEHICLE_CLASSES:
            lbl = QLabel(f"{cls}: 0")
            lbl.setStyleSheet("font-size: 14px; padding: 4px;")
            self.vehicle_labels[cls] = lbl
            side_layout.addWidget(lbl)
        side_layout.addStretch()

        # --- MAIN LAYOUT ---
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 3)
        main_layout.addWidget(self.side_frame, 1)
        self.setLayout(main_layout)

        # --- Event Binding ---
        self.btn_input.clicked.connect(self.load_video)
        self.btn_play.clicked.connect(self.play_video)
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_capture.clicked.connect(self.capture_frame)
        self.btn_export.clicked.connect(self.export_data)
        self.btn_detail.clicked.connect(self.show_detail)
        self.btn_filter.clicked.connect(self.show_filter)

        self.set_controls_enabled(False)

    # ====================== VIDEO HANDLER ======================
    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Gagal membuka video.")
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setMaximum(self.total_frames)
        self.set_controls_enabled(True)

    def play_video(self):
        if self.cap:
            self.timer.start(33)

    def pause_video(self):
        self.timer.stop()

    def update_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        if self.detector:
            try:
                annotated = self.detector.process_frame(frame)
            except Exception:
                annotated = frame
        else:
            annotated = frame

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pix)

        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.slider.setValue(pos)

    def capture_frame(self):
        if self.current_frame is None:
            return
        fname = os.path.join(self.capture_dir, f"capture_{time.strftime('%Y%m%d-%H%M%S')}.png")
        cv2.imwrite(fname, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
        QMessageBox.information(self, "Capture", f"Frame disimpan ke {fname}")

    def set_video_position(self):
        if self.cap:
            pos = int(self.slider.value())
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    def show_detail(self):
        dlg = DetailWindow(self.vehicle_counts_total, self)
        dlg.exec()

    def show_filter(self):
        QMessageBox.information(self, "Filter", "Fitur filter kendaraan akan datang ✨")

    def export_data(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Simpan Data", "", "Excel Files (*.xlsx);;PDF Files (*.pdf)")
        if not fname:
            return
        if fname.endswith(".xlsx"):
            self.export_excel(fname)
        elif fname.endswith(".pdf"):
            self.export_pdf(fname)

    def export_excel(self, fname):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Kelas Kendaraan", "Jumlah"])
        for cls, count in self.vehicle_counts_total.items():
            ws.append([cls, count])
        wb.save(fname)
        QMessageBox.information(self, "Export", f"Data disimpan ke {fname}")

    def export_pdf(self, fname):
        c = canvas.Canvas(fname, pagesize=A4)
        width, height = A4
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, height - 50, "Traffic Vision - Laporan Kendaraan")
        y = height - 100
        for cls, count in self.vehicle_counts_total.items():
            c.drawString(100, y, f"{cls}: {count}")
            y -= 20
        c.save()
        QMessageBox.information(self, "Export", f"PDF disimpan ke {fname}")

    def set_controls_enabled(self, enabled: bool):
        for btn in [self.btn_play, self.btn_pause, self.btn_capture,
                    self.btn_export, self.btn_detail, self.btn_filter]:
            btn.setEnabled(enabled)
        self.slider.setEnabled(enabled)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrafficVisionApp()
    window.show()
    sys.exit(app.exec())