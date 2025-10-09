# ui/home_window.py
import sys
import cv2
import time
import os
import openpyxl
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QSlider, QMessageBox,
    QTableWidget, QTableWidgetItem, QDialog, QFrame, QComboBox
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt

from constants.vehicle_classes import VEHICLE_CLASSES

# import detector (pastikan core/detector_yolo.py sesuai API)
try:
    from core.detector_yolo import YOLODetector
    DETECTOR_AVAILABLE = True
except Exception as e:
    print("Detector import failed:", e)
    YOLODetector = None
    DETECTOR_AVAILABLE = False


class DetailWindow(QDialog):
    """Popup detail jumlah kendaraan (TOTAL sepanjang video)."""
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
        self.setGeometry(100, 100, 1800, 1169)

        # --- Video + timer ---
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = None
        self.total_frames = 0
        self.current_frame_index = 0

        # --- Detector (optional) ---
        self.detector = None
        if DETECTOR_AVAILABLE:
            # ubah path model sesuai tempat model kamu (atau None untuk demo)
            model_path = os.path.join("models", "yolov8n.pt")
            try:
                self.detector = YOLODetector(model_path=model_path)
            except Exception as e:
                print("Gagal inisialisasi detector:", e)
                self.detector = None

        # --- Simulasi processing timer ---
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self.finish_processing)

        # --- capture folder ---
        self.capture_dir = "captures"
        os.makedirs(self.capture_dir, exist_ok=True)

        # --- Counters ---
        # keys = keys of VEHICLE_CLASSES (whatever format you used there)
        self.vehicle_counts_total = {cls: 0 for cls in VEHICLE_CLASSES}
        self.vehicle_counts_live = {cls: 0 for cls in VEHICLE_CLASSES}

        # --- active filter (key from VEHICLE_CLASSES or None) ---
        self.active_filter = None

        # --- UI: video preview ---
        self.video_label = QLabel("Video Preview")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setFixedSize(1280, 720)

        # --- UI: live panel ---
        self.count_frame = QFrame()
        self.count_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.count_frame.setStyleSheet("background-color: #1e1e1e; color: white; border-radius: 10px;")
        self.count_layout = QVBoxLayout(self.count_frame)
        self.title_label = QLabel("Live Vehicle Count")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.count_layout.addWidget(self.title_label)

        self.vehicle_labels = {}
        for cls in VEHICLE_CLASSES:
            lbl = QLabel(f"{cls}: 0")
            lbl.setStyleSheet("font-size: 14px; padding: 3px; color: white;")
            self.vehicle_labels[cls] = lbl
            self.count_layout.addWidget(lbl)
        self.count_layout.addStretch()

        # --- Buttons ---
        self.btn_input = QPushButton("Input Video")
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_capture = QPushButton("Capture")
        self.btn_export = QPushButton("Export Data")
        self.btn_detail = QPushButton("Detail Data")
        self.btn_filter = QPushButton("Filter")

        self.btn_input.clicked.connect(self.load_video)
        self.btn_play.clicked.connect(self.play_video)
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_capture.clicked.connect(self.capture_frame)
        self.btn_export.clicked.connect(self.export_data)
        self.btn_detail.clicked.connect(self.show_detail)
        self.btn_filter.clicked.connect(self.show_filter)

        btn_layout = QHBoxLayout()
        for b in [self.btn_input, self.btn_play, self.btn_pause, self.btn_capture,
                  self.btn_export, self.btn_detail, self.btn_filter]:
            btn_layout.addWidget(b)

        # --- Slider ---
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderReleased.connect(self.set_video_position)

        # --- Loading label ---
        self.loading_label = QLabel("Processing video...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet("color: orange; font-size: 16px;")
        self.loading_label.hide()

        # --- Layout utama ---
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)
        left_layout.addLayout(btn_layout)
        left_layout.addWidget(self.slider)
        left_layout.addWidget(self.loading_label)
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.count_frame, stretch=0)

        self.setLayout(main_layout)
        self.set_controls_enabled(False)

    # ----------------------
    # helper mapping function
    # ----------------------
    def _map_detector_label_to_clskey(self, det_label):
        """
        Map a detector class name (e.g. '1' or 'Sepeda motor') to the key in VEHICLE_CLASSES.
        Returns the key if found, else None.
        """
        sdet = str(det_label)
        # direct match with keys
        if sdet in VEHICLE_CLASSES:
            return sdet
        # match against values (human label)
        for k, v in VEHICLE_CLASSES.items():
            if str(v) == sdet:
                return k
        # not found
        return None

    # ----------------------
    def set_controls_enabled(self, enabled: bool):
        for btn in [self.btn_play, self.btn_pause, self.btn_capture,
                    self.btn_export, self.btn_detail, self.btn_filter]:
            btn.setEnabled(enabled)
        self.slider.setEnabled(enabled)

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if video_path:
            # open capture
            self.cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.slider.setMaximum(self.total_frames)

            # reset UI counters and detector internal state
            self.vehicle_counts_total = {cls: 0 for cls in VEHICLE_CLASSES}
            self.vehicle_counts_live = {cls: 0 for cls in VEHICLE_CLASSES}
            if self.detector:
                try:
                    self.detector.reset()
                except Exception:
                    pass

            self.loading_label.show()
            self.set_controls_enabled(False)
            self.processing_timer.start(1500)  # short dummy processing

    def finish_processing(self):
        self.processing_timer.stop()
        self.loading_label.hide()
        self.set_controls_enabled(True)

    def play_video(self):
        if self.cap is not None and self.cap.isOpened():
            self.timer.start(30)

    def pause_video(self):
        self.timer.stop()

    def capture_frame(self):
        if self.current_frame is not None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.capture_dir, f"capture_{timestamp}.png")
            # current_frame is RGB for display; convert back to BGR for saving
            cv2.imwrite(filename, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
            QMessageBox.information(self, "Capture", f"Frame disimpan ke {filename}")

    def update_frame(self):
        """Main loop while video playing. robust try/except to avoid sudden crash."""
        try:
            if self.cap is None or not self.cap.isOpened():
                return

            ret, frame = self.cap.read()
            if not ret:
                self.timer.stop()
                return

            # If detector available, use it (preferred). detector.process_frame should:
            #  - return annotated BGR frame
            #  - update detector.current_counts (dict) and detector.total_counts (dict)
            annotated = frame
            if self.detector:
                try:
                    # if user selected a filter (key from VEHICLE_CLASSES), map to possible detector labels
                    allowed = None
                    if self.active_filter:
                        # allow both key and human label value as possible detector labels
                        allowed = [str(self.active_filter)]
                        human = VEHICLE_CLASSES.get(self.active_filter)
                        if human:
                            allowed.append(str(human))
                    # call process_frame with allowed_classes optional param
                    # detector should handle allowed_classes == None (draw all)
                    annotated = self.detector.process_frame(frame, allowed_classes=allowed)
                except Exception as e:
                    print("Detector processing error:", e)
                    # fallback: annotated = frame (no boxes)
                    annotated = frame

                # read counts from detector (these should be class names or numeric strings)
                det_live = getattr(self.detector, "current_counts", {}) or {}
                det_total = getattr(self.detector, "total_counts", {}) or {}

                # reset UI counters, then fill using mapping function
                self.vehicle_counts_live = {cls: 0 for cls in VEHICLE_CLASSES}
                self.vehicle_counts_total = {cls: 0 for cls in VEHICLE_CLASSES}

                for dlabel, cnt in det_live.items():
                    key = self._map_detector_label_to_clskey(dlabel)
                    if key:
                        self.vehicle_counts_live[key] = cnt

                for dlabel, cnt in det_total.items():
                    key = self._map_detector_label_to_clskey(dlabel)
                    if key:
                        self.vehicle_counts_total[key] = cnt

            else:
                # no detector installed -> demo increment (so UI doesn't stay empty)
                demo_key = list(self.vehicle_counts_live.keys())[0]
                self.vehicle_counts_live[demo_key] += 1
                self.vehicle_counts_total[demo_key] += 1
                annotated = frame

            # convert annotated BGR -> RGB for Qt
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            self.current_frame = rgb
            self.current_frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            self.slider.setValue(self.current_frame_index)

            # update UI labels
            self.update_live_counts()

            # show image
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio
            )
            self.video_label.setPixmap(pixmap)

        except Exception as ex:
            print("❌ update_frame error:", ex)
            # stop to avoid crash loop
            try:
                self.timer.stop()
            except Exception:
                pass

    def update_live_counts(self):
        """Update UI labels on the right panel"""
        for cls, lbl in self.vehicle_labels.items():
            lbl.setText(f"{cls}: {self.vehicle_counts_live.get(cls, 0)}")

    def set_video_position(self):
        if self.cap is not None:
            pos = int(self.slider.value())
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            self.current_frame_index = pos

    def show_detail(self):
        dlg = DetailWindow(self.vehicle_counts_total, self)
        dlg.exec()

    def show_filter(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Pilih Filter Kendaraan")

        combo = QComboBox()
        combo.addItem("Tampilkan Semua")
        for cls in VEHICLE_CLASSES:
            combo.addItem(str(cls))

        btn_ok = QPushButton("OK")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Filter kendaraan yang ingin ditampilkan:"))
        layout.addWidget(combo)
        layout.addWidget(btn_ok)
        dlg.setLayout(layout)

        def apply_filter():
            selected = combo.currentText()
            if selected == "Tampilkan Semua":
                self.active_filter = None
            else:
                self.active_filter = selected
            dlg.accept()
            QMessageBox.information(self, "Filter Diterapkan",
                                    f"Menampilkan hanya kendaraan: {self.active_filter or 'Semua'}")

        btn_ok.clicked.connect(apply_filter)
        dlg.exec()

    def export_data(self):
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Simpan Data", "", "Excel Files (*.xlsx);;PDF Files (*.pdf)"
        )
        if not filename:
            return
        if filename.endswith(".xlsx"):
            self.export_excel(filename)
        elif filename.endswith(".pdf"):
            self.export_pdf(filename)
        else:
            if "Excel" in selected_filter:
                filename += ".xlsx"
                self.export_excel(filename)
            elif "PDF" in selected_filter:
                filename += ".pdf"
                self.export_pdf(filename)

    def export_excel(self, filename):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Vehicle Data"
        ws.append(["Kelas Kendaraan", "Jumlah"])
        for cls, count in self.vehicle_counts_total.items():
            ws.append([cls, count])
        wb.save(filename)
        QMessageBox.information(self, "Export", f"Data berhasil diexport ke {filename}")

    def export_pdf(self, filename):
        c = canvas.Canvas(filename, pagesize=A4)
        width, height = A4
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, height - 50, "Traffic Vision - Vehicle Report")
        c.setFont("Helvetica", 12)
        y = height - 100
        for cls, count in self.vehicle_counts_total.items():
            c.drawString(100, y, f"{cls}: {count}")
            y -= 20
        c.save()
        QMessageBox.information(self, "Export", f"Data berhasil diexport ke {filename}")