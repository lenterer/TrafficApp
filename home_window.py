import sys
import cv2
import time
import os
import random
import openpyxl
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QSlider, QMessageBox,
    QTableWidget, QTableWidgetItem, QDialog, QFrame, QGridLayout
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt

from constants.vehicle_classes import VEHICLE_CLASSES


class DetailWindow(QDialog):
    """Popup detail jumlah kendaraan"""
    def __init__(self, vehicle_counts, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detail Data Kendaraan")
        self.resize(400, 300)

        self.table = QTableWidget(self)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Kelas Kendaraan", "Jumlah"])

        self.table.setRowCount(len(VEHICLE_CLASSES))
        for i, cls in enumerate(VEHICLE_CLASSES):
            count = vehicle_counts.get(cls, 0)
            self.table.setItem(i, 0, QTableWidgetItem(cls))
            self.table.setItem(i, 1, QTableWidgetItem(str(count)))

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)


class TrafficVisionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Traffic Vision 🚦")
        self.setGeometry(100, 100, 1800, 1169)

        #Variabel video
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = None
        self.total_frames = 0
        self.current_frame_index = 0

        #Timer simulasi processing
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self.finish_processing)

        #Buat folder capture kalau belum ada
        self.capture_dir = "captures"
        os.makedirs(self.capture_dir, exist_ok=True)

        #Data kendaraan
        self.vehicle_counts = {cls: 0 for cls in VEHICLE_CLASSES}

        #Video Preview (16:9)
        self.video_label = QLabel("Video Preview")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setFixedSize(1280, 720)

        #Panel live count kendaraan
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
            lbl.setStyleSheet("font-size: 14px; padding: 3px;")
            self.vehicle_labels[cls] = lbl
            self.count_layout.addWidget(lbl)

        self.count_layout.addStretch()

        #Tombol kontrol
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
        for btn in [self.btn_input, self.btn_play, self.btn_pause, self.btn_capture, self.btn_export, self.btn_detail, self.btn_filter]:
            btn_layout.addWidget(btn)

        #timeline video
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderReleased.connect(self.set_video_position)

        #Loading spinner (dummy)
        self.loading_label = QLabel("Processing video...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet("color: orange; font-size: 16px;")
        self.loading_label.hide()

        #Layout utama (horizontal)
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)
        left_layout.addLayout(btn_layout)
        left_layout.addWidget(self.slider)
        left_layout.addWidget(self.loading_label)
        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.count_frame, stretch=1)

        self.setLayout(main_layout)
        self.set_controls_enabled(False)

    def set_controls_enabled(self, enabled: bool):
        for btn in [self.btn_play, self.btn_pause, self.btn_capture, self.btn_export, self.btn_detail, self.btn_filter]:
            btn.setEnabled(enabled)
        self.slider.setEnabled(enabled)

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setMaximum(self.total_frames)

            self.loading_label.show()
            self.set_controls_enabled(False)
            self.processing_timer.start(3000)

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
            cv2.imwrite(filename, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
            print(f"Frame disimpan ke {filename}")

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.timer.stop()
                return

            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.setValue(self.current_frame_index)

            # Simulasi: random nambah count kendaraan
            cls = random.choice(list(VEHICLE_CLASSES.keys()))
            self.vehicle_counts[cls] += 1
            self.update_live_counts()

            h, w, ch = self.current_frame.shape
            qimg = QImage(self.current_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio
            )
            self.video_label.setPixmap(pixmap)

    def update_live_counts(self):
        """Update text pada live count panel"""
        for cls, lbl in self.vehicle_labels.items():
            lbl.setText(f"{cls}: {self.vehicle_counts[cls]}")

    def set_video_position(self):
        if self.cap is not None:
            pos = self.slider.value()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            self.current_frame_index = pos

    def show_detail(self):
        dlg = DetailWindow(self.vehicle_counts, self)
        dlg.exec()

    def show_filter(self):
        QMessageBox.information(self, "Filter", "Filter kendaraan masih dummy 🚧")

    def export_data(self):
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Simpan Data", "", "Excel Files (*.xlsx);;PDF Files (*.pdf)"
        )
        if filename:
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
        for cls, count in self.vehicle_counts.items():
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
        for cls, count in self.vehicle_counts.items():
            c.drawString(100, y, f"{cls}: {count}")
            y -= 20
        c.save()
        QMessageBox.information(self, "Export", f"Data berhasil diexport ke {filename}")