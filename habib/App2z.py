from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QStackedWidget, QSizePolicy, QToolButton, QListWidget, QMessageBox, 
    QListWidgetItem, QProgressBar, QGraphicsDropShadowEffect, QFrame, QDialog
)
from PyQt5.QtGui import QFont, QIcon, QPixmap, QImage, QColor, QPainter, QPen, QConicalGradient, QBrush
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import Qt, QSize, QUrl, QTimer, QRectF, QPoint, QEvent
import sys
import os
import cv2
import time

from Appz import VideoPlayer

class TutorialDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Tutorial Penggunaan")
        
        # Setup Window
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog) 
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Hitung ukuran awal
        self.update_geometry_to_parent()

        # Data Tutorial (Sama seperti sebelumnya)
        self.steps = [
            {"title": "Selamat Datang di TrafficVision!", "text": "Aplikasi ini membantu Anda mendeteksi dan menghitung volume kendaraan secara otomatis menggunakan AI.", "image": "desain/step1.png"},
            {"title": "Langkah 1: Buka Video", "text": "Klik tombol 'New Video' di halaman utama, lalu pilih file CCTV (mp4/avi) dari komputer Anda.", "image": "desain/step2.png"},
            {"title": "Langkah 2: Gambar Garis Deteksi", "text": "Klik tombol 'Draw Line', lalu tarik garis di jalan pada video. Kendaraan yang melewati garis ini akan dihitung.", "image": "desain/step3.png"},
            {"title": "Langkah 3: Jalankan Deteksi", "text": "Klik tombol 'Run'. Sistem AI akan mulai memproses. Anda bisa melihat log detail setelah proses selesai.", "image": "desain/step4.png"}
        ]
        self.current_step = 0

        # --- LAYOUT UTAMA ---
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10) 

        self.container = QFrame()
        self.container.setStyleSheet("QFrame { background-color: #1e1e1e; border: 1px solid #333; border-radius: 20px; }")
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0,0,0, 150))
        self.container.setGraphicsEffect(shadow)

        self.inner_layout = QVBoxLayout(self.container)
        self.inner_layout.setContentsMargins(30, 30, 30, 30)

        # --- KONTEN ---
        self.lbl_title = QLabel()
        self.lbl_title.setStyleSheet("font-size: 20px; font-weight: bold; color: white; border: none; background: transparent;")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: #2c2c2c; border-radius: 10px; border: 1px solid #444;")
        self.lbl_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
        self.lbl_image.setMinimumSize(200, 150)

        self.lbl_text = QLabel()
        self.lbl_text.setWordWrap(True)
        self.lbl_text.setAlignment(Qt.AlignCenter)
        self.lbl_text.setStyleSheet("font-size: 14px; color: #cccccc; border: none; background: transparent; margin-top: 10px;")
        self.lbl_text.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.btn_layout = QHBoxLayout()
        self.btn_prev = QPushButton("← Sebelumnya")
        self.btn_prev.clicked.connect(self.prev_step)
        self.btn_prev.setStyleSheet("background: transparent; color: #888; font-weight: bold; text-align: left;")
        self.btn_prev.setCursor(Qt.PointingHandCursor)
        
        self.btn_next = QPushButton("Lanjut →")
        self.btn_next.clicked.connect(self.next_step)
        self.btn_next.setFixedSize(120, 40)
        self.btn_next.setCursor(Qt.PointingHandCursor)
        self.btn_next.setStyleSheet("QPushButton { background-color: #007acc; color: white; border-radius: 20px; font-weight: bold; } QPushButton:hover { background-color: #005f99; }")

        self.btn_layout.addWidget(self.btn_prev)
        self.btn_layout.addStretch()
        self.btn_layout.addWidget(self.btn_next)

        self.inner_layout.addWidget(self.lbl_title)
        self.inner_layout.addSpacing(10)
        self.inner_layout.addWidget(self.lbl_image)
        self.inner_layout.addWidget(self.lbl_text)
        self.inner_layout.addSpacing(15)
        self.inner_layout.addLayout(self.btn_layout)

        self.layout.addWidget(self.container)
        
        # Install Event Filter ke Parent agar bisa mendeteksi resize
        if self.parent_window:
            self.parent_window.installEventFilter(self)

        QTimer.singleShot(10, self.update_content)

    def update_geometry_to_parent(self):
        """Fungsi pintar untuk menyesuaikan ukuran dan posisi ke tengah parent"""
        if self.parent_window:
            # Ukuran 85% lebar dan 80% tinggi parent
            target_w = int(self.parent_window.width() * 0.85)
            target_h = int(self.parent_window.height() * 0.80)
            
            # Set ukuran
            self.resize(target_w, target_h)
            
            # Pindahkan ke tengah-tengah parent (Centering)
            parent_geo = self.parent_window.geometry()
            x = parent_geo.x() + (parent_geo.width() - target_w) // 2
            y = parent_geo.y() + (parent_geo.height() - target_h) // 2
            self.move(x, y)
        
        # Update konten gambar juga agar tidak gepeng
        self.update_content()

    def eventFilter(self, obj, event):
        # Jika parent di-resize atau dipindah, tutorial ikut menyesuaikan
        if obj == self.parent_window and event.type() in [event.Resize, event.Move]:
            self.update_geometry_to_parent()
        return super().eventFilter(obj, event)

    def update_content(self):
        if not self.isVisible(): return
        
        # Pastikan index valid
        if self.current_step < 0 or self.current_step >= len(self.steps): return

        data = self.steps[self.current_step]
        self.lbl_title.setText(data["title"])
        self.lbl_text.setText(data["text"])
        
        # Logic gambar responsif
        avail_w = self.lbl_image.width()
        avail_h = self.lbl_image.height()
        
        if avail_w > 10 and avail_h > 10:
            if os.path.exists(data["image"]):
                pix = QPixmap(data["image"])
                scaled_pix = pix.scaled(avail_w - 10, avail_h - 10, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.lbl_image.setPixmap(scaled_pix)
            else:
                self.lbl_image.clear()
                self.lbl_image.setText(f"[Gambar tidak ditemukan]")

        # Update tombol
        self.btn_prev.setVisible(self.current_step > 0)
        
        if self.current_step == len(self.steps) - 1:
            self.btn_next.setText("Selesai")
            self.btn_next.setStyleSheet("background-color: #27ae60; color: white; border-radius: 20px; font-weight: bold;")
        else:
            self.btn_next.setText("Lanjut →")
            self.btn_next.setStyleSheet("background-color: #007acc; color: white; border-radius: 20px; font-weight: bold;")

    def next_step(self):
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.update_content()
        else:
            self.accept() # Tutup dialog

    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.update_content()

class CustomTitleBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        # Tinggi Title Bar
        self.setFixedHeight(45) 
        # Background Abu-abu gelap (sedikit lebih terang dari body app)
        self.setStyleSheet("background-color: #1f1f1f; border-bottom: 1px solid #333;")
        
        # Layout Horizontal
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 0, 0, 0) # Margin Kiri 15px, Kanan 0 (biar tombol mepet)
        layout.setSpacing(15)

        # --- 1. LOGO (FIX GEPENG & UKURAN) ---
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(30, 30)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet("border: none; background: transparent;")
        
        if os.path.exists("desain/logoonly.png"):
            pix = QPixmap("desain/logoonly.png")
            # Gunakan scaledToHeight agar aspek rasio terjaga (tidak gepeng)
            scaled_pix = pix.scaledToHeight(24, Qt.SmoothTransformation)
            self.icon_label.setPixmap(scaled_pix)
        else:
            self.icon_label.setText("🚀")

        # --- 2. JUDUL APLIKASI ---
        self.title_label = QLabel("TRAFFICVISION")
        self.title_label.setStyleSheet("""
            color: #eeeeee; 
            font-weight: bold; 
            font-family: 'Segoe UI', Arial; 
            font-size: 14px; 
            border: none; 
            background: transparent;
        """)

        # --- 3. TOMBOL KONTROL WINDOW ---
        # Fungsi helper untuk membuat tombol seragam
        def create_btn(text, tooltip):
            btn = QPushButton(text)
            btn.setFixedSize(50, 45) # Lebar 50, Tinggi Full Bar (45)
            btn.setToolTip(tooltip)
            btn.setCursor(Qt.PointingHandCursor)
            return btn

        # Tombol Minimize (-)
        self.btn_min = create_btn("-", "Minimize")
        self.btn_min.clicked.connect(self.minimize_window)
        self.btn_min.setStyleSheet("""
            QPushButton { background: transparent; border: none; color: #cccccc; font-size: 20px; font-weight: bold; }
            QPushButton:hover { background-color: #333333; color: white; }
        """)

        # Tombol Maximize ([ ])
        self.btn_max = create_btn("□", "Maximize") # Pakai kotak ASCII standar
        self.btn_max.clicked.connect(self.toggle_max)
        self.btn_max.setStyleSheet("""
            QPushButton { background: transparent; border: none; color: #cccccc; font-size: 16px; }
            QPushButton:hover { background-color: #333333; color: white; }
        """)

        # Tombol Close (X) - Merah saat hover
        self.btn_close = create_btn("✕", "Close")
        self.btn_close.clicked.connect(self.close_window)
        self.btn_close.setStyleSheet("""
            QPushButton { background: transparent; border: none; color: #cccccc; font-size: 16px; }
            QPushButton:hover { background-color: #e81123; color: white; }
        """)

        # --- MENYUSUN LAYOUT ---
        layout.addWidget(self.icon_label)
        layout.addWidget(self.title_label)
        
        layout.addStretch() # Mendorong tombol ke pojok kanan
        
        layout.addWidget(self.btn_min)
        layout.addWidget(self.btn_max)
        layout.addWidget(self.btn_close)

        # Variabel untuk Dragging Window
        self.start = QPoint(0, 0)
        self.pressing = False

    # --- LOGIKA TOMBOL ---
    def minimize_window(self):
        self.window().showMinimized()

    def close_window(self):
        self.window().close()

    def toggle_max(self):
        if self.window().isMaximized():
            self.window().showNormal()
            self.btn_max.setText("□") # Ikon kotak satu
        else:
            self.window().showMaximized()
            self.btn_max.setText("❐") # Ikon kotak tumpuk (restore)

    # --- LOGIKA DRAG WINDOW ---
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Cek apakah klik berada di tepi paling atas (untuk resize vertikal)
            # Jika klik di 5 pixel teratas, JANGAN geser window (abaikan event ini di sini)
            if event.y() < 5 and not self.window().isMaximized():
                event.ignore() # Lempar event ke parent (MainWindow) untuk di-handle sebagai resize
                return

            self.start = self.mapToGlobal(event.pos())
            self.pressing = True

    def mouseMoveEvent(self, event):
        if self.pressing and not self.window().isMaximized():
            end = self.mapToGlobal(event.pos())
            movement = end - self.start
            self.window().setGeometry(self.window().x() + movement.x(),
                                    self.window().y() + movement.y(),
                                    self.window().width(),
                                    self.window().height())
            self.start = end

    def mouseReleaseEvent(self, event):
        self.pressing = False
    
    # Fitur Double Click Title Bar untuk Maximize
    def mouseDoubleClickEvent(self, event):
        self.toggle_max()

class LoadingSpinner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 60)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.timer.start(16) # 60 FPS (makin kecil makin halus)
        
        self.setAttribute(Qt.WA_TranslucentBackground)

    def rotate(self):
        # Putar 10 derajat per frame agar mulus
        self.angle = (self.angle + 10) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        painter.translate(w / 2, h / 2) # Titik tengah
        painter.rotate(self.angle)      # Rotasi kanvas

        # --- MEMBUAT EFEK ULAR (GRADASI CONICAL) ---
        # Gradasi melingkar: Mulai dari Transparan -> Biru Penuh
        gradient = QConicalGradient(0, 0, -90.0)
        
        # Warna Ekor (Transparan)
        gradient.setColorAt(0, Qt.transparent)
        
        # Warna Badan (Setengah pudar)
        gradient.setColorAt(0.5, QColor(0, 170, 255, 50)) 
        
        # Warna Kepala (Biru Penuh)
        gradient.setColorAt(1, QColor(0, 170, 255, 255))

        # Setup Pen dengan Brush Gradasi
        pen = QPen(QBrush(gradient), 6) # Ketebalan 6
        pen.setCapStyle(Qt.RoundCap)    # Ujung bulat
        
        painter.setPen(pen)
        
        # Gambar lingkaran penuh (namun terlihat sebagian karena gradasi)
        # drawArc(x, y, w, h, startAngle, spanAngle)
        # Satuan angle di Qt adalah 1/16 derajat. Jadi 360 * 16 = lingkaran penuh.
        painter.drawArc(-20, -20, 40, 40, 0, 360 * 16)

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(680, 400)
        self.setWindowFlags(Qt.FramelessWindowHint) # Hilangkan bingkai window
        self.setAttribute(Qt.WA_TranslucentBackground) # Background transparan

        # Layout Utama
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Frame Container (Kotak Utama)
        self.container = QFrame(self)
        self.container.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                color: white;
                border-radius: 20px;
            }
        """)
        
        # Efek Bayangan (Shadow)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.container.setGraphicsEffect(shadow)

        self.inner_layout = QVBoxLayout(self.container)
        self.inner_layout.setAlignment(Qt.AlignCenter)
        self.inner_layout.setSpacing(20)

        # 1. LOGO APLIKASI
        self.lbl_logo = QLabel()
        self.lbl_logo.setAlignment(Qt.AlignCenter)
        # Ganti dengan path logomu, jika tidak ada pakai teks emotikon
        if os.path.exists("desain/logonobg.png"):
            self.lbl_logo.setPixmap(QPixmap("desain/logonobg.png").scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.lbl_logo.setText("📷") 
            self.lbl_logo.setStyleSheet("font-size: 80px; background: transparent;")

        # 2. DESKRIPSI / STATUS LOADING
        self.lbl_desc = QLabel("Memuat modul kecerdasan buatan...")
        self.lbl_desc.setAlignment(Qt.AlignCenter)
        # Tambahkan margin-top agar tidak terlalu mepet dengan spinner
        self.lbl_desc.setStyleSheet("font-family: 'Segoe UI'; font-size: 14px; color: #888888; background: transparent; margin-top: 10px;")
        
        # 3. SPINNER (Pengganti Progress Bar)
        self.spinner = LoadingSpinner()
        
        # Layouting agar spinner ada di tengah bawah
        spinner_layout = QHBoxLayout()
        spinner_layout.addStretch()
        spinner_layout.addWidget(self.spinner)
        spinner_layout.addStretch()

        # --- PENYUSUNAN LAYOUT (URUTAN BARU) ---
        self.inner_layout.addStretch()
        
        # A. Logo Paling Atas
        self.inner_layout.addWidget(self.lbl_logo)
        self.inner_layout.addSpacing(30) # Jarak antara Logo dan Spinner
        
        # B. Spinner di Tengah (Di bawah logo)
        self.inner_layout.addLayout(spinner_layout)
        
        # C. Teks di Bawah (Di bawah Spinner)
        self.inner_layout.addWidget(self.lbl_desc)
        
        self.inner_layout.addStretch()

        self.layout.addWidget(self.container)

    def update_progress(self, value):
        
        # Update teks loading biar terlihat canggih
        if value < 20: self.lbl_desc.setText("Initializing system...")
        elif value < 40: self.lbl_desc.setText("Loading PyTorch & YOLO libraries...")
        elif value < 70: self.lbl_desc.setText("Preparing graphical interface...")
        elif value < 90: self.lbl_desc.setText("Checking GPU drivers...")
        else: self.lbl_desc.setText("Done! Launching TrafficVision...")

class HomePage(QWidget):
    def __init__(self, stacked):
        super().__init__()
        self.stacked = stacked

        self.tutorial_dialog = None
        
        # --- 1. DEFINISI LAYOUT UTAMA (WAJIB PERTAMA) ---
        mainLayout = QVBoxLayout(self)
        mainLayout.setAlignment(Qt.AlignCenter)
        mainLayout.setSpacing(40) 

        # --- 2. BUAT HEADER (TOMBOL TUTORIAL) ---
        header_layout = QHBoxLayout()
        header_layout.addStretch() # Dorong tombol ke kanan
        
        btn_help = QPushButton("?")
        btn_help.setFixedSize(50, 50)
        btn_help.setToolTip("Cara Penggunaan")
        btn_help.setCursor(Qt.PointingHandCursor)
        # Hubungkan ke fungsi tutorial
        btn_help.clicked.connect(self.show_tutorial) 
        
        btn_help.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                border-radius: 20px;
                font-size: 18px;
                font-weight: bold;
                border: 2px solid #555;
            }
            QPushButton:hover {
                background-color: #007acc;
                border-color: #007acc;
            }
        """)
        
        header_layout.addWidget(btn_help)
        
        # Masukkan header ke layout utama PALING ATAS
        mainLayout.addLayout(header_layout)

        # --- 3. MEMBUAT LOGO ---
        lbl_logo = QLabel()
        lbl_logo.setAlignment(Qt.AlignCenter)
        lbl_logo.setStyleSheet("border: none; background: transparent;")
        
        if os.path.exists("desain/logonobg.png"):
            pixmap = QPixmap("desain/logonobg.png") 
            lbl_logo.setPixmap(pixmap.scaledToWidth(350, Qt.SmoothTransformation))
        else:
            lbl_logo.setText("LOGO") 
            lbl_logo.setStyleSheet("font-size: 30px; font-weight: bold; color: white;")

        # --- 4. MEMBUAT TOMBOL MENU ---
        btnLayout = QHBoxLayout()
        btnLayout.setAlignment(Qt.AlignCenter)
        btnLayout.setSpacing(60)

        # Tombol New Video
        btn_new = self.createMenuButton("Desain/video.png", "New Analysis", "Start new video detection")
        btn_new.clicked.connect(lambda: stacked.setCurrentIndex(1))

        # Tombol Open Results
        btn_open = self.createMenuButton("Desain/file.png", "View History", "View previous analysis results")
        btn_open.clicked.connect(lambda: stacked.setCurrentIndex(2))

        btnLayout.addWidget(btn_new)
        btnLayout.addWidget(btn_open)
        
        # --- 5. SUSUN SISANYA ---
        mainLayout.addStretch() # Spacer agar logo & menu ada di tengah vertikal
        mainLayout.addWidget(lbl_logo)
        mainLayout.addLayout(btnLayout)
        mainLayout.addStretch() # Spacer bawah

        self.setLayout(mainLayout)

    # --- FUNGSI HELPER UNTUK TOMBOL MENU ---
    def createMenuButton(self, icon_path, title_text, sub_text):
        btn = QPushButton()
        btn.setFixedSize(350, 400)
        btn.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(btn)
        layout.setAlignment(Qt.AlignCenter)
        
        lbl_icon = QLabel()
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path).scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl_icon.setPixmap(pixmap)
        else:
            lbl_icon.setText("📷") 
            lbl_icon.setStyleSheet("font-size: 80px;")
        lbl_icon.setAlignment(Qt.AlignCenter)
        
        lbl_title = QLabel(title_text)
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("font-size: 24px; font-weight: bold; margin-top: 20px; background: transparent; border: none; color: white;")
        
        lbl_sub = QLabel(sub_text)
        lbl_sub.setAlignment(Qt.AlignCenter)
        lbl_sub.setWordWrap(True)
        lbl_sub.setStyleSheet("font-size: 14px; color: #888; margin-top: 5px; background: transparent; border: none;")

        layout.addWidget(lbl_icon)
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_sub)
        
        btn.setStyleSheet("""
            QPushButton {
                background-color: #252525;
                border: 2px solid #333;
                border-radius: 20px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #2f2f2f;
                border: 2px solid #00aaff;
                transform: scale(1.05);
            }
            QPushButton QLabel {
                background: transparent;
                border: none;
            }
        """)
        
        return btn

    # --- FUNGSI MEMBUKA TUTORIAL ---
    def show_tutorial(self):
        # Cek apakah dialog sudah ada dan masih terbuka
        if self.tutorial_dialog is not None and self.tutorial_dialog.isVisible():
            # Jika sudah ada, angkat ke depan (fokus) dan update posisinya
            self.tutorial_dialog.raise_()
            self.tutorial_dialog.activateWindow()
            self.tutorial_dialog.update_geometry_to_parent() # Paksa update posisi
            return # Jangan buat baru!

        # Jika belum ada atau sudah ditutup, buat baru
        # Kita kirim 'self.window()' agar parent-nya adalah MainWindow utama
        self.tutorial_dialog = TutorialDialog(self.window())
        
        # Saat dialog ditutup, reset variabel jadi None agar bersih
        self.tutorial_dialog.finished.connect(self.on_tutorial_closed)
        
        self.tutorial_dialog.show() # Gunakan show() bukan exec_() agar non-blocking (bisa resize parent)

    def on_tutorial_closed(self):
        self.tutorial_dialog = None

class NewVideoPage(QWidget):
    def __init__(self, stacked):
        super().__init__()
        layout = QVBoxLayout()
        lbl = QLabel("New Video Page")
        lbl.setFont(QFont("Arial", 22))
        lbl.setAlignment(Qt.AlignCenter)

        back = QPushButton("← Back")
        back.clicked.connect(lambda: stacked.setCurrentIndex(0))

        layout.addWidget(lbl)
        layout.addWidget(back, alignment=Qt.AlignCenter)
        self.setLayout(layout)


class ResultsPage(QWidget):
    def __init__(self, stacked):
        super().__init__()
        self.stacked = stacked

        layout = QVBoxLayout(self)

        # Back button di pojok kiri atas
        btn_layout = QHBoxLayout()
        back_btn = QPushButton("← Back")
        back_btn.setFixedSize(120, 50)
        back_btn.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        btn_layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Label Judul
        title = QLabel("List of Video Results")
        title.setFont(QFont("Arial", 26, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # List thumbnail video
        self.video_list = QListWidget()
        self.video_list.setViewMode(QListWidget.IconMode)
        self.video_list.setIconSize(QSize(240, 160))
        self.video_list.setResizeMode(QListWidget.Adjust)
        self.video_list.setSpacing(25)
        self.video_list.setMovement(QListWidget.Static)
        self.video_list.setGridSize(QSize(260, 200))
        self.video_list.itemDoubleClicked.connect(self.open_video)
        layout.addWidget(self.video_list)

        # Folder hasil
        self.results_folder = "hasil_test"
        self.load_video_list()

    def load_video_list(self):
        if not os.path.exists(self.results_folder):
            QMessageBox.warning(self, "Folder Not Found",
                                f"Folder '{self.results_folder}' tidak ditemukan!")
            return

        video_ext = (".mp4", ".avi", ".mov", ".mkv")
        video_files = [f for f in os.listdir(self.results_folder)
                       if f.lower().endswith(video_ext)]

        self.video_list.clear()

        for filename in video_files:
            item = QListWidgetItem(filename)

            # Load thumbnail video
            video_path = os.path.join(self.results_folder, filename)
            pixmap = self.get_video_thumbnail(video_path)

            if pixmap:
                icon = QIcon(pixmap)
            else:
                icon = QIcon("Desain/video.png")  # fallback icon

            item.setIcon(icon)
            item.setTextAlignment(Qt.AlignCenter)
            self.video_list.addItem(item)

    def get_video_thumbnail(self, video_path):
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()

        if not success:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        return pixmap.scaled(240, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def open_video(self, item):
        file_name = item.text()
        file_path = os.path.join(self.results_folder, file_name)

        # Pindah ke page VideoPlayer
        self.stacked.setCurrentIndex(1)

        # Ambil VideoPlayer instance dari stacked widget index 1
        video_player = self.stacked.widget(1)

        # Jalankan proses load video seperti open_file()
        video_player.current_video_path = file_path
        video_player.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))

        # Tampilkan judul file
        video_player.label_title.setText(file_name)

        # Reset UI controls
        video_player.info_progress.setValue(0)
        video_player.slider.setEnabled(True)
        video_player.play_video()
        video_player.reset_info_view()
        video_player.load_csv_summary(file_path)
        video_player.check_and_update_button(file_path)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        # 1. Matikan Title Bar
        self.setWindowFlags(Qt.FramelessWindowHint) 
        
        # 2. Setup Mouse Tracking (Wajib untuk Parent DAN Child utama)
        self.setMouseTracking(True) 
        
        # 3. Variabel Resize
        self.resize_margin = 5 
        self.resize_dir = None
        self.drag_pos = None

        # 4. Layout Utama
        window_layout = QVBoxLayout(self)
        window_layout.setContentsMargins(0, 0, 0, 0)
        window_layout.setSpacing(0)

        # 5. Container Utama
        self.main_container = QFrame()
        # Aktifkan tracking di container agar mouse terdeteksi di atasnya
        self.main_container.setMouseTracking(True) 
        # Pasang mata-mata (Event Filter)
        self.main_container.installEventFilter(self) 
        
        self.main_container.setStyleSheet("""
            QFrame {
                background-color: #121212; 
                border: 1px solid #333; 
            }
        """)
        
        container_layout = QVBoxLayout(self.main_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # 6. Pasang Isi (Title Bar & Stacked)
        self.title_bar = CustomTitleBar(self)
        container_layout.addWidget(self.title_bar)

        self.stacked = QStackedWidget()
        self.home_page = HomePage(self.stacked)
        self.video_page = VideoPlayer(self.stacked)
        self.results_page = ResultsPage(self.stacked)

        self.stacked.addWidget(self.home_page)
        self.stacked.addWidget(self.video_page)
        self.stacked.addWidget(self.results_page)
        
        container_layout.addWidget(self.stacked)
        window_layout.addWidget(self.main_container)

        self.resize(1024, 768)

    # --- INI KUNCI RAHASIANYA (EVENT FILTER) ---
    def eventFilter(self, obj, event):
        # Jika event terjadi di main_container (taplak meja)
        if obj == self.main_container:
            # Jika mouse bergerak (Hover atau Drag)
            if event.type() == QEvent.MouseMove:
                self.mouseMoveEvent(event) # Paksa panggil fungsi resize kita
                return True # Event dianggap sudah ditangani
            
            # Jika mouse diklik (Mulai Resize)
            elif event.type() == QEvent.MouseButtonPress:
                self.mousePressEvent(event)
                return True
                
            # Jika mouse dilepas (Selesai Resize)
            elif event.type() == QEvent.MouseButtonRelease:
                self.mouseReleaseEvent(event)
                return True
                
        return super().eventFilter(obj, event)

    # --- LOGIKA RESIZE (Sama seperti sebelumnya, tapi sekarang pasti terpanggil) ---
    def get_resize_direction(self, global_pos):
        # Konversi posisi global (layar) ke posisi lokal (window)
        pos = self.mapFromGlobal(global_pos)
        
        w, h = self.width(), self.height()
        x, y = pos.x(), pos.y()
        m = self.resize_margin

        on_left = x < m
        on_right = x > w - m
        on_top = y < m
        on_bottom = y > h - m

        if on_top and on_left: return "top_left"
        if on_top and on_right: return "top_right"
        if on_bottom and on_left: return "bottom_left"
        if on_bottom and on_right: return "bottom_right"
        if on_top: return "top"
        if on_bottom: return "bottom"
        if on_left: return "left"
        if on_right: return "right"
        return None

    def set_cursor_shape(self, direction):
        if direction in ["top", "bottom"]: self.setCursor(Qt.SizeVerCursor)
        elif direction in ["left", "right"]: self.setCursor(Qt.SizeHorCursor)
        elif direction in ["top_left", "bottom_right"]: self.setCursor(Qt.SizeFDiagCursor)
        elif direction in ["top_right", "bottom_left"]: self.setCursor(Qt.SizeBDiagCursor)
        else: self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Gunakan globalPos agar akurat
            direction = self.get_resize_direction(event.globalPos())
            if direction:
                self.resize_dir = direction
                self.drag_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if self.isMaximized():
            self.setCursor(Qt.ArrowCursor) # Pastikan kursor normal
            return super().mouseMoveEvent(event) # Kembalikan ke default

        # 1. LOGIKA DRAG (RESIZE)
        if self.resize_dir:
            delta = event.globalPos() - self.drag_pos
            self.drag_pos = event.globalPos()
            
            geo = self.geometry()
            x, y, w, h = geo.x(), geo.y(), geo.width(), geo.height()
            dx, dy = delta.x(), delta.y()

            if "right" in self.resize_dir: w += dx
            if "bottom" in self.resize_dir: h += dy
            if "left" in self.resize_dir:
                if w - dx > self.minimumWidth():
                    x += dx; w -= dx
            if "top" in self.resize_dir:
                if h - dy > self.minimumHeight():
                    y += dy; h -= dy
            
            self.setGeometry(x, y, w, h)
            
        # 2. LOGIKA HOVER (UBAH KURSOR)
        else:
            direction = self.get_resize_direction(event.globalPos())
            self.set_cursor_shape(direction)

    def mouseReleaseEvent(self, event):
        self.resize_dir = None
        self.setCursor(Qt.ArrowCursor)

if __name__ == '__main__':
    # Fix untuk VLC path di Windows (jika diperlukan)
    if sys.platform == "win32":
        try:
            vlc_path = os.path.dirname(vlc.__file__)
            os.add_dll_directory(vlc_path)
        except: pass

    app = QApplication(sys.argv)
    
    splash = SplashScreen()
    splash.show()

    # Loop simulasi loading
    for i in range(101):
        # Panggil fungsi update status teks saja (spinner berputar otomatis)
        splash.update_progress(i) 
        
        app.processEvents()
        time.sleep(0.03) 
    
    splash.close()

    win = MainWindow()
    win.showMaximized()
    
    sys.exit(app.exec_())