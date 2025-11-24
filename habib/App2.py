from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QStackedWidget, QSizePolicy, QToolButton, QListWidget, QMessageBox, 
    QListWidgetItem
)
from PyQt5.QtGui import QFont, QIcon, QPixmap, QImage
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import Qt, QSize, QUrl
import sys
import os
import cv2

from App import VideoPlayer

class HomePage(QWidget):
    def __init__(self, stacked):
        super().__init__()
        self.stacked = stacked
        mainLayout = QVBoxLayout(self)
        mainLayout.setAlignment(Qt.AlignCenter)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(200)

        # Tombol New Video
        btn_new = self.createMenuButton("Desain/video.png", "New Video")
        btn_new.clicked.connect(lambda: stacked.setCurrentIndex(1))

        # Tombol Open Results
        btn_open = self.createMenuButton("Desain/file.png", "Open Results")
        btn_open.clicked.connect(lambda: stacked.setCurrentIndex(2))

        layout.addWidget(btn_new)
        layout.addWidget(btn_open)
        mainLayout.addLayout(layout)
        self.setLayout(mainLayout)

        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QToolButton {
                background-color: #d5d5d5;
                border-radius: 18px;
                padding: 30px;
            }
            QToolButton:hover {
                background-color: #c5c7c8;
            }
        """)

    def createMenuButton(self, icon_path, label_text):
        btn = QToolButton()
        btn.setIcon(QIcon(icon_path))
        btn.setIconSize(QSize(200, 200))
        btn.setFont(QFont("Arial", 26, QFont.Bold))
        btn.setText(label_text)
        btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn.setMinimumSize(400, 400)
        btn.setMaximumSize(500, 500)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn.setStyleSheet("""
            QToolButton {
                background-color: #d5d5d5;
                border-radius: 22px;
                padding: 40px;
                color: #2c2c2c;
            }
            QToolButton:hover {
                background-color: #c5c7c8;
            }
        """)
        btn.setIconSize(btn.sizeHint())
        return btn


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
        title = QLabel("Daftar Hasil Video")
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

        self.stacked = QStackedWidget()

        self.stacked.addWidget(HomePage(self.stacked))  # Index 0
        self.video_page = VideoPlayer(self.stacked)
        self.stacked.addWidget(self.video_page)  # Index 1
        self.stacked.addWidget(ResultsPage(self.stacked))  # Index 2

        layout = QVBoxLayout()
        layout.addWidget(self.stacked)
        self.setLayout(layout)

        self.setWindowTitle("Multi Page UI Example")
        self.resize(2560, 1560)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #fcf3eb;
            }
            QToolButton {
                background-color: #2b2b2b;
                border-radius: 22px;
                padding: 40px;
                color: #ffffff;
            }
            QToolButton:hover {
                background-color: #3c3c3c;
            }
        """)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())