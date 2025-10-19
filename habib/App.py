import sys
import vlc
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel
    , QSlider, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer
from deteksi2 import jalankan_deteksi

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player (PyQt + VLC)")
        self.setGeometry(100, 100, 1000, 700)
        self.current_video_path = None # Path video

        # Layout utama
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Label judul video (pojok kiri atas)
        self.label_title = QLabel("", self)
        self.label_title.setStyleSheet("""
            color: white;
            background-color: rgba(0, 0, 0, 80);
            padding: 5px;
            font-size: 14px;
        """)
        self.label_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label_title.setFixedHeight(30)

        # Tombol kontrol
        self.controls_layout = QHBoxLayout()
        self.btn_open = QPushButton("Open")
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        self.run_button = QPushButton("Run")

        for btn in [self.btn_open, self.btn_play, self.btn_pause, self.btn_stop]:
            btn.setFixedSize(80, 30)

        self.controls_layout.addStretch()
        self.controls_layout.addWidget(self.btn_open)
        self.controls_layout.addWidget(self.btn_play)
        self.controls_layout.addWidget(self.btn_pause)
        self.controls_layout.addWidget(self.btn_stop)
        self.controls_layout.addWidget(self.run_button)
        self.controls_layout.addStretch()

        # Inisialisasi VLC
        self.instance = vlc.Instance()
        self.media_player = self.instance.media_player_new()
        
        # Progress bar layout (slider + waktu)
        self.progress_layout = QHBoxLayout()
        self.label_current = QLabel("00:00")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setEnabled(False)
        self.label_total = QLabel("00:00")

        self.progress_layout.addWidget(self.label_current)
        self.progress_layout.addWidget(self.slider, stretch=1)
        self.progress_layout.addWidget(self.label_total)

        # --- Bagian utama: video + panel info berdampingan ---
        self.video_info_layout = QHBoxLayout()

        # Area video
        self.video_frame = QWidget(self)
        self.video_frame.setStyleSheet("background-color: black;")

        # Panel informasi di kanan
        self.info_panel = QVBoxLayout()
        self.info_label = QLabel("Status: Menunggu video...")
        self.info_label.setStyleSheet("font-size: 14px; color: white; background-color: #333; padding: 6px; border-radius: 4px;")
        self.info_progress = QProgressBar()
        self.info_progress.setRange(0, 100)
        self.info_progress.setValue(0)
        self.info_panel.addWidget(self.info_label)
        self.info_panel.addWidget(self.info_progress)

        # Gabungkan video + info panel
        self.video_info_layout.addWidget(self.video_frame, stretch=3)
        self.video_info_layout.addLayout(self.info_panel, stretch=1)

        # Masukkan widget ke layout utama
        self.layout.addWidget(self.label_title)
        self.layout.addLayout(self.video_info_layout, stretch=1)
        self.layout.addLayout(self.progress_layout)
        self.layout.addLayout(self.controls_layout)

        # Hubungkan tombol
        self.btn_open.clicked.connect(self.open_file)
        self.btn_play.clicked.connect(self.play_video)
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_stop.clicked.connect(self.stop_video)
        self.run_button.clicked.connect(self.run_detection)
        self.slider.sliderMoved.connect(self.set_position)

        # Timer untuk update slider
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.update_ui)

        # Hubungkan media player ke widget video
        if sys.platform.startswith("linux"):
            self.media_player.set_xwindow(self.video_frame.winId())
        elif sys.platform == "win32":
            self.media_player.set_hwnd(self.video_frame.winId())
        elif sys.platform == "darwin":
            self.media_player.set_nsobject(int(self.video_frame.winId()))

        self.current_media = None

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if file_name != "":
            self.current_video_path = file_name  # simpan path asli
            # Set media baru
            self.current_media = self.instance.media_new(file_name)
            self.media_player.set_media(self.current_media)

            # Tampilkan nama video (maksimal 10 huruf)
            short_name = file_name.split("/")[-1]
            if len(short_name) > 10:
                short_name = short_name[:10] + "..."
            self.label_title.setText(short_name)
            self.info_label.setText(f"Status: Memutar {file_name.split('/')[-1][:10]}...")
            self.info_progress.setValue(0)

            # Langsung mainkan video
            self.media_player.play()
            self.slider.setEnabled(True)
            self.timer.start()

    def play_video(self):
        if self.current_media is not None:
            self.media_player.play()
            self.timer.start()

    def pause_video(self):
        self.media_player.pause()

    def stop_video(self):
        self.media_player.stop()
        
    def run_detection(self):
        if self.current_media is not None:
            self.info_label.setText("Status: Memproses video...")
            self.info_progress.setValue(0)
            QApplication.processEvents()

            def update_progress(value):
                self.info_progress.setValue(value)
                QApplication.processEvents()

            try:
                # Kirim callback ke fungsi deteksi
                jalankan_deteksi(self.current_video_path, progress_callback=update_progress)
                self.info_label.setText("Status: Deteksi selesai ✅")
            except Exception as e:
                self.info_label.setText(f"Error: {str(e)}")
        else:
            self.info_label.setText("Status: Tidak ada video untuk diproses!")
        
    def set_position(self, position):
        """Ubah posisi video berdasarkan slider"""
        if self.media_player.is_playing():
            self.media_player.set_position(position / 1000.0)

    def update_ui(self):
        """Update waktu dan posisi slider"""
        if self.media_player is not None:
            media_length = self.media_player.get_length()
            current_time = self.media_player.get_time()

            if media_length > 0:
                pos = int((current_time / media_length) * 1000)
                self.slider.blockSignals(True)
                self.slider.setValue(pos)
                self.slider.blockSignals(False)

                # Update label waktu
                self.label_current.setText(self.format_time(current_time))
                self.label_total.setText(self.format_time(media_length))

            # Hentikan timer kalau video selesai
            if self.media_player.get_state() == vlc.State.Ended:
                self.timer.stop()

    def format_time(self, ms):
        """Konversi milidetik ke format jj:mm:ss"""
        total_seconds = int(ms / 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
