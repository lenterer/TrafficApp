import sys
import os
# --- PERBAIKAN 1: Menambahkan ini untuk stabilitas threading ---
os.environ["OMP_NUM_THREADS"] = "1" 
import vlc
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, 
    QSlider, QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot

# Import library untuk deteksi
try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Error: Library yang dibutuhkan tidak ditemukan -> {e}")
    print("Pastikan Anda sudah menginstal: pip install ultralytics opencv-python numpy python-vlc")
    sys.exit()

# ==============================================================================
# BAGIAN 1: LOGIKA DETEKSI DAN PENGHITUNGAN (BACKEND)
# ==============================================================================

def jalankan_deteksi_dan_hitung(video_path, model_path, output_path="hasil_deteksi.mp4", progress_callback=None):
    """
    Memproses video, menyimpan hasilnya, MENGHITUNG kendaraan, dan me-return hasil hitungan.
    """
    try:
        CLASS_NAMES = ['1', '2', '3', '4', '5a', '5b', '6a', '6b', '7a', '7b', '7c', '8']
        
        # --- PERBAIKAN 2: Tambahkan print untuk memberi tahu model sedang dimuat ---
        print("Memuat model YOLO... Ini mungkin butuh waktu...")
        model = YOLO(model_path)
        print("Model berhasil dimuat.")
        
        cap = cv2.VideoCapture(video_path)
        model.overrides["verbose"] = False 

        # Info video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            raise ValueError("Tidak bisa membaca video atau video kosong.")

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        # --- Inisialisasi Logika Counting ---
        LINE_RED_Y = 380   # SESUAIKAN
        LINE_GREEN_Y = 420 # SESUAIKAN

        object_history = {}
        counted_ids_down = set()
        counts_top_to_bottom = {class_name: 0 for class_name in CLASS_NAMES}
        counted_ids_up = set()
        counts_bottom_to_top = {class_name: 0 for class_name in CLASS_NAMES}
        
        frame_count = 0
        
        # --- PERBAIKAN 3: Variabel untuk melacak progres terakhir ---
        last_percent = -1 
        
        print(f"Mulai deteksi, tracking, dan counting: {video_path} ({total_frames} frame)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            
            annotated_frame = frame.copy() 

            cv2.line(annotated_frame, (0, LINE_RED_Y), (width, LINE_RED_Y), (0, 0, 255), 2)
            cv2.line(annotated_frame, (0, LINE_GREEN_Y), (width, LINE_GREEN_Y), (0, 255, 0), 2)

            if results[0].boxes.id is not None:
                for box in results[0].boxes:
                    confidence = box.conf[0]
                    if confidence > 0.1:
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = coords
                        class_id = int(box.cls[0])
                        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "Unknown"
                        obj_id = int(box.id[0])
                        current_y_ref = y2

                        if obj_id in object_history:
                            prev_y_ref = object_history[obj_id]
                            
                            if (prev_y_ref < LINE_RED_Y) and (current_y_ref >= LINE_RED_Y):
                                if obj_id not in counted_ids_down:
                                    counts_top_to_bottom[class_name] += 1
                                    counted_ids_down.add(obj_id)
                                    cv2.line(annotated_frame, (0, LINE_RED_Y), (width, LINE_RED_Y), (255, 255, 255), 3)

                            if (prev_y_ref > LINE_GREEN_Y) and (current_y_ref <= LINE_GREEN_Y):
                                if obj_id not in counted_ids_up:
                                    counts_bottom_to_top[class_name] += 1
                                    counted_ids_up.add(obj_id)
                                    cv2.line(annotated_frame, (0, LINE_GREEN_Y), (width, LINE_GREEN_Y), (255, 255, 255), 3)

                        object_history[obj_id] = current_y_ref
                        
                        label = f'ID: {obj_id} {class_name}'
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            out.write(annotated_frame)
            frame_count += 1
            
            if total_frames > 0:
                percent = int((frame_count / total_frames) * 100)
                
                # --- PERBAIKAN 4: Hanya kirim sinyal jika persen berubah ---
                if percent > last_percent:
                    if progress_callback:
                        progress_callback(percent)
                    last_percent = percent
                    
                # Update console bisa tetap jalan
                sys.stdout.write(f"\r[{percent:3d}%] Memproses frame {frame_count}/{total_frames}")
                sys.stdout.flush()

        print("\n✅ Deteksi dan Counting selesai. Hasil disimpan di:", output_path)
        
        if progress_callback:
            progress_callback(100) # Pastikan 100% terkirim
            
        return counts_top_to_bottom, counts_bottom_to_top

    except Exception as e:
        print(f"\nTerjadi error saat deteksi: {e}")
        return None, None
    
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals() and out.isOpened():
            out.release()

# ==============================================================================
# BAGIAN 2: THREAD WORKER UNTUK MENJALANKAN DETEKSI
# ==============================================================================

class DetectionThread(QThread):
    progress_updated = pyqtSignal(int)
    counting_finished = pyqtSignal(dict, dict)
    processing_finished = pyqtSignal(str)
    processing_error = pyqtSignal(str)

    def __init__(self, video_path, model_path):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.output_path = "hasil_deteksi.mp4"

    def run(self):
        try:
            counts_down, counts_up = jalankan_deteksi_dan_hitung(
                self.video_path, 
                self.model_path, 
                self.output_path,
                progress_callback=self.progress_updated.emit
            )
            
            self.processing_finished.emit(self.output_path)
            
            if counts_down is not None and counts_up is not None:
                self.counting_finished.emit(counts_down, counts_up)
            
        except Exception as e:
            self.processing_error.emit(f"Error pada thread: {str(e)}")

# ==============================================================================
# BAGIAN 3: GUI UTAMA (FRONTEND)
# ==============================================================================

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrafficApp")
        self.setGeometry(100, 100, 1200, 800)
        self.current_video_path = None
        self.model_path = 'C:/HDD/TUGAS KULIAH/TrafficCountYOLO/Protel/TrafficApp/Model1/best.pt'

        # Layout utama
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label_title = QLabel("Silakan Buka File Video", self)
        self.label_title.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 80); padding: 5px; font-size: 14px;")
        self.label_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label_title.setFixedHeight(30)
        
        self.controls_layout = QHBoxLayout()
        self.btn_open = QPushButton("Open")
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        self.run_button = QPushButton("Run Detection")
        self.run_button.setEnabled(False) # Awalnya nonaktif
        self.run_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

        for btn in [self.btn_open, self.btn_play, self.btn_pause, self.btn_stop, self.run_button]:
            btn.setFixedSize(100, 30)

        self.controls_layout.addStretch()
        self.controls_layout.addWidget(self.btn_open)
        self.controls_layout.addWidget(self.btn_play)
        self.controls_layout.addWidget(self.btn_pause)
        self.controls_layout.addWidget(self.btn_stop)
        self.controls_layout.addWidget(self.run_button)
        self.controls_layout.addStretch()

        self.instance = vlc.Instance()
        self.media_player = self.instance.media_player_new()
        
        self.progress_layout = QHBoxLayout()
        self.label_current = QLabel("00:00:00")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setEnabled(False)
        self.label_total = QLabel("00:00:00")

        self.progress_layout.addWidget(self.label_current)
        self.progress_layout.addWidget(self.slider, stretch=1)
        self.progress_layout.addWidget(self.label_total)

        self.video_info_layout = QHBoxLayout()
        self.video_frame = QWidget(self)
        self.video_frame.setStyleSheet("background-color: black;")

        # --- Panel Informasi di Kanan (di-upgrade) ---
        self.info_panel = QVBoxLayout()
        self.info_panel.setContentsMargins(10, 0, 10, 0)
        
        # Simpan daftar kelas untuk referensi
        self.CLASS_NAMES = ['1', '2', '3', '4', '5a', '5b', '6a', '6b', '7a', '7b', '7c', '8']

        self.info_label = QLabel("Status: Idle")
        self.info_label.setStyleSheet("font-size: 14px; color: white; background-color: #333; padding: 6px; border-radius: 4px;")
        
        self.info_progress = QProgressBar()
        self.info_progress.setRange(0, 100)
        self.info_progress.setValue(0)
        self.info_progress.setTextVisible(True)
        
        # --- Label untuk hasil counting (DIMODIFIKASI) ---
        
        # Helper function untuk membuat teks default
        def get_default_text(class_names):
            parts = [f"Gol. {name}: 0" for name in class_names]
            return "\n".join(parts)
        
        default_text = get_default_text(self.CLASS_NAMES)

        # Judul untuk Arah Bawah
        self.label_title_down = QLabel("Total Arah Bawah (Merah):")
        self.label_title_down.setStyleSheet("font-size: 14px; color: #FF5733; font-weight: bold; margin-top: 10px;")

        # Daftar hitungan Arah Bawah
        self.count_down_label = QLabel(default_text)
        self.count_down_label.setStyleSheet("font-size: 12px; color: black; padding-left: 5px;")
        self.count_down_label.setAlignment(Qt.AlignTop) # Penting agar rata atas
        
        # Judul untuk Arah Atas
        self.label_title_up = QLabel("Total Arah Atas (Hijau):")
        self.label_title_up.setStyleSheet("font-size: 14px; color: #33FF57; font-weight: bold; margin-top: 10px;")

        # Daftar hitungan Arah Atas
        self.count_up_label = QLabel(default_text)
        self.count_up_label.setStyleSheet("font-size: 12px; color: black; padding-left: 5px;")
        self.count_up_label.setAlignment(Qt.AlignTop) # Penting agar rata atas

        # Masukkan widget baru ke layout panel info
        self.info_panel.addWidget(self.info_label)
        self.info_panel.addWidget(self.info_progress)
        self.info_panel.addWidget(self.label_title_down)  # <-- Baru
        self.info_panel.addWidget(self.count_down_label)  # <-- Dimodifikasi
        self.info_panel.addWidget(self.label_title_up)    # <-- Baru
        self.info_panel.addWidget(self.count_up_label)    # <-- Dimodifikasi
        self.info_panel.addStretch() # Mendorong widget ke atas
        # ----------------------------------------------

        self.video_info_layout.addWidget(self.video_frame, stretch=3)
        self.video_info_layout.addLayout(self.info_panel, stretch=1)

        self.layout.addWidget(self.label_title)
        self.layout.addLayout(self.video_info_layout, stretch=1)
        self.layout.addLayout(self.progress_layout)
        self.layout.addLayout(self.controls_layout)

        self.btn_open.clicked.connect(self.open_file)
        self.btn_play.clicked.connect(self.play_video)
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_stop.clicked.connect(self.stop_video)
        self.run_button.clicked.connect(self.run_detection_thread)
        self.slider.sliderMoved.connect(self.set_position)

        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.update_ui)

        if sys.platform.startswith("linux"):
            self.media_player.set_xwindow(self.video_frame.winId())
        elif sys.platform == "win32":
            self.media_player.set_hwnd(self.video_frame.winId())
        elif sys.platform == "darwin":
            self.media_player.set_nsobject(int(self.video_frame.winId()))

        self.current_media = None

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.current_video_path = file_name
            self.current_media = self.instance.media_new(file_name)
            self.media_player.set_media(self.current_media)

            short_name = os.path.basename(file_name)
            self.label_title.setText(short_name)
            self.info_label.setText(f"Status: Siap memutar {short_name[:20]}...")
            
            # Reset UI
            self.info_progress.setValue(0)
            
            # --- Reset label hitungan (DIMODIFIKASI dari kode Anda sebelumnya) ---
            def get_default_text(class_names):
                parts = [f"Gol. {name}: 0" for name in class_names]
                return "\n".join(parts)
            
            default_text = get_default_text(self.CLASS_NAMES)
            self.count_down_label.setText(default_text)
            self.count_up_label.setText(default_text)
            # ------------------------------------------
            
            # Aktifkan tombol Run
            self.run_button.setEnabled(True)

            self.slider.setEnabled(True)
            
            self.play_video()

    def play_video(self):
        if self.current_media is not None:
            self.media_player.play()
            self.timer.start()

    def pause_video(self):
        self.media_player.pause()

    def stop_video(self):
        self.media_player.stop()
        self.timer.stop()
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.label_current.setText("00:00:00")
        
    def run_detection_thread(self):
        if not self.current_video_path:
            QMessageBox.warning(self, "Peringatan", "Silakan buka file video terlebih dahulu.")
            return

        # Nonaktifkan tombol untuk mencegah klik ganda
        self.run_button.setEnabled(False)
        self.btn_open.setEnabled(False)

        # Reset UI
        self.info_label.setText("Status: Memproses video...")
        self.info_progress.setValue(0)
        
        # --- Reset label hitungan (DIMODIFIKASI) ---
        def get_default_text(class_names):
            parts = [f"Gol. {name}: 0" for name in class_names]
            return "\n".join(parts)
        
        default_text = get_default_text(self.CLASS_NAMES)
        self.count_down_label.setText(default_text)
        self.count_up_label.setText(default_text)
        # ------------------------------------------

        # Buat dan jalankan thread
        self.detection_thread = DetectionThread(self.current_video_path, self.model_path)
        
        # Hubungkan sinyal dari thread ke slot di GUI
        self.detection_thread.progress_updated.connect(self.update_progress)
        self.detection_thread.counting_finished.connect(self.update_final_counts)
        self.detection_thread.processing_finished.connect(self.on_processing_finished)
        self.detection_thread.processing_error.connect(self.on_processing_error)
        
        self.detection_thread.start()

    @pyqtSlot(int)
    def update_progress(self, value):
        self.info_progress.setValue(value)

    @pyqtSlot(dict, dict)
    def update_final_counts(self, counts_down, counts_up):
        """
        Menerima hasil hitungan akhir dari thread dan memformatnya 
        menjadi daftar per golongan.
        """
        
        # --- Helper function untuk format teks ---
        def format_counts_text(class_names_list, counts_dict):
            parts = []
            for class_name in class_names_list:
                # Ambil hitungan dari dict, default 0 jika tidak ada
                count = counts_dict.get(class_name, 0)
                parts.append(f"Gol. {class_name}: {count}")
            # Gabungkan dengan newline
            return "\n".join(parts)
        # ----------------------------------------
        
        # Buat teks untuk Arah Bawah
        text_down = format_counts_text(self.CLASS_NAMES, counts_down)
        self.count_down_label.setText(text_down)
        
        # Buat teks untuk Arah Atas
        text_up = format_counts_text(self.CLASS_NAMES, counts_up)
        self.count_up_label.setText(text_up)
        
        print("--- Hasil Akhir Perhitungan (UI) ---")
        print(f"Arah Bawah:\n{text_down}")
        print(f"\nArah Atas:\n{text_up}")

    @pyqtSlot(str)
    def on_processing_finished(self, output_path):
        self.info_label.setText("Status: Deteksi Selesai ✅")
        QMessageBox.information(self, "Selesai", f"Video berhasil diproses!\nOutput disimpan di:\n{os.path.abspath(output_path)}")
        self.run_button.setEnabled(True)
        self.btn_open.setEnabled(True)
        # Tawarkan untuk memutar video hasil
        reply = QMessageBox.question(self, 'Putar Hasil', 'Apakah Anda ingin memutar video hasil deteksi?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.current_video_path = os.path.abspath(output_path)
            self.current_media = self.instance.media_new(self.current_video_path)
            self.media_player.set_media(self.current_media)
            self.label_title.setText(os.path.basename(self.current_video_path))
            self.play_video()

    @pyqtSlot(str)
    def on_processing_error(self, message):
        self.info_label.setText("Status: Error!")
        QMessageBox.critical(self, "Error", message)
        self.run_button.setEnabled(True)
        self.btn_open.setEnabled(True)

    def set_position(self, position):
        if self.current_media is not None:
            self.media_player.set_position(position / 1000.0)

    def update_ui(self):
        if self.media_player is not None:
            media_length = self.media_player.get_length()
            current_time = self.media_player.get_time()

            if media_length > 0:
                pos = int((current_time / media_length) * 1000)
                self.slider.blockSignals(True)
                self.slider.setValue(pos)
                self.slider.blockSignals(False)
                
                self.label_current.setText(self.format_time(current_time))
                self.label_total.setText(self.format_time(media_length))

            if self.media_player.get_state() == vlc.State.Ended:
                self.timer.stop()
                self.slider.setEnabled(False)

    def format_time(self, ms):
        total_seconds = int(ms / 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def closeEvent(self, event):
        """Pastikan thread berhenti saat jendela ditutup."""
        if hasattr(self, 'detection_thread') and self.detection_thread.isRunning():
            self.detection_thread.quit() # Minta thread untuk berhenti
            self.detection_thread.wait() # Tunggu sampai benar-benar berhenti
        event.accept()

if __name__ == "__main__":
    # Perbaikan bug VLC di beberapa sistem
    if sys.platform == "win32":
        os.add_dll_directory(os.path.dirname(vlc.__file__))
        
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())