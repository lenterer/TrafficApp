import sys
import os
import csv
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel,
    QSlider, QProgressBar, QGraphicsView, QGraphicsScene, QGraphicsLineItem, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QSizePolicy, QInputDialog, QMessageBox
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt5.QtCore import Qt, QUrl, QLineF, QSizeF, QPointF, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap

# --- Import fungsi deteksi ---
try:
    from deteksihitungz import jalankan_deteksi_dan_hitung
except ImportError:
    print("Warning: testhitung.py tidak ditemukan. Menggunakan dummy.")
    def jalankan_deteksi_dan_hitung(video_path, model_path, output_path, lines_coords, progress_callback):
        pass

# Daftar Kelas Valid (Harus sama dengan yang ada di testhitung.py)
VALID_CLASSES = ['1', '2', '3', '4', '5a', '5b', '6a', '6b', '7a', '7b', '7c', '8', 'Unknown']

class DetectionWorker(QThread):
    progress_changed = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, video_path, model_path, output_path, lines_coords):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.output_path = output_path
        self.line_coords = lines_coords

    def run(self):
        jalankan_deteksi_dan_hitung(
            self.video_path,
            model_path=self.model_path,
            output_path=self.output_path,
            lines_coords=self.line_coords,
            progress_callback=self.progress_changed.emit
        )
        self.finished.emit()

class MyVideoWidget(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.is_active = False
        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)
        self.overlay_lines = []
        self.start_point = None
        self.temp_line = None
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(Qt.black)
        self.line_colors = [QColor("#ff0000"), QColor("#00ff00"), QColor("#0000ff"), QColor("#ffff00"), QColor("#ff00ff"), QColor("#00ffff"), QColor("#ffa500")]
        self.next_color_index = 0
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def setMediaPlayer(self, player):
        player.setVideoOutput(self.video_item)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        new_size = QSizeF(event.size())
        self.video_item.setSize(new_size)
        self.scene.setSceneRect(0, 0, new_size.width(), new_size.height())

    def mousePressEvent(self, event):
        if not self.is_active: return
        if event.button() == Qt.LeftButton:
            self.start_point = self.mapToScene(event.pos())

    def mouseReleaseEvent(self, event):
        if not self.is_active: return
        if event.button() == Qt.LeftButton and self.start_point:
            if self.temp_line:
                self.scene.removeItem(self.temp_line)
                self.temp_line = None
            end_point = self.mapToScene(event.pos())
            rect = self.scene.sceneRect()
            end_point = QPointF(min(max(end_point.x(), rect.left()), rect.right()), min(max(end_point.y(), rect.top()), rect.bottom()))
            color = self.line_colors[self.next_color_index]
            self.next_color_index = (self.next_color_index + 1) % len(self.line_colors)
            line = QGraphicsLineItem(QLineF(self.start_point, end_point))
            line.setPen(QPen(color, 3))
            self.scene.addItem(line)
            self.overlay_lines.append(line)
            self.start_point = None

    def mouseMoveEvent(self, event):
        if not self.is_active or self.start_point is None: return
        current_point = self.mapToScene(event.pos())
        color = self.line_colors[self.next_color_index]
        if self.temp_line is None:
            self.temp_line = QGraphicsLineItem(QLineF(self.start_point, current_point))
            self.temp_line.setPen(QPen(color, 1, Qt.DashLine))
            self.scene.addItem(self.temp_line)
        else:
            self.temp_line.setLine(QLineF(self.start_point, current_point))

class VideoPlayer(QWidget):
    def __init__(self, stacked):
        super().__init__()
        self.stacked = stacked
        self.setGeometry(100, 100, 1100, 700)
        self.current_video_path = None
        self.is_showing_log = False 

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        lbl_small_logo = QLabel()
        pix_small = QPixmap("desain/logonobg.png").scaledToHeight(30, Qt.SmoothTransformation)
        lbl_small_logo.setPixmap(pix_small)
        lbl_small_logo.setStyleSheet("margin-right: 10px;") # Beri jarak
        
        # 🔹 Tambahkan tombol BACK
        btn_back = QPushButton("← Back")
        btn_back.setFixedHeight(45)
        btn_back.clicked.connect(self.go_back)
        self.layout.addWidget(btn_back, alignment=Qt.AlignLeft | Qt.AlignTop)

        self.label_title = QLabel("", self)
        self.label_title.setStyleSheet("color: white; background-color: #222; padding: 5px; font-size: 14px; font-weight: bold;")
        self.label_title.setFixedHeight(30)

        self.controls_layout = QHBoxLayout()
        self.btn_open = QPushButton("Open")
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        self.run_button = QPushButton("Run") 
        self.run_button.setStyleSheet("""QPushButton { background-color: #3498db; color: white; font-weight: bold; } QPushButton:hover { background-color: #2980b9; }""")
        self.btn_draw = QPushButton("Draw Line")
        self.btn_clear = QPushButton("Clear Line")

        for btn in [self.btn_open, self.btn_play, self.btn_pause, self.btn_stop, self.run_button, self.btn_draw, self.btn_clear]:
            # Hapus atau comment baris ini:
            # btn.setFixedSize(90, 30) 
            
            # Ganti dengan ini:
            btn.setMinimumHeight(40)  # Tinggi tetap
            btn.setMinimumWidth(80)   # Lebar minimal, tapi bisa melar jika teks panjang
            btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.controls_layout.addStretch()
        self.controls_layout.addWidget(self.btn_open)
        self.controls_layout.addWidget(self.btn_play)
        self.controls_layout.addWidget(self.btn_pause)
        self.controls_layout.addWidget(self.btn_stop)
        self.controls_layout.addWidget(self.run_button)
        self.controls_layout.addWidget(self.btn_draw)
        self.controls_layout.addWidget(self.btn_clear)
        self.controls_layout.addStretch()

        self.is_drawing_enabled = False
        self.btn_draw.setCheckable(True)
        self.btn_draw.clicked.connect(self.toggle_draw_mode)
        self.btn_clear.clicked.connect(self.clear_lines)

        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.video_widget = MyVideoWidget()
        self.video_widget.setMediaPlayer(self.media_player)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.progress_layout = QHBoxLayout()
        self.label_current = QLabel("00:00:00")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setEnabled(False)
        self.label_total = QLabel("00:00:00")
        self.progress_layout.addWidget(self.label_current)
        self.progress_layout.addWidget(self.slider, stretch=1)
        self.progress_layout.addWidget(self.label_total)

        # --- Panel Info ---
        self.info_panel = QVBoxLayout()
        self.lbl_info_header = QLabel("DETECTION INFORMATION")
        self.lbl_info_header.setAlignment(Qt.AlignCenter)
        self.lbl_info_header.setStyleSheet("font-weight: bold; color: white; margin-bottom: 5px;")
        self.lbl_info_header.setFixedHeight(25)

        self.info_label = QLabel("Status: Waiting for video...")
        self.info_label.setStyleSheet("font-size: 20px; color: white; background-color: #333; padding: 10px; border-radius: 4px;")
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.info_label.setWordWrap(True)
        self.info_label.setMinimumHeight(100) 

        # --- Tabel Log Detail (MODIFIKASI: Tambah Kolom Aksi) ---
        self.table_csv = QTableWidget()
        self.table_csv.setColumnCount(5) # Waktu, Kelas, ID, Garis, Edit
        self.table_csv.setHorizontalHeaderLabels(["Waktu", "Kelas", "ID", "Garis", "Aksi"])
        self.table_csv.setStyleSheet("""
            QTableWidget { background-color: #222; color: white; gridline-color: #444; font-size: 11px; }
            QHeaderView::section { background-color: #444; color: white; padding: 4px; }
            QTableWidget::item:selected { background-color: #e67e22; color: white; }
        """)
        self.table_csv.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_csv.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents) # Kolom aksi pas konten
        self.table_csv.verticalHeader().setVisible(False)
        self.table_csv.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_csv.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_csv.cellClicked.connect(self.seek_video_from_table)
        self.table_csv.setVisible(False) 

        self.info_progress = QProgressBar()
        self.info_progress.setRange(0, 100)
        self.info_progress.setValue(0)
        self.info_progress.setFixedHeight(15)

        self.info_panel.addWidget(self.lbl_info_header)
        self.info_panel.addWidget(self.info_label)
        self.info_panel.addWidget(self.table_csv)
        self.info_panel.addWidget(self.info_progress)

        self.video_info_layout = QHBoxLayout()
        self.video_info_layout.addWidget(self.video_widget, stretch=7)
        self.video_info_layout.addLayout(self.info_panel, stretch=3)

        self.layout.addWidget(self.label_title)
        self.layout.addLayout(self.video_info_layout)
        self.layout.addLayout(self.progress_layout)
        self.layout.addLayout(self.controls_layout)

        self.btn_open.clicked.connect(self.open_file)
        self.btn_play.clicked.connect(self.play_video)
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_stop.clicked.connect(self.stop_video)
        self.run_button.clicked.connect(self.run_detection)
        self.slider.sliderMoved.connect(self.set_position)
        self.media_player.positionChanged.connect(self.update_position)
        self.media_player.durationChanged.connect(self.update_duration)
        self.media_player.stateChanged.connect(self.handle_state_changed)

    def go_back(self):
        self.stacked.setCurrentIndex(0)
    
    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if file_name != "":
            self.current_video_path = file_name
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_name)))
            
            short_name = file_name.split("/")[-1]
            self.label_title.setText(short_name)
            self.info_progress.setValue(0)
            self.slider.setEnabled(True)
            self.play_video()
            self.reset_info_view()
            self.load_csv_summary(file_name)
            self.check_and_update_button(file_name)

    def reset_info_view(self):
        self.is_showing_log = False
        self.info_label.setVisible(True)
        self.table_csv.setVisible(False)
        self.lbl_info_header.setText("RINGKASAN HASIL")

    def check_and_update_button(self, video_path):
        base_name = os.path.splitext(video_path)[0]
        log_path = f"{base_name}_log_detail.csv"
        try: self.run_button.clicked.disconnect()
        except TypeError: pass 
        if os.path.exists(log_path):
            self.run_button.setText("View Log")
            self.run_button.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
            self.run_button.clicked.connect(self.toggle_log_view)
        else:
            self.run_button.setText("Run")
            self.run_button.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
            self.run_button.clicked.connect(self.run_detection)
            self.info_label.setText("Status: Video siap diproses.\nSilakan gambar garis lalu klik Run.")

    def toggle_log_view(self):
        if not self.is_showing_log:
            self.load_log_to_table()
            self.info_label.setVisible(False)
            self.table_csv.setVisible(True)
            self.run_button.setText("Ringkasan")
            self.run_button.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold;")
            self.lbl_info_header.setText("LOG DETAIL KENDARAAN")
            self.is_showing_log = True
        else:
            self.load_csv_summary(self.current_video_path) # Reload ringkasan terbaru
            self.table_csv.setVisible(False)
            self.info_label.setVisible(True)
            self.run_button.setText("Lihat Log")
            self.run_button.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
            self.lbl_info_header.setText("RINGKASAN HASIL")
            self.is_showing_log = False

    def load_csv_summary(self, video_path):
        base_video_path = os.path.splitext(video_path)[0]
        csv_path = base_video_path + "_ringkasan.csv"
        if not os.path.exists(csv_path): return
        try:
            text_output = ""
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                text_output += "<b>TOTAL DETEKSI:</b><br>"
                for line in lines:
                    line = line.strip()
                    if "TOTAL PER KELAS" in line: continue
                    if "TOTAL PER GARIS" in line:
                        text_output += "<br><b>PER GARIS:</b><br>"
                        continue
                    if not line or "Kelas,Jumlah" in line: continue
                    parts = line.split(',')
                    if len(parts) == 2:
                        if parts[0] == "Total Semua": text_output += f"<br><b>TOTAL: {parts[1]}</b><br>"
                        else: text_output += f"- {parts[0]}: {parts[1]}<br>"
            self.info_label.setText(text_output)
        except Exception as e:
            self.info_label.setText(f"Gagal baca ringkasan: {e}")

    def load_log_to_table(self):
        if not self.current_video_path: return
        base_name = os.path.splitext(self.current_video_path)[0]
        log_path = f"{base_name}_log_detail.csv"
        if not os.path.exists(log_path): return

        self.table_csv.setRowCount(0)
        try:
            row_data_list = []
            with open(log_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    waktu = row.get('Waktu Video', '')
                    kelas = row.get('Jenis Kendaraan', '')
                    oid = row.get('ID Objek', '')
                    garis = row.get('Garis', '')
                    # Simpan index baris asli di CSV untuk keperluan update nanti
                    row_data_list.append({'data': (waktu, kelas, oid, garis), 'raw': row})
            
            self.table_csv.setRowCount(len(row_data_list))
            for i, item in enumerate(row_data_list):
                waktu, kelas, oid, garis = item['data']
                
                # --- Kolom Data ---
                item_waktu = QTableWidgetItem(waktu)
                ms_val = self.parse_timestamp_to_ms(waktu)
                item_waktu.setData(Qt.UserRole, ms_val)
                self.table_csv.setItem(i, 0, item_waktu)
                self.table_csv.setItem(i, 1, QTableWidgetItem(kelas))
                self.table_csv.setItem(i, 2, QTableWidgetItem(oid))
                self.table_csv.setItem(i, 3, QTableWidgetItem(garis))

                # --- Kolom Tombol Edit (MODIFIKASI BARU) ---
                btn_edit = QPushButton("Edit")
                btn_edit.setStyleSheet("background-color: #f39c12; color: white; border: none; padding: 2px;")
                # Gunakan lambda dengan argumen default untuk menangkap nilai i saat ini
                btn_edit.clicked.connect(lambda checked, row=i: self.edit_vehicle_class(row))
                self.table_csv.setCellWidget(i, 4, btn_edit)

        except Exception as e:
            print(f"Error loading table: {e}")

    def parse_timestamp_to_ms(self, time_str):
        try:
            parts = time_str.split(':')
            if len(parts) == 3:
                h = int(parts[0]); m = int(parts[1]); s = float(parts[2])
                return int((h*3600 + m*60 + s) * 1000)
        except: return 0
        return 0

    def seek_video_from_table(self, row, col):
        # Jika yang diklik kolom tombol (kolom 4), jangan seek (biar tombol yang handle)
        if col == 4: return
        item = self.table_csv.item(row, 0)
        if item:
            ms = item.data(Qt.UserRole)
            if ms is not None:
                self.media_player.setPosition(ms)
                self.media_player.pause()
                self.update_position(ms)

    # =========================================================
    # FITUR EDIT DAN SIMPAN (BARU)
    # =========================================================
    
    def edit_vehicle_class(self, row_idx):
        """Munculkan dialog edit kelas."""
        # Pause video saat mengedit
        self.media_player.pause()

        # Ambil data saat ini
        current_class = self.table_csv.item(row_idx, 1).text()
        current_id = self.table_csv.item(row_idx, 2).text()
        
        # Cari index kelas saat ini di daftar valid
        current_idx = 0
        if current_class in VALID_CLASSES:
            current_idx = VALID_CLASSES.index(current_class)

        # Tampilkan Dialog
        item, ok = QInputDialog.getItem(
            self, "Edit Kelas Kendaraan", 
            f"Ubah kelas untuk ID {current_id}:", 
            VALID_CLASSES, current_idx, False
        )

        if ok and item:
            if item != current_class:
                # Update Tabel UI
                self.table_csv.item(row_idx, 1).setText(item)
                
                # Update CSV dan Hitung Ulang Summary
                self.update_csv_files(row_idx, item)
                QMessageBox.information(self, "Sukses", "Data berhasil diubah dan disimpan.")

    def update_csv_files(self, row_idx, new_class):
        """Mengupdate log CSV dan menghitung ulang summary."""
        if not self.current_video_path: return

        base_name = os.path.splitext(self.current_video_path)[0]
        log_path = f"{base_name}_log_detail.csv"
        summary_path = f"{base_name}_ringkasan.csv"

        # 1. BACA SEMUA DATA LOG
        rows = []
        header = []
        try:
            with open(log_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader) # Simpan header
                rows = list(reader)
        except Exception as e:
            print(f"Error reading log for update: {e}")
            return

        # 2. UPDATE BARIS YANG SESUAI (Header: Garis, Jenis Kendaraan, ID Objek, Waktu Video, Frame)
        # Asumsi urutan di tabel UI sama dengan urutan baris di CSV (karena kita load berurutan)
        # Kolom 'Jenis Kendaraan' biasanya index ke-1 (cek header di testhitung.py)
        
        # Kita cari index kolom 'Jenis Kendaraan' di header
        try:
            cls_col_idx = header.index('Jenis Kendaraan')
        except ValueError:
            cls_col_idx = 1 # Fallback default

        if row_idx < len(rows):
            rows[row_idx][cls_col_idx] = new_class

        # 3. TULIS ULANG FILE LOG
        try:
            with open(log_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
        except Exception as e:
            print(f"Error writing updated log: {e}")
            return

        # 4. HITUNG ULANG SUMMARY (RINGKASAN)
        counts_total = {cls: 0 for cls in VALID_CLASSES}
        # Bersihkan counts agar tidak ada key sisa
        
        # Kita juga perlu menghitung per garis, cari index kolom 'Garis'
        try:
            line_col_idx = header.index('Garis')
        except ValueError:
            line_col_idx = 0

        counts_per_line = {} # Dictionary dinamis karena nama garis bisa apa saja

        for row in rows:
            c_cls = row[cls_col_idx]
            c_line = row[line_col_idx]
            
            # Hitung Total Kelas
            if c_cls in counts_total:
                counts_total[c_cls] += 1
            else:
                # Jika ada kelas aneh (hasil edit manual user di luar daftar), tambahkan
                counts_total[c_cls] = counts_total.get(c_cls, 0) + 1
            
            # Hitung Per Garis
            counts_per_line[c_line] = counts_per_line.get(c_line, 0) + 1

        # 5. TULIS ULANG FILE RINGKASAN
        try:
            with open(summary_path, mode="w", newline="", encoding="utf-8") as file:
                writer_summary = csv.writer(file)
                
                writer_summary.writerow(["=== RINGKASAN TOTAL PER KELAS ==="])
                writer_summary.writerow(["Kelas", "Jumlah"])
                total_kendaraan = 0
                
                # Urutkan berdasarkan VALID_CLASSES agar rapi
                for cls in VALID_CLASSES:
                    val = counts_total.get(cls, 0)
                    writer_summary.writerow([cls, val])
                    total_kendaraan += val
                
                # Tulis kelas lain yang mungkin muncul (tidak ada di VALID_CLASSES)
                for cls, val in counts_total.items():
                    if cls not in VALID_CLASSES:
                        writer_summary.writerow([cls, val])
                        total_kendaraan += val

                writer_summary.writerow(["Total Semua", total_kendaraan])
                
                writer_summary.writerow([]) 
                writer_summary.writerow(["=== RINGKASAN TOTAL PER GARIS ==="])
                
                # Urutkan garis (Garis 1, Garis 2...)
                sorted_lines = sorted(counts_per_line.keys())
                for line_key in sorted_lines:
                    writer_summary.writerow([line_key, counts_per_line[line_key]])

            print("Summary updated successfully.")
        except Exception as e:
            print(f"Error updating summary: {e}")

    def run_detection(self):
        if self.current_video_path is None: return

        # 1. Ambil koordinat garis
        coords_list = self.get_all_line_coords()
        total_lines = len(coords_list)
        
        if not coords_list:
            QMessageBox.warning(self, "Peringatan", "Gambar garis deteksi terlebih dahulu!")
            return

        # 2. Hitung Estimasi Waktu (Hanya perkiraan)
        cap = cv2.VideoCapture(self.current_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        duration_sec = frame_count / fps if fps > 0 else 0
        est_time = duration_sec  # Anggap proses realtime (bisa disesuaikan)
        est_min = int(est_time // 60)
        est_sec = int(est_time % 60)

        # 3. Tampilkan Popup Konfirmasi
        msg = QMessageBox()
        msg.setWindowTitle("Konfirmasi Deteksi")
        msg.setText(
            f"Jumlah garis deteksi: {total_lines}\n"
            f"Estimasi waktu proses: +/- {est_min} menit {est_sec} detik\n\n"
            f"Mulai pemrosesan deteksi?"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        result = msg.exec_()

        if result != QMessageBox.Yes:
            return  # Batal jika user klik No

        # Update UI
        self.info_label.setText("Status: Memproses video... Mohon tunggu.")
        self.info_progress.setValue(0)
        self.run_button.setEnabled(False) # Matikan tombol agar tidak di-klik double
        
        # ==============================================================
        # BAGIAN YANG HILANG SEBELUMNYA (FIX)
        # ==============================================================
        
        # A. Format Koordinat
        # Fungsi di backend (testhitung.py) butuh list tuple [(x1,y1,x2,y2), ...], 
        # sedangkan get_all_line_coords mengembalikan list of dictionary.
        # Kita perlu mengekstraknya:
        raw_lines = []
        for item in coords_list:
            x1, y1, x2, y2 = item['coords']
            # Paksa ubah jadi integer (bilangan bulat)
            raw_lines.append((int(x1), int(y1), int(x2), int(y2)))

        # B. Tentukan Path
        # Pastikan path model benar. Sesuaikan jika nama file model Anda beda.
        model_path = "model/best.pt"  
        
        # Path output video
        base_name = os.path.splitext(self.current_video_path)[0]
        output_path = f"{base_name}_hasil.mp4"

        # C. Inisialisasi Thread Worker
        # Kita simpan di self.worker agar garbage collector tidak menghapusnya
        self.worker = DetectionWorker(self.current_video_path, model_path, output_path, raw_lines)
        
        # D. Hubungkan Signal (Events)
        self.worker.progress_changed.connect(self.info_progress.setValue)
        self.worker.finished.connect(self.on_detection_finished)
        
        # E. Start Thread
        self.worker.start()

    def on_detection_finished(self):
        self.info_label.setText("Selesai! Klik 'Lihat Log' untuk melihat detail.")
        self.info_progress.setValue(100)
        if self.current_video_path:
            self.load_csv_summary(self.current_video_path)
            self.check_and_update_button(self.current_video_path)

    def play_video(self): self.media_player.play()
    def pause_video(self): self.media_player.pause()
    def stop_video(self): self.media_player.stop()
    def set_position(self, pos): self.media_player.setPosition(int(self.media_player.duration() * (pos/1000)))
    def update_position(self, pos):
        if self.media_player.duration() > 0:
            self.slider.blockSignals(True)
            self.slider.setValue(int((pos/self.media_player.duration())*1000))
            self.slider.blockSignals(False)
            self.label_current.setText(self.format_time(pos))
    def update_duration(self, dur):
        self.slider.setEnabled(dur > 0)
        self.label_total.setText(self.format_time(dur))
    def handle_state_changed(self, s):
        if s == QMediaPlayer.StoppedState: 
            self.slider.setValue(0); self.label_current.setText("00:00:00")
    def format_time(self, ms):
        s = ms // 1000; m = (s % 3600) // 60; h = s // 3600; s = s % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
    def toggle_draw_mode(self):
        self.is_drawing_enabled = self.btn_draw.isChecked()
        self.video_widget.is_active = self.is_drawing_enabled
        self.btn_draw.setText("Drawing: ON" if self.is_drawing_enabled else "Draw Line")
    def clear_lines(self):
        for l in list(self.video_widget.overlay_lines): self.video_widget.scene.removeItem(l)
        self.video_widget.overlay_lines.clear()
    def get_all_line_coords(self):
        coords = []
        for idx, line_item in enumerate(self.video_widget.overlay_lines):
            line = line_item.line()
            color = line_item.pen().color().name()
            coords.append({"index": idx, "coords": (line.x1(), line.y1(), line.x2(), line.y2()), "color": color})
        return coords
    def closeEvent(self, event):
        self.media_player.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())