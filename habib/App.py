import sys
import os
import csv
import cv2
import io 
import pandas as pd
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel,
    QSlider, QProgressBar, QGraphicsView, QGraphicsScene, QGraphicsLineItem, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QSizePolicy, QInputDialog, QMessageBox, QComboBox
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt5.QtCore import Qt, QUrl, QLineF, QSizeF, QPointF, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor

# --- Import fungsi deteksi ---
try:
    from deteksihitung import jalankan_deteksi_dan_hitung
except ImportError:
    print("File deteksihitung tidak ditemukan")
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

class CanvasGrafik(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=80):
        # Membuat Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        
        # --- UBAH BACKGROUND FIG (LUAR) KE GELAP ---
        self.fig.patch.set_facecolor('#1e1e1e') 
        
        # --- KEMBALI KE LAYOUT AWAL (Split 3:1) ---
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 1])
        
        self.ax_grafik = self.fig.add_subplot(gs[0, 0]) # Axis Kiri
        self.ax_teks = self.fig.add_subplot(gs[0, 1])   # Axis Kanan
        
        # --- UBAH BACKGROUND AXES (DALAM) KE GELAP ---
        self.ax_grafik.set_facecolor('#1e1e1e')
        self.ax_teks.set_facecolor('#1e1e1e')

        super(CanvasGrafik, self).__init__(self.fig)
        self.setMinimumHeight(200)
        
        # Setup Interaksi Mouse
        self.bar_containers = []
        self.annot = None
        self.mpl_connect("motion_notify_event", self.on_hover)

    def update_data(self, filename):
        try:
            # 1. BACA FILE
            raw_data = ""
            is_reading = False
            
            if not os.path.exists(filename): return

            with open(filename, 'r') as f:
                lines = f.readlines()

            for line in lines:
                clean_line = line.strip()
                if "=== DETAIL PER GARIS & KELAS ===" in clean_line:
                    is_reading = True
                    continue
                if is_reading:
                    if "===" in clean_line or clean_line == "":
                        if len(raw_data) > 0: break 
                        else: continue
                    raw_data += line

            if not raw_data:
                self.ax_grafik.clear()
                self.ax_teks.clear()
                # Teks Putih
                self.ax_grafik.text(0.5, 0.5, "Data Kosong", ha='center', fontname='Segoe UI', color='white')
                self.draw()
                return

            # 2. PROSES DATA
            df_raw = pd.read_csv(io.StringIO(raw_data), sep=',') 
            df_raw.columns = df_raw.columns.str.strip()
            df = df_raw.pivot_table(index='Kelas', columns='Garis', values='Jumlah')
            df = df.fillna(0)

            self.kategori = df.index
            self.nama_kolom_data = df.columns
            
            jumlah_baris = len(self.kategori)
            jumlah_data_series = len(self.nama_kolom_data)

            # 3. GAMBAR GRAFIK (DI KIRI)
            self.ax_grafik.clear()
            self.ax_teks.clear()
            self.bar_containers = []
            
            # Reset warna background setelah clear()
            self.ax_grafik.set_facecolor('#1e1e1e')
            self.ax_teks.set_facecolor('#1e1e1e')
            
            ax = self.ax_grafik 
            
            # --- JUDUL (WARNA PUTIH) ---
            ax.text(0, 1.15, "Volume Kendaraan", 
                    transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom', fontname='Segoe UI', color='white')
            # Subjudul Abu-abu
            ax.text(0, 1.08, "Di update : Otomatis dari CSV", 
                    transform=ax.transAxes, fontsize=9, color='#aaaaaa', va='bottom', fontname='Segoe UI')

            x = np.arange(jumlah_baris) 
            total_width = 0.8
            single_bar_width = total_width / max(1, jumlah_data_series)
            warna = ['#007bff', '#00c0c0', '#ffc107', '#dc3545', '#6610f2']

            for i, kol_name in enumerate(self.nama_kolom_data):
                y_values = df[kol_name]
                posisi_bar = x - (total_width / 2) + (i * single_bar_width) + (single_bar_width / 2)
                warna_bar = warna[i % len(warna)]
                bars = ax.bar(posisi_bar, y_values, single_bar_width, label=kol_name, color=warna_bar)
                self.bar_containers.append(bars)

            # Styling Grafik (WARNA PUTIH/ABU)
            ax.set_xticks(x)
            ax.set_xticklabels(self.kategori, fontsize=9, fontweight='bold', fontname='Segoe UI', color='white')
            ax.tick_params(axis='y', labelsize=9, colors='white') # Angka Y Putih
            ax.tick_params(axis='x', colors='white')              # Garis Tick X Putih
            
            # Legend (Teks Putih)
            legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False, fontsize=8)
            for text in legend.get_texts():
                text.set_color("white")
            
            # Grid Abu-abu Gelap
            ax.yaxis.grid(True, color='#444444', linestyle='-', zorder=0)
            ax.set_axisbelow(True)
            
            # Hilangkan garis atas/kanan, ubah warna garis kiri/bawah jadi abu-abu
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#888888')
            ax.spines['bottom'].set_color('#888888')

            # 4. GAMBAR KETERANGAN TEKS (DI KANAN / SIDEBAR)
            ax_t = self.ax_teks
            ax_t.axis('off') 

            list_info = [
                "1: Sepeda Motor", "2: Sedan, Jeep", "3: Minibus / Angkot",
                "4: Pick Up, Blindvan, mobil hantaran", "5a: Bus Kecil", "5b: Bus Besar",
                "6a: Truk Ringan 2 Sumbu", "6b: Truk Sedang 2 Sumbu",
                "7a: Truk 3 Sumbu", "7b: Truk Gandeng", "7c: Truk Semitrailer",
                "8: Kendaraan Tidak Bermotor"
            ]

            # Judul List (Putih)
            ax_t.text(0, 1.0, "KETERANGAN GOLONGAN:", fontweight='bold', fontsize=9, fontname='Segoe UI', color='white')

            # Loop Isi List (Putih)
            posisi_y = 0.92 
            for info in list_info:
                ax_t.text(0, posisi_y, info, fontsize=10, va='top', color='white')
                posisi_y -= 0.08 

            # Tooltip setup (Hidden awal)
            self.annot = ax.annotate("", xy=(0,0), xytext=(0,10),
                                     textcoords="offset points",
                                     bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.9),
                                     arrowprops=dict(arrowstyle="->"))
            self.annot.set_visible(False)

            self.fig.subplots_adjust(top=0.82, bottom=0.1, left=0.1, right=0.98, wspace=0.1)
            self.draw()

        except Exception as e:
            print(f"Error Grafik: {e}")
            self.ax_grafik.clear()
            self.ax_grafik.text(0.5, 0.5, "Gagal Memuat Grafik", ha='center', color='red', fontname='Segoe UI')
            self.draw()

    def on_hover(self, event):
        if event.inaxes == self.ax_grafik:
            found = False
            if self.annot is None: return # Mencegah error jika annot belum dibuat
            
            for i, bars in enumerate(self.bar_containers):
                for j, bar in enumerate(bars):
                    cont, _ = bar.contains(event)
                    if cont:
                        nama_garis = self.nama_kolom_data[i]
                        nama_golongan = self.kategori[j] 
                        nilai = bar.get_height()
                        
                        x = bar.get_x() + bar.get_width() / 2
                        y = bar.get_height()
                        self.annot.xy = (x, y)
                        text = f"{nama_golongan}\n{nama_garis}: {int(nilai)}"
                        self.annot.set_text(text)
                        self.annot.get_bbox_patch().set_edgecolor(bar.get_facecolor())
                        self.annot.set_visible(True)
                        self.draw_idle()
                        found = True
                        break
                if found: break
            if not found and self.annot.get_visible():
                self.annot.set_visible(False)
                self.draw_idle()

class TrafficAnalysisWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Konfigurasi
        self.interval_menit = 5
        self.df = pd.DataFrame()
        self.list_garis = []
        self.list_kelas = []
        self.current_csv = None

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 10, 0, 0)

        # --- 1. HEADER & TOOLBAR ---
        self.top_layout = QHBoxLayout()
        
        self.lbl_judul = QLabel("Analisis Waktu")
        self.lbl_judul.setStyleSheet("font-weight: bold; color: white; font-family: 'Segoe UI'; border: none;")
        self.top_layout.addWidget(self.lbl_judul)
        
        self.top_layout.addSpacing(20)

        # --- Style Dasar Label ---
        style_label = "color: #e0e0e0; font-weight: bold; font-size: 13px; font-family: 'Segoe UI'; border: none;"

 # --- Style Khusus ComboBox (Dropdown Dark Mode - FIXED) ---
        style_combo = """
            /* 1. Bagian Utama ComboBox */
            QComboBox {
                background-color: #333333;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
                padding-left: 10px;
                min-width: 10px;
            }
            
            QComboBox:hover {
                border: 1px solid #007acc; /* Border biru saat hover */
                background-color: #3a3a3a;
            }
            
            QComboBox:on { /* Saat menu terbuka */
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }

            /* 2. Bagian Tombol Panah (Kanan) */
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 25px;
                
                border-left-width: 1px;
                border-left-color: #555;
                border-left-style: solid; /* Garis pemisah vertikal */
                
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
                background-color: #333333; /* Pastikan background sama */
            }

            /* 3. Ikon Panah (Segitiga CSS) */
            QComboBox::down-arrow {
                image: none;
                width: 0; 
                height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #cccccc; /* Warna Panah Abu Terang */
                margin-right: 2px;
            }

            /* 4. Daftar Pilihan (Popup List) */
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #555;
                selection-background-color: #007acc;
                selection-color: white;
                
                /* INI SOLUSINYA: Hilangkan garis putus-putus oranye */
                outline: 0px; 
            }
        """

        # --- Style Khusus Tombol Interval (Toggle Dark Mode) ---
        style_btn_interval = """
            QPushButton {
                background-color: #333333;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:checked {
                background-color: #007acc; /* Biru saat aktif */
                border: 1px solid #005a9e;
                font-weight: bold;
            }
        """

        # --- SETUP COMBOBOX GARIS ---
        self.combo_garis = QComboBox()
        self.combo_garis.setStyleSheet(style_combo) # Terapkan Style
        self.combo_garis.addItems(["Semua"])
        self.combo_garis.currentIndexChanged.connect(self.update_chart)
        
        lbl_garis = QLabel("Garis:")
        lbl_garis.setStyleSheet(style_label)
        self.top_layout.addWidget(lbl_garis)
        self.top_layout.addWidget(self.combo_garis)
        
        self.top_layout.addSpacing(10)

        # --- SETUP COMBOBOX KELAS ---
        self.combo_kelas = QComboBox()
        self.combo_kelas.setStyleSheet(style_combo) # Terapkan Style
        self.combo_kelas.addItems(["Semua"])
        self.combo_kelas.currentIndexChanged.connect(self.update_chart)
        
        lbl_kelas = QLabel("Kelas:")
        lbl_kelas.setStyleSheet(style_label)
        self.top_layout.addWidget(lbl_kelas)
        self.top_layout.addWidget(self.combo_kelas)

        self.top_layout.addSpacing(20)

        # --- SETUP TOMBOL INTERVAL ---
        self.btn_5m = QPushButton("5m"); self.btn_5m.setCheckable(True)
        self.btn_10m = QPushButton("10m"); self.btn_10m.setCheckable(True)
        self.btn_30m = QPushButton("30m"); self.btn_30m.setCheckable(True)
        
        # Terapkan Style ke semua tombol
        self.btn_5m.setStyleSheet(style_btn_interval)
        self.btn_10m.setStyleSheet(style_btn_interval)
        self.btn_30m.setStyleSheet(style_btn_interval)
        
        self.buttons = [self.btn_5m, self.btn_10m, self.btn_30m]
        self.intervals = [5, 10, 30]
        
        lbl_interval = QLabel("Interval:")
        lbl_interval.setStyleSheet(style_label)
        self.top_layout.addWidget(lbl_interval)
        
        for btn, interval in zip(self.buttons, self.intervals):
            btn.setFixedSize(40, 25)
            btn.clicked.connect(lambda checked, val=interval: self.set_interval(val))
            self.top_layout.addWidget(btn)
        
        self.btn_5m.setChecked(True) # Default aktif

        self.top_layout.addStretch()
        self.main_layout.addLayout(self.top_layout)

        # --- 2. AREA GRAFIK (PERBAIKAN DISINI) ---
        # Kita definisikan self.fig agar bisa diakses oleh subplots_adjust nanti
        self.fig = Figure(figsize=(10, 3), dpi=80)
        self.fig.patch.set_facecolor('#f9f9f9')
        
        self.canvas = FigureCanvas(self.fig) 
        self.main_layout.addWidget(self.canvas)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.text(0.5, 0.5, 
            "Belum ada data", 
            ha='center', 
            va='center',       # Agar teks benar-benar di tengah vertikal
            color='#cccccc',   # Warna abu-abu terang (lebih lembut dari putih murni)
            fontsize=14,       # Ukuran font
            fontweight='bold', # Tebal huruf (bold/normal/light)
            fontname='Segoe UI' # Jenis font (samakan dengan UI aplikasi)
        )

        # --- SETUP DARK MODE UNTUK GRAFIK ---
        # 1. Warna Background (Samakan dengan warna aplikasi #1e1e1e atau sedikit lebih terang #2c2c2c)
        self.fig.patch.set_facecolor('#1e1e1e') 
        self.ax.set_facecolor('#1e1e1e')

        # 2. Warna Garis Sumbu (Spines) menjadi Putih/Abu-abu
        self.ax.spines['bottom'].set_color('#aaaaaa')
        self.ax.spines['top'].set_color('#aaaaaa')
        self.ax.spines['left'].set_color('#aaaaaa')
        self.ax.spines['right'].set_color('#aaaaaa')

        # 3. Warna Teks Label & Tick (Angka) menjadi Putih
        self.ax.tick_params(axis='x', colors='#cccccc')
        self.ax.tick_params(axis='y', colors='#cccccc')
        self.ax.yaxis.label.set_color('#cccccc')
        self.ax.xaxis.label.set_color('#cccccc')
        self.ax.title.set_color('#ffffff')

        # 4. (Opsional) Grid Tipis agar terlihat modern
        self.ax.grid(color='#444444', linestyle='--', linewidth=0.5, alpha=0.5)

    def load_data(self, csv_path):
        self.current_csv = csv_path
        print(f"\n[TrafficWidget] Mencoba membaca CSV: {csv_path}")

        try:
            if not os.path.exists(csv_path):
                print(f"[TrafficWidget] FILE TIDAK DITEMUKAN: {csv_path}")
                self.df = pd.DataFrame()
                self.update_chart()
                return

            self.df = pd.read_csv(csv_path)
            self.df.columns = self.df.columns.str.strip()
            
            print(f"[TrafficWidget] Kolom ditemukan: {list(self.df.columns)}")

            required_cols = ['Waktu Video', 'Jenis Kendaraan', 'Garis']
            if not all(col in self.df.columns for col in required_cols):
                print("[TrafficWidget] Format CSV Log tidak sesuai.")
                self.ax.clear()
                self.ax.text(0.5, 0.5, "Format CSV Salah", ha='center', fontname='Segoe UI')
                self.canvas.draw()
                return

            self.list_garis = sorted(self.df['Garis'].unique().astype(str).tolist())
            self.list_kelas = sorted(self.df['Jenis Kendaraan'].unique().astype(str).tolist())

            self.combo_garis.blockSignals(True)
            self.combo_kelas.blockSignals(True)
            self.combo_garis.clear(); self.combo_garis.addItems(["Semua"] + self.list_garis)
            self.combo_kelas.clear(); self.combo_kelas.addItems(["Semua"] + self.list_kelas)
            self.combo_garis.blockSignals(False)
            self.combo_kelas.blockSignals(False)

            def parse_time(t_str):
                try: return pd.to_datetime(str(t_str), format='%H:%M:%S.%f')
                except:
                    try: return pd.to_datetime(str(t_str), format='%H:%M:%S')
                    except: return pd.to_datetime("00:" + str(t_str))

            self.df['datetime'] = self.df['Waktu Video'].apply(lambda x: parse_time(x))
            
            print(f"[TrafficWidget] Berhasil memuat {len(self.df)} baris data.")
            self.update_chart()

        except Exception as e:
            print(f"[TrafficWidget] Error loading traffic data: {e}")
            import traceback
            traceback.print_exc() # Print detail error ke terminal untuk debugging
            self.df = pd.DataFrame()
            self.update_chart()

    def set_interval(self, menit):
        self.interval_menit = menit
        for btn, val in zip(self.buttons, self.intervals):
            btn.setChecked(val == menit)
        self.update_chart()

    def update_chart(self):
        self.ax.clear()
        
        if self.df.empty:
            self.ax.text(0.5, 0.5, "Data Kosong / Tidak Ada File Log", ha='center', fontsize=9, fontname='Segoe UI', color='white')
            self.canvas.draw()
            return

        # 1. FILTERING
        df_filtered = self.df.copy()
        
        pilihan_garis = self.combo_garis.currentText()
        if pilihan_garis != "Semua":
            df_filtered = df_filtered[df_filtered['Garis'] == pilihan_garis]
            
        pilihan_kelas = self.combo_kelas.currentText()
        if pilihan_kelas != "Semua":
            df_filtered = df_filtered[df_filtered['Jenis Kendaraan'] == pilihan_kelas]

        if df_filtered.empty:
            self.ax.text(0.5, 0.5, "Tidak ada data pada filter ini", ha='center', fontsize=9, fontname='Segoe UI')
            self.canvas.draw()
            return

        # 2. RESAMPLING
        df_filtered = df_filtered.set_index('datetime')
        
        # PERBAIKAN WARNING: Ganti 'T' (deprecated) menjadi 'min'
        rule = f'{self.interval_menit}min' 
        
        resampled = df_filtered.resample(rule)['Garis'].count()
        
        x_raw = np.arange(len(resampled))
        y_values = resampled.values 
        x_labels = [dt.strftime('%H:%M') for dt in resampled.index]

        if len(y_values) > 0:
            nilai_rata = y_values.mean()
        else:
            nilai_rata = 0

        # 3. PLOTTING
        self.ax.bar(x_raw, y_values, width=0.6, color='#4fc3f7', label='Volume', alpha=0.8)

        label_rata = f'Avg: {int(nilai_rata)}'
        self.ax.axhline(y=nilai_rata, color='#FF5722', linestyle='--', linewidth=1.5, label=label_rata)

        # Styling
        self.ax.set_xticks(x_raw)
        self.ax.set_xticklabels(x_labels, rotation=0, fontsize=8)
        self.ax.tick_params(axis='y', labelsize=8)
        
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
        self.ax.legend(loc='upper right', frameon=False, fontsize=8)
        self.ax.set_title(f"Volume per {self.interval_menit} Menit", fontsize=10, fontweight='bold', color='white', fontname='Segoe UI')
        
        # PERBAIKAN: self.fig sekarang sudah didefinisikan di __init__
        self.fig.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.98)
        self.canvas.draw()

class VideoPlayer(QWidget):
    def __init__(self, stacked):
        super().__init__()
        self.stacked = stacked
        
        rect = QApplication.desktop().availableGeometry()
        available_height = rect.height()
        
        # Tinggi Video & Ringkasan = 55% dari layar
        self.target_height = int(available_height * 0.5)

        self.current_video_path = None
        self.is_showing_log = False 

        # --- KONFIGURASI LAYOUT UTAMA ---
        # Gunakan QVBoxLayout (Vertikal) sebagai layout utama halaman
        self.layout = QVBoxLayout() 
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        self.setLayout(self.layout)

        # --- 1. HEADER (TOMBOL BACK & JUDUL) ---
        header_layout = QHBoxLayout()
        header_layout.setSpacing(15) # Jarak antara tombol back dan judul

        # A. Tombol Back (Dark Mode)
        self.btn_back = QPushButton("← Back")
        self.btn_back.setFixedSize(80, 35)
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.clicked.connect(self.go_back)
        
        self.btn_back.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #3e3e3e;
                border-radius: 6px;
                font-weight: bold;
                font-family: 'Segoe UI';
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border: 1px solid #555;
                color: white;
            }
            QPushButton:pressed {
                background-color: #151515;
            }
        """)

        # B. Label Judul Video (Tanpa Border/Background)
        self.label_title = QLabel("")
        self.label_title.setStyleSheet("""
            color: #ffffff;
            font-family: 'Segoe UI';
            font-size: 16px;
            font-weight: bold;
            background: transparent; /* Hilangkan kotak abu-abu */
            border: none;            /* Hilangkan border */
            padding-left: 5px;
        """)
        self.label_title.setFixedHeight(35) # Samakan tinggi dengan tombol

        # Masukkan ke Header Layout
        header_layout.addWidget(self.btn_back)
        header_layout.addWidget(self.label_title)
        header_layout.addStretch() # Dorong semuanya ke kiri

        # Masukkan Header ke Layout Utama
        self.layout.addLayout(header_layout)

        # --- SETUP CONTROLS (Tombol Bawah) ---
        self.controls_layout = QHBoxLayout()
        self.btn_open = QPushButton("Open")
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        self.run_button = QPushButton("Run") 
        self.btn_draw = QPushButton("Draw Line")
        self.btn_clear = QPushButton("Clear Line")

        style_btn_dark = """
            QPushButton {
                background-color: #2b2b2b;
                color: #e0e0e0;
                border: 1px solid #3e3e3e;
                border-radius: 6px;
                font-family: 'Segoe UI';
                font-size: 13px;
                font-weight: 600;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #3a3a3a; /* Lebih terang saat hover */
                border: 1px solid #555;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #151515; /* Lebih gelap saat ditekan */
            }
            QPushButton:checked {
                background-color: #005a9e; /* Khusus tombol Draw saat aktif */
                border: 1px solid #007acc;
                color: white;
            }
        """

        # Terapkan style dark ke tombol-tombol standar
        self.btn_open.setStyleSheet(style_btn_dark)
        self.btn_play.setStyleSheet(style_btn_dark)
        self.btn_pause.setStyleSheet(style_btn_dark)
        self.btn_stop.setStyleSheet(style_btn_dark)
        self.btn_draw.setStyleSheet(style_btn_dark)
        self.btn_clear.setStyleSheet(style_btn_dark)

        # 3. Style Khusus Tombol UTAMA (Run / Lihat Log)
        # Kita buat warnanya Biru/Hijau mencolok tapi tetap modern
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4; /* Biru Windows 11 */
                color: white;
                border: none;
                border-radius: 6px;
                font-family: 'Segoe UI';
                font-weight: bold;
                font-size: 13px;
                padding: 6px 15px;
            }
            QPushButton:hover {
                background-color: #0063b1;
            }
            QPushButton:pressed {
                background-color: #004e8c;
            }
        """)

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
        
        # --- 2. SETUP VIDEO (KIRI) ---
        # --- 2. SETUP VIDEO (KIRI) ---
        self.video_widget = MyVideoWidget()
        self.video_widget.setMediaPlayer(self.media_player)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_widget.setMinimumHeight(500)
        
        # --- TAMBAHKAN STYLE INI UNTUK ROUNDED BORDER ---
        self.video_widget.setStyleSheet("""
            QGraphicsView {
                border: 1px solid #333;  /* Garis tepi tipis warna abu gelap */
                border-radius: 15px;     /* Membuat sudut melengkung */
                background-color: black; /* Pastikan background hitam */
            }
        """)
        # -----------------------------------------------

        self.progress_layout = QHBoxLayout()
        
        # --- Bagian label (Kode Anda sebelumnya) ---
        self.label_current = QLabel("00:00:00")
        self.label_current.setStyleSheet("color: #ffffff; font-weight: bold; font-family: 'Segoe UI'; border: none;") 
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setEnabled(False)
        
        self.label_total = QLabel("00:00:00")
        self.label_total.setStyleSheet("color: #ffffff; font-weight: bold; font-family: 'Segoe UI'; border: none;")
        
        self.progress_layout.addWidget(self.label_current)
        self.progress_layout.addWidget(self.slider, stretch=1)
        self.progress_layout.addWidget(self.label_total)

        # --- 3. SETUP INFO PANEL (KANAN) ---
        self.right_panel_widget = QWidget()
        self.right_panel_layout = QVBoxLayout(self.right_panel_widget)
        self.right_panel_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_info_header = QLabel("Informasi Deteksi")
        self.lbl_info_header.setAlignment(Qt.AlignCenter)
        self.lbl_info_header.setStyleSheet("font-weight: bold; color: white; margin-bottom: 5px; font-family: 'Segoe UI'; border: none;")
        self.lbl_info_header.setFixedHeight(25)

        self.info_label = QLabel("Status: Menunggu video...")
        self.info_label.setStyleSheet("font-weight: bold; font-size: 14px; color: white; background-color: #333; padding: 10px; border-radius: 4px; font-family: 'Segoe UI'")
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.info_label.setWordWrap(True)
        self.info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.chart_widget = CanvasGrafik(self, width=5, height=4, dpi=80)
        self.chart_widget.setVisible(False)
        self.chart_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.table_csv = QTableWidget()
        self.table_csv.setColumnCount(5) 
        self.table_csv.setHorizontalHeaderLabels(["Waktu", "Kelas", "ID", "Garis", "Aksi"])
        self.table_csv.setStyleSheet("QTableWidget { background-color: #222; color: white; gridline-color: #444; font-size: 11px; }")
        self.table_csv.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_csv.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_csv.verticalHeader().setVisible(False)
        self.table_csv.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_csv.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_csv.cellClicked.connect(self.seek_video_from_table)
        self.table_csv.setVisible(False)

        self.info_progress = QProgressBar()
        self.info_progress.setRange(0, 100)
        self.info_progress.setValue(0)
        self.info_progress.setFixedHeight(15)

        self.info_progress.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #333333; /* Warna Background Batang Kosong */
                border-radius: 5px;
                color: #ffffff;            /* <--- INI KUNCINYA (Warna Teks Putih) */
                text-align: center;        /* Opsi: Menaruh teks di tengah batang agar lebih rapi */
                font-weight: bold;
                font-family: Segoe UI;
            }
            QProgressBar::chunk {
                background-color: #27ae60; /* Warna Hijau (sesuai screenshot Anda) */
                border-radius: 5px;
            }
        """)

        self.right_panel_layout.addWidget(self.lbl_info_header)
        self.right_panel_layout.addWidget(self.info_label)
        self.right_panel_layout.addWidget(self.chart_widget)
        self.right_panel_layout.addWidget(self.table_csv)
        self.right_panel_layout.addWidget(self.info_progress)
        
        self.right_panel_widget.setMinimumHeight(500)
        # self.right_panel_widget.setMinimumHeight(self.target_height)

        # --- 4. GABUNGKAN LAYOUT TENGAH (VIDEO + RINGKASAN) ---
        self.video_info_layout = QHBoxLayout()
        self.video_info_layout.addWidget(self.video_widget, stretch=1)
        self.video_info_layout.addWidget(self.right_panel_widget, stretch=1)

        self.layout.addWidget(self.label_title)
        self.layout.addLayout(self.video_info_layout)

        # --- 5. [UBAH DISINI] WIDGET ANALISIS BAWAH (FILL SPACE) ---
        self.traffic_widget = TrafficAnalysisWidget()
        
        # PENTING: Set Policy ke Expanding agar mengisi sisa ruang vertikal
        self.traffic_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # PENTING: Hapus setMaximumHeight agar tidak dibatasi
        self.traffic_widget.setMaximumHeight(300) # <--- DIBUANG
        
        # Tambahkan ke layout (stretch=1 agar mengambil sisa ruang)
        self.layout.addWidget(self.traffic_widget, stretch=1)

        # PENTING: Hapus spacer (addStretch) yang sebelumnya ada disini
        # self.layout.addStretch(1) <--- DIBUANG

        # Tombol tetap di paling bawah
        self.layout.addLayout(self.progress_layout)
        self.layout.addLayout(self.controls_layout)

        # Koneksi Signal
        self.btn_open.clicked.connect(self.open_file)
        self.btn_play.clicked.connect(self.play_video)
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_stop.clicked.connect(self.stop_video)
        self.run_button.clicked.connect(self.run_detection)
        self.slider.sliderMoved.connect(self.set_position)
        self.media_player.positionChanged.connect(self.update_position)
        self.media_player.durationChanged.connect(self.update_duration)
        self.media_player.stateChanged.connect(self.handle_state_changed)

    # --- FUNGSI HELPER BARU (Tambahkan ini di dalam class VideoPlayer) ---
    def get_result_path(self, video_path, suffix):
        """
        Logika Satu CSV untuk Semua:
        1. Cars.mp4       -> Root: Cars       -> CSV: Cars_hasil...
        2. Cars_hasil.mp4 -> Root: Cars       -> CSV: Cars_hasil...
        """
        if not video_path: return ""
        
        filename = os.path.basename(video_path)
        base_name = os.path.splitext(filename)[0]
        
        # 1. BERSIHKAN: Jika nama file sudah mengandung '_hasil' di belakang, buang dulu.
        if base_name.endswith("_hasil"):
            clean_name = base_name[:-6] # Buang 6 karakter terakhir ("_hasil")
        else:
            clean_name = base_name
            
        # 2. BENTUK ULANG: Selalu gunakan format [NamaBersih]_hasil[Suffix]
        return os.path.join("hasil_test", f"{clean_name}_hasil{suffix}")
        """
        Mengubah path video input menjadi path file hasil di folder 'hasil_test'.
        Contoh: C:/Video/Cars.mp4 -> hasil_test/Cars_hasil_log_detail.csv
        """
        if not video_path: return ""
        
        # 1. Ambil nama file asli (Cars.mp4)
        filename = os.path.basename(video_path) 
        
        # 2. Buang ekstensinya (Cars)
        base_name = os.path.splitext(filename)[0]
        
        # 3. Gabungkan dengan folder hasil_test DAN tambahkan "_hasil"
        # Karena log menyimpan sebagai: Cars_hasil_log_detail.csv
        return os.path.join("hasil_test", f"{base_name}_hasil{suffix}")

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
            
            # --- PERBAIKAN PATH DI SINI ---
            self.check_and_update_button(file_name)
            
            # Gunakan helper untuk mencari log di hasil_test
            log_path = self.get_result_path(file_name, "_log_detail.csv")
            self.load_csv_summary(file_name) # Summary juga perlu perbaikan path di dalamnya
            
            # Load data ke widget traffic bawah
            self.traffic_widget.load_data(log_path)

    def reset_info_view(self):
        self.is_showing_log = False
        self.lbl_info_header.setText("RINGKASAN HASIL")
        
        self.table_csv.setVisible(False)
        self.chart_widget.setVisible(False) # Reset grafik
        self.info_label.setVisible(True)    # Tampilkan status text
        self.info_label.setText("Status: Menunggu video...")

    def check_and_update_button(self, video_path):
        # --- PERBAIKAN PATH ---
        log_path = self.get_result_path(video_path, "_log_detail.csv")
        
        try: self.run_button.clicked.disconnect()
        except TypeError: pass 

        # --- DEFINISI BENTUK TOMBOL (AGAR KONSISTEN) ---
        # Kita simpan style bentuk di sini agar tidak hilang saat ganti warna
        base_shape = """
            color: white;
            border: none;
            border-radius: 6px; /* INI KUNCINYA (Membuat sudut tumpul) */
            font-family: 'Segoe UI';
            font-weight: bold;
            font-size: 13px;
            padding: 6px 15px;
        """

        if os.path.exists(log_path):
            self.run_button.setText("Lihat Log")
            # Gabungkan Bentuk + Warna Hijau
            self.run_button.setStyleSheet(f"""
                QPushButton {{
                    {base_shape}
                    background-color: #27ae60; 
                }}
                QPushButton:hover {{ background-color: #219150; }}
            """)
            self.run_button.clicked.connect(self.toggle_log_view)
        else:
            self.run_button.setText("Run")
            # Gabungkan Bentuk + Warna Biru
            self.run_button.setStyleSheet(f"""
                QPushButton {{
                    {base_shape}
                    background-color: #3498db;
                }}
                QPushButton:hover {{ background-color: #2980b9; }}
            """)
            self.run_button.clicked.connect(self.run_detection)
            self.info_label.setText("Status: Video siap diproses.\nSilakan gambar garis lalu klik Run.")

    def toggle_log_view(self):
        # --- DEFINISI BENTUK TOMBOL (SAMA SEPERTI DI ATAS) ---
        base_shape = """
            color: white;
            border: none;
            border-radius: 6px;
            font-family: 'Segoe UI';
            font-weight: bold;
            font-size: 13px;
            padding: 6px 15px;
        """

        if not self.is_showing_log:
            # MODE: LIHAT LOG DETAIL (Tabel)
            self.load_log_to_table()
            self.info_label.setVisible(False)
            self.chart_widget.setVisible(False) 
            self.table_csv.setVisible(True)     
            
            self.run_button.setText("Ringkasan")
            # Gabungkan Bentuk + Warna Orange
            self.run_button.setStyleSheet(f"""
                QPushButton {{
                    {base_shape}
                    background-color: #e67e22;
                }}
                QPushButton:hover {{ background-color: #d35400; }}
            """)
            
            self.lbl_info_header.setText("LOG DETAIL KENDARAAN")
            self.is_showing_log = True
        else:
            # MODE: LIHAT RINGKASAN (Grafik)
            self.load_csv_summary(self.current_video_path) 
            self.table_csv.setVisible(False)    
            
            self.run_button.setText("Lihat Log")
            # Gabungkan Bentuk + Warna Hijau
            self.run_button.setStyleSheet(f"""
                QPushButton {{
                    {base_shape}
                    background-color: #27ae60;
                }}
                QPushButton:hover {{ background-color: #219150; }}
            """)
            
            self.lbl_info_header.setText("GRAFIK VOLUME")
            self.is_showing_log = False

    def load_csv_summary(self, video_path):
        """
        Fungsi baru: Memuat grafik dari CSV Ringkasan
        """
        # --- PERBAIKAN PATH ---
        csv_path = self.get_result_path(video_path, "_ringkasan.csv")
        
        if not os.path.exists(csv_path):
            self.chart_widget.setVisible(False)
            self.info_label.setVisible(True)
            # self.info_label.setText("Belum ada data ringkasan.") # Optional
            return

        # Sembunyikan teks status, tampilkan grafik
        self.info_label.setVisible(False)
        self.chart_widget.setVisible(True)
        
        self.chart_widget.update_data(csv_path)

    def load_log_to_table(self):
        if not self.current_video_path: return
        
        # --- PERBAIKAN PATH ---
        log_path = self.get_result_path(self.current_video_path, "_log_detail.csv")
        
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
                    row_data_list.append({'data': (waktu, kelas, oid, garis), 'raw': row})
            
            self.table_csv.setRowCount(len(row_data_list))
            for i, item in enumerate(row_data_list):
                waktu, kelas, oid, garis = item['data']
                
                item_waktu = QTableWidgetItem(waktu)
                ms_val = self.parse_timestamp_to_ms(waktu)
                item_waktu.setData(Qt.UserRole, ms_val)
                self.table_csv.setItem(i, 0, item_waktu)
                self.table_csv.setItem(i, 1, QTableWidgetItem(kelas))
                self.table_csv.setItem(i, 2, QTableWidgetItem(oid))
                self.table_csv.setItem(i, 3, QTableWidgetItem(garis))

                btn_edit = QPushButton("Edit")
                btn_edit.setStyleSheet("background-color: #f39c12; color: white; border: none; padding: 2px;")
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

        log_path = self.get_result_path(self.current_video_path, "_log_detail.csv")
        summary_path = self.get_result_path(self.current_video_path, "_ringkasan.csv")
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
            
            # TAMBAHKAN INI: Reload data untuk widget traffic bawah
            self.traffic_widget.load_data(log_path)
        except Exception as e:
            print(f"Error updating summary: {e}")


    # =========================================================
    # FUNGSI DETEKSI & HELPER (SAMA SEPERTI SEBELUMNYA)
    # =========================================================
    def run_detection(self):
        if self.current_video_path is None: return
        
        # 1. Ambil koordinat mentah dari GUI
        coords_list = self.get_all_line_coords()
        if not coords_list:
            QMessageBox.warning(self, "Peringatan", "Gambar garis dulu!")
            return

        # --- [MULAI PERUBAHAN: HITUNG SKALA] ---
        # Kita buka video sebentar untuk tahu ukuran aslinya
        cap = cv2.VideoCapture(self.current_video_path)
        real_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        real_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        # Ambil ukuran video yang tampil di layar aplikasi (GUI)
        gui_width = self.video_widget.video_item.size().width()
        gui_height = self.video_widget.video_item.size().height()

        # Hitung faktor skala (Ratio)
        # Jika GUI 0 (error), hindari pembagian nol
        if gui_width == 0 or gui_height == 0:
            scale_x = 1
            scale_y = 1
        else:
            scale_x = real_width / gui_width
            scale_y = real_height / gui_height

        # Terapkan skala ke setiap garis
        raw_lines = []
        for item in coords_list:
            x1, y1, x2, y2 = item['coords']
            
            # Kalikan koordinat GUI dengan skala agar sesuai video asli
            real_x1 = int(x1 * scale_x)
            real_y1 = int(y1 * scale_y)
            real_x2 = int(x2 * scale_x)
            real_y2 = int(y2 * scale_y)
            
            raw_lines.append((real_x1, real_y1, real_x2, real_y2))
        # --- [AKHIR PERUBAHAN] ---

        # Konfirmasi User
        est_min = int((frame_count / fps) // 60)
        msg = QMessageBox()
        msg.setWindowTitle("Konfirmasi")
        msg.setText(f"Video Asli: {int(real_width)}x{int(real_height)}\n"
                    f"Tampilan GUI: {int(gui_width)}x{int(gui_height)}\n"
                    f"Ratio Skala: {scale_x:.2f} x {scale_y:.2f}\n\n"
                    f"Mulai deteksi? (+/- {est_min} menit)")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        
        if msg.exec_() != QMessageBox.Yes: return

        # Siapkan UI untuk proses
        # self.chart_canvas.axes.clear()
        # self.chart_canvas.axes.text(0.5, 0.5, "Sedang memproses...", ha='center')
        # self.chart_canvas.draw()
        
        self.info_progress.setValue(0)
        self.run_button.setEnabled(False)
        
        input_filename = os.path.basename(self.current_video_path)
        base_name = os.path.splitext(input_filename)[0]

        # BERSIHKAN nama juga disini (Sama seperti get_result_path)
        # Agar output tidak menjadi 'Cars_hasil_hasil.mp4'
        if base_name.endswith("_hasil"):
            clean_name = base_name[:-6]
        else:
            clean_name = base_name

        # Setting Path Output
        model_path = "model/best.pt"
        # Output video selalu: [NamaBersih]_hasil.mp4
        output_path = os.path.join("hasil_test", f"{clean_name}_hasil.mp4")

        # Pastikan folder hasil_test ada
        if not os.path.exists("hasil_test"):
            os.makedirs("hasil_test")

        # --- Akhir Perubahan Penamaan ---

        # Jalankan Worker (Gunakan output_path yang baru)
        self.worker = DetectionWorker(self.current_video_path, model_path, output_path, raw_lines) # Pastikan raw_lines sudah dihitung skala (kode sebelumnya)
        self.worker.progress_changed.connect(self.info_progress.setValue)
        self.worker.finished.connect(self.on_detection_finished)
        self.worker.start()        # Setting Path

    def on_detection_finished(self):
        self.info_label.setText("Selesai! Klik 'Lihat Log' untuk melihat detail.")
        self.info_progress.setValue(100)
        self.run_button.setEnabled(True) # Jangan lupa enable tombol kembali
        
        if self.current_video_path:
            # Update Ringkasan (Grafik Atas)
            self.load_csv_summary(self.current_video_path)
            self.check_and_update_button(self.current_video_path)
            
            # --- PERBAIKAN PATH UNTUK TRAFFIC WIDGET ---
            log_path = self.get_result_path(self.current_video_path, "_log_detail.csv")
            self.traffic_widget.load_data(log_path)

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