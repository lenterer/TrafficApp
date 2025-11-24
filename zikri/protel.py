import cv2
from ultralytics import YOLO
import numpy as np # Diperlukan untuk koordinat
import torch

print(torch.cuda.get_device_name(0))

MODEL_PATH = 'C:/HDD/TUGAS KULIAH/TrafficCountYOLO/Protel/TrafficApp/Model1/best.pt' 
VIDEO_SOURCE = 'cctv.mp4' 
CLASS_NAMES = ['1', '2', '3', '4', '5a', '5b', '6a', '6b', '7a', '7b', '7c', '8']
CONF_THRESHOLD = 0.1

# --- PERUBAHAN: Inisialisasi untuk DUA Garis ---

# Tentukan posisi Y untuk kedua garis
# SESUAIKAN DUA NILAI INI berdasarkan video Anda
LINE_RED_Y = 380   # Garis untuk kendaraan DARI ATAS KE BAWAH
LINE_GREEN_Y = 420 # Garis untuk kendaraan DARI BAWAH KE ATAS

# Dictionary untuk menyimpan posisi Y terakhir dari setiap objek
# Format: { obj_id: last_y }
object_history = {}

# --- Kita butuh DUA set ID dan DUA dict hitungan ---
# Set 1: Untuk arah Atas ke Bawah (Melintasi Garis Merah)
counted_ids_down = set()
counts_top_to_bottom = {class_name: 0 for class_name in CLASS_NAMES}

# Set 2: Untuk arah Bawah ke Atas (Melintasi Garis Hijau)
counted_ids_up = set()
counts_bottom_to_top = {class_name: 0 for class_name in CLASS_NAMES}
# ----------------------------------------------------


try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Tidak dapat membuka sumber video '{VIDEO_SOURCE}'")
    exit()

# Dapatkan dimensi frame untuk menggambar garis
ret, frame = cap.read()
if not ret:
    print("Tidak bisa membaca frame pertama.")
    exit()
FRAME_WIDTH = frame.shape[1]
FRAME_HEIGHT = frame.shape[0] # Kita mungkin perlu ini untuk dashboard
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Kembalikan video ke awal

print("Mulai deteksi dan tracking... Tekan 'q' pada jendela video untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video selesai atau gagal membaca frame.")
        break

    results = model.track(frame, persist=True)

    # --- PERUBAHAN: Gambar DUA Garis ---
    # Garis Merah (Atas ke Bawah)
    cv2.line(frame, (0, LINE_RED_Y), (FRAME_WIDTH, LINE_RED_Y), (0, 0, 255), 1)
    # Garis Hijau (Bawah ke Atas)
    cv2.line(frame, (0, LINE_GREEN_Y), (FRAME_WIDTH, LINE_GREEN_Y), (0, 255, 0), 1)
    # ------------------------------------

    if results[0].boxes.id is not None:
        for box in results[0].boxes:
            confidence = box.conf[0]

            if confidence > CONF_THRESHOLD:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                
                class_id = int(box.cls[0])
                class_name = "Unknown"
                if class_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[class_id]

                obj_id = int(box.id[0])
                
                # Gunakan titik tengah-bawah (y2) sebagai referensi
                current_y_ref = y2 

                # --- PERUBAHAN: Logika Pengecekan DUA ARAH ---
                if obj_id in object_history:
                    prev_y_ref = object_history[obj_id]
                    
                    # 1. Cek Arah ATAS ke BAWAH (Melintasi Garis Merah)
                    if (prev_y_ref < LINE_RED_Y) and (current_y_ref >= LINE_RED_Y):
                        if obj_id not in counted_ids_down:
                            counts_top_to_bottom[class_name] += 1
                            counted_ids_down.add(obj_id)
                            # Flash garis MERAH (ubah jadi putih terang) saat dilintasi
                            cv2.line(frame, (0, LINE_RED_Y), (FRAME_WIDTH, LINE_RED_Y), (255, 255, 255), 3)

                    # 2. Cek Arah BAWAH ke ATAS (Melintasi Garis Hijau)
                    if (prev_y_ref > LINE_GREEN_Y) and (current_y_ref <= LINE_GREEN_Y):
                        if obj_id not in counted_ids_up:
                            counts_bottom_to_top[class_name] += 1
                            counted_ids_up.add(obj_id)
                            # Flash garis HIJAU (ubah jadi putih terang) saat dilintasi
                            cv2.line(frame, (0, LINE_GREEN_Y), (FRAME_WIDTH, LINE_GREEN_Y), (255, 255, 255), 3)

                object_history[obj_id] = current_y_ref
                # -------------------------------------------------

                label = f'ID: {obj_id} {class_name}'

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # --- PERUBAHAN: Tampilkan Dashboard untuk DUA Arah ---
    
    # Hitungan Arah Bawah (di Kiri Atas)
    y_offset_down = 30
    total_down = sum(counts_top_to_bottom.values())
    cv2.putText(frame, f'Arah Bawah (Merah): {total_down}', (10, y_offset_down), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Detail Arah Bawah
    for name, count in counts_top_to_bottom.items():
        if count > 0:
            y_offset_down += 25
            cv2.putText(frame, f'  {name}: {count}', (10, y_offset_down), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Hitungan Arah Atas (di Kanan Atas)
    y_offset_up = 30
    x_offset_up = FRAME_WIDTH - 250 # Sesuaikan '250' jika perlu
    total_up = sum(counts_bottom_to_top.values())
    cv2.putText(frame, f'Arah Atas (Hijau): {total_up}', (x_offset_up, y_offset_up), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Detail Arah Atas
    for name, count in counts_bottom_to_top.items():
        if count > 0:
            y_offset_up += 25
            cv2.putText(frame, f'  {name}: {count}', (x_offset_up + 150, y_offset_up), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # ----------------------------------------------------

    cv2.imshow('Deteksi Kendaraan - Tekan q untuk Keluar', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program dihentikan.")

# --- PERUBAHAN: Hasil Akhir Perhitungan ---
print("--- Hasil Akhir (Arah Atas ke Bawah) ---")
for name, count in counts_top_to_bottom.items():
    if count > 0:
        print(f"{name}: {count}")

print("\n--- Hasil Akhir (Arah Bawah ke Atas) ---")
for name, count in counts_bottom_to_top.items():
    if count > 0:
        print(f"{name}: {count}")