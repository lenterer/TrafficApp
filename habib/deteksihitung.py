import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
from ultralytics import YOLO
import cv2
import numpy as np
import csv
import datetime
import torch

def jalankan_deteksi_dan_hitung(video_path, model_path, output_path="hasil.mp4",
                               lines_coords=None, progress_callback=None):
    CLASS_NAMES = ['1', '2', '3', '4', '5a', '5b', '6a', '6b', '7a', '7b', '7c', '8']

    # ... (CLASS_COLORS dan setup folder tetap sama) ...
    CLASS_COLORS = {
        '1': (0, 255, 0), '2': (255, 0, 0), '3': (0, 0, 255), '4': (255, 255, 0),
        '5a': (255, 0, 255), '5b': (0, 255, 255), '6a': (128, 0, 128), '6b': (0, 128, 255),
        '7a': (0, 165, 255), '7b': (128, 128, 0), '7c': (0, 128, 0), '8': (128, 0, 0),
    }

    if torch.cuda.is_available():
        device = 0  # Gunakan GPU pertama (index 0)
        nama_device = torch.cuda.get_device_name(0)
        print(f"[INFO] Menggunakan GPU: {nama_device}")
    else:
        device = 'cpu'
        print("[INFO] GPU tidak ditemukan, menggunakan CPU.")
    
    hasil_folder = "hasil_test"
    os.makedirs(hasil_folder, exist_ok=True)
    
    print("Memuat model YOLO...")
    model = YOLO(model_path)
    print("Model berhasil dimuat.")

    cap = cv2.VideoCapture(video_path)
    model.overrides["verbose"] = False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 1. TENTUKAN SKALA (Misal 50% dari ukuran asli)
    scale_output = 0.5 
    
    # Hitung ukuran baru
    out_width = int(width * scale_output)
    out_height = int(height * scale_output)
    
    # ... (Setup VideoWriter tetap sama) ...
    output_filename = os.path.basename(output_path)
    output_path = os.path.join(hasil_folder, output_filename)
    base, ext = os.path.splitext(output_path)
    counter = 1
    while os.path.exists(output_path):
        output_path = f"{base}_{counter}{ext}"
        counter += 1
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, out_height))

    if lines_coords is None:
        lines_coords = [
            (0, int(height * 0.4), width, int(height * 0.4)),
            (0, int(height * 0.7), width, int(height * 0.7))
        ]

    line_colors = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (0, 0, 255), (255, 128, 0)]

    object_history = {}
    counts_total = {cls: 0 for cls in CLASS_NAMES}
    counted_ids_per_line = [set() for _ in lines_coords]
    counts_per_line = [0] * len(lines_coords)
    
    # === [MODIFIKASI 1] Variabel baru untuk menghitung per Garis DAN per Kelas ===
    # Struktur: { indeks_garis: { '1': 0, '2': 0, ... }, ... }
    counts_detailed = {i: {cls: 0 for cls in CLASS_NAMES} for i in range(len(lines_coords))}
    # ============================================================================

    crossing_events = []
    last_counted_time = {} 

    frame_count = 0
    last_percent = -1
    start_time = datetime.datetime.now()

    print(f"Mulai memproses {video_path} ...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        current_video_time = frame_count / fps
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", device=device)
        annotated_frame = frame.copy()

        for i, (x1, y1, x2, y2) in enumerate(lines_coords):
            color = line_colors[i % len(line_colors)]
            cv2.line(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f"Line {i+1}: {counts_per_line[i]}", (x1 + 10, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if results[0].boxes.id is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf < 0.1: continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                obj_id = int(box.id[0])
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "Unknown"
                x_ref, y_ref = int((x1 + x2) / 2), y2

                if obj_id in object_history:
                    prev_pos = object_history[obj_id]
                    for i, (x1_line, y1_line, x2_line, y2_line) in enumerate(lines_coords):
                        if is_crossing_line(prev_pos, (x_ref, y_ref), (x1_line, y1_line), (x2_line, y2_line)) \
                                and (obj_id not in counted_ids_per_line[i]):
                            
                            counts_total[class_name] += 1
                            counts_per_line[i] += 1
                            counted_ids_per_line[i].add(obj_id)
                            
                            # === [MODIFIKASI 2] Tambahkan hitungan ke variable detail ===
                            if class_name in counts_detailed[i]:
                                counts_detailed[i][class_name] += 1
                            # ============================================================
                            
                            last_counted_time[obj_id] = current_video_time

                            video_timestamp_delta = datetime.timedelta(seconds=current_video_time)
                            video_timestamp = str(video_timestamp_delta)
                            if '.' in video_timestamp:
                                parts = video_timestamp.split('.')
                                video_timestamp = parts[0] + '.' + parts[1][:3]
                            
                            crossing_events.append({
                                'Garis': f"Garis {i+1}",
                                'Jenis Kendaraan': class_name,
                                'ID Objek': obj_id,
                                'Waktu Video': video_timestamp,
                                'Frame': frame_count
                            })
                
                object_history[obj_id] = (x_ref, y_ref)

                # ... (Bagian Visualisasi Gambar Kotak & Teks tetap sama) ...
                final_color = CLASS_COLORS.get(class_name, (0, 255, 0))
                thickness = 2
                label = f"ID:{obj_id} {class_name}"
                text_color = (255, 255, 255)
                
                if obj_id in last_counted_time:
                    time_diff = current_video_time - last_counted_time[obj_id]
                    if time_diff < 0.5: 
                        final_color = (255, 255, 255)
                        thickness = 6
                        label = f"ID:{obj_id} [OK]"
                        text_color = (0, 0, 0)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), final_color, thickness)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), final_color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        frame_resized = cv2.resize(annotated_frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
        out.write(frame_resized)
        frame_count += 1
        
        if total_frames > 0:
            percent = int((frame_count / total_frames) * 100)
            if percent > last_percent:
                sys.stdout.write(f"\rProgres: {percent}% ({frame_count}/{total_frames})")
                sys.stdout.flush()
                if progress_callback: progress_callback(percent)
                last_percent = percent

    cap.release()
    out.release()
    print("\n✅ Selesai! Video hasil disimpan di:", output_path)
    
    end_time = datetime.datetime.now()
    total_duration = end_time - start_time
    print(f"⏱️ Waktu total proses: {total_duration}")

    # === Simpan Log Detail (Tetap Sama) ===
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    log_csv_path = os.path.join(hasil_folder, f"{base_name}_log_detail.csv")
    # ... (logika nama file unik log detail) ...
    base, ext = os.path.splitext(log_csv_path)
    counter = 1
    while os.path.exists(log_csv_path):
        log_csv_path = f"{base}_{counter}{ext}"
        counter += 1

    try:
        with open(log_csv_path, mode="w", newline="", encoding="utf-8") as file:
            if crossing_events:
                headers = crossing_events[0].keys()
                writer = csv.DictWriter(file, fieldnames=headers)
                writer.writeheader()
                writer.writerows(crossing_events)
            else:
                file.write("Tidak ada kendaraan yang terdeteksi melintasi garis.\n")
        print(f"📄 Log detail disimpan ke: {log_csv_path}")
    except Exception as e:
        print(f"Gagal menyimpan log detail CSV: {e}")

    # === Simpan Ringkasan (MODIFIKASI UTAMA DI SINI) ===
    summary_csv_path = os.path.join(hasil_folder, f"{base_name}_ringkasan.csv")
    base, ext = os.path.splitext(summary_csv_path)
    counter = 1
    while os.path.exists(summary_csv_path):
        summary_csv_path = f"{base}_{counter}{ext}"
        counter += 1

    try:
        with open(summary_csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer_summary = csv.writer(file)
            
            # 1. Ringkasan Global (Semua Garis)
            writer_summary.writerow(["=== TOTAL SEMUA GARIS (PER KELAS) ==="])
            writer_summary.writerow(["Kelas", "Jumlah Total"])
            total_kendaraan = 0
            for cls, val in counts_total.items():
                writer_summary.writerow([cls, val])
                total_kendaraan += val
            writer_summary.writerow(["GRAND TOTAL", total_kendaraan])
            writer_summary.writerow([]) 

            # 2. Ringkasan Per Garis (Total Only)
            writer_summary.writerow(["=== TOTAL PER GARIS ==="])
            writer_summary.writerow(["Garis", "Jumlah Total"])
            for i, val in enumerate(counts_per_line):
                writer_summary.writerow([f"Garis {i+1}", val])
            writer_summary.writerow([])

            # === [MODIFIKASI 3] Menulis Detail: Garis | Kelas | Jumlah ===
            writer_summary.writerow(["=== DETAIL PER GARIS & KELAS ==="])
            writer_summary.writerow(["Garis", "Kelas", "Jumlah"])
            
            for i in sorted(counts_detailed.keys()):
                nama_garis = f"Garis {i+1}"
                # Loop setiap kelas
                for cls in CLASS_NAMES:
                    jumlah = counts_detailed[i][cls]
                    # Kita tulis semua kelas meskipun jumlahnya 0 agar rapi, 
                    # atau Anda bisa tambahkan 'if jumlah > 0:' jika ingin yang ada isinya saja.
                    writer_summary.writerow([nama_garis, cls, jumlah])
            
            writer_summary.writerow([])
            # ==============================================================

            writer_summary.writerow(["=== WAKTU PEMROSESAN ==="])
            writer_summary.writerow(["Waktu Mulai", start_time.strftime("%Y-%m-%d %H:%M:%S")])
            writer_summary.writerow(["Waktu Selesai", end_time.strftime("%Y-%m-%d %H:%M:%S")])
            writer_summary.writerow(["Durasi", str(total_duration)])

        print(f"📄 Ringkasan disimpan ke: {summary_csv_path}")
    except Exception as e:
        print(f"Gagal menyimpan ringkasan CSV: {e}")

    return counts_total, counts_per_line, crossing_events

def is_crossing_line(p1, p2, line_p1, line_p2):
    # P1 = Posisi Objek Sebelumnya
    # P2 = Posisi Objek Sekarang
    # line_p1, line_p2 = Koordinat Garis Deteksi

    # Fungsi Helper: Menentukan orientasi 3 titik (Counter-Clockwise)
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # Agar dianggap melintas (valid), kedua kondisi harus terpenuhi (True):
    # 1. Pergerakan objek (p1->p2) memotong garis tak terbatas dari line_p1-line_p2
    # 2. Garis deteksi (line_p1->line_p2) memotong garis tak terbatas dari p1-p2
    # Ini membatasi deteksi HANYA pada panjang segmen garis yang digambar.
    
    return (ccw(p1, line_p1, line_p2) != ccw(p2, line_p1, line_p2)) and \
           (ccw(p1, p2, line_p1) != ccw(p1, p2, line_p2))

# if __name__ == "__main__":
#     video_path = "Cars.mp4"
#     model_path = "model/best.pt"
#     output_path = "hasil_deteksi_multi_garis.mp4"

#     counts, per_line, events = jalankan_deteksi_dan_hitung(video_path, model_path, output_path)

#     print("\n=== HASIL COUNTING TOTAL ===")
#     for cls, val in counts.items():
#         print(f"   {cls}: {val}")

#     print("\n=== HASIL PER GARIS ===")
#     for i, val in enumerate(per_line):
#         print(f"   Garis {i+1}: {val}")

#     print(f"\n=== CONTOH LOG EVENT (Total: {len(events)}) ===")
#     for event in events[:10]:
#         print(f"   {event}")

#     hasil_path = os.path.abspath("hasil_test")
#     print(f"\nVideo dan CSV disimpan di folder: {hasil_path}")
#     print("Membuka folder hasil...")
#     if sys.platform.startswith("win"):
#         os.startfile(hasil_path)
#     elif sys.platform == "darwin":
#         os.system(f"open '{hasil_path}'")
#     else:
#         os.system(f"xdg-open '{hasil_path}'")