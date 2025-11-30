from ultralytics import YOLO
import cv2
import sys
import os

def jalankan_deteksi(video_path, model_path='C:/HDD/TUGAS KULIAH/TrafficCountYOLO/Protel/TrafficApp/Model1/best.pt', progress_callback=None):
    # Cek apakah model ada sebelum loading
    if not os.path.exists(model_path):
        print(f"Error: Model tidak ditemukan di {model_path}")
        return

    print("Sedang memuat model...")
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    model.overrides["verbose"] = False  # Nonaktifkan log YOLO bawaan

    # Ambil info video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # File output
    output_filename = "hasil_video.mp4"
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Mulai deteksi video: {video_path} ({total_frames} frame)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi objek per frame
        results = model(frame)

        # Gambar hasil deteksi
        annotated_frame = results[0].plot()

        # Simpan ke file output
        out.write(annotated_frame)

        frame_count += 1
        # Hitung dan tampilkan progress 0–100%
        if total_frames > 0:
            percent = int((frame_count / total_frames) * 100)
            
            # 🔹 Update progress GUI jika callback tersedia
            if progress_callback:
                progress_callback(percent)
                
            sys.stdout.write(f"\r[{percent:3d}%] Memproses frame {frame_count}/{total_frames}")
            sys.stdout.flush()

    # Bersihkan resource
    cap.release()
    out.release()
    print(f"\n✅ Deteksi selesai. Hasil disimpan di: {output_filename}")
    
    # 🔹 Beri tahu GUI kalau sudah selesai
    if progress_callback:
        progress_callback(100)

# --- BAGIAN EKSEKUSI ---
if __name__ == "__main__":
    # Ganti nama file ini sesuai video yang mau dites
    video_input = 'cctv.mp4' 
    
    if os.path.exists(video_input):
        jalankan_deteksi(video_input)
    else:
        print(f"Error: File '{video_input}' tidak ditemukan. Pastikan path benar.")