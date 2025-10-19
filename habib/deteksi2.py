from ultralytics import YOLO
import cv2
import sys

def jalankan_deteksi(video_path, model_path="besti.pt", progress_callback=None):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    model.overrides["verbose"] = False  # Nonaktifkan log YOLO bawaan

    # Ambil info video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # File output
    out = cv2.VideoWriter("hasil_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

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
        percent = int((frame_count / total_frames) * 100)
        
        # 🔹 Update progress GUI jika callback tersedia
        if progress_callback:
            progress_callback(percent)
            
        sys.stdout.write(f"\r[{percent:3d}%] Memproses frame {frame_count}/{total_frames}")
        sys.stdout.flush()

    # Bersihkan resource
    cap.release()
    out.release()
    print("\n✅ Deteksi selesai. Hasil disimpan di: hasil_video.mp4")
    
    # 🔹 Beri tahu GUI kalau sudah selesai
    if progress_callback:
        progress_callback(100)
