from ultralytics import YOLO
import cv2

# Load model YOLOv8 nano (pretrained COCO)
model = YOLO("besti.pt")

# Buka file video
video_path = "Cars.mp4"
cap = cv2.VideoCapture(video_path)

# Dapatkan info video (ukuran & fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Simpan hasil video ke file baru
out = cv2.VideoWriter("hasil_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek per frame
    results = model(frame)

    # Gambar hasil deteksi pada frame
    annotated_frame = results[0].plot()

    # Tampilkan hasil
    cv2.imshow("Deteksi YOLOv8", annotated_frame)

    # Simpan ke file output
    out.write(annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Bersihkan resource
cap.release()
out.release()
cv2.destroyAllWindows()
