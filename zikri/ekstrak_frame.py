import cv2
import os
import time
import threading

CCTV_STREAMS = {
    "hk_atp": "https://extstream.hk-opt2.com/LiveApp/streams/756751654695732090756915.m3u8",
    "km60_000b": "https://extstream.hk-opt2.com/LiveApp/streams/381024135984437558197100.m3u8"
}
folder_utama_output = "dataset_images"
ambil_setiap_detik = 1

if not os.path.exists(folder_utama_output):
    os.makedirs(folder_utama_output)

def grab_frames(cctv_id, stream_url):
    print(f"[{cctv_id}] Thread dimulai...")
    
    folder_cctv = os.path.join(folder_utama_output, cctv_id)
    if not os.path.exists(folder_cctv):
        os.makedirs(folder_cctv)
        print(f"[{cctv_id}] Folder '{folder_cctv}' telah dibuat.")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"[{cctv_id}] Error: Gagal membuka stream.")
        return

    saved_count = 0
    last_capture_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{cctv_id}] Stream terputus. Mencoba menyambung kembali...")
            cap.release()
            time.sleep(5)
            cap = cv2.VideoCapture(stream_url)
            continue

        current_time = time.time()
        if current_time - last_capture_time >= ambil_setiap_detik:
            nama_gambar = os.path.join(folder_cctv, f"frame_{cctv_id}_{saved_count:05d}.jpg")
            cv2.imwrite(nama_gambar, frame)
            print(f"[{cctv_id}] Menyimpan {nama_gambar}")
            
            saved_count += 1
            last_capture_time = current_time

if __name__ == '__main__':
    threads = []
    
    print("Memulai pengambilan frame dari semua stream CCTV...")
    
    for cctv_id, stream_url in CCTV_STREAMS.items():
        thread = threading.Thread(target=grab_frames, args=(cctv_id, stream_url), daemon=True)
        threads.append(thread)
        thread.start()
        time.sleep(1)

    print(f"\n{len(threads)} thread telah dimulai untuk {len(CCTV_STREAMS)} stream.")
    print("Program berjalan di latar belakang. Tekan Ctrl+C di terminal ini untuk menghentikan semua proses.")

    try:

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCtrl+C terdeteksi. Menghentikan semua thread pekerja...")
    
    print("Program Selesai.")