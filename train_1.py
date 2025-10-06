from ultralytics import YOLO
import os, shutil

# cek folder dataset
# !ls /kaggle/input/12golongan

# buat dataset.yaml
dataset_yaml = """
path: /kaggle/input/12golongan
train: train/images
val: valid/images
test: test/images

nc: 12
names: ["1", "2", "3", "4", "5a", "5b", "6a", "6b", "7a", "7b", "7c", "8"]
"""
with open("dataset.yaml", "w") as f:
    f.write(dataset_yaml)
print(open("dataset.yaml").read())

# load model pre-trained
model = YOLO("yolo11s.pt")

# training
model.train(
    data="dataset.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    workers=2,
    device=0
)

# evaluasi
metrics_val = model.val(split="val")
print("Validation metrics:", metrics_val)

metrics_test = model.val(split="test")
print("Test metrics:", metrics_test)

# prediksi + export
results = model.predict(
    source="/kaggle/input/12golongan/test/images",
    save=True
)
print("Output folder:", results[0].save_dir)

model.export(format="onnx")

# =========================
# SIMPAN MODEL SUPAYA AMAN
# =========================
best_model_path = "/kaggle/working/runs/detect/train/weights/best.pt"
last_model_path = "/kaggle/working/runs/detect/train/weights/last.pt"

# copy ke /kaggle/working (root)
shutil.copy(best_model_path, "/kaggle/working/best.pt")
shutil.copy(last_model_path, "/kaggle/working/last.pt")

# copy juga ke /kaggle/output biar bisa didownload dari tab Output Files
os.makedirs("/kaggle/output", exist_ok=True)
shutil.copy(best_model_path, "/kaggle/output/best.pt")
shutil.copy(last_model_path, "/kaggle/output/last.pt")

print("✅ Model disalin ke /kaggle/working/")
# !ls -lh /kaggle/working/*.pt