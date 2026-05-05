from ultralytics import YOLO
from roboflow import Roboflow
import os

# API anahtarını buraya ekledim
rf = Roboflow(api_key="WfPQq5lEH5KmnAlIk71P")
project = rf.workspace("ivtu900mplm5uodtbx0jiweozdt2").project("telestroke")
dataset = project.version(1).download("yolov8")

# 2. MODELİ YÜKLE
# Sıfırdan bir eğitim için yolov8n.pt kullanıyoruz
model = YOLO('yolov8n.pt') 

# 3. EĞİTİMİ BAŞLAT (MAC M2 İÇİN OPTİMİZE EDİLDİ)
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=99,      # Modeli 100 tur eğitir
    imgsz=640,       # Görüntü boyutu
    device='mps',    # M2 GPU (Metal Performance Shaders) desteği
    plots=True,      # Başarı grafiklerini oluşturur
    name='faststroke_v1' # Eğitim sonuçlarını bu isimle kaydeder
)

