import cv2
import time
import torch
import numpy as np
import sounddevice as sd
from ultralytics import YOLO
from transformers import VitsModel, AutoTokenizer
import threading

# 1. Load models
model_yolo = YOLO("yolov8n.pt")
names = model_yolo.names

try:
    money_model_yolo = YOLO("d:/CE180136/SE/K7/EXE101/FINAL/SafeEye/src/model/money_yolov8.pt")
    money_names_yolo = money_model_yolo.names
    print("[SafeEye] Tải model tiền YOLO thành công, classes:", money_names_yolo)
except Exception as e:
    print("[SafeEye] Lỗi tải model tiền YOLO:", e)
    money_model_yolo = None

try:
    best_model_yolo = YOLO("d:/CE180136/SE/K7/EXE101/FINAL WEB/SafeEye/src/model/best.pt")
    print("[SafeEye] Tải model vật cản (best.pt) thành công")
except Exception as e:
    print("[SafeEye] Lỗi tải model vật cản (best.pt):", e)
    best_model_yolo = None

BEST_LABELS_VI = {
    0: "nắp cống đóng",
    1: "nắp cống mở",
    2: "bậc vỉa hè"
}

target_classes = [
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    4,   # airplane
    5,   # bus
    6,   # train
    7,   # truck
    8,   # boat
    9,   # traffic light
    10,  # fire hydrant
    11,  # stop sign
    12,  # parking meter
    13,  # bench
    15,  # cat
    16,  # dog
    19,  # cow
    24,  # backpack
    25,  # umbrella
    26,  # handbag
    28,  # suitcase
    39,  # bottle
    41,  # cup
    56,  # chair
    60,  # dining table
    67,  # cell phone
]

CLASS_LABELS_VI = {
    0:  "Người",
    1:  "Xe đạp",
    2:  "Ô tô",
    3:  "Xe máy",
    4:  "Máy bay",
    5:  "Xe buýt",
    6:  "Tàu hỏa",
    7:  "Xe tải",
    8:  "Thuyền",
    9:  "Đèn giao thông",
    10: "Họng cứu hỏa",
    11: "Biển dừng",
    12: "Đồng hồ đỗ xe",
    13: "Ghế băng",
    15: "Mèo",
    16: "Chó",
    19: "Bò",
    24: "Ba lô",
    25: "Ô/Dù",
    26: "Túi xách",
    28: "Vali",
    39: "Chai/Bình",
    41: "Ly/Cốc",
    56: "Ghế",
    60: "Bàn ăn",
    67: "Điện thoại",
}

tts_model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

def speak(text):
    def run():
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = tts_model(**inputs).waveform
        audio = output.squeeze().cpu().numpy()
        sd.stop()
        sd.play(audio, samplerate=tts_model.config.sampling_rate)

    threading.Thread(target=run, daemon=True).start()

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.1.235:5000/video")

last_speak_time = 0
interval = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_small = cv2.resize(frame, (640, 480))

    # Nhận diện vật thể thường
    results = model_yolo(frame_small, classes=target_classes, conf=0.5, verbose=False)

    detected_classes_vi = set()

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            
            label_vi = CLASS_LABELS_VI.get(cls, names[cls])
            detected_classes_vi.add(label_vi.lower())
            
            # Vẽ bounding box
            cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_small, label_vi, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Nhận diện tiền
    if money_model_yolo is not None:
        m_results = money_model_yolo(frame_small, conf=0.6, verbose=False)
        for r in m_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                m_label = money_names_yolo[cls]
                label_vi_money = f"Tiền {m_label} VNĐ"
                detected_classes_vi.add(label_vi_money.lower())
                
                # Vẽ bounding box
                cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame_small, label_vi_money, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Nhận diện vật cản (nắp cống, bậc vỉa hè)
    if best_model_yolo is not None:
        b_results = best_model_yolo(frame_small, conf=0.4, verbose=False)
        for r in b_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                
                label_vi_best = BEST_LABELS_VI.get(cls, f"Vật cản {cls}")
                detected_classes_vi.add(label_vi_best)
                
                # Vẽ bounding box (màu đỏ)
                cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_small, label_vi_best, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    current_time = time.time()
    if current_time - last_speak_time > interval:
        if detected_classes_vi:
            sentence = "Cảnh báo! phía trước có " + ", ".join(list(detected_classes_vi))
            print("TTS:", sentence)
            speak(sentence)
            last_speak_time = current_time

    cv2.imshow("Detection + TTS", frame_small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()