# === BẮT BUỘC: Đặt CUDA_VISIBLE_DEVICES TRƯỚC KHI IMPORT BẤT KỲ THƯ VIỆN CUDA NÀO ===
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
import threading
import concurrent.futures
import requests
import json

# Bắt buộc chạy trên GPU 1 (NVIDIA GeForce GTX 1650)
if not torch.cuda.is_available():
    raise RuntimeError("[SafeEye] LỖI: Không tìm thấy GPU NVIDIA CUDA. Bắt buộc phải chạy trên GPU 1.")
CUDA_DEVICE = "cuda:0"  # Thực tế là GPU 1 vật lý nhờ CUDA_VISIBLE_DEVICES=1
print(f"[SafeEye] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"[SafeEye] GPU đang dùng: {torch.cuda.get_device_name(0)}")
print(f"[SafeEye] Số GPU nhìn thấy: {torch.cuda.device_count()}")

# 1. Load models
model_yolo = YOLO("yolov8n.pt", task="detect")
model_yolo.to(CUDA_DEVICE)
names = model_yolo.names

try:
    money_model_yolo = YOLO("d:/CE180136/SE/K7/EXE101/FINAL WEB/SafeEye/src/model/money_yolov8.pt", task="detect")
    money_model_yolo.to(CUDA_DEVICE)
    money_names_yolo = money_model_yolo.names
    print("[SafeEye] Tải model tiền YOLO thành công, classes:", money_names_yolo)
except Exception as e:
    print("[SafeEye] Lỗi tải model tiền YOLO:", e)
    money_model_yolo = None

try:
    best_model_yolo = YOLO("d:/CE180136/SE/K7/EXE101/FINAL WEB/SafeEye/src/model/best.pt", task="detect")
    best_model_yolo.to(CUDA_DEVICE)
    print("[SafeEye] Tải model vật cản (.pt) thành công")
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

# Đã loại bỏ model TTS để tối ưu FPS. Giọng nói sẽ được phát thông qua Web Browser hoặc Raspberry Pi.

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.1.211:8000/stream.mjpg")

last_speak_time = 0
interval = 5
last_firebase_push_time = 0

def push_to_firebase(data):
    url = "https://sos-app-8ba8b-default-rtdb.asia-southeast1.firebasedatabase.app/detections.json"
    try:
        # Nếu data rỗng, ta có thể đẩy rỗng hoặc thông báo an toàn
        requests.put(url, json=data)
    except Exception as e:
        print(f"[SafeEye] Lỗi đẩy dữ liệu lên Firebase: {e}")

# === CHẾ ĐỘ DEBUG: Đặt True để hiển thị cả vật bị lọc (khung xám) ===
DEBUG_MODE = True

if not cap.isOpened():
    print(f"[SafeEye] LỖI: Không thể kết nối tới IP Camera (http://192.168.1.211:8000/). Hãy kiểm tra lại địa chỉ IP, cổng, hoặc đường dẫn luồng video (ví dụ: /video, /stream.mjpg).")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[SafeEye] Bị mất kết nối với Camera hoặc luồng video kết thúc.")
        break

    frame_small = cv2.resize(frame, (1080, 720), interpolation=cv2.INTER_AREA)

    # Ngưỡng diện tích riêng cho từng vật thể (pixel²)
    MIN_AREA_PER_CLASS = {
        0:  3000,   # person        - người (thân hình dài, dễ nhận)
        1:  2500,   # bicycle       - xe đạp
        2:  5000,   # car           - ô tô (to, chỉ báo khi gần)
        3:  3000,   # motorcycle    - xe máy
        4:  4000,   # airplane      - máy bay
        5:  6000,   # bus           - xe buýt (rất to, chỉ báo khi gần)
        6:  6000,   # train         - tàu hỏa
        7:  5000,   # truck         - xe tải
        8:  3000,   # boat          - thuyền
        9:  800,    # traffic light - đèn giao thông (nhỏ trên cao)
        10: 1500,   # fire hydrant  - họng cứu hỏa (nhỏ dưới đất)
        11: 1000,   # stop sign     - biển dừng (nhỏ trên cao)
        12: 1000,   # parking meter - đồng hồ đỗ xe
        13: 3000,   # bench         - ghế băng
        15: 1500,   # cat           - mèo (nhỏ)
        16: 2000,   # dog           - chó
        19: 4000,   # cow           - bò
        24: 1500,   # backpack      - ba lô
        25: 1500,   # umbrella      - ô dù
        26: 1000,   # handbag       - túi xách (nhỏ)
        28: 2000,   # suitcase      - vali
        39: 800,    # bottle        - chai (nhỏ)
        41: 600,    # cup           - ly/cốc (rất nhỏ)
        56: 3000,   # chair         - ghế
        60: 4000,   # dining table  - bàn ăn (to)
        67: 500,    # cell phone    - điện thoại (rất nhỏ)
    }
    MIN_AREA_MONEY = 1000
    MIN_AREA_OBSTACLE = 3000

    # Nhận diện ĐA LUỒNG (Song song) 3 model cùng lúc
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_general = executor.submit(model_yolo, frame_small, classes=target_classes, conf=0.5, verbose=False, imgsz=320, device=CUDA_DEVICE)
        future_money = executor.submit(money_model_yolo, frame_small, conf=0.6, verbose=False, imgsz=320, device=CUDA_DEVICE) if money_model_yolo else None
        future_best = executor.submit(best_model_yolo, frame_small, conf=0.4, verbose=False, imgsz=320, device=CUDA_DEVICE) if best_model_yolo else None
        
        results = future_general.result()
        m_results = future_money.result() if future_money else []
        b_results = future_best.result() if future_best else []

    detected_classes_vi = set()
    class_counts = {}

    # Xử lý kết quả model chung

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Lọc theo diện tích riêng từng class
            cls = int(box.cls[0])
            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h
            min_area = MIN_AREA_PER_CLASS.get(cls, 4000)
            label_vi = CLASS_LABELS_VI.get(cls, names[cls])

            if box_area < min_area:
                # DEBUG: Vẽ khung xám nét đứt cho vật bị lọc
                if DEBUG_MODE:
                    cv2.rectangle(frame_small, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(frame_small, f"[LOC] {label_vi} ({box_w}x{box_h}) < {min_area}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                continue
            
            detected_classes_vi.add(label_vi.lower())
            class_counts[label_vi.lower()] = class_counts.get(label_vi.lower(), 0) + 1
            
            # Vẽ bounding box
            cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_small, f"{label_vi} ({box_w}x{box_h})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Xử lý kết quả model tiền
    if money_model_yolo is not None:
        for r in m_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Lọc theo diện tích
                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h
                cls = int(box.cls[0])
                m_label = money_names_yolo[cls]
                label_vi_money = f"Tiền {m_label} VNĐ"

                if box_area < MIN_AREA_MONEY:
                    if DEBUG_MODE:
                        cv2.rectangle(frame_small, (x1, y1), (x2, y2), (128, 128, 128), 1)
                        cv2.putText(frame_small, f"[LOC] {label_vi_money} ({box_w}x{box_h}) < {MIN_AREA_MONEY}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                    continue

                detected_classes_vi.add(label_vi_money.lower())
                class_counts[label_vi_money.lower()] = class_counts.get(label_vi_money.lower(), 0) + 1
                
                # Vẽ bounding box
                cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame_small, f"{label_vi_money} ({box_w}x{box_h})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Xử lý kết quả model vật cản (nắp cống, bậc vỉa hè)
    if best_model_yolo is not None:
        for r in b_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Lọc theo diện tích
                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h
                cls = int(box.cls[0])
                label_vi_best = BEST_LABELS_VI.get(cls, f"Vật cản {cls}")

                if box_area < MIN_AREA_OBSTACLE:
                    if DEBUG_MODE:
                        cv2.rectangle(frame_small, (x1, y1), (x2, y2), (128, 128, 128), 1)
                        cv2.putText(frame_small, f"[LOC] {label_vi_best} ({box_w}x{box_h}) < {MIN_AREA_OBSTACLE}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                    continue

                detected_classes_vi.add(label_vi_best)
                class_counts[label_vi_best] = class_counts.get(label_vi_best, 0) + 1
                
                # Vẽ bounding box (màu đỏ)
                cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_small, f"{label_vi_best} ({box_w}x{box_h})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    current_time = time.time()
    
    # Gửi dữ liệu Firebase mỗi giây
    if current_time - last_firebase_push_time > 1.0:
        threading.Thread(target=push_to_firebase, args=(class_counts,), daemon=True).start()
        last_firebase_push_time = current_time

    if current_time - last_speak_time > interval:
        if detected_classes_vi:
            sentence = "Cảnh báo! phía trước có " + ", ".join(list(detected_classes_vi))
            print("LOG CẢNH BÁO (Sẽ gửi qua Web/Pi để phát loa):", sentence)
            last_speak_time = current_time

    cv2.imshow("Detection + TTS", frame_small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()