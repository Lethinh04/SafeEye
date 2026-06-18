"""
SafeEye Demo API - Flask server để xử lý nhận diện vật thể từ webcam
Chạy: python src/model/demo_api.py
Port: 5050
"""

import base64
import io
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from keras.models import load_model
import pickle

from transformers import VitsModel, AutoTokenizer
import sounddevice as sd
import threading
import torch
import soundfile as sf
import base64
import io

app = Flask(__name__)
CORS(app)  # Cho phép cross-origin từ Node.js Express

# Load model YOLOv8
print("[SafeEye] Đang tải model YOLOv8...")
model = YOLO("yolov8n.pt")
names = model.names

print("[SafeEye] Đang tải model TTS tiếng Việt...")
tts_model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

print("[SafeEye] Đang tải model YOLO nhận diện tiền VN...")
try:
    money_model_yolo = YOLO("d:/CE180136/SE/K7/EXE101/FINAL/SafeEye/src/model/money_yolov8.pt")
    money_names_yolo = money_model_yolo.names
    print("[SafeEye] Tải model tiền YOLO thành công, classes:", money_names_yolo)
except Exception as e:
    print("[SafeEye] Lỗi tải model tiền YOLO:", e)
    money_model_yolo = None

# Các class cần nhận diện (theo demo.py gốc)
TARGET_CLASSES = [
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

# Màu sắc cho từng class
CLASS_COLORS = {
    0:  "#3b82f6",  # person        - xanh dương
    1:  "#06b6d4",  # bicycle       - xanh cyan
    2:  "#f43f5e",  # car           - hồng đỏ
    3:  "#a855f7",  # motorcycle    - tím
    4:  "#0ea5e9",  # airplane      - xanh trời
    5:  "#f59e0b",  # bus           - vàng cam
    6:  "#6366f1",  # train         - indigo
    7:  "#dc2626",  # truck         - đỏ
    8:  "#22d3ee",  # boat          - cyan nhạt
    9:  "#facc15",  # traffic light - vàng
    10: "#fb7185",  # fire hydrant  - hồng
    11: "#ef4444",  # stop sign     - đỏ tươi
    12: "#a3a3a3",  # parking meter - xám
    13: "#84cc16",  # bench         - xanh lá vàng
    15: "#f97316",  # cat           - cam
    16: "#b45309",  # dog           - nâu
    19: "#4d7c0f",  # cow           - xanh lá đậm
    24: "#7c3aed",  # backpack      - tím đậm
    25: "#0284c7",  # umbrella      - xanh biển
    26: "#db2777",  # handbag       - hồng đậm
    28: "#64748b",  # suitcase      - xám xanh
    39: "#10b981",  # bottle        - xanh lá
    41: "#f97316",  # cup           - cam
    56: "#f59e0b",  # chair         - vàng cam
    60: "#8b5cf6",  # dining table  - tím nhạt
    67: "#ef4444",  # cell phone    - đỏ
}

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

fps_counter = {"last_time": time.time(), "count": 0, "fps": 0}

VI_MAP = {
    "person":        "người",
    "bicycle":       "xe đạp",
    "car":           "ô tô",
    "motorcycle":    "xe máy",
    "airplane":      "máy bay",
    "bus":           "xe buýt",
    "train":         "tàu hỏa",
    "truck":         "xe tải",
    "boat":          "thuyền",
    "traffic light": "đèn giao thông",
    "fire hydrant":  "họng cứu hỏa",
    "stop sign":     "biển dừng",
    "parking meter": "đồng hồ đỗ xe",
    "bench":         "ghế băng",
    "cat":           "mèo",
    "dog":           "chó",
    "cow":           "bò",
    "backpack":      "ba lô",
    "umbrella":      "ô dù",
    "handbag":       "túi xách",
    "suitcase":      "vali",
    "bottle":        "bình nước",
    "cup":           "cốc",
    "chair":         "ghế",
    "dining table":  "bàn",
    "cell phone":    "điện thoại",
}

last_speak_time = 0
SPEAK_INTERVAL = 5  # giây
last_spoken_classes = set()

def decode_base64_image(data_url: str) -> np.ndarray:
    """Giải mã base64 image từ canvas.toDataURL() sang numpy array."""
    # Bỏ tiêu đề "data:image/jpeg;base64,"
    if "," in data_url:
        data_url = data_url.split(",")[1]
    img_bytes = base64.b64decode(data_url)
    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return frame


def calculate_fps():
    global fps_counter
    current_time = time.time()
    fps_counter["count"] += 1
    elapsed = current_time - fps_counter["last_time"]
    if elapsed >= 1.0:
        fps_counter["fps"] = round(fps_counter["count"] / elapsed, 1)
        fps_counter["count"] = 0
        fps_counter["last_time"] = current_time
    return fps_counter["fps"]

def speak(text):
    def run():
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = tts_model(**inputs).waveform
        audio = output.squeeze().cpu().numpy()
        sd.stop()
        sd.play(audio, samplerate=tts_model.config.sampling_rate)

    threading.Thread(target=run, daemon=True).start()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "yolov8n", "message": "SafeEye AI sẵn sàng"})


@app.route("/detect", methods=["POST"])
def detect():
    """
    Nhận base64 image, chạy inference, trả về kết quả detection.
    Request body: { "image": "data:image/jpeg;base64,..." }
    Response: { "detections": [...], "fps": N, "total": N }
    """
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Thiếu trường 'image'"}), 400

        frame = decode_base64_image(data["image"])
        if frame is None:
            return jsonify({"error": "Không thể giải mã ảnh"}), 400

        h, w = frame.shape[:2]

        # Chạy YOLO inference
        conf_threshold = float(data.get("conf", 0.5))  # ngưỡng mặc định 0.5 để giảm false positive
        results = model(frame, classes=TARGET_CLASSES, conf=conf_threshold, verbose=False)

        detections = []
        class_counts = {}

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                label_vi = CLASS_LABELS_VI.get(cls, names[cls])
                color = CLASS_COLORS.get(cls, "#ffffff")

                detections.append({
                    "class_id": cls,
                    "label": label_vi,
                    "label_en": names[cls],
                    "confidence": round(conf, 3),
                    "color": color,
                    "bbox": {
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        # normalized (0..1) cho frontend scale theo kích thước video
                        "x1n": round(x1 / w, 4),
                        "y1n": round(y1 / h, 4),
                        "x2n": round(x2 / w, 4),
                        "y2n": round(y2 / h, 4),
                    }
                })

                class_counts[label_vi] = class_counts.get(label_vi, 0) + 1

        # Nhận diện tiền VN bằng YOLO
        if money_model_yolo is not None:
            try:
                # Chạy YOLO cho tiền (conf 0.4 để nhạy hơn)
                m_results = money_model_yolo(frame, conf=0.4, verbose=False)
                for r in m_results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        m_conf = float(box.conf[0])
                        
                        m_label = money_names_yolo[cls] # ví dụ '1000', '5000'
                        label_vi_money = f"Tiền {m_label} VNĐ"
                        
                        MONEY_COLORS = {
                            "1000": "#64748b",   # Xám
                            "2000": "#78716c",   # Nâu xám
                            "5000": "#3b82f6",   # Xanh dương đậm
                            "10000": "#eab308",  # Vàng
                            "20000": "#0ea5e9",  # Xanh nhạt
                            "50000": "#ec4899",  # Hồng
                            "100000": "#22c55e", # Xanh lá
                            "200000": "#ef4444", # Đỏ
                            "500000": "#14b8a6", # Xanh ngọc
                        }
                        m_color = MONEY_COLORS.get(str(m_label), "#facc15")
                        
                        detections.append({
                            "class_id": 999 + cls,
                            "label": label_vi_money,
                            "label_en": f"Money {m_label}",
                            "confidence": round(m_conf, 3),
                            "color": m_color,
                            "bbox": {
                                "x1": x1, "y1": y1,
                                "x2": x2, "y2": y2,
                                "x1n": round(x1 / w, 4),
                                "y1n": round(y1 / h, 4),
                                "x2n": round(x2 / w, 4),
                                "y2n": round(y2 / h, 4),
                            }
                        })
                        class_counts[label_vi_money] = class_counts.get(label_vi_money, 0) + 1
            except Exception as e:
                print("[SafeEye] Lỗi nhận diện tiền YOLO:", e)

        fps = calculate_fps()

        # Tạo câu nói
        detected_classes_vi = [CLASS_LABELS_VI.get(d["class_id"], d["label"]) for d in detections]
        unique_classes = list(set(detected_classes_vi))
        unique_classes.sort()

        global last_speak_time, last_spoken_classes
        current_time = time.time()
        
        audio_base64 = ""
        speech_text = ""

        if tuple(unique_classes) != last_spoken_classes or (current_time - last_speak_time > SPEAK_INTERVAL):
            if unique_classes:
                speech_text = "Cảnh báo phía trước có " + ", ".join(unique_classes)
            else:
                speech_text = "Phía trước an toàn"

            try:
                # Tạo audio từ text
                inputs = tokenizer(speech_text, return_tensors="pt")
                with torch.no_grad():
                    output = tts_model(**inputs).waveform

                audio = output.squeeze().cpu().numpy()

                buf = io.BytesIO()
                sf.write(buf, audio, tts_model.config.sampling_rate, format="WAV")
                audio_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                
                last_spoken_classes = tuple(unique_classes)
                last_speak_time = current_time
            except Exception as e:
                print(f"[SafeEye] Lỗi TTS: {e}")
        else:
            # Nếu chưa quá 5 giây và không đổi class, ta không sinh audio để tránh lag
            pass

        return jsonify({
            "detections": detections,
            "class_counts": class_counts,
            "total": len(detections),
            "fps": fps,
            "frame_size": {"width": w, "height": h},
            "speech": speech_text,
            "audio": audio_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("[SafeEye] API server đang khởi động tại http://localhost:5050")
    print("[SafeEye] Endpoint: POST /detect  |  GET /health")
    app.run(host="0.0.0.0", port=5050, debug=False)