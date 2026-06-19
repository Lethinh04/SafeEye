"""
SafeEye Demo API - Flask server để xử lý nhận diện vật thể từ webcam
Chạy: python src/model/demo_api.py
Port: 5050
"""

# === BẮT BUỘC: Đặt CUDA_VISIBLE_DEVICES TRƯỚC KHI IMPORT BẤT KỲ THƯ VIỆN CUDA NÀO ===
# GPU 1 (NVIDIA GeForce GTX 1650) sẽ trở thành device duy nhất mà process này nhìn thấy.
# Điều này ép cả PyTorch, ONNX Runtime, và mọi thư viện CUDA chỉ dùng GPU 1.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Đảm bảo thứ tự GPU khớp với nvidia-smi

import base64
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app)  # Cho phép cross-origin từ Node.js Express

import torch

# === CẤU HÌNH GPU ===
# Sau khi đặt CUDA_VISIBLE_DEVICES=1, GPU 1 (NVIDIA) trở thành cuda:0 trong PyTorch
if not torch.cuda.is_available():
    raise RuntimeError("[SafeEye] LỖI: Không tìm thấy GPU NVIDIA CUDA. Bắt buộc phải chạy trên GPU 1 (NVIDIA).")

CUDA_DEVICE = "cuda:0"  # Thực tế là GPU 1 vật lý nhờ CUDA_VISIBLE_DEVICES=1
print(f"[SafeEye] === GPU CONFIGURATION ===")
print(f"[SafeEye] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"[SafeEye] PyTorch device: {CUDA_DEVICE}")
print(f"[SafeEye] GPU Name: {torch.cuda.get_device_name(0)}")
print(f"[SafeEye] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"[SafeEye] Số GPU nhìn thấy: {torch.cuda.device_count()}")
print(f"[SafeEye] ===========================")

# Load model YOLOv8 trên GPU (dùng .onnx với onnxruntime-gpu trên CUDA)
print(f"[SafeEye] Đang tải model YOLOv8 (.onnx) trên GPU 1 (NVIDIA)...")
model = YOLO("yolov8n.onnx", task="detect")
names = model.names

# === RESIZE cho inference (giữ đúng tỷ lệ 16:9 của camera gốc 4608x2592) ===
INFERENCE_WIDTH = 640
INFERENCE_HEIGHT = 360

# Đã loại bỏ model TTS để tối ưu FPS. Giọng nói sẽ được phát thông qua Web Browser hoặc Raspberry Pi.

print("[SafeEye] Đang tải model YOLO nhận diện tiền VN (.onnx)...")
try:
    money_model_yolo = YOLO("d:/CE180136/SE/K7/EXE101/FINAL WEB/SafeEye/src/model/money_yolov8.onnx", task="detect")
    money_names_yolo = money_model_yolo.names
    print("[SafeEye] Tải model tiền YOLO (.onnx) thành công, classes:", money_names_yolo)
except Exception as e:
    print("[SafeEye] Lỗi tải model tiền YOLO:", e)
    money_model_yolo = None

try:
    best_model_yolo = YOLO("d:/CE180136/SE/K7/EXE101/FINAL WEB/SafeEye/src/model/best.onnx", task="detect")
    print("[SafeEye] Tải model vật cản (.onnx) thành công")
except Exception as e:
    print("[SafeEye] Lỗi tải model vật cản (best.onnx):", e)
    best_model_yolo = None

BEST_LABELS_VI = {
    0: "nắp cống đóng",
    1: "nắp cống mở",
    2: "bậc vỉa hè"
}

BEST_COLORS = {
    0: "#ef4444", # Đỏ
    1: "#ef4444", # Đỏ
    2: "#ef4444", # Đỏ
}

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
    
    if frame is not None:
        frame = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        
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

def hex_to_bgr(hex_color: str) -> tuple:
    """Convert hex color string (#RRGGBB) to BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip('#')
    try:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r) # OpenCV uses BGR
    except:
        return (255, 255, 255)

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to RGB tuple for PIL."""
    hex_color = hex_color.lstrip('#')
    try:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except:
        return (255, 255, 255)

# === PIL cho tiếng Việt Unicode (cv2.putText không hỗ trợ Unicode) ===
from PIL import Image, ImageDraw, ImageFont

# Tải font hỗ trợ tiếng Việt (dùng Arial có sẵn trên Windows)
try:
    _pil_font = ImageFont.truetype("arial.ttf", 16)
    _pil_font_small = ImageFont.truetype("arial.ttf", 14)
    print("[SafeEye] Đã tải font Arial cho tiếng Việt")
except:
    _pil_font = ImageFont.load_default()
    _pil_font_small = _pil_font
    print("[SafeEye] Không tìm thấy Arial, dùng font mặc định")

def draw_text_vi(frame, text, position, color_hex, font=None):
    """Vẽ chữ tiếng Việt lên frame OpenCV bằng PIL."""
    if font is None:
        font = _pil_font
    color_rgb = hex_to_rgb(color_hex)
    
    # Convert OpenCV BGR -> RGB -> PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Vẽ viền đen (outline) để dễ đọc trên mọi nền
    x, y = position
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=color_rgb)
    
    # Convert PIL -> OpenCV BGR
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)



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

        # Chạy YOLO inference SONG SONG (3 model cùng lúc, tận dụng overlap preprocessing/postprocessing)
        conf_threshold = float(data.get("conf", 0.5))  # ngưỡng mặc định 0.5 để giảm false positive
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_general = executor.submit(model, frame, classes=TARGET_CLASSES, conf=conf_threshold, verbose=False, imgsz=320, device=CUDA_DEVICE)
            future_money = executor.submit(money_model_yolo, frame, conf=0.6, verbose=False, imgsz=320, device=CUDA_DEVICE) if money_model_yolo else None
            future_best = executor.submit(best_model_yolo, frame, conf=0.4, verbose=False, imgsz=320, device=CUDA_DEVICE) if best_model_yolo else None
            
            results = future_general.result()
            m_results = future_money.result() if future_money else []
            b_results = future_best.result() if future_best else []

        detections = []
        filtered_detections = []  # Dùng khi DEBUG_MODE = True
        class_counts = {}
        _text_draw_list = []  # Thu thập text để vẽ 1 lần bằng PIL (hỗ trợ tiếng Việt)

        # === CHẾ ĐỘ DEBUG: Đặt False để tăng FPS (bỏ xây dựng danh sách filtered) ===
        DEBUG_MODE = False

        # Ngưỡng diện tích riêng cho từng vật thể (pixel²)
        MIN_AREA_PER_CLASS = {
            0:  3000,   # person        - người
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
                conf = float(box.conf[0])
                label_vi = CLASS_LABELS_VI.get(cls, names[cls])

                if box_area < min_area:
                    # DEBUG: Trả về vật bị lọc với khung xám
                    if DEBUG_MODE:
                        filtered_detections.append({
                            "class_id": cls,
                            "label": f"[LỌC] {label_vi} ({box_w}x{box_h}) < {min_area}",
                            "label_tts": "",
                            "label_en": names[cls],
                            "confidence": round(conf, 3),
                            "color": "#6b7280",
                            "filtered": True,
                            "bbox": {
                                "x1": x1, "y1": y1,
                                "x2": x2, "y2": y2,
                                "x1n": round(x1 / w, 4),
                                "y1n": round(y1 / h, 4),
                                "x2n": round(x2 / w, 4),
                                "y2n": round(y2 / h, 4),
                            }
                        })
                    continue

                label_vi = CLASS_LABELS_VI.get(cls, names[cls])
                color_hex = CLASS_COLORS.get(cls, "#ffffff")
                color_bgr = hex_to_bgr(color_hex)

                detections.append({
                    "class_id": cls,
                    "label": f"{label_vi} ({box_w}x{box_h})",
                    "label_tts": label_vi,
                    "label_en": names[cls],
                    "confidence": round(conf, 3),
                    "color": color_hex,
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
                
                # DRAW BOX ON FRAME (rectangle = OpenCV nhanh, text = thu thập để vẽ 1 lần bằng PIL)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
                label_text = f"{label_vi} {int(conf*100)}%"
                _text_draw_list.append((label_text, (x1, max(y1 - 22, 2)), color_hex))

        # Xử lý kết quả model tiền
        if money_model_yolo is not None:
            try:
                for r in m_results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        box_w = x2 - x1
                        box_h = y2 - y1
                        box_area = box_w * box_h
                        cls = int(box.cls[0])
                        m_label_raw = money_names_yolo[cls]

                        if box_area < MIN_AREA_MONEY:
                            if DEBUG_MODE:
                                filtered_detections.append({
                                    "class_id": 999 + cls,
                                    "label": f"[LỌC] Tiền {m_label_raw} VNĐ ({box_w}x{box_h}) < {MIN_AREA_MONEY}",
                                    "label_tts": "",
                                    "label_en": f"Money {m_label_raw}",
                                    "confidence": round(float(box.conf[0]), 3),
                                    "color": "#6b7280",
                                    "filtered": True,
                                    "bbox": {
                                        "x1": x1, "y1": y1,
                                        "x2": x2, "y2": y2,
                                        "x1n": round(x1 / w, 4),
                                        "y1n": round(y1 / h, 4),
                                        "x2n": round(x2 / w, 4),
                                        "y2n": round(y2 / h, 4),
                                    }
                                })
                            continue

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
                        m_color_hex = MONEY_COLORS.get(str(m_label), "#facc15")
                        m_color_bgr = hex_to_bgr(m_color_hex)
                        
                        detections.append({
                            "class_id": 999 + cls,
                            "label": f"{label_vi_money} ({box_w}x{box_h})",
                            "label_tts": label_vi_money,
                            "label_en": f"Money {m_label}",
                            "confidence": round(m_conf, 3),
                            "color": m_color_hex,
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
                        
                        # DRAW BOX ON FRAME
                        cv2.rectangle(frame, (x1, y1), (x2, y2), m_color_bgr, 2)
                        _text_draw_list.append((f"{label_vi_money} {int(m_conf*100)}%", (x1, max(y1 - 22, 2)), m_color_hex))
            except Exception as e:
                print("[SafeEye] Lỗi nhận diện tiền YOLO:", e)

        # Xử lý kết quả model vật cản bằng YOLO
        if best_model_yolo is not None:
            try:
                for r in b_results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        box_w = x2 - x1
                        box_h = y2 - y1
                        box_area = box_w * box_h
                        cls = int(box.cls[0])
                        b_conf = float(box.conf[0])
                        label_vi_best = BEST_LABELS_VI.get(cls, f"Vật cản {cls}")

                        if box_area < MIN_AREA_OBSTACLE:
                            if DEBUG_MODE:
                                filtered_detections.append({
                                    "class_id": 2000 + cls,
                                    "label": f"[LỌC] {label_vi_best} ({box_w}x{box_h}) < {MIN_AREA_OBSTACLE}",
                                    "label_tts": "",
                                    "label_en": f"Obstacle {cls}",
                                    "confidence": round(b_conf, 3),
                                    "color": "#6b7280",
                                    "filtered": True,
                                    "bbox": {
                                        "x1": x1, "y1": y1,
                                        "x2": x2, "y2": y2,
                                        "x1n": round(x1 / w, 4),
                                        "y1n": round(y1 / h, 4),
                                        "x2n": round(x2 / w, 4),
                                        "y2n": round(y2 / h, 4),
                                    }
                                })
                            continue

                        b_color_hex = BEST_COLORS.get(cls, "#ef4444")
                        b_color_bgr = hex_to_bgr(b_color_hex)
                        
                        detections.append({
                            "class_id": 2000 + cls,
                            "label": f"{label_vi_best} ({box_w}x{box_h})",
                            "label_tts": label_vi_best,
                            "label_en": f"Obstacle {cls}",
                            "confidence": round(b_conf, 3),
                            "color": b_color_hex,
                            "bbox": {
                                "x1": x1, "y1": y1,
                                "x2": x2, "y2": y2,
                                "x1n": round(x1 / w, 4),
                                "y1n": round(y1 / h, 4),
                                "x2n": round(x2 / w, 4),
                                "y2n": round(y2 / h, 4),
                            }
                        })
                        class_counts[label_vi_best] = class_counts.get(label_vi_best, 0) + 1
                        
                        # DRAW BOX ON FRAME
                        cv2.rectangle(frame, (x1, y1), (x2, y2), b_color_bgr, 2)
                        _text_draw_list.append((f"{label_vi_best} {int(b_conf*100)}%", (x1, max(y1 - 22, 2)), b_color_hex))
            except Exception as e:
                print("[SafeEye] Lỗi nhận diện vật cản YOLO:", e)

        # === VẼ TẤT CẢ TEXT TIẾNG VIỆT 1 LẦN DUY NHẤT BẰNG PIL (tối ưu tốc độ) ===
        if _text_draw_list:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            for text, pos, c_hex in _text_draw_list:
                c_rgb = hex_to_rgb(c_hex)
                x, y = pos
                # Viền đen
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text((x + dx, y + dy), text, font=_pil_font, fill=(0, 0, 0))
                draw.text((x, y), text, font=_pil_font, fill=c_rgb)
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        fps = calculate_fps()

        # Tạo câu nói
        detected_classes_vi = [d.get("label_tts", CLASS_LABELS_VI.get(d["class_id"], d["label"])) for d in detections]
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

            # Đã loại bỏ TTS inference (tts_model) ở đây
            # Client (Web/Pi) sẽ nhận `speech_text` và tự động đọc bằng TTS tích hợp (VD: window.speechSynthesis)
            
            last_spoken_classes = tuple(unique_classes)
            last_speak_time = current_time

        # Encode processed frame back to base64 (quality 60 = cân bằng tốc độ vs chất lượng)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        processed_img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        response_data = {
            "detections": detections,
            "class_counts": class_counts,
            "total": len(detections),
            "fps": fps,
            "frame_size": {"width": w, "height": h},
            "speech": speech_text,
            "audio": audio_base64,
            "processed_image": processed_img_b64
        }

        # DEBUG: Thêm danh sách vật bị lọc vào response
        if DEBUG_MODE and filtered_detections:
            response_data["filtered"] = filtered_detections
            response_data["filtered_count"] = len(filtered_detections)

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # === WARM-UP: Chạy inference giả để GPU khởi tạo sẵn, tránh frame đầu bị chậm ===
    print(f"[SafeEye] Đang warm-up {CUDA_DEVICE}...")
    dummy = np.zeros((INFERENCE_HEIGHT, INFERENCE_WIDTH, 3), dtype=np.uint8)
    model(dummy, verbose=False, imgsz=320, device=CUDA_DEVICE)
    if money_model_yolo:
        money_model_yolo(dummy, verbose=False, imgsz=320, device=CUDA_DEVICE)
    if best_model_yolo:
        best_model_yolo(dummy, verbose=False, imgsz=320, device=CUDA_DEVICE)
    print("[SafeEye] Warm-up hoàn tất!")
    
    print("[SafeEye] API server đang khởi động tại http://localhost:5050")
    print("[SafeEye] Endpoint: POST /detect  |  GET /health")
    app.run(host="0.0.0.0", port=5050, debug=False)