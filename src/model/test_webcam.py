import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def get_model_path(filename):
    return os.path.join(MODEL_DIR, filename)

# ==================== CẤU HÌNH GPU ====================
CUDA_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[Test Webcam] Device: {CUDA_DEVICE}")

# ==================== LOAD MODELS ====================
print("[Test Webcam] Đang tải các model...")
try:
    model = YOLO(get_model_path("yolov8n.onnx"), task="detect")
    names = model.names
except Exception as e:
    print("Lỗi load yolov8n:", e)
    model = None

try:
    money_model = YOLO(get_model_path("money_yolov8.onnx"), task="detect")
    money_names = money_model.names
except Exception as e:
    print("Lỗi load money_yolov8:", e)
    money_model = None

try:
    best_model = YOLO(get_model_path("best.onnx"), task="detect")
except Exception as e:
    print("Lỗi load best.onnx:", e)
    best_model = None

try:
    curb_model = YOLO(get_model_path("curb.onnx"), task="detect")
except Exception as e:
    print("Lỗi load curb.onnx:", e)
    curb_model = None

# ==================== CẤU HÌNH HIỂN THỊ ====================
def hex_to_bgr(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip('#')
    try:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)
    except:
        return (255, 255, 255)

def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip('#')
    try:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except:
        return (255, 255, 255)

try:
    _pil_font = ImageFont.truetype("arial.ttf", 20)
except:
    _pil_font = ImageFont.load_default()

# Các dict phụ trợ như ở demo_api
TARGET_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 19, 24, 25, 26, 28, 39, 41, 56, 60, 67]
CLASS_LABELS_VI = {
    0: "Người", 1: "Xe đạp", 2: "Ô tô", 3: "Xe máy", 5: "Xe buýt", 7: "Xe tải",
    9: "Đèn giao thông", 11: "Biển dừng", 15: "Mèo", 16: "Chó", 56: "Ghế"
}
BEST_LABELS_VI = {0: "nắp cống đóng", 1: "nắp cống mở", 2: "bậc vỉa hè"}

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Lỗi] Không thể mở webcam!")
        return

    # Resize để chạy model nhanh hơn
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[Test Webcam] Bắt đầu nhận diện. Nhấn 'q' để thoát.")
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Tính FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        _text_draw_list = []

        # 1. Chạy YOLOv8 base
        if model:
            results = model(frame, classes=TARGET_CLASSES, conf=0.5, verbose=False, imgsz=320, device=CUDA_DEVICE)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = CLASS_LABELS_VI.get(cls, names[cls] if cls in names else str(cls))
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    _text_draw_list.append((f"{label} {int(conf*100)}%", (x1, max(y1-25, 0)), "#3b82f6"))

        # 2. Chạy model nhận diện tiền
        if money_model:
            m_results = money_model(frame, conf=0.6, verbose=False, imgsz=320, device=CUDA_DEVICE)
            for r in m_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"Tiền {money_names[cls]} VNĐ" if cls in money_names else "Tiền"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    _text_draw_list.append((f"{label} {int(conf*100)}%", (x1, max(y1-25, 0)), "#22c55e"))

        # 3. Chạy model vật cản
        if best_model:
            b_results = best_model(frame, conf=0.4, verbose=False, imgsz=320, device=CUDA_DEVICE)
            for r in b_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls == 2: continue # Bỏ qua bậc vỉa từ best.onnx
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    label = BEST_LABELS_VI.get(cls, f"Vật cản {cls}")
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    _text_draw_list.append((f"{label} {int(conf*100)}%", (x1, max(y1-25, 0)), "#ef4444"))

        # 4. Chạy model bậc vỉa
        if curb_model:
            c_results = curb_model(frame, conf=0.4, verbose=False, imgsz=320, device=CUDA_DEVICE)
            for r in c_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    label = "Bậc vỉa hè"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    _text_draw_list.append((f"{label} {int(conf*100)}%", (x1, max(y1-25, 0)), "#f59e0b"))

        # Vẽ text tiếng Việt
        if _text_draw_list:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            for text, pos, c_hex in _text_draw_list:
                c_rgb = hex_to_rgb(c_hex)
                x, y = pos
                # Vẽ viền
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        draw.text((x+dx, y+dy), text, font=_pil_font, fill=(0,0,0))
                draw.text((x, y), text, font=_pil_font, fill=c_rgb)
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Hiển thị FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Test Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
