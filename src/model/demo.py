import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

target_classes = [0, 56, 60, 39, 67]

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.1.235:5000/video")

names = model.names

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1080, 720), interpolation=cv2.INTER_AREA)

    results = model(frame, classes=target_classes, conf=0.3)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = f"{names[cls]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Detection (5 objects)", frame)

    # nhấn q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()