from ultralytics import YOLO
import cv2

# Load YOLO model once
yolo_model = YOLO("models/yolov8n.pt")

def detect_cats(image):
    """
    Detect cats in the image using YOLOv8.
    Returns list of bounding boxes [(x1,y1,x2,y2)]
    """
    results = yolo_model(image)
    boxes = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])

            # COCO class 15 = cat
            if cls_id == 15:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))

    return boxes