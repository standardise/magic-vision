from ultralytics import YOLO
import numpy as np

class YOLOv8Agnostic:
    def __init__(self, model_path: str = "weights/yolov8_agnostic.pt", conf_threshold: float = 0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        รับภาพ 1 เฟรม (BGR จาก OpenCV)
        Return: Bounding boxes ในรูปแบบ numpy array รูปร่าง (N, 5) -> [x1, y1, x2, y2, confidence]
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]

        boxes_list = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            boxes_list.append([x1, y1, x2, y2, conf])
        
        if len(boxes_list) == 0:
            return np.empty((0, 5))  # คืนค่า array ว่างถ้าไม่มีการตรวจจับ
        
        return np.array(boxes_list)