import numpy as np
import cv2

from app.ml.detector import YOLOv8Agnostic
from app.ml.encoder import DINOv2Encoder
from app.ml.pipeline import VisionCountingPipeline

def main():
    print("Testing VisionCountingPipeline with dummy data...")

    yolo = YOLOv8Agnostic(model_path="./models/yolov8n.pt", conf_threshold=0.5)
    dino = DINOv2Encoder(model_path="./models/dinov2_vits14.pth")

    dummy_vector = np.random.rand(384).astype(np.float32)
    dummy_vector = dummy_vector / np.linalg.norm(dummy_vector)

    db_prototypes = {
        "screw": dummy_vector,
        "unknown_part": dummy_vector
    }

    pipline = VisionCountingPipeline(
        yolo_model=yolo,
        dino_model=dino,
        target_classes=["screw"]
    )

    pipline.counting_line = ((50, 240), (600, 240))
    cap = cv2.VideoCapture(0)
    total_counted = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from video source")
            return

        frame = cv2.resize(frame, (640, 480))

        processed_frame, count_in_frame = pipline.process_frame(frame=frame, db_prototypes=db_prototypes)

        total_counted += count_in_frame

        cv2.rectangle(processed_frame, (10, 10), (350, 70), (0, 0, 0), -1)
        cv2.putText(processed_frame, f"TOTAL COUNT: {total_counted}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        cv2.imshow("Few-Shot Counting System (Test Mode)", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
        