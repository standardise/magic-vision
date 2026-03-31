import supervision as sv
import numpy as np

class TrackedObject:
    """
    คลาสจำลองเพื่อเก็บข้อมูลของวัตถุที่ถูก Track แล้ว
    เพื่อให้ pipeline.py สามารถเรียกใช้ obj.track_id และ obj.tlbr ได้ทันที
    """
    def __init__(self, track_id: int, tlbr: np.ndarray):
        self.track_id = track_id
        self.tlbr = tlbr # [x1, y1, x2, y2]
        self.prev_centroid = None # จะถูกจัดการโดย pipeline.py

class ByteTrack:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30):
        """
        เริ่มต้น ByteTrack Tracker โดยเรียกใช้จาก supervision library
        """
        print("Initializing ByteTrack (Supervision)...")
        
        # สร้าง Instance ของ ByteTrack จากไลบรารี supervision
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate
        )

    def update(self, boxes: np.ndarray) -> list:
        """
        รับ Bounding Boxes จาก YOLOv8 และคืนค่ากลับเป็น List ของ TrackedObject
        :param boxes: numpy array ขนาด (N, 5) -> [x1, y1, x2, y2, confidence]
        :return: List[TrackedObject]
        """
        # 1. ถ้าเฟรมนี้ YOLO หาอะไรไม่เจอเลย ให้คืนค่าลิสต์ว่างกลับไป
        if len(boxes) == 0:
            return []

        # 2. แยกข้อมูลจาก numpy array ของเรา
        xyxy = boxes[:, :4]           # พิกัดกล่อง
        confidence = boxes[:, 4]      # ค่าความมั่นใจ (Confidence Score)
        
        # สมมติ class_id เป็น 0 ทั้งหมด เพราะ YOLO ของเราทำงานแบบ Class-Agnostic
        class_id = np.zeros(len(boxes), dtype=int) 

        # 3. แปลงข้อมูลให้อยู่ในรูปแบบ Detections object ที่ supervision ต้องการ
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        # 4. รัน ByteTrack Algorithm
        # ขั้นตอนนี้ Tracker จะทำ Kalman Filter & Bipartite Matching เพื่อจำแนก ID
        tracked_detections = self.tracker.update_with_detections(detections)

        # 5. แปลงผลลัพธ์กลับมาเป็นโครงสร้าง TrackedObject ให้ pipeline.py ของเราใช้งาน
        tracked_objects = []
        
        # หากมีวัตถุที่ถูก Track ได้
        if tracked_detections.tracker_id is not None:
            for i in range(len(tracked_detections)):
                t_id = tracked_detections.tracker_id[i]
                box_coords = tracked_detections.xyxy[i]
                
                tracked_objects.append(TrackedObject(track_id=t_id, tlbr=box_coords))

        return tracked_objects