import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from numpy.linalg import norm

# สมมติว่า ByteTrack ถูกนำมาเขียนไว้ใน module นี้นะครับ
# (คุณสามารถใช้ library อย่าง 'yolox.tracker.byte_tracker' หรือตัวอื่นๆ ที่หาได้ใน Github)
from app.ml.tracker import ByteTrack 

class VisionCountingPipeline:
    def __init__(self, yolo_model, dino_model, target_classes: List[str]):
        self.yolo = yolo_model
        self.dino = dino_model
        
        # กำหนด Tracker (สมมติว่าปรับค่า frame_rate และ track_buffer ตามความเหมาะสม)
        self.tracker = ByteTrack(track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30)
        
        # State ของ Session นี้
        self.target_classes = target_classes
        self.track_memory = {}   # {track_id: {"votes": ["obj_1", "obj_1"], "final_class": "obj_1"}}
        self.counted_ids = set() # เก็บ ID ที่นับผ่านเส้นไปแล้ว
        
        # เส้นสมมติสำหรับการนับ (Virtual Line) กำหนดเป็นพิกัด (x1, y1), (x2, y2)
        # ตัวอย่าง: ลากเส้นแนวนอนกลางหน้าจอ
        self.counting_line = ((100, 500), (1000, 500)) 
        
        # สีสำหรับวาด (เพื่อความสวยงามตอน Debug)
        self.colors = np.random.uniform(0, 255, size=(1000, 3))

    def process_frame(self, frame: np.ndarray, db_prototypes: Dict[str, np.ndarray]) -> Tuple[np.ndarray, int]:
        """รัน 1 เฟรมภาพ ผ่าน Pipeline ทั้งหมด"""
        
        # เก็บจำนวนนับในเฟรมนี้
        current_counts = 0
        
        # 1. Detect (หาตำแหน่งวัตถุ)
        boxes = self.yolo.detect(frame)
        
        # 2. Track (ผูก ID ให้วัตถุ)
        # Tracker จะคืนค่า list ของ object ที่มี properties: id, bbox, centroid, prev_centroid
        tracked_objects = self.tracker.update(boxes)
        
        # วาดเส้นทับลงไปบนภาพเพื่อให้เห็นว่าเส้นอยู่ตรงไหน
        cv2.line(frame, self.counting_line[0], self.counting_line[1], (0, 0, 255), 3)
        
        for obj in tracked_objects:
            track_id = obj.track_id # เปลี่ยนชื่อเรียกตาม implementation ของ ByteTrack
            
            # สมมติ bbox มาในรูปแบบ [x1, y1, x2, y2]
            box = obj.tlbr 
            
            # คำนวณจุดกึ่งกลาง (Centroid) ล่าสุด
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            centroid = (x_center, y_center)
            
            # ดึงจุดกึ่งกลางของเฟรมที่แล้วมา (ถ้ามี)
            prev_centroid = getattr(obj, 'prev_centroid', None)
            obj.prev_centroid = centroid # อัปเดตเก็บไว้ใช้รอบหน้า
            
            # --- 3. Lazy Feature Matching ---
            if track_id not in self.track_memory:
                self.track_memory[track_id] = {"votes": [], "final_class": None}
                
            memory = self.track_memory[track_id]
            
            # ถ้ายืนยันคลาสไม่ได้ (โหวตยังไม่ครบ 3 เฟรม) ให้รัน DINO
            if memory["final_class"] is None and len(memory["votes"]) < 3:
                cropped_img = self._crop_image(frame, box)
                
                # เช็คว่าภาพ crop ไม่ใช่ภาพว่างเปล่า
                if cropped_img.size > 0:
                    vector = self.dino.encode(cropped_img)
                    
                    # เทียบ Vector ล่าสุดกับ Database Prototypes
                    best_match = self._find_best_match(vector, db_prototypes)
                    
                    if best_match:
                        memory["votes"].append(best_match)
                    
                # ถ้าโหวตครบ 3 ครั้ง หา Majority Vote (หาตัวที่ซ้ำเยอะสุด)
                if len(memory["votes"]) == 3:
                    memory["final_class"] = max(set(memory["votes"]), key=memory["votes"].count)
            
            # --- 4. Line Crossing Logic (ลอจิกการนับ) ---
            final_class = memory.get("final_class")
            
            # ถ้ารู้แล้วว่าคืออะไร และเป็นคลาสที่เราสนใจนับ
            if final_class in self.target_classes:
                # ตรวจสอบการตัดเส้น (ต้องมีจุดก่อนหน้าถึงจะลากเส้นเทียบได้)
                if prev_centroid is not None:
                    if self._is_crossing_line(prev_centroid, centroid, self.counting_line):
                        if track_id not in self.counted_ids:
                            self.counted_ids.add(track_id)
                            current_counts += 1 # นับเพิ่ม!
                            
            # --- 5. Visualization (วาดกล่องและข้อความทับลงไปบนภาพ) ---
            color = self.colors[track_id % 1000]
            label = final_class if final_class else "Identifying..."
            
            # วาดกรอบ
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            
            # เขียนข้อความบอก ID และ Class
            text = f"ID:{track_id} {label}"
            cv2.putText(frame, text, (int(box[0]), int(box[1] - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # เปลี่ยนสีข้อความถ้านับไปแล้ว
            if track_id in self.counted_ids:
                cv2.putText(frame, "COUNTED", (int(box[0]), int(box[3] + 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
        return frame, current_counts

    # ==========================================
    # Helper Functions
    # ==========================================

    def _crop_image(self, frame: np.ndarray, box: List[float]) -> np.ndarray:
        """ตัดรูปภาพจาก Bounding Box โดยป้องกันการทะลุขอบหน้าจอ (Out of bounds)"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        return frame[y1:y2, x1:x2]

    def _find_best_match(self, vector: np.ndarray, db_prototypes: Dict[str, np.ndarray]) -> Optional[str]:
        """หา Class ที่มีความเหมือนของ Vector (Cosine Similarity) สูงที่สุด"""
        best_class = None
        best_sim = -1.0 # Cosine Sim มีค่าตั้งแต่ -1 (ตรงข้าม) ถึง 1 (เหมือนกันเป๊ะ)
        
        for class_name, proto_vec in db_prototypes.items():
            # สูตร Cosine Similarity = (A dot B) / (||A|| * ||B||)
            sim = np.dot(vector, proto_vec) / (norm(vector) * norm(proto_vec) + 1e-10)
            
            if sim > best_sim:
                best_sim = sim
                best_class = class_name
                
        # หากต้องการความแม่นยำ สามารถตั้ง Threshold ได้ เช่น if best_sim > 0.8: return best_class
        # ในตอนนี้เรา return ตัวที่ใกล้เคียงที่สุดเสมอ
        return best_class

    def _is_crossing_line(self, p1, p2, line):
        """
        p1: จุดในเฟรมก่อน (x, y)
        p2: จุดในเฟรมปัจจุบัน (x, y)
        line: ((x1, y1), (x2, y2))
        """
        (lx1, ly1), (lx2, ly2) = line
    
        # กรณีเส้นแนวนอน (Horizontal Line) - นิยมที่สุดในสายพาน
        if ly1 == ly2:
            # เช็คว่าจุดหนึ่งอยู่เหนือเส้น อีกจุดอยู่ใต้เส้น (หรือทับเส้น)
            crossed = (p1[1] <= ly1 and p2[1] > ly1) or (p1[1] >= ly1 and p2[1] < ly1)
            # และต้องอยู่ในช่วงความกว้างของเส้น x1 ถึง x2 ด้วย
            in_width = min(lx1, lx2) <= p2[0] <= max(lx1, lx2)
            return crossed and in_width

        # กรณีเส้นแนวตั้ง (Vertical Line)
        elif lx1 == lx2:
            crossed = (p1[0] <= lx1 and p2[0] > lx1) or (p1[0] >= lx1 and p2[0] < lx1)
            in_height = min(ly1, ly2) <= p2[1] <= max(ly1, ly2)
            return crossed and in_height
    
        # ถ้าเป็นเส้นเฉียง ให้ใช้ Logic เดิม (CCW) ได้ครับ
        return self._is_line_intersect(p1, p2, line[0], line[1])

    def _ccw(self, A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
        """เช็คทิศทางทวนเข็มนาฬิกา (Counter-Clockwise) สำหรับสมการตัดเส้น"""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])