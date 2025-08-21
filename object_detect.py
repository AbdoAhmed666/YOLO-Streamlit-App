import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        results = self.model(frame, stream=True)
        detections = []

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)  # x1, y1, x2, y2
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls in zip(boxes, confs, clss):
                x1, y1, x2, y2 = box
                detections.append({
                    "box": (x1, y1, x2, y2),
                    "conf": float(conf),
                    "cls": int(cls),
                    "label": self.model.names[cls]
                })

                # Draw on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{self.model.names[cls]} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        return frame, detections
