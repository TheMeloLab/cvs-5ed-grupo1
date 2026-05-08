from ultralytics import YOLO
import cv2


class VehiclesDetection: 
    def __init__(self, model_path="yolov8n.pt", confidence=0.4):
        self.model = YOLO(model_path)
        self.confidence = confidence

        # COCO vehicle classes
        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

    def detect_cars(self, frame):
        results = self.model(frame, conf=self.confidence)

        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])

                if class_id in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])

                    detections.append({
                        "class": self.vehicle_classes[class_id],
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2)
                    })

        return detections

    def draw_detections(self, frame, detections):
        frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f'{det["class"]}: {det["confidence"]:.2f}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return frame