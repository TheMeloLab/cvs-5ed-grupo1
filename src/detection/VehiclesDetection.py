from ultralytics import YOLO
import cv2
import pytesseract
import re
from difflib import SequenceMatcher

pytesseract.pytesseract.tesseract_cmd = (
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)


class VehiclesDetection:

    def __init__(self,
                 plate_model_path="models/best.pt",
                 confidence=0.4):

        # MODELO VEÍCULOS (COCO)
        self.vehicle_model = YOLO("yolov8n.pt")

        # MODELO MATRÍCULAS
        self.plate_model = YOLO(plate_model_path)

        self.confidence = confidence

        # MATRÍCULAS JÁ DETECTADAS
        self.detected_plates = set()

        # TRACK_ID -> MATRÍCULA
        self.tracked_plates = {}

        # CLASSES COCO
        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

    def detect_cars(self, frame):

        detections = []

        # =========================
        # DETEÇÃO/TRACKING VEÍCULOS
        # =========================

        vehicle_results = self.vehicle_model.track(
            frame,
            conf=self.confidence,
            persist=True,
            tracker="bytetrack.yaml"
        )

        for result in vehicle_results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])

                if class_id not in self.vehicle_classes:
                    continue

                vehicle_name = self.vehicle_classes[class_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0])

                track_id = (
                    int(box.id[0])
                    if box.id is not None
                    else -1
                )

                # =========================
                # CROP DO VEÍCULO
                # =========================

                vehicle_crop = frame[y1:y2, x1:x2]

                # =========================
                # DETEÇÃO MATRÍCULA
                # =========================

                plate_results = self.plate_model(
                    vehicle_crop,
                    conf=0.3
                )

                plate_text = "Reading..."

                plate_bbox_global = None

                for plate_result in plate_results:

                    if plate_result.boxes is None:
                        continue

                    for plate_box in plate_result.boxes:

                        px1, py1, px2, py2 = map(
                            int,
                            plate_box.xyxy[0]
                        )

                        # Coordenadas globais
                        global_px1 = x1 + px1
                        global_py1 = y1 + py1
                        global_px2 = x1 + px2
                        global_py2 = y1 + py2

                        plate_bbox_global = (
                            global_px1,
                            global_py1,
                            global_px2,
                            global_py2
                        )

                        # Crop matrícula
                        plate_crop = frame[
                            global_py1:global_py2,
                            global_px1:global_px2
                        ]

                        # OCR
                        text = self.read_plate(plate_crop)

                        # Limpeza
                        text = self.clean_plate(text)

                        # Validar OCR
                        if (
                            len(text) >= 6 and
                            any(c.isdigit() for c in text)
                        ):

                            # Atualiza tracking
                            if track_id not in self.tracked_plates:

                                if not self.is_similar_plate(text):

                                    self.tracked_plates[track_id] = text
                                    self.detected_plates.add(text)

                            # reutiliza matrícula conhecida
                            plate_text = self.tracked_plates.get(
                                track_id,
                                text
                            )

                        break

                detections.append({
                    "id": track_id,
                    "vehicle": vehicle_name,
                    "confidence": confidence,
                    "vehicle_bbox": (x1, y1, x2, y2),
                    "plate_bbox": plate_bbox_global,
                    "plate": plate_text
                })

        return detections

    def draw_detections(self, frame, detections):

        frame = frame.copy()

        for det in detections:

            x1, y1, x2, y2 = det["vehicle_bbox"]

            plate = det["plate"]

            vehicle = det["vehicle"]

            track_id = det["id"]

            # =========================
            # BOX VEÍCULO
            # =========================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                1
            )

            label = f'{vehicle.upper()}'

            cv2.rectangle(
                frame,
                (x1, y1 - 30),
                (x1 + 260, y1),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                1
            )

            # =========================
            # BOX MATRÍCULA
            # =========================

            if det["plate_bbox"] is not None:

                px1, py1, px2, py2 = det["plate_bbox"]

                cv2.rectangle(
                    frame,
                    (px1, py1),
                    (px2, py2),
                    (255, 0, 0),
                    1
                )

                cv2.rectangle(
                    frame,
                    (px1, py1 - 30),
                    (px1 + 220, py1),
                    (255, 0, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    plate,
                    (px1 + 5, py1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

        return frame

    def read_plate(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        text = pytesseract.image_to_string(
            gray,
            config='--psm 7'
        )

        return text.strip()

    def clean_plate(self, text):

        text = text.upper()

        text = re.sub(r'[^A-Z0-9]', '', text)

        return text

    def is_similar_plate(self,
                         new_plate,
                         threshold=0.8):

        for saved_plate in self.detected_plates:

            similarity = SequenceMatcher(
                None,
                new_plate,
                saved_plate
            ).ratio()

            if similarity >= threshold:
                return True

        return False