from ultralytics import YOLO
import cv2
import pytesseract
import re
from difflib import SequenceMatcher

pytesseract.pytesseract.tesseract_cmd = (r'C:\Program Files\Tesseract-OCR\tesseract.exe')

class VehiclesDetection:

    def __init__(self, plate_model_path="models/best.pt", confidence=0.25, preprocessor=None):
        self.vehicle_model = YOLO("yolov8n.pt") # MODELO VEÍCULOS (COCO)
        self.plate_model = YOLO(plate_model_path) # MODELO MATRÍCULAS
        self.confidence = confidence
        self.preprocessor = preprocessor
        self.detected_plates = set() # MATRÍCULAS JÁ DETECTADAS
        self.tracked_plates = {} # TRACK_ID -> MATRÍCULA
        self.plate_votes = {} 

        # CLASSES COCO
        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"}

    def detect_cars(self, frame):
        detections = []
        vehicle_results = self.vehicle_model.track(
            frame,
            conf=self.confidence,
            persist=True,
            tracker="bytetrack.yaml")

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
                track_id = (int(box.id[0]) if box.id is not None else -1)

                # =========================
                # CROP DO VEÍCULO
                # =========================

                vehicle_crop = frame[y1:y2, x1:x2]

                # =========================
                # DETEÇÃO MATRÍCULA
                # =========================

                plate_results = self.plate_model(
                    vehicle_crop,
                    conf=0.25)

                plate_text = "Reading..."
                plate_bbox_global = None

                for plate_result in plate_results:

                    if plate_result.boxes is None:
                        continue

                    for plate_box in plate_result.boxes:

                        px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                        global_px1 = x1 + px1
                        global_py1 = y1 + py1
                        global_px2 = x1 + px2
                        global_py2 = y1 + py2

                        plate_bbox_global = (
                            global_px1,
                            global_py1,
                            global_px2,
                            global_py2)

                        # Crop matrícula
                        plate_crop = frame[
                            global_py1:global_py2,
                            global_px1:global_px2]

                        # OCR com pré-processamento se disponível
                        text = self.read_plate(plate_crop)

                        # Limpeza
                        text = self.clean_plate(text)

                        # Validar OCR
                        if (len(text) >= 6 and
                            any(c.isdigit() for c in text)):

                            # Atualiza tracking
                            if (len(text) >= 6 and any(c.isdigit() for c in text)):

                                # Adiciona voto para este texto
                                if track_id not in self.plate_votes:
                                    self.plate_votes[track_id] = {}
                                self.plate_votes[track_id][text] = self.plate_votes[track_id].get(text, 0) + 1

                                # Escolhe o texto mais votado
                                best_text = max(self.plate_votes[track_id], key=self.plate_votes[track_id].get)

                                # Atualiza se o mais votado mudou
                                previous = self.tracked_plates.get(track_id)
                                if best_text != previous:
                                    if previous:
                                        self.detected_plates.discard(previous)
                                    if not self.is_similar_plate(best_text):
                                        self.tracked_plates[track_id] = best_text
                                        self.detected_plates.add(best_text)

                                plate_text = self.tracked_plates.get(track_id, text)

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

            # =========================
            # BOX VEÍCULO
            # =========================

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            label = f'{vehicle.upper()}'
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + 260, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

            # =========================
            # BOX MATRÍCULA
            # =========================

            if det["plate_bbox"] is not None:

                px1, py1, px2, py2 = det["plate_bbox"]
                cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 1)
                cv2.rectangle(frame, (px1, py1 - 30), (px1 + 220, py1), (255, 0, 0), -1)
                cv2.putText(frame, plate, (px1 + 5, py1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def read_plate(self, image):
        # Se o preprocessor estiver disponível, usa o pipeline completo
        # if self.preprocessor is not None:
            # image = self.preprocessor.process_plate_crop(image)
        # else:
            # Fallback: processamento simples
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.GaussianBlur(gray, (3, 3), 0)

        # text = pytesseract.image_to_string(image, config='--psm 7')
        text = pytesseract.image_to_string(image, config='--psm 8')
        return text.strip()

    def clean_plate(self, text):
        text = text.upper()
        text = re.sub(r'[^A-Z0-9]', '', text)
        return text

    def is_similar_plate(self, new_plate, threshold=0.8):
        for saved_plate in self.detected_plates:
            similarity = SequenceMatcher(None, new_plate, saved_plate).ratio()
            if similarity >= threshold:
                return True
        return False