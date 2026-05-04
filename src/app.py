import cv2
import gradio as gr
import pytesseract
from ultralytics import YOLO
from detection import VehiclesDetection
import os

detector = VehiclesDetection(
    model_path="models/yolov8n.pt",
    confidence=0.25
)

os.makedirs("outputs", exist_ok=True)


def process_video(video_path):
    if video_path is None:
        return None, "Please upload a video."

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, "Could not open video."

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "outputs/annotated_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    detection_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        detections = detector.detect_cars(frame)
        detection_count += len(detections)

        annotated_frame = detector.draw_detections(frame, detections)

        out.write(annotated_frame)

    cap.release()
    out.release()

    summary = f"Processed {frame_count} frames.\nDetected {detection_count} vehicles."

    return output_path, summary


app = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload video"),
    outputs=[
        gr.Video(label="Video with Vehicle Detection"),
        gr.Textbox(label="Detection Summary")
    ],
    title="CVGAI I01 - License Plate Recognition",
    description="Upload a video to detect vehicles and draw green bounding boxes."
)


if __name__ == "__main__":
    app.launch()