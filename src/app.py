import cv2
import gradio as gr
from detection import VehiclesDetection
import os
from preprocessor import FramePreprocessor
from extraction import PlateExtractor

detector = VehiclesDetection(
    model_path="models/best.pt",
    confidence=0.25
)

preprocessor = FramePreprocessor()
extractor = PlateExtractor()

os.makedirs("outputs", exist_ok=True)


def process_video(video_path):
    if video_path is None:
        return None, "Please upload a video."

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, "Could not open video."

    fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, test_frame = cap.read()
    processed_test = preprocessor.process(test_frame)
    processed_height, processed_width = processed_test.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_path = "outputs/annotated_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (processed_width, processed_height))

    frame_count = 0
    detection_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        frame = preprocessor.process(frame) 

        detections = detector.detect_cars(frame)
        detection_count += len(detections)

        for det in detections:                          
            x1, y1, x2, y2 = det['bbox']
            plate_crop = frame[y1:y2, x1:x2]
            plate_ready = preprocessor.process_plate_crop(plate_crop)
            text = extractor.extract(plate_ready)
            if text:
                det['plate_text'] = text

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