import cv2
import gradio as gr
from detection import VehiclesDetection
from preprocessor import FramePreprocessor
import os
import time

preprocessor = FramePreprocessor()
detector = VehiclesDetection(preprocessor=preprocessor)

os.makedirs("outputs", exist_ok=True)

def process_video(video_path):
    start = time.time() 
    if video_path is None:
        return None, "Please upload a video."

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, "Could not open video."

    fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, test_frame = cap.read()
    if not ret:
        return None, "Video is empty."
    processed_test = preprocessor.process(test_frame)
    out_h, out_w = processed_test.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_path = "outputs/annotated_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"avc1") # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # frame = preprocessor.process(frame)
        detections = detector.detect_cars(frame)
        annotated_frame = detector.draw_detections(frame, detections)
        out.write(annotated_frame)

    cap.release()
    out.release()

    plate_list = sorted(detector.detected_plates)
    plate_summary = ", ".join(plate_list) if plate_list else "Nenhuma matrícula detectada"
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    summary = f"Processed {frame_count} frames.\nDetected plates: {plate_summary}"

    return output_path, summary


app = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload video"),
    outputs=[gr.Video(label="Video with Vehicle and plates Detection"),
            gr.Textbox(label="Detection Summary")],
    title="CVGAI I01 - License Plate Recognition",
    description="Upload a video to detect license plates and draw blue bounding boxes.")


if __name__ == "__main__":
    app.launch()