import streamlit as st
import cv2
import tempfile
import os
from object_detect import ObjectDetector

st.title("🎯 YOLO Object Detection App")
st.write("Upload a video to detect and track objects using YOLOv8.")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Temporary save video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Open video
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    detector = ObjectDetector("yolov8n.pt")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, detections = detector.detect_frame(frame)

        # Show video frame
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    os.unlink(tfile.name)
