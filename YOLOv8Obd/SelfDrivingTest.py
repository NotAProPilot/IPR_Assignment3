import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import threading
import time

# Function to detect lanes in a frame
def detect_lanes(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges using Canny
    edges = cv2.Canny(blur, 50, 150)
    # Define a region of interest (ROI) to focus on lanes
    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (width * 0.1, height),
        (width * 0.4, height * 0.6),
        (width * 0.6, height * 0.6),
        (width * 0.9, height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    # Find lines using Hough transform
    lines = cv2.HoughLinesP(cropped_edges, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=25)
    # Draw lines on the frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return frame

# Function to process a video file and detect objects and lanes
def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_lanes(frame)
        cv2.imshow('Video Processing', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to start video processing in a separate thread
def start_video_processing(file_path):
    processing_thread = threading.Thread(target=process_video, args=(file_path,))
    processing_thread.start()

# Function to open file dialog and select a video file
def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if file_path:
        start_video_processing(file_path)

# Function to initiate real-time object detection (using webcam)
def real_time_detection():
    cap = cv2.VideoCapture(0)  # Open the default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_lanes(frame)
        cv2.imshow('Real-time Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Create a simple GUI with Tkinter
root = tk.Tk()
root.title("Video Processing Application")
root.geometry("400x200")

# Button to upload a video file and process it
upload_button = tk.Button(root, text="Upload Video", command=upload_video)
upload_button.pack(pady=20)

# Button to start real-time detection (using webcam)
real_time_button = tk.Button(root, text="Real-time Detection", command=real_time_detection)
real_time_button.pack(pady=20)

root.mainloop()