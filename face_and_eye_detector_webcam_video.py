'''Drowsy driver detection: Beeps when closed eyes are detected'''

# Import necessary libraries
import cv2 as cv
import numpy as np
import winsound
import time
import threading

# Load Haarcascade classifiers
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")

# Check if cascades are loaded
if face_cascade.empty() or eye_cascade.empty():
    print("Error loading cascade files.")
    exit()

# Alarm control variables
last_alarm_time = 0  # Last beep time

# Function to play sound in a separate thread
def play_alarm():
    global last_alarm_time
    if time.time() - last_alarm_time > 5:  # Beep only once every 5 sec
        last_alarm_time = time.time()
        threading.Thread(target=winsound.Beep, args=(1000, 500), daemon=True).start()  # Non-blocking sound

# Capture video from webcam
video_capture = cv.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv.flip(frame, 1)  # Mirror effect
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # **Improve face & eye detection by enhancing contrast**
    gray = cv.equalizeHist(gray)  

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80))

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes inside the detected face
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

        if len(eyes) == 0:  # No eyes detected = Possibly closed eyes
            play_alarm()

        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Show the video feed
    cv.imshow('Drowsiness Detector', frame)

    # Exit when 'Esc' key is pressed
    if cv.waitKey(1) == 27:  
        break

# Release video capture and close all windows
video_capture.release()
cv.destroyAllWindows()
