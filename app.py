import os
import cv2
import time
import numpy as np
from collections import Counter
from tensorflow.keras.models import load_model
import webbrowser

# Constants
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
WIDTH, HEIGHT = 48, 48
MODEL_PATH = os.getenv('MODEL_PATH', 'emotion_detector.h5')

def load_emotion_model():
    return load_model(MODEL_PATH)

def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

def predict_emotion(face, model):
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (WIDTH, HEIGHT))
    face_normalized = face_resized / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=0)
    face_expanded = np.expand_dims(face_expanded, axis=-1)
    return EMOTIONS[np.argmax(model.predict(face_expanded))]

def play_music_on_spotify(url):
    webbrowser.open(url)

def main():
    model = load_emotion_model()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    emotions_detected = []
    start_time = time.time()
    while time.time() - start_time < 5:  # Run for 5 seconds
        ret, frame = cap.read()
        faces = detect_faces(frame, face_cascade)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            emotion = predict_emotion(face, model)
            emotions_detected.append(emotion)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    most_common_emotion = Counter(emotions_detected).most_common(1)[0][0]
    
    if most_common_emotion == "sad":
        sad_music_url = "https://open.spotify.com/track/6ITfsvK4yudOL4pK4QIFJP?si=553346ccfd484a56"
        play_music_on_spotify(sad_music_url)

    if most_common_emotion == "happy":
        happy_music_url = "https://open.spotify.com/track/1nInOsHbtotAmEOQhtvnzP?si=49c0e004880b4545"
        play_music_on_spotify(happy_music_url)

if __name__ == "__main__":
    main()
