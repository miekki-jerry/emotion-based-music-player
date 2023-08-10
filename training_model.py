import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
NUM_CLASSES = 7  # Number of emotions
WIDTH, HEIGHT = 48, 48  # Dimensions of input images
BATCH_SIZE = 64  # Size of batches for training
EPOCHS = 100  # Number of epochs for training
DATASET_DIR = '/Users/piter/Documents/programowanko/Emotion_recognition'  # Directory of your dataset

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(WIDTH, HEIGHT, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Use ImageDataGenerator to load and preprocess the dataset
datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values

# Load the dataset
train_generator = datagen.flow_from_directory(os.path.join(DATASET_DIR, 'train'), target_size=(WIDTH, HEIGHT), color_mode='grayscale', batch_size=BATCH_SIZE, class_mode='categorical')
test_generator = datagen.flow_from_directory(os.path.join(DATASET_DIR, 'test'), target_size=(WIDTH, HEIGHT), color_mode='grayscale', batch_size=BATCH_SIZE, class_mode='categorical')

# Train the model
model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator)

# Save the model for later use
model.save('emotion_detector.h5')

# Real-time emotion detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert color to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (WIDTH, HEIGHT))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)
        
        # Predict the emotion
        emotion = EMOTIONS[np.argmax(model.predict(face))]
        
        # Display the emotion
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
