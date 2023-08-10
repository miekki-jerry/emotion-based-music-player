# Emotion-based Music Player
## Description
The Emotion-based Music Player is an innovative and personalized music experience that aligns music with the listener's current emotional state. By utilizing real-time facial recognition and emotion analysis, the program can detect emotions such as happiness, sadness, excitement, and relaxation, and then play music that resonates with those feelings.

## Why Is It Cool?
**Personalized Experience:** Tailors the music selection to the listener's current mood, enhancing the connection between music and emotions.

**Real-Time Emotion Analysis**: Utilizes the camera to analyze facial expressions in real-time, creating a dynamic and engaging user experience.

**Easy Integration with Spotify:** Allows users to play music directly from Spotify, providing access to a vast collection of tracks and playlists.

**Expandable for More Emotions:** Can be easily extended to recognize and respond to a wider range of emotions.

### How to Install
1. Clone the Repository: Clone the code repository to your local machine using Git.

2. Install required Dependencies using pip.
3. Run the main script to start the emotion recognition and music playback:
   
```
python app.py
```

## Technology Stack
**Facial Recognition:** OpenCV for face detection and capturing real-time video feed.
**Emotion Analysis:** TensorFlow with a pre-trained model (e.g., AffectNet) for emotion classification.
**Music Integration:** Uses the Spotipy library to open Spotify URLs for music playback.
**Web Integration:** Uses the webbrowser module to open Spotify links in the default web browser.

## Conclusion
The Emotion-based Music Player is <strike> the most stupid program I've done so far </strike> an engaging and unique way to connect music with emotions. It offers a seamless and personalized listening experience that can be further expanded and customized. Whether you're looking to lift your spirits with an upbeat track or find solace in soothing melodies, this application has you covered.

Feel free to contribute, explore, and enjoy the power of music aligned with emotions!

**DISCLAIMER:** To train this model, I used the 'train' and 'test' folders and the 'training_model.py' file. You don't have to do this because the model is already trained (emotion_detector.h5).

Also you can use Spotify API instead of basic webbrowser package.
