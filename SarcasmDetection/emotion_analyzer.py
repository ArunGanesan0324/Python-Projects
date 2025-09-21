from transformers import pipeline
import numpy as np

# --- 1. Load the UPGRADED Emotion Detection Model (Analyzes AUDIO TONE) ---
# We are changing the model name here to get 8 emotions instead of 4.
print("Emotion Analyzer: Loading 8-emotion recognition model...")
try:
    # THIS IS THE ONLY LINE YOU NEED TO CHANGE
    emotion_recognizer = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    print("Emotion Analyzer: 8-emotion model loaded successfully.")
except Exception as e:
    print(f"Emotion Analyzer: Failed to load model. Error: {e}")
    emotion_recognizer = None

def analyze_emotion_from_audio(raw_audio_data, sample_rate):
    """
    Analyzes raw audio data to detect the primary emotion from the speaker's tone.

    Args:
        raw_audio_data (bytes): The raw audio bytes from the microphone.
        sample_rate (int): The sample rate of the audio (e.g., 16000).

    Returns:
        tuple: A tuple containing the emotion label (str) and score (float),
               or (None, None) if the model isn't loaded.
    """
    if not emotion_recognizer:
        return "Model not loaded", 0.0

    # The model expects the audio data as a NumPy array of floats.
    # We convert the raw bytes to a NumPy array of 16-bit integers,
    # then normalize it to a float array between -1 and 1.
    audio_array = np.frombuffer(raw_audio_data, dtype=np.int16)
    audio_array_float = audio_array.astype(np.float32) / 32768.0

    # Get a list of all emotion predictions from the model
    results = emotion_recognizer(audio_array_float, sampling_rate=sample_rate)
    
    # Find the emotion with the highest confidence score
    top_emotion = max(results, key=lambda x: x['score'])
    
    label = top_emotion['label'].upper()
    score = top_emotion['score']
    
    return label, score
