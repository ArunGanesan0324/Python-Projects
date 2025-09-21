import speech_recognition as sr
from transformers import pipeline
import emotion_analyzer
import conversational_responder # <-- Using the dynamic (generative) responder

# --- 1. Load the Sarcasm Detection Model (Analyzes TEXT) ---
print("Main: Loading the sarcasm detection model...")
sarcasm_detector = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-irony")
print("Main: Sarcasm model loaded successfully!")

# --- 2. Initialize the Voice Recognizer ---
recognizer = sr.Recognizer()
microphone = sr.Microphone()

with microphone as source:
    print("\nMain: Calibrating microphone for ambient noise...")
    recognizer.adjust_for_ambient_noise(source, duration=1)

RECORDING_DURATION = 7
print(f"\n🎙️  Ready! The program will record for {RECORDING_DURATION} seconds each time.")
print("(Press Ctrl+C in the terminal to stop the program)")

# --- 3. Start the Real-Time Recognition Loop ---
while True:
    try:
        with microphone as source:
            print(f"\nRecording for {RECORDING_DURATION} seconds...")
            audio = recognizer.record(source, duration=RECORDING_DURATION)

        print("Processing audio...")

        # --- Analysis 1: Get Emotion from Tone ---
        raw_audio_data = audio.get_raw_data()
        sample_rate = audio.sample_rate
        emotion_label, emotion_score = emotion_analyzer.analyze_emotion_from_audio(raw_audio_data, sample_rate)
        print(f"--> Emotion from Tone: {emotion_label} (Confidence: {emotion_score:.2f})")

        # --- Analysis 2: Get Sarcasm from Text ---
        is_sarcastic = False
        user_text = "" # Initialize user_text
        try:
            user_text = recognizer.recognize_google(audio)
            print(f"You said: \"{user_text}\"")

            results = sarcasm_detector([user_text])
            result = results[0]
            label = result['label'].upper()
            score = result['score']
            
            if label == 'IRONY' and score > 0.5:
                verdict = "Sarcastic"
                is_sarcastic = True
            else:
                verdict = "Not Sarcastic"
            
            print(f"--> Sarcasm from Text: {verdict} (Confidence: {score:.2f})")

        except sr.UnknownValueError:
            print("--> Sarcasm from Text: Could not understand the audio to transcribe.")

        # --- Step 3: Generate and Speak the DYNAMIC Reply ---
        # THIS IS THE CORRECTED LINE: We now pass all three required arguments.
        if user_text:
            conversational_responder.generate_dynamic_reply(is_sarcastic, emotion_label, user_text)

    except sr.RequestError as e:
        print(f"--> API error: {e}")
    except KeyboardInterrupt:
        print("\n\nProgram stopped by user. Goodbye!")
        break