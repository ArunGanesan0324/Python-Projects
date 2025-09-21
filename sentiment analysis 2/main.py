import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Text Preprocessing Function ---
# IMPORTANT: You must use the EXACT same preprocessing function that you used
# during training for the model to work correctly.
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and prepares text data for the model.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)
# -----------------------------------


def predict_sentiment_and_emotion(sentence: str, model):
    """
    Predicts the sentiment and a corresponding emotion for a given sentence.

    Args:
        sentence (str): The input sentence to analyze.
        model: The trained sentiment analysis model.

    Returns:
        dict: A dictionary containing the 'sentiment' and 'emotion'.
    """
    # Preprocess the new sentence from the user
    processed_sentence = preprocess_text(sentence)

    # Use the loaded model to predict the sentiment (returns 0 or 1)
    prediction = model.predict([processed_sentence])[0]
    
    # Map the numeric prediction to human-readable sentiment and emotion
    if prediction == 1:
        sentiment = "Positive"
        emotion = "Happy 😊"
    else:
        sentiment = "Negative"
        emotion = "Sad/Angry 😠"
        
    return {"sentiment": sentiment, "emotion": emotion}


# This block runs when the script is executed directly
if __name__ == "__main__":
    # 1. Load the saved model from the file
    model_filename = 'sentiment_model.joblib'
    try:
        sentiment_model = joblib.load(model_filename)
        print(f"✅ Sentiment analysis model '{model_filename}' loaded successfully!")
    except FileNotFoundError:
        print(f"❌ Error: Model file '{model_filename}' not found.")
        print("Please run the `train_model.py` script first to create and save the model file.")
        exit()

    # 2. Example sentences to test
    test_sentences = [
        "This is a fantastic movie, I loved every minute of it!",
        "What a complete waste of time, the plot was terrible.",
        "The service was bad"
    ]

    print("\n🤖 Analyzing example sentences...")
    for sentence in test_sentences:
        result = predict_sentiment_and_emotion(sentence, sentiment_model)
        print(f"\nSentence: '{sentence}'")
        print(f" -> Predicted Sentiment: {result['sentiment']}")
        print(f" -> Associated Emotion: {result['emotion']}")
