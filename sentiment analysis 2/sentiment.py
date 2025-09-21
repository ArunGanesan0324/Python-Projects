import time
import string
import pandas as pd
import joblib  # Used for saving the model

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# We'll use the NLTK library for advanced text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- One-time setup for NLTK ---
# You only need to run these download commands once per computer.
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK resources (stopwords and wordnet)...")
    nltk.download('stopwords')
    nltk.download('wordnet')
# ---------------------------------

# 1. Load the dataset with custom logic to handle commas within sentences
dataset_filename = 'sentiment_dataset.csv'
print(f"Loading and parsing dataset from '{dataset_filename}'...")

sentences = []
labels_list = []

try:
    with open(dataset_filename, 'r', encoding='utf-8') as f:
        # Read and discard the header line
        next(f)
        # Process each subsequent line in the file
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Find the position of the last comma in the line
            last_comma_index = line.rfind(',')

            # Check if a comma was found
            if last_comma_index != -1:
                # The sentence is everything before the last comma
                sentence = line[:last_comma_index].strip()
                # The label is everything after the last comma
                label_str = line[last_comma_index + 1:].strip()

                # Ensure the label is a valid number (0 or 1) before adding
                if label_str in ('0', '1'):
                    sentences.append(sentence)
                    labels_list.append(int(label_str))

except FileNotFoundError:
    print(f"Error: The dataset file '{dataset_filename}' was not found.")
    print("Please make sure the dataset file is in the same directory as this script.")
    exit()

# Now assign the parsed data to the variables used by the script
texts = sentences
labels = labels_list


# 2. Advanced Text Preprocessing Function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Ensure text is a string
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply the preprocessing
print("Preprocessing text data...")
processed_texts = [preprocess_text(t) for t in texts]

# 3. Split data into training and testing sets
# Using stratify ensures that the proportion of labels is the same in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 4. Create the machine learning pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(solver='liblinear', random_state=42)),
])

# 5. Define the Hyperparameter Grid for GridSearchCV
# These are the settings that GridSearchCV will test to find the best combination.
parameters = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # Unigrams or bigrams
    'tfidf__max_df': [0.75, 1.0],           # Ignore words that are too frequent
    'clf__C': [0.1, 1, 10],                 # Regularization strength
}

# Create the GridSearchCV object to find the best model
# cv=5 means 5-fold cross-validation for robust performance checking
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)

print("\n🚀 Starting Hyperparameter Tuning...")
start_time = time.time()

# 6. Train the model
grid_search.fit(X_train, y_train)

end_time = time.time()
print(f"\n✅ Tuning finished in {end_time - start_time:.2f} seconds.")

# 7. Print the best results found by the search
print("\nBest parameters set found:")
print(grid_search.best_params_)

# 8. Evaluate the BEST model on the unseen test data
print("\n🔍 Evaluating the best model on the test set...")
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# 9. Save the final, trained model to a file
model_filename = 'sentiment_model.joblib'
print(f"\n💾 Saving the best model to `{model_filename}`...")
# We save the .best_estimator_, which is the actual trained Pipeline
joblib.dump(grid_search.best_estimator_, model_filename)
print("✨ Model saved successfully!")

