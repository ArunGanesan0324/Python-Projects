import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
file_path = r"Backend\projectIdea.xlsx"  # Ensure this file is in the same directory
df = pd.read_excel(file_path)

# Encode difficulty and domain as numbers
difficulty_encoder = LabelEncoder()
domain_encoder = LabelEncoder()

df["Difficulty"] = difficulty_encoder.fit_transform(df["Difficulty"])
df["Domain"] = domain_encoder.fit_transform(df["Domain"])

# Prepare input (X) and output (Y)
X = df[["Difficulty", "Domain"]].values
y = df["Project Idea"].values

# Convert text output (Project Ideas) into numerical format using encoding
idea_encoder = LabelEncoder()
y_encoded = idea_encoder.fit_transform(y)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create Neural Network Model
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(2,)),  # Input: Difficulty & Domain
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(len(idea_encoder.classes_), activation='softmax')  # Output: Project Idea Categories
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("project_idea_nn_model.h5")

print("Model training complete and saved as 'project_idea_nn_model.h5'!")

# Function to predict project ideas
def predict_project_idea(difficulty, domain):
    difficulty_encoded = difficulty_encoder.transform([difficulty])[0]
    domain_encoded = domain_encoder.transform([domain])[0]
    prediction = model.predict(np.array([[difficulty_encoded, domain_encoded]]))
    predicted_index = np.argmax(prediction)
    return idea_encoder.inverse_transform([predicted_index])[0]

# Example usage
difficulty_input = "Intermediate"  # Example input
domain_input = "AI/ML"  # Example input
predicted_idea = predict_project_idea(difficulty_input, domain_input)
print(f"Suggested Project Idea: {predicted_idea}")