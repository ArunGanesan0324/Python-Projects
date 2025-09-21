import sys
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = keras.models.load_model("project_idea_nn_model.h5")

# Load dataset
file_path = r"E:\reactnative\IdeaGenerator\Backend\projectIdea.xlsx"
df = pd.read_excel(file_path)

# Encode columns
difficulty_encoder = LabelEncoder()
domain_encoder = LabelEncoder()
idea_encoder = LabelEncoder()

df["Difficulty"] = difficulty_encoder.fit_transform(df["Difficulty"])
df["Domain"] = domain_encoder.fit_transform(df["Domain"])
df["Project Idea"] = idea_encoder.fit_transform(df["Project Idea"])

# Get input parameters
domain = sys.argv[1]
difficulty = sys.argv[2] if len(sys.argv) > 2 else None

# Predict function
def predict_project_idea(domain, difficulty):
    try:
        domain_encoded = domain_encoder.transform([domain])[0]
    except ValueError:
        return f"Invalid domain '{domain}'. Available: {list(domain_encoder.classes_)}"

    if difficulty:
        try:
            difficulty_encoded = difficulty_encoder.transform([difficulty])[0]
            matching_projects = df[(df["Domain"] == domain_encoded) & (df["Difficulty"] == difficulty_encoded)]["Project Idea"].values
        except ValueError:
            return f" Invalid difficulty '{difficulty}'. Available: {list(difficulty_encoder.classes_)}"
    else:
        matching_projects = df[df["Domain"] == domain_encoded]["Project Idea"].values

    if len(matching_projects) > 0:
        project_names = idea_encoder.inverse_transform(matching_projects)
        np.random.shuffle(project_names)
        return "\n".join(project_names[:5])  # Return 5 projects
    return f" No projects found for '{difficulty}' in '{domain}'"

# Get prediction and print (to be read by Node.js)
print(predict_project_idea(domain, difficulty))