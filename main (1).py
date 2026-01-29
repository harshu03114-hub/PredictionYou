from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib

app = FastAPI(title="Predicted You vs Real You API")

# =====================================================
# Load models (load the embedder directly without saving it)
# =====================================================
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Load Sentence-BERT model

models = {
    "Extraversion": joblib.load("E_label_classifier.pkl"),
    "Openness": joblib.load("J_label_classifier.pkl"),
    "Conscientiousness": joblib.load("N_label_classifier.pkl"),
    "Agreeableness": joblib.load("T_label_classifier.pkl"),
}

# =====================================================
# Schema (define the expected input)
# =====================================================
class TextInput(BaseModel):
    text: str  # Input text field

# =====================================================
# Inference (use the loaded embedder and classifiers)
# =====================================================
def predict_personality(text: str):
    # Convert text to embedding using the embedder
    embedding = embedder.encode([text])

    prediction = {}
    confidence = {}

    # For each trait, run inference using the respective model
    for trait, model in models.items():
        pred = model.predict(embedding)[0]
        prob = model.predict_proba(embedding)[0][pred]

        # Store predictions and confidence scores
        prediction[trait] = "High" if pred == 1 else "Low"
        confidence[trait] = round(float(prob), 3)

    return {
        "prediction": prediction,
        "confidence": confidence
    }

# API endpoint for predictions
@app.post("/predict")
def predict(input: TextInput):
    return predict_personality(input.text)
