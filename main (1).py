from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib

app = FastAPI(title="Predicted You vs Real You API")

embedder = None
models = {}

# =====================================================
# Load models ON STARTUP (not at import time)
# =====================================================
@app.on_event("startup")
def load_models():
    global embedder, models

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    models = {
        "Extraversion": joblib.load("E_label_classifier.pkl"),
        "Openness": joblib.load("J_label_classifier.pkl"),
        "Conscientiousness": joblib.load("N_label_classifier.pkl"),
        "Agreeableness": joblib.load("T_label_classifier.pkl"),
    }

# =====================================================
# Schema
# =====================================================
class TextInput(BaseModel):
    text: str

# =====================================================
# Inference
# =====================================================
def predict_personality(text: str):
    embedding = embedder.encode([text])

    prediction = {}
    confidence = {}

    for trait, model in models.items():
        pred = model.predict(embedding)[0]
        prob = model.predict_proba(embedding)[0][pred]

        prediction[trait] = "High" if pred == 1 else "Low"
        confidence[trait] = round(float(prob), 3)

    return {
        "prediction": prediction,
        "confidence": confidence
    }

@app.post("/predict")
def predict(input: TextInput):
    return predict_personality(input.text)
