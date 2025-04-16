from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load trained model and vectorizer
model = joblib.load("model.pkl")        # This must be trained
vectorizer = joblib.load("vectorizer.pkl")

class ComplaintInput(BaseModel):
    description: str

@app.post("/predict")
def predict(data: ComplaintInput):
    vectorized_text = vectorizer.transform([data.description])
    prediction = model.predict(vectorized_text)[0]
    return {"category": prediction}
