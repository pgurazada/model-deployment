import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Predicting wine quality")

# Represents a particular wine 
class Wine(BaseModel):
    fixed_acidity: float = 7.4
    volatile_acidity: float = 0.7
    citric_acid: float = 0.0
    residual_sugar: float = 1.9
    chlorides: float = 0.076
    free_sulfur_dioxide: float = 11.0
    total_sulfur_dioxide: float = 34.0
    density: float = 0.9978
    pH: float = 3.51 
    sulphates: float = 0.56
    alcohol: float = 9.4

@app.on_event("startup")
def load_classifier():
    # Load classifier from pickle file
    with open("wine-quality-prediction-model.pkl", "rb") as file:
        global classifier
        classifier = pickle.load(file)


@app.get("/")
def home():
    return "Welcome, for predictions go to http://localhost:8000/docs"


@app.post("/predict/v1")
def predict(wine: Wine):
    data_point = np.array(
        [
            [
                wine.fixed_acidity,
                wine.volatile_acidity,
                wine.citric_acid,
                wine.residual_sugar,
                wine.chlorides,
                wine.free_sulfur_dioxide,
                wine.total_sulfur_dioxide,
                wine.density,
                wine.pH,
                wine.sulphates,
                wine.alcohol
            ]
        ]
    )

    pred = classifier.predict(data_point).tolist()
    pred = pred[0]
    print(pred)
    return {"Prediction": pred}