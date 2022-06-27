import pickle
import pandas as pd

from flask import Flask, request, jsonify

def predict_single(individual, ct, model):
    X = ct.transform(pd.DataFrame([individual]))
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


with open('income-prediction-model.bin', 'rb') as f_in:
    ct, model = pickle.load(f_in)

app = Flask('incomepredictor')

@app.route("/")
def main_message():
    return "Income Predictor"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    individual = request.get_json()

    prediction = predict_single(individual, ct, model)
    highincome = prediction >= 0.5
    
    result = {
        'prediction_probability': float(prediction),
        'highincome': bool(highincome),
    }

    return jsonify(result)






