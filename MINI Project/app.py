from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# --- Load pickle files ---
with open("preprocessing_pipeline.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("label_encoder_district.pkl", "rb") as f:
    district_label_encoder = pickle.load(f)

with open("label_encoder_disease.pkl", "rb") as f:
    disease_label_encoder = pickle.load(f)

with open("disease_outbreak_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- State list for dropdown ---
STATE_LIST = [
    "Tamil Nadu", "Kerala", "Karnataka", "Andhra Pradesh", "Telangana",
    "Maharashtra", "Gujarat", "Delhi", "Rajasthan", "West Bengal",
    "Uttar Pradesh", "Madhya Pradesh", "Bihar", "Punjab", "Odisha"
]


@app.route('/')
def home():
    return render_template('index.html', states=STATE_LIST)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- Get user input ---
        week_of_outbreak = float(request.form['week'])
        state_ut = request.form['state'].strip()
        district = request.form['district'].strip()
        Cases = float(request.form['cases'])
        Deaths = float(request.form['deaths'])
        mon = int(request.form['mon'])
        year = int(request.form['year'])
        preci = float(request.form['preci'])
        LAI = float(request.form['lai'])
        Temp = float(request.form['temp'])

        # --- Encode district ---
        if district in district_label_encoder.classes_:
            district_encoded = district_label_encoder.transform([district])[0]
        else:
            # Assign a neutral value for unseen district
            district_encoded = -1

        # --- Build dataframe ---
        input_df = pd.DataFrame([{
            'week_of_outbreak': week_of_outbreak,
            'state_ut': state_ut,
            'district': district_encoded,
            'Cases': Cases,
            'Deaths': Deaths,
            'mon': mon,
            'year': year,
            'preci': preci,
            'LAI': LAI,
            'Temp': Temp
        }])

        # --- Enforce data types ---
        numeric_cols = ['week_of_outbreak', 'Cases', 'Deaths', 'mon', 'year', 'preci', 'LAI', 'Temp', 'district']
        categorical_cols = ['state_ut']

        # Convert numeric columns to float
        input_df[numeric_cols] = input_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        # Convert categorical to string (important fix!)
        input_df[categorical_cols] = input_df[categorical_cols].astype(str)

        # --- Apply preprocessing ---
        processed_input = preprocessor.transform(input_df)

        # --- Predict disease probabilities ---
        probs = model.predict_proba(processed_input)[0]
        pred_index = np.argmax(probs)
        pred_disease = disease_label_encoder.inverse_transform([pred_index])[0]
        pred_prob = round(probs[pred_index] * 100, 2)

        result_text = f"The probability of {pred_disease} is {pred_prob}%"
        return render_template('result.html', prediction=result_text)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
