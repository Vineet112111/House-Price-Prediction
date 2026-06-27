import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import os


app = Flask(__name__, template_folder='templates')

# Load CSV and model once at startup
data = pd.read_csv('data/Cleaned_Data.csv')

with open('models/RidgeModel.pkle', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location   = request.form.get('location')
        bhk        = int(request.form.get('bhk'))
        bath       = int(request.form.get('bath'))
        total_sqft = float(request.form.get('total_sqft'))

        # Build a one-row DataFrame with the same columns the model was trained on
        input_df = pd.DataFrame(
            [[location, total_sqft, bath, bhk]],
            columns=['location', 'total_sqft', 'bath', 'bhk']
        )

        # model.predict returns price in Lakhs (standard for Bangalore datasets)
        price = model.predict(input_df)[0]

        return jsonify({'status': 'ok', 'price': round(float(price), 2)})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
