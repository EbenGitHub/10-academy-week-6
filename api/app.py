from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load models
with open('log_reg_model.pkl', 'rb') as f:
    log_reg_model = pickle.load(f)
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('gbm_model.pkl', 'rb') as f:
    gbm_model = pickle.load(f)

@app.route('/predict/<model>', methods=['POST'])
def predict(model):
    data = request.get_json(force=True)
    input_df = pd.DataFrame([data], columns=['Recency', 'Frequency', 'Monetary', 'Size'])
    if model == 'logistic_regression':
        prediction = log_reg_model.predict(input_df)[0]
    elif model == 'random_forest':
        prediction = rf_model.predict(input_df)[0]
    elif model == 'gradient_boosting':
        prediction = gbm_model.predict(input_df)[0]
    else:
        return jsonify({'error': 'Invalid model name'}), 400
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
