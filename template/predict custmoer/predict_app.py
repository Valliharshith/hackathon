import os
from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Base path is current script's folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(os.path.dirname(BASE_DIR), 'templates')
CSV_PATH = os.path.join(BASE_DIR, 'social_ads.csv')

# Check if required files/folders exist
if not os.path.exists(TEMPLATE_PATH):
    raise FileNotFoundError(f"Templates folder not found: {TEMPLATE_PATH}")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV dataset file not found: {CSV_PATH}")

print("Template folder path being used:", TEMPLATE_PATH)
print("Files in templates folder:", os.listdir(TEMPLATE_PATH))
print("CSV file path being used:", CSV_PATH)

# Flask app with correct template path
app = Flask(__name__, template_folder=TEMPLATE_PATH)

# Load dataset and train model
df = pd.read_csv(CSV_PATH)
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        salary = float(request.form['salary'])

        input_scaled = scaler.transform([[age, salary]])
        proba = model.predict_proba(input_scaled)[0][1]
        prediction = int(proba >= 0.5)

        result_text = f"{'Will Purchase' if prediction else 'Will NOT Purchase'} (Probability: {round(proba, 2)})"
        return render_template('predict.html', prediction_text=result_text)
    except Exception as e:
        return render_template('predict.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    print("Starting Flask app on port 8080...")
    app.run(debug=True, port=8080)
