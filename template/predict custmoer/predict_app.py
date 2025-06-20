import os
import random
import pandas as pd
from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set paths
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
data_path = r'C:\Users\kanki\consumers-buying-behavior\social_ads_extended.csv'
base_path = r'C:\Users\kanki\consumers-buying-behavior\social_ads.csv'

# Initialize Flask
app = Flask(__name__, template_folder=template_dir)

# Create extended dataset if not exists
if not os.path.exists(data_path):
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"❌ Could not find base dataset: {base_path}")
    df_base = pd.read_csv(base_path)
    df_base['Year'] = [random.randint(2018, 2024) for _ in range(len(df_base))]
    df_base['Category'] = [random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports', 'Beauty', 'Toys']) for _ in range(len(df_base))]
    df_base['Price'] = [round(random.uniform(20, 500), 2) for _ in range(len(df_base))]
    df_base['Device'] = [random.choice(['Mobile', 'Desktop', 'Tablet']) for _ in range(len(df_base))]
    df_base['PreviousPurchases'] = [random.randint(0, 10) for _ in range(len(df_base))]
    df_base['TimeOnSite'] = [round(random.uniform(10, 300), 1) for _ in range(len(df_base))]
    df_base.to_csv(data_path, index=False)
    print("✅ social_ads_extended.csv created.")

# Load extended dataset
df = pd.read_csv(data_path)

# Features & target
features = ['Age', 'EstimatedSalary', 'Year', 'Category', 'Price', 'Device', 'PreviousPurchases', 'TimeOnSite']
target = 'Purchased'
X = df[features]
y = df[target]

# Define column types
numeric_features = ['Age', 'EstimatedSalary', 'Year', 'Price', 'PreviousPurchases', 'TimeOnSite']
categorical_features = ['Category', 'Device']

# Pipeline: scale + encode + model
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression())
])

pipeline.fit(X, y)

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'Age': float(request.form['age']),
            'EstimatedSalary': float(request.form['salary']),
            'Year': int(request.form['year']),
            'Category': request.form['category'],
            'Price': float(request.form['price']),
            'Device': request.form['device'],
            'PreviousPurchases': int(request.form['prev_purchases']),
            'TimeOnSite': float(request.form['time_on_site']),
        }

        input_df = pd.DataFrame([input_data])
        proba = pipeline.predict_proba(input_df)[0][1]
        prediction = int(proba >= 0.5)

        result_text = f"{'🟢 Will Purchase' if prediction else '🔴 Will NOT Purchase'} (Probability: {round(proba, 2)})"
        return render_template('predict.html', prediction_text=result_text)
    except Exception as e:
        return render_template('predict.html', prediction_text=f"⚠️ Error: {e}")

if __name__ == '__main__':
    print("🚀 Starting Flask app on http://localhost:8080")
    app.run(debug=True, port=8080)
