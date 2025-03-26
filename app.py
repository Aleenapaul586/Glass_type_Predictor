from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Load trained model and scaler
with open("knn_model.pkl", "rb") as model_file:
    knn_model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Get input values from form
            features = [
                float(request.form["Refractive Index"]),
                float(request.form["Sodium"]),
                float(request.form["Magnesium"]),
                float(request.form["Aluminum"]),
                float(request.form["Silicon"]),
                float(request.form["Potassium"]),
                float(request.form["Calcium"]),
                float(request.form["Barium"]),
                float(request.form["Iron"]),
            ]

            # Convert to NumPy array and reshape for model
            input_data = np.array(features).reshape(1, -1)

            # Standardize input features
            input_data_scaled = scaler.transform(input_data)

            # Predict
            prediction = knn_model.predict(input_data_scaled)[0]
            prediction = "Window" if prediction == 0 else "Container"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
