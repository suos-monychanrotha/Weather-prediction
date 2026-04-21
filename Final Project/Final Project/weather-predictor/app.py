from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import json

app = Flask(__name__)

# ──────────────────────────────────────────────
# Load the trained model, scaler, and chart data on startup
# ──────────────────────────────────────────────
model = joblib.load("model/rain_model.pkl")
scaler = joblib.load("model/scaler.pkl")

with open("model/chart_data.json", "r") as f:
    chart_data = json.load(f)

FEATURE_KEYS = [
    "MinTemp", "MaxTemp", "Rainfall",
    "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm",
    "Temp9am", "Temp3pm",
    "WindGustSpeed"
]


@app.route("/")
def index():
    """Render the main prediction page."""
    return render_template("index.html")


@app.route("/stats")
def stats():
    """Render the stats dashboard page."""
    return render_template("stats.html")


@app.route("/chart-data")
def get_chart_data():
    """Return class distribution and feature importance as JSON."""
    return jsonify(chart_data)


@app.route("/predict", methods=["POST"])
def predict():
    """Accept JSON input, scale it, and return a rain prediction."""
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "Invalid request. Please send JSON data."}), 400

        # Validate that all required fields are present and numeric
        missing = [k for k in FEATURE_KEYS if k not in data]
        if missing:
            return jsonify({
                "error": f"Missing fields: {', '.join(missing)}"
            }), 400

        values = []
        for key in FEATURE_KEYS:
            val = data[key]
            if val is None or val == "":
                return jsonify({
                    "error": f"Field '{key}' cannot be empty."
                }), 400
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                return jsonify({
                    "error": f"Field '{key}' must be a valid number, got: {val}"
                }), 400

        input_array = np.array([values])

        # Scale the input using the loaded scaler
        input_scaled = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        # Get confidence for the predicted class
        confidence = float(probabilities[int(prediction)] * 100)
        label = "Rain" if prediction == 1 else "No Rain"

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
