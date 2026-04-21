# 🌦️ Weather Predictor

A machine learning web application that predicts whether it will rain tomorrow based on today's weather data. Built with **Flask**, **scikit-learn**, and **XGBoost**, using the [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) dataset.

## Features

- **Rain Prediction** — Enter 10 weather parameters and get an instant rain/no-rain forecast with confidence percentage
- **Stats Dashboard** — View dataset class distribution and model feature importance via interactive Chart.js charts
- **Model Comparison** — Trains both Random Forest and XGBoost classifiers, automatically saves the best performer

## Project Structure

```
weather-predictor/
├── data/
│   └── weatherAUS.csv          # Rain in Australia dataset
├── model/
│   ├── rain_model.pkl          # Best trained model (saved by train.py)
│   ├── scaler.pkl              # StandardScaler (saved by train.py)
│   └── chart_data.json         # Chart data for stats dashboard
├── static/
│   └── style.css               # Application styles
├── templates/
│   ├── index.html              # Prediction page
│   └── stats.html              # Stats dashboard page
├── app.py                      # Flask web application
├── train.py                    # Data processing & model training script
├── requirements.txt            # Python dependencies
└── README.md
```

## Installation

### 1. Clone or navigate to the project directory

```bash
cd weather-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

Make sure the dataset file `weatherAUS.csv` is in the `data/` folder, then run:

```bash
python train.py
```

This will:
- Load and clean the dataset (handle missing values, encode features)
- Train a Random Forest and an XGBoost classifier
- Compare both models and save the best one to `model/`
- Generate chart data for the stats dashboard

### 4. Run the Flask app

```bash
python app.py
```

The app will start at **http://127.0.0.1:5000**

## Usage

### Predictor Page (`/`)

Enter the following weather parameters and click **Predict**:

| Parameter | Description |
|-----------|-------------|
| MinTemp | Minimum temperature (°C) |
| MaxTemp | Maximum temperature (°C) |
| Rainfall | Rainfall amount (mm) |
| WindGustSpeed | Wind gust speed (km/h) |
| Humidity9am | Humidity at 9 AM (%) |
| Humidity3pm | Humidity at 3 PM (%) |
| Pressure9am | Atmospheric pressure at 9 AM (hPa) |
| Pressure3pm | Atmospheric pressure at 3 PM (hPa) |
| Temp9am | Temperature at 9 AM (°C) |
| Temp3pm | Temperature at 3 PM (°C) |

### Stats Dashboard (`/stats`)

View interactive charts showing:
- **Class Distribution** — Count of rainy vs non-rainy days in the dataset
- **Feature Importance** — Which weather features matter most for prediction

## Tech Stack

- **Backend**: Flask (Python)
- **ML Models**: scikit-learn (Random Forest), XGBoost
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Data Processing**: pandas, NumPy
