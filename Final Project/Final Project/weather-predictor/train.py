import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

# ──────────────────────────────────────────────
# 1. Load the dataset
# ──────────────────────────────────────────────
df = pd.read_csv("data/weatherAUS.csv")
print(f"Original shape: {df.shape}")

# ──────────────────────────────────────────────
# 2. Drop columns with more than 30% missing values
# ──────────────────────────────────────────────
threshold = 0.3
missing_ratio = df.isnull().mean()
cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
print(f"Dropping columns (>{threshold*100:.0f}% missing): {cols_to_drop}")
df.drop(columns=cols_to_drop, inplace=True)

# ──────────────────────────────────────────────
# 3. Fill remaining missing values
#    - Median for numeric columns
#    - Mode for categorical columns
# ──────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=["object", "string"]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ──────────────────────────────────────────────
# 4. Encode the target column "RainTomorrow"
#    Yes -> 1, No -> 0
# ──────────────────────────────────────────────
df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})

# Also encode RainToday if it still exists
if "RainToday" in df.columns:
    df["RainToday"] = df["RainToday"].map({"Yes": 1, "No": 0})

# ──────────────────────────────────────────────
# 5. One-hot encode categorical columns
# ──────────────────────────────────────────────
# Drop the 'Date' column if present (not useful for modelling)
if "Date" in df.columns:
    df.drop(columns=["Date"], inplace=True)

# Identify remaining categorical columns after encoding
cat_cols_to_encode = [
    col for col in df.select_dtypes(include=["object", "string"]).columns
    if col in ["Location", "WindGustDir", "WindDir9am", "WindDir3pm"]
]

# Drop any other remaining non-numeric columns not in our encode list
remaining_obj = [
    col for col in df.select_dtypes(include=["object", "string"]).columns
    if col not in cat_cols_to_encode
]
if remaining_obj:
    print(f"Dropping extra non-numeric columns: {remaining_obj}")
    df.drop(columns=remaining_obj, inplace=True)

df = pd.get_dummies(df, columns=cat_cols_to_encode, drop_first=True)

# ──────────────────────────────────────────────
# 6. Print cleaned dataframe info
# ──────────────────────────────────────────────
print(f"\nCleaned dataframe shape: {df.shape}")
print(f"\nClass distribution of RainTomorrow:\n{df['RainTomorrow'].value_counts()}")
print(f"\nClass proportions:\n{df['RainTomorrow'].value_counts(normalize=True)}")

# ══════════════════════════════════════════════
#  MODEL TRAINING
# ══════════════════════════════════════════════

# ──────────────────────────────────────────────
# 7. Select features
# ──────────────────────────────────────────────
features = [
    "MinTemp", "MaxTemp", "Rainfall",
    "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm",
    "Temp9am", "Temp3pm",
    "WindGustSpeed"
]

X = df[features]
y = df["RainTomorrow"]

print(f"\nSelected features ({len(features)}): {features}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# ──────────────────────────────────────────────
# 8. Split into train/test sets (80/20)
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# ──────────────────────────────────────────────
# 9. Normalize features using StandardScaler
# ──────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ──────────────────────────────────────────────
# 10. Train Random Forest Classifier
# ──────────────────────────────────────────────
print("\nTraining Random Forest Classifier (n_estimators=100)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
print("Training complete!")

rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"\n{'='*50}")
print(f" Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print(f"{'='*50}")
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, rf_pred, target_names=["No Rain", "Rain"]))
print("Confusion Matrix (Random Forest):")
print(confusion_matrix(y_test, rf_pred))

# ──────────────────────────────────────────────
# 11. Train XGBoost Classifier
# ──────────────────────────────────────────────
print("\nTraining XGBoost Classifier (n_estimators=100)...")
xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
xgb_model.fit(X_train_scaled, y_train)
print("Training complete!")

xgb_pred = xgb_model.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, xgb_pred)

print(f"\n{'='*50}")
print(f" XGBoost Accuracy: {xgb_accuracy:.4f} ({xgb_accuracy*100:.2f}%)")
print(f"{'='*50}")
print("\nClassification Report (XGBoost):")
print(classification_report(y_test, xgb_pred, target_names=["No Rain", "Rain"]))
print("Confusion Matrix (XGBoost):")
print(confusion_matrix(y_test, xgb_pred))

# ──────────────────────────────────────────────
# 12. Compare models and save the best one
# ──────────────────────────────────────────────
print(f"\n{'='*50}")
print(" MODEL COMPARISON")
print(f"{'='*50}")
print(f" Random Forest : {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print(f" XGBoost       : {xgb_accuracy:.4f} ({xgb_accuracy*100:.2f}%)")

diff = abs(rf_accuracy - xgb_accuracy)

if xgb_accuracy > rf_accuracy:
    best_model = xgb_model
    winner = "XGBoost"
elif rf_accuracy > xgb_accuracy:
    best_model = rf_model
    winner = "Random Forest"
else:
    best_model = rf_model
    winner = "Random Forest (tie — defaulting to RF)"

print(f"\n🏆 Winner: {winner} (by {diff*100:.2f}%)")
print(f"{'='*50}")

joblib.dump(best_model, "model/rain_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print(f"\nBest model ({winner}) saved to model/rain_model.pkl")
print("Scaler saved to model/scaler.pkl")

# ──────────────────────────────────────────────
# 13. Save chart data for the stats dashboard
# ──────────────────────────────────────────────
import json

class_counts = y.value_counts().sort_index()
rf_importances = rf_model.feature_importances_.tolist()

chart_data = {
    "class_distribution": {
        "labels": ["No Rain", "Rain"],
        "counts": [int(class_counts[0]), int(class_counts[1])]
    },
    "feature_importance": {
        "labels": features,
        "values": [round(v, 4) for v in rf_importances]
    }
}

with open("model/chart_data.json", "w") as f:
    json.dump(chart_data, f, indent=2)

print("Chart data saved to model/chart_data.json")
