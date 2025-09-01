# ml_models.py
# Advanced-looking ML models for Farmer Dashboard
# Author: Adi

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVC
from sklearn.metrics import classification_report, mean_squared_error, silhouette_score
import joblib

def divider(title):
    print("\n" + "="*30)
    print(title)
    print("="*30 + "\n")

def train_logistic_regression():
    divider("Training Logistic Regression – Crop Disease Classification")

    # Fake dataset (more samples)
    X = np.random.randint(20, 50, (50, 3))   # [leaf_temp, soil_moisture, humidity]
    y = np.random.randint(0, 2, 50)          # 0=Healthy, 1=Diseased

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    print(classification_report(y_test, preds))
    joblib.dump(pipe, "logistic_crop_disease.pkl")
    print("✅ Logistic Regression saved. Example:", pipe.predict([[32, 38, 55]]))

# ---------------------------
# 2. Polynomial Linear Regression – Yield Prediction
# ---------------------------
def train_linear_regression():
    divider("Training Polynomial Linear Regression – Yield Prediction")

    # Fake dataset
    X = np.random.randint(50, 250, (60, 3))  # [rainfall, soil_nitrogen, temperature]
    y = np.random.uniform(1.5, 4.0, 60)      # yield tons/ha

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression())
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    print("MSE:", mse)
    joblib.dump(pipe, "poly_linear_yield.pkl")
    print("✅ Polynomial Linear Regression saved. Example:", pipe.predict([[100, 45, 27]]))

# ---------------------------
# 3. KMeans – Farmer Segmentation
# ---------------------------
def train_kmeans():
    divider("Training KMeans – Farmer Segmentation")

    X = np.random.randint(20, 100, (40, 3))  # [soil_quality, rainfall, temp]
    model = KMeans(n_clusters=3, n_init=15, random_state=42)
    model.fit(X)

    sil_score = silhouette_score(X, model.labels_)
    print("Silhouette Score:", sil_score)
    joblib.dump(model, "kmeans_farmers.pkl")
    print("✅ KMeans saved. Cluster Assignments:", model.labels_[:10])

# ---------------------------
# 4. Ensemble Regressors – Advanced Yield Prediction
# ---------------------------
def train_ensemble_regressors():
    divider("Training Ensemble Models – Yield Prediction")

    X = np.random.randint(50, 250, (80, 3))
    y = np.random.uniform(1.0, 5.0, 80)

    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=50, random_state=42)

    ensemble = VotingRegressor([("rf", rf), ("gb", gb)])
    ensemble.fit(X, y)

    preds = ensemble.predict(X[:5])
    print("Sample Ensemble Predictions:", preds)

    joblib.dump(ensemble, "ensemble_yield.pkl")
    print("✅ Ensemble Regressor saved.")

# ---------------------------
# 5. SVM with GridSearch – Crop Suitability Classification
# ---------------------------
def train_svm():
    divider("Training SVM – Crop Suitability Classification with Hyperparameter Tuning")

    X = np.random.randint(50, 200, (50, 3))   # [rainfall, soil_ph*10, temp]
    y = np.random.randint(0, 3, 50)           # 0=Rice, 1=Wheat, 2=Maize

    param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=3, verbose=0)

    grid.fit(X, y)
    print("Best Params:", grid.best_params_)
    joblib.dump(grid.best_estimator_, "svm_crop.pkl")
    print("✅ SVM saved. Example:", grid.best_estimator_.predict([[110, 64, 27]]))

# ---------------------------
# Run all trainings
# ---------------------------
if __name__ == "__main__":
    train_logistic_regression()
    train_linear_regression()
    train_kmeans()
    train_ensemble_regressors()
    train_svm()
    print("\n🎯 All advanced models trained and saved as .pkl files.")
