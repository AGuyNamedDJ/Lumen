import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Paths to saved models
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_NAME_HIST = 'Lumen2_historical.keras'  # Historical model name
MODEL_NAME_REAL = 'Lumen2_real_time.keras'   # Real-time model name

# Load models
model_hist = load_model(os.path.join(MODEL_DIR, MODEL_NAME_HIST))
model_real = load_model(os.path.join(MODEL_DIR, MODEL_NAME_REAL))

# Load test data


def load_test_data():
    X_test_hist = np.load(os.path.join(MODEL_DIR, 'X_test_hist.npy'))
    y_test_hist = np.load(os.path.join(MODEL_DIR, 'y_test_hist.npy'))
    X_test_real = np.load(os.path.join(MODEL_DIR, 'X_test_real.npy'))
    y_test_real = np.load(os.path.join(MODEL_DIR, 'y_test_real.npy'))

    return X_test_hist, y_test_hist, X_test_real, y_test_real


def evaluate_model(model, X_test, y_test, model_name, expected_features):
    logging.debug(f"Evaluating model: {model_name}")

    # Check if the test data has the correct number of features, if not, adjust
    if X_test.shape[2] != expected_features:
        logging.debug(f"Trimming test data to {expected_features} features")
        # Keep only the first N features
        X_test = X_test[:, :, :expected_features]

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Log the results
    logging.info(f"{model_name} Evaluation Results:")
    logging.info(f"MSE: {mse}")
    logging.info(f"RMSE: {rmse}")
    logging.info(f"R²: {r2}")

    return mse, rmse, r2, y_pred


def plot_predictions_vs_actual(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.title(f"{model_name} Predictions vs. Actual")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()


def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.75)
    plt.title(f"{model_name} Residuals (Errors)")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.show()


def main():
    X_test_hist, y_test_hist, X_test_real, y_test_real = load_test_data()

    # Evaluate the historical model with 41 features
    mse_hist, rmse_hist, r2_hist, y_pred_hist = evaluate_model(
        model_hist, X_test_hist, y_test_hist, "Historical Model", expected_features=41)

    # Evaluate the real-time model with 36 features
    mse_real, rmse_real, r2_real, y_pred_real = evaluate_model(
        model_real, X_test_real, y_test_real, "Real-Time Model", expected_features=36)

    print("Evaluation Results:")
    print(
        f"Historical Model - MSE: {mse_hist}, RMSE: {rmse_hist}, R²: {r2_hist}")
    print(
        f"Real-Time Model - MSE: {mse_real}, RMSE: {rmse_real}, R²: {r2_real}")

    # # Plot predictions vs actual
    # plot_predictions_vs_actual(y_test_hist, y_pred_hist, "Historical Model")
    # plot_predictions_vs_actual(y_test_real, y_pred_real, "Real-Time Model")

    # # Plot residuals (errors)
    # plot_residuals(y_test_hist, y_pred_hist, "Historical Model")
    # plot_residuals(y_test_real, y_pred_real, "Real-Time Model")


if __name__ == "__main__":
    main()
