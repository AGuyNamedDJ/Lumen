import os
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Set the absolute paths for predictions and true values
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', 'lumen_2')

# Paths for real-time predictions and true values
predictions_real_path = os.path.join(models_dir, 'predictions_real.npy')
true_values_real_path = os.path.join(models_dir, 'y_test_real.npy')
# Paths for historical predictions and true values
predictions_hist_path = os.path.join(models_dir, 'predictions_hist.npy')
true_values_hist_path = os.path.join(models_dir, 'y_test_hist.npy')

# Print the file paths to check if they are correct
print(
    f"Attempting to load real-time predictions from: {predictions_real_path}")
print(
    f"Attempting to load real-time true values from: {true_values_real_path}")
print(f"Attempting to load historical predictions from: {
      predictions_hist_path}")
print(f"Attempting to load historical true values from: {
      true_values_hist_path}")


def evaluate_predictions(predictions_path, true_values_path, model_name):
    # Load the predicted values
    predictions = np.load(predictions_path).flatten()
    logging.debug(f"Loaded {len(predictions)} predictions for {model_name}.")

    # Load the true values
    y_true = np.load(true_values_path)
    logging.debug(f"Loaded {len(y_true)} true values for {model_name}.")

    # Check if the predictions and true values are aligned
    if len(predictions) != len(y_true):
        logging.error(
            f"Mismatch between predictions and true values for {model_name}.")
        logging.error(f"Predictions length: {len(predictions)}")
        logging.error(f"True values length: {len(y_true)}")
        raise ValueError(
            f"Mismatch between predictions and true values for {model_name}.")

    # Print first 10 rows of predictions and true values for manual inspection
    print(f"First 10 predictions for {model_name}: {predictions[:10]}")
    print(f"First 10 true values for {model_name}: {y_true[:10]}")

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, predictions)
    logging.info(f"{model_name} - Mean Squared Error (MSE): {mse}")

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    logging.info(f"{model_name} - Root Mean Squared Error (RMSE): {rmse}")

    # Calculate R-squared (R²)
    r2 = r2_score(y_true, predictions)
    logging.info(f"{model_name} - R-squared (R²): {r2}")

    # Optionally print or save these metrics to a file
    print(f"{model_name} - MSE: {mse}")
    print(f"{model_name} - RMSE: {rmse}")
    print(f"{model_name} - R²: {r2}")


if __name__ == "__main__":
    # Evaluate real-time model
    evaluate_predictions(predictions_real_path,
                         true_values_real_path, "Real-Time Model")

    # Evaluate historical model
    evaluate_predictions(predictions_hist_path,
                         true_values_hist_path, "Historical Model")
