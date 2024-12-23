import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Paths to saved models
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME_HIST = 'Lumen2_historical.keras'
MODEL_NAME_REAL = 'Lumen2_real_time.keras'

# Load models
model_hist = load_model(os.path.join(MODEL_DIR, MODEL_NAME_HIST))
model_real = load_model(os.path.join(MODEL_DIR, MODEL_NAME_REAL))

# Function to load test data using memory-mapping


def load_test_data():
    try:
        # Load historical test data normally
        X_test_hist = np.load(os.path.join(MODEL_DIR, 'X_test_hist.npy'))
        y_test_hist = np.load(os.path.join(MODEL_DIR, 'y_test_hist.npy'))

        # Memory-map the real-time test data
        X_test_real = np.load(os.path.join(
            MODEL_DIR, 'X_test_real.npy'), mmap_mode='r')
        y_test_real = np.load(os.path.join(
            MODEL_DIR, 'y_test_real.npy'), mmap_mode='r')

    except FileNotFoundError as e:
        logging.error(f"Test data file not found: {e}")
        raise e

    return X_test_hist, y_test_hist, X_test_real, y_test_real

# Data generator for memory-mapped arrays


def memmap_data_generator(X, y, batch_size, expected_features):
    num_samples = X.shape[0]
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        # Check for feature mismatch and pad if necessary
        if X_batch.shape[2] != expected_features:
            missing_feature_count = expected_features - X_batch.shape[2]
            if missing_feature_count > 0:
                logging.warning(
                    f"Adding {missing_feature_count} missing features to test data.")
                X_batch = np.pad(
                    X_batch, ((0, 0), (0, 0), (0, missing_feature_count)),
                    'constant', constant_values=0)
            else:
                logging.error(f"Feature mismatch: expected {
                              expected_features}, got {X_batch.shape[2]}")
                raise ValueError(f"Feature mismatch: expected {
                                 expected_features}, got {X_batch.shape[2]}")

        yield X_batch.astype(np.float32), y_batch.astype(np.float32)

# Evaluate model function using generator


def evaluate_model_generator(model, data_gen, num_samples, model_name, expected_features, batch_size=256):
    logging.debug(f"Evaluating model: {model_name}")

    # Initialize lists to store predictions and true values
    y_true = []
    y_pred = []

    steps = int(np.ceil(num_samples / batch_size))

    for _ in range(steps):
        try:
            X_batch, y_batch = next(data_gen)
            y_batch_pred = model.predict(X_batch)
            y_true.append(y_batch)
            y_pred.append(y_batch_pred)
        except StopIteration:
            break

    # Concatenate all predictions and true values
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    # Proceed with evaluation as before
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Log the results
    logging.info(f"{model_name} Evaluation Results:")
    logging.info(f"MSE: {mse}")
    logging.info(f"RMSE: {rmse}")
    logging.info(f"R²: {r2}")

    return mse, rmse, r2, y_pred

# Evaluate model function for smaller datasets (historical data)


def evaluate_model(model, X_test, y_test, model_name, expected_features):
    logging.debug(f"Evaluating model: {model_name}")

    if X_test.shape[0] == 0:
        logging.error(f"No test data available for {model_name}.")
        raise ValueError(f"No test data available for {model_name}.")

    # Check for feature mismatch and pad if necessary
    if X_test.shape[2] != expected_features:
        missing_feature_count = expected_features - X_test.shape[2]
        if missing_feature_count > 0:
            logging.warning(f"Adding {missing_feature_count} missing features to {
                            model_name} test data.")
            X_test = np.pad(
                X_test, ((0, 0), (0, 0), (0, missing_feature_count)),
                'constant', constant_values=0)
        else:
            logging.error(f"Feature mismatch: expected {
                          expected_features}, got {X_test.shape[2]}")
            raise ValueError(f"Feature mismatch: expected {
                             expected_features}, got {X_test.shape[2]}")

    # Ensure data types are optimized
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Predict
    y_pred = model.predict(X_test)

    # Proceed with evaluation as before
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Log the results
    logging.info(f"{model_name} Evaluation Results:")
    logging.info(f"MSE: {mse}")
    logging.info(f"RMSE: {rmse}")
    logging.info(f"R²: {r2}")

    return mse, rmse, r2, y_pred

# Main function to evaluate both models


def main():
    X_test_hist, y_test_hist, X_test_real, y_test_real = load_test_data()

    # Print shapes for historical data
    print('X_test_hist shape:', X_test_hist.shape)
    print('y_test_hist shape:', y_test_hist.shape)
    print('X_test_real shape:', X_test_real.shape)
    print('y_test_real shape:', y_test_real.shape)

    # Get expected features from model input shapes
    expected_input_shape_hist = model_hist.input_shape
    expected_features_hist = expected_input_shape_hist[2]  # Should be 44
    expected_input_shape_real = model_real.input_shape
    expected_features_real = expected_input_shape_real[2]  # Should be 58

    # Evaluate the historical model
    mse_hist, rmse_hist, r2_hist, y_pred_hist = evaluate_model(
        model_hist, X_test_hist, y_test_hist, "Historical Model", expected_features=expected_features_hist)

    # Evaluate the real-time model using the generator
    batch_size = 256  # Adjust based on your system's capacity

    data_gen = memmap_data_generator(
        X_test_real, y_test_real, batch_size, expected_features_real)

    num_samples = X_test_real.shape[0]

    mse_real, rmse_real, r2_real, y_pred_real = evaluate_model_generator(
        model_real, data_gen, num_samples, "Real-Time Model", expected_features_real, batch_size=batch_size)

    print("Evaluation Results:")
    print(
        f"Historical Model - MSE: {mse_hist}, RMSE: {rmse_hist}, R²: {r2_hist}")
    print(
        f"Real-Time Model - MSE: {mse_real}, RMSE: {rmse_real}, R²: {r2_real}")


if __name__ == "__main__":
    main()
