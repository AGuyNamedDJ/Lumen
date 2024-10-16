import os
import sys
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
import numpy as np
import logging
from tensorflow.keras.models import load_model
# from definitions_lumen_2 import ReduceMeanLayer
from models.lumen_2.definitions_lumen_2 import ReduceMeanLayer

# Add the correct path for module discovery
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define current_dir to refer to the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))


def load_data_for_prediction(file_path, is_historical=False):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Check if it's historical or real-time data and use appropriate time column
    time_col = 'date' if is_historical else 'timestamp'

    # Convert 'date' or 'timestamp' to datetime
    if time_col in data.columns:
        data[time_col] = pd.to_datetime(data[time_col], errors='coerce')
    else:
        raise KeyError(f"Column '{time_col}' not found in the dataset.")

    # Extract useful time-based features
    data['day_of_week'] = data[time_col].dt.dayofweek   # Monday=0, Sunday=6
    data['hour_of_day'] = data[time_col].dt.hour  # Hour of the day (0-23)
    data['day_of_month'] = data[time_col].dt.day  # Day of the month (1-31)
    data['month_of_year'] = data[time_col].dt.month     # Month (1-12)
    data['seconds_since_start'] = (
        data[time_col] - data[time_col].min()).dt.total_seconds()

    # Drop the 'FEDFUNDS' column if it exists
    if 'FEDFUNDS' in data.columns:
        data = data.drop(columns=['FEDFUNDS'])
        print("Dropped 'FEDFUNDS' column.")

    # Drop any remaining non-numeric columns (in case there are other string columns)
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"Dropping non-numeric columns: {non_numeric_cols}")
        data = data.drop(columns=non_numeric_cols)

    # Drop the original 'date' or 'timestamp' column and any other unnecessary columns
    data = data.drop(columns=[time_col], errors='ignore')

    # Check if data has any non-numeric columns remaining
    print(data.dtypes)

    # Convert to float32 to ensure compatibility with TensorFlow models
    try:
        data = data.astype(np.float32)
    except ValueError as e:
        print(f"Error converting data to float32: {e}")
        raise

    # Check for NaNs and fill if needed
    if data.isnull().values.any():
        print("Warning: Data contains NaN values. Filling with 0.")
        data = data.fillna(0)

    return data


def load_trained_model(model_path):
    # Load the saved hybrid model with custom layer
    try:
        model = load_model(model_path, custom_objects={
                           'ReduceMeanLayer': ReduceMeanLayer})
        logging.debug(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load the model: {e}")
        return None
    return model


def predict_with_model(model, input_data):
    # Extract the input shape expected by the model
    expected_shape = model.input_shape
    expected_timesteps = expected_shape[1]
    expected_features = expected_shape[2]

    # Convert DataFrame to NumPy array
    input_data = input_data.values

    # Ensure that input_data has the correct number of features
    if input_data.shape[1] > expected_features:
        print(f"Trimming input data to {expected_features} features")
        input_data = input_data[:, :expected_features]

    # Create sequences of the expected length
    sequences = []
    for i in range(len(input_data) - expected_timesteps + 1):
        sequences.append(input_data[i:i + expected_timesteps])
    sequences = np.array(sequences)

    # Check the reshaped input data shape
    print(f"Reshaped input data shape: {sequences.shape}")

    # Make predictions using the model
    predictions = model.predict(sequences)

    return predictions


if __name__ == "__main__":
    # === Predict for Historical Model ===
    # Load test sequences
    X_test_hist = np.load(os.path.join(
        current_dir, 'X_test_hist.npy'))

    # Load the historical model
    hist_model_file = os.path.join(
        current_dir, 'Lumen2_historical.keras')
    hist_model = load_trained_model(hist_model_file)

    if hist_model is not None:
        # Make predictions on historical test data
        hist_predictions = hist_model.predict(X_test_hist)
        print(hist_predictions)

        # Save historical predictions to a file
        np.save(os.path.join(current_dir, 'predictions_hist.npy'), hist_predictions)
        logging.debug(
            f"Historical predictions saved to models/predictions_hist.npy")
    else:
        print("Failed to load historical model.")

    # === Predict for Real-Time Model ===
    # Load test sequences
    X_test_real = np.load(os.path.join(
        current_dir, 'X_test_real.npy'))

    # Load the real-time model
    real_time_model_file = os.path.join(
        current_dir, 'Lumen2_real_time.keras')
    real_time_model = load_trained_model(real_time_model_file)

    if real_time_model is not None:
        # Make predictions on real-time test data
        real_time_predictions = real_time_model.predict(X_test_real)
        print(real_time_predictions)

        # Save real-time predictions to a file
        np.save(os.path.join(current_dir, 'predictions_real.npy'),
                real_time_predictions)
        logging.debug(
            f"Real-time predictions saved to models/predictions_real.npy")
    else:
        print("Failed to load real-time model.")
