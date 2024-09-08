import numpy as np
import os
import logging
from tensorflow.keras.models import load_model
from definitions_lumen_2 import ReduceMeanLayer
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def load_data_for_prediction(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Convert 'timestamp' to datetime if it's not already
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

    # Extract useful time-based features from the 'timestamp' column
    data['day_of_week'] = data['timestamp'].dt.dayofweek   # Monday=0, Sunday=6
    data['hour_of_day'] = data['timestamp'].dt.hour  # Hour of the day (0-23)
    data['day_of_month'] = data['timestamp'].dt.day  # Day of the month (1-31)
    data['month_of_year'] = data['timestamp'].dt.month     # Month (1-12)
    data['seconds_since_start'] = (
        data['timestamp'] - data['timestamp'].min()).dt.total_seconds()

    # Drop the original 'timestamp' and any other unnecessary columns
    data = data.drop(
        columns=['timestamp', 'conditions_real_time_spy'], errors='ignore')

    # Check if data has any non-numeric columns
    print(data.dtypes)

    # Convert to float32 to ensure compatibility with TensorFlow models
    data = data.astype(np.float32)

    # Check for NaNs and fill if needed
    if data.isnull().values.any():
        print("Warning: Data contains NaN values. Filling with 0.")
        data = data.fillna(0)

    return data


def load_trained_model(model_path):
    # Load the saved hybrid model with custom layer
    try:
        model = load_model(model_path, custom_objects={
                           'ReduceMeanLayer': ReduceMeanLayer})  # Custom layer passed here
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

    # Convert DataFrame to NumPy array before reshaping
    input_data = input_data.values

    # Check the shape of input_data before reshaping
    print(f"Input data shape before reshaping: {input_data.shape}")

    # Ensure that input_data has the correct number of features (expected_features)
    if input_data.shape[1] > expected_features:
        print(f"Trimming input data to {expected_features} features")
        # Keep only the first N features
        input_data = input_data[:, :expected_features]

    # If input_data is 2D, add an extra dimension to make it 3D
    if len(input_data.shape) == 2:
        # Reshape to (samples, timesteps=expected_timesteps, features=expected_features)
        input_data = input_data.reshape(
            (input_data.shape[0], expected_timesteps, expected_features))

    # Check the reshaped input data shape
    print(f"Reshaped input data shape: {input_data.shape}")

    # Make predictions using the model
    predictions = model.predict(input_data)

    return predictions


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    data_file = os.path.join(current_dir, 'combined_real_time_data.csv')
    model_file = os.path.join(current_dir, 'models/Lumen2_real_time.keras')

    # Load data
    new_data = load_data_for_prediction(data_file)

    # Load the trained model
    model = load_trained_model(model_file)

    if model is not None:
        # Make predictions
        predictions = predict_with_model(model, new_data)
        print(predictions)
    else:
        print("Failed to load real-time model.")

# Save predictions to a .npy file
np.save(os.path.join(current_dir, 'models', 'predictions_real.npy'), predictions)

print(predictions)
