# Import the necessary modules
import os
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
import logging

# Load environment variables
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the LSTM model


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'trained_model.keras')
    model = tf.keras.models.load_model(model_path)
    logging.info("Model loaded successfully.")
    return model

# Prepare data for prediction


def prepare_data_for_prediction(data):
    try:
        df = pd.DataFrame([data])
        logging.info(f"Data prepared for prediction: {df}")
        return df
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        return None


def get_predictions(model, data):
    try:
        features = data[['open', 'high', 'low',
                         'close', 'volume', 'ema_10', 'ema_50']]
        predictions = model.predict(features)
        return predictions[-1]
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        return None


# Load the model once and keep it ready for predictions
model = load_model()


def predict_next_day_close(data):
    if not model:
        return "Model is not loaded"
    prepared_data = prepare_data_for_prediction(data)
    if prepared_data is None:
        return "Error in preparing data"
    prediction = get_predictions(model, prepared_data)
    if prediction is None:
        return "Error in making predictions"
    return prediction[0]
