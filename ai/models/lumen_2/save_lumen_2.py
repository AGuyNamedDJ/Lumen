import os
import logging
from tensorflow.keras.models import save_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define the base directory and model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Paths for different models
HISTORICAL_MODEL_NAME = 'Lumen2_historical_finetuned.keras'
REAL_TIME_MODEL_NAME = 'Lumen2_real_time_finetuned.keras'
HISTORICAL_MODEL_PATH = os.path.join(MODEL_DIR, HISTORICAL_MODEL_NAME)
REAL_TIME_MODEL_PATH = os.path.join(MODEL_DIR, REAL_TIME_MODEL_NAME)


def save_lumen2_model(model, model_path):
    """
    Save the Lumen2 model.
    """
    try:
        logging.debug(f"Saving Lumen2 model to {model_path}")
        save_model(model, model_path)
        logging.debug(f"Lumen2 model saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save the Lumen2 model: {e}")


if __name__ == '__main__':
    # Use relative import for local files
    from load_lumen_2 import load_lumen2_model

    # Load historical and real-time models
    historical_model = load_lumen2_model(
        os.path.join(MODEL_DIR, 'Lumen2_historical.keras'))
    real_time_model = load_lumen2_model(
        os.path.join(MODEL_DIR, 'Lumen2_real_time.keras'))

    if historical_model:
        # Fine-tune the historical model here if needed
        # Example: historical_model.fit(train_data, train_labels, epochs=5)

        # Save the fine-tuned historical model
        save_lumen2_model(historical_model, HISTORICAL_MODEL_PATH)

    if real_time_model:
        # Fine-tune the real-time model here if needed
        # Example: real_time_model.fit(train_data, train_labels, epochs=5)

        # Save the fine-tuned real-time model
        save_lumen2_model(real_time_model, REAL_TIME_MODEL_PATH)
