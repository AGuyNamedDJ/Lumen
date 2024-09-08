import os
import logging
from tensorflow.keras.models import load_model
from definitions_lumen_2 import ReduceMeanLayer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define the base directory and model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Paths for different models
HISTORICAL_MODEL_PATH = os.path.join(MODEL_DIR, 'Lumen2_historical.keras')
REAL_TIME_MODEL_PATH = os.path.join(MODEL_DIR, 'Lumen2_real_time.keras')
MAIN_MODEL_PATH = os.path.join(MODEL_DIR, 'Lumen2.keras')


def load_lumen2_model(model_path):
    try:
        logging.debug(f"Loading model from {model_path}")
        model = load_model(model_path, custom_objects={
                           'ReduceMeanLayer': ReduceMeanLayer})
        logging.debug(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        return None


if __name__ == '__main__':
    # Load and print summaries for each model
    historical_model = load_lumen2_model(HISTORICAL_MODEL_PATH)
    real_time_model = load_lumen2_model(REAL_TIME_MODEL_PATH)
    main_model = load_lumen2_model(MAIN_MODEL_PATH)

    if historical_model:
        logging.info("Historical Model Summary:")
        historical_model.summary()
    if real_time_model:
        logging.info("Real-Time Model Summary:")
        real_time_model.summary()
    if main_model:
        logging.info("Main Lumen2 Model Summary:")
        main_model.summary()
