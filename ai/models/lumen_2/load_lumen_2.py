import os
import sys
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
MODEL_DIR = os.path.join(BASE_DIR)  # Use BASE_DIR directly

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


def get_input_shape(model):
    if isinstance(model.input, list):
        # If the model has multiple input layers, process each one
        return [inp.shape for inp in model.input]
    else:
        # If it's a single input layer
        return model.input.shape


if __name__ == '__main__':
    # Load and print summaries for each model
    historical_model = load_lumen2_model(HISTORICAL_MODEL_PATH)
    real_time_model = load_lumen2_model(REAL_TIME_MODEL_PATH)
    main_model = load_lumen2_model(MAIN_MODEL_PATH)

    if historical_model:
        logging.info("Historical Model Summary:")
        historical_model.summary()
        input_shape_hist = get_input_shape(historical_model)  # Get input shape
        logging.info(f"Expected input shape for Historical Model: {
                     input_shape_hist}")

    if real_time_model:
        logging.info("Real-Time Model Summary:")
        real_time_model.summary()
        input_shape_real = get_input_shape(real_time_model)  # Get input shape
        logging.info(
            f"Expected input shape for Real-Time Model: {input_shape_real}")

    if main_model:
        logging.info("Main Lumen2 Model Summary:")
        main_model.summary()
        input_shape_main = get_input_shape(main_model)  # Get input shape
        logging.info(f"Expected input shape for Main Lumen2 Model: {
                     input_shape_main}")
