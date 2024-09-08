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
MODEL_NAME = 'Lumen2_finetuned.keras'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)


def save_lumen2_model(model):
    """
    Save the Lumen2 model.
    """
    try:
        logging.debug(f"Saving Lumen2 model to {MODEL_PATH}")
        save_model(model, MODEL_PATH)
        logging.debug(f"Lumen2 model saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save the Lumen2 model: {e}")


if __name__ == '__main__':
    # Use relative import for local files
    from load_lumen_2 import load_lumen2_model

    # Load the model
    model = load_lumen2_model()

    if model:
        # Example logic: train or fine-tune the model if necessary

        # Save the model
        save_lumen2_model(model)
