import os
import logging
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# Load environment variables
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def load_lumen_model():
    model_path = os.path.join(os.path.dirname(__file__), 'trained_model.keras')
    try:
        model = load_model(model_path)
        logging.debug("Lumen model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading Lumen model: {e}")
        raise e


def load_preprocessed_data():
    data_path = os.path.join(os.path.dirname(
        __file__), '../../data/lumen_1/processed/fine_tune_data.json')
    try:
        with open(data_path, 'r') as file:
            data = json.load(file)
        logging.debug("Preprocessed data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading preprocessed data: {e}")
        raise e


if __name__ == "__main__":
    model = load_lumen_model()
    data = load_preprocessed_data()
    logging.debug(f"Loaded model: {model}")
    logging.debug(f"Loaded preprocessed data: {data}")
