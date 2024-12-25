import os
import sys
import logging
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from definitions_lumen_2 import ReduceMeanLayer

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

try:
    from ai.utils.aws_s3_utils import download_file_from_s3
except ImportError:
    logging.warning("Could not import 'download_file_from_s3' from ai.utils.aws_s3_utils. "
                    "Check your import paths!")

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Default local filename
LOCAL_MODEL_FILENAME = "Lumen2_from_s3.keras"
# Default S3 key (path in your S3 bucket).
# Example: "models/lumen_2/trained/Lumen2.keras"
S3_KEY = "models/lumen_2/trained/Lumen2.keras"

def load_lumen2_model_from_s3(s3_key=S3_KEY, local_filename=LOCAL_MODEL_FILENAME):
    """
    Downloads the trained Lumen2 model from S3, saves it locally, then loads it
    with custom layer support.

    :param s3_key: The path in S3 where your model file is stored.
    :param local_filename: Where to save the file locally.
    :return: A Keras model object or None if something fails.
    """
    # 1) Ensure we have a function to download from S3
    if 'download_file_from_s3' not in globals() or not callable(download_file_from_s3):
        logging.error("download_file_from_s3 is not available. Cannot pull model from S3.")
        return None

    # 2) Download from S3 → local
    logging.info(f"Attempting to download model from S3 key: {s3_key} → {local_filename}")
    try:
        download_file_from_s3(s3_key, local_filename)
    except Exception as e:
        logging.error(f"Failed to download from S3: {e}")
        return None

    if not os.path.exists(local_filename):
        logging.error(f"File {local_filename} not found after download. Cannot load model.")
        return None

    # 3) Load with custom objects
    logging.info(f"Loading model from local file: {local_filename}")
    try:
        model = load_model(local_filename, custom_objects={'ReduceMeanLayer': ReduceMeanLayer})
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {local_filename}: {e}")
        return None


if __name__ == "__main__":
    """
    Example usage: 
      python load_lumen2.py
    This will:
      - Download the model from s3://<bucket>/{S3_KEY} to local {LOCAL_MODEL_FILENAME}.
      - Load it with custom layer 'ReduceMeanLayer'.
      - Print a summary if loaded successfully.
    """
    model = load_lumen2_model_from_s3(
        s3_key=S3_KEY,
        local_filename=LOCAL_MODEL_FILENAME
    )
    if model:
        logging.info("Printing model summary:")
        model.summary()
    else:
        logging.error("Failed to load the model from S3.")