import os
import sys
import logging
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

# Local definition for custom layer
from definitions_lumen_2 import ReduceMeanLayer

##############################################################################
# 1) PATHS & ENV
##############################################################################
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

load_dotenv()
logging.basicConfig(level=logging.INFO)

##############################################################################
# 2) OPTIONAL S3 UTILS
##############################################################################
try:
    from ai.utils.aws_s3_utils import download_file_from_s3
except ImportError:
    download_file_from_s3 = None
    logging.warning("Could not import 'download_file_from_s3' from ai.utils.aws_s3_utils. "
                    "Falling back to local-only model usage.")

##############################################################################
# DEFAULT CONSTANTS
##############################################################################
LOCAL_MODEL_FILENAME = "Lumen2_from_s3.keras"
S3_KEY = "models/lumen_2/trained/Lumen2.keras"  # Adjust if your S3 path differs

##############################################################################
# 3) MODEL-LOADING FUNCTION
##############################################################################
def load_lumen2_model_from_s3(
    s3_key: str = S3_KEY,
    local_filename: str = LOCAL_MODEL_FILENAME,
    force_download: bool = True
):
    """
    Downloads the trained Lumen2 model from S3 (if possible), then loads it
    with your custom 'ReduceMeanLayer'.

    :param s3_key: Path in your S3 bucket where the model file (.keras) is stored.
    :param local_filename: Local path/filename to store the downloaded model.
    :param force_download: If True, always re-download from S3. If False and
                           local_filename exists, skip the download step.
    :return: Keras model object, or None if there's an issue.
    """
    # 1) Check S3 availability
    if not download_file_from_s3 or not callable(download_file_from_s3):
        logging.warning(
            "download_file_from_s3 is unavailable. Trying to load local model only."
        )
        if os.path.exists(local_filename):
            logging.info(f"Local file {local_filename} found. Attempting load...")
        else:
            logging.error("No local model file found => cannot load model.")
            return None
    else:
        # 2) Possibly skip re-download if local file is already there
        if not force_download and os.path.exists(local_filename):
            logging.info(f"Model file {local_filename} already exists locally. "
                         "Skipping S3 download (force_download=False).")
        else:
            logging.info(f"Downloading model from s3_key={s3_key} → {local_filename}")
            try:
                download_file_from_s3(s3_key, local_filename)
                logging.info("Download completed.")
            except Exception as e:
                logging.error(f"Failed to download {s3_key} from S3: {e}")
                return None

    # 3) Attempt to load the local model file
    if not os.path.exists(local_filename):
        logging.error(f"File {local_filename} not found; cannot load model.")
        return None

    logging.info(f"Loading Lumen2 model from local file: {local_filename}")
    try:
        model = load_model(
            local_filename,
            custom_objects={'ReduceMeanLayer': ReduceMeanLayer}
        )
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {local_filename}: {e}")
        return None

##############################################################################
# 4) MAIN (SAMPLE USAGE)
##############################################################################
if __name__ == "__main__":
    """
    Example usage:
      python load_lumen2.py
    This will:
      1) Attempt to download s3://<your-bucket>/{S3_KEY} => local .keras file
      2) Load it with your custom layer 'ReduceMeanLayer'
      3) Print a summary if loaded successfully
    """
    # If you want to skip download if the file already exists locally, set force_download=False
    model = load_lumen2_model_from_s3(S3_KEY, LOCAL_MODEL_FILENAME, force_download=True)
    if model:
        logging.info("Printing model summary.")
        model.summary()
    else:
        logging.error("Failed to load the Lumen2 model from S3 or local file.")