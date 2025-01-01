import os
import sys
import logging
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

# Local definition for the custom layer
from definitions_lumen_2 import ReduceMeanLayer

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

load_dotenv()
logging.basicConfig(level=logging.INFO)

try:
    from ai.utils.aws_s3_utils import download_file_from_s3
except ImportError:
    download_file_from_s3 = None
    logging.warning(
        "Could not import 'download_file_from_s3' from ai.utils.aws_s3_utils. "
        "Falling back to local-only model usage."
    )

LOCAL_MODEL_FILENAME = "Lumen2_from_s3.keras"
S3_KEY = "models/lumen_2/trained/Lumen2.keras"

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
    """
    # 1) Check if we have an S3 download function
    if not download_file_from_s3 or not callable(download_file_from_s3):
        logging.warning(
            "download_file_from_s3 is unavailable. We'll try to load from local file only."
        )
        if os.path.exists(local_filename):
            logging.info(f"Local file {local_filename} found. Attempting load...")
        else:
            logging.error("No local model file found => cannot load model.")
            return None
    else:
        # 2) If force_download=True, remove any existing local file to ensure fresh S3 pull
        if force_download and os.path.exists(local_filename):
            logging.info(f"[load_lumen2_model_from_s3] Removing existing local file => {local_filename}")
            try:
                os.remove(local_filename)
            except OSError as e:
                logging.warning(f"Could not remove local file {local_filename}: {e}")

        # If the file doesn't exist or was removed, we download from S3
        if not os.path.exists(local_filename):
            logging.info(f"Downloading model from s3://<bucket>/{s3_key} → {local_filename}")
            try:
                download_file_from_s3(s3_key, local_filename)
                logging.info("Download completed.")
            except Exception as e:
                logging.error(f"Failed to download {s3_key} from S3: {e}")
                return None
        else:
            logging.info(
                f"Model file {local_filename} already exists locally. "
                "Skipping S3 download (force_download=False)."
            )

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

    model = load_lumen2_model_from_s3(
        s3_key=S3_KEY,
        local_filename=LOCAL_MODEL_FILENAME,
        force_download=True 
    )
    if model:
        logging.info("Printing model summary now...")
        model.summary()
    else:
        logging.error("Failed to load the Lumen2 model from S3 or local file.")