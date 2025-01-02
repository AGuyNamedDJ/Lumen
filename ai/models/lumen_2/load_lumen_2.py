import os
import sys
import logging
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from models.lumen_2.definitions_lumen_2 import ReduceMeanLayer
except ImportError:
    logging.error("Could not import ReduceMeanLayer. Check your import paths.")
    ReduceMeanLayer = None

try:
    from ai.utils.aws_s3_utils import download_file_from_s3
except ImportError:
    download_file_from_s3 = None
    logging.warning("No 'download_file_from_s3' found; will do local-file-only model loading.")

# 3) ENV + LOGGING
load_dotenv()
logging.basicConfig(level=logging.INFO)

LOCAL_MODEL_FILENAME = "Lumen2_from_s3.keras"
S3_KEY = "models/lumen_2/trained/Lumen2.keras"

def download_file_if_needed(s3_key, local_path, force_download=True):
    """
    If S3 download is available, optionally remove any existing local file,
    then download from s3://<bucket>/<s3_key>.
    """
    if not download_file_from_s3 or not callable(download_file_from_s3):
        logging.warning("No S3 download function => skipping S3 pull.")
        return

    if force_download and os.path.exists(local_path):
        logging.info(f"[download_file_if_needed] Removing existing {local_path}")
        os.remove(local_path)

    if not os.path.exists(local_path):
        bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
        logging.info(f"Downloading model s3://{bucket_name}/{s3_key} → {local_path}")
        try:
            download_file_from_s3(s3_key, local_path)
            logging.info("Download complete.")
        except Exception as exc:
            logging.error(f"Error downloading {s3_key} => {exc}")
    else:
        logging.info(f"Local model file {local_path} already exists => skipping download.")

def load_lumen2_model_from_s3(
    s3_key: str = S3_KEY,
    local_filename: str = LOCAL_MODEL_FILENAME,
    force_download: bool = True
):
    """
    If `force_download=True`, remove any local .keras file, then pull from S3.
    Finally, load the model with custom layer `ReduceMeanLayer`.
    """
    download_file_if_needed(s3_key, local_filename, force_download)
    if not os.path.exists(local_filename):
        logging.error(f"Local file {local_filename} not found => cannot load model.")
        return None

    logging.info(f"Loading Lumen2 model from {local_filename} ...")
    try:
        model = load_model(local_filename, custom_objects={"ReduceMeanLayer": ReduceMeanLayer})
        logging.info("Model loaded successfully.")
        return model
    except Exception as exc:
        logging.error(f"Error loading model from {local_filename}: {exc}")
        return None

if __name__ == "__main__":
    model = load_lumen2_model_from_s3(
        s3_key=S3_KEY,
        local_filename=LOCAL_MODEL_FILENAME,
        force_download=True
    )
    if model:
        logging.info("Model loaded. Printing summary:")
        model.summary()
    else:
        logging.error("Failed to load the Lumen2 model from S3 or local disk.")