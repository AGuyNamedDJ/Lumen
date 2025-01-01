import os
import sys
import logging
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .definitions_lumen_2 import ReduceMeanLayer
try:
    from ..utils.aws_s3_utils import download_file_from_s3
except ImportError:
    download_file_from_s3 = None
    logging.warning("No download_file_from_s3 found; local usage only.")

load_dotenv()
logging.basicConfig(level=logging.INFO)

LOCAL_MODEL_FILENAME = "Lumen2_from_s3.keras"
S3_KEY = "models/lumen_2/trained/Lumen2.keras"

def get_s3_client():
    import boto3
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

def download_file_if_needed(s3_key, local_path, force_download=True):
    if not download_file_from_s3 or not callable(download_file_from_s3):
        logging.warning("No S3 download function, skipping S3 pull.")
        return

    if force_download and os.path.exists(local_path):
        logging.info(f"Removing existing local file => {local_path}")
        os.remove(local_path)

    if not os.path.exists(local_path):
        bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
        logging.info(f"Downloading model s3://{bucket_name}/{s3_key} → {local_path}")
        download_file_from_s3(s3_key, local_path)
        logging.info("Download complete.")
    else:
        logging.info(f"Local file {local_path} already present, skipping S3 download.")

def load_lumen2_model_from_s3(s3_key=S3_KEY, local_filename=LOCAL_MODEL_FILENAME, force_download=True):
    download_file_if_needed(s3_key, local_filename, force_download)
    if not os.path.exists(local_filename):
        logging.error(f"File {local_filename} not found; cannot load model.")
        return None

    logging.info(f"Loading model from {local_filename}...")
    try:
        model = load_model(local_filename, custom_objects={'ReduceMeanLayer': ReduceMeanLayer})
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {local_filename}: {e}")
        return None

if __name__ == "__main__":
    model = load_lumen2_model_from_s3(force_download=True)
    if model:
        logging.info("Printing model summary now...")
        model.summary()
    else:
        logging.error("Failed to load the Lumen2 model from S3/local.")