import os
import sys
import logging
import numpy as np
import boto3
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

try:
    from definitions_lumen_2 import ReduceMeanLayer
except ImportError:
    class ReduceMeanLayer:
        pass

# ------------------------------------------------------------------------
# ENV + LOGGING
# ------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------
# BOTO3 S3 UTILS
# ------------------------------------------------------------------------
def get_s3_client():
    """Creates a boto3 S3 client from environment variables (region/key/secret)."""
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

def download_file_from_s3(s3_key: str, local_path: str):
    """
    Downloads a file from s3://<bucket>/<s3_key> → local_path.
    Overwrites local_path if it already exists.
    """
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
    s3 = get_s3_client()
    if os.path.exists(local_path):
        logging.info(f"[download_file_from_s3] Removing existing local file => {local_path}")
        os.remove(local_path)
    logging.info(f"[download_file_from_s3] Downloading s3://{bucket_name}/{s3_key} → {local_path}")
    s3.download_file(bucket_name, s3_key, local_path)
    logging.info("[download_file_from_s3] Done.")

def upload_file_to_s3(local_path: str, s3_key: str):
    """
    Uploads a local file to s3://<bucket>/<s3_key>.
    """
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
    s3 = get_s3_client()
    logging.info(f"[upload_file_to_s3] Uploading {local_path} → s3://{bucket_name}/{s3_key}")
    s3.upload_file(local_path, bucket_name, s3_key)
    logging.info("[upload_file_to_s3] Done.")

# ------------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

TRAINED_DIR = os.path.join(script_dir, "trained")
os.makedirs(TRAINED_DIR, exist_ok=True)

# Default model path
MODEL_FILENAME = "Lumen2.keras"
MODEL_PATH     = os.path.join(TRAINED_DIR, MODEL_FILENAME)

TEST_X_S3_KEY  = "data/lumen2/featured/sequences/spx_spy_vix_test_X_3D_part0.npy"
TEST_Y_S3_KEY  = "data/lumen2/featured/sequences/spx_spy_vix_test_Y_3D_part0.npy"
LOCAL_TEST_X   = os.path.join(TRAINED_DIR, "spx_spy_vix_test_X_3D_part0.npy")
LOCAL_TEST_Y   = os.path.join(TRAINED_DIR, "spx_spy_vix_test_Y_3D_part0.npy")

PREDICTIONS_NPY = os.path.join(TRAINED_DIR, "predictions_test.npy")
PREDICTIONS_S3  = "models/lumen_2/trained/predictions_test.npy" 

# ------------------------------------------------------------------------
# 1) Download test sequences + model if needed
# ------------------------------------------------------------------------
def download_test_data():
    """Download test X, Y from S3 => local, so we can do predictions."""
    logging.info("[download_test_data] Ensuring test data is local.")
    download_file_from_s3(TEST_X_S3_KEY, LOCAL_TEST_X)
    download_file_from_s3(TEST_Y_S3_KEY, LOCAL_TEST_Y)

def maybe_download_model():
    """If the model isn't local, download it from S3, or skip if local file is present."""
    if os.path.exists(MODEL_PATH):
        logging.info(f"Model file already local: {MODEL_PATH}")
        return
    # If you store the model on S3, define S3 key here:
    model_s3_key = "models/lumen_2/trained/Lumen2.keras"
    download_file_from_s3(model_s3_key, MODEL_PATH)

# ------------------------------------------------------------------------
# 2) Load Model + Validate shapes
# ------------------------------------------------------------------------
def load_lumen2_model():
    """Loads Lumen2.keras with a custom layer if needed."""
    if not os.path.exists(MODEL_PATH):
        logging.error(f"No model at {MODEL_PATH}.")
        return None
    try:
        model = load_model(MODEL_PATH, custom_objects={'ReduceMeanLayer': ReduceMeanLayer})
        logging.info(f"Loaded model from {MODEL_PATH}")
        return model
    except Exception as exc:
        logging.error(f"Error loading model: {exc}")
        return None

# ------------------------------------------------------------------------
# 3) Prediction Flow
# ------------------------------------------------------------------------
def main():
    logging.info("[predict_lumen_2] Starting prediction on test set.")
    os.makedirs(TRAINED_DIR, exist_ok=True)

    # Download test npy files + model
    download_test_data()
    maybe_download_model()

    # Load model
    model = load_lumen2_model()
    if model is None:
        logging.error("Failed to load model => abort.")
        return

    # Load test arrays
    if not os.path.exists(LOCAL_TEST_X):
        logging.error(f"Missing test X => {LOCAL_TEST_X}. Exiting.")
        return
    X_test = np.load(LOCAL_TEST_X)
    logging.info(f"Loaded X_test => shape={X_test.shape}")

    if os.path.exists(LOCAL_TEST_Y):
        Y_test = np.load(LOCAL_TEST_Y)
        logging.info(f"Loaded Y_test => shape={Y_test.shape}")
    else:
        Y_test = None
        logging.warning("No local Y test found => skipping error metrics.")

    # Validate feature shape
    seq_len_model = model.input_shape[1]  # e.g., 60
    feat_model    = model.input_shape[2]  # e.g., 31
    seq_len_data  = X_test.shape[1]
    feat_data     = X_test.shape[2]

    if seq_len_model != seq_len_data or feat_model != feat_data:
        logging.warning(f"Model expects (None, {seq_len_model}, {feat_model}), got {X_test.shape}.")

    # Convert to float32 and predict
    X_test = X_test.astype("float32")
    logging.info("Predicting on test data...")
    preds = model.predict(X_test, verbose=0)
    logging.info(f"Predictions shape: {preds.shape}")

    # Save predictions
    np.save(PREDICTIONS_NPY, preds)
    logging.info(f"Saved predictions => {PREDICTIONS_NPY}")

    # Optional: upload predictions to S3
    try:
        upload_file_to_s3(PREDICTIONS_NPY, PREDICTIONS_S3)
        logging.info(f"Uploaded predictions to s3://<bucket>/{PREDICTIONS_S3}")
    except Exception as exc:
        logging.warning(f"Could not upload predictions to S3: {exc}")

    logging.info("[predict_lumen_2] Done.")

if __name__ == "__main__":
    main()