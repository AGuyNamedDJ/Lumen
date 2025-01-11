import os
import sys
import logging
import numpy as np
import boto3
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, r2_score

##############################################################################
# 1) AWS S3 HELPERS
##############################################################################
def get_s3_client():
    """
    Creates a boto3 client pinned to region us-east-2 (or whichever region you use).
    Pulls credentials from environment variables.
    """
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name="us-east-2"
    )

def download_file_from_s3(s3_key: str, local_path: str):
    """
    Always remove any local file, then download from s3://<bucket>/<s3_key> into local_path.
    """
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "lumenaibucket")
    s3 = get_s3_client()

    if os.path.exists(local_path):
        logging.info(f"[download_file_from_s3] Removing existing => {local_path}")
        os.remove(local_path)

    logging.info(f"[download_file_from_s3] Downloading s3://{bucket_name}/{s3_key} → {local_path}")
    try:
        s3.download_file(bucket_name, s3_key, local_path)
        logging.info("[download_file_from_s3] Done.")
    except Exception as exc:
        logging.error(f"Error downloading s3://{bucket_name}/{s3_key}: {exc}")
        raise

##############################################################################
# 2) ENV + LOGGING SETUP
##############################################################################
load_dotenv()
logging.basicConfig(level=logging.INFO)

##############################################################################
# 3) LOCAL PATHS + S3 KEYS FOR EVALUATION
##############################################################################
script_dir = os.path.dirname(os.path.abspath(__file__))
TRAINED_DIR = os.path.join(script_dir, "trained")
os.makedirs(TRAINED_DIR, exist_ok=True)

S3_PREDICTIONS_TEST = "models/lumen_2/trained/predictions_test.npy"
S3_TRUE_VALUES_TEST = "models/lumen_2/trained/y_test.npy"

LOCAL_PREDICTIONS_TEST = os.path.join(TRAINED_DIR, "predictions_test.npy")
LOCAL_TRUE_VALUES_TEST = os.path.join(TRAINED_DIR, "y_test.npy")

##############################################################################
# 4) DOWNLOAD ANY NEEDED FILES
##############################################################################
def download_test_files():
    """
    Force-downloads predictions_test.npy and y_test.npy from S3 => local,
    overwriting local files.
    """
    download_file_from_s3(S3_PREDICTIONS_TEST, LOCAL_PREDICTIONS_TEST)
    download_file_from_s3(S3_TRUE_VALUES_TEST, LOCAL_TRUE_VALUES_TEST)

##############################################################################
# 5) EVALUATION LOGIC
##############################################################################
def evaluate_predictions(predictions_path, groundtruth_path, label="Test Set"):
    """
    Loads 'predictions' & 'y_true' from .npy files,
    checks shape, then prints MSE, RMSE, and R².
    """
    if not os.path.exists(predictions_path):
        logging.error(f"[{label}] Missing predictions => {predictions_path}")
        return
    if not os.path.exists(groundtruth_path):
        logging.error(f"[{label}] Missing ground-truth => {groundtruth_path}")
        return

    preds = np.load(predictions_path)
    y_true = np.load(groundtruth_path)

    logging.info(f"[{label}] preds shape={preds.shape}, y_true shape={y_true.shape}")
    if len(preds) != len(y_true):
        logging.error(f"[{label}] Mismatch: preds={len(preds)}, y_true={len(y_true)}")
        return

    preds_flat = preds.ravel()
    y_true_flat = y_true.ravel()

    mse_val  = mean_squared_error(y_true_flat, preds_flat)
    rmse_val = np.sqrt(mse_val)
    r2_val   = r2_score(y_true_flat, preds_flat)

    logging.info(f"[{label}] => MSE: {mse_val:.6f}, RMSE: {rmse_val:.6f}, R²: {r2_val:.6f}")

##############################################################################
# 6) MAIN
##############################################################################
def main():
    logging.info("=== evaluate_predictions => Start ===")

    download_test_files()

    evaluate_predictions(LOCAL_PREDICTIONS_TEST, LOCAL_TRUE_VALUES_TEST, label="Test Set")

    logging.info("=== evaluate_predictions => Done ===")

if __name__ == "__main__":
    main()