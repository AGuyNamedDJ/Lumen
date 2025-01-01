import os
import sys
import logging
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, r2_score
import boto3

##############################################################################
# 1) AWS S3 HELPERS
##############################################################################
def get_s3_client():
    """Create an S3 client pinned to region us-east-2 (update if needed)."""
    return boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = "us-east-2"
    )

def download_file_from_s3(s3_key: str, local_path: str):
    """
    Downloads a file from s3://<bucket>/<s3_key> to the local path.
    Replaces the local file if it exists (to ensure fresh data).
    """
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "lumenaibucket")
    s3 = get_s3_client()

    if os.path.exists(local_path):
        logging.info(f"[download_file_from_s3] Removing existing local file => {local_path}")
        os.remove(local_path)

    logging.info(f"[download_file_from_s3] Downloading s3://{bucket_name}/{s3_key} → {local_path}")
    try:
        s3.download_file(bucket_name, s3_key, local_path)
        logging.info("[download_file_from_s3] Done.")
    except Exception as e:
        logging.error(f"Error downloading {s3_key} from S3: {e}")
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

# S3 keys for real-time predictions & ground truth
S3_PREDICTIONS_REAL = "models/lumen_2/trained/predictions_real.npy"
S3_TRUE_VALUES_REAL = "models/lumen_2/trained/y_test_real.npy"

# Local file paths for real-time predictions & ground truth
LOCAL_PREDICTIONS_REAL = os.path.join(TRAINED_DIR, "predictions_real.npy")
LOCAL_TRUE_VALUES_REAL = os.path.join(TRAINED_DIR, "y_test_real.npy")

##############################################################################
# 4) DOWNLOAD ANY NEEDED FILES
##############################################################################
def maybe_download_real_time_files():
    """
    Downloads predictions and y_test arrays for real-time evaluation if not present locally.
    """
    # Predictions
    if not os.path.exists(LOCAL_PREDICTIONS_REAL):
        logging.info("[maybe_download_real_time_files] Predictions not found locally; downloading...")
        try:
            download_file_from_s3(S3_PREDICTIONS_REAL, LOCAL_PREDICTIONS_REAL)
        except Exception as e:
            logging.warning(f"Could not fetch real-time predictions from S3: {e}")
    else:
        logging.info(f"Real-time predictions already exist locally: {LOCAL_PREDICTIONS_REAL}")

    # True values
    if not os.path.exists(LOCAL_TRUE_VALUES_REAL):
        logging.info("[maybe_download_real_time_files] True-values not found locally; downloading...")
        try:
            download_file_from_s3(S3_TRUE_VALUES_REAL, LOCAL_TRUE_VALUES_REAL)
        except Exception as e:
            logging.warning(f"Could not fetch real-time true-values from S3: {e}")
    else:
        logging.info(f"Real-time true-values already exist locally: {LOCAL_TRUE_VALUES_REAL}")

##############################################################################
# 5) EVALUATION LOGIC
##############################################################################
def evaluate_predictions(predictions_path, true_values_path, label="Real-Time"):
    """
    Loads preds & ground truth from .npy files, then computes MSE, RMSE, R².
    """
    if not os.path.exists(predictions_path):
        logging.error(f"[{label}] Missing predictions file => {predictions_path}")
        return

    if not os.path.exists(true_values_path):
        logging.error(f"[{label}] Missing true-values file => {true_values_path}")
        return

    preds = np.load(predictions_path)
    y_true = np.load(true_values_path)

    logging.info(f"[{label}] preds shape={preds.shape}, y_true shape={y_true.shape}")

    if preds.shape[0] != y_true.shape[0]:
        logging.error(f"[{label}] Mismatch: preds={preds.shape[0]}, y_true={y_true.shape[0]}")
        return

    # Flatten if needed
    preds_flat = preds.flatten()
    y_true_flat = y_true.flatten()

    mse_val = mean_squared_error(y_true_flat, preds_flat)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(y_true_flat, preds_flat)

    logging.info(f"[{label}] MSE={mse_val:.6f}, RMSE={rmse_val:.6f}, R²={r2_val:.6f}")

##############################################################################
# 6) MAIN
##############################################################################
def main():
    logging.info("=== evaluate_lumen_2: Start ===")
    maybe_download_real_time_files()

    evaluate_predictions(LOCAL_PREDICTIONS_REAL, LOCAL_TRUE_VALUES_REAL, "Real-Time")

    logging.info("=== evaluate_lumen_2: Done ===")

if __name__ == "__main__":
    main()