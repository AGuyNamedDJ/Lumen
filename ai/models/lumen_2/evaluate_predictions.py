import os
import sys
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv
import boto3

def get_s3_client():
    """Create an S3 client pinned to region us-east-2 (update if needed)."""
    return boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = "us-east-2"
    )

def download_file_from_s3(s3_key: str, local_path: str):
    """Download a file from S3 to local filesystem."""
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "lumenaibucket")
    s3 = get_s3_client()
    try:
        logging.info(f"Downloading s3://{bucket_name}/{s3_key} → {local_path}")
        s3.download_file(bucket_name, s3_key, local_path)
        logging.info(f"Downloaded s3://{bucket_name}/{s3_key} → {local_path}")
    except Exception as e:
        logging.error(f"Error downloading {s3_key} from S3: {e}")
        raise

# ------------------------------------------------------------------------
# Environment + logging
# ------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------
# Local paths + S3 references for real-time predictions and y_test
# ------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
TRAINED_DIR = os.path.join(script_dir, "trained")

# S3 keys for real-time
S3_PREDICTIONS_REAL = "models/lumen_2/trained/predictions_real.npy"
S3_TRUE_VALUES_REAL = "models/lumen_2/trained/y_test_real.npy"

# Local files to store them
PREDICTIONS_REAL_PATH = os.path.join(TRAINED_DIR, "predictions_real.npy")
TRUE_REAL_PATH        = os.path.join(TRAINED_DIR, "y_test_real.npy")

def maybe_download_real_time_files():
    """Download real-time predictions + y_test arrays from S3 if not present."""
    os.makedirs(TRAINED_DIR, exist_ok=True)

    # Predictions
    if not os.path.exists(PREDICTIONS_REAL_PATH):
        try:
            download_file_from_s3(S3_PREDICTIONS_REAL, PREDICTIONS_REAL_PATH)
        except Exception as e:
            logging.warning(f"Could not fetch real-time predictions from S3: {e}")
    else:
        logging.info(f"Real-time predictions already exist locally: {PREDICTIONS_REAL_PATH}")

    # True values
    if not os.path.exists(TRUE_REAL_PATH):
        try:
            download_file_from_s3(S3_TRUE_VALUES_REAL, TRUE_REAL_PATH)
        except Exception as e:
            logging.warning(f"Could not fetch real-time true-values from S3: {e}")
    else:
        logging.info(f"Real-time true-values already exist locally: {TRUE_REAL_PATH}")

def evaluate_predictions(predictions_path, true_values_path, label="Real-Time Model"):
    """
    Loads predictions & true arrays from disk, checks shape & prints MSE, RMSE, and R².
    """
    # Ensure files exist
    if not os.path.exists(predictions_path):
        logging.error(f"[{label}] missing predictions file: {predictions_path}")
        return
    if not os.path.exists(true_values_path):
        logging.error(f"[{label}] missing true-values file: {true_values_path}")
        return

    # Load arrays
    preds = np.load(predictions_path)
    y_true = np.load(true_values_path)

    logging.info(f"[{label}] predictions shape={preds.shape}, true shape={y_true.shape}")

    # Sanity check
    if preds.shape[0] != y_true.shape[0]:
        logging.error(f"[{label}] mismatch: preds={len(preds)}, true={len(y_true)}")
        return

    # Flatten if needed
    preds = preds.flatten()
    y_true = y_true.flatten()

    # Calculate metrics
    mse_val  = mean_squared_error(y_true, preds)
    rmse_val = np.sqrt(mse_val)
    r2_val   = r2_score(y_true, preds)

    logging.info(f"[{label}] MSE={mse_val:.6f}, RMSE={rmse_val:.6f}, R²={r2_val:.6f}")

def main():
    # 1) Possibly download real-time predictions + real-time y_test from S3
    maybe_download_real_time_files()

    # 2) Evaluate real-time model predictions
    evaluate_predictions(PREDICTIONS_REAL_PATH, TRUE_REAL_PATH, "Real-Time Model")

if __name__ == "__main__":
    main()