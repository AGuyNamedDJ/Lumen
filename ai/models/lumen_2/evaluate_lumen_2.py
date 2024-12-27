import os
import sys
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model

import boto3

def get_s3_client():
    """Hard-code region to us-east-2 (or whatever region your S3 bucket is in)."""
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name="us-east-2"
    )

def download_file_from_s3(s3_key: str, local_path: str):
    """Download a file from your S3 bucket -> local filesystem."""
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "lumenaibucket")
    s3 = get_s3_client()
    try:
        s3.download_file(bucket_name, s3_key, local_path)
        logging.info(f"Downloaded s3://{bucket_name}/{s3_key} → {local_path}")
    except Exception as e:
        logging.error(f"Error downloading {s3_key} from S3: {e}")
        raise

# --------------------------------------------------------------------------------
# Evaluate script focusing on Real-Time test arrays only
# --------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

logging.basicConfig(level=logging.INFO)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAINED_DIR = os.path.join(BASE_DIR, "trained")

MODEL_FILENAME = "Lumen2.keras"
MODEL_PATH     = os.path.join(TRAINED_DIR, MODEL_FILENAME)

# The S3 keys & local paths for real-time test data:
S3_X_TEST_REAL = "models/lumen_2/trained/X_test_real.npy"
S3_Y_TEST_REAL = "models/lumen_2/trained/y_test_real.npy"

LOCAL_X_TEST_REAL = os.path.join(TRAINED_DIR, "X_test_real.npy")
LOCAL_Y_TEST_REAL = os.path.join(TRAINED_DIR, "y_test_real.npy")

def maybe_download_test_files():
    """Only download the real-time arrays from S3."""
    os.makedirs(TRAINED_DIR, exist_ok=True)

    def attempt(s3_key, local_path):
        logging.info(f"Downloading {s3_key} → {local_path}")
        try:
            download_file_from_s3(s3_key, local_path)
        except Exception as exc:
            logging.error(f"Error downloading {s3_key} from S3: {exc}")

    attempt(S3_X_TEST_REAL, LOCAL_X_TEST_REAL)
    attempt(S3_Y_TEST_REAL, LOCAL_Y_TEST_REAL)

def load_npy(npy_path):
    """Helper to load a single .npy file if it exists; otherwise None."""
    if not os.path.exists(npy_path):
        return None
    return np.load(npy_path)

def evaluate_realtime(model, X_test, y_test):
    """Evaluate the model on real-time test data."""
    if X_test is None or y_test is None:
        logging.error("No real-time test data loaded. Skipping evaluation.")
        return

    if len(X_test) != len(y_test):
        logging.error(f"Mismatch: X={X_test.shape[0]}, y={y_test.shape[0]}")
        return

    # Check feature count
    expected_feats = model.input_shape[-1]
    if X_test.shape[-1] != expected_feats:
        diff = expected_feats - X_test.shape[-1]
        if diff > 0:
            logging.warning(f"Padding X by {diff} features with zeros.")
            X_test = np.pad(
                X_test, ((0,0),(0,0),(0,diff)), mode="constant", constant_values=0
            )
        else:
            logging.error(f"Feature mismatch: model wants {expected_feats}, got {X_test.shape[-1]}")
            return

    # Evaluate
    X_test = X_test.astype("float32")
    y_test = y_test.astype("float32")

    predictions = model.predict(X_test, verbose=0)
    mse_val = mean_squared_error(y_test, predictions)
    rmse_val = np.sqrt(mse_val)
    r2_val   = r2_score(y_test, predictions)

    logging.info("=== REAL-TIME EVALUATION ===")
    logging.info(f"MSE:  {mse_val:.6f}")
    logging.info(f"RMSE: {rmse_val:.6f}")
    logging.info(f"R²:   {r2_val:.6f}")

def main():
    # 1) Download real-time test arrays
    maybe_download_test_files()

    # 2) Check local model
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model not found at {MODEL_PATH}. Exiting.")
        return

    # 3) Load model
    model = load_model(MODEL_PATH)
    logging.info(f"Loaded model from {MODEL_PATH}")

    # 4) Load test arrays
    X_test_real = load_npy(LOCAL_X_TEST_REAL)
    y_test_real = load_npy(LOCAL_Y_TEST_REAL)
    logging.info(f"X_test_real shape: {None if X_test_real is None else X_test_real.shape}")
    logging.info(f"y_test_real shape: {None if y_test_real is None else y_test_real.shape}")

    # 5) Evaluate real-time
    evaluate_realtime(model, X_test_real, y_test_real)

if __name__ == "__main__":
    main()