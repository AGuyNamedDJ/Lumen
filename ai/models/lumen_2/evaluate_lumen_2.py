import os
import sys
import logging
import numpy as np
import joblib

from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
import boto3

##############################################################################
# 1) ENV + LOGGING
##############################################################################
load_dotenv()
logging.basicConfig(level=logging.INFO)

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

##############################################################################
# 2) AWS S3 UTILS
##############################################################################
def get_s3_client():
    """Create a boto3 S3 client from environment variables."""
    return boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = "us-east-2"
    )

def download_file_from_s3(s3_key: str, local_path: str):
    """
    Always force-download from s3://<bucket>/<s3_key> to local_path,
    removing local_path if it already exists.
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
    except Exception as e:
        logging.error(f"Error downloading {s3_key} from s3://{bucket_name}: {e}")
        raise

##############################################################################
# 3) PATHS
##############################################################################
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAINED_DIR = os.path.join(BASE_DIR, "trained")
os.makedirs(TRAINED_DIR, exist_ok=True)

MODEL_FILE        = "Lumen2.keras"
MODEL_LOCAL       = os.path.join(TRAINED_DIR, MODEL_FILE)
MODEL_S3_KEY      = "models/lumen_2/trained/Lumen2.keras"

X_TEST_S3_KEY     = "models/lumen_2/trained/X_test.npy"
Y_TEST_S3_KEY     = "models/lumen_2/trained/y_test.npy"

LOCAL_X_TEST      = os.path.join(TRAINED_DIR, "X_test.npy")
LOCAL_Y_TEST      = os.path.join(TRAINED_DIR, "y_test.npy")

TARGET_SCALER_S3_KEY = "models/lumen_2/scalers/spx_target_scaler.joblib"
LOCAL_TARGET_SCALER  = os.path.join(TRAINED_DIR, "spx_spy_vix_target_scaler.joblib")

##############################################################################
# 4) DOWNLOAD
##############################################################################
def maybe_download_test_arrays():
    """
    Force-download the single test set .npy files (X_test and y_test) from S3 => local,
    removing any existing local copies.
    """
    download_file_from_s3(X_TEST_S3_KEY, LOCAL_X_TEST)
    download_file_from_s3(Y_TEST_S3_KEY, LOCAL_Y_TEST)

def maybe_download_model():
    """Force-download the trained Lumen2 model from S3 => local."""
    download_file_from_s3(MODEL_S3_KEY, MODEL_LOCAL)

def maybe_download_target_scaler():
    """Force-download the target scaler if it exists on S3."""
    try:
        download_file_from_s3(TARGET_SCALER_S3_KEY, LOCAL_TARGET_SCALER)
        logging.info("[maybe_download_target_scaler] Target scaler downloaded.")
    except Exception as exc:
        logging.warning(f"[maybe_download_target_scaler] Could not download target scaler => {exc}")

##############################################################################
# 5) LOAD .NPY
##############################################################################
def load_npy(local_path):
    """Loads a .npy file or returns None if missing/empty."""
    if not os.path.exists(local_path):
        logging.error(f"{local_path} not found => skipping load.")
        return None
    arr = np.load(local_path, allow_pickle=False)
    if arr.size == 0:
        logging.error(f"{local_path} is empty => skipping load.")
        return None
    return arr

##############################################################################
# 6) EVALUATION
##############################################################################
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on scaled domain. Also attempt real-domain metrics if target scaler is present.
    Logs MSE, RMSE, and R².
    """
    if X_test is None or y_test is None:
        logging.error("[evaluate_model] No X_test or y_test => cannot evaluate.")
        return

    if len(X_test) != len(y_test):
        logging.error(f"[evaluate_model] Mismatch: X_test={len(X_test)}, y_test={len(y_test)} => abort.")
        return

    expected_feats = model.input_shape[-1]
    actual_feats   = X_test.shape[-1]
    if actual_feats < expected_feats:
        diff = expected_feats - actual_feats
        logging.warning(f"Padding X_test from {actual_feats} => {expected_feats} feats w/ zeros.")
        X_test = np.pad(X_test, ((0,0),(0,0),(0,diff)), mode="constant", constant_values=0)
    elif actual_feats > expected_feats:
        diff = actual_feats - expected_feats
        logging.warning(f"Truncating X_test from {actual_feats} => {expected_feats} feats.")
        X_test = X_test[:,:, :expected_feats]

    X_test = X_test.astype("float32")
    y_test = y_test.astype("float32")

    preds = model.predict(X_test, verbose=0)

    mse_val  = mean_squared_error(y_test, preds)
    rmse_val = np.sqrt(mse_val)
    r2_val   = r2_score(y_test, preds)

    logging.info("=== EVALUATION in scaled domain ===")
    logging.info(f"MSE:  {mse_val:.8f}")
    logging.info(f"RMSE: {rmse_val:.8f}")
    logging.info(f"R²:   {r2_val:.6f}")

    if os.path.exists(LOCAL_TARGET_SCALER):
        try:
            import joblib
            target_scaler = joblib.load(LOCAL_TARGET_SCALER)
            preds_inv = target_scaler.inverse_transform(preds.reshape(-1,1))
            y_test_inv= target_scaler.inverse_transform(y_test.reshape(-1,1))

            mse_real  = mean_squared_error(y_test_inv, preds_inv)
            rmse_real = np.sqrt(mse_real)
            r2_real   = r2_score(y_test_inv, preds_inv)

            logging.info("=== EVALUATION in real-price domain ===")
            logging.info(f"MSE:  {mse_real:.4f}")
            logging.info(f"RMSE: {rmse_real:.4f}")
            logging.info(f"R²:   {r2_real:.6f}")
        except Exception as exc:
            logging.warning(f"[evaluate_model] Could not inverse-transform => {exc}")

##############################################################################
# 7) MAIN
##############################################################################
def main():
    logging.info("=== evaluate_lumen_2 => Starting ===")

    maybe_download_test_arrays()
    maybe_download_model()
    maybe_download_target_scaler()

    if not os.path.exists(MODEL_LOCAL):
        logging.error(f"Model file not found => {MODEL_LOCAL}")
        return
    model = load_model(MODEL_LOCAL)
    logging.info(f"Model loaded from => {MODEL_LOCAL}")

    X_test = load_npy(LOCAL_X_TEST)
    y_test = load_npy(LOCAL_Y_TEST)
    if X_test is None or y_test is None:
        logging.error("[main] Missing test arrays => abort.")
        return

    logging.info(f"[main] X_test => {X_test.shape}, y_test => {y_test.shape}")

    evaluate_model(model, X_test, y_test)

    logging.info("=== evaluate_lumen_2 => Done ===")


if __name__ == "__main__":
    main()