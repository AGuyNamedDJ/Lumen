import os
import sys
import logging
import numpy as np
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
import boto3

# -----------------------------------------------------------------------------
# Setup environment + logging
# -----------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)


def get_s3_client():
    """
    Returns a boto3 S3 client object using AWS credentials from environment variables.
    Hard-coded region to 'us-east-2' or whichever region your bucket is in.
    """
    return boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = "us-east-2"
    )

def download_file_from_s3(s3_key: str, local_path: str, force_download: bool = False):
    """
    Downloads a file from s3://<bucket>/<s3_key> into local_path.
    If force_download=True, remove any existing local file and always re-download.
    """
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "lumenaibucket")
    s3 = get_s3_client()

    if os.path.exists(local_path) and not force_download:
        logging.info(f"[download_file_from_s3] File already exists locally: {local_path}")
        return

    if os.path.exists(local_path) and force_download:
        logging.info(f"[download_file_from_s3] Removing existing local file => {local_path}")
        os.remove(local_path)

    logging.info(f"[download_file_from_s3] Downloading s3://{bucket_name}/{s3_key} → {local_path}")
    try:
        s3.download_file(bucket_name, s3_key, local_path)
        logging.info("[download_file_from_s3] Done.")
    except Exception as e:
        logging.error(f"Error downloading {s3_key} from s3://{bucket_name}: {e}")
        raise

# -----------------------------------------------------------------------------
# PATHS FOR TEST ARRAYS AND MODEL
# -----------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TRAINED_DIR  = os.path.join(BASE_DIR, "trained")
os.makedirs(TRAINED_DIR, exist_ok=True)

# If your model is stored locally or in S3:
MODEL_FILENAME = "Lumen2.keras"   # the trained model name
MODEL_LOCAL    = os.path.join(TRAINED_DIR, MODEL_FILENAME)
MODEL_S3_KEY   = "models/lumen_2/trained/Lumen2.keras"

# Paths for your spx_spy_vix_test_X_3D_part0.npy and Y_3D_part0.npy in S3:
S3_X_TEST = "data/lumen2/featured/sequences/spx_spy_vix_test_X_3D_part0.npy"
S3_Y_TEST = "data/lumen2/featured/sequences/spx_spy_vix_test_Y_3D_part0.npy"

# Local equivalents
LOCAL_X_TEST = os.path.join(TRAINED_DIR, "spx_spy_vix_test_X_3D_part0.npy")
LOCAL_Y_TEST = os.path.join(TRAINED_DIR, "spx_spy_vix_test_Y_3D_part0.npy")

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------
def maybe_download_test_arrays(force_download: bool = False):
    """
    Downloads the test .npy arrays from S3 if not already present (or if forced),
    storing them in the 'trained' subfolder.
    """
    os.makedirs(TRAINED_DIR, exist_ok=True)
    download_file_from_s3(S3_X_TEST, LOCAL_X_TEST, force_download=force_download)
    download_file_from_s3(S3_Y_TEST, LOCAL_Y_TEST, force_download=force_download)

def maybe_download_model(force_download: bool = False):
    """
    Downloads the model .keras file from S3 if not already present (or if forced),
    storing it in the 'trained' subfolder.
    """
    os.makedirs(TRAINED_DIR, exist_ok=True)
    download_file_from_s3(MODEL_S3_KEY, MODEL_LOCAL, force_download=force_download)

def load_npy_or_none(local_path: str) -> np.ndarray:
    """
    Loads a .npy file from local filesystem or returns None if not found/empty.
    """
    if not os.path.exists(local_path):
        logging.warning(f"{local_path} not found.")
        return None
    arr = np.load(local_path, allow_pickle=False)
    if arr.size == 0:
        logging.warning(f"{local_path} is empty.")
        return None
    return arr

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on X_test / y_test arrays using MSE, RMSE, R².
    """
    if X_test is None or y_test is None:
        logging.error("X_test or y_test is None => cannot evaluate.")
        return

    # Ensure dimensional match
    if len(X_test) != len(y_test):
        logging.error(f"Mismatch: X_test has {len(X_test)} samples, y_test has {len(y_test)}.")
        return

    # Check feature dimension
    expected_feats = model.input_shape[-1]
    actual_feats   = X_test.shape[-1]
    if actual_feats < expected_feats:
        diff = expected_feats - actual_feats
        logging.warning(f"Padding X_test from {actual_feats} → {expected_feats} features with zeros.")
        X_test = np.pad(X_test, ((0,0), (0,0), (0,diff)), mode="constant", constant_values=0)
    elif actual_feats > expected_feats:
        diff = actual_feats - expected_feats
        logging.warning(f"Truncating X_test from {actual_feats} → {expected_feats} features.")
        X_test = X_test[:, :, :expected_feats]

    # Convert dtypes for safety
    X_test = X_test.astype("float32")
    y_test = y_test.astype("float32")

    # Predict
    preds = model.predict(X_test, verbose=0)

    # MSE, RMSE, R²
    mse_val = mean_squared_error(y_test, preds)
    rmse_val = np.sqrt(mse_val)
    r2_val   = r2_score(y_test, preds)

    logging.info("--- EVALUATION RESULTS (TEST) ---")
    logging.info(f"MSE:  {mse_val:.6f}")
    logging.info(f"RMSE: {rmse_val:.6f}")
    logging.info(f"R²:   {r2_val:.6f}")
    logging.info("---------------------------------")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    logging.info("=== evaluate_lumen_2: Start ===")

    # 1) Ensure test arrays are downloaded
    maybe_download_test_arrays(force_download=False)

    # 2) Ensure model is downloaded
    maybe_download_model(force_download=False)

    # 3) Load the model from local
    if not os.path.exists(MODEL_LOCAL):
        logging.error(f"Model file not found => {MODEL_LOCAL}")
        return
    model = load_model(MODEL_LOCAL)
    logging.info(f"Loaded model from {MODEL_LOCAL}")

    # 4) Load test data
    X_test = load_npy_or_none(LOCAL_X_TEST)
    y_test = load_npy_or_none(LOCAL_Y_TEST)
    if X_test is None or y_test is None:
        logging.error("Missing test arrays => cannot evaluate.")
        return

    logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # 5) Evaluate
    evaluate_model(model, X_test, y_test)

    logging.info("=== evaluate_lumen_2: Done ===")


if __name__ == "__main__":
    main()