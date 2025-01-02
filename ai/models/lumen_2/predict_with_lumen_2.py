import os
import sys
import logging
import numpy as np
import boto3
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

try:
    from models.lumen_2.definitions_lumen_2 import ReduceMeanLayer
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
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

def download_file_from_s3(s3_key: str, local_path: str, force: bool = False):
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
    s3 = get_s3_client()
    if os.path.exists(local_path) and not force:
        logging.info(f"[download_file_from_s3] Already exists: {local_path}")
        return
    if os.path.exists(local_path) and force:
        logging.info(f"[download_file_from_s3] Removing existing => {local_path}")
        os.remove(local_path)
    logging.info(f"[download_file_from_s3] Downloading s3://{bucket_name}/{s3_key} → {local_path}")
    s3.download_file(bucket_name, s3_key, local_path)
    logging.info("[download_file_from_s3] Done.")

def upload_file_to_s3(local_path: str, s3_key: str):
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
    s3 = get_s3_client()
    logging.info(f"[upload_file_to_s3] {local_path} → s3://{bucket_name}/{s3_key}")
    s3.upload_file(local_path, bucket_name, s3_key)
    logging.info("[upload_file_to_s3] Done.")

# ------------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------------
script_dir    = os.path.dirname(os.path.abspath(__file__))
project_root  = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

TRAINED_DIR   = os.path.join(script_dir, "trained")
os.makedirs(TRAINED_DIR, exist_ok=True)

MODEL_NAME    = "Lumen2.keras"
MODEL_S3_KEY  = "models/lumen_2/trained/Lumen2.keras"
MODEL_LOCAL   = os.path.join(TRAINED_DIR, MODEL_NAME)

TEST_X_S3_KEY = "data/lumen2/featured/sequences/spx_spy_vix_test_X_3D_part0.npy"
TEST_Y_S3_KEY = "data/lumen2/featured/sequences/spx_spy_vix_test_Y_3D_part0.npy"
LOCAL_X_TEST  = os.path.join(TRAINED_DIR, "spx_spy_vix_test_X_3D_part0.npy")
LOCAL_Y_TEST  = os.path.join(TRAINED_DIR, "spx_spy_vix_test_Y_3D_part0.npy")

PREDICTIONS_NPY = os.path.join(TRAINED_DIR, "predictions_test.npy")
PREDICTIONS_S3  = "models/lumen_2/trained/predictions_test.npy"

# ------------------------------------------------------------------------
# DOWNLOAD UTILS
# ------------------------------------------------------------------------
def download_test_data(force=False):
    logging.info("[download_test_data] Ensuring test data is local.")
    download_file_from_s3(TEST_X_S3_KEY, LOCAL_X_TEST, force)
    download_file_from_s3(TEST_Y_S3_KEY, LOCAL_Y_TEST, force)

def download_model(force=False):
    if os.path.exists(MODEL_LOCAL) and not force:
        logging.info(f"Model file already local: {MODEL_LOCAL}")
        return
    download_file_from_s3(MODEL_S3_KEY, MODEL_LOCAL, force)

# ------------------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------------------
def load_lumen2_model():
    if not os.path.exists(MODEL_LOCAL):
        logging.error(f"No model file => {MODEL_LOCAL}")
        return None
    try:
        model = load_model(MODEL_LOCAL, custom_objects={'ReduceMeanLayer': ReduceMeanLayer})
        logging.info(f"Loaded model from {MODEL_LOCAL}")
        return model
    except Exception as exc:
        logging.error(f"Error loading model: {exc}")
        return None

# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------
def main():
    logging.info("[predict_lumen_2] Starting predictions.")
    os.makedirs(TRAINED_DIR, exist_ok=True)

    download_test_data(force=False)
    download_model(force=False)

    model = load_lumen2_model()
    if model is None:
        logging.error("Could not load Lumen2 model => abort.")
        return

    if not os.path.exists(LOCAL_X_TEST):
        logging.error(f"No test X => {LOCAL_X_TEST}")
        return
    X_test = np.load(LOCAL_X_TEST)
    logging.info(f"Loaded X_test => shape={X_test.shape}")

    Y_test = None
    if os.path.exists(LOCAL_Y_TEST):
        Y_test = np.load(LOCAL_Y_TEST)
        logging.info(f"Loaded Y_test => shape={Y_test.shape}")

    seq_len_model = model.input_shape[1]
    feat_model    = model.input_shape[2]
    if X_test.shape[1] != seq_len_model or X_test.shape[2] != feat_model:
        logging.warning(f"Shape mismatch. Model expects (None,{seq_len_model},{feat_model}); got {X_test.shape}")
        if X_test.shape[2] < feat_model:
            diff = feat_model - X_test.shape[2]
            logging.warning(f"Padding test data from {X_test.shape[2]} → {feat_model} features.")
            X_test = np.pad(X_test, ((0,0),(0,0),(0,diff)), mode="constant", constant_values=0)
        elif X_test.shape[2] > feat_model:
            diff = X_test.shape[2] - feat_model
            logging.warning(f"Trimming test data from {X_test.shape[2]} → {feat_model} features.")
            X_test = X_test[:,:,:feat_model]

    # 5) Predict
    X_test = X_test.astype("float32")
    logging.info("Predicting test data now...")
    preds = model.predict(X_test, verbose=0)
    logging.info(f"Predictions shape => {preds.shape}")

    np.save(PREDICTIONS_NPY, preds)
    logging.info(f"Saved predictions => {PREDICTIONS_NPY}")

    try:
        upload_file_to_s3(PREDICTIONS_NPY, PREDICTIONS_S3)
        logging.info(f"Uploaded => s3://<bucket>/{PREDICTIONS_S3}")
    except Exception as exc:
        logging.warning(f"Could not upload predictions => {exc}")

    logging.info("[predict_lumen_2] Done.")

if __name__ == "__main__":
    main()