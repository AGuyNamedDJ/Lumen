import os
import sys
import logging
import re
import numpy as np
import boto3
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

try:
    from models.lumen_2.definitions_lumen_2 import ReduceMeanLayer
except ImportError:
    class ReduceMeanLayer:
        pass

##############################################################################
# 1) ENV + LOGGING
##############################################################################
load_dotenv()
logging.basicConfig(level=logging.INFO)

##############################################################################
# 2) BOTO3 S3 UTILS
##############################################################################
def get_s3_client():
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

def download_file_from_s3(s3_key: str, local_path: str):
    """
    Always remove any local file, then download s3://<bucket>/<s3_key> => local_path.
    """
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
    s3 = get_s3_client()

    if os.path.exists(local_path):
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

##############################################################################
# 3) LIST + CONCAT PARTS
##############################################################################
def list_s3_objects_with_prefix(prefix):
    """
    Lists all objects in the S3 bucket that start with 'prefix'.
    Returns a list of keys.
    """
    s3 = get_s3_client()
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")

    results = []
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                results.append(obj['Key'])
    return results

def download_and_combine_testX(s3_prefix="data/lumen2/featured/sequences/", pattern=r"spx_test_X_3D_part\d+\.npy"):
    """
    1. List all parted X test files in S3 matching pattern (e.g., spx_test_X_3D_part0, part1...).
    2. Download & concatenate them into one big X_test (N, seq_len, feats).
    Returns the combined array or None if none found.
    """
    s3_keys = list_s3_objects_with_prefix(s3_prefix)
    part_keys = []
    for key in s3_keys:
        fname = os.path.basename(key)
        if re.match(pattern, fname):
            part_keys.append(key)

    if not part_keys:
        logging.warning(f"No parted test X found matching pattern {pattern} => cannot combine.")
        return None

    logging.info(f"Found {len(part_keys)} parted X test files:")
    for pk in part_keys:
        logging.info(f" - s3://.../{pk}")

    local_tmp = "tmp_testX"
    if not os.path.exists(local_tmp):
        os.makedirs(local_tmp, exist_ok=True)

    all_arrays = []
    for pk in sorted(part_keys):
        base = os.path.basename(pk)
        local_path = os.path.join(local_tmp, base)

        if os.path.exists(local_path):
            os.remove(local_path)

        logging.info(f"Downloading parted test X => {pk}")
        download_file_from_s3(pk, local_path)
        arr = np.load(local_path)
        all_arrays.append(arr)

    if not all_arrays:
        logging.warning("No parted arrays loaded => returning None.")
        return None

    combined = np.concatenate(all_arrays, axis=0)  # shape => (N, seq_len, feats)
    logging.info(f"Combined X_test shape => {combined.shape}")
    return combined

##############################################################################
# 4) PATHS
##############################################################################
script_dir    = os.path.dirname(os.path.abspath(__file__))
project_root  = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

TRAINED_DIR   = os.path.join(script_dir, "trained")
os.makedirs(TRAINED_DIR, exist_ok=True)

MODEL_NAME       = "Lumen2.keras"
MODEL_S3_KEY     = "models/lumen_2/trained/Lumen2.keras"
MODEL_LOCAL      = os.path.join(TRAINED_DIR, MODEL_NAME)

PREDICTIONS_NPY  = os.path.join(TRAINED_DIR, "predictions_test.npy")
PREDICTIONS_S3   = "models/lumen_2/trained/predictions_test.npy"

##############################################################################
# 5) DOWNLOAD MODEL
##############################################################################
def download_model():
    logging.info("[download_model] Force-downloading model file.")
    download_file_from_s3(MODEL_S3_KEY, MODEL_LOCAL)

##############################################################################
# 6) LOAD MODEL
##############################################################################
def load_lumen2_model():
    """
    Loads the local model file with the custom ReduceMeanLayer.
    Assumes the .keras file has been freshly downloaded.
    """
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

##############################################################################
# 7) MAIN
##############################################################################
def main():
    logging.info("[predict_lumen_2] Starting predictions.")
    download_model()

    model = load_lumen2_model()
    if model is None:
        logging.error("Could not load Lumen2 model => abort.")
        return

    X_test = download_and_combine_testX(
        s3_prefix="data/lumen2/featured/sequences/", 
        pattern=r"spx_test_X_3D_part\d+\.npy"
    )
    if X_test is None:
        logging.error("No parted X test data => cannot predict.")
        return

    logging.info(f"[predict_lumen_2] X_test shape => {X_test.shape}")

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

    X_test = X_test.astype("float32")
    logging.info("Predicting now on the combined parted test set...")
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