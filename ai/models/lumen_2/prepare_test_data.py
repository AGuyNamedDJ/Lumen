import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import boto3
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

##############################################################################
# 1) AWS S3 UTILS
##############################################################################
def get_s3_client():
    """
    Build a boto3 S3 client using the AWS creds set in environment variables.
    """
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

def download_file_from_s3(s3_key: str, local_path: str):
    """
    Force-download a file from s3://<bucket>/<s3_key> into local_path.
    If local_path already exists, remove it and re-download.
    """
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
    s3 = get_s3_client()

    if os.path.exists(local_path):
        logging.info(f"[download_file_from_s3] Removing existing local file => {local_path}")
        os.remove(local_path)

    logging.info(f"[download_file_from_s3] Downloading s3://{bucket_name}/{s3_key} → {local_path}")
    s3.download_file(bucket_name, s3_key, local_path)
    logging.info("[download_file_from_s3] Done.")

def upload_file_to_s3(local_path, s3_key):
    """
    Upload local_path to s3://<bucket>/<s3_key>.
    """
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
    s3 = get_s3_client()
    logging.info(f"[upload_file_to_s3] Uploading {local_path} → s3://{bucket_name}/{s3_key}")
    s3.upload_file(local_path, bucket_name, s3_key)
    logging.info("[upload_file_to_s3] Done.")

##############################################################################
# 2) ENV, LOGGING, PATHS
##############################################################################
load_dotenv()
logging.basicConfig(level=logging.INFO)

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

BASE_DIR      = os.path.abspath(os.path.join(script_dir, "..", ".."))
DATA_DIR      = os.path.join(BASE_DIR, "data", "lumen_2", "processed")
FEATURED_DIR  = os.path.join(BASE_DIR, "data", "lumen_2", "featured")
SEQUENCES_DIR = os.path.join(FEATURED_DIR, "sequences")
SCALER_DIR    = os.path.join(BASE_DIR, "models", "lumen_2", "scalers")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FEATURED_DIR, exist_ok=True)
os.makedirs(SEQUENCES_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# The final test CSV & scaler we want to load from S3
MERGED_CSV_S3_KEY  = "data/lumen2/featured/spx_spy_vix_merged_features.csv"
LOCAL_MERGED_CSV   = os.path.join(FEATURED_DIR, "spx_spy_vix_merged_realtime.csv")

SCALER_S3_KEY      = "models/lumen_2/scalers/spx_spy_vix_scaler.joblib"
LOCAL_SCALER       = os.path.join(SCALER_DIR, "spx_spy_vix_scaler.joblib")

##############################################################################
# 3) DOWNLOAD TEST DATA & SCALER
##############################################################################
def download_test_data_and_scaler():
    """
    Force-downloads the 'spx_spy_vix_merged_features.csv' (test data)
    and the 'spx_spy_vix_scaler.joblib' (scaler) from S3.
    """
    download_file_from_s3(MERGED_CSV_S3_KEY, LOCAL_MERGED_CSV)
    download_file_from_s3(SCALER_S3_KEY, LOCAL_SCALER)

##############################################################################
# 4) LOAD + SCALE + CREATE SEQUENCES
##############################################################################
def load_and_scale():
    """
    Loads spx_spy_vix_merged_realtime.csv, applies the same MinMaxScaler
    used in feature engineering (spx_spy_vix_scaler.joblib), and returns a DataFrame.
    If the scaler is missing, returns the DataFrame unscaled.
    """
    if not os.path.exists(LOCAL_MERGED_CSV):
        logging.error(f"No CSV found at {LOCAL_MERGED_CSV} => cannot proceed.")
        return pd.DataFrame()

    df = pd.read_csv(LOCAL_MERGED_CSV)
    if df.empty:
        logging.warning("[load_and_scale] CSV is empty => no data.")
        return pd.DataFrame()

    if not os.path.exists(LOCAL_SCALER):
        logging.warning(f"No local scaler found at {LOCAL_SCALER}; using unscaled data.")
        return df

    # Load the previously fitted MinMaxScaler
    scaler = joblib.load(LOCAL_SCALER)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude columns we don't want to scale (e.g. 'timestamp', 'target_1h' if present)
    for exclude_col in ["timestamp", "target_1h"]:
        if exclude_col in numeric_cols:
            numeric_cols.remove(exclude_col)

    # Replace infs with NaNs, then drop rows if necessary
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=numeric_cols, inplace=True)

    # Scale the numeric columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df

def create_sequences(df, seq_len=60, prefix="spx_spy_vix_test"):
    """
    Builds test sequences (X_3D, Y_3D) from the scaled DataFrame, using
    'target_1h' as the label. Saves them to the sequences folder as .npy files.
    """
    if "target_1h" not in df.columns:
        logging.warning("[create_sequences] 'target_1h' missing => no Y array.")
        return

    # X is all numeric columns except 'target_1h'
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]) if c != "target_1h"]
    X_arr = df[numeric_cols].values
    Y_arr = df["target_1h"].values

    X_seq, Y_seq = [], []
    for i in range(len(X_arr) - seq_len):
        X_seq.append(X_arr[i : i + seq_len])
        Y_seq.append(Y_arr[i + seq_len])

    X_seq = np.array(X_seq, dtype=np.float32)
    Y_seq = np.array(Y_seq, dtype=np.float32).reshape(-1, 1)

    if len(X_seq) == 0:
        logging.warning("[create_sequences] Not enough rows => no sequences.")
        return

    # Save as a single chunk (part0)
    x_filename = f"{prefix}_X_3D_part0.npy"
    y_filename = f"{prefix}_Y_3D_part0.npy"
    x_path = os.path.join(SEQUENCES_DIR, x_filename)
    y_path = os.path.join(SEQUENCES_DIR, y_filename)

    np.save(x_path, X_seq)
    np.save(y_path, Y_seq)

    logging.info(f"[create_sequences] Saved => {x_path}: {X_seq.shape}, {y_path}: {Y_seq.shape}")

##############################################################################
# 5) MAIN
##############################################################################
def main():
    logging.info("[prepare_spx_spy_vix_test_data] Starting test data preparation...")
    download_test_data_and_scaler()

    df = load_and_scale()
    if df.empty:
        logging.error("DataFrame is empty => cannot create sequences.")
        return

    create_sequences(df, seq_len=60, prefix="spx_spy_vix_test")
    logging.info("[prepare_spx_spy_vix_test_data] Done.")

if __name__ == "__main__":
    main()