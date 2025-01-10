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
    Create a boto3 S3 client using AWS credentials from env vars.
    """
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

def download_file_from_s3(s3_key: str, local_path: str):
    """
    Force-download a file from s3://<bucket>/<s3_key> to local_path,
    removing local_path if it already exists.
    """
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
    s3 = get_s3_client()

    # Always remove any local copy first to ensure we download fresh
    if os.path.exists(local_path):
        logging.info(f"[download_file_from_s3] Removing existing => {local_path}")
        os.remove(local_path)

    logging.info(f"[download_file_from_s3] Downloading s3://{bucket_name}/{s3_key} → {local_path}")
    s3.download_file(bucket_name, s3_key, local_path)
    logging.info("[download_file_from_s3] Done.")

def upload_file_to_s3(local_path: str, s3_key: str):
    """
    Uploads local_path => s3://<bucket>/<s3_key>.
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

# These S3 keys should point to the correct, updated files
MERGED_CSV_S3_KEY = "data/lumen2/featured/spx_vix_test.csv"
LOCAL_MERGED_CSV  = os.path.join(FEATURED_DIR, "spx_spy_vix_merged_realtime.csv")

SCALER_S3_KEY = "models/lumen_2/scalers/spx_feature_scaler.joblib"
LOCAL_SCALER  = os.path.join(SCALER_DIR, "spx_feature_scaler.joblib")

##############################################################################
# 3) DOWNLOAD
##############################################################################
def download_test_data_and_scaler():
    """
    Force-download the test CSV + the fitted MinMaxScaler from S3 => local.
    Always removes local copies first to ensure a fresh download.
    """
    download_file_from_s3(MERGED_CSV_S3_KEY, LOCAL_MERGED_CSV)
    download_file_from_s3(SCALER_S3_KEY,  LOCAL_SCALER)

##############################################################################
# 4) LOAD + ALIGN COLUMNS + SCALE
##############################################################################
def load_and_scale():
    """
    Loads spx_spy_vix_merged_realtime.csv → aligns columns w/ scaler.feature_names_in_
    → transforms the numeric columns. Returns the DataFrame.
    """
    if not os.path.exists(LOCAL_MERGED_CSV):
        logging.error(f"[load_and_scale] Missing CSV => {LOCAL_MERGED_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(LOCAL_MERGED_CSV)
    if df.empty:
        logging.warning("[load_and_scale] Merged CSV is empty => returning empty df.")
        return pd.DataFrame()

    # If there's no local scaler, skip scaling
    if not os.path.exists(LOCAL_SCALER):
        logging.warning(f"[load_and_scale] No scaler found => unscaled data returned.")
        return df

    scaler: MinMaxScaler = joblib.load(LOCAL_SCALER)
    scaler_cols = list(scaler.feature_names_in_)

    # 1) Identify numeric columns from DF
    df_num = df.select_dtypes(include=[np.number]).copy()

    # 2) Drop non-feature columns
    for skip_col in ["timestamp", "target_1h"]:
        if skip_col in df_num.columns:
            df_num.drop(columns=[skip_col], inplace=True, errors="ignore")

    # 3) Align columns to scaler_cols
    aligned_df = pd.DataFrame(0.0, index=df_num.index, columns=scaler_cols)
    common_cols = set(df_num.columns).intersection(scaler_cols)
    for c in common_cols:
        aligned_df[c] = df_num[c]

    # 4) Replace inf, drop rows w/ any NaN (strict drop)
    aligned_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    aligned_df.dropna(axis=0, how="any", inplace=True)

    # 5) Apply the scaler
    scaled_arr = scaler.transform(aligned_df[scaler_cols])
    aligned_df[scaler_cols] = scaled_arr

    # 6) Re-attach scaled columns to original df, removing old numeric columns
    df.drop(columns=df_num.columns, inplace=True, errors="ignore")
    for c in scaler_cols:
        df[c] = aligned_df[c].values

    return df

##############################################################################
# 5) CREATE SEQUENCES
##############################################################################
def create_sequences(df, seq_len=60, prefix="spx_spy_vix_test"):
    """
    Builds X_3D, Y_3D from the final DataFrame. 'target_1h' is the label if present.
    Saves as {prefix}_X_3D_part0.npy and {prefix}_Y_3D_part0.npy.
    """
    if "target_1h" not in df.columns:
        logging.warning("[create_sequences] 'target_1h' not found => skipping Y.")
        return

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "target_1h"]
    X_vals = df[numeric_cols].values
    y_vals = df["target_1h"].values

    X_seq, y_seq = [], []
    for i in range(len(X_vals) - seq_len):
        X_seq.append(X_vals[i : i + seq_len])
        y_seq.append(y_vals[i + seq_len])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)

    if len(X_seq) == 0:
        logging.warning(f"[create_sequences] Not enough rows => 0 sequences built.")
        return

    x_path = os.path.join(SEQUENCES_DIR, f"{prefix}_X_3D_part0.npy")
    y_path = os.path.join(SEQUENCES_DIR, f"{prefix}_Y_3D_part0.npy")

    np.save(x_path, X_seq)
    np.save(y_path, y_seq)
    logging.info(f"[create_sequences] => {x_path}: {X_seq.shape}, {y_path}: {y_seq.shape}")

##############################################################################
# 6) MAIN
##############################################################################
def main():
    logging.info("[prepare_spx_spy_vix_test_data] Starting test-data prep.")
    download_test_data_and_scaler()

    df = load_and_scale()
    if df.empty:
        logging.error("[main] DataFrame is empty => aborting.")
        return

    create_sequences(df, seq_len=60, prefix="spx_spy_vix_test")
    logging.info("[prepare_spx_spy_vix_test_data] Done.")

if __name__ == "__main__":
    main()