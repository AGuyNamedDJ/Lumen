import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

try:
    from ai.utils.aws_s3_utils import download_file_from_s3, auto_upload_file_to_s3
except ImportError:
    download_file_from_s3 = None
    auto_upload_file_to_s3 = None
    logging.warning("S3 utilities not found; skipping cloud operations.")

load_dotenv()
logging.basicConfig(level=logging.INFO)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FEATURED_DIR= os.path.join(BASE_DIR, "../../data/lumen_2/featured")
MODEL_DIR   = os.path.join(BASE_DIR, "../../models/lumen_2")
os.makedirs(MODEL_DIR, exist_ok=True)

# Only handle real_time_spy in this file:
DATA_S3_CSV      = "data/lumen2/featured/featured_real_time_spy.csv"
SCALER_S3_PATH   = "models/lumen_2/scalers/real_time_spy_scaler.joblib"

LOCAL_CSV        = os.path.join(FEATURED_DIR, "real_time_spy.csv")
SCALER_LOCAL_DIR = os.path.join(MODEL_DIR, "scalers")
SCALER_LOCAL     = os.path.join(SCALER_LOCAL_DIR, "real_time_spy_data_scaler.joblib")

TARGET_COL   = "current_price"
TIMESTAMP_COL= "timestamp"
SEQ_LEN      = 60  # match whatever seq_len your model expects (60 in your logs)
X_TEST_REAL  = os.path.join(MODEL_DIR, "X_test_real.npy")
Y_TEST_REAL  = os.path.join(MODEL_DIR, "y_test_real.npy")


def download_csv():
    if not download_file_from_s3:
        logging.warning("download_file_from_s3 not available; expecting local CSV only.")
        return
    os.makedirs(FEATURED_DIR, exist_ok=True)
    try:
        logging.info(f"Downloading {DATA_S3_CSV} → {LOCAL_CSV}")
        download_file_from_s3(DATA_S3_CSV, LOCAL_CSV)
    except Exception as e:
        logging.error(f"Failed to download CSV from S3: {e}")


def download_scaler():
    if not download_file_from_s3:
        logging.warning("No S3 download_file_from_s3; expecting local scaler.")
        return
    os.makedirs(SCALER_LOCAL_DIR, exist_ok=True)
    try:
        logging.info(f"Downloading {SCALER_S3_PATH} → {SCALER_LOCAL}")
        download_file_from_s3(SCALER_S3_PATH, SCALER_LOCAL)
    except Exception as e:
        logging.error(f"Failed to download scaler from S3: {e}")


def load_dataframe():
    if not os.path.exists(LOCAL_CSV):
        logging.error(f"CSV file missing at {LOCAL_CSV}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(LOCAL_CSV, parse_dates=[TIMESTAMP_COL])
    except Exception:
        logging.warning(f"parse_dates failed on {TIMESTAMP_COL}, reading plain CSV.")
        df = pd.read_csv(LOCAL_CSV)
    if df.empty:
        logging.warning("Loaded real_time_spy CSV is empty.")
    return df


def load_scaler():
    if not os.path.exists(SCALER_LOCAL):
        logging.warning(f"Local scaler not found at {SCALER_LOCAL}. Data will remain unscaled.")
        return None
    return joblib.load(SCALER_LOCAL)


def apply_scaler(df, scaler):
    if df.empty:
        return df
    # Drop the timestamp + target from feature set
    feats = df.drop(columns=[TIMESTAMP_COL, TARGET_COL], errors="ignore")

    if not scaler:
        logging.warning("No scaler loaded; returning unscaled features.")
        return df  # unchanged

    # Ensure columns match scaler
    want_cols = getattr(scaler, "feature_names_in_", feats.columns)
    feats = feats.reindex(columns=want_cols, fill_value=0)
    scaled_arr = scaler.transform(feats)
    scaled_df = pd.DataFrame(scaled_arr, columns=want_cols, index=df.index)

    # Recombine: timestamp, scaled feats, target
    out = pd.concat([
        df[[TIMESTAMP_COL]].reset_index(drop=True),
        scaled_df.reset_index(drop=True),
        df[[TARGET_COL]].reset_index(drop=True)
    ], axis=1)
    return out


def build_sequences(df, seq_len=60):
    if df.empty:
        return None, None

    if TIMESTAMP_COL in df.columns:
        df = df.drop(columns=[TIMESTAMP_COL], errors="ignore")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target '{TARGET_COL}' missing from DataFrame columns: {df.columns.tolist()}")

    # Attempt to load feature_names_real (optional)
    feats_list_path = os.path.join(MODEL_DIR, "feature_names_real.npy")
    try:
        feats_list = np.load(feats_list_path, allow_pickle=True).tolist()
        logging.info(f"Using {len(feats_list)} features from feature_names_real.npy")
    except FileNotFoundError:
        feats_list = None
        logging.warning("feature_names_real.npy not found; using all columns except target.")

    if feats_list:
        feats_list = [c for c in feats_list if c in df.columns]
        feats_df   = df[feats_list]
    else:
        feats_df   = df.drop(columns=[TARGET_COL], errors="ignore")

    X_vals = feats_df.values
    y_vals = df[TARGET_COL].values

    X_seq, y_seq = [], []
    for i in range(len(X_vals) - seq_len):
        X_seq.append(X_vals[i : i + seq_len])
        y_seq.append(y_vals[i + seq_len])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)
    return X_seq, y_seq


def main():
    # 1) Possibly download from S3
    download_csv()
    download_scaler()

    # 2) Load
    df = load_dataframe()
    if df.empty:
        logging.error("No real_time_spy data found; abort.")
        return

    # 3) Load scaler & apply
    scl = load_scaler()
    df_scaled = apply_scaler(df, scl)
    if df_scaled.empty:
        logging.error("After applying scaler, DataFrame is empty. Abort.")
        return

    # 4) Build sequences
    X_real, y_real = build_sequences(df_scaled, seq_len=SEQ_LEN)
    if X_real is None or y_real is None:
        logging.error("No real_time_spy sequences generated.")
        return

    # 5) Save .npy
    np.save(X_TEST_REAL, X_real)
    np.save(Y_TEST_REAL, y_real)
    logging.info(f"Saved real-time test arrays: X_test_real={X_real.shape}, y_test_real={y_real.shape}")

    # 6) Optionally upload .npy to S3
    if auto_upload_file_to_s3:
        try:
            auto_upload_file_to_s3(X_TEST_REAL, "models/lumen_2/trained")
            auto_upload_file_to_s3(Y_TEST_REAL, "models/lumen_2/trained")
            logging.info("Uploaded X_test_real & y_test_real to S3 => models/lumen_2/trained/")
        except Exception as exc:
            logging.warning(f"Could not upload test arrays to S3: {exc}")

    logging.info("Real-time SPY test data preparation complete!")


if __name__ == "__main__":
    main()