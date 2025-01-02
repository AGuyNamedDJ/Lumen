#!/usr/bin/env python3
import os
import sys
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
import joblib
import boto3

##############################################################################
# 1) AWS S3 HELPER FUNCTIONS
##############################################################################
def get_s3_client():
    """
    Builds a boto3 S3 client with region/key/secret from environment variables.
    """
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

def download_file_from_s3(s3_key: str, local_path: str):
    """
    Always re-download a file from s3://<bucket>/<s3_key>, overwriting local_path if it exists.
    """
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
    s3 = get_s3_client()
    if os.path.exists(local_path):
        logging.info(f"[download_file_from_s3] Removing existing local file => {local_path}")
        os.remove(local_path)

    logging.info(f"[download_file_from_s3] Downloading s3://{bucket_name}/{s3_key} → {local_path}")
    s3.download_file(bucket_name, s3_key, local_path)
    logging.info("[download_file_from_s3] Done.")

def upload_file_to_s3(local_path: str, s3_key: str):
    """
    Uploads a local file to the given s3://<bucket>/<s3_key>.
    """
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
    s3 = get_s3_client()
    try:
        logging.info(f"[upload_file_to_s3] Uploading {local_path} → s3://{bucket_name}/{s3_key}")
        s3.upload_file(local_path, bucket_name, s3_key)
        logging.info("[upload_file_to_s3] Done.")
    except Exception as e:
        logging.error(f"Error uploading {local_path} to S3: {e}")
        raise

def auto_upload_file_to_s3(local_path: str, s3_subfolder: str = ""):
    """
    Convenience wrapper that derives the filename from local_path
    and optionally places it in s3_subfolder.
    """
    base_name = os.path.basename(local_path)
    if s3_subfolder:
        s3_key = f"{s3_subfolder}/{base_name}"
    else:
        s3_key = base_name
    upload_file_to_s3(local_path, s3_key)

##############################################################################
# 2) ENV, LOGGING, AND PATH SETUP
##############################################################################
load_dotenv()
logging.basicConfig(level=logging.INFO)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

BASE_DIR     = os.path.abspath(os.path.join(script_dir, "..", ".."))
DATA_DIR     = os.path.join(BASE_DIR, "data", "lumen_2", "processed")
FEATURED_DIR = os.path.join(BASE_DIR, "data", "lumen_2", "featured")
SEQUENCES_DIR= os.path.join(FEATURED_DIR, "sequences")
MODEL_DIR    = os.path.join(BASE_DIR, "models", "lumen_2")  # <-- Define MODEL_DIR
SCALER_DIR   = os.path.join(MODEL_DIR, "scalers")           # <-- And SCALER_DIR

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FEATURED_DIR, exist_ok=True)
os.makedirs(SEQUENCES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

##############################################################################
# 3) DOWNLOAD PROCESSED CSVs
##############################################################################
def download_processed_csvs():
    """
    Retrieves processed CSVs for real_time_spx, real_time_spy, real_time_vix from S3
    and overwrites local versions if they exist (due to updated data).
    """
    spx_s3_key = "data/lumen2/processed/processed_real_time_spx.csv"
    spy_s3_key = "data/lumen2/processed/processed_real_time_spy.csv"
    vix_s3_key = "data/lumen2/processed/processed_real_time_vix.csv"

    spx_local = os.path.join(DATA_DIR, "processed_real_time_spx.csv")
    spy_local = os.path.join(DATA_DIR, "processed_real_time_spy.csv")
    vix_local = os.path.join(DATA_DIR, "processed_real_time_vix.csv")

    download_file_from_s3(spx_s3_key, spx_local)
    download_file_from_s3(spy_s3_key, spy_local)
    download_file_from_s3(vix_s3_key, vix_local)

##############################################################################
# 4) MERGE SPX, SPY, VIX
##############################################################################
def merge_spx_spy_vix_3min():
    spx_path = os.path.join(DATA_DIR, "processed_real_time_spx.csv")
    spy_path = os.path.join(DATA_DIR, "processed_real_time_spy.csv")
    vix_path = os.path.join(DATA_DIR, "processed_real_time_vix.csv")

    logging.info(f"[merge_spx_spy_vix_3min] Reading {spx_path}, {spy_path}, {vix_path}")
    spx = pd.read_csv(spx_path)
    spy = pd.read_csv(spy_path)
    vix = pd.read_csv(vix_path)

    spx["timestamp"] = pd.to_datetime(spx["timestamp"], errors="coerce")
    spy["timestamp"] = pd.to_datetime(spy["timestamp"], errors="coerce")
    vix["timestamp"] = pd.to_datetime(vix["timestamp"], errors="coerce")

    spx.sort_values("timestamp", inplace=True)
    spy.sort_values("timestamp", inplace=True)
    vix.sort_values("timestamp", inplace=True)

    spx.set_index("timestamp", inplace=True)
    spy.set_index("timestamp", inplace=True)
    vix.set_index("timestamp", inplace=True)

    # rename
    if "current_price" in spx.columns:
        spx.rename(columns={"current_price": "spx_price"}, inplace=True)
    if "volume" in spx.columns:
        spx.rename(columns={"volume": "spx_volume"}, inplace=True)
    if "current_price" in spy.columns:
        spy.rename(columns={"current_price": "spy_price"}, inplace=True)
    if "volume" in spy.columns:
        spy.rename(columns={"volume": "spy_volume"}, inplace=True)

    # upsample vix => 3min
    vix_3m = vix.resample("3Min").last().ffill()
    if "current_price" in vix_3m.columns:
        vix_3m.rename(columns={"current_price":"vix_price"}, inplace=True)
    if "volume" in vix_3m.columns:
        vix_3m.rename(columns={"volume":"vix_volume"}, inplace=True)

    # merge spx & spy
    spx_spy = spx.join(spy, how="outer", lsuffix="_spx", rsuffix="_spy")
    merged  = spx_spy.join(vix_3m, how="outer", rsuffix="_vix")
    merged.sort_index(inplace=True)
    merged.ffill(inplace=True)
    merged.reset_index(inplace=True)
    merged.rename(columns={"index":"timestamp"}, inplace=True)

    logging.info(f"[merge_spx_spy_vix_3min] Final shape after merge+ffill: {merged.shape}")
    return merged

##############################################################################
# 5) INDICATOR FUNCTIONS
##############################################################################
def add_spx_indicators(df):
    if "spx_price" not in df.columns:
        return df
    df.sort_values("timestamp", inplace=True)
    p = df["spx_price"]
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    df["SPX_MACD"]        = ema12 - ema26
    df["SPX_MACD_Signal"] = df["SPX_MACD"].ewm(span=9, adjust=False).mean()

    sma20 = p.rolling(20).mean()
    std20 = p.rolling(20).std()
    df["SPX_BollU"] = sma20 + 2*std20
    df["SPX_BollL"] = sma20 - 2*std20

    # RSI
    delta = p.diff()
    gain  = delta.where(delta>0, 0).rolling(14).mean()
    loss  = (-delta.where(delta<0,0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["SPX_RSI"] = 100 - 100/(1+rs)

    # OBV if spx_volume
    if "spx_volume" in df.columns:
        sign_price = np.sign(delta.fillna(0))
        df["SPX_OBV"] = (sign_price * df["spx_volume"]).fillna(0).cumsum()
    return df

def add_spy_indicators(df):
    if "spy_price" not in df.columns:
        return df
    df.sort_values("timestamp", inplace=True)
    p = df["spy_price"]
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    df["SPY_MACD"]        = ema12 - ema26
    df["SPY_MACD_Signal"] = df["SPY_MACD"].ewm(span=9, adjust=False).mean()

    sma20 = p.rolling(20).mean()
    std20 = p.rolling(20).std()
    df["SPY_BollU"] = sma20 + 2*std20
    df["SPY_BollL"] = sma20 - 2*std20

    delta = p.diff()
    gain  = delta.where(delta>0,0).rolling(14).mean()
    loss  = (-delta.where(delta<0,0)).rolling(14).mean()
    rs    = gain/(loss+1e-9)
    df["SPY_RSI"] = 100 - 100/(1+rs)

    if "spy_volume" in df.columns:
        sign_price = np.sign(delta.fillna(0))
        df["SPY_OBV"] = (sign_price * df["spy_volume"]).fillna(0).cumsum()
    return df

def add_vix_indicators(df):
    if "vix_price" not in df.columns:
        return df
    df.sort_values("timestamp", inplace=True)
    p = df["vix_price"]

    delta = p.diff()
    gain  = delta.where(delta>0,0).rolling(14).mean()
    loss  = (-delta.where(delta<0,0)).rolling(14).mean()
    rs    = gain/(loss+1e-9)
    df["VIX_RSI"] = 100 - 100/(1+rs)

    df["VIX_EMA_10"] = p.ewm(span=10, adjust=False).mean()
    df["VIX_EMA_20"] = p.ewm(span=20, adjust=False).mean()

    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    df["VIX_MACD"] = ema12 - ema26
    df["VIX_MACD_Signal"] = df["VIX_MACD"].ewm(span=9, adjust=False).mean()
    return df

def add_spy_vix_rolling_corr(df, window=30):
    if "spy_price" not in df.columns or "vix_price" not in df.columns:
        return df
    df.sort_values("timestamp", inplace=True)
    sp = df["spy_price"]
    vx = df["vix_price"]
    df["SPY_VIX_RollCorr_30"] = sp.rolling(window).corr(vx)
    return df

##############################################################################
# 6) CREATE TARGET
##############################################################################
def create_target_1h(df):
    if "spy_price" in df.columns:
        df["target_1h"] = df["spy_price"].shift(-20)
    else:
        logging.warning("No spy_price => cannot create target_1h.")
    return df


##############################################################################
# 7) SEPARATE FEATURE + TARGET SCALER
##############################################################################
def separate_feature_and_target_scaling(df):
    if "target_1h" not in df.columns:
        logging.warning("No target_1h => skipping separate scaling.")
        return df

    # 1) Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove("target_1h")

    # Fill any NaN
    if df[numeric_cols].isna().any().any():
        logging.warning("NaNs in feature columns => filling with 0.0")
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

    feat_scaler = MinMaxScaler()
    df[numeric_cols] = feat_scaler.fit_transform(df[numeric_cols])

    # Save feature scaler
    feat_scaler_path = os.path.join(SCALER_DIR, "spx_spy_vix_scaler.joblib")
    joblib.dump(feat_scaler, feat_scaler_path)
    logging.info(f"Saved feature scaler => {feat_scaler_path}")
    try:
        auto_upload_file_to_s3(feat_scaler_path, "models/lumen_2/scalers")
    except Exception as exc:
        logging.warning(f"Could not upload feature scaler => {exc}")

    # 2) Target scaler
    tgt_scaler = MinMaxScaler()
    tvals = df["target_1h"].fillna(method="ffill").fillna(method="bfill").values.reshape(-1,1)
    df["target_1h"] = tgt_scaler.fit_transform(tvals)

    tgt_scaler_path = os.path.join(SCALER_DIR, "spx_spy_vix_target_scaler.joblib")
    joblib.dump(tgt_scaler, tgt_scaler_path)
    logging.info(f"Saved target scaler => {tgt_scaler_path}")
    try:
        auto_upload_file_to_s3(tgt_scaler_path, "models/lumen_2/scalers")
    except Exception as exc:
        logging.warning(f"Could not upload target scaler => {exc}")

    # 3) Save final feature columns (not including target_1h)
    feat_list = np.array(numeric_cols, dtype=object)
    feat_list_path = os.path.join(SCALER_DIR, "feature_names.npy")
    np.save(feat_list_path, feat_list)
    logging.info(f"Saved feature columns => {feat_list_path}")
    try:
        auto_upload_file_to_s3(feat_list_path, "models/lumen_2/scalers")
    except Exception as exc:
        logging.warning(f"Could not upload feature_names => {exc}")

    return df

##############################################################################
# 8) TIME-SERIES SPLIT
##############################################################################
def time_series_split(df, train_frac=0.7, val_frac=0.15):
    df_sorted = df.sort_values("timestamp").copy()
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    df_train = df_sorted.iloc[:train_end]
    df_val   = df_sorted.iloc[train_end:val_end]
    df_test  = df_sorted.iloc[val_end:]
    logging.info(f"[time_series_split] -> Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")
    return df_train, df_val, df_test

##############################################################################
# 9) CREATE SEQUENCES
##############################################################################
def create_sequences_in_chunks(df, prefix="spx_spy_vix", seq_len=60, chunk_size=10000):
    timestamps = df.pop("timestamp")
    all_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "target_1h" not in all_cols:
        logging.warning(f"No target_1h => skipping sequences.")
        return
    all_cols.remove("target_1h")

    X_arr = df[all_cols].values
    Y_arr = df["target_1h"].values
    n = len(X_arr)
    if n < seq_len:
        logging.warning(f"Not enough rows => {n} < seq_len={seq_len}")
        return

    logging.info(f"[create_sequences_in_chunks] => seq_len={seq_len}, total_rows={n}")
    start = 0
    chunk_idx = 0
    while start + seq_len <= n:
        end = min(start + chunk_size, n - seq_len + 1)
        X_list, Y_list = [], []
        for i in range(start, end):
            X_list.append(X_arr[i : i + seq_len])
            # using last step as label
            Y_list.append(Y_arr[i + seq_len - 1])

        X_np = np.array(X_list, dtype=np.float32)
        Y_np = np.array(Y_list, dtype=np.float32).reshape(-1,1)

        x_file = os.path.join(SEQUENCES_DIR, f"{prefix}_X_3D_part{chunk_idx}.npy")
        y_file = os.path.join(SEQUENCES_DIR, f"{prefix}_Y_3D_part{chunk_idx}.npy")
        np.save(x_file, X_np)
        np.save(y_file, Y_np)
        logging.info(f"[create_sequences_in_chunks] chunk {chunk_idx} => X={X_np.shape}, Y={Y_np.shape}")

        # Optional S3
        chunk_s3 = "data/lumen2/featured/sequences"
        try:
            auto_upload_file_to_s3(x_file, chunk_s3)
            auto_upload_file_to_s3(y_file, chunk_s3)
        except Exception as exc:
            logging.warning(f"Could not upload part{chunk_idx} => {exc}")

        chunk_idx += 1
        start = end

    df.insert(0, "timestamp", timestamps)

##############################################################################
# 10) MAIN
##############################################################################
def main():
    logging.info("=== Feature Engineering with SPX, SPY, VIX data (S3) ===")

    # 1) Download CSV
    download_processed_csvs()

    # 2) Merge => add indicators => create target
    df_merged = merge_spx_spy_vix_3min()
    logging.info(f"[main] shape after merge: {df_merged.shape}")

    df_merged = add_spx_indicators(df_merged)
    df_merged = add_spy_indicators(df_merged)
    df_merged = add_vix_indicators(df_merged)
    df_merged = add_spy_vix_rolling_corr(df_merged)
    logging.info(f"[main] shape after indicators: {df_merged.shape}")

    df_merged = create_target_1h(df_merged)
    logging.info(f"[main] shape after create_target_1h: {df_merged.shape}")

    # Drop rows missing target_1h
    df_merged.dropna(subset=["target_1h"], inplace=True)
    if len(df_merged) < 70:
        logging.warning("Less than 70 rows => might hamper training. Proceeding anyway.")

    # 3) Replace inf with NaN, fill or drop
    df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_merged.dropna(subset=["target_1h"], how="any", inplace=True)

    # 4) Actually do separate scaling
    df_merged = separate_feature_and_target_scaling(df_merged)
    logging.info(f"[main] shape after separate scaling: {df_merged.shape}")

    # 5) time-split
    df_train, df_val, df_test = time_series_split(df_merged, train_frac=0.7, val_frac=0.15)

    # Save CSV
    def save_csv(subdf, name):
        out_path = os.path.join(FEATURED_DIR, name)
        subdf.to_csv(out_path, index=False)
        logging.info(f"[save_csv] => {out_path}")
        try:
            auto_upload_file_to_s3(out_path, "data/lumen2/featured")
        except Exception as exc:
            logging.warning(f"Could not upload {name} => {exc}")

    save_csv(df_train, "spx_spy_vix_merged_features_train.csv")
    save_csv(df_val,   "spx_spy_vix_merged_features_val.csv")
    save_csv(df_test,  "spx_spy_vix_merged_features_test.csv")

    # 6) sequences
    create_sequences_in_chunks(df_train, prefix="spx_spy_vix_train", seq_len=60)
    create_sequences_in_chunks(df_val,   prefix="spx_spy_vix_val",   seq_len=60)
    create_sequences_in_chunks(df_test,  prefix="spx_spy_vix_test",  seq_len=60)

    logging.info("=== Done with feature engineering ===")


if __name__ == "__main__":
    main()