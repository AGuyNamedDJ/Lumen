import os
import sys
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
import joblib
import boto3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

##############################################################################
# AWS S3 HELPER FUNCTIONS
##############################################################################
def get_s3_client():
    """
    Creates a boto3 S3 client using environment variables for credentials.
    Defaults to region us-east-2 (adjust if needed).
    """
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

def download_file_from_s3(s3_key: str, local_path: str):
    """
    Removes any local file that already exists at local_path,
    then downloads from s3://<bucket>/<s3_key> → local_path.
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
    """
    Uploads local_path => s3://<bucket>/<s3_key>.
    """
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
    s3 = get_s3_client()
    logging.info(f"[upload_file_to_s3] Uploading {local_path} → s3://{bucket_name}/{s3_key}")
    s3.upload_file(local_path, bucket_name, s3_key)
    logging.info("[upload_file_to_s3] Done.")

def auto_upload_file_to_s3(local_path: str, s3_subfolder: str = ""):
    """
    Helper to upload a file to S3 under a subfolder if provided.
    """
    base_name = os.path.basename(local_path)
    if s3_subfolder:
        s3_key = f"{s3_subfolder}/{base_name}"
    else:
        s3_key = base_name
    upload_file_to_s3(local_path, s3_key)

##############################################################################
# ENV, LOGGING, PATHS
##############################################################################
load_dotenv()
logging.basicConfig(level=logging.INFO)

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

BASE_DIR     = os.path.abspath(os.path.join(script_dir, "..", ".."))
DATA_DIR     = os.path.join(BASE_DIR, "data", "lumen_2", "processed")
FEATURED_DIR = os.path.join(BASE_DIR, "data", "lumen_2", "featured")
SEQUENCES_DIR= os.path.join(FEATURED_DIR, "sequences")
MODEL_DIR    = os.path.join(BASE_DIR, "models", "lumen_2")
SCALER_DIR   = os.path.join(MODEL_DIR, "scalers")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FEATURED_DIR, exist_ok=True)
os.makedirs(SEQUENCES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

##############################################################################
# 1) Optional: Filter for Market Hours
##############################################################################
def filter_market_hours(df):
    """
    (Optional) Keep only Monday–Friday (dayofweek < 5),
    and limit timestamps to ~9:00–15:59 (adjust as needed).
    This ensures we’re mostly dealing with active market hours.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # Monday=0 ... Friday=4, so we exclude dayofweek >= 5 (weekends)
    df = df[df["timestamp"].dt.dayofweek < 5]

    # Example range: 9:00–15:59 local time (tweak to your actual session times)
    df = df[(df["timestamp"].dt.hour >= 9) & (df["timestamp"].dt.hour <= 15)]
    return df

##############################################################################
def download_processed_csvs():
    """
    Downloads the processed SPX & VIX CSVs from S3 => local DATA_DIR.
    The CSVs are assumed to be pre-cleaned for missing data, duplicates, etc.
    """
    spx_s3_key = "data/lumen2/processed/processed_real_time_spx.csv"
    vix_s3_key = "data/lumen2/processed/processed_real_time_vix.csv"
    spx_local = os.path.join(DATA_DIR, "processed_real_time_spx.csv")
    vix_local = os.path.join(DATA_DIR, "processed_real_time_vix.csv")

    download_file_from_s3(spx_s3_key, spx_local)
    download_file_from_s3(vix_s3_key, vix_local)

##############################################################################
# 2) Merge SPX + VIX into a single DataFrame
##############################################################################
def merge_spx_vix_3min():
    """
    Loads processed spx, vix CSVs → merges them at 3-minute intervals,
    keeping only M–F, ~9:00–15:59 times if we want market hours only.
    """
    spx_csv = os.path.join(DATA_DIR, "processed_real_time_spx.csv")
    vix_csv = os.path.join(DATA_DIR, "processed_real_time_vix.csv")

    logging.info(f"[merge_spx_vix_3min] Reading {spx_csv}, {vix_csv}")
    spx = pd.read_csv(spx_csv)
    vix = pd.read_csv(vix_csv)

    # Filter out weekend / non-market hours (optional)
    spx = filter_market_hours(spx)
    vix = filter_market_hours(vix)

    spx["timestamp"] = pd.to_datetime(spx["timestamp"], errors="coerce")
    vix["timestamp"] = pd.to_datetime(vix["timestamp"], errors="coerce")
    spx.sort_values("timestamp", inplace=True)
    vix.sort_values("timestamp", inplace=True)

    # Use timestamp as index
    spx.set_index("timestamp", inplace=True)
    vix.set_index("timestamp", inplace=True)

    # Resample at 3-minute intervals and forward-fill
    spx = spx.resample("3T").last().ffill()
    vix = vix.resample("3T").last().ffill()

    # Rename columns if present
    if "current_price" in spx.columns:
        spx.rename(columns={"current_price": "spx_price"}, inplace=True)
    if "volume" in spx.columns:
        spx.rename(columns={"volume": "spx_volume"}, inplace=True)

    if "current_price" in vix.columns:
        vix.rename(columns={"current_price": "vix_price"}, inplace=True)
    if "volume" in vix.columns:
        vix.rename(columns={"volume": "vix_volume"}, inplace=True)

    # Outer join on timestamps
    merged = spx.join(vix, how="outer", rsuffix="_vix")

    # Drop rows missing either spx or vix
    merged.dropna(subset=["spx_price", "vix_price"], how="any", inplace=True)

    # Sort again, reset index => final DataFrame
    merged.sort_index(inplace=True)
    merged.reset_index(inplace=True)
    merged.rename(columns={"index": "timestamp"}, inplace=True)
    logging.info(f"[merge_spx_vix_3min] Final shape => {merged.shape}")
    return merged

##############################################################################
# 3) Add Indicators
##############################################################################
def add_spx_indicators(df):
    """
    Adds SPX technical indicators (MACD, Bollinger, RSI, ROC, LogPrice, etc.) in raw domain.
    """
    if "spx_price" not in df.columns:
        return df
    df.sort_values("timestamp", inplace=True)
    p = df["spx_price"]

    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    df["SPX_MACD"] = ema12 - ema26
    df["SPX_MACD_Signal"] = df["SPX_MACD"].ewm(span=9, adjust=False).mean()

    sma20 = p.rolling(20).mean()
    std20 = p.rolling(20).std()
    df["SPX_BollU"] = sma20 + 2.0 * std20
    df["SPX_BollL"] = sma20 - 2.0 * std20

    delta = p.diff()
    gain  = delta.where(delta>0, 0).rolling(14).mean()
    loss  = (-delta.where(delta<0,0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["SPX_RSI"] = 100 - 100.0/(1+rs)

    df["SPX_ROC_10"] = p.pct_change(10) * 100.0
    df["SPX_LogPrice"] = np.log1p(p.clip(lower=1e-9))

    if "spx_volume" in df.columns:
        sign_price = np.sign(delta.fillna(0))
        df["SPX_OBV"] = (sign_price * df["spx_volume"]).fillna(0).cumsum()
        df["SPX_VolumeLog"] = np.log1p(df["spx_volume"].clip(lower=1e-9))
    return df

def add_vix_indicators(df):
    """
    Adds VIX technical indicators (RSI, EMAs, LogPrice) in raw domain.
    """
    if "vix_price" not in df.columns:
        return df
    df.sort_values("timestamp", inplace=True)
    p = df["vix_price"]

    delta = p.diff()
    gain  = delta.where(delta>0,0).rolling(14).mean()
    loss  = (-delta.where(delta<0,0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["VIX_RSI"] = 100 - 100.0/(1+rs)
    df["VIX_EMA_10"] = p.ewm(span=10, adjust=False).mean()
    df["VIX_EMA_20"] = p.ewm(span=20, adjust=False).mean()
    df["VIX_LogPrice"] = np.log1p(p.clip(lower=1e-9))
    return df

def add_spx_vix_ratio(df):
    """
    Adds spx_vix_ratio = spx_price / vix_price (if vix>0) in raw domain.
    """
    if "spx_price" in df.columns and "vix_price" in df.columns:
        df["spx_vix_ratio"] = df.apply(
            lambda row: row["spx_price"]/row["vix_price"] if row["vix_price"]>0 else 0.0,
            axis=1
        )
    return df

def add_time_features(df):
    """
    Adds day-of-week, hour, plus sin/cos transforms in raw domain.
    """
    df.sort_values("timestamp", inplace=True)
    df["dow"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour
    df["dow_sin"] = np.sin(2.0*np.pi*df["dow"]/7.0)
    df["dow_cos"] = np.cos(2.0*np.pi*df["dow"]/7.0)
    df["hour_sin"] = np.sin(2.0*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2.0*np.pi*df["hour"]/24.0)
    return df

def create_target_1h(df):
    """
    Creates target_1h = spx_price shifted up by 20 rows (i.e., next 1 hour if data is 3-min).
    Remains in raw domain.
    """
    if "spx_price" in df.columns:
        df["target_1h"] = df["spx_price"].shift(-20)
    else:
        logging.warning("No spx_price => cannot create target_1h.")
    return df

##############################################################################
# 4) Visualization & Correlation
##############################################################################
def visualize_correlations(df, output_png=None):
    """
    Plots a correlation heatmap in raw domain. Saves to output_png if provided,
    else shows interactively.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        logging.warning("[visualize_correlations] No numeric columns => skipping.")
        return

    corr = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="YlGnBu", annot=False)
    plt.title("Feature Correlation Heatmap (SPX & VIX)")

    if output_png:
        plt.savefig(output_png, dpi=150, bbox_inches="tight")
        logging.info(f"[visualize_correlations] Saved heatmap => {output_png}")

        s3_key = "data/lumen2/featured/spx_vix_corr_heatmap.png"
        logging.info(f"[visualize_correlations] Uploading => s3://<bucket>/{s3_key}")
        upload_file_to_s3(output_png, s3_key)
    else:
        plt.show()
    plt.close()

def rank_features_by_target_corr(df, target_col="target_1h"):
    """
    Returns a Series of absolute correlations with target_col, sorted descending,
    excluding the target_col itself.
    """
    numeric_df = df.select_dtypes(include=[float, int])
    if target_col not in numeric_df.columns:
        return None
    cmat = numeric_df.corr()
    if target_col not in cmat.columns:
        return None
    s = cmat[target_col].abs().sort_values(ascending=False)
    s = s.drop(labels=[target_col], errors="ignore")
    return s

##############################################################################
# 5) Optional correlation-based features
##############################################################################
def drop_low_corr_features(df, threshold=0.05):
    """
    Drops columns that have correlation < threshold with target_1h.
    """
    if "target_1h" not in df.columns:
        return df
    cmat = df.corr(numeric_only=True)
    if "target_1h" not in cmat.columns:
        return df

    target_corr = cmat["target_1h"].abs().sort_values(ascending=False)
    keep = target_corr[target_corr >= threshold].index.tolist()
    keep = set(keep + ["timestamp", "target_1h"])
    drop_list = [c for c in df.columns if c not in keep]
    if drop_list:
        logging.info(f"[drop_low_corr_features] Dropping => {drop_list}")
        df.drop(columns=drop_list, inplace=True)
    return df

def keep_topN_features(df, n=15, target_col="target_1h"):
    """
    Retains only the top N correlated features w.r.t. target_1h + target_1h + timestamp.
    """
    if target_col not in df.columns:
        logging.warning("[keep_topN_features] target not found => skipping.")
        return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        logging.warning("[keep_topN_features] target not numeric => skipping.")
        return df
    cmat = df.corr(numeric_only=True)
    if target_col not in cmat.columns:
        logging.warning("[keep_topN_features] target not in correlation matrix => skipping.")
        return df

    target_corr = cmat[target_col].abs().sort_values(ascending=False)
    feats_sorted = target_corr.index.tolist()
    feats_sorted.remove(target_col)
    topN = feats_sorted[:n]
    keep_set = set(topN + [target_col, "timestamp"])
    drop_list = [c for c in df.columns if c not in keep_set]
    if drop_list:
        logging.info(f"[keep_topN_features] Dropping => {drop_list}")
        df.drop(columns=drop_list, inplace=True)
    logging.info(f"[keep_topN_features] Keeping => {list(keep_set)}")
    return df

##############################################################################
# 6) Fitting Scalers Without Transforming
##############################################################################
def separate_feature_and_target_scaling(df):
    """
    Fits the feature-scaler on numeric columns (excluding target_1h),
    and fits a separate target-scaler on target_1h. Saves both as .joblib,
    but does NOT transform the DataFrame. The CSV remains in real domain.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "target_1h" not in numeric_cols:
        return df

    numeric_cols.remove("target_1h")

    # Fill any NaNs
    if df[numeric_cols].isna().any().any():
        logging.warning("[separate_feature_and_target_scaling] NaNs => filling with 0.0")
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # Fit feature scaler (no transform)
    feat_scaler = MinMaxScaler()
    feat_scaler.fit(df[numeric_cols])
    fs_path = os.path.join(SCALER_DIR, "spx_feature_scaler.joblib")
    joblib.dump(feat_scaler, fs_path)
    try:
        auto_upload_file_to_s3(fs_path, "models/lumen_2/scalers")
    except:
        pass

    # Fit target scaler (no transform)
    tgt_scaler = MinMaxScaler()
    tvals = df["target_1h"].fillna(method="ffill").fillna(method="bfill").values.reshape(-1,1)
    logging.info(f"[Scaler Validation] target_1h raw range: {tvals.min()} to {tvals.max()}")
    tgt_scaler.fit(tvals)
    ts_path = os.path.join(SCALER_DIR, "spx_target_scaler.joblib")
    joblib.dump(tgt_scaler, ts_path)
    try:
        auto_upload_file_to_s3(ts_path, "models/lumen_2/scalers")
    except:
        pass

    # Save feature names
    feats = np.array(numeric_cols, dtype=object)
    feat_path = os.path.join(SCALER_DIR, "feature_names.npy")
    np.save(feat_path, feats)

    logging.info("[separate_feature_and_target_scaling] Fitted scalers (did not transform CSV).")
    logging.info(f"[separate_feature_and_target_scaling] Final numeric cols => {feats.tolist()}")

    return df

##############################################################################
# 7) Time-based Splitting
##############################################################################
def time_series_split(df, train_frac=0.7, val_frac=0.15):
    """
    Chronological split => train, val, test.
    """
    df_sorted = df.sort_values("timestamp").copy()
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))
    tr = df_sorted.iloc[:train_end]
    va = df_sorted.iloc[train_end:val_end]
    te = df_sorted.iloc[val_end:]
    logging.info(f"[time_series_split] => Train:{tr.shape}, Val:{va.shape}, Test:{te.shape}")
    return tr, va, te

##############################################################################
# 8) Build Sequences in Chunks
##############################################################################
def create_sequences_in_chunks(df, prefix="spx", seq_len=60, chunk_size=10000):
    """
    Creates parted 3D arrays => (N, seq_len, features). Saves them as .npy,
    optionally uploads them to S3. Raw domain, no scaling done here.
    """
    timestamps = df.pop("timestamp")
    all_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "target_1h" not in all_cols:
        logging.warning("[create_sequences_in_chunks] No target_1h => skipping.")
        return

    all_cols.remove("target_1h")

    Xvals = df[all_cols].values
    Yvals = df["target_1h"].values
    n = len(Xvals)
    if n < seq_len:
        logging.warning(f"[create_sequences_in_chunks] Not enough rows => {n} < seq_len={seq_len}")
        df.insert(0, "timestamp", timestamps)
        return

    logging.info(f"[create_sequences_in_chunks] => seq_len={seq_len}, total_rows={n}")

    start = 0
    part_i = 0
    while start + seq_len <= n:
        end = min(start + chunk_size, n - seq_len + 1)
        X_list, Y_list = [], []
        for i in range(start, end):
            X_list.append(Xvals[i : i + seq_len])
            Y_list.append(Yvals[i + seq_len - 1])

        X_np = np.array(X_list, dtype=np.float32)
        Y_np = np.array(Y_list, dtype=np.float32).reshape(-1, 1)

        x_file = os.path.join(SEQUENCES_DIR, f"{prefix}_X_3D_part{part_i}.npy")
        y_file = os.path.join(SEQUENCES_DIR, f"{prefix}_Y_3D_part{part_i}.npy")

        np.save(x_file, X_np)
        np.save(y_file, Y_np)
        logging.info(f"[create_sequences_in_chunks] part={part_i}, X={X_np.shape}, Y={Y_np.shape}")

        # Optionally upload to S3
        chunk_s3 = "data/lumen2/featured/sequences"
        try:
            auto_upload_file_to_s3(x_file, chunk_s3)
            auto_upload_file_to_s3(y_file, chunk_s3)
        except Exception as e:
            logging.warning(f"chunk {part_i} => {e}")

        part_i += 1
        start = end

    df.insert(0, "timestamp", timestamps)

##############################################################################
# MAIN
##############################################################################
def main():
    logging.info("=== Feature Engineering (SPX + VIX) with Approach A (no in-place scaling) ===")

    # 1) Download processed CSVs
    download_processed_csvs()

    # 2) Merge SPX and VIX data
    df_merged = merge_spx_vix_3min()
    logging.info(f"[main] shape after merge => {df_merged.shape}")

    # 3) Add indicators/time features
    df_merged = add_spx_indicators(df_merged)
    df_merged = add_vix_indicators(df_merged)
    df_merged = add_spx_vix_ratio(df_merged)
    df_merged = add_time_features(df_merged)

    # 4) Create target_1h
    df_merged = create_target_1h(df_merged)
    df_merged.dropna(subset=["target_1h"], inplace=True)

    # 5) Visualize correlation (optional) & drop low-corr, keep top-N
    heatmap_path = os.path.join(FEATURED_DIR, "spx_vix_corr_heatmap.png")
    visualize_correlations(df_merged, output_png=heatmap_path)

    df_merged = drop_low_corr_features(df_merged, threshold=0.05)
    df_merged = keep_topN_features(df_merged, n=15, target_col="target_1h")

    # 6) Fit scalers only (leave CSV in real domain)
    df_merged = separate_feature_and_target_scaling(df_merged)

    # 7) Time-series splits
    df_train, df_val, df_test = time_series_split(df_merged, train_frac=0.7, val_frac=0.15)

    # 8) (Optional) Save these splits to CSV
    def save_csv(sub_df, name):
        out_path = os.path.join(FEATURED_DIR, name)
        sub_df.to_csv(out_path, index=False)
        logging.info(f"[save_csv] => {out_path}")
        try:
            auto_upload_file_to_s3(out_path, "data/lumen2/featured")
        except:
            pass

    save_csv(df_train, "spx_vix_train.csv")
    save_csv(df_val,   "spx_vix_val.csv")
    save_csv(df_test,  "spx_vix_test.csv")

    # 9) Create parted 3D sequences from each split
    create_sequences_in_chunks(df_train, prefix="spx_train", seq_len=60)
    create_sequences_in_chunks(df_val,   prefix="spx_val",   seq_len=60)
    create_sequences_in_chunks(df_test,  prefix="spx_test",  seq_len=60)

    logging.info("=== Done with feature engineering ===")

if __name__ == "__main__":
    main()