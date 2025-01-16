import os
import sys
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import boto3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

def download_file_from_s3(s3_key: str, local_path: str):
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
    logging.info(f"[upload_file_to_s3] Uploading {local_path} → s3://{bucket_name}/{s3_key}")
    s3.upload_file(local_path, bucket_name, s3_key)
    logging.info("[upload_file_to_s3] Done.")

def auto_upload_file_to_s3(local_path: str, s3_subfolder: str = ""):
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

def download_processed_csvs():
    """
    Example function that downloads the processed CSVs from S3
    (e.g. real-time spx/vix) into DATA_DIR, but does no scaling.
    """
    spx_s3_key = "data/lumen2/processed/processed_real_time_spx.csv"
    vix_s3_key = "data/lumen2/processed/processed_real_time_vix.csv"
    spx_local = os.path.join(DATA_DIR, "processed_real_time_spx.csv")
    vix_local = os.path.join(DATA_DIR, "processed_real_time_vix.csv")
    download_file_from_s3(spx_s3_key, spx_local)
    download_file_from_s3(vix_s3_key, vix_local)

def merge_spx_vix_3min():
    """
    Simple merging/resampling of the spx and vix data. No scaling here.
    """
    spx_csv = os.path.join(DATA_DIR, "processed_real_time_spx.csv")
    vix_csv = os.path.join(DATA_DIR, "processed_real_time_vix.csv")
    logging.info(f"[merge_spx_vix_3min] Reading {spx_csv}, {vix_csv}")
    spx = pd.read_csv(spx_csv)
    vix = pd.read_csv(vix_csv)

    spx["timestamp"] = pd.to_datetime(spx["timestamp"], errors="coerce")
    vix["timestamp"] = pd.to_datetime(vix["timestamp"], errors="coerce")
    spx.sort_values("timestamp", inplace=True)
    vix.sort_values("timestamp", inplace=True)

    spx.set_index("timestamp", inplace=True)
    vix.set_index("timestamp", inplace=True)

    if "current_price" in spx.columns:
        spx.rename(columns={"current_price": "spx_price"}, inplace=True)
    if "volume" in spx.columns:
        spx.rename(columns={"volume": "spx_volume"}, inplace=True)

    vix_3m = vix.resample("3Min").last().ffill()
    if "current_price" in vix_3m.columns:
        vix_3m.rename(columns={"current_price": "vix_price"}, inplace=True)
    if "volume" in vix_3m.columns:
        vix_3m.rename(columns={"volume": "vix_volume"}, inplace=True)

    merged = spx.join(vix_3m, how="outer", rsuffix="_vix")
    merged.sort_index(inplace=True)
    merged.ffill(inplace=True)
    merged.reset_index(inplace=True)
    merged.rename(columns={"index": "timestamp"}, inplace=True)
    logging.info(f"[merge_spx_vix_3min] Final shape => {merged.shape}")
    return merged

def add_spx_indicators(df):
    """
    Example: Add SPX indicators (MACD, RSI, Bollinger, etc.).
    Still in raw domain, no scaling.
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
    Example: Add VIX indicators. No scaling here.
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
    Example ratio of spx/vix. No scaling here.
    """
    if "spx_price" in df.columns and "vix_price" in df.columns:
        df["spx_vix_ratio"] = df.apply(
            lambda row: row["spx_price"]/row["vix_price"] if row["vix_price"]>0 else 0.0,
            axis=1
        )
    return df

def add_time_features(df):
    """
    Example time-based features (day-of-week, hour, sin/cos).
    No scaling here.
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
    Example: shift spx_price -20 steps => target_1h. No scaling.
    """
    if "spx_price" in df.columns:
        df["target_1h"] = df["spx_price"].shift(-20)
    else:
        logging.warning("No spx_price => cannot create target_1h.")
    return df

def visualize_correlations(df, output_png=None):
    """
    Optional: Just a correlation heatmap in raw domain.
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
        logging.info(f"[visualize_correlations] Uploading heatmap => s3://<bucket>/{s3_key}")
        upload_file_to_s3(output_png, s3_key)
    else:
        plt.show()
    plt.close()

def drop_low_corr_features(df, threshold=0.05):
    """
    Optionally remove columns that have < threshold correlation w/ target_1h.
    No scaling here.
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
    Optionally keep top N correlated columns with target_1h. No scaling here.
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

def time_series_split(df, train_frac=0.7, val_frac=0.15):
    """
    Split the final DataFrame into train/val/test by chronological order.
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

def create_sequences_in_chunks(df, prefix="spx", seq_len=60, chunk_size=10000):
    """
    Example chunk-based approach to build 3D (N, seq_len, features) arrays from df.
    No scaling is applied here.
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

        # Save parted .npy
        x_file = os.path.join(SEQUENCES_DIR, f"{prefix}_X_3D_part{part_i}.npy")
        y_file = os.path.join(SEQUENCES_DIR, f"{prefix}_Y_3D_part{part_i}.npy")

        np.save(x_file, X_np)
        np.save(y_file, Y_np)
        logging.info(f"[create_sequences_in_chunks] part={part_i}, X={X_np.shape}, Y={Y_np.shape}")

        # Optional: upload parted files to S3
        chunk_s3 = "data/lumen2/featured/sequences"
        try:
            auto_upload_file_to_s3(x_file, chunk_s3)
            auto_upload_file_to_s3(y_file, chunk_s3)
        except Exception as e:
            logging.warning(f"chunk {part_i} => {e}")

        part_i += 1
        start = end

    # Reinsert timestamp if needed
    df.insert(0, "timestamp", timestamps)

def main():
    logging.info("=== Preprocess => Merging spx + vix (no scaling) ===")

    # 1. Download the processed CSVs (real_time_spx, real_time_vix) from S3
    download_processed_csvs()

    # 2. Merge SPX + VIX data => raw domain
    df_merged = merge_spx_vix_3min()
    logging.info(f"[main] shape after merge => {df_merged.shape}")

    # 3. Add indicators/time features => still raw domain
    df_merged = add_spx_indicators(df_merged)
    df_merged = add_vix_indicators(df_merged)
    df_merged = add_spx_vix_ratio(df_merged)
    df_merged = add_time_features(df_merged)
    df_merged = create_target_1h(df_merged)
    df_merged.dropna(subset=["target_1h"], inplace=True)

    # 4. Optional correlation heatmap => raw domain
    heatmap_path = os.path.join(FEATURED_DIR, "spx_vix_corr_heatmap.png")
    visualize_correlations(df_merged, output_png=heatmap_path)

    # 5. Drop low-corr feats or keep topN feats => no scaling
    df_merged = drop_low_corr_features(df_merged, threshold=0.05)
    df_merged = keep_topN_features(df_merged, n=15, target_col="target_1h")

    # 6. (No scaling here) => Save final CSV or parted .npy if needed
    output_path = os.path.join(FEATURED_DIR, "merged_spx_vix.csv")
    df_merged.to_csv(output_path, index=False)
    logging.info(f"[main] Saved final merged CSV => {output_path}")

    # 7. Create parted 3D sequences in raw domain if needed
    create_sequences_in_chunks(df_merged, prefix="spx_test", seq_len=60)
    logging.info("=== Done with preprocess => raw domain only ===")

if __name__ == "__main__":
    main()