import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from functools import reduce
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

try:
    from ai.utils.aws_s3_utils import download_file_from_s3
except ImportError:
    logging.error("download_file_from_s3 not available, cannot pull from S3. Exiting.")
    sys.exit(1)

load_dotenv()
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURED_DIR = os.path.join(BASE_DIR, "../../data/lumen_2/featured")
MODEL_DIR = os.path.join(BASE_DIR, "../../models/lumen_2")

###############################################################################
# Data Keys, CSVs, Scalers in S3
###############################################################################
DATA_KEYS = [
    "consumer_confidence", "consumer_sentiment", "core_inflation", "cpi", "gdp",
    "industrial_production", "interest_rate", "labor_force", "nonfarm_payroll",
    "personal_consumption", "ppi", "unemployment_rate",
    "historical_spx", "historical_spy", "historical_vix",
    "real_time_spx", "real_time_spy", "real_time_vix"
]

DATA_S3 = {
    "consumer_confidence":      "data/lumen2/featured/featured_consumer_confidence_data.csv",
    "consumer_sentiment":       "data/lumen2/featured/featured_consumer_sentiment_data.csv",
    "core_inflation":           "data/lumen2/featured/featured_core_inflation_data.csv",
    "cpi":                      "data/lumen2/featured/featured_cpi_data.csv",
    "gdp":                      "data/lumen2/featured/featured_gdp_data.csv",
    "industrial_production":    "data/lumen2/featured/featured_industrial_production_data.csv",
    "interest_rate":            "data/lumen2/featured/featured_interest_rate_data.csv",
    "labor_force":              "data/lumen2/featured/featured_labor_force_participation_rate_data.csv",
    "nonfarm_payroll":          "data/lumen2/featured/featured_nonfarm_payroll_employment_data.csv",
    "personal_consumption":     "data/lumen2/featured/featured_personal_consumption_expenditures_data.csv",
    "ppi":                      "data/lumen2/featured/featured_ppi_data.csv",
    "unemployment_rate":        "data/lumen2/featured/featured_unemployment_rate_data.csv",
    "historical_spx":           "data/lumen2/featured/featured_historical_spx.csv",
    "historical_spy":           "data/lumen2/featured/featured_historical_spy.csv",
    "historical_vix":           "data/lumen2/featured/featured_historical_vix.csv",
    "real_time_spx":            "data/lumen2/featured/featured_real_time_spx.csv",
    "real_time_spy":            "data/lumen2/featured/featured_real_time_spy.csv",
    "real_time_vix":            "data/lumen2/featured/featured_real_time_vix.csv",
}

# Local ephemeral CSV paths
CSV_TMP = {
    k: os.path.join(FEATURED_DIR, f"{k}.csv") for k in DATA_KEYS
}

SCALERS_S3 = {
    "consumer_confidence":      "models/lumen_2/scalers/consumer_confidence_data_scaler.joblib",
    "consumer_sentiment":       "models/lumen_2/scalers/consumer_sentiment_data_scaler.joblib",
    "core_inflation":           "models/lumen_2/scalers/core_inflation_data_scaler.joblib",
    "cpi":                      "models/lumen_2/scalers/cpi_data_scaler.joblib",
    "gdp":                      "models/lumen_2/scalers/gdp_data_scaler.joblib",
    "industrial_production":    "models/lumen_2/scalers/industrial_production_data_scaler.joblib",
    "interest_rate":            "models/lumen_2/scalers/interest_rate_data_scaler.joblib",
    "labor_force":              "models/lumen_2/scalers/labor_force_participation_rate_data_scaler.joblib",
    "nonfarm_payroll":          "models/lumen_2/scalers/nonfarm_payroll_employment_data_scaler.joblib",
    "personal_consumption":     "models/lumen_2/scalers/personal_consumption_expenditures_data_scaler.joblib",
    "ppi":                      "models/lumen_2/scalers/ppi_data_scaler.joblib",
    "unemployment_rate":        "models/lumen_2/scalers/unemployment_rate_data_scaler.joblib",
    "historical_spx":           "models/lumen_2/scalers/historical_spx_scaler.joblib",
    "historical_spy":           "models/lumen_2/scalers/historical_spy_scaler.joblib",
    "historical_vix":           "models/lumen_2/scalers/historical_vix_scaler.joblib",
    "real_time_spx":            "models/lumen_2/scalers/real_time_spx_scaler.joblib",
    "real_time_spy":            "models/lumen_2/scalers/real_time_spy_scaler.joblib",
    "real_time_vix":            "models/lumen_2/scalers/real_time_vix_scaler.joblib",
}

SCALERS_LOCAL = {
    k: os.path.join(MODEL_DIR, "scalers", f"{k}_data_scaler.joblib") for k in DATA_KEYS
}


def download_csvs_from_s3():
    """Download each CSV from S3 into FEATURED_DIR, parse date/timestamp if possible."""
    if not os.path.exists(FEATURED_DIR):
        os.makedirs(FEATURED_DIR, exist_ok=True)

    dataframes = {}
    for key in DATA_KEYS:
        s3_key = DATA_S3[key]
        local_csv = CSV_TMP[key]

        logging.info(f"[{key}] Downloading CSV from S3 → {local_csv}")
        try:
            download_file_from_s3(s3_key, local_csv)
        except Exception as e:
            logging.error(f"[{key}] Could not download CSV from S3: {e}")
            continue

        # Attempt parse
        if "real_time" in key:
            dt_col = "timestamp"
        else:
            dt_col = "date"

        with open(local_csv, "r") as fcheck:
            snippet = fcheck.read(4096)
            if dt_col not in snippet:
                logging.warning(f"[{key}] Column '{dt_col}' not found. Reading w/o parse_dates.")
                df = pd.read_csv(local_csv)
            else:
                try:
                    df = pd.read_csv(local_csv, parse_dates=[dt_col])
                except Exception:
                    logging.warning(f"[{key}] parse_dates failed on '{dt_col}'. Reading fallback.")
                    df = pd.read_csv(local_csv)

        dataframes[key] = df

    return dataframes


def download_scalers_from_s3():
    """Download each scaler from S3 into local MODEL_DIR/scalers, then load them."""
    local_scaler_dir = os.path.join(MODEL_DIR, "scalers")
    os.makedirs(local_scaler_dir, exist_ok=True)

    scalers = {}
    for key in DATA_KEYS:
        s3_path = SCALERS_S3[key]
        local_path = SCALERS_LOCAL[key]
        logging.info(f"[{key}] Downloading scaler from S3 → {local_path}")
        try:
            download_file_from_s3(s3_path, local_path)
            scalers[key] = joblib.load(local_path)
        except Exception as e:
            logging.error(f"[{key}] Could not download scaler from S3: {e}")
    return scalers


def unify_historical_columns(dataframes):
    """
    For each historical DataFrame:
      - If 'timestamp' is present but 'date' is missing, rename 'timestamp'->'date'.
      - Then forcibly convert 'date' to datetime64[ns].
    """
    hist_keys = [
        k for k in dataframes
        if any(x in k for x in [
            "historical", "consumer", "core_inflation", "cpi", "gdp",
            "industrial", "interest_rate", "labor_force", "nonfarm",
            "personal_consumption", "ppi", "unemployment_rate"
        ])
    ]

    for k in hist_keys:
        df = dataframes[k]
        # If 'date' not in columns but 'timestamp' is, rename
        if "date" not in df.columns and "timestamp" in df.columns:
            logging.info(f"[{k}] Renaming 'timestamp' -> 'date' for historical unify.")
            df.rename(columns={"timestamp": "date"}, inplace=True)

        if "date" in df.columns:
            # Force convert to datetime
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            logging.warning(f"[{k}] No 'date' or 'timestamp' columns found after unify.")
    return dataframes


def force_datetime(df, colname):
    """Ensure colname is datetime64[ns] if present."""
    if colname in df.columns:
        df[colname] = pd.to_datetime(df[colname], errors="coerce")


def apply_scalers(dataframes, scalers):
    """
    Align each DataFrame with its associated scaler if found, scaling the features,
    and prefixing columns with the dataset key.
    """
    for key, df in dataframes.items():
        # decide dt col
        if "real_time" in key:
            dt_cols = []
            if "timestamp" in df.columns:
                dt_cols = ["timestamp"]
                force_datetime(df, "timestamp")
        else:
            dt_cols = []
            if "date" in df.columns:
                dt_cols = ["date"]
                force_datetime(df, "date")

        # Identify target(s)
        target_cols = []
        if "close" in df.columns:
            target_cols.append("close")
        if "current_price" in df.columns:
            target_cols.append("current_price")

        df_targets = df[target_cols] if target_cols else pd.DataFrame(index=df.index)

        # remove dt cols + target cols from feature set
        df_features = df.drop(columns=(dt_cols + target_cols), errors="ignore")

        scaler = scalers.get(key, None)
        if not scaler:
            logging.warning(f"[{key}] No scaler found; leaving columns as-is.")
            df_features.columns = [f"{key}_{c}" for c in df_features.columns]
            if not df_targets.empty:
                df_targets.columns = [f"{key}_{c}" for c in df_targets.columns]

            dataframes[key] = pd.concat(
                [
                    df[dt_cols].reset_index(drop=True),  # keep dt if present
                    df_features.reset_index(drop=True),
                    df_targets.reset_index(drop=True),
                ],
                axis=1,
            )
            continue

        # if scaler found, align columns
        expected_cols = getattr(scaler, "feature_names_in_", df_features.columns)
        df_features = df_features.reindex(columns=expected_cols, fill_value=0)

        # scale
        scaled_vals = scaler.transform(df_features)
        df_scaled = pd.DataFrame(scaled_vals, columns=expected_cols)
        df_scaled.columns = [f"{key}_{c}" for c in df_scaled.columns]

        # rename targets
        if not df_targets.empty:
            df_targets.columns = [f"{key}_{c}" for c in df_targets.columns]

        dataframes[key] = pd.concat(
            [
                df[dt_cols].reset_index(drop=True),
                df_scaled.reset_index(drop=True),
                df_targets.reset_index(drop=True),
            ],
            axis=1,
        )
    return dataframes


def merge_dataframes(frames_dict, on_col):
    """
    Outer-merge all frames on `on_col`. If a frame doesn't have that column at all,
    skip it with a warning. Also ensure all 'on_col' are datetime dtype.
    """
    valid_frames = []
    for k, df in frames_dict.items():
        if on_col not in df.columns:
            logging.warning(f"[{k}] Missing merge col '{on_col}'. Skipping from merge.")
            continue
        # Force col to datetime to avoid object/datetime conflicts
        df[on_col] = pd.to_datetime(df[on_col], errors="coerce")
        valid_frames.append(df)

    if not valid_frames:
        logging.warning(f"No valid frames with '{on_col}' to merge on. Returning empty.")
        return pd.DataFrame()

    # now reduce-merge
    merged = reduce(lambda L, R: pd.merge(L, R, on=on_col, how="outer"), valid_frames)
    # sort
    if on_col in merged.columns:
        # ensure it's datetime
        merged[on_col] = pd.to_datetime(merged[on_col], errors="coerce")
        merged.sort_values(by=on_col, inplace=True)
        merged.ffill(inplace=True)
        merged.bfill(inplace=True)
    return merged


def prepare_data_for_lstm(df, target_col, seq_len=30, feature_list=None):
    """
    Build sequences of length `seq_len`. Optionally restrict columns to `feature_list`.
    """
    df = df.copy()
    # drop date / timestamp if present
    for c in ["date", "timestamp"]:
        if c in df.columns:
            df.drop(columns=c, inplace=True, errors="ignore")

    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' missing from columns {df.columns.tolist()}")

    if feature_list is not None:
        feature_list = [f for f in feature_list if f in df.columns]
        df_features = df[feature_list]
    else:
        df_features = df.drop(columns=[target_col], errors="ignore")

    X_all = df_features.values
    y_all = df[target_col].values

    if np.isnan(X_all).any() or np.isinf(X_all).any():
        raise ValueError(f"NaN/Inf found in features for target '{target_col}'")
    if np.isnan(y_all).any() or np.isinf(y_all).any():
        raise ValueError(f"NaN/Inf found in target '{target_col}'")

    X_seq, y_seq = [], []
    for i in range(len(X_all) - seq_len):
        X_seq.append(X_all[i : i + seq_len])
        y_seq.append(y_all[i + seq_len])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def main():
    # 1) Download CSVs from S3
    dfs = download_csvs_from_s3()
    # 1a) unify columns for historical sets
    dfs = unify_historical_columns(dfs)

    # 2) Download scalers
    scalers = download_scalers_from_s3()

    # 3) Apply scalers
    dfs = apply_scalers(dfs, scalers)

    # Partition sets
    hist_keys = [
        k for k in dfs
        if any(x in k for x in [
            "historical", "consumer", "core_inflation", "cpi", "gdp",
            "industrial", "interest_rate", "labor_force", "nonfarm",
            "personal_consumption", "ppi", "unemployment_rate"
        ])
    ]
    rt_keys = [k for k in dfs if "real_time" in k]

    hist_dfs = {k: dfs[k] for k in hist_keys}
    rt_dfs   = {k: dfs[k] for k in rt_keys}

    # 4) Merge historical on 'date'
    combined_hist = merge_dataframes(hist_dfs, on_col="date")
    logging.info(f"[Historical] final shape: {combined_hist.shape}")

    # 5) Merge real-time on 'timestamp'
    combined_rt = merge_dataframes(rt_dfs, on_col="timestamp")
    logging.info(f"[Real-Time] final shape: {combined_rt.shape}")

    # Example: build sequences for historical
    if not combined_hist.empty:
        target_hist = "historical_spx_close"
        seq_len = 30
        try:
            f_hist = np.load(os.path.join(MODEL_DIR, "feature_names_hist.npy"), allow_pickle=True)
            feature_hist = f_hist.tolist() if isinstance(f_hist, np.ndarray) else None
        except FileNotFoundError:
            feature_hist = None

        X_hist, y_hist = prepare_data_for_lstm(combined_hist, target_hist, seq_len, feature_list=feature_hist)
        np.save(os.path.join(MODEL_DIR, "X_test_hist.npy"), X_hist)
        np.save(os.path.join(MODEL_DIR, "y_test_hist.npy"), y_hist)
        logging.info(f"[Historical Sequences] X={X_hist.shape}, y={y_hist.shape}")

    # Example: build sequences for real-time
    if not combined_rt.empty:
        target_rt = "real_time_spx_current_price"
        seq_len = 30
        try:
            f_rt = np.load(os.path.join(MODEL_DIR, "feature_names_real.npy"), allow_pickle=True)
            feature_rt = f_rt.tolist() if isinstance(f_rt, np.ndarray) else None
        except FileNotFoundError:
            feature_rt = None

        X_rt, y_rt = prepare_data_for_lstm(combined_rt, target_rt, seq_len, feature_list=feature_rt)
        np.save(os.path.join(MODEL_DIR, "X_test_real.npy"), X_rt)
        np.save(os.path.join(MODEL_DIR, "y_test_real.npy"), y_rt)
        logging.info(f"[Real-Time Sequences] X={X_rt.shape}, y={y_rt.shape}")

    logging.info("Test data preparation from S3 complete!")


if __name__ == "__main__":
    main()