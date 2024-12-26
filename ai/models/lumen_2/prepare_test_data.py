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
    from ai.utils.aws_s3_utils import download_file_from_s3, auto_upload_file_to_s3
except ImportError:
    download_file_from_s3 = None
    auto_upload_file_to_s3 = None
    logging.warning("AWS S3 utilities not available. No cloud downloads/uploads will occur.")

load_dotenv()
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURED_DIR = os.path.join(BASE_DIR, "../../data/lumen_2/featured")
MODEL_DIR = os.path.join(BASE_DIR, "../../models/lumen_2")

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

CSV_TMP = {k: os.path.join(FEATURED_DIR, f"{k}.csv") for k in DATA_KEYS}
SCALERS_LOCAL = {
    k: os.path.join(MODEL_DIR, "scalers", f"{k}_data_scaler.joblib") for k in DATA_KEYS
}

def download_file(key_s3, key_local):
    if not download_file_from_s3:
        logging.warning("download_file_from_s3 not available. Skipping.")
        return
    logging.info(f"Downloading {key_s3} → {key_local}")
    download_file_from_s3(key_s3, key_local)

def download_csvs():
    os.makedirs(FEATURED_DIR, exist_ok=True)
    dataframes = {}
    for k in DATA_KEYS:
        s3_key = DATA_S3[k]
        local_csv = CSV_TMP[k]
        try:
            download_file(s3_key, local_csv)
        except Exception as e:
            logging.error(f"[{k}] CSV download error: {e}")
            continue
        # Decide date col
        dt_col = "timestamp" if "real_time" in k else "date"
        try:
            with open(local_csv, "r") as f:
                snippet = f.read(2048)
                if dt_col in snippet:
                    df = pd.read_csv(local_csv, parse_dates=[dt_col])
                else:
                    logging.warning(f"[{k}] Missing '{dt_col}' in snippet. Reading plain.")
                    df = pd.read_csv(local_csv)
        except Exception:
            df = pd.read_csv(local_csv)
        dataframes[k] = df
    return dataframes

def download_scalers():
    local_scalers = {}
    os.makedirs(os.path.join(MODEL_DIR, "scalers"), exist_ok=True)
    for k in DATA_KEYS:
        try:
            remote_path = SCALERS_S3[k]
            local_path = SCALERS_LOCAL[k]
            download_file(remote_path, local_path)
            local_scalers[k] = joblib.load(local_path)
        except Exception as e:
            logging.error(f"[{k}] Scaler download error: {e}")
    return local_scalers

def unify_historical(dfmap):
    """
    For each historical DataFrame:
    - rename 'timestamp'→'date' if no 'date' column found
    - convert 'date'→datetime
    """
    hist_keys = [
        x for x in dfmap
        if any(z in x for z in [
            "historical", "consumer", "core_inflation", "cpi", "gdp",
            "industrial", "interest_rate", "labor_force", "nonfarm",
            "personal_consumption", "ppi", "unemployment_rate"
        ])
    ]
    for k in hist_keys:
        df = dfmap[k]
        if "date" not in df.columns and "timestamp" in df.columns:
            logging.info(f"[{k}] rename 'timestamp' -> 'date'")
            df.rename(columns={"timestamp": "date"}, inplace=True)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return dfmap

def prefix_columns(df, prefix):
    df.columns = [f"{prefix}_{c}" for c in df.columns]
    return df

def apply_scaler_to_df(df, scaler):
    cols = df.columns
    arr = scaler.transform(df)
    return pd.DataFrame(arr, columns=cols, index=df.index)

def apply_all_scalers(dfmap, scalers):
    for k, df in dfmap.items():
        # Identify date/timestamp col
        if "real_time" in k:
            dt_cols = ["timestamp"] if "timestamp" in df.columns else []
        else:
            dt_cols = ["date"] if "date" in df.columns else []
        # Identify possible target columns
        tcols = []
        if "close" in df.columns:
            tcols.append("close")
        if "current_price" in df.columns:
            tcols.append("current_price")
        feats = df.drop(columns=dt_cols + tcols, errors="ignore")
        scl = scalers.get(k)
        if not scl:
            logging.warning(f"[{k}] No scaler found, leaving data unscaled.")
            feats = prefix_columns(feats, k)
            # rename target columns
            for c in tcols:
                df.rename(columns={c: f"{k}_{c}"}, inplace=True)
            dfmap[k] = pd.concat([df[dt_cols], feats, df[[col for col in df.columns if col.startswith(f"{k}_")]]],
                                 axis=1)
            continue
        want_cols = getattr(scl, "feature_names_in_", feats.columns)
        feats = feats.reindex(columns=want_cols, fill_value=0)
        scaled = apply_scaler_to_df(feats, scl)
        scaled = prefix_columns(scaled, k)
        # rename target columns
        for c in tcols:
            df.rename(columns={c: f"{k}_{c}"}, inplace=True)
        dfmap[k] = pd.concat([
            df[dt_cols].reset_index(drop=True),
            scaled.reset_index(drop=True),
            df[[col for col in df.columns if col.startswith(f"{k}_")]].reset_index(drop=True)
        ], axis=1)
    return dfmap

def outer_merge(dfs, on_col):
    valid = []
    for k, df in dfs.items():
        if on_col not in df.columns:
            logging.warning(f"[{k}] missing {on_col}, skip.")
            continue
        df[on_col] = pd.to_datetime(df[on_col], errors="coerce")
        valid.append(df)
    if not valid:
        return pd.DataFrame()
    merged = reduce(lambda L, R: pd.merge(L, R, on=on_col, how="outer"), valid)
    merged.sort_values(by=on_col, inplace=True)
    merged.ffill(inplace=True)
    merged.bfill(inplace=True)
    return merged

def fix_historical_spx_close(df):
    """
    If 'historical_spx_close' is missing, but we see 'close_x' or 'close_y' 
    from merges, we unify them into 'historical_spx_close'.
    """
    # typical merges produce close_x / close_y if there's also a close from spy
    # We only fix if historical_spx_close does not exist
    if "historical_spx_close" not in df.columns:
        # check if there's a 'close_x' or 'close_y' from the historical_spx merges
        # we can attempt to unify into one column named 'historical_spx_close'
        potential = [col for col in df.columns if col.startswith("close_") or col == "close"]
        # Typically merges produce 'close_x' and 'close_y'
        # but we only unify if at least one is not all NaN
        for c in potential:
            if df[c].notna().any():
                logging.info(f"Renaming {c} → 'historical_spx_close'")
                df.rename(columns={c: "historical_spx_close"}, inplace=True)
                break
    return df

def build_sequences(df, target_col, seq_len=30, feat_list=None):
    # drop date/timestamp
    for c in ["date", "timestamp"]:
        if c in df.columns:
            df.drop(columns=c, inplace=True, errors="ignore")
    if target_col not in df.columns:
        raise ValueError(f"Missing {target_col} in {df.columns.tolist()}")
    if feat_list:
        feat_list = [f for f in feat_list if f in df.columns]
        Xdf = df[feat_list]
    else:
        Xdf = df.drop(columns=[target_col], errors="ignore")
    Xarr = Xdf.values
    yarr = df[target_col].values
    X_seq, y_seq = [], []
    for i in range(len(Xarr) - seq_len):
        X_seq.append(Xarr[i:i+seq_len])
        y_seq.append(yarr[i+seq_len])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

def main():
    if not download_file_from_s3:
        logging.warning("No S3 download; only local usage possible?")

    # 1) Download CSVs
    dfmap = download_csvs()
    # 1a) unify historical date
    dfmap = unify_historical(dfmap)

    # 2) Download scalers
    scs = download_scalers()

    # 3) Apply scalers
    dfmap = apply_all_scalers(dfmap, scs)

    # Partition sets
    hist_keys = [
        k for k in dfmap
        if any(z in k for z in [
            "historical", "consumer", "core_inflation", "cpi", "gdp",
            "industrial", "interest_rate", "labor_force", "nonfarm",
            "personal_consumption", "ppi", "unemployment_rate"
        ])
    ]
    rt_keys = [k for k in dfmap if "real_time" in k]
    hist_map = {k: dfmap[k] for k in hist_keys}
    rt_map = {k: dfmap[k] for k in rt_keys}

    hist_merged = outer_merge(hist_map, "date")
    logging.info(f"[Historical] final shape: {hist_merged.shape}")
    rt_merged = outer_merge(rt_map, "timestamp")
    logging.info(f"[Real-Time] final shape: {rt_merged.shape}")

    # Additional fix: unify 'close_x' or 'close_y' -> 'historical_spx_close' if needed
    hist_merged = fix_historical_spx_close(hist_merged)

    # 4) If historical not empty
    if not hist_merged.empty:
        target_hist = "historical_spx_close"
        try:
            feats_hist = np.load(os.path.join(MODEL_DIR, "feature_names_hist.npy"), allow_pickle=True).tolist()
        except FileNotFoundError:
            feats_hist = None

        Xh, yh = build_sequences(hist_merged, target_hist, seq_len=30, feat_list=feats_hist)
        np.save(os.path.join(MODEL_DIR, "X_test_hist.npy"), Xh)
        np.save(os.path.join(MODEL_DIR, "y_test_hist.npy"), yh)
        logging.info(f"[Historical Sequences] X={Xh.shape}, y={yh.shape}")
        # push to S3
        if auto_upload_file_to_s3:
            auto_upload_file_to_s3(os.path.join(MODEL_DIR, "X_test_hist.npy"), "models/lumen_2/trained")
            auto_upload_file_to_s3(os.path.join(MODEL_DIR, "y_test_hist.npy"), "models/lumen_2/trained")

    # 5) If real-time not empty
    if not rt_merged.empty:
        target_rt = "real_time_spx_current_price"
        try:
            feats_rt = np.load(os.path.join(MODEL_DIR, "feature_names_real.npy"), allow_pickle=True).tolist()
        except FileNotFoundError:
            feats_rt = None

        Xr, yr = build_sequences(rt_merged, target_rt, seq_len=30, feat_list=feats_rt)
        np.save(os.path.join(MODEL_DIR, "X_test_real.npy"), Xr)
        np.save(os.path.join(MODEL_DIR, "y_test_real.npy"), yr)
        logging.info(f"[Real-Time Sequences] X={Xr.shape}, y={yr.shape}")
        # push to S3
        if auto_upload_file_to_s3:
            auto_upload_file_to_s3(os.path.join(MODEL_DIR, "X_test_real.npy"), "models/lumen_2/trained")
            auto_upload_file_to_s3(os.path.join(MODEL_DIR, "y_test_real.npy"), "models/lumen_2/trained")

    logging.info("Test data preparation complete (with .npy upload to S3).")

if __name__ == "__main__":
    main()