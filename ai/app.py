import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import boto3

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import load_model

# Local definition: custom layer
from models.lumen_2.definitions_lumen_2 import ReduceMeanLayer
# Some fallback GPT logic
from models.lumen_2.conversation import gpt_4o_mini_response

##############################################################################
# ENV & LOGGING
##############################################################################
load_dotenv()
logging.basicConfig(level=logging.INFO)

##############################################################################
# FLASK SETUP
##############################################################################
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000",
                                         "https://lumen-1.netlify.app/"]}},
     supports_credentials=True)

##############################################################################
# DB SETUP (if used)
##############################################################################
Base = declarative_base()
db_url = os.getenv("DB_URL")
if not db_url:
    raise ValueError("DB_URL not set")

engine = create_engine(db_url)
Session = sessionmaker(bind=engine)

##############################################################################
# PATHS & S3
##############################################################################
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "models", "lumen_2")

# ---------------------------------------------------------------------------
# 1) Model + CSV + Scaler keys
# ---------------------------------------------------------------------------
REALTIME_MODEL_S3_KEY   = "models/lumen_2/trained/Lumen2.keras"
FEATURES_S3_KEY         = "data/lumen2/featured/spx_vix_test.csv"  
SCALER_S3_KEY           = "models/lumen_2/scalers/spx_feature_scaler.joblib"
TARGET_SCALER_S3_KEY    = "models/lumen_2/scalers/spx_target_scaler.joblib"

# ---------------------------------------------------------------------------
# 2) Local paths
# ---------------------------------------------------------------------------
LOCAL_MODEL_PATH          = os.path.join(MODEL_DIR, "Lumen2.keras")
LOCAL_FEATURES_PATH       = os.path.join(BASE_DIR, "data", "lumen_2", "featured", "spx_spy_vix_merged_features.csv")
LOCAL_SCALER_PATH         = os.path.join(MODEL_DIR, "scalers", "spx_feature_scaler.joblib")
LOCAL_TARGET_SCALER_PATH  = os.path.join(MODEL_DIR, "scalers", "spx_target_scaler.joblib")

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

def download_file_if_needed(s3_key, local_path, force=False):
    """
    Download file from s3://<bucket>/<s3_key> => local_path, removing local_path if force=True.
    """
    bucket = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
    s3 = get_s3_client()

    if force and os.path.exists(local_path):
        logging.info(f"Removing existing => {local_path}")
        os.remove(local_path)

    if not os.path.exists(local_path):
        logging.info(f"Downloading s3://{bucket}/{s3_key} → {local_path}")
        try:
            s3.download_file(bucket, s3_key, local_path)
            logging.info("Download complete.")
        except Exception as exc:
            logging.error(f"Error downloading {s3_key} => {exc}")
            raise
    else:
        logging.info(f"{local_path} already present; skipping S3 download.")

##############################################################################
# Ensure everything is local
##############################################################################
def ensure_model(force=True):
    """
    Force a fresh model download so we don't accidentally keep an older local copy.
    """
    download_file_if_needed(REALTIME_MODEL_S3_KEY, LOCAL_MODEL_PATH, force=force)

def ensure_features(force=False):
    csv_dir = os.path.dirname(LOCAL_FEATURES_PATH)
    os.makedirs(csv_dir, exist_ok=True)
    download_file_if_needed(FEATURES_S3_KEY, LOCAL_FEATURES_PATH, force=force)

def ensure_scaler(force=False):
    sc_dir = os.path.dirname(LOCAL_SCALER_PATH)
    os.makedirs(sc_dir, exist_ok=True)
    try:
        download_file_if_needed(SCALER_S3_KEY, LOCAL_SCALER_PATH, force=force)
    except Exception as exc:
        logging.warning(f"No feature scaling => {exc}")

def ensure_target_scaler(force=False):
    """
    Download the target scaler for inverse-transforming predictions.
    We'll pass force=True if we want to guarantee a fresh copy.
    """
    sc_dir = os.path.dirname(LOCAL_TARGET_SCALER_PATH)
    os.makedirs(sc_dir, exist_ok=True)
    try:
        download_file_if_needed(TARGET_SCALER_S3_KEY, LOCAL_TARGET_SCALER_PATH, force=force)
    except Exception as exc:
        logging.warning(f"No target scaler => {exc}")

##############################################################################
# LOAD MODEL ON STARTUP
##############################################################################
try:
    ensure_model(force=True)  # Force model
    Lumen2_real_time = load_model(LOCAL_MODEL_PATH, custom_objects={"ReduceMeanLayer": ReduceMeanLayer})
    logging.info("Single real-time Lumen2 model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Lumen2.keras => {e}")
    raise

##############################################################################
# HELPER: CLASSIFICATION
##############################################################################
def categorize_question(message: str):
    msg_lower = message.lower()
    if "spx" in msg_lower:
        return "spx"
    elif "spy" in msg_lower:
        return "spy"
    # If certain words => treat as market analysis
    if any(k in msg_lower for k in ["price","forecast","trend","technical","valuation"]):
        return "market_analysis"
    return "general"

def classify_message(msg: str) -> dict:
    cat = categorize_question(msg)
    is_related = cat in ["spx","spy","market_analysis"]
    return {"is_stock_related": is_related, "classification": cat}

##############################################################################
# HELPER: LOAD FEATURES
##############################################################################
def load_features(symbol: str) -> pd.DataFrame:
    ensure_features(force=False)
    if not os.path.exists(LOCAL_FEATURES_PATH):
        logging.error("Features CSV missing after ensure_features call.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(LOCAL_FEATURES_PATH)
        logging.info(f"Loaded {len(df)} rows from {LOCAL_FEATURES_PATH}")

        skip = ["timestamp", "target_1h"]
        for c in df.columns:
            if c not in skip:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Drop columns that are entirely NaN
        for c in df.columns:
            if c not in skip and df[c].isna().all():
                logging.warning(f"Dropping column '{c}' => all NaN")
                df.drop(columns=[c], inplace=True)

        return df
    except Exception as exc:
        logging.error(f"load_features error: {exc}")
        return pd.DataFrame()

##############################################################################
# HELPER: PREPARE
##############################################################################
def prepare_input_data(df: pd.DataFrame) -> (np.ndarray, list):
    """
    Takes last 60 rows => shape => (1,60,#features)
    """
    seq_len = 60
    if len(df) < seq_len:
        logging.warning("Dataframe < 60 rows => can't form a 60-step sequence.")
        return None, []

    tail = df.tail(seq_len).copy()
    skip = ["timestamp","target_1h"]
    feat_cols = [c for c in tail.columns if c not in skip]
    if not feat_cols:
        logging.error("No numeric columns remain => cannot prepare input.")
        return None, []

    arr = tail[feat_cols].values
    if np.isnan(arr).any():
        logging.warning("NaN found => filling with 0.0")
        arr = np.nan_to_num(arr, nan=0.0)

    arr_3d = arr[np.newaxis,:,:]  # => (1,60,#)
    return arr_3d, feat_cols

##############################################################################
# SCALING
##############################################################################
def apply_feature_scaling(arr_3d: np.ndarray, colnames: list) -> np.ndarray:
    if not os.path.exists(LOCAL_SCALER_PATH):
        logging.warning("Scaler missing => skipping feature scaling entirely.")
        return arr_3d

    try:
        sc = joblib.load(LOCAL_SCALER_PATH)
        if not hasattr(sc, "feature_names_in_"):
            logging.warning("Scaler has no feature_names_in_, skipping scaling.")
            return arr_3d

        scaler_cols = list(sc.feature_names_in_)
        logging.info(f"[DEBUG] Scaler expects these {len(scaler_cols)} columns => {scaler_cols}")

        tmpdf = pd.DataFrame(arr_3d[0], columns=colnames)
        logging.info(f"[DEBUG] Real-time DataFrame columns => {tmpdf.columns.tolist()}")
        logging.info(f"[DEBUG] Real-time DataFrame shape => {tmpdf.shape}")

        # Fill missing => 0.0
        for col in scaler_cols:
            if col not in tmpdf.columns:
                logging.info(f"[DEBUG] Missing column '{col}' => filling with 0.0")
                tmpdf[col] = 0.0

        # Drop extras
        drop_extras = [c for c in tmpdf.columns if c not in scaler_cols]
        if drop_extras:
            logging.info(f"[DEBUG] Dropping extra columns => {drop_extras}")
            tmpdf.drop(columns=drop_extras, inplace=True)

        # Re-order
        tmpdf = tmpdf[scaler_cols]
        logging.info(f"[DEBUG] Final DataFrame columns => {tmpdf.columns.tolist()}")
        logging.info(f"[DEBUG] Final DataFrame shape => {tmpdf.shape}")

        shape_b4 = (1, tmpdf.shape[0], tmpdf.shape[1])
        flat = tmpdf.values.reshape(-1, tmpdf.shape[1])
        scaled_flat = sc.transform(flat)
        shaped = scaled_flat.reshape(shape_b4)

        # Possibly trim if shaped features > model
        expected_feats = Lumen2_real_time.input_shape[-1]
        if shaped.shape[2] > expected_feats:
            logging.warning(f"Trimming shaped from {shaped.shape[2]} => {expected_feats}")
            shaped = shaped[:,:,:expected_feats]

        logging.info(f"[DEBUG] Final shaped for model => {shaped.shape}")
        return shaped
    except Exception as exc:
        logging.warning(f"Scaling error => skipping feature scaling => {exc}")
        return arr_3d

##############################################################################
# ROUTES
##############################################################################
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        rsp = app.make_default_options_response()
        hdr = rsp.headers
        hdr["Access-Control-Allow-Origin"] = "*"
        hdr["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, PUT, DELETE"
        hdr["Access-Control-Allow-Credentials"] = "true"
        hdr["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        return rsp

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    msg = data.get("message","")
    return jsonify(classify_message(msg)), 200

@app.route("/conversation", methods=["POST"])
def conversation():
    data = request.get_json()
    message = data.get("message", "")
    class_res = classify_message(message)

    # If not stock-related => fallback
    if not class_res["is_stock_related"]:
        fallback = gpt_4o_mini_response(message)
        if fallback.get("success"):
            return jsonify({"response":fallback["response"]}), 200
        return jsonify({"error":fallback.get("error","Unknown")}), 500

    # 1) Load features -> prepare array
    symbol = class_res["classification"]
    df_feat = load_features(symbol)
    if df_feat.empty:
        return jsonify({"error":"No features loaded"}), 500

    arr_3d, colnames = prepare_input_data(df_feat)
    if arr_3d is None:
        return jsonify({"error":"Not enough data or no numeric columns"}), 500

    # 2) Scale
    try:
        ensure_scaler(force=False)
        shaped = apply_feature_scaling(arr_3d, colnames)
    except Exception as exc:
        logging.warning(f"Skipping scaling => {exc}")
        shaped = arr_3d

    # 3) Pad/trim to match model feats
    expected_feats = Lumen2_real_time.input_shape[-1]
    actual_feats = shaped.shape[2]
    if actual_feats > expected_feats:
        logging.warning(f"Trimming from {actual_feats} feats => {expected_feats}")
        shaped = shaped[:, :, :expected_feats]
    elif actual_feats < expected_feats:
        diff = expected_feats - actual_feats
        logging.warning(f"Padding from {actual_feats} feats => {expected_feats}")
        shaped = np.pad(shaped, ((0,0),(0,0),(0,diff)), mode="constant", constant_values=0)

    # 4) Predict in scaled domain
    raw_pred = Lumen2_real_time.predict(shaped)
    out_val_scaled = float(raw_pred[0][0])
    logging.info(f"[conversation] Scaled prediction => {out_val_scaled}")

    # 5) Inverse-transform => real domain
    try:
        ensure_target_scaler(force=True)  # Force download the updated target scaler
        target_scaler = joblib.load(LOCAL_TARGET_SCALER_PATH)
        arr_scaled = np.array([[out_val_scaled]], dtype=np.float32)
        arr_real = target_scaler.inverse_transform(arr_scaled)
        real_price = float(arr_real[0][0])
        logging.info(f"[conversation] Real domain price => {real_price}")
    except Exception as e:
        logging.warning(f"No target scaling => {e}")
        real_price = out_val_scaled  # fallback

    return jsonify({
        "predicted_price": real_price,
        "scaled_value": out_val_scaled
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)