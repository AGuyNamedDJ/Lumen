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
# DB SETUP (if you use a DB)
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

# Our model & CSV on S3
REALTIME_MODEL_S3_KEY = "models/lumen_2/trained/Lumen2.keras"
FEATURES_S3_KEY       = "data/lumen2/featured/spx_spy_vix_merged_features.csv"
SCALER_S3_KEY         = "models/lumen_2/scalers/spx_spy_vix_scaler.joblib"

LOCAL_MODEL_PATH      = os.path.join(MODEL_DIR, "Lumen2.keras")
LOCAL_FEATURES_PATH   = os.path.join(BASE_DIR, "data", "lumen_2", "featured", "spx_spy_vix_merged_features.csv")
LOCAL_SCALER_PATH     = os.path.join(MODEL_DIR, "scalers", "spx_spy_vix_scaler.joblib")

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

def download_file_if_needed(s3_key, local_path, force=False):
    bucket = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
    s3 = get_s3_client()

    if force and os.path.exists(local_path):
        logging.info(f"Removing existing => {local_path}")
        os.remove(local_path)

    if not os.path.exists(local_path):
        logging.info(f"Downloading s3://{bucket}/{s3_key} → {local_path}")
        s3.download_file(bucket, s3_key, local_path)
        logging.info("Download complete.")
    else:
        logging.info(f"{local_path} already present; skipping S3 download.")

##############################################################################
# Ensure Model/CSV/Scaler are local
##############################################################################
def ensure_model(force=False):
    download_file_if_needed(REALTIME_MODEL_S3_KEY, LOCAL_MODEL_PATH, force)

def ensure_features(force=False):
    # Make sure we have the CSV
    # Also ensure local data dirs exist
    csv_dir = os.path.dirname(LOCAL_FEATURES_PATH)
    os.makedirs(csv_dir, exist_ok=True)
    download_file_if_needed(FEATURES_S3_KEY, LOCAL_FEATURES_PATH, force)

def ensure_scaler(force=False):
    # We only have a feature-scaler for now
    sc_dir = os.path.dirname(LOCAL_SCALER_PATH)
    os.makedirs(sc_dir, exist_ok=True)
    download_file_if_needed(SCALER_S3_KEY, LOCAL_SCALER_PATH, force)

##############################################################################
# LOAD MODEL ON STARTUP
##############################################################################
try:
    ensure_model(force=False)
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

        # Convert columns (except 'timestamp','target_1h') to numeric
        skip = ["timestamp","target_1h"]
        for c in df.columns:
            if c not in skip:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Possibly drop columns that are entirely NaN
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
    seq_len = 60
    if len(df) < seq_len:
        logging.warning("Dataframe < 60 rows => can't form a sequence")
        return None, []

    # Slice last 60
    tail = df.tail(seq_len).copy()
    skip = ["timestamp","target_1h"]
    feat_cols = [c for c in tail.columns if c not in skip]
    if not feat_cols:
        logging.error("No numeric columns remain => can't prepare input.")
        return None, []

    arr = tail[feat_cols].values
    # Fill any NaN => 0.0
    if np.isnan(arr).any():
        logging.warning("NaN found => filling with 0.0")
        arr = np.nan_to_num(arr, nan=0.0)

    # shape => (1,60,N)
    arr_3d = arr[np.newaxis,:,:]
    return arr_3d, feat_cols

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

    # If not stock-related => fallback to GPT
    if not class_res["is_stock_related"]:
        fallback = gpt_4o_mini_response(message)
        if fallback.get("success"):
            return jsonify({"response":fallback["response"]}),200
        return jsonify({"error":fallback.get("error","Unknown")}),500

    # Otherwise => numeric prediction path
    symbol = class_res["classification"]
    df_feat = load_features(symbol)
    if df_feat.empty:
        return jsonify({"error":"No features loaded"}),500

    arr_3d, colnames = prepare_input_data(df_feat)
    if arr_3d is None:
        return jsonify({"error":"Not enough data or no numeric columns"}),500

    # Attempt to scale with the feature scaler
    shaped = arr_3d
    try:
        ensure_scaler(force=False)
        if not os.path.exists(LOCAL_SCALER_PATH):
            logging.warning("Scaler missing => skipping feature scaling")
        else:
            sc = joblib.load(LOCAL_SCALER_PATH)

            # sc.feature_names_in_ might exist if sc was fit with column names
            scaler_cols = list(sc.feature_names_in_)
            # Rebuild a small DataFrame with the same columns:
            tmpdf = pd.DataFrame(shaped[0], columns=colnames) # shape => (60,#)
            # Add any missing columns => 0.0
            for col in scaler_cols:
                if col not in tmpdf.columns:
                    tmpdf[col] = 0.0
            # Now only keep those scaler_cols in order
            tmpdf = tmpdf[scaler_cols]

            # Flatten => transform => reshape
            shape_b4 = (1, tmpdf.shape[0], tmpdf.shape[1]) # (1,60,N?)
            flat = tmpdf.values.reshape(-1, tmpdf.shape[1])
            scaled_flat = sc.transform(flat)
            shaped = scaled_flat.reshape(shape_b4)
    except Exception as exc:
        logging.warning(f"No feature scaling => {exc}")
        shaped = arr_3d

    # 2) Predict
    raw_pred = Lumen2_real_time.predict(shaped)
    out_val = float(raw_pred[0][0])

    # 3) If you had a target-scaler => do inverse transform. But you only have
    #    spx_spy_vix_scaler.joblib in S3, so let's skip it for now.
    #    If you do create & upload a target scaler, you'd do:
    """
    try:
        target_scaler_path = os.path.join(MODEL_DIR,"scalers","spx_spy_vix_target_scaler.joblib")
        download_file_if_needed("models/lumen_2/scalers/spx_spy_vix_target_scaler.joblib", target_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        arr_resh = raw_pred.reshape(-1,1)
        inv = target_scaler.inverse_transform(arr_resh)
        out_val = float(inv[0][0])
    except Exception as exc:
        logging.warning(f"No target scaling => {exc}")
    """

    return jsonify({"predicted_price": out_val}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)