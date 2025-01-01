import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import boto3
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from tensorflow.keras.models import load_model

# Local definitions
from models.lumen_2.definitions_lumen_2 import ReduceMeanLayer
from models.lumen_2.conversation import gpt_4o_mini_response

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:3000",
    "https://lumen-1.netlify.app/"
]}}, supports_credentials=True)

Base = declarative_base()
db_url = os.getenv("DB_URL")
if not db_url:
    raise ValueError("DB_URL not set")

engine = create_engine(db_url)
Session = sessionmaker(bind=engine)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "lumen_2")

real_time_model_path = os.path.join(MODEL_DIR, "Lumen2.keras")
try:
    Lumen2_real_time = load_model(real_time_model_path, custom_objects={"ReduceMeanLayer": ReduceMeanLayer})
    logging.info("Loaded single real-time model.")
except Exception as e:
    logging.error(f"Error loading Lumen2_real_time model: {e}")
    raise

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

def download_file_from_s3(s3_key: str, local_path: str):
    bname = os.getenv("LUMEN_S3_BUCKET_NAME", "your-default-bucket")
    s3 = get_s3_client()
    if os.path.exists(local_path):
        logging.info(f"[download_file_from_s3] Removing existing => {local_path}")
        os.remove(local_path)
    logging.info(f"[download_file_from_s3] s3://{bname}/{s3_key} → {local_path}")
    s3.download_file(bname, s3_key, local_path)
    logging.info("[download_file_from_s3] Done.")

def categorize_question(msg: str):
    ml = msg.lower()
    if "spx" in ml: return "spx"
    elif "spy" in ml: return "spy"
    mk = ["price","forecast","trend","technical","valuation"]
    if any(k in ml for k in mk): return "market_analysis"
    return "general"

def classify_message(msg: str) -> dict:
    cat = categorize_question(msg)
    return {"is_stock_related": (cat in ["spx","spy","market_analysis"]),
            "classification": cat}

def get_latest_stock_price(cls: str):
    session = Session()
    metadata = MetaData()
    metadata.reflect(bind=engine)
    tbl_name = "real_time_spx" if cls == "spx" else "real_time_spy"
    if tbl_name not in metadata.tables:
        session.close()
        logging.warning(f"Table {tbl_name} not found.")
        return None
    tbl = metadata.tables[tbl_name]
    stmt = select(tbl).order_by(tbl.c.timestamp.desc()).limit(1)
    row = session.execute(stmt).mappings().first()
    session.close()
    return row["current_price"] if row else None

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        hdr = resp.headers
        hdr["Access-Control-Allow-Origin"] = "*"
        hdr["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, PUT, DELETE"
        hdr["Access-Control-Allow-Credentials"] = "true"
        hdr["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        return resp

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    msg = data.get("message", "")
    return jsonify(classify_message(msg)), 200

def load_features(cls: str) -> pd.DataFrame:
    try:
        csv_path = os.path.join(BASE_DIR, "data", "lumen_2", "featured", "spx_spy_vix_merged_features.csv")
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded {len(df)} rows from {csv_path}")
        return df
    except Exception as e:
        logging.error(f"load_features error: {e}")
        return pd.DataFrame()

def prepare_input_data(df: pd.DataFrame) -> np.ndarray:
    seq_len = 60
    if len(df) < seq_len:
        logging.warning("Not enough rows for a 60-step sequence.")
        return None
    tail = df.tail(seq_len).copy()
    drop_cols = ["timestamp","target_1h"]
    feats = [c for c in tail.columns if c not in drop_cols]
    arr = tail[feats].values
    return arr[np.newaxis,:,:]

@app.route("/conversation", methods=["POST"])
def conversation():
    data = request.get_json()
    message = data.get("message","")
    result = classify_message(message)
    if result["is_stock_related"]:
        cls = result["classification"]
        df_feat = load_features(cls)
        if df_feat.empty:
            return jsonify({"error":"No features loaded"}),500
        arr_3d = prepare_input_data(df_feat)
        if arr_3d is None:
            return jsonify({"error":"Not enough data"}),500
        model = Lumen2_real_time
        try:
            sc_path = os.path.join(MODEL_DIR,"scalers","spx_spy_vix_scaler.joblib")
            sc_obj = joblib.load(sc_path)
            shape_b4 = arr_3d.shape
            flat = arr_3d.reshape(-1, shape_b4[-1])
            scaled = sc_obj.transform(flat).reshape(shape_b4)
        except Exception as exc:
            logging.warning(f"No feature scaling: {exc}")
            scaled = arr_3d
        raw_pred = model.predict(scaled)
        val_pred = raw_pred[0][0]
        try:
            tsc_path = os.path.join(MODEL_DIR,"scalers","spx_spy_vix_target_scaler.joblib")
            tsc_obj = joblib.load(tsc_path)
            resh = raw_pred.reshape(-1,1)
            inv = tsc_obj.inverse_transform(resh)
            val_pred = inv[0][0]
        except Exception as exc:
            logging.warning(f"No target scaling: {exc}")
        return jsonify({"predicted_price":float(val_pred)}),200
    else:
        fallback = gpt_4o_mini_response(message)
        if fallback.get("success"):
            return jsonify({"response":fallback["response"]}),200
        return jsonify({"error":fallback.get("error","Unknown")}),500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)