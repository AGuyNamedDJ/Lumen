# conversation.py

import os
import logging
import random
import datetime
import dateparser
import pytz
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from tensorflow.keras.models import load_model

# If you have a local file "aws_s3_utils.py" in the same or parent directory,
# adjust the import below accordingly or remove if unused.
try:
    from utils.aws_s3_utils import download_file_from_s3
except ImportError:
    download_file_from_s3 = None
    logging.warning("download_file_from_s3 not found; skipping S3 usage.")

# definitions_lumen_2 should be in the same directory (i.e. “./definitions_lumen_2.py”).
from .definitions_lumen_2 import ReduceMeanLayer

load_dotenv()
logging.basicConfig(level=logging.INFO)
timezone = pytz.timezone("America/Chicago")

# Single real-time model references:
MODEL_S3_KEY = "models/lumen_2/trained/Lumen2.keras"
LOCAL_MODEL_FILENAME = os.path.join(os.path.dirname(__file__), "Lumen2.keras")

def ensure_lumen2_model(force_download=False):
    """Ensures a local Lumen2.keras file, optionally pulled from S3."""
    if force_download and os.path.exists(LOCAL_MODEL_FILENAME):
        logging.info(f"Removing existing => {LOCAL_MODEL_FILENAME}")
        os.remove(LOCAL_MODEL_FILENAME)

    if not os.path.exists(LOCAL_MODEL_FILENAME):
        logging.info(f"Downloading from s3://<bucket>/{MODEL_S3_KEY}")
        if not download_file_from_s3:
            raise FileNotFoundError("No S3 function and local model missing.")
        try:
            download_file_from_s3(MODEL_S3_KEY, LOCAL_MODEL_FILENAME)
        except Exception as e:
            raise FileNotFoundError(f"Failed pulling {MODEL_S3_KEY} from S3: {e}")
        logging.info("Download complete.")
    else:
        logging.info(f"{LOCAL_MODEL_FILENAME} is already present locally.")

    try:
        model = load_model(LOCAL_MODEL_FILENAME, custom_objects={"ReduceMeanLayer": ReduceMeanLayer})
        logging.info("Lumen2 single real-time model loaded.")
        return model
    except Exception as e:
        raise FileNotFoundError(f"Error loading {LOCAL_MODEL_FILENAME}: {e}")

try:
    Lumen2_model = ensure_lumen2_model(force_download=False)
except FileNotFoundError as exc:
    logging.error(str(exc))
    Lumen2_model = None

db_url = os.getenv("DB_URL")
if not db_url:
    raise ValueError("DB_URL is not set in environment.")

engine = create_engine(db_url)
Session = sessionmaker(bind=engine)

def get_db_connection():
    return Session()

def gpt_4o_mini_response(user_msg):
    """A simple fallback stub, or replace with an actual GPT call."""
    return {
        "success": True,
        "response": f"GPT fallback: {user_msg}"
    }

market_state = {
    "spx": {"current_price": None, "last_updated": None},
    "spy": {"current_price": None, "last_updated": None},
    "vix": {"current_price": None, "last_updated": None},
    "market_open": "08:30:00",
    "market_close": "15:00:00"
}

def get_table_for_symbol(sym):
    mapping = {
        "spx": "real_time_spx",
        "spy": "real_time_spy",
        "vix": "real_time_vix"
    }
    return mapping.get(sym.lower())

def update_market_state(sym):
    session = get_db_connection()
    try:
        tbl = get_table_for_symbol(sym)
        if not tbl:
            return
        q = text(f"SELECT current_price, timestamp FROM {tbl} ORDER BY timestamp DESC LIMIT 1")
        row = session.execute(q).fetchone()
        if row:
            market_state[sym]["current_price"] = row["current_price"]
            market_state[sym]["last_updated"] = row["timestamp"]
    except Exception as e:
        logging.error(f"Error updating {sym}: {e}")
    finally:
        session.close()

def get_current_time():
    return datetime.datetime.now(timezone)

def is_market_open():
    now_t = get_current_time().time()
    open_t = datetime.datetime.strptime(market_state["market_open"], "%H:%M:%S").time()
    close_t = datetime.datetime.strptime(market_state["market_close"], "%H:%M:%S").time()
    return open_t <= now_t <= close_t

def choose_random_response(lst, **kwargs):
    return random.choice(lst).format(**kwargs)

def extract_date_from_message(message):
    parsed = dateparser.parse(message, settings={"PREFER_DATES_FROM": "future"})
    if parsed:
        return parsed.date().isoformat()
    # Default to tomorrow if none found
    tmr = (get_current_time().date() + datetime.timedelta(days=1))
    return tmr.isoformat()

price_prediction_responses = [
    "Expect {symbol_upper} near ${predicted_price:.2f} on {requested_date}.",
    "We see {symbol_upper} around ${predicted_price:.2f} on {requested_date}."
]
current_price_responses = [
    "{symbol_upper} is around ${current_price:.2f} now.",
    "Currently, {symbol_upper} = ${current_price:.2f}."
]
market_hours_responses = [
    "Market is open now.",
    "Market is closed now."
]

def handle_price_prediction_for_date(symbol, requested_date):
    cp = market_state[symbol]["current_price"]
    if cp is None or not Lumen2_model:
        return f"No model or price for {symbol.upper()}"

    inp = np.array([[cp]], dtype="float32")
    pred_raw = Lumen2_model.predict(inp)
    predicted_price = float(pred_raw[0][0])
    return choose_random_response(
        price_prediction_responses,
        symbol_upper=symbol.upper(),
        predicted_price=predicted_price,
        requested_date=requested_date
    )

def handle_current_price_request(symbol):
    cp = market_state[symbol]["current_price"]
    if cp is not None:
        return choose_random_response(
            current_price_responses,
            symbol_upper=symbol.upper(),
            current_price=cp
        )
    return f"Couldn't get {symbol.upper()} price from DB."

def handle_market_hours_request():
    return market_hours_responses[0] if is_market_open() else market_hours_responses[1]

def categorize_question(msg: str):
    ml = msg.lower()
    if any(k in ml for k in ["price", "forecast", "predict", "close"]):
        return "market_analysis"
    elif any(k in ml for k in ["current", "live", "now"]):
        return "current_price"
    elif "market" in ml and any(k in ml for k in ["open", "close", "hour"]):
        return "market_hours"
    return "general"

def process_conversation(user_message: str):
    if "spy" in user_message.lower():
        sym = "spy"
    elif "vix" in user_message.lower():
        sym = "vix"
    else:
        sym = "spx"

    update_market_state(sym)
    cat = categorize_question(user_message)

    if cat == "market_analysis":
        req_date = extract_date_from_message(user_message)
        return handle_price_prediction_for_date(sym, req_date)
    elif cat == "current_price":
        return handle_current_price_request(sym)
    elif cat == "market_hours":
        return handle_market_hours_request()
    else:
        fallback = gpt_4o_mini_response(user_message)
        return fallback["response"] if fallback["success"] else "Error from GPT"

def select_model_for_prediction(is_real_time: bool):
    """We only have one real-time model => always return that."""
    logging.debug("Selecting single real-time Lumen2 model.")
    return Lumen2_model

def gpt_4o_mini_response(msg: str):
    """Stub GPT fallback."""
    return {"success": True, "response": f"GPT fallback: {msg}"}