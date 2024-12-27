import os
import sys
import logging
import numpy as np
import boto3
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

# ------------------------------------------------------------------------
# If you need a custom layer from definitions_lumen_2:
# ------------------------------------------------------------------------
try:
    from definitions_lumen_2 import ReduceMeanLayer
except ImportError:
    # If definitions_lumen_2 is not accessible or no custom layer is needed
    class ReduceMeanLayer:
        pass

# ------------------------------------------------------------------------
# Load environment variables + set up logging
# ------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------
# Hard-coded region us-east-2; update if you truly use a different region
# ------------------------------------------------------------------------
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = "us-east-2"
    )

def download_file_from_s3(s3_key: str, local_path: str):
    """Download a file from S3 to local filesystem."""
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "lumenaibucket")
    s3 = get_s3_client()
    try:
        logging.info(f"Downloading s3://{bucket_name}/{s3_key} → {local_path}")
        s3.download_file(bucket_name, s3_key, local_path)
        logging.info(f"Downloaded s3://{bucket_name}/{s3_key} → {local_path}")
    except Exception as e:
        logging.error(f"Error downloading {s3_key} from S3: {e}")
        raise

def upload_file_to_s3(local_path: str, s3_key: str):
    """Upload a local file to the S3 bucket/key specified."""
    bucket_name = os.getenv("LUMEN_S3_BUCKET_NAME", "lumenaibucket")
    s3 = get_s3_client()
    try:
        logging.info(f"Uploading {local_path} → s3://{bucket_name}/{s3_key}")
        s3.upload_file(local_path, bucket_name, s3_key)
        logging.info(f"Uploaded {local_path} → s3://{bucket_name}/{s3_key}")
    except Exception as e:
        logging.error(f"Error uploading {local_path} to s3://{bucket_name}/{s3_key}: {e}")
        raise

# ------------------------------------------------------------------------
# Local + S3 file references
# ------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

# We'll store everything in 'trained' subfolder for consistency
TRAINED_DIR         = os.path.join(script_dir, "trained")
MODEL_PATH          = os.path.join(TRAINED_DIR, "Lumen2.keras")  # single real-time model
LOCAL_X_TEST_REAL   = os.path.join(TRAINED_DIR, "X_test_real.npy")  # real-time test data
LOCAL_PRED_REAL     = os.path.join(TRAINED_DIR, "predictions_real.npy")  # predictions

# Adjust these S3 keys if you want the script to automatically pull/push them:
S3_X_TEST_REAL      = "models/lumen_2/trained/X_test_real.npy"
S3_PREDICTIONS_REAL = "models/lumen_2/trained/predictions_real.npy"

def maybe_download_test_data():
    """Ensure the real-time test array (X_test_real.npy) is present locally by pulling from S3."""
    os.makedirs(TRAINED_DIR, exist_ok=True)
    if not os.path.exists(LOCAL_X_TEST_REAL):
        try:
            download_file_from_s3(S3_X_TEST_REAL, LOCAL_X_TEST_REAL)
        except Exception as exc:
            logging.error(f"Could not download {S3_X_TEST_REAL} from S3: {exc}")
    else:
        logging.info(f"Real-time test data already exists locally: {LOCAL_X_TEST_REAL}")

def load_trained_model(model_path):
    """Load your single Lumen2 model with any needed custom objects."""
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}")
        return None
    try:
        model = load_model(model_path, custom_objects={'ReduceMeanLayer': ReduceMeanLayer})
        logging.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def main():
    # 1) Possibly download the test data from S3
    maybe_download_test_data()

    # 2) Load the trained model
    model = load_trained_model(MODEL_PATH)
    if model is None:
        logging.error("Could not load real-time Lumen2 model. Exiting.")
        return

    # 3) Check if local X_test_real exists now
    if not os.path.exists(LOCAL_X_TEST_REAL):
        logging.error(f"No real-time test data found at {LOCAL_X_TEST_REAL}. Exiting.")
        return

    # 4) Load the real-time test data
    X_test = np.load(LOCAL_X_TEST_REAL)
    logging.info(f"Loaded real-time test data: shape={X_test.shape}")

    # 5) Compare to the model’s expected shape
    exp_timesteps = model.input_shape[1]
    exp_features  = model.input_shape[2]
    logging.info(f"Model expects shape: (batch, {exp_timesteps}, {exp_features})")

    actual_feats = X_test.shape[2]
    if actual_feats < exp_features:
        diff = exp_features - actual_feats
        logging.warning(f"Padding X_test from {actual_feats} → {exp_features} features with zeros.")
        X_test = np.pad(X_test, ((0,0), (0,0), (0,diff)), mode="constant", constant_values=0)
    elif actual_feats > exp_features:
        diff = actual_feats - exp_features
        logging.warning(f"Trimming X_test from {actual_feats} → {exp_features} features.")
        X_test = X_test[:, :, :exp_features]

    # 6) Predict
    logging.info("Predicting on real-time test data...")
    X_test = X_test.astype("float32")
    preds = model.predict(X_test, verbose=0)
    logging.info(f"Predictions shape: {preds.shape}")

    # 7) Save predictions locally
    np.save(LOCAL_PRED_REAL, preds)
    logging.info(f"Saved predictions to {LOCAL_PRED_REAL}")

    # 8) Optionally upload predictions back to S3
    try:
        upload_file_to_s3(LOCAL_PRED_REAL, S3_PREDICTIONS_REAL)
    except Exception as exc:
        logging.warning(f"Could not upload predictions to S3: {exc}")

if __name__ == "__main__":
    main()