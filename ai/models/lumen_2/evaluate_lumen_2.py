import os
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "trained", "Lumen2.keras")
# If you've renamed your real-time test arrays to X_test.npy / y_test.npy:
X_TEST_PATH = os.path.join(BASE_DIR, "X_test.npy")  
Y_TEST_PATH = os.path.join(BASE_DIR, "y_test.npy")

def load_test_data():
    if not (os.path.exists(X_TEST_PATH) and os.path.exists(Y_TEST_PATH)):
        logging.error("X_test.npy or y_test.npy not found.")
        return None, None

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    logging.info(f"Loaded test data: X_test={X_test.shape}, y_test={y_test.shape}")
    return X_test, y_test

def evaluate_lumen2_model(model, X_test, y_test):
    if X_test is None or y_test is None:
        logging.error("No test data to evaluate.")
        return

    if len(X_test) != len(y_test):
        logging.error(f"Mismatch in sample counts: X_test={X_test.shape[0]}, y_test={y_test.shape[0]}")
        return

    expected_features = model.input_shape[-1]
    if X_test.shape[-1] != expected_features:
        diff = expected_features - X_test.shape[-1]
        if diff > 0:
            logging.warning(f"Padding X_test with {diff} missing feature(s).")
            X_test = np.pad(
                X_test, ((0,0),(0,0),(0,diff)), mode="constant", constant_values=0
            )
        else:
            logging.error(f"Feature mismatch: model expects {expected_features}, got {X_test.shape[-1]}")
            return

    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    y_pred = model.predict(X_test, verbose=0)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    logging.info("=== Evaluation Results ===")
    logging.info(f"MSE:  {mse}")
    logging.info(f"RMSE: {rmse}")
    logging.info(f"R²:   {r2}")

def main():
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found at {MODEL_PATH}")
        return

    model = load_model(MODEL_PATH)
    logging.info(f"Loaded model from {MODEL_PATH}")

    X_test, y_test = load_test_data()
    if X_test is None or y_test is None:
        logging.error("Could not load test data; aborting evaluation.")
        return

    evaluate_lumen2_model(model, X_test, y_test)

if __name__ == "__main__":
    main()