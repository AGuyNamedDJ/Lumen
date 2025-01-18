import os
import sys
import logging
import numpy as np
from dotenv import load_dotenv
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
import boto3
import joblib
from sklearn.preprocessing import MinMaxScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

try:
    from ai.utils.aws_s3_utils import auto_upload_file_to_s3
except ImportError:
    logging.info("auto_upload_file_to_s3 not available. S3 uploads will be skipped.")
    auto_upload_file_to_s3 = None

# IMPORTANT: This import must match your updated create_hybrid_model
from definitions_lumen_2 import create_hybrid_model

# -----------------------------------------------------------------------------
# Environment & Logging
# -----------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "..", "..", "models", "lumen_2")
TRAINED_DIR = os.path.join(MODEL_DIR, "trained")
os.makedirs(TRAINED_DIR, exist_ok=True)

FEATURED_DIR  = os.path.join(BASE_DIR, "..", "..", "data", "lumen_2", "featured")
SEQUENCES_DIR = os.path.join(FEATURED_DIR, "sequences")
MODEL_NAME    = "Lumen2"

if not os.path.exists(SEQUENCES_DIR):
    logging.warning(f"[train_lumen_2] {SEQUENCES_DIR} not found. Did you run feature_engineering?")

# -----------------------------------------------------------------------------
# Data Generator
# -----------------------------------------------------------------------------
class NpyDataGenerator(Sequence):
    """
    Simple generator that yields batches from in-memory X, y arrays.
    """
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.X))
        batch_indices = self.indices[start:end]
        return self.X[batch_indices], self.y[batch_indices]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# -----------------------------------------------------------------------------
# Utility: find & load parted .npy sequences
# -----------------------------------------------------------------------------
def find_npy_parts(directory, prefix):
    """
    Return a sorted list of .npy files in `directory` that start with `prefix`,
    sorted by the integer found after 'part'.
    """
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith(".npy")
    ]
    def part_num(fname):
        base = os.path.basename(fname)
        segs = base.split("part")
        if len(segs) > 1:
            try:
                return int(segs[-1].replace(".npy", ""))
            except ValueError:
                return 0
        return 0
    files.sort(key=part_num)
    return files

def load_npy_data(x_prefix, y_prefix):
    """
    Loads parted .npy sequences (X and Y) from SEQUENCES_DIR,
    combining them if multiple parts are found.
    """
    x_files = find_npy_parts(SEQUENCES_DIR, x_prefix)
    y_files = find_npy_parts(SEQUENCES_DIR, y_prefix)

    if not x_files:
        logging.error(f"No X part files found for '{x_prefix}' in {SEQUENCES_DIR}.")
        return None, None
    if not y_files:
        logging.error(f"No Y part files found for '{y_prefix}' in {SEQUENCES_DIR}.")
        return None, None

    X_parts = [np.load(xf) for xf in x_files]
    Y_parts = [np.load(yf) for yf in y_files]

    X = np.concatenate(X_parts, axis=0) if len(X_parts) > 1 else X_parts[0]
    y = np.concatenate(Y_parts, axis=0) if len(Y_parts) > 1 else Y_parts[0]

    if len(X) != len(y):
        logging.error(f"Mismatch => X has {len(X)} rows, y has {len(y)} rows.")
        return None, None

    logging.info(f"Loaded X from '{x_prefix}': shape={X.shape}")
    logging.info(f"Loaded Y from '{y_prefix}': shape={y.shape}")

    if X.size > 0:
        logging.info(f"  X range => [{X.min():.4f}, {X.max():.4f}]")
    if y.size > 0:
        logging.info(f"  y range => [{y.min():.4f}, {y.max():.4f}]")

    return X, y

# -----------------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------------
def main():
    # 1. Load raw .npy data
    X_train, y_train = load_npy_data("spx_train_X_3D", "spx_train_Y_3D")
    if X_train is None or y_train is None:
        logging.error("[main] Missing or invalid train sequences => aborting.")
        return

    X_val, y_val = load_npy_data("spx_val_X_3D", "spx_val_Y_3D")
    if X_val is None or y_val is None:
        logging.error("[main] Missing or invalid val sequences => aborting.")
        return

    X_test, y_test = load_npy_data("spx_test_X_3D", "spx_test_Y_3D")
    if X_test is None or y_test is None:
        logging.warning("[main] Missing test => skipping test evaluation.")
        X_test, y_test = None, None

    # 2. Load the previously-fitted scalers (feature + target)
    fs_path = os.path.join(MODEL_DIR, "scalers", "spx_feature_scaler.joblib")
    ts_path = os.path.join(MODEL_DIR, "scalers", "spx_target_scaler.joblib")

    try:
        feat_scaler = joblib.load(fs_path)
        tgt_scaler  = joblib.load(ts_path)
        logging.info("[main] Successfully loaded feature & target scalers.")
    except Exception as e:
        logging.error(f"[main] Error loading scalers => {e}")
        return

    # 3. scale_X(...) and scale_y(...) for transforming the raw sequences
    def scale_X(X_raw):
        # shape (num_samples, seq_len, num_feats)
        num_samples, seq_len, num_feats = X_raw.shape
        X_2D = X_raw.reshape(num_samples * seq_len, num_feats)
        X_2D_scaled = feat_scaler.transform(X_2D)
        return X_2D_scaled.reshape(num_samples, seq_len, num_feats)

    def scale_y(y_raw):
        # shape (num_samples, 1)
        return tgt_scaler.transform(y_raw)

    # Scale training
    X_train_scaled = scale_X(X_train)
    y_train_scaled = scale_y(y_train)
    logging.info(f"[main] => X_train scaled range => [{X_train_scaled.min()}, {X_train_scaled.max()}]")
    logging.info(f"[main] => y_train scaled range => [{y_train_scaled.min()}, {y_train_scaled.max()}]")

    # Scale validation
    X_val_scaled, y_val_scaled = None, None
    if X_val is not None and y_val is not None:
        X_val_scaled = scale_X(X_val)
        y_val_scaled = scale_y(y_val)
        logging.info(f"[main] => X_val scaled range => [{X_val_scaled.min()}, {X_val_scaled.max()}]")
        logging.info(f"[main] => y_val scaled range => [{y_val_scaled.min()}, {y_val_scaled.max()}]")

    # Scale test
    X_test_scaled, y_test_scaled = None, None
    if X_test is not None and y_test is not None:
        X_test_scaled = scale_X(X_test)
        y_test_scaled = scale_y(y_test)
        logging.info(f"[main] => X_test scaled range => [{X_test_scaled.min()}, {X_test_scaled.max()}]")
        logging.info(f"[main] => y_test scaled range => [{y_test_scaled.min()}, {y_test_scaled.max()}]")

    # 4. Set up data generators
    batch_size = 32
    train_gen  = NpyDataGenerator(X_train_scaled, y_train_scaled, batch_size=batch_size, shuffle=True)

    val_gen = None
    if X_val_scaled is not None:
        val_gen = NpyDataGenerator(X_val_scaled, y_val_scaled, batch_size=batch_size, shuffle=False)

    seq_len   = X_train.shape[1]
    num_feats = X_train.shape[2]

    # FIX: Check if X_val_scaled is None *explicitly* for logging
    val_count = len(X_val_scaled) if X_val_scaled is not None else 0
    logging.info(f"[main] => Train samples = {len(X_train_scaled)}, Val = {val_count}")
    logging.info(f"[main] => seq_len = {seq_len}, num_feats = {num_feats}")

    # 5. Build your updated model with reduced complexity
    model = create_hybrid_model(
        input_shape=(seq_len, num_feats),
        num_lstm_layers=1,       # fewer layers
        lstm_hidden=64,          # smaller hidden dimension
        num_transformer_heads=2, # fewer attention heads
        dropout_rate=0.3,        # higher dropout
        learning_rate=3e-4       # lower LR
    )

    logging.info("Model summary =>")
    model.summary(print_fn=logging.info)

    # Callbacks
    checkpoint_path = os.path.join(TRAINED_DIR, f"{MODEL_NAME}.keras")
    chkpt_cb = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1
    )
    early_cb = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    reduce_lr_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6)

    # 6. Train
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=[chkpt_cb, early_cb, reduce_lr_cb],
        verbose=1
    )

    logging.info(f"[main] Training complete. Best model => {checkpoint_path}")

    # 7. Optionally upload the best model to S3
    if chkpt_cb.best is not None and callable(auto_upload_file_to_s3):
        try:
            auto_upload_file_to_s3(checkpoint_path, "models/lumen_2/trained")
            logging.info(f"Uploaded model => s3://<bucket>/models/lumen_2/trained/{MODEL_NAME}.keras")
        except Exception as e:
            logging.warning(f"Could not upload final model => {e}")

    # 8. Evaluate on test set if it exists
    if X_test_scaled is not None and y_test_scaled is not None:
        logging.info(f"[main] Evaluating on test => {len(X_test_scaled)} samples.")
        test_gen = NpyDataGenerator(X_test_scaled, y_test_scaled, batch_size=batch_size, shuffle=False)
        test_loss = model.evaluate(test_gen, verbose=1)
        logging.info(f"[main] Test Loss => {test_loss:.6f}")

if __name__ == "__main__":
    main()