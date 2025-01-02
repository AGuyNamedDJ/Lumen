import os
import sys
import logging
import numpy as np
from dotenv import load_dotenv
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

# Adjust Python path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

try:
    from ai.utils.aws_s3_utils import auto_upload_file_to_s3
except ImportError:
    logging.info("auto_upload_file_to_s3 not available. S3 uploads will be skipped.")
    auto_upload_file_to_s3 = None

from definitions_lumen_2 import create_hybrid_model

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

# --------------------- Data Generator --------------------- #
class NpyDataGenerator(Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end   = min(start + self.batch_size, len(self.X))
        batch_indices = self.indices[start:end]
        return self.X[batch_indices], self.y[batch_indices]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# --------------------- Find .npy part files --------------------- #
def find_npy_parts(directory, prefix):
    """
    Returns sorted .npy files in `directory` that start with `prefix`,
    sorted by integer after 'part'.
    """
    all_files = [
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
    all_files.sort(key=part_num)
    return all_files

# --------------------- Load X, y from part files --------------------- #
def load_npy_data(x_prefix, y_prefix):
    x_files = find_npy_parts(SEQUENCES_DIR, x_prefix)
    if not x_files:
        logging.error(f"No X part files for '{x_prefix}' in {SEQUENCES_DIR}.")
        return None, None

    y_files = find_npy_parts(SEQUENCES_DIR, y_prefix)
    if not y_files:
        logging.error(f"No Y part files for '{y_prefix}' in {SEQUENCES_DIR}.")
        return None, None

    X_parts = [np.load(xf) for xf in x_files]
    Y_parts = [np.load(yf) for yf in y_files]

    X = np.concatenate(X_parts, axis=0) if len(X_parts) > 1 else X_parts[0]
    y = np.concatenate(Y_parts, axis=0) if len(Y_parts) > 1 else Y_parts[0]

    if len(X) != len(y):
        logging.error(f"Mismatch: X has {len(X)} rows, y has {len(y)}.")
        return None, None

    logging.info(f"Loaded {x_prefix}: X={X.shape}, y={y.shape}")
    if X.size > 0:
        logging.info(f"  X range: [{np.min(X):.4f}, {np.max(X):.4f}]")
    if y.size > 0:
        logging.info(f"  y range: [{np.min(y):.4f}, {np.max(y):.4f}]")
    return X, y

# --------------------- Main Train Function --------------------- #
def main():
    # 1) Load train
    X_train, y_train = load_npy_data("spx_spy_vix_train_X_3D", "spx_spy_vix_train_Y_3D")
    if X_train is None or y_train is None:
        logging.error("Could not load TRAIN => abort.")
        return

    # 2) Load val
    X_val, y_val = load_npy_data("spx_spy_vix_val_X_3D", "spx_spy_vix_val_Y_3D")
    if X_val is None or y_val is None:
        logging.error("Could not load VAL => abort.")
        return

    # 3) (Optional) load test
    X_test, y_test = load_npy_data("spx_spy_vix_test_X_3D", "spx_spy_vix_test_Y_3D")
    if X_test is None or y_test is None:
        logging.warning("No TEST => skipping final test evaluation.")
        X_test, y_test = None, None

    # 4) DataGenerators
    batch_size = 32
    train_gen = NpyDataGenerator(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_gen   = NpyDataGenerator(X_val,   y_val,   batch_size=batch_size, shuffle=False)

    seq_len   = X_train.shape[1]
    num_feats = X_train.shape[2]
    logging.info(f"[main] Train samples={len(X_train)}, Val samples={len(X_val)}")
    logging.info(f"[main] seq_len={seq_len}, num_feats={num_feats}")

    # 5) Build model
    model = create_hybrid_model(
        input_shape=(seq_len, num_feats),
        num_lstm_layers=2,
        num_cnn_filters=64,
        num_transformer_heads=4,
        dropout_rate=0.2
    )
    optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    logging.info("Model summary:")
    model.summary(print_fn=logging.info)

    # Checkpoint
    checkpoint_path = os.path.join(TRAINED_DIR, f"{MODEL_NAME}.keras")
    chkpt_cb = ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1
    )
    early_cb = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    rl_cb    = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                 verbose=1, min_lr=1e-6)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=[chkpt_cb, early_cb, rl_cb],
        steps_per_epoch=None,
        validation_steps=None,
        verbose=1
    )
    logging.info(f"Training done. Best model => {checkpoint_path}")

    # S3 upload final model
    if chkpt_cb.best and callable(auto_upload_file_to_s3):
        try:
            auto_upload_file_to_s3(checkpoint_path, "models/lumen_2/trained")
            logging.info(f"Uploaded => s3://<bucket>/models/lumen_2/trained/{MODEL_NAME}.keras")
        except Exception as e:
            logging.warning(f"Upload model to S3 failed => {e}")

    # Evaluate on test if available
    if X_test is not None and y_test is not None:
        logging.info(f"Evaluating on test => {len(X_test)} samples.")
        test_gen = NpyDataGenerator(X_test, y_test, batch_size=batch_size, shuffle=False)
        test_loss = model.evaluate(test_gen, verbose=1)
        logging.info(f"Test Loss => {test_loss:.6f}")


if __name__ == "__main__":
    main()