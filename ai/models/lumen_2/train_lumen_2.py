import sys
import os
import numpy as np
import logging
from dotenv import load_dotenv
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

# Adjust your Python path to see ai/utils, definitions, etc.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

try:
    from ai.utils.aws_s3_utils import auto_upload_file_to_s3
except ImportError:
    logging.info("auto_upload_file_to_s3 not available. Skipping S3 upload.")

from definitions_lumen_2 import create_hybrid_model

# Environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "models", "lumen_2")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAINED_DIR = os.path.join(MODEL_DIR, "trained")
os.makedirs(TRAINED_DIR, exist_ok=True)

MODEL_NAME = "Lumen2"

FEATURED_DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data", "lumen_2", "featured")
SEQUENCES_DIR = os.path.join(FEATURED_DATA_DIR, "sequences")

if not os.path.exists(FEATURED_DATA_DIR):
    raise FileNotFoundError(f"{FEATURED_DATA_DIR} does not exist. Check your data paths!")
if not os.path.exists(SEQUENCES_DIR):
    logging.warning(f"{SEQUENCES_DIR} does not exist. Did you run feature engineering?")

print("Files in featured directory:", os.listdir(FEATURED_DATA_DIR))
if os.path.exists(SEQUENCES_DIR):
    print("Files in sequences directory:", os.listdir(SEQUENCES_DIR))
else:
    print("Sequences directory not found or empty.")


class NpyDataGenerator(Sequence):
    """
    Loads entire X, y in memory, yields mini-batches.
    Leftover partial batch is effectively dropped if steps_per_epoch < len(X)/batch_size.
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
        # We'll let Keras rely on steps_per_epoch, so __len__ can just be the full # of batches
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.X[batch_indices], self.y[batch_indices]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def find_npy_parts(directory, prefix):
    """Finds all .npy with a given prefix, sorts them by 'partN' integer."""
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith(".npy")
    ]

    def part_num(fname):
        base = os.path.basename(fname)
        segments = base.split("part")
        if len(segments) > 1:
            num_str = segments[-1].replace(".npy", "")
            try:
                return int(num_str)
            except ValueError:
                return 0
        return 0

    files.sort(key=part_num)
    return files


def load_npy_data(x_prefix, y_prefix):
    """
    Loads X, y from .npy part files in SEQUENCES_DIR, checks for shape consistency, logs debug info.
    """
    x_files = find_npy_parts(SEQUENCES_DIR, x_prefix)
    if not x_files:
        logging.error(f"No X part files for '{x_prefix}' in {SEQUENCES_DIR}. Exiting.")
        return None, None

    y_files = find_npy_parts(SEQUENCES_DIR, y_prefix)
    if not y_files:
        single_y = os.path.join(SEQUENCES_DIR, f"{y_prefix}.npy")
        if os.path.exists(single_y):
            y_files = [single_y]
        else:
            logging.error(f"No Y part files for '{y_prefix}' in {SEQUENCES_DIR}. Exiting.")
            return None, None

    # Concatenate any multi-part X
    X_parts = [np.load(xf) for xf in x_files]
    X = np.concatenate(X_parts, axis=0) if len(X_parts) > 1 else X_parts[0]

    # Concatenate any multi-part Y
    Y_parts = [np.load(yf) for yf in y_files]
    y = np.concatenate(Y_parts, axis=0) if len(Y_parts) > 1 else Y_parts[0]

    # Check consistency
    if len(X) != len(y):
        logging.error(f"Mismatch X vs y count: {len(X)} vs {len(y)}. Exiting.")
        return None, None

    logging.info(f"Loaded X={X.shape}, y={y.shape}")
    logging.info(f"X range: min={X.min():.4f}, max={X.max():.4f}")
    logging.info(f"y range: min={y.min():.4f}, max={y.max():.4f}")

    return X, y


def main():
    """Trains the hybrid model with a memory-based generator, ensuring consistent steps_per_epoch."""
    # Decide which files to load: e.g. real_time_spy_X_3D / real_time_spy_Y_3D
    x_prefix = "real_time_spy_X_3D"
    y_prefix = "real_time_spy_Y_3D"

    X, y = load_npy_data(x_prefix, y_prefix)
    if X is None or y is None:
        print("Could not load data. Verify .npy files in the sequences folder.")
        return

    seq_len = X.shape[1]
    num_feats = X.shape[2]
    logging.info(f"Sequence length={seq_len}, Features={num_feats}, Samples={len(X)}")

    # Build generator
    batch_size = 32
    train_generator = NpyDataGenerator(X, y, batch_size=batch_size, shuffle=True)

    # steps_per_epoch to fully utilize the data
    steps_per_epoch = len(X) // batch_size  # floor division
    logging.info(f"steps_per_epoch={steps_per_epoch} for {len(X)} samples @ batch_size={batch_size}")

    # Build model
    model = create_hybrid_model(
        input_shape=(seq_len, num_feats),
        num_lstm_layers=2,
        num_cnn_filters=64,
        num_transformer_heads=4,
        dropout_rate=0.2
    )
    optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mean_squared_error")

    # Logging
    logging.info("Model summary:")
    model.summary(print_fn=logging.info)

    # Checkpoints
    checkpoint_path = os.path.join(TRAINED_DIR, f"{MODEL_NAME}.keras")
    checkpoint_cb = ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor="loss",
        mode="min",
        verbose=1
    )
    early_stopping_cb = EarlyStopping(
        monitor="loss",
        patience=5,
        verbose=1
    )

    # Fit
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=20,
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=1
    )

    logging.info(f"Training complete. Model saved to {checkpoint_path}")

    # Optionally upload checkpoint to S3
    if checkpoint_cb.best is not None:
        if auto_upload_file_to_s3 and callable(auto_upload_file_to_s3):
            try:
                auto_upload_file_to_s3(checkpoint_path, "models/lumen_2/trained")
                logging.info(f"Uploaded {checkpoint_path} to S3 => models/lumen_2/trained")
            except Exception as exc:
                logging.warning(f"Could not upload to S3: {exc}")
    else:
        logging.info("No best checkpoint found to upload.")

if __name__ == "__main__":
    main()