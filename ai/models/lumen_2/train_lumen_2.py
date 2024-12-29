#!/usr/bin/env python3
import os
import sys
import numpy as np
import logging
from dotenv import load_dotenv
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, 
                                       ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

# Adjust Python path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

try:
    from ai.utils.aws_s3_utils import auto_upload_file_to_s3
except ImportError:
    logging.info("auto_upload_file_to_s3 not available. Skipping S3 upload.")
    auto_upload_file_to_s3 = None

from definitions_lumen_2 import create_hybrid_model

load_dotenv()
logging.basicConfig(level=logging.INFO)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "..", "..", "models", "lumen_2")
TRAINED_DIR = os.path.join(MODEL_DIR, "trained")
os.makedirs(TRAINED_DIR, exist_ok=True)

FEATURED_DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data", "lumen_2", "featured")
SEQUENCES_DIR     = os.path.join(FEATURED_DATA_DIR, "sequences")

MODEL_NAME = "Lumen2"

if not os.path.exists(SEQUENCES_DIR):
    logging.warning(f"{SEQUENCES_DIR} does not exist. Did you run feature engineering?")


##############################################################################
# NpyDataGenerator: Memory-based generator that includes leftover partial batch
##############################################################################
class NpyDataGenerator(Sequence):
    """
    A memory-based generator that yields (X, y) batches from NumPy arrays.
    This version DOES include leftover partial batches.
    """
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        # Use ceil so partial leftover batch is included
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end   = min(start + self.batch_size, len(self.X))
        batch_indices = self.indices[start:end]
        return self.X[batch_indices], self.y[batch_indices]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def find_npy_parts(directory, prefix):
    """Locate all .npy files with a given prefix, sorted by the integer after 'part'."""
    all_files = [
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
    
    all_files.sort(key=part_num)
    return all_files


def load_npy_data(x_prefix, y_prefix):
    """Load X, y from .npy 'part' files; returns concatenated arrays."""
    x_files = find_npy_parts(SEQUENCES_DIR, x_prefix)
    if not x_files:
        logging.error(f"No X part files for '{x_prefix}' in {SEQUENCES_DIR}.")
        return None, None

    y_files = find_npy_parts(SEQUENCES_DIR, y_prefix)
    if not y_files:
        logging.error(f"No Y part files for '{y_prefix}' in {SEQUENCES_DIR}.")
        return None, None

    X_parts = [np.load(f) for f in x_files]
    Y_parts = [np.load(f) for f in y_files]

    X = np.concatenate(X_parts, axis=0) if len(X_parts) > 1 else X_parts[0]
    y = np.concatenate(Y_parts, axis=0) if len(Y_parts) > 1 else Y_parts[0]

    if len(X) != len(y):
        logging.error(f"Mismatch: X has {len(X)} rows, y has {len(y)}.")
        return None, None

    logging.info(f"Loaded {x_prefix}: X={X.shape}, y={y.shape}")
    logging.info(f"  X range: [{X.min():.4f}, {X.max():.4f}]")
    logging.info(f"  y range: [{y.min():.4f}, {y.max():.4f}]")
    return X, y


def main():
    """
    Example training routine using separate train/val/test .npy 
    from spx_spy_vix_{train,val,test} sequences in 'featured/sequences'.
    """
    # 1) Load train
    train_x_prefix = "spx_spy_vix_train_X_3D"
    train_y_prefix = "spx_spy_vix_train_Y_3D"
    X_train, y_train = load_npy_data(train_x_prefix, train_y_prefix)
    if X_train is None or y_train is None:
        logging.error("Could not load train data => Exiting.")
        return

    # 2) Load val
    val_x_prefix = "spx_spy_vix_val_X_3D"
    val_y_prefix = "spx_spy_vix_val_Y_3D"
    X_val, y_val = load_npy_data(val_x_prefix, val_y_prefix)
    if X_val is None or y_val is None:
        logging.error("Could not load val data => Exiting.")
        return

    # 3) Load test (optional)
    test_x_prefix = "spx_spy_vix_test_X_3D"
    test_y_prefix = "spx_spy_vix_test_Y_3D"
    X_test, y_test = load_npy_data(test_x_prefix, test_y_prefix)

    # 4) Build DataGenerators
    batch_size = 32
    train_gen  = NpyDataGenerator(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_gen    = NpyDataGenerator(X_val,   y_val,   batch_size=batch_size, shuffle=False)

    # IMPORTANT: let Keras derive steps from generator => steps_per_epoch=None
    # This avoids partial leftover batch issues and "End of sequence" warnings.
    steps_per_epoch      = None
    validation_steps     = None

    seq_len, num_feats   = X_train.shape[1], X_train.shape[2]

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

    checkpoint_path = os.path.join(TRAINED_DIR, f"{MODEL_NAME}.keras")
    
    # Monitor val_loss for best checkpoint
    checkpoint_cb = ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1
    )

    # Early stopping with higher patience => more epochs if improvements fluctuate
    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=1
    )

    # LR schedule for plateaus
    reduce_lr_cb = ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.5, 
        patience=3, 
        verbose=1, 
        min_lr=1e-6
    )

    # 6) Fit
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        steps_per_epoch=steps_per_epoch,        # Let Keras infer from generator
        validation_steps=validation_steps,      # Let Keras infer from generator
        verbose=1
    )

    logging.info(f"Training complete. Best model saved to {checkpoint_path}")

    # 7) Optional checkpoint upload to S3
    if checkpoint_cb.best is not None and callable(auto_upload_file_to_s3):
        try:
            auto_upload_file_to_s3(checkpoint_path, "models/lumen_2/trained")
            logging.info(f"Uploaded {checkpoint_path} to S3 => models/lumen_2/trained")
        except Exception as exc:
            logging.warning(f"Could not upload to S3: {exc}")

    # 8) Evaluate on test if found
    if X_test is not None and y_test is not None:
        logging.info(f"Evaluating on test set => {len(X_test)} samples.")
        test_gen = NpyDataGenerator(X_test, y_test, batch_size=batch_size, shuffle=False)
        test_loss  = model.evaluate(test_gen, verbose=1) 
        logging.info(f"Test Loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()