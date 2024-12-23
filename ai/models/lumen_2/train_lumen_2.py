import os
import numpy as np
import logging
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from dotenv import load_dotenv

# Make sure definitions_lumen_2.py has the correct final Dense(...) 
# If single-horizon => Dense(1, activation='linear')
# If multi-horizon => Dense(5, activation='linear'), etc.
from definitions_lumen_2 import create_hybrid_model

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, '..', '..', 'models', 'lumen_2')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

MODEL_NAME = 'Lumen2'

# This is still your overall "featured" dir
FEATURED_DATA_DIR = os.path.join(BASE_DIR, '../../data/lumen_2/featured')
if not os.path.exists(FEATURED_DATA_DIR):
    raise FileNotFoundError(f"The directory {FEATURED_DATA_DIR} does not exist.")

# But now we also define the subdirectory where .npy parts reside
SEQUENCES_DIR = os.path.join(FEATURED_DATA_DIR, 'sequences')
if not os.path.exists(SEQUENCES_DIR):
    logging.warning(f"Sequences directory {SEQUENCES_DIR} does not exist. Did you run feature engineering?")


print("Files in featured directory:", os.listdir(FEATURED_DATA_DIR))
if os.path.exists(SEQUENCES_DIR):
    print("Files in sequences subdirectory:", os.listdir(SEQUENCES_DIR))
else:
    print("Sequences subdirectory does not exist.")


class NpyDataGenerator(Sequence):
    """
    Simple generator that loads entire X, y in memory, then yields mini-batches.
    Leftover partial batch is effectively dropped because we'll set steps_per_epoch
    to num_samples // batch_size.
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
        """
        Return the total number of batches if we used all data. But for Approach #1,
        we won't rely on this to define steps_per_epoch. 
        """
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def find_npy_parts(directory, prefix):
    """
    Finds all parts (part0, part1, ...) for a given prefix in the .npy filenames,
    then sorts them in ascending order by part index.
    """
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith('.npy')
    ]

    def part_num(fname):
        base = os.path.basename(fname)
        segments = base.split('part')
        if len(segments) > 1:
            num_str = segments[-1].replace('.npy', '')
            try:
                return int(num_str)
            except ValueError:
                return 0
        return 0

    files.sort(key=part_num)
    return files


def load_npy_data(x_prefix, y_prefix):
    """
    Loads and concatenates all partial .npy files for X and Y from SEQUENCES_DIR,
    ensuring shapes match and checking for NaNs or infs.
    """
    x_files = find_npy_parts(SEQUENCES_DIR, x_prefix)
    if len(x_files) == 0:
        logging.error(f"No X files found with prefix '{x_prefix}' in {SEQUENCES_DIR}")
        return None, None

    y_files = find_npy_parts(SEQUENCES_DIR, y_prefix)
    if len(y_files) == 0:
        # Maybe there's a single .npy that doesn't have 'part0', etc.
        single_y_file = os.path.join(SEQUENCES_DIR, f"{y_prefix}.npy")
        if os.path.exists(single_y_file):
            y_files = [single_y_file]
        else:
            logging.error(f"No Y files found with prefix '{y_prefix}', "
                          f"and no '{y_prefix}.npy' found.")
            return None, None

    # Load X
    X_parts = [np.load(xf) for xf in x_files]
    X = np.concatenate(X_parts, axis=0) if len(X_parts) > 1 else X_parts[0]

    # Load Y
    Y_parts = [np.load(yf) for yf in y_files]
    y = np.concatenate(Y_parts, axis=0) if len(Y_parts) > 1 else Y_parts[0]

    if len(X) != len(y):
        logging.error(f"X and y lengths do not match. X length={len(X)}, y length={len(y)}")
        return None, None

    # Debug checks for NaNs or inf
    logging.info(f"Loaded X shape: {X.shape}, Y shape: {y.shape}")
    if np.isnan(X).any():
        logging.warning("NaNs detected in X array!")
    if np.isinf(X).any():
        logging.warning("Inf values detected in X array!")
    if np.isnan(y).any():
        logging.warning("NaNs detected in y array!")
    if np.isinf(y).any():
        logging.warning("Inf values detected in y array!")

    # Print min/max to check scale
    logging.info(f"X min={X.min()}, X max={X.max()}")
    logging.info(f"y min={y.min()}, y max={y.max()}")

    return X, y


def main():
    """
    1) Loads X, y from .npy parts (now from SEQUENCES_DIR).
    2) Performs debug checks on data (shapes, NaNs, etc.).
    3) Builds and compiles a hybrid model.
    4) Trains the model with a generator (NpyDataGenerator),
       but sets steps_per_epoch so partial leftover batch is ignored.
    5) Saves the best model to disk.
    """

    # Adjust these for whichever dataset you’re training on:
    x_prefix = 'real_time_spy_X_3D'
    y_prefix = 'real_time_spy_Y_3D'

    X, y = load_npy_data(x_prefix, y_prefix)
    if X is None or y is None:
        print("Could not load data. Please verify .npy files exist and shapes match.")
        return

    # Basic shape checks
    logging.info(f"Final X shape: {X.shape}, y shape: {y.shape}")
    # Example: print a small sample
    logging.info(f"Example X[0, 0]: {X[0, 0]}")
    logging.info(f"Example y[0]: {y[0]}")

    sequence_length = X.shape[1]
    num_features = X.shape[2]

    # Approach #1: define batch_size and steps_per_epoch explicitly,
    # ignoring leftover partial batches.
    batch_size = 32
    num_samples = X.shape[0]
    steps_per_epoch = num_samples // batch_size  # integer division => leftover ignored

    # Create generator
    train_generator = NpyDataGenerator(X, y, batch_size=batch_size, shuffle=True)
    
    # Build model
    input_shape = (sequence_length, num_features)
    model = create_hybrid_model(
        input_shape=input_shape,
        num_lstm_layers=2,
        num_cnn_filters=64,
        num_transformer_heads=4,
        dropout_rate=0.2
    )

    # If you see loss=nan, you can lower your LR to 1e-5 or 1e-6
    optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    logging.info("Model summary:")
    model.summary(print_fn=logging.info)

    # Checkpoints + early stopping
    checkpoint_path = os.path.join(MODEL_DIR, f'{MODEL_NAME}.keras')
    checkpoint_cb = ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor='loss',
        mode='min',
        verbose=1
    )
    early_stopping_cb = EarlyStopping(
        monitor='loss',
        patience=5,
        verbose=1
    )

    # Train
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=20,
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=1
    )

    print("Training complete. Model saved to:", checkpoint_path)


if __name__ == "__main__":
    main()