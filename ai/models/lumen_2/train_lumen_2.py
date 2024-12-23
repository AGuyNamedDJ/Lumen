import os
import numpy as np
import logging
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from dotenv import load_dotenv
from definitions_lumen_2 import create_hybrid_model

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', '..', 'models', 'lumen_2')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_NAME = 'Lumen2'

FEATURED_DATA_DIR = os.path.join(BASE_DIR, '../../data/lumen_2/featured')
if not os.path.exists(FEATURED_DATA_DIR):
    raise FileNotFoundError(f"The directory {FEATURED_DATA_DIR} does not exist.")

print("Files in featured directory:", os.listdir(FEATURED_DATA_DIR))

class NpyDataGenerator(Sequence):
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
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def find_npy_parts(directory, prefix):
    files = [os.path.join(directory, f) for f in os.listdir(directory) 
             if f.startswith(prefix) and f.endswith('.npy')]
    def part_num(fname):
        base = os.path.basename(fname)
        segments = base.split('part')
        if len(segments) > 1:
            num_str = segments[-1].replace('.npy', '')
            try:
                return int(num_str)
            except ValueError:
                return 0
        else:
            return 0
    files.sort(key=part_num)
    return files

def load_npy_data(x_prefix, y_prefix):
    x_files = find_npy_parts(FEATURED_DATA_DIR, x_prefix)
    if len(x_files) == 0:
        logging.error(f"No X files found with prefix '{x_prefix}' in {FEATURED_DATA_DIR}")
        return None, None

    y_files = find_npy_parts(FEATURED_DATA_DIR, y_prefix)
    if len(y_files) == 0:
        # Check if there's a single y file (without 'partX')
        single_y_file = os.path.join(FEATURED_DATA_DIR, f"{y_prefix}.npy")
        if os.path.exists(single_y_file):
            y_files = [single_y_file]
        else:
            logging.error(f"No Y files found with prefix '{y_prefix}', and no '{y_prefix}.npy' found.")
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

    return X, y

def main():
    # Update these prefixes according to your dataset
    # For example, if you have files like real_time_spy_X_3D_part0.npy and real_time_spy_Y_3D_part0.npy:
    x_prefix = 'real_time_spy_X_3D'
    y_prefix = 'real_time_spy_Y_3D'

    X, y = load_npy_data(x_prefix, y_prefix)
    if X is None or y is None:
        print("Could not load data. Ensure X and y npy files are generated and match in length.")
        return

    sequence_length = X.shape[1]
    num_features = X.shape[2]

    batch_size = 32
    train_generator = NpyDataGenerator(X, y, batch_size=batch_size, shuffle=True)

    input_shape = (sequence_length, num_features)
    model = create_hybrid_model(input_shape=input_shape)
    optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    steps_per_epoch = len(train_generator)
    checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, f'{MODEL_NAME}.keras'),
                                 save_best_only=True, monitor='loss', mode='min')
    early_stopping = EarlyStopping(monitor='loss', patience=5)

    model.fit(train_generator,
              steps_per_epoch=steps_per_epoch,
              epochs=20,
              callbacks=[checkpoint, early_stopping],
              verbose=1)

    print("Training complete. Model saved to:", os.path.join(MODEL_DIR, f'{MODEL_NAME}.keras'))

if __name__ == "__main__":
    main()