import numpy as np
import os

# Define the correct paths to your test data files
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the test data
X_test_hist = np.load(os.path.join(MODEL_DIR, 'X_test_hist.npy'))
y_test_hist = np.load(os.path.join(MODEL_DIR, 'y_test_hist.npy'))
X_test_real = np.load(os.path.join(MODEL_DIR, 'X_test_real.npy'))
y_test_real = np.load(os.path.join(MODEL_DIR, 'y_test_real.npy'))

# Print shapes of the test datasets to verify number of features and time steps
print("X_test_hist shape:", X_test_hist.shape)  # Should be (samples, 30, 41)
print("y_test_hist shape:", y_test_hist.shape)
print("X_test_real shape:", X_test_real.shape)  # Should be (samples, 30, 36)
print("y_test_real shape:", y_test_real.shape)
