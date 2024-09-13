import numpy as np

# Load real-time files
y_test_real = np.load('y_test_real.npy')
predictions_real = np.load('predictions_real.npy')

# Load historical files
y_test_hist = np.load('y_test_hist.npy')
predictions_hist = np.load('predictions_hist.npy')

# Print the shape and first few elements of y_test_real
print(f"Shape of y_test_real: {y_test_real.shape}")
print(f"First 10 elements of y_test_real: {y_test_real[:10]}")

# Print the shape and first few elements of predictions_real
print(f"Shape of predictions_real: {predictions_real.shape}")
print(f"First 10 elements of predictions_real: {predictions_real[:10]}")

# Print the shape and first few elements of y_test_hist
print(f"Shape of y_test_hist: {y_test_hist.shape}")
print(f"First 10 elements of y_test_hist: {y_test_hist[:10]}")

# Print the shape and first few elements of predictions_hist
print(f"Shape of predictions_hist: {predictions_hist.shape}")
print(f"First 10 elements of predictions_hist: {predictions_hist[:10]}")
