import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Define model directory path
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# Step 2: Load the true test values for real-time (y_test_real.npy)
y_true_real = np.load(os.path.join(MODEL_DIR, 'y_test_real.npy'))

# Step 3: Load the saved predictions from predictions_real.npy
predictions = np.load(os.path.join(MODEL_DIR, 'predictions_real.npy'))

# Step 4: Evaluate the model's performance
mse = mean_squared_error(y_true_real, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_true_real, predictions)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Step 5: Plot Predicted vs True values
plt.figure(figsize=(10, 6))
plt.scatter(y_true_real, predictions, alpha=0.5)
plt.plot([min(y_true_real), max(y_true_real)], [
         min(y_true_real), max(y_true_real)], 'r--')
plt.title('Predicted vs. True Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.show()

# Step 6: Plot Residuals
residuals = y_true_real - predictions
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, alpha=0.75)
plt.title('Residuals (Errors) Distribution')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()
