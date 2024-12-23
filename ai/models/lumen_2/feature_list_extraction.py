import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct path to the featured data directory
FEATURED_DATA_DIR = os.path.join(
    BASE_DIR, '..', '..', 'data', 'lumen_2', 'featured')

# Ensure the directory exists
if not os.path.exists(FEATURED_DATA_DIR):
    raise FileNotFoundError(
        f"The directory {FEATURED_DATA_DIR} does not exist.")

# Debug: List the contents of the featured directory
print("Files in featured directory:", os.listdir(FEATURED_DATA_DIR))

# **For the Real-Time Model, use real-time data for both training and test datasets**

# Load the combined real-time data as training data
training_data_path = os.path.join(
    FEATURED_DATA_DIR, 'combined_real_time_data.csv')
if not os.path.exists(training_data_path):
    raise FileNotFoundError(f"The training data file {
                            training_data_path} does not exist.")

training_data = pd.read_csv(training_data_path)

# Replace 'current_price' with your actual target column name if different
target_column = 'current_price'  # Adjust based on your data

# Exclude the target column to get the list of feature columns
feature_columns_training = [
    col for col in training_data.columns if col != target_column]

# Print the list of features used in the training data
print("\nFeatures used in the training data (Real-Time):")
for feature in feature_columns_training:
    print(feature)

# Optionally, save the feature list to a file for later reference
with open('original_model_features_real_time.txt', 'w') as f:
    for feature in feature_columns_training:
        f.write(f"{feature}\n")

# Load your test data and get the list of features
# For the test data, you might use 'featured_real_time_spx_featured.csv' or a separate test dataset

test_data_path = os.path.join(
    FEATURED_DATA_DIR, 'featured_real_time_spx_featured.csv')
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"The test data file {
                            test_data_path} does not exist.")

test_data = pd.read_csv(test_data_path)

# Exclude the target column to get the list of feature columns in test data
feature_columns_test = [
    col for col in test_data.columns if col != target_column]

# Print the list of features available in the test data
print("\nFeatures available in the test data (Real-Time):")
for feature in feature_columns_test:
    print(feature)

# Identify missing features (present in training data but not in test data)
missing_features = list(set(feature_columns_training) -
                        set(feature_columns_test))

# Identify common features (present in both training and test data)
common_features = list(set(feature_columns_training)
                       & set(feature_columns_test))

# Print missing features
print("\nMissing features (present in training data but not in test data):")
for feature in missing_features:
    print(feature)

# Optionally, save the list of missing features to a file
with open('missing_features_real_time.txt', 'w') as f:
    for feature in missing_features:
        f.write(f"{feature}\n")

# Print common features
print("\nCommon features (present in both training and test data):")
for feature in common_features:
    print(feature)
