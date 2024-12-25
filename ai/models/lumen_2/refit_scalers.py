# import os
# import pandas as pd
# import joblib
# from sklearn.preprocessing import MinMaxScaler

# # Define paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# FEATURED_DIR = os.path.join(BASE_DIR, '../../data/lumen_2/featured')
# MODEL_DIR = os.path.join(BASE_DIR, '../../models/lumen_2')

# # List of datasets to fit scalers on
# data_paths = {
#     # Economic indicators
#     'consumer_confidence': os.path.join(FEATURED_DIR, 'featured_consumer_confidence_data_featured.csv'),
#     'consumer_sentiment': os.path.join(FEATURED_DIR, 'featured_consumer_sentiment_data_featured.csv'),
#     'core_inflation': os.path.join(FEATURED_DIR, 'featured_core_inflation_data_featured.csv'),
#     'cpi': os.path.join(FEATURED_DIR, 'featured_cpi_data_featured.csv'),
#     'gdp': os.path.join(FEATURED_DIR, 'featured_gdp_data_featured.csv'),
#     'industrial_production': os.path.join(FEATURED_DIR, 'featured_industrial_production_data_featured.csv'),
#     'interest_rate': os.path.join(FEATURED_DIR, 'featured_interest_rate_data_featured.csv'),
#     'labor_force': os.path.join(FEATURED_DIR, 'featured_labor_force_participation_rate_data_featured.csv'),
#     'nonfarm_payroll': os.path.join(FEATURED_DIR, 'featured_nonfarm_payroll_employment_data_featured.csv'),
#     'personal_consumption': os.path.join(FEATURED_DIR, 'featured_personal_consumption_expenditures_data_featured.csv'),
#     'ppi': os.path.join(FEATURED_DIR, 'featured_ppi_data_featured.csv'),
#     'unemployment_rate': os.path.join(FEATURED_DIR, 'featured_unemployment_rate_data_featured.csv'),
#     # Historical data
#     'historical_spx': os.path.join(FEATURED_DIR, 'featured_historical_spx_featured.csv'),
#     'historical_spy': os.path.join(FEATURED_DIR, 'featured_historical_spy_featured.csv'),
#     'historical_vix': os.path.join(FEATURED_DIR, 'featured_historical_vix_featured.csv'),
#     # Real-time data
#     'real_time_spx': os.path.join(FEATURED_DIR, 'featured_real_time_spx_featured.csv'),
#     'real_time_spy': os.path.join(FEATURED_DIR, 'featured_real_time_spy_featured.csv'),
#     'real_time_vix': os.path.join(FEATURED_DIR, 'featured_real_time_vix_featured.csv'),
# }

# for dataset_name in data_paths:
#     data_path = data_paths.get(dataset_name)
#     if data_path is None or not os.path.exists(data_path):
#         print(f"Data file {data_path} does not exist! Skipping.")
#         continue

#     scaler_path = os.path.join(
#         MODEL_DIR, f'{dataset_name}_feature_scaler.joblib')

#     # Load data
#     df = pd.read_csv(data_path)

#     # Identify datetime and target columns
#     datetime_columns = ['date', 'timestamp']
#     target_columns = ['close', 'current_price', 'value']

#     # Drop datetime and target columns
#     df_features = df.drop(columns=datetime_columns +
#                           target_columns, errors='ignore')

#     # Check if df_features is empty
#     if df_features.empty:
#         print(f"No features to scale in {dataset_name}. Skipping.")
#         continue

#     # Check for NaN values and handle them
#     if df_features.isnull().all().all():
#         print(f"All features are NaN in {dataset_name}. Skipping.")
#         continue
#     df_features.fillna(0, inplace=True)

#     # Fit scaler
#     scaler = MinMaxScaler()
#     scaler.fit(df_features)

#     # Save scaler
#     joblib.dump(scaler, scaler_path)
#     print(f"Scaler for {dataset_name} saved at {scaler_path}")

# # Verification loop after all scalers are saved
# for verify_dataset_name in data_paths:
#     scaler_path = os.path.join(
#         MODEL_DIR, f'{verify_dataset_name}_feature_scaler.joblib')
#     if os.path.exists(scaler_path):
#         scaler = joblib.load(scaler_path)
#         expected_columns = scaler.feature_names_in_
#         if 'close' in expected_columns:
#             print(f"Warning: 'close' is still in expected features for {
#                   verify_dataset_name}")
#         else:
#             print(f"'close' is not in expected features for {
#                   verify_dataset_name}")
