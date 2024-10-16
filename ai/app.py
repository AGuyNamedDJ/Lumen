# Standard library imports
import os
import logging
import datetime

# Third-party imports
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, MetaData, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from openai import OpenAI
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Local imports
from models.lumen_2.definitions_lumen_2 import ReduceMeanLayer
from models.lumen_2.conversation import select_model_for_prediction, gpt_4o_mini_response

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=api_key)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000",
                                         "https://lumen-1.netlify.app/"]}}, supports_credentials=True)

# Set up SQLAlchemy Base and database session
Base = declarative_base()

# Database setup
db_url = os.getenv('DB_URL')
if not db_url:
    raise ValueError("DB_URL environment variable is not set")
logging.debug(f"DB_URL: {db_url}")

engine = create_engine(db_url)
Session = sessionmaker(bind=engine)

# Load models
MODEL_DIR = os.path.join(os.getcwd(), 'models', 'lumen_2')
historic_model_path = os.path.join(MODEL_DIR, 'Lumen2_historical.keras')
real_time_model_path = os.path.join(MODEL_DIR, 'Lumen2_real_time.keras')

# Load models with custom objects if necessary
try:
    Lumen2_historic = load_model(
        historic_model_path, custom_objects={'ReduceMeanLayer': ReduceMeanLayer})
    logging.debug("Lumen2_historic model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Lumen2_historic model: {e}")
    raise e

try:
    Lumen2_real_time = load_model(
        real_time_model_path, custom_objects={'ReduceMeanLayer': ReduceMeanLayer})
    logging.debug("Lumen2_real_time model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Lumen2_real_time model: {e}")
    raise e


def load_features(stock_classification):
    # Real-time data with 'timestamp'
    real_time_csv_files = [
        'data/lumen_2/featured/featured_real_time_spx_featured.csv',  # SPX real-time
        'data/lumen_2/featured/featured_real_time_spy_featured.csv',  # SPY real-time
        'data/lumen_2/featured/featured_real_time_vix_featured.csv'   # VIX real-time
    ]

    # Historical and other data without 'timestamp'
    historical_csv_files = [
        'data/lumen_2/featured/featured_consumer_confidence_data_featured.csv',
        'data/lumen_2/featured/featured_consumer_sentiment_data_featured.csv',
        'data/lumen_2/featured/featured_core_inflation_data_featured.csv',
        'data/lumen_2/featured/featured_cpi_data_featured.csv',
        'data/lumen_2/featured/featured_gdp_data_featured.csv',
        'data/lumen_2/featured/featured_historical_spx_featured.csv',
        'data/lumen_2/featured/featured_historical_spy_featured.csv',
        'data/lumen_2/featured/featured_historical_vix_featured.csv',
        'data/lumen_2/featured/featured_industrial_production_data_featured.csv',
        'data/lumen_2/featured/featured_interest_rate_data_featured.csv',
        'data/lumen_2/featured/featured_labor_force_participation_rate_data_featured.csv',
        'data/lumen_2/featured/featured_nonfarm_payroll_employment_data_featured.csv',
        'data/lumen_2/featured/featured_personal_consumption_expenditures_data_featured.csv',
        'data/lumen_2/featured/featured_ppi_data_featured.csv',
        'data/lumen_2/featured/featured_unemployment_rate_data_featured.csv',
    ]

    # Load real-time data
    real_time_dataframes = []
    for csv_file in real_time_csv_files:
        csv_file_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), csv_file)
        try:
            df = pd.read_csv(csv_file_path)
            logging.debug(f"Loaded features from {csv_file}")

            # Print the columns of each real-time CSV
            # Prints to console
            print(f"Real-time CSV: {csv_file}, Columns: {df.columns.tolist()}")

            # Ensure 'timestamp' column exists
            if 'timestamp' not in df.columns:
                logging.warning(f"'timestamp' column not found in {
                                csv_file}, skipping this file.")
                continue  # Skip this CSV if 'timestamp' is missing

            real_time_dataframes.append(df)
        except Exception as e:
            logging.error(f"Error reading CSV file {csv_file}: {e}")
            return None

    # Merge real-time DataFrames on 'timestamp'
    if real_time_dataframes:
        from functools import reduce
        merged_real_time_df = reduce(lambda left, right: pd.merge(
            left, right, on='timestamp', how='outer'), real_time_dataframes)
        merged_real_time_df.sort_values(by='timestamp', inplace=True)
        merged_real_time_df.ffill(inplace=True)
        merged_real_time_df.bfill(inplace=True)

        # Clean up column names (remove _x, _y suffixes)
        merged_real_time_df.columns = merged_real_time_df.columns.str.replace(
            '_x', '').str.replace('_y', '')

    else:
        merged_real_time_df = None

    # Load historical data
    historical_dataframes = []
    for csv_file in historical_csv_files:
        csv_file_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), csv_file)
        try:
            df = pd.read_csv(csv_file_path)
            logging.debug(f"Loaded features from {csv_file}")

            # Print the columns of each historical CSV
            print(f"Historical CSV: {csv_file}, Columns: {
                df.columns.tolist()}")  # Prints to console

            historical_dataframes.append(df)
        except Exception as e:
            logging.error(f"Error reading CSV file {csv_file}: {e}")
            return None

    # Combine historical data (since they don't have 'timestamp', we won't merge on 'timestamp')
    if historical_dataframes:
        combined_historical_df = pd.concat(historical_dataframes, axis=1)
    else:
        combined_historical_df = None

    # Combine real-time and historical data
    if merged_real_time_df is not None and combined_historical_df is not None:
        final_combined_df = pd.concat(
            [merged_real_time_df, combined_historical_df], axis=1)

        # Clean up column names (remove _x, _y suffixes)
        final_combined_df.columns = final_combined_df.columns.str.replace(
            '_x', '').str.replace('_y', '')

    elif merged_real_time_df is not None:
        final_combined_df = merged_real_time_df
    elif combined_historical_df is not None:
        final_combined_df = combined_historical_df
    else:
        logging.error("No valid data loaded.")
        return None

    return final_combined_df


def merge_dataframes(dataframes):
    from functools import reduce

    # Ensure all elements are valid DataFrames before attempting to merge
    valid_dataframes = [
        df for df in dataframes if isinstance(df, pd.DataFrame)]

    if not valid_dataframes:
        logging.error("No valid DataFrames to merge.")
        return None

    try:
        # Assuming all DataFrames have a 'timestamp' column
        merged_df = reduce(lambda left, right: pd.merge(
            left, right, on='timestamp'), valid_dataframes)

        logging.debug(f"Merged DataFrame: {merged_df.head()}")

        # Log the final merged DataFrame columns
        logging.debug(f"Final merged DataFrame columns: {
                      merged_df.columns.tolist()}")

        return merged_df
    except Exception as e:
        logging.error(f"Error merging DataFrames: {e}")
        return None


def prepare_input_data(df, expected_features_list):
    sequence_length = 30  # As per your model's expected sequence length

    # Ensure the DataFrame has enough rows
    if len(df) < sequence_length:
        logging.error("Not enough data to form a complete sequence.")
        return None

    # Log the available columns in the DataFrame
    logging.debug(f"Available DataFrame columns: {df.columns.tolist()}")

    # Select the last `sequence_length` rows
    df_sequence = df.tail(sequence_length)

    # Check for missing features
    available_features = [
        feature for feature in expected_features_list if feature in df_sequence.columns]
    missing_features = set(expected_features_list) - set(available_features)

    # Log available and missing features
    logging.debug(f"Available features: {available_features}")
    logging.debug(f"Missing features: {missing_features}")

    if missing_features:
        logging.warning(f"Missing features: {missing_features}")
        logging.info(f"Proceeding with available features: {
                     available_features}")
        # Add default values for missing features if necessary
        for feature in missing_features:
            # Fill missing features with 0 or appropriate default value
            df_sequence[feature] = 0

    # Ensure that you only use the expected features (36 in this case)
    # Select only the first 36 features
    df_sequence = df_sequence[expected_features_list[:36]]

    # Convert to NumPy array and reshape
    input_array = df_sequence.values
    # Shape: (1, sequence_length, num_features)
    input_array = input_array[np.newaxis, :, :]

    return input_array


def apply_scaling(input_array, scaler):
    # Reshape input_array for scaling
    original_shape = input_array.shape
    reshaped_input = input_array.reshape(-1, original_shape[-1])

    # Apply scaler
    scaled_input = scaler.transform(reshaped_input)

    # Reshape back to original shape
    input_array_scaled = scaled_input.reshape(original_shape)
    return input_array_scaled


# Define the categorize_question function


def categorize_question(message):
    """
    Categorizes user queries into categories like market analysis, SPX, SPY, or others.
    """
    message_lower = message.lower()

    # SPX/Stock related queries
    if "spx" in message_lower:
        return "spx"
    elif "spy" in message_lower:
        return "spy"

    # Market analysis keywords
    market_analysis_keywords = [
        "price", "prices", "forecast", "predict", "trend", "valuation",
        "estimate", "outlook", "technical analysis"
    ]

    if any(keyword in message_lower for keyword in market_analysis_keywords):
        return "market_analysis"

    # General fallback category
    return "general"

# Define the classify_message function


def classify_message(message):
    logging.debug("Classifying message using categorize_question function.")
    try:
        # Call categorize_question to determine the category
        category = categorize_question(message)

        # Treat SPX, SPY, and market analysis as stock-related
        if category in ["market_analysis", "spx", "spy"]:
            logging.debug(f"Message classified as stock-related: {category}")
            return {"is_stock_related": True, "classification": category}
        else:
            logging.debug(
                f"Message classified as non-stock-related: {category}")
            return {"is_stock_related": False}
    except Exception as e:
        logging.error(f"Error in message classification: {e}")
        return {"is_stock_related": False, "error": str(e)}

# Define get_latest_stock_price function


def get_latest_stock_price(stock_classification):
    """
    Fetches the latest price for the given stock classification (SPX or SPY).
    """
    try:
        session = Session()

        # Reflect the database schema
        metadata = MetaData()
        metadata.reflect(bind=engine)

        # Log all available tables
        logging.debug(f"Available tables: {list(metadata.tables.keys())}")

        if stock_classification == "spx":
            table_name = 'real_time_spx'
        elif stock_classification == "spy":
            table_name = 'real_time_spy'
        else:
            raise ValueError(f"Unknown stock classification: {
                             stock_classification}")

        # Check if the table exists in metadata
        if table_name in metadata.tables:
            table = metadata.tables[table_name]
        else:
            logging.error(f"Table '{table_name}' not found in metadata.")
            return None

        # Build and execute the query
        stmt = select(table).order_by(table.c.timestamp.desc()).limit(1)
        result = session.execute(stmt).mappings().first()

        # Close the session
        session.close()

        if result:
            return result['current_price']
        else:
            logging.error(f"No entries found for {
                          stock_classification.upper()}")
            return None
    except Exception as e:
        logging.error(
            f"Error fetching latest {stock_classification.upper()} price: {e}", exc_info=True)
        return None

# Handle preflight requests


@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        headers = response.headers
        headers['Access-Control-Allow-Origin'] = '*'
        headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS, PUT, DELETE'
        headers['Access-Control-Allow-Credentials'] = 'true'
        headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
        return response

# Define the classify route


@app.route('/classify', methods=['POST'])
def classify():
    request_data = request.get_json()
    message = request_data.get('message')
    if not message:
        return jsonify({"error": "No message provided"}), 400
    result = classify_message(message)
    return jsonify(result), 200

# Define the conversation route


@app.route('/conversation', methods=['POST'])
def conversation():
    try:
        logging.debug("Received a request at /conversation endpoint")
        request_data = request.get_json()
        message = request_data.get('message')
        if not message:
            logging.error("No message provided in the request")
            return jsonify({"error": "No message provided"}), 400

        # Classify the message
        classification_result = classify_message(message)
        logging.debug(f"Classification result: {classification_result}")

        if classification_result.get('is_stock_related'):
            stock_classification = classification_result.get('classification')
            logging.debug(
                f"Message classified as stock-related, classification: {stock_classification}")

            # Load features
            df_features = load_features(stock_classification)
            if df_features is None:
                return jsonify({"error": "Failed to load features"}), 500

            # Log the DataFrame columns
            logging.debug(f"DataFrame columns: {df_features.columns.tolist()}")

            # Decide which model to use
            is_real_time = stock_classification == 'spx'  # Adjust this condition as needed
            model_to_use = select_model_for_prediction(is_real_time)

            # Determine the target column based on stock classification
            if is_real_time:
                target_column = 'current_price'
            else:
                target_column = 'close'  # Adjust as necessary

            # Define the expected features list
            expected_features_list = [
                # Consumer Confidence Features
                'consumer_confidence_cumulative_sum',
                'days_since_consumer_confidence_peak',
                'days_since_consumer_confidence_trough',
                # Consumer Sentiment Features
                'consumer_sentiment_rolling_6m_avg',
                'consumer_sentiment_rolling_12m_avg',
                # Core Inflation Features
                'core_inflation_value',
                'core_inflation_ema_12',
                'core_inflation_ema_26',
                'core_inflation_cumulative_sum',
                'core_inflation_z_score',
                'core_inflation_trend',
                # CPI Features
                'feature_cpi_value',
                'feature_cpi_trend',
                'feature_cpi_cumulative_sum',
                'feature_cpi_days_since_peak',
                'dfeature_cpi_days_since_trough',
                # GDP Features
                'feature_gdp_value',
                'feature_gdp_lag',
                'feature_gdp_rolling_mean',
                'feature_gdp_rolling_std',
                'feature_gdp_cumulative_sum',
                'feature_gdp_cumulative_product',
                'feature_gdp_trend',
                'feature_gdp_ema',
                # Industrial Production Features
                'feature_industrial_production_value',
                'feature_industrial_production_lag',
                'feature_industrial_production_rolling_mean',
                'feature_industrial_production_cumulative_sum',
                'feature_industrial_production_cumulative_product',
                'feature_industrial_production_trend',
                'feature_industrial_production_ema',
                'feature_industrial_production_z_score',
                'feature_industrial_production_days_since_trough',
                # Interest Rate Features
                'feature_interest_rate_value',
                'feature_interest_rate_lag',
                'feature_interest_rate_rolling_mean',
                'feature_interest_rate_ema',
                'feature_interest_rate_days_since_peak',
                'feature_interest_rate_days_since_trough',
                # Labor Force Features
                'feature_labor_force_value',
                'feature_labor_force_lag',
                'feature_labor_force_rolling_mean',
                'feature_labor_force_ema',
                'feature_labor_force_days_since_peak',
                'feature_labor_force_days_since_trough',
                # Nonfarm Features
                'feature_nonfarm_value',
                'feature_nonfarm_lag',
                'feature_nonfarm_rolling_mean',
                'feature_nonfarm_ema',
                'feature_nonfarm_cumulative_sum',
                'feature_nonfarm_cumulative_product',
                'feature_nonfarm_days_since_trough',
                # PCE Features
                'feature_pce_value',
                'feature_pce_lag',
                'feature_pce_rolling_mean',
                'feature_pce_cumulative_sum',
                'feature_pce_cumulative_product',
                'feature_pce_trend',
                'feature_pce_ema',
                'feature_pce_z_score',
                'feature_pce_days_since_peak',
                # PPI Features
                'feature_ppi_value',
                'feature_ppi_lag',
                'feature_ppi_rolling_mean',
                'feature_ppi_cumulative_sum',
                'feature_ppi_cumulative_product',
                'feature_ppi_trend',
                'feature_ppi_ema',
                'feature_ppi_z_score',
                'feature_ppi_days_since_trough',
                # Unemployment Rate Features
                'feature_unemployment_rate_cumulative_sum',
                'feature_unemployment_rate_days_since_trough',
                # Historical Features
                'feature_historical_indicator_ema_12',
                'feature_historical_indicator_rsi',
                # SPX Market Features
                'feature_spx_ema',
                'feature_spx_sma',
                'feature_spx_drawdown_recovery',
                # SPY Market Features
                'feature_spy_price_data',
                'feature_spy_atr',
                'feature_spy_drawdown_recovery',
                # SPX Features
                'Real_Time_Indicator_Lag_1',
                'Real_Time_Indicator_SMA_20',
                'Real_Time_Indicator_SMA_50',
                'Real_Time_Indicator_EMA_12',
                'Real_Time_Indicator_EMA_26',
                'Real_Time_SPX_VIX_Correlation',
                # SPY Features
                'Real_Time_Indicator_Lag_1',
                'Real_Time_Indicator_SMA_20',
                'Real_Time_Indicator_SMA_50',
                'Real_Time_Indicator_EMA_12',
                'Real_Time_Indicator_EMA_26',
                'Real_Time_SPY_VIX_Correlation'
            ]

            # Prepare input data
            input_array = prepare_input_data(
                df_features, expected_features_list)
            if input_array is None:
                return jsonify({"error": "Failed to prepare input data"}), 500

            # Load the appropriate scaler
            if is_real_time:
                scaler_path = 'models/lumen_2/scaler_current_price.joblib'
            else:
                scaler_path = 'models/lumen_2/scaler_close.joblib'

            try:
                scaler = joblib.load(scaler_path)
            except Exception as e:
                logging.error(f"Error loading feature scaler: {e}")
                return jsonify({"error": "Failed to load feature scaler"}), 500

            # Apply scaling to input features
            input_array = apply_scaling(input_array, feature_scaler)

            # Make prediction
            predicted_price = model_to_use.predict(input_array)

            # Load the target scaler
            try:
                target_scaler = joblib.load(
                    'models/lumen_2/target_scaler.joblib')
            except Exception as e:
                logging.error(f"Error loading target scaler: {e}")
                return jsonify({"error": "Failed to load target scaler"}), 500

            # Reshape and inverse transform the predicted price
            predicted_value_reshaped = np.array(predicted_price).reshape(-1, 1)
            predicted_price_value = target_scaler.inverse_transform(
                predicted_value_reshaped)[0][0]

            logging.debug(f"Predicted price (after inverse scaling): {
                          predicted_price_value}")
            return jsonify({"predicted_price": float(predicted_price_value)}), 200

        else:
            # Handle non-stock-related messages
            logging.debug(
                "Message not stock-related, falling back to GPT-4o-mini")
            gpt_result = gpt_4o_mini_response(message)
            if gpt_result.get('success'):
                return jsonify({"response": gpt_result['response']}), 200
            else:
                return jsonify({"error": gpt_result.get('error', 'Unknown error')}), 500

    except Exception as e:
        logging.error(f"Unexpected error in /conversation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
