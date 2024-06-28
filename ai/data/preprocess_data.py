import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

# Environment variables
DB_URL = os.getenv('DB_URL')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Function to establish a database connection using SQLAlchemy


def get_db_connection():
    try:
        engine = create_engine(DB_URL)
        return engine
    except Exception as e:
        print(f"Error creating database connection: {e}")
        return None

# Function to fetch data from the database


def fetch_data():
    try:
        engine = get_db_connection()
        if engine is None:
            raise Exception("Database connection failed")

        query = "SELECT * FROM historical_spx"
        data = pd.read_sql(query, engine)
        print("Data fetched successfully from the database.")
        return data
    except Exception as e:
        print(f"Error fetching data from database: {e}")
        return None

# Function to handle missing values in the dataset


def handle_missing_values(data):
    data.ffill(inplace=True)  # Forward fill missing values
    data.bfill(inplace=True)  # Backward fill missing values
    return data

# Function to normalize the data using MinMaxScaler


def normalize_data(data):
    scaler = MinMaxScaler()
    data[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(
        data[['open', 'high', 'low', 'close', 'volume']].fillna(0))
    return data

# Function to preprocess the fetched data


def preprocess_data(data):
    # Ensure consistent column naming
    if 'date' in data.columns:
        data.rename(columns={'date': 'timestamp'}, inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    data = handle_missing_values(data)
    data = normalize_data(data)
    return data

# Main function to orchestrate data fetching and preprocessing


def main():
    print("Starting data preprocessing...")
    data = fetch_data()
    if data is not None:
        print(f"Data shape before preprocessing: {data.shape}")
        preprocessed_data = preprocess_data(data)
        preprocessed_data.to_csv(os.path.join(os.path.dirname(
            __file__), 'processed/preprocessed_spx_data.csv'))
        print("Data preprocessed and saved successfully.")
    else:
        print("Failed to preprocess data.")


# Entry point for the script
if __name__ == "__main__":
    main()
