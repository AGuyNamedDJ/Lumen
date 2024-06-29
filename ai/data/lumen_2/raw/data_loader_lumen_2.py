import os
import boto3
import psycopg2
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

# Read environment variables
DATABASE_URL = os.getenv("DB_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Print environment variables for debugging
print("Environment Variables:")
print(f"DATABASE_URL: {DATABASE_URL}")
print(f"AWS_ACCESS_KEY_ID: {AWS_ACCESS_KEY_ID}")
print(f"AWS_SECRET_ACCESS_KEY: {AWS_SECRET_ACCESS_KEY}")
print(f"AWS_REGION: {AWS_REGION}")
print(f"S3_BUCKET_NAME: {S3_BUCKET_NAME}")


def check_db_connection():
    """Check if the database connection can be established."""
    try:
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor()
        cursor.execute("SELECT 1;")
        cursor.close()
        connection.close()
        print("Database connection established successfully.")
    except Exception as e:
        print(f"Error establishing database connection: {e}")
        return False
    return True


def load_data_from_db(query):
    """Load data from the PostgreSQL database based on the provided query."""
    try:
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=columns)
        print(f"Data loaded from database with query '{query}' successfully.")
        cursor.close()
        connection.close()
        return df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return pd.DataFrame()


def load_data_from_s3():
    """Load historical data from CSV files stored in an AWS S3 bucket."""
    all_data = []
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
        for obj in response.get('Contents', []):
            key = obj['Key']
            csv_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
            body = csv_obj['Body'].read().decode('utf-8')
            data = pd.read_csv(StringIO(body))
            all_data.append(data)
        print(f"Data loaded from S3 bucket '{S3_BUCKET_NAME}' successfully.")
    except Exception as e:
        print(f"Error loading data from S3: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def insert_data_into_db(data):
    """Insert data into the PostgreSQL database."""
    try:
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor()
        for index, row in data.iterrows():
            try:
                volume = None
                if 'Volume' in row:
                    volume = int(row['Volume'].replace(',', '')) if isinstance(
                        row['Volume'], str) and row['Volume'] else None

                close_value = None
                if 'Close/Last' in row:
                    close_value = float(row['Close/Last'].replace(',', '')) if isinstance(
                        row['Close/Last'], str) else row['Close/Last']
                elif 'Close' in row:
                    close_value = float(row['Close'].replace(',', '')) if isinstance(
                        row['Close'], str) else row['Close']

                if close_value is None:
                    print(f"Skipping row {index} due to missing 'Close' value")
                    continue

                cursor.execute(
                    """
                    INSERT INTO historical_spx (timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                    (
                        row['Date'],
                        float(row['Open'].replace(',', '')) if isinstance(
                            row['Open'], str) else row['Open'],
                        float(row['High'].replace(',', '')) if isinstance(
                            row['High'], str) else row['High'],
                        float(row['Low'].replace(',', '')) if isinstance(
                            row['Low'], str) else row['Low'],
                        close_value,
                        volume
                    )
                )
                connection.commit()  # Commit each row separately
            except Exception as e:
                connection.rollback()  # Rollback if there is an error with this row
                print(f"Error inserting row {index} into the database: {e}")
        cursor.close()
        connection.close()
        print(f"Inserted {len(data)} records into the database successfully.")
    except Exception as e:
        print(f"Error inserting data into the database: {e}")


def main():
    # Check database connection
    if not check_db_connection():
        print("Exiting script due to database connection issues.")
        return

    # Load historical data from DB
    historical_data_query = "SELECT * FROM historical_spx;"
    historical_data_db = load_data_from_db(historical_data_query)

    # Load live data from DB
    live_data_query = "SELECT * FROM real_time_spx;"
    live_data_db = load_data_from_db(live_data_query)

    # Load historical data from S3
    historical_data_s3 = load_data_from_s3()

    # Combine historical data from DB and S3
    combined_historical_data = pd.concat(
        [historical_data_db, historical_data_s3], ignore_index=True)

    # Combine all data
    combined_data = pd.concat(
        [combined_historical_data, live_data_db], ignore_index=True)

    # Save combined data to CSV
    combined_data.to_csv('data/processed/combined_data.csv', index=False)
    print("Combined data saved to data/processed/combined_data.csv")

    # Print the first few rows of the combined data for verification
    print("First few rows of the combined data:")
    print(combined_data.head())


if __name__ == "__main__":
    main()
