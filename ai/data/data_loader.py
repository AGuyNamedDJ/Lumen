import os
import boto3
import psycopg2
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

# Read environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
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


def load_data_from_db():
    """Load data from the PostgreSQL database."""
    try:
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM historical_spx;")
        data = cursor.fetchall()
        print("Data loaded from database successfully.")
        cursor.close()
        connection.close()
        return data
    except Exception as e:
        print(f"Error loading data from database: {e}")


def load_data_from_s3():
    """Load data from CSV files stored in an AWS S3 bucket."""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
        print(f"S3 list objects response: {response}")
        for obj in response.get('Contents', []):
            key = obj['Key']
            print(f"Loading data from {key}...")
            csv_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
            body = csv_obj['Body'].read().decode('utf-8')
            data = pd.read_csv(StringIO(body))
            print(f"CSV data loaded successfully from {key}.")
            insert_data_into_db(data)
    except Exception as e:
        print(f"Error loading data from S3: {e}")


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


if __name__ == "__main__":
    db_data = load_data_from_db()
    s3_data = load_data_from_s3()
