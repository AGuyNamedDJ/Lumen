import os
import pandas as pd
import json
import openai
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from openai.error import APIError, RateLimitError, APIConnectionError


# Load environment variables
load_dotenv()

# Base directory settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database settings from environment variables
DB_URL = os.getenv('DB_URL')
if not DB_URL:
    raise ValueError("DB_URL environment variable is missing.")

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is missing.")
openai.api_key = OPENAI_API_KEY

# Model directory and featured data directory paths
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FEATURED_DATA_DIR = os.path.join(BASE_DIR, '../../data/lumen_2/featured')

# Ensure model and featured data directories exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(FEATURED_DATA_DIR):
    raise FileNotFoundError(f"The directory {FEATURED_DATA_DIR} does not exist.")

# Database connection function
def get_db_connection():
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    return Session()

# Prepare fine-tuning data
def prepare_fine_tuning_data(input_csv, output_jsonl):
    # Load the data from CSV
    df = pd.read_csv(input_csv)

    # Check for missing values and drop rows with missing values
    if df.isnull().values.any():
        print("Warning: Missing values found in the dataset. Dropping rows with missing values.")
        df = df.dropna()

    # Prepare chat-formatted data for fine-tuning
    chat_data = []
    for i in range(len(df) - 1):
        message = f"Given the market data: Open: {df.loc[i, 'open']}, High: {df.loc[i, 'high']}, Low: {df.loc[i, 'low']}, Close: {df.loc[i, 'close']}, Volume: {df.loc[i, 'volume']}, EMA_10: {df.loc[i, 'ema_10']}, EMA_50: {df.loc[i, 'ema_50']}, what will be the closing price of SPX tomorrow?"
        response = f"{df.loc[i + 1, 'close']}"
        
        chat_data.append({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]
        })

    # Save the prepared data to a JSONL file
    with open(output_jsonl, 'w') as f:
        for item in chat_data:
            f.write(json.dumps(item) + "\n")

    print(f"Fine-tuning data saved to {output_jsonl}")


def fine_tune_model():
    input_csv = 'models/lumen_2/final_combined_data.csv'
    output_jsonl = 'data/processed/fine_tune_data.jsonl'  # JSONL file for fine-tuning
    
    # Prepare the fine-tuning data from CSV
    prepare_fine_tuning_data(input_csv, output_jsonl)

    try:
        # Upload the training file to OpenAI
        with open(output_jsonl, "rb") as f:
            file_response = openai.File.create(
                file=f,
                purpose='fine-tune'
            )
        training_file_id = file_response.id
        print(f"Uploaded fine-tuning file ID: {training_file_id}")

        # Create a fine-tuning job
        fine_tune_response = openai.FineTune.create(
            training_file=training_file_id,
            model="gpt-4o-mini",  
            hyperparameters={
                "n_epochs": 2,  # Number of epochs (adjust based on your data and needs)
                "batch_size": 8,  # Batch size (adjust for large/small datasets)
                "learning_rate_multiplier": 0.1  # Learning rate multiplier for fine-tuning
            }
        )
        print(f"Fine-tuning job created: {fine_tune_response}")

    except (APIError, RateLimitError, APIConnectionError) as e:
        print(f"OpenAI API error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    fine_tune_model()


if __name__ == "__main__":
    main()