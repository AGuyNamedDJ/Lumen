import os
import pandas as pd
from dotenv import load_dotenv
import json
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))
api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
client = OpenAI(api_key=api_key)


def prepare_fine_tuning_data(input_csv, output_jsonl):
    # Load the data
    df = pd.read_csv(input_csv)

    # Check for missing values
    if df.isnull().values.any():
        print("Warning: Missing values found in the dataset.")
        df = df.dropna()  # Drop rows with missing values

    # Print data summary
    print("Data Summary:")
    print(df.describe())
    print(f"Total data points: {len(df)}")

    # Prepare chat-formatted data
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

    # Save to JSONL file
    with open(output_jsonl, 'w') as f:
        for item in chat_data:
            f.write(json.dumps(item) + "\n")


def fine_tune_model():
    # Prepare the fine-tuning data
    input_csv = 'data/processed/preprocessed_spx_data.csv'
    output_jsonl = 'data/processed/fine_tune_data.jsonl'
    prepare_fine_tuning_data(input_csv, output_jsonl)

    # Upload the training file using new API method
    try:
        with open(output_jsonl, "rb") as f:
            response = client.files.create(
                file=f,
                purpose='fine-tune'
            )
        training_file_id = response.id
        print(f"Uploaded fine-tuning file ID: {training_file_id}")

        # Create a fine-tuning job using new API method
        response = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model="gpt-3.5-turbo",
            hyperparameters={
                "n_epochs": 2,  # Adjust if be
                "batch_size": 8,  # Adjust on data size
                "learning_rate_multiplier": 0.1  # Default or adjusted based on experiments
            }
        )
        print(f"Fine-tuning job response: {response}")

    except (APIError, RateLimitError, APIConnectionError) as e:
        print(f"OpenAI API error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    fine_tune_model()


if __name__ == "__main__":
    main()
