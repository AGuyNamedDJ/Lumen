import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

# Environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def prepare_fine_tuning_data(input_csv, output_jsonl):
    """
    Prepare the data for fine-tuning.

    :param input_csv: Path to the input CSV file.
    :param output_jsonl: Path to the output JSONL file.
    """
    df = pd.read_csv(input_csv)
    with open(output_jsonl, 'w') as f:
        for i, row in df.iterrows():
            f.write(
                f'{{"prompt": "The close price is {row["close"]}", "completion": "{row["close"]}"}}\n')


def fine_tune_model(training_file_id):
    """
    Fine-tune the model using the training file.

    :param training_file_id: ID of the training file uploaded to OpenAI.
    """
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model="gpt-3.5-turbo"
    )
    print("Fine-tuning job response:", response)


def main():
    input_csv = 'data/processed/preprocessed_spx_data.csv'
    output_jsonl = 'data/processed/spx_fine_tuning_data.jsonl'

    # Prepare the fine-tuning data
    prepare_fine_tuning_data(input_csv, output_jsonl)

    # Upload the fine-tuning data
    response = client.files.create(
        file=open(output_jsonl, 'rb'),
        purpose='fine-tune'
    )
    training_file_id = response.id
    print("Uploaded fine-tuning file ID:", training_file_id)

    # Start fine-tuning
    fine_tune_model(training_file_id)


if __name__ == "__main__":
    main()
