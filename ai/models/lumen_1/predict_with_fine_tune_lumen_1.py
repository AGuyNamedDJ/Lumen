import os
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Function to prepare data for prediction


def prepare_data_for_prediction(input_csv):
    df = pd.read_csv(input_csv)
    # Assuming your data preprocessing steps here
    return df


# Load data for prediction
data_for_prediction = prepare_data_for_prediction(
    'data/processed/preprocessed_spx_data.csv')

# Convert your data to the format expected by OpenAI
# Here, convert to JSON with necessary fields
data_for_prediction_json = data_for_prediction.to_json(orient='records')

# Function to use the fine-tuned model for prediction


def get_predictions(data):
    response = client.chat.completions.create(
        model='ft:gpt-3.5-turbo:your-org:custom_suffix:id',  # Replace fine-tuned model IDs
        messages=[
            {"role": "user", "content": f"Predict SPX values: {data}"}
        ]
    )
    return response.choices[0].message['content']


# Get predictions
predictions = get_predictions(data_for_prediction_json)
print(predictions)
