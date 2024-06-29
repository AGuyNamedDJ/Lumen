import os
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    client = OpenAI(api_key=api_key)
else:
    print("API key is not set. Please check the .env file.")
    client = None


# Function to prepare data for prediction
def prepare_data_for_prediction(input_csv, subset_size):
    df = pd.read_csv(input_csv)
    # Use only the first `subset_size` rows of data
    # Our data is currently too massive and exceeding our token limit
    df_subset = df.head(subset_size)
    return df_subset


# Path to the preprocessed data CSV
data_path = os.path.join(os.path.dirname(
    __file__), '../../data/lumen_1/processed/preprocessed_spx_data.csv')

# Load data for prediction with a smaller subset size
subset_size = 50  # Adjust this size to ensure it fits within the token limit
data_for_prediction = prepare_data_for_prediction(data_path, subset_size)

# Convert to JSON expected by OpenAI with necessary fields
data_for_prediction_json = data_for_prediction.to_json(orient='records')


def get_predictions(data):
    try:
        response = client.chat.completions.create(
            model='ft:gpt-3.5-turbo-0125:personal::9fB2NCNH',
            messages=[
                {"role": "user", "content": f"Given the following market data: {data}, predict the closing price of SPX for tomorrow."}
            ]
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None


def convert_to_percentage_change(raw_prediction, last_close_price):
    try:
        predicted_price = float(raw_prediction)
        percentage_change = (
            (predicted_price - last_close_price) / last_close_price) * 100
        return percentage_change
    except ValueError:
        print("Error converting prediction to float.")
        return None


# Get the last close price from the dataset
last_close_price = data_for_prediction.iloc[-1]['close']

# Get predictions
if client:
    predictions = get_predictions(data_for_prediction_json)
    if predictions:
        percentage_change = convert_to_percentage_change(
            predictions, last_close_price)
        if percentage_change is not None:
            print(f"Predicted percentage change: {percentage_change:.2f}%")
    else:
        print("Prediction could not be made.")
else:
    print("Client is not initialized. Exiting...")
