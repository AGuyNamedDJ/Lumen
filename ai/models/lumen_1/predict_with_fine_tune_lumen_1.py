import os
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from openai import OpenAI

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
    df_subset = df.head(subset_size)
    return df_subset


# Path to the preprocessed data CSV
data_path = os.path.join(os.path.dirname(
    __file__), '../../data/lumen_1/processed/preprocessed_spx_data.csv')

# Load data for prediction with a smaller subset size
subset_size = 50  # Adjust this size to ensure it fits within the token limit
data_for_prediction = prepare_data_for_prediction(data_path, subset_size)
data_for_prediction_json = data_for_prediction.to_json(orient='records')


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'trained_model.keras')
    model = tf.keras.models.load_model(model_path)
    return model


def get_predictions(model, data):
    try:
        # Convert JSON data to DataFrame
        df = pd.read_json(data, orient='records')
        # Extract features and make predictions
        # Adjust according to your features
        features = df[['feature1', 'feature2', 'feature3']]
        predictions = model.predict(features)
        return predictions[-1]  # Return the latest prediction
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None


def get_openai_response(prompt, max_tokens=800):
    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7  # 0.5 is predictable & 0.9 is varied  keep 0.7 for now
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in OpenAI response: {e}")
        return None


def convert_to_percentage_change(raw_prediction, last_close_price):
    try:
        predicted_price = float(raw_prediction)
        percentage_change = (
            predicted_price - last_close_price) / last_close_price * 100
        return percentage_change
    except ValueError:
        print("Error converting prediction to float.")
        return None


# Load the trained LSTM model
model = load_model()

# Get the last close price from the dataset
last_close_price = data_for_prediction.iloc[-1]['close']

# Get predictions
if client:
    model_predictions = get_predictions(model, data_for_prediction_json)
    if model_predictions:
        # Get the percentage change prediction
        percentage_change = convert_to_percentage_change(
            model_predictions, last_close_price)
        if percentage_change is not None:
            # Use OpenAI to interpret the prediction
            prompt = f"The model predicts a {percentage_change:.2f}% change in the SPX closing price tomorrow. Provide insights or potential reasons for this prediction."
            openai_response = get_openai_response(prompt)
            if openai_response:
                print(f"OpenAI Response: {openai_response}")
            else:
                print("OpenAI response could not be retrieved.")
        else:
            print("Percentage change could not be calculated.")
    else:
        print("Model predictions could not be made.")
else:
    print("Client is not initialized. Exiting...")
