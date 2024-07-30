import os
import logging
import requests
from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=api_key)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000",
                                         "https://lumen-1.netlify.app/"]}}, supports_credentials=True)


def classify_message(message):
    logging.debug(f"Classifying message: {message}")
    try:
        content = (
            "Is the following message related to stocks? Answer with 'True' or 'False'.\n"
            "Message: " + message + "\nAnswer:"
        )

        response = client.chat.completions.create(model="gpt-4o-mini",
                                                  messages=[
                                                      {"role": "system",
                                                       "content": "You are a helpful assistant."},
                                                      {"role": "user",
                                                       "content": content}
                                                  ],
                                                  max_tokens=1,
                                                  temperature=0)
        classification = response.choices[0].message.content.strip().lower()
        logging.debug(f"Classification response: {classification}")
        return {"is_stock_related": classification == 'true'}
    except Exception as e:
        logging.error(f"Error in message classification: {e}")
        return {"is_stock_related": False, "error": str(e)}


def get_spx_price():
    try:
        # Adjust URL as needed
        response = requests.get('http://localhost:3001/api/spx-price')
        response.raise_for_status()
        data = response.json()
        return data['price']
    except requests.RequestException as e:
        logging.error(f"Error fetching SPX price: {e}")
        return None


def get_current_spx_price():
    try:
        response = requests.get(
            'https://lumen-0q0f.onrender.com/api/spx-price')
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        return data.get('price')
    except requests.RequestException as e:
        logging.error(f"Error fetching current SPX price: {e}")
        return None


def process_lumen_model(message, reference_price=None):
    try:
        data_part = message.split(": ", 1)[1]
        data_points = data_part.split(", ")
        data_dict = {}
        for dp in data_points:
            try:
                if ':' in dp:
                    key, value = dp.split(": ")
                    data_dict[key.lower()] = float(value)
                else:
                    logging.warning(f"Skipping non-data point: {dp}")
            except ValueError as e:
                logging.error(f"Error parsing data point '{dp}': {e}")
                raise ValueError(f"Error parsing data point '{dp}': {e}")

        logging.debug(f"Extracted data: {data_dict}")

        default_values = {
            'open': 1.0,
            'high': 1.0,
            'low': 1.0,
            'close': 1.0,
            'volume': 0.0,
            'ema_10': 1.0,
            'ema_50': 1.0
        }

        for key, default_value in default_values.items():
            data_dict.setdefault(key, default_value)

        logging.debug(f"Data with defaults: {data_dict}")

        normalized_closing_price = 0.999

        if reference_price:
            predicted_price = normalized_closing_price * reference_price
            current_close = data_dict['close'] * reference_price
            percentage_change = (
                (predicted_price - current_close) / current_close) * 100
        else:
            predicted_price = normalized_closing_price
            percentage_change = None

        return {
            "predicted_closing_price": round(predicted_price, 2),
            "percentage_change": round(percentage_change, 2) if percentage_change is not None else None
        }
    except Exception as e:
        logging.error(f"Error processing conversation with Lumen model: {e}")
        return {"error": str(e)}


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


@app.route('/classify', methods=['POST'])
def classify():
    request_data = request.get_json()
    message = request_data.get('message')
    if not message:
        return jsonify({"error": "No message provided"}), 400
    result = classify_message(message)
    return jsonify(result), 200


@app.route('/conversation', methods=['POST'])
def conversation():
    request_data = request.get_json()
    message = request_data.get('message')
    logging.debug(f"Received a request at /conversation endpoint")
    logging.debug(f"Request JSON data: {request_data}")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Fetch the current SPX price
    current_spx_price = get_current_spx_price()
    if not current_spx_price:
        return jsonify({"error": "Could not fetch current SPX price"}), 500

    classification_result = classify_message(message)
    logging.debug(f"Classification result: {classification_result}")

    if classification_result.get('is_stock_related'):
        logging.debug(f"Message is stock-related, processing with Lumen model")
        lumen_result = process_lumen_model(message, current_spx_price)
        if 'error' in lumen_result:
            logging.debug(
                f"Lumen model encountered an error, falling back to ChatGPT-4o-mini")
            return fallback_to_gpt(message)

        # Convert dictionary response to a string
        lumen_response_text = f"Predicted closing price: {
            lumen_result['predicted_closing_price']}, Percentage change: {lumen_result['percentage_change']}"
        return jsonify({"response": lumen_response_text}), 200
    else:
        return fallback_to_gpt(message)


def fallback_to_gpt(message):
    logging.debug(f"Falling back to ChatGPT-4o-mini")
    try:
        response = client.chat.completions.create(model="gpt-4o-mini",
                                                  messages=[
                                                      {"role": "user", "content": message}],
                                                  max_tokens=150,
                                                  temperature=0.7)
        ai_response = response.choices[0].message.content.strip()
        logging.debug(f"AI response: {ai_response}")
        return jsonify({"response": ai_response}), 200
    except Exception as e:
        logging.error(
            f"Error processing conversation with ChatGPT-4o-mini: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
