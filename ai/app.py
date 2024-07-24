import os
import logging
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}},
     supports_credentials=True)  # Enable CORS


def classify_message(message):
    logging.debug(f"Classifying message: {message}")
    try:
        content = (
            "Is the following message related to stocks? Answer with 'True' or 'False'.\n"
            "Message: " + message + "\nAnswer:"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}
            ],
            max_tokens=1,
            temperature=0
        )
        classification = response.choices[0].message.content.strip().lower()
        logging.debug(f"Classification response: {classification}")
        return {"is_stock_related": classification == 'true'}
    except Exception as e:
        logging.error(f"Error in message classification: {e}")
        return {"is_stock_related": False, "error": str(e)}


def process_lumen_model(message, reference_price=None):
    try:
        # Remove 'Given the market data: ' part from the message
        data_part = message.split(": ", 1)[1]

        # Splitting the message and creating a dictionary with proper conversion
        data_points = data_part.split(", ")
        data_dict = {}
        for dp in data_points:
            try:
                if ':' in dp:  # Ensure it has the correct format
                    key, value = dp.split(": ")
                    data_dict[key.lower()] = float(value)
                else:
                    logging.warning(f"Skipping non-data point: {dp}")
            except ValueError as e:
                logging.error(f"Error parsing data point '{dp}': {e}")
                raise ValueError(f"Error parsing data point '{dp}': {e}")

        logging.debug(f"Extracted data: {data_dict}")

        # Provide default values if any data points are missing
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

        # Placeholder for the Lumen model prediction
        normalized_closing_price = 0.999  # Placeholder normalized value

        if reference_price:
            predicted_price = normalized_closing_price * reference_price
            current_close = data_dict['close'] * reference_price
            percentage_change = (
                (predicted_price - current_close) / current_close) * 100
        else:
            predicted_price = normalized_closing_price  # Default to normalized value
            percentage_change = None

        return {
            "predicted_closing_price": round(predicted_price, 2),
            "percentage_change": round(percentage_change, 2) if percentage_change is not None else None
        }
    except Exception as e:
        logging.error(f"Error processing conversation with Lumen model: {e}")
        return {"error": str(e)}


@app.route('/conversation', methods=['POST'])
def conversation():
    request_data = request.get_json()
    message = request_data.get('message')
    reference_price = request_data.get('reference_price')
    logging.debug(f"Received a request at /conversation endpoint")
    logging.debug(f"Request JSON data: {request_data}")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    classification_result = classify_message(message)
    logging.debug(f"Classification result: {classification_result}")

    if classification_result.get('is_stock_related'):
        logging.debug(f"Message is stock-related, processing with Lumen model")
        lumen_result = process_lumen_model(message, reference_price)
        if 'error' in lumen_result:
            return jsonify({"error": lumen_result['error']}), 500
        return jsonify(lumen_result), 200
    else:
        logging.debug(f"Message is general, processing with ChatGPT-4o-mini")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": message}],
                max_tokens=150,
                temperature=0.7
            )
            ai_response = response.choices[0].message.content.strip()
            logging.debug(f"AI response: {ai_response}")
            return jsonify({"response": ai_response}), 200
        except Exception as e:
            logging.error(
                f"Error processing conversation with ChatGPT-4o-mini: {e}")
            return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
