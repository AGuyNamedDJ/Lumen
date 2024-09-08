import os
import logging
import requests
from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai_integration.classify import stock_related_keywords, classify_message
from models.lumen_2.conversation import gpt_4o_mini_response, predict_next_day_close
from models.lumen_2.load_lumen_2 import load_lumen2_model
from models.lumen_2.definitions_lumen_2 import ReduceMeanLayer

# Load environment variables
load_dotenv()

# Load the Lumen2 model
lumen2_model = load_lumen2_model()

# Check if model is loaded
if lumen2_model is None:
    raise RuntimeError("Failed to load Lumen2 model")

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
    logging.debug("Classifying message: %s", message)
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
        logging.debug("Classification response: %s", classification)
        return {"is_stock_related": classification == 'true'}
    except Exception as e:
        logging.error("Error in message classification: %s", e)
        return {"is_stock_related": False, "error": str(e)}


def get_current_spx_price():
    """
    Fetches the current SPX price.
    """
    try:
        response = requests.get(
            'https://lumen-0q0f.onrender.com/api/spx-price')
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        return data.get('price')
    except requests.RequestException as e:
        logging.error("Error fetching current SPX price: %s", e)
        return None


def process_lumen2_model(message, model):
    """
    Process the stock-related message using the Lumen2 model.
    """
    try:
        # Extract features from the message (this step will depend on how your data is structured)
        input_data = extract_features_from_message(message)

        # Predict using the Lumen2 model
        prediction = model.predict(input_data)

        # Return the result in a user-friendly format
        return {"predicted_closing_price": prediction[0]}
    except Exception as e:
        logging.error(f"Error processing with Lumen2 model: {e}")
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
    """
    Main conversation endpoint for Lumen 2 model interaction.
    """
    try:
        # Log the start of the request
        logging.debug("Received a request at /conversation endpoint")

        # Extract request data and message
        request_data = request.get_json()
        logging.debug("Request JSON data: %s", request_data)

        message = request_data.get('message')
        if not message:
            logging.error("No message provided in the request")
            return jsonify({"error": "No message provided"}), 400

        # Classify the message to determine if it’s stock-related
        logging.debug("Classifying message: %s", message)
        classification_result = classify_message(message)
        logging.debug(f"Classification result: {classification_result}")

        # Fetch the current SPX price if the message is stock-related
        if classification_result.get('is_stock_related'):
            logging.debug(
                "Message classified as stock-related, processing with Lumen 2 model")

            # Fetch current SPX price
            current_spx_price = get_current_spx_price()
            logging.debug(f"Current SPX price: {current_spx_price}")

            if not current_spx_price:
                logging.error("Could not fetch current SPX price")
                return jsonify({"error": "Could not fetch current SPX price"}), 500

            # Now use Lumen2 model to predict the next day's close or any other task
            input_data = {
                'current_price': current_spx_price,
                # Add other input features required by the model
            }

            # Make the prediction using Lumen2
            predicted_price = lumen2_model.predict(input_data)
            logging.debug(f"Predicted closing price: {predicted_price}")

            # Return the Lumen2 model result
            return jsonify({"response": f"The predicted closing price is {predicted_price}."}), 200

        else:
            # If not stock-related, fall back to GPT-4o-mini
            logging.debug(
                "Message not stock-related, falling back to GPT-4o-mini")
            gpt_result = gpt_4o_mini_response(message)
            logging.debug(f"GPT-4o-mini result: {gpt_result}")

            if 'success' in gpt_result and gpt_result['success']:
                return jsonify({"response": gpt_result['response']}), 200
            else:
                logging.error(f"GPT-4o-mini error: {gpt_result.get('error')}")
                return jsonify({"error": gpt_result.get('error', 'Unknown error')}), 500

    except Exception as e:
        logging.error(f"Unexpected error in /conversation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
