import os
from openai import OpenAI
import logging
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from models.lumen_2.conversation import select_model_for_prediction, gpt_4o_mini_response, categorize_question

# Log the current directory and contents
current_dir = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.DEBUG)
logging.debug(f"Current directory: {current_dir}")
logging.debug(f"Files in current directory: {os.listdir(current_dir)}")

# Check and log models/lumen_2 directory
models_dir = os.path.join(current_dir, 'models', 'lumen_2')
logging.debug(f"Files in models/lumen_2: {os.listdir(models_dir)}")

# Try to import definitions_lumen_2 and catch any errors
try:
    from models.lumen_2.definitions_lumen_2 import ReduceMeanLayer
    logging.debug(
        "Successfully imported ReduceMeanLayer from definitions_lumen_2")
except ImportError as e:
    logging.error(f"Error importing ReduceMeanLayer: {e}")

# Load environment variables
logging.debug("Loading environment variables from .env...")
load_dotenv()

# Initialize the OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set")

client = OpenAI(
    api_key=api_key
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000",
     "https://lumen-1.netlify.app/"]}}, supports_credentials=True)


def classify_message(message):
    logging.debug("Classifying message using categorize_question function.")
    try:
        category = categorize_question(message)

        # Treat categories like "market_analysis" and "current_price" as stock-related
        if category in ["market_analysis", "current_price"]:
            logging.debug(f"Message classified as stock-related: {category}")
            return {"is_stock_related": True}
        else:
            logging.debug(
                f"Message classified as non-stock-related: {category}")
            return {"is_stock_related": False}
    except Exception as e:
        logging.error(f"Error in message classification: {e}")
        return {"is_stock_related": False, "error": str(e)}

# def classify_message(message):
#     logging.debug("Classifying message: %s", message)
#     try:
#         content = (
#             "Is the following message related to stocks? Answer with 'True' or 'False'.\n"
#             "Message: " + message + "\nAnswer:"
#         )

#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": content}
#             ],
#             max_tokens=1,
#             temperature=0
#         )

#         # Ensure you're using the right attribute for the response:
#         # Correct attribute
#         classification = response.choices[0].message['content'].strip().lower()

#         logging.debug("Classification response: %s", classification)
#         return {"is_stock_related": classification == 'true'}
#     except Exception as e:
#         logging.error("Error in message classification: %s", e)
#         return {"is_stock_related": False, "error": str(e)}


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

            # Determine which model to use
            if select_model_for_prediction:  # Define your logic for when to use the real-time model
                predicted_price = Lumen2_real_time.predict(input_data)
            else:
                predicted_price = Lumen2_historic.predict(input_data)

            logging.debug(f"Predicted closing price: {predicted_price}")

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
