from flask import Flask, request, jsonify
from flask_cors import CORS
from models.lumen_1.conversation import process_conversation
import logging
import os
from urllib.parse import quote as url_quote

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.DEBUG)


@app.route('/conversation', methods=['POST'])
def conversation():
    logging.debug("Received a request at /conversation endpoint")
    data = request.json
    logging.debug(f"Request JSON data: {data}")

    user_message = data.get('message')
    logging.debug(f"User message: {user_message}")
    if not user_message:
        logging.debug("No user message found in the request")
        return jsonify({'error': 'User message is required'}), 400

    ai_response = process_conversation(user_message)
    logging.debug(f"AI response: {ai_response}")
    return jsonify({'response': ai_response})


@app.route('/classify', methods=['POST'])
def classify():
    logging.debug("Received a request at /classify endpoint")
    data = request.json
    logging.debug(f"Request JSON data: {data}")

    user_message = data.get('message')
    logging.debug(f"User message: {user_message}")
    if not user_message:
        logging.debug("No user message found in the request")
        return jsonify({'error': 'User message is required'}), 400

    # Example classification logic based on keywords
    stock_keywords = ["SPX", "stock", "SPY", "bull", "option", ",buy",
                      "sell", "open", "close", "bear", "market", "S&P", "trading", "$"]
    category = 'stock-related' if any(
        keyword in user_message for keyword in stock_keywords) else 'general'

    logging.debug(f"Message classified as: {category}")
    return jsonify({'category': category})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))  # Updated to 8000 for consistency
    logging.debug(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port)
