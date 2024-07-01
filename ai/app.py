from flask import Flask, request, jsonify
from models.lumen_1.conversation import process_conversation
import logging
import os

app = Flask(__name__)

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


if __name__ == '__main__':
    # Use port from environment  or default
    port = int(os.environ.get('PORT', 8000))
    logging.debug(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port)
