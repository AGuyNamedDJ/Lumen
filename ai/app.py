from flask import Flask, request, jsonify
# Import AI conversation handler
from models.lumen_1.conversation import handle_conversation

app = Flask(__name__)  # Initialize


# Define endpoint for conversation requests
@app.route('/conversation', methods=['POST'])
def conversation():
    data = request.json  # Extract JSON data from request
    user_message = data.get('message')
    if not user_message:
        return jsonify({'error': 'User message is required'}), 400

    # Process message with AI handler
    ai_response = handle_conversation(user_message)
    return jsonify({'response': ai_response})  # Return AI response as JSON


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)  # Run
