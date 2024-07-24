import os
from flask import Flask, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)


@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Flask and Flask-CORS are working correctly"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
