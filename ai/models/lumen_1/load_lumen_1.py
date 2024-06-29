import os
import json
from openai import OpenAI
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


def load_fine_tuned_model(model_path):
    if client is None:
        print("Client is not initialized. Exiting...")
        return

    try:
        # Load the saved model from file
        with open(model_path, 'r') as f:
            model = json.load(f)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def main():
    model_path = 'models/lumen_1/saved_model_lumen_1.json'
    model = load_fine_tuned_model(model_path)


if __name__ == "__main__":
    main()
