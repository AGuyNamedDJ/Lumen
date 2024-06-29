import os
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load environment variables
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)

# Verify .env file loading
if os.path.exists(dotenv_path):
    print(f".env file loaded successfully from: {dotenv_path}")
else:
    print(f".env file not found at: {dotenv_path}")

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')

# Verify API key loading
if api_key:
    print(f"API Key: {api_key}")
    client = OpenAI(api_key=api_key)
else:
    print("API key is not set. Please check the .env file.")
    client = None


def save_fine_tuned_model(model_id, output_path):
    if client is None:
        print("Client is not initialized. Exiting...")
        return

    try:
        # Retrieve the fine-tuned model
        response = client.models.retrieve(model=model_id)
        model = response.to_dict()

        # Save the model to a file
        with open(output_path, 'w') as f:
            f.write(json.dumps(model, indent=4))
        print(f"Model saved to {output_path}")

    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    # Replace with fine-tuned model ID
    model_id = 'ft:gpt-3.5-turbo-0125:personal::9fB2NCNH'
    output_path = 'models/lumen_1/saved_model_lumen_1.json'
    save_fine_tuned_model(model_id, output_path)


if __name__ == "__main__":
    main()
