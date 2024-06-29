import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='../.env')

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")

# Print the Python path
print(f"Python path: {sys.path}")

# Check if .env is found and loaded
if os.path.exists('../.env'):
    print(".env file found!")
else:
    print(".env file NOT found!")

# Print the environment variable for the OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
print(f"OPENAI_API_KEY: {api_key}")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)


def test_openai_response(prompt):
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-3.5-turbo",
            max_tokens=50,
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    prompt = "Hello, how are you?"
    response = test_openai_response(prompt)
    print(f"OpenAI Response: {response}")
