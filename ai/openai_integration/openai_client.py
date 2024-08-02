import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv(dotenv_path='../.env')

# Retrieve the API key
api_key = os.getenv('OPENAI_API_KEY')

if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not found")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def get_openai_response(prompt, max_tokens=800):
    logging.debug(f"Sending prompt to OpenAI: {prompt}")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        logging.debug(f"OpenAI response: {response}")
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in OpenAI response: {e}")
        return f"Error: {e}"


def handle_conversation(user_message):
    prompt = f"User: {user_message}\nAI:"
    logging.debug(f"Formatted prompt: {prompt}")
    return get_openai_response(prompt)
