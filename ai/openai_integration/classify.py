import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=api_key)

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def classify_message(message):
    logging.debug(f"Classifying message: {message}")
    try:
        response = client.chat.completions.create(model="gpt-4o-mini",
                                                  messages=[
                                                      {"role": "system",
                                                          "content": "You are a helpful assistant."},
                                                      {"role": "user",
                                                          "content": f"Is the following message related to stocks? Answer with 'True' or 'False'.\nMessage: {message}\nAnswer:"}
                                                  ],
                                                  max_tokens=1,
                                                  temperature=0)
        classification = response.choices[0].message["content"].strip().lower()
        logging.debug(f"Classification response: {classification}")
        return {"is_stock_related": classification == 'true'}
    except Exception as e:
        logging.error(f"Error in message classification: {e}")
        return {"is_stock_related": False, "error": str(e)}
