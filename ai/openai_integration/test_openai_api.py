from openai import OpenAI
import os
from dotenv import load_dotenv

# Load env
load_dotenv()

# Set OpenAI API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def test_openai_api():
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ]
        )
        print("OpenAI API Response:")
        print(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_openai_api()
