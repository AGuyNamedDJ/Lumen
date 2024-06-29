import openai
import os

# Load the OpenAI API key from .env
openai.api_key = os.getenv('OPENAI_API_KEY')


def get_ai_response(prompt, model="gpt-3.5-turbo", max_tokens=150):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens
    )
    return response.choices[0].text.strip()
