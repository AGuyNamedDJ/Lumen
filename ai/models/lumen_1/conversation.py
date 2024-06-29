from openai_integration.openai_client import get_ai_response


def handle_conversation(user_message):
    # Processes the user message and returns the AI response
    response = get_ai_response(user_message)
    return response
