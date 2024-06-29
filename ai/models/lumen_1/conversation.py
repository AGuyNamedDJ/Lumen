from openai_integration.openai_client import handle_conversation
import logging


def process_conversation(user_message):
    logging.debug(f"Processing conversation for user message: {user_message}")
    return handle_conversation(user_message)
