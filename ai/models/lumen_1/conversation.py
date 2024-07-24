import logging
from .predict_with_fine_tune_lumen_1 import predict_next_day_close


def process_conversation(user_message):
    logging.debug(f"Processing conversation for user message: {user_message}")

    # Extract relevant data from the user message for prediction
    try:
        data = extract_data_from_message(user_message)
        logging.debug(f"Extracted data: {data}")
        prediction = predict_next_day_close(data)
        response = f"Based on the market data provided, the predicted closing price of SPX tomorrow is: {
            prediction}."
        logging.debug(f"Lumen model prediction: {response}")
    except Exception as e:
        logging.error(
            f"Error in processing conversation with Lumen model: {e}")
        response = "An error occurred while processing your request with the Lumen model."

    return response


def extract_data_from_message(message):
    try:
        parts = message.split(',')
        data = {part.split(':')[0].strip().lower(): float(
            part.split(':')[1].strip()) for part in parts[1:]}
        logging.debug(f"Extracted data from message: {data}")
        return data
    except Exception as e:
        logging.error(f"Error extracting data from message: {e}")
        raise ValueError("Invalid message format")
