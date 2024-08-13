import logging
# Assuming new prediction functions
from .predict_with_lumen_2 import predict_next_day_close, predict_other_scenarios


def process_conversation(user_message):
    logging.debug(f"Processing conversation for user message: {user_message}")

    # Extract relevant data from the user message for prediction
    try:
        data = extract_data_from_message(user_message)
        logging.debug(f"Extracted data: {data}")

        # Decide which prediction function to call based on the data or message content
        if 'scenario' in data:
            # Hypothetical function for other predictions
            prediction = predict_other_scenarios(data)
        else:
            prediction = predict_next_day_close(data)

        response = f"Based on the market data provided, the predicted closing price of SPX tomorrow is: {
            prediction}."
        logging.debug(f"Lumen model prediction: {response}")

    except ValueError as ve:
        logging.error(f"ValueError in processing conversation: {ve}")
        response = "It seems there was an issue with the data format. Please check your input and try again."
    except Exception as e:
        logging.error(
            f"Error in processing conversation with Lumen model: {e}")
        response = "An error occurred while processing your request with the Lumen model."

    return response


def extract_data_from_message(message):
    try:
        # Assuming more complex message parsing is needed
        parts = message.split(',')
        data = {part.split(':')[0].strip().lower(): float(
            part.split(':')[1].strip()) for part in parts[1:]}

        logging.debug(f"Extracted data from message: {data}")
        return data

    except ValueError as ve:
        logging.error(f"Error extracting data from message - ValueError: {ve}")
        raise ValueError("Invalid message format")
    except Exception as e:
        logging.error(f"Error extracting data from message: {e}")
        raise ValueError("Unexpected error in message parsing")
