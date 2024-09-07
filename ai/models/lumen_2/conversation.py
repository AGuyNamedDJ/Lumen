import os
import logging
import openai
from datetime import datetime, timedelta
import pytz
import random
import dateparser
from .predict_with_lumen_2 import predict_next_day_close, predict_other_scenarios
from .state_manager import get_current_spx_price, get_market_state
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize a time zone (assuming the market operates in New York)
timezone = pytz.timezone('America/Chicago')

# Define SQLAlchemy base and session
def get_db_connection():
    db_url = os.getenv('DB_URL')
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    return Session()

# Define state object
market_state = {
    'spx': {
        'current_price': None,
        'last_updated': None
    },
    'spy': {
        'current_price': None,
        'last_updated': None
    },
    'vix': {
        'current_price': None,
        'last_updated': None
    },
    'market_open': '08:30:00',
    'market_close': '15:00:00'
}

# Define conversational responses (whitelist approach)
price_prediction_responses = [
    "Based on the latest data, I expect {symbol_upper} to close at ${predicted_price:.2f} on {requested_date}.",
    "It looks like {symbol_upper} will close at around ${predicted_price:.2f} on {requested_date}, according to the data.",
    "For {requested_date}, I predict {symbol_upper} will close at ${predicted_price:.2f}.",
    "The forecast shows that {symbol_upper} will likely close at ${predicted_price:.2f} on {requested_date}.",
    "My calculations indicate that {symbol_upper} will close near ${predicted_price:.2f} on {requested_date}.",
    "The prediction suggests {symbol_upper} will finish the day at ${predicted_price:.2f} on {requested_date}.",
    "I'm expecting {symbol_upper} to close at about ${predicted_price:.2f} on {requested_date}.",
    "The data points toward {symbol_upper} closing at ${predicted_price:.2f} on {requested_date}.",
    "{symbol_upper} is predicted to close at ${predicted_price:.2f} on {requested_date}, based on current trends.",
    "According to the latest analysis, {symbol_upper} will likely close at ${predicted_price:.2f} on {requested_date}.",
    "The anticipated closing price for {symbol_upper} on {requested_date} is around ${predicted_price:.2f}.",
    "It seems like {symbol_upper} is trending towards a closing price of ${predicted_price:.2f} on {requested_date}."
]

current_price_responses = [
    "The current price of {symbol_upper} is ${current_price:.2f}.",
    "{symbol_upper} is currently priced at ${current_price:.2f}.",
    "Right now, {symbol_upper} is trading at ${current_price:.2f}.",
    "At this moment, {symbol_upper} stands at ${current_price:.2f}.",
    "{symbol_upper} is trading at approximately ${current_price:.2f} at the moment.",
    "The latest data shows {symbol_upper} is priced at ${current_price:.2f}.",
    "Currently, {symbol_upper} is valued at ${current_price:.2f}.",
    "As of now, {symbol_upper} is sitting at ${current_price:.2f}.",
    "The real-time price of {symbol_upper} is ${current_price:.2f}.",
    "At present, {symbol_upper} is trading around ${current_price:.2f}.",
    "The most recent price for {symbol_upper} is ${current_price:.2f}.",
]

market_hours_responses = [
    "The market is open right now.",
    "Yes, the market is open at the moment.",
    "The market is currently trading.",
    "The market is closed at the moment.",
    "No, the market has already closed for today.",
    "The market is currently not trading.",
    "The trading floor is open right now.",
    "Yes, the market is live right now.",
    "No, the market is closed at this time.",
    "The market has closed, trading will resume tomorrow.",
    "The market is closed, but will open again during the next trading session.",
    "The market isn't open at the moment."
]


# Dynamic market data retrieval function
def get_table_for_symbol(symbol):
    """Maps the symbol to the appropriate table name."""
    table_mapping = {
        'spx': 'real_time_spx',
        'spy': 'real_time_spy',
        'vix': 'real_time_vix'
    }
    return table_mapping.get(symbol.lower(), None)

def update_market_state(symbol):
    """
    Fetches the most recent entry from the database for the given symbol (SPX, SPY, or VIX).
    Updates the market_state for that symbol with the current price and timestamp.
    """
    session = get_db_connection()

    try:
        # Get the appropriate table for the symbol
        table_name = get_table_for_symbol(symbol)
        if not table_name:
            logging.error(f"Invalid symbol '{symbol}' provided. Cannot fetch data.")
            return

        # SQL query to get the most recent current_price and timestamp
        query = text(f"SELECT current_price, timestamp FROM {table_name} ORDER BY timestamp DESC LIMIT 1")

        # Fetch the most recent data for the symbol (SPX, SPY, or VIX)
        result = session.execute(query).fetchone()

        if result:
            # Update the market state with the latest price and timestamp
            market_state[symbol]['current_price'] = result['current_price']
            market_state[symbol]['last_updated'] = result['timestamp']
            logging.debug(f"Market state updated for {symbol.upper()}: {market_state[symbol]}")
        else:
            logging.error(f"No data found for {symbol.upper()} in the database.")
        
    except Exception as e:
        logging.error(f"Error updating market state for {symbol.upper()}: {e}")
    finally:
        session.close()

def get_current_timestamp():
    """
    Returns the current timestamp in the appropriate timezone.
    """
    return datetime.now(timezone)

def extract_date_from_message(message):
    """
    Extracts a date from a user message and returns the parsed date.
    If no valid date is found, it returns None.
    Automatically logs the current timestamp when the function is called.
    """
    # Log the current timestamp for the incoming query
    current_timestamp = get_current_timestamp()
    logging.debug(f"Current Timestamp for message: {current_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    # Extract the date from the message using natural language processing
    parsed_date = dateparser.parse(message, settings={'PREFER_DATES_FROM': 'future'})

    if parsed_date:
        return parsed_date.date()
    else:
        return None
        
# Helper to get the current date and time
def get_current_time():
    return datetime.now(timezone)

def get_todays_date():
    return get_current_time().strftime('%Y-%m-%d')

def get_tomorrows_date():
    return (get_current_time() + timedelta(days=1)).strftime('%Y-%m-%d')

def is_market_open():
    now = get_current_time()
    open_time = datetime.strptime(market_state['market_open'], '%H:%M:%S').time()
    close_time = datetime.strptime(market_state['market_close'], '%H:%M:%S').time()
    return open_time <= now.time() <= close_time

# Question categorization function with keyword lists for each category
def categorize_question(message):
    """
    Categorizes user queries into categories like market analysis, news, general inquiries, options, or others.
    """
    message_lower = message.lower()

    # Define keyword lists for each category
    market_analysis_keywords = [
    "price", "prices", "close", "closing", "forecast", "forecasts", "predict", "prediction", 
    "predictions", "trends", "trend", "movement", "movements", "target", "targets", 
    "valuation", "valuations", "estimate", "estimates", "outlook", "outlooks", 
    "future price", "future prices", "closing price", "closing prices", "projections", 
    "projection", "market direction", "directions", "expected", "expecting", 
    "anticipation", "anticipating", "next", "upcoming", "evaluation", "evaluations", 
    "technical analysis", "analyze", "analyzing", "evaluating"]
    current_price_keywords = [
    "current price", "live price", "price now", "real-time price", "latest price", 
    "price at the moment", "today's price", "price update", "current value", 
    "price check", "spot price", "market price", "price right now", "present price", 
    "price as of now", "latest value", "recent price", "live value", "current market value", 
    "current stock price", "price today", "what's the price", "today’s stock price"]
    market_hours_keywords = [
    "market", "open", "close", "trading hours", "market hours", "market open time", 
    "market close time", "opening bell", "closing bell", "when does the market open", 
    "when does the market close", "trading start time", "trading end time", 
    "stock market open", "stock market close", "market schedule", 
    "market opening", "market closing", "stock market hours", "market session", 
    "opening time", "closing time", "market timings", "trading session"]
    date_keywords = [
    "date", "today", "tomorrow", "yesterday", "day", "this week", "next week", 
    "next day", "current date", "what day", "what date", "this month", 
    "next month", "last week", "last month", "previous day", "previous date", 
    "day after tomorrow", "day before yesterday", "next trading day", 
    "last trading day", "weekend", "weekday", "next business day", 
    "upcoming date", "past date", "historical date"]
    options_keywords = [
    "options", "strike price", "expiry", "contract", "call option", "put option", 
    "option chain", "option premium", "expiration date", "implied volatility", 
    "open interest", "in the money", "out of the money", "at the money", 
    "long call", "long put", "short call", "short put", "covered call", 
    "naked call", "naked put", "iron condor", "butterfly spread", 
    "vertical spread", "horizontal spread", "bull call spread", "bull put spread", 
    "bear call spread", "bear put spread", "credit spread", "debit spread", 
    "calendar spread", "straddle", "strangle", "ratio spread", "protective put", 
    "synthetic call", "synthetic put", "collar", "covered put", 
    "exercise", "assignment", "option greeks", "delta", "gamma", "theta", 
    "vega", "rho", "delta neutral", "volatility skew", "risk reversal", 
    "synthetic long", "synthetic short", "iron butterfly", "leaps", 
    "rollover", "expiration cycle", "time decay", "intrinsic value", 
    "extrinsic value", "bid-ask spread", "gamma squeeze", "vega risk", 
    "skew", "skewness", "risk premium", "long straddle", "short straddle", 
    "long strangle", "short strangle", "cash-secured put", "poor man's covered call"]

    # Market analysis category
    if any(keyword in message_lower for keyword in market_analysis_keywords):
        return "market_analysis"

    # Current price category
    elif any(keyword in message_lower for keyword in current_price_keywords):
        return "current_price"

    # Market hours category
    elif any(keyword in message_lower for keyword in market_hours_keywords):
        return "market_hours"

    # Date inquiry category
    elif any(keyword in message_lower for keyword in date_keywords):
        return "date_inquiry"

    # Options-related questions category
    elif any(keyword in message_lower for keyword in options_keywords):
        return "options"

    # General category (fallback)
    else:
        return "general"
    

# Helper functions
def choose_random_response(response_list, **kwargs):
    response_template = random.choice(response_list)
    return response_template.format(**kwargs)

def use_nlg_model(user_message):
    """
    Generates a natural language response using OpenAI's GPT-4o-mini model.
    """
    try:
        response = openai.Completion.create(
            model="gpt-4o-mini",
            prompt=user_message,
            max_tokens=150,
            temperature=0.7
        )
        generated_text = response.choices[0].text.strip()
        return generated_text
    except Exception as e:
        logging.error(f"Error generating NLG response: {e}")
        return "Sorry, I couldn't generate a response at the moment."

def gpt_40_mini_response(user_message):
    """
    Fallback to GPT-4o-mini if the system cannot find a suitable whitelist response.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"Error with GPT-4o-mini: {e}")
        return "I'm sorry, I'm having trouble processing your request right now."
    

# In-memory storage for session memory
session_memory = {}

def start_new_session(user_id):
    """
    Starts a new session by creating an empty history for the user.
    """
    session_memory[user_id] = {"history": []}

def process_conversation(user_id, user_message):
    """
    Process the user's message within the context of their session.
    """
    logging.debug(f"Processing conversation for user {user_id} with message: {user_message}")

    # Check if the session exists for the user
    if user_id not in session_memory:
        logging.debug(f"Starting a new session for user {user_id}")
        start_new_session(user_id)

    # Log the user message into their session history
    session_memory[user_id]["history"].append({"user": user_message})

    # Categorize the question
    question_category = categorize_question(user_message)
    logging.debug(f"Question category: {question_category}")

    # Determine which symbol the user is asking about (SPX, SPY, or VIX)
    if "spx" in user_message.lower():
        symbol = 'spx'
    elif "spy" in user_message.lower():
        symbol = 'spy'
    elif "vix" in user_message.lower():
        symbol = 'vix'
    else:
        response = "Please specify SPX, SPY, or VIX."
        session_memory[user_id]["history"].append({"response": response})
        return response

    # Update the market state for the requested symbol
    update_market_state(symbol)

    # Handle the question based on its category
    if question_category == "market_analysis":
        requested_date = extract_date_from_message(user_message)
        if not requested_date:
            requested_date = get_tomorrows_date()
        response = handle_price_prediction_for_date(symbol, requested_date)
        
    elif question_category == "current_price":
        response = handle_current_price_request(symbol)

    elif question_category == "market_hours":
        response = handle_market_hours_request()

    elif question_category == "date_inquiry":
        requested_date = extract_date_from_message(user_message)
        if requested_date:
            response = f"The date you're asking about is {requested_date}."
        else:
            response = f"Today is {get_todays_date()} and tomorrow is {get_tomorrows_date()}."

    elif question_category == "options":
        response = handle_options_query(user_message)

    else:
        # If the question doesn't fit any known category, fall back to NLG
        response = use_nlg_model(user_message)

    # Log the response into session history
    session_memory[user_id]["history"].append({"response": response})

    logging.debug(f"Lumen model response for user {user_id}: {response}")
    return response

def retrieve_conversation_history(user_id):
    """
    Retrieve the conversation history for the given user.
    """
    if user_id in session_memory:
        return session_memory[user_id]["history"]
    return []

def clear_session(user_id):
    """
    Clears the session memory for a user, effectively ending their session.
    """
    if user_id in session_memory:
        del session_memory[user_id]
        logging.debug(f"Session cleared for user {user_id}")
    else:
        logging.debug(f"No session found for user {user_id}")

        
def handle_price_prediction_for_date(symbol, requested_date):
    logging.debug(f"Handling price prediction for {symbol.upper()} at close on {requested_date}")

    # Get current price for the symbol
    current_price = market_state[symbol]['current_price']
    if not current_price:
        return f"I'm unable to fetch the current {symbol.upper()} price at the moment."

    # Predict the closing price for the requested date
    predicted_price = predict_next_day_close(market_state[symbol], requested_date)

    response = choose_random_response(price_prediction_responses, symbol_upper=symbol.upper(), predicted_price=predicted_price, requested_date=requested_date)
    logging.debug(f"Prediction for {symbol.upper()} close on {requested_date}: {predicted_price:.2f}")

    return response

def handle_current_price_request(symbol):
    current_price = market_state[symbol]['current_price']
    if current_price:
        return choose_random_response(current_price_responses, symbol_upper=symbol.upper(), current_price=current_price)
    else:
        return f"Sorry, I couldn't retrieve the current {symbol.upper()} price."

def handle_market_hours_request():
    if is_market_open():
        return choose_random_response(market_hours_responses[:3])  # Use the open responses
    else:
        return choose_random_response(market_hours_responses[3:])  # Use the closed responses

def predict_next_day_close(market_state, target_date):
    """
    Uses the Lumen model to predict the SPX, SPY, or VIX closing price for a given target date based on current data.
    """
    # Get the current price
    current_price = market_state['current_price']
    
    # This should be where you plug in your machine learning model to predict based on indicators
    # Adjust the prediction logic depending on how far the target_date is from today.
    
    # Mock prediction (e.g., simple 1% increase for each day beyond today)
    days_ahead = (target_date - datetime.strptime(get_todays_date(), '%Y-%m-%d')).days
    predicted_price = current_price * (1 + 0.01 * days_ahead)

    return predicted_price