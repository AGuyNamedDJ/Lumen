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

# Whitelist
stock_related_keywords = [
    # General Stock Terms
    "stock", "stocks", "price", "prices", "market", "shares", "equity", "equities", "index", "indices", "ETF", "ETFs", "fund", "funds",
    "dividend", "dividends", "stock price", "market price", "share price", "market cap", "capitalization", "liquidity",
    "buy", "sell", "buying", "selling", "trade", "trading", "bull", "bear", "bullish", "bearish",

    # Specific Stock Symbols and Indexes
    "SPX", "SPY", "VIX", "S&P",

    # Financial Metrics and Indicators
    "P/E ratio", "price-to-earnings", "EPS", "earnings per share", "revenue", "growth", "earnings", "profit", "profit margin",
    "gross margin", "operating margin", "net income", "debt", "debt-to-equity", "leverage", "ROE", "return on equity", "ROI",
    "return on investment", "cash flow", "EBITDA", "valuation", "market value", "book value",

    # Technical Analysis Terms
    "technical analysis", "support", "resistance", "trend", "uptrend", "downtrend", "chart", "patterns", "candlestick",
    "moving average", "MA", "EMA", "SMA", "MACD", "RSI", "Bollinger Bands", "Fibonacci", "volume", "volatility", "VIX",
    "momentum", "oscillator", "stochastic", "ADX", "CCI", "Ichimoku", "breakout", "pullback", "retracement", "divergence",
    "overbought", "oversold",

    # Economic and Market Events
    "yield curve", "interest rate", "Federal Reserve", "Fed", "CPI", "inflation", "GDP", "economic growth",

    # Options and Derivatives
    "options", "calls", "puts", "strike price", "expiration date", "contract", "premium", "option chain", "implied volatility",
    "IV", "open interest", "greeks", "delta", "gamma", "theta", "vega", "rho", "straddle", "strangle", "iron condor",
    "butterfly", "vertical spread", "horizontal spread", "bull call spread", "bear put spread", "covered call",

    # Market Types and Trading Strategies
    "long", "short", "position", "day trading", "swing trading", "scalping", "buy and hold", "hedging", "arbitrage", "high-frequency trading", "algorithmic trading", "dark pool", "liquidity",

    # Financial News and Events
    "market sentiment", "consumer sentiment", "economic indicator", "jobs report",
    "unemployment rate", "labor market", "housing market", "consumer confidence", "retail sales", "industrial production",
]


def classify_message(message, classification_prompt=None):
    """
    Classifies a message using a whitelist first, and if it doesn't match, uses GPT-4o-mini as a fallback.

    :param message: The user message to classify.
    :param classification_prompt: Custom prompt for classification (optional).
    :return: Dictionary with classification result.
    """
    logging.debug(f"Classifying message: {message}")

    # First, try local classification using the whitelist
    is_stock_related_local = any(keyword.lower() in message.lower()
                                 for keyword in stock_related_keywords)

    if is_stock_related_local:
        logging.debug(
            "Message classified as stock-related using local whitelist")
        return {"is_stock_related": True}

    # If not found locally, fall back to OpenAI classification
    try:
        # Default classification for stock-related messages
        if not classification_prompt:
            classification_prompt = f"Is the following message related to stocks? Answer with 'True' or 'False'.\nMessage: {
                message}\nAnswer:"

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": classification_prompt}
            ],
            max_tokens=1,
            temperature=0
        )

        classification = response['choices'][0]['message']['content'].strip(
        ).lower()
        logging.debug(f"Classification response from OpenAI: {classification}")

        return {"is_stock_related": classification == 'true'}

    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        return {"is_stock_related": False, "error": str(e)}

    except Exception as e:
        logging.error(f"Unexpected error during classification: {e}")
        return {"is_stock_related": False, "error": str(e)}
