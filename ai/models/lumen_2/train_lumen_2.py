import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from definitions_lumen_2 import create_hybrid_model
import logging
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt


# Load environment variables
load_dotenv()

# Base directory settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths from environment variables
DATA_DIR = os.getenv('DATA_DIR', 'data/lumen_2/processed')
MODEL_DIR = os.getenv('MODEL_DIR', 'models/lumen_2')
MODEL_NAME = 'lumen_2'

# Load historical SPX, SPY, and VIX data
historical_spx = pd.read_csv(os.path.join(
    DATA_DIR, 'processed_historical_spx.csv'))
historical_spy = pd.read_csv(os.path.join(
    DATA_DIR, 'processed_historical_spy.csv'))
historical_vix = pd.read_csv(os.path.join(
    DATA_DIR, 'processed_historical_vix.csv'))

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load preprocessed data


def load_data():
    average_hourly_earnings_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_average_hourly_earnings_data.csv'))
    consumer_confidence_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_consumer_confidence_data.csv'))
    consumer_sentiment_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_consumer_sentiment_data.csv'))
    core_inflation_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_core_inflation_data.csv'))
    cpi_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_cpi_data.csv'))
    gdp_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_gdp_data.csv'))
    industrial_production_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_industrial_production_data.csv'))
    interest_rate_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_interest_rate_data.csv'))
    labor_force_participation_rate_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_labor_force_participation_rate_data.csv'))
    nonfarm_payroll_employment_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_nonfarm_payroll_employment_data.csv'))
    personal_consumption_expenditures_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_personal_consumption_expenditures.csv'))
    ppi_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_ppi_data.csv'))
    real_time_spx = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_real_time_spx.csv'))
    real_time_spy = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_real_time_spy.csv'))
    real_time_vix = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_real_time_vix.csv'))
    unemployment_rate_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_unemployment_rate_data.csv'))

    return {
        'average_hourly_earnings_data': average_hourly_earnings_data,
        'consumer_confidence_data': consumer_confidence_data,
        'consumer_sentiment_data': consumer_sentiment_data,
        'core_inflation_data': core_inflation_data,
        'cpi_data': cpi_data,
        'gdp_data': gdp_data,
        'historical_spx': historical_spx,
        'historical_spy': historical_spy,
        'historical_vix': historical_vix,
        'industrial_production_data': industrial_production_data,
        'interest_rate_data': interest_rate_data,
        'labor_force_participation_rate_data': labor_force_participation_rate_data,
        'nonfarm_payroll_employment_data': nonfarm_payroll_employment_data,
        'personal_consumption_expenditures_data': personal_consumption_expenditures_data,
        'ppi_data': ppi_data,
        'real_time_spx': real_time_spx,
        'real_time_spy': real_time_spy,
        'real_time_vix': real_time_vix,
        'unemployment_rate_data': unemployment_rate_data,
    }


# Feature Engineering
def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_obv(df):
    # Determine the correct price column to use
    price_column = 'close' if 'close' in df.columns else 'current_price'

    obv = [0]
    for i in range(1, len(df)):
        if df[price_column].iloc[i] > df[price_column].iloc[i - 1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df[price_column].iloc[i] < df[price_column].iloc[i - 1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    return df


def calculate_vma(df, window=20):
    df[f'VMA_{window}'] = df['volume'].rolling(window=window).mean()
    return df


def calculate_vpt(df):
    # Determine the correct price column to use
    price_column = 'close' if 'close' in df.columns else 'current_price'

    vpt = [(df[price_column].iloc[0] - df[price_column].iloc[0])
           * df['volume'].iloc[0]]
    for i in range(1, len(df)):
        vpt.append(vpt[-1] + (df[price_column].iloc[i] - df[price_column].iloc[i-1]
                              ) / df[price_column].iloc[i-1] * df['volume'].iloc[i])
    df['VPT'] = vpt
    return df


def classify_market_condition(df, threshold=0.20):
    """
    Classify the market condition as bull or bear based on a 20% threshold.

    Parameters:
    - df: DataFrame containing the SPX/SPY price data.
    - threshold: Percentage change threshold to define bull/bear market.

    Returns:
    - Series with market condition classification.
    """
    # Calculate percentage change from recent peak
    df['peak'] = df['close'].cummax()
    df['trough'] = df['close'].cummin()
    df['drawdown'] = (df['close'] - df['peak']) / df['peak']
    df['recovery'] = (df['close'] - df['trough']) / df['trough']

    # Classify market condition
    conditions = []
    for index, row in df.iterrows():
        if row['drawdown'] <= -threshold:
            conditions.append('bear')
        elif row['recovery'] >= threshold:
            conditions.append('bull')
        else:
            conditions.append('neutral')
    
    return pd.Series(conditions, index=df.index, name='market_condition')


def classify_vix_regime(vix_value):
    if vix_value < 12:
        return 'low'
    elif 12 <= vix_value < 20:
        return 'moderate'
    elif 20 <= vix_value < 30:
        return 'high'
    else:
        return 'extreme'

def recompute_indicators(df, tf, has_close=True):
    # Timeframe dictionary to define rolling windows
    timeframe = {
        '1min': {'SMA_20': 20, 'SMA_50': 50, 'SMA_100': 100, 'EMA_12': 12, 'EMA_26': 26, 'RSI': 14},
        '5min': {'SMA_20': 100, 'SMA_50': 250, 'SMA_100': 500, 'EMA_12': 60, 'EMA_26': 130, 'RSI': 14},
        '30min': {'SMA_20': 600, 'SMA_50': 1500, 'SMA_100': 3000, 'EMA_12': 360, 'EMA_26': 780, 'RSI': 14},
        '1hour': {'SMA_20': 1200, 'SMA_50': 3000, 'SMA_100': 6000, 'EMA_12': 720, 'EMA_26': 1560, 'RSI': 14},
        'daily': {'SMA_20': 20, 'SMA_50': 50, 'SMA_100': 100, 'EMA_12': 12, 'EMA_26': 26, 'RSI': 14},
        'weekly': {'SMA_20': 104, 'SMA_50': 260, 'SMA_100': 520, 'EMA_12': 52, 'EMA_26': 104, 'RSI': 14},
        'monthly': {'SMA_20': 240, 'SMA_50': 600, 'SMA_100': 1200, 'EMA_12': 240, 'EMA_26': 520, 'RSI': 14}
    }

    # Get the appropriate settings for the timeframe
    settings = timeframe.get(tf, timeframe['daily'])

    # List of indicators to calculate
    indicators = []

    # Choose the appropriate price column based on whether 'close' exists
    price_column = 'close' if has_close else 'current_price'

    # Debug: print before processing
    print("Before indicator calculation:")
    print(df.head())

    # Ensure rolling operations have enough data
    if len(df) < settings['SMA_20']:
        print("Not enough data to calculate indicators for this timeframe.")
        return df

    # Calculate indicators and append to the list using the timeframe settings
    indicators.append(
        df[price_column].rolling(
            window=settings['SMA_20']).mean().rename('SMA_20')
    )
    indicators.append(
        df[price_column].rolling(
            window=settings['SMA_50']).mean().rename('SMA_50')
    )
    indicators.append(
        df[price_column].rolling(window=settings['SMA_100']
                                 ).mean().rename('SMA_100')
    )
    indicators.append(
        df[price_column].ewm(span=settings['EMA_12'],
                             adjust=False).mean().rename('EMA_12')
    )
    indicators.append(
        df[price_column].ewm(span=settings['EMA_26'],
                             adjust=False).mean().rename('EMA_26')
    )
    indicators.append(
        (df[price_column].rolling(window=settings['SMA_20']).mean()
         + 2 * df[price_column].rolling(window=settings['SMA_20']).std()).rename('Bollinger_Upper')
    )
    indicators.append(
        (df[price_column].rolling(window=settings['SMA_20']).mean()
         - 2 * df[price_column].rolling(window=settings['SMA_20']).std()).rename('Bollinger_Lower')
    )
    indicators.append(
        calculate_rsi(df[price_column], period=settings['RSI']).rename('RSI')
    )
    # Concatenate all indicators at once using pd.concat with batching
    batch_size = 4  # Number of indicators to concatenate in each batch
    for i in range(0, len(indicators), batch_size):
        df = pd.concat([df] + indicators[i:i + batch_size], axis=1)

    # Debug: print after processing
    print("After indicator calculation:")
    print(df.head())

    return df.dropna()  # Drop rows with NaNs after recalculating indicators

# Correlations


def average_hourly_earnings_correlation(data_dict):
    # Extract the relevant data
    average_hourly_earnings = data_dict['average_hourly_earnings_data']
    spx_data = data_dict['historical_spx']
    spy_data = data_dict['historical_spy']

    # Truncate to the shortest length to avoid length mismatch
    min_length = min(len(average_hourly_earnings),
                     len(spx_data), len(spy_data))

    # Truncate datasets
    average_hourly_earnings = average_hourly_earnings.iloc[:min_length]
    spx_data = spx_data.iloc[:min_length]
    spy_data = spy_data.iloc[:min_length]

    # Calculate correlations
    spx_corr = average_hourly_earnings['value'].corr(spx_data['close'])
    spy_corr = average_hourly_earnings['value'].corr(spy_data['close'])

    return spx_corr, spy_corr

def lagged_correlation_analysis(vix_df, spx_df, spy_df, max_lag=10):
    correlations = {'Lag': [], 'VIX-SPX': [], 'VIX-SPY': []}

    for lag in range(0, max_lag + 1):
        shifted_vix = vix_df['close'].shift(lag)

        vix_spx_corr = shifted_vix.corr(spx_df['close'])
        vix_spy_corr = shifted_vix.corr(spy_df['close'])

        correlations['Lag'].append(lag)
        correlations['VIX-SPX'].append(vix_spx_corr)
        correlations['VIX-SPY'].append(vix_spy_corr)

    corr_df = pd.DataFrame(correlations)

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_df.set_index('Lag'), annot=True, cmap='coolwarm', center=0)
    plt.title('Lagged VIX-Price Correlations')
    plt.show()


def perform_average_hourly_earnings_analysis(data_dict):
    spx_corr, spy_corr = average_hourly_earnings_correlation(data_dict)

    # Print out correlations
    print(f"Average Hourly Earnings - SPX Correlation: {spx_corr}")
    print(f"Average Hourly Earnings - SPY Correlation: {spy_corr}")

    # Create a heatmap to visualize the correlations
    correlations = pd.DataFrame({
        'SPX': [spx_corr],
        'SPY': [spy_corr],
    }, index=['Average Hourly Earnings'])

    sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Average Hourly Earnings - Price Correlations')
    plt.show()


# Calculate Correlations for Indicators

def indicator_correlation(df, target='close'):
    # Select columns that are indicators (you may need to adjust this based on your indicators)
    indicator_columns = ['SMA_20', 'SMA_50', 'EMA_12',
                         'EMA_26', 'RSI', 'OBV', 'VMA_20', 'VPT']

    correlations = {}
    for indicator in indicator_columns:
        if indicator in df.columns:
            correlation = df[indicator].corr(df[target])
            correlations[indicator] = correlation
    return correlations

# Perform Indicator Correlation Analysis


def perform_indicator_correlation_analysis(data_dict):
    historical_spx = data_dict['historical_spx']
    historical_spy = data_dict['historical_spy']

    spx_correlations = indicator_correlation(historical_spx)
    spy_correlations = indicator_correlation(historical_spy)

    # Print out correlations
    print("SPX Indicator Correlations:")
    for indicator, correlation in spx_correlations.items():
        print(f"{indicator}: {correlation}")

    print("\nSPY Indicator Correlations:")
    for indicator, correlation in spy_correlations.items():
        print(f"{indicator}: {correlation}")

    # Create a heatmap to visualize the correlations
    correlation_df = pd.DataFrame(
        [spx_correlations, spy_correlations], index=['SPX', 'SPY'])

    sns.heatmap(correlation_df, annot=True, cmap='coolwarm')
    plt.title('Indicator-Price Correlations')
    plt.show()


def perform_market_condition_correlation_analysis(data_dict):
    # Get the historical data
    historical_spx = data_dict['historical_spx']
    historical_spy = data_dict['historical_spy']

    # Apply market condition classification
    historical_spx['market_condition'] = classify_market_condition(historical_spx)
    historical_spy['market_condition'] = classify_market_condition(historical_spy)

    # Split data into bull and bear markets
    bull_market_spx = historical_spx[historical_spx['market_condition'] == 'bull']
    bear_market_spx = historical_spx[historical_spx['market_condition'] == 'bear']
    bull_market_spy = historical_spy[historical_spy['market_condition'] == 'bull']
    bear_market_spy = historical_spy[historical_spy['market_condition'] == 'bear']

    # Select only numeric columns for correlation
    numeric_columns_spx = bull_market_spx.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns_spy = bull_market_spy.select_dtypes(include=['float64', 'int64']).columns

    # Calculate correlations for each market condition
    bull_corr_spx = bull_market_spx[numeric_columns_spx].corr()['close']
    bear_corr_spx = bear_market_spx[numeric_columns_spx].corr()['close']
    bull_corr_spy = bull_market_spy[numeric_columns_spy].corr()['close']
    bear_corr_spy = bear_market_spy[numeric_columns_spy].corr()['close']

    # Print correlations
    print("Bull Market SPX Correlations:")
    print(bull_corr_spx)
    print("\nBear Market SPX Correlations:")
    print(bear_corr_spx)
    print("\nBull Market SPY Correlations:")
    print(bull_corr_spy)
    print("\nBear Market SPY Correlations:")
    print(bear_corr_spy)

    # Visualize correlations using heatmaps
    plt.figure(figsize=(12, 6))
    sns.heatmap(pd.DataFrame({'Bull Market SPX': bull_corr_spx, 'Bear Market SPX': bear_corr_spx}),
                annot=True, cmap='coolwarm')
    plt.title('SPX Correlations in Bull vs. Bear Markets')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.heatmap(pd.DataFrame({'Bull Market SPY': bull_corr_spy, 'Bear Market SPY': bear_corr_spy}),
                annot=True, cmap='coolwarm')
    plt.title('SPY Correlations in Bull vs. Bear Markets')
    plt.show()

def volume_price_correlation(df, price_column='close'):
    # Calculate the correlation between volume and price
    correlation = df['volume'].corr(df[price_column])
    return correlation


def perform_volume_correlation_analysis(data_dict):
    # Get the data
    historical_spx = data_dict['historical_spx']
    historical_spy = data_dict['historical_spy']
    historical_vix = data_dict['historical_vix']
    real_time_spx = data_dict['real_time_spx']
    real_time_spy = data_dict['real_time_spy']
    real_time_vix = data_dict['real_time_vix']

    # Calculate correlations
    spx_correlation = volume_price_correlation(
        historical_spx) if 'volume' in historical_spx.columns else None
    spy_correlation = volume_price_correlation(
        historical_spy) if 'volume' in historical_spy.columns else None
    # Skipping volume correlation for VIX as it doesn't have a volume column
    vix_correlation = None

    spx_real_time_correlation = volume_price_correlation(
        real_time_spx, price_column='current_price') if 'volume' in real_time_spx.columns else None
    spy_real_time_correlation = volume_price_correlation(
        real_time_spy, price_column='current_price') if 'volume' in real_time_spy.columns else None
    # Skipping volume correlation for real-time VIX as well
    vix_real_time_correlation = None

    # Print out correlations
    if spx_correlation is not None:
        print(f"SPX Volume-Price Correlation: {spx_correlation}")
    if spy_correlation is not None:
        print(f"SPY Volume-Price Correlation: {spy_correlation}")
    if vix_correlation is not None:
        print(f"VIX Volume-Price Correlation: {vix_correlation}")
    if spx_real_time_correlation is not None:
        print(
            f"Real-Time SPX Volume-Price Correlation: {spx_real_time_correlation}")
    if spy_real_time_correlation is not None:
        print(
            f"Real-Time SPY Volume-Price Correlation: {spy_real_time_correlation}")

    # Create a heatmap to visualize the correlations
    correlations = {
        'SPX': [spx_correlation, spx_real_time_correlation],
        'SPY': [spy_correlation, spy_real_time_correlation],
        'VIX': [None, None]  # No volume correlation for VIX
    }
    correlations = {k: v for k, v in correlations.items(
    ) if v[0] is not None}  # Filter out None values

    correlation_df = pd.DataFrame(
        correlations, index=['Historical', 'Real-Time'])

    sns.heatmap(correlation_df, annot=True, cmap='coolwarm')
    plt.title('Volume-Price Correlations')
    plt.show()


# VIX-SPX/SPY Correlation Analysis


def vix_price_correlation(vix_df, spx_df, spy_df, lag=0):
    # Ensure the DataFrames have either 'timestamp' or 'date' columns
    for df_name, df in zip(['VIX', 'SPX', 'SPY'], [vix_df, spx_df, spy_df]):
        logging.debug(f"Checking columns for {df_name} DataFrame: {df.columns.tolist()}")

        # Rename to 'timestamp' if 'date' is used
        if 'timestamp' in df.columns:
            logging.debug(f"{df_name} DataFrame has 'timestamp' column.")
        elif 'date' in df.columns:
            df.rename(columns={'date': 'timestamp'}, inplace=True)
            logging.debug(f"Renamed 'date' to 'timestamp' for {df_name} DataFrame.")
        else:
            logging.error(f"{df_name} DataFrame is missing both 'timestamp' and 'date' columns.")
            raise KeyError(f"{df_name} DataFrame must have either a 'timestamp' or 'date' column for correlation.")

    # Convert 'timestamp' to datetime format for consistency
    for df_name, df in zip(['VIX', 'SPX', 'SPY'], [vix_df, spx_df, spy_df]):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logging.debug(f"Converted 'timestamp' to datetime for {df_name} DataFrame.")

    # Align data based on 'timestamp'
    aligned_data = vix_df.set_index('timestamp').join(
        spx_df.set_index('timestamp'), lsuffix='_vix', rsuffix='_spx', how='inner'
    ).join(spy_df.set_index('timestamp'), lsuffix='_spx', rsuffix='_spy', how='inner')

    logging.debug(f"Columns in aligned_data: {aligned_data.columns.tolist()}")

    # Verify that expected suffixes are present in the merged data
    if not any(col.endswith('_spy') for col in aligned_data.columns):
        logging.error("SPY DataFrame merging didn't result in expected '_spy' suffixed columns.")
        raise KeyError("SPY DataFrame merging didn't result in expected '_spy' suffixed columns.")

    # Calculate correlations with lag if specified
    if lag:
        aligned_data['vix_shifted'] = aligned_data['close_vix'].shift(lag)
        vix_spx_corr = aligned_data['vix_shifted'].corr(aligned_data['close_spx'])
        vix_spy_corr = aligned_data['vix_shifted'].corr(aligned_data.filter(regex='_spy').iloc[:, 0])
    else:
        vix_spx_corr = aligned_data['close_vix'].corr(aligned_data['close_spx'])
        vix_spy_corr = aligned_data['close_vix'].corr(aligned_data.filter(regex='_spy').iloc[:, 0])

    return vix_spx_corr, vix_spy_corr


def perform_vix_price_correlation_analysis(data_dict, lag=None):
    # Get the data
    historical_vix = data_dict['historical_vix']
    historical_spx = data_dict['historical_spx']
    historical_spy = data_dict['historical_spy']

    # Ensure consistency by using 'timestamp'
    for df in [historical_vix, historical_spx, historical_spy]:
        if 'date' in df.columns:
            df.rename(columns={'date': 'timestamp'}, inplace=True)
        elif 'timestamp' not in df.columns:
            raise KeyError("Dataframes must have either a 'timestamp' or 'date' column for correlation.")

    # Now all DataFrames should have a 'timestamp' column, and no need to rename to 'time_index'
    
    # Calculate VIX-SPX and VIX-SPY correlations with lag
    vix_spx_corr, vix_spy_corr = vix_price_correlation(historical_vix, historical_spx, historical_spy, lag=lag)

    # Print out correlations
    print(f"VIX-SPX Price Correlation (Lag {lag}): {vix_spx_corr}")
    print(f"VIX-SPY Price Correlation (Lag {lag}): {vix_spy_corr}")

    # Optionally, plot the results as a heatmap
    correlation_df = pd.DataFrame({
        f'VIX-SPX (Lag {lag})': [vix_spx_corr],
        f'VIX-SPY (Lag {lag})': [vix_spy_corr]
    })

    sns.heatmap(correlation_df, annot=True, cmap='coolwarm')
    plt.title(f'VIX-Price Correlations (Lag {lag})')
    plt.show()

def rolling_correlation_analysis(vix_df, spx_df, spy_df, window=30):
    vix_spx_rolling_corr = vix_df['close'].rolling(window=window).corr(spx_df['close'])
    vix_spy_rolling_corr = vix_df['close'].rolling(window=window).corr(spy_df['close'])

    plt.figure(figsize=(14, 7))
    plt.plot(vix_spx_rolling_corr, label='VIX-SPX Rolling Correlation', color='blue')
    plt.plot(vix_spy_rolling_corr, label='VIX-SPY Rolling Correlation', color='red')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title(f'Rolling Correlation (Window = {window} days)')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.legend()
    plt.show()


# Ensure the function definition is properly aligned with other functions
def volatility_regimes_analysis(vix_df, spx_df, spy_df):
    # Define volatility regimes based on VIX values
    vix_df['volatility_regime'] = pd.cut(vix_df['close'], bins=[0, 12, 20, 30, np.inf], labels=['Low', 'Moderate', 'High', 'Extreme'])

    correlations = {
        'Volatility Regime': [],
        'VIX-SPX Correlation': [],
        'VIX-SPY Correlation': []
    }

    # Loop through each volatility regime
    for regime in ['Low', 'Moderate', 'High', 'Extreme']:
        regime_mask = vix_df['volatility_regime'] == regime

        # Align indices and use .loc to filter
        aligned_spx = spx_df.loc[regime_mask.index.intersection(spx_df.index)]
        aligned_spy = spy_df.loc[regime_mask.index.intersection(spy_df.index)]
        aligned_vix = vix_df.loc[regime_mask.index.intersection(vix_df.index)]

        # Apply mask after alignment
        regime_vix = aligned_vix[regime_mask]['close']
        regime_spx = aligned_spx[regime_mask]['close']
        regime_spy = aligned_spy[regime_mask]['close']

        vix_spx_corr = regime_vix.corr(regime_spx)
        vix_spy_corr = regime_vix.corr(regime_spy)

        correlations['Volatility Regime'].append(regime)
        correlations['VIX-SPX Correlation'].append(vix_spx_corr)
        correlations['VIX-SPY Correlation'].append(vix_spy_corr)

    corr_df = pd.DataFrame(correlations)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Volatility Regime', y='VIX-SPX Correlation', data=corr_df, color='blue', label='VIX-SPX')
    sns.barplot(x='Volatility Regime', y='VIX-SPY Correlation', data=corr_df, color='red', label='VIX-SPY', alpha=0.7)
    plt.title('Correlation by Volatility Regime')
    plt.ylabel('Correlation')
    plt.legend()
    plt.show()

def engineer_features(df, tf='daily', has_close=True):
    logging.debug("Starting feature engineering...")

    # Adjust the price column based on whether 'close' exists or not
    price_column = 'close' if has_close else 'current_price'

    # Ensure missing columns exist and are initialized to default values
    if 'volume' not in df.columns:
        df['volume'] = 0.0  # Default value for volume
    if 'conditions' not in df.columns:
        df['conditions'] = ''  # Default value for conditions (empty string)


    # Classify VIX regime if VIX data is available
    if 'VIX' in df.columns:
        df['vix_regime'] = df['VIX'].apply(classify_vix_regime)


    # Print the DataFrame before recomputing indicators for debugging
    print("DataFrame before recomputing indicators:")
    print(df.head())  # Debugging step

    # Check for NaNs in the key column before processing
    if df[price_column].isna().all():
        print(f"All values in {price_column} are NaN, skipping recompute.")
        return df  # Skip recomputation if there's no valid data

    # Recompute indicators based on the timeframe, ensuring that we have enough data
    df = recompute_indicators(df, tf, has_close=has_close)

    # Add volume-based features
    df = calculate_obv(df)
    df = calculate_vma(df)
    df = calculate_vpt(df)

    # Print the DataFrame after recomputing indicators for debugging
    print("DataFrame after recomputing indicators, before ffill:")
    print(df.head())  # Debugging step

    # Forward fill any remaining NaNs and drop rows that are still NaN
    df = df.ffill().dropna(subset=[price_column])

    # Print the DataFrame after filling and dropping NaNs for debugging
    print("DataFrame after ffill and dropna:")
    print(df.head())  # Debugging step

    # Log shape of data after feature engineering
    logging.debug(f"Feature engineering complete. Data shape: {df.shape}")

    return df


def prepare_data(df):
    # Include volume-related features along with price indicators
    columns_to_keep = [col for col in df.columns if col not in ['id']]

    X = df[columns_to_keep].fillna(method='ffill').dropna().values  # Features
    y = df['close'].values  # Target

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reshape for LSTM
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    return X, y


def train_model(X_train, X_test, y_train, y_test):
    logging.debug("Starting model training...")

    # Create hybrid model
    model = create_hybrid_model(input_shape=(
        X_train.shape[1], X_train.shape[2]))

    # Set up model checkpointing
    checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, MODEL_NAME + '.h5'),
                                 save_best_only=True, monitor='val_loss', mode='min')

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_data=(X_test, y_test),
              callbacks=[checkpoint, early_stopping])

    logging.debug("Model training complete.")
    return model


def test_engineer_features():
    data_dict = load_data()

    # Perform correlation analysis for volume
    perform_volume_correlation_analysis(data_dict)

    # Perform indicator correlation analysis
    perform_indicator_correlation_analysis(data_dict)

    # Perform VIX-SPX/SPY correlation analysis
    perform_vix_price_correlation_analysis(data_dict)

    # Perform Average Hourly Earnings analysis
    perform_average_hourly_earnings_analysis(data_dict)

    # Perform Bull vs. Bear Market correlation analysis
    perform_market_condition_correlation_analysis(data_dict)

     # Perform Lagged Correlation Analysis
    historical_vix = data_dict['historical_vix']
    historical_spx = data_dict['historical_spx']
    historical_spy = data_dict['historical_spy']
    lagged_correlation_analysis(historical_vix, historical_spx, historical_spy, max_lag=10)

    # Perform Rolling Correlation Analysis
    rolling_correlation_analysis(historical_vix, historical_spx, historical_spy, window=30)

    # Perform Volatility Regimes Analysis
    volatility_regimes_analysis(historical_vix, historical_spx, historical_spy)

    # Print summary of data for verification
    for key, df in data_dict.items():
        print(f"{key} data head:\n{df.head()}\n")


def main():
    data_dict = load_data()

    # Process historical SPX, SPY, and VIX data
    df_spx = engineer_features(data_dict['historical_spx'], vix_df=data_dict['historical_vix'])
    df_spy = engineer_features(data_dict['historical_spy'], vix_df=data_dict['historical_vix'])
    df_vix = engineer_features(data_dict['historical_vix'])

    # Process real-time SPX, SPY, and VIX data
    real_time_spx = engineer_features(
        data_dict['real_time_spx'], tf='1min', has_close=False, vix_df=data_dict['real_time_vix'])
    real_time_spy = engineer_features(
        data_dict['real_time_spy'], tf='1min', has_close=False, vix_df=data_dict['real_time_vix'])
    real_time_vix = engineer_features(
        data_dict['real_time_vix'], tf='1min', has_close=True)

    # Optionally combine historical and real-time data
    combined_spx = pd.concat([df_spx, real_time_spx], ignore_index=True)
    combined_spy = pd.concat([df_spy, real_time_spy], ignore_index=True)
    combined_vix = pd.concat([df_vix, real_time_vix], ignore_index=True)

    # Perform correlation analysis
    perform_volume_correlation_analysis(data_dict)
    perform_indicator_correlation_analysis(data_dict)
    perform_vix_price_correlation_analysis(data_dict)

    # Perform Lagged Correlation Analysis
    lagged_correlation_analysis(historical_vix, historical_spx, historical_spy, max_lag=10)

    # Perform Rolling Correlation Analysis
    rolling_correlation_analysis(historical_vix, historical_spx, historical_spy, window=30)

    # Perform Volatility Regimes Analysis
    volatility_regimes_analysis(historical_vix, historical_spx, historical_spy)

    # Prepare data for the model
    X_spx, y_spx = prepare_data(combined_spx)
    X_spy, y_spy = prepare_data(combined_spy)
    X_vix, y_vix = prepare_data(combined_vix)

    # Split into train and test sets for SPX
    X_train_spx, X_test_spx, y_train_spx, y_test_spx = train_test_split(
        X_spx, y_spx, test_size=0.2, random_state=42)

    # Split into train and test sets for SPY
    X_train_spy, X_test_spy, y_train_spy, y_test_spy = train_test_split(
        X_spy, y_spy, test_size=0.2, random_state=42)

    # Split into train and test sets for VIX
    X_train_vix, X_test_vix, y_train_vix, y_test_vix = train_test_split(
        X_vix, y_vix, test_size=0.2, random_state=42)

    # Train the models
    trained_model_spx = train_model(
        X_train_spx, X_test_spx, y_train_spx, y_test_spx)
    trained_model_spy = train_model(
        X_train_spy, X_test_spy, y_train_spy, y_test_spy)
    trained_model_vix = train_model(
        X_train_vix, X_test_vix, y_train_vix, y_test_vix)

if __name__ == "__main__":
    test_engineer_features()