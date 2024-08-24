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

# Average Hourly Earnings: Historical
def average_hourly_earnings_correlation_historical(data_dict):
    print("Starting historical average hourly earnings correlation analysis...")

    historical_average_hourly_earnings = data_dict['average_hourly_earnings_data']
    historical_spx_data = data_dict['historical_spx']
    historical_spy_data = data_dict['historical_spy']

    # Truncate to the shortest length to avoid length mismatch
    min_length = min(len(historical_average_hourly_earnings), len(historical_spx_data), len(historical_spy_data))

    # Truncate datasets
    historical_average_hourly_earnings = historical_average_hourly_earnings.iloc[:min_length]
    historical_spx_data = historical_spx_data.iloc[:min_length]
    historical_spy_data = historical_spy_data.iloc[:min_length]

    # Calculate correlations
    historical_spx_corr = historical_average_hourly_earnings['value'].corr(historical_spx_data['close'])
    historical_spy_corr = historical_average_hourly_earnings['value'].corr(historical_spy_data['close'])

    print(f"Historical SPX Correlation: {historical_spx_corr}, Historical SPY Correlation: {historical_spy_corr}")

    return historical_spx_corr, historical_spy_corr

# Average Hourly Earnings: Real Time
def average_hourly_earnings_correlation_real_time(data_dict):
    print("Starting real-time average hourly earnings correlation analysis...")

    real_time_average_hourly_earnings = data_dict['average_hourly_earnings_data']
    real_time_spx_data = data_dict['real_time_spx']
    real_time_spy_data = data_dict['real_time_spy']

    # Aggregate real-time data to daily to match real_time_average_hourly_earnings
    real_time_spx_data_daily = real_time_spx_data.groupby('date')['current_price'].mean().reset_index()
    real_time_spy_data_daily = real_time_spy_data.groupby('date')['current_price'].mean().reset_index()

    # Align the dates
    merged_real_time_spx = pd.merge(real_time_average_hourly_earnings, real_time_spx_data_daily, on='date', how='inner')
    merged_real_time_spy = pd.merge(real_time_average_hourly_earnings, real_time_spy_data_daily, on='date', how='inner')

    # Calculate correlations
    real_time_spx_corr = merged_real_time_spx['value'].corr(merged_real_time_spx['current_price'])
    real_time_spy_corr = merged_real_time_spy['value'].corr(merged_real_time_spy['current_price'])

    print(f"Real-Time SPX Correlation: {real_time_spx_corr}, Real-Time SPY Correlation: {real_time_spy_corr}")

    return real_time_spx_corr, real_time_spy_corr


def perform_average_hourly_earnings_historical_analysis(data_dict):
    # Perform historical correlation analysis
    historical_spx_corr, historical_spy_corr = average_hourly_earnings_correlation_historical(data_dict)

    # Print out historical correlations
    print(f"Historical Average Hourly Earnings - SPX Correlation: {historical_spx_corr}")
    print(f"Historical Average Hourly Earnings - SPY Correlation: {historical_spy_corr}")

    # Create a heatmap for historical data
    correlations_hist = pd.DataFrame({
        'SPX': [historical_spx_corr],
        'SPY': [historical_spy_corr],
    }, index=['Historical Average Hourly Earnings'])

    sns.heatmap(correlations_hist, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Historical Average Hourly Earnings - Price Correlations')
    plt.show()

def perform_average_hourly_earnings_real_time_analysis(data_dict):
    # Perform real-time correlation analysis
    real_time_spx_corr, real_time_spy_corr = average_hourly_earnings_correlation_real_time(data_dict)

    # Print out real-time correlations
    print(f"Real-Time Average Hourly Earnings - SPX Correlation: {real_time_spx_corr}")
    print(f"Real-Time Average Hourly Earnings - SPY Correlation: {real_time_spy_corr}")

    # Create a heatmap for real-time data
    correlations_real_time = pd.DataFrame({
        'SPX': [real_time_spx_corr],
        'SPY': [real_time_spy_corr],
    }, index=['Real-Time Average Hourly Earnings'])

    sns.heatmap(correlations_real_time, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Real-Time Average Hourly Earnings - Price Correlations')
    plt.show()

# Consumer Confidence Analysis: Historic
def create_consumer_confidence_heatmap_historic(data_dict):
    """
    This function calculates the correlations between consumer confidence data and SPX/SPY close prices,
    and then creates heatmaps to visualize these correlations.
    
    Parameters:
    - data_dict: A dictionary containing DataFrames for consumer confidence and historical SPX, SPY data.
    """
    # Load the consumer confidence data
    consumer_confidence_df = data_dict['consumer_confidence_data']
        
    # Load the historical SPX and SPY data
    spx_df = data_dict['historical_spx']
    spy_df = data_dict['historical_spy']
    
    # Merge consumer confidence data with SPX and SPY to align dates
    merged_spx = pd.merge(consumer_confidence_df, spx_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spx'))
    merged_spy = pd.merge(consumer_confidence_df, spy_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spy'))
    
    # Select only numeric columns for correlation
    numeric_cols_spx = merged_spx.select_dtypes(include=[float, int]).columns
    numeric_cols_spy = merged_spy.select_dtypes(include=[float, int]).columns
    
    # Calculate correlation with SPX and SPY close prices
    spx_correlations = merged_spx[numeric_cols_spx].corr()['close']
    spy_correlations = merged_spy[numeric_cols_spy].corr()['close']
    
    # Combine the correlations into a single DataFrame for visualization
    combined_correlations = pd.DataFrame({'SPX': spx_correlations, 'SPY': spy_correlations}).dropna()
    
    # Plot the heatmap
    plt.figure(figsize=(10, len(combined_correlations) * 0.5))
    sns.heatmap(combined_correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Consumer Confidence Correlation with SPX and SPY')
    plt.show()

# Consumer Confidence Analysis: Real Time
def create_consumer_confidence_heatmap_real_time(data_dict):
    """
    This function calculates the correlations between consumer confidence data and real-time SPX/SPY current prices,
    and then creates heatmaps to visualize these correlations.

    Parameters:
    - data_dict: A dictionary containing DataFrames for consumer confidence and real-time SPX, SPY data.
    """
    # Load the consumer confidence data
    consumer_confidence_df = data_dict['consumer_confidence_data']
    
    # Load the real-time SPX and SPY data
    spx_df = data_dict['real_time_spx']
    spy_df = data_dict['real_time_spy']
    
    # Convert the 'date' column to datetime if not already done
    consumer_confidence_df['date'] = pd.to_datetime(consumer_confidence_df['date'])
    spx_df['date'] = pd.to_datetime(spx_df['date'])
    spy_df['date'] = pd.to_datetime(spy_df['date'])
    
    # Aggregate real-time data to daily to match consumer confidence granularity
    spx_daily = spx_df.groupby('date')['current_price'].mean().reset_index()
    spy_daily = spy_df.groupby('date')['current_price'].mean().reset_index()
    
    # Merge consumer confidence data with aggregated SPX and SPY data to align dates
    merged_spx = pd.merge(consumer_confidence_df, spx_daily, on='date', how='inner', suffixes=('', '_spx'))
    merged_spy = pd.merge(consumer_confidence_df, spy_daily, on='date', how='inner', suffixes=('', '_spy'))
    
    # Debugging: Print the first few rows of the merged data
    print("Merged SPX DataFrame:")
    print(merged_spx.head())
    print("Merged SPY DataFrame:")
    print(merged_spy.head())
    
    # Select only numeric columns for correlation
    numeric_cols_spx = merged_spx.select_dtypes(include=[float, int]).columns
    numeric_cols_spy = merged_spy.select_dtypes(include=[float, int]).columns
    
    # Calculate correlation with SPX and SPY current prices
    spx_correlations = merged_spx[numeric_cols_spx].corr()['current_price']
    spy_correlations = merged_spy[numeric_cols_spy].corr()['current_price']
    
    # Combine the correlations into a single DataFrame for visualization
    combined_correlations = pd.DataFrame({'SPX': spx_correlations, 'SPY': spy_correlations}).dropna()
    
    # Debugging: Print combined correlations
    print("Combined Correlations DataFrame:")
    print(combined_correlations)
    print("Shape:", combined_correlations.shape)
    
    # Check if DataFrame is empty
    if combined_correlations.empty:
        print("No data to plot for consumer confidence correlation with real-time SPX and SPY.")
        return
    
    # Plot the heatmap with adjusted figure size to avoid aspect ratio issues
    plt.figure(figsize=(10, max(1, len(combined_correlations) * 0.5)))
    sns.heatmap(combined_correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Consumer Confidence Correlation with Real-Time SPX and SPY')
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

def create_consumer_sentiment_heatmap(data_dict):
    """
    This function calculates the correlations between consumer sentiment data and SPX/SPY close prices,
    and then creates heatmaps to visualize these correlations.
    
    Parameters:
    - data_dict: A dictionary containing DataFrames for consumer sentiment and historical SPX, SPY data.
    """
    # Load the consumer sentiment data
    consumer_sentiment_df = data_dict['consumer_sentiment_data']
    
    # Load the historical SPX and SPY data
    spx_df = data_dict['historical_spx']
    spy_df = data_dict['historical_spy']
    
    # Merge consumer sentiment data with SPX and SPY to align dates
    merged_spx = pd.merge(consumer_sentiment_df, spx_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spx'))
    merged_spy = pd.merge(consumer_sentiment_df, spy_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spy'))
    
    # Select only numeric columns for correlation
    numeric_cols_spx = merged_spx.select_dtypes(include=[float, int]).columns
    numeric_cols_spy = merged_spy.select_dtypes(include=[float, int]).columns
    
    # Calculate correlation with SPX and SPY close prices
    spx_correlations = merged_spx[numeric_cols_spx].corr()['close']
    spy_correlations = merged_spy[numeric_cols_spy].corr()['close']
    
    # Combine the correlations into a single DataFrame for visualization
    combined_correlations = pd.DataFrame({'SPX': spx_correlations, 'SPY': spy_correlations}).dropna()
    
    # Plot the heatmap
    plt.figure(figsize=(10, len(combined_correlations) * 0.5))
    sns.heatmap(combined_correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Consumer Sentiment Correlation with SPX and SPY')
    plt.show()

# Core Inflation Analysis
def create_core_inflation_heatmap_historical(data_dict):
    """
    This function calculates the correlations between core inflation data and historical SPX/SPY close prices,
    and then creates heatmaps to visualize these correlations.
    
    Parameters:
    - data_dict: A dictionary containing DataFrames for core inflation and historical SPX, SPY data.
    """
    # Load the core inflation data
    core_inflation_df = data_dict['core_inflation_data']
        
    # Load the historical SPX and SPY data
    historical_spx_df = data_dict['historical_spx']
    historical_spy_df = data_dict['historical_spy']
    
    # Merge core inflation data with historical SPX and SPY to align dates
    merged_spx = pd.merge(core_inflation_df, historical_spx_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spx'))
    merged_spy = pd.merge(core_inflation_df, historical_spy_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spy'))
    
    # Select only numeric columns for correlation
    numeric_cols_spx = merged_spx.select_dtypes(include=[float, int]).columns
    numeric_cols_spy = merged_spy.select_dtypes(include=[float, int]).columns
    
    # Calculate correlation with SPX and SPY close prices
    spx_correlations = merged_spx[numeric_cols_spx].corr()['close']
    spy_correlations = merged_spy[numeric_cols_spy].corr()['close']
    
    # Combine the correlations into a single DataFrame for visualization
    combined_correlations = pd.DataFrame({'SPX': spx_correlations, 'SPY': spy_correlations}).dropna()
    
    # Plot the heatmap
    plt.figure(figsize=(10, len(combined_correlations) * 0.5))
    sns.heatmap(combined_correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Core Inflation Correlation with Historical SPX and SPY')
    plt.show()

# CPI Data
def create_cpi_correlation_heatmap_historical(data_dict):
    """
    This function calculates the correlations between CPI data and historical SPX/SPY close prices,
    and then creates heatmaps to visualize these correlations.
    
    Parameters:
    - data_dict: A dictionary containing DataFrames for CPI and historical SPX, SPY data.
    """
    # Load the CPI data
    cpi_df = data_dict['cpi_data']
        
    # Load the historical SPX and SPY data
    historical_spx_df = data_dict['historical_spx']
    historical_spy_df = data_dict['historical_spy']
    
    # Merge CPI data with historical SPX and SPY to align dates
    merged_spx = pd.merge(cpi_df, historical_spx_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spx'))
    merged_spy = pd.merge(cpi_df, historical_spy_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spy'))
    
    # Select only numeric columns for correlation
    numeric_cols_spx = merged_spx.select_dtypes(include=[float, int]).columns
    numeric_cols_spy = merged_spy.select_dtypes(include=[float, int]).columns
    
    # Calculate correlation with SPX and SPY close prices
    spx_correlations = merged_spx[numeric_cols_spx].corr()['close']
    spy_correlations = merged_spy[numeric_cols_spy].corr()['close']
    
    # Combine the correlations into a single DataFrame for visualization
    combined_correlations = pd.DataFrame({'SPX': spx_correlations, 'SPY': spy_correlations}).dropna()
    
    # Plot the heatmap
    plt.figure(figsize=(10, len(combined_correlations) * 0.5))
    sns.heatmap(combined_correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('CPI Correlation with Historical SPX and SPY')
    plt.show()

# GDP Data
def create_gdp_correlation_heatmap_historical(data_dict):
    """
    This function calculates the correlations between GDP data and historical SPX/SPY close prices,
    and then creates heatmaps to visualize these correlations.
    
    Parameters:
    - data_dict: A dictionary containing DataFrames for GDP and historical SPX, SPY data.
    """
    # Load the GDP data
    gdp_df = data_dict['gdp_data']
        
    # Load the historical SPX and SPY data
    historical_spx_df = data_dict['historical_spx']
    historical_spy_df = data_dict['historical_spy']
    
    # Merge GDP data with historical SPX and SPY to align dates
    merged_spx = pd.merge(gdp_df, historical_spx_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spx'))
    merged_spy = pd.merge(gdp_df, historical_spy_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spy'))
    
    # Select only numeric columns for correlation
    numeric_cols_spx = merged_spx.select_dtypes(include=[float, int]).columns
    numeric_cols_spy = merged_spy.select_dtypes(include=[float, int]).columns
    
    # Calculate correlation with SPX and SPY close prices
    spx_correlations = merged_spx[numeric_cols_spx].corr()['close']
    spy_correlations = merged_spy[numeric_cols_spy].corr()['close']
    
    # Combine the correlations into a single DataFrame for visualization
    combined_correlations = pd.DataFrame({'SPX': spx_correlations, 'SPY': spy_correlations}).dropna()
    
    # Plot the heatmap
    plt.figure(figsize=(10, len(combined_correlations) * 0.5))
    sns.heatmap(combined_correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('GDP Correlation with Historical SPX and SPY')
    plt.show()

# Industrial Production Data
def create_industrial_production_correlation_heatmap_historical(data_dict):
    """
    This function calculates the correlations between Industrial Production data and historical SPX/SPY close prices,
    and then creates heatmaps to visualize these correlations.
    
    Parameters:
    - data_dict: A dictionary containing DataFrames for Industrial Production and historical SPX, SPY data.
    """
    # Load the Industrial Production data
    industrial_production_df = data_dict['industrial_production_data']
        
    # Load the historical SPX and SPY data
    historical_spx_df = data_dict['historical_spx']
    historical_spy_df = data_dict['historical_spy']
    
    # Merge Industrial Production data with historical SPX and SPY to align dates
    merged_spx = pd.merge(industrial_production_df, historical_spx_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spx'))
    merged_spy = pd.merge(industrial_production_df, historical_spy_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spy'))
    
    # Select only numeric columns for correlation
    numeric_cols_spx = merged_spx.select_dtypes(include=[float, int]).columns
    numeric_cols_spy = merged_spy.select_dtypes(include=[float, int]).columns
    
    # Calculate correlation with SPX and SPY close prices
    spx_correlations = merged_spx[numeric_cols_spx].corr()['close']
    spy_correlations = merged_spy[numeric_cols_spy].corr()['close']
    
    # Combine the correlations into a single DataFrame for visualization
    combined_correlations = pd.DataFrame({'SPX': spx_correlations, 'SPY': spy_correlations}).dropna()
    
    # Plot the heatmap
    plt.figure(figsize=(10, len(combined_correlations) * 0.5))
    sns.heatmap(combined_correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Industrial Production Correlation with Historical SPX and SPY')
    plt.show()

# Interest Rate Data
def create_interest_rate_correlation_heatmap_historical(data_dict):
    """
    This function calculates the correlations between interest rate data and historical SPX/SPY close prices,
    and then creates heatmaps to visualize these correlations.
    
    Parameters:
    - data_dict: A dictionary containing DataFrames for interest rate and historical SPX, SPY data.
    """
    # Load the interest rate data
    interest_rate_df = data_dict['interest_rate_data']
        
    # Load the historical SPX and SPY data
    historical_spx_df = data_dict['historical_spx']
    historical_spy_df = data_dict['historical_spy']
    
    # Merge interest rate data with historical SPX and SPY to align dates
    merged_spx = pd.merge(interest_rate_df, historical_spx_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spx'))
    merged_spy = pd.merge(interest_rate_df, historical_spy_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spy'))
    
    # Select only numeric columns for correlation
    numeric_cols_spx = merged_spx.select_dtypes(include=[float, int]).columns
    numeric_cols_spy = merged_spy.select_dtypes(include=[float, int]).columns
    
    # Calculate correlation with SPX and SPY close prices
    spx_correlations = merged_spx[numeric_cols_spx].corr()['close']
    spy_correlations = merged_spy[numeric_cols_spy].corr()['close']
    
    # Combine the correlations into a single DataFrame for visualization
    combined_correlations = pd.DataFrame({'SPX': spx_correlations, 'SPY': spy_correlations}).dropna()
    
    # Plot the heatmap
    plt.figure(figsize=(10, len(combined_correlations) * 0.5))
    sns.heatmap(combined_correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Interest Rate Correlation with Historical SPX and SPY')
    plt.show()

# Labor Force Participation
def create_labor_force_participation_correlation_heatmap_historical(data_dict):
    """
    This function calculates the correlations between labor force participation data and historical SPX/SPY close prices,
    and then creates heatmaps to visualize these correlations.
    
    Parameters:
    - data_dict: A dictionary containing DataFrames for labor force participation and historical SPX, SPY data.
    """
    # Load the labor force participation data
    labor_force_participation_df = data_dict['labor_force_participation_rate_data']
        
    # Load the historical SPX and SPY data
    historical_spx_df = data_dict['historical_spx']
    historical_spy_df = data_dict['historical_spy']
    
    # Merge labor force participation data with historical SPX and SPY to align dates
    merged_spx = pd.merge(labor_force_participation_df, historical_spx_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spx'))
    merged_spy = pd.merge(labor_force_participation_df, historical_spy_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spy'))
    
    # Select only numeric columns for correlation
    numeric_cols_spx = merged_spx.select_dtypes(include=[float, int]).columns
    numeric_cols_spy = merged_spy.select_dtypes(include=[float, int]).columns
    
    # Calculate correlation with SPX and SPY close prices
    spx_correlations = merged_spx[numeric_cols_spx].corr()['close']
    spy_correlations = merged_spy[numeric_cols_spy].corr()['close']
    
    # Combine the correlations into a single DataFrame for visualization
    combined_correlations = pd.DataFrame({'SPX': spx_correlations, 'SPY': spy_correlations}).dropna()
    
    # Plot the heatmap
    plt.figure(figsize=(10, len(combined_correlations) * 0.5))
    sns.heatmap(combined_correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Labor Force Participation Correlation with Historical SPX and SPY')
    plt.show()

# Non Farm Payroll Employment Rate
def create_nonfarm_payroll_correlation_heatmap_historical(data_dict):
    """
    This function calculates the correlations between non-farm payroll employment data and SPX/SPY close prices,
    and then creates heatmaps to visualize these correlations.

    Parameters:
    - data_dict: A dictionary containing DataFrames for non-farm payroll employment and historical SPX, SPY data.
    """
    # Load the non-farm payroll employment data
    nonfarm_payroll_df = data_dict['nonfarm_payroll_employment_data']

    # Load the historical SPX and SPY data
    spx_df = data_dict['historical_spx']
    spy_df = data_dict['historical_spy']

    # Merge non-farm payroll employment data with SPX and SPY to align dates
    merged_spx = pd.merge(nonfarm_payroll_df, spx_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spx'))
    merged_spy = pd.merge(nonfarm_payroll_df, spy_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spy'))

    # Select only numeric columns for correlation
    numeric_cols_spx = merged_spx.select_dtypes(include=[float, int]).columns
    numeric_cols_spy = merged_spy.select_dtypes(include=[float, int]).columns

    # Calculate correlation with SPX and SPY close prices
    spx_correlations = merged_spx[numeric_cols_spx].corr()['close']
    spy_correlations = merged_spy[numeric_cols_spy].corr()['close']

    # Combine the correlations into a single DataFrame for visualization
    combined_correlations = pd.DataFrame({'SPX': spx_correlations, 'SPY': spy_correlations}).dropna()

    # Plot the heatmap
    if not combined_correlations.empty:
        plt.figure(figsize=(10, len(combined_correlations) * 0.5))
        sns.heatmap(combined_correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Non-Farm Payroll Employment Correlation with SPX and SPY')
        plt.show()
    else:
        print("No data to plot for non-farm payroll employment correlation with SPX and SPY.")

# Personal Consumption Expenditures Data
def create_personal_consumption_expenditures_correlation_heatmap_historical(data_dict):
    """
    This function calculates the correlations between personal consumption expenditures data and SPX/SPY close prices,
    and then creates heatmaps to visualize these correlations.

    Parameters:
    - data_dict: A dictionary containing DataFrames for personal consumption expenditures and historical SPX, SPY data.
    """
    # Load the personal consumption expenditures data
    personal_consumption_expenditures_df = data_dict['personal_consumption_expenditures_data']

    # Load the historical SPX and SPY data
    historical_spx_df = data_dict['historical_spx']
    historical_spy_df = data_dict['historical_spy']

    # Merge personal consumption expenditures data with SPX and SPY to align dates
    merged_spx = pd.merge(personal_consumption_expenditures_df, historical_spx_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spx'))
    merged_spy = pd.merge(personal_consumption_expenditures_df, historical_spy_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spy'))

    # Select only numeric columns for correlation
    numeric_cols_spx = merged_spx.select_dtypes(include=[float, int]).columns
    numeric_cols_spy = merged_spy.select_dtypes(include=[float, int]).columns

    # Calculate correlation with SPX and SPY close prices
    spx_correlations = merged_spx[numeric_cols_spx].corr()['close']
    spy_correlations = merged_spy[numeric_cols_spy].corr()['close']

    # Combine the correlations into a single DataFrame for visualization
    combined_correlations = pd.DataFrame({'SPX': spx_correlations, 'SPY': spy_correlations}).dropna()

    # Plot the heatmap
    if not combined_correlations.empty:
        plt.figure(figsize=(10, len(combined_correlations) * 0.5))
        sns.heatmap(combined_correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Personal Consumption Expenditures Correlation with SPX and SPY')
        plt.show()
    else:
        print("No data to plot for personal consumption expenditures correlation with SPX and SPY.")

# PPI Data
def create_ppi_correlation_heatmap_historical(data_dict):
    """
    This function calculates the correlations between PPI data and SPX/SPY close prices,
    and then creates heatmaps to visualize these correlations.

    Parameters:
    - data_dict: A dictionary containing DataFrames for PPI and historical SPX, SPY data.
    """
    # Load the PPI data
    ppi_df = data_dict['ppi_data']

    # Load the historical SPX and SPY data
    historical_spx_df = data_dict['historical_spx']
    historical_spy_df = data_dict['historical_spy']

    # Merge PPI data with SPX and SPY to align dates
    merged_spx = pd.merge(ppi_df, historical_spx_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spx'))
    merged_spy = pd.merge(ppi_df, historical_spy_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spy'))

    # Select only numeric columns for correlation
    numeric_cols_spx = merged_spx.select_dtypes(include=[float, int]).columns
    numeric_cols_spy = merged_spy.select_dtypes(include=[float, int]).columns

    # Calculate correlation with SPX and SPY close prices
    spx_correlations = merged_spx[numeric_cols_spx].corr()['close']
    spy_correlations = merged_spy[numeric_cols_spy].corr()['close']

    # Combine the correlations into a single DataFrame for visualization
    combined_correlations = pd.DataFrame({'SPX': spx_correlations, 'SPY': spy_correlations}).dropna()

    # Plot the heatmap
    if not combined_correlations.empty:
        plt.figure(figsize=(10, len(combined_correlations) * 0.5))
        sns.heatmap(combined_correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('PPI Correlation with SPX and SPY')
        plt.show()
    else:
        print("No data to plot for PPI correlation with SPX and SPY.")

# Unemployment Rate Data
def create_unemployment_rate_correlation_heatmap_historical(data_dict):
    """
    This function calculates the correlations between Unemployment Rate data and SPX/SPY close prices,
    and then creates heatmaps to visualize these correlations.

    Parameters:
    - data_dict: A dictionary containing DataFrames for Unemployment Rate and historical SPX, SPY data.
    """
    # Load the Unemployment Rate data
    unemployment_rate_df = data_dict['unemployment_rate_data']

    # Load the historical SPX and SPY data
    historical_spx_df = data_dict['historical_spx']
    historical_spy_df = data_dict['historical_spy']

    # Merge Unemployment Rate data with SPX and SPY to align dates
    merged_spx = pd.merge(unemployment_rate_df, historical_spx_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spx'))
    merged_spy = pd.merge(unemployment_rate_df, historical_spy_df[['date', 'close']], on='date', how='inner', suffixes=('', '_spy'))

    # Select only numeric columns for correlation
    numeric_cols_spx = merged_spx.select_dtypes(include=[float, int]).columns
    numeric_cols_spy = merged_spy.select_dtypes(include=[float, int]).columns

    # Calculate correlation with SPX and SPY close prices
    spx_correlations = merged_spx[numeric_cols_spx].corr()['close']
    spy_correlations = merged_spy[numeric_cols_spy].corr()['close']

    # Combine the correlations into a single DataFrame for visualization
    combined_correlations = pd.DataFrame({'SPX': spx_correlations, 'SPY': spy_correlations}).dropna()

    # Plot the heatmap
    if not combined_correlations.empty:
        plt.figure(figsize=(10, len(combined_correlations) * 0.5))
        sns.heatmap(combined_correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Unemployment Rate Correlation with SPX and SPY')
        plt.show()
    else:
        print("No data to plot for Unemployment Rate correlation with SPX and SPY.")


# Lagged Correlation
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

# Real Time Indicator Correlation
def real_time_indicator_correlation(data_dict):
    """
    This function calculates the correlations between real-time SPX and SPY prices and various indicators.
    
    Parameters:
    - data_dict: A dictionary containing DataFrames for real-time SPX, SPY, and other indicators.
    """
    real_time_spx = data_dict['real_time_spx']
    real_time_spy = data_dict['real_time_spy']
    
    # List of indicators to analyze
    indicators = ['Lag_1', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'Bollinger_Upper', 'Bollinger_Lower', 'MACD', 'MACD_Signal', 'RSI', 'ATR']
    
    # Calculate correlations for SPX
    spx_correlations = real_time_spx[indicators].corrwith(real_time_spx['current_price'])
    
    # Calculate correlations for SPY
    spy_correlations = real_time_spy[indicators].corrwith(real_time_spy['current_price'])
    
    # Combine the correlations into a single DataFrame for visualization
    combined_correlations = pd.DataFrame({'SPX': spx_correlations, 'SPY': spy_correlations})
    
    # Plot the heatmap
    plt.figure(figsize=(10, len(combined_correlations) * 0.5))
    sns.heatmap(combined_correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Real-Time Indicator Correlation with SPX and SPY')
    plt.show()

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


def real_time_bull_bear_market_analysis(data_dict):
    """
    This function performs bull and bear market analysis using real-time SPX data
    and moving average crossover method.
    
    Parameters:
    - data_dict: A dictionary containing DataFrames for real-time SPX data.
    """
    real_time_spx = data_dict['real_time_spx']
    
    # Define short-term and long-term moving averages
    short_window = 50
    long_window = 200
    
    # Calculate moving averages
    real_time_spx['SMA_short'] = real_time_spx['current_price'].rolling(window=short_window).mean()
    real_time_spx['SMA_long'] = real_time_spx['current_price'].rolling(window=long_window).mean()
    
    # Determine bull and bear markets based on moving average crossover
    real_time_spx['market_condition'] = np.where(
        real_time_spx['SMA_short'] > real_time_spx['SMA_long'], 'Bull', 'Bear'
    )
    
    # Filter out periods where moving averages are not defined
    real_time_spx = real_time_spx.dropna(subset=['SMA_short', 'SMA_long'])
    
    # Calculate average SPX price in bull and bear markets
    bull_market_avg = real_time_spx[real_time_spx['market_condition'] == 'Bull']['current_price'].mean()
    bear_market_avg = real_time_spx[real_time_spx['market_condition'] == 'Bear']['current_price'].mean()
    
    print(f"Average SPX Price in Bull Market: {bull_market_avg}")
    print(f"Average SPX Price in Bear Market: {bear_market_avg}")
    
    # Optionally visualize the data
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='market_condition', y='current_price', data=real_time_spx)
    plt.title('SPX Price Distribution in Bull and Bear Markets')
    plt.xlabel('Market Condition')
    plt.ylabel('SPX Current Price')
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

# VIX Price Correlation
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

# Rolling Correlation Analysis
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


# Function to compute correlations between real-time VIX and SPX/SPY
def spx_vix_correlation_real_time(spx_df, vix_df, lag=0):
    print("Starting SPX-VIX real-time correlation analysis...")

    # Rename columns if necessary
    if 'timestamp' not in spx_df.columns:
        if 'date' in spx_df.columns:
            spx_df.rename(columns={'date': 'timestamp'}, inplace=True)
            print("Renamed 'date' to 'timestamp' in SPX DataFrame.")
        else:
            raise KeyError("'timestamp' or 'date' column not found in SPX DataFrame")

    if 'timestamp' not in vix_df.columns:
        if 'date' in vix_df.columns:
            vix_df.rename(columns={'date': 'timestamp'}, inplace=True)
            print("Renamed 'date' to 'timestamp' in VIX DataFrame.")
        else:
            raise KeyError("'timestamp' or 'date' column not found in VIX DataFrame")

    # Convert to datetime
    spx_df['timestamp'] = pd.to_datetime(spx_df['timestamp'], errors='coerce')
    vix_df['timestamp'] = pd.to_datetime(vix_df['timestamp'], errors='coerce')

    print(f"Initial SPX timestamps:\n{spx_df[['timestamp']].head()}")
    print(f"Initial VIX timestamps:\n{vix_df[['timestamp']].head()}")

    # Aggregate SPX and VIX data to daily levels
    spx_df['date'] = spx_df['timestamp'].dt.date
    vix_df['date'] = vix_df['timestamp'].dt.date

    spx_daily = spx_df.groupby('date').agg({'current_price': 'mean'}).reset_index()
    vix_daily = vix_df.groupby('date').agg({'close': 'mean'}).reset_index()

    print(f"SPX DataFrame after daily aggregation:\n{spx_daily.head()}")
    print(f"VIX DataFrame after daily aggregation:\n{vix_daily.head()}")

    # Align data based on date
    merged_data = pd.merge(spx_daily, vix_daily, on='date', how='inner', suffixes=('_spx', '_vix'))

    if lag:
        merged_data['close_vix_shifted'] = merged_data['close'].shift(lag)
        correlation = merged_data['close_vix_shifted'].corr(merged_data['current_price'])
    else:
        correlation = merged_data['close'].corr(merged_data['current_price'])

    print(f"SPX-VIX Real-Time Correlation: {correlation}")

    return correlation

def spy_vix_correlation_real_time(spy_df, vix_df, lag=0):
    print("Starting SPY-VIX real-time correlation analysis...")

    # Rename columns if necessary
    if 'timestamp' not in spy_df.columns:
        if 'date' in spy_df.columns:
            spy_df.rename(columns={'date': 'timestamp'}, inplace=True)
            print("Renamed 'date' to 'timestamp' in SPY DataFrame.")
        else:
            raise KeyError("'timestamp' or 'date' column not found in SPY DataFrame")

    if 'timestamp' not in vix_df.columns:
        if 'date' in vix_df.columns:
            vix_df.rename(columns={'date': 'timestamp'}, inplace=True)
            print("Renamed 'date' to 'timestamp' in VIX DataFrame.")
        else:
            raise KeyError("'timestamp' or 'date' column not found in VIX DataFrame")

    # Convert to datetime
    spy_df['timestamp'] = pd.to_datetime(spy_df['timestamp'], errors='coerce')
    vix_df['timestamp'] = pd.to_datetime(vix_df['timestamp'], errors='coerce')

    print(f"Initial SPY timestamps:\n{spy_df[['timestamp']].head()}")
    print(f"Initial VIX timestamps:\n{vix_df[['timestamp']].head()}")

    # Aggregate SPY and VIX data to daily levels
    spy_df['date'] = spy_df['timestamp'].dt.date
    vix_df['date'] = vix_df['timestamp'].dt.date

    spy_daily = spy_df.groupby('date').agg({'current_price': 'mean'}).reset_index()
    vix_daily = vix_df.groupby('date').agg({'close': 'mean'}).reset_index()

    print(f"SPY DataFrame after daily aggregation:\n{spy_daily.head()}")
    print(f"VIX DataFrame after daily aggregation:\n{vix_daily.head()}")

    # Align data based on date
    merged_data = pd.merge(spy_daily, vix_daily, on='date', how='inner', suffixes=('_spy', '_vix'))

    if lag:
        merged_data['close_vix_shifted'] = merged_data['close'].shift(lag)
        correlation = merged_data['close_vix_shifted'].corr(merged_data['current_price'])
    else:
        correlation = merged_data['close'].corr(merged_data['current_price'])

    print(f"SPY-VIX Real-Time Correlation: {correlation}")

    return correlation



# Engineering Features
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

    # FRED API
    # Perform Average Hourly Earnings Analysis: Historical & Real-Time
    perform_average_hourly_earnings_historical_analysis(data_dict)
    perform_average_hourly_earnings_real_time_analysis(data_dict)

    # Perform Consumer Confidence Analysis
    create_consumer_confidence_heatmap_historic(data_dict)
    create_consumer_confidence_heatmap_real_time(data_dict)

    # Perform Consumer Sentiment Analysis
    create_consumer_sentiment_heatmap(data_dict)

    # Perform Core Inflation Analysis
    create_core_inflation_heatmap_historical(data_dict)

    # Perform CPI Analysis
    create_cpi_correlation_heatmap_historical(data_dict)

    # Perform GDP Data Analysis
    create_gdp_correlation_heatmap_historical(data_dict)

    # Perform Industrial Production Analysis
    create_industrial_production_correlation_heatmap_historical(data_dict)

    # Perform Interest Rate Data
    create_interest_rate_correlation_heatmap_historical(data_dict)

    # Perform Labor Force Partipcipation Analysis
    create_labor_force_participation_correlation_heatmap_historical(data_dict)

    # Perform Non Farm Payroll EMployment Rate Analysis
    create_nonfarm_payroll_correlation_heatmap_historical(data_dict)

    # Perform Personal Consumption Expenditures Analysis
    create_personal_consumption_expenditures_correlation_heatmap_historical(data_dict)

    # Perform PPI Data Analysis
    create_ppi_correlation_heatmap_historical(data_dict)

    # Perofrm Unemployment Rate Analysis
    create_unemployment_rate_correlation_heatmap_historical(data_dict)

    # Indicators
    # Perform indicator correlation analysis
    perform_indicator_correlation_analysis(data_dict)

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

    # Perform correlation analysis for volume
    perform_volume_correlation_analysis(data_dict)

    # Perform SPX-VIX real-time correlation analysis
    spx_vix_corr = spx_vix_correlation_real_time(data_dict['real_time_spx'], data_dict['real_time_vix'], lag=0)

    # Perform SPY-VIX real-time correlation analysis
    spy_vix_corr = spy_vix_correlation_real_time(data_dict['real_time_spy'], data_dict['real_time_vix'], lag=0)

    # Perform correlation with indicators for real-time data
    real_time_indicator_correlation(data_dict)

    # Perform bull and bear market analysis using real-time data
    real_time_bull_bear_market_analysis(data_dict)

    print(f"Final SPX-VIX Correlation: {spx_vix_corr}")
    print(f"Final SPY-VIX Correlation: {spy_vix_corr}")

    # Optionally visualize the correlations
    correlations = pd.DataFrame({
        'SPX-VIX': [spx_vix_corr],
        'SPY-VIX': [spy_vix_corr],
    })

    sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Real-Time SPX-VIX and SPY-VIX Correlations')
    plt.show()

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