import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths from environment variables
DATA_DIR = os.path.join(BASE_DIR, 'data/lumen_2/processed') 
FEATURED_DIR = os.path.join(BASE_DIR, 'data/lumen_2/featured') 
MODEL_DIR = os.getenv('MODEL_DIR', 'models/lumen_2')
MODEL_NAME = 'lumen_2'

# Function to ensure the directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

# Filter SPX data
def filter_spx_data(df):
    if 'symbol' in df.columns:
        return df[df['symbol'] == 'SPX']
    else:
        print("No 'symbol' column found, returning the original DataFrame.")
        return df

# Filter SPY data
def filter_spy_data(df):
    if 'symbol' in df.columns:
        return df[df['symbol'] == 'SPY']
    else:
        print("No 'symbol' column found, returning the original DataFrame.")
        return df

# Filter VIX data
    if 'symbol' in df.columns:
        return df[df['symbol'] == 'VIX']
    else:
        print("No 'symbol' column found, returning the original DataFrame.")
        return df

def load_data():
    consumer_confidence_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_consumer_confidence_data.csv'))
    consumer_sentiment_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_consumer_sentiment_data.csv'))
    core_inflation_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_core_inflation_data.csv'))
    cpi_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_cpi_data.csv'))
    gdp_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_gdp_data.csv'))
    historical_spx = pd.read_csv(os.path.join(DATA_DIR, 'processed_historical_spx.csv'))
    historical_spy = pd.read_csv(os.path.join(DATA_DIR, 'processed_historical_spy.csv'))
    historical_vix = pd.read_csv(os.path.join(DATA_DIR, 'processed_historical_vix.csv'))
    industrial_production_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_industrial_production_data.csv'))
    interest_rate_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_interest_rate_data.csv'))
    labor_force_participation_rate_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_labor_force_participation_rate_data.csv'))
    nonfarm_payroll_employment_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_nonfarm_payroll_employment_data.csv'))
    personal_consumption_expenditures_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_personal_consumption_expenditures.csv'))
    ppi_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_ppi_data.csv'))
    real_time_spx = pd.read_csv(os.path.join(DATA_DIR, 'processed_real_time_spx.csv'))
    real_time_spy = pd.read_csv(os.path.join(DATA_DIR, 'processed_real_time_spy.csv'))
    real_time_vix = pd.read_csv(os.path.join(DATA_DIR, 'processed_real_time_vix.csv'))
    unemployment_rate_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_unemployment_rate_data.csv'))

    return {
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
# Feature Engineering: Consumer Confidence
def feature_consumer_confidence_cumulative_sum(df):
    if 'value' in df.columns:
        df['Consumer_Confidence_Cumulative_Sum'] = df['value'].cumsum()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_consumer_confidence_days_since_peak(df):
    if 'value' in df.columns:
        df['Consumer_Confidence_Peak'] = df['value'].expanding().max()
        peak_dates = df[df['value'] == df['Consumer_Confidence_Peak']].groupby('Consumer_Confidence_Peak').apply(lambda x: x.index[0]).reset_index(drop=True)
        df['Consumer_Confidence_Days_Since_Peak'] = df.apply(lambda x: (x.name - peak_dates[peak_dates == x['Consumer_Confidence_Peak']].values[0]).days if any(peak_dates == x['Consumer_Confidence_Peak']) else np.nan, axis=1)
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_consumer_confidence_days_since_trough(df):
    if 'value' in df.columns:
        df['Consumer_Confidence_Trough'] = df['value'].expanding().min()
        trough_dates = df[df['value'] == df['Consumer_Confidence_Trough']].groupby('Consumer_Confidence_Trough').apply(lambda x: x.index[0]).reset_index(drop=True)
        df['Consumer_Confidence_Days_Since_Trough'] = df.apply(lambda x: (x.name - trough_dates[trough_dates == x['Consumer_Confidence_Trough']].values[0]).days if any(trough_dates == x['Consumer_Confidence_Trough']) else np.nan, axis=1)
    else:
        print("Column 'value' not found in DataFrame.")
    return df


# Feature Engineering: Consumer Sentiment
def feature_consumer_sentiment_rolling_6m_average(df):
    if 'value' in df.columns:
        df['Consumer_Sentiment_Rolling_6M_Avg'] = df['value'].rolling(window=6).mean()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_consumer_sentiment_rolling_12m_average(df):
    if 'value' in df.columns:
        df['Consumer_Sentiment_Rolling_12M_Avg'] = df['value'].rolling(window=12).mean()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

# Feature Engineering: Core Inflation
def feature_core_inflation_value(df):
    if 'value' in df.columns:
        df['Core_Inflation_Value'] = df['value']
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_core_inflation_ema_12(df):
    if 'value' in df.columns:
        df['Core_Inflation_EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_core_inflation_ema_26(df):
    if 'value' in df.columns:
        df['Core_Inflation_EMA_26'] = df['value'].ewm(span=26, adjust=False).mean()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_core_inflation_cumulative_sum(df):
    if 'value' in df.columns:
        df['Core_Inflation_Cumulative_Sum'] = df['value'].cumsum()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_core_inflation_z_score(df):
    if 'value' in df.columns:
        df['Core_Inflation_Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_core_inflation_trend(df):
    if 'value' in df.columns:
        df['Core_Inflation_Trend'] = df['value'].rolling(window=12).mean()
    else:
        print("Column 'value' not found in DataFrame.")
    return df


# Feature Engineering: CPI
def feature_cpi_value(df):
    if 'value' in df.columns:
        df['CPI_Value'] = df['value']
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_cpi_trend(df):
    if 'value' in df.columns:
        df['CPI_Trend'] = df['value'].rolling(window=12).mean()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_cpi_cumulative_sum(df):
    if 'value' in df.columns:
        df['CPI_Cumulative_Sum'] = df['value'].cumsum()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_cpi_days_since_peak(df):
    if 'value' in df.columns:
        df['CPI_Peak'] = df['value'].expanding().max()
        peak_dates = df[df['value'] == df['CPI_Peak']].groupby('CPI_Peak').apply(lambda x: x.index[0]).reset_index(drop=True)
        df['CPI_Days_Since_Peak'] = df.apply(lambda x: (x.name - peak_dates[peak_dates == x['CPI_Peak']].values[0]).days if any(peak_dates == x['CPI_Peak']) else np.nan, axis=1)
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_cpi_days_since_trough(df):
    if 'value' in df.columns:
        df['CPI_Trough'] = df['value'].expanding().min()
        trough_dates = df[df['value'] == df['CPI_Trough']].groupby('CPI_Trough').apply(lambda x: x.index[0]).reset_index(drop=True)
        df['CPI_Days_Since_Trough'] = df.apply(lambda x: (x.name - trough_dates[trough_dates == x['CPI_Trough']].values[0]).days if any(trough_dates == x['CPI_Trough']) else np.nan, axis=1)
    else:
        print("Column 'value' not found in DataFrame.")
    return df


# Feature Engineering: GDP
def feature_gdp_value(df):
    if 'value' in df.columns:
        df['GDP_Value'] = df['value']
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_gdp_lag(df, lag_periods):
    for lag in lag_periods:
        if 'value' in df.columns:
            df[f'GDP_Lag_{lag}'] = df['value'].shift(lag)
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_gdp_rolling_mean(df, windows):
    for window in windows:
        if 'value' in df.columns:
            df[f'GDP_Rolling_Mean_{window}Q'] = df['value'].rolling(window).mean()
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_gdp_rolling_std(df, windows):
    for window in windows:
        if 'value' in df.columns:
            df[f'GDP_Rolling_Std_{window}Q'] = df['value'].rolling(window).std()
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_gdp_cumulative_sum(df):
    if 'value' in df.columns:
        df['GDP_Cumulative_Sum'] = df['value'].cumsum()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_gdp_cumulative_product(df):
    if 'value' in df.columns:
        df['GDP_Cumulative_Product'] = df['value'].cumprod()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_gdp_trend(df):
    if 'value' in df.columns:
        df['GDP_Trend'] = df['value'].rolling(window=12).mean()  # Assuming a yearly trend for simplicity
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_gdp_ema(df, spans):
    for span in spans:
        if 'value' in df.columns:
            df[f'GDP_EMA_{span}'] = df['value'].ewm(span=span, adjust=False).mean()
        else:
            print("Column 'value' not found in DataFrame.")
    return df

# Feature Engineering: Industrial Production
def feature_industrial_production_value(df):
    if 'value' in df.columns:
        df['Industrial_Production_Value'] = df['value']
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_industrial_production_lag(df, lag_periods):
    for lag in lag_periods:
        if 'value' in df.columns:
            df[f'Industrial_Production_Lag_{lag}'] = df['value'].shift(lag)
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_industrial_production_rolling_mean(df, windows):
    for window in windows:
        if 'value' in df.columns:
            df[f'Industrial_Production_Rolling_Mean_{window}M'] = df['value'].rolling(window).mean()
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_industrial_production_cumulative_sum(df):
    if 'value' in df.columns:
        df['Industrial_Production_Cumulative_Sum'] = df['value'].cumsum()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_industrial_production_cumulative_product(df):
    if 'value' in df.columns:
        df['Industrial_Production_Cumulative_Product'] = df['value'].cumprod()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_industrial_production_trend(df):
    if 'value' in df.columns:
        df['Industrial_Production_Trend'] = df['value'].rolling(window=12).mean()  # Assuming a yearly trend for simplicity
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_industrial_production_ema(df, spans):
    for span in spans:
        if 'value' in df.columns:
            df[f'Industrial_Production_EMA_{span}'] = df['value'].ewm(span=span, adjust=False).mean()
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_industrial_production_z_score(df):
    if 'value' in df.columns:
        df['Industrial_Production_Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_industrial_production_days_since_trough(df):
    if 'value' in df.columns:
        df['Industrial_Production_Trough'] = df['value'].expanding().min()
        trough_dates = df[df['value'] == df['Industrial_Production_Trough']].groupby('Industrial_Production_Trough').apply(lambda x: x.index[0]).reset_index(drop=True)
        df['Industrial_Production_Days_Since_Trough'] = df.apply(lambda x: (x.name - trough_dates[trough_dates == x['Industrial_Production_Trough']].values[0]).days if any(trough_dates == x['Industrial_Production_Trough']) else np.nan, axis=1)
    else:
        print("Column 'value' not found in DataFrame.")
    return df

# Feature Engineering: Interest Rate
def feature_interest_rate_value(df):
    # Assuming 'value' column exists and directly using it without renaming
    if 'value' in df.columns:
        df['Interest_Rate_Value'] = df['value']
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_interest_rate_lag(df, lag_periods):
    for lag in lag_periods:
        if 'value' in df.columns:
            df[f'Interest_Rate_Lag_{lag}'] = df['value'].shift(lag)
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_interest_rate_rolling_mean(df, windows):
    for window in windows:
        if 'value' in df.columns:
            df[f'Interest_Rate_Rolling_Mean_{window}M'] = df['value'].rolling(window).mean()
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_interest_rate_ema(df, spans):
    for span in spans:
        if 'value' in df.columns:
            df[f'Interest_Rate_EMA_{span}'] = df['value'].ewm(span=span, adjust=False).mean()
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_interest_rate_days_since_peak(df):
    if 'value' in df.columns:
        df['Interest_Rate_Peak'] = df['value'].expanding().max()
        peak_dates = df[df['value'] == df['Interest_Rate_Peak']].groupby('Interest_Rate_Peak')['value'].apply(lambda x: x.index[0]).reset_index(drop=True)
        df['Interest_Rate_Days_Since_Peak'] = df.apply(lambda x: (x.name - peak_dates[peak_dates == x['Interest_Rate_Peak']].values[0]).days if any(peak_dates == x['Interest_Rate_Peak']) else np.nan, axis=1)
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_interest_rate_days_since_trough(df):
    if 'value' in df.columns:
        df['Interest_Rate_Trough'] = df['value'].expanding().min()
        trough_dates = df[df['value'] == df['Interest_Rate_Trough']].groupby('Interest_Rate_Trough')['value'].apply(lambda x: x.index[0]).reset_index(drop=True)
        df['Interest_Rate_Days_Since_Trough'] = df.apply(lambda x: (x.name - trough_dates[trough_dates == x['Interest_Rate_Trough']].values[0]).days if any(trough_dates == x['Interest_Rate_Trough']) else np.nan, axis=1)
    else:
        print("Column 'value' not found in DataFrame.")
    return df


# Feature Engineering: Labor Force Participation with SPY
def feature_labor_force_value(df):
    if 'value' in df.columns:
        df['Labor_Force_Value'] = df['value']
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_labor_force_lag(df, lag_periods):
    for lag in lag_periods:
        if 'value' in df.columns:
            df[f'Labor_Force_Lag_{lag}'] = df['value'].shift(lag)
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_labor_force_rolling_mean(df, windows):
    for window in windows:
        if 'value' in df.columns:
            df[f'Labor_Force_Rolling_Mean_{window}M'] = df['value'].rolling(window).mean()
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_labor_force_ema(df, spans):
    for span in spans:
        if 'value' in df.columns:
            df[f'Labor_Force_EMA_{span}'] = df['value'].ewm(span=span, adjust=False).mean()
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_labor_force_days_since_peak(df):
    if 'value' in df.columns:
        df['Labor_Force_Peak'] = df['value'].expanding().max()
        peak_dates = df[df['value'] == df['Labor_Force_Peak']].groupby('Labor_Force_Peak').apply(lambda x: x.index[0]).reset_index(drop=True)
        df['Labor_Force_Days_Since_Peak'] = df.apply(lambda x: (x.name - peak_dates[peak_dates == x['Labor_Force_Peak']].values[0]).days if any(peak_dates == x['Labor_Force_Peak']) else np.nan, axis=1)
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_labor_force_days_since_trough(df):
    if 'value' in df.columns:
        df['Labor_Force_Trough'] = df['value'].expanding().min()
        trough_dates = df[df['value'] == df['Labor_Force_Trough']].groupby('Labor_Force_Trough').apply(lambda x: x.index[0]).reset_index(drop=True)
        df['Labor_Force_Days_Since_Trough'] = df.apply(lambda x: (x.name - trough_dates[trough_dates == x['Labor_Force_Trough']].values[0]).days if any(trough_dates == x['Labor_Force_Trough']) else np.nan, axis=1)
    else:
        print("Column 'value' not found in DataFrame.")
    return df

# Feature Engineering: Non-Farm Payroll Employment
def feature_nonfarm_value(df):
    if 'value' in df.columns:
        df['NonFarm_Value'] = df['value']
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_nonfarm_lag(df, lag_periods):
    for lag in lag_periods:
        if 'value' in df.columns:
            df[f'NonFarm_Lag_{lag}'] = df['value'].shift(lag)
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_nonfarm_rolling_mean(df, windows):
    for window in windows:
        if 'value' in df.columns:
            df[f'NonFarm_Rolling_Mean_{window}M'] = df['value'].rolling(window).mean()
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_nonfarm_ema(df, spans):
    for span in spans:
        if 'value' in df.columns:
            df[f'NonFarm_EMA_{span}'] = df['value'].ewm(span=span, adjust=False).mean()
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_nonfarm_cumulative_sum(df):
    if 'value' in df.columns:
        df['NonFarm_Cumulative_Sum'] = df['value'].cumsum()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_nonfarm_cumulative_product(df):
    if 'value' in df.columns:
        df['NonFarm_Cumulative_Product'] = df['value'].cumprod()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_nonfarm_days_since_trough(df):
    if 'value' in df.columns:
        df['NonFarm_Trough'] = df['value'].expanding().min()
        trough_dates = df[df['value'] == df['NonFarm_Trough']].groupby('NonFarm_Trough').apply(lambda x: x.index[0]).reset_index(drop=True)
        df['NonFarm_Days_Since_Trough'] = df.apply(lambda x: (x.name - trough_dates[trough_dates == x['NonFarm_Trough']].values[0]).days if any(trough_dates == x['NonFarm_Trough']) else np.nan, axis=1)
    else:
        print("Column 'value' not found in DataFrame.")
    return df

# Feature Engineering: Personal Consumption Expenditures (PCE)
def feature_pce_value(df):
    if 'value' in df.columns:
        df['PCE_Value'] = df['value']
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_pce_lag(df, lag_periods):
    for lag in lag_periods:
        if 'value' in df.columns:
            df[f'PCE_Lag_{lag}'] = df['value'].shift(lag)
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_pce_rolling_mean(df, windows):
    for window in windows:
        if 'value' in df.columns:
            df[f'PCE_Rolling_Mean_{window}M'] = df['value'].rolling(window).mean()
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_pce_cumulative_sum(df):
    if 'value' in df.columns:
        df['PCE_Cumulative_Sum'] = df['value'].cumsum()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_pce_cumulative_product(df):
    if 'value' in df.columns:
        df['PCE_Cumulative_Product'] = df['value'].cumprod()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_pce_trend(df):
    if 'value' in df.columns:
        df['PCE_Trend'] = df['value'].ewm(span=12, adjust=False).mean()  # Example: Using EMA as a trend indicator
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_pce_ema(df, spans):
    for span in spans:
        if 'value' in df.columns:
            df[f'PCE_EMA_{span}'] = df['value'].ewm(span=span, adjust=False).mean()
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_pce_z_score(df):
    if 'value' in df.columns:
        df['PCE_Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_pce_days_since_peak(df):
    if 'value' in df.columns:
        df['PCE_Peak'] = df['value'].expanding().max()
        peak_dates = df[df['value'] == df['PCE_Peak']].groupby('PCE_Peak').apply(lambda x: x.index[0]).reset_index(drop=True)
        df['PCE_Days_Since_Peak'] = df.apply(lambda x: (x.name - peak_dates[peak_dates == x['PCE_Peak']].values[0]).days if any(peak_dates == x['PCE_Peak']) else np.nan, axis=1)
    else:
        print("Column 'value' not found in DataFrame.")
    return df

# Feature Engineering: Producer Price Index (PPI)
def feature_ppi_value(df):
    if 'value' in df.columns:
        df['PPI_Value'] = df['value']
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_ppi_lag(df, lag_periods):
    for lag in lag_periods:
        if 'value' in df.columns:
            df[f'PPI_Lag_{lag}'] = df['value'].shift(lag)
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_ppi_rolling_mean(df, windows):
    for window in windows:
        if 'value' in df.columns:
            df[f'PPI_Rolling_Mean_{window}M'] = df['value'].rolling(window).mean()
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_ppi_cumulative_sum(df):
    if 'value' in df.columns:
        df['PPI_Cumulative_Sum'] = df['value'].cumsum()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_ppi_cumulative_product(df):
    if 'value' in df.columns:
        df['PPI_Cumulative_Product'] = df['value'].cumprod()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_ppi_trend(df):
    if 'value' in df.columns:
        df['PPI_Trend'] = df['value'].ewm(span=12, adjust=False).mean()  # Example: Using EMA as a trend indicator
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_ppi_ema(df, spans):
    for span in spans:
        if 'value' in df.columns:
            df[f'PPI_EMA_{span}'] = df['value'].ewm(span=span, adjust=False).mean()
        else:
            print("Column 'value' not found in DataFrame.")
    return df

def feature_ppi_z_score(df):
    if 'value' in df.columns:
        df['PPI_Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_ppi_days_since_trough(df):
    if 'value' in df.columns:
        df['PPI_Trough'] = df['value'].expanding().min()
        trough_dates = df[df['value'] == df['PPI_Trough']].groupby('PPI_Trough').apply(lambda x: x.index[0]).reset_index(drop=True)
        df['PPI_Days_Since_Trough'] = df.apply(lambda x: (x.name - trough_dates[trough_dates == x['PPI_Trough']].values[0]).days if any(trough_dates == x['PPI_Trough']) else np.nan, axis=1)
    else:
        print("Column 'value' not found in DataFrame.")
    return df

# Feature Engineering: Unemployment Rate
def feature_unemployment_rate_cumulative_sum(df):
    if 'value' in df.columns:
        df['Unemployment_Rate_Cumulative_Sum'] = df['value'].cumsum()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_unemployment_rate_days_since_trough(df):
    if 'value' in df.columns:
        df['Unemployment_Rate_Trough'] = df['value'].expanding().min()
        trough_dates = df[df['value'] == df['Unemployment_Rate_Trough']].groupby('Unemployment_Rate_Trough').apply(lambda x: x.index[0]).reset_index(drop=True)
        df['Unemployment_Rate_Days_Since_Trough'] = df.apply(lambda x: (x.name - trough_dates[trough_dates == x['Unemployment_Rate_Trough']].values[0]).days if any(trough_dates == x['Unemployment_Rate_Trough']) else np.nan, axis=1)
    else:
        print("Column 'value' not found in DataFrame.")
    return df

# Feature Engineering: Historical Indicators
def feature_historical_indicator_ema_12(df):
    if 'value' in df.columns:
        df['Historical_Indicator_EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    else:
        print("Column 'value' not found in DataFrame.")
    return df

def feature_historical_indicator_rsi(df, window=14):
    if 'value' in df.columns:
        delta = df['value'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df['Historical_Indicator_RSI'] = 100 - (100 / (1 + rs))
    else:
        print("Column 'value' not found in DataFrame.")
    return df

# Feature Engineering: SPX Bull vs. Bear Markets
def feature_spx_ema(df, spans):
    for span in spans:
        if 'close' in df.columns:
            df[f'SPX_EMA_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        else:
            print("Column 'close' not found in DataFrame.")
    return df

def feature_spx_sma(df, windows):
    for window in windows:
        if 'close' in df.columns:
            df[f'SPX_SMA_{window}'] = df['close'].rolling(window=window).mean()
        else:
            print("Column 'close' not found in DataFrame.")
    return df

def feature_spx_drawdown_recovery(df):
    if 'drawdown' in df.columns:
        df['SPX_Drawdown'] = df['drawdown']
    else:
        print("Column 'drawdown' not found in DataFrame.")
    
    if 'recovery' in df.columns:
        df['SPX_Recovery'] = df['recovery']
    else:
        print("Column 'recovery' not found in DataFrame.")
    
    return df

# Feature Engineering: SPY Bull vs. Bear Markets
def feature_spy_price_data(df):
    if 'open' in df.columns:
        df['SPY_Open'] = df['open']
    if 'high' in df.columns:
        df['SPY_High'] = df['high']
    if 'low' in df.columns:
        df['SPY_Low'] = df['low']
    if 'close' in df.columns:
        df['SPY_Close'] = df['close']
    return df

def feature_spy_atr(df):
    if 'ATR' in df.columns:
        df['SPY_ATR'] = df['ATR']
    else:
        print("Column 'ATR' not found in DataFrame.")
    return df

def feature_spy_drawdown_recovery(df):
    if 'drawdown' in df.columns:
        df['SPY_Drawdown'] = df['drawdown']
    else:
        print("Column 'drawdown' not found in DataFrame.")
    
    if 'recovery' in df.columns:
        df['SPY_Recovery'] = df['recovery']
    else:
        print("Column 'recovery' not found in DataFrame.")
    
    return df

# Feature Engineering: Real Time Indicators
def feature_real_time_indicator_lag_1(df):
    if 'current_price' in df.columns:
        df['Real_Time_Indicator_Lag_1'] = df['current_price'].shift(1)
    else:
        print("Column 'current_price' not found in DataFrame. Skipping feature 'Real_Time_Indicator_Lag_1'.")
    return df

def feature_real_time_indicator_sma_20(df):
    if 'current_price' in df.columns:
        df['Real_Time_Indicator_SMA_20'] = df['current_price'].rolling(window=20).mean()
    else:
        print("Column 'current_price' not found in DataFrame. Skipping feature 'Real_Time_Indicator_SMA_20'.")
    return df

def feature_real_time_indicator_sma_50(df):
    if 'current_price' in df.columns:
        df['Real_Time_Indicator_SMA_50'] = df['current_price'].rolling(window=50).mean()
    else:
        print("Column 'current_price' not found in DataFrame. Skipping feature 'Real_Time_Indicator_SMA_50'.")
    return df

def feature_real_time_indicator_ema_12(df):
    if 'current_price' in df.columns:
        df['Real_Time_Indicator_EMA_12'] = df['current_price'].ewm(span=12, adjust=False).mean()
    else:
        print("Column 'current_price' not found in DataFrame. Skipping feature 'Real_Time_Indicator_EMA_12'.")
    return df

def feature_real_time_indicator_ema_26(df):
    if 'current_price' in df.columns:
        df['Real_Time_Indicator_EMA_26'] = df['current_price'].ewm(span=26, adjust=False).mean()
    else:
        print("Column 'current_price' not found in DataFrame. Skipping feature 'Real_Time_Indicator_EMA_26'.")
    return df

def feature_real_time_indicator_bollinger_bands(df):
    if 'current_price' in df.columns:
        sma = df['current_price'].rolling(window=20).mean()
        std = df['current_price'].rolling(window=20).std()
        df['Real_Time_Indicator_Bollinger_Upper'] = sma + (std * 2)
        df['Real_Time_Indicator_Bollinger_Lower'] = sma - (std * 2)
    else:
        print("Column 'current_price' not found in DataFrame. Skipping Bollinger Bands features.")
    return df

# Feature Engineering: Real Time SPX-VIX Correlation
def feature_real_time_spx_vix_correlation(df_spx, df_vix):
    # Use the appropriate date column
    date_column_spx = 'date'
    date_column_vix = 'timestamp'

    # Convert to datetime
    df_spx[date_column_spx] = pd.to_datetime(df_spx[date_column_spx], errors='coerce')
    df_vix[date_column_vix] = pd.to_datetime(df_vix[date_column_vix], errors='coerce')

    # Rename to 'date' for consistency
    df_spx.rename(columns={date_column_spx: 'date'}, inplace=True)
    df_vix.rename(columns={date_column_vix: 'date'}, inplace=True)

    # Set 'date' as the index
    df_spx.set_index('date', inplace=True)
    df_vix.set_index('date', inplace=True)

    # Concatenate the DataFrames
    df_combined = pd.concat([df_spx['current_price'], df_vix['current_price']], axis=1, keys=['close_spx', 'close_vix'])

    # Calculate the rolling correlation
    df_combined['Real_Time_SPX_VIX_Correlation'] = df_combined['close_spx'].rolling(window=30).corr(df_combined['close_vix'])

    return df_combined[['Real_Time_SPX_VIX_Correlation']].reset_index()


# Feature Engineering: Real Time SPY-VIX Correlation
def feature_real_time_spy_vix_correlation(df_spy, df_vix):
    # Use the appropriate date column
    date_column_spy = 'timestamp'
    date_column_vix = 'timestamp'

    # Convert to datetime
    df_spy[date_column_spy] = pd.to_datetime(df_spy[date_column_spy], errors='coerce')
    df_vix[date_column_vix] = pd.to_datetime(df_vix[date_column_vix], errors='coerce')

    # Rename to 'date' for consistency
    df_spy.rename(columns={date_column_spy: 'date'}, inplace=True)
    df_vix.rename(columns={date_column_vix: 'date'}, inplace=True)

    # Set 'date' as the index
    df_spy.set_index('date', inplace=True)
    df_vix.set_index('date', inplace=True)

    # Concatenate the DataFrames
    df_combined = pd.concat([df_spy['current_price'], df_vix['current_price']], axis=1, keys=['close_spy', 'close_vix'])

    # Calculate the rolling correlation
    df_combined['Real_Time_SPY_VIX_Correlation'] = df_combined['close_spy'].rolling(window=30).corr(df_combined['close_vix'])

    return df_combined[['Real_Time_SPY_VIX_Correlation']].reset_index()
    
        
# Main functions for Feature Engineering
def main_consumer_confidence_features(df):
    df = feature_consumer_confidence_cumulative_sum(df)
    df = feature_consumer_confidence_days_since_peak(df)
    df = feature_consumer_confidence_days_since_trough(df)
    return df

def main_consumer_sentiment_features(df):
    df = feature_consumer_sentiment_rolling_6m_average(df)
    df = feature_consumer_sentiment_rolling_12m_average(df)
    return df

def main_core_inflation_features(df):
    df = feature_core_inflation_value(df)
    df = feature_core_inflation_ema_12(df)
    df = feature_core_inflation_ema_26(df)
    df = feature_core_inflation_cumulative_sum(df)
    df = feature_core_inflation_z_score(df)
    df = feature_core_inflation_trend(df)
    return df

def main_cpi_features(df):
    df = feature_cpi_value(df)
    df = feature_cpi_trend(df)
    df = feature_cpi_cumulative_sum(df)
    df = feature_cpi_days_since_peak(df)
    df = feature_cpi_days_since_trough(df)
    return df

def main_gdp_features(df):
    df = feature_gdp_value(df)
    df = feature_gdp_lag(df, [1, 2, 4, 8])
    df = feature_gdp_rolling_mean(df, [4, 8])
    df = feature_gdp_rolling_std(df, [4, 8])
    df = feature_gdp_cumulative_sum(df)
    df = feature_gdp_cumulative_product(df)
    df = feature_gdp_trend(df)
    df = feature_gdp_ema(df, [4, 8])
    return df

def main_industrial_production_features(df):
    df = feature_industrial_production_value(df)
    df = feature_industrial_production_lag(df, [1, 3, 12])
    df = feature_industrial_production_rolling_mean(df, [3, 6, 12])
    df = feature_industrial_production_cumulative_sum(df)
    df = feature_industrial_production_cumulative_product(df)
    df = feature_industrial_production_trend(df)
    df = feature_industrial_production_ema(df, [12, 26, 50])
    df = feature_industrial_production_z_score(df)
    df = feature_industrial_production_days_since_trough(df)
    return df

def main_interest_rate_features(df):
    if 'symbol' in df.columns:
        df = filter_spx_data(df)  # or filter_spy_data(df) based on the context
    df = feature_interest_rate_value(df)
    df = feature_interest_rate_lag(df, [1, 3, 12])
    df = feature_interest_rate_rolling_mean(df, [3, 6, 12])
    df = feature_interest_rate_ema(df, [12, 26, 50])
    df = feature_interest_rate_days_since_peak(df)
    df = feature_interest_rate_days_since_trough(df)
    return df

def main_labor_force_features(df):
    # Check if 'symbol' column exists before filtering
    if 'symbol' in df.columns:
        df = filter_spy_data(df)  # Only apply if 'symbol' column exists
    df = feature_labor_force_value(df)
    df = feature_labor_force_lag(df, [1, 3, 12])
    df = feature_labor_force_rolling_mean(df, [3, 6, 12])
    df = feature_labor_force_ema(df, [12, 26, 50])
    df = feature_labor_force_days_since_peak(df)
    df = feature_labor_force_days_since_trough(df)
    return df

def main_nonfarm_features(df):
    df = feature_nonfarm_value(df)
    df = feature_nonfarm_lag(df, [1, 3, 12])
    df = feature_nonfarm_rolling_mean(df, [3, 6, 12])
    df = feature_nonfarm_ema(df, [12, 26, 50])
    df = feature_nonfarm_cumulative_sum(df)
    df = feature_nonfarm_cumulative_product(df)
    df = feature_nonfarm_days_since_trough(df)
    return df

def main_pce_features(df):
    df = feature_pce_value(df)
    df = feature_pce_lag(df, [1, 3, 12])
    df = feature_pce_rolling_mean(df, [3, 6, 12])
    df = feature_pce_cumulative_sum(df)
    df = feature_pce_cumulative_product(df)
    df = feature_pce_trend(df)
    df = feature_pce_ema(df, [12, 26, 50])
    df = feature_pce_z_score(df)
    df = feature_pce_days_since_peak(df)
    return df

def main_ppi_features(df):
    df = feature_ppi_value(df)
    df = feature_ppi_lag(df, [1, 3, 12])
    df = feature_ppi_rolling_mean(df, [3, 6, 12])
    df = feature_ppi_cumulative_sum(df)
    df = feature_ppi_cumulative_product(df)
    df = feature_ppi_trend(df)
    df = feature_ppi_ema(df, [12, 26, 50])
    df = feature_ppi_z_score(df)
    df = feature_ppi_days_since_trough(df)
    return df

def main_unemployment_rate_features(df):
    df = feature_unemployment_rate_cumulative_sum(df)
    df = feature_unemployment_rate_days_since_trough(df)
    return df

def main_historical_indicator_features(df):
    df = feature_historical_indicator_ema_12(df)
    df = feature_historical_indicator_rsi(df)
    return df

def main_spx_market_features(df):
    df = feature_spx_ema(df, [12, 26])
    df = feature_spx_sma(df, [50, 100])
    df = feature_spx_drawdown_recovery(df)
    return df

def main_spy_market_features(df):
    df = feature_spy_price_data(df)
    df = feature_spy_atr(df)
    df = feature_spy_drawdown_recovery(df)
    return df

def main_real_time_spx_features(df):
    df = filter_spx_data(df)
    df = feature_real_time_indicator_lag_1(df)
    df = feature_real_time_indicator_sma_20(df)
    df = feature_real_time_indicator_sma_50(df)
    df = feature_real_time_indicator_ema_12(df)
    df = feature_real_time_indicator_ema_26(df)
    df = feature_real_time_indicator_bollinger_bands(df)
    return df

def main_real_time_spy_features(df):
    df = filter_spy_data(df)
    df = feature_real_time_indicator_lag_1(df)
    df = feature_real_time_indicator_sma_20(df)
    df = feature_real_time_indicator_sma_50(df)
    df = feature_real_time_indicator_ema_12(df)
    df = feature_real_time_indicator_ema_26(df)
    df = feature_real_time_indicator_bollinger_bands(df)
    return df

def main_real_time_vix_correlation_features(df_spx, df_spy, df_vix):
    df_spx_vix_correlation = feature_real_time_spx_vix_correlation(df_spx, df_vix)
    df_spy_vix_correlation = feature_real_time_spy_vix_correlation(df_spy, df_vix)
    
    # Merging the SPX-VIX and SPY-VIX correlations into respective dataframes
    df_spx = pd.merge(df_spx, df_spx_vix_correlation, left_index=True, right_index=True, how='left')
    df_spy = pd.merge(df_spy, df_spy_vix_correlation, left_index=True, right_index=True, how='left')
    
    return df_spx, df_spy


# Apply Features (Main Main)
def apply_features(df, dataset_name, real_time_vix=None):
    if dataset_name == 'consumer_confidence_data':
        return main_consumer_confidence_features(df)
    elif dataset_name == 'consumer_sentiment_data':
        return main_consumer_sentiment_features(df)
    elif dataset_name == 'core_inflation_data':
        return main_core_inflation_features(df)
    elif dataset_name == 'cpi_data':
        return main_cpi_features(df)
    elif dataset_name == 'gdp_data':
        return main_gdp_features(df)
    elif dataset_name == 'industrial_production_data':
        return main_industrial_production_features(df)
    elif dataset_name == 'interest_rate_data':
        return main_interest_rate_features(df)
    elif dataset_name == 'labor_force_participation_rate_data':
        return main_labor_force_features(df)
    elif dataset_name == 'nonfarm_payroll_employment_data':
        return main_nonfarm_features(df)
    elif dataset_name == 'personal_consumption_expenditures_data':
        return main_pce_features(df)
    elif dataset_name == 'ppi_data':
        return main_ppi_features(df)
    elif dataset_name == 'unemployment_rate_data':
        return main_unemployment_rate_features(df)
    elif dataset_name == 'historical_indicator_data':
        return main_historical_indicator_features(df)
    elif dataset_name == 'spx_market_data': 
        return main_spx_market_features(df)
    elif dataset_name == 'spy_market_data': 
        return main_spy_market_features(df)
    elif dataset_name == 'real_time_spx':
        df = main_real_time_spx_features(df)
        if real_time_vix is not None:
            spx_vix_corr = feature_real_time_spx_vix_correlation(df, real_time_vix)
            df = df.join(spx_vix_corr)
        return df
    elif dataset_name == 'real_time_spy':
        df = main_real_time_spy_features(df)
        if real_time_vix is not None:
            spy_vix_corr = feature_real_time_spy_vix_correlation(df, real_time_vix)
            df = df.join(spy_vix_corr)
        return df
    else:
        print(f"No specific feature engineering function defined for {dataset_name}.")
        return df

# Save enhanced data to the featured directory
def save_enhanced_data(df, filename):
    enhanced_filename = f"featured_{filename}"
    df.to_csv(os.path.join(FEATURED_DIR, enhanced_filename))
    print(f"Saved enhanced data to {os.path.join(FEATURED_DIR, enhanced_filename)}")

# MAIN function to perform feature engineering
def main():
    # Load all data
    data_dict = load_data()
    
    # Extract the real_time_vix DataFrame
    real_time_vix = data_dict.get('real_time_vix')

    # Process each dataset
    for data_name, df in data_dict.items():
        print(f"Processing {data_name}...")

        # Ensure the date column is in datetime format
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            print("No 'date' column found, using default index.")
        
        # Apply specific feature engineering based on dataset, passing real_time_vix as needed
        df = apply_features(df, data_name, real_time_vix)
        
        # Save the resulting DataFrame to the featured directory
        enhanced_filename = data_name + '_featured.csv'
        save_enhanced_data(df, enhanced_filename)
        
    # Ensure the directory exists before running main
if __name__ == "__main__":
    ensure_directory_exists(FEATURED_DIR)
    main()