import sys
import os
import pandas as pd
import logging
import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam

# Add this:
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)

try:
    from ai.utils.aws_s3_utils import auto_upload_file_to_s3
except ImportError:
    logging.warning("Could not import 'auto_upload_file_to_s3' from ai.utils.aws_s3_utils. Check your import paths!")


# Load environment variables
load_dotenv()

# Base directory settings
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

# Paths from environment variables
DATA_DIR = os.path.join(BASE_DIR, 'data/lumen_2/processed')
FEATURED_DIR = os.path.join(BASE_DIR, 'data/lumen_2/featured')
MODEL_DIR = os.getenv('MODEL_DIR', 'models/lumen_2')
MODEL_NAME = 'lumen_2'

# Initialize logging
logging.basicConfig(level=logging.INFO)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

# 1) Define a folder for sequences inside FEATURED_DIR
SEQUENCES_DIR = os.path.join(FEATURED_DIR, 'sequences')


def ensure_timestamp_column(df, dataset_name):
    if dataset_name == 'real_time_vix':
        if 'timestamp' not in df.columns:
            if 'date' in df.columns:
                df.rename(columns={'date': 'timestamp'}, inplace=True)
                logging.info(f"{dataset_name}: Renamed 'date' column to 'timestamp'.")
            else:
                logging.warning(f"{dataset_name}: No 'timestamp' or 'date' column found. Creating dummy timestamps.")
                df['timestamp'] = pd.date_range(start='2000-01-01', periods=len(df), freq='D')
    elif dataset_name == 'historical_vix':
        if 'timestamp' not in df.columns and 'date' not in df.columns:
            logging.warning(f"{dataset_name}: No 'timestamp' or 'date' column found. Leaving as is.")
    return df

def load_data():
    consumer_confidence_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_consumer_confidence_data.csv'))
    consumer_sentiment_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_consumer_sentiment_data.csv'))
    core_inflation_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_core_inflation_data.csv'))
    cpi_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_cpi_data.csv'))
    gdp_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_gdp_data.csv'))
    historical_spx = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_historical_spx.csv'))
    historical_spy = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_historical_spy.csv'))
    historical_vix = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_historical_vix.csv'))
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

    logging.info("[load_data - Immediately After Reading real_time_vix] Columns: %s", real_time_vix.columns.tolist())
    logging.info("[load_data - Immediately After Reading real_time_vix] Head:\n%s", real_time_vix.head().to_string())

    real_time_vix = ensure_timestamp_column(real_time_vix, 'real_time_vix')

    logging.info("[load_data - After Ensuring Timestamp] real_time_vix head:")
    logging.info(real_time_vix.head().to_string())
    unemployment_rate_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_unemployment_rate_data.csv'))
    

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

def memory_friendly_rolling_corr(x, y, window):
    import collections
    from math import sqrt
    x = x.astype('float32')
    y = y.astype('float32')
    n = len(x)
    result = np.full(n, np.nan, dtype='float32')
    if window > n:
        return pd.Series(result, index=x.index)
    window_x = collections.deque()
    window_y = collections.deque()
    sum_x = sum_y = sum_xy = sum_x2 = sum_y2 = 0.0
    for i in range(n):
        xx = x[i]
        yy = y[i]
        window_x.append(xx)
        window_y.append(yy)
        sum_x += xx
        sum_y += yy
        sum_xy += xx*yy
        sum_x2 += xx*xx
        sum_y2 += yy*yy
        if i >= window:
            oldx = window_x.popleft()
            oldy = window_y.popleft()
            sum_x -= oldx
            sum_y -= oldy
            sum_xy -= oldx*oldy
            sum_x2 -= oldx*oldx
            sum_y2 -= oldy*oldy
        if i >= window-1:
            numerator = (window*sum_xy - sum_x*sum_y)
            denominator = (window*sum_x2 - sum_x*sum_x)*(window*sum_y2 - sum_y*sum_y)
            if denominator <= 0:
                c = np.nan
            else:
                c = numerator/np.sqrt(denominator)
            result[i] = c
    return pd.Series(result, index=x.index)

def feature_consumer_confidence_cumulative_sum(df):
    if 'value' in df.columns:
        df['Consumer_Confidence_Cumulative_Sum'] = df['value'].cumsum()
    return df

def feature_consumer_confidence_days_since_peak(df):
    if 'value' in df.columns:
        df['Consumer_Confidence_Peak'] = df['value'].expanding().max()
    return df

def feature_consumer_confidence_days_since_trough(df):
    if 'value' in df.columns:
        df['Consumer_Confidence_Trough'] = df['value'].expanding().min()
    return df

def feature_consumer_sentiment_rolling_6m_average(df):
    if 'value' in df.columns:
        df['Consumer_Sentiment_Rolling_6M_Avg'] = df['value'].rolling(window=6).mean()
    return df

def feature_consumer_sentiment_rolling_12m_average(df):
    if 'value' in df.columns:
        df['Consumer_Sentiment_Rolling_12M_Avg'] = df['value'].rolling(window=12).mean()
    return df

def feature_core_inflation_value(df):
    if 'value' in df.columns:
        df['Core_Inflation_Value'] = df['value']
    return df

def feature_core_inflation_ema_12(df):
    if 'value' in df.columns:
        df['Core_Inflation_EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    return df

def feature_core_inflation_ema_26(df):
    if 'value' in df.columns:
        df['Core_Inflation_EMA_26'] = df['value'].ewm(span=26, adjust=False).mean()
    return df

def feature_core_inflation_cumulative_sum(df):
    if 'value' in df.columns:
        df['Core_Inflation_Cumulative_Sum'] = df['value'].cumsum()
    return df

def feature_core_inflation_z_score(df):
    if 'value' in df.columns:
        df['Core_Inflation_Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()
    return df

def feature_core_inflation_trend(df):
    if 'value' in df.columns:
        df['Core_Inflation_Trend'] = df['value'].rolling(window=12).mean()
    return df

def feature_cpi_value(df):
    if 'value' in df.columns:
        df['CPI_Value'] = df['value']
    return df

def feature_cpi_trend(df):
    if 'value' in df.columns:
        df['CPI_Trend'] = df['value'].rolling(window=12).mean()
    return df

def feature_cpi_cumulative_sum(df):
    if 'value' in df.columns:
        df['CPI_Cumulative_Sum'] = df['value'].cumsum()
    return df

def feature_cpi_days_since_peak(df):
    if 'value' in df.columns:
        df['CPI_Peak'] = df['value'].expanding().max()
    return df

def feature_cpi_days_since_trough(df):
    if 'value' in df.columns:
        df['CPI_Trough'] = df['value'].expanding().min()
    return df

def feature_gdp_value(df):
    if 'value' in df.columns:
        df['GDP_Value'] = df['value']
    return df

def feature_gdp_lag(df, lag_periods):
    for lag in lag_periods:
        if 'value' in df.columns:
            df[f'GDP_Lag_{lag}'] = df['value'].shift(lag)
    return df

def feature_gdp_rolling_mean(df, windows):
    for window in windows:
        if 'value' in df.columns:
            df[f'GDP_Rolling_Mean_{window}Q'] = df['value'].rolling(window).mean()
    return df

def feature_gdp_rolling_std(df, windows):
    for window in windows:
        if 'value' in df.columns:
            df[f'GDP_Rolling_Std_{window}Q'] = df['value'].rolling(window).std()
    return df

def feature_gdp_cumulative_sum(df):
    if 'value' in df.columns:
        df['GDP_Cumulative_Sum'] = df['value'].cumsum()
    return df

def feature_gdp_cumulative_product(df):
    if 'value' in df.columns:
        df['GDP_Cumulative_Product'] = df['value'].cumprod()
    return df

def feature_gdp_trend(df):
    if 'value' in df.columns:
        df['GDP_Trend'] = df['value'].rolling(window=12).mean()
    return df

def feature_gdp_ema(df, spans):
    for span in spans:
        if 'value' in df.columns:
            df[f'GDP_EMA_{span}'] = df['value'].ewm(span=span, adjust=False).mean()
    return df

def feature_industrial_production_value(df):
    if 'value' in df.columns:
        df['Industrial_Production_Value'] = df['value']
    return df

def feature_industrial_production_lag(df, lag_periods):
    for lag in lag_periods:
        if 'value' in df.columns:
            df[f'Industrial_Production_Lag_{lag}'] = df['value'].shift(lag)
    return df

def feature_industrial_production_rolling_mean(df, windows):
    for window in windows:
        if 'value' in df.columns:
            df[f'Industrial_Production_Rolling_Mean_{window}M'] = df['value'].rolling(window).mean()
    return df

def feature_industrial_production_cumulative_sum(df):
    if 'value' in df.columns:
        df['Industrial_Production_Cumulative_Sum'] = df['value'].cumsum()
    return df

def feature_industrial_production_cumulative_product(df):
    if 'value' in df.columns:
        df['Industrial_Production_Cumulative_Product'] = df['value'].cumprod()
    return df

def feature_industrial_production_trend(df):
    if 'value' in df.columns:
        df['Industrial_Production_Trend'] = df['value'].rolling(window=12).mean()
    return df

def feature_industrial_production_ema(df, spans):
    for span in spans:
        if 'value' in df.columns:
            df[f'Industrial_Production_EMA_{span}'] = df['value'].ewm(span=span, adjust=False).mean()
    return df

def feature_industrial_production_z_score(df):
    if 'value' in df.columns:
        df['Industrial_Production_Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()
    return df

def feature_industrial_production_days_since_trough(df):
    if 'value' in df.columns:
        df['Industrial_Production_Trough'] = df['value'].expanding().min()
    return df

def feature_interest_rate_value(df):
    if 'value' in df.columns:
        df['Interest_Rate_Value'] = df['value']
    return df

def feature_interest_rate_lag(df, lag_periods):
    for lag in lag_periods:
        if 'value' in df.columns:
            df[f'Interest_Rate_Lag_{lag}'] = df['value'].shift(lag)
    return df

def feature_interest_rate_rolling_mean(df, windows):
    for window in windows:
        if 'value' in df.columns:
            df[f'Interest_Rate_Rolling_Mean_{window}M'] = df['value'].rolling(window).mean()
    return df

def feature_interest_rate_ema(df, spans):
    for span in spans:
        if 'value' in df.columns:
            df[f'Interest_Rate_EMA_{span}'] = df['value'].ewm(span=span, adjust=False).mean()
    return df

def feature_interest_rate_days_since_peak(df):
    if 'value' in df.columns:
        df['Interest_Rate_Peak'] = df['value'].expanding().max()
    return df

def feature_interest_rate_days_since_trough(df):
    if 'value' in df.columns:
        df['Interest_Rate_Trough'] = df['value'].expanding().min()
    return df

def feature_labor_force_value(df):
    if 'value' in df.columns:
        df['Labor_Force_Value'] = df['value']
    return df

def feature_labor_force_lag(df, lag_periods):
    for lag in lag_periods:
        if 'value' in df.columns:
            df[f'Labor_Force_Lag_{lag}'] = df['value'].shift(lag)
    return df

def feature_labor_force_rolling_mean(df, windows):
    for window in windows:
        if 'value' in df.columns:
            df[f'Labor_Force_Rolling_Mean_{window}M'] = df['value'].rolling(window).mean()
    return df

def feature_labor_force_ema(df, spans):
    for span in spans:
        if 'value' in df.columns:
            df[f'Labor_Force_EMA_{span}'] = df['value'].ewm(span=span, adjust=False).mean()
    return df

def feature_labor_force_days_since_peak(df):
    if 'value' in df.columns:
        df['Labor_Force_Peak'] = df['value'].expanding().max()
    return df

def feature_labor_force_days_since_trough(df):
    if 'value' in df.columns:
        df['Labor_Force_Trough'] = df['value'].expanding().min()
    return df

def feature_nonfarm_value(df):
    if 'value' in df.columns:
        df['NonFarm_Value'] = df['value']
    return df

def feature_nonfarm_lag(df, lag_periods):
    for lag in lag_periods:
        if 'value' in df.columns:
            df[f'NonFarm_Lag_{lag}'] = df['value'].shift(lag)
    return df

def feature_nonfarm_rolling_mean(df, windows):
    for window in windows:
        if 'value' in df.columns:
            df[f'NonFarm_Rolling_Mean_{window}M'] = df['value'].rolling(window).mean()
    return df

def feature_nonfarm_ema(df, spans):
    for span in spans:
        if 'value' in df.columns:
            df[f'NonFarm_EMA_{span}'] = df['value'].ewm(span=span, adjust=False).mean()
    return df

def feature_nonfarm_cumulative_sum(df):
    if 'value' in df.columns:
        df['NonFarm_Cumulative_Sum'] = df['value'].cumsum()
    return df

def feature_nonfarm_cumulative_product(df):
    if 'value' in df.columns:
        df['NonFarm_Cumulative_Product'] = df['value'].cumprod()
    return df

def feature_nonfarm_days_since_trough(df):
    if 'value' in df.columns:
        df['NonFarm_Trough'] = df['value'].expanding().min()
    return df

def feature_pce_value(df):
    if 'value' in df.columns:
        df['PCE_Value'] = df['value']
    return df

def feature_pce_lag(df, lag_periods):
    for lag in lag_periods:
        if 'value' in df.columns:
            df[f'PCE_Lag_{lag}'] = df['value'].shift(lag)
    return df

def feature_pce_rolling_mean(df, windows):
    for window in windows:
        if 'value' in df.columns:
            df[f'PCE_Rolling_Mean_{window}M'] = df['value'].rolling(window).mean()
    return df

def feature_pce_cumulative_sum(df):
    if 'value' in df.columns:
        df['PCE_Cumulative_Sum'] = df['value'].cumsum()
    return df

def feature_pce_cumulative_product(df):
    if 'value' in df.columns:
        df['PCE_Cumulative_Product'] = df['value'].cumprod()
    return df

def feature_pce_trend(df):
    if 'value' in df.columns:
        df['PCE_Trend'] = df['value'].ewm(span=12, adjust=False).mean()
    return df

def feature_pce_ema(df, spans):
    for span in spans:
        if 'value' in df.columns:
            df[f'PCE_EMA_{span}'] = df['value'].ewm(span=span, adjust=False).mean()
    return df

def feature_pce_z_score(df):
    if 'value' in df.columns:
        df['PCE_Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()
    return df

def feature_pce_days_since_peak(df):
    if 'value' in df.columns:
        df['PCE_Peak'] = df['value'].expanding().max()
    return df

def feature_ppi_value(df):
    if 'value' in df.columns:
        df['PPI_Value'] = df['value']
    return df

def feature_ppi_lag(df, lag_periods):
    for lag in lag_periods:
        if 'value' in df.columns:
            df[f'PPI_Lag_{lag}'] = df['value'].shift(lag)
    return df

def feature_ppi_rolling_mean(df, windows):
    for window in windows:
        if 'value' in df.columns:
            df[f'PPI_Rolling_Mean_{window}M'] = df['value'].rolling(window).mean()
    return df

def feature_ppi_cumulative_sum(df):
    if 'value' in df.columns:
        df['PPI_Cumulative_Sum'] = df['value'].cumsum()
    return df

def feature_ppi_cumulative_product(df):
    if 'value' in df.columns:
        df['PPI_Cumulative_Product'] = df['value'].cumprod()
    return df

def feature_ppi_trend(df):
    if 'value' in df.columns:
        df['PPI_Trend'] = df['value'].ewm(span=12, adjust=False).mean()
    return df

def feature_ppi_ema(df, spans):
    for span in spans:
        if 'value' in df.columns:
            df[f'PPI_EMA_{span}'] = df['value'].ewm(span=span, adjust=False).mean()
    return df

def feature_ppi_z_score(df):
    if 'value' in df.columns:
        df['PPI_Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()
    return df

def feature_ppi_days_since_trough(df):
    if 'value' in df.columns:
        df['PPI_Trough'] = df['value'].expanding().min()
    return df

def feature_unemployment_rate_cumulative_sum(df):
    if 'value' in df.columns:
        df['Unemployment_Rate_Cumulative_Sum'] = df['value'].cumsum()
    return df

def feature_unemployment_rate_days_since_trough(df):
    if 'value' in df.columns:
        df['Unemployment_Rate_Trough'] = df['value'].expanding().min()
    return df

def feature_historical_indicator_ema_12(df):
    if 'value' in df.columns:
        df['Historical_Indicator_EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    return df

def feature_historical_indicator_rsi(df, window=14):
    if 'value' in df.columns:
        delta = df['value'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df['Historical_Indicator_RSI'] = 100 - (100 / (1 + rs))
    return df

def feature_spx_ema(df, spans):
    for span in spans:
        if 'close' in df.columns:
            df[f'SPX_EMA_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    return df

def feature_spx_sma(df, windows):
    for window in windows:
        if 'close' in df.columns:
            df[f'SPX_SMA_{window}'] = df['close'].rolling(window=window).mean()
    return df

def feature_spx_drawdown_recovery(df):
    if 'drawdown' in df.columns:
        df['SPX_Drawdown'] = df['drawdown']
    if 'recovery' in df.columns:
        df['SPX_Recovery'] = df['recovery']
    return df

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
    return df

def feature_spy_drawdown_recovery(df):
    if 'drawdown' in df.columns:
        df['SPY_Drawdown'] = df['drawdown']
    if 'recovery' in df.columns:
        df['SPY_Recovery'] = df['recovery']
    return df

def feature_real_time_indicator_lag_1(df):
    if 'current_price' in df.columns:
        df['Real_Time_Indicator_Lag_1'] = df['current_price'].shift(1)
    return df

def feature_real_time_indicator_sma_20(df):
    if 'current_price' in df.columns:
        df['Real_Time_Indicator_SMA_20'] = df['current_price'].rolling(window=20).mean()
    return df

def feature_real_time_indicator_sma_50(df):
    if 'current_price' in df.columns:
        df['Real_Time_Indicator_SMA_50'] = df['current_price'].rolling(window=50).mean()
    return df

def feature_real_time_indicator_ema_12(df):
    if 'current_price' in df.columns:
        df['Real_Time_Indicator_EMA_12'] = df['current_price'].ewm(span=12, adjust=False).mean()
    return df

def feature_real_time_indicator_ema_26(df):
    if 'current_price' in df.columns:
        df['Real_Time_Indicator_EMA_26'] = df['current_price'].ewm(span=26, adjust=False).mean()
    return df

def feature_real_time_spx_vix_correlation(df_spx, df_vix):
    logging.info(f"[feature_real_time_spx_vix_correlation] df_spx columns: {df_spx.columns.tolist()}")
    logging.info(f"[feature_real_time_spx_vix_correlation] df_vix columns: {df_vix.columns.tolist()}")

    if not isinstance(df_vix, pd.DataFrame):
        logging.error("df_vix is not a pandas DataFrame.")
        return pd.DataFrame()

    logging.info("Inspecting df_vix DataFrame for SPX correlation:")
    logging.info(df_vix.head())
    logging.info(f"Columns in df_vix: {df_vix.columns.tolist()}")

    if 'timestamp' not in df_spx.columns:
        logging.error("The 'timestamp' column is missing from the SPX DataFrame. Exiting function.")
        return pd.DataFrame()

    if 'timestamp' not in df_vix.columns:
        logging.error("The 'timestamp' column is missing from the VIX DataFrame. Cannot proceed.")
        return pd.DataFrame()

    df_spx_copy = df_spx.copy()
    df_vix_copy = df_vix.copy()

    # Convert to datetime just in case
    df_spx_copy['timestamp'] = pd.to_datetime(df_spx_copy['timestamp'], errors='coerce')
    df_vix_copy['timestamp'] = pd.to_datetime(df_vix_copy['timestamp'], errors='coerce')

    # Reset indices if they were set to 'timestamp'
    if df_spx_copy.index.name == 'timestamp':
        df_spx_copy.reset_index(drop=True, inplace=True)
    if df_vix_copy.index.name == 'timestamp':
        df_vix_copy.reset_index(drop=True, inplace=True)

    # Use 'timestamp' as the new index
    df_spx_copy.set_index('timestamp', inplace=True, drop=True)
    df_vix_copy.set_index('timestamp', inplace=True, drop=True)

    # --- NEW PART: rename 'close' -> 'current_price' if needed ---
    if 'current_price' not in df_spx_copy.columns and 'close' in df_spx_copy.columns:
        df_spx_copy.rename(columns={'close': 'current_price'}, inplace=True)

    if 'current_price' not in df_vix_copy.columns and 'close' in df_vix_copy.columns:
        df_vix_copy.rename(columns={'close': 'current_price'}, inplace=True)

    # Check again
    if 'current_price' not in df_spx_copy.columns or 'current_price' not in df_vix_copy.columns:
        logging.error("Missing 'current_price' column in SPX or VIX DataFrame.")
        return pd.DataFrame()

    # Align on timestamps
    common_index = df_spx_copy.index.intersection(df_vix_copy.index)
    df_spx_aligned = df_spx_copy.loc[common_index, ['current_price']].copy()
    df_vix_aligned = df_vix_copy.loc[common_index, ['current_price']].copy()

    # Perform rolling correlation
    df_spx_aligned['Real_Time_SPX_VIX_Correlation'] = memory_friendly_rolling_corr(
        df_spx_aligned['current_price'],
        df_vix_aligned['current_price'],
        window=30
    )

    final_df = df_spx_aligned[['Real_Time_SPX_VIX_Correlation']].reset_index()

    logging.info("Completed SPX-VIX correlation feature calculation in a memory-friendly manner.")
    return final_df


def feature_real_time_spy_vix_correlation(df_spy, df_vix):
    print("\n--- Starting SPY-VIX Correlation ---")
    print("Initial df_spy state:")
    print(df_spy.head())
    print("Columns in df_spy:", df_spy.columns)

    print("Initial df_vix state:")
    print(df_vix.head())
    print("Columns in df_vix:", df_vix.columns)

    if 'timestamp' not in df_spy.columns:
        print("The 'timestamp' column is missing from the SPY DataFrame. Exiting function.")
        return pd.DataFrame()

    if 'timestamp' not in df_vix.columns:
        logging.warning("The 'timestamp' column is missing from the VIX DataFrame. Adding default 'timestamp' column.")
        df_vix['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df_vix), freq='T')

    df_spy_copy = df_spy.copy()
    df_vix_copy = df_vix.copy()

    df_spy_copy['timestamp'] = pd.to_datetime(df_spy_copy['timestamp'], errors='coerce')
    df_vix_copy['timestamp'] = pd.to_datetime(df_vix_copy['timestamp'], errors='coerce')

    if df_spy_copy.index.name == 'timestamp':
        df_spy_copy.reset_index(drop=True, inplace=True)
    if df_vix_copy.index.name == 'timestamp':
        df_vix_copy.reset_index(drop=True, inplace=True)

    df_spy_copy.set_index('timestamp', inplace=True, drop=True)
    df_vix_copy.set_index('timestamp', inplace=True, drop=True)

    # --- NEW PART: rename 'close' -> 'current_price' if needed ---
    if 'current_price' not in df_spy_copy.columns and 'close' in df_spy_copy.columns:
        df_spy_copy.rename(columns={'close': 'current_price'}, inplace=True)

    if 'current_price' not in df_vix_copy.columns and 'close' in df_vix_copy.columns:
        df_vix_copy.rename(columns={'close': 'current_price'}, inplace=True)

    if 'current_price' not in df_spy_copy.columns or 'current_price' not in df_vix_copy.columns:
        print("Missing 'current_price' column in SPY or VIX DataFrame.")
        return pd.DataFrame()

    common_index = df_spy_copy.index.intersection(df_vix_copy.index)
    df_spy_aligned = df_spy_copy.loc[common_index, ['current_price']].copy()
    df_vix_aligned = df_vix_copy.loc[common_index, ['current_price']].copy()

    try:
        df_spy_aligned['Real_Time_SPY_VIX_Correlation'] = memory_friendly_rolling_corr(
            df_spy_aligned['current_price'],
            df_vix_aligned['current_price'],
            window=30
        )
    except Exception as e:
        print("An error occurred during the memory-friendly correlation calculation:")
        print(e)
        return pd.DataFrame()

    final_df = df_spy_aligned[['Real_Time_SPY_VIX_Correlation']].reset_index()
    return final_df

def feature_vix_ema(df: pd.DataFrame, span_list: list) -> pd.DataFrame:
    if 'current_price' not in df.columns:
        logging.error("Missing 'current_price' column in VIX DataFrame. EMA calculation skipped.")
        return df
    for span in span_list:
        ema_column = f'VIX_EMA_{span}'
        df[ema_column] = df['current_price'].ewm(span=span, adjust=False).mean()
        logging.info(f"Calculated {ema_column} for VIX.")
    return df

def feature_vix_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    if 'current_price' not in df.columns:
        logging.error("Missing 'current_price' column in VIX DataFrame. RSI calculation skipped.")
        return df
    delta = df['current_price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['VIX_RSI'] = 100 - (100 / (1 + rs))
    logging.info("Calculated VIX_RSI.")
    return df

def feature_vix_macd(df: pd.DataFrame, span_short: int = 12, span_long: int = 26, span_signal: int = 9) -> pd.DataFrame:
    if 'current_price' not in df.columns:
        logging.error("Missing 'current_price' column in VIX DataFrame. MACD calculation skipped.")
        return df
    ema_short = df['current_price'].ewm(span=span_short, adjust=False).mean()
    ema_long = df['current_price'].ewm(span=span_long, adjust=False).mean()
    df['VIX_MACD'] = ema_short - ema_long
    df['VIX_MACD_Signal'] = df['VIX_MACD'].ewm(span=span_signal, adjust=False).mean()
    logging.info("Calculated VIX_MACD and VIX_MACD_Signal.")
    return df

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
    df = feature_interest_rate_value(df)
    df = feature_interest_rate_lag(df, [1, 3, 12])
    df = feature_interest_rate_rolling_mean(df, [3, 6, 12])
    df = feature_interest_rate_ema(df, [12, 26, 50])
    df = feature_interest_rate_days_since_peak(df)
    df = feature_interest_rate_days_since_trough(df)
    return df

def main_labor_force_features(df):
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

def main_vix_features(df):
    logging.info("[main_vix_features] Processing VIX features.")
    df = feature_vix_ema(df, [12, 26])
    df = feature_vix_rsi(df)
    df = feature_vix_macd(df)
    logging.info("[main_vix_features] Completed VIX feature engineering.")
    return df

def main_real_time_spx_features(df):
    logging.info("[main_real_time_spx_features] Processing SPX real-time features.")
    if 'timestamp' in df.index.names and 'timestamp' not in df.columns:
        logging.info("[main_real_time_spx_features] 'timestamp' in index but not in columns. Resetting index.")
        df.reset_index(inplace=True)
    elif 'timestamp' in df.index.names and 'timestamp' in df.columns:
        logging.info("[main_real_time_spx_features] 'timestamp' is both index and column. Dropping from index.")
        df.reset_index(drop=True, inplace=True)

    if 'timestamp' not in df.columns:
        logging.warning("[main_real_time_spx_features] The 'timestamp' column is missing. Cannot proceed.")
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    if 'current_price' in df.columns:
        df['target_1h'] = df['current_price'].shift(-20)
    else:
        logging.warning("[main_real_time_spx_features] 'current_price' not found, cannot create targets.")

    df = feature_real_time_indicator_lag_1(df)
    df = feature_real_time_indicator_sma_20(df)
    df = feature_real_time_indicator_sma_50(df)
    df = feature_real_time_indicator_ema_12(df)
    df = feature_real_time_indicator_ema_26(df)

    logging.info("[main_real_time_spx_features] Completed feature engineering for SPX real-time.")
    return df

def main_real_time_spy_features(df):
    logging.info("[main_real_time_spy_features] Processing SPY real-time features.")
    if 'timestamp' in df.index.names and 'timestamp' not in df.columns:
        logging.info("[main_real_time_spy_features] 'timestamp' in index but not in columns. Resetting index.")
        df.reset_index(inplace=True)
    elif 'timestamp' in df.index.names and 'timestamp' in df.columns:
        logging.info("[main_real_time_spy_features] 'timestamp' is both index and column. Dropping from index.")
        df.reset_index(drop=True, inplace=True)

    if 'timestamp' not in df.columns:
        logging.warning("[main_real_time_spy_features] The 'timestamp' column is missing. Cannot proceed.")
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    if 'current_price' in df.columns:
        df['target_1h'] = df['current_price'].shift(-20)
    else:
        logging.warning("[main_real_time_spy_features] 'current_price' not found, cannot create targets.")

    df = feature_real_time_indicator_lag_1(df)
    df = feature_real_time_indicator_sma_20(df)
    df = feature_real_time_indicator_sma_50(df)
    df = feature_real_time_indicator_ema_12(df)
    df = feature_real_time_indicator_ema_26(df)

    logging.info("[main_real_time_spy_features] Completed feature engineering for SPY real-time.")
    return df

missing_columns_dict = {}

def require_columns(df, required_cols, dataset_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logging.warning(f"{dataset_name}: Missing required columns {missing}")
        missing_columns_dict[dataset_name] = missing
        return False
    return True

def apply_days_since_peak_or_trough(df, feature_peak_col, feature_days_col, dataset_name):
    datetime_col = 'date' if 'date' in df.columns else 'timestamp' if 'timestamp' in df.columns else None
    if datetime_col is None:
        logging.warning(f"{dataset_name}: No datetime column for {feature_days_col}. Skipping.")
        df[feature_days_col] = np.nan
        return df
    unique_vals = df[feature_peak_col].dropna().unique()
    val_to_index = {}
    for val in unique_vals:
        first_occ_idx = df[df[feature_peak_col] == val].index[0]
        val_to_index[val] = first_occ_idx
    def calc_days_since(row):
        val = row[feature_peak_col]
        if pd.isna(val):
            return np.nan
        if val not in val_to_index:
            return np.nan
        peak_idx = val_to_index[val]
        if peak_idx not in df.index or row.name not in df.index:
            return np.nan
        if pd.isna(df.at[peak_idx, datetime_col]) or pd.isna(df.at[row.name, datetime_col]):
            return np.nan
        delta = df.at[row.name, datetime_col] - df.at[peak_idx, datetime_col]
        return delta.days if pd.notna(delta) else np.nan
    df[feature_days_col] = df.apply(calc_days_since, axis=1)
    return df

def apply_features(df, dataset_name, data_dict):
    if dataset_name == 'real_time_vix':
        logging.info(f"[apply_features - {dataset_name}] Initial columns: {df.columns.tolist()}")
        if 'timestamp' not in df.columns:
            logging.warning(f"[apply_features - {dataset_name}] 'timestamp' missing at the start of apply_features.")
        else:
            logging.info(f"[apply_features - {dataset_name}] 'timestamp' found before feature engineering.")
        if 'timestamp' not in df.columns:
            logging.error(f"[apply_features - {dataset_name}] 'timestamp' must be present. Cannot proceed.")
            return df

        logging.info(f"[apply_features - {dataset_name}] Applying main_vix_features...")
        df = main_vix_features(df)
        logging.info(f"[apply_features - {dataset_name}] After main_vix_features columns: {df.columns.tolist()}")

        if 'timestamp' not in df.columns:
            logging.warning(f"[apply_features - {dataset_name}] 'timestamp' missing after main_vix_features.")
        else:
            logging.info(f"[apply_features - {dataset_name}] 'timestamp' still present after main_vix_features.")

        logging.info(f"[apply_features - {dataset_name}] Final columns at end of function: {df.columns.tolist()}")
        return df

    macro_required = ['value']

    if dataset_name == 'consumer_confidence_data':
        if not require_columns(df, macro_required, dataset_name):
            return df
        df = main_consumer_confidence_features(df)
        df = apply_days_since_peak_or_trough(df,
                                             'Consumer_Confidence_Peak',
                                             'Consumer_Confidence_Days_Since_Peak',
                                             dataset_name)
        df = apply_days_since_peak_or_trough(df,
                                             'Consumer_Confidence_Trough',
                                             'Consumer_Confidence_Days_Since_Trough',
                                             dataset_name)
        return df

    elif dataset_name == 'consumer_sentiment_data':
        if not require_columns(df, macro_required, dataset_name):
            return df
        return main_consumer_sentiment_features(df)

    elif dataset_name == 'core_inflation_data':
        if not require_columns(df, macro_required, dataset_name):
            return df
        return main_core_inflation_features(df)

    elif dataset_name == 'cpi_data':
        if not require_columns(df, macro_required, dataset_name):
            return df
        df = main_cpi_features(df)
        df = apply_days_since_peak_or_trough(df,
                                             'CPI_Peak',
                                             'CPI_Days_Since_Peak',
                                             dataset_name)
        df = apply_days_since_peak_or_trough(df,
                                             'CPI_Trough',
                                             'CPI_Days_Since_Trough',
                                             dataset_name)
        return df

    elif dataset_name == 'gdp_data':
        if not require_columns(df, macro_required, dataset_name):
            return df
        return main_gdp_features(df)

    elif dataset_name == 'industrial_production_data':
        if not require_columns(df, macro_required, dataset_name):
            return df
        df = main_industrial_production_features(df)
        df = apply_days_since_peak_or_trough(df,
                                             'Industrial_Production_Trough',
                                             'Industrial_Production_Days_Since_Trough',
                                             dataset_name)
        return df

    elif dataset_name == 'interest_rate_data':
        if not require_columns(df, macro_required, dataset_name):
            return df
        df = main_interest_rate_features(df)
        df = apply_days_since_peak_or_trough(df,
                                             'Interest_Rate_Peak',
                                             'Interest_Rate_Days_Since_Peak',
                                             dataset_name)
        df = apply_days_since_peak_or_trough(df,
                                             'Interest_Rate_Trough',
                                             'Interest_Rate_Days_Since_Trough',
                                             dataset_name)
        return df

    elif dataset_name == 'labor_force_participation_rate_data':
        if not require_columns(df, macro_required, dataset_name):
            return df
        df = main_labor_force_features(df)
        df = apply_days_since_peak_or_trough(df,
                                             'Labor_Force_Peak',
                                             'Labor_Force_Days_Since_Peak',
                                             dataset_name)
        df = apply_days_since_peak_or_trough(df,
                                             'Labor_Force_Trough',
                                             'Labor_Force_Days_Since_Trough',
                                             dataset_name)
        return df

    elif dataset_name == 'nonfarm_payroll_employment_data':
        if not require_columns(df, macro_required, dataset_name):
            return df
        df = main_nonfarm_features(df)
        df = apply_days_since_peak_or_trough(df,
                                             'NonFarm_Trough',
                                             'NonFarm_Days_Since_Trough',
                                             dataset_name)
        return df

    elif dataset_name == 'personal_consumption_expenditures':
        if not require_columns(df, macro_required, dataset_name):
            return df
        df = main_pce_features(df)
        df = apply_days_since_peak_or_trough(df,
                                             'PCE_Peak',
                                             'PCE_Days_Since_Peak',
                                             dataset_name)
        return df

    elif dataset_name == 'ppi_data':
        if not require_columns(df, macro_required, dataset_name):
            return df
        df = main_ppi_features(df)
        df = apply_days_since_peak_or_trough(df,
                                             'PPI_Trough',
                                             'PPI_Days_Since_Trough',
                                             dataset_name)
        return df

    elif dataset_name == 'unemployment_rate_data':
        if not require_columns(df, macro_required, dataset_name):
            return df
        df = main_unemployment_rate_features(df)
        df = apply_days_since_peak_or_trough(df,
                                             'Unemployment_Rate_Trough',
                                             'Unemployment_Rate_Days_Since_Trough',
                                             dataset_name)
        return df

    elif dataset_name == 'historical_indicator_data':
        if not require_columns(df, macro_required, dataset_name):
            return df
        return main_historical_indicator_features(df)

    elif dataset_name == 'spx_market_data':
        if not require_columns(df, ['close'], dataset_name):
            return df
        return main_spx_market_features(df)

    elif dataset_name == 'spy_market_data':
        if not require_columns(df, ['close'], dataset_name):
            return df
        return main_spy_market_features(df)

    elif dataset_name in ['historical_spx', 'historical_spy']:
        if dataset_name == 'historical_spx':
            if not require_columns(df, ['close'], dataset_name):
                return df
            df = main_spx_market_features(df)
            correlation_function = feature_real_time_spx_vix_correlation
        else:
            if not require_columns(df, ['close'], dataset_name):
                return df
            df = main_spy_market_features(df)
            correlation_function = feature_real_time_spy_vix_correlation

        vix_data = data_dict.get('historical_vix', None)
        if vix_data is not None and not vix_data.empty:
            corr_features = correlation_function(df, vix_data)
            if 'timestamp' in corr_features.columns:
                df = pd.merge(df, corr_features, on='timestamp', how='left')
        return df
    
    elif dataset_name in ['real_time_spx', 'real_time_spy']:
        if dataset_name == 'real_time_spx':
            if not require_columns(df, ['current_price'], dataset_name):
                return df
            df = main_real_time_spx_features(df)
            correlation_function = feature_real_time_spx_vix_correlation
        else:
            if not require_columns(df, ['current_price'], dataset_name):
                return df
            df = main_real_time_spy_features(df)
            correlation_function = feature_real_time_spy_vix_correlation

        vix_data = data_dict.get('real_time_vix', None)
        if vix_data is not None and not vix_data.empty:
            corr_features = correlation_function(df, vix_data)
            if 'timestamp' in corr_features.columns:
                df = pd.merge(df, corr_features, on='timestamp', how='left')
        return df

    elif dataset_name in ['historical_vix', 'real_time_vix']:
        if 'timestamp' not in df.columns:
            return df
        df = main_vix_features(df)
        return df

    else:
        return df
    
def create_XY_sequences(df, seq_len=60, single_horizon_target=None):
    """
    Create sequences for both X and Y arrays from a given DataFrame.

    1) If df has all 5 multi-horizon columns: [target_1h, target_3h, target_6h, target_1d, target_3d],
       we create Y as a 5-dimensional output (shape: (num_samples, 5)).

    2) Otherwise, if 'single_horizon_target' is provided and present in df, we produce Y with shape (num_samples, 1).

    3) If neither multi-horizon columns nor a single_horizon_target is found, we produce only X (no Y).
    """
    import logging
    horizon_cols = ['target_1h', 'target_3h', 'target_6h', 'target_1d', 'target_3d']
    available_horizons = [col for col in horizon_cols if col in df.columns]

    # 1) Multi-horizon case
    if len(available_horizons) == 5:
        logging.info("Creating multi-horizon Y with shape=(...,5).")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for hc in horizon_cols:
            if hc in numeric_cols:
                numeric_cols.remove(hc)
        X_array = df[numeric_cols].values
        y_array = df[horizon_cols].values  # shape => (N, 5)

    # 2) Single-horizon fallback
    elif single_horizon_target and single_horizon_target in df.columns:
        logging.info(f"Creating single-horizon Y using {single_horizon_target}.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if single_horizon_target in numeric_cols:
            numeric_cols.remove(single_horizon_target)
        X_array = df[numeric_cols].values
        y_array = df[single_horizon_target].values.reshape(-1, 1)

    # 3) No valid target => only X
    else:
        logging.warning("No multi-horizon or single-horizon target found; only X will be returned.")
        X_array = df.select_dtypes(include=[np.number]).values
        y_array = None

    # If not enough data for a single sequence:
    if len(X_array) < seq_len:
        logging.warning("Not enough rows to form even one sequence.")
        return None, None

    # Build sequences
    X_seq, y_seq = [], []
    for i in range(len(X_array) - seq_len):
        X_seq.append(X_array[i : i + seq_len])
        if y_array is not None:
            y_seq.append(y_array[i + seq_len])  # offset

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32) if y_array is not None else None
    return X_seq, y_seq


def save_sequences_in_parts(X, Y, prefix, chunk_size=10000):
    """
    Saves 3D X and Y arrays in multiple smaller .npy chunks, optionally uploading them to S3.
    If Y is None, only X parts are saved/uploaded.

    :param X: 3D NumPy array, shape [samples, sequence_length, features]
    :param Y: 2D or 3D NumPy array, or None if no target, shape [samples, ...]
    :param prefix: Filename prefix (e.g. 'real_time_spx')
    :param chunk_size: Maximum number of sequences to store in each .npy file
    """
    ensure_directory_exists(SEQUENCES_DIR)  # Ensure local subfolder is created

    # 1) Save X in parts
    num_chunks_x = (X.shape[0] + chunk_size - 1) // chunk_size
    for i in range(num_chunks_x):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, X.shape[0])

        X_part = X[start:end]
        x_filename = f"{prefix}_X_3D_part{i}.npy"
        x_filepath = os.path.join(SEQUENCES_DIR, x_filename)

        # Save locally
        np.save(x_filepath, X_part)
        logging.info(f"Saved X chunk to {x_filepath}")

        # Auto-upload to S3 if available
        if callable(globals().get("auto_upload_file_to_s3")):
            auto_upload_file_to_s3(
                local_path=x_filepath,
                s3_subfolder="data/lumen2/featured/sequences"
            )

    # 2) If Y is provided, save it in parts as well
    if Y is not None:
        if len(Y) != len(X):
            logging.warning("X and Y lengths differ. Verify alignment before saving.")
        num_chunks_y = (Y.shape[0] + chunk_size - 1) // chunk_size
        for j in range(num_chunks_y):
            start = j * chunk_size
            end = min((j + 1) * chunk_size, Y.shape[0])

            Y_part = Y[start:end]
            y_filename = f"{prefix}_Y_3D_part{j}.npy"
            y_filepath = os.path.join(SEQUENCES_DIR, y_filename)

            # Save locally
            np.save(y_filepath, Y_part)
            logging.info(f"Saved Y chunk to {y_filepath}")

            # Auto-upload to S3 if available
            if callable(globals().get("auto_upload_file_to_s3")):
                auto_upload_file_to_s3(
                    local_path=y_filepath,
                    s3_subfolder="data/lumen2/featured/sequences"
                )

def scale_all_features(df, dataset_name, target_column=None):
    datetime_columns_local = [col for col in ["timestamp", "date"] if col in df.columns]
    multi_target_cols = [col for col in df.columns if col.startswith("target_")]
    target_columns_local = []

    if target_column and target_column in df.columns:
        target_columns_local.append(target_column)

    target_columns_local = list(set(target_columns_local + multi_target_cols))

    # For real_time data, drop columns that are entirely NaN
    if dataset_name in ["real_time_spx", "real_time_spy", "real_time_vix"]:
        df.dropna(axis=1, how="all", inplace=True)

    # Separate out date/time columns and any target columns
    datetime_df = df[datetime_columns_local] if datetime_columns_local else pd.DataFrame()
    target_df = df[target_columns_local] if target_columns_local else pd.DataFrame()

    # The "feature" columns are all numeric columns excluding date/time + targets
    feature_columns = [c for c in df.columns if c not in datetime_columns_local + target_columns_local]
    feature_df = df[feature_columns].apply(pd.to_numeric, errors="coerce")

    # Downcast numeric columns to save memory
    for col in feature_df.columns:
        if pd.api.types.is_float_dtype(feature_df[col]):
            feature_df[col] = pd.to_numeric(feature_df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(feature_df[col]):
            feature_df[col] = pd.to_numeric(feature_df[col], downcast="integer")

    # If there are no features, return early
    if feature_df.empty:
        if not (datetime_df.empty and target_df.empty):
            return pd.concat([datetime_df, target_df], axis=1)
        return df

    # Replace inf with NaN, then fill NaNs with column mean
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_df.fillna(feature_df.mean(numeric_only=True), inplace=True)

    # If still no valid numeric columns, skip
    if feature_df.empty:
        scaled_feature_df = feature_df
    else:
        # --- Here we actually fit/transform with MinMaxScaler ---
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        try:
            scaled_vals = scaler.fit_transform(feature_df)
        except ValueError:
            # If something goes wrong with fitting, at least return unscaled data
            if not (datetime_df.empty and target_df.empty):
                return pd.concat([datetime_df, target_df], axis=1)
            return df

        # Create DataFrame of scaled features
        scaled_feature_df = pd.DataFrame(
            scaled_vals.astype("float32"),
            columns=feature_df.columns,
            index=feature_df.index
        )

        # Save the scaler to local disk
        import joblib
        scalers_dir = os.path.join(BASE_DIR, "models", "lumen_2", "scalers")
        os.makedirs(scalers_dir, exist_ok=True)
        scaler_filename = f"{dataset_name}_scaler.joblib"
        scaler_path = os.path.join(scalers_dir, scaler_filename)

        joblib.dump(scaler, scaler_path)
        logging.info(f"Saved scaler for '{dataset_name}' to {scaler_path}")

        if callable(globals().get("auto_upload_file_to_s3")):
            try:
                auto_upload_file_to_s3(
                    local_path=scaler_path,
                    s3_subfolder="models/lumen_2/scalers"
                )
            except Exception as upload_err:
                logging.warning(f"Could not upload scaler to S3: {upload_err}")

    # Re-combine any datetime columns + scaled features + target columns
    parts = []
    if not datetime_df.empty:
        datetime_df.reset_index(drop=True, inplace=True)
        parts.append(datetime_df)

    scaled_feature_df.reset_index(drop=True, inplace=True)
    parts.append(scaled_feature_df)

    if not target_df.empty:
        target_df.reset_index(drop=True, inplace=True)
        parts.append(target_df)

    return pd.concat(parts, axis=1)


def save_enhanced_data(df, filename):
    enhanced_filename = f"featured_{filename}"
    filepath = os.path.join(FEATURED_DIR, enhanced_filename)
    df.to_csv(filepath, index=False)
    logging.info(f"Saved enhanced data to {filepath}")

    # Now auto-upload to S3, if the function is available
    if callable(globals().get("auto_upload_file_to_s3")):
        auto_upload_file_to_s3(
            local_path=filepath,
            s3_subfolder="data/lumen2/featured"
        )

def extract_and_save_feature_list(data_dict: dict, output_file: str, datetime_columns: list, target_columns: dict):
    feature_list = []
    for dataset_name, df in data_dict.items():
        target_column = target_columns.get(dataset_name, None)
        exclude_cols = datetime_columns.copy()
        if target_column:
            exclude_cols.append(target_column)
        features = [col for col in df.columns if col not in exclude_cols]
        for idx, feature in enumerate(features, start=1):
            feature_list.append({
                'Dataset': dataset_name,
                'Feature_Number': idx,
                'Feature_Name': feature
            })
    feature_df = pd.DataFrame(feature_list)
    feature_df.to_csv(output_file, index=False)
    logging.info(f"Feature list saved to {output_file}")

    summary = feature_df.groupby('Dataset').agg(
        Total_Features=('Feature_Name', 'count'),
        Feature_Names=('Feature_Name',
                       lambda x: ', '.join(x[:5]) + ('...' if len(x) > 5 else ''))
    ).reset_index()

    logging.info("=== Feature Summary ===")
    for _, row in summary.iterrows():
        logging.info(f"Dataset '{row['Dataset']}': {row['Total_Features']} features.")
        logging.info(f"Sample Features: {row['Feature_Names']}")
    logging.info("========================")


datetime_columns = ['timestamp', 'date']
target_columns = {
    'historical_spx': 'close',
    'historical_spy': 'close',
    'real_time_spx': 'current_price',
    'real_time_spy': 'current_price',
    'unemployment_rate_data': 'unemployment_rate',
}


def create_sequences(data, seq_len=60):
    sequences = []
    for i in range(len(data) - seq_len + 1):
        seq = data[i : i + seq_len]
        sequences.append(seq)
    return np.array(sequences)


def create_and_save_sequences_in_chunks(df, data_name, seq_len, target_column, chunk_size=10000):
    """
    Create sequences for X and Y arrays in small increments rather than all at once,
    preventing massive memory usage. Saves each chunk locally, then uploads
    to S3 if 'auto_upload_file_to_s3' is available.

    :param df: A DataFrame with numeric columns.
    :param data_name: A string identifier, e.g. "real_time_spx".
    :param seq_len: Sequence length (e.g. 60).
    :param target_column: The name of the column to predict, or None.
    :param chunk_size: Max number of sequences per chunk file.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    has_target = target_column and (target_column in df.columns)

    # Separate features vs. target
    if has_target:
        numeric_cols.remove(target_column)
        X_array = df[numeric_cols].values
        y_array = df[target_column].values
    else:
        X_array = df[numeric_cols].values
        y_array = None

    total_length = len(X_array)
    if total_length < seq_len:
        logging.warning(
            f"[{data_name}] Not enough data to create sequences of length {seq_len}. Skipping."
        )
        return

    start = 0
    x_part_count = 0
    y_part_count = 0

    while start + seq_len <= total_length:
        end = min(start + chunk_size, total_length - seq_len + 1)

        X_seq_list = []
        Y_seq_list = []

        # Build partial sequences
        for i in range(start, end):
            X_seq_list.append(X_array[i : i + seq_len])
            if has_target:
                # Example: take the last item in the 60-length window
                Y_seq_list.append(y_array[i + seq_len - 1])

        # Convert to NumPy arrays
        X_seq_arr = np.array(X_seq_list, dtype=np.float32)
        Y_seq_arr = np.array(Y_seq_list, dtype=np.float32) if has_target else None

        # Save local X chunk
        X_part_filename = os.path.join(SEQUENCES_DIR, f"{data_name}_X_3D_part{x_part_count}.npy")
        np.save(X_part_filename, X_seq_arr)
        logging.info(f"Saved X chunk to {X_part_filename}")

        # Attempt auto-upload to S3
        if callable(globals().get("auto_upload_file_to_s3")):
            auto_upload_file_to_s3(
                local_path=X_part_filename,
                s3_subfolder="data/lumen2/featured/sequences"
            )

        x_part_count += 1

        # Save local Y chunk if applicable
        if has_target:
            Y_part_filename = os.path.join(SEQUENCES_DIR, f"{data_name}_Y_3D_part{y_part_count}.npy")
            np.save(Y_part_filename, Y_seq_arr)
            logging.info(f"Saved Y chunk to {Y_part_filename}")

            # Attempt auto-upload to S3
            if callable(globals().get("auto_upload_file_to_s3")):
                auto_upload_file_to_s3(
                    local_path=Y_part_filename,
                    s3_subfolder="data/lumen2/featured/sequences"
                )

            y_part_count += 1

        start = end

def main():
    ensure_directory_exists(FEATURED_DIR)
    ensure_directory_exists(SEQUENCES_DIR)
    logging.info("[main] Starting feature engineering process.")

    data_dict = load_data()  # Your custom load_data function

    for data_name, df in data_dict.items():
        logging.info(f"[main] Processing {data_name}...")

        # Convert existing timestamp/date
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            logging.info(f"[main] 'timestamp' column converted for {data_name}.")
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            logging.info(f"[main] 'date' column converted for {data_name}.")
        else:
            logging.warning(f"[main] No 'timestamp' or 'date' column in {data_name}.")

        # Apply cleaning + features
        try:
            df = apply_features(df, data_name, data_dict)
        except Exception as e:
            logging.error(f"[main] Error applying features for {data_name}: {e}", exc_info=True)
            continue

        # If real_time_spx or real_time_spy, we do special target creation & then drop NaNs
        if data_name in ['real_time_spx', 'real_time_spy']:
            # After shift, any row without a future target is NaN → drop them
            if 'target_1h' in df.columns:
                before_drop = len(df)
                df.dropna(subset=['target_1h'], inplace=True)
                after_drop = len(df)
                dropped = before_drop - after_drop
                logging.info(f"[main] {data_name} => dropped {dropped} rows lacking valid target_1h.")

        # Scale
        target_column = target_columns.get(data_name, None)
        try:
            df = scale_all_features(df, data_name, target_column)
        except Exception as e:
            logging.error(f"[main] Error scaling features for {data_name}: {e}", exc_info=True)
            continue

        if df.empty:
            logging.warning(f"[main] {data_name} is empty after scaling. Skipping saving.")
            continue

        # Check NaNs
        df_numeric = df.select_dtypes(include=[np.number])
        nan_counts = df_numeric.isna().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            logging.warning(f"[main] {data_name} contains {total_nans} NaNs across numeric columns!")
            nan_columns = nan_counts[nan_counts > 0]
            logging.warning(f"NaN details for {data_name}:\n{nan_columns.to_string()}")

        logging.info(f"[main] {data_name} final shape: {df.shape}")

        # Save enhanced CSV
        save_enhanced_data(df, f'{data_name}.csv')

        # Optionally build sequences
        create_and_save_sequences_in_chunks(
            df,
            data_name,
            seq_len=60,
            target_column=target_column,
            chunk_size=10000
        )


    # Extract and save the full feature list
    output_file = os.path.join(BASE_DIR, 'feature_list.csv')
    extract_and_save_feature_list(data_dict, output_file, datetime_columns, target_columns)

    # If any columns were missing across datasets, show a summary
    if missing_columns_dict:
        logging.info("=== Missing Columns Summary ===")
        for d_name, cols in missing_columns_dict.items():
            logging.info(f"{d_name} is missing: {cols}")
        logging.info("==============================")

    logging.info("[main] Feature engineering process completed.")


if __name__ == "__main__":
    ensure_directory_exists(FEATURED_DIR)
    main()