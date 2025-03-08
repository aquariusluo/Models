import pandas as pd
import requests
import numpy as np  # Import numpy for floor function

# Function to fetch OHLC data from Binance
def fetch_binance_ohlc(symbol, interval, start_time, end_time):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000
    }
    all_data = []
    while True:
        response = requests.get(url, params=params)
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        params["startTime"] = data[-1][0] + 1  # Increment to fetch next set
        if len(data) < 1000:
            break
    return all_data

# Process Binance data into a DataFrame
def process_binance_data(raw_data):
    columns = ["timestamp", "open", "high", "low", "close", "volume", "close_time",
               "quote_asset_volume", "number_of_trades", "taker_buy_base_volume", 
               "taker_buy_quote_volume", "ignore"]
    df = pd.DataFrame(raw_data, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["open"] = pd.to_numeric(df["open"])
    df["high"] = pd.to_numeric(df["high"])
    df["low"] = pd.to_numeric(df["low"])
    df["close"] = pd.to_numeric(df["close"])
    return df[["timestamp", "open", "high", "low", "close"]]

# Generate lagged features for OHLC columns
def generate_lagged_features(df, asset):
    for lag in range(1, 11):  # Generate lag1 to lag10
        for col in ["open", "high", "low", "close"]:
            df[f"{col}_{asset}_lag{lag}"] = df[col].shift(lag)
    return df

# Calculate 1-hour return for ETHUSDT
def calculate_target(df):
    df["target_ETHUSDT"] = (df["close_ETHUSDT"].shift(-1) - df["close_ETHUSDT"]) / df["close_ETHUSDT"]
    return df

# Fetch and process data for ETHUSDT and BTCUSDT
def fetch_and_process_data(symbol, interval, start_date, end_date):
    start_time_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_time_ms = int(pd.Timestamp(end_date).timestamp() * 1000)
    raw_data = fetch_binance_ohlc(symbol, interval, start_time_ms, end_time_ms)
    df = process_binance_data(raw_data)
    return df

# Define parameters
interval = "1h"
start_date = "2024-11-30T23:00:00"
end_date = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")

# Fetch data for ETHUSDT and BTCUSDT
eth_df = fetch_and_process_data("ETHUSDT", interval, start_date, end_date)
btc_df = fetch_and_process_data("BTCUSDT", interval, start_date, end_date)

# Generate lagged features for both datasets
eth_df = generate_lagged_features(eth_df, "ETHUSDT")
btc_df = generate_lagged_features(btc_df, "BTCUSDT")

# Merge ETHUSDT and BTCUSDT data on timestamp
merged_df = pd.merge(eth_df, btc_df, on="timestamp", suffixes=("_ETHUSDT", "_BTCUSDT"))

# Add hour_of_day feature (rounded down to one decimal place)
merged_df["hour_of_day"] = np.floor(merged_df["timestamp"].dt.hour * 10) / 10

# Calculate the target (1-hour return for ETHUSDT)
merged_df = calculate_target(merged_df)

# Drop rows with null values (caused by lagging)
merged_df = merged_df.dropna()

# Remove original OHLC columns for both ETHUSDT and BTCUSDT
columns_to_remove = [
    "open_ETHUSDT", "high_ETHUSDT", "low_ETHUSDT", "close_ETHUSDT",
    "open_BTCUSDT", "high_BTCUSDT", "low_BTCUSDT", "close_BTCUSDT"
]
merged_df = merged_df.drop(columns=columns_to_remove)

# Move timestamp to the last column
timestamp_col = merged_df.pop("timestamp")
merged_df["timestamp"] = timestamp_col  # Add it back at the end

# Save the dataset with timestamp at the end
output_file_with_timestamp = "../data/ETHUSDT_BTCUSDT_cleaned_no_ohlc_hour_decimal.csv"
merged_df.to_csv(output_file_with_timestamp, index=False)
print(f"✅ Dataset saved as {output_file_with_timestamp} (timestamp moved to end)")

# ✅ Merge `ETHUSDT_1h_spot_forecast_training_timestamp.csv` with `ETHUSDT_BTCUSDT_cleaned_no_ohlc_hour_decimal.csv`
old_data_path_timestamp = "../data/ETHUSDT_1h_spot_forecast_training_timestamp.csv"
new_data_path_timestamp = "../data/ETHUSDT_BTCUSDT_cleaned_no_ohlc_hour_decimal.csv"
merged_output_path_timestamp = "../data/ETHUSDT_1h_spot_forecast_training_new_timestamp.csv"

# Load both datasets
df_old_timestamp = pd.read_csv(old_data_path_timestamp)
df_new_timestamp = pd.read_csv(new_data_path_timestamp)

# Merge both datasets
df_combined_timestamp = pd.concat([df_old_timestamp, df_new_timestamp], ignore_index=True)

# Remove duplicate rows if any
df_combined_timestamp.drop_duplicates(inplace=True)

# Save the merged dataset with timestamp
df_combined_timestamp.to_csv(merged_output_path_timestamp, index=False)
print(f"✅ Merged dataset saved as {merged_output_path_timestamp} (with timestamp)")

# ✅ Generate `ETHUSDT_1h_spot_forecast_training_new.csv` (without timestamp)
df_combined_no_timestamp = df_combined_timestamp.drop(columns=["timestamp"])
merged_output_path_no_timestamp = "../data/ETHUSDT_1h_spot_forecast_training_new.csv"

df_combined_no_timestamp.to_csv(merged_output_path_no_timestamp, index=False)
print(f"✅ Merged dataset saved as {merged_output_path_no_timestamp} (without timestamp)")
