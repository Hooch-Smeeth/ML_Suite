import requests
import pandas as pd
import pickle
import datetime
import logging
import os
from time import sleep

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_klines(pair, interval, start_time, end_time):
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': pair,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        klines = response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f'Error fetching klines: {e}')
        return None
    return klines

def save_data(df, filename):
    with open(filename, 'wb') as file:
        pickle.dump(df, file)
    logging.info(f'Data saved to {filename}')

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def main():
    """
    Fetches kline (candlestick) data from the Binance API for a specified currency pair and interval.

    Args:
        pair (str): The currency pair to fetch (e.g., 'BTCUSDT').
        interval (str): The interval for klines (e.g., '1m').
        start_time (int): The start time for data in milliseconds since epoch.
        end_time (int): The end time for data in milliseconds since epoch.

    Returns:
        list: A list of klines, each represented as a list of values.
    """
    pair = 'BTCUSDT'
    interval = '1m'
    filename = f'{pair}_{interval}_Binance.pkl'

    if os.path.exists(filename):
        logging.info('Data file exists, loading data...')
        df = load_data(filename)
        last_timestamp = int(df.iloc[-1]['Open time'].timestamp() * 1000)
        start_time = last_timestamp + 60000
    else:
        logging.info('Data file does not exist, downloading all available historical data...')
        start_time = int(datetime.datetime(2021, 1, 1).timestamp() * 1000)

    end_time = int(datetime.datetime.now().timestamp() * 1000)

    all_klines = []

    while start_time < end_time:
        klines = fetch_klines(pair, interval, start_time, end_time)
        if klines is None:
            logging.error('Failed to fetch klines, exiting.')
            return

        if len(klines) == 0:
            break

        all_klines.extend(klines)
        start_time = klines[-1][0] + 60000
        sleep(0.2)  # adhere to Binance's API rate limit requirements

    if len(all_klines) > 0:
        new_df = pd.DataFrame(all_klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                                   'Close time', 'Quote asset volume', 'Number of trades',
                                                   'Taker buy base asset volume', 'Taker buy quote asset volume',
                                                   'Ignore'])
        new_df['Open time'] = pd.to_datetime(new_df['Open time'], unit='ms')
        new_df['Close time'] = pd.to_datetime(new_df['Close time'], unit='ms')

        if os.path.exists(filename):
            df = df.append(new_df, ignore_index=True)
        else:
            df = new_df

        save_data(df, filename)

    print(df[['Open time', 'Open', 'High', 'Low', 'Close']].head())
    print(df[['Open time', 'Open', 'High', 'Low', 'Close']].tail())

if __name__ == '__main__':
    main()

