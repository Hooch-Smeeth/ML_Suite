import numpy as np
import pandas as pd

def add_sinusoidal_features(df, datetime_col, periods, time_resolution) -> pd.DataFrame:
    """
    Adds sinusoidal features to a DataFrame based on specified time periods and time resolution.

    Args:
        df (pd.DataFrame): The DataFrame to be modified.
        datetime_col (str): The name of the column containing datetime data.
        periods (List[str]): List of periods to create sinusoidal features for.
                             Valid periods: ['1hr', '4hr', '1day', '1week', '1month', '1year']
        time_resolution (int): Time resolution in seconds of the input data (e.g., 60 for 1 minute, 3600 for 1 hour).

    Returns:
        pd.DataFrame: The DataFrame with added sinusoidal features.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected 'df' to be a pandas DataFrame.")

    if not isinstance(datetime_col, str):
        raise TypeError("Expected 'datetime_col' to be a string.")

    if not isinstance(periods, list) or not all(isinstance(period, str) for period in periods):
        raise TypeError("Expected 'periods' to be a list of strings.")

    if not isinstance(time_resolution, int):
        raise TypeError("Expected 'time_resolution' to be an integer.")

    # Define the periods in seconds
    period_values = {
        '1hr': 3600,
        '4hr': 14400,
        '1day': 86400,
        '1week': 604800,
        '1month': 2592000,  # Approx. 30 days
        '1year': 31536000   # Approx. 365 days
    }

    # Temporary variable for seconds since start
    _seconds_since_start = (pd.to_datetime(df[datetime_col]) - pd.to_datetime(df[datetime_col]).iloc[0]).dt.total_seconds()

    # Add sinusoidal columns for each specified period
    for period in periods:
        if period in period_values:
            period_in_seconds = period_values[period]
            col_name = f'sinusoid_{period}'
            df[col_name] = np.sin(2 * np.pi * _seconds_since_start / period_in_seconds)
        else:
            print(f"Period '{period}' is not valid. Skipping.")

    return df
