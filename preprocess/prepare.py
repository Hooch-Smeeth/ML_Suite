from typing import Any, List, Tuple, Union
import numpy as np
import pandas as pd
import os


def file_extension(filepath) -> str:
    # Extract and return the file extension from a file path as a string
    _, extension = os.path.splitext(filepath)
    return extension.lower()


def load_dataframe(filepath, start_idx=0, end_idx=None) -> pd.DataFrame:
  """
  Load data from various file types into a pandas DataFrame.
  """
  if not os.path.exists(filepath):
    raise FileNotFoundError(f"The file {filepath} does not exist.")
        
  match file_extension(filepath):
    case '.pkl':
        df = pd.read_pickle(filepath)
        df = df[start_idx:end_idx] if end_idx else df[start_idx:]
        return df

    case '.jpg' | '.jpeg' | '.png' | '.bmp':
        # Process image file
        # Load image data here
        pass

    case '.csv':
        df = pd.read_csv(filepath)
        df = df[start_idx:end_idx] if end_idx else df[start_idx:]
        return df

    case _:
        raise ValueError(f"Unsupported file type: {filepath}")


def dataframe_columns_astype(df, target_type, columns=None) -> pd.DataFrame:
    """
    Converts specified columns of a DataFrame to a given data type. If no columns are
    specified, all columns are converted.

    Args:
        df (pd.DataFrame): The DataFrame to be modified.
        target_type (type): The target data type to convert the columns to.
        columns (List[str], optional): List of column names to convert. If None, all columns are converted.
    """
    if columns is None:
        columns = df.columns

    for col in columns:
        if col in df.columns:
            try:
                first_non_null_element = df[col].dropna().iloc[0] if not df[col].isnull().all() else None
                if not isinstance(first_non_null_element, target_type):
                    df[col] = df[col].astype(target_type)
            except Exception as e:
                print(f"Error converting column {col}: {e}")
        else:
            print(f"Column {col} not found in DataFrame.")

    return df


def normalize_columns(df, columns) -> pd.DataFrame:
    """
    Normalizes specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to be normalized.
        columns (List[str]): List of column names to normalize.

    Returns:
        pd.DataFrame: The DataFrame with specified columns normalized.
    """
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")

    try:
        column_means = df[columns].mean()
        column_stds = df[columns].std()
        df[columns] = (df[columns] - column_means) / column_stds
    except Exception as e:
        raise ValueError(f"Error during normalization: {e}")

    return df