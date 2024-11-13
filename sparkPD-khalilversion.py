import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Function to fill missing string values in a DataFrame column
def patch_empty_strings(df, column, fill_value='Unavailable'):
    col_data = df[column].copy()
    while True:
        if fill_value in np.unique(col_data.dropna().unique().to_numpy()):
            print(f'The value ({fill_value}) is already in "{column}"')
            user_decision = input('Replace anyway? [Y/n]: ')
            if user_decision.lower() == 'y':
                col_data = col_data.fillna(fill_value)
                return col_data
            elif user_decision.lower() == 'n':
                new_fill_value = input('Enter a new string to use: ')
                if new_fill_value not in np.unique(col_data.dropna().unique().to_numpy()):
                    col_data = col_data.fillna(new_fill_value)
                    return col_data
            else:
                print('Invalid input. Please enter Y or n.')
        else:
            col_data = col_data.fillna(fill_value)
            return col_data

# Function to fill missing numeric values in a DataFrame column
def impute_numeric_values(df, column, strategy='mode'):
    if strategy not in ['mode', 'median', 'mean']:
        print('Invalid imputation strategy. Choose "mode", "median", or "mean".')
        return
    if strategy == 'mode':
        mode_value = df[column].mode()[0] if not df[column].mode().empty else 0
        return df[column].fillna(mode_value)
    elif strategy == 'median':
        return df[column].fillna(df[column].median())
    elif strategy == 'mean':
        return df[column].fillna(df[column].mean())

# Function to apply string and numeric imputation across all DataFrame columns
def impute_dataframe(df, string_fill='Unavailable', num_strategy='mode'):
    result_df = df.copy()
    for col in result_df.columns:
        if result_df[col].isnull().any():
            if result_df[col].dtype == 'object':
                result_df[col] = patch_empty_strings(result_df, col, string_fill)
            else:
                result_df[col] = impute_numeric_values(result_df, col, num_strategy)
    return result_df

# Function to encode string column values to numerical identifiers
def encode_categorical_column(df, column):
    col_data = df[column].fillna('Unknown').astype(str)
    unique_values = sorted(col_data.unique())
    encoding_map = {val: idx for idx, val in enumerate(unique_values)}
    return col_data.map(encoding_map), encoding_map

# Function to encode all categorical columns in a DataFrame
def encode_categorical_dataframe(df):
    encoded_df = df.copy()
    encoding_maps = {}
    for col in encoded_df.columns:
        if encoded_df[col].dtype == 'object':
            encoded_df[col], encoding_maps[col] = encode_categorical_column(encoded_df, col)
    return encoded_df, encoding_maps
