import pandas as pd
import numpy as np

def load_and_preprocess_target(data_filename: str, target_column_name: str) -> pd.DataFrame:
    """Loads data and preprocesses the specified target column."""
    try:
        data = pd.read_csv(data_filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Data file '{data_filename}' not found.")

    if target_column_name in data.columns:
        # Convert target to numeric, handling errors
        data[target_column_name] = pd.to_numeric(data[target_column_name], errors='coerce')
        # Drop rows where target became NaN
        initial_rows = len(data)
        data.dropna(subset=[target_column_name], inplace=True)
        if len(data) < initial_rows:
            print(f"Warning: Dropped {initial_rows - len(data)} rows due to non-numeric target values.")

        # Convert target to binary (0 or 1)
        data[target_column_name] = np.where(data[target_column_name] > 0, 1, 0).astype(int)
        print(f"Target column '{target_column_name}' processed successfully.")
        return data
    else:
        raise ValueError(f"Error: Target column '{target_column_name}' not found in '{data_filename}'.")

