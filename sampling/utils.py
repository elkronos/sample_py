import numpy as np
import random
import logging
import pandas as pd
from typing import Optional, List

def set_seed(seed: Optional[int] = None) -> None:
    """
    Set the random seed for both NumPy and Python's random module.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

def get_random_state(seed: Optional[int] = None) -> np.random.RandomState:
    """
    Return a NumPy RandomState initialized with the given seed.
    If seed is None, returns a RandomState based on system randomness.
    """
    return np.random.RandomState(seed)

def setup_logging(level=logging.INFO) -> None:
    """
    Set up logging with the specified level and a standard format.
    """
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_dataframe(data: pd.DataFrame, required_columns: Optional[List[str]] = None) -> None:
    """
    Validate that a DataFrame is not empty and (optionally) contains required columns.
    
    Raises:
        ValueError: If the DataFrame is empty or if any required column is missing.
    """
    if data.empty:
        logging.error("The dataset is empty.")
        raise ValueError("The dataset is empty.")
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"The dataset is missing required columns: {', '.join(missing_columns)}")
            raise ValueError(f"The dataset is missing required columns: {', '.join(missing_columns)}")

def apply_max_rows(df: pd.DataFrame, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Return the first `max_rows` rows of the DataFrame if max_rows is provided.
    """
    if max_rows is not None and max_rows > 0:
        return df.iloc[:max_rows]
    return df
