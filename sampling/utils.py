import numpy as np
import random
import logging
import pandas as pd
from typing import Optional, List

def set_seed(seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_dataframe(data: pd.DataFrame, required_columns: Optional[List[str]] = None):
    if data.empty:
        logging.error("The dataset is empty.")
        raise ValueError("The dataset is empty.")
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"The dataset is missing required columns: {', '.join(missing_columns)}")
            raise ValueError(f"The dataset is missing required columns: {', '.join(missing_columns)}")
