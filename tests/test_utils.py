import pytest
import numpy as np
import random
from sampling import utils

def test_set_seed():
    utils.set_seed(42)
    assert np.random.randint(0, 100) == 51  # Consistent result with seed 42
    assert random.randint(0, 100) == 81  # Consistent result with seed 42

def test_validate_dataframe():
    import pandas as pd
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    # This should pass without exception
    utils.validate_dataframe(data, required_columns=['col1', 'col2'])

    with pytest.raises(ValueError):
        # This should raise an error because 'col3' is missing
        utils.validate_dataframe(data, required_columns=['col1', 'col3'])
