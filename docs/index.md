# Sample_Py Documentation

## Overview

Sample_Py is a Python project that provides various sampling methods for data analysis. These methods include:

- Simple Random Sampling
- Stratified Sampling
- Systematic Sampling
- Cluster Sampling
- Multi-Stage Sampling
- Weighted Sampling
- Reservoir Sampling
- Bootstrap Sampling
- Temporal Sampling
- Spatial Sampling

## Installation

Clone the repository and install the dependencies using `pip`:

```bash
git clone https://github.com/elkronos/sample_py.git
cd sample_py
pip install -r requirements.txt
```

## Usage

Below are detailed descriptions and examples of the sampling functions provided in this project.

### Simple Random Sampling

**Function**: `simple_random_sampling`

Performs simple random sampling on a given DataFrame.

**Parameters**:
- `data` (pd.DataFrame): The input DataFrame.
- `sample_size` (int): Number of samples to draw.
- `max_rows` (Optional[int]): Maximum number of rows to return. Defaults to None.
- `replace` (bool): Sample with replacement. Defaults to False.
- `seed` (Optional[int]): Seed for random number generator. Defaults to None.

**Example**:
```python
import pandas as pd
from sampling import sampling_methods

data = pd.read_csv('your_dataset.csv')
sampled_data = sampling_methods.simple_random_sampling(data, sample_size=100, seed=42)
print(sampled_data)
```

### Stratified Sampling

**Function**: `stratified_sampling`

Performs stratified sampling on a DataFrame based on a specified column.

**Parameters**:
- `data` (pd.DataFrame): The input DataFrame.
- `strata_column` (str): Column used for stratification.
- `sample_size` (int): Number of samples to draw from each stratum.
- `max_rows` (Optional[int]): Maximum number of rows to return. Defaults to None.
- `replace` (bool): Sample with replacement. Defaults to False.
- `equal_samples` (bool): Sample an equal number from each stratum. Defaults to False.
- `seed` (Optional[int]): Seed for random number generator. Defaults to None.

**Example**:
```python
import pandas as pd
from sampling import sampling_methods

data = pd.read_csv('your_dataset.csv')
stratified_sampled_data = sampling_methods.stratified_sampling(data, strata_column='category', sample_size=50, seed=42)
print(stratified_sampled_data)
```

### Systematic Sampling

**Function**: `systematic_sampling`

Performs systematic sampling on a DataFrame.

**Parameters**:
- `data` (pd.DataFrame): The input DataFrame.
- `interval` (int): Interval at which to sample.
- `start` (Optional[int]): Starting index for sampling. Defaults to None.
- `max_rows` (Optional[int]): Maximum number of rows to return. Defaults to None.
- `sort_column` (Optional[str]): Column to sort the DataFrame before sampling. Defaults to None.
- `seed` (Optional[int]): Seed for random number generator. Defaults to None.

**Example**:
```python
import pandas as pd
from sampling import sampling_methods

data = pd.read_csv('your_dataset.csv')
systematic_sampled_data = sampling_methods.systematic_sampling(data, interval=10, seed=42)
print(systematic_sampled_data)
```

### Cluster Sampling

**Function**: `cluster_sampling`

Performs cluster sampling on a DataFrame based on a specified column.

**Parameters**:
- `data` (pd.DataFrame): The input DataFrame.
- `cluster_column` (str): Column used for clustering.
- `num_clusters` (int): Number of clusters to sample.
- `max_rows` (Optional[int]): Maximum number of rows to return. Defaults to None.
- `balanced` (bool): Balance the samples across clusters. Defaults to False.
- `seed` (Optional[int]): Seed for random number generator. Defaults to None.

**Example**:
```python
import pandas as pd
from sampling import sampling_methods

data = pd.read_csv('your_dataset.csv')
cluster_sampled_data = sampling_methods.cluster_sampling(data, cluster_column='cluster_id', num_clusters=5, seed=42)
print(cluster_sampled_data)
```

### Multi-Stage Sampling

**Function**: `multi_stage_sampling`

Performs multi-stage sampling on a DataFrame based on a specified cluster column.

**Parameters**:
- `data` (pd.DataFrame): The input DataFrame.
- `cluster_column` (str): Column used for clustering.
- `num_clusters` (int): Number of clusters to sample in the first stage.
- `stage_two_sample_size` (int): Number of samples from each selected cluster in the second stage.
- `max_rows` (Optional[int]): Maximum number of rows to return. Defaults to None.
- `proportional_stage_two` (bool): Sample proportionally from each cluster in the second stage. Defaults to False.
- `replace` (bool): Sample with replacement. Defaults to False.
- `seed` (Optional[int]): Seed for random number generator. Defaults to None.

**Example**:
```python
import pandas as pd
from sampling import sampling_methods

data = pd.read_csv('your_dataset.csv')
multi_stage_sampled_data = sampling_methods.multi_stage_sampling(
    data, 
    cluster_column='cluster_id', 
    num_clusters=5, 
    stage_two_sample_size=10, 
    seed=42
)
print(multi_stage_sampled_data)
```

### Weighted Sampling

**Function**: `weighted_sampling`

Performs weighted sampling on a DataFrame based on a specified weights column.

**Parameters**:
- `data` (pd.DataFrame): The input DataFrame.
- `weights_column` (str): Column used for weights.
- `sample_size` (int): Number of samples to draw.
- `max_rows` (Optional[int]): Maximum number of rows to return. Defaults to None.
- `replace` (bool): Sample with replacement. Defaults to False.
- `normalization` (Optional[str]): Weight normalization method ('min-max' or 'z-score'). Defaults to None.
- `seed` (Optional[int]): Seed for random number generator. Defaults to None.

**Example**:
```python
import pandas as pd
from sampling import sampling_methods

data = pd.read_csv('your_dataset.csv')
weighted_sampled_data = sampling_methods.weighted_sampling(data, weights_column='weights', sample_size=100, seed=42)
print(weighted_sampled_data)
```

### Reservoir Sampling

**Function**: `reservoir_sampling`

Performs reservoir sampling on a data stream or DataFrame.

**Parameters**:
- `data_stream` (Union[pd.DataFrame, List]): Input data stream or DataFrame.
- `sample_size` (int): Number of samples to draw.
- `max_rows` (Optional[int]): Maximum number of rows to return. Defaults to None.
- `handle_infinite_stream` (bool): Handle infinite data streams by breaking early. Defaults to False.
- `seed` (Optional[int]): Seed for random number generator. Defaults to None.

**Example**:
```python
import pandas as pd
from sampling import sampling_methods

data = pd.read_csv('your_dataset.csv')
data_stream = data.to_dict('records')
reservoir_sample = sampling_methods.reservoir_sampling(data_stream, sample_size=100, seed=42)
print(reservoir_sample)
```

### Bootstrap Sampling

**Function**: `bootstrap_sampling`

Performs bootstrap sampling on a DataFrame.

**Parameters**:
- `data` (pd.DataFrame): The input DataFrame.
- `num_samples` (int): Number of bootstrap samples to generate.
- `sample_size` (int): Size of each bootstrap sample.
- `max_rows` (Optional[int]): Maximum number of rows to return. Defaults to None.
- `method` (str): Bootstrap sampling method ('simple' or 'block'). Defaults to 'simple'.
- `seed` (Optional[int]): Seed for random number generator. Defaults to None.

**Example**:
```python
import pandas as pd
from sampling import sampling_methods

data = pd.read_csv('your_dataset.csv')
bootstrap_samples = sampling_methods.bootstrap_sampling(data, num_samples=10, sample_size=100, seed=42)

for i, sample in enumerate(bootstrap_samples):
    print(f"Bootstrap Sample {i+1}")
    print(sample)
```

### Temporal Sampling

**Function**: `temporal_sampling`

Performs temporal sampling on a DataFrame based on a specified time column.

**Parameters**:
- `data` (pd.DataFrame): The input DataFrame.
- `time_column` (str): Column used for time-based sampling.
- `start_time` (pd.Timestamp): Start time for sampling.
- `end_time` (pd.Timestamp): End time for sampling.
- `interval` (int): Interval at which to sample.
- `sample_size` (int): Number of samples to draw in each interval.
- `max_rows` (Optional[int]): Maximum number of rows to return. Defaults to None.
- `time_zone` (Optional[str]): Time zone for the time column. Defaults to None.
- `unit` (str): Unit of time for the interval ('days', 'weeks', 'months'). Defaults to 'days'.
- `seed` (Optional[int]): Seed for random number generator. Defaults to None.

**Example**:
```python
import pandas as pd
from sampling import sampling_methods

data = pd.read_csv('your_dataset.csv')
temporal_sampled_data = sampling_methods.temporal_sampling(
    data, 
    time_column='timestamp', 
    start_time=pd.Timestamp('2021-01-01'), 
    end_time=pd.Timestamp('2021-12-31'), 
    interval=7, 
    sample_size=10, 
    seed=42
)
print(temporal_sampled_data)
```

### Spatial Sampling

**Function**: `spatial_sampling`

Performs spatial sampling on a DataFrame based on specified latitude and longitude columns.

**Parameters**:
- `data` (pd.DataFrame): The input DataFrame.
- `latitude_column` (str): Column used for latitude.
- `longitude_column` (str): Column used for longitude.
- `region` (Union[Polygon, List[Polygon]]): Region or list of regions to sample within.
- `sample_size` (int): Number of samples to draw.
- `max_rows` (Optional[int]): Maximum number of rows to return. Defaults to None.
- `complex_region` (bool): Indicates if the region is a complex polygon. Defaults to False.
- `seed` (Optional[int]): Seed for random number generator. Defaults to None.

**Example**:
```python
import pandas as pd
from shapely.geometry import Polygon
from sampling import sampling_methods

data = pd.read_csv('your_dataset.csv')
region = Polygon([(-90, -180), (-90, 180), (90, 180), (90, -180)])
spatial_sampled_data = sampling_methods.spatial_sampling(
    data, 
    latitude_column='lat', 
    longitude_column='lon', 
    region=region, 
    sample_size=100, 
    seed=42
)
print(spatial_sampled_data)
```

## Contributing

Contributions to Sample_Py are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Write your code and tests.
4. Ensure all tests pass.
5. Submit a pull request with a description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

## Contact Information

For questions, suggestions, or issues, please contact the project maintainers:

- Project Maintainer: [J](mailto:jchase.msu@gmail.com)

## Changelog

All notable changes to this project will be documented here.

### [Unreleased]

- Initial release with the following sampling methods:
  - Simple Random Sampling
  - Stratified Sampling
  - Systematic Sampling
  - Cluster Sampling
  - Multi-Stage Sampling
  - Weighted Sampling
  - Reservoir Sampling
  - Bootstrap Sampling
  - Temporal Sampling
  - Spatial Sampling
