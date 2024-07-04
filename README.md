# Sample_Py

A Python project implementing various sampling methods for data analysis.

## Overview

This project provides a collection of functions for performing different types of sampling on datasets. The sampling methods include:

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

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Simple Random Sampling](#simple-random-sampling)
  - [Stratified Sampling](#stratified-sampling)
  - [Systematic Sampling](#systematic-sampling)
  - [Cluster Sampling](#cluster-sampling)
  - [Multi-Stage Sampling](#multi-stage-sampling)
  - [Weighted Sampling](#weighted-sampling)
  - [Reservoir Sampling](#reservoir-sampling)
  - [Bootstrap Sampling](#bootstrap-sampling)
  - [Temporal Sampling](#temporal-sampling)
  - [Spatial Sampling](#spatial-sampling)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To install the project, you can clone the repository and install the dependencies using `pip`.

```bash
git clone https://github.com/elkronos/sample_py.git
cd sample_py
pip install -r requirements.txt
```

## Usage

Here are examples of how to use the sampling functions provided in this project:

### Simple Random Sampling

```python
import pandas as pd
from sampling import sampling_methods

# Load your data
data = pd.read_csv('your_dataset.csv')

# Perform simple random sampling
sampled_data = sampling_methods.simple_random_sampling(data, sample_size=100, seed=42)

print(sampled_data)
```

### Stratified Sampling

```python
# Perform stratified sampling
stratified_sampled_data = sampling_methods.stratified_sampling(data, strata_column='category', sample_size=50, seed=42)

print(stratified_sampled_data)
```

### Systematic Sampling

```python
# Perform systematic sampling
systematic_sampled_data = sampling_methods.systematic_sampling(data, interval=10, seed=42)

print(systematic_sampled_data)
```

### Cluster Sampling

```python
# Perform cluster sampling
cluster_sampled_data = sampling_methods.cluster_sampling(data, cluster_column='cluster_id', num_clusters=5, seed=42)

print(cluster_sampled_data)
```

### Multi-Stage Sampling

```python
# Perform multi-stage sampling
multi_stage_sampled_data = sampling_methods.multi_stage_sampling(data, cluster_column='cluster_id', num_clusters=5, stage_two_sample_size=10, seed=42)

print(multi_stage_sampled_data)
```

### Weighted Sampling

```python
# Perform weighted sampling
weighted_sampled_data = sampling_methods.weighted_sampling(data, weights_column='weights', sample_size=100, seed=42)

print(weighted_sampled_data)
```

### Reservoir Sampling

```python
# Perform reservoir sampling on a data stream
data_stream = data.to_dict('records')
reservoir_sample = sampling_methods.reservoir_sampling(data_stream, sample_size=100, seed=42)

print(reservoir_sample)
```

### Bootstrap Sampling

```python
# Perform bootstrap sampling
bootstrap_samples = sampling_methods.bootstrap_sampling(data, num_samples=10, sample_size=100, seed=42)

for i, sample in enumerate(bootstrap_samples):
    print(f"Bootstrap Sample {i+1}")
    print(sample)
```

### Temporal Sampling

```python
# Perform temporal sampling
temporal_sampled_data = sampling_methods.temporal_sampling(data, time_column='timestamp', start_time=pd.Timestamp('2021-01-01'), end_time=pd.Timestamp('2021-12-31'), interval=7, sample_size=10, seed=42)

print(temporal_sampled_data)
```

### Spatial Sampling

```python
from shapely.geometry import Polygon

# Define a region as a polygon
region = Polygon([(-10, -10), (-10, 10), (10, 10), (10, -10)])

# Perform spatial sampling
spatial_sampled_data = sampling_methods.spatial_sampling(data, latitude_column='lat', longitude_column='lon', region=region, sample_size=100, seed=42)

print(spatial_sampled_data)
```

## Testing

To run the tests, use the following command:

```bash
pytest
```

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch-name`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-branch-name`.
5. Submit a pull request.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

If you have any questions or suggestions, feel free to reach out to:

- jchase.msu@gmail.com
