import pandas as pd
import numpy as np
import random
import logging
from shapely.geometry import Point, Polygon
from typing import List, Optional, Union
from .utils import set_seed, validate_dataframe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def simple_random_sampling(data: pd.DataFrame, sample_size: int, max_rows: Optional[int] = None, replace: bool = False, seed: Optional[int] = None) -> pd.DataFrame:
    logging.info("Starting simple random sampling.")
    validate_dataframe(data)
    set_seed(seed)
    if sample_size > len(data) and not replace:
        logging.error("Sample size cannot be larger than population size when sampling without replacement.")
        raise ValueError("Sample size cannot be larger than population size when sampling without replacement.")
    sampled_data = data.sample(n=sample_size, replace=replace, random_state=seed)
    if max_rows is not None and max_rows > 0:
        sampled_data = sampled_data.iloc[:max_rows]
    logging.info("Simple random sampling completed.")
    return sampled_data

def stratified_sampling(data: pd.DataFrame, strata_column: str, sample_size: int, max_rows: Optional[int] = None, replace: bool = False, equal_samples: bool = False, seed: Optional[int] = None) -> pd.DataFrame:
    logging.info("Starting stratified sampling.")
    validate_dataframe(data, required_columns=[strata_column])
    set_seed(seed)
    strata = data[strata_column].unique()
    stratified_sample = pd.DataFrame()
    for stratum in strata:
        stratum_data = data[data[strata_column] == stratum]
        stratum_sample_size = sample_size if equal_samples else int(sample_size * (len(stratum_data) / len(data)))
        if stratum_sample_size > len(stratum_data) and not replace:
            logging.error(f"Sample size cannot be larger than stratum size when sampling without replacement for stratum {stratum}.")
            raise ValueError(f"Sample size cannot be larger than stratum size when sampling without replacement for stratum {stratum}.")
        stratified_sample = pd.concat([
            stratified_sample,
            stratum_data.sample(n=stratum_sample_size, replace=replace, random_state=seed)
        ])
    if max_rows is not None and max_rows > 0:
        stratified_sample = stratified_sample.iloc[:max_rows]
    logging.info("Stratified sampling completed.")
    return stratified_sample

def systematic_sampling(data: pd.DataFrame, interval: int, start: Optional[int] = None, max_rows: Optional[int] = None, sort_column: Optional[str] = None, seed: Optional[int] = None) -> pd.DataFrame:
    logging.info("Starting systematic sampling.")
    validate_dataframe(data)
    set_seed(seed)
    if sort_column is not None:
        data = data.sort_values(by=sort_column).reset_index(drop=True)
    if start is None:
        start = np.random.randint(0, interval)
    sampled_data = data.iloc[start::interval]
    if max_rows is not None and max_rows > 0:
        sampled_data = sampled_data.iloc[:max_rows]
    logging.info("Systematic sampling completed.")
    return sampled_data

def cluster_sampling(data: pd.DataFrame, cluster_column: str, num_clusters: int, max_rows: Optional[int] = None, balanced: bool = False, seed: Optional[int] = None) -> pd.DataFrame:
    logging.info("Starting cluster sampling.")
    validate_dataframe(data, required_columns=[cluster_column])
    set_seed(seed)
    clusters = data[cluster_column].unique()
    if num_clusters > len(clusters):
        logging.error("Number of clusters to sample cannot be greater than the total number of clusters.")
        raise ValueError("Number of clusters to sample cannot be greater than the total number of clusters.")
    sampled_clusters = np.random.choice(clusters, num_clusters, replace=False)
    sampled_data = data[data[cluster_column].isin(sampled_clusters)]
    if balanced:
        cluster_sizes = sampled_data[cluster_column].value_counts()
        min_cluster_size = cluster_sizes.min()
        sampled_data = sampled_data.groupby(cluster_column).apply(lambda x: x.sample(n=min_cluster_size)).reset_index(drop=True)
    if max_rows is not None and max_rows > 0:
        sampled_data = sampled_data.iloc[:max_rows]
    logging.info("Cluster sampling completed.")
    return sampled_data

def multi_stage_sampling(data: pd.DataFrame, cluster_column: str, num_clusters: int, stage_two_sample_size: int, max_rows: Optional[int] = None, proportional_stage_two: bool = False, replace: bool = False, seed: Optional[int] = None) -> pd.DataFrame:
    logging.info("Starting multi-stage sampling.")
    validate_dataframe(data, required_columns=[cluster_column])
    set_seed(seed)
    clusters = data[cluster_column].unique()
    if num_clusters > len(clusters):
        logging.error("Number of clusters to sample cannot be greater than the total number of clusters.")
        raise ValueError("Number of clusters to sample cannot be greater than the total number of clusters.")
    sampled_clusters = np.random.choice(clusters, num_clusters, replace=False)
    stage_two_sample = pd.DataFrame()
    for cluster in sampled_clusters:
        cluster_data = data[data[cluster_column] == cluster]
        cluster_sample_size = int(stage_two_sample_size * (len(cluster_data) / len(data))) if proportional_stage_two else stage_two_sample_size
        if cluster_sample_size > len(cluster_data) and not replace:
            logging.error(f"Sample size cannot be larger than cluster size when sampling without replacement for cluster {cluster}.")
            raise ValueError(f"Sample size cannot be larger than cluster size when sampling without replacement for cluster {cluster}.")
        stage_two_sample = pd.concat([
            stage_two_sample,
            cluster_data.sample(n=cluster_sample_size, replace=replace, random_state=seed)
        ])
    if max_rows is not None and max_rows > 0:
        stage_two_sample = stage_two_sample.iloc[:max_rows]
    logging.info("Multi-stage sampling completed.")
    return stage_two_sample

def weighted_sampling(data: pd.DataFrame, weights_column: str, sample_size: int, max_rows: Optional[int] = None, replace: bool = False, normalization: Optional[str] = None, seed: Optional[int] = None) -> pd.DataFrame:
    logging.info("Starting weighted sampling.")
    validate_dataframe(data, required_columns=[weights_column])
    set_seed(seed)
    if normalization == 'min-max':
        data[weights_column] = (data[weights_column] - data[weights_column].min()) / (data[weights_column].max() - data[weights_column].min())
    elif normalization == 'z-score':
        data[weights_column] = (data[weights_column] - data[weights_column].mean()) / data[weights_column].std()
    if any(data[weights_column] <= 0):
        logging.error("Weights must be positive.")
        raise ValueError("Weights must be positive.")
    sampled_data = data.sample(n=sample_size, weights=data[weights_column], replace=replace, random_state=seed)
    if max_rows is not None and max_rows > 0:
        sampled_data = sampled_data.iloc[:max_rows]
    logging.info("Weighted sampling completed.")
    return sampled_data

def reservoir_sampling(data_stream: Union[pd.DataFrame, List], sample_size: int, max_rows: Optional[int] = None, handle_infinite_stream: bool = False, seed: Optional[int] = None) -> List:
    logging.info("Starting reservoir sampling.")
    set_seed(seed)
    reservoir = []
    for i, item in enumerate(data_stream):
        if i < sample_size:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < sample_size:
                reservoir[j] = item
        if handle_infinite_stream and len(reservoir) > sample_size:
            break
    if max_rows is not None and max_rows > 0:
        reservoir = reservoir[:max_rows]
    logging.info("Reservoir sampling completed.")
    return reservoir

def bootstrap_sampling(data: pd.DataFrame, num_samples: int, sample_size: int, max_rows: Optional[int] = None, method: str = 'simple', seed: Optional[int] = None) -> List[pd.DataFrame]:
    logging.info("Starting bootstrap sampling.")
    validate_dataframe(data)
    set_seed(seed)
    bootstrap_samples = []
    for _ in range(num_samples):
        if method == 'simple':
            sample = data.sample(n=sample_size, replace=True, random_state=seed)
        elif method == 'block':
            block_indices = list(range(0, len(data), sample_size))
            blocks = [data.iloc[i:i+sample_size] for i in block_indices]
            chosen_blocks = random.choices(blocks, k=num_samples)
            sample = pd.concat(chosen_blocks).sample(n=sample_size, replace=True, random_state=seed)
        bootstrap_samples.append(sample)
    if max_rows is not None and max_rows > 0:
        combined_samples = pd.concat(bootstrap_samples).drop_duplicates().reset_index(drop=True)
        combined_samples = combined_samples.iloc[:max_rows]
        bootstrap_samples = [combined_samples.iloc[i:i+sample_size] for i in range(0, len(combined_samples), sample_size)]
    logging.info("Bootstrap sampling completed.")
    return bootstrap_samples

def temporal_sampling(data: pd.DataFrame, time_column: str, start_time: pd.Timestamp, end_time: pd.Timestamp, interval: int, sample_size: int, max_rows: Optional[int] = None, time_zone: Optional[str] = None, unit: str = 'days', seed: Optional[int] = None) -> pd.DataFrame:
    logging.info("Starting temporal sampling.")
    validate_dataframe(data, required_columns=[time_column])
    set_seed(seed)
    if time_zone is not None:
        data[time_column] = pd.to_datetime(data[time_column]).dt.tz_localize('UTC').dt.tz_convert(time_zone)
    else:
        data[time_column] = pd.to_datetime(data[time_column])
    if unit == 'days':
        interval = pd.Timedelta(days=interval)
    elif unit == 'weeks':
        interval = pd.Timedelta(weeks=interval)
    elif unit == 'months':
        interval = pd.DateOffset(months=interval)
    data = data[(data[time_column] >= start_time) & (data[time_column] <= end_time)]
    sampled_data = pd.DataFrame()
    current_time = start_time
    while current_time < end_time:
        interval_data = data[(data[time_column] >= current_time) & (data[time_column] < current_time + interval)]
        if len(interval_data) < sample_size:
            sampled_data = pd.concat([sampled_data, interval_data])
        else:
            sampled_data = pd.concat([sampled_data, interval_data.sample(n=sample_size, random_state=seed)])
        current_time += interval
    if max_rows is not None and max_rows > 0:
        sampled_data = sampled_data.iloc[:max_rows]
    logging.info("Temporal sampling completed.")
    return sampled_data

def spatial_sampling(data: pd.DataFrame, latitude_column: str, longitude_column: str, region: Union[Polygon, List[Polygon]], sample_size: int, max_rows: Optional[int] = None, complex_region: bool = False, seed: Optional[int] = None) -> pd.DataFrame:
    logging.info("Starting spatial sampling.")
    validate_dataframe(data, required_columns=[latitude_column, longitude_column])
    set_seed(seed)
    def point_in_region(lat: float, lon: float, region: Union[Polygon, List[Polygon]]) -> bool:
        point = Point(lon, lat)
        if complex_region:
            return any(poly.contains(point) for poly in region)
        return region.contains(point)
    data['in_region'] = data.apply(lambda row: point_in_region(row[latitude_column], row[longitude_column], region), axis=1)
    region_data = data[data['in_region']]
    if sample_size > len(region_data):
        logging.error("Sample size cannot be larger than the number of points within the region.")
        raise ValueError("Sample size cannot be larger than the number of points within the region.")
    sampled_data = region_data.sample(n=sample_size, random_state=seed)
    if max_rows is not None and max_rows > 0:
        sampled_data = sampled_data.iloc[:max_rows]
    logging.info("Spatial sampling completed.")
    return sampled_data
