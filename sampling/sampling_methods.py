import pandas as pd
import numpy as np
import random
import logging
from shapely.geometry import Point, Polygon
from typing import List, Optional, Union
from .utils import set_seed, get_random_state, validate_dataframe, apply_max_rows

# -----------------------------------------------------------------------------
# SIMPLE RANDOM SAMPLING
# -----------------------------------------------------------------------------
def simple_random_sampling(data: pd.DataFrame,
                           sample_size: int,
                           max_rows: Optional[int] = None,
                           replace: bool = False,
                           seed: Optional[int] = None) -> pd.DataFrame:
    """
    Perform simple random sampling on a DataFrame.
    
    Args:
        data: Input DataFrame.
        sample_size: Number of rows to sample.
        max_rows: Maximum rows to return (after sampling).
        replace: Whether sampling is with replacement.
        seed: Seed for reproducibility.
        
    Returns:
        A DataFrame containing the sampled rows.
        
    Raises:
        ValueError: If sample_size exceeds population size when not sampling with replacement.
    """
    logging.info("Starting simple random sampling.")
    validate_dataframe(data)
    set_seed(seed)
    if sample_size > len(data) and not replace:
        logging.error("Sample size cannot be larger than population size when sampling without replacement.")
        raise ValueError("Sample size cannot be larger than population size when sampling without replacement.")
    sampled_data = data.sample(n=sample_size, replace=replace, random_state=seed)
    sampled_data = apply_max_rows(sampled_data, max_rows)
    logging.info("Simple random sampling completed.")
    return sampled_data

# -----------------------------------------------------------------------------
# STRATIFIED SAMPLING
# -----------------------------------------------------------------------------
def stratified_sampling(data: pd.DataFrame,
                        strata_column: str,
                        sample_size: int,
                        max_rows: Optional[int] = None,
                        replace: bool = False,
                        equal_samples: bool = False,
                        seed: Optional[int] = None) -> pd.DataFrame:
    """
    Perform stratified sampling by splitting the data into strata based on a column.
    
    Args:
        data: Input DataFrame.
        strata_column: Column name defining the strata.
        sample_size: Overall sample size or per-stratum sample size if equal_samples is True.
        max_rows: Maximum rows to return.
        replace: Whether to sample with replacement.
        equal_samples: If True, sample 'sample_size' rows from each stratum; 
                       otherwise, allocate proportionally.
        seed: Seed for reproducibility.
        
    Returns:
        A DataFrame with the stratified sample.
        
    Raises:
        ValueError: If sample size exceeds stratum size (when not replacing).
    """
    logging.info("Starting stratified sampling.")
    validate_dataframe(data, required_columns=[strata_column])
    set_seed(seed)
    rng = get_random_state(seed)
    
    strata = data[strata_column].unique()
    stratified_sample = pd.DataFrame()
    total_population = len(data)
    
    for stratum in strata:
        stratum_data = data[data[strata_column] == stratum]
        if equal_samples:
            stratum_sample_size = sample_size
        else:
            # Proportional allocation (rounding to nearest integer)
            stratum_sample_size = int(round(sample_size * (len(stratum_data) / total_population)))
            # Ensure at least one sample if there is any data
            stratum_sample_size = max(1, stratum_sample_size)
            
        if stratum_sample_size > len(stratum_data) and not replace:
            logging.error(f"Sample size cannot be larger than stratum size when sampling without replacement for stratum {stratum}.")
            raise ValueError(f"Sample size cannot be larger than stratum size when sampling without replacement for stratum {stratum}.")
        
        # Use a new seed for each stratum to ensure variability
        current_seed = rng.randint(0, 2**32)
        sampled = stratum_data.sample(n=stratum_sample_size, replace=replace, random_state=current_seed)
        stratified_sample = pd.concat([stratified_sample, sampled])
    
    stratified_sample = apply_max_rows(stratified_sample, max_rows)
    logging.info("Stratified sampling completed.")
    return stratified_sample

# -----------------------------------------------------------------------------
# SYSTEMATIC SAMPLING
# -----------------------------------------------------------------------------
def systematic_sampling(data: pd.DataFrame,
                        interval: int,
                        start: Optional[int] = None,
                        max_rows: Optional[int] = None,
                        sort_column: Optional[str] = None,
                        seed: Optional[int] = None) -> pd.DataFrame:
    """
    Perform systematic sampling on a DataFrame.
    
    Args:
        data: Input DataFrame.
        interval: Sampling interval.
        start: Starting index; if None, a random start between 0 and interval-1 is chosen.
        max_rows: Maximum rows to return.
        sort_column: Column name to sort by prior to sampling.
        seed: Seed for reproducibility.
        
    Returns:
        A DataFrame with the systematically sampled rows.
    """
    logging.info("Starting systematic sampling.")
    validate_dataframe(data)
    set_seed(seed)
    if sort_column:
        data = data.sort_values(by=sort_column).reset_index(drop=True)
    if start is None:
        start = np.random.randint(0, interval)
    sampled_data = data.iloc[start::interval]
    sampled_data = apply_max_rows(sampled_data, max_rows)
    logging.info("Systematic sampling completed.")
    return sampled_data

# -----------------------------------------------------------------------------
# CLUSTER SAMPLING
# -----------------------------------------------------------------------------
def cluster_sampling(data: pd.DataFrame,
                     cluster_column: str,
                     num_clusters: int,
                     max_rows: Optional[int] = None,
                     balanced: bool = False,
                     seed: Optional[int] = None) -> pd.DataFrame:
    """
    Perform cluster sampling by selecting clusters and then including all points (or a balanced subsample).
    
    Args:
        data: Input DataFrame.
        cluster_column: Column name that defines clusters.
        num_clusters: Number of clusters to sample.
        max_rows: Maximum rows to return.
        balanced: If True, select the same number of rows from each cluster.
        seed: Seed for reproducibility.
        
    Returns:
        A DataFrame with the sampled clusters.
        
    Raises:
        ValueError: If num_clusters exceeds the total number of clusters.
    """
    logging.info("Starting cluster sampling.")
    validate_dataframe(data, required_columns=[cluster_column])
    set_seed(seed)
    rng = get_random_state(seed)
    
    clusters = data[cluster_column].unique()
    if num_clusters > len(clusters):
        logging.error("Number of clusters to sample cannot be greater than the total number of clusters.")
        raise ValueError("Number of clusters to sample cannot be greater than the total number of clusters.")
    
    sampled_clusters = rng.choice(clusters, num_clusters, replace=False)
    sampled_data = data[data[cluster_column].isin(sampled_clusters)]
    
    if balanced:
        cluster_sizes = sampled_data[cluster_column].value_counts()
        min_cluster_size = cluster_sizes.min()
        sampled_data = sampled_data.groupby(cluster_column).apply(lambda x: x.sample(n=min_cluster_size, random_state=rng.randint(0, 2**32))).reset_index(drop=True)
    
    sampled_data = apply_max_rows(sampled_data, max_rows)
    logging.info("Cluster sampling completed.")
    return sampled_data

# -----------------------------------------------------------------------------
# MULTI-STAGE SAMPLING
# -----------------------------------------------------------------------------
def multi_stage_sampling(data: pd.DataFrame,
                         cluster_column: str,
                         num_clusters: int,
                         stage_two_sample_size: int,
                         max_rows: Optional[int] = None,
                         proportional_stage_two: bool = False,
                         replace: bool = False,
                         seed: Optional[int] = None) -> pd.DataFrame:
    """
    Perform multi-stage sampling: first select clusters, then sample within each cluster.
    
    Args:
        data: Input DataFrame.
        cluster_column: Column name that defines clusters.
        num_clusters: Number of clusters to sample.
        stage_two_sample_size: Number of samples to draw within each selected cluster 
                               (or a baseline for proportional allocation).
        max_rows: Maximum rows to return.
        proportional_stage_two: If True, scale stage_two_sample_size proportional to the cluster size.
        replace: Whether to sample with replacement in the second stage.
        seed: Seed for reproducibility.
        
    Returns:
        A DataFrame with the multi-stage sample.
        
    Raises:
        ValueError: If the second-stage sample size exceeds the cluster size when not sampling with replacement.
    """
    logging.info("Starting multi-stage sampling.")
    validate_dataframe(data, required_columns=[cluster_column])
    set_seed(seed)
    rng = get_random_state(seed)
    
    clusters = data[cluster_column].unique()
    if num_clusters > len(clusters):
        logging.error("Number of clusters to sample cannot be greater than the total number of clusters.")
        raise ValueError("Number of clusters to sample cannot be greater than the total number of clusters.")
    
    sampled_clusters = rng.choice(clusters, num_clusters, replace=False)
    stage_two_sample = pd.DataFrame()
    
    for cluster in sampled_clusters:
        cluster_data = data[data[cluster_column] == cluster]
        if proportional_stage_two:
            # Scale sample size proportional to the cluster's share of total data
            cluster_sample_size = int(round(stage_two_sample_size * (len(cluster_data) / len(data))))
            cluster_sample_size = max(1, cluster_sample_size)
        else:
            cluster_sample_size = stage_two_sample_size
        
        if cluster_sample_size > len(cluster_data) and not replace:
            logging.error(f"Sample size cannot be larger than cluster size when sampling without replacement for cluster {cluster}.")
            raise ValueError(f"Sample size cannot be larger than cluster size when sampling without replacement for cluster {cluster}.")
        
        current_seed = rng.randint(0, 2**32)
        sampled = cluster_data.sample(n=cluster_sample_size, replace=replace, random_state=current_seed)
        stage_two_sample = pd.concat([stage_two_sample, sampled])
    
    stage_two_sample = apply_max_rows(stage_two_sample, max_rows)
    logging.info("Multi-stage sampling completed.")
    return stage_two_sample

# -----------------------------------------------------------------------------
# WEIGHTED SAMPLING
# -----------------------------------------------------------------------------
def weighted_sampling(data: pd.DataFrame,
                      weights_column: str,
                      sample_size: int,
                      max_rows: Optional[int] = None,
                      replace: bool = False,
                      normalization: Optional[str] = None,
                      seed: Optional[int] = None) -> pd.DataFrame:
    """
    Perform weighted sampling using a specified weights column.
    
    Args:
        data: Input DataFrame.
        weights_column: Column name with weights.
        sample_size: Number of rows to sample.
        max_rows: Maximum rows to return.
        replace: Whether sampling is with replacement.
        normalization: Normalization method ('min-max' or 'z-score') to adjust weights.
        seed: Seed for reproducibility.
        
    Returns:
        A DataFrame with the weighted sample.
        
    Raises:
        ValueError: If any weight is non-positive after normalization.
    """
    logging.info("Starting weighted sampling.")
    validate_dataframe(data, required_columns=[weights_column])
    set_seed(seed)
    rng = get_random_state(seed)
    
    data_copy = data.copy()  # work on a copy to avoid modifying the original
    if normalization == 'min-max':
        data_copy[weights_column] = (data_copy[weights_column] - data_copy[weights_column].min()) / (data_copy[weights_column].max() - data_copy[weights_column].min())
    elif normalization == 'z-score':
        data_copy[weights_column] = (data_copy[weights_column] - data_copy[weights_column].mean()) / data_copy[weights_column].std()
    
    if (data_copy[weights_column] <= 0).any():
        logging.error("Weights must be positive.")
        raise ValueError("Weights must be positive.")
    
    sampled_data = data_copy.sample(n=sample_size, weights=data_copy[weights_column], replace=replace, random_state=rng.randint(0, 2**32))
    sampled_data = apply_max_rows(sampled_data, max_rows)
    logging.info("Weighted sampling completed.")
    return sampled_data

# -----------------------------------------------------------------------------
# RESERVOIR SAMPLING
# -----------------------------------------------------------------------------
def reservoir_sampling(data_stream: Union[pd.DataFrame, List],
                       sample_size: int,
                       max_rows: Optional[int] = None,
                       handle_infinite_stream: bool = False,
                       seed: Optional[int] = None) -> List:
    """
    Perform reservoir sampling on an iterable data stream.
    
    Args:
        data_stream: An iterable data stream (e.g., list or generator).
                     (Note: If a DataFrame is passed, iteration will occur over its rows via .itertuples().)
        sample_size: The fixed size of the reservoir.
        max_rows: Maximum number of items to return from the reservoir.
        handle_infinite_stream: If True, stop sampling once the reservoir is full.
        seed: Seed for reproducibility.
        
    Returns:
        A list containing the reservoir sample.
    """
    logging.info("Starting reservoir sampling.")
    set_seed(seed)
    reservoir = []
    
    # If data_stream is a DataFrame, iterate over its rows
    if isinstance(data_stream, pd.DataFrame):
        iterator = data_stream.itertuples(index=False)
    else:
        iterator = iter(data_stream)
    
    for i, item in enumerate(iterator):
        if i < sample_size:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < sample_size:
                reservoir[j] = item
        if handle_infinite_stream and i > 10000:  # safety break for an infinite stream example
            break
    
    if max_rows is not None and max_rows > 0:
        reservoir = reservoir[:max_rows]
    logging.info("Reservoir sampling completed.")
    return reservoir

# -----------------------------------------------------------------------------
# BOOTSTRAP SAMPLING
# -----------------------------------------------------------------------------
def bootstrap_sampling(data: pd.DataFrame,
                       num_samples: int,
                       sample_size: int,
                       max_rows: Optional[int] = None,
                       method: str = 'simple',
                       seed: Optional[int] = None) -> List[pd.DataFrame]:
    """
    Perform bootstrap sampling to generate multiple samples from the data.
    
    Args:
        data: Input DataFrame.
        num_samples: Number of bootstrap samples to generate.
        sample_size: Number of rows in each bootstrap sample.
        max_rows: Maximum total rows to return (after combining and de-duplicating).
        method: 'simple' for simple bootstrap or 'block' for block bootstrap.
        seed: Seed for reproducibility.
        
    Returns:
        A list of DataFrames, each representing one bootstrap sample.
    """
    logging.info("Starting bootstrap sampling.")
    validate_dataframe(data)
    set_seed(seed)
    rng = get_random_state(seed)
    bootstrap_samples = []
    
    if method not in ['simple', 'block']:
        logging.error("Method must be either 'simple' or 'block'.")
        raise ValueError("Method must be either 'simple' or 'block'.")
    
    if method == 'simple':
        for _ in range(num_samples):
            current_seed = rng.randint(0, 2**32)
            sample = data.sample(n=sample_size, replace=True, random_state=current_seed)
            bootstrap_samples.append(sample)
    elif method == 'block':
        # Divide data into blocks of size sample_size.
        block_indices = list(range(0, len(data), sample_size))
        blocks = [data.iloc[i:i+sample_size] for i in block_indices if not data.iloc[i:i+sample_size].empty]
        if not blocks:
            logging.error("No blocks created; check the sample_size and data length.")
            raise ValueError("No blocks created; check the sample_size and data length.")
        for _ in range(num_samples):
            current_seed = rng.randint(0, 2**32)
            chosen_blocks = random.choices(blocks, k=1)  # choose one block at a time
            sample = pd.concat(chosen_blocks).sample(n=sample_size, replace=True, random_state=current_seed)
            bootstrap_samples.append(sample)
    
    # Optionally combine samples, remove duplicates, and re-partition if max_rows is set.
    if max_rows is not None and max_rows > 0:
        combined_samples = pd.concat(bootstrap_samples).drop_duplicates().reset_index(drop=True)
        combined_samples = apply_max_rows(combined_samples, max_rows)
        # Partition back into samples of size sample_size (last sample may be smaller)
        bootstrap_samples = [combined_samples.iloc[i:i+sample_size] for i in range(0, len(combined_samples), sample_size)]
    
    logging.info("Bootstrap sampling completed.")
    return bootstrap_samples

# -----------------------------------------------------------------------------
# TEMPORAL SAMPLING
# -----------------------------------------------------------------------------
def temporal_sampling(data: pd.DataFrame,
                      time_column: str,
                      start_time: pd.Timestamp,
                      end_time: pd.Timestamp,
                      interval: int,
                      sample_size: int,
                      max_rows: Optional[int] = None,
                      time_zone: Optional[str] = None,
                      unit: str = 'days',
                      seed: Optional[int] = None) -> pd.DataFrame:
    """
    Perform temporal sampling by dividing the data into intervals and sampling within each.
    
    Args:
        data: Input DataFrame.
        time_column: Column containing time/date information.
        start_time: Start of the time window.
        end_time: End of the time window.
        interval: Interval between samples (in the given unit).
        sample_size: Number of samples to draw per interval.
        max_rows: Maximum rows to return.
        time_zone: Time zone for conversion (if needed).
        unit: Unit of the interval ('days', 'weeks', or 'months').
        seed: Seed for reproducibility.
        
    Returns:
        A DataFrame with the temporal sample.
    """
    logging.info("Starting temporal sampling.")
    validate_dataframe(data, required_columns=[time_column])
    set_seed(seed)
    rng = get_random_state(seed)
    
    data_copy = data.copy()
    if time_zone is not None:
        data_copy[time_column] = pd.to_datetime(data_copy[time_column]).dt.tz_localize('UTC').dt.tz_convert(time_zone)
    else:
        data_copy[time_column] = pd.to_datetime(data_copy[time_column])
    
    # Filter the data to the desired time window
    data_copy = data_copy[(data_copy[time_column] >= start_time) & (data_copy[time_column] <= end_time)]
    
    if unit == 'days':
        delta = pd.Timedelta(days=interval)
    elif unit == 'weeks':
        delta = pd.Timedelta(weeks=interval)
    elif unit == 'months':
        # For months, use DateOffset
        delta = pd.DateOffset(months=interval)
    else:
        logging.error("Unsupported unit. Use 'days', 'weeks', or 'months'.")
        raise ValueError("Unsupported unit. Use 'days', 'weeks', or 'months'.")
    
    sampled_data = pd.DataFrame()
    current_time = start_time
    while current_time < end_time:
        next_time = current_time + delta
        interval_data = data_copy[(data_copy[time_column] >= current_time) & (data_copy[time_column] < next_time)]
        if len(interval_data) <= sample_size:
            sampled_data = pd.concat([sampled_data, interval_data])
        else:
            current_seed = rng.randint(0, 2**32)
            sampled_interval = interval_data.sample(n=sample_size, random_state=current_seed)
            sampled_data = pd.concat([sampled_data, sampled_interval])
        # For DateOffset (months) addition, it works directly; for Timedelta, addition is also supported.
        current_time = next_time
    
    sampled_data = apply_max_rows(sampled_data, max_rows)
    logging.info("Temporal sampling completed.")
    return sampled_data

# -----------------------------------------------------------------------------
# SPATIAL SAMPLING
# -----------------------------------------------------------------------------
def spatial_sampling(data: pd.DataFrame,
                     latitude_column: str,
                     longitude_column: str,
                     region: Union[Polygon, List[Polygon]],
                     sample_size: int,
                     max_rows: Optional[int] = None,
                     complex_region: bool = False,
                     seed: Optional[int] = None) -> pd.DataFrame:
    """
    Perform spatial sampling by selecting points within a specified region.
    
    Args:
        data: Input DataFrame.
        latitude_column: Column name with latitude values.
        longitude_column: Column name with longitude values.
        region: A shapely Polygon or a list of Polygons defining the region(s).
        sample_size: Number of points to sample from within the region.
        max_rows: Maximum rows to return.
        complex_region: If True, treat `region` as a list and check if point is in any.
        seed: Seed for reproducibility.
        
    Returns:
        A DataFrame with the spatial sample.
        
    Raises:
        ValueError: If sample_size is greater than the number of points within the region.
    """
    logging.info("Starting spatial sampling.")
    validate_dataframe(data, required_columns=[latitude_column, longitude_column])
    set_seed(seed)
    rng = get_random_state(seed)
    
    data_copy = data.copy()
    
    def point_in_region(lat: float, lon: float, region: Union[Polygon, List[Polygon]]) -> bool:
        pt = Point(lon, lat)
        if complex_region and isinstance(region, list):
            return any(poly.contains(pt) for poly in region)
        elif isinstance(region, Polygon):
            return region.contains(pt)
        else:
            logging.error("Invalid region type. Must be a Polygon or list of Polygons.")
            raise ValueError("Invalid region type. Must be a Polygon or list of Polygons.")
    
    data_copy['in_region'] = data_copy.apply(lambda row: point_in_region(row[latitude_column], row[longitude_column], region), axis=1)
    region_data = data_copy[data_copy['in_region']].drop(columns=['in_region'])
    
    if sample_size > len(region_data):
        logging.error("Sample size cannot be larger than the number of points within the region.")
        raise ValueError("Sample size cannot be larger than the number of points within the region.")
    
    sampled_data = region_data.sample(n=sample_size, random_state=rng.randint(0, 2**32))
    sampled_data = apply_max_rows(sampled_data, max_rows)
    logging.info("Spatial sampling completed.")
    return sampled_data
