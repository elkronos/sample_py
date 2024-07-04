import pytest
import pandas as pd
import numpy as np
from sampling import sampling_methods

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'id': range(1000),
        'category': ['A', 'B', 'C', 'D'] * 250,
        'value': np.random.randn(1000),
        'weights': np.random.rand(1000),
        'lat': np.random.uniform(-90, 90, 1000),
        'lon': np.random.uniform(-180, 180, 1000),
        'timestamp': pd.date_range(start='2021-01-01', periods=1000, freq='H')
    })

def test_simple_random_sampling(sample_data):
    sampled_data = sampling_methods.simple_random_sampling(sample_data, sample_size=100, seed=42)
    assert len(sampled_data) == 100

def test_stratified_sampling(sample_data):
    stratified_sampled_data = sampling_methods.stratified_sampling(sample_data, strata_column='category', sample_size=50, seed=42)
    assert len(stratified_sampled_data) == 200  # 50 samples per stratum

def test_systematic_sampling(sample_data):
    systematic_sampled_data = sampling_methods.systematic_sampling(sample_data, interval=10, seed=42)
    assert len(systematic_sampled_data) == len(sample_data) // 10

def test_cluster_sampling(sample_data):
    cluster_sampled_data = sampling_methods.cluster_sampling(sample_data, cluster_column='category', num_clusters=2, seed=42)
    assert 'category' in cluster_sampled_data.columns

def test_multi_stage_sampling(sample_data):
    multi_stage_sampled_data = sampling_methods.multi_stage_sampling(sample_data, cluster_column='category', num_clusters=2, stage_two_sample_size=10, seed=42)
    assert 'category' in multi_stage_sampled_data.columns

def test_weighted_sampling(sample_data):
    weighted_sampled_data = sampling_methods.weighted_sampling(sample_data, weights_column='weights', sample_size=100, seed=42)
    assert len(weighted_sampled_data) == 100

def test_reservoir_sampling(sample_data):
    data_stream = sample_data.to_dict('records')
    reservoir_sample = sampling_methods.reservoir_sampling(data_stream, sample_size=100, seed=42)
    assert len(reservoir_sample) == 100

def test_bootstrap_sampling(sample_data):
    bootstrap_samples = sampling_methods.bootstrap_sampling(sample_data, num_samples=10, sample_size=100, seed=42)
    assert len(bootstrap_samples) == 10
    assert all(len(sample) == 100 for sample in bootstrap_samples)

def test_temporal_sampling(sample_data):
    temporal_sampled_data = sampling_methods.temporal_sampling(
        sample_data, 
        time_column='timestamp', 
        start_time=pd.Timestamp('2021-01-01'), 
        end_time=pd.Timestamp('2021-12-31'), 
        interval=7, 
        sample_size=10, 
        seed=42
    )
    assert 'timestamp' in temporal_sampled_data.columns

def test_spatial_sampling(sample_data):
    from shapely.geometry import Polygon

    region = Polygon([(-90, -180), (-90, 180), (90, 180), (90, -180)])
    spatial_sampled_data = sampling_methods.spatial_sampling(
        sample_data, 
        latitude_column='lat', 
        longitude_column='lon', 
        region=region, 
        sample_size=100, 
        seed=42
    )
    assert 'lat' in spatial_sampled_data.columns
    assert 'lon' in spatial_sampled_data.columns
