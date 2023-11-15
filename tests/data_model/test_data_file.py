"""Tests of DataFile pydantic models."""
import datetime
import json
import time

import bson
import numpy as np
from svalbard.data_model.data_file import (
    Data,
    MaskMetaData,
    MeasurementHandle,
    MetaData,
)


def test_data(data: Data):
    """Test that 'data' generated from fixture in
    conftest.py propertly generates a Data object"""
    assert isinstance(data, Data)
    assert hasattr(data, "datasets")
    assert isinstance(data.datasets, list)
    for dataset in data.datasets:
        assert isinstance(dataset, Data.DataSet)
        assert hasattr(dataset, "name")
        assert hasattr(dataset, "memory")
    assert hasattr(data, "handle")
    assert isinstance(data.handle, MeasurementHandle)


def test_metadata():
    mdat = MetaData(
        name="test_name",
        user="test_user",
        station="test_station",
        cooldown=datetime.datetime(2023, 9, 12),
    )
    l_mdat = MetaData(**mdat.dict())
    assert mdat == l_mdat


def test_datafile_bson(metadata: MetaData):
    """Test that DataFile can be converted to bson"""
    bson_encoded = bson.BSON.encode(json.loads(metadata.json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_metadata = MetaData(**bson_decoded)
    assert new_metadata == metadata


def test_mask_metadata_bson(mask_metadata: MaskMetaData):
    """Test that DataFile using MaskMetaData can be converted to bson"""
    bson_encoded = bson.BSON.encode(json.loads(mask_metadata.json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_metadata = MaskMetaData(**bson_decoded)
    assert new_metadata == mask_metadata


def test_metadata_datetime():
    mdat1 = MetaData(name="test_name1", user="test_user", station="test_station")
    time.sleep(0.001)
    mdat2 = MetaData(name="test_name2", user="test_user", station="test_station")

    assert mdat1.date < mdat2.date


def test_data_from_measurement(measurement):
    """Test that Data can be created from a measurement"""
    data_from_measurement = Data.from_measurement(measurement)
    test_data(data_from_measurement)

    assert len(data_from_measurement.datasets) == 2
    assert data_from_measurement.datasets[0].name == "test_name1"
    assert data_from_measurement.datasets[0].memory.shape == (7,)
    assert data_from_measurement.datasets[0].memory.dtype == "float64"
    assert data_from_measurement.datasets[0].memory.to_array().shape == (7,)
    assert np.allclose(
        data_from_measurement.datasets[0].memory.to_array(),
        np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]),
    )
    assert data_from_measurement.datasets[0].memory.to_array().dtype == np.float64

    assert data_from_measurement.datasets[1].name == "test_name3"
    assert data_from_measurement.datasets[1].memory.shape == (7,)
    assert data_from_measurement.datasets[1].memory.dtype == "complex128"
    assert data_from_measurement.datasets[1].memory.to_array().shape == (7,)
    assert data_from_measurement.datasets[1].memory.to_array().dtype == np.complex128


def test_dataset_from_array():
    """Test that a DataSet can be created from a numpy array and name str"""
    arr = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    name = "test_dataset"
    dataset = Data.DataSet.from_array(name, arr)
    assert isinstance(dataset, Data.DataSet)
    assert dataset.name == "test_dataset"
    assert dataset.memory.shape == (7,)
    assert dataset.memory.dtype == "float64"
    assert dataset.memory.to_array().shape == (7,)
    assert dataset.memory.to_array().dtype == np.float64
