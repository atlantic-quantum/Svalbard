import json

import bson
import numpy as np
import pytest

from svalbard.data_model.measurement.lookup import LookupInterpolation, LookupTable


@pytest.fixture(name="lookup_table")
def fixture_lookup_table():
    data = np.arange(10)
    return LookupTable(
        xy={x: y for x, y in zip(data, data[::-1])},
        interpolation=LookupInterpolation.LINEAR,
    )


def test_lookup_table_bson(lookup_table: LookupTable):
    """Test that DataFile can be converted to bson"""
    bson_encoded = bson.BSON.encode(json.loads(lookup_table.model_dump_json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_lookup_table = LookupTable(**bson_decoded)
    assert new_lookup_table == lookup_table


def test_lookup_table_calculate_values(lookup_table: LookupTable):
    """Test that calculate_values works as expected"""
    x_values = np.linspace(0, 9, 100)
    y_values = lookup_table.calculate_values(x_values)
    assert np.allclose(y_values, x_values[::-1])


@pytest.fixture(name="lookup_table_random")
def fixture_lookup_table_random():
    data_x = np.random.random(10)
    data_y = np.random.random(10)
    return LookupTable.from_arrays(data_x, data_y, LookupInterpolation.CUBIC)


def test_lookup_table_sorted(lookup_table_random: LookupTable):
    """Test that x_sorted is sorted and y_sorted is sorted by x_sorted"""
    assert np.all(lookup_table_random.x_sorted[:-1] <= lookup_table_random.x_sorted[1:])

    for x, y in zip(lookup_table_random.x_sorted, lookup_table_random.y_sorted):
        assert y == lookup_table_random.xy[x]


# todo tests for all LookupInterpolation types ?
