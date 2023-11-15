import json

import bson
import numpy as np
import pytest
from svalbard.data_model.measurement.step_item import StepItem, StepRange


def test_step_item_json(step_item: StepItem):
    new_step_item = StepItem(**json.loads(step_item.json()))
    assert new_step_item == step_item


def test_step_item_bson(step_item: StepItem):
    """Test that DataFile can be converted to bson"""
    bson_encoded = bson.BSON.encode(json.loads(step_item.json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    # print(bson_decoded)
    new_step_item = StepItem(**bson_decoded)
    assert new_step_item == step_item


def test_step_item_values(step_item: StepItem):
    values = step_item.calculate_values()
    assert np.allclose(values, np.array([1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))


@pytest.mark.parametrize("index", [0, None])
def test_add_range(index, step_item: StepItem):
    step_range = StepRange(start=0, stop=0, step_count=3)
    assert step_range not in step_item.ranges
    step_item.add_range(step_range, index=index)
    assert step_range in step_item.ranges


def test_get_range(step_item: StepItem):
    step_range = step_item.get_range(0)
    assert step_range is step_item.ranges[0]
    step_range = step_item.get_range(1)
    assert step_range is step_item.ranges[1]

    with pytest.raises(IndexError):
        assert step_item.get_range(2)


def test_remove_range(step_item: StepItem):
    step_range = step_item.get_range(0)
    assert step_range is step_item.ranges[0]
    step_item.remove_range(0)
    assert step_range not in step_item.ranges

    with pytest.raises(IndexError):
        assert step_item.remove_range(1)


def test_set_range_position(step_item: StepItem):
    step_range = step_item.get_range(0)
    assert step_range is step_item.ranges[0]
    step_item.set_range_position(0, 1)
    assert step_range is step_item.ranges[1]
