import json

import bson
import numpy as np
import pytest

from svalbard.data_model.measurement.channel import Channel
from svalbard.data_model.measurement.step_item import (
    RangeTypes,
    StepConfig,
    StepItem,
    StepRange,
)
from svalbard.data_model.measurement.step_range import StepTypes


def test_step_item_json(step_item: StepItem):
    new_step_item = StepItem(**json.loads(step_item.model_dump_json()))
    assert new_step_item == step_item


def test_step_item_bson(step_item: StepItem):
    """Test that DataFile can be converted to bson"""
    bson_encoded = bson.BSON.encode(json.loads(step_item.model_dump_json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_step_item = StepItem(**bson_decoded)
    assert new_step_item == step_item


def test_step_item_values(step_item: StepItem):
    values = step_item.calculate_values()
    assert isinstance(values, np.ndarray)
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


def test_step_range_all_same_type_error(step_item: StepItem):
    step_range = StepRange(values=["a", "b", "c"], range_type=RangeTypes.VALUES)
    step_item.add_range(step_range)
    with pytest.raises(ValueError):
        step_item.calculate_values()


def test_step_range_list():
    step_range = StepRange(values=["a", "b", "c"], range_type=RangeTypes.VALUES)
    step_item = StepItem(name="test_name", config=StepConfig(), ranges=[step_range])
    assert step_item.calculate_values() == ["a", "b", "c"]


@pytest.fixture(name="test_channel")
def fixture_test_channel():
    yield Channel(
        name="test",
        instrument_identity="test",
        instrument_setting_name="test",
        unit_physical="test_unit",
    )


def test_step_item_get_range_strings_empty(test_channel: Channel):
    step_item = StepItem(name="test_name", config=StepConfig(), ranges=[])
    assert step_item.get_range_strings(test_channel.unit_physical) == (
        "",
        "",
    )


def test_step_item_name_pythonic():
    step_item = StepItem(name="test name", config=StepConfig(), ranges=[])
    assert step_item.name == "test_name"


@pytest.mark.parametrize(
    "range_type", [RangeTypes.VALUES, RangeTypes.START_STOP, RangeTypes.CENTER_SPAN]
)
@pytest.mark.parametrize("step_type", [StepTypes.STEP_SIZE, StepTypes.STEP_COUNT])
def test_step_item_get_range_strings_single_start_stop(
    test_channel: Channel, range_type, step_type
):
    step_item = StepItem(
        name="test_name",
        config=StepConfig(),
        ranges=[
            StepRange(
                range_type=range_type,
                start=0.0,
                stop=10.0,
                center=0.0,
                span=10.0,
                step_count=11,
                step_size=1.0,
                step_type=step_type,
                values=[i for i in range(11)],
            )
        ],
    )
    expected_stop = (
        9
        if range_type == RangeTypes.START_STOP and step_type == StepTypes.STEP_SIZE
        else 10
    )
    expected_range_str = (
        f"{'c = ' if range_type==RangeTypes.CENTER_SPAN else ''}0 test_unit"
        f"{', w = ' if range_type == RangeTypes.CENTER_SPAN else ' - '}{expected_stop}"
        " test_unit"
    )
    expected_pts_str = (
        "11 pts"
        if step_type == StepTypes.STEP_COUNT or range_type == RangeTypes.VALUES
        else "1 test_unit"
    )

    range_str, pts_str = step_item.get_range_strings(test_channel.unit_physical)
    assert expected_range_str == range_str
    assert expected_pts_str == pts_str


def test_step_item_get_range_strings_multiple(test_channel: Channel):
    step_item = StepItem(
        name="test_name",
        config=StepConfig(),
        ranges=[
            StepRange(
                range_type=RangeTypes.CENTER_SPAN,
                center=0.0,
                span=10.0,
                step_count=11,
            ),
            StepRange(
                range_type=RangeTypes.START_STOP,
                start=10.0,
                stop=100.0,
                step_count=11,
            ),
        ],
    )
    assert step_item.get_range_strings(test_channel) == (  # type: ignore
        "-5 test_unit - 100 test_unit",
        "22 pts",
    )


def test_step_item_list_range_strings():
    step_range = StepRange(values=["a", "b", "c"], range_type=RangeTypes.VALUES)
    step_item = StepItem(name="test_name", config=StepConfig(), ranges=[step_range])
    assert step_item.get_range_strings("test_unit") == ("", "3 pts")


def test_step_item_step_count(step_item: StepItem):
    assert step_item.step_count == 14


def test_step_item_get_step_value(step_item: StepItem):
    for index, expected in enumerate(step_item.calculate_values()):
        assert step_item.get_step_value(index) == expected

    with pytest.raises(IndexError):
        step_item.get_step_value(14)

    with pytest.raises(IndexError):
        step_item.get_step_value(-1)
