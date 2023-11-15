import json

import bson
import numpy as np
import pytest
from svalbard.data_model.measurement.channel import Channel
from svalbard.data_model.measurement.measurement import Measurement, step_arrays
from svalbard.data_model.measurement.relation import (
    RelationParameters,
    RelationSettings,
)
from svalbard.data_model.measurement.step_config import StepConfig
from svalbard.data_model.measurement.step_item import StepItem
from svalbard.data_model.measurement.step_range import StepRange


def test_measurements_bson(measurement: Measurement):
    """Test that Measurement can be converted to bson"""
    bson_encoded = bson.BSON.encode(json.loads(measurement.json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_measurement = Measurement(**bson_decoded)
    assert new_measurement == measurement


def test_measurements_values_basic(measurement: Measurement):
    measurement.add_log("test_name1")
    values = measurement.calculate_values()
    assert np.allclose(
        values["test_name1"], np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    )


def test_measurement_add_step(
    measurement: Measurement, step_item: StepItem, channel: Channel
):
    measurement.add_step(channel, [11, 22], 1)
    measurement.calculate_values()

    measurement.add_step("test_name3", np.array([111, 222]), 2)

    with pytest.raises(ValueError):
        measurement.add_step("unknown_channel", step_item.calculate_values(), 3)

    new_channel = Channel(
        name="test_name4",
        unit_physical="test_unit",
        instrument_identity="test_instrument",
        instrument_setting_name="test_setting",
        unit_instrument="test_unit",
    )
    measurement.add_step(new_channel, 5.5, 3)

    values = measurement.calculate_values()
    assert np.allclose(
        values["test_name1"], np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0] * 4)
    )
    assert np.allclose(
        values["test_name"],
        np.array([11] * 7 + [22] * 7 + [11] * 7 + [22] * 7),
    )
    assert np.allclose(
        values["test_name3"],
        np.array([111] * 14 + [222] * 14),
    )
    assert np.allclose(values["test_name4"], np.array([5.5] * 28))


@pytest.mark.parametrize("index", [0, None])
def test_measurement_add_step_item(
    index, measurement: Measurement, step_item: StepItem, channel: Channel
):
    measurement.add_channel(channel)
    measurement.add_step_item(step_item, index)
    measurement.calculate_values()


def test_add_step_item_error(
    measurement: Measurement, step_item: StepItem, channel: Channel
):
    # invalid name
    with pytest.raises(ValueError):
        measurement.add_step_item(
            StepItem(
                name="invalid_name",
                config=StepConfig(),
                ranges=[StepRange(start=0, stop=1, step_count=2)],
            ),
        )

    # duplicate name
    measurement.add_channel(channel)
    measurement.add_step_item(step_item)
    with pytest.raises(ValueError):
        measurement.add_step_item(step_item)


def test_measurement_values_relations(measurement: Measurement):
    measurement.add_relation(
        RelationSettings(
            name="test_name2",
            config=StepConfig(),
            equation="-x",
            enable=True,
            parameters=[RelationParameters(variable="x", source_name="test_name1")],
        )
    )
    values = measurement.calculate_values()
    assert np.allclose(
        values["test_name1"], np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    )
    assert np.allclose(
        values["test_name2"], -np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    )


def test_measurement_values_relations_2d(measurement: Measurement):
    measurement.add_relation(
        RelationSettings(
            name="test_name2",
            config=StepConfig(),
            equation="-x",
            enable=True,
            parameters=[RelationParameters(variable="x", source_name="test_name1")],
        )
    )
    measurement.add_step("test_name3", [11, 22], 1)
    values = measurement.calculate_values()
    assert np.allclose(
        values["test_name1"], np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0] * 2)
    )
    assert np.allclose(
        values["test_name2"], -np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0] * 2)
    )
    assert np.allclose(values["test_name3"], np.array([11] * 7 + [22] * 7))


def test_measurement_values_error(measurement: Measurement):
    measurement.add_relation(
        RelationSettings(
            name="test_name2",
            config=StepConfig(),
            parameters=[RelationParameters(variable="x", source_name="invalid_name")],
        )
    )
    with pytest.raises(ValueError):
        measurement.calculate_values()


def test_step_arrays_2d():
    arrays = {"test1": np.array([0, 1]), "test2": np.array([2, 3])}
    new_arrays = step_arrays(arrays)

    assert list(arrays.keys()) == ["test1", "test2"]
    assert list(new_arrays.keys()) == ["test1", "test2"]

    assert np.allclose(new_arrays["test1"], np.array([0, 1, 0, 1]))
    assert np.allclose(new_arrays["test2"], np.array([2, 2, 3, 3]))


def test_step_arrays_3d():
    arrays = {
        "test1": np.array([0, 1]),
        "test2": np.array([2, 3]),
        "test3": np.array([4, 5]),
    }
    new_arrays = step_arrays(arrays)

    assert list(arrays.keys()) == ["test1", "test2", "test3"]
    assert list(new_arrays.keys()) == ["test1", "test2", "test3"]

    assert np.allclose(new_arrays["test1"], np.array([0, 1, 0, 1, 0, 1, 0, 1]))
    assert np.allclose(new_arrays["test2"], np.array([2, 2, 3, 3, 2, 2, 3, 3]))
    assert np.allclose(new_arrays["test3"], np.array([4, 4, 4, 4, 5, 5, 5, 5]))


def test_measurement_relation_validator(channel: Channel):
    Measurement(
        channels=[channel],
        step_items=[],
        relations=[
            RelationSettings(
                name="test_name",
                config=StepConfig(),
                equation="-x",
                enable=True,
                parameters=[RelationParameters(variable="x", source_name="test_name1")],
            )
        ],
        log_channels=["test_name"],
    )


def test_add_relations_error(measurement: Measurement):
    # invalid name
    with pytest.raises(ValueError):
        measurement.add_relation(
            RelationSettings(
                name="invalid_name",
                config=StepConfig(),
                equation="-x",
                enable=True,
                parameters=[RelationParameters(variable="x", source_name="test_name1")],
            )
        )

    # duplicate name
    measurement.add_relation(
        RelationSettings(
            name="test_name2",
            config=StepConfig(),
            equation="-x",
            enable=True,
            parameters=[RelationParameters(variable="x", source_name="test_name1")],
        )
    )
    with pytest.raises(ValueError):
        measurement.add_relation(
            RelationSettings(
                name="test_name2",
                config=StepConfig(),
                equation="-x",
                enable=True,
                parameters=[RelationParameters(variable="x", source_name="test_name1")],
            )
        )


def test_add_channel(measurement: Measurement, channel: Channel):
    measurement.add_channel(channel)
    assert channel in measurement.channels

    # duplicate name error
    with pytest.raises(ValueError):
        measurement.add_channel(channel)


def test_add_log(measurement: Measurement):
    measurement.add_log("test_name2")
    assert "test_name2" in measurement.log_channels

    # invalid name
    with pytest.raises(ValueError):
        measurement.add_log("invalid_name")


def test_get_channel(measurement: Measurement, channel: Channel):
    measurement.add_channel(channel)
    assert measurement.get_channel("test_name") == channel

    # invalid name
    with pytest.raises(ValueError):
        measurement.get_channel("invalid_name")


def test_get_step_item(measurement: Measurement, step_item: StepItem, channel: Channel):
    measurement.add_channel(channel)
    measurement.add_step_item(step_item)
    assert measurement.get_step_item("test_name") == step_item

    # invalid name
    with pytest.raises(ValueError):
        measurement.get_step_item("invalid_name")


def test_get_relation(measurement: Measurement):
    measurement.add_relation(
        RelationSettings(
            name="test_name2",
            config=StepConfig(),
            equation="-x",
            enable=True,
            parameters=[RelationParameters(variable="x", source_name="test_name1")],
        )
    )
    assert measurement.get_relation("test_name2") == measurement.relations[0]

    # invalid name
    with pytest.raises(ValueError):
        measurement.get_relation("invalid_name")


def test_remove_channel(measurement: Measurement, channel: Channel):
    measurement.add_channel(channel)
    assert channel in measurement.channels
    measurement.remove_channel(channel.name)
    assert channel not in measurement.channels

    # invalid channel
    with pytest.raises(ValueError):
        measurement.remove_channel(channel.name)


def test_remove_step_item(
    measurement: Measurement, step_item: StepItem, channel: Channel
):
    measurement.add_channel(channel)
    measurement.add_step_item(step_item)
    assert step_item in measurement.step_items
    measurement.remove_step_item(step_item.name)
    assert step_item not in measurement.step_items

    # invalid step item
    with pytest.raises(ValueError):
        measurement.remove_step_item(step_item.name)


def test_remove_relations(measurement: Measurement):
    measurement.add_relation(
        RelationSettings(
            name="test_name2",
            config=StepConfig(),
            equation="-x",
            enable=True,
            parameters=[RelationParameters(variable="x", source_name="test_name1")],
        )
    )
    assert measurement.relations
    measurement.remove_relation("test_name2")
    assert not measurement.relations

    # invalid relation
    with pytest.raises(ValueError):
        measurement.remove_relation("invalid_name")


def test_remove_log(measurement: Measurement):
    measurement.add_log("test_name2")
    assert "test_name2" in measurement.log_channels
    measurement.remove_log("test_name2")
    assert "test_name2" not in measurement.log_channels

    # invalid name
    with pytest.raises(ValueError):
        measurement.remove_log("invalid_name")


def test_set_log_position(measurement: Measurement):
    measurement.add_log("test_name2")
    assert measurement.log_channels.index("test_name2") == 1
    measurement.set_log_position("test_name2", 0)
    assert measurement.log_channels.index("test_name2") == 0

    # invalid name
    with pytest.raises(ValueError):
        measurement.set_log_position("invalid_name", 0)


def test_set_step_item_position(
    measurement: Measurement, step_item: StepItem, channel: Channel
):
    measurement.add_channel(channel)
    measurement.add_step_item(step_item)
    assert measurement.step_items.index(step_item) == 1
    measurement.set_step_item_position(step_item.name, 0)
    assert measurement.step_items.index(step_item) == 0

    # invalid name
    with pytest.raises(ValueError):
        measurement.set_step_item_position("invalid_name", 0)


def test_empty_measurement():
    measurement = Measurement.create_empty()
    assert measurement.calculate_values() == {}
    assert measurement.channels == []
    assert measurement.step_items == []
    assert measurement.relations == []
    assert measurement.log_channels == []
