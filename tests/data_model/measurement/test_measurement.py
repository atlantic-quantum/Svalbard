import json
import warnings
from copy import deepcopy

import bson
import numpy as np
import pytest

from svalbard.data_model.data_file import Data
from svalbard.data_model.measurement.channel import Channel
from svalbard.data_model.measurement.log_channel import LogChannel
from svalbard.data_model.measurement.measurement import (
    Measurement,
    meshgrid,
    step_arrays,
)
from svalbard.data_model.measurement.relation import (
    RelationParameters,
    RelationSettings,
)
from svalbard.data_model.measurement.step_config import StepConfig
from svalbard.data_model.measurement.step_item import StepItem
from svalbard.data_model.measurement.step_range import StepRange


def test_measurement_basic(measurement: Measurement):
    """Test that Measurement can be dumped to dict and back"""
    dict_ = measurement.model_dump()
    assert dict_ is not None
    new_measurement = Measurement(**dict_)
    assert new_measurement == measurement


def test_measurement_basic_json(measurement: Measurement):
    """Test that Measurement can be converted to json and back"""
    json_string = measurement.model_dump_json(indent=2)
    assert json_string is not None
    new_measurement = Measurement(**json.loads(json_string))
    assert new_measurement == measurement


def test_measurements_bson(measurement: Measurement):
    """Test that Measurement can be converted to bson"""
    bson_encoded = bson.BSON.encode(json.loads(measurement.model_dump_json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_measurement = Measurement(**bson_decoded)
    assert new_measurement == measurement


def test_measurements_values_basic(measurement: Measurement):
    measurement.add_log("test_name1")
    values = measurement.calculate_values()
    assert np.allclose(
        np.array(values["test_name1"]), np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
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
        instrument_setting_name="test_setting2",
    )
    measurement.add_step(new_channel, 5.5, 3)

    values = measurement.calculate_values()
    assert np.allclose(
        np.array(values["test_name1"]),
        np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0] * 4),
    )
    assert np.allclose(
        np.array(values["test_name"]),
        np.array([11] * 7 + [22] * 7 + [11] * 7 + [22] * 7),
    )
    assert np.allclose(
        np.array(values["test_name3"]),
        np.array([111] * 14 + [222] * 14),
    )
    assert np.allclose(np.array(values["test_name4"]), np.array([5.5] * 28))


def test_measurement_add_step_no_index(
    measurement: Measurement, step_item: StepItem, channel: Channel
):
    measurement.add_step(channel, [11, 22])
    assert measurement.step_items[-1].name == channel.name


@pytest.mark.parametrize("index", [1, None])
def test_measurement_add_step_item(
    index, measurement: Measurement, step_item: StepItem, channel: Channel
):
    assert measurement.max_step_index == 0
    measurement.add_channel(channel)
    measurement.add_step_item(step_item, index)
    assert measurement.max_step_index == 1
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
        np.array(values["test_name1"]), np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    )
    assert np.allclose(
        np.array(values["test_name2"]), -np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
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
        np.array(values["test_name1"]),
        np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0] * 2),
    )
    assert np.allclose(
        np.array(values["test_name2"]),
        -np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0] * 2),
    )
    assert np.allclose(np.array(values["test_name3"]), np.array([11] * 7 + [22] * 7))


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


def test_validate_step_items_indices_order():
    """Test two orders of step items indices, one valid and one invalid."""
    step_items = [
        StepItem(
            name="test_name1",
            config=StepConfig(),
            ranges=[StepRange(start=0, stop=1, step_count=2)],
            index=1,
            hw_swept=True,
        ),
        StepItem(
            name="test_name1",
            config=StepConfig(),
            ranges=[StepRange(start=0, stop=1, step_count=2)],
            index=0,
        ),
    ]
    with pytest.raises(ValueError):
        Measurement.validate_step_items_indices_order(step_items)  # type: ignore
    step_items[0].index = 0
    step_items[1].index = 1
    Measurement.validate_step_items_indices_order(step_items)  # type: ignore


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
        log_channels=[LogChannel(name="test_name")],
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

    channel2 = deepcopy(channel)
    channel2.name = "new_channel_name"
    # duplicate instrument identity and setting name error
    with pytest.raises(ValueError):
        measurement.add_channel(channel2)


def test_add_log(measurement: Measurement):
    measurement.add_log("test_name2")
    test_log = measurement.get_log_channel("test_name2")
    assert test_log in measurement.log_channels

    # invalid name
    with pytest.raises(ValueError):
        measurement.add_log(LogChannel(name="invalid_name"))


def test_get_channel(measurement: Measurement, channel: Channel):
    measurement.add_channel(channel)
    assert measurement.get_channel("test_name") == channel

    assert measurement.get_channel(channel) == channel

    # invalid name
    with pytest.raises(ValueError):
        measurement.get_channel("invalid_name")


def test_get_step_item(measurement: Measurement, step_item: StepItem, channel: Channel):
    measurement.add_channel(channel)
    max_index = measurement.max_step_index
    measurement.add_step_item(step_item)
    step_item.index = max_index + 1

    assert measurement.get_step_item("test_name") == StepItem(**step_item.model_dump())

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
    step_item_int = measurement.get_step_item(step_item.name)
    assert step_item_int in measurement.step_items
    measurement.remove_step_item(step_item.name)
    assert step_item_int not in measurement.step_items

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
    test_log = measurement.get_log_channel("test_name2")
    assert test_log in measurement.log_channels
    assert measurement.log_channels_names == ["test_name3", "test_name2"]
    measurement.remove_log("test_name2")
    assert test_log not in measurement.log_channels
    assert measurement.log_channels_names == ["test_name3"]

    # invalid name
    with pytest.raises(ValueError):
        measurement.remove_log("invalid_name")


def test_set_log_position(measurement: Measurement):
    measurement.add_log("test_name2")
    test_log = measurement.get_log_channel("test_name2")
    assert measurement.log_channels.index(test_log) == 1
    measurement.set_log_position("test_name2", 0)
    assert measurement.log_channels.index(test_log) == 0

    # invalid name
    with pytest.raises(ValueError):
        measurement.set_log_position("invalid_name", 0)


def test_set_step_item_position(
    measurement: Measurement, step_item: StepItem, channel: Channel
):
    measurement.add_channel(channel)
    measurement.add_step_item(step_item)
    step_item_int = measurement.get_step_item(step_item.name)
    assert measurement.step_items.index(step_item_int) == 1
    measurement.set_step_item_position(step_item.name, 0)
    assert measurement.step_items.index(step_item_int) == 0

    # invalid name
    with pytest.raises(ValueError):
        measurement.set_step_item_position("invalid_name", 0)


def test_set_step_item_index(
    measurement: Measurement, step_item: StepItem, channel: Channel
):
    measurement.add_channel(channel)
    measurement.add_step_item(step_item)
    step_item_int = measurement.get_step_item(step_item.name)
    last_index = step_item_int.index
    assert last_index != 3
    measurement.set_step_item_index(step_item.name, 3)
    assert step_item_int.index == 3

    # invalid name
    with pytest.raises(ValueError):
        measurement.set_step_item_index("invalid_name", 0)


def test_empty_measurement():
    measurement = Measurement.create_empty()
    assert measurement.calculate_values() == {}
    assert measurement.channels == []
    assert measurement.step_items == []
    assert measurement.relations == []
    assert measurement.log_channels == []


def test_invalid_step_item_error():
    with pytest.raises(ValueError):
        Measurement(
            channels=[],
            step_items=[
                StepItem(
                    name="test_name",
                    config=StepConfig(),
                    ranges=[StepRange(start=0, stop=1, step_count=2)],
                    index=0,
                )
            ],
            relations=[],
            log_channels=[],
        )


def test_invalid_relation_error():
    with pytest.raises(ValueError):
        Measurement(
            channels=[],
            step_items=[],
            relations=[
                RelationSettings(
                    name="test_name2",
                    config=StepConfig(),
                    equation="-x",
                    enable=True,
                    parameters=[
                        RelationParameters(variable="x", source_name="test_name1")
                    ],
                )
            ],
            log_channels=[],
        )


def test_measurement_log_shape(measurement: Measurement):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert measurement.log_shape == (7,)


def test_empty_measurement_log_shape():
    measurement = Measurement.create_empty()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert measurement.log_shape == (1,)


def test_measurement_swept_shape(measurement: Measurement):
    assert measurement.swept_shape == (7,)


def test_empty_measurement_swept_shape():
    measurement = Measurement.create_empty()
    assert measurement.swept_shape == (1,)


def test_measurement_log_shapes(measurement: Measurement):
    assert measurement.log_shapes == {"test_name3": (7,)}


def test_meshgrid():
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    c = np.array([6, 7, 8])

    arrs = meshgrid(a, b, c, indicies=[0, 1, 0])

    assert np.allclose(arrs[0].T.flatten(), np.array([1, 2, 3, 1, 2, 3]))
    assert np.allclose(arrs[1].T.flatten(), np.array([4, 4, 4, 5, 5, 5]))
    assert np.allclose(arrs[2].T.flatten(), np.array([6, 7, 8, 6, 7, 8]))

    arrs = meshgrid(a, c, b, indicies=[0, 0, 1])

    assert np.allclose(arrs[0].T.flatten(), np.array([1, 2, 3, 1, 2, 3]))
    assert np.allclose(arrs[1].T.flatten(), np.array([6, 7, 8, 6, 7, 8]))
    assert np.allclose(arrs[2].T.flatten(), np.array([4, 4, 4, 5, 5, 5]))


def test_meshgrid_2():
    a = np.array([1, 2])
    b = np.array([3, 4])
    c = np.array([5, 6])

    arrs = meshgrid(a, b, c, indicies=[0, 1, 2])
    assert np.allclose(arrs, np.meshgrid(a, b, c, indexing="ij"))

    assert np.allclose(arrs[0].T.flatten(), np.array([1, 2, 1, 2, 1, 2, 1, 2]))
    assert np.allclose(arrs[1].T.flatten(), np.array([3, 3, 4, 4, 3, 3, 4, 4]))
    assert np.allclose(arrs[2].T.flatten(), np.array([5, 5, 5, 5, 6, 6, 6, 6]))

    arrs = meshgrid(a, b, c, indicies=[2, 1, 0])

    assert np.allclose(arrs[0].T.flatten(), np.array([1, 1, 1, 1, 2, 2, 2, 2]))
    assert np.allclose(arrs[1].T.flatten(), np.array([3, 3, 4, 4, 3, 3, 4, 4]))
    assert np.allclose(arrs[2].T.flatten(), np.array([5, 6, 5, 6, 5, 6, 5, 6]))

    arrs = meshgrid(a, b, c, indicies=[0, 1, 0])

    assert np.allclose(arrs[0].T.flatten(), np.array([1, 2, 1, 2]))
    assert np.allclose(arrs[1].T.flatten(), np.array([3, 3, 4, 4]))
    assert np.allclose(arrs[2].T.flatten(), np.array([5, 6, 5, 6]))

    arrs = meshgrid(a, b, c, indicies=[0, 0, 1])

    assert np.allclose(arrs[0].T.flatten(), np.array([1, 2, 1, 2]))
    assert np.allclose(arrs[1].T.flatten(), np.array([3, 4, 3, 4]))
    assert np.allclose(arrs[2].T.flatten(), np.array([5, 5, 6, 6]))

    arrs = meshgrid(a, b, c, indicies=[1, 0, 0])

    assert np.allclose(arrs[0].T.flatten(), np.array([1, 1, 2, 2]))
    assert np.allclose(arrs[1].T.flatten(), np.array([3, 4, 3, 4]))
    assert np.allclose(arrs[2].T.flatten(), np.array([5, 6, 5, 6]))

    arrs = meshgrid(a, b, c, indicies=[1, 1, 0])

    assert np.allclose(arrs[0].T.flatten(), np.array([1, 1, 2, 2]))
    assert np.allclose(arrs[1].T.flatten(), np.array([3, 3, 4, 4]))
    assert np.allclose(arrs[2].T.flatten(), np.array([5, 6, 5, 6]))

    arrs = meshgrid(a, b, c, indicies=[1, 0, 1])

    assert np.allclose(arrs[0].T.flatten(), np.array([1, 1, 2, 2]))
    assert np.allclose(arrs[1].T.flatten(), np.array([3, 4, 3, 4]))
    assert np.allclose(arrs[2].T.flatten(), np.array([5, 5, 6, 6]))

    arrs = meshgrid(a, b, c, indicies=[0, 1, 1])

    assert np.allclose(arrs[0].T.flatten(), np.array([1, 2, 1, 2]))
    assert np.allclose(arrs[1].T.flatten(), np.array([3, 3, 4, 4]))
    assert np.allclose(arrs[2].T.flatten(), np.array([5, 5, 6, 6]))

    arrs = meshgrid(a, b, c, indicies=[0, 0, 0])

    assert np.allclose(arrs[0].T.flatten(), np.array([1, 2]))
    assert np.allclose(arrs[1].T.flatten(), np.array([3, 4]))
    assert np.allclose(arrs[2].T.flatten(), np.array([5, 6]))


def test_measurement_step_item_index_values(measurement: Measurement):
    measurement.remove_step_item("test_name1")
    measurement.add_step("test_name1", [0, 1], 0)
    measurement.add_step("test_name2", [2, 3], 0)
    measurement.add_step("test_name3", [4, 5], 1)

    values = measurement.calculate_values()
    assert np.allclose(np.array(values["test_name1"]), np.array([0, 1, 0, 1]))
    assert np.allclose(np.array(values["test_name2"]), np.array([2, 3, 2, 3]))
    assert np.allclose(np.array(values["test_name3"]), np.array([4, 4, 5, 5]))


def test_measurement_step_item_strings(measurement: Measurement):
    measurement.remove_step_item("test_name1")
    measurement.add_step("test_name1", [0, 1], 0)
    measurement.add_step("test_name2", ["a", "b"], 0)
    measurement.add_step("test_name3", ["c", "d"], 1)

    values = measurement.calculate_values()
    assert np.allclose(np.array(values["test_name1"]), np.array([0, 1, 0, 1]))
    assert np.all(values["test_name2"] == np.array(["a", "b", "a", "b"]))
    assert np.all(values["test_name3"] == np.array(["c", "c", "d", "d"]))

    values = measurement.calculate_values(expand_arrays=False)
    assert np.allclose(np.array(values["test_name1"]), np.array([0, 1]))
    assert np.all(values["test_name2"] == np.array(["a", "b"]))
    assert np.all(values["test_name3"] == np.array(["c", "d"]))


def test_measurement_step_item_lists(measurement: Measurement):
    measurement.remove_step_item("test_name1")
    measurement.add_step("test_name1", [0, 1], 0)
    measurement.add_step("test_name2", [["a1", "a1"], ["b1", "b2"]], 0)
    measurement.add_step("test_name3", [[2, 3], [4, 5]], 1)

    values = measurement.calculate_values()
    assert np.allclose(np.array(values["test_name1"]), np.array([0, 1, 0, 1]))
    assert np.all(
        values["test_name2"]
        == np.array([["a1", "a1"], ["b1", "b2"], ["a1", "a1"], ["b1", "b2"]])
    )
    assert np.all(values["test_name3"] == np.array([[2, 3], [2, 3], [4, 5], [4, 5]]))

    values = measurement.calculate_values(expand_arrays=False)
    assert np.allclose(np.array(values["test_name1"]), np.array([0, 1]))
    assert np.all(values["test_name2"] == np.array([["a1", "a1"], ["b1", "b2"]]))
    assert np.all(values["test_name3"] == np.array([[2, 3], [4, 5]]))

    measurement.remove_log("test_name3")
    data = Data.from_measurement(measurement)
    assert data.get_dataset("test_name1").memory.to_array().shape == (2,)
    assert data.get_dataset("test_name2").memory.to_array().shape == (2, 2)
    assert data.get_dataset("test_name3").memory.to_array().shape == (2, 2)


def test_measurement_step_item_lists_uneven(measurement: Measurement):
    measurement.remove_step_item("test_name1")
    measurement.add_step("test_name1", [0, 1], 0)
    measurement.add_step("test_name2", [["a1", "a1"], ["b1", "b2", "b3"]], 0)
    measurement.add_step("test_name3", [[2, 3, 6], [4, 5]], 1)

    values = measurement.calculate_values()
    assert np.allclose(np.array(values["test_name1"]), np.array([0, 1, 0, 1]))
    assert values["test_name2"] == [
        ["a1", "a1"],
        ["b1", "b2", "b3"],
        ["a1", "a1"],
        ["b1", "b2", "b3"],
    ]

    assert values["test_name3"] == [[2, 3, 6], [2, 3, 6], [4, 5], [4, 5]]

    values = measurement.calculate_values(expand_arrays=False)
    assert np.allclose(np.array(values["test_name1"]), np.array([0, 1]))
    assert values["test_name2"] == [["a1", "a1"], ["b1", "b2", "b3"]]
    assert values["test_name3"] == [[2, 3, 6], [4, 5]]

    measurement.remove_log("test_name3")
    data = Data.from_measurement(measurement)
    assert data.get_dataset("test_name1").memory.to_array().shape == (2,)
    assert data.get_dataset("test_name2").memory.to_array().shape == (2,)
    assert data.get_dataset("test_name3").memory.to_array().shape == (2,)


def test_meshgrid_error():
    with pytest.raises(ValueError):
        meshgrid([0, 1], [2, 3], indicies=[0, 1, 2])


def test_measurement_step_item_index_None():
    Measurement(
        channels=[
            Channel(
                name="test_name1",
                instrument_identity="test_instrument1",
                instrument_setting_name="test_setting1",
                unit_physical="Phys_unit1",
            )
        ],
        step_items=[
            StepItem(
                name="test_name1",
                config=StepConfig(),
                ranges=[
                    StepRange(start=0.0, stop=3.0, step_count=4),
                    StepRange(start=2.0, stop=0.0, step_count=3),
                ],
                index=0,
            )  # type: ignore
        ],
        relations=[],
        log_channels=[],
    )


def test_measurement_step_item_from_dict():
    Measurement(
        channels=[
            Channel(
                name="test_name1",
                instrument_identity="test_instrument1",
                instrument_setting_name="test_setting1",
                unit_physical="Phys_unit1",
            )
        ],
        step_items=[
            {
                "name": "test_name1",
                "config": {},
                "ranges": [{"start": 0.0, "stop": 3.0, "step_count": 4}],
                "index": None,
            }  # type: ignore
        ],
        relations=[],
        log_channels=[],
    )


@pytest.fixture(name="channels")
def channels_fixtures():
    yield [
        Channel(
            name="test_name1",
            instrument_identity="test_instrument1",
            instrument_setting_name="test_setting1",
            unit_physical="Phys_unit1",
        ),
        Channel(
            name="test_name2",
            instrument_identity="test_instrument2",
            instrument_setting_name="test_setting2",
            unit_physical="Phys_unit2",
        ),
        Channel(
            name="test_name3",
            instrument_identity="test_instrument3",
            instrument_setting_name="test_setting3",
            unit_physical="Phys_unit3",
        ),
        Channel(
            name="test_name4",
            instrument_identity="test_instrument4",
            instrument_setting_name="test_setting4",
            unit_physical="Phys_unit4",
        ),
        Channel(
            name="test_name5",
            instrument_identity="test_instrument5",
            instrument_setting_name="test_setting5",
            unit_physical="Phys_unit5",
        ),
    ]


def test_measurement_step_item_indicies(channels: list[Channel]):
    meas = Measurement(
        channels=channels,
        log_channels=[],
        relations=[],
        step_items=[
            StepItem(name="test_name1", config=StepConfig(), ranges=[], index=0),
            StepItem(name="test_name2", config=StepConfig(), ranges=[]),
            StepItem(name="test_name3", config=StepConfig(), ranges=[], index=3),
            StepItem(name="test_name4", config=StepConfig(), ranges=[], index=-2),
            StepItem(name="test_name5", config=StepConfig(), ranges=[]),
        ],  # type: ignore
    )
    assert meas.get_step_item("test_name1").index == 0
    assert meas.get_step_item("test_name2").index == 4
    assert meas.get_step_item("test_name3").index == 3
    assert meas.get_step_item("test_name4").index == -2
    assert meas.get_step_item("test_name5").index == 5


def test_measurement_step_item_no_indicies(channels: list[Channel]):
    meas = Measurement(
        channels=channels,
        log_channels=[],
        relations=[],
        step_items=[
            StepItem(name="test_name1", config=StepConfig(), ranges=[]),
            StepItem(name="test_name2", config=StepConfig(), ranges=[]),
            StepItem(name="test_name3", config=StepConfig(), ranges=[]),
            StepItem(name="test_name4", config=StepConfig(), ranges=[]),
            StepItem(name="test_name5", config=StepConfig(), ranges=[]),
        ],  # type: ignore
    )
    assert meas.get_step_item("test_name1").index == 0
    assert meas.get_step_item("test_name2").index == 1
    assert meas.get_step_item("test_name3").index == 2
    assert meas.get_step_item("test_name4").index == 3
    assert meas.get_step_item("test_name5").index == 4


def test_measurement_step_item_negative_indices(channels: list[Channel]):
    meas = Measurement(
        channels=channels,
        log_channels=[],
        relations=[],
        step_items=[
            StepItem(name="test_name1", config=StepConfig(), ranges=[]),
            StepItem(name="test_name2", config=StepConfig(), ranges=[]),
            StepItem(name="test_name3", config=StepConfig(), ranges=[], index=-100),
            StepItem(name="test_name4", config=StepConfig(), ranges=[]),
            StepItem(name="test_name5", config=StepConfig(), ranges=[]),
        ],  # type: ignore
    )
    assert meas.get_step_item("test_name1").index == -99
    assert meas.get_step_item("test_name2").index == -98
    assert meas.get_step_item("test_name3").index == -100
    assert meas.get_step_item("test_name4").index == -97
    assert meas.get_step_item("test_name5").index == -96


def test_calculate_values_relations(channels: list[Channel]):
    meas = Measurement(
        channels=channels,
        log_channels=[],
        relations=[
            RelationSettings(
                name="test_name3",
                config=StepConfig(),
                equation="x+y",
                enable=True,
                parameters=[
                    RelationParameters(variable="x", source_name="test_name1"),
                    RelationParameters(variable="y", source_name="test_name2"),
                ],
            )
        ],
        step_items=[
            StepItem(
                name="test_name1",
                config=StepConfig(),
                ranges=[StepRange(start=0, stop=1, step_count=2)],
                index=0,
            ),
            StepItem(
                name="test_name2",
                config=StepConfig(),
                ranges=[StepRange(start=2, stop=4, step_count=3)],
                index=4,
            ),
        ],
    )
    values = meas.calculate_values()
    assert np.allclose(np.array(values["test_name1"]), np.array([0, 1, 0, 1, 0, 1]))
    assert np.allclose(np.array(values["test_name2"]), np.array([2, 2, 3, 3, 4, 4]))
    assert np.allclose(np.array(values["test_name3"]), np.array([2, 3, 3, 4, 4, 5]))

    values = meas.calculate_values(expand_arrays=False)
    assert np.allclose(np.array(values["test_name1"]), np.array([0, 1]))
    assert np.allclose(np.array(values["test_name2"]), np.array([2, 3, 4]))
    assert np.allclose(np.array(values["test_name3"]), np.array([[2, 3, 4], [3, 4, 5]]))


def test_measurement_step_item_mismatching_length(measurement: Measurement):
    measurement.add_step_item(
        StepItem(
            name="test_name2",
            config=StepConfig(),
            ranges=[
                StepRange(start=0.0, stop=3.0, step_count=4),
                StepRange(start=2.0, stop=0.0, step_count=4),
            ],
            index=0,
        )  # type: ignore
    )
    with pytest.raises(ValueError):
        measurement.swept_shape


def test_json_serial_deserialization(measurement: Measurement):
    new_measurement = Measurement.model_validate_json(measurement.model_dump_json())
    assert new_measurement == measurement
