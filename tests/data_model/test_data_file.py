"""Tests of DataFile pydantic models."""

import datetime
import json
import time

import bson
import numpy as np
import pytest

from svalbard.data_model.data_file import (
    Data,
    DataFile,
    MaskMetaData,
    MeasurementHandle,
    MetaData,
    _flatten_list,
)
from svalbard.data_model.instruments import (
    Drain,
    InstrumentModel,
    InstrumentSetting,
    SettingType,
)
from svalbard.data_model.measurement import Measurement, RangeTypes, StepItem, StepRange


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

    for dataset in data.datasets:
        assert data.get_dataset(dataset.name) == dataset


def test_data_get_dataset_error(data: Data):
    with pytest.raises(ValueError):
        data.get_dataset("nonexistent_dataset")


def test_metadata():
    mdat = MetaData(
        name="test_name",
        user="test_user",
        station="test_station",
        cooldown=datetime.datetime(2023, 9, 12),
    )
    l_mdat = MetaData(**mdat.model_dump())
    assert mdat == l_mdat


def test_datafile_bson(metadata: MetaData):
    """Test that DataFile can be converted to bson"""
    bson_encoded = bson.BSON.encode(json.loads(metadata.model_dump_json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_metadata = MetaData(**bson_decoded)
    assert new_metadata == metadata


def test_datafile_add_step_values(metadata: MetaData):
    metadata.add_step("test_hardware", "test_name0", [0, 1, 2])
    assert len(metadata.measurement.channels) == 1
    assert metadata.measurement.channels[0].name == "test_hardware___test_name0"
    assert len(metadata.measurement.step_items) == 1
    assert metadata.measurement.step_items[0].name == "test_hardware___test_name0"


def test_datafile_add_step(metadata: MetaData):
    metadata.add_step("test_hardware", "test_name0", 3.0)
    assert len(metadata.measurement.channels) == 0
    assert len(metadata.measurement.step_items) == 0
    instr_model = metadata.get_instrument_model("test_hardware")
    instr_setting = instr_model.get_setting("test_name0")
    assert instr_setting.value == 3.0


def test_mask_metadata_bson(mask_metadata: MaskMetaData):
    """Test that DataFile using MaskMetaData can be converted to bson"""
    bson_encoded = bson.BSON.encode(json.loads(mask_metadata.model_dump_json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_metadata = MaskMetaData(**bson_decoded)
    assert new_metadata == mask_metadata


def test_metadata_datetime():
    mdat1 = MetaData(name="test_name1", user="test_user", station="test_station")
    time.sleep(0.001)
    mdat2 = MetaData(name="test_name2", user="test_user", station="test_station")

    assert mdat1.date < mdat2.date


def test_data_from_measurement(measurement: Measurement):
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


def test_data_from_measurement_no_step_items(measurement: Measurement):
    """Test that Data can be created from a measurement without step items"""
    for step_item in measurement.step_item_names:
        measurement.remove_step_item(step_item)
    data_from_measurement = Data.from_measurement(measurement)
    test_data(data_from_measurement)

    assert len(data_from_measurement.datasets) == 1
    assert data_from_measurement.datasets[0].name == "test_name3"
    assert data_from_measurement.datasets[0].memory.shape == (1,)
    assert data_from_measurement.datasets[0].memory.dtype == "complex128"
    assert data_from_measurement.datasets[0].memory.to_array().shape == (1,)
    assert data_from_measurement.datasets[0].memory.to_array().dtype == np.complex128


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


def test_add_instrument_model(metadata: MetaData):
    instr_model = InstrumentModel(identity="adding_model")
    metadata.add_instrument_model(instr_model)
    assert "adding_model" in metadata.instruments

    with pytest.raises(ValueError):
        metadata.add_instrument_model(instr_model)


def test_datafile_add_new_step(metadata: MetaData):
    instr_model1 = InstrumentModel(identity="new_settings", allow_new_settings=True)
    assert not instr_model1.has_setting("new_setting")
    metadata.add_instrument_model(instr_model1)
    metadata.add_step("new_settings", "new_setting", [0, 1, 2])
    assert instr_model1.has_setting("new_setting")

    instr_model2 = InstrumentModel(identity="no_new_settings", allow_new_settings=False)
    assert not instr_model2.has_setting("new_setting2")
    metadata.add_instrument_model(instr_model2)
    with pytest.raises(ValueError):
        metadata.add_step("no_new_settings", "new_setting", [0, 1, 2])
    assert not instr_model2.has_setting("new_setting2")

    with pytest.raises(ValueError):
        metadata.add_step("wrong_instrument", "new_setting", 3.0)


def test_datafile_configure_new_step(metadata: MetaData):
    # ! remove once ParamsGenerators are deprecated
    instr_model1 = InstrumentModel(identity="new_settings", allow_new_settings=True)
    assert not instr_model1.has_setting("new_setting")
    metadata.add_instrument_model(instr_model1)
    metadata.configure_setting("new_settings", "new_setting", [0, 1, 2])
    assert instr_model1.has_setting("new_setting")

    instr_model2 = InstrumentModel(identity="no_new_settings", allow_new_settings=False)
    metadata.add_instrument_model(instr_model2)
    with pytest.raises(ValueError):
        metadata.configure_setting("no_new_settings", "new_setting", [0, 1, 2])
    assert not instr_model2.has_setting("new_setting2")

    with pytest.raises(ValueError):
        metadata.add_step("wrong_instrument", "new_setting", 3.0)


def test_json_serial_deserialization_meta_data(
    metadata: MetaData, measurement: Measurement
):
    metadata.measurement = measurement
    new_metadata = MetaData.model_validate_json(metadata.model_dump_json())
    assert new_metadata == metadata


def test_json_serial_deserialization_data_file(
    data_file: DataFile, measurement: Measurement
):
    data_file.metadata.measurement = measurement
    new_data_file = DataFile.model_validate_json(data_file.model_dump_json())
    assert new_data_file == data_file


@pytest.mark.parametrize(
    "input, converted",
    [
        (np.array([0, 1, 2]), [0, 1, 2]),
        ([0, 1, 2], [0, 1, 2]),
        (1, [1]),
        ("test", ["test"]),
        (np.float32(1.0), [1.0]),
        (np.uint32(1), [1]),
        ([np.uint(2), np.uint(1)], [2, 1]),
        ((np.uint(2), np.uint(1)), [2, 1]),
    ],
)
def test_add_step_type_conversion(metadata: MetaData, input, converted):
    metadata.add_step("test_hardware", "test_name0", input)
    instr_model = metadata.get_instrument_model("test_hardware")
    if len(converted) > 1:
        assert np.allclose(
            metadata.measurement.step_items[0].ranges[0].values,  # type: ignore
            converted,
        )
    else:
        assert instr_model.get_setting("test_name0").value == converted[0]


def test_add_step_item(metadata: MetaData):
    step_item = StepItem(
        name="test_hardware___test_name0",
        ranges=[StepRange(values=[0, 1, 2], range_type=RangeTypes.VALUES)],
    )
    metadata._add_step(step_item)
    assert len(metadata.measurement.step_items) == 1
    assert metadata.measurement.step_items[0] == step_item
    assert step_item.name in metadata.measurement.step_item_names


def test_add_step_no_values_error(metadata: MetaData):
    with pytest.raises(ValueError):
        metadata._add_step("test_hardware___test_name0", None)


def test_add_log_new_channel(metadata: MetaData):
    metadata.add_log("test_hardware___test_name4")
    assert "test_hardware___test_name4" in metadata.measurement.log_channels_names


def test_create_channel_instrument_error(metadata: MetaData):
    with pytest.raises(ValueError):
        metadata._create_channel("wrong_instrument___test_name0")


def test_metadata_update_setting(metadata: MetaData):
    instr_model = metadata.get_instrument_model("test_hardware")
    instr_setting = instr_model.get_setting("test_name0")
    assert instr_setting.unit == ""
    metadata.update_setting("test_hardware", "test_name0", 3.0, "new_unit")
    assert instr_setting.value == 3.0
    assert instr_setting.unit == "new_unit"

    with pytest.raises(ValueError):
        metadata.update_setting("wrong_instrument", "test_name1", 3.0)


def test_metadata_add_setting(metadata: MetaData):
    instr_model1 = InstrumentModel(identity="new_settings", allow_new_settings=True)
    metadata.add_instrument_model(instr_model1)
    metadata.add_setting("new_settings", "test_name0", 3.0)
    instr_setting = metadata.get_instrument_model("new_settings").get_setting(
        "test_name0"
    )
    assert instr_setting.value == 3.0

    with pytest.raises(ValueError):
        metadata.add_setting("wrong_hardware", "test_name0", 3.0)


def test_metadata_get_channel_name(metadata: MetaData):
    metadata.add_step("test_hardware", "test_name0", [0, 1, 2])
    channel = metadata.measurement.channels[0]
    assert (
        metadata.get_channel_name(
            channel.instrument_identity, channel.instrument_setting_name
        )
        == channel.name
    )
    instrument_id = "wrong_instrument"
    setting_name = "test_name0"
    assert (
        metadata.get_channel_name(instrument_id, setting_name)
        == f"{instrument_id}___{setting_name}"
    )


def test_metadata_get_instrument_model_error(metadata: MetaData):
    with pytest.raises(ValueError):
        metadata.get_instrument_model("wrong_instrument")


def test_metadata_configure_setting_step(metadata: MetaData):
    metadata.configure_setting("test_hardware", "test_name0", [3.0, 4.0])
    channel_name = metadata.get_channel_name("test_hardware", "test_name0")
    step_item = metadata.measurement.get_step_item(channel_name)
    assert np.allclose(step_item.calculate_values(), [3.0, 4.0])  # type: ignore

    metadata.measurement.remove_step_item(channel_name)
    metadata.configure_setting("test_hardware", "test_name0", np.array([5.0, 6.0]))
    step_item = metadata.measurement.get_step_item(channel_name)
    assert np.allclose(step_item.calculate_values(), [5.0, 6.0])  # type: ignore

    metadata.measurement.remove_step_item(channel_name)
    metadata.configure_setting("test_hardware", "test_name0", (6.0, 7.0))
    step_item = metadata.measurement.get_step_item(channel_name)
    assert np.allclose(step_item.calculate_values(), [6.0, 7.0])  # type: ignore


def test_metadata_configure_setting_no_step(metadata: MetaData):
    metadata.configure_setting("test_hardware", "test_name0", 3.0)
    instrument = metadata.get_instrument_model("test_hardware")
    assert instrument.get_setting("test_name0").value == 3.0

    metadata.configure_setting("test_hardware", "test_name0", [4.0])
    assert instrument.get_setting("test_name0").value == 4.0

    metadata.configure_setting("test_hardware", "test_name0", np.array([5.0]))
    assert instrument.get_setting("test_name0").value == 5.0

    metadata.configure_setting("test_hardware", "test_name0", (6.0,))
    assert instrument.get_setting("test_name0").value == 6.0


def test_metadata_configure_setting_list_no_step(metadata: MetaData):
    instrument = metadata.get_instrument_model("test_hardware")
    instrument.settings["list_setting"] = InstrumentSetting(
        name="list_setting", value=[0, 1, 2], dtype=SettingType.LIST_INT
    )
    metadata.configure_setting(
        "test_hardware", "list_setting", [3, 4], list_setting=True
    )
    assert instrument.get_setting("list_setting").value == [3, 4]


def test_metadata_configure_setting_list_step(metadata: MetaData):
    instrument = metadata.get_instrument_model("test_hardware")
    instrument.settings["list_setting"] = InstrumentSetting(
        name="list_setting", value=[0, 1, 2], dtype=SettingType.LIST_INT
    )
    metadata.configure_setting(
        "test_hardware", "list_setting", [[3, 4], [1, 2, 3]], list_setting=True
    )
    channel_name = metadata.get_channel_name("test_hardware", "list_setting")
    step_item = metadata.measurement.get_step_item(channel_name)
    assert step_item.calculate_values()[0] == [3, 4]
    assert step_item.calculate_values()[1] == [1, 2, 3]


def test_flatten_list():
    assert _flatten_list([1, 2, 3]) == [1, 2, 3]
    assert _flatten_list([[1], [2, 3]]) == [1, 2, 3]
    assert _flatten_list(3) == [3]
    assert _flatten_list("test") == ["test"]


def test_adding_drains(metadata: MetaData):
    model_setting = InstrumentSetting(
        name="model", value="test", dtype=SettingType.MODEL
    )
    drain_instrument = InstrumentModel(
        identity="drain_model", settings={"model": model_setting}
    )
    metadata.add_instrument_model(drain_instrument)

    object_setting = InstrumentSetting(
        name="source_model", value="test", dtype=SettingType.OBJECT
    )
    source_instrument = InstrumentModel(
        identity="source_model", settings={"source_model": object_setting}
    )
    metadata.add_instrument_model(source_instrument)

    assert object_setting.drains == []
    metadata.configure_setting("drain_model", "model", "source_model")
    assert object_setting.drains == [Drain(identity="drain_model", setting="model")]
