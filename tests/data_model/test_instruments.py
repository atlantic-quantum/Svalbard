import json
from enum import Enum

import bson
import numpy as np
import pytest
from svalbard.data_model.instruments import (
    InstrumentConnection,
    InstrumentModel,
    InstrumentSetting,
    SettingType,
    SupportsInstrumentModel,
)

val_and_type = [
    (1.0, float, SettingType.FLOAT, "V"),
    (2, int, SettingType.INT, "A"),
    (False, bool, SettingType.BOOL, ""),
]


@pytest.mark.parametrize("value, ttype, setting_type, unit", val_and_type)
def test_instrument_setting_serialiastion(value, ttype, setting_type, unit):
    instr_setting = InstrumentSetting(
        name="test_name", value=value, dtype=setting_type, unit=unit
    )

    assert instr_setting.name == "test_name"
    assert instr_setting.value == value
    assert type(instr_setting.value) == ttype
    assert instr_setting.dtype == setting_type
    assert instr_setting.unit == unit

    loaded_instr_setting = InstrumentSetting(**instr_setting.dict())

    assert loaded_instr_setting.name == "test_name"
    assert loaded_instr_setting.value == value
    assert type(loaded_instr_setting.value) == ttype
    assert loaded_instr_setting.dtype == setting_type
    assert loaded_instr_setting.unit == unit


@pytest.mark.parametrize("ttype", [SettingType.COMPLEX, SettingType.NONE, "complex"])
def test_instrument_setting_serialiastion_complex(ttype):
    instr_setting = InstrumentSetting(
        name="test_name", value=1.0 + 2.0j, dtype=ttype, unit="V"
    )

    assert instr_setting.name == "test_name"
    assert instr_setting.value == "1.0+2.0j"
    assert type(instr_setting.value) == str
    assert instr_setting.dtype == SettingType.COMPLEX
    assert instr_setting.unit == "V"

    loaded_instr_setting = InstrumentSetting(**instr_setting.dict())

    assert loaded_instr_setting.name == "test_name"
    assert loaded_instr_setting.value == "1.0+2.0j"
    assert type(loaded_instr_setting.value) == str
    assert loaded_instr_setting.dtype == SettingType.COMPLEX
    assert loaded_instr_setting.unit == "V"


class test_enum(Enum):
    enum1 = 1
    enum2 = 2


class test_enum_text(Enum):
    enum1 = "enum1"
    enum2 = "enum2"


val_and_type_enum = [
    (test_enum.enum1, int, SettingType.ENUM, "", 1),
    (test_enum_text.enum1, str, SettingType.ENUM, "", "enum1"),
    (2, int, SettingType.ENUM, "", 2),
    ("enum2", str, SettingType.ENUM, "", "enum2"),
]


@pytest.mark.parametrize("value, ttype, setting_type, unit, e_value", val_and_type_enum)
def test_instrument_setting_enum(value, ttype, setting_type, unit, e_value):
    instr_setting = InstrumentSetting(
        name="test_name", value=value, dtype=setting_type, unit=unit
    )

    assert instr_setting.value == e_value
    assert type(instr_setting.value) == ttype
    assert instr_setting.dtype == SettingType.ENUM
    assert instr_setting == InstrumentSetting(**instr_setting.dict())


val_and_enum_no_type = [
    (test_enum.enum1, "", 1),
    (test_enum_text.enum1, "", "enum1"),
]


@pytest.mark.parametrize("value, unit, e_value", val_and_enum_no_type)
def test_instrument_setting_enum_no_type(value, unit, e_value):
    instr_setting = InstrumentSetting(name="test_name", value=value, unit=unit)

    assert instr_setting.value == e_value
    assert instr_setting.dtype == SettingType.ENUM
    assert instr_setting == InstrumentSetting(**instr_setting.dict())


@pytest.mark.parametrize("value, ttype, setting_type, unit", val_and_type)
def test_instrument_setting_stype_none(value, ttype, setting_type, unit):
    instr_setting = InstrumentSetting(
        name="test_name", value=value, dtype=SettingType.NONE, unit=unit
    )
    assert instr_setting.value == value
    assert type(value) == ttype
    assert type(instr_setting.value) == ttype
    assert instr_setting.dtype == setting_type


@pytest.mark.parametrize("value, ttype, setting_type, unit", val_and_type)
def test_instrument_setting_no_type(value, ttype, setting_type, unit):
    instr_setting = InstrumentSetting(name="test_name", value=value)

    assert instr_setting.value == value
    assert type(instr_setting.value) == ttype
    assert instr_setting.dtype == setting_type


def test_instrument_setting_value_zero():
    instr_setting = InstrumentSetting(name="test_name", value=0)

    assert instr_setting.value == 0


@pytest.mark.xfail
def test_instrument_setting_value_zero_numpy_int():
    # todo fix this
    InstrumentSetting(name="test_name", value=np.int64(0))


def test_instrument_setting_str():
    InstrumentSetting(name="test_name", value="test_value")


@pytest.mark.xfail
def test_instrument_setting_array():
    InstrumentSetting(name="test_name", value=np.asarray([1, 2, 3]))


@pytest.mark.xfail
def test_instrument_setting_list():
    InstrumentSetting(name="test_name", value=[1, 2, 3])


def test_instrument_setting_stype_none_error():
    with pytest.raises(ValueError):
        InstrumentSetting(
            name="test_name", value={"dict": 1}, dtype=SettingType.FLOAT, unit=""
        )

    with pytest.raises(ValueError):
        InstrumentSetting(
            name="test_name", value={"dict": 1}, dtype=SettingType.NONE, unit=""
        )


def test_supports_instrument_model():
    """
    test that the SupportsInstrumentModel class raises NotImplementedError
    """

    class test_sim:
        def __init__(self):
            pass

        def to_instrument_settings(self) -> list[InstrumentSetting]:
            return []

        def from_instrument_model(self, instrument: InstrumentModel):
            self.instrument = instrument

        def to_instrument_model(self) -> InstrumentModel:
            return InstrumentModel()

    tsim = test_sim()

    with pytest.raises(NotImplementedError):
        SupportsInstrumentModel.to_insturment_settings(tsim)

    with pytest.raises(NotImplementedError):
        SupportsInstrumentModel.to_instrument_model(tsim)

    with pytest.raises(NotImplementedError):
        SupportsInstrumentModel.from_instrument_model(InstrumentModel())


def test_instrument_connection_bson():
    instr_connection = InstrumentConnection()
    bson_encoded = bson.BSON.encode(json.loads(instr_connection.json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_instr_connection = InstrumentConnection(**bson_decoded)
    assert new_instr_connection == instr_connection


def test_instrument_connection_json():
    instr_connection = InstrumentConnection()
    conn_json = json.loads(instr_connection.json())
    assert instr_connection == InstrumentConnection(**conn_json)


_val_and_setting_type = [
    (1, SettingType.INT),
    (1.0, SettingType.FLOAT),
    (False, SettingType.BOOL),
    (test_enum.enum1, SettingType.ENUM),
    ("test", SettingType.STR),
]


@pytest.mark.parametrize("value, setting_type", _val_and_setting_type)
def test_instrument_setting_bson(value, setting_type):
    instr_setting = InstrumentSetting(name="test_name", value=value, dtype=setting_type)
    bson_encoded = bson.BSON.encode(json.loads(instr_setting.json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_instr_setting = InstrumentSetting(**bson_decoded)
    assert new_instr_setting == instr_setting


@pytest.mark.parametrize("value, setting_type", _val_and_setting_type)
def test_instrument_setting_json(value, setting_type):
    instr_setting = InstrumentSetting(name="test_name", value=value, dtype=setting_type)
    setting_json = json.loads(instr_setting.json())
    assert instr_setting == InstrumentSetting(**setting_json)
