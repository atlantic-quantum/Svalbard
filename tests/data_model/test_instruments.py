import json
from enum import Enum
from types import NoneType

import bson
import numpy as np
import pytest

from svalbard.data_model._setting_types import NoneSettingType
from svalbard.data_model.instruments import (
    Drain,
    InstrumentConnection,
    InstrumentModel,
    InstrumentSetting,
    SettingType,
    SupportsInstrumentModel,
)

val_and_type = [
    (1.0, float, SettingType.FLOAT, "V", "1 V"),
    (2, int, SettingType.INT, "A", "2 A"),
    (1, int, SettingType.INT, "A", "1 A"),
    (0, int, SettingType.INT, "A", "0 A"),
    (True, bool, SettingType.BOOL, "", "On"),
    (False, bool, SettingType.BOOL, "", "Off"),
    ("test", str, SettingType.STR, "", "test"),
    ([1.0, 2.0], list, SettingType.LIST_FLOAT, "V", "['1 V', '2 V']"),
    ([2, 4], list, SettingType.LIST_INT, "A", "['2 A', '4 A']"),
    ([0, 1], list, SettingType.LIST_INT, "A", "['0 A', '1 A']"),
    ([False, True], list, SettingType.LIST_BOOL, "", "['Off', 'On']"),
    (["test", "more_test"], list, SettingType.LIST_STR, "", "['test', 'more_test']"),
    (None, NoneType, SettingType.NONE, "", ""),
    ("object_name", str, SettingType.MODEL, "", "object_name"),
]


@pytest.mark.parametrize("value, ttype, setting_type, unit, str_rep", val_and_type)
def test_instrument_setting_serialiastion(value, ttype, setting_type, unit, str_rep):
    instr_setting = InstrumentSetting(
        name="test_name", value=value, dtype=setting_type, unit=unit
    )

    assert instr_setting.name == "test_name"
    assert instr_setting.value == value
    assert isinstance(instr_setting.value, ttype)
    assert instr_setting.dtype == setting_type
    assert instr_setting.unit == unit

    loaded_instr_setting = InstrumentSetting(**instr_setting.model_dump())

    assert loaded_instr_setting.name == "test_name"
    assert loaded_instr_setting.value == value
    assert isinstance(loaded_instr_setting.value, ttype)
    assert loaded_instr_setting.dtype == setting_type
    assert loaded_instr_setting.unit == unit

    assert instr_setting.get_value_string() == str_rep

    assert instr_setting == loaded_instr_setting


@pytest.mark.parametrize("ttype", [SettingType.COMPLEX, SettingType.NONE, "complex"])
def test_instrument_setting_serialiastion_complex(ttype):
    instr_setting = InstrumentSetting(
        name="test_name", value=1.0 + 2.0j, dtype=ttype, unit="V"  # type: ignore
    )

    assert instr_setting.name == "test_name"
    assert instr_setting.value == "1.0+2.0j"
    assert isinstance(instr_setting.value, str)
    assert instr_setting.dtype == SettingType.COMPLEX
    assert instr_setting.unit == "V"

    loaded_instr_setting = InstrumentSetting(**instr_setting.model_dump())

    assert loaded_instr_setting.name == "test_name"
    assert loaded_instr_setting.value == "1.0+2.0j"
    assert isinstance(loaded_instr_setting.value, str)
    assert loaded_instr_setting.dtype == SettingType.COMPLEX
    assert loaded_instr_setting.unit == "V"

    assert instr_setting.get_value_string() == "2.23607 V < 63.4°"
    assert instr_setting == loaded_instr_setting


@pytest.mark.parametrize(
    "ttype", [SettingType.LIST_COMPLEX, SettingType.NONE, "list[complex]"]
)
def test_instrument_setting_serialiastion_complex_list(ttype):
    instr_setting = InstrumentSetting(
        name="test_name",
        value=[1.0 + 2.0j, 2 + 1j],  # type: ignore
        dtype=ttype,
        unit="V",
    )

    assert instr_setting.name == "test_name"
    assert instr_setting.value == ["1.0+2.0j", "2.0+1.0j"]
    assert isinstance(instr_setting.value, list)
    assert isinstance(instr_setting.value[0], str)
    assert instr_setting.dtype == SettingType.LIST_COMPLEX
    assert instr_setting.unit == "V"
    assert (
        instr_setting.get_value_string() == "['2.23607 V < 63.4°', '2.23607 V < 26.6°']"
    )

    loaded_instr_setting = InstrumentSetting(**instr_setting.model_dump())

    assert loaded_instr_setting.name == "test_name"
    assert loaded_instr_setting.value == ["1.0+2.0j", "2.0+1.0j"]
    assert isinstance(instr_setting.value, list)
    assert isinstance(instr_setting.value[0], str)
    assert loaded_instr_setting.dtype == SettingType.LIST_COMPLEX
    assert loaded_instr_setting.unit == "V"

    assert instr_setting == loaded_instr_setting


class test_enum(Enum):
    enum1 = 1
    enum2 = 2


class test_enum_text(Enum):
    enum1 = "enum1"
    enum2 = "enum2"


val_and_type_enum = [
    (test_enum.enum1, int, SettingType.ENUM, "", 1, "1"),
    (test_enum_text.enum1, str, SettingType.ENUM, "", "enum1", "enum1"),
    (2, int, SettingType.ENUM, "", 2, "2"),
    ("enum2", str, SettingType.ENUM, "", "enum2", "enum2"),
    (
        [test_enum.enum1, test_enum.enum2],
        list,
        SettingType.LIST_ENUM,
        "",
        [1, 2],
        "['1', '2']",
    ),
    (
        [test_enum_text.enum1, test_enum_text.enum2],
        list,
        SettingType.LIST_ENUM,
        "",
        ["enum1", "enum2"],
        "['enum1', 'enum2']",
    ),
    ([1, 2], list, SettingType.LIST_ENUM, "", [1, 2], "['1', '2']"),
    (
        ["enum1", "enum2"],
        list,
        SettingType.LIST_ENUM,
        "",
        ["enum1", "enum2"],
        "['enum1', 'enum2']",
    ),
]


@pytest.mark.parametrize(
    "value, ttype, setting_type, unit, e_value, val_str", val_and_type_enum
)
def test_instrument_setting_enum(value, ttype, setting_type, unit, e_value, val_str):
    instr_setting = InstrumentSetting(
        name="test_name", value=value, dtype=setting_type, unit=unit
    )

    assert instr_setting.value == e_value
    assert isinstance(instr_setting.value, ttype)
    assert instr_setting.dtype == setting_type
    assert instr_setting == InstrumentSetting(**instr_setting.model_dump())

    assert instr_setting.get_value_string() == val_str


val_and_enum_no_type = [
    (test_enum.enum1, "", 1),
    (test_enum_text.enum1, "", "enum1"),
]


@pytest.mark.parametrize("value, unit, e_value", val_and_enum_no_type)
def test_instrument_setting_enum_no_type(value, unit, e_value):
    instr_setting = InstrumentSetting(name="test_name", value=value, unit=unit)

    assert instr_setting.value == e_value
    assert instr_setting.dtype == SettingType.ENUM
    assert instr_setting == InstrumentSetting(**instr_setting.model_dump())


val_and_enum_no_type_list = [
    ([test_enum.enum1, test_enum.enum2], "", [1, 2]),
    ([test_enum_text.enum1, test_enum_text.enum2], "", ["enum1", "enum2"]),
]


@pytest.mark.parametrize("value, unit, e_value", val_and_enum_no_type_list)
def test_instrument_setting_enum_no_type_list(value, unit, e_value):
    instr_setting = InstrumentSetting(name="test_name", value=value, unit=unit)

    assert instr_setting.value == e_value
    assert instr_setting.dtype == SettingType.LIST_ENUM
    assert instr_setting == InstrumentSetting(**instr_setting.model_dump())


@pytest.mark.parametrize("value, ttype, setting_type, unit, str_rep", val_and_type[:-1])
def test_instrument_setting_stype_none(value, ttype, setting_type, unit, str_rep):
    instr_setting = InstrumentSetting(
        name="test_name", value=value, dtype=SettingType.NONE, unit=unit
    )
    assert instr_setting.value == value
    assert isinstance(value, ttype)
    assert isinstance(instr_setting.value, ttype)
    assert instr_setting.dtype == setting_type

    assert InstrumentSetting(**instr_setting.model_dump()) == instr_setting
    setting_json = json.loads(instr_setting.model_dump_json())
    assert instr_setting == InstrumentSetting(**setting_json)


@pytest.mark.parametrize("value, ttype, setting_type, unit, str_rep", val_and_type[:-1])
def test_instrument_setting_no_type(value, ttype, setting_type, unit, str_rep):
    instr_setting = InstrumentSetting(name="test_name", value=value)

    assert instr_setting.value == value
    assert isinstance(instr_setting.value, ttype)
    assert instr_setting.dtype == setting_type

    assert InstrumentSetting(**instr_setting.model_dump()) == instr_setting
    setting_json = json.loads(instr_setting.model_dump_json())
    assert instr_setting == InstrumentSetting(**setting_json)


def test_instrument_setting_value_zero():
    instr_setting = InstrumentSetting(name="test_name", value=0)

    assert instr_setting.value == 0


def test_instrument_setting_value_zero_numpy_int():
    instr_setting = InstrumentSetting(
        name="test_name",
        value=np.int64(0),  # type: ignore
    )
    assert instr_setting.value == 0


def test_instrument_setting_str():
    instr_setting = InstrumentSetting(name="test_name", value="test_value")
    assert instr_setting.value == "test_value"


def test_instrument_setting_array():
    instr_setting = InstrumentSetting(
        name="test_name",
        value=np.asarray([1, 2, 3]),  # type: ignore
    )
    assert instr_setting.value == [1, 2, 3]


def test_instrument_setting_list():
    instr_setting = InstrumentSetting(name="test_name", value=[1, 2, 3])
    assert instr_setting.value == [1, 2, 3]


def test_instrument_setting_stype_none_error():
    with pytest.raises(TypeError):
        InstrumentSetting(
            name="test_name",
            value={"dict": 1},  # type: ignore
            dtype=SettingType.FLOAT,
            unit="",
        )

    with pytest.raises(ValueError):
        InstrumentSetting(
            name="test_name",
            value={"dict": 1},  # type: ignore
            dtype=4,  # type: ignore
            unit="",
        )

    with pytest.raises(ValueError):
        InstrumentSetting(
            name="test_name",
            value={"dict": 1},  # type: ignore
            dtype=SettingType.NONE,
            unit="",
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
        SupportsInstrumentModel.to_instrument_settings(tsim)  # type: ignore

    with pytest.raises(NotImplementedError):
        SupportsInstrumentModel.to_instrument_model(tsim)  # type: ignore

    with pytest.raises(NotImplementedError):
        SupportsInstrumentModel.from_instrument_model(InstrumentModel())  # type: ignore


def test_instrument_connection_bson():
    instr_connection = InstrumentConnection()
    bson_encoded = bson.BSON.encode(json.loads(instr_connection.model_dump_json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_instr_connection = InstrumentConnection(**bson_decoded)
    assert new_instr_connection == instr_connection


def test_instrument_connection_json():
    instr_connection = InstrumentConnection()
    conn_json = json.loads(instr_connection.model_dump_json())
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
    bson_encoded = bson.BSON.encode(json.loads(instr_setting.model_dump_json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_instr_setting = InstrumentSetting(**bson_decoded)
    assert new_instr_setting == instr_setting


@pytest.mark.parametrize("value, setting_type", _val_and_setting_type)
def test_instrument_setting_json(value, setting_type):
    instr_setting = InstrumentSetting(name="test_name", value=value, dtype=setting_type)
    setting_json = json.loads(instr_setting.model_dump_json())
    assert instr_setting == InstrumentSetting(**setting_json)


def test_instrument_model_id_str(instruments: dict[str, InstrumentModel]):
    for instrument in instruments.values():
        assert (
            instrument.get_id_string()
            == f"{instrument.hardware} - {instrument.identity} - {instrument.serial}"
        )


def test_add_get_update_setting():
    instr_setting1 = InstrumentSetting(name="test_1", value=1, dtype=SettingType.INT)
    instr_setting2 = InstrumentSetting(name="test_2", value=2, dtype=SettingType.INT)
    instr_setting3 = InstrumentSetting(name="test_3", value=3, dtype=SettingType.INT)

    instr_model = InstrumentModel(
        hardware="test",
        identity="test",
        serial="test",
        settings={
            instr_setting1.name: instr_setting1,
            instr_setting2.name: instr_setting2,
        },
        allow_new_settings=True,
    )

    instr_model.add_setting(instr_setting3)

    assert instr_model.get_setting("test_1") == instr_setting1
    assert instr_model.get_setting("test_2") == instr_setting2
    assert instr_model.get_setting("test_3") == instr_setting3

    assert instr_setting1.value == 1
    instr_model.update_setting("test_1", 4)
    assert instr_setting1.value == 4

    instr_model.update_setting("test_1", np.int64(5))  # type: ignore
    assert instr_setting1.value == 5

    with pytest.raises(ValueError):
        instr_model.update_setting("test_1", np.array([1, 2, 3]))  # type: ignore

    with pytest.raises(ValueError):  # already exists
        instr_model.add_setting(instr_setting1)

    with pytest.raises(ValueError):
        instr_model.get_setting("not_a_setting")

    with pytest.raises(ValueError):
        instr_model.update_setting(instr_setting1.name, None)  # type: ignore

    instr_setting4 = InstrumentSetting(name="test_4", value=4, dtype=SettingType.INT)
    instr_model.allow_new_settings = False
    with pytest.raises(ValueError):  # new settings not allowed
        instr_model.add_setting(instr_setting4)

    assert instr_model.needs_drainage("test_1") is False

    assert instr_setting1.unit == ""
    instr_model.update_setting("test_1", 1, "new_unit")
    assert instr_setting1.unit == "new_unit"


def test_update_list_settings():
    instr_setting_list = InstrumentSetting(
        name="test_list", value=[3], dtype=SettingType.LIST_INT
    )
    instr_model = InstrumentModel(
        hardware="test",
        identity="test",
        serial="test",
        settings={
            instr_setting_list.name: instr_setting_list,
        },
        allow_new_settings=True,
    )
    with pytest.raises(ValueError):
        instr_model.update_setting("test_list", 1)

    instr_model.update_setting("test_list", [1, 2, 3])
    assert instr_setting_list.value == [1, 2, 3]


def test_needs_drainage():
    instr_setting_model = InstrumentSetting(
        name="test_model", value="some_object", dtype=SettingType.MODEL
    )
    instr_model = InstrumentModel(
        hardware="test",
        identity="test",
        serial="test",
        settings={
            instr_setting_model.name: instr_setting_model,
        },
        allow_new_settings=True,
    )
    assert instr_model.needs_drainage(instr_setting_model.name) is True


def test_none_setting_type():
    assert NoneSettingType.to_value(None) is None
    assert NoneSettingType.to_value([]) == []
    with pytest.raises(ValueError):
        assert NoneSettingType.to_value([1]) == []


def test_add_remove_drain():
    instr_setting_model = InstrumentSetting(
        name="test_model", value="some_object", dtype=SettingType.MODEL
    )
    assert instr_setting_model.drains == []

    instr_setting_model.add_drain("test_instrument", "test_setting")
    assert instr_setting_model.drains == [
        Drain(identity="test_instrument", setting="test_setting")
    ]

    # add same drain again is ok and does nothing
    instr_setting_model.add_drain("test_instrument", "test_setting")
    assert instr_setting_model.drains == [
        Drain(identity="test_instrument", setting="test_setting")
    ]

    instr_setting_model.add_drain("test_instrument", "test_setting_2")
    assert instr_setting_model.drains == [
        Drain(identity="test_instrument", setting="test_setting"),
        Drain(identity="test_instrument", setting="test_setting_2"),
    ]

    instr_setting_model.remove_drain("test_instrument", "test_setting")
    assert instr_setting_model.drains == [
        Drain(identity="test_instrument", setting="test_setting_2"),
    ]

    # remove non-existing drain is ok and does nothing
    instr_setting_model.remove_drain("test_instrument", "test_setting")
    assert instr_setting_model.drains == [
        Drain(identity="test_instrument", setting="test_setting_2"),
    ]
