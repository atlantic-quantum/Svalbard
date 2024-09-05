import json

import bson
import pytest
from pydantic import ValidationError

from svalbard.data_model.instruments import (
    InstrumentModel,
    InstrumentSetting,
    SettingType,
)
from svalbard.data_model.measurement.channel import Channel


def test_channel_bson(channel: Channel):
    """Test that DataFile can be converted to bson"""
    bson_encoded = bson.BSON.encode(json.loads(channel.model_dump_json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_channel = Channel(**bson_decoded)
    assert new_channel == channel


def test_channel_error():
    with pytest.raises(ValidationError):
        Channel(
            name="1not_number_at_beggining",
            instrument_identity="test",
            instrument_setting_name="test",
            unit_physical="test",
        )

    # keyword
    with pytest.raises(ValidationError):
        Channel(
            name="while",
            instrument_identity="test",
            instrument_setting_name="test",
            unit_physical="test",
        )


def test_channel_name_modification():
    ch = Channel(
        name="s p a c e s",
        instrument_identity="test",
        instrument_setting_name="test",
        unit_physical="test",
    )
    assert ch.name == "s_p_a_c_e_s"


def test_from_instrument_model_setting():
    instr_setting = InstrumentSetting(
        name="test_name", value=100, dtype=SettingType.INT, unit="hz"
    )
    instr_model = InstrumentModel(
        identity="test_instr", settings={instr_setting.name: instr_setting}
    )

    channel = Channel.from_instrument_model_and_setting(
        "test", instr_model, instr_setting, unit="s"
    )

    assert channel.name == "test"
    assert channel.instrument_identity == "test_instr"
    assert channel.instrument_setting_name == "test_name"
    assert channel.unit_physical == "s"
    assert channel.get_instrument_unit({"test_instr": instr_model}) == "hz"
    assert channel.dtype == SettingType.INT
    assert channel.label == "test (s)"

    instr_settin2 = InstrumentSetting(
        name="test_name2", value=100, dtype=SettingType.INT, unit="hz"
    )

    with pytest.raises(ValueError):
        # setting must be in model
        Channel.from_instrument_model_and_setting("test", instr_model, instr_settin2)

    channel2 = Channel.from_instrument_model_and_setting(
        "test", instr_model, instr_setting, dtype="bool"
    )

    assert channel2.dtype == SettingType.BOOL
    assert channel2.unit_physical == "hz"
    assert channel2.label == "test (hz)"


def test_get_instrument_setting():
    instr_setting = InstrumentSetting(
        name="test_name", value=100, dtype=SettingType.INT, unit="hz"
    )
    instr_model = InstrumentModel(
        identity="test_instr", settings={instr_setting.name: instr_setting}
    )

    channel = Channel.from_instrument_model_and_setting(
        "test", instr_model, instr_setting, unit="s"
    )

    assert instr_setting == channel.get_instrument_setting(
        {instr_model.identity: instr_model}
    )
    assert instr_setting.unit == channel.get_instrument_unit(
        {instr_model.identity: instr_model}
    )

    assert instr_setting.dtype == channel.dtype


def test_get_instrument_setting_error():
    instr_setting = InstrumentSetting(
        name="test_name", value=100, dtype=SettingType.INT, unit="hz"
    )
    instr_model = InstrumentModel(
        identity="test_instr", settings={instr_setting.name: instr_setting}
    )
    instr_model2 = InstrumentModel(identity="test_instr", settings={})

    channel = Channel.from_instrument_model_and_setting(
        "test", instr_model, instr_setting, unit="s"
    )

    with pytest.raises(ValueError):
        channel.get_instrument_setting({})

    with pytest.raises(ValueError):
        channel.get_instrument_setting({instr_model2.identity: instr_model2})


def test_json_serial_deserialization(channel: Channel):
    new_channel = Channel.model_validate_json(channel.model_dump_json())
    assert new_channel == channel


def test_json_serial_deserialization_no_limit(channel_no_limit: Channel):
    new_channel = Channel.model_validate_json(channel_no_limit.model_dump_json())
    assert new_channel == channel_no_limit


def test_json_serial_deserialization_none_limit(channel_none_limit: Channel):
    new_channel = Channel.model_validate_json(channel_none_limit.model_dump_json())
    assert new_channel == channel_none_limit


@pytest.mark.parametrize("setting", ["setting", "setting___extra"])
def test_default_channel_name(setting: str):
    instrument = "instrument"
    assert Channel.default_name(instrument, setting) == f"{instrument}___{setting}"
    assert Channel.get_instrument_and_setting(
        Channel.default_name(instrument, setting)
    ) == (instrument, setting)

    with pytest.raises(ValueError):
        Channel.get_instrument_and_setting("not_valid")
