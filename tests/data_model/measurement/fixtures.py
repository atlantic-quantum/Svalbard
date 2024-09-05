import pytest

from svalbard.data_model.measurement.channel import Channel, LimitAction, SettingType
from svalbard.data_model.measurement.measurement import Measurement
from svalbard.data_model.measurement.step_config import StepConfig
from svalbard.data_model.measurement.step_item import StepItem
from svalbard.data_model.measurement.step_range import (
    InterpolationTypes,
    RangeTypes,
    StepRange,
    StepTypes,
)


@pytest.fixture(name="channel")
def fixture_channel():
    """Fixture for creating a Channel"""
    yield Channel(
        name="test_name",
        instrument_identity="test_instrument",
        instrument_setting_name="test_setting",
        unit_physical="test_unit",
        gain=1.0,
        offset=0.0,
        amplification=1.0,
        high_limit=1.0,
        low_limit=0.0,
        limit_action=LimitAction.CONTINUE,
    )


@pytest.fixture(name="channel_no_limit")
def fixture_channel_no_limit():
    """Fixture for creating a Channel"""
    yield Channel(
        name="test_name",
        instrument_identity="test_instrument",
        instrument_setting_name="test_setting",
        unit_physical="test_unit",
        gain=1.0,
        offset=0.0,
        amplification=1.0,
        limit_action=LimitAction.CONTINUE,
    )


@pytest.fixture(name="channel_none_limit")
def fixture_channel_none_limit():
    """Fixture for creating a Channel"""
    yield Channel(
        name="test_name",
        instrument_identity="test_instrument",
        instrument_setting_name="test_setting",
        unit_physical="test_unit",
        gain=1.0,
        offset=0.0,
        amplification=1.0,
        limit_action=LimitAction.CONTINUE,
        low_limit=None,  # type: ignore
        high_limit=None,  # type: ignore
    )


@pytest.fixture(name="step_range_center_span_linear_count")
def fixture_step_range_center_span_linear_count():
    """Fixture for creating a StepRange with center and span, linear step count"""
    yield StepRange(
        range_type=RangeTypes.CENTER_SPAN,
        step_type=StepTypes.STEP_COUNT,
        interpolation_type=InterpolationTypes.LINEAR,
        center=0.0,
        span=1.0,
        step_count=11,
    )


@pytest.fixture(name="step_item")
def fixture_step_item(channel: Channel):
    step_item = StepItem(
        name=channel.name,
        config=StepConfig(),
        ranges=[
            StepRange(
                range_type=RangeTypes.VALUES,
                values=[1, 2, 3],
                interpolation_type=InterpolationTypes.LINEAR,
            ),
            StepRange(
                range_type=RangeTypes.START_STOP,
                start=0.0,
                stop=10.0,
                step_count=11,
            ),
        ],
    )
    yield step_item


@pytest.fixture(name="measurement")
def fixture_measurement():
    measurement = Measurement(
        channels=[
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
                dtype=SettingType.COMPLEX,
            ),
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
            )
        ],
        relations=[],
        log_channels=["test_name3"],  # type: ignore
    )
    yield measurement
