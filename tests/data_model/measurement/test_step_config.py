import json

import bson
import pytest

from svalbard.data_model.measurement.step_config import (
    AfterLastStep,
    StepConfig,
    StepUnits,
    SweepMode,
)

step_configs: list[StepConfig] = [
    StepConfig(),
    StepConfig(wait_after=1.0),
    StepConfig(alternate_direction=True),
    StepConfig(step_unit=StepUnits.PHYSICAL),
    StepConfig(after_last_step=AfterLastStep.STAY_AT_LAST),
    StepConfig(after_last_step=AfterLastStep.GOTO_VALUE, final_value=1.0),
    StepConfig(sweep_mode=SweepMode.BETWEEN, sweep_rate=1.0),
    StepConfig(sweep_mode=SweepMode.CONTINUOUS, sweep_rate=1.0),
    StepConfig(
        step_unit=StepUnits.PHYSICAL,
        wait_after=1.0,
        after_last_step=AfterLastStep.GOTO_VALUE,
        final_value=1.0,
        sweep_mode=SweepMode.BETWEEN,
        sweep_rate=1.0,
        alternate_direction=True,
    ),
]


@pytest.mark.parametrize("step_config", step_configs)
def test_step_config_bson(step_config: StepConfig):
    """Test that DataFile can be converted to bson"""
    bson_encoded = bson.BSON.encode(json.loads(step_config.model_dump_json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_step_config = StepConfig(**bson_decoded)
    assert new_step_config == step_config


def test_step_config_negative_wait_after_error():
    """Test that negative wait_after raises error"""
    with pytest.raises(ValueError):
        StepConfig(wait_after=-1.0)


def test_step_config_sweep_rate_swept_error():
    """Test that sweep rate is not allowed in direct mode"""
    with pytest.raises(ValueError):
        StepConfig(sweep_mode=SweepMode.CONTINUOUS, sweep_rate=None)

    with pytest.raises(ValueError):
        StepConfig(sweep_mode=SweepMode.BETWEEN, sweep_rate=None)


def test_step_config_final_value_none_error():
    """Test that final value is not allowed to be None"""
    with pytest.raises(ValueError):
        StepConfig(after_last_step=AfterLastStep.GOTO_VALUE, final_value=None)
