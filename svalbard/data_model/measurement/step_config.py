"""
Configuration of a step item or a relation setting in a measurement, defines
if physical or instrument units are used, if/how to sweep, and what to do after the
last step. also defines a wait time after each step and if the sweep direction
should alternate

"""

from enum import Enum

from pydantic import BaseModel, validator


class StepUnits(Enum):
    """Enum for step units, either instrument or physical"""

    INSTRUMENT = "Instrument"
    PHYSICAL = "Physical"


class AfterLastStep(Enum):
    """
    Enum for what to do after the last step in step item or relation setting,
    either go to first step, stay at last step or go to a specific value
    """

    GOTO_FIRST = "Goto first"
    STAY_AT_LAST = "Stay at last"
    GOTO_VALUE = "Goto value"


class SweepMode(Enum):
    """
    Enumerations for sweep mode, either direct, between or continuous
    in direct mode the instrument is set to the step value and data is acquired
    in between mode the instrument is swept between step values and the sweep pauses
    at each step value while data is acquired
    in continuous mode the instrument is swept continuously, data acquisition starts
    at each step value but sweeps continuous while data is acquired
    """

    DIRECT = "Direct"
    BETWEEN = "Between"
    CONTINUOUS = "Continuous"


class StepConfig(BaseModel):
    """
    Pydantic model for step configuration of a step item or a relation setting
    in a measurement, defines if physical or instrument units are used, if/how to sweep,
    and what to do after the last step. also defines a wait time after each step and
    if the sweep direction should alternate.

    Args:
        step_unit (StepUnits, optional):
            Unit of the step_item / relation, either instrument or physical.
            Defaults to StepUnits.INSTRUMENT.
        wait_after (float, optional):
            Wait time after each step in seconds. Defaults to 0.0.
        after_last_step (AfterLastStep, optional):
            What to do after the last step, either go to first step, stay at last step
            or go to a specific value. Defaults to AfterLastStep.GOTO_FIRST.
        final_value (float, optional):
            Final value to go to after the last step if after_last_step is goto_value.
            defaults to 0.0
        sweep_mode (SweepMode, optional):
            Sweep mode, either direct, between or continuous.
              - In direct mode the instrument is set to the step value
                and data is acquired
              - In between mode the instrument is swept between step values
                the sweep pauses at each step value while data is acquired
              - In continuous mode the instrument is swept continuously
                aquisition starts at each step value but sweeps continuous
                while data is acquired
            Defaults to SweepMode.DIRECT.
        sweep_rate (float, optional):
            Sweep rate in units per second, defaults to 0.0
            only used if sweep mode is between or continuous.
        alternate_direction (bool, optional):
            If the sweep direction should alternate, defaults to False
    """

    step_unit: StepUnits = StepUnits.INSTRUMENT
    wait_after: float = 0.0
    after_last_step: AfterLastStep = AfterLastStep.GOTO_FIRST
    final_value: float | None = 0.0
    sweep_mode: SweepMode = SweepMode.DIRECT
    sweep_rate: float | None = 0.0
    alternate_direction: bool = False

    @validator("wait_after")
    def wait_after_positive(cls, v: float):
        """wait after must be positive"""
        assert v >= 0.0
        return v

    @validator("sweep_rate")
    def sweep_rate_validator(cls, v: float | None, values: dict):
        """if sweep mode is direct then the sweep rate must not be None"""
        assert "sweep_mode" in values
        if values["sweep_mode"] != SweepMode.DIRECT:
            assert v is not None
        return v

    @validator("final_value")
    def final_value_validator(cls, v: float | None, values: dict):
        """if after_last_step is goto_value then final_value must not be None"""
        assert "after_last_step" in values
        if values["after_last_step"] == AfterLastStep.GOTO_VALUE:
            assert v is not None
        return v
