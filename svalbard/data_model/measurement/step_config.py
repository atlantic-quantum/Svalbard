"""
Configuration of a step item or a relation setting in a measurement, defines
if physical or instrument units are used, if/how to sweep, and what to do after the
last step. also defines a wait time after each step and if the sweep direction
should alternate

"""

from enum import Enum
from typing import Self

from pydantic import BaseModel, field_validator, model_validator


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

    @field_validator("wait_after")
    @classmethod
    def wait_after_positive(cls, v: float):
        """wait after must be positive"""
        if v < 0.0:
            raise ValueError("wait_after must be positive")
        return v

    @model_validator(mode="after")
    def sweep_rate_validator(self) -> Self:
        """if sweep mode is not direct then the sweep rate must not be None"""
        if self.sweep_mode != SweepMode.DIRECT:
            if self.sweep_rate is None:
                raise ValueError(
                    "sweep_rate must not be None if sweep_mode is not direct"
                )
        return self

    @model_validator(mode="after")
    def final_value_validator(self) -> Self:
        """if after_last_step is goto_value then final_value must not be None"""
        if self.after_last_step == AfterLastStep.GOTO_VALUE:
            if self.final_value is None:
                raise ValueError(
                    "final_value must not be None if after_last_step is goto_value"
                )
        return self
