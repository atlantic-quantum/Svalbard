"""
Data model for step items. Step items define how to step a single channel
"""

import asteval
import numpy as np
from pydantic import BaseModel, field_validator

from ...typing import TSettingValue, TSettingValueSweep
from ...utility.str_helper import get_si_string
from .channel import Channel
from .step_config import StepConfig
from .step_range import RangeTypes, StepRange


class StepItem(BaseModel):
    """
    Pydantic model for step items. Step items define how to step a single channel

    Args:
        name (str):
            Name of the channel this step item is connected to.
        config (StepConfig):
            Configuration of the step item, see StepConfig model for details.
        ranges (list[StepRange]):
            List of step ranges, each step range defines a range of values to step
            see StepRange model for details.
        index (int | None, optional):
            Index of the step item in the list of step items. Defaults to None.
        hw_swept (bool, optional):
            Whether the step item should be swept in hardware. Defaults to False.
            To work correctly, the underlying instrument settings must support hardware
            sweeping.
    """

    name: str
    ranges: list[StepRange]
    config: StepConfig = StepConfig()
    index: int | None = None
    hw_swept: bool = False

    @field_validator("name")
    @classmethod
    def source_name_must_be_valid_symbol_name(cls, v: str):
        """Validate that source_name is a valid asteval symbol name"""
        if not asteval.valid_symbol_name(v):
            v = Channel.make_pythonic(v)
        return v

    def calculate_values(self) -> np.ndarray | TSettingValueSweep:
        """Calculate values for all step ranges

        Returns:
            np.ndarray: all values that the step item should step through
        """
        all_step_values = [r.step_values() for r in self.ranges]
        for step_values in all_step_values[1:]:
            if not isinstance(step_values, type(all_step_values[0])):
                raise ValueError("All ranges must generate values of the same type")

        if isinstance(all_step_values[0], np.ndarray):
            np_values: np.ndarray = np.concatenate(all_step_values)  # type: ignore
            return np_values

        if isinstance(all_step_values[0], list):
            values: TSettingValueSweep = [
                value for step_values in all_step_values for value in step_values
            ]  # type: ignore
            return values
        raise ValueError("Invalid step values type")  # pragma: no cover

    def get_range_strings(self, unit: str) -> tuple[str, str]:
        """Get descriptive strings for the range of step item, for use in UI

        Args:
            unit (str):
                unit to use for formatting the range values

        Returns:
            tuple[str, str]: Strings describing range of values and step size/n pts
        """
        # formatting depends on range type
        if len(self.ranges) == 0:
            # no ranges, return empty strings
            return ("", "")

        vals = self.calculate_values()
        # default step str is using n pts formatting
        step_str = f"{len(vals)} pts"

        if not isinstance(vals, np.ndarray):
            return ("", step_str)

        if isinstance(unit, Channel):
            unit = unit.unit_physical
        if len(self.ranges) > 1:
            # multiple numeric ranges, use max/min and n pts format
            min_str = get_si_string(min(vals), unit, decimals=6)
            max_str = get_si_string(max(vals), unit, decimals=6)
            return (f"{min_str} - {max_str}", step_str)

        range_class = RangeTypes.get_range_type_class(self.ranges[0].range_type)
        return range_class.from_step_range(self.ranges[0]).range_strings(unit)

    @property
    def step_count(self) -> int:
        """Calculate the total number of steps in the step item

        Returns:
            int: total number of steps
        """
        return sum(r.n_steps() for r in self.ranges)

    def get_step_value(self, index: int) -> TSettingValue:
        """Get the step value at a given index

        Args:
            index (int): index of the step value to get, count starts at first
                step range and continues through all step ranges in order of the ranges
                list.

        raises:
            IndexError: if index is out of range (negative indices do not wrap around)

        Returns:
            TSettingValue: step value at the given index
        """
        if index < 0:
            raise IndexError("Index out of range")
        for step_range in self.ranges:
            if index < step_range.n_steps():
                return step_range.get_step_value(index)
            index -= step_range.n_steps()
        raise IndexError("Index out of range")

    def add_range(self, step_range: StepRange, index: int | None = None):
        """Add a step range to the step item

        Args:
            step_range (StepRange):
                Step range to add
            index (int | None, optional):
                index the new step range is inserted at into the list of step ranges.
                Defaults to None. If None the step range is appended to the end of list.
        """
        if index is None:
            self.ranges.append(step_range)
        else:
            self.ranges.insert(index, step_range)

    def get_range(self, index: int) -> StepRange:
        """
        Get a step range from the list of step ranges

        Args:
            index (int):
                index of the step range to get

        Returns:
            StepRange: step range at the given index
        """
        return self.ranges[index]

    def remove_range(self, index: int):
        """
        Remove a step range from the list of step ranges

        Args:
            index (int):
                index of the step range to remove
        """
        self.ranges.remove(self.ranges[index])

    def set_range_position(self, index: int, new_index: int):
        """
        Change the position of a step range in the list of step ranges

        Args:
            index (int):
                current index of the step range to move
            new_index (int):
                index to move the step range to
        """
        step_range = self.get_range(index)
        self.ranges.remove(step_range)
        self.ranges.insert(new_index, step_range)
