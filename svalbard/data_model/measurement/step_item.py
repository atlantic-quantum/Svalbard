"""
Data model for step items. Step items define how to step a single channel
"""

import numpy as np
from pydantic import BaseModel

from .step_config import StepConfig
from .step_range import StepRange


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
    """

    name: str
    config: StepConfig
    ranges: list[StepRange]

    def calculate_values(self) -> np.ndarray:
        """calculate values for all step ranges

        Returns:
            np.ndarray: all values that the step item should step through
        """
        return np.concatenate([r.step_values() for r in self.ranges])

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
