"""
Data model for a Log channel.

Log channels are Channels that is logged by a measurement
log channels do not neccessarily have the same shape as the measurement used to
log them, e.g. if a log channel represents an averaged/processed value over one of
the step items in the measurement.

"""

import asteval
from pydantic import BaseModel, field_validator, model_validator

from .channel import Channel


class LogChannel(BaseModel):
    """
    Log channels are Channels that is logged by a measurement
    log channels do not neccessarily have the same shape as the measurement used to
    log them, e.g. if a log channel represents an averaged/processed value over one of
    the step items in the measurement.

    Args:
        name (str):
            Name of the channel. This name is used to refer to the channel other places
            within the measurement. The name should be pythonic (lowercase, no spaces,
            no special characters).
        base_shape (tuple[int, ...]):
            shape of the log channel data. Defaults to (1,), which means that the log
            channel has a single value for each step in the measurement.
        inclusive (bool):
            Whether the step names in step_names should be included or excluded from
            the log channel shape. Defaults to False, which means that the step names
            in step_names should be excluded in the log channel shape.
        step_names (set[str]):
            set of step item names that the log channel is based on. Defaults to an
            empty set, which means that the log channel is based on all step items in
            the measurement in exclusive mode and None in inclusive mode.

            Example:
                Let's say that a measurement has two step items named "voltage" and
                "current" with "voltage" being the first step item and "current" being
                the second step item.

                If the log channel is inclusive and is based on a step item named
                "voltage" and a step item named "current", the set should be
                {"voltage", "current"}. The log channel will then be a 2D array with
                the first dimension representing changes in the log channel versus
                voltage and the second dimension representing changes in the log channel
                versus current.

                If the log channel is only based on a step item named "voltage", the
                set should be {"voltage"}.
                The log channel will then be a 1D array of measured log channel values
                versus voltage.

                Note that in both cases the measurement is still two dimensional, but
                the latter case it is assumed that the raw data has somehow been
                processed to a single value for each voltage step.

    """

    name: str
    base_shape: tuple[int, ...] = (1,)
    inclusive: bool = False
    step_names: set[str] = set()

    @field_validator("name")
    @classmethod
    def source_name_must_be_valid_symbol_name(cls, v: str):
        """Validate that source_name is a valid asteval symbol name"""
        if not asteval.valid_symbol_name(v):
            v = Channel.make_pythonic(v)
        return v

    @model_validator(mode="before")  # type: ignore
    @classmethod
    def cast_step_names(cls, data):
        """Cast step_names to a set if it is a string or a list."""
        if "step_names" not in data:
            return data
        if isinstance(data["step_names"], str):
            data["step_names"] = {data["step_names"]}
        if isinstance(data["step_names"], list):
            data["step_names"] = set(data["step_names"])
        return data

    def shape(self, idx_size_steps: dict[int, tuple[int, set[str]]]) -> tuple[int, ...]:
        """
        Returns the shape of the log channel data based on the shape of the step items
        in the measurement.

        Args:
            idx_size_steps (dict[idx, tuple[int, set[str]]]):
                dictionary with the size of the step item at each index and a set of
                step item names at each index. The keys are the indices of the steps.

        Returns:
            tuple[int, ...]:
                shape of the log channel data
        """
        if not idx_size_steps:
            return self.base_shape
        idx_size = {idx: size_steps[0] for idx, size_steps in idx_size_steps.items()}
        idx_steps = {idx: size_steps[1] for idx, size_steps in idx_size_steps.items()}
        step_shape_list = self._step_shapes(idx_size, idx_steps)
        if self.base_shape != (1,):
            return tuple(list(self.base_shape) + step_shape_list)
        return tuple(step_shape_list) if step_shape_list else self.base_shape

    def _step_shapes(
        self, index_size: dict[int, int], index_steps: dict[int, set[str]]
    ) -> list[int]:
        """
        Raw shape of the log channel data based on the shape of the step items in the
        measurement and the step names in the log channel. Excludes the base shape of
        the log channel.

        Args:
            index_size (dict[int, int]):
                dictionary keyed by the index of the step item with the size of the
                step item as the value.
            index_steps (dict[int, set[str]]):
                dictionary keyed by the index of the step item with a set of step names
                as the value.

        Returns:
            list[int]: shape of the log channel data
        """
        if not self.step_names:
            if self.inclusive:
                return []
            return [size for _, size in sorted(index_size.items())]
        return [
            index_size[index]
            for index, step_names in sorted(index_steps.items())
            if self.inclusive == bool(self.step_names.intersection(step_names))
        ]
