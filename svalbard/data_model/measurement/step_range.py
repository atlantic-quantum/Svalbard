"""
Data model for step ranges. Step ranges define a range of values to step through
"""

from abc import ABC
from enum import Enum
from numbers import Number
from typing import Callable, Self

import numpy as np
from pydantic import BaseModel, field_validator, model_validator

from ...typing import TSettingValue, TSettingValueSweep
from ...utility.str_helper import get_si_string


class BaseRangeType(ABC):  # pragma: no cover
    """
    Base Class for range types. the range type classes define how values are generated
    for the range type.
    """

    range_type: "RangeTypes"

    def step_values(self) -> np.ndarray | TSettingValueSweep:
        """
        Returns:
            np.ndarray | TSettingValueList:
                The values to step through for the range type instance
        """
        raise NotImplementedError

    def n_steps(self) -> int:
        """
        Returns:
            int: Number of steps in the range type instance
        """
        raise NotImplementedError

    def range_strings(self, unit: str) -> tuple[str, str]:
        """
        Returns:
            tuple[str, str]:
                tuple of strings representing the range of values for the range type
                instance and the step size or number of steps for the range type.
        """
        raise NotImplementedError

    @classmethod
    def from_step_range(cls, step_range: "StepRange") -> Self:
        """
        Returns:
            Self: Instance of the range type class created from a StepRange instance
        """
        raise NotImplementedError

    @classmethod
    def update_step_range(cls, step_range: "StepRange") -> None:
        """
        Update the parameters of a StepRange instance to match the current values.

        Example:
            for a start_stop range type the center and span values are updated to match
            the current start and stop values.

        Args:
            step_range (StepRange): StepRange instance to update
        """
        raise NotImplementedError


class RangeTypes(str, Enum):
    """
    Enumeration for range types, either start_stop, center_span or values

    in start_stop mode the range is defined by a start and stop value
    in center_span mode the range is defined by a center value and a span
    in values mode the range is defined directly by a list of values
    """

    START_STOP = "Start stop"
    CENTER_SPAN = "Center span"
    VALUES = "Values"

    @staticmethod
    def get_range_type_class(range_type: "RangeTypes") -> type[BaseRangeType]:
        """
        Raises:
            ValueError: if range_type is not a valid RangeTypes

        Returns:
            type[BaseRangeType]:
                BaseRangeType subclass corresponding to the range type
        """
        range_classes = {
            range_class.range_type: range_class
            for range_class in BaseRangeType.__subclasses__()
        }
        if range_type not in range_classes:
            raise ValueError("Invalid range type")
        return range_classes[range_type]


class StepTypes(str, Enum):
    """
    Enumeration for step types, either step_size or step_count

    in step_size mode the step size between each step is defined
    in step_count mode the number of steps is defined
    """

    STEP_SIZE = "Step size"
    STEP_COUNT = "Step count"


# define number of steps per decade for logdecade interpolation
DECADES = [
    [1.0],
    [1.0, 3.0],
    [1.0, 2.0, 5.0],
    [1.0, 2.0, 3.0, 6.0],
    [1.0, 1.6, 2.5, 4, 6.4],
    [1.0, 1.5, 2.2, 3.2, 4.6, 6.8],
    [1.0, 1.4, 1.9, 2.7, 3.7, 5.2, 7.2],
    [1.0, 1.3, 1.8, 2.4, 3.2, 4.2, 5.6, 7.5],
    [1.0, 1.3, 1.7, 2.2, 2.8, 3.6, 4.6, 6.0, 7.7],
    [1.0, 1.3, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.3, 8.0],
]


def decadespace(start: float, stop: float, points_per_decade: int) -> np.ndarray:
    """
    create a logarithmically spaced array with a number of points per decade

    Args:
        start (float):
            start value
        stop (float):
            stop value
        points_per_decade (int):
            points per decade, if between 1 and 10 then a predefined 'nice' values
            are used for each decade, otherwise the number of points per decade is used
            exactly

    Returns:
        np.ndarray: logarithmically spaced array with a number of points per decade
    """
    if 1 <= points_per_decade <= 10:
        points_in_decade = np.array(DECADES[int(points_per_decade) - 1])
        first_decade = int(np.floor(np.log10(start)))
        last_decade = int(np.ceil(np.log10(stop)))
        decade_values = np.concatenate(
            [
                points_in_decade * 10 ** (first_decade + i)
                for i in range(last_decade - first_decade)
            ]
        )
        space_values = np.concatenate(
            [
                [start],
                decade_values[
                    np.logical_and(start < decade_values, decade_values < stop)
                ],
                [stop],
            ]
        )
        r_idx_low = 1 if np.allclose(space_values[1], start) else 0
        r_idx_high = -1 if np.allclose(space_values[-2], stop) else -0
        return space_values[r_idx_low : space_values.size + r_idx_high]
    decades = np.log10(stop / start)
    points = int(decades * points_per_decade) + 1
    return np.geomspace(start, stop, points)


class InterpolationTypes(str, Enum):
    """
    Interpolation types for step ranges, either linear, log or logdecade

    in linear mode the steps are linearly spaced
    in log mode the steps are logarithmically spaced
    in logdecade mode the steps are logarithmically spaced with a number of steps
    per each decade
    """

    LINEAR = "Linear"
    LOG = "Log"
    LOGDECADE = "Log,#/decade"

    @staticmethod
    def get_interpolation_function(
        interpolation_type: "InterpolationTypes",
    ) -> Callable[..., np.ndarray]:
        if interpolation_type == InterpolationTypes.LINEAR:
            return np.linspace
        if interpolation_type == InterpolationTypes.LOG:
            return np.geomspace
        if interpolation_type == InterpolationTypes.LOGDECADE:
            return decadespace
        raise ValueError("Invalid interpolation type")


def _step_values_size(
    start: float, stop: float, size: float, interpolation_type: InterpolationTypes
) -> np.ndarray:
    """
    Calculate step values for the step range step_type is STEP_SIZE (for range_type
    START_STOP or CENTER_SPAN)

    Raises:
        ValueError:
            if interpolation type is not linear

    Returns:
        np.ndarray: values to step through
    """
    if interpolation_type != InterpolationTypes.LINEAR:
        raise ValueError(
            f"Invalid interpolation type {interpolation_type} for step_size"
        )
    return np.arange(start, stop, size)


def _step_values_count(
    start: float, stop: float, step_count: int, interpolation_type: InterpolationTypes
) -> np.ndarray:
    """
    Calculate step values for the step range step_type is STEP_COUNT (for range_type
    START_STOP or CENTER_SPAN)

    Raises:
        ValueError:
            if interpolation type is not known InterpolationTypes

    Returns:
        np.ndarray: values to step through
    """
    return InterpolationTypes.get_interpolation_function(interpolation_type)(
        start, stop, step_count
    )


def _update_parameter_step_size_and_count(step_range: "StepRange"):
    """Updates the step size and count parameters of a StepRange instance

    Args:
        step_range (StepRange): StepRange instance to update

    Raises:
        ValueError: if start or stop is None
        ValueError: if step_type is not a known StepType
        ValueError: if interpolation type is not known InterpolationTypes
        ValueError: if step_size is None and step_type is STEP_SIZE
        ValueError: if step_count is None and step_type is STEP_COUNT
        ValueError: if interpolation type is not LINEAR for step_type STEP_SIZE
    """
    if step_range.start is None:
        raise ValueError("start must not be None when updating step size or count")
    if step_range.stop is None:
        raise ValueError("stop must not be None if range_type is START_STOP")
    if not isinstance(step_range.step_type, StepTypes):
        raise ValueError("Invalid step type")

    if step_range.step_type == StepTypes.STEP_SIZE:
        if step_range.interpolation_type != InterpolationTypes.LINEAR:
            estr = "Invalid interpolation type {:s} for step type STEP_SIZE"
            raise ValueError(estr.format(step_range.interpolation_type))
        if step_range.step_size is None:
            raise ValueError("step_size must not be None if step_type is STEP_SIZE")
        step_range.step_count = (
            int((step_range.stop - step_range.start) / step_range.step_size) + 1
        )
    else:
        if step_range.step_count is None:
            raise ValueError("step_count must not be None if step_type is STEP_COUNT")
        if step_range.interpolation_type == InterpolationTypes.LINEAR:
            step_range.step_size = (step_range.stop - step_range.start) / (
                step_range.step_count - 1
            )
        else:
            step_range.step_size = None


class StartStopRange(BaseRangeType):
    """
    Range type class for a range type defined by a start and stop value

    Args:
        start (float):
            start value of the range
        stop (float):
            stop value of the range
        step_size (float, optional):
            step size between each step in the range. Defaults to None.
            Used if step_type is STEP_SIZE
        step_count (int, optional):
            number of steps in the range. Defaults to None.
            Used if step_type is STEP_COUNT
        interpolation_type (InterpolationTypes, optional):
            interpolation type for the range. Defaults to InterpolationTypes.LINEAR
        step_type (StepTypes, optional):
            step type for the range. Defaults to StepTypes.STEP_COUNT

    """

    range_type: RangeTypes = RangeTypes.START_STOP

    def __init__(
        self,
        start: float,
        stop: float,
        step_size: float | None = None,
        step_count: int | None = None,
        interpolation_type: InterpolationTypes = InterpolationTypes.LINEAR,
        step_type: StepTypes = StepTypes.STEP_COUNT,
    ):
        self.start = start
        self.stop = stop
        self.step_size = step_size
        self.step_count = step_count
        self.interpolation_type = interpolation_type
        self.step_type = step_type

    @classmethod
    def from_step_range(cls, step_range: "StepRange") -> "StartStopRange":
        if step_range.start is None:
            raise ValueError("start must not be None if range_type is START_STOP")
        if step_range.stop is None:
            raise ValueError("stop must not be None if range_type is START_STOP")
        return cls(
            step_range.start,
            step_range.stop,
            step_range.step_size,
            step_range.step_count,
            step_range.interpolation_type,
            step_range.step_type,
        )

    def step_values(self) -> np.ndarray:
        """
        Calculate step values for the step range if range_type is START_STOP

        Raises:
            ValueError:
                if both step_size and step_count are None

        Returns:
            np.ndarray: values to step through
        """
        if self.step_size is not None and self.step_type == StepTypes.STEP_SIZE:
            return _step_values_size(
                self.start, self.stop, self.step_size, self.interpolation_type
            )
        if self.step_count is not None and self.step_type == StepTypes.STEP_COUNT:
            return _step_values_count(
                self.start, self.stop, self.step_count, self.interpolation_type
            )
        raise ValueError("step_size or step_count must be set and match step_type")

    def n_steps(self) -> int:
        if self.step_type == StepTypes.STEP_COUNT and self.step_count is not None:
            if self.interpolation_type == InterpolationTypes.LOGDECADE:
                return len(self.step_values())
            return self.step_count
        if self.step_type == StepTypes.STEP_SIZE and self.step_size is not None:
            return len(self.step_values())
        raise ValueError("step_size or step_count must be set and match step_type")

    def range_strings(self, unit: str) -> tuple[str, str]:
        vals = self.step_values()
        step_str = f"{len(vals)} pts"
        start_str = get_si_string(vals[0], unit, decimals=6)
        stop_str = get_si_string(vals[-1], unit, decimals=6)
        range_str = f"{start_str} - {stop_str}"
        # in this mode, step_str can be step size
        if self.step_size is not None and self.step_type == StepTypes.STEP_SIZE:
            step_str = get_si_string(self.step_size, unit, decimals=6)
        return (range_str, step_str)

    @classmethod
    def update_step_range(cls, step_range: "StepRange") -> None:
        if step_range.start is None:
            raise ValueError("start must not be None if range_type is START_STOP")
        if step_range.stop is None:
            raise ValueError("stop must not be None if range_type is START_STOP")
        step_range.center = (step_range.start + step_range.stop) / 2
        step_range.span = step_range.stop - step_range.start
        _update_parameter_step_size_and_count(step_range)


class CenterSpanRange(BaseRangeType):
    """
    Range type class for a range type defined by a center and span value

    Args:
        center (float):
            center value of the range
        span (float):
            center value of the range
        step_size (float, optional):
            step size between each step in the range. Defaults to None.
            Used if step_type is STEP_SIZE
        step_count (int, optional):
            number of steps in the range. Defaults to None.
            Used if step_type is STEP_COUNT
        interpolation_type (InterpolationTypes, optional):
            interpolation type for the range. Defaults to InterpolationTypes.LINEAR
        step_type (StepTypes, optional):
            step type for the range. Defaults to StepTypes.STEP_COUNT

    """

    range_type: RangeTypes = RangeTypes.CENTER_SPAN

    def __init__(
        self,
        center: float,
        span: float,
        step_size: float | None = None,
        step_count: int | None = None,
        interpolation_type: InterpolationTypes = InterpolationTypes.LINEAR,
        step_type: StepTypes = StepTypes.STEP_COUNT,
    ):
        self.center = center
        self.span = span
        self.step_size = step_size
        self.step_count = step_count
        self.interpolation_type = interpolation_type
        self.step_type = step_type

    @classmethod
    def from_step_range(cls, step_range: "StepRange") -> "CenterSpanRange":
        if step_range.center is None:
            raise ValueError("center must not be None if range_type is CENTER_SPAN")
        if step_range.span is None:
            raise ValueError("span must not be None if range_type is CENTER_SPAN")
        return cls(
            step_range.center,
            step_range.span,
            step_range.step_size,
            step_range.step_count,
            step_range.interpolation_type,
            step_range.step_type,
        )

    def step_values(self) -> np.ndarray:
        """
        Calculate step values for the step range if range_type is CENTER_SPAN

        Raises:
            ValueError:
                if both step_size and step_count are None

        Returns:
            np.ndarray: values to step through
        """
        start = self.center - self.span / 2
        stop = self.center + self.span / 2
        if self.step_size is not None and self.step_type == StepTypes.STEP_SIZE:
            return _step_values_size(
                start, stop, self.step_size, self.interpolation_type
            )
        if self.step_count is not None and self.step_type == StepTypes.STEP_COUNT:
            return _step_values_count(
                start, stop, self.step_count, self.interpolation_type
            )
        raise ValueError("step_size or step_count must be set and match step_type")

    def n_steps(self) -> int:
        if self.step_type == StepTypes.STEP_COUNT and self.step_count is not None:
            if self.interpolation_type == InterpolationTypes.LOGDECADE:
                return len(self.step_values())
            return self.step_count
        if self.step_type == StepTypes.STEP_SIZE and self.step_size is not None:
            return len(self.step_values())
        raise ValueError("step_size or step_count must be set and match step_type")

    def range_strings(self, unit: str) -> tuple[str, str]:
        center_str = get_si_string(self.center, unit, decimals=6)
        span_str = get_si_string(self.span, unit, decimals=6)
        range_str = f"c = {center_str}, w = {span_str}"
        # in this mode, step_str can be step size
        if self.step_size is not None and self.step_type == StepTypes.STEP_SIZE:
            step_str = get_si_string(self.step_size, unit, decimals=6)
        if self.step_count is not None and self.step_type == StepTypes.STEP_COUNT:
            step_str = f"{self.n_steps()} pts"
        return (range_str, step_str)

    @classmethod
    def update_step_range(cls, step_range: "StepRange") -> None:
        if step_range.center is None:
            raise ValueError("center must not be None if range_type is CENTER_SPAN")
        if step_range.span is None:
            raise ValueError("span must not be None if range_type is CENTER_SPAN")
        step_range.start = step_range.center - step_range.span / 2
        step_range.stop = step_range.center + step_range.span / 2
        _update_parameter_step_size_and_count(step_range)


class ValuesRange(BaseRangeType):
    """
    Range type class for a range type defined by a list of values

    Args:
        values (TSettingValueList):
            list of values in the range.
    """

    range_type: RangeTypes = RangeTypes.VALUES

    def __init__(self, values: TSettingValueSweep):
        self.values = values

    @classmethod
    def from_step_range(cls, step_range: "StepRange") -> "ValuesRange":
        if step_range.values is None:
            raise ValueError("values must not be None if range_type is VALUES")
        return cls(step_range.values)

    def step_values(self) -> np.ndarray | TSettingValueSweep:
        """
        Calculate step values for the step range if range_type is VALUES

        Raises:
            ValueError:
                if values is None

        Returns:
            np.ndarray: values to step through
        """
        if isinstance(self.values[0], Number):
            return np.array(self.values)
        return self.values

    def range_strings(self, unit: str) -> tuple[str, str]:
        # in generic case, use max/min format
        vals = self.step_values()
        step_str = f"{len(vals)} pts"
        if isinstance(vals[0], Number):
            assert isinstance(vals, np.ndarray)
            min_str = get_si_string(min(vals), unit, decimals=6)
            max_str = get_si_string(max(vals), unit, decimals=6)
            range_str = f"{min_str} - {max_str}"
        else:
            range_str = ""
        return (range_str, step_str)

    def n_steps(self) -> int:
        return len(self.values)

    @classmethod
    def update_step_range(cls, step_range: "StepRange") -> None:
        if step_range.values is None:
            raise ValueError("values must not be None if range_type is VALUES")
        step_range.start = None
        step_range.stop = None
        step_range.center = None
        step_range.span = None
        step_range.step_size = None
        step_range.step_count = len(step_range.values)


class StepRange(BaseModel):
    """
    Pydantic model for step ranges. Step ranges define a range of values to step through

    Args:
        range_type (RangeTypes, optional):
            Type of the step range, either start_stop, center_span or values.
            Defaults to RangeTypes.START_STOP.
        step_type (StepTypes, optional):
            Type of the step, either step_size or step_count.
            Defaults to StepTypes.STEP_COUNT.
        interpolation_type (InterpolationTypes, optional):
            Interpolation type, either linear, log or logdecade.
            Defaults to InterpolationTypes.LINEAR.
                in log mode step type must be step_count
                in logdecade mode step type must be step_count
        start (float, optional):
            Start value of the step range. Defaults to None.
            Used if range_type is start_stop.
        stop (float, optional):
            Stop value of the step range. Defaults to None.
            Used if range_type is start_stop.
        center (float, optional):
            Center value of the step range. Defaults to None.
            Used if range_type is center_span.
        span (float, optional):
            Span value of the step range. Defaults to None.
            Used if range_type is center_span.
        step_size (float, optional):
            Step size of the step range. Defaults to None.
            Used if step_type is step_size.
        step_count (int, optional):
            Number of steps in the step range. Defaults to None.
            Used if step_type is step_count.
            Must be used if InterpolationTypes is log or logdecade.
        values (list[float], optional):
            List of values in the step range. Defaults to None.
            Used if range_type is values.
    """

    range_type: RangeTypes = RangeTypes.START_STOP
    step_type: StepTypes = StepTypes.STEP_COUNT
    interpolation_type: InterpolationTypes = InterpolationTypes.LINEAR
    start: float | None = None
    stop: float | None = None
    center: float | None = None
    span: float | None = None
    step_size: float | None = None
    step_count: int | None = None  # use this for #/decade as well
    values: TSettingValueSweep | None = None

    @model_validator(mode="after")
    def validate_start_step(self) -> Self:
        """validate that start and stop are not None if range_type is START_STOP"""
        if self.range_type == RangeTypes.START_STOP:
            if self.start is None:
                raise ValueError("start must not be None if range_type is START_STOP")
            if self.stop is None:
                raise ValueError("stop must not be None if range_type is START_STOP")
        return self

    @model_validator(mode="after")
    def validate_center_span(self) -> Self:
        """validate that center and span are not None if range_type is CENTER_SPAN"""
        if self.range_type == RangeTypes.CENTER_SPAN:
            if self.center is None:
                raise ValueError("center must not be None if range_type is CENTER_SPAN")
            if self.span is None:
                raise ValueError("span must not be None if range_type is CENTER_SPAN")
        return self

    @model_validator(mode="after")
    def validate_step_size(self) -> Self:
        """validate that step size is not None if step_type is step_size"""
        if self.step_type == StepTypes.STEP_SIZE:
            if self.step_size is None:
                raise ValueError("step_size must not be None if step_type is step_size")
            if self.step_size == 0:
                raise ValueError("step_size must not be 0 if step_type is step_size")
        return self

    @model_validator(mode="after")
    def validate_step_count(self) -> Self:
        """validate that step count is not None if step_type is step_count"""
        if self.range_type == RangeTypes.VALUES:
            return self
        if self.step_type == StepTypes.STEP_COUNT:
            if self.step_count is None:
                raise ValueError(
                    "step_count must not be None if step_type is STEP_COUNT"
                )
            if self.step_count <= 0:
                raise ValueError(
                    "step_count must positive integer when step_type is STEP_COUNT"
                )
        return self

    @model_validator(mode="after")
    def validate_interpolation_type(self) -> Self:
        """validate that interpolation type is valid for step type"""
        if self.interpolation_type != InterpolationTypes.LINEAR:
            if self.step_type != StepTypes.STEP_COUNT:
                raise ValueError(
                    f"Invalid interpolation type {self.interpolation_type}"
                    " for step_count"
                )
        return self

    @field_validator("values", mode="before")
    @classmethod
    def values_must_be_list(cls, v):
        """validate that values is a list"""
        if v is None:
            return None
        if isinstance(v, np.ndarray):
            if v.ndim != 1:
                raise ValueError("values must be 1D array if supplied as ndarray")
            return v.tolist()
        if not isinstance(v, list):
            return [v]
        return v

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        """validate that values is not None if range_type is VALUES"""
        if self.range_type == RangeTypes.VALUES:
            if self.values is None:
                raise ValueError("values must not be None if range_type is VALUES")
        return self

    def step_values(self) -> np.ndarray | TSettingValueSweep:
        """
        Calculate step values for the step range

        Raises:
            ValueError:
                if range_type is not known RangeTypes

        Returns:
            np.ndarray: values to step through
        """
        range_class = RangeTypes.get_range_type_class(self.range_type)
        return range_class.from_step_range(self).step_values()

    def update_parameters(self):
        """
        Update parameters of the step range to match the current values,

        i.e.
        if the step range is in start_stop mode then the center and span values are
        updated to match the current start and stop values.

        if the interpolation type is linear and the step type is step_size then the
        step count is updated to match the current step size and start and stop values.


        Raises:
            ValueError:
                if range_type is not known RangeTypes
            ValueError:
                if step_type is not known StepTypes
        """
        range_class = RangeTypes.get_range_type_class(self.range_type)
        range_class.update_step_range(self)

    def n_steps(self) -> int:
        """
        Get the number of steps in the step range

        Returns:
            int: number of steps
        """
        range_class = RangeTypes.get_range_type_class(self.range_type)
        return range_class.from_step_range(self).n_steps()

    def get_step_value(self, index: int) -> TSettingValue:
        """
        Get the value of the step at the given index

        Args:
            index (int):
                index of the step

        Returns:
            TSettingValue: value of the step
        """
        return self.step_values()[index]

    def get_range_strings(self, unit: str) -> tuple[str, str]:
        """
        Get the range strings for the step range

        Args:
            unit (str): unit string, appended to the range strings

        Returns:
            tuple[str, str]: range and step strings
        """
        range_class = RangeTypes.get_range_type_class(self.range_type)
        return range_class.from_step_range(self).range_strings(unit)
