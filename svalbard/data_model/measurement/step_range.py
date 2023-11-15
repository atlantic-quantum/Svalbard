"""
Data model for step ranges. Step ranges define a range of values to step through
"""

from enum import Enum

import numpy as np
from pydantic import BaseModel, validator


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


class StepTypes(str, Enum):
    """
    Enumeration for step types, either step_size or step_count

    in step_size mode the step size between each step is defined
    in step_count mode the number of steps is defined
    """

    STEP_SIZE = "Step size"
    STEP_COUNT = "Step count"


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
    values: list[float] | None = None

    @validator("start", "stop")
    def start_stop_validator(cls, v: float | None, values: dict):
        """validate that start and stop are not None if range type is start_stop"""
        assert "range_type" in values
        if values["range_type"] == RangeTypes.START_STOP:
            assert v is not None
        return v

    @validator("center", "span")
    def center_span_validator(cls, v: float | None, values: dict):
        """validate that center and span are not None if range type is center_span"""
        assert "range_type" in values
        if values["range_type"] == RangeTypes.CENTER_SPAN:
            assert v is not None
        return v

    @validator("step_size")
    def step_size_validator(cls, v: float | None, values: dict):
        """validate that step size is not None if step type is step_size"""
        assert "step_type" in values
        if values["step_type"] == StepTypes.STEP_SIZE:
            assert v is not None
        return v

    @validator("step_count")
    def step_count_validator(cls, v: int | None, values: dict):
        """validate that step count is not None if step type is step_count"""
        assert "step_type" in values
        assert "range_type" in values
        if values["range_type"] == RangeTypes.VALUES:
            return v
        if values["step_type"] == StepTypes.STEP_COUNT:
            assert v is not None
        return v

    @validator("interpolation_type")
    def interp_type_validator(cls, v: InterpolationTypes, values: dict):
        """validate that step type is valid for interpolation type"""
        assert "step_type" in values
        if v != InterpolationTypes.LINEAR:
            assert values["step_type"] == StepTypes.STEP_COUNT
        return v

    @validator("values", pre=True)
    def values_validator(cls, v: list[float] | None, values: dict):
        """
        validate that values is not None if range type is values, and convert to list
        """
        assert "range_type" in values
        if values["range_type"] == RangeTypes.VALUES:
            assert v is not None
            if isinstance(v, np.ndarray):
                assert v.ndim == 1
                v = v.tolist()
            elif isinstance(v, list):
                pass
            elif isinstance(v, float):
                v = [v]
        return v

    def step_values(self) -> np.ndarray:
        """
        Calculate step values for the step range

        Raises:
            ValueError:
                if interpolation type is log or logdecade and step type is step_size
            ValueError:
                if interpolation type is not known InterpolationTypes
            ValueError:
                if range type is not known RangeTypes
            ValueError:
                if step type is not known StepTypes

        Returns:
            np.ndarray: values to step through
        """
        if self.range_type == RangeTypes.VALUES:
            assert self.values is not None
            return np.array(self.values)
        if self.range_type == RangeTypes.START_STOP:
            assert self.start is not None
            assert self.stop is not None
            if self.step_type == StepTypes.STEP_SIZE:
                assert self.step_size is not None
                if self.interpolation_type == InterpolationTypes.LINEAR:
                    return np.arange(self.start, self.stop, self.step_size)
                estr = "Invalid interpolation type {i_type} for step size"
                raise ValueError(estr.format(i_type=self.interpolation_type))
            if self.step_type == StepTypes.STEP_COUNT:
                assert self.step_count is not None
                if self.interpolation_type == InterpolationTypes.LINEAR:
                    return np.linspace(self.start, self.stop, self.step_count)
                if self.interpolation_type == InterpolationTypes.LOG:
                    return np.geomspace(self.start, self.stop, self.step_count)
                if self.interpolation_type == InterpolationTypes.LOGDECADE:
                    return decadespace(self.start, self.stop, self.step_count)
                raise ValueError("Invalid interpolation type")
            raise ValueError("Invalid step type")
        if self.range_type == RangeTypes.CENTER_SPAN:
            assert self.center is not None
            assert self.span is not None
            if self.step_type == StepTypes.STEP_SIZE:
                assert self.step_size is not None
                if self.interpolation_type == InterpolationTypes.LINEAR:
                    return np.arange(
                        self.center - self.span / 2,
                        self.center + self.span / 2,
                        self.step_size,
                    )
                estr = "Invalid interpolation type {i_type} for step size"
                raise ValueError(estr.format(i_type=self.interpolation_type))
            if self.step_type == StepTypes.STEP_COUNT:
                assert self.step_count is not None
                if self.interpolation_type == InterpolationTypes.LINEAR:
                    return np.linspace(
                        self.center - self.span / 2,
                        self.center + self.span / 2,
                        self.step_count,
                    )
                if self.interpolation_type == InterpolationTypes.LOG:
                    return np.geomspace(
                        self.center - self.span / 2,
                        self.center + self.span / 2,
                        self.step_count,
                    )
                if self.interpolation_type == InterpolationTypes.LOGDECADE:
                    return decadespace(
                        self.center - self.span / 2,
                        self.center + self.span / 2,
                        self.step_count,
                    )
                raise ValueError("Invalid interpolation type")
            raise ValueError("Invalid step type")
        raise ValueError("Invalid range type")

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
                if range type is not known RangeTypes
            ValueError:
                if step type is not known StepTypes
        """
        if self.range_type == RangeTypes.VALUES:
            assert self.values is not None
            self.start = None
            self.stop = None
            self.center = None
            self.span = None
            self.step_count = len(self.values)
        else:
            if self.range_type == RangeTypes.START_STOP:
                assert self.start is not None
                assert self.stop is not None
                self.center = (self.start + self.stop) / 2
                self.span = self.stop - self.start
            elif self.range_type == RangeTypes.CENTER_SPAN:
                assert self.center is not None
                assert self.span is not None
                self.start = self.center - self.span / 2
                self.stop = self.center + self.span / 2
            else:
                raise ValueError("Invalid range type")

            if self.interpolation_type == InterpolationTypes.LINEAR:
                if self.step_type == StepTypes.STEP_SIZE:
                    assert self.step_size is not None
                    assert self.stop is not None
                    assert self.start is not None
                    self.step_count = int((self.stop - self.start) / self.step_size) + 1
                elif self.step_type == StepTypes.STEP_COUNT:
                    assert self.step_count is not None
                    assert self.stop is not None
                    assert self.start is not None
                    self.step_size = (self.stop - self.start) / (self.step_count - 1)
                else:
                    raise ValueError("Invalid step type")


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
