import json
from numbers import Number

import bson
import numpy as np
import pytest

from svalbard.data_model.measurement import step_range as aq_step_range
from svalbard.data_model.measurement.step_range import (
    DECADES,
    InterpolationTypes,
    RangeTypes,
    StepRange,
    StepTypes,
    decadespace,
)

step_ranges = {
    "start_stop_linear_count": StepRange(
        range_type=RangeTypes.START_STOP,
        step_type=StepTypes.STEP_COUNT,
        interpolation_type=InterpolationTypes.LINEAR,
        start=0.0,
        stop=1.0,
        step_count=11,
    ),
    "start_stop_linear_size": StepRange(
        range_type=RangeTypes.START_STOP,
        step_type=StepTypes.STEP_SIZE,
        interpolation_type=InterpolationTypes.LINEAR,
        start=0.0,
        stop=1.0,
        step_size=0.1,
    ),
    "start_stop_log_count": StepRange(
        range_type=RangeTypes.START_STOP,
        step_type=StepTypes.STEP_COUNT,
        interpolation_type=InterpolationTypes.LOG,
        start=0.1,
        stop=1.0,
        step_count=10,
    ),
    "start_stop_logdecade": StepRange(  # disable this option.
        range_type=RangeTypes.START_STOP,
        step_type=StepTypes.STEP_COUNT,
        interpolation_type=InterpolationTypes.LOGDECADE,
        start=0.1,
        stop=1.0,
        step_count=3,
    ),
    "center_span_linear_count": StepRange(
        range_type=RangeTypes.CENTER_SPAN,
        step_type=StepTypes.STEP_COUNT,
        interpolation_type=InterpolationTypes.LINEAR,
        center=0.5,
        span=1.0,
        step_count=11,
    ),
    "center_span_linear_size": StepRange(
        range_type=RangeTypes.CENTER_SPAN,
        step_type=StepTypes.STEP_SIZE,
        interpolation_type=InterpolationTypes.LINEAR,
        center=0.5,
        span=1.0,
        step_size=0.1,
    ),
    "center_span_log_count": StepRange(
        range_type=RangeTypes.CENTER_SPAN,
        step_type=StepTypes.STEP_COUNT,
        interpolation_type=InterpolationTypes.LOG,
        center=0.6,
        span=1.0,
        step_count=11,
    ),
    "center_span_logdecade": StepRange(
        range_type=RangeTypes.CENTER_SPAN,
        step_type=StepTypes.STEP_COUNT,
        interpolation_type=InterpolationTypes.LOGDECADE,
        center=0.6,
        span=1.0,
        step_count=3,
    ),
    "values": StepRange(
        range_type=RangeTypes.VALUES,
        values=[0.0, 0.4, 0.1, 0.3, 0.2],
    ),
    "values_np": StepRange(
        range_type=RangeTypes.VALUES,
        values=np.array([0.0, 0.4, 0.1, 0.3, 0.2]),  # type: ignore
    ),
    "values_single": StepRange(
        range_type=RangeTypes.VALUES,
        values=2.0,  # type: ignore
    ),
    "values_list": StepRange(
        range_type=RangeTypes.VALUES,
        values=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # type: ignore
    ),
    "values_str": StepRange(
        range_type=RangeTypes.VALUES, values=["a", "b", "c", "d", "e"]
    ),
}


@pytest.mark.parametrize("step_range", step_ranges.values(), ids=step_ranges.keys())
def test_step_range_inits(step_range: StepRange):
    """Test that StepRange can be initialized"""
    step_range.step_values()
    step_range.update_parameters()


@pytest.mark.parametrize("step_range", step_ranges.values(), ids=step_ranges.keys())
def test_step_range_bson(step_range: StepRange):
    bson_encoded = bson.BSON.encode(json.loads(step_range.model_dump_json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_step_range = StepRange(**bson_decoded)
    assert new_step_range == step_range


step_range_expected_values = {
    "start_stop_linear_count": np.linspace(0.0, 1.0, 11),
    "start_stop_linear_size": np.arange(0.0, 1.0, 0.1),
    "start_stop_log_count": np.geomspace(0.1, 1.0, 10),
    "start_stop_logdecade": np.array([0.1, 0.2, 0.5, 1.0]),
    "center_span_linear_count": np.linspace(0.0, 1.0, 11),
    "center_span_linear_size": np.arange(0.0, 1.0, 0.1),
    "center_span_log_count": np.geomspace(0.1, 1.1, 11),
    "center_span_logdecade": np.array([0.1, 0.2, 0.5, 1.0, 1.1]),
    "values": np.array([0.0, 0.4, 0.1, 0.3, 0.2]),
    "values_np": np.array([0.0, 0.4, 0.1, 0.3, 0.2]),
    "values_single": np.array([2.0]),
    "values_list": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    "values_str": np.array(["a", "b", "c", "d", "e"]),
}


@pytest.mark.parametrize(
    "step_range, expected",
    [
        (sr, e)
        for sr, e in zip(step_ranges.values(), step_range_expected_values.values())
    ],
    ids=step_ranges.keys(),
)
def test_step_values_expected(step_range: StepRange, expected: np.ndarray):
    try:
        generated = step_range.step_values()
        assert isinstance(generated, np.ndarray)
        assert np.allclose(generated, expected)
    except (np.exceptions.DTypePromotionError, AssertionError):
        assert np.all(step_range.step_values() == expected)

    for index, expected_value in enumerate(expected):
        try:
            assert np.allclose(
                step_range.get_step_value(index),  # type: ignore
                expected_value,
            )
        except (np.exceptions.DTypePromotionError, AssertionError):
            assert step_range.get_step_value(index) == expected_value


step_range_expected_n_steps = {
    "start_stop_linear_count": 11,
    "start_stop_linear_size": 10,
    "start_stop_log_count": 10,
    "start_stop_logdecade": 4,
    "center_span_linear_count": 11,
    "center_span_linear_size": 10,
    "center_span_log_count": 11,
    "center_span_logdecade": 5,
    "values": 5,
    "values_np": 5,
    "values_single": 1,
    "values_list": 3,
    "values_str": 5,
}


@pytest.mark.parametrize(
    "step_range, expected",
    [
        (sr, e)
        for sr, e in zip(step_ranges.values(), step_range_expected_n_steps.values())
    ],
    ids=step_ranges.keys(),
)
def test_step_values_expected_n_steps(step_range: StepRange, expected: np.ndarray):
    assert step_range.n_steps() == expected


def test_n_steps_errors():
    with pytest.raises(ValueError):
        aq_step_range.StartStopRange(1, 2).n_steps()

    with pytest.raises(ValueError):
        aq_step_range.CenterSpanRange(1, 2).n_steps()


step_range_expected_range_strings = {
    "start_stop_linear_count": ("0 V - 1 V", "11 pts"),
    "start_stop_linear_size": ("0 V - 900 mV", "100 mV"),
    "start_stop_log_count": ("100 mV - 1 V", "10 pts"),
    "start_stop_logdecade": ("100 mV - 1 V", "4 pts"),
    "center_span_linear_count": ("c = 500 mV, w = 1 V", "11 pts"),
    "center_span_linear_size": ("c = 500 mV, w = 1 V", "100 mV"),
    "center_span_log_count": ("c = 600 mV, w = 1 V", "11 pts"),
    "center_span_logdecade": ("c = 600 mV, w = 1 V", "5 pts"),
    "values": ("0 V - 400 mV", "5 pts"),
    "values_np": ("0 V - 400 mV", "5 pts"),
    "values_single": ("2 V - 2 V", "1 pts"),
    "values_list": ("", "3 pts"),
    "values_str": ("", "5 pts"),
}


@pytest.mark.parametrize(
    "step_range, expected",
    [
        (sr, e)
        for sr, e in zip(
            step_ranges.values(), step_range_expected_range_strings.values()
        )
    ],
    ids=step_ranges.keys(),
)
def test_step_values_expected_range_strings(
    step_range: StepRange, expected: np.ndarray
):
    assert step_range.get_range_strings("V") == expected


step_range_update_paremeters_expected = {
    "start_stop_linear_count": {
        "start": 0.0,
        "stop": 1.0,
        "step_count": 11,
        "step_size": 0.1,
        "span": 1.0,
        "center": 0.5,
        "values": None,
    },
    "start_stop_linear_size": {
        "start": 0.0,
        "stop": 1.0,
        "step_count": 11,
        "step_size": 0.1,
        "span": 1.0,
        "center": 0.5,
        "values": None,
    },
    "start_stop_log_count": {
        "start": 0.1,
        "stop": 1.0,
        "step_count": 10,
        "step_size": None,
        "span": 0.9,
        "center": 0.55,
        "values": None,
    },
    "start_stop_logdecade": {
        "start": 0.1,
        "stop": 1.0,
        "step_count": 3,
        "step_size": None,
        "span": 0.9,
        "center": 0.55,
        "values": None,
    },
    "center_span_linear_count": {
        "start": 0.0,
        "stop": 1.0,
        "step_count": 11,
        "step_size": 0.1,
        "span": 1.0,
        "center": 0.5,
        "values": None,
    },
    "center_span_linear_size": {
        "start": 0.0,
        "stop": 1.0,
        "step_count": 11,
        "step_size": 0.1,
        "span": 1.0,
        "center": 0.5,
        "values": None,
    },
    "center_span_log_count": {
        "start": 0.1,
        "stop": 1.1,
        "step_count": 11,
        "step_size": None,
        "span": 1.0,
        "center": 0.6,
        "values": None,
    },
    "center_span_logdecade": {
        "start": 0.1,
        "stop": 1.1,
        "step_count": 3,
        "step_size": None,
        "span": 1.0,
        "center": 0.6,
        "values": None,
    },
    "values": {
        "start": None,
        "stop": None,
        "step_count": 5,
        "step_size": None,
        "span": None,
        "center": None,
        "values": [0.0, 0.4, 0.1, 0.3, 0.2],
    },
    "values_np": {
        "start": None,
        "stop": None,
        "step_count": 5,
        "step_size": None,
        "span": None,
        "center": None,
        "values": [0.0, 0.4, 0.1, 0.3, 0.2],
    },
    "values_single": {
        "start": None,
        "stop": None,
        "step_count": 1,
        "step_size": None,
        "span": None,
        "center": None,
        "values": [2.0],
    },
    "values_list": {
        "start": None,
        "stop": None,
        "step_count": 3,
        "step_size": None,
        "span": None,
        "center": None,
        "values": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    },
    "values_str": {
        "start": None,
        "stop": None,
        "step_count": 5,
        "step_size": None,
        "span": None,
        "center": None,
        "values": ["a", "b", "c", "d", "e"],
    },
}


@pytest.mark.parametrize(
    "step_range, expected",
    [
        (sr, e)
        for sr, e in zip(
            step_ranges.values(), step_range_update_paremeters_expected.values()
        )
    ],
    ids=step_ranges.keys(),
)
def test_update_parameters_expected(step_range: StepRange, expected):
    """Test that StepRange.update_parameters() sets the correct values"""
    step_range.update_parameters()
    for k, v in step_range.model_dump().items():
        if k in expected:
            if isinstance(v, Number):
                assert np.allclose(v, expected[k])  # type: ignore
            else:
                assert v == expected[k]


def test_step_values_errors(step_range_center_span_linear_count: StepRange):
    """Test that StepRange.step_values() raises errors"""
    step_range_center_span_linear_count.update_parameters()
    step_range_center_span_linear_count.interpolation_type = "invalid"  # type: ignore

    with pytest.raises(ValueError):
        step_range_center_span_linear_count.step_values()

    step_range_center_span_linear_count.step_type = "invalid"  # type: ignore
    with pytest.raises(ValueError):
        step_range_center_span_linear_count.step_values()

    step_range_center_span_linear_count.range_type = RangeTypes.START_STOP
    step_range_center_span_linear_count.step_type = StepTypes.STEP_COUNT
    with pytest.raises(ValueError):
        step_range_center_span_linear_count.step_values()

    step_range_center_span_linear_count.step_type = "invalid"  # type: ignore
    with pytest.raises(ValueError):
        step_range_center_span_linear_count.step_values()

    step_range_center_span_linear_count.range_type = "invalid"  # type: ignore
    with pytest.raises(ValueError):
        step_range_center_span_linear_count.step_values()

    step_range_center_span_linear_count.range_type = RangeTypes.START_STOP
    step_range_center_span_linear_count.step_type = StepTypes.STEP_SIZE
    step_range_center_span_linear_count.interpolation_type = InterpolationTypes.LOG
    with pytest.raises(ValueError):
        step_range_center_span_linear_count.step_values()

    step_range_center_span_linear_count.range_type = RangeTypes.CENTER_SPAN
    step_range_center_span_linear_count.step_type = StepTypes.STEP_SIZE
    step_range_center_span_linear_count.interpolation_type = InterpolationTypes.LOG
    with pytest.raises(ValueError):
        step_range_center_span_linear_count.step_values()


def test_step_range_update_parameters_errors(
    step_range_center_span_linear_count: StepRange,
):
    step_range_center_span_linear_count.step_type = "invalid"  # type: ignore
    with pytest.raises(ValueError):
        step_range_center_span_linear_count.update_parameters()

    step_range_center_span_linear_count.range_type = "invalid"  # type: ignore
    with pytest.raises(ValueError):
        step_range_center_span_linear_count.update_parameters()


def test_step_range_update_parameters_errors_size_count(
    step_range_center_span_linear_count: StepRange,
):
    step_range_center_span_linear_count.update_parameters()
    step_range_center_span_linear_count.stop = None
    with pytest.raises(ValueError):
        aq_step_range._update_parameter_step_size_and_count(
            step_range_center_span_linear_count
        )

    step_range_center_span_linear_count.start = None
    with pytest.raises(ValueError):
        aq_step_range._update_parameter_step_size_and_count(
            step_range_center_span_linear_count
        )


def test_step_range_start_stop_error():
    with pytest.raises(ValueError):
        StepRange(
            range_type=RangeTypes.START_STOP,
            step_type=StepTypes.STEP_COUNT,
            interpolation_type=InterpolationTypes.LINEAR,
            start=None,
            stop=1.0,
            step_count=1,
        )

    with pytest.raises(ValueError):
        StepRange(
            range_type=RangeTypes.START_STOP,
            step_type=StepTypes.STEP_COUNT,
            interpolation_type=InterpolationTypes.LINEAR,
            start=0.1,
            stop=None,
            step_count=1,
        )


def test_step_range_center_span_error():
    with pytest.raises(ValueError):
        StepRange(
            range_type=RangeTypes.CENTER_SPAN,
            step_type=StepTypes.STEP_COUNT,
            interpolation_type=InterpolationTypes.LINEAR,
            center=None,
            span=1.0,
            step_count=1,
        )

    with pytest.raises(ValueError):
        StepRange(
            range_type=RangeTypes.CENTER_SPAN,
            step_type=StepTypes.STEP_COUNT,
            interpolation_type=InterpolationTypes.LINEAR,
            center=0.1,
            span=None,
            step_count=1,
        )


def test_step_range_step_size_error():
    with pytest.raises(ValueError):
        StepRange(
            range_type=RangeTypes.CENTER_SPAN,
            step_type=StepTypes.STEP_SIZE,
            interpolation_type=InterpolationTypes.LINEAR,
            center=0.1,
            span=1.0,
            step_size=None,
        )

    with pytest.raises(ValueError):
        StepRange(
            range_type=RangeTypes.CENTER_SPAN,
            step_type=StepTypes.STEP_SIZE,
            interpolation_type=InterpolationTypes.LINEAR,
            center=0.1,
            span=1.0,
            step_size=0.0,
        )


def test_step_range_step_count_error():
    with pytest.raises(ValueError):
        StepRange(
            range_type=RangeTypes.CENTER_SPAN,
            step_type=StepTypes.STEP_COUNT,
            interpolation_type=InterpolationTypes.LINEAR,
            center=0.1,
            span=1.0,
            step_count=None,
        )

    with pytest.raises(ValueError):
        StepRange(
            range_type=RangeTypes.CENTER_SPAN,
            step_type=StepTypes.STEP_COUNT,
            interpolation_type=InterpolationTypes.LINEAR,
            center=0.1,
            span=1.0,
            step_count=0,
        )


def test_step_range_interpolation_type_error():
    with pytest.raises(ValueError):
        StepRange(
            range_type=RangeTypes.CENTER_SPAN,
            step_type=StepTypes.STEP_SIZE,
            interpolation_type=InterpolationTypes.LOG,
            center=0.1,
            span=1.0,
            step_size=1,
        )


def test_step_range_numpy_dim_error():
    with pytest.raises(ValueError):
        StepRange(
            range_type=RangeTypes.VALUES,
            step_type=StepTypes.STEP_SIZE,
            interpolation_type=InterpolationTypes.LOG,
            center=0.1,
            span=1.0,
            values=np.array([[1, 2, 3], [2, 4, 5]]),  # type: ignore
        )


def test_step_range_values_error():
    with pytest.raises(ValueError):
        StepRange(
            range_type=RangeTypes.VALUES,
            step_type=StepTypes.STEP_SIZE,
            interpolation_type=InterpolationTypes.LINEAR,
            center=0.1,
            span=1.0,
            step_size=1,
            values=None,
        )


def test_step_range_step_values_values_error():
    step_range_values = StepRange(
        range_type=RangeTypes.VALUES,
        values=[1.0],
    )
    step_range_values.update_parameters()
    step_range_values.step_values()
    step_range_values.values = None
    with pytest.raises(ValueError):
        step_range_values.step_values()


def test_step_range_step_values_size_error():
    step_range_size = StepRange(
        step_type=StepTypes.STEP_SIZE,
        start=0.0,
        stop=1.0,
        step_size=0.1,
    )
    step_range_size.update_parameters()
    step_range_size.step_values()
    step_range_size.step_size = None
    with pytest.raises(ValueError):
        step_range_size.step_values()


def test_step_range_step_values_count_error():
    step_range_count = StepRange(
        start=0.0,
        stop=1.0,
        step_count=11,
    )
    step_range_count.update_parameters()
    step_range_count.step_values()
    step_range_count.step_count = None
    with pytest.raises(ValueError):
        step_range_count.step_values()


def test_step_range_step_values_start_stop_error():
    step_range_start_stop = StepRange(
        start=0.0,
        stop=1.0,
        step_count=11,
    )
    step_range_start_stop.update_parameters()
    step_range_start_stop.step_values()
    step_range_start_stop.start = None
    with pytest.raises(ValueError):
        step_range_start_stop.step_values()

    step_range_start_stop.start = 0.0
    step_range_start_stop.step_values()
    step_range_start_stop.stop = None
    with pytest.raises(ValueError):
        step_range_start_stop.step_values()


def test_step_range_step_values_center_span_error():
    step_range_center_span = StepRange(
        range_type=RangeTypes.CENTER_SPAN,
        center=0.5,
        span=1.0,
        step_count=11,
    )
    step_range_center_span.update_parameters()
    step_range_center_span.step_values()
    step_range_center_span.center = None
    with pytest.raises(ValueError):
        step_range_center_span.step_values()

    step_range_center_span.center = 0.5
    step_range_center_span.step_values()
    step_range_center_span.span = None
    with pytest.raises(ValueError):
        step_range_center_span.step_values()


def test_step_range_update_paramters_values_error():
    step_range_values = StepRange(
        range_type=RangeTypes.VALUES,
        values=[1.0],
    )
    step_range_values.update_parameters()
    step_range_values.values = None
    with pytest.raises(ValueError):
        step_range_values.update_parameters()


def test_step_range_update_paramters_size_error():
    step_range_size = StepRange(
        step_type=StepTypes.STEP_SIZE,
        start=0.0,
        stop=1.0,
        step_size=0.1,
    )
    step_range_size.update_parameters()
    step_range_size.step_size = None
    with pytest.raises(ValueError):
        step_range_size.update_parameters()

    step_range_size.step_size = 0.1
    step_range_size.update_parameters()
    step_range_size.interpolation_type = InterpolationTypes.LOG
    with pytest.raises(ValueError):
        step_range_size.update_parameters()


def test_step_range_update_paramters_count_error():
    step_range_count = StepRange(
        start=0.0,
        stop=1.0,
        step_count=11,
    )
    step_range_count.update_parameters()
    step_range_count.step_count = None
    with pytest.raises(ValueError):
        step_range_count.update_parameters()


def test_step_range_update_paramters_start_stop_error():
    step_range_start_stop = StepRange(
        start=0.0,
        stop=1.0,
        step_count=11,
    )
    step_range_start_stop.update_parameters()
    step_range_start_stop.start = None
    with pytest.raises(ValueError):
        step_range_start_stop.update_parameters()

    step_range_start_stop.start = 0.0
    step_range_start_stop.update_parameters()
    step_range_start_stop.stop = None
    with pytest.raises(ValueError):
        step_range_start_stop.update_parameters()


def test_step_range_update_paramters_center_span_error():
    step_range_center_span = StepRange(
        range_type=RangeTypes.CENTER_SPAN,
        center=0.5,
        span=1.0,
        step_count=11,
    )
    step_range_center_span.update_parameters()
    step_range_center_span.center = None
    with pytest.raises(ValueError):
        step_range_center_span.update_parameters()

    step_range_center_span.center = 0.5
    step_range_center_span.update_parameters()
    step_range_center_span.span = None
    with pytest.raises(ValueError):
        step_range_center_span.update_parameters()


@pytest.mark.parametrize("decade", DECADES)
def test_decadespace_targeted(decade):
    dspace = decadespace(1, 10, len(decade))
    assert np.allclose(dspace[:-1], decade)


def test_decadespace_targeted_2():
    dspace = decadespace(0.1, 1.1, 3)
    assert np.allclose(dspace, np.array([0.1, 0.2, 0.5, 1.0, 1.1]))
    dspace2 = decadespace(0.6 - 1.0 / 2, 0.6 + 1.0 / 2, 3)
    assert np.allclose(dspace2, np.array([0.1, 0.2, 0.5, 1.0, 1.1]))


def test_decadespace_large_points_per_decade():
    dspace = decadespace(0.1, 10, 20)
    assert np.allclose(dspace, np.geomspace(0.1, 10, 41))


def test_decadespace_small_points_per_decade():
    dspace = decadespace(0.1, 1000, 0.5)  # type: ignore
    assert np.allclose(dspace, np.geomspace(0.1, 1000, 3))


# @pytest.mark.parametrize("start", np.linspace(0.1, 1.0, 10))
# @pytest.mark.parametrize("stop", np.linspace(0.2, 1.0, 10))
# @pytest.mark.parametrize("step_count", np.arange(0.5, 20, 1))
# def test_decadespace_big_bang(start, stop, step_count):
#     decadespace(start, 10 * stop, step_count)
