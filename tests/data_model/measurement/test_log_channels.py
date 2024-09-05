import json

import pytest

from svalbard.data_model.measurement.log_channel import LogChannel


@pytest.fixture(name="log_channel")
def fixture_log_channel() -> LogChannel:
    return LogChannel(
        name="log_channel",
        step_names=set(),
    )


def test_log_channel_io(log_channel: LogChannel):
    n_log_channel = LogChannel(**log_channel.model_dump())
    assert n_log_channel == log_channel


def test_log_channel_io_json(log_channel: LogChannel):
    n_log_channel = LogChannel(**json.loads(log_channel.model_dump_json()))
    assert n_log_channel == log_channel

    n_log_channel = LogChannel.model_validate_json(log_channel.model_dump_json())
    assert n_log_channel == log_channel


steps = [("time", 100), ("frequency", 50), ("shots", 1000)]


@pytest.mark.parametrize("shape", [(1,), (8,), (2, 4)])
@pytest.mark.parametrize("shots", [None, steps[2]])
@pytest.mark.parametrize("frequency", [None, steps[1]])
@pytest.mark.parametrize("time", [None, steps[0]])
@pytest.mark.parametrize("inclusive", [True, False])
def test_log_channel_shape(time, frequency, shots, shape, inclusive):
    log_channel_steps = list(
        filter(lambda step: step is not None, [time, frequency, shots])
    )
    step_names = set([step[0] for step in log_channel_steps])
    if inclusive:
        expected_shape = tuple([step[1] for step in log_channel_steps])
    else:
        expected_shape = tuple(
            [
                shapes[1]
                for shapes, include in zip(steps, [time, frequency, shots])
                if include is None
            ]
        )

    log_channel = LogChannel(
        name="log_channel",
        step_names=step_names,
        base_shape=shape,
        inclusive=inclusive,
    )
    shape = () if shape == (1,) and expected_shape != () else shape
    expected_shape = tuple(list(shape) + list(expected_shape))
    idx_size_steps = {
        i: (size, set([step_names])) for i, (step_names, size) in enumerate(steps)
    }
    assert log_channel.shape(idx_size_steps) == expected_shape


def test_step_names_str():
    log_channel = LogChannel(
        name="log_channel",
        step_names="time",  # type: ignore
    )
    assert log_channel.step_names == {"time"}


@pytest.mark.parametrize("shape", [(1,), (8,), (2, 4)])
def test_log_channel_shape_no_steps(shape):
    log_channel = LogChannel(
        name="log_channel",
        base_shape=shape,
    )
    assert log_channel.shape({}) == shape


def test_log_channel_name_pythonic():
    log_channel = LogChannel(name="log channel")
    assert log_channel.name == "log_channel"
