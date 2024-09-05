"""Data Model Fixtures"""

from pathlib import Path

import numpy as np
import pytest
from svalbard.data_model.data_file import (
    Data,
    DataFile,
    MaskMetaData,
    MeasurementHandle,
    MetaData,
)
from svalbard.data_model.instruments import (
    InstrumentModel,
    InstrumentSetting,
    SettingType,
)
from svalbard.data_model.ipc import BufferReference, EndStreamModel, StartStreamModel
from svalbard.data_model.memory_models import SharedMemoryIn, SharedMemoryOut


@pytest.fixture(name="instrument_settings")
def fixture_instrument_settings():
    _val_and_setting_type = [
        (1, SettingType.INT),
        (1.0, SettingType.FLOAT),
        (False, SettingType.BOOL),
        ("enum1", SettingType.ENUM),
        (1, SettingType.ENUM),
        ("test", SettingType.STR),
    ]
    yield {
        f"test_name{i}": InstrumentSetting(
            name=f"test_name{i}", value=val, dtype=setting_type
        )
        for i, (val, setting_type) in enumerate(_val_and_setting_type)
    }


@pytest.fixture(name="metadata")
def fixture_metadata(instrument_settings: dict[str, InstrumentSetting]):
    """MetaData Fixture"""
    yield MetaData(
        name="test_name",
        tags=["test_tag1", "test_tag2"],
        instruments={
            "test_hardware": InstrumentModel(
                identity="test_hardware",
                hardware="test_hardware",
                model="test_model",
                settings=instrument_settings,
            )
        },
    )


@pytest.fixture(name="mask_metadata")
def fixture_mask_metadata():
    """MaskMetaData fixture"""
    yield MaskMetaData(
        name="test_name", data_path=Path("svalbard/tests/"), files=[Path("test.gds")]
    )


@pytest.fixture(name="shapes_and_dtypes")
def fixture_shapes_and_dtypes():
    """Fixture for creating various shapes and datatypes"""
    shapes = [(100,), (10, 10), (3, 3, 3), (2, 2, 2, 10)]
    dtypes = [np.dtype("float"), np.dtype("float"), np.dtype("int"), np.dtype("bool")]
    yield shapes, dtypes


@pytest.fixture(name="shared_memories")
def fixture_shared_memories(shapes_and_dtypes):
    """Shared Memories Fixture"""
    shapes, dtypes = shapes_and_dtypes
    mems_in = [
        SharedMemoryIn(dtype=dtype, shape=shape)
        for (shape, dtype) in zip(shapes, dtypes)
    ]
    mems_out = [SharedMemoryOut.from_memory_in(mem_in) for mem_in in mems_in]
    rng = np.random.default_rng()
    for mem_out in mems_out:
        mem_out.to_array()[:] = rng.standard_normal(size=mem_out.shape).astype(
            mem_out.dtype
        )

    yield mems_out


@pytest.fixture(name="datasets")
def fixture_datasets(shared_memories: list[SharedMemoryOut]):
    """Datasets Fixture"""
    yield [
        Data.DataSet(name=f"test_name_{i}", memory=memory)
        for i, memory in enumerate(shared_memories)
    ]


@pytest.fixture(name="data")
def fixture_data(datasets):
    """Data fixture"""
    yield Data(handle=MeasurementHandle.new(), datasets=datasets)


@pytest.fixture(name="buffer_references")
def fixture_buffer_references():
    """BufferReferences fixture"""
    buffer_sizes = [
        (1, 10),
        (1, 5, 2),
    ]
    smis = [
        SharedMemoryIn(dtype="float64", shape=buffer_size)
        for buffer_size in buffer_sizes
    ]
    buffer_references = [BufferReference.from_memory_in(smi) for smi in smis]
    yield buffer_references


@pytest.fixture(name="buffer_references_2d")
def fixture_buffer_references_2d():
    """BufferReferences fixture"""
    buffer_sizes = [
        (1, 1, 10),
        (1, 1, 5, 2),
    ]
    smis = [
        SharedMemoryIn(dtype="float64", shape=buffer_size)
        for buffer_size in buffer_sizes
    ]
    buffer_references = [BufferReference.from_memory_in(smi) for smi in smis]
    yield buffer_references


@pytest.fixture(name="streamed_datasets")
def fixture_streamed_datasets(buffer_references: list[BufferReference]):
    """DataSet fixture for streamed data"""
    yield [
        Data.DataSet(name=f"test_name_{i}", memory=memory)
        for i, memory in enumerate(buffer_references)
    ]


@pytest.fixture(name="streamed_datasets_2d")
def fixture_streamed_datasets_2d(buffer_references_2d: list[BufferReference]):
    """DataSet fixture for streamed data"""
    yield [
        Data.DataSet(name=f"test_name_{i}", memory=memory)
        for i, memory in enumerate(buffer_references_2d)
    ]


@pytest.fixture(name="streamed_data")
def fixture_stream_setup(streamed_datasets: list[Data.DataSet]):
    """StreamingSetup fixture"""
    yield Data(handle=MeasurementHandle.new(), datasets=streamed_datasets)


@pytest.fixture(name="streamed_data_2d")
def fixture_stream_setup_2d(streamed_datasets_2d: list[Data.DataSet]):
    """StreamingSetup fixture"""
    yield Data(handle=MeasurementHandle.new(), datasets=streamed_datasets_2d)


@pytest.fixture(name="data_file")
def fixture_data_file(data: Data, metadata: MetaData):
    """Fixture for creating a DataFile"""
    yield DataFile(data=data, metadata=metadata)


@pytest.fixture(name="streamed_data_file")
def fixture_streamed_data_file(streamed_data: Data, metadata: MetaData):
    """Fixture for creating a DataFile for streaming"""
    yield DataFile(data=streamed_data, metadata=metadata)


@pytest.fixture(name="start_stream_model")
def fixture_start_stream(streamed_data_file: DataFile):
    """Fixture for creating a StartStreamModel"""
    handle = MeasurementHandle.new()
    assert streamed_data_file.data is not None
    streamed_data_file.data.handle = handle
    yield StartStreamModel(handle=handle, data_file=streamed_data_file)


@pytest.fixture(name="end_stream_model")
def fixture_end_stream(start_stream_model: StartStreamModel):
    """Fixture for creating a EndStreamModel"""
    yield EndStreamModel(handle=start_stream_model.handle)


@pytest.fixture(name="streamed_data_file_2d")
def fixture_streamed_data_file_2d(streamed_data_2d: Data, metadata: MetaData):
    """Fixture for creating a DataFile for streaming"""
    return DataFile(data=streamed_data_2d, metadata=metadata)


@pytest.fixture(name="instruments")
def fixture_instruments():
    instruments = {
        "test_instrument1": InstrumentModel(
            identity="test_instrument1",
            settings={
                "test_setting1": InstrumentSetting(
                    name="test_setting1",
                    dtype=SettingType.FLOAT,
                    value=1.0,
                )
            },
        ),
        "test_instrument2": InstrumentModel(
            identity="test_instrument2",
            settings={
                "test_setting2": InstrumentSetting(
                    name="test_setting2",
                    dtype=SettingType.FLOAT,
                    value=1.0,
                )
            },
        ),
        "test_instrument3": InstrumentModel(
            identity="test_instrument3",
            settings={
                "test_setting3": InstrumentSetting(
                    name="test_setting3",
                    dtype=SettingType.COMPLEX,
                    value=1.0,
                )
            },
        ),
    }
    yield instruments
