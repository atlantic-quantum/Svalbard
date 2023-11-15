"""functions that are common between multiple test modules of the data server"""
from pathlib import Path

import numpy as np
from svalbard.data_model.data_file import Data, DataFile
from svalbard.data_model.ipc import BufferReference, MeasurementHandle
from svalbard.data_model.memory_models import SharedMemoryIn, SharedMemoryOut
from svalbard.data_server.data_backend.abstract_data_backend import AbstractDataBackend
from svalbard.data_server.frontend.abstract_frontend import AbstractFrontend
from svalbard.utility.resize_array import new_shape


def index(dataset: Data.DataSet, indices: tuple) -> tuple:
    """calculate the index for where to insert data in buffer"""
    # this and the next function should be part fo the module
    return tuple(
        dataset.memory.shape[i] * idx for i, idx in enumerate(indices)
    ) + tuple(0 for _ in dataset.memory.shape[len(indices) :])


def new_slice(buffer: BufferReference, indx: tuple) -> list[slice]:
    """calculate the slices to access new data in buffer"""
    return list(slice(idx, idx + buffer.shape[i]) for i, idx in enumerate(indx))


def buffer_upload_data(
    buffers: list[BufferReference], sweep_shape: tuple
) -> list[np.ndarray]:
    """Generate random data to upload in buffers"""
    shapes = [
        tuple(
            buffer.shape[i] * shape_index for i, shape_index in enumerate(sweep_shape)
        )
        + buffer.shape[len(sweep_shape) :]
        for buffer in buffers
    ]
    return [
        np.arange(np.prod(shape)).reshape(shape)
        for buffer, shape in zip(buffers, shapes)
    ]


async def save_buffer_test(
    buffer: BufferReference,
    dataset: Data.DataSet,
    handle: MeasurementHandle,
    buffer_data: np.ndarray,
    idx: tuple,
    caller: AbstractFrontend | AbstractDataBackend,
    path: Path,
    serial: bool = True,
):
    """function for uploading data in buffer reference,
    downloading it and testing the downloaded data"""

    async def _current_shape(
        caller: AbstractFrontend | AbstractDataBackend, path: Path
    ) -> tuple:
        l_data = await caller.load(path)
        if isinstance(l_data, DataFile):
            l_data = l_data.data
        assert l_data is not None
        l_dataset = [dset for dset in l_data.datasets if dset.name == dataset.name][0]
        [SharedMemoryOut.close(dset.memory.name) for dset in l_data.datasets]
        return l_dataset.memory.shape

    current_shape = await _current_shape(caller, path)
    new_buffer, slices = new_buffer_and_slices(dataset, idx, buffer, buffer_data)
    await caller.save_buffer(handle, dataset.name, new_buffer, index(dataset, idx))
    l_data = await caller.load(path)
    if isinstance(l_data, DataFile):
        l_data = l_data.data
    assert l_data is not None
    l_dataset = [dset for dset in l_data.datasets if dset.name == dataset.name][0]
    assert dataset.memory.dtype == l_dataset.memory.dtype
    if serial:
        assert (
            new_shape(current_shape, slices, dataset.memory.shape)
            == l_dataset.memory.shape
        )
    [SharedMemoryOut.close(dset.memory.name) for dset in l_data.datasets]
    SharedMemoryOut.close(new_buffer.name)
    # assert np.all(buffer_data[slices] == l_dataset.memory.to_array()[slices])


def sync_save_buffer_test(
    buffer: BufferReference,
    dataset: Data.DataSet,
    handle: MeasurementHandle,
    buffer_data: np.ndarray,
    idx: tuple,
    caller: AbstractFrontend | AbstractDataBackend,
    path: Path,
    serial: bool = True,
):
    """function for uploading data in buffer reference,
    downloading it and testing the downloaded data"""

    def _current_shape(
        caller: AbstractFrontend | AbstractDataBackend, path: Path
    ) -> tuple:
        l_data = caller.load(path)
        if isinstance(l_data, DataFile):
            l_data = l_data.data
        assert l_data is not None
        assert isinstance(l_data, Data)
        l_dataset = [dset for dset in l_data.datasets if dset.name == dataset.name][0]
        [SharedMemoryOut.close(dset.memory.name) for dset in l_data.datasets]
        return l_dataset.memory.shape

    current_shape = _current_shape(caller, path)
    new_buffer, slices = new_buffer_and_slices(dataset, idx, buffer, buffer_data)
    caller.save_buffer(
        handle,
        dataset.name,
        new_buffer,
        index(dataset, idx),
    )  # type: ignore
    l_data = caller.load(path)
    if isinstance(l_data, DataFile):
        l_data = l_data.data
    assert l_data is not None
    assert isinstance(l_data, Data)
    l_dataset = [dset for dset in l_data.datasets if dset.name == dataset.name][0]
    assert dataset.memory.dtype == l_dataset.memory.dtype
    if serial:
        assert (
            new_shape(current_shape, slices, dataset.memory.shape)
            == l_dataset.memory.shape
        )
    [SharedMemoryOut.close(dset.memory.name) for dset in l_data.datasets]
    SharedMemoryOut.close(new_buffer.name)
    # assert np.all(buffer_data[slices] == l_dataset.memory.to_array()[slices])


def compare_data_in_datasets(dataset1: np.ndarray, dataset2: np.ndarray):
    if np.isnan(dataset1).all() and np.isnan(dataset2).all():
        return True
    return np.all(dataset1 == dataset2)


def assert_datasets_match(
    datasets: list[Data.DataSet],
    l_datasets: list[Data.DataSet],
    buffer_data: list[np.ndarray] | None = None,
):
    """checks if two datasets matchs

    Args:
        datasets (list[Data.DataSet]): original dataset
        l_datasets (list[Data.DataSet]): loaded dataset
    """
    if buffer_data is None:
        buffer_data = [dataset.memory.to_array() for dataset in datasets]
    assert len(datasets) == len(l_datasets)
    assert len(l_datasets) == len(buffer_data)
    for dataset, l_dataset, b_data in zip(datasets, l_datasets, buffer_data):
        assert dataset.name == l_dataset.name
        assert dataset.memory.dtype == l_dataset.memory.dtype
        assert b_data.shape == l_dataset.memory.shape
        assert compare_data_in_datasets(b_data, l_dataset.memory.to_array())


def assert_partial_datasets_match(
    datasets: list[Data.DataSet],
    l_datasets: list[Data.DataSet],
    slice_lists: list[list[slice]],
    buffer_data: list[np.ndarray] | None = None,
):
    if buffer_data is None:
        buffer_data = [
            dataset.memory.to_array()[tuple(sliced)]
            for dataset, sliced in zip(datasets, slice_lists)
        ]
    assert len(datasets) == len(l_datasets)
    assert len(l_datasets) == len(buffer_data)
    for dataset, l_dataset, b_data in zip(datasets, l_datasets, buffer_data):
        assert dataset.name == l_dataset.name
        assert dataset.memory.dtype == l_dataset.memory.dtype
        assert b_data.shape == l_dataset.memory.shape
        assert compare_data_in_datasets(b_data, l_dataset.memory.to_array())


def assert_metadata_and_handle_match(datafile: DataFile, l_datafile: DataFile):
    """Check that loaded metadata and handle in a datafile match

    Args:
        datafile (DataFile): original datafile
        l_datafile (DataFile): loaded datafile
    """
    assert l_datafile.metadata == datafile.metadata
    if l_datafile.data is not None:
        assert datafile.data is not None
        assert l_datafile.data.handle == datafile.data.handle
    else:
        assert datafile.data is None


def new_buffer_and_slices(
    dataset: Data.DataSet, idx: tuple, buffer: BufferReference, buffer_data: np.ndarray
):
    idex = index(dataset, idx)
    slices = new_slice(buffer, idex)

    smi = SharedMemoryIn(dtype=buffer.dtype, shape=buffer.shape)
    new_buffer = BufferReference.from_memory_in(smi)
    new_buffer.to_array()[:] = buffer_data[tuple(slices)]
    return new_buffer, slices
