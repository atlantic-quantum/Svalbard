"""Test fsspec FileSystem based backend using a MemoryFileSystem"""
import asyncio
import filecmp
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from fsspec.implementations.memory import MemoryFileSystem
from svalbard.data_model.data_file import Data, MeasurementHandle
from svalbard.data_model.ipc import BufferReference
from svalbard.data_model.memory_models import (
    SharedMemoryIn,
    SharedMemoryOut,
    __memory_reference__,
)
from svalbard.data_server.data_backend.fs_backend import FSBackend, FSBackendConfig
from svalbard.data_server.errors import (
    BufferShapeError,
    DataNotInitializedError,
    StreamAlreadyPreparedError,
    StreamNotPreparedError,
)

from ..utility import assert_datasets_match, buffer_upload_data, save_buffer_test


@pytest.mark.asyncio
async def test_init_backend_from_config(fs_config: FSBackendConfig):
    """Test creating a fs backend from fs backend config"""
    fs_config.init()


@pytest.mark.asyncio
async def test_end_to_end_save_load(event_loop):
    """End to end save/load tests
    that contains all the steps instead of using fixture"""
    # create data
    smi = SharedMemoryIn(dtype="float", shape=(10, 10))
    smo = SharedMemoryOut.from_memory_in(smi)
    smo.to_array()[:] = np.arange(100).reshape((10, 10))
    datasets = [Data.DataSet(name=f"test_name_{i}", memory=smo) for i in range(4)]
    data = Data(handle=MeasurementHandle.new(), datasets=datasets)

    # create the backend
    mfs = MemoryFileSystem()
    path_base = Path("memory:/tmp/test/fs_backend")
    # event loop is a asyncio loop createdd for tests marked asyncio
    data_backend = FSBackend(mfs, loop=event_loop, path_base=path_base)

    # save data
    path = await data_backend.save(data)
    assert str(path) == f"{path_base/str(data.handle.handle)}"

    # load data
    l_data = await data_backend.load(path)
    assert data.handle.handle == l_data.handle.handle
    for dataset, l_dataset in zip(data.datasets, l_data.datasets):
        assert dataset.name == l_dataset.name
        assert dataset.memory.shape == l_dataset.memory.shape
        assert dataset.memory.dtype == l_dataset.memory.dtype
        assert np.all(dataset.memory.to_array() == l_dataset.memory.to_array())

    # cleanup to have the test not leave files, should not be done in production
    try:
        mfs.rm(str(path_base), recursive=True)
    except FileNotFoundError:
        pass
    SharedMemoryOut.close(smo.name)
    [SharedMemoryOut.close(dset.memory.name) for dset in l_data.datasets]


@pytest.mark.asyncio
async def test_end_to_end_streaming(event_loop):
    """End to end streaming test
    that contains all the steps instead of using fixture"""
    # create buffers
    buffer_sizes = [
        (1, 10),
        (1, 5, 2),
    ]
    smis = [
        SharedMemoryIn(dtype="float64", shape=buffer_size)
        for buffer_size in buffer_sizes
    ]
    buffer_references = [BufferReference.from_memory_in(smi) for smi in smis]

    # create Data object to set up streaming
    datasets = [
        Data.DataSet(name=f"test_name_{i}", memory=memory)
        for i, memory in enumerate(buffer_references)
    ]
    data = Data(handle=MeasurementHandle.new(), datasets=datasets)

    # create array with data to stream from
    data_sizes = [
        (10, 10),
        (10, 5, 2),
    ]
    streamed_data = [np.arange(np.prod(shape)).reshape(shape) for shape in data_sizes]

    # create the backend
    mfs = MemoryFileSystem()
    path_base = Path("memory:/tmp/test/fs_backend")
    # event loop is a asyncio loop createdd for tests marked asyncio
    data_backend = FSBackend(mfs, loop=event_loop, path_base=path_base)

    # prepare for stream
    path = await data_backend.prepare_stream(data)
    assert str(path) == f"{path_base/str(data.handle.handle)}"

    # stream data
    for j, buffer in enumerate(buffer_references):
        for i in range(10):
            buffer.to_array()[:] = streamed_data[j][i]
            await data_backend.save_buffer(
                data.handle,
                datasets[j].name,
                buffer,
                (i,) + tuple(0 for _ in buffer.shape[1:]),
            )

    # finalize stream
    await data_backend.finalize_stream(data.handle)

    # load data and compare to uploaded.
    l_data = await data_backend.load(path)
    assert data.handle.handle == l_data.handle.handle
    for dataset, l_dataset, streamed in zip(
        data.datasets, l_data.datasets, streamed_data
    ):
        assert dataset.name == l_dataset.name
        assert streamed.shape == l_dataset.memory.shape
        assert dataset.memory.dtype == l_dataset.memory.dtype
        assert np.all(streamed == l_dataset.memory.to_array())

    # cleanup to have the test not leave files, should not be done in production
    try:
        mfs.rm(str(path_base), recursive=True)
    except FileNotFoundError:
        pass

    [SharedMemoryOut.close(br.name) for br in buffer_references]
    [SharedMemoryOut.close(dset.memory.name) for dset in l_data.datasets]


class FSBackendTests:
    """Class for performing tests on Filesystem Backends,
    subclasses of FSBackend should subclass this class for testing."""

    @pytest.fixture(name="backend")
    def fixture_fs_backend(self, fs_backend):
        """Fixture for creating fs backend for use in other tests"""
        yield fs_backend

    @pytest.mark.asyncio
    async def test_save_load(self, backend: FSBackend, data: Data):
        """test basic saving and loading of data"""
        # use a different handle to avoid conflicts when creating multiple data
        data.handle = MeasurementHandle.new()
        path = await backend.save(data)
        assert str(path) == f"{backend.path_base/str(data.handle.handle)}"
        l_data = await backend.load(path)
        assert data.handle.handle == l_data.handle.handle
        assert len(data.datasets) != 0
        assert len(l_data.datasets) != 0
        assert len(data.datasets) == len(l_data.datasets)
        for dataset, l_dataset in zip(data.datasets, l_data.datasets):
            assert dataset.name == l_dataset.name
            assert dataset.memory.shape == l_dataset.memory.shape
            assert dataset.memory.dtype == l_dataset.memory.dtype
            assert np.all(dataset.memory.to_array() == l_dataset.memory.to_array())
            SharedMemoryOut.close(l_dataset.memory.name)

    @pytest.mark.asyncio
    async def test_save_partial_load(self, backend: FSBackend, data: Data):
        """test saving and partial loading"""
        data.handle = MeasurementHandle.new()
        path = await backend.save(data)
        assert str(path) == f"{backend.path_base/str(data.handle.handle)}"
        slice_lists = [
            [slice(3, 7, 2)],
            [slice(0, 2), slice(0, 2)],
            [slice(0, 2), slice(0, 3), slice(0, 3)],
            [slice(0, 2), slice(0, 2), slice(0, 2), slice(0, 10)],
        ]
        l_data = await backend.load(path, slice_lists=slice_lists)
        assert data.handle.handle == l_data.handle.handle
        assert len(data.datasets) != 0
        assert len(l_data.datasets) != 0
        assert len(data.datasets) == len(l_data.datasets)
        for dataset, l_dataset, sliced in zip(
            data.datasets, l_data.datasets, slice_lists
        ):
            assert dataset.name == l_dataset.name
            assert dataset.memory.dtype == l_dataset.memory.dtype
            sliced_data = dataset.memory.to_array()[tuple(sliced)]
            assert np.all(sliced_data == l_dataset.memory.to_array())
            SharedMemoryOut.close(l_dataset.memory.name)

    @pytest.mark.asyncio
    async def test_loading_a_non_existent_file(self, backend: FSBackend, data: Data):
        """test loading a file that does not exist raises a FileNotFoundError"""
        # use a different handle to avoid conflicts when creating multiple data
        data.handle = MeasurementHandle.new()
        path = backend.path(data)
        with pytest.raises(FileNotFoundError):
            await backend.load(path)

    @pytest.mark.asyncio
    async def test_loading_a_non_existent_file_partial(
        self, backend: FSBackend, data: Data
    ):
        """test loading a file that does not exist raises a FileNotFoundError"""
        # use a different handle to avoid conflicts when creating multiple data
        data.handle = MeasurementHandle.new()
        path = backend.path(data)
        with pytest.raises(FileNotFoundError):
            await backend.load(path, slice_lists=[[slice(0, 1)]])

    @pytest.mark.asyncio
    async def test_save_load_files_from_data(self, backend: FSBackend, data: Data):
        """test saving and loading files specified in data"""
        pbase = Path(__file__).parent.resolve() / "test_fs_backend_files"
        data.files = [
            pbase / "in/test_dir/test.txt",
            pbase / "in/test_dir2/test-2.txt",
            pbase / "in/test_dir2/test-3.txt",
        ]
        save_path = await backend.save(data)
        assert save_path == backend.path_base / str(data.handle.handle)
        assert backend.filesystem.exists(save_path.as_posix())
        load_path = pbase / "files/"
        l_data = await backend.load(save_path, load_path=load_path)
        test_files = [
            load_path / "test.txt",
            load_path / "test-2.txt",
            load_path / "test-3.txt",
        ]
        assert len(l_data.files) == len(test_files)
        for test_file, file in zip(test_files, data.files):
            assert test_file.exists()
            assert filecmp.cmp(str(test_file), str(file), shallow=False)
            test_file.unlink()
        load_path.rmdir()

    @pytest.mark.asyncio
    async def test_save_load_single_file(self, backend: FSBackend):
        """test saving and loading one file at a time"""
        testpath = Path(__file__).parent.resolve() / "test_fs_backend_files/"
        tpath_save = await backend.save_file(
            testpath / "in/test_dir/test.txt", backend.path_base
        )
        assert backend.filesystem.exists(str(tpath_save.as_posix()))
        tpath_load = await backend.load_file(tpath_save, testpath / "out_file/")
        assert (tpath_load).exists()
        assert filecmp.cmp(
            str(testpath / "in/test_dir/test.txt"), str(tpath_load), shallow=False
        )
        tpath_load.unlink()
        (testpath / "out_file/").rmdir()

    @pytest.mark.asyncio
    async def test_save_load_nonexistent_path(self, backend: FSBackend):
        """test saving and loading files with nonexistent file paths
        to raise FileNotFound error"""
        testpath = Path(__file__).parent.resolve()
        with pytest.raises(FileNotFoundError):
            await backend.save_file(testpath / "in/errtest.txt", backend.path_base)
        with pytest.raises(FileNotFoundError):
            await backend.load_file(
                backend.path_base / "errtest.txt", testpath / "out_file/"
            )

    @pytest.mark.asyncio
    async def test_init_data(self, backend: FSBackend, data: Data):
        """test initialising data in the FileSystem"""
        # use a different handle to avoid conflicts when creating multiple data
        data.handle = MeasurementHandle.new()
        path = await backend.init_data(data)
        assert str(path) == f"{backend.path_base/str(data.handle.handle)}"
        l_data = await backend.load(path)
        assert len(data.datasets) != 0
        assert len(l_data.datasets) == 4
        await backend.close(data)

    @pytest.mark.asyncio
    async def test_close_data(self, backend: FSBackend, data: Data):
        """Test closing data"""
        for dataset in data.datasets:
            assert dataset.memory.name in __memory_reference__
        await backend.close(data)
        for dataset in data.datasets:
            assert dataset.memory.name not in __memory_reference__

    @pytest.mark.asyncio
    async def test_update_data(self, backend: FSBackend, data: Data):
        """test updating data in the FileSystem"""
        data.handle = MeasurementHandle.new()
        path = await backend.init_data(data)
        assert str(path) == f"{backend.path_base/str(data.handle.handle)}"

        await backend.update(data)
        l_data = await backend.load(path)
        for dataset, l_dataset in zip(data.datasets, l_data.datasets):
            assert dataset.name == l_dataset.name
            assert dataset.memory.shape == l_dataset.memory.shape
            assert dataset.memory.dtype == l_dataset.memory.dtype
            assert np.all(dataset.memory.to_array() == l_dataset.memory.to_array())

        await backend.close(data)
        await backend.close(l_data)

    @pytest.mark.asyncio
    async def test_update_data_not_init_error(self, backend: FSBackend, data: Data):
        """test updating data in the FileSystem"""
        data.handle = MeasurementHandle.new()
        with pytest.raises(DataNotInitializedError):
            await backend.update(data)

    @pytest.mark.asyncio
    async def test_update_data_partial(self, backend: FSBackend, data: Data):
        data.handle = MeasurementHandle.new()
        path = await backend.init_data(data)
        await backend.update(data)

        slice_lists = [
            [slice(0, 10)],
            [slice(0, 2), slice(0, 2)],
            [slice(0, 2), slice(0, 3), slice(0, 3)],
            [slice(0, 2), slice(0, 2), slice(0, 2), slice(0, 2)],
        ]
        update_sizes = [(10,), (2, 2), (2, 3, 3), (2, 2, 2, 2)]

        rng = np.random.default_rng()
        for dataset, update_size, slice_list in zip(
            data.datasets, update_sizes, slice_lists
        ):
            mem_out = dataset.memory.to_array()[tuple(slice_list)]
            mem_out[:] = rng.standard_normal(size=update_size).astype(mem_out.dtype)
            if len(slice_list) == 4:
                mem_out[:] = False

        l_data = await backend.load(path)
        for dataset, l_dataset in zip(data.datasets, l_data.datasets):
            assert np.any(dataset.memory.to_array() != l_dataset.memory.to_array())

        await backend.update(data, slice_lists=slice_lists)
        l_data = await backend.load(path)
        for dataset, l_dataset in zip(data.datasets, l_data.datasets):
            assert dataset.name == l_dataset.name
            assert dataset.memory.shape == l_dataset.memory.shape
            assert dataset.memory.dtype == l_dataset.memory.dtype
            assert np.all(dataset.memory.to_array() == l_dataset.memory.to_array())

        await backend.close(data)
        await backend.close(l_data)

    @pytest.mark.asyncio
    async def test_update_data_resize(
        self, backend: FSBackend, data: Data, slice_lists: list[list[slice]]
    ):
        data.handle = MeasurementHandle.new()
        path = await backend.init_data(data)
        await backend.update(data)
        update_sizes = [(100,), (10, 10), (3, 3, 3), (2, 2, 2, 10)]
        updated_sizes = [(200,), (20, 10), (6, 3, 3), (4, 2, 2, 10)]

        rng = np.random.default_rng()
        for dataset, update_size, slice_list in zip(
            data.datasets, update_sizes, slice_lists
        ):
            mem_out = dataset.memory.to_array()
            mem_out[:] = rng.standard_normal(size=update_size).astype(mem_out.dtype)
            if len(slice_list) == 4:
                mem_out[:] = False

        l_data = await backend.load(path)
        for dataset, l_dataset in zip(data.datasets, l_data.datasets):
            assert np.any(dataset.memory.to_array() != l_dataset.memory.to_array())

        await backend.update(data, slice_lists=slice_lists)
        l_data2 = await backend.load(path)
        for dataset, l_dataset, u_shape, slice_list in zip(
            data.datasets, l_data2.datasets, updated_sizes, slice_lists
        ):
            assert dataset.name == l_dataset.name
            assert u_shape == l_dataset.memory.shape
            assert dataset.memory.dtype == l_dataset.memory.dtype
            assert np.all(
                dataset.memory.to_array()
                == l_dataset.memory.to_array()[tuple(slice_list)]
            )

        await backend.close(data)


class TestFSBackendWithStreaming(FSBackendTests):
    """Streaming tests for data backend"""

    @pytest.mark.asyncio
    async def test_prepare_finalize_stream(
        self, backend: FSBackend, streamed_data: Data
    ):
        """test preparing and finalizing a stream for FS backend and that measurment
        handles are tracked"""
        await backend.prepare_stream(streamed_data)
        assert streamed_data.handle.handle in backend.handles
        await backend.finalize_stream(streamed_data.handle)
        assert streamed_data.handle.handle not in backend.handles

    @pytest.mark.asyncio
    async def test_stream_data_1d(
        self,
        backend: FSBackend,
        streamed_data: Data,
        buffer_references: list[BufferReference],
    ):
        """Test preparing a stream, streaming data to the
        backend and finalizing the stream"""
        async_lock = asyncio.Lock()
        async for save_buffer in self.streaming_test_generator(
            backend, streamed_data, buffer_references, (10,)
        ):
            async with async_lock:
                await save_buffer

    @pytest.mark.asyncio
    async def test_stream_data_1d_random(
        self,
        backend: FSBackend,
        streamed_data: Data,
        buffer_references: list[BufferReference],
    ):
        """Test preparing a stream, streaming data to the
        backend, in a random order, and finalizing the stream"""
        async_lock = asyncio.Lock()
        async for save_buffer in self.streaming_test_generator(
            backend, streamed_data, buffer_references, (10,), random=True
        ):
            async with async_lock:
                await save_buffer

    @pytest.mark.asyncio
    async def test_stream_data_1d_gather(
        self,
        backend: FSBackend,
        streamed_data: Data,
        buffer_references: list[BufferReference],
    ):
        """Test preparing a stream, streaming data to the
        backend asynchronously and finalizing the stream"""
        async for save_buffer in self.streaming_test_generator(
            backend, streamed_data, buffer_references, (10,)
        ):
            await save_buffer

    @pytest.mark.asyncio
    async def test_stream_data_1d_gather_random(
        self,
        backend: FSBackend,
        streamed_data: Data,
        buffer_references: list[BufferReference],
    ):
        """Test preparing a stream, streaming data to the
        backend asynchronously, in random order, and finalizing the stream"""
        async for save_buffer in self.streaming_test_generator(
            backend, streamed_data, buffer_references, (10,), random=True
        ):
            await save_buffer

    @pytest.mark.asyncio
    async def test_streaming_2d(
        self,
        backend: FSBackend,
        streamed_data_2d: Data,
        buffer_references_2d: list[BufferReference],
    ):
        """Test preaparing stream saving buffers and finalizing stream"""
        async_lock = asyncio.Lock()
        async for save_buffer in self.streaming_test_generator(
            backend, streamed_data_2d, buffer_references_2d, (2, 3)
        ):
            async with async_lock:
                await save_buffer

    @pytest.mark.asyncio
    async def test_not_prepare_same_stream_multiple_times(
        self, backend: FSBackend, streamed_data: Data
    ):
        """Preparing a stream with a same handle as an already prepared stream should
        raise a StreamAlreadyPreparedError"""
        await backend.prepare_stream(streamed_data)
        with pytest.raises(StreamAlreadyPreparedError):
            await backend.prepare_stream(streamed_data)

    @pytest.mark.asyncio
    async def test_not_finalize_stream_more_than_once(
        self, backend: FSBackend, streamed_data: Data
    ):
        """Test that finalizing a stream that has already been finalised
        or not prepared raises a KeyError"""
        # finalising never prepared stream raises StreamNotPreparedError
        with pytest.raises(StreamNotPreparedError):
            await backend.finalize_stream(MeasurementHandle.new())

        # finalising stream twice raises KeyError the 2nd time
        await backend.prepare_stream(streamed_data)
        await backend.finalize_stream(streamed_data.handle)
        with pytest.raises(StreamNotPreparedError):
            await backend.finalize_stream(streamed_data.handle)

    @pytest.mark.asyncio
    async def test_saving_buffer_require_preparation(
        self,
        backend: FSBackend,
        streamed_data: Data,
        buffer_references: list[BufferReference],
    ):
        """test that saving a buffer using a un prepared measurment handle raises
        a StreamNotPreparedError"""
        await backend.prepare_stream(streamed_data)
        with pytest.raises(StreamNotPreparedError):
            await backend.save_buffer(
                MeasurementHandle.new(), "dummy_name", buffer_references[0], (0,)
            )

    @pytest.mark.asyncio
    async def test_saving_buffer_require_prepared_name(
        self,
        backend: FSBackend,
        streamed_data: Data,
        buffer_references: list[BufferReference],
    ):
        """test that saving a buffer using a name not part of the streaming setup raises
        a NameError"""
        await backend.prepare_stream(streamed_data)
        with pytest.raises(NameError):
            await backend.save_buffer(
                streamed_data.handle, "not_in_setup", buffer_references[0], (0,)
            )

    @pytest.mark.asyncio
    async def test_saving_buffer_require_prepared_buffer_size(
        self, backend: FSBackend, streamed_data: Data
    ):
        """test that saving a buffer using a shape different than the prepared shape
        (used as chunk size) raises a BufferShapeError"""
        await backend.prepare_stream(streamed_data)
        valid_name = streamed_data.datasets[0].name
        invalid_buffer = BufferReference(dtype="float", shape=(999,), name="dummy_name")
        with pytest.raises(BufferShapeError):
            await backend.save_buffer(
                streamed_data.handle, valid_name, invalid_buffer, (0,)
            )
        SharedMemoryOut.close(invalid_buffer.name)

    @pytest.mark.asyncio
    async def test_saving_buffer_require_prepared_datatype(
        self, backend: FSBackend, streamed_data: Data
    ):
        """test that saving a buffer using a data type different than the prepared
        data type raises a type error"""
        await backend.prepare_stream(streamed_data)
        valid_name = streamed_data.datasets[0].name
        invalid_buffer = BufferReference(
            dtype="uint",
            shape=streamed_data.datasets[0].memory.shape,
            name="dummy_name",
        )
        with pytest.raises(TypeError):
            await backend.save_buffer(
                streamed_data.handle, valid_name, invalid_buffer, (0,)
            )
        SharedMemoryOut.close(invalid_buffer.name)

    @pytest.mark.asyncio
    async def test_saving_buffer_require_matching_index_shape(
        self, backend: FSBackend, streamed_data: Data
    ):
        """test that saving a buffer using a index with different dimensions
        than the prepared dataset raises a value error"""
        await backend.prepare_stream(streamed_data)
        valid_name = streamed_data.datasets[0].name
        valid_buffer = BufferReference(
            dtype="float",
            shape=streamed_data.datasets[0].memory.shape,
            name="dummy_name",
        )
        with pytest.raises(ValueError):
            await backend.save_buffer(
                streamed_data.handle,
                valid_name,
                valid_buffer,
                (0, 0, 0),
            )
        SharedMemoryOut.close(valid_buffer.name)

    async def streaming_test_generator(
        self,
        backend: FSBackend,
        streamed_data: Data,
        buffer_references: list[BufferReference],
        size: tuple,
        random: bool = False,
    ):
        """Generate buffer lists for streaming tests

        Args:
            frontend (FrontendV1): frontend to use for the testing
            streamed_data_file (DataFile): datafile to use for the testing
            buffer_references (list[BufferReference]): buffer references to use
            size: (tuple):
                the size of the test to iterate over.
                e.g.
                    size = (10,) would do 10 buffers in 1d,
                    size = (2,3) would do 6 buffers in 2d

        Yields:
            _type_: coroutines to save buffers with
        """
        path = await backend.prepare_stream(streamed_data)
        l_data = await backend.load(path)
        assert_datasets_match(streamed_data.datasets, l_data.datasets)

        assert len(l_data.datasets) == len(buffer_references)
        [SharedMemoryOut.close(dset.memory.name) for dset in l_data.datasets]
        buffer_data = buffer_upload_data(buffer_references, size)
        arr = np.arange(np.prod(size)).reshape(size)
        if random:
            np.random.shuffle(arr)
        arr_iter = np.nditer(arr, flags=["multi_index"])
        for val in arr_iter:
            idx = np.where(arr == val)
            for buffer, dataset, b_data in zip(
                buffer_references, streamed_data.datasets, buffer_data
            ):
                yield save_buffer_test(
                    buffer,
                    dataset,
                    streamed_data.handle,
                    b_data,
                    tuple(i[0] for i in idx),
                    backend,
                    path,
                    serial=False,
                )

        l_data = await backend.load(path)
        assert_datasets_match(streamed_data.datasets, l_data.datasets, buffer_data)

        await backend.finalize_stream(streamed_data.handle)
        [SharedMemoryOut.close(dset.memory.name) for dset in l_data.datasets]


class TestFSBackendFromConfig(FSBackendTests):
    """Test FSBackend created from config"""

    @pytest_asyncio.fixture(name="backend")
    async def fixture_fs_backend_config(self, fs_config: FSBackendConfig):
        """Fixture for creating fs backend for use in other tests"""
        fsb = fs_config.init()
        yield fsb
        try:
            fsb.filesystem.rm(str(fsb.path_base), recursive=True)
        except FileNotFoundError:
            pass


def test_extract_shape(fs_backend: FSBackend):
    slice_list = [slice(0, 10, 2), slice(0, 7), slice(10), slice(None)]
    defaults = (5, 10, 10, 20)

    e_shape = fs_backend._extract_shape(slice_list, defaults)
    assert e_shape == (5, 7, 10, 20)
