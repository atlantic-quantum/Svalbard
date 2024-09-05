"""Test data server frontend v1 both with:
    a. Mongo DB metadata + fsspec FileSystem data backends
    b. Mongo DB metadata + GCS data backends
"""

import filecmp
import time
from pathlib import Path

import numpy as np
import pytest
from fsspec.asyn import get_loop
from fsspec.implementations.memory import MemoryFileSystem
from gcsfs import GCSFileSystem
from gcsfs.retry import HttpError

from svalbard.data_model.data_file import Data, DataFile, MeasurementHandle, MetaData
from svalbard.data_model.ipc import BufferReference
from svalbard.data_model.memory_models import SharedMemoryIn, SharedMemoryOut
from svalbard.data_server.data_backend.fs_backend import FSBackend
from svalbard.data_server.errors import (
    StreamAlreadyPreparedError,
    StreamNotPreparedError,
)
from svalbard.data_server.frontend.sync_frontend_v1 import (
    SyncFrontendV1,
    SyncFrontendV1Config,
)
from svalbard.data_server.metadata_backend.mongodb_backend import MongoDBBackend

from ..utility import (
    assert_datasets_match,
    assert_metadata_and_handle_match,
    assert_partial_datasets_match,
    buffer_upload_data,
    sync_save_buffer_test,
)

# todo benchmarking


def test_end_to_end_save_load(server_address):
    """End to end test to save and load data, that implements everything here"""
    # create data file
    # create metadata
    mdata = MetaData(name="test_data")
    smi = SharedMemoryIn(dtype="float", shape=(10, 10))
    smo = SharedMemoryOut.from_memory_in(smi)
    smo.to_array()[:] = np.arange(100).reshape((10, 10))
    datasets = [Data.DataSet(name=f"test_name_{i}", memory=smo) for i in range(4)]
    # create data
    data = Data(handle=MeasurementHandle.new(), datasets=datasets)
    # finally create the datafile
    datafile = DataFile(data=data, metadata=mdata)

    # create the frontend
    # create metadata backend,
    # use server address from fixture as it is location dependant
    mdb_backend = MongoDBBackend(
        f"mongodb://root:example@{server_address}:27017",
        "aq_test",
        "test",
    )
    # create the backend
    mfs = MemoryFileSystem()
    path_base = Path("memory:/tmp/test/fs_backend")
    # event loop is a asyncio loop createdd for tests marked asyncio

    data_backend = FSBackend(mfs, loop=get_loop(), path_base=path_base)
    # finally create the frontend
    frontend = SyncFrontendV1(data_backend=data_backend, metadata_backend=mdb_backend)

    # save the datafile
    path, save_task = frontend.save(datafile)
    assert isinstance(path, Path)

    # load the data and make sure loaded data matches saved data
    l_data_file = frontend.load(path)
    assert l_data_file.metadata == datafile.metadata
    assert datafile.data is not None
    assert l_data_file.data is not None
    assert l_data_file.data.handle == datafile.data.handle
    assert len(datasets) == len(l_data_file.data.datasets)
    for dataset, l_dataset in zip(datasets, l_data_file.data.datasets):
        assert dataset.name == l_dataset.name
        assert dataset.memory.dtype == l_dataset.memory.dtype
        assert dataset.memory.shape == l_dataset.memory.shape
        assert (
            dataset.memory.name != l_dataset.memory.name
        )  # data and loaded data should not be in the same shared memeory
        assert np.all(dataset.memory.to_array() == l_dataset.memory.to_array())
    # cleanup to have the test not leave files, should not be done in production
    try:
        mfs.rm(str(path_base), recursive=True)
    except FileNotFoundError:
        pass


def test_end_to_end_streaming(server_address):
    """End to end streaming test
    that contains all the steps instead of using fixture"""
    handle = MeasurementHandle.new()
    # create data file
    # create metadata
    mdata = MetaData(name="test_data")
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
    data = Data(handle=handle, datasets=datasets)
    # finally create the datafile
    datafile = DataFile(data=data, metadata=mdata)

    # create array with data to stream from
    data_sizes: list[tuple[int, ...]] = [(10, 10), (10, 5, 2)]
    streamed_data = [np.arange(np.prod(shape)).reshape(shape) for shape in data_sizes]

    # create the frontend
    # create metadata backend,
    # use server address from fixture as it is location dependant
    mdb_backend = MongoDBBackend(
        f"mongodb://root:example@{server_address}:27017",
        "aq_test",
        "test",
    )
    # create the backend
    mfs = MemoryFileSystem()
    path_base = Path("memory:/tmp/test/fs_backend")
    # event loop is a asyncio loop createdd for tests marked asyncio
    data_backend = FSBackend(mfs, loop=get_loop(), path_base=path_base)
    # finally create the frontend
    frontend = SyncFrontendV1(data_backend=data_backend, metadata_backend=mdb_backend)

    # prepare for stream
    path = frontend.prepare_stream(handle, datafile)

    # stream data
    for j, buffer in enumerate(buffer_references):
        for i in range(10):
            buffer.to_array()[:] = streamed_data[j][i]
            frontend.save_buffer(
                data.handle,
                datasets[j].name,
                buffer,
                (i,) + tuple(0 for _ in buffer.shape[1:]),
            )

    # finalize stream
    frontend.finalize_stream(data.handle)

    # load data and compare to uploaded.
    l_data_file = frontend.load(path)
    assert l_data_file.data is not None
    for dataset, l_dataset, streamed in zip(
        data.datasets, l_data_file.data.datasets, streamed_data
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


@pytest.mark.asyncio
async def test_init_frontend_fs_from_config(
    sync_frontend_fs_config: SyncFrontendV1Config,
):
    """Test creating a sync frontend with mongo db metadata
    backend and fsspec filesytem based data backend"""
    sync_frontend_fs_config.init()


# def test_init_frontend_gcs_from_config(
#     sync_frontend_gcs_config: SyncFrontendV1Config, gcs_filesystem: GCSFileSystem
# ):
#     """Test creating a frontend with mongo db metadata
#     backend and fsspec filesytem based data backend"""
#     sync_frontend_gcs_config.init()


class SyncFrontendV1Tests:
    """Class for testing data server frontend v1 and it's subclasses"""

    @pytest.fixture(name="frontend", scope="class")
    def frontend_v1_fs(self, sync_frontend_mdb_fs: SyncFrontendV1):
        """Fixture that yields a frontend,
        used for subclassing multiple Test SyncFrontendV1 classes"""
        yield sync_frontend_mdb_fs

    def test_save_load(self, frontend: SyncFrontendV1, data_file: DataFile):
        """Test Saving and Loading DataFile"""
        path, save_task = frontend.save(data_file)
        assert isinstance(path, Path)
        while list(frontend.background_tasks):  # type: ignore
            time.sleep(0.01)
        l_data_file = frontend.load(path)
        # assert l_data_file.metadata.data_path != None
        # l_data_file.metadata.data_path = None
        assert l_data_file.metadata == data_file.metadata
        assert data_file.data is not None
        assert l_data_file.data is not None
        assert l_data_file.data.handle == data_file.data.handle
        assert_datasets_match(l_data_file.data.datasets, data_file.data.datasets)

    def test_save_partial_load(self, frontend: SyncFrontendV1, data_file: DataFile):
        """Test Saving and Partially Loading DataFile"""
        path, save_task = frontend.save(data_file)
        assert isinstance(path, Path)
        while list(frontend.background_tasks):  # type: ignore
            time.sleep(0.01)
        slice_lists = [
            [slice(3, 7, 2)],
            [slice(0, 2), slice(0, 2)],
            [slice(0, 2), slice(0, 3), slice(0, 3)],
            [slice(0, 2), slice(0, 2), slice(0, 2), slice(0, 10)],
        ]
        l_data_file = frontend.load(path, slice_lists=slice_lists)
        assert l_data_file.data is not None
        assert data_file.data is not None
        assert l_data_file.data.handle == data_file.data.handle
        assert_partial_datasets_match(
            data_file.data.datasets, l_data_file.data.datasets, slice_lists
        )

    def test_save_load_files(self, frontend: SyncFrontendV1, data_file: DataFile):
        """Test saving and loading files from DataFile Data"""
        in_files_path_base = (
            Path(__file__).parent.parent.resolve()
            / "data_backend/test_fs_backend_files/in"
        )
        data_file.data.files = [
            in_files_path_base / "test_dir2/test-2.txt",
            in_files_path_base / "test_dir2/test-3.txt",
            in_files_path_base / "test_dir/test.txt",
        ]
        assert not data_file.metadata.files
        path, save_task = frontend.save(data_file)
        # time.sleep(10)
        assert isinstance(path, Path)
        assert data_file.metadata.files
        assert len(data_file.data.files) == len(data_file.metadata.files)
        assert data_file.metadata.files == [
            Path("test-2.txt"),
            Path("test-3.txt"),
            Path("test.txt"),
        ]
        total_time = 0
        while list(frontend.background_tasks) and total_time < 5:  # type: ignore
            total_time += 0.01
            time.sleep(0.01)
        load_files_path_base = (
            Path(__file__).parent.resolve() / "test_frontend_v1_files/"
        )
        # try:
        l_data_file = frontend.load(path, True, load_files_path_base)
        # except FileNotFoundError:
        #     time.sleep(100)
        assert len(data_file.metadata.files) == len(l_data_file.data.files)
        for file, md_file, l_file in zip(
            data_file.data.files, data_file.metadata.files, l_data_file.data.files
        ):
            assert md_file == Path(l_file.name)
            assert l_file.exists()
            assert filecmp.cmp(str(file), str(l_file), shallow=False)
            l_file.unlink()
        load_files_path_base.rmdir()

    def test_init(self, frontend: SyncFrontendV1, data_file: DataFile):
        """Test initialising DataFile"""
        path = frontend.init_data(data_file)
        assert isinstance(path, Path)

    def test_close(self, frontend: SyncFrontendV1, data_file: DataFile):
        """Test closing data in DataFile"""
        assert data_file.data
        for dataset in data_file.data.datasets:
            dataset.memory.to_array()
        frontend.close(data_file)
        for dataset in data_file.data.datasets:
            with pytest.raises(FileNotFoundError):
                dataset.memory.to_array()

    def test_save_load_only_metadata(
        self, frontend: SyncFrontendV1, data_file: DataFile
    ):
        """Test saving and loading and only metadata"""
        data_file.data = None
        assert data_file.metadata.data_path is None
        path, save_task = frontend.save(data_file)
        assert isinstance(path, Path)
        l_data_file = frontend.load(path)
        assert l_data_file.data is None
        assert l_data_file.metadata == data_file.metadata

    def test_loading_non_existent_file(self, frontend: SyncFrontendV1):
        """test that loading a file that does not exist raises a FileNotFoundError"""
        # test saving and loading
        with pytest.raises(FileNotFoundError):
            frontend.load(Path("/this/does/not/exist"))

    def test_update(
        self,
        frontend: SyncFrontendV1,
        data_file: DataFile,
        slice_lists: list[list[slice]],
    ):
        assert data_file.data is not None
        path = frontend.init_data(data_file)
        data_file.metadata.data_path = frontend.data_backend.path(data_file.data)
        frontend.update(path, data_file)
        frontend.complete_background_tasks()

        update_sizes = [(100,), (10, 10), (3, 3, 3), (2, 2, 2, 10)]
        updated_sizes = [(200,), (20, 10), (6, 3, 3), (4, 2, 2, 10)]

        rng = np.random.default_rng()
        for dataset, update_size, slice_list in zip(
            data_file.data.datasets, update_sizes, slice_lists
        ):
            mem_out = dataset.memory.to_array()
            mem_out[:] = rng.standard_normal(size=update_size).astype(mem_out.dtype)
            if len(slice_list) == 4:
                mem_out[:] = False

        l_data = frontend.load(path)
        assert l_data.data is not None
        for dataset, l_dataset in zip(data_file.data.datasets, l_data.data.datasets):
            assert np.any(dataset.memory.to_array() != l_dataset.memory.to_array())

        frontend.update(path, data_file, slice_lists=slice_lists)
        frontend.complete_background_tasks()
        l_data2 = frontend.load(path)
        assert l_data2.data is not None
        for dataset, l_dataset, u_shape, slice_list in zip(
            data_file.data.datasets, l_data2.data.datasets, updated_sizes, slice_lists
        ):
            assert dataset.name == l_dataset.name
            assert u_shape == l_dataset.memory.shape
            assert dataset.memory.dtype == l_dataset.memory.dtype
            assert np.all(
                dataset.memory.to_array()
                == l_dataset.memory.to_array()[tuple(slice_list)]
            )


class TestSyncFrontendV1WithStreaming(SyncFrontendV1Tests):
    """Streaming Tests for Frontend V1"""

    def test_streaming_1d(
        self,
        frontend: SyncFrontendV1,
        streamed_data_file: DataFile,
        buffer_references: list[BufferReference],
    ):
        """Test preaparing stream saving buffers and finalizing stream"""
        for save_buffer in self.streaming_test_generator(
            frontend, streamed_data_file, buffer_references, (10,)
        ):
            save_buffer

    def test_streaming_1d_random(
        self,
        frontend: SyncFrontendV1,
        streamed_data_file: DataFile,
        buffer_references: list[BufferReference],
    ):
        """Test preaparing stream saving buffers in a random order
        and finalizing stream"""
        for save_buffer in self.streaming_test_generator(
            frontend, streamed_data_file, buffer_references, (10,), random=True
        ):
            save_buffer

    def test_streaming_2d(
        self,
        frontend: SyncFrontendV1,
        streamed_data_file_2d: DataFile,
        buffer_references_2d: list[BufferReference],
    ):
        """Test preaparing stream saving buffers and finalizing stream, 2d loop"""
        for save_buffer in self.streaming_test_generator(
            frontend, streamed_data_file_2d, buffer_references_2d, (2, 3)
        ):
            save_buffer

    def test_streaming_no_prepare_twice(
        self, frontend: SyncFrontendV1, streamed_data_file: DataFile
    ):
        """Test preaparing stream with same handle twice raises ValueError"""
        assert streamed_data_file.data is not None
        handle = streamed_data_file.data.handle
        frontend.prepare_stream(handle, streamed_data_file)
        with pytest.raises(StreamAlreadyPreparedError):
            frontend.prepare_stream(handle, streamed_data_file)

    def test_prepare_streaming_requires_data(
        self, frontend: SyncFrontendV1, streamed_data_file: DataFile
    ):
        """Test that preparing a stream for a datafile with no data
        raises ValueError"""
        streamed_data_file.data = None
        with pytest.raises(ValueError):
            frontend.prepare_stream(MeasurementHandle.new(), streamed_data_file)

    def test_streaming_no_finalize_twice(
        self, frontend: SyncFrontendV1, streamed_data_file: DataFile
    ):
        """Test finalising a stream with same handle twice raises ValueError"""
        assert streamed_data_file.data is not None
        handle = streamed_data_file.data.handle
        frontend.prepare_stream(handle, streamed_data_file)
        frontend.finalize_stream(streamed_data_file.data.handle)
        with pytest.raises(StreamNotPreparedError):
            frontend.finalize_stream(streamed_data_file.data.handle)

    def streaming_test_generator(
        self,
        frontend: SyncFrontendV1,
        streamed_data_file: DataFile,
        buffer_references: list[BufferReference],
        size: tuple,
        random: bool = False,
    ):
        """Generate buffer lists for streaming tests

        Args:
            frontend ( SyncFrontendV1): frontend to use for the testing
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
        assert streamed_data_file.data is not None
        handle = streamed_data_file.data.handle
        path = frontend.prepare_stream(handle, streamed_data_file)
        l_data = frontend.load(path)
        assert_metadata_and_handle_match(streamed_data_file, l_data)
        assert_datasets_match(
            l_data.data.datasets,  # type: ignore
            streamed_data_file.data.datasets,
        )

        assert len(l_data.data.datasets) == len(buffer_references)  # type: ignore
        assert isinstance(l_data.data, Data)
        buffer_data = buffer_upload_data(buffer_references, size)
        arr = np.arange(np.prod(size)).reshape(size)
        if random:
            np.random.shuffle(arr)
        arr_iter = np.nditer(arr, flags=["multi_index"])
        for val in arr_iter:
            for buffer, dataset, b_data in zip(
                buffer_references, streamed_data_file.data.datasets, buffer_data
            ):
                idx = np.where(arr == val)
                idx = tuple(i[0] for i in idx)
                yield sync_save_buffer_test(
                    buffer,
                    dataset,
                    streamed_data_file.data.handle,
                    b_data,
                    idx,
                    frontend,
                    path,
                    serial=False,
                )

        l_data = frontend.load(path)
        assert_metadata_and_handle_match(streamed_data_file, l_data)
        assert_datasets_match(
            streamed_data_file.data.datasets,
            l_data.data.datasets,  # type: ignore
            buffer_data,
        )

        frontend.finalize_stream(streamed_data_file.data.handle)
        assert isinstance(l_data.data, Data)


class TestSyncFrontendV1GCS(TestSyncFrontendV1WithStreaming):
    """Class for testing data server frontend v1 with Mongo DB and GCS Backends"""

    @pytest.fixture(name="frontend", scope="class")
    def frontend_v1_gcs(self, sync_frontend_mdb_gcs: SyncFrontendV1):
        """Fixture that yields a frontend,
        with a Mongo DB metadata backend and a GCS data backend"""
        yield sync_frontend_mdb_gcs


class TestSyncFrontendV1Config(SyncFrontendV1Tests):
    """Class for testing data server frontend v1 with Mongo DB and FS Backends
    created from config"""

    def frontend_v1_fs_config(self, sync_frontend_fs_config: SyncFrontendV1Config):
        """Fixture that yields a frontend, created from config,
        with a Mongo DB metadata backend and a FS data backend"""
        fend = sync_frontend_fs_config.init()
        yield fend
        assert isinstance(fend.data_backend, FSBackend)
        try:
            fend.data_backend.filesystem.rm(
                str(fend.data_backend.path_base), recursive=True
            )
        except FileNotFoundError:
            pass


class TestSyncFrontendV1GCSConfig(SyncFrontendV1Tests):
    """Class for testing data server frontend v1 with Mongo DB and GCS Backends"""

    def frontend_v1_gcs_config(
        self,
        sync_frontend_gcs_config: SyncFrontendV1Config,
        gcs_filesystem: GCSFileSystem,
    ):
        """Fixture that yields a frontend, created from config,
        with a Mongo DB metadata backend and a GCS data backend"""
        test_bucket = "aq_test"
        try:
            # ensure we're empty.
            try:
                gcs_filesystem.rm(test_bucket, recursive=True)
            except FileNotFoundError:
                pass
            try:
                gcs_filesystem.mkdir(test_bucket)
            except HttpError:
                pass
            yield sync_frontend_gcs_config.init()
        finally:
            try:
                gcs_filesystem.rm(gcs_filesystem.find(test_bucket))
            except FileNotFoundError:
                pass
