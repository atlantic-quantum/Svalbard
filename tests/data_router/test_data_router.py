"""Test rest API of Data Server app"""
import asyncio
import filecmp
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient
from svalbard.data_model.data_file import Data, DataFile, MeasurementHandle, MetaData
from svalbard.data_model.ipc import (
    BufferReference,
    DataFileAndSliceModel,
    EndStreamModel,
    SaveBufferModel,
    SavedPath,
    SliceListModel,
    StartStreamModel,
)
from svalbard.data_model.memory_models import (
    SharedMemoryIn,
    SharedMemoryOut,
    __memory_reference__,
)
from svalbard.data_router.data_router import DataServer, ds_router
from svalbard.data_server.frontend.frontend_v1 import FrontendV1, FrontendV1Config
from svalbard.utility.resize_array import new_shape

from ..data_server.utility import (
    assert_datasets_match,
    assert_metadata_and_handle_match,
    buffer_upload_data,
    index,
    new_buffer_and_slices,
)


@pytest.mark.asyncio
async def test_uvicorn_server_started(startup_and_shutdown_server):
    """A simple websocket test"""
    # any test code here
    server = startup_and_shutdown_server
    assert server.started


@pytest.mark.asyncio
async def test_callback_app(
    startup_and_shutdown_server, callback_app, buffer_references
):
    """Test that calling buffer saved for the callback app
    adds the buffer to the list of saved buffers"""
    assert not callback_app.saved_buffers  # type: ignore
    assert startup_and_shutdown_server.started
    async with AsyncClient(base_url="http://127.0.0.1:8000") as capp:
        for buffer in buffer_references:
            res = await capp.post("/buffer/saved", content=buffer.json())
            assert res.status_code == 200
            assert buffer in callback_app.saved_buffers  # type: ignore


@pytest.mark.asyncio
async def test_data_server(frontend_mdb_fs: FrontendV1, frontend_mdb_gcs: FrontendV1):
    """Tests assigning and getting the frontend of the DataServer"""
    assert frontend_mdb_fs != frontend_mdb_gcs
    assert DataServer.frontend is None
    ds1 = DataServer()
    ds2 = DataServer()
    assert ds1.frontend is None
    assert ds2.frontend is None
    DataServer.set_frontend(frontend_mdb_fs)
    assert DataServer.frontend is not None
    assert ds1.frontend is not None
    assert ds2.frontend is not None
    # pylint: disable=comparison-with-callable
    assert DataServer.frontend == frontend_mdb_fs
    assert DataServer.frontend == ds1.frontend
    assert DataServer.frontend == ds2.frontend
    ds1.set_frontend(frontend_mdb_gcs)
    assert DataServer.frontend == frontend_mdb_gcs
    # pylint: enable=comparison-with-callable
    assert ds1.frontend == frontend_mdb_gcs
    assert ds2.frontend == frontend_mdb_gcs
    DataServer.set_frontend(None)  # type: ignore
    assert ds1.frontend is None
    assert ds2.frontend is None


@pytest.mark.asyncio
async def test_data_server_frontend_must_be_assigned(data_file: DataFile):
    """test that calling api without assigning the dataserver frontend
    results in 503"""
    assert DataServer.frontend is None
    app = FastAPI()
    app.include_router(ds_router, prefix="/data")
    async with AsyncClient(app=app, base_url="http://test") as ds_app:
        response = await ds_app.post("/data/save", content=data_file.json())
        assert response.status_code == 503
        assert response.json()["detail"] == DataServer.NOT_SET_UP


@pytest.mark.asyncio
async def test_end_to_end_save_load(server_address):
    """End to end save/load tests
    that contains all the steps instead of using fixture"""
    # setup data server from config
    config = {
        "data_backend": {
            "cls": "fsspec.implementations.memory.MemoryFileSystem",
            "protocol": "memory",
            "args": [],
            "path": "memory:/tmp/test/fs_backend_config",
        },
        "metadata_backend": {
            "url": f"mongodb://root:example@{server_address}:27017",
            "database": "aq_test",
            "collection": "test",
        },
    }
    frontend_config = FrontendV1Config(**config)  # type: ignore

    # create a datafile
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

    # create the rest api app that runs the data server
    app = FastAPI()
    app.include_router(ds_router, prefix="/data")
    client = AsyncClient(app=app, base_url="http://test")

    async with client:
        # setup the data server
        assert DataServer.frontend is None
        await client.post("/data/setup", content=frontend_config.json())
        assert DataServer.frontend is not None

        # save data
        s_response = await client.post("/data/save", content=datafile.json())
        assert s_response.status_code == 201
        path = SavedPath(**s_response.json()).path

        # load data
        l_response = await client.get(f"/data/load/{str(path)}")
        assert l_response.status_code == 200
        l_datafile = DataFile(**l_response.json())
        # make sure loaded data is the same as saved data
        # the data path is determined during the save step, is None before then
        assert datafile.metadata.data_path is None
        assert l_datafile.metadata.data_path is not None
        l_datafile.metadata.data_path = None
        # saved and loaded metadata the same,
        # after we put the data path of the loaded data to None
        assert l_datafile.metadata == datafile.metadata
        assert l_datafile.data is not None
        assert datafile.data is not None
        for l_dataset, dataset in zip(l_datafile.data.datasets, datafile.data.datasets):
            assert l_dataset.memory.shape == dataset.memory.shape
            assert l_dataset.memory.dtype == dataset.memory.dtype
            assert np.all(l_dataset.memory.to_array() == dataset.memory.to_array())
            SharedMemoryOut.close(l_dataset.memory.name)

    # reset the dataserver for subsequent tests
    DataServer.set_frontend(None)  # type: ignore
    SharedMemoryOut.close(smo.name)


@pytest.mark.asyncio
async def test_end_to_end_streaming(server_address):
    """End to end streaming test
    that contains all the steps instead of using fixture"""
    # setup data server from config
    config = {
        "data_backend": {
            "cls": "fsspec.implementations.memory.MemoryFileSystem",
            "protocol": "memory",
            "args": [],
            "path": "memory:/tmp/test/fs_backend_config",
        },
        "metadata_backend": {
            "url": f"mongodb://root:example@{server_address}:27017",
            "database": "aq_test",
            "collection": "test",
        },
    }
    frontend_config = FrontendV1Config(**config)

    handle = MeasurementHandle.new()

    # create data file
    mdata = MetaData(name="test_data")
    buffer_sizes = [(1, 10), (1, 5, 2)]
    smis = [
        SharedMemoryIn(dtype="float64", shape=buffer_size)
        for buffer_size in buffer_sizes
    ]
    buffer_references = [BufferReference.from_memory_in(smi) for smi in smis]
    datasets = [
        Data.DataSet(name=f"test_name_{i}", memory=memory)
        for i, memory in enumerate(buffer_references)
    ]
    data = Data(handle=handle, datasets=datasets)
    datafile = DataFile(data=data, metadata=mdata)

    # create start end stream models
    start_stream_model = StartStreamModel(handle=handle, file=datafile)
    end_stream_model = EndStreamModel(handle=handle)

    # create array with data to stream from
    data_sizes = [(10, 10), (10, 5, 2)]
    streamed_data = [np.arange(np.prod(shape)).reshape(shape) for shape in data_sizes]

    # create the rest api app that runs the data server
    app = FastAPI()
    app.include_router(ds_router, prefix="/data")
    client = AsyncClient(app=app, base_url="http://test")

    async with client:
        # setup the data server
        assert DataServer.frontend is None
        await client.post("/data/setup", content=frontend_config.json())
        assert DataServer.frontend is not None

        # prepare for stream
        res = await client.post("/data/stream/start", content=start_stream_model.json())
        assert res.status_code == 201
        path = SavedPath(**res.json()).path

        # stream data
        for j, buffer in enumerate(buffer_references):
            for i in range(10):
                buffer.to_array()[:] = streamed_data[j][i]
                sbm = SaveBufferModel(
                    handle=data.handle,
                    name=datasets[j].name,
                    buffer=buffer,
                    index=list((i,) + tuple(0 for _ in buffer.shape[1:])),
                )
                res = await client.post("/data/save/buffer", content=sbm.json())
                assert res.status_code == 202
                assert res.json() == {"message": "buffer posted"}

        # finalize stream
        await client.post("/data/stream/end", content=end_stream_model.json())

        # load data and compare to uploaded.
        l_response = await client.get(f"/data/load/{str(path)}")
        assert l_response.status_code == 200
        l_datafile = DataFile(**l_response.json())
        assert l_datafile.data is not None
        for dataset, l_dataset, streamed in zip(
            data.datasets, l_datafile.data.datasets, streamed_data
        ):
            assert dataset.name == l_dataset.name
            assert streamed.shape == l_dataset.memory.shape
            assert dataset.memory.dtype == l_dataset.memory.dtype
            assert np.all(streamed == l_dataset.memory.to_array())
            SharedMemoryOut.close(l_dataset.memory.name)
    [SharedMemoryOut.close(br.name) for br in buffer_references]


class DataRouterAppTests:
    """Class for testing data server app with frontend v1 and it's subclasses"""

    def create_app(self, frontend: FrontendV1) -> AsyncClient:
        """create an app that includeds
        a data server router (ds_router) and a frontend"""
        app = FastAPI()
        app.include_router(ds_router, prefix="/data")
        DataServer.set_frontend(frontend)  # type: ignore
        return AsyncClient(app=app, base_url="http://test")

    async def _load(
        self,
        ds_app: AsyncClient,
        path: Path,
        return_df: bool = False,
        load_path: Path | None = None,
    ) -> DataFile | Data:
        """Helper function for loading data,
        should be run within 'with ds_app:'
        """
        params = {"load_path": load_path}
        l_res = await ds_app.get(
            f"/data/load/{str(path)}",
            params=params,  # type: ignore
        )
        assert l_res.status_code == 200
        l_data_file = DataFile(**l_res.json())
        if return_df:
            return l_data_file
        assert l_data_file.data is not None
        return l_data_file.data

    async def _load_partial(
        self,
        ds_app: AsyncClient,
        path: Path,
        slm: SliceListModel,
        return_df: bool = False,
    ) -> DataFile | Data:
        """Helper function for loading data,
        should be run within 'with ds_app:'
        """
        l_res = await ds_app.post(f"/data/load_partial/{str(path)}", content=slm.json())
        assert l_res.status_code == 200
        data_file = DataFile(**l_res.json())
        if return_df:
            return data_file
        assert data_file.data is not None
        return data_file.data

    @pytest.fixture(name="ds_app")
    def fixture_ds_app_fs(self, frontend_mdb_fs: FrontendV1):
        """create an app that includeds a data server router (ds_router)
        and a frontend with mongo and fsspec filesystem"""
        yield self.create_app(frontend_mdb_fs)
        DataServer.set_frontend(None)  # type: ignore

    @pytest.mark.asyncio
    async def test_save_load(self, ds_app: AsyncClient, data_file: DataFile):
        """Basic saving and loading test using the data server app"""
        async with ds_app:
            response = await ds_app.post("/data/save", content=data_file.json())
            assert response.status_code == 201
            path = SavedPath(**response.json()).path
            assert isinstance(DataServer.frontend, FrontendV1)
            await DataServer.frontend.complete_background_tasks()
            l_data_file = await self._load(ds_app, path, return_df=True)
            assert isinstance(l_data_file, DataFile)
            assert data_file.metadata.data_path is None
            assert l_data_file.metadata.data_path is not None
            l_data_file.metadata.data_path = None
            assert l_data_file.metadata == data_file.metadata
            assert l_data_file.data is not None
            assert data_file.data is not None
            for l_dataset, dataset in zip(
                l_data_file.data.datasets, data_file.data.datasets
            ):
                assert l_dataset.memory.shape == dataset.memory.shape
                assert l_dataset.memory.dtype == dataset.memory.dtype
                assert np.all(l_dataset.memory.to_array() == dataset.memory.to_array())
                SharedMemoryOut.close(l_dataset.memory.name)

    @pytest.mark.asyncio
    async def test_save_load_files(self, ds_app: AsyncClient, data_file: DataFile):
        async with ds_app:
            in_path = (
                Path(__file__).parent.parent.resolve()
                / "data_server/data_backend/test_fs_backend_files/in"
            )
            assert data_file.data is not None
            data_file.data.files = [
                in_path / "test_dir2/test-2.txt",
                in_path / "test_dir2/test-3.txt",
                in_path / "test_dir/test.txt",
            ]
            response = await ds_app.post("/data/save", content=data_file.json())
            assert response.status_code == 201
            path = SavedPath(**response.json()).path
            assert isinstance(DataServer.frontend, FrontendV1)
            await DataServer.frontend.complete_background_tasks()
            load_path = Path(__file__).parent.resolve() / "test_data_router_files/"
            l_data_file = await self._load(
                ds_app, path, return_df=True, load_path=load_path
            )
            assert isinstance(l_data_file, DataFile)
            # files check
            assert l_data_file.data is not None
            assert len(data_file.data.files) == len(l_data_file.data.files)
            assert l_data_file.data.files == [
                load_path / "test-2.txt",
                load_path / "test-3.txt",
                load_path / "test.txt",
            ]
            for file, md_file, l_file in zip(
                data_file.data.files, l_data_file.metadata.files, l_data_file.data.files
            ):
                assert md_file == Path(l_file.name)
                assert l_file.exists()
                assert filecmp.cmp(str(file), str(l_file), shallow=False)
                l_file.unlink()
            load_path.rmdir()
            # metadata check
            assert l_data_file.metadata.files
            assert len(data_file.data.files) == len(l_data_file.metadata.files)
            assert l_data_file.metadata.files == [
                Path("test-2.txt"),
                Path("test-3.txt"),
                Path("test.txt"),
            ]
            l_data_file.metadata.data_path = None
            l_data_file.metadata.files = []
            assert l_data_file.metadata == data_file.metadata

    @pytest.mark.asyncio
    async def test_save_partial_load(self, ds_app: AsyncClient, data_file: DataFile):
        """Basic saving and partial loading test using the data server app"""
        slice_list = [
            [slice(3, 7, 2)],
            [slice(0, 2), slice(0, 2)],
            [slice(0, 2), slice(0, 3), slice(0, 3)],
            [slice(0, 2), slice(0, 2), slice(0, 2), slice(0, 10)],
        ]
        async with ds_app:
            response = await ds_app.post("/data/save", content=data_file.json())
            assert response.status_code == 201
            path = SavedPath(**response.json()).path
            assert isinstance(DataServer.frontend, FrontendV1)
            await DataServer.frontend.complete_background_tasks()
            l_data_file = await self._load_partial(
                ds_app,
                path,
                slm=SliceListModel.from_slice_lists(slice_list),
                return_df=True,
            )
            assert isinstance(l_data_file, DataFile)
            assert data_file.metadata.data_path is None
            assert l_data_file.metadata.data_path is not None
            l_data_file.metadata.data_path = None
            assert l_data_file.metadata == data_file.metadata
            assert l_data_file.data is not None
            assert data_file.data is not None
            for l_dataset, dataset, sliced in zip(
                l_data_file.data.datasets, data_file.data.datasets, slice_list
            ):
                assert l_dataset.memory.dtype == dataset.memory.dtype
                sliced_data = dataset.memory.to_array()[tuple(sliced)]
                assert np.all(sliced_data == l_dataset.memory.to_array())
                SharedMemoryOut.close(l_dataset.memory.name)

    @pytest.mark.asyncio
    async def test_load_non_existent_file(
        self, ds_app: AsyncClient, data_file: DataFile
    ):
        """Test that trying to load a file that does not exist raises a ???"""
        async with ds_app:
            l_res = await ds_app.get("/data/load/this/file/does/not/exist")
            assert l_res.status_code == 404

    @pytest.mark.asyncio
    async def test_load_partial_non_existent_file(
        self, ds_app: AsyncClient, data_file: DataFile
    ):
        slice_lists = [[slice(3, 7, 2)]]
        async with ds_app:
            l_res = await ds_app.post(
                "/data/load_partial/this/file/does/not/exist",
                content=SliceListModel.from_slice_lists(slice_lists).json(),
            )
            assert l_res.status_code == 404

    @pytest.mark.asyncio
    async def test_load_only_metadata(self, ds_app: AsyncClient, data_file: DataFile):
        """Test that loading only metadata does not load data"""
        async with ds_app:
            response = await ds_app.post("/data/save", content=data_file.json())
            assert response.status_code == 201
            path = SavedPath(**response.json()).path
            assert isinstance(DataServer.frontend, FrontendV1)
            await DataServer.frontend.complete_background_tasks()

            l_res = await ds_app.get(
                f"/data/load/{str(path)}", params={"load_data": False}
            )
            assert l_res.status_code == 200
            l_data_file = DataFile(**l_res.json())
            assert l_data_file.metadata.data_path is not None
            l_data_file.metadata.data_path = None
            assert l_data_file.metadata == data_file.metadata
            assert l_data_file.data is None
            assert data_file.data is not None

    @pytest.mark.asyncio
    async def test_init(self, ds_app: AsyncClient, data_file: DataFile):
        """Test initializing a data store"""
        async with ds_app:
            response = await ds_app.post("/data/init", content=data_file.json())
            assert response.status_code == 201
            assert isinstance(DataServer.frontend, FrontendV1)
            await DataServer.frontend.complete_background_tasks()

    @pytest.mark.asyncio
    async def test_update_metadata(self, ds_app: AsyncClient, data_file: DataFile):
        """Test updating metadata"""
        async with ds_app:
            response = await ds_app.post("/data/init", content=data_file.json())
            assert response.status_code == 201
            assert isinstance(DataServer.frontend, FrontendV1)
            assert data_file.data is not None
            data_file.metadata.data_path = DataServer.frontend.data_backend.path(
                data_file.data
            )
            path = SavedPath(**response.json()).path
            response = await ds_app.post(
                f"/data/update/metadata/{str(path)}", content=data_file.json()
            )
            assert response.status_code == 200
            await DataServer.frontend.complete_background_tasks()
            l_res = await ds_app.get(
                f"/data/load/{str(path)}", params={"load_data": False}
            )
            assert l_res.status_code == 200
            l_data_file = DataFile(**l_res.json())
            assert l_data_file.metadata.data_path is not None
            assert l_data_file.metadata == data_file.metadata
            assert l_data_file.data is None
            assert data_file.data is not None

    @pytest.mark.asyncio
    async def test_update_data(
        self, ds_app: AsyncClient, data_file: DataFile, slice_list_model: SliceListModel
    ):
        """Test updating data"""
        async with ds_app:
            response = await ds_app.post("/data/init", content=data_file.json())
            assert response.status_code == 201
            assert isinstance(DataServer.frontend, FrontendV1)
            assert data_file.data is not None
            data_file.metadata.data_path = DataServer.frontend.data_backend.path(
                data_file.data
            )
            path = SavedPath(**response.json()).path
            response = await ds_app.post(
                f"/data/update/metadata/{str(path)}", content=data_file.json()
            )
            assert response.status_code == 200
            response = await ds_app.post(
                "/data/update/data/",
                content=DataFileAndSliceModel(data_file=data_file).json(),
            )
            assert response.status_code == 202
            await asyncio.sleep(0.01)
            await DataServer.frontend.complete_background_tasks()

            slice_lists = slice_list_model
            update_sizes = [(100,), (10, 10), (3, 3, 3), (2, 2, 2, 10)]
            updated_sizes = [(200,), (20, 10), (6, 3, 3), (4, 2, 2, 10)]

            rng = np.random.default_rng()
            for i, (dataset, update_size) in enumerate(
                zip(data_file.data.datasets, update_sizes)
            ):
                mem_out = dataset.memory.to_array()
                mem_out[:] = rng.standard_normal(size=update_size).astype(mem_out.dtype)
                if i == 2:
                    mem_out[:] = 2
                if i == 3:
                    mem_out[:] = False

            l_data = await self._load(ds_app, path, return_df=True)
            assert isinstance(l_data, DataFile)
            assert l_data.data is not None
            for dataset, l_dataset in zip(
                data_file.data.datasets, l_data.data.datasets
            ):
                assert dataset.memory.name != l_dataset.memory.name
                assert np.any(dataset.memory.to_array() != l_dataset.memory.to_array())

            response = await ds_app.post(
                "/data/update/data/",
                content=DataFileAndSliceModel(
                    data_file=data_file, slices=slice_lists
                ).json(),
            )
            assert response.status_code == 202
            await asyncio.sleep(0.01)
            await DataServer.frontend.complete_background_tasks()
            l_data2 = await self._load(ds_app, path, return_df=True)
            assert isinstance(l_data2, DataFile)
            assert l_data2.data is not None
            for dataset, l_dataset, u_shape, slice_list in zip(
                data_file.data.datasets,
                l_data2.data.datasets,
                updated_sizes,
                slice_lists.to_slice_lists(),
            ):
                assert dataset.name == l_dataset.name
                assert u_shape == l_dataset.memory.shape
                assert dataset.memory.dtype == l_dataset.memory.dtype
                assert np.all(
                    dataset.memory.to_array()
                    == l_dataset.memory.to_array()[tuple(slice_list)]
                )

            response = await ds_app.post("data/close", content=data_file.json())
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_update_data_callback(
        self,
        ds_app: AsyncClient,
        data_file: DataFile,
        callback_app: FastAPI,
        startup_and_shutdown_server,
    ):
        """test update with callback"""
        assert startup_and_shutdown_server.started
        async with ds_app:
            response = await ds_app.post("/data/init", content=data_file.json())
            assert isinstance(DataServer.frontend, FrontendV1)
            assert data_file.data is not None
            data_file.metadata.data_path = DataServer.frontend.data_backend.path(
                data_file.data
            )
            path = SavedPath(**response.json()).path
            await ds_app.post(
                f"/data/update/metadata/{str(path)}", content=data_file.json()
            )
            response = await ds_app.post(
                "/data/update/data/",
                content=DataFileAndSliceModel(data_file=data_file).json(),
                params={"callback_url": "http://127.0.0.1:8000/"},
            )
            assert response.status_code == 202
            assert response.json() == {"message": "data updating"}
            assert data_file.data.handle in callback_app.updated_handles  # type: ignore

    @pytest.mark.asyncio
    async def test_update_data_file(
        self, ds_app: AsyncClient, data_file: DataFile, slice_list_model: SliceListModel
    ):
        """Test updating data file"""
        async with ds_app:
            response = await ds_app.post("/data/init", content=data_file.json())
            assert response.status_code == 201
            assert isinstance(DataServer.frontend, FrontendV1)
            assert data_file.data is not None
            data_file.metadata.data_path = DataServer.frontend.data_backend.path(
                data_file.data
            )
            path = SavedPath(**response.json()).path
            response = await ds_app.post(
                f"/data/update/{str(path)}",
                content=DataFileAndSliceModel(data_file=data_file).json(),
            )
            assert response.status_code == 200
            await asyncio.sleep(0.01)
            await DataServer.frontend.complete_background_tasks()

            slice_lists = slice_list_model
            update_sizes = [(100,), (10, 10), (3, 3, 3), (2, 2, 2, 10)]
            updated_sizes = [(200,), (20, 10), (6, 3, 3), (4, 2, 2, 10)]

            await DataServer.frontend.complete_background_tasks()
            rng = np.random.default_rng()
            for i, (dataset, update_size) in enumerate(
                zip(data_file.data.datasets, update_sizes)
            ):
                mem_out = dataset.memory.to_array()
                mem_out[:] = rng.standard_normal(size=update_size).astype(mem_out.dtype)
                if i == 2:
                    mem_out[:] = 2
                if i == 3:
                    mem_out[:] = False

            l_data = await self._load(ds_app, path, return_df=True)
            assert isinstance(l_data, DataFile)
            assert l_data.data is not None
            for dataset, l_dataset in zip(
                data_file.data.datasets, l_data.data.datasets
            ):
                assert dataset.memory.name != l_dataset.memory.name
                assert np.any(dataset.memory.to_array() != l_dataset.memory.to_array())

            response = await ds_app.post(
                f"/data/update/{path}",
                content=DataFileAndSliceModel(
                    data_file=data_file, slices=slice_lists
                ).json(),
            )
            assert response.status_code == 200
            await asyncio.sleep(0.01)
            await DataServer.frontend.complete_background_tasks()
            l_data2 = await self._load(ds_app, path, return_df=True)
            assert isinstance(l_data2, DataFile)
            assert l_data2.data is not None
            for dataset, l_dataset, u_shape, slice_list in zip(
                data_file.data.datasets,
                l_data2.data.datasets,
                updated_sizes,
                slice_lists.to_slice_lists(),
            ):
                assert dataset.name == l_dataset.name
                assert u_shape == l_dataset.memory.shape
                assert dataset.memory.dtype == l_dataset.memory.dtype
                assert np.all(
                    dataset.memory.to_array()
                    == l_dataset.memory.to_array()[tuple(slice_list)]
                )

            response = await ds_app.post("data/close", content=data_file.json())
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_close(self, ds_app: AsyncClient, data_file: DataFile):
        """Test closing data in DataFile"""
        assert data_file.data
        for dataset in data_file.data.datasets:
            assert dataset.memory.name in __memory_reference__
        response = await ds_app.post("data/close", content=data_file.json())
        assert response.status_code == 200
        for dataset in data_file.data.datasets:
            assert dataset.memory.name not in __memory_reference__

    @pytest.mark.asyncio
    async def test_create_delete_memory(self, ds_app: AsyncClient, shapes_and_dtypes):
        shapes, dtypes = shapes_and_dtypes
        mems_in = [
            SharedMemoryIn(dtype=dtype, shape=shape)
            for (shape, dtype) in zip(shapes, dtypes)
        ]
        for mem_in in mems_in:
            response = await ds_app.post("/data/memory", content=mem_in.json())
            assert response.status_code == 200
            mem_out = SharedMemoryOut(**response.json())
            assert mem_out.name in __memory_reference__
            assert mem_out.shape == mem_in.shape
            assert mem_out.dtype == mem_in.dtype
            assert mem_out.to_array().shape == mem_in.shape
            assert mem_out.to_array().dtype == mem_in.dtype

            res2 = await ds_app.delete(f"/data/memory/{mem_out.name}")
            assert res2.status_code == 200
            assert mem_out.name not in __memory_reference__


class TestDataRouterAppStreaming(DataRouterAppTests):
    """Streaming Tests for Data Router"""

    @pytest.mark.asyncio
    async def test_streaming_1d(
        self,
        ds_app: AsyncClient,
        streamed_data_file: DataFile,
        buffer_references: list[BufferReference],
    ):
        """Test preaparing stream saving buffers and finalizing stream"""
        async_lock = asyncio.Lock()
        async for save_buffer in self.streaming_test_generator(
            ds_app, streamed_data_file, buffer_references, (10,)
        ):
            async with async_lock:
                await save_buffer

    @pytest.mark.asyncio
    async def test_streaming_1d_random(
        self,
        ds_app: AsyncClient,
        streamed_data_file: DataFile,
        buffer_references: list[BufferReference],
    ):
        """Test preaparing stream saving buffers in a random order
        and finalizing stream"""
        async_lock = asyncio.Lock()
        async for save_buffer in self.streaming_test_generator(
            ds_app, streamed_data_file, buffer_references, (10,), random=True
        ):
            async with async_lock:
                await save_buffer

    @pytest.mark.asyncio
    async def test_streaming_1d_gather(
        self,
        ds_app: AsyncClient,
        streamed_data_file: DataFile,
        buffer_references: list[BufferReference],
    ):
        """Test preaparing stream saving buffers asynchronously
        and finalizing stream"""
        async for save_buffer in self.streaming_test_generator(
            ds_app, streamed_data_file, buffer_references, (10,)
        ):
            await save_buffer

    @pytest.mark.asyncio
    async def test_streaming_1d_gather_random(
        self,
        ds_app: AsyncClient,
        streamed_data_file: DataFile,
        buffer_references: list[BufferReference],
    ):
        """Test preaparing stream saving buffers asynchronously and in a random order
        and finalizing stream"""
        async for save_buffer in self.streaming_test_generator(
            ds_app, streamed_data_file, buffer_references, (10,), random=True
        ):
            await save_buffer

    @pytest.mark.asyncio
    async def test_streaming_2d(
        self,
        ds_app: AsyncClient,
        streamed_data_file_2d: DataFile,
        buffer_references_2d: list[BufferReference],
    ):
        """Test preaparing stream saving buffers and finalizing stream 2d"""
        async for save_buffer in self.streaming_test_generator(
            ds_app, streamed_data_file_2d, buffer_references_2d, (2, 3), random=True
        ):
            await save_buffer

    async def streaming_test_generator(
        self,
        ds_app: AsyncClient,
        streamed_data_file: DataFile,
        buffer_references: list[BufferReference],
        size: tuple,
        random: bool = False,
    ):
        """Generate buffer lists for streaming tests

        Args:
            ds_app (AsyncClient): client to communicate with data server
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
        handle = MeasurementHandle.new()
        assert streamed_data_file.data is not None
        streamed_data_file.data.handle = handle
        start_stream_model = StartStreamModel(handle=handle, file=streamed_data_file)
        assert start_stream_model.file.data is not None
        assert start_stream_model.handle == start_stream_model.file.data.handle

        end_stream_model = EndStreamModel(handle=handle)

        async with ds_app:
            res = await ds_app.post(
                "/data/stream/start", content=start_stream_model.json()
            )
            assert res.status_code == 201
            path = SavedPath(**res.json()).path
            l_data = await self._load(ds_app, path, return_df=True)
            assert isinstance(l_data, DataFile)
            assert streamed_data_file.metadata.data_path is None
            assert l_data.metadata.data_path is not None
            l_data.metadata.data_path = None
            assert_metadata_and_handle_match(streamed_data_file, l_data)
            assert streamed_data_file.data is not None
            assert_datasets_match(
                l_data.data.datasets,  # type: ignore
                streamed_data_file.data.datasets,
            )

            assert len(l_data.data.datasets) == len(buffer_references)  # type: ignore
            assert l_data.data is not None
            [
                SharedMemoryOut.close(dataset.memory.name)
                for dataset in l_data.data.datasets
            ]
            buffer_data = buffer_upload_data(buffer_references, size)
            arr = np.arange(np.prod(size)).reshape(size)
            if random:
                np.random.shuffle(arr)
            arr_iter = np.nditer(arr, flags=["multi_index"])
            for val in arr_iter:
                idx = np.where(arr == val)
                for buffer, dataset, b_data in zip(
                    buffer_references, streamed_data_file.data.datasets, buffer_data
                ):
                    yield self._save_buffer_test(
                        buffer,
                        dataset,
                        start_stream_model.handle,
                        b_data,
                        tuple(i[0] for i in idx),
                        ds_app,
                        path,
                        serial=False,
                    )

            l_data = await self._load(ds_app, path, return_df=True)
            assert isinstance(l_data, DataFile)
            assert streamed_data_file.metadata.data_path is None
            assert l_data.metadata.data_path is not None
            l_data.metadata.data_path = None
            assert_metadata_and_handle_match(streamed_data_file, l_data)
            assert_datasets_match(
                streamed_data_file.data.datasets,
                l_data.data.datasets,  # type: ignore
                buffer_data,
            )

            await ds_app.post("/data/stream/end", content=end_stream_model.json())
            assert l_data.data is not None
            [
                SharedMemoryOut.close(dataset.memory.name)
                for dataset in l_data.data.datasets
            ]

    async def _save_buffer_test(
        self,
        buffer: BufferReference,
        dataset: Data.DataSet,
        handle: MeasurementHandle,
        buffer_data: np.ndarray,
        idx: tuple,
        ds_app: AsyncClient,
        path: Path,
        serial: bool = True,
    ):
        """function for uploading data in buffer reference,
        downloading it and testing the downloaded data"""

        async def _current_shape(path: Path) -> tuple:
            l_data = await self._load(ds_app, path)
            assert isinstance(l_data, Data)
            l_dataset = [dset for dset in l_data.datasets if dset.name == dataset.name][
                0
            ]
            [SharedMemoryOut.close(dataset.memory.name) for dataset in l_data.datasets]
            return l_dataset.memory.shape

        # get current shape before uploading new data
        # such we can verify that the reshaping works correctly
        current_shape = await _current_shape(path)

        new_buffer, slices = new_buffer_and_slices(dataset, idx, buffer, buffer_data)
        sbm = SaveBufferModel(
            handle=handle,
            name=dataset.name,
            buffer=new_buffer,
            index=list(index(dataset, idx)),
        )
        res = await ds_app.post("/data/save/buffer", content=sbm.json())
        assert res.status_code == 202
        assert res.json() == {"message": "buffer posted"}
        l_data = await self._load(ds_app, path)
        assert isinstance(l_data, Data)
        assert l_data is not None
        l_dataset = [dset for dset in l_data.datasets if dset.name == dataset.name][0]
        assert dataset.memory.dtype == l_dataset.memory.dtype
        if serial:
            assert (
                new_shape(current_shape, slices, dataset.memory.shape)
                == l_dataset.memory.shape
            )
        [SharedMemoryOut.close(dataset.memory.name) for dataset in l_data.datasets]
        SharedMemoryOut.close(new_buffer.name)

    @pytest.mark.asyncio
    async def test_streaming_callback(
        self,
        ds_app: AsyncClient,
        callback_app: FastAPI,
        streamed_data_file: DataFile,
        buffer_references: list[BufferReference],
        startup_and_shutdown_server,
    ):
        """Test preaparing stream saving buffers with callback"""
        handle = MeasurementHandle.new()
        assert streamed_data_file.data is not None
        streamed_data_file.data.handle = handle
        start_stream_model = StartStreamModel(handle=handle, file=streamed_data_file)
        assert start_stream_model.file.data is not None
        assert start_stream_model.handle == start_stream_model.file.data.handle

        assert startup_and_shutdown_server.started

        size = (10,)

        async with ds_app:
            await ds_app.post("/data/stream/start", content=start_stream_model.json())

            arr = np.arange(np.prod(size)).reshape(size)
            arr_iter = np.nditer(arr, flags=["multi_index"])
            for val in arr_iter:
                idex = np.where(arr == val)
                for buffer, dataset, b_data in zip(
                    buffer_references,
                    streamed_data_file.data.datasets,
                    buffer_upload_data(buffer_references, size),
                ):
                    idx = tuple(i[0] for i in idex)
                    new_buffer, _ = new_buffer_and_slices(dataset, idx, buffer, b_data)
                    sbm = SaveBufferModel(
                        handle=handle,
                        name=dataset.name,
                        buffer=new_buffer,
                        index=list(index(dataset, idx)),
                    )
                    res = await ds_app.post(
                        "/data/save/buffer",
                        content=sbm.json(),
                        params={"callback_url": "http://127.0.0.1:8000/"},
                    )
                    assert res.status_code == 202
                    assert res.json() == {"message": "buffer posted"}
                    assert new_buffer in callback_app.saved_buffers  # type: ignore
                    SharedMemoryOut.close(new_buffer.name)
            await ds_app.post(
                "/data/stream/end", content=EndStreamModel(handle=handle).json()
            )
        [SharedMemoryOut.close(br.name) for br in buffer_references]

    @pytest.mark.asyncio
    async def test_stream_prepare_twice(
        self,
        ds_app: AsyncClient,
        start_stream_model: StartStreamModel,
    ):
        """Test preaparing a stream twice result in 409 after 2nd call"""
        async with ds_app:
            res1 = await ds_app.post(
                "/data/stream/start", content=start_stream_model.json()
            )
            assert res1.status_code == 201
            res2 = await ds_app.post(
                "/data/stream/start", content=start_stream_model.json()
            )
            assert res2.status_code == 409

    @pytest.mark.asyncio
    async def test_stream_prepare_no_data(
        self,
        ds_app: AsyncClient,
        start_stream_model: StartStreamModel,
    ):
        """Test preaparing a stream with no data results in 415"""
        start_stream_model.file.data = None
        async with ds_app:
            res1 = await ds_app.post(
                "/data/stream/start", content=start_stream_model.json()
            )
            assert res1.status_code == 415

    @pytest.mark.asyncio
    async def test_end_unprepared_or_closed_stream(
        self,
        ds_app: AsyncClient,
        end_stream_model: EndStreamModel,
        start_stream_model: StartStreamModel,
    ):
        """Test preaparing a stream with no data results in 415"""
        assert start_stream_model.handle == end_stream_model.handle
        async with ds_app:
            res1 = await ds_app.post(
                "/data/stream/end", content=end_stream_model.json()
            )
            assert res1.status_code == 409

            res2 = await ds_app.post(
                "/data/stream/start", content=start_stream_model.json()
            )
            assert res2.status_code == 201

            res3 = await ds_app.post(
                "/data/stream/end", content=end_stream_model.json()
            )
            assert res3.status_code == 200

            res4 = await ds_app.post(
                "/data/stream/end", content=end_stream_model.json()
            )
            assert res4.status_code == 409

    @pytest.mark.asyncio
    async def test_save_buffer_for_unprepared_stream(
        self,
        ds_app: AsyncClient,
        buffer_references: list[BufferReference],
    ):
        """Test saving buffer with a unprepared handle results in 409"""

        handle = MeasurementHandle.new()
        sbm = SaveBufferModel(
            handle=handle,
            name="n/a",
            buffer=buffer_references[0],
            index=[0, 1],
        )
        async with ds_app:
            res1 = await ds_app.post("/data/save/buffer", content=sbm.json())
            assert res1.status_code == 409
            assert (
                "Stream has not been prepared using MeasurmentHandle"
                in res1.json()["detail"]
            )

    @pytest.mark.asyncio
    async def test_save_buffer_for_wrong_name(
        self,
        ds_app: AsyncClient,
        buffer_references: list[BufferReference],
        start_stream_model: StartStreamModel,
    ):
        """Test preaparing a stream twice result in 409 after 2nd call"""

        sbm = SaveBufferModel(
            handle=start_stream_model.handle,
            name="wrong_name",
            buffer=buffer_references[0],
            index=[0, 1],
        )
        async with ds_app:
            res1 = await ds_app.post(
                "/data/stream/start", content=start_stream_model.json()
            )
            assert res1.status_code == 201

            res2 = await ds_app.post("/data/save/buffer", content=sbm.json())
            assert res2.status_code == 409
            assert (
                "'wrong_name':not in names initialised for handle"
                in res2.json()["detail"]
            )

    @pytest.mark.asyncio
    async def test_save_buffer_for_wrong_shape(
        self,
        ds_app: AsyncClient,
        start_stream_model: StartStreamModel,
    ):
        """test saveing buffer with wrong shape gives 415"""
        dataset = start_stream_model.file.data.datasets[0]  # type: ignore
        buffer = BufferReference(
            dtype=dataset.memory.dtype,
            shape=(999,),
            name="dummy",
        )

        sbm = SaveBufferModel(
            handle=start_stream_model.handle,
            name=dataset.name,  # type: ignore
            buffer=buffer,
            index=[2, 0],
        )
        async with ds_app:
            res1 = await ds_app.post(
                "/data/stream/start", content=start_stream_model.json()
            )
            assert res1.status_code == 201

            res2 = await ds_app.post("/data/save/buffer", content=sbm.json())
            assert res2.status_code == 415
            assert (
                "'test_name_0':Shape of data referenced by buffer reference (999,)"
                in res2.json()["detail"]
            )

    @pytest.mark.asyncio
    async def test_save_buffer_for_wrong_datatype(
        self,
        ds_app: AsyncClient,
        start_stream_model: StartStreamModel,
    ):
        """test saveing buffer with wrong datatype gives 415"""
        dataset = start_stream_model.file.data.datasets[0]  # type: ignore
        buffer = BufferReference(
            dtype="complex",
            shape=(1, 10),
            name="dummy",
        )

        sbm = SaveBufferModel(
            handle=start_stream_model.handle,
            name=dataset.name,  # type: ignore
            buffer=buffer,
            index=[2, 0],
        )
        async with ds_app:
            res1 = await ds_app.post(
                "/data/stream/start", content=start_stream_model.json()
            )
            assert res1.status_code == 201

            res2 = await ds_app.post("/data/save/buffer", content=sbm.json())
            assert res2.status_code == 415
            assert (
                "'test_name_0:'datatype referenced by buffer reference complex128"
                in res2.json()["detail"]
            )

    @pytest.mark.asyncio
    async def test_save_buffer_for_wrong_index(
        self,
        ds_app: AsyncClient,
        start_stream_model: StartStreamModel,
    ):
        """test saveing buffer with wrong index length gives 415"""
        dataset = start_stream_model.file.data.datasets[0]  # type: ignore
        buffer = BufferReference(
            dtype="float",
            shape=(1, 10),
            name="dummy",
        )

        sbm = SaveBufferModel(
            handle=start_stream_model.handle,
            name=dataset.name,  # type: ignore
            buffer=buffer,
            index=[1, 0, 0],
        )
        async with ds_app:
            res1 = await ds_app.post(
                "/data/stream/start", content=start_stream_model.json()
            )
            assert res1.status_code == 201

            res2 = await ds_app.post("/data/save/buffer", content=sbm.json())
            assert res2.status_code == 415
            assert (
                "'test_name_0:': index (3) and buffer (2) dimensions don't match"
                in res2.json()["detail"]
            )


class TestDataRouterGCSApp(TestDataRouterAppStreaming):
    """Class for testing data server app with frontend v1 using gcs data backend"""

    @pytest.fixture(name="ds_app")
    def fixture_ds_app_gcs(self, frontend_mdb_gcs: FrontendV1):
        """create an app that includeds a data server router (ds_router)
        and a frontend with mongo and fsspec filesystem"""
        yield self.create_app(frontend_mdb_gcs)


class TestDataRouterAppFromConfig(DataRouterAppTests):
    """Class for testing data server app with frontend v1 using fs data backend,
    created from config"""

    @pytest_asyncio.fixture(name="frontend_config")
    async def fixture_frontend_fs_config(self, frontend_fs_config):
        """Fixture that returns a frontend mdb/fs backed config"""
        yield frontend_fs_config

    @pytest_asyncio.fixture(name="ds_app")
    async def fixture_ds_app_fs_config(self, frontend_config: FrontendV1Config):
        """create an app that includeds a data server router (ds_router)
        and a frontend with mongo and fsspec filesystem"""
        DataServer.set_frontend(None)  # type: ignore
        assert DataServer.frontend is None
        app = FastAPI()
        app.include_router(ds_router, prefix="/data")
        client = AsyncClient(app=app, base_url="http://test")
        assert DataServer.frontend is None
        await client.post("/data/setup", content=frontend_config.json())
        assert DataServer.frontend is not None
        yield AsyncClient(app=app, base_url="http://test")
        DataServer.set_frontend(None)  # type: ignore

    @pytest.mark.asyncio
    async def test_reseting_data_server(self, ds_app: AsyncClient):
        """Test reseting the data server"""
        async with ds_app:
            assert DataServer.frontend is not None
            res = await ds_app.put("/data/reset")
            assert res.status_code == 202
            assert DataServer.frontend is None

    @pytest.mark.asyncio
    async def test_setting_data_server_twice(
        self, ds_app: AsyncClient, frontend_config: FrontendV1Config
    ):
        """Test that setting the dataserver up twice results in 409"""
        # data server should already be setup by the ds_app fixture
        async with ds_app:
            assert DataServer.frontend is not None
            res = await ds_app.post("/data/setup", content=frontend_config.json())
            assert res.status_code == 409
            assert DataServer.frontend is not None


class TestDataRouterGCSAppFromConfig(DataRouterAppTests):
    """Class for testing data server app with frontend v1 using gcs data backend,
    created from config"""

    @pytest_asyncio.fixture(name="frontend_config")
    async def fixture_frontend_fs_config(self, frontend_gcs_config):
        """Fixture that returns a frontend mdb/fs backed config"""
        yield frontend_gcs_config
