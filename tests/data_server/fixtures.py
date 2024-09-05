import asyncio
import sys
from pathlib import Path

import pytest
import pytest_asyncio
import requests
from fsspec.asyn import get_loop
from fsspec.implementations.memory import MemoryFileSystem
from gcsfs import GCSFileSystem
from gcsfs.retry import HttpError

from svalbard.data_server.data_backend.fs_backend import FSBackend, FSBackendConfig
from svalbard.data_server.data_backend.gcs_backend import (
    GCSBackend,
    GCSBackendConfig,
    Scopes,
)
from svalbard.data_server.frontend.frontend_v1 import FrontendV1, FrontendV1Config
from svalbard.data_server.frontend.sync_frontend_v1 import (
    SyncFrontendV1,
    SyncFrontendV1Config,
)
from svalbard.data_server.metadata_backend.mongodb_backend import (
    MongoDBBackend,
    MongoDBConfig,
)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(name="test_bucket", scope="session")
def fixture_test_bucket():
    yield "aq_test"


@pytest.fixture(name="server_address", scope="session")
def fixture_server_address():
    """fixture for correctly setting the server address
    such that both local and CI server tests can run"""
    yield "mongodb" if sys.platform == "linux" else "localhost"


@pytest.fixture(name="mdb_backend", scope="session")
def fixture_mdb_backend(server_address):
    """Create a MongoDBBackend for testing purposes"""
    mdb_backend = MongoDBBackend(
        f"mongodb://root:example@{server_address}:27017",
        "aq_test",
        "test",
    )
    yield mdb_backend
    mdb_backend.client.close()


@pytest.fixture(name="mdb_backend_module", scope="module")
def fixture_mdb_backend_module(server_address):
    """Create a MongoDBBackend for testing purposes"""
    mdb_backend = MongoDBBackend(
        f"mongodb://root:example@{server_address}:27017",
        "aq_test",
        "test",
    )
    yield mdb_backend
    mdb_backend.client.close()


@pytest_asyncio.fixture(name="fs_backend", scope="session")
def fixture_fs_backend(event_loop):
    """Fixture for creating fs backend for use in other tests"""
    mfs = MemoryFileSystem()
    path_base = Path("memory:/tmp/test/fs_backend")
    yield FSBackend(
        mfs,
        loop=event_loop,
        path_base=path_base,
    )
    try:
        mfs.rm(str(path_base), recursive=True)
    except FileNotFoundError:
        pass


@pytest.fixture(name="sync_fs_backend", scope="session")
def fixture_sync_fs_backend():
    """Fixture for creating fs backend for use in other tests"""
    mfs = MemoryFileSystem()
    path_base = Path("memory:/tmp/test/fs_backend")
    yield FSBackend(
        mfs,
        loop=get_loop(),
        path_base=path_base,
    )
    try:
        mfs.rm(str(path_base), recursive=True)
    except FileNotFoundError:
        pass


@pytest.fixture(name="server_address_gcs", scope="session")
def fixture_server_address_gcs():
    """fixture for correctly setting the server address
    such that both local and CI server tests can run"""
    try:
        requests.get("http://localhost:4443/storage/v1/b", timeout=0.1)
        yield "localhost"
    except requests.exceptions.ConnectionError:
        yield "fake-gcs-server"


@pytest.fixture(name="gcs_filesystem", scope="session")
def fixture_gcsfs(server_address_gcs, test_bucket):
    """Fixture for creating a gcsfs with an initialised bucket"""
    gcsfs = GCSFileSystem(endpoint_url=f"http://{server_address_gcs}:4443", timeout=1)
    try:
        gcsfs.mkdir(test_bucket)
    except HttpError:
        pass
    yield gcsfs


@pytest_asyncio.fixture(name="gcs_backend", scope="session")
def fixture_gcs_backend(event_loop, test_bucket, gcs_filesystem: GCSFileSystem):
    """Fixture for creating fs backend for use in other tests"""
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

        path_base = Path(test_bucket)
        yield GCSBackend(
            gcs_filesystem,
            loop=event_loop,
            path_base=path_base,
        )
    finally:
        try:
            gcs_filesystem.rm(gcs_filesystem.find(test_bucket))
        except FileNotFoundError:
            pass


@pytest_asyncio.fixture(name="sync_gcs_backend", scope="session")
def fixture_sync_gcs_backend(test_bucket, gcs_filesystem: GCSFileSystem):
    """Fixture for creating fs backend for use in other tests"""
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

        path_base = Path(test_bucket)
        yield GCSBackend(
            gcs_filesystem,
            loop=get_loop(),
            path_base=path_base,
        )
    finally:
        try:
            gcs_filesystem.rm(gcs_filesystem.find(test_bucket))
        except FileNotFoundError:
            pass


@pytest_asyncio.fixture(name="frontend_mdb_fs", scope="session")
def fixture_frontend_with_mongo_and_fs(
    mdb_backend: MongoDBBackend, fs_backend: FSBackend
):
    """Fixture that creates a frontend with
    Mongo DB metadata backend and FS data backend"""
    yield FrontendV1(fs_backend, mdb_backend)


@pytest_asyncio.fixture(name="frontend_mdb_gcs", scope="session")
def fixture_frontend_with_mongo_and_gcs(
    mdb_backend: MongoDBBackend, gcs_backend: GCSBackend
):
    """Fixture that creates a frontend with
    Mongo DB metadata backend and FS data backend"""
    yield FrontendV1(gcs_backend, mdb_backend)


@pytest.fixture(name="mdb_config", scope="session")
def fixture_mdb_config(server_address):
    """Fixture for creating a valid mongo db config"""
    yield MongoDBConfig(
        url=f"mongodb://root:example@{server_address}:27017",
        database="aq_test",
        collection="test",
    )


@pytest.fixture(name="fs_config", scope="session")
def fixture_fs_config():
    """fixture for creating a fs backend config"""
    yield FSBackendConfig(path=Path("memory:/tmp/test/fs_backend_config"))


@pytest.fixture(name="gcs_config", scope="session")
def fixture_gcs_config(server_address_gcs, test_bucket):
    """fixture for creating a fs backend config"""
    yield GCSBackendConfig(
        path=Path(test_bucket),
        access=Scopes.FULL_CONTROL,
        endpoint_url=f"http://{server_address_gcs}:4443",
        timeout=1,
    )


@pytest.fixture(name="frontend_fs_config", scope="session")
def fixture_frontend_fs(mdb_config: MongoDBConfig, fs_config: FSBackendConfig):
    """Fixutre for creating a frontend with mongo db metadata
    backend and fsspec filesytem based data backend"""
    yield FrontendV1Config(data_backend=fs_config, metadata_backend=mdb_config)


@pytest.fixture(name="frontend_gcs_config", scope="session")
def fixture_frontend_fsgcs(mdb_config: MongoDBConfig, gcs_config: GCSBackendConfig):
    """Fixutre for creating a frontend with mongo db metadata
    backend and gcsfs filesytem based data backend"""
    yield FrontendV1Config(data_backend=gcs_config, metadata_backend=mdb_config)


@pytest.fixture(name="sync_frontend_fs_config", scope="session")
def fixture_sync_frontend_fs(mdb_config: MongoDBConfig, fs_config: FSBackendConfig):
    """Fixutre for creating a frontend with mongo db metadata
    backend and fsspec filesytem based data backend"""
    yield SyncFrontendV1Config(data_backend=fs_config, metadata_backend=mdb_config)


@pytest.fixture(name="sync_frontend_gcs_config", scope="session")
def fixture_sync_frontend_fsgcs(
    mdb_config: MongoDBConfig, gcs_config: GCSBackendConfig
):
    """Fixutre for creating a frontend with mongo db metadata
    backend and gcsfs filesytem based data backend"""
    yield SyncFrontendV1Config(data_backend=gcs_config, metadata_backend=mdb_config)


@pytest.fixture(name="sync_frontend_mdb_fs", scope="module")
def fixture_sync_frontend_with_mongo_and_fs(
    mdb_backend_module: MongoDBBackend, sync_fs_backend: FSBackend
):
    """Fixture that creates a frontend with
    Mongo DB metadata backend and FS data backend"""
    yield SyncFrontendV1(sync_fs_backend, mdb_backend_module)


@pytest.fixture(name="sync_frontend_mdb_gcs", scope="module")
def fixture_sync_frontend_with_mongo_and_gcs(
    mdb_backend_module: MongoDBBackend, sync_gcs_backend: GCSBackend
):
    """Fixture that creates a frontend with
    Mongo DB metadata backend and FS data backend"""
    yield SyncFrontendV1(sync_gcs_backend, mdb_backend_module)


@pytest.fixture(name="final_fixture", scope="session")
async def fixture_final_fixture(mdb_backend: MongoDBBackend):
    """Fixture that is run once at the end of all tests"""
    yield
    await mdb_backend.client.drop_database(mdb_backend.database.name())  # type: ignore
    mdb_backend.client.close()


@pytest.fixture(name="slice_lists", scope="session")
def fixture_slice_lists():
    """Fixture for creating slice lists"""
    yield [
        [slice(100, 200)],
        [slice(10, 20), slice(0, 10)],
        [slice(3, 6), slice(0, 3), slice(0, 3)],
        [slice(2, 4), slice(0, 2), slice(0, 2), slice(0, 10)],
    ]
