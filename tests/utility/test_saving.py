# import multiprocessing

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from svalbard.data_router import ds_router, lifespan
from svalbard.data_server.frontend.frontend_v1 import FrontendV1Config
from svalbard.launch import DEFAULT_SERVER_URL
from svalbard.utility.saving import (
    save_data_file,
    save_data_file_async,
    save_data_file_in_background,
)

from ..data_router.utility import UvicornTestServer


@pytest.fixture(name="data_server", autouse=False, scope="module")
async def data_server_fixture(frontend_gcs_config: FrontendV1Config):
    """Fixture for creating a data server"""
    EXAMPLES_FOLDER = Path(__file__).parent.parent.parent / "examples"
    LOCAL_CONFIG_PATH = EXAMPLES_FOLDER / "data_server_local.json"
    lifespan_ = functools.partial(
        lifespan,
        data_config_path=LOCAL_CONFIG_PATH,
    )

    app = FastAPI(lifespan=lifespan_)
    app.include_router(ds_router, prefix="/data")
    server = UvicornTestServer(app, port=5000)
    await server.bring_up()
    async with httpx.AsyncClient(base_url=DEFAULT_SERVER_URL) as client:
        await client.put("/data/reset")
        response = await client.post(
            "/data/setup", content=frontend_gcs_config.model_dump_json()
        )
        assert response.status_code == 201
    yield server
    await server.down()


@pytest.mark.asyncio
async def test_save_data_file_async(data_server, data_file):
    try:
        async for ds in data_server:
            await asyncio.sleep(0.1)
            await save_data_file_async(data_file)
    except Exception as e:
        await ds.down()
        raise e


@pytest.mark.asyncio
async def test_already_setup(data_server, server_address):
    """Test that the data server can be started"""
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
    async for ds in data_server:
        async with httpx.AsyncClient(base_url=DEFAULT_SERVER_URL) as client:
            response = await client.post(
                "/data/setup", content=frontend_config.model_dump_json()
            )
            assert response.status_code == 409


@pytest.mark.asyncio
async def test_save_data_file_async_error(data_server, metadata):
    try:
        async for ds in data_server:
            with pytest.raises(httpx.HTTPStatusError):
                await save_data_file_async(metadata)
    except Exception as e:
        await ds.down()
        raise e


@pytest.mark.asyncio
async def test_save_data_file_in_background(data_server, data_file):
    try:
        async for ds in data_server:
            task = save_data_file_in_background(data_file)
            await task
            task.result()
    except Exception as e:
        await ds.down()
        raise e


@pytest.mark.asyncio
async def test_save_data_file_in_background_error(data_server, metadata):
    try:
        async for ds in data_server:
            with pytest.raises(httpx.HTTPStatusError):
                task = save_data_file_in_background(metadata)
                await task
    except Exception as e:
        await ds.down()
        raise e


@pytest.mark.asyncio
async def test_save_data_file_in_background_timeout_warning(data_server, data_file):
    try:
        async for ds in data_server:
            with pytest.warns(UserWarning):
                task = save_data_file_in_background(data_file, timeout=0.01)
                with pytest.raises(httpx.ReadTimeout):
                    await task
    except Exception as e:
        await ds.down()
        raise e


def make_awaitable(func, *args, **kwargs):
    pool = ThreadPoolExecutor()
    future = pool.submit(func, *args, **kwargs)
    return asyncio.wrap_future(future)


@pytest.mark.asyncio
async def test_save_data_file(data_server, data_file):
    try:
        async for ds in data_server:
            await make_awaitable(save_data_file, data_file)
    except Exception as e:
        await ds.down()
        raise e


@pytest.mark.asyncio
async def test_save_data_file_err(data_server, metadata):
    try:
        async for ds in data_server:
            with pytest.raises(httpx.HTTPStatusError):
                await make_awaitable(save_data_file, metadata)
    except Exception as e:
        await ds.down()
        raise e
