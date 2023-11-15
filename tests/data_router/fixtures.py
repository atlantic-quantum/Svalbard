import pytest
import pytest_asyncio
from fastapi import FastAPI
from svalbard.data_model.ipc import BufferReference, MeasurementHandle, SliceListModel
from svalbard.utility.logger import logger

from .utility import UvicornTestServer


@pytest.fixture(name="callback_app")
def fixture_callback_app():
    """fixture for creating a fastAPI app that recieves callbacks"""
    app = FastAPI()
    app.saved_buffers = []  # type: ignore
    app.updated_handles = []  # type: ignore

    @app.post("/buffer/saved")
    def buffer_saved(buffer: BufferReference):
        logger.debug("callback app called with buffer: %s", buffer)
        app.saved_buffers.append(buffer)  # type: ignore

    @app.post("/data/updated")
    def data_updated(handle: MeasurementHandle):
        logger.debug("callback app called with buffer: %s", handle)
        app.updated_handles.append(handle)  # type: ignore

    yield app


@pytest_asyncio.fixture
async def startup_and_shutdown_server(callback_app, capsys):
    """Start server as test fixture and tear down after test"""
    server = UvicornTestServer(callback_app)
    await server.bring_up()
    yield server
    await server.down()
    # capsys.readouterr()


@pytest.fixture(name="slice_list_model")
def fixture_slice_list_model(slice_lists: list[list[slice]]):
    """Fixture for creating a SliceListModel"""
    yield SliceListModel.from_slice_lists(slice_lists)
