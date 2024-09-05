"""Fast API Router for the Data Server"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from functools import partial, wraps
from pathlib import Path

import httpx
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, status
from pydantic import AnyHttpUrl

from ..data_model.data_file import DataFile
from ..data_model.ipc import (
    BufferReference,
    DataFileAndSliceModel,
    EndStreamModel,
    MeasurementHandle,
    SaveBufferModel,
    SavedPath,
    SliceListModel,
    StartStreamModel,
)
from ..data_model.memory_models import SharedMemoryIn, SharedMemoryOut
from ..data_server.errors import (
    BufferShapeError,
    StreamAlreadyPreparedError,
    StreamNotPreparedError,
)
from ..data_server.frontend.frontend_v1 import FrontendV1, FrontendV1Config
from ..utility.constants import ASYNCIO_SLEEP, MONITOR_TIMOUT
from ..utility.logger import logger
from ..utility.shared_array import SharedCounter

DATA_SERVER_CONFIG = "data_server.json"
DATA_SERVER_TEST_CONFIG = "data_server_test.json"
DATA_SERVER_LOCAL_CONFIG = "data_server_local.json"
DATA_SERVER_FILE_CONFIG = "data_server_file.json"
CONFIG_PATH = Path("~/.aq_config").expanduser()
DATA_CONFIG_PATH = CONFIG_PATH / DATA_SERVER_CONFIG
DATA_CONFIG_TEST_PATH = CONFIG_PATH / DATA_SERVER_TEST_CONFIG
DATA_CONFIG_LOCAL_PATH = CONFIG_PATH / DATA_SERVER_LOCAL_CONFIG
DATA_CONFIG_FILE_PATH = CONFIG_PATH / DATA_SERVER_FILE_CONFIG


class DataServer:
    """Container class around the dataserver frontend"""

    _FRONTEND: FrontendV1 = None  # type: ignore
    NOT_SET_UP: str = (
        "Service Unavailable: Data Server has not been set up,"
        + "call '/setup' with a frontend config model to setup"
    )
    ALREADY_SET_UP: str = (
        "DataServer is already set up, use '/reset' to reset the data server"
        + " to an empty state then post a frontend config using '/setup'"
    )
    _MEMORIES: dict[str, SharedMemoryOut] = {}

    @classmethod
    def get_frontend(cls) -> FrontendV1:
        """frontend of the data server

        Returns:
            AbstractFrontend: frontend of the data server
        """
        return cls._FRONTEND

    @classmethod
    def set_frontend(cls, frontend: FrontendV1):
        """Sets the frontend of the data server

        Args:
            frontend (AbstractFrontend):
                set the frontend of the data server to this frontend
        """
        cls._FRONTEND = frontend


TAGS = ["Data", "Data Server"]

ds_router = APIRouter(
    tags=TAGS,  # type: ignore
)
callback_router = APIRouter(tags=["Callback"])

PATHS = [
    "/setup",
    "/reset",
    "/save",
    "/load",
    "/load_partial",
    "/init",
    "/stream/start",
    "/save/buffer",
    "/stream/end",
]


@asynccontextmanager
async def lifespan(  # pylint: disable=unused-argument
    app: FastAPI, data_config_path: str | Path = DATA_CONFIG_PATH
):  # pragma: no cover
    """Event called when the data server is started"""
    logger.info("DataServer started")
    if os.path.exists(data_config_path):
        logger.info("Loading DataServer from config")
        config_json = json.loads(Path(data_config_path).read_text(encoding="utf-8"))
        config = FrontendV1Config(**config_json)
        DataServer.set_frontend(config.init())
    else:
        logger.warning("No DataServer config found")
    yield
    logger.info("DataServer stopped")


@ds_router.post("/setup", status_code=status.HTTP_201_CREATED)
async def setup_data_server(config: FrontendV1Config):
    """Setup the DateServer by creating a frontend from the posted config.

    Args:
        config (FrontendV1Config):
            a frontend for the dataserver is created using this config

    Raises:
        HTTPException: 409 if a dataserver has already been set up
    """
    logger.debug("Setting up DataServer from config: %s", config)
    if DataServer.get_frontend() is not None:
        logger.warning("DataServer frontend already set up")
        raise HTTPException(status.HTTP_409_CONFLICT, detail=DataServer.ALREADY_SET_UP)
    DataServer.set_frontend(config.init())


@ds_router.put("/reset", status_code=status.HTTP_202_ACCEPTED)
async def reset_data_server():
    """Resets the DataServer by setting the frontend to None"""
    logger.debug("Reseting DataServer to no frontend")
    DataServer.set_frontend(None)  # type: ignore


def ds_setup_wrapper(coroutine):
    """Wraps error handling regarding the data server
    not being set up around other api routing functions

    Args:
        coroutine (asyncio.coroutine): the coroutine to wrap

    Raises:
        HTTPException: if the data server has not been set up

    Returns:
        asyncio.coroutine: wrapped coroutine
    """

    @wraps(coroutine)
    async def with_ds_not_setup_error(*args, **kwargs):
        try:
            return await coroutine(*args, **kwargs)
        except AttributeError as attr_e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=DataServer.NOT_SET_UP,
            ) from attr_e

    return with_ds_not_setup_error


def save_callback():
    """Callback function for when a file has finished saving"""
    logger.debug("File has finished saving in data_router")


@ds_router.post("/save", response_model=SavedPath, status_code=status.HTTP_201_CREATED)
@ds_setup_wrapper
async def save(file: DataFile) -> SavedPath:
    """Save the DataFile 'file' using the data
    server and returns the path to the saved metadata.
    Also collects callback from frontend to notify when saving to GCS is complete

    Args:
        file (DataFile): to be saved

    Raises:
        HTTPException: if the DataServer has not been setup

    Returns:
        Path: the path to the saved metadata
    """
    path, save_task = await DataServer.get_frontend().save(file)
    save_task.add_done_callback(partial(lambda e: save_callback()))
    return SavedPath(path=path)


@ds_router.get("/load/{file_path:path}", response_model=DataFile)
@ds_setup_wrapper
async def load(
    file_path: str,
    background_tasks: BackgroundTasks,
    load_data: bool = True,
    load_path: Path | None = None,
) -> DataFile:
    """Load a DataFile using the metadata path for the datafile

    Args:
        file (str):
            path to metadata on the metadata server used to save the metadata
        load_data (bool, optional):
            if False only the metadata for the file and not the data is loaded.
            Defaults to True.
        load_path (Path, optional):
            path to the target local directory to which files are to be loaded.
            If None, files not loaded. Defaults to None.

    Raises:
        HTTPException: if the DataServer has not been setup

    Returns:
        DataFile: datafile loaded from the metadata path
    """
    try:
        datafile = await DataServer.get_frontend().load(
            Path(file_path), load_data=load_data, load_path=load_path
        )
        if load_data and datafile.data is not None:
            memories = [dataset.memory for dataset in datafile.data.datasets]
            background_tasks.add_task(_monitor_counters, _increase_counters(memories))
        return datafile
    except FileNotFoundError as ferr:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(ferr)) from ferr


@ds_router.post("/load_partial/{file_path:path}", response_model=DataFile)
@ds_setup_wrapper
async def load_partial(
    file_path: str, slice_list: SliceListModel, background_tasks: BackgroundTasks
) -> DataFile:
    """Partially load a DataFile using the metadata path for the datafile and
    the slice_list for portions to load

    Args:
        file (str):
            path to metadata on the metadata server used to save the metadata
        slice_list (SliceListModel):
            SliceListModel object that contains a list of lists of SliceModel objects
            that represent the slices to load

    Raises:
        HTTPException: if the DataServer has not been setup

    Returns:
        DataFile: datafile loaded from the metadata path
    """
    slice_lists = slice_list.to_slice_lists()
    try:
        datafile = await DataServer.get_frontend().load(
            Path(file_path), slice_lists=slice_lists
        )
        if datafile.data is not None:
            memories = [dataset.memory for dataset in datafile.data.datasets]
            background_tasks.add_task(_monitor_counters, _increase_counters(memories))
        return datafile
    except FileNotFoundError as ferr:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(ferr)) from ferr


@ds_router.post("/init", response_model=SavedPath, status_code=status.HTTP_201_CREATED)
@ds_setup_wrapper
async def init_data(file: DataFile) -> SavedPath:
    """Initialises stores for the DataFile 'file' using the data
    server and returns the path the metadata would be saved to

    Args:
        file (DataFile): for which data stores should be initialized

    Raises:
        HTTPException: if the DataServer has not been setup

    Returns:
        Path: the path to the initialized metadata store
    """
    path = await DataServer.get_frontend().init_data(file)
    return SavedPath(path=path)


@ds_router.post("/update/metadata/{file_path:path}", status_code=status.HTTP_200_OK)
@ds_setup_wrapper
async def update_metadata(file_path: str, file: DataFile):
    """Updates the metadata for the DataFile 'file' using the data server

    Args:
        file_path (str): path to the metadata to update
        file (DataFile): metadata to update the metadata on the metadata server to

    Raises:
        HTTPException: if the DataServer has not been setup
    """
    await DataServer.get_frontend().update_metadata(Path(file_path), file)


@ds_router.post("/update/data/", status_code=status.HTTP_202_ACCEPTED)
@ds_setup_wrapper
async def update_data(
    data_and_slices: DataFileAndSliceModel,
    background_tasks: BackgroundTasks,
    callback_url: AnyHttpUrl | None = None,
):
    """Updates the data for the DataFile 'file' using the data server

    Args:
        file_path (str): path to the metadata to update
        file (DataFile): metadata to update the metadata on the metadata server to

    Raises:
        HTTPException: if the DataServer has not been setup
    """
    background_tasks.add_task(_update_data_task, data_and_slices, callback_url)
    return {"message": "data updating"}
    # todo error handling


async def _update_data_task(
    data_and_slices: DataFileAndSliceModel,
    callback_url: AnyHttpUrl | None = None,
):
    """Background task that updates the data for a DataFile"""
    assert data_and_slices.data_file.data is not None
    logger.debug("Updating Data %s", data_and_slices.data_file.data.handle)
    if callback_url:
        logger.debug("Updating Data got callback url: %s", callback_url)
    slice_lists = (
        data_and_slices.slices.to_slice_lists() if data_and_slices.slices else None
    )
    await DataServer.get_frontend().update_data(
        data_and_slices.data_file, slice_lists=slice_lists
    )
    if callback_url:
        async with httpx.AsyncClient(base_url=str(callback_url)) as client:
            await client.post(
                "/data/updated",
                content=data_and_slices.data_file.data.handle.model_dump_json(),
            )


# pylint: disable=unused-argument
@callback_router.post("{$callback_url}/data/updated")
async def data_updated(handle: MeasurementHandle):
    """API that should be implemented by a service, that posts data
    to be updated to the data server, such that it can recieve callback from
    the data server when the data server has updated the data in the datafile

    Args:
        handle (MeasurementHandle):
            A handle to identify which data is being updated
    """
    # this is not executed in the data server app but serves as documentation
    # for how other apps should implement the callback api such that they can recieve
    # a callback. See https://fastapi.tiangolo.com/advanced/openapi-callbacks/ for
    # further info


# pylint: enable=unused-argument


@ds_router.post("/update/{file_path:path}", status_code=status.HTTP_200_OK)
@ds_setup_wrapper
async def update(file_path: str, data_and_slices: DataFileAndSliceModel):
    """Updates the data and metadata for the DataFile 'file' using the data server

    Args:
        file_path (str): path to the metadata to update
        file (DataFile): metadata to update the metadata on the metadata server to

    Raises:
        HTTPException: if the DataServer has not been setup
    """
    slice_lists = (
        data_and_slices.slices.to_slice_lists() if data_and_slices.slices else None
    )
    await DataServer.get_frontend().update(
        Path(file_path), data_and_slices.data_file, slice_lists=slice_lists
    )


@ds_router.post("/memory", response_model=SharedMemoryOut)
async def create_memory(
    memory_in: SharedMemoryIn, background_tasks: BackgroundTasks
) -> SharedMemoryOut:
    """Creates a SharedMemoryOut object from a SharedMemoryIn object

    Args:
        memory_in (SharedMemoryIn):
            a object describing the shape and datatype of an unallocated array

    Returns:
        SharedMemoryOut:
            a object describing the shape and datatype of an allocated array
    """
    memory_out = SharedMemoryOut.from_memory_in(memory_in)
    background_tasks.add_task(_monitor_counters, _increase_counters([memory_out]))
    return memory_out


@ds_router.delete("/memory/{name:path}")
async def delete_memory(name: str):
    """Deletes a SharedMemoryOut object

    Args:
        name (str):
            name of the shared memory of the shared memory out object
    """
    SharedMemoryOut.close(name)


@ds_router.post("/close", status_code=status.HTTP_200_OK)
@ds_setup_wrapper
async def close(file: DataFile):
    """Closes stores for the DataFile 'file' using the data server

    Args:
        file (DataFile): for which data stores should be closed

    Raises:
        HTTPException: if the DataServer has not been setup
    """
    await DataServer.get_frontend().close(file)


@ds_router.post(
    "/stream/start", response_model=SavedPath, status_code=status.HTTP_201_CREATED
)
@ds_setup_wrapper
async def prepare_stream(ssm: StartStreamModel) -> SavedPath:
    """Signals to the data server that is should prepare for streaming

    Args:
        ssm (StartStreamModel): IPC model for signaling stream is starting

    Raises:
        HTTPException (503):
            if the DataServer has not been setup
        HTTPException (409):
            is stream has already been prepared with the supplied handle
        HTTPException (415):
            if the supplied datafile has no data

    Returns:
        Path: the path to the initialized metadata store

    The StartStreamModel has the following attributes
        handle (MeasurementHandle):
            a handle to identify which measurement is being finalized
        file (DataFile):
            The metadata for the DataFile is saved and
            the Data is used to determine the expected buffer sizes
    """
    try:
        path = await DataServer.get_frontend().prepare_stream(ssm.handle, ssm.data_file)
        return SavedPath(path=path)
    except StreamAlreadyPreparedError as serr:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(serr)) from serr
    except ValueError as verr:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="To prepare a stream for a datafile the datafile must have data",
        ) from verr


# pylint: disable=unused-argument
@callback_router.post("{$callback_url}/buffer/saved")
async def buffer_saved(buffer: BufferReference):
    """API that should be implemented by a service, that posts buffers
    to be saved to the data server, such that it can recieve callback from
    the data server when the data server has saved the data in the buffer

    Args:
        buffer (BufferReference):
            A reference to a buffer the data server has saved the data in.
    """
    # this is not executed in the data server app but serves as documentation
    # for how other apps should implement the callback api such that they can recieve
    # a callback. See https://fastapi.tiangolo.com/advanced/openapi-callbacks/ for
    # further info


# pylint: enable=unused-argument


async def save_buffer_task(
    sbm: SaveBufferModel,
    validated: bool = False,
    callback_url: AnyHttpUrl | None = None,
):
    """saves the data in a buffer referenced by the SaveBufferModel,
    if a callback_url is provided then, when the data in the buffer has been saved,
    a callback is performed indicating the data server
    is done with the data in the buffer

    Args:
        sbm (SaveBufferModel):
            IPC model for saving data in buffers
        validated (bool, optional):
            indicates if calling save_buffer using the data in the SaveBufferModel
            has been validated, defaults to False
        callback_url (AnyHTTPUrl, optional):
            url to call to inform that the buffer has been saved,
            defaults to None

    The SaveBufferModel has the following attributes
        handle (MeasurementHandle):
            A handle to identify which measurement (and therefore where) the buffer
            belongs to and where it should be saved.
        name (str):
            name of dataset the buffer belongs to.
        buffer (BufferReference):
            A Reference to the buffer that can be used to correctly extract the
            data from it
        index (tuple):
            the starting index for the data being saved, i.e. if 1000x1000 large
            data is saved in 100x100 size buffers then a buffer with index
            (300, 400) will write data into the [300:400,400:500] slice of the
            final data array.
    """
    logger.debug("Saving buffer %s", SaveBufferModel)
    logger.debug("Saving buffer got callback url: %s", callback_url)
    await DataServer.get_frontend().save_buffer(
        sbm.handle,
        sbm.name,
        sbm.buffer,
        sbm.index,  # type: ignore
        validated=validated,
    )
    if callback_url:
        async with httpx.AsyncClient(base_url=str(callback_url)) as client:
            await client.post("/buffer/saved", content=sbm.buffer.model_dump_json())


@ds_router.post(
    "/save/buffer",
    status_code=status.HTTP_202_ACCEPTED,
    callbacks=callback_router.routes,
)
@ds_setup_wrapper
async def save_buffer(
    sbm: SaveBufferModel,
    background_tasks: BackgroundTasks,
    callback_url: AnyHttpUrl | None = None,
):
    """starts a background task that saves the data in a buffer
    referenced by the SaveBufferModel, if callback_url is provided then,
    when the data in the buffer has been saved, a callback is performed
    indicating the data server is done with the data in the buffer

    Args:
        sbm (SaveBufferModel):
            IPC model for saving data in buffers
        callback_url (AnyHTTPUrl, optional):
            url to call to inform that the buffer has been saved,
            defaults to None

    Raises:
        HTTPException (503):
            if the DataServer has not been setup
        HTTPException (409):
            a) a stream with a handle matching sbm.handle has not been prepared
            b) if a dataset with a name matching sbm.name was not prepared
               for the stream
        HTTPException (415):
            a) Shape of data in the referenced buffer does not match
               the chunksize of the matching dataset
            b) Datatype of the data in the referenced buffer does not match
               the datatype of the matching dataset
            c) Length of index does not match the dimension fo the matcing dataset


    The SaveBufferModel has the following attributes
        handle (MeasurementHandle):
            A handle to identify which measurement (and therefore where) the buffer
            belongs to and where it should be saved.
        name (str):
            name of dataset the buffer belongs to.
        buffer (BufferReference):
            A Reference to the buffer that can be used to correctly extract the
            data from it
        index (tuple):
            the starting index for the data being saved, i.e. if 1000x1000 large
            data is saved in 100x100 size buffers then a buffer with index
            (300, 400) will write data into the [300:400,400:500] slice of the
            final data array.
    """
    try:
        DataServer.get_frontend().validate_save_buffer_call(
            sbm.handle,
            sbm.name,
            sbm.buffer,
            sbm.index,  # type:ignore
        )
        background_tasks.add_task(save_buffer_task, sbm, True, callback_url)
        # Background tasks run in fastapi loop,
        # should not be cpu intensive or they block
        return {"message": "buffer posted"}
    except StreamNotPreparedError as serr:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(serr)) from serr
    except NameError as nerr:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(nerr)) from nerr
    except BufferShapeError as berr:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(berr)
        ) from berr
    except TypeError as terr:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(terr)
        ) from terr
    except ValueError as verr:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(verr)
        ) from verr


@ds_router.post("/stream/end")
@ds_setup_wrapper
async def finalize_stream(esm: EndStreamModel):
    """Signal to the data server that streaming associated with a MeasurementHandle
    is finished. Optionally overwrites the metadata with new values
    from the DataFile 'file' if supplied.

    Args:
        esm (EndStreamModel): IPC model for signaling streaming is finished

    Raises:
        HTTPException (503):
            if the DataServer has not been setup
        HTTPException (409):
            a stream with a handle matching esm.handle has not been prepared

    The EndStreamModel has the following attributes
        handle (MeasurementHandle):
            a handle to identify which measurement is being finalized
        file (DataFile | None, optional):
            If supplied used to overwrites metadata.
            Defaults to None.
    """
    try:
        await DataServer.get_frontend().finalize_stream(esm.handle, esm.data_file)
    except StreamNotPreparedError as serr:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(serr)) from serr


def _increase_counters(memories: list[SharedMemoryOut]) -> dict[str, int]:
    """
    Increment all the counters of the provided memories,

    Called before returning the memories using the DataRouter to the user such that the
    memories are not deallocated before the user has recieved them.

    Args:
        datafile (DataFile): the datafile to increment counters for

    Returns:
        dict[str, int]: a dictionary of the current counter values
    """
    current_counter_values = {}
    for memory in memories:
        memory_name = memory.name
        counter = SharedCounter(name=memory_name)
        counter.increase()
        current_counter_values[memory_name] = counter.value
    return current_counter_values


async def _monitor_counters(current_counter_values: dict[str, int]):
    """
    Monitor counters. If the counters are changed then the counters are decreased,
    allowing this process to no longer keep refencees to the memories and them to be
    closed if all processes have finished with them.

    If the counters are not changed within 10 seconds then the counters are decreased.

    Args:
        current_counter_values (dict[str, int]):
            dictionary of the current counter values, the keys are the memory names
    """

    time_start = time.time()

    while time.time() - time_start < MONITOR_TIMOUT:
        if not current_counter_values:
            return
        await asyncio.sleep(ASYNCIO_SLEEP)
        decreased_counters = []
        for memory_name, value in current_counter_values.items():
            counter = SharedCounter(name=memory_name)
            if counter.value != value:
                counter.decrease()
                decreased_counters.append(memory_name)
        for memory_name in decreased_counters:
            current_counter_values.pop(memory_name)
    for memory_name in current_counter_values:
        SharedCounter(name=memory_name).decrease()
