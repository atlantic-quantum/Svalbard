"""Frontend for DataServer"""
import asyncio
import logging
import uuid
from functools import partial
from pathlib import Path
from typing import Coroutine

from pydantic import validator

from ...data_model.data_file import DataFile, MeasurementHandle
from ...data_model.ipc import BufferReference
from ...utility.logger import logger
from ..data_backend.abstract_data_backend import AbstractDataBackend
from ..data_backend.fs_backend import FSBackendConfig
from ..data_backend.gcs_backend import GCSBackendConfig
from ..errors import StreamAlreadyPreparedError, StreamNotPreparedError
from ..metadata_backend.abstract_metadata_backend import AbstractMetadataBackend
from ..metadata_backend.mongodb_backend import MongoDBConfig
from .abstract_frontend import AbstractFrontend, AbstractFrontendConfig


class FrontendV1(AbstractFrontend):
    """Frontend for DataServer"""

    def __init__(
        self,
        data_backend: AbstractDataBackend,
        metadata_backend: AbstractMetadataBackend,
    ) -> None:
        """Create a frontend"""
        self._data_backend = data_backend
        self._metadata_backend = metadata_backend
        self._handles: set[uuid.UUID] = set()
        self._background_tasks: set[asyncio.Task] = set()
        self._logger = logger  # logging.getLogger("Frontend")
        self._logger.setLevel(logging.DEBUG)
        self._tracked_tasks: dict[uuid.UUID, asyncio.Task] = {}

    @property
    def data_backend(self):
        return self._data_backend

    @property
    def metadata_backend(self):
        return self._metadata_backend

    @property
    def handles(self):
        """A set of MeasurmentHandles tracked by the FrontEnd"""
        return self._handles

    @property
    def logger(self):
        """A logger for the frontend"""
        return self._logger

    @property
    def background_tasks(self):
        """A set of asyncio tasks, used to keep references to the tasks
        so they don't get garbage collected until completed"""
        return self._background_tasks

    @property
    def tracked_tasks(self):
        """A dictionary of tasks tracked by the frontend in order to ensure that
        certain tasks are completed before others"""
        return self._tracked_tasks

    def save_callback(self):
        self.logger.debug("File has finished saving")
        return

    async def save(self, datafile: DataFile) -> tuple[Path, asyncio.Task | None]:
        """Save a datafile to the backend and return the metadata path and save_task.
        Also notify the frontend that the file has finished saving with task callback"""
        if datafile.data:
            datafile.metadata.data_path = self.data_backend.path(datafile.data)
            datafile.metadata.files = [Path(f.name) for f in datafile.data.files]
            save_task = self._add_background_task(self.data_backend.save(datafile.data))
            self._track_task(datafile.data.handle, save_task)
            save_task.add_done_callback(partial(lambda e: self.save_callback()))
        else:
            save_task = None
        self.logger.debug("entering save metadata")
        return await self.metadata_backend.save(datafile.metadata), save_task

    async def load(
        self,
        path: Path,
        load_data: bool = True,
        load_path: Path | None = None,
        slice_lists: list[list[slice]] | None = None,
    ) -> DataFile:
        metadata = await self.metadata_backend.load(path)
        if (
            load_data
            and metadata.data_path
            and uuid.UUID(metadata.data_path.name) in self.tracked_tasks
        ):
            await self.tracked_tasks[uuid.UUID(metadata.data_path.name)]
        data = (
            await self.data_backend.load(metadata.data_path, load_path, slice_lists)
            if metadata.data_path and load_data
            else None
        )
        return DataFile(data=data, metadata=metadata)  # type: ignore

    async def init_data(self, datafile: DataFile) -> Path:
        if datafile.data:
            task = self._add_background_task(self.data_backend.init_data(datafile.data))
            self._track_task(datafile.data.handle, task)
        return await self.metadata_backend.init_data(datafile.metadata)

    async def update_metadata(self, path: Path, data_file: DataFile):
        if data_file.metadata.data_path is None and data_file.data is not None:
            data_file.metadata.data_path = self.data_backend.path(data_file.data)
        await self.metadata_backend.update(path, data_file.metadata)

    async def update_data(
        self, data_file: DataFile, slice_lists: list[list[slice]] | None = None
    ):
        if data_file.data:
            if data_file.data.handle.handle in self.tracked_tasks:
                await self.tracked_tasks[data_file.data.handle.handle]
            task = self._add_background_task(
                self.data_backend.update(data_file.data, slice_lists)
            )
            task.add_done_callback(partial(lambda e: self.save_callback()))
            return task

    async def update(
        self,
        path: Path,
        data_file: DataFile,
        slice_lists: list[list[slice]] | None = None,
    ):
        self._add_background_task(self.update_data(data_file, slice_lists))
        await self.update_metadata(path, data_file)

    async def close(self, datafile: DataFile):
        if datafile.data:
            if datafile.data.handle.handle in self.tracked_tasks:
                await self.tracked_tasks[datafile.data.handle.handle]
            await self.data_backend.close(datafile.data)
        # commented out as no metadata backend currently requries closing metadata
        # await self.metadata_backend.close(datafile.metadata)

    async def prepare_stream(
        self, handle: MeasurementHandle, datafile: DataFile
    ) -> Path:
        if handle.handle in self.handles:
            raise StreamAlreadyPreparedError(handle, self.__class__.__name__)
        if datafile.data is None:
            raise ValueError(
                f"{self.__class__.__name__}: Stream can't"
                + " be prepared for a datafile that has no data"
            )
        self.handles.add(handle.handle)
        self.logger.debug("Preparing stream with handle: %s", handle.handle)
        datafile.metadata.data_path = self.data_backend.path(datafile.data)
        await self._add_background_task(self.data_backend.prepare_stream(datafile.data))
        return await self.metadata_backend.prepare_stream(handle, datafile.metadata)

    def validate_save_buffer_call(
        self,
        handle: MeasurementHandle,
        name: str,
        buffer: BufferReference,
        index: tuple,
    ):
        if handle.handle not in self.handles:
            raise StreamNotPreparedError(handle, self.__class__.__name__)
        self.data_backend.validate_save_buffer_call(handle, name, buffer, index)

    async def save_buffer(
        self,
        handle: MeasurementHandle,
        name: str,
        buffer: BufferReference,
        index: tuple,
        validated: bool = False,
    ):
        self.logger.debug("Saving Buffer for handle: %s", handle.handle)
        self.logger.debug("Tracked Handles: %s", self.handles)
        if not validated:
            self.validate_save_buffer_call(handle, name, buffer, index)
        await self._add_background_task(
            self.data_backend.save_buffer(handle, name, buffer, index, validated=True)
        )

    async def finalize_stream(
        self, handle: MeasurementHandle, data_file: DataFile | None = None
    ):
        if handle.handle not in self.handles:
            raise StreamNotPreparedError(handle, self.__class__.__name__)
        self.logger.debug("Finalizing stream with handle: %s", handle.handle)
        data_finalize_stream_task = self._add_background_task(
            self.data_backend.finalize_stream(handle)
        )
        metadata_finalize_stream_task = self._add_background_task(
            self.metadata_backend.finalize_stream(
                handle, data_file.metadata if data_file else None
            )
        )
        await data_finalize_stream_task
        await metadata_finalize_stream_task
        self.handles.discard(handle.handle)

    def _add_background_task(self, coroutine: Coroutine) -> asyncio.Task:
        """Creates a background task and adds it to the background_tasks set
        when the task is completed, it is removed from the set

        Args:
            coroutine (Coroutine):
                coroutine to create a task from and run in the background

        Returns:
            asyncio.Task: background task created from coroutine
        """
        task = asyncio.create_task(coroutine)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    def _track_task(self, handle: MeasurementHandle, task: asyncio.Task):
        """Tracks a task by handle, so that it can be waited on later to ensure
        that the task is completed before other tasks that may depend on it are
        started

        Args:
            handle (MeasurementHandle): unique handle for the Data being tracked
            task (asyncio.Task): task to track

        Raises:
            ValueError:
                if a task is already being tracked by the handle
        """
        if handle.handle in self.tracked_tasks:
            raise ValueError(
                f"Task for handle {handle.handle} already being tracked"
                + f" by {self.tracked_tasks[handle.handle]}"
            )
        self.tracked_tasks[handle.handle] = task

        def remove_task(task):
            self.tracked_tasks.pop(handle.handle)

        task.add_done_callback(remove_task)

    async def complete_background_tasks(self):
        self.logger.debug("Completing background tasks")
        tasks = list(self.background_tasks)
        for task in tasks:
            await task


class FrontendV1Config(AbstractFrontendConfig):
    """Configuration for Frontend V1"""

    data_backend: FSBackendConfig | GCSBackendConfig
    metadata_backend: MongoDBConfig

    # todo use field discriminators instead of this hack
    # https://docs.pydantic.dev/usage/types/#discriminated-unions-aka-tagged-unions
    @validator("data_backend")
    def cast_data_backend(cls, data_backend: FSBackendConfig | GCSBackendConfig):
        if data_backend.protocol == "gcs":
            return GCSBackendConfig(**data_backend.dict())
        return data_backend

    def init(self) -> FrontendV1:
        return FrontendV1(self.data_backend.init(), self.metadata_backend.init())
