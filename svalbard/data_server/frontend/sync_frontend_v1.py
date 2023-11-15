"""Synchronous frontend for the data server."""
import asyncio
from pathlib import Path

from fsspec.asyn import get_loop, sync_wrapper

from ...data_model.data_file import DataFile, MeasurementHandle
from ...data_model.ipc import BufferReference
from .frontend_v1 import (
    AbstractDataBackend,
    AbstractMetadataBackend,
    FrontendV1,
    FrontendV1Config,
)


class SyncFrontendV1(FrontendV1):
    def __init__(
        self,
        data_backend: AbstractDataBackend,
        metadata_backend: AbstractMetadataBackend,
    ) -> None:
        super().__init__(data_backend, metadata_backend)
        self._loop = get_loop()

    @property
    def loop(self):
        return self._loop

    @sync_wrapper
    async def save(self, datafile: DataFile) -> tuple[Path, asyncio.Task | None]:
        return await super().save(datafile)

    @sync_wrapper
    async def load(
        self,
        path: Path,
        load_data: bool = True,
        load_path: Path | None = None,
        slice_lists: list[list[slice]] | None = None,
    ) -> DataFile:
        return await super().load(path, load_data, load_path, slice_lists)

    @sync_wrapper
    async def init_data(self, datafile: DataFile) -> Path:
        path = await super().init_data(datafile)
        await super().complete_background_tasks()
        return path

    @sync_wrapper
    async def update_metadata(self, path: Path, data_file: DataFile):
        return await super().update_metadata(path, data_file)

    @sync_wrapper
    async def update_data(
        self, data_file: DataFile, slice_lists: list[list[slice]] | None = None
    ):
        return await super().update_data(data_file, slice_lists)

    def update(
        self,
        path: Path,
        data_file: DataFile,
        slice_lists: list[list[slice]] | None = None,
    ):
        self.update_data(data_file, slice_lists)  # type: ignore
        self.update_metadata(path, data_file)  # type: ignore

    @sync_wrapper
    async def close(self, datafile: DataFile):
        return await super().close(datafile)

    @sync_wrapper
    async def prepare_stream(
        self, handle: MeasurementHandle, datafile: DataFile
    ) -> Path:
        return await super().prepare_stream(handle, datafile)

    @sync_wrapper
    async def save_buffer(
        self,
        handle: MeasurementHandle,
        name: str,
        buffer: BufferReference,
        index: tuple,
        validated: bool = False,
    ):
        return await super().save_buffer(handle, name, buffer, index, validated)

    @sync_wrapper
    async def finalize_stream(
        self, handle: MeasurementHandle, data_file: DataFile | None = None
    ):
        return await super().finalize_stream(handle, data_file)

    @sync_wrapper
    async def complete_background_tasks(self):
        return await super().complete_background_tasks()


class SyncFrontendV1Config(FrontendV1Config):
    def init(self) -> SyncFrontendV1:
        return SyncFrontendV1(self.data_backend.init(), self.metadata_backend.init())
