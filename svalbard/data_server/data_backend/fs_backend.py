"""General fsspec based data backend"""

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import lru_cache, partial
from pathlib import Path, PurePath
from typing import Awaitable, Callable, Literal

import numpy as np
import zarr
from fsspec import AbstractFileSystem
from zarr.errors import GroupNotFoundError

from ...data_model.data_file import Data, MeasurementHandle
from ...data_model.ipc import BufferReference
from ...data_model.memory_models import SharedMemoryIn, SharedMemoryOut
from ...utility.logger import logger
from ...utility.resize_array import new_shape
from ..errors import (
    BufferShapeError,
    DataNotInitializedError,
    StreamAlreadyPreparedError,
    StreamNotPreparedError,
)
from .abstract_data_backend import AbstractDataBackend, AbstractDataBackendConfig

# todo implement background tasks where appropriate
# todo save handle as array attribute instead of relying on path being handle?


class FSBackend(AbstractDataBackend):
    """General fsspec base data backend"""

    def __init__(
        self,
        filesystem: AbstractFileSystem,
        executor: ThreadPoolExecutor | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        path_base: Path = Path("/"),
    ) -> None:
        self._fs = filesystem
        self._executor = executor or ThreadPoolExecutor()
        self._loop = loop or asyncio.get_running_loop()
        self._handles: dict[uuid.UUID, dict[str, Data.DataSet]] = {}
        self._path_base = path_base
        self._logger = logger
        self._logger.setLevel(logging.DEBUG)
        self._groups: dict[uuid.UUID, zarr.Group] = {}

    @property
    def filesystem(self):
        """the filesystem the backend is connected to"""
        return self._fs

    @property
    def executor(self) -> ThreadPoolExecutor:
        """the thread pool executor used for asynchronous z_arrayrr operation"""
        return self._executor

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Event loop to schedule futures from thread pool executor"""
        return self._loop

    @property
    def handles(self) -> dict[uuid.UUID, dict[str, Data.DataSet]]:
        """A dictionary with keys of tracked measurement handles and
        values of dataset names initialised for each measurement handle"""
        return self._handles

    @property
    def logger(self) -> logging.Logger:
        """logger for the data backend"""
        return self._logger

    @property
    def path_base(self) -> Path:
        """returns the base path of the backend"""
        return self._path_base

    def path(self, data: Data) -> Path:
        return self._handle_to_path(data.handle.handle)

    async def init_data(self, data: Data) -> Path:
        self.logger.debug("initializing data at path: %s", self.path(data))
        path = self.path(data)
        group = await self._create_group(path)
        self._groups[data.handle.handle] = group
        create_coros = [
            self._create_dataset(
                group,
                ds.name,
                shape=ds.memory.shape,
                dtype=ds.memory.dtype,
                fill_value=np.NAN if np.dtype(ds.memory.dtype).kind != "i" else 0,
            )
            for ds in data.datasets
        ]
        await asyncio.gather(*create_coros)
        await self._run(zarr.consolidate_metadata, group.store)
        self.logger.debug("initialized data at path: %s", path)
        return path

    async def save(self, data: Data) -> Path:
        path = self.path(data)  # path that data is saved to --> memory:/...
        self.logger.debug("Saving data at path: %s", path)
        group = await self._create_group(path)
        if data.files:
            fcoros = [self.save_file(Path(spath), path) for spath in data.files]
            await asyncio.gather(*fcoros)
        coros = [self._save_memory(group, ds.name, ds.memory) for ds in data.datasets]
        await asyncio.gather(*coros)
        await self._run(zarr.consolidate_metadata, group.store)
        self.logger.debug("Saved data at path: %s", path)
        return path

    async def save_file(self, spath: Path, tpath: Path) -> Path:
        if not spath.exists():
            raise FileNotFoundError(f"save_file - no file found at: {spath}")
        tpath = tpath / "files" / spath.name
        await self._run(
            self.filesystem.put_file, str(spath.as_posix()), str(tpath.as_posix())
        )
        return tpath

    async def load(
        self,
        path: Path,
        load_path: Path | None = None,
        slice_lists: list[list[slice]] | list[None] | None = None,
    ) -> Data:
        self.logger.debug("Loading data at path: %s", path)
        try:
            z_group = await self._open_group(path)
        except (GroupNotFoundError, KeyError) as gerr:
            raise FileNotFoundError(f"no data found at: {path}") from gerr
        arrs = await self._run(z_group.array_keys)
        names = [str(name) for name in arrs]
        slice_lists = slice_lists or [None for _ in names]
        coros = [
            self._load_memory(z_group, name, slices)
            for (name, slices) in zip(names, slice_lists)
        ]
        memories = await asyncio.gather(*coros)
        files = []  # if load_path is None
        if load_path and await self._exists(path / "files"):
            file_paths = await self._ls(path / "files")
            fcoros = [self.load_file(spath, load_path) for spath in file_paths]
            files = await asyncio.gather(*fcoros)
        self.logger.debug("Loaded data at path: %s", path)
        return Data(
            handle=self._path_to_handle(path),
            datasets=[
                Data.DataSet(name=name, memory=mem)
                for name, mem in zip(names, memories)
            ],
            files=files,
        )

    async def update(
        self, data: Data, slice_lists: list[list[slice]] | list[None] | None = None
    ):
        try:
            z_group = self._groups[data.handle.handle]
        except KeyError as kerr:
            raise DataNotInitializedError(
                data.handle, self.__class__.__name__
            ) from kerr
        self.logger.debug("Updating data at path: %s", self.path(data))
        slice_lists = slice_lists or [None for _ in data.datasets]
        update_coros = [
            self._run(self._sync_update_dataset, z_group, ds.name, ds.memory, slices)
            for ds, slices in zip(data.datasets, slice_lists)
        ]
        await asyncio.gather(*update_coros)
        await self._run(zarr.consolidate_metadata, z_group.store)
        self.logger.debug("Updated data at path: %s", self.path(data))

    async def load_file(self, spath: Path, tpath: Path) -> Path:
        if not await self._exists(spath):
            raise FileNotFoundError(f"load_file - no file found at: {spath}")
        if not tpath.exists():  # create directory if does not exist.
            tpath.mkdir()
        await self._run(
            self.filesystem.get_file, spath.as_posix(), (tpath / spath.name).as_posix()
        )
        return tpath / spath.name

    async def close(self, data: Data):
        for dataset in data.datasets:
            logger.debug("close memory: %s", dataset.memory.name)
            SharedMemoryOut.close(dataset.memory.name)
        try:
            del self._groups[data.handle.handle]
        except KeyError:
            pass

    async def prepare_stream(self, data: Data) -> Path:
        if data.handle.handle in self.handles:
            raise StreamAlreadyPreparedError(data.handle, self.__class__.__name__)
        self.logger.debug("Preparing stream with handle: %s", data.handle.handle)
        self.handles[data.handle.handle] = {
            dataset.name: dataset for dataset in data.datasets
        }
        path = self.path(data)
        z_group = await self._create_group(path)
        self._groups[data.handle.handle] = z_group
        coros = [
            self._create_dataset(
                z_group,
                dataset.name,
                shape=dataset.memory.shape,
                chunks=dataset.memory.shape,
                dtype=dataset.memory.dtype,
                cache_metadata=False,
                fill_value=np.NAN,
            )
            for dataset in data.datasets
        ]
        await asyncio.gather(*coros)
        await self._run(zarr.consolidate_metadata, z_group.store)
        return path

    def validate_save_buffer_call(
        self,
        handle: MeasurementHandle,
        name: str,
        buffer: BufferReference,
        index: tuple,
    ) -> bool:
        self.logger.debug("Verifying saving buffer call:")
        if handle.handle not in self.handles:
            raise StreamNotPreparedError(handle, self.__class__.__name__)
        if name not in self.handles[handle.handle]:
            raise NameError(
                f"{self.__class__.__name__}: name '{name}':"
                + f"not in names initialised for handle {handle.handle}, "
                + f"initialized names {self.handles[handle.handle]}"
            )
        if buffer.shape != self.handles[handle.handle][name].memory.shape:
            raise BufferShapeError(
                f"{self.__class__.__name__}: name '{name}':"
                + f"Shape of data referenced by buffer reference {buffer.shape} "
                + "does not match the chunk size that was created for the dataset"
                + f" {self.handles[handle.handle][name].memory.shape}"
            )
        if buffer.dtype != self.handles[handle.handle][name].memory.dtype:
            raise TypeError(
                f"{self.__class__.__name__}: name '{name}:'"
                + f"datatype referenced by buffer reference {buffer.dtype} "
                + "does not match the datatype of the the dataset"
                + f" {self.handles[handle.handle][name].memory.dtype}"
            )
        if len(index) != len(buffer.shape):
            raise ValueError(
                f"{self.__class__.__name__}: name '{name}:': index ({len(index)})"
                + f" and buffer ({len(buffer.shape)}) dimensions don't match:"
            )
        self.logger.debug("Save buffer call verified")
        return True

    async def save_buffer(  # pylint: disable=too-many-arguments
        self,
        handle: MeasurementHandle,
        name: str,
        buffer: BufferReference,
        index: tuple,
        validated: bool = False,
    ):
        if not validated:
            self.validate_save_buffer_call(handle, name, buffer, index)
        self.logger.debug(
            "Saving buffer for handle: %s, dataset: %s, index %s",
            handle.handle,
            name,
            index,
        )
        z_group = self._groups[handle.handle]
        await self._add_data(z_group, name, buffer, index)
        await self._run(zarr.consolidate_metadata, z_group.store)

    async def finalize_stream(self, handle: MeasurementHandle):
        try:
            del self.handles[handle.handle]
            del self._groups[handle.handle]
        except KeyError as kerr:
            raise StreamNotPreparedError(handle, self.__class__.__name__) from kerr

    async def _create_group(self, path: PurePath) -> zarr.Group:
        """Wrapper for creating a group in a thread pool executor

        Args:
            path (Path): location of group in filesystem

        Returns:
            zarr.Group: a zarr group located at path
        """
        store = self.filesystem.get_mapper(str(path))
        return await self._run(
            partial(zarr.group, synchronizer=zarr.ThreadSynchronizer()), store
        )

    async def _open_group(self, path: PurePath) -> zarr.Group:
        """Wrapper toopen a group that already exists using a thread pool executor

        Args:
            path (Path): location of group in filesystem

        Returns:
            zarr.Group: a zarr group located at path
        """
        store = self.filesystem.get_mapper(str(path))
        return await self._run(  # type: ignore
            partial(
                zarr.open_consolidated,
                mode="r+",
                synchronizer=zarr.ThreadSynchronizer(),
            ),
            store,
        )

    async def _create_dataset(
        self,
        z_group: zarr.Group,
        name: str,
        **kwargs,
    ) -> zarr.Array:
        """Wrapper for creating a dataset in a thread pool executor

        Args:
            z_group (zarr.Group): group to create dataset in
            name (str): name of dataset (key in z_group) to create
            kwargs (dict): kwargs for zarr.Group.create_dataset

        Returns:
            zarr.Array: the created dataset (z_group[name])
        """
        return await self._run(partial(z_group.create_dataset, **kwargs), name)

    async def _save_memory(self, group: zarr.Group, name: str, memory: SharedMemoryOut):
        """
        Wrapper for saving a memory to a dataset in a thread pool executor

        Args:
            group (zarr.Group): zarr group to save dataset to
            name (str): name of dataset to save to
            memory (SharedMemoryOut): memory to save to dataset
        """

        def _add_data(memory: SharedMemoryOut):
            group.create_dataset(name, data=memory.to_array())

        return await self._run(_add_data, memory)

    def _sync_update_dataset(
        self,
        z_group: zarr.Group,
        name: str,
        memory: SharedMemoryOut,
        slice_list: list[slice] | None = None,
    ):
        """
        Updates a dataset synchronously. This functions locks the dataset before
        updating it to prevent concurrent updating of the same dataset

        Args:
            z_group (zarr.Group): the zarr group to update the dataset in
            name (str): name of dataset to update
            memory (SharedMemoryOut): memory to update dataset with
            slice_list (list[slice] | None, optional):
                list of slices to load from dataset one for each dimensions of the
                dataset, defaults to None, which loads the entire dataset

        Returns:
            _type_: _description_
        """
        z_array = z_group[name]
        z_shape = z_array.shape
        lock = z_array.synchronizer[name]  # type: ignore
        with lock:
            n_shape = new_shape(
                z_shape,  # type: ignore
                slice_list or [slice(size) for size in z_shape],
                z_shape,  # type: ignore
            )
            if n_shape != z_shape:
                self.logger.debug(
                    "update data: rezising array from %s to %s", z_shape, n_shape
                )
                z_array.resize(n_shape)  # type: ignore

            shape = self._extract_shape(slice_list, n_shape) if slice_list else n_shape
            selection = slice_list or [slice(None) for _ in n_shape]
            selection = tuple(selection)

            # todo maybe dedent this
            if shape != memory.shape:
                z_array[selection] = memory.to_array()[selection]
            else:
                z_array[selection] = memory.to_array()[:]

    async def _load_memory(
        self, z_group: zarr.Group, name: str, slice_list: list[slice] | None = None
    ) -> SharedMemoryOut:
        """
        Wrapper for loading a dataset to a memory in a thread pool executor

        Args:
            z_group (zarr.Group): zarr group to load dataset from
            name (str): name of dataset to load from
            slice_list (list[slice] | None, Optional):
                list of slices to load from dataset one for each dimensions of the
                dataset, defaults to None, which loads the entire dataset

        Returns:
            SharedMemoryOut: Loaded memory
        """
        z_array: zarr.Array = await self._run(z_group.__getitem__, name)
        z_shape = z_array.shape
        shape_slices = (
            self._extract_shape(slice_list, z_shape) if slice_list else z_shape
        )
        smi = SharedMemoryIn(dtype=z_array.dtype, shape=shape_slices)  # type: ignore
        smo = SharedMemoryOut.from_memory_in(smi)

        def _assign_values(smo: SharedMemoryOut, z_array: zarr.Array):
            selection = slice_list or [slice(None) for _ in z_shape]
            smo.to_array()[:] = z_array.get_orthogonal_selection(tuple(selection))

        await self._run(_assign_values, smo, z_array)
        return smo

    @lru_cache(maxsize=128)  # saves up to maxsize recent calls
    def _handle_to_path(self, handle: uuid.UUID) -> Path:
        """return a save path for a handle (uuid),
        as uuid is hashable this function can be cached

        Args:
            handle (uuid.UUID): the handle to determine a save path for

        Returns:
            Path: the save path for the handle
        """
        return self.path_base / str(handle)

    @staticmethod
    @lru_cache(maxsize=128)
    def _path_to_handle(path: Path) -> MeasurementHandle:
        """return a handle (uuid) for a path,

        Args:
            path (Path): the save path to determine a handle for

        Returns:
            MeasurementHandle: the handle for the path
        """
        return MeasurementHandle(handle=uuid.UUID(path.name))

    def _sync_add_data(
        self, z_group: zarr.Group, name: str, buffer: BufferReference, index: tuple
    ):
        """
        Adds data to a dataset synchronously. This functions locks the dataset
        before adding the data to prevent concurrent addition of data to the same
        dataset

        Args:
            z_group (zarr.Group): _description_
            name (str): _description_
            buffer (BufferReference): _description_
            index (tuple): _description_
        """
        z_array = z_group[name]
        new_data_slice = tuple(
            slice(idx, idx + buffer.shape[i]) for i, idx in enumerate(index)
        )

        lock = z_array.synchronizer[name]  # type: ignore
        with lock:
            self.logger.debug("streaming: locked with %s", lock)
            z_shape = z_array.shape
            n_shape = new_shape(z_shape, new_data_slice, z_array.chunks)  # type: ignore
            if n_shape != z_shape:
                self.logger.debug(
                    "streaming: rezising array from %s to %s", z_shape, n_shape
                )
                z_array.resize(n_shape)  # type: ignore
            self.logger.debug("streaming: insertion index: %s", index)
            z_array[new_data_slice] = buffer.to_array()[:]
            self.logger.debug("streaming: exiting lock, array shape: %s", z_array.shape)

    async def _add_data(
        self, z_group: zarr.Group, name: str, buffer: BufferReference, index: tuple
    ):
        """Wrapper for adding data to a dataset in a thread pool executor from a buffer

        Args:
            z_group (zarr.Group): group to add data to
            name (str): name of dataset to add data to
            buffer (BufferReference): buffer to add data from
            index (tuple): starting index for data to be added at
        """
        return await self._run(self._sync_add_data, z_group, name, buffer, index)

    async def _exists(self, path: PurePath) -> bool:
        """Returns True if path exists in filesystem

        Args:
            path (PurePath): path to check

        Returns:
            bool: true if path exists, false otherwise
        """
        return self.filesystem.exists(path.as_posix())

    async def _ls(self, path: PurePath) -> list[Path]:
        """Returns a list of files and directories in path

        Args:
            path (PurePath): path to list

        Returns:
            list[str]: list of files and directories in path
        """
        files = self.filesystem.ls(path.as_posix(), detail=False)
        return [Path(file) for file in files]

    async def _run(self, method: Awaitable | Callable, *args):
        """
        wrapper for running a method in the thread pool executor

        Args:
            method (Awaitable | Callable): async or sync method to run

        Returns:
            Awaitable: future for the method runnning in the thread pool executor
        """
        task = self.loop.run_in_executor(self.executor, method, *args)  # type: ignore
        return await task

    @staticmethod
    def _extract_shape(
        slice_list: list[slice], default: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Determines the shape of a slice list

        Args:
            slice_list (list[slice]): list of slices, one for each dimension of the data

        Returns:
            tuple[int, ...]: the shape of the slice list
        """
        shape = [0] * len(slice_list)
        for i, s in enumerate(slice_list):
            if s.stop is None:
                s = slice(s.start, default[i], s.step)
            if s.start is None:
                s = slice(0, s.stop, s.step)
            if s.step is not None:
                shape[i] = divmod((s.stop - s.start), s.step)[0]
                if shape[i] * s.step + s.start < s.stop:
                    shape[i] += 1  # ! why?
            else:
                shape[i] = s.stop - s.start
        return tuple(shape)


class Scopes(Enum):
    """Enumeration of GCS access scopes"""

    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    FULL_CONTROL = "full_control"


class FSBackendConfig(AbstractDataBackendConfig):
    """Configuration for fsspec FileSystem Based Data Backend"""

    # todo fix model loading instead of hacking all parameters gcs config in here

    cls: str = "fsspec.implementations.memory.MemoryFileSystem"
    protocol: Literal["memory", "local"] = "memory"
    args: list = []
    path: Path = Path("/")
    project: str = ""  #
    access: Scopes = Scopes.READ_ONLY
    token: str | dict | None = None
    endpoint_url: str | None = None
    timeout: float | None = None

    def init(self) -> FSBackend:
        return FSBackend(self.filesystem(), path_base=self.path)

    def filesystem(self) -> AbstractFileSystem:
        """Creates a filesystem from the configuration"""
        return AbstractFileSystem.from_json(self.model_dump_json())
