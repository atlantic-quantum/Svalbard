"""
Abstract base class that defines the Front end interface.
"""
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from ...data_model.data_file import DataFile, MeasurementHandle
from ...data_model.ipc import BufferReference
from ..data_backend.abstract_data_backend import AbstractDataBackendConfig
from ..metadata_backend.abstract_metadata_backend import AbstractMetadataBackendConfig


class AbstractFrontend(ABC):
    """
    Abstract base class that defines the Front end interface.
    """

    @property
    @abstractmethod
    def data_backend(self):
        """data backend for the DataServer"""

    @property
    @abstractmethod
    def metadata_backend(self):
        """metadata backend for the DataServer"""

    @abstractmethod
    async def save(self, datafile: DataFile) -> Path:
        """Saves metadata and returns the location it was saved to

        Args:
            datafile (DataFile):
                the DataFile to be saved, including data and metadata

        Returns:
            Path: path to saved metadata
        """

    @abstractmethod
    async def load(
        self,
        path: Path,
        load_data: bool = True,
        load_path: Path | None = None,
        slice_lists: list[list[slice]] | None = None,
    ) -> DataFile:
        """Loads datafile from path

        Args:
            path (Path): path to metadata of datafile
            load_data (bool, optional),
                if false only the metadata is loaded, defaults to true
            load_path (Path, optional),
                path to which files are to be loaded, defaults to None
            slice_lists (list): list of lists (for each dimension for each array)
                of slices to load, defaults to None, which loads the entire data

        Raises:
            FileNotFoundError:
                If no metadata is found at path or
                (if load_data=True) no data is found at datapath
        # todo should we have DataNotFoundError and MetadataNotFoundErrors?

        Returns:
            DataFile: loaded datafile from path
        """

    @abstractmethod
    async def init_data(self, datafile: DataFile) -> Path:
        """Any operations to initialise storage for datafile before saving it

        Args:
            datafile (DataFile): the datafile to initilise storage for

        Returns:
            Path: The path to location initialised for DataFile storage.
        """

    @abstractmethod
    async def update_metadata(self, path: Path, metadata: DataFile):
        """Updates metadata at path with new metadata

        Args:
            path (Path): path to metadata to be updated
            metadata (BaseMetaData): new metadata

        Raises:
            FileNotFoundError: if no metadata is found at path
        """

    @abstractmethod
    async def update_data(
        self, data: DataFile, slice_lists: list[list[slice]] | None = None
    ):
        """Updates data at path with new data

        Args:
            data (DataFile): new data

        Raises:
            FileNotFoundError: if no data is found at path
        """

    @abstractmethod
    async def update(
        self,
        path: Path,
        datafile: DataFile,
        slice_lists: list[list[slice]] | None = None,
    ):
        """Updates both metadata at path and data with new datafile

        Args:
            path (Path): path to datafile to be updated
            datafile (DataFile): new datafile

        Raises:
            FileNotFoundError: if no datafile is found at path
        """

    @abstractmethod
    async def close(self, datafile: DataFile):
        """Any operation to close a storage assocated with a datafile
        once it's been used

        Args:
            datafile (DataFile): the Datafile to close
        """

    @abstractmethod
    async def prepare_stream(
        self, handle: MeasurementHandle, datafile: DataFile
    ) -> Path:
        """Prepares for streaming data by initializing data stores in the backends.

        Args:
            handle (MeasurementHandle):
                A handle to identify the measurement a stream is being prepared for,
                this handle should be used when saveing buffers to the stream
            datafile (DataFile):
                A datafile containing the metadata for the data to be streamed
                and information about the names of the datasets to be streamed,
                their datatypes and buffer sizes

        Raises:
            StreamAlreadyPreparedError:
                If attempting to prepare a stream using a handle that a stream
                has already been prepared for.
            ValueError:
                If attempting to prepare a stream with a datafile with
                datafile.data = None

        Returns:
            Path: The path to the metadata saved for the stream.
        """

    @abstractmethod
    def validate_save_buffer_call(
        self,
        handle: MeasurementHandle,
        name: str,
        buffer: BufferReference,
        index: tuple,
    ):
        """Validates calling the save_buffer function with these arguments.
        This function should call the validate_save_buffer_call function of the
        data backend.
        Note that this function is synchronous

        Args:
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

        Raises:
            StreamNotPreparedError: if a stream has not been prepared for the handle
        """

    @abstractmethod
    async def save_buffer(
        self,
        handle: MeasurementHandle,
        name: str,
        buffer: BufferReference,
        index: tuple,
        validated: bool = False,
    ):
        """saves the data pointed to by the BufferReference where data assoicated
        with the MeasurementHandle is saved and as instructed by the index tuple.
        Once the data is saved the backend will notify the frontend using a callback

        Args:
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
            validated (bool, optional):
                Indicates that the other arguments in the function call have been
                validated, defaults to False

        Raises:
            StreamNotPreparedError: if a stream has not been prepared for the handle
        """

    @abstractmethod
    async def finalize_stream(
        self, handle: MeasurementHandle, data_file: DataFile | None = None
    ):
        """finalizes the streaming process,
        optionally overwrites the metadata from the datafile

        Args:
            handle (MeasurementHandle):
                a handle to identify which measurement is being finalized
            data_file (DataFile | None, optional):
                If supplied used to overwrites metadata.
                Defaults to None.

        Raises:
            StreamNotPreparedError:
                a) if a stream has not been prepared for the handle
                b) if a stream has already been finalized
        """


class AbstractFrontendConfig(ABC, BaseModel):
    """configuration for Frontend"""

    data_backend: AbstractDataBackendConfig
    metadata_backend: AbstractMetadataBackendConfig

    @abstractmethod
    def init(self) -> AbstractFrontend:
        """Initialise a frontend from the configuration"""
