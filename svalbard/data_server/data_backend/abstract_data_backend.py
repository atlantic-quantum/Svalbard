"""
Abstract base class that defines the Data Backend interface
"""
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from ...data_model.data_file import Data, MeasurementHandle
from ...data_model.ipc import BufferReference


class AbstractDataBackend(ABC):
    """
    Abstract base class that defines the Data Backend interface
    """

    @abstractmethod
    def path(self, data: Data) -> Path:
        """Returns a path that data would be saved to.
        meant to be a faster method for getting the path than e.g. saving the data

        Args:
            data (Data): data to get path for

        Returns:
            Path: the path the data would be saved to
        """

    @abstractmethod
    async def save(self, data: Data) -> Path:
        """Saves data and returns the path the data is saved to

        Args:
            data (Data): the data to be saved

        Returns:
            Path: the path the data was saved to
        """

    @abstractmethod
    async def save_file(self, spath: Path, tpath: Path) -> Path:
        """Saves a local file into the data_backend storage and returns
        the path to the file

        Args:
            spath (Path): the source path, to an arbitrary file in local storage
            tpath (Path): the target directory path, in data_backend storage
                Note: files are then stored in a "/files" subfolder in this target dir

        Raises:
            FileNotFoundError:
                if no file is found at spath

        Returns:
            tpath (Path): the path to the file  in the fs
        """

    @abstractmethod
    async def load(
        self,
        path: Path,
        load_path: Path | None = None,
        slice_lists: list[list[slice]] | None = None,
    ) -> Data:
        """Load data from path

        Args:
            path (Path): the path to the data to be loaded
            load_path (Path, optional),
                if not None loads all files in data to this path, defaults to None.
            slice_lists (list):
                list of lists (one for each dimension of each array)
                of slices to load, defaults to None, which loads the whole array
        Raises:
            FileNotFoundError:
                if no data is found at path

        Returns:
            AbstractData: the loaded data
        """

    @abstractmethod
    async def load_file(self, spath: Path, tpath: Path) -> Path:
        """Loads a file from the data_backend storage into local storage
        and returns the path to the file.

        Args:
            spath (Path): the source path, in data_backend storage
            tpath (Path): the target directory path, to a local directory
                Note: files are then stored in a "/files" subfolder in this target dir

        Raises:
            FileNotFoundError:
                if no file is found with fname in data_backend storage

        Returns:
            tpath (Path): the path to the file locally
        """

    @abstractmethod
    async def init_data(self, data: Data) -> Path:
        """Any operations to initialise storage for data before saving it

        Args:
            data (Data): the data to initilise storage for

        Returns:
            Path: The path to location initialised for data.
        """

    @abstractmethod
    async def update(self, data: Data, slice_lists: list[list[slice]] | None = None):
        """Updates data in storage with new data

        Args:
            data (Data): data to be updated
            slice_lists (list[list[slice]] | None, optional):
                list of lists (one for each dimension of an array) to save.
                Defaults to None, which saves the whole array.
        """

    async def close(self, data: Data):
        """Any operation to close data storage once it's been used

        Args:
            data (Data): the data storage to close
        """

    @abstractmethod
    async def prepare_stream(self, data: Data) -> Path:
        """Prepares for streaming data by initializing data stores.

        Args:
            data (Data):
                a structure containign information about the names of
                the datasets to be streamed, their datatypes and buffer sizes

        Raises:
            StreamAlreadyPreparedError:
                If attempting to prepare a stream using a handle that a stream
                has already been prepared for.

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
            StreamNotPreparedError:
                If attempting to save a buffer using 'handle' that a stream
                has not been prepared for.
            NameError:
                If 'name' is not among the dataset names the stream was prepared for
            BufferShapeError:
                If the shape of the referenced buffer does not match the chuncksize
                of the dataset that the stream was prepared for
            TypeError:
                If the datatype of the refrenced buffer does not match the datatype
                of the dataset that the stream was prepared for
            ValueError:
                If the length of the index does not match the dimension of the dataset
                that the stream was prepared for
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
                validated, defaults to Fals

        Raises:
            StreamNotPreparedError:
                If attempting to save a buffer using 'handle' that a stream
                has not been prepared for.
            NameError:
                If 'name' is not among the dataset names the stream was prepared for
            BufferShapeError:
                If the shape of the referenced buffer does not match the chuncksize
                of the dataset that the stream was prepared for
            TypeError:
                If the datatype of the refrenced buffer does not match the datatype
                of the dataset that the stream was prepared for
            ValueError:
                If the length of the index does not match the dimension of the dataset
                that the stream was prepared for
        """

    @abstractmethod
    async def finalize_stream(self, handle: MeasurementHandle):
        """Any code required to finalize a stream is exectued here

        Args:
            handle (MeasurementHandle):
                A handle to identify which measurement is being finalized

        Raises:
            StreamNotPreparedError:
                If attempting to save a buffer using 'handle' that a stream
                has not been prepared for.
        """


class AbstractDataBackendConfig(ABC, BaseModel):
    """configuration for DataBackend"""

    @abstractmethod
    def init(self) -> AbstractDataBackend:
        """Initialize data backend from configuration"""
