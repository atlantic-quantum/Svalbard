"""
Abstract base class that defines the Metadata Backend interface
"""

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from ...data_model.data_file import BaseMetaData, MeasurementHandle


class AbstractMetadataBackend(ABC):
    """
    Abstract base class that defines the Metadata Backend interface
    """

    @abstractmethod
    async def save(self, metadata: BaseMetaData) -> Path:
        """Saves metadata and returns the location it was saved to

        Args:
            metadata (BaseMetaData): the metadata to be saved

        Returns:
            Path: path to saved metadata
        """

    @abstractmethod
    async def update(self, path: Path, metadata: BaseMetaData):
        """Updated metadata at path too new values (overwrites)

        Args:
            path (Path): the path to the metadata to update
            metadata (BaseMetaData): the new metadata, overwrites the old metadata
        """

    @abstractmethod
    async def init_data(self, metadata: BaseMetaData) -> Path:
        """Any operations to initialise storage for metadata before saving it

        Args:
            metadata (BaseMetaData): the metadata to initilise storage for

        Returns:
            Path: The path to location initialised for metadata.
        """

    # commented out as no current storage requires closing metadata
    # @abstractmethod
    # async def close(self, metadata: MetaData):
    #     """Any operations to close a storage for metadata once it's been used

    #     Args:
    #         metadata (MetaData): the metadata to close
    #     """

    @abstractmethod
    async def load(self, path: Path) -> BaseMetaData:
        """Loads metadata from path

        Args:
            path (Path): path to metadata

        Raises:
            FileNotFoundError:
                If attempting to load metadata at a path that does not exist

        Returns:
            BaseMetaData: loaded metadata from path
        """

    @abstractmethod
    async def prepare_stream(
        self, handle: MeasurementHandle, metadata: BaseMetaData
    ) -> Path:
        """Prepare for a streaming measurement

        Args:
            handle (MeasurementHandle): identifies the streaming measurement
            metadata (BaseMetaData): saved to the metadata backend

        Raises:
            StreamAlreadyPreparedError:
                If attempting to prepare a stream using a handle that a stream
                has already been prepared for.

        Returns:
            Path: path to location prepared for streaming
        """

    @abstractmethod
    async def finalize_stream(
        self, handle: MeasurementHandle, metadata: BaseMetaData | None = None
    ):
        """Finalise a streaming measurement

        Args:
            handle (MeasurementHandle): identifies the streaming measurement
            metadata (BaseMetaData | None, optional):
                If provided overwrites the metadata supplied when streaming was
                prepared.
                Defaults to None.

        Raises:
            StreamNotPreparedError:
                a) if a stream has not been prepared for the handle
                b) if a stream has already been finalize
        """


class AbstractMetadataBackendConfig(ABC, BaseModel):
    """Configurationo for MetadataBackend"""

    @abstractmethod
    def init(self) -> AbstractMetadataBackend:
        """Initialize metadata backend from configuration"""
