"""
Pydantic models to be used fro inter-process communication
"""
from pathlib import Path

from pydantic import BaseModel

from .data_file import DataFile, MeasurementHandle
from .memory_models import SharedMemoryOut


class BufferReference(SharedMemoryOut):
    """Model for storing references to shared memory buffers for ipc communication"""


class SavedPath(BaseModel):
    """Model for returning paths to data from data server"""

    path: Path


class StartStreamModel(BaseModel):
    """IPC model for passing information required to
    start streaming data to the data server"""

    handle: MeasurementHandle
    file: DataFile


class EndStreamModel(BaseModel):
    """IPC model for passing information required to
    end streaming data to the data server"""

    handle: MeasurementHandle
    file: DataFile | None = None


class SaveBufferModel(BaseModel):
    """IPC model for passing information required to
    save buffer in a correct location to the data server"""

    handle: MeasurementHandle
    name: str  # name of dataset the data in the buffer belongs to
    buffer: BufferReference
    index: list[int]  # data array index of where to insert buffer


class SliceModel(BaseModel):
    """
    Abstract Base Class from which memory can be allocated
    """

    start: int | None = None
    stop: int | None = None
    step: int | None = None

    def to_slice(self) -> slice:
        """Converts SliceModel to slice"""
        if self.step is None:
            return slice(self.start, self.stop)
        else:
            return slice(self.start, self.stop, self.step)

    @classmethod
    def from_slice(cls, slice_: slice) -> "SliceModel":
        """Converts slice to SliceModel"""
        return cls(start=slice_.start, stop=slice_.stop, step=slice_.step)


class SliceListModel(BaseModel):
    """
    Model for storing a list of list SliceModels (for each dimension of each array)
    """

    slice_lists: list[list[SliceModel]] = []

    def to_slice_lists(self) -> list[list[slice]]:
        """
        returns list of lists of slices created from a list of lists of SliceModels
        """
        return [
            [slice_.to_slice() for slice_ in slice_list]
            for slice_list in self.slice_lists
        ]

    @classmethod
    def from_slice_lists(cls, slice_lists: list[list[slice]]) -> "SliceListModel":
        """returns SliceListModel from list of list of lists of slices"""
        return cls(
            slice_lists=[
                [SliceModel.from_slice(slice_) for slice_ in slice_list]
                for slice_list in slice_lists
            ]
        )


class DataFileAndSliceModel(BaseModel):
    """
    Modelf for storing DataFile and SliceListModel
    """

    data_file: DataFile
    slices: SliceListModel | None = None
