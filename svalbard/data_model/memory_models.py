""" Module for Pydantic models for Memory Management """

from abc import ABC, abstractmethod
from multiprocessing.resource_tracker import unregister
from multiprocessing.shared_memory import ShareableList
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from ..utility.shared_array import SharedArray, SharedCounter, SharedMemory

# todo look inty pydantic and numpy dtypelike

__memory_list_reference__: dict[str, ShareableList] = {}

# Limit of SharedMemory name length
SHAREDLISTNAMELENGTH = 14


class AbstractMemoryIn(ABC, BaseModel):
    """
    Abstract Base Class from which memory can be allocated
    """

    dtype: str  # numpy dtype.name
    shape: tuple[int, ...]

    @abstractmethod
    def size(self) -> int:
        """returns the size of the array created using dtype and shape in bytes"""

    @abstractmethod
    def allocate(self) -> str:
        """Allocate the memory

        Returns:
            str: a string that can be used to reference the allocated memory
        """


class AbstractMemoryOut(AbstractMemoryIn, ABC):
    """Abstrace Base Class from which already allocated memory can be accessed"""

    name: str

    @abstractmethod
    def to_array(self) -> np.ndarray:
        "returns an array of the type and size specified using the named shared memory"

    @classmethod
    @abstractmethod
    def from_memory_in(cls, memory_in: AbstractMemoryIn) -> Self:
        """Create a MemoryOut from a MemoryIn and allocate the memory

        Args:
            memory_in (AbstractMemoryIn): MemoryIn to create a MemoryOut from

        Returns:
            Self: MemoryOut that has been allocated
        """


class SharedMemoryIn(AbstractMemoryIn):
    """
    Pydantic Model for creating shared memory that fits a
    numpy array of a given datatype and size
    """

    model_config = ConfigDict(frozen=True)

    @field_validator("dtype", mode="before")
    @classmethod
    def dtype_must_be_numpy_dtype(cls, dtype: str | np.dtype | type):
        """validates that dtype is a valid datatype"""
        if isinstance(dtype, np.dtype | str):
            try:
                return str(np.dtype(dtype))
            except TypeError as exc:
                raise ValueError(
                    f"MemoryIn.dtype: value {dtype} is not a valid "
                    + "representation of a numpy datatype or numpy datatype"
                ) from exc
        if isinstance(dtype, type):  # pylint: disable=W1116
            # e.g. np.int64 is of type type not np.dtype
            return str(np.dtype(dtype))
        # reaches here if dtype is 'non-sensical' such as a dict
        raise ValueError(f"Memory.dtype: value {dtype} neither np.dtype nor str")

    def size(self) -> int:
        """returns the size of the array created using dtype and shape in bytes"""
        return int(np.dtype(self.dtype).itemsize * np.prod(self.shape))

    def allocate(self) -> str:
        shm = SharedMemory(None, create=True, size=self.size(), track=False)
        sha = SharedArray(
            self.shape, dtype=np.dtype(self.dtype), buffer=shm.buf, shm=shm
        )
        if np.dtype(self.dtype).kind != "i":
            sha.fill(np.NAN)
        return shm.name


class SharedMemoryOut(SharedMemoryIn, AbstractMemoryOut):
    """dataclass to share info required for
    accessing shared memory correctly between processes"""

    name: str

    @field_validator("name", mode="after")
    @classmethod
    def increment_counter(cls, name: str) -> str:
        """increment the counter"""
        try:
            counter = SharedCounter(name=name)
        except FileNotFoundError:
            counter = SharedCounter(name=name, create=True)
        counter.increase()
        return name

    def to_array(self) -> SharedArray:
        "returns an array of the type and size specified using the named shared memory"
        shm = SharedMemory(self.name, track=False)
        return SharedArray(
            self.shape, dtype=np.dtype(self.dtype), buffer=shm.buf, shm=shm
        )

    @classmethod
    def from_memory_in(cls, memory_in: SharedMemoryIn) -> Self:
        name = memory_in.allocate()
        return cls(name=name, shape=memory_in.shape, dtype=memory_in.dtype)

    @classmethod
    def close(cls, name: str):
        """
        manually close and unlink a shared memory

        Args:
            name (str): name of the shared memory to close
        """
        try:
            shm = SharedMemory(name, track=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

    def __del__(self):
        counter = SharedCounter(name=self.name)
        counter.decrease()
        if counter.value == 0:
            self.close(self.name)
            counter.close()


class ShareableListOut(BaseModel):
    """
    ShareableListOut stores the current measurement
    progress in the following shape:
        [0] -> the total number of steps
        [1] -> the current step
        [2] -> the elapsed time as reported by tqdm
        [3] -> the rate as reported by tqdm as a way to calculate
        time remaining

    The Measurement Executor will update this list during execution time.
    """

    name: str

    @classmethod
    def create_shareable_list(cls, name: str) -> ShareableList:
        """
        Create ShareableList for Measurement Executor Progress tracking.
        """
        short_name = name[:SHAREDLISTNAMELENGTH]
        sl = ShareableList(name=short_name, sequence=[100, 1, 0.0, 0.0])
        __memory_list_reference__[short_name] = sl
        return sl

    @classmethod
    def get_shareable_list(cls, name: str) -> ShareableList:
        """Return ShareableList based on name"""
        short_name = name[:SHAREDLISTNAMELENGTH]
        try:
            sl = __memory_list_reference__[short_name]
        except KeyError:  # pragma: no cover
            sl = ShareableList(name=short_name)
            unregister("/" + short_name, "shared_memory")
            shm = SharedMemory(short_name, track=False)
            sl.shm = shm
            __memory_list_reference__[short_name] = sl
        finally:
            return sl

    @classmethod
    def close(cls, name: str):
        """Clean up ShareableList"""
        try:
            short_name = name[:SHAREDLISTNAMELENGTH]
            sl = __memory_list_reference__[short_name]
            sl.shm.close()
            sl.shm.unlink()
            del __memory_list_reference__[short_name]
        except KeyError:
            pass
