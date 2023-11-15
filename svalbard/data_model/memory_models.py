""" Module for Pydantic models for Memory Management """
from abc import ABC, abstractmethod
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from pydantic import BaseConfig, BaseModel, validator
from typing_extensions import Self

from ..utility.shared_array import SharedArray

# todo look inty pydantic and numpy dtypelike

__memory_reference__: dict[str, SharedMemory] = {}


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

    @validator("dtype", pre=True)
    def dtype_must_be_numpy_dtype(cls, dtype: str | np.dtype | type):
        """validates that dtype is a valid datatype"""
        if isinstance(dtype, np.dtype):
            return dtype.name
        if isinstance(dtype, str | type):  # pylint: disable=W1116
            # e.g. np.int64 is of type type not np.dtype
            try:
                np_v = np.dtype(dtype)
                return np_v.name
            except TypeError as exc:
                raise ValueError(
                    f"MemoryIn.dtype: value {dtype} is not a valid "
                    + "string representation of a numpy datatype or numpy datatype"
                ) from exc
        else:
            raise ValueError(f"Memory.dtype: value {dtype} neither np.dtype nor str")

    class Config(BaseConfig):
        """BaseModel confic class customization"""

        allow_mutation = False

    def size(self) -> int:
        """returns the size of the array created using dtype and shape in bytes"""
        return int(np.dtype(self.dtype).itemsize * np.prod(self.shape))

    def allocate(self) -> str:
        shm = SharedMemory(None, create=True, size=self.size())
        __memory_reference__[shm.name] = shm
        # ! this may get garbaage collected on function return
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

    def to_array(self) -> SharedArray:
        "returns an array of the type and size specified using the named shared memory"
        shm = SharedMemory(self.name)
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
            shm = __memory_reference__[name]
            shm.close()
            shm.unlink()
            del __memory_reference__[name]
        except KeyError:
            pass
