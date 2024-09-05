"""
Extend numpy ndarray to also store a reference to a shared memory,
not just the buffer of a shared memory.
Adapted from https://numpy.org/doc/stable/user/basics.subclassing.html
"""

import logging
import sys
import threading
import time
from multiprocessing import resource_tracker as _mprt
from multiprocessing import shared_memory as _mpshm
from pathlib import Path

import numpy as np
from fasteners import InterProcessReaderWriterLock as IPRWLock

from ..utility.logger import logger as data_server_logger

_SHM_NAME_PREFIX = "psm_"  # will not work on windows

# minimal functional backport of SharedMemory from Python 3.13
# from https://github.com/python/cpython/issues/82300#issuecomment-2169035092
if sys.version_info >= (3, 13):
    SharedMemory = _mpshm.SharedMemory
else:

    class SharedMemory(_mpshm.SharedMemory):
        __lock = threading.Lock()

        def __init__(
            self,
            name: str | None = None,
            create: bool = False,
            size: int = 0,
            *,
            track: bool = True,
        ) -> None:
            self._track = track

            # if tracking, normal init will suffice
            if track:
                return super().__init__(name=name, create=create, size=size)

            # lock so that other threads don't attempt to use the
            # register function during this time
            with self.__lock:
                # temporarily disable registration during initialization
                orig_register = _mprt.register
                _mprt.register = self.__tmp_register

                # initialize; ensure original register function is
                # re-instated
                try:
                    super().__init__(name=name, create=create, size=size)
                finally:
                    _mprt.register = orig_register

        @staticmethod
        def __tmp_register(*args, **kwargs) -> None:
            return

        def unlink(self) -> None:
            if _mpshm._USE_POSIX and self._name:  # type: ignore
                _mpshm._posixshmem.shm_unlink(self._name)  # type: ignore
                if self._track:
                    _mprt.unregister(self._name, "shared_memory")  # type: ignore


class InterProcessRWRLock(IPRWLock):
    """
    Reentrant extension to fasteners.InterProcessReaderWriterLock
    Reentrant extension inspired by portalocker RLock.
    """

    def __init__(
        self,
        path: Path | str,
        sleep_func=time.sleep,
        logger: logging.Logger | None = None,
    ):
        super().__init__(path, sleep_func, logger=logger)
        self._acquire_count = 0

    # pylint: disable=too-many-arguments
    def _acquire(
        self,
        blocking=True,
        delay=IPRWLock.DELAY_INCREMENT,
        max_delay=IPRWLock.MAX_DELAY,
        timeout=None,
        exclusive=True,
    ):
        acquired = (
            super()._acquire(blocking, delay, max_delay, timeout, exclusive)
            if self._acquire_count == 0
            else True
        )
        if acquired:
            self._acquire_count += 1
        return acquired

    def _release(self, release_method):
        if self._acquire_count == 0:
            raise ValueError("Lock released more than aquired")
        if self._acquire_count == 1:
            release_method()
        self._acquire_count -= 1

    def release_read_lock(self):
        return self._release(super().release_read_lock)

    def release_write_lock(self):
        return self._release(super().release_write_lock)


class SharedCounter:
    """Shared counter using shared memory

    Args:
        name (str, optional): name of the shared memory. Defaults to None.
        create (bool, optional): whether to create the shared memory. Defaults to False.

    Note:
        The counter can either be created or connected to an existing shared memory.
        If create is True, name must be None. If create is False, name must be provided.

    Properties:
        value (int): value of the counter
        name (str): name of the shared memory used for the counter

    Raises:
        ValueError: If create is True and name is not None
        ValueError: If create is False and name is None
    """

    MAX = 255
    MIN = 0
    COUNTER_PREFIX = "cnt_"

    def __init__(self, name: str, create: bool = False):
        if name is None and not create:
            raise ValueError("name must be provided if create is False")
        if not name.startswith(_SHM_NAME_PREFIX):
            raise ValueError(f"name must start with {_SHM_NAME_PREFIX}")
        name = name[len(_SHM_NAME_PREFIX) :]
        name = f"{self.COUNTER_PREFIX}{name}"
        self.__memory = SharedMemory(name=name, create=create, size=1)
        if create:
            self.__memory.buf[0] = 0

    def increment(self, n: int = 1):
        """Increment the counter by n

        Args:
            n (int, optional): amount to increment the counter by. Defaults to 1.

        Raises:
            ValueError: if the counter would be negative after increment
            ValueError: if the counter would be greater than 255 after increment
        """
        if self.__memory.buf[0] + n < self.MIN:
            raise ValueError("Counter cannot be negative")
        if self.__memory.buf[0] + n > self.MAX:
            raise ValueError("Counter cannot exceed sys.maxsize")
        self.__memory.buf[0] += n

    def increase(self):
        """Increment the counter by 1"""
        self.increment(1)

    def decrease(self):
        """Decrement the counter by 1"""
        self.increment(-1)

    @property
    def value(self) -> int:
        """Value of the counter"""
        return self.__memory.buf[0]

    @property
    def name(self) -> str:
        """Name of the shared memory used for the counter"""
        return self.__memory.name

    def close(self):
        """Close the shared memory"""
        self.__memory.close()
        self.__memory.unlink()
        del self.__memory


class SharedArray(np.ndarray):
    """Extended numpy array to keep track of shared memory"""

    # pylint: disable=too-many-arguments
    def __new__(
        cls,
        shape,
        dtype: np.dtype = np.dtype(np.float64),
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        shm: SharedMemory | None = None,
    ):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = super().__new__(cls, shape, dtype, buffer, offset, strides, order)
        # set the new 'info' attribute to the value passed
        obj.shm = shm
        if shm:
            obj._lock = InterProcessRWRLock(
                f"/tmp/locks/{shm.name}", logger=data_server_logger
            )
            # obj._lock.logger.setLevel(5)
        # Finally, we must return the newly created object:
        return obj

    # pylint: enable=too-many-arguments
    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None:
            return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.shm = getattr(obj, "shm", None)
        self._lock = getattr(obj, "_lock", None)
        # We do not need to return anything

    def __getitem__(self, key):
        with self._lock.read_lock():  # type: ignore
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        with self._lock.write_lock():  # type: ignore
            super().__setitem__(key, value)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # If any operations are performed on the shared array, copy the data to a new
        # numpy array and then perform the operation. ensures that the result is not
        # a shared array.
        inputs = list(inputs)
        for i, input_ in enumerate(inputs):
            if isinstance(input_, SharedArray):
                inputs[i] = np.array(input_).copy()
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
