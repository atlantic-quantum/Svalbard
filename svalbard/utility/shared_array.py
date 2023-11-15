"""
Extend numpy ndarray to also store a reference to a shared memory,
not just the buffer of a shared memory.
Adapted from https://numpy.org/doc/stable/user/basics.subclassing.html
"""
import logging
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
from fasteners import InterProcessReaderWriterLock as IPRWLock

from ..utility.logger import logger as data_server_logger


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
