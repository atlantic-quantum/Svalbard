import numpy as np

from svalbard.utility.shared_array import SharedArray, SharedMemory


def test_operating_changes_to_numpy_array():
    arr = np.array([1, 2, 3, 4, 5])
    size = int(np.dtype(arr.dtype).itemsize * np.prod(arr.shape))
    sm = SharedMemory(create=True, size=size)
    sarr = SharedArray(arr.shape, dtype=arr.dtype, buffer=sm.buf, shm=sm)
    sarr[:] = arr
    assert isinstance(sarr, SharedArray)
    assert sarr.shape == arr.shape
    assert sarr.dtype == arr.dtype
    assert np.all(sarr == arr)
    not_sarr = sarr * 2
    assert not isinstance(not_sarr, SharedArray)
    assert np.all(not_sarr == arr * 2)
