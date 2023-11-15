"""Tests for memory models"""
import numpy as np
import pytest
from pydantic import ValidationError
from svalbard.data_model.memory_models import SharedMemoryIn, SharedMemoryOut


def test_shared_memory_in_validator():
    """test if dtypes of MemoryIn correctly validate to strigns that
    represent numpy datatypes"""
    mem_in = SharedMemoryIn(dtype="int64", shape=(1,))
    assert mem_in.dtype == "int64"
    assert mem_in.size() == 8
    mem_in = SharedMemoryIn(dtype=np.int64, shape=(1,))  # type: ignore
    assert mem_in.dtype == np.dtype(np.int64).name
    mem_in = SharedMemoryIn(dtype=np.dtype("int64"), shape=(1,))  # type: ignore
    assert mem_in.dtype == np.dtype("int64").name
    with pytest.raises(ValidationError):
        SharedMemoryIn(dtype="not_numpy_str", shape=(1,))
    with pytest.raises(ValidationError):
        SharedMemoryIn(dtype={}, shape=(1,))  # type: ignore


def test_shared_memory_out_to_array():
    """test creating an array from shared memory"""
    mem_in = SharedMemoryIn(dtype="int64", shape=(3, 3))
    mem_out = SharedMemoryOut.from_memory_in(mem_in)
    arr = mem_out.to_array()
    assert arr.dtype == np.dtype(mem_in.dtype)
    assert arr.shape == mem_in.shape
    SharedMemoryOut.close(mem_out.name)


def test_close_pass_on_key_error():
    """test that close passes on key errors"""
    SharedMemoryOut.close("not_a_memory")
