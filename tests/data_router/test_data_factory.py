""" Data Factory function along with tests for it"""

import numpy as np
import pytest

from svalbard.data_model.data_file import Data, DataFile, MeasurementHandle, MetaData
from svalbard.data_model.memory_models import SharedMemoryIn, SharedMemoryOut


def data_factory(size: int, type_list: list | None = None, shapes: list | None = None):
    """
        Factory function for creating a DataFile with random data
        Args:
            size: size of the data (number of elements)
            type_list (optional): list of numpy dtypes for each array, if none specified
                all arrays are float
            shapes (optional): list of shapes for each array, if none specified, one
                array of size is created
    s
        Returns:
            DataFile with random data (can be of specified type and shape)
    """
    if shapes is None or shapes == []:
        shapes = [(size,)]
    if type_list is None or type_list == []:
        type_list = [np.dtype("float")] * len(shapes)
    total_shape = sum([np.prod(shape) for shape in shapes])
    if total_shape < size:
        shapes.append((size - total_shape,))
    elif total_shape > size:
        raise Exception("shapes declared are greater than size")
    if len(type_list) < len(shapes):
        type_list.extend([np.dtype("float")] * (len(shapes) - len(type_list)))
    elif len(type_list) > len(shapes):
        raise Exception("type_list declared is greater than shapes")
    mems_in = [
        SharedMemoryIn(dtype=str(dtype), shape=shape)
        for (shape, dtype) in zip(shapes, type_list)
    ]
    mems_out = [SharedMemoryOut.from_memory_in(mem_in) for mem_in in mems_in]
    rng = np.random.default_rng()
    for mem_out in mems_out:
        mem_out.to_array()[:] = rng.standard_normal(size=mem_out.shape).astype(
            mem_out.dtype
        )
    datasets = [
        Data.DataSet(name=f"demo_data_name_{i}", memory=memory)
        for i, memory in enumerate(mems_out)
    ]
    handle = MeasurementHandle.new()
    metadata = MetaData(name="demo_metadata")
    data = Data(handle=handle, datasets=datasets)
    # total_mem = sum([mem_in.size() for mem_in in mems_in])
    # print(f"Datafile is {len(datasets)} arrays totalling {total_mem} bytes")
    return DataFile(data=data, metadata=metadata)


def test_size_equal_parameters():
    """
    Test of ideal parameter input, where size of the data is equal to the sum of the
        sizes of the shapes and the number of types matches the number of
        shapes declared
    """
    shapes = [(100,), (10, 10), (3, 3, 3), (20, 20, 20, 10)]
    dtypes = [np.dtype("float"), np.dtype("float"), np.dtype("int"), np.dtype("bool")]
    datafile = data_factory(80227, dtypes, shapes)
    assert datafile.data is not None
    assert (
        datafile.data.datasets[0].memory.size()
        == np.prod(shapes[0]) * dtypes[0].itemsize
    )
    assert (
        datafile.data.datasets[1].memory.size()
        == np.prod(shapes[1]) * dtypes[1].itemsize
    )
    assert (
        datafile.data.datasets[2].memory.size()
        == np.prod(shapes[2]) * dtypes[2].itemsize
    )
    assert (
        datafile.data.datasets[3].memory.size()
        == np.prod(shapes[3]) * dtypes[3].itemsize
    )
    assert datafile.data.datasets[0].memory.dtype == dtypes[0]
    assert datafile.data.datasets[1].memory.dtype == dtypes[1]
    assert datafile.data.datasets[2].memory.dtype == dtypes[2]
    assert datafile.data.datasets[3].memory.dtype == dtypes[3]
    assert datafile.data.datasets[0].memory.shape == shapes[0]
    assert datafile.data.datasets[1].memory.shape == shapes[1]
    assert datafile.data.datasets[2].memory.shape == shapes[2]
    assert datafile.data.datasets[3].memory.shape == shapes[3]


def test_size_greater_parameters():
    """
    Test when declared size is greater than the declared shapes, should automatically
        create a new array with the remaining size. Assumes number of types matches
        number of shapes declared
    """
    shapes = [(100,), (10, 10), (3, 3, 3), (20, 20, 20, 10)]
    dtypes = [np.dtype("float"), np.dtype("float"), np.dtype("int"), np.dtype("bool")]
    datafile = data_factory(100000, dtypes, shapes)
    assert datafile.data is not None
    assert (
        datafile.data.datasets[0].memory.size()
        == np.prod(shapes[0]) * dtypes[0].itemsize
    )
    assert (
        datafile.data.datasets[1].memory.size()
        == np.prod(shapes[1]) * dtypes[1].itemsize
    )
    assert (
        datafile.data.datasets[2].memory.size()
        == np.prod(shapes[2]) * dtypes[2].itemsize
    )
    assert (
        datafile.data.datasets[3].memory.size()
        == np.prod(shapes[3]) * dtypes[3].itemsize
    )
    assert (
        datafile.data.datasets[4].memory.size()
        == np.prod(shapes[4]) * dtypes[4].itemsize
    )
    assert datafile.data.datasets[4].memory.size() != 0
    assert datafile.data.datasets[0].memory.dtype == dtypes[0]
    assert datafile.data.datasets[1].memory.dtype == dtypes[1]
    assert datafile.data.datasets[2].memory.dtype == dtypes[2]
    assert datafile.data.datasets[3].memory.dtype == dtypes[3]
    assert datafile.data.datasets[4].memory.dtype == dtypes[4]
    assert datafile.data.datasets[0].memory.shape == shapes[0]
    assert datafile.data.datasets[1].memory.shape == shapes[1]
    assert datafile.data.datasets[2].memory.shape == shapes[2]
    assert datafile.data.datasets[3].memory.shape == shapes[3]
    assert datafile.data.datasets[4].memory.shape == shapes[4]


def test_only_size():
    """Tests when only size is declared and neither optional parameter, should create
    a single array of size size"""
    datafile = data_factory(1000)
    assert datafile.data is not None
    assert datafile.data.datasets[0].memory.size() == 1000 * np.dtype("float").itemsize
    assert datafile.data.datasets[0].memory.dtype == np.dtype("float")
    assert datafile.data.datasets[0].memory.shape == (1000,)


def test_only_size_and_type():
    """
    Tests when only the size and type list are declared, not the shapes. Should throw
        an exception
    """
    dtypes = [np.dtype("float"), np.dtype("float"), np.dtype("int"), np.dtype("bool")]
    with pytest.raises(Exception) as context:
        data_factory(1000, dtypes)
    assert "type_list declared is greater than shapes" in str(context.value)


def test_only_size_and_shapes():
    """
    Tests when only the size and shapes are declared, not the type list. Should create
        a typelist of all floats
    """
    shapes = [(100,), (10, 10), (3, 3, 3), (20, 20, 20, 10)]
    datafile = data_factory(100000, shapes=shapes)
    assert datafile.data is not None
    assert (
        datafile.data.datasets[0].memory.size()
        == np.prod(shapes[0]) * np.dtype("float").itemsize
    )
    assert (
        datafile.data.datasets[1].memory.size()
        == np.prod(shapes[1]) * np.dtype("float").itemsize
    )
    assert (
        datafile.data.datasets[2].memory.size()
        == np.prod(shapes[2]) * np.dtype("float").itemsize
    )
    assert (
        datafile.data.datasets[3].memory.size()
        == np.prod(shapes[3]) * np.dtype("float").itemsize
    )
    assert (
        datafile.data.datasets[4].memory.size()
        == np.prod(shapes[4]) * np.dtype("float").itemsize
    )
    assert datafile.data.datasets[4].memory.size() != 0
    assert datafile.data.datasets[0].memory.dtype == np.dtype("float")


def test_fewer_types_than_shapes():
    """
    Test when the number of types is fewer than the number of shapes, should add
        remaining "float" types
    """
    shapes = [(100,), (10, 10), (3, 3, 3), (20, 20, 20, 10)]
    dtypes = [np.dtype("float"), np.dtype("float"), np.dtype("int")]
    datafile = data_factory(100000, dtypes, shapes)
    assert datafile.data is not None
    assert (
        datafile.data.datasets[0].memory.size()
        == np.prod(shapes[0]) * dtypes[0].itemsize
    )
    assert (
        datafile.data.datasets[1].memory.size()
        == np.prod(shapes[1]) * dtypes[1].itemsize
    )
    assert (
        datafile.data.datasets[2].memory.size()
        == np.prod(shapes[2]) * dtypes[2].itemsize
    )
    assert (
        datafile.data.datasets[3].memory.size()
        == np.prod(shapes[3]) * np.dtype("float").itemsize
    )
    assert (
        datafile.data.datasets[4].memory.size()
        == np.prod(shapes[4]) * np.dtype("float").itemsize
    )
    assert datafile.data.datasets[0].memory.dtype == dtypes[0]
    assert datafile.data.datasets[1].memory.dtype == dtypes[1]
    assert datafile.data.datasets[2].memory.dtype == dtypes[2]
    assert datafile.data.datasets[3].memory.dtype == np.dtype("float")
    assert datafile.data.datasets[4].memory.dtype == np.dtype("float")


def test_fewer_shapes_than_types():
    """
    Test when the number of shapes is fewer than the number of types, should throw an
        exception
    """
    shapes = [(100,), (10, 10), (3, 3, 3)]
    dtypes = [
        np.dtype("float"),
        np.dtype("float"),
        np.dtype("int"),
        np.dtype("bool"),
        np.dtype("float"),
    ]
    with pytest.raises(Exception) as context:
        data_factory(100000, dtypes, shapes)
    assert "type_list declared is greater than shapes" in str(context.value)


def test_size_less_than_shape():
    """
    Test when the size is less than the shape, should throw an exception
    """
    shapes = [(100,), (10, 10), (3, 3, 3), (20, 20, 20, 10)]
    with pytest.raises(Exception) as context:
        data_factory(1000, shapes=shapes)
    assert "shapes declared are greater than size" in str(context.value)
