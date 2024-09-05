"""Test determining new_shape for chunked array"""

import pytest

from svalbard.utility.resize_array import new_shape

# todo loop over random shapes
# todo type hinting for tuple of ints


def test_new_shape():
    """Test functionality of the new_shape function for resizing arrays"""
    arr_shape = (3, 7)
    chunk_shape = (3, 7)
    slices = (slice(3, 6), slice(7, 14))
    new_s = new_shape(arr_shape, slices, chunk_shape)
    assert new_s == (6, 14)


def test_lengths_new_shape_slices():
    """Test that slices of different length than array raises ValueError"""
    with pytest.raises(ValueError):
        new_shape((1, 1), (1,), (1, 1))  # type: ignore


def test_lengths_new_shape_chunk():
    """Test that chunk shape of different length than array raises ValueError"""
    with pytest.raises(ValueError):
        new_shape((1, 1), (1, 1), (1,))  # type: ignore
