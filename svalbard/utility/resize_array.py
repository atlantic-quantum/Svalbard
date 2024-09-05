"""Utility function for resizing an zarr array"""

import numpy as np


def new_shape(
    array_shape: tuple[int, ...], slices: list[slice], chunk_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """calculates a new shape of an array while accomodating chunk shape

    Args:
        array_shape (tuple[int]): current shape of array
        slices (list[slice]): desired insertion shape
        chunk_shape (tuple[int]): chunk shape of array

    Raises:
        ValueError: if array dimension and slices dimension don't match
        ValueError: if array dimension and chunk dimension don't match

    Returns:
        tuple[int]: the new shape of the array
    """
    # ! todo replace with typevartuple when update to 3.11
    if len(array_shape) != len(slices):
        raise ValueError(
            f"resize_array: dimension of array ({len(array_shape)}"
            + f"and slices ({len(slices)}) does not match"
        )
    if len(array_shape) != len(chunk_shape):
        raise ValueError(
            f"resize_array: dimension of array ({len(array_shape)}"
            + f"and chunk ({len(chunk_shape)}) does not match"
        )
    max_shape = np.array(
        [
            max(a_dim, s_dim.stop if s_dim.stop else a_dim)
            for a_dim, s_dim in zip(array_shape, slices)
        ]
    )
    return tuple(
        (np.ceil(max_shape / np.array(chunk_shape)) * np.array(chunk_shape)).astype(int)
    )
