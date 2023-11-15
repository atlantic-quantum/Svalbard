"""
Lookup table for interpolation of values.

The lookup table is used by relation parameters to interpolate values
"""

from enum import Enum

import numpy as np
from pydantic import BaseModel
from scipy.interpolate import interp1d


class LookupInterpolation(str, Enum):
    """
    Enum for lookup table interpolation methos,
    covers all scipy interpolation methods
    """

    LINEAR = "linear"
    NEAREST = "nearest"
    NEAREST_UP = "nearest-up"
    ZERO = "zero"
    SLINEAR = "slinear"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    PREVIOUS = "previous"
    NEXT = "next"


class LookupTable(BaseModel):
    """
    Pydantic model for a lookup table, an internal dictionary is used to store a
    mapping from x to y values. which is used to create a interpolation function


    Args:
        xy (dict[float, float]):
            mapping from x to y values,
            x values are dictionary keys,
            y values are dictionary values
        interpolation (LookupInterpolation, optional):
            interpolation method used when creating a function interpolating
            the mapping from x to y values. Defaults to LookupInterpolation.LINEAR.

    """

    xy: dict[
        float, float
    ]  # make sure values are not rounded when serializing to database
    interpolation: LookupInterpolation = LookupInterpolation.LINEAR

    def calculate_values(self, x_values: np.ndarray) -> np.ndarray:
        """
        Calculate y values for given x values using interpolation

        Args:
            x_values (np.ndarray): x values for which to calculate y values

        returns:
            y_values (np.ndarray): y values for given x values via interpolation
        """
        interp_f = interp1d(self.x, self.y, kind=self.interpolation.value)
        return interp_f(x_values)

    @property
    def x(self) -> np.ndarray:
        """Return x values as numpy array"""
        return np.fromiter(self.xy.keys(), dtype=float)

    @property
    def y(self) -> np.ndarray:
        """Return y values as numpy array"""
        return np.fromiter(self.xy.values(), dtype=float)

    @property
    def x_sorted(self) -> np.ndarray:
        """Return x values as numpy array, sorted by x"""
        return np.sort(self.x)

    @property
    def y_sorted(self) -> np.ndarray:
        """Return y values as numpy array, sorted by x"""
        return np.array(self.y[np.argsort(self.x)])

    @classmethod
    def from_arrays(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        interpolation: LookupInterpolation = LookupInterpolation.LINEAR,
    ) -> "LookupTable":
        """
        Create a lookup table from x and y arrays, optionally set interpolation method.
        when calculating values from the lookup table a interpolation is created using a
        mapping from x to y values

        Args:
            x (np.ndarray):
                x values
            y (np.ndarray):
                y values (same length as x), mapped to x values for interpolation
            interpolation (LookupInterpolation, optional):
                interpolation method used when creating a function by mapping the
                y values from the x values. Defaults to LookupInterpolation.LINEAR.

        Returns:
            LookupTable:
                lookup table with x and y values and interpolation method
        """
        assert x.shape == y.shape
        return LookupTable(xy=dict(zip(x, y)), interpolation=interpolation)
