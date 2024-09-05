"""
Module for type aliases used in the project.
"""

from enum import Enum
from pathlib import Path
from typing import TypeVar

import numpy as np

PathLike = str | Path
ArrayLike = np.ndarray | list
TSettingTypeBase = TypeVar("TSettingTypeBase", bool, int, float, Enum, str)
TSettingValue = TSettingTypeBase | list[TSettingTypeBase]
TSettingValueSweep = list[TSettingTypeBase] | list[list[TSettingTypeBase]]
