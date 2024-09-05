"""
Settings types for instrument settings
"""

from abc import ABC
from enum import Enum, StrEnum

import numpy as np

from ..typing import TSettingValue
from ..utility.str_helper import get_si_string


class BaseSettingType(ABC):  # pragma: no cover
    """
    Base class for instrument setting types, used to convert values to the correct type
    and to get a string representation of the value
    """

    @staticmethod
    def to_value(value):
        """
        Returns the value converted to the correct type such that it can be used
        as a value of an instrument setting

        Args:
            value (_type_): value to convert, type depends on the setting type class
        """
        raise NotImplementedError

    @staticmethod
    def get_value_string(value, unit: str) -> str:
        """
        Args:
            value (_type_):
                value to convert to string, type depends on the setting type class
            unit (str):
                unit of the value

        Returns:
            str: Formatted string representation of the value with the unit
        """
        raise NotImplementedError

    @property
    def setting_type(self) -> "SettingType":
        """
        Returns:
            SettingType: The "SettingType" of the setting type class
        """
        raise NotImplementedError


class SettingType(StrEnum):
    """Enumeration for instrument setting types"""

    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    ENUM = "enum"
    STR = "str"
    COMPLEX = "complex"
    MODEL = "model"
    LIST_FLOAT = "list[float]"
    LIST_INT = "list[int]"
    LIST_BOOL = "list[bool]"
    LIST_ENUM = "list[enum]"
    LIST_STR = "list[str]"
    LIST_COMPLEX = "list[complex]"
    LIST_MODEL = "list[model]"
    OBJECT = "object"
    NONE = "none"

    @staticmethod
    def determine_setting_type(
        value: TSettingValue, setting_type: "SettingType | str | None"
    ) -> "SettingType":
        """
        Use the value and setting type to determine the setting type

        Args:
            value (TSettingValue):
                Value to determine the setting type from
            setting_type (SettingType | str | None):
                SettingType, string or None to determine the setting type from

        Raises:
            ValueError: If a valid setting type cannot be determined

        Returns:
            SettingType: Setting type determined from the value and setting type
        """
        if setting_type == SettingType.NONE:
            return _determine_setting_type_from_value(value)
        if isinstance(setting_type, SettingType):
            return setting_type
        if isinstance(setting_type, str):
            return _str_to_setting_type.get(setting_type.lower(), SettingType.NONE)
        if setting_type is not None:
            raise ValueError(f"Invalid setting type {setting_type}")
        return _determine_setting_type_from_value(value)

    @staticmethod
    def get_setting_type_class(setting_type: "SettingType") -> type[BaseSettingType]:
        """Get the setting type class for the given setting type"""
        setting_type_class_map: dict[SettingType, type[BaseSettingType]] = {
            type_class().setting_type: type_class
            for type_class in BaseSettingType.__subclasses__()
        }
        return setting_type_class_map[setting_type]


def _determine_setting_type_from_value(value: TSettingValue) -> SettingType:
    """Determine a setting type from a value"""
    if isinstance(value, list):
        if len(value) == 0:
            return SettingType.NONE
        type_str = f"list[{_type_str(value[0])}]"
    else:
        type_str = _type_str(value)
    return _str_to_setting_type.get(type_str, SettingType.NONE)


def _type_str(value: TSettingValue) -> str:
    """Get a SettingType string representation of the type of a value"""
    if value is None:
        return SettingType.NONE.value
    if isinstance(value, complex):
        return SettingType.COMPLEX.value
    if isinstance(value, float):
        return SettingType.FLOAT.value
    if isinstance(value, bool):
        return SettingType.BOOL.value
    if isinstance(value, int):
        return SettingType.INT.value
    if isinstance(value, Enum):
        return SettingType.ENUM.value
    if isinstance(value, str):
        return SettingType.STR.value
    raise ValueError(f"Invalid value {value}, type {type(value)}")


_str_to_setting_type: dict[str, SettingType] = {
    setting_type.value: setting_type for setting_type in SettingType
}

LIST_SETTING_TYPES = [
    setting_type for setting_type in SettingType if "list" in setting_type
]


def _complex_to_str(value: complex | str) -> str:
    """Convert complex value to string"""
    if isinstance(value, str):
        try:
            complex(value)
        except ValueError:
            raise ValueError(f"Invalid complex value {value}")
        return value
    sign = "+" if value.imag >= 0 else ""
    return f"{value.real}{sign}{value.imag}j"


def _enum_to_value(value: Enum | str | int) -> str | int:
    """Convert Enum value to string"""
    if isinstance(value, Enum):
        return value.value
    return value


class FloatSettingType(BaseSettingType):
    """Implementation of BaseSettingType for float settings"""

    @staticmethod
    def to_value(value: float) -> float:
        return float(value)

    @staticmethod
    def get_value_string(value: float, unit: str) -> str:
        return get_si_string(value, unit, decimals=6)

    @property
    def setting_type(self) -> SettingType:
        return SettingType.FLOAT


class IntSettingType(BaseSettingType):
    """Implementation of BaseSettingType for integer settings"""

    @staticmethod
    def to_value(value: int) -> int:
        return int(value)

    @staticmethod
    def get_value_string(value: int, unit: str) -> str:
        if unit:
            return f"{value} {unit}"
        return str(value)

    @property
    def setting_type(self) -> SettingType:
        return SettingType.INT


class BoolSettingType(BaseSettingType):
    """Implementation of BaseSettingType for boolean settings"""

    @staticmethod
    def to_value(value: bool) -> bool:
        return bool(value)

    @staticmethod
    def get_value_string(value: bool, unit: str) -> str:
        return "On" if value else "Off"

    @property
    def setting_type(self) -> SettingType:
        return SettingType.BOOL


class EnumSettingType(BaseSettingType):
    """Implementation of BaseSettingType for enumeration settings"""

    @staticmethod
    def to_value(value: Enum | str | int) -> str | int:
        return _enum_to_value(value)

    @staticmethod
    def get_value_string(value: Enum | str | int, unit: str) -> str:
        if isinstance(value, Enum):
            return str(value.value)
        return str(value)

    @property
    def setting_type(self) -> SettingType:
        return SettingType.ENUM


class StrSettingType(BaseSettingType):
    """Implementation of BaseSettingType for string settings"""

    @staticmethod
    def to_value(value: str) -> str:
        return str(value)

    @staticmethod
    def get_value_string(value: str, unit: str) -> str:
        return value

    @property
    def setting_type(self) -> SettingType:
        return SettingType.STR


class ComplexSettingType(BaseSettingType):
    """Implementation of BaseSettingType for complex settings"""

    @staticmethod
    def to_value(value: complex | str) -> str:
        # complex values can't be converted to json directly so we store them as strings
        return _complex_to_str(value)

    @staticmethod
    def get_value_string(value: complex | str, unit: str) -> str:
        if isinstance(value, str):
            value = complex(value)
        magnitude = abs(value)
        angle = np.angle(value) * 180.0 / np.pi
        # create string for magnitude and use float for angle
        magnitude_str = get_si_string(magnitude, unit, decimals=6)
        return f"{magnitude_str} < {angle:.1f}°"

    @property
    def setting_type(self) -> SettingType:
        return SettingType.COMPLEX


class ModelSettingType(BaseSettingType):
    """
    Implementation of BaseSettingType for a setting that stores the name of a model
    """

    @staticmethod
    def to_value(value: str) -> str:
        return value

    @staticmethod
    def get_value_string(value: str, unit: str) -> str:
        return value

    @property
    def setting_type(self) -> SettingType:
        return SettingType.MODEL


class ObjectSettingType(BaseSettingType):
    """
    Implementation of BaseSettingType for a setting that stores the name of an object
    """

    @staticmethod
    def to_value(value: str) -> str:
        return value

    @staticmethod
    def get_value_string(value: str, unit: str) -> str:
        return value

    @property
    def setting_type(self) -> SettingType:
        return SettingType.OBJECT


class NoneSettingType(BaseSettingType):
    """Implementation of BaseSettingType for a setting that stores no value"""

    @staticmethod
    def to_value(value: list | None) -> list | None:
        if isinstance(value, list):
            if len(value) == 0:
                return []
        if value is None:
            return None
        raise ValueError(f"Invalid value {value} for setting type {SettingType.NONE}")

    @staticmethod
    def get_value_string(value: list | None, unit: str) -> str:
        return ""

    @property
    def setting_type(self) -> SettingType:
        return SettingType.NONE


def list_setting_type_factory(setting_type_class: type[BaseSettingType]):
    """Create a list setting type class from a setting type class"""
    stype = setting_type_class().setting_type.value

    fmt = "Implementation of BaseSettingType for a setting that is a list of {}"
    doc = fmt.format(stype)

    def to_value(value: list) -> list:
        return [setting_type_class.to_value(i) for i in value]

    def get_value_string(value: list, unit: str) -> str:
        return f"{[setting_type_class.get_value_string(i, unit) for i in value]}"

    def setting_type(self) -> SettingType:
        return SettingType(f"list[{stype}]")

    ListSettingType = type(
        "List" + setting_type_class.__name__,
        (BaseSettingType,),
        {
            "to_value": to_value,
            "get_value_string": get_value_string,
            "setting_type": property(setting_type),
            "__doc__": doc,
        },
    )

    return ListSettingType


ListFloatSettingType = list_setting_type_factory(FloatSettingType)
ListIntSettingType = list_setting_type_factory(IntSettingType)
ListBoolSettingType = list_setting_type_factory(BoolSettingType)
ListEnumSettingType = list_setting_type_factory(EnumSettingType)
ListStrSettingType = list_setting_type_factory(StrSettingType)
ListComplexSettingType = list_setting_type_factory(ComplexSettingType)
ListModelSettingType = list_setting_type_factory(ModelSettingType)
