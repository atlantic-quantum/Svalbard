"""
Data models for devices, including serial number and designer
"""
from typing import Literal

from pydantic import BaseModel, Field, root_validator


class DeviceInfo(BaseModel):
    """
    Data model for a generic device

    Args:
        serial_number (str):
            serial number of the device, defaults to "unknown"
        model (Literal["other"]):
            data model for device info, defaults to "other".
            Used as discriminator between different device info models
    """

    serial_number: str = "unknown"
    model: Literal["other"] = "other"


class AtlanticDeviceInfo(DeviceInfo):
    """Data model for a device from Atlantic Quantum

    Parameter description copied from slack discussion with Sergey

    Args:
        mask (str):
            mask name (Alewife, Betta, Cod, Dorada, etc.)
            -- these are unique but might have a revision appended to the name,
            e.g. Cod rev. 1
        wafer (str):
            wafer name (e.g. W1-2023-03-21)
            -- unique in combination with the mask name
        chip (str):
            chip name (e.g. Chip 1, Chip 2b, etc.
            -- these are not unique as there are multiple copies of them)
        die (str):
            die name (e.g. E5) -- unique positions on the wafer
        serial_number (str):
            serial number of the device, automatically generated from the mask, wafer,
            chip, and die fields during validation.
        model (Literal["atlantic"]):
            data model for device info, defaults to "atlantic".
            Used as discriminator between different device info models
    """

    mask: str = "unknown"
    wafer: str = "unknown"
    chip: str = "unknown"
    die: str = "unknown"
    model: Literal["atlantic"] = "atlantic"

    @root_validator
    def combine_info_to_serial(cls, values: dict) -> dict:
        """
        root_validator to combine the mask, wafer, chip, and die fields into a
        serial number, called automatically during validation. Does not need to be
        called manually.

        Args:
            values (dict):
                dictionary of constructor arguments (note that pydantic model
                constructors are called with keyword arguments only)

        Returns:
            dict: modified values dictionary with the serial_number field added
        """
        values[
            "serial_number"
        ] = f"{values['mask']}-{values['wafer']}-{values['chip']}-{values['die']}"
        return values


class Device(BaseModel):
    device_info: DeviceInfo | AtlanticDeviceInfo = Field(
        default_factory=AtlanticDeviceInfo, discriminator="model"
    )
    device_designer: str = "atlantic"
