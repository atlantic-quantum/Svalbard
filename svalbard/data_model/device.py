"""
Data models for devices, including serial number and designer
"""

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class DeviceInfo(BaseModel):
    """
    Data model for a generic device info

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
    """Device info data model for a device from Atlantic Quantum

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

    @model_validator(mode="after")
    def combine_info_to_serial(self) -> Self:
        """
        model_validator to combine the mask, wafer, chip, and die fields into a
        serial number, called automatically during validation. Does not need to be
        called manually.

        Returns:
            AtlanticDeviceInfo:
                self, with the serial_number field updated to the combined fields
        """
        self.serial_number = f"{self.mask}-{self.wafer}-{self.chip}-{self.die}"
        return self


class Device(BaseModel):
    """
    Data model for a device

    Args:
        device_info (DeviceInfo | AtlanticDeviceInfo):
            device info data model for the device, defaults to AtlanticDeviceInfo
        device_designer (str):
            name of the device designer, defaults to "atlantic"
    """

    device_info: DeviceInfo | AtlanticDeviceInfo = Field(
        default_factory=AtlanticDeviceInfo, discriminator="model"
    )
    device_designer: str = "atlantic"
