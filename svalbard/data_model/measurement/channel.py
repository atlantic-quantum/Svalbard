"""
Data model for channels. Channels define instrument settings and simple names
that can be used to refer to the channel in the measurement. Channels also define
units and scaling factors that are used to convert the raw data from the instrument
to physical units.
"""

from enum import Enum

import asteval
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from ..instruments import InstrumentModel, InstrumentSetting, SettingType


class LimitAction(str, Enum):
    """
    Enumeration of possible actions when a limit is reached.
    """

    CONTINUE = "Continue"
    NOTHING = "Nothing"
    STOP = "Stop at current value"
    STOP_RESET = "Stop go to init/final value"


class Channel(BaseModel):
    """
    Data model for channels. Channels define instrument settings and simple names
    that can be used to refer to the channel in the measurement. Channels also define
    Units and scaling factors that are used to convert the raw data from the instrument
    to physical units.

    Note on gain, offset and amplification:
        The raw data from the instrument is converted to physical units using the
        following formula:

            raw_data = (physical_values * gain + offset) * amplification

    Args:
        name (str):
            Name of the channel. This name is used to refer to the channel other places
            within the measurement. The name should be pythonic (lowercase, no spaces,
            no special characters).
        instrument_identity (str):
            Identity of the instrument that the channel is connected to. points to an
            an instrument id as defined in the Instrument model.
        instrument_setting_name (str):
            Name of the instrument setting that the channel is connected to. Points to
            an instrument setting name as defined in the InstrumentSetting model.
        unit_physical (str):
            Physical unit of the channel. This is the unit that the data is converted to
            after the raw data is read from the instrument.
        gain (float):
            Gain of the channel. This is used to convert the raw data from the
            instrument to physical units. default is 1.0
        offset (float):
            Offset of the channel. This is used to convert the raw data from the
            instrument to physical units. default is 0.0
        amplification (float):
            Amplification of the channel. This is used to convert the raw data from the
            instrument to physical units. default is 1.0
        high_limit (float):
            High limit of the channel. If the data exceeds this limit, the limit action
            will be performed. default is np.inf
        low_limit (float):
            Low limit of the channel. If the data exceeds this limit, the limit action
            will be performed. default is -np.inf
        limit_action (LimitAction):
            Action to perform if the data exceeds the high or low limit. default is
            CONTINUE
        dtype (str):
            Data type of the channel. This is the data type that is used to create the
            shared memory array that the data is stored in. default is "float"

    """

    model_config = ConfigDict(ser_json_inf_nan="constants")

    name: str
    instrument_identity: str
    instrument_setting_name: str  # instrument setting object instead?
    unit_physical: str
    gain: float = 1.0
    offset: float = 0.0
    amplification: float = 1.0
    high_limit: float = np.inf
    low_limit: float = -np.inf
    limit_action: LimitAction = LimitAction.CONTINUE
    dtype: SettingType = SettingType.FLOAT

    @field_validator("name")
    @classmethod
    def name_must_be_pythonic(cls, v):
        """
        Validator that checks that the name is pythonic (lowercase, no spaces,
        no special characters, not a reserved keyword).
        """
        if not asteval.valid_symbol_name(v):
            v = cls.make_pythonic(v)
            if not asteval.valid_symbol_name(v):
                raise ValueError(f"{v} is not a valid python symbol name")
        return v

    @field_validator("high_limit", mode="before")
    @classmethod
    def high_limit_validation(cls, h):
        """
        Validator that checks that the high_limit is set to np.inf if currently
        None
        """
        if h is None:
            return np.inf
        return h

    @field_validator("low_limit", mode="before")
    @classmethod
    def low_limit_validation(cls, h):
        """
        Validator that checks that the low_limit is set to -np.inf if currently None
        """
        if h is None:
            return -np.inf
        return h

    @staticmethod
    def make_pythonic(name: str) -> str:
        """Makes name a valid python variable name

        Args:
            name: name to be made valid

        Returns:
            Valid name
        """
        return (
            name.replace(".", "_")
            .replace("$", "_")
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )

    @classmethod
    def from_instrument_model_and_setting(  # pylint: disable=too-many-arguments
        cls,
        name: str,
        model: InstrumentModel,
        setting: InstrumentSetting,
        unit: str | None = None,
        gain: float = 1.0,
        offset: float = 0.0,
        amplification: float = 1.0,
        dtype: SettingType | str | None = None,
    ) -> "Channel":
        """
        create a channel from an instrument model and setting

        Args:
            name (str):
                Name of the channel. This name is used to refer to the channel in other
                data models for measurements.
            model (InstrumentModel):
                Instrument model that the channel is connected to. used to determine
                the instrument identity.
            setting (InstrumentSetting):
                Instrument setting that the channel is connected to. used to determine
                instrument setting name and unit.
            unit (str, optional):
                Physical unit of the channel. This is the unit that the data is
                converted to after the raw data is read from the instrument.
                Defaults to None, in which case the instrument unit
                (from instrument settings) is used.
            gain (float):
                Gain of the channel. This is used to convert the raw data from the
                instrument to physical units. default is 1.0
            offset (float):
                Offset of the channel. This is used to convert the raw data from the
                instrument to physical units. default is 0.0
            amplification (float):
                Amplification of the channel. This is used to convert the raw data from
                the instrument to physical units. default is 1.0
            dtype (SettingType, str, optional):
                Data type of the channel. If supplied as str it must be a valid
                SettingType value, default is None, in which case the dtype of the
                InstrumentSetting is used.

        """
        model.get_setting(setting.name)  # raises value error if setting does not exist
        if isinstance(dtype, str):
            dtype = SettingType(dtype)
        return Channel(
            name=name,
            instrument_identity=model.identity,
            instrument_setting_name=setting.name,
            unit_physical=unit or setting.unit,
            gain=gain,
            offset=offset,
            amplification=amplification,
            dtype=dtype or setting.dtype,
        )

    def get_instrument_setting(
        self, instruments: dict[str, InstrumentModel]
    ) -> InstrumentSetting:
        """
        get the instrument setting of the channel

        Args:
            instruments (list[InstrumentModel]):
                list of instrument models to search for the instrument identity

        Raises:
            ValueError: if the instrument identity is not found in the list of
                instrument models
            ValueError: if the instrument setting name is not found in the list of
                instrument settings for the identified instrument model

        Returns:
            str: instrument unit
        """
        try:
            instrument = instruments[self.instrument_identity]
        except KeyError as err:
            raise ValueError(
                f"Could not find instrument {self.instrument_identity}"
            ) from err
        try:
            return instrument.settings[self.instrument_setting_name]
        except KeyError as err:
            estr = (
                f"Could not find setting {self.instrument_setting_name} in"
                + f" instrument {self.instrument_identity}"
            )
            raise ValueError(estr) from err

    def get_instrument_unit(self, instruments: dict[str, InstrumentModel]) -> str:
        """
        get the instrument unit of the channel

        Args:
            instruments (list[InstrumentModel]):
                list of instrument models to search for the instrument identity

        Raises:
            ValueError: if the instrument identity is not found in the list of
                instrument models or if the instrument setting name is not found in
                in the list of instrument settings for the identified instrument model

        Returns:
            str: instrument unit
        """
        return self.get_instrument_setting(instruments).unit

    @property
    def label(self) -> str:
        """
        return label with unit for the channel
        """
        return f"{self.name} ({self.unit_physical})"

    @staticmethod
    def default_name(instrument_id: str, setting_name: str) -> str:
        """
        Create a default channel name from the instrument id and setting name

        Args:
            instrument_id (str): identity of the instrument
            setting_name (str): name of the setting

        Returns:
            str: default channel name in the form of "{instrument_id}___{setting_name}"
        """
        return f"{instrument_id}___{setting_name}"

    @staticmethod
    def get_instrument_and_setting(default_channel_name: str) -> tuple[str, str]:
        """
        Get the instrument identity and setting name from the default channel name

        Args:
            default_channel_name (str): default channel name

        Returns:
            tuple[str, str]: instrument identity and setting name

        Raises:
            ValueError: if the channel name is not in the form
                {instrument_id}___{setting_name}
        """
        split_name = default_channel_name.split("___", 1)
        if len(split_name) != 2:
            # fall back on the old format (split on '/') for backwards compatibility
            # ! revert to new format once aq_measurement master is updatad
            split_name = default_channel_name.split("/", 1)
            if len(split_name) != 2:
                raise ValueError(
                    f"Channel name {default_channel_name} is not in the form"
                    + " {instrument_id}/{setting_name}"
                )
            # raise ValueError(
            #     f"Channel name {default_channel_name} is not in the form"
            #     + " {instrument_id}___{setting_name}"
            # )
        return split_name[0], split_name[1]
