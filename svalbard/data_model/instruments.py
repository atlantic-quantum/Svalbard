"""
Data models for instruments, including settings, connections, and hardware
"""

from abc import abstractmethod
from enum import Enum
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ..typing import TSettingValue
from ._setting_types import LIST_SETTING_TYPES, SettingType


class SweepCapability(Enum):
    """Enumeration of different sweep capabilities for instrument settings"""

    NONE = "none"
    SW_GENERATED_SW_SET = "sw_generated_sw_set"
    SW_GENERATED_HW_SET = "sw_generated_hw_set"
    HW_GENERATED_HW_SET = "hw_generated_hw_set"


class Drain(BaseModel):
    """
    Drain data model, used to specify which instrument setting is used as a drain.
    i.e. settings of other instruments that should be set to the value of this setting

    Args:
        identity (str): instrument identity of the setting
        setting (str): name of instrument setting used as drain
    """

    model_config = ConfigDict(frozen=True)
    identity: str
    setting: str


class InstrumentSetting(BaseModel):
    """
    Instrument setting data model

    Args:
        name (str):
            name of the setting
        value (TSettingValue | None):
            value of the setting, can be a single value or a list of values.
            A single value can be a float, int, bool, Enum, or str.
            A list of values can be a list of floats, ints, bools, Enums, or strs.
            Complex values are stored as strings, e.g. "1.0+2.0j"
        dtype (SettingType):
            type of the setting, defaults to "SettingType.None"
        unit (str):
            unit of the setting, defaults to ""
        sweep_capability (SweepCapability):
            sweep capability of the setting, defaults to "sw_generated_sw_set"
        drains (list[Drain]):
            list of drains, i.e. settings of other instruments that should be set to the
            value of this setting, when this setting is changed, defaults to []
    """

    name: str
    value: TSettingValue | None
    dtype: SettingType = SettingType.NONE
    unit: str = ""
    sweep_capability: SweepCapability = SweepCapability.SW_GENERATED_SW_SET
    drains: list[Drain] = []
    # todo should this have shape e.g. for settings that are waveforms or traces?

    def get_value_string(self) -> str:
        """Get value as string for the instrument setting, for use in UI

        Returns:
            str: String describing value of instrument setting
        """
        return SettingType.get_setting_type_class(self.dtype).get_value_string(
            self.value, self.unit
        )

    @model_validator(mode="before")  # type: ignore
    @classmethod
    def validate_value_and_type(cls, values: dict):
        """
        if type is given, cast value to that type
        if type is none, try to infer type from value
        note that type is always given when deserializing from json

        Args:
            values (dict): the values used to create the model

        Returns:
            dict: parsed values
        """
        if isinstance(values["value"], np.generic):
            values["value"] = values["value"].item()
        if isinstance(values["value"], np.ndarray):
            values["value"] = values["value"].tolist()

        values["dtype"] = SettingType.determine_setting_type(
            values["value"], values.get("dtype", None)
        )
        setting_type = values["dtype"]
        if values.get("value", None) is not None:
            _check_value_correct_type(setting_type, values["value"], values["name"])
            values["value"] = SettingType.get_setting_type_class(setting_type).to_value(
                values["value"]
            )
        return values

    def add_drain(self, target_instrument: str, target_setting: str) -> None:
        """Add a drain to the instrument setting

        Args:
            target_instrument (str): identity of the target instrument
            target_setting (str): name of the target setting on the target instrument
        """
        drain = Drain(identity=target_instrument, setting=target_setting)
        if drain not in self.drains:
            self.drains.append(drain)

    def remove_drain(self, target_instrument: str, target_setting: str) -> None:
        """Remove a drain from the instrument setting

        Args:
            target_instrument (str): identity of the target instrument
            target_setting (str): name of the target setting on the target instrument
        """
        drain = Drain(identity=target_instrument, setting=target_setting)
        if drain in self.drains:
            self.drains.remove(drain)


class StartupBehaviour(Enum):
    """Enum for startup behaviour"""

    NO_ACTION = 0
    READ_SETTINGS = 1
    WRITE_SETTINGS = 2


class TerminationCharater(Enum):
    """Enum for termination characters"""

    CR = "\r"
    LF = "\n"
    CRLF = "\r\n"
    NONE = ""


class ConnectionInterface(Enum):
    """Enum for supported connection interfaces"""

    GPIB = "GPIB"
    TCPIP = "TCPIP"
    LABONE = "LABONE"
    USB = "USB"
    PXI = "PXI"
    SERIAL = "SERIAL"
    VISA = "VISA"
    OTHER = "OTHER"
    NONE = "NONE"


class InstrumentConnection(BaseModel):
    """Instrument connection settings

    # todo have Simon check this especially the defaults and docstring

    Args:
        interface (ConnectionInterface):
            interface used to connect to the instrument
        address (str):
            address of the instrument, e.g. "TCPIP::xxx.xxx.xxx.xxx::INSTR"
        server (str):
            server address, e.g. "localhost"
        startup (StartupBehaviour):
            startup behaviour, defaults to "NO_ACTION"
        lock (bool):
            lock the instrument and keep it from communicating using other methods,
            defaults to False
        timeout (float):
            timeout in seconds, defaults to 0.0
        termination_character (TeminationCharater):
            character used to terminate a command, defaults to "LF"
        send_end_on_write (bool):
            send end character on write, defaults to True
        lock_visa_resource (bool):
            lock the visa resource, defaults to False
            # todo overlap with lock?
        suppress_end_bit_termination_on_read (bool):
            suppress end bit termination on read, defaults to False
        use_tcp_port (int):
            tcp port to use, defaults to 0
        baud_rate (int):
            baud rate, defaults to 9600
        data_bits (int):
            data bits, defaults to 8
        stop_bits (int):
            stop bits, defaults to 1
        parity (int):
            parity, defaults to 0
        gpib_board_number (int):
            gpib board number, defaults to 0
        send_gpib_go_local_at_close (bool):
            send gpib go local at close, defaults to True

    """

    interface: ConnectionInterface = ConnectionInterface.LABONE
    address: str = ""
    server: str = ""
    startup: StartupBehaviour = StartupBehaviour.NO_ACTION
    lock: bool = False
    timeout: float = 0.0
    termination_character: TerminationCharater = TerminationCharater.LF
    send_end_on_write: bool = True
    lock_visa_resource: bool = False
    suppress_end_bit_termination_on_read: bool = False
    use_tcp_port: bool = False
    tcp_port: int = 0
    baud_rate: int = 9600
    data_bits: int = 8
    stop_bits: int = 1
    parity: int = 0
    gpib_board_number: int = 0
    send_gpib_go_local_at_close: bool = True


class InstrumentModel(BaseModel):
    """
    Instrument data model

    Args:
        hardware (str):
            hardware type of the instrument, e.g. HDAWG,
            used to identify the driver for the instrument
        version (str):
            driver version used to communicate with the instrument for an experiment
            e.g. 1.0.3
        model (str):
            model name of the instrument, e.g. HDAWG8
        serial (str):
            serial number of the instrument, e.g. DEV8568
        identity (str):
            human readable identity of the instrument, e.g. flux_q0-q7
            this value is not used for anything internally, but is useful for the user
            instead the hardware, model, and serial are used to identify the instrument
        connection (InstrumentConnection):
            connection settings for the instrument.
            See InstrumentConnection documentation for details.
        settings (dict[str, InstrumentSetting]):
            dictionary of instrument settings, keyed by setting name.
            defaults to {}
            See InstrumentSetting documentation for details.
        allow_new_settings (bool):
            allow new settings to be added to the settings dictionary.
            defaults to False

    """

    hardware: str = "unknown"
    version: str = "unknown"
    model: str = "unknown"
    serial: str = "unknown"
    identity: str = "unknown"
    connection: InstrumentConnection = InstrumentConnection()
    settings: dict[str, InstrumentSetting] = {}
    allow_new_settings: bool = False

    @field_validator("settings")
    @classmethod
    def validate_settings(cls, settings: dict[str, InstrumentSetting]):
        """Validate that the keys of settings are the same as their names"""
        for key, setting in settings.items():
            assert key == setting.name
        return settings

    def get_id_string(self) -> str:
        """Get a string that uniquely identifies the instrument"""
        return f"{self.hardware} - {self.identity} - {self.serial}"

    def has_setting(self, setting_name: str) -> bool:
        """
        Returns:
            True: If the InstrumentModel has an InstrumentSetting name 'setting_name'
            False: Otherwise
        """
        return setting_name in self.settings

    def update_setting(
        self, setting: str, value: TSettingValue, unit: str | None = None
    ):
        """Update the settings of the instrument with new settings

        Args:
            setting (str):
                name of the setting to update
            value (str):
                new value for the setting
            unit (str, optional):
                new unit for the setting. Default: None -> unit is unchanged
        """
        if value is None:
            raise ValueError("Value of InstrumentSetting may not be set to None")
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, np.generic):
            value = value.item()
        isetting = self.get_setting(setting)
        _check_value_correct_type(isetting.dtype, value, setting, self.identity)
        isetting.value = value
        if unit is not None:
            isetting.unit = unit

    def add_setting(self, setting: InstrumentSetting):
        """Add a new setting to the instrument

        Args:
            setting (InstrumentSetting):
                new setting to add to the instrument

        Raises:
            ValueError: if setting already exists or new settings are not allowed
        """
        if not self.allow_new_settings:
            raise ValueError(
                f"New settings are not allowed for this model {self.identity}"
            )
        if self.has_setting(setting.name):
            raise ValueError(f"Setting {setting.name} already exists")
        self.settings[setting.name] = setting

    def get_setting(self, setting: str) -> InstrumentSetting:
        """Get a setting from the instrument

        Args:
            setting (str):
                name of the setting to get

        Returns:
            InstrumentSetting:
                setting from the instrument

        Raises:
            ValueError: if the setting does not exist.
        """
        if not self.has_setting(setting):
            raise ValueError(
                f"Instrument model: '{self.identity}' doesn't have setting: '{setting}'"
            )
        return self.settings[setting]

    def needs_drainage(self, setting: str) -> bool:
        """
        Args:
            setting (str): name of the setting to check

        Returns:
            bool:
                True if the setting needs to be drained to from other instruments.
                False otherwise.
        """
        instrument_setting = self.get_setting(setting)
        if instrument_setting.dtype in [SettingType.MODEL, SettingType.LIST_MODEL]:
            return True
        return False


class SupportsInstrumentModel(Protocol):
    """
    A Protocol for classes that support conversion to and from InstrumentModel objects
    and lists of InstrumentSetting representing the settings of the object.

    see https://peps.python.org/pep-0544/ for more info on Protocols
    """

    @abstractmethod
    def to_instrument_settings(self) -> list[InstrumentSetting]:
        """
        Convert the object to a list of InstrumentSetting objects

        Returns:
            list[InstrumentSetting]:
                List of InstrumentSetting representing the settings of the object
        """
        raise NotImplementedError

    @abstractmethod
    def to_instrument_model(self) -> InstrumentModel:
        """
        Convert the object to an InstrumentModel object

        Returns:
            InstrumentModel:
                InstrumentModel representation of the object
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_instrument_model(cls, instrument: InstrumentModel):
        """
        Convert an InstrumentModel object to an instance of the class

        Args:
            instrument (InstrumentModel):
                InstrumentModel object to convert
        """
        raise NotImplementedError


def _check_value_correct_type(
    setting_type: SettingType,
    value: TSettingValue,
    setting_name: str,
    instrument_name: str = "",
):
    """Minimal type checking for setting values

    Args:
        setting_type (SettingType):
            type of the settings value
        value (TSettingValue):
            value of the setting
        setting_name (str):
            name of the setting, used for error messages
        instrument_name (str, optional):
            name of instrument the setting belongs to. Defaults to "",
            used for error messages.

    Raises:
        ValueError: If the setting type is a list and the value is not a list
        ValueError: If the setting type is not a list and the value is a list
    """
    istr = f"for instrument '{instrument_name}'" if instrument_name else ""
    sstr = f"Value of setting '{setting_name}' "
    if setting_type in LIST_SETTING_TYPES:
        if not isinstance(value, list):
            raise ValueError(f"{sstr}{istr} must be a list, got '{value}'")
    else:
        if isinstance(value, list):
            raise ValueError(f"{sstr}{istr} must be a single value, got '{value}'")
