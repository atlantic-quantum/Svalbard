"""
Data models for instruments, including settings, connections, and hardware
"""
from abc import abstractmethod
from enum import Enum
from typing import Protocol

from pydantic import (
    BaseModel,
    StrictBool,
    StrictFloat,
    StrictInt,
    root_validator,
    validator,
)


class SettingType(Enum):
    """Enumeration for instrument setting types"""

    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    ENUM = "enum"
    STR = "str"
    NONE = "none"
    COMPLEX = "complex"


_setting_type_map = {
    SettingType.FLOAT: float,
    "float": float,
    SettingType.INT: int,
    "int": int,
    SettingType.BOOL: bool,
    "bool": bool,
    SettingType.STR: str,
    "str": str,
    SettingType.COMPLEX: complex,
}


class InstrumentSetting(BaseModel):
    """
    Instrument setting data model

    Args:
        name (str):
            name of the setting
        value (float | int | bool):
            value of the setting
        type (str):
            type of the setting, defaults to "float"
        unit (str):
            unit of the setting, defaults to ""
    """

    name: str
    value: StrictBool | StrictInt | StrictFloat | Enum | str
    dtype: SettingType = SettingType.NONE
    unit: str = ""
    # todo should this have shape e.g. for settings that are waveforms or traces?

    @root_validator(pre=True)
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
        values["dtype"] = values.get("dtype", SettingType.NONE)

        if values["dtype"] == SettingType.ENUM or values["dtype"] == "enum":
            values["dtype"] = SettingType.ENUM
            if isinstance(values["value"], Enum):
                values["value"] = values["value"].value
            return values

        if values["dtype"] == SettingType.COMPLEX or values["dtype"] == "complex":
            values["dtype"] = SettingType.COMPLEX
            if isinstance(values["value"], complex):
                values["value"] = f"{values['value'].real}+{values['value'].imag}j"
            return values

        if values["dtype"] != SettingType.NONE:
            try:
                values["value"] = _setting_type_map[values["dtype"]](values["value"])
            except (ValueError, TypeError) as err:
                raise ValueError(
                    f"Value {values['value']} could not be cast to {values['dtype']}"
                ) from err
            return values

        if isinstance(values["value"], complex):
            values["dtype"] = SettingType.COMPLEX
            values["value"] = f"{values['value'].real}+{values['value'].imag}j"
        elif isinstance(values["value"], float):
            values["dtype"] = SettingType.FLOAT
        elif isinstance(values["value"], bool):
            # bool is a subclass of int, so this must be checked first
            values["dtype"] = SettingType.BOOL
        elif isinstance(values["value"], int):
            values["dtype"] = SettingType.INT
        elif isinstance(values["value"], Enum):
            values["dtype"] = SettingType.ENUM
            values["value"] = values["value"].value
        elif isinstance(values["value"], str):
            values["dtype"] = SettingType.STR
        else:
            raise ValueError(
                f"Could not infer type from value {values['value']}, "
                f"please specify type explicitly"
            )
        return values


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

    """

    hardware: str = "unknown"
    version: str = "unknown"
    model: str = "unknown"
    serial: str = "unknown"
    identity: str = "unknown"
    connection: InstrumentConnection = InstrumentConnection()
    settings: dict[str, InstrumentSetting] = {}

    @validator("settings")
    def validate_settings(cls, settings: dict[str, InstrumentSetting]):
        """Validate that the keys of settings are the same as their names"""
        for key, setting in settings.items():
            assert key == setting.name
        return settings


class SupportsInstrumentModel(Protocol):
    """
    A Protocol for classes that support conversion to and from InstrumentModel objects
    and lists of InstrumentSetting representing the settings of the object.

    see https://peps.python.org/pep-0544/ for more info on Protocols
    """

    @abstractmethod
    def to_insturment_settings(self) -> list[InstrumentSetting]:
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
