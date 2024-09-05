"""
Pydantic models that defines the data file structure
"""

import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeVar, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..typing import TSettingValue, TSettingValueSweep
from .compiler import AQCompilerDataModel, AQCompilerDataModelV2, CompilerDataModel
from .device import Device
from .instruments import InstrumentModel, InstrumentSetting
from .measurement import Channel, LogChannel, Measurement, StepItem
from .memory_models import SharedMemoryIn, SharedMemoryOut

# todo: measurement handle is not saved with metadata, should it be?


class Flags(StrEnum):
    """
    Enum for flags that can be set on a data file metadata
    """

    ORANGE = "orange"
    RED = "red"
    PURPLE = "purple"
    BLUE = "blue"
    YELLOW = "yellow"
    GREEN = "green"
    GRAY = "gray"
    NONE = "none"


class MeasurementHandle(BaseModel):
    """Model for storing a handle to a measurement for ipc communication"""

    handle: uuid.UUID

    @classmethod
    def new(cls):
        """Create a new MeasurementHandle"""
        return cls(handle=uuid.uuid1())


class Data(BaseModel):
    """Pydantic model for aqcuired data

    Args:
        handle (MeasurementHandle):
            unique identifier for a given data_file
        datasets (list[DataSet]):
            list of datasets associated with the data_file
        files (list[Path]):
            list of relative paths from repo dir to files associated with the data_file

    """

    class DataSet(BaseModel):
        """Pydantic model for datasets"""

        name: str
        memory: SharedMemoryOut

        @classmethod
        def from_array(cls, name: str, array: np.ndarray) -> "Data.DataSet":
            """
            Constructs a DataSet from a name and numpy array of values.

            Args:
                name (str):
                    The name of the dataset.
                array (np.ndarray):
                    The array of values for the dataset.

            Returns:
                Data.DataSet:
                    DataSet object created from numpy array.
            """
            dataset = cls(
                name=name,
                memory=SharedMemoryOut.from_memory_in(
                    SharedMemoryIn(
                        dtype=array.dtype,
                        shape=array.shape,
                    ),
                ),
            )
            dataset.memory.to_array()[:] = array
            return dataset

    handle: MeasurementHandle
    datasets: list[DataSet]
    files: list[Path] = []

    def get_dataset(self, name: str) -> DataSet:
        """
        Get the dataset with the given name

        Args:
            name (str):
                name of the dataset to get

        Returns:
            DataSet:
                dataset with the given name

        Raises:
            ValueError:
                if no dataset with the given name exists
        """
        datasets = {dataset.name: dataset for dataset in self.datasets}
        try:
            return datasets[name]
        except KeyError:
            raise ValueError(f"Dataset with name {name} not found")

    @classmethod
    def from_measurement(cls, measurement: Measurement) -> "Data":
        """
        Create a Data object from a measurement object. Datasets are created for
        both step channels, relations and log channels. The values of the step channels
        and relations have already been calculated and are stored in the memories
        associated with each corresponding dataset.

        Args:
            measurement (Measurement):
                measurement object to create Data object from

        Returns:
            Data:
                Data object created from measurement object,
        """
        datasets = [
            Data.DataSet.from_array(name=channel_name, array=values)
            for (channel_name, values) in measurement.calculate_dataset_values().items()
        ]
        log_shapes = measurement.log_shapes
        for log in measurement.log_channels:
            datasets.append(
                Data.DataSet(
                    name=log.name,
                    memory=SharedMemoryOut.from_memory_in(
                        SharedMemoryIn(
                            dtype=measurement.get_channel(log.name).dtype.value,
                            shape=log_shapes[log.name],
                        ),
                    ),
                )
            )

        return Data(handle=MeasurementHandle.new(), datasets=datasets)


class BaseMetaData(BaseModel):
    """
    Pydantic model for BaseMetaData, the parent model for all other Metadata models

    Args:
        metadata_model (Literal):
            what type of metadata is associated with the data. Default empty string.
            Should not be changed by the user.
        name (str):
            name assoicated with the data_file the metadata belongs to
        user (str):
            user who created the data_file
        tags (list[str]):
            list of tags associated with the data_file, defaults to []
        flag (Flags):
            flag set on the datafile metadata. Defaults to none.
        date (datetime):
            date and time the data_file was created, should not be changed by the user,
            defaults to datetime.now() in milliseconds (isoformat)
        version (str):
            version of the data_file format, should not be changed by the user,
            defaults to "0.0.1"
        data_path (Path):
            path on object storage where the data is stored,
            populated by the data server, should not be changed by the user
        files (list[Path]):
            list of names of all files associated with a given data_file

    """

    metadata_model: Literal[""] = ""
    name: str = ""
    user: str = "unknown"
    tags: list[str] = []
    flag: Flags = Flags.NONE
    date: datetime = Field(
        default_factory=lambda: datetime.fromisoformat(
            datetime.now().isoformat(timespec="milliseconds")
        )
    )
    version: str = "0.0.1"
    data_path: Path | None = None  # todo AQC-289 - change to PurePosixPath
    comment: str = ""
    files: list[Path] = []


class MetaData(BaseMetaData):
    """
    Pydantic model for Metadata

    Args:
        metadata_model (Literal):
            what type of metadata is associated with the data. Default "measurement".
            Should not be changed by the user.
        name (str):
            name assoicated with the data_file the metadata belongs to
        user (str):
            user who created the data_file
        station (str):
            station (e.g. atlantis2) where the data_file was created
        tags (list[str]):
            list of tags associated with the data_file, defaults to []
        flag (Flags):
            flag set on the datafile metadata. Defaults to none.
        device (Device):
            data model description of the device measured to acquire the data
            The device model has the following fields:
                mask (str), wafer (str), chip (str), die (str).
                see Device documentation for details.
        instruments (dict[str, InstrumentModel]):
            dictionary of instrument models, keyed by instrument identity,
            defaults to {}
                see InstrumentModel documentation for details on its structure
        compiler_data (CompilerDataModel | AQCompilerDataModel):
            data model description of the compiler and source files used to generate
            instrument specific code, defaults to AQCompilerDataModel
        date (datetime):
            date and time the data_file was created, should not be changed by the user,
            defaults to datetime.now() in milliseconds (isoformat)
        cooldown (datetime):
            date of the cooldown the data belongs to, do not specify time.
            defaults to 1900-1-1
        version (str):
            version of the data_file format, should not be changed by the user,
            defaults to "0.0.4"
        data_path (Path):
            path on object storage where the data is stored,
            populated by the data server, should not be changed by the user
        files (list[Path]):
            list of names of all files associated with a given data_file
        data_size (list[int]):
            dimensions of the data stored in the data_file

    """

    model_config = ConfigDict(validate_assignment=True)

    metadata_model: Literal["measurement"] = "measurement"
    station: str = "unknown"
    device: Device = Device()
    instruments: dict[str, InstrumentModel] = {}
    cooldown: datetime = Field(default_factory=lambda: datetime(1900, 1, 1))

    compiler_data: CompilerDataModel | AQCompilerDataModel | AQCompilerDataModelV2 = (
        Field(default_factory=AQCompilerDataModel, discriminator="compiler")
    )
    measurement: Measurement = Measurement.create_empty()
    version: str = "0.1.3"
    data_size: list[int] = []

    @field_validator("instruments")
    @classmethod
    def validate_instruments(cls, instruments: dict[str, InstrumentModel]):
        """Validate that the keys of instruments are the same as their identities"""
        for key, instrument_model in instruments.items():
            assert key == instrument_model.identity
        return instruments

    def add_instrument_model(self, instrument_model: InstrumentModel) -> None:
        """
        Add an instrument model to the metadata

        Args:
            instrument_model (InstrumentModel):
                instrument model to add

        Raises:
            ValueError:
                if instrument with the same identity already exists
        """
        if self.has_instrument_model(instrument_model.identity):
            raise ValueError(
                f"Instrument with identity {instrument_model.identity} already exists"
            )
        self.instruments[instrument_model.identity] = instrument_model

    def has_instrument_model(self, instrument_model_identity: str) -> bool:
        """
        Returns:
            True: if metadata model has an instrument model with identity
                'instrument_model_identity'.
            False: Otherwise
        """
        return instrument_model_identity in self.instruments

    def get_instrument_model(self, instrument_model_identity: str) -> InstrumentModel:
        """
        Get the instrument model with the identity 'instrument_model_identity'

        Args:
            instrument_model_identity (str):
                identity of the instrument model to get

        Returns:
            InstrumentModel:
                instrument model with the given identity

        Raises:
            ValueError:
                if no instrument model with the given identity exists
        """
        if not self.has_instrument_model(instrument_model_identity):
            raise ValueError(
                f"Instrument with identity {instrument_model_identity} not in metadata"
            )
        return self.instruments[instrument_model_identity]

    def configure_setting(
        self,
        instrument: str,
        setting: str,
        values: TSettingValue | TSettingValueSweep,
        unit: str = "",
        index: int | None = None,
        hw_swept: bool = False,
        list_setting: bool = False,
    ) -> None:
        """
        Configures the InstrumentSetting 'setting' of the InstrumentModel
        'instrument'. If 'is_sweep' is True, the setting is swept over the 'values'
        provided. If 'is_sweep' is False, the setting is set to the 'single' value
        provided. Note that this 'single' value may in some cases be a list of values.

        Args:
            instrument (str):
                instrument identity
            setting (str):
                setting name
            values (TSettingValue | TSettingValueSweep):
                values to set or sweep for the parameter.
            unit (str):
                unit of values, defaults to "", only used if the setting for add step
                does not already exist in the instrument model. REMOVE once
                ParamGenerators are addressed.
            index (int):
                The index used to determine the order this step item is stepped in with
                respect to other step items. If not provided, the step item is given
                the highest current index + 1.
            hw_swept (bool):
                if the step item is should be hardware swept
            list_setting (bool):
                True if the setting dtype is a list of values, False otherwise.
                defaults to False.
                ! REMOVE once ParamGenerators are addressed.
        """
        # for when ParamGenerators are removed
        # instr_setting = self.get_instrument_model(instrument).get_setting(setting)
        # list_setting = instr_setting.dtype in LIST_SETTING_TYPES
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if isinstance(values, tuple):
            values = list(values)
        stepped_list_setting = list_setting and values and isinstance(values[0], list)
        stepped_setting = not list_setting and isinstance(values, list)
        if not self.get_instrument_model(instrument).has_setting(setting):
            stepped_setting = True  # ! ParamsGenerators hack
        if stepped_list_setting or stepped_setting:
            self.add_step(
                instrument, setting, values, unit=unit, index=index, hw_swept=hw_swept
            )
        else:
            self.update_setting(instrument, setting, values, unit=unit)

        if self.get_instrument_model(instrument).needs_drainage(setting):
            # if needs_drainage is True, the setting is a model or list_model and the
            # values are strings of instrument identities that need to be drained.
            drained_instruments: set[str] = set(_flatten_list(values))
            self._add_drains(drained_instruments, instrument, setting)

    # ! this remains the API method until ParamGenerators are removed.
    def add_step(
        self,
        instrument: str,
        setting: str,
        values: TSettingValue | TSettingValueSweep,
        unit: str | None = None,
        index: int | None = None,
        hw_swept: bool = False,
    ) -> None:
        """
        Add a StepItem to MetaData if array of values or update instrument setting to
        single value.

        Args:
            instrument (str):
                instrument identity
            setting (str):
                setting string
            values (TSettingValue | TSettingValueSweep):
                values to sweep / create StepItem for or value to set setting to
            unit (str):
                unit of values, defaults to "", only used if the setting for add step
                does not already exist in the instrument model. REMOVE once
                ParamGenerators are addressed.
            index (int):
                The index used to determine the order this step item is stepped in with
                respect to other step items. If not provided, the step item is given
                the highest current index + 1.
            hw_swept (bool):
                if the step item is should be hardware swept
        """
        if not self.has_instrument_model(instrument):
            raise ValueError(f"Instrument '{instrument}' not in metadata")
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if isinstance(values, tuple):
            values = list(values)
        if not isinstance(values, list):
            values = [values]
        init_value = values[0]
        if self.get_instrument_model(instrument).has_setting(setting):
            self.update_setting(instrument, setting, init_value, unit=unit)
        # remove below once ParamGenerators addressed
        else:
            self.add_setting(instrument, setting, init_value, unit=unit or "")
        # remove above once ParamGenerators addressed
        if len(values) > 1:
            self._add_step(
                Channel.default_name(instrument, setting), values, index, hw_swept
            )

    def _add_drains(
        self,
        drained_instruments: set[str],
        target_instrument: str,
        target_setting: str,
    ) -> None:
        """
        Called when type of target_setting is MODEL or LIST_MODEL and add drains to
        the object setting of the source_instruments. The drains are to the
        target_setting of the target_instrument.

        Args:
            drained_instruments (set[str]):
                set of instrument identities to add drains to.
            target_instrument (str):
                Added drains will target this instrument
            target_setting (str):
                Added drains will target this setting of the target_instrument
        """
        for drained_instrument in drained_instruments:
            instrument = self.get_instrument_model(drained_instrument)
            setting = instrument.get_setting(drained_instrument)
            setting.add_drain(target_instrument, target_setting)

    def _add_step(
        self,
        step_name: str | StepItem,
        values: TSettingValueSweep | None = None,
        index: int | None = None,
        hw_swept: bool = False,
    ) -> None:
        """
        Add a StepItem to MetaData, either directly or as an array of values

        Args:
            step_name (str | StepItem):
                step item or name of step items
            values (list[bool | int | float | Enum] | None):
                values to sweep create StepItem for or value for setting, should be none
                if step item is being added directly
            index (int):
                StepItem index
            hw_swept (bool):
                if the step item should be hw swepts.
        """
        try:
            self.measurement.get_channel(step_name)
        except ValueError:
            channel = self._create_channel(step_name)
            self.measurement.add_channel(channel)
        if isinstance(step_name, StepItem):
            self.measurement.add_step_item(step_name)
        else:
            if values is None:
                raise ValueError("No values provided when creating a StepItem.")
            self.measurement.add_step(step_name, values, index, hw_swept=hw_swept)

    def add_log(self, log_name: str | LogChannel) -> None:
        """Adds a log channel to metadata.

        Args:
            log_name (str | LogChannel):
                name of log channel to add or a LogChannel object to add. If a channel
                with log_name does not exist one will be created provided the name is
                in the format
        """
        try:
            self.measurement.add_log(log_name)
        except ValueError:  # channel does not exist
            channel = self._create_channel(log_name)
            self.measurement.add_channel(channel)
            self.measurement.add_log(log_name)

    def _create_channel(self, channel_name: str | LogChannel | StepItem) -> Channel:
        """
        Creates a Channel from a channel_name, the name must be in the format

            f"{instrument_id}___{setting_name}"

        Args:
            channel_name (str | LogChannel | StepItem):
                Name of channel to create. if channel_name is not string the 'name' of
                the LogChannel/StepItem object is used instead.

        Returns:
            Channel, with name channel_name.

        Raises:
            ValueError: if channels instrument identity is not in metadata
        """
        if not isinstance(channel_name, str):
            channel_name = channel_name.name
        instrument_id, setting_name = Channel.get_instrument_and_setting(channel_name)
        if not self.has_instrument_model(instrument_id):
            raise ValueError(f"Instrument {instrument_id} not in metadata")
        instrument_setting = self.get_instrument_model(instrument_id).get_setting(
            setting_name
        )
        return Channel.from_instrument_model_and_setting(
            channel_name, self.get_instrument_model(instrument_id), instrument_setting
        )

    def update_setting(
        self,
        instrument: str,
        setting: str,
        value: TSettingValue,
        unit: str | None = None,
    ) -> None:
        """
        Update the value of an instrument setting

        Args:
            instrument (str):
                instrument identity of the instrument whose setting to update
            setting (str):
                name of the setting to update
            value (TSettingValue):
                value to update the setting to
            unit (str, optional):
                new unit for the setting, default: None -> unit is unchanged

        """
        if not self.has_instrument_model(instrument):
            raise ValueError(f"Instrument {instrument} not in metadata")
        self.get_instrument_model(instrument).update_setting(setting, value, unit)

    # ! the add settings method can be removed once ParamGenerators are addressed.
    def add_setting(
        self,
        instrument: str,
        setting: str,
        value: TSettingValue,
        unit: str = "",
        dtype: str | None = None,
    ) -> None:
        """
        Add a setting to the metadata object

        Args:
            instrument (str):
                instrument identity of the instrument whose setting to update
            setting (str):
                name of the setting to update
            value (TSettingValue):
                value to update the setting to
            unit (str, optional):
                unit of the setting, defaults to ""
            dtype (str, optional):
                data type of the setting, defaults to None
        """
        if not self.has_instrument_model(instrument):
            raise ValueError(f"Instrument {instrument} not in metadata")
        instrument_setting = InstrumentSetting(
            name=setting,
            value=value,
            unit=unit,
            dtype=dtype,  # type: ignore
        )
        self.get_instrument_model(instrument).add_setting(instrument_setting)

    def get_channel_name(self, instrument: str, setting: str) -> str:
        """
        Get the name of a channel from an instrument and setting, if the channel
        exists in the measurement object. If the channel does not exist, a default
        name is returned.

        Args:
            instrument: instrument identity
            setting: setting name

        Returns:
            str: channel name
        """
        for channel in self.measurement.channels:
            if (
                channel.instrument_identity == instrument
                and channel.instrument_setting_name == setting
            ):
                return channel.name
        return Channel.default_name(instrument, setting)


class MaskMetaData(BaseMetaData):
    """
    Pydantic model for mask Metadata

    Args:
        metadata_model (Literal):
            what type of metadata is associated with the data. Default "mask".
            Should not be changed by the user.
        name (str):
            name of the mask associated with the mask_params
        date (datetime):
            date and time the data_file was created, should not be changed by the user,
            defaults to datetime.now() in milliseconds (isoformat)
        data_path (Path):
            path on object storage where the data is stored,
            populated by the data server, should not be changed by the user
        mask_params (dict):
            json-serializable dict which stores the layout parameters of the mask
            (specified in aq_design/masks/mask_params.py)
        chips_params (dict):
            json-serializable dict which stores the layout parameters of all the
            chips on the mask (specified in aq_design/masks/chip_xx.py)
        files (list[Path]):
            list of names of all files associated with the given mask datafile
            (e.g. gds)
    """

    metadata_model: Literal["mask"] = "mask"
    mask_params: dict = {}
    chips_params: dict = {}
    version: str = "0.0.1"


class EMSimMetaData(BaseMetaData):
    """
    Pydantic model for EM Simulation Metadata

    Args:
        metadata_model (Literal):
            what type of metadata is associated with the data. Default "em_simulation".
            Should not be changed by the user.
        name (str):
            name of the EM simulation associated with the results files
        user (str):
            user who created the data_file
        tags (list[str]):
            list of tags associated with the data_file, including type of simulation
            (e.g. "hfss", "maxwell"). Defaults to []
        device (Device):
            data model description of the device measured to acquire the data
            The device model has the following fields:
                mask (str), wafer (str), chip (str), die (str).
                see Device documentation for details.
        date (datetime):
            date and time the data_file was created, should not be changed by the user,
            defaults to datetime.now() in milliseconds (isoformat)
        data_path (Path):
            path on object storage where the data is stored,
            populated by the data server, should not be changed by the user
        files (list[Path]):
            list of names of all results files associated with the given EM simulation
            datafile (e.g. .csv, .eig)
    """

    metadata_model: Literal["em_simulation"] = "em_simulation"
    device: Device = Device()
    version: str = "0.0.1"


class MetaDataDiscriminator(BaseMetaData):
    """Model for discriminating between different metadata models"""

    metadata: Union[MetaData, MaskMetaData, EMSimMetaData] = Field(
        discriminator="metadata_model"
    )


class DataFile(BaseModel):
    """Pydantic model for DataFiles"""

    data: Data | None = None
    # metadata: Union[MetaData, MaskMetaData, EMSimMetaData] = Field(
    #     discriminator="metadata_model"
    # )
    metadata: MetaData


T = TypeVar("T")


def _flatten_list(lst: T | list[T] | list[list[T]]) -> list[T]:
    """Takes a value, a list of values or lists of lists of values and flattens it"""

    def _to_list(lst: T | list[T]) -> list[T]:
        if isinstance(lst, list):
            return lst
        return [lst]

    if isinstance(lst, list):
        return [item for sublist in lst for item in _to_list(sublist)]
    return [lst]
