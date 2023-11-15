"""
Pydantic models that defines the data file structure
"""
import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Union

import numpy as np
from pydantic import BaseConfig, BaseModel, Field, validator

from .compiler import AQCompilerDataModel, CompilerDataModel
from .device import Device
from .instruments import InstrumentModel
from .measurement import Measurement
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
            for (channel_name, values) in measurement.calculate_values(False).items()
        ]
        for log_name in measurement.log_channels:
            datasets.append(
                Data.DataSet(
                    name=log_name,
                    memory=SharedMemoryOut.from_memory_in(
                        SharedMemoryIn(
                            dtype=measurement.get_channel(log_name).dtype.value,
                            shape=measurement.log_shape,
                            # ! what if log channel is not single value but array?
                            # ! mainly an issue if multiple log channels
                            # ! todo: with different shapes are used
                        ),
                    ),
                )
            )

        return Data(handle=MeasurementHandle.new(), datasets=datasets)


class BaseMetaData(BaseModel):
    """
    Pydantic model for BaseMetaData, the parent model for all other Metadata models

    Args:
        model_type (Literal):
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

    model_type: Literal[""] = ""
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
        model_type (Literal):
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

    """

    model_type: Literal["measurement"] = "measurement"
    station: str = "unknown"
    device: Device = Device()
    instruments: dict[str, InstrumentModel] = {}
    cooldown: datetime = Field(default_factory=lambda: datetime(1900, 1, 1))

    compiler_data: CompilerDataModel | AQCompilerDataModel = Field(
        default_factory=AQCompilerDataModel, discriminator="compiler"
    )
    measurement: Measurement = Measurement.create_empty()
    version: str = "0.1.2"

    class Config(BaseConfig):
        """Pydantic Model Config customization for MetaData models"""

        validate_assignment = True

    @validator("instruments")
    def validate_instruments(cls, instruments: dict[str, InstrumentModel]):
        """Validate that the keys of instruments are the same as their identities"""
        for key, instrument_model in instruments.items():
            assert key == instrument_model.identity
        return instruments


class MaskMetaData(BaseMetaData):
    """
    Pydantic model for mask Metadata

    Args:
        model_type (Literal):
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

    model_type: Literal["mask"] = "mask"
    mask_params: dict = {}
    chips_params: dict = {}
    version: str = "0.0.1"


class EMSimMetaData(BaseMetaData):
    """
    Pydantic model for EM Simulation Metadata

    Args:
        model_type (Literal):
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

    model_type: Literal["em_simulation"] = "em_simulation"
    device: Device = Device()
    version: str = "0.0.1"


class MetaDataDiscriminator(BaseMetaData):
    metadata: Union[MetaData, MaskMetaData, EMSimMetaData] = Field(
        discriminator="model_type"
    )


class DataFile(BaseModel):
    """Pydantic model for DataFiles"""

    data: Data | None
    metadata: Union[MetaData, MaskMetaData, EMSimMetaData] = Field(
        discriminator="model_type"
    )
