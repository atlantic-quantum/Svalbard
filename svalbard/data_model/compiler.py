"""
Data models for source code and settings and compiler used to generate instrument
specific code
"""

from typing import Literal

from pydantic import BaseModel, field_validator

from .instruments import InstrumentSetting


class CompilerDataModel(BaseModel):
    """
    Data model for compiler data

    Args:
        qasm_src (str):
            string containing qasm source code
        setup (dict):
            dictionary containing setup data
        compiler (str):
            name of the compiler used to generate the code
        compiler_version (str):
            version of the compiler used to generate the code
    """

    qasm_src: str = ""
    setup: dict = {}
    compiler: Literal["unknown"] = "unknown"
    compiler_version: str = "unknown"


class AQCompilerDataModel(BaseModel):
    """
    Data model for compiler data of the aq_compiler

    Args:
        qasm_src (str):
            string containing qasm source code
        setup (dict):
            dictionary containing setup data
        compiler (str):
            name of the compiler used to generate the code
        compiler_version (str):
            version of the compiler used to generate the code
        settings (dict[str, list[InstrumentSetting]]):
            dictionary containing settings for each Core as a list of InstrumentSetting
        compiled_code (dict[str, str]):
            dictionary containing compiled code for each Core as a string
    """

    # todo need source code here has well
    # todo do we also need a field for input dict?
    # todo should there be a new model for the compiler input e.g. using filenames etc.
    setup: dict = {}  # aq_compiler.pulse.Setup
    compiled_code: dict[str, str] = {}
    settings: dict[str, list[InstrumentSetting]] = {}
    extracted_values: dict = {}  # todo update once AQC-230 and AQC-229 are done
    compiler: Literal["aq_compiler"] = "aq_compiler"
    compiler_version: str = "unknown"

    # key of compiled_code and settings are intended to be core identifiers
    # on the format InstrumentNameInSetup-CoreNumber-CoreType


class AQCompilerDataModelV2(BaseModel):
    """
    Data model for compiler data of the aq_compiler

    Args:
        qasm_src (str):
            string containing qasm source code
        setup (dict):
            dictionary containing setup data
        compiler (str):
            name of the compiler used to generate the code
        compiler_version (str):
            version of the compiler used to generate the code
        settings (dict[str, list[InstrumentSetting]]):
            dictionary containing settings for each Core as a list of InstrumentSetting
        sequence_mapping (dict[tuple[str, int], str):
            dictionary containing mapping from sequence identifiers and waveform
            indices to frame names
        input_dict (dict):
            dictionary containing input values for the qasm code
    """

    qasm_src: str = ""
    setup: dict = {}  # aq_compiler.setup.ExternalSetup
    compiler: Literal["aq_compiler_v2"] = "aq_compiler_v2"
    compiler_version: str = "unknown"
    settings: dict[str, list[InstrumentSetting]] = {}
    sequence_mapping: dict[tuple[str, int] | int, str] = {}
    input_dict: dict = {}

    # keys of settings are intended to be core identifiers (CoreTuple)
    # on the format InstrumentNameInSetup-CoreNumber-CoreType

    @field_validator("sequence_mapping", mode="before")
    @classmethod
    def sequence_mapping_keys(cls, sequence_mapping: dict[str, str]):
        transfromed_mapping: dict[tuple[str, int] | int, str] = {}
        for key, value in sequence_mapping.items():
            if isinstance(key, str):
                key = key.split(",")
                if len(key) == 2:
                    transfromed_mapping[(key[0], int(key[1]))] = value
                else:
                    transfromed_mapping[int(key[0])] = value
            else:
                transfromed_mapping[key] = value  # type: ignore
        return transfromed_mapping
