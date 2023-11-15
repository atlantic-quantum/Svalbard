"""
Data models for source code and settings and compiler used to generate instrument
specific code
"""

from typing import Literal

from pydantic import BaseModel

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
