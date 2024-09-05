from svalbard.data_model.compiler import (
    AQCompilerDataModel,
    AQCompilerDataModelV2,
    CompilerDataModel,
)
from svalbard.data_model.data_file import MetaData


def test_compiler_data_model_serialisation():
    cdm = CompilerDataModel()
    l_cdm = CompilerDataModel(**cdm.model_dump())

    assert cdm == l_cdm


def test_aq_compiler_data_model_serialisation():
    acdm = AQCompilerDataModel()
    l_acdm = AQCompilerDataModel(**acdm.model_dump())

    assert acdm == l_acdm


def test_aq_compiler_data_model_serialisation_json():
    acdm = AQCompilerDataModel(settings={"test-1-test": []})
    acdm_json = acdm.model_dump_json()

    l_acdm = AQCompilerDataModel.model_validate_json(acdm_json)

    assert acdm == l_acdm

    mdat = MetaData(name="test", compiler_data=acdm)
    l_mdat = MetaData.model_validate_json(mdat.model_dump_json())
    assert mdat == l_mdat


def test_aq_compiler_data_model_v2_serialisation():
    acdm = AQCompilerDataModelV2()
    l_acdm = AQCompilerDataModelV2(**acdm.model_dump())
    assert acdm == l_acdm

    acdm_json = acdm.model_dump_json()
    l_acdm_json = AQCompilerDataModelV2.model_validate_json(acdm_json)
    assert acdm == l_acdm_json


def test_json_serial_deserialization_aqcdm_v2():
    acdm = AQCompilerDataModelV2(
        sequence_mapping={
            ("seq", 0): "drive_frame",
            ("seq", 1): "map_frame",
            1: "flux_frame",
        }
    )
    new_acdm = AQCompilerDataModelV2.model_validate_json(acdm.model_dump_json())
    assert new_acdm == acdm
