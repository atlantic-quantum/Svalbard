from svalbard.data_model.compiler import AQCompilerDataModel, CompilerDataModel
from svalbard.data_model.data_file import MetaData


def test_compiler_data_model_serialisation():
    cdm = CompilerDataModel()
    l_cdm = CompilerDataModel(**cdm.dict())

    assert cdm == l_cdm


def test_aq_compiler_data_model_serialisation():
    acdm = AQCompilerDataModel()
    l_acdm = AQCompilerDataModel(**acdm.dict())

    assert acdm == l_acdm


def test_aq_compiler_data_model_serialisation_json():
    acdm = AQCompilerDataModel(settings={"test-1-test": []})
    acdm_json = acdm.json()

    l_acdm = AQCompilerDataModel.parse_raw(acdm_json)

    assert acdm == l_acdm

    mdat = MetaData(name="test", compiler_data=acdm)
    l_mdat = MetaData.parse_raw(mdat.json())
    assert mdat == l_mdat
